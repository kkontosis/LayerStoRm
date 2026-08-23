// DSpark DFlash backbone runtime (DSP-3): aux-hidden ingest + the semi-AR
// parallel backbone forward that D_CMD_RUN_DSPARK_STEP drives.
//
// Semantics (authoritative refs: ref/vllm qwen3_dflash.py / qwen3_dspark.py +
// v1/worker/gpu/spec_decode/{dflash,dspark}/speculator.py; DeepSpec
// modeling/dspark/common.py):
//
//   * The target's aux hidden states (input of target layers
//     aux_hidden_state_layer_ids — post-residual output of layer id-1, the
//     vLLM aux capture convention, INV-DSPARK-AUX) are concatenated
//     [T, N_aux*H] -> fc GEMM -> [T, H] fused context hiddens.  These do NOT
//     feed the query stack directly: hidden_norm + per-layer {k,v}_proj +
//     K-norm + RoPE turn them into the draft's CONTEXT KV, keyed by absolute
//     target position (DFlash "context KV precompute").
//   * One RUN_DSPARK_STEP = ONE backbone forward over the whole block:
//     query rows = [anchor_token, mask_token x ndraft] at positions
//     anchor_pos + [0..ndraft], NON-causal inside the block, attending to
//     all context positions < anchor_pos.  Speculators-format checkpoints
//     use the DFlash 1+N "bonus anchor" convention (INV-DSPARK-ANCHOR,
//     vLLM dspark_bonus_anchor=True): the anchor row is NON-predicting and
//     each mask row's logits predict ITS OWN position — draft slot k =
//     physical row k+1, predicting anchor_pos+k+1.  Outputs: base_logits
//     [rows, V] FP32 + hidden_out [rows, H] BF16 (post final norm — DSP-4's
//     Markov head and DSP-6's confidence head consume rows [1..ndraft]).
//     The layout is UNCONDITIONAL — the dense-DeepSpec reading (every row
//     sampled, slot k predicts query_pos+1) mis-drafts every checkpoint
//     this loader can load and is not implemented.
//   * run_markov_head (DSP-4, INV-DSPARK-MARKOV) then chains the sequential
//     stage on the same stream: per step, base_logits[k] plus the low-rank
//     transition bias of the ACTUALLY-SAMPLED predecessor, greedy argmax,
//     all on-device — leaving draft_tokens [gamma] i32 + corrected_logits
//     [gamma, V] FP32 for DSP-5 verification.
//
// TP seam (TD-DSPARK-DRAFT-SHARD, LANDED for exactly 2 ranks): the backbone
// forward is parameterized on the draft DEVICE SET (ranks_); every
// projection routes through the column-/row-parallel helpers with
// n_local = dim / num_ranks.  Sharded (nr == 2) semantics:
//   * Weights split per dspark_tensor_shard_kind (loader): q/k/v/gate/up +
//     lm_head column-parallel (row slices), o_proj/down_proj row-parallel
//     (K-column windows, quant scale groups verified aligned), norms
//     replicated, embed/fc/markov/confidence/d2t single-homed on rank 0.
//   * The residual stream (q_x/q_normed/hidden_out) is REPLICATED: each rank
//     runs the elementwise adds + norms on identical inputs (deterministic
//     kernels -> bit-identical replicas); attention is head-sharded
//     (n_heads/nr per rank) over a per-rank context-KV arena shard.
//   * The row-parallel combine (o_proj / down_proj partials) is a D2D
//     cross-copy + commutative local add (a+b bitwise == b+a at nr == 2),
//     staged per (layer, site) — NOT NCCL: the target's TP=2 attention
//     collectives run CONCURRENTLY on the same 5090 pair from different
//     streams, and un-stream-ordered NCCL kernels from a second comm (or
//     interleaved ops on the shared comm) can cross-deadlock.  The D2D
//     pattern is the sanctioned INV-MOE-EP-XTP fold (dispatch_moe.cpp
//     ep_xtp_broadcast).  DcpCommunicator is deliberately NOT used.
//   * lm_head runs column-parallel per rank; rank 1's logits shard is
//     2D-gathered into rank 0's base_logits; the Markov/confidence heads +
//     EPM tap stay single-homed on rank 0 (their sweeps are second-order).
//   * Numerics: K-split partial sums change the FP reduction order — the
//     sharded forward is FP-band-equivalent (draft ids expected-identical),
//     never bit-identical, to the single-rank forward.  nr == 1 keeps the
//     exact legacy kernel sequence (bit-identical).
// DCP does NOT apply (standard MHA, tiny KV).
//
// Threading: daemon thread only (INV-3.4.2).  CUDA-free TU (INV-GPU-1):
// device work goes through DeviceBackend + kernel launchers.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model/weight_loader/dspark_loader.h"

namespace layerstorm::config {
struct Config;
}

namespace layerstorm::compute {
class DeviceBackend;
class ExpertDevice;
}

namespace layerstorm::memory {
class NumaManager;
}

namespace layerstorm::parallelism {
class DcpCommunicator;
}

namespace layerstorm::speculation {

class DsparkRuntime {
public:
    /// One draft device of the device set.  DSP-3: exactly one rank; the
    /// forward loops over ranks_ so TP=2 across both 5080s later is the
    /// sharded branch + config, not a rewrite.
    struct Rank {
        compute::DeviceBackend* backend = nullptr;  ///< non-owning
        void* stream = nullptr;                     ///< non-owning (StreamManager)
    };

    /// Build the runtime: loads the checkpoint from
    /// cfg.speculation.dspark.checkpoint_path, cross-validates the config,
    /// uploads each rank's draft weight shard onto its GPU and allocates
    /// all device scratch (aux staging, context-KV arena, query buffers).
    /// One rank = the whole draft (legacy); two ranks = the TP=2 shard
    /// (TD-DSPARK-DRAFT-SHARD); more ranks fail closed (the D2D combine is
    /// a 2-rank cross-exchange).
    /// When `arenas` is non-empty (the engine path; one region per rank,
    /// arenas.size() == ranks.size()), weights AND scratch are carved from
    /// those device regions instead of fresh device_allocs — the
    /// LayerRegistry budget already accounted them as each draft GPU's
    /// pinned region (VramAllocator), so a second allocation would
    /// DOUBLE-BOOK the VRAM (observed: the 7.61 GB device_alloc fails on
    /// the 15 GB 5080 whose pinned region already holds the charge). Empty
    /// arenas (unit tests) device_allocs and owns everything. Throws on any
    /// failure (fail closed), including insufficient arena capacity.
    ///
    /// `numa` (EPM-1, optional): NUMA manager for the epm-dump D2H host
    /// staging — allocated on the DRAFT GPU's local node (P-22 home_node
    /// pattern) then register-pinned. Null (unit tests) falls back to
    /// backend host_alloc_pinned. Only consulted when the EPM dump is
    /// enabled (speculation.dspark.epm_dump_dir / LS_EPM_DUMP).
    static std::unique_ptr<DsparkRuntime> create(
        const config::Config& cfg, std::vector<Rank> ranks,
        parallelism::DcpCommunicator* communicator = nullptr,
        std::vector<void*> arenas = {},
        std::vector<int64_t> arena_bytes = {},
        memory::NumaManager* numa = nullptr);

    ~DsparkRuntime();
    DsparkRuntime(const DsparkRuntime&) = delete;
    DsparkRuntime& operator=(const DsparkRuntime&) = delete;

    // ── Aux-hidden export (target side) ────────────────────────────────────
    // Called by the CommandDispatcher hook at the START of target layer L's
    // attention dispatch, when L is one of aux_hidden_state_layer_ids: the
    // primary rank's attn_buf then holds the post-residual output of layer
    // L-1 == the input of layer L (INV-DSPARK-AUX).

    /// aux slot index for a target layer, or -1 if not an aux layer.
    int aux_slot_for_layer(uint32_t target_layer) const;

    /// Enqueue the cross-GPU copy of `rows` hidden rows (BF16 [rows, H],
    /// tight) from the TARGET GPU's attn_buf into aux-staging column slot
    /// `slot` on the draft GPU, on `src_stream` (the target's kAttention
    /// stream — ordered after the previous layer's MoE commit).  When the
    /// LAST slot of a step arrives, chains the ingest (fc -> hidden_norm ->
    /// per-layer context-KV append at positions [start_pos, start_pos+rows))
    /// onto the draft stream and advances the tracked context.
    ///
    /// Steps larger than aux_capture_max_rows are ingested in staging-sized
    /// pieces per slot (chunked fc accumulation; ctx_hidden_ is the
    /// accumulator) — whole-prompt prefills stay draftable up to the
    /// context-arena capacity (TD-DSPARK-PREFILL-CAP resolved).
    ///
    /// Superchunk prefill (TD-DSPARK-SUPERCHUNK-CAPTURE resolved): each aux
    /// layer may deliver its rows as MULTIPLE contiguous windows chunk-major
    /// (slot 0's windows all arrive before slot 1's first — target layer
    /// order).  A slot-0 window at start_pos == slot 0's covered end EXTENDS
    /// the open epoch; every slot must then cover the identical window
    /// [epoch base, slot-0 end) contiguously before the ingest finalizes at
    /// the last slot.  Rows land in the draft context keyed by ABSOLUTE
    /// position regardless of arrival interleaving; numerics follow the
    /// chunked per-slot fc accumulation (the single-window paths are
    /// byte-identical to before).
    ///
    /// Fail-closed contract: any shape the DSP-3 exporter cannot ingest
    /// (position gap, sequence switch mid-context, context-arena overflow
    /// beyond draft_context_capacity_tokens) marks the tracked context
    /// INVALID for drafting and returns; the target forward is NEVER
    /// affected.
    void capture_aux(int slot, const void* target_attn_buf, int rows,
                     uint64_t seq_id, uint32_t start_pos,
                     compute::DeviceBackend& src_backend, void* src_stream);

    /// Disable drafting for the tracked context (fail-closed hook for target
    /// step shapes DSP-3 does not export: B>1 decode batches, drafts, ...).
    void invalidate_context(const char* why);

    /// Aux-hidden row width the loaded draft checkpoint expects (its
    /// hidden_size; 0 before load). V4-5b mHC: the capture hook fails loud
    /// when this differs from the target's hc_streams*hidden export width.
    int aux_hidden_width() const { return H_; }

    /// Ticket J: the V4 dflash draft consumes the target aux hiddens as the
    /// MEAN over the hc residual streams ([rows, hidden], ref/vllm
    /// deepseek_v4/nvidia/model.py:1101) — the capture hook must reduce the
    /// hc-wide rows target-side (launch_hc_stream_mean) before the copy,
    /// and additionally deliver the FINAL-residual tap (aux id ==
    /// num_target_layers) from the output-head path.
    bool aux_stream_mean() const { return ckpt_.is_v4_dflash; }

    // ── Backbone forward (D_CMD_RUN_DSPARK_STEP) ───────────────────────────

    /// ONE backbone forward over the gamma block.  num_query == 0 selects
    /// the config speculative_tokens.  On success base_logits()/hidden_out()
    /// hold [num_query, V] FP32 / [num_query, H] BF16 (device, draft GPU)
    /// once the draft stream drains.  Returns false with *err set on any
    /// validation failure (unarmed context, seq mismatch, anchor_pos beyond
    /// the ingested context, num_query > block_size); enqueues nothing then.
    bool run_step(uint64_t seq_id, uint32_t anchor_token_id,
                  uint32_t anchor_pos, int num_query, std::string* err);

    // ── Sequential Markov head (DSP-4, INV-DSPARK-MARKOV) ──────────────────

    /// Left-to-right greedy sample loop over the LAST run_step outputs:
    /// per step k, logits_k = base_logits[k] + markov_w2 @ markov_w1[x_{k-1}]
    /// (x_{-1} = the anchor token, still device-resident in q_ids_[0]);
    /// x_k = argmax(logits_k) feeds the next step's lookup ON-DEVICE (fused
    /// finalize kernel writes x_k AND gathers the next e — no host round-
    /// trip; the whole gamma chain is stream-ordered on the draft stream).
    /// Must be called after a successful run_step and before the next one.
    /// On success draft_tokens() [last_num_query] i32 and corrected_logits()
    /// [last_num_query, V] FP32 are valid once the draft stream drains.
    /// Fails closed (false + *err) on: no run_step outputs, non-vanilla
    /// checkpoint head (TD-DSPARK-HEAD-VARIANTS), out-of-range markov_rank.
    bool run_markov_head(std::string* err);

    /// Sampled draft token ids [last_num_query] i32 (device, draft GPU) —
    /// what DSP-5 verify consumes.  ALWAYS TARGET-vocab ids: reduced-vocab
    /// checkpoints remap the draft-space argmax on-device via the d2t map
    /// inside the finalize kernel (TD-DSPARK-VOCAB-REMAP).  Valid after
    /// run_markov_head's stream work completes.
    const int32_t* draft_tokens() const;

    /// Markov-corrected logits [last_num_query, draft_vocab] FP32 (device) —
    /// base_logits + per-step transition bias (DSP-5/DSP-6 consumers).
    /// DRAFT-vocab space: for reduced-vocab checkpoints a probabilistic
    /// verifier must scatter columns into target vocab via d2t (vLLM
    /// _draft_scatter_buf pattern) — the greedy path never reads these.
    const float* corrected_logits() const;

    // ── Trained confidence head (DSP-6, INV-DSPARK-CONF) ────────────────────

    /// c_k = sigmoid(confidence_head.proj · [hidden_k ; markov_w1[x_{k-1}]]
    /// + bias) for k in [0, last_num_query) — the TRAINED per-position
    /// survival probability (conditional on all predecessors accepted;
    /// cumprod-composable), ONE kernel over the gamma positions.  hidden_k
    /// and the per-step markov embeds are already device-resident from
    /// run_step / run_markov_head (the DSP-4 e-chain is stashed per step).
    /// Must be called after run_markov_head (the with_markov head consumes
    /// the e stash; hidden-only checkpoints only need run_step).  Fails
    /// closed (false + *err) on: checkpoint without a confidence head, no
    /// run_step outputs, with_markov head before run_markov_head.
    /// DISTINCT from the output-head {top1_prob, entropy} heuristic
    /// (compute/kernels/confidence/) — INV-DSPARK-CONF.
    bool run_confidence_head(std::string* err);

    /// Per-position survival probabilities [last_num_query] FP32 (device),
    /// each in (0,1).  Valid after run_confidence_head's stream work
    /// completes.  Raw (pre-STS-calibration, DSP-7) values.
    const float* confidence() const;

    /// The loaded checkpoint ships a usable confidence head. The bias is
    /// required only for formats that carry one (the V4 dflash GGUF's
    /// conf_proj has none — confidence_has_bias false, kernel treats a null
    /// bias as 0).
    bool has_confidence_head() const {
        return ckpt_.enable_confidence_head &&
               weights_.confidence_proj_weight.ptr != nullptr &&
               (!ckpt_.confidence_has_bias ||
                weights_.confidence_proj_bias.ptr != nullptr);
    }

    // ── EPM-1 training-data dump (Phase 29, spec/MoE-SpeQ_NOTES.md §7) ─────

    /// The feature-side dump is armed (speculation.dspark.epm_dump_dir or
    /// LS_EPM_DUMP resolved non-empty at create() AND the dump file opened).
    /// When false, run_step contains ZERO extra work — no staging buffer is
    /// allocated and the per-layer tap is a single null-pointer check.
    bool epm_dump_enabled() const { return epm_writer_ != nullptr; }

    /// Write one EPMB block record for the LAST run_step: D2H of the
    /// per-backbone-layer residual-stream hiddens [gamma, L, H] BF16 (the
    /// tap copies q_x_ after every layer's MLP residual add), the DSP-4
    /// sampled draft ids, and — when `conf_valid` — the raw DSP-6 c_k;
    /// then a device sync and an append to <dump_dir>/epm_blocks.bin.
    /// Call AFTER run_markov_head (+ run_confidence_head when enabled).
    /// No-op when the dump is disabled or there is no pending step.
    void epm_write_block_record(bool conf_valid);

    // ── Introspection (tests / DSP-4 consumers) ────────────────────────────

    int draft_gpu_position() const;
    void* draft_stream() const { return ranks_[0].stream; }
    compute::DeviceBackend* draft_backend() const { return ranks_[0].backend; }

    /// Device pointers into draft-GPU scratch (valid after run_step's stream
    /// work completes).  DSP-4 (Markov head) reads both on-device.
    const float* base_logits() const;   ///< [last_num_query, draft_vocab] FP32
    const void* hidden_out() const;     ///< [last_num_query, H] BF16
    int last_num_query() const { return last_num_query_; }
    /// Physical row offset of draft slot 0 in hidden_out/base_logits: 1 —
    /// row 0 is always the non-predicting bonus-anchor row (the layout
    /// speculators-format checkpoints are trained in; INV-DSPARK-ANCHOR).
    int sample_off() const { return 1; }

    /// Tracked context state (host-side mirror of the ingested KV arena).
    uint64_t ctx_seq_id() const { return ctx_seq_id_; }
    int ctx_len() const { return ctx_len_; }
    bool ctx_valid() const { return ctx_valid_; }

    /// TD-V4-SPEC-PREFILL-CTX: open capture epoch awaiting ONLY the final
    /// aux slot.  True iff the context is valid, every slot except the last
    /// has captured, and the second-to-last slot's coverage reached the
    /// epoch target — i.e. the exact state a headless chunk (or any step
    /// whose head has not yet fired) leaves behind after the last layer's
    /// MoE. Outputs the window the final slot must cover: [start, end) of
    /// sequence `seq`.  n_aux < 2 (final-tap-only checkpoints) is not
    /// resolvable from the epoch — returns false (head-sited capture only).
    bool pending_final_window(uint64_t* seq, uint32_t* start,
                              uint32_t* end) const {
        const int n = aux_count();
        if (!ctx_valid_ || n < 2) return false;
        const uint32_t want = (1u << (n - 1)) - 1u;  // slots 0..n-2
        if (cap_slot_mask_ != want) return false;
        if (cap_slot_end_.size() < static_cast<size_t>(n)) return false;
        const uint32_t target = cap_slot_end_[0];
        if (cap_slot_end_[static_cast<size_t>(n) - 2] != target) return false;
        if (target <= cap_start_pos_) return false;
        *seq = cap_seq_;
        *start = cap_start_pos_;
        *end = target;
        return true;
    }

    /// Model dims (from the checkpoint).
    const model::DsparkCheckpointConfig& ckpt() const { return ckpt_; }
    int aux_count() const { return static_cast<int>(aux_ids_.size()); }

    /// Aux staging view (tests): device ptr [aux_capture_max_rows, N_aux*H].
    const void* aux_stage() const { return aux_stage_; }

private:
    DsparkRuntime() = default;

    /// fc + hidden_norm + per-layer KV projection/append for `rows` staged
    /// aux rows at absolute positions [start_pos, start_pos + rows).
    void ingest_context(int rows, uint32_t start_pos);

    /// Chunked-capture pieces (TD-DSPARK-PREFILL-CAP; steps larger than the
    /// aux staging).  ensure_capture_event: lazy per-source-backend event
    /// creation shared by both paths (false + invalidate on backend change).
    /// capture_slot_chunked: per staging-sized piece, copy slot rows +
    /// fold the per-slot fc partial into the ctx_hidden_ accumulator
    /// (fc distributes over the concat).  finalize_context_chunked:
    /// hidden_norm + per-layer KV append over the accumulated rows, in
    /// staging-sized pieces.  append_context_kv: the shared per-layer K/V
    /// projection/append body (single-shot ingest + chunked finalize).
    bool ensure_capture_event(compute::DeviceBackend& src_backend);
    void capture_slot_chunked(int slot, const void* target_attn_buf,
                              int rows, int64_t acc_row_base,
                              compute::DeviceBackend& src_backend,
                              void* src_stream);
    /// TD-DSPARK-SUPERCHUNK-CAPTURE: transition an open single-window epoch
    /// into per-slot fc accumulation when slot 0's SECOND contiguous window
    /// arrives (chunk-major superchunk prefill).  Folds the already-staged
    /// first window's slot-0 fc partial into ctx_hidden_ (no-op when the
    /// first window went through the chunked path already).  False (+
    /// invalidate) on capture-event backend change.
    bool begin_multi_window_epoch(compute::DeviceBackend& src_backend,
                                  void* src_stream);
    void finalize_context_chunked(int rows, uint32_t start_pos);
    void append_context_kv(const void* normed, int rows, uint32_t start_pos,
                           void* stream);

    // Weight-consuming GEMM seam (TD-DSPARK-DRAFT-QUANT): every projection
    // GEMM routes through weight_gemm, which dispatches on the device
    // tensor's storage dtype — BF16 tensors through
    // launch_bf16_gemm_nt_strided (bit-identical to the legacy direct
    // calls: lda == ldw == K collapses to launch_bf16_gemm_nt), quantized
    // tensors through the fused-dequant launch_wq_gemm_nt (no BF16 weight
    // scratch).  `k_off` selects a column window of W's stored rows (the
    // chunked-fc slot GEMMs; scale groups are anchored at the row start so
    // any k_off is legal); `row_off` selects a row window (col-parallel
    // rank shards).  out_fp32 = FP32 C (the lm_head site); else BF16.
    void weight_gemm(void* C, const void* A,
                     const model::DsparkDeviceTensor& W, int M, int N, int K,
                     int64_t lda, int64_t ldw, int64_t k_off, int64_t row_off,
                     bool out_fp32, void* stream) const;

    // Sharded-GEMM helpers (the TP seam).  W is rank `rank`'s SHARD of the
    // [N, K] row-major projection (the loader uploaded rows
    // [rank*N/nr, (rank+1)*N/nr) as a tight [N/nr, K] tensor) — the GEMM
    // computes the rank's n_local output columns into a tight [M, n_local]
    // buffer on that rank's stream.  At nr==1 the shard IS the whole tensor
    // (bit-identical legacy path).
    void col_parallel_gemm(size_t rank, void* C, const void* A,
                           const model::DsparkDeviceTensor& W,
                           int M, int N, int K) const;
    // Row-parallel combine point (o_proj: site 0 on q_oproj; down_proj:
    // site 1 on q_mlp).  nr==1: no-op (the single rank holds the full sum).
    // nr==2: D2D cross-copy of each rank's [num_tokens, H] partial into the
    // peer's (layer, site) staging slot + a commutative in-place add on
    // each rank — both ranks end with the bit-identical full sum (IEEE add
    // commutes).  NOT NCCL (see the header block: concurrent target TP
    // collectives on the same devices make shared/second comms unsafe).
    void allreduce_seam(int layer, int site, int num_tokens) const;

    // ── Immutable after create() ──
    model::DsparkCheckpointConfig ckpt_;
    model::DsparkDeviceWeights weights_;   ///< rank 0's (whole draft at nr==1)
    std::vector<Rank> ranks_;
    parallelism::DcpCommunicator* communicator_ = nullptr;
    std::vector<int> aux_ids_;
    int H_ = 0, n_heads_ = 0, head_dim_ = 0, q_dim_ = 0, kv_dim_ = 0;
    // Per-rank local dims (== the full dims at nr==1).
    int n_heads_local_ = 0, q_dim_local_ = 0, kv_dim_local_ = 0,
        I_local_ = 0;
    int64_t Vd_local_ = 0;
    int I_ = 0, L_ = 0, block_size_ = 0, spec_tokens_ = 0;
    int64_t V_ = 0;   ///< target/embed vocab (embed_tokens, markov_w1 rows)
    int64_t Vd_ = 0;  ///< draft vocab (lm_head/markov_w2 rows; == V_ unless
                      ///< the checkpoint ships a reduced vocab + d2t map —
                      ///< TD-DSPARK-VOCAB-REMAP)
    int mask_token_id_ = 0;
    float eps_ = 1e-5f;
    float theta_ = 8e6f;
    int ctx_cap_ = 0;        ///< draft_context_capacity_tokens
    int aux_rows_cap_ = 0;   ///< aux_capture_max_rows

    // ── Device scratch (rank 0's GPU; freed in dtor) ──
    void* aux_stage_ = nullptr;    ///< [aux_rows_cap, N_aux*H] BF16
    void* ctx_hidden_ = nullptr;   ///< [max(aux_rows_cap, ctx_cap), H] BF16
                                   ///< (fc out; chunked-capture accumulator)
    void* ctx_normed_ = nullptr;   ///< [aux_rows_cap, H] BF16
    void* ctx_ktmp_ = nullptr;     ///< [aux_rows_cap, kv_dim] BF16
    void* kv_arena_ = nullptr;     ///< L x {K,V} x [ctx_cap, kv_dim] BF16
    void* q_ids_ = nullptr;        ///< [block_size] i32
    void* q_x_ = nullptr;          ///< [block_size, H] BF16 (residual stream)
    void* q_normed_ = nullptr;     ///< [block_size, H] BF16
    void* q_tmp_ = nullptr;        ///< [block_size, q_dim] BF16 (pre-norm qk)
    void* q_q_ = nullptr;          ///< [block_size, q_dim] BF16
    void* q_k_ = nullptr;          ///< [block_size, q_dim] BF16
    void* q_v_ = nullptr;          ///< [block_size, q_dim] BF16
    void* q_attn_ = nullptr;       ///< [block_size, q_dim] BF16
    void* q_oproj_ = nullptr;      ///< [block_size, H] BF16
    void* q_gate_ = nullptr;       ///< [block_size, I] BF16
    void* q_up_ = nullptr;         ///< [block_size, I] BF16
    void* q_act_ = nullptr;        ///< [block_size, I] BF16
    void* q_mlp_ = nullptr;        ///< [block_size, H] BF16
    void* hidden_out_ = nullptr;   ///< [block_size, H] BF16 (post final norm)
    void* base_logits_ = nullptr;  ///< [block_size, Vd] FP32 (draft vocab)
    void* draft_ids_ = nullptr;    ///< [block_size] i32 (DSP-4 sampled tokens,
                                   ///< TARGET-vocab ids — d2t-mapped)
    void* markov_e_ = nullptr;     ///< [block_size, r] BF16 — slot k holds
                                   ///< markov_w1[x_{k-1}] (the DSP-4 e-chain,
                                   ///< stashed per step for DSP-6)
    void* markov_partials_ = nullptr;   ///< [num_blocks(Vd)] argmax partials
    void* corrected_logits_ = nullptr;  ///< [block_size, Vd] FP32 (base + bias)
    void* conf_out_ = nullptr;     ///< [block_size] FP32 (DSP-6 c_k)
    void* epm_stage_ = nullptr;    ///< EPM-1: [block_size, L, H] BF16 dump
                                   ///< staging (null unless dump armed)
    std::vector<void*> owned_dev_;  ///< dtor-freed allocs (empty on arena path)

    // ── V4 dflash draft state (ticket J; unset unless ckpt_.is_v4_dflash) ──
    // The V4 arm reuses the base machinery (context tracking + chunked
    // capture + fc accumulation, q_ids_/draft_ids_/base_logits_/hidden_out_
    // + the DSP-4/DSP-6 markov/confidence kernels) and swaps the backbone
    // forward + the context-KV append for the V4 layer math (latent MQA +
    // sinks + grouped o_proj + mHC + MoE with native-MXFP4 experts on an
    // owned ExpertDevice). Single-rank ONLY (fail-closed in create()).
    // kv arena in V4 mode: L x [ctx_cap, head_dim] BF16 SINGLE-copy roped
    // latents (K == V), row index == absolute position.
    std::unique_ptr<compute::ExpertDevice> v4_expert_dev_;
    void* v4_rope_table_ = nullptr;  ///< [ctx_cap+block, rope_dim] f32 (base)
    void* v4_pos_ = nullptr;         ///< [block_size] i32 query positions
    void* v4_x1_ = nullptr;          ///< [bs, H] hc_pre collapse / head hidden
    void* v4_post_ = nullptr;        ///< [bs, hc] f32 mix
    void* v4_comb_ = nullptr;        ///< [bs, hc*hc] f32 mix
    void* v4_qlat_ = nullptr;        ///< [bs, q_lora] q_a latent
    void* v4_qheads_ = nullptr;      ///< [bs, h_q*head_dim] q_b out
    void* v4_qn_ = nullptr;          ///< [bs, h_q, head_dim] q_nope
    void* v4_qr_ = nullptr;          ///< [bs, h_q, rope_dim] q_rope
    void* v4_kvlat_ = nullptr;       ///< [bs, head_dim] block kv latent
    void* v4_blkkv_ = nullptr;       ///< [bs, head_dim] roped block kv
    void* v4_attn_ = nullptr;        ///< [bs, h_q, head_dim] attention out
    void* v4_lse_ = nullptr;         ///< [bs, h_q] f32
    void* v4_oa_ = nullptr;          ///< [bs, o_groups*o_lora] oproj stage-1
    void* v4_module_out_ = nullptr;  ///< [bs, H] module (attn/moe) output
    void* v4_logits_e_ = nullptr;    ///< [bs, E] f32 router logits
    void* v4_topk_w_ = nullptr;      ///< [bs, topk] f32
    void* v4_topk_idx_ = nullptr;    ///< [bs, topk] i32
    void* v4_perm_in_ = nullptr;     ///< [bs*topk, H] permuted activations
    void* v4_exp_off_ = nullptr;     ///< [E+1] i32
    void* v4_s2d_ = nullptr;         ///< [bs*topk] i32
    void* v4_perm_idx_ = nullptr;    ///< [bs*topk] i32
    void* v4_perm_ws_ = nullptr;     ///< moe_permute workspace
    void* v4_gate_out_ = nullptr;    ///< [bs*topk, I]
    void* v4_up_out_ = nullptr;      ///< [bs*topk, I]
    void* v4_gu_ = nullptr;          ///< [bs*topk, 2I] gate|up concat
    void* v4_act_ = nullptr;         ///< [bs*topk, I]
    void* v4_expert_out_ = nullptr;  ///< [bs*topk, H]
    void* v4_sh_g_ = nullptr;        ///< [bs, I] shexp gate / act reuse
    void* v4_sh_u_ = nullptr;        ///< [bs, I]
    void* v4_sh_gu_ = nullptr;       ///< [bs, 2I]
    void* v4_sh_out_ = nullptr;      ///< [bs, H]
    void* v4_gemm_ws_ = nullptr;     ///< gguf grouped int-GEMM workspace
    int64_t v4_gemm_ws_bytes_ = 0;
    std::array<int32_t, 16> host_pos_{};  ///< persistent H2D position source

    /// V4 backbone forward body (run_step dispatches here when
    /// ckpt_.is_v4_dflash; validation + entry drain already done).
    bool run_step_v4(uint32_t anchor_token_id, uint32_t anchor_pos, int nq,
                     std::string* err);
    /// V4 per-layer context-KV append (wkv → full-D kv_norm → rope tail →
    /// arena rows [start_pos, start_pos+rows)).
    void v4_append_context_kv(const void* normed, int rows,
                              uint32_t start_pos, void* stream);
    /// V4 MoE FFN (routed MXFP4 + shared) over `rows` of q_normed_ →
    /// v4_module_out_.
    void v4_moe_ffn(int layer, int rows, void* stream);

    // ev_capture_done_ is recorded on the TARGET's stream, and
    // cudaEventRecord requires event and stream on the SAME device — so it
    // is created LAZILY by the source (target) backend at the first capture
    // (capture_ev_backend_ destroys it). ev_ingest_done_ is recorded on the
    // draft stream (created on the draft device). cudaStreamWaitEvent is
    // legal cross-device for both.
    void* ev_capture_done_ = nullptr;              ///< recorded on target stream
    compute::DeviceBackend* capture_ev_backend_ = nullptr;  ///< creator/owner
    void* ev_ingest_done_ = nullptr;   ///< recorded on draft stream
    void* ev_step_sync_ = nullptr;     ///< run_step entry drain: recorded +
                                       ///< host-spun on the DRAFT STREAM only
                                       ///< (a device-wide sync would serialize
                                       ///< against unrelated same-GPU work —
                                       ///< expert GEMMs on a 5080 host, the
                                       ///< whole TP pipeline on a 5090 host)

    // ── Sharded-draft per-rank state (TD-DSPARK-DRAFT-SHARD; ranks 1..nr-1,
    //    empty at nr==1).  Rank 0 keeps the legacy members above; the
    //    buf_*(r) accessors in the .cpp select between them. ──
    struct ShardRank {
        model::DsparkDeviceWeights weights;  ///< this rank's weight shard
        // Scratch (this rank's GPU; local dims where sharded).
        void* ctx_normed = nullptr;   ///< [aux_rows_cap, H] broadcast replica
        void* ctx_ktmp = nullptr;     ///< [aux_rows_cap, kv_dim_local]
        void* kv_arena = nullptr;     ///< L x {K,V} x [ctx_cap, kv_dim_local]
        void* q_x = nullptr;          ///< [bs, H] replicated residual
        void* q_normed = nullptr;     ///< [bs, H] replicated
        void* q_tmp = nullptr;        ///< [bs, q_dim_local]
        void* q_q = nullptr;          ///< [bs, q_dim_local]
        void* q_k = nullptr;          ///< [bs, q_dim_local]
        void* q_v = nullptr;          ///< [bs, q_dim_local]
        void* q_attn = nullptr;       ///< [bs, q_dim_local]
        void* q_oproj = nullptr;      ///< [bs, H] row-parallel partial/sum
        void* q_gate = nullptr;       ///< [bs, I_local]
        void* q_up = nullptr;         ///< [bs, I_local]
        void* q_act = nullptr;        ///< [bs, I_local]
        void* q_mlp = nullptr;        ///< [bs, H] row-parallel partial/sum
        void* hidden_out = nullptr;   ///< [bs, H] replicated final hidden
        void* logits_shard = nullptr; ///< [bs, Vd_local] FP32 lm_head shard
        void* peer_stage = nullptr;   ///< [L, 2, bs, H] BF16 peer partials
        void* ev_step_sync = nullptr; ///< run_step entry drain (this rank)
        void* ev_xfer = nullptr;      ///< cross-rank copy ordering (recorded
                                      ///< on THIS rank's stream)
        void* ev_append = nullptr;    ///< ctx-KV shard append done (guards
                                      ///< rank 0's next ctx_normed broadcast
                                      ///< against the replica WAR)
        std::vector<void*> owned_dev;  ///< dtor-freed (empty on arena path)
    };
    std::vector<ShardRank> shard_;
    // Rank-0 extras, allocated only at nr>1.
    void* logits_shard0_ = nullptr;  ///< [bs, Vd_local] FP32
    void* peer_stage0_ = nullptr;    ///< [L, 2, bs, H] BF16
    void* ev_xfer0_ = nullptr;       ///< recorded on rank 0's stream

    std::array<int32_t, 16> host_ids_{};  ///< persistent H2D source

    // ── Host-side context tracking (daemon thread only) ──
    uint64_t ctx_seq_id_ = ~0ULL;
    int ctx_len_ = 0;
    bool ctx_valid_ = false;
    bool ingest_recorded_ = false;  ///< ev_ingest_done_ has been recorded

    // Per-step capture bookkeeping.  An "epoch" is the row window
    // [cap_start_pos_, cap_slot_end_[0]) currently being assembled: the
    // classic slot-major step is a single window per slot; a superchunk
    // prefill (TD-DSPARK-SUPERCHUNK-CAPTURE) delivers each slot's rows as
    // MULTIPLE contiguous windows chunk-major (all of slot 0's windows
    // before slot 1's first — target layer order), tracked per slot in
    // cap_slot_end_ and folded through the per-slot fc accumulation.
    int cap_rows_ = 0;
    uint32_t cap_start_pos_ = 0;
    uint64_t cap_seq_ = ~0ULL;
    uint32_t cap_slot_mask_ = 0;
    bool cap_multi_ = false;              ///< epoch has >1 window per slot
    std::vector<uint32_t> cap_slot_end_;  ///< per-slot coverage end (abs pos)

    int last_num_query_ = 0;
    bool markov_ran_ = false;  ///< run_markov_head chained after the last
                               ///< run_step (the e stash is valid)

    // ── EPM-1 dump state (all inert when epm_writer_ is null) ──
    std::unique_ptr<class EpmBlockDumper> epm_writer_;
    void* epm_host_ = nullptr;      ///< NUMA-local pinned D2H staging
    size_t epm_host_bytes_ = 0;     ///< page-aligned allocation size
    int epm_host_node_ = -1;        ///< NUMA node (-1 = pinned fallback)
    bool epm_host_registered_ = false;  ///< host_register_pinned_portable ok
    bool epm_host_from_numa_ = false;   ///< freed via NumaManager vs backend
    memory::NumaManager* numa_ = nullptr;  ///< non-owning (dtor free path)
    int64_t epm_hid_cap_bytes_ = 0;  ///< block_size*L*H*2 (host layout)
    uint64_t epm_last_seq_ = ~0ULL;      ///< run_step stash for the record
    uint32_t epm_last_anchor_pos_ = 0;
    uint32_t epm_last_anchor_token_ = 0;
    uint64_t epm_ctr_seq_ = ~0ULL;   ///< per-seq block_idx counter key
    uint32_t epm_next_block_idx_ = 0;

    /// per-layer K/V arena bases (rank r's shard arena; kv_dim_local_ row
    /// stride — the full kv_dim_ at nr==1)
    void* k_base(size_t rank, int layer) const;
    void* v_base(size_t rank, int layer) const;

    /// Per-rank buffer/weight selectors (rank 0 = the legacy members).
    const model::DsparkDeviceWeights& rank_weights(size_t r) const {
        return r == 0 ? weights_ : shard_[r - 1].weights;
    }
};

/// Device scratch bytes create() will allocate beyond the weight arena, from
/// config + checkpoint dims only (no GPU needed) — used by LayerRegistry's
/// draft-GPU budget accounting alongside dspark_draft_bytes(). Includes the
/// EPM-1 hidden-dump staging IFF the dump is enabled (epm_dump_dir /
/// LS_EPM_DUMP) so budget and real allocations can never drift.
/// rank/num_ranks (TD-DSPARK-DRAFT-SHARD): rank `rank`'s per-rank scratch
/// under a num_ranks-way shard; (0, 1) is the legacy single-rank total.
int64_t dspark_runtime_scratch_bytes(const config::Config& cfg,
                                     const model::DsparkCheckpointConfig& ck,
                                     int rank = 0, int num_ranks = 1);

/// Alignment slack the LayerRegistry budget adds on top of the exact
/// weights+scratch byte totals: the arena carve 256-aligns every weight
/// tensor (64), quant scale block (<= 37 under draft_weights_quant —
/// TD-DSPARK-DRAFT-QUANT) and scratch item (~21), so the placed total
/// exceeds the exact sums by < 122 * 256 B. 1 MiB covers it with margin.
inline constexpr int64_t kDsparkArenaAlignSlack = int64_t{1} << 20;

}  // namespace layerstorm::speculation

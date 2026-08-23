// DSpark draft-checkpoint weight loader (DSP-2).
//
// Loads the RedHatAI GLM-5.2 DSpark speculator checkpoint — **speculators
// library format v0.5** (NOT DeepSpec-native): a single `model.safetensors`
// (64 BF16 tensors) + `config.json` (DSparkSpeculatorConfig, subclass of
// DFlashSpeculatorConfig). Ground truth recorded from the shipped checkpoint
// (RedHatAI/GLM-5.2-speculator.dspark @ speculators 0.5.0.dev38):
//
//   embed_tokens.weight            [V=154880, H=6144]   draft's OWN embedding
//   lm_head.weight                 [V, H]               draft's OWN LM head
//                                                       (tie_word_embeddings=false;
//                                                       NOT shared with the target,
//                                                       contrary to the paper §3.3
//                                                       sharing assumption)
//   fc.weight                      [H, N_aux*H=30720]   EAGLE-style aux-hidden
//                                                       fusion: concat of the target's
//                                                       aux_hidden_state_layer_ids
//                                                       ([8,23,39,55,70]) hiddens → H
//   hidden_norm.weight / norm.weight  [H]
//   layers.{0..4}.self_attn.{q,k,v}_proj.weight [4096, H], .o_proj.weight [H, 4096]
//   layers.{0..4}.self_attn.{q,k}_norm.weight   [head_dim=64]  (per-head QK RMSNorm)
//   layers.{0..4}.{input,post_attention}_layernorm.weight [H]
//   layers.{0..4}.mlp.{gate,up}_proj.weight [I=12288, H], .down_proj.weight [H, I]
//   markov_head.markov_w1.weight   [V, r=256]   previous-token embedding lookup
//   markov_head.markov_w2.weight   [V, r]       logit-bias projection (Linear(r→V)
//                                               storage layout; logical W2 is [r, V])
//   confidence_head.proj.weight    [1, H + r = 6400], .bias [1]
//                                               (confidence_head_with_markov=true)
//
// There is NO `mask_embedding` tensor in this format: mask_token_id (154856)
// indexes a row of embed_tokens.
//
// Reduced draft vocab (TD-DSPARK-VOCAB-REMAP; the shipped checkpoint drafts
// in the FULL target vocab): when `draft_vocab_size < vocab_size` the
// checkpoint additionally ships (vLLM qwen3_dflash.py d2t convention)
//   d2t                            [Vd] I64     draft->target id OFFSETS:
//                                               target_id = draft_id + d2t[draft_id]
// and the draft-vocab-sized tensors shrink: lm_head [Vd, H] and
// markov_head.markov_w2 [Vd, r] (the bias is DRAFT-space).  embed_tokens
// and markov_head.markov_w1 stay TARGET-vocab [V, ...] — their inputs are
// target ids (anchor + previously-sampled tokens).  The loader validates
// every d2t entry maps into [0, vocab_size) host-side (fail closed).
//
// The DFlash backbone is a dense Qwen3 transformer (5 layers, full attention,
// QK-norm, rope_theta 8e6) — dims come from `transformer_layer_config` in the
// checkpoint's config.json; the loader validates every tensor shape against
// them and fails closed on ANY unmapped, missing, mis-shaped, or non-BF16
// tensor (INV-DSPARK-CKPT).
//
// Device placement (DSP-2 decision): the whole 7.61 GB BF16 draft is placed
// REPLICATED on ONE draft GPU — `speculation.dspark.draft_gpus[0]` when set,
// else the first non-TP GPU (the otherwise-idle 5080s on the reference box;
// the 5090s carry the tiered target). TP-sharding across the draft GPUs is
// deferred (TD-DSPARK-DRAFT-SHARD): the dense backbone forward lands in DSP-3
// on a single device; sharding would need TP attention + allreduce on the
// draft GPUs for a ~3.8 GB/GPU saving.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "model/weight_loader/gguf_reader.h"
#include "model/weight_loader/safetensors_reader.h"
#include "model/weight_loader/weight_handler.h"

namespace layerstorm::config {
struct Config;
struct DsparkConfig;
enum class DsparkDraftWeightsQuant : int;  // config_parser.h (generated)
}  // namespace layerstorm::config

namespace layerstorm::compute {
class DeviceBackend;
}  // namespace layerstorm::compute

namespace layerstorm::model {

// ── DsparkCheckpointConfig ──────────────────────────────────────────────────
// Parsed from the checkpoint's config.json (speculators v0.5
// DSparkSpeculatorConfig). Authoritative for tensor shapes; the engine's
// speculation.dspark knobs are cross-validated against it
// (validate_dspark_config_against_checkpoint).

// ── V4 dflash GGUF checkpoint extras (ticket J) ────────────────────────────
// The DeepSeek-V4-Flash dspark speculator ships as a single `dflash`-arch
// GGUF (dspark-DeepSeek-V4-Flash-*.gguf): 3 full V4-shaped SWA-only decoder
// blocks (latent-MQA attention + sinks + grouped o_proj + mHC + MoE with
// MXFP4 routed experts), one model-level markov/confidence head pair, an
// fc + enc-norm target-feature fusion, and NO embed/lm_head (aliased from
// the TARGET GGUF). Dims below beyond the shared DsparkCheckpointConfig
// fields; consumed only by the V4 arm of DsparkRuntime.
struct DsparkV4CheckpointDims {
    int64_t q_lora_rank = 0;        // 1024
    int64_t o_groups = 0;           // 8
    int64_t o_lora_rank = 0;        // 1024
    int64_t rope_dim = 0;           // 64
    int64_t sliding_window = 0;     // 128
    int64_t n_routed_experts = 0;   // 256
    int64_t n_expert_used = 0;      // 6 (top-k)
    int64_t moe_intermediate = 0;   // 2048 (routed AND shared width)
    double routed_scaling = 0.0;    // 1.5
    bool norm_topk_prob = true;
    double swiglu_limit = 0.0;      // 10.0
    int hc_mult = 0;                // 4
    int hc_sinkhorn_iters = 0;      // 20
    double hc_eps = 0.0;            // 1e-6
};

struct DsparkCheckpointConfig {
    // Speculator-level fields.
    int block_size = 0;                  // γ (8 in the shipped checkpoint)
    int markov_rank = 0;                 // r (256)
    std::string markov_head_type;        // "vanilla" | "gated" | "rnn"
    bool enable_confidence_head = false;
    bool confidence_head_with_markov = false;
    int64_t draft_vocab_size = 0;        // 154880 in the shipped ckpt (== the
                                         // GLM-5.2 vocab: identity remap);
                                         // < vocab_size selects the reduced-
                                         // vocab d2t path
    int64_t mask_token_id = -1;          // 154856 (row of embed_tokens)
    int max_anchors = 0;                 // 1024
    std::vector<int> aux_hidden_state_layer_ids;  // [8,23,39,55,70]
    bool tie_word_embeddings = false;    // false: own lm_head tensor
    int speculative_tokens = 0;          // proposal_methods[0].speculative_tokens (7)

    // DFlash backbone dims (transformer_layer_config — Qwen3 dense).
    std::string model_type;              // "qwen3"
    int num_hidden_layers = 0;           // 5
    int64_t hidden_size = 0;             // 6144
    int num_attention_heads = 0;         // 64
    int num_key_value_heads = 0;         // 64
    int64_t head_dim = 0;                // 64
    int64_t intermediate_size = 0;       // 12288
    double rms_norm_eps = 0.0;           // 1e-5
    double rope_theta = 0.0;             // 8e6
    int64_t vocab_size = 0;              // 154880 (== draft_vocab_size)

    // ── V4 dflash GGUF arm (ticket J) ──
    bool is_v4_dflash = false;           // checkpoint_path was a dflash .gguf
    bool confidence_has_bias = true;     // safetensors ships a bias; the V4
                                         // GGUF conf_proj has none
    DsparkV4CheckpointDims v4;           // valid iff is_v4_dflash
};

/// Parse <checkpoint_dir>/config.json. Throws std::runtime_error on missing
/// file, wrong speculators_model_type, or missing required fields.
DsparkCheckpointConfig parse_dspark_checkpoint_config(
    const std::filesystem::path& checkpoint_dir);

// ── Host-side draft weights ─────────────────────────────────────────────────

/// One DFlash backbone (Qwen3) decoder layer. All RawTensors view the mmap'd
/// shard (kept alive by DsparkDraftWeights::shard).
struct DsparkLayerWeights {
    RawTensor q_proj;                    // [n_heads*head_dim, H]
    RawTensor k_proj;                    // [n_kv_heads*head_dim, H]
    RawTensor v_proj;                    // [n_kv_heads*head_dim, H]
    RawTensor o_proj;                    // [H, n_heads*head_dim]
    RawTensor q_norm;                    // [head_dim]
    RawTensor k_norm;                    // [head_dim]
    RawTensor input_layernorm;           // [H]
    RawTensor post_attention_layernorm;  // [H]
    RawTensor gate_proj;                 // [I, H]
    RawTensor up_proj;                   // [I, H]
    RawTensor down_proj;                 // [H, I]
};

// ── V4 dflash host tensors (ticket J) ───────────────────────────────────────
// RawTensor views either into the GGUF mmaps (MXFP4 packed experts, F32
// heads) or into `owned` dequant buffers (Q8_0 → BF16 at load; F32 norms
// consumed by the BF16 rmsnorm kernel → BF16-converted, matching the V4
// TARGET's norm upload convention). Layout is logical row-major (GGUF ne
// order reversed).
struct DsparkV4LayerHostWeights {
    RawTensor attn_norm, ffn_norm;                       // BF16 (from F32)
    RawTensor q_a, q_a_norm, q_b, kv, kv_norm;           // BF16
    RawTensor o_a, o_b;                                  // BF16
    RawTensor sinks;                                     // F32 [h_q]
    RawTensor hc_attn_fn, hc_attn_base, hc_attn_scale;   // F32
    RawTensor hc_ffn_fn, hc_ffn_base, hc_ffn_scale;      // F32
    RawTensor gate_inp;                                  // BF16 [E, H]
    RawTensor exp_probs_b;                               // F32 [E]
    RawTensor shexp_gate, shexp_up, shexp_down;          // BF16
    RawTensor exps_gate, exps_up, exps_down;             // MXFP4 packed,
                                                         // expert-major 3D
};

struct DsparkV4HostWeights {
    std::vector<DsparkV4LayerHostWeights> layers;        // [3]
    RawTensor output_hc_fn, output_hc_base, output_hc_scale;  // F32
    // Keep-alive backing: the draft GGUF (MXFP4/F32 spans) + the target
    // shards that own token_embd/output (embed/lm_head spans), + owned
    // dequant/convert buffers (inner heap blocks are pointer-stable).
    GgufReader draft_gguf;
    std::vector<GgufReader> target_shards;
    std::vector<std::vector<std::byte>> owned;
};

/// All draft weights, mapped 1:1 from the checkpoint (zero missing / zero
/// unmapped enforced by load_dspark_draft — INV-DSPARK-CKPT).
struct DsparkDraftWeights {
    DsparkCheckpointConfig ckpt;

    RawTensor embed_tokens;              // [V, H]  (V = target/embed vocab)
    RawTensor lm_head;                   // [Vd, H] (Vd = draft vocab; == V full)
    RawTensor fc;                        // [H, N_aux*H] aux-hidden fusion
    RawTensor hidden_norm;               // [H]
    RawTensor final_norm;                // "norm.weight" [H]
    RawTensor markov_w1;                 // [V, r]  (embeds SAMPLED target ids)
    RawTensor markov_w2;                 // [Vd, r] (Linear(r→Vd) storage)
    RawTensor confidence_proj_weight;    // [1, H(+r)] — empty span if head disabled
    RawTensor confidence_proj_bias;      // [1]        — empty span if head disabled
    RawTensor d2t;                       // [Vd] I64 draft->target offsets —
                                         // empty span for full-vocab drafts

    std::vector<DsparkLayerWeights> layers;  // [num_hidden_layers]

    /// V4 dflash GGUF checkpoint (ticket J): the V4-shaped layer stack +
    /// heads + backing storage. Non-null iff ckpt.is_v4_dflash — the Qwen3
    /// `layers` above stay empty then; embed_tokens/lm_head come from the
    /// TARGET GGUF and fc/hidden_norm/final_norm/markov/confidence reuse
    /// the base slots.
    std::unique_ptr<DsparkV4HostWeights> v4;

    /// Shard handle. Keep alive for mmap data lifetime.
    SafetensorsReader shard;

    int total_tensors_loaded = 0;
    int64_t total_weight_bytes = 0;
};

/// Load the DSpark draft checkpoint from `checkpoint_dir` (config.json +
/// model.safetensors). Strict coverage contract (INV-DSPARK-CKPT): throws
/// std::runtime_error listing the offenders if ANY checkpoint tensor is
/// unmapped, ANY expected slot is missing, ANY shape/dtype mismatches the
/// config-derived expectation (weights BF16; d2t I64), or — reduced vocab —
/// ANY d2t entry maps outside [0, vocab_size).
DsparkDraftWeights load_dspark_draft(const std::filesystem::path& checkpoint_dir,
                                     bool use_mmap = true);

/// Exact draft DEVICE weight bytes from the safetensors header only (no data
/// load, no mmap of the data region), for the given upload quantization
/// (TD-DSPARK-DRAFT-QUANT): GEMM operands (dspark_tensor_is_gemm_operand)
/// count at their quantized size + scale bytes, everything else at the BF16
/// checkpoint size. quant == bf16 reproduces the raw checkpoint total.
/// Used for VRAM budget accounting (LayerRegistry::estimate_gpu_budgets).
/// Throws on unreadable checkpoint.
/// rank/num_ranks (TD-DSPARK-DRAFT-SHARD): device bytes of rank `rank`'s
/// SHARD under a num_ranks-way split (dspark_tensor_shard_kind). (0, 1) is
/// the whole-draft legacy total.
int64_t dspark_draft_bytes(
    const std::filesystem::path& checkpoint_dir,
    config::DsparkDraftWeightsQuant quant =
        static_cast<config::DsparkDraftWeightsQuant>(0) /* bf16 */,
    int rank = 0, int num_ranks = 1);

/// True for the draft tensors whose GEMM operands requantize under a non-bf16
/// draft_weights_quant: the 7 per-layer projections (layers.*.…_proj.weight),
/// fc.weight and lm_head.weight. Norms, embeddings, markov/confidence heads
/// and d2t never quantize. Shared by the upload layout and the budget helper
/// so they cannot drift.
bool dspark_tensor_is_gemm_operand(const std::string& name);

// ── TP=2 draft shard classification (TD-DSPARK-DRAFT-SHARD) ─────────────────
// How each checkpoint tensor is distributed across a multi-rank draft device
// set. Shared by the sharded upload, the per-rank budget helper and the
// runtime — they must never drift.
enum class DsparkShardKind {
    kPrimaryOnly,   // rank 0 only: embed_tokens, fc, hidden_norm, markov
                    // heads, confidence head, d2t (the ctx-fc/heads pipeline
                    // is single-homed on rank 0)
    kReplicated,    // full copy on every rank: layer norms, QK norms,
                    // final norm (the residual stream is replicated)
    kColParallel,   // rank r holds output rows [r*N/nr, (r+1)*N/nr) of the
                    // [N, K] storage: q/k/v/gate/up projections + lm_head.
                    // Quant scales split WITH their rows (scale rows are
                    // per-output-row) — no group-boundary concern.
    kRowParallel,   // rank r holds the K-column window [r*K/nr, (r+1)*K/nr)
                    // of every row: o_proj + down_proj (K-strided partials,
                    // combined by the runtime's allreduce seam). Under a
                    // quant format K/nr must be a scale-group multiple so
                    // the shard's group boundaries (anchored at the slice
                    // start) coincide with the full tensor's — verified
                    // fail-closed at upload/budget time.
};
DsparkShardKind dspark_tensor_shard_kind(const std::string& name);

/// Logical shard shape of one tensor for rank `rank` of `num_ranks` (empty =
/// this rank holds no part — kPrimaryOnly on rank > 0). Validates shard
/// divisibility and (row-parallel, quant != bf16) scale-group alignment at
/// the split boundary; throws on violation. num_ranks == 1 returns `shape`.
std::vector<int64_t> dspark_shard_shape(
    const std::string& name, const std::vector<int64_t>& shape, int rank,
    int num_ranks, config::DsparkDraftWeightsQuant quant);

// ── Draft device resolution ─────────────────────────────────────────────────

/// Resolve the GPU (position in config.hardware.gpus[], per INV-4.18) that
/// hosts the draft: speculation.dspark.draft_gpus[0] when non-empty (range
/// checked), else the first GPU NOT in hardware.tp_array (the secondary /
/// non-TP GPUs are otherwise idle while the TP GPUs run the target). Throws
/// std::runtime_error when no non-TP GPU exists and draft_gpus is empty
/// (fail closed: set speculation.dspark.draft_gpus explicitly).
int resolve_dspark_draft_gpu(const config::Config& cfg);

/// Resolve the FULL draft device set (TD-DSPARK-DRAFT-SHARD): every entry of
/// speculation.dspark.draft_gpus (range-checked, duplicates rejected) when
/// non-empty, else the single auto-resolved GPU. Size > 1 selects the TP=2
/// sharded draft (the runtime enforces exactly 2 ranks for the sharded
/// branch). Throws like resolve_dspark_draft_gpu on failure.
std::vector<int> resolve_dspark_draft_gpus(const config::Config& cfg);

/// Cross-validate the engine's speculation.dspark config against the parsed
/// checkpoint config. Throws std::runtime_error on ANY mismatch of
/// block_size, markov_rank, draft_vocab_size, mask_token_id, max_anchors,
/// aux_hidden_state_layer_ids, speculative_tokens, or head-type/confidence
/// compatibility (confidence_enabled=true requires enable_confidence_head;
/// head_type markov ↔ checkpoint "vanilla").
void validate_dspark_config_against_checkpoint(
    const config::DsparkConfig& dspark_cfg, const DsparkCheckpointConfig& ckpt);

// ── Device placement ────────────────────────────────────────────────────────

/// Device storage dtype of one draft tensor (TD-DSPARK-DRAFT-QUANT). BF16 =
/// checkpoint verbatim; the quant formats apply ONLY to GEMM operands
/// (dspark_tensor_is_gemm_operand) and follow the model/quantization/
/// kgroup_quant.h packing consumed by launch_wq_gemm_nt.
enum class DsparkWeightDtype {
    kBF16,
    kFp8E4M3,  // q [N, K] u8 + FP32 scales [N, ceil(K/128)]
    kNvfp4,    // q [N, ceil(K/2)] packed u8 + UE8M0 scales [N, ceil(K/16)]
};

/// One tensor resident on the draft device.
struct DsparkDeviceTensor {
    void* ptr = nullptr;                 // device pointer into the arena
    int64_t bytes = 0;                   // device bytes at `ptr` (quantized
                                         // size when dtype != kBF16)
    std::vector<int64_t> shape;          // row-major LOGICAL shape, as in the
                                         // checkpoint (independent of dtype)
    DsparkWeightDtype dtype = DsparkWeightDtype::kBF16;
    void* scales = nullptr;              // per-K-group scales (quant only)
    int64_t scales_bytes = 0;
    int64_t k_groups = 0;                // scale groups per row =
                                         // ceil(shape.back()/group) — the
                                         // GEMM lds (0 for kBF16)
};

/// Device mirror of DsparkV4LayerHostWeights (ticket J). MXFP4 expert
/// tensors stay packed; exps_*_ptrs are device `const void*[E]` arrays of
/// per-expert block pointers (GgufGroupedGemmParams::B_ptrs), laid out in
/// the same arena.
struct DsparkV4DeviceLayer {
    DsparkDeviceTensor attn_norm, ffn_norm;
    DsparkDeviceTensor q_a, q_a_norm, q_b, kv, kv_norm;
    DsparkDeviceTensor o_a, o_b;
    DsparkDeviceTensor sinks;
    DsparkDeviceTensor hc_attn_fn, hc_attn_base, hc_attn_scale;
    DsparkDeviceTensor hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    DsparkDeviceTensor gate_inp, exp_probs_b;
    DsparkDeviceTensor shexp_gate, shexp_up, shexp_down;
    DsparkDeviceTensor exps_gate, exps_up, exps_down;
    void* exps_gate_ptrs = nullptr;   // device const void*[E]
    void* exps_up_ptrs = nullptr;
    void* exps_down_ptrs = nullptr;
};

struct DsparkV4DeviceWeights {
    std::vector<DsparkV4DeviceLayer> layers;
    DsparkDeviceTensor output_hc_fn, output_hc_base, output_hc_scale;
};

/// Device mirror of DsparkLayerWeights.
struct DsparkDeviceLayer {
    DsparkDeviceTensor q_proj, k_proj, v_proj, o_proj;
    DsparkDeviceTensor q_norm, k_norm;
    DsparkDeviceTensor input_layernorm, post_attention_layernorm;
    DsparkDeviceTensor gate_proj, up_proj, down_proj;
};

/// All draft weights resident on ONE draft GPU as a single contiguous arena
/// (BF16, tensor offsets 256-byte aligned). Owns the arena (frees it through
/// the backend on destruction) UNLESS it was placed into a caller-provided
/// region (owns_arena == false — DSP-3: the engine places the draft inside
/// the VramAllocator's pinned region on the draft GPU, which the budget
/// already carved; a second device_alloc would double-book the VRAM).
/// Move-only.
struct DsparkDeviceWeights {
    DsparkDeviceWeights() = default;
    ~DsparkDeviceWeights();
    DsparkDeviceWeights(DsparkDeviceWeights&& other) noexcept;
    DsparkDeviceWeights& operator=(DsparkDeviceWeights&& other) noexcept;
    DsparkDeviceWeights(const DsparkDeviceWeights&) = delete;
    DsparkDeviceWeights& operator=(const DsparkDeviceWeights&) = delete;

    compute::DeviceBackend* backend = nullptr;  // non-owning; frees the arena
    void* arena = nullptr;
    int64_t arena_bytes = 0;             // padded total (>= host total_weight_bytes)
    bool owns_arena = true;              // false: caller-provided region (no free)

    DsparkDeviceTensor embed_tokens, lm_head, fc, hidden_norm, final_norm;
    DsparkDeviceTensor markov_w1, markov_w2;
    DsparkDeviceTensor confidence_proj_weight, confidence_proj_bias;
    DsparkDeviceTensor d2t;              // I64 — null ptr for full-vocab drafts
    std::vector<DsparkDeviceLayer> layers;

    /// V4 dflash arm (ticket J): non-null iff the checkpoint was a dflash
    /// GGUF — `layers` above stays empty then.
    std::unique_ptr<DsparkV4DeviceWeights> v4;

    int total_tensors_uploaded = 0;
};

/// Upload all draft weights onto `backend`'s GPU: one arena + per-tensor
/// synchronous H2D copies (init path; mmap source is not pinned). When
/// `quant` != bf16 the GEMM operands (dspark_tensor_is_gemm_operand) are
/// REQUANTIZED host-side at upload (TD-DSPARK-DRAFT-QUANT) — per-K-group
/// packing via model/quantization/kgroup_quant.h, scales carried in the same
/// arena — and everything else stays BF16 verbatim; quant == bf16 is the
/// byte-identical legacy upload. When `arena` is null a fresh device_alloc
/// arena is made (owned); otherwise the tensors are placed at the given
/// device region (capacity checked; NOT owned — DSP-3 engine path: the
/// VramAllocator pinned region the budget already carved). Throws on
/// allocation failure / insufficient capacity.
/// rank/num_ranks (TD-DSPARK-DRAFT-SHARD): upload only rank `rank`'s shard
/// of every tensor per dspark_tensor_shard_kind — col-parallel tensors as a
/// contiguous row slice, row-parallel tensors as a host-repacked K-column
/// window (requantized on the slice; group boundaries verified aligned),
/// kPrimaryOnly tensors skipped on rank > 0 (their DsparkDeviceTensor stays
/// null). (0, 1) is the byte-identical legacy whole-draft upload.
DsparkDeviceWeights upload_dspark_draft(
    const DsparkDraftWeights& weights, compute::DeviceBackend& backend,
    void* arena = nullptr, int64_t arena_capacity = 0,
    config::DsparkDraftWeightsQuant quant =
        static_cast<config::DsparkDraftWeightsQuant>(0) /* bf16 */,
    int rank = 0, int num_ranks = 1);

// ── V4 dflash GGUF checkpoint arm (ticket J; dspark_gguf_loader.cpp) ───────
// Dispatch rule: a checkpoint_path that IS a regular file ending in ".gguf"
// selects the dflash-GGUF loader; a directory keeps the speculators-v0.5
// byte-identical legacy path. parse_dspark_checkpoint_config /
// dspark_draft_bytes / upload_dspark_draft dispatch internally;
// load_dspark_draft cannot (the GGUF arm additionally needs the TARGET
// weights path for the aliased embed/lm_head) — callers with a Config use
// load_dspark_v4_gguf_draft directly on the GGUF branch.

/// True when `checkpoint_path` names a dflash GGUF FILE (vs a checkpoint dir).
bool dspark_checkpoint_is_gguf(const std::filesystem::path& checkpoint_path);

/// Header-only parse of the dflash GGUF → DsparkCheckpointConfig
/// (is_v4_dflash set; γ = dflash.block_size, aux ids = dflash.target_layers,
/// markov rank / vocab / confidence anatomy derived from the tensor infos).
DsparkCheckpointConfig parse_dspark_gguf_checkpoint_config(
    const std::filesystem::path& gguf_file);

/// Load the V4 dflash draft: draft GGUF tensors (Q8_0 → BF16 dequant at
/// load, MXFP4 experts packed verbatim, F32 heads verbatim, F32 norms →
/// BF16) + the target GGUF's token_embd/output as the aliased
/// embed_tokens/lm_head (resolve_gguf_files over `target_weights_path`).
/// Strict coverage (INV-DSPARK-CKPT): throws listing offenders on any
/// unmapped/missing/mis-shaped tensor.
DsparkDraftWeights load_dspark_v4_gguf_draft(
    const std::filesystem::path& gguf_file,
    const std::string& target_weights_path, bool use_mmap = true);

/// Device weight bytes of the V4 dflash draft (header-only; single-rank,
/// bf16/native — draft_weights_quant does not apply to a pre-quantized
/// artifact). Includes the target-aliased embed/lm_head and the per-layer
/// expert B_ptrs arrays.
int64_t dspark_gguf_draft_bytes(const std::filesystem::path& gguf_file);

/// Upload the V4 dflash draft onto `backend`'s GPU (single rank only).
DsparkDeviceWeights upload_dspark_v4_gguf_draft(
    const DsparkDraftWeights& weights, compute::DeviceBackend& backend,
    void* arena = nullptr, int64_t arena_capacity = 0);

}  // namespace layerstorm::model

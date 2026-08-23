// Fused attention dispatch driver: command handling, phase sequencing
// (guards → kv metadata → provisioning → arch execute → residual/mHC →
// fused gating → tiering sweeps). Moved verbatim from dispatch_attention.cpp
// (attention refactor V2 P1); the per-arch phase classes arrive later in
// P1/P2.

#include "daemon/command_dispatcher.h"
#include "daemon/attention/arch_base.h"          // attention refactor V2 P1
#include "daemon/attention/arch_mla.h"           // MLA phase hooks
#include "daemon/attention/arch_deepseek_v4.h"   // V4 phase hooks
#include "daemon/v4_kv_tiering.h"
#include "daemon/dispatch_detail.h"
#include "daemon/kv_shard_math.h"  // KVS-2: sharded-KV token→rank math
#include "daemon/kv_tiering_manager.h"  // GLM-25k: DSA-guided KV tiering

#include <algorithm>
#include <atomic>   // TD-DRIFT-ROOTCAUSE diagnostic dump
#include <cstdio>   // TD-DRIFT-ROOTCAUSE diagnostic dump
#include <cstdlib>  // TD-DRIFT-ROOTCAUSE getenv gate
#include <mutex>    // KVS-4: once-flag fail-closed warning
#include <vector>   // TD-DRIFT-ROOTCAUSE D2H staging

#include <spdlog/spdlog.h>

#include "compute/stream_manager.h"
#include "core/attention_device.h"
#include "core/cuda_hardware_query.h"  // EPM-1: host_register_pinned_portable
#include "core/device_backend.h"
#include "core/memory/numa_manager.h"  // EPM-1: NUMA-local D2H staging
#include "core/memory/page_allocator.h"
#include "speculation/epm_dump.h"      // EPM-1: routing-label dump
#include "compute/kernels/elementwise/residual_add.h"
#include "compute/kernels/mhc/mhc.h"
#include "compute/kernels/norm/rmsnorm.h"
#include "compute/kernels/moe/router_projection.h"  // F-1: fused gate in attention
#include "parallelism/dcp_communicator.h"
#include "parallelism/dcp_executor.h"
#include "compute/kernels/moe/hash_gating.h"  // V4-4 hash-layer gating
#include "sm120/gating/topk_gating.h"  // F-1: launch_topk_gating

namespace layerstorm::daemon {

namespace {
// TD-DRIFT-ROOTCAUSE: DIAGNOSTIC-ONLY routing dump (off by default; gated on
// LS_DRIFT_DUMP=<path>). The keeper's FETCH_AND_RUN path routes via the gate
// FUSED INTO ATTENTION (emit_gating), so the pre-argmax router logits + the
// selected top-K are produced HERE (not in dispatch_moe's self-gating branch).
// Dumps one binary record per (seq, layer, gpu) right after the top-K gate so
// two runs (same config twice, or decay-on vs off) can be diffed at the logit
// level to locate + classify the FIRST divergence. UNSET = zero work. When SET
// it adds a D2H + full device sync (timing changes; used to compare dump-on vs
// dump-on, and as the race probe: serialization must not change the routing).
// Record (LE): int32 hdr[6]={seq,layer,gpu,num_tokens,n_experts,topk};
//   float logits[nt*ne]; int32 idx[nt*topk]; float w[nt*topk].
void drift_dump_routing(compute::DeviceBackend* dev_be, void* stream,
                        const void* router_logits_dev,
                        const void* topk_indices_dev,
                        const void* topk_weights_dev,
                        int num_tokens, int n_experts, int topk,
                        int layer_idx, int gpu) {
    static const char* path = std::getenv("LS_DRIFT_DUMP");
    if (!path || !*path) return;
    if (!dev_be || !router_logits_dev || !topk_indices_dev || !topk_weights_dev)
        return;
    if (num_tokens <= 0 || n_experts <= 0 || topk <= 0) return;
    static std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return;
    static std::atomic<uint64_t> seq{0};
    const size_t nlog = static_cast<size_t>(num_tokens) * n_experts;
    const size_t nk   = static_cast<size_t>(num_tokens) * topk;
    std::vector<float>   logits(nlog);
    std::vector<int32_t> idx(nk);
    std::vector<float>   w(nk);
    dev_be->set_device();
    dev_be->memcpy_d2h_async(logits.data(), router_logits_dev,
                             nlog * sizeof(float), stream);
    dev_be->memcpy_d2h_async(idx.data(), topk_indices_dev,
                             nk * sizeof(int32_t), stream);
    dev_be->memcpy_d2h_async(w.data(), topk_weights_dev,
                             nk * sizeof(float), stream);
    dev_be->synchronize_device();
    const uint64_t s = seq.fetch_add(1);
    int32_t hdr[6] = {static_cast<int32_t>(s), layer_idx, gpu,
                      num_tokens, n_experts, topk};
    std::fwrite(hdr, sizeof(int32_t), 6, fp);
    std::fwrite(logits.data(), sizeof(float), nlog, fp);
    std::fwrite(idx.data(), sizeof(int32_t), nk, fp);
    std::fwrite(w.data(), sizeof(float), nk, fp);
    std::fflush(fp);
}
}  // namespace

// ── Fused compute command handler (IPC-8d) ────────────────────────────────

void CommandDispatcher::handle_fused_compute_command(const ipc::Command& cmd) {
    if (!deps_.sideband_base) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                    "fused compute: sideband not configured");
        return;
    }

    const auto type = static_cast<ipc::CmdType>(cmd.cmd_type);
    uint32_t layer_idx = 0;
    uint8_t emit_ckpt = 0;
    uint8_t ckpt_type = 0;

    bool dispatch_ok = false;

    // LS_ATTN_CHUNK_PROF: timing-event pair bracketing a chunk-shaped
    // attention dispatch on the primary kAttention stream (GPU stream-chain
    // wall from first to last enqueued op; reaped through the generic
    // compute_t_start/end path in poll_compute_completions).
    void* attn_prof_ev_start = nullptr;
    void* attn_prof_ev_end   = nullptr;
    int   attn_prof_gpu      = -1;

    switch (type) {
        case ipc::D_B_CMD_RUN_ATTENTION: {
            auto& p = cmd.run_attention;
            layer_idx = p.layer_idx;
            if (p.num_seqs == 0 || p.num_seqs > ipc::kMaxBatchDescriptors) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "run_attention: num_seqs out of range");
                return;
            }
            // TD-PREFILL-SUPERCHUNK: a sub-chunk's hidden rows + fused-gate
            // topk rows must fit the superchunk-sized buffers
            // (superchunk_rows_ = max(max_batch, min(superchunk, capacity))).
            if (p.row_offset != 0
                && p.row_offset + p.num_seqs
                       > static_cast<uint32_t>(superchunk_rows_)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "run_attention: row_offset + num_seqs exceeds the "
                            "superchunk row capacity");
                return;
            }
            if (attn_chunk_prof_ && p.is_prefill != 0 && deps_.stream_manager
                && deps_.dcp_executor && !deps_.dcp_executor->gpus().empty()) {
                const int g = deps_.dcp_executor->gpus()[0].position;
                if (g >= 0 && g < static_cast<int>(deps_.device_backends.size())
                    && deps_.device_backends[g]) {
                    auto* be = deps_.device_backends[g];
                    be->set_device();
                    attn_prof_ev_start = be->create_timing_event();
                    if (attn_prof_ev_start) {
                        be->record_event(attn_prof_ev_start,
                            deps_.stream_manager->stream(
                                g, compute::StreamId::kAttention));
                        attn_prof_gpu = g;
                    }
                }
            }
            dispatch_ok = dispatch_fused_attention(cmd);
            if (attn_prof_ev_start) {
                auto* be = deps_.device_backends[attn_prof_gpu];
                be->set_device();
                attn_prof_ev_end = be->create_timing_event();
                if (attn_prof_ev_end)
                    be->record_event(attn_prof_ev_end,
                        deps_.stream_manager->stream(
                            attn_prof_gpu, compute::StreamId::kAttention));
            }
            emit_ckpt = p.emit_checkpoint;
            ckpt_type = static_cast<uint8_t>(ipc::CheckpointType::kHiddenState);
            break;
        }
        case ipc::D_B_CMD_RUN_MOE: {
            auto& p = cmd.run_moe;
            layer_idx = p.layer_idx;
            // TD-PREFILL-SUPERCHUNK: MoE batch capacity may exceed the legacy
            // kMaxBatchDescriptors when a superchunk is configured.
            if (p.num_seqs == 0
                || p.num_seqs > static_cast<uint32_t>(moe_batch_capacity_)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "run_moe: num_seqs out of range");
                return;
            }
            if (p.moe_mode > 2) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "run_moe: invalid moe_mode");
                return;
            }
            dispatch_ok = dispatch_fused_moe(cmd);
            emit_ckpt = p.emit_checkpoint;
            ckpt_type = static_cast<uint8_t>(ipc::CheckpointType::kGatingOutput);
            break;
        }
        default:
            write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                        "unhandled fused compute command type");
            return;
    }

    // TD-40a: if dispatch failed, write error completion instead of PendingCompute
    if (!dispatch_ok) {
        if (attn_prof_ev_start) {  // LS_ATTN_CHUNK_PROF cleanup on error
            auto* be = deps_.device_backends[attn_prof_gpu];
            be->destroy_event(attn_prof_ev_start);
            if (attn_prof_ev_end) be->destroy_event(attn_prof_ev_end);
        }
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    last_internal_error_msg_
                        ? last_internal_error_cat_
                        : ipc::CmpErrorCategory::kComputeValidation,
                    last_internal_error_msg_
                        ? last_internal_error_msg_
                        : "fused compute: dispatch failed (see log)");
        return;
    }

    // Enqueue async completion via event on the correct stream for this command.
    // Fused commands dispatch on fixed streams (E1): attention on kAttention,
    // MoE on kExpertFfn. This overrides the advisory stream_id in the header.
    auto event_stream = (type == ipc::D_B_CMD_RUN_ATTENTION)
        ? compute::StreamId::kAttention
        : compute::StreamId::kExpertFfn;

    // TD-40f: record event on DcpExecutor's primary GPU (where work actually ran)
    // TODO:DEBT TD-51i: attention completion gpu_idx set to DCP primary GPU, differs from cmd.gpu_idx
    int event_gpu = static_cast<int>(cmd.gpu_idx);
    if (type == ipc::D_B_CMD_RUN_ATTENTION && deps_.dcp_executor &&
        !deps_.dcp_executor->gpus().empty()) {
        event_gpu = deps_.dcp_executor->gpus()[0].position;
    }

    void* event = create_and_record_event(event_gpu, event_stream);

    PendingCompute pc{};
    pc.cmd_seq    = cmd.cmd_seq;
    pc.gpu_idx    = static_cast<uint32_t>(event_gpu);
    pc.cmd_type   = cmd.cmd_type;
    pc.layer_idx  = layer_idx;
    pc.cuda_event = event;

    // TD-40e: is_draft checkpoint carries correct data_bytes
    // TODO:DEBT TD-51d: is_draft D2H copy not enacted — data_bytes set but no async memcpy enqueued
    if (emit_ckpt) {
        if (type == ipc::D_B_CMD_RUN_MOE) {
            // F-7: the MoE op publishes the seam routing (top-K weights+indices) to
            // kSeamCheckpointOff in dispatch_moe_internal. Carry the real
            // {kSeamRouting, offset, bytes} it recorded instead of the old zero-data
            // path, so the orchestrator/decider can read routing at the seam.
            pc.checkpoint_data = PendingCompute::CheckpointData{
                .checkpoint_type = last_seam_checkpoint_.checkpoint_type,
                .host_buf_offset = last_seam_checkpoint_.host_buf_offset,
                .data_bytes      = last_seam_checkpoint_.data_bytes,
            };
        } else {
            // Attention's kHiddenState path unchanged.
            uint32_t data_bytes = 0;
            if (type == ipc::D_B_CMD_RUN_ATTENTION &&
                cmd.run_attention.is_draft && deps_.live_config) {
                data_bytes = cmd.run_attention.num_seqs
                           * static_cast<uint32_t>(deps_.live_config->model.hidden_size)
                           * 2u;  // BF16
            }
            pc.checkpoint_data = PendingCompute::CheckpointData{
                .checkpoint_type = ckpt_type,
                .host_buf_offset = 0,
                .data_bytes = data_bytes,
            };
        }
    }

    // TD-89m: propagate routed expert miss count for MoE completions.
    if (type == ipc::D_B_CMD_RUN_MOE) {
        pc.routed_miss_count = last_moe_miss_count_;
    }

    // LS_ATTN_CHUNK_PROF: attach the dispatch decomposition + GPU timing pair.
    if (attn_prof_ev_start && attn_prof_ev_end && attn_prof_have_
        && static_cast<int>(pc.gpu_idx) == attn_prof_gpu) {
        pc.compute_t_start = attn_prof_ev_start;
        pc.compute_t_end   = attn_prof_ev_end;
        PendingCompute::AttnChunkProf ap;
        const auto ms = [](std::chrono::steady_clock::time_point a,
                           std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<float, std::milli>(b - a).count();
        };
        const auto now = std::chrono::steady_clock::now();
        ap.host_pre_ms  = ms(attn_prof_enter_, attn_prof_pre_exec_);
        ap.host_exec_ms = ms(attn_prof_pre_exec_, attn_prof_post_exec_);
        ap.host_post_ms = ms(attn_prof_post_exec_, now);
        ap.dispatch_end = now;
        pc.attn_prof = ap;
    } else if (attn_prof_ev_start) {
        // Pair unusable (end-event alloc failed / gpu mismatch) — drop it.
        auto* be = deps_.device_backends[attn_prof_gpu];
        be->destroy_event(attn_prof_ev_start);
        if (attn_prof_ev_end) be->destroy_event(attn_prof_ev_end);
    }

    pending_compute_.push_back(pc);
}

// ── KD-3a/3d: Fused attention dispatch ───────────────────────────────────

bool CommandDispatcher::dispatch_attention_internal(const InternalAttentionParams& p) {
    last_internal_error_msg_ = nullptr;  // TD-GOLDEN-KV-EXHAUST: reset per dispatch

    // LS_ATTN_CHUNK_PROF: stamp the host-side dispatch phases for chunk-shaped
    // attention (is_prefill / chunked-prefill). Zero work when the flag is off.
    const bool attn_prof_this =
        attn_chunk_prof_ && (p.is_prefill != 0 || p.chunk_len > 0);
    attn_prof_have_ = false;
    if (attn_prof_this)
        attn_prof_enter_ = std::chrono::steady_clock::now();

    // TD-40h/40a: require DcpExecutor
    if (!deps_.dcp_executor) {
        spdlog::warn("dispatch_attention: no dcp_executor, layer={}", p.layer_idx);
        return false;
    }

    const int dcp_size = deps_.dcp_executor->dcp_size();
    if (static_cast<int>(deps_.hidden_state_pairs.size()) < dcp_size) {
        spdlog::warn("dispatch_attention: hidden_state_pairs too small ({} < {})",
                     deps_.hidden_state_pairs.size(), dcp_size);
        return false;
    }

    // Wait for prior MoE copy-back (kExpertFfn → attn_buf) before this
    // layer's attention reads attn_buf on kAttention.  No-op on the first
    // layer (event unrecorded).  Mirrors forward_one_layer Step 0.
    // For TP>1: wait on ALL TP ranks — DcpExecutor reads all attn_bufs.
    if (deps_.stream_manager) {
        const bool tp_active = deps_.dcp_communicator
                             && deps_.dcp_communicator->is_active();
        if (tp_active) {
            const auto& tp_gpus = deps_.dcp_executor->gpus();
            for (int r = 0; r < static_cast<int>(tp_gpus.size()); ++r) {
                const int pos = tp_gpus[r].position;
                const int pi = resolve_pair_idx(static_cast<uint32_t>(pos));
                if (pi >= 0 && deps_.hidden_state_pairs[pi].moe_attn_event) {
                    deps_.stream_manager->wait_event(
                        pos, compute::StreamId::kAttention,
                        deps_.hidden_state_pairs[pi].moe_attn_event);
                }
            }
        } else {
            const int pair_idx = resolve_pair_idx(p.gpu_idx);
            if (pair_idx >= 0) {
                const auto& pair = deps_.hidden_state_pairs[pair_idx];
                if (pair.moe_attn_event) {
                    deps_.stream_manager->wait_event(
                        static_cast<int>(p.gpu_idx),
                        compute::StreamId::kAttention,
                        pair.moe_attn_event);
                }
            }
        }
    }

    // TD-PREFILL-NONDET diagnostic: layer-INPUT hidden of every rank at the
    // top of the attention dispatch (post moe_attn_event waits) — 'Ain '.
    // Layer 0's record is the embedding output. Env-gated, zero work off.
    if (deps_.live_config) {
        const int hs = deps_.live_config->model.hidden_size * deps_.hc_streams;
        for (size_t pi = 0; pi < deps_.hidden_state_pairs.size(); ++pi) {
            const auto& pr = deps_.hidden_state_pairs[pi];
            if (pr.attn_buf)
                seam_dump_hidden(0x206e6941u /*'Ain '*/,
                                 static_cast<int>(p.layer_idx),
                                 pr.gpu_position, pr.attn_buf,
                                 static_cast<int>(p.num_seqs), hs);
        }
    }

    // DSP-3: aux-hidden export hook (INV-DSPARK-AUX).  When the DSpark
    // runtime is armed and this layer is one of aux_hidden_state_layer_ids,
    // the primary rank's attn_buf currently holds the post-residual output
    // of layer_idx-1 (== this layer's input — the vLLM aux capture
    // convention).  The copy is enqueued on the primary kAttention stream
    // AFTER the moe_attn_event waits above, so the previous layer's MoE
    // commit is visible.  Read-only on target state; no-op when unarmed.
    // TD-DSPARK-SUPERCHUNK-CAPTURE: superchunk sub-launches (row_offset > 0)
    // capture too — the hook offsets the source by row_offset rows and the
    // runtime folds the chunk-major windows per slot at their absolute
    // positions, so long-prompt prefills arm the draft context.
    if (deps_.dspark) maybe_dspark_capture(p);

    // V4-7b (ticket H, resolves TD-V4-ATTN-ROUTING): deepseek_v4 routes to
    // DcpExecutor::execute_attention_v4 (via ArchDeepseekV4).
    const bool is_v4 = deps_.live_config &&
        deps_.live_config->model.architecture ==
            config::Architecture::deepseek_v4;
    // Attention refactor V2 P1 (arch_base.h): the model-special driver
    // phases run through the AttentionArch hooks. Selection is the SAME
    // is_v4 condition as the legacy inline branches; the arch objects are
    // stateless facades over this dispatcher (friends), constructed once.
    if (!arch_mla_) {
        arch_mla_ = std::make_unique<ArchMla>(*this);
        arch_v4_  = std::make_unique<ArchDeepseekV4>(*this);
    }
    AttentionArch& arch = is_v4 ? *arch_v4_ : *arch_mla_;

    // Phase A: arch shape legality gate + batch cap. V4: fail-closed shapes
    // + the verify-chunk descriptor bound (TD-V4-CHUNK-PREFILL); MLA:
    // max_batch_size (TD-40g: validate batch_size against DcpExecutor
    // buffer cap).
    int batch_cap = 0;
    if (!arch.validate_shape(p, batch_cap)) return false;

    const int batch_size = static_cast<int>(p.num_seqs);
    if (batch_size > batch_cap) {
        spdlog::error("dispatch_attention: batch_size {} > max {}",
                      batch_size, batch_cap);
        return false;
    }

    // TD-40b/c: use per-layer weights from Deps (resolves nullptr weights + RMSNorm)
    const int layer = static_cast<int>(p.layer_idx);
    std::vector<const parallelism::AttentionLayerWeights*> weight_ptrs(dcp_size);
    if (layer < static_cast<int>(deps_.per_layer_attn_weights.size()) &&
        static_cast<int>(deps_.per_layer_attn_weights[layer].size()) == dcp_size) {
        for (int r = 0; r < dcp_size; ++r)
            weight_ptrs[r] = &deps_.per_layer_attn_weights[layer][r];
    } else {
        spdlog::warn("dispatch_attention: no weights for layer {}", layer);
        return false;
    }

    // KD-4e: Build KV cache metadata from sideband batch descriptors.
    std::vector<const int*> seqlens_k_ptrs(dcp_size, nullptr);
    std::vector<const int*> block_table_ptrs(dcp_size, nullptr);
    std::vector<const int*> slot_mapping_ptrs(dcp_size, nullptr);
    // KVS-2 (sharded KV only): per-rank device GLOBAL seqlens + per-rank host
    // LOCAL shard lengths (see AttentionExecParams docs).
    std::vector<const int*> global_seqlens_ptrs(dcp_size, nullptr);
    std::vector<const int*> host_local_seqlens_ptrs(dcp_size, nullptr);
    const bool kv_sharded = kv_sharded_ && dcp_size >= 2;

    // GLM-25k (TD-KVT-SPEC-FORK): a rewind INTO demoted territory (the
    // step's k_append rewrites token_pos, so every position >= token_pos
    // must be HOT) re-promotes the affected cold pages BEFORE the kv-meta
    // build — the alloc seam un-neutralizes seq_pages_ and poisons the
    // dirty guard, so THIS step's block tables / slot mappings are built
    // from the fresh VRAM handles.  No-op unless the step actually reaches
    // below a demoted frontier (normal forward steps never do — the
    // retention window keeps the frontier behind the append position).
    // VRAM-full / missing seam fails the step CLOSED (capacity, not
    // correctness — INV-KVT-2).
    if (kv_tiering_ && kv_tiering_->has_demotions() && deps_.sideband_base) {
        const auto* rbe = reinterpret_cast<const ipc::BatchDescriptorEntry*>(
            deps_.sideband_base + ipc::IpcLayout::kBatchDescriptorOff);
        for (int b = 0; b < batch_size; ++b) {
            if (!kv_tiering_->seq_has_demotions(rbe[b].seq_id)) continue;
            if (!kv_tiering_->repromote_for_rewind(rbe[b].seq_id,
                                                   rbe[b].token_pos)) {
                last_internal_error_cat_ =
                    ipc::CmpErrorCategory::kKvPoolExhausted;
                last_internal_error_msg_ =
                    "attention: rewind into demoted KV territory — cold-page "
                    "re-promotion failed (VRAM capacity; fail-closed, "
                    "TD-KVT-SPEC-FORK)";
                return false;
            }
        }
    }

    const auto kv_meta_res = build_kv_metadata(batch_size, dcp_size);
    if (kv_meta_res == KvMetaResult::kFailed) {
        // TD-GOLDEN-KV-EXHAUST: never run attention with a missing slot
        // mapping — k_append would silently write to physical page 0.
        last_internal_error_cat_ = ipc::CmpErrorCategory::kKvPoolExhausted;
        last_internal_error_msg_ =
            "attention: KV page pool exhausted or unknown seq_id";
        return false;
    }
    const bool kv_meta_ok = (kv_meta_res == KvMetaResult::kOk);
    if (kv_meta_ok) {
        // TD-GOLDEN-KVMETA-PER-LAYER: scratch holds all layers — offset the
        // pointers to this layer's slice (layer stride = batch_size).
        const int L = kv_layers_ > 0 ? kv_layers_ : 1;
        const int lyr = (layer >= 0 && layer < L) ? layer : 0;
        const size_t bt_off = static_cast<size_t>(lyr) * batch_size
                            * max_blocks_per_seq_;
        const size_t sm_off = static_cast<size_t>(lyr) * batch_size;
        for (int r = 0; r < dcp_size; ++r) {
            const auto& m = kv_meta_scratch_[r];
            if (deps_.cuda_kernels_enabled) {
                if (m.dev_seqlens_k)
                    seqlens_k_ptrs[r] = static_cast<const int*>(m.dev_seqlens_k);
                if (m.dev_block_tables)
                    block_table_ptrs[r] =
                        static_cast<const int*>(m.dev_block_tables) + bt_off;
                if (m.dev_slot_mappings)
                    slot_mapping_ptrs[r] =
                        static_cast<const int*>(m.dev_slot_mappings) + sm_off;
                if (kv_sharded && m.dev_global_seqlens)
                    global_seqlens_ptrs[r] =
                        static_cast<const int*>(m.dev_global_seqlens);
            } else {
                seqlens_k_ptrs[r]    = m.host_seqlens_k.data();
                block_table_ptrs[r]  = m.host_block_tables.data() + bt_off;
                slot_mapping_ptrs[r] = m.host_slot_mappings.data() + sm_off;
                if (kv_sharded)
                    global_seqlens_ptrs[r] = host_global_seqlens_.data();
            }
            if (kv_sharded)
                host_local_seqlens_ptrs[r] = m.host_seqlens_k.data();
        }
    }

    parallelism::AttentionExecParams params{};
    params.layer_idx         = layer;
    params.batch_size        = batch_size;
    // TD-PREFILL-SUPERCHUNK: a sub-chunk reads/writes hidden rows
    // [row_offset, row_offset+batch_size). Offset the per-rank base pointers
    // here so the executor stays row-offset-agnostic for projections; the
    // executor uses batch_row_offset ONLY for its persistent sparse top-k
    // rows + IndexShare reuse keys.
    const size_t sc_row_bytes = static_cast<size_t>(p.row_offset)
        * (deps_.live_config ? deps_.live_config->model.hidden_size : 0)
        * deps_.hc_streams * 2;
    if (deps_.hc_streams > 1) {
        // V4-5b mHC: collapse the hc-stream residual to the attention module
        // input. hc_pre reads attn_buf rows [row_offset, row_offset+B) and
        // writes hc_attn_x rows [0, B) (+ the post/comb coefficients the
        // attention-side hc_post consumes after o_proj). The executor then
        // consumes hc_attn_x like any 4096-wide hidden.
        const auto& mcfg = deps_.live_config->model;
        for (int r = 0; r < dcp_size; ++r) {
            const auto& pair = deps_.hidden_state_pairs[r];
            const parallelism::AttentionLayerWeights* lw = nullptr;
            if (layer < static_cast<int>(deps_.per_layer_attn_weights.size()) &&
                pair.rank >= 0 &&
                pair.rank < static_cast<int>(deps_.per_layer_attn_weights[layer].size()))
                lw = &deps_.per_layer_attn_weights[layer][pair.rank];
            if (!lw || !lw->hc_attn_fn || !lw->hc_attn_base || !lw->hc_attn_scale ||
                r >= static_cast<int>(deps_.hc_attn_x.size()) ||
                !deps_.hc_attn_x[r] || !pair.attn_buf) {
                spdlog::error("dispatch_attention: mHC active but hc_attn "
                              "weights/scratch missing (layer {}, rank {})",
                              layer, r);
                return false;
            }
            deps_.device_backends[pair.gpu_position]->set_device();
            void* stream_r = deps_.stream_manager->stream(
                pair.gpu_position, compute::StreamId::kAttention);
            const void* residual_src =
                static_cast<const uint8_t*>(pair.attn_buf) + sc_row_bytes;
            compute::launch_mhc_pre(
                deps_.hc_attn_x[r], deps_.hc_attn_post[r], deps_.hc_attn_comb[r],
                residual_src, lw->hc_attn_fn, lw->hc_attn_scale, lw->hc_attn_base,
                mcfg.rms_norm_eps, mcfg.hc_eps, 2.0f, mcfg.hc_sinkhorn_iters,
                batch_size, deps_.hc_streams, mcfg.hidden_size, stream_r);
        }
        params.hidden_states = deps_.hc_attn_x.data();
    } else if (p.row_offset > 0 && sc_row_bytes > 0) {
        attn_bufs_offset_.resize(attn_bufs_.size());
        for (size_t i = 0; i < attn_bufs_.size(); ++i)
            attn_bufs_offset_[i] = attn_bufs_[i]
                ? static_cast<uint8_t*>(attn_bufs_[i]) + sc_row_bytes
                : nullptr;
        params.hidden_states = attn_bufs_offset_.data();
    } else {
        params.hidden_states = attn_bufs_.data();
    }
    params.batch_row_offset  = static_cast<int>(p.row_offset);
    params.superchunk        = p.superchunk;  // SC: V4 layer-sweep replays
    params.cache_stride_block = deps_.kv_cache_stride_block;
    params.cache_stride_row  = deps_.kv_cache_stride_row;
    params.page_size         = deps_.kv_page_size;
    params.weights           = weight_ptrs.data();
    params.use_graph         = (p.use_graph != 0) && (p.is_prefill == 0);

    // KD-4e: wire KV metadata (nullptr if build_kv_metadata returned false).
    if (kv_meta_ok) {
        // Host-side max KV length for the nongraph gather/dequant bound.
        // Replicated: per-rank host arrays are identical, rank 0 suffices.
        // Sharded (KVS-2): use the GLOBAL lengths — a safe upper bound on
        // every rank's local shard length.
        int max_seqlen = 0;
        if (kv_sharded) {
            for (int b = 0; b < batch_size; ++b)
                max_seqlen = std::max(max_seqlen, host_global_seqlens_[b]);
        } else {
            for (int b = 0; b < batch_size; ++b)
                max_seqlen = std::max(max_seqlen,
                                      kv_meta_scratch_[0].host_seqlens_k[b]);
        }
        params.max_seqlen_k      = max_seqlen;
        params.seqlens_k         = seqlens_k_ptrs.data();
        params.block_tables      = block_table_ptrs.data();
        params.max_blocks_per_seq = max_blocks_per_seq_;
        params.slot_mappings     = slot_mapping_ptrs.data();
        params.kv_cache_ptrs     = kv_cache_base_ptrs_.data();
        if (kv_sharded) {
            // KVS-2: sharded-KV extras — per-rank device GLOBAL seqlens
            // (RoPE positions) + per-rank host LOCAL shard lengths.
            params.global_seqlens_k     = global_seqlens_ptrs.data();
            params.host_local_seqlens_k = host_local_seqlens_ptrs.data();
        }
    }

    // TD-51h RESOLVED: is_sparse remains false until sparse_indices are populated
    // by DSA indexer infrastructure. Dense prefill is correct for all layers until then.

    // Phase B: arch step staging — MLA: the DSA indexer coverage state
    // machine + paged indexer-K provisioning + chunk-descriptor synthesis
    // (arch_mla.cpp); V4: the chunk-descriptor copy (arch_deepseek_v4.cpp).
    if (!arch.stage_step(p, params, batch_size, layer, dcp_size,
                         kv_meta_ok))
        return false;

    // Per-entry host lengths (kv-meta staging). Replicated: rank rows are
    // identical, rank 0 suffices. Sharded (KVS-2): rank rows hold LOCAL
    // shard lengths — hand the executor the GLOBAL lengths here (positions,
    // indexer keys); per-rank locals travel via host_local_seqlens_k.
    if (kv_meta_ok)
        params.host_seqlens_k = kv_sharded
            ? host_global_seqlens_.data()
            : kv_meta_scratch_[0].host_seqlens_k.data();

    if (attn_prof_this)
        attn_prof_pre_exec_ = std::chrono::steady_clock::now();

    // Phase C: arch execution — V4: side-tier provisioning + V4Step
    // assembly + execute_attention_v4 + the V4 KVT demote sweep
    // (arch_deepseek_v4.cpp); MLA: DcpExecutor::execute_attention
    // (arch_mla.cpp). Errors surface as command completions.
    if (!arch.execute(p, params, batch_size, layer, dcp_size, kv_meta_ok))
        return false;

    if (attn_prof_this) {
        attn_prof_post_exec_ = std::chrono::steady_clock::now();
        attn_prof_have_ = true;
    }

    // KD-R2: Wire attention residual — h_attn = h + attn_output.
    // DcpExecutor wrote o_proj+allreduce to hidden_out_[rank]. Copy to
    // pair.moe_buf and add pair.attn_buf as the residual connection.
    if (deps_.cuda_kernels_enabled) {
        const auto& attn_out = deps_.dcp_executor->hidden_out();
        const int hidden_size = deps_.live_config
            ? deps_.live_config->model.hidden_size : 0;
        const size_t copy_bytes =
            static_cast<size_t>(batch_size) * hidden_size * 2;  // BF16

        // TD-PREFILL-SUPERCHUNK: the sub-chunk's post-attention hidden rows
        // land at [row_offset, row_offset+batch_size) of moe_buf — moe_buf IS
        // the superchunk staging (zero-copy): after K sub-launches it holds
        // all K×chunk rows and ONE MoE command consumes them.
        // V4-5b mHC: rows are hc_streams*hidden wide and the residual update
        // is hc_post (R' = post·y + combᵀ·R) instead of memcpy+add.
        const size_t row_bytes = static_cast<size_t>(p.row_offset)
                               * hidden_size * deps_.hc_streams * 2;
        for (int r = 0; r < dcp_size; ++r) {
            const auto& pair = deps_.hidden_state_pairs[r];
            if (!pair.moe_buf || !attn_out[r] || !pair.attn_buf
                || hidden_size <= 0)
                continue;

            void* stream_r = deps_.stream_manager->stream(
                pair.gpu_position, compute::StreamId::kAttention);
            void* moe_dst = static_cast<uint8_t*>(pair.moe_buf) + row_bytes;
            const void* attn_src =
                static_cast<const uint8_t*>(pair.attn_buf) + row_bytes;

            if (deps_.hc_streams > 1) {
                compute::launch_mhc_post(
                    moe_dst, attn_out[r], attn_src,
                    deps_.hc_attn_post[r], deps_.hc_attn_comb[r],
                    batch_size, deps_.hc_streams, hidden_size, stream_r);
                continue;
            }

            // Copy attn_output to moe_buf (same GPU, D2D).
            deps_.device_backends[pair.gpu_position]->memcpy_d2d_async(
                moe_dst, attn_out[r], copy_bytes, stream_r);

            // Residual add: moe_buf += attn_buf
            compute::launch_residual_add(
                moe_dst, attn_src,
                batch_size * hidden_size, stream_r);
        }

        // F-1: fused gate. When emit_gating is set, run the router projection +
        // topk gating here — the post-attn hidden (pair.moe_buf, just residual-
        // added and TP all-reduced above on kAttention) is the only input gating
        // needs and is ready now. Result (topk_weights/topk_indices) is written
        // into moe_scratch_[gpu], the same scratch the MoE step reads. This is
        // the earliest valid point; default off keeps the path byte-for-byte
        // unchanged.
        //
        // Reuses the exact gate dispatch_moe_internal runs (post-attn RMSNorm →
        // launch_router_projection → launch_topk_gating). Runs on kAttention,
        // ordered in-stream after the residual add and BEFORE the attn_moe_event
        // record below — so the MoE step's wait on attn_moe_event also covers
        // the gate's writes to moe_scratch_ (no cross-stream race). Dense layers
        // (< first_k_dense_replace) and layers without router weights are skipped
        // (no routed experts → nothing to gate).
        if (p.emit_gating && deps_.live_config) {
            const auto& mc = deps_.live_config->model;
            const int n_experts   = mc.n_routed_experts;
            const int topk        = mc.num_experts_per_tok;
            // #16: MTP layers (>= num_hidden_layers) are full MoE blocks with
            // their own router (uploaded at index num_hidden_layers + mi) —
            // include them so the MTP draft step can ride the production
            // routing-export + FETCH_AND_RUN seam.
            const bool is_routed_layer = layer >= mc.first_k_dense_replace
                                       && layer < mc.num_hidden_layers
                                              + mc.num_nextn_predict_layers
                                       && n_experts > 0 && topk > 0
                                       && hidden_size > 0;
            if (is_routed_layer) {
                for (int r = 0; r < dcp_size; ++r) {
                    const auto& pair = deps_.hidden_state_pairs[r];
                    const uint32_t gpu = static_cast<uint32_t>(pair.gpu_position);
                    if (gpu >= moe_scratch_.size())
                        continue;
                    auto& scratch = moe_scratch_[gpu];
                    if (!pair.moe_buf || !scratch.normalized_hidden
                        || !scratch.router_logits || !scratch.topk_weights
                        || !scratch.topk_indices)
                        continue;

                    // Resolve per-rank post-attn norm + per-GPU router weight.
                    const void* norm_w = nullptr;
                    if (layer < static_cast<int>(deps_.per_layer_attn_weights.size())) {
                        const int rr = pair.rank;
                        if (rr >= 0 && rr < static_cast<int>(
                                deps_.per_layer_attn_weights[layer].size()))
                            norm_w = deps_.per_layer_attn_weights[layer][rr]
                                         .post_attention_layernorm;
                    }
                    const void* router_w = nullptr;
                    if (layer < static_cast<int>(deps_.router_weight_ptrs.size())
                        && gpu < deps_.router_weight_ptrs[layer].size())
                        router_w = deps_.router_weight_ptrs[layer][gpu];
                    if (!norm_w || !router_w)
                        continue;  // no router on this layer/GPU — nothing to gate

                    // Match dispatch_moe_internal: set active device for the
                    // router projection GEMM + kernel launches on this rank.
                    deps_.device_backends[pair.gpu_position]->set_device();
                    void* stream_g = deps_.stream_manager->stream(
                        pair.gpu_position, compute::StreamId::kAttention);

                    // TD-PREFILL-SUPERCHUNK: gate the sub-chunk's rows —
                    // input from moe_buf at the row offset, topk stored at
                    // the SAME row offset in moe_scratch_ (the superchunk
                    // FETCH_AND_RUN consumes the full K×chunk topk via
                    // use_precomputed_gating, TD-FAR-GATING). normalized_
                    // hidden/router_logits stay launch-local rows [0, B).
                    const void* gate_in = static_cast<const uint8_t*>(
                        pair.moe_buf) + static_cast<size_t>(p.row_offset)
                        * hidden_size * deps_.hc_streams * 2;
                    // V4-5b mHC: the gate input is the FFN-stage collapsed x
                    // — recompute hc_pre(ffn) launch-locally into scratch.hc_x
                    // rows [0, B) (the MoE command recomputes the full batch
                    // bit-identically before consuming hc_post/hc_comb).
                    if (deps_.hc_streams > 1) {
                        const auto& lwg =
                            deps_.per_layer_attn_weights[layer][pair.rank];
                        if (!lwg.hc_ffn_fn || !scratch.hc_x) {
                            spdlog::error("fused gate: mHC active but hc_ffn "
                                          "weights/scratch missing (layer {})",
                                          layer);
                            continue;
                        }
                        compute::launch_mhc_pre(
                            scratch.hc_x, scratch.hc_post, scratch.hc_comb,
                            gate_in, lwg.hc_ffn_fn, lwg.hc_ffn_scale,
                            lwg.hc_ffn_base, mc.rms_norm_eps, mc.hc_eps, 2.0f,
                            mc.hc_sinkhorn_iters, batch_size,
                            deps_.hc_streams, hidden_size, stream_g);
                        gate_in = scratch.hc_x;
                    }
                    float* topk_w_dst =
                        static_cast<float*>(scratch.topk_weights)
                        + static_cast<size_t>(p.row_offset) * topk;
                    int32_t* topk_i_dst =
                        static_cast<int32_t*>(scratch.topk_indices)
                        + static_cast<size_t>(p.row_offset) * topk;

                    // Post-attn RMSNorm of the residual-added hidden.
                    compute::launch_rmsnorm(
                        scratch.normalized_hidden, gate_in, norm_w,
                        mc.rms_norm_eps, batch_size, hidden_size,
                        compute::NormDtype::kBFloat16, stream_g);

                    // Router projection: normalized hidden → router_logits.
                    compute::launch_router_projection(
                        static_cast<float*>(scratch.router_logits),
                        scratch.normalized_hidden, router_w,
                        batch_size, n_experts, hidden_size, stream_g);

                    // V4-4 hash layers (layer < num_hash_layers): expert ids
                    // come from tid2eid[token_id], weights from the router
                    // logits restricted to those experts; exp_probs_b never
                    // applies. Otherwise: learned TopK gating.
                    if (layer < deps_.moe_hash_layers) {
                        const int32_t* table = nullptr;
                        if (layer < static_cast<int>(
                                deps_.hash_gating_table_ptrs.size())
                            && gpu < deps_.hash_gating_table_ptrs[layer].size())
                            table = deps_.hash_gating_table_ptrs[layer][gpu];
                        const int32_t* tok_ids = scratch.moe_token_ids
                            ? static_cast<const int32_t*>(
                                  scratch.moe_token_ids) + p.row_offset
                            : nullptr;
                        if (!table || !tok_ids) {
                            spdlog::error(
                                "fused gate: hash layer {} on gpu {} missing "
                                "{} — routing skipped",
                                layer, gpu,
                                table ? "token-id scratch" : "tid2eid table");
                            continue;
                        }
                        compute::HashGatingParams hp{};
                        hp.num_tokens            = batch_size;
                        hp.num_experts           = n_experts;
                        hp.topk                  = topk;
                        hp.vocab_size            = mc.vocab_size;
                        hp.routed_scaling_factor =
                            static_cast<float>(mc.routed_scaling_factor);
                        hp.renormalize           = mc.norm_topk_prob;
                        hp.scoring_func          =
                            to_scoring_func(mc.gating_score_fn);
                        compute::launch_hash_gating(
                            topk_w_dst, topk_i_dst,
                            static_cast<const float*>(scratch.router_logits),
                            table, tok_ids, hp, stream_g);
                    } else {
                    // TopK gating: router_logits → topk_weights, topk_indices.
                    compute::TopkGatingParams gp{};
                    gp.num_tokens            = batch_size;
                    gp.num_experts           = n_experts;
                    gp.topk                  = topk;
                    gp.n_group               = mc.n_group;
                    gp.topk_group            = mc.topk_group;
                    gp.routed_scaling_factor =
                        static_cast<float>(mc.routed_scaling_factor);
                    gp.renormalize           = mc.norm_topk_prob;
                    // V4-4a: sigmoid (V3.2/GLM) vs sqrtsoftplus (V4).
                    gp.scoring_func          =
                        to_scoring_func(mc.gating_score_fn);

                    const float* bias = nullptr;
                    if (layer < static_cast<int>(deps_.gating_bias_ptrs.size())
                        && gpu < deps_.gating_bias_ptrs[layer].size())
                        bias = deps_.gating_bias_ptrs[layer][gpu];

                    compute::launch_topk_gating(
                        topk_w_dst,
                        topk_i_dst,
                        static_cast<const float*>(scratch.router_logits),
                        bias, gp, stream_g);
                    }  // V4-4 hash/learned gating branch

                    // TD-DRIFT-ROOTCAUSE: gated logit-level routing dump (off by
                    // default). This is the keeper's actual routing producer.
                    drift_dump_routing(
                        deps_.device_backends[pair.gpu_position], stream_g,
                        scratch.router_logits, topk_i_dst,
                        topk_w_dst, batch_size, n_experts, topk,
                        layer, static_cast<int>(gpu));

                    // EPM-1 (Phase 29): routing-label training dump (off by
                    // default; config/LS_EPM_DUMP-gated). PRIMARY rank only —
                    // this is the exact topk_indices/topk_weights buffer set
                    // the F-3 routing export and the FETCH_AND_RUN expert
                    // list are built from. One int compare when off.
                    if (r == 0 && epm_dump_state_ != 0)
                        epm_capture_routing(p, layer, batch_size, gpu,
                                            stream_g);
                }

                // F-3: export the primary rank's routed top-K to the canonical
                // routing slot at attention-end — the orchestrator reads routing
                // BEFORE the MoE op (prefetch seam). Same slot/layout as F-4's
                // MoE-side producer (publish_routing_export). On kAttention,
                // ordered before attn_moe_event so the slot is filled when the
                // op's CMP_COMPUTE_DONE fires.
                if (p.store_gating && dcp_size > 0) {
                    const auto& pair0 = deps_.hidden_state_pairs[0];
                    const uint32_t gpu0 =
                        static_cast<uint32_t>(pair0.gpu_position);
                    if (gpu0 < moe_scratch_.size()) {
                        deps_.device_backends[pair0.gpu_position]->set_device();
                        void* stream0 = deps_.stream_manager->stream(
                            pair0.gpu_position, compute::StreamId::kAttention);
                        // TD-PREFILL-SUPERCHUNK: export THIS sub-chunk's rows
                        // (stored at row_offset) to sideband rows [0, B).
                        publish_routing_export(gpu0, batch_size, topk,
                                               static_cast<uint32_t>(layer),
                                               stream0,
                                               static_cast<int>(p.row_offset));
                    }
                }
            }
        }

        // KD-R2: record sync event after residual add so MoE on
        // kExpertFfn can wait before reading moe_buf.
        for (int r = 0; r < dcp_size; ++r) {
            const auto& pair = deps_.hidden_state_pairs[r];
            if (!pair.moe_buf || !attn_out[r] || !pair.attn_buf
                || hidden_size <= 0)
                continue;  // skip ranks where residual add was not executed
            if (pair.attn_moe_event) {
                deps_.stream_manager->record_event(
                    pair.attn_moe_event,
                    pair.gpu_position,
                    compute::StreamId::kAttention);
            }
        }
    }

    return true;
}

// ── EPM-1 (Phase 29): routing-label training dump ───────────────────────────
// Tap at the F-1 attention-end gate — the production routing producer for
// decode under FETCH_AND_RUN (the InternalMoeParams path consumes these very
// topk buffers via use_precomputed_gating; the F-3 routing export the
// orchestrator builds the FETCH_AND_RUN expert list from is a byte-copy of
// the same device buffers). Dumps, per (seq, decode position), one EPMR row
// per routed layer: full 256 pre-top-k router logits (FP32 D2H, FP16 on
// host — decision (A)) + the top-8 ids/weights. Decode-only (batch 1,
// non-draft, non-prefill); MTP layers (>= num_hidden_layers) excluded.
// OFF (default): the call site pays one int compare (epm_dump_state_).
// ON: adds a D2H + device sync per routed layer — collection mode only.

void CommandDispatcher::epm_capture_routing(const InternalAttentionParams& p,
                                            int layer, int batch_size,
                                            uint32_t gpu, void* stream) {
    // Lazy one-time resolve of the enable gate (config + LS_EPM_DUMP env).
    if (epm_dump_state_ < 0) {
        std::string dir;
        if (deps_.live_config)
            dir = speculation::epm_dump_dir(
                deps_.live_config->speculation.dspark);
        if (dir.empty()) {
            epm_dump_state_ = 0;
        } else {
            const auto& mc = deps_.live_config->model;
            epm_routing_dump_ =
                std::make_unique<speculation::EpmRoutingDumper>(
                    dir, mc.n_routed_experts, mc.num_experts_per_tok);
            if (!epm_routing_dump_->ok()) epm_routing_dump_.reset();
            epm_dump_state_ = epm_routing_dump_ ? 1 : 0;
        }
    }
    if (epm_dump_state_ != 1 || !deps_.live_config || !deps_.sideband_base)
        return;

    // Decode positions only: single-token, non-draft, non-prefill steps of
    // REAL routed layers (verify feeds + plain AR feeds — the AR positions
    // feed the EPM-2 B0 temporal-locality baseline for free).
    if (batch_size != 1 || p.is_draft || p.is_prefill) return;
    const auto& mc = deps_.live_config->model;
    if (layer < mc.first_k_dense_replace || layer >= mc.num_hidden_layers)
        return;
    if (gpu >= moe_scratch_.size() ||
        gpu >= deps_.device_backends.size() || !deps_.device_backends[gpu])
        return;
    const auto& scratch = moe_scratch_[gpu];
    if (!scratch.router_logits || !scratch.topk_indices ||
        !scratch.topk_weights)
        return;
    auto* dev_be = deps_.device_backends[gpu];
    const int n_experts = mc.n_routed_experts;
    const int topk = mc.num_experts_per_tok;
    if (n_experts <= 0 || topk <= 0) return;

    // Lazy D2H staging, NUMA-local to the SOURCE (primary-rank) GPU's home
    // node (the P-22 allocate_for_gpu pattern) then register-pinned;
    // fallback = backend pinned alloc when no NumaManager is wired (tests).
    if (!epm_route_host_) {
        const size_t need =
            static_cast<size_t>(n_experts) * sizeof(float) +
            static_cast<size_t>(topk) * (sizeof(int32_t) + sizeof(float));
        if (deps_.numa_manager) {
            auto buf = deps_.numa_manager->allocate_for_gpu(
                need, static_cast<int>(gpu));
            epm_route_host_ = buf.data;
            epm_route_host_bytes_ = buf.size;
            epm_route_host_node_ = buf.numa_node;
            epm_route_host_from_numa_ = true;
            epm_route_host_registered_ =
                core::host_register_pinned_portable(buf.data, buf.size) == 0;
        } else {
            epm_route_host_ = dev_be->host_alloc_pinned(need);
            epm_route_host_bytes_ = need;
        }
        epm_route_host_gpu_ = static_cast<int>(gpu);
        if (!epm_route_host_) {
            spdlog::error("EPM-1: routing D2H staging alloc failed — "
                          "routing dump disabled");
            epm_routing_dump_.reset();
            epm_dump_state_ = 0;
            return;
        }
        spdlog::info("EPM-1: routing dump staging armed on gpu {} "
                     "(node {}, {} B)", gpu, epm_route_host_node_,
                     epm_route_host_bytes_);
    }
    if (epm_route_host_gpu_ != static_cast<int>(gpu))
        return;  // staging is bound to the primary rank's GPU

    const auto* bd = reinterpret_cast<const ipc::BatchDescriptorEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kBatchDescriptorOff);
    const uint64_t seq_id = bd[0].seq_id;
    const uint32_t token_pos = bd[0].token_pos;

    auto* hl = static_cast<float*>(epm_route_host_);
    auto* hi = reinterpret_cast<int32_t*>(hl + n_experts);
    auto* hw = reinterpret_cast<float*>(hi + topk);
    dev_be->set_device();
    dev_be->memcpy_d2h_async(hl, scratch.router_logits,
                             static_cast<size_t>(n_experts) * sizeof(float),
                             stream);
    dev_be->memcpy_d2h_async(hi, scratch.topk_indices,
                             static_cast<size_t>(topk) * sizeof(int32_t),
                             stream);
    dev_be->memcpy_d2h_async(hw, scratch.topk_weights,
                             static_cast<size_t>(topk) * sizeof(float),
                             stream);
    dev_be->synchronize_device();

    epm_routing_dump_->add_row(seq_id, token_pos,
                               static_cast<uint32_t>(layer), hl, hi, hw,
                               layer == mc.num_hidden_layers - 1);
}

std::pair<int, uint32_t>
CommandDispatcher::indexer_coverage(uint64_t seq_id) const {
    // mode == kUnset ⇔ untracked: a dispatched step always leaves the
    // coverage machine in a non-kUnset mode, so the default-initialized
    // state is exactly the old "no map entry" case.
    const auto* st = find_seq(seq_id);
    if (!st || st->indexer_cov.mode == IndexerSeqMode::kUnset) return {-1, 0};
    return {static_cast<int>(st->indexer_cov.mode), st->indexer_cov.next_pos};
}

bool CommandDispatcher::dispatch_fused_attention(const ipc::Command& cmd) {
    InternalAttentionParams p{};
    p.layer_idx   = cmd.run_attention.layer_idx;
    p.num_seqs    = cmd.run_attention.num_seqs;
    p.gpu_idx     = cmd.gpu_idx;
    p.is_prefill  = cmd.run_attention.is_prefill;
    p.use_graph   = cmd.run_attention.use_graph;
    p.is_draft    = cmd.run_attention.is_draft;
    p.chunk_start = cmd.run_attention.chunk_start;
    p.chunk_len   = cmd.run_attention.chunk_len;
    p.emit_gating = cmd.run_attention.emit_gating != 0;  // F-1
    p.store_gating = cmd.run_attention.store_gating != 0;  // F-3
    p.row_offset  = cmd.run_attention.row_offset;         // TD-PREFILL-SUPERCHUNK
    p.superchunk  = cmd.run_attention.superchunk != 0;    // TD-PREFILL-SUPERCHUNK
    return dispatch_attention_internal(p);
}

}  // namespace layerstorm::daemon

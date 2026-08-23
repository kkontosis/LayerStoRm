// Standalone compute kernel dispatch.
// Part of CommandDispatcher — see command_dispatcher.h.

#include "daemon/command_dispatcher.h"
#include "daemon/dispatch_detail.h"
#include "compute/kernels/mhc/mhc.h"

#include <algorithm>

#include <spdlog/spdlog.h>

#include "compute/graphs/graph_registry.h"
#include "compute/stream_manager.h"
#include "core/attention_device.h"
#include "core/device_backend.h"
#include "core/expert_device.h"
#include "daemon/buffer_registry.h"
#include "parallelism/dcp_communicator.h"
#include "parallelism/dcp_executor.h"

// Standalone kernel launchers (KD-2)
#include "compute/kernels/sampling/sampling.h"
#include "compute/kernels/embedding/embedding.h"
#include "compute/kernels/confidence/confidence.h"
#include "compute/kernels/norm/rmsnorm.h"
#include "compute/kernels/moe/hash_gating.h"  // V4-4 hash-layer gating
#include "sm120/gating/topk_gating.h"
#include "smxx/quant/dynamic_fp8_quant.h"

namespace layerstorm::daemon {

// ── TD-GOLDEN-EMB-OOB: TP-sharded embedding lookup ─────────────────────────

int CommandDispatcher::embedding_tp_degree() const {
    if (!deps_.live_config) return 1;
    const int mode = deps_.live_config->memory.tp_mode_per_layer.embedding;
    if (mode > 0) return mode;
    return deps_.dcp_executor ? std::max(1, deps_.dcp_executor->dcp_size()) : 1;
}

bool CommandDispatcher::dispatch_embedding_lookup_sharded(int num_tokens,
                                                          uint32_t row_offset) {
    if (!deps_.cuda_kernels_enabled || !deps_.live_config
        || !deps_.stream_manager || !deps_.sideband_base
        || !deps_.dcp_executor || !deps_.dcp_communicator
        || !deps_.dcp_communicator->is_active())
        return false;

    const int dcp_size = deps_.dcp_executor->dcp_size();
    const int embed_tp = embedding_tp_degree();
    if (embed_tp <= 1 || embed_tp != dcp_size) {
        spdlog::error("embedding_lookup_sharded: embed_tp {} vs dcp_size {} "
                      "unsupported", embed_tp, dcp_size);
        return false;
    }
    if (static_cast<int>(deps_.hidden_state_pairs.size()) < dcp_size
        || static_cast<int>(attn_bufs_.size()) < dcp_size)
        return false;

    const auto& mc = deps_.live_config->model;
    const int local_vocab = mc.vocab_size / embed_tp;
    if (local_vocab * embed_tp != mc.vocab_size) {
        spdlog::error("embedding_lookup_sharded: vocab {} not divisible by "
                      "embed_tp {}", mc.vocab_size, embed_tp);
        return false;
    }

    static constexpr int kMaxTp = 8;
    if (dcp_size > kMaxTp) return false;
    void* streams[kMaxTp];

    // Validate every rank's resources before launching anything.
    for (int r = 0; r < dcp_size; ++r) {
        const auto& pair = deps_.hidden_state_pairs[r];
        const auto pos = static_cast<size_t>(pair.gpu_position);
        if (!pair.attn_buf
            || pos >= embedding_token_scratch_.size()
            || !embedding_token_scratch_[pos]
            || pos >= deps_.embedding_table_ptrs.size()
            || !deps_.embedding_table_ptrs[pos])
            return false;
        streams[r] = deps_.stream_manager->stream(
            pair.gpu_position, compute::StreamId::kAttention);
        if (!streams[r]) return false;
    }

    const auto* host_token_ids = reinterpret_cast<const int32_t*>(
        deps_.sideband_base + ipc::IpcLayout::kTokenIdsOff);

    // TD-PREFILL-NONDET (root cause): the embedding write into attn_buf on
    // kAttention RACED the PREVIOUS step's last MoE residual-add/commit
    // (kExpertFfn → attn_buf) at the prefill chunk seam. Every attn_buf
    // consumer waits the pair's moe_attn_event (dispatch_attention Step 0,
    // forward_one_layer, output head) — but embedding, the FIRST WRITER of
    // the next chunk, did not: the command-level CMP of the previous
    // FETCH_AND_RUN covers only the PRIMARY rank's kExpertFfn event, so a
    // non-primary rank's residual add could still be in flight and land
    // AFTER the embedding overwrote the rows (hidden = embedding +
    // stale-layer FFN output — all rows perturbed at once, the chunk-2
    // first-gate fork signature). Decode was immune only because the
    // interposed OUTPUT_HEAD waits the event on the same kAttention stream.
    // Wait each rank's moe_attn_event before any embedding write.
    for (int r = 0; r < dcp_size; ++r) {
        const auto& pair = deps_.hidden_state_pairs[r];
        if (pair.moe_attn_event)
            deps_.stream_manager->wait_event(
                pair.gpu_position, compute::StreamId::kAttention,
                pair.moe_attn_event);
    }

    // TD-PREFILL-SUPERCHUNK: destination rows [row_offset, row_offset+n).
    // V4-5b mHC: residual rows are hc_streams*hidden wide; the lookup goes
    // through a collapsed 4096-wide staging (moe_scratch_.hc_x), the
    // allreduce runs at the collapsed width, and the repeat-expansion writes
    // the hc-stream rows (llama.cpp hc_init: R = repeat(embed, hc)).
    const int hc = deps_.hc_streams;
    const size_t row_bytes = static_cast<size_t>(row_offset)
                           * mc.hidden_size * hc * 2;

    // Masked lookup on every rank: rows outside the rank's vocab shard come
    // out zero, so the allreduce-sum reconstructs the full embedding on all
    // ranks (mirrors the lm_head local_vocab + collective pattern).
    for (int r = 0; r < dcp_size; ++r) {
        const auto& pair = deps_.hidden_state_pairs[r];
        const int pos = pair.gpu_position;
        const int rank = pair.rank >= 0 ? pair.rank : r;
        deps_.device_backends[pos]->set_device();
        auto* device_token_ids =
            static_cast<int32_t*>(embedding_token_scratch_[pos]);
        deps_.device_backends[pos]->memcpy_h2d_async(
            device_token_ids, host_token_ids,
            static_cast<size_t>(num_tokens) * sizeof(int32_t), streams[r]);
        // V4-4 hash gating: persist this launch's token ids AT row_offset in
        // the per-GPU step buffer (embedding_token_scratch_ is per-launch at
        // base row 0 — a later sub-chunk overwrites it before gating replays
        // under TD-PREFILL-SUPERCHUNK). Same kAttention stream ⇒ in-stream
        // ordered before any fused gate / MoE gate that reads it.
        if (deps_.moe_hash_layers > 0
            && static_cast<size_t>(pos) < moe_scratch_.size()
            && moe_scratch_[pos].moe_token_ids) {
            deps_.device_backends[pos]->memcpy_h2d_async(
                static_cast<int32_t*>(moe_scratch_[pos].moe_token_ids)
                    + row_offset,
                host_token_ids,
                static_cast<size_t>(num_tokens) * sizeof(int32_t), streams[r]);
        }
        void* lookup_dst = static_cast<uint8_t*>(pair.attn_buf) + row_bytes;
        if (hc > 1) {
            if (static_cast<size_t>(pos) >= moe_scratch_.size()
                || !moe_scratch_[pos].hc_x) {
                spdlog::error("embedding: mHC active but hc_x staging missing "
                              "on gpu {}", pos);
                return false;
            }
            lookup_dst = moe_scratch_[pos].hc_x;
        }
        compute::launch_embedding_lookup(
            lookup_dst,
            deps_.embedding_table_ptrs[pos],
            device_token_ids, num_tokens, mc.vocab_size, mc.hidden_size,
            compute::EmbeddingDtype::kBFloat16, streams[r],
            rank * local_vocab, local_vocab);
    }

    if (hc > 1) {
        // Allreduce the collapsed staging, then expand into the hc-stream
        // residual rows on every rank (in-stream on the same kAttention
        // streams — ordered before any consumer).
        hc_embed_stage_.resize(dcp_size);
        for (int r = 0; r < dcp_size; ++r)
            hc_embed_stage_[r] =
                moe_scratch_[deps_.hidden_state_pairs[r].gpu_position].hc_x;
        deps_.dcp_communicator->allreduce_hidden(
            hc_embed_stage_.data(), num_tokens, streams);
        for (int r = 0; r < dcp_size; ++r) {
            const auto& pair = deps_.hidden_state_pairs[r];
            deps_.device_backends[pair.gpu_position]->set_device();
            compute::launch_hc_expand_repeat(
                static_cast<uint8_t*>(pair.attn_buf) + row_bytes,
                hc_embed_stage_[r], num_tokens, hc, mc.hidden_size,
                streams[r]);
        }
    } else if (row_bytes == 0) {
        deps_.dcp_communicator->allreduce_hidden(
            attn_bufs_.data(), num_tokens, streams);
    } else {
        attn_bufs_offset_.resize(attn_bufs_.size());
        for (size_t i = 0; i < attn_bufs_.size(); ++i)
            attn_bufs_offset_[i] = attn_bufs_[i]
                ? static_cast<uint8_t*>(attn_bufs_[i]) + row_bytes
                : nullptr;
        deps_.dcp_communicator->allreduce_hidden(
            attn_bufs_offset_.data(), num_tokens, streams);
    }
    return true;
}

// ── Compute command dispatch (KD-2: validates + launches kernels) ─────────

void CommandDispatcher::handle_compute_command(const ipc::Command& cmd) {
    if (!deps_.buffer_registry) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                    "buffer registry not configured");
        return;
    }

    auto* reg = deps_.buffer_registry;
    const auto type = static_cast<ipc::CmdType>(cmd.cmd_type);
    const auto gpu = static_cast<int>(cmd.gpu_idx);
    const auto sid = to_stream(cmd.stream_id);
    // TODO:DEBT TD-37: stream_manager dereferenced without null check
    void* stream = deps_.stream_manager->stream(gpu, sid);
    uint32_t layer_idx = 0;
    bool confidence_launched = false;
    // CMD_OUTPUT_HEAD readback (TD-14i, #91): when readback_to_host is set,
    // the sampled (argmax) token is D2H-copied into the sideband spec
    // readback scratch and the completion carries its offset/size so the
    // Python orchestrator reads it without touching CUDA.
    uint32_t readback_off = 0;
    uint32_t readback_bytes = 0;

    // Per-command: validate buf_ids, then launch kernel on the target stream.
    switch (type) {

        // ── Attention commands → AttentionDevice ───────────────────────────

        // TD-ATTN-LEGACY (resolved): the granular attention commands route
        // through the SAME production path as D_B_CMD_RUN_ATTENTION —
        // dispatch_attention_internal → DcpExecutor (W_UK absorption, RoPE,
        // model softmax scale, KV metadata from the batch-descriptor
        // sideband). The old handler called dev->prefill_attention directly
        // with the raw hidden-state buffer as q (no absorption, no RoPE) —
        // never valid attention. The payload's buf_ids are ignored: hidden
        // state and KV cache come from the engine-wired buffers, like the
        // fused command. Requires a wired DcpExecutor (CMP_ERROR otherwise).
        case ipc::CMD_ATTENTION_DECODE:
        case ipc::CMD_ATTENTION_PREFILL: {
            auto& p = cmd.attention;
            layer_idx = p.layer_idx;
            if (p.batch_size == 0
                || p.batch_size > ipc::kMaxBatchDescriptors) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "attention: batch_size out of range");
                return;
            }
            InternalAttentionParams ap{};
            ap.layer_idx  = p.layer_idx;
            ap.num_seqs   = p.batch_size;
            ap.gpu_idx    = cmd.gpu_idx;
            ap.is_prefill = (type == ipc::CMD_ATTENTION_PREFILL) ? 1u : 0u;
            ap.use_graph  = p.use_graph;
            if (!dispatch_attention_internal(ap)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx,
                            last_internal_error_msg_
                                ? last_internal_error_cat_
                                : ipc::CmpErrorCategory::kComputeValidation,
                            last_internal_error_msg_
                                ? last_internal_error_msg_
                                : "attention: internal dispatch failed "
                                  "(DcpExecutor required)");
                return;
            }
            break;
        }

        case ipc::CMD_RMSNORM: {
            auto& p = cmd.rmsnorm;
            if (!reg->resolve(p.input_buf_id) ||
                !reg->resolve(p.output_buf_id) ||
                !reg->resolve(p.weight_buf_id)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "rmsnorm: invalid buf_id");
                return;
            }
            auto* dev = attn_dev(cmd.gpu_idx);
            if (!dev) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "rmsnorm: no attention device for gpu_idx");
                return;
            }
            dev->set_device();
            dev->rmsnorm(reg->resolve(p.output_buf_id),
                         reg->resolve(p.input_buf_id),
                         reg->resolve(p.weight_buf_id),
                         p.eps, static_cast<int>(p.num_tokens),
                         static_cast<int>(p.hidden_size),
                         /*row_stride=*/static_cast<int>(p.hidden_size),
                         stream);
            break;
        }

        case ipc::CMD_DYNAMIC_FP8_QUANT: {
            auto& p = cmd.dynamic_fp8_quant;
            if (!reg->resolve(p.input_buf_id) ||
                !reg->resolve(p.output_buf_id) ||
                !reg->resolve(p.scales_buf_id)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "dynamic_fp8_quant: invalid buf_id");
                return;
            }
            auto* dev = attn_dev(cmd.gpu_idx);
            if (!dev) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "dynamic_fp8_quant: no attention device for gpu_idx");
                return;
            }
            dev->set_device();
            compute::DynamicFp8QuantParams qp{};
            qp.num_tokens  = static_cast<int>(p.num_tokens);
            qp.hidden_size = static_cast<int>(p.hidden_dim);
            qp.input       = reg->resolve(p.input_buf_id);
            qp.output      = reg->resolve(p.output_buf_id);
            qp.scales      = reg->resolve(p.scales_buf_id);
            dev->quantize_fp8(qp, stream);
            break;
        }

        case ipc::CMD_DCP_CORRECTION: {
            auto& p = cmd.dcp_correction;
            if (!reg->resolve(p.input_buf_id) ||
                !reg->resolve(p.output_buf_id) ||
                !reg->resolve(p.lse_buf_id)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "dcp_correction: invalid buf_id");
                return;
            }
            auto* dev = attn_dev(cmd.gpu_idx);
            if (!dev) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "dcp_correction: no attention device for gpu_idx");
                return;
            }
            dev->set_device();
            {
                compute::GraphKey gkey{compute::GraphType::kDcpAllreduce,
                                       gpu, static_cast<int>(p.batch_size)};
                auto* entry = deps_.graph_registry->find(gkey);
                if (entry) {
                    dev->dcp_graph_replay(*entry, stream);
                }
            }
            break;
        }

        case ipc::CMD_NCCL_ALLREDUCE: {
            auto& p = cmd.nccl_allreduce;
            if (!reg->resolve(p.buf_id)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "nccl_allreduce: invalid buf_id");
                return;
            }
            // NCCL allreduce is a multi-rank collective — all TP ranks must
            // issue their call inside ncclGroupStart/End. Wired in KD-3 via
            // fused attention (D_B_CMD_RUN_ATTENTION). TP=1: no-op.
            break;
        }

        // ── MoE commands → ExpertDevice ────────────────────────────────────

        case ipc::CMD_GATING: {
            auto& p = cmd.gating;
            layer_idx = p.layer_idx;
            if (!reg->resolve(p.input_buf_id) ||
                !reg->resolve(p.output_weights_buf_id) ||
                !reg->resolve(p.output_indices_buf_id)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "gating: invalid buf_id");
                return;
            }
            // V4-4: hash layers route by TOKEN ID (tid2eid) — the standalone
            // CMD_GATING payload carries no token ids. Fail-closed: use the
            // fused emit_gating / RUN_MOE paths for hash layers.
            if (static_cast<int>(p.layer_idx) < deps_.moe_hash_layers) {
                write_error(cmd.cmd_seq, cmd.gpu_idx,
                            ipc::CmpErrorCategory::kComputeValidation,
                            "gating: V4 hash layer requires token ids "
                            "(unavailable in CMD_GATING)");
                return;
            }
            if (deps_.cuda_kernels_enabled && deps_.live_config) {
                // TD-74o: guard against null expert device.
                auto* dev = expert_dev(cmd.gpu_idx);
                if (!dev) {
                    write_error(cmd.cmd_seq, cmd.gpu_idx,
                                ipc::CmpErrorCategory::kComputeValidation,
                                "gating: no expert device for gpu_idx");
                    return;
                }
                dev->set_device();
                const auto& mc = deps_.live_config->model;
                compute::TopkGatingParams gp{};
                gp.num_tokens            = static_cast<int>(p.num_tokens);
                gp.num_experts           = mc.n_routed_experts;
                gp.topk                  = mc.num_experts_per_tok;
                gp.n_group               = mc.n_group;
                gp.topk_group            = mc.topk_group;
                gp.routed_scaling_factor = static_cast<float>(mc.routed_scaling_factor);
                gp.renormalize           = mc.norm_topk_prob;
                // V4-4a: sigmoid (V3.2/GLM) vs sqrtsoftplus (V4).
                gp.scoring_func          = to_scoring_func(mc.gating_score_fn);
                const float* bias = nullptr;
                if (p.layer_idx < deps_.gating_bias_ptrs.size()
                    && cmd.gpu_idx < deps_.gating_bias_ptrs[p.layer_idx].size()) {
                    bias = deps_.gating_bias_ptrs[p.layer_idx][cmd.gpu_idx];
                }
                compute::launch_topk_gating(
                    static_cast<float*>(reg->resolve(p.output_weights_buf_id)),
                    static_cast<int32_t*>(reg->resolve(p.output_indices_buf_id)),
                    static_cast<const float*>(reg->resolve(p.input_buf_id)),
                    bias, gp, stream);
            }
            break;
        }

        case ipc::CMD_EXPERT_FFN: {
            auto& p = cmd.expert_ffn;
            layer_idx = p.layer_idx;
            if (!reg->resolve(p.permuted_input_buf_id) ||
                !reg->resolve(p.output_buf_id) ||
                !reg->resolve(p.expert_offsets_buf_id) ||
                (p.weights_buf_id && !reg->resolve(p.weights_buf_id)) ||
                (p.workspace_buf_id && !reg->resolve(p.workspace_buf_id))) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "expert_ffn: invalid buf_id");
                return;
            }
            auto* dev = expert_dev(cmd.gpu_idx);
            if (!dev) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "expert_ffn: no expert device for gpu_idx");
                return;
            }
            dev->set_device();
            if (!p.weights_buf_id || !p.workspace_buf_id) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "expert_ffn: weights_buf_id and workspace_buf_id required");
                return;
            }
            if (p.quant_mode == 0) {
                compute::Nvfp4GroupedGemmParams gp{};
                gp.num_experts    = static_cast<int>(p.num_experts);
                gp.N              = static_cast<int>(p.hidden_dim);
                gp.K              = 0;
                gp.A_base         = reg->resolve(p.permuted_input_buf_id);
                gp.B_base         = reg->resolve(p.weights_buf_id);
                gp.D_base         = reg->resolve(p.output_buf_id);
                gp.scale_A_base   = nullptr;
                gp.scale_B_base   = nullptr;
                gp.alphas         = nullptr;  // TODO(TD-69t): needs per-expert alpha array
                gp.expert_offsets = static_cast<const int32_t*>(
                    reg->resolve(p.expert_offsets_buf_id));
                gp.sf_offsets     = nullptr;
                gp.problem_sizes  = nullptr;
                gp.output_dtype   = compute::GemmOutputDtype::kBFloat16;
                auto* ws = reg->resolve(p.workspace_buf_id);
                const auto* ws_entry = reg->lookup(p.workspace_buf_id);
                dev->nvfp4_grouped_gemm(gp, ws,
                    ws_entry ? static_cast<size_t>(ws_entry->size_bytes) : 0, stream);
            } else if (p.quant_mode == 1) {
                compute::Fp8GroupedGemmParams gp{};
                gp.num_experts    = static_cast<int>(p.num_experts);
                gp.N              = static_cast<int>(p.hidden_dim);
                gp.K              = 0;
                gp.A_base         = reg->resolve(p.permuted_input_buf_id);
                gp.B_base         = reg->resolve(p.weights_buf_id);
                gp.D_base         = reg->resolve(p.output_buf_id);
                gp.scale_A_base   = nullptr;
                gp.scale_B_base   = nullptr;
                gp.expert_offsets = static_cast<const int32_t*>(
                    reg->resolve(p.expert_offsets_buf_id));
                gp.problem_sizes  = nullptr;
                gp.output_dtype   = compute::GemmOutputDtype::kBFloat16;
                auto* ws = reg->resolve(p.workspace_buf_id);
                const auto* ws_entry = reg->lookup(p.workspace_buf_id);
                dev->fp8_grouped_gemm(gp, ws,
                    ws_entry ? static_cast<size_t>(ws_entry->size_bytes) : 0, stream);
            } else if (p.quant_mode == 2) {
                // GG-5: GGUF grouped path. N=hidden_dim, K=k_dim. The weight quant
                // family is uniform across experts (GG-6: stacked ffn_*_exps has one
                // ggml_type per projection) and arrives as the model::GgufKQuantType
                // ordinal in p.gguf_type; it shares the canonical value order with
                // compute::GgufQuantType (Q2_K=0..Q8_0=5), so map by cast. Strategy
                // (int vs dequant) is a live-config knob.
                if (p.gguf_type > static_cast<uint8_t>(compute::GgufQuantType::Q8_0)) {
                    write_error(cmd.cmd_seq, cmd.gpu_idx,
                                ipc::CmpErrorCategory::kComputeValidation,
                                "expert_ffn: invalid gguf_type");
                    return;
                }
                compute::GgufGroupedGemmParams gp{};
                gp.type        = static_cast<compute::GgufQuantType>(p.gguf_type);
                gp.strategy    =
                    (deps_.live_config->quantization.gguf_strategy ==
                     config::GgufStrategy::dequant)
                        ? compute::GgufGemmStrategy::dequant
                        : compute::GgufGemmStrategy::int_strategy;
                gp.num_experts = static_cast<int>(p.num_experts);
                gp.N           = static_cast<int>(p.hidden_dim);
                gp.K           = static_cast<int>(p.k_dim);
                // GG-5d: permuted row count for the device-fused int kernel.
                gp.total_tokens = static_cast<int>(p.total_tokens);
                gp.A_base      = reg->resolve(p.permuted_input_buf_id);
                gp.D_base      = reg->resolve(p.output_buf_id);
                gp.expert_offsets = static_cast<const int32_t*>(
                    reg->resolve(p.expert_offsets_buf_id));
                // weights_buf_id resolves to the [num_experts] device array of
                // per-expert packed GGUF weight-block pointers (B_ptrs).
                gp.B_ptrs      = static_cast<const void**>(
                    reg->resolve(p.weights_buf_id));
                auto* ws = reg->resolve(p.workspace_buf_id);
                const auto* ws_entry = reg->lookup(p.workspace_buf_id);
                dev->gguf_grouped_gemm(gp, ws,
                    ws_entry ? static_cast<size_t>(ws_entry->size_bytes) : 0, stream);
            } else {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "expert_ffn: invalid quant_mode");
                return;
            }
            break;
        }

        case ipc::CMD_SWIGLU: {
            auto& p = cmd.swiglu;
            if (!reg->resolve(p.input_buf_id) ||
                !reg->resolve(p.output_buf_id)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "swiglu: invalid buf_id");
                return;
            }
            auto* dev = expert_dev(cmd.gpu_idx);
            if (!dev) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "swiglu: no expert device for gpu_idx");
                return;
            }
            dev->set_device();
            compute::FusedSwigluParams sp{};
            sp.num_tokens = static_cast<int>(p.num_tokens);
            sp.d          = static_cast<int>(p.hidden_dim);
            // V4-4b: model-wide SwiGLU clamp (llama.cpp DEEPSEEK4 semantics;
            // 0.0 = off, V3.2/GLM unchanged).
            if (deps_.live_config)
                sp.swiglu_limit =
                    static_cast<float>(deps_.live_config->model.swiglu_limit);
            dev->fused_swiglu(reg->resolve(p.output_buf_id),
                              reg->resolve(p.input_buf_id),
                              sp, /*elem_size_bytes=*/2, stream);
            break;
        }

        case ipc::CMD_MOE_PERMUTE: {
            auto& p = cmd.moe_permute;
            if (!reg->resolve(p.input_buf_id) ||
                !reg->resolve(p.output_buf_id) ||
                !reg->resolve(p.indices_buf_id) ||
                !reg->resolve(p.offsets_buf_id) ||
                (p.workspace_buf_id && !reg->resolve(p.workspace_buf_id))) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "moe_permute: invalid buf_id");
                return;
            }
            auto* dev = expert_dev(cmd.gpu_idx);
            if (!dev) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "moe_permute: no expert device for gpu_idx");
                return;
            }
            dev->set_device();
            void* ws = p.workspace_buf_id ? reg->resolve(p.workspace_buf_id) : nullptr;
            dev->moe_permute(
                reg->resolve(p.output_buf_id),
                static_cast<int32_t*>(reg->resolve(p.offsets_buf_id)),
                ws ? static_cast<int32_t*>(ws) : nullptr,
                ws ? static_cast<int32_t*>(ws) + p.num_tokens * p.topk : nullptr,
                reg->resolve(p.input_buf_id),
                static_cast<const int32_t*>(reg->resolve(p.indices_buf_id)),
                static_cast<int>(p.num_tokens), static_cast<int>(p.topk),
                static_cast<int>(p.hidden_dim), static_cast<int>(p.num_experts),
                /*elem_size_bytes=*/2, ws, stream);
            break;
        }

        case ipc::CMD_MOE_UNPERMUTE: {
            auto& p = cmd.moe_unpermute;
            if (!reg->resolve(p.input_buf_id) ||
                !reg->resolve(p.output_buf_id) ||
                !reg->resolve(p.indices_buf_id) ||
                (p.weights_buf_id && !reg->resolve(p.weights_buf_id))) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "moe_unpermute: invalid buf_id");
                return;
            }
            auto* dev = expert_dev(cmd.gpu_idx);
            if (!dev) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "moe_unpermute: no expert device for gpu_idx");
                return;
            }
            dev->set_device();
            dev->moe_unpermute(
                reg->resolve(p.output_buf_id),
                reg->resolve(p.input_buf_id),
                p.weights_buf_id
                    ? static_cast<const float*>(reg->resolve(p.weights_buf_id))
                    : nullptr,
                static_cast<const int32_t*>(reg->resolve(p.indices_buf_id)),
                static_cast<int>(p.num_tokens), static_cast<int>(p.topk),
                static_cast<int>(p.hidden_dim), /*elem_size_bytes=*/2, stream);
            break;
        }

        // ── Boundary commands → free-function kernels ──────────────────────

        case ipc::CMD_EMBEDDING_LOOKUP: {
            auto& p = cmd.embedding_lookup;
            if (p.num_tokens == 0 || p.num_tokens > ipc::kMaxSidebandTokenIds) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "embedding_lookup: num_tokens out of range");
                return;
            }
            // TD-74k: output buffer is sized for max_batch_size tokens —
            // TD-PREFILL-SUPERCHUNK: or for the superchunk rows when one is
            // configured (row_offset targets a sub-chunk's rows;
            // superchunk_rows_ = max(max_batch, min(superchunk, capacity))).
            const int hidden_rows = superchunk_rows_;
            if (static_cast<int>(p.row_offset + p.num_tokens) > hidden_rows
                || static_cast<int>(p.num_tokens) > hidden_rows) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "embedding_lookup: row_offset + num_tokens exceeds "
                            "max_batch_size (superchunk hidden rows)");
                return;
            }
            if (!deps_.sideband_base) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "embedding_lookup: sideband not configured");
                return;
            }
            if (!reg->resolve(p.output_buf_id)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "embedding_lookup: invalid output_buf_id");
                return;
            }
            if (deps_.cuda_kernels_enabled
                && cmd.gpu_idx < deps_.embedding_table_ptrs.size()
                && deps_.embedding_table_ptrs[cmd.gpu_idx]
                && deps_.live_config) {
                // TD-74n: ensure correct device context before kernel launch.
                deps_.device_backends[cmd.gpu_idx]->set_device();
                const auto& mc = deps_.live_config->model;

                // TD-GOLDEN-EMB-OOB: with a vocab-sharded table (TP), a
                // full-vocab lookup on one rank reads past the shard for
                // token ids >= vocab/tp. Run the masked per-rank lookup +
                // allreduce instead — never fall through to the single-table
                // path with a sharded table.
                if (embedding_tp_degree() > 1) {
                    if (!dispatch_embedding_lookup_sharded(
                            static_cast<int>(p.num_tokens), p.row_offset)) {
                        write_error(cmd.cmd_seq, cmd.gpu_idx,
                                    ipc::CmpErrorCategory::kComputeValidation,
                                    "embedding_lookup: TP-sharded lookup "
                                    "failed (see log)");
                        return;
                    }
                    // The sharded path fills every rank's attn_buf; if the
                    // requested output buffer is a different buffer, mirror
                    // rank 0's result into it (same GPU, in-stream).
                    void* out = reg->resolve(p.output_buf_id);
                    if (out && !attn_bufs_.empty() && out != attn_bufs_[0]) {
                        const auto& pair0 = deps_.hidden_state_pairs[0];
                        void* s0 = deps_.stream_manager->stream(
                            pair0.gpu_position, compute::StreamId::kAttention);
                        const size_t mr_off = static_cast<size_t>(p.row_offset)
                                            * mc.hidden_size * 2;
                        deps_.device_backends[pair0.gpu_position]
                            ->memcpy_d2d_async(
                                static_cast<uint8_t*>(out) + mr_off,
                                static_cast<const uint8_t*>(attn_bufs_[0])
                                    + mr_off,
                                static_cast<size_t>(p.num_tokens)
                                    * mc.hidden_size * 2, s0);
                    }
                    break;
                }

                // KD-4b: H2D copy token IDs from sideband (host) to device scratch.
                if (cmd.gpu_idx >= embedding_token_scratch_.size()
                    || !embedding_token_scratch_[cmd.gpu_idx]) {
                    write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                                "embedding_lookup: no token scratch for gpu_idx");
                    return;
                }
                // TD-PREFILL-NONDET: order the embedding write after the
                // previous step's MoE commit on THIS rank (see the sharded
                // path above for the full race account).
                {
                    const int pi = resolve_pair_idx(cmd.gpu_idx);
                    if (pi >= 0 && deps_.hidden_state_pairs[pi].moe_attn_event
                        && deps_.stream_manager)
                        deps_.device_backends[cmd.gpu_idx]->stream_wait_event(
                            stream, deps_.hidden_state_pairs[pi].moe_attn_event);
                }
                const auto* host_token_ids = reinterpret_cast<const int32_t*>(
                    deps_.sideband_base + ipc::IpcLayout::kTokenIdsOff);
                auto* device_token_ids = static_cast<int32_t*>(
                    embedding_token_scratch_[cmd.gpu_idx]);
                deps_.device_backends[cmd.gpu_idx]->memcpy_h2d_async(
                    device_token_ids, host_token_ids,
                    p.num_tokens * sizeof(int32_t), stream);
                // V4-4 hash gating: persist token ids at row_offset (see the
                // sharded path above for the superchunk rationale).
                if (deps_.moe_hash_layers > 0
                    && cmd.gpu_idx < moe_scratch_.size()
                    && moe_scratch_[cmd.gpu_idx].moe_token_ids) {
                    deps_.device_backends[cmd.gpu_idx]->memcpy_h2d_async(
                        static_cast<int32_t*>(
                            moe_scratch_[cmd.gpu_idx].moe_token_ids)
                            + p.row_offset,
                        host_token_ids,
                        p.num_tokens * sizeof(int32_t), stream);
                }
                // V4-5b mHC: residual rows are hc-wide; lookup into the
                // collapsed staging then repeat-expand (see sharded path).
                const int hc = deps_.hc_streams;
                const size_t st_row_off = static_cast<size_t>(p.row_offset)
                                        * mc.hidden_size * hc * 2;
                uint8_t* embed_dst =
                    static_cast<uint8_t*>(reg->resolve(p.output_buf_id))
                    + st_row_off;
                void* lookup_dst = embed_dst;
                if (hc > 1) {
                    if (cmd.gpu_idx >= moe_scratch_.size()
                        || !moe_scratch_[cmd.gpu_idx].hc_x) {
                        spdlog::error("embedding: mHC active but hc_x staging "
                                      "missing on gpu {}", cmd.gpu_idx);
                        break;
                    }
                    lookup_dst = moe_scratch_[cmd.gpu_idx].hc_x;
                }
                compute::launch_embedding_lookup(
                    lookup_dst,
                    deps_.embedding_table_ptrs[cmd.gpu_idx],
                    device_token_ids, static_cast<int>(p.num_tokens),
                    mc.vocab_size, mc.hidden_size,
                    compute::EmbeddingDtype::kBFloat16, stream);
                if (hc > 1) {
                    compute::launch_hc_expand_repeat(
                        embed_dst, lookup_dst,
                        static_cast<int>(p.num_tokens), hc, mc.hidden_size,
                        stream);
                }

                // TD-73i: Broadcast embedding output to all other TP ranks.
                // The kernel wrote to the primary rank's attn_buf.  For TP>1,
                // copy to each other rank's attn_buf on their kAttention stream
                // so DcpExecutor::execute_attention reads valid hidden state.
                if (deps_.dcp_executor
                    && deps_.dcp_executor->dcp_size() > 1
                    && deps_.hidden_state_pairs.size() > 1) {
                    const int src_pair = resolve_pair_idx(cmd.gpu_idx);
                    if (src_pair >= 0) {
                        const void* src_buf =
                            static_cast<const uint8_t*>(
                                reg->resolve(p.output_buf_id)) + st_row_off;
                        const size_t embed_bytes =
                            static_cast<size_t>(p.num_tokens)
                            * mc.hidden_size * deps_.hc_streams * 2;  // BF16

                        // Record event on source stream after kernel.
                        void* embed_evt =
                            deps_.device_backends[cmd.gpu_idx]->create_event();
                        deps_.device_backends[cmd.gpu_idx]->record_event(
                            embed_evt, stream);

                        for (size_t r = 0;
                             r < deps_.hidden_state_pairs.size(); ++r) {
                            if (static_cast<int>(r) == src_pair) continue;
                            const auto& dp = deps_.hidden_state_pairs[r];
                            if (!dp.attn_buf) continue;
                            const int dst_pos = dp.gpu_position;
                            void* dst_stream =
                                deps_.stream_manager->stream(
                                    dst_pos,
                                    compute::StreamId::kAttention);
                            deps_.device_backends[dst_pos]->set_device();
                            // TD-PREFILL-NONDET: dst rank's prior MoE commit
                            // must land before this broadcast overwrites its
                            // attn_buf.
                            if (dp.moe_attn_event)
                                deps_.device_backends[dst_pos]
                                    ->stream_wait_event(dst_stream,
                                                        dp.moe_attn_event);
                            deps_.device_backends[dst_pos]->stream_wait_event(
                                dst_stream, embed_evt);
                            deps_.device_backends[dst_pos]->memcpy_async(
                                static_cast<uint8_t*>(dp.attn_buf)
                                    + st_row_off,
                                src_buf,
                                embed_bytes, dst_stream);
                        }

                        // Restore device context for completion event.
                        deps_.device_backends[cmd.gpu_idx]->set_device();
                        deps_.device_backends[cmd.gpu_idx]->destroy_event(
                            embed_evt);
                    }
                }
            }
            break;
        }

        // TODO:DEBT TD-14l: num_logprobs field ignored — top-K logprobs not returned
        // TODO:DEBT TD-14fa: no num_tokens range check — confidence scratch overflow if > kMaxBatchDescriptors
        // TODO:DEBT TD-14fc: no num_tokens range check — norm scratch overflow if > max_batch_size
        case ipc::CMD_OUTPUT_HEAD: {
            auto& p = cmd.output_head;
            if (!reg->resolve(p.input_buf_id) ||
                !reg->resolve(p.output_buf_id)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "output_head: invalid buf_id");
                return;
            }
            // TD-14i RESOLVED (#91): readback_to_host samples the last
            // token's ARGMAX (temperature=0 — the greedy production decode;
            // non-greedy sampling still goes through CMD_SAMPLE_TOKENS) and
            // D2H-copies the token id (u32) into the sideband spec readback
            // scratch — the same slice the fused MTP/self-spec steps use;
            // per-request command chains are serialized by the orchestrator
            // so the slice is never concurrently owned.  top1_prob still
            // arrives via the completion field (host staging).
            uint8_t* readback_dst = nullptr;
            if (p.readback_to_host && deps_.sideband_base) {
                readback_off = static_cast<uint32_t>(
                    ipc::IpcLayout::kSpecCheckpointOff + 2560);
                // Batched-verify heads (num_tokens > 1) read back ALL
                // sampled argmax tokens (u32 each) — one per hidden row —
                // so the orchestrator can apply the greedy acceptance rule
                // over a K-token verify pass in ONE completion.  Capped to
                // the spec readback slice (kMaxOutputHeadReadbackTokens);
                // num_tokens == 1 is the historical 4-byte readback.
                const uint32_t n = std::min(
                    std::max(p.num_tokens, 1u),
                    ipc::kMaxOutputHeadReadbackTokens);
                readback_bytes = n * static_cast<uint32_t>(sizeof(uint32_t));
                readback_dst = deps_.sideband_base + readback_off;
            }
            // KD-R5: delegate to unified output head helper.
            confidence_launched = dispatch_output_head({
                .gpu_idx           = cmd.gpu_idx,
                .num_tokens        = static_cast<int>(p.num_tokens),
                .input             = reg->resolve(p.input_buf_id),
                .logits_out        = static_cast<float*>(reg->resolve(p.output_buf_id)),
                .stream            = stream,
                .stream_id         = sid,
                .compute_confidence = (p.compute_confidence != 0),
                .do_sample         = (readback_dst != nullptr),
                .readback_host_dst = readback_dst,
                // #16: mtp_head selects the MTP shared head (mtp_idx + 1).
                .mtp_head_idx      = p.mtp_head
                                         ? static_cast<int>(p.mtp_head) - 1
                                         : -1,
                // TD-SERVE-NAMED-TOOL-CHOICE: guided decoding — full logits
                // of the last row D2H into the pinned readback row.
                .logits_host_dst   = (p.readback_logits && logits_readback_host_)
                                         ? static_cast<uint8_t*>(
                                               logits_readback_host_)
                                         : nullptr,
            });
            break;
        }

        case ipc::CMD_SAMPLE_TOKENS: {
            auto& p = cmd.sample_tokens;
            if (p.num_tokens == 0 || p.num_tokens > ipc::kMaxSidebandTokenIds) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "sample_tokens: num_tokens out of range");
                return;
            }
            if (p.vocab_size == 0) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "sample_tokens: vocab_size must be > 0");
                return;
            }
            if (!deps_.sideband_base) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "sample_tokens: sideband not configured");
                return;
            }
            if (!reg->resolve(p.logits_buf_id)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "sample_tokens: invalid logits_buf_id");
                return;
            }
            if (p.top_p < 0.0f || p.top_p > 1.0f) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "sample_tokens: top_p must be in [0, 1]");
                return;
            }
            if (deps_.cuda_kernels_enabled
                && cmd.gpu_idx < sampling_scratch_.size()
                && sampling_scratch_[cmd.gpu_idx]) {
                // TD-74n: ensure correct device context before kernel launch.
                deps_.device_backends[cmd.gpu_idx]->set_device();
                auto* device_ids = static_cast<int32_t*>(sampling_scratch_[cmd.gpu_idx]);
                compute::launch_sample_tokens(
                    device_ids,
                    static_cast<float*>(reg->resolve(p.logits_buf_id)),
                    static_cast<int>(p.num_tokens),
                    static_cast<int>(p.vocab_size),
                    p.temperature, p.top_p, static_cast<int>(p.top_k),
                    p.random_seed, stream);
                void* host_dst = deps_.sideband_base + ipc::IpcLayout::kTokenIdsOff;
                size_t copy_bytes = p.num_tokens * sizeof(int32_t);
                deps_.stream_manager->memcpy_d2h_async(
                    host_dst, device_ids, copy_bytes, gpu, sid);
            }
            break;
        }

        // ── Deferred commands (validate only, kernel in #63b) ──────────────

        case ipc::CMD_PRESCOPE_GATING: {
            auto& p = cmd.prescope;
            layer_idx = p.target_layer_idx;
            if (!reg->resolve(p.hidden_state_buf_id) ||
                !reg->resolve(p.output_buf_id)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "prescope_gating: invalid buf_id");
                return;
            }
            break;
        }

        case ipc::CMD_PROBE_MLP: {
            auto& p = cmd.probe;
            if (!reg->resolve(p.hidden_state_buf_id) ||
                !reg->resolve(p.output_buf_id)) {
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                            "probe_mlp: invalid buf_id");
                return;
            }
            break;
        }

        default:
            write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kComputeValidation,
                        "unhandled compute command type");
            return;
    }

    // All validation + kernel launch succeeded.  Record event for async completion.
    void* event = create_and_record_event(gpu, sid);

    PendingCompute pc{};
    pc.cmd_seq    = cmd.cmd_seq;
    pc.gpu_idx    = cmd.gpu_idx;
    pc.cmd_type   = cmd.cmd_type;
    pc.layer_idx  = layer_idx;
    pc.cuda_event = event;

    if (type == ipc::CMD_SAMPLE_TOKENS) {
        pc.data_bytes = cmd.sample_tokens.num_tokens
                        * static_cast<uint32_t>(sizeof(uint32_t));
    }
    // TD-14i (#91): CMD_OUTPUT_HEAD readback — the completion points at the
    // sampled token id in the sideband readback scratch (event above is
    // recorded AFTER the D2H enqueue on the same stream, so the data is
    // host-visible when CMP_COMPUTE_DONE fires).
    if (readback_bytes > 0) {
        pc.host_buf_offset = readback_off;
        pc.data_bytes      = readback_bytes;
    }
    pc.has_confidence = confidence_launched;

    // TODO:DEBT TD-39: CUDA event leaked if push_back throws std::bad_alloc
    pending_compute_.push_back(pc);
}

}  // namespace layerstorm::daemon

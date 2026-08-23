// Forward pass, output head, and speculation pipelines (MTP, self-spec).
// Part of CommandDispatcher — see command_dispatcher.h.

#include "daemon/command_dispatcher.h"
#include "daemon/dispatch_detail.h"

#include <algorithm>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <spdlog/spdlog.h>

#include "compute/stream_manager.h"
#include "core/device_backend.h"
#include "core/expert_device.h"
#include "daemon/buffer_registry.h"
#include "compute/kernels/mhc/mhc.h"
#include "compute/kernels/mhc/hc_stream_mean.h"  // ticket J: V4 aux mean
#include "parallelism/dcp_communicator.h"
#include "compute/kernels/confidence/confidence.h"
#include "compute/kernels/embedding/embedding.h"
#include "compute/kernels/norm/rmsnorm.h"
#include "compute/kernels/sampling/sampling.h"
#include "compute/kernels/similarity/cosine_sim.h"
#include "compute/kernels/sm120/gemm/bf16_gemm.h"  // #16: BF16 eh_proj GEMM
#include "speculation/dspark_runtime.h"            // DSP-3: DFlash backbone

namespace layerstorm::daemon {

// ── KD-R1: Shared per-layer forward pass ────────────────────────────────────

bool CommandDispatcher::forward_one_layer(const ForwardLayerOpts& opts) {
    last_internal_error_msg_ = nullptr;  // TD-GOLDEN-KV-EXHAUST: reset per dispatch
    const uint32_t gpu = opts.gpu_idx;
    const bool tp_active = deps_.dcp_communicator
                         && deps_.dcp_communicator->is_active()
                         && deps_.dcp_executor;

    // KD-R2: resolve pair for sync events.
    const int pair_idx = resolve_pair_idx(gpu);

    // Step 0: Wait for prior MoE copy-back (kExpertFfn) before this layer's
    // attention reads attn_buf (kAttention).  No-op on first layer
    // (event unrecorded) or when not in a multi-layer loop.
    // KD-4g: with TP>1, wait on ALL TP ranks (attention reads all attn_bufs).
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
    } else if (pair_idx >= 0) {
        const auto& pair = deps_.hidden_state_pairs[pair_idx];
        if (pair.moe_attn_event) {
            deps_.stream_manager->wait_event(
                static_cast<int>(gpu),
                compute::StreamId::kAttention,
                pair.moe_attn_event);
        }
    }

    // Step 1: Attention — reads attn_buf, produces attn_out,
    // copies to moe_buf + residual, records attn_moe sync.
    InternalAttentionParams ap{};
    ap.layer_idx  = opts.layer_idx;
    ap.num_seqs   = opts.num_seqs;
    ap.gpu_idx    = opts.gpu_idx;
    ap.is_prefill = opts.is_prefill ? 1u : 0u;
    ap.use_graph  = opts.use_graph  ? 1u : 0u;
    ap.is_draft   = opts.is_draft   ? 1u : 0u;
    if (!dispatch_attention_internal(ap))
        return false;

    // Step 2: MoE — dense layers handled by early-out in dispatch_moe_internal.
    // Waits attn_moe sync, full pipeline, commits moe_buf → attn_buf.
    // KD-4g: with TP>1, dispatch all ranks with allreduce coordination.
    InternalMoeParams mp{};
    mp.layer_idx     = opts.layer_idx;
    mp.num_seqs      = opts.num_seqs;
    mp.gpu_idx       = opts.gpu_idx;
    mp.topk_override = opts.topk_override;
    mp.store_gating  = opts.store_gating;
    const bool moe_ok = tp_active
        ? dispatch_moe_all_ranks(mp)
        : dispatch_moe_internal(mp);
    if (!moe_ok)
        return false;

    // moe_attn_event is recorded inside dispatch_moe_internal() after commit
    // (TD-73l). No redundant recording needed here.

    return true;
}

// ── KD-R5: Unified output head dispatch ─────────────────────────────────────

namespace {
// #16: resolve the pre-head norm + head weight for one GPU, honoring the MTP
// shared-head override.  MTP shared_head.norm replaces final_norm; MTP
// shared_head.head replaces the lm_head when present (GGUF checkpoints dedup
// it into the main head → the fallback is exact by construction).
struct HeadWeights {
    const void* norm = nullptr;
    const void* head = nullptr;
};
HeadWeights resolve_head_weights(const CommandDispatcher::Deps& deps,
                                 size_t gpu, int mtp_head_idx) {
    HeadWeights w;
    if (gpu < deps.final_norm_ptrs.size())
        w.norm = deps.final_norm_ptrs[gpu];
    if (gpu < deps.output_head_weight_ptrs.size())
        w.head = deps.output_head_weight_ptrs[gpu];
    if (mtp_head_idx >= 0) {
        const auto mi = static_cast<size_t>(mtp_head_idx);
        if (mi < deps.mtp_shared_head_norm_ptrs.size()
            && gpu < deps.mtp_shared_head_norm_ptrs[mi].size()
            && deps.mtp_shared_head_norm_ptrs[mi][gpu])
            w.norm = deps.mtp_shared_head_norm_ptrs[mi][gpu];
        if (mi < deps.mtp_shared_head_weight_ptrs.size()
            && gpu < deps.mtp_shared_head_weight_ptrs[mi].size()
            && deps.mtp_shared_head_weight_ptrs[mi][gpu])
            w.head = deps.mtp_shared_head_weight_ptrs[mi][gpu];
    }
    return w;
}
}  // namespace

bool CommandDispatcher::dispatch_output_head(const OutputHeadOpts& opts) {
    const uint32_t gpu = opts.gpu_idx;
    if (!deps_.cuda_kernels_enabled
        || gpu >= deps_.output_head_weight_ptrs.size()
        || !deps_.output_head_weight_ptrs[gpu]
        || !deps_.live_config)
        return false;

    // KD-4g: TP>1 — dispatch all ranks, allgather partial logits.
    if (deps_.dcp_communicator && deps_.dcp_communicator->is_active()
        && deps_.dcp_executor)
        return dispatch_output_head_tp(opts);

    // TD-74t: ensure correct device context before kernel launches.
    deps_.device_backends[gpu]->set_device();

    const auto& mc = deps_.live_config->model;
    const void* head_input = opts.input;

    // Ticket J: V4 dflash final-residual aux tap (before the collapse
    // consumes the hc-wide rows; no-op unless the runtime wants it).
    if (deps_.dspark)
        maybe_dspark_capture_final(gpu, head_input, opts.num_tokens,
                                   opts.mtp_head_idx, opts.stream);

    // V4-5b mHC: the input buffer holds the hc-stream residual — collapse it
    // with the output_hc weights (build_hc_head) before norm + head GEMM.
    if (deps_.hc_streams > 1) {
        void* hscr = (gpu < moe_scratch_.size()) ? moe_scratch_[gpu].hc_x
                                                 : nullptr;
        const void* fn = (gpu < deps_.output_hc_fn_ptrs.size())
            ? deps_.output_hc_fn_ptrs[gpu] : nullptr;
        const void* base = (gpu < deps_.output_hc_base_ptrs.size())
            ? deps_.output_hc_base_ptrs[gpu] : nullptr;
        const void* scale = (gpu < deps_.output_hc_scale_ptrs.size())
            ? deps_.output_hc_scale_ptrs[gpu] : nullptr;
        if (!fn || !base || !scale || !hscr) {
            spdlog::error("dispatch_output_head: mHC active but output_hc "
                          "weights/scratch missing on gpu {}", gpu);
            return false;
        }
        compute::launch_mhc_head(
            hscr, head_input, fn, scale, base,
            static_cast<float>(mc.rms_norm_eps),
            static_cast<float>(mc.hc_eps),
            opts.num_tokens, deps_.hc_streams, mc.hidden_size, opts.stream);
        head_input = hscr;
    }

    // #16: norm/head weight selection (MTP shared-head override when set).
    const HeadWeights hw = resolve_head_weights(deps_, gpu, opts.mtp_head_idx);

    // Step 1: Final RMSNorm (fixes TD-50x — speculation paths were missing this).
    if (opts.apply_final_norm
        && hw.norm
        && gpu < output_norm_scratch_.size()
        && output_norm_scratch_[gpu]) {
        compute::launch_rmsnorm(
            output_norm_scratch_[gpu], head_input,
            hw.norm,
            static_cast<float>(mc.rms_norm_eps),
            opts.num_tokens, mc.hidden_size,
            compute::NormDtype::kBFloat16, opts.stream);
        head_input = output_norm_scratch_[gpu];
    }

    // Step 2: Output GEMM.
    const float* bias = (gpu < deps_.output_head_bias_ptrs.size())
        ? deps_.output_head_bias_ptrs[gpu] : nullptr;
    compute::launch_output_head(
        opts.logits_out, head_input,
        hw.head, bias,
        opts.num_tokens, mc.vocab_size, mc.hidden_size,
        compute::EmbeddingDtype::kBFloat16, opts.stream);

    // TD-SERVE-NAMED-TOOL-CHOICE / TD-ORCH-SAMPLED-SPEC: full-logits
    // readback — D2H the FIRST min(num_tokens, kMaxLogitsReadbackRows)
    // rows contiguously on the same stream, so they are host-visible when
    // the completion event fires.  num_tokens == 1 is the historical
    // guided-decoding single-row copy byte-for-byte; the speculative
    // sampled/logprobs verify chunk consumes all R rows.
    if (opts.logits_host_dst && opts.num_tokens > 0) {
        const size_t n_rows = std::min<size_t>(
            static_cast<size_t>(opts.num_tokens),
            ipc::kMaxLogitsReadbackRows);
        deps_.stream_manager->memcpy_d2h_async(
            opts.logits_host_dst,
            opts.logits_out,
            n_rows * static_cast<size_t>(mc.vocab_size) * sizeof(float),
            static_cast<int>(gpu), opts.stream_id);
    }

    // Step 3: Confidence (fixes TD-50sa — always use device scratch, never nullptr;
    //                      fixes TD-50s — D2H from device scratch, not host ptr).
    bool confidence_launched = false;
    // TD-56b RESOLVED: guard both top1 and entropy scratch symmetrically.
    if (opts.compute_confidence && opts.num_tokens > 0
        && gpu < confidence_top1_scratch_.size()
        && confidence_top1_scratch_[gpu]
        && gpu < confidence_entropy_scratch_.size()
        && confidence_entropy_scratch_[gpu]) {
        auto* top1_dev = static_cast<float*>(confidence_top1_scratch_[gpu]);
        auto* entr_dev = static_cast<float*>(confidence_entropy_scratch_[gpu]);
        compute::launch_compute_confidence(
            opts.logits_out, top1_dev, entr_dev,
            opts.num_tokens, mc.vocab_size, opts.stream);

        const int last = opts.num_tokens - 1;
        if (opts.readback_host_dst) {
            // Speculation/readback path: D2H top1_prob AFTER the token
            // block — readback[4n..4n+4) where n = min(num_tokens, cap).
            // n == 1 is the historical [4..7] layout byte-for-byte.
            const int n = std::min(
                opts.num_tokens,
                static_cast<int>(ipc::kMaxOutputHeadReadbackTokens));
            deps_.stream_manager->memcpy_d2h_async(
                opts.readback_host_dst + 4 * static_cast<size_t>(n),
                top1_dev + last,
                sizeof(float), static_cast<int>(gpu), opts.stream_id);
        }
        // Host staging is ALWAYS filled (TD-14i/#91): the CMD_OUTPUT_HEAD
        // ring path with readback_to_host needs the token via the sideband
        // readback AND top1_prob via the completion field (staging).  Spec
        // pipelines never set has_confidence, so the extra write is inert.
        if (gpu < confidence_host_staging_.size()) {
            auto& staging = confidence_host_staging_[gpu];
            deps_.stream_manager->memcpy_d2h_async(
                &staging.top1_prob, top1_dev + last,
                sizeof(float), static_cast<int>(gpu), opts.stream_id);
            deps_.stream_manager->memcpy_d2h_async(
                &staging.entropy, entr_dev + last,
                sizeof(float), static_cast<int>(gpu), opts.stream_id);
        }
        confidence_launched = true;
    }

    // Step 4: Sampling (speculation pipelines only).  Batched-verify heads
    // (num_tokens > 1) read back ALL sampled tokens — one per hidden row.
    if (opts.do_sample && opts.num_tokens > 0
        && gpu < spec_scratch_.size() && spec_scratch_[gpu].readback) {
        auto* sample_dev = static_cast<int32_t*>(spec_scratch_[gpu].readback);
        compute::launch_sample_tokens(
            sample_dev, opts.logits_out,
            opts.num_tokens, mc.vocab_size,
            opts.temperature, opts.top_p, opts.top_k, opts.seed, opts.stream);
        if (opts.readback_host_dst) {
            const int n = std::min(
                opts.num_tokens,
                static_cast<int>(ipc::kMaxOutputHeadReadbackTokens));
            deps_.stream_manager->memcpy_d2h_async(
                opts.readback_host_dst, sample_dev,
                static_cast<size_t>(n) * sizeof(int32_t),
                static_cast<int>(gpu), opts.stream_id);
        }
    }

    return confidence_launched;
}

// ── KD-4g: TP>1 output head — allgather partial logits ──────────────────────

bool CommandDispatcher::dispatch_output_head_tp(const OutputHeadOpts& opts) {
    const auto& mc = deps_.live_config->model;
    const auto& tp_gpus = deps_.dcp_executor->gpus();
    const int dcp_size = deps_.dcp_executor->dcp_size();
    const int local_vocab = mc.vocab_size / dcp_size;
    const int num_tokens = opts.num_tokens;
    const uint32_t primary_gpu = opts.gpu_idx;

    // Step 1: RMSNorm + partial output GEMM on each TP rank.
    // Rank 0 uses opts.stream; other ranks use kExpertFfn.
    static constexpr int kMaxTp = 8;
    if (dcp_size > kMaxTp) {
        spdlog::critical("dispatch_output_head_tp: dcp_size {} exceeds kMaxTp {}", dcp_size, kMaxTp);
        std::abort();
    }
    const float* send_bufs[kMaxTp] = {};
    float* recv_bufs[kMaxTp] = {};
    void* streams[kMaxTp] = {};

    for (int r = 0; r < dcp_size; ++r) {
        const int gpu_pos = tp_gpus[r].position;

        // Resolve hidden state input for this rank.
        const void* head_input = nullptr;
        if (static_cast<uint32_t>(gpu_pos) == primary_gpu && opts.input) {
            head_input = opts.input;
        } else {
            const int pi = resolve_pair_idx(static_cast<uint32_t>(gpu_pos));
            if (pi >= 0)
                head_input = deps_.hidden_state_pairs[pi].attn_buf;
        }
        if (!head_input) continue;

        deps_.device_backends[gpu_pos]->set_device();
        void* stream_r = (static_cast<uint32_t>(gpu_pos) == primary_gpu)
            ? opts.stream
            : deps_.stream_manager->stream(gpu_pos, compute::StreamId::kExpertFfn);
        streams[r] = stream_r;

        // #16: norm/head selection (MTP shared-head override when set).  The
        // MTP shared_head.weight is vocab-sharded exactly like the main head.
        const HeadWeights hw = resolve_head_weights(
            deps_, static_cast<size_t>(gpu_pos), opts.mtp_head_idx);

        // V4-5b mHC: collapse the hc-stream residual first (per rank).
        if (deps_.hc_streams > 1) {
            const size_t gp = static_cast<size_t>(gpu_pos);
            void* hscr = (gp < moe_scratch_.size()) ? moe_scratch_[gp].hc_x
                                                    : nullptr;
            const void* fn = (gp < deps_.output_hc_fn_ptrs.size())
                ? deps_.output_hc_fn_ptrs[gp] : nullptr;
            const void* base = (gp < deps_.output_hc_base_ptrs.size())
                ? deps_.output_hc_base_ptrs[gp] : nullptr;
            const void* scale = (gp < deps_.output_hc_scale_ptrs.size())
                ? deps_.output_hc_scale_ptrs[gp] : nullptr;
            if (!fn || !base || !scale || !hscr) {
                spdlog::error("dispatch_output_head_tp: mHC active but "
                              "output_hc weights/scratch missing on gpu {}",
                              gpu_pos);
                return false;
            }
            compute::launch_mhc_head(
                hscr, head_input, fn, scale, base,
                static_cast<float>(mc.rms_norm_eps),
                static_cast<float>(mc.hc_eps),
                num_tokens, deps_.hc_streams, mc.hidden_size, stream_r);
            head_input = hscr;
        }

        // Final RMSNorm
        if (opts.apply_final_norm
            && hw.norm
            && static_cast<size_t>(gpu_pos) < output_norm_scratch_.size()
            && output_norm_scratch_[gpu_pos]) {
            compute::launch_rmsnorm(
                output_norm_scratch_[gpu_pos], head_input,
                hw.norm,
                static_cast<float>(mc.rms_norm_eps),
                num_tokens, mc.hidden_size,
                compute::NormDtype::kBFloat16, stream_r);
            head_input = output_norm_scratch_[gpu_pos];
        }

        // Partial output GEMM: [num_tokens, local_vocab]
        auto* partial = static_cast<float*>(
            (static_cast<size_t>(gpu_pos) < partial_logits_scratch_.size())
                ? partial_logits_scratch_[gpu_pos] : nullptr);
        if (!partial || !hw.head)
            continue;

        const float* bias = (static_cast<size_t>(gpu_pos) < deps_.output_head_bias_ptrs.size())
            ? deps_.output_head_bias_ptrs[gpu_pos] : nullptr;

        compute::launch_output_head(
            partial, head_input,
            hw.head, bias,
            num_tokens, local_vocab, mc.hidden_size,
            compute::EmbeddingDtype::kBFloat16, stream_r);

        send_bufs[r] = partial;
    }

    // Step 2: Allgather partial logits → full logits.
    // NCCL allgather concatenates rank-major: [tp, batch, local_vocab].
    // For batch=1 this equals [vocab_size] — correct, allgather directly into logits_out.
    // For batch>1 the layout is wrong — allgather into scratch, then transpose.
    const bool needs_transpose = (num_tokens > 1 && logits_gather_scratch_);

    for (int r = 0; r < dcp_size; ++r) {
        const int gpu_pos = tp_gpus[r].position;
        if (static_cast<uint32_t>(gpu_pos) == primary_gpu) {
            recv_bufs[r] = needs_transpose
                ? static_cast<float*>(logits_gather_scratch_)
                : opts.logits_out;
        } else {
            recv_bufs[r] = static_cast<float*>(
                (static_cast<size_t>(gpu_pos) < logits_scratch_.size())
                    ? logits_scratch_[gpu_pos] : nullptr);
        }
    }

    // TD-72h: verify all ranks have valid send/recv/stream before NCCL call.
    for (int r = 0; r < dcp_size; ++r) {
        if (!send_bufs[r] || !recv_bufs[r] || !streams[r]) {
            spdlog::error("dispatch_output_head_tp: rank {} has null buffer/stream", r);
            return false;
        }
    }

    deps_.dcp_communicator->allgather_logits(
        send_bufs, recv_bufs,
        num_tokens, local_vocab, streams);

    // TD-72a: transpose [tp, batch, local_vocab] → [batch, vocab_size] via strided copies.
    // Same approach as vLLM/SGLang (reshape + permute(1,0,2) + reshape).
    if (needs_transpose) {
        auto* src = static_cast<const char*>(logits_gather_scratch_);
        auto* dst = reinterpret_cast<char*>(opts.logits_out);
        const size_t row_bytes = static_cast<size_t>(local_vocab) * sizeof(float);
        const int vocab_size = dcp_size * local_vocab;
        auto* dev = deps_.device_backends[primary_gpu];
        for (int r = 0; r < dcp_size; ++r) {
            for (int t = 0; t < num_tokens; ++t) {
                const size_t src_off = (static_cast<size_t>(r) * num_tokens + t) * row_bytes;
                const size_t dst_off = (static_cast<size_t>(t) * vocab_size + r * local_vocab)
                                     * sizeof(float);
                dev->memcpy_d2d_async(dst + dst_off, src + src_off, row_bytes, opts.stream);
            }
        }
    }

    // Step 3: Confidence + sampling on primary GPU only.
    // opts.logits_out now has full [num_tokens, vocab_size] on the primary GPU.
    // Restore CUDA device to primary (loop left it on last non-primary rank).
    deps_.device_backends[primary_gpu]->set_device();

    // TD-SERVE-NAMED-TOOL-CHOICE / TD-ORCH-SAMPLED-SPEC: full-logits
    // readback of the FIRST min(num_tokens, kMaxLogitsReadbackRows) rows
    // after the allgather (+ transpose) — stream-ordered on opts.stream so
    // the completion event gates host visibility.  num_tokens == 1 is the
    // historical guided-decoding single-row copy byte-for-byte.
    if (opts.logits_host_dst && num_tokens > 0) {
        const size_t n_rows = std::min<size_t>(
            static_cast<size_t>(num_tokens), ipc::kMaxLogitsReadbackRows);
        deps_.stream_manager->memcpy_d2h_async(
            opts.logits_host_dst,
            opts.logits_out,
            n_rows * static_cast<size_t>(mc.vocab_size) * sizeof(float),
            static_cast<int>(primary_gpu), opts.stream_id);
    }

    bool confidence_launched = false;
    if (opts.compute_confidence && num_tokens > 0
        && primary_gpu < confidence_top1_scratch_.size()
        && confidence_top1_scratch_[primary_gpu]
        && primary_gpu < confidence_entropy_scratch_.size()
        && confidence_entropy_scratch_[primary_gpu]) {
        auto* top1_dev = static_cast<float*>(confidence_top1_scratch_[primary_gpu]);
        auto* entr_dev = static_cast<float*>(confidence_entropy_scratch_[primary_gpu]);
        compute::launch_compute_confidence(
            opts.logits_out, top1_dev, entr_dev,
            num_tokens, mc.vocab_size, opts.stream);

        const int last = num_tokens - 1;
        if (opts.readback_host_dst) {
            // top1_prob after the token block — see the non-TP variant.
            const int n = std::min(
                num_tokens,
                static_cast<int>(ipc::kMaxOutputHeadReadbackTokens));
            deps_.stream_manager->memcpy_d2h_async(
                opts.readback_host_dst + 4 * static_cast<size_t>(n),
                top1_dev + last,
                sizeof(float), static_cast<int>(primary_gpu), opts.stream_id);
        }
        // Host staging ALWAYS filled (TD-14i/#91) — see the non-TP variant.
        if (primary_gpu < confidence_host_staging_.size()) {
            auto& staging = confidence_host_staging_[primary_gpu];
            deps_.stream_manager->memcpy_d2h_async(
                &staging.top1_prob, top1_dev + last,
                sizeof(float), static_cast<int>(primary_gpu), opts.stream_id);
            deps_.stream_manager->memcpy_d2h_async(
                &staging.entropy, entr_dev + last,
                sizeof(float), static_cast<int>(primary_gpu), opts.stream_id);
        }
        confidence_launched = true;
    }

    if (opts.do_sample && num_tokens > 0
        && primary_gpu < spec_scratch_.size()
        && spec_scratch_[primary_gpu].readback) {
        auto* sample_dev = static_cast<int32_t*>(spec_scratch_[primary_gpu].readback);
        compute::launch_sample_tokens(
            sample_dev, opts.logits_out,
            num_tokens, mc.vocab_size,
            opts.temperature, opts.top_p, opts.top_k, opts.seed, opts.stream);
        if (opts.readback_host_dst) {
            // Batched-verify heads read back ALL sampled tokens.
            const int n = std::min(
                num_tokens,
                static_cast<int>(ipc::kMaxOutputHeadReadbackTokens));
            deps_.stream_manager->memcpy_d2h_async(
                opts.readback_host_dst, sample_dev,
                static_cast<size_t>(n) * sizeof(int32_t),
                static_cast<int>(primary_gpu), opts.stream_id);
        }
    }

    return confidence_launched;
}

void CommandDispatcher::handle_run_prefetch_probe(const ipc::Command& cmd) {
    if (!deps_.buffer_registry) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kElmExpertOp,
                    "run_prefetch_probe: buffer registry not configured");
        return;
    }

    const auto& p = cmd.run_prefetch_probe;
    // Validate basic fields.
    if (p.num_tokens == 0 || p.num_tokens > ipc::kMaxBatchDescriptors) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kElmExpertOp,
                    "run_prefetch_probe: num_tokens out of range");
        return;
    }

    // Async completion via event.
    const int probe_gpu = static_cast<int>(cmd.gpu_idx);
    void* event = create_and_record_event(probe_gpu, to_stream(cmd.stream_id));

    PendingCompute pc{};
    pc.cmd_seq    = cmd.cmd_seq;
    pc.gpu_idx    = cmd.gpu_idx;
    pc.cmd_type   = cmd.cmd_type;
    pc.layer_idx  = p.target_layer;
    pc.cuda_event = event;
    pending_compute_.push_back(pc);
}

void CommandDispatcher::handle_run_adapter_forward(const ipc::Command& cmd) {
    if (!deps_.buffer_registry) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kAdapterForward,
                    "run_adapter_forward: buffer registry not configured");
        return;
    }

    const auto& p = cmd.run_adapter_forward;
    // Resolve adapter weights buffer.
    auto resolved = deps_.buffer_registry->resolve(p.adapter_weights_buf_id);
    if (!resolved) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kAdapterForward,
                    "run_adapter_forward: invalid adapter_weights_buf_id");
        return;
    }

    // Async completion via event.
    const int adapter_gpu = static_cast<int>(cmd.gpu_idx);
    void* event = create_and_record_event(adapter_gpu, to_stream(cmd.stream_id));

    PendingCompute pc{};
    pc.cmd_seq    = cmd.cmd_seq;
    pc.gpu_idx    = cmd.gpu_idx;
    pc.cmd_type   = cmd.cmd_type;
    pc.layer_idx  = 0;
    pc.cuda_event = event;
    pending_compute_.push_back(pc);
}

void CommandDispatcher::handle_run_mtp_step(const ipc::Command& cmd) {
    // KD-3c: fused MTP draft step pipeline.
    // If CUDA pipeline prerequisites met, dispatch real GPU work;
    // otherwise fall through to stub path (preserves backward compat for tests).
    if (deps_.cuda_kernels_enabled && deps_.live_config && deps_.sideband_base
        && cmd.gpu_idx < spec_scratch_.size() && spec_scratch_[cmd.gpu_idx].hidden_a) {
        run_mtp_pipeline(cmd);
        return;
    }

    // Stub path: emit completion sequence without GPU work.
    const auto& p = cmd.run_mtp_step;

    write_checkpoint_completion(
        cmd.cmd_type, cmd.cmd_seq, cmd.gpu_idx,
        p.mtp_layer_idx,
        static_cast<uint8_t>(ipc::CheckpointType::kGatingOutput),
        /*host_buf_offset=*/0, /*data_bytes=*/0);

    const int mtp_gpu = static_cast<int>(cmd.gpu_idx);
    void* event = create_and_record_event(mtp_gpu, to_stream(cmd.stream_id));

    PendingCompute pc{};
    pc.cmd_seq     = cmd.cmd_seq;
    pc.gpu_idx     = cmd.gpu_idx;
    pc.cmd_type    = cmd.cmd_type;
    pc.layer_idx   = p.mtp_layer_idx;
    pc.cuda_event  = event;
    pc.data_bytes  = 8;
    pending_compute_.push_back(pc);
}

// ── DSP-3/DSP-4: DSpark draft step (backbone + sequential Markov head) ──────
// ONE backbone forward over the whole gamma block (draft latency
// ~independent of gamma — the DFlash point).  The runtime already holds the
// ingested context KV (aux-hidden export, maybe_dspark_capture); this
// command embeds [anchor, mask x (gamma-1)], runs the 5-layer dense Qwen3
// stack with non-causal block attention (base_logits [gamma, V] FP32 +
// hidden_out [gamma, H] BF16), then chains the DSP-4 sequential Markov head
// (INV-DSPARK-MARKOV) on the same stream: per-step low-rank transition bias
// from the actually-sampled predecessor + greedy argmax, leaving
// draft_tokens [gamma] i32 + corrected_logits [gamma, V] FP32 device-
// resident for DSP-5 — plus, when confidence_enabled, the DSP-6 trained
// confidence head (INV-DSPARK-CONF) over [hidden_k ; markov e-chain].
// Completion: one CMP_COMPUTE_DONE off an event on the draft stream (no
// per-position relaunch).  KD-4h: no host sideband pointers enter kernels;
// outputs stay device-resident.

void CommandDispatcher::handle_run_dspark_step(const ipc::Command& cmd) {
    const auto& p = cmd.run_dspark_step;
    auto* rt = deps_.dspark;
    if (!rt || !deps_.cuda_kernels_enabled || !deps_.stream_manager) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kComputeValidation,
                    "dspark_step: runtime not armed "
                    "(speculation.method=dspark + CUDA required)");
        return;
    }

    // LS_DSPARK_PROF=1: daemon-thread wall split of the whole handler
    // (DSP52_BOOST lever-2 — this runs on the daemon thread, so its CPU
    // cost serializes ALL command dispatch and defeats command-level
    // overlap; the runtime prints the run_step sync/enqueue split).
    static const bool dspark_prof = [] {
        const char* e = std::getenv("LS_DSPARK_PROF");
        return e && *e == '1';
    }();
    const auto t0 = std::chrono::steady_clock::now();

    std::string err;
    if (!rt->run_step(p.seq_id, p.anchor_token_id, p.anchor_pos,
                      static_cast<int>(p.num_query), &err)) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kComputeValidation, err.c_str());
        return;
    }
    const auto t_fwd = std::chrono::steady_clock::now();

    // DSP-4: sequential Markov head — the left-to-right greedy sample loop
    // chains on the same draft stream (INV-DSPARK-MARKOV), leaving
    // draft_tokens [gamma] i32 + corrected_logits [gamma, V] FP32
    // device-resident for DSP-5 verification.
    if (!rt->run_markov_head(&err)) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kComputeValidation, err.c_str());
        return;
    }

    // DSP-6: trained confidence head — gated on
    // speculation.dspark.confidence_enabled (validated against the
    // checkpoint's enable_confidence_head at load); when off, the path is
    // inert and DSP-5 behavior is unchanged.  One kernel over the gamma
    // positions, chained on the same draft stream after the Markov e-chain
    // it consumes (INV-DSPARK-CONF).
    const bool conf_enabled =
        deps_.live_config &&
        deps_.live_config->speculation.dspark.confidence_enabled;
    if (conf_enabled && !rt->run_confidence_head(&err)) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kComputeValidation, err.c_str());
        return;
    }

    // EPM-1 (Phase 29): feature-side training dump — one EPMB record per
    // block (per-backbone-layer hiddens + draft ids + c_k), keyed
    // (seq_id, anchor_pos). Inert single branch when the dump is off
    // (epm_dump_enabled is a null-check); when on it syncs the draft GPU
    // (collection mode — timing changes expected and confined to dump-ON).
    if (rt->epm_dump_enabled()) rt->epm_write_block_record(conf_enabled);

    // DSP-5/DSP-6 readback — async D2H into the sideband readback scratch
    // (kSpecCheckpointOff + 2560, the same slice the MTP/self-spec fused
    // steps use; commands are mutually exclusive per ring ordering),
    // stream-ordered AFTER the Markov chain (+ confidence head) on the
    // draft stream, BEFORE the completion event — so when CMP_COMPUTE_DONE
    // fires the driver (DsparkStepExecutor / Python loop) reads straight
    // from sideband + host_buf_offset without touching CUDA (KD-4h: the D2H
    // goes through the DeviceBackend scratch-readback seam).  Layout:
    // gamma i32 sampled ids, then — iff confidence_enabled — gamma f32 raw
    // survival probabilities c_k (data_bytes = gamma * 4 * (1 + conf); both
    // sides know confidence_enabled from config, gamma*8 <= 128 fits the
    // 1536-byte scratch).
    const int gamma = rt->last_num_query();
    const uint32_t readback_off = static_cast<uint32_t>(
        ipc::IpcLayout::kSpecCheckpointOff + 2560);
    const uint32_t ids_bytes = static_cast<uint32_t>(gamma) * sizeof(int32_t);
    const uint32_t readback_bytes = conf_enabled ? 2 * ids_bytes : ids_bytes;
    if (deps_.sideband_base && gamma > 0) {
        rt->draft_backend()->memcpy_d2h_async(
            deps_.sideband_base + readback_off, rt->draft_tokens(),
            ids_bytes, rt->draft_stream());
        if (conf_enabled)
            rt->draft_backend()->memcpy_d2h_async(
                deps_.sideband_base + readback_off + ids_bytes,
                rt->confidence(), static_cast<uint32_t>(gamma) * sizeof(float),
                rt->draft_stream());
    }

    // Completion event on the DRAFT stream (the whole block pipeline is
    // stream-ordered there).  The draft GPU position overrides the header
    // gpu_idx: the command always executes on the draft device.
    if (dspark_prof) {
        const auto t_now = std::chrono::steady_clock::now();
        std::fprintf(
            stderr,
            "[dspark-prof] handler: run_step=%.3f ms heads+readback=%.3f ms\n",
            std::chrono::duration<double, std::milli>(t_fwd - t0).count(),
            std::chrono::duration<double, std::milli>(t_now - t_fwd).count());
    }
    const int draft_gpu = rt->draft_gpu_position();
    // Record on the RUNTIME'S DRAFT STREAM, not the draft GPU's kAttention
    // stream. On a non-TP draft GPU they are the same stream (engine wires
    // the draft onto kAttention there — byte-identical behavior). On a TP
    // draft GPU (TD-DSPARK-DRAFT-QUANT placement) kAttention is the
    // TARGET's attention stream: recording there fired the completion
    // BEFORE the draft pipeline + sideband readback finished (latent
    // early-fire race — masked in overlap mode by the ~150 ms plain step,
    // but a sequential driver could read stale draft ids).
    void* event = deps_.stream_manager->create_event(draft_gpu);
    rt->draft_backend()->record_event(event, rt->draft_stream());
    PendingCompute pc{};
    pc.cmd_seq    = cmd.cmd_seq;
    pc.gpu_idx    = static_cast<uint32_t>(draft_gpu);
    pc.cmd_type   = cmd.cmd_type;
    pc.layer_idx  = p.step_idx;
    pc.cuda_event = event;
    if (deps_.sideband_base && gamma > 0) {
        pc.host_buf_offset = readback_off;
        pc.data_bytes      = readback_bytes;
    }
    pending_compute_.push_back(pc);
}

// ── DSP-3: aux-hidden export hook (INV-DSPARK-AUX) ───────────────────────────
// Runs at the start of dispatch_attention_internal, after the moe_attn_event
// waits: the primary rank's attn_buf holds the post-residual output of
// layer_idx-1 == the INPUT of layer_idx — exactly what vLLM's aux capture
// (deepseek_v2.py: `hidden_states + residual` at the top of layer idx)
// feeds the speculators-format DSpark checkpoint.  Read-only on target
// state; every unsupported step shape fails CLOSED inside the runtime
// (drafting disabled, target untouched).

void CommandDispatcher::maybe_dspark_capture(const InternalAttentionParams& p) {
    auto* rt = deps_.dspark;
    if (!rt || !deps_.cuda_kernels_enabled || !deps_.sideband_base ||
        !deps_.stream_manager)
        return;
    // LS_DSPARK_AUX_SHIFT=1 (TD-DSPARK-ACCEPT-SHORTPROMPT A/B probe):
    // DeepSpec TRAINS on the OUTPUT of aux layer L (forward hook on the
    // layer module / HF hidden_states[L+1]; vLLM converts DFlash ids with
    // +1 — gpu_model_runner._get_eagle3_aux_layers_from_config, "#40727:
    // DFlash configs use different indexing"), while this hook exports the
    // INPUT of layer L (= output of L-1).  The shift interprets aux id L
    // as "capture at the dispatch of layer L+1", i.e. exports layer L's
    // post-residual OUTPUT.  Layer 0 wraps to UINT32_MAX -> no aux match.
    static const bool aux_shift = [] {
        const char* e = std::getenv("LS_DSPARK_AUX_SHIFT");
        return e && *e == '1';
    }();
    const int slot = rt->aux_slot_for_layer(
        aux_shift ? p.layer_idx - 1 : p.layer_idx);
    if (slot < 0) return;

    // Draft (speculation) passes must not extend the real context.
    if (p.is_draft) return;

    const int pair_idx = resolve_pair_idx(p.gpu_idx);
    if (pair_idx < 0 || !deps_.hidden_state_pairs[pair_idx].attn_buf) return;
    const auto& pair = deps_.hidden_state_pairs[pair_idx];

    const auto* bd = reinterpret_cast<const ipc::BatchDescriptorEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kBatchDescriptorOff);

    // Step shape → (rows, start_pos).  Decode: one row per sequence at
    // token_pos (single-sequence only — TD-DSPARK-BATCH).  Prefill: rows =
    // chunk tokens at chunk_start (whole-prompt prefill carries the token
    // count in num_seqs, mirroring dispatch_attention's TD-40j synthesis).
    int rows;
    uint32_t start_pos;
    if (p.is_prefill) {
        if (p.chunk_len > 0) {
            rows = static_cast<int>(p.chunk_len);
            start_pos = p.chunk_start;
        } else {
            rows = static_cast<int>(p.num_seqs);
            start_pos = bd[0].token_pos;
        }
    } else {
        if (p.num_seqs != 1) {
            rt->invalidate_context("multi-sequence decode batch "
                                   "(TD-DSPARK-BATCH)");
            return;
        }
        rows = 1;
        start_pos = bd[0].token_pos;
    }

    // TD-DSPARK-SUPERCHUNK-CAPTURE: a superchunk sub-launch's hidden rows
    // land at row_offset in the pair buffer — offset the capture source so
    // each window exports ITS rows (passing the buffer base re-exported
    // sub-chunk 1's rows for every window: the row_offset defect).
    // V4-5b mHC (INV-DSPARK-AUX): the export representation is the FLATTENED
    // hc-stream residual — hc_streams*hidden per row (16384 for V4-Flash).
    // The draft checkpoint's own hidden width (DsparkRuntime::H_) must match,
    // else the fusion GEMM K-dim silently corrupts — fail loud here.
    const char* src = static_cast<const char*>(pair.attn_buf);
    const int hs_row = deps_.live_config
        ? deps_.live_config->model.hidden_size * deps_.hc_streams : 0;

    // Ticket J: the V4 dflash draft wants the stream-MEAN representation —
    // reduce the hc-wide rows to [rows, hidden] on the TARGET GPU, then
    // capture the compact rows (the draft's H is `hidden`, not hc*hidden).
    if (rt->aux_stream_mean()) {
        const int hidden =
            deps_.live_config ? deps_.live_config->model.hidden_size : 0;
        if (deps_.hc_streams <= 1 || hidden <= 0
            || rt->aux_hidden_width() != hidden) {
            rt->invalidate_context("V4 aux capture: stream-mean draft needs "
                                   "an hc-wide target with draft H == "
                                   "hidden");
            return;
        }
        if (rows > kDsparkMeanScratchRows) {
            rt->invalidate_context("V4 aux capture rows exceed the stream-"
                                   "mean staging");
            return;
        }
        if (dspark_mean_scratch_.size() < deps_.device_backends.size())
            dspark_mean_scratch_.resize(deps_.device_backends.size(),
                                        nullptr);
        void*& scr = dspark_mean_scratch_[p.gpu_idx];
        auto* be = deps_.device_backends[p.gpu_idx];
        if (!scr) {
            be->set_device();
            scr = be->device_alloc(
                static_cast<size_t>(kDsparkMeanScratchRows) *
                static_cast<size_t>(hidden) * 2);
            if (!scr) {
                rt->invalidate_context("V4 aux stream-mean staging alloc "
                                       "failed");
                return;
            }
        }
        const char* mean_src = src;
        if (p.row_offset > 0)
            mean_src += static_cast<size_t>(p.row_offset)
                        * static_cast<size_t>(hs_row) * 2;
        void* src_stream = deps_.stream_manager->stream(
            static_cast<int>(p.gpu_idx), compute::StreamId::kAttention);
        // Stream-ordered on the same kAttention stream the capture copy
        // uses — no extra events (the previous step's capture copy from
        // this staging precedes this overwrite in stream order).
        compute::launch_hc_stream_mean(scr, mean_src, rows,
                                       deps_.hc_streams, hidden, src_stream);
        rt->capture_aux(slot, scr, rows, bd[0].seq_id, start_pos, *be,
                        src_stream);
        return;
    }

    if (deps_.hc_streams > 1 && rt->aux_hidden_width() > 0
        && rt->aux_hidden_width() != hs_row) {
        rt->invalidate_context("aux-hidden width mismatch: draft expects a "
                               "different width than hc_streams*hidden");
        return;
    }
    if (p.row_offset > 0) {
        if (hs_row <= 0) {
            rt->invalidate_context("superchunk capture without model config");
            return;
        }
        src += static_cast<size_t>(p.row_offset) * static_cast<size_t>(hs_row)
               * 2 /* BF16 */;
    }

    void* src_stream = deps_.stream_manager->stream(
        static_cast<int>(p.gpu_idx), compute::StreamId::kAttention);
    rt->capture_aux(slot, src, rows, bd[0].seq_id, start_pos,
                    *deps_.device_backends[p.gpu_idx], src_stream);
}

void CommandDispatcher::maybe_dspark_capture_final(uint32_t gpu,
                                                   const void* head_input,
                                                   int num_tokens,
                                                   int mtp_head_idx,
                                                   void* stream) {
    // Ticket J: the V4 dflash draft's LAST aux tap (aux id == num target
    // layers = the final post-layer residual, vLLM `idx+1 == end_layer`
    // convention) fires from the output-head path: head_input holds the
    // committed hc-wide residual the hc_head collapse is about to consume.
    auto* rt = deps_.dspark;
    if (!rt || !rt->aux_stream_mean() || mtp_head_idx >= 0
        || !deps_.cuda_kernels_enabled || !deps_.sideband_base
        || !deps_.live_config || !deps_.stream_manager || num_tokens <= 0
        || !head_input)
        return;
    const auto& mc = deps_.live_config->model;
    const int slot = rt->aux_slot_for_layer(
        static_cast<uint32_t>(mc.num_hidden_layers));
    if (slot < 0) return;
    const int hidden = mc.hidden_size;
    if (deps_.hc_streams <= 1 || rt->aux_hidden_width() != hidden) return;
    // All capture traffic must ride ONE stream (the last slot's
    // ev_capture_done_ covers every slot's copy only then) — the bridge/
    // orchestrator chains run every command on kAttention (stream_id 0).
    void* attn_stream = deps_.stream_manager->stream(
        static_cast<int>(gpu), compute::StreamId::kAttention);
    if (stream != attn_stream) {
        rt->invalidate_context("V4 final-residual capture off the "
                               "kAttention stream (unsupported head stream)");
        return;
    }
    if (num_tokens > kDsparkMeanScratchRows) {
        rt->invalidate_context("V4 final-residual capture rows exceed the "
                               "stream-mean staging");
        return;
    }
    if (dspark_mean_scratch_.size() < deps_.device_backends.size())
        dspark_mean_scratch_.resize(deps_.device_backends.size(), nullptr);
    void*& scr = dspark_mean_scratch_[gpu];
    auto* be = deps_.device_backends[gpu];
    if (!scr) {
        be->set_device();
        scr = be->device_alloc(static_cast<size_t>(kDsparkMeanScratchRows) *
                               static_cast<size_t>(hidden) * 2);
        if (!scr) {
            rt->invalidate_context("V4 aux stream-mean staging alloc failed");
            return;
        }
    }
    const auto* bd = reinterpret_cast<const ipc::BatchDescriptorEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kBatchDescriptorOff);
    // TD-V4-SPEC-PREFILL-CTX dedupe: if the MoE-final tap already captured
    // this window at the last layer's FETCH_AND_RUN finalize, this head's
    // rows are the same committed bytes — a second slot capture would
    // invalidate the freshly closed epoch (duplicate-slot contract). Skip.
    if (dspark_moe_final_mark_.valid
        && bd[0].seq_id == dspark_moe_final_mark_.seq_id
        && bd[0].token_pos >= dspark_moe_final_mark_.start
        && static_cast<uint64_t>(bd[0].token_pos) + num_tokens
               <= dspark_moe_final_mark_.end)
        return;
    compute::launch_hc_stream_mean(scr, head_input, num_tokens,
                                   deps_.hc_streams, hidden, stream);
    rt->capture_aux(slot, scr, num_tokens, bd[0].seq_id, bd[0].token_pos,
                    *be, stream);
    be->set_device();  // capture_aux restores the source device; be explicit
}

void CommandDispatcher::maybe_dspark_capture_moe_final(uint32_t gpu,
                                                       uint32_t layer_idx,
                                                       int num_tokens) {
    // TD-V4-SPEC-PREFILL-CTX: the V4 dflash final aux tap (id == num
    // layers) is head-sited, but headless prefill chunks/superchunks never
    // run a head — their capture epoch stays open and the next step's
    // slot-0 capture hits a "position gap", killing the draft context.
    // At the LAST layer's MoE finalize the pair attn_buf holds exactly the
    // bytes a head would consume (the committed hc-wide final residual for
    // every window row), so fire the final tap from here whenever the
    // runtime's epoch awaits only the final slot. The window comes from
    // the RUNTIME (pending_final_window), not the batch descriptors — at
    // MOE_BIG time the descriptors hold only the last attention sub-chunk.
    auto* rt = deps_.dspark;
    if (!rt || !rt->aux_stream_mean() || !deps_.cuda_kernels_enabled
        || !deps_.live_config || !deps_.stream_manager || num_tokens <= 0)
        return;
    const auto& mc = deps_.live_config->model;
    if (static_cast<int>(layer_idx) != mc.num_hidden_layers - 1) return;
    const int slot = rt->aux_slot_for_layer(
        static_cast<uint32_t>(mc.num_hidden_layers));
    if (slot < 0 || slot != rt->aux_count() - 1) return;
    const int hidden = mc.hidden_size;
    if (deps_.hc_streams <= 1 || rt->aux_hidden_width() != hidden) return;

    uint64_t seq = 0;
    uint32_t start = 0, end = 0;
    if (!rt->pending_final_window(&seq, &start, &end)) return;
    const int rows_total = static_cast<int>(end - start);
    // The epoch window must be exactly this MoE step's rows — anything else
    // is a foreign/stale epoch: leave it to the existing fail-closed
    // contracts (the next slot-0 capture invalidates on the gap).
    if (rows_total != num_tokens) return;

    const int pair_idx = resolve_pair_idx(gpu);
    if (pair_idx < 0 || !deps_.hidden_state_pairs[pair_idx].attn_buf) return;
    const auto& pair = deps_.hidden_state_pairs[pair_idx];

    // All capture traffic rides the primary rank's kAttention stream;
    // order it after the MoE residual commit (recorded on kExpertFfn).
    void* attn_stream = deps_.stream_manager->stream(
        static_cast<int>(gpu), compute::StreamId::kAttention);
    if (pair.moe_attn_event)
        deps_.stream_manager->wait_event(static_cast<int>(gpu),
                                         compute::StreamId::kAttention,
                                         pair.moe_attn_event);

    if (dspark_mean_scratch_.size() < deps_.device_backends.size())
        dspark_mean_scratch_.resize(deps_.device_backends.size(), nullptr);
    void*& scr = dspark_mean_scratch_[gpu];
    auto* be = deps_.device_backends[gpu];
    if (!scr) {
        be->set_device();
        scr = be->device_alloc(static_cast<size_t>(kDsparkMeanScratchRows) *
                               static_cast<size_t>(hidden) * 2);
        if (!scr) {
            rt->invalidate_context("V4 aux stream-mean staging alloc failed");
            return;
        }
    }

    const size_t hs_row_bytes =
        static_cast<size_t>(hidden) * deps_.hc_streams * 2;
    const char* buf = static_cast<const char*>(pair.attn_buf);
    for (int off = 0; off < rows_total; off += kDsparkMeanScratchRows) {
        const int rows = std::min(kDsparkMeanScratchRows, rows_total - off);
        be->set_device();
        // Stream-ordered staging reuse: the previous piece's capture copy
        // from `scr` precedes this overwrite on the same kAttention stream.
        compute::launch_hc_stream_mean(
            scr, buf + static_cast<size_t>(off) * hs_row_bytes, rows,
            deps_.hc_streams, hidden, attn_stream);
        rt->capture_aux(slot, scr, rows, seq,
                        start + static_cast<uint32_t>(off), *be, attn_stream);
        if (!rt->ctx_valid()) break;  // fail-closed inside the runtime
    }
    dspark_moe_final_mark_ = {seq, start, end, /*valid=*/true};
    be->set_device();  // capture_aux restores the source device; be explicit
}

void CommandDispatcher::handle_run_self_spec_forward(const ipc::Command& cmd) {
    // KD-3c: fused self-spec forward pass pipeline.
    if (deps_.cuda_kernels_enabled && deps_.live_config && deps_.sideband_base
        && cmd.gpu_idx < spec_scratch_.size() && spec_scratch_[cmd.gpu_idx].hidden_a) {
        run_self_spec_pipeline(cmd);
        return;
    }

    // Stub path: emit completion sequence without GPU work.
    const auto& p = cmd.self_spec_forward;
    const uint64_t skip_lo = p.skip_mask_lo;
    const uint64_t skip_hi = p.skip_mask_hi;

    uint32_t num_layers = 61;
    uint32_t first_moe_layer = 3;
    if (deps_.live_config) {
        num_layers = static_cast<uint32_t>(
            deps_.live_config->model.num_hidden_layers);
        first_moe_layer = static_cast<uint32_t>(
            deps_.live_config->model.first_k_dense_replace);
    }

    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        const bool skipped = (layer < 64)
            ? ((skip_lo >> layer) & 1u) != 0
            : (layer < 128) ? ((skip_hi >> (layer - 64)) & 1u) != 0 : false;
        if (skipped) continue;

        write_checkpoint_completion(
            cmd.cmd_type, cmd.cmd_seq, cmd.gpu_idx, layer,
            static_cast<uint8_t>(ipc::CheckpointType::kLayerSimilarity),
            /*host_buf_offset=*/0, /*data_bytes=*/4);

        if (p.store_gating && layer >= first_moe_layer) {
            write_checkpoint_completion(
                cmd.cmd_type, cmd.cmd_seq, cmd.gpu_idx, layer,
                static_cast<uint8_t>(ipc::CheckpointType::kGatingOutput),
                /*host_buf_offset=*/0, /*data_bytes=*/0);
        }
    }

    const int spec_gpu = static_cast<int>(cmd.gpu_idx);
    void* event = create_and_record_event(spec_gpu, to_stream(cmd.stream_id));

    PendingCompute pc{};
    pc.cmd_seq     = cmd.cmd_seq;
    pc.gpu_idx     = cmd.gpu_idx;
    pc.cmd_type    = cmd.cmd_type;
    pc.layer_idx   = num_layers - 1;
    pc.cuda_event  = event;
    pc.data_bytes  = 8;
    pending_compute_.push_back(pc);
}

// ── F-5: Autonomous one-layer forward ───────────────────────────────────────
// E_FORWARD_ONE_LAYER delegates a full decode layer (attention + gate + MoE)
// to the existing forward_one_layer() primitive as a single daemon op.
// NOTE (production path decision, 2026-07-05): this composite's routed-MoE
// stage predates the FETCH_AND_RUN_MOE production seam — a production
// single-op forward must be rebased on E_CMD_FETCH_AND_RUN_MOE semantics
// (routed-list fetch+run) before it ships on the hot path.
// Output is identical to issuing D_B_CMD_RUN_ATTENTION then D_B_CMD_RUN_MOE for
// the same layer — forward_one_layer() is exactly what the two-op pair composes
// into (see dispatch_forward.cpp::forward_one_layer). Mirrors the completion
// contract of handle_fused_compute_command's D_B_CMD_RUN_MOE branch: one
// CMP_COMPUTE_DONE on the MoE stream (kExpertFfn, the layer's last stage),
// carrying the routed-expert miss count (TD-89m).

void CommandDispatcher::handle_forward_one_layer(const ipc::Command& cmd) {
    if (!deps_.sideband_base) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kForwardOneLayer,
                    "forward_one_layer: sideband not configured");
        return;
    }

    const auto& p = cmd.forward_one_layer;
    if (p.num_seqs == 0 || p.num_seqs > ipc::kMaxBatchDescriptors) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kForwardOneLayer,
                    "forward_one_layer: num_seqs out of range");
        return;
    }

    // Delegate the whole layer to the shared primitive (attention + gate + MoE).
    // The batch descriptor is read from sideband by dispatch_attention_internal /
    // dispatch_moe_internal, exactly as for the D_B_CMD_RUN_ATTENTION/MOE pair.
    const bool ok = forward_one_layer({
        .layer_idx    = p.layer_idx,
        .num_seqs     = p.num_seqs,
        .gpu_idx      = cmd.gpu_idx,
        .is_prefill   = p.is_prefill != 0,
        .use_graph    = p.use_graph != 0,
        .is_draft     = p.is_draft != 0,
        .topk_override = p.topk_override,
        .store_gating = p.store_gating != 0,
    });

    if (!ok) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    last_internal_error_msg_
                        ? last_internal_error_cat_
                        : ipc::CmpErrorCategory::kForwardOneLayer,
                    last_internal_error_msg_
                        ? last_internal_error_msg_
                        : "forward_one_layer: dispatch failed (see log)");
        return;
    }

    // Async completion via event on the MoE stream (kExpertFfn) — MoE is the
    // layer's final stage, matching the D_B_CMD_RUN_MOE completion path.
    void* event = create_and_record_event(static_cast<int>(cmd.gpu_idx),
                                          compute::StreamId::kExpertFfn);

    PendingCompute pc{};
    pc.cmd_seq          = cmd.cmd_seq;
    pc.gpu_idx          = cmd.gpu_idx;
    pc.cmd_type         = cmd.cmd_type;
    pc.layer_idx        = p.layer_idx;
    pc.cuda_event       = event;
    // TD-89m: propagate routed expert miss count, as the RUN_MOE branch does.
    pc.routed_miss_count = last_moe_miss_count_;
    pending_compute_.push_back(pc);
}

// ── #16 / GLM-25g: MTP projection (resolves TD-50m) ─────────────────────────
// eh_proj(concat(enorm(Emb(token)), hnorm(prev_hidden))) → attn_buf on every
// TP rank.  DeepSeek V3 §2.2 eq. 21: h'_k = M_k [RMSNorm(h_{k-1}); RMSNorm(
// Emb(t_k))].  prev_hidden is each rank's CURRENT attn_buf content — the
// trunk hidden after a main-model step (draft step 0) or the MTP layer's
// committed output after a chained draft step (the recurrence anchor); the
// caller sequences commands so the right hidden is resident.
//
// Stream discipline: the primary rank runs on kExpertFfn (the speculation
// pipeline stream, matching setup_spec_pipeline); every other rank runs on
// its own kAttention stream (where the sharded-embed lookup/allreduce and
// the TD-73n broadcast land, so all attn_buf reads/writes are in-order).
// The ONE cross-stream hazard — the primary's hnorm (kAttention-visible
// attn_buf read... actually issued on kExpertFfn) vs the embed's attn_buf
// overwrite — is closed by issuing the primary hnorm on kExpertFfn BEFORE
// setup_spec_pipeline (same stream, in-order) and each secondary hnorm on
// that rank's kAttention BEFORE the embed work lands there.

bool CommandDispatcher::dispatch_mtp_projection(
        uint32_t cmd_seq, uint32_t gpu, uint32_t token_id,
        int mtp_layer_idx, int hidden_row,
        void*& stream_out, int& pair_idx_out) {
    if (!deps_.cuda_kernels_enabled || !deps_.live_config
        || !deps_.sideband_base || !deps_.stream_manager) {
        write_error(cmd_seq, gpu, ipc::CmpErrorCategory::kComputeValidation,
                    "mtp_project: CUDA pipeline prerequisites missing");
        return false;
    }
    const auto& mc = deps_.live_config->model;
    const int NH = mc.num_hidden_layers;
    const int mi = mtp_layer_idx - NH;
    if (mi < 0 || mi >= static_cast<int>(deps_.mtp_eh_proj_ptrs.size())
        || mi >= static_cast<int>(deps_.mtp_enorm_ptrs.size())
        || mi >= static_cast<int>(deps_.mtp_hnorm_ptrs.size())) {
        write_error(cmd_seq, gpu, ipc::CmpErrorCategory::kComputeValidation,
                    "mtp_project: mtp_layer_idx out of range (no MTP weights)");
        return false;
    }
    const int H = mc.hidden_size;
    const size_t h_bytes = static_cast<size_t>(H) * 2;

    // Participating TP ranks = the hidden-state pairs (one per DCP rank).
    const int tp = std::max(
        1, deps_.live_config->parallelism.tensor_parallelism);
    if (static_cast<int>(deps_.hidden_state_pairs.size()) < tp
        || H % tp != 0) {
        write_error(cmd_seq, gpu, ipc::CmpErrorCategory::kComputeValidation,
                    "mtp_project: hidden-state pairs/tp mismatch");
        return false;
    }
    const int n_local = H / tp;  // eh_proj output-row shard per rank

    // Validate every rank's weights + scratch up front (fail loud, launch
    // nothing on a partial config).
    for (int r = 0; r < tp; ++r) {
        const auto& pair = deps_.hidden_state_pairs[r];
        const auto pos = static_cast<size_t>(pair.gpu_position);
        const bool ok = pair.attn_buf
            && pos < spec_scratch_.size()
            && spec_scratch_[pos].hidden_a && spec_scratch_[pos].mtp_concat
            && pos < deps_.mtp_enorm_ptrs[mi].size()
            && deps_.mtp_enorm_ptrs[mi][pos]
            && pos < deps_.mtp_hnorm_ptrs[mi].size()
            && deps_.mtp_hnorm_ptrs[mi][pos]
            && pos < deps_.mtp_eh_proj_ptrs[mi].size()
            && deps_.mtp_eh_proj_ptrs[mi][pos];
        if (!ok) {
            write_error(cmd_seq, gpu,
                        ipc::CmpErrorCategory::kComputeValidation,
                        "mtp_project: MTP projection weights/scratch missing "
                        "on a TP rank (enorm/hnorm/eh_proj)");
            return false;
        }
    }

    const int primary_pair = resolve_pair_idx(gpu);
    if (primary_pair < 0) {
        write_error(cmd_seq, gpu, ipc::CmpErrorCategory::kComputeValidation,
                    "mtp_project: no hidden-state pair for primary GPU");
        return false;
    }

    // Batched-verify catch-ups (hidden_row > 0): prev_hidden is a higher
    // row of the K-token verify pass's trunk hiddens.  attn_buf holds
    // max_batch_size rows; fail loud on an out-of-range row rather than
    // hnorm-ing past the buffer.
    if (hidden_row < 0 || hidden_row >= std::max(1, deps_.max_batch_size)) {
        write_error(cmd_seq, gpu, ipc::CmpErrorCategory::kComputeValidation,
                    "mtp_project: hidden_row out of range");
        return false;
    }

    auto rank_stream = [&](int r) -> void* {
        const auto& pair = deps_.hidden_state_pairs[r];
        return (r == primary_pair)
            ? deps_.stream_manager->stream(pair.gpu_position,
                                           compute::StreamId::kExpertFfn)
            : deps_.stream_manager->stream(pair.gpu_position,
                                           compute::StreamId::kAttention);
    };

    const float eps = static_cast<float>(mc.rms_norm_eps);

    // Step 1: hnorm(prev_hidden = attn_buf_r row `hidden_row`) →
    // concat[H..2H) on each rank, BEFORE the embed overwrites attn_buf row
    // 0.  Wait the pair's moe_attn_event first so the previous layer's MoE
    // commit (kExpertFfn) is visible.  hidden_row > 0 selects a trunk
    // hidden of a batched verify pass (rows [0..K) of attn_buf) — every
    // write below touches ONLY row 0, so higher rows survive a sequential
    // chain of MTP steps.
    const size_t row_off = static_cast<size_t>(hidden_row) * h_bytes;
    for (int r = 0; r < tp; ++r) {
        const auto& pair = deps_.hidden_state_pairs[r];
        const int pos = pair.gpu_position;
        auto& ssr = spec_scratch_[pos];
        void* s = rank_stream(r);
        deps_.device_backends[pos]->set_device();
        if (pair.moe_attn_event)
            deps_.device_backends[pos]->stream_wait_event(
                s, pair.moe_attn_event);
        auto* concat_hi = static_cast<char*>(ssr.mtp_concat) + h_bytes;
        const void* prev_hidden =
            static_cast<const char*>(pair.attn_buf) + row_off;
        compute::launch_rmsnorm(
            concat_hi, prev_hidden, deps_.mtp_hnorm_ptrs[mi][pos],
            eps, 1, H, compute::NormDtype::kBFloat16, s);
    }

    // Step 2: embed token → primary ss.hidden_a + every rank's attn_buf
    // (setup_spec_pipeline: sharded lookup+allreduce or lookup+broadcast).
    // Non-primary hnorms above are on the same kAttention streams the embed
    // writes land on (in-order); the primary hnorm is on kExpertFfn, the
    // same stream setup uses for its attn_buf overwrite (in-order).
    auto& ss = spec_scratch_[gpu];
    if (!setup_spec_pipeline(cmd_seq, gpu, token_id, "mtp_project",
                             ss, stream_out, pair_idx_out))
        return false;

    // Step 3: enorm(embedding) → concat[0..H) on each rank.  Primary reads
    // ss.hidden_a (kExpertFfn, in-order after the embed); secondaries read
    // their attn_buf copy (kAttention, in-order after the broadcast).
    for (int r = 0; r < tp; ++r) {
        const auto& pair = deps_.hidden_state_pairs[r];
        const int pos = pair.gpu_position;
        auto& ssr = spec_scratch_[pos];
        void* s = rank_stream(r);
        const void* embed_src =
            (r == primary_pair) ? ss.hidden_a : pair.attn_buf;
        deps_.device_backends[pos]->set_device();
        compute::launch_rmsnorm(
            ssr.mtp_concat, embed_src, deps_.mtp_enorm_ptrs[mi][pos],
            eps, 1, H, compute::NormDtype::kBFloat16, s);
    }

    // Step 4: eh_proj partial GEMM per rank.  Each rank holds the output-row
    // shard [n_local, 2H]; it GEMMs into its shard's offset of a ZEROED
    // full-H buffer so the allreduce-sum below reconstructs the full
    // projection on every rank (same collective pattern as the sharded
    // embedding lookup, TD-GOLDEN-EMB-OOB).
    const bool have_gguf_meta =
        mi < static_cast<int>(deps_.mtp_eh_proj_is_gguf.size());
    for (int r = 0; r < tp; ++r) {
        const auto& pair = deps_.hidden_state_pairs[r];
        const int pos = pair.gpu_position;
        const int rank = pair.rank >= 0 ? pair.rank : r;
        auto& ssr = spec_scratch_[pos];
        void* s = rank_stream(r);
        deps_.device_backends[pos]->set_device();
        deps_.device_backends[pos]->memset_async(ssr.hidden_a, 0, h_bytes, s);
        void* c_shard = static_cast<char*>(ssr.hidden_a)
                      + static_cast<size_t>(rank) * n_local * 2;
        const bool is_gguf = have_gguf_meta
            && pos < deps_.mtp_eh_proj_is_gguf[mi].size()
            && deps_.mtp_eh_proj_is_gguf[mi][pos] != 0;
        if (is_gguf) {
            // Packed k-quant eh_proj (GLM-5.2 GGUF: Q8_0) — GGUF GEMM route.
            auto* attn_dev = (pos < static_cast<int>(
                                  deps_.attention_devices.size()))
                ? deps_.attention_devices[pos] : nullptr;
            if (!deps_.dcp_executor || !attn_dev) {
                write_error(cmd_seq, gpu,
                            ipc::CmpErrorCategory::kComputeValidation,
                            "mtp_project: GGUF eh_proj requires DCP executor "
                            "+ attention device");
                return false;
            }
            deps_.dcp_executor->route_gguf_gemm(
                attn_dev, rank, /*M=*/1, /*N=*/n_local, /*K=*/2 * H,
                ssr.mtp_concat, deps_.mtp_eh_proj_ptrs[mi][pos], c_shard,
                deps_.mtp_eh_proj_gguf_type[mi][pos], s);
        } else {
            // BF16 eh_proj rows (safetensors checkpoints).
            compute::launch_bf16_gemm_nt(
                c_shard, ssr.mtp_concat, deps_.mtp_eh_proj_ptrs[mi][pos],
                /*M=*/1, /*N=*/n_local, /*K=*/2 * H,
                compute::GemmInDtype::kBFloat16,
                compute::GemmAccOutDtype::kBFloat16, s);
        }
    }

    // Step 5: allreduce-sum the per-rank partials → full projection everywhere.
    if (tp > 1) {
        if (!deps_.dcp_communicator || !deps_.dcp_communicator->is_active()) {
            write_error(cmd_seq, gpu,
                        ipc::CmpErrorCategory::kComputeValidation,
                        "mtp_project: TP>1 eh_proj needs an active DCP "
                        "communicator");
            return false;
        }
        static constexpr int kMaxTp = 8;
        if (tp > kMaxTp) {
            write_error(cmd_seq, gpu,
                        ipc::CmpErrorCategory::kComputeValidation,
                        "mtp_project: tp exceeds kMaxTp");
            return false;
        }
        void* bufs[kMaxTp] = {};
        void* streams[kMaxTp] = {};
        for (int r = 0; r < tp; ++r) {
            bufs[r] = spec_scratch_[
                deps_.hidden_state_pairs[r].gpu_position].hidden_a;
            streams[r] = rank_stream(r);
        }
        deps_.dcp_communicator->allreduce_hidden(bufs, 1, streams);
    }

    // Step 6: projected hidden → attn_buf on every rank (the MTP layer's
    // attention input) + re-record the pair's moe_attn_event so the layer's
    // step-0 wait covers this write.
    for (int r = 0; r < tp; ++r) {
        const auto& pair = deps_.hidden_state_pairs[r];
        const int pos = pair.gpu_position;
        void* s = rank_stream(r);
        deps_.device_backends[pos]->set_device();
        deps_.device_backends[pos]->memcpy_d2d_async(
            pair.attn_buf, spec_scratch_[pos].hidden_a, h_bytes, s);
        if (pair.moe_attn_event)
            deps_.device_backends[pos]->record_event(pair.moe_attn_event, s);
    }

    // Restore primary device context for the caller's next launches.
    deps_.device_backends[gpu]->set_device();
    return true;
}

// D_CMD_MTP_PROJECT: standalone projection so the production-seam MTP draft
// step composes from existing commands (see ipc_protocol.h payload docs):
//   MTP_PROJECT → RUN_ATTENTION(mtp_layer, emit/store gating) →
//   FETCH_AND_RUN_MOE(mtp_layer) → OUTPUT_HEAD(mtp_head) → SAMPLE_TOKENS.
void CommandDispatcher::handle_mtp_project(const ipc::Command& cmd) {
    const auto& p = cmd.mtp_project;
    const uint32_t gpu = cmd.gpu_idx;

    void* stream = nullptr;
    int pair_idx = -1;
    if (!dispatch_mtp_projection(cmd.cmd_seq, gpu, p.input_token_id,
                                 static_cast<int>(p.mtp_layer_idx),
                                 static_cast<int>(p.hidden_row),
                                 stream, pair_idx))
        return;  // CMP_ERROR already written

    void* event = create_and_record_event(static_cast<int>(gpu),
                                          compute::StreamId::kExpertFfn);
    PendingCompute pc{};
    pc.cmd_seq    = cmd.cmd_seq;
    pc.gpu_idx    = gpu;
    pc.cmd_type   = cmd.cmd_type;
    pc.layer_idx  = p.mtp_layer_idx;
    pc.cuda_event = event;
    pending_compute_.push_back(pc);
}

// ── KD-3c: MTP pipeline ──────────────────────────────────────────────────────

void CommandDispatcher::run_mtp_pipeline(const ipc::Command& cmd) {
    const auto& p = cmd.run_mtp_step;
    const uint32_t gpu = cmd.gpu_idx;
    const auto& mc = deps_.live_config->model;
    auto& ss = spec_scratch_[gpu];

    // TD-50m RESOLVED (#16): full MTP projection — embed + enorm/hnorm +
    // eh_proj — instead of feeding the raw embedding to the layer.
    void* stream = nullptr;
    int pair_idx = -1;
    if (!dispatch_mtp_projection(cmd.cmd_seq, gpu, p.input_token_id,
                                 static_cast<int>(p.mtp_layer_idx),
                                 /*hidden_row=*/0,
                                 stream, pair_idx))
        return;

    // TD-51b RESOLVED: abort the pipeline with CMP_ERROR on layer failure.
    // NOTE (TD-MTP-FUSED-RUNMOE): this fused step predates the production
    // FETCH_AND_RUN seam — its routed MoE runs dispatch_moe_internal
    // (resident experts only).  The production-seam composition is the
    // D_CMD_MTP_PROJECT command chain (see handle_mtp_project).
    if (!forward_one_layer({
            .layer_idx   = p.mtp_layer_idx,
            .num_seqs    = 1,
            .gpu_idx     = gpu,
            .store_gating = true,
        })) {
        write_error(cmd.cmd_seq, gpu,
                    last_internal_error_msg_
                        ? last_internal_error_cat_
                        : ipc::CmpErrorCategory::kComputeValidation,
                    last_internal_error_msg_
                        ? last_internal_error_msg_
                        : "mtp_step: forward_one_layer failed (see log)");
        return;
    }

    // Copy MoE output back to hidden_a for output head.
    if (deps_.cuda_kernels_enabled && pair_idx >= 0
        && deps_.hidden_state_pairs[pair_idx].attn_buf) {
        const auto& pair = deps_.hidden_state_pairs[pair_idx];
        const size_t h_bytes = static_cast<size_t>(mc.hidden_size) * 2;
        deps_.device_backends[gpu]->memcpy_d2d_async(
            ss.hidden_a, pair.attn_buf, h_bytes, stream);
    }

    // Step 5: Gating checkpoint — copy topk_weights to sideband.
    const auto& scratch = moe_scratch_[gpu];
    const int topk = mc.num_experts_per_tok;
    const uint32_t gating_offset = static_cast<uint32_t>(
        ipc::IpcLayout::kSpecCheckpointOff + 512);
    const uint32_t gating_bytes = static_cast<uint32_t>(topk * sizeof(float));
    if (deps_.cuda_kernels_enabled) {
        deps_.device_backends[gpu]->memcpy_d2h_async(
            deps_.sideband_base + gating_offset,
            scratch.topk_weights, gating_bytes, stream);
    } else {
        std::memcpy(deps_.sideband_base + gating_offset,
                    scratch.topk_weights, gating_bytes);
    }

    void* gating_event = create_and_record_event(static_cast<int>(gpu), compute::StreamId::kExpertFfn);

    // Step 7-8: Output head + confidence + sampling (KD-R5).
    auto* readback = deps_.sideband_base
        + ipc::IpcLayout::kSpecCheckpointOff + 2560;
    dispatch_output_head({
        .gpu_idx            = gpu,
        .input              = ss.hidden_a,
        .logits_out         = static_cast<float*>(ss.logits),
        .stream             = stream,
        .stream_id          = compute::StreamId::kExpertFfn,
        .compute_confidence = true,
        .do_sample          = true,
        .readback_host_dst  = readback,
        // #16: MTP shared head — shared_head.norm (+ shared_head.head when
        // not deduped into the main lm_head) instead of final_norm/lm_head.
        .mtp_head_idx       = static_cast<int>(p.mtp_layer_idx)
                              - mc.num_hidden_layers,
    });

    // Record final event.
    void* final_event = create_and_record_event(static_cast<int>(gpu), compute::StreamId::kExpertFfn);

    PendingCompute pc{};
    pc.cmd_seq    = cmd.cmd_seq;
    pc.gpu_idx    = gpu;
    pc.cmd_type   = cmd.cmd_type;
    pc.layer_idx  = p.mtp_layer_idx;
    pc.cuda_event = final_event;
    pc.data_bytes = 8;
    pc.host_buf_offset = static_cast<uint32_t>(
        ipc::IpcLayout::kSpecCheckpointOff + 2560);
    pc.pipeline_checkpoints.push_back({
        .cuda_event      = gating_event,
        .layer_idx       = p.mtp_layer_idx,
        .checkpoint_type = static_cast<uint8_t>(ipc::CheckpointType::kGatingOutput),
        .host_buf_offset = gating_offset,
        .data_bytes      = gating_bytes,
    });
    pending_compute_.push_back(std::move(pc));
}

// ── KD-3c: Self-spec pipeline ────────────────────────────────────────────────

void CommandDispatcher::run_self_spec_pipeline(const ipc::Command& cmd) {
    const auto& p = cmd.self_spec_forward;
    const uint32_t gpu = cmd.gpu_idx;
    const auto& mc = deps_.live_config->model;
    // V4-5b mHC: the self-spec/MTP pipeline's hidden buffers and layer loop
    // are single-stream (GLM/V3.2). V4 speculation is the embedded-dspark
    // path (ticket J) — fail loud rather than mis-execute (TD-V4-SELFSPEC).
    if (deps_.hc_streams > 1) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kAdapterForward,
                    "run_self_spec_pipeline: unsupported with mHC residual "
                    "streams — V4 speculation is the embedded-dspark path "
                    "(TD-V4-SELFSPEC)");
        return;
    }
    auto& ss = spec_scratch_[gpu];
    const uint64_t skip_lo = p.skip_mask_lo;
    const uint64_t skip_hi = p.skip_mask_hi;
    const uint32_t num_layers = static_cast<uint32_t>(mc.num_hidden_layers);
    const uint32_t first_moe_layer = static_cast<uint32_t>(mc.first_k_dense_replace);
    const int hidden = mc.hidden_size;

    void* stream = nullptr;
    int pair_idx = -1;
    if (!setup_spec_pipeline(cmd.cmd_seq, gpu, p.input_token_id,
                             "self_spec", ss, stream, pair_idx))
        return;

    // Per-layer pipeline with checkpoints.
    auto* gpu_dev = deps_.device_backends[gpu];
    const size_t h_bytes = static_cast<size_t>(hidden) * 2;

    PendingCompute pc{};
    pc.cmd_seq  = cmd.cmd_seq;
    pc.gpu_idx  = gpu;
    pc.cmd_type = cmd.cmd_type;

    void* attn_buf = (pair_idx >= 0)
        ? deps_.hidden_state_pairs[pair_idx].attn_buf : nullptr;

    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        const bool skipped = (layer < 64)
            ? ((skip_lo >> layer) & 1u) != 0
            : (layer < 128) ? ((skip_hi >> (layer - 64)) & 1u) != 0 : false;
        if (skipped) continue;

        // Snapshot pre-layer hidden state for cos_sim.
        if (deps_.cuda_kernels_enabled && attn_buf && ss.hidden_b) {
            gpu_dev->memcpy_d2d_async(ss.hidden_b, attn_buf, h_bytes, stream);
        } else if (ss.hidden_b && ss.hidden_a) {
            std::memcpy(ss.hidden_b, ss.hidden_a, h_bytes);
        }

        // Attention + MoE via shared forward_one_layer.
        const bool is_moe_layer = (layer >= first_moe_layer);
        const int topk_ov = is_moe_layer
            ? static_cast<int>(p.draft_expert_count) : 1;
        const bool do_gating = is_moe_layer && (p.store_gating != 0);

        // TD-51b RESOLVED: abort the pipeline with CMP_ERROR on layer failure.
        if (!forward_one_layer({
                .layer_idx    = layer,
                .num_seqs     = 1,
                .gpu_idx      = gpu,
                .topk_override = topk_ov,
                .store_gating = do_gating,
            })) {
            write_error(cmd.cmd_seq, gpu,
                        last_internal_error_msg_
                            ? last_internal_error_cat_
                            : ipc::CmpErrorCategory::kComputeValidation,
                        last_internal_error_msg_
                            ? last_internal_error_msg_
                            : "self_spec: forward_one_layer failed (see log)");
            return;
        }

        // Copy MoE output back to hidden_a for cos_sim.
        if (deps_.cuda_kernels_enabled && attn_buf) {
            gpu_dev->memcpy_d2d_async(ss.hidden_a, attn_buf, h_bytes, stream);
        }

        // TD-74v: restore device context after forward_one_layer (may switch for TP>1).
        gpu_dev->set_device();

        // Cosine similarity between pre-layer (hidden_b) and post-layer (hidden_a).
        compute::launch_cosine_similarity(
            static_cast<float*>(ss.cos_sim_out),
            ss.hidden_b, ss.hidden_a, hidden, stream);

        // D2H: copy cos_sim to sideband.
        // TODO:DEBT TD-50q: no bounds validation on spec checkpoint sideband region (4KB limit)
        const uint32_t cos_offset = static_cast<uint32_t>(
            ipc::IpcLayout::kSpecCheckpointOff + layer * sizeof(float));
        if (deps_.cuda_kernels_enabled) {
            gpu_dev->memcpy_d2h_async(deps_.sideband_base + cos_offset,
                                      ss.cos_sim_out, sizeof(float), stream);
        } else {
            std::memcpy(deps_.sideband_base + cos_offset,
                        ss.cos_sim_out, sizeof(float));
        }

        void* sim_event = create_and_record_event(static_cast<int>(gpu), compute::StreamId::kExpertFfn);

        pc.pipeline_checkpoints.push_back({
            .cuda_event      = sim_event,
            .layer_idx       = layer,
            .checkpoint_type = static_cast<uint8_t>(ipc::CheckpointType::kLayerSimilarity),
            .host_buf_offset = cos_offset,
            .data_bytes      = 4,
        });

        // Gating checkpoint for MoE layers when store_gating enabled.
        if (p.store_gating && is_moe_layer) {
            const auto& scratch = moe_scratch_[gpu];
            const int topk = (p.draft_expert_count > 0)
                ? static_cast<int>(p.draft_expert_count)
                : mc.num_experts_per_tok;
            const uint32_t gating_offset = static_cast<uint32_t>(
                ipc::IpcLayout::kSpecCheckpointOff + 512
                + layer * static_cast<uint32_t>(topk) * sizeof(float));
            const uint32_t gating_bytes = static_cast<uint32_t>(topk * sizeof(float));
            if (deps_.cuda_kernels_enabled) {
                gpu_dev->memcpy_d2h_async(deps_.sideband_base + gating_offset,
                                          scratch.topk_weights, gating_bytes,
                                          stream);
            } else {
                std::memcpy(deps_.sideband_base + gating_offset,
                            scratch.topk_weights, gating_bytes);
            }

            void* gating_event = create_and_record_event(static_cast<int>(gpu), compute::StreamId::kExpertFfn);

            pc.pipeline_checkpoints.push_back({
                .cuda_event      = gating_event,
                .layer_idx       = layer,
                .checkpoint_type = static_cast<uint8_t>(ipc::CheckpointType::kGatingOutput),
                .host_buf_offset = gating_offset,
                .data_bytes      = gating_bytes,
            });
        }
    }

    // Step 3: Output head + confidence + sampling (KD-R5).
    auto* readback = deps_.sideband_base
        + ipc::IpcLayout::kSpecCheckpointOff + 2560;
    dispatch_output_head({
        .gpu_idx            = gpu,
        .input              = ss.hidden_a,
        .logits_out         = static_cast<float*>(ss.logits),
        .stream             = stream,
        .stream_id          = compute::StreamId::kExpertFfn,
        .compute_confidence = true,
        .do_sample          = true,
        .readback_host_dst  = readback,
    });

    // Final event.
    void* final_event = create_and_record_event(static_cast<int>(gpu), compute::StreamId::kExpertFfn);

    pc.layer_idx  = num_layers - 1;
    pc.cuda_event = final_event;
    pc.data_bytes = 8;
    pc.host_buf_offset = static_cast<uint32_t>(
        ipc::IpcLayout::kSpecCheckpointOff + 2560);
    pending_compute_.push_back(std::move(pc));
}

}  // namespace layerstorm::daemon

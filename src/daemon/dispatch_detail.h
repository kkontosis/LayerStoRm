// Internal helpers for CommandDispatcher split files.
//
// Free-function utilities shared across dispatch_*.cpp translation units.
// Do NOT include from modules outside the CommandDispatcher implementation.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <spdlog/spdlog.h>

#include "compute/stream_manager.h"
#include "config/config_parser.h"          // GatingScoreFn (V4-4a)
#include "core/device_backend.h"
#include "core/expert_device.h"
#include "core/memory/eviction_policy.h"   // ExpertKey, CacheZone, SubComponent
#include "sm120/gating/topk_gating.h"      // ScoringFunc (V4-4a)
#include "smxx/quant/dynamic_fp8_quant.h"
#include "smxx/quant/bf16_to_nvfp4_grouped.h"
#include "smxx/quant/silu_mul_to_nvfp4_grouped.h"

#include <stdexcept>

namespace layerstorm::daemon {

// ── Shared type conversion helpers ────────────────────────────────────────
// Migrated from anonymous namespace so all split files can use them.

inline memory::ExpertKey make_key(uint32_t layer, uint16_t expert) {
    return {layer, expert};
}

// ── Gating scoring-function mapping (V4-4a) ───────────────────────────────
// config gating_score_fn → kernel ScoringFunc for launch_topk_gating.
// `softmax` has NO kernel implementation anywhere and is rejected at config
// validation (fail-closed); the throw keeps the dispatcher fail-loud should
// a config bypass validation.
inline compute::ScoringFunc to_scoring_func(config::GatingScoreFn fn) {
    switch (fn) {
        case config::GatingScoreFn::sigmoid:
            return compute::ScoringFunc::kSigmoid;
        case config::GatingScoreFn::sqrtsoftplus:
            return compute::ScoringFunc::kSqrtSoftplus;
        case config::GatingScoreFn::softmax:
            break;
    }
    throw std::runtime_error(
        "gating_score_fn=softmax has no gating-kernel ScoringFunc mapping "
        "(config validation should have rejected it)");
}

// ── MoE-BIG x-ray + wave-mask switches (read once) ────────────────────────

/// LS_MOE_BIG_XRAY (default OFF): clean per-command phase decomposition for
/// the FETCH_AND_RUN_MOE(_BIG) progressive machine — fetch-issue / fetch-WAIT
/// / wave-GPU / drain — logged once per command at completion reap. The
/// perf_trace markers pair garbage for the BIG path (cmd_seq collisions), so
/// this is the authoritative BIG-path timing (spec/bloat/PREFILL_FETCH_OVERHEAD.md §3).
inline bool moe_big_xray_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("LS_MOE_BIG_XRAY");
        return e && e[0] && e[0] != '0';
    }();
    return v;
}

/// LS_MOE_WAVE_MASK (default ON, "0" disables): wave-masked permute for the
/// chunked (BIG) rolling-wave passes — non-wave experts' top-K entries are
/// masked to the permute drop sentinel so each wave's grouped GEMMs cover only
/// its own experts' rows instead of the full-batch zero-weight sweep
/// (TD-MOE-BIG-GEMM-SWEEP). Kill-switch for A/B: bitwise-identical output.
inline bool moe_wave_mask_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("LS_MOE_WAVE_MASK");
        return !(e && e[0] == '0');
    }();
    return v;
}

inline uint64_t xray_now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline memory::CacheZone to_zone(uint8_t z) {
    return z == 0 ? memory::CacheZone::kStable : memory::CacheZone::kStreaming;
}

inline compute::StreamId to_stream(uint32_t s) {
    if (s >= static_cast<uint32_t>(compute::StreamId::kCount)) {
        spdlog::warn("CommandDispatcher: invalid stream_id {}, clamping to 0", s);
        return compute::StreamId::kAttention;
    }
    return static_cast<compute::StreamId>(s);
}

// ── Activation quantization helper ────────────────────────────────────────
// Replaces 6 identical FP8/NVFP4 quant branches in MoE dispatch.

struct ActivationQuantArgs {
    bool use_fp8;
    int num_tokens;
    int hidden_size;             // K dimension
    const void* input;
    void* quant_act;             // output: quantized activations
    void* quant_scale;           // output: quantization scales
    size_t quant_scale_bytes;    // for memset in NVFP4 path
    const void* expert_offsets;  // [num_experts+1] INT32 (NVFP4 grouping)
    const void* sf_offsets;      // [num_experts+1] INT32 (NVFP4 only)
    int total_tokens;            // may differ from num_tokens for routed experts
    int num_experts;             // 1 for dense/shared, E for routed
    compute::DeviceBackend* gpu_dev;  // for memset_async
    void* stream;
    // NVFP4 only: [num_experts] device float per-expert calibrated activation
    // global scales (TRT-LLM input_scale; pack-normalized). The consuming
    // GEMM's alpha must include the SAME values (alpha = ws2 * is). nullptr
    // (and entries <= 0) behave as 1.0.
    const void* input_scales = nullptr;
};

inline void launch_activation_quant(const ActivationQuantArgs& a) {
    if (a.use_fp8) {
        compute::DynamicFp8QuantParams qp{};
        qp.num_tokens  = a.num_tokens;
        qp.hidden_size = a.hidden_size;
        qp.input       = a.input;
        qp.output      = a.quant_act;
        qp.scales      = a.quant_scale;
        compute::launch_dynamic_fp8_quant(qp, a.stream);
    } else {
        a.gpu_dev->memset_async(a.quant_scale, 0, a.quant_scale_bytes, a.stream);
        compute::Bf16ToNvfp4GroupedParams qp{};
        qp.input          = a.input;
        qp.output_packed  = a.quant_act;
        qp.output_scales  = a.quant_scale;
        qp.expert_offsets = a.expert_offsets;
        qp.sf_offsets     = a.sf_offsets;
        qp.total_tokens   = a.total_tokens;
        qp.num_experts    = a.num_experts;
        qp.K              = a.hidden_size;
        qp.input_scales   = a.input_scales;
        compute::launch_bf16_to_nvfp4_grouped(qp, a.stream);
    }
}

// ── Grouped GEMM helper ──────────────────────────────────────────────────
// Replaces 6 identical FP8/NVFP4 GEMM branches in MoE dispatch.

struct GroupedGemmArgs {
    bool use_fp8;
    int num_experts;
    int N;                         // output dimension
    int K;                         // input dimension
    const void* A_base;            // quantized activations (BF16 for GGUF)
    const void* B_base;            // weights (nullptr for routed — filled by cache)
    void* D_base;                  // output
    const void* scale_A_base;      // activation scales
    const void* scale_B_base;      // weight scales
    const float* alphas;           // NVFP4 per-expert output scale (nullptr for FP8)
    const int32_t* expert_offsets;
    const int32_t* sf_offsets;     // NVFP4 only (nullptr for FP8)
    const int32_t* problem_sizes;
    compute::ExpertDevice* dev;
    void* gemm_workspace;
    size_t gemm_workspace_bytes;
    void* stream;

    // Optional per-expert pointer arrays (for scattered ExpertCache slots).
    // When non-null, override B_base/scale_B_base in the GEMM params.
    const void** B_ptrs = nullptr;
    const void** scale_B_ptrs = nullptr;

    // GG-5b: GGUF path. When use_gguf is true, this builds GgufGroupedGemmParams
    // and calls dev->gguf_grouped_gemm instead of the FP8/NVFP4 GEMM. GGUF feeds
    // the BF16 activation directly as A_base (the int kernel self-quantizes to
    // Q8_1; dequant needs none) — scale_A_base / scale_B_* / alphas / sf_offsets /
    // problem_sizes are all unused. The packed weight blocks come from B_ptrs (a
    // device [num_experts] array; for dense/shared a 1-element array). There is NO
    // separate weight-scale region (GGUF scales are intra-block), so scale_B_ptrs
    // is dropped. use_gguf takes precedence over use_fp8.
    bool use_gguf = false;
    compute::GgufQuantType gguf_type = compute::GgufQuantType::Q4_K;
    compute::GgufGemmStrategy gguf_strategy =
        compute::GgufGemmStrategy::int_strategy;
    // GG-5d: permuted row count (== expert_offsets[num_experts]) for the
    // device-fused GGUF int kernel. Routed: expanded_tokens; dense/shared:
    // num_tokens. Set at each GGUF call site alongside use_gguf.
    int gguf_total_tokens = 0;
};

inline void launch_grouped_gemm(const GroupedGemmArgs& g) {
    if (g.use_gguf) {
        compute::GgufGroupedGemmParams p{};
        p.type           = g.gguf_type;
        p.strategy       = g.gguf_strategy;
        p.num_experts    = g.num_experts;
        p.N              = g.N;
        p.K              = g.K;
        p.A_base         = g.A_base;   // BF16 activations (no pre-quant)
        p.D_base         = g.D_base;
        p.expert_offsets = g.expert_offsets;
        p.B_ptrs         = g.B_ptrs;
        p.total_tokens   = g.gguf_total_tokens;
        g.dev->gguf_grouped_gemm(p, g.gemm_workspace,
                                 g.gemm_workspace_bytes, g.stream);
    } else if (g.use_fp8) {
        compute::Fp8GroupedGemmParams p{};
        p.num_experts   = g.num_experts;
        p.N             = g.N;
        p.K             = g.K;
        p.A_base        = g.A_base;
        p.B_base        = g.B_base;
        p.D_base        = g.D_base;
        p.scale_A_base  = g.scale_A_base;
        p.scale_B_base  = g.scale_B_base;
        p.expert_offsets = g.expert_offsets;
        p.problem_sizes  = g.problem_sizes;
        p.output_dtype   = compute::GemmOutputDtype::kBFloat16;
        p.B_ptrs         = g.B_ptrs;
        p.scale_B_ptrs   = g.scale_B_ptrs;
        g.dev->fp8_grouped_gemm(p, g.gemm_workspace,
                                g.gemm_workspace_bytes, g.stream);
    } else {
        compute::Nvfp4GroupedGemmParams p{};
        p.num_experts   = g.num_experts;
        p.N             = g.N;
        p.K             = g.K;
        p.A_base        = g.A_base;
        p.B_base        = g.B_base;
        p.D_base        = g.D_base;
        p.scale_A_base  = g.scale_A_base;
        p.scale_B_base  = g.scale_B_base;
        p.alphas        = g.alphas;
        p.expert_offsets = g.expert_offsets;
        p.sf_offsets     = g.sf_offsets;
        p.problem_sizes  = g.problem_sizes;
        p.output_dtype   = compute::GemmOutputDtype::kBFloat16;
        p.B_ptrs         = g.B_ptrs;
        p.scale_B_ptrs   = g.scale_B_ptrs;
        g.dev->nvfp4_grouped_gemm(p, g.gemm_workspace,
                                  g.gemm_workspace_bytes, g.stream);
    }
}

// ── GG-S1: MoeGemmEmitter — single home for the routed grouped-GEMM pick ─────
// One object owns the "which kernel, built how" decision for the routed MoE FFN
// grouped GEMMs (gate / up / down), so the fp8 / nvfp4 / gguf param-build lives
// in exactly ONE place instead of being re-scattered at every projection site.
//
// It derives the GGUF permuted-row count (`total_tokens`, == expert_offsets
// [num_experts]) EXACTLY ONCE at construction and stamps it into every GGUF
// GEMM it emits — this kills the per-site `gguf_total_tokens = …` field that
// TD-GG5D-GGUF-TOTAL-TOKENS-SILENT-ZERO warns is easy to forget (a forgotten
// 0 silently degrades the int kernel). Constructed once per dispatch from the
// per-call invariants; `routed_gemm()` is a flat, allocation-free emit (no
// virtual, no heap) so the decode token×layer hot loop pays nothing.
//
// CUDA-free (INV-GPU-1): only forwards to launch_grouped_gemm, which forwards
// to the device backend. Lives on the dispatch (no-CUDA) side.
struct MoeGemmEmitter {
    compute::ExpertDevice* dev = nullptr;
    void*  gemm_workspace = nullptr;
    size_t gemm_workspace_bytes = 0;
    void*  stream = nullptr;
    int    num_experts = 0;
    bool   use_fp8 = false;
    bool   use_gguf = false;
    compute::GgufGemmStrategy gguf_strategy =
        compute::GgufGemmStrategy::int_strategy;
    int    total_tokens = 0;   // GGUF permuted row count, derived ONCE
    const int32_t* expert_offsets = nullptr;
    const int32_t* sf_offsets = nullptr;
    const int32_t* problem_sizes = nullptr;

    // Emit one routed projection grouped GEMM.
    //  - GGUF : A is the BF16 activation; `gguf_type` selects the k-quant;
    //           scale_A / alphas / sb_ptrs are unused (intra-block scales).
    //  - NVFP4/FP8: A is the quantized activation; scale_A = activation scales,
    //           alphas = per-expert output scale (NVFP4), sb_ptrs = scale-B.
    void routed_gemm(int N, int K, const void* A, void* D,
                     const void** b_ptrs, const void** sb_ptrs,
                     const void* scale_A, const float* alphas,
                     compute::GgufQuantType gguf_type) const {
        GroupedGemmArgs g{use_fp8, num_experts, N, K, A, nullptr, D,
            scale_A, nullptr, alphas, expert_offsets, sf_offsets, problem_sizes,
            dev, gemm_workspace, gemm_workspace_bytes, stream,
            b_ptrs, use_gguf ? nullptr : sb_ptrs};
        if (use_gguf) {
            g.use_gguf = true;
            g.gguf_type = gguf_type;
            g.gguf_strategy = gguf_strategy;
            g.gguf_total_tokens = total_tokens;
        }
        launch_grouped_gemm(g);
    }

    // ── GG-S1 RESERVATION (design-only; do NOT implement here) ───────────────
    // AttnSequence is the planned sibling of MoeSequence over this SAME emitter.
    // Attention does a per-projection SINGLE GEMM (num_experts==1, M = batch rows)
    // whose GGUF strategy×M pick is exactly dcp_executor::route_gguf_gemm:
    //     int + M<=8 -> mmvq ; int + M>8 -> mmq ; dequant -> gguf_dequant_gemm.
    // That is the SAME "which kernel, built how" decision routed_gemm() makes for
    // the grouped case — it belongs on this emitter as a future `single_gemm(M,
    // …)` method, leaving dcp_executor a pure orchestrator and GG-4's three
    // gguf_* AttentionDevice virtuals merely relocated (decision: option a). The
    // shared runner's warmup() hook already covers the only attention-touching
    // debt (mmq capture warmup). This struct is intentionally general enough to
    // admit that method without reshaping; AttnSequence earns its keep only once
    // a concrete attention-decode-graph driver exists, so it is NOT built now.
};

// ── GGUF dense/shared gate_up stage (GG-5c) ─────────────────────────────────
// Dense-FFN and shared-expert FFN concatenate their own `ffn_gate`‖`ffn_up`
// weights (gate-block then up-block, contiguous) and normally run them as one
// fused `[2*I, H]` GGUF GEMM. That single GEMM carries ONE k-quant type, so it
// is only valid when this unit's gate and up tensors share the same type. On a
// mixed-precision GGUF they can differ (e.g. gate Q4_K, up Q6_K) — then we SPLIT
// into two single-type `[I, H]` GEMMs: gate at the weight base (gate_type) and
// up at base + up_block_offset (up_type, the up block is separately addressable,
// just concatenated after gate). The two `[., I]` outputs are interleaved into
// the `[., 2*I]` gate_up buffer (mirroring the routed split's 2× memcpy_2d) so
// the caller's plain BF16 SwiGLU consumes the same layout the fused path emits.
// Returns 1 (fused) or 2 (split) — the number of GEMMs launched.
struct GgufDenseGateUpArgs {
    int num_tokens;            // M (rows)
    int intermediate_local;    // I — per-projection N (the gate/up half width)
    int hidden;                // K
    const void* a_base;        // BF16 input (norm_input), fed directly to the int kernel
    void* gate_up_output;      // [M, 2*I] BF16 — fused GEMM output / split interleave dest
    void* gate_scratch;        // [M, I]   BF16 — split gate GEMM output
    void* up_scratch;          // [M, >=I] BF16 — split up GEMM output (sized for I)
    compute::GgufQuantType gate_type;
    compute::GgufQuantType up_type;
    int64_t up_block_offset;   // split only: gguf_packed_bytes(I, H, gate_type)
    compute::GgufGemmStrategy strategy;
    const void* gate_up_weight;     // device weight base pointer (gate ‖ up); its VALUE is known host-side and staged H2D into single_b_ptr, then dereferenced as B_ptrs[0] on-device
    void* single_b_ptr;             // device [1] void* — the GEMM's B_ptrs array
    const int32_t* expert_offsets;  // device [2] {0, num_tokens}
    compute::ExpertDevice* dev;
    compute::DeviceBackend* gpu_dev;
    void* gemm_workspace;
    size_t gemm_workspace_bytes;
    void* stream;
};

inline int launch_gguf_dense_gate_up(const GgufDenseGateUpArgs& a) {
    // Bind the B pointer through a local staging slot (pageable H2D copies
    // synchronously, so reusing one slot across the gate/up rebinds is safe —
    // the same synchronicity the surrounding single_b_ptr binds rely on) and
    // launch one single-expert GGUF GEMM of output width N into `d`.
    const void* b_staging = nullptr;
    auto emit = [&](const void* b_base, int N, compute::GgufQuantType t, void* d) {
        b_staging = b_base;
        a.gpu_dev->memcpy_h2d_async(a.single_b_ptr, &b_staging,
                                    sizeof(void*), a.stream);
        GroupedGemmArgs g{/*use_fp8=*/false, /*num_experts=*/1, N, a.hidden,
            a.a_base, /*B_base=*/nullptr, d,
            /*scale_A_base=*/nullptr, /*scale_B_base=*/nullptr, /*alphas=*/nullptr,
            a.expert_offsets, /*sf_offsets=*/nullptr, /*problem_sizes=*/nullptr,
            a.dev, a.gemm_workspace, a.gemm_workspace_bytes, a.stream};
        g.use_gguf = true;
        g.gguf_type = t;
        g.gguf_strategy = a.strategy;
        g.gguf_total_tokens = a.num_tokens;  // dense/shared: 1 expert, M rows
        g.B_ptrs = static_cast<const void**>(a.single_b_ptr);
        launch_grouped_gemm(g);
    };

    if (a.gate_type == a.up_type) {
        // Fused (common, fast): one [2*I, H] GEMM with the shared type.
        emit(a.gate_up_weight, 2 * a.intermediate_local, a.gate_type,
             a.gate_up_output);
        return 1;
    }

    // Split: gate (base, gate_type) → gate_scratch; up (base+offset, up_type) →
    // up_scratch; then interleave both halves into gate_up_output [., 2*I].
    // GG-S1 Phase 4 (TD-GG5C-SPLIT-BUFFER-ALWAYS-ALLOC): up_scratch
    // (MoeScratch::gguf_gate_up_split) is only allocated when the load-time
    // gate≠up scan flagged a split unit. Reaching here with a null up_scratch
    // means the scan missed this unit — defensive guard against the GG-9
    // uninit-tail class. Should be impossible once the scan is correct.
    if (a.up_scratch == nullptr) {
        spdlog::error("launch_gguf_dense_gate_up: SPLIT path (gate_type != "
                      "up_type) hit with null up_scratch — gguf_gate_up_split "
                      "was gated off but a gate≠up unit fired (scan bug). "
                      "Skipping up GEMM to avoid OOB write.");
        return 1;
    }
    emit(a.gate_up_weight, a.intermediate_local, a.gate_type, a.gate_scratch);
    emit(static_cast<const std::byte*>(a.gate_up_weight) + a.up_block_offset,
         a.intermediate_local, a.up_type, a.up_scratch);
    if (a.num_tokens > 0) {
        const size_t I_bytes = static_cast<size_t>(a.intermediate_local) * 2;
        a.gpu_dev->memcpy_2d_async(a.gate_up_output, I_bytes * 2,
                                   a.gate_scratch, I_bytes,
                                   I_bytes, a.num_tokens, a.stream);
        a.gpu_dev->memcpy_2d_async(
            static_cast<std::byte*>(a.gate_up_output) + I_bytes, I_bytes * 2,
            a.up_scratch, I_bytes, I_bytes, a.num_tokens, a.stream);
    }
    return 2;
}

// ── SwiGLU helper ─────────────────────────────────────────────────────────
// Replaces 3 identical SwiGLU launch blocks.

inline void launch_swiglu(compute::ExpertDevice* dev,
                          void* output, const void* input,
                          int num_tokens, int d,
                          void* stream, float swiglu_limit) {
    compute::FusedSwigluParams sp{};
    sp.num_tokens   = num_tokens;
    sp.d            = d;
    // V4-4b: llama.cpp DEEPSEEK4 clamp — SiLU(min(g,L)) * clamp(up,-L,L).
    // 0.0 = off (V3.2/GLM byte-identical legacy behavior).
    sp.swiglu_limit = swiglu_limit;
    dev->fused_swiglu(output, input, sp, /*elem_size_bytes=*/2, stream);
}

// ── Fused SwiGLU + NVFP4 grouped quant (NVFP4 routed/dense/shared path) ─────
// Reads the gate + up grouped-GEMM outputs directly and writes FP4 + Sm1xx
// scales for the down GEMM, replacing interleave(2x memcpy2D) + SwiGLU +
// bf16_to_nvfp4_grouped. Byte-identical to that sequence (GK proves it). The
// down GEMM's per-expert alphas must carry the SAME input_scales (alpha=ws2*is).
struct FusedSwigluNvfp4QuantArgs {
    const void* gate;            // [total_tokens, I] BF16 (gate GEMM output)
    const void* up;              // [total_tokens, I] BF16 (up GEMM output)
    void* quant_act;             // out: [total_tokens, I/2] packed FP4
    void* quant_scale;           // out: Sm1xx UE4M3 scales
    size_t quant_scale_bytes;    // for the zeroing memset
    const void* expert_offsets;  // [num_experts+1] INT32
    const void* sf_offsets;      // [num_experts+1] INT32
    int total_tokens;
    int num_experts;
    int intermediate;           // K = I
    const void* input_scales = nullptr;  // [num_experts] FLOAT32 down fc2 scales
    compute::DeviceBackend* gpu_dev;     // for the scale memset
    void* stream;
};

inline void launch_fused_swiglu_nvfp4_quant(const FusedSwigluNvfp4QuantArgs& a) {
    a.gpu_dev->memset_async(a.quant_scale, 0, a.quant_scale_bytes, a.stream);
    compute::SiluMulToNvfp4GroupedParams qp{};
    qp.gate           = a.gate;
    qp.up             = a.up;
    qp.output_packed  = a.quant_act;
    qp.output_scales  = a.quant_scale;
    qp.expert_offsets = a.expert_offsets;
    qp.sf_offsets     = a.sf_offsets;
    qp.total_tokens   = a.total_tokens;
    qp.num_experts    = a.num_experts;
    qp.K              = a.intermediate;
    qp.input_scales   = a.input_scales;
    compute::launch_silu_mul_to_nvfp4_grouped(qp, a.stream);
}

}  // namespace layerstorm::daemon

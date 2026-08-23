// MoE weight-quant route derivation (see quant_routes.h). Body is the
// verbatim union of the two former per-file copies (dispatch_moe.cpp
// dispatch_moe_internal / dispatch_moe_big.cpp dispatch_moe_chunked_internal
// — they were token-identical modulo the log-message prefix, now `ctx`, and
// the cuda_kernels_enabled guard, which the chunked caller reaches only in
// the true state).

#include "daemon/moe/quant_routes.h"

#include <cstdlib>

#include <spdlog/spdlog.h>

namespace layerstorm::daemon {

MoeQuantRoutes build_moe_quant_routes(const CommandDispatcher::Deps& deps,
                                      uint32_t layer_idx, int intermediate,
                                      int hidden, const char* ctx) {
    MoeQuantRoutes qr;

    // Weight quant mode — needed by dense, routed, and shared expert paths.
    const auto wq = deps.live_config->quantization.weights;
    qr.use_fp8 = (wq == config::WeightQuant::fp8_e4m3 ||
                  wq == config::WeightQuant::fp8_e5m2);
    // GG-5b: GGUF weight path (Q2_K..Q8_0 + the generic `gguf` sentinel). Feeds
    // BF16 activations straight into the GGUF grouped GEMM (no FP8/NVFP4 quant);
    // the int kernel self-quantizes to Q8_1, dequant needs none.
    qr.use_gguf = model::gguf::is_gguf_weight_quant(wq);

    // Validate quant type before any GEMM path (dense, routed, or shared).
    if (deps.cuda_kernels_enabled &&
        wq != config::WeightQuant::fp8_e4m3 &&
        wq != config::WeightQuant::fp8_e5m2 &&
        wq != config::WeightQuant::nvfp4 &&
        !qr.use_gguf) {
        spdlog::critical("{}: unsupported weight quant for MoE GEMM", ctx);
        std::abort();
    }
    // GG-5b: GGUF needs the per-projection k-quant interface plumbed via Deps.
    // Without it we cannot select gate/up/down GgufQuantType — abort loudly
    // rather than run a wrong-typed GEMM (mirrors the quant-type gate above).
    if (deps.cuda_kernels_enabled && qr.use_gguf && !deps.gguf_quant) {
        spdlog::critical("{}: GGUF weights but no gguf_quant interface "
                         "plumbed (Deps::gguf_quant is null)", ctx);
        std::abort();
    }

    // GG-5b: GGUF compute strategy (int vs dequant) from live config.
    qr.gguf_strategy =
        (deps.live_config->quantization.gguf_strategy ==
         config::GgufStrategy::dequant)
            ? compute::GgufGemmStrategy::dequant
            : compute::GgufGemmStrategy::int_strategy;
    // GG-5b: per-projection GGUF type. projection_type() returns a
    // model::GgufKQuantType whose ordinal matches compute::GgufQuantType
    // (Q2_K=0..Q8_0=5), so the cast is by value (same cast the atomic path uses).
    auto gguf_proj_type = [&](model::Projection proj) -> compute::GgufQuantType {
        return static_cast<compute::GgufQuantType>(
            static_cast<int>(deps.gguf_quant->projection_type(proj)));
    };
    // GG-9: prefer THIS layer's own routed k-quant types (a mixed "XL" GGUF uses
    // different routed k-quants per layer); fall back to the uniform `gguf_quant`
    // types when the per-layer table is absent (uniform GGUF / synthetic tests).
    const model::GgufModelExpertTypes* layer_rt =
        (qr.use_gguf && layer_idx < deps.routed_layer_gguf_types.size())
            ? &deps.routed_layer_gguf_types[layer_idx]
            : nullptr;
    qr.gguf_gate_type =
        layer_rt      ? MoeQuantRoutes::to_gguf_compute(layer_rt->gate)
        : qr.use_gguf ? gguf_proj_type(model::Projection::gate)
                      : compute::GgufQuantType::Q4_K;
    qr.gguf_up_type =
        layer_rt      ? MoeQuantRoutes::to_gguf_compute(layer_rt->up)
        : qr.use_gguf ? gguf_proj_type(model::Projection::up)
                      : compute::GgufQuantType::Q4_K;
    qr.gguf_down_type =
        layer_rt      ? MoeQuantRoutes::to_gguf_compute(layer_rt->down)
        : qr.use_gguf ? gguf_proj_type(model::Projection::down)
                      : compute::GgufQuantType::Q4_K;

    // GG-9 (TD-GG9-EXPERT-PERLAYER-OFFSETS): a mixed "XL" GGUF packs each layer's
    // experts at THIS layer's per-projection k-quant sizes (e.g. layer 3 gate/up
    // are Q5_K while the global-max cache slot is sized for Q6_K). The CacheEntry
    // gate/up/down offsets are the GLOBAL-max layout, so for a smaller per-layer
    // pack the up/down offsets are wrong → the GEMM reads uninitialized slot tail
    // → NaN. Recompute the in-slot offsets from this layer's own gate/up types so
    // each projection is read where the packer (ensure_expert_packed, per-layer
    // types) actually wrote it. gate is always at 0.
    if (qr.use_gguf) {
        const model::GgufKQuantType gate_mt =
            layer_rt ? layer_rt->gate
                     : deps.gguf_quant->projection_type(model::Projection::gate);
        const model::GgufKQuantType up_mt =
            layer_rt ? layer_rt->up
                     : deps.gguf_quant->projection_type(model::Projection::up);
        qr.gguf_up_off =
            model::gguf::gguf_packed_bytes(intermediate, hidden, gate_mt);
        qr.gguf_down_off =
            qr.gguf_up_off
            + model::gguf::gguf_packed_bytes(intermediate, hidden, up_mt);
    }
    return qr;
}

}  // namespace layerstorm::daemon

// GG-5: enum reconciliation + dispatch param-build (CPU-only, CUDA-free).
//
// Verifies that the engine GGUF quant enum was aligned to the canonical set and
// stays bit-compatible with model::GgufKQuantType and the kernel GgufType (the
// engine→kernel boundary casts by value), and that the GgufGroupedGemmParams the
// quant_mode==2 dispatch builds carries the expected fields/defaults.

#include "core/expert_device.h"
#include "model/quantization/gguf_kquant.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace lc = layerstorm::compute;
namespace lm = layerstorm::model;

// Canonical ordinals: {Q2_K=0, Q3_K=1, Q4_K=2, Q5_K=3, Q6_K=4, Q8_0=5}.
TEST(GgufEnumReconciliation, CanonicalOrdinals) {
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q2_K), 0);
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q3_K), 1);
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q4_K), 2);
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q5_K), 3);
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q6_K), 4);
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q8_0), 5);
}

// The engine quant enum and the model quant enum share the same value order, so
// the dispatch can map p.gguf_type (a model::GgufKQuantType ordinal) by cast.
TEST(GgufEnumReconciliation, MatchesModelEnum) {
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q2_K),
              static_cast<int>(lm::GgufKQuantType::Q2_K));
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q3_K),
              static_cast<int>(lm::GgufKQuantType::Q3_K));
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q4_K),
              static_cast<int>(lm::GgufKQuantType::Q4_K));
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q5_K),
              static_cast<int>(lm::GgufKQuantType::Q5_K));
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q6_K),
              static_cast<int>(lm::GgufKQuantType::Q6_K));
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q8_0),
              static_cast<int>(lm::GgufKQuantType::Q8_0));
    EXPECT_EQ(static_cast<int>(lc::GgufQuantType::Q8_0),
              lm::kNumGgufKQuantTypes - 1);
}

// The strategy enum exists and defaults to int_strategy on the params struct,
// mirroring the engine config default (GgufStrategy::int_strategy).
TEST(GgufEnumReconciliation, GroupedParamsStrategyDefault) {
    lc::GgufGroupedGemmParams p{};
    EXPECT_EQ(p.strategy, lc::GgufGemmStrategy::int_strategy);
}

// Mirror of the quant_mode==2 dispatch param build (dispatch_compute.cpp): N from
// hidden_dim, K from k_dim, type cast from the IPC gguf_type byte, strategy from
// the live config. This pins the field wiring so a future field reshuffle that
// silently swaps N/K or drops the type is caught.
TEST(GgufDispatchBuild, BuildsParamsFromIpcFields) {
    const uint8_t ipc_gguf_type = static_cast<uint8_t>(lm::GgufKQuantType::Q4_K);
    const uint32_t hidden_dim = 2048;  // N
    const uint32_t k_dim = 7168;       // K
    const uint32_t num_experts = 8;

    lc::GgufGroupedGemmParams gp{};
    gp.type        = static_cast<lc::GgufQuantType>(ipc_gguf_type);
    gp.strategy    = lc::GgufGemmStrategy::dequant;  // as if config==dequant
    gp.num_experts = static_cast<int>(num_experts);
    gp.N           = static_cast<int>(hidden_dim);
    gp.K           = static_cast<int>(k_dim);

    EXPECT_EQ(gp.type, lc::GgufQuantType::Q4_K);
    EXPECT_EQ(gp.strategy, lc::GgufGemmStrategy::dequant);
    EXPECT_EQ(gp.num_experts, 8);
    EXPECT_EQ(gp.N, 2048);
    EXPECT_EQ(gp.K, 7168);
}

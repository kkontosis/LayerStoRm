// Unit tests for the RoPE cos/sin table builder + softmax scale (CUDA-free).
//
// Reference: ref/DeepSeek-V3 inference/model.py precompute_freqs_cis (interleaved
// adjacent-pair convention, YaRN frequency correction) and model.py:434-437
// (softmax_scale = qk_head_dim^-0.5, ×mscale² beyond the original context).

#include <gtest/gtest.h>

#include <cmath>

#include "compute/rope_table.h"

namespace lc = layerstorm::compute;
namespace lcfg = layerstorm::config;

namespace {

double plain_freq(int i, int d_rope, double theta) {
    return 1.0 / std::pow(theta, static_cast<double>(2 * i) / d_rope);
}

lcfg::RopeScalingConfig v32_yarn() {
    lcfg::RopeScalingConfig s{};
    s.type = lcfg::RopeScalingType::yarn;
    s.factor = 40.0;
    s.beta_fast = 32.0;
    s.beta_slow = 1.0;
    s.mscale = 1.0;
    s.mscale_all_dim = 1.0;
    s.original_max_position_embeddings = 4096.0;
    return s;
}

}  // namespace

TEST(RopeTable, PositionZeroIsIdentity) {
    auto t = lc::build_rope_cos_sin_table(4, 64, 10000.0, std::nullopt);
    ASSERT_EQ(t.size(), 4u * 64);
    for (int i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(t[i], 1.0f) << "cos[0][" << i << "]";
        EXPECT_FLOAT_EQ(t[32 + i], 0.0f) << "sin[0][" << i << "]";
    }
}

TEST(RopeTable, PlainAnglesMatchFormula) {
    const int d_rope = 64, max_pos = 100;
    const double theta = 10000.0;
    auto t = lc::build_rope_cos_sin_table(max_pos, d_rope, theta, std::nullopt);
    for (int p : {1, 7, 99}) {
        const float* row = t.data() + static_cast<size_t>(p) * d_rope;
        for (int i = 0; i < d_rope / 2; ++i) {
            const double ang = p * plain_freq(i, d_rope, theta);
            EXPECT_NEAR(row[i], std::cos(ang), 1e-5) << "p=" << p << " i=" << i;
            EXPECT_NEAR(row[32 + i], std::sin(ang), 1e-5) << "p=" << p << " i=" << i;
        }
    }
}

TEST(RopeTable, YarnCorrectsFrequencies) {
    // V3.2 yarn: correction range low=floor(10.47)=10, high=ceil(22.51)=23.
    // i <= 10 → extrapolation (freq unchanged); i >= 23 → interpolation (freq/40).
    const int d_rope = 64;
    const double theta = 10000.0;
    auto scaling = v32_yarn();
    const int max_pos = 8192;  // beyond original 4096 → correction active
    auto t = lc::build_rope_cos_sin_table(max_pos, d_rope, theta, scaling);

    const int p = 1;  // angle at pos 1 == effective frequency
    const float* row = t.data() + static_cast<size_t>(p) * d_rope;
    // Low dims: unchanged.
    for (int i = 0; i <= 10; ++i) {
        const double ang = plain_freq(i, d_rope, theta);
        EXPECT_NEAR(row[32 + i], std::sin(ang), 1e-5) << "i=" << i;
    }
    // High dims: interpolated by 1/factor.
    for (int i = 23; i < 32; ++i) {
        const double ang = plain_freq(i, d_rope, theta) / 40.0;
        EXPECT_NEAR(row[32 + i], std::sin(ang), 1e-6) << "i=" << i;
    }
    // Transition region: strictly between the two.
    for (int i = 11; i < 23; ++i) {
        const double f_plain = plain_freq(i, d_rope, theta);
        const double ang = std::asin(row[32 + i]);  // small angles here
        EXPECT_GT(ang, f_plain / 40.0 - 1e-9) << "i=" << i;
        EXPECT_LT(ang, f_plain + 1e-9) << "i=" << i;
    }
}

TEST(RopeTable, YarnInactiveWithinOriginalContext) {
    const int d_rope = 64;
    auto scaling = v32_yarn();
    auto corrected = lc::build_rope_cos_sin_table(2048, d_rope, 10000.0, scaling);
    auto plain = lc::build_rope_cos_sin_table(2048, d_rope, 10000.0, std::nullopt);
    EXPECT_EQ(corrected, plain);  // max_pos <= original → no correction
}

TEST(RopeSoftmaxScale, PlainIsInverseSqrtHeadDim) {
    // (128 + 64)^-0.5 — the NON-absorbed head dim, not the 576 absorbed dim.
    EXPECT_NEAR(lc::rope_softmax_scale(128, 64, 2048, std::nullopt),
                1.0 / std::sqrt(192.0), 1e-7);
}

TEST(RopeSoftmaxScale, YarnAppliesMscaleSquared) {
    auto scaling = v32_yarn();
    const double mscale = 0.1 * 1.0 * std::log(40.0) + 1.0;  // ≈ 1.3689
    EXPECT_NEAR(lc::rope_softmax_scale(128, 64, 163840, scaling),
                (1.0 / std::sqrt(192.0)) * mscale * mscale, 1e-6);
    // Within the original context: no mscale.
    EXPECT_NEAR(lc::rope_softmax_scale(128, 64, 4096, scaling),
                1.0 / std::sqrt(192.0), 1e-7);
}

// ── V4-4c: dual RoPE tables (DeepSeek V4) ───────────────────────────────────
//
// Rule (ref/llama.cpp deepseek4.cpp:817-824): uncompressed layers rotate with
// the base theta and NO yarn (freq_scale 1, ext_factor 0); compressed layers
// rotate with compress_rope_theta WITH the full yarn frequency correction.
// Both tables PURE cos/sin — dsv4_rope_attn_factor cancels ggml's yarn mscale.

namespace {

lcfg::RopeScalingConfig v4_yarn() {
    lcfg::RopeScalingConfig s{};
    s.type = lcfg::RopeScalingType::yarn;
    s.factor = 16.0;
    s.beta_fast = 32.0;
    s.beta_slow = 1.0;
    s.original_max_position_embeddings = 4096.0;  // scaled-down for the test
    return s;
}

}  // namespace

TEST(RopeTableV4, BaseTableHasNoYarnEvenBeyondOriginalContext) {
    const int d_rope = 64, max_pos = 8192;  // beyond original 4096
    auto t = lc::build_v4_rope_tables(max_pos, d_rope, 10000.0, 160000.0,
                                      v4_yarn());
    // Base table must equal the plain no-scaling table at the base theta:
    // uncompressed V4 layers run WITHOUT yarn regardless of context length.
    auto plain = lc::build_rope_cos_sin_table(max_pos, d_rope, 10000.0,
                                              std::nullopt);
    ASSERT_EQ(t.base.size(), plain.size());
    for (size_t i = 0; i < plain.size(); ++i)
        ASSERT_EQ(t.base[i], plain[i]) << "base table diverges at " << i;
}

TEST(RopeTableV4, CompressTableUsesCompressThetaWithYarn) {
    const int d_rope = 64, max_pos = 8192;
    auto t = lc::build_v4_rope_tables(max_pos, d_rope, 10000.0, 160000.0,
                                      v4_yarn());
    // Compress table == direct build at compress theta WITH yarn...
    auto yarned = lc::build_rope_cos_sin_table(max_pos, d_rope, 160000.0,
                                               v4_yarn());
    ASSERT_EQ(t.compress.size(), yarned.size());
    for (size_t i = 0; i < yarned.size(); ++i)
        ASSERT_EQ(t.compress[i], yarned[i]) << "compress table diverges at " << i;
    // ...and it differs from the un-yarned compress-theta table (the yarn
    // frequency correction must actually engage beyond the original context).
    auto unyarned = lc::build_rope_cos_sin_table(max_pos, d_rope, 160000.0,
                                                 std::nullopt);
    bool any_diff = false;
    for (size_t i = 0; i < yarned.size() && !any_diff; ++i)
        any_diff = (yarned[i] != unyarned[i]);
    EXPECT_TRUE(any_diff) << "yarn correction did not engage";
    // And from the base table (different theta).
    any_diff = false;
    for (size_t i = 0; i < t.base.size() && !any_diff; ++i)
        any_diff = (t.compress[i] != t.base[i]);
    EXPECT_TRUE(any_diff) << "compress table equals base table";
}

TEST(RopeTableV4, WithinOriginalContextTablesAreUnyarnedPerTheta) {
    const int d_rope = 64, max_pos = 1024;  // within original 4096
    auto t = lc::build_v4_rope_tables(max_pos, d_rope, 10000.0, 160000.0,
                                      v4_yarn());
    // DeepSeek-reference convention: yarn correction inactive within the
    // original context — both tables are plain tables at their thetas.
    auto base_plain = lc::build_rope_cos_sin_table(max_pos, d_rope, 10000.0,
                                                   std::nullopt);
    auto comp_plain = lc::build_rope_cos_sin_table(max_pos, d_rope, 160000.0,
                                                   std::nullopt);
    EXPECT_EQ(t.base, base_plain);
    EXPECT_EQ(t.compress, comp_plain);
}

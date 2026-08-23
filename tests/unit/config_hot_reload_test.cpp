#include <cstring>

#include <gtest/gtest.h>

#include "config/config_parser.h"

using namespace layerstorm::config;

// ── is_changeable ──────────────────────────────────────────────────────────

TEST(ConfigHotReload, ChangeableFieldsMarkedCorrectly) {
    EXPECT_TRUE(is_changeable(FieldId::kPrefetchPrescopeEnabled));
    EXPECT_TRUE(is_changeable(FieldId::kPrefetchPrescopeTopK));
    EXPECT_TRUE(is_changeable(FieldId::kPrefetchPrescopeScoreThreshold));
    EXPECT_TRUE(is_changeable(FieldId::kPrefetchPrescopePredictorEnabled));
    EXPECT_TRUE(is_changeable(FieldId::kInternalPrescopeLearningRate));
    EXPECT_TRUE(is_changeable(FieldId::kSpeculationEnabled));
    EXPECT_TRUE(is_changeable(FieldId::kSpeculationCalibrationMinAcceptanceRate));
    EXPECT_TRUE(is_changeable(FieldId::kMemoryExpertCacheStableZoneFraction));
}

TEST(ConfigHotReload, NonChangeableFieldsRejected) {
    EXPECT_FALSE(is_changeable(FieldId::kModelArchitecture));
    EXPECT_FALSE(is_changeable(FieldId::kModelHiddenSize));
    EXPECT_FALSE(is_changeable(FieldId::kInternalPrescopeHiddenSize));
    EXPECT_FALSE(is_changeable(FieldId::kInternalPrescopePcaDim));
    EXPECT_FALSE(is_changeable(FieldId::kPrefetchPrescopeLookaheadLayers));
}

// ── field_name ─────────────────────────────────────────────────────────────

TEST(ConfigHotReload, FieldNameReturnsPath) {
    EXPECT_STREQ(field_name(FieldId::kPrefetchPrescopeTopK),
                 "prefetch.prescope.top_k");
    EXPECT_STREQ(field_name(FieldId::kSpeculationEnabled),
                 "speculation.enabled");
    EXPECT_STREQ(field_name(FieldId::kInternalPrescopeLearningRate),
                 "_internal-prescope.learning_rate");
}

// ── apply_field_update ─────────────────────────────────────────────────────

TEST(ConfigHotReload, ApplyBoolUpdate) {
    Config cfg;
    EXPECT_TRUE(cfg.prefetch.prescope.enabled);
    EXPECT_TRUE(apply_field_update(cfg, FieldId::kPrefetchPrescopeEnabled,
                                   0, 0));
    EXPECT_FALSE(cfg.prefetch.prescope.enabled);
    EXPECT_TRUE(apply_field_update(cfg, FieldId::kPrefetchPrescopeEnabled,
                                   0, 1));
    EXPECT_TRUE(cfg.prefetch.prescope.enabled);
}

TEST(ConfigHotReload, ApplyIntUpdate) {
    Config cfg;
    EXPECT_EQ(cfg.prefetch.prescope.top_k, 8);
    EXPECT_TRUE(apply_field_update(cfg, FieldId::kPrefetchPrescopeTopK,
                                   1, 16));
    EXPECT_EQ(cfg.prefetch.prescope.top_k, 16);
}

TEST(ConfigHotReload, ApplyFloatUpdate) {
    Config cfg;
    EXPECT_DOUBLE_EQ(cfg.prefetch.prescope.score_threshold, 0.01);
    float new_val = 0.05f;
    uint32_t raw;
    std::memcpy(&raw, &new_val, sizeof(float));
    EXPECT_TRUE(apply_field_update(cfg, FieldId::kPrefetchPrescopeScoreThreshold,
                                   2, raw));
    EXPECT_NEAR(cfg.prefetch.prescope.score_threshold, 0.05, 1e-6);
}

TEST(ConfigHotReload, RejectNonChangeable) {
    Config cfg;
    cfg.model.hidden_size = 7168;
    EXPECT_FALSE(apply_field_update(cfg, FieldId::kModelHiddenSize,
                                    1, 4096));
    EXPECT_EQ(cfg.model.hidden_size, 7168);
}

TEST(ConfigHotReload, RejectTypeMismatch) {
    Config cfg;
    // top_k is int (type 1), sending bool (type 0)
    EXPECT_FALSE(apply_field_update(cfg, FieldId::kPrefetchPrescopeTopK,
                                    0, 1));
    EXPECT_EQ(cfg.prefetch.prescope.top_k, 8);
}

TEST(ConfigHotReload, RejectOutOfRangeFieldId) {
    Config cfg;
    auto bad_id = static_cast<FieldId>(9999);
    EXPECT_FALSE(apply_field_update(cfg, bad_id, 1, 42));
}

TEST(ConfigHotReload, PredictorFieldUpdate) {
    Config cfg;
    EXPECT_DOUBLE_EQ(cfg._internal_prescope.focal_loss_gamma, 2.0);
    float new_val = 1.5f;
    uint32_t raw;
    std::memcpy(&raw, &new_val, sizeof(float));
    EXPECT_TRUE(apply_field_update(
        cfg, FieldId::kInternalPrescopeFocalLossGamma, 2, raw));
    EXPECT_NEAR(cfg._internal_prescope.focal_loss_gamma, 1.5, 1e-6);
}

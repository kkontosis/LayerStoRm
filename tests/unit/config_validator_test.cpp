#include <gtest/gtest.h>

#include "config/config_validator.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/registry.h"

using namespace layerstorm::config;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a valid DeepSeek V3.2-like Config for use as a baseline.
/// Tests modify specific fields to trigger validation failures.
static Config valid_config() {
    Config cfg;

    // Model (DeepSeek V3.2-like)
    cfg.model.architecture = Architecture::deepseek_v3;
    cfg.model.weights_path = "/data/models/deepseek-v3.2/";
    cfg.model.weights_format = WeightsFormat::safetensors;
    cfg.model.num_hidden_layers = 61;
    cfg.model.hidden_size = 7168;
    cfg.model.num_attention_heads = 128;
    cfg.model.num_key_value_heads = 128;
    cfg.model.kv_lora_rank = 512;
    cfg.model.qk_rope_head_dim = 64;
    cfg.model.qk_nope_head_dim = 128;
    cfg.model.v_head_dim = 128;
    cfg.model.q_lora_rank = 1536;
    cfg.model.intermediate_size = 18432;
    cfg.model.n_routed_experts = 256;
    cfg.model.n_shared_experts = 1;
    cfg.model.num_experts_per_tok = 8;
    cfg.model.n_group = 8;
    cfg.model.topk_group = 4;
    cfg.model.vocab_size = 129280;
    cfg.model.max_position_embeddings = 163840;
    cfg.model.rope_theta = 10000.0;
    cfg.model.rms_norm_eps = 1e-6;
    cfg.model.first_k_dense_replace = 3;
    cfg.model.moe_intermediate_size = 2048;
    cfg.model.moe_layer_freq = 1;

    // Quantization
    cfg.quantization.weights = WeightQuant::nvfp4;
    cfg.quantization.attention_compute = AttentionQuant::fp8_e4m3;
    cfg.quantization.kv_cache = KvCacheQuant::fp8_e4m3;
    cfg.quantization.gating_compute = GatingQuant::fp32;

    // Hardware: 2x5090 + 2x5080
    cfg.hardware.gpus = {
        {0, GpuType::rtx5090, 32, 5, 16, 0, 0.85, std::nullopt, 0,
         {GpuRole::attention, GpuRole::resident, GpuRole::expert_streaming}},
        {1, GpuType::rtx5090, 32, 5, 16, 0, 0.85, std::nullopt, 0,
         {GpuRole::attention, GpuRole::resident, GpuRole::expert_streaming}},
        {2, GpuType::rtx5080, 16, 5, 16, 1, 0.85, std::nullopt, 0,
         {GpuRole::expert_streaming}},
        {3, GpuType::rtx5080, 16, 5, 16, 1, 0.85, std::nullopt, 0,
         {GpuRole::expert_streaming}},
    };
    cfg.hardware.tp_array = {0, 1};
    cfg.hardware.system_ram_gb = 256;

    // Memory
    cfg.memory.pinned_layers.attention = std::string{"all"};
    cfg.memory.pinned_layers.dense_ffn_layers = {0, 1, 2};
    cfg.memory.pinned_layers.embedding = true;
    cfg.memory.pinned_layers.output_head = true;
    cfg.memory.pinned_layers.gating = std::string{"all"};
    cfg.memory.tp_mode_per_layer.default_mode = 1;
    cfg.memory.tp_mode_per_layer.gating = 2;
    cfg.memory.tp_mode_per_layer.pinned_dense_ffn = 2;
    cfg.memory.tp_mode_per_layer.attention = 1;
    cfg.memory.kv_cache.max_pages_per_gpu = std::string{"auto"};
    cfg.memory.kv_cache.page_size_tokens = 16;
    cfg.memory.kv_cache.speculation_pool_fraction = 0.15;

    // Expert cache (defaults are fine)
    // Orchestrator (defaults are fine)
    // Prefetch (defaults are fine)
    // Speculation (defaults are fine)
    // Compute (defaults are fine)
    // Serving (defaults are fine)

    return cfg;
}

static bool has_error(const ValidationResult& r, const std::string& field_substr) {
    for (const auto& e : r.errors) {
        if (e.field.find(field_substr) != std::string::npos)
            return true;
    }
    return false;
}

static bool has_warning(const ValidationResult& r, const std::string& field_substr) {
    for (const auto& w : r.warnings) {
        if (w.field.find(field_substr) != std::string::npos)
            return true;
    }
    return false;
}

// ── Valid config passes ─────────────────────────────────────────────────────

TEST(ConfigValidator, ValidConfigPasses) {
    auto cfg = valid_config();
    auto result = validate_config(cfg);
    EXPECT_TRUE(result.valid())
        << "Errors: " << (result.errors.empty() ? "(none)" : result.errors[0].field + ": " + result.errors[0].message);
}

// ── TP array validation ─────────────────────────────────────────────────────

TEST(ConfigValidator, RejectTpArrayOnNon5090) {
    auto cfg = valid_config();
    // Make both GPUs 5080
    cfg.hardware.gpus[0].type = GpuType::rtx5080;
    cfg.hardware.gpus[0].vram_gb = 16;
    cfg.hardware.gpus[1].type = GpuType::rtx5080;
    cfg.hardware.gpus[1].vram_gb = 16;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "tp_array"));
}

TEST(ConfigValidator, RejectTpArrayWithOnlyOneNon5090) {
    auto cfg = valid_config();
    cfg.hardware.gpus[1].type = GpuType::rtx5080;
    cfg.hardware.gpus[1].vram_gb = 16;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "tp_array[1]"));
}

TEST(ConfigValidator, RejectTpArrayWithNonexistentGpu) {
    auto cfg = valid_config();
    cfg.hardware.tp_array = {0, 99};

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "tp_array"));
}

TEST(ConfigValidator, RejectTpArraySameGpu) {
    auto cfg = valid_config();
    cfg.hardware.tp_array = {0, 0};

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "tp_array"));
}

TEST(ConfigValidator, RejectTpArrayNonPowerOf2) {
    auto cfg = valid_config();
    cfg.hardware.tp_array = {0, 1, 2};

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "tp_array"));
}

TEST(ConfigValidator, RejectTpModeWithoutTpArray) {
    auto cfg = valid_config();
    cfg.hardware.tp_array.clear();
    cfg.memory.tp_mode_per_layer.gating = 2;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "tp_mode_per_layer"));
}

TEST(ConfigValidator, NoTpArrayIsOkWhenAllTpModeIs1) {
    auto cfg = valid_config();
    cfg.hardware.tp_array.clear();
    cfg.memory.tp_mode_per_layer.default_mode = 1;
    cfg.memory.tp_mode_per_layer.gating = 1;
    cfg.memory.tp_mode_per_layer.pinned_dense_ffn = 1;
    cfg.memory.tp_mode_per_layer.attention = 1;

    auto result = validate_config(cfg);
    EXPECT_FALSE(has_error(result, "tp_mode_per_layer"));
}

// ── DCP validation ──────────────────────────────────────────────────────────

TEST(ConfigValidator, RejectDcpWithoutTpArray) {
    // dcp_enabled is auto-derived by resolver; simulate inconsistency
    auto cfg = valid_config();
    cfg.hardware.dcp_enabled = true;
    cfg.hardware.tp_array.clear();
    cfg.memory.tp_mode_per_layer.gating = 1;
    cfg.memory.tp_mode_per_layer.pinned_dense_ffn = 1;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "dcp_enabled"));
}

TEST(ConfigValidator, DcpWithTpArrayPasses) {
    auto cfg = valid_config();
    cfg.hardware.dcp_enabled = true;
    cfg.hardware.tp_array = {0, 1};

    auto result = validate_config(cfg);
    EXPECT_FALSE(has_error(result, "dcp_enabled"));
}

TEST(ConfigValidator, DcpDefaultTrueRoundTrip) {
    auto cfg = valid_config();
    // Schema default is true (matches tensor_parallelism=2).
    EXPECT_TRUE(cfg.hardware.dcp_enabled);
}

TEST(ConfigValidator, DcpIndexerModeLocalWarnsWhenDsaDisabled) {
    auto cfg = valid_config();
    cfg.hardware.dcp_indexer_mode = DcpIndexerMode::local;
    cfg.model.index_topk = 0;  // DSA disabled

    auto result = validate_config(cfg);
    EXPECT_TRUE(has_warning(result, "dcp_indexer_mode"));
}

TEST(ConfigValidator, DcpIndexerModeLocalNoWarnWhenDsaEnabled) {
    auto cfg = valid_config();
    cfg.hardware.dcp_indexer_mode = DcpIndexerMode::local;
    cfg.model.index_topk = 2048;  // DSA enabled

    auto result = validate_config(cfg);
    EXPECT_FALSE(has_warning(result, "dcp_indexer_mode"));
}

TEST(ConfigValidator, DcpIndexerModeReplicatedNoWarn) {
    auto cfg = valid_config();
    cfg.hardware.dcp_indexer_mode = DcpIndexerMode::replicated;
    cfg.model.index_topk = 0;  // DSA disabled

    auto result = validate_config(cfg);
    EXPECT_FALSE(has_warning(result, "dcp_indexer_mode"));
}

// ── NVMe validation ─────────────────────────────────────────────────────────

TEST(ConfigValidator, RejectNvmeWithoutPaths) {
    auto cfg = valid_config();
    cfg.memory.nvme_tier.enabled = true;
    cfg.hardware.nvme_paths = {};

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "nvme_paths"));
}

TEST(ConfigValidator, NvmeWithPathsIsValid) {
    auto cfg = valid_config();
    cfg.memory.nvme_tier.enabled = true;
    cfg.hardware.nvme_paths = {"/mnt/nvme0"};
    cfg.hardware.nvme_capacity_gb = 1000;

    auto result = validate_config(cfg);
    EXPECT_FALSE(has_error(result, "nvme_paths"));
}

TEST(ConfigValidator, NvmeMultiplePathsValid) {
    auto cfg = valid_config();
    cfg.memory.nvme_tier.enabled = true;
    cfg.hardware.nvme_paths = {"/mnt/nvme0", "/mnt/nvme1"};
    cfg.hardware.nvme_capacity_gb = 4000;

    auto result = validate_config(cfg);
    EXPECT_FALSE(has_error(result, "nvme_paths"));
}

TEST(ConfigValidator, WarnNvmePathsSetButDisabled) {
    auto cfg = valid_config();
    cfg.memory.nvme_tier.enabled = false;
    cfg.hardware.nvme_paths = {"/mnt/nvme0"};

    auto result = validate_config(cfg);
    EXPECT_TRUE(result.valid());  // warning, not error
    EXPECT_TRUE(has_warning(result, "nvme_tier"));
}

TEST(ConfigValidator, WarnNvmeEnabledWithoutCapacity) {
    auto cfg = valid_config();
    cfg.memory.nvme_tier.enabled = true;
    cfg.hardware.nvme_paths = {"/mnt/nvme0"};
    cfg.hardware.nvme_capacity_gb = std::nullopt;

    auto result = validate_config(cfg);
    EXPECT_TRUE(has_warning(result, "nvme_capacity_gb"));
}

// ── VRAM budget ─────────────────────────────────────────────────────────────

TEST(ConfigValidator, RejectInsufficientVram) {
    auto cfg = valid_config();
    // Single tiny GPU — clearly insufficient for DeepSeek V3.2
    cfg.hardware.gpus = {
        {0, GpuType::rtx5090, 2, 5, 16, 0, 0.85, std::nullopt, 0,
         {GpuRole::attention, GpuRole::resident, GpuRole::expert_streaming}},
    };
    cfg.hardware.tp_array.clear();
    cfg.memory.tp_mode_per_layer.gating = 1;
    cfg.memory.tp_mode_per_layer.pinned_dense_ffn = 1;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "hardware.gpus"));
}

TEST(ConfigValidator, SufficientVramPasses) {
    auto cfg = valid_config();
    // 4 GPUs with 96 GB total — reasonable for NVFP4 DeepSeek
    auto result = validate_config(cfg);
    EXPECT_FALSE(has_error(result, "hardware.gpus"));
}

TEST(ConfigValidator, RejectPinnedPlusExpertCacheOverVram) {
    auto cfg = valid_config();

    const auto& quant = layerstorm::model::get_format(cfg.quantization.weights);
    layerstorm::model::ModelConfig model_cfg(cfg);
    layerstorm::model::LayerRegistry reg(model_cfg, cfg, quant);
    auto budgets = reg.estimate_gpu_budgets();
    ASSERT_FALSE(budgets.empty());

    // Set TP GPU VRAM to exactly pinned + safety margin (no room for expert cache)
    double tight_vram_gb = static_cast<double>(budgets[0].pinned_bytes + 512LL * 1024 * 1024)
                           / (1024.0 * 1024 * 1024);
    for (auto& g : cfg.hardware.gpus)
        g.vram_gb = tight_vram_gb;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "hardware.gpus"));
}

// ── GPU config validation ───────────────────────────────────────────────────

TEST(ConfigValidator, RejectNoGpus) {
    auto cfg = valid_config();
    cfg.hardware.gpus.clear();
    cfg.hardware.tp_array.clear();
    cfg.memory.tp_mode_per_layer.gating = 1;
    cfg.memory.tp_mode_per_layer.pinned_dense_ffn = 1;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "hardware.gpus"));
}

TEST(ConfigValidator, RejectDuplicateGpuIds) {
    auto cfg = valid_config();
    cfg.hardware.gpus[1].id = 0;  // duplicate with gpus[0]

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, ".id"));
}

TEST(ConfigValidator, RejectNegativeVram) {
    auto cfg = valid_config();
    cfg.hardware.gpus[0].vram_gb = -1.0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "vram_gb"));
}

TEST(ConfigValidator, ZeroVramIsAutoDetectSentinel) {
    auto cfg = valid_config();
    cfg.hardware.gpus[0].vram_gb = 0.0;

    auto result = validate_config(cfg);
    // vram_gb=0 means auto-detect; budget check is skipped, no vram_gb error
    EXPECT_FALSE(has_error(result, "vram_gb"));
}

TEST(ConfigValidator, RejectNegativeSystemRam) {
    auto cfg = valid_config();
    cfg.hardware.system_ram_gb = -1;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "system_ram_gb"));
}

// ── Model dimension validation ──────────────────────────────────────────────

TEST(ConfigValidator, RejectZeroHiddenLayers) {
    auto cfg = valid_config();
    cfg.model.num_hidden_layers = 0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "num_hidden_layers"));
}

TEST(ConfigValidator, RejectZeroHiddenSize) {
    auto cfg = valid_config();
    cfg.model.hidden_size = 0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "hidden_size"));
}

TEST(ConfigValidator, RejectExpertsPerTokExceedsTotal) {
    auto cfg = valid_config();
    cfg.model.num_experts_per_tok = 300;  // > n_routed_experts (256)

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "num_experts_per_tok"));
}

TEST(ConfigValidator, RejectTopkGroupExceedsNGroup) {
    auto cfg = valid_config();
    cfg.model.topk_group = 10;
    cfg.model.n_group = 8;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "topk_group"));
}

TEST(ConfigValidator, RejectExpertsNotDivisibleByGroups) {
    auto cfg = valid_config();
    cfg.model.n_routed_experts = 255;  // not divisible by n_group=8

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "n_group"));
}

TEST(ConfigValidator, RejectFirstKDenseExceedsLayers) {
    auto cfg = valid_config();
    cfg.model.first_k_dense_replace = 100;  // > 61

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "first_k_dense_replace"));
}

TEST(ConfigValidator, RejectZeroMoeLayerFreq) {
    auto cfg = valid_config();
    cfg.model.moe_layer_freq = 0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "moe_layer_freq"));
}

// ── Pinned layer indices ────────────────────────────────────────────────────

TEST(ConfigValidator, RejectPinnedLayerOutOfBounds) {
    auto cfg = valid_config();
    cfg.memory.pinned_layers.dense_ffn_layers = {0, 1, 2, 99};  // 99 > 60

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "dense_ffn_layers"));
}

TEST(ConfigValidator, RejectPinnedAttentionLayerOutOfBounds) {
    auto cfg = valid_config();
    cfg.memory.pinned_layers.attention = std::vector<int>{0, 1, 70};

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "attention"));
}

// ── Eviction weights ────────────────────────────────────────────────────────

TEST(ConfigValidator, RejectAllZeroEvictionWeights) {
    auto cfg = valid_config();
    cfg.memory.expert_cache.eviction_alpha_recency = 0.0;
    cfg.memory.expert_cache.eviction_beta_frequency = 0.0;
    cfg.memory.expert_cache.eviction_gamma_routing_weight = 0.0;
    cfg.memory.expert_cache.eviction_delta_temporal_autocorr = 0.0;
    cfg.memory.expert_cache.eviction_epsilon_coactivation = 0.0;
    cfg.memory.expert_cache.eviction_zeta_prefetch_score = 0.0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "eviction_*"));
}

TEST(ConfigValidator, RejectNegativeEvictionWeight) {
    auto cfg = valid_config();
    cfg.memory.expert_cache.eviction_alpha_recency = -0.1;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "eviction_alpha_recency"));
}

// ── Performance objective ───────────────────────────────────────────────────

TEST(ConfigValidator, RejectNegativeLatencyWeight) {
    auto cfg = valid_config();
    cfg.orchestrator.performance_objective.latency_weight = -1.0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "latency_weight"));
}

TEST(ConfigValidator, RejectBothWeightsZero) {
    auto cfg = valid_config();
    cfg.orchestrator.performance_objective.latency_weight = 0.0;
    cfg.orchestrator.performance_objective.throughput_weight = 0.0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "performance_objective"));
}

// ── Speculation pool sizing ─────────────────────────────────────────────────

TEST(ConfigValidator, RejectInsufficientSpeculationPool) {
    auto cfg = valid_config();
    // Explicit page count with tiny pool
    cfg.memory.kv_cache.max_pages_per_gpu = 100;
    cfg.memory.kv_cache.speculation_pool_fraction = 0.01;  // 1 page
    cfg.serving.max_concurrent_requests = 32;
    cfg.speculation.enabled = true;
    cfg.speculation.mtp.enabled = true;
    cfg.speculation.mtp.max_depth = 5;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "kv_cache"));
}

TEST(ConfigValidator, SpeculationPoolAutoNotChecked) {
    auto cfg = valid_config();
    cfg.memory.kv_cache.max_pages_per_gpu = std::string{"auto"};
    cfg.speculation.enabled = true;

    auto result = validate_config(cfg);
    // "auto" pages skip pool size check — no error on kv_cache for pool sizing
    EXPECT_FALSE(has_error(result, "kv_cache"));
}

TEST(ConfigValidator, SpeculationDisabledSkipsPoolCheck) {
    auto cfg = valid_config();
    cfg.speculation.enabled = false;
    cfg.memory.kv_cache.max_pages_per_gpu = 10;
    cfg.memory.kv_cache.speculation_pool_fraction = 0.01;

    auto result = validate_config(cfg);
    EXPECT_FALSE(has_error(result, "kv_cache"));
}

TEST(ConfigValidator, RejectSpecPoolFractionOutOfRange) {
    auto cfg = valid_config();
    cfg.memory.kv_cache.speculation_pool_fraction = 1.5;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "speculation_pool_fraction"));
}

TEST(ConfigValidator, RejectZeroSpecPoolWhenSpecEnabled) {
    auto cfg = valid_config();
    cfg.speculation.enabled = true;
    cfg.memory.kv_cache.speculation_pool_fraction = 0.0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "speculation_pool_fraction"));
}

// ── CUDA graphs INV-0.6 ────────────────────────────────────────────────────
// capture_expert_ffn is now a const field (always false) in the schema,
// so it's not present in the struct and needs no runtime validation test.

// ── Serving config ──────────────────────────────────────────────────────────

TEST(ConfigValidator, RejectZeroConcurrentRequests) {
    auto cfg = valid_config();
    cfg.serving.max_concurrent_requests = 0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "max_concurrent_requests"));
}

TEST(ConfigValidator, RejectInvalidPort) {
    auto cfg = valid_config();
    cfg.serving.port = 0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "port"));
}

// ── Stable zone fraction ────────────────────────────────────────────────────

TEST(ConfigValidator, RejectStableZoneFractionOutOfRange) {
    auto cfg = valid_config();
    cfg.memory.expert_cache.stable_zone_fraction = 1.5;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "stable_zone_fraction"));
}

TEST(ConfigValidator, RejectPerGpuStableZoneFractionOutOfRange) {
    auto cfg = valid_config();
    cfg.hardware.gpus[0].vram_allocation_gb = VramAllocationConfig{};
    cfg.hardware.gpus[0].vram_allocation_gb->stable_zone_fraction = 1.5;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "stable_zone_fraction"));
}

// ── Warnings (non-fatal) ───────────────────────────────────────────────────

TEST(ConfigValidator, WarnUnusualPcieGen) {
    auto cfg = valid_config();
    cfg.hardware.gpus[0].pcie_gen = 2;

    auto result = validate_config(cfg);
    EXPECT_TRUE(has_warning(result, "pcie_gen"));
}

TEST(ConfigValidator, WarnAtypicalVram5090) {
    auto cfg = valid_config();
    cfg.hardware.gpus[0].vram_gb = 24;  // 5090 is typically 32

    auto result = validate_config(cfg);
    EXPECT_TRUE(has_warning(result, "vram_gb"));
}

// ── Multiple errors accumulated ─────────────────────────────────────────────

// ── TurboQuant attention backend validation ─────────────────────────────────

TEST(ConfigValidator, RejectTqWithKvLoraRankZero) {
    auto cfg = valid_config();
    cfg.compute.attention_backend = AttentionBackendType::turboquant_mla;
    cfg.model.kv_lora_rank = 0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "attention_backend"));
}

TEST(ConfigValidator, AcceptTqWithKvLoraRankPositive) {
    auto cfg = valid_config();
    cfg.compute.attention_backend = AttentionBackendType::turboquant_mla;
    // valid_config() has kv_lora_rank=512

    auto result = validate_config(cfg);
    EXPECT_FALSE(has_error(result, "attention_backend"));
}

// ── DSpark config contract (DSP-1) ──────────────────────────────────────────

TEST(ConfigValidator, DsparkStsTemperaturesLengthMustMatchBlockSize) {
    auto cfg = valid_config();
    cfg.speculation.method = SpeculationMethodType::dspark;
    cfg.speculation.dspark.block_size = 5;
    cfg.speculation.dspark.sts_temperatures = {1.0, 1.1, 0.9};  // 3 != 5

    auto result = validate_config(cfg);
    EXPECT_TRUE(has_error(result, "speculation.dspark.sts_temperatures"));

    // Empty = identity (no calibration) — always valid.
    cfg.speculation.dspark.sts_temperatures.clear();
    EXPECT_FALSE(has_error(validate_config(cfg),
                           "speculation.dspark.sts_temperatures"));

    // Exact γ-length — valid.
    cfg.speculation.dspark.sts_temperatures = {1.0, 1.0, 1.0, 1.0, 1.0};
    EXPECT_FALSE(has_error(validate_config(cfg),
                           "speculation.dspark.sts_temperatures"));
}

TEST(ConfigValidator, DsparkSchedulerRequiresConfidenceHead) {
    auto cfg = valid_config();
    cfg.speculation.method = SpeculationMethodType::dspark;
    cfg.speculation.dspark.scheduler_mode = DsparkSchedulerMode::throughput;
    cfg.speculation.dspark.confidence_enabled = false;

    EXPECT_TRUE(has_error(validate_config(cfg),
                          "speculation.dspark.scheduler_mode"));

    cfg.speculation.dspark.confidence_enabled = true;
    EXPECT_FALSE(has_error(validate_config(cfg),
                           "speculation.dspark.scheduler_mode"));
}

TEST(ConfigValidator, DsparkContractInertWhenMethodNotDspark) {
    // The dspark knob contract binds only when the method is selected.
    auto cfg = valid_config();
    cfg.speculation.method = SpeculationMethodType::none;
    cfg.speculation.dspark.sts_temperatures = {1.0, 1.1};  // mismatched, inert
    cfg.speculation.dspark.scheduler_mode = DsparkSchedulerMode::throughput;

    EXPECT_FALSE(has_error(validate_config(cfg), "speculation.dspark"));
}

TEST(ConfigValidator, AccumulatesMultipleErrors) {
    auto cfg = valid_config();
    cfg.model.hidden_size = 0;
    cfg.model.num_hidden_layers = 0;
    cfg.model.vocab_size = 0;
    cfg.hardware.system_ram_gb = 0;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_GE(result.errors.size(), 3u);
}

// ── memory.arena_attach.on_conflict × persist (P-24b) ───────────────────────

TEST(ConfigValidator, ArenaOnConflictKillRejectedUnderPersist) {
    // persist means the store is never wiped — 'kill' is a flat contradiction.
    auto cfg = valid_config();
    cfg.memory.arena_attach.persist = true;
    cfg.memory.arena_attach.on_conflict = ArenaOnConflict::kill;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "memory.arena_attach.on_conflict"));
}

TEST(ConfigValidator, ArenaOnConflictNewRejectedUnderPersist) {
    // persist declares the store authoritative — silently serving from a
    // throwaway private arena would bypass it (and double-allocate host RAM).
    auto cfg = valid_config();
    cfg.memory.arena_attach.persist = true;
    cfg.memory.arena_attach.on_conflict = ArenaOnConflict::new_arena;

    auto result = validate_config(cfg);
    EXPECT_FALSE(result.valid());
    EXPECT_TRUE(has_error(result, "memory.arena_attach.on_conflict"));
}

TEST(ConfigValidator, ArenaOnConflictAllowedCombinations) {
    // persist=true + 'fail' (or absent) is the only persist-compatible pair;
    // persist=false accepts every mode.
    auto cfg = valid_config();

    cfg.memory.arena_attach.persist = true;
    cfg.memory.arena_attach.on_conflict = ArenaOnConflict::fail;
    EXPECT_FALSE(has_error(validate_config(cfg), "arena_attach"));
    cfg.memory.arena_attach.on_conflict = ArenaOnConflict::unset;
    EXPECT_FALSE(has_error(validate_config(cfg), "arena_attach"));

    cfg.memory.arena_attach.persist = false;
    for (auto oc : {ArenaOnConflict::unset, ArenaOnConflict::new_arena,
                    ArenaOnConflict::fail, ArenaOnConflict::kill}) {
        cfg.memory.arena_attach.on_conflict = oc;
        EXPECT_FALSE(has_error(validate_config(cfg), "arena_attach"));
    }
}

TEST(ConfigValidator, ResolvedArenaOnConflictDerivation) {
    // Absent derives from persist — byte-for-byte the historical matrix:
    // persist=false → kill (wipe + cold rebuild), persist=true → fail.
    ArenaAttachConfig a;
    a.on_conflict = ArenaOnConflict::unset;
    a.persist = false;
    EXPECT_EQ(resolved_arena_on_conflict(a), ArenaOnConflict::kill);
    a.persist = true;
    EXPECT_EQ(resolved_arena_on_conflict(a), ArenaOnConflict::fail);

    // An explicit value always wins over the derivation.
    a.persist = false;
    a.on_conflict = ArenaOnConflict::new_arena;
    EXPECT_EQ(resolved_arena_on_conflict(a), ArenaOnConflict::new_arena);
    a.on_conflict = ArenaOnConflict::fail;
    EXPECT_EQ(resolved_arena_on_conflict(a), ArenaOnConflict::fail);
    a.on_conflict = ArenaOnConflict::kill;
    EXPECT_EQ(resolved_arena_on_conflict(a), ArenaOnConflict::kill);
}

// ═══════════════════════════════════════════════════════════════════════════
// DeepSeek-V4 rules (spec/DEEPSEEK4_PLAN.md V4-1c)
// ═══════════════════════════════════════════════════════════════════════════

/// Valid V4-Flash-shaped Config (mirrors test-data/config/deepseek_v4_flash_gguf.json).
static Config valid_v4_config() {
    Config cfg = valid_config();
    auto& m = cfg.model;
    m.architecture = Architecture::deepseek_v4;
    m.weights_path = "/data/models/deepseek-v4-flash/";
    m.weights_format = WeightsFormat::gguf;
    // V4 experts are GGUF (MXFP4) — swiglu_limit > 0 + nvfp4 is rejected
    // (V4-4b: fused SiLU-mul->NVFP4 kernel has no clamp support).
    cfg.quantization.weights = WeightQuant::gguf;
    m.num_hidden_layers = 43;
    m.hidden_size = 4096;
    m.num_attention_heads = 64;
    m.num_key_value_heads = 1;
    m.head_dim = 512;
    m.qk_rope_head_dim = 64;
    m.q_lora_rank = 1024;
    m.intermediate_size = 2048;
    m.num_experts_per_tok = 6;
    m.n_group = 1;
    m.topk_group = 1;
    m.first_k_dense_replace = 0;
    m.compress_ratios.assign({0, 0});
    for (int i = 0; i < 20; ++i) {
        m.compress_ratios.push_back(4);
        m.compress_ratios.push_back(128);
    }
    m.compress_ratios.push_back(4);  // 43 entries
    m.compress_rope_theta = 160000.0;
    m.sliding_window = 128;
    m.swiglu_limit = 10.0;
    m.num_hash_layers = 3;
    m.o_groups = 8;
    m.o_lora_rank = 1024;
    m.hc_mult = 4;
    m.hc_sinkhorn_iters = 20;
    m.hc_eps = 1e-6;
    m.gating_score_fn = GatingScoreFn::sqrtsoftplus;
    m.index_topk = 512;
    cfg.compute.attention_backend = AttentionBackendType::csa_hca;
    return cfg;
}

static bool has_error_on(const ValidationResult& r, const std::string& field) {
    for (const auto& e : r.errors)
        if (e.field == field) return true;
    return false;
}

TEST(ConfigValidatorV4, ValidV4ConfigPasses) {
    auto r = validate_config(valid_v4_config());
    for (const auto& e : r.errors)
        ADD_FAILURE() << "unexpected error on " << e.field << ": " << e.message;
    EXPECT_TRUE(r.valid());
}

// V4-2c (TD-V4-TP): TP sharding constraints — group-aligned rank shards.
TEST(ConfigValidatorV4, TpShardingRules) {
    {   // tp=2 on the Flash shape (64 heads / 8 groups) is legal.
        auto cfg = valid_v4_config();
        cfg.parallelism.tensor_parallelism = 2;
        EXPECT_TRUE(validate_config(cfg).valid());
    }
    {   // o_groups must divide by tp.
        auto cfg = valid_v4_config();
        cfg.parallelism.tensor_parallelism = 2;
        cfg.model.o_groups = 3;
        // (also breaks heads%o_groups — expect invalid regardless)
        EXPECT_FALSE(validate_config(cfg).valid());
    }
    {   // heads must divide by tp.
        auto cfg = valid_v4_config();
        cfg.parallelism.tensor_parallelism = 3;
        EXPECT_FALSE(validate_config(cfg).valid());
    }
}

TEST(ConfigValidatorV4, RequiresSingleKvHead) {
    auto cfg = valid_v4_config();
    cfg.model.num_key_value_heads = 64;  // the MLA convention — invalid for V4
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.num_key_value_heads"));
}

TEST(ConfigValidatorV4, RequiresHeadDim) {
    auto cfg = valid_v4_config();
    cfg.model.head_dim = 0;
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.head_dim"));
    cfg.model.head_dim = 64;  // must exceed qk_rope_head_dim
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.head_dim"));
}

TEST(ConfigValidatorV4, CompressRatiosLengthAndValues) {
    auto cfg = valid_v4_config();
    cfg.model.compress_ratios.resize(40);  // too short
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.compress_ratios"));

    cfg = valid_v4_config();
    cfg.model.compress_ratios[5] = 7;  // not in {0,4,128}
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.compress_ratios"));
}

TEST(ConfigValidatorV4, CompressRatiosAcceptsTrailingMtpZeros) {
    // The HF/GGUF array ships 46 = 43 layers + 3 MTP zeros — accepted.
    auto cfg = valid_v4_config();
    cfg.model.compress_ratios.insert(cfg.model.compress_ratios.end(), {0, 0, 0});
    ASSERT_EQ(cfg.model.compress_ratios.size(), 46u);
    EXPECT_FALSE(has_error_on(validate_config(cfg), "model.compress_ratios"));

    // …but non-zero padding beyond num_hidden_layers is rejected.
    cfg.model.compress_ratios[44] = 4;
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.compress_ratios"));
}

TEST(ConfigValidatorV4, RequiresCompressRopeThetaAndSlidingWindow) {
    auto cfg = valid_v4_config();
    cfg.model.compress_rope_theta = 0.0;
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.compress_rope_theta"));

    cfg = valid_v4_config();
    cfg.model.sliding_window = 0;
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.sliding_window"));
}

TEST(ConfigValidatorV4, GroupedOProjRules) {
    auto cfg = valid_v4_config();
    cfg.model.o_groups = 7;  // does not divide 64 heads
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.o_groups"));

    cfg = valid_v4_config();
    cfg.model.o_lora_rank = 0;  // required when o_groups > 1
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.o_lora_rank"));
}

TEST(ConfigValidatorV4, MhcRules) {
    auto cfg = valid_v4_config();
    cfg.model.hc_eps = 0.0;
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.hc_eps"));
}

TEST(ConfigValidatorV4, MlaBackendsRejectedForV4) {
    for (auto backend : {AttentionBackendType::snapmla,
                         AttentionBackendType::turboquant_mla}) {
        auto cfg = valid_v4_config();
        cfg.compute.attention_backend = backend;
        EXPECT_TRUE(has_error_on(validate_config(cfg),
                                 "compute.attention_backend"));
    }
}

TEST(ConfigValidatorV4, AllThreeV4BackendsAccepted) {
    for (auto backend : {AttentionBackendType::csa_hca,
                         AttentionBackendType::csa_hca_tq,
                         AttentionBackendType::csa_hca_tq_mix}) {
        auto cfg = valid_v4_config();
        cfg.compute.attention_backend = backend;
        EXPECT_FALSE(has_error_on(validate_config(cfg),
                                  "compute.attention_backend"));
    }
}

TEST(ConfigValidatorV4, V4BackendRejectedForNonV4) {
    auto cfg = valid_config();  // deepseek_v3
    cfg.compute.attention_backend = AttentionBackendType::csa_hca;
    EXPECT_TRUE(has_error_on(validate_config(cfg), "compute.attention_backend"));
}

TEST(ConfigValidatorV4, V32BaselineStillPasses) {
    // No-collateral-damage regression: the V3.2 baseline stays green.
    auto r = validate_config(valid_config());
    EXPECT_TRUE(r.valid());
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU-ONLY structural check of the SHIPPED V4-Flash config — NO CUDA, NO
// weights, NO decode (clone of the Dsp52ConfigCpu / Keeper52ConfigResolves
// pattern): parses test-data/config/deepseek_v4_flash_gguf.json and asserts
// the V4 feature surface is present and validator-clean.
// ═══════════════════════════════════════════════════════════════════════════
#include <fstream>
#include <nlohmann/json.hpp>

TEST(V4ConfigCpu, V4FlashGgufConfigResolves) {
    static const char* candidates[] = {
        "../../test-data/config/deepseek_v4_flash_gguf.json",
        "../test-data/config/deepseek_v4_flash_gguf.json",
        "test-data/config/deepseek_v4_flash_gguf.json",
#ifdef LAYERSTORM_SOURCE_DIR
        LAYERSTORM_SOURCE_DIR "/test-data/config/deepseek_v4_flash_gguf.json",
#endif
    };
    nlohmann::json j;
    bool found = false;
    for (const char* path : candidates) {
        std::ifstream f(path);
        if (f.is_open()) { j = nlohmann::json::parse(f); found = true; break; }
    }
    if (!found) GTEST_SKIP() << "deepseek_v4_flash_gguf.json not found";

    // JSON-level feature surface.
    EXPECT_EQ(j["model"]["architecture"], "deepseek_v4");
    EXPECT_EQ(j["model"]["num_key_value_heads"], 1);
    EXPECT_EQ(j["model"]["head_dim"], 512);
    EXPECT_EQ(j["model"]["gating_score_fn"], "sqrtsoftplus");
    EXPECT_EQ(j["model"]["hc_mult"], 4);
    EXPECT_EQ(j["compute"]["attention_backend"], "csa_hca");
    EXPECT_EQ(j["model"]["weights_format"], "gguf");
    // Legacy MLA approximations must be gone.
    EXPECT_FALSE(j["model"].contains("kv_lora_rank"));
    EXPECT_FALSE(j["model"].contains("qk_nope_head_dim"));
    EXPECT_FALSE(j["model"].contains("v_head_dim"));

    // Pure-CPU parse (throws on any schema/type error).
    Config cfg;
    ASSERT_NO_THROW(cfg = parse_config(j));

    EXPECT_EQ(cfg.model.architecture, Architecture::deepseek_v4);
    EXPECT_EQ(cfg.compute.attention_backend, AttentionBackendType::csa_hca);
    EXPECT_EQ(cfg.model.gating_score_fn, GatingScoreFn::sqrtsoftplus);
    ASSERT_EQ(cfg.model.compress_ratios.size(), 43u);
    EXPECT_EQ(cfg.model.compress_ratios[2], 4);
    EXPECT_EQ(cfg.model.compress_ratios[3], 128);
    EXPECT_EQ(cfg.model.compress_ratios[42], 4);
    EXPECT_DOUBLE_EQ(cfg.model.compress_rope_theta, 160000.0);
    EXPECT_EQ(cfg.model.sliding_window, 128);
    EXPECT_DOUBLE_EQ(cfg.model.swiglu_limit, 10.0);
    EXPECT_EQ(cfg.model.num_hash_layers, 3);
    EXPECT_EQ(cfg.model.o_groups, 8);
    EXPECT_EQ(cfg.model.o_lora_rank, 1024);
    EXPECT_EQ(cfg.model.hc_mult, 4);
    EXPECT_EQ(cfg.model.hc_sinkhorn_iters, 20);
    EXPECT_EQ(cfg.model.index_topk, 512);

    // Validator-clean on the V4 surface (model.* / compute.*). Whole-config
    // validity additionally needs resolve_config's hardware derivation
    // (dcp_enabled etc.), which a pure-CPU structural test skips.
    auto r = validate_config(cfg);
    for (const auto& e : r.errors) {
        if (e.field.rfind("model.", 0) == 0 || e.field.rfind("compute.", 0) == 0)
            ADD_FAILURE() << "unexpected error on " << e.field << ": "
                          << e.message;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU-ONLY structural check of the SHIPPED V4 SERVING config
// (config/examples/deepseek_v4_flash_serve.json): the production serving
// artifact must carry the FULL-RESIDENCY arena contract — prepacked source +
// pinned host pool + preload + persistent holder attach with persist=true
// (the only protection a store has since the holder honors reasoned wipes) —
// and must NOT resolve on_conflict to a wiping mode.
// ═══════════════════════════════════════════════════════════════════════════
TEST(V4ConfigCpu, V4FlashServeConfigCarriesResidentArenaContract) {
    static const char* candidates[] = {
        "../../config/examples/deepseek_v4_flash_serve.json",
        "../config/examples/deepseek_v4_flash_serve.json",
        "config/examples/deepseek_v4_flash_serve.json",
#ifdef LAYERSTORM_SOURCE_DIR
        LAYERSTORM_SOURCE_DIR "/config/examples/deepseek_v4_flash_serve.json",
#endif
    };
    nlohmann::json j;
    bool found = false;
    for (const char* path : candidates) {
        std::ifstream f(path);
        if (f.is_open()) { j = nlohmann::json::parse(f); found = true; break; }
    }
    if (!found) GTEST_SKIP() << "deepseek_v4_flash_serve.json not found";

    Config cfg;
    ASSERT_NO_THROW(cfg = parse_config(j));
    EXPECT_EQ(cfg.model.architecture, Architecture::deepseek_v4);

    // Residency: prepacked source + pinned arena + eager preload (the arena is
    // the warm tier; nothing must depend on page-cache warmth).
    EXPECT_FALSE(cfg.preprocessing.prepacked_dir.empty());
    EXPECT_TRUE(cfg.memory.preload_expert_buffers);
    EXPECT_TRUE(cfg.memory.pin_host_expert_pool);
    EXPECT_TRUE(cfg.memory.pin_host_expert_pool_preload);
    EXPECT_TRUE(cfg.memory.pin_host_expert_pool_direct_load.value_or(false));

    // Persistence + protection.
    EXPECT_TRUE(cfg.memory.arena_attach.enabled);
    EXPECT_TRUE(cfg.memory.arena_attach.persist);
    EXPECT_EQ(resolved_arena_on_conflict(cfg.memory.arena_attach),
              ArenaOnConflict::fail);

    // Capacity: tier-1 (GPU-attached) nodes plus spill banks must be able to
    // hold the whole routed set — the spill list is what makes it full-fit.
    EXPECT_TRUE(cfg.memory.cross_node_spill.enabled);
    EXPECT_FALSE(cfg.memory.cross_node_spill.nodes.empty());

    // Host-local placement (INV/D2H rule): every declared GPU sits on the NUMA
    // node the topology reports, so arena tier-1 == the fetching GPU's node.
    ASSERT_GE(cfg.hardware.gpus.size(), 1u);
    for (const auto& g : cfg.hardware.gpus) EXPECT_GE(g.numa_node, 0);

    auto r = validate_config(cfg);
    for (const auto& e : r.errors) {
        if (e.field.rfind("model.", 0) == 0 ||
            e.field.rfind("compute.", 0) == 0 ||
            e.field.rfind("memory.arena", 0) == 0 ||
            e.field.rfind("memory.pin_host", 0) == 0 ||
            e.field.rfind("memory.cross_node_spill", 0) == 0)
            ADD_FAILURE() << "unexpected error on " << e.field << ": "
                          << e.message;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// V4-4a/b gating scoring + SwiGLU clamp surface (validate_gating_activation)
// ═══════════════════════════════════════════════════════════════════════════

TEST(ConfigValidatorGating, SoftmaxScoringRejected) {
    // `softmax` is a schema enum value with NO gating-kernel implementation —
    // fail-closed until a kernel exists.
    auto cfg = valid_config();
    cfg.model.gating_score_fn = GatingScoreFn::softmax;
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.gating_score_fn"));

    auto v4 = valid_v4_config();
    v4.model.gating_score_fn = GatingScoreFn::softmax;
    EXPECT_TRUE(has_error_on(validate_config(v4), "model.gating_score_fn"));
}

TEST(ConfigValidatorGating, SigmoidAndSqrtsoftplusAccepted) {
    auto cfg = valid_config();
    cfg.model.gating_score_fn = GatingScoreFn::sigmoid;
    EXPECT_FALSE(has_error_on(validate_config(cfg), "model.gating_score_fn"));

    auto v4 = valid_v4_config();  // sqrtsoftplus
    EXPECT_FALSE(has_error_on(validate_config(v4), "model.gating_score_fn"));
}

TEST(ConfigValidatorGating, SwigluLimitRules) {
    // Negative limit invalid.
    auto cfg = valid_config();
    cfg.model.swiglu_limit = -1.0;
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.swiglu_limit"));

    // limit > 0 + NVFP4 expert weights: the fused SiLU-mul->NVFP4 kernel has
    // no clamp support — rejected.
    cfg = valid_config();  // quantization.weights = nvfp4
    cfg.model.swiglu_limit = 10.0;
    EXPECT_TRUE(has_error_on(validate_config(cfg), "model.swiglu_limit"));

    // limit > 0 + GGUF experts (the V4 surface): accepted.
    auto v4 = valid_v4_config();  // swiglu_limit 10, weights gguf
    EXPECT_FALSE(has_error_on(validate_config(v4), "model.swiglu_limit"));

    // limit 0 (V3.2/GLM default) always accepted, nvfp4 included.
    cfg = valid_config();
    cfg.model.swiglu_limit = 0.0;
    EXPECT_FALSE(has_error_on(validate_config(cfg), "model.swiglu_limit"));
}

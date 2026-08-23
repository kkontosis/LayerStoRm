#include <fstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "config/config_parser.h"

using namespace layerstorm::config;

// ── Helpers ──────────────────────────────────────────────────────────────────

static nlohmann::json minimal_json() {
    return {
        {"model", {
            {"architecture",           "deepseek_v3"},
            {"weights_path",           "/data/models/test/"},
            {"weights_format",         "safetensors"},
            {"num_hidden_layers",      61},
            {"hidden_size",            7168},
            {"num_attention_heads",    128},
            {"num_key_value_heads",    128},
            {"intermediate_size",      18432},
            {"n_routed_experts",       256},
            {"num_experts_per_tok",    8},
            {"vocab_size",             129280},
            {"max_position_embeddings", 163840},
        }},
        {"quantization", {
            {"weights",           "nvfp4"},
            {"attention_compute", "fp8_e4m3"},
            {"kv_cache",          "fp8_e4m3"},
            {"gating_compute",    "fp32"},
        }},
        {"hardware", {
            {"gpus", {{
                {"id",       0},
                {"type",     "rtx5090"},
                {"vram_gb",  32},
            }}},
            {"system_ram_gb", 256},
        }},
    };
}

static nlohmann::json load_reference_json() {
    // Try several paths: from build dir and from source root
    static const char* candidates[] = {
        "../../test-data/config/valid_deepseek_v3_2.json",
        "../test-data/config/valid_deepseek_v3_2.json",
        "test-data/config/valid_deepseek_v3_2.json",
#ifdef LAYERSTORM_SOURCE_DIR
        LAYERSTORM_SOURCE_DIR "/test-data/config/valid_deepseek_v3_2.json",
#endif
    };
    for (const char* path : candidates) {
        std::ifstream f(path);
        if (f.is_open()) return nlohmann::json::parse(f);
    }
    throw std::runtime_error("Could not open valid_deepseek_v3_2.json");
}

// ── Required-field parsing ───────────────────────────────────────────────────

// TD-VOCAB-AUTODETECT: model.vocab_size is OPTIONAL — absent (or 0) parses
// to 0, the "autodetect from the weights at load" sentinel; an explicit
// value is preserved (and cross-checked against the weights at load).
TEST(ConfigParser, VocabSizeOptionalDefaultsToAutodetect) {
    auto j = minimal_json();
    j["model"].erase("vocab_size");
    EXPECT_EQ(parse_config(j).model.vocab_size, 0);

    j["model"]["vocab_size"] = 0;
    EXPECT_EQ(parse_config(j).model.vocab_size, 0);

    j["model"]["vocab_size"] = 154880;
    EXPECT_EQ(parse_config(j).model.vocab_size, 154880);
}

TEST(ConfigParser, ParsesRequiredFields) {
    auto j = minimal_json();
    auto cfg = parse_config(j);

    EXPECT_EQ(cfg.model.architecture, Architecture::deepseek_v3);
    EXPECT_EQ(cfg.model.weights_path, "/data/models/test/");
    EXPECT_EQ(cfg.model.weights_format, WeightsFormat::safetensors);
    EXPECT_EQ(cfg.model.num_hidden_layers, 61);
    EXPECT_EQ(cfg.model.hidden_size, 7168);
    EXPECT_EQ(cfg.model.num_attention_heads, 128);
    EXPECT_EQ(cfg.model.num_key_value_heads, 128);
    EXPECT_EQ(cfg.model.intermediate_size, 18432);
    EXPECT_EQ(cfg.model.n_routed_experts, 256);
    EXPECT_EQ(cfg.model.num_experts_per_tok, 8);
    EXPECT_EQ(cfg.model.vocab_size, 129280);
    EXPECT_EQ(cfg.model.max_position_embeddings, 163840);

    EXPECT_EQ(cfg.quantization.weights, WeightQuant::nvfp4);
    EXPECT_EQ(cfg.quantization.attention_compute, AttentionQuant::fp8_e4m3);
    EXPECT_EQ(cfg.quantization.kv_cache, KvCacheQuant::fp8_e4m3);
    EXPECT_EQ(cfg.quantization.gating_compute, GatingQuant::fp32);

    ASSERT_EQ(cfg.hardware.gpus.size(), 1u);
    EXPECT_EQ(cfg.hardware.gpus[0].id, 0);
    EXPECT_EQ(cfg.hardware.gpus[0].type, GpuType::rtx5090);
    EXPECT_EQ(cfg.hardware.gpus[0].vram_gb, 32);
    EXPECT_EQ(cfg.hardware.system_ram_gb, 256);
}

// ── Default values for optional fields ──────────────────────────────────────

TEST(ConfigParser, DefaultsModel) {
    auto j = minimal_json();
    auto cfg = parse_config(j);

    // Model defaults
    EXPECT_EQ(cfg.model.kv_lora_rank, 512);
    EXPECT_EQ(cfg.model.qk_rope_head_dim, 64);
    EXPECT_EQ(cfg.model.qk_nope_head_dim, 128);
    EXPECT_EQ(cfg.model.v_head_dim, 128);
    EXPECT_EQ(cfg.model.q_lora_rank, 1536);
    EXPECT_EQ(cfg.model.n_shared_experts, 1);
    EXPECT_EQ(cfg.model.n_group, 8);
    EXPECT_EQ(cfg.model.topk_group, 4);
    EXPECT_DOUBLE_EQ(cfg.model.rope_theta, 10000.0);
    EXPECT_DOUBLE_EQ(cfg.model.rms_norm_eps, 1e-6);
    EXPECT_EQ(cfg.model.num_nextn_predict_layers, 1);
    EXPECT_EQ(cfg.model.index_topk, 2048);
    EXPECT_EQ(cfg.model.index_n_heads, 64);
    EXPECT_EQ(cfg.model.index_head_dim, 128);
    EXPECT_FALSE(cfg.model.rope_interleave);
    EXPECT_FALSE(cfg.model.indexer_rope_interleave);
    EXPECT_EQ(cfg.model.first_k_dense_replace, 3);
    EXPECT_DOUBLE_EQ(cfg.model.routed_scaling_factor, 2.5);
    EXPECT_EQ(cfg.model.moe_intermediate_size, 2048);
    EXPECT_EQ(cfg.model.moe_layer_freq, 1);
    EXPECT_TRUE(cfg.model.norm_topk_prob);
    EXPECT_EQ(cfg.model.gating_score_fn, GatingScoreFn::sigmoid);
    EXPECT_FALSE(cfg.model.rope_scaling.has_value());
    EXPECT_FALSE(cfg.model.vision.has_value());
}

TEST(ConfigParser, DefaultsHardware) {
    auto j = minimal_json();
    auto cfg = parse_config(j);

    // x-override fields: absent in JSON → sentinel value (auto-detect from hardware)
    EXPECT_EQ(cfg.hardware.gpus[0].pcie_gen, 0);     // sentinel: auto-detect
    EXPECT_EQ(cfg.hardware.gpus[0].pcie_width, 0);   // sentinel: auto-detect
    EXPECT_EQ(cfg.hardware.gpus[0].numa_node, -1);    // sentinel: auto-detect
    EXPECT_DOUBLE_EQ(cfg.hardware.gpus[0].pcie_bandwidth_loss_ratio, 0.85);
    EXPECT_EQ(cfg.hardware.gpus[0].vram_budget_gb, 0);
    ASSERT_EQ(cfg.hardware.gpus[0].roles.size(), 3u);
    EXPECT_TRUE(cfg.hardware.tp_array.empty());
    EXPECT_TRUE(cfg.hardware.nvme_paths.empty());
    EXPECT_FALSE(cfg.hardware.nvme_capacity_gb.has_value());
}

TEST(ConfigParser, DefaultsMemory) {
    auto j = minimal_json();
    auto cfg = parse_config(j);

    // pinned_layers defaults
    ASSERT_TRUE(std::holds_alternative<std::string>(cfg.memory.pinned_layers.attention));
    EXPECT_EQ(std::get<std::string>(cfg.memory.pinned_layers.attention), "all");
    ASSERT_TRUE(std::holds_alternative<std::string>(cfg.memory.pinned_layers.gating));
    EXPECT_EQ(std::get<std::string>(cfg.memory.pinned_layers.gating), "all");
    EXPECT_TRUE(cfg.memory.pinned_layers.embedding);
    EXPECT_TRUE(cfg.memory.pinned_layers.output_head);

    // kv_cache defaults
    ASSERT_TRUE(std::holds_alternative<std::string>(cfg.memory.kv_cache.max_pages_per_gpu));
    EXPECT_EQ(std::get<std::string>(cfg.memory.kv_cache.max_pages_per_gpu), "auto");
    EXPECT_EQ(cfg.memory.kv_cache.page_size_tokens, 16);
    EXPECT_DOUBLE_EQ(cfg.memory.kv_cache.speculation_pool_fraction, 0.15);

    // expert_cache defaults
    EXPECT_EQ(cfg.memory.expert_cache.eviction_policy, EvictionPolicy::impact_weighted_lru);
    EXPECT_DOUBLE_EQ(cfg.memory.expert_cache.eviction_alpha_recency, 0.4);
    EXPECT_DOUBLE_EQ(cfg.memory.expert_cache.eviction_beta_frequency, 0.35);
    EXPECT_DOUBLE_EQ(cfg.memory.expert_cache.eviction_gamma_routing_weight, 0.25);
    EXPECT_DOUBLE_EQ(cfg.memory.expert_cache.eviction_delta_temporal_autocorr, 0.0);
    EXPECT_DOUBLE_EQ(cfg.memory.expert_cache.eviction_epsilon_coactivation, 0.0);
    EXPECT_DOUBLE_EQ(cfg.memory.expert_cache.eviction_zeta_prefetch_score, 0.0);
    EXPECT_EQ(cfg.memory.expert_cache.duplication_frequency_threshold_percentile, 95);
    EXPECT_DOUBLE_EQ(cfg.memory.expert_cache.stable_zone_fraction, 0.7);

    // nvme_tier defaults
    EXPECT_FALSE(cfg.memory.nvme_tier.enabled);
    EXPECT_EQ(cfg.memory.nvme_tier.io_engine, IoEngine::io_uring);
    EXPECT_EQ(cfg.memory.nvme_tier.queue_depth, 64);

    // numa defaults
    EXPECT_EQ(cfg.memory.numa.interleave_strategy, NumaStrategy::affinity_based);
    EXPECT_EQ(cfg.memory.numa.fallback, NumaFallback::round_robin);
}

TEST(ConfigParser, DefaultsOrchestrator) {
    auto j = minimal_json();
    auto cfg = parse_config(j);

    EXPECT_DOUBLE_EQ(cfg.orchestrator.performance_objective.latency_weight, 0.6);
    EXPECT_DOUBLE_EQ(cfg.orchestrator.performance_objective.throughput_weight, 0.4);
    EXPECT_EQ(cfg.orchestrator.max_batch_size, 64);
    EXPECT_TRUE(cfg.orchestrator.expert_batching.enabled);
    EXPECT_EQ(cfg.orchestrator.expert_batching.max_wait_us, 50);
    EXPECT_EQ(cfg.orchestrator.expert_batching.min_batch_tokens_per_expert, 4);
    EXPECT_TRUE(cfg.orchestrator.coactivation_graph.enabled);
    EXPECT_DOUBLE_EQ(cfg.orchestrator.coactivation_graph.decay_factor, 0.999);
    EXPECT_EQ(cfg.orchestrator.coactivation_graph.reoptimize_interval_seconds, 300);
    EXPECT_DOUBLE_EQ(cfg.orchestrator.coactivation_graph.workload_shift_decay, 0.1);
    EXPECT_DOUBLE_EQ(cfg.orchestrator.workload_detection.ewma_alpha, 0.01);
    EXPECT_DOUBLE_EQ(cfg.orchestrator.workload_detection.shift_threshold_std_devs, 3.0);
}

TEST(ConfigParser, DefaultsSpeculation) {
    auto j = minimal_json();
    auto cfg = parse_config(j);

    EXPECT_TRUE(cfg.speculation.enabled);
    EXPECT_TRUE(cfg.speculation.mtp.enabled);
    EXPECT_EQ(cfg.speculation.mtp.max_depth, 3);
    EXPECT_TRUE(cfg.speculation.mtp.dynamic_depth);
    EXPECT_TRUE(cfg.speculation.self_speculative.enabled);
    EXPECT_EQ(cfg.speculation.self_speculative.draft_expert_count, 1);
    EXPECT_TRUE(cfg.speculation.prompt_lookup.enabled);
    EXPECT_EQ(cfg.speculation.prompt_lookup.max_ngram_size, 4);
    EXPECT_DOUBLE_EQ(cfg.speculation.verification.adaptive_topk_threshold, 0.92);
    EXPECT_DOUBLE_EQ(cfg.speculation.verification.verification_quality_floor, 0.85);
    EXPECT_EQ(cfg.speculation.verification.in_flight_transfer_mode, InFlightTransferMode::conservative);
    EXPECT_FALSE(cfg.speculation.layer_skip_draft.enabled);
    EXPECT_FALSE(cfg.speculation.reasoning_mode.enabled);
    EXPECT_DOUBLE_EQ(cfg.speculation.calibration.min_acceptance_rate, 0.3);
    EXPECT_TRUE(cfg.speculation.utility_scorer.enabled);
    EXPECT_DOUBLE_EQ(cfg.speculation.utility_scorer.utility_threshold, 1.0);
}

TEST(ConfigParser, DefaultsCompute) {
    auto j = minimal_json();
    auto cfg = parse_config(j);

    EXPECT_TRUE(cfg.compute.cuda_graphs.enabled);
    EXPECT_TRUE(cfg.compute.cuda_graphs.capture_attention);
    EXPECT_TRUE(cfg.compute.cuda_graphs.capture_gating);
    EXPECT_TRUE(cfg.compute.cuda_graphs.capture_tp_allreduce);
    // INV-0.6: capture_expert_ffn is a const field (always false), not in the struct
    EXPECT_EQ(cfg.compute.gemm.backend, GemmBackend::cutlass);
    EXPECT_TRUE(cfg.compute.gemm.grouped_gemm);
    EXPECT_TRUE(cfg.compute.gemm.fused_swiglu);
    EXPECT_EQ(cfg.compute.attention_backend, AttentionBackendType::snapmla);
    // TD-14d: the IPC sideband region is CUDA-pinned by default.
    EXPECT_TRUE(cfg.compute.ipc_pin);
}

// ── TD-14d: compute.ipc_pin (IPC sideband cudaHostRegister) ──────────────────

TEST(ConfigParser, IpcPinDefaultsOnAndParsesOff) {
    auto j = minimal_json();
    EXPECT_TRUE(parse_config(j).compute.ipc_pin);

    j["compute"]["ipc_pin"] = false;
    auto cfg = parse_config(j);
    EXPECT_FALSE(cfg.compute.ipc_pin);

    // Round-trips through config_to_json (the live-config / hot-reload seam).
    EXPECT_FALSE(parse_config(config_to_json(cfg)).compute.ipc_pin);

    j["compute"]["ipc_pin"] = true;
    EXPECT_TRUE(parse_config(j).compute.ipc_pin);
    EXPECT_TRUE(config_to_json(parse_config(j))["compute"]["ipc_pin"].get<bool>());
}

// ── INV-0.6: capture_expert_ffn is a const field ─────────────────────────────
// capture_expert_ffn is always false in the schema (const). The codegen skips
// it from the struct entirely — it's only emitted as a literal in to_json.
// No runtime test needed; JSON with capture_expert_ffn: true is silently ignored.

// ── Layer pin spec variants ───────────────────────────────────────────────────

TEST(ConfigParser, LayerPinSpecAll) {
    auto j = minimal_json();
    j["memory"] = {{"pinned_layers", {{"attention", "all"}}}};
    auto cfg = parse_config(j);
    ASSERT_TRUE(std::holds_alternative<std::string>(cfg.memory.pinned_layers.attention));
    EXPECT_EQ(std::get<std::string>(cfg.memory.pinned_layers.attention), "all");
}

TEST(ConfigParser, LayerPinSpecList) {
    auto j = minimal_json();
    j["memory"] = {{"pinned_layers", {{"attention", {0, 5, 10}}}}};
    auto cfg = parse_config(j);
    ASSERT_TRUE(std::holds_alternative<std::vector<int>>(cfg.memory.pinned_layers.attention));
    const auto& v = std::get<std::vector<int>>(cfg.memory.pinned_layers.attention);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], 0);
    EXPECT_EQ(v[1], 5);
    EXPECT_EQ(v[2], 10);
}

// ── AutoOrInt variants ───────────────────────────────────────────────────────

TEST(ConfigParser, AutoOrIntAuto) {
    auto j = minimal_json();
    j["memory"] = {{"kv_cache", {{"max_pages_per_gpu", "auto"}}}};
    auto cfg = parse_config(j);
    ASSERT_TRUE(std::holds_alternative<std::string>(cfg.memory.kv_cache.max_pages_per_gpu));
    EXPECT_EQ(std::get<std::string>(cfg.memory.kv_cache.max_pages_per_gpu), "auto");
}

TEST(ConfigParser, AutoOrIntValue) {
    auto j = minimal_json();
    j["memory"] = {{"kv_cache", {{"max_pages_per_gpu", 4096}}}};
    auto cfg = parse_config(j);
    ASSERT_TRUE(std::holds_alternative<int>(cfg.memory.kv_cache.max_pages_per_gpu));
    EXPECT_EQ(std::get<int>(cfg.memory.kv_cache.max_pages_per_gpu), 4096);
}

// ── TP array parsing ─────────────────────────────────────────────────────────

TEST(ConfigParser, TpArrayPresent) {
    auto j = minimal_json();
    j["hardware"]["tp_array"] = {0, 1};
    auto cfg = parse_config(j);
    ASSERT_EQ(cfg.hardware.tp_array.size(), 2u);
    EXPECT_EQ(cfg.hardware.tp_array[0], 0);
    EXPECT_EQ(cfg.hardware.tp_array[1], 1);
}

TEST(ConfigParser, TpArrayNull) {
    auto j = minimal_json();
    j["hardware"]["tp_array"] = nullptr;
    auto cfg = parse_config(j);
    EXPECT_TRUE(cfg.hardware.tp_array.empty());
}

TEST(ConfigParser, TpArrayFourGpus) {
    auto j = minimal_json();
    j["hardware"]["tp_array"] = {0, 1, 2, 3};
    auto cfg = parse_config(j);
    ASSERT_EQ(cfg.hardware.tp_array.size(), 4u);
    EXPECT_EQ(cfg.hardware.tp_array[0], 0);
    EXPECT_EQ(cfg.hardware.tp_array[3], 3);
}

// ── NVMe paths ───────────────────────────────────────────────────────────────

TEST(ConfigParser, NvmePathsEmpty) {
    auto j = minimal_json();
    j["hardware"]["nvme_paths"] = nlohmann::json::array();
    auto cfg = parse_config(j);
    EXPECT_TRUE(cfg.hardware.nvme_paths.empty());
}

TEST(ConfigParser, NvmePathsSingle) {
    auto j = minimal_json();
    j["hardware"]["nvme_paths"]       = {"/mnt/nvme0"};
    j["hardware"]["nvme_capacity_gb"] = 2000;
    auto cfg = parse_config(j);
    ASSERT_EQ(cfg.hardware.nvme_paths.size(), 1u);
    EXPECT_EQ(cfg.hardware.nvme_paths[0], "/mnt/nvme0");
    ASSERT_TRUE(cfg.hardware.nvme_capacity_gb.has_value());
    EXPECT_EQ(*cfg.hardware.nvme_capacity_gb, 2000);
}

TEST(ConfigParser, NvmePathsMulti) {
    auto j = minimal_json();
    j["hardware"]["nvme_paths"]       = {"/mnt/nvme0", "/mnt/nvme1", "/mnt/nvme2"};
    j["hardware"]["nvme_capacity_gb"] = 6000;
    auto cfg = parse_config(j);
    ASSERT_EQ(cfg.hardware.nvme_paths.size(), 3u);
    EXPECT_EQ(cfg.hardware.nvme_paths[0], "/mnt/nvme0");
    EXPECT_EQ(cfg.hardware.nvme_paths[1], "/mnt/nvme1");
    EXPECT_EQ(cfg.hardware.nvme_paths[2], "/mnt/nvme2");
}

// ── RopeScaling optional field ────────────────────────────────────────────────

TEST(ConfigParser, RopeScalingPresent) {
    auto j = minimal_json();
    j["model"]["rope_scaling"] = {
        {"type",    "yarn"},
        {"factor",  4.0},
        {"beta_fast", 32.0},
        {"beta_slow", 1.0},
    };
    auto cfg = parse_config(j);
    ASSERT_TRUE(cfg.model.rope_scaling.has_value());
    EXPECT_EQ(cfg.model.rope_scaling->type, RopeScalingType::yarn);
    ASSERT_TRUE(cfg.model.rope_scaling->factor.has_value());
    EXPECT_DOUBLE_EQ(*cfg.model.rope_scaling->factor, 4.0);
    ASSERT_TRUE(cfg.model.rope_scaling->beta_fast.has_value());
    EXPECT_DOUBLE_EQ(*cfg.model.rope_scaling->beta_fast, 32.0);
    EXPECT_FALSE(cfg.model.rope_scaling->mscale.has_value());
}

// ── Vision optional field ────────────────────────────────────────────────────

TEST(ConfigParser, VisionPresent) {
    auto j = minimal_json();
    j["model"]["vision"] = {
        {"enabled",     true},
        {"hidden_size", 1152},
        {"num_layers",  27},
        {"num_heads",   16},
        {"patch_size",  14},
        {"image_size",  384},
    };
    auto cfg = parse_config(j);
    ASSERT_TRUE(cfg.model.vision.has_value());
    EXPECT_TRUE(cfg.model.vision->enabled);
    EXPECT_EQ(cfg.model.vision->hidden_size, 1152);
}

// ── GPU roles ────────────────────────────────────────────────────────────────

TEST(ConfigParser, GpuRoles) {
    auto j = minimal_json();
    j["hardware"]["gpus"][0]["roles"] = {"attention", "expert_streaming"};
    auto cfg = parse_config(j);
    ASSERT_EQ(cfg.hardware.gpus[0].roles.size(), 2u);
    EXPECT_EQ(cfg.hardware.gpus[0].roles[0], GpuRole::attention);
    EXPECT_EQ(cfg.hardware.gpus[0].roles[1], GpuRole::expert_streaming);
}

// ── VramAllocation optional field ────────────────────────────────────────────

TEST(ConfigParser, VramAllocationPresent) {
    auto j = minimal_json();
    j["hardware"]["gpus"][0]["vram_allocation_gb"] = {
        {"resident",            8.0},
        {"expert_streaming",    20.0},
        {"stable_zone_fraction", 0.8},
    };
    auto cfg = parse_config(j);
    ASSERT_TRUE(cfg.hardware.gpus[0].vram_allocation_gb.has_value());
    EXPECT_DOUBLE_EQ(cfg.hardware.gpus[0].vram_allocation_gb->resident, 8.0);
    EXPECT_DOUBLE_EQ(cfg.hardware.gpus[0].vram_allocation_gb->stable_zone_fraction, 0.8);
}

// ── Type errors ──────────────────────────────────────────────────────────────

TEST(ConfigParser, ThrowsOnMissingRequiredField) {
    auto j = minimal_json();
    j["model"].erase("architecture");
    EXPECT_THROW(parse_config(j), nlohmann::json::exception);
}

TEST(ConfigParser, ThrowsOnUnknownArchitecture) {
    auto j = minimal_json();
    j["model"]["architecture"] = "unknown_arch";
    EXPECT_THROW(parse_config(j), std::runtime_error);
}

TEST(ConfigParser, ThrowsOnUnknownWeightQuant) {
    auto j = minimal_json();
    j["quantization"]["weights"] = "fp3";
    EXPECT_THROW(parse_config(j), std::runtime_error);
}

// ── GGUF weight quant types + strategy (GG-1) ─────────────────────────────────

TEST(ConfigParser, GgufWeightQuantTypesRoundTrip) {
    // Each GGUF weight type parses, and survives enum → JSON → enum round-trip.
    struct Case { const char* str; WeightQuant val; };
    const Case cases[] = {
        {"gguf",      WeightQuant::gguf},       // generic: per-tensor mixed types from the file
        {"gguf_q2_k", WeightQuant::gguf_q2_k},
        {"gguf_q3_k", WeightQuant::gguf_q3_k},
        {"gguf_q4_k", WeightQuant::gguf_q4_k},
        {"gguf_q5_k", WeightQuant::gguf_q5_k},
        {"gguf_q6_k", WeightQuant::gguf_q6_k},
        {"gguf_q8_0", WeightQuant::gguf_q8_0},
    };
    for (const auto& c : cases) {
        auto j = minimal_json();
        j["quantization"]["weights"] = c.str;
        auto cfg = parse_config(j);
        EXPECT_EQ(cfg.quantization.weights, c.val) << "parsing " << c.str;
        // Round-trip: enum → JSON string → enum
        auto j2 = config_to_json(cfg);
        EXPECT_EQ(j2["quantization"]["weights"].get<std::string>(), c.str);
        auto cfg2 = parse_config(j2);
        EXPECT_EQ(cfg2.quantization.weights, c.val) << "round-trip " << c.str;
    }
}

TEST(ConfigParser, GgufStrategyDefaultsToInt) {
    auto j = minimal_json();
    // gguf_strategy omitted entirely from quantization block.
    ASSERT_FALSE(j["quantization"].contains("gguf_strategy"));
    auto cfg = parse_config(j);
    EXPECT_EQ(cfg.quantization.gguf_strategy, GgufStrategy::int_strategy);
}

TEST(ConfigParser, GgufStrategyRoundTripInt) {
    auto j = minimal_json();
    j["quantization"]["gguf_strategy"] = "int";
    auto cfg = parse_config(j);
    EXPECT_EQ(cfg.quantization.gguf_strategy, GgufStrategy::int_strategy);
    auto j2 = config_to_json(cfg);
    EXPECT_EQ(j2["quantization"]["gguf_strategy"].get<std::string>(), "int");
    auto cfg2 = parse_config(j2);
    EXPECT_EQ(cfg2.quantization.gguf_strategy, GgufStrategy::int_strategy);
}

TEST(ConfigParser, GgufStrategyRoundTripDequant) {
    auto j = minimal_json();
    j["quantization"]["gguf_strategy"] = "dequant";
    auto cfg = parse_config(j);
    EXPECT_EQ(cfg.quantization.gguf_strategy, GgufStrategy::dequant);
    auto j2 = config_to_json(cfg);
    EXPECT_EQ(j2["quantization"]["gguf_strategy"].get<std::string>(), "dequant");
    auto cfg2 = parse_config(j2);
    EXPECT_EQ(cfg2.quantization.gguf_strategy, GgufStrategy::dequant);
}

TEST(ConfigParser, ThrowsOnUnknownGgufStrategy) {
    auto j = minimal_json();
    j["quantization"]["gguf_strategy"] = "int8_tc";
    EXPECT_THROW(parse_config(j), std::runtime_error);
}

TEST(ConfigParser, ThrowsOnMissingGpuArray) {
    auto j = minimal_json();
    j["hardware"].erase("gpus");
    EXPECT_THROW(parse_config(j), nlohmann::json::exception);
}

TEST(ConfigParser, ThrowsOnBadLayerPinSpec) {
    auto j = minimal_json();
    j["memory"] = {{"pinned_layers", {{"attention", "partial"}}}};
    EXPECT_THROW(parse_config(j), std::runtime_error);
}

TEST(ConfigParser, ThrowsOnBadAutoOrInt) {
    auto j = minimal_json();
    j["memory"] = {{"kv_cache", {{"max_pages_per_gpu", "some_string"}}}};
    EXPECT_THROW(parse_config(j), std::runtime_error);
}

// ── Round-trip: JSON -> Config -> JSON ───────────────────────────────────────

TEST(ConfigParser, RoundTripMinimal) {
    auto j = minimal_json();
    auto cfg = parse_config(j);
    auto j2  = config_to_json(cfg);
    // Re-parse and verify key fields survived round-trip
    auto cfg2 = parse_config(j2);
    EXPECT_EQ(cfg2.model.architecture,       cfg.model.architecture);
    EXPECT_EQ(cfg2.model.weights_path,        cfg.model.weights_path);
    EXPECT_EQ(cfg2.model.num_hidden_layers,   cfg.model.num_hidden_layers);
    EXPECT_EQ(cfg2.model.n_routed_experts,    cfg.model.n_routed_experts);
    EXPECT_EQ(cfg2.quantization.weights,      cfg.quantization.weights);
    EXPECT_EQ(cfg2.hardware.gpus.size(),      cfg.hardware.gpus.size());
    EXPECT_EQ(cfg2.hardware.system_ram_gb,    cfg.hardware.system_ram_gb);
}

// ── Full reference config ────────────────────────────────────────────────────

TEST(ConfigParser, ParsesReferenceConfig) {
    nlohmann::json j;
    try {
        j = load_reference_json();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping reference config test: " << e.what();
    }
    Config cfg;
    ASSERT_NO_THROW(cfg = parse_config(j));

    // Spot-check DeepSeek V3.2 values
    EXPECT_EQ(cfg.model.architecture, Architecture::deepseek_v3);
    EXPECT_EQ(cfg.model.num_hidden_layers, 61);
    EXPECT_EQ(cfg.model.hidden_size, 7168);
    EXPECT_EQ(cfg.model.n_routed_experts, 256);
    EXPECT_EQ(cfg.model.num_experts_per_tok, 8);
    EXPECT_EQ(cfg.model.kv_lora_rank, 512);
    EXPECT_EQ(cfg.model.n_group, 8);
    EXPECT_EQ(cfg.model.num_nextn_predict_layers, 1);
    EXPECT_EQ(cfg.model.gating_score_fn, GatingScoreFn::sigmoid);
    EXPECT_TRUE(cfg.model.norm_topk_prob);

    EXPECT_EQ(cfg.model.q_lora_rank, 1536);
    EXPECT_EQ(cfg.model.topk_group, 4);
    EXPECT_EQ(cfg.model.index_topk, 2048);
    EXPECT_EQ(cfg.model.first_k_dense_replace, 3);
    EXPECT_DOUBLE_EQ(cfg.model.routed_scaling_factor, 2.5);
    EXPECT_EQ(cfg.model.moe_intermediate_size, 2048);
    EXPECT_EQ(cfg.model.moe_layer_freq, 1);

    EXPECT_EQ(cfg.quantization.weights, WeightQuant::nvfp4);
    EXPECT_EQ(cfg.quantization.attention_compute, AttentionQuant::fp8_e4m3);

    ASSERT_EQ(cfg.hardware.gpus.size(), 4u);
    EXPECT_EQ(cfg.hardware.gpus[0].type, GpuType::rtx5090);
    EXPECT_EQ(cfg.hardware.gpus[2].type, GpuType::rtx5080);
    ASSERT_EQ(cfg.hardware.tp_array.size(), 2u);
    EXPECT_EQ(cfg.hardware.tp_array[0], 0);
    EXPECT_EQ(cfg.hardware.tp_array[1], 1);
    EXPECT_EQ(cfg.hardware.system_ram_gb, 256);
    EXPECT_TRUE(cfg.hardware.nvme_paths.empty());
    EXPECT_EQ(cfg.hardware.dcp_indexer_mode, DcpIndexerMode::replicated);

    // Memory
    ASSERT_TRUE(std::holds_alternative<std::string>(cfg.memory.pinned_layers.attention));
    EXPECT_EQ(std::get<std::string>(cfg.memory.pinned_layers.attention), "all");
    ASSERT_EQ(cfg.memory.pinned_layers.dense_ffn_layers.size(), 5u);
    EXPECT_EQ(cfg.memory.pinned_layers.dense_ffn_layers[0], 0);
    // tp_mode_per_layer: all null (sentinel 0) — inherit from tensor_parallelism
    EXPECT_EQ(cfg.memory.tp_mode_per_layer.default_mode, 0);
    EXPECT_EQ(cfg.memory.tp_mode_per_layer.attention, 0);
    EXPECT_EQ(cfg.memory.tp_mode_per_layer.gating, 0);
    // KV cache DCP fields
    EXPECT_EQ(cfg.memory.kv_cache.indexer_k_page_size_tokens, 64);
    EXPECT_DOUBLE_EQ(cfg.memory.kv_cache.prefill_scratch_preallocated_gb, 0.5);
    EXPECT_DOUBLE_EQ(cfg.memory.kv_cache.streaming_spill_fraction, 0.34);
    EXPECT_FALSE(cfg.memory.kv_cache.naive_prefill_context_ceiling.has_value());
    // Expert cache
    EXPECT_DOUBLE_EQ(cfg.memory.expert_cache.eviction_alpha_recency, 0.4);
    EXPECT_DOUBLE_EQ(cfg.memory.expert_cache.stable_zone_fraction, 0.7);
    // NVMe tier
    EXPECT_FALSE(cfg.memory.nvme_tier.enabled);
    EXPECT_TRUE(cfg.memory.nvme_tier.direct_io);
    EXPECT_FALSE(cfg.memory.nvme_tier.host_ram_cache_gb.has_value());

    // Parallelism
    EXPECT_EQ(cfg.parallelism.tensor_parallelism, 2);

    // Speculation
    EXPECT_TRUE(cfg.speculation.enabled);
    EXPECT_TRUE(cfg.speculation.mtp.enabled);
    EXPECT_EQ(cfg.speculation.mtp.max_depth, 3);
    EXPECT_DOUBLE_EQ(cfg.speculation.verification.adaptive_topk_threshold, 0.92);
    EXPECT_DOUBLE_EQ(cfg.speculation.verification.verification_quality_floor, 0.85);
    EXPECT_EQ(cfg.speculation.verification.in_flight_transfer_mode, InFlightTransferMode::conservative);
    EXPECT_FALSE(cfg.speculation.layer_skip_draft.enabled);
    EXPECT_FALSE(cfg.speculation.reasoning_mode.enabled);

    // Round-trip
    auto j2   = config_to_json(cfg);
    auto cfg2 = parse_config(j2);
    EXPECT_EQ(cfg2.model.architecture,    cfg.model.architecture);
    EXPECT_EQ(cfg2.model.n_routed_experts, cfg.model.n_routed_experts);
    EXPECT_EQ(cfg2.hardware.gpus.size(),  cfg.hardware.gpus.size());
}

// ── Multiple GPU types ───────────────────────────────────────────────────────

TEST(ConfigParser, MultipleGpuTypes) {
    auto j = minimal_json();
    j["hardware"]["gpus"] = {
        {{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}},
        {{"id", 1}, {"type", "rtx5080"}, {"vram_gb", 16}},
    };
    auto cfg = parse_config(j);
    ASSERT_EQ(cfg.hardware.gpus.size(), 2u);
    EXPECT_EQ(cfg.hardware.gpus[0].type, GpuType::rtx5090);
    EXPECT_EQ(cfg.hardware.gpus[0].vram_gb, 32);
    EXPECT_EQ(cfg.hardware.gpus[1].type, GpuType::rtx5080);
    EXPECT_EQ(cfg.hardware.gpus[1].vram_gb, 16);
}

// ── All architectures ────────────────────────────────────────────────────────

TEST(ConfigParser, ArchitectureGlmMoeDsa) {
    auto j = minimal_json();
    j["model"]["architecture"] = "glm_moe_dsa";
    auto cfg = parse_config(j);
    EXPECT_EQ(cfg.model.architecture, Architecture::glm_moe_dsa);
}

TEST(ConfigParser, ArchitectureKimiK25) {
    auto j = minimal_json();
    j["model"]["architecture"] = "kimi_k25";
    auto cfg = parse_config(j);
    EXPECT_EQ(cfg.model.architecture, Architecture::kimi_k25);
}

// ── Verification modes ────────────────────────────────────────────────────────

TEST(ConfigParser, VerificationOptimisticMode) {
    auto j = minimal_json();
    j["speculation"] = {{"verification", {{"in_flight_transfer_mode", "optimistic"}}}};
    auto cfg = parse_config(j);
    EXPECT_EQ(cfg.speculation.verification.in_flight_transfer_mode, InFlightTransferMode::optimistic);
}

TEST(ConfigParser, RankingStrategyStaticCalibration) {
    auto j = minimal_json();
    j["speculation"] = {{"verification", {{"ranking_strategy", "static_calibration"}}}};
    auto cfg = parse_config(j);
    EXPECT_EQ(cfg.speculation.verification.ranking_strategy, RankingStrategy::static_calibration);
    // Round-trip
    auto j2 = config_to_json(cfg);
    EXPECT_EQ(j2["speculation"]["verification"]["ranking_strategy"].get<std::string>(), "static_calibration");
}

// ── Prefetch probe_points ──────────────────────────────────────────────────────

TEST(ConfigParser, ProbePoints) {
    auto j = minimal_json();
    j["prefetch"] = {{"probe", {{"probe_points", {0.33, 0.66}}}}};
    auto cfg = parse_config(j);
    ASSERT_EQ(cfg.prefetch.probe.probe_points.size(), 2u);
    EXPECT_DOUBLE_EQ(cfg.prefetch.probe.probe_points[0], 0.33);
    EXPECT_DOUBLE_EQ(cfg.prefetch.probe.probe_points[1], 0.66);
}

// ── Layer skip exit_layer null/set ────────────────────────────────────────────

TEST(ConfigParser, LayerSkipExitLayerNull) {
    auto j = minimal_json();
    j["speculation"] = {{"layer_skip_draft", {{"exit_layer", nullptr}}}};
    auto cfg = parse_config(j);
    EXPECT_FALSE(cfg.speculation.layer_skip_draft.exit_layer.has_value());
}

TEST(ConfigParser, LayerSkipExitLayerSet) {
    auto j = minimal_json();
    j["speculation"] = {{"layer_skip_draft", {{"exit_layer", 30}}}};
    auto cfg = parse_config(j);
    ASSERT_TRUE(cfg.speculation.layer_skip_draft.exit_layer.has_value());
    EXPECT_EQ(*cfg.speculation.layer_skip_draft.exit_layer, 30);
}

// ── Prescope predictor ────────────────────────────────────────────────────────

TEST(ConfigParser, PrescopePredictorDefaults) {
    auto j = minimal_json();
    auto cfg = parse_config(j);
    EXPECT_TRUE(cfg.prefetch.prescope.enabled);
    EXPECT_EQ(cfg.prefetch.prescope.top_k, 8);
    EXPECT_DOUBLE_EQ(cfg.prefetch.prescope.score_threshold, 0.01);
    EXPECT_FALSE(cfg.prefetch.prescope.predictor_enabled);
    // Internal predictor params via _internal-prescope
    EXPECT_EQ(cfg._internal_prescope.hidden_size, 7168);
    EXPECT_EQ(cfg._internal_prescope.pca_dim, 256);
    EXPECT_EQ(cfg._internal_prescope.hidden_dim, 256);
    EXPECT_DOUBLE_EQ(cfg._internal_prescope.learning_rate, 0.001);
    EXPECT_DOUBLE_EQ(cfg._internal_prescope.momentum, 0.9);
    EXPECT_EQ(cfg._internal_prescope.training_batch_size, 32);
    EXPECT_EQ(cfg._internal_prescope.training_buffer_size, 2048);
    EXPECT_DOUBLE_EQ(cfg._internal_prescope.focal_loss_gamma, 2.0);
    EXPECT_DOUBLE_EQ(cfg._internal_prescope.balance_loss_lambda, 0.5);
    EXPECT_DOUBLE_EQ(cfg._internal_prescope.input_group_frac, 0.25);
    EXPECT_DOUBLE_EQ(cfg._internal_prescope.output_group_frac, 0.25);
    EXPECT_EQ(cfg._internal_prescope.min_samples_before_predict, 64);
    EXPECT_DOUBLE_EQ(cfg._internal_prescope.freq_ema_alpha, 0.01);
}

TEST(ConfigParser, PrescopePredictorOverride) {
    auto j = minimal_json();
    j["prefetch"] = {{"prescope", {
        {"top_k", 16},
        {"score_threshold", 0.05},
        {"predictor_enabled", true},
    }}};
    j["_internal-prescope"] = {
        {"hidden_size", 4096},
        {"learning_rate", 0.01},
        {"training_batch_size", 64},
    };
    auto cfg = parse_config(j);
    EXPECT_EQ(cfg.prefetch.prescope.top_k, 16);
    EXPECT_DOUBLE_EQ(cfg.prefetch.prescope.score_threshold, 0.05);
    EXPECT_TRUE(cfg.prefetch.prescope.predictor_enabled);
    EXPECT_EQ(cfg._internal_prescope.hidden_size, 4096);
    EXPECT_DOUBLE_EQ(cfg._internal_prescope.learning_rate, 0.01);
    EXPECT_EQ(cfg._internal_prescope.training_batch_size, 64);
    // Unset fields keep defaults
    EXPECT_EQ(cfg._internal_prescope.pca_dim, 256);
}

TEST(ConfigParser, PrescopeRoundTrip) {
    auto j = minimal_json();
    j["prefetch"] = {{"prescope", {
        {"top_k", 12},
        {"predictor_enabled", true},
    }}};
    j["_internal-prescope"] = {{"focal_loss_gamma", 3.0}};
    auto cfg = parse_config(j);
    auto j2 = config_to_json(cfg);
    auto cfg2 = parse_config(j2);
    EXPECT_EQ(cfg2.prefetch.prescope.top_k, 12);
    EXPECT_TRUE(cfg2.prefetch.prescope.predictor_enabled);
    EXPECT_DOUBLE_EQ(cfg2._internal_prescope.focal_loss_gamma, 3.0);
}

// ── Preprocessing defaults ──────────────────────────────────────────────────

TEST(ConfigParser, DefaultsPreprocessing) {
    auto j = minimal_json();
    auto cfg = parse_config(j);
    EXPECT_EQ(cfg.preprocessing.prepacked_dir, "");
    EXPECT_FALSE(cfg.preprocessing.auto_preprocess);
    EXPECT_EQ(cfg.preprocessing.auto_preprocess_target, "");
    EXPECT_EQ(cfg.preprocessing.host_cache_mode, HostCacheMode::mmap);
}

TEST(ConfigParser, PreprocessingOverride) {
    auto j = minimal_json();
    j["preprocessing"] = {
        {"prepacked_dir", "/data/prepacked"},
        {"auto_preprocess", true},
        {"auto_preprocess_target", "/data/prepacked-out"},
        {"host_cache_mode", "explicit_alloc"},
    };
    auto cfg = parse_config(j);
    EXPECT_EQ(cfg.preprocessing.prepacked_dir, "/data/prepacked");
    EXPECT_TRUE(cfg.preprocessing.auto_preprocess);
    EXPECT_EQ(cfg.preprocessing.auto_preprocess_target, "/data/prepacked-out");
    EXPECT_EQ(cfg.preprocessing.host_cache_mode, HostCacheMode::explicit_alloc);
}

TEST(ConfigParser, PreprocessingRoundTrip) {
    auto j = minimal_json();
    j["preprocessing"] = {
        {"prepacked_dir", "/mnt/nvme/prepacked"},
        {"host_cache_mode", "mmap"},
    };
    auto cfg = parse_config(j);
    auto j2 = config_to_json(cfg);
    auto cfg2 = parse_config(j2);
    EXPECT_EQ(cfg2.preprocessing.prepacked_dir, "/mnt/nvme/prepacked");
    EXPECT_EQ(cfg2.preprocessing.host_cache_mode, HostCacheMode::mmap);
}

TEST(ConfigParser, PreprocessingBadHostCacheMode) {
    auto j = minimal_json();
    j["preprocessing"] = {{"host_cache_mode", "invalid"}};
    EXPECT_THROW(parse_config(j), std::runtime_error);
}

// ── memory.arena_attach.on_conflict (P-24b) ─────────────────────────────────

TEST(ConfigParser, ArenaAttachOnConflictAbsentIsUnset) {
    // Absent field = the C++-only "unset" sentinel (default derives from
    // persist at resolve time, resolved_arena_on_conflict) — NOT any of the
    // three JSON values.
    auto j = minimal_json();
    auto cfg = parse_config(j);
    EXPECT_EQ(cfg.memory.arena_attach.on_conflict, ArenaOnConflict::unset);
}

TEST(ConfigParser, ArenaAttachOnConflictParsesAllModes) {
    {
        auto j = minimal_json();
        j["memory"]["arena_attach"]["on_conflict"] = "new";
        EXPECT_EQ(parse_config(j).memory.arena_attach.on_conflict,
                  ArenaOnConflict::new_arena);
    }
    {
        auto j = minimal_json();
        j["memory"]["arena_attach"]["on_conflict"] = "fail";
        EXPECT_EQ(parse_config(j).memory.arena_attach.on_conflict,
                  ArenaOnConflict::fail);
    }
    {
        auto j = minimal_json();
        j["memory"]["arena_attach"]["on_conflict"] = "kill";
        EXPECT_EQ(parse_config(j).memory.arena_attach.on_conflict,
                  ArenaOnConflict::kill);
    }
}

TEST(ConfigParser, ArenaAttachOnConflictRejectsUnknownValue) {
    auto j = minimal_json();
    j["memory"]["arena_attach"]["on_conflict"] = "wipe";
    EXPECT_THROW(parse_config(j), std::runtime_error);
    // The C++-only sentinel has no JSON spelling either.
    j["memory"]["arena_attach"]["on_conflict"] = "unset";
    EXPECT_THROW(parse_config(j), std::runtime_error);
}

TEST(ConfigParser, ArenaAttachOnConflictRoundTrip) {
    // Absent round-trips as absent (unset is never serialized)…
    auto j = minimal_json();
    auto out = config_to_json(parse_config(j));
    EXPECT_FALSE(out["memory"]["arena_attach"].contains("on_conflict"));
    // …and an explicit value round-trips verbatim.
    j["memory"]["arena_attach"]["on_conflict"] = "kill";
    out = config_to_json(parse_config(j));
    ASSERT_TRUE(out["memory"]["arena_attach"].contains("on_conflict"));
    EXPECT_EQ(out["memory"]["arena_attach"]["on_conflict"], "kill");
    EXPECT_EQ(parse_config(out).memory.arena_attach.on_conflict,
              ArenaOnConflict::kill);
}

// ── DeepSeek-V4 fields (spec/DEEPSEEK4_PLAN.md V4-1a) ────────────────────────

static nlohmann::json v4_minimal_json() {
    auto j = minimal_json();
    j["model"]["architecture"] = "deepseek_v4";
    j["model"]["num_hidden_layers"] = 43;
    j["model"]["hidden_size"] = 4096;
    j["model"]["num_attention_heads"] = 64;
    j["model"]["num_key_value_heads"] = 1;
    j["model"]["intermediate_size"] = 2048;
    j["model"]["num_experts_per_tok"] = 6;
    j["model"]["head_dim"] = 512;
    // 43-layer pattern: [0,0,(4,128)x20,4]
    std::vector<int> ratios{0, 0};
    for (int i = 0; i < 20; ++i) { ratios.push_back(4); ratios.push_back(128); }
    ratios.push_back(4);
    j["model"]["compress_ratios"] = ratios;
    j["model"]["compress_rope_theta"] = 160000.0;
    j["model"]["sliding_window"] = 128;
    j["model"]["swiglu_limit"] = 10.0;
    j["model"]["num_hash_layers"] = 3;
    j["model"]["o_groups"] = 8;
    j["model"]["o_lora_rank"] = 1024;
    j["model"]["hc_mult"] = 4;
    j["model"]["hc_sinkhorn_iters"] = 20;
    j["model"]["hc_eps"] = 1e-6;
    j["model"]["gating_score_fn"] = "sqrtsoftplus";
    return j;
}

TEST(ConfigParser, V4FieldsParse) {
    auto cfg = parse_config(v4_minimal_json());
    EXPECT_EQ(cfg.model.architecture, Architecture::deepseek_v4);
    EXPECT_EQ(cfg.model.head_dim, 512);
    ASSERT_EQ(cfg.model.compress_ratios.size(), 43u);
    EXPECT_EQ(cfg.model.compress_ratios[0], 0);
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
    EXPECT_DOUBLE_EQ(cfg.model.hc_eps, 1e-6);
    EXPECT_EQ(cfg.model.gating_score_fn, GatingScoreFn::sqrtsoftplus);
}

TEST(ConfigParser, V4AttentionBackendsParse) {
    auto j = v4_minimal_json();
    for (const char* name : {"csa_hca", "csa_hca_tq", "csa_hca_tq_mix"}) {
        j["compute"]["attention_backend"] = name;
        EXPECT_NO_THROW(parse_config(j)) << name;
    }
    EXPECT_EQ(parse_config(j).compute.attention_backend,
              AttentionBackendType::csa_hca_tq_mix);
}

TEST(ConfigParser, V4DefaultsLeaveV32Unchanged) {
    // A V3.2-style config picks up ONLY inert defaults for the V4 fields.
    auto cfg = parse_config(minimal_json());
    EXPECT_EQ(cfg.model.head_dim, 0);
    EXPECT_TRUE(cfg.model.compress_ratios.empty());
    EXPECT_DOUBLE_EQ(cfg.model.compress_rope_theta, 0.0);
    EXPECT_EQ(cfg.model.sliding_window, 0);
    EXPECT_DOUBLE_EQ(cfg.model.swiglu_limit, 0.0);
    EXPECT_EQ(cfg.model.num_hash_layers, 0);
    EXPECT_EQ(cfg.model.o_groups, 1);
    EXPECT_EQ(cfg.model.o_lora_rank, 0);
    EXPECT_EQ(cfg.model.hc_mult, 1);
    EXPECT_EQ(cfg.model.hc_sinkhorn_iters, 20);
    EXPECT_DOUBLE_EQ(cfg.model.hc_eps, 1e-6);
    EXPECT_EQ(cfg.model.gating_score_fn, GatingScoreFn::sigmoid);
}

TEST(ConfigParser, V4FieldsRoundTrip) {
    auto cfg = parse_config(v4_minimal_json());
    auto cfg2 = parse_config(config_to_json(cfg));
    EXPECT_EQ(cfg2.model.architecture, Architecture::deepseek_v4);
    EXPECT_EQ(cfg2.model.head_dim, cfg.model.head_dim);
    EXPECT_EQ(cfg2.model.compress_ratios, cfg.model.compress_ratios);
    EXPECT_DOUBLE_EQ(cfg2.model.compress_rope_theta, cfg.model.compress_rope_theta);
    EXPECT_EQ(cfg2.model.sliding_window, cfg.model.sliding_window);
    EXPECT_DOUBLE_EQ(cfg2.model.swiglu_limit, cfg.model.swiglu_limit);
    EXPECT_EQ(cfg2.model.num_hash_layers, cfg.model.num_hash_layers);
    EXPECT_EQ(cfg2.model.o_groups, cfg.model.o_groups);
    EXPECT_EQ(cfg2.model.o_lora_rank, cfg.model.o_lora_rank);
    EXPECT_EQ(cfg2.model.hc_mult, cfg.model.hc_mult);
    EXPECT_EQ(cfg2.model.hc_sinkhorn_iters, cfg.model.hc_sinkhorn_iters);
    EXPECT_DOUBLE_EQ(cfg2.model.hc_eps, cfg.model.hc_eps);
    EXPECT_EQ(cfg2.model.gating_score_fn, GatingScoreFn::sqrtsoftplus);
}

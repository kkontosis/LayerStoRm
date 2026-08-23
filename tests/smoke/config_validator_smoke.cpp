// Smoke test: validate_config() with real hardware topology.
//
// Expected hardware (dev machine):
//   GPU 0: RTX 5090, 31 GiB VRAM (CUDA-reported), PCIe 5.0 x16, NUMA 2
//   GPU 1: RTX 5090, 31 GiB VRAM (CUDA-reported), PCIe 5.0 x16, NUMA 3
//   GPU 2: RTX 5080, 15 GiB VRAM (CUDA-reported), PCIe 5.0 x16, NUMA 0
//   GPU 3: RTX 5080, 15 GiB VRAM (CUDA-reported), PCIe 5.0 x16, NUMA 2
//   TP array: GPU 0 + GPU 1 | System RAM: ~503 GiB
//
// Key property: the VRAM budget check must pass using CUDA-reported GiB
// values, not the marketed 32/16 GiB numbers.
//
// Build: cmake --build build --target config_validator_smoke
// Run:   ./build/tests/smoke/config_validator_smoke

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <iostream>
#include <string>

#include "config/config_resolver.h"
#include "config/config_validator.h"

namespace lc = layerstorm::config;

// ── Helpers ────────────────────────────────────────────────────────────────

static lc::Config build_resolved_config() {
    int count = 0;
    cudaGetDeviceCount(&count);

    lc::Config cfg;
    for (int i = 0; i < count; ++i) {
        lc::GpuConfig g;
        g.id = i;
        cfg.hardware.gpus.push_back(g);
    }
    lc::resolve_config(cfg);

    // Fill roles based on GPU type.
    for (auto& g : cfg.hardware.gpus) {
        if (g.type == lc::GpuType::rtx5090)
            g.roles = {lc::GpuRole::attention, lc::GpuRole::resident,
                       lc::GpuRole::expert_streaming};
        else
            g.roles = {lc::GpuRole::expert_streaming};
    }

    cfg.model.architecture = lc::Architecture::deepseek_v3;
    cfg.model.weights_path = "/data/models/deepseek-v3.2/";
    cfg.model.weights_format = lc::WeightsFormat::safetensors;
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

    cfg.quantization.weights = lc::WeightQuant::nvfp4;
    cfg.quantization.attention_compute = lc::AttentionQuant::fp8_e4m3;
    cfg.quantization.kv_cache = lc::KvCacheQuant::fp8_e4m3;
    cfg.quantization.gating_compute = lc::GatingQuant::fp32;

    cfg.memory.pinned_layers.attention = std::string{"all"};
    cfg.memory.pinned_layers.dense_ffn_layers = {0, 1, 2};
    cfg.memory.pinned_layers.embedding = true;
    cfg.memory.pinned_layers.output_head = true;
    cfg.memory.pinned_layers.gating = std::string{"all"};
    cfg.memory.tp_mode_per_layer.default_mode = 1;
    cfg.memory.tp_mode_per_layer.gating = lc::has_tp_array(cfg.hardware) ? 2 : 1;
    cfg.memory.tp_mode_per_layer.pinned_dense_ffn = lc::has_tp_array(cfg.hardware) ? 2 : 1;
    cfg.memory.tp_mode_per_layer.attention = 1;
    cfg.memory.kv_cache.max_pages_per_gpu = std::string{"auto"};
    cfg.memory.kv_cache.page_size_tokens = 16;
    cfg.memory.kv_cache.speculation_pool_fraction = 0.15;

    return cfg;
}

// ── Fixture ────────────────────────────────────────────────────────────────

class ConfigValidatorSmoke : public ::testing::Test {
   protected:
    void SetUp() override {
        std::cout << "\nExpected hardware (dev machine):\n"
                  << "  GPU 0: RTX 5090, 31 GiB (CUDA-reported), PCIe 5.0 x16, NUMA 2\n"
                  << "  GPU 1: RTX 5090, 31 GiB (CUDA-reported), PCIe 5.0 x16, NUMA 3\n"
                  << "  GPU 2: RTX 5080, 15 GiB (CUDA-reported), PCIe 5.0 x16, NUMA 0\n"
                  << "  GPU 3: RTX 5080, 15 GiB (CUDA-reported), PCIe 5.0 x16, NUMA 2\n"
                  << "  TP array: GPU 0 + GPU 1 | System RAM: ~503 GiB\n\n";

        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            GTEST_SKIP() << "No CUDA GPU present — cannot run hardware smoke test";

        cfg = build_resolved_config();

        std::cout << "Detected hardware (vram_gb from CUDA totalGlobalMem):\n";
        for (const auto& g : cfg.hardware.gpus) {
            std::cout << "  GPU " << g.id << ": "
                      << (g.type == lc::GpuType::rtx5090 ? "RTX 5090" : "RTX 5080")
                      << ", vram_gb=" << g.vram_gb << " GiB\n";
        }
        double total_gib = 0.0;
        for (const auto& g : cfg.hardware.gpus)
            total_gib += g.vram_gb;
        std::cout << "  Total: " << total_gib << " GiB\n\n";
    }

    lc::Config cfg;
};

// ── Tests ──────────────────────────────────────────────────────────────────

TEST_F(ConfigValidatorSmoke, ValidConfigPassesWithRealVram) {
    auto result = lc::validate_config(cfg);
    EXPECT_TRUE(result.valid())
        << "Unexpected errors with CUDA-reported VRAM values:\n"
        << [&] {
               std::string s;
               for (const auto& e : result.errors)
                   s += "  " + e.field + ": " + e.message + "\n";
               return s;
           }();
}

TEST_F(ConfigValidatorSmoke, NoVramBudgetError) {
    auto result = lc::validate_config(cfg);
    for (const auto& e : result.errors) {
        if (e.field.find("hardware.gpus") != std::string::npos &&
            e.message.find("VRAM") != std::string::npos) {
            FAIL() << "VRAM budget rejected with real hardware GiB values.\n"
                   << "  " << e.message << "\n"
                   << "Safety margin may need to increase to cover firmware overhead.";
        }
    }
}

TEST_F(ConfigValidatorSmoke, InsufficientVramRejected) {
    lc::Config tiny = cfg;
    tiny.hardware.gpus = {{0, lc::GpuType::rtx5090, 2.0, 5, 16, 0, 0.85, 0,
                           {lc::GpuRole::attention, lc::GpuRole::resident,
                            lc::GpuRole::expert_streaming}}};
    tiny.hardware.tp_array.clear();
    tiny.memory.tp_mode_per_layer.gating = 1;
    tiny.memory.tp_mode_per_layer.pinned_dense_ffn = 1;

    auto result = lc::validate_config(tiny);
    EXPECT_FALSE(result.valid());
    bool found = false;
    for (const auto& e : result.errors)
        if (e.field.find("hardware.gpus") != std::string::npos) found = true;
    EXPECT_TRUE(found) << "Expected VRAM budget error on a 2 GiB GPU";
}

TEST_F(ConfigValidatorSmoke, TpArrayValidOnRealHardware) {
    auto fives = lc::indices_by_type(cfg.hardware, lc::GpuType::rtx5090);
    if (fives.size() < 2)
        GTEST_SKIP() << "Need at least two RTX 5090s for TP array test";

    auto result = lc::validate_config(cfg);
    for (const auto& e : result.errors)
        EXPECT_EQ(e.field.find("tp_array"), std::string::npos)
            << "Unexpected TP array error: " << e.field << ": " << e.message;
}

TEST_F(ConfigValidatorSmoke, FourGpusDetected) {
    EXPECT_EQ(lc::gpu_count(cfg.hardware), 4)
        << "Expected 2x5090 + 2x5080; got " << lc::gpu_count(cfg.hardware) << " GPU(s)";
}

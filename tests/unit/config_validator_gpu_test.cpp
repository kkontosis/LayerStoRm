#include "../gpu_test_utils.h"

#include "config/config_resolver.h"
#include "config/config_validator.h"

using namespace layerstorm::config;

// ── validate_config with real hardware VRAM ──────────────────────────────
// Skipped on headless machines. Verifies the VRAM budget check accepts the
// CUDA-reported GiB values (e.g. 31.25 for RTX 5090, not the marketed 32).

TEST(ConfigValidatorGpu, VramBudgetPassesWithRealHardwareValues) {
    REQUIRES_GPU();

    // Build a skeleton config and resolve hardware fields.
    int count = 0;
    cudaGetDeviceCount(&count);

    Config cfg;
    for (int i = 0; i < count; ++i) {
        GpuConfig gc;
        gc.id = i;
        cfg.hardware.gpus.push_back(gc);
    }
    resolve_config(cfg);

    // Fill roles and remaining config fields.
    for (auto& g : cfg.hardware.gpus)
        g.roles = {GpuRole::attention, GpuRole::resident, GpuRole::expert_streaming};

    // Model: DeepSeek V3.2
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

    // Memory
    cfg.memory.pinned_layers.attention = std::string{"all"};
    cfg.memory.pinned_layers.dense_ffn_layers = {0, 1, 2};
    cfg.memory.pinned_layers.embedding = true;
    cfg.memory.pinned_layers.output_head = true;
    cfg.memory.pinned_layers.gating = std::string{"all"};
    cfg.memory.tp_mode_per_layer.default_mode = 1;
    cfg.memory.tp_mode_per_layer.gating = has_tp_array(cfg.hardware) ? 2 : 1;
    cfg.memory.tp_mode_per_layer.pinned_dense_ffn = has_tp_array(cfg.hardware) ? 2 : 1;
    cfg.memory.tp_mode_per_layer.attention = 1;
    cfg.memory.kv_cache.max_pages_per_gpu = std::string{"auto"};
    cfg.memory.kv_cache.page_size_tokens = 16;
    cfg.memory.kv_cache.speculation_pool_fraction = 0.15;

    auto result = validate_config(cfg);
    bool vram_error = false;
    for (const auto& e : result.errors)
        if (e.field.find("hardware.gpus") != std::string::npos &&
            e.message.find("VRAM") != std::string::npos)
            vram_error = true;
    EXPECT_FALSE(vram_error)
        << "VRAM budget rejected with real CUDA-reported GiB values — "
           "safety margin may be too tight";
}

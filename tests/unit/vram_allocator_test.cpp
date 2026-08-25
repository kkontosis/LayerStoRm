#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "config/config_resolver.h"
#include "core/device_backend.h"
#include "core/memory/vram_allocator.h"
#include "core/null_device_backend.h"
#include "model/quantization/fp8.h"
#include "model/quantization/nvfp4.h"

namespace lc = layerstorm::config;
namespace lcomp = layerstorm::compute;
namespace lmem = layerstorm::memory;
namespace lmod = layerstorm::model;

// ── Config helpers ──────────────────────────────────────────────────────────

namespace {

lc::Config v32_config() {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",            "deepseek_v3"},
            {"weights_path",            "/data/models/deepseek-v3.2/"},
            {"weights_format",          "safetensors"},
            {"num_hidden_layers",       61},
            {"hidden_size",             7168},
            {"num_attention_heads",     128},
            {"num_key_value_heads",     128},
            {"intermediate_size",       18432},
            {"n_routed_experts",        256},
            {"n_shared_experts",        1},
            {"num_experts_per_tok",     8},
            {"n_group",                 8},
            {"topk_group",              4},
            {"vocab_size",              129280},
            {"max_position_embeddings", 163840},
            {"kv_lora_rank",            512},
            {"q_lora_rank",             1536},
            {"qk_rope_head_dim",        64},
            {"qk_nope_head_dim",        128},
            {"v_head_dim",              128},
            {"first_k_dense_replace",   3},
            {"moe_layer_freq",          1},
            {"index_topk",              2048},
            {"index_n_heads",           64},
            {"index_head_dim",          128},
            {"num_nextn_predict_layers", 1},
            {"rms_norm_eps",            1e-6},
            {"rope_theta",              10000.0},
            {"routed_scaling_factor",   2.5},
            {"moe_intermediate_size",   2048},
        }},
        {"quantization", {{"weights", "nvfp4"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {
            {"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}},
                      {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 32}},
                      {{"id", 2}, {"type", "rtx5080"}, {"vram_gb", 16}},
                      {{"id", 3}, {"type", "rtx5080"}, {"vram_gb", 16}}}},
            {"tp_array", {0, 1}},
            {"system_ram_gb", 256}}},
    };
    return lc::parse_config(j);
}

// Minimal standard-attention model (no MLA, no MoE)
lc::Config dense_mha_config() {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",            "deepseek_v3"},
            {"weights_path",            "/data/models/test/"},
            {"weights_format",          "safetensors"},
            {"num_hidden_layers",       4},
            {"hidden_size",             256},
            {"num_attention_heads",     4},
            {"num_key_value_heads",     4},
            {"intermediate_size",       512},
            {"n_routed_experts",        0},
            {"n_shared_experts",        0},
            {"num_experts_per_tok",     0},
            {"n_group",                 1},
            {"topk_group",              1},
            {"vocab_size",              1024},
            {"max_position_embeddings", 2048},
            {"kv_lora_rank",            0},        // no MLA
            {"q_lora_rank",             0},
            {"qk_rope_head_dim",        32},
            {"qk_nope_head_dim",        32},
            {"v_head_dim",              64},
            {"first_k_dense_replace",   999},      // all dense
            {"moe_layer_freq",          1},
            {"index_topk",              0},
            {"num_nextn_predict_layers", 0},
            {"moe_intermediate_size",   512},
        }},
        {"quantization", {{"weights", "fp8_e4m3"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp16"}, {"gating_compute", "fp32"}}},
        {"hardware", {
            {"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
            {"system_ram_gb", 64}}},
    };
    return lc::parse_config(j);
}

}  // namespace

// ── kv_bytes_per_token ──────────────────────────────────────────────────────

TEST(VramAllocatorKv, MlaFp8BytesPerToken) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    // MLA: kv_lora_rank(512) * 1.0 (fp8) + qk_rope_head_dim(64) * 2.0 (bf16) + 4 (scale) = 644
    EXPECT_EQ(lmem::kv_bytes_per_token(mcfg, lc::KvCacheQuant::fp8_e4m3,
                                        lc::AttentionBackendType::snapmla), 644);
}

TEST(VramAllocatorKv, MlaFp16BytesPerToken) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    // MLA: 512 * 2.0 + 64 * 2.0 = 1152
    EXPECT_EQ(lmem::kv_bytes_per_token(mcfg, lc::KvCacheQuant::fp16,
                                        lc::AttentionBackendType::snapmla), 1152);
}

TEST(VramAllocatorKv, MlaFp8E5M2BytesPerToken) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    // Same as fp8_e4m3
    EXPECT_EQ(lmem::kv_bytes_per_token(mcfg, lc::KvCacheQuant::fp8_e5m2,
                                        lc::AttentionBackendType::snapmla), 644);
}

TEST(VramAllocatorKv, StandardAttentionFp16BytesPerToken) {
    auto cfg = dense_mha_config();
    lmod::ModelConfig mcfg(cfg);
    // Standard: 2 * num_kv_heads(4) * (32+32) * 2.0 (fp16) = 1024
    EXPECT_EQ(lmem::kv_bytes_per_token(mcfg, lc::KvCacheQuant::fp16,
                                        lc::AttentionBackendType::snapmla), 1024);
}

TEST(VramAllocatorKv, StandardAttentionFp8BytesPerToken) {
    auto cfg = dense_mha_config();
    lmod::ModelConfig mcfg(cfg);
    // Standard: 2 * 4 * 64 * 1.0 = 512
    EXPECT_EQ(lmem::kv_bytes_per_token(mcfg, lc::KvCacheQuant::fp8_e4m3,
                                        lc::AttentionBackendType::snapmla), 512);
}

// ── TurboQuant kv_bytes_per_token ───��──────────────────────────────────────

TEST(VramAllocatorKv, MlaTurboQuantBytesPerToken) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    // TQ: kv_lora_rank(512) * 0.5 + 2 (FP16 norm) + qk_rope_head_dim(64) * 2.0 = 386
    EXPECT_EQ(lmem::kv_bytes_per_token(mcfg, lc::KvCacheQuant::fp8_e4m3,
                                        lc::AttentionBackendType::turboquant_mla), 386);
}

TEST(VramAllocatorKv, TurboQuantModel1BytesPerToken) {
    // MODEL1-like config with kv_lora_rank=448
    auto cfg = v32_config();
    cfg.model.kv_lora_rank = 448;
    lmod::ModelConfig mcfg(cfg);
    // TQ: 448 * 0.5 + 2 + 64 * 2.0 = 354
    EXPECT_EQ(lmem::kv_bytes_per_token(mcfg, lc::KvCacheQuant::fp8_e4m3,
                                        lc::AttentionBackendType::turboquant_mla), 354);
}

TEST(VramAllocatorKv, TurboQuantIgnoresKvQuant) {
    // TQ formula is independent of kv_cache quant setting (always 4-bit packed)
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    EXPECT_EQ(lmem::kv_bytes_per_token(mcfg, lc::KvCacheQuant::fp16,
                                        lc::AttentionBackendType::turboquant_mla), 386);
    EXPECT_EQ(lmem::kv_bytes_per_token(mcfg, lc::KvCacheQuant::fp8_e5m2,
                                        lc::AttentionBackendType::turboquant_mla), 386);
}

TEST(VramAllocatorKv, StandardMhaIgnoresBackend) {
    auto cfg = dense_mha_config();
    lmod::ModelConfig mcfg(cfg);
    // Standard MHA (kv_lora_rank=0): backend doesn't matter
    EXPECT_EQ(lmem::kv_bytes_per_token(mcfg, lc::KvCacheQuant::fp8_e4m3,
                                        lc::AttentionBackendType::snapmla),
              lmem::kv_bytes_per_token(mcfg, lc::KvCacheQuant::fp8_e4m3,
                                        lc::AttentionBackendType::turboquant_mla));
}

// ── kv_bytes_per_page ───────────────────────────────────────────────────────

TEST(VramAllocatorKv, V32BytesPerPage16) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    // 644 * 16 = 10304
    EXPECT_EQ(lmem::kv_bytes_per_page(mcfg, cfg), 10304);
}

TEST(VramAllocatorKv, BytesPerPage1Token) {
    auto cfg = v32_config();
    cfg.memory.kv_cache.page_size_tokens = 1;
    lmod::ModelConfig mcfg(cfg);
    EXPECT_EQ(lmem::kv_bytes_per_page(mcfg, cfg), 644);
}

TEST(VramAllocatorKv, TurboQuantV32BytesPerPage16) {
    auto cfg = v32_config();
    cfg.compute.attention_backend = lc::AttentionBackendType::turboquant_mla;
    lmod::ModelConfig mcfg(cfg);
    // 386 * 16 = 6176
    EXPECT_EQ(lmem::kv_bytes_per_page(mcfg, cfg), 6176);
}

TEST(VramAllocatorKv, TurboQuantMoreKvPagesThanSnapMla) {
    // Same VRAM budget: TQ gets more KV pages than SnapMLA
    auto cfg_snap = v32_config();
    auto cfg_tq = v32_config();
    cfg_tq.compute.attention_backend = lc::AttentionBackendType::turboquant_mla;

    lmod::ModelConfig mcfg_snap(cfg_snap);
    lmod::ModelConfig mcfg_tq(cfg_tq);
    lmod::Nvfp4 nvfp4_snap, nvfp4_tq;
    lmod::LayerRegistry reg_snap(mcfg_snap, cfg_snap, nvfp4_snap);
    lmod::LayerRegistry reg_tq(mcfg_tq, cfg_tq, nvfp4_tq);

    auto layout_snap = lmem::compute_vram_layout(cfg_snap, reg_snap, mcfg_snap);
    auto layout_tq = lmem::compute_vram_layout(cfg_tq, reg_tq, mcfg_tq);

    // TQ pages are smaller → more pages fit
    EXPECT_LT(layout_tq.kv_bytes_per_page, layout_snap.kv_bytes_per_page);
    // TP GPUs should get more KV pages with TQ
    for (size_t i = 0; i < layout_snap.gpus.size(); ++i) {
        if (layout_snap.gpus[i].kv_main_pages > 0) {
            EXPECT_GT(layout_tq.gpus[i].kv_main_pages,
                      layout_snap.gpus[i].kv_main_pages)
                << "GPU " << layout_snap.gpus[i].gpu_id;
        }
    }
}

// ── Budget calculation: regions sum to total_vram ───────────────────────────

TEST(VramAllocatorBudget, RegionsSumToTotal) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    ASSERT_EQ(layout.gpus.size(), 4u);

    for (const auto& gpu : layout.gpus) {
        int64_t sum = gpu.pinned_bytes + gpu.kv_main_bytes +
                      gpu.kv_speculation_bytes + gpu.indexer_k_bytes +
                      gpu.expert_stable_bytes +
                      gpu.expert_streaming_bytes + gpu.safety_margin_bytes;
        EXPECT_EQ(sum, gpu.total_vram_bytes)
            << "GPU " << gpu.gpu_id << ": sum=" << sum
            << " total=" << gpu.total_vram_bytes;
    }
}

// ── TP=2 pinned distribution ────────────────────────────────────────────────

TEST(VramAllocatorBudget, TpGpusPinnedNonTpUnpinned) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);

    // GPUs 0,1 (TP) should have pinned bytes
    EXPECT_GT(layout.gpus[0].pinned_bytes, 0);
    EXPECT_GT(layout.gpus[1].pinned_bytes, 0);
    EXPECT_EQ(layout.gpus[0].pinned_bytes, layout.gpus[1].pinned_bytes);

    // GPUs 2,3 (non-TP) should have pinned=0
    EXPECT_EQ(layout.gpus[2].pinned_bytes, 0);
    EXPECT_EQ(layout.gpus[3].pinned_bytes, 0);
}

// ── Safety margin ───────────────────────────────────────────────────────────

TEST(VramAllocatorBudget, SafetyMarginFromConfig) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    int64_t expected = lc::vram_gb_to_bytes(cfg.memory.vram_safety_margin_gb);
    for (const auto& gpu : layout.gpus) {
        EXPECT_EQ(gpu.safety_margin_bytes, expected);
    }

    // Custom value
    cfg.memory.vram_safety_margin_gb = 2.0;
    auto layout2 = lmem::compute_vram_layout(cfg, reg, mcfg);
    int64_t expected2 = lc::vram_gb_to_bytes(2.0);
    for (const auto& gpu : layout2.gpus) {
        EXPECT_EQ(gpu.safety_margin_bytes, expected2);
    }
}

// ── KV page counts ──────────────────────────────────────────────────────────

TEST(VramAllocatorBudget, KvPageCountsConsistent) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    for (const auto& gpu : layout.gpus) {
        EXPECT_GE(gpu.max_kv_pages, 0);
        // Page counts derived from byte allocations via floor division,
        // so sum may be ≤ max_kv_pages (off-by-one from speculation floor).
        EXPECT_LE(gpu.kv_main_pages + gpu.kv_speculation_pages,
                  gpu.max_kv_pages)
            << "GPU " << gpu.gpu_id;
        EXPECT_GE(gpu.kv_main_pages + gpu.kv_speculation_pages,
                  gpu.max_kv_pages - 1)
            << "GPU " << gpu.gpu_id;
        // Page count * bytes_per_page <= actual byte allocation
        EXPECT_LE(static_cast<int64_t>(gpu.kv_main_pages) * layout.kv_bytes_per_page,
                  gpu.kv_main_bytes)
            << "GPU " << gpu.gpu_id;
    }
}

// ── TD-GOLDEN-KV-SPEC: auto KV sizing counts MTP layers ────────────────────

TEST(VramAllocatorBudget, AutoKvPagesIncludeMtpLayers) {
    // The dispatcher allocates one physical page per (logical page, layer)
    // INCLUDING the MTP layer (INV-KV-LAYER) — auto sizing must too. With a
    // small model the request is not VRAM-capped, so max_kv_pages scales
    // exactly with the layer count: (4 + 1) / 4.
    auto cfg = dense_mha_config();
    lmod::ModelConfig mcfg0(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg0(mcfg0, cfg, fp8);
    auto layout0 = lmem::compute_vram_layout(cfg, reg0, mcfg0);
    ASSERT_GT(layout0.gpus[0].max_kv_pages, 0);
    ASSERT_EQ(layout0.gpus[0].max_kv_pages % 4, 0);  // not VRAM-capped

    cfg.model.num_nextn_predict_layers = 1;
    lmod::ModelConfig mcfg1(cfg);
    lmod::LayerRegistry reg1(mcfg1, cfg, fp8);
    auto layout1 = lmem::compute_vram_layout(cfg, reg1, mcfg1);

    EXPECT_EQ(layout1.gpus[0].max_kv_pages,
              layout0.gpus[0].max_kv_pages / 4 * 5);
}

// ── Speculation pool fraction ───────────────────────────────────────────────

TEST(VramAllocatorBudget, SpeculationPoolFraction) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    double spec_frac = cfg.memory.kv_cache.speculation_pool_fraction;

    // Region sizes are 256-byte aligned (SPEC_UPDATES line 73).
    auto align256 = [](int64_t sz) { return (sz + 255) & ~int64_t{255}; };

    for (const auto& gpu : layout.gpus) {
        int64_t kv_total = gpu.kv_total_bytes();
        if (kv_total > 0) {
            // Spec fraction applies to KV data (excluding scratch), then aligned
            int64_t kv_data = kv_total - gpu.prefill_scratch_preallocated_bytes;
            int64_t expected_spec = align256(static_cast<int64_t>(
                std::floor(static_cast<double>(kv_data) * spec_frac)));
            EXPECT_EQ(gpu.kv_speculation_bytes, expected_spec)
                << "GPU " << gpu.gpu_id;
        }
    }
}

// ── Stable/streaming zone split ─────────────────────────────────────────────

TEST(VramAllocatorBudget, StableStreamingZoneSplit) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    double stable_frac = cfg.memory.expert_cache.stable_zone_fraction;

    // Streaming is 256-aligned; stable absorbs the alignment cost.
    auto align256 = [](int64_t sz) { return (sz + 255) & ~int64_t{255}; };

    for (const auto& gpu : layout.gpus) {
        int64_t expert_total = gpu.expert_total_bytes();
        if (expert_total > 0) {
            int64_t raw_stable = static_cast<int64_t>(
                std::floor(static_cast<double>(expert_total) * stable_frac));
            int64_t expected_streaming = align256(expert_total - raw_stable);
            int64_t expected_stable = expert_total - expected_streaming;
            EXPECT_EQ(gpu.expert_stable_bytes, expected_stable)
                << "GPU " << gpu.gpu_id;
            EXPECT_EQ(gpu.expert_streaming_bytes, expected_streaming)
                << "GPU " << gpu.gpu_id;
        }
    }
}

// ── Explicit max_pages_per_gpu ──────────────────────────────────────────────

TEST(VramAllocatorBudget, ExplicitMaxPages) {
    auto cfg = v32_config();
    cfg.memory.kv_cache.max_pages_per_gpu = 100;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    // TP GPUs (0,1) get the explicit page count; non-TP GPUs (2,3) get 0
    for (int i : {0, 1}) {
        EXPECT_EQ(layout.gpus[i].max_kv_pages, 100) << "TP GPU " << i;
    }
    for (int i : {2, 3}) {
        EXPECT_EQ(layout.gpus[i].max_kv_pages, 0) << "Non-TP GPU " << i;
    }
}

// ── Auto max_pages_per_gpu capped by VRAM ───────────────────────────────────

TEST(VramAllocatorBudget, AutoPagesCapByVram) {
    auto cfg = v32_config();
    // Set very high sequence length and concurrency to exceed VRAM
    cfg.serving.max_sequence_length = 1000000;
    cfg.serving.max_concurrent_requests = 100;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    for (const auto& gpu : layout.gpus) {
        // KV bytes should not exceed total - pinned - safety
        int64_t max_kv = gpu.total_vram_bytes - gpu.pinned_bytes -
                         gpu.safety_margin_bytes;
        EXPECT_LE(gpu.kv_total_bytes(), max_kv)
            << "GPU " << gpu.gpu_id;
    }
}

// ── Insufficient VRAM throws ────────────────────────────────────────────────

TEST(VramAllocatorBudget, InsufficientVramThrows) {
    auto cfg = v32_config();
    // Tiny GPUs that can't fit pinned + safety + min expert cache
    for (auto& g : cfg.hardware.gpus) {
        g.vram_gb = 0.5;  // 512 MB — way too small
    }
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    EXPECT_THROW(lmem::compute_vram_layout(cfg, reg, mcfg), std::runtime_error);
}

// ── Per-GPU resident override ───────────────────────────────────────────────

TEST(VramAllocatorOverride, ResidentOverride) {
    auto cfg = v32_config();
    // Override GPU 0's pinned bytes to 20 GB (well above computed layout).
    cfg.hardware.gpus[0].vram_allocation_gb = lc::VramAllocationConfig{};
    cfg.hardware.gpus[0].vram_allocation_gb->resident = 20.0;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    // Override is large enough to be accepted without clamping.
    EXPECT_GE(layout.gpus[0].pinned_bytes, lc::vram_gb_to_bytes(20.0));
    // GPU 1 still gets its normal LayerRegistry value
    EXPECT_NE(layout.gpus[1].pinned_bytes, lc::vram_gb_to_bytes(20.0));
}

TEST(VramAllocatorOverride, ResidentOverrideClampsToLayout) {
    // TD-54c: override smaller than computed pinned layout is clamped up.
    auto cfg = v32_config();
    cfg.hardware.gpus[0].vram_allocation_gb = lc::VramAllocationConfig{};
    cfg.hardware.gpus[0].vram_allocation_gb->resident = 0.001;  // ~1 MB, way below layout
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    // The override is too small — clamped to at least what GPU 1 (no override) gets.
    EXPECT_EQ(layout.gpus[0].pinned_bytes, layout.gpus[1].pinned_bytes);
}

// ── Per-GPU expert_streaming override ───────────────────────────────────────

TEST(VramAllocatorOverride, ExpertStreamingOverride) {
    auto cfg = v32_config();
    // Override GPU 2's expert cache to 5 GB
    cfg.hardware.gpus[2].vram_allocation_gb = lc::VramAllocationConfig{};
    cfg.hardware.gpus[2].vram_allocation_gb->expert_streaming = 5.0;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    int64_t expected = lc::vram_gb_to_bytes(5.0);
    EXPECT_EQ(layout.gpus[2].expert_total_bytes(), expected);
}

// ── Per-GPU stable_zone_fraction override ───────────────────────────────────

TEST(VramAllocatorOverride, StableZoneFractionOverride) {
    auto cfg = v32_config();
    cfg.hardware.gpus[3].vram_allocation_gb = lc::VramAllocationConfig{};
    cfg.hardware.gpus[3].vram_allocation_gb->stable_zone_fraction = 0.5;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    // Streaming is 256-aligned; stable absorbs the alignment cost.
    auto align256 = [](int64_t sz) { return (sz + 255) & ~int64_t{255}; };

    // GPU 3 should use 0.5, other GPUs use global default (0.7)
    int64_t et = layout.gpus[3].expert_total_bytes();
    int64_t raw_stable = static_cast<int64_t>(std::floor(et * 0.5));
    int64_t expected_stable = et - align256(et - raw_stable);
    EXPECT_EQ(layout.gpus[3].expert_stable_bytes, expected_stable);

    // GPU 0 uses global 0.7
    int64_t et0 = layout.gpus[0].expert_total_bytes();
    int64_t raw_stable0 = static_cast<int64_t>(std::floor(et0 * 0.7));
    int64_t expected_stable0 = et0 - align256(et0 - raw_stable0);
    EXPECT_EQ(layout.gpus[0].expert_stable_bytes, expected_stable0);
}

TEST(VramAllocatorOverride, StableZoneFractionClampedToRange) {
    // TD-91c: values outside [0.0, 1.0] are clamped.
    auto cfg = v32_config();
    // Global config: set to something out of range (schema prevents this,
    // but code-level clamp should still handle it).
    cfg.memory.expert_cache.stable_zone_fraction = 1.5;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    // Clamped to 1.0 → all expert cache is stable, streaming = 0.
    for (auto& gpu : layout.gpus) {
        EXPECT_GE(gpu.expert_stable_bytes, 0) << "GPU " << gpu.gpu_id;
        EXPECT_GE(gpu.expert_streaming_bytes, 0) << "GPU " << gpu.gpu_id;
    }
}

// ── Single GPU, no TP ───────────────────────────────────────────────────────

TEST(VramAllocatorBudget, SingleGpuNoTp) {
    auto cfg = v32_config();
    cfg.hardware.gpus = {{0, lc::GpuType::rtx5090, 32.0}};
    cfg.hardware.tp_array = {};
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    ASSERT_EQ(layout.gpus.size(), 1u);
    auto& gpu = layout.gpus[0];

    // Regions sum to total
    int64_t sum = gpu.pinned_bytes + gpu.kv_main_bytes +
                  gpu.kv_speculation_bytes + gpu.indexer_k_bytes +
                  gpu.expert_stable_bytes +
                  gpu.expert_streaming_bytes + gpu.safety_margin_bytes;
    EXPECT_EQ(sum, gpu.total_vram_bytes);
}

// ── All-dense model: zero expert cache ──────────────────────────────────────

TEST(VramAllocatorBudget, AllDenseModelNoExpertMinimum) {
    auto cfg = dense_mha_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    ASSERT_EQ(layout.gpus.size(), 1u);

    // No MoE experts: expert cache may be whatever is left, no minimum
    // Just verify regions sum to total and nothing crashes
    auto& gpu = layout.gpus[0];
    int64_t sum = gpu.pinned_bytes + gpu.kv_main_bytes +
                  gpu.kv_speculation_bytes + gpu.indexer_k_bytes +
                  gpu.expert_stable_bytes +
                  gpu.expert_streaming_bytes + gpu.safety_margin_bytes;
    EXPECT_EQ(sum, gpu.total_vram_bytes);
}

// ── KV bytes per page stored in layout ──────────────────────────────────────

TEST(VramAllocatorBudget, KvBytesPerPageStored) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    EXPECT_EQ(layout.kv_bytes_per_page, 10304);
}

// ── V3.2 realistic: 4 GPUs, ~11GB pinned per 5090 ──────────────────────────

TEST(VramAllocatorRealistic, V32FourGpuPinnedSizes) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);

    // With TP+DCP, per-GPU pinned bytes > total/2 because MLA replicated
    // projections (q_a, kv_a, norms, DSA indexer) are NOT halved by TP.
    // Both TP GPUs should get the same (larger) amount.
    EXPECT_EQ(layout.gpus[0].pinned_bytes, layout.gpus[1].pinned_bytes);
    EXPECT_GT(layout.gpus[0].pinned_bytes, reg.total_pinned_bytes() / 2);

    // 5090s still have room for expert cache
    EXPECT_GT(layout.gpus[0].expert_total_bytes(), 0);
    EXPECT_GT(layout.gpus[1].expert_total_bytes(), 0);

    // 5080s have full VRAM minus safety for expert cache
    EXPECT_GT(layout.gpus[2].expert_total_bytes(), 0);
    EXPECT_GT(layout.gpus[3].expert_total_bytes(), 0);
}

// ── Non-negative region bytes ───────────────────────────────────────────────

TEST(VramAllocatorBudget, AllRegionsNonNegative) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    for (const auto& gpu : layout.gpus) {
        EXPECT_GE(gpu.pinned_bytes, 0) << "GPU " << gpu.gpu_id;
        EXPECT_GE(gpu.kv_main_bytes, 0) << "GPU " << gpu.gpu_id;
        EXPECT_GE(gpu.kv_speculation_bytes, 0) << "GPU " << gpu.gpu_id;
        EXPECT_GE(gpu.expert_stable_bytes, 0) << "GPU " << gpu.gpu_id;
        EXPECT_GE(gpu.expert_streaming_bytes, 0) << "GPU " << gpu.gpu_id;
        EXPECT_GE(gpu.safety_margin_bytes, 0) << "GPU " << gpu.gpu_id;
    }
}

// ── KV total bytes helper ───────────────────────────────────────────────────

TEST(VramAllocatorBudget, KvTotalBytesHelper) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    for (const auto& gpu : layout.gpus) {
        EXPECT_EQ(gpu.kv_total_bytes(),
                  gpu.kv_main_bytes + gpu.kv_speculation_bytes);
    }
}

// ── Expert total bytes helper ───────────────────────────────────────────────

TEST(VramAllocatorBudget, ExpertTotalBytesHelper) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    for (const auto& gpu : layout.gpus) {
        EXPECT_EQ(gpu.expert_total_bytes(),
                  gpu.expert_stable_bytes + gpu.expert_streaming_bytes);
    }
}

// ── Zero speculation fraction ───────────────────────────────────────────────

TEST(VramAllocatorBudget, ZeroSpeculationFraction) {
    auto cfg = v32_config();
    cfg.memory.kv_cache.speculation_pool_fraction = 0.0;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    for (const auto& gpu : layout.gpus) {
        EXPECT_EQ(gpu.kv_speculation_bytes, 0) << "GPU " << gpu.gpu_id;
        EXPECT_EQ(gpu.kv_speculation_pages, 0) << "GPU " << gpu.gpu_id;
        EXPECT_EQ(gpu.kv_main_bytes, gpu.kv_total_bytes()) << "GPU " << gpu.gpu_id;
    }
}

// ── Explicit pages: large enough to exceed VRAM → clamped ───────────────────

TEST(VramAllocatorBudget, ExplicitPagesClampedByVram) {
    // Use all-dense model (no expert minimum) so clamping doesn't throw
    auto cfg = dense_mha_config();
    cfg.memory.kv_cache.max_pages_per_gpu = 999999999;  // absurd
    lmod::ModelConfig mcfg(cfg);
    lmod::Fp8E4M3 fp8;
    lmod::LayerRegistry reg(mcfg, cfg, fp8);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    for (const auto& gpu : layout.gpus) {
        // KV total clamped to available (with up to 512 bytes alignment overhead
        // from 256-byte region alignment of kv_speculation + kv_main)
        int64_t available = gpu.total_vram_bytes - gpu.pinned_bytes -
                            gpu.safety_margin_bytes;
        EXPECT_LE(gpu.kv_total_bytes(), available + 512)
            << "GPU " << gpu.gpu_id;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VramAllocator class tests (using heap backend — no CUDA needed)
// ═══════════════════════════════════════════════════════════════════════════

namespace {

/// Build NullDeviceBackend instances for a VramLayout.
/// Must outlive any VramAllocator created from these pointers.
struct NullBackends {
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned;
    std::vector<lcomp::DeviceBackend*> ptrs;

    explicit NullBackends(const lmem::VramLayout& layout) {
        for (size_t i = 0; i < layout.gpus.size(); ++i) {
            lc::GpuRef ref{.position = static_cast<int>(i),
                           .id = layout.gpus[i].gpu_id};
            owned.push_back(lcomp::make_null_device_backend(ref));
            ptrs.push_back(owned.back().get());
        }
    }

    explicit NullBackends(int num_gpus) {
        for (int i = 0; i < num_gpus; ++i) {
            lc::GpuRef ref{.position = i, .id = i};
            owned.push_back(lcomp::make_null_device_backend(ref));
            ptrs.push_back(owned.back().get());
        }
    }
};

/// RAII wrapper: backends + VramAllocator (backends must outlive allocator).
/// Provides VramAllocator-like accessors for convenience.
struct TestVramAllocator {
    NullBackends backends;
    lmem::VramAllocator alloc;

    TestVramAllocator(lmem::VramLayout layout, NullBackends backends_)
        : backends(std::move(backends_)),
          alloc(std::move(layout), backends.ptrs) {}

    // Convenience accessors (delegate to alloc)
    int gpu_count() const { return alloc.gpu_count(); }
    bool owns_memory() const { return alloc.owns_memory(); }
    const lmem::GpuRegion& region(int i) const { return alloc.region(i); }
    const lmem::VramLayout& layout() const { return alloc.layout(); }
};

TestVramAllocator make_heap_allocator(const lc::Config& cfg) {
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    NullBackends nb(layout);
    return TestVramAllocator(std::move(layout), std::move(nb));
}

}  // namespace

// ── Construction allocates, destruction frees ────────────────────────────────

TEST(VramAllocatorClass, ConstructAllocatesAllGpus) {
    auto alloc = make_heap_allocator(v32_config());
    EXPECT_EQ(alloc.gpu_count(), 4);
    EXPECT_TRUE(alloc.owns_memory());

    for (int i = 0; i < alloc.gpu_count(); ++i) {
        const auto& reg = alloc.region(i);
        EXPECT_NE(reg.base, nullptr) << "GPU " << i;
        EXPECT_GT(reg.allocated_bytes, 0) << "GPU " << i;
    }
}

// ── Region pointers are contiguous offsets into base ─────────────────────────

TEST(VramAllocatorClass, RegionsContiguous) {
    auto alloc = make_heap_allocator(v32_config());

    for (int i = 0; i < alloc.gpu_count(); ++i) {
        const auto& reg = alloc.region(i);
        const auto& gpu = alloc.layout().gpus[i];
        auto* base = static_cast<char*>(reg.base);

        // Layout: pinned | kv_speculation | indexer_k | kv_main (+scratch) | expert_streaming | expert_stable
        EXPECT_EQ(reg.pinned, base);
        EXPECT_EQ(reg.kv_speculation, base + gpu.pinned_bytes);
        EXPECT_EQ(reg.indexer_k,
                  base + gpu.pinned_bytes + gpu.kv_speculation_bytes);
        EXPECT_EQ(reg.kv_main,
                  base + gpu.pinned_bytes + gpu.kv_speculation_bytes +
                  gpu.indexer_k_bytes);
        EXPECT_EQ(reg.expert_streaming,
                  base + gpu.pinned_bytes + gpu.kv_speculation_bytes +
                  gpu.indexer_k_bytes + gpu.kv_main_bytes);
        EXPECT_EQ(reg.expert_stable,
                  base + gpu.pinned_bytes + gpu.kv_speculation_bytes +
                  gpu.indexer_k_bytes + gpu.kv_main_bytes +
                  gpu.expert_streaming_bytes);
    }
}

// ── Allocated bytes = total - safety margin ──────────────────────────────────

TEST(VramAllocatorClass, AllocatedBytesMatchLayout) {
    auto alloc = make_heap_allocator(v32_config());

    for (int i = 0; i < alloc.gpu_count(); ++i) {
        const auto& reg = alloc.region(i);
        const auto& gpu = alloc.layout().gpus[i];
        EXPECT_EQ(reg.allocated_bytes,
                  gpu.total_vram_bytes - gpu.safety_margin_bytes);
    }
}

// ── Regions are writable (heap backend) ──────────────────────────────────────

TEST(VramAllocatorClass, RegionsWritable) {
    auto alloc = make_heap_allocator(v32_config());

    for (int i = 0; i < alloc.gpu_count(); ++i) {
        const auto& reg = alloc.region(i);
        // Write first and last byte of each region
        auto write_byte = [](void* ptr, int64_t size, uint8_t val) {
            if (size > 0 && ptr) {
                static_cast<uint8_t*>(ptr)[0] = val;
                static_cast<uint8_t*>(ptr)[size - 1] = val;
            }
        };
        const auto& gpu = alloc.layout().gpus[i];
        write_byte(reg.pinned, gpu.pinned_bytes, 0xAA);
        write_byte(reg.kv_main, gpu.kv_main_bytes, 0xBB);
        write_byte(reg.expert_stable, gpu.expert_stable_bytes, 0xCC);
    }
}

// ── Move constructor transfers ownership ─────────────────────────────────────

TEST(VramAllocatorClass, MoveConstructor) {
    auto ctx = make_heap_allocator(v32_config());
    void* original_base = ctx.region(0).base;

    lmem::VramAllocator moved(std::move(ctx.alloc));

    EXPECT_TRUE(moved.owns_memory());
    EXPECT_EQ(moved.gpu_count(), 4);
    EXPECT_EQ(moved.region(0).base, original_base);

    // Source is empty after move
    EXPECT_FALSE(ctx.alloc.owns_memory());
    EXPECT_EQ(ctx.alloc.gpu_count(), 0);
}

// ── Move assignment transfers ownership ──────────────────────────────────────

TEST(VramAllocatorClass, MoveAssignment) {
    auto alloc1 = make_heap_allocator(v32_config());
    void* base1 = alloc1.region(0).base;

    auto cfg2 = dense_mha_config();
    auto alloc2 = make_heap_allocator(cfg2);

    // Move-assign: alloc2's old memory freed, alloc1's memory transferred
    alloc2.alloc = std::move(alloc1.alloc);
    EXPECT_EQ(alloc2.region(0).base, base1);
    EXPECT_EQ(alloc2.gpu_count(), 4);
    EXPECT_FALSE(alloc1.owns_memory());
}

// ── Layout accessible after construction ─────────────────────────────────────

TEST(VramAllocatorClass, LayoutPreserved) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    int64_t expected_kv_per_page = layout.kv_bytes_per_page;
    int expected_gpus = static_cast<int>(layout.gpus.size());

    NullBackends nb(layout);
    lmem::VramAllocator alloc(std::move(layout), nb.ptrs);

    EXPECT_EQ(alloc.layout().kv_bytes_per_page, expected_kv_per_page);
    EXPECT_EQ(static_cast<int>(alloc.layout().gpus.size()), expected_gpus);
}

// ── Single GPU allocation ────────────────────────────────────────────────────

TEST(VramAllocatorClass, SingleGpu) {
    auto cfg = v32_config();
    cfg.hardware.gpus = {{0, lc::GpuType::rtx5090, 32.0}};
    cfg.hardware.tp_array = {};

    auto alloc = make_heap_allocator(cfg);
    EXPECT_EQ(alloc.gpu_count(), 1);
    EXPECT_NE(alloc.region(0).base, nullptr);
}

// ── Alloc tracking: track total allocated per backend call ───────────────────

TEST(VramAllocatorClass, TrackAllocFreeCallCount) {
    int alloc_count = 0;
    int free_count = 0;

    // Counting backend that records alloc/free calls.
    struct CountingBackend : lcomp::NullDeviceBackend {
        int& alloc_cnt;
        int& free_cnt;
        CountingBackend(lc::GpuRef gpu, int& ac, int& fc)
            : NullDeviceBackend(gpu), alloc_cnt(ac), free_cnt(fc) {}
        void* device_alloc(size_t bytes) override {
            ++alloc_cnt;
            return NullDeviceBackend::device_alloc(bytes);
        }
        void device_free(void* ptr) override {
            if (ptr) ++free_cnt;
            NullDeviceBackend::device_free(ptr);
        }
    };

    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);

    // Build counting backends
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned;
    std::vector<lcomp::DeviceBackend*> ptrs;
    for (size_t i = 0; i < layout.gpus.size(); ++i) {
        lc::GpuRef ref{.position = static_cast<int>(i),
                       .id = layout.gpus[i].gpu_id};
        owned.push_back(std::make_unique<CountingBackend>(
            ref, alloc_count, free_count));
        ptrs.push_back(owned.back().get());
    }

    {
        lmem::VramAllocator alloc(std::move(layout), ptrs);

        // One allocation per GPU
        EXPECT_EQ(alloc_count, 4);
        EXPECT_EQ(free_count, 0);
    }
    // Destructor frees all
    EXPECT_EQ(free_count, 4);
}

// ── Non-TP GPU: pinned region is zero-size but pointer still valid ───────────

TEST(VramAllocatorClass, NonTpGpuZeroPinnedRegion) {
    auto alloc = make_heap_allocator(v32_config());
    // GPUs 2,3 are non-TP: pinned_bytes=0, kv=0, indexer_k=0
    const auto& gpu2 = alloc.layout().gpus[2];
    EXPECT_EQ(gpu2.pinned_bytes, 0);

    // Non-TP: pinned,kv_speculation,indexer_k,kv_main all collapsed at base
    const auto& reg2 = alloc.region(2);
    EXPECT_EQ(reg2.pinned, reg2.base);
    EXPECT_EQ(reg2.kv_speculation, reg2.base);
    EXPECT_EQ(reg2.indexer_k, reg2.base);
    EXPECT_EQ(reg2.kv_main, reg2.base);
}

// ── TP-aware KV cache allocation ───────────────────────────────────────────

TEST(VramAllocatorBudget, NonTpGpuNoKvCache) {
    auto cfg = v32_config();  // TP=[0,1], 4 GPUs
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);

    // GPUs 2,3 are non-TP: no KV cache (they don't run attention)
    for (int i : {2, 3}) {
        EXPECT_EQ(layout.gpus[i].kv_main_bytes, 0)
            << "Non-TP GPU " << i << " should have kv_main_bytes=0";
        EXPECT_EQ(layout.gpus[i].kv_speculation_bytes, 0)
            << "Non-TP GPU " << i << " should have kv_speculation_bytes=0";
        EXPECT_EQ(layout.gpus[i].max_kv_pages, 0)
            << "Non-TP GPU " << i << " should have max_kv_pages=0";
    }
}

TEST(VramAllocatorBudget, TpGpuHasKvCache) {
    auto cfg = v32_config();  // TP=[0,1], 4 GPUs
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);

    // GPUs 0,1 are TP: should have KV cache
    for (int i : {0, 1}) {
        EXPECT_GT(layout.gpus[i].kv_main_bytes, 0)
            << "TP GPU " << i << " should have kv_main_bytes>0";
        EXPECT_GT(layout.gpus[i].max_kv_pages, 0)
            << "TP GPU " << i << " should have max_kv_pages>0";
    }
}

// ── Indexer K region ─────────────────────────────────────────────────────────

TEST(VramAllocatorIndexerK, ZeroForNonDsa) {
    auto cfg = v32_config();
    // Make it non-DSA by setting index_topk=0
    cfg.model.index_topk = 0;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    for (const auto& gpu : layout.gpus) {
        EXPECT_EQ(gpu.indexer_k_bytes, 0) << "GPU " << gpu.gpu_id;
        EXPECT_EQ(gpu.indexer_k_pages, 0) << "GPU " << gpu.gpu_id;
    }
    EXPECT_EQ(layout.indexer_k_bytes_per_page, 0);
}

TEST(VramAllocatorIndexerK, NonZeroForDsa) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    // TP GPUs (0,1) should have indexer K
    EXPECT_GT(layout.gpus[0].indexer_k_bytes, 0);
    EXPECT_GT(layout.gpus[0].indexer_k_pages, 0);
    EXPECT_GT(layout.gpus[1].indexer_k_bytes, 0);
    // Non-TP GPUs (2,3) should not have indexer K
    EXPECT_EQ(layout.gpus[2].indexer_k_bytes, 0);
    EXPECT_EQ(layout.gpus[3].indexer_k_bytes, 0);
    EXPECT_GT(layout.indexer_k_bytes_per_page, 0);
}

TEST(VramAllocatorIndexerK, IndexerKBytesPerToken) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    // V3.2: 1 head * 128 dim * 1.0 (FP8) + 4 B F32 per-token absmax scale
    // (TD-GLM-INDEXER-PAGED page layout) = 132 bytes/token/layer.
    EXPECT_EQ(lmem::indexer_k_bytes_per_token(mcfg, lc::KvCacheQuant::fp8_e4m3), 132);
}

// INV-KVT-14 (GLM-25k 1M capacity smoke): the kIndexerK pool is provisioned
// only for indexer-COMPUTING layers — IndexShare full ∪ layer 0, the exact
// ensure_indexer_pages rule. SHARED layers (and MTP layers, shared by
// construction) reuse the preceding full layer's selection and never
// allocate a page, so sizing them wastes VRAM (~3.8× on GLM-5.2: 10.9 GB vs
// 2.9 GB per TP GPU at max_sequence_length=1M). Non-IndexShare DSA models
// (index_topk_freq<=0) keep the historical all-hidden-layers count.
TEST(VramAllocatorIndexerK, IndexShareAwareSizing) {
    lmod::Nvfp4 nvfp4;

    // Baseline: V3.2, no IndexShare (freq<=0) — every hidden layer computes,
    // MTP layer never does. pages_per_seq = ceil(32768 / 8192) = 4.
    auto base = v32_config();
    base.serving.max_concurrent_requests = 1;   // per-seq shape only
    lmod::ModelConfig mcfg_all(base);
    lmod::LayerRegistry reg_all(mcfg_all, base, nvfp4);
    auto layout_all = lmem::compute_vram_layout(base, reg_all, mcfg_all);
    ASSERT_EQ(layout_all.gpus[0].indexer_k_pages % base.model.num_hidden_layers, 0);
    const int pages_per_seq =
        layout_all.gpus[0].indexer_k_pages / base.model.num_hidden_layers;
    ASSERT_GT(pages_per_seq, 0);

    // IndexShare (GLM-5.2 pattern): freq=4, offset=3 → full layers are
    // {0,1,2} ∪ {6,10,...,58} = 17 of 61 hidden; the MTP layer (61) is
    // shared by construction.
    auto shared = v32_config();
    shared.serving.max_concurrent_requests = 1;  // per-seq shape only
    shared.model.index_topk_freq = 4;
    shared.model.index_skip_topk_offset = 3;
    lmod::ModelConfig mcfg_sh(shared);
    lmod::LayerRegistry reg_sh(mcfg_sh, shared, nvfp4);
    auto layout_sh = lmem::compute_vram_layout(shared, reg_sh, mcfg_sh);

    int computing = 0;
    const int n_idx_layers = shared.model.num_hidden_layers
                           + shared.model.num_nextn_predict_layers;
    for (int l = 0; l < n_idx_layers; ++l)
        if (mcfg_sh.is_full_index_layer(l) || l == 0) ++computing;
    EXPECT_EQ(computing, 17);
    EXPECT_EQ(layout_sh.gpus[0].indexer_k_pages, pages_per_seq * computing);
    EXPECT_LT(layout_sh.gpus[0].indexer_k_bytes,
              layout_all.gpus[0].indexer_k_bytes);
}

// INV-KVT-14b / TD-INDEXER-POOL-EVICT: the kIndexerK pool must cover
// serving.max_concurrent_requests CONCURRENT sequences, not one. Sizing it
// for a single sequence made it a hard per-sequence wall — serving always
// has more than one live sequence (every prefix-cache holder is one, pinning
// a CoW frontier page group of num_dsa_layers pages), so the second live
// sequence exhausted the pool, silently downgraded a forked long prefix to
// dense, and fail-closed the request through the tiering re-promotion lift
// (2026-08-24 serving incident).
TEST(VramAllocatorIndexerK, PoolScalesWithMaxConcurrentRequests) {
    lmod::Nvfp4 nvfp4;

    auto one = v32_config();
    one.serving.max_concurrent_requests = 1;
    lmod::ModelConfig mcfg1(one);
    lmod::LayerRegistry reg1(mcfg1, one, nvfp4);
    auto layout1 = lmem::compute_vram_layout(one, reg1, mcfg1);
    const int64_t pages1 = layout1.gpus[0].indexer_k_pages;
    ASSERT_GT(pages1, 0);

    for (int max_req : {2, 4, 8}) {   // all below this config's VRAM cap
        auto cfg = v32_config();
        cfg.serving.max_concurrent_requests = max_req;
        lmod::ModelConfig mcfg(cfg);
        lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
        auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
        EXPECT_EQ(layout.gpus[0].indexer_k_pages, pages1 * max_req)
            << "max_concurrent_requests=" << max_req;
        // Bytes track pages (region alignment only rounds UP).
        EXPECT_GE(layout.gpus[0].indexer_k_bytes,
                  layout1.gpus[0].indexer_k_bytes * max_req)
            << "max_concurrent_requests=" << max_req;
    }

    // The per-sequence shape is unchanged: pages/seq × computing layers.
    auto cfg2 = v32_config();
    cfg2.serving.max_concurrent_requests = 2;
    lmod::ModelConfig mcfg2(cfg2);
    lmod::LayerRegistry reg2(mcfg2, cfg2, nvfp4);
    auto layout2 = lmem::compute_vram_layout(cfg2, reg2, mcfg2);
    const int pages_per_seq =
        (cfg2.serving.max_sequence_length
         + cfg2.memory.kv_cache.indexer_k_page_size_tokens - 1)
        / cfg2.memory.kv_cache.indexer_k_page_size_tokens;
    int computing = 0;
    const int n_idx = cfg2.model.num_hidden_layers
                    + cfg2.model.num_nextn_predict_layers;
    for (int l = 0; l < n_idx; ++l)
        if (mcfg2.is_full_index_layer(l) || l == 0) ++computing;
    EXPECT_EQ(layout2.gpus[0].indexer_k_pages,
              static_cast<int64_t>(pages_per_seq) * computing * 2);
}

// The concurrency multiple is BOUNDED by what the KV pool can spare: the
// indexer pool is carved BEFORE KV, so every indexer byte is taken 1:1 from
// the residual. An aspirational max_concurrent_requests must never drive that
// residual to zero (measured 2026-08-25 on the champion box: an unbounded
// `* max_concurrent_requests` took all 86 MiB that GPU 2 had left and gave
// kv_main ZERO pages). At most a quarter of the one-sequence residual, whole
// sequences only, one sequence as the hard floor.
TEST(VramAllocatorIndexerK, PoolConcurrencyCappedByVramHeadroom) {
    lmod::Nvfp4 nvfp4;
    auto cfg = v32_config();
    cfg.serving.max_concurrent_requests = 4096;   // absurd on purpose
    lmod::ModelConfig mcfg(cfg);
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);

    const auto& gpu = layout.gpus[0];
    ASSERT_GT(gpu.indexer_k_pages, 0);
    // THE regression guard: KV still gets pages. This is what an unbounded
    // multiple destroyed.
    EXPECT_GT(gpu.max_kv_pages, 0)
        << "indexer pool starved the KV residual it is carved out of";
    EXPECT_GT(gpu.kv_main_bytes, 0);

    // Floor: never below one sequence, and always a WHOLE number of them.
    auto one = v32_config();
    one.serving.max_concurrent_requests = 1;
    lmod::ModelConfig mcfg1(one);
    lmod::LayerRegistry reg1(mcfg1, one, nvfp4);
    auto layout1 = lmem::compute_vram_layout(one, reg1, mcfg1);
    ASSERT_GT(layout1.gpus[0].indexer_k_pages, 0);
    EXPECT_GE(gpu.indexer_k_pages, layout1.gpus[0].indexer_k_pages);
    EXPECT_EQ(gpu.indexer_k_pages % layout1.gpus[0].indexer_k_pages, 0);
    // ... and the one-sequence carve is what KV was measured against, so the
    // capped pool must not have cost KV more than the quarter it may spend.
    EXPECT_GE(gpu.max_kv_pages, layout1.gpus[0].max_kv_pages * 3 / 4);

    // TP ranks stay SYMMETRIC: replicated indexer pages are claimed in
    // lockstep, so a richer rank must not be given pages the tightest rank
    // cannot match (they would be unusable).
    for (const auto& g : layout.gpus)
        if (g.indexer_k_pages > 0)
            EXPECT_EQ(g.indexer_k_pages, gpu.indexer_k_pages)
                << "GPU " << g.gpu_id;
}

// ── DCP sharding ────────────────────────────────────────────────────────────

// TD-KV-REP-POOL-HALVED / INV-KV-SIZE-SHARD: DCP with the opt-in
// dcp_kv_mode=replicated must NOT divide auto KV pages — each TP GPU claims
// the same page_idx in lockstep (INV-KV-REP), holding FULL pages. Only
// dcp_kv_mode=sharded (the DEFAULT) splits page demand across ranks.
TEST(VramAllocatorDcp, DcpReplicatedKvKeepsFullPages) {
    auto cfg = v32_config();
    cfg.hardware.dcp_enabled = false;  // explicit baseline: no DCP
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout_no_dcp = lmem::compute_vram_layout(cfg, reg, mcfg);
    int kv_no_dcp = layout_no_dcp.gpus[0].max_kv_pages;

    cfg.hardware.dcp_enabled = true;
    cfg.hardware.dcp_kv_mode = lc::DcpKvMode::replicated;  // opt-in mode
    auto layout_dcp = lmem::compute_vram_layout(cfg, reg, mcfg);
    int kv_dcp = layout_dcp.gpus[0].max_kv_pages;

    // Replicated KV: full page count — identical to the no-DCP layout.
    EXPECT_GT(kv_no_dcp, 0);
    EXPECT_EQ(kv_dcp, kv_no_dcp);
    EXPECT_EQ(layout_dcp.gpus[0].dcp_kv_shard_factor, 1);

    // Sharded KV keeps the per-rank division (this config is VRAM-bound, so
    // the sharded per-rank demand is still above the cap — the pool may use
    // the full available VRAM per rank; see ShardedKvPoolVramBound below).
    cfg.hardware.dcp_kv_mode = lc::DcpKvMode::sharded;
    auto layout_sh = lmem::compute_vram_layout(cfg, reg, mcfg);
    EXPECT_EQ(layout_sh.gpus[0].dcp_kv_shard_factor, 2);
}

// ── Auto KV sizing vs dcp_kv_mode (TD-KV-REP-POOL-HALVED / INV-KV-SIZE-SHARD)

namespace {
// Region sum must equal total VRAM on every GPU — the budget invariant
// (KV pool + expert stable/streaming + pinned + indexer-K + safety margin
// exactly tile usable VRAM; prefill scratch lives inside kv_main).
void expect_regions_sum_to_total(const lmem::VramLayout& layout,
                                 const char* tag) {
    for (const auto& gpu : layout.gpus) {
        int64_t sum = gpu.pinned_bytes + gpu.kv_main_bytes +
                      gpu.kv_speculation_bytes + gpu.indexer_k_bytes +
                      gpu.expert_stable_bytes +
                      gpu.expert_streaming_bytes + gpu.safety_margin_bytes;
        EXPECT_EQ(sum, gpu.total_vram_bytes)
            << tag << ": GPU " << gpu.gpu_id << " sum=" << sum
            << " total=" << gpu.total_vram_bytes;
    }
}
}  // namespace

// Demand-bound config (not VRAM-capped): replicated KV gets the FULL auto
// page count (max_req × ceil(max_seq/page) × kv_layers); sharded KV gets the
// per-rank split. Budget still balances in both modes.
TEST(VramAllocatorBudget, KvPoolReplicatedFullShardedSplit) {
    auto cfg = v32_config();
    cfg.serving.max_concurrent_requests = 1;  // demand-bound: ~1.3 GB of KV
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    const int page = cfg.memory.kv_cache.page_size_tokens;
    const int pages_per_seq =
        (cfg.serving.max_sequence_length + page - 1) / page;  // 2048
    const int kv_layers = cfg.model.num_hidden_layers +
                          cfg.model.num_nextn_predict_layers;  // 62

    // DCP enabled (default), dcp_kv_mode=replicated (opt-in): each TP GPU
    // holds full pages in lockstep (INV-KV-REP) — full demand, NOT halved.
    ASSERT_TRUE(cfg.hardware.dcp_enabled);
    ASSERT_EQ(cfg.hardware.dcp_kv_mode, lc::DcpKvMode::sharded);  // schema dflt
    cfg.hardware.dcp_kv_mode = lc::DcpKvMode::replicated;
    auto layout_rep = lmem::compute_vram_layout(cfg, reg, mcfg);
    EXPECT_EQ(layout_rep.gpus[0].max_kv_pages, pages_per_seq * kv_layers);
    EXPECT_EQ(layout_rep.gpus[1].max_kv_pages, pages_per_seq * kv_layers);
    expect_regions_sum_to_total(layout_rep, "replicated");

    // Sharded: round-robin-by-chunk ownership splits page demand across the
    // 2 DCP ranks. dcp_chunk_size=16 = 1 page/chunk, 2048 even → exactly half.
    cfg.hardware.dcp_kv_mode = lc::DcpKvMode::sharded;
    auto layout_sh = lmem::compute_vram_layout(cfg, reg, mcfg);
    EXPECT_EQ(layout_sh.gpus[0].max_kv_pages,
              (pages_per_seq / 2) * kv_layers);
    EXPECT_EQ(layout_sh.gpus[0].max_kv_pages,
              layout_rep.gpus[0].max_kv_pages / 2);
    expect_regions_sum_to_total(layout_sh, "sharded");

    // Non-TP GPUs never hold KV in either mode.
    for (int i : {2, 3}) {
        EXPECT_EQ(layout_rep.gpus[i].max_kv_pages, 0) << "GPU " << i;
        EXPECT_EQ(layout_sh.gpus[i].max_kv_pages, 0) << "GPU " << i;
    }
}

// Sharded per-rank demand is sized for the WORST-CASE rank at chunk
// granularity (rank 0 owns ceil), not a floor divide of the total — a
// floor divide would underprovision rank 0 for odd page counts (same
// rationale as the indexer-local ceil, TD-GLM-INDEXER-LOCAL-MERGE).
TEST(VramAllocatorBudget, KvPoolShardedCeilsAtChunkGranularity) {
    auto cfg = v32_config();
    cfg.serving.max_concurrent_requests = 1;
    cfg.serving.max_sequence_length = 3 * cfg.memory.kv_cache.page_size_tokens;
    cfg.hardware.dcp_kv_mode = lc::DcpKvMode::sharded;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    const int kv_layers = cfg.model.num_hidden_layers +
                          cfg.model.num_nextn_predict_layers;  // 62

    // 3 pages/seq, chunk = 1 page, dcp_size = 2: rank 0 owns pages {0, 2}
    // → 2 pages, not floor(3/2) = 1.
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    EXPECT_EQ(layout.gpus[0].max_kv_pages, 2 * kv_layers);
    expect_regions_sum_to_total(layout, "sharded-odd");
}

// VRAM-bound config: the cap applies AFTER the per-rank shard division, so a
// sharded pool still uses the full available VRAM per rank (each rank's
// pages hold only its owned tokens — that is the dcp_size× context capacity
// Helix sharding provides). Replicated is capped identically.
TEST(VramAllocatorBudget, KvPoolVramBoundBothModes) {
    auto cfg = v32_config();
    cfg.serving.max_sequence_length = 1000000;
    cfg.serving.max_concurrent_requests = 100;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout_rep = lmem::compute_vram_layout(cfg, reg, mcfg);
    cfg.hardware.dcp_kv_mode = lc::DcpKvMode::sharded;
    auto layout_sh = lmem::compute_vram_layout(cfg, reg, mcfg);

    EXPECT_GT(layout_rep.gpus[0].max_kv_pages, 0);
    EXPECT_EQ(layout_sh.gpus[0].max_kv_pages,
              layout_rep.gpus[0].max_kv_pages)
        << "VRAM-capped pool must be identical across kv modes";
    for (const auto* layout : {&layout_rep, &layout_sh}) {
        for (const auto& gpu : layout->gpus) {
            int64_t max_kv = gpu.total_vram_bytes - gpu.pinned_bytes -
                             gpu.safety_margin_bytes;
            EXPECT_LE(gpu.kv_total_bytes(), max_kv) << "GPU " << gpu.gpu_id;
        }
    }
    expect_regions_sum_to_total(layout_rep, "vram-bound replicated");
    expect_regions_sum_to_total(layout_sh, "vram-bound sharded");
}

TEST(VramAllocatorDcp, DcpDoesNotAffectIndexerK) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout_no_dcp = lmem::compute_vram_layout(cfg, reg, mcfg);

    cfg.hardware.dcp_enabled = true;
    auto layout_dcp = lmem::compute_vram_layout(cfg, reg, mcfg);

    // Indexer K should be identical (replicated mode is default)
    EXPECT_EQ(layout_no_dcp.gpus[0].indexer_k_bytes,
              layout_dcp.gpus[0].indexer_k_bytes);
    EXPECT_EQ(layout_no_dcp.gpus[0].indexer_k_pages,
              layout_dcp.gpus[0].indexer_k_pages);
}

TEST(VramAllocatorDcp, DcpIndexerModeReplicated) {
    // Explicit replicated mode gives same result as default (no DCP effect on indexer K)
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    // Baseline: no DCP
    auto layout_no_dcp = lmem::compute_vram_layout(cfg, reg, mcfg);

    // DCP enabled with explicit replicated mode
    cfg.hardware.dcp_enabled = true;
    cfg.hardware.dcp_indexer_mode = lc::DcpIndexerMode::replicated;
    auto layout_dcp = lmem::compute_vram_layout(cfg, reg, mcfg);

    // Indexer K should be identical — replicated mode keeps full pages per GPU
    for (int i : {0, 1}) {
        EXPECT_EQ(layout_no_dcp.gpus[i].indexer_k_bytes,
                  layout_dcp.gpus[i].indexer_k_bytes)
            << "TP GPU " << i << ": replicated mode should match no-DCP indexer K";
        EXPECT_EQ(layout_no_dcp.gpus[i].indexer_k_pages,
                  layout_dcp.gpus[i].indexer_k_pages)
            << "TP GPU " << i;
    }
}

TEST(VramAllocatorDcp, DcpIndexerModeLocal) {
    // Local mode with DCP enabled halves indexer K pages (dcp_shard_factor=2)
    auto cfg = v32_config();
    // Per-SEQUENCE shape only: pin the concurrency multiple (INV-KVT-14b) to
    // one so the VRAM cap on the pool cannot bind differently between the two
    // arms (local pages/seq are half, so an equal cap would buy local MORE
    // sequences and break the 2:1 comparison).
    cfg.serving.max_concurrent_requests = 1;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    // Baseline: DCP enabled, replicated (default)
    cfg.hardware.dcp_enabled = true;
    cfg.hardware.dcp_indexer_mode = lc::DcpIndexerMode::replicated;
    auto layout_rep = lmem::compute_vram_layout(cfg, reg, mcfg);

    // Local mode
    cfg.hardware.dcp_indexer_mode = lc::DcpIndexerMode::local;
    auto layout_local = lmem::compute_vram_layout(cfg, reg, mcfg);

    // TP GPUs should have halved indexer K pages and bytes
    for (int i : {0, 1}) {
        EXPECT_GT(layout_rep.gpus[i].indexer_k_pages, 0) << "TP GPU " << i;
        EXPECT_GT(layout_local.gpus[i].indexer_k_pages, 0) << "TP GPU " << i;

        // pages_per_seq is halved (integer division), so indexer_k_pages should be ~half
        // With TP=2, dcp_shard_factor=2: local pages = replicated pages / 2
        int expected_pages = layout_rep.gpus[i].indexer_k_pages / 2;
        EXPECT_EQ(layout_local.gpus[i].indexer_k_pages, expected_pages)
            << "TP GPU " << i << ": local mode should halve indexer K pages";

        int64_t expected_bytes = static_cast<int64_t>(expected_pages) *
                                 layout_rep.indexer_k_bytes_per_page;
        EXPECT_EQ(layout_local.gpus[i].indexer_k_bytes, expected_bytes)
            << "TP GPU " << i << ": local mode should halve indexer K bytes";
    }

    // Non-TP GPUs should still have zero indexer K
    for (int i : {2, 3}) {
        EXPECT_EQ(layout_local.gpus[i].indexer_k_bytes, 0) << "Non-TP GPU " << i;
        EXPECT_EQ(layout_local.gpus[i].indexer_k_pages, 0) << "Non-TP GPU " << i;
    }

    // Regions should still sum to total VRAM
    for (const auto& gpu : layout_local.gpus) {
        int64_t sum = gpu.pinned_bytes + gpu.kv_main_bytes +
                      gpu.kv_speculation_bytes + gpu.indexer_k_bytes +
                      gpu.expert_stable_bytes +
                      gpu.expert_streaming_bytes + gpu.safety_margin_bytes;
        EXPECT_EQ(sum, gpu.total_vram_bytes)
            << "GPU " << gpu.gpu_id << ": regions must sum to total VRAM";
    }
}

// ── Contiguity: kv_main end == expert_streaming start ────────────────────────

TEST(VramAllocatorClass, KvMainAdjacentToStreaming) {
    auto alloc = make_heap_allocator(v32_config());

    for (int i = 0; i < alloc.gpu_count(); ++i) {
        const auto& reg = alloc.region(i);
        const auto& gpu = alloc.layout().gpus[i];
        if (gpu.kv_main_bytes > 0 || gpu.expert_streaming_bytes > 0) {
            auto* kv_main_end = static_cast<char*>(reg.kv_main) + gpu.kv_main_bytes;
            EXPECT_EQ(kv_main_end, reg.expert_streaming)
                << "GPU " << i << ": kv_main end must be contiguous with expert_streaming start";
        }
    }
}

// ── Expert stable is last region ────────────────────────────────────────────

TEST(VramAllocatorClass, StableZoneIsLast) {
    auto alloc = make_heap_allocator(v32_config());

    for (int i = 0; i < alloc.gpu_count(); ++i) {
        const auto& reg = alloc.region(i);
        const auto& gpu = alloc.layout().gpus[i];
        auto* stable_end = static_cast<char*>(reg.expert_stable) + gpu.expert_stable_bytes;
        auto* alloc_end = static_cast<char*>(reg.base) + reg.allocated_bytes;
        EXPECT_EQ(stable_end, alloc_end) << "GPU " << i;
    }
}

// ── Streaming spill + prefetch == streaming total ────────────────────────────

TEST(VramAllocatorBudget, StreamingSpillPlusPrefetchEqualsTotal) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    for (const auto& gpu : layout.gpus) {
        EXPECT_EQ(gpu.streaming_spill_bytes + gpu.streaming_prefetch_bytes,
                  gpu.expert_streaming_bytes) << "GPU " << gpu.gpu_id;
    }
}

// ── Prefill scratch present on TP GPUs ───────────────────────────────────────

TEST(VramAllocatorBudget, PrefillScratchOnTpGpus) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    // TP GPUs get prefill scratch
    EXPECT_GT(layout.gpus[0].prefill_scratch_preallocated_bytes, 0);
    EXPECT_GT(layout.gpus[1].prefill_scratch_preallocated_bytes, 0);
    // Non-TP GPUs don't
    EXPECT_EQ(layout.gpus[2].prefill_scratch_preallocated_bytes, 0);
    EXPECT_EQ(layout.gpus[3].prefill_scratch_preallocated_bytes, 0);
}

// ── TP-aware KV cache allocation ───────────────────────────────────────────

TEST(VramAllocatorBudget, NoTpAllGpusGetKv) {
    auto cfg = v32_config();
    // Clear TP array — all GPUs should get KV cache
    cfg.hardware.tp_array.clear();

    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);

    for (size_t i = 0; i < layout.gpus.size(); ++i) {
        EXPECT_GT(layout.gpus[i].kv_main_bytes, 0)
            << "GPU " << i << " should have kv_main_bytes>0 when no TP configured";
        EXPECT_GT(layout.gpus[i].max_kv_pages, 0)
            << "GPU " << i << " should have max_kv_pages>0 when no TP configured";
    }
}

// ── naive_prefill_scratch_bytes ─────────────────────────────────────────────

TEST(VramAllocatorNaiveScratch, ComputesDecompressionBufferSize) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    // kv_b_proj decompression: 128 heads × (128+128) dim × seq_len × 2 (BF16)
    int seq = 4096;
    int64_t expected = static_cast<int64_t>(128) * 256 * seq * 2;
    EXPECT_EQ(lmem::naive_prefill_scratch_bytes(mcfg, seq), expected);
}

TEST(VramAllocatorNaiveScratch, ZeroForZeroSeqLen) {
    auto cfg = v32_config();
    lmod::ModelConfig mcfg(cfg);
    EXPECT_EQ(lmem::naive_prefill_scratch_bytes(mcfg, 0), 0);
}

// ── Ceiling-aware scratch sizing ────────────────────────────────────────────

TEST(VramAllocatorCeiling, NonDsaCeilingShrinksScratch) {
    // Non-DSA model with a low ceiling should have smaller scratch
    auto cfg = v32_config();
    cfg.model.index_topk = 0;  // Non-DSA
    cfg.serving.max_sequence_length = 131072;
    cfg.memory.kv_cache.prefill_scratch_preallocated_gb = 2.0;

    // No ceiling — scratch capped by naive scratch for max_seq
    lmod::ModelConfig mcfg_no(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg_no(mcfg_no, cfg, nvfp4);
    auto layout_no = lmem::compute_vram_layout(cfg, reg_no, mcfg_no);

    // With ceiling at 4096 tokens
    cfg.memory.kv_cache.naive_prefill_context_ceiling = 4096;
    lmod::ModelConfig mcfg_ceil(cfg);
    lmod::LayerRegistry reg_ceil(mcfg_ceil, cfg, nvfp4);
    auto layout_ceil = lmem::compute_vram_layout(cfg, reg_ceil, mcfg_ceil);

    // TP GPU scratch should be smaller with ceiling
    EXPECT_LT(layout_ceil.gpus[0].prefill_scratch_preallocated_bytes,
              layout_no.gpus[0].prefill_scratch_preallocated_bytes)
        << "Ceiling should reduce pre-allocated scratch for non-DSA";
}

TEST(VramAllocatorCeiling, NonDsaCeilingReducesSpillZone) {
    auto cfg = v32_config();
    cfg.model.index_topk = 0;  // Non-DSA
    cfg.serving.max_sequence_length = 131072;
    cfg.memory.kv_cache.prefill_scratch_preallocated_gb = 0.5;

    // With very low ceiling, Tier 1 covers all scratch, spill zone = 0
    cfg.memory.kv_cache.naive_prefill_context_ceiling = 1024;
    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);

    for (const auto& gpu : layout.gpus) {
        if (gpu.kv_main_bytes > 0) {
            // Naive scratch at 1024 tokens:
            // 128 × 256 × 1024 × 2 = 67 MB < 0.5 GB Tier 1
            // So spill_scratch_needed = 0, spill zone = 0
            EXPECT_EQ(gpu.streaming_spill_bytes, 0)
                << "GPU " << gpu.gpu_id
                << ": spill zone should be 0 when ceiling scratch fits in Tier 1";
            // All streaming goes to prefetch
            EXPECT_EQ(gpu.streaming_prefetch_bytes, gpu.expert_streaming_bytes);
        }
    }
}

// TD-KV-DCP-SCRATCH-CEILING (verified) / INV-KV-NAIVE-SCRATCH: the DCP
// division of the naive scratch ceiling is correct under BOTH dcp_kv_mode
// settings — per-rank naive decompression demand is (H × T) / factor in each:
//   - replicated KV: kv_b_proj is column-TP-sharded (INV-MLA-1), so a rank
//     decompresses only its H/tp head shard, over the FULL sequence
//     (INV-KV-REP all-ranks residency);
//   - sharded KV: post Q-allgather (INV-KVS-QAG) a rank attends ALL heads,
//     over its T/dcp token shard.
// Both equal naive_prefill_scratch_bytes(full-H, ceiling/dcp_shard_factor)
// while dcp_size == tp_degree.
TEST(VramAllocatorCeiling, DcpDividesNaiveScratchInBothKvModes) {
    auto cfg = v32_config();
    cfg.model.index_topk = 0;                 // non-DSA (naive scratch path)
    cfg.serving.max_sequence_length = 131072;
    cfg.serving.max_concurrent_requests = 1;  // demand-bound KV pool
    cfg.memory.kv_cache.naive_prefill_context_ceiling = 4096;
    cfg.memory.kv_cache.prefill_scratch_preallocated_gb = 2.0;  // cap binds

    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);

    // Baseline: no DCP — full-ceiling scratch.
    cfg.hardware.dcp_enabled = false;
    auto layout_no = lmem::compute_vram_layout(cfg, reg, mcfg);
    EXPECT_EQ(layout_no.gpus[0].prefill_scratch_preallocated_bytes,
              lmem::naive_prefill_scratch_bytes(mcfg, 4096));

    // DCP + replicated KV (opt-in mode): divided — head-sharded demand.
    cfg.hardware.dcp_enabled = true;
    cfg.hardware.dcp_kv_mode = lc::DcpKvMode::replicated;
    auto layout_rep = lmem::compute_vram_layout(cfg, reg, mcfg);
    EXPECT_EQ(layout_rep.gpus[0].prefill_scratch_preallocated_bytes,
              lmem::naive_prefill_scratch_bytes(mcfg, 4096 / 2));

    // DCP + sharded KV: SAME divided size — token-sharded demand.
    cfg.hardware.dcp_kv_mode = lc::DcpKvMode::sharded;
    auto layout_sh = lmem::compute_vram_layout(cfg, reg, mcfg);
    EXPECT_EQ(layout_sh.gpus[0].prefill_scratch_preallocated_bytes,
              layout_rep.gpus[0].prefill_scratch_preallocated_bytes);
}

TEST(VramAllocatorCeiling, DsaUnaffectedByCeiling) {
    // DSA model — ceiling should not reduce scratch or spill zone
    auto cfg = v32_config();
    // V3.2 is already DSA (index_topk=2048)

    lmod::ModelConfig mcfg_no(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg_no(mcfg_no, cfg, nvfp4);
    auto layout_no = lmem::compute_vram_layout(cfg, reg_no, mcfg_no);

    cfg.memory.kv_cache.naive_prefill_context_ceiling = 512;
    lmod::ModelConfig mcfg_ceil(cfg);
    lmod::LayerRegistry reg_ceil(mcfg_ceil, cfg, nvfp4);
    auto layout_ceil = lmem::compute_vram_layout(cfg, reg_ceil, mcfg_ceil);

    // DSA: scratch and spill zone should be identical regardless of ceiling
    EXPECT_EQ(layout_ceil.gpus[0].prefill_scratch_preallocated_bytes,
              layout_no.gpus[0].prefill_scratch_preallocated_bytes)
        << "DSA scratch should not change with naive ceiling";
    EXPECT_EQ(layout_ceil.gpus[0].streaming_spill_bytes,
              layout_no.gpus[0].streaming_spill_bytes)
        << "DSA spill zone should not change with naive ceiling";
}

TEST(VramAllocatorCeiling, CeilingZeroMeansNeverNaive) {
    // ceiling=0 → naive scratch needed = 0, all scratch eliminated for non-DSA
    auto cfg = v32_config();
    cfg.model.index_topk = 0;  // Non-DSA
    cfg.memory.kv_cache.naive_prefill_context_ceiling = 0;
    cfg.memory.kv_cache.prefill_scratch_preallocated_gb = 0.5;

    lmod::ModelConfig mcfg(cfg);
    lmod::Nvfp4 nvfp4;
    lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);

    for (const auto& gpu : layout.gpus) {
        if (gpu.kv_main_bytes > 0) {
            EXPECT_EQ(gpu.prefill_scratch_preallocated_bytes, 0)
                << "GPU " << gpu.gpu_id << ": ceiling=0 means zero scratch";
            EXPECT_EQ(gpu.streaming_spill_bytes, 0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DeepSeek-V4 (V4-3b): 3-tier KV layout
// ═══════════════════════════════════════════════════════════════════════════

#include "model/quantization/gguf_kquant.h"

namespace {

// Mirrors test-data/config/deepseek_v4_flash_gguf.json: 2 SWA + 21 CSA +
// 20 HCA layers, 32k max_seq, 32 concurrent requests.
lc::Config v4_config(const std::string& backend = "csa_hca") {
    std::vector<int> ratios{0, 0};
    for (int l = 2; l < 43; ++l) ratios.push_back(l % 2 == 0 ? 4 : 128);
    auto j = nlohmann::json{
        {"model", {
            {"architecture",           "deepseek_v4"},
            {"weights_path",           "/data/models/deepseek-v4-flash.gguf"},
            {"weights_format",         "gguf"},
            {"num_hidden_layers",      43},
            {"hidden_size",            4096},
            {"num_attention_heads",    64},
            {"num_key_value_heads",    1},
            {"head_dim",               512},
            {"qk_rope_head_dim",       64},
            {"q_lora_rank",            1024},
            {"compress_ratios",        ratios},
            {"compress_rope_theta",    160000.0},
            {"sliding_window",         128},
            {"intermediate_size",      2048},
            {"n_routed_experts",       256},
            {"n_shared_experts",       1},
            {"num_experts_per_tok",    6},
            {"n_group",                1},
            {"topk_group",             1},
            {"vocab_size",             129280},
            {"max_position_embeddings", 1048576},
            {"rope_theta",             10000.0},
            {"rms_norm_eps",           1e-6},
            {"num_nextn_predict_layers", 1},
            {"first_k_dense_replace",  0},
            {"routed_scaling_factor",  1.5},
            {"moe_intermediate_size",  2048},
            {"moe_layer_freq",         1},
            {"gating_score_fn",        "sqrtsoftplus"},
            {"swiglu_limit",           10.0},
            {"num_hash_layers",        3},
            {"o_groups",               8},
            {"o_lora_rank",            1024},
            {"hc_mult",                4},
            {"index_topk",             512},
            {"index_n_heads",          64},
            {"index_head_dim",         128},
        }},
        {"quantization", {{"weights", "gguf"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp16"}}},
        {"compute", {{"attention_backend", backend}}},
        {"parallelism", {{"tensor_parallelism", 1}}},
        {"memory", {{"kv_cache", {
            {"page_size_tokens", 16},
            {"speculation_pool_fraction", 0.15},
            {"indexer_k_page_size_tokens", 8192},
        }}}},
        {"serving", {{"max_concurrent_requests", 32},
                     {"max_sequence_length", 32768}}},
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                      {"tp_array", {0}},
                      {"system_ram_gb", 512}}},
    };
    return lc::parse_config(j);
}

lmem::VramLayout v4_layout(const lc::Config& cfg) {
    lmod::ModelConfig mcfg(cfg);
    lmod::GgufQuantInterface mxfp4{lmod::GgufKQuantType::MXFP4};
    lmod::LayerRegistry reg(mcfg, cfg, mxfp4);
    return lmem::compute_vram_layout(cfg, reg, mcfg);
}

}  // namespace

TEST(VramAllocatorV4, TierGeometryFp8) {
    auto cfg = v4_config("csa_hca");
    lmod::ModelConfig mcfg(cfg);
    auto v4 = lmem::compute_v4_kv_layout(cfg, mcfg);

    EXPECT_TRUE(v4.enabled);
    EXPECT_EQ(v4.logical_block_tokens, 256);
    EXPECT_EQ(v4.csa_entries_per_page, 64);
    EXPECT_EQ(v4.hca_entries_per_page, 2);
    EXPECT_EQ(v4.swa_page_tokens, 128);

    EXPECT_EQ(v4.csa_entry_bytes, 1160);   // deps csa_fp8/params.h V4CacheLayout
    EXPECT_EQ(v4.hca_entry_bytes, 1160);
    EXPECT_EQ(v4.swa_entry_bytes, 1160);   // SWA tier shares the V4 entry format
    EXPECT_EQ(v4.csa_bytes_per_page, 64 * 1160);   // 74240
    EXPECT_EQ(v4.hca_bytes_per_page, 2 * 1160);    // 2320
    EXPECT_EQ(v4.swa_bytes_per_page, 128 * 1160);  // 148480
    EXPECT_EQ(v4.csa_format, lmem::KvCacheFormat::kV4Fp8);
    EXPECT_EQ(v4.hca_format, lmem::KvCacheFormat::kV4Fp8);

    // Lightning-Indexer tier: 128 B FP8 key + 4 B F32 scale per compressed
    // block; 8192-token page → 2048 entries.
    EXPECT_EQ(v4.indexer_entry_bytes, 132);
    EXPECT_EQ(v4.indexer_bytes_per_page, 2048 * 132);  // 270336

    EXPECT_EQ(v4.num_csa_layers, 21);
    EXPECT_EQ(v4.num_hca_layers, 20);
    EXPECT_EQ(v4.num_swa_layers, 2);
    EXPECT_EQ(v4.num_swa_kv_layers, 44);   // 43 hidden + 1 nextn (SWA-only)
}

TEST(VramAllocatorV4, TierGeometryTqAndMix) {
    lmod::ModelConfig mcfg_tq(v4_config("csa_hca_tq"));
    auto tq = lmem::compute_v4_kv_layout(v4_config("csa_hca_tq"), mcfg_tq);
    EXPECT_EQ(tq.csa_entry_bytes, 644);
    EXPECT_EQ(tq.hca_entry_bytes, 644);
    EXPECT_EQ(tq.swa_entry_bytes, 1160);   // SWA always FP8
    EXPECT_EQ(tq.csa_format, lmem::KvCacheFormat::kV4Tq);
    EXPECT_EQ(tq.csa_bytes_per_page, 64 * 644);

    lmod::ModelConfig mcfg_mix(v4_config("csa_hca_tq_mix"));
    auto mix = lmem::compute_v4_kv_layout(v4_config("csa_hca_tq_mix"), mcfg_mix);
    EXPECT_EQ(mix.csa_entry_bytes, 644);   // V4-5Mb: CSA → TQ
    EXPECT_EQ(mix.hca_entry_bytes, 1160);  //         HCA → FP8
    EXPECT_EQ(mix.csa_format, lmem::KvCacheFormat::kV4Tq);
    EXPECT_EQ(mix.hca_format, lmem::KvCacheFormat::kV4Fp8);
}

TEST(VramAllocatorV4, LayoutPageCountsDemandDriven) {
    auto layout = v4_layout(v4_config());
    ASSERT_EQ(layout.gpus.size(), 1u);
    const auto& gpu = layout.gpus[0];

    EXPECT_EQ(layout.kv_bytes_per_page, 74240);  // CSA bucket page
    EXPECT_EQ(layout.kv_cache_format, lmem::KvCacheFormat::kV4Fp8);
    EXPECT_EQ(layout.indexer_k_bytes_per_page, 270336);

    // 32 req × ceil(32768/256)=128 blocks × layer counts (demand fits VRAM).
    EXPECT_EQ(gpu.kv_main_pages, 32 * 128 * 21);        // 86016 CSA pages
    EXPECT_EQ(gpu.kv_hca_pages, 32 * 128 * 20);         // 81920 HCA pages
    // SWA/raw: per seq — 2 SWA layers×2 + 21 CSA×3 + 20 HCA×3 + 1 MTP×2.
    EXPECT_EQ(gpu.kv_swa_pages, 32 * (2 * 2 + 21 * 3 + 20 * 3 + 2));
    // Speculation: CSA-page-size sibling, 15 % of CSA demand.
    EXPECT_EQ(gpu.kv_speculation_pages,
              static_cast<int>(32 * 128 * 21 * 0.15));
    // Indexer: ceil(32768/8192)=4 pages/seq × 21 CSA layers × 32 req.
    EXPECT_EQ(gpu.indexer_k_pages, 4 * 21 * 32);
    EXPECT_EQ(gpu.max_kv_pages, gpu.kv_main_pages + gpu.kv_speculation_pages);
}

TEST(VramAllocatorV4, RegionsSumToTotal) {
    auto layout = v4_layout(v4_config());
    const auto& gpu = layout.gpus[0];
    int64_t sum = gpu.pinned_bytes + gpu.kv_main_bytes +
                  gpu.kv_speculation_bytes + gpu.indexer_k_bytes +
                  gpu.kv_hca_bytes + gpu.kv_swa_bytes +
                  gpu.expert_stable_bytes + gpu.expert_streaming_bytes +
                  gpu.safety_margin_bytes;
    EXPECT_EQ(sum, gpu.total_vram_bytes);
    EXPECT_GT(gpu.kv_hca_bytes, 0);
    EXPECT_GT(gpu.kv_swa_bytes, 0);
    EXPECT_GT(gpu.expert_streaming_bytes + gpu.expert_stable_bytes, 0);
}

TEST(VramAllocatorV4, TqShrinksCsaBytes) {
    auto fp8 = v4_layout(v4_config("csa_hca"));
    auto tq = v4_layout(v4_config("csa_hca_tq"));
    // Same demand-driven page counts, smaller pages under TQ.
    EXPECT_EQ(tq.gpus[0].kv_main_pages, fp8.gpus[0].kv_main_pages);
    EXPECT_LT(tq.kv_bytes_per_page, fp8.kv_bytes_per_page);
    EXPECT_LT(tq.gpus[0].kv_main_bytes, fp8.gpus[0].kv_main_bytes);
    // SWA tier identical (always FP8).
    EXPECT_EQ(tq.gpus[0].kv_swa_bytes, fp8.gpus[0].kv_swa_bytes);
}

TEST(VramAllocatorV4, KvFootprintFarBelowMlaBaseline) {
    // The whole point of the 3-tier scheme: compressed KV.  Compare the V4 KV
    // carve against a naive uniform 1160 B/token/layer baseline at the same
    // serving load (32 req × 32768 tokens × 44 layers ≈ 53 GB).
    auto layout = v4_layout(v4_config());
    const auto& gpu = layout.gpus[0];
    const int64_t v4_kv = gpu.kv_main_bytes + gpu.kv_speculation_bytes +
                          gpu.kv_hca_bytes + gpu.kv_swa_bytes;
    const int64_t naive = 32LL * 32768 * 44 * 1160;
    EXPECT_LT(v4_kv, naive / 5);
}

TEST(VramAllocatorV4, LegacyUniformSizingThrows) {
    auto cfg = v4_config();
    lmod::ModelConfig mcfg(cfg);
    EXPECT_THROW(lmem::kv_bytes_per_token(mcfg, cfg.quantization.kv_cache,
                                          cfg.compute.attention_backend),
                 std::logic_error);
    EXPECT_THROW(lmem::kv_bytes_per_page(mcfg, cfg), std::logic_error);
}

TEST(VramAllocatorV4, MainTierEntryBytesPerLayer) {
    auto cfg = v4_config("csa_hca_tq_mix");
    lmod::ModelConfig mcfg(cfg);
    EXPECT_EQ(lmem::v4_main_tier_entry_bytes(cfg, mcfg, 0), 0);     // SWA
    EXPECT_EQ(lmem::v4_main_tier_entry_bytes(cfg, mcfg, 2), 644);   // CSA (TQ)
    EXPECT_EQ(lmem::v4_main_tier_entry_bytes(cfg, mcfg, 3), 1160);  // HCA (FP8)
}

// TD-V4-KVT RESOLVED (P3, 2026-08-21): kv_tiering under V4 no longer
// fails the layout — it routes to the CSA-bucket V4KvTiering manager
// (no layout change: the CSA tier IS kv_main).
TEST(VramAllocatorV4, KvTieringAllowed) {
    auto cfg = v4_config();
    cfg.memory.kv_tiering.enabled = true;
    lmod::ModelConfig mcfg(cfg);
    lmod::GgufQuantInterface mxfp4{lmod::GgufKQuantType::MXFP4};
    lmod::LayerRegistry reg(mcfg, cfg, mxfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
    EXPECT_TRUE(layout.v4.enabled);
}

TEST(VramAllocatorV4, FailClosedDcp) {
    auto cfg = v4_config();
    cfg.hardware.gpus.push_back(cfg.hardware.gpus[0]);
    cfg.hardware.gpus[1].id = 1;
    cfg.hardware.tp_array = {0, 1};
    cfg.parallelism.tensor_parallelism = 2;
    cfg.hardware.dcp_enabled = true;
    lmod::ModelConfig mcfg(cfg);
    lmod::GgufQuantInterface mxfp4{lmod::GgufKQuantType::MXFP4};
    lmod::LayerRegistry reg(mcfg, cfg, mxfp4);
    // Schema default dcp_kv_mode = sharded → still fail-closed
    // (TD-V4-DCP-KV narrowed: sharded only).
    EXPECT_THROW(lmem::compute_vram_layout(cfg, reg, mcfg),
                 std::invalid_argument);
    // V4-2c: REPLICATED KV at tp=2 is the V4 TP mode — allowed.
    cfg.hardware.dcp_kv_mode = lc::DcpKvMode::replicated;
    lmod::ModelConfig mcfg2(cfg);
    lmod::LayerRegistry reg2(mcfg2, cfg, mxfp4);
    auto layout = lmem::compute_vram_layout(cfg, reg2, mcfg2);
    EXPECT_TRUE(layout.v4.enabled);
}

TEST(VramAllocatorV4, FailClosedExplicitPageCount) {
    auto cfg = v4_config();
    cfg.memory.kv_cache.max_pages_per_gpu = 1000;
    lmod::ModelConfig mcfg(cfg);
    lmod::GgufQuantInterface mxfp4{lmod::GgufKQuantType::MXFP4};
    lmod::LayerRegistry reg(mcfg, cfg, mxfp4);
    EXPECT_THROW(lmem::compute_vram_layout(cfg, reg, mcfg),
                 std::invalid_argument);
}

TEST(VramAllocatorV4, AllocatorPartitionsTierRegions) {
    auto cfg = v4_config();
    auto layout = v4_layout(cfg);
    std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned;
    std::vector<lcomp::DeviceBackend*> ptrs;
    for (size_t i = 0; i < layout.gpus.size(); ++i) {
        lc::GpuRef ref{.position = static_cast<int>(i),
                       .id = layout.gpus[i].gpu_id};
        owned.push_back(lcomp::make_null_device_backend(ref));
        ptrs.push_back(owned.back().get());
    }
    const auto snapshot = layout;  // VramAllocator consumes the layout
    lmem::VramAllocator vram(std::move(layout), ptrs);
    const auto& reg0 = vram.region(0);
    const auto& g = snapshot.gpus[0];

    // Ordering: pinned < kv_speculation < indexer_k < kv_hca < kv_swa <
    // kv_main < expert_streaming < expert_stable.
    EXPECT_LT(reg0.pinned, reg0.kv_speculation);
    EXPECT_LT(reg0.kv_speculation, reg0.indexer_k);
    EXPECT_LT(reg0.indexer_k, reg0.kv_hca);
    EXPECT_LT(reg0.kv_hca, reg0.kv_swa);
    EXPECT_LT(reg0.kv_swa, reg0.kv_main);
    EXPECT_LT(reg0.kv_main, reg0.expert_streaming);

    // Region spans hold their page pools.
    auto span = [](void* a, void* b) {
        return static_cast<char*>(b) - static_cast<char*>(a);
    };
    EXPECT_GE(span(reg0.kv_hca, reg0.kv_swa),
              static_cast<int64_t>(g.kv_hca_pages) *
                  snapshot.v4.hca_bytes_per_page);
    EXPECT_GE(span(reg0.kv_swa, reg0.kv_main),
              static_cast<int64_t>(g.kv_swa_pages) *
                  snapshot.v4.swa_bytes_per_page);
}

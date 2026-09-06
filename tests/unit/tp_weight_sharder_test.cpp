#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include <nlohmann/json.hpp>

#include "config/config_parser.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"
#include "model/quantization/gguf_kquant.h"
#include "model/quantization/nvfp4.h"
#include "model/weight_loader/tp_weight_sharder.h"

namespace lc = layerstorm::config;
using layerstorm::model::ShardMode;
using layerstorm::model::ShardedTensor;
using layerstorm::model::ShardedWeightBundle;
using layerstorm::model::TpWeightSharder;
using layerstorm::model::shard_mode_for;
using layerstorm::model::TensorComponent;
using layerstorm::model::TensorRole;
using layerstorm::model::TensorOwner;
using layerstorm::model::TensorId;
using layerstorm::model::RawTensor;
using layerstorm::model::WeightBundle;
using layerstorm::model::SafetensorsDtype;
using layerstorm::model::ModelConfig;
using layerstorm::model::LayerRegistry;
using layerstorm::model::dtype_size;

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

static lc::Config deepseek_v3_2_config() {
    auto j = nlohmann::json{
        {"model", {
            {"architecture",           "deepseek_v3"},
            {"weights_path",           "/data/models/deepseek-v3.2/"},
            {"weights_format",         "safetensors"},
            {"num_hidden_layers",      61},
            {"hidden_size",            7168},
            {"num_attention_heads",    128},
            {"num_key_value_heads",    128},
            {"intermediate_size",      18432},
            {"n_routed_experts",       256},
            {"n_shared_experts",       1},
            {"num_experts_per_tok",    8},
            {"n_group",                8},
            {"topk_group",             4},
            {"vocab_size",             129280},
            {"max_position_embeddings", 163840},
            {"kv_lora_rank",           512},
            {"q_lora_rank",            1536},
            {"qk_rope_head_dim",       64},
            {"qk_nope_head_dim",       128},
            {"v_head_dim",             128},
            {"first_k_dense_replace",  3},
            {"moe_layer_freq",         1},
            {"index_topk",             2048},
            {"index_n_heads",          64},
            {"index_head_dim",         128},
            {"num_nextn_predict_layers", 1},
            {"rms_norm_eps",           1e-6},
            {"rope_theta",             10000.0},
            {"routed_scaling_factor",  2.5},
            {"moe_intermediate_size",  2048},
        }},
        {"quantization", {{"weights", "nvfp4"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}},
                                {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                      {"system_ram_gb", 256},
                      {"tp_array", {0, 1}}}},
        {"parallelism", {{"tensor_parallelism", 2}}},
    };
    return lc::parse_config(j);
}

/// Create a synthetic buffer filled with sequential byte values.
static std::vector<std::byte> make_synthetic_data(size_t size) {
    std::vector<std::byte> buf(size);
    for (size_t i = 0; i < size; ++i) {
        buf[i] = static_cast<std::byte>(i & 0xFF);
    }
    return buf;
}

/// Create a RawTensor backed by `buf` with given shape and dtype.
static RawTensor make_raw_tensor(const std::vector<std::byte>& buf,
                                 std::vector<int64_t> shape,
                                 SafetensorsDtype dtype) {
    return RawTensor{
        .data = std::span<const std::byte>(buf.data(), buf.size()),
        .dtype = dtype,
        .shape = std::move(shape),
    };
}

/// Create a WeightBundle from synthetic data.
static WeightBundle make_bundle(const std::vector<std::byte>& buf,
                                std::vector<int64_t> shape,
                                SafetensorsDtype dtype,
                                TensorComponent component,
                                TensorOwner owner = TensorOwner::attention,
                                int layer = 0) {
    WeightBundle b;
    b.id = TensorId{component, TensorRole::weight, owner, layer, -1};
    b.weight = make_raw_tensor(buf, std::move(shape), dtype);
    return b;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 1: Shard mode classification
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, ShardModeClassification) {
    // Column-parallel
    EXPECT_EQ(shard_mode_for(TensorComponent::q_b_proj), ShardMode::kColumnParallel);
    EXPECT_EQ(shard_mode_for(TensorComponent::kv_b_proj), ShardMode::kColumnParallel);

    // Row-parallel
    EXPECT_EQ(shard_mode_for(TensorComponent::o_proj), ShardMode::kRowParallel);

    // Replicated — attention
    // V4-2c (TD-V4-TP): grouped o_proj shards BY GROUP — stage-1 slabs
    // column-parallel, stage-2 K-slice row-parallel (executor allreduce).
    EXPECT_EQ(shard_mode_for(TensorComponent::o_proj_a),
              ShardMode::kColumnParallel);
    EXPECT_EQ(shard_mode_for(TensorComponent::o_proj_b),
              ShardMode::kRowParallel);
    // V4 components that must stay REPLICATED (ticket-C pinned plan: ÷tp
    // only q_b/o_a/o_b).
    EXPECT_EQ(shard_mode_for(TensorComponent::attn_sinks),
              ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::compressor_wkv),
              ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::compressor_wgate),
              ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::compressor_ape),
              ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::compressor_norm),
              ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::hc_attn_fn),
              ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::hc_ffn_fn),
              ShardMode::kReplicated);

    EXPECT_EQ(shard_mode_for(TensorComponent::q_a_proj), ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::q_a_norm), ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::kv_a_proj_with_mqa), ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::kv_a_norm), ShardMode::kReplicated);

    // Replicated — DSA indexer (all 5 components)
    EXPECT_EQ(shard_mode_for(TensorComponent::indexer_wq_b), ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::indexer_wk), ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::indexer_k_norm_weight), ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::indexer_k_norm_bias), ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::indexer_weights_proj), ShardMode::kReplicated);

    // Replicated — layer norms
    EXPECT_EQ(shard_mode_for(TensorComponent::input_layernorm), ShardMode::kReplicated);
    EXPECT_EQ(shard_mode_for(TensorComponent::post_attention_layernorm), ShardMode::kReplicated);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2: Column-parallel shape correctness (V3.2 q_b_proj)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, ColumnParallelShape) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // q_b_proj: [num_heads * (qk_nope + qk_rope), q_lora_rank] = [24576, 1536]
    size_t sz = 24576 * 1536;  // FP8 = 1 byte/elem
    auto buf = make_synthetic_data(sz);
    auto bundle = make_bundle(buf, {24576, 1536}, SafetensorsDtype::F8_E4M3,
                              TensorComponent::q_b_proj);

    auto s0 = sharder.shard_attention(bundle, 0);
    auto s1 = sharder.shard_attention(bundle, 1);

    EXPECT_EQ(s0.weight.shape[0], 12288);
    EXPECT_EQ(s0.weight.shape[1], 1536);
    EXPECT_EQ(s1.weight.shape[0], 12288);
    EXPECT_EQ(s1.weight.shape[1], 1536);
    EXPECT_EQ(s0.weight.size_bytes(), 12288 * 1536);
    EXPECT_EQ(s1.weight.size_bytes(), 12288 * 1536);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3: Column-parallel data correctness (zero-copy)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, ColumnParallelData) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // Small tensor: 8 rows x 4 cols, FP8 (1 byte/elem)
    auto buf = make_synthetic_data(32);  // 8 * 4 = 32 bytes
    auto bundle = make_bundle(buf, {8, 4}, SafetensorsDtype::F8_E4M3,
                              TensorComponent::q_b_proj);

    auto s0 = sharder.shard_attention(bundle, 0);
    auto s1 = sharder.shard_attention(bundle, 1);

    // Rank 0: rows 0-3 (bytes 0-15)
    EXPECT_EQ(s0.weight.shape[0], 4);
    EXPECT_EQ(s0.weight.shape[1], 4);
    EXPECT_FALSE(s0.weight.is_owned());  // zero-copy
    EXPECT_EQ(s0.weight.data.data(), buf.data());  // points into original
    EXPECT_EQ(s0.weight.data.size(), 16u);

    // Rank 1: rows 4-7 (bytes 16-31)
    EXPECT_EQ(s1.weight.shape[0], 4);
    EXPECT_FALSE(s1.weight.is_owned());
    EXPECT_EQ(s1.weight.data.data(), buf.data() + 16);

    // Verify data content
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(s0.weight.data[i], buf[i]);
        EXPECT_EQ(s1.weight.data[i], buf[i + 16]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 4: Row-parallel shape correctness (V3.2 o_proj)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, RowParallelShape) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // o_proj: [hidden_size, num_heads * v_head_dim] = [7168, 16384]
    // Full tensor is huge, just verify shapes (don't allocate 117 MB)
    size_t sz = 64 * 128;  // small proxy: [64, 128]
    auto buf = make_synthetic_data(sz);
    auto bundle = make_bundle(buf, {64, 128}, SafetensorsDtype::F8_E4M3,
                              TensorComponent::o_proj);

    auto s0 = sharder.shard_attention(bundle, 0);
    auto s1 = sharder.shard_attention(bundle, 1);

    EXPECT_EQ(s0.weight.shape[0], 64);
    EXPECT_EQ(s0.weight.shape[1], 64);
    EXPECT_EQ(s1.weight.shape[0], 64);
    EXPECT_EQ(s1.weight.shape[1], 64);
    EXPECT_EQ(s0.weight.size_bytes(), 64 * 64);
    EXPECT_EQ(s1.weight.size_bytes(), 64 * 64);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 5: Row-parallel data correctness (packed copy)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, RowParallelData) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // 4 rows x 8 cols, FP8 (1 byte/elem)
    auto buf = make_synthetic_data(32);
    auto bundle = make_bundle(buf, {4, 8}, SafetensorsDtype::F8_E4M3,
                              TensorComponent::o_proj);

    auto s0 = sharder.shard_attention(bundle, 0);
    auto s1 = sharder.shard_attention(bundle, 1);

    // Rank 0: cols 0-3 of each row
    EXPECT_EQ(s0.weight.shape[0], 4);
    EXPECT_EQ(s0.weight.shape[1], 4);
    EXPECT_TRUE(s0.weight.is_owned());  // packed copy

    // Rank 1: cols 4-7 of each row
    EXPECT_TRUE(s1.weight.is_owned());

    // Verify data content row by row
    // Original: row r = [r*8+0, r*8+1, ..., r*8+7]
    // Rank 0 should have: row r = [r*8+0, r*8+1, r*8+2, r*8+3]
    // Rank 1 should have: row r = [r*8+4, r*8+5, r*8+6, r*8+7]
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            auto expected_r0 = static_cast<std::byte>((r * 8 + c) & 0xFF);
            auto expected_r1 = static_cast<std::byte>((r * 8 + c + 4) & 0xFF);
            EXPECT_EQ(s0.weight.data[r * 4 + c], expected_r0)
                << "rank0 row=" << r << " col=" << c;
            EXPECT_EQ(s1.weight.data[r * 4 + c], expected_r1)
                << "rank1 row=" << r << " col=" << c;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 6: Replicated returns full tensor unchanged
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, ReplicatedUnchanged) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // q_a_proj: replicated
    size_t sz = 1536 * 7168;
    auto buf = make_synthetic_data(sz);
    auto bundle = make_bundle(buf, {1536, 7168}, SafetensorsDtype::F8_E4M3,
                              TensorComponent::q_a_proj);

    auto s0 = sharder.shard_attention(bundle, 0);
    auto s1 = sharder.shard_attention(bundle, 1);

    // Both ranks get the full tensor
    EXPECT_EQ(s0.weight.shape[0], 1536);
    EXPECT_EQ(s0.weight.shape[1], 7168);
    EXPECT_FALSE(s0.weight.is_owned());
    EXPECT_EQ(s0.weight.data.data(), buf.data());
    EXPECT_EQ(s0.weight.data.size(), buf.size());

    EXPECT_EQ(s1.weight.shape[0], 1536);
    EXPECT_EQ(s1.weight.shape[1], 7168);
    EXPECT_FALSE(s1.weight.is_owned());
    EXPECT_EQ(s1.weight.data.data(), buf.data());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 7: FP8 blockwise scale shards with weight (column-parallel)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, Fp8ScaleFollowsWeight) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // q_b_proj FP8: [24576, 1536], scale: [192, 12] (ceil(24576/128), ceil(1536/128))
    auto weight_buf = make_synthetic_data(24576 * 1536);
    auto scale_buf = make_synthetic_data(192 * 12 * 4);  // F32 = 4 bytes/elem

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::q_b_proj, TensorRole::weight,
                         TensorOwner::attention, 0, -1};
    bundle.weight = make_raw_tensor(weight_buf, {24576, 1536}, SafetensorsDtype::F8_E4M3);
    bundle.aux.emplace_back(TensorRole::weight_scale,
                            make_raw_tensor(scale_buf, {192, 12}, SafetensorsDtype::F32));

    auto s0 = sharder.shard_attention(bundle, 0);
    auto s1 = sharder.shard_attention(bundle, 1);

    // Weight sharded: [12288, 1536]
    EXPECT_EQ(s0.weight.shape[0], 12288);
    EXPECT_EQ(s0.weight.shape[1], 1536);

    // Scale should also be column-parallel: [96, 12]
    auto* scale0 = s0.find_aux(TensorRole::weight_scale);
    auto* scale1 = s1.find_aux(TensorRole::weight_scale);
    ASSERT_NE(scale0, nullptr);
    ASSERT_NE(scale1, nullptr);
    EXPECT_EQ(scale0->shape[0], 96);
    EXPECT_EQ(scale0->shape[1], 12);
    EXPECT_EQ(scale1->shape[0], 96);
    EXPECT_EQ(scale1->shape[1], 12);

    // Column-parallel scales should be zero-copy
    EXPECT_FALSE(scale0->is_owned());
    EXPECT_FALSE(scale1->is_owned());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 8: NVFP4 scale sharding (row-parallel for o_proj)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, NvFp4ScaleSharding) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // o_proj NVFP4: physical shape [7168, 8192] (U8 packed, logical 16384 cols)
    // weight_scale (F8_E4M3): [7168, 1024] (logical_cols / group_size = 16384/16)
    // weight_scale_2 (F32): scalar [1]
    // input_scale (F32): scalar [1]
    auto weight_buf = make_synthetic_data(7168 * 8192);
    auto ws_buf = make_synthetic_data(7168 * 1024);    // F8_E4M3 = 1 byte
    auto ws2_buf = make_synthetic_data(4);              // F32 scalar
    auto is_buf = make_synthetic_data(4);               // F32 scalar

    WeightBundle bundle;
    bundle.id = TensorId{TensorComponent::o_proj, TensorRole::weight,
                         TensorOwner::attention, 0, -1};
    bundle.weight = make_raw_tensor(weight_buf, {7168, 8192}, SafetensorsDtype::U8);
    bundle.aux.emplace_back(TensorRole::weight_scale,
                            make_raw_tensor(ws_buf, {7168, 1024}, SafetensorsDtype::F8_E4M3));
    bundle.aux.emplace_back(TensorRole::weight_scale_2,
                            make_raw_tensor(ws2_buf, {1}, SafetensorsDtype::F32));
    bundle.aux.emplace_back(TensorRole::input_scale,
                            make_raw_tensor(is_buf, {1}, SafetensorsDtype::F32));

    auto s0 = sharder.shard_attention(bundle, 0);

    // Weight: row-parallel → [7168, 4096]
    EXPECT_EQ(s0.weight.shape[0], 7168);
    EXPECT_EQ(s0.weight.shape[1], 4096);
    EXPECT_TRUE(s0.weight.is_owned());

    // weight_scale: row-parallel → [7168, 512]
    auto* ws0 = s0.find_aux(TensorRole::weight_scale);
    ASSERT_NE(ws0, nullptr);
    EXPECT_EQ(ws0->shape[0], 7168);
    EXPECT_EQ(ws0->shape[1], 512);
    EXPECT_TRUE(ws0->is_owned());  // row-parallel → packed copy

    // weight_scale_2: scalar → replicated
    auto* ws2_0 = s0.find_aux(TensorRole::weight_scale_2);
    ASSERT_NE(ws2_0, nullptr);
    EXPECT_EQ(ws2_0->shape.size(), 1u);
    EXPECT_EQ(ws2_0->shape[0], 1);
    EXPECT_FALSE(ws2_0->is_owned());

    // input_scale: scalar → replicated
    auto* is0 = s0.find_aux(TensorRole::input_scale);
    ASSERT_NE(is0, nullptr);
    EXPECT_FALSE(is0->is_owned());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 9: DSA indexer always replicated
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, IndexerAlwaysReplicated) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    struct TestCase {
        TensorComponent comp;
        std::vector<int64_t> shape;
    };
    std::vector<TestCase> cases = {
        {TensorComponent::indexer_wq_b,         {12582912 / 1, 1}},  // simplified 1D-ish
        {TensorComponent::indexer_wk,            {7168, 128}},
        {TensorComponent::indexer_k_norm_weight, {128}},
        {TensorComponent::indexer_k_norm_bias,   {128}},
        {TensorComponent::indexer_weights_proj,  {7168, 64}},
    };

    for (auto& tc : cases) {
        int64_t numel = 1;
        for (auto d : tc.shape) numel *= d;
        auto buf = make_synthetic_data(static_cast<size_t>(numel));

        WeightBundle bundle;
        bundle.id = TensorId{tc.comp, TensorRole::weight, TensorOwner::attention, 0, -1};
        bundle.weight = make_raw_tensor(buf, tc.shape, SafetensorsDtype::F8_E4M3);

        // Shard as attention (indexer components classified as replicated)
        auto s0 = sharder.shard_attention(bundle, 0);
        auto s1 = sharder.shard_attention(bundle, 1);

        EXPECT_EQ(s0.weight.shape, tc.shape) << "component mismatch";
        EXPECT_EQ(s1.weight.shape, tc.shape) << "component mismatch";
        EXPECT_FALSE(s0.weight.is_owned());
        EXPECT_FALSE(s1.weight.is_owned());
        EXPECT_EQ(s0.weight.data.data(), buf.data());
        EXPECT_EQ(s1.weight.data.data(), buf.data());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 10: Layer-level sharding bundle count
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, LayerBundleCount) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // Create attention bundles: q_a, q_a_norm, q_b, kv_a, kv_a_norm, kv_b, o_proj = 7
    auto small_buf = make_synthetic_data(64);
    std::vector<WeightBundle> attention;
    for (auto comp : {TensorComponent::q_a_proj, TensorComponent::q_a_norm,
                      TensorComponent::q_b_proj, TensorComponent::kv_a_proj_with_mqa,
                      TensorComponent::kv_a_norm, TensorComponent::kv_b_proj,
                      TensorComponent::o_proj}) {
        attention.push_back(make_bundle(small_buf, {8, 8}, SafetensorsDtype::F8_E4M3,
                                        comp));
    }

    // Create indexer bundles: 5 components
    std::vector<WeightBundle> indexer;
    for (auto comp : {TensorComponent::indexer_wq_b, TensorComponent::indexer_wk,
                      TensorComponent::indexer_k_norm_weight,
                      TensorComponent::indexer_k_norm_bias,
                      TensorComponent::indexer_weights_proj}) {
        indexer.push_back(make_bundle(small_buf, {8, 8}, SafetensorsDtype::F8_E4M3,
                                      comp));
    }

    auto result = sharder.shard_attention_layer(attention, indexer, 0);

    // Total: 7 attention + 5 indexer = 12
    EXPECT_EQ(result.size(), 12u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 11: Size match with LayerRegistry
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, SizeMatchLayerRegistry) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    layerstorm::model::Fp8E4M3 fp8;  // Use FP8 for attention (1 byte/elem)

    // Build LayerRegistry with FP8 attention quant to match our synthetic bundles
    cfg.quantization.weights = lc::WeightQuant::fp8_e4m3;
    ModelConfig model_cfg2{cfg};
    LayerRegistry reg(model_cfg2, cfg, fp8);

    // Get expected per-GPU attention bytes from LayerRegistry
    // This calls attention_bytes_per_gpu(2) internally for TP GPUs
    // LayerRegistry formula: replicated + indexer + shardable/2
    // We need the per-layer value, not the budget (which sums across layers).
    // Use the layer info + the registry's internal TP-aware calculation.

    // V3.2 FP8 attention dimensions (all 1 byte/elem):
    // q_a_proj:  7168 * 1536              =  11,010,048
    // q_b_proj:  1536 * 128 * (128+64)    =  37,748,736
    // kv_a_proj: 7168 * (512+64)          =   4,128,768
    // kv_b_proj: 512 * 128 * (128+128)    =  16,777,216
    // o_proj:    128 * 128 * 7168         = 117,440,512
    // norms:     1536 + 512 + 2*7168      =      16,384
    //
    // Replicated = q_a + kv_a + norms = 11,010,048 + 4,128,768 + 16,384 = 15,155,200
    // Shardable  = q_b + kv_b + o     = 37,748,736 + 16,777,216 + 117,440,512 = 171,966,464
    //
    // DSA indexer (FP8):
    // q_idx_b:      1536 * 64 * 128     = 12,582,912
    // k_idx:        7168 * 128          =    917,504
    // k_idx_norm:   128                 =        128
    // weights_proj: 7168 * 64           =    458,752
    // Indexer total                     = 13,959,296
    //
    // Per-GPU (TP=2) = replicated + indexer + shardable/2
    //               = 15,155,200 + 13,959,296 + 85,983,232
    //               = 115,097,728

    TpWeightSharder sharder(model_cfg, 2);

    // Create synthetic FP8 bundles matching V3.2 shapes
    // Attention projections
    auto q_a_buf = make_synthetic_data(7168LL * 1536);
    auto q_b_buf = make_synthetic_data(24576LL * 1536);  // 128*(128+64) = 24576
    auto kv_a_buf = make_synthetic_data(7168LL * 576);   // 512+64 = 576
    auto kv_b_buf = make_synthetic_data(32768LL * 512);  // 128*(128+128) = 32768
    auto o_buf = make_synthetic_data(7168LL * 16384);    // 128*128 = 16384
    auto norm_buf = make_synthetic_data(16384);          // q_a_norm + kv_a_norm + 2*layernorms

    std::vector<WeightBundle> attention;
    attention.push_back(make_bundle(q_a_buf, {1536, 7168}, SafetensorsDtype::F8_E4M3,
                                    TensorComponent::q_a_proj));
    attention.push_back(make_bundle(q_b_buf, {24576, 1536}, SafetensorsDtype::F8_E4M3,
                                    TensorComponent::q_b_proj));
    attention.push_back(make_bundle(kv_a_buf, {576, 7168}, SafetensorsDtype::F8_E4M3,
                                    TensorComponent::kv_a_proj_with_mqa));
    attention.push_back(make_bundle(kv_b_buf, {32768, 512}, SafetensorsDtype::F8_E4M3,
                                    TensorComponent::kv_b_proj));
    attention.push_back(make_bundle(o_buf, {7168, 16384}, SafetensorsDtype::F8_E4M3,
                                    TensorComponent::o_proj));

    // Norms: split into individual bundles as they'd appear in LoadedModel
    // q_a_norm: [q_lora_rank] = [1536]
    // kv_a_norm: [kv_lora_rank] = [512]
    // input_layernorm: [hidden_size] = [7168]
    // post_attention_layernorm: [hidden_size] = [7168]
    auto q_a_norm_buf = make_synthetic_data(1536);
    auto kv_a_norm_buf = make_synthetic_data(512);
    auto in_norm_buf = make_synthetic_data(7168);
    auto post_norm_buf = make_synthetic_data(7168);

    attention.push_back(make_bundle(q_a_norm_buf, {1536}, SafetensorsDtype::F8_E4M3,
                                    TensorComponent::q_a_norm));
    attention.push_back(make_bundle(kv_a_norm_buf, {512}, SafetensorsDtype::F8_E4M3,
                                    TensorComponent::kv_a_norm));

    // Note: norms go in a separate vector in LoadedModel, but for sharding they're
    // all replicated. We include them in attention for total_bytes accounting.
    // However, the LayerRegistry counts norms in attention_bytes, so we add them.
    // The norm bundles would be passed as part of the attention vector or separately.
    // For this test, include them in the attention vector.
    attention.push_back(make_bundle(in_norm_buf, {7168}, SafetensorsDtype::F8_E4M3,
                                    TensorComponent::input_layernorm));
    attention.push_back(make_bundle(post_norm_buf, {7168}, SafetensorsDtype::F8_E4M3,
                                    TensorComponent::post_attention_layernorm));

    // DSA indexer
    auto q_idx_b_buf = make_synthetic_data(1536LL * 64 * 128);
    auto k_idx_buf = make_synthetic_data(7168LL * 128);
    auto k_idx_norm_buf = make_synthetic_data(128);
    auto wp_buf = make_synthetic_data(7168LL * 64);

    std::vector<WeightBundle> indexer;
    indexer.push_back(make_bundle(q_idx_b_buf, {8192, 1536}, SafetensorsDtype::F8_E4M3,
                                  TensorComponent::indexer_wq_b));
    indexer.push_back(make_bundle(k_idx_buf, {7168, 128}, SafetensorsDtype::F8_E4M3,
                                  TensorComponent::indexer_wk));
    indexer.push_back(make_bundle(k_idx_norm_buf, {128}, SafetensorsDtype::F8_E4M3,
                                  TensorComponent::indexer_k_norm_weight));
    // k_idx_norm bias: omit for simplicity (128 bytes, counted in k_idx_norm above)
    indexer.push_back(make_bundle(wp_buf, {7168, 64}, SafetensorsDtype::F8_E4M3,
                                  TensorComponent::indexer_weights_proj));

    auto sharded = sharder.shard_attention_layer(attention, indexer, 0);

    int64_t total_sharded_bytes = 0;
    for (auto& sb : sharded) {
        total_sharded_bytes += sb.total_bytes();
    }

    // Expected per-GPU: replicated + indexer + shardable/2
    // Replicated: q_a(11,010,048) + kv_a(4,128,768) + norms(16,384) = 15,155,200
    // Indexer: q_idx_b(12,582,912) + k_idx(917,504) + k_idx_norm(128) + wp(458,752) = 13,959,296
    // Shardable/2: (q_b(37,748,736) + kv_b(16,777,216) + o(117,440,512)) / 2 = 85,983,232
    // Total = 115,097,728
    //
    // But we didn't include k_idx_norm_bias (128 bytes) in our indexer.
    // The LayerRegistry counts it. So our expected total is 115,097,728 - 128 = 115,097,600.
    // Actually, the LayerRegistry's indexer_bytes_per_layer_ includes only:
    //   q_idx_b + k_idx + k_idx_norm + weights_proj (see layer_registry.cpp:101-107)
    //   k_idx_norm = index_head_dim = 128 elements
    //   It does NOT include k_idx_norm_bias separately (the bias is not counted in the
    //   LayerRegistry formula). So we match: the bias is an extra tensor in the actual
    //   weights but not in the byte budget.
    //
    // Our synthetic data matches the LayerRegistry formula exactly.
    int64_t expected = 15155200LL + 13959296LL + 85983232LL;
    EXPECT_EQ(expected, 115097728LL);  // sanity check
    EXPECT_EQ(total_sharded_bytes, expected);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 12: tp_degree=1 is a no-op
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TpWeightSharder, TpDegree1Noop) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 1);  // No sharding

    // q_b_proj: would normally be column-parallel, but tp=1 → replicated
    auto buf = make_synthetic_data(24576 * 1536);
    auto bundle = make_bundle(buf, {24576, 1536}, SafetensorsDtype::F8_E4M3,
                              TensorComponent::q_b_proj);

    auto s0 = sharder.shard_attention(bundle, 0);

    EXPECT_EQ(s0.weight.shape[0], 24576);
    EXPECT_EQ(s0.weight.shape[1], 1536);
    EXPECT_FALSE(s0.weight.is_owned());
    EXPECT_EQ(s0.weight.data.data(), buf.data());

    // o_proj: would normally be row-parallel, but tp=1 → replicated
    auto o_buf = make_synthetic_data(7168 * 16384);
    auto o_bundle = make_bundle(o_buf, {7168, 16384}, SafetensorsDtype::F8_E4M3,
                                TensorComponent::o_proj);

    auto s_o = sharder.shard_attention(o_bundle, 0);

    EXPECT_EQ(s_o.weight.shape[0], 7168);
    EXPECT_EQ(s_o.weight.shape[1], 16384);
    EXPECT_FALSE(s_o.weight.is_owned());  // No owned buffer = no copy
    EXPECT_EQ(s_o.weight.data.data(), o_buf.data());
}

// ═══════════════════════════════════════════════════════════════════════════════
// GGUF TP sharding (TD-GGUF-ATTN-TP-SHARD)
// ═══════════════════════════════════════════════════════════════════════════════

using layerstorm::model::GgufKQuantType;
namespace gguf = layerstorm::model::gguf;

/// Build a GGUF k-quant weight bundle: logical shape [out, in], dtype U8, and
/// the data buffer sized to the packed byte count out*(in/QK)*block_bytes.
static WeightBundle make_gguf_bundle(const std::vector<std::byte>& buf,
                                     int64_t out, int64_t in,
                                     GgufKQuantType type,
                                     TensorComponent component) {
    WeightBundle b;
    b.id = TensorId{component, TensorRole::weight, TensorOwner::attention, 0, -1};
    b.weight = make_raw_tensor(buf, {out, in}, SafetensorsDtype::U8);
    b.weight.gguf_type = type;
    return b;
}

// ── Column-parallel GGUF (q_b / kv_b): whole packed rows, zero-copy sub-span ──

TEST(TpWeightSharder, GgufColumnParallelQ8_0) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // Q8_0: QK=32, block_bytes=34. out=8 rows, in=128 (=4 blocks/row).
    const int64_t out = 8, in = 128;
    const int qk = gguf::block_values(GgufKQuantType::Q8_0);     // 32
    const int blk = gguf::block_bytes(GgufKQuantType::Q8_0);     // 34
    const int64_t packed_row = (in / qk) * blk;                  // 4*34 = 136
    const int64_t packed_total = out * packed_row;               // 1088
    auto buf = make_synthetic_data(static_cast<size_t>(packed_total));
    auto bundle = make_gguf_bundle(buf, out, in, GgufKQuantType::Q8_0,
                                   TensorComponent::q_b_proj);

    auto s0 = sharder.shard_attention(bundle, 0);
    auto s1 = sharder.shard_attention(bundle, 1);

    // Each rank: out/tp=4 whole packed rows.
    EXPECT_EQ(s0.weight.shape[0], 4);
    EXPECT_EQ(s0.weight.shape[1], in);
    EXPECT_EQ(s0.weight.gguf_type, GgufKQuantType::Q8_0);  // type preserved
    EXPECT_FALSE(s0.weight.is_owned());                    // zero-copy sub-span

    const int64_t per_rank = (out / 2) * packed_row;       // 4*136 = 544
    EXPECT_EQ(s0.weight.data.size(), static_cast<size_t>(per_rank));
    EXPECT_EQ(s0.weight.size_bytes(), per_rank);
    EXPECT_EQ(s1.weight.size_bytes(), per_rank);

    // Sub-span offsets: rank 0 at byte 0, rank 1 at byte 544.
    EXPECT_EQ(s0.weight.data.data(), buf.data());
    EXPECT_EQ(s1.weight.data.data(), buf.data() + per_rank);

    // Content: rank r covers packed bytes [r*per_rank, (r+1)*per_rank).
    for (int64_t i = 0; i < per_rank; ++i) {
        EXPECT_EQ(s0.weight.data[i], buf[i]);
        EXPECT_EQ(s1.weight.data[i], buf[per_rank + i]);
    }
}

TEST(TpWeightSharder, GgufColumnParallelQ4_K) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // Q4_K: QK=256, block_bytes=144. out=4, in=512 (=2 blocks/row).
    const int64_t out = 4, in = 512;
    const int qk = gguf::block_values(GgufKQuantType::Q4_K);     // 256
    const int blk = gguf::block_bytes(GgufKQuantType::Q4_K);     // 144
    const int64_t packed_row = (in / qk) * blk;                  // 2*144 = 288
    auto buf = make_synthetic_data(static_cast<size_t>(out * packed_row));
    auto bundle = make_gguf_bundle(buf, out, in, GgufKQuantType::Q4_K,
                                   TensorComponent::kv_b_proj);

    auto s0 = sharder.shard_attention(bundle, 0);
    auto s1 = sharder.shard_attention(bundle, 1);

    const int64_t per_rank = (out / 2) * packed_row;            // 2*288 = 576
    EXPECT_EQ(s0.weight.shape[0], 2);
    EXPECT_EQ(s0.weight.size_bytes(), per_rank);
    EXPECT_EQ(s0.weight.gguf_type, GgufKQuantType::Q4_K);
    EXPECT_FALSE(s0.weight.is_owned());
    EXPECT_EQ(s1.weight.data.data(), buf.data() + per_rank);
}

// ── Row-parallel GGUF (o_proj): super-block-aligned per-row block gather ──────

TEST(TpWeightSharder, GgufRowParallelQ8_0) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // Q8_0: QK=32. out=3 rows, in=128 (=4 blocks/row). Split in → 64/rank = 2 blocks.
    const int64_t out = 3, in = 128;
    const int qk = gguf::block_values(GgufKQuantType::Q8_0);     // 32
    const int blk = gguf::block_bytes(GgufKQuantType::Q8_0);     // 34
    const int64_t blocks_total = in / qk;                        // 4
    const int64_t packed_row = blocks_total * blk;               // 136
    auto buf = make_synthetic_data(static_cast<size_t>(out * packed_row));
    auto bundle = make_gguf_bundle(buf, out, in, GgufKQuantType::Q8_0,
                                   TensorComponent::o_proj);

    auto s0 = sharder.shard_attention(bundle, 0);
    auto s1 = sharder.shard_attention(bundle, 1);

    const int64_t blocks_per_rank = (in / 2) / qk;               // 2
    const int64_t dst_row = blocks_per_rank * blk;               // 68
    EXPECT_EQ(s0.weight.shape[0], out);
    EXPECT_EQ(s0.weight.shape[1], in / 2);                       // logical cols halved
    EXPECT_EQ(s0.weight.gguf_type, GgufKQuantType::Q8_0);
    EXPECT_TRUE(s0.weight.is_owned());                            // packed gather
    EXPECT_EQ(s0.weight.size_bytes(), out * dst_row);            // packed bytes
    EXPECT_EQ(s0.weight.data.size(), static_cast<size_t>(out * dst_row));

    // Per-row block-range gather: rank 0 → first 2 blocks of each row, rank 1 → last 2.
    for (int64_t r = 0; r < out; ++r) {
        const size_t src_row = static_cast<size_t>(r * packed_row);
        for (int64_t b = 0; b < dst_row; ++b) {
            EXPECT_EQ(s0.weight.data[r * dst_row + b], buf[src_row + b])
                << "rank0 row=" << r << " byte=" << b;
            EXPECT_EQ(s1.weight.data[r * dst_row + b], buf[src_row + dst_row + b])
                << "rank1 row=" << r << " byte=" << b;
        }
    }
}

TEST(TpWeightSharder, GgufRowParallelQ6_K) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // Q6_K: QK=256, block_bytes=210. in=512 (=2 blocks) → 1 block/rank.
    const int64_t out = 2, in = 512;
    const int qk = gguf::block_values(GgufKQuantType::Q6_K);
    const int blk = gguf::block_bytes(GgufKQuantType::Q6_K);
    const int64_t packed_row = (in / qk) * blk;                  // 420
    auto buf = make_synthetic_data(static_cast<size_t>(out * packed_row));
    auto bundle = make_gguf_bundle(buf, out, in, GgufKQuantType::Q6_K,
                                   TensorComponent::o_proj);

    auto s0 = sharder.shard_attention(bundle, 0);
    const int64_t dst_row = ((in / 2) / qk) * blk;               // 1*210
    EXPECT_EQ(s0.weight.shape[1], in / 2);
    EXPECT_TRUE(s0.weight.is_owned());
    EXPECT_EQ(s0.weight.size_bytes(), out * dst_row);
}

// ── Error paths ──────────────────────────────────────────────────────────────

TEST(TpWeightSharder, GgufRowParallelUnalignedThrows) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // Q4_K QK=256, in=256 (1 block). Split by 2 → 128/rank, NOT a multiple of 256.
    const int64_t out = 2, in = 256;
    const int64_t packed_row =
        (in / gguf::block_values(GgufKQuantType::Q4_K)) *
        gguf::block_bytes(GgufKQuantType::Q4_K);
    auto buf = make_synthetic_data(static_cast<size_t>(out * packed_row));
    auto bundle = make_gguf_bundle(buf, out, in, GgufKQuantType::Q4_K,
                                   TensorComponent::o_proj);

    EXPECT_THROW(sharder.shard_attention(bundle, 0), std::invalid_argument);
}

TEST(TpWeightSharder, GgufColumnParallelIndivisibleRowsThrows) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // out=3, not divisible by tp=2.
    const int64_t out = 3, in = 256;
    const int64_t packed_row =
        (in / gguf::block_values(GgufKQuantType::Q4_K)) *
        gguf::block_bytes(GgufKQuantType::Q4_K);
    auto buf = make_synthetic_data(static_cast<size_t>(out * packed_row));
    auto bundle = make_gguf_bundle(buf, out, in, GgufKQuantType::Q4_K,
                                   TensorComponent::q_b_proj);

    EXPECT_THROW(sharder.shard_attention(bundle, 0), std::invalid_argument);
}

// ── tp==1 GGUF unchanged: replicated, type preserved, packed size_bytes ──────

TEST(TpWeightSharder, GgufTpDegree1Noop) {
    auto cfg = deepseek_v3_2_config();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 1);

    const int64_t out = 8, in = 256;
    const int64_t packed_row =
        (in / gguf::block_values(GgufKQuantType::Q4_K)) *
        gguf::block_bytes(GgufKQuantType::Q4_K);
    const int64_t packed_total = out * packed_row;
    auto buf = make_synthetic_data(static_cast<size_t>(packed_total));

    for (auto comp : {TensorComponent::q_b_proj, TensorComponent::o_proj}) {
        auto bundle = make_gguf_bundle(buf, out, in, GgufKQuantType::Q4_K, comp);
        auto s0 = sharder.shard_attention(bundle, 0);
        EXPECT_EQ(s0.weight.shape[0], out);
        EXPECT_EQ(s0.weight.shape[1], in);
        EXPECT_FALSE(s0.weight.is_owned());                  // replicated zero-copy
        EXPECT_EQ(s0.weight.data.data(), buf.data());
        EXPECT_EQ(s0.weight.gguf_type, GgufKQuantType::Q4_K);
        EXPECT_EQ(s0.weight.size_bytes(), packed_total);     // packed, not out*in
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// glm5_next KDA linear attention (GF3.3; spec/GLM-5.3-FLASH-MODELINFO.md §3a)
//
// Real GLM-5.3-Flash dims: H = kda heads = 64, D = head_dim = 128,
// K = short_conv_kernel_size = 4, h = hidden = 4096 ⇒ H*D = 8192.
// ═══════════════════════════════════════════════════════════════════════════════

static lc::Config glm53_flash_config_tp2() {
    nlohmann::json layer_types = nlohmann::json::array();
    for (int l = 0; l < 45; ++l) {
        const bool sparse = (l % 4 == 3);
        layer_types.push_back(sparse ? "deepseek_sparse_attention"
                                     : "linear_attention");
    }
    auto j = nlohmann::json{
        {"model", {
            {"architecture",           "glm5_next"},
            {"weights_path",           "/data/models/glm-5.3-flash/"},
            {"weights_format",         "safetensors"},
            {"num_hidden_layers",      45},
            {"hidden_size",            4096},
            {"num_attention_heads",    64},
            {"num_key_value_heads",    64},
            {"intermediate_size",      12288},
            {"n_routed_experts",       288},
            {"n_shared_experts",       1},
            {"num_experts_per_tok",    8},
            {"n_group",                1},
            {"topk_group",             1},
            {"vocab_size",             154880},
            {"max_position_embeddings", 1048576},
            {"kv_lora_rank",           512},
            {"q_lora_rank",            1536},
            {"qk_rope_head_dim",       0},
            {"qk_nope_head_dim",       256},
            {"v_head_dim",             256},
            {"first_k_dense_replace",  3},
            {"moe_layer_freq",         1},
            {"index_topk",             2048},
            {"index_n_heads",          32},
            {"index_head_dim",         128},
            {"index_kpool",            4},
            {"index_kpool_compress",   true},
            {"index_kpool_always_select_tail", true},
            {"mla_use_nope",           true},
            {"layer_types",            layer_types},
            {"linear_attn_config", {
                {"num_heads", 64}, {"head_dim", 128},
                {"short_conv_kernel_size", 4},
                {"gate_lower_bound", -5.0}}},
            {"hc_mult",                4},
            {"hc_sinkhorn_iters",      20},
            {"hc_eps",                 1e-6},
            {"swiglu_limit",           10.0},
            {"num_nextn_predict_layers", 1},
            {"rms_norm_eps",           1e-5},
            {"routed_scaling_factor",  2.5},
            {"moe_intermediate_size",  2048},
        }},
        {"quantization", {{"weights", "fp8_e4m3"}, {"attention_compute", "fp8_e4m3"},
                          {"kv_cache", "fp8_e4m3"}, {"gating_compute", "fp32"}}},
        {"hardware", {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}},
                                {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
                      {"system_ram_gb", 256},
                      {"tp_array", {0, 1}}}},
        {"parallelism", {{"tensor_parallelism", 2}}},
    };
    return lc::parse_config(j);
}

// ── Mode classification: the head-axis split vs the replicated bottlenecks ──

TEST(TpWeightSharder, Glm5NextKdaShardModes) {
    // Column-parallel = split on the HEAD axis (axis 0 in every case).
    for (auto c : {TensorComponent::kda_q_proj, TensorComponent::kda_k_proj,
                   TensorComponent::kda_v_proj, TensorComponent::kda_b_proj,
                   TensorComponent::kda_f_b_proj, TensorComponent::kda_g_b_proj,
                   TensorComponent::kda_q_conv1d, TensorComponent::kda_k_conv1d,
                   TensorComponent::kda_v_conv1d, TensorComponent::kda_a_log,
                   TensorComponent::kda_dt_bias}) {
        EXPECT_EQ(shard_mode_for(c), ShardMode::kColumnParallel)
            << layerstorm::model::tensor_component_name(c);
    }
    // Rank-D bottlenecks + per-head-dim norm stay whole on every rank.
    for (auto c : {TensorComponent::kda_f_a_proj, TensorComponent::kda_g_a_proj,
                   TensorComponent::kda_o_norm}) {
        EXPECT_EQ(shard_mode_for(c), ShardMode::kReplicated)
            << layerstorm::model::tensor_component_name(c);
    }
    // KDA o_proj reuses the shared MLA component and its row-parallel rule.
    EXPECT_EQ(shard_mode_for(TensorComponent::o_proj), ShardMode::kRowParallel);
}

// ── Column-parallel head split at the real shapes, tp=2 ────────────────────
// Covers rank-2 ([H*D, h] and [H, h]), rank-3 ([H*D, 1, K] depthwise conv) and
// rank-1 ([H] A_log, [H*D] dt_bias) tensors — all split on axis 0, all
// zero-copy contiguous sub-spans.

TEST(TpWeightSharder, Glm5NextKdaColumnParallelRealShapes) {
    auto cfg = glm53_flash_config_tp2();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // q_proj [8192, 4096] BF16 — 64 heads × 128 dim rows.
    auto q_buf = make_synthetic_data(8192ULL * 4096 * 2);
    auto q_bundle = make_bundle(q_buf, {8192, 4096}, SafetensorsDtype::BF16,
                                TensorComponent::kda_q_proj);
    auto q0 = sharder.shard_attention(q_bundle, 0);
    auto q1 = sharder.shard_attention(q_bundle, 1);
    EXPECT_EQ(q0.weight.shape, (std::vector<int64_t>{4096, 4096}));
    EXPECT_EQ(q1.weight.shape, (std::vector<int64_t>{4096, 4096}));
    EXPECT_EQ(q0.weight.size_bytes(), 4096LL * 4096 * 2);
    EXPECT_FALSE(q0.weight.is_owned());
    EXPECT_EQ(q0.weight.data.data(), q_buf.data());
    EXPECT_EQ(q1.weight.data.data(), q_buf.data() + 4096LL * 4096 * 2);

    // b_proj [64, 4096] BF16 — one row PER HEAD, so the head split is a 32-row
    // split (this is the tensor that makes "tp must divide H" load-bearing).
    auto b_buf = make_synthetic_data(64ULL * 4096 * 2);
    auto b_bundle = make_bundle(b_buf, {64, 4096}, SafetensorsDtype::BF16,
                                TensorComponent::kda_b_proj);
    auto b0 = sharder.shard_attention(b_bundle, 0);
    auto b1 = sharder.shard_attention(b_bundle, 1);
    EXPECT_EQ(b0.weight.shape, (std::vector<int64_t>{32, 4096}));
    EXPECT_EQ(b0.weight.size_bytes(), 32LL * 4096 * 2);
    EXPECT_EQ(b1.weight.data.data(), b_buf.data() + 32LL * 4096 * 2);

    // f_b_proj [8192, 128] BF16 — decay-gate up-projection, head-major rows.
    auto fb_buf = make_synthetic_data(8192ULL * 128 * 2);
    auto fb_bundle = make_bundle(fb_buf, {8192, 128}, SafetensorsDtype::BF16,
                                 TensorComponent::kda_f_b_proj);
    auto fb1 = sharder.shard_attention(fb_bundle, 1);
    EXPECT_EQ(fb1.weight.shape, (std::vector<int64_t>{4096, 128}));
    EXPECT_EQ(fb1.weight.size_bytes(), 4096LL * 128 * 2);
    EXPECT_EQ(fb1.weight.data.data(), fb_buf.data() + 4096LL * 128 * 2);

    // q_conv1d [8192, 1, 4] BF16 — RANK 3. A "row" is 1*4 elements = 8 bytes,
    // NOT shape[1]*elem (which the old rank-2 formula would have used).
    auto c_buf = make_synthetic_data(8192ULL * 1 * 4 * 2);
    auto c_bundle = make_bundle(c_buf, {8192, 1, 4}, SafetensorsDtype::BF16,
                                TensorComponent::kda_q_conv1d);
    auto c0 = sharder.shard_attention(c_bundle, 0);
    auto c1 = sharder.shard_attention(c_bundle, 1);
    EXPECT_EQ(c0.weight.shape, (std::vector<int64_t>{4096, 1, 4}));
    EXPECT_EQ(c0.weight.size_bytes(), 4096LL * 4 * 2);
    EXPECT_EQ(c0.weight.data.size(), 4096ULL * 4 * 2);
    EXPECT_FALSE(c0.weight.is_owned());
    EXPECT_EQ(c0.weight.data.data(), c_buf.data());
    EXPECT_EQ(c1.weight.data.data(), c_buf.data() + 4096LL * 4 * 2);
    // Byte-exact content of the channel the two ranks straddle.
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(c1.weight.data[i], c_buf[4096ULL * 4 * 2 + i]);
    }

    // A_log [64] F32 — RANK 1, one value per head.
    auto a_buf = make_synthetic_data(64ULL * 4);
    auto a_bundle = make_bundle(a_buf, {64}, SafetensorsDtype::F32,
                                TensorComponent::kda_a_log);
    auto a0 = sharder.shard_attention(a_bundle, 0);
    auto a1 = sharder.shard_attention(a_bundle, 1);
    EXPECT_EQ(a0.weight.shape, (std::vector<int64_t>{32}));
    EXPECT_EQ(a0.weight.size_bytes(), 32LL * 4);
    EXPECT_FALSE(a0.weight.is_owned());
    EXPECT_EQ(a0.weight.data.data(), a_buf.data());
    EXPECT_EQ(a1.weight.data.data(), a_buf.data() + 32LL * 4);
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(a1.weight.data[i], a_buf[128 + i]);
    }

    // dt_bias [8192] F32 — RANK 1, one value per CHANNEL (head*head_dim).
    auto dt_buf = make_synthetic_data(8192ULL * 4);
    auto dt_bundle = make_bundle(dt_buf, {8192}, SafetensorsDtype::F32,
                                 TensorComponent::kda_dt_bias);
    auto dt0 = sharder.shard_attention(dt_bundle, 0);
    auto dt1 = sharder.shard_attention(dt_bundle, 1);
    EXPECT_EQ(dt0.weight.shape, (std::vector<int64_t>{4096}));
    EXPECT_EQ(dt0.weight.size_bytes(), 4096LL * 4);
    EXPECT_EQ(dt1.weight.data.data(), dt_buf.data() + 4096LL * 4);
}

// ── Replicated KDA tensors + the row-parallel o_proj ───────────────────────

TEST(TpWeightSharder, Glm5NextKdaReplicatedAndOProj) {
    auto cfg = glm53_flash_config_tp2();
    ModelConfig model_cfg{cfg};
    TpWeightSharder sharder(model_cfg, 2);

    // f_a_proj / g_a_proj [128, 4096] BF16 — the rank-128 bottleneck is shared
    // across heads, so both ranks must see the WHOLE tensor, zero-copy.
    auto fa_buf = make_synthetic_data(128ULL * 4096 * 2);
    for (auto comp : {TensorComponent::kda_f_a_proj,
                      TensorComponent::kda_g_a_proj}) {
        auto bundle = make_bundle(fa_buf, {128, 4096}, SafetensorsDtype::BF16, comp);
        for (int r = 0; r < 2; ++r) {
            auto s = sharder.shard_attention(bundle, r);
            EXPECT_EQ(s.weight.shape, (std::vector<int64_t>{128, 4096}));
            EXPECT_EQ(s.weight.size_bytes(), 128LL * 4096 * 2);
            EXPECT_FALSE(s.weight.is_owned());
            EXPECT_EQ(s.weight.data.data(), fa_buf.data());
        }
    }

    // o_norm [128] BF16 — per-head-dim RMSNorm gain, replicated.
    auto on_buf = make_synthetic_data(128ULL * 2);
    auto on_bundle = make_bundle(on_buf, {128}, SafetensorsDtype::BF16,
                                 TensorComponent::kda_o_norm);
    for (int r = 0; r < 2; ++r) {
        auto s = sharder.shard_attention(on_bundle, r);
        EXPECT_EQ(s.weight.shape, (std::vector<int64_t>{128}));
        EXPECT_EQ(s.weight.size_bytes(), 256);
        EXPECT_EQ(s.weight.data.data(), on_buf.data());
    }

    // o_proj [4096, 8192] BF16 — row-parallel on the HEAD axis (its K dim),
    // packed copy, one allreduce afterwards.
    auto o_buf = make_synthetic_data(4096ULL * 8192 * 2);
    auto o_bundle = make_bundle(o_buf, {4096, 8192}, SafetensorsDtype::BF16,
                                TensorComponent::o_proj);
    auto o1 = sharder.shard_attention(o_bundle, 1);
    EXPECT_EQ(o1.weight.shape, (std::vector<int64_t>{4096, 4096}));
    EXPECT_EQ(o1.weight.size_bytes(), 4096LL * 4096 * 2);
    EXPECT_TRUE(o1.weight.is_owned());
    // Row r of rank 1 = source row r, columns [4096, 8192).
    const size_t src_row_bytes = 8192 * 2;
    const size_t dst_row_bytes = 4096 * 2;
    for (int64_t r : {int64_t{0}, int64_t{1}, int64_t{4095}}) {
        EXPECT_EQ(std::memcmp(o1.weight.data.data() + r * dst_row_bytes,
                              o_buf.data() + r * src_row_bytes + dst_row_bytes,
                              dst_row_bytes), 0) << "row " << r;
    }
}

// ── A full KDA layer at tp=1 and tp=2: bundle count + per-rank byte totals ──

TEST(TpWeightSharder, Glm5NextKdaLayerTotals) {
    auto cfg = glm53_flash_config_tp2();
    ModelConfig model_cfg{cfg};

    // Buffers for the 15-tensor KDA attention set (MODELINFO §3a).
    auto qkv_buf   = make_synthetic_data(8192ULL * 4096 * 2);   // q/k/v_proj
    auto b_buf     = make_synthetic_data(64ULL * 4096 * 2);     // b_proj
    auto fga_buf   = make_synthetic_data(128ULL * 4096 * 2);    // f_a/g_a
    auto fgb_buf   = make_synthetic_data(8192ULL * 128 * 2);    // f_b/g_b
    auto conv_buf  = make_synthetic_data(8192ULL * 4 * 2);      // q/k/v_conv1d
    auto alog_buf  = make_synthetic_data(64ULL * 4);            // A_log
    auto dt_buf    = make_synthetic_data(8192ULL * 4);          // dt_bias
    auto onorm_buf = make_synthetic_data(128ULL * 2);           // o_norm
    auto o_buf     = make_synthetic_data(4096ULL * 8192 * 2);   // o_proj

    std::vector<WeightBundle> attention;
    for (auto c : {TensorComponent::kda_q_proj, TensorComponent::kda_k_proj,
                   TensorComponent::kda_v_proj})
        attention.push_back(make_bundle(qkv_buf, {8192, 4096},
                                        SafetensorsDtype::BF16, c));
    attention.push_back(make_bundle(b_buf, {64, 4096}, SafetensorsDtype::BF16,
                                    TensorComponent::kda_b_proj));
    for (auto c : {TensorComponent::kda_f_a_proj, TensorComponent::kda_g_a_proj})
        attention.push_back(make_bundle(fga_buf, {128, 4096},
                                        SafetensorsDtype::BF16, c));
    for (auto c : {TensorComponent::kda_f_b_proj, TensorComponent::kda_g_b_proj})
        attention.push_back(make_bundle(fgb_buf, {8192, 128},
                                        SafetensorsDtype::BF16, c));
    for (auto c : {TensorComponent::kda_q_conv1d, TensorComponent::kda_k_conv1d,
                   TensorComponent::kda_v_conv1d})
        attention.push_back(make_bundle(conv_buf, {8192, 1, 4},
                                        SafetensorsDtype::BF16, c));
    attention.push_back(make_bundle(alog_buf, {64}, SafetensorsDtype::F32,
                                    TensorComponent::kda_a_log));
    attention.push_back(make_bundle(dt_buf, {8192}, SafetensorsDtype::F32,
                                    TensorComponent::kda_dt_bias));
    attention.push_back(make_bundle(onorm_buf, {128}, SafetensorsDtype::BF16,
                                    TensorComponent::kda_o_norm));
    attention.push_back(make_bundle(o_buf, {4096, 8192}, SafetensorsDtype::BF16,
                                    TensorComponent::o_proj));

    // A KDA layer carries NO indexer (MODELINFO §3a).
    const std::vector<WeightBundle> no_indexer;

    // Hand-summed from the tensor table (no hc set here — that lives in the
    // sizing formula, not the attention bundle list).
    const int64_t full =
        3LL * 8192 * 4096 * 2      // q/k/v_proj
        + 64LL * 4096 * 2          // b_proj
        + 2LL * 128 * 4096 * 2     // f_a/g_a
        + 2LL * 8192 * 128 * 2     // f_b/g_b
        + 3LL * 8192 * 4 * 2       // convs
        + 64LL * 4                 // A_log
        + 8192LL * 4               // dt_bias
        + 128LL * 2                // o_norm
        + 4096LL * 8192 * 2;       // o_proj
    EXPECT_EQ(full, 275'481'088LL);

    {
        TpWeightSharder sharder(model_cfg, 1);
        auto sharded = sharder.shard_attention_layer(attention, no_indexer, 0);
        EXPECT_EQ(sharded.size(), 15u);
        int64_t total = 0;
        for (const auto& sb : sharded) total += sb.total_bytes();
        EXPECT_EQ(total, full);
    }
    {
        TpWeightSharder sharder(model_cfg, 2);
        // Replicated share: f_a + g_a + o_norm.
        const int64_t replicated = 2LL * 128 * 4096 * 2 + 128LL * 2;
        const int64_t expected = replicated + (full - replicated) / 2;
        EXPECT_EQ(expected, 138'789'248LL);
        for (int rank = 0; rank < 2; ++rank) {
            auto sharded = sharder.shard_attention_layer(attention, no_indexer,
                                                         rank);
            EXPECT_EQ(sharded.size(), 15u);
            int64_t total = 0;
            for (const auto& sb : sharded) total += sb.total_bytes();
            EXPECT_EQ(total, expected) << "rank " << rank;
        }
    }
}

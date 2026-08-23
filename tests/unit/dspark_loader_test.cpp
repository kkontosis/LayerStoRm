// DSP-2 coverage gate: DSpark draft-checkpoint weight loading.
//
// Three tiers:
//   1. Synthetic tiny checkpoint (always runs): full slot coverage on a
//      generated speculators-v0.5 checkpoint + strictness (unmapped / missing /
//      shape / dtype all rejected — INV-DSPARK-CKPT), draft-GPU resolution,
//      config↔checkpoint cross-validation, LayerRegistry budget accounting.
//   2. Real checkpoint (test-data/GLM-5.2-speculator.dspark, skipped
//      gracefully when absent): ALL 64 tensors mapped, zero missing / zero
//      unmapped, shapes match the config-derived expectations, byte totals
//      exact.
//   3. Device placement (real checkpoint + CUDA GPU, skipped otherwise):
//      the 7.61 GB BF16 draft uploads onto the configured draft GPU as one
//      arena; readback spot-checks byte identity.

#include "model/weight_loader/dspark_loader.h"

#include "config/config_parser.h"
#include "compute/cuda_sm120_device_backend.h"
#include "core/device_backend.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/kgroup_quant.h"
#include "model/quantization/nvfp4.h"
#include "speculation/dspark_runtime.h"  // DSP-3: scratch sizing in budget

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace lc = layerstorm::config;
namespace lm = layerstorm::model;

namespace {

namespace fs = std::filesystem;

// ── Real checkpoint location ────────────────────────────────────────────────

fs::path real_checkpoint_dir() {
    return fs::path(LAYERSTORM_SOURCE_DIR) / "test-data" /
           "GLM-5.2-speculator.dspark";
}

bool real_checkpoint_present() {
    std::error_code ec;
    return fs::exists(real_checkpoint_dir() / "model.safetensors", ec);
}

bool has_cuda_gpu() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// ── Synthetic tiny checkpoint (speculators v0.5 layout, small dims) ─────────
// H=8, layers=2, heads=2, kv_heads=2, head_dim=4, I=16, V=32, r=4,
// aux ids [0,1] (N_aux=2), γ=8, speculative_tokens=7, confidence head on
// (with markov → conf input 8+4=12). 7 top-level + 2 confidence + 2*11 layer
// tensors = 31 expected slots.

struct TinyDims {
    int64_t H = 8, V = 32, r = 4, D = 4, I = 16;
    int layers = 2, heads = 2, kv_heads = 2;
};

nlohmann::json tiny_config_json() {
    TinyDims d;
    return nlohmann::json{
        {"speculators_model_type", "dspark"},
        {"block_size", 8},
        {"markov_rank", d.r},
        {"markov_head_type", "vanilla"},
        {"enable_confidence_head", true},
        {"confidence_head_with_markov", true},
        {"draft_vocab_size", d.V},
        {"mask_token_id", 5},
        {"max_anchors", 16},
        {"aux_hidden_state_layer_ids", {0, 1}},
        {"tie_word_embeddings", false},
        {"speculators_config",
         {{"proposal_methods",
           {{{"proposal_type", "greedy"}, {"speculative_tokens", 7}}}}}},
        {"transformer_layer_config",
         {{"model_type", "qwen3"},
          {"num_hidden_layers", d.layers},
          {"hidden_size", d.H},
          {"num_attention_heads", d.heads},
          {"num_key_value_heads", d.kv_heads},
          {"head_dim", d.D},
          {"intermediate_size", d.I},
          {"rms_norm_eps", 1e-5},
          {"vocab_size", d.V},
          {"rope_parameters", {{"rope_theta", 8000000.0}}}}},
    };
}

struct NamedTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype = "BF16";  // 2 bytes/elem for BF16; 4 for F32; 8 for I64
    std::vector<char> raw;       // explicit payload (e.g. d2t offsets);
                                 // empty = the ascending-byte pattern
};

/// The full expected tensor list for the tiny checkpoint.
std::vector<NamedTensor> tiny_tensor_list() {
    TinyDims d;
    const int64_t Q = d.heads * d.D;    // 8
    const int64_t KV = d.kv_heads * d.D;
    std::vector<NamedTensor> t = {
        {"embed_tokens.weight", {d.V, d.H}},
        {"lm_head.weight", {d.V, d.H}},
        {"fc.weight", {d.H, 2 * d.H}},
        {"hidden_norm.weight", {d.H}},
        {"norm.weight", {d.H}},
        {"markov_head.markov_w1.weight", {d.V, d.r}},
        {"markov_head.markov_w2.weight", {d.V, d.r}},
        {"confidence_head.proj.weight", {1, d.H + d.r}},
        {"confidence_head.proj.bias", {1}},
    };
    for (int l = 0; l < d.layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        t.push_back({p + "self_attn.q_proj.weight", {Q, d.H}});
        t.push_back({p + "self_attn.k_proj.weight", {KV, d.H}});
        t.push_back({p + "self_attn.v_proj.weight", {KV, d.H}});
        t.push_back({p + "self_attn.o_proj.weight", {d.H, Q}});
        t.push_back({p + "self_attn.q_norm.weight", {d.D}});
        t.push_back({p + "self_attn.k_norm.weight", {d.D}});
        t.push_back({p + "input_layernorm.weight", {d.H}});
        t.push_back({p + "post_attention_layernorm.weight", {d.H}});
        t.push_back({p + "mlp.gate_proj.weight", {d.I, d.H}});
        t.push_back({p + "mlp.up_proj.weight", {d.I, d.H}});
        t.push_back({p + "mlp.down_proj.weight", {d.H, d.I}});
    }
    return t;
}

int64_t dtype_bytes(const std::string& dtype) {
    if (dtype == "I64") return 8;
    return dtype == "F32" ? 4 : 2;
}

/// I64 payload helper (d2t draft->target offset tensors).
std::vector<char> i64_raw(const std::vector<int64_t>& v) {
    std::vector<char> raw(v.size() * 8);
    std::memcpy(raw.data(), v.data(), raw.size());
    return raw;
}

/// Write a minimal valid safetensors file (data = ascending bytes so device
/// readback can verify content identity).
void write_safetensors(const fs::path& path, const std::vector<NamedTensor>& tensors) {
    nlohmann::json header;
    int64_t offset = 0;
    for (const auto& t : tensors) {
        int64_t numel = 1;
        for (int64_t s : t.shape) numel *= s;
        const int64_t bytes = numel * dtype_bytes(t.dtype);
        header[t.name] = {{"dtype", t.dtype},
                          {"shape", t.shape},
                          {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    const std::string hj = header.dump();
    const uint64_t hlen = hj.size();

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(f.is_open()) << path;
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(hj.data(), static_cast<std::streamsize>(hj.size()));
    std::vector<char> data(static_cast<size_t>(offset));
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<char>(i % 251);
    // Explicit payloads (raw) overwrite the pattern at each tensor's offsets.
    int64_t off = 0;
    for (const auto& t : tensors) {
        int64_t numel = 1;
        for (int64_t s : t.shape) numel *= s;
        const int64_t bytes = numel * dtype_bytes(t.dtype);
        if (!t.raw.empty()) {
            ASSERT_EQ(static_cast<int64_t>(t.raw.size()), bytes) << t.name;
            std::memcpy(data.data() + off, t.raw.data(), t.raw.size());
        }
        off += bytes;
    }
    f.write(data.data(), static_cast<std::streamsize>(data.size()));
    ASSERT_TRUE(f.good());
}

/// Materialize a synthetic checkpoint dir under the gtest temp dir.
fs::path make_tiny_checkpoint(const std::string& tag,
                              const std::vector<NamedTensor>& tensors,
                              nlohmann::json cfg = tiny_config_json()) {
    fs::path dir = fs::path(::testing::TempDir()) / ("dspark_tiny_" + tag);
    fs::create_directories(dir);
    { std::ofstream(dir / "config.json") << cfg.dump(2); }
    write_safetensors(dir / "model.safetensors", tensors);
    return dir;
}

/// Engine DsparkConfig matching the tiny checkpoint.
lc::DsparkConfig tiny_engine_config() {
    lc::DsparkConfig d;
    d.block_size = 8;
    d.speculative_tokens = 7;
    d.markov_rank = 4;
    d.draft_vocab_size = 32;
    d.mask_token_id = 5;
    d.max_anchors = 16;
    d.aux_hidden_state_layer_ids = {0, 1};
    return d;
}

}  // namespace

// ── Tier 1: synthetic checkpoint ────────────────────────────────────────────

TEST(DsparkLoaderSynthetic, LoadsTinyCheckpointFullCoverage) {
    auto dir = make_tiny_checkpoint("ok", tiny_tensor_list());
    auto w = lm::load_dspark_draft(dir);

    EXPECT_EQ(w.total_tensors_loaded, 31);  // 9 top-level + 2*11 layer tensors
    EXPECT_EQ(w.ckpt.block_size, 8);
    EXPECT_EQ(w.ckpt.speculative_tokens, 7);
    EXPECT_EQ(w.ckpt.markov_rank, 4);
    EXPECT_EQ(w.ckpt.num_hidden_layers, 2);
    ASSERT_EQ(w.layers.size(), 2u);

    // Every slot filled, shapes as expected.
    EXPECT_EQ(w.embed_tokens.shape, (std::vector<int64_t>{32, 8}));
    EXPECT_EQ(w.lm_head.shape, (std::vector<int64_t>{32, 8}));
    EXPECT_EQ(w.fc.shape, (std::vector<int64_t>{8, 16}));
    EXPECT_EQ(w.markov_w1.shape, (std::vector<int64_t>{32, 4}));
    EXPECT_EQ(w.markov_w2.shape, (std::vector<int64_t>{32, 4}));
    EXPECT_EQ(w.confidence_proj_weight.shape, (std::vector<int64_t>{1, 12}));
    EXPECT_EQ(w.confidence_proj_bias.shape, (std::vector<int64_t>{1}));
    EXPECT_EQ(w.layers[1].down_proj.shape, (std::vector<int64_t>{8, 16}));
    EXPECT_NE(w.layers[0].q_norm.data.data(), nullptr);

    // Byte totals: sum over all entries; header-only scan agrees.
    int64_t expected = 0;
    for (const auto& t : tiny_tensor_list()) {
        int64_t numel = 1;
        for (int64_t s : t.shape) numel *= s;
        expected += numel * 2;
    }
    EXPECT_EQ(w.total_weight_bytes, expected);
    EXPECT_EQ(lm::dspark_draft_bytes(dir), expected);
}

TEST(DsparkLoaderSynthetic, RejectsUnmappedTensor) {
    auto tensors = tiny_tensor_list();
    tensors.push_back({"mask_embedding.weight", {8}});  // not in v0.5 format
    auto dir = make_tiny_checkpoint("unmapped", tensors);
    EXPECT_THROW(
        {
            try {
                lm::load_dspark_draft(dir);
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("unmapped"),
                          std::string::npos);
                EXPECT_NE(std::string(e.what()).find("mask_embedding.weight"),
                          std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST(DsparkLoaderSynthetic, RejectsMissingTensor) {
    auto tensors = tiny_tensor_list();
    std::erase_if(tensors, [](const NamedTensor& t) {
        return t.name == "markov_head.markov_w2.weight";
    });
    auto dir = make_tiny_checkpoint("missing", tensors);
    EXPECT_THROW(
        {
            try {
                lm::load_dspark_draft(dir);
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("missing"),
                          std::string::npos);
                EXPECT_NE(
                    std::string(e.what()).find("markov_head.markov_w2.weight"),
                    std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST(DsparkLoaderSynthetic, RejectsShapeMismatch) {
    auto tensors = tiny_tensor_list();
    for (auto& t : tensors)
        if (t.name == "fc.weight") t.shape = {8, 24};  // wrong N_aux*H
    auto dir = make_tiny_checkpoint("shape", tensors);
    EXPECT_THROW(
        {
            try {
                lm::load_dspark_draft(dir);
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("shape"),
                          std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST(DsparkLoaderSynthetic, RejectsNonBf16Dtype) {
    auto tensors = tiny_tensor_list();
    for (auto& t : tensors)
        if (t.name == "hidden_norm.weight") t.dtype = "F32";
    auto dir = make_tiny_checkpoint("dtype", tensors);
    EXPECT_THROW(lm::load_dspark_draft(dir), std::runtime_error);
}

TEST(DsparkLoaderSynthetic, RejectsWrongSpeculatorsModelType) {
    auto cfg = tiny_config_json();
    cfg["speculators_model_type"] = "eagle3";
    auto dir = make_tiny_checkpoint("modeltype", tiny_tensor_list(), cfg);
    EXPECT_THROW(lm::parse_dspark_checkpoint_config(dir), std::runtime_error);
}

// ── Config ↔ checkpoint cross-validation ────────────────────────────────────

TEST(DsparkLoaderSynthetic, CrossValidationAcceptsMatchingConfig) {
    auto dir = make_tiny_checkpoint("xval_ok", tiny_tensor_list());
    auto ckpt = lm::parse_dspark_checkpoint_config(dir);
    EXPECT_NO_THROW(
        lm::validate_dspark_config_against_checkpoint(tiny_engine_config(), ckpt));
}

TEST(DsparkLoaderSynthetic, CrossValidationRejectsMismatch) {
    auto dir = make_tiny_checkpoint("xval_bad", tiny_tensor_list());
    auto ckpt = lm::parse_dspark_checkpoint_config(dir);

    auto d = tiny_engine_config();
    d.block_size = 5;  // stale DSP-1 default; checkpoint γ=8
    d.sts_temperatures = {};
    EXPECT_THROW(lm::validate_dspark_config_against_checkpoint(d, ckpt),
                 std::runtime_error);

    d = tiny_engine_config();
    d.aux_hidden_state_layer_ids = {8, 23, 39, 55, 70};  // GLM defaults ≠ tiny
    EXPECT_THROW(lm::validate_dspark_config_against_checkpoint(d, ckpt),
                 std::runtime_error);
}

// ── Reduced draft vocab (TD-DSPARK-VOCAB-REMAP) ─────────────────────────────
// Vd=16 < V=32: lm_head/markov_w2 shrink to Vd rows, a d2t [Vd] I64 offset
// tensor is REQUIRED (target_id = draft_id + d2t[draft_id], vLLM
// qwen3_dflash.py convention), embed_tokens/markov_w1 stay V rows.
// mask_token_id=20 deliberately sits in [Vd, V): it indexes embed_tokens
// (EMBED vocab), not the draft vocab.

namespace {

constexpr int64_t kVd = 16;  // reduced draft vocab (embed vocab V=32)

nlohmann::json reduced_config_json() {
    auto cfg = tiny_config_json();
    cfg["draft_vocab_size"] = kVd;
    cfg["mask_token_id"] = 20;  // >= Vd, < V: embed-vocab row
    return cfg;
}

/// Offsets j -> target 2j+1 (all non-zero, injective, max 31 < V=32).
std::vector<int64_t> reduced_d2t_offsets() {
    std::vector<int64_t> off(static_cast<size_t>(kVd));
    for (int64_t j = 0; j < kVd; ++j) off[static_cast<size_t>(j)] = j + 1;
    return off;
}

std::vector<NamedTensor> reduced_tensor_list() {
    TinyDims d;
    auto t = tiny_tensor_list();
    for (auto& nt : t) {
        if (nt.name == "lm_head.weight") nt.shape = {kVd, d.H};
        if (nt.name == "markov_head.markov_w2.weight") nt.shape = {kVd, d.r};
    }
    t.push_back({"d2t", {kVd}, "I64", i64_raw(reduced_d2t_offsets())});
    return t;
}

}  // namespace

TEST(DsparkLoaderReducedVocab, LoadsReducedCheckpointWithD2t) {
    auto dir = make_tiny_checkpoint("reduced_ok", reduced_tensor_list(),
                                    reduced_config_json());
    auto w = lm::load_dspark_draft(dir);

    EXPECT_EQ(w.total_tensors_loaded, 32);  // 31 + d2t
    EXPECT_EQ(w.ckpt.draft_vocab_size, kVd);
    EXPECT_EQ(w.ckpt.vocab_size, 32);
    EXPECT_EQ(w.ckpt.mask_token_id, 20);
    EXPECT_EQ(w.embed_tokens.shape, (std::vector<int64_t>{32, 8}));
    EXPECT_EQ(w.markov_w1.shape, (std::vector<int64_t>{32, 4}));
    EXPECT_EQ(w.lm_head.shape, (std::vector<int64_t>{kVd, 8}));
    EXPECT_EQ(w.markov_w2.shape, (std::vector<int64_t>{kVd, 4}));
    ASSERT_EQ(w.d2t.shape, (std::vector<int64_t>{kVd}));
    ASSERT_NE(w.d2t.data.data(), nullptr);

    // Byte-exact d2t payload through the mmap.
    const auto want = reduced_d2t_offsets();
    const auto* got = reinterpret_cast<const int64_t*>(w.d2t.data.data());
    for (int64_t j = 0; j < kVd; ++j)
        EXPECT_EQ(got[j], want[static_cast<size_t>(j)]) << "d2t[" << j << "]";
}

TEST(DsparkLoaderReducedVocab, RejectsMissingD2t) {
    auto tensors = reduced_tensor_list();
    std::erase_if(tensors,
                  [](const NamedTensor& t) { return t.name == "d2t"; });
    auto dir = make_tiny_checkpoint("reduced_missing_d2t", tensors,
                                    reduced_config_json());
    EXPECT_THROW(
        {
            try {
                lm::load_dspark_draft(dir);
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("missing"),
                          std::string::npos);
                EXPECT_NE(std::string(e.what()).find("d2t"),
                          std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST(DsparkLoaderReducedVocab, RejectsWrongD2tDtype) {
    auto tensors = reduced_tensor_list();
    for (auto& t : tensors)
        if (t.name == "d2t") {
            t.dtype = "BF16";  // must be I64
            t.raw.clear();
        }
    auto dir = make_tiny_checkpoint("reduced_d2t_dtype", tensors,
                                    reduced_config_json());
    EXPECT_THROW(
        {
            try {
                lm::load_dspark_draft(dir);
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("expected I64"),
                          std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST(DsparkLoaderReducedVocab, RejectsOutOfRangeD2tMapping) {
    auto offsets = reduced_d2t_offsets();
    offsets[3] = 32 - 3;  // draft 3 -> target 32 == V: out of range
    auto tensors = reduced_tensor_list();
    for (auto& t : tensors)
        if (t.name == "d2t") t.raw = i64_raw(offsets);
    auto dir = make_tiny_checkpoint("reduced_d2t_range", tensors,
                                    reduced_config_json());
    EXPECT_THROW(
        {
            try {
                lm::load_dspark_draft(dir);
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("outside target vocab"),
                          std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST(DsparkLoaderReducedVocab, RejectsD2tInFullVocabCheckpoint) {
    // Full-vocab checkpoints must NOT ship d2t (strict coverage: unmapped).
    auto tensors = tiny_tensor_list();
    tensors.push_back({"d2t", {32}, "I64",
                       i64_raw(std::vector<int64_t>(32, 0))});
    auto dir = make_tiny_checkpoint("full_with_d2t", tensors);
    EXPECT_THROW(
        {
            try {
                lm::load_dspark_draft(dir);
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("unmapped"),
                          std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST(DsparkLoaderReducedVocab, RejectsDraftVocabLargerThanEmbedVocab) {
    auto cfg = tiny_config_json();
    cfg["draft_vocab_size"] = 64;  // > transformer vocab_size 32
    auto dir = make_tiny_checkpoint("reduced_too_big", tiny_tensor_list(), cfg);
    EXPECT_THROW(lm::parse_dspark_checkpoint_config(dir), std::runtime_error);
}

// ── Draft GPU resolution ────────────────────────────────────────────────────

namespace {

lc::Config four_gpu_config(std::vector<int> tp_array, std::vector<int> draft_gpus) {
    auto j = nlohmann::json{
        {"model",
         {{"architecture", "glm_moe_dsa"},
          {"weights_path", "/data/models/glm-5.2/"},
          {"weights_format", "safetensors"},
          {"num_hidden_layers", 78},
          {"hidden_size", 6144},
          {"num_attention_heads", 64},
          {"num_key_value_heads", 64},
          {"intermediate_size", 12288},
          {"n_routed_experts", 256},
          {"n_shared_experts", 1},
          {"num_experts_per_tok", 8},
          {"n_group", 1},
          {"topk_group", 1},
          {"vocab_size", 154880},
          {"max_position_embeddings", 202752},
          {"kv_lora_rank", 512},
          {"q_lora_rank", 2048},
          {"qk_rope_head_dim", 64},
          {"qk_nope_head_dim", 192},
          {"v_head_dim", 256},
          {"first_k_dense_replace", 3},
          {"moe_layer_freq", 1},
          {"index_topk", 2048},
          {"index_n_heads", 32},
          {"index_head_dim", 128},
          {"num_nextn_predict_layers", 1},
          {"rms_norm_eps", 1e-5},
          {"rope_theta", 1000000.0},
          {"routed_scaling_factor", 2.5},
          {"moe_intermediate_size", 2048}}},
        {"quantization",
         {{"weights", "nvfp4"},
          {"attention_compute", "fp8_e4m3"},
          {"kv_cache", "fp8_e4m3"},
          {"gating_compute", "fp32"}}},
        {"hardware",
         {{"gpus",
           {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}},
            {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 32}},
            {{"id", 2}, {"type", "rtx5080"}, {"vram_gb", 16}},
            {{"id", 3}, {"type", "rtx5080"}, {"vram_gb", 16}}}},
          {"system_ram_gb", 256}}},
    };
    auto cfg = lc::parse_config(j);
    cfg.hardware.tp_array = std::move(tp_array);
    cfg.speculation.method = lc::SpeculationMethodType::dspark;
    cfg.speculation.dspark.draft_gpus = std::move(draft_gpus);
    return cfg;
}

}  // namespace

TEST(DsparkDraftGpu, ExplicitDraftGpuWins) {
    auto cfg = four_gpu_config({0, 1}, {3});
    EXPECT_EQ(lm::resolve_dspark_draft_gpu(cfg), 3);
}

TEST(DsparkDraftGpu, AutoSelectsFirstNonTpGpu) {
    auto cfg = four_gpu_config({0, 1}, {});
    EXPECT_EQ(lm::resolve_dspark_draft_gpu(cfg), 2);  // first 5080
}

TEST(DsparkDraftGpu, ThrowsWhenAllGpusInTp) {
    auto cfg = four_gpu_config({0, 1, 2, 3}, {});
    EXPECT_THROW(lm::resolve_dspark_draft_gpu(cfg), std::runtime_error);
}

TEST(DsparkDraftGpu, ThrowsOnOutOfRangeExplicitGpu) {
    auto cfg = four_gpu_config({0, 1}, {7});
    EXPECT_THROW(lm::resolve_dspark_draft_gpu(cfg), std::runtime_error);
}

// ── LayerRegistry budget accounting (synthetic checkpoint) ──────────────────

TEST(DsparkBudget, ChargesDraftBytesOnDraftGpu) {
    auto dir = make_tiny_checkpoint("budget", tiny_tensor_list());
    auto cfg = four_gpu_config({0, 1}, {});
    cfg.speculation.dspark.checkpoint_path = dir.string();

    // DSP-3: the budget charges weights + the DsparkRuntime device scratch
    // (aux staging, context-KV arena, query buffers).
    const int64_t draft_bytes =
        lm::dspark_draft_bytes(dir) +
        layerstorm::speculation::dspark_runtime_scratch_bytes(
            cfg, lm::parse_dspark_checkpoint_config(dir)) +
        layerstorm::speculation::kDsparkArenaAlignSlack;
    ASSERT_GT(draft_bytes, 0);

    lm::ModelConfig model_cfg(cfg);
    lm::Nvfp4 nvfp4;
    lm::LayerRegistry reg(model_cfg, cfg, nvfp4);

    EXPECT_EQ(reg.dspark_draft_gpu(), 2);  // first non-TP GPU
    EXPECT_EQ(reg.dspark_draft_bytes(), draft_bytes);

    auto budgets = reg.estimate_gpu_budgets();
    ASSERT_EQ(budgets.size(), 4u);

    // Baseline without dspark: same config, method=none.
    auto cfg_none = four_gpu_config({0, 1}, {});
    cfg_none.speculation.method = lc::SpeculationMethodType::none;
    lm::ModelConfig model_none(cfg_none);
    lm::LayerRegistry reg_none(model_none, cfg_none, nvfp4);
    auto base = reg_none.estimate_gpu_budgets();

    EXPECT_EQ(budgets[2].pinned_bytes, base[2].pinned_bytes + draft_bytes);
    EXPECT_EQ(budgets[2].available_for_cache_bytes,
              base[2].available_for_cache_bytes - draft_bytes);
    for (int i : {0, 1, 3}) {
        EXPECT_EQ(budgets[static_cast<size_t>(i)].pinned_bytes,
                  base[static_cast<size_t>(i)].pinned_bytes)
            << "gpu " << i;
    }
}

TEST(DsparkBudget, RegistryFailsClosedOnMissingCheckpoint) {
    auto cfg = four_gpu_config({0, 1}, {});
    cfg.speculation.dspark.checkpoint_path = "/nonexistent/dspark-checkpoint";
    lm::ModelConfig model_cfg(cfg);
    lm::Nvfp4 nvfp4;
    EXPECT_THROW(lm::LayerRegistry(model_cfg, cfg, nvfp4), std::runtime_error);
}

// ── TD-DSPARK-DRAFT-SHARD: shard classification + per-rank budget ───────────

TEST(DsparkShard, ShardKindClassification) {
    using K = lm::DsparkShardKind;
    EXPECT_EQ(lm::dspark_tensor_shard_kind("lm_head.weight"), K::kColParallel);
    EXPECT_EQ(lm::dspark_tensor_shard_kind("layers.0.self_attn.q_proj.weight"),
              K::kColParallel);
    EXPECT_EQ(lm::dspark_tensor_shard_kind("layers.4.mlp.up_proj.weight"),
              K::kColParallel);
    EXPECT_EQ(lm::dspark_tensor_shard_kind("layers.1.self_attn.o_proj.weight"),
              K::kRowParallel);
    EXPECT_EQ(lm::dspark_tensor_shard_kind("layers.3.mlp.down_proj.weight"),
              K::kRowParallel);
    EXPECT_EQ(lm::dspark_tensor_shard_kind("layers.0.self_attn.q_norm.weight"),
              K::kReplicated);
    EXPECT_EQ(lm::dspark_tensor_shard_kind("layers.0.input_layernorm.weight"),
              K::kReplicated);
    EXPECT_EQ(lm::dspark_tensor_shard_kind("norm.weight"), K::kReplicated);
    for (const char* n :
         {"embed_tokens.weight", "fc.weight", "hidden_norm.weight",
          "markov_head.markov_w1.weight", "markov_head.markov_w2.weight",
          "confidence_head.proj.weight", "confidence_head.proj.bias", "d2t"})
        EXPECT_EQ(lm::dspark_tensor_shard_kind(n), K::kPrimaryOnly) << n;
}

TEST(DsparkShard, ShardShapeSplitsAndValidates) {
    const auto bf16 = static_cast<lc::DsparkDraftWeightsQuant>(0);
    // Col-parallel: output rows split.
    EXPECT_EQ(lm::dspark_shard_shape("lm_head.weight", {32, 8}, 0, 2, bf16),
              (std::vector<int64_t>{16, 8}));
    // Row-parallel: K columns split.
    EXPECT_EQ(lm::dspark_shard_shape("layers.0.self_attn.o_proj.weight",
                                     {8, 8}, 1, 2, bf16),
              (std::vector<int64_t>{8, 4}));
    // Replicated: whole shape on every rank.
    EXPECT_EQ(lm::dspark_shard_shape("norm.weight", {8}, 1, 2, bf16),
              (std::vector<int64_t>{8}));
    // Primary-only: empty on rank > 0, whole on rank 0.
    EXPECT_TRUE(
        lm::dspark_shard_shape("embed_tokens.weight", {32, 8}, 1, 2, bf16)
            .empty());
    EXPECT_EQ(
        lm::dspark_shard_shape("embed_tokens.weight", {32, 8}, 0, 2, bf16),
        (std::vector<int64_t>{32, 8}));
    // num_ranks == 1: identity.
    EXPECT_EQ(lm::dspark_shard_shape("lm_head.weight", {32, 8}, 0, 1, bf16),
              (std::vector<int64_t>{32, 8}));
    // Indivisible dims fail closed.
    EXPECT_THROW(
        lm::dspark_shard_shape("lm_head.weight", {33, 8}, 0, 2, bf16),
        std::runtime_error);
    // Quantized row-parallel shard whose K/nr is not a scale-group multiple
    // fails closed (INV-DSPARK-QUANT group-alignment clause): nvfp4 group 16.
    const auto nvfp4 = lc::DsparkDraftWeightsQuant::nvfp4;
    EXPECT_THROW(lm::dspark_shard_shape("layers.0.self_attn.o_proj.weight",
                                        {8, 8}, 0, 2, nvfp4),
                 std::runtime_error);
    // Group-aligned quant shard passes (K/nr = 2048 % 16 == 0).
    EXPECT_EQ(lm::dspark_shard_shape("layers.0.self_attn.o_proj.weight",
                                     {6144, 4096}, 1, 2, nvfp4),
              (std::vector<int64_t>{6144, 2048}));
}

TEST(DsparkShard, PerRankBytesSumToWholePlusReplicated) {
    auto dir = make_tiny_checkpoint("shardbytes", tiny_tensor_list());
    const auto bf16 = static_cast<lc::DsparkDraftWeightsQuant>(0);
    const int64_t whole = lm::dspark_draft_bytes(dir, bf16);
    const int64_t r0 = lm::dspark_draft_bytes(dir, bf16, 0, 2);
    const int64_t r1 = lm::dspark_draft_bytes(dir, bf16, 1, 2);
    // Col/row-parallel tensors split exactly; kPrimaryOnly count once; the
    // kReplicated norms count on BOTH ranks — the sum exceeds the whole by
    // exactly the replicated bytes: per layer q_norm[D]+k_norm[D]+2*ln[H],
    // plus the final norm [H], all BF16.
    TinyDims d;
    const int64_t repl =
        (d.layers * (2 * d.D + 2 * d.H) + d.H) * 2;
    EXPECT_EQ(r0 + r1, whole + repl);
    EXPECT_GT(r0, r1);  // rank 0 carries embed/fc/markov/confidence
}

TEST(DsparkShard, TinyQuantShardRejectsMisalignedGroups) {
    // The tiny checkpoint's o_proj K shard (8/2 = 4) is not an nvfp4
    // group-16 multiple — per-rank budget under quant must fail closed.
    auto dir = make_tiny_checkpoint("shardq", tiny_tensor_list());
    EXPECT_THROW(lm::dspark_draft_bytes(
                     dir, lc::DsparkDraftWeightsQuant::nvfp4, 0, 2),
                 std::runtime_error);
}

TEST(DsparkDraftGpu, ResolveListReturnsAllValidatedEntries) {
    auto cfg = four_gpu_config({0, 1}, {0, 1});
    EXPECT_EQ(lm::resolve_dspark_draft_gpus(cfg), (std::vector<int>{0, 1}));
    auto cfg_auto = four_gpu_config({0, 1}, {});
    EXPECT_EQ(lm::resolve_dspark_draft_gpus(cfg_auto),
              (std::vector<int>{2}));
    auto cfg_dup = four_gpu_config({0, 1}, {1, 1});
    EXPECT_THROW(lm::resolve_dspark_draft_gpus(cfg_dup), std::runtime_error);
    auto cfg_oob = four_gpu_config({0, 1}, {0, 7});
    EXPECT_THROW(lm::resolve_dspark_draft_gpus(cfg_oob), std::runtime_error);
}

TEST(DsparkBudget, ShardedChargesPerRankGpu) {
    auto dir = make_tiny_checkpoint("budget2", tiny_tensor_list());
    auto cfg = four_gpu_config({0, 1}, {0, 1});
    cfg.speculation.dspark.checkpoint_path = dir.string();

    std::array<int64_t, 2> charges{};
    for (int r = 0; r < 2; ++r) {
        charges[static_cast<size_t>(r)] =
            lm::dspark_draft_bytes(
                dir, cfg.speculation.dspark.draft_weights_quant, r, 2) +
            layerstorm::speculation::dspark_runtime_scratch_bytes(
                cfg, lm::parse_dspark_checkpoint_config(dir), r, 2) +
            layerstorm::speculation::kDsparkArenaAlignSlack;
        ASSERT_GT(charges[static_cast<size_t>(r)], 0);
    }

    lm::ModelConfig model_cfg(cfg);
    lm::Nvfp4 nvfp4;
    lm::LayerRegistry reg(model_cfg, cfg, nvfp4);
    EXPECT_EQ(reg.dspark_rank_gpus(), (std::vector<int>{0, 1}));
    ASSERT_EQ(reg.dspark_rank_charges().size(), 2u);
    EXPECT_EQ(reg.dspark_rank_charges()[0], charges[0]);
    EXPECT_EQ(reg.dspark_rank_charges()[1], charges[1]);

    auto budgets = reg.estimate_gpu_budgets();
    ASSERT_EQ(budgets.size(), 4u);
    auto cfg_none = four_gpu_config({0, 1}, {});
    cfg_none.speculation.method = lc::SpeculationMethodType::none;
    lm::ModelConfig model_none(cfg_none);
    lm::LayerRegistry reg_none(model_none, cfg_none, nvfp4);
    auto base = reg_none.estimate_gpu_budgets();
    for (int i : {0, 1}) {
        EXPECT_EQ(budgets[static_cast<size_t>(i)].pinned_bytes,
                  base[static_cast<size_t>(i)].pinned_bytes +
                      charges[static_cast<size_t>(i)])
            << "gpu " << i;
    }
    for (int i : {2, 3}) {
        EXPECT_EQ(budgets[static_cast<size_t>(i)].pinned_bytes,
                  base[static_cast<size_t>(i)].pinned_bytes)
            << "gpu " << i;
    }
}

// ── Tier 2: real checkpoint (skip when absent) ──────────────────────────────

TEST(DsparkLoaderRealCheckpoint, ParsesCheckpointConfig) {
    if (!real_checkpoint_present())
        GTEST_SKIP() << "DSpark checkpoint not present at "
                     << real_checkpoint_dir();
    auto c = lm::parse_dspark_checkpoint_config(real_checkpoint_dir());
    EXPECT_EQ(c.block_size, 16);  // glm-5.2-dspark-preview (shipped ckpt was 8)
    EXPECT_EQ(c.markov_rank, 256);
    EXPECT_EQ(c.markov_head_type, "vanilla");
    EXPECT_TRUE(c.enable_confidence_head);
    EXPECT_TRUE(c.confidence_head_with_markov);
    EXPECT_EQ(c.draft_vocab_size, 154880);
    EXPECT_EQ(c.mask_token_id, 154856);
    EXPECT_EQ(c.max_anchors, 1024);
    EXPECT_EQ(c.aux_hidden_state_layer_ids, (std::vector<int>{8, 23, 39, 55, 70}));
    EXPECT_FALSE(c.tie_word_embeddings);
    EXPECT_EQ(c.speculative_tokens, 15);  // glm-5.2-dspark-preview (shipped ckpt was 7)
    EXPECT_EQ(c.model_type, "qwen3");
    EXPECT_EQ(c.num_hidden_layers, 5);
    EXPECT_EQ(c.hidden_size, 6144);
    EXPECT_EQ(c.num_attention_heads, 64);
    EXPECT_EQ(c.num_key_value_heads, 64);
    EXPECT_EQ(c.head_dim, 64);
    EXPECT_EQ(c.intermediate_size, 12288);
    EXPECT_DOUBLE_EQ(c.rope_theta, 8000000.0);
}

TEST(DsparkLoaderRealCheckpoint, CoverageAll64TensorsMappedZeroMissingZeroUnmapped) {
    if (!real_checkpoint_present())
        GTEST_SKIP() << "DSpark checkpoint not present at "
                     << real_checkpoint_dir();

    // load_dspark_draft throws on ANY unmapped/missing/shape/dtype offender —
    // reaching the assertions below proves zero of each (INV-DSPARK-CKPT).
    auto w = lm::load_dspark_draft(real_checkpoint_dir());

    EXPECT_EQ(w.total_tensors_loaded, 64);
    EXPECT_EQ(static_cast<int>(w.shard.entries().size()), 64);
    ASSERT_EQ(w.layers.size(), 5u);

    // Shapes match the checkpoint ground truth (dspark_loader.h header).
    EXPECT_EQ(w.embed_tokens.shape, (std::vector<int64_t>{154880, 6144}));
    EXPECT_EQ(w.lm_head.shape, (std::vector<int64_t>{154880, 6144}));
    EXPECT_EQ(w.fc.shape, (std::vector<int64_t>{6144, 30720}));
    EXPECT_EQ(w.hidden_norm.shape, (std::vector<int64_t>{6144}));
    EXPECT_EQ(w.final_norm.shape, (std::vector<int64_t>{6144}));
    EXPECT_EQ(w.markov_w1.shape, (std::vector<int64_t>{154880, 256}));
    EXPECT_EQ(w.markov_w2.shape, (std::vector<int64_t>{154880, 256}));
    EXPECT_EQ(w.confidence_proj_weight.shape, (std::vector<int64_t>{1, 6400}));
    EXPECT_EQ(w.confidence_proj_bias.shape, (std::vector<int64_t>{1}));
    for (const auto& L : w.layers) {
        EXPECT_EQ(L.q_proj.shape, (std::vector<int64_t>{4096, 6144}));
        EXPECT_EQ(L.k_proj.shape, (std::vector<int64_t>{4096, 6144}));
        EXPECT_EQ(L.v_proj.shape, (std::vector<int64_t>{4096, 6144}));
        EXPECT_EQ(L.o_proj.shape, (std::vector<int64_t>{6144, 4096}));
        EXPECT_EQ(L.q_norm.shape, (std::vector<int64_t>{64}));
        EXPECT_EQ(L.k_norm.shape, (std::vector<int64_t>{64}));
        EXPECT_EQ(L.input_layernorm.shape, (std::vector<int64_t>{6144}));
        EXPECT_EQ(L.post_attention_layernorm.shape, (std::vector<int64_t>{6144}));
        EXPECT_EQ(L.gate_proj.shape, (std::vector<int64_t>{12288, 6144}));
        EXPECT_EQ(L.up_proj.shape, (std::vector<int64_t>{12288, 6144}));
        EXPECT_EQ(L.down_proj.shape, (std::vector<int64_t>{6144, 12288}));
        EXPECT_EQ(L.q_proj.dtype, lm::SafetensorsDtype::BF16);
    }

    // Byte totals exact: sum over header entries == loaded total ==
    // header-only budget scan.
    int64_t sum = 0;
    for (const auto& e : w.shard.entries())
        sum += static_cast<int64_t>(e.data_size_bytes);
    EXPECT_EQ(w.total_weight_bytes, sum);
    EXPECT_EQ(lm::dspark_draft_bytes(real_checkpoint_dir()), sum);
    // 7.61 GB BF16 draft (the 5080 placement rationale).
    EXPECT_GT(sum, int64_t{7} * 1000 * 1000 * 1000);
    EXPECT_LT(sum, int64_t{8} * 1000 * 1000 * 1000);
}

TEST(DsparkLoaderRealCheckpoint, DefaultEngineConfigMatchesCheckpoint) {
    if (!real_checkpoint_present())
        GTEST_SKIP() << "DSpark checkpoint not present at "
                     << real_checkpoint_dir();
    auto ckpt = lm::parse_dspark_checkpoint_config(real_checkpoint_dir());
    // A config matching the checkpoint's γ/speculative_tokens (the rest are the
    // DSP-2 schema defaults: r=256, V=154880, mask 154856, anchors 1024,
    // aux [8,23,39,55,70]) must accept the shipped checkpoint out of the box.
    // (γ/spec are derived because the symlinked model varies: shipped ckpt was
    // γ=8/spec=7; glm-5.2-dspark-preview is γ=16/spec=15.)
    lc::DsparkConfig cfg;
    cfg.block_size = ckpt.block_size;
    cfg.speculative_tokens = ckpt.speculative_tokens;
    EXPECT_NO_THROW(
        lm::validate_dspark_config_against_checkpoint(cfg, ckpt));
}

// ── Tier 3: device placement (real checkpoint + CUDA GPU) ──────────────────

TEST(DsparkLoaderRealCheckpoint, UploadsDraftOntoDraftGpu) {
    if (!real_checkpoint_present())
        GTEST_SKIP() << "DSpark checkpoint not present at "
                     << real_checkpoint_dir();
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";

    auto w = lm::load_dspark_draft(real_checkpoint_dir());

    // Enough free VRAM on device 0 (run under CUDA_VISIBLE_DEVICES=0,1 with
    // PCI_BUS_ID order → the 5080s; a squatting process may shrink this).
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    size_t free_b = 0, total_b = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_b, &total_b), cudaSuccess);
    if (static_cast<int64_t>(free_b) < w.total_weight_bytes + (int64_t{1} << 30))
        GTEST_SKIP() << "Insufficient free VRAM on device 0: " << free_b;

    auto backend = layerstorm::compute::make_cuda_sm120_device_backend(
        layerstorm::config::GpuRef{.position = 0, .id = 0,
                                   .type = layerstorm::config::GpuType::rtx5080});
    auto dev = lm::upload_dspark_draft(w, *backend);

    EXPECT_EQ(dev.total_tensors_uploaded, 64);
    EXPECT_NE(dev.arena, nullptr);
    EXPECT_GE(dev.arena_bytes, w.total_weight_bytes);
    // 256-byte alignment padding is bounded: 64 tensors * 255 bytes.
    EXPECT_LE(dev.arena_bytes, w.total_weight_bytes + 64 * 255);
    ASSERT_EQ(dev.layers.size(), 5u);
    EXPECT_NE(dev.embed_tokens.ptr, nullptr);
    EXPECT_EQ(dev.embed_tokens.bytes,
              int64_t{154880} * 6144 * 2);  // BF16 preserved
    EXPECT_EQ(dev.markov_w2.shape, (std::vector<int64_t>{154880, 256}));
    EXPECT_NE(dev.layers[4].down_proj.ptr, nullptr);

    // Readback spot-checks: byte identity host ↔ device.
    auto check_roundtrip = [&](const lm::RawTensor& host,
                               const lm::DsparkDeviceTensor& devt,
                               size_t probe_bytes) {
        ASSERT_EQ(static_cast<int64_t>(host.data.size()), devt.bytes);
        probe_bytes = std::min(probe_bytes, host.data.size());
        std::vector<std::byte> buf(probe_bytes);
        ASSERT_EQ(cudaMemcpy(buf.data(), devt.ptr, probe_bytes,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(std::memcmp(buf.data(), host.data.data(), probe_bytes), 0);
        // Tail probe too (catches offset bugs).
        ASSERT_EQ(cudaMemcpy(buf.data(),
                             static_cast<const std::byte*>(devt.ptr) +
                                 devt.bytes - static_cast<int64_t>(probe_bytes),
                             probe_bytes, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(std::memcmp(buf.data(),
                              host.data.data() + host.data.size() - probe_bytes,
                              probe_bytes),
                  0);
    };
    check_roundtrip(w.embed_tokens, dev.embed_tokens, 4096);
    check_roundtrip(w.fc, dev.fc, 4096);
    check_roundtrip(w.markov_w1, dev.markov_w1, 4096);
    check_roundtrip(w.confidence_proj_bias, dev.confidence_proj_bias, 2);
    check_roundtrip(w.layers[2].gate_proj, dev.layers[2].gate_proj, 4096);
}

// Synthetic device upload (any CUDA GPU, no checkpoint required): verifies the
// arena layout + per-tensor offsets end-to-end at tiny scale.
TEST(DsparkLoaderSynthetic, UploadsTinyDraftAndReadsBack) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    auto dir = make_tiny_checkpoint("upload", tiny_tensor_list());
    auto w = lm::load_dspark_draft(dir);

    auto backend = layerstorm::compute::make_cuda_sm120_device_backend(
        layerstorm::config::GpuRef{.position = 0, .id = 0,
                                   .type = layerstorm::config::GpuType::rtx5080});
    auto dev = lm::upload_dspark_draft(w, *backend);
    EXPECT_EQ(dev.total_tensors_uploaded, 31);
    EXPECT_GE(dev.arena_bytes, w.total_weight_bytes);

    // Full-tensor readback identity on every slot.
    auto check = [&](const lm::RawTensor& host, const lm::DsparkDeviceTensor& d) {
        ASSERT_EQ(static_cast<int64_t>(host.data.size()), d.bytes);
        std::vector<std::byte> buf(host.data.size());
        ASSERT_EQ(cudaMemcpy(buf.data(), d.ptr, buf.size(),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(std::memcmp(buf.data(), host.data.data(), buf.size()), 0);
    };
    check(w.embed_tokens, dev.embed_tokens);
    check(w.lm_head, dev.lm_head);
    check(w.fc, dev.fc);
    check(w.hidden_norm, dev.hidden_norm);
    check(w.final_norm, dev.final_norm);
    check(w.markov_w1, dev.markov_w1);
    check(w.markov_w2, dev.markov_w2);
    check(w.confidence_proj_weight, dev.confidence_proj_weight);
    check(w.confidence_proj_bias, dev.confidence_proj_bias);
    for (size_t l = 0; l < w.layers.size(); ++l) {
        check(w.layers[l].q_proj, dev.layers[l].q_proj);
        check(w.layers[l].k_proj, dev.layers[l].k_proj);
        check(w.layers[l].v_proj, dev.layers[l].v_proj);
        check(w.layers[l].o_proj, dev.layers[l].o_proj);
        check(w.layers[l].q_norm, dev.layers[l].q_norm);
        check(w.layers[l].k_norm, dev.layers[l].k_norm);
        check(w.layers[l].input_layernorm, dev.layers[l].input_layernorm);
        check(w.layers[l].post_attention_layernorm,
              dev.layers[l].post_attention_layernorm);
        check(w.layers[l].gate_proj, dev.layers[l].gate_proj);
        check(w.layers[l].up_proj, dev.layers[l].up_proj);
        check(w.layers[l].down_proj, dev.layers[l].down_proj);
    }
}

// ── TD-DSPARK-DRAFT-QUANT: requant-at-upload ────────────────────────────────

TEST(DsparkQuant, GemmOperandClassification) {
    EXPECT_TRUE(lm::dspark_tensor_is_gemm_operand("lm_head.weight"));
    EXPECT_TRUE(lm::dspark_tensor_is_gemm_operand("fc.weight"));
    EXPECT_TRUE(lm::dspark_tensor_is_gemm_operand(
        "layers.3.self_attn.q_proj.weight"));
    EXPECT_TRUE(lm::dspark_tensor_is_gemm_operand(
        "layers.0.mlp.down_proj.weight"));
    EXPECT_FALSE(lm::dspark_tensor_is_gemm_operand("embed_tokens.weight"));
    EXPECT_FALSE(lm::dspark_tensor_is_gemm_operand(
        "markov_head.markov_w1.weight"));
    EXPECT_FALSE(lm::dspark_tensor_is_gemm_operand(
        "layers.1.self_attn.q_norm.weight"));
    EXPECT_FALSE(lm::dspark_tensor_is_gemm_operand(
        "layers.1.input_layernorm.weight"));
    EXPECT_FALSE(lm::dspark_tensor_is_gemm_operand(
        "confidence_head.proj.weight"));
    EXPECT_FALSE(lm::dspark_tensor_is_gemm_operand("d2t"));
}

TEST(DsparkQuant, BudgetBytesMatchQuantSizing) {
    namespace kg = layerstorm::model::kgroup;
    auto dir = make_tiny_checkpoint("qbudget", tiny_tensor_list());

    const int64_t bf16 = lm::dspark_draft_bytes(dir);
    int64_t fp8_expect = 0, nvfp4_expect = 0;
    for (const auto& t : tiny_tensor_list()) {
        int64_t numel = 1;
        for (int64_t s : t.shape) numel *= s;
        const int64_t raw = numel * dtype_bytes(t.dtype);
        if (lm::dspark_tensor_is_gemm_operand(t.name) && t.shape.size() == 2) {
            const int64_t n = t.shape[0], k = t.shape[1];
            fp8_expect += kg::fp8_weight_bytes(n, k) +
                          kg::fp8_scale_bytes(n, k);
            nvfp4_expect += kg::nvfp4_weight_bytes(n, k) +
                            kg::nvfp4_scale_bytes(n, k);
        } else {
            fp8_expect += raw;
            nvfp4_expect += raw;
        }
    }
    EXPECT_EQ(lm::dspark_draft_bytes(
                  dir, lc::DsparkDraftWeightsQuant::fp8_e4m3),
              fp8_expect);
    EXPECT_EQ(lm::dspark_draft_bytes(dir, lc::DsparkDraftWeightsQuant::nvfp4),
              nvfp4_expect);
    EXPECT_LT(fp8_expect, bf16);
    EXPECT_LT(nvfp4_expect, fp8_expect);
}

// Quantized tiny upload (GPU): GEMM operands carry dtype + scales, the
// device bytes match the packed sizing, and a readback DEQUANT of every
// quantized tensor reproduces the BF16 source within the format band
// (per-group amax/28 for FP8, amax/3 for NVFP4 — kgroup_quant_test bounds).
// Non-GEMM tensors stay byte-identical BF16.
TEST(DsparkQuant, UploadsQuantizedTinyDraftAndDequantMatches) {
    namespace kg = layerstorm::model::kgroup;
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    auto dir = make_tiny_checkpoint("qupload", tiny_tensor_list());
    auto w = lm::load_dspark_draft(dir);

    auto backend = layerstorm::compute::make_cuda_sm120_device_backend(
        layerstorm::config::GpuRef{.position = 0, .id = 0,
                                   .type = layerstorm::config::GpuType::rtx5080});

    for (auto quant : {lc::DsparkDraftWeightsQuant::fp8_e4m3,
                       lc::DsparkDraftWeightsQuant::nvfp4}) {
        const bool fp8 = quant == lc::DsparkDraftWeightsQuant::fp8_e4m3;
        auto dev = lm::upload_dspark_draft(w, *backend, nullptr, 0, quant);
        EXPECT_EQ(dev.total_tensors_uploaded, 31);

        // Non-GEMM tensors: BF16 verbatim.
        EXPECT_EQ(dev.embed_tokens.dtype, lm::DsparkWeightDtype::kBF16);
        EXPECT_EQ(dev.embed_tokens.scales, nullptr);
        EXPECT_EQ(dev.markov_w1.dtype, lm::DsparkWeightDtype::kBF16);
        {
            std::vector<std::byte> buf(w.embed_tokens.data.size());
            ASSERT_EQ(cudaMemcpy(buf.data(), dev.embed_tokens.ptr,
                                 buf.size(), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(std::memcmp(buf.data(), w.embed_tokens.data.data(),
                                  buf.size()),
                      0);
        }

        // GEMM operands: quantized + scaled; dequant matches the source.
        auto check_quant = [&](const lm::RawTensor& host,
                               const lm::DsparkDeviceTensor& d) {
            ASSERT_EQ(d.dtype, fp8 ? lm::DsparkWeightDtype::kFp8E4M3
                                   : lm::DsparkWeightDtype::kNvfp4);
            ASSERT_NE(d.scales, nullptr);
            const int64_t n = host.shape[0], k = host.shape[1];
            const int group = fp8 ? kg::kFp8GroupSize : kg::kNvfp4GroupSize;
            EXPECT_EQ(d.k_groups, (k + group - 1) / group);
            EXPECT_EQ(d.bytes, fp8 ? kg::fp8_weight_bytes(n, k)
                                   : kg::nvfp4_weight_bytes(n, k));
            EXPECT_EQ(d.scales_bytes, fp8 ? kg::fp8_scale_bytes(n, k)
                                          : kg::nvfp4_scale_bytes(n, k));

            std::vector<uint8_t> q(static_cast<size_t>(d.bytes));
            std::vector<uint8_t> s(static_cast<size_t>(d.scales_bytes));
            ASSERT_EQ(cudaMemcpy(q.data(), d.ptr, q.size(),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpy(s.data(), d.scales, s.size(),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            std::vector<float> deq(static_cast<size_t>(n * k));
            if (fp8)
                kg::dequantize_rows_fp8_e4m3(
                    q.data(), reinterpret_cast<const float*>(s.data()), n, k,
                    deq.data());
            else
                kg::dequantize_rows_nvfp4(q.data(), s.data(), n, k,
                                          deq.data());
            const auto* src =
                reinterpret_cast<const uint16_t*>(host.data.data());
            const int64_t groups = (k + group - 1) / group;
            for (int64_t r = 0; r < n; ++r)
                for (int64_t g = 0; g < groups; ++g) {
                    const int64_t k0 = g * group;
                    const int64_t k1 = std::min(k, k0 + group);
                    float amax = 0.0f;
                    for (int64_t j = k0; j < k1; ++j) {
                        const uint32_t u =
                            static_cast<uint32_t>(src[r * k + j]) << 16;
                        float f;
                        std::memcpy(&f, &u, 4);
                        amax = std::max(amax, std::fabs(f));
                    }
                    const float band =
                        (fp8 ? amax / 28.0f : amax / 3.0f) + 1e-7f;
                    for (int64_t j = k0; j < k1; ++j) {
                        const uint32_t u =
                            static_cast<uint32_t>(src[r * k + j]) << 16;
                        float f;
                        std::memcpy(&f, &u, 4);
                        ASSERT_LE(
                            std::fabs(deq[static_cast<size_t>(r * k + j)] - f),
                            band)
                            << "row " << r << " col " << j;
                    }
                }
        };
        check_quant(w.lm_head, dev.lm_head);
        check_quant(w.fc, dev.fc);
        for (size_t l = 0; l < w.layers.size(); ++l) {
            check_quant(w.layers[l].q_proj, dev.layers[l].q_proj);
            check_quant(w.layers[l].o_proj, dev.layers[l].o_proj);
            check_quant(w.layers[l].gate_proj, dev.layers[l].gate_proj);
            check_quant(w.layers[l].down_proj, dev.layers[l].down_proj);
        }
        // Norms inside layers stay BF16.
        EXPECT_EQ(dev.layers[0].q_norm.dtype, lm::DsparkWeightDtype::kBF16);
        EXPECT_EQ(dev.layers[0].input_layernorm.dtype,
                  lm::DsparkWeightDtype::kBF16);

        // Arena accounting: exact packed total <= arena <= total + align pad.
        const int64_t packed = lm::dspark_draft_bytes(dir, quant);
        EXPECT_GE(dev.arena_bytes, packed);
        EXPECT_LE(dev.arena_bytes, packed + 128 * 255);
    }
}

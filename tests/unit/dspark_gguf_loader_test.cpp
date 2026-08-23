// DSpark V4 dflash GGUF loader tests (ticket J).
//
// Synthetic mini dflash checkpoint (small dims, real GGUF v3 bytes) through
// the REAL parse/load path: config derivation from metadata + tensor infos,
// strict coverage (INV-DSPARK-CKPT), load-time conversions (Q8_0 -> BF16,
// F32 norms -> BF16, MXFP4 packed verbatim), target-aliased embed/lm_head,
// and the header-only device-bytes budget.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "model/quantization/gguf_kquant.h"
#include "model/weight_loader/dspark_loader.h"

namespace layerstorm::model {
namespace {

namespace fs = std::filesystem;

// ── Minimal GGUF v3 writer with full KV support ────────────────────────────

constexpr int32_t kF32 = 0, kQ8_0 = 8, kBF16 = 30, kMXFP4 = 39;
constexpr int32_t kKvU32 = 4, kKvI32 = 5, kKvF32 = 6, kKvBool = 7,
                  kKvStr = 8, kKvArr = 9;

class MiniGguf {
public:
    void kv_str(const std::string& k, const std::string& v) {
        begin_kv(k, kKvStr);
        put_str(v);
    }
    void kv_u32(const std::string& k, uint32_t v) {
        begin_kv(k, kKvU32);
        put(&v, 4);
    }
    void kv_f32(const std::string& k, float v) {
        begin_kv(k, kKvF32);
        put(&v, 4);
    }
    void kv_bool(const std::string& k, bool v) {
        begin_kv(k, kKvBool);
        int8_t b = v ? 1 : 0;
        put(&b, 1);
    }
    void kv_arr_i32(const std::string& k, const std::vector<int32_t>& v) {
        begin_kv(k, kKvArr);
        int32_t et = kKvI32;
        put(&et, 4);
        uint64_t n = v.size();
        put(&n, 8);
        for (int32_t x : v) put(&x, 4);
    }
    void kv_arr_f32(const std::string& k, const std::vector<float>& v) {
        begin_kv(k, kKvArr);
        int32_t et = kKvF32;
        put(&et, 4);
        uint64_t n = v.size();
        put(&n, 8);
        for (float x : v) put(&x, 4);
    }

    void tensor(const std::string& name, int32_t type,
                std::vector<int64_t> ne) {
        tensors_.push_back({name, type, std::move(ne)});
    }

    static uint64_t type_bytes(int32_t t, int64_t numel) {
        switch (t) {
        case kF32: return static_cast<uint64_t>(numel) * 4;
        case kBF16: return static_cast<uint64_t>(numel) * 2;
        case kQ8_0: return static_cast<uint64_t>(numel / 32) * 34;
        case kMXFP4: return static_cast<uint64_t>(numel / 32) * 17;
        }
        return 0;
    }

    void write(const fs::path& path) {
        std::vector<uint8_t> hdr;
        auto out = [&hdr](const void* p, size_t n) {
            const auto* b = static_cast<const uint8_t*>(p);
            hdr.insert(hdr.end(), b, b + n);
        };
        const char magic[4] = {'G', 'G', 'U', 'F'};
        out(magic, 4);
        uint32_t ver = 3;
        out(&ver, 4);
        int64_t nt = static_cast<int64_t>(tensors_.size());
        int64_t nk = n_kv_;
        out(&nt, 8);
        out(&nk, 8);
        hdr.insert(hdr.end(), kv_.begin(), kv_.end());

        std::vector<uint64_t> offsets, sizes;
        uint64_t cursor = 0;
        for (const auto& t : tensors_) {
            int64_t numel = 1;
            for (auto d : t.ne) numel *= d;
            offsets.push_back(cursor);
            sizes.push_back(type_bytes(t.type, numel));
            cursor = (cursor + sizes.back() + 31) & ~uint64_t{31};
        }
        for (size_t i = 0; i < tensors_.size(); ++i) {
            const auto& t = tensors_[i];
            uint64_t len = t.name.size();
            out(&len, 8);
            out(t.name.data(), t.name.size());
            uint32_t nd = static_cast<uint32_t>(t.ne.size());
            out(&nd, 4);
            for (auto d : t.ne) out(&d, 8);
            out(&t.type, 4);
            out(&offsets[i], 8);
        }
        const uint64_t blob_start = (hdr.size() + 31) & ~uint64_t{31};
        std::vector<uint8_t> file(blob_start, 0);
        std::memcpy(file.data(), hdr.data(), hdr.size());
        file.resize(blob_start + cursor, 0);
        for (size_t i = 0; i < tensors_.size(); ++i) {
            uint8_t* dst = file.data() + blob_start + offsets[i];
            for (uint64_t b = 0; b < sizes[i]; ++b)
                dst[b] = static_cast<uint8_t>((i * 37 + b) & 0xFF);
        }
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(file.data()),
                static_cast<std::streamsize>(file.size()));
    }

private:
    struct T {
        std::string name;
        int32_t type;
        std::vector<int64_t> ne;
    };
    void begin_kv(const std::string& k, int32_t vt) {
        put_str(k);
        put(&vt, 4);
        ++n_kv_;
    }
    void put(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        kv_.insert(kv_.end(), b, b + n);
    }
    void put_str(const std::string& s) {
        uint64_t n = s.size();
        put(&n, 8);
        put(s.data(), s.size());
    }
    std::vector<uint8_t> kv_;
    int64_t n_kv_ = 0;
    std::vector<T> tensors_;
};

// ── Mini dflash checkpoint fixture ─────────────────────────────────────────
// H=32, 2 heads x head_dim 32 (rope 4), q_lora 32, o_groups 2 x o_lora 4,
// E=4 experts top-2, I=32, hc=4, r=4, V=64, 2 blocks, target_layers [1,2].

struct MiniDims {
    int64_t H = 32, V = 64, r = 4, D = 32, HQ = 2, QL = 32, OG = 2, OLR = 4;
    int64_t E = 4, I = 32, hc = 4;
    int layers = 2;
};

void add_dflash_meta(MiniGguf& g, const MiniDims& d) {
    g.kv_str("general.architecture", "dflash");
    g.kv_u32("dflash.block_count", static_cast<uint32_t>(d.layers));
    g.kv_u32("dflash.embedding_length", static_cast<uint32_t>(d.H));
    g.kv_u32("dflash.attention.head_count", static_cast<uint32_t>(d.HQ));
    g.kv_u32("dflash.attention.head_count_kv", 1);
    g.kv_u32("dflash.attention.key_length", static_cast<uint32_t>(d.D));
    g.kv_u32("dflash.attention.value_length", static_cast<uint32_t>(d.D));
    g.kv_u32("dflash.rope.dimension_count", 4);
    g.kv_f32("dflash.rope.freq_base", 10000.0f);
    g.kv_f32("dflash.attention.layer_norm_rms_epsilon", 1e-6f);
    g.kv_u32("dflash.attention.q_lora_rank", static_cast<uint32_t>(d.QL));
    g.kv_u32("dflash.attention.sliding_window", 16);
    g.kv_u32("dflash.expert_count", static_cast<uint32_t>(d.E));
    g.kv_u32("dflash.expert_used_count", 2);
    g.kv_u32("dflash.expert_gating_func", 4);
    g.kv_u32("dflash.expert_feed_forward_length",
             static_cast<uint32_t>(d.I));
    g.kv_u32("dflash.expert_shared_count", 1);
    g.kv_f32("dflash.expert_weights_scale", 1.5f);
    g.kv_bool("dflash.expert_weights_norm", true);
    g.kv_u32("dflash.attention.output_group_count",
             static_cast<uint32_t>(d.OG));
    g.kv_u32("dflash.attention.output_lora_rank",
             static_cast<uint32_t>(d.OLR));
    g.kv_arr_i32("dflash.attention.compress_ratios", {0, 0});
    g.kv_u32("dflash.hyper_connection.count", static_cast<uint32_t>(d.hc));
    g.kv_u32("dflash.hyper_connection.sinkhorn_iterations", 20);
    g.kv_f32("dflash.hyper_connection.epsilon", 1e-6f);
    g.kv_u32("dflash.hash_layer_count", 0);
    g.kv_u32("dflash.block_size", 5);
    g.kv_arr_i32("dflash.target_layers", {1, 2});
    g.kv_arr_f32("dflash.swiglu_clamp_exp", {10.0f, 10.0f});
    g.kv_arr_f32("dflash.swiglu_clamp_shexp", {10.0f, 10.0f});
    g.kv_u32("tokenizer.ggml.mask_token_id", 60);
}

void add_dflash_tensors(MiniGguf& g, const MiniDims& d, bool with_conf = true,
                        const std::string& skip = {}) {
    const int64_t mix = (2 + d.hc) * d.hc;
    auto add = [&](const std::string& name, int32_t t,
                   std::vector<int64_t> ne) {
        if (name == skip) return;
        g.tensor(name, t, std::move(ne));
    };
    add("fc.weight", kQ8_0, {2 * d.H, d.H});  // ne0 = n_aux*H
    add("enc.output_norm.weight", kF32, {d.H});
    add("output_norm.weight", kF32, {d.H});
    add("output_hc_fn.weight", kF32, {d.hc * d.H, d.hc});
    add("output_hc_base.weight", kF32, {d.hc});
    add("output_hc_scale.weight", kF32, {1});
    add("markov_w1.weight", kBF16, {d.r, d.V});
    add("markov_w2.weight", kBF16, {d.r, d.V});
    if (with_conf) add("conf_proj.weight", kBF16, {d.H + d.r, 1});
    for (int l = 0; l < d.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        add(p + "attn_norm.weight", kF32, {d.H});
        add(p + "ffn_norm.weight", kF32, {d.H});
        add(p + "attn_q_a.weight", kQ8_0, {d.H, d.QL});
        add(p + "attn_q_a_norm.weight", kF32, {d.QL});
        add(p + "attn_q_b.weight", kQ8_0, {d.QL, d.HQ * d.D});
        add(p + "attn_kv.weight", kQ8_0, {d.H, d.D});
        add(p + "attn_kv_a_norm.weight", kF32, {d.D});
        add(p + "attn_output_a.weight", kQ8_0,
            {d.HQ * d.D / d.OG, d.OG * d.OLR});
        add(p + "attn_output_b.weight", kQ8_0, {d.OG * d.OLR, d.H});
        add(p + "attn_sinks.weight", kF32, {d.HQ});
        add(p + "hc_attn_fn.weight", kF32, {d.hc * d.H, mix});
        add(p + "hc_attn_base.weight", kF32, {mix});
        add(p + "hc_attn_scale.weight", kF32, {3});
        add(p + "hc_ffn_fn.weight", kF32, {d.hc * d.H, mix});
        add(p + "hc_ffn_base.weight", kF32, {mix});
        add(p + "hc_ffn_scale.weight", kF32, {3});
        add(p + "ffn_gate_inp.weight", kBF16, {d.H, d.E});
        add(p + "exp_probs_b.bias", kF32, {d.E});
        add(p + "ffn_gate_shexp.weight", kQ8_0, {d.H, d.I});
        add(p + "ffn_up_shexp.weight", kQ8_0, {d.H, d.I});
        add(p + "ffn_down_shexp.weight", kQ8_0, {d.I, d.H});
        add(p + "ffn_gate_exps.weight", kMXFP4, {d.H, d.I, d.E});
        add(p + "ffn_up_exps.weight", kMXFP4, {d.H, d.I, d.E});
        add(p + "ffn_down_exps.weight", kMXFP4, {d.I, d.H, d.E});
    }
}

fs::path write_target_gguf(const fs::path& dir, const MiniDims& d) {
    MiniGguf g;
    g.kv_str("general.architecture", "deepseek4");
    g.tensor("token_embd.weight", kBF16, {d.H, d.V});
    g.tensor("output.weight", kBF16, {d.H, d.V});
    const auto p = dir / "target.gguf";
    g.write(p);
    return p;
}

fs::path write_draft_gguf(const fs::path& dir, const MiniDims& d,
                          bool with_conf = true,
                          const std::string& skip = {}) {
    MiniGguf g;
    add_dflash_meta(g, d);
    add_dflash_tensors(g, d, with_conf, skip);
    const auto p = dir / "draft.gguf";
    g.write(p);
    return p;
}

struct TmpDir {
    TmpDir() {
        path = fs::temp_directory_path() /
               ("dspark_gguf_test_" + std::to_string(::getpid()) + "_" +
                std::to_string(counter++));
        fs::create_directories(path);
    }
    ~TmpDir() { std::error_code ec; fs::remove_all(path, ec); }
    fs::path path;
    static inline int counter = 0;
};

// ── Tests ──────────────────────────────────────────────────────────────────

TEST(DsparkGgufLoader, DispatchPredicate) {
    EXPECT_TRUE(dspark_checkpoint_is_gguf("foo/draft.gguf"));
    EXPECT_FALSE(dspark_checkpoint_is_gguf("test-data/GLM.dspark"));
    EXPECT_FALSE(dspark_checkpoint_is_gguf("dir/checkpoint"));
}

TEST(DsparkGgufLoader, ParseDflashConfig) {
    TmpDir tmp;
    MiniDims d;
    const auto p = write_draft_gguf(tmp.path, d);
    const auto c = parse_dspark_checkpoint_config(p);  // dispatches
    EXPECT_TRUE(c.is_v4_dflash);
    EXPECT_EQ(c.block_size, 5);
    EXPECT_EQ(c.speculative_tokens, 4);  // block_size - 1 (llama.cpp rule)
    EXPECT_EQ(c.mask_token_id, 60);
    EXPECT_EQ(c.markov_rank, 4);
    EXPECT_EQ(c.vocab_size, 64);
    EXPECT_EQ(c.draft_vocab_size, 64);
    EXPECT_EQ(c.num_hidden_layers, 2);
    EXPECT_EQ(c.hidden_size, 32);
    EXPECT_EQ(c.num_key_value_heads, 1);
    EXPECT_EQ(c.head_dim, 32);
    EXPECT_EQ(c.aux_hidden_state_layer_ids, (std::vector<int>{1, 2}));
    EXPECT_TRUE(c.enable_confidence_head);
    EXPECT_TRUE(c.confidence_head_with_markov);
    EXPECT_FALSE(c.confidence_has_bias);
    EXPECT_EQ(c.v4.n_routed_experts, 4);
    EXPECT_EQ(c.v4.n_expert_used, 2);
    EXPECT_DOUBLE_EQ(c.v4.swiglu_limit, 10.0);
    EXPECT_EQ(c.v4.hc_mult, 4);
    EXPECT_EQ(c.v4.o_groups, 2);
    EXPECT_EQ(c.v4.rope_dim, 4);
    EXPECT_DOUBLE_EQ(c.v4.routed_scaling, 1.5);
}

TEST(DsparkGgufLoader, LoadConvertsAndAliasesTarget) {
    TmpDir tmp;
    MiniDims d;
    const auto draft = write_draft_gguf(tmp.path, d);
    const auto target = write_target_gguf(tmp.path, d);
    auto w = load_dspark_v4_gguf_draft(draft, target.string());
    ASSERT_TRUE(w.v4 != nullptr);
    ASSERT_EQ(w.v4->layers.size(), 2u);

    // Target-aliased embed/lm_head: BF16 views of the target mmap.
    EXPECT_EQ(w.embed_tokens.shape, (std::vector<int64_t>{d.V, d.H}));
    EXPECT_EQ(w.embed_tokens.dtype, SafetensorsDtype::BF16);
    EXPECT_EQ(w.lm_head.shape, (std::vector<int64_t>{d.V, d.H}));

    // fc: Q8_0 dequanted to BF16 at load, [H, n_aux*H] logical.
    EXPECT_EQ(w.fc.shape, (std::vector<int64_t>{d.H, 2 * d.H}));
    EXPECT_EQ(w.fc.dtype, SafetensorsDtype::BF16);
    EXPECT_EQ(static_cast<int64_t>(w.fc.data.size()), d.H * 2 * d.H * 2);

    // Norms: F32 -> BF16 converted.
    EXPECT_EQ(w.hidden_norm.dtype, SafetensorsDtype::BF16);
    EXPECT_EQ(w.final_norm.dtype, SafetensorsDtype::BF16);
    const auto& L0 = w.v4->layers[0];
    EXPECT_EQ(L0.attn_norm.dtype, SafetensorsDtype::BF16);
    EXPECT_EQ(L0.kv_norm.dtype, SafetensorsDtype::BF16);

    // F32 heads stay F32 (mhc / sinks / gating bias consumers).
    EXPECT_EQ(L0.sinks.dtype, SafetensorsDtype::F32);
    EXPECT_EQ(L0.hc_attn_fn.dtype, SafetensorsDtype::F32);
    EXPECT_EQ(L0.exp_probs_b.dtype, SafetensorsDtype::F32);
    EXPECT_EQ(w.v4->output_hc_fn.shape,
              (std::vector<int64_t>{d.hc, d.hc * d.H}));

    // MXFP4 experts: packed verbatim, expert-major 3D.
    EXPECT_EQ(L0.exps_gate.shape, (std::vector<int64_t>{d.E, d.I, d.H}));
    ASSERT_TRUE(L0.exps_gate.gguf_type.has_value());
    EXPECT_EQ(*L0.exps_gate.gguf_type, GgufKQuantType::MXFP4);
    EXPECT_EQ(static_cast<int64_t>(L0.exps_gate.data.size()),
              d.E * (d.I * d.H / 32) * 17);

    // Confidence: weight present, NO bias.
    EXPECT_NE(w.confidence_proj_weight.data.data(), nullptr);
    EXPECT_EQ(w.confidence_proj_bias.data.data(), nullptr);
    EXPECT_EQ(w.confidence_proj_weight.shape,
              (std::vector<int64_t>{1, d.H + d.r}));
}

TEST(DsparkGgufLoader, MissingTensorFailsClosed) {
    TmpDir tmp;
    MiniDims d;
    const auto draft = write_draft_gguf(tmp.path, d, /*with_conf=*/true,
                                        /*skip=*/"blk.1.attn_kv.weight");
    const auto target = write_target_gguf(tmp.path, d);
    EXPECT_THROW(load_dspark_v4_gguf_draft(draft, target.string()),
                 std::runtime_error);
}

TEST(DsparkGgufLoader, NoConfidenceHeadIsOptional) {
    TmpDir tmp;
    MiniDims d;
    const auto draft = write_draft_gguf(tmp.path, d, /*with_conf=*/false);
    const auto c = parse_dspark_gguf_checkpoint_config(draft);
    EXPECT_FALSE(c.enable_confidence_head);
    const auto target = write_target_gguf(tmp.path, d);
    auto w = load_dspark_v4_gguf_draft(draft, target.string());
    EXPECT_EQ(w.confidence_proj_weight.data.data(), nullptr);
}

TEST(DsparkGgufLoader, DraftBytesCoversConvertedTotals) {
    TmpDir tmp;
    MiniDims d;
    const auto draft = write_draft_gguf(tmp.path, d);
    const auto target = write_target_gguf(tmp.path, d);
    const int64_t budget = dspark_gguf_draft_bytes(draft);
    auto w = load_dspark_v4_gguf_draft(draft, target.string());
    // After load-time conversion host bytes == device bytes per tensor; the
    // budget additionally covers per-tensor 256-alignment + the expert
    // B_ptrs arrays. It must bound the raw converted total and stay within
    // the alignment + ptr-array slack.
    EXPECT_GE(budget, w.total_weight_bytes);
    const int64_t slack =
        256 * (w.total_tensors_loaded + 1) +
        d.layers * 3 * ((d.E * 8 + 255) / 256 * 256 + 256);
    EXPECT_LE(budget, w.total_weight_bytes + slack);
    // Requant / sharding do not apply to the pre-quantized artifact.
    EXPECT_THROW(dspark_draft_bytes(
                     draft, config::DsparkDraftWeightsQuant::fp8_e4m3),
                 std::runtime_error);
    EXPECT_THROW(dspark_draft_bytes(
                     draft, config::DsparkDraftWeightsQuant::bf16, 0, 2),
                 std::runtime_error);
}

TEST(DsparkGgufLoader, LegacyDirLoaderRejectsGgufPath) {
    TmpDir tmp;
    MiniDims d;
    const auto draft = write_draft_gguf(tmp.path, d);
    EXPECT_THROW(load_dspark_draft(draft), std::runtime_error);
}

}  // namespace
}  // namespace layerstorm::model

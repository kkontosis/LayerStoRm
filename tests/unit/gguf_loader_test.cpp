#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "config/config_parser.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/fp8.h"
#include "model/quantization/gguf_kquant.h"
#include "model/weight_loader/gguf_reader.h"
#include "model/weight_loader/tensor_id.h"
#include "model/weight_loader/weight_handler.h"
#include "model/weight_loader/weight_loader.h"
#include "model/weight_pipeline/manifest.h"
#include "model/weight_pipeline/prepacked_format.h"

#include <unistd.h>

#include <nlohmann/json.hpp>

using namespace layerstorm::model;
using namespace layerstorm;

// ── Synthetic GGUF blob builder ─────────────────────────────────────────────
// Writes the GGUF v3 byte format by hand: magic | version | n_tensors | n_kv |
// KV pairs | tensor infos | <pad to alignment> | data blob. Just enough to
// exercise the reader + name parsing + de-stacking + prepack sizing.

namespace {

// ggml_type values (ref ggml.h).
constexpr int32_t kGGML_F32  = 0;
constexpr int32_t kGGML_Q4_K = 12;
constexpr int32_t kGGML_Q6_K = 14;

// GGUF KV scalar types (ref gguf.h).
constexpr int32_t kKV_UINT32 = 4;
constexpr int32_t kKV_INT32  = 5;
constexpr int32_t kKV_ARRAY  = 9;

struct PlannedTensor {
    std::string name;
    int32_t ggml_type;
    std::vector<int64_t> dims;  // GGUF order (dims[0] fastest)
};

class GgufBlobBuilder {
public:
    explicit GgufBlobBuilder(uint64_t alignment = 32) : alignment_(alignment) {}

    void add_tensor(std::string name, int32_t ggml_type, std::vector<int64_t> dims) {
        planned_.push_back({std::move(name), ggml_type, std::move(dims)});
    }

    // A small INT32 metadata ARRAY (GGUF kv type 9, element type 5) — the shape
    // `glm5next.attention.head_count_kv` uses. GgufReader retains numeric
    // arrays up to kSmallArrayCap elements.
    void add_kv_i32_array(std::string key, std::vector<int32_t> values) {
        i32_arrays_.push_back({std::move(key), std::move(values)});
    }

    // Build the full blob; fills each tensor's data region with a per-tensor
    // byte pattern (index-based) so de-stacking correctness can be checked.
    std::vector<std::byte> build() {
        std::vector<uint8_t> hdr;
        auto put = [&](const void* p, size_t n) {
            const auto* b = static_cast<const uint8_t*>(p);
            hdr.insert(hdr.end(), b, b + n);
        };
        auto put_u32 = [&](uint32_t v) { put(&v, 4); };
        auto put_i32 = [&](int32_t v) { put(&v, 4); };
        auto put_i64 = [&](int64_t v) { put(&v, 8); };
        auto put_u64 = [&](uint64_t v) { put(&v, 8); };
        auto put_str = [&](const std::string& s) {
            put_u64(s.size());
            put(s.data(), s.size());
        };

        // Magic + version + counts.
        const char magic[4] = {'G', 'G', 'U', 'F'};
        put(magic, 4);
        put_u32(3);
        put_i64(static_cast<int64_t>(planned_.size()));
        put_i64(1 + static_cast<int64_t>(i32_arrays_.size()));  // general.alignment + arrays

        // KV: general.alignment (uint32).
        put_str("general.alignment");
        put_i32(kKV_UINT32);
        put_u32(static_cast<uint32_t>(alignment_));

        // KV: numeric INT32 arrays.
        for (const auto& [key, vals] : i32_arrays_) {
            put_str(key);
            put_i32(kKV_ARRAY);
            put_i32(kKV_INT32);
            put_u64(vals.size());
            for (int32_t v : vals) put_i32(v);
        }

        // Tensor infos + compute per-tensor sizes/offsets within the blob.
        std::vector<uint64_t> sizes;
        std::vector<uint64_t> offsets;
        uint64_t cursor = 0;
        for (const auto& t : planned_) {
            int64_t numel = 1;
            for (auto d : t.dims) numel *= d;
            uint64_t bytes = tensor_bytes(t.ggml_type, numel);
            offsets.push_back(cursor);
            sizes.push_back(bytes);
            cursor = align_up(cursor + bytes, alignment_);
        }

        for (size_t i = 0; i < planned_.size(); ++i) {
            const auto& t = planned_[i];
            put_str(t.name);
            put_u32(static_cast<uint32_t>(t.dims.size()));
            for (auto d : t.dims) put_i64(d);
            put_i32(t.ggml_type);
            put_u64(offsets[i]);
        }

        // Pad header to alignment, then append the data blob.
        uint64_t blob_start = align_up(hdr.size(), alignment_);
        std::vector<std::byte> out(blob_start, std::byte{0});
        std::memcpy(out.data(), hdr.data(), hdr.size());

        uint64_t blob_bytes = cursor;
        out.resize(blob_start + blob_bytes, std::byte{0});
        for (size_t i = 0; i < planned_.size(); ++i) {
            uint8_t* dst = reinterpret_cast<uint8_t*>(out.data()) + blob_start + offsets[i];
            for (uint64_t b = 0; b < sizes[i]; ++b) {
                dst[b] = static_cast<uint8_t>((i * 31 + b) & 0xFF);
            }
            // GF3.9: ssm_a must model the REAL checkpoint convention
            // A = -exp(A_log) < 0 (measured on the unsloth GGUF; the
            // GF3.6 goldens' safe gate exponentiates A_log, so exp > 0
            // always). The loader's convention guard REFUSES a
            // non-negative value — index-pattern garbage bytes here would
            // be a fixture that never matched any artifact.
            if (planned_[i].name.size() >= 5
                && planned_[i].name.compare(planned_[i].name.size() - 5, 5,
                                            "ssm_a") == 0) {
                auto* f = reinterpret_cast<float*>(dst);
                for (uint64_t e = 0; e < sizes[i] / 4; ++e)
                    f[e] = -1.5f - 0.01f * static_cast<float>(e % 64);
            }
        }
        return out;
    }

    static uint64_t tensor_bytes(int32_t ggml_type, int64_t numel) {
        switch (ggml_type) {
            case kGGML_F32:  return static_cast<uint64_t>(numel) * 4;
            case kGGML_Q4_K: return static_cast<uint64_t>(numel / 256) * 144;
            case kGGML_Q6_K: return static_cast<uint64_t>(numel / 256) * 210;
            case 8:  return static_cast<uint64_t>(numel / 32) * 34;  // Q8_0
            case 26: return static_cast<uint64_t>(numel) * 4;        // I32
            case 30: return static_cast<uint64_t>(numel) * 2;        // BF16
            case 39: return static_cast<uint64_t>(numel / 32) * 17;  // MXFP4
        }
        return 0;
    }

private:
    static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }
    uint64_t alignment_;
    std::vector<PlannedTensor> planned_;
    std::vector<std::pair<std::string, std::vector<int32_t>>> i32_arrays_;
};

}  // namespace

// ── GgufReader: parse tensor infos + per-tensor types ───────────────────────

TEST(GgufLoader, ReaderParsesTensorInfosAndTypes) {
    GgufBlobBuilder b;
    // A float norm, a Q4_K projection, and a stacked Q4_K expert tensor.
    b.add_tensor("output_norm.weight", kGGML_F32, {256});
    b.add_tensor("blk.0.attn_q_a.weight", kGGML_Q4_K, {512, 256});  // {in, out}
    b.add_tensor("blk.3.ffn_gate_exps.weight", kGGML_Q4_K, {512, 256, 4});  // {in,out,n_exp}
    auto blob = b.build();

    GgufReader r = GgufReader::from_buffer(blob, "synthetic");
    ASSERT_EQ(r.entries().size(), 3u);
    EXPECT_EQ(r.alignment(), 32u);

    const auto& norm = r.entries()[0];
    EXPECT_EQ(norm.name, "output_norm.weight");
    EXPECT_FALSE(norm.is_kquant());
    EXPECT_EQ(norm.data_size_bytes, 256u * 4);

    const auto& qa = r.entries()[1];
    EXPECT_TRUE(qa.is_kquant());
    EXPECT_EQ(qa.kquant_type(), GgufKQuantType::Q4_K);
    // 512*256 / 256 * 144 = 512 * 144.
    EXPECT_EQ(qa.data_size_bytes, static_cast<uint64_t>(512) * 144);

    const auto& exps = r.entries()[2];
    EXPECT_TRUE(exps.is_kquant());
    EXPECT_EQ(exps.dims.size(), 3u);
    EXPECT_EQ(exps.dims[2], 4);  // 4 experts
    // 512*256*4 / 256 * 144 = 512*4*144.
    EXPECT_EQ(exps.data_size_bytes, static_cast<uint64_t>(512) * 4 * 144);

    // Data span has the right length and byte pattern (tensor index 1).
    auto span = r.tensor_data(qa);
    EXPECT_EQ(span.size(), qa.data_size_bytes);
    EXPECT_EQ(static_cast<uint8_t>(span[0]), static_cast<uint8_t>(1 * 31 + 0));
}

TEST(GgufLoader, ReaderRejectsBadMagic) {
    std::vector<std::byte> blob(64, std::byte{0});
    blob[0] = std::byte{'X'};
    EXPECT_THROW(GgufReader::from_buffer(blob, "bad"), std::runtime_error);
}

TEST(GgufLoader, ReaderRejectsUnsupportedKquant) {
    // ggml IQ2_XXS = 16 is not supported; non-multiple-of-QK is caught earlier,
    // so use a recognized float type vs. an unsupported quant via the helper.
    EXPECT_THROW(gguf_kquant_from_ggml(16), std::runtime_error);
    EXPECT_FALSE(is_supported_gguf_kquant(16));
    EXPECT_TRUE(is_supported_gguf_kquant(kGGML_Q6_K));
}

// ── Split-set resolution: -NNNNN-of-MMMMM.gguf expansion ────────────────────

TEST(GgufLoader, ResolveSplitMembersExpandsFullSet) {
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() /
        ("lstorm_split_" + std::to_string(::testing::UnitTest::GetInstance()
                                              ->random_seed()));
    fs::create_directories(dir);
    const std::string stem = "MODEL-UD-Q4_K_XL";
    auto member = [&](int i) {
        char buf[6];
        std::snprintf(buf, sizeof(buf), "%05d", i);
        return dir / (stem + "-" + buf + "-of-00003.gguf");
    };
    for (int i = 1; i <= 3; ++i) { std::ofstream(member(i)) << "x"; }

    // Passing ANY member expands to the full ordered set.
    auto files = resolve_gguf_files(member(2).string());
    ASSERT_EQ(files.size(), 3u);
    EXPECT_EQ(files[0].filename().string(), stem + "-00001-of-00003.gguf");
    EXPECT_EQ(files[1].filename().string(), stem + "-00002-of-00003.gguf");
    EXPECT_EQ(files[2].filename().string(), stem + "-00003-of-00003.gguf");

    // A plain single .gguf (no split suffix) resolves to just itself.
    fs::path single = dir / "plain.gguf";
    std::ofstream(single) << "x";
    auto one = resolve_gguf_files(single.string());
    ASSERT_EQ(one.size(), 1u);
    EXPECT_EQ(one[0].filename().string(), "plain.gguf");

    // A missing member throws (incomplete download).
    fs::remove(member(3));
    EXPECT_THROW(resolve_gguf_files(member(1).string()), std::runtime_error);

    fs::remove_all(dir);
}

// ── TD-VOCAB-AUTODETECT: GGUF vocab-row detection (dims are reversed) ───────

TEST(GgufLoader, VocabAutodetectFromTokenEmbd) {
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() /
        ("lstorm_vocab_" + std::to_string(::testing::UnitTest::GetInstance()
                                              ->random_seed()));
    fs::create_directories(dir);

    GgufBlobBuilder b;
    // GGUF order: dims[0] fastest (hidden columns) — vocab is the LAST dim.
    b.add_tensor("token_embd.weight", kGGML_F32, {8, 1024});
    b.add_tensor("output.weight", kGGML_F32, {8, 1024});
    auto blob = b.build();
    fs::path f = dir / "model.gguf";
    {
        std::ofstream ofs(f, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(blob.data()),
                  static_cast<std::streamsize>(blob.size()));
    }

    layerstorm::config::Config cfg;
    cfg.model.weights_path = f.string();
    cfg.model.weights_format = layerstorm::config::WeightsFormat::gguf;
    cfg.model.vocab_size = 0;

    EXPECT_EQ(detect_weights_vocab_rows(cfg), 1024);
    resolve_vocab_size(cfg);
    EXPECT_EQ(cfg.model.vocab_size, 1024);   // adopted

    cfg.model.vocab_size = 129280;           // mismatched paste → fail loud
    EXPECT_THROW(resolve_vocab_size(cfg), std::runtime_error);

    fs::remove_all(dir);
}

// ── GLM-1: split MLA up-projection (attn_k_b / attn_v_b) ────────────────────

namespace {

// fp16 (binary16) bit pattern of a float (round-toward-zero mantissa; the test
// values below are all exactly representable so rounding is irrelevant).
uint16_t f32_to_fp16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FFu;
    if (((x >> 23) & 0xFF) == 0) { exp = 0; mant = 0; }  // zero/subnormal → 0
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp & 0x1F) << 10) | mant);
}

// Reference bf16 (round-to-nearest-even) — mirrors the engine's f32_to_bf16.
uint16_t f32_to_bf16_ref(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

// Encode one Q8_0 block (32 values) into 34 bytes: fp16 scale d, then 32 int8.
// Picks d so the values quantize exactly (values chosen as d * small ints).
void encode_q8_0_block(std::vector<std::byte>& out, const float* vals, float d) {
    uint16_t dh = f32_to_fp16_bits(d);
    auto* p = reinterpret_cast<uint8_t*>(&dh);
    out.push_back(std::byte{p[0]});
    out.push_back(std::byte{p[1]});
    for (int i = 0; i < 32; ++i) {
        int q = static_cast<int>(std::lround(vals[i] / d));
        q = std::max(-127, std::min(127, q));
        out.push_back(static_cast<std::byte>(static_cast<int8_t>(q)));
    }
}

// Build a Q8_0 RawTensor from a flat f32 array (ggml row-major, fastest dim a
// multiple of 32). Returns {bytes, RawTensor-view}; bytes must outlive the view.
RawTensor make_q8_0_raw(std::vector<std::byte>& bytes, const std::vector<float>& flat,
                        std::vector<int64_t> shape, float d) {
    bytes.clear();
    for (size_t b = 0; b < flat.size(); b += 32) {
        encode_q8_0_block(bytes, &flat[b], d);
    }
    RawTensor t;
    t.data = std::span<const std::byte>(bytes.data(), bytes.size());
    t.dtype = SafetensorsDtype::U8;
    t.shape = std::move(shape);
    t.gguf_type = GgufKQuantType::Q8_0;
    return t;
}

}  // namespace

TEST(GgufLoader, ParseNameMapsSplitMlaUpProj) {
    auto kb = parse_gguf_name("blk.11.attn_k_b.weight");
    ASSERT_TRUE(kb.has_value());
    EXPECT_EQ(kb->component, TensorComponent::mla_k_b_split);
    EXPECT_EQ(kb->owner, TensorOwner::attention);
    EXPECT_EQ(kb->layer_idx, 11);
    EXPECT_EQ(kb->role, TensorRole::weight);

    auto vb = parse_gguf_name("blk.11.attn_v_b.weight");
    ASSERT_TRUE(vb.has_value());
    EXPECT_EQ(vb->component, TensorComponent::mla_v_b_split);
    EXPECT_EQ(vb->layer_idx, 11);
}

// Assemble combined kv_b_proj from split Q8_0 tensors and compare against an
// independent reference (explicit per-head dequant + transpose of k_b + stack).
TEST(GgufLoader, AssembleSplitKvBMatchesReference) {
    // Non-trivial, GLM-like head count with the real per-head dims (QK=32 must
    // divide the packed/fastest dim: P=192, L=512 both ÷32). Use H=3 heads.
    const int64_t H = 3, P = 192, V = 256, L = 512;

    // attn_k_b : [H, L, P]  (W_UK transposed). attn_v_b : [H, V, L] (W_UV).
    // Build distinct, exactly-Q8_0-representable values: d * integer.
    const float dk = 0.5f, dv = 0.25f;
    std::vector<float> kflat(static_cast<size_t>(H * L * P));
    std::vector<float> vflat(static_cast<size_t>(H * V * L));
    auto kval = [&](int64_t h, int64_t l, int64_t p) {
        return dk * static_cast<float>(((h * 7 + l) % 11 + p % 5) - 5);
    };
    auto vval = [&](int64_t h, int64_t vr, int64_t l) {
        return dv * static_cast<float>(((h * 3 + vr) % 9 + l % 4) - 4);
    };
    for (int64_t h = 0; h < H; ++h)
        for (int64_t l = 0; l < L; ++l)
            for (int64_t p = 0; p < P; ++p)
                kflat[static_cast<size_t>((h * L + l) * P + p)] = kval(h, l, p);
    for (int64_t h = 0; h < H; ++h)
        for (int64_t vr = 0; vr < V; ++vr)
            for (int64_t l = 0; l < L; ++l)
                vflat[static_cast<size_t>((h * V + vr) * L + l)] = vval(h, vr, l);

    std::vector<std::byte> kbytes, vbytes;
    RawTensor kb = make_q8_0_raw(kbytes, kflat, {H, L, P}, dk);
    RawTensor vb = make_q8_0_raw(vbytes, vflat, {H, V, L}, dv);

    AssembledKvB out = assemble_split_kv_b_proj(kb, vb);

    const int64_t rows = H * (P + V);
    ASSERT_EQ(out.shape.size(), 2u);
    EXPECT_EQ(out.shape[0], rows);
    EXPECT_EQ(out.shape[1], L);
    ASSERT_EQ(out.buf->size(), static_cast<size_t>(rows * L) * sizeof(uint16_t));
    const auto* got = reinterpret_cast<const uint16_t*>(out.buf->data());

    // Reference: per head, rows [0,P) = W_UK = transpose(k_b[h]) i.e.
    // out[h,p,l] = kval(h,l,p); rows [P,P+V) = W_UV = vval(h,vr,l). Compare bf16.
    for (int64_t h = 0; h < H; ++h) {
        const int64_t base = h * (P + V);
        for (int64_t p = 0; p < P; ++p)
            for (int64_t l = 0; l < L; ++l) {
                uint16_t ref = f32_to_bf16_ref(kval(h, l, p));
                ASSERT_EQ(got[(base + p) * L + l], ref)
                    << "W_UK h=" << h << " p=" << p << " l=" << l;
            }
        for (int64_t vr = 0; vr < V; ++vr)
            for (int64_t l = 0; l < L; ++l) {
                uint16_t ref = f32_to_bf16_ref(vval(h, vr, l));
                ASSERT_EQ(got[(base + P + vr) * L + l], ref)
                    << "W_UV h=" << h << " vr=" << vr << " l=" << l;
            }
    }
}

// Shape-mismatch and unsupported-type paths throw rather than corrupt.
TEST(GgufLoader, AssembleSplitKvBRejectsBadInput) {
    std::vector<std::byte> kbytes, vbytes;
    std::vector<float> kflat(static_cast<size_t>(2 * 64 * 32), 0.5f);  // [H=2,L=64,P=32]
    std::vector<float> vflat(static_cast<size_t>(3 * 32 * 64), 0.25f); // [H=3,V=32,L=64]
    RawTensor kb = make_q8_0_raw(kbytes, kflat, {2, 64, 32}, 0.5f);
    RawTensor vb = make_q8_0_raw(vbytes, vflat, {3, 32, 64}, 0.25f);
    EXPECT_THROW(assemble_split_kv_b_proj(kb, vb), std::runtime_error);  // H mismatch

    // Unsupported gguf type for host dequant → throws with guidance. (Q4_K/Q5_K/
    // Q6_K are now supported by the GG-9 host dequant; Q2_K/Q3_K still throw.)
    RawTensor bad = kb;
    bad.gguf_type = GgufKQuantType::Q2_K;
    RawTensor vb_ok = make_q8_0_raw(vbytes, std::vector<float>(2 * 32 * 64, 0.25f),
                                    {2, 32, 64}, 0.25f);
    EXPECT_THROW(assemble_split_kv_b_proj(bad, vb_ok), std::runtime_error);
}

// ── parse_gguf_name ─────────────────────────────────────────────────────────

TEST(GgufLoader, ParseNameMapsToCanonicalIds) {
    struct Case {
        std::string name;
        TensorComponent comp;
        TensorOwner owner;
        int layer;
        int expert;
        TensorRole role;
    };
    const Case cases[] = {
        {"token_embd.weight", TensorComponent::embedding, TensorOwner::model_level, -1, -1, TensorRole::weight},
        {"output.weight", TensorComponent::output_head, TensorOwner::model_level, -1, -1, TensorRole::weight},
        {"output_norm.weight", TensorComponent::final_norm, TensorOwner::model_level, -1, -1, TensorRole::weight},
        {"blk.3.attn_q_a.weight", TensorComponent::q_a_proj, TensorOwner::attention, 3, -1, TensorRole::weight},
        {"blk.3.attn_q_b.weight", TensorComponent::q_b_proj, TensorOwner::attention, 3, -1, TensorRole::weight},
        {"blk.3.attn_kv_a_mqa.weight", TensorComponent::kv_a_proj_with_mqa, TensorOwner::attention, 3, -1, TensorRole::weight},
        {"blk.3.attn_kv_b.weight", TensorComponent::kv_b_proj, TensorOwner::attention, 3, -1, TensorRole::weight},
        {"blk.3.attn_output.weight", TensorComponent::o_proj, TensorOwner::attention, 3, -1, TensorRole::weight},
        {"blk.3.attn_q_a_norm.weight", TensorComponent::q_a_norm, TensorOwner::attention, 3, -1, TensorRole::weight},
        {"blk.3.attn_kv_a_norm.weight", TensorComponent::kv_a_norm, TensorOwner::attention, 3, -1, TensorRole::weight},
        {"blk.3.attn_norm.weight", TensorComponent::input_layernorm, TensorOwner::attention, 3, -1, TensorRole::weight},
        {"blk.3.ffn_norm.weight", TensorComponent::post_attention_layernorm, TensorOwner::attention, 3, -1, TensorRole::weight},
        {"blk.3.ffn_gate_inp.weight", TensorComponent::gate_weight, TensorOwner::gating, 3, -1, TensorRole::weight},
        {"blk.3.exp_probs_b.bias", TensorComponent::gate_e_score_correction_bias, TensorOwner::gating, 3, -1, TensorRole::bias},
        {"blk.5.ffn_gate_exps.weight", TensorComponent::gate_proj, TensorOwner::routed_expert, 5, -1, TensorRole::weight},
        {"blk.5.ffn_up_exps.weight", TensorComponent::up_proj, TensorOwner::routed_expert, 5, -1, TensorRole::weight},
        {"blk.5.ffn_down_exps.weight", TensorComponent::down_proj, TensorOwner::routed_expert, 5, -1, TensorRole::weight},
        {"blk.5.ffn_gate_shexp.weight", TensorComponent::gate_proj, TensorOwner::shared_expert, 5, -1, TensorRole::weight},
        {"blk.5.ffn_down_shexp.weight", TensorComponent::down_proj, TensorOwner::shared_expert, 5, -1, TensorRole::weight},
        {"blk.2.ffn_gate.weight", TensorComponent::gate_proj, TensorOwner::dense_ffn, 2, -1, TensorRole::weight},
        {"blk.2.ffn_down.weight", TensorComponent::down_proj, TensorOwner::dense_ffn, 2, -1, TensorRole::weight},
        {"blk.7.indexer.attn_q_b.weight", TensorComponent::indexer_wq_b, TensorOwner::attention, 7, -1, TensorRole::weight},
        {"blk.7.indexer.attn_k.weight", TensorComponent::indexer_wk, TensorOwner::attention, 7, -1, TensorRole::weight},
        {"blk.7.indexer.proj.weight", TensorComponent::indexer_weights_proj, TensorOwner::attention, 7, -1, TensorRole::weight},
        {"blk.7.indexer.k_norm.weight", TensorComponent::indexer_k_norm_weight, TensorOwner::attention, 7, -1, TensorRole::weight},
        {"blk.7.indexer.k_norm.bias", TensorComponent::indexer_k_norm_bias, TensorOwner::attention, 7, -1, TensorRole::bias},
        // MTP / NextN block (llama.cpp blk.N.nextn.* naming, GLM-5.2 blk.78)
        {"blk.78.nextn.eh_proj.weight", TensorComponent::mtp_eh_proj, TensorOwner::mtp, 78, -1, TensorRole::weight},
        {"blk.78.nextn.embed_tokens.weight", TensorComponent::mtp_embed_tokens, TensorOwner::mtp, 78, -1, TensorRole::weight},
        {"blk.78.nextn.enorm.weight", TensorComponent::mtp_enorm, TensorOwner::mtp, 78, -1, TensorRole::weight},
        {"blk.78.nextn.hnorm.weight", TensorComponent::mtp_hnorm, TensorOwner::mtp, 78, -1, TensorRole::weight},
        {"blk.78.nextn.shared_head_head.weight", TensorComponent::mtp_shared_head_weight, TensorOwner::mtp, 78, -1, TensorRole::weight},
        {"blk.78.nextn.shared_head_norm.weight", TensorComponent::mtp_shared_head_norm, TensorOwner::mtp, 78, -1, TensorRole::weight},
    };
    for (const auto& c : cases) {
        auto id = parse_gguf_name(c.name);
        ASSERT_TRUE(id.has_value()) << c.name;
        EXPECT_EQ(id->component, c.comp) << c.name;
        EXPECT_EQ(id->owner, c.owner) << c.name;
        EXPECT_EQ(id->layer_idx, c.layer) << c.name;
        EXPECT_EQ(id->expert_idx, c.expert) << c.name;
        EXPECT_EQ(id->role, c.role) << c.name;
    }
}

// GLM-25e: the COMPLETE tensor-name surface of the real GLM-5.2 Q4_K_XL GGUF
// (1809 tensors, 33 unique patterns enumerated from the 11 shard headers on
// 2026-07-05). Every pattern must map — an unmapped name would silently drop
// a weight at load. Notable: the MTP block (blk.78) carries FULL attention +
// indexer + MoE tensors (79/76 counts), and the converter dedups
// nextn.embed_tokens/shared_head_head into token_embd/output (absent here,
// still mapped above for other exports).
TEST(GgufLoader, ParseNameCoversGlm52GgufSurface) {
    const char* patterns[] = {
        "blk.5.attn_k_b.weight",       "blk.5.attn_kv_a_mqa.weight",
        "blk.5.attn_kv_a_norm.weight", "blk.5.attn_norm.weight",
        "blk.5.attn_output.weight",    "blk.5.attn_q_a.weight",
        "blk.5.attn_q_a_norm.weight",  "blk.5.attn_q_b.weight",
        "blk.5.attn_v_b.weight",       "blk.5.exp_probs_b.bias",
        "blk.2.ffn_down.weight",       "blk.5.ffn_down_exps.weight",
        "blk.5.ffn_down_shexp.weight", "blk.2.ffn_gate.weight",
        "blk.5.ffn_gate_exps.weight",  "blk.5.ffn_gate_inp.weight",
        "blk.5.ffn_gate_shexp.weight", "blk.5.ffn_norm.weight",
        "blk.2.ffn_up.weight",         "blk.5.ffn_up_exps.weight",
        "blk.5.ffn_up_shexp.weight",   "blk.5.indexer.attn_k.weight",
        "blk.5.indexer.attn_q_b.weight", "blk.5.indexer.k_norm.bias",
        "blk.5.indexer.k_norm.weight", "blk.5.indexer.proj.weight",
        "blk.78.nextn.eh_proj.weight", "blk.78.nextn.enorm.weight",
        "blk.78.nextn.hnorm.weight",   "blk.78.nextn.shared_head_norm.weight",
        "output.weight", "output_norm.weight", "token_embd.weight",
    };
    for (const char* p : patterns)
        EXPECT_TRUE(parse_gguf_name(p).has_value()) << "unmapped: " << p;
}

TEST(GgufLoader, ParseNameRejectsUnknown) {
    EXPECT_FALSE(parse_gguf_name("not.a.tensor").has_value());
    EXPECT_FALSE(parse_gguf_name("blk.x.attn_q_a.weight").has_value());
    EXPECT_FALSE(parse_gguf_name("blk.0.mystery.weight").has_value());
}

// ── De-stacking yields correct per-expert spans ──────────────────────────────

TEST(GgufLoader, DestackingExpertSpans) {
    // hidden=256, intermediate=512, 4 experts. gate: {in=hidden, out=inter}.
    const int hidden = 256, inter = 512, n_exp = 4;
    GgufBlobBuilder b;
    b.add_tensor("blk.0.ffn_gate_exps.weight", kGGML_Q4_K, {hidden, inter, n_exp});
    auto blob = b.build();

    GgufReader r = GgufReader::from_buffer(blob, "exps");
    const auto& e = r.entries()[0];
    auto data = r.tensor_data(e);

    int64_t per_expert =
        gguf::gguf_packed_bytes(inter, hidden, GgufKQuantType::Q4_K);
    EXPECT_EQ(per_expert * n_exp, static_cast<int64_t>(data.size()));

    // Each expert's span is the e-th contiguous slice; verify the byte pattern
    // at the start of expert 2's slice matches the global pattern.
    int64_t off2 = 2 * per_expert;
    EXPECT_EQ(static_cast<uint8_t>(data[off2]),
              static_cast<uint8_t>((0 * 31 + off2) & 0xFF));
}

// ── Prepack slot bytes match make_gguf_quant / gguf_packed_bytes ─────────────

TEST(GgufLoader, ManifestRoundTripAndVerify) {
    // Build a GGUF manifest for a mixed expert (gate/up=Q4_K, down=Q6_K) and
    // confirm it round-trips through JSON and verifies against the rebuilt quant.
    const ExpertShape shape{256, 512};  // hidden, intermediate (QK=256 divides both)
    GgufQuantInterface quant =
        make_gguf_quant(GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q6_K);

    Manifest m;
    m.format_version = std::string{prepacked::kFormatVersion};
    m.engine_version = std::string{prepacked::kFormatVersion};
    m.source_model_path = "/tmp/model.gguf";
    m.quant_format = std::string{prepacked::kGgufPackedQuantFormat};
    m.n_routed_experts = 4;
    m.n_expert_files = 4;
    m.expert_dimensions = ManifestExpertDimensions{256, 512};
    m.moe_layers.count = 2;
    m.moe_layers.first_moe_layer = 0;
    m.moe_layers.last_moe_layer = 1;
    m.moe_layers.indices = {0, 1};
    m.gguf_types = GgufExpertTypes{GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q6_K};
    // GG-10: per-layer triples (mandatory for gguf, one per MoE layer). Both
    // layers uniform here — mixed-per-layer coverage is in
    // ManifestPerLayerTypesRoundTrip below.
    m.gguf_types_per_layer = {
        GgufExpertTypes{GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q6_K},
        GgufExpertTypes{GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q6_K},
    };
    m.slot = build_slot_from_quant(quant, shape);

    // Slot byte accounting matches gguf_packed_bytes per projection.
    int64_t gate = gguf::gguf_packed_bytes(512, 256, GgufKQuantType::Q4_K);
    int64_t up   = gguf::gguf_packed_bytes(512, 256, GgufKQuantType::Q4_K);
    int64_t down = gguf::gguf_packed_bytes(256, 512, GgufKQuantType::Q6_K);
    EXPECT_EQ(m.slot.slot_size_bytes, gate + up + down);
    EXPECT_EQ(m.slot.projections[0].total_bytes, gate);
    EXPECT_EQ(m.slot.projections[2].total_bytes, down);
    EXPECT_EQ(m.slot.projections[0].scale_bytes, 0);  // k-quant scales in-block

    // JSON round-trip preserves the gguf_types block.
    auto j = manifest_to_json(m);
    Manifest m2 = manifest_from_json(j);
    ASSERT_TRUE(m2.gguf_types.has_value());
    EXPECT_EQ(m2.gguf_types->gate, GgufKQuantType::Q4_K);
    EXPECT_EQ(m2.gguf_types->down, GgufKQuantType::Q6_K);
    EXPECT_EQ(m2.quant_format, "gguf");

    // verify_manifest passes against the rebuilt interface.
    GgufQuantInterface rebuilt = m2.gguf_types->to_quant();
    auto vr = verify_manifest(m2, rebuilt);
    EXPECT_TRUE(vr.ok) << vr.error;

    // A different down type fails verification (catches mix mismatch).
    GgufQuantInterface wrong =
        make_gguf_quant(GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q4_K);
    auto vr_wrong = verify_manifest(m2, wrong);
    EXPECT_FALSE(vr_wrong.ok);
}

// ── GG-10: per-layer mixed k-quant triples in the manifest ───────────────────

namespace {

// Base gguf manifest with 3 MoE layers of MIXED per-layer triples; the global
// gguf_types block is the per-projection MAX (Q5_K/Q5_K/Q6_K) that sizes the slot.
Manifest make_per_layer_gguf_manifest() {
    const ExpertShape shape{256, 512};
    GgufQuantInterface quant = make_gguf_quant(
        GgufKQuantType::Q5_K, GgufKQuantType::Q5_K, GgufKQuantType::Q6_K);

    Manifest m;
    m.format_version = std::string{prepacked::kFormatVersion};
    m.engine_version = std::string{prepacked::kFormatVersion};
    m.source_model_path = "/tmp/model.gguf";
    m.quant_format = std::string{prepacked::kGgufPackedQuantFormat};
    m.n_routed_experts = 4;
    m.n_expert_files = 4;
    m.expert_dimensions = ManifestExpertDimensions{256, 512};
    m.moe_layers.count = 3;
    m.moe_layers.first_moe_layer = 1;
    m.moe_layers.last_moe_layer = 3;
    m.moe_layers.indices = {1, 2, 3};
    m.gguf_types = GgufExpertTypes{
        GgufKQuantType::Q5_K, GgufKQuantType::Q5_K, GgufKQuantType::Q6_K};
    m.gguf_types_per_layer = {
        GgufExpertTypes{GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q5_K},
        GgufExpertTypes{GgufKQuantType::Q5_K, GgufKQuantType::Q5_K, GgufKQuantType::Q6_K},
        GgufExpertTypes{GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q6_K},
    };
    m.slot = build_slot_from_quant(quant, shape);
    return m;
}

}  // namespace

TEST(GgufLoader, ManifestPerLayerTypesRoundTrip) {
    Manifest m = make_per_layer_gguf_manifest();
    GgufQuantInterface quant = m.gguf_types->to_quant();

    // JSON round-trip preserves order and every mixed entry.
    auto j = manifest_to_json(m);
    Manifest m2 = manifest_from_json(j);
    ASSERT_EQ(m2.gguf_types_per_layer.size(), 3u);
    EXPECT_EQ(m2.gguf_types_per_layer[0].gate, GgufKQuantType::Q4_K);
    EXPECT_EQ(m2.gguf_types_per_layer[0].up,   GgufKQuantType::Q4_K);
    EXPECT_EQ(m2.gguf_types_per_layer[0].down, GgufKQuantType::Q5_K);
    EXPECT_EQ(m2.gguf_types_per_layer[1].gate, GgufKQuantType::Q5_K);
    EXPECT_EQ(m2.gguf_types_per_layer[1].down, GgufKQuantType::Q6_K);
    EXPECT_EQ(m2.gguf_types_per_layer[2].gate, GgufKQuantType::Q4_K);
    EXPECT_EQ(m2.gguf_types_per_layer[2].down, GgufKQuantType::Q6_K);

    auto vr = verify_manifest(m2, quant);
    EXPECT_TRUE(vr.ok) << vr.error;
}

TEST(GgufLoader, ManifestPerLayerTypesRejections) {
    GgufQuantInterface quant = make_gguf_quant(
        GgufKQuantType::Q5_K, GgufKQuantType::Q5_K, GgufKQuantType::Q6_K);

    // Count mismatch: entries != moe_layers.count.
    {
        Manifest m = make_per_layer_gguf_manifest();
        m.gguf_types_per_layer.pop_back();
        auto vr = verify_manifest(m, quant);
        EXPECT_FALSE(vr.ok);
        EXPECT_NE(vr.error.find("gguf_types_per_layer"), std::string::npos)
            << vr.error;
    }

    // Missing block entirely on a current-version gguf manifest.
    {
        Manifest m = make_per_layer_gguf_manifest();
        m.gguf_types_per_layer.clear();
        auto vr = verify_manifest(m, quant);
        EXPECT_FALSE(vr.ok);
        EXPECT_NE(vr.error.find("missing gguf_types_per_layer"),
                  std::string::npos) << vr.error;
    }

    // Bad k-quant name is rejected at parse time (type_from_name throws).
    {
        Manifest m = make_per_layer_gguf_manifest();
        auto j = manifest_to_json(m);
        j["gguf_types_per_layer"][0]["gate"] = "q9_z";
        EXPECT_THROW(manifest_from_json(j), std::runtime_error);
    }

    // A layer whose packed total exceeds the (global-MAX-sized) slot.
    {
        Manifest m = make_per_layer_gguf_manifest();
        m.gguf_types_per_layer[0] = GgufExpertTypes{
            GgufKQuantType::Q8_0, GgufKQuantType::Q8_0, GgufKQuantType::Q8_0};
        auto vr = verify_manifest(m, quant);
        EXPECT_FALSE(vr.ok);
        EXPECT_NE(vr.error.find("exceeds slot size"), std::string::npos)
            << vr.error;
    }
}

// ── pack_gguf_expert: slot bytes match make_gguf_quant ───────────────────────

TEST(GgufLoader, PackExpertSlotBytesMatchQuant) {
    const int hidden = 256, inter = 512;  // QK=256 divides both
    const ExpertShape shape{hidden, inter};
    // gate/up Q4_K, down Q6_K.
    GgufQuantInterface quant =
        make_gguf_quant(GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q6_K);

    int64_t gate_b = quant.bytes_per_projection(shape, Projection::gate);
    int64_t up_b   = quant.bytes_per_projection(shape, Projection::up);
    int64_t down_b = quant.bytes_per_projection(shape, Projection::down);

    // Build three de-stacked bundles with their own block-byte buffers (a single
    // expert's gate/up/down slices, non-contiguous → forces the copy path).
    std::vector<std::byte> gate_buf(gate_b), up_buf(up_b), down_buf(down_b);
    for (int64_t i = 0; i < gate_b; ++i) gate_buf[i] = std::byte{0xA0};
    for (int64_t i = 0; i < up_b; ++i)   up_buf[i]   = std::byte{0xB0};
    for (int64_t i = 0; i < down_b; ++i) down_buf[i] = std::byte{0xC0};

    auto make_bundle = [](TensorComponent c, GgufKQuantType t,
                          std::span<const std::byte> data) {
        WeightBundle b;
        b.id = TensorId{c, TensorRole::weight, TensorOwner::routed_expert, 0, 0};
        b.weight.data = data;
        b.weight.dtype = SafetensorsDtype::U8;
        b.weight.gguf_type = t;
        return b;
    };
    std::vector<WeightBundle> bundles;
    bundles.push_back(make_bundle(TensorComponent::gate_proj, GgufKQuantType::Q4_K, gate_buf));
    bundles.push_back(make_bundle(TensorComponent::up_proj, GgufKQuantType::Q4_K, up_buf));
    bundles.push_back(make_bundle(TensorComponent::down_proj, GgufKQuantType::Q6_K, down_buf));

    EXPECT_TRUE(pack_gguf_expert(bundles, shape));

    ASSERT_FALSE(bundles[0].packed_slot.empty());
    EXPECT_EQ(static_cast<int64_t>(bundles[0].packed_slot.size()),
              gate_b + up_b + down_b);
    EXPECT_EQ(bundles[0].packed_slot.size(),
              static_cast<size_t>(quant.bytes_per_expert(shape)));
    // gate | up | down concatenation order, byte-verbatim.
    const auto& slot = bundles[0].packed_slot;
    EXPECT_EQ(static_cast<uint8_t>(slot[0]), 0xA0);
    EXPECT_EQ(static_cast<uint8_t>(slot[gate_b]), 0xB0);
    EXPECT_EQ(static_cast<uint8_t>(slot[gate_b + up_b]), 0xC0);

    // GG-10 strict validation: an expected-triple that doesn't match the
    // bundles' own types is rejected (stale per-layer type table).
    std::vector<WeightBundle> bundles2;
    bundles2.push_back(make_bundle(TensorComponent::gate_proj, GgufKQuantType::Q4_K, gate_buf));
    bundles2.push_back(make_bundle(TensorComponent::up_proj, GgufKQuantType::Q4_K, up_buf));
    bundles2.push_back(make_bundle(TensorComponent::down_proj, GgufKQuantType::Q6_K, down_buf));
    const GgufModelExpertTypes wrong{
        GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q4_K};
    EXPECT_FALSE(pack_gguf_expert(bundles2, shape, &wrong));
    EXPECT_TRUE(bundles2[0].packed_slot.empty());
    const GgufModelExpertTypes right{
        GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q6_K};
    EXPECT_TRUE(pack_gguf_expert(bundles2, shape, &right));
    EXPECT_FALSE(bundles2[0].packed_slot.empty());
}

// ── GGUF weight handler ──────────────────────────────────────────────────────

TEST(GgufLoader, HandlerSelectedByGgufType) {
    WeightBundle b;
    b.weight.dtype = SafetensorsDtype::U8;  // same dtype as NVFP4
    b.weight.gguf_type = GgufKQuantType::Q4_K;
    const WeightHandler& h = handler_for_bundle(b);
    EXPECT_EQ(h.name(), "gguf");
    EXPECT_TRUE(h.expected_aux_roles().empty());

    // No aux → valid. (ModelConfig not used by the GGUF validator.)
    // We can't construct a ModelConfig trivially here; validate's model_cfg is
    // unused, so a nullptr-equivalent is not needed — skip the validate() call
    // and just confirm handler selection + aux-role contract.

    // Without a gguf_type, U8 selects the NVFP4 handler.
    WeightBundle nv;
    nv.weight.dtype = SafetensorsDtype::U8;
    EXPECT_EQ(handler_for_bundle(nv).name(), "nvfp4");
}

// ═══════════════════════════════════════════════════════════════════════════
// DeepSeek-V4 GGUF surface (spec/DEEPSEEK4_PLAN.md V4-2a; DS4_DOSSIER.md §0.3)
// ═══════════════════════════════════════════════════════════════════════════

TEST(GgufLoader, ParseNameCoversV4GgufSurface) {
    struct Case {
        const char* name;
        TensorComponent comp;
        TensorOwner owner;
        int layer;
    };
    const Case cases[] = {
        {"blk.7.attn_kv.weight", TensorComponent::kv_a_proj_with_mqa,
         TensorOwner::attention, 7},
        {"blk.7.attn_output_a.weight", TensorComponent::o_proj_a,
         TensorOwner::attention, 7},
        {"blk.7.attn_output_b.weight", TensorComponent::o_proj_b,
         TensorOwner::attention, 7},
        {"blk.7.attn_sinks.weight", TensorComponent::attn_sinks,
         TensorOwner::attention, 7},
        {"blk.7.attn_compressor_kv.weight", TensorComponent::compressor_wkv,
         TensorOwner::attention, 7},
        {"blk.7.attn_compressor_gate.weight", TensorComponent::compressor_wgate,
         TensorOwner::attention, 7},
        {"blk.7.attn_compressor_ape.weight", TensorComponent::compressor_ape,
         TensorOwner::attention, 7},
        {"blk.7.attn_compressor_norm.weight", TensorComponent::compressor_norm,
         TensorOwner::attention, 7},
        {"blk.7.indexer_compressor_kv.weight",
         TensorComponent::indexer_compressor_wkv, TensorOwner::attention, 7},
        {"blk.7.indexer_compressor_gate.weight",
         TensorComponent::indexer_compressor_wgate, TensorOwner::attention, 7},
        {"blk.7.indexer_compressor_ape.weight",
         TensorComponent::indexer_compressor_ape, TensorOwner::attention, 7},
        {"blk.7.indexer_compressor_norm.weight",
         TensorComponent::indexer_compressor_norm, TensorOwner::attention, 7},
        // V4 reuses the DSA indexer names for the Lightning Indexer.
        {"blk.7.indexer.attn_q_b.weight", TensorComponent::indexer_wq_b,
         TensorOwner::attention, 7},
        {"blk.7.indexer.proj.weight", TensorComponent::indexer_weights_proj,
         TensorOwner::attention, 7},
        {"blk.7.hc_attn_fn.weight", TensorComponent::hc_attn_fn,
         TensorOwner::attention, 7},
        {"blk.7.hc_attn_base.weight", TensorComponent::hc_attn_base,
         TensorOwner::attention, 7},
        {"blk.7.hc_attn_scale.weight", TensorComponent::hc_attn_scale,
         TensorOwner::attention, 7},
        {"blk.7.hc_ffn_fn.weight", TensorComponent::hc_ffn_fn,
         TensorOwner::attention, 7},
        {"blk.7.hc_ffn_base.weight", TensorComponent::hc_ffn_base,
         TensorOwner::attention, 7},
        {"blk.7.hc_ffn_scale.weight", TensorComponent::hc_ffn_scale,
         TensorOwner::attention, 7},
        {"blk.7.ffn_gate_tid2eid.weight", TensorComponent::gate_tid2eid,
         TensorOwner::gating, 7},
        {"output_hc_fn.weight", TensorComponent::output_hc_fn,
         TensorOwner::model_level, -1},
        {"output_hc_base.weight", TensorComponent::output_hc_base,
         TensorOwner::model_level, -1},
        {"output_hc_scale.weight", TensorComponent::output_hc_scale,
         TensorOwner::model_level, -1},
    };
    for (const auto& c : cases) {
        auto id = parse_gguf_name(c.name);
        ASSERT_TRUE(id.has_value()) << c.name;
        EXPECT_EQ(id->component, c.comp) << c.name;
        EXPECT_EQ(id->owner, c.owner) << c.name;
        EXPECT_EQ(id->layer_idx, c.layer) << c.name;
        EXPECT_EQ(id->role, TensorRole::weight) << c.name;
    }
    // V4 ships NO kv_b decompression tensors — but the legacy names still parse
    // (V3.2/GLM regression).
    EXPECT_TRUE(parse_gguf_name("blk.3.attn_kv_b.weight").has_value());
    EXPECT_TRUE(parse_gguf_name("blk.3.attn_kv_a_mqa.weight").has_value());
    EXPECT_TRUE(parse_gguf_name("blk.3.attn_kv_a_norm.weight").has_value());
}

// ── End-to-end V4 mini-model GGUF load: placement + graceful nextn skip ──────

TEST(GgufLoader, V4MiniModelLoadPlacesAllFamilies) {
    // hidden=32, vocab=16, 1 all-MoE hash layer, 2 experts (Q8_0), hc_mult=4.
    constexpr int kH = 32, kV = 16, kE = 2, kHc = 4;

    GgufBlobBuilder b;
    constexpr int32_t F32 = 0, Q8_0 = 8, I32 = 26, BF16 = 30;
    // model-level
    b.add_tensor("token_embd.weight", F32, {kH, kV});
    b.add_tensor("output_norm.weight", F32, {kH});
    b.add_tensor("output.weight", F32, {kH, kV});
    b.add_tensor("output_hc_fn.weight", F32, {kHc * kH, kHc});
    b.add_tensor("output_hc_base.weight", F32, {kHc});
    b.add_tensor("output_hc_scale.weight", F32, {1});
    // layer 0 attention (BF16-native like the real artifact)
    b.add_tensor("blk.0.attn_norm.weight", F32, {kH});
    b.add_tensor("blk.0.ffn_norm.weight", F32, {kH});
    b.add_tensor("blk.0.attn_sinks.weight", F32, {1});
    b.add_tensor("blk.0.attn_q_a.weight", BF16, {kH, kH});
    b.add_tensor("blk.0.attn_q_a_norm.weight", F32, {kH});
    b.add_tensor("blk.0.attn_q_b.weight", BF16, {kH, kH});
    b.add_tensor("blk.0.attn_kv.weight", BF16, {kH, kH});
    b.add_tensor("blk.0.attn_kv_a_norm.weight", F32, {kH});
    b.add_tensor("blk.0.attn_output_a.weight", BF16, {kH, kH});
    b.add_tensor("blk.0.attn_output_b.weight", BF16, {kH, kH});
    b.add_tensor("blk.0.hc_attn_fn.weight", F32, {kHc * kH, (2 + kHc) * kHc});
    b.add_tensor("blk.0.hc_attn_base.weight", F32, {(2 + kHc) * kHc});
    b.add_tensor("blk.0.hc_attn_scale.weight", F32, {3});
    b.add_tensor("blk.0.hc_ffn_fn.weight", F32, {kHc * kH, (2 + kHc) * kHc});
    b.add_tensor("blk.0.hc_ffn_base.weight", F32, {(2 + kHc) * kHc});
    b.add_tensor("blk.0.hc_ffn_scale.weight", F32, {3});
    b.add_tensor("blk.0.attn_compressor_kv.weight", BF16, {kH, 2 * kH});
    b.add_tensor("blk.0.attn_compressor_gate.weight", BF16, {kH, 2 * kH});
    b.add_tensor("blk.0.attn_compressor_ape.weight", F32, {2 * kH, 4});
    b.add_tensor("blk.0.attn_compressor_norm.weight", F32, {kH});
    b.add_tensor("blk.0.indexer.proj.weight", BF16, {kH, 2});
    b.add_tensor("blk.0.indexer.attn_q_b.weight", BF16, {kH, kH});
    b.add_tensor("blk.0.indexer_compressor_kv.weight", BF16, {kH, kH});
    b.add_tensor("blk.0.indexer_compressor_gate.weight", BF16, {kH, kH});
    b.add_tensor("blk.0.indexer_compressor_ape.weight", F32, {kH, 4});
    b.add_tensor("blk.0.indexer_compressor_norm.weight", F32, {kH / 2});
    // gating: hash layer → tid2eid (I32), no exp_probs_b
    b.add_tensor("blk.0.ffn_gate_inp.weight", BF16, {kH, kE});
    b.add_tensor("blk.0.ffn_gate_tid2eid.weight", I32, {1, kV});
    // experts (stacked 3D Q8_0) + shared expert (BF16)
    b.add_tensor("blk.0.ffn_gate_exps.weight", Q8_0, {kH, kH, kE});
    b.add_tensor("blk.0.ffn_up_exps.weight", Q8_0, {kH, kH, kE});
    b.add_tensor("blk.0.ffn_down_exps.weight", Q8_0, {kH, kH, kE});
    b.add_tensor("blk.0.ffn_gate_shexp.weight", BF16, {kH, kH});
    b.add_tensor("blk.0.ffn_up_shexp.weight", BF16, {kH, kH});
    b.add_tensor("blk.0.ffn_down_shexp.weight", BF16, {kH, kH});

    auto blob = b.build();

    // Write to a temp .gguf and load through the real path.
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / "ls3_v4_mini_gguf";
    fs::create_directories(dir);
    fs::path file = dir / "v4-mini.gguf";
    {
        std::ofstream f(file, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(blob.data()),
                static_cast<std::streamsize>(blob.size()));
    }

    config::Config cfg;
    cfg.model.architecture = config::Architecture::deepseek_v4;
    cfg.model.weights_path = file.string();
    cfg.model.weights_format = config::WeightsFormat::gguf;
    cfg.model.num_hidden_layers = 1;
    cfg.model.hidden_size = kH;
    cfg.model.num_attention_heads = 1;
    cfg.model.num_key_value_heads = 1;
    cfg.model.head_dim = kH;
    cfg.model.q_lora_rank = kH;
    cfg.model.intermediate_size = kH;
    cfg.model.moe_intermediate_size = kH;
    cfg.model.n_routed_experts = kE;
    cfg.model.n_shared_experts = 1;
    cfg.model.num_experts_per_tok = 1;
    cfg.model.vocab_size = kV;
    cfg.model.max_position_embeddings = 128;
    cfg.model.first_k_dense_replace = 0;
    cfg.model.moe_layer_freq = 1;
    cfg.model.num_nextn_predict_layers = 1;  // graceful skip: no nextn tensors
    cfg.model.num_hash_layers = 1;
    cfg.model.compress_ratios = {4};
    cfg.model.hc_mult = kHc;
    cfg.model.index_topk = 2;
    cfg.quantization.weights = config::WeightQuant::fp8_e4m3;
    cfg.quantization.attention_compute = config::AttentionQuant::fp8_e4m3;
    cfg.quantization.gating_compute = config::GatingQuant::fp32;
    cfg.hardware.gpus.push_back(config::GpuConfig{
        .id = 0, .type = config::GpuType::rtx5090, .vram_gb = 32.0});
    cfg.hardware.tp_array = {0};

    ModelConfig mcfg(cfg);
    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    // num_nextn_predict_layers=1 with zero nextn tensors must NOT throw for V4.
    LoadedModel model;
    ASSERT_NO_THROW(model = load_weights(cfg, mcfg, registry));

    // Model-level: output_hc trio present.
    EXPECT_TRUE(model.embedding.has_value());
    EXPECT_TRUE(model.output_head.has_value());
    EXPECT_TRUE(model.final_norm.has_value());
    ASSERT_EQ(model.output_hc.size(), 3u);
    EXPECT_FALSE(model.mtp.has_value());

    ASSERT_EQ(model.layers.size(), 1u);
    const auto& layer = model.layers[0];

    auto has_comp = [](const std::vector<WeightBundle>& v, TensorComponent c) {
        for (const auto& b2 : v)
            if (b2.id.component == c) return true;
        return false;
    };

    // Attention families all placed in layer.attention.
    for (auto c : {TensorComponent::q_a_proj, TensorComponent::q_b_proj,
                   TensorComponent::kv_a_proj_with_mqa, TensorComponent::o_proj_a,
                   TensorComponent::o_proj_b, TensorComponent::attn_sinks,
                   TensorComponent::compressor_wkv, TensorComponent::compressor_wgate,
                   TensorComponent::compressor_ape, TensorComponent::compressor_norm,
                   TensorComponent::hc_attn_fn, TensorComponent::hc_attn_base,
                   TensorComponent::hc_attn_scale, TensorComponent::hc_ffn_fn,
                   TensorComponent::hc_ffn_base, TensorComponent::hc_ffn_scale}) {
        EXPECT_TRUE(has_comp(layer.attention, c))
            << tensor_component_name(c) << " missing from layer.attention";
    }
    // Indexer families (incl. V4 indexer-compressor) in layer.indexer.
    for (auto c : {TensorComponent::indexer_wq_b, TensorComponent::indexer_weights_proj,
                   TensorComponent::indexer_compressor_wkv,
                   TensorComponent::indexer_compressor_wgate,
                   TensorComponent::indexer_compressor_ape,
                   TensorComponent::indexer_compressor_norm}) {
        EXPECT_TRUE(has_comp(layer.indexer, c))
            << tensor_component_name(c) << " missing from layer.indexer";
    }
    // Gating: router + hash table.
    EXPECT_TRUE(has_comp(layer.gating, TensorComponent::gate_weight));
    EXPECT_TRUE(has_comp(layer.gating, TensorComponent::gate_tid2eid));
    for (const auto& g : layer.gating) {
        if (g.id.component == TensorComponent::gate_tid2eid) {
            EXPECT_EQ(g.weight.dtype, SafetensorsDtype::I32);
            ASSERT_EQ(g.weight.shape.size(), 2u);
            EXPECT_EQ(g.weight.shape[0], kV);  // row-major [vocab, topk] after reversal
            EXPECT_EQ(g.weight.shape[1], 1);
        }
    }
    // GG-9 requant is GATED OFF for V4: BF16 projections stay float.
    for (const auto& a : layer.attention) {
        if (a.id.component == TensorComponent::q_a_proj ||
            a.id.component == TensorComponent::q_b_proj ||
            a.id.component == TensorComponent::kv_a_proj_with_mqa) {
            EXPECT_FALSE(a.weight.gguf_type.has_value())
                << tensor_component_name(a.id.component)
                << " was requantized — GG-9 must be V4-gated";
        }
    }
    // Experts de-stacked with Q8_0 typing.
    ASSERT_EQ(layer.routed_experts.size(), 2u);
    for (const auto& expert : layer.routed_experts) {
        ASSERT_EQ(expert.size(), 3u);
        for (const auto& e : expert) {
            ASSERT_TRUE(e.weight.gguf_type.has_value());
            EXPECT_EQ(*e.weight.gguf_type, GgufKQuantType::Q8_0);
        }
    }
    EXPECT_FALSE(layer.shared_expert.empty());

    fs::remove_all(dir);
}

// ── MXFP4 stacked experts: reader accepts type 39 + de-stack sizing ─────────

TEST(GgufLoader, Mxfp4ExpertEntriesParseAndSize) {
    // V4-Flash-shaped (scaled down): gate [4096-in, 2048-out] per expert.
    const int hidden = 128, inter = 64, n_exp = 4;
    constexpr int32_t kGGML_MXFP4 = 39;
    GgufBlobBuilder b;
    b.add_tensor("blk.0.ffn_gate_exps.weight", kGGML_MXFP4, {hidden, inter, n_exp});
    auto blob = b.build();

    GgufReader r = GgufReader::from_buffer(blob, "mxfp4-exps");
    const auto& e = r.entries()[0];
    EXPECT_TRUE(e.is_kquant());
    EXPECT_EQ(e.kquant_type(), GgufKQuantType::MXFP4);
    EXPECT_TRUE(is_supported_gguf_kquant(kGGML_MXFP4));

    auto data = r.tensor_data(e);
    const int64_t per_expert =
        gguf::gguf_packed_bytes(inter, hidden, GgufKQuantType::MXFP4);
    EXPECT_EQ(per_expert, static_cast<int64_t>(inter) * (hidden / 32) * 17);
    EXPECT_EQ(per_expert * n_exp, static_cast<int64_t>(data.size()));
}

// ═══════════════════════════════════════════════════════════════════════════
// glm5_next (GLM-5.3-Flash) GGUF surface — GF3.9
// ═══════════════════════════════════════════════════════════════════════════

// The COMPLETE measured tensor-name surface of the real GLM-5.3-Flash
// UD-Q4_K_XL GGUF (unsloth, 1412 tensors over 6 shards, dumped 2026-08-29 —
// scratchpad/gf39/SURVEY_BOOT.md §3.7): blk.0 = KDA linear layer with dense
// FFN (26 tensors), blk.3 = sparse MLA + IndexPool indexer + MoE (31), blk.45 =
// MTP/nextn block (29), plus the 3 non-blk tensors. Every name must map, and to
// the RIGHT component — an unmapped name is silently dropped at load
// (weight_loader.cpp "Unrecognized GGUF tensor name") and a mis-mapped one
// loads the wrong anatomy.
TEST(GgufLoader, ParseNameCoversGlm53GgufSurface) {
    struct Case {
        const char* name;
        TensorComponent comp;
        TensorOwner owner;
        int layer;
        TensorRole role = TensorRole::weight;
    };
    const Case cases[] = {
        // ── blk.0 — KDA linear-attention layer, dense FFN ──
        {"blk.0.attn_norm.weight", TensorComponent::input_layernorm,
         TensorOwner::attention, 0},
        {"blk.0.attn_q.weight", TensorComponent::kda_q_proj,
         TensorOwner::attention, 0},
        {"blk.0.attn_k.weight", TensorComponent::kda_k_proj,
         TensorOwner::attention, 0},
        {"blk.0.attn_v.weight", TensorComponent::kda_v_proj,
         TensorOwner::attention, 0},
        {"blk.0.attn_output.weight", TensorComponent::o_proj,
         TensorOwner::attention, 0},
        // NO role suffix in the file — the narrow escape in parse_gguf_name.
        {"blk.0.ssm_a", TensorComponent::kda_a_log, TensorOwner::attention, 0},
        {"blk.0.ssm_beta.weight", TensorComponent::kda_b_proj,
         TensorOwner::attention, 0},
        {"blk.0.ssm_conv1d_q.weight", TensorComponent::kda_q_conv1d,
         TensorOwner::attention, 0},
        {"blk.0.ssm_conv1d_k.weight", TensorComponent::kda_k_conv1d,
         TensorOwner::attention, 0},
        {"blk.0.ssm_conv1d_v.weight", TensorComponent::kda_v_conv1d,
         TensorOwner::attention, 0},
        // `.bias` suffix, but it IS the main tensor: role normalized to weight.
        {"blk.0.ssm_dt.bias", TensorComponent::kda_dt_bias,
         TensorOwner::attention, 0},
        {"blk.0.ssm_f_a.weight", TensorComponent::kda_f_a_proj,
         TensorOwner::attention, 0},
        {"blk.0.ssm_f_b.weight", TensorComponent::kda_f_b_proj,
         TensorOwner::attention, 0},
        {"blk.0.ssm_g_a.weight", TensorComponent::kda_g_a_proj,
         TensorOwner::attention, 0},
        {"blk.0.ssm_g_b.weight", TensorComponent::kda_g_b_proj,
         TensorOwner::attention, 0},
        {"blk.0.ssm_norm.weight", TensorComponent::kda_o_norm,
         TensorOwner::attention, 0},
        {"blk.0.ffn_norm.weight", TensorComponent::post_attention_layernorm,
         TensorOwner::attention, 0},
        {"blk.0.ffn_gate.weight", TensorComponent::gate_proj,
         TensorOwner::dense_ffn, 0},
        {"blk.0.ffn_up.weight", TensorComponent::up_proj,
         TensorOwner::dense_ffn, 0},
        {"blk.0.ffn_down.weight", TensorComponent::down_proj,
         TensorOwner::dense_ffn, 0},
        {"blk.0.hc_attn_base.weight", TensorComponent::hc_attn_base,
         TensorOwner::attention, 0},
        {"blk.0.hc_attn_fn.weight", TensorComponent::hc_attn_fn,
         TensorOwner::attention, 0},
        {"blk.0.hc_attn_scale.weight", TensorComponent::hc_attn_scale,
         TensorOwner::attention, 0},
        {"blk.0.hc_ffn_base.weight", TensorComponent::hc_ffn_base,
         TensorOwner::attention, 0},
        {"blk.0.hc_ffn_fn.weight", TensorComponent::hc_ffn_fn,
         TensorOwner::attention, 0},
        {"blk.0.hc_ffn_scale.weight", TensorComponent::hc_ffn_scale,
         TensorOwner::attention, 0},

        // ── blk.3 — sparse MLA + IndexPool indexer + MoE ──
        {"blk.3.attn_norm.weight", TensorComponent::input_layernorm,
         TensorOwner::attention, 3},
        {"blk.3.attn_q_a.weight", TensorComponent::q_a_proj,
         TensorOwner::attention, 3},
        {"blk.3.attn_q_a_norm.weight", TensorComponent::q_a_norm,
         TensorOwner::attention, 3},
        {"blk.3.attn_q_b.weight", TensorComponent::q_b_proj,
         TensorOwner::attention, 3},
        {"blk.3.attn_kv_a_mqa.weight", TensorComponent::kv_a_proj_with_mqa,
         TensorOwner::attention, 3},
        {"blk.3.attn_kv_a_norm.weight", TensorComponent::kv_a_norm,
         TensorOwner::attention, 3},
        {"blk.3.attn_k_b.weight", TensorComponent::mla_k_b_split,
         TensorOwner::attention, 3},
        {"blk.3.attn_v_b.weight", TensorComponent::mla_v_b_split,
         TensorOwner::attention, 3},
        {"blk.3.attn_output.weight", TensorComponent::o_proj,
         TensorOwner::attention, 3},
        {"blk.3.indexer.attn_q_b.weight", TensorComponent::indexer_wq_b,
         TensorOwner::attention, 3},
        {"blk.3.indexer.attn_k.weight", TensorComponent::indexer_wk,
         TensorOwner::attention, 3},
        {"blk.3.indexer.k_norm.weight", TensorComponent::indexer_k_norm_weight,
         TensorOwner::attention, 3},
        {"blk.3.indexer.k_norm.bias", TensorComponent::indexer_k_norm_bias,
         TensorOwner::attention, 3, TensorRole::bias},
        {"blk.3.indexer.proj.weight", TensorComponent::indexer_weights_proj,
         TensorOwner::attention, 3},
        {"blk.3.indexer_compressor_ape.weight",
         TensorComponent::indexer_compressor_ape, TensorOwner::attention, 3},
        {"blk.3.indexer_compressor_gate.weight",
         TensorComponent::indexer_compressor_wgate, TensorOwner::attention, 3},
        {"blk.3.ffn_norm.weight", TensorComponent::post_attention_layernorm,
         TensorOwner::attention, 3},
        {"blk.3.ffn_gate_inp.weight", TensorComponent::gate_weight,
         TensorOwner::gating, 3},
        {"blk.3.exp_probs_b.bias",
         TensorComponent::gate_e_score_correction_bias, TensorOwner::gating, 3,
         TensorRole::bias},
        {"blk.3.ffn_gate_exps.weight", TensorComponent::gate_proj,
         TensorOwner::routed_expert, 3},
        {"blk.3.ffn_up_exps.weight", TensorComponent::up_proj,
         TensorOwner::routed_expert, 3},
        {"blk.3.ffn_down_exps.weight", TensorComponent::down_proj,
         TensorOwner::routed_expert, 3},
        {"blk.3.ffn_gate_shexp.weight", TensorComponent::gate_proj,
         TensorOwner::shared_expert, 3},
        {"blk.3.ffn_up_shexp.weight", TensorComponent::up_proj,
         TensorOwner::shared_expert, 3},
        {"blk.3.ffn_down_shexp.weight", TensorComponent::down_proj,
         TensorOwner::shared_expert, 3},
        {"blk.3.hc_attn_base.weight", TensorComponent::hc_attn_base,
         TensorOwner::attention, 3},
        {"blk.3.hc_attn_fn.weight", TensorComponent::hc_attn_fn,
         TensorOwner::attention, 3},
        {"blk.3.hc_attn_scale.weight", TensorComponent::hc_attn_scale,
         TensorOwner::attention, 3},
        {"blk.3.hc_ffn_base.weight", TensorComponent::hc_ffn_base,
         TensorOwner::attention, 3},
        {"blk.3.hc_ffn_fn.weight", TensorComponent::hc_ffn_fn,
         TensorOwner::attention, 3},
        {"blk.3.hc_ffn_scale.weight", TensorComponent::hc_ffn_scale,
         TensorOwner::attention, 3},

        // ── blk.45 — MTP / nextn block (no hc_*, no embed/head extras) ──
        {"blk.45.attn_norm.weight", TensorComponent::input_layernorm,
         TensorOwner::attention, 45},
        {"blk.45.attn_q_a.weight", TensorComponent::q_a_proj,
         TensorOwner::attention, 45},
        {"blk.45.attn_q_a_norm.weight", TensorComponent::q_a_norm,
         TensorOwner::attention, 45},
        {"blk.45.attn_q_b.weight", TensorComponent::q_b_proj,
         TensorOwner::attention, 45},
        {"blk.45.attn_kv_a_mqa.weight", TensorComponent::kv_a_proj_with_mqa,
         TensorOwner::attention, 45},
        {"blk.45.attn_kv_a_norm.weight", TensorComponent::kv_a_norm,
         TensorOwner::attention, 45},
        {"blk.45.attn_k_b.weight", TensorComponent::mla_k_b_split,
         TensorOwner::attention, 45},
        {"blk.45.attn_v_b.weight", TensorComponent::mla_v_b_split,
         TensorOwner::attention, 45},
        {"blk.45.attn_output.weight", TensorComponent::o_proj,
         TensorOwner::attention, 45},
        {"blk.45.exp_probs_b.bias",
         TensorComponent::gate_e_score_correction_bias, TensorOwner::gating, 45,
         TensorRole::bias},
        {"blk.45.ffn_down_exps.weight", TensorComponent::down_proj,
         TensorOwner::routed_expert, 45},
        {"blk.45.ffn_down_shexp.weight", TensorComponent::down_proj,
         TensorOwner::shared_expert, 45},
        {"blk.45.ffn_gate_exps.weight", TensorComponent::gate_proj,
         TensorOwner::routed_expert, 45},
        {"blk.45.ffn_up_exps.weight", TensorComponent::up_proj,
         TensorOwner::routed_expert, 45},
        {"blk.45.ffn_gate_inp.weight", TensorComponent::gate_weight,
         TensorOwner::gating, 45},
        {"blk.45.ffn_norm.weight", TensorComponent::post_attention_layernorm,
         TensorOwner::attention, 45},
        {"blk.45.ffn_gate_shexp.weight", TensorComponent::gate_proj,
         TensorOwner::shared_expert, 45},
        {"blk.45.ffn_up_shexp.weight", TensorComponent::up_proj,
         TensorOwner::shared_expert, 45},
        {"blk.45.indexer.attn_k.weight", TensorComponent::indexer_wk,
         TensorOwner::attention, 45},
        {"blk.45.indexer.attn_q_b.weight", TensorComponent::indexer_wq_b,
         TensorOwner::attention, 45},
        {"blk.45.indexer.k_norm.weight", TensorComponent::indexer_k_norm_weight,
         TensorOwner::attention, 45},
        {"blk.45.indexer.k_norm.bias", TensorComponent::indexer_k_norm_bias,
         TensorOwner::attention, 45, TensorRole::bias},
        {"blk.45.indexer.proj.weight", TensorComponent::indexer_weights_proj,
         TensorOwner::attention, 45},
        {"blk.45.indexer_compressor_ape.weight",
         TensorComponent::indexer_compressor_ape, TensorOwner::attention, 45},
        {"blk.45.indexer_compressor_gate.weight",
         TensorComponent::indexer_compressor_wgate, TensorOwner::attention, 45},
        {"blk.45.nextn.eh_proj.weight", TensorComponent::mtp_eh_proj,
         TensorOwner::mtp, 45},
        {"blk.45.nextn.enorm.weight", TensorComponent::mtp_enorm,
         TensorOwner::mtp, 45},
        {"blk.45.nextn.hnorm.weight", TensorComponent::mtp_hnorm,
         TensorOwner::mtp, 45},
        {"blk.45.nextn.shared_head_norm.weight",
         TensorComponent::mtp_shared_head_norm, TensorOwner::mtp, 45},

        // ── the 3 non-blk tensors ──
        {"token_embd.weight", TensorComponent::embedding,
         TensorOwner::model_level, -1},
        {"output.weight", TensorComponent::output_head,
         TensorOwner::model_level, -1},
        {"output_norm.weight", TensorComponent::final_norm,
         TensorOwner::model_level, -1},
    };
    for (const auto& c : cases) {
        auto id = parse_gguf_name(c.name);
        ASSERT_TRUE(id.has_value()) << "unmapped: " << c.name;
        EXPECT_EQ(id->component, c.comp) << c.name;
        EXPECT_EQ(id->owner, c.owner) << c.name;
        EXPECT_EQ(id->layer_idx, c.layer) << c.name;
        EXPECT_EQ(id->role, c.role) << c.name;
    }

    // The suffix-less escape is NARROW: only `ssm_a`.
    EXPECT_FALSE(parse_gguf_name("blk.0.ssm_b").has_value());
    EXPECT_FALSE(parse_gguf_name("blk.0.ssm_xyz.weight").has_value());
    EXPECT_FALSE(parse_gguf_name("blk.0.ssm_conv1d_x.weight").has_value());
    EXPECT_FALSE(parse_gguf_name("ssm_a").has_value());
    EXPECT_FALSE(parse_gguf_name("blk.x.ssm_a").has_value());
}

// ── End-to-end mini-glm5_next GGUF load ─────────────────────────────────────
// Geometry mirrors GLM-5.3-Flash scaled down: 2 hidden layers (0 = KDA linear
// with dense FFN, 1 = sparse MLA + IndexPool indexer + MoE) plus one MTP block,
// in the REAL measured GGUF naming + dtype scheme (Q8_0 projections, F32 norms/
// A_log/dt_bias/ape, Q4_K stacked experts).

namespace {

constexpr int64_t kG5Hidden = 256;
constexpr int64_t kG5KdaHeads = 4;
constexpr int64_t kG5KdaDim = 64;   // heads*dim == hidden
constexpr int64_t kG5ConvK = 4;
constexpr int64_t kG5Heads = 4;
constexpr int64_t kG5QLora = 128;
constexpr int64_t kG5KvLora = 128;
constexpr int64_t kG5QkNope = 64;
constexpr int64_t kG5VHead = 64;
constexpr int64_t kG5IdxDim = 64;
constexpr int64_t kG5IdxHeads = 4;
constexpr int64_t kG5Kpool = 4;
constexpr int64_t kG5Vocab = 512;
constexpr int64_t kG5Experts = 4;
constexpr int64_t kG5MoeInter = 256;  // Q4_K needs in_features % 256 == 0
constexpr int64_t kG5Inter = 256;
constexpr int64_t kG5HcMult = 4;
constexpr int kG5Layers = 2;       // hidden layers
constexpr int kG5SparseLayer = 1;
constexpr int kG5MtpLayer = 2;     // blk index of the MTP block

/// One synthetic GLM-5.3-Flash-shaped GGUF. `head_count_kv` is written as the
/// `glm5next.attention.head_count_kv` array when non-empty (0 = KDA layer).
std::vector<std::byte> build_glm53_mini_gguf(
    const std::vector<int32_t>& head_count_kv) {
    constexpr int32_t F32 = 0, Q4_K = 12, Q8_0 = 8;
    GgufBlobBuilder b;
    if (!head_count_kv.empty())
        b.add_kv_i32_array("glm5next.attention.head_count_kv", head_count_kv);

    // Model level (both Q8_0 and untied, as measured).
    b.add_tensor("token_embd.weight", Q8_0, {kG5Hidden, kG5Vocab});
    b.add_tensor("output.weight", Q8_0, {kG5Hidden, kG5Vocab});
    b.add_tensor("output_norm.weight", F32, {kG5Hidden});

    auto add_hc = [&](const std::string& blk) {
        for (const char* w : {"attn", "ffn"}) {
            b.add_tensor(blk + ".hc_" + w + "_base.weight", F32,
                         {(2 + kG5HcMult) * kG5HcMult});
            b.add_tensor(blk + ".hc_" + w + "_fn.weight", Q8_0,
                         {kG5HcMult * kG5Hidden, (2 + kG5HcMult) * kG5HcMult});
            b.add_tensor(blk + ".hc_" + w + "_scale.weight", F32, {3});
        }
    };

    // ── blk.0: KDA linear attention + dense FFN ──
    b.add_tensor("blk.0.attn_norm.weight", F32, {kG5Hidden});
    for (const char* p : {"attn_q", "attn_k", "attn_v"})
        b.add_tensor(std::string("blk.0.") + p + ".weight", Q8_0,
                     {kG5Hidden, kG5KdaHeads * kG5KdaDim});
    b.add_tensor("blk.0.attn_output.weight", Q8_0,
                 {kG5KdaHeads * kG5KdaDim, kG5Hidden});
    b.add_tensor("blk.0.ssm_a", F32, {kG5KdaHeads});              // no suffix
    b.add_tensor("blk.0.ssm_beta.weight", Q8_0, {kG5Hidden, kG5KdaHeads});
    for (const char* c : {"q", "k", "v"})
        b.add_tensor(std::string("blk.0.ssm_conv1d_") + c + ".weight", F32,
                     {kG5ConvK, 1, kG5KdaHeads * kG5KdaDim});
    b.add_tensor("blk.0.ssm_dt.bias", F32, {kG5KdaHeads * kG5KdaDim});
    b.add_tensor("blk.0.ssm_f_a.weight", Q8_0, {kG5Hidden, kG5KdaDim});
    b.add_tensor("blk.0.ssm_f_b.weight", Q8_0,
                 {kG5KdaDim, kG5KdaHeads * kG5KdaDim});
    b.add_tensor("blk.0.ssm_g_a.weight", Q8_0, {kG5Hidden, kG5KdaDim});
    b.add_tensor("blk.0.ssm_g_b.weight", Q8_0,
                 {kG5KdaDim, kG5KdaHeads * kG5KdaDim});
    b.add_tensor("blk.0.ssm_norm.weight", F32, {kG5KdaDim});
    b.add_tensor("blk.0.ffn_norm.weight", F32, {kG5Hidden});
    b.add_tensor("blk.0.ffn_gate.weight", Q8_0, {kG5Hidden, kG5Inter});
    b.add_tensor("blk.0.ffn_up.weight", Q8_0, {kG5Hidden, kG5Inter});
    b.add_tensor("blk.0.ffn_down.weight", Q8_0, {kG5Inter, kG5Hidden});
    add_hc("blk.0");

    // ── sparse MLA + indexer + MoE, shared by blk.1 and the MTP blk.2 ──
    auto add_sparse = [&](const std::string& blk, bool with_moe) {
        b.add_tensor(blk + ".attn_norm.weight", F32, {kG5Hidden});
        b.add_tensor(blk + ".attn_q_a.weight", Q8_0, {kG5Hidden, kG5QLora});
        b.add_tensor(blk + ".attn_q_a_norm.weight", F32, {kG5QLora});
        b.add_tensor(blk + ".attn_q_b.weight", Q8_0,
                     {kG5QLora, kG5Heads * kG5QkNope});
        b.add_tensor(blk + ".attn_kv_a_mqa.weight", Q8_0,
                     {kG5Hidden, kG5KvLora});
        b.add_tensor(blk + ".attn_kv_a_norm.weight", F32, {kG5KvLora});
        // Split MLA up-projection: file order {P,L,H} / {L,V,H}.
        b.add_tensor(blk + ".attn_k_b.weight", Q8_0,
                     {kG5QkNope, kG5KvLora, kG5Heads});
        b.add_tensor(blk + ".attn_v_b.weight", Q8_0,
                     {kG5KvLora, kG5VHead, kG5Heads});
        b.add_tensor(blk + ".attn_output.weight", Q8_0,
                     {kG5Heads * kG5VHead, kG5Hidden});
        b.add_tensor(blk + ".indexer.attn_q_b.weight", Q8_0,
                     {kG5QLora, kG5IdxHeads * kG5IdxDim});
        b.add_tensor(blk + ".indexer.attn_k.weight", Q8_0,
                     {kG5Hidden, kG5IdxDim});
        b.add_tensor(blk + ".indexer.k_norm.weight", F32, {kG5IdxDim});
        b.add_tensor(blk + ".indexer.k_norm.bias", F32, {kG5IdxDim});
        b.add_tensor(blk + ".indexer.proj.weight", F32,
                     {kG5Hidden, kG5IdxHeads});
        // IndexPool compressor: ape ships F32, gate ships Q8_0 (the GF3.9
        // load-time dequant target).
        b.add_tensor(blk + ".indexer_compressor_ape.weight", F32,
                     {kG5IdxDim, kG5Kpool});
        b.add_tensor(blk + ".indexer_compressor_gate.weight", Q8_0,
                     {kG5Hidden, kG5IdxDim});
        b.add_tensor(blk + ".ffn_norm.weight", F32, {kG5Hidden});
        if (!with_moe) return;
        b.add_tensor(blk + ".ffn_gate_inp.weight", F32, {kG5Hidden, kG5Experts});
        b.add_tensor(blk + ".exp_probs_b.bias", F32, {kG5Experts});
        b.add_tensor(blk + ".ffn_gate_exps.weight", Q4_K,
                     {kG5Hidden, kG5MoeInter, kG5Experts});
        b.add_tensor(blk + ".ffn_up_exps.weight", Q4_K,
                     {kG5Hidden, kG5MoeInter, kG5Experts});
        b.add_tensor(blk + ".ffn_down_exps.weight", Q4_K,
                     {kG5MoeInter, kG5Hidden, kG5Experts});
        b.add_tensor(blk + ".ffn_gate_shexp.weight", Q8_0,
                     {kG5Hidden, kG5MoeInter});
        b.add_tensor(blk + ".ffn_up_shexp.weight", Q8_0,
                     {kG5Hidden, kG5MoeInter});
        b.add_tensor(blk + ".ffn_down_shexp.weight", Q8_0,
                     {kG5MoeInter, kG5Hidden});
    };
    add_sparse("blk.1", /*with_moe=*/true);
    add_hc("blk.1");

    // ── blk.2: MTP block — sparse MLA, NO hc_*, 4 nextn extras ──
    add_sparse("blk.2", /*with_moe=*/false);
    b.add_tensor("blk.2.nextn.eh_proj.weight", Q8_0,
                 {2 * kG5Hidden, kG5Hidden});
    b.add_tensor("blk.2.nextn.enorm.weight", F32, {kG5Hidden});
    b.add_tensor("blk.2.nextn.hnorm.weight", F32, {kG5Hidden});
    b.add_tensor("blk.2.nextn.shared_head_norm.weight", F32, {kG5Hidden});

    return b.build();
}

/// The engine-side recipe for the mini GGUF (GLM-5.3-Flash shape, shrunk).
/// `layer_types` deliberately mirrors the head_count_kv the test writes — the
/// mismatch tests perturb one or the other.
layerstorm::config::Config glm53_mini_config(
    const std::filesystem::path& gguf_file, bool layer0_linear = true) {
    nlohmann::json layer_types = nlohmann::json::array();
    for (int l = 0; l < kG5Layers; ++l) {
        const bool linear = (l == 0) ? layer0_linear : (l != kG5SparseLayer);
        layer_types.push_back(linear ? "linear_attention"
                                     : "deepseek_sparse_attention");
    }
    nlohmann::json j = {
        {"model",
         {{"architecture", "glm5_next"},
          {"weights_path", gguf_file.string()},
          {"weights_format", "gguf"},
          {"num_hidden_layers", kG5Layers},
          {"hidden_size", kG5Hidden},
          {"num_attention_heads", kG5Heads},
          {"num_key_value_heads", kG5Heads},
          {"intermediate_size", kG5Inter},
          {"n_routed_experts", kG5Experts},
          {"n_shared_experts", 1},
          {"num_experts_per_tok", 2},
          {"n_group", 1},
          {"topk_group", 1},
          {"vocab_size", kG5Vocab},
          {"max_position_embeddings", 4096},
          {"kv_lora_rank", kG5KvLora},
          {"q_lora_rank", kG5QLora},
          {"qk_rope_head_dim", 0},
          {"qk_nope_head_dim", kG5QkNope},
          {"v_head_dim", kG5VHead},
          {"first_k_dense_replace", kG5SparseLayer},
          {"moe_layer_freq", 1},
          {"index_topk", 16},
          {"index_n_heads", kG5IdxHeads},
          {"index_head_dim", kG5IdxDim},
          {"index_kpool", kG5Kpool},
          {"index_kpool_compress", true},
          {"index_kpool_always_select_tail", true},
          {"mla_use_nope", true},
          {"layer_types", layer_types},
          {"linear_attn_config",
           {{"num_heads", kG5KdaHeads},
            {"head_dim", kG5KdaDim},
            {"short_conv_kernel_size", kG5ConvK},
            {"gate_lower_bound", -5.0}}},
          {"hc_mult", kG5HcMult},
          {"hc_sinkhorn_iters", 20},
          {"hc_eps", 1e-6},
          {"swiglu_limit", 10.0},
          {"num_nextn_predict_layers", 1},
          {"rms_norm_eps", 1e-5},
          {"routed_scaling_factor", 2.5},
          {"moe_intermediate_size", kG5MoeInter}}},
        {"quantization",
         {{"weights", "fp8_e4m3"},
          {"attention_compute", "fp8_e4m3"},
          {"kv_cache", "fp8_e4m3"},
          {"gating_compute", "fp32"}}},
        {"hardware",
         {{"gpus", {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}}}},
          {"system_ram_gb", 256}}},
    };
    return layerstorm::config::parse_config(j);
}

/// Writes one mini GGUF into a fresh temp dir; removes it on destruction.
class Glm53MiniGgufFixture {
public:
    explicit Glm53MiniGgufFixture(const std::string& tag,
                                  const std::vector<int32_t>& head_count_kv) {
        namespace fs = std::filesystem;
        dir_ = fs::temp_directory_path() /
               ("ls3_glm53_mini_" + tag + "_" + std::to_string(::getpid()));
        std::error_code ec;
        fs::remove_all(dir_, ec);
        fs::create_directories(dir_);
        file_ = dir_ / "glm53-mini.gguf";
        const auto blob = build_glm53_mini_gguf(head_count_kv);
        std::ofstream f(file_, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(blob.data()),
                static_cast<std::streamsize>(blob.size()));
    }
    ~Glm53MiniGgufFixture() {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }
    const std::filesystem::path& file() const { return file_; }

private:
    std::filesystem::path dir_, file_;
};

const WeightBundle* find_in(const std::vector<WeightBundle>& v,
                            TensorComponent c) {
    for (const auto& b : v)
        if (b.id.component == c) return &b;
    return nullptr;
}

}  // namespace

// The Q8_0 IndexPool compressor gate must arrive at the executor as BF16 (the
// executor's kpool GEMM is BF16-only and nothing tags this component as a GGUF
// operand — SURVEY_BOOT.md §4.5 / R7). Everything else keeps its shipped dtype.
TEST(GgufLoader, Glm53MiniGgufLoadDequantsCompressorGate) {
    Glm53MiniGgufFixture fx("gate", {0, 1, 1});
    auto cfg = glm53_mini_config(fx.file());
    ModelConfig mcfg(cfg);
    ASSERT_TRUE(mcfg.is_glm5_next());
    ASSERT_TRUE(mcfg.has_index_pool());
    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    LoadedModel model;
    ASSERT_NO_THROW(model = load_weights(cfg, mcfg, registry));
    ASSERT_EQ(model.layers.size(), static_cast<size_t>(kG5Layers));

    // ── layer 0: the full KDA anatomy parsed and placed ──
    const auto& l0 = model.layers[0];
    for (auto c : {TensorComponent::kda_q_proj, TensorComponent::kda_k_proj,
                   TensorComponent::kda_v_proj, TensorComponent::kda_b_proj,
                   TensorComponent::kda_f_a_proj, TensorComponent::kda_f_b_proj,
                   TensorComponent::kda_g_a_proj, TensorComponent::kda_g_b_proj,
                   TensorComponent::kda_q_conv1d, TensorComponent::kda_k_conv1d,
                   TensorComponent::kda_v_conv1d, TensorComponent::kda_a_log,
                   TensorComponent::kda_dt_bias, TensorComponent::kda_o_norm,
                   TensorComponent::o_proj}) {
        EXPECT_NE(find_in(l0.attention, c), nullptr)
            << "layer 0 missing " << tensor_component_name(c);
    }
    EXPECT_TRUE(l0.indexer.empty());
    EXPECT_FALSE(l0.dense_ffn.empty());

    const auto* a_log = find_in(l0.attention, TensorComponent::kda_a_log);
    ASSERT_NE(a_log, nullptr);
    EXPECT_EQ(a_log->weight.dtype, SafetensorsDtype::F32);
    EXPECT_EQ(a_log->id.role, TensorRole::weight);
    EXPECT_EQ(a_log->weight.shape, (std::vector<int64_t>{kG5KdaHeads}));

    const auto* dt = find_in(l0.attention, TensorComponent::kda_dt_bias);
    ASSERT_NE(dt, nullptr);
    EXPECT_EQ(dt->id.role, TensorRole::weight);  // `.bias` normalized to weight
    EXPECT_EQ(dt->weight.dtype, SafetensorsDtype::F32);
    EXPECT_TRUE(dt->aux.empty());

    const auto* conv = find_in(l0.attention, TensorComponent::kda_q_conv1d);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->weight.shape,
              (std::vector<int64_t>{kG5KdaHeads * kG5KdaDim, 1, kG5ConvK}));

    const auto* kq = find_in(l0.attention, TensorComponent::kda_q_proj);
    ASSERT_NE(kq, nullptr);
    ASSERT_TRUE(kq->weight.gguf_type.has_value());  // Q8_0 stays packed
    EXPECT_EQ(*kq->weight.gguf_type, GgufKQuantType::Q8_0);

    // ── layer 1: the compressor gate is BF16, the ape untouched F32 ──
    const auto& l1 = model.layers[kG5SparseLayer];
    const auto* gate = find_in(l1.indexer, TensorComponent::indexer_compressor_wgate);
    ASSERT_NE(gate, nullptr);
    EXPECT_FALSE(gate->weight.gguf_type.has_value());
    EXPECT_EQ(gate->weight.dtype, SafetensorsDtype::BF16);
    EXPECT_EQ(gate->weight.shape, (std::vector<int64_t>{kG5IdxDim, kG5Hidden}));
    EXPECT_EQ(static_cast<int64_t>(gate->weight.data.size()),
              kG5IdxDim * kG5Hidden * 2);

    const auto* ape = find_in(l1.indexer, TensorComponent::indexer_compressor_ape);
    ASSERT_NE(ape, nullptr);
    EXPECT_FALSE(ape->weight.gguf_type.has_value());
    EXPECT_EQ(ape->weight.dtype, SafetensorsDtype::F32);
    EXPECT_EQ(ape->weight.shape, (std::vector<int64_t>{kG5Kpool, kG5IdxDim}));

    // Other Q8_0 indexer weights are NOT dequanted (only the gate is).
    const auto* iwk = find_in(l1.indexer, TensorComponent::indexer_wk);
    ASSERT_NE(iwk, nullptr);
    EXPECT_TRUE(iwk->weight.gguf_type.has_value());

    // Split kv_b assembled to BF16 on both sparse blocks.
    const auto* kvb = find_in(l1.attention, TensorComponent::kv_b_proj);
    ASSERT_NE(kvb, nullptr);
    EXPECT_EQ(kvb->weight.dtype, SafetensorsDtype::BF16);

    // ── the MTP block's gate is dequanted too ──
    ASSERT_TRUE(model.mtp.has_value());
    ASSERT_EQ(model.mtp->block_layers.size(), 1u);
    const auto& mtp_blk = model.mtp->block_layers[0];
    EXPECT_EQ(mtp_blk.layer_idx, kG5MtpLayer);
    const auto* mtp_gate =
        find_in(mtp_blk.indexer, TensorComponent::indexer_compressor_wgate);
    ASSERT_NE(mtp_gate, nullptr);
    EXPECT_FALSE(mtp_gate->weight.gguf_type.has_value());
    EXPECT_EQ(mtp_gate->weight.dtype, SafetensorsDtype::BF16);
}

// A recipe whose layer_types disagrees with the checkpoint's head_count_kv
// would load the wrong anatomy into a layer, silently. It must throw, naming
// the first mismatching layer (SURVEY_BOOT.md R3).
TEST(GgufLoader, Glm53MiniGgufRejectsLayerTypesMismatch) {
    // Checkpoint says blk.0 is MLA (head_count_kv[0] != 0); the recipe says KDA.
    Glm53MiniGgufFixture fx("mismatch", {1, 1, 1});
    auto cfg = glm53_mini_config(fx.file());
    ModelConfig mcfg(cfg);
    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    try {
        (void)load_weights(cfg, mcfg, registry);
        FAIL() << "expected a layer_types/head_count_kv mismatch throw";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("layer_types"), std::string::npos) << msg;
        EXPECT_NE(msg.find("layer 0"), std::string::npos) << msg;
    }
}

TEST(GgufLoader, Glm53MiniGgufRejectsHeadCountKvLengthMismatch) {
    // 2 entries for a 2 + 1 (hidden + MTP) block model.
    Glm53MiniGgufFixture fx("len", {0, 1});
    auto cfg = glm53_mini_config(fx.file());
    ModelConfig mcfg(cfg);
    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    try {
        (void)load_weights(cfg, mcfg, registry);
        FAIL() << "expected a head_count_kv length throw";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("head_count_kv"), std::string::npos) << msg;
    }
}

// Exporters that drop the key must still load (warn-and-continue).
TEST(GgufLoader, Glm53MiniGgufLoadsWithoutHeadCountKvMetadata) {
    Glm53MiniGgufFixture fx("nokey", {});
    auto cfg = glm53_mini_config(fx.file());
    ModelConfig mcfg(cfg);
    layerstorm::model::Fp8E4M3 quant;
    LayerRegistry registry(mcfg, cfg, quant);

    LoadedModel model;
    ASSERT_NO_THROW(model = load_weights(cfg, mcfg, registry));
    const auto* gate = find_in(model.layers[kG5SparseLayer].indexer,
                               TensorComponent::indexer_compressor_wgate);
    ASSERT_NE(gate, nullptr);
    EXPECT_EQ(gate->weight.dtype, SafetensorsDtype::BF16);
}

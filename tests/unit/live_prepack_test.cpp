// LIVE PREPACK byte-identity gate (INV-LIVE-PREPACK-IDENTITY).
//
// Level (i) unit proof: for a synthetic on-disk GGUF with per-layer MIXED
// k-quant stacked routed experts (GG-10), the slot bytes LiveGgufExpertSource
// synthesizes must be BYTE-IDENTICAL — content AND zero tail up to the
// aligned stride — to the slots prepack_experts writes (read back through
// PrepackedSource) from the same source. Also covers the O_DIRECT bounce
// path (when the tmp filesystem grants O_DIRECT), multithreaded load_into
// (per-thread scratch), and the type/coverage parity queries.
//
// Level (ii) real-box proof: LiveIdentityRealModel (env-gated,
// LS_LIVE_IDENT_CONFIG=<engine config json with model.weights_path +
// preprocessing.prepacked_dir>) compares live-transformed slots against
// targeted preads of the REAL prepacked set for sampled experts.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "config/config_parser.h"
#include "core/parallel_for.h"
#include "model/model_config.h"
#include "model/quantization/gguf_kquant.h"
#include "model/weight_loader/gguf_reader.h"
#include "model/weight_loader/tensor_id.h"
#include "model/weight_loader/weight_loader.h"
#include "model/weight_pipeline/expert_prepacker.h"
#include "model/weight_pipeline/live_gguf_source.h"
#include "model/weight_pipeline/prepacked_format.h"
#include "model/weight_pipeline/prepacked_source.h"

using namespace layerstorm;
using namespace layerstorm::model;
namespace fs = std::filesystem;

namespace {

// ── Synthetic GGUF file builder ─────────────────────────────────────────────
// Same byte format as gguf_loader_test's GgufBlobBuilder (GGUF v3), written
// to a FILE so both prepack_experts (via mmap'd de-stacked bundles) and
// LiveGgufExpertSource (via O_DIRECT/pread) read the identical source.

constexpr int32_t kKV_UINT32 = 4;

struct PlannedTensor {
    std::string name;
    int32_t ggml_type;
    std::vector<int64_t> dims;  // GGUF order (dims[0] fastest)
};

class GgufFileBuilder {
public:
    explicit GgufFileBuilder(uint64_t alignment = 32) : alignment_(alignment) {}

    void add_tensor(std::string name, int32_t t, std::vector<int64_t> dims) {
        planned_.push_back({std::move(name), t, std::move(dims)});
    }

    static uint64_t tensor_bytes(int32_t ggml_type, int64_t numel) {
        switch (ggml_type) {
            case 0:  return static_cast<uint64_t>(numel) * 4;         // F32
            case 8:  return static_cast<uint64_t>(numel / 32) * 34;   // Q8_0
            case 12: return static_cast<uint64_t>(numel / 256) * 144; // Q4_K
            case 13: return static_cast<uint64_t>(numel / 256) * 176; // Q5_K
            case 14: return static_cast<uint64_t>(numel / 256) * 210; // Q6_K
            case 39: return static_cast<uint64_t>(numel / 32) * 17;   // MXFP4
        }
        return 0;
    }

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

        const char magic[4] = {'G', 'G', 'U', 'F'};
        put(magic, 4);
        put_u32(3);
        put_i64(static_cast<int64_t>(planned_.size()));
        put_i64(1);
        put_str("general.alignment");
        put_i32(kKV_UINT32);
        put_u32(static_cast<uint32_t>(alignment_));

        std::vector<uint64_t> sizes, offsets;
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

        uint64_t blob_start = align_up(hdr.size(), alignment_);
        std::vector<std::byte> out(blob_start, std::byte{0});
        std::memcpy(out.data(), hdr.data(), hdr.size());
        out.resize(blob_start + cursor, std::byte{0});
        for (size_t i = 0; i < planned_.size(); ++i) {
            uint8_t* dst =
                reinterpret_cast<uint8_t*>(out.data()) + blob_start + offsets[i];
            // Per-byte varying pattern (tensor-index seeded LCG) so layout /
            // offset bugs corrupt visibly.
            uint32_t lcg = static_cast<uint32_t>(i * 2654435761u + 12345u);
            for (uint64_t b = 0; b < sizes[i]; ++b) {
                lcg = lcg * 1664525u + 1013904223u;
                dst[b] = static_cast<uint8_t>(lcg >> 24);
            }
        }
        return out;
    }

private:
    static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }
    uint64_t alignment_;
    std::vector<PlannedTensor> planned_;
};

// Map GgufKQuantType → raw ggml_type id for the builder.
int32_t ggml_id(GgufKQuantType t) {
    switch (t) {
        case GgufKQuantType::Q4_K: return 12;
        case GgufKQuantType::Q5_K: return 13;
        case GgufKQuantType::Q6_K: return 14;
        case GgufKQuantType::Q8_0: return 8;
        case GgufKQuantType::MXFP4: return 39;
        default: return -1;
    }
}

struct LayerTypes {
    GgufKQuantType gate, up, down;
};

class LivePrepackTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "layerstorm_live_prepack_test";
        fs::create_directories(tmp_dir_);
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmp_dir_, ec);
    }

    config::Config make_config(int n_layers, int n_experts, int hidden,
                               int inter, int first_moe) {
        config::Config cfg;
        cfg.model.num_hidden_layers = n_layers;
        cfg.model.hidden_size = hidden;
        cfg.model.moe_intermediate_size = inter;
        cfg.model.n_routed_experts = n_experts;
        cfg.model.first_k_dense_replace = first_moe;
        cfg.model.moe_layer_freq = 1;
        cfg.model.weights_path = "/fake/model";
        return cfg;
    }

    /// Write a synthetic GGUF with stacked routed experts at the given
    /// per-layer types; returns the file path.
    fs::path write_gguf(int n_layers, int first_moe, int n_experts, int hidden,
                        int inter, const std::vector<LayerTypes>& types) {
        GgufFileBuilder b;
        b.add_tensor("output_norm.weight", /*F32*/ 0, {hidden});
        for (int l = first_moe, i = 0; l < n_layers; ++l, ++i) {
            const auto& t = types[static_cast<size_t>(i)];
            const std::string p = "blk." + std::to_string(l) + ".";
            b.add_tensor(p + "ffn_gate_exps.weight", ggml_id(t.gate),
                         {hidden, inter, n_experts});
            b.add_tensor(p + "ffn_up_exps.weight", ggml_id(t.up),
                         {hidden, inter, n_experts});
            b.add_tensor(p + "ffn_down_exps.weight", ggml_id(t.down),
                         {inter, hidden, n_experts});
        }
        auto blob = b.build();
        const fs::path path = tmp_dir_ / "synthetic.gguf";
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        ofs.write(reinterpret_cast<const char*>(blob.data()),
                  static_cast<std::streamsize>(blob.size()));
        ofs.close();
        return path;
    }

    /// De-stack the file's routed experts into a LoadedModel (the same spans
    /// destack_expert_tensor would produce), keeping the reader alive inside
    /// the model.
    LoadedModel destack_model(const fs::path& gguf, int n_layers,
                              int n_experts) {
        LoadedModel model;
        model.layers.resize(static_cast<size_t>(n_layers));
        for (int l = 0; l < n_layers; ++l) model.layers[l].layer_idx = l;
        GgufReader r = GgufReader::open(gguf);
        for (const auto& e : r.entries()) {
            auto id = parse_gguf_name(e.name);
            if (!id || id->owner != TensorOwner::routed_expert) continue;
            if (e.dims.size() != 3 || !e.is_kquant()) continue;
            auto data = r.tensor_data(e);
            const int64_t per_expert = gguf::gguf_packed_bytes(
                e.dims[1], e.dims[0], e.kquant_type());
            auto& layer = model.layers[static_cast<size_t>(id->layer_idx)];
            if (static_cast<int>(layer.routed_experts.size()) < n_experts)
                layer.routed_experts.resize(static_cast<size_t>(n_experts));
            for (int ei = 0; ei < n_experts; ++ei) {
                RawTensor raw{
                    .data = data.subspan(
                        static_cast<size_t>(ei) *
                            static_cast<size_t>(per_expert),
                        static_cast<size_t>(per_expert)),
                    .dtype = SafetensorsDtype::U8,
                    .shape = {e.dims[1], e.dims[0]},
                    .gguf_type = e.kquant_type(),
                };
                WeightBundle bundle;
                bundle.id = TensorId{id->component, TensorRole::weight,
                                     TensorOwner::routed_expert, id->layer_idx,
                                     ei};
                bundle.weight = std::move(raw);
                layer.routed_experts[static_cast<size_t>(ei)].push_back(
                    std::move(bundle));
            }
        }
        model.gguf_shards.push_back(std::move(r));
        return model;
    }

    fs::path tmp_dir_;
};

}  // namespace

// ── Level (i): synthetic mixed-type byte identity ───────────────────────────

TEST_F(LivePrepackTest, MixedKQuantByteIdentityVsPrepackedSet) {
    constexpr int kLayers = 5;
    constexpr int kFirstMoe = 2;
    constexpr int kExperts = 4;
    constexpr int kHidden = 512;
    constexpr int kInter = 256;
    const std::vector<LayerTypes> layer_types{
        {GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q5_K},
        {GgufKQuantType::Q5_K, GgufKQuantType::Q6_K, GgufKQuantType::Q8_0},
        {GgufKQuantType::MXFP4, GgufKQuantType::MXFP4, GgufKQuantType::MXFP4},
    };

    const fs::path gguf = write_gguf(kLayers, kFirstMoe, kExperts, kHidden,
                                     kInter, layer_types);
    auto cfg = make_config(kLayers, kExperts, kHidden, kInter, kFirstMoe);
    cfg.model.weights_path = gguf.string();
    ModelConfig model_cfg(cfg);

    // Global slot sizing = per-projection MAX (GG-9): gate Q5_K, up Q6_K,
    // down Q8_0 for this mix (MXFP4 is the smallest).
    GgufQuantInterface quant = make_gguf_quant(
        GgufKQuantType::Q5_K, GgufKQuantType::Q6_K, GgufKQuantType::Q8_0);
    const int64_t stride =
        prepacked::aligned_slot_stride(quant.bytes_per_expert(
            ExpertShape{kHidden, kInter}));

    // Reference arm: the offline tool (prepack_experts) → PrepackedSource.
    auto model = destack_model(gguf, kLayers, kExperts);
    const fs::path outdir = tmp_dir_ / "prepacked";
    auto result = prepack_experts(model, model_cfg, quant, cfg, outdir);
    ASSERT_TRUE(result.error.empty()) << result.error;
    PrepackedSource psrc(outdir, quant);
    ASSERT_EQ(psrc.slot_size_bytes(), stride);

    // Live arm: transform straight from the GGUF (O_DIRECT attempted; the
    // ctor falls back per-file to buffered pread when refused).
    std::vector<GgufReader> shards;
    shards.push_back(GgufReader::open(gguf));
    LiveGgufExpertSource live(shards, model_cfg, quant, kExperts,
                              /*o_direct=*/true);
    ASSERT_EQ(live.slot_size_bytes(), stride);
    if (!live.shard_is_o_direct(0)) {
        std::printf("live_prepack_test: tmp fs refused O_DIRECT — buffered "
                    "pread path exercised instead\n");
    }

    // Byte identity for EVERY (layer, expert), full stride (content + pad).
    std::vector<std::byte> buf(static_cast<size_t>(stride),
                               std::byte{0xEE});  // poison: pad must be zeroed
    for (int l = 0; l < kLayers; ++l) {
        for (int e = 0; e < kExperts; ++e) {
            memory::ExpertKey key{static_cast<uint32_t>(l),
                                  static_cast<uint16_t>(e)};
            ASSERT_EQ(live.has(key), psrc.has(key)) << l << "," << e;
            if (!live.has(key)) continue;
            std::fill(buf.begin(), buf.end(), std::byte{0xEE});
            ASSERT_TRUE(live.load_into(key, buf.data())) << l << "," << e;
            const void* ref = psrc.resolve(key);
            ASSERT_NE(ref, nullptr) << l << "," << e;
            EXPECT_EQ(std::memcmp(buf.data(), ref,
                                  static_cast<size_t>(stride)), 0)
                << "slot bytes differ at layer " << l << " expert " << e;
        }
    }

    // GG-10 per-layer type parity with the manifest.
    for (int l = 0; l < kLayers; ++l) {
        auto lt = live.gguf_types_for_layer(l);
        auto pt = psrc.gguf_types_for_layer(l);
        ASSERT_EQ(lt.has_value(), pt.has_value()) << l;
        if (lt) {
            EXPECT_EQ(lt->gate, pt->gate);
            EXPECT_EQ(lt->up, pt->up);
            EXPECT_EQ(lt->down, pt->down);
        }
    }
}

TEST_F(LivePrepackTest, BufferedModeIdenticalToODirectMode) {
    constexpr int kLayers = 3;
    constexpr int kFirstMoe = 1;
    constexpr int kExperts = 3;
    constexpr int kHidden = 512;
    constexpr int kInter = 256;
    const std::vector<LayerTypes> layer_types{
        {GgufKQuantType::Q4_K, GgufKQuantType::Q5_K, GgufKQuantType::Q6_K},
        {GgufKQuantType::Q8_0, GgufKQuantType::Q8_0, GgufKQuantType::Q8_0},
    };
    const fs::path gguf = write_gguf(kLayers, kFirstMoe, kExperts, kHidden,
                                     kInter, layer_types);
    auto cfg = make_config(kLayers, kExperts, kHidden, kInter, kFirstMoe);
    ModelConfig model_cfg(cfg);
    GgufQuantInterface quant = make_gguf_quant(
        GgufKQuantType::Q8_0, GgufKQuantType::Q8_0, GgufKQuantType::Q8_0);
    const int64_t stride = prepacked::aligned_slot_stride(
        quant.bytes_per_expert(ExpertShape{kHidden, kInter}));

    std::vector<GgufReader> shards;
    shards.push_back(GgufReader::open(gguf));
    LiveGgufExpertSource direct(shards, model_cfg, quant, kExperts, true);
    LiveGgufExpertSource buffered(shards, model_cfg, quant, kExperts, false);

    std::vector<std::byte> a(static_cast<size_t>(stride));
    std::vector<std::byte> b(static_cast<size_t>(stride));
    for (int l = kFirstMoe; l < kLayers; ++l) {
        for (int e = 0; e < kExperts; ++e) {
            memory::ExpertKey key{static_cast<uint32_t>(l),
                                  static_cast<uint16_t>(e)};
            ASSERT_TRUE(direct.load_into(key, a.data()));
            ASSERT_TRUE(buffered.load_into(key, b.data()));
            EXPECT_EQ(std::memcmp(a.data(), b.data(),
                                  static_cast<size_t>(stride)), 0)
                << l << "," << e;
        }
    }
}

TEST_F(LivePrepackTest, MultithreadedLoadIntoIsRaceFreeAndIdentical) {
    constexpr int kLayers = 4;
    constexpr int kFirstMoe = 0;
    constexpr int kExperts = 8;
    constexpr int kHidden = 512;
    constexpr int kInter = 256;
    const std::vector<LayerTypes> layer_types{
        {GgufKQuantType::Q4_K, GgufKQuantType::Q4_K, GgufKQuantType::Q4_K},
        {GgufKQuantType::Q5_K, GgufKQuantType::Q5_K, GgufKQuantType::Q5_K},
        {GgufKQuantType::Q6_K, GgufKQuantType::Q6_K, GgufKQuantType::Q6_K},
        {GgufKQuantType::Q8_0, GgufKQuantType::Q8_0, GgufKQuantType::Q8_0},
    };
    const fs::path gguf = write_gguf(kLayers, kFirstMoe, kExperts, kHidden,
                                     kInter, layer_types);
    auto cfg = make_config(kLayers, kExperts, kHidden, kInter, kFirstMoe);
    ModelConfig model_cfg(cfg);
    GgufQuantInterface quant = make_gguf_quant(
        GgufKQuantType::Q8_0, GgufKQuantType::Q8_0, GgufKQuantType::Q8_0);
    const size_t stride = static_cast<size_t>(prepacked::aligned_slot_stride(
        quant.bytes_per_expert(ExpertShape{kHidden, kInter})));

    std::vector<GgufReader> shards;
    shards.push_back(GgufReader::open(gguf));
    LiveGgufExpertSource live(shards, model_cfg, quant, kExperts, true);

    // Single-threaded reference.
    const size_t n = static_cast<size_t>(kLayers) * kExperts;
    std::vector<std::vector<std::byte>> ref(n);
    for (size_t i = 0; i < n; ++i) {
        memory::ExpertKey key{static_cast<uint32_t>(i / kExperts),
                              static_cast<uint16_t>(i % kExperts)};
        ref[i].resize(stride);
        ASSERT_TRUE(live.load_into(key, ref[i].data()));
    }
    // Parallel arm (per-thread scratch buffers must not cross-contaminate).
    std::vector<std::vector<std::byte>> par(n);
    core::parallel_for(n, [&](size_t i) {
        memory::ExpertKey key{static_cast<uint32_t>(i / kExperts),
                              static_cast<uint16_t>(i % kExperts)};
        par[i].resize(stride);
        ASSERT_TRUE(live.load_into(key, par[i].data()));
    }, 8);
    for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(std::memcmp(ref[i].data(), par[i].data(), stride), 0) << i;
}

// ── Level (ii): real-box sampled identity (env-gated) ───────────────────────
// LS_LIVE_IDENT_CONFIG=<config.json> — must carry model.weights_path (GGUF)
// and preprocessing.prepacked_dir (a prepack of the SAME GGUF). Compares live
// slots against targeted preads of the prepacked files for sampled keys.

TEST(LivePrepackRealModel, SampledSlotIdentity) {
    const char* cfg_path = std::getenv("LS_LIVE_IDENT_CONFIG");
    if (!cfg_path || !*cfg_path)
        GTEST_SKIP() << "LS_LIVE_IDENT_CONFIG unset";

    auto cfg = config::parse_config(std::string(cfg_path));
    ASSERT_FALSE(cfg.preprocessing.prepacked_dir.empty());
    ModelConfig model_cfg(cfg);
    auto types = gguf_expert_types_from_path(cfg.model.weights_path);
    GgufQuantInterface quant =
        make_gguf_quant(types.gate, types.up, types.down);

    // Prepacked arm: direct pread (no mmap page-cache pollution of the box).
    PrepackedSource psrc(cfg.preprocessing.prepacked_dir, quant,
                         /*direct_io=*/true, /*o_direct=*/false);
    // Live arm over the real GGUF shards.
    std::vector<GgufReader> shards;
    for (const auto& f : resolve_gguf_files(cfg.model.weights_path))
        shards.push_back(GgufReader::open(f));
    LiveGgufExpertSource live(shards, model_cfg, quant,
                              cfg.model.n_routed_experts, /*o_direct=*/true);
    const int64_t stride = psrc.slot_size_bytes();
    ASSERT_EQ(live.slot_size_bytes(), stride);

    const auto& moe = model_cfg.moe_layer_indices();
    ASSERT_FALSE(moe.empty());
    std::vector<int> sample_layers;
    sample_layers.push_back(moe.front());
    sample_layers.push_back(moe[moe.size() / 2]);
    sample_layers.push_back(moe.back());
    std::vector<int> sample_experts{0, 1, cfg.model.n_routed_experts / 2,
                                    cfg.model.n_routed_experts - 1};

    std::vector<std::byte> lbuf(static_cast<size_t>(stride), std::byte{0xEE});
    std::vector<std::byte> pbuf(static_cast<size_t>(stride), std::byte{0x11});
    int compared = 0;
    for (int l : sample_layers) {
        for (int e : sample_experts) {
            memory::ExpertKey key{static_cast<uint32_t>(l),
                                  static_cast<uint16_t>(e)};
            ASSERT_TRUE(psrc.has(key)) << l << "," << e;
            ASSERT_TRUE(live.has(key)) << l << "," << e;
            ASSERT_TRUE(psrc.pread_into(key, pbuf.data())) << l << "," << e;
            ASSERT_TRUE(live.load_into(key, lbuf.data())) << l << "," << e;
            EXPECT_EQ(std::memcmp(lbuf.data(), pbuf.data(),
                                  static_cast<size_t>(stride)), 0)
                << "REAL slot bytes differ at layer " << l << " expert " << e;
            ++compared;
        }
    }
    std::printf("LivePrepackRealModel: %d sampled slots byte-identical "
                "(stride %ld)\n", compared, static_cast<long>(stride));
}

// EPM-1 (Phase 29) coverage gate: training-data dump hooks.
//
// Tiers:
//   0. Host-only: FP32->FP16 conversion (vs the compiler's _Float16),
//      epm_dump_dir config/env resolution, and the hooks-OFF contracts
//      (no scratch growth, no files, dump calls inert).
//   1. Block-record round trip through the REAL DsparkRuntime capture path
//      (CUDA GPU required; mirrors DsparkForward.MatchesCpuReferenceSmallDims):
//      a tiny synthetic speculators-v0.5 checkpoint runs capture_aux ->
//      run_step -> run_markov_head -> run_confidence_head ->
//      epm_write_block_record; the dumped [γ, L, H] hiddens are compared
//      element-wise against a CPU float reference of the PER-LAYER post-MLP
//      residual stream (BF16-rounded at every write, as the sibling test),
//      the dumped ids/c_k bitwise against the runtime's own device outputs,
//      and the forward is proven bit-identical dump-ON vs dump-OFF.
//   2. Routing-record round trip against the REAL top-K gating kernel
//      (CUDA GPU required): launch_topk_gating on synthetic router logits,
//      capture through EpmRoutingDumper exactly as the dispatcher does
//      (D2H + host FP16 conversion), and cross-check the dumped top-8 ids
//      bitwise against the kernel's topk_indices buffer — the single
//      source both the F-3 routing export (which the orchestrator turns
//      into the FETCH_AND_RUN expert list, entry-for-entry) and the
//      dispatch_moe precomputed-gating consumer read — plus a CPU
//      reference top-8 selection from the raw logits (label sanity).
//
// Tiny-checkpoint + CPU-reference machinery adapted from
// tests/unit/dspark_forward_test.cpp (same dims, same BF16 rounding model).

#include "speculation/epm_dump.h"
#include "speculation/dspark_runtime.h"

#include "compute/cuda_sm120_device_backend.h"
#include "config/config_parser.h"
#include "core/device_backend.h"
#include "model/weight_loader/dspark_loader.h"
#include "sm120/gating/topk_gating.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace lc = layerstorm::config;
namespace lm = layerstorm::model;
namespace lspec = layerstorm::speculation;
namespace lcomp = layerstorm::compute;

namespace {

namespace fs = std::filesystem;

bool has_cuda_gpu() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// ── BF16 helpers (as dspark_forward_test.cpp) ────────────────────────────────

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding) >> 16);
}

float bf2f(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float bfr(float f) { return bf2f(f2bf(f)); }

// ── Tiny synthetic checkpoint (speculators v0.5, dims as the DSP-3 test) ─────

struct TinyDims {
    int64_t H = 8, V = 32, r = 4, D = 4, I = 16;
    int layers = 2, heads = 2;
    int64_t Q() const { return heads * D; }
};

struct NamedTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<uint16_t> data;  // BF16
};

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    return n;
}

std::vector<NamedTensor> tiny_tensors(uint32_t seed) {
    TinyDims d;
    std::mt19937 rng(seed);
    auto fill = [&](const std::vector<int64_t>& shape, float lo, float hi) {
        std::uniform_real_distribution<float> u(lo, hi);
        std::vector<uint16_t> v(static_cast<size_t>(numel(shape)));
        for (auto& x : v) x = f2bf(u(rng));
        return v;
    };
    auto w = [&](std::string name, std::vector<int64_t> shape, float lo,
                 float hi) {
        return NamedTensor{std::move(name), shape, fill(shape, lo, hi)};
    };

    std::vector<NamedTensor> t;
    t.push_back(w("embed_tokens.weight", {d.V, d.H}, -0.5f, 0.5f));
    t.push_back(w("lm_head.weight", {d.V, d.H}, -0.3f, 0.3f));
    t.push_back(w("fc.weight", {d.H, 2 * d.H}, -0.25f, 0.25f));
    t.push_back(w("hidden_norm.weight", {d.H}, 0.8f, 1.2f));
    t.push_back(w("norm.weight", {d.H}, 0.8f, 1.2f));
    t.push_back(w("markov_head.markov_w1.weight", {d.V, d.r}, -0.3f, 0.3f));
    t.push_back(w("markov_head.markov_w2.weight", {d.V, d.r}, -0.3f, 0.3f));
    t.push_back(w("confidence_head.proj.weight", {1, d.H + d.r}, -0.3f, 0.3f));
    t.push_back(w("confidence_head.proj.bias", {1}, -0.1f, 0.1f));
    for (int l = 0; l < d.layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        t.push_back(w(p + "self_attn.q_proj.weight", {d.Q(), d.H}, -0.3f, 0.3f));
        t.push_back(w(p + "self_attn.k_proj.weight", {d.Q(), d.H}, -0.3f, 0.3f));
        t.push_back(w(p + "self_attn.v_proj.weight", {d.Q(), d.H}, -0.3f, 0.3f));
        t.push_back(w(p + "self_attn.o_proj.weight", {d.H, d.Q()}, -0.3f, 0.3f));
        t.push_back(w(p + "self_attn.q_norm.weight", {d.D}, 0.8f, 1.2f));
        t.push_back(w(p + "self_attn.k_norm.weight", {d.D}, 0.8f, 1.2f));
        t.push_back(w(p + "input_layernorm.weight", {d.H}, 0.8f, 1.2f));
        t.push_back(w(p + "post_attention_layernorm.weight", {d.H}, 0.8f, 1.2f));
        t.push_back(w(p + "mlp.gate_proj.weight", {d.I, d.H}, -0.3f, 0.3f));
        t.push_back(w(p + "mlp.up_proj.weight", {d.I, d.H}, -0.3f, 0.3f));
        t.push_back(w(p + "mlp.down_proj.weight", {d.H, d.I}, -0.25f, 0.25f));
    }
    return t;
}

fs::path make_tiny_checkpoint(const std::string& tag,
                              const std::vector<NamedTensor>& tensors) {
    TinyDims d;
    nlohmann::json cfg{
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
          {"num_key_value_heads", d.heads},
          {"head_dim", d.D},
          {"intermediate_size", d.I},
          {"rms_norm_eps", 1e-5},
          {"vocab_size", d.V},
          {"rope_parameters", {{"rope_theta", 8000000.0}}}}},
    };

    fs::path dir = fs::path(::testing::TempDir()) / ("epm_dump_" + tag);
    fs::create_directories(dir);
    { std::ofstream(dir / "config.json") << cfg.dump(2); }

    nlohmann::json header;
    int64_t offset = 0;
    for (const auto& t : tensors) {
        const int64_t bytes = numel(t.shape) * 2;
        header[t.name] = {{"dtype", "BF16"},
                          {"shape", t.shape},
                          {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    const std::string hj = header.dump();
    const uint64_t hlen = hj.size();
    std::ofstream f(dir / "model.safetensors", std::ios::binary);
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(hj.data(), static_cast<std::streamsize>(hj.size()));
    for (const auto& t : tensors)
        f.write(reinterpret_cast<const char*>(t.data.data()),
                static_cast<std::streamsize>(t.data.size() * 2));
    return dir;
}

lc::Config runtime_config(const fs::path& ckpt_dir) {
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
         {{"gpus", {{{"id", 0}, {"type", "rtx5080"}, {"vram_gb", 16}}}},
          {"system_ram_gb", 256}}},
    };
    auto cfg = lc::parse_config(j);
    cfg.speculation.method = lc::SpeculationMethodType::dspark;
    auto& dc = cfg.speculation.dspark;
    dc.checkpoint_path = ckpt_dir.string();
    dc.draft_gpus = {0};
    dc.block_size = 8;
    dc.speculative_tokens = 7;
    dc.aux_hidden_state_layer_ids = {0, 1};
    dc.mask_token_id = 5;
    dc.max_anchors = 16;
    dc.draft_vocab_size = 32;
    dc.markov_rank = 4;
    dc.draft_context_capacity_tokens = 256;
    dc.aux_capture_max_rows = 16;
    dc.confidence_enabled = true;
    return cfg;
}

struct Harness {
    std::unique_ptr<layerstorm::compute::DeviceBackend> backend;
    void* stream = nullptr;
    std::unique_ptr<lspec::DsparkRuntime> rt;

    explicit Harness(const lc::Config& cfg) {
        backend = layerstorm::compute::make_cuda_sm120_device_backend(
            lc::GpuRef{.position = 0, .id = 0,
                       .type = lc::GpuType::rtx5080});
        backend->set_device();
        stream = backend->create_stream();
        std::vector<lspec::DsparkRuntime::Rank> ranks;
        ranks.push_back({backend.get(), stream});
        rt = lspec::DsparkRuntime::create(cfg, std::move(ranks));
    }
    ~Harness() {
        rt.reset();
        if (backend && stream) backend->destroy_stream(stream);
    }
};

// ── CPU float reference with PER-LAYER residual capture ──────────────────────
// Mirrors DsparkRuntime ingest_context + run_step exactly (BF16 rounding at
// every buffer write, FP32 dot products) — as dspark_forward_test.cpp's
// cpu_reference, extended to record x (the residual stream) after each
// layer's MLP add: exactly what the EPM-1 tap copies into the dump staging.

using Vec = std::vector<float>;

Vec bf16_of(const lm::RawTensor& t) {
    const auto* p = reinterpret_cast<const uint16_t*>(t.data.data());
    Vec v(static_cast<size_t>(t.numel()));
    for (size_t i = 0; i < v.size(); ++i) v[i] = bf2f(p[i]);
    return v;
}

Vec gemm_nt(const Vec& A, const Vec& W, int M, int N, int K) {
    Vec C(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[static_cast<size_t>(m) * K + k] *
                       W[static_cast<size_t>(n) * K + k];
            C[static_cast<size_t>(m) * N + n] = bfr(acc);
        }
    return C;
}

Vec rmsnorm(const Vec& x, const Vec& w, int rows, int dim, float eps) {
    Vec out(x.size());
    for (int r = 0; r < rows; ++r) {
        float ss = 0.0f;
        for (int d = 0; d < dim; ++d) {
            const float v = x[static_cast<size_t>(r) * dim + d];
            ss += v * v;
        }
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
        for (int d = 0; d < dim; ++d)
            out[static_cast<size_t>(r) * dim + d] =
                bfr(x[static_cast<size_t>(r) * dim + d] * inv * w[d]);
    }
    return out;
}

void rope(Vec& x, int rows, int heads, int D, int base, float theta) {
    const int half = D / 2;
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < heads; ++h) {
            float* row = x.data() + (static_cast<size_t>(t) * heads + h) * D;
            for (int i = 0; i < half; ++i) {
                const float inv_freq = std::pow(
                    theta, -2.0f * static_cast<float>(i) /
                               static_cast<float>(D));
                const float ang = static_cast<float>(base + t) * inv_freq;
                const float c = std::cos(ang), s = std::sin(ang);
                const float x0 = row[i], x1 = row[i + half];
                row[i] = bfr(x0 * c - x1 * s);
                row[i + half] = bfr(x0 * s + x1 * c);
            }
        }
}

/// Per-layer post-MLP residuals [L][nq*H] of one run_step.
std::vector<Vec> cpu_layer_residuals(const lm::DsparkDraftWeights& w,
                                     const Vec& aux_concat, int ctx_rows,
                                     int anchor_token, int anchor_pos,
                                     int nq) {
    const auto& ck = w.ckpt;
    const int H = static_cast<int>(ck.hidden_size);
    const int heads = ck.num_attention_heads;
    const int D = static_cast<int>(ck.head_dim);
    const int QD = heads * D;
    const int I = static_cast<int>(ck.intermediate_size);
    const int L = ck.num_hidden_layers;
    const int n_aux = static_cast<int>(ck.aux_hidden_state_layer_ids.size());
    const float eps = static_cast<float>(ck.rms_norm_eps);
    const float theta = static_cast<float>(ck.rope_theta);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    const Vec fc = bf16_of(w.fc);
    const Vec hidden_norm_w = bf16_of(w.hidden_norm);
    Vec ctx_hidden = gemm_nt(aux_concat, fc, ctx_rows, H, n_aux * H);
    Vec ctx_normed = rmsnorm(ctx_hidden, hidden_norm_w, ctx_rows, H, eps);

    std::vector<Vec> ctx_k(static_cast<size_t>(L)),
        ctx_v(static_cast<size_t>(L));
    for (int l = 0; l < L; ++l) {
        const auto& lw = w.layers[static_cast<size_t>(l)];
        Vec k = gemm_nt(ctx_normed, bf16_of(lw.k_proj), ctx_rows, QD, H);
        k = rmsnorm(k, bf16_of(lw.k_norm), ctx_rows * heads, D, eps);
        rope(k, ctx_rows, heads, D, /*base=*/0, theta);
        ctx_k[static_cast<size_t>(l)] = std::move(k);
        ctx_v[static_cast<size_t>(l)] =
            gemm_nt(ctx_normed, bf16_of(lw.v_proj), ctx_rows, QD, H);
    }

    const Vec embed = bf16_of(w.embed_tokens);
    Vec x(static_cast<size_t>(nq) * H);
    for (int t = 0; t < nq; ++t) {
        const int id = (t == 0) ? anchor_token
                                : static_cast<int>(ck.mask_token_id);
        for (int d = 0; d < H; ++d)
            x[static_cast<size_t>(t) * H + d] =
                embed[static_cast<size_t>(id) * H + d];
    }

    Vec normed = rmsnorm(x, bf16_of(w.layers[0].input_layernorm), nq, H, eps);

    std::vector<Vec> residuals;
    for (int l = 0; l < L; ++l) {
        const auto& lw = w.layers[static_cast<size_t>(l)];
        Vec q = gemm_nt(normed, bf16_of(lw.q_proj), nq, QD, H);
        q = rmsnorm(q, bf16_of(lw.q_norm), nq * heads, D, eps);
        rope(q, nq, heads, D, anchor_pos, theta);
        Vec k = gemm_nt(normed, bf16_of(lw.k_proj), nq, QD, H);
        k = rmsnorm(k, bf16_of(lw.k_norm), nq * heads, D, eps);
        rope(k, nq, heads, D, anchor_pos, theta);
        Vec v = gemm_nt(normed, bf16_of(lw.v_proj), nq, QD, H);

        const int cu = anchor_pos;
        Vec attn(static_cast<size_t>(nq) * QD);
        for (int t = 0; t < nq; ++t)
            for (int h = 0; h < heads; ++h) {
                const int total = cu + nq;
                std::vector<float> sc(static_cast<size_t>(total));
                float mx = -INFINITY;
                for (int jj = 0; jj < total; ++jj) {
                    const float* kr =
                        (jj < cu)
                            ? &ctx_k[static_cast<size_t>(l)]
                                    [(static_cast<size_t>(jj) * heads + h) *
                                     D]
                            : &k[(static_cast<size_t>(jj - cu) * heads + h) *
                                 D];
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d)
                        dot += q[(static_cast<size_t>(t) * heads + h) * D + d]
                               * kr[d];
                    sc[static_cast<size_t>(jj)] = dot * scale;
                    mx = std::max(mx, sc[static_cast<size_t>(jj)]);
                }
                float lsum = 0.0f;
                std::vector<float> acc(static_cast<size_t>(D), 0.0f);
                for (int jj = 0; jj < total; ++jj) {
                    const float wj =
                        std::exp(sc[static_cast<size_t>(jj)] - mx);
                    lsum += wj;
                    const float* vr =
                        (jj < cu)
                            ? &ctx_v[static_cast<size_t>(l)]
                                    [(static_cast<size_t>(jj) * heads + h) *
                                     D]
                            : &v[(static_cast<size_t>(jj - cu) * heads + h) *
                                 D];
                    for (int d = 0; d < D; ++d)
                        acc[static_cast<size_t>(d)] += wj * vr[d];
                }
                for (int d = 0; d < D; ++d)
                    attn[(static_cast<size_t>(t) * heads + h) * D + d] =
                        bfr(acc[static_cast<size_t>(d)] / lsum);
            }

        Vec o = gemm_nt(attn, bf16_of(lw.o_proj), nq, H, QD);
        for (size_t i = 0; i < x.size(); ++i) x[i] = bfr(x[i] + o[i]);
        normed = rmsnorm(x, bf16_of(lw.post_attention_layernorm), nq, H, eps);

        Vec gate = gemm_nt(normed, bf16_of(lw.gate_proj), nq, I, H);
        Vec up = gemm_nt(normed, bf16_of(lw.up_proj), nq, I, H);
        Vec act(gate.size());
        for (size_t i = 0; i < gate.size(); ++i)
            act[i] = bfr(gate[i] / (1.0f + std::exp(-gate[i])) * up[i]);
        Vec mlp = gemm_nt(act, bf16_of(lw.down_proj), nq, H, I);

        for (size_t i = 0; i < x.size(); ++i) x[i] = bfr(x[i] + mlp[i]);
        residuals.push_back(x);  // <-- what the EPM-1 tap dumps for layer l

        const bool last = (l + 1 == L);
        normed = rmsnorm(
            x,
            bf16_of(last ? w.final_norm
                         : w.layers[static_cast<size_t>(l) + 1]
                               .input_layernorm),
            nq, H, eps);
    }
    return residuals;
}

// ── Raw record parsers (mirror src/speculation/epm_dump.h layouts) ───────────

struct ParsedBlock {
    uint64_t seq_id;
    uint32_t block_idx, anchor_pos, anchor_token, gamma, n_layers, hidden,
        has_conf;
    std::vector<int32_t> ids;
    std::vector<float> conf;
    std::vector<uint16_t> hiddens;  // [gamma, n_layers, hidden]
};

std::vector<ParsedBlock> parse_blocks(const fs::path& file) {
    std::ifstream f(file, std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    std::vector<ParsedBlock> out;
    size_t off = 0;
    auto rd = [&](void* dst, size_t n) {
        EXPECT_LE(off + n, data.size()) << "truncated EPMB stream";
        std::memcpy(dst, data.data() + off, n);
        off += n;
    };
    while (off < data.size()) {
        ParsedBlock b{};
        uint32_t magic = 0, ver = 0;
        rd(&magic, 4);
        rd(&ver, 4);
        EXPECT_EQ(magic, 0x424D5045u);
        EXPECT_EQ(ver, 1u);
        rd(&b.seq_id, 8);
        rd(&b.block_idx, 4);
        rd(&b.anchor_pos, 4);
        rd(&b.anchor_token, 4);
        rd(&b.gamma, 4);
        rd(&b.n_layers, 4);
        rd(&b.hidden, 4);
        rd(&b.has_conf, 4);
        b.ids.resize(b.gamma);
        rd(b.ids.data(), b.gamma * 4);
        if (b.has_conf) {
            b.conf.resize(b.gamma);
            rd(b.conf.data(), b.gamma * 4);
        }
        b.hiddens.resize(static_cast<size_t>(b.gamma) * b.n_layers *
                         b.hidden);
        rd(b.hiddens.data(), b.hiddens.size() * 2);
        out.push_back(std::move(b));
    }
    EXPECT_EQ(off, data.size());
    return out;
}

struct ParsedRouteRow {
    uint32_t layer;
    std::vector<uint16_t> logits_f16;
    std::vector<int32_t> ids;
    std::vector<float> w;
};

struct ParsedRoute {
    uint64_t seq_id;
    uint32_t token_pos, n_experts, topk;
    std::vector<ParsedRouteRow> rows;
};

std::vector<ParsedRoute> parse_routes(const fs::path& file) {
    std::ifstream f(file, std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    std::vector<ParsedRoute> out;
    size_t off = 0;
    auto rd = [&](void* dst, size_t n) {
        EXPECT_LE(off + n, data.size()) << "truncated EPMR stream";
        std::memcpy(dst, data.data() + off, n);
        off += n;
    };
    while (off < data.size()) {
        ParsedRoute r{};
        uint32_t magic = 0, ver = 0, n_rows = 0;
        rd(&magic, 4);
        rd(&ver, 4);
        EXPECT_EQ(magic, 0x524D5045u);
        EXPECT_EQ(ver, 1u);
        rd(&r.seq_id, 8);
        rd(&r.token_pos, 4);
        rd(&n_rows, 4);
        rd(&r.n_experts, 4);
        rd(&r.topk, 4);
        for (uint32_t i = 0; i < n_rows; ++i) {
            ParsedRouteRow row{};
            rd(&row.layer, 4);
            row.logits_f16.resize(r.n_experts);
            rd(row.logits_f16.data(), r.n_experts * 2);
            row.ids.resize(r.topk);
            rd(row.ids.data(), r.topk * 4);
            row.w.resize(r.topk);
            rd(row.w.data(), r.topk * 4);
            r.rows.push_back(std::move(row));
        }
        out.push_back(std::move(r));
    }
    EXPECT_EQ(off, data.size());
    return out;
}

fs::path fresh_dir(const std::string& tag) {
    fs::path dir = fs::path(::testing::TempDir()) / ("epm_out_" + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════════════
// Tier 0: host-only — FP16 conversion + gating resolution + hooks-off
// ═════════════════════════════════════════════════════════════════════════════

TEST(EpmDump, F32ToF16MatchesCompilerFloat16) {
#ifdef __FLT16_MANT_DIG__
    // Exact RNE parity with the compiler's IEEE binary16 over randoms +
    // edge cases (the dumped router logits must be faithful, decision (A)).
    std::vector<float> cases = {0.0f, -0.0f, 1.0f, -1.0f, 65504.0f,
                                -65504.0f, 65520.0f, 1e-8f, -1e-8f,
                                5.9604645e-08f, 6.097555e-05f,
                                INFINITY, -INFINITY};
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-30.0f, 30.0f);
    std::uniform_real_distribution<float> big(-1e5f, 1e5f);
    std::uniform_real_distribution<float> tiny(-1e-4f, 1e-4f);
    for (int i = 0; i < 20000; ++i) cases.push_back(u(rng));
    for (int i = 0; i < 2000; ++i) cases.push_back(big(rng));
    for (int i = 0; i < 2000; ++i) cases.push_back(tiny(rng));
    for (float f : cases) {
        const uint16_t got = lspec::epm_f32_to_f16(f);
        const _Float16 h = static_cast<_Float16>(f);
        uint16_t want;
        std::memcpy(&want, &h, 2);
        EXPECT_EQ(got, want) << "f=" << f;
    }
    // NaN maps to a NaN pattern (payload unspecified).
    const uint16_t n = lspec::epm_f32_to_f16(NAN);
    EXPECT_EQ(n & 0x7C00u, 0x7C00u);
    EXPECT_NE(n & 0x03FFu, 0u);
#else
    GTEST_SKIP() << "_Float16 unsupported by this compiler";
#endif
}

TEST(EpmDump, DumpDirResolutionEnvOverridesConfig) {
    lc::DsparkConfig dc;
    unsetenv("LS_EPM_DUMP");
    EXPECT_EQ(lspec::epm_dump_dir(dc), "");  // OFF by default
    dc.epm_dump_dir = "/tmp/epm_cfg";
    EXPECT_EQ(lspec::epm_dump_dir(dc), "/tmp/epm_cfg");
    setenv("LS_EPM_DUMP", "/tmp/epm_env", 1);
    EXPECT_EQ(lspec::epm_dump_dir(dc), "/tmp/epm_env");
    setenv("LS_EPM_DUMP", "0", 1);  // forced OFF beats config
    EXPECT_EQ(lspec::epm_dump_dir(dc), "");
    unsetenv("LS_EPM_DUMP");
    EXPECT_EQ(lspec::epm_dump_dir(dc), "/tmp/epm_cfg");
    dc.epm_dump_dir = "";
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 1: block-record round trip through the real DsparkRuntime capture path
// ═════════════════════════════════════════════════════════════════════════════

TEST(EpmDump, BlockRecordRoundTripMatchesForwardHiddens) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    unsetenv("LS_EPM_DUMP");

    const auto tensors = tiny_tensors(/*seed=*/1234);
    const auto ckpt = make_tiny_checkpoint("blockrt", tensors);
    auto host = lm::load_dspark_draft(ckpt);

    // ── Hooks-OFF baseline: no staging in the budget, no dump methods. ──
    auto cfg_off = runtime_config(ckpt);
    const int64_t scratch_off =
        lspec::dspark_runtime_scratch_bytes(cfg_off, host.ckpt);

    // ── Dump-ON config. ──
    const fs::path dump_dir = fresh_dir("blockrt");
    auto cfg_on = runtime_config(ckpt);
    cfg_on.speculation.dspark.epm_dump_dir = dump_dir.string();
    const int64_t scratch_on =
        lspec::dspark_runtime_scratch_bytes(cfg_on, host.ckpt);
    // Budget grows by exactly the [block_size, L, H] BF16 staging.
    EXPECT_EQ(scratch_on - scratch_off, int64_t{8} * 2 * 8 * 2);

    // Shared synthetic context rows (identical for both runtimes).
    const TinyDims d;
    const int ctx_rows = 3, H = static_cast<int>(d.H);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> u(-0.8f, 0.8f);
    Vec slot_rows[2];
    for (auto& s : slot_rows) {
        s.resize(static_cast<size_t>(ctx_rows) * H);
        for (auto& x : s) x = bfr(u(rng));
    }
    std::vector<uint16_t> bf(static_cast<size_t>(ctx_rows) * H);

    const uint64_t seq = 21;
    const int anchor_token = 3, anchor_pos = 3, nq = 4;
    const int V = static_cast<int>(d.V);

    auto run_block = [&](Harness& h, bool dump) -> Vec {
        void* src[2];
        for (int s = 0; s < 2; ++s) {
            for (size_t i = 0; i < bf.size(); ++i)
                bf[i] = f2bf(slot_rows[s][i]);
            src[s] = h.backend->device_alloc(bf.size() * 2);
            EXPECT_NE(src[s], nullptr);
            h.backend->memcpy_h2d(src[s], bf.data(), bf.size() * 2);
        }
        h.rt->capture_aux(0, src[0], ctx_rows, seq, 0, *h.backend, h.stream);
        h.rt->capture_aux(1, src[1], ctx_rows, seq, 0, *h.backend, h.stream);
        EXPECT_TRUE(h.rt->ctx_valid());
        std::string err;
        EXPECT_TRUE(h.rt->run_step(seq, anchor_token, anchor_pos, nq, &err))
            << err;
        EXPECT_TRUE(h.rt->run_markov_head(&err)) << err;
        EXPECT_TRUE(h.rt->run_confidence_head(&err)) << err;
        // The dispatcher's exact call shape (handle_run_dspark_step):
        // inert branch when the dump is off.
        if (h.rt->epm_dump_enabled()) h.rt->epm_write_block_record(true);
        h.backend->synchronize_device();
        EXPECT_EQ(h.rt->epm_dump_enabled(), dump);
        Vec logits(static_cast<size_t>(nq) * V);
        EXPECT_EQ(cudaMemcpy(logits.data(), h.rt->base_logits(),
                             logits.size() * 4, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (void* p : src) h.backend->device_free(p);
        return logits;
    };

    // Dump-OFF: identical command path, no files.
    Harness h_off(cfg_off);
    const Vec logits_off = run_block(h_off, /*dump=*/false);

    // Dump-ON: same forward + one EPMB record per block.
    Harness h_on(cfg_on);
    const Vec logits_on = run_block(h_on, /*dump=*/true);

    // (1) Zero behavior change: forward outputs bit-identical ON vs OFF.
    EXPECT_EQ(std::memcmp(logits_on.data(), logits_off.data(),
                          logits_off.size() * 4),
              0)
        << "EPM dump taps changed the forward";

    // Read back the ON runtime's own device outputs (record ground truth).
    std::vector<int32_t> dev_ids(static_cast<size_t>(nq));
    ASSERT_EQ(cudaMemcpy(dev_ids.data(), h_on.rt->draft_tokens(),
                         dev_ids.size() * 4, cudaMemcpyDeviceToHost),
              cudaSuccess);
    Vec dev_conf(static_cast<size_t>(nq));
    ASSERT_EQ(cudaMemcpy(dev_conf.data(), h_on.rt->confidence(),
                         dev_conf.size() * 4, cudaMemcpyDeviceToHost),
              cudaSuccess);

    // (2) Round trip: parse the record and check every field.
    auto blocks = parse_blocks(dump_dir / "epm_blocks.bin");
    ASSERT_EQ(blocks.size(), 1u);
    const auto& b = blocks[0];
    EXPECT_EQ(b.seq_id, seq);
    EXPECT_EQ(b.block_idx, 0u);
    EXPECT_EQ(b.anchor_pos, static_cast<uint32_t>(anchor_pos));
    EXPECT_EQ(b.anchor_token, static_cast<uint32_t>(anchor_token));
    EXPECT_EQ(b.gamma, static_cast<uint32_t>(nq));
    EXPECT_EQ(b.n_layers, 2u);
    EXPECT_EQ(b.hidden, static_cast<uint32_t>(H));
    ASSERT_EQ(b.has_conf, 1u);
    EXPECT_EQ(std::memcmp(b.ids.data(), dev_ids.data(), nq * 4), 0)
        << "dumped draft ids != device draft ids";
    EXPECT_EQ(std::memcmp(b.conf.data(), dev_conf.data(), nq * 4), 0)
        << "dumped c_k != device confidence output";

    // (3) The dumped hiddens ARE the forward's per-layer residuals: CPU
    // reference with the same BF16 rounding model, same tolerance as the
    // DSP-3 forward test (FP32 accumulation order is the only slack).
    Vec aux_concat(static_cast<size_t>(ctx_rows) * 2 * H);
    for (int r = 0; r < ctx_rows; ++r)
        for (int s = 0; s < 2; ++s)
            for (int c = 0; c < H; ++c)
                aux_concat[(static_cast<size_t>(r) * 2 + s) * H + c] =
                    slot_rows[s][static_cast<size_t>(r) * H + c];
    const auto ref =
        cpu_layer_residuals(host, aux_concat, ctx_rows, anchor_token,
                            anchor_pos, nq);
    ASSERT_EQ(ref.size(), 2u);
    for (int l = 0; l < 2; ++l)
        for (int k = 0; k < nq; ++k)
            for (int c = 0; c < H; ++c) {
                const float got = bf2f(
                    b.hiddens[(static_cast<size_t>(k) * 2 + l) * H + c]);
                const float want =
                    ref[static_cast<size_t>(l)]
                       [static_cast<size_t>(k) * H + c];
                EXPECT_LE(std::abs(got - want),
                          0.06f + 0.05f * std::abs(want))
                    << "hidden[k=" << k << ", l=" << l << ", c=" << c
                    << "] gpu=" << got << " ref=" << want;
            }

    // (4) Determinism + per-seq block counter: a second identical block
    // appends record #1 with bit-identical hiddens.
    (void)run_block(h_on, /*dump=*/true);
    blocks = parse_blocks(dump_dir / "epm_blocks.bin");
    ASSERT_EQ(blocks.size(), 2u);
    EXPECT_EQ(blocks[1].block_idx, 1u);
    EXPECT_EQ(blocks[1].seq_id, seq);
    ASSERT_EQ(blocks[1].hiddens.size(), blocks[0].hiddens.size());
    EXPECT_EQ(std::memcmp(blocks[1].hiddens.data(), blocks[0].hiddens.data(),
                          blocks[0].hiddens.size() * 2),
              0)
        << "identical blocks dumped different hiddens";

    // (5) Hooks-OFF: the disabled runtime produced NO dump artifacts and
    // its dump entry points are inert no-ops.
    h_off.rt->epm_write_block_record(true);  // must be a no-op
    EXPECT_FALSE(h_off.rt->epm_dump_enabled());
    EXPECT_FALSE(fs::exists(dump_dir / "nonexistent"));
    size_t files_in_dump = 0;
    for (auto& e : fs::directory_iterator(dump_dir)) {
        (void)e;
        ++files_in_dump;
    }
    EXPECT_EQ(files_in_dump, 1u) << "only epm_blocks.bin expected";
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 2: routing-record round trip against the real top-K gating kernel
// ═════════════════════════════════════════════════════════════════════════════

TEST(EpmDump, RoutingRecordMatchesGatingKernelAndFetchSeam) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";

    auto backend = layerstorm::compute::make_cuda_sm120_device_backend(
        lc::GpuRef{.position = 0, .id = 0, .type = lc::GpuType::rtx5080});
    backend->set_device();
    void* stream = backend->create_stream();

    // GLM-5.2 routing shape: 256 experts, top-8, n_group=1 (noaux_tc simple
    // path), sigmoid scoring, routed_scaling_factor 2.5, renormalize.
    const int E = 256, K = 8;
    const fs::path dump_dir = fresh_dir("routing");
    lspec::EpmRoutingDumper dumper(dump_dir.string(), E, K);
    ASSERT_TRUE(dumper.ok());

    void* d_logits = backend->device_alloc(E * sizeof(float));
    void* d_w = backend->device_alloc(K * sizeof(float));
    void* d_idx = backend->device_alloc(K * sizeof(int32_t));
    ASSERT_TRUE(d_logits && d_w && d_idx);
    // Pinned host staging, exactly the dispatcher's layout (logits|ids|w).
    auto* h_stage = static_cast<float*>(
        backend->host_alloc_pinned(E * 4 + K * 4 + K * 4));
    ASSERT_NE(h_stage, nullptr);

    std::mt19937 rng(4242);
    std::normal_distribution<float> g(0.0f, 2.0f);

    const uint64_t seq = 9;
    const int layers_per_pos = 3;  // e.g. layers 3, 4, 5 of a feed
    std::vector<std::vector<int32_t>> kernel_ids_by_row;
    std::vector<std::vector<float>> kernel_w_by_row;
    std::vector<std::vector<float>> logits_by_row;

    auto feed_position = [&](uint32_t pos) {
        for (int li = 0; li < layers_per_pos; ++li) {
            std::vector<float> logits(E);
            for (auto& x : logits) x = g(rng);
            backend->memcpy_h2d(d_logits, logits.data(), E * 4);

            lcomp::TopkGatingParams gp{};
            gp.num_tokens = 1;
            gp.num_experts = E;
            gp.topk = K;
            gp.n_group = 1;
            gp.topk_group = 1;
            gp.routed_scaling_factor = 2.5f;
            gp.renormalize = true;
            lcomp::launch_topk_gating(static_cast<float*>(d_w),
                                      static_cast<int32_t*>(d_idx),
                                      static_cast<const float*>(d_logits),
                                      /*bias=*/nullptr, gp, stream);

            // The dispatcher's exact capture: D2H logits+ids+w, sync, row.
            auto* hl = h_stage;
            auto* hi = reinterpret_cast<int32_t*>(hl + E);
            auto* hw = reinterpret_cast<float*>(hi + K);
            backend->memcpy_d2h_async(hl, d_logits, E * 4, stream);
            backend->memcpy_d2h_async(hi, d_idx, K * 4, stream);
            backend->memcpy_d2h_async(hw, d_w, K * 4, stream);
            backend->synchronize_device();

            kernel_ids_by_row.emplace_back(hi, hi + K);
            kernel_w_by_row.emplace_back(hw, hw + K);
            logits_by_row.push_back(logits);
            dumper.add_row(seq, pos, /*layer=*/3 + li, hl, hi, hw,
                           /*last_layer=*/li == layers_per_pos - 1);
        }
    };

    feed_position(100);
    feed_position(101);  // second decode position -> second record

    auto records = parse_routes(dump_dir / "epm_routing.bin");
    ASSERT_EQ(records.size(), 2u);
    for (int rec = 0; rec < 2; ++rec) {
        const auto& r = records[static_cast<size_t>(rec)];
        EXPECT_EQ(r.seq_id, seq);
        EXPECT_EQ(r.token_pos, 100u + static_cast<uint32_t>(rec));
        EXPECT_EQ(r.n_experts, static_cast<uint32_t>(E));
        EXPECT_EQ(r.topk, static_cast<uint32_t>(K));
        ASSERT_EQ(r.rows.size(), static_cast<size_t>(layers_per_pos));
        for (int li = 0; li < layers_per_pos; ++li) {
            const auto& row = r.rows[static_cast<size_t>(li)];
            const size_t flat =
                static_cast<size_t>(rec) * layers_per_pos + li;
            EXPECT_EQ(row.layer, 3u + static_cast<uint32_t>(li));

            // (a) Dumped top-8 ids/weights == the gating kernel's output
            // BITWISE. These ids are the single source of the F-3 routing
            // export the orchestrator copies entry-for-entry into the
            // E_CMD_FETCH_AND_RUN_MOE sideband expert list — so dumped
            // labels == the experts FETCH_AND_RUN actually requests.
            EXPECT_EQ(std::memcmp(row.ids.data(),
                                  kernel_ids_by_row[flat].data(), K * 4),
                      0)
                << "rec " << rec << " layer row " << li;
            EXPECT_EQ(std::memcmp(row.w.data(),
                                  kernel_w_by_row[flat].data(), K * 4),
                      0);

            // (b) Label sanity vs raw logits: sigmoid scoring is monotone
            // and bias is null, so the kernel's top-8 SET must equal the
            // 8 largest raw logits.
            std::vector<int> order(E);
            std::iota(order.begin(), order.end(), 0);
            const auto& lg = logits_by_row[flat];
            std::sort(order.begin(), order.end(), [&](int a, int c) {
                return lg[static_cast<size_t>(a)] >
                       lg[static_cast<size_t>(c)];
            });
            std::vector<int32_t> want(order.begin(), order.begin() + K);
            std::vector<int32_t> got_sorted = row.ids;
            std::sort(want.begin(), want.end());
            std::sort(got_sorted.begin(), got_sorted.end());
            EXPECT_EQ(got_sorted, want)
                << "dumped top-8 set != CPU top-8 of the raw logits";

            // (c) Full 256 pre-top-k logits: FP16 of the exact FP32 values
            // the router produced (decision (A)).
            for (int e = 0; e < E; ++e)
                EXPECT_EQ(row.logits_f16[static_cast<size_t>(e)],
                          lspec::epm_f32_to_f16(lg[static_cast<size_t>(e)]))
                    << "expert " << e;
        }
    }

    backend->host_free_pinned(h_stage);
    backend->device_free(d_logits);
    backend->device_free(d_w);
    backend->device_free(d_idx);
    backend->destroy_stream(stream);
}

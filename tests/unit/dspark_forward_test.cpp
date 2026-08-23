// DSP-3 coverage gate: DSpark DFlash backbone forward + aux-hidden ingest.
//
// Tiers:
//   1. Small-dims CPU-reference test (CUDA GPU required): a synthetic
//      speculators-v0.5 checkpoint with well-behaved random BF16 weights is
//      loaded through the real DSP-2 loader, ingested through the real
//      capture_aux/ingest_context path (fc fusion -> hidden_norm -> per-layer
//      context KV with K-norm + RoPE), and one run_step (anchor + mask block,
//      non-causal attention — INV-DSPARK-ANCHOR) is compared element-wise
//      against a CPU float reference that rounds to BF16 at every buffer
//      write, mirroring the GPU pipeline.
//   2. Real checkpoint (test-data/GLM-5.2-speculator.dspark + a >=10 GB free
//      GPU, skipped otherwise): the backbone runs on the real 7.61 GB
//      weights, produces FINITE base logits [gamma, V] + hiddens [gamma, H],
//      maps the aux layer ids [8,23,39,55,70] to slots 0..4, and is
//      bit-deterministic across two identical runs.
//   3. run_step fail-closed contracts (no valid context / bad anchor_pos /
//      num_query > block_size).

#include "speculation/dspark_runtime.h"

#include "compute/cuda_sm120_device_backend.h"
#include "compute/stream_manager.h"
#include "config/config_parser.h"
#include "core/device_backend.h"
#include "model/quantization/kgroup_quant.h"
#include "model/weight_loader/dspark_loader.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace lc = layerstorm::config;
namespace lm = layerstorm::model;
namespace lspec = layerstorm::speculation;

namespace {

namespace fs = std::filesystem;

// ── BF16 helpers ─────────────────────────────────────────────────────────────

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    // round-to-nearest-even, matching __float2bfloat16.
    const uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding) >> 16);
}

float bf2f(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float bfr(float f) { return bf2f(f2bf(f)); }  // round-trip (BF16 rounding)

// ── Tiny synthetic checkpoint (speculators v0.5) ─────────────────────────────
// H=8, layers=2, heads=2 (== kv heads), head_dim=4, I=16, V=32, r=4,
// aux ids [0,1], block 8 / speculative 7, mask token 5.

struct TinyDims {
    int64_t H = 8, V = 32, r = 4, D = 4, I = 16;
    int layers = 2, heads = 2;
    int64_t Q() const { return heads * D; }
};

struct NamedTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<uint16_t> data;  // BF16 (used when dtype == "BF16")
    std::string dtype = "BF16";
    std::vector<char> raw;       // non-BF16 payload (e.g. d2t I64 offsets)
};

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    return n;
}

// `markov_scale` widens the markov_w1/w2 ranges so the DSP-4 transition
// bias DOMINATES the base logits (the non-no-op control: the Markov head
// must change the argmax vs base-only).  Default keeps DSP-3 behavior.
std::vector<NamedTensor> tiny_tensors(uint32_t seed,
                                      float markov_scale = 0.3f) {
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
    t.push_back(w("markov_head.markov_w1.weight", {d.V, d.r}, -markov_scale,
                  markov_scale));
    t.push_back(w("markov_head.markov_w2.weight", {d.V, d.r}, -markov_scale,
                  markov_scale));
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

// `draft_vocab` == 0 keeps the full-vocab config (draft_vocab_size == V);
// non-zero writes a REDUCED-vocab config (TD-DSPARK-VOCAB-REMAP).
fs::path make_tiny_checkpoint(const std::string& tag,
                              const std::vector<NamedTensor>& tensors,
                              int64_t draft_vocab = 0) {
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

    if (draft_vocab > 0) cfg["draft_vocab_size"] = draft_vocab;

    fs::path dir = fs::path(::testing::TempDir()) / ("dspark_fwd_" + tag);
    fs::create_directories(dir);
    { std::ofstream(dir / "config.json") << cfg.dump(2); }

    nlohmann::json header;
    int64_t offset = 0;
    for (const auto& t : tensors) {
        const int64_t bytes =
            t.dtype == "BF16" ? numel(t.shape) * 2
                              : static_cast<int64_t>(t.raw.size());
        header[t.name] = {{"dtype", t.dtype},
                          {"shape", t.shape},
                          {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    const std::string hj = header.dump();
    const uint64_t hlen = hj.size();
    std::ofstream f(dir / "model.safetensors", std::ios::binary);
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(hj.data(), static_cast<std::streamsize>(hj.size()));
    for (const auto& t : tensors) {
        if (t.dtype == "BF16")
            f.write(reinterpret_cast<const char*>(t.data.data()),
                    static_cast<std::streamsize>(t.data.size() * 2));
        else
            f.write(t.raw.data(), static_cast<std::streamsize>(t.raw.size()));
    }
    return dir;
}

// ── CPU float reference (BF16-rounded at every buffer write) ─────────────────

using Vec = std::vector<float>;

Vec bf16_of(const lm::RawTensor& t) {
    const auto* p = reinterpret_cast<const uint16_t*>(t.data.data());
    Vec v(static_cast<size_t>(t.numel()));
    for (size_t i = 0; i < v.size(); ++i) v[i] = bf2f(p[i]);
    return v;
}

// TD-DSPARK-DRAFT-QUANT: reference view of a GEMM operand under the upload
// requantization — quantize+dequantize through the SAME kgroup packers the
// loader uses (bit-matched to the fused GEMM's in-kernel dequant), so the
// GPU/CPU comparison bands stay the accumulation-order-only bands.
enum class WqMode { kBf16, kFp8, kNvfp4 };

Vec wq_of(const lm::RawTensor& t, WqMode mode) {
    namespace kg = layerstorm::model::kgroup;
    if (mode == WqMode::kBf16 || t.shape.size() != 2) return bf16_of(t);
    const int64_t n = t.shape[0], k = t.shape[1];
    const auto* p = reinterpret_cast<const uint16_t*>(t.data.data());
    Vec out(static_cast<size_t>(n * k));
    if (mode == WqMode::kFp8) {
        std::vector<uint8_t> q(static_cast<size_t>(kg::fp8_weight_bytes(n, k)));
        std::vector<float> s(
            static_cast<size_t>(kg::fp8_scale_bytes(n, k)) / sizeof(float));
        kg::quantize_rows_fp8_e4m3(p, n, k, q.data(), s.data());
        kg::dequantize_rows_fp8_e4m3(q.data(), s.data(), n, k, out.data());
    } else {
        std::vector<uint8_t> q(
            static_cast<size_t>(kg::nvfp4_weight_bytes(n, k)));
        std::vector<uint8_t> s(
            static_cast<size_t>(kg::nvfp4_scale_bytes(n, k)));
        kg::quantize_rows_nvfp4(p, n, k, q.data(), s.data());
        kg::dequantize_rows_nvfp4(q.data(), s.data(), n, k, out.data());
    }
    return out;
}

// C[M,N] = A[M,K] @ W[N,K]^T, FP32 accumulate, BF16-rounded store.
Vec gemm_nt(const Vec& A, const Vec& W, int M, int N, int K,
            bool round_bf16 = true) {
    Vec C(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[static_cast<size_t>(m) * K + k] *
                       W[static_cast<size_t>(n) * K + k];
            C[static_cast<size_t>(m) * N + n] = round_bf16 ? bfr(acc) : acc;
        }
    return C;
}

// RMSNorm rows of `dim`, weight [dim], BF16-rounded store.
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

// In-place NEOX RoPE on [rows, heads, D], row t at position base+t.
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

struct CpuRef {
    // Post-final-norm hidden [nq, H] and logits [nq, V].
    Vec hidden;
    Vec logits;
};

// Mirrors DsparkRuntime ingest_context + run_step exactly (BF16 rounding at
// each write; FP32 dot products). `wq` != kBf16 mirrors the quantized-draft
// forward: GEMM operands go through the kgroup quant round-trip
// (TD-DSPARK-DRAFT-QUANT); norms/embeddings stay BF16.
CpuRef cpu_reference(const lm::DsparkDraftWeights& w, const Vec& aux_concat,
                     int ctx_rows, int anchor_token, int anchor_pos, int nq,
                     WqMode wq = WqMode::kBf16) {
    const auto& ck = w.ckpt;
    const int H = static_cast<int>(ck.hidden_size);
    const int heads = ck.num_attention_heads;
    const int D = static_cast<int>(ck.head_dim);
    const int QD = heads * D;
    const int I = static_cast<int>(ck.intermediate_size);
    const int L = ck.num_hidden_layers;
    // lm_head is DRAFT-vocab sized (== vocab_size unless reduced —
    // TD-DSPARK-VOCAB-REMAP).
    const int Vd = static_cast<int>(ck.draft_vocab_size);
    const int n_aux = static_cast<int>(ck.aux_hidden_state_layer_ids.size());
    const float eps = static_cast<float>(ck.rms_norm_eps);
    const float theta = static_cast<float>(ck.rope_theta);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Context ingest: fc -> hidden_norm -> per-layer K (norm+rope) / V.
    const Vec fc = wq_of(w.fc, wq);
    const Vec hidden_norm_w = bf16_of(w.hidden_norm);
    Vec ctx_hidden = gemm_nt(aux_concat, fc, ctx_rows, H, n_aux * H);
    Vec ctx_normed = rmsnorm(ctx_hidden, hidden_norm_w, ctx_rows, H, eps);

    std::vector<Vec> ctx_k(static_cast<size_t>(L)), ctx_v(static_cast<size_t>(L));
    for (int l = 0; l < L; ++l) {
        const auto& lw = w.layers[static_cast<size_t>(l)];
        Vec k = gemm_nt(ctx_normed, wq_of(lw.k_proj, wq), ctx_rows, QD, H);
        k = rmsnorm(k, bf16_of(lw.k_norm), ctx_rows * heads, D, eps);
        rope(k, ctx_rows, heads, D, /*base=*/0, theta);
        ctx_k[static_cast<size_t>(l)] = std::move(k);
        ctx_v[static_cast<size_t>(l)] =
            gemm_nt(ctx_normed, wq_of(lw.v_proj, wq), ctx_rows, QD, H);
    }

    // Query block: [anchor, mask x (nq-1)].
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

    CpuRef out;
    for (int l = 0; l < L; ++l) {
        const auto& lw = w.layers[static_cast<size_t>(l)];
        Vec q = gemm_nt(normed, wq_of(lw.q_proj, wq), nq, QD, H);
        q = rmsnorm(q, bf16_of(lw.q_norm), nq * heads, D, eps);
        rope(q, nq, heads, D, anchor_pos, theta);
        Vec k = gemm_nt(normed, wq_of(lw.k_proj, wq), nq, QD, H);
        k = rmsnorm(k, bf16_of(lw.k_norm), nq * heads, D, eps);
        rope(k, nq, heads, D, anchor_pos, theta);
        Vec v = gemm_nt(normed, wq_of(lw.v_proj, wq), nq, QD, H);

        // Non-causal attention: ctx positions < anchor_pos + full block.
        const int cu = anchor_pos;  // context rows attended
        Vec attn(static_cast<size_t>(nq) * QD);
        for (int t = 0; t < nq; ++t)
            for (int h = 0; h < heads; ++h) {
                const int total = cu + nq;
                std::vector<float> sc(static_cast<size_t>(total));
                float mx = -INFINITY;
                for (int j = 0; j < total; ++j) {
                    const float* kr =
                        (j < cu)
                            ? &ctx_k[static_cast<size_t>(l)]
                                    [(static_cast<size_t>(j) * heads + h) * D]
                            : &k[(static_cast<size_t>(j - cu) * heads + h) *
                                 D];
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d)
                        dot += q[(static_cast<size_t>(t) * heads + h) * D + d]
                               * kr[d];
                    sc[static_cast<size_t>(j)] = dot * scale;
                    mx = std::max(mx, sc[static_cast<size_t>(j)]);
                }
                float lsum = 0.0f;
                std::vector<float> acc(static_cast<size_t>(D), 0.0f);
                for (int j = 0; j < total; ++j) {
                    const float wj = std::exp(sc[static_cast<size_t>(j)] - mx);
                    lsum += wj;
                    const float* vr =
                        (j < cu)
                            ? &ctx_v[static_cast<size_t>(l)]
                                    [(static_cast<size_t>(j) * heads + h) * D]
                            : &v[(static_cast<size_t>(j - cu) * heads + h) *
                                 D];
                    for (int d = 0; d < D; ++d)
                        acc[static_cast<size_t>(d)] += wj * vr[d];
                }
                for (int d = 0; d < D; ++d)
                    attn[(static_cast<size_t>(t) * heads + h) * D + d] =
                        bfr(acc[static_cast<size_t>(d)] / lsum);
            }

        Vec o = gemm_nt(attn, wq_of(lw.o_proj, wq), nq, H, QD);
        // fused add + rmsnorm: residual (x) += o; normed = rmsnorm(residual).
        for (size_t i = 0; i < x.size(); ++i) x[i] = bfr(x[i] + o[i]);
        normed = rmsnorm(x, bf16_of(lw.post_attention_layernorm), nq, H, eps);

        Vec gate = gemm_nt(normed, wq_of(lw.gate_proj, wq), nq, I, H);
        Vec up = gemm_nt(normed, wq_of(lw.up_proj, wq), nq, I, H);
        Vec act(gate.size());
        for (size_t i = 0; i < gate.size(); ++i)
            act[i] = bfr(gate[i] / (1.0f + std::exp(-gate[i])) * up[i]);
        Vec mlp = gemm_nt(act, wq_of(lw.down_proj, wq), nq, H, I);

        for (size_t i = 0; i < x.size(); ++i) x[i] = bfr(x[i] + mlp[i]);
        const bool last = (l + 1 == L);
        normed = rmsnorm(
            x,
            bf16_of(last ? w.final_norm
                         : w.layers[static_cast<size_t>(l) + 1]
                               .input_layernorm),
            nq, H, eps);
    }

    out.hidden = normed;  // post final norm
    out.logits = gemm_nt(normed, wq_of(w.lm_head, wq), nq, Vd, H,
                         /*round_bf16=*/false);
    return out;
}

// ── GPU harness ──────────────────────────────────────────────────────────────

bool has_cuda_gpu() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
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
    cfg.speculation.dspark.checkpoint_path = ckpt_dir.string();
    cfg.speculation.dspark.draft_gpus = {0};
    // Derive gamma/speculative_tokens from the actual checkpoint so the loader
    // cross-validation matches whatever model is symlinked (shipped ckpt
    // gamma=8/spec=7; glm-5.2-dspark-preview gamma=16/spec=15). config.json is
    // present for both real and tiny synthetic checkpoints passed here.
    {
        auto ck = lm::parse_dspark_checkpoint_config(ckpt_dir);
        cfg.speculation.dspark.block_size = ck.block_size;
        cfg.speculation.dspark.speculative_tokens = ck.speculative_tokens;
    }
    return cfg;
}

/// Two-rank harness (TD-DSPARK-DRAFT-SHARD): DsparkRuntime sharded across
/// CUDA devices 0 and 1 (backends + streams owned).
struct ShardHarness {
    std::unique_ptr<layerstorm::compute::DeviceBackend> backends[2];
    void* streams[2] = {nullptr, nullptr};
    std::unique_ptr<lspec::DsparkRuntime> rt;

    explicit ShardHarness(const lc::Config& cfg) {
        std::vector<lspec::DsparkRuntime::Rank> ranks;
        for (int i = 0; i < 2; ++i) {
            backends[i] = layerstorm::compute::make_cuda_sm120_device_backend(
                lc::GpuRef{.position = i, .id = i,
                           .type = lc::GpuType::rtx5080});
            backends[i]->set_device();
            streams[i] = backends[i]->create_stream();
            ranks.push_back({backends[i].get(), streams[i]});
        }
        rt = lspec::DsparkRuntime::create(cfg, std::move(ranks));
    }
    ~ShardHarness() {
        rt.reset();
        for (int i = 0; i < 2; ++i)
            if (backends[i] && streams[i]) {
                backends[i]->set_device();
                backends[i]->destroy_stream(streams[i]);
            }
    }
};

/// Minimal single-GPU harness around DsparkRuntime (backend + stream owned).
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
        rt.reset();  // frees draft scratch through the backend
        if (backend && stream) backend->destroy_stream(stream);
    }
};

}  // namespace

// ═════════════════════════════════════════════════════════════════════════════
// Tier 1: small-dims CPU reference
// ═════════════════════════════════════════════════════════════════════════════

TEST(DsparkForward, MatchesCpuReferenceSmallDims) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";

    const auto tensors = tiny_tensors(/*seed=*/1234);
    const auto dir = make_tiny_checkpoint("cpuref", tensors);
    auto host = lm::load_dspark_draft(dir);

    auto cfg = runtime_config(dir);
    auto& dc = cfg.speculation.dspark;
    dc.block_size = 8;
    dc.speculative_tokens = 7;
    dc.aux_hidden_state_layer_ids = {0, 1};
    dc.mask_token_id = 5;
    dc.max_anchors = 16;
    dc.draft_vocab_size = 32;
    dc.markov_rank = 4;
    dc.draft_context_capacity_tokens = 256;
    dc.aux_capture_max_rows = 16;

    Harness h(cfg);
    auto* rt = h.rt.get();
    ASSERT_EQ(rt->aux_slot_for_layer(0), 0);
    ASSERT_EQ(rt->aux_slot_for_layer(1), 1);
    ASSERT_EQ(rt->aux_slot_for_layer(2), -1);

    // Synthetic target hiddens: 3 context rows x 2 aux slots x H, distinct
    // values per slot, BF16-exact.
    const TinyDims d;
    const int ctx_rows = 3, H = static_cast<int>(d.H);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> u(-0.8f, 0.8f);
    Vec slot_rows[2];
    for (auto& s : slot_rows) {
        s.resize(static_cast<size_t>(ctx_rows) * H);
        for (auto& x : s) x = bfr(u(rng));
    }

    // Stage each slot's rows in a device buffer and run the REAL capture
    // path (same-GPU "cross-device" copy: identical code path).
    std::vector<uint16_t> bf(static_cast<size_t>(ctx_rows) * H);
    void* src[2];
    for (int s = 0; s < 2; ++s) {
        for (size_t i = 0; i < bf.size(); ++i) bf[i] = f2bf(slot_rows[s][i]);
        src[s] = h.backend->device_alloc(bf.size() * 2);
        ASSERT_NE(src[s], nullptr);
        h.backend->memcpy_h2d(src[s], bf.data(), bf.size() * 2);
    }
    const uint64_t seq = 7;
    rt->capture_aux(0, src[0], ctx_rows, seq, /*start_pos=*/0, *h.backend,
                    h.stream);
    rt->capture_aux(1, src[1], ctx_rows, seq, /*start_pos=*/0, *h.backend,
                    h.stream);
    EXPECT_TRUE(rt->ctx_valid());
    EXPECT_EQ(rt->ctx_len(), ctx_rows);
    EXPECT_EQ(rt->ctx_seq_id(), seq);

    // One backbone forward: anchor token 3 at position 3, ndraft 4.
    const int anchor_token = 3, anchor_pos = 3, nq = 4;
    std::string err;
    ASSERT_TRUE(rt->run_step(seq, anchor_token, anchor_pos, nq, &err)) << err;
    h.backend->synchronize_device();

    // Physical rows = ndraft + the bonus-anchor row (default layout); the
    // CPU reference mirrors the PHYSICAL forward, so compare all rows.
    const int rows = nq + rt->sample_off();

    // Read back GPU logits + hidden.
    const int V = static_cast<int>(d.V);
    Vec gpu_logits(static_cast<size_t>(rows) * V);
    ASSERT_EQ(cudaMemcpy(gpu_logits.data(), rt->base_logits(),
                         gpu_logits.size() * 4, cudaMemcpyDeviceToHost),
              cudaSuccess);
    std::vector<uint16_t> gpu_hidden_bf(static_cast<size_t>(rows) * H);
    ASSERT_EQ(cudaMemcpy(gpu_hidden_bf.data(), rt->hidden_out(),
                         gpu_hidden_bf.size() * 2, cudaMemcpyDeviceToHost),
              cudaSuccess);

    // CPU reference from the same host weights + aux rows.
    Vec aux_concat(static_cast<size_t>(ctx_rows) * 2 * H);
    for (int r = 0; r < ctx_rows; ++r)
        for (int s = 0; s < 2; ++s)
            for (int c = 0; c < H; ++c)
                aux_concat[(static_cast<size_t>(r) * 2 + s) * H + c] =
                    slot_rows[s][static_cast<size_t>(r) * H + c];
    const CpuRef ref = cpu_reference(host, aux_concat, ctx_rows, anchor_token,
                                     anchor_pos, rows);

    // Element-wise comparison.  Both sides round to BF16 at each buffer
    // write; residual differences come from FP32 accumulation order only.
    float max_logit_err = 0.0f;
    for (size_t i = 0; i < gpu_logits.size(); ++i) {
        ASSERT_TRUE(std::isfinite(gpu_logits[i])) << "logit " << i;
        const float e = std::abs(gpu_logits[i] - ref.logits[i]);
        max_logit_err = std::max(max_logit_err, e);
        EXPECT_LE(e, 0.06f + 0.05f * std::abs(ref.logits[i]))
            << "logit[" << i << "] gpu=" << gpu_logits[i]
            << " ref=" << ref.logits[i];
    }
    for (size_t i = 0; i < gpu_hidden_bf.size(); ++i) {
        const float g = bf2f(gpu_hidden_bf[i]);
        EXPECT_LE(std::abs(g - ref.hidden[i]),
                  0.06f + 0.05f * std::abs(ref.hidden[i]))
            << "hidden[" << i << "] gpu=" << g << " ref=" << ref.hidden[i];
    }
    SUCCEED() << "max logit err " << max_logit_err;

    // Fail-closed contracts.
    EXPECT_FALSE(rt->run_step(seq + 1, anchor_token, anchor_pos, nq, &err));
    EXPECT_FALSE(rt->run_step(seq, anchor_token, /*anchor_pos=*/99, nq, &err));
    EXPECT_FALSE(rt->run_step(seq, anchor_token, anchor_pos, /*nq=*/9, &err));
    // Bonus-anchor layout: ndraft == block_size also fails (ndraft+1 rows).
    EXPECT_FALSE(rt->run_step(seq, anchor_token, anchor_pos, /*nq=*/8, &err));

    for (void* p : src) h.backend->device_free(p);
}

// TD-DSPARK-DRAFT-SHARD: the TP=2 sharded forward vs the single-device
// forward on the SAME tiny checkpoint + identical ingest. Two CUDA devices
// required (any pair — the shard semantics are device-agnostic). Gates:
//   * base logits within the FP-reduction band (the K-split o_proj /
//     down_proj partial sums legally reorder the accumulation; everything
//     else is element-identical by construction),
//   * DSP-4 sampled draft ids IDENTICAL (INV-DSPARK-MARKOV discipline: the
//     Markov chain runs on rank 0 over gathered logits; argmax ties are
//     index-deterministic),
//   * the sharded forward is bit-deterministic across two identical runs.
TEST(DsparkForward, ShardedForwardMatchesSingleDevice) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 2)
        GTEST_SKIP() << "needs 2 CUDA devices";

    const auto tensors = tiny_tensors(/*seed=*/555);
    const auto dir = make_tiny_checkpoint("shardfwd", tensors);

    auto cfg = runtime_config(dir);
    auto& dc = cfg.speculation.dspark;
    dc.block_size = 8;
    dc.speculative_tokens = 7;
    dc.aux_hidden_state_layer_ids = {0, 1};
    dc.mask_token_id = 5;
    dc.max_anchors = 16;
    dc.draft_vocab_size = 32;
    dc.markov_rank = 4;
    dc.draft_context_capacity_tokens = 256;
    dc.aux_capture_max_rows = 16;

    const TinyDims d;
    const int ctx_rows = 3, H = static_cast<int>(d.H);
    const int V = static_cast<int>(d.V);
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> u(-0.8f, 0.8f);
    Vec slot_rows[2];
    for (auto& s : slot_rows) {
        s.resize(static_cast<size_t>(ctx_rows) * H);
        for (auto& x : s) x = bfr(u(rng));
    }
    const uint64_t seq = 11;
    const int anchor_token = 3, anchor_pos = 3, nq = 4;
    const int rows = nq + 1;  // + bonus-anchor row

    // Drive one runtime (any rank count) through the identical capture +
    // forward + Markov chain; return logits [rows, V] + draft ids [nq].
    auto drive = [&](lspec::DsparkRuntime* rt,
                     layerstorm::compute::DeviceBackend* cap_backend,
                     void* cap_stream, Vec& logits,
                     std::vector<int32_t>& ids) {
        cap_backend->set_device();
        std::vector<uint16_t> bf(static_cast<size_t>(ctx_rows) * H);
        void* src[2];
        for (int s = 0; s < 2; ++s) {
            for (size_t i = 0; i < bf.size(); ++i)
                bf[i] = f2bf(slot_rows[s][i]);
            src[s] = cap_backend->device_alloc(bf.size() * 2);
            ASSERT_NE(src[s], nullptr);
            cap_backend->memcpy_h2d(src[s], bf.data(), bf.size() * 2);
        }
        rt->capture_aux(0, src[0], ctx_rows, seq, /*start_pos=*/0,
                        *cap_backend, cap_stream);
        rt->capture_aux(1, src[1], ctx_rows, seq, /*start_pos=*/0,
                        *cap_backend, cap_stream);
        ASSERT_TRUE(rt->ctx_valid());
        ASSERT_EQ(rt->ctx_len(), ctx_rows);
        std::string err;
        ASSERT_TRUE(rt->run_step(seq, anchor_token, anchor_pos, nq, &err))
            << err;
        ASSERT_TRUE(rt->run_markov_head(&err)) << err;
        rt->draft_backend()->set_device();
        rt->draft_backend()->synchronize_device();
        logits.resize(static_cast<size_t>(rows) * V);
        ASSERT_EQ(cudaMemcpy(logits.data(), rt->base_logits(),
                             logits.size() * 4, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ids.resize(static_cast<size_t>(nq));
        ASSERT_EQ(cudaMemcpy(ids.data(), rt->draft_tokens(),
                             ids.size() * 4, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        cap_backend->set_device();
        for (void* p : src) cap_backend->device_free(p);
    };

    Vec single_logits, shard_logits, shard_logits2;
    std::vector<int32_t> single_ids, shard_ids, shard_ids2;
    {
        Harness h(cfg);
        drive(h.rt.get(), h.backend.get(), h.stream, single_logits,
              single_ids);
        if (HasFatalFailure()) return;
    }
    {
        ShardHarness h(cfg);
        // Capture sources live on rank 0's GPU (the aux staging home).
        drive(h.rt.get(), h.backends[0].get(), h.streams[0], shard_logits,
              shard_ids);
        if (HasFatalFailure()) return;
        // Determinism: a second identical block on the SAME sharded runtime
        // (rewind: same anchor/pos overwrite semantics) must bit-reproduce.
        drive(h.rt.get(), h.backends[0].get(), h.streams[0], shard_logits2,
              shard_ids2);
        if (HasFatalFailure()) return;
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < single_logits.size(); ++i) {
        ASSERT_TRUE(std::isfinite(shard_logits[i])) << "logit " << i;
        const float e = std::abs(shard_logits[i] - single_logits[i]);
        max_err = std::max(max_err, e);
        EXPECT_LE(e, 0.06f + 0.05f * std::abs(single_logits[i]))
            << "logit[" << i << "] shard=" << shard_logits[i]
            << " single=" << single_logits[i];
    }
    EXPECT_EQ(shard_ids, single_ids)
        << "sharded draft ids diverged from single-device";
    // Bit-determinism of the sharded route.
    EXPECT_EQ(shard_ids, shard_ids2);
    for (size_t i = 0; i < shard_logits.size(); ++i)
        ASSERT_EQ(shard_logits[i], shard_logits2[i]) << "logit " << i;
    SUCCEED() << "max shard-vs-single logit err " << max_err;
}

// Chunked capture (TD-DSPARK-PREFILL-CAP): a target step LARGER than the aux
// staging must ingest in staging-sized pieces per slot (per-slot fc
// accumulation) instead of disabling drafting — and must produce the same
// context as a single-shot ingest of the identical rows.  40 rows through a
// 16-row staging (pieces 16/16/8, non-divisible tail) vs a 64-row staging
// (single shot), cross-checked against the CPU reference.
TEST(DsparkForward, ChunkedCaptureMatchesSingleShotAndCpuRef) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";

    const auto tensors = tiny_tensors(/*seed=*/31337);
    const auto dir = make_tiny_checkpoint("chunked", tensors);
    auto host = lm::load_dspark_draft(dir);

    auto make_cfg = [&](int rows_cap) {
        auto cfg = runtime_config(dir);
        auto& dc = cfg.speculation.dspark;
        dc.block_size = 8;
        dc.speculative_tokens = 7;
        dc.aux_hidden_state_layer_ids = {0, 1};
        dc.mask_token_id = 5;
        dc.max_anchors = 16;
        dc.draft_vocab_size = 32;
        dc.markov_rank = 4;
        dc.draft_context_capacity_tokens = 256;
        dc.aux_capture_max_rows = rows_cap;
        return cfg;
    };

    const TinyDims d;
    const int ctx_rows = 40, H = static_cast<int>(d.H);
    const int V = static_cast<int>(d.V);
    std::mt19937 rng(4711);
    std::uniform_real_distribution<float> u(-0.8f, 0.8f);
    Vec slot_rows[2];
    for (auto& s : slot_rows) {
        s.resize(static_cast<size_t>(ctx_rows) * H);
        for (auto& x : s) x = bfr(u(rng));
    }

    const int anchor_token = 3, anchor_pos = ctx_rows, nq = 4;
    const int rows = nq + 1;  // + the bonus-anchor row (default layout)
    const uint64_t seq = 21;

    // Run the identical capture + forward through a harness with the given
    // staging row cap; return the GPU logits.
    auto run_with_cap = [&](int rows_cap, Vec& logits, bool expect_chunked) {
        auto cfg = make_cfg(rows_cap);
        Harness h(cfg);
        auto* rt = h.rt.get();

        std::vector<uint16_t> bf(static_cast<size_t>(ctx_rows) * H);
        void* src[2];
        for (int s = 0; s < 2; ++s) {
            for (size_t i = 0; i < bf.size(); ++i)
                bf[i] = f2bf(slot_rows[s][i]);
            src[s] = h.backend->device_alloc(bf.size() * 2);
            ASSERT_NE(src[s], nullptr);
            h.backend->memcpy_h2d(src[s], bf.data(), bf.size() * 2);
        }
        ASSERT_EQ(expect_chunked, ctx_rows > rows_cap);
        rt->capture_aux(0, src[0], ctx_rows, seq, /*start_pos=*/0, *h.backend,
                        h.stream);
        rt->capture_aux(1, src[1], ctx_rows, seq, /*start_pos=*/0, *h.backend,
                        h.stream);
        // The oversized step must STAY ARMED (pre-fix it invalidated
        // fail-closed with "rows exceed aux_capture_max_rows").
        ASSERT_TRUE(rt->ctx_valid());
        ASSERT_EQ(rt->ctx_len(), ctx_rows);
        ASSERT_EQ(rt->ctx_seq_id(), seq);

        std::string err;
        ASSERT_TRUE(rt->run_step(seq, anchor_token, anchor_pos, nq, &err))
            << err;
        h.backend->synchronize_device();
        ASSERT_EQ(rt->sample_off() + nq, rows);
        logits.resize(static_cast<size_t>(rows) * V);
        ASSERT_EQ(cudaMemcpy(logits.data(), rt->base_logits(),
                             logits.size() * 4, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (void* p : src) h.backend->device_free(p);
    };

    Vec chunked_logits, single_logits;
    run_with_cap(/*rows_cap=*/16, chunked_logits, /*expect_chunked=*/true);
    if (HasFatalFailure()) return;
    run_with_cap(/*rows_cap=*/64, single_logits, /*expect_chunked=*/false);
    if (HasFatalFailure()) return;

    // CPU reference over the identical aux rows (single concat fc GEMM).
    Vec aux_concat(static_cast<size_t>(ctx_rows) * 2 * H);
    for (int r = 0; r < ctx_rows; ++r)
        for (int s = 0; s < 2; ++s)
            for (int c = 0; c < H; ++c)
                aux_concat[(static_cast<size_t>(r) * 2 + s) * H + c] =
                    slot_rows[s][static_cast<size_t>(r) * H + c];
    const CpuRef ref = cpu_reference(host, aux_concat, ctx_rows, anchor_token,
                                     anchor_pos, rows);

    // Chunked vs single-shot: same context up to the per-slot partial-sum
    // BF16 roundings (n_aux-1 extra roundings per fc element).  Both must
    // also sit on the CPU reference within the standard backbone tolerance.
    for (size_t i = 0; i < chunked_logits.size(); ++i) {
        ASSERT_TRUE(std::isfinite(chunked_logits[i])) << "logit " << i;
        EXPECT_LE(std::abs(chunked_logits[i] - single_logits[i]),
                  0.06f + 0.05f * std::abs(single_logits[i]))
            << "chunked[" << i << "]=" << chunked_logits[i]
            << " single=" << single_logits[i];
        EXPECT_LE(std::abs(chunked_logits[i] - ref.logits[i]),
                  0.06f + 0.05f * std::abs(ref.logits[i]))
            << "chunked[" << i << "]=" << chunked_logits[i]
            << " ref=" << ref.logits[i];
    }
}

// Superchunk chunk-major capture (TD-DSPARK-SUPERCHUNK-CAPTURE): a superchunk
// prefill delivers each aux slot's rows as MULTIPLE contiguous windows,
// CHUNK-MAJOR — all of slot 0's windows arrive before slot 1's first (target
// layer order), each window sourced at its row offset (the fixed
// maybe_dspark_capture hook offsets attn_buf by row_offset rows).  The
// ingested context must equal the single-shot ingest of the identical rows
// (per-slot fc-accumulation tolerance band, as the chunked-capture gate).
// Windows 16/16/8 through (a) a 16-row staging (each window one piece) and
// (b) an 8-row staging (16-row windows split into pieces), vs a 64-row
// single-shot capture and the CPU reference.
TEST(DsparkForward, SuperchunkChunkMajorCaptureMatchesSingleShot) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";

    const auto tensors = tiny_tensors(/*seed=*/24601);
    const auto dir = make_tiny_checkpoint("superchunk", tensors);
    auto host = lm::load_dspark_draft(dir);

    auto make_cfg = [&](int rows_cap) {
        auto cfg = runtime_config(dir);
        auto& dc = cfg.speculation.dspark;
        dc.block_size = 8;
        dc.speculative_tokens = 7;
        dc.aux_hidden_state_layer_ids = {0, 1};
        dc.mask_token_id = 5;
        dc.max_anchors = 16;
        dc.draft_vocab_size = 32;
        dc.markov_rank = 4;
        dc.draft_context_capacity_tokens = 256;
        dc.aux_capture_max_rows = rows_cap;
        return cfg;
    };

    const TinyDims d;
    const int ctx_rows = 40, H = static_cast<int>(d.H);
    const int V = static_cast<int>(d.V);
    std::mt19937 rng(1848);
    std::uniform_real_distribution<float> u(-0.8f, 0.8f);
    Vec slot_rows[2];
    for (auto& s : slot_rows) {
        s.resize(static_cast<size_t>(ctx_rows) * H);
        for (auto& x : s) x = bfr(u(rng));
    }

    const int anchor_token = 3, anchor_pos = ctx_rows, nq = 4;
    const int rows = nq + 1;  // + the bonus-anchor row (default layout)
    const uint64_t seq = 33;

    // Superchunk sub-chunk windows (chunk-major arrival, non-divisible tail).
    struct Win { int start, len; };
    const Win wins[3] = {{0, 16}, {16, 16}, {32, 8}};

    auto upload = [&](Harness& h, void* src[2]) {
        std::vector<uint16_t> bf(static_cast<size_t>(ctx_rows) * H);
        for (int s = 0; s < 2; ++s) {
            for (size_t i = 0; i < bf.size(); ++i)
                bf[i] = f2bf(slot_rows[s][i]);
            src[s] = h.backend->device_alloc(bf.size() * 2);
            ASSERT_NE(src[s], nullptr);
            h.backend->memcpy_h2d(src[s], bf.data(), bf.size() * 2);
        }
    };

    auto read_logits = [&](Harness& h, lspec::DsparkRuntime* rt, Vec& out) {
        std::string err;
        ASSERT_TRUE(rt->run_step(seq, anchor_token, anchor_pos, nq, &err))
            << err;
        h.backend->synchronize_device();
        ASSERT_EQ(rt->sample_off() + nq, rows);
        out.resize(static_cast<size_t>(rows) * V);
        ASSERT_EQ(cudaMemcpy(out.data(), rt->base_logits(),
                             out.size() * 4, cudaMemcpyDeviceToHost),
                  cudaSuccess);
    };

    // Chunk-major window feed, per staging cap.
    auto run_superchunk = [&](int rows_cap, Vec& logits) {
        auto cfg = make_cfg(rows_cap);
        Harness h(cfg);
        auto* rt = h.rt.get();
        void* src[2];
        upload(h, src);
        if (HasFatalFailure()) return;
        for (int s = 0; s < 2; ++s) {
            for (const auto& w : wins) {
                const auto* p = static_cast<const char*>(src[s]) +
                                static_cast<size_t>(w.start) * H * 2;
                rt->capture_aux(s, p, w.len, seq,
                                static_cast<uint32_t>(w.start), *h.backend,
                                h.stream);
            }
        }
        // The chunk-major superchunk shape must STAY ARMED and cover the
        // whole window (pre-fix: sub-chunks >= 2 were skipped or gapped ->
        // long prompts never drafted).
        ASSERT_TRUE(rt->ctx_valid());
        ASSERT_EQ(rt->ctx_len(), ctx_rows);
        ASSERT_EQ(rt->ctx_seq_id(), seq);
        read_logits(h, rt, logits);
        for (void* p : src) h.backend->device_free(p);
    };

    // Single-shot baseline: one 40-row window per slot, 64-row staging.
    auto run_single = [&](Vec& logits) {
        auto cfg = make_cfg(/*rows_cap=*/64);
        Harness h(cfg);
        auto* rt = h.rt.get();
        void* src[2];
        upload(h, src);
        if (HasFatalFailure()) return;
        rt->capture_aux(0, src[0], ctx_rows, seq, /*start_pos=*/0, *h.backend,
                        h.stream);
        rt->capture_aux(1, src[1], ctx_rows, seq, /*start_pos=*/0, *h.backend,
                        h.stream);
        ASSERT_TRUE(rt->ctx_valid());
        ASSERT_EQ(rt->ctx_len(), ctx_rows);
        read_logits(h, rt, logits);
        for (void* p : src) h.backend->device_free(p);
    };

    Vec sc16_logits, sc8_logits, single_logits;
    run_superchunk(/*rows_cap=*/16, sc16_logits);
    if (HasFatalFailure()) return;
    run_superchunk(/*rows_cap=*/8, sc8_logits);
    if (HasFatalFailure()) return;
    run_single(single_logits);
    if (HasFatalFailure()) return;

    // CPU reference over the identical aux rows (single concat fc GEMM).
    Vec aux_concat(static_cast<size_t>(ctx_rows) * 2 * H);
    for (int r = 0; r < ctx_rows; ++r)
        for (int s = 0; s < 2; ++s)
            for (int c = 0; c < H; ++c)
                aux_concat[(static_cast<size_t>(r) * 2 + s) * H + c] =
                    slot_rows[s][static_cast<size_t>(r) * H + c];
    const CpuRef ref = cpu_reference(host, aux_concat, ctx_rows, anchor_token,
                                     anchor_pos, rows);

    for (size_t i = 0; i < single_logits.size(); ++i) {
        for (const Vec* got : {&sc16_logits, &sc8_logits}) {
            const float g = (*got)[i];
            ASSERT_TRUE(std::isfinite(g)) << "logit " << i;
            EXPECT_LE(std::abs(g - single_logits[i]),
                      0.06f + 0.05f * std::abs(single_logits[i]))
                << "superchunk[" << i << "]=" << g
                << " single=" << single_logits[i];
            EXPECT_LE(std::abs(g - ref.logits[i]),
                      0.06f + 0.05f * std::abs(ref.logits[i]))
                << "superchunk[" << i << "]=" << g
                << " ref=" << ref.logits[i];
        }
    }
}

// Rewind + gap semantics of the capture path (host bookkeeping only needs a
// GPU for the enqueue; assertions are on the tracked context).
TEST(DsparkForward, CaptureRewindAndGapContracts) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";

    const auto dir = make_tiny_checkpoint("capture", tiny_tensors(7));
    auto cfg = runtime_config(dir);
    auto& dc = cfg.speculation.dspark;
    dc.block_size = 8;
    dc.speculative_tokens = 7;
    dc.aux_hidden_state_layer_ids = {0, 1};
    dc.mask_token_id = 5;
    dc.max_anchors = 16;
    dc.draft_vocab_size = 32;
    dc.markov_rank = 4;
    dc.draft_context_capacity_tokens = 256;
    dc.aux_capture_max_rows = 16;
    Harness h(cfg);
    auto* rt = h.rt.get();

    const int H = 8;
    void* src = h.backend->device_alloc(16 * H * 2);
    ASSERT_NE(src, nullptr);

    auto step = [&](uint64_t seq, uint32_t pos, int rows) {
        rt->capture_aux(0, src, rows, seq, pos, *h.backend, h.stream);
        rt->capture_aux(1, src, rows, seq, pos, *h.backend, h.stream);
    };

    step(1, 0, 2);  // arm: positions 0..1
    EXPECT_TRUE(rt->ctx_valid());
    EXPECT_EQ(rt->ctx_len(), 2);
    step(1, 2, 1);  // extend
    EXPECT_EQ(rt->ctx_len(), 3);
    step(1, 1, 1);  // rewind (verification rollback): overwrite in place
    EXPECT_TRUE(rt->ctx_valid());
    EXPECT_EQ(rt->ctx_len(), 2);
    step(1, 5, 1);  // gap -> fail closed
    EXPECT_FALSE(rt->ctx_valid());
    step(1, 0, 1);  // position-0 restart re-arms
    EXPECT_TRUE(rt->ctx_valid());
    EXPECT_EQ(rt->ctx_len(), 1);
    step(2, 3, 1);  // sequence switch not at 0 -> fail closed
    EXPECT_FALSE(rt->ctx_valid());
    step(2, 0, 1);  // new sequence from 0 re-arms
    EXPECT_TRUE(rt->ctx_valid());
    EXPECT_EQ(rt->ctx_seq_id(), 2u);

    // TD-DSPARK-CTX-CAP: context-arena overflow (start_pos + rows beyond
    // draft_context_capacity_tokens = 256) degrades GRACEFULLY — drafting
    // disables via invalidate_context (warn-once: ctx_valid_ flips false, so
    // repeats stay silent), no crash/UB, run_step fails closed with a clean
    // error, and a position-0 restart re-arms.
    step(2, 250, 10);  // 260 > 256 -> overflow, fail closed
    EXPECT_FALSE(rt->ctx_valid());
    std::string err;
    EXPECT_FALSE(rt->run_step(2, /*anchor=*/3, /*anchor_pos=*/1, 4, &err));
    EXPECT_NE(err.find("no valid ingested context"), std::string::npos) << err;
    step(2, 251, 1);  // within cap but context invalid -> stays dormant
    EXPECT_FALSE(rt->ctx_valid());
    step(2, 0, 1);  // position-0 restart re-arms
    EXPECT_TRUE(rt->ctx_valid());
    EXPECT_EQ(rt->ctx_len(), 1);

    // ── TD-DSPARK-SUPERCHUNK-CAPTURE: chunk-major multi-window epochs ──
    auto cap = [&](int slot, uint64_t seq, uint32_t pos, int rows) {
        rt->capture_aux(slot, src, rows, seq, pos, *h.backend, h.stream);
    };
    // Superchunk extend from the current frontier (ctx_len 1): slot 0's two
    // contiguous windows arrive before slot 1's (target layer order); the
    // context advances only at the last slot's full coverage.
    cap(0, 2, 1, 2);  // slot 0 window [1,3)
    cap(0, 2, 3, 1);  // slot 0 window [3,4) -> multi-window epoch
    EXPECT_EQ(rt->ctx_len(), 1);  // not finalized yet
    cap(1, 2, 1, 2);  // slot 1 window [1,3)
    cap(1, 2, 3, 1);  // slot 1 window [3,4) -> coverage complete, finalize
    EXPECT_TRUE(rt->ctx_valid());
    EXPECT_EQ(rt->ctx_len(), 4);
    // Slot-0 window GAP between sub-chunks -> fail closed.
    cap(0, 2, 4, 1);  // new epoch [4,5)
    cap(0, 2, 6, 1);  // gap (covered end 5) -> fail closed
    EXPECT_FALSE(rt->ctx_valid());
    // Re-arm chunk-major from 0; a later slot OVERSHOOTING slot 0's
    // coverage -> fail closed.
    cap(0, 2, 0, 2);  // re-arm, slot 0 [0,2)
    cap(0, 2, 2, 2);  // slot 0 [2,4)
    EXPECT_TRUE(rt->ctx_valid());
    cap(1, 2, 0, 2);  // slot 1 [0,2)
    cap(1, 2, 2, 3);  // slot 1 [2,5) overshoots target 4 -> fail closed
    EXPECT_FALSE(rt->ctx_valid());
    // Re-arm; a slot left SHORT of coverage means the next feed's slot-0
    // window sits beyond the (unadvanced) context -> gap, fail closed.
    cap(0, 2, 0, 2);
    cap(0, 2, 2, 2);
    cap(1, 2, 0, 2);  // slot 1 covers only [0,2)
    cap(0, 2, 4, 1);  // next feed: epoch never finalized, ctx_len still 0
    EXPECT_FALSE(rt->ctx_valid());
    // Position-0 restart re-arms after all of the above.
    step(2, 0, 1);
    EXPECT_TRUE(rt->ctx_valid());
    EXPECT_EQ(rt->ctx_len(), 1);

    h.backend->synchronize_device();
    h.backend->device_free(src);
}

// TD-V4-SPEC-PREFILL-CTX: pending_final_window exposes the exact epoch
// state a headless chunk leaves behind (all slots but the last captured),
// and the chunk-final tap closes it by capturing the LAST slot over that
// window — single-window and multi-window (superchunk) epochs, in pieces.
TEST(DsparkForward, PendingFinalWindowChunkFinalTap) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";

    const auto dir = make_tiny_checkpoint("pendingfinal", tiny_tensors(7));
    auto cfg = runtime_config(dir);
    auto& dc = cfg.speculation.dspark;
    dc.block_size = 8;
    dc.speculative_tokens = 7;
    dc.aux_hidden_state_layer_ids = {0, 1};
    dc.mask_token_id = 5;
    dc.max_anchors = 16;
    dc.draft_vocab_size = 32;
    dc.markov_rank = 4;
    dc.draft_context_capacity_tokens = 256;
    dc.aux_capture_max_rows = 16;
    Harness h(cfg);
    auto* rt = h.rt.get();

    const int H = 8;
    void* src = h.backend->device_alloc(16 * H * 2);
    ASSERT_NE(src, nullptr);
    auto cap = [&](int slot, uint64_t seq, uint32_t pos, int rows) {
        rt->capture_aux(slot, src, rows, seq, pos, *h.backend, h.stream);
    };
    uint64_t seq = 0;
    uint32_t s = 0, e = 0;

    // No epoch yet -> nothing pending.
    EXPECT_FALSE(rt->pending_final_window(&seq, &s, &e));

    // Single-window epoch (a headless <=512-row chunk): slot 0 captured,
    // final slot missing -> pending exposes the window; the chunk-final
    // tap closes it and the context advances.
    cap(0, 1, 0, 4);
    ASSERT_TRUE(rt->pending_final_window(&seq, &s, &e));
    EXPECT_EQ(seq, 1u);
    EXPECT_EQ(s, 0u);
    EXPECT_EQ(e, 4u);
    cap(1, 1, 0, 4);  // the tap
    EXPECT_TRUE(rt->ctx_valid());
    EXPECT_EQ(rt->ctx_len(), 4);
    EXPECT_FALSE(rt->pending_final_window(&seq, &s, &e));  // epoch closed

    // Multi-window epoch (superchunk sub-chunks fold chunk-major): slot 0's
    // windows accumulate; pending reports the FULL folded window and the
    // tap may close it in pieces (the dispatcher's staging-sized loop).
    cap(0, 1, 4, 2);
    cap(0, 1, 6, 2);  // fold -> multi-window epoch [4,8)
    ASSERT_TRUE(rt->pending_final_window(&seq, &s, &e));
    EXPECT_EQ(s, 4u);
    EXPECT_EQ(e, 8u);
    cap(1, 1, 4, 3);  // piece 1
    EXPECT_EQ(rt->ctx_len(), 4);  // not finalized yet
    cap(1, 1, 7, 1);  // piece 2 -> coverage complete, finalize
    EXPECT_TRUE(rt->ctx_valid());
    EXPECT_EQ(rt->ctx_len(), 8);

    // A decode-shaped step keeps the same contract (rows == 1).
    cap(0, 1, 8, 1);
    ASSERT_TRUE(rt->pending_final_window(&seq, &s, &e));
    EXPECT_EQ(s, 8u);
    EXPECT_EQ(e, 9u);
    cap(1, 1, 8, 1);
    EXPECT_EQ(rt->ctx_len(), 9);

    // Invalid context -> never pending.
    cap(0, 1, 20, 1);  // gap -> fail closed
    EXPECT_FALSE(rt->ctx_valid());
    EXPECT_FALSE(rt->pending_final_window(&seq, &s, &e));

    h.backend->synchronize_device();
    h.backend->device_free(src);
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 2: real checkpoint smoke (finite logits on the real loaded weights)
// ═════════════════════════════════════════════════════════════════════════════

namespace {
fs::path real_checkpoint_dir() {
    return fs::path(LAYERSTORM_SOURCE_DIR) / "test-data" /
           "GLM-5.2-speculator.dspark";
}
}  // namespace

TEST(DsparkForwardRealCheckpoint, FiniteLogitsAndDeterminism) {
    if (!fs::exists(real_checkpoint_dir() / "model.safetensors"))
        GTEST_SKIP() << "DSpark checkpoint not present";
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    size_t free_b = 0, total_b = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_b, &total_b), cudaSuccess);
    if (free_b < (size_t{10} << 30))
        GTEST_SKIP() << "Insufficient free VRAM: " << free_b;

    auto cfg = runtime_config(real_checkpoint_dir());
    // Keep the smoke small: 256-token context arena + 16-row staging.
    cfg.speculation.dspark.draft_context_capacity_tokens = 256;
    cfg.speculation.dspark.aux_capture_max_rows = 16;
    Harness h(cfg);
    auto* rt = h.rt.get();

    // Aux layer id -> slot mapping (the export contract, INV-DSPARK-AUX).
    ASSERT_EQ(rt->aux_slot_for_layer(8), 0);
    ASSERT_EQ(rt->aux_slot_for_layer(23), 1);
    ASSERT_EQ(rt->aux_slot_for_layer(39), 2);
    ASSERT_EQ(rt->aux_slot_for_layer(55), 3);
    ASSERT_EQ(rt->aux_slot_for_layer(70), 4);
    ASSERT_EQ(rt->aux_slot_for_layer(9), -1);
    ASSERT_EQ(rt->aux_slot_for_layer(78), -1);

    // Two synthetic context rows per aux slot (smooth values, BF16).
    const int H = 6144, rows = 2;
    std::vector<uint16_t> bf(static_cast<size_t>(rows) * H);
    void* src = h.backend->device_alloc(bf.size() * 2);
    ASSERT_NE(src, nullptr);
    for (int s = 0; s < 5; ++s) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < H; ++c)
                bf[static_cast<size_t>(r) * H + c] = f2bf(
                    0.02f * std::sin(0.37f * c + 1.3f * s + 0.7f * r));
        h.backend->memcpy_h2d(src, bf.data(), bf.size() * 2);
        rt->capture_aux(s, src, rows, /*seq=*/1, /*start_pos=*/0, *h.backend,
                        h.stream);
    }
    ASSERT_TRUE(rt->ctx_valid());
    ASSERT_EQ(rt->ctx_len(), rows);

    // Default gamma = speculative_tokens (from the checkpoint); anchor " the"-ish.
    std::string err;
    ASSERT_TRUE(rt->run_step(1, /*anchor=*/290, /*anchor_pos=*/2,
                             /*num_query=*/0, &err))
        << err;
    h.backend->synchronize_device();
    const int nq = cfg.speculation.dspark.speculative_tokens;
    ASSERT_EQ(rt->last_num_query(), nq);

    const int64_t V = 154880;
    std::vector<float> logits(static_cast<size_t>(nq) * V);
    ASSERT_EQ(cudaMemcpy(logits.data(), rt->base_logits(), logits.size() * 4,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    double sum_abs = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        ASSERT_TRUE(std::isfinite(logits[i])) << "logit " << i;
        sum_abs += std::abs(logits[i]);
    }
    EXPECT_GT(sum_abs, 0.0) << "all-zero logits";
    std::vector<uint16_t> hid(static_cast<size_t>(nq) * H);
    ASSERT_EQ(cudaMemcpy(hid.data(), rt->hidden_out(), hid.size() * 2,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (size_t i = 0; i < hid.size(); ++i)
        ASSERT_TRUE(std::isfinite(bf2f(hid[i]))) << "hidden " << i;

    // Bit-determinism: identical run -> identical logits.
    ASSERT_TRUE(rt->run_step(1, 290, 2, 0, &err)) << err;
    h.backend->synchronize_device();
    std::vector<float> logits2(logits.size());
    ASSERT_EQ(cudaMemcpy(logits2.data(), rt->base_logits(),
                         logits2.size() * 4, cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(std::memcmp(logits.data(), logits2.data(), logits.size() * 4),
              0);

    h.backend->device_free(src);
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 1b (DSP-4): sequential Markov head — CPU reference (INV-DSPARK-MARKOV)
// ═════════════════════════════════════════════════════════════════════════════

namespace {

// CPU float reference of the DSP-4 sequential Markov sample loop over GIVEN
// base logits: e = w1[prev]; bias_v = sum_i w2[v,i] * e_i (FP32); corrected =
// base + bias; x_k = argmax (FIRST index on exact ties, torch.argmax).
// `chain=false` freezes prev at the anchor (the control proving the
// sequential dependency matters).
struct CpuMarkovOut {
    std::vector<int32_t> ids;
    Vec corrected;
};

CpuMarkovOut cpu_markov_loop(const Vec& base, const Vec& w1, const Vec& w2,
                             int nq, int V, int r, int anchor, bool chain) {
    CpuMarkovOut out;
    out.corrected.resize(base.size());
    int prev = anchor;
    for (int k = 0; k < nq; ++k) {
        int am = 0;
        float best = -INFINITY;
        for (int v = 0; v < V; ++v) {
            float bias = 0.0f;
            for (int i = 0; i < r; ++i)
                bias += w2[static_cast<size_t>(v) * r + i] *
                        w1[static_cast<size_t>(prev) * r + i];
            const float logit = base[static_cast<size_t>(k) * V + v] + bias;
            out.corrected[static_cast<size_t>(k) * V + v] = logit;
            if (logit > best) {  // strict > keeps the FIRST max
                best = logit;
                am = v;
            }
        }
        out.ids.push_back(am);
        if (chain) prev = am;
    }
    return out;
}

}  // namespace

TEST(DsparkMarkov, MatchesCpuReferenceSequentialLoop) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";

    // markov_scale 2.0 makes the transition bias DOMINATE the base logits so
    // the head is provably non-no-op (and each step's bias depends on the
    // actually-sampled predecessor).
    const auto tensors = tiny_tensors(/*seed=*/2026, /*markov_scale=*/2.0f);
    const auto dir = make_tiny_checkpoint("markov", tensors);
    auto host = lm::load_dspark_draft(dir);

    auto cfg = runtime_config(dir);
    auto& dc = cfg.speculation.dspark;
    dc.block_size = 8;
    dc.speculative_tokens = 7;
    dc.aux_hidden_state_layer_ids = {0, 1};
    dc.mask_token_id = 5;
    dc.max_anchors = 16;
    dc.draft_vocab_size = 32;
    dc.markov_rank = 4;
    dc.draft_context_capacity_tokens = 256;
    dc.aux_capture_max_rows = 16;

    Harness h(cfg);
    auto* rt = h.rt.get();

    // Fail-closed: the Markov head needs run_step outputs.
    std::string err;
    EXPECT_FALSE(rt->run_markov_head(&err));

    // Context ingest through the real capture path (as the DSP-3 test).
    const TinyDims d;
    const int ctx_rows = 3, H = static_cast<int>(d.H);
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> u(-0.8f, 0.8f);
    std::vector<uint16_t> bf(static_cast<size_t>(ctx_rows) * H);
    void* src[2];
    for (auto*& sp : src) {
        for (auto& x : bf) x = f2bf(u(rng));
        sp = h.backend->device_alloc(bf.size() * 2);
        ASSERT_NE(sp, nullptr);
        h.backend->memcpy_h2d(sp, bf.data(), bf.size() * 2);
    }
    const uint64_t seq = 11;
    rt->capture_aux(0, src[0], ctx_rows, seq, 0, *h.backend, h.stream);
    rt->capture_aux(1, src[1], ctx_rows, seq, 0, *h.backend, h.stream);
    ASSERT_TRUE(rt->ctx_valid());

    // Backbone forward + on-device sequential Markov loop.
    const int anchor = 3, nq = 4, V = static_cast<int>(d.V),
              r = static_cast<int>(d.r);
    ASSERT_TRUE(rt->run_step(seq, anchor, /*anchor_pos=*/3, nq, &err)) << err;
    ASSERT_TRUE(rt->run_markov_head(&err)) << err;
    h.backend->synchronize_device();

    // Read back the GPU base logits (the Markov loop's input — isolates the
    // head from backbone tolerance), corrected logits, and sampled ids.
    // The head consumes physical rows [sample_off, sample_off+nq) — slice
    // the bonus-anchor row off so the CPU chain sees the sampled rows.
    const int off = rt->sample_off();
    Vec base_phys(static_cast<size_t>(off + nq) * V);
    ASSERT_EQ(cudaMemcpy(base_phys.data(), rt->base_logits(),
                         base_phys.size() * 4, cudaMemcpyDeviceToHost),
              cudaSuccess);
    const Vec base(base_phys.begin() + static_cast<size_t>(off) * V,
                   base_phys.end());
    Vec corr(static_cast<size_t>(nq) * V);
    std::vector<int32_t> ids(static_cast<size_t>(nq));
    ASSERT_EQ(cudaMemcpy(corr.data(), rt->corrected_logits(),
                         corr.size() * 4, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ids.data(), rt->draft_tokens(), ids.size() * 4,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    // CPU reference: sequential chain from the anchor.
    const Vec w1 = bf16_of(host.markov_w1), w2 = bf16_of(host.markov_w2);
    const auto chained = cpu_markov_loop(base, w1, w2, nq, V, r, anchor,
                                         /*chain=*/true);

    // Control 1 (non-no-op): the transition bias changes the argmax vs
    // base-only for at least one step.
    std::vector<int32_t> base_only(static_cast<size_t>(nq));
    for (int k = 0; k < nq; ++k) {
        int am = 0;
        float best = -INFINITY;
        for (int v = 0; v < V; ++v)
            if (base[static_cast<size_t>(k) * V + v] > best) {
                best = base[static_cast<size_t>(k) * V + v];
                am = v;
            }
        base_only[static_cast<size_t>(k)] = am;
    }
    EXPECT_NE(base_only, chained.ids)
        << "Markov bias never changed the argmax — head is a no-op";

    // Control 2 (sequential dependency): freezing prev at the anchor gives a
    // DIFFERENT sequence, and the GPU matches the CHAINED one.
    const auto anchor_fixed = cpu_markov_loop(base, w1, w2, nq, V, r, anchor,
                                              /*chain=*/false);
    EXPECT_NE(anchor_fixed.ids, chained.ids)
        << "chained vs anchor-fixed identical — sequential feed untested";

    // Exact integer token-id match; corrected logits within FP32
    // reduce-order tolerance.
    for (int k = 0; k < nq; ++k)
        EXPECT_EQ(ids[static_cast<size_t>(k)],
                  chained.ids[static_cast<size_t>(k)])
            << "draft token " << k;
    for (size_t i = 0; i < corr.size(); ++i)
        EXPECT_LE(std::abs(corr[i] - chained.corrected[i]),
                  1e-4f + 1e-4f * std::abs(chained.corrected[i]))
            << "corrected[" << i << "] gpu=" << corr[i]
            << " ref=" << chained.corrected[i];

    for (void* p : src) h.backend->device_free(p);
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 1b-2 (TD-DSPARK-VOCAB-REMAP): reduced draft vocab — d2t remap
// ═════════════════════════════════════════════════════════════════════════════
// Vd=12 < V=32; d2t maps draft id j -> target id 2j+1 (offset j+1: every
// offset non-zero so an unremapped id can NEVER pass).  The whole pipeline
// runs hermetically on a synthetic checkpoint: base logits come out
// DRAFT-vocab sized, the Markov bias/argmax runs in draft space, and the
// finalize kernel writes TARGET ids whose markov_w1 rows (target vocab)
// feed the next step's e-chain — matched against a CPU reference of the
// same chain.

TEST(DsparkMarkovReducedVocab, RemapsDraftArgmaxToTargetIds) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";

    const TinyDims d;
    const int Vd = 12, V = static_cast<int>(d.V),
              H = static_cast<int>(d.H), r = static_cast<int>(d.r);

    // Reduced tensor set: lm_head/markov_w2 shrink to Vd rows; d2t added.
    // markov_scale 2.0 keeps the transition bias dominant (non-no-op head).
    auto tensors = tiny_tensors(/*seed=*/9091, /*markov_scale=*/2.0f);
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& t : tensors) {
        if (t.name == "lm_head.weight") {
            t.shape = {Vd, d.H};
            t.data.resize(static_cast<size_t>(Vd) * H);
            for (auto& x : t.data) x = f2bf(0.3f * u(rng));
        } else if (t.name == "markov_head.markov_w2.weight") {
            t.shape = {Vd, d.r};
            t.data.resize(static_cast<size_t>(Vd) * r);
            for (auto& x : t.data) x = f2bf(2.0f * u(rng));
        }
    }
    std::vector<int64_t> d2t(static_cast<size_t>(Vd));
    for (int j = 0; j < Vd; ++j) d2t[static_cast<size_t>(j)] = j + 1;
    NamedTensor d2t_t;
    d2t_t.name = "d2t";
    d2t_t.shape = {Vd};
    d2t_t.dtype = "I64";
    d2t_t.raw.resize(d2t.size() * 8);
    std::memcpy(d2t_t.raw.data(), d2t.data(), d2t_t.raw.size());
    tensors.push_back(std::move(d2t_t));

    const auto dir = make_tiny_checkpoint("reduced", tensors,
                                          /*draft_vocab=*/Vd);
    auto host = lm::load_dspark_draft(dir);
    ASSERT_EQ(host.ckpt.draft_vocab_size, Vd);
    ASSERT_NE(host.d2t.data.data(), nullptr);

    auto cfg = runtime_config(dir);
    auto& dc = cfg.speculation.dspark;
    dc.block_size = 8;
    dc.speculative_tokens = 7;
    dc.aux_hidden_state_layer_ids = {0, 1};
    dc.mask_token_id = 5;
    dc.max_anchors = 16;
    dc.draft_vocab_size = Vd;  // cross-validated against the checkpoint
    dc.markov_rank = 4;
    dc.draft_context_capacity_tokens = 256;
    dc.aux_capture_max_rows = 16;

    Harness h(cfg);
    auto* rt = h.rt.get();

    // Context ingest through the real capture path.
    const int ctx_rows = 3;
    std::vector<uint16_t> bf(static_cast<size_t>(ctx_rows) * H);
    void* src[2];
    for (auto*& sp : src) {
        for (auto& x : bf) x = f2bf(0.8f * u(rng));
        sp = h.backend->device_alloc(bf.size() * 2);
        ASSERT_NE(sp, nullptr);
        h.backend->memcpy_h2d(sp, bf.data(), bf.size() * 2);
    }
    const uint64_t seq = 17;
    rt->capture_aux(0, src[0], ctx_rows, seq, 0, *h.backend, h.stream);
    rt->capture_aux(1, src[1], ctx_rows, seq, 0, *h.backend, h.stream);
    ASSERT_TRUE(rt->ctx_valid());

    // Backbone + Markov chain.  Anchor 25 >= Vd proves the anchor path is
    // TARGET-vocab (its markov_w1/embed rows exist only in the full vocab).
    const int anchor = 25, nq = 4;
    std::string err;
    ASSERT_TRUE(rt->run_step(seq, anchor, /*anchor_pos=*/3, nq, &err)) << err;
    ASSERT_TRUE(rt->run_markov_head(&err)) << err;
    h.backend->synchronize_device();

    // GPU outputs: draft-vocab base/corrected logits + TARGET-vocab ids.
    // Slice the bonus-anchor row off the physical base logits (the head
    // consumes rows [sample_off, sample_off+nq)).
    const int off = rt->sample_off();
    Vec base_phys(static_cast<size_t>(off + nq) * Vd);
    ASSERT_EQ(cudaMemcpy(base_phys.data(), rt->base_logits(),
                         base_phys.size() * 4, cudaMemcpyDeviceToHost),
              cudaSuccess);
    const Vec base(base_phys.begin() + static_cast<size_t>(off) * Vd,
                   base_phys.end());
    Vec corr(static_cast<size_t>(nq) * Vd);
    std::vector<int32_t> ids(static_cast<size_t>(nq));
    ASSERT_EQ(cudaMemcpy(corr.data(), rt->corrected_logits(),
                         corr.size() * 4, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ids.data(), rt->draft_tokens(), ids.size() * 4,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Base logits vs the CPU backbone reference (draft-vocab lm_head).
    Vec aux_concat(static_cast<size_t>(ctx_rows) * 2 * H);
    {
        // Re-read the staged slot rows from the device for the reference
        // concat (they were generated above per slot).
        std::vector<uint16_t> row(static_cast<size_t>(ctx_rows) * H);
        for (int s = 0; s < 2; ++s) {
            ASSERT_EQ(cudaMemcpy(row.data(), src[s], row.size() * 2,
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            for (int rr = 0; rr < ctx_rows; ++rr)
                for (int c = 0; c < H; ++c)
                    aux_concat[(static_cast<size_t>(rr) * 2 + s) * H + c] =
                        bf2f(row[static_cast<size_t>(rr) * H + c]);
        }
    }
    // The CPU reference mirrors the PHYSICAL forward (off + nq rows in the
    // bonus-anchor layout) — compare all physical rows.
    const CpuRef ref = cpu_reference(host, aux_concat, ctx_rows, anchor,
                                     /*anchor_pos=*/3, off + nq);
    ASSERT_EQ(ref.logits.size(), base_phys.size());
    for (size_t i = 0; i < base_phys.size(); ++i) {
        ASSERT_TRUE(std::isfinite(base_phys[i])) << "base logit " << i;
        EXPECT_LE(std::abs(base_phys[i] - ref.logits[i]),
                  0.06f + 0.05f * std::abs(ref.logits[i]))
            << "base[" << i << "] gpu=" << base_phys[i]
            << " ref=" << ref.logits[i];
    }

    // CPU Markov chain over the GPU's OWN base logits (isolates the head +
    // remap from backbone tolerance): draft-space bias/argmax, d2t map,
    // TARGET-vocab markov_w1 for the next e.
    const Vec w1 = bf16_of(host.markov_w1), w2 = bf16_of(host.markov_w2);
    std::vector<int32_t> want_ids;
    Vec want_corr(corr.size());
    int prev = anchor;  // target id
    for (int k = 0; k < nq; ++k) {
        int am = 0;
        float best = -INFINITY;
        for (int v = 0; v < Vd; ++v) {
            float bias = 0.0f;
            for (int i = 0; i < r; ++i)
                bias += w2[static_cast<size_t>(v) * r + i] *
                        w1[static_cast<size_t>(prev) * r + i];
            const float logit =
                base[static_cast<size_t>(k) * Vd + v] + bias;
            want_corr[static_cast<size_t>(k) * Vd + v] = logit;
            if (logit > best) {  // strict > keeps the FIRST max
                best = logit;
                am = v;
            }
        }
        const int target =
            am + static_cast<int>(d2t[static_cast<size_t>(am)]);
        want_ids.push_back(target);
        prev = target;  // the chain feeds TARGET ids into markov_w1
    }

    for (int k = 0; k < nq; ++k) {
        EXPECT_EQ(ids[static_cast<size_t>(k)],
                  want_ids[static_cast<size_t>(k)])
            << "draft token " << k;
        // Every sampled id must be a d2t image (odd, in [1, 2*Vd-1]) — an
        // unremapped draft id would be even or out of that range.
        EXPECT_EQ(ids[static_cast<size_t>(k)] % 2, 1)
            << "id " << ids[static_cast<size_t>(k)] << " not d2t-mapped";
        EXPECT_GE(ids[static_cast<size_t>(k)], 1);
        EXPECT_LT(ids[static_cast<size_t>(k)], V);
    }
    for (size_t i = 0; i < corr.size(); ++i)
        EXPECT_LE(std::abs(corr[i] - want_corr[i]),
                  1e-4f + 1e-4f * std::abs(want_corr[i]))
            << "corrected[" << i << "] gpu=" << corr[i]
            << " ref=" << want_corr[i];

    for (void* p : src) h.backend->device_free(p);
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 2b (DSP-4): real checkpoint — finite in-vocab draft ids, deterministic
// ═════════════════════════════════════════════════════════════════════════════

// ═════════════════════════════════════════════════════════════════════════════
// Tier 1c (DSP-6): trained confidence head — CPU reference (INV-DSPARK-CONF)
// ═════════════════════════════════════════════════════════════════════════════
// c_k = sigmoid(proj · [hidden_k ; markov_w1[x_{k-1}]] + bias): hidden-FIRST
// concat, x_{k-1} = the ACTUALLY-SAMPLED predecessor (anchor at k=0) — the
// DeepSpec AcceptRatePredictor semantics (qwen3/modeling.py
// predict_confidence_step; eval draft_ops.py _predict_confidence_logits).
// The reference consumes the GPU's OWN hidden_out + draft ids (isolates the
// head from backbone/Markov tolerances).

TEST(DsparkConfidence, MatchesCpuReferenceAndNonDegenerate) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";

    const auto tensors = tiny_tensors(/*seed=*/4242, /*markov_scale=*/2.0f);
    const auto dir = make_tiny_checkpoint("confidence", tensors);
    auto host = lm::load_dspark_draft(dir);

    auto cfg = runtime_config(dir);
    auto& dc = cfg.speculation.dspark;
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

    Harness h(cfg);
    auto* rt = h.rt.get();
    ASSERT_TRUE(rt->has_confidence_head());

    // Fail-closed: no run_step outputs yet.
    std::string err;
    EXPECT_FALSE(rt->run_confidence_head(&err));

    // Context ingest through the real capture path.
    const TinyDims d;
    const int ctx_rows = 3, H = static_cast<int>(d.H);
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> u(-0.8f, 0.8f);
    std::vector<uint16_t> bf(static_cast<size_t>(ctx_rows) * H);
    void* src[2];
    for (auto*& sp : src) {
        for (auto& x : bf) x = f2bf(u(rng));
        sp = h.backend->device_alloc(bf.size() * 2);
        ASSERT_NE(sp, nullptr);
        h.backend->memcpy_h2d(sp, bf.data(), bf.size() * 2);
    }
    const uint64_t seq = 13;
    rt->capture_aux(0, src[0], ctx_rows, seq, 0, *h.backend, h.stream);
    rt->capture_aux(1, src[1], ctx_rows, seq, 0, *h.backend, h.stream);
    ASSERT_TRUE(rt->ctx_valid());

    const int anchor = 3, nq = 4, r = static_cast<int>(d.r);
    ASSERT_TRUE(rt->run_step(seq, anchor, /*anchor_pos=*/3, nq, &err)) << err;

    // Fail-closed: the with_markov head needs the DSP-4 e-chain first.
    EXPECT_FALSE(rt->run_confidence_head(&err));
    EXPECT_NE(err.find("run_markov_head"), std::string::npos) << err;

    ASSERT_TRUE(rt->run_markov_head(&err)) << err;
    ASSERT_TRUE(rt->run_confidence_head(&err)) << err;
    h.backend->synchronize_device();

    // Read back the GPU's hidden + sampled ids (the head's inputs) + c_k.
    // Slice the bonus-anchor row off the physical hidden (the confidence
    // head consumes rows [sample_off, sample_off+nq)).
    const int off = rt->sample_off();
    std::vector<uint16_t> hid_phys(static_cast<size_t>(off + nq) * H);
    ASSERT_EQ(cudaMemcpy(hid_phys.data(), rt->hidden_out(),
                         hid_phys.size() * 2, cudaMemcpyDeviceToHost),
              cudaSuccess);
    const std::vector<uint16_t> hid(
        hid_phys.begin() + static_cast<size_t>(off) * H, hid_phys.end());
    std::vector<int32_t> ids(static_cast<size_t>(nq));
    ASSERT_EQ(cudaMemcpy(ids.data(), rt->draft_tokens(), ids.size() * 4,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    Vec conf(static_cast<size_t>(nq));
    ASSERT_EQ(cudaMemcpy(conf.data(), rt->confidence(), conf.size() * 4,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    // CPU float reference: concat (hidden FIRST) -> dot + bias -> sigmoid.
    const Vec w1 = bf16_of(host.markov_w1);
    const Vec wc = bf16_of(host.confidence_proj_weight);
    const Vec bc = bf16_of(host.confidence_proj_bias);
    ASSERT_EQ(wc.size(), static_cast<size_t>(H + r));
    ASSERT_EQ(bc.size(), 1u);
    float cmin = 1.0f, cmax = 0.0f;
    for (int k = 0; k < nq; ++k) {
        const int prev = (k == 0) ? anchor : ids[static_cast<size_t>(k - 1)];
        float logit = bc[0];
        for (int i = 0; i < H; ++i)
            logit += bf2f(hid[static_cast<size_t>(k) * H + i]) * wc[static_cast<size_t>(i)];
        for (int i = 0; i < r; ++i)
            logit += w1[static_cast<size_t>(prev) * r + i] *
                     wc[static_cast<size_t>(H + i)];
        const float ref = 1.0f / (1.0f + std::exp(-logit));
        EXPECT_GT(conf[static_cast<size_t>(k)], 0.0f) << "c_" << k;
        EXPECT_LT(conf[static_cast<size_t>(k)], 1.0f) << "c_" << k;
        EXPECT_LE(std::abs(conf[static_cast<size_t>(k)] - ref),
                  1e-4f + 1e-3f * std::abs(ref))
            << "c_" << k << " gpu=" << conf[static_cast<size_t>(k)]
            << " ref=" << ref << " (logit " << logit << ")";
        cmin = std::min(cmin, conf[static_cast<size_t>(k)]);
        cmax = std::max(cmax, conf[static_cast<size_t>(k)]);
    }
    // Non-degenerate control: c_k varies with the draft (hidden_k and the
    // sampled predecessor differ per position) — not a constant / all-0.5.
    EXPECT_GT(cmax - cmin, 1e-6f)
        << "confidence head emitted a constant — degenerate";

    for (void* p : src) h.backend->device_free(p);
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 2c (DSP-6): real checkpoint — finite deterministic c_k in (0,1)
// ═════════════════════════════════════════════════════════════════════════════

TEST(DsparkConfidenceRealCheckpoint, FiniteDeterministicSurvival) {
    if (!fs::exists(real_checkpoint_dir() / "model.safetensors"))
        GTEST_SKIP() << "DSpark checkpoint not present";
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    size_t free_b = 0, total_b = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_b, &total_b), cudaSuccess);
    if (free_b < (size_t{10} << 30))
        GTEST_SKIP() << "Insufficient free VRAM: " << free_b;

    auto cfg = runtime_config(real_checkpoint_dir());
    cfg.speculation.dspark.draft_context_capacity_tokens = 256;
    cfg.speculation.dspark.aux_capture_max_rows = 16;
    cfg.speculation.dspark.confidence_enabled = true;
    Harness h(cfg);
    auto* rt = h.rt.get();
    ASSERT_TRUE(rt->has_confidence_head());
    ASSERT_TRUE(rt->ckpt().confidence_head_with_markov);

    // Synthetic context rows per aux slot (as the DSP-3/DSP-4 real tests).
    const int H = 6144, rows = 2;
    std::vector<uint16_t> bf(static_cast<size_t>(rows) * H);
    void* src = h.backend->device_alloc(bf.size() * 2);
    ASSERT_NE(src, nullptr);
    for (int s = 0; s < 5; ++s) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < H; ++c)
                bf[static_cast<size_t>(r) * H + c] = f2bf(
                    0.02f * std::sin(0.37f * c + 1.3f * s + 0.7f * r));
        h.backend->memcpy_h2d(src, bf.data(), bf.size() * 2);
        rt->capture_aux(s, src, rows, /*seq=*/1, /*start_pos=*/0, *h.backend,
                        h.stream);
    }
    ASSERT_TRUE(rt->ctx_valid());

    // nq from the checkpoint (shipped ckpt 7; glm-5.2-dspark-preview 15).
    const int nq = cfg.speculation.dspark.speculative_tokens;
    auto run_once = [&](std::vector<float>& conf) {
        std::string err;
        ASSERT_TRUE(rt->run_step(1, /*anchor=*/290, /*anchor_pos=*/2,
                                 /*num_query=*/0, &err))
            << err;
        ASSERT_TRUE(rt->run_markov_head(&err)) << err;
        ASSERT_TRUE(rt->run_confidence_head(&err)) << err;
        h.backend->synchronize_device();
        ASSERT_EQ(rt->last_num_query(), nq);
        conf.resize(static_cast<size_t>(nq));
        ASSERT_EQ(cudaMemcpy(conf.data(), rt->confidence(), conf.size() * 4,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
    };

    std::vector<float> c1, c2;
    run_once(c1);
    ASSERT_EQ(c1.size(), static_cast<size_t>(nq));  // run_once ASSERT-returned?
    float cmin = 1.0f, cmax = 0.0f;
    double a = 1.0;  // cumulative survival must stay in (0,1], non-increasing
    for (int k = 0; k < nq; ++k) {
        ASSERT_TRUE(std::isfinite(c1[static_cast<size_t>(k)])) << "c_" << k;
        EXPECT_GT(c1[static_cast<size_t>(k)], 0.0f) << "c_" << k;
        EXPECT_LT(c1[static_cast<size_t>(k)], 1.0f) << "c_" << k;
        const double prev_a = a;
        a *= static_cast<double>(c1[static_cast<size_t>(k)]);
        EXPECT_LE(a, prev_a) << "cumulative survival increased at " << k;
        cmin = std::min(cmin, c1[static_cast<size_t>(k)]);
        cmax = std::max(cmax, c1[static_cast<size_t>(k)]);
    }
    // Non-degenerate on the real trained head: c_k varies across positions.
    EXPECT_GT(cmax - cmin, 1e-6f)
        << "real confidence head emitted a constant — degenerate";

    // Bit-determinism across two identical full draft steps.
    run_once(c2);
    ASSERT_EQ(c2.size(), c1.size());
    EXPECT_EQ(std::memcmp(c1.data(), c2.data(), c1.size() * 4), 0);

    std::string cs;
    for (float c : c1) {
        char buf[32];
        snprintf(buf, sizeof buf, " %.4f", c);
        cs += buf;
    }
    std::cout << "[ DSP-6    ] real-checkpoint c_k:" << cs << "\n";

    h.backend->device_free(src);
}

TEST(DsparkMarkovRealCheckpoint, FiniteDeterministicDraftTokens) {
    if (!fs::exists(real_checkpoint_dir() / "model.safetensors"))
        GTEST_SKIP() << "DSpark checkpoint not present";
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    size_t free_b = 0, total_b = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_b, &total_b), cudaSuccess);
    if (free_b < (size_t{10} << 30))
        GTEST_SKIP() << "Insufficient free VRAM: " << free_b;

    auto cfg = runtime_config(real_checkpoint_dir());
    cfg.speculation.dspark.draft_context_capacity_tokens = 256;
    cfg.speculation.dspark.aux_capture_max_rows = 16;
    Harness h(cfg);
    auto* rt = h.rt.get();

    // Synthetic context rows per aux slot (as the DSP-3 real test).
    const int H = 6144, rows = 2;
    std::vector<uint16_t> bf(static_cast<size_t>(rows) * H);
    void* src = h.backend->device_alloc(bf.size() * 2);
    ASSERT_NE(src, nullptr);
    for (int s = 0; s < 5; ++s) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < H; ++c)
                bf[static_cast<size_t>(r) * H + c] = f2bf(
                    0.02f * std::sin(0.37f * c + 1.3f * s + 0.7f * r));
        h.backend->memcpy_h2d(src, bf.data(), bf.size() * 2);
        rt->capture_aux(s, src, rows, /*seq=*/1, /*start_pos=*/0, *h.backend,
                        h.stream);
    }
    ASSERT_TRUE(rt->ctx_valid());

    // nq from the checkpoint (shipped ckpt 7; glm-5.2-dspark-preview 15).
    const int nq = cfg.speculation.dspark.speculative_tokens;
    auto run_once = [&](std::vector<int32_t>& ids, std::vector<float>& corr) {
        std::string err;
        ASSERT_TRUE(rt->run_step(1, /*anchor=*/290, /*anchor_pos=*/2,
                                 /*num_query=*/0, &err))
            << err;
        ASSERT_TRUE(rt->run_markov_head(&err)) << err;
        h.backend->synchronize_device();
        ASSERT_EQ(rt->last_num_query(), nq);
        ids.resize(static_cast<size_t>(nq));
        ASSERT_EQ(cudaMemcpy(ids.data(), rt->draft_tokens(), ids.size() * 4,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        corr.resize(static_cast<size_t>(nq) * 154880);
        ASSERT_EQ(cudaMemcpy(corr.data(), rt->corrected_logits(),
                             corr.size() * 4, cudaMemcpyDeviceToHost),
                  cudaSuccess);
    };

    const int64_t V = 154880;
    std::vector<int32_t> ids1, ids2;
    std::vector<float> corr1, corr2;
    run_once(ids1, corr1);
    ASSERT_EQ(ids1.size(), static_cast<size_t>(nq));  // run_once ASSERT-return

    // Gamma finite in-vocab token ids; corrected logits finite and NOT equal
    // to the base logits (the bias is doing something on real weights).
    for (int k = 0; k < nq; ++k) {
        EXPECT_GE(ids1[static_cast<size_t>(k)], 0) << "draft token " << k;
        EXPECT_LT(ids1[static_cast<size_t>(k)], V) << "draft token " << k;
    }
    for (size_t i = 0; i < corr1.size(); ++i)
        ASSERT_TRUE(std::isfinite(corr1[i])) << "corrected " << i;
    std::vector<float> base(corr1.size());
    ASSERT_EQ(cudaMemcpy(base.data(), rt->base_logits(), base.size() * 4,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    double bias_mass = 0.0;
    for (size_t i = 0; i < base.size(); ++i)
        bias_mass += std::abs(corr1[i] - base[i]);
    EXPECT_GT(bias_mass, 0.0) << "corrected == base: Markov bias is a no-op";

    // Bit-determinism across two identical full draft steps.
    run_once(ids2, corr2);
    ASSERT_EQ(ids2.size(), ids1.size());
    ASSERT_EQ(corr2.size(), corr1.size());
    EXPECT_EQ(std::memcmp(ids1.data(), ids2.data(), ids1.size() * 4), 0);
    EXPECT_EQ(std::memcmp(corr1.data(), corr2.data(), corr1.size() * 4), 0);

    std::string tok_str;
    for (int32_t t : ids1) tok_str += std::to_string(t) + " ";
    std::cout << "[ DSP-4    ] real-checkpoint draft tokens: " << tok_str
              << "\n";

    h.backend->device_free(src);
}

// ═════════════════════════════════════════════════════════════════════════════
// TD-DSPARK-DRAFT-QUANT: quantized-draft forward
// ═════════════════════════════════════════════════════════════════════════════

// Small-dims CPU reference under both quant formats: the GPU runtime uploads
// with draft_weights_quant set (requant-at-upload + fused-dequant GEMMs) and
// must match the CPU reference whose GEMM operands went through the SAME
// kgroup quant round-trip — the comparison band stays the accumulation-order
// band of the BF16 test because the dequantized weights are bit-identical on
// both sides.
TEST(DsparkForwardQuant, MatchesCpuReferenceSmallDims) {
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";

    for (auto [quant, wq] :
         {std::pair{lc::DsparkDraftWeightsQuant::fp8_e4m3, WqMode::kFp8},
          std::pair{lc::DsparkDraftWeightsQuant::nvfp4, WqMode::kNvfp4}}) {
        SCOPED_TRACE(wq == WqMode::kFp8 ? "fp8_e4m3" : "nvfp4");
        const auto tensors = tiny_tensors(/*seed=*/1234);
        const auto dir = make_tiny_checkpoint(
            wq == WqMode::kFp8 ? "quant_fp8" : "quant_nvfp4", tensors);
        auto host = lm::load_dspark_draft(dir);

        auto cfg = runtime_config(dir);
        auto& dc = cfg.speculation.dspark;
        dc.block_size = 8;
        dc.speculative_tokens = 7;
        dc.aux_hidden_state_layer_ids = {0, 1};
        dc.mask_token_id = 5;
        dc.max_anchors = 16;
        dc.draft_vocab_size = 32;
        dc.markov_rank = 4;
        dc.draft_context_capacity_tokens = 256;
        dc.aux_capture_max_rows = 16;
        dc.draft_weights_quant = quant;

        Harness h(cfg);
        auto* rt = h.rt.get();

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
        void* src[2];
        for (int s = 0; s < 2; ++s) {
            for (size_t i = 0; i < bf.size(); ++i)
                bf[i] = f2bf(slot_rows[s][i]);
            src[s] = h.backend->device_alloc(bf.size() * 2);
            ASSERT_NE(src[s], nullptr);
            h.backend->memcpy_h2d(src[s], bf.data(), bf.size() * 2);
        }
        const uint64_t seq = 7;
        rt->capture_aux(0, src[0], ctx_rows, seq, 0, *h.backend, h.stream);
        rt->capture_aux(1, src[1], ctx_rows, seq, 0, *h.backend, h.stream);
        ASSERT_TRUE(rt->ctx_valid());

        const int anchor_token = 3, anchor_pos = 3, nq = 4;
        std::string err;
        ASSERT_TRUE(rt->run_step(seq, anchor_token, anchor_pos, nq, &err))
            << err;
        h.backend->synchronize_device();

        const int rows = nq + rt->sample_off();
        const int V = static_cast<int>(d.V);
        Vec gpu_logits(static_cast<size_t>(rows) * V);
        ASSERT_EQ(cudaMemcpy(gpu_logits.data(), rt->base_logits(),
                             gpu_logits.size() * 4, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        std::vector<uint16_t> gpu_hidden_bf(static_cast<size_t>(rows) * H);
        ASSERT_EQ(cudaMemcpy(gpu_hidden_bf.data(), rt->hidden_out(),
                             gpu_hidden_bf.size() * 2,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);

        Vec aux_concat(static_cast<size_t>(ctx_rows) * 2 * H);
        for (int r = 0; r < ctx_rows; ++r)
            for (int s = 0; s < 2; ++s)
                for (int c = 0; c < H; ++c)
                    aux_concat[(static_cast<size_t>(r) * 2 + s) * H + c] =
                        slot_rows[s][static_cast<size_t>(r) * H + c];
        const CpuRef ref = cpu_reference(host, aux_concat, ctx_rows,
                                         anchor_token, anchor_pos, rows, wq);

        float max_err = 0.0f;
        for (size_t i = 0; i < gpu_logits.size(); ++i) {
            ASSERT_TRUE(std::isfinite(gpu_logits[i])) << "logit " << i;
            const float e = std::abs(gpu_logits[i] - ref.logits[i]);
            max_err = std::max(max_err, e);
            EXPECT_LE(e, 0.06f + 0.05f * std::abs(ref.logits[i]))
                << "logit[" << i << "] gpu=" << gpu_logits[i]
                << " ref=" << ref.logits[i];
        }
        for (size_t i = 0; i < gpu_hidden_bf.size(); ++i) {
            const float g = bf2f(gpu_hidden_bf[i]);
            EXPECT_LE(std::abs(g - ref.hidden[i]),
                      0.06f + 0.05f * std::abs(ref.hidden[i]))
                << "hidden[" << i << "] gpu=" << g << " ref=" << ref.hidden[i];
        }
        SUCCEED() << "max logit err " << max_err;
        for (void* p : src) h.backend->device_free(p);
    }
}

// Real-checkpoint smoke (validation tier 2): quantized upload of the real
// draft, finite base logits + hiddens, DSP-4 draft ids finite AND
// bit-deterministic across identical runs, for both formats.
TEST(DsparkForwardQuantRealCheckpoint, FiniteDeterministicQuantDraft) {
    if (!fs::exists(real_checkpoint_dir() / "model.safetensors"))
        GTEST_SKIP() << "DSpark checkpoint not present";
    if (!has_cuda_gpu()) GTEST_SKIP() << "No CUDA GPU";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    size_t free_b = 0, total_b = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_b, &total_b), cudaSuccess);
    if (free_b < (size_t{7} << 30))
        GTEST_SKIP() << "Insufficient free VRAM: " << free_b;

    for (auto quant : {lc::DsparkDraftWeightsQuant::fp8_e4m3,
                       lc::DsparkDraftWeightsQuant::nvfp4}) {
        SCOPED_TRACE(quant == lc::DsparkDraftWeightsQuant::fp8_e4m3
                         ? "fp8_e4m3"
                         : "nvfp4");
        auto cfg = runtime_config(real_checkpoint_dir());
        cfg.speculation.dspark.draft_context_capacity_tokens = 256;
        cfg.speculation.dspark.aux_capture_max_rows = 16;
        cfg.speculation.dspark.draft_weights_quant = quant;
        Harness h(cfg);
        auto* rt = h.rt.get();

        const int H = 6144, rows = 2;
        std::vector<uint16_t> bf(static_cast<size_t>(rows) * H);
        void* src = h.backend->device_alloc(bf.size() * 2);
        ASSERT_NE(src, nullptr);
        for (int s = 0; s < rt->aux_count(); ++s) {
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < H; ++c)
                    bf[static_cast<size_t>(r) * H + c] = f2bf(
                        0.02f * std::sin(0.37f * c + 1.3f * s + 0.7f * r));
            h.backend->memcpy_h2d(src, bf.data(), bf.size() * 2);
            rt->capture_aux(s, src, rows, /*seq=*/1, /*start_pos=*/0,
                            *h.backend, h.stream);
            h.backend->synchronize_device();
        }
        ASSERT_TRUE(rt->ctx_valid());

        std::string err;
        ASSERT_TRUE(rt->run_step(1, /*anchor=*/290, /*anchor_pos=*/2,
                                 /*num_query=*/0, &err))
            << err;
        ASSERT_TRUE(rt->run_markov_head(&err)) << err;
        h.backend->synchronize_device();
        const int nq = rt->last_num_query();
        ASSERT_GT(nq, 0);

        const int64_t V = 154880;
        std::vector<float> logits(static_cast<size_t>(nq) * V);
        ASSERT_EQ(cudaMemcpy(logits.data(), rt->base_logits(),
                             logits.size() * 4, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        double sum_abs = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
            ASSERT_TRUE(std::isfinite(logits[i])) << "logit " << i;
            sum_abs += std::abs(logits[i]);
        }
        EXPECT_GT(sum_abs, 0.0);
        std::vector<int32_t> ids(static_cast<size_t>(nq));
        ASSERT_EQ(cudaMemcpy(ids.data(), rt->draft_tokens(), ids.size() * 4,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (int32_t id : ids) {
            EXPECT_GE(id, 0);
            EXPECT_LT(id, V);
        }

        // Bit-determinism across an identical re-run.
        ASSERT_TRUE(rt->run_step(1, 290, 2, 0, &err)) << err;
        ASSERT_TRUE(rt->run_markov_head(&err)) << err;
        h.backend->synchronize_device();
        std::vector<float> logits2(logits.size());
        ASSERT_EQ(cudaMemcpy(logits2.data(), rt->base_logits(),
                             logits2.size() * 4, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(std::memcmp(logits.data(), logits2.data(),
                              logits.size() * 4),
                  0);
        std::vector<int32_t> ids2(ids.size());
        ASSERT_EQ(cudaMemcpy(ids2.data(), rt->draft_tokens(), ids2.size() * 4,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(ids, ids2);

        h.backend->device_free(src);
    }
}

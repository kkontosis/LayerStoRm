// GG-7b: GPU numerical test of the absorbed-MLA W_UV value-side GGUF branch.
//
// The MLA value side projects the compressed latent back to V using W_UV (the
// V-half of kv_b_proj, rows [h*(P+V)+P .. +V) per head). The KvBvDequantPool
// extracts + dequants those V rows into a contiguous BF16 buffer [HL, V, D_c]
// that the batched value GEMM consumes. For an FP8 checkpoint this is an
// FP8→BF16 dequant; GG-7b adds the GGUF branch: kv_b_proj is a packed GGUF
// k-quant and the V rows are dequanted per element into BF16 (dequant-only —
// bit-equal in spirit to a load-time dequant, the same philosophy as
// q_absorb's W_UK GGUF branch). NO activation quant / int path.
//
// This test drives the REAL CudaSm120DeviceBackend::kv_bv_extract_dequant on
// hardware with a GGUF-packed kv_b (Q4_K, Q6_K, Q8_0) and compares the BF16
// output to the FP32 dequant of the SAME packed bytes (sliced to the V rows).
// Because the path is dequant-only, the two must agree to cosine >= 0.99999
// (the only difference is the BF16 rounding of each dequanted element).
//
// GPU-required (REQUIRES_GPU); SKIPPED on headless CI.

#include "compute/cuda_sm120_device_backend.h"
#include "compute/kernels/smxx/quant/kv_bv_extract_dequant.h"
#include "core/attention_device.h"
#include "model/quantization/gguf_kquant.h"

#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace ik = layerstorm::compute::cpu::ik;
namespace lc = layerstorm::compute;
namespace lm = layerstorm::model;
namespace cfg = layerstorm::config;

namespace {

float bf2f(__nv_bfloat16 b) { return __bfloat162float(b); }

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += double(a[i]) * b[i];
        na += double(a[i]) * a[i];
        nb += double(b[i]) * b[i];
    }
    if (na == 0 || nb == 0) return 1.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

struct TypeTags {
    ik::GgufType ik_t;
    lm::GgufKQuantType mod_t;
};
constexpr TypeTags kQ4_K{ik::GgufType::q4_k, lm::GgufKQuantType::Q4_K};
constexpr TypeTags kQ6_K{ik::GgufType::q6_k, lm::GgufKQuantType::Q6_K};
constexpr TypeTags kQ8_0{ik::GgufType::q8_0, lm::GgufKQuantType::Q8_0};

cfg::GpuRef make_gpu() { return {0, 0, cfg::GpuType::rtx5090}; }

// Pack `N` random weight rows of `K` cols at `t` (ik reference packer =
// byte-identical to the ggml block layout the GPU kernel consumes) and return
// the parallel FP32 dequant of the SAME bytes.
struct PackedWeight {
    std::vector<uint8_t> bytes;
    std::vector<float> dq;   // [N, K] FP32
    size_t row_bytes;
};
PackedWeight pack_weight(ik::GgufType t, int N, int K, std::mt19937& rng) {
    std::uniform_real_distribution<float> wd(-0.6f, 0.6f);
    PackedWeight p;
    p.row_bytes = ik::weight_row_bytes(t, K);
    p.bytes.resize(static_cast<size_t>(N) * p.row_bytes);
    p.dq.resize(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n) {
        std::vector<float> wr(K);
        for (int k = 0; k < K; ++k) wr[k] = wd(rng);
        void* dst = p.bytes.data() + static_cast<size_t>(n) * p.row_bytes;
        ik::quantize_weight(t, wr.data(), dst, K);
        ik::dequantize_weight(t, dst, p.dq.data() + static_cast<size_t>(n) * K, K);
    }
    return p;
}

void run_case(TypeTags t, int HL, int P, int V, int D_c, double min_cos,
              uint32_t seed) {
    if (!ik::gguf_supported(t.ik_t)) GTEST_SKIP() << "ik type unsupported in build";
    std::mt19937 rng(seed);

    const int kv_row = P + V;        // kv_b_proj per-head row count
    const int N = HL * kv_row;       // kv_b_proj rows = HL*(P+V)

    // kv_b_proj packed as [N, D_c] (each row D_c cols → row-major GGUF blocks).
    PackedWeight w = pack_weight(t.ik_t, N, D_c, rng);

    lc::CudaSm120DeviceBackend dev(make_gpu());
    dev.set_device();

    // Upload packed GGUF bytes.
    std::byte* d_w = nullptr;
    cudaMalloc(&d_w, w.bytes.size());
    cudaMemcpy(d_w, w.bytes.data(), w.bytes.size(), cudaMemcpyHostToDevice);

    // Output [HL, V, D_c] BF16.
    const size_t out_n = static_cast<size_t>(HL) * V * D_c;
    __nv_bfloat16* d_out = nullptr;
    cudaMalloc(&d_out, out_n * sizeof(__nv_bfloat16));
    cudaMemset(d_out, 0, out_n * sizeof(__nv_bfloat16));

    lc::KvBvExtractDequantParams p{};
    p.num_heads_local = HL;
    p.qk_nope_head_dim = P;
    p.v_head_dim = V;
    p.kv_lora_rank = D_c;
    p.kv_b_proj = d_w;
    p.scales = nullptr;                       // GGUF: scales are in-block
    p.output = d_out;
    p.gguf_type = static_cast<int>(t.mod_t);

    dev.kv_bv_extract_dequant(p, nullptr);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> out(out_n);
    cudaMemcpy(out.data(), d_out, out_n * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);

    // FP32 reference: the V rows of the SAME dequanted weight.
    //   ref[h, v, c] = dq[(h*(P+V) + P + v) * D_c + c]
    std::vector<float> ref(out_n), gpu(out_n);
    for (int h = 0; h < HL; ++h)
        for (int v = 0; v < V; ++v) {
            const size_t src_row = static_cast<size_t>(h) * kv_row + P + v;
            const size_t dst_row = static_cast<size_t>(h) * V + v;
            for (int c = 0; c < D_c; ++c) {
                ref[dst_row * D_c + c] = w.dq[src_row * D_c + c];
                gpu[dst_row * D_c + c] = bf2f(out[dst_row * D_c + c]);
            }
        }

    const double cos = cosine(gpu, ref);
    EXPECT_GE(cos, min_cos) << "GGUF V-extract vs FP32 dequant cos=" << cos;

    // Per-element BF16-rounding check (catches a structurally-broken kernel that
    // still happens to correlate): each output is within BF16 ULP of the FP32
    // dequant of the same element.
    int bad = 0;
    for (size_t i = 0; i < out_n; ++i) {
        const float r = ref[i];
        const float diff = std::fabs(gpu[i] - r);
        const float denom = std::max(std::fabs(r), 1e-6f);
        if (diff / denom > 1e-2f && diff > 1e-2f && bad++ < 5)
            ADD_FAILURE() << "elem " << i << " gpu=" << gpu[i] << " ref=" << r;
    }

    cudaFree(d_w);
    cudaFree(d_out);
}

}  // namespace

// DeepSeek-V3.2-like absorbed-MLA shapes: P=128, V=128, D_c=512.
// D_c=512 satisfies QK=256 (k-quants) and QK=32 (Q8_0).
TEST(KvBvExtractDequantGgufGpu, Q4K_DequantOnlyMatchesFp32) {
    REQUIRES_GPU();
    run_case(kQ4_K, /*HL=*/4, /*P=*/128, /*V=*/128, /*D_c=*/512,
             /*min_cos=*/0.99999, 0x5B01);
}
TEST(KvBvExtractDequantGgufGpu, Q6K_DequantOnlyMatchesFp32) {
    REQUIRES_GPU();
    run_case(kQ6_K, /*HL=*/4, /*P=*/128, /*V=*/128, /*D_c=*/512,
             /*min_cos=*/0.99999, 0x5B02);
}
TEST(KvBvExtractDequantGgufGpu, Q8_0_DequantOnlyMatchesFp32) {
    REQUIRES_GPU();
    // QK=32 family — exercises the blocks_per_row = D_c/32 = 16 path.
    run_case(kQ8_0, /*HL=*/3, /*P=*/128, /*V=*/128, /*D_c=*/512,
             /*min_cos=*/0.99999, 0x5B03);
}
// GLM-style asymmetric head dims (P != V): V rows still slice correctly.
TEST(KvBvExtractDequantGgufGpu, Q4K_AsymmetricHeadDims) {
    REQUIRES_GPU();
    run_case(kQ4_K, /*HL=*/2, /*P=*/192, /*V=*/256, /*D_c=*/512,
             /*min_cos=*/0.99999, 0x5B04);
}

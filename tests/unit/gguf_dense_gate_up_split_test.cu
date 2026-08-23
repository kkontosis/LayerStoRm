// GG-5c: GPU numerical test of the dense/shared gate_up fused-vs-split path.
//
// Resolves TD-GG5C-SPLIT-E2E-UNCOVERED. The CUDA-free
// GgufFusedMoeGateUp.SplitWhenGateDiffersFromUp test pins only the GEMM
// EMISSION (which GEMMs, types, the up-block offset) via a mock device. It does
// NOT validate NUMERICS. This test drives the REAL helper
// daemon::launch_gguf_dense_gate_up (dispatch_detail.h) on actual hardware with
// a real CudaSm120ExpertDevice (gguf_grouped_gemm) + CudaSm120DeviceBackend
// (the single_b_ptr H2D rebind + the 2x memcpy_2d interleave), then checks the
// pre-SwiGLU gate_up_output[M, 2*I] against a per-projection BF16-dequant
// reference. This exercises the two num_experts==1 GEMMs at the computed up-block
// offset (gguf::gguf_packed_bytes(I,H,gate_type)) and the interleave into the
// [.,2*I] buffer — the split's real, previously-unverified risk.
//
// Cases: SPLIT with gate_type != up_type for two distinct type pairs that vary
// QK + block size + offset — Q4_K(gate)/Q6_K(up) (QK=256) and Q8_0(gate)/Q4_K(up)
// (QK=32 vs 256) — under BOTH int (cosine >= 0.999, Q8_1 act-quant noise) and
// dequant (cosine >= 0.9999) strategies, asserting EACH half independently and
// cross-checking that decoding the UP half as the GATE type yields a WORSE cosine
// (catches a wrong type/offset). Plus a FUSED case (gate_type == up_type)
// asserting parity with a single [2*I, H] GEMM reference.
//
// GPU-required (REQUIRES_GPU); SKIPPED on headless CI.

#include "daemon/dispatch_detail.h"
#include "core/expert_device.h"
#include "compute/cuda_sm120_expert_device.h"
#include "compute/cuda_sm120_device_backend.h"
#include "model/quantization/gguf_kquant.h"

#include "sm120/gemm/gguf/gguf_grouped_gemm.h"   // gguf_grouped_gemm_workspace_bytes
#include "sm120/gemm/gguf/gguf_dequant_gemm.h"   // gguf_block_bytes/values

#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace ik = layerstorm::compute::cpu::ik;
namespace lc = layerstorm::compute;
namespace lm = layerstorm::model;
namespace ld = layerstorm::daemon;
namespace cfg = layerstorm::config;

namespace {

__nv_bfloat16 f2bf(float f) { return __float2bfloat16(f); }
float bf2f(__nv_bfloat16 b) { return __bfloat162float(b); }

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

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

// The three parallel type tags for one GGUF family: the CPU ik packer tag, the
// engine GgufQuantType (drives the helper / kernel), and the model
// GgufKQuantType (drives the host byte-offset math).
struct TypeTags {
    ik::GgufType ik_t;
    lc::GgufQuantType eng_t;
    lm::GgufKQuantType mod_t;
};

constexpr TypeTags kQ4_K{ik::GgufType::q4_k, lc::GgufQuantType::Q4_K, lm::GgufKQuantType::Q4_K};
constexpr TypeTags kQ6_K{ik::GgufType::q6_k, lc::GgufQuantType::Q6_K, lm::GgufKQuantType::Q6_K};
constexpr TypeTags kQ8_0{ik::GgufType::q8_0, lc::GgufQuantType::Q8_0, lm::GgufKQuantType::Q8_0};

cfg::GpuRef make_gpu() { return {0, 0, cfg::GpuType::rtx5090}; }

// Pack `N` random weight rows of `K` cols at `t` precision into a host byte
// buffer (ik reference packer = byte-identical to the ggml block layout the GPU
// kernel consumes), and return the parallel FP32 dequant for the reference.
struct PackedWeight {
    std::vector<uint8_t> bytes;
    std::vector<float> dq;        // [N, K] FP32 dequant of the SAME packed bytes
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

// Dequantize `src` (N rows, stride `src_row_bytes`) AS IF it were type
// `decode_as`. Each row is copied into a zero-padded buffer sized to
// weight_row_bytes(decode_as, K) so a cross-type decode (the wrong-half check)
// never reads out of bounds when decode_as needs more bytes/row than src has.
std::vector<float> dequant_rows(ik::GgufType decode_as, const uint8_t* src,
                                size_t src_row_bytes, int N, int K) {
    const size_t need = ik::weight_row_bytes(decode_as, K);
    std::vector<float> out(static_cast<size_t>(N) * K);
    std::vector<uint8_t> rowbuf(need, 0);
    const size_t copy = std::min(need, src_row_bytes);
    for (int n = 0; n < N; ++n) {
        std::memset(rowbuf.data(), 0, need);
        std::memcpy(rowbuf.data(), src + static_cast<size_t>(n) * src_row_bytes, copy);
        ik::dequantize_weight(decode_as, rowbuf.data(), out.data() + static_cast<size_t>(n) * K, K);
    }
    return out;
}

// ref[m,n] = sum_k A_f[m,k] * Wdq[n,k]   (one projection: [M,K] x [N,K]^T -> [M,N])
std::vector<float> matmul_ref(const std::vector<float>& A_f, const std::vector<float>& Wdq,
                              int M, int N, int K) {
    std::vector<float> out(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            const float* ar = A_f.data() + static_cast<size_t>(m) * K;
            const float* wr = Wdq.data() + static_cast<size_t>(n) * K;
            for (int k = 0; k < K; ++k) acc += double(ar[k]) * wr[k];
            out[static_cast<size_t>(m) * N + n] = float(acc);
        }
    return out;
}

// Random BF16 activations [M, K]; returns the BF16 device-upload vector and the
// bf-rounded FP32 mirror for the reference.
struct Activations {
    std::vector<__nv_bfloat16> bf;
    std::vector<float> f;
};
Activations make_acts(int M, int K, std::mt19937& rng) {
    std::uniform_real_distribution<float> ad(-1.0f, 1.0f);
    Activations a;
    a.bf.resize(static_cast<size_t>(M) * K);
    a.f.resize(static_cast<size_t>(M) * K);
    for (size_t i = 0; i < a.f.size(); ++i) {
        a.bf[i] = f2bf(ad(rng));
        a.f[i] = bf2f(a.bf[i]);
    }
    return a;
}

// Slice the [M, 2*I] gate_up_output into the gate ([0,I)) or up ([I,2I)) half.
std::vector<float> half(const std::vector<float>& full, int M, int I, bool up) {
    std::vector<float> out(static_cast<size_t>(M) * I);
    const int base = up ? I : 0;
    for (int m = 0; m < M; ++m)
        for (int i = 0; i < I; ++i)
            out[static_cast<size_t>(m) * I + i] = full[static_cast<size_t>(m) * 2 * I + base + i];
    return out;
}

// ── SPLIT case: gate_type != up_type ────────────────────────────────────────
// Packs gate (type A) then up (type B) CONTIGUOUSLY (up block begins at
// gguf_packed_bytes(I,H,gate)), runs the REAL helper, checks each half + the
// wrong-type cross-check.
void run_split_case(TypeTags gate, TypeTags up, lc::GgufGemmStrategy strat,
                    int M, int I, int H, double min_cos, uint32_t seed) {
    if (!ik::gguf_supported(gate.ik_t) || !ik::gguf_supported(up.ik_t)) {
        GTEST_SKIP() << "ik type unsupported in this build";
    }
    std::mt19937 rng(seed);

    PackedWeight gp = pack_weight(gate.ik_t, I, H, rng);
    PackedWeight upk = pack_weight(up.ik_t, I, H, rng);

    // The up block lives immediately after the gate block. The model byte
    // primitive MUST agree with the ik packer's total gate-block size.
    const int64_t up_off = lm::gguf::gguf_packed_bytes(I, H, gate.mod_t);
    ASSERT_EQ(up_off, static_cast<int64_t>(gp.bytes.size()));
    // Cross-check the GPU kernel's expected packed-row size matches ik's (both
    // halves). The block helpers take the kernel GgufType (same ordinal order).
    const auto gate_krn = static_cast<lc::GgufType>(static_cast<int>(gate.eng_t));
    const auto up_krn = static_cast<lc::GgufType>(static_cast<int>(up.eng_t));
    ASSERT_EQ(static_cast<int>(gp.row_bytes),
              (H / lc::gguf_block_values(gate_krn)) * lc::gguf_block_bytes(gate_krn));
    ASSERT_EQ(static_cast<int>(upk.row_bytes),
              (H / lc::gguf_block_values(up_krn)) * lc::gguf_block_bytes(up_krn));

    Activations act = make_acts(M, H, rng);

    // ── Devices (real GPU) ───────────────────────────────────────────────────
    auto edev = lc::make_cuda_sm120_expert_device(make_gpu());
    auto bdev = lc::make_cuda_sm120_device_backend(make_gpu());
    edev->set_device();

    // ── Device buffers ───────────────────────────────────────────────────────
    __nv_bfloat16* dA = upload(act.bf);

    // Contiguous gate||up weight blob (gate at 0, up at up_off).
    std::byte* dW = nullptr;
    cudaMalloc(&dW, gp.bytes.size() + upk.bytes.size());
    cudaMemcpy(dW, gp.bytes.data(), gp.bytes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dW + gp.bytes.size(), upk.bytes.data(), upk.bytes.size(),
               cudaMemcpyHostToDevice);

    std::vector<__nv_bfloat16> zero_gu(static_cast<size_t>(M) * 2 * I, f2bf(0.0f));
    std::vector<__nv_bfloat16> zero_half(static_cast<size_t>(M) * I, f2bf(0.0f));
    __nv_bfloat16* dGU = upload(zero_gu);
    __nv_bfloat16* dGate = upload(zero_half);
    __nv_bfloat16* dUp = upload(zero_half);

    std::vector<int32_t> offsets = {0, M};
    int32_t* dOff = upload(offsets);
    void** dBptr = nullptr;
    cudaMalloc(&dBptr, sizeof(void*));

    void* ws = nullptr;
    size_t ws_bytes = 0;
    if (strat == lc::GgufGemmStrategy::int_strategy) {
        ws_bytes = lc::gguf_grouped_gemm_workspace_bytes(M, H, /*num_experts=*/1);
        cudaMalloc(&ws, ws_bytes);
    }

    // ── Drive the REAL helper ────────────────────────────────────────────────
    ld::GgufDenseGateUpArgs a{};
    a.num_tokens = M;
    a.intermediate_local = I;
    a.hidden = H;
    a.a_base = dA;
    a.gate_up_output = dGU;
    a.gate_scratch = dGate;
    a.up_scratch = dUp;
    a.gate_type = gate.eng_t;
    a.up_type = up.eng_t;
    a.up_block_offset = up_off;
    a.strategy = strat;
    a.gate_up_weight = dW;
    a.single_b_ptr = dBptr;
    a.expert_offsets = dOff;
    a.dev = edev.get();
    a.gpu_dev = bdev.get();
    a.gemm_workspace = ws;
    a.gemm_workspace_bytes = ws_bytes;
    a.stream = nullptr;

    const int n = ld::launch_gguf_dense_gate_up(a);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(n, 2) << "gate_type != up_type must SPLIT into two GEMMs";

    std::vector<__nv_bfloat16> gu_bf(zero_gu.size());
    cudaMemcpy(gu_bf.data(), dGU, gu_bf.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    std::vector<float> gu(gu_bf.size());
    for (size_t i = 0; i < gu.size(); ++i) gu[i] = bf2f(gu_bf[i]);

    // ── Reference + assertions (per half) ────────────────────────────────────
    std::vector<float> ref_gate = matmul_ref(act.f, gp.dq, M, I, H);   // type A
    std::vector<float> ref_up = matmul_ref(act.f, upk.dq, M, I, H);    // type B
    std::vector<float> out_gate = half(gu, M, I, /*up=*/false);
    std::vector<float> out_up = half(gu, M, I, /*up=*/true);

    const double cos_gate = cosine(out_gate, ref_gate);
    const double cos_up = cosine(out_up, ref_up);
    EXPECT_GE(cos_gate, min_cos) << "GATE half (type A) cos=" << cos_gate;
    EXPECT_GE(cos_up, min_cos) << "UP half (type B) cos=" << cos_up;

    // Wrong-half cross-check: decode the UP bytes AS THE GATE TYPE. A wrong type
    // tag (or a single-type decode of both halves) would make this the matching
    // reference. The correct type-B decode passed `min_cos` above; the wrong
    // type-A decode must NOT — reinterpreting the up bytes through the gate type's
    // block layout / fp16 scales gives garbage (a low cosine, or NaN/Inf from a
    // bogus scale bit-pattern — both clear misses; NaN >= min_cos is false). The
    // gap between cos_up (>= min_cos) and this proves the per-half type is honored.
    std::vector<float> up_dq_as_gate =
        dequant_rows(gate.ik_t, upk.bytes.data(), upk.row_bytes, I, H);
    std::vector<float> ref_up_wrong = matmul_ref(act.f, up_dq_as_gate, M, I, H);
    const double cos_up_wrong = cosine(out_up, ref_up_wrong);
    EXPECT_FALSE(cos_up_wrong >= min_cos)
        << "UP half ALSO matched the gate-type decode (cos_A=" << cos_up_wrong
        << ") — type tag not honored (cos_B=" << cos_up << ")";
    // Offset cross-check: the up half must NOT match the (independent) gate
    // weights — i.e. the helper read the up block at base+offset, not at base.
    const double cos_up_vs_gate = cosine(out_up, ref_gate);
    EXPECT_GT(cos_up, cos_up_vs_gate + 0.01)
        << "UP half matched the GATE weights — wrong up-block offset? cos_up="
        << cos_up << " cos_up_vs_gate=" << cos_up_vs_gate;

    cudaFree(dA); cudaFree(dW); cudaFree(dGU); cudaFree(dGate); cudaFree(dUp);
    cudaFree(dOff); cudaFree(dBptr);
    if (ws) cudaFree(ws);
}

// ── FUSED case: gate_type == up_type ────────────────────────────────────────
// One [2*I, H] GEMM with the shared type; parity vs a single [2*I,H] reference
// (the fast path must still be correct).
void run_fused_case(TypeTags t, lc::GgufGemmStrategy strat,
                    int M, int I, int H, double min_cos, uint32_t seed) {
    if (!ik::gguf_supported(t.ik_t)) GTEST_SKIP() << "ik type unsupported";
    std::mt19937 rng(seed);

    // One [2*I, H] weight (gate rows then up rows — same type ⇒ one block).
    PackedWeight w = pack_weight(t.ik_t, 2 * I, H, rng);
    Activations act = make_acts(M, H, rng);

    auto edev = lc::make_cuda_sm120_expert_device(make_gpu());
    auto bdev = lc::make_cuda_sm120_device_backend(make_gpu());
    edev->set_device();

    __nv_bfloat16* dA = upload(act.bf);
    std::byte* dW = nullptr;
    cudaMalloc(&dW, w.bytes.size());
    cudaMemcpy(dW, w.bytes.data(), w.bytes.size(), cudaMemcpyHostToDevice);

    std::vector<__nv_bfloat16> zero_gu(static_cast<size_t>(M) * 2 * I, f2bf(0.0f));
    std::vector<__nv_bfloat16> zero_half(static_cast<size_t>(M) * I, f2bf(0.0f));
    __nv_bfloat16* dGU = upload(zero_gu);
    __nv_bfloat16* dGate = upload(zero_half);
    __nv_bfloat16* dUp = upload(zero_half);

    std::vector<int32_t> offsets = {0, M};
    int32_t* dOff = upload(offsets);
    void** dBptr = nullptr;
    cudaMalloc(&dBptr, sizeof(void*));

    void* ws = nullptr;
    size_t ws_bytes = 0;
    if (strat == lc::GgufGemmStrategy::int_strategy) {
        ws_bytes = lc::gguf_grouped_gemm_workspace_bytes(M, H, /*num_experts=*/1);
        cudaMalloc(&ws, ws_bytes);
    }

    ld::GgufDenseGateUpArgs a{};
    a.num_tokens = M;
    a.intermediate_local = I;
    a.hidden = H;
    a.a_base = dA;
    a.gate_up_output = dGU;
    a.gate_scratch = dGate;
    a.up_scratch = dUp;
    a.gate_type = t.eng_t;
    a.up_type = t.eng_t;       // fused
    a.up_block_offset = 0;     // unused on the fused path
    a.strategy = strat;
    a.gate_up_weight = dW;
    a.single_b_ptr = dBptr;
    a.expert_offsets = dOff;
    a.dev = edev.get();
    a.gpu_dev = bdev.get();
    a.gemm_workspace = ws;
    a.gemm_workspace_bytes = ws_bytes;
    a.stream = nullptr;

    const int n = ld::launch_gguf_dense_gate_up(a);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(n, 1) << "gate_type == up_type must take the FUSED single GEMM";

    std::vector<__nv_bfloat16> gu_bf(zero_gu.size());
    cudaMemcpy(gu_bf.data(), dGU, gu_bf.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    std::vector<float> gu(gu_bf.size());
    for (size_t i = 0; i < gu.size(); ++i) gu[i] = bf2f(gu_bf[i]);

    // Single [2*I, H] GEMM reference over the full output.
    std::vector<float> ref = matmul_ref(act.f, w.dq, M, 2 * I, H);
    const double cos = cosine(gu, ref);
    EXPECT_GE(cos, min_cos) << "fused [2*I,H] parity cos=" << cos;

    cudaFree(dA); cudaFree(dW); cudaFree(dGU); cudaFree(dGate); cudaFree(dUp);
    cudaFree(dOff); cudaFree(dBptr);
    if (ws) cudaFree(ws);
}

}  // namespace

// ── SPLIT: Q4_K(gate) / Q6_K(up) — QK=256 both, distinct block size + offset ──
TEST(GgufDenseGateUpSplitGpu, Q4K_Q6K_Int) {
    REQUIRES_GPU();
    // M=16 > 8 exercises the int mmq_mma tile-map path for each split half.
    run_split_case(kQ4_K, kQ6_K, lc::GgufGemmStrategy::int_strategy,
                   /*M=*/16, /*I=*/256, /*H=*/512, 0.999, 0x5C01);
}
TEST(GgufDenseGateUpSplitGpu, Q4K_Q6K_Dequant) {
    REQUIRES_GPU();
    run_split_case(kQ4_K, kQ6_K, lc::GgufGemmStrategy::dequant,
                   /*M=*/8, /*I=*/256, /*H=*/512, 0.9999, 0x5C02);
}

// ── SPLIT: Q8_0(gate) / Q4_K(up) — QK=32 vs 256, different offset math ────────
TEST(GgufDenseGateUpSplitGpu, Q8_0_Q4K_Int) {
    REQUIRES_GPU();
    // M=8 hits the int mmvq path (M_e <= 8).
    run_split_case(kQ8_0, kQ4_K, lc::GgufGemmStrategy::int_strategy,
                   /*M=*/8, /*I=*/256, /*H=*/512, 0.999, 0x5C03);
}
TEST(GgufDenseGateUpSplitGpu, Q8_0_Q4K_Dequant) {
    REQUIRES_GPU();
    run_split_case(kQ8_0, kQ4_K, lc::GgufGemmStrategy::dequant,
                   /*M=*/12, /*I=*/256, /*H=*/512, 0.9999, 0x5C04);
}

// ── FUSED parity: gate_type == up_type equals a single [2*I, H] GEMM ──────────
TEST(GgufDenseGateUpSplitGpu, FusedQ4K_Int) {
    REQUIRES_GPU();
    run_fused_case(kQ4_K, lc::GgufGemmStrategy::int_strategy,
                   /*M=*/16, /*I=*/256, /*H=*/512, 0.999, 0x5C05);
}
TEST(GgufDenseGateUpSplitGpu, FusedQ4K_Dequant) {
    REQUIRES_GPU();
    run_fused_case(kQ4_K, lc::GgufGemmStrategy::dequant,
                   /*M=*/8, /*I=*/256, /*H=*/512, 0.9999, 0x5C06);
}

// Shared CPU GGUF grouped-GEMM driver (C-6). See cpu_gguf_gemm.h.
//
// CPU-only TU — NO CUDA (INV-GPU-1). The ik GEMM kernels are vendored verbatim
// under compute/cpu/ik_vendor; this file is the LayerStoRm threaded driver glue.

#include "compute/cpu/cpu_gguf_gemm.h"

#include "compute/cpu/numa_thread_pool.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace layerstorm::compute::cpu {

namespace {

inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    bits += 0x7FFF + ((bits >> 16) & 1);  // round-to-nearest-even
    return static_cast<uint16_t>(bits >> 16);
}

// Per-thread output-row slice of `nrows` rows: for each active expert, run the
// verbatim ik GEMM over weight rows [w_row0, w_row0+nrows) of that expert's slab
// against the expert's quantized tokens, then scatter FP32 -> BF16 into D columns
// [d_col0, d_col0+nrows). For the single-node path w_row0 == d_col0 (the slab is
// the full weight matrix). For a multi-node row-slice, the slab base already
// starts at the slice (w_row0 == 0) while d_col0 == the slice's global N offset.
void gemm_rows(uint16_t* D, ik::GgufType type, const void* const* B_ptrs,
               const uint8_t* quant_a, size_t act_rb,
               const int32_t* expert_offsets, int num_experts,
               int N, int K, int w_row0, int d_col0, int nrows,
               std::vector<float>& fp32_out) {
    if (nrows <= 0) return;
    const size_t wrow = ik::weight_row_bytes(type, K);
    for (int e = 0; e < num_experts; ++e) {
        const int m0 = expert_offsets[e];
        const int m1 = expert_offsets[e + 1];
        const int my = m1 - m0;
        if (my <= 0) continue;
        const auto* W = static_cast<const uint8_t*>(B_ptrs[e]) +
                        static_cast<size_t>(w_row0) * wrow;
        const void* A = quant_a + static_cast<size_t>(m0) * act_rb;
        if (fp32_out.size() < static_cast<size_t>(nrows) * my)
            fp32_out.resize(static_cast<size_t>(nrows) * my);
        // C[ix, iy] at C + iy*stride_C + ix; stride_C = nrows => [my, nrows].
        if (!ik::gguf_gemm_one(type, nrows, my, K, W, A, fp32_out.data(),
                               /*stride_C=*/nrows)) {
            throw std::runtime_error(
                "cpu_gguf_grouped_gemm: ik kernel unsupported for type/host");
        }
        for (int iy = 0; iy < my; ++iy) {
            const int m = m0 + iy;
            uint16_t* drow = D + static_cast<size_t>(m) * N;
            const float* src = fp32_out.data() + static_cast<size_t>(iy) * nrows;
            for (int ix = 0; ix < nrows; ++ix)
                drow[d_col0 + ix] = f32_to_bf16(src[ix]);
        }
    }
}

}  // namespace

void cpu_gguf_grouped_gemm(uint16_t* D, const uint16_t* A,
                           ik::GgufType type, const void* const* B_ptrs,
                           const int32_t* expert_offsets, int num_experts,
                           int N, int K, NumaThreadPool* pool) {
    if (num_experts == 0 || N == 0) return;
    if (!B_ptrs)
        throw std::runtime_error("cpu_gguf_grouped_gemm requires B_ptrs");
    if (!ik::gguf_supported(type))
        throw std::runtime_error(
            "cpu_gguf_grouped_gemm: type unsupported on this build/host");

    const int total_tokens = expert_offsets[num_experts];
    if (total_tokens == 0) return;

    // BF16 acts -> FP32 -> ik activation format (once, all rows).
    const size_t act_rb = ik::activation_row_bytes(type, K);
    std::vector<uint8_t> quant_a(static_cast<size_t>(total_tokens) * act_rb);
    std::vector<float> fa(static_cast<size_t>(K));
    for (int m = 0; m < total_tokens; ++m) {
        const uint16_t* arow = A + static_cast<size_t>(m) * K;
        for (int k = 0; k < K; ++k) fa[k] = bf16_to_f32(arow[k]);
        ik::quantize_activations(type, fa.data(),
                                 quant_a.data() + static_cast<size_t>(m) * act_rb,
                                 1, K);
    }

    if (pool == nullptr || pool->num_threads() <= 1) {
        std::vector<float> fp32_out;
        gemm_rows(D, type, B_ptrs, quant_a.data(), act_rb, expert_offsets,
                  num_experts, N, K, /*w_row0=*/0, /*d_col0=*/0, /*nrows=*/N,
                  fp32_out);
        return;
    }
    pool->parallel_for([&](int ith, int nth, CpuBarrierState& /*b*/) {
        const int base = N / nth, rem = N % nth;
        const int n0 = ith * base + std::min(ith, rem);
        const int n1 = n0 + base + (ith < rem ? 1 : 0);
        std::vector<float> fp32_out;
        gemm_rows(D, type, B_ptrs, quant_a.data(), act_rb, expert_offsets,
                  num_experts, N, K, /*w_row0=*/n0, /*d_col0=*/n0,
                  /*nrows=*/n1 - n0, fp32_out);
    });
}

// ── Multi-NUMA split-N building blocks (TD-IK-MULTINODE) ─────────────────────

void cpu_gguf_quantize_activations(const uint16_t* A, ik::GgufType type,
                                   int total_tokens, int K, uint8_t* quant_a_out) {
    if (total_tokens == 0 || K == 0) return;
    if (!A || !quant_a_out)
        throw std::runtime_error("cpu_gguf_quantize_activations: null buffer");
    if (!ik::gguf_supported(type))
        throw std::runtime_error(
            "cpu_gguf_quantize_activations: type unsupported on this build/host");
    const size_t act_rb = ik::activation_row_bytes(type, K);
    std::vector<float> fa(static_cast<size_t>(K));
    for (int m = 0; m < total_tokens; ++m) {
        const uint16_t* arow = A + static_cast<size_t>(m) * K;
        for (int k = 0; k < K; ++k) fa[k] = bf16_to_f32(arow[k]);
        ik::quantize_activations(type, fa.data(),
                                 quant_a_out + static_cast<size_t>(m) * act_rb,
                                 1, K);
    }
}

void cpu_gguf_grouped_gemm_slice(uint16_t* D, const uint8_t* quant_a,
                                 ik::GgufType type, const void* const* B_slabs,
                                 const int32_t* expert_offsets, int num_experts,
                                 int N, int K, int n0, int nslice,
                                 NumaThreadPool* pool) {
    if (num_experts == 0 || nslice <= 0) return;
    if (!B_slabs || !quant_a)
        throw std::runtime_error("cpu_gguf_grouped_gemm_slice requires buffers");
    if (!ik::gguf_supported(type))
        throw std::runtime_error(
            "cpu_gguf_grouped_gemm_slice: type unsupported on this build/host");
    const int total_tokens = expert_offsets[num_experts];
    if (total_tokens == 0) return;
    const size_t act_rb = ik::activation_row_bytes(type, K);

    // The per-expert slabs already start at the slice (local row 0 == global n0);
    // weight read offset is therefore 0, output columns start at n0.
    if (pool == nullptr || pool->num_threads() <= 1) {
        std::vector<float> fp32_out;
        gemm_rows(D, type, B_slabs, quant_a, act_rb, expert_offsets, num_experts,
                  N, K, /*w_row0=*/0, /*d_col0=*/n0, /*nrows=*/nslice, fp32_out);
        return;
    }
    pool->parallel_for([&](int ith, int nth, CpuBarrierState& /*b*/) {
        const int base = nslice / nth, rem = nslice % nth;
        const int r0 = ith * base + std::min(ith, rem);
        const int r1 = r0 + base + (ith < rem ? 1 : 0);
        std::vector<float> fp32_out;
        gemm_rows(D, type, B_slabs, quant_a, act_rb, expert_offsets, num_experts,
                  N, K, /*w_row0=*/r0, /*d_col0=*/n0 + r0, /*nrows=*/r1 - r0,
                  fp32_out);
    });
}

}  // namespace layerstorm::compute::cpu

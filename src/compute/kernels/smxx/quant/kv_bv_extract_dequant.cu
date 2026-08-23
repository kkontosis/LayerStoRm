// FP8 → BF16 strided dequantization kernel for kv_b_proj V portion.
//
// One CUDA block per (head, v_row) pair.  Each thread processes multiple
// columns (kv_lora_rank dimension) in a strided loop.  Reads FP8 values
// from the interleaved kv_b_proj layout and writes contiguous BF16 output.

#include "compute/kernels/smxx/quant/kv_bv_extract_dequant.h"

#include "formats/gguf_dequant_one.h"  // GG-7b: per-element GGUF dequant policies

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

static constexpr int kThreadsPerBlock = 256;
static constexpr int kScaleBlockSize = 128;

/// Convert FP8 E4M3 byte to float32.
__device__ __forceinline__
float fp8e4m3_to_float(uint8_t bits) {
    __nv_fp8_e4m3 fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&bits);
    return float(fp8);
}

__global__ void __launch_bounds__(kThreadsPerBlock)
kv_bv_extract_dequant_kernel(
    __nv_bfloat16* __restrict__ output,      // [HL, V, D_c] row-major
    const uint8_t* __restrict__ fp8_src,      // [HL*(P+V), D_c] row-major
    const float*   __restrict__ scales,       // [ceil(D_c/128), ceil(N_total/128)] K-major
    int HL, int P, int V, int D_c,
    int N_total,                              // = HL * (P + V)
    int N_scale_blocks)                       // = ceil(N_total / 128)
{
    // Block (head, v_row) mapping.
    const int block_id = blockIdx.x;
    const int head = block_id / V;
    const int v_row = block_id % V;
    if (head >= HL) return;

    // Source row in the full kv_b_proj: skip P (K_nope) rows per head.
    const int src_row = head * (P + V) + P + v_row;
    const uint8_t* __restrict__ src_row_ptr = fp8_src +
        static_cast<int64_t>(src_row) * D_c;

    // N-block index for this source row (for scale lookup).
    const int n_block = src_row / kScaleBlockSize;

    // Destination row in contiguous output.
    const int dst_row = head * V + v_row;
    __nv_bfloat16* __restrict__ dst_row_ptr = output +
        static_cast<int64_t>(dst_row) * D_c;

    // Process columns with strided loop.
    for (int c = threadIdx.x; c < D_c; c += kThreadsPerBlock) {
        // Read FP8 value.
        float val = fp8e4m3_to_float(__ldg(&src_row_ptr[c]));

        // Blockwise scale: scales[k_block * N_scale_blocks + n_block]
        const int k_block = c / kScaleBlockSize;
        float scale = __ldg(&scales[k_block * N_scale_blocks + n_block]);

        dst_row_ptr[c] = __float2bfloat16(val * scale);
    }
}

// ---------------------------------------------------------------------------
// GG-7b: GGUF W_UV value-side path. kv_b_proj is a packed GGUF weight; the V
// rows are dequanted per element into BF16 (dequant-only — bit-equal to a
// load-time dequant). MIRRORS q_absorb's W_UK GGUF branch (same packed-row
// math), just sliced to the V rows (src_row = head*(P+V) + P + v_row). Each
// packed row carries (D_c/VALS) super-blocks; dequant_q*_one reads block
// (src_row*blocks_per_row + kb) at position p = c % VALS. GGUF scales are
// in-block, so there is no separate scale region.
// ---------------------------------------------------------------------------

namespace gguf_pol {
namespace fmt = layerstorm::formats;
struct Q2K { static constexpr int VALS = 256, BYTES = fmt::BLOCK_Q2K_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q2k_one(*(const fmt::block_q2_K*)b, p);} };
struct Q3K { static constexpr int VALS = 256, BYTES = fmt::BLOCK_Q3K_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q3k_one(*(const fmt::block_q3_K*)b, p);} };
struct Q4K { static constexpr int VALS = 256, BYTES = fmt::BLOCK_Q4K_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q4k_one(*(const fmt::block_q4_K*)b, p);} };
struct Q5K { static constexpr int VALS = 256, BYTES = fmt::BLOCK_Q5K_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q5k_one(*(const fmt::block_q5_K*)b, p);} };
struct Q6K { static constexpr int VALS = 256, BYTES = fmt::BLOCK_Q6K_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q6k_one(*(const fmt::block_q6_K*)b, p);} };
struct Q8_0 { static constexpr int VALS = 32, BYTES = fmt::BLOCK_Q8_0_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q8_0_one(*(const fmt::block_q8_0*)b, p);} };
}  // namespace gguf_pol

template <class P_>
__global__ void __launch_bounds__(kThreadsPerBlock)
kv_bv_extract_dequant_gguf_kernel(
    __nv_bfloat16* __restrict__ output,        // [HL, V, D_c] row-major
    const uint8_t* __restrict__ gguf_src,      // [HL*(P+V), (D_c/QK)*bytes] packed
    int HL, int P, int V, int D_c)
{
    const int block_id = blockIdx.x;
    const int head = block_id / V;
    const int v_row = block_id % V;
    if (head >= HL) return;

    // Packed source row (same V-row slice as the FP8 path: skip P K-rows/head).
    const int src_row = head * (P + V) + P + v_row;
    const int blocks_per_row = D_c / P_::VALS;
    const uint8_t* __restrict__ src_row_ptr = gguf_src +
        static_cast<int64_t>(src_row) * blocks_per_row * P_::BYTES;

    const int dst_row = head * V + v_row;
    __nv_bfloat16* __restrict__ dst_row_ptr = output +
        static_cast<int64_t>(dst_row) * D_c;

    for (int c = threadIdx.x; c < D_c; c += kThreadsPerBlock) {
        const int kb = c / P_::VALS;
        const int p  = c % P_::VALS;
        const void* blk = src_row_ptr + static_cast<int64_t>(kb) * P_::BYTES;
        dst_row_ptr[c] = __float2bfloat16(P_::at(blk, p));
    }
}

}  // namespace

void launch_kv_bv_extract_dequant(const KvBvExtractDequantParams& params,
                                   void* stream) {
    if (params.num_heads_local <= 0 || params.v_head_dim <= 0
        || params.kv_lora_rank <= 0)
        return;

    const bool is_gguf = params.gguf_type >= 0;

    // FP8 needs scales; GGUF carries scales in-block (scales == nullptr).
    if (!params.kv_b_proj || !params.output || (!is_gguf && !params.scales)) {
        throw std::runtime_error(
            "launch_kv_bv_extract_dequant: null pointer in params");
    }

    const int HL = params.num_heads_local;
    const int P  = params.qk_nope_head_dim;
    const int V  = params.v_head_dim;
    const int D_c = params.kv_lora_rank;

    // One block per (head, v_row) pair.
    const int num_blocks = HL * V;
    auto cu_stream = static_cast<cudaStream_t>(stream);

    if (is_gguf) {
        // Per-element GGUF dequant of the V rows → BF16. Requires D_c % QK == 0
        // (a per-V-row is whole super-blocks on the kv_lora axis); the engine
        // guards this (shared L%QK rule) before dispatching here.
        const uint8_t* src = static_cast<const uint8_t*>(params.kv_b_proj);
        __nv_bfloat16* out = static_cast<__nv_bfloat16*>(params.output);
        switch (params.gguf_type) {  // 0=Q2_K,1=Q3_K,2=Q4_K,3=Q5_K,4=Q6_K,5=Q8_0
            case 0: kv_bv_extract_dequant_gguf_kernel<gguf_pol::Q2K> <<<num_blocks, kThreadsPerBlock, 0, cu_stream>>>(out, src, HL, P, V, D_c); break;
            case 1: kv_bv_extract_dequant_gguf_kernel<gguf_pol::Q3K> <<<num_blocks, kThreadsPerBlock, 0, cu_stream>>>(out, src, HL, P, V, D_c); break;
            case 2: kv_bv_extract_dequant_gguf_kernel<gguf_pol::Q4K> <<<num_blocks, kThreadsPerBlock, 0, cu_stream>>>(out, src, HL, P, V, D_c); break;
            case 3: kv_bv_extract_dequant_gguf_kernel<gguf_pol::Q5K> <<<num_blocks, kThreadsPerBlock, 0, cu_stream>>>(out, src, HL, P, V, D_c); break;
            case 4: kv_bv_extract_dequant_gguf_kernel<gguf_pol::Q6K> <<<num_blocks, kThreadsPerBlock, 0, cu_stream>>>(out, src, HL, P, V, D_c); break;
            case 5: kv_bv_extract_dequant_gguf_kernel<gguf_pol::Q8_0><<<num_blocks, kThreadsPerBlock, 0, cu_stream>>>(out, src, HL, P, V, D_c); break;
            default:
                throw std::runtime_error(
                    "launch_kv_bv_extract_dequant: unknown gguf_type "
                    + std::to_string(params.gguf_type));
        }
        return;
    }

    const int N_total = HL * (P + V);
    const int N_scale_blocks = (N_total + kScaleBlockSize - 1) / kScaleBlockSize;

    kv_bv_extract_dequant_kernel<<<num_blocks, kThreadsPerBlock, 0, cu_stream>>>(
        static_cast<__nv_bfloat16*>(params.output),
        static_cast<const uint8_t*>(params.kv_b_proj),
        static_cast<const float*>(params.scales),
        HL, P, V, D_c,
        N_total, N_scale_blocks);
}

}  // namespace layerstorm::compute

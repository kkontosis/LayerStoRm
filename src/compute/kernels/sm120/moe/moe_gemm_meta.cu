// Populate grouped GEMM metadata from device-resident expert_offsets.
// One 256-thread block (num_experts <= 256): each thread owns one expert; the
// sf_offsets exclusive prefix sum of pad128(M_e) is a shared-memory block scan.
// Bit-identical to the former single-thread sequential scan (was a <<<1,1>>>
// one-thread kernel ~8 us/launch, 2x/layer/GPU = ~4.5 ms/tok aggregate at
// decode; see spec/reports/DECODE_KERNEL_HUNT_HANDOFF.md 8b).

#include "compute/kernels/moe/moe_gemm_meta.h"

#include <cuda_runtime.h>

namespace layerstorm::compute {

namespace {

// One CUDA block covers up to kBlockExperts experts (max block dim = 1024,
// enough for every supported model: GLM 256, DSV3 256, Kimi K2.5 384).
constexpr int kBlockExperts = 1024;

// Parallel path: one 1024-thread block, each thread owns one expert; the
// sf_offsets exclusive prefix sum of pad128(M_e) is a shared-memory block scan.
__global__ void populate_gemm_meta_kernel(
    int32_t* __restrict__ problem_sizes,
    int32_t* __restrict__ sf_offsets,
    const int32_t* __restrict__ expert_offsets,
    int N, int K, int num_experts) {

    const int e = threadIdx.x;

    // problem_sizes: one independent (M_e, N, K) triple per expert.
    int32_t M_e = 0;
    if (e < num_experts) {
        M_e = expert_offsets[e + 1] - expert_offsets[e];
        problem_sizes[e * 3]     = M_e;
        problem_sizes[e * 3 + 1] = N;
        problem_sizes[e * 3 + 2] = K;
    }

    if (!sf_offsets) return;

    // sf_offsets[e] = exclusive prefix sum of pad128(M_e) over experts 0..e-1,
    // with sf_offsets[num_experts] = total. Hillis-Steele inclusive scan in
    // shared memory, then shift to exclusive. Deterministic (fixed order),
    // bit-identical to the former single-thread sequential accumulate.
    __shared__ int32_t scan[kBlockExperts];
    const int32_t pad = (e < num_experts) ? ((M_e + 127) / 128) * 128 : 0;
    scan[e] = pad;
    __syncthreads();

    for (int off = 1; off < kBlockExperts; off <<= 1) {
        int32_t add = (e >= off) ? scan[e - off] : 0;
        __syncthreads();
        scan[e] += add;
        __syncthreads();
    }

    // scan[e] is now the INCLUSIVE prefix sum; exclusive = scan[e] - pad.
    if (e < num_experts) sf_offsets[e] = scan[e] - pad;
    if (e == num_experts - 1) sf_offsets[num_experts] = scan[e];
}

// Fallback for the (currently unreached) num_experts > kBlockExperts case:
// the original single-thread sequential scan. Preserved so the kernel never
// silently drops experts on a hypothetical >1024-expert model.
__global__ void populate_gemm_meta_serial_kernel(
    int32_t* __restrict__ problem_sizes,
    int32_t* __restrict__ sf_offsets,
    const int32_t* __restrict__ expert_offsets,
    int N, int K, int num_experts) {

    if (threadIdx.x != 0) return;

    int32_t sf_acc = 0;
    for (int e = 0; e < num_experts; ++e) {
        int32_t M_e = expert_offsets[e + 1] - expert_offsets[e];
        problem_sizes[e * 3]     = M_e;
        problem_sizes[e * 3 + 1] = N;
        problem_sizes[e * 3 + 2] = K;
        if (sf_offsets) {
            sf_offsets[e] = sf_acc;
            sf_acc += ((M_e + 127) / 128) * 128;
        }
    }
    if (sf_offsets) {
        sf_offsets[num_experts] = sf_acc;
    }
}

}  // namespace

void launch_populate_gemm_meta(
    int32_t* problem_sizes,
    int32_t* sf_offsets,
    const int32_t* expert_offsets,
    int N, int K, int num_experts,
    void* stream) {

    if (num_experts <= 0) return;

    if (num_experts <= kBlockExperts) {
        // One block; threads past num_experts contribute a pad of 0 to the scan
        // and skip the metadata writes.
        populate_gemm_meta_kernel<<<1, kBlockExperts, 0,
                                    static_cast<cudaStream_t>(stream)>>>(
            problem_sizes, sf_offsets, expert_offsets, N, K, num_experts);
    } else {
        populate_gemm_meta_serial_kernel<<<1, 1, 0,
                                           static_cast<cudaStream_t>(stream)>>>(
            problem_sizes, sf_offsets, expert_offsets, N, K, num_experts);
    }
}

// ── Wave-masked top-K indices (TD-MOE-BIG-GEMM-SWEEP) ───────────────────────

namespace {

__global__ void mask_topk_indices_kernel(
    int32_t* __restrict__ masked_indices,
    const int32_t* __restrict__ topk_indices,
    const uint8_t* __restrict__ expert_mask,
    int count, int num_experts) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        const int32_t e = topk_indices[i];
        masked_indices[i] =
            (e >= 0 && e < num_experts && expert_mask[e]) ? e : -1;
    }
}

}  // namespace

void launch_mask_topk_indices(
    int32_t* masked_indices,
    const int32_t* topk_indices,
    const uint8_t* expert_mask,
    int count,
    int num_experts,
    void* stream) {

    if (count <= 0) return;

    const int threads = 256;
    int blocks = (count + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;  // grid-stride loop covers the rest
    mask_topk_indices_kernel<<<blocks, threads, 0,
                               static_cast<cudaStream_t>(stream)>>>(
        masked_indices, topk_indices, expert_mask, count, num_experts);
}

// ── Gather per-expert NVFP4 alphas from scattered cache slots ──────────────

namespace {

__global__ void gather_alphas_kernel(
    float* __restrict__ alphas,
    const void* const* __restrict__ b_ptrs,
    int64_t alpha_offset,
    int num_experts) {

    int e = threadIdx.x;
    if (e >= num_experts) return;

    auto* base = static_cast<const char*>(b_ptrs[e]);
    alphas[e] = *reinterpret_cast<const float*>(base + alpha_offset);
}

__global__ void gather_alphas_scaled_kernel(
    float* __restrict__ alphas,
    float* __restrict__ input_scales,
    const void* const* __restrict__ b_ptrs,
    int64_t ws2_offset,
    int64_t is_offset,
    int num_experts) {

    int e = threadIdx.x;
    if (e >= num_experts) return;

    auto* base = static_cast<const char*>(b_ptrs[e]);
    float ws2 = *reinterpret_cast<const float*>(base + ws2_offset);
    float is = *reinterpret_cast<const float*>(base + is_offset);
    // Missing experts point at a zero buffer: is reads 0.0 — treat as 1.0 so
    // the quantizer never divides by zero (the zero weights null the GEMM).
    if (!(is > 0.0f)) is = 1.0f;
    alphas[e] = ws2 * is;
    if (input_scales) input_scales[e] = is;
}

}  // namespace

void launch_gather_alphas(
    float* alphas_out,
    const void* const* b_ptrs,
    int64_t alpha_offset,
    int num_experts,
    void* stream) {

    if (num_experts <= 0) return;

    gather_alphas_kernel<<<1, num_experts, 0, static_cast<cudaStream_t>(stream)>>>(
        alphas_out, b_ptrs, alpha_offset, num_experts);
}

void launch_gather_alphas_scaled(
    float* alphas_out,
    float* input_scales_out,
    const void* const* b_ptrs,
    int64_t ws2_offset,
    int64_t is_offset,
    int num_experts,
    void* stream) {

    if (num_experts <= 0) return;

    gather_alphas_scaled_kernel<<<1, num_experts, 0,
                                  static_cast<cudaStream_t>(stream)>>>(
        alphas_out, input_scales_out, b_ptrs, ws2_offset, is_offset,
        num_experts);
}

}  // namespace layerstorm::compute

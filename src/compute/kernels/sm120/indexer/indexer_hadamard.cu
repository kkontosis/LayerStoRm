// TD-GLM-INDEXER-HADAMARD: in-place FWHT of indexer q/k rows. See
// indexer_hadamard.h for the llama.cpp convention this matches.

#include "compute/kernels/sm120/indexer/indexer_hadamard.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

// One CTA per row. The row is staged in shared memory as FP32; log2(dim)
// butterfly stages (natural-order FWHT == Sylvester Hadamard, no permutation),
// then scaled by 1/sqrt(dim) and written back as BF16. blockDim.x threads
// cooperate over dim/2 butterfly pairs per stage.
__global__ void indexer_hadamard_kernel(__nv_bfloat16* x, int rows, int dim,
                                        float inv_sqrt_dim) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __nv_bfloat16* xr = x + static_cast<int64_t>(row) * dim;

    extern __shared__ float s[];  // dim floats
    const int tid = threadIdx.x;

    for (int i = tid; i < dim; i += blockDim.x) s[i] = __bfloat162float(xr[i]);
    __syncthreads();

    const int half = dim >> 1;
    for (int h = 1; h < dim; h <<= 1) {
        // Pair p ∈ [0, dim/2): lower index i = 2h·(p/h) + (p mod h).
        for (int p = tid; p < half; p += blockDim.x) {
            const int i = ((p / h) * (h << 1)) + (p % h);
            const float a = s[i];
            const float b = s[i + h];
            s[i]     = a + b;
            s[i + h] = a - b;
        }
        __syncthreads();
    }

    for (int i = tid; i < dim; i += blockDim.x)
        xr[i] = __float2bfloat16(s[i] * inv_sqrt_dim);
}

bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

}  // namespace

void launch_indexer_hadamard(void* x_bf16, int rows, int dim,
                             cudaStream_t stream) {
    if (rows == 0) return;
    if (rows < 0 || !x_bf16 || dim < 2 || dim > 4096 || !is_pow2(dim))
        throw std::runtime_error("launch_indexer_hadamard: invalid params "
                                 "(dim must be a power of two in [2, 4096])");
    // Power-of-two threads ≥ 32, ≤ min(dim/2, 512): each stage has dim/2 pairs.
    int bdim = 32;
    while (bdim < dim / 2 && bdim < 512) bdim <<= 1;
    indexer_hadamard_kernel<<<rows, bdim, dim * sizeof(float), stream>>>(
        static_cast<__nv_bfloat16*>(x_bf16), rows, dim,
        1.0f / std::sqrt(static_cast<float>(dim)));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_indexer_hadamard: ")
                                 + cudaGetErrorString(err));
}

}  // namespace layerstorm::compute

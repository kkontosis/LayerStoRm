// GLM-25k (KV tiering): generic device row gather/scatter.
// One CTA per row; rows are opaque byte blobs.  Vectorized 16B path when the
// row is 16B-aligned at both ends (SnapMLA rows are: stride_row is a multiple
// of 16 and every source/destination is a multiple-of-stride offset into a
// 256B-aligned device allocation); byte fallback otherwise.

#include "compute/kernels/kv/kv_row_copy.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

__device__ __forceinline__ void copy_row(
    char* __restrict__ dst, const char* __restrict__ src, int row_bytes) {
    const bool vec16 =
        ((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)
          | static_cast<uintptr_t>(row_bytes)) & 15u) == 0;
    if (vec16) {
        auto* d4 = reinterpret_cast<uint4*>(dst);
        const auto* s4 = reinterpret_cast<const uint4*>(src);
        const int n4 = row_bytes / 16;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) d4[i] = s4[i];
    } else {
        for (int i = threadIdx.x; i < row_bytes; i += blockDim.x)
            dst[i] = src[i];
    }
}

__global__ void kv_row_gather_kernel(
    char* __restrict__ dst_base, int64_t dst_stride_block,
    int dst_stride_row, int page_size,
    const void* const* __restrict__ src_ptrs,
    int n_rows, int row_bytes) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const char* src = static_cast<const char*>(src_ptrs[row]);
    if (!src) return;
    char* dst = dst_base
        + static_cast<int64_t>(row / page_size) * dst_stride_block
        + static_cast<int64_t>(row % page_size) * dst_stride_row;
    copy_row(dst, src, row_bytes);
}

__global__ void kv_row_scatter_kernel(
    void* const* __restrict__ dst_ptrs,
    const char* __restrict__ src_base, int64_t src_stride_row,
    const int* __restrict__ src_row_idx,
    int n_rows, int row_bytes) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    char* dst = static_cast<char*>(dst_ptrs[row]);
    if (!dst) return;
    const char* src = src_base
        + static_cast<int64_t>(src_row_idx[row]) * src_stride_row;
    copy_row(dst, src, row_bytes);
}

void check_last(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " failed: "
                                 + cudaGetErrorString(err));
    }
}

}  // namespace

void launch_kv_row_gather(
    void* dst_base, int64_t dst_stride_block, int dst_stride_row,
    int page_size, const void* const* src_ptrs,
    int n_rows, int row_bytes, void* stream) {
    if (n_rows <= 0) return;
    kv_row_gather_kernel<<<n_rows, 128, 0,
                           static_cast<cudaStream_t>(stream)>>>(
        static_cast<char*>(dst_base), dst_stride_block, dst_stride_row,
        page_size, src_ptrs, n_rows, row_bytes);
    check_last("launch_kv_row_gather");
}

void launch_kv_row_scatter(
    void* const* dst_ptrs, const void* src_base, int64_t src_stride_row,
    const int* src_row_idx, int n_rows, int row_bytes, void* stream) {
    if (n_rows <= 0) return;
    kv_row_scatter_kernel<<<n_rows, 128, 0,
                            static_cast<cudaStream_t>(stream)>>>(
        dst_ptrs, static_cast<const char*>(src_base), src_stride_row,
        src_row_idx, n_rows, row_bytes);
    check_last("launch_kv_row_scatter");
}

}  // namespace layerstorm::compute

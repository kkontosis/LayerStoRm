// Kernel-node replacements for memcpy/memset nodes inside captured decode
// graphs. See compute/kernels/elementwise/graph_node_ops.h for rationale.
// Every kernel here is a pure byte mover — results are byte-identical to the
// memcpy/memset calls they replace.

#include "compute/kernels/elementwise/graph_node_ops.h"

#include <cuda_runtime.h>

#include <algorithm>

namespace layerstorm::compute {

namespace {

struct PtrCopyParams {
    const uint64_t* src[kMaxPtrCopyArrays];  // pinned host (UVA-readable)
    uint64_t* dst[kMaxPtrCopyArrays];        // device
};

// grid.y = array index, grid.x * block covers n_ptrs 8-byte entries.
__global__ void copy_ptr_arrays_kernel(PtrCopyParams p, int n_ptrs) {
    const int seg = blockIdx.y;
    const uint64_t* __restrict__ src = p.src[seg];
    uint64_t* __restrict__ dst = p.dst[seg];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_ptrs;
         i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

// Interleave: one thread moves one 16-byte chunk of one source row into its
// interleaved slot; each thread handles the A-half and B-half chunk at the
// same (row, col) so a single grid covers both sources.
__global__ void interleave_rows_kernel(
    uint4* __restrict__ dst,
    const uint4* __restrict__ src_a,
    const uint4* __restrict__ src_b,
    long long total_vec,  // rows * row_vec
    int row_vec) {        // row_bytes / 16
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total_vec;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long row = i / row_vec;
        const int col = static_cast<int>(i - row * row_vec);
        const long long dst_row = row * 2LL * row_vec;
        dst[dst_row + col] = src_a[i];
        dst[dst_row + row_vec + col] = src_b[i];
    }
}

__global__ void zero_fill_kernel(unsigned char* __restrict__ ptr,
                                 unsigned long long bytes) {
    // Vectorized body over the 16-byte-aligned middle, byte tail/head edges.
    const unsigned long long tid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    const unsigned long long addr = reinterpret_cast<unsigned long long>(ptr);
    const unsigned long long head =
        std::min(bytes, (16ULL - (addr & 15ULL)) & 15ULL);
    const unsigned long long body_vec = (bytes - head) / 16ULL;
    uint4* __restrict__ v = reinterpret_cast<uint4*>(ptr + head);
    const uint4 z = {0u, 0u, 0u, 0u};
    for (unsigned long long i = tid; i < body_vec; i += stride) v[i] = z;
    // Head + tail bytes (at most 30 total) by the first threads.
    const unsigned long long tail_start = head + body_vec * 16ULL;
    if (tid < head) ptr[tid] = 0;
    const unsigned long long tail = bytes - tail_start;
    if (tid < tail) ptr[tail_start + tid] = 0;
}

}  // namespace

void launch_copy_ptr_arrays(
    const PtrArrayCopyDesc* descs, int n_arrays, int n_ptrs, void* stream) {
    if (n_arrays <= 0 || n_ptrs <= 0) return;
    PtrCopyParams p{};
    const int na = std::min(n_arrays, kMaxPtrCopyArrays);
    for (int a = 0; a < na; ++a) {
        p.src[a] = reinterpret_cast<const uint64_t*>(descs[a].src);
        p.dst[a] = reinterpret_cast<uint64_t*>(descs[a].dst);
    }
    constexpr int kBlock = 256;
    dim3 grid((n_ptrs + kBlock - 1) / kBlock, na);
    copy_ptr_arrays_kernel<<<grid, kBlock, 0,
                             static_cast<cudaStream_t>(stream)>>>(p, n_ptrs);
}

void launch_interleave_rows(
    void* dst, const void* src_a, const void* src_b,
    int rows, size_t row_bytes, void* stream) {
    if (rows <= 0 || row_bytes == 0) return;
    const int row_vec = static_cast<int>(row_bytes / 16);
    const long long total_vec = static_cast<long long>(rows) * row_vec;
    constexpr int kBlock = 256;
    const int grid = static_cast<int>(
        std::min<long long>((total_vec + kBlock - 1) / kBlock, 65535));
    interleave_rows_kernel<<<grid, kBlock, 0,
                             static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint4*>(dst), static_cast<const uint4*>(src_a),
        static_cast<const uint4*>(src_b), total_vec, row_vec);
}

void launch_zero_fill(void* ptr, size_t bytes, void* stream) {
    if (bytes == 0) return;
    constexpr int kBlock = 256;
    const unsigned long long body_vec = bytes / 16ULL + 1ULL;
    const int grid = static_cast<int>(
        std::min<unsigned long long>((body_vec + kBlock - 1) / kBlock, 65535));
    zero_fill_kernel<<<grid, kBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<unsigned char*>(ptr),
        static_cast<unsigned long long>(bytes));
}

}  // namespace layerstorm::compute

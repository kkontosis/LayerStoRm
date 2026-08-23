// GLM-25k (KV tiering): generic device row gather/scatter for the tiered
// sparse-KV materialization path.
//
// CUDA-free binding header — implementations live in
// compute/kernels/smxx/kv/kv_row_copy.cu.  Rows are opaque byte blobs
// (dtype/layout-agnostic): the tiering manager copies whole KV cache rows
// (cache_stride_row bytes, self-contained per SnapMLA layout: FP8 c_kv |
// f32 scale | pre-scaled BF16 rope) between the paged pool, the hot row
// cache, the packed cold-incoming staging, and the dense per-layer scratch.

#pragma once

#include <cstdint>

namespace layerstorm::compute {

/// Gather n_rows rows into a dense fake-paged scratch.
///   dst row i lands at
///     dst_base + (i / page_size) * dst_stride_block
///              + (i % page_size) * dst_stride_row
///   src row i is read from src_ptrs[i] (device array of device pointers;
///   entries may alias pool pages, row-cache slots, or the cold-incoming
///   staging).  row_bytes bytes are copied per row.
void launch_kv_row_gather(
    void* dst_base, int64_t dst_stride_block, int dst_stride_row,
    int page_size,
    const void* const* src_ptrs /*device*/,
    int n_rows, int row_bytes,
    void* stream /*cudaStream_t*/);

/// Scatter n_rows rows from a packed source buffer to arbitrary destinations.
///   dst row i is written at dst_ptrs[i] (device array of device pointers,
///   e.g. hot row-cache slots); src row i is read from
///     src_base + src_row_idx[i] * src_stride_row
///   (src_row_idx: device int array — the row's index inside the packed
///   cold-incoming buffer).
void launch_kv_row_scatter(
    void* const* dst_ptrs /*device*/,
    const void* src_base, int64_t src_stride_row,
    const int* src_row_idx /*device*/,
    int n_rows, int row_bytes,
    void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute

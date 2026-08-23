// Block table linearization kernel for prefill dequantization (TD-71a:
// algorithm-agnostic — both the SnapMLA and TurboQuant AttentionDevices
// use it to feed their indexed KV dequant).
//
// Converts paged block_tables + seqlens_k into flat slot indices (e.g.
// TqDequantCKVIndexedParams::indices).  One thread per token position per
// sequence.
//
// Input:
//   block_tables[batch_size, max_blocks] — page-to-physical-page mapping
//   seqlens_k[batch_size]                — per-sequence KV lengths
//
// Output:
//   indices_out[sum(seqlens_k)]          — flat slot indices into KV cache
//   num_fetch_out[1]                     — total number of slots (atomic sum)

#include <cuda_runtime.h>
#include <cstdint>

namespace layerstorm::compute {

namespace {

__global__ void linearize_block_tables_kernel(
    const int* __restrict__ block_tables,
    const int* __restrict__ seqlens_k,
    int batch_size,
    int max_blocks,
    int page_size,
    int* __restrict__ indices_out,
    int* __restrict__ seq_offsets)   // [batch_size] — pre-computed prefix sums
{
    const int seq_idx = blockIdx.y;
    if (seq_idx >= batch_size) return;

    const int seq_len = seqlens_k[seq_idx];
    const int out_offset = seq_offsets[seq_idx];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= seq_len) return;

    const int page_idx = tid / page_size;
    const int page_off = tid % page_size;
    const int phys_page = block_tables[seq_idx * max_blocks + page_idx];
    indices_out[out_offset + tid] = phys_page * page_size + page_off;
}

// Small kernel to compute prefix sums of seqlens_k for the output offsets.
// Single block, suitable for batch_size <= 1024.
__global__ void compute_seq_offsets_kernel(
    const int* __restrict__ seqlens_k,
    int batch_size,
    int* __restrict__ seq_offsets,
    int* __restrict__ total_out)
{
    // Simple serial prefix sum (batch_size is small, typically 1 for prefill)
    int sum = 0;
    for (int i = 0; i < batch_size; ++i) {
        seq_offsets[i] = sum;
        sum += seqlens_k[i];
    }
    if (total_out) *total_out = sum;
}

}  // anonymous namespace

void launch_linearize_block_tables(
    const int* block_tables,
    const int* seqlens_k,
    int batch_size,
    int max_blocks,
    int page_size,
    int max_seq_len,
    int* indices_out,
    int* seq_offsets_scratch,
    int* num_fetch_out,
    void* stream)
{
    // KVS-3 (INV-KVS-EMPTY): max_seq_len == 0 is a legal empty DCP shard —
    // grid.x would be 0 (invalid launch config). Nothing to linearize; the
    // caller also skips the dequant, and the attention kernel handles s_kv=0.
    if (batch_size <= 0 || max_seq_len <= 0) return;
    auto s = static_cast<cudaStream_t>(stream);

    // Step 1: compute prefix sums of seqlens_k
    compute_seq_offsets_kernel<<<1, 1, 0, s>>>(
        seqlens_k, batch_size, seq_offsets_scratch, num_fetch_out);

    // Step 2: linearize block tables
    const int threads = 256;
    const int blocks_x = (max_seq_len + threads - 1) / threads;
    dim3 grid(blocks_x, batch_size);
    linearize_block_tables_kernel<<<grid, threads, 0, s>>>(
        block_tables, seqlens_k, batch_size, max_blocks, page_size,
        indices_out, seq_offsets_scratch);
}

}  // namespace layerstorm::compute

// Block table linearization for prefill dequantization (TD-71a:
// algorithm-agnostic — used by both the SnapMLA and TurboQuant
// AttentionDevices).
//
// Converts paged block_tables + seqlens_k into flat slot indices consumed
// by the indexed KV dequant (e.g. TqDequantCKVIndexedParams::indices).

#pragma once

namespace layerstorm::compute {

/// Linearize block tables into flat slot indices for indexed KV dequant.
///
/// @param block_tables   [batch_size, max_blocks] page→physical mapping (device)
/// @param seqlens_k      [batch_size] per-sequence KV lengths (device)
/// @param batch_size     number of sequences
/// @param max_blocks     second dimension of block_tables
/// @param page_size      tokens per page
/// @param max_seq_len    upper bound on any single sequence length (for grid sizing)
/// @param indices_out    [sum(seqlens_k)] output flat slot indices (device)
/// @param seq_offsets_scratch [batch_size] scratch for prefix sums (device)
/// @param num_fetch_out  [1] total slots written (device, may be nullptr)
/// @param stream         CUDA stream
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
    void* stream);

}  // namespace layerstorm::compute

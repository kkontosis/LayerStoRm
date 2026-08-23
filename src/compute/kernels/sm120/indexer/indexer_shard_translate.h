#pragma once

// KVS-4: GLOBAL→LOCAL DSA sparse-index translation for sequence-sharded KV
// (hardware.dcp_kv_mode = sharded, INV-4.9e round-robin-by-chunk partition).
//
// Under dcp_indexer_mode=replicated every rank derives the IDENTICAL global
// top-k (indexer-K is replicated, positions are global). Under sharded KV,
// however, the sparse-attention consumer indexes rank r's KV STAGING, which
// holds only rank r's token shard in local ascending order — global position
// ids are meaningless there. This kernel filters each batch row's global
// top-k list down to the positions rank r OWNS and rewrites them as LOCAL
// slot indices into r's staging:
//
//   owner(g) = (g / chunk_tokens) % dcp_size          (INV-4.9e; must match
//              PageAllocator::dcp_rank_for_token / kv_shard_math::owner_rank)
//   local(g) = (g / (chunk_tokens*dcp_size)) * chunk_tokens + g % chunk_tokens
//              (the number of positions owned by `rank` strictly before g —
//              kv_shard_math::owned_len(rank, g), valid when owner(g)==rank)
//
// Input indices are the lightning-topk output: sorted ascending, padded with
// −1, with global_lengths[b] valid entries. Ownership filtering of an
// ascending list preserves ascending order and local(g) is monotonic on the
// owned subset, so the output is again sorted ascending, compacted, padded
// with −1, with local_lengths[b] = |owned ∩ top-k| (may be 0: the rank owns
// none of the selected positions — the sparse kernel then emits zero output
// + lse=+inf and the DCP QAG combine weights the rank 0, INV-KVS-EMPTY).
//
// Deterministic: warp-ballot compaction, no atomics.

#include <cuda_runtime.h>

namespace layerstorm::compute {

struct IndexerShardTranslateParams {
    const int* global_indices;  ///< [num_tokens, topk] device, sorted asc, pad −1
    const int* global_lengths;  ///< [num_tokens] device, valid-entry counts
    int* local_indices;         ///< [num_tokens, topk] device out, pad −1
    int* local_lengths;         ///< [num_tokens] device out
    int num_tokens;             ///< batch rows
    int topk;                   ///< row stride / max entries (index_topk)
    int chunk_tokens;           ///< dcp_chunk_size (INV-4.9e round-robin unit)
    int dcp_size;               ///< number of DCP ranks (>= 2)
    int rank;                   ///< this rank (0..dcp_size-1)
};

void launch_indexer_shard_translate(const IndexerShardTranslateParams& p,
                                    cudaStream_t stream);

}  // namespace layerstorm::compute

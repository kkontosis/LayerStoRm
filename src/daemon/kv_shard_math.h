#pragma once

// KVS-2: pure token→rank sharding math for DCP sequence-sharded KV
// (hardware.dcp_kv_mode = "sharded", INV-DCP-6 Helix sequence sharding).
//
// The partition is ROUND-ROBIN BY CHUNK (Helix S2.3, INV-4.9e): global token
// position p belongs to rank (p / chunk_tokens) % dcp_size. This matches
// PageAllocator::dcp_rank_for_token exactly — page ALLOCATION has routed by
// this rule since bring-up, so the metadata built here references pages that
// physically live on the owning rank's GPU. chunk_tokens is a multiple of
// page_size_tokens (asserted in PageAllocator::set_dcp_config), so every KV
// page is owned entirely by ONE rank and within-page row offsets are
// identical in global and rank-local numbering.
//
// Rationale for round-robin chunks (vs contiguous halves): load balance —
// every rank's shard grows uniformly with context length, so per-rank
// attention work and KV reads stay ~equal at any sequence length. The DCP
// LSE combine (KVS-3, INV-DCP-2/-3) is partition-agnostic: it softmax-merges
// per-rank partials over ANY disjoint token subsets, so the choice is free
// for correctness and round-robin wins on balance. Contiguous halves would
// idle rank 1 until the sequence crosses the midpoint.
//
// Header-only, dependency-free, CPU-pure: unit-testable without CUDA.

#include <cstdint>

namespace layerstorm::daemon::kvshard {

/// Rank owning global token position `pos`.
/// Mirrors PageAllocator::dcp_rank_for_token (kept in sync by unit test).
inline int owner_rank(uint32_t pos, int chunk_tokens, int dcp_size) {
    if (dcp_size <= 1) return 0;
    return static_cast<int>((pos / static_cast<uint32_t>(chunk_tokens))
                            % static_cast<uint32_t>(dcp_size));
}

/// Number of tokens in [0, n_tokens) owned by `rank` — the rank's LOCAL KV
/// length once positions 0..n_tokens-1 are appended. Disjoint + complete:
/// sum over ranks == n_tokens (INV-4.9e).
inline int owned_len(int rank, uint32_t n_tokens, int chunk_tokens,
                     int dcp_size) {
    if (dcp_size <= 1) return static_cast<int>(n_tokens);
    const uint32_t chunk = static_cast<uint32_t>(chunk_tokens);
    const uint32_t cycle = chunk * static_cast<uint32_t>(dcp_size);
    const uint32_t full_cycles = n_tokens / cycle;
    const uint32_t rem = n_tokens % cycle;
    const uint32_t r_start = static_cast<uint32_t>(rank) * chunk;
    uint32_t rem_owned = 0;
    if (rem > r_start)
        rem_owned = (rem - r_start < chunk) ? (rem - r_start) : chunk;
    return static_cast<int>(full_cycles * chunk + rem_owned);
}

/// Rank owning global LOGICAL page `global_page` (page fully owned because
/// chunk_tokens % page_size_tokens == 0).
inline int page_owner_rank(int global_page, int pages_per_chunk,
                           int dcp_size) {
    if (dcp_size <= 1) return 0;
    return (global_page / pages_per_chunk) % dcp_size;
}

/// LOCAL logical page index of `global_page` within its owner rank's KV:
/// the number of pages owned by that rank strictly before it. The owner's
/// block table row j = its j-th owned global page, ascending.
inline int local_page_index(int global_page, int pages_per_chunk,
                            int dcp_size) {
    if (dcp_size <= 1) return global_page;
    const int cycle_pages = pages_per_chunk * dcp_size;
    const int cycles = global_page / cycle_pages;
    const int within = global_page % pages_per_chunk;
    return cycles * pages_per_chunk + within;
}

/// Inverse of local_page_index for a given rank (test/diagnostic helper):
/// the global logical page backing `rank`'s local page `local_page`.
inline int global_page_of_local(int rank, int local_page, int pages_per_chunk,
                                int dcp_size) {
    if (dcp_size <= 1) return local_page;
    const int cycles = local_page / pages_per_chunk;
    const int within = local_page % pages_per_chunk;
    return (cycles * dcp_size + rank) * pages_per_chunk + within;
}

}  // namespace layerstorm::daemon::kvshard

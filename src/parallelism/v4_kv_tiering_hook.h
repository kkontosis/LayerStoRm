// TD-V4-KVT: DeepSeek-V4 CSA-bucket KV tiering — executor-side hook seam.
//
// The V4 tiering manager (daemon::V4KvTiering) demotes COLD CSA kMain pages
// (64 compressed entries = 256 native tokens each) behind a retention
// frontier to pinned host RAM, and RE-PROMOTES exactly the pages containing
// the lightning selection's cold blocks before the attention gather — the
// unmodified attention then reads real pool pages (INV-KVT-1: tiering
// changes PLACEMENT, never selection or values; byte-identical entries).
//
// Bucket policy (P3, 2026-08-21): ONLY the CSA main tier is tiered.
// HCA (dense visibility every step — repromote-everything degenerate),
// LID (the lightning scorer reads the whole tier every step) and SWA
// (bounded ring) are exempt by construction.
//
// Dependency-light: parallelism must not depend on daemon, so DcpExecutor
// consumes this abstract interface via AttentionExecParams::v4_tiering.

#pragma once

#include <cstdint>

namespace layerstorm::parallelism {

class V4KvTieringHook {
public:
    virtual ~V4KvTieringHook() = default;

    /// Cheap host check: does (seq, layer) currently have demoted CSA
    /// pages? False ⇒ the executor takes the untouched fast path.
    virtual bool has_demotions(uint64_t seq_id, int layer) const = 0;

    /// Ensure the CSA pages containing the given LOGICAL block ids are hot
    /// (repromote cold pages: kMain page alloc via the dispatcher seam +
    /// per-rank H2D enqueued on each rank's attention stream + seq_pages_
    /// un-neutralize + in-place update of the CURRENT step's host
    /// block-table rows). `logical_ids` nullptr ⇒ all blocks [0, n)
    /// (the executor's deterministic IOTA visibility path); otherwise `n`
    /// ids, -1 entries ignored (topk padding). THROWS on kMain pool
    /// exhaustion — capacity fail-closed, never a silent wrong read
    /// (INV-KVT-2).
    virtual void ensure_hot(uint64_t seq_id, int layer,
                            const int* logical_ids, int n) = 0;
};

}  // namespace layerstorm::parallelism

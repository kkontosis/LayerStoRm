// GLM-25k: DSA-guided KV tiering — executor-side hook seam.
//
// The tiering manager (daemon::KvTieringManager) demotes cold main-KV pages
// to pinned host RAM; before each tiered layer's sparse attention it
// materializes EXACTLY the indexer's selected rows into a dense per-layer
// FP8 scratch and presents that scratch as a small fake paged cache.  The
// executor then calls the UNMODIFIED prefill_attention with the fake view +
// IDENTITY sparse indices (topk_lengths unchanged) — kernel-change-free and
// bit-identical to the non-tiered path (INV-KVT-1: tiering changes
// PLACEMENT, never SELECTION or values).
//
// Dependency-light: parallelism must not depend on daemon, so DcpExecutor
// consumes this abstract interface via AttentionExecParams::kv_tiering.

#pragma once

namespace layerstorm::parallelism {

/// Result of a per-(rank, layer) tiered materialization: a fake paged-cache
/// view over the dense scratch holding the selected rows in selection order.
struct TieredKvView {
    /// Dense scratch presented as a contiguous fake paged cache (same
    /// cache_stride_block / cache_stride_row / page_size as the real cache).
    void* kv_cache = nullptr;
    /// Device iota block table (fake page j = scratch page j).
    const int* block_tables = nullptr;
    /// Device int holding the row count n (aliases the producer's
    /// topk_lengths buffer — B==1).
    const int* seqlens_k = nullptr;
    int max_blocks_per_seq = 0;   ///< number of fake pages
    int seq_len_kv = 0;           ///< host-side n (rows materialized)
    /// Device identity iota [0..index_topk) — row i of the scratch IS
    /// selected row i, so identity indices reproduce the exact selection.
    const int* sparse_indices = nullptr;
};

/// Abstract tiering hook consumed by DcpExecutor (implemented by
/// daemon::KvTieringManager).
class KvTieringHook {
public:
    virtual ~KvTieringHook() = default;

    /// TD-KVT-SYNC: optional selection-readback overlap seam.  Called by the
    /// executor for (rank, layer) as soon as this layer's selection is fully
    /// enqueued on `stream` (the rank's attention stream) — i.e. right after
    /// the indexer top-k (and, in local mode, its cross-rank merge) — and
    /// BEFORE the executor enqueues the projection/staging work between the
    /// indexer and the sparse consumer.  The hook may start an async D2H of
    /// the selection here so the later materialize() finds it already
    /// host-resident (the host wait overlaps the executor's own enqueueing).
    /// `selection_fresh` = false means the producer REUSED the previous
    /// produce's buffers (IndexShare shared layer: byte-identical content) —
    /// the hook may then skip the readback entirely and reuse its host copy.
    /// Pure hint: materialize() must stay correct if prepare was never
    /// called.  Default no-op.
    virtual void prepare(int rank, int layer_idx,
                         const int* sparse_indices_dev,
                         const int* topk_lengths_dev,
                         int batch_size, void* stream,
                         bool selection_fresh) {
        (void)rank; (void)layer_idx; (void)sparse_indices_dev;
        (void)topk_lengths_dev; (void)batch_size; (void)stream;
        (void)selection_fresh;
    }

    /// Materialize the selected rows for (rank, layer) on `stream` (the
    /// rank's attention stream).  sparse_indices_dev / topk_lengths_dev are
    /// the producer's device buffers (GLOBAL positions).  Returns true and
    /// fills `out` when the layer must consume the tiered view; false when
    /// the layer has no cold pages (original full-residency path is valid).
    /// Throws on unrecoverable inconsistency — never silently falls back
    /// once pages are cold.
    virtual bool materialize(int rank, int layer_idx,
                             const int* sparse_indices_dev,
                             const int* topk_lengths_dev,
                             int batch_size, void* stream,
                             TieredKvView* out) = 0;

    /// Called when a DSA-capable decode layer falls back to DENSE attention.
    /// Dense staging reads the FULL prefix through the real block tables, so
    /// it is only legal while every page of the layer is VRAM-resident.
    /// Throws if the layer has cold pages (INV-KVT-2 fail-loud) — a silent
    /// fallback would read stale/reused pages.
    virtual void on_dense_layer(int layer_idx) = 0;
};

}  // namespace layerstorm::parallelism

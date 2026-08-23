// TD-V4-KVT (capability-completion P3, 2026-08-21): DeepSeek-V4 CSA-bucket
// KV tiering manager.
//
// Page-granular demote/repromote over the V4 CSA kMain tier (64 compressed
// entries × 1160 B = one 256-native-token logical block per page):
//   • DEMOTE: after a layer's attention, every CSA page of the stepped
//     sequence whose ENTIRE token span lies behind the retention frontier
//     (pos+1 − retention_tokens) is copied D2H (rank 0's replica — V4
//     replicated computation is bit-identical across ranks, so ONE cold
//     copy suffices) into pinned host RAM, the VRAM page returns to the
//     allocator, and the dispatcher's seq_pages_ entry is neutralized
//     (free_page seam — the GLM KvTieringManager contract).
//   • REPROMOTE (selection-driven): before the attention gather the
//     executor hands the lightning selection's logical block ids (or the
//     IOTA visibility range) to ensure_hot(); cold pages containing
//     selected blocks get a fresh kMain page (alloc_page seam —
//     allocate_replicated lockstep under replicated KV), the cold bytes
//     H2D to EVERY rank's pool, seq_pages_ un-neutralized, and the CURRENT
//     step's host block-table rows updated in place (set_step_bt) so the
//     executor's per-row page-table upload sees the fresh page.
//
// Placement-only by construction: repromoted bytes are the demoted bytes,
// attention reads real pool pages — token-identical to tiering-off
// (INV-KVT-1/INV-V4-KVT). Capacity failures are fail-closed (throw), a
// full cold pool skips further demotions (fail-safe, GLM semantics).

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "core/device_backend.h"
#include "core/memory/page_allocator.h"
#include "parallelism/v4_kv_tiering_hook.h"

namespace layerstorm::daemon {

class V4KvTiering final : public parallelism::V4KvTieringHook {
public:
    struct Options {
        int dcp_size = 1;
        /// Per-rank device backends (rank order) + kMain pool bases.
        std::vector<compute::DeviceBackend*> device_backends;
        std::vector<void*> kv_main_bases;
        /// Per-rank attention streams (H2D repromotes enqueue here so the
        /// following kernels are stream-ordered). Null entries → sync copy.
        std::vector<void*> attn_streams;
        /// Dispatcher seams (GLM KvTieringManager contract): free returns
        /// the page + neutralizes seq_pages_; alloc allocates a fresh kMain
        /// page (replicated lockstep), un-neutralizes seq_pages_ and
        /// poisons the kv-meta dirty guard. alloc nullopt = pool full.
        std::function<void(uint64_t seq, int layer, int logical,
                           const memory::PageHandle&)> free_page;
        std::function<std::optional<memory::PageHandle>(
            uint64_t seq, int layer, int logical)> alloc_page;
        int64_t page_bytes = 0;        ///< CSA page bytes (64 × 1160)
        int csa_ratio = 4;             ///< native tokens per CSA entry
        int page_tokens = 256;         ///< native tokens per CSA page
        /// Retention window in native tokens (memory.kv_tiering
        /// .hot_buffer_slots; 0 → 2 × index_topk).
        int retention_tokens = 0;
        int index_topk = 512;
        /// Cold-pool capacity = ratio × retention pages per CSA layer
        /// (memory.kv_tiering.host_to_device_ratio).
        double host_to_device_ratio = 8.0;
        /// CSA layer mask [num_hidden_layers] (1 = CSA).
        std::vector<uint8_t> csa_layers;
    };

    explicit V4KvTiering(Options opts);
    ~V4KvTiering() override;

    // ── V4KvTieringHook (executor side) ─────────────────────────────────
    bool has_demotions(uint64_t seq_id, int layer) const override;
    void ensure_hot(uint64_t seq_id, int layer,
                    const int* logical_ids, int n) override;

    // ── Dispatcher side ─────────────────────────────────────────────────
    /// Point the manager at the CURRENT step's host block-table rows (one
    /// per rank; each row = per-logical-block page ids for `layer`).
    /// Repromotes update these rows in place. Rows stay valid for the
    /// whole executor call (kv_meta_scratch_ storage).
    void set_step_bt(uint64_t seq_id, int layer, std::vector<int*> rows,
                     int len);

    /// Demote sweep after a CSA layer's attention at `pos` (the step's
    /// LAST row position): pages fully behind pos+1 − retention demote.
    /// Synchronous D2H (v1 — 74 KB/page is noise on the V4 walls).
    void after_attention(uint64_t seq_id, int layer, int pos,
                         const std::vector<memory::PageHandle>& layer_pages,
                         int num_logical, int handle_stride);

    /// Drop a sequence's cold copies.
    void free_sequence(uint64_t seq_id);

    /// TD-V4-SERVE-PREFIX: deep-copy src's cold pages to dst at
    /// CMD_SEQ_FORK (host byte vectors; the child repromotes into its own
    /// fresh VRAM pages). No-op when src has no demotions.
    void on_seq_fork(uint64_t src_id, uint64_t dst_id);

    bool seq_has_demotions(uint64_t seq_id) const;

    int64_t demoted_pages_total() const { return stat_demotes_; }
    int64_t repromoted_pages_total() const { return stat_repromotes_; }

private:
    struct ColdPage {
        std::vector<uint8_t> bytes;   // page_bytes (host copy, rank 0)
    };
    struct SeqState {
        // (layer << 32 | logical) → cold copy
        std::unordered_map<uint64_t, ColdPage> cold;
    };
    static uint64_t key(int layer, int logical) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32)
             | static_cast<uint32_t>(logical);
    }

    Options opts_;
    int retention_ = 0;
    int64_t cold_cap_pages_ = 0;      // fail-safe demotion cap (per seq)
    std::unordered_map<uint64_t, SeqState> seqs_;
    // Current-step block-table rows (per rank) + the layer they belong to.
    int step_layer_ = -1;
    std::vector<int*> step_bt_;
    int step_bt_len_ = 0;
    // The stepped sequence (set_step_bt is per dispatch; ensure_hot only
    // ever runs for it).
    uint64_t step_seq_ = 0;
    int64_t stat_demotes_ = 0;
    int64_t stat_repromotes_ = 0;
    int64_t stat_skipped_full_ = 0;

    void maybe_dump_stats();
};

}  // namespace layerstorm::daemon

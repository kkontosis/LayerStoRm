// KV cache metadata build (block tables / seqlens / slot mappings, replicated
// + KVS-2 sharded) — moved verbatim from dispatch_attention.cpp (attention
// refactor V2 P1). Still CommandDispatcher members.

#include "daemon/command_dispatcher.h"
#include "daemon/v4_kv_tiering.h"
#include "daemon/dispatch_detail.h"
#include "daemon/kv_shard_math.h"  // KVS-2: sharded-KV token→rank math
#include "daemon/kv_tiering_manager.h"  // GLM-25k: DSA-guided KV tiering

#include <algorithm>
#include <atomic>   // TD-DRIFT-ROOTCAUSE diagnostic dump
#include <cstdio>   // TD-DRIFT-ROOTCAUSE diagnostic dump
#include <cstdlib>  // TD-DRIFT-ROOTCAUSE getenv gate
#include <mutex>    // KVS-4: once-flag fail-closed warning
#include <vector>   // TD-DRIFT-ROOTCAUSE D2H staging

#include <spdlog/spdlog.h>

#include "compute/stream_manager.h"
#include "core/attention_device.h"
#include "core/cuda_hardware_query.h"  // EPM-1: host_register_pinned_portable
#include "core/device_backend.h"
#include "core/memory/numa_manager.h"  // EPM-1: NUMA-local D2H staging
#include "core/memory/page_allocator.h"
#include "speculation/epm_dump.h"      // EPM-1: routing-label dump
#include "compute/kernels/elementwise/residual_add.h"
#include "compute/kernels/mhc/mhc.h"
#include "compute/kernels/norm/rmsnorm.h"
#include "compute/kernels/moe/router_projection.h"  // F-1: fused gate in attention
#include "parallelism/dcp_communicator.h"
#include "parallelism/dcp_executor.h"
#include "compute/kernels/moe/hash_gating.h"  // V4-4 hash-layer gating
#include "sm120/gating/topk_gating.h"  // F-1: launch_topk_gating

namespace layerstorm::daemon {

// ── KD-4e: Build KV cache metadata from sideband + seq_pages_ ──────────

CommandDispatcher::KvMetaResult
CommandDispatcher::build_kv_metadata(int batch_size, int dcp_size) {
    if (!deps_.sideband_base || !deps_.page_allocator)
        return KvMetaResult::kUnavailable;
    if (kv_meta_scratch_.empty() || max_blocks_per_seq_ <= 0)
        return KvMetaResult::kUnavailable;

    const auto* batch_entries = reinterpret_cast<const ipc::BatchDescriptorEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kBatchDescriptorOff);

    // TD-51cb: skip rebuild when sideband hasn't changed (multi-layer
    // pipelines). TD-GOLDEN-KVMETA-PER-LAYER: the build covers ALL layers,
    // so the fingerprint has no layer term — within one token the rebuild
    // and its H2D copies run exactly once.
    if (batch_size == kv_meta_last_batch_size_
        && batch_size > 0
        && batch_entries[0].seq_id == kv_meta_last_seq_id_
        && batch_entries[0].token_pos == kv_meta_last_token_pos_) {
        return KvMetaResult::kOk;  // device scratch already has correct data
    }

    const int page_size = deps_.kv_page_size > 0 ? deps_.kv_page_size : 64;
    const int L = kv_layers_ > 0 ? kv_layers_ : 1;
    // Layer stride uses the CURRENT batch size (compact layout, one
    // contiguous H2D); the dirty guard ties device contents to batch_size.
    const size_t bt_bytes = static_cast<size_t>(L) * batch_size
                          * max_blocks_per_seq_ * sizeof(int);
    const size_t sm_bytes = static_cast<size_t>(L) * batch_size * sizeof(int);

    // TD-GOLDEN-KV-EXHAUST: any failure below must poison the dirty guard
    // (both levels — TD-PREFILL-KVMETA-BT-SPLIT) and fail the attention
    // command — a zero slot mapping would silently corrupt physical page 0.
    auto fail = [this]() {
        invalidate_kv_meta();
        kv_bt_built_epoch_ = ~0ULL;
        return KvMetaResult::kFailed;
    };

    // TD-PREFILL-KVMETA-BT-SPLIT: the block tables depend only on the batch
    // row→seq mapping and each seq's allocated page list — NOT on token_pos.
    // Pass 1 below stages the PER-STEP arrays (seqlens + slot_mappings,
    // ~82 KiB) and detects page growth; the block tables (165.7 MiB/rank at
    // max_batch 256 / 32k window) rebuild + re-upload in pass 2 ONLY when
    // the page-topology epoch advanced, the batch size changed, or the
    // row→seq composition changed. A chunked/superchunk prefill advances
    // token_pos every sub-chunk (12+ fingerprint changes/layer × 78 layers);
    // without the split each one re-staged + re-uploaded the identical
    // 165.7 MiB per rank from pageable memory (~30 s of a 103 s 2876-token
    // prefill wall).
    bool bt_dirty = kv_bt_built_epoch_ != kv_pages_epoch_
                 || kv_bt_batch_size_ != batch_size
                 || static_cast<int>(kv_bt_seq_ids_.size()) != batch_size;

    // Zero the per-step host staging for all ranks (block tables in pass 2).
    for (int r = 0; r < dcp_size; ++r) {
        auto& m = kv_meta_scratch_[r];
        std::memset(m.host_seqlens_k.data(), 0,
                    static_cast<size_t>(batch_size) * sizeof(int));
        std::memset(m.host_slot_mappings.data(), 0, sm_bytes);
    }

    // KVS-2: sequence-sharded KV routing (hardware.dcp_kv_mode = sharded).
    const bool sharded = kv_sharded_ && dcp_size >= 2;
    if (sharded)
        std::memset(host_global_seqlens_.data(), 0,
                    static_cast<size_t>(batch_size) * sizeof(int));

    const int chunk = kv_dcp_chunk_tokens_ > 0 ? kv_dcp_chunk_tokens_
                                               : page_size;
    if (sharded && chunk % page_size != 0) {
        spdlog::error("build_kv_metadata: dcp_chunk_size {} not a multiple "
                      "of page_size {} (sharded KV)", chunk, page_size);
        return fail();
    }
    const int ppc = chunk / page_size;  // pages per ownership chunk

    // ── Pass 1: per-step arrays (seqlens, slot_mappings, global seqlens) +
    //    page-growth and row→seq composition tracking. ────────────────────
    for (int b = 0; b < batch_size; ++b) {
        const auto& entry = batch_entries[b];
        const uint64_t seq_id = entry.seq_id;
        const uint32_t token_pos = entry.token_pos;

        // Row→seq composition: any change invalidates the block tables.
        if (!bt_dirty && kv_bt_seq_ids_[static_cast<size_t>(b)] != seq_id)
            bt_dirty = true;

        // Look up sequence pages.
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) {
            spdlog::error("build_kv_metadata: unknown seq_id {} at batch idx {}",
                          seq_id, b);
            return fail();
        }

        // KD-4e1: auto-grow pages if token_pos exceeds current allocation.
        // Growth mutates the page list → the block tables must rebuild.
        const size_t pages_before = it->second.kv_pages.size();
        if (!ensure_pages(seq_id, token_pos))
            return fail();
        if (it->second.kv_pages.size() != pages_before)
            bt_dirty = true;

        const auto& pages = it->second.kv_pages;
        // TD-GOLDEN: pages are layer-major per logical page —
        // handle (logical j, layer l) at pages[j * kv_layers_ + l].
        const int num_logical = static_cast<int>(pages.size()) / L;
        const int logical_page = static_cast<int>(token_pos / page_size);
        if (logical_page >= num_logical) {
            // Growth was capped (max_blocks_per_seq_) or otherwise short.
            spdlog::error("build_kv_metadata: no page for seq {} token_pos {} "
                          "(have {} logical pages)",
                          seq_id, token_pos, num_logical);
            return fail();
        }

        if (!sharded) {
            // KD-4f-d.1b: replicate KV metadata across all ranks for TP>1.
            // MLA attention uses replicated KV — every rank sees identical
            // KV data.
            for (int rank = 0; rank < dcp_size; ++rank) {
                auto& m = kv_meta_scratch_[rank];

                // seqlens_k: attend to tokens 0..token_pos after k_append.
                m.host_seqlens_k[b] = static_cast<int>(token_pos + 1);

                for (int l = 0; l < L; ++l) {
                    // slot_mappings: flat slot for k_append write position.
                    const int phys_page_idx =
                        pages[static_cast<size_t>(logical_page) * L + l]
                            .page_idx;
                    m.host_slot_mappings[
                        static_cast<size_t>(l) * batch_size + b] =
                        phys_page_idx * page_size
                        + static_cast<int>(token_pos % page_size);
                }
            }
        } else {
            // KVS-2: SHARDED branch — token → rank by position (INV-4.9e
            // round-robin chunks, kv_shard_math.h; matches
            // PageAllocator::dcp_rank_for_token, so every referenced page
            // physically lives on the referencing rank's GPU). Per-rank
            // DIVERGENT metadata: seqlens_k[r] = rank-LOCAL shard length,
            // slot_mappings[r] = real slot on the OWNING rank / rank-local
            // trash slot elsewhere (the k_append kernels have no skip path;
            // the trash page is never referenced by any block table).
            // GLOBAL lengths (RoPE positions) go to host_global_seqlens_ /
            // dev_global_seqlens. Block tables (rank r's owned pages in
            // local order) build in pass 2.
            const int n_tokens = static_cast<int>(token_pos) + 1;
            host_global_seqlens_[b] = n_tokens;
            const int owner =
                kvshard::owner_rank(token_pos, chunk, dcp_size);

            for (int rank = 0; rank < dcp_size; ++rank) {
                auto& m = kv_meta_scratch_[rank];

                // Rank-LOCAL shard length after this step's k_append.
                const int local_len = kvshard::owned_len(
                    rank, static_cast<uint32_t>(n_tokens), chunk, dcp_size);
                m.host_seqlens_k[b] = local_len;

                // Non-owner rows write to the rank's trash slot — it must
                // exist.
                if (rank != owner && kv_trash_slot_base_[rank] < 0) {
                    spdlog::error("build_kv_metadata: no trash page on rank "
                                  "{} (sharded KV)", rank);
                    return fail();
                }

                for (int l = 0; l < L; ++l) {
                    // slot_mappings: owning rank appends for real; other
                    // ranks write to their trash page.
                    int slot;
                    if (rank == owner) {
                        const auto& h =
                            pages[static_cast<size_t>(logical_page) * L + l];
                        slot = h.page_idx * page_size
                             + static_cast<int>(token_pos % page_size);
                    } else {
                        slot = kv_trash_slot_base_[rank]
                             + (b % page_size);
                    }
                    m.host_slot_mappings[
                        static_cast<size_t>(l) * batch_size + b] = slot;
                }
            }
        }
    }

    // ── Pass 2 (bt_dirty only): rebuild + restage the block tables. ──────
    if (bt_dirty) {
        for (int r = 0; r < dcp_size; ++r)
            std::memset(kv_meta_scratch_[r].host_block_tables.data(), 0,
                        bt_bytes);

        for (int b = 0; b < batch_size; ++b) {
            const auto& entry = batch_entries[b];
            const uint64_t seq_id = entry.seq_id;
            // Pass 1 verified presence + ran growth; the map is unchanged
            // since (single daemon thread).
            const auto& pages = sequences_.find(seq_id)->second.kv_pages;
            const int num_logical = static_cast<int>(pages.size()) / L;

            if (!sharded) {
                for (int rank = 0; rank < dcp_size; ++rank) {
                    auto& m = kv_meta_scratch_[rank];
                    for (int l = 0; l < L; ++l) {
                        // block_tables: layer l's physical page indices.
                        int* bt = m.host_block_tables.data()
                                + (static_cast<size_t>(l) * batch_size + b)
                                  * max_blocks_per_seq_;
                        for (int j = 0;
                             j < num_logical && j < max_blocks_per_seq_; ++j)
                            bt[j] = pages[static_cast<size_t>(j) * L + l]
                                        .page_idx;
                    }
                }
            } else {
                // TD-PREFILL-NONDET ROOT CAUSE (fixed 2026-08-02): build each
                // row's table over the FULL allocation — every owned page of
                // the seq's num_logical global pages — NOT truncated to the
                // row's current token count. The bt_dirty guard deliberately
                // has no token_pos term (TD-PREFILL-KVMETA-BT-SPLIT: "tables
                // depend only on row→seq + allocated pages"), and the
                // REPLICATED branch honors that (writes all num_logical).
                // This branch used to stop at ceil(owned_len(token_pos+1) /
                // page_size): a chunked prefill advances token_pos WITHOUT
                // changing epoch/batch/seq-ids (pages preallocated at
                // creation), so chunk N+1 reused chunk N's tables whose
                // coverage ended at chunk N's length — the chunk_causal
                // union staging then linearized ZEROED entries → physical
                // page 0, i.e. a foreign page (receiving colliding same-
                // launch k_append writes) → the staged ctx KV rows were both
                // WRONG and run-to-run NONDETERMINISTIC. That was the
                // chunk-2 first-gate fork: Ain/Qin/Lin identical, staged
                // 'Kst' rows for the second global-page cycle differing at
                // O(1) with in-run duplicate rows (both stale entries → the
                // same phys page 0). Full coverage restores the guard's
                // premise; reads stay bounded by per-row seqlens.
                for (int rank = 0; rank < dcp_size; ++rank) {
                    auto& m = kv_meta_scratch_[rank];
                    const int rank_gpu =
                        rank < static_cast<int>(
                                   deps_.hidden_state_pairs.size())
                            ? deps_.hidden_state_pairs[rank].gpu_position
                            : -1;
                    for (int l = 0; l < L; ++l) {
                        int* bt = m.host_block_tables.data()
                                + (static_cast<size_t>(l) * batch_size + b)
                                  * max_blocks_per_seq_;
                        for (int j = 0; j < max_blocks_per_seq_; ++j) {
                            const int g = kvshard::global_page_of_local(
                                rank, j, ppc, dcp_size);
                            if (g >= num_logical)
                                break;  // past the allocation — row done
                            const auto& h =
                                pages[static_cast<size_t>(g) * L + l];
                            if (h.gpu_idx != rank_gpu) {
                                // Allocation/routing mismatch — never
                                // reference a page on another GPU
                                // (INV-4.9e).
                                spdlog::error(
                                    "build_kv_metadata: page for global "
                                    "logical {} lives on gpu {} but rank {} "
                                    "is gpu {} (seq {})",
                                    g, h.gpu_idx, rank, rank_gpu, seq_id);
                                return fail();
                            }
                            bt[j] = h.page_idx;
                        }
                    }
                }
            }
        }
    }

    // H2D copy to device scratch (only when real CUDA is available).
    // Per-step arrays on every build; block tables only when bt_dirty
    // (TD-PREFILL-KVMETA-BT-SPLIT) — all layers at once.
    if (deps_.cuda_kernels_enabled && deps_.stream_manager) {
        for (int r = 0; r < dcp_size; ++r) {
            auto& m = kv_meta_scratch_[r];
            if (!m.dev_seqlens_k || !m.dev_block_tables
                || !m.dev_slot_mappings)
                continue;
            if (r >= static_cast<int>(deps_.hidden_state_pairs.size()))
                continue;

            const auto& pair = deps_.hidden_state_pairs[r];
            void* stream = deps_.stream_manager->stream(
                pair.gpu_position, compute::StreamId::kAttention);
            auto* dev = deps_.device_backends[pair.gpu_position];
            dev->set_device();

            dev->memcpy_h2d_async(m.dev_seqlens_k, m.host_seqlens_k.data(),
                                  static_cast<size_t>(batch_size) * sizeof(int),
                                  stream);
            if (bt_dirty)
                dev->memcpy_h2d_async(m.dev_block_tables,
                                      m.host_block_tables.data(),
                                      bt_bytes, stream);
            dev->memcpy_h2d_async(m.dev_slot_mappings, m.host_slot_mappings.data(),
                                  sm_bytes, stream);
            // KVS-2: GLOBAL seqlens (RoPE positions under sharded KV).
            if (sharded && m.dev_global_seqlens)
                dev->memcpy_h2d_async(m.dev_global_seqlens,
                                      host_global_seqlens_.data(),
                                      static_cast<size_t>(batch_size)
                                          * sizeof(int),
                                      stream);
        }
    }

    // TD-51cb: cache sideband fingerprint for dirty guard.
    kv_meta_last_batch_size_ = batch_size;
    if (batch_size > 0) {
        kv_meta_last_seq_id_    = batch_entries[0].seq_id;
        kv_meta_last_token_pos_ = batch_entries[0].token_pos;
    }
    // TD-PREFILL-KVMETA-BT-SPLIT: record the block-table build state.
    if (bt_dirty) {
        kv_bt_built_epoch_ = kv_pages_epoch_;
        kv_bt_batch_size_  = batch_size;
        kv_bt_seq_ids_.resize(static_cast<size_t>(batch_size));
        for (int b = 0; b < batch_size; ++b)
            kv_bt_seq_ids_[static_cast<size_t>(b)] = batch_entries[b].seq_id;
    }

    return KvMetaResult::kOk;
}

CommandDispatcher::KvMetaHostView
CommandDispatcher::kv_meta_host_view(int rank) const {
    KvMetaHostView v{};
    if (rank < 0 || rank >= static_cast<int>(kv_meta_scratch_.size()))
        return v;
    const auto& m = kv_meta_scratch_[rank];
    v.seqlens_k         = m.host_seqlens_k.data();
    v.block_tables      = m.host_block_tables.data();
    v.slot_mappings     = m.host_slot_mappings.data();
    v.max_blocks_per_seq = max_blocks_per_seq_;
    v.kv_layers         = kv_layers_;
    if (kv_sharded_) {
        v.global_seqlens = host_global_seqlens_.data();
        if (rank < static_cast<int>(kv_trash_slot_base_.size()))
            v.trash_slot_base = kv_trash_slot_base_[rank];
    }
    return v;
}

}  // namespace layerstorm::daemon

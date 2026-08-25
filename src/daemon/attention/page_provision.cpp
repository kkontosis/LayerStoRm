// Per-sequence attention page provisioning: GLM/DSA paged indexer-K
// (ensure_indexer_pages) and V4 side tiers kSwa/kHca/LID
// (ensure_v4_tier_pages) — two parallel provisioners moved verbatim from
// dispatch_attention.cpp (attention refactor V2 P1); dedup target of P2.

#include "daemon/command_dispatcher.h"
#include "daemon/attention/kv_codec.h"  // codec axis: per-tier entry bytes
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

// ── TD-GLM-INDEXER-PAGED/-BATCH/-DCP: paged indexer-K provisioning ──────────
//
// Provisions Pool::kIndexerK pages for every computing layer (IndexShare full
// ∪ layer 0) covering token_pos of one sequence, and writes that sequence's
// device page-base pointers into batch row `batch_slot` of the rank host
// tables ([batch * batch_stride + layer * page_stride + page_slot]).
// Replicated dcp_indexer_mode: allocate_indexer_k_for_dcp returns one handle
// per TP GPU in rank order — rank r's table gets rank r's GPU's replica at
// the GLOBAL logical page slot.
// Local mode (TD-GLM-INDEXER-LOCAL-MERGE): ONE handle per page, on the
// OWNER rank's GPU (round-robin by indexer page: owner = pg % dcp); it is
// written ONLY into the owner's table at the LOCAL-compacted slot pg / dcp
// (the executor's producer scores rank r's owned pages as a contiguous
// local run).
//
// TD-INDEXER-POOL-EVICT: the two failure modes are DISTINGUISHED.
// kUnavailable (beyond the serving window / bad slot / dcp mismatch) is a
// permanent property of this shape — the producer falls back (replicated
// B==1: executor arena; local: dense), never an error. kExhausted means the
// POOL is full of OTHER live sequences' pages and is RETRYABLE: the caller
// may raise a pool-exhaustion CMP_ERROR so the orchestrator evicts a prefix
// holder and re-issues (a holder's pages are freed with its sequence).
CommandDispatcher::IndexerPageResult
CommandDispatcher::ensure_indexer_pages(uint64_t seq_id,
                                        uint32_t token_pos,
                                        int batch_slot, int dcp_size) {
    using R = IndexerPageResult;
    if (!deps_.page_allocator || !deps_.live_config) return R::kUnavailable;
    const auto& cfg = *deps_.live_config;
    const int PT = cfg.memory.kv_cache.indexer_k_page_size_tokens;
    if (PT <= 0 || dcp_size < 1) return R::kUnavailable;

    const int n_layers = cfg.model.num_hidden_layers
                       + cfg.model.num_nextn_predict_layers;
    if (indexer_computes_.empty()) {
        // Same rule as DcpExecutor's slot map: IndexShare full ∪ layer 0.
        model::ModelConfig mc(cfg);
        indexer_computes_.assign(static_cast<size_t>(n_layers), 0);
        for (int l = 0; l < n_layers; ++l)
            indexer_computes_[l] =
                (mc.is_full_index_layer(l) || l == 0) ? 1 : 0;
        const int max_seq = cfg.serving.max_sequence_length;
        indexer_page_stride_ = (max_seq + PT - 1) / PT;
        indexer_batch_stride_ = n_layers * indexer_page_stride_;
        indexer_page_table_.assign(dcp_size, {});
        for (auto& t : indexer_page_table_)
            t.assign(static_cast<size_t>(ipc::kMaxBatchDescriptors)
                         * indexer_batch_stride_, nullptr);
        indexer_table_bases_.assign(dcp_size, nullptr);
        for (int r = 0; r < dcp_size; ++r)
            indexer_table_bases_[r] = indexer_page_table_[r].data();
    }
    if (static_cast<int>(indexer_page_table_.size()) != dcp_size)
        return R::kUnavailable;  // dcp_size changed under us

    const int need = static_cast<int>(token_pos) / PT + 1;
    if (need > indexer_page_stride_)
        return R::kUnavailable;  // beyond serving window
    if (batch_slot < 0
        || batch_slot >= static_cast<int>(ipc::kMaxBatchDescriptors))
        return R::kUnavailable;

    // TD-GLM-INDEXER-LOCAL-MERGE: local mode allocates ONE page per (pg, l)
    // on the owner rank's GPU; replicated allocates dcp_size replicas.
    const bool local = dcp_size >= 2
        && cfg.hardware.dcp_indexer_mode == config::DcpIndexerMode::local;
    const int per_group = local ? 1 : dcp_size;

    auto& handles = sequences_[seq_id].indexer_pages;
    // Handles are (logical_page, computing-layer, rank)-ordered; count per
    // (page, layer) is per_group, so coverage = size / (n_computing * group).
    int n_computing = 0;
    for (uint8_t c : indexer_computes_) n_computing += c;
    if (n_computing == 0) return R::kUnavailable;
    int have = static_cast<int>(handles.size()) / (n_computing * per_group);

    for (int pg = have; pg < need; ++pg) {
        for (int l = 0; l < n_layers; ++l) {
            if (!indexer_computes_[l]) continue;
            auto page = deps_.page_allocator->allocate_indexer_k_for_dcp(
                seq_id, static_cast<uint32_t>(pg) * PT,
                static_cast<uint32_t>(l));
            bool ok = static_cast<int>(page.size()) == per_group;
            for (const auto& h : page) ok = ok && h.gpu_ptr;
            if (!ok) {
                // Return any partial allocation of this (pg, l) group.
                for (auto& h : page) deps_.page_allocator->free(h);
                spdlog::warn("ensure_indexer_pages: kIndexerK exhausted at "
                             "seq {} page {} layer {} (seq has {} of {} "
                             "logical pages; pool is sized for "
                             "serving.max_concurrent_requests sequences at "
                             "max_sequence_length) — capacity, retryable",
                             seq_id, pg, l, have, need);
                return R::kExhausted;
            }
            for (const auto& h : page) handles.push_back(h);
        }
    }

    // (Re)write this sequence's rows into batch row `batch_slot`.
    // Replicated: rank r's table gets rank r's replica at the GLOBAL page
    // slot pg. Local: the single handle goes ONLY into the OWNER rank's
    // table (owner = pg % dcp, matching allocate_indexer_k_for_dcp's
    // page-round-robin routing) at the LOCAL-compacted slot pg / dcp; other
    // ranks' slots are untouched (never read — the producer bounds each
    // rank's row by its own owned-page count).
    size_t h = 0;
    for (int pg = 0; pg < need; ++pg) {
        for (int l = 0; l < n_layers; ++l) {
            if (!indexer_computes_[l]) continue;
            if (local) {
                const int owner = pg % dcp_size;
                indexer_page_table_[owner][static_cast<size_t>(batch_slot)
                        * indexer_batch_stride_
                    + static_cast<size_t>(l) * indexer_page_stride_
                    + pg / dcp_size] = handles[h++].gpu_ptr;
            } else {
                for (int r = 0; r < dcp_size; ++r) {
                    indexer_page_table_[r][static_cast<size_t>(batch_slot)
                            * indexer_batch_stride_
                        + static_cast<size_t>(l) * indexer_page_stride_ + pg] =
                        handles[h++].gpu_ptr;
                }
            }
        }
    }
    return R::kOk;
}

// ── Shared side-tier page claim (attention refactor V2 P2 dedup) ──────────
//
// One pool page on `gpu` for (seq, layer): allocate → stamp meta → zero the
// first `zero_bytes` on the kAttention stream. Zeroing is the ticket-J
// determinism contract: side-tier pages are REUSED across sequences — a new
// sequence must never inherit the previous holder's residue (fresh VRAM
// reads ~zero, the state the ticket-H goldens validated; residue made
// run-to-run decode drift). Returns nullopt on pool exhaustion (caller
// logs + fails closed).
std::optional<memory::PageHandle> CommandDispatcher::claim_zeroed_tier_page(
        int gpu, memory::Pool pool, uint64_t seq_id, int layer,
        int64_t zero_bytes) {
    auto h = deps_.page_allocator->allocate(gpu, pool);
    if (!h || !h->gpu_ptr) return std::nullopt;
    auto& meta = deps_.page_allocator->meta(*h);
    meta.sequence_id = seq_id;
    meta.layer_index = static_cast<uint32_t>(layer);
    if (h->gpu_ptr && static_cast<size_t>(gpu) < deps_.device_backends.size()
        && deps_.device_backends[static_cast<size_t>(gpu)]
        && deps_.stream_manager) {
        auto* be = deps_.device_backends[static_cast<size_t>(gpu)];
        be->set_device();
        be->memset_async(h->gpu_ptr, 0, static_cast<size_t>(zero_bytes),
                         deps_.stream_manager->stream(
                             gpu, compute::StreamId::kAttention));
    }
    return h;
}

// ── V4-7b (ticket H): per-seq V4 side-tier page provisioning ──────────────
//
// kSwa: ONE ring page per layer (window == page_tokens ⇒ appends at
// pos % window overwrite exactly the token leaving the window — decode-
// exact). kHca: 2-entry pages per HCA layer grown at 128-boundaries.
// LID (kIndexerK): pages per CSA layer; entry = one CSA compressed block
// (4 tokens), entries per page = indexer_k_page_size_tokens / 4. All
// allocations are fail-closed: a V4 step never runs with missing slots.
bool CommandDispatcher::ensure_v4_tier_pages(uint64_t seq_id,
                                             uint32_t token_pos) {
    if (!deps_.page_allocator || !deps_.live_config || !deps_.dcp_executor)
        return false;
    const auto& cfg = *deps_.live_config;
    const int n_layers = cfg.model.num_hidden_layers;
    const auto& gpus = deps_.dcp_executor->gpus();
    const int n_ranks = gpus.empty() ? 1 : static_cast<int>(gpus.size());

    // Per-tier entry accounting comes from the allocator's V4 layout — the
    // single sizing authority (kv_codec.h, attention refactor V2 P2). All
    // FP8 arms zero the same byte counts as the former local constants
    // (1160 / 132); TQ arms (P3) inherit arm-correct sizes for free.
    const auto& v4l = deps_.page_allocator->v4_layout();

    // V4-2c TP: every rank keeps its replicated tiers on its own GPU.
    auto& t = sequences_[seq_id].v4_tiers;
    if (t.swa.empty()) {
        t.swa.resize(static_cast<size_t>(n_ranks));
        t.hca.resize(static_cast<size_t>(n_ranks));
        t.lid.resize(static_cast<size_t>(n_ranks));
        for (int r = 0; r < n_ranks; ++r) {
            const int gpu = gpus.empty() ? 0 : gpus[static_cast<size_t>(r)]
                                                   .position;
            auto& tr_swa = t.swa[static_cast<size_t>(r)];
            t.hca[static_cast<size_t>(r)]
                .resize(static_cast<size_t>(n_layers));
            t.lid[static_cast<size_t>(r)]
                .resize(static_cast<size_t>(n_layers));
            tr_swa.resize(static_cast<size_t>(n_layers));
            for (int l = 0; l < n_layers; ++l) {
                auto h = claim_zeroed_tier_page(
                    gpu, memory::Pool::kSwa, seq_id, l,
                    static_cast<int64_t>(cfg.model.sliding_window) *
                        v4_tier_entry_bytes(v4l, memory::Pool::kSwa));
                if (!h) {
                    spdlog::error("ensure_v4_tier_pages: kSwa exhausted at "
                                  "seq {} rank {} layer {}", seq_id, r, l);
                    return false;
                }
                tr_swa[static_cast<size_t>(l)] = *h;
            }
        }
    }

    const int PT = cfg.memory.kv_cache.indexer_k_page_size_tokens;
    for (int r = 0; r < n_ranks; ++r) {
        const int gpu = gpus.empty() ? 0
                                     : gpus[static_cast<size_t>(r)].position;
        for (int l = 0; l < n_layers; ++l) {
            const int ratio =
                l < static_cast<int>(cfg.model.compress_ratios.size())
                    ? cfg.model.compress_ratios[static_cast<size_t>(l)] : 0;
            if (ratio <= 0) continue;
            const int entries = static_cast<int>(
                (token_pos + 1) / static_cast<uint32_t>(ratio));
            if (entries <= 0) continue;
            const bool csa = ratio == memory::kV4CsaRatio;
            const int epp = csa ? PT / memory::kV4CsaRatio
                                : memory::kV4LogicalBlockTokens
                                      / memory::kV4HcaRatio;  // 2
            if (epp <= 0) return false;
            auto& vec = csa
                ? t.lid[static_cast<size_t>(r)][static_cast<size_t>(l)]
                : t.hca[static_cast<size_t>(r)][static_cast<size_t>(l)];
            const int need = (entries + epp - 1) / epp;
            const auto pool = csa ? memory::Pool::kIndexerK
                                  : memory::Pool::kHca;
            const int64_t page_zero_bytes =
                epp * v4_tier_entry_bytes(v4l, pool);
            while (static_cast<int>(vec.size()) < need) {
                auto h = claim_zeroed_tier_page(gpu, pool, seq_id, l,
                                                page_zero_bytes);
                if (!h) {
                    spdlog::error("ensure_v4_tier_pages: {} exhausted at seq "
                                  "{} rank {} layer {} page {}",
                                  csa ? "kIndexerK" : "kHca", seq_id, r, l,
                                  vec.size());
                    return false;
                }
                vec.push_back(*h);
            }
        }
    }
    return true;
}

}  // namespace layerstorm::daemon

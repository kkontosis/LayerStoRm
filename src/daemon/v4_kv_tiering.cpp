// TD-V4-KVT (capability-completion P3): V4 CSA-bucket KV tiering — see the
// header for the design record. INV-V4-KVT: placement-only, token-identical
// to tiering-off; capacity fail-closed on repromote, fail-safe (skip) on
// demote.

#include "daemon/v4_kv_tiering.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

namespace layerstorm::daemon {

V4KvTiering::V4KvTiering(Options opts) : opts_(std::move(opts)) {
    // hot_buffer_slots carries the retention window. GLM doc: ~2x
    // index_topk SLOTS; a V4 selection slot is a compressed BLOCK
    // (csa_ratio native tokens), so auto = 2 x index_topk x csa_ratio
    // native tokens (Flash: 4096).
    retention_ = opts_.retention_tokens > 0
        ? opts_.retention_tokens
        : 2 * std::max(opts_.index_topk, 1) * std::max(opts_.csa_ratio, 1);
    const int64_t hot_pages =
        (retention_ + opts_.page_tokens - 1) / opts_.page_tokens;
    const int64_t n_csa = static_cast<int64_t>(
        std::count(opts_.csa_layers.begin(), opts_.csa_layers.end(), 1));
    cold_cap_pages_ = static_cast<int64_t>(
        opts_.host_to_device_ratio * static_cast<double>(hot_pages)
        * static_cast<double>(std::max<int64_t>(n_csa, 1)));
    spdlog::info(
        "V4KvTiering: CSA-bucket tiering armed — retention {} tokens, "
        "page {} tokens/{} B, cold cap {} pages/seq, {} CSA layers "
        "(HCA/LID/SWA exempt by policy)",
        retention_, opts_.page_tokens, opts_.page_bytes, cold_cap_pages_,
        n_csa);
}

V4KvTiering::~V4KvTiering() { maybe_dump_stats(); }

void V4KvTiering::maybe_dump_stats() {
    // Anti-vacuous-gate seam (trap #12): LS_V4_KVT_STATS=<file> appends the
    // counters so the needle test can assert demotions actually happened.
    const char* p = std::getenv("LS_V4_KVT_STATS");
    if (!p || !*p) return;
    if (FILE* f = std::fopen(p, "a")) {
        std::fprintf(f, "demotes=%lld repromotes=%lld skipped_full=%lld\n",
                     static_cast<long long>(stat_demotes_),
                     static_cast<long long>(stat_repromotes_),
                     static_cast<long long>(stat_skipped_full_));
        std::fclose(f);
    }
}

bool V4KvTiering::has_demotions(uint64_t seq_id, int layer) const {
    auto it = seqs_.find(seq_id);
    if (it == seqs_.end() || it->second.cold.empty()) return false;
    // Per-layer scan is bounded by the cold set (small); the common
    // untiered path exits on the map lookup above.
    for (const auto& [k, cp] : it->second.cold) {
        if (static_cast<int>(k >> 32) == layer) return true;
    }
    return false;
}

bool V4KvTiering::seq_has_demotions(uint64_t seq_id) const {
    auto it = seqs_.find(seq_id);
    return it != seqs_.end() && !it->second.cold.empty();
}

void V4KvTiering::set_step_bt(uint64_t seq_id, int layer,
                              std::vector<int*> rows, int len) {
    step_seq_ = seq_id;
    step_layer_ = layer;
    step_bt_ = std::move(rows);
    step_bt_len_ = len;
}

void V4KvTiering::ensure_hot(uint64_t seq_id, int layer,
                             const int* logical_ids, int n) {
    auto it = seqs_.find(seq_id);
    if (it == seqs_.end() || it->second.cold.empty()) return;
    if (seq_id != step_seq_ || layer != step_layer_) {
        throw std::runtime_error(
            "V4KvTiering::ensure_hot: step context mismatch (dispatcher "
            "must set_step_bt for this (seq, layer) before the executor "
            "call)");
    }
    auto& cold = it->second.cold;
    const int ppt = std::max(opts_.page_tokens / opts_.csa_ratio, 1);

    // Collect the needed cold LOGICAL PAGES (dedup via a small stamp set —
    // selections are <= topk ids).
    std::vector<int> pages;
    auto need_page = [&](int block) {
        if (block < 0) return;
        const int lp = block / ppt;
        auto cit = cold.find(key(layer, lp));
        if (cit == cold.end()) return;
        for (int p : pages)
            if (p == lp) return;
        pages.push_back(lp);
    };
    if (logical_ids) {
        for (int i = 0; i < n; ++i) need_page(logical_ids[i]);
    } else {
        for (int b = 0; b < n; ++b) need_page(b);
    }
    if (pages.empty()) return;

    bool any_h2d = false;
    for (int lp : pages) {
        auto cit = cold.find(key(layer, lp));
        auto h = opts_.alloc_page ? opts_.alloc_page(seq_id, layer, lp)
                                  : std::nullopt;
        if (!h) {
            throw std::runtime_error(
                "V4KvTiering::ensure_hot: kMain pool exhausted repromoting "
                "seq " + std::to_string(seq_id) + " layer "
                + std::to_string(layer) + " page " + std::to_string(lp)
                + " — fail-closed (capacity, not correctness)");
        }
        // H2D the cold bytes into EVERY rank's replica page (replicated
        // lockstep page_idx valid against each rank's base).
        for (int r = 0; r < opts_.dcp_size; ++r) {
            auto* be = static_cast<size_t>(r) < opts_.device_backends.size()
                ? opts_.device_backends[static_cast<size_t>(r)] : nullptr;
            void* base = static_cast<size_t>(r) < opts_.kv_main_bases.size()
                ? opts_.kv_main_bases[static_cast<size_t>(r)] : nullptr;
            if (!be || !base) continue;
            be->set_device();
            void* dst = static_cast<uint8_t*>(base)
                      + static_cast<int64_t>(h->page_idx) * opts_.page_bytes;
            void* stream =
                static_cast<size_t>(r) < opts_.attn_streams.size()
                    ? opts_.attn_streams[static_cast<size_t>(r)] : nullptr;
            be->memcpy_h2d_async(dst, cit->second.bytes.data(),
                                 static_cast<size_t>(opts_.page_bytes),
                                 stream);
            any_h2d = true;
        }
        // Update the CURRENT step's host block-table rows in place so the
        // executor's per-row page-table H2D picks up the fresh page.
        for (int* row : step_bt_) {
            if (row && lp < step_bt_len_) row[lp] = h->page_idx;
        }
        ++stat_repromotes_;
        // Drop the cold copy AFTER the copies are durable (sync below —
        // pageable-async sources must outlive stream execution otherwise).
    }
    if (any_h2d) {
        for (int r = 0; r < opts_.dcp_size
                 && static_cast<size_t>(r) < opts_.device_backends.size();
             ++r) {
            if (opts_.device_backends[static_cast<size_t>(r)]) {
                opts_.device_backends[static_cast<size_t>(r)]->set_device();
                opts_.device_backends[static_cast<size_t>(r)]->device_sync();
            }
        }
    }
    for (int lp : pages) cold.erase(key(layer, lp));
    if (it->second.cold.empty()) { /* seq fully hot again */ }
}

void V4KvTiering::after_attention(
        uint64_t seq_id, int layer, int pos,
        const std::vector<memory::PageHandle>& layer_pages, int num_logical,
        int handle_stride) {
    if (layer < 0
        || layer >= static_cast<int>(opts_.csa_layers.size())
        || !opts_.csa_layers[static_cast<size_t>(layer)])
        return;  // policy: only the CSA bucket tiers
    const int frontier = pos + 1 - retention_;
    if (frontier <= 0) return;
    const int last_cold_page = frontier / opts_.page_tokens;  // exclusive
    if (last_cold_page <= 0) return;

    auto& st = seqs_[seq_id];
    for (int lp = 0; lp < std::min(last_cold_page, num_logical); ++lp) {
        const size_t hidx = static_cast<size_t>(lp) * handle_stride
                          + static_cast<size_t>(layer);
        if (hidx >= layer_pages.size()) break;
        const auto& h = layer_pages[hidx];
        if (h.page_idx < 0 || !h.gpu_ptr) continue;   // already cold/neutral
        if (static_cast<int64_t>(st.cold.size()) >= cold_cap_pages_) {
            ++stat_skipped_full_;
            return;  // fail-safe: pool full, page stays hot
        }
        ColdPage cp;
        cp.bytes.resize(static_cast<size_t>(opts_.page_bytes));
        // Rank-0 D2H (replicated computation is bit-identical per rank).
        auto* be0 = opts_.device_backends.empty()
            ? nullptr : opts_.device_backends[0];
        void* base0 = opts_.kv_main_bases.empty()
            ? nullptr : opts_.kv_main_bases[0];
        if (!be0 || !base0) return;
        be0->set_device();
        const void* src = static_cast<const uint8_t*>(base0)
            + static_cast<int64_t>(h.page_idx) * opts_.page_bytes;
        void* stream = opts_.attn_streams.empty()
            ? nullptr : opts_.attn_streams[0];
        be0->memcpy_d2h_async(cp.bytes.data(), src,
                              static_cast<size_t>(opts_.page_bytes), stream);
        be0->device_sync();  // v1: synchronous demote (74 KB — wall noise)
        st.cold.emplace(key(layer, lp), std::move(cp));
        if (opts_.free_page) opts_.free_page(seq_id, layer, lp, h);
        ++stat_demotes_;
    }
}

void V4KvTiering::free_sequence(uint64_t seq_id) { seqs_.erase(seq_id); }

void V4KvTiering::on_seq_fork(uint64_t src_id, uint64_t dst_id) {
    // TD-V4-SERVE-PREFIX: the cold copies are per-seq HOST byte vectors —
    // a deep copy gives the child its own cold pages (its kv_pages carry
    // the same neutralized handles as the parent's; a child repromote
    // allocates the child's OWN fresh VRAM pages through alloc_page).
    // A parent with no demotions is a no-op.
    auto it = seqs_.find(src_id);
    if (it == seqs_.end()) return;
    seqs_[dst_id] = it->second;
}

}  // namespace layerstorm::daemon

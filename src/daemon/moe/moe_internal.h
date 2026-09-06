// Internal helpers shared by the src/daemon/moe/ TUs (driver + arch files).
// Not part of any public interface. Precedent: parallelism/dcp_executor_internal.h.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <spdlog/spdlog.h>

#include "core/device_backend.h"

namespace layerstorm::daemon {

// ── 44z health tripwire: TD-91d zero-fill guard counters ──────────────────────
// A bitset-resident expert whose cache entry is missing (or addressless) at
// pointer-table fill time falls back to the zero-weight / excluded buffer: its
// w_k * expert_out_k contribution to the token is silently ZEROED. That is the
// same silent-loss class as TD-GLM53-EP4-DEGENERATE-GENERATION — plausible but
// wrong tokens, at an INFLATED throughput, with moe_degraded_layers stuck at 0.
//
// Under 44z KV <-> expert zone rebalancing this is the exact residual hazard of
// ticket §5: a reclaim evicts elastic-zone experts through the ordinary evict
// path, and "a dispatch consulted a reclaimed slot" manifests EXACTLY as one of
// these guard fires. So the fires are counted here, and moe_progressive.cpp
// diffs the total around each dispatch and ORs a nonzero diff into st.degraded
// — which reaches pc.moe_degraded and therefore [orch-stats]
// moe_degraded_layers, the counter every gate, ledger and discard rule already
// trusts. No new plumbing.
//
// DEVIATION (44z stage 5, deliberate): the natural home for this field is the
// per-GPU MoeScratch struct held as moe_scratch_[gpu] — but that struct is
// declared in daemon/command_dispatcher.h, which this change may not touch
// (owned by a parallel change). The counters therefore live here as a
// process-wide per-GPU bank. Semantics are unchanged for the single
// CommandDispatcher the daemon runs: the reader diffs a monotonic total around
// one dispatch, and the writers are relaxed atomics because
// dispatch_moe_all_ranks may fill several ranks' pointer tables concurrently.
// Index kZoneGuardMaxGpus is the "unknown gpu" bucket — an out-of-range gpu id
// must never make the signal disappear.
inline constexpr int kZoneGuardMaxGpus = 64;

inline std::atomic<int64_t>* zone_guard_counter_bank() {
    static std::atomic<int64_t> counters[kZoneGuardMaxGpus + 1] = {};
    return counters;
}

/// 44z: TD-91d guard fires on `gpu` — a bitset-resident expert had no cache
/// entry at pointer-table fill time; its contribution was ZEROED. Same silent-
/// loss class as TD-GLM53-EP4; nonzero within a dispatch degrades the layer.
inline int64_t zone_guard_zero_fills(int gpu) {
    const int slot = (gpu < 0 || gpu >= kZoneGuardMaxGpus) ? kZoneGuardMaxGpus : gpu;
    return zone_guard_counter_bank()[slot].load(std::memory_order_relaxed);
}

/// Sum over every GPU (plus the unknown-gpu bucket). This is the honest form
/// for a dispatch that may compute on several ranks: dispatch_moe_all_ranks
/// fills a pointer table on EVERY gpu it touches, so a per-gpu diff would miss
/// fires on the other ranks.
inline int64_t zone_guard_zero_fills_total() {
    int64_t total = 0;
    for (int g = 0; g <= kZoneGuardMaxGpus; ++g)
        total += zone_guard_counter_bank()[g].load(std::memory_order_relaxed);
    return total;
}

/// Record one guard fire. Returns the count already recorded on this gpu BEFORE
/// this one.
inline int64_t zone_guard_note_zero_fill(int gpu) {
    const int slot = (gpu < 0 || gpu >= kZoneGuardMaxGpus) ? kZoneGuardMaxGpus : gpu;
    return zone_guard_counter_bank()[slot].fetch_add(1, std::memory_order_relaxed);
}

/// One guard fire at a pointer-table fill site: count it, and warn LOUDLY on
/// the FIRST fire of this fill (`fires_this_fill` is the caller's per-fill
/// counter, so a whole-layer wipe costs one line, not n_experts lines).
inline void zone_guard_zero_fill_hit(uint32_t layer_idx, int expert, int gpu,
                                     int& fires_this_fill) {
    zone_guard_note_zero_fill(gpu);
    if (fires_this_fill++ == 0) {
        spdlog::warn(
            "44z/TD-91d guard: layer {} expert {} on gpu {} is marked RESIDENT "
            "in the dispatch bitset but has NO addressable cache entry at "
            "pointer-table fill — its contribution to every routed token is "
            "ZEROED (slot reclaimed/evicted mid-flight). Further fires in this "
            "fill are counted silently; the layer will be marked DEGRADED.",
            layer_idx, expert, gpu);
    }
}

/// Close out a fill: report the total when more than one expert was zeroed
/// (the single-fire case is already fully described by the warn above).
inline void zone_guard_zero_fill_fill_done(uint32_t layer_idx, int gpu,
                                           int fires_this_fill) {
    if (fires_this_fill > 1) {
        spdlog::warn("44z/TD-91d guard: {} expert contribution(s) ZEROED in one "
                     "pointer-table fill on layer {} gpu {}.",
                     fires_this_fill, layer_idx, gpu);
    }
}

// ── TD-DRIFT-ROOTCAUSE: DIAGNOSTIC-ONLY routing dump (off by default) ──────────
// Gated on LS_DRIFT_DUMP=<path>. When set, dumps a binary record per
// (sequence, layer, gpu) immediately after the top-K gate runs: the full
// pre-argmax router logit vector + the selected top-K expert ids + their gating
// weights. Two runs (same config twice, OR decay-on vs decay-off) can then be
// diffed at the logit level to locate the FIRST divergence and classify it
// (ULP-scale near-tie argmax flip vs large/garbage). UNSET = zero work, zero
// production impact. When SET it adds a D2H + full device sync, which DOES change
// timing — used only to compare dump-on vs dump-on, and (deliberately) as the
// race probe: forcing this serialization must not change the dumped routing if
// the pipeline is race-free. Record layout (little-endian):
//   int32 hdr[6] = {seq, layer_idx, gpu, num_tokens, n_experts, topk}
//   float logits[num_tokens*n_experts]
//   int32 idx   [num_tokens*topk]
//   float w     [num_tokens*topk]
// (Moved verbatim from dispatch_moe.cpp's anonymous namespace; `inline` so the
// gating hooks in arch_mla_moe.cpp / arch_deepseek_v4_moe.cpp share it.)
inline void drift_dump_routing(compute::DeviceBackend* dev_be, void* stream,
                               const void* router_logits_dev,
                               const void* topk_indices_dev,
                               const void* topk_weights_dev,
                               int num_tokens, int n_experts, int topk,
                               uint32_t layer_idx, int gpu) {
    static const char* path = std::getenv("LS_DRIFT_DUMP");
    if (!path || !*path) return;
    if (!dev_be || !router_logits_dev || !topk_indices_dev || !topk_weights_dev)
        return;
    if (num_tokens <= 0 || n_experts <= 0 || topk <= 0) return;
    static std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return;
    static std::atomic<uint64_t> seq{0};
    const size_t nlog = static_cast<size_t>(num_tokens) * n_experts;
    const size_t nk   = static_cast<size_t>(num_tokens) * topk;
    std::vector<float>   logits(nlog);
    std::vector<int32_t> idx(nk);
    std::vector<float>   w(nk);
    dev_be->set_device();
    dev_be->memcpy_d2h_async(logits.data(), router_logits_dev,
                             nlog * sizeof(float), stream);
    dev_be->memcpy_d2h_async(idx.data(), topk_indices_dev,
                             nk * sizeof(int32_t), stream);
    dev_be->memcpy_d2h_async(w.data(), topk_weights_dev,
                             nk * sizeof(float), stream);
    dev_be->synchronize_device();
    const uint64_t s = seq.fetch_add(1);
    int32_t hdr[6] = {static_cast<int32_t>(s), static_cast<int32_t>(layer_idx),
                      gpu, num_tokens, n_experts, topk};
    std::fwrite(hdr, sizeof(int32_t), 6, fp);
    std::fwrite(logits.data(), sizeof(float), nlog, fp);
    std::fwrite(idx.data(), sizeof(int32_t), nk, fp);
    std::fwrite(w.data(), sizeof(float), nk, fp);
    std::fflush(fp);
}

}  // namespace layerstorm::daemon

// Shared reroute-eviction fallback. See far_evict.h. Lifted verbatim from
// CommandDispatcher::handle_fetch_and_run_moe (dispatch_moe.cpp).
#include "core/gpu_loader/far_evict.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/gpu_loader/loader_evict_scores.h"
#include "core/memory/expert_cache.h"

namespace layerstorm::gpu_loader {

void apply_far_evictions(memory::ExpertCache& cache, EvictScoreBoard* board,
                         bool have_evict_map, bool lru_fallback,
                         std::span<const FarEvictFetch> fetches,
                         uint32_t cur_layer, const EvictFn& evict_fn,
                         FarEvictStats& stats) {
  // Per target GPU: how many NEW (missing, to-be-fetched) experts need slots,
  // and the set of experts that must stay resident this layer (the routed K).
  std::unordered_map<int, int> need_by_gpu;
  std::unordered_map<int, std::unordered_set<uint16_t>> keep_by_gpu;
  for (const auto& er : fetches) {
    keep_by_gpu[er.target_gpu].insert(er.key.expert_idx);
    if (er.is_real_fetch) ++need_by_gpu[er.target_gpu];
  }

  // ── 13c-2.0 Option A: honor an orchestrator-supplied eviction map ──
  if (have_evict_map) {
    for (const auto& er : fetches) {
      if (!er.is_real_fetch) continue;  // not a real fetch
      if (!er.has_victim) continue;     // sentinel → fallback
      const int gpu = er.target_gpu;
      if (cache.free_slots(gpu, memory::CacheZone::kStable) >= need_by_gpu[gpu])
        continue;  // GPU already has room
      const auto vkey = er.victim;
      if (vkey.layer_idx == cur_layer &&
          keep_by_gpu[gpu].count(vkey.expert_idx)) {
        ++stats.rejected;
        continue;  // needed this layer
      }
      if (cache.is_locked(vkey, gpu)) {
        ++stats.rejected;
        continue;
      }
      if (evict_fn(vkey, gpu))
        ++stats.honored;
      else
        ++stats.rejected;
    }
  }

  // ── Local fallback: free any remaining shortfall per GPU ──
  // Guard-aware eviction of one candidate key (shared by both passes): never
  // evict an expert this command needs THIS layer / a locked entry / an
  // ELM-pending load; on success decrement to_free + bump the counter.
  auto try_evict = [&](memory::ExpertKey vk, int gpu, uint32_t cur_layer_,
                       const std::unordered_set<uint16_t>& keep, int& to_free) {
    if (vk.layer_idx == cur_layer_ && keep.count(vk.expert_idx)) return;
    if (cache.is_locked(vk, gpu)) return;
    if (evict_fn(vk, gpu)) {
      --to_free;
      ++stats.fallback;
    }
  };
  std::vector<memory::ExpertKey> victims;  // reused per GPU (heap walk)
  for (const auto& [gpu, need] : need_by_gpu) {
    int free = cache.free_slots(gpu, memory::CacheZone::kStable);
    if (free >= need) continue;
    int to_free = need - free;
    const auto& keep = keep_by_gpu[gpu];
    const int g = gpu;
    const bool board_path = board && lru_fallback;
    if (board_path) {
      // O(to_free·log N) walk: the to_free cheapest (LRU) board victims, which
      // ARE ExpertCache stable residents (parity).
      int board_resident = 0;
      board->cheapest_keys(g, to_free, victims, &board_resident);
      for (const auto& vk : victims) {
        if (to_free <= 0) break;
        try_evict(vk, gpu, cur_layer, keep, to_free);
      }
      // Guards skipped some of the cheapest → widen to the full ascending
      // residency (rare; O(N·log N) only when guards bite).
      if (to_free > 0 && static_cast<int>(victims.size()) < board_resident) {
        board->cheapest_keys(g, board_resident, victims, nullptr);
        for (const auto& vk : victims) {
          if (to_free <= 0) break;
          try_evict(vk, gpu, cur_layer, keep, to_free);
        }
      }
      if (to_free <= 0) continue;
    }
    // Residual / legacy fall-through: ExpertCache truth in hash order. The
    // SOLE, BYTE-IDENTICAL path when board_path is false (loader-off baseline /
    // kill-switch); also a final safety net if the board path could not free
    // enough (all cheap victims guarded out).
    auto residents = cache.eviction_inputs(gpu, memory::CacheZone::kStable);
    for (const auto& ri : residents) {
      if (to_free <= 0) break;
      try_evict(ri.key, gpu, cur_layer, keep, to_free);
    }
  }
}

}  // namespace layerstorm::gpu_loader

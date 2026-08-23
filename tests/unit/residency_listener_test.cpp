// CPU unit tests for the TD-EVICT-BOARD-DESYNC listener-interface fix.
//
// Covers:
//   (A) ExpertCache fires memory::ResidencyListener at its OWN stable-zone
//       choke-points — reserve(kStable)/promote → on_resident_added; evict of a
//       stable entry / demote / reserve-cancel (cancel routes through evict) →
//       on_resident_removed; the streaming zone is NEVER reported; a null
//       listener (default) is byte-identical (no fire).
//   (A') EvictScoreBoard-as-listener PARITY: after an arbitrary cache add/evict
//       sequence the board's per-GPU residency EXACTLY equals the cache's
//       stable-zone residency (the debug parity invariant).
//   (B) Always-on per-access recency: advance_recency + on_resident_added (stamp
//       at the current clock) + touch_existing (re-stamp routed use) orders
//       victims by TRUE per-layer-grouped LRU with NO decay feed.
//   (C) cheapest_keys driven over the board returns the LRU victims ascending
//       (oldest first) — the eviction-fallback selection equivalence.
//
// Pure CPU (NullDeviceBackend), no CUDA, no GPU.
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "config/config_resolver.h"
#include "core/device_backend.h"
#include "core/gpu_loader/loader_evict_scores.h"
#include "core/memory/eviction_policy.h"
#include "core/memory/expert_cache.h"
#include "core/memory/residency_listener.h"
#include "core/memory/vram_allocator.h"
#include "core/null_device_backend.h"
#include "model/quantization/nvfp4.h"

namespace lc = layerstorm::config;
namespace lcomp = layerstorm::compute;
namespace lmem = layerstorm::memory;
namespace lmod = layerstorm::model;
namespace gl = layerstorm::gpu_loader;

namespace {

lmem::ExpertKey key(uint32_t layer, uint16_t expert) { return {layer, expert}; }

// ── A recording listener: captures every add/remove the cache fires ─────────
struct RecordingListener : public lmem::ResidencyListener {
  std::vector<std::pair<lmem::ExpertKey, int>> added;
  std::vector<std::pair<lmem::ExpertKey, int>> removed;
  void on_resident_added(lmem::ExpertKey k, int g) override {
    added.emplace_back(k, g);
  }
  void on_resident_removed(lmem::ExpertKey k, int g) override {
    removed.emplace_back(k, g);
  }
  int added_count(lmem::ExpertKey k, int g) const {
    return static_cast<int>(std::count(added.begin(), added.end(),
                                        std::make_pair(k, g)));
  }
  int removed_count(lmem::ExpertKey k, int g) const {
    return static_cast<int>(std::count(removed.begin(), removed.end(),
                                        std::make_pair(k, g)));
  }
};

lc::Config v32_config() {
  auto j = nlohmann::json{
      {"model",
       {{"architecture", "deepseek_v3"},
        {"weights_path", "/data/models/deepseek-v3.2/"},
        {"weights_format", "safetensors"},
        {"num_hidden_layers", 61},
        {"hidden_size", 7168},
        {"num_attention_heads", 128},
        {"num_key_value_heads", 128},
        {"intermediate_size", 18432},
        {"n_routed_experts", 256},
        {"n_shared_experts", 1},
        {"num_experts_per_tok", 8},
        {"n_group", 8},
        {"topk_group", 4},
        {"vocab_size", 129280},
        {"max_position_embeddings", 163840},
        {"kv_lora_rank", 512},
        {"q_lora_rank", 1536},
        {"qk_rope_head_dim", 64},
        {"qk_nope_head_dim", 128},
        {"v_head_dim", 128},
        {"first_k_dense_replace", 3},
        {"moe_layer_freq", 1},
        {"index_topk", 2048},
        {"index_n_heads", 64},
        {"index_head_dim", 128},
        {"num_nextn_predict_layers", 1},
        {"rms_norm_eps", 1e-6},
        {"rope_theta", 10000.0},
        {"routed_scaling_factor", 2.5},
        {"moe_intermediate_size", 2048}}},
      {"quantization",
       {{"weights", "nvfp4"},
        {"attention_compute", "fp8_e4m3"},
        {"kv_cache", "fp8_e4m3"},
        {"gating_compute", "fp32"}}},
      {"serving", {{"max_sequence_length", 4096}, {"max_concurrent_requests", 4}}},
      {"hardware",
       {{"gpus",
         {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 32}},
          {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 32}},
          {{"id", 2}, {"type", "rtx5080"}, {"vram_gb", 16}},
          {{"id", 3}, {"type", "rtx5080"}, {"vram_gb", 16}}}},
        {"tp_array", {0, 1}},
        {"system_ram_gb", 256}}}};
  return lc::parse_config(j);
}

struct NullBackends {
  std::vector<std::unique_ptr<lcomp::DeviceBackend>> owned;
  std::vector<lcomp::DeviceBackend*> ptrs;
  explicit NullBackends(const lmem::VramLayout& layout) {
    for (size_t i = 0; i < layout.gpus.size(); ++i) {
      lc::GpuRef ref{.position = static_cast<int>(i), .id = layout.gpus[i].gpu_id};
      owned.push_back(lcomp::make_null_device_backend(ref));
      ptrs.push_back(owned.back().get());
    }
  }
};

struct Ctx {
  lc::Config cfg;
  lmod::ModelConfig mcfg;
  int64_t expert_bytes;
  NullBackends backends;
  lmem::VramAllocator vram;
  lmem::ExpertCache cache;
};

Ctx make_ctx() {
  auto cfg = v32_config();
  lmod::ModelConfig mcfg(cfg);
  lmod::Nvfp4 nvfp4;
  lmod::LayerRegistry reg(mcfg, cfg, nvfp4);
  int64_t expert_bytes = reg.per_routed_expert_bytes();
  auto layout = lmem::compute_vram_layout(cfg, reg, mcfg);
  NullBackends nb(layout);
  auto vram = lmem::VramAllocator(std::move(layout), nb.ptrs);
  auto cache = lmem::ExpertCache(vram, cfg, expert_bytes);
  return Ctx{std::move(cfg), std::move(mcfg), expert_bytes,
             std::move(nb), std::move(vram), std::move(cache)};
}

// Brute-force: the cache's stable residents on GPU g (the parity reference).
std::unordered_set<lmem::ExpertKey> cache_stable(const lmem::ExpertCache& c,
                                                 int g) {
  std::unordered_set<lmem::ExpertKey> s;
  for (const auto& ri : c.residency_snapshot(g))
    if (ri.zone == lmem::CacheZone::kStable) s.insert(ri.key);
  return s;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// (A) ExpertCache fires the listener at its stable-zone choke-points
// ═══════════════════════════════════════════════════════════════════════════

// Default (null) listener: every op succeeds and NOTHING is reported (the
// loader-off baseline is byte-identical — no listener present).
TEST(ResidencyListener, NullListenerNeverFires) {
  auto ctx = make_ctx();
  EXPECT_EQ(ctx.cache.residency_listener(), nullptr);
  ASSERT_NE(ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable), nullptr);
  ASSERT_NE(ctx.cache.reserve(key(3, 1), 0, lmem::CacheZone::kStreaming), nullptr);
  EXPECT_TRUE(ctx.cache.evict(key(3, 0), 0));
  // No crash, no listener → the path is unchanged.
  EXPECT_EQ(ctx.cache.residency_listener(), nullptr);
}

// A stable reserve fires on_resident_added; evict of that stable entry fires
// on_resident_removed. The cancel / never-arrived path routes through the SAME
// evict(), so it is covered identically (no separate cancel hook needed).
TEST(ResidencyListener, StableReserveAndEvictFire) {
  auto ctx = make_ctx();
  RecordingListener rec;
  ctx.cache.set_residency_listener(&rec);
  EXPECT_EQ(ctx.cache.residency_listener(), &rec);

  ASSERT_NE(ctx.cache.reserve(key(3, 10), 0, lmem::CacheZone::kStable), nullptr);
  EXPECT_EQ(rec.added_count(key(3, 10), 0), 1);
  EXPECT_EQ(rec.removed_count(key(3, 10), 0), 0);

  // Evict (== ordinary evict AND the cancel / never-arrived path).
  EXPECT_TRUE(ctx.cache.evict(key(3, 10), 0));
  EXPECT_EQ(rec.removed_count(key(3, 10), 0), 1);
}

// The streaming zone is NEVER reported — the listener tracks STABLE only.
TEST(ResidencyListener, StreamingZoneIsNeverReported) {
  auto ctx = make_ctx();
  RecordingListener rec;
  ctx.cache.set_residency_listener(&rec);

  ASSERT_NE(ctx.cache.reserve(key(5, 7), 2, lmem::CacheZone::kStreaming), nullptr);
  EXPECT_TRUE(rec.added.empty()) << "streaming reserve must NOT fire added";
  EXPECT_TRUE(ctx.cache.evict(key(5, 7), 2));
  EXPECT_TRUE(rec.removed.empty()) << "streaming evict must NOT fire removed";
}

// promote (streaming→stable) fires added; demote (stable→streaming) fires
// removed — the zone-crossing choke-points.
TEST(ResidencyListener, PromoteDemoteFireAddedRemoved) {
  auto ctx = make_ctx();
  RecordingListener rec;
  ctx.cache.set_residency_listener(&rec);

  ASSERT_NE(ctx.cache.reserve(key(4, 3), 1, lmem::CacheZone::kStreaming), nullptr);
  EXPECT_TRUE(rec.added.empty());           // streaming reserve: silent
  EXPECT_TRUE(ctx.cache.promote(key(4, 3), 1));
  EXPECT_EQ(rec.added_count(key(4, 3), 1), 1);   // promote → added
  EXPECT_TRUE(ctx.cache.demote(key(4, 3), 1));
  EXPECT_EQ(rec.removed_count(key(4, 3), 1), 1);  // demote → removed
}

// ═══════════════════════════════════════════════════════════════════════════
// (A') EvictScoreBoard-as-listener: residency PARITY with the cache stable zone
// ═══════════════════════════════════════════════════════════════════════════

TEST(ResidencyListener, BoardMirrorsCacheStableResidencyParity) {
  auto ctx = make_ctx();
  const int M = ctx.cache.gpu_count();
  gl::EvictScoreBoard board(M, 64);
  ctx.cache.set_residency_listener(&board);

  // An arbitrary add/evict/promote/demote sequence across two GPUs.
  ASSERT_NE(ctx.cache.reserve(key(3, 0), 0, lmem::CacheZone::kStable), nullptr);
  ASSERT_NE(ctx.cache.reserve(key(3, 1), 0, lmem::CacheZone::kStable), nullptr);
  ASSERT_NE(ctx.cache.reserve(key(3, 2), 1, lmem::CacheZone::kStable), nullptr);
  ASSERT_NE(ctx.cache.reserve(key(4, 5), 1, lmem::CacheZone::kStreaming), nullptr);
  ctx.cache.promote(key(4, 5), 1);            // streaming→stable → board add
  ctx.cache.evict(key(3, 0), 0);              // board remove
  ASSERT_NE(ctx.cache.reserve(key(3, 9), 0, lmem::CacheZone::kStable), nullptr);
  ctx.cache.demote(key(3, 1), 0);             // stable→streaming → board remove

  // PARITY: board residency == cache stable residency, per GPU.
  std::vector<lmem::ExpertKey> bkeys;
  for (int g = 0; g < M; ++g) {
    auto truth = cache_stable(ctx.cache, g);
    board.resident_keys(g, bkeys);
    std::unordered_set<lmem::ExpertKey> bset(bkeys.begin(), bkeys.end());
    EXPECT_EQ(bset.size(), truth.size()) << "g=" << g;
    EXPECT_EQ(bset, truth) << "g=" << g;
    EXPECT_TRUE(board.check_invariant(g)) << "g=" << g;
  }
  // Concretely: g0 has {(3,1) demoted-out? no → removed; (3,9)}; (3,0) evicted.
  EXPECT_TRUE(board.is_resident(0, key(3, 9)));
  EXPECT_FALSE(board.is_resident(0, key(3, 0)));  // evicted
  EXPECT_FALSE(board.is_resident(0, key(3, 1)));  // demoted out of stable
  EXPECT_TRUE(board.is_resident(1, key(3, 2)));
  EXPECT_TRUE(board.is_resident(1, key(4, 5)));   // promoted in
}

// ═══════════════════════════════════════════════════════════════════════════
// (B) Always-on per-access recency: per-layer-grouped LRU, no decay feed
// ═══════════════════════════════════════════════════════════════════════════

TEST(EvictScoreBoardRecency, AddStampsCurrentClockAndTouchReStamps) {
  gl::EvictScoreBoard b(1, 64);

  // layer-visit 1: add k1 (fresh = MRU at clock 1).
  b.advance_recency();
  b.on_resident_added(key(3, 1), 0);
  // layer-visit 2: add k2.
  b.advance_recency();
  b.on_resident_added(key(3, 2), 0);
  // layer-visit 3: add k3.
  b.advance_recency();
  b.on_resident_added(key(3, 3), 0);

  // raw == recency clock at add time: k1<k2<k3 → k1 is the oldest (cheapest).
  EXPECT_LT(b.raw_of(0, key(3, 1)), b.raw_of(0, key(3, 2)));
  EXPECT_LT(b.raw_of(0, key(3, 2)), b.raw_of(0, key(3, 3)));

  std::vector<lmem::ExpertKey> out;
  b.cheapest_keys(0, 3, out, nullptr);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0], key(3, 1));  // oldest first
  EXPECT_EQ(out[1], key(3, 2));
  EXPECT_EQ(out[2], key(3, 3));

  // layer-visit 4: routed re-use of k1 → re-stamp to the current clock → k1 is
  // now the NEWEST, so k2 becomes the oldest victim.
  b.advance_recency();
  b.touch_existing(0, key(3, 1));
  b.cheapest_keys(0, 3, out, nullptr);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0], key(3, 2));  // k2 now oldest
  EXPECT_EQ(out[1], key(3, 3));
  EXPECT_EQ(out[2], key(3, 1));  // k1 freshest
  EXPECT_TRUE(b.check_invariant(0));
}

// touch_existing MUST NOT insert a non-resident key (would re-open desync source
// (b): a board entry for a not-yet-resident fetch). Insertion is the listener's
// job (on_resident_added) when the cache reserves the slot.
TEST(EvictScoreBoardRecency, TouchExistingNeverInserts) {
  gl::EvictScoreBoard b(2, 16);
  b.advance_recency();
  b.touch_existing(0, key(3, 99));  // absent → silently ignored
  EXPECT_FALSE(b.is_resident(0, key(3, 99)));
  EXPECT_EQ(b.resident_count(0), 0);
  b.touch_existing(5, key(3, 1));   // out-of-range GPU → safe no-op
  EXPECT_TRUE(b.check_invariant(0));
}

// Within a single layer-visit, all added/touched keys share ONE recency value
// (per-layer-grouped LRU): ties within a layer, strict order across layers.
TEST(EvictScoreBoardRecency, PerLayerGroupedTies) {
  gl::EvictScoreBoard b(1, 64);
  b.advance_recency();                 // layer A
  b.on_resident_added(key(3, 1), 0);
  b.on_resident_added(key(3, 2), 0);   // same layer → same recency as k1
  EXPECT_DOUBLE_EQ(b.raw_of(0, key(3, 1)), b.raw_of(0, key(3, 2)));
  b.advance_recency();                 // layer B
  b.on_resident_added(key(3, 3), 0);   // strictly newer than A's keys
  EXPECT_LT(b.raw_of(0, key(3, 1)), b.raw_of(0, key(3, 3)));
  EXPECT_TRUE(b.check_invariant(0));
}

// ═══════════════════════════════════════════════════════════════════════════
// (C) cheapest_keys-driven eviction selection equivalence (LRU victims first)
// ═══════════════════════════════════════════════════════════════════════════

// With always-on recency feeding raw scores, cheapest_keys yields EXACTLY the
// least-recently-used residents, ascending — the victim order the reroute
// fallback now drives off (replacing the O(N) nth_element over cache truth).
TEST(EvictScoreBoardRecency, CheapestKeysAreLruVictims) {
  gl::EvictScoreBoard b(1, 256);
  const int N = 40;
  // Each expert added in its own layer-visit → strictly increasing recency.
  for (int e = 0; e < N; ++e) {
    b.advance_recency();
    b.on_resident_added(key(3, static_cast<uint16_t>(e)), 0);
  }
  EXPECT_TRUE(b.check_invariant(0));

  // Re-touch the first 10 (oldest) in fresh layer-visits → they become newest,
  // so the NEXT-oldest band (experts 10..) are now the cheapest victims.
  for (int e = 0; e < 10; ++e) {
    b.advance_recency();
    b.touch_existing(0, key(3, static_cast<uint16_t>(e)));
  }

  std::vector<lmem::ExpertKey> victims;
  int rc = 0;
  const int to_free = 8;
  int got = b.cheapest_keys(0, to_free, victims, &rc);
  EXPECT_EQ(got, to_free);
  EXPECT_EQ(rc, N);
  ASSERT_EQ(victims.size(), static_cast<size_t>(to_free));
  // The 8 cheapest are experts 10..17 (now the least-recently-used).
  for (int i = 0; i < to_free; ++i)
    EXPECT_EQ(victims[i], key(3, static_cast<uint16_t>(10 + i))) << "i=" << i;

  // Brute-force: the victim set == ALL residents sorted by recency ascending.
  std::vector<lmem::ExpertKey> all;
  for (int e = 0; e < N; ++e) all.push_back(key(3, static_cast<uint16_t>(e)));
  std::stable_sort(all.begin(), all.end(),
                   [&](lmem::ExpertKey a, lmem::ExpertKey c) {
                     return b.score(0, a) < b.score(0, c);
                   });
  std::vector<lmem::ExpertKey> full;
  b.cheapest_keys(0, N, full, nullptr);
  ASSERT_EQ(full.size(), all.size());
  for (size_t i = 0; i < all.size(); ++i)
    EXPECT_NEAR(b.score(0, full[i]), b.score(0, all[i]), 1e-12) << "i=" << i;
}

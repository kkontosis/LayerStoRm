// CPU-only offline replay driving the REAL engine eviction. See offline_sim.h.
#include "offline_sim.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "config/config_resolver.h"
#include "core/device_backend.h"
#include "core/gpu_loader/far_evict.h"
#include "core/gpu_loader/loader_evict_scores.h"
#include "core/memory/eviction_policy.h"
#include "core/memory/expert_cache.h"
#include "core/memory/vram_allocator.h"
#include "core/null_device_backend.h"
#include "daemon/ep_residency_dedup.h"  // SHARED force-ON EP-combine dedup (REUSE)

namespace layerstorm::offline_sim {

namespace lc = layerstorm::config;
namespace lmem = layerstorm::memory;
namespace gl = layerstorm::gpu_loader;

// ── trace loading ───────────────────────────────────────────────────────────
std::vector<TraceRow> load_trace(const std::string& csv_path) {
  std::ifstream in(csv_path);
  if (!in) throw std::runtime_error("offline_sim: cannot open trace " + csv_path);
  std::vector<TraceRow> rows;
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    TraceRow r;
    std::stringstream ss(line);
    std::string cell;
    auto next_int = [&](int& out) {
      if (!std::getline(ss, cell, ',')) throw std::runtime_error("offline_sim: short trace row");
      out = std::stoi(cell);
    };
    next_int(r.token);
    next_int(r.layer);
    next_int(r.slot);
    next_int(r.expert_idx);
    next_int(r.bank);
    rows.push_back(r);  // remaining oracle columns (oj,j,cached0,cached1) ignored
  }
  return rows;
}

std::vector<LayerInstance> group_layers(const std::vector<TraceRow>& rows) {
  std::vector<LayerInstance> out;
  for (const auto& r : rows) {
    if (out.empty() || out.back().token != r.token || out.back().layer != r.layer) {
      LayerInstance li;
      li.token = r.token;
      li.layer = r.layer;
      out.push_back(std::move(li));
    }
    out.back().expert_idx.push_back(r.expert_idx);
    out.back().bank.push_back(r.bank);
  }
  return out;
}

int num_tokens(const std::vector<TraceRow>& rows) {
  int mx = -1;
  for (const auto& r : rows) mx = std::max(mx, r.token);
  return mx + 1;
}

// ── CacheSim (CEILING reference only) ────────────────────────────────────────
CacheSim::CacheSim(int num_gpus, int capacity)
    : g_(num_gpus), capacity_(capacity) {}

void CacheSim::touch(int gpu, uint64_t key) {
  Gpu& g = g_[gpu];
  auto it = g.map.find(key);
  if (it == g.map.end()) return;
  g.order.erase(it->second);
  g.order.push_back(key);
  it->second = std::prev(g.order.end());
}

std::vector<uint64_t> CacheSim::insert(
    int gpu, uint64_t key, const std::unordered_set<uint64_t>& needed) {
  Gpu& g = g_[gpu];
  std::vector<uint64_t> evicted;
  if (g.map.count(key)) { touch(gpu, key); return evicted; }
  while (static_cast<int>(g.order.size()) >= capacity_) {
    bool found = false;
    for (auto it = g.order.begin(); it != g.order.end(); ++it) {
      if (needed.count(*it)) continue;
      uint64_t v = *it;
      g.map.erase(v);
      g.order.erase(it);
      evicted.push_back(v);
      found = true;
      break;
    }
    if (!found) break;
  }
  g.order.push_back(key);
  g.map[key] = std::prev(g.order.end());
  return evicted;
}

// ── helpers ─────────────────────────────────────────────────────────────────
int bank_slot_of_node(const gpu_loader::LoaderConstants& k, int node) {
  for (int b = 0; b < k.num_banks; ++b)
    if (k.banks[b].node == node) return b;
  return 0;
}

double mean_bank_egress_us(const gpu_loader::LoaderConstants& k) {
  double sum = 0.0;
  int cnt = 0;
  for (int b = 0; b < k.num_banks; ++b) { sum += k.banks[b].egress_us; ++cnt; }
  return (cnt > 0 && sum > 0.0) ? (sum / cnt) : 600.0;
}

// ── oracle loader (DEBUG/ASSERTION ONLY) ─────────────────────────────────────
OracleMap load_oracle(const std::string& csv_path) {
  std::ifstream in(csv_path);
  if (!in) throw std::runtime_error("offline_sim: cannot open oracle " + csv_path);
  OracleMap out;
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string cell;
    std::vector<int> v;
    while (std::getline(ss, cell, ',')) v.push_back(std::stoi(cell));
    if (v.size() < 9) continue;
    out[OracleKey{v[0], v[1], v[3]}] = OracleRow{v[5], v[6], v[7], v[8]};
  }
  return out;
}

// ── EP-combine compute-owner dedup (REUSES daemon::dedup_ep_residency) ────────
std::vector<int> ep_combine_owners(
    int num_slots, int num_gpus,
    const std::function<bool(int, int)>& resident,
    const std::vector<int>& target, int* dup_count) {
  const int bytes = (num_slots + 7) / 8;
  // Per-GPU residency bitsets over THIS token's routed slots — the same little-
  // endian bit-per-expert layout dedup_ep_residency expects (slot s ⇒ byte s/8,
  // bit s%8). Local scratch only: the real cache is never touched.
  std::vector<std::vector<uint8_t>> bs(num_gpus, std::vector<uint8_t>(bytes, 0));
  std::vector<uint8_t*> ptrs(num_gpus);
  for (int g = 0; g < num_gpus; ++g) {
    ptrs[g] = bs[g].data();
    for (int i = 0; i < num_slots; ++i)
      if (resident(i, g)) bs[g][i / 8] |= static_cast<uint8_t>(1u << (i % 8));
  }
  const int dups = layerstorm::daemon::dedup_ep_residency(
      ptrs.data(), num_gpus, num_slots);
  if (dup_count) *dup_count = dups;
  std::vector<int> owner(num_slots);
  for (int i = 0; i < num_slots; ++i) {
    int o = -1;  // post-dedup exactly one rank holds each slot (the lowest holder)
    for (int g = 0; g < num_gpus; ++g)
      if (bs[g][i / 8] & (1u << (i % 8))) { o = g; break; }
    owner[i] = (o >= 0) ? o : target[i];  // non-resident slot → keep naive target
  }
  return owner;
}

// ── clean-LRU ceiling (local CacheSim, no reroute) ───────────────────────────
double clean_lru_ceiling(const std::vector<LayerInstance>& layers, int M,
                         int capacity) {
  CacheSim cache(M, capacity);
  uint64_t lookups = 0, hits = 0;
  for (const auto& li : layers) {
    const int K = static_cast<int>(li.expert_idx.size());
    std::vector<std::unordered_set<uint64_t>> needed(M);
    std::vector<uint8_t> hit(K);
    for (int i = 0; i < K; ++i) {
      const int home = li.expert_idx[i] % M;
      needed[home].insert(CacheSim::make_key(li.layer, li.expert_idx[i]));
      hit[i] = cache.resident(home, li.layer, li.expert_idx[i]) ? 1 : 0;
    }
    for (int i = 0; i < K; ++i) {
      ++lookups;
      if (hit[i]) ++hits;
    }
    for (int i = 0; i < K; ++i)
      if (hit[i])
        cache.touch(li.expert_idx[i] % M,
                    CacheSim::make_key(li.layer, li.expert_idx[i]));
    for (int i = 0; i < K; ++i) {
      if (hit[i]) continue;
      const int home = li.expert_idx[i] % M;
      auto key = CacheSim::make_key(li.layer, li.expert_idx[i]);
      cache.insert(home, key, needed[home]);
      cache.touch(home, key);
    }
  }
  return lookups ? static_cast<double>(hits) / lookups : 0.0;
}

// ═══════════════════════════════════════════════════════════════════════════
// Real-engine driver internals
// ═══════════════════════════════════════════════════════════════════════════
namespace {

// Cache slot size for the CPU ExpertCache. Residency logic is purely SLOT-COUNT
// based (eviction frees a slot; a miss reserves a slot), so the slot byte size is
// irrelevant to hit/miss — we pick a tiny value so the NullDeviceBackend's real
// aligned_alloc stays trivially small. The T(j) COST MODEL uses the REAL
// expert_bytes from the calibration JSON (LoaderConstants), independently.
constexpr int64_t kSlotBytes = 256;

// CPU-only DeviceBackend set (one NullDeviceBackend per GPU), mirroring
// residency_listener_test::NullBackends.
struct NullBackends {
  std::vector<std::unique_ptr<compute::DeviceBackend>> owned;
  std::vector<compute::DeviceBackend*> ptrs;
  explicit NullBackends(const lmem::VramLayout& layout) {
    for (size_t i = 0; i < layout.gpus.size(); ++i) {
      lc::GpuRef ref{.position = static_cast<int>(i), .id = layout.gpus[i].gpu_id};
      owned.push_back(compute::make_null_device_backend(ref));
      ptrs.push_back(owned.back().get());
    }
  }
};

// Minimal 2x rtx5090 config (tp_array {0,1}); only the expert_cache / duplication
// fields are consulted by the ExpertCache constructor.
lc::Config make_config(int M) {
  nlohmann::json gpus = nlohmann::json::array();
  for (int i = 0; i < M; ++i)
    gpus.push_back({{"id", i}, {"type", "rtx5090"}, {"vram_gb", 32}});
  nlohmann::json tp = nlohmann::json::array();
  for (int i = 0; i < M; ++i) tp.push_back(i);
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
       {{"gpus", gpus}, {"tp_array", tp}, {"system_ram_gb", 256}}}};
  return lc::parse_config(j);
}

// Manual VRAM layout: capacity stable slots/GPU, everything else zero, so the
// real aligned_alloc is tiny (~capacity*256 B/GPU).
lmem::VramLayout make_layout(int M, int capacity) {
  lmem::VramLayout layout;
  layout.kv_bytes_per_page = 1;
  for (int i = 0; i < M; ++i) {
    lmem::GpuVramLayout g{};
    g.gpu_id = i;
    g.safety_margin_bytes = 0;
    g.pinned_bytes = 0;
    g.kv_main_bytes = 0;
    g.kv_speculation_bytes = 0;
    g.indexer_k_bytes = 0;
    g.expert_streaming_bytes = 0;
    g.streaming_spill_bytes = 0;
    g.streaming_prefetch_bytes = 0;
    g.expert_stable_bytes = static_cast<int64_t>(capacity) * kSlotBytes;
    // + alignment slack so region partitioning never runs past the allocation.
    g.total_vram_bytes = g.expert_stable_bytes + 4096;
    layout.gpus.push_back(g);
  }
  return layout;
}

// EvictScoreBoard-fed convex evict_cum, mirroring shadow_solve_and_log
// (dispatch_loader.cpp). unit already folds in the γ horizon weight.
void build_evict_cum(const gl::EvictScoreBoard& board, const lmem::ExpertCache& cache,
                     int M, int num_experts, double unit,
                     std::vector<std::vector<double>>& out) {
  out.assign(M, {});
  const size_t curve_len = static_cast<size_t>(num_experts) + 1;
  std::vector<double> vcost;
  for (int j = 0; j < M; ++j) {
    const int pos = j;  // calib devices[j].position == j (symmetric 2x5090)
    int resident_count = 0;
    board.cheapest_scores_sorted(pos, num_experts, vcost, &resident_count);
    for (double& v : vcost) v *= unit;
    const double max_v = vcost.empty() ? unit : vcost.back();
    const int fs = cache.free_slots(pos, lmem::CacheZone::kStable);
    auto& cum = out[j];
    cum.assign(curve_len, 0.0);
    double acc = 0.0;
    for (size_t n = 1; n < curve_len; ++n) {
      const int evicts = static_cast<int>(n) - fs;
      if (evicts <= 0) { cum[n] = 0.0; continue; }
      const int vi = evicts - 1;
      acc += (vi < static_cast<int>(vcost.size())) ? vcost[vi] : max_v;
      cum[n] = acc;
    }
  }
}

// Orchestrator per-GPU clean LRU (decode_step_fetch_and_run's have_evict_map
// model). Keyed by (layer,expert); front = MRU, back = LRU. Models per-GPU
// residency by e%tp assignment — diverges from the real cache under reroute,
// exactly as in the keeper harness.
struct GpuLru {
  struct K { uint32_t layer; uint16_t expert;
             bool operator==(const K& o) const {
               return layer == o.layer && expert == o.expert; } };
  struct Kh { size_t operator()(const K& k) const {
    return (static_cast<size_t>(k.layer) << 16) | k.expert; } };
  int capacity = 0;
  std::deque<K> order;
  std::unordered_set<K, Kh> resident;
  void touch(const K& k) {
    for (auto it = order.begin(); it != order.end(); ++it)
      if (*it == k) { order.erase(it); break; }
    order.push_front(k);
  }
};

}  // namespace

VariantResult run_variant(const VariantSpec& spec,
                          const std::vector<LayerInstance>& layers,
                          int num_tokens_,
                          const gpu_loader::LoaderConstants& k,
                          int capacity, const OracleMap* oracle,
                          DecisionLog* dlog) {
  const int M = k.num_devices;

  // Real CPU engine state.
  lc::Config cfg = make_config(M);
  lmem::VramLayout layout = make_layout(M, capacity);
  NullBackends nb(layout);
  lmem::VramAllocator vram(std::move(layout), nb.ptrs);
  lmem::ExpertCache cache(vram, cfg, kSlotBytes);
  gl::EvictScoreBoard board(M, capacity + 64);
  if (spec.use_board) cache.set_residency_listener(&board);

  gpu_loader::LoaderSolver solver, eval_solver;
  std::vector<GpuLru> orch(M);
  for (auto& o : orch) o.capacity = capacity;

  const double unit = mean_bank_egress_us(k) * kEvictWeight;

  // DRIFT-PROBE bounded-recency prototype: a sim-side monotonic recency counter
  // that REPLACES the board's unbounded internal clock when spec.recency_inc > 0.
  // Fed to the board through update_existing_only (public API) so the raw spread
  // stays O(base_score) and the rank·base duplicate penalty actually bites.
  const bool bounded = spec.recency_inc > 0.0;
  double brec = 0.0;

  VariantResult res;
  res.name = spec.name;
  res.tokens = num_tokens_;

  auto ready = [&](lmem::ExpertKey key, int g) {
    const auto* ce = cache.lookup(key, g);
    return ce && (ce->sub_components_ready & lmem::SubComponent::kAll) ==
                     lmem::SubComponent::kAll;
  };

  int last_layer = -1;
  bool first = true;
  for (const auto& li : layers) {
    const int K = static_cast<int>(li.expert_idx.size());
    const uint32_t cur_layer = static_cast<uint32_t>(li.layer);
    const bool new_token = first || li.layer < last_layer;
    first = false;
    last_layer = li.layer;

    std::vector<lmem::ExpertKey> key(K);
    std::vector<int> home(K), target(K);
    for (int i = 0; i < K; ++i) {
      key[i] = {cur_layer, static_cast<uint16_t>(li.expert_idx[i])};
      home[i] = li.expert_idx[i] % M;
      target[i] = home[i];
    }

    // ── orchestrator GpuLru victim map (have_evict_map source) ──────────────
    std::vector<bool> has_victim(K, false);
    std::vector<lmem::ExpertKey> victim(K);
    for (int g = 0; g < M; ++g) {
      std::vector<uint16_t> want;
      std::vector<int> miss_ei;
      for (int i = 0; i < K; ++i) {
        if (home[i] != g) continue;
        GpuLru::K lk{cur_layer, static_cast<uint16_t>(li.expert_idx[i])};
        want.push_back(static_cast<uint16_t>(li.expert_idx[i]));
        if (orch[g].resident.count(lk)) orch[g].touch(lk);
        else miss_ei.push_back(i);
      }
      auto needed_now = [&](const GpuLru::K& kk) {
        if (kk.layer != cur_layer) return false;
        return std::find(want.begin(), want.end(), kk.expert) != want.end();
      };
      int need_room = static_cast<int>(miss_ei.size());
      size_t vi = 0;
      while (static_cast<int>(orch[g].resident.size()) + need_room > orch[g].capacity
             && vi < miss_ei.size()) {
        GpuLru::K v{}; bool found = false;
        for (auto it = orch[g].order.rbegin(); it != orch[g].order.rend(); ++it)
          if (!needed_now(*it)) { v = *it; found = true; break; }
        if (!found) break;
        int ei = miss_ei[vi++];
        has_victim[ei] = true;
        victim[ei] = {v.layer, v.expert};
        orch[g].resident.erase(v);
        for (auto it = orch[g].order.begin(); it != orch[g].order.end(); ++it)
          if (*it == v) { orch[g].order.erase(it); break; }
      }
      for (int ei : miss_ei) {
        GpuLru::K lk{cur_layer, static_cast<uint16_t>(li.expert_idx[ei])};
        orch[g].resident.insert(lk);
        orch[g].touch(lk);
      }
    }

    // ── residency at e%tp home (was_cached) ──────────────────────────────────
    std::vector<uint8_t> was_cached(K);
    for (int i = 0; i < K; ++i)
      was_cached[i] = ready(key[i], home[i]) ? 1 : 0;

    // ── solve-time residency snapshot (oracle cached0/cached1 diff) ──────────
    std::vector<uint8_t> c0(K), c1(K);
    for (int i = 0; i < K; ++i) {
      c0[i] = cache.is_resident(key[i], 0) ? 1 : 0;
      c1[i] = (M > 1 && cache.is_resident(key[i], 1)) ? 1 : 0;
    }

    // ── route_moe_by_loader: recency + decay + solve + reroute ───────────────
    if (spec.use_board) {
      if (bounded) {
        // Bounded external recency: advance the sim counter once per layer-visit
        // and stamp this layer's routed-resident experts at their HOME GPU.
        brec += spec.recency_inc;
        for (int i = 0; i < K; ++i) {
          if (!cache.is_resident(key[i], home[i])) continue;
          const gl::ScoreUpdate u{home[i], key[i], brec};
          board.update_existing_only(std::span<const gl::ScoreUpdate>(&u, 1));
        }
      } else {
        board.advance_recency();
        for (int i = 0; i < K; ++i) board.touch_existing(home[i], key[i]);
      }
      if (spec.decay && new_token) board.decay_all(kDecay);
    }

    gpu_loader::SolveRequest req;
    req.num_devices = M;
    req.num_experts = K;
    req.bank_of.resize(K);
    req.cached.assign(static_cast<size_t>(K) * M, 0);
    req.subprep_us.assign(K, 0.0);
    req.place.clear();
    req.clamp_place = true;
    for (int i = 0; i < K; ++i) {
      req.bank_of[i] = bank_slot_of_node(k, li.bank[i]);
      for (int j = 0; j < M; ++j)
        if (ready(key[i], j)) req.cached[static_cast<size_t>(i) * M + j] = 1;
    }
    if (spec.use_board)
      build_evict_cum(board, cache, M, K, unit, req.evict_cum);

    if (spec.loader_act) {
      const auto r = solver.solve(k, req);
      for (int i = 0; i < K; ++i) {
        const int sp = (i < r.n) ? r.assignment[i] : -1;
        if (sp >= 0 && sp < M && !was_cached[i] && sp != target[i]) {
          target[i] = sp;  // reroute this miss
          ++res.reroute_count;
          // A reroute landing on a GPU that already holds this expert is a free
          // hit (no H2D, no duplicate); a reroute onto a GPU that does NOT hold it
          // while a copy is resident elsewhere is what mints a cross-GPU duplicate.
          if (cache.is_resident(key[i], sp)) ++res.reroute_to_cached;
        }
      }
    }

    // ── F-6 decider (all missing fetched, no caps) + locks ───────────────────
    for (int i = 0; i < K; ++i)
      if (was_cached[i]) cache.lock(key[i], target[i]);

    // ── eviction via the SHARED engine helper ────────────────────────────────
    std::vector<gl::FarEvictFetch> fetches(K);
    for (int i = 0; i < K; ++i) {
      auto& f = fetches[i];
      f.key = key[i];
      f.target_gpu = target[i];
      f.is_real_fetch = !was_cached[i];
      f.has_victim = has_victim[i];
      if (has_victim[i]) f.victim = victim[i];
    }
    gl::FarEvictStats st;
    const int li_token = li.token;
    gl::apply_far_evictions(
        cache, spec.use_board ? &board : nullptr, /*have_evict_map=*/true,
        spec.lru_fallback, std::span<const gl::FarEvictFetch>(fetches), cur_layer,
        [&](lmem::ExpertKey vk, int g) {
          const bool ok = cache.evict(vk, g);
          if (ok && dlog)
            dlog->evicts.push_back(DecisionLog::Evict{
                li_token, static_cast<int>(cur_layer), g, vk.layer_idx,
                vk.expert_idx});
          return ok;
        },
        st);
    res.ev_honored  += st.honored;
    res.ev_rejected += st.rejected;
    res.ev_fallback += st.fallback;

    // ── fetch misses + hit/miss accounting (keeper: miss == H2D transfer) ────
    for (int i = 0; i < K; ++i) {
      ++res.lookups;
      uint8_t hit = 0;
      if (was_cached[i]) {           // resident at home → no transfer
        ++res.hits;
        hit = 1;
      } else if (cache.is_resident(key[i], target[i])) {  // rerouted target resident
        ++res.hits;                                        // no H2D
        hit = 1;
      } else {
        void* addr = cache.reserve(key[i], target[i], lmem::CacheZone::kStable,
                                   /*is_duplicate=*/false);
        if (addr) {
          cache.mark_all_ready(key[i], target[i]);
          // Bounded recency: stamp the freshly-resident copy at the current brec
          // (the listener's on_resident_added used the bounded-mode 0 clock).
          if (spec.use_board && bounded) {
            const gl::ScoreUpdate u{target[i], key[i], brec};
            board.update_existing_only(std::span<const gl::ScoreUpdate>(&u, 1));
          }
          ++res.transfers;
          if (cache.resident_gpus(key[i]).size() > 1) ++res.dup_inserts;
        } else {
          ++res.reserve_failures;    // non-resident, no slot, no transfer
        }
      }
      if (dlog)
        dlog->lookups.push_back(DecisionLog::Lookup{
            li.token, li.layer, li.expert_idx[i], home[i], target[i],
            was_cached[i], hit});
    }

    // ── unlock cached experts (end-of-command unlock) ────────────────────────
    for (int i = 0; i < K; ++i)
      if (was_cached[i]) cache.unlock(key[i], target[i]);

    // ── cross-GPU duplicate-residency census (fidelity instrumentation) ──────
    // Count (layer-visit-weighted) how many distinct resident keys currently sit
    // on >1 GPU vs the total resident set. This is the steady-state duplicate
    // pressure the EvictScoreBoard's duplicate-aware penalty is meant to bite on
    // — the smoking-gun the hypothesis predicts ACT placement should generate.
    {
      const auto snap = cache.residency_snapshot();
      std::unordered_map<uint64_t, int> count;
      count.reserve(snap.size() * 2);
      for (const auto& ri : snap) {
        if (ri.zone != lmem::CacheZone::kStable) continue;
        const uint64_t fk = (static_cast<uint64_t>(ri.key.layer_idx) << 16) |
                            ri.key.expert_idx;
        ++count[fk];
      }
      for (const auto& [fk, c] : count) {
        ++res.resident_layersum;
        if (c > 1) ++res.dup_resident_layersum;
      }
    }

    // ── EP-combine compute-owner dedup (mirror dispatch_moe_all_ranks force-ON) ──
    // The engine runs EVERY GPU over its resident routed subset and the cross-GPU
    // SUM-combine assumes each routed expert is computed on EXACTLY ONE GPU. The
    // TD-MOE-EP-COMBINE-RESIDUAL fix makes the daemon UNCONDITIONALLY dedup a
    // duplicated routed expert to its lowest-rank holder before the combine
    // (src/daemon/ep_residency_dedup.h). We mirror it here: the per-token compute
    // owner of each routed expert is the canonical (lowest-rank) holder among its
    // post-fetch resident-AND-ready GPUs, so LoaderSolver::evaluate single-counts
    // each routed expert exactly as the engine does — and never attributes the
    // SAME expert's compute to two GPUs (the cost-model double-count this fixes).
    // Pure accounting: the cache is NOT mutated. Disjoint baseline ⇒ owner ==
    // target, dup == 0 ⇒ byte-identical predicted_us (baseline reproduction kept).
    int ep_dups = 0;
    const std::vector<int> exec = ep_combine_owners(
        K, M, [&](int i, int g) { return ready(key[i], g); }, target, &ep_dups);
    res.ep_combine_dups += static_cast<uint64_t>(ep_dups);
    for (int i = 0; i < K; ++i)
      if (exec[i] != target[i]) ++res.ep_owner_shifts;

    // ── T(j) cost of the executed assignment ─────────────────────────────────
    // Evaluate the per-token time from the transfer/compute makespan only: the
    // evict_cum term is a placement-steering future-token CONSEQUENCE, not actual
    // wall time, and the always-on recency clock makes its raw magnitude grow
    // unbounded (TD-EVICT-RECENCY-MAGNITUDE) — feeding it into the cost would
    // explode T(j) by 50-400x. Clearing it leaves the makespan driven by the REAL
    // per-layer miss counts (the fetch-time deltas the tok/s estimate is about).
    req.evict_cum.clear();
    gpu_loader::SolveResult out;
    eval_solver.evaluate(k, req, exec, &out);
    res.predicted_us_total += out.predicted_us;

    // ── oracle cached diff (assertion/debug only) ────────────────────────────
    if (oracle) {
      for (int i = 0; i < K; ++i) {
        auto it = oracle->find(OracleKey{li.token, li.layer, li.expert_idx[i]});
        if (it == oracle->end()) continue;
        ++res.oracle_rows;
        if (it->second.c0 != c0[i] || it->second.c1 != c1[i]) {
          ++res.cached_divergences;
          if (res.first_divergence.empty())
            res.first_divergence =
                "CACHED@(t=" + std::to_string(li.token) + ",l=" +
                std::to_string(li.layer) + ",e=" +
                std::to_string(li.expert_idx[i]) + ") sim=(" +
                std::to_string(c0[i]) + "," + std::to_string(c1[i]) +
                ") oracle=(" + std::to_string(it->second.c0) + "," +
                std::to_string(it->second.c1) + ")";
        }
      }
    }
  }

  res.max_recency = bounded ? brec : board.recency_now();
  res.hit_rate = res.lookups
      ? 1.0 - static_cast<double>(res.transfers) / res.lookups : 0.0;
  res.transfers_per_token = res.tokens
      ? static_cast<double>(res.transfers) / res.tokens : 0.0;
  res.t_pred_ms_per_token = res.tokens
      ? res.predicted_us_total / res.tokens / 1000.0 : 0.0;
  return res;
}

}  // namespace layerstorm::offline_sim

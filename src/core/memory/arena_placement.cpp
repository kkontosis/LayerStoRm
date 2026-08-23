// Arena host-slot placement policy (Wave-2 M3) — see arena_placement.h.

#include "core/memory/arena_placement.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace layerstorm::memory {

namespace {

// FNV-1a-64 (matches arena_cache.cpp's constants; kept local — the policy
// identity is its own namespace, only stability across runs matters).
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

uint64_t fnv1a(const void* data, size_t n, uint64_t h = kFnvOffset) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h;
}

}  // namespace

std::string ArenaPlacementPolicy::resolved_path(
        const std::string& config_freq_table) {
    const char* env = std::getenv("LS_ARENA_PLACE_FREQ");
    if (env && *env) {
        const std::string v(env);
        if (v == "off" || v == "none" || v == "0")
            return "";  // explicit env disable overrides the config default
        return v;       // env path overrides the config
    }
    return config_freq_table;  // "" = disabled (legacy placement)
}

ArenaPlacementPolicy ArenaPlacementPolicy::resolve(
        const std::string& config_freq_table) {
    ArenaPlacementPolicy p;
    p.freq_path = resolved_path(config_freq_table);
    if (p.freq_path.empty()) {
        p.freq_path.clear();
        return p;  // disabled — legacy placement, identity 0
    }

    std::ifstream f(p.freq_path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(
            "arena placement (memory.arena_placement.freq_table / "
            "LS_ARENA_PLACE_FREQ): cannot open frequency table '" +
            p.freq_path + "' (fail-closed: a silent fallback to the legacy "
            "placement would invalidate placement-gated measurements)");
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string bytes = ss.str();
    if (bytes.empty())
        throw std::runtime_error(
            "arena placement: frequency table '" + p.freq_path +
            "' is empty");

    // Identity = policy tag + table CONTENT (not the path — the same table
    // copied elsewhere must adopt the same persisted store).
    static constexpr char kTag[] = "arena-place-freqhbm-v1";
    uint64_t h = fnv1a(kTag, sizeof(kTag) - 1);
    h = fnv1a(bytes.data(), bytes.size(), h);
    if (h == 0) h = 1;  // 0 is reserved for "disabled"
    p.identity = h;
    p.enabled = true;
    return p;
}

ArenaPlacementPolicy ArenaPlacementPolicy::from_env() { return resolve(""); }

ArenaFreqTable ArenaFreqTable::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error(
            "ArenaFreqTable: cannot open '" + path + "'");
    ArenaFreqTable t;
    std::string line;
    size_t lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        // Trim leading whitespace; skip blanks and comments.
        size_t s = line.find_first_not_of(" \t\r");
        if (s == std::string::npos) continue;
        if (line[s] == '#') continue;
        long layer = -1, expert = -1;
        double count = 0.0;
        char c1 = 0, c2 = 0;
        std::istringstream ls(line.substr(s));
        if (!(ls >> layer >> c1 >> expert >> c2 >> count) || c1 != ',' ||
            c2 != ',' || layer < 0 || expert < 0 || expert > 0xFFFF ||
            count < 0.0)
            throw std::runtime_error(
                "ArenaFreqTable: malformed row " + std::to_string(lineno) +
                " in '" + path + "': " + line);
        ExpertKey k{static_cast<uint32_t>(layer),
                    static_cast<uint16_t>(expert)};
        t.counts[k] += count;  // duplicate rows accumulate
    }
    if (t.counts.empty())
        throw std::runtime_error(
            "ArenaFreqTable: no data rows in '" + path + "'");
    return t;
}

ArenaPlacementMap compute_arena_placement(
        const ArenaFreqTable& freq, const std::vector<ExpertKey>& keys,
        std::vector<ArenaPlacementNode> nodes) {
    ArenaPlacementMap out;
    if (keys.empty() || nodes.empty()) return out;

    // Deterministic node order (ascending id) within each class.
    std::sort(nodes.begin(), nodes.end(),
              [](const ArenaPlacementNode& a, const ArenaPlacementNode& b) {
                  return a.node < b.node;
              });
    std::vector<size_t> hbm_idx, ddr_idx;
    for (size_t i = 0; i < nodes.size(); ++i)
        (nodes[i].is_hbm ? hbm_idx : ddr_idx).push_back(i);

    // Global frequency order (freq desc, then key asc — fully deterministic).
    std::vector<ExpertKey> order(keys);
    std::sort(order.begin(), order.end(),
              [&](const ExpertKey& a, const ExpertKey& b) {
                  const double fa = freq.at(a), fb = freq.at(b);
                  if (fa != fb) return fa > fb;
                  return a < b;
              });

    // Round-robin-by-per-layer-rank assignment of `set` over node class
    // `cls` (indices into nodes). Target = cls[(rank + layer) % n]; when
    // full, fall back to the class node with the most free slots. Keys that
    // fit nowhere in the class are returned (for the caller's next stage).
    auto assign_rr = [&](const std::vector<ExpertKey>& set,
                         const std::vector<size_t>& cls)
            -> std::vector<ExpertKey> {
        std::vector<ExpertKey> unplaced;
        if (cls.empty()) return set;
        // Group by layer, preserving the (freq desc) order within each layer.
        std::map<uint32_t, std::vector<ExpertKey>> by_layer;
        for (const auto& k : set) by_layer[k.layer_idx].push_back(k);
        for (auto& [layer, lk] : by_layer) {
            size_t rank = 0;
            for (const auto& k : lk) {
                size_t tgt = cls[(rank + layer) % cls.size()];
                if (nodes[tgt].free_slots == 0) {
                    // Fallback: most free slots (ties: lowest node id).
                    // NOTE: cls holds indices into `nodes` (NOT 0..cls.size()),
                    // so the not-found sentinel must be out-of-band (SIZE_MAX).
                    size_t best = SIZE_MAX;
                    for (size_t ci : cls)
                        if (nodes[ci].free_slots > 0 &&
                            (best == SIZE_MAX ||
                             nodes[ci].free_slots > nodes[best].free_slots))
                            best = ci;
                    if (best == SIZE_MAX) { unplaced.push_back(k); ++rank; continue; }
                    tgt = best;
                }
                --nodes[tgt].free_slots;
                out.emplace(k, nodes[tgt].node);
                ++rank;
            }
        }
        return unplaced;
    };

    // 1. HBM-FIRST: the globally hottest keys fill total HBM free capacity.
    size_t hbm_cap = 0;
    for (size_t i : hbm_idx) hbm_cap += nodes[i].free_slots;
    const size_t n_hot = std::min(hbm_cap, order.size());
    std::vector<ExpertKey> hot(order.begin(), order.begin() + n_hot);
    std::vector<ExpertKey> rest(order.begin() + n_hot, order.end());
    // Anti-gang WITHIN the HBM class too (cheap, and keeps per-layer hot
    // members off a single HBM node).
    std::vector<ExpertKey> hbm_overflow = assign_rr(hot, hbm_idx);
    // Capacity race (shouldn't happen — n_hot == free capacity): push any
    // overflow to the DDR stage.
    rest.insert(rest.begin(), hbm_overflow.begin(), hbm_overflow.end());

    // 2. ANTI-GANG over DDR for everything else (zero-freq tail included —
    // the arena stays full-fit). Whatever fits nowhere is left unassigned
    // (caller's legacy tiered fill handles it).
    assign_rr(rest, ddr_idx);
    return out;
}

}  // namespace layerstorm::memory

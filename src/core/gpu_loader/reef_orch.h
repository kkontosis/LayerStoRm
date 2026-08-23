#pragma once

// REEF orchestrator decision stack (P-25 KEEPER52_REEF_ORCH — extracted from
// tests/integration/{keeper52,dsp52}_test.cpp, 2026-08-18, byte-identical).
//
// The orchestrator-side placement + eviction decision maker over the SAME
// pure components the engine's I8-ACT path uses: the calibrated LoaderSolver
// (LoaderSolver256 for batched-verify unions > kMaxExperts) and the
// EvictScoreBoard (freq-bonus recency, pooled cheapest victims, reuse place
// reward). Residency is SELF-CONSISTENT MODEL STATE — the board is driven
// exactly like dispatch_loader drives the engine's: advance_recency once per
// layer, touch_existing_all(key, w·f) for every routed expert BEFORE the
// solve (i-order), inserts at the current clock when a miss is admitted
// (on_resident_added semantics: update() with score=recency_now), evict()
// for chosen victims. Decisions ship as explicit per-expert gpu_idx + the
// 13c-2.0 victim map (index-aligned; 0xFFFF sentinel = no victim).
//
// CONSUMERS: the keeper52/dsp52 C++ fixtures (test-side arm,
// KEEPER52_REEF_ORCH=1) and the daemon's E_CMD_REEF_ROUTE service (the
// Python orchestrator's REEF arm — the P-18 production line). Both must
// produce the identical decision stream for identical inputs.
//
// Test-only diagnostics stay OUT of the library behind two optional seams
// (null = the plain path, byte-identical to hooks-absent):
//   bank_node_fn(layer, expert) -> host NUMA node (-1 unknown) — the solver
//     bank lookup source (tests: the LS_ARENA_MAP_DUMP CSV; daemon: the
//     live PinnedExpertArena::location_node ground truth the CSV dumps).
//   evict_oracle — the Belady next-use diagnostic (DSP52_EVICT_ORACLE).
//
// CUDA-free (INV-GPU-1): pure CPU decision logic.

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/gpu_loader/loader_constants.h"
#include "core/gpu_loader/loader_evict_scores.h"
#include "core/gpu_loader/loader_solver.h"
#include "core/memory/eviction_policy.h"  // ExpertKey

namespace layerstorm::memory {
class PinnedExpertArena;  // bank-snapshot source (install_arena_bank_seam)
}

namespace layerstorm::gpu_loader {

// Sideband entry VIEWS — layout mirrors ipc::ExpertPrefetchEntry /
// ipc::ExpertEvictionEntry byte-for-byte (8 B each) without a core→daemon
// include. Consumers that hold the ipc types reinterpret_cast and
// static_assert layout equality in their own TU.
struct ReefEntry {
    uint32_t layer_idx;
    uint16_t expert_idx;
    uint8_t  zone;       // 0=stable, 1=streaming (ELB stream-served)
    uint8_t  gpu_idx;
};
static_assert(sizeof(ReefEntry) == 8);

struct ReefVictim {
    uint32_t layer_idx;
    uint16_t expert_idx;  // 0xFFFF = no victim (router/local fallback picks)
    uint8_t  gpu_idx;
    uint8_t  _pad;
};
static_assert(sizeof(ReefVictim) == 8);

/// Belady next-use eviction oracle seam (diagnostic, NOT deployable — uses
/// recorded FUTURE routing). Implemented test-side (NextUseOracle).
class ReefEvictOracle {
public:
    virtual ~ReefEvictOracle() = default;
    /// Next demand STEP of (layer, expert) at/after (cur_step, cur_layer),
    /// or -1 if never again.
    virtual int32_t next_use_step(uint32_t layer, uint16_t expert,
                                  int64_t cur_step, int cur_layer) const = 0;
    /// Belady distance in global layer-visit time (never-again = huge).
    virtual int64_t next_use_dist(uint32_t layer, uint16_t expert,
                                  int64_t cur_step, int cur_layer) const = 0;
};

struct ReefOrch {
    LoaderConstants K;
    LoaderSolver solver;
    // Large unions (> kMaxExperts: batched-verify / prompt-prefill chunks):
    // the 256-bound instantiation. Unions <= 64 keep the frozen production
    // solver (decode byte-identity); beyond kMaxExpertsLarge callers fail
    // loud before route.
    LoaderSolver256 solver_big;
    EvictScoreBoard board;
    SolveRequest req;  // persistent scratch
    std::unordered_map<memory::ExpertKey, float> freq;
    std::vector<int> cap;   // per-GPU stable-zone slot count
    int M = 0;
    // Landed REEF policy defaults (mirror dispatch_loader's env defaults).
    double freq_w = 60.0, freq_mult = 0.9048374180359595;  // exp(-0.1)
    double reuse_w = 2000.0, reuse_tau = 300.0;
    // EPM-0 keeppred (LS_LOADER_KEEPPRED_W; default 0 = OFF). Retention bias
    // on eviction victim selection: keeppred_union[L] = experts routed at
    // layer L the previous time L was visited (the free prev-round-union
    // temporal signal). A resident victim in its layer's union is biased UP
    // the keep order. Never fetches, never changes placement/routing.
    // keeppred_w==0 ⇒ victim order byte-identical to OFF.
    double keeppred_w = 0.0;
    std::vector<std::unordered_set<uint16_t>> keeppred_union;
    // ── Optional seams (null = plain path, byte-identical) ──
    std::function<int(uint32_t layer, uint16_t expert)> bank_node_fn;
    ReefEvictOracle* evict_oracle = nullptr;  // non-owning
    int evict_bridge = 0;  // >0: "recurs<=k tok" bridge instead of Belady
    // Monotonic solve counter (incremented at the end of every
    // reef_orch_route). Correlates out-of-band events (e.g. the arena
    // relocation trace, LS_REEF_RELOC_TRACE) with the decision stream:
    // an event stamped `solve_count == k` happened after the k-th solve.
    uint64_t solve_count = 0;
    // Decision-stream dump (TD-BRIDGE-CPP-GAP probe): armed by
    // LS_REEF_DECISION_DUMP=<file> in make_reef_orch. One line per
    // route ("R <layer> <n> | e:assign:bank:pinned ...") and one per
    // apply with victims ("A <layer> | vl:ve:g ...") — identical format
    // from both drivers (test-side stack and the daemon service) so the
    // first divergence is a plain `diff`. nullptr = zero-cost off.
    std::FILE* decision_dump = nullptr;

    ReefOrch(int tp, std::vector<int> caps, LoaderConstants k);
};

/// Decide `layer`'s placement for the routed union `topk` (single decode
/// row's top-K OR a deduped multi-row chunk union): recency+freq touch pass →
/// SolveRequest (hits pinned to their resident device, reuse place reward,
/// bank_of via bank_node_fn) → exact solve (64) / bounded greedy (256).
/// Fills assign[i] (size >= topk.size()) with the target GPU position.
void reef_orch_route(ReefOrch& o, int layer,
                     const std::vector<uint16_t>& topk,
                     std::vector<uint8_t>& assign);

/// Apply `layer`'s decisions to the model: per GPU ascending, pick pooled
/// cheapest victims for overflowing misses (needed_now excluded; keeppred /
/// Belady variants when armed), fill the 13c-2.0 victim map, evict them from
/// the board, then admit the misses at the current clock. `stream_mask[i]`
/// nonzero = entry served by an ELB streaming-zone copy (no victim, no
/// stable-board admission). cur_step feeds the Belady oracle only.
void reef_orch_apply(ReefOrch& o, int layer, const ReefEntry* entries,
                     ReefVictim* evicts, uint32_t count,
                     const std::vector<uint8_t>* stream_mask = nullptr,
                     int64_t cur_step = -1);

/// INV-REEF-BANK (2026-08-23, TD-BRIDGE-CPP-GAP Q1 flip): install the ONE
/// shared paired-bank-input seam used by BOTH consumers (the daemon REEF
/// service and the C++ test-side stack). Bank inputs come from the arena's
/// PUBLISHED epoch-latched location snapshot (frozen per placement epoch;
/// republished only at online-migrator commit boundaries), with HBM→CPU
/// affinity pairing applied ALWAYS via `paired_node` (raw HBM holding
/// nodes match no calibration bank → silent bank-0 mis-costing; the
/// measured live-raw bridge gap). Before the first publication the seam
/// returns -1 (bank 0) — byte-compatible with the retired boot-CSV race.
/// The closure caches the snapshot per epoch: one atomic epoch load per
/// lookup, refetch only on change. Single-threaded per consumer.
void install_arena_bank_seam(ReefOrch& o,
                             const memory::PinnedExpertArena* arena,
                             std::function<int(int)> paired_node);

/// Shared construction (tests + the daemon E_CMD_REEF_ROUTE service):
/// calibration from `calib_path` (already resolved), caps = per-GPU stable
/// slot counts, policy from env LS_LOADER_POLICY (artifact) + env overrides
/// else built-in defaults, keeppred_w from LS_LOADER_KEEPPRED_W. THROWS
/// std::runtime_error on calibration load failure or a device-count
/// mismatch. desc_out (nullable) receives the policy-source description.
std::unique_ptr<ReefOrch> make_reef_orch(const std::string& calib_path,
                                         int num_gpus,
                                         std::vector<int> caps,
                                         std::string* desc_out = nullptr);

}  // namespace layerstorm::gpu_loader

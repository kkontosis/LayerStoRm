// keeper52 — the full-fit 100-token decode benchmark, ported to GLM-5.2.
//
// This is a CLONE of the production-path offload keeper
//   RoutedExpertLruFullFitDirectTest.HundredTokenDecodeFetchAndRun_FullFit_EP2
//   (tests/integration/first_token_test.cpp; spec/BENCHMARK_FULLFIT.md)
// — the E_CMD_FETCH_AND_RUN_MOE variant (FETCH_AND_RUN_MOE is the engine-wide
// production routed-MoE path; the host-LRU PREFETCH_BATCH + D_B_CMD_RUN_MOE
// TopKLRU keeper is the legacy comparison base). Same fixture structure and
// every keeper property inherited:
//   - decode via decode_step_fetch_and_run(): RUN_ATTENTION with the fused
//     router gate (emit_gating/store_gating) → read the routed top-K from the
//     sideband → ONE E_CMD_FETCH_AND_RUN_MOE per MoE layer spanning BOTH GPUs
//     via per-expert gpu_idx (EP=2 split, e%tp), fetch+progressive-compute
//     fused in the daemon (both PCIe links overlap by construction);
//   - the 13c-2.0 orchestrator LRU eviction map (make_lrus(): per-GPU global
//     (layer,expert) LRU sized to the stable zone; have_evict_map=1 supplies
//     LRU-chosen victims instead of the daemon's arbitrary unranked pick);
//   - seed token 15234 (env LS_KEEPER_SEED_TOKEN overrides);
//   - O_DIRECT expert preload (pin_host_expert_pool_direct_o_direct, the
//     ~3.37 GB/s Gen3-x4 link cap — hit-mostly-safe for full-fit);
//   - overlapped arena registration (pin_host_expert_pool_preload triggers
//     engine.cpp's defer_registration → register_all() on a side thread while
//     the io_uring preload fills the prefaulted arena — TD-INIT-OVERLAP);
//   - TP=2 / EP=2 split on the two RTX 5090s (the 5080s are hidden from CUDA
//     itself so cudaHostRegister does not map the arenas into their contexts);
//   - the full-fit host arena (fraction_total + cross_node_spill on all 4 NUMA
//     nodes, 0 skipped) and precomputed gating (the fused gate IS the gating —
//     FETCH_AND_RUN consumes the stored routing, no double-gate);
//   - the I8 loader-solver wiring: LS_LOADER_SHADOW enables
//     gpu_loader.enabled + calibration_path (default
//     gpu_loader_calibration_5090x2.json weights-adjacent, LS_LOADER_CALIB
//     overrides) so the FETCH handler runs the (behavior-neutral) solver and
//     logs j[·] vs the orchestrator's e%tp; the TearDown trainer hook
//     (LS_LOADER_TRAIN_OUT + LS_LOADER_SHADOW_DUMP) shells out to
//     tools/loader_xray/trainer_apply.py — identical to the FirstTokenTest
//     fixture in first_token_test.cpp.
//
// keeper52 = that keeper, but the GLM-5.2 (GGUF Q4_K_XL) model plus FOUR
// feature deltas, all wired together in ONE fixture:
//   1. MODEL — GLM-5.2 GGUF Q4_K_XL 11-shard set + the glm_5_2_gguf config
//      (glm_moe_dsa: MLA-256 + DSA lightning indexer + MoE 256-expert top-8,
//      78 layers, first_k_dense_replace=3), loaded exactly like
//      glm52_gguf_golden_test / glm52_longctx_golden_test (config
//      test-data/config/glm_5_2_gguf.json, weights
//      test-data/GLM-5.2-GGUF-Q4_K_XL/...-00001-of-00011.gguf, experts from the
//      prepacked set test-data/GLM-5.2-prepacked).
//   2. ARENA BUMP — GLM-5.2's routed set is larger than the keeper's DeepSeek
//      set, so the host arena fraction is raised 0.76 → 0.90 (env-overridable
//      KEEPER52_ARENA_FRACTION). Fit math (mirrors BENCHMARK_FULLFIT):
//        per expert (gate+up+down) = 3 * 2048 * 6144 = 37,748,736 weights;
//        Q4_K = 0.5625 B/weight → 20.25 MiB/expert;
//        MoE layers = 78 - 3 = 75; experts = 75 * 256 = 19,200;
//        Q4_K routed footprint = 19,200 * 20.25 MiB = 379.7 GiB ≈ 407.6 GB.
//      Box: 4 NUMA nodes * ~126 GB ≈ 504 GB usable; 0.76 → 382.6 GB (fits
//      DeepSeek's 343 GB but UNDER-covers GLM-5.2's 407.6 GB → cold reads on
//      decode). 0.90 → ~454 GB pinned FULLY covers the 407.6 GB Q4_K routed
//      footprint with ~46 GB headroom (tiering cold pool + engine). The
//      Q4_K_XL mixed-precision prepacked arena is ~494 GB stride-padded; a
//      strict 0-skip full-fit of that on a 504 GB-usable box needs f≈0.98
//      (unsafe), so keeper52 targets full-fit of the Q4_K base footprint and
//      the padded tail slab-LRUs via O_DIRECT-on-miss (TD-KEEPER52-ARENA-RAM).
//   3. HiSparse SPARSE ATTENTION — memory.kv_tiering enabled (DSA-guided KV
//      tiering, "HiSparse-style"): cold main-KV pages demote to NUMA-local
//      pinned host RAM and the indexer's selected rows materialize into a
//      dense per-layer scratch before sparse decode attention. Placement-only
//      (INV-KVT-1). index_topk=2048 in the config keeps DSA active.
//   4. SHARDED DCP — hardware.dcp_kv_mode = sharded (KVS sequence-sharded KV;
//      effective at dcp_size>=2, i.e. TP=2). Each rank tiers ITS token shard
//      (TD-KVT-DCP-SHARDED / INV-KVT-9). Indexer stays replicated (tiering's
//      dcp>=2 gate requires replicated indexer).
//   5. TQ — compute.attention_backend = turboquant_mla (TQ_MSE 4-bit MLA,
//      386 B/tok). The tiering gate accepts the TurboQuant-MSE4 cache format
//      as row-self-contained (TD-KVT-TQ resolved, command_dispatcher.cpp).
//
// All four compose per the CURRENT runtime gate (command_dispatcher.cpp
// ~L807-846): the tiering manager is built for BOTH replicated and sharded KV
// and for BOTH SnapMLA-FP8 and TurboQuant-MSE4 cache formats. (The schema
// description at config/schema.json:641 still says "replicated KV, snapmla
// backend" — that text is stale; the gate is the source of truth.)
//
// It skips gracefully (like the goldens' count_big_sm120 guard) when two
// >=28 GB SM120 GPUs, the GGUF, or the prepacked set are absent, so it is safe
// to build into the default `layerstorm_integration_tests` target.
//
// A companion CPU-only test (Keeper52ConfigResolves) builds the exact keeper52
// config JSON, parses it, and runs the pure-CPU config validator — proving the
// config is structurally valid and carries all four features WITHOUT touching
// CUDA / loading weights / running a decode.
//
// ── HOW TO RUN (needs two IDLE RTX 5090s + the GLM-5.2 assets) ───────────────
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
//     LS_LOADER_SHADOW=1 \
//     LS_PERF_TRACE=1 LS_PERF_TRACE_OUT=/tmp/keeper52_xray.csv \
//     ./build/tests/integration/keeper52_test \
//       --gtest_filter='Keeper52Test.HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52'
// ── EP=4 MODE (KEEPER52_EP=4, INV-MOE-EP-XTP) ────────────────────────────────
//   Attention/KV stay TP=2 on the two 5090s; the routed experts split e%4
//   across ALL FOUR GPUs — the 5080s are EXPERT-ONLY ranks (roles=
//   [expert_streaming]: no attention, no pinned weights, no KV → ~their whole
//   VRAM is expert window). Per-GPU decode working set halves to 75x2=150
//   experts, which FITS the stable zone → decode finally caches. Run:
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 KEEPER52_EP=4 \
//     ./build/tests/integration/keeper52_test \
//       --gtest_filter='Keeper52Test.HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52'
//   EP=4-only env: KEEPER52_EXPERT_CACHE_GB_5080 (default 11.0 — 16 GB card
//   minus arena page tables + margin); KEEPER52_STABLE_FRAC defaults to 0.95
//   (the EP=4 working set must fit the STABLE zone). The loader calibration
//   defaults to gpu_loader_calibration_ep4x4.json (4-GPU topology; keeps the
//   EP=2 5090x2 capture intact).
//
// ── FASTEST KNOWN RECIPE (superopt fab2, 2026-07-12 — 5.787 tok/s measured) ──
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
//     KEEPER52_EP=4 KEEPER52_AFFINITY=1 LS_MOE_FFN_GRAPH=1 \
//     KEEPER52_EXPERT_CACHE_GB=5.5 KEEPER52_EXPERT_CACHE_GB_5080=12 \
//     LS_LOADER_SHADOW=0 \
//     ./build/tests/integration/keeper52_test \
//       --gtest_filter='Keeper52Test.HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52'
//   (affinity routing +17%, FFN decode graph +2.7%, window sweeps ~+5%,
//    shadow-solve off +5.9% — see spec/CHANGELOG.md 2026-07-12.)
//
//   Optional env:
//     KEEPER52_EP=<2|4>            EP degree (default 2 — the committed keeper)
//     KEEPER52_AFFINITY=1          cache-affinity expert->GPU routing (replaces
//                                  the static e%tp split; default off)
//     KEEPER52_TOKENS=<n>          decode length override (PROFILING ONLY)
//     LS_KEEPER_NSYS=<a>,<b>       cudaProfilerStart/Stop around tokens a..b
//                                  (use with `nsys profile -c cudaProfilerApi`)
//     KEEPER52_ARENA_FRACTION=<f>  override the 0.90 host-arena fraction_total
//     GLM52_KV_TIERING=<slots>     per-layer hot buffer (default 512 = 2x a
//                                  small window; the config also sets it, this
//                                  env just lets you sweep it)
//     GLM52_KVT_RATIO=<r>          cold-pool multiple of the hot buffer
//                                  (default 16 — the historical goldens' value)
//     LS_KEEPER_SEED_TOKEN=<t>     override the 15234 decode seed token
//     KEEPER52_SEEDS=<a,b,...>     multi-prompt eval IN-PROCESS (one arena
//                                  load): seed 0 cold cache, later seeds warm;
//                                  per-seed stats + median-of-medians.
//                                  BARE-SEED regime only — rejected loud with
//                                  KEEPER52_PROMPT / KEEPER52_FORCE_TRAJ
//     LS_KEEPER_DUMP_ALL=1         print every step (default: 0th + every 10th)
//     LS_LOADER_SHADOW=1           I8: run the loader-solver in shadow mode
//     LS_LOADER_CALIB=<path>       I8: calibration JSON (default
//                                  gpu_loader_calibration_5090x2.json,
//                                  weights-adjacent; recalibrates when absent
//                                  or dims mismatch)
//     LS_LOADER_SHADOW_DUMP=<path> I8: per-layer solver JSONL x-ray dump
//     LS_LOADER_TRAIN_OUT=<path>   I8: fit+bake corrected LoaderConstants after
//                                  the run (needs SHADOW_DUMP + perf_trace CSV)
//     LS_LOADER_PIN_HITS=0         I8: full-domain solve (default: hits pinned
//                                  to their resident device — 91x faster solve)
//     LS_LOADER_REUSE_W=<us>       I8: place_cons reuse reward weight (default
//                                  2000; 0 disables); LS_LOADER_REUSE_TAU=<lv>
//                                  age half-scale in layer-visits (default 300)
//     LS_LOADER_POLICY=<json>      P-26: policy-weights artifact (deploy_fit.py
//                                  output). Precedence: explicit env vars above
//                                  > artifact > built-in defaults; malformed
//                                  artifact fails loud. Same semantics engine-
//                                  side (init_loader_from_env).
//     LS_LOADER_SHADOW_LOG=1       I8: re-enable the per-layer solver-vs-orch
//                                  warn line (default off — hot-path cost)
//
// ── MEASUREMENT REGIMES (spec/reports/DSP52_BOOST.md; TD-PREFILL-NONDET) ─────
//   keeper52 runs in ONE of three prompt regimes (defaults reproduce the
//   canonical bare-seed benchmark BYTE-IDENTICALLY — that is THE protected
//   reportable number):
//   1. BARE-SEED (default, canonical): decode from seed 15234 at position 0,
//      no prompt (~11.5-11.9 tok/s class). B=1 decode is bit-stable across
//      processes, so trajectories AND walls are cross-run comparable.
//   2. PROMPT-FED (KEEPER52_PROMPT=<file> [+ KEEPER52_PROMPT_TOKENS=<cap>]):
//      chunked prefill of a real token-id context before the decode loop
//      (canonical: test-data/prompts/glm52_longctx_tokens3.txt capped 512).
//      Realistic routing/hit profile (~7.7-8.8 tok/s class). Chunked
//      prefill is cross-process DETERMINISTIC since the TD-PREFILL-NONDET
//      root-cause fix (2026-08-02, INV-KVMETA-BT-COV stale-block-table
//      coverage): trajectories bit-converge across processes; walls remain
//      the noisy quantity. NOTE: pre-fix prompt-fed numbers were per-run
//      lottery samples on corrupted mid-prompt ctx KV — do not compare
//      against them.
//   3. TEACHER-FORCED (KEEPER52_FORCE_TRAJ=<file>, normally + KEEPER52_PROMPT):
//      commits come from a pre-recorded continuation; every decode step still
//      executes (walls honest, report notes FORCED). Pins the trajectory —
//      the ONLY cross-run-comparable prompt-conditioned regime (identical
//      routing/hit profile by construction; the DSP52_BOOST saga-conclusion
//      verdict methodology). Fixture recipes: test-data/prompts/README.md.
//
//   Regime env (both default OFF = bare-seed):
//     KEEPER52_PROMPT=<file>       PROMPT-FED: token-id file (one id per
//                                  whitespace token) fed BEFORE the decode
//                                  loop — chunked prefill of the first L-1
//                                  tokens (64-token chunks through the SAME
//                                  fill_moe_entries placement), then the LAST
//                                  prompt token via the loop's first decode
//                                  step (its step produces generated token 0,
//                                  so the loop still measures exactly
//                                  KEEPER52_TOKENS generated tokens; positions
//                                  start at the prompt offset). Prefill is
//                                  EXCLUDED from tok/s + hit stats (walls sum
//                                  decode steps only; the H2D miss counter is
//                                  re-baselined after prefill).
//                                  LS_KEEPER_SEED_TOKEN is ignored (seed =
//                                  last prompt token); multi-seed
//                                  KEEPER52_SEEDS fails loud.
//     KEEPER52_PROMPT_TOKENS=<n>   cap the prompt to its first n ids
//                                  (0/absent = whole file; canonical cap 512)
//     KEEPER52_FORCE_TRAJ=<file>   TEACHER-FORCED: commit the file's tokens
//                                  instead of the sampled ones (steps beyond
//                                  the file's length fall back to sampled).
//                                  Walls honest — every step still runs; the
//                                  model-match rate is reported. Multi-seed
//                                  KEEPER52_SEEDS fails loud. NOT a perf
//                                  regime on its own merits vs bare-seed —
//                                  it exists to pin prompt-conditioned A/Bs.
//
//   The fixture hides non-5090 GPUs from CUDA (nvidia-smi UUID probe → the two
//   largest-VRAM cards) unless CUDA_VISIBLE_DEVICES is already set.
//   NOTE: at LS_PERF_TRACE=1 the 8.4M-event ring can overflow on PollTick spam
//   a few steps in — the run completes and the end-of-run [TD-100j x-ray] log
//   line still aggregates ALL fetches; raise LS_PERF_TRACE=<events> for a full
//   CSV.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "config/config_parser.h"
#include "core/gpu_loader/loader_constants.h"
#include "core/gpu_loader/loader_evict_scores.h"
#include "core/gpu_loader/loader_keeppred.h"
#include "core/gpu_loader/loader_policy.h"
#include "core/gpu_loader/loader_solver.h"
#include "core/gpu_loader/reef_orch.h"
#include "core/memory/expert_cache.h"
#include "core/memory/numa_manager.h"        // INV-REEF-BANK pairing
#include "core/memory/pinned_expert_arena.h"  // INV-REEF-BANK bank seam
#include "core/memory/vram_allocator.h"
#include "core/perf_report.h"
#include "core/perf_trace.h"
#include "daemon/engine.h"
#include "daemon/buffer_registry.h"
#include "daemon/ipc_protocol.h"

namespace lipc = layerstorm::ipc;
namespace ldam = layerstorm::daemon;
namespace fs   = std::filesystem;

namespace {
const char* kConfigRel = "/test-data/config/glm_5_2_gguf.json";
const char* kGgufRel =
    "/test-data/GLM-5.2-GGUF-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf";
const char* kPrepackedRel = "/test-data/GLM-5.2-prepacked";

// keeper52 defaults (the arena bump is the only structural knob change vs the
// original keeper's 0.76; see the fit math in the file header).
constexpr double kDefaultArenaFraction = 0.90;
constexpr int    kDefaultHotBufferSlots = 512;    // 2x a small decode window
constexpr double kDefaultColdRatio = 16.0;        // historical golden value
constexpr int    kNumTokens = 100;

// KEEPER52_EP: expert-parallel degree. 2 (default) = the committed keeper —
// TP=2 / EP=2 on the two 5090s only (5080s hidden from CUDA). 4 = EP-beyond-TP
// (INV-MOE-EP-XTP): attention/KV stay TP=2 on the 5090s; experts split e%4
// across ALL FOUR GPUs — the 5080s are EXPERT-ONLY ranks (no attention, no
// pinned weights, no KV → almost their whole VRAM is expert-cache window).
// EP=4 halves the per-GPU decode working set (75 MoE layers x ceil(8/4) = 150
// experts/GPU vs 300 at EP=2), which is what lets decode finally CACHE.
int keeper52_ep() {
    static const int ep = [] {
        const char* e = std::getenv("KEEPER52_EP");
        if (e && *e && std::atoi(e) == 4) return 4;
        return 2;
    }();
    return ep;
}

// KEEPER52_FORCE_TRAJ=<file> (TEACHER-FORCED regime, TD-PREFILL-NONDET
// pivot — the dsp52 DSP52_FORCE_TRAJ mechanism, e5d71ab3): commit the file's
// pre-recorded continuation (one token id per whitespace token — a prior
// run's committed step tokens) instead of the sampled tokens. Walls stay
// honest (every decode step still executes; only the NEXT-token feed is
// replaced), so two arms forcing the same file see the identical
// routing/hit profile by construction — the only cross-run-comparable
// prompt-conditioned regime (spec/reports/DSP52_BOOST.md saga conclusion).
// An unreadable file FAILS LOUD (a silently-disabled forcing would
// invalidate the A/B it exists for). nullptr = regime off.
const std::vector<uint32_t>* keeper52_force_traj() {
    static const std::vector<uint32_t> traj = [] {
        std::vector<uint32_t> v;
        const char* p = std::getenv("KEEPER52_FORCE_TRAJ");
        if (!p || !*p) return v;
        std::ifstream f(p);
        if (!f.is_open()) {
            ADD_FAILURE() << "KEEPER52_FORCE_TRAJ file not readable: " << p;
            return v;
        }
        long long t;
        while (f >> t)
            if (t >= 0) v.push_back(static_cast<uint32_t>(t));
        std::cerr << "  [keeper52] FORCE_TRAJ: " << v.size()
                  << " forced continuation token(s) from " << p << "\n";
        return v;
    }();
    return traj.empty() ? nullptr : &traj;
}

// EP=4 GPU probe: all visible SM120 devices with their total VRAM (GiB),
// sorted LARGEST-first. Initializes CUDA on the visible set (intended — EP=4
// uses every GPU).
struct ProbedGpu { int ordinal; double gib; };
std::vector<ProbedGpu> probe_sm120_gpus() {
    std::vector<ProbedGpu> v;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return v;
    for (int i = 0; i < count; ++i) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, i)
                != cudaSuccess || major < 12)
            continue;
        size_t freeb = 0, totalb = 0;
        if (cudaSetDevice(i) != cudaSuccess) continue;
        if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) continue;
        v.push_back({i, static_cast<double>(totalb) / (1024.0 * 1024 * 1024)});
    }
    std::sort(v.begin(), v.end(),
              [](const ProbedGpu& a, const ProbedGpu& b) { return a.gib > b.gib; });
    return v;
}

// Build the keeper52 config JSON from the glm_5_2_gguf preset + the four
// feature deltas + the full-fit O_DIRECT overlapped-registration arena.
// `have_gpus` decides whether to bake the two-5090 TP=2 hardware block (the
// GPU run) or leave the preset's single-GPU block (CPU-only structural check).
nlohmann::json build_keeper52_config(const std::string& src, bool have_gpus,
                                     bool have_prepacked) {
    const std::string cfg_path = src + kConfigRel;
    std::ifstream f(cfg_path);
    nlohmann::json j = nlohmann::json::parse(f);

    j["model"]["weights_path"] = src + kGgufRel;
    j["quantization"]["gguf_strategy"] = "int";

    // ── Hardware block: TP=2/EP=2 on the two 5090s (keeper shape, default) or
    // TP=2/EP=4 across all four GPUs (KEEPER52_EP=4, INV-MOE-EP-XTP) ─────────
    if (have_gpus) {
        // TD-MOE-EXPERT-WINDOW: reserve a real per-GPU expert-cache window
        // (vram_allocation_gb.expert_streaming = TOTAL expert cache, GB) so the
        // decode fetch hit-rate + any BIG-prefill get a fat window instead of the
        // 211 MiB auto-minimum. KEEPER52_EXPERT_CACHE_GB overrides (0 = legacy).
        double expert_cache_gb = 4.5;
        if (const char* eg = std::getenv("KEEPER52_EXPERT_CACHE_GB"); eg && *eg)
            expert_cache_gb = std::atof(eg);
        // KEEPER52_STABLE_FRAC: put (nearly) the whole window in the STABLE zone
        // — decode (FETCH_AND_RUN_MOE) reserves kStable, so only the stable
        // fraction is usable for the per-token cache; the streaming zone is idle
        // in this benchmark. <0 leaves the config default (0.5). EP=4 defaults
        // to 0.95: the whole point of EP=4 is a per-GPU working set (150
        // experts ≈ 4.05 GiB) that FITS the stable zone — at the 0.5 default
        // the 4.5 GB 5090 window would thrash.
        double stable_frac = (keeper52_ep() == 4) ? 0.95 : -1.0;
        if (const char* sf = std::getenv("KEEPER52_STABLE_FRAC"); sf && *sf)
            stable_frac = std::atof(sf);
        if (keeper52_ep() == 4) {
            // EP=4: hardware.gpus ORDERED 5090s-first (the engine takes the
            // first tensor_parallelism entries as the DCP/TP group). ids are
            // the live CUDA ordinals from the probe; positions 0,1 = 5090s
            // (attention + resident + expert window), positions 2,3 = 5080s
            // (expert-ONLY — no attention/pinned/KV; roles=[expert_streaming]).
            // 5080 window: the card is 16 GB but must also absorb the pinned-
            // arena page tables (~3 MiB/GiB of pinned host arena ≈ 1.4 GB) +
            // context, all inside the declared 15 with the 3.0 safety margin —
            // 11 GB ≈ 390 stable slots ≫ the 150-expert working set.
            // KEEPER52_EXPERT_CACHE_GB_5080 overrides.
            double expert_cache_gb_5080 = 11.0;
            if (const char* eg = std::getenv("KEEPER52_EXPERT_CACHE_GB_5080");
                eg && *eg)
                expert_cache_gb_5080 = std::atof(eg);
            auto gpus = probe_sm120_gpus();  // largest-first
            if (gpus.size() < 4) {
                fprintf(stderr, "[keeper52] EP=4 needs 4 SM120 GPUs, found %zu\n",
                        gpus.size());
                return j;  // caller's skip guard already vetoed this
            }
            auto all_roles = nlohmann::json::array(
                {"attention", "resident", "expert_streaming"});
            auto expert_only = nlohmann::json::array({"expert_streaming"});
            nlohmann::json arr = nlohmann::json::array();
            for (int p = 0; p < 4; ++p) {
                const bool big = p < 2;  // largest-first probe → 0,1 = 5090s
                nlohmann::json g = {
                    {"id", gpus[static_cast<size_t>(p)].ordinal},
                    {"type", big ? "rtx5090" : "rtx5080"},
                    {"vram_gb", big ? 30 : 15},
                    {"pcie_gen", 5}, {"pcie_width", 16},
                    {"roles", big ? all_roles : expert_only}};
                const double win = big ? expert_cache_gb : expert_cache_gb_5080;
                if (win > 0.0) {
                    g["vram_allocation_gb"]["expert_streaming"] = win;
                    if (stable_frac >= 0.0)
                        g["vram_allocation_gb"]["stable_zone_fraction"] =
                            stable_frac;
                }
                arr.push_back(g);
            }
            j["hardware"]["gpus"] = arr;
        } else {
            j["hardware"]["gpus"] = nlohmann::json::array(
                {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 30},
                  {"pcie_gen", 5}, {"pcie_width", 16}},
                 {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 30},
                  {"pcie_gen", 5}, {"pcie_width", 16}}});
            if (expert_cache_gb > 0.0)
                for (auto& g : j["hardware"]["gpus"]) {
                    g["vram_allocation_gb"]["expert_streaming"] = expert_cache_gb;
                    if (stable_frac >= 0.0)
                        g["vram_allocation_gb"]["stable_zone_fraction"] = stable_frac;
                }
        }
        j["hardware"]["tp_array"] = nlohmann::json::array({0, 1});
        j["parallelism"]["tensor_parallelism"] = 2;
    }

    // ── Feature 4: sharded DCP (KVS). Effective at dcp_size>=2 (TP=2). The
    // indexer stays replicated (tiering's dcp>=2 gate requires it). ────────
    j["hardware"]["dcp_kv_mode"] = "sharded";
    j["hardware"]["dcp_indexer_mode"] = "replicated";
    // dcp_chunk_size default (16) is already a page_size_tokens (16) multiple,
    // which the tiering shard_ok gate requires; set it explicitly for clarity.
    j["memory"]["kv_cache"]["dcp_chunk_size"] = 16;

    // ── Feature 5: TurboQuant 4-bit MLA attention backend ────────────────
    j["compute"]["attention_backend"] = "turboquant_mla";

    // ── Feature 3: HiSparse DSA-guided KV tiering ────────────────────────
    // A hot buffer smaller than the 100-token decode window forces real
    // demotion + cold materialization on the tiered decode steps. Env
    // GLM52_KV_TIERING / GLM52_KVT_RATIO override (parity with the goldens).
    int hot_slots = kDefaultHotBufferSlots;
    if (const char* kt = std::getenv("GLM52_KV_TIERING"); kt && *kt)
        if (int v = std::atoi(kt); v > 0) hot_slots = v;
    double cold_ratio = kDefaultColdRatio;
    if (const char* kr = std::getenv("GLM52_KVT_RATIO"); kr && *kr)
        if (double v = std::atof(kr); v >= 1.0) cold_ratio = v;
    j["memory"]["kv_tiering"] = {
        {"enabled", true},
        {"hot_buffer_slots", hot_slots},
        {"host_to_device_ratio", cold_ratio}};

    // GLM-5.2's larger pinned MLA-256 attention needs more free-VRAM headroom
    // than the DeepSeek keeper's 2.25 GB (matches the glm52 goldens' 3.0). TQ
    // shrinks the KV cache (386 vs 644 B/tok) so this is comfortable.
    j["memory"]["vram_safety_margin_gb"] = 3.0;
    // 100-token decode grows KV by ~1 page/step; a modest page budget is ample.
    j["memory"]["kv_cache"]["max_pages_per_gpu"] = 4096;
    j["memory"]["kv_cache"]["page_growth_chunk_tokens"] = 16;

    // ── Feature 2 + the inherited full-fit O_DIRECT overlapped-register
    // arena. Same knobs as RoutedExpertLruFullFitDirectTest::patch_config,
    // with the fraction bumped 0.76 → 0.90 for GLM-5.2's larger routed set. ─
    double arena_frac = kDefaultArenaFraction;
    if (const char* af = std::getenv("KEEPER52_ARENA_FRACTION"); af && *af)
        if (double v = std::atof(af); v > 0.0 && v <= 1.0) arena_frac = v;
    if (have_prepacked) {
        j["preprocessing"]["prepacked_dir"] = src + kPrepackedRel;
        j["memory"]["preload_expert_buffers"] = true;
        j["memory"]["pin_host_expert_pool"] = true;
        j["memory"]["pin_host_expert_pool_direct_load"] = true;
        j["memory"]["arena_attach"] = {
            // keeper52 defaults arena attach ON (user decision 2026-08-01):
            // this test IS the box's recurring canonical workflow, so the
            // persistent holder should hold ITS geometry — first run after
            // a foreign-geometry store kWipes + cold-rebuilds once, then
            // every keeper/dsp52 run warm-adopts (~75 s init).  CAVEAT:
            // geometry-affecting knob overrides (KEEPER52_ARENA_FRACTION /
            // EP degree / tiering sizes) wipe the champion store per sweep
            // arm — set GLM52_ARENA_ATTACH=0 for such sweeps to keep the
            // canonical store warm (private-arena runs need the holder's
            // RAM free, so don't mix them while a full-fit store is held).
            {"enabled", [] {
                const char* aa = std::getenv("GLM52_ARENA_ATTACH");
                return !(aa && *aa == '0');
            }()}};
        // pin_host_expert_pool_preload=true triggers engine.cpp's
        // defer_registration → register_all() overlapped with the io_uring
        // preload (TD-INIT-OVERLAP), inherited unchanged from the keeper.
        j["memory"]["pin_host_expert_pool_preload"] = true;
        j["memory"]["pin_host_expert_pool_direct_o_direct"] = true;
        j["memory"]["pin_host_expert_pool_sizing"] = {
            {"mode", "fraction_total"}, {"value", arena_frac}};
        // Arenas on the GPU-less NUMA nodes (0,1) too — 0 skipped. A bare bool
        // parses to an empty node list → no spill arenas → a third of the set
        // stays cold (see BENCHMARK_FULLFIT). Must be an OBJECT with explicit
        // nodes.
        // Cross-node RAM spill on nodes 0,1 (fraction_total) PLUS the CPU-less HBM
        // nodes 4-7 as extra warm capacity (TD-NUMA-HBM-BANKS). The HBM nodes are
        // small (16 GB) and partially occupied, so they use per-node fraction_free
        // (% of AVAILABLE, safe margin) — % of total would over-commit and re-trigger
        // the MPOL_BIND OOM that made us exclude them. Proven host-pinnable (mbind +
        // cudaHostRegister succeed, 39 ms/GB). RAM 0.90 + HBM spill → full routed-fit.
        j["memory"]["cross_node_spill"] = {
            {"enabled", true},
            {"nodes", nlohmann::json::array({
                {{"node", 0}, {"weight", 1}},
                {{"node", 1}, {"weight", 1}},
                {{"node", 4}, {"weight", 1}},
                {{"node", 5}, {"weight", 1}},
                {{"node", 6}, {"weight", 1}},
                {{"node", 7}, {"weight", 1}},
            })},
            {"sizing_mode", "fraction_total"},
            {"sizing_value", arena_frac},
            {"per_node", nlohmann::json::array({
                {{"node", 4}, {"mode", "fraction_free"}, {"value", 0.80}},
                {{"node", 5}, {"mode", "fraction_free"}, {"value", 0.80}},
                {{"node", 6}, {"mode", "fraction_free"}, {"value", 0.80}},
                {{"node", 7}, {"mode", "fraction_free"}, {"value", 0.80}},
            })},
        };
    }
    return j;
}
}  // namespace

// ── GPU probe (skip guard) ───────────────────────────────────────────────────

// Counts visible SM120+ devices with >= min_gb total VRAM (TP=2 needs two).
static int count_big_sm120(double min_gb) {
    int count = 0, big = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    for (int i = 0; i < count; ++i) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, i)
                != cudaSuccess || major < 12)
            continue;
        size_t freeb = 0, totalb = 0;
        if (cudaSetDevice(i) != cudaSuccess) continue;
        if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) continue;
        if (static_cast<double>(totalb) / (1024.0 * 1024 * 1024) >= min_gb)
            ++big;
    }
    return big;
}

// ── Per-GPU test-side LRU over routed experts (keeper machinery, 13c-2.0) ────

struct LruKey {
    uint32_t layer;
    uint16_t expert;
    bool operator==(const LruKey& o) const {
        return layer == o.layer && expert == o.expert;
    }
};
struct LruKeyHash {
    size_t operator()(const LruKey& k) const {
        return (static_cast<size_t>(k.layer) << 16) ^ k.expert;
    }
};
// Global recency tick shared by all per-GPU LRUs — lets the affinity router
// (KEEPER52_AFFINITY) compare victim ages ACROSS GPUs so miss placement
// approximates one pooled global LRU over all stable slots instead of four
// partitioned ones.
static uint64_t g_lru_tick = 0;

struct GpuLru {
    int capacity = 0;
    std::deque<LruKey> order;                          // front = MRU, back = LRU
    std::unordered_set<LruKey, LruKeyHash> resident;   // (layer,expert) in VRAM
    std::unordered_map<LruKey, uint64_t, LruKeyHash> last_tick;  // global recency
    void touch(const LruKey& k) {
        for (auto it = order.begin(); it != order.end(); ++it)
            if (*it == k) { order.erase(it); break; }
        order.push_front(k);
        last_tick[k] = ++g_lru_tick;
    }
};

// KEEPER52_AFFINITY=1: orchestrator-side cache-affinity expert→GPU routing.
// Replaces the static e%tp split: a routed expert runs on the GPU where it is
// ALREADY resident (per the 13c-2.0 LRU model — no fetch at all), and a miss
// is placed to balance the per-layer fetch load across the PCIe links
// (argmin misses-assigned-this-layer), tie-broken by the globally OLDEST
// evictable victim so the four per-GPU LRUs behave like one pooled LRU
// (bigger windows naturally absorb more novel experts). Engine-side this is
// legal by construction: FETCH_AND_RUN takes per-expert gpu_idx, residency
// bitsets are per-GPU, and INV-MOE-EP-DISJOINT dedup + the zero-resident
// rank path already handle arbitrary placement (the ACT reroute exercises
// the same seam). Default OFF — the committed e%tp baselines stay intact.
static bool keeper52_affinity() {
    static const bool on = [] {
        const char* e = std::getenv("KEEPER52_AFFINITY");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}

// ── NUMA-aware miss placement (§12h). The engine dumps each expert's pinned
// host-slot NUMA node (LS_ARENA_MAP_DUMP; HBM banks pre-paired to their CPU
// node) once preload is done; this router places each MISS on a GPU local to
// that node — measured H2D is ~56 GB/s local vs ~29 blended in-engine
// (bench/h2d numa matrix + handoff §12h) — and hits are unaffected (they run
// where resident). The map file appears after the first MoE command, so early
// lookups fall back to pure load/age placement (warmup tokens only).
// KEEPER52_NUMA_ROUTE=0 disables. Load (miss_cnt) + victim age still break
// ties INSIDE the candidate set, so per-layer balance is preserved.
struct ArenaNodeMap {
    bool loaded = false;
    std::array<int, 8> gpu_node{};                 // gpu position -> NUMA node
    std::unordered_map<uint32_t, uint8_t> node;    // (layer<<16|e) -> paired node
    bool ensure(const char* path) {
        if (loaded || !path) return loaded;
        std::ifstream f(path);
        if (!f.is_open()) return false;            // dump not written yet
        gpu_node.fill(-1);
        std::string line;
        while (std::getline(f, line)) {
            int a = 0, b = 0, c = 0;
            if (std::sscanf(line.c_str(), "g,%d,%d", &a, &b) == 2) {
                if (a >= 0 && a < 8) gpu_node[static_cast<size_t>(a)] = b;
            } else if (std::sscanf(line.c_str(), "e,%d,%d,%d", &a, &b, &c) == 3) {
                node[(static_cast<uint32_t>(a) << 16)
                     | static_cast<uint32_t>(b)] = static_cast<uint8_t>(c);
            }
        }
        loaded = !node.empty();
        return loaded;
    }
};
static bool keeper52_numa_route() {
    // DEFAULT OFF (§12j): every placement-side locality policy LOST e2e —
    // hard-local 9.69 tok/s @ hit 0.756, rate-slack 9.89 @ 0.760, even
    // tiebreak-only 10.22 @ 0.722 vs baseline 11.41 @ 0.791. DMA does drop
    // (719 → 572-660 us/xfer) but placement bias breaks the pooled-LRU
    // (the oldest-victim tiebreak IS the cache policy); 1pp hit_rate ≈
    // 3.6 ms/tok >> the locality saving. Fetch locality must move the DATA
    // (P-22a host-slot migration), not the consumer.
    static const bool on = [] {
        const char* e = std::getenv("KEEPER52_NUMA_ROUTE");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}
// Candidate policy: 1 = rate-slack constraint (GPUs within KEEPER52_NUMA_SLACK
// GB/s of the best link for the expert's host node; multi-GPU sets preserve
// cache balance), 2 = pure tiebreak (load first, then link rate, then age —
// zero placement distortion). Hard node-local (the first attempt) collapsed
// hit_rate 0.791→0.756: single-GPU candidate sets fought cache capacity.
static int keeper52_numa_mode() {
    static const int m = [] {
        const char* e = std::getenv("KEEPER52_NUMA_MODE");
        return (e && *e) ? std::atoi(e) : 1;
    }();
    return m;
}
static double keeper52_numa_slack() {
    static const double v = [] {
        const char* e = std::getenv("KEEPER52_NUMA_SLACK");
        return (e && *e) ? std::atof(e) : 8.0;
    }();
    return v;
}
// rate[gpu][node] GB/s, measured 2026-07-13 (bench/h2d 'numa', 24.77 MB).
// HBM banks are pre-paired to CPU nodes in the engine dump, so only nodes
// 0-3 appear here. Re-measure after any hardware change.
static constexpr double kH2dRate[4][4] = {
    {42.6, 34.3, 56.2, 51.7},   // GPU0 (5090, node2)
    {34.8, 42.2, 49.2, 56.2},   // GPU1 (5090, node3)
    {56.3, 48.4, 44.2, 35.8},   // GPU2 (5080, node0)
    {43.2, 34.6, 56.3, 50.4},   // GPU3 (5080, node2)
};
// P-25 decision-3 scaffold (KEEPER52_REEF_ORCH=1): the TEST is the REEF
// orchestrator. All placement + eviction decisions are made HERE with the same
// pure components the engine's I8-ACT path uses — the real LoaderSolver over
// the real calibration JSON and the real EvictScoreBoard (freq-bonus recency,
// pooled cheapest victims, reuse place reward) — and shipped as explicit
// per-expert gpu_idx + the 13c-2.0 victim map. The ENGINE loader stays OFF
// (SetUp forces LS_LOADER_SHADOW=0): hits arrive was_cached at dispatch parse
// and nothing reroutes — the zero-reroute dispatch shape worth ~0.3-0.4 tok/s
// over daemon-side acting (DETHRONE_AFFINITY_IDEAS.md UPDATE 6). This is the
// A-compatible scaffold for orchestrator-side loop closure (2026-07-18 user
// decision): decision state lives with the decision maker; engine untouched.
static bool keeper52_reef_orch() {
    static const bool on = [] {
        const char* e = std::getenv("KEEPER52_REEF_ORCH");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}
// Resolve the map path once, wire the engine's dump env BEFORE engine boot
// (same process), and clear any stale map from a previous run.
static const char* keeper52_arena_map_path() {
    static const std::string p = [] {
        const char* e = std::getenv("KEEPER52_ARENA_MAP");
        std::string path = (e && *e) ? e : "/tmp/keeper52_arena_map.csv";
        // INV-REEF-BANK: the REEF arm no longer arms the CSV export — its
        // bank seam is the shared arena snapshot (install_arena_bank_seam).
        if (keeper52_numa_route()) {
            ::unlink(path.c_str());
            ::setenv("LS_ARENA_MAP_DUMP", path.c_str(), /*overwrite=*/0);
        }
        return path;
    }();
    return p.c_str();
}
static const char* g_keeper52_arena_map_init = keeper52_arena_map_path();
static ArenaNodeMap g_arena_map;

// ── KEEPER52_REEF_ORCH decision state (see keeper52_reef_orch() note) ────────
// The board is driven exactly like dispatch_loader drives the engine's:
// advance_recency once per layer, touch_existing_all(key, w·f) for every
// routed expert BEFORE the solve (i-order), inserts at the current clock when
// a miss is admitted (the engine's on_resident_added semantics — update() with
// score=recency_now), evict() for chosen victims. Residency is self-consistent
// model state (the same contract the affinity arm's GpuLru model holds).
// ── REEF decision stack: EXTRACTED to the library (2026-08-18) ───────────────
// ReefOrch + reef_orch_route + reef_orch_apply now live in
// src/core/gpu_loader/reef_orch.{h,cpp} (byte-identical extraction; shared
// with dsp52_test and the daemon's E_CMD_REEF_ROUTE service). Test-only bank
// seam (arena-map CSV) is installed in make_reef_orch below.
using ReefOrch = layerstorm::gpu_loader::ReefOrch;
using layerstorm::gpu_loader::reef_orch_route;

static_assert(sizeof(layerstorm::gpu_loader::ReefEntry) ==
                  sizeof(lipc::ExpertPrefetchEntry) &&
              offsetof(layerstorm::gpu_loader::ReefEntry, expert_idx) ==
                  offsetof(lipc::ExpertPrefetchEntry, expert_idx) &&
              offsetof(layerstorm::gpu_loader::ReefEntry, gpu_idx) ==
                  offsetof(lipc::ExpertPrefetchEntry, gpu_idx));
static_assert(sizeof(layerstorm::gpu_loader::ReefVictim) ==
                  sizeof(lipc::ExpertEvictionEntry) &&
              offsetof(layerstorm::gpu_loader::ReefVictim, expert_idx) ==
                  offsetof(lipc::ExpertEvictionEntry, expert_idx) &&
              offsetof(layerstorm::gpu_loader::ReefVictim, gpu_idx) ==
                  offsetof(lipc::ExpertEvictionEntry, gpu_idx));

static void reef_orch_apply(ReefOrch& o, int layer,
                            const lipc::ExpertPrefetchEntry* entries,
                            lipc::ExpertEvictionEntry* evicts,
                            uint32_t count) {
    layerstorm::gpu_loader::reef_orch_apply(
        o, layer,
        reinterpret_cast<const layerstorm::gpu_loader::ReefEntry*>(entries),
        reinterpret_cast<layerstorm::gpu_loader::ReefVictim*>(evicts), count);
}

// ── Decode result + timing (mirrors first_token_test.cpp) ────────────────────

struct StepTimings {
    double embedding_ms = 0;
    std::vector<double> attention_ms;
    std::vector<double> moe_ms;
    double output_head_ms = 0;
    double sample_ms = 0;
    double total_ms = 0;
};

struct DecodeResult {
    uint32_t sampled_token = 0;
    float top1_prob = 0.f;
    float entropy = 0.f;
    StepTimings timings;
    uint32_t moe_lookups = 0;  // routed experts examined this token (Σ top-K per MoE layer)
};

// ── Fixture ──────────────────────────────────────────────────────────────────

class Keeper52Test : public ::testing::Test {
protected:
    // Hide non-5090 GPUs from CUDA entirely (same rationale as the original
    // keeper): device probing + cudaHostRegister(Portable) would otherwise
    // create contexts on the 5080s and map the whole 0.90-fraction arena into
    // them (4 contexts instead of 2 → slower registration). Reads UUIDs via an
    // nvidia-smi subprocess (does NOT initialize this process's CUDA) and keeps
    // the two largest-VRAM GPUs. A user-set CUDA_VISIBLE_DEVICES is respected.
    // Runs before the FIRST CUDA call — keeper52 is one-per-process (documented).
    void hide_small_gpus() {
        if (std::getenv("CUDA_VISIBLE_DEVICES")) return;
        FILE* p = popen(
            "nvidia-smi --query-gpu=uuid,memory.total "
            "--format=csv,noheader,nounits 2>/dev/null", "r");
        std::vector<std::pair<size_t, std::string>> gpus;  // {MiB, uuid}
        if (p) {
            char line[256];
            while (fgets(line, sizeof(line), p)) {
                std::string s(line);
                auto comma = s.find(',');
                if (comma == std::string::npos) continue;
                std::string uuid = s.substr(0, comma);
                size_t mib = std::strtoull(s.c_str() + comma + 1, nullptr, 10);
                gpus.emplace_back(mib, uuid);
            }
            pclose(p);
        }
        if (gpus.size() > 2) {
            std::sort(gpus.rbegin(), gpus.rend());
            setenv("CUDA_VISIBLE_DEVICES",
                   (gpus[0].second + "," + gpus[1].second).c_str(), 1);
        }
    }

    // Bring up the GLM-5.2 engine with the keeper52 config. Returns via the
    // caller's HasFailure on setup errors.
    void start_engine() {
        const std::string src = LAYERSTORM_SOURCE_DIR;
        const bool have_prepacked =
            fs::exists(src + kPrepackedRel + "/manifest.json");
        if (!have_prepacked)
            fprintf(stderr,
                    "[keeper52] WARNING: prepacked set %s absent — falling back "
                    "to the GGUF-mmap expert path (NOT full-fit; the keeper "
                    "envelope needs the prepacked O_DIRECT arena)\n",
                    (src + kPrepackedRel).c_str());

        nlohmann::json j =
            build_keeper52_config(src, /*have_gpus=*/true, have_prepacked);
        weights_path_ = j["model"]["weights_path"].get<std::string>();
        // P1 shadow (I8): when LS_LOADER_SHADOW is set, enable the GPU-loader
        // calibration so the FETCH handler runs the (behavior-neutral) solver
        // and logs j[·] vs the orchestrator's e%tp. calibration_path defaults
        // to the 2×5090 capture (weights-adjacent — for GLM-5.2 that is the
        // GGUF dir; absent or dims-mismatched → the engine recalibrates and
        // saves it there); LS_LOADER_CALIB overrides. Identical wiring to
        // FirstTokenTest::SetUp (first_token_test.cpp).
        // LS_LOADER_SHADOW defaults ON for keeper52 — the I8 loader/solver IS part
        // of this benchmark. setenv (no-overwrite) so BOTH the config below AND the
        // daemon-side dispatcher (dispatch_loader.cpp reads the env directly) see it.
        // LS_LOADER_ACT=1 then makes the solver's balanced j[·] actually place miss
        // experts (the "I8 NEW" variant) instead of the orchestrator's e%tp.
        // KEEPER52_REEF_ORCH: the TEST is the REEF orchestrator — the engine
        // loader must stay fully OFF (no shadow solve, no act, no board) or
        // both sides would decide. Forced (overwrite) BEFORE the default-on.
        if (keeper52_reef_orch()) setenv("LS_LOADER_SHADOW", "0", 1);
        if (!std::getenv("LS_LOADER_SHADOW")) setenv("LS_LOADER_SHADOW", "1", 0);
        // Canonical placement-invariant EP combine DEFAULTS ON for the keeper
        // (user decision 2026-07-18): every arm shares ONE routing trajectory,
        // so A/Bs measure POLICY, not the FP-drift trace lottery (the ~±1pp
        // equilibrium noise that dominated gate-off comparisons). Costs ~9%
        // wall vs gate-off — historical gate-off numbers are NOT comparable.
        // Override with LAYERSTORM_DETERMINISTIC_EP_COMBINE=0 for legacy runs.
        setenv("LAYERSTORM_DETERMINISTIC_EP_COMBINE", "1", 0);
        setenv("LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION", "bf16", 0);
        // Small-M GEMM route: TEST-INTERFACE compat. The engine default is ON
        // since 2026-08-17 (TD-CHUNK-SMALLM-DEFAULT resolved) with the inverse
        // flag LS_NO_CHUNK_SMALLM=1; historical keeper/dsp52 recipes drive this
        // fixture with LS_CHUNK_SMALLM=0/1, so translate the disable form here
        // and keep those recipes meaningful (=1/unset both mean default ON).
        // No-overwrite: an explicit LS_NO_CHUNK_SMALLM always wins.
        if (const char* sm = std::getenv("LS_CHUNK_SMALLM");
            sm && *sm && sm[0] == '0')
            setenv("LS_NO_CHUNK_SMALLM", "1", 0);
        {
            j["gpu_loader"]["enabled"] = true;
            const char* cp = std::getenv("LS_LOADER_CALIB");
            // EP=4 gets its OWN default calibration file: the 2-GPU capture's
            // dims mismatch a 4-GPU topology, and letting the engine
            // recalibrate into the 5090x2 name would clobber the EP=2 keeper's
            // calibration.
            loader_calib_path_ = cp ? cp
                : (keeper52_ep() == 4 ? "gpu_loader_calibration_ep4x4.json"
                                      : "gpu_loader_calibration_5090x2.json");
            j["gpu_loader"]["calibration_path"] = loader_calib_path_;
        }
        config_path_ = "/tmp/keeper52_config.json";
        { std::ofstream o(config_path_); o << j.dump(2); }

        vocab_size_      = j["model"]["vocab_size"].get<int>();
        first_moe_layer_ = j["model"].value("first_k_dense_replace", 3);
        topk_            = j["model"].value("num_experts_per_tok", 8);

        auto backends = ldam::default_backends();
        backends.skip_cuda_graphs = true;
        engine_ = std::make_unique<ldam::Engine>(config_path_,
                                                 std::move(backends));

        auto& info = engine_->info();
        auto* base = reinterpret_cast<uint8_t*>(info.ipc_base);
        cmd_ring_ = std::make_unique<lipc::CommandRing>(
            base + info.cmd_ring_offset);
        cmp_ring_ = std::make_unique<lipc::CompletionRing>(
            base + info.cmp_ring_offset);
        sideband_    = base + info.sideband_offset;
        num_layers_  = info.num_layers;
        num_experts_ = info.num_experts;

        auto* reg = engine_->buffer_registry();
        hidden_buf_id_ = find_buf(reg, "hidden_state.attn.rank0");
        logits_buf_id_ = find_buf(reg, "logits_scratch.pos0");
        ASSERT_NE(hidden_buf_id_, 0u);
        ASSERT_NE(logits_buf_id_, 0u);
    }

    void TearDown() override {
        // shutdown() flushes the perf_trace CSV + closes the shadow dump.
        if (engine_) { engine_->shutdown(); engine_.reset(); }
        maybe_run_loader_trainer();
        if (!config_path_.empty()) std::remove(config_path_.c_str());
    }

    // I8 trainer feedback hook (dev-only, env-gated, GPU-irrelevant): after the
    // run, if LS_LOADER_TRAIN_OUT is set AND the shadow dump
    // (LS_LOADER_SHADOW_DUMP) and perf_trace CSV (LS_PERF_TRACE_OUT, default
    // /tmp/perf_trace.csv) both exist, shell out (std::system) to
    // tools/loader_xray/trainer_apply.py to join->fit->bake a corrected
    // LoaderConstants JSON at $LS_LOADER_TRAIN_OUT. CPU-only Python; no CUDA —
    // INV-GPU-1 unaffected. Logs the command + exit status. Identical to
    // FirstTokenTest::maybe_run_loader_trainer (first_token_test.cpp).
    void maybe_run_loader_trainer() {
        const char* train_out = std::getenv("LS_LOADER_TRAIN_OUT");
        if (!train_out || !*train_out) return;
        const char* dump = std::getenv("LS_LOADER_SHADOW_DUMP");
        if (!dump || !*dump) {
            std::fprintf(stderr,
                "[loader-train] LS_LOADER_TRAIN_OUT set but LS_LOADER_SHADOW_DUMP unset"
                " — skipping (no model dump to fit)\n");
            return;
        }
        const char* trace_env = std::getenv("LS_PERF_TRACE_OUT");
        std::string trace = (trace_env && *trace_env) ? trace_env : "/tmp/perf_trace.csv";
        if (!fs::exists(dump)) {
            std::fprintf(stderr, "[loader-train] dump %s missing — skipping\n", dump);
            return;
        }
        if (!fs::exists(trace)) {
            std::fprintf(stderr, "[loader-train] trace %s missing — skipping\n", trace.c_str());
            return;
        }
        std::string calib = loader_calib_path_.empty()
                              ? std::string("gpu_loader_calibration_5090x2.json")
                              : loader_calib_path_;
        // Resolve like the engine (Step 14c): a bare/relative path is
        // weights-adjacent — and weights_path is the GGUF FILE, so adjacent
        // means its parent directory.
        if (!calib.empty() && calib[0] != '/') {
            fs::path wdir(weights_path_);
            if (!wdir.empty() && !fs::is_directory(wdir)) wdir = wdir.parent_path();
            calib = (wdir / calib).string();
        }
        const std::string xray = std::string(LAYERSTORM_SOURCE_DIR) + "/tools/loader_xray";
        const char* model = std::getenv("LS_LOADER_TRAIN_MODEL");  // current|contention_bank
        std::string cmd = "python3 '" + xray + "/trainer_apply.py'"
                        + " --in-calib '" + calib + "'"
                        + " --model-jsonl '" + std::string(dump) + "'"
                        + " --trace '" + trace + "'"
                        + " --model " + (model && *model ? model : "current")
                        + " --out-calib '" + std::string(train_out) + "'";
        std::fprintf(stderr, "[loader-train] running: %s\n", cmd.c_str());
        const int rc = std::system(cmd.c_str());
        std::fprintf(stderr, "[loader-train] trainer_apply.py exit status = %d (wrote %s)\n",
                     rc, train_out);
    }

    static uint32_t find_buf(const ldam::BufferRegistry* reg,
                             const std::string& prefix) {
        for (const auto& [id, name] : reg->all_named_entries())
            if (name.rfind(prefix, 0) == 0) return id;
        return 0;
    }

    // ── Command helpers ──────────────────────────────────────────────────
    lipc::Command make_cmd(lipc::CmdType type, uint32_t gpu = 0) {
        lipc::Command c{};
        c.cmd_type  = static_cast<uint32_t>(type);
        c.cmd_seq   = cmd_seq_++;
        c.gpu_idx   = gpu;
        c.stream_id = 0;
        return c;
    }
    void send(const lipc::Command& c) {
        ASSERT_TRUE(cmd_ring_->try_write(&c)) << "cmd ring full";
    }
    bool wait(lipc::Completion& out, uint32_t expected, int timeout_s = 300) {
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::seconds(timeout_s);
        while (std::chrono::steady_clock::now() < deadline) {
            if (cmp_ring_->try_read(&out)) {
                if (out.cmp_type == static_cast<uint32_t>(lipc::CMP_ERROR)) {
                    ADD_FAILURE() << "CMP_ERROR: " << out.error.message;
                    return false;
                }
                if (out.cmp_type == static_cast<uint32_t>(lipc::CMP_CHECKPOINT))
                    continue;
                if (out.cmp_type == expected) return true;
            }
            std::this_thread::yield();
        }
        ADD_FAILURE() << "timeout waiting for cmp 0x" << std::hex << expected;
        return false;
    }

    void create_sequence(uint64_t seq_id, uint32_t prompt_len) {
        lipc::Completion cmp{};
        auto c = make_cmd(lipc::CMD_SEQ_CREATE);
        c.seq_create.seq_id     = seq_id;
        c.seq_create.prompt_len = prompt_len;
        c.seq_create.pool       = 0;
        send(c);
        ASSERT_TRUE(wait(cmp, lipc::CMP_SEQ_OP_DONE));
        ASSERT_EQ(cmp.status, 0u);
    }
    void free_sequence(uint64_t seq_id) {
        lipc::Completion cmp{};
        auto c = make_cmd(lipc::CMD_SEQ_FREE);
        c.seq_free.seq_id = seq_id;
        send(c);
        wait(cmp, lipc::CMP_SEQ_OP_DONE);
    }

    // KEEPER52_REEF_ORCH: build the test-side REEF decision stack — the real
    // calibration (resolved weights-adjacent like the engine), the real solver,
    // the real board — sized to the engine cache's stable-zone capacities.
    std::unique_ptr<ReefOrch> make_reef_orch() {
        if (!keeper52_reef_orch()) return nullptr;
        namespace fs = std::filesystem;
        std::string calib = loader_calib_path_.empty()
                              ? std::string("gpu_loader_calibration_ep4x4.json")
                              : loader_calib_path_;
        if (!calib.empty() && calib[0] != '/') {
            fs::path wdir(weights_path_);
            if (!wdir.empty() && !fs::is_directory(wdir)) wdir = wdir.parent_path();
            calib = (wdir / calib).string();
        }
        const int tp = engine_->info().num_gpus;
        std::vector<int> caps(static_cast<size_t>(tp));
        for (int g = 0; g < tp; ++g)
            caps[static_cast<size_t>(g)] = engine_->expert_cache()->total_slots(
                g, layerstorm::memory::CacheZone::kStable);
        // Library construction (policy precedence P-26: explicit env vars >
        // LS_LOADER_POLICY artifact > built-in defaults; keeppred via
        // LS_LOADER_KEEPPRED_W). Safe to share the env names — under
        // REEF_ORCH the engine loader is forced OFF, so only this side
        // reads them. A malformed artifact fails the test loud.
        std::unique_ptr<ReefOrch> o;
        std::string psrc;
        try {
            o = layerstorm::gpu_loader::make_reef_orch(calib, tp,
                                                       std::move(caps), &psrc);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "KEEPER52_REEF_ORCH: " << e.what();
            return nullptr;
        }
        // Bank seam (INV-REEF-BANK): the SHARED epoch-latched paired
        // snapshot builder — identical to the daemon REEF service's seam
        // (the private LS_ARENA_MAP_DUMP CSV read is retired; epoch 1
        // publishes daemon-side at the first routed-MoE fetch, refreshes
        // at online-migrator commit boundaries).
        {
            const auto* nm = engine_->numa_manager();
            layerstorm::gpu_loader::install_arena_bank_seam(
                *o, engine_->pinned_arena(), [nm](int n) {
                    if (n < 0 || !nm || !nm->node_is_hbm(n)) return n;
                    for (const auto& h : nm->hbm_nodes())
                        if (h.node == n) return h.cpu_affinity_node;
                    return n;
                });
        }
        std::cerr << "  [REEF-ORCH] test-side solver+board armed (calib="
                  << calib << ", policy=" << psrc << ") freq_w=" << o->freq_w
                  << " reuse_w=" << o->reuse_w << " reuse_tau=" << o->reuse_tau
                  << " keeppred_w=" << o->keeppred_w
                  << "\n";
        return o;
    }
    std::unique_ptr<ReefOrch> reef_orch_;

    // Per-GPU LRU model sized to each GPU's stable-zone slot count — the
    // 13c-2.0 orchestrator eviction map for the FETCH keeper (make_lrus() in
    // RoutedExpertLruFullFitDirectTest).
    std::vector<GpuLru> make_lrus() {
        const int tp = engine_->info().num_gpus;
        EXPECT_NE(engine_->expert_cache(), nullptr);
        std::vector<GpuLru> lrus(tp);
        for (int g = 0; g < tp; ++g)
            lrus[g].capacity = engine_->expert_cache()->total_slots(
                g, layerstorm::memory::CacheZone::kStable);
        return lrus;
    }

    // ── Shared expert placement + eviction map ───────────────────────────
    // Places the routed experts `topk` (a single decode row's top-K OR a
    // deduped multi-row prefill-chunk UNION — the placement machinery is
    // list-shaped either way) for `layer` and fills the sideband prefetch +
    // eviction entries. Placement: REEF-ORCH > KEEPER52_AFFINITY > static
    // e%tp — the decode_step_fetch_and_run MoE block factored out (dsp52
    // fill_moe_entries parity) so the KEEPER52_PROMPT prefill chunks route
    // through the SAME code. Returns the entry count (== topk.size()).
    uint32_t fill_moe_entries(int layer, const std::vector<uint16_t>& topk,
                              std::vector<GpuLru>* lrus) {
        const int tp = engine_->info().num_gpus;
        std::vector<uint8_t> assign(topk.size());
        if (reef_orch_ && !topk.empty()) {
            // P-25 decision-3 scaffold: test-side REEF placement (real
            // solver + board); victims applied after entries are built.
            // CAPACITY BOUND (prefill unions): <=64 experts take the frozen
            // production solver; (64, 256] route to LoaderSolver256 inside
            // reef_orch_route; beyond kMaxExpertsLarge fail loud.
            if (topk.size() >
                static_cast<size_t>(layerstorm::gpu_loader::kMaxExpertsLarge)) {
                ADD_FAILURE() << "REEF-ORCH: routed union " << topk.size()
                              << " exceeds LoaderSolver256 kMaxExpertsLarge="
                              << layerstorm::gpu_loader::kMaxExpertsLarge
                              << " (layer " << layer << ") — split the solve "
                                 "or run without KEEPER52_REEF_ORCH";
                return 0;
            }
            reef_orch_route(*reef_orch_, layer, topk, assign);
        } else if (keeper52_affinity() && lrus) {
            auto& Ls = *lrus;
            std::vector<uint32_t> miss_pos;          // topk[] index of misses
            std::vector<int> miss_cnt(static_cast<size_t>(tp), 0);
            for (size_t i = 0; i < topk.size(); ++i) {
                const LruKey k{static_cast<uint32_t>(layer), topk[i]};
                int g_res = -1;
                for (int g = 0; g < tp; ++g)
                    if (Ls[g].resident.count(k)) { g_res = g; break; }
                if (g_res >= 0) assign[i] = static_cast<uint8_t>(g_res);
                else            miss_pos.push_back(static_cast<uint32_t>(i));
            }
            // Experts routed THIS layer are un-evictable on their GPU.
            auto needed_now = [&](int g, const LruKey& k) {
                if (k.layer != static_cast<uint32_t>(layer)) return false;
                for (size_t i = 0; i < topk.size(); ++i)
                    if (topk[i] == static_cast<uint16_t>(k.expert)
                        && (i >= assign.size() ? false
                            : assign[i] == static_cast<uint8_t>(g)))
                        return true;
                return false;
            };
            // §12h: NUMA candidate mask per miss — GPUs local to the
            // expert's pinned host slot (paired node from the engine's
            // arena map). All-GPUs when the map is absent/unknown.
            const bool numa_rt = keeper52_numa_route()
                && g_arena_map.ensure(keeper52_arena_map_path());
            for (uint32_t mi : miss_pos) {
                int best = -1;
                uint64_t best_age = ~0ULL;
                uint32_t numa_ok = ~0u;
                int home = -1;
                if (numa_rt) {
                    auto it = g_arena_map.node.find(
                        (static_cast<uint32_t>(layer) << 16)
                        | topk[mi]);
                    if (it != g_arena_map.node.end()
                        && it->second < 4) {
                        home = static_cast<int>(it->second);
                        if (keeper52_numa_mode() == 1 && tp == 4) {
                            double best_r = 0;
                            for (int g = 0; g < tp; ++g)
                                best_r = std::max(best_r,
                                                  kH2dRate[g][home]);
                            uint32_t m = 0;
                            for (int g = 0; g < tp; ++g)
                                if (kH2dRate[g][home] >= best_r
                                        - keeper52_numa_slack())
                                    m |= (1u << g);
                            if (m) numa_ok = m;
                        }
                    }
                }
                for (int pass = 0; pass < 2 && best < 0; ++pass)
                for (int g = 0; g < tp; ++g) {
                    if (pass == 0 && !(numa_ok & (1u << g))) continue;
                    // Oldest evictable victim's global tick on g (the
                    // pooled-LRU signal). A not-full LRU is "age 0"
                    // (free slot — always preferred at equal load).
                    uint64_t age = 0;
                    if (static_cast<int>(Ls[g].resident.size())
                            >= Ls[g].capacity) {
                        age = ~0ULL;  // no evictable victim → last resort
                        for (auto it = Ls[g].order.rbegin();
                             it != Ls[g].order.rend(); ++it) {
                            if (needed_now(g, *it)) continue;
                            auto t = Ls[g].last_tick.find(*it);
                            age = (t == Ls[g].last_tick.end()) ? 0
                                                               : t->second;
                            break;
                        }
                    }
                    const double rate_g = (home >= 0 && tp == 4)
                        ? kH2dRate[g][home] : 0.0;
                    const double rate_b = (home >= 0 && tp == 4
                                           && best >= 0)
                        ? kH2dRate[best][home] : 0.0;
                    if (best < 0
                        || miss_cnt[static_cast<size_t>(g)]
                               < miss_cnt[static_cast<size_t>(best)]
                        || (miss_cnt[static_cast<size_t>(g)]
                                == miss_cnt[static_cast<size_t>(best)]
                            && (rate_g > rate_b + 1e-9
                                || (rate_g >= rate_b - 1e-9
                                    && age < best_age)))) {
                        best = g;
                        best_age = age;
                    }
                }
                assign[mi] = static_cast<uint8_t>(best < 0 ? 0 : best);
                ++miss_cnt[static_cast<size_t>(assign[mi])];
            }
        } else {
            for (size_t i = 0; i < topk.size(); ++i)
                assign[i] = static_cast<uint8_t>(topk[i] % tp);
        }
        auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
            sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
        auto* evicts = reinterpret_cast<lipc::ExpertEvictionEntry*>(
            sideband_ + lipc::IpcLayout::kExpertEvictionOff);
        uint32_t count = 0;
        for (size_t i = 0; i < topk.size(); ++i) {
            const uint16_t e = topk[i];
            entries[count].layer_idx  = static_cast<uint32_t>(layer);
            entries[count].expert_idx = e;
            entries[count].zone       = 0;
            entries[count].gpu_idx    = assign[i];
            // 13c-2.0: default = no victim (router/local fallback picks).
            evicts[count].layer_idx  = static_cast<uint32_t>(layer);
            evicts[count].expert_idx = 0xFFFF;   // sentinel
            evicts[count].gpu_idx    = assign[i];
            evicts[count]._pad       = 0;
            ++count;
        }

        // 13c-2.0 Option A: supply an LRU-chosen victim per overflowing
        // fetch (orchestrator-guided eviction), replacing the daemon's
        // arbitrary unranked choice. Only attaches a victim to a miss
        // when admitting the routed K would overflow the stable zone.
        // REEF-ORCH: the board (freq-protected pooled LRU) chooses the
        // victims instead of the plain GpuLru model — same map format.
        if (reef_orch_) {
            reef_orch_apply(*reef_orch_, layer, entries, evicts, count);
        } else if (lrus) {
            auto& Ls = *lrus;
            for (int g = 0; g < tp; ++g) {
                GpuLru& lru = Ls[g];
                // This GPU's needed experts this layer + their fetch slots.
                std::vector<uint16_t> want;
                std::vector<uint32_t> miss_ei;   // entry idx of misses on g
                for (uint32_t i = 0; i < count; ++i) {
                    if (entries[i].gpu_idx != g) continue;
                    uint16_t e = entries[i].expert_idx;
                    want.push_back(e);
                    LruKey k{static_cast<uint32_t>(layer), e};
                    if (lru.resident.count(k)) lru.touch(k);
                    else miss_ei.push_back(i);
                }
                auto needed_now = [&](const LruKey& k) {
                    if (k.layer != static_cast<uint32_t>(layer)) return false;
                    return std::find(want.begin(), want.end(),
                                     static_cast<uint16_t>(k.expert)) != want.end();
                };
                // Evict global LRU victims while admitting the misses
                // would overflow; pin each victim to a distinct miss entry.
                int need_room = static_cast<int>(miss_ei.size());
                size_t vi = 0;
                while (static_cast<int>(lru.resident.size()) + need_room
                           > lru.capacity && vi < miss_ei.size()) {
                    LruKey victim{}; bool found = false;
                    for (auto it = lru.order.rbegin(); it != lru.order.rend(); ++it)
                        if (!needed_now(*it)) { victim = *it; found = true; break; }
                    if (!found) break;
                    uint32_t ei = miss_ei[vi++];
                    evicts[ei].layer_idx  = victim.layer;
                    evicts[ei].expert_idx = victim.expert;
                    evicts[ei].gpu_idx    = static_cast<uint8_t>(g);
                    lru.resident.erase(victim);
                    lru.last_tick.erase(victim);
                    for (auto it = lru.order.begin(); it != lru.order.end(); ++it)
                        if (*it == victim) { lru.order.erase(it); break; }
                }
                // Admit the misses (they will be fetched + resident).
                for (uint32_t ei : miss_ei) {
                    LruKey k{static_cast<uint32_t>(layer), entries[ei].expert_idx};
                    lru.resident.insert(k);
                    lru.touch(k);
                }
            }
        }
        return count;
    }

    // ── PROMPT-FED regime (KEEPER52_PROMPT): chunked prefill ─────────────
    // One REAL prefill chunk over `n` prompt tokens at positions
    // [pos0, pos0+n) — the dsp52 / glm52 golden prefill_step pattern on
    // keeper52's placement machinery: is_prefill=1 chunk attention with the
    // fused gate, DEDUPED routed union through fill_moe_entries (REEF /
    // affinity / LRU bookkeeping stays consistent with the decode loop),
    // ONE FETCH_AND_RUN_MOE per layer. No output head — prefill feeds KV
    // only.
    void prefill_chunk_fetch_and_run(const uint32_t* toks, uint32_t n,
                                     uint64_t seq_id, uint32_t pos0,
                                     std::vector<GpuLru>* lrus) {
        lipc::Completion cmp{};
        ASSERT_LE(n, lipc::kMaxSidebandTokenIds);
        ASSERT_LE(n, lipc::kMaxBatchDescriptors);
        auto* token_ids = reinterpret_cast<uint32_t*>(
            sideband_ + lipc::IpcLayout::kTokenIdsOff);
        for (uint32_t i = 0; i < n; ++i) token_ids[i] = toks[i];

        auto embed = make_cmd(lipc::CMD_EMBEDDING_LOOKUP);
        embed.embedding_lookup.num_tokens    = n;
        embed.embedding_lookup.output_buf_id = hidden_buf_id_;
        send(embed);
        EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE));
        EXPECT_EQ(cmp.status, 0u);

        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        for (uint32_t b = 0; b < n; ++b) {
            batch[b].seq_id    = seq_id;
            batch[b].token_pos = pos0 + b;
            batch[b]._pad      = 0;
        }

        for (int layer = 0; layer < num_layers_; ++layer) {
            const bool is_moe = layer >= first_moe_layer_;

            auto a = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
            a.run_attention.layer_idx   = static_cast<uint32_t>(layer);
            a.run_attention.num_seqs    = n;
            a.run_attention.is_prefill  = 1;
            a.run_attention.use_graph   = 0;
            a.run_attention.chunk_start = pos0;
            a.run_attention.chunk_len   = n;
            a.run_attention.emit_gating  = is_moe ? 1 : 0;
            a.run_attention.store_gating = is_moe ? 1 : 0;
            send(a);
            EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "prompt prefill attn L" << layer;
            EXPECT_EQ(cmp.status, 0u);
            if (::testing::Test::HasFailure()) return;

            if (is_moe) {
                const auto* hdr = reinterpret_cast<const lipc::RoutingExportHeader*>(
                    sideband_ + lipc::IpcLayout::kRoutingExportOff);
                EXPECT_EQ(hdr->num_tokens, n)
                    << "routing export rows != prefill chunk rows L" << layer;
                EXPECT_EQ(hdr->layer_idx, static_cast<uint32_t>(layer))
                    << "routing-export layer mismatch in prompt prefill";
                const auto* ridx = reinterpret_cast<const int32_t*>(
                    sideband_ + lipc::IpcLayout::kRoutingExportIndicesOff);
                const uint32_t rn = hdr->num_tokens * hdr->topk;

                // DEDUPED union across the chunk's rows, first-occurrence
                // order (the verify-chunk / golden prefill_step shape).
                std::vector<uint8_t> seen(static_cast<size_t>(num_experts_), 0);
                std::vector<uint16_t> topk;
                for (uint32_t k = 0; k < rn; ++k) {
                    if (ridx[k] < 0 || ridx[k] >= num_experts_) continue;
                    if (seen[static_cast<size_t>(ridx[k])]) continue;
                    seen[static_cast<size_t>(ridx[k])] = 1;
                    topk.push_back(static_cast<uint16_t>(ridx[k]));
                }
                EXPECT_GT(topk.size(), 0u)
                    << "no routed experts exported L" << layer;
                if (topk.empty()) return;

                const uint32_t count = fill_moe_entries(layer, topk, lrus);
                if (::testing::Test::HasFailure()) return;

                auto m = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
                m.fetch_and_run_moe.layer_idx    = static_cast<uint32_t>(layer);
                m.fetch_and_run_moe.num_seqs     = n;
                m.fetch_and_run_moe.expert_count = count;
                m.fetch_and_run_moe.have_evict_map =
                    (reef_orch_ || lrus) ? 1 : 0;
                m.fetch_and_run_moe.timeout_us   = 120000000;  // 120 s
                m.fetch_and_run_moe.moe_mode     = 0;
                send(m);
            } else {
                auto m = make_cmd(lipc::D_B_CMD_RUN_MOE);
                m.run_moe.layer_idx = static_cast<uint32_t>(layer);
                m.run_moe.num_seqs  = n;
                m.run_moe.moe_mode  = 0;
                m.run_moe.apply_residual_correction = 0;
                m.run_moe.store_gating_output       = 0;
                m.run_moe.emit_checkpoint           = 0;
                send(m);
            }
            EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "prompt prefill moe L" << layer;
            EXPECT_EQ(cmp.status, 0u);
            if (::testing::Test::HasFailure()) return;
        }
    }

    // Feed the KEEPER52_PROMPT prompt's first L-1 tokens as 64-token prefill
    // chunks on `seq_id` (the caller has created the sequence with
    // prompt_len = L). The LAST prompt token is NOT fed here — the caller
    // feeds it through ONE plain decode step (the loop's first step), so the
    // loop still measures exactly num_tokens generated tokens. The prefill
    // wall prints and is EXCLUDED from every perf stat (the caller
    // re-baselines the H2D miss counter after this returns; walls only sum
    // decode steps).
    void feed_prompt_prefill(uint64_t seq_id,
                             const std::vector<uint32_t>& prompt,
                             std::vector<GpuLru>* lrus) {
        constexpr uint32_t kChunk = 64;
        const uint32_t pre = static_cast<uint32_t>(prompt.size()) - 1;
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t pos = 0; pos < pre && !::testing::Test::HasFailure();
             pos += kChunk) {
            const uint32_t len = std::min(kChunk, pre - pos);
            prefill_chunk_fetch_and_run(prompt.data() + pos, len, seq_id,
                                        pos, lrus);
        }
        const double s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cerr << "  [keeper52] prompt prefill: " << pre << " tokens in "
                  << std::fixed << std::setprecision(1) << s << " s (chunk "
                  << kChunk << "; last prompt token via the first decode "
                     "step; excluded from perf stats)\n"
                  << std::setprecision(3);
    }

    // One decode step via the production routed-MoE path: RUN_ATTENTION with
    // the fused router gate → read the routed top-K from the sideband → ONE
    // E_CMD_FETCH_AND_RUN_MOE per MoE layer spanning BOTH GPUs via per-expert
    // gpu_idx (EP split, e%tp), with the 13c-2.0 LRU eviction map when `lrus`
    // is supplied. Faithful clone of decode_step_fetch_and_run in
    // first_token_test.cpp (minus the TD-FAR-DIVERGE dump hooks); the
    // placement block is factored into fill_moe_entries (shared with the
    // KEEPER52_PROMPT prefill chunks — dsp52 parity).
    DecodeResult decode_step_fetch_and_run(uint32_t input_token, uint64_t seq_id,
                                           uint32_t token_pos,
                                           std::vector<GpuLru>* lrus = nullptr) {
        using clock = std::chrono::steady_clock;
        auto step_start = clock::now();
        lipc::Completion cmp{};
        DecodeResult result{};
        result.timings.attention_ms.resize(num_layers_);
        result.timings.moe_ms.resize(num_layers_);

        // 1. Write token ID to sideband
        auto* token_ids = reinterpret_cast<uint32_t*>(
            sideband_ + lipc::IpcLayout::kTokenIdsOff);
        token_ids[0] = input_token;

        // 2. CMD_EMBEDDING_LOOKUP
        auto t0 = clock::now();
        auto embed_cmd = make_cmd(lipc::CMD_EMBEDDING_LOOKUP);
        embed_cmd.embedding_lookup.num_tokens    = 1;
        embed_cmd.embedding_lookup.output_buf_id = hidden_buf_id_;
        send(embed_cmd);
        EXPECT_TRUE(wait(cmp, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)));
        EXPECT_EQ(cmp.status, 0u);
        result.timings.embedding_ms =
            std::chrono::duration<double, std::milli>(clock::now() - t0).count();

        // 3. Write batch descriptor
        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        batch[0].seq_id    = seq_id;
        batch[0].token_pos = token_pos;
        batch[0]._pad      = 0;

        // 4. Per-layer attention + MoE
        for (int layer = 0; layer < num_layers_; ++layer) {
            const bool is_moe = layer >= first_moe_layer_;

            auto ta = clock::now();
            auto attn_cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
            attn_cmd.run_attention.layer_idx = static_cast<uint32_t>(layer);
            attn_cmd.run_attention.num_seqs  = 1;
            attn_cmd.run_attention.is_prefill = 0;
            attn_cmd.run_attention.use_graph  = 0;
            // F-1/F-3: fuse the router gate into attention and export the
            // routed top-K to the sideband, so FETCH_AND_RUN can fetch ONLY
            // the routed experts (not all N) — the production-realistic seam.
            attn_cmd.run_attention.emit_gating  = is_moe ? 1 : 0;
            attn_cmd.run_attention.store_gating = is_moe ? 1 : 0;
            send(attn_cmd);
            EXPECT_TRUE(wait(cmp,
                static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)));
            EXPECT_EQ(cmp.status, 0u);
            result.timings.attention_ms[layer] =
                std::chrono::duration<double, std::milli>(clock::now() - ta).count();
            if (::testing::Test::HasFailure()) return result;

            auto tm = clock::now();

            if (is_moe) {
                // Read the routed top-K exported by attention's fused gate.
                const auto* hdr = reinterpret_cast<const lipc::RoutingExportHeader*>(
                    sideband_ + lipc::IpcLayout::kRoutingExportOff);
                EXPECT_EQ(hdr->num_tokens, 1u);
                EXPECT_EQ(hdr->layer_idx, static_cast<uint32_t>(layer))
                    << "routing-export layer mismatch in FETCH_AND_RUN";
                const auto* ridx = reinterpret_cast<const int32_t*>(
                    sideband_ + lipc::IpcLayout::kRoutingExportIndicesOff);
                const uint32_t rn = hdr->num_tokens * hdr->topk;

                std::vector<uint16_t> topk;
                for (uint32_t k = 0; k < rn; ++k)
                    if (ridx[k] >= 0) topk.push_back(static_cast<uint16_t>(ridx[k]));

                // Send ONLY the routed experts to FETCH_AND_RUN. Placement +
                // 13c-2.0/REEF victim map: fill_moe_entries (REEF-ORCH >
                // KEEPER52_AFFINITY > static e%tp — the factored block; see
                // the method comment).
                const uint32_t count = fill_moe_entries(layer, topk, lrus);
                if (::testing::Test::HasFailure()) return result;

                result.moe_lookups += count;  // x-ray: routed experts examined

                auto moe_cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
                moe_cmd.fetch_and_run_moe.layer_idx    = static_cast<uint32_t>(layer);
                moe_cmd.fetch_and_run_moe.num_seqs     = 1;
                moe_cmd.fetch_and_run_moe.expert_count = count;
                moe_cmd.fetch_and_run_moe.have_evict_map =
                    (reef_orch_ || lrus) ? 1 : 0;
                // Correctness gate: wait long enough for the routed experts to
                // load even from a cold O_DIRECT read. The hot arena needs only
                // ~ms; this generous bound just guarantees the routed K are
                // resident before compute so FETCH is numerically equivalent
                // to RUN_MOE.
                moe_cmd.fetch_and_run_moe.timeout_us   = 5000000;  // 5s
                moe_cmd.fetch_and_run_moe.moe_mode     = 0;
                // F-6 decider params left at their make_cmd() zero-init
                // defaults (weight_count=0, min_experts=0, max_new_fetches=0,
                //  gating_weight_threshold=0) → fetch every (routed) entry.
                send(moe_cmd);
            } else {
                // Dense layers: use legacy D_B_CMD_RUN_MOE.
                auto moe_cmd = make_cmd(lipc::D_B_CMD_RUN_MOE);
                moe_cmd.run_moe.layer_idx = static_cast<uint32_t>(layer);
                moe_cmd.run_moe.num_seqs  = 1;
                moe_cmd.run_moe.moe_mode  = 0;
                moe_cmd.run_moe.apply_residual_correction = 0;
                moe_cmd.run_moe.store_gating_output       = 0;
                moe_cmd.run_moe.emit_checkpoint           = 0;
                send(moe_cmd);
            }

            EXPECT_TRUE(wait(cmp,
                static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)));
            EXPECT_EQ(cmp.status, 0u);
            result.timings.moe_ms[layer] =
                std::chrono::duration<double, std::milli>(clock::now() - tm).count();
            if (::testing::Test::HasFailure()) return result;
        }

        // 5. CMD_OUTPUT_HEAD
        auto to = clock::now();
        auto head_cmd = make_cmd(lipc::CMD_OUTPUT_HEAD);
        head_cmd.output_head.num_tokens         = 1;
        head_cmd.output_head.input_buf_id       = hidden_buf_id_;
        head_cmd.output_head.output_buf_id      = logits_buf_id_;
        head_cmd.output_head.compute_confidence = 1;
        send(head_cmd);
        EXPECT_TRUE(wait(cmp, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)));
        EXPECT_EQ(cmp.status, 0u);
        result.top1_prob = cmp.compute.top1_prob;
        result.entropy   = cmp.compute.entropy;
        result.timings.output_head_ms =
            std::chrono::duration<double, std::milli>(clock::now() - to).count();

        // 6. CMD_SAMPLE_TOKENS
        auto ts = clock::now();
        auto sample_cmd = make_cmd(lipc::CMD_SAMPLE_TOKENS);
        sample_cmd.sample_tokens.num_tokens    = 1;
        sample_cmd.sample_tokens.logits_buf_id = logits_buf_id_;
        sample_cmd.sample_tokens.vocab_size    = static_cast<uint32_t>(vocab_size_);
        sample_cmd.sample_tokens.temperature   = 0.0f;  // argmax
        sample_cmd.sample_tokens.top_p         = 1.0f;
        sample_cmd.sample_tokens.top_k         = 0;
        sample_cmd.sample_tokens.random_seed   = 42;
        send(sample_cmd);
        EXPECT_TRUE(wait(cmp, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)));
        EXPECT_EQ(cmp.status, 0u);
        result.timings.sample_ms =
            std::chrono::duration<double, std::milli>(clock::now() - ts).count();

        // 7. Read sampled token (D2H complete by CUDA FIFO ordering).
        result.sampled_token = token_ids[0];

        result.timings.total_ms =
            std::chrono::duration<double, std::milli>(clock::now() - step_start).count();
        return result;
    }

    // Append the per-TP-GPU VRAM breakdown grid (regions in MiB + live
    // used/free + expert cache slots) to an x-ray Report, then emit it (table
    // to stderr + JSON to LS_XRAY_JSON or /tmp/xray_keeper52.json). Clone of
    // RoutedExpertLruTest::finish_xray_report.
    void finish_xray_report(layerstorm::perf_report::Report& rep) {
        namespace pr = layerstorm::perf_report;
        if (const auto* va = engine_->vram_allocator()) {
            const auto* ec = engine_->expert_cache();
            const int ng = va->gpu_count();
            auto MiB = [](int64_t b) {
                std::ostringstream s;
                s << std::fixed << std::setprecision(0) << (b / 1048576.0);
                return s.str();
            };
            std::vector<std::string> hdr = {"region"};
            for (int g = 0; g < ng; ++g)
                hdr.push_back("GPU" + std::to_string(va->region(g).gpu.id));
            pr::Grid& vg = rep.grid("VRAM per TP GPU (MiB)", hdr);
            auto add_region = [&](const char* name, auto field) {
                std::vector<std::string> row = {name};
                for (int g = 0; g < ng; ++g) row.push_back(MiB(field(va->layout().gpus[g])));
                vg.add(row);
            };
            add_region("pinned weights",  [](const auto& L) { return L.pinned_bytes; });
            add_region("KV main",         [](const auto& L) { return L.kv_main_bytes; });
            add_region("KV speculation",  [](const auto& L) { return L.kv_speculation_bytes; });
            add_region("indexer K",       [](const auto& L) { return L.indexer_k_bytes; });
            add_region("expert stable",   [](const auto& L) { return L.expert_stable_bytes; });
            add_region("expert streaming",[](const auto& L) { return L.expert_streaming_bytes; });
            add_region("prefill scratch", [](const auto& L) { return L.prefill_scratch_preallocated_bytes; });
            add_region("safety margin",   [](const auto& L) { return L.safety_margin_bytes; });
            add_region("allocated",       [](const auto& L) { return L.total_vram_bytes - L.safety_margin_bytes; });
            {  // live used/free (cudaMemGetInfo per device)
                std::vector<std::string> used = {"live used"}, freer = {"live free"};
                int cur = 0; cudaGetDevice(&cur);
                for (int g = 0; g < ng; ++g) {
                    size_t fr = 0, tt = 0;
                    cudaSetDevice(va->region(g).gpu.id);
                    if (cudaMemGetInfo(&fr, &tt) == cudaSuccess) {
                        used.push_back(MiB(static_cast<int64_t>(tt - fr)));
                        freer.push_back(MiB(static_cast<int64_t>(fr)));
                    } else { used.push_back("?"); freer.push_back("?"); }
                }
                cudaSetDevice(cur);
                vg.add(used); vg.add(freer);
            }
            if (ec) {  // expert cache slots (used/total) per zone + slot size
                std::vector<std::string> st = {"expert slots stable (used/total)"};
                std::vector<std::string> sr = {"expert slots streaming (used/total)"};
                std::vector<std::string> sz = {"expert slot size MiB"};
                for (int g = 0; g < ng; ++g) {
                    using Z = layerstorm::memory::CacheZone;
                    st.push_back(std::to_string(ec->used_slots(g, Z::kStable)) +
                                 "/" + std::to_string(ec->total_slots(g, Z::kStable)));
                    sr.push_back(std::to_string(ec->used_slots(g, Z::kStreaming)) +
                                 "/" + std::to_string(ec->total_slots(g, Z::kStreaming)));
                    sz.push_back(MiB(ec->expert_bytes()));
                }
                vg.add(st); vg.add(sr); vg.add(sz);
            }
        }
        const char* xj = std::getenv("LS_XRAY_JSON");
        pr::emit(rep, xj ? xj : "/tmp/xray_keeper52.json");
    }

    std::unique_ptr<ldam::Engine> engine_;
    std::unique_ptr<lipc::CommandRing> cmd_ring_;
    std::unique_ptr<lipc::CompletionRing> cmp_ring_;
    uint8_t* sideband_ = nullptr;
    std::string config_path_;
    std::string weights_path_;
    std::string loader_calib_path_;  // I8: gpu_loader.calibration_path when shadow on
    uint32_t cmd_seq_ = 1;
    int num_layers_ = 0, num_experts_ = 0, first_moe_layer_ = 3, topk_ = 8;
    int vocab_size_ = 0;
    uint32_t hidden_buf_id_ = 0, logits_buf_id_ = 0;
};

// ═══════════════════════════════════════════════════════════════════════════
// THE keeper52 benchmark — GPU run (skips gracefully when assets absent).
// Faithful clone of HundredTokenDecodeFetchAndRun_FullFit_EP2 with the GLM-5.2
// keeper52 config: E_CMD_FETCH_AND_RUN_MOE + the 13c-2.0 LRU eviction map +
// seed 15234 + the unified fetch-path x-ray.
// ═══════════════════════════════════════════════════════════════════════════
TEST_F(Keeper52Test, HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52) {
    const std::string src = LAYERSTORM_SOURCE_DIR;
    if (!fs::exists(src + kConfigRel))
        GTEST_SKIP() << "GLM-5.2 config missing: " << src + kConfigRel;
    if (!fs::exists(src + kGgufRel))
        GTEST_SKIP() << "GLM-5.2 GGUF not present: " << src + kGgufRel;

    if (keeper52_ep() == 4) {
        // EP=4 (INV-MOE-EP-XTP): need ALL FOUR GPUs — 2× >=28 GB (5090 TP
        // ranks) + 2 more >=14 GB (5080 expert-only ranks). No hiding; the
        // probe initializes CUDA on the full visible set.
        auto gpus = probe_sm120_gpus();
        if (gpus.size() < 4 || gpus[1].gib < 28.0 || gpus[3].gib < 14.0)
            GTEST_SKIP() << "KEEPER52_EP=4 needs 4 SM120 GPUs "
                            "(2x >=28 GB + 2x >=14 GB); found " << gpus.size();
    } else {
        // Hide the 5080s BEFORE any CUDA call, then require the two 5090s.
        hide_small_gpus();
        if (count_big_sm120(28.0) < 2)
            GTEST_SKIP() << "need TWO SM120+ GPUs with >=28 GB visible (TP=2)";
    }

    start_engine();
    if (::testing::Test::HasFailure()) return;

    // KEEPER52_TOKENS: dev/profiling knob — shorten (or lengthen) the decode.
    // The KEEPER measurement is the 100-token default; a shortened run is for
    // nsys/x-ray iteration only, never a reportable number.
    int num_tokens = kNumTokens;
    if (const char* nt = std::getenv("KEEPER52_TOKENS"); nt && *nt)
        if (int v = std::atoi(nt); v > 0) num_tokens = v;
    // The established keeper seed; env LS_KEEPER_SEED_TOKEN overrides (probe
    // whether a result generalizes across prompts — TD-DECAY-ROOTCAUSE).
    uint32_t seed_token = 15234;
    if (const char* st = std::getenv("LS_KEEPER_SEED_TOKEN"); st && *st) {
        long v = std::atol(st);
        if (v >= 0 && v < static_cast<long>(vocab_size_))
            seed_token = static_cast<uint32_t>(v);
    }
    // KEEPER52_SEEDS: comma-separated seed tokens measured IN-PROCESS — a
    // multi-prompt eval pays ONE arena/engine load, not one NVMe reload per
    // prompt. Seed 0 starts with a cold expert cache (comparable with
    // historical single-seed numbers); later seeds inherit whatever the
    // previous trajectory left cached (warm start — per-seed cache stats use
    // H2D counter deltas). Unset → single established seed, output unchanged.
    std::vector<uint32_t> seeds{seed_token};
    if (const char* ms = std::getenv("KEEPER52_SEEDS"); ms && *ms) {
        seeds.clear();
        std::stringstream seed_ss(ms);
        std::string item;
        while (std::getline(seed_ss, item, ',')) {
            const long v = std::atol(item.c_str());
            if (v >= 0 && v < static_cast<long>(vocab_size_))
                seeds.push_back(static_cast<uint32_t>(v));
        }
        if (seeds.empty()) seeds.push_back(seed_token);
    }
    // ── PROMPT-FED regime (KEEPER52_PROMPT): real-context token-id prompt ──
    // (see the MEASUREMENT REGIMES header block; canonical =
    // test-data/prompts/glm52_longctx_tokens3.txt capped to 512).
    std::vector<uint32_t> prompt;
    if (const char* pp = std::getenv("KEEPER52_PROMPT"); pp && *pp) {
        std::ifstream tf(pp);
        ASSERT_TRUE(tf.is_open())
            << "KEEPER52_PROMPT token file not readable: " << pp;
        int64_t v = 0;
        while (tf >> v) {
            ASSERT_GE(v, 0) << "KEEPER52_PROMPT id negative";
            ASSERT_LT(v, static_cast<int64_t>(vocab_size_))
                << "KEEPER52_PROMPT id out of vocab";
            prompt.push_back(static_cast<uint32_t>(v));
        }
        if (const char* pc = std::getenv("KEEPER52_PROMPT_TOKENS"); pc && *pc)
            if (size_t cap = static_cast<size_t>(std::atoll(pc));
                cap > 0 && cap < prompt.size())
                prompt.resize(cap);
        ASSERT_GE(prompt.size(), size_t{2})
            << "KEEPER52_PROMPT too short: " << pp;
        std::cerr << "  [keeper52] PROMPT-FED: " << prompt.size()
                  << " prompt tokens from " << pp
                  << " (seed = last prompt token " << prompt.back()
                  << "; LS_KEEPER_SEED_TOKEN ignored)\n";
    }
    const std::vector<uint32_t>* prompt_p = prompt.empty() ? nullptr : &prompt;
    // ── TEACHER-FORCED regime (KEEPER52_FORCE_TRAJ) ──────────────────────
    const std::vector<uint32_t>* forced = keeper52_force_traj();
    if (::testing::Test::HasFailure()) return;  // unreadable forced file
    if (forced)
        for (uint32_t t : *forced)
            ASSERT_LT(t, static_cast<uint32_t>(vocab_size_))
                << "KEEPER52_FORCE_TRAJ id out of vocab";
    // Prompt/forced are SINGLE-trajectory regimes: a multi-seed sweep would
    // silently mix regimes (seed 1+ would re-prefill / replay the same
    // forced stream against a warm cache) — fail loud instead.
    ASSERT_TRUE(!(prompt_p || forced) || seeds.size() == 1)
        << "KEEPER52_SEEDS multi-seed is incompatible with KEEPER52_PROMPT / "
           "KEEPER52_FORCE_TRAJ (single-trajectory regimes) — run one seed";
    // 13c-2.0: per-GPU global (layer,expert) LRU model that supplies the
    // orchestrator eviction map (have_evict_map) so victims are LRU-chosen
    // instead of the daemon's arbitrary unranked pick. Persists across tokens.
    std::vector<GpuLru> lrus = make_lrus();
    // P-25 decision-3 scaffold: arm the test-side REEF orchestrator (real
    // solver + board). When armed, it overrides both placement and the victim
    // map; the GpuLru model above stays untouched/unused.
    reef_orch_ = make_reef_orch();
    {   // x-ray: GPU→NUMA-node map for offline local/cross-NUMA classification.
        const auto& gpus = engine_->config().hardware.gpus;
        std::string m = "  GPU->NUMA node:";
        for (size_t p = 0; p < gpus.size(); ++p)
            m += " gpu" + std::to_string(p) + "=node"
                 + std::to_string(gpus[p].numa_node);
        std::cerr << m << "\n";
    }
    const int tp = engine_->info().num_gpus;
    const int num_moe_layers = num_layers_ - first_moe_layer_;
    const int per_gpu_k = (topk_ + tp - 1) / tp;
    const int worst_ws  = num_moe_layers * per_gpu_k;
    const int cap0 = lrus.empty() ? 0 : lrus[0].capacity;
    // Pooled (all-seed) accumulators + per-seed records (KEEPER52_SEEDS).
    struct SeedResult { uint32_t seed; double med_ms, hit, attn_ms, moe_ms; };
    std::vector<SeedResult> seed_results;
    // KEEPER52_PROMPT: pooled-x-ray H2D fields are re-baselined to the
    // post-prefill snapshot (prefill fetches excluded). Bare-seed: zero
    // base — the established process-cumulative pooled report, unchanged.
    lipc::EngineH2dPathStats pooled_h2d_base{};
    int force_match_total = 0;  // KEEPER52_FORCE_TRAJ: model-match count
    double g_total_ms = 0, g_embed = 0, g_attn = 0, g_moe = 0, g_head = 0,
           g_sample = 0;
    uint64_t g_lookups = 0;
    int g_nan = 0, g_tokens = 0;
    double total_ms = 0, sum_embed = 0, sum_attn = 0, sum_moe = 0,
           sum_head = 0, sum_sample = 0;
    uint64_t sum_lookups = 0;
    int nan_count = 0;
    // Per-token wall times, so the report can show median/min/max and a
    // steady-state (token-0-excluded) mean — a plain mean is skewed by the
    // first token's graph-capture/warmup and by occasional H2D-straggler tokens,
    // so the mean can differ materially from the typical (median) token.
    std::vector<double> per_token_ms, per_token_attn_ms, per_token_moe_ms;
    per_token_ms.reserve(num_tokens);
    per_token_attn_ms.reserve(num_tokens);
    per_token_moe_ms.reserve(num_tokens);
    std::cerr << "\n=== keeper52 HundredTokenDecodeFetchAndRun (GLM-5.2 GGUF, "
                 "EP=" << keeper52_ep() << " split (e%" << tp << " over " << tp
              << " GPUs), full-fit O_DIRECT, E_CMD_FETCH_AND_RUN_MOE, LRU "
                 "eviction map, TQ + sharded-KV + HiSparse tiering) ===\n";
    {   // Per-GPU stable capacities (heterogeneous at EP=4: 5090s ≠ 5080s).
        int min_cap = cap0;
        std::string caps;
        for (size_t g = 0; g < lrus.size(); ++g) {
            caps += (g ? "," : "") + std::to_string(lrus[g].capacity);
            min_cap = std::min(min_cap, lrus[g].capacity);
        }
        std::cerr << std::fixed << std::setprecision(3)
                  << "  layers=" << num_layers_ << " first_moe=" << first_moe_layer_
                  << " experts=" << num_experts_ << " topk=" << topk_
                  << " GPUs=" << tp << "\n"
                  << "  stable slots/GPU=[" << caps << "], per-GPU K=" << per_gpu_k
                  << ", worst-case working set=" << num_moe_layers << "x"
                  << per_gpu_k << "=" << worst_ws << " experts; "
                  << (worst_ws <= min_cap ? "FITS in cache (cross-token hits)"
                                          : "EXCEEDS cache (LRU thrash)")
                  << "\n";
    }
    for (size_t si = 0; si < seeds.size(); ++si) {
        // Per-seed H2D delta base: engine counters are cumulative
        // process-wide; per-seed cache stats subtract this snapshot.
        auto h2d_base = engine_->h2d_path_stats();
        const uint32_t prompt_len =
            prompt_p ? static_cast<uint32_t>(prompt_p->size()) : 0;
        create_sequence(1, prompt_len ? prompt_len : 1);
        uint32_t token = seeds[si];
        uint32_t pos0  = 0;
        if (prompt_len) {
            // PROMPT-FED regime: chunked prefill of the first L-1 prompt
            // tokens; the LAST prompt token becomes the loop's first fed
            // token (its decode step produces generated token 0), so the
            // loop still measures exactly num_tokens generated tokens.
            feed_prompt_prefill(1, *prompt_p, &lrus);
            if (::testing::Test::HasFailure()) { free_sequence(1); return; }
            token = (*prompt_p)[prompt_len - 1];
            pos0  = prompt_len - 1;
            // Re-baseline: prompt-prefill fetches are EXCLUDED from the
            // decode hit/miss + H2D stats (per-seed AND pooled x-ray);
            // walls already only sum decode steps.
            h2d_base = engine_->h2d_path_stats();
            pooled_h2d_base = h2d_base;
        }
        if (seeds.size() > 1)
            std::cerr << "\n  ── seed " << token << " (" << (si + 1) << "/"
                      << seeds.size() << ", " << (si ? "warm" : "cold")
                      << " expert cache) ──\n";
        total_ms = sum_embed = sum_attn = sum_moe = sum_head = sum_sample = 0.0;
        sum_lookups = 0;
        nan_count = 0;
        per_token_ms.clear();
        per_token_attn_ms.clear();
        per_token_moe_ms.clear();
        // LS_KEEPER_NSYS=<a>,<b>: dev/profiling knob — cudaProfilerStart() before
        // token a, cudaProfilerStop() after token b (use with
        // `nsys profile -c cudaProfilerApi`). Steady-state decode only; no effect
        // without an attached profiler.
        int prof_a = -1, prof_b = -1;
        if (const char* pw = std::getenv("LS_KEEPER_NSYS"); pw && *pw)
            if (std::sscanf(pw, "%d,%d", &prof_a, &prof_b) != 2) { prof_a = prof_b = -1; }
        int force_match = 0;  // KEEPER52_FORCE_TRAJ: sampled == forced count
        for (int i = 0; i < num_tokens; ++i) {
            if (i == prof_a) cudaProfilerStart();
            auto r = decode_step_fetch_and_run(
                token, 1, pos0 + static_cast<uint32_t>(i), &lrus);
            if (i == prof_b) cudaProfilerStop();
            if (::testing::Test::HasFailure()) { free_sequence(1); return; }
            ASSERT_LT(r.sampled_token, static_cast<uint32_t>(vocab_size_))
                << "Invalid token at step " << i;
            if (!std::isfinite(r.top1_prob) || !std::isfinite(r.entropy)) ++nan_count;
            total_ms   += r.timings.total_ms;
            per_token_ms.push_back(r.timings.total_ms);
            sum_embed  += r.timings.embedding_ms;
            sum_head   += r.timings.output_head_ms;
            sum_sample += r.timings.sample_ms;
            double tok_attn = 0, tok_moe = 0;
            for (double a : r.timings.attention_ms) tok_attn += a;
            for (double m : r.timings.moe_ms)       tok_moe  += m;
            sum_attn += tok_attn;
            sum_moe  += tok_moe;
            per_token_attn_ms.push_back(tok_attn);   // for median/steady-state
            per_token_moe_ms.push_back(tok_moe);
            sum_lookups += r.moe_lookups;
            // LS_KEEPER_DUMP_ALL=1 prints EVERY token so two runs' full 100-token
            // sequences can be diffed for bit-identity. Default OFF preserves the
            // established sparse output.
            const bool dump_all = std::getenv("LS_KEEPER_DUMP_ALL") != nullptr;
            if (dump_all || (i + 1) % 10 == 0 || i == 0)
                std::cerr << "  step " << std::setw(3) << i << ": token="
                          << std::setw(6) << r.sampled_token << "  prob="
                          << r.top1_prob << "  ms=" << r.timings.total_ms << "\n";
            // KEEPER52_FORCE_TRAJ: commit the forced token (walls honest —
            // the step already ran); model-match tracked for the report.
            // Steps beyond the forced tail fall back to the sampled token.
            uint32_t next = r.sampled_token;
            if (forced && static_cast<size_t>(i) < forced->size()) {
                if (r.sampled_token == (*forced)[static_cast<size_t>(i)])
                    ++force_match;
                next = (*forced)[static_cast<size_t>(i)];
            }
            token = next;
        }
        if (forced) {
            force_match_total += force_match;
            std::cerr << "  [keeper52] FORCED trajectory: model matched "
                      << force_match << "/" << num_tokens
                      << " forced tokens (commits came from the file; walls "
                         "honest)\n";
        }
        free_sequence(1);

        // Cache hit/miss + H2D sub-phase x-ray (481-3). phase_count = expert H2D
        // transfers during decode = MISSES; lookups = routed experts examined.
        auto h2d = engine_->h2d_path_stats();
        h2d.direct -= h2d_base.direct;
        h2d.staged -= h2d_base.staged;
        h2d.memcpy_ns -= h2d_base.memcpy_ns;
        h2d.dma_wait_ns -= h2d_base.dma_wait_ns;
        h2d.phase_count -= h2d_base.phase_count;
        h2d.dma_gpu_ns -= h2d_base.dma_gpu_ns;
        h2d.dma_gpu_count -= h2d_base.dma_gpu_count;
        const uint64_t misses = h2d.phase_count;
        const double hit_rate = sum_lookups
            ? 1.0 - static_cast<double>(misses) / static_cast<double>(sum_lookups)
            : 0.0;
        const double dma_us = misses ? (h2d.dma_wait_ns / 1e3) / misses : 0.0;
        const double cpy_us = misses ? (h2d.memcpy_ns   / 1e3) / misses : 0.0;
        // perf_trace pure-DMA x-ray: GPU memcpy time vs host detection/queue lag.
        const double gpu_us = h2d.dma_gpu_count
            ? (h2d.dma_gpu_ns / 1e3) / h2d.dma_gpu_count : 0.0;
        const double host_us = dma_us - gpu_us;  // detection + pre-start stream backlog
        const double per_tok = total_ms / num_tokens;
        // Median / steady-state / spread of per-token wall (warmup-robust): the plain
        // Avg above is skewed by token 0 (graph capture + first-token warmup) and by
        // occasional H2D-straggler tokens, so it can diverge from the TYPICAL token.
        // The median tok/s is the more stable figure to compare across kernel A/Bs.
        // (median, steady-state-mean-excl-tok0) of a per-token series — warmup-robust.
        auto med_ss = [](const std::vector<double>& v) -> std::pair<double, double> {
            if (v.empty()) return {0.0, 0.0};
            std::vector<double> srt(v);
            std::sort(srt.begin(), srt.end());
            const size_t n = srt.size();
            const double med = (n & 1) ? srt[n / 2]
                                       : 0.5 * (srt[n / 2 - 1] + srt[n / 2]);
            double ss = v[0];
            if (n > 1) {
                double s = 0;
                for (size_t k = 1; k < v.size(); ++k) s += v[k];
                ss = s / (v.size() - 1);
            }
            return {med, ss};
        };
        double med_ms = 0, ss_mean_ms = 0, min_ms = 0, max_ms = 0;
        if (!per_token_ms.empty()) {
            const auto ms_stat = med_ss(per_token_ms);
            med_ms = ms_stat.first;
            ss_mean_ms = ms_stat.second;
            auto mm = std::minmax_element(per_token_ms.begin(), per_token_ms.end());
            min_ms = *mm.first;
            max_ms = *mm.second;
        }
        const auto attn_stat = med_ss(per_token_attn_ms);
        const auto moe_stat  = med_ss(per_token_moe_ms);
        // fetch/evict falls out as the gap (same accounting as the keeper).
        const double gap = (total_ms - sum_embed - sum_attn - sum_moe
                            - sum_head - sum_sample) / num_tokens;
        std::cerr << "\n  Total: " << total_ms << " ms for " << num_tokens
                  << " tokens\n  Avg: " << per_tok << " ms/token ("
                  << (1000.0 / per_tok) << " tok/s)\n"
                  << "  Median: " << med_ms << " ms/token ("
                  << (med_ms > 0 ? 1000.0 / med_ms : 0.0) << " tok/s)  "
                  << "SteadyState(excl tok0): " << ss_mean_ms << " ms/token ("
                  << (ss_mean_ms > 0 ? 1000.0 / ss_mean_ms : 0.0) << " tok/s)\n"
                  << "  per-token ms min/med/max: " << min_ms << " / " << med_ms
                  << " / " << max_ms << "\n"
                  << "  NaN: " << nan_count << " / "
                  << num_tokens
                  << "\n  fetch/evict gap: " << gap << " ms/token  (= total - "
                     "embed/attn/moe/head/sample)\n"
                  << "  attention: " << sum_attn / num_tokens << " ms/token  moe: "
                  << sum_moe / num_tokens << " ms/token\n"
                  << "    (median attn " << attn_stat.first << " / steady "
                  << attn_stat.second << " ms;  median moe " << moe_stat.first
                  << " / steady " << moe_stat.second << " ms)\n";
        std::cerr << "  cache: lookups=" << sum_lookups << " misses=" << misses
                  << " hit_rate=" << std::setprecision(4) << hit_rate
                  << std::setprecision(3) << " (fetched " << misses / num_tokens
                  << "/token)\n"
                  << "  H2D x-ray: direct=" << h2d.direct << " staged=" << h2d.staged
                  << "  DMA+detect " << dma_us << " us/xfer  host-copy " << cpy_us
                  << " us/xfer\n";
        if (h2d.dma_gpu_count)
            std::cerr << "  pure-DMA x-ray: GPU-memcpy " << gpu_us
                      << " us/xfer  host detect+queue " << host_us
                      << " us/xfer  (window " << dma_us << ", GPU "
                      << std::setprecision(1) << 100.0 * gpu_us / dma_us << "%)"
                      << std::setprecision(3) << "\n";

        seed_results.push_back({seeds[si], med_ms, hit_rate,
                                sum_attn / num_tokens, sum_moe / num_tokens});
        g_total_ms += total_ms; g_embed += sum_embed; g_attn += sum_attn;
        g_moe += sum_moe; g_head += sum_head; g_sample += sum_sample;
        g_lookups += sum_lookups; g_nan += nan_count; g_tokens += num_tokens;
    }  // ── seed loop (KEEPER52_SEEDS) ──

    if (seed_results.size() > 1) {
        std::cerr << "\n  ── multi-seed summary (" << seed_results.size()
                  << " seeds, one arena load) ──\n" << std::fixed
                  << std::setprecision(3);
        std::vector<double> meds;
        for (const auto& sr : seed_results) {
            std::cerr << "  seed " << std::setw(6) << sr.seed << ": median "
                      << (sr.med_ms > 0 ? 1000.0 / sr.med_ms : 0.0)
                      << " tok/s  hit " << std::setprecision(4) << sr.hit
                      << std::setprecision(3) << "  attn " << sr.attn_ms
                      << "  moe " << sr.moe_ms << " ms/tok\n";
            meds.push_back(sr.med_ms);
        }
        std::sort(meds.begin(), meds.end());
        const size_t nmed = meds.size();
        const double mom = (nmed & 1)
            ? meds[nmed / 2] : 0.5 * (meds[nmed / 2 - 1] + meds[nmed / 2]);
        std::cerr << "  median-of-medians: " << mom << " ms/token ("
                  << (mom > 0 ? 1000.0 / mom : 0.0) << " tok/s)\n";
    }

    // Pooled view for the X-RAY report below (single-seed runs: identical
    // values to the pre-KEEPER52_SEEDS report).
    num_tokens = g_tokens;
    total_ms = g_total_ms; sum_embed = g_embed; sum_attn = g_attn;
    sum_moe = g_moe; sum_head = g_head; sum_sample = g_sample;
    sum_lookups = g_lookups; nan_count = g_nan;
    // Pooled H2D: minus the KEEPER52_PROMPT post-prefill baseline (zero in
    // bare-seed mode — the established process-cumulative report).
    auto h2d = engine_->h2d_path_stats();
    h2d.direct        -= pooled_h2d_base.direct;
    h2d.staged        -= pooled_h2d_base.staged;
    h2d.memcpy_ns     -= pooled_h2d_base.memcpy_ns;
    h2d.dma_wait_ns   -= pooled_h2d_base.dma_wait_ns;
    h2d.phase_count   -= pooled_h2d_base.phase_count;
    h2d.dma_gpu_ns    -= pooled_h2d_base.dma_gpu_ns;
    h2d.dma_gpu_count -= pooled_h2d_base.dma_gpu_count;
    const uint64_t misses = h2d.phase_count;
    const double hit_rate = sum_lookups
        ? 1.0 - static_cast<double>(misses) / static_cast<double>(sum_lookups)
        : 0.0;
    const double dma_us = misses ? (h2d.dma_wait_ns / 1e3) / misses : 0.0;
    const double gpu_us = h2d.dma_gpu_count
        ? (h2d.dma_gpu_ns / 1e3) / h2d.dma_gpu_count : 0.0;
    const double host_us = dma_us - gpu_us;
    const double per_tok = total_ms / num_tokens;

    {  // ── Unified x-ray report: coarse token breakdown + indented daemon MoE
       // phases (perf_trace) + stats + per-TP-GPU VRAM grid; table + JSON sidecar.
        namespace pr = layerstorm::perf_report;
        const double nt = num_tokens;
        const double total_gap = total_ms - sum_embed - sum_attn - sum_moe
                                 - sum_head - sum_sample;
        auto pctf = [&](double v) { return 100.0 * v / total_ms; };
        pr::Report rep;
        rep.title = "X-RAY";
        rep.row("embedding",  0, sum_embed / nt, pctf(sum_embed), -1.0, 1.0);
        rep.row("attention",  0, sum_attn / nt,  pctf(sum_attn),
                sum_attn / (nt * num_layers_), static_cast<double>(num_layers_));
        rep.row("moe (fused fetch+compute)", 0, sum_moe / nt, pctf(sum_moe),
                sum_moe / (nt * num_layers_), static_cast<double>(num_layers_));
        rep.row("output_head", 0, sum_head / nt,  pctf(sum_head),  -1.0, 1.0);
        rep.row("sample",      0, sum_sample / nt, pctf(sum_sample), -1.0, 1.0);
        rep.row("fetch/evict gap", 0, total_gap / nt, pctf(total_gap));
        const auto agg = layerstorm::perf_trace::aggregate();
        if (agg.available) {
            using SD = layerstorm::perf_trace::StageDelta;
            // Each level's % is OF ITS PARENT, so every indent level sums to 100%.
            auto lrow = [&](const char* n, int indent, const SD& d, double parent_ms) {
                rep.row(n, indent, d.total_ms / nt,
                        parent_ms > 0 ? 100.0 * d.total_ms / parent_ms : -1.0,
                        d.mean_ms(), d.count / nt);
            };
            // L1 parent is the CONSUMER moe wall (sum_moe) — which includes the
            // ASYNC GPU compute the daemon timeline (t0→t3) ends before. So the
            // L1 rows are setup + fetch wait + launch + async-compute, summing
            // to the real MoE wall (100%), not just the daemon total.
            const double moe_ms  = sum_moe;                  // L1 parent (consumer wall)
            const double wait_ms = agg.moe_wait.total_ms;    // L2 parent
            const double fw_ms   = agg.fetch_wall.total_ms;  // L3 parent
            // gap = wait − fetch_wall (last byte → finalize enter).
            SD fgap; fgap.count = agg.moe_wait.count;
            fgap.total_ms = std::max(0.0, wait_ms - fw_ms);
            // GPU compute + completion (async): the consumer waits for this
            // after kMoeFinalizeExit (kExpertFfn event reap). = consumer moe −
            // daemon t0→t3, split into the timed GPU-compute wall (kComputeGpu)
            // + completion handoff.
            SD async_compute; async_compute.count = agg.moe_total.count;
            async_compute.total_ms = std::max(0.0, sum_moe - agg.moe_total.total_ms);
            rep.section("MoE timeline (% of MoE wall — each level → 100%)");
            lrow("setup (enter→fetch_issued)",   1, agg.moe_fetch,   moe_ms);
            lrow("fetch wait (issued→finalize)", 1, agg.moe_wait,    moe_ms);
            lrow("fetch wall (first→last byte)", 2, agg.fetch_wall,  wait_ms);
            lrow("fastest GPU",                  3, agg.fetch_fastest, fw_ms);
            lrow("dead / straggler",             3, agg.fetch_dead,    fw_ms);
            lrow("gap (last byte→finalize)",     2, fgap,            wait_ms);
            lrow("compute launch (finalize,host)",1, agg.moe_compute, moe_ms);
            lrow("compute+completion (async tail)",1, async_compute, moe_ms);
            // NOTE: GPU compute (kComputeGpu) is an OVERLAPPING metric — the
            // GPU runs the layer's kernels CONCURRENTLY with the daemon's
            // (un-graphed) host LAUNCH window for the SAME layer. Reported in
            // stats, NOT the partition.
        }
        auto f = [](double v, int p) {
            std::ostringstream s; s << std::fixed << std::setprecision(p) << v;
            return s.str();
        };
        rep.stat("tok/s",             f(1000.0 / per_tok, 3));
        rep.stat("ms/token",          f(per_tok, 3));
        rep.stat("hit_rate",          f(hit_rate, 4));
        rep.stat("lookups/misses",    std::to_string(sum_lookups) + "/" +
                                      std::to_string(misses));
        rep.stat("fetched/token",     f(static_cast<double>(misses) / nt, 1));
        rep.stat("NaN",               std::to_string(nan_count) + "/" +
                                      std::to_string(num_tokens));
        rep.stat("H2D direct/staged", std::to_string(h2d.direct) + "/" +
                                      std::to_string(h2d.staged));
        rep.stat("DMA+detect us/xfer",       f(dma_us, 1));
        rep.stat("pure GPU-memcpy us/xfer",  f(gpu_us, 1));
        rep.stat("host detect+queue us/xfer", f(host_us, 1));
        if (dma_us > 0.0) rep.stat("GPU % of DMA window", f(100.0 * gpu_us / dma_us, 1));
        const auto agg2 = layerstorm::perf_trace::aggregate();
        if (agg2.available) {
            // Overlapping / derived (NOT a partition): summed across both
            // engines, so it can exceed the fetch wall — kept out of the 100%.
            rep.stat("DMA gpu Σ-both-GPUs ms/tok", f(agg2.dma_gpu.total_ms / nt, 3));
            rep.stat("daemon MoE total ms/tok",    f(agg2.moe_total.total_ms / nt, 3));
            rep.stat("fetch wall ms/tok",          f(agg2.fetch_wall.total_ms / nt, 3));
            rep.stat("fetch dead ms/tok",          f(agg2.fetch_dead.total_ms / nt, 3));
            // Compute overlap diagnostic (kComputeGpu): the real per-GPU GEMM
            // wall, how much of it overlapped the daemon's host LAUNCH window
            // for the same layer, and the serial tail after launch.
            const double async_tail = std::max(0.0, sum_moe - agg2.moe_total.total_ms);
            const double gpu_comp   = agg2.compute_gpu.total_ms;
            rep.stat("MoE GPU compute wall ms/tok",   f(gpu_comp / nt, 3));
            rep.stat("compute overlapped w/ launch ms/tok",
                     f(std::max(0.0, gpu_comp - async_tail) / nt, 3));
            rep.stat("compute serial tail ms/tok", f(async_tail / nt, 3));
            rep.stat("perf_trace events", std::to_string(agg2.total_events) +
                                          (agg2.overflow ? " (OVERFLOW)" : ""));
        }
        // Regime markers (absent in the canonical bare-seed report).
        if (prompt_p)
            rep.stat("prompt", std::to_string(prompt.size()) +
                     " tokens (prefill EXCLUDED from all stats)");
        if (forced)
            rep.stat("FORCED trajectory",
                     std::to_string(forced->size()) + " tokens (model matched "
                     + std::to_string(force_match_total) + "/"
                     + std::to_string(num_tokens) + ")");
        finish_xray_report(rep);
    }
    EXPECT_EQ(nan_count, 0) << "Some tokens produced NaN prob/entropy";
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU-ONLY structural check — NO CUDA, NO weights, NO decode. Builds the exact
// keeper52 config, parses it, runs the pure-CPU validator, and asserts all four
// features are present. This is the "config resolves without throwing" gate the
// task asks for; it never initializes CUDA (single-GPU preset block, no engine).
// ═══════════════════════════════════════════════════════════════════════════
TEST(Keeper52ConfigCpu, Keeper52ConfigResolves) {
    namespace cfgns = layerstorm::config;
    const std::string src = LAYERSTORM_SOURCE_DIR;
    if (!fs::exists(src + kConfigRel))
        GTEST_SKIP() << "GLM-5.2 config missing: " << src + kConfigRel;

    // have_gpus=false → keep the preset's single-GPU hardware block (no CUDA
    // probe). have_prepacked=false → skip the O_DIRECT arena knobs (they need
    // the on-disk set); the four feature deltas are asserted regardless.
    nlohmann::json j =
        build_keeper52_config(src, /*have_gpus=*/false, /*have_prepacked=*/false);

    // The four feature deltas are in the JSON as authored.
    EXPECT_EQ(j["compute"]["attention_backend"], "turboquant_mla");
    EXPECT_EQ(j["hardware"]["dcp_kv_mode"], "sharded");
    EXPECT_EQ(j["memory"]["kv_tiering"]["enabled"], true);
    EXPECT_GT(j["model"]["index_topk"].get<int>(), 0);   // DSA active
    EXPECT_EQ(j["model"]["weights_format"], "gguf");     // GLM-5.2 GGUF model

    const std::string tmp = "/tmp/keeper52_cpu_config.json";
    { std::ofstream o(tmp); o << j.dump(2); }

    // Pure-CPU parse (no CUDA). parse_config throws on any schema/type error.
    cfgns::Config cfg;
    ASSERT_NO_THROW(cfg = cfgns::parse_config(tmp));
    std::remove(tmp.c_str());

    // The parsed Config carries the four features.
    EXPECT_EQ(cfg.compute.attention_backend,
              cfgns::AttentionBackendType::turboquant_mla);
    EXPECT_EQ(cfg.hardware.dcp_kv_mode, cfgns::DcpKvMode::sharded);
    EXPECT_TRUE(cfg.memory.kv_tiering.enabled);
    EXPECT_GT(cfg.model.index_topk, 0);
    EXPECT_GT(cfg.model.kv_lora_rank, 0);  // turboquant_mla requires MLA

    // NOTE: the full config validator (config_validator.cpp) is NOT run here.
    // For a GGUF model it sizes the routed experts, which is per-tensor,
    // data-dependent and REQUIRES parsing the actual GGUF file (see
    // make_gguf_quant / gguf_expert_types_from_model). That is not a pure-CPU,
    // file-free check, so full validation is deferred to engine construction in
    // the GPU test. parse_config above (pure CPU) already proves the keeper52
    // config is structurally well-formed and carries all four features.

    // Sanity: the GLM-5.2 routed-expert Q4_K footprint the arena bump targets.
    // 75 MoE layers * 256 experts * (3 * 2048 * 6144 weights * 0.5625 B) ≈
    // 407.6 GB — the number justifying the 0.76 → 0.90 fraction bump.
    const int moe_layers = cfg.model.num_hidden_layers
                         - cfg.model.first_k_dense_replace;
    EXPECT_EQ(moe_layers, 75);
    const double bytes_per_expert =
        3.0 * cfg.model.moe_intermediate_size * cfg.model.hidden_size * 0.5625;
    const double routed_gb =
        static_cast<double>(moe_layers) * cfg.model.n_routed_experts
        * bytes_per_expert / 1e9;
    fprintf(stderr,
            "[keeper52-cpu] Q4_K routed footprint = %.1f GB (%d MoE layers x "
            "%d experts x %.2f MiB); 0.90 fraction ~= 454 GB pinned FITS\n",
            routed_gb, moe_layers, cfg.model.n_routed_experts,
            bytes_per_expert / (1024.0 * 1024.0));
    EXPECT_GT(routed_gb, 400.0);
    EXPECT_LT(routed_gb, 415.0);
}

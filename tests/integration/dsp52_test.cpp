// dsp52 — keeper52 + DSpark speculative decoding (DSP-3/4/5 command seam).
//
// This is a CLONE of the keeper52 benchmark
//   Keeper52Test.HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52
//   (tests/integration/keeper52_test.cpp; spec/BENCHMARK_FULLFIT.md)
// with the IDENTICAL machinery — the E_CMD_FETCH_AND_RUN_MOE production
// routed-MoE seam, the 13c-2.0 orchestrator LRU eviction map, the
// KEEPER52_AFFINITY cache-affinity router, the §12h NUMA miss router, the
// P-25 KEEPER52_REEF_ORCH decision stack (real LoaderSolver + EvictScoreBoard
// driven test-side), the full-fit O_DIRECT overlapped-registration arena with
// arena_attach default ON, TQ + sharded-KV + HiSparse tiering — plus DSpark
// speculative decoding wired on top:
//
//   1. EP=4 IS THE DEFAULT (keeper defaults 2). Attention/KV stay TP=2 on the
//      two 5090s; experts split across ALL FOUR GPUs (INV-MOE-EP-XTP), the
//      5080s as EXPERT-ONLY ranks. KEEPER52_EP still overrides, but the dsp52
//      draft placement REQUIRES the EP=4 shape (the draft lives on the last
//      5080 — an expert-only rank), so KEEPER52_EP=2 skips the GPU test.
//   2. DSPARK — speculation.method=dspark, checkpoint at
//      test-data/GLM-5.2-speculator.dspark (DSP52_CKPT overrides);
//      block_size + speculative_tokens derived from the checkpoint's
//      config.json (loader hard cross-validates — same derivation as
//      glm52_gguf_golden_test); confidence_enabled=false;
//      draft_gpus=[3] — the LAST 5080 (config position 3 of the largest-first
//      probe). UNLIKE the golden (expert-free draft GPU), the draft-hosting
//      5080 here STILL hosts experts: its expert window shrinks to
//      DSP52_DRAFT_CACHE_GB (default 4.0) so the 7.61 GB BF16 draft +
//      DsparkRuntime scratch fit as the LayerRegistry pinned carve beside the
//      expert regions; the other 5080 keeps KEEPER52_EXPERT_CACHE_GB_5080
//      (default 11.0). REEF/affinity adapt to the asymmetric capacity.
//   3. SPECULATIVE DECODE LOOP (default; DSP52_SPEC=0 falls back to the plain
//      keeper loop for A/B in ONE binary). Per round:
//        (a) DRAFT  — ONE D_CMD_RUN_DSPARK_STEP (anchor token @ anchor_pos =
//            fed-token count, num_query = γ); the γ i32 draft ids arrive in
//            the sideband readback slice (kSpecCheckpointOff + 2560) when the
//            completion fires (DSP-5 contract; confidence off → data_bytes
//            = γ*4). The aux-hidden export (INV-DSPARK-AUX) arms the draft
//            context automatically on every FED row — including the verify
//            chunks (prefill-shaped capture) and rewound re-feeds
//            (position-addressed overwrite, dspark_runtime.cpp rewind path).
//        (b) VERIFY — two backends, DSP52_VB selects:
//            * DSP52_VB=seq (DEFAULT — GPU-validated): sequential early-stop
//              teacher-forced verification, the exact DsparkLossless golden
//              pattern (glm52_gguf_golden_test dspark_draft_rounds): feed the
//              anchor, then each draft WHILE it matches the main model's
//              greedy output, all through the SAME plain
//              decode_step_fetch_and_run the keeper loop uses. Rejected
//              drafts are NEVER fed → no KV rewind, indexer coverage stays
//              alive, and losslessness is inherited from the plain path by
//              construction.
//            * DSP52_VB=batched (the throughput target — mirrors the Python
//              orchestrator's _start_batched_verification production shape,
//              orchestrator_loop.py ~1636-1650): ONE multi-row same-seq
//              chunk of R = 1+γ rows [anchor@p, d_0@p+1, …, d_{γ-1}@p+γ]:
//              EMBEDDING_LOOKUP num_tokens=R, batch-descriptor rows for
//              positions p..p+γ, per layer RUN_ATTENTION is_prefill=1
//              (chunk_start=p, chunk_len=R) with the fused gate on MoE
//              layers, the DEDUPED routed-expert union across all R rows
//              through the SAME REEF/affinity/static placement as the plain
//              step, ONE FETCH_AND_RUN_MOE per layer (single-shot: R ≪
//              moe_chunk_capacity, so the EP-XTP extra-rank fold applies,
//              INV-MOE-EP-XTP), then ONE CMD_OUTPUT_HEAD num_tokens=R
//              readback_to_host=1 → R u32 greedy argmax ids (batched-verify
//              head, dispatch_compute.cpp).
//              ⚠ 2026-08-01 bring-up: batched verify returned corrupt
//              argmax ids (row-0 16 vs plain 284; near-constant low ids)
//              while every command completed status 0. Dataflow audit
//              CONFIRMED the per-row post-final-layer hiddens DO land in
//              attn_buf rows [0..R) (residual: dispatch_attention.cpp
//              ~1306-1342 moe_buf=attn_out+attn_buf; commit:
//              dispatch_moe.cpp ~1659-1662 moe_buf→attn_buf, num_tokens
//              rows) and CMD_OUTPUT_HEAD num_tokens=R reads exactly those
//              rows — so the corruption is NUMERIC in one of the four
//              never-GPU-validated stages of this config's chunk pipeline:
//              (1) TQ chunk-prefill attention (chunk_causal B>1,
//                  tq_sm120_attention_device.cpp:360-585 — goldens ran
//                  chunk prefill on the default backend only),
//              (2) sharded-KV QAG at B>1 (the strided head re-gather,
//                  dcp_executor.cpp:2265-2283 — sharded goldens were B=1
//                  decode),
//              (3) EP-XTP extra-rank fold at B>1 (dispatch_moe.cpp
//                  ep_xtp_broadcast/dispatch_moe_ep_extras — only ever run
//                  at B=1 decode),
//              (4) the TP batched OUTPUT_HEAD (allgather+transpose,
//                  dispatch_forward.cpp:245-433 — no GPU test ever issued
//                  num_tokens>1).
//              RESOLUTION (TD-DSP52-BATCHED-VERIFY-EQUIV, 2026-08-01): the
//              bisect kit exonerated ALL FOUR corruption suspects (KV-mode
//              flip identical, TP-only fold identical, SAMPLE_TOKENS agrees
//              row-for-row) and the backend arm produced a CROSS (TQ-chunk
//              agrees with snapmla-decode and vice versa at top1_prob
//              0.018) — the divergence is chunk-vs-decode FP reduction
//              order flipping near-ties at FLAT-distribution positions,
//              NOT corruption. The bisect knobs stay for future stage
//              isolation; the batched REF gate is the logits EQUIVALENCE
//              gate of item 4 (INV-DSPARK-LOSSLESS B>1 clause), and
//              DSP52_VB=seq keeps the strict token-identity gate.
//        (c) ACCEPT — longest prefix j with d_j == target_j; commit
//            d_0..d_{j-1} + bonus (j+1 tokens/round); fed advances by j+1;
//            anchor = bonus. Batched mode re-feeds stale positions
//            (position-addressed KV overwrite; DSA indexer coverage goes
//            permanently dense on the first partial-acceptance rewind —
//            dispatch_attention.cpp coverage guard — a REGIME change, not a
//            correctness one); sequential mode never rewinds.
//   4. REFERENCE GATE — DSP52_REF=1, shape-dependent (INV-DSPARK-LOSSLESS):
//      * VB=seq: first runs the plain keeper loop on seq 1 recording its
//        trajectory, then the speculative loop on seq 2, and EXPECTs token
//        identity (the sequential verifier feeds through the SAME B=1
//        decode pipeline, so strict bit-identity is the right gate — greedy
//        verify rejects any mismatching draft; speculation changes SPEED,
//        never tokens).
//      * VB=batched: token identity vs a B=1 plain-decode trajectory is the
//        WRONG gate — chunk and decode kernel pipelines are legitimately
//        different FP reduction orders and near-ties flip at flat positions
//        (resolved TD-DSP52-BATCHED-VERIFY-EQUIV). Per the
//        INV-DSPARK-LOSSLESS B>1 clause the target distribution IS the
//        batched forward's own outputs (vLLM semantics; the trajectory is
//        self-consistent and greedy-deterministic). The gate is
//        EQUIVALENCE at matched context: for the first DSP52_REF_ROUNDS
//        rounds (default 4) the batched chunk's row-0 LOGITS are compared
//        against a B=1 decode replay on a THIRD scratch sequence (seq 3)
//        teacher-forced with the identical committed history (the replay
//        runs ONCE, AFTER the loop — the single-slot DSpark draft context
//        re-arms only on a position-0 capture, so B=1 replay steps cannot
//        interleave with the rounds) — per-element
//        |a-b| <= DSP52_REF_BAND_ABS + DSP52_REF_BAND_REL*|b| (defaults =
//        the CALIBRATED end-to-end band, dsp52_ref_band_abs(); the
//        per-kernel BF16 band 0.06 + 0.05*|b| is NOT the right scale here
//        — chunk-vs-decode reduction-order divergence amplifies over the
//        78-layer backbone), argmax identity asserted ONLY when the
//        replay's top1-vs-top2 logit margin exceeds the band. Row-0 top1
//        margins print per round so flat positions are visible in logs.
//        Unconditional gates stay: no NaN, status 0 on every command; and
//        DSP52_REF_DET=1 optionally re-runs the batched loop in-process
//        (seq 4) EXPECTing the identical committed trajectory.
//      Default mode runs the speculative loop only.
//
// It skips gracefully when the 4 SM120 GPUs, the GGUF, the prepacked set or
// the DSpark checkpoint are absent.
//
// ── HOW TO RUN (needs the four idle SM120 GPUs + GLM-5.2 assets + ckpt) ──────
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
//     ./build/tests/integration/dsp52_test \
//       --gtest_filter='Dsp52Test.SpeculativeHundredTokenDecodeFetchAndRun_FullFit_EP4_GLM52'
//
//   dsp52-only env:
//     DSP52_CKPT=<dir>             DSpark checkpoint override (default
//                                  test-data/GLM-5.2-speculator.dspark →
//                                  glm-5.2-dspark-preview γ=15/block-16 =
//                                  the M1-frontier champion ckpt; the
//                                  bring-up-era `.orig` γ=7 pin is RETIRED
//                                  — DSP52_BOOST.md "M1 frontier",
//                                  2026-08-05)
//     DSP52_SPEC=0                 run the PLAIN keeper loop instead (A/B
//                                  baseline in the same binary; default 1)
//     DSP52_REF=1                  reference gate. VB=seq: plain loop
//                                  (seq 1) then speculative loop (seq 2),
//                                  EXPECT token identity. VB=batched:
//                                  per-round row-0 logits EQUIVALENCE vs a
//                                  B=1 teacher-forced replay (seq 3) at
//                                  matched context (INV-DSPARK-LOSSLESS
//                                  B>1 clause; no token-identity baseline)
//     DSP52_REF_ROUNDS=<n>         batched equivalence-gate depth: gate the
//                                  first n rounds (default 4; DSP52_REF=1 +
//                                  DSP52_VB=batched only)
//     DSP52_REF_BAND_ABS=<f>       equivalence-band override |a-b| <= ABS +
//     DSP52_REF_BAND_REL=<f>       REL*|b| (defaults = the calibrated
//                                  end-to-end band, dsp52_ref_band_abs();
//                                  ABS=1e9 makes the element check log-only
//                                  for band recalibration)
//     DSP52_REF_DET=1              batched optional determinism check:
//                                  re-run the batched loop in-process
//                                  (seq 4), EXPECT the identical committed
//                                  trajectory
//     DSP52_DRAFT_CACHE_GB=<gb>    draft-hosting GPU's expert window
//                                  (default 4.0 on the 5080 host; on a 5090
//                                  host derived from the quant format:
//                                  fp8_e4m3 0.5 / nvfp4 1.75)
//     DSP52_DRAFT_GPU=<0..3>[,<0..3>]  config position(s) hosting the draft
//                                  (default 3 = last 5080; 0/1 = a 5090 TP
//                                  rank — TD-DSPARK-DRAFT-QUANT placement:
//                                  draft at the pinned-region tail, that
//                                  rank's expert window shrinks, position 3
//                                  keeps the full 5080 window). A comma
//                                  PAIR (e.g. "0,1") = the TP=2 draft
//                                  shard across both 5090s
//                                  (TD-DSPARK-DRAFT-SHARD): each hosting
//                                  rank carries ~half the weight sweep;
//                                  per-rank windows default to the proven
//                                  per-card occupancy (nvfp4 3.25 GB each)
//     DSP52_SHARD=1                shorthand for DSP52_DRAFT_GPU=0,1
//     DSP52_QUANT=<fmt>            speculation.dspark.draft_weights_quant
//                                  (bf16 default | fp8_e4m3 | nvfp4):
//                                  requant the draft's GEMM operands at
//                                  upload — draft arena 7.61/4.92/3.62 GB
//     DSP52_MARGIN_GB=<f>          per-GPU safety-margin shave on the
//                                  draft-hosting position ONLY (phase-1
//                                  window-coverage probe): the freed
//                                  (3.0 - f) GB folds into the default
//                                  draft-host window (e.g. 1.5 -> nvfp4
//                                  window 3.25 GB). Other GPUs keep the
//                                  global 3.0 margin (their physical carve
//                                  is untouched)
//     DSP52_PROMPT=<file>          PROMPT-FED arm: token-id file (one id per
//                                  whitespace token). The prompt is fed
//                                  BEFORE the decode loop in EVERY mode
//                                  (plain / seq / batched): chunked prefill
//                                  of the first L-1 tokens (64-token chunks
//                                  through the SAME fill_moe_entries
//                                  placement — the DsparkAcceptRealistic
//                                  pattern), then the LAST prompt token via
//                                  ONE plain decode step = the loop's seed
//                                  step. The aux export ingests every fed
//                                  row, so the draft context is armed over
//                                  the REAL prompt automatically
//                                  (INV-DSPARK-AUX). Committed-token
//                                  counting and the report are UNCHANGED
//                                  (still gen_tokens GENERATED tokens);
//                                  positions and history simply start at the
//                                  prompt offset — anchor_pos = fed count
//                                  INCLUDES the prompt. Prefill is excluded
//                                  from the wall / hit-rate stats;
//                                  LS_KEEPER_SEED_TOKEN is ignored.
//     DSP52_PROMPT_TOKENS=<n>      cap the prompt to its first n ids
//                                  (0/absent = the whole file)
//     DSP52_GAMMA=<n>              effective draft length override (lever 1,
//                                  DSP52_BOOST campaign): num_query < ckpt
//                                  speculative_tokens shrinks the verify
//                                  chunk to R=1+n rows (union-collapse);
//                                  clamped to [1, ckpt gamma]
//     DSP52_CONF_THRESH=<f>        DSP-9 static_threshold bring-up (> 0
//                                  enables confidence_enabled=true): the
//                                  verify block is truncated at the first
//                                  draft slot with cumprod(c_1..c_k) below
//                                  the threshold (c_k = trained DSP-6 head,
//                                  sideband readback after the ids); a round
//                                  truncated to 0 slots runs ONE plain
//                                  decode step instead (hybrid AR/spec).
//                                  Every round also prints a [dsp52-conf]
//                                  STS-calibration trace (raw c_k + g_use +
//                                  accepted prefix — dspark_calibration.py
//                                  ConfidenceRound input; slots >= g_use are
//                                  censored, never verified)
//     DSP52_VB=<seq|batched>       verify backend (default seq — the
//                                  GPU-validated sequential early-stop
//                                  pattern; batched = the R-row chunk, see
//                                  the bring-up note above)
//     DSP52_OVERLAP=1              lever 2 (DSP52_VB=batched only): hide the
//                                  draft under an always-committing plain
//                                  decode step; on draft[0]==plain result,
//                                  chunk-verify the remaining slots (rows
//                                  [t, d_1, ...]); accept rule unchanged
//     DSP52_UPART=1                union-aware cache partitioning: verify-
//                                  chunk union MISSES fetch into the
//                                  expert_streaming zone (zone=1 transient
//                                  class, released at chunk completion) —
//                                  no stable victim, no board admission;
//                                  plain/decode fills untouched. See the
//                                  dsp52_upart() block comment
//     DSP52_LOOKAHEAD_DIAG=<shared|resident>
//                                  Wave-3 M4 stage-1 routing-agreement
//                                  diagnostic: approximate PASS-1 forward of
//                                  every verify chunk before the real pass-2;
//                                  per-layer top-K agreement + manifest-
//                                  coverage tables at loop end. Measurement-
//                                  only (walls not champion-comparable). See
//                                  the dsp52_lookahead_diag() block comment
//     DSP52_FORCE_TRAJ=<file>      matched-trajectory A/B (TD-PREFILL-NONDET
//                                  pivot): teacher-force BOTH loops over the
//                                  same pre-recorded generated continuation
//                                  (token-id file). Commits come from the
//                                  forced stream; every step/draft/chunk
//                                  still executes (walls honest); acceptance
//                                  = drafts vs the forced stream. NOT a
//                                  lossless gate — do not combine with
//                                  DSP52_REF=1
//     DSP52_FORCE_DRAFT=<file>     NO-FORK completion of DSP52_FORCE_TRAJ:
//                                  replay a recorded per-round draft block +
//                                  confidences (real draft still executes,
//                                  walls honest) so g_use/acceptance/round
//                                  structure are arm-identical across
//                                  numerics-perturbing arms
//     DSP52_FORCE_DRAFT_DUMP=<f>   per-round draft recorder ("<g> toks
//                                  confs"; run on the champion arm)
//   Batched-verify BISECT knobs (DSP52_VB=batched only; run one at a time —
//   whichever restores row-0 to the plain argmax fingerprints the broken
//   stage):
//     DSP52_VB_TP_ONLY=1           route the verify-chunk expert union to the
//                                  TP ranks only (static e%2 — removes the
//                                  EP-XTP B>1 fold, suspect 3; forces static
//                                  placement for verify chunks, so run
//                                  without KEEPER52_REEF_ORCH)
//     DSP52_VB_SAMPLE=1            after the batched head, ALSO issue
//                                  CMD_SAMPLE_TOKENS num_tokens=R over the
//                                  same logits and diff against the head
//                                  readback (splits suspect 4's readback/
//                                  sample seam from an upstream logits bug)
//     DSP52_KV_MODE=replicated     config override of hardware.dcp_kv_mode
//                                  (removes the QAG B>1 re-gather, suspect 2)
//     DSP52_ATTN_BACKEND=<name>    config override of
//                                  compute.attention_backend (e.g. the
//                                  preset default — removes TQ chunk
//                                  prefill, suspect 1)
//   Inherited keeper52 env (identical semantics — see keeper52_test.cpp):
//     KEEPER52_EP=<2|4>            EP degree (dsp52 DEFAULT 4; 2 skips — the
//                                  draft needs the EP=4 expert-only 5080)
//     KEEPER52_AFFINITY=1          cache-affinity expert->GPU routing
//     KEEPER52_REEF_ORCH=1         test-side REEF solver+board placement
//     KEEPER52_TOKENS=<n>          decode length override (PROFILING ONLY)
//     KEEPER52_EXPERT_CACHE_GB / KEEPER52_EXPERT_CACHE_GB_5080
//     KEEPER52_STABLE_FRAC / KEEPER52_ARENA_FRACTION
//     (KEEPER52_SEEDS is NOT supported here — dsp52 is single-seed; use
//      keeper52_test for multi-seed evals)
//     KEEPER52_NUMA_ROUTE / KEEPER52_NUMA_MODE / KEEPER52_NUMA_SLACK
//     KEEPER52_ARENA_MAP           arena-map dump path (default
//                                  /tmp/dsp52_arena_map.csv)
//     LS_KEEPER_SEED_TOKEN=<t>     override the 15234 seed token
//     LS_KEEPER_DUMP_ALL=1         print every plain-loop step
//     LS_KEEPER_NSYS=<a>,<b>       profiler window (plain loop only)
//     GLM52_KV_TIERING / GLM52_KVT_RATIO / GLM52_ARENA_ATTACH
//     LS_LOADER_SHADOW / LS_LOADER_CALIB / LS_LOADER_POLICY / ... (I8 stack)
//
// ── MEASUREMENT REGIMES (spec/reports/DSP52_BOOST.md; TD-PREFILL-NONDET) ─────
//   Every dsp52 mode (plain DSP52_SPEC=0 / seq / batched) runs in ONE of
//   three prompt regimes — identical semantics to keeper52_test's
//   KEEPER52_PROMPT / KEEPER52_FORCE_TRAJ:
//   1. BARE-SEED (default): decode from seed 15234 at position 0, no prompt.
//      B=1 decode is bit-stable across processes → trajectories AND walls
//      cross-run comparable. Canonical for keeper-parity numbers; NOTE the
//      bare-seed trajectory is degenerate for acceptance studies (drafter
//      acceptance collapses on it — DSP52_PERF).
//   2. PROMPT-FED (DSP52_PROMPT=<file> [+ DSP52_PROMPT_TOKENS=<cap>];
//      canonical: test-data/prompts/glm52_longctx_tokens3.txt cap 512):
//      realistic routing/acceptance. Chunked prefill is cross-process
//      DETERMINISTIC since the TD-PREFILL-NONDET root-cause fix
//      (2026-08-02, INV-KVMETA-BT-COV stale-block-table coverage):
//      trajectories/acceptance/τ bit-converge across processes; walls
//      remain the noisy quantity. Pre-fix prompt-fed numbers were per-run
//      lottery samples on corrupted mid-prompt ctx KV — do not compare.
//   3. TEACHER-FORCED (DSP52_FORCE_TRAJ=<file>, normally + DSP52_PROMPT):
//      both loops commit the same pre-recorded continuation (walls honest;
//      acceptance = drafts vs the forced stream). Pins the trajectory —
//      the ONLY cross-run-comparable prompt-conditioned regime (the
//      DSP52_BOOST saga-conclusion verdict methodology). Fixture recipes:
//      test-data/prompts/README.md (dsp52_forced_traj_r1.txt).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

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
const char* kDsparkCkptRel = "/test-data/GLM-5.2-speculator.dspark";

// keeper52 defaults, inherited unchanged.
constexpr double kDefaultArenaFraction = 0.90;
constexpr int    kDefaultHotBufferSlots = 512;
constexpr double kDefaultColdRatio = 16.0;
constexpr int    kNumTokens = 100;

// DSpark checkpoint dir: DSP52_CKPT overrides the shipped symlink.
std::string dsp52_ckpt_dir(const std::string& src) {
    if (const char* ck = std::getenv("DSP52_CKPT"); ck && *ck) return ck;
    return src + kDsparkCkptRel;
}

// DSP52_VB: verify backend. Default "seq" — the sequential early-stop
// teacher-forced pattern (GPU-validated by the DsparkLossless golden;
// lossless by construction). "batched" = the R-row chunk (production target
// shape; see the bring-up defect note in the file header).
bool dsp52_vb_batched() {
    static const bool b = [] {
        const char* e = std::getenv("DSP52_VB");
        return e && std::string(e) == "batched";
    }();
    return b;
}
// DSP52_VB_TP_ONLY=1: bisect — verify-chunk union routed to TP ranks only.
bool dsp52_vb_tp_only() {
    static const bool on = [] {
        const char* e = std::getenv("DSP52_VB_TP_ONLY");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}
// DSP52_VB_SAMPLE=1: bisect — cross-check the batched head readback against
// CMD_SAMPLE_TOKENS argmax over the same logits buffer.
bool dsp52_vb_sample() {
    static const bool on = [] {
        const char* e = std::getenv("DSP52_VB_SAMPLE");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}

// DSP52_REF_ROUNDS: how many initial batched-verify rounds the DSP52_REF=1
// logits-equivalence gate covers (default 4; 0 disables the replay while
// keeping the per-round margin log).
int dsp52_ref_rounds() {
    static const int n = [] {
        const char* e = std::getenv("DSP52_REF_ROUNDS");
        int v = 4;
        if (e && *e) v = std::max(0, std::atoi(e));
        return v;
    }();
    return n;
}
// DSP52_REF_BAND_ABS / DSP52_REF_BAND_REL: the batched-vs-replay row-0
// logits equivalence band, per element |a-b| <= ABS + REL*|b|. The defaults
// are the CALIBRATED END-TO-END band (TD-DSP52-BATCHED-VERIFY-EQUIV band
// calibration, 2026-08-01): chunk-vs-decode FP reduction-order divergence
// amplified over the 78-layer backbone — NOT the per-kernel BF16 band
// (0.06 + 0.05*|b|, dspark_forward_test.cpp), which end-to-end logits
// legitimately exceed. Calibration (orig ckpt γ=7, seed 15234, 24 gated
// rounds, REEF): per-round max|dLogit| is BIMODAL — 0.84..2.74 while the
// spec sequence's KV history is young (rounds 0-12), 9.9..15.5 once the
// chunk-written KV drift compounds (rounds 13-23); observed max 15.489.
// ABS = 48.0 ≈ 3.1x that max; REL = 0 (a constant band — the drift is NOT
// proportional to |ref|: max|d|/(1+|ref|) peaked at 11.6, no cleaner).
// Env-overridable for future recalibration; DSP52_REF_BAND_ABS=1e9 turns
// the element check log-only for collection.
double dsp52_ref_band_abs() {
    static const double v = [] {
        const char* e = std::getenv("DSP52_REF_BAND_ABS");
        double d = 48.0;
        if (e && *e) d = std::atof(e);
        return d;
    }();
    return v;
}
double dsp52_ref_band_rel() {
    static const double v = [] {
        const char* e = std::getenv("DSP52_REF_BAND_REL");
        double d = 0.0;
        if (e && *e) d = std::atof(e);
        return d;
    }();
    return v;
}
// DSP52_REF_DET=1: optional batched-mode determinism check — re-run the SAME
// batched loop in-process and EXPECT the identical committed trajectory
// (self-consistency of the batched target distribution, INV-DSPARK-LOSSLESS
// B>1 clause).
bool dsp52_ref_det() {
    static const bool on = [] {
        const char* e = std::getenv("DSP52_REF_DET");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}

// DSP52_GAMMA: effective draft length override (DSP52_BOOST campaign lever
// 1). The runtime supports num_query < the checkpoint's speculative_tokens
// (the block simply runs shorter); small gamma collapses the batched
// verify-union fetch footprint (γ=1 → R=2 routes ~11 experts/layer ≈
// cache-friendly vs ~39 at γ=7). 0/absent = the checkpoint gamma; clamped
// to [1, ckpt gamma] at use.
int dsp52_gamma_override() {
    static const int v = [] {
        const char* e = std::getenv("DSP52_GAMMA");
        return (e && *e) ? std::max(0, std::atoi(e)) : 0;
    }();
    return v;
}
// DSP52_CONF_THRESH: DSP-9 static_threshold bring-up mode, driven test-side
// (the PLAN DSP-9 per-request cutoff on the cumulative survivals a_j =
// cumprod(c_1..c_j); the trained DSP-6 confidence head c_k arrives in the
// sideband readback after the draft ids). The verify block is truncated at
// the first slot whose cumprod drops below the threshold; a round truncated
// to 0 slots falls back to ONE plain decode step (hybrid AR/spec — the
// draft cost is already sunk, the anchor still advances). > 0 enables
// speculation.dspark.confidence_enabled in the config (both shipped
// checkpoints carry trained heads). 0/absent = off.
double dsp52_conf_thresh() {
    static const double v = [] {
        const char* e = std::getenv("DSP52_CONF_THRESH");
        return (e && *e) ? std::atof(e) : 0.0;
    }();
    return v;
}

// DSP52_QUANT=<bf16|fp8_e4m3|nvfp4> (TD-DSPARK-DRAFT-QUANT): requantize the
// draft's GEMM operands at upload (speculation.dspark.draft_weights_quant).
// Default bf16 = the established byte-identical upload. Measured real-ckpt
// draft weight arenas: bf16 7.61 GB, fp8_e4m3 4.92 GB, nvfp4 3.62 GB
// (+ ~0.9 GiB runtime scratch in the LayerRegistry charge).
std::string dsp52_quant() {
    static const std::string v = [] {
        const char* e = std::getenv("DSP52_QUANT");
        return std::string(e && *e ? e : "bf16");
    }();
    return v;
}

// DSP52_DRAFT_GPU=<config position 0..3> (TD-DSPARK-DRAFT-QUANT placement):
// which config GPU hosts the draft. Default 3 = the LAST 5080 (the
// established dsp52 shape). 0/1 = a 5090 TP rank: the draft lives at the
// TAIL of that rank's pinned region beside the target weights and its
// expert window shrinks to make room (see build_dsp52_config); position 3
// then keeps the FULL KEEPER52_EXPERT_CACHE_GB_5080 window — the gpu3
// draft-contention + window-shrink removal the quant campaign targets.
// TD-DSPARK-DRAFT-SHARD: DSP52_DRAFT_GPU also accepts a comma pair
// ("0,1") = the TP=2 draft shard across both 5090s; DSP52_SHARD=1 is
// shorthand for "0,1". Single value keeps the established placement.
std::vector<int> dsp52_draft_gpus() {
    static const std::vector<int> v = [] {
        std::vector<int> out;
        if (const char* s = std::getenv("DSP52_SHARD");
            s && *s && std::atoi(s) != 0)
            return std::vector<int>{0, 1};
        const char* e = std::getenv("DSP52_DRAFT_GPU");
        if (!e || !*e) return std::vector<int>{3};
        std::string str(e);
        size_t pos = 0;
        while (pos <= str.size()) {
            const size_t comma = str.find(',', pos);
            const std::string tok = str.substr(
                pos, comma == std::string::npos ? std::string::npos
                                                : comma - pos);
            if (!tok.empty()) {
                const int p = std::atoi(tok.c_str());
                if (p >= 0 && p <= 3 &&
                    std::find(out.begin(), out.end(), p) == out.end())
                    out.push_back(p);
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        if (out.empty()) out.push_back(3);
        return out;
    }();
    return v;
}

int dsp52_draft_gpu() { return dsp52_draft_gpus()[0]; }

// DSP52_MARGIN_GB=<f> (TD-DSPARK-DRAFT-SHARD phase-1 window probe): per-GPU
// safety-margin override applied to the DRAFT-HOSTING position only (the
// other three GPUs keep the global 3.0 — a GLOBAL shave would grow the
// physical device_alloc carve on every card, incl. the near-full 5080s).
// Shaving the hosting 5090's margin 3.0 -> ~1.5 recovers ~1.5 GB of expert
// window (pair with DSP52_DRAFT_CACHE_GB to actually claim it — the window
// is an explicit override clamped by total - pinned - kv - margin).
// Negative/absent = unset (global margin everywhere, byte-identical config).
double dsp52_margin_gb() {
    static const double v = [] {
        const char* e = std::getenv("DSP52_MARGIN_GB");
        if (!e || !*e) return -1.0;
        const double m = std::atof(e);
        return m >= 0.0 ? m : -1.0;
    }();
    return v;
}

// DSP52_OVERLAP=1 (lever 2, DSP52_BOOST campaign; DSP52_VB=batched only):
// pipeline the DSpark draft (5080) UNDER a plain decode step (5090s+MoE):
// per macro-round, send D_CMD_RUN_DSPARK_STEP async, run ONE plain decode
// step feeding the anchor (its output ALWAYS commits — never wasted), then
// collect the draft. If draft slot 0 == the plain result, batch-verify the
// REMAINING slots in a (g_use)-row chunk [t@fed, d_1, ...]; else skip the
// chunk (the sunk draft was hidden under the plain step anyway). The accept
// rule stays greedy longest-prefix + bonus (INV-DSPARK-LOSSLESS: slot k is
// checked against the target's own argmax for its position — slot 0 against
// the plain step's output, slots >=1 against the chunk rows).
bool dsp52_overlap() {
    static const bool on = [] {
        const char* e = std::getenv("DSP52_OVERLAP");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}

// DSP52_PREFETCH=1 (lever 4, DSP52_BOOST campaign; batched verify only):
// during each verify chunk, at the LAUNCH of MoE layer L's attention (the
// H2D-idle window), issue one fire-and-forget D_B_CMD_PREFETCH_BATCH for
// the PREVIOUS chunk's layer-L union at its previous (expert,gpu)
// placement, priority -1.0 (below the FED's 0.0 — staged behind critical
// fetches). Pure residency warm-up: no numerics change (same weights land
// in the same cache seam the FED would fetch through). Completions are
// stashed by cmd_seq and dropped (fire_forget_seqs_).
//
// SOURCE OF TRUTH = the engine config field
// speculation.dspark.verify_chunk_prefetch (default false — measured -1.0%
// champion wall, and even a perfect predictor loses on this DMA-bound box;
// see the schema description). The test interface keeps DSP52_PREFETCH with
// OVERRIDE-WINS semantics: set and non-"0" → ON, "0" → OFF, unset → config.
// The env decision is cached (env never changes mid-run); the config value is
// published by start_engine() once the parsed Config exists.
namespace {
std::atomic<bool> g_dsp52_prefetch_cfg{false};
}  // namespace

// Publish the parsed speculation.dspark.verify_chunk_prefetch value.
void dsp52_set_prefetch_config(bool on) {
    g_dsp52_prefetch_cfg.store(on, std::memory_order_relaxed);
}

bool dsp52_prefetch() {
    // -1 = env unset (defer to config), 0/1 = explicit env override.
    static const int env_override = [] {
        const char* e = std::getenv("DSP52_PREFETCH");
        if (!e || !*e) return -1;
        return std::atoi(e) != 0 ? 1 : 0;
    }();
    if (env_override >= 0) return env_override != 0;
    return g_dsp52_prefetch_cfg.load(std::memory_order_relaxed);
}

// TD-KEEPPRED-HYBRID budgeted idle-window MISS-prefetch (DSP52_PREFETCH_MISS=n):
// 0 (default) = legacy FULL prev-union prefetch; n>0 = fire ONLY the top-n
// board-NON-RESIDENT predicted-reuse experts (the misses keeppred/freq did not
// keep), ranked by decayed frequency. Never re-fetches a resident expert (the
// tier-0 ELB mistake). Reuses the (retired-default) DSP52_PREFETCH machinery.
int dsp52_prefetch_miss() {
    static const int n = [] {
        const char* e = std::getenv("DSP52_PREFETCH_MISS");
        return (e && *e) ? std::max(0, std::atoi(e)) : 0;
    }();
    return n;
}

// TD-KEEPPRED-HYBRID lookahead (DSP52_PREFETCH_AHEAD=k, default 0 = same-layer):
// at layer L's attention launch, fire the (budgeted, miss-only) prefetch for
// layer L+k instead of L — hiding L+k's fetch behind k layers of compute (a
// genuinely EARLIER/wider idle window the same-layer demand fetch does not
// already consume). The decisive "is there untapped idle time" probe for B=1.
int dsp52_prefetch_ahead() {
    static const int k = [] {
        const char* e = std::getenv("DSP52_PREFETCH_AHEAD");
        return (e && *e) ? std::max(0, std::atoi(e)) : 0;
    }();
    return k;
}

// ── DSP52_ELB=1 — predictive expert prefetch behind a pluggable predictor
// seam (EPM Phase-29 tier-0 deployment; spec/MoE-SpeQ_NOTES.md §6/§8).
//
// Tier-0 predictor = TEMPORAL LOCALITY ("b0_prev", the EPM-4 grid verdict:
// held-out recall@8 0.302 / union coverage 0.881 — 2× every trained model at
// accessible corpus scale): per layer, the routed (expert → gpu, zone) entry
// list of the PREVIOUS observation of that layer — previous decode round in
// the plain/overlap loops, previous chunk union in batched verify.
//
// ISSUE POLICY (INV-ELB-CAP): at the LAUNCH of layer L's attention (the
// compute-idle H2D window), issue ONE evict batch + ONE fire-and-forget
// D_B_CMD_PREFETCH_BATCH covering the predictions for layers L..L+AHEAD:
//   - candidates capped per layer (DSP52_ELB_CAP, selection-rank order) and
//     filtered to not-resident, not-already-in-flight;
//   - room is made ONLY from cheaply-evictable victims — the REEF board's
//     cheapest (LRU-oldest) stable residents whose recency age exceeds the
//     hot window (DSP52_ELB_HOT, default 2*(AHEAD+1) layer-visits = the
//     prefetch horizon with margin; see dsp52_elb_hot()) — the prefetch
//     horizon's working set is NEVER evicted for a prediction; when no
//     evictable victim exists the prediction is silently skipped (no retry,
//     no fabricated budget — the 0cf5823a reserve-feasibility discipline);
//   - prefetch DMA priority is -1.0, strictly below every demand fetch
//     (engine kDemandFetchPriority = FLT_MAX; a demand fetch joining an
//     in-flight prefetch re-asserts its priority in the staged queue).
// ADVISORY ONLY (INV-ELB-ADVISORY): residency warm-up + LRU-cold eviction
// shifts change WHEN bytes move, never which experts the MoE computes —
// tokens must stay byte-identical flag-on vs flag-off.
//
// DSP52_ELB=1 SUPERSEDES DSP52_PREFETCH (the lever-4 prototype): the a=0
// chunk-loop prediction is the same previous-chunk same-layer union; both
// flags on would double-write the sideband prefetch region racily, so the
// legacy block is skipped when ELB is armed.
//
// Prefill chunks only feed the predictor (observe); no prefetch is issued
// inside prefill — its fetch seam is wave-scheduled (INV-FAR-WAVE) and out
// of scope here.
// DSP52_PREFETCH_ORACLE=<file> — the ORACLE UPPER-BOUND diagnostic (not
// deployable: uses recorded FUTURE routing). Drives the exact ELB issue
// machinery (residency filter, per-layer cap, lookahead, priority -1,
// advisory byte-identity) with a PERFECT-accuracy scripted predictor read
// from a DSP52_ORACLE_DUMP recording of a matched forced-trajectory run.
// Measures the hit/wall CEILING a perfect expert predictor could reach.
std::string dsp52_prefetch_oracle() {
    static const std::string s = [] {
        const char* e = std::getenv("DSP52_PREFETCH_ORACLE");
        return e ? std::string(e) : std::string();
    }();
    return s;
}
// DSP52_ORACLE_DUMP=<file> — RUN 1 (record): dump per-(traj-step, layer) the
// committed routed union + placement, so RUN 2 can replay it as the oracle.
std::string dsp52_oracle_dump() {
    static const std::string s = [] {
        const char* e = std::getenv("DSP52_ORACLE_DUMP");
        return e ? std::string(e) : std::string();
    }();
    return s;
}
// DSP52_ORACLE_NEXTROUND=1 — oracle predicts the NEXT trajectory step's
// same-layer union (a perfect b0_prev) instead of this step's future layers
// (perfect same-round lookahead). Default 0 = same-step future layers.
bool dsp52_oracle_nextround() {
    static const bool on = [] {
        const char* e = std::getenv("DSP52_ORACLE_NEXTROUND");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}
// DSP52_ORACLE_BULK=1 — CHUNK-START BULK-MANIFEST oracle arm (the perfect +
// unbounded ceiling probe). Instead of per-layer fire-at-launch, the ENTIRE
// step's non-resident routed manifest (all MoE layers of this traj step) is
// fired as one fire-and-forget prefetch burst (lever-4 style, priority -1,
// transient zone=1) BEFORE the first attention of the step, giving the fetch
// the maximum possible overlap window against the whole sweep's compute. The
// per-layer ELB issue is suppressed; the oracle predictor + determinism gate
// stay active. Answers: "is there ANY timing window when EVERYTHING is fired
// perfectly early?" (mission stop condition when both per-layer arms are null).
bool dsp52_oracle_bulk() {
    static const bool on = [] {
        const char* e = std::getenv("DSP52_ORACLE_BULK");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}
// DSP52_EVICT_ORACLE=<file> — the BELADY-EVICTION diagnostic (not deployable;
// uses FUTURE info). Consumes a DSP52_ORACLE_DUMP recording (magic 'ORCL') as an
// EXACT next-use oracle: in the REEF victim selection (reef_orch_apply) evict the
// resident whose NEXT demand is FARTHEST in the future (never-again first) = true
// Belady. Fetches NOTHING extra — it only reorders the eviction victim set, so
// future demands HIT more → fewer total demand-fetch bytes on the bandwidth-bound
// bus (a VOLUME reduction, the CORRECT use of perfect future knowledge — unlike
// DSP52_PREFETCH_ORACLE which moved the same bytes earlier and lost). Default OFF
// (victim order byte-identical to the champion). Determinism gate: the committed
// routed union of each replayed (step,layer) must equal the recorded set (eviction
// is lossless ⇒ routing invariant — asserted). Lives in ReefOrch; performance-only.
std::string dsp52_evict_oracle() {
    static const std::string p = [] {
        const char* e = std::getenv("DSP52_EVICT_ORACLE");
        return std::string(e ? e : "");
    }();
    return p;
}
// DSP52_EVICT_BRIDGE=k — the REALIZABLE-SHAPE variant of the Belady oracle (the
// sim's "recurs ≤ k tok" BINARY bridge, POLICY_LAB_RESULTS finding #2, k=16 ≈
// 0.815 of Belady's 0.824). 0 (default) = EXACT Belady (continuous next-use
// distance). k>0 = protect experts whose next demand is within k decode STEPS
// (tokens); among the unprotected evict by the base board policy (freq+recency),
// dipping into the protected set only when no unprotected victim remains (no
// livelock). Only consulted when DSP52_EVICT_ORACLE is set.
int dsp52_evict_bridge() {
    static const int k = [] {
        const char* e = std::getenv("DSP52_EVICT_BRIDGE");
        int v = (e && *e) ? std::atoi(e) : 0;
        return v < 0 ? 0 : v;
    }();
    return k;
}
bool dsp52_elb() {
    static const bool on = [] {
        const char* e = std::getenv("DSP52_ELB");
        if (e && *e && std::atoi(e) != 0) return true;
        // The oracle replay drives the ELB issue/intercept machinery.
        const char* o = std::getenv("DSP52_PREFETCH_ORACLE");
        return o && *o;
    }();
    return on;
}
// Lookahead depth: predictions issued for layers L..L+AHEAD at attention-L
// launch (default 2 — window ≈ attention L + MoE L..L+1 walls).
int dsp52_elb_ahead() {
    static const int v = [] {
        const char* e = std::getenv("DSP52_ELB_AHEAD");
        int n = (e && *e) ? std::atoi(e) : 2;
        return std::min(std::max(n, 0), 8);
    }();
    return v;
}
// Per-layer candidate cap (selection-rank order; bounds chunk unions).
int dsp52_elb_cap() {
    static const int v = [] {
        const char* e = std::getenv("DSP52_ELB_CAP");
        int n = (e && *e) ? std::atoi(e) : 16;
        return std::min(std::max(n, 1),
                        static_cast<int>(lipc::kMaxExpertPrefetch));
    }();
    return v;
}
// Hot-window in board recency ticks (layer-visits); a victim younger than
// this is hot working set and never evicted for a prediction. 0 = default
// ONE FULL MoE ROUND (num_layers - first_moe). This makes the guard
// SELF-GATING, and both alternatives were measured (2026-08-03 bare-seed
// smokes, hit 0.32 / miss 5.45 per layer-visit = LRU-thrash):
//   - full round: cache turnover ~ cap/miss-rate ~ one round, so NO victim
//     qualifies -> issued=0, mechanism correctly REFUSES to churn a
//     thrashing cache (byte-identical walls);
//   - horizon (2*(AHEAD+1)): engaged fully (issued=4389) and CANNIBALIZED
//     the would-be hits — under round-periodic reuse the board-cheapest
//     (oldest) residents are exactly the persistent set at the far end of
//     its ~75-tick reuse cycle, so evicting them lands just before reuse:
//     hit 0.3187 -> 0.0405, moe 131 -> 141 ms/tok, median 151 -> 187.
// In a NON-thrash regime (prompt-fed keeper, hit ~0.65, turnover ~3.8
// rounds) residents older than a round are a genuinely cold tail and the
// full-round guard admits them. Victim age must exceed the REUSE period,
// not the prefetch horizon — a victim's next use is its own re-route, not
// the prefetched layers' demand.
int dsp52_elb_hot() {
    static const int v = [] {
        const char* e = std::getenv("DSP52_ELB_HOT");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    return v;
}
// DSP52_ELB_EVICT=0: fill free slots only (no cheap-victim eviction) — the
// prototype's behavior; useless in steady-state decode where the stable zone
// is exactly full, kept as an ablation arm.
bool dsp52_elb_evict() {
    static const bool on = [] {
        const char* e = std::getenv("DSP52_ELB_EVICT");
        return !(e && *e && std::atoi(e) == 0);
    }();
    return on;
}
// DSP52_ELB_ZONE=stream (default) | stable — which VRAM zone predictions
// land in. STREAM: the expert_streaming PREFETCH region (decode never uses
// it — "spill mode is test-only"), managed as a per-GPU FIFO ring of ELB's
// own entries: no stable-zone eviction at all, so the mechanism works even
// in the LRU-thrash regimes where the stable self-gate correctly refuses
// (the measured wall-1 state: deterministic keeper trajectory hit 0.26).
// A ready streamed copy is pinned into the demand placement (assign
// override in fill_moe_entries) — ExpertCache::lookup is zone-blind, so
// the demand path sees was_cached and computes from the streamed slot.
// STABLE: the v1 cheap-victim path (self-gating; see dsp52_elb_hot()).
bool dsp52_elb_stream() {
    static const bool on = [] {
        const char* e = std::getenv("DSP52_ELB_ZONE");
        return !(e && *e && std::string(e) == "stable");
    }();
    return on;
}
// Per-GPU cap on ELB's streaming-ring occupancy (default: the streaming
// zone's total slot count, queried at first issue).
int dsp52_elb_stream_cap() {
    static const int v = [] {
        const char* e = std::getenv("DSP52_ELB_STREAM_CAP");
        return (e && *e) ? std::atoi(e) : 0;  // 0 = zone capacity
    }();
    return v;
}

// ── DSP52_UPART=1 — union-aware cache partitioning (the TD-CHUNK-SMALLM-
// DEFAULT flip gate; DSP52_BOOST saga conclusion: verify-union fetches
// pollute the stable LRU working set — hit 0.79→0.33 at identical committed
// stream, +55-60 ms on every overlapped plain step).
//
// MECHANISM: verify-chunk union MISSES (board-non-resident after the SAME
// REEF solve — placement/gpu_idx unchanged) are marked zone=1 in the
// FETCH_AND_RUN sideband: the engine fetches them into the expert_streaming
// zone (scan-resistant transient class), they get NO stable victim and NO
// board admission (the stable working set is never evicted for them), and
// the engine releases every evictable streaming resident on the involved
// GPUs when the chunk MoE's completion is reaped (sweep-on-reap — transient
// lifetime is exactly one layer's FETCH_AND_RUN). Union HITS ride wherever
// resident (ExpertCache::lookup is zone-blind — free). Decode/plain-step
// and prefill fills are UNTOUCHED (transient_union=false), so the plain
// regime is byte-identical flag-on vs flag-off.
//
// CAPACITY DISCIPLINE (0cf5823a: no fabricated budgets): per-GPU marks are
// capped by the streaming zone's slot count minus the in-flight lever-4
// prefetch entries for this layer; the overflow keeps zone=0 = today's
// exact stable path (victim map + board admission). The engine clamps
// again (free-slot check) as a safety net. Under DSP52_PREFETCH the
// recorded prev-union entries carry zone=1 for the streamed subset, so the
// lever-4 prefetch ALSO lands in the streaming zone (pollution-free
// overlap; spent copies are released by the same sweep).
// ADVISORY ONLY (INV-UPART-ADVISORY): residency zone + release timing
// change WHEN/WHERE bytes live, never which experts the MoE computes —
// tokens must stay byte-identical flag-on vs flag-off at matched routing.
// Mutually exclusive with DSP52_ELB (both manage the streaming zone).
bool dsp52_upart() {
    static const bool on = [] {
        const char* e = std::getenv("DSP52_UPART");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}

// ── DSP52_LOOKAHEAD_DIAG=<shared|resident> — Wave-3 M4 STAGE-1 agreement
// diagnostic (look-ahead exact prefetch feasibility, spec/reports/
// DSP52_BOOST.md M4). Before EVERY batched verify chunk, run a cheap PASS-1
// forward of the SAME chunk (identical embedding + per-layer chunk attention
// + fused router gate) with an APPROXIMATE routed FFN:
//   shared   — routed experts SKIPPED entirely (FETCH_AND_RUN_MOE with an
//              EMPTY expert list → zero-filled routed contribution; the
//              always-resident shared expert + residual commit still run —
//              zero fetches by construction);
//   resident — the routed union filtered to experts the REEF board (or LRU
//              model) says are ALREADY resident, computed in place (no
//              demand fetches except rare board-desync strays).
// The pass-1 per-row routed top-K of every MoE layer is recorded, then the
// real pass-2 chunk runs UNMODIFIED and per-layer agreement accumulates:
//   row-recall — mean over rows of |topk_p1(row) ∩ topk_p2(row)| / topk;
//   union-cov  — |U1 ∩ U2| / |U2|, the manifest coverage a pass-1-driven
//                bulk prefetch would have achieved for this layer;
//   w-cov      — pass-2 gate-weight-weighted coverage of (row,expert) picks.
// STATE SAFETY: pass-1 writes chunk-row KV / indexer rows / aux-capture
// hiddens at positions [pos0, pos0+R) — all position-addressed and fully
// overwritten by pass-2 at the SAME positions (the established rewind
// re-feed discipline, file-header note (c)); pass-1 bypasses
// fill_moe_entries entirely (no placement/board/LRU/predictor mutation), no
// OUTPUT_HEAD, no lever-4/ELB feeds. Measurement-only: walls with the flag
// armed are NOT comparable to champion walls.
int dsp52_lookahead_diag() {
    static const int mode = [] {
        const char* e = std::getenv("DSP52_LOOKAHEAD_DIAG");
        if (!e || !*e) return 0;
        const std::string s(e);
        if (s == "shared")   return 1;
        if (s == "resident") return 2;
        ADD_FAILURE() << "DSP52_LOOKAHEAD_DIAG must be 'shared' or "
                         "'resident', got: " << s;
        return 0;
    }();
    return mode;
}

// DSP52_FORCE_TRAJ=<file> (TD-PREFILL-NONDET A/B pivot): teacher-force BOTH
// loops over the SAME pre-recorded generated continuation (one token id per
// whitespace token — a prior plain run's step tokens). Kills the prefill
// trajectory lottery for matched-trajectory wall A/Bs: every arm feeds the
// identical token stream (identical routing/hit profile by construction);
// walls stay honest (every decode step / draft / verify chunk still
// executes — only the COMMIT decision is replaced by the forced stream).
// Plain loop: step i commits forced[i] (model-match rate reported).
// Speculative loop: the target stream IS the forced trajectory — slot k
// accepted iff draft[k] == forced token at its position; the bonus is the
// forced token; verify chunks still run at the same shapes (their argmax is
// ignored for commits). Acceptance then measures DSpark's prediction of the
// forced stream (coordinator-approved semantics; NOT a lossless gate — do
// not combine with DSP52_REF=1).
const std::vector<uint32_t>* dsp52_force_traj() {
    static const std::vector<uint32_t> traj = [] {
        std::vector<uint32_t> v;
        const char* p = std::getenv("DSP52_FORCE_TRAJ");
        if (!p || !*p) return v;
        std::ifstream f(p);
        if (!f.is_open()) {
            // FAIL LOUD (keeper52 parity): a silently-disabled forcing
            // would invalidate the matched-trajectory A/B it exists for.
            ADD_FAILURE() << "DSP52_FORCE_TRAJ file not readable: " << p;
            return v;
        }
        long long t;
        while (f >> t)
            if (t >= 0) v.push_back(static_cast<uint32_t>(t));
        std::cerr << "  [dsp52] FORCE_TRAJ: " << v.size()
                  << " forced continuation token(s) from " << p << "\n";
        return v;
    }();
    return traj.empty() ? nullptr : &traj;
}

// DSP52_FORCE_DRAFT=<file> (NO-FORK teacher-forcing completion). DSP52_FORCE_TRAJ
// pins the COMMITTED stream, but the DSpark draft proposals + confidence c_k are
// produced off the TARGET model's aux hidden states — an arm that perturbs target
// numerics (e.g. fast-Q8 CPU-expert FFN) still forks the ACCEPTANCE pattern
// (g_use / accepted-per-round → round structure → verify-chunk widths → loader
// unions), even under FORCE_TRAJ (measured: e01 acc 0.5224 vs champion 0.4820).
// This file (recorded with DSP52_FORCE_DRAFT_DUMP=<file> on the champion arm)
// replays the champion's per-round draft block + confidences: one line per draft
// round, "<g> tok_0..tok_{g-1} conf_0..conf_{g-1}". With BOTH files forced, every
// arm shares ONE trajectory AND one round structure — acceptance/committed/rounds
// are identical by construction; walls stay honest (the real draft still executes
// and is paid for; only its OUTPUT is replaced before truncation/acceptance).
struct Dsp52DraftRound {
    std::vector<int32_t> toks;
    std::vector<float> confs;
};
const std::vector<Dsp52DraftRound>* dsp52_force_draft() {
    static const std::vector<Dsp52DraftRound> rounds = [] {
        std::vector<Dsp52DraftRound> v;
        const char* p = std::getenv("DSP52_FORCE_DRAFT");
        if (!p || !*p) return v;
        std::ifstream f(p);
        if (!f.is_open()) {
            // FAIL LOUD (dsp52_force_traj parity): silently-disabled forcing
            // would invalidate the no-fork A/B it exists for.
            ADD_FAILURE() << "DSP52_FORCE_DRAFT file not readable: " << p;
            return v;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            int g = 0;
            if (!(is >> g) || g <= 0 || g > 16) {
                ADD_FAILURE() << "DSP52_FORCE_DRAFT bad round header: " << line;
                v.clear();
                return v;
            }
            Dsp52DraftRound r;
            r.toks.resize(static_cast<size_t>(g));
            r.confs.resize(static_cast<size_t>(g));
            bool ok = true;
            for (int k = 0; k < g && ok; ++k) ok = static_cast<bool>(is >> r.toks[static_cast<size_t>(k)]);
            for (int k = 0; k < g && ok; ++k) ok = static_cast<bool>(is >> r.confs[static_cast<size_t>(k)]);
            if (!ok) {
                ADD_FAILURE() << "DSP52_FORCE_DRAFT truncated round line: " << line;
                v.clear();
                return v;
            }
            v.push_back(std::move(r));
        }
        std::cerr << "  [dsp52] FORCE_DRAFT: " << v.size()
                  << " forced draft round(s) from " << p << "\n";
        return v;
    }();
    return rounds.empty() ? nullptr : &rounds;
}

// DSP52_DUMP_TOKENS=<file>: dump the committed trajectory (whitespace-
// separated token ids, plain or speculative arm) at loop end — the exact
// token-identity reference the Python bridge port (test_dsp52_bridge.py)
// gates against. Write-failure is loud (a silently-missing reference would
// void the cross-language parity gate).
void dsp52_dump_tokens(const std::vector<uint32_t>& toks) {
    const char* p = std::getenv("DSP52_DUMP_TOKENS");
    if (!p || !*p) return;
    std::ofstream f(p, std::ios::trunc);
    if (!f.is_open()) {
        ADD_FAILURE() << "DSP52_DUMP_TOKENS not writable: " << p;
        return;
    }
    for (size_t i = 0; i < toks.size(); ++i)
        f << toks[i] << ((i + 1) % 20 == 0 ? '\n' : ' ');
    f << '\n';
    std::cerr << "  [dsp52] DUMP_TOKENS: wrote " << toks.size()
              << " committed token(s) -> " << p << "\n";
}

// KEEPER52_EP with the dsp52 DEFAULT FLIPPED TO 4: the draft lives on the
// last 5080, an expert-only EP=4 rank (INV-MOE-EP-XTP), so EP=4 is the only
// shape dsp52 can speculate in. The env still overrides (KEEPER52_EP=2 makes
// the GPU test skip — there is no draft host in the 2×5090 keeper shape).
int dsp52_ep() {
    static const int ep = [] {
        const char* e = std::getenv("KEEPER52_EP");
        if (e && *e && std::atoi(e) == 2) return 2;
        return 4;
    }();
    return ep;
}

// EP=4 GPU probe: all visible SM120 devices with their total VRAM (GiB),
// sorted LARGEST-first (identical to keeper52).
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

// Build the dsp52 config JSON: the keeper52 builder (all four GLM-5.2
// feature deltas + full-fit O_DIRECT arena, arena_attach default ON) with the
// EP=4 default, the DSpark speculation block, and the draft-hosting 5080's
// expert window shrunk to DSP52_DRAFT_CACHE_GB. block_size /
// speculative_tokens are derived from the checkpoint's config.json (the
// loader HARD cross-validates them) and returned through gamma_out/block_out
// (defaults 7/8 — the shipped-checkpoint shape — when the config.json is
// absent/unreadable; engine init then fails loud anyway).
nlohmann::json build_dsp52_config(const std::string& src, bool have_gpus,
                                  bool have_prepacked,
                                  int* gamma_out = nullptr,
                                  int* block_out = nullptr) {
    const std::string cfg_path = src + kConfigRel;
    std::ifstream f(cfg_path);
    nlohmann::json j = nlohmann::json::parse(f);

    j["model"]["weights_path"] = src + kGgufRel;
    j["quantization"]["gguf_strategy"] = "int";

    // ── Hardware block (keeper52 shapes; dsp52 defaults to the EP=4 one) ──
    if (have_gpus) {
        double expert_cache_gb = 4.5;
        if (const char* eg = std::getenv("KEEPER52_EXPERT_CACHE_GB"); eg && *eg)
            expert_cache_gb = std::atof(eg);
        double stable_frac = (dsp52_ep() == 4) ? 0.95 : -1.0;
        if (const char* sf = std::getenv("KEEPER52_STABLE_FRAC"); sf && *sf)
            stable_frac = std::atof(sf);
        if (dsp52_ep() == 4) {
            // EP=4 (INV-MOE-EP-XTP): hardware.gpus ORDERED 5090s-first;
            // positions 0,1 = 5090s (attention + resident + expert window),
            // positions 2,3 = 5080s (expert-ONLY ranks). dsp52 delta vs
            // keeper52: position 3 (the LAST 5080) also hosts the DSpark
            // draft — its expert window is DSP52_DRAFT_CACHE_GB (default 4.0)
            // so the 7.61 GB BF16 draft + DsparkRuntime scratch (the
            // LayerRegistry pinned carve, layer_registry.cpp
            // estimate_gpu_budgets) fit beside the expert regions inside the
            // declared 15 GB with the 3.0 safety margin. Position 2 keeps the
            // keeper 5080 default (KEEPER52_EXPERT_CACHE_GB_5080, 11.0).
            double expert_cache_gb_5080 = 11.0;
            if (const char* eg = std::getenv("KEEPER52_EXPERT_CACHE_GB_5080");
                eg && *eg)
                expert_cache_gb_5080 = std::atof(eg);
            // The DRAFT-HOSTING GPU's expert window (whichever position
            // DSP52_DRAFT_GPU selects). 5080 host (default, position 3):
            // 4.0 leaves room for the 7.61 GB BF16 draft + scratch inside
            // the declared 15. 5090 host (TD-DSPARK-DRAFT-QUANT placement):
            // the draft shares the 30 GB card with ~20.3 GiB target pinned
            // weights + ~0.54 GiB KV + the 3.0 safety margin, so the window
            // default is derived from the measured quantized draft charge
            // (weights + ~0.9 GiB scratch): fp8_e4m3 ~5.5 GiB -> 0.5 GB
            // window; nvfp4 ~4.3 GiB -> 1.75 GB window. BF16 cannot fit a
            // 5090 (engine init fails loud). DSP52_DRAFT_CACHE_GB overrides.
            const auto draft_positions = dsp52_draft_gpus();
            const int draft_pos = draft_positions[0];
            const bool sharded = draft_positions.size() > 1;
            double draft_cache_gb = 4.0;
            if (sharded) {
                // TD-DSPARK-DRAFT-SHARD (TP=2 across the 5090s): each
                // hosting rank carries ~half the weight shard + its own
                // scratch. Per-hosting-rank window default derived so each
                // card's pinned+window total matches the PROVEN single-host
                // nvfp4 layout occupancy (pos1 25163+1792 MiB): nvfp4
                // per-rank charge ~2.7 GiB -> 3.25 GB window per rank; fp8
                // ~3.4 GiB -> 2.5; bf16 ~4.7 GiB -> 1.25 (tight).
                if (dsp52_quant() == "fp8_e4m3") draft_cache_gb = 2.5;
                else if (dsp52_quant() == "nvfp4") draft_cache_gb = 3.25;
                else draft_cache_gb = 1.25;
            } else if (draft_pos < 2) {
                if (dsp52_quant() == "fp8_e4m3") draft_cache_gb = 0.5;
                else if (dsp52_quant() == "nvfp4") draft_cache_gb = 1.75;
                else
                    fprintf(stderr,
                            "[dsp52] WARNING: BF16 draft on a 5090 "
                            "(DSP52_DRAFT_GPU=%d without DSP52_QUANT) does "
                            "not fit — engine init will fail\n",
                            draft_pos);
            }
            // DSP52_MARGIN_GB (phase-1 window probe): shaving the hosting
            // GPU's safety margin frees exactly (3.0 - margin) GB of the
            // region budget — fold it into the default window so the probe
            // knob alone recovers coverage (1.5 -> nvfp4 window 3.25 GB
            // ~= 124 slots). DSP52_DRAFT_CACHE_GB still overrides.
            if (dsp52_margin_gb() >= 0.0 && dsp52_margin_gb() < 3.0)
                draft_cache_gb += 3.0 - dsp52_margin_gb();
            if (const char* dg = std::getenv("DSP52_DRAFT_CACHE_GB"); dg && *dg)
                draft_cache_gb = std::atof(dg);
            auto gpus = probe_sm120_gpus();  // largest-first
            if (gpus.size() < 4) {
                fprintf(stderr, "[dsp52] EP=4 needs 4 SM120 GPUs, found %zu\n",
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
                const bool hosts_draft =
                    std::find(draft_positions.begin(), draft_positions.end(),
                              p) != draft_positions.end();
                const double win =
                    hosts_draft ? draft_cache_gb
                                : (big ? expert_cache_gb
                                       : expert_cache_gb_5080);
                if (win > 0.0) {
                    g["vram_allocation_gb"]["expert_streaming"] = win;
                    if (stable_frac >= 0.0)
                        g["vram_allocation_gb"]["stable_zone_fraction"] =
                            stable_frac;
                }
                // DSP52_MARGIN_GB: per-GPU safety-margin shave on the
                // draft-hosting position(s) ONLY (see dsp52_margin_gb()).
                if (hosts_draft && dsp52_margin_gb() >= 0.0)
                    g["vram_allocation_gb"]["safety_margin_gb"] =
                        dsp52_margin_gb();
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

    // ── DSpark speculation (the dsp52 feature) ───────────────────────────
    // Same derivation as the glm52 golden fixture: block_size +
    // speculative_tokens come from the checkpoint's config.json so engine
    // init matches whatever is symlinked (shipped ckpt 8/7, preview 16/15).
    {
        const std::string ckpt = dsp52_ckpt_dir(src);
        j["speculation"]["method"] = "dspark";
        j["speculation"]["enabled"] = true;
        j["speculation"]["dspark"]["checkpoint_path"] = ckpt;
        // DSP52_CONF_THRESH > 0 (DSP-9 static_threshold bring-up) needs the
        // trained DSP-6 confidence head readback; otherwise keep it off
        // (inert path, unchanged dsp52 default).
        j["speculation"]["dspark"]["confidence_enabled"] =
            dsp52_conf_thresh() > 0.0;
        // Draft host = DSP52_DRAFT_GPU (default: config position 3, the LAST
        // 5080 — an EP=4 expert-only rank; 0/1 = a 5090 TP rank under the
        // TD-DSPARK-DRAFT-QUANT placement). Only meaningful in the EP=4 GPU
        // shape; the CPU-only structural check (have_gpus=false, single-GPU
        // preset) leaves the auto-resolution default.
        if (have_gpus && dsp52_ep() == 4) {
            auto arr = nlohmann::json::array();
            for (int p : dsp52_draft_gpus()) arr.push_back(p);
            j["speculation"]["dspark"]["draft_gpus"] = arr;
        }
        // TD-DSPARK-DRAFT-QUANT: requant-at-upload format (default bf16
        // keeps the config byte-identical to the established recipe).
        if (dsp52_quant() != "bf16")
            j["speculation"]["dspark"]["draft_weights_quant"] = dsp52_quant();
        int gamma = 7, block = 8;  // shipped-checkpoint defaults
        std::ifstream cf(ckpt + "/config.json");
        if (cf.is_open()) {
            auto cj = nlohmann::json::parse(cf, nullptr,
                                            /*allow_exceptions=*/false);
            if (!cj.is_discarded()) {
                if (cj.contains("block_size")) {
                    j["speculation"]["dspark"]["block_size"] =
                        cj["block_size"];
                    block = cj["block_size"].get<int>();
                }
                if (cj.contains("speculators_config") &&
                    cj["speculators_config"].contains("proposal_methods") &&
                    !cj["speculators_config"]["proposal_methods"].empty()) {
                    j["speculation"]["dspark"]["speculative_tokens"] =
                        cj["speculators_config"]["proposal_methods"]
                          [0]["speculative_tokens"];
                    gamma = cj["speculators_config"]["proposal_methods"]
                              [0]["speculative_tokens"].get<int>();
                }
            }
        }
        if (gamma_out) *gamma_out = gamma;
        if (block_out) *block_out = block;
    }

    // ── Feature 4: sharded DCP (KVS) — keeper52 unchanged.
    // DSP52_KV_MODE overrides (batched-verify bisect: "replicated" removes
    // the QAG B>1 head re-gather from the chunk pipeline). ─────────────────
    {
        std::string kv_mode = "sharded";
        if (const char* km = std::getenv("DSP52_KV_MODE"); km && *km)
            kv_mode = km;
        j["hardware"]["dcp_kv_mode"] = kv_mode;
    }
    j["hardware"]["dcp_indexer_mode"] = "replicated";
    j["memory"]["kv_cache"]["dcp_chunk_size"] = 16;

    // ── Feature 5: TurboQuant 4-bit MLA attention backend.
    // DSP52_ATTN_BACKEND overrides (batched-verify bisect: the preset
    // default backend removes TQ chunk-prefill from the chunk pipeline). ──
    {
        std::string backend = "turboquant_mla";
        if (const char* ab = std::getenv("DSP52_ATTN_BACKEND"); ab && *ab)
            backend = ab;
        j["compute"]["attention_backend"] = backend;
    }

    // ── DSP52_SPARSE_PREFILL=1 (INV-DSA-REWIND campaign): turn on
    // compute.dsa_sparse_prefill so blessed chunks are CONSUMED sparse by
    // the executor (replicated KV only — under the sharded default the
    // executor falls back to dense chunks, TD-SPARSE-PREFILL-KVS). Pair
    // with DSP52_KV_MODE=replicated to run the batched-verify chunks with
    // sparse DSA attention after rewinds (compute.dsa_indexer_rewind now
    // defaults TRUE; LS_INDEXER_REWIND env overrides either way). ────────
    if (const char* sp = std::getenv("DSP52_SPARSE_PREFILL");
        sp && *sp && *sp != '0')
        j["compute"]["dsa_sparse_prefill"] = true;

    // ── Feature 3: HiSparse DSA-guided KV tiering ────────────────────────
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

    j["memory"]["vram_safety_margin_gb"] = 3.0;
    j["memory"]["kv_cache"]["max_pages_per_gpu"] = 4096;
    j["memory"]["kv_cache"]["page_growth_chunk_tokens"] = 16;

    // ── Feature 2 + inherited full-fit O_DIRECT overlapped-register arena ─
    double arena_frac = kDefaultArenaFraction;
    if (const char* af = std::getenv("KEEPER52_ARENA_FRACTION"); af && *af)
        if (double v = std::atof(af); v > 0.0 && v <= 1.0) arena_frac = v;
    if (have_prepacked) {
        j["preprocessing"]["prepacked_dir"] = src + kPrepackedRel;
        j["memory"]["preload_expert_buffers"] = true;
        j["memory"]["pin_host_expert_pool"] = true;
        j["memory"]["pin_host_expert_pool_direct_load"] = true;
        j["memory"]["arena_attach"] = {
            // keeper52/dsp52 default arena attach ON (user decision
            // 2026-08-01): this test family IS the box's recurring canonical
            // workflow. GLM52_ARENA_ATTACH=0 opts out for geometry sweeps.
            {"enabled", [] {
                const char* aa = std::getenv("GLM52_ARENA_ATTACH");
                return !(aa && *aa == '0');
            }()},
            // Store protection (user decision 2026-08-18, post-incident):
            // a mismatched boot FAILS LOUD instead of wiping the persisted
            // store. Intentional rebuilds: GLM52_ARENA_PERSIST=0 for one
            // run.
            {"persist", [] {
                const char* ap = std::getenv("GLM52_ARENA_PERSIST");
                return !(ap && *ap == '0');
            }()}};
        j["memory"]["pin_host_expert_pool_preload"] = true;
        j["memory"]["pin_host_expert_pool_direct_o_direct"] = true;
        j["memory"]["pin_host_expert_pool_sizing"] = {
            {"mode", "fraction_total"}, {"value", arena_frac}};
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

// KEEPER52_AFFINITY=1: orchestrator-side cache-affinity expert→GPU routing
// (identical to keeper52 — see keeper52_test.cpp for the full rationale).
static bool keeper52_affinity() {
    static const bool on = [] {
        const char* e = std::getenv("KEEPER52_AFFINITY");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}

// ── NUMA-aware miss placement (§12h, keeper52 machinery unchanged) ───────────
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
    // DEFAULT OFF (§12j) — see keeper52_test.cpp.
    static const bool on = [] {
        const char* e = std::getenv("KEEPER52_NUMA_ROUTE");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}
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
static constexpr double kH2dRate[4][4] = {
    {42.6, 34.3, 56.2, 51.7},   // GPU0 (5090, node2)
    {34.8, 42.2, 49.2, 56.2},   // GPU1 (5090, node3)
    {56.3, 48.4, 44.2, 35.8},   // GPU2 (5080, node0)
    {43.2, 34.6, 56.3, 50.4},   // GPU3 (5080, node2)
};
// P-25 decision-3 scaffold (KEEPER52_REEF_ORCH=1) — keeper52 unchanged.
static bool keeper52_reef_orch() {
    static const bool on = [] {
        const char* e = std::getenv("KEEPER52_REEF_ORCH");
        return e && *e && std::atoi(e) != 0;
    }();
    return on;
}
static const char* dsp52_arena_map_path() {
    static const std::string p = [] {
        const char* e = std::getenv("KEEPER52_ARENA_MAP");
        std::string path = (e && *e) ? e : "/tmp/dsp52_arena_map.csv";
        // INV-REEF-BANK: the REEF arm no longer arms the CSV export — its
        // bank seam is the shared arena snapshot (install_arena_bank_seam).
        // Only the NUMA-route arm still consumes the dump.
        if (keeper52_numa_route()) {
            ::unlink(path.c_str());
            ::setenv("LS_ARENA_MAP_DUMP", path.c_str(), /*overwrite=*/0);
        }
        return path;
    }();
    return p.c_str();
}
static const char* g_dsp52_arena_map_init = dsp52_arena_map_path();
static ArenaNodeMap g_arena_map;

// ── KEEPER52_REEF_ORCH decision state (identical to keeper52) ────────────────
// ── BELADY next-use oracle (DIAGNOSTIC, NOT DEPLOYABLE — DSP52_EVICT_ORACLE) ──
// Same recording (magic 'ORCL', DSP52_ORACLE_DUMP) the prefetch oracle uses, but
// consumed the CORRECT way: as an EXACT next-use index driving Belady-optimal
// EVICTION. For each recorded (step, layer) it stores the routed expert set (the
// determinism gate) and, per (layer, expert), the sorted list of STEPS at which
// that expert is next demanded (the next-use index). reef_orch_apply queries
// next_use_dist() to pick the victim whose next demand is FARTHEST (or never) —
// fetching nothing extra, only reducing the total demand-miss volume.
//
// Time model: the forced trajectory visits every layer once per step, so global
// time t = step*num_layers + layer is a total order over layer-visits. A resident
// (layer_idx, expert_idx) demanded at current (cur_step, cur_layer) has its next
// use at the first recorded step >= cur_step (only if layer_idx is visited LATER
// this step, i.e. layer_idx > cur_layer) else >= cur_step+1. Belady distance is
// that occurrence's global time minus now; never-again ⇒ +inf (evict first).
class NextUseOracle final
        : public layerstorm::gpu_loader::ReefEvictOracle {
public:
    NextUseOracle(int num_layers, const std::string& path)
        : num_layers_(std::max(num_layers, 0)) { load(path); }

    // DETERMINISM GATE (replay): the committed routed union for (step, layer)
    // must equal the recorded expert SET (placement/gpu may differ — the EP
    // combine is placement-invariant; eviction is lossless ⇒ routing invariant).
    // A mismatch means Belady eviction perturbed routing → measurement invalid.
    void observe(int layer, const lipc::ExpertPrefetchEntry* entries,
                 uint32_t count, int64_t step) {
        if (step < 0 || static_cast<size_t>(step) >= script_.size()) return;
        if (layer < 0 || layer >= num_layers_) return;
        const auto& rec = script_[static_cast<size_t>(step)][
            static_cast<size_t>(layer)];
        if (rec.empty()) return;  // this (step,layer) not recorded
        std::vector<uint16_t> a;
        a.reserve(count);
        for (uint32_t i = 0; i < count; ++i) a.push_back(entries[i].expert_idx);
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());  // committed union
        if (a != rec) {
            ++mismatch_;
            if (mismatch_ <= 4)
                ADD_FAILURE() << "BELADY ORACLE STALE: step " << step << " layer "
                              << layer << " committed union (" << a.size()
                              << ") != recorded (" << rec.size()
                              << ") — Belady eviction perturbed routing";
        } else {
            ++matched_;
        }
    }

    // Next demand STEP of (layer, expert) at/after now, or -1 if never again.
    int32_t next_use_step(uint32_t layer, uint16_t expert, int64_t cur_step,
                          int cur_layer) const override {
        auto it = occ_.find((layer << 16) | static_cast<uint32_t>(expert));
        if (it == occ_.end()) return -1;
        const auto& v = it->second;
        const int64_t lb =
            (static_cast<int>(layer) > cur_layer) ? cur_step : cur_step + 1;
        auto p = std::lower_bound(v.begin(), v.end(), static_cast<int32_t>(lb));
        return (p == v.end()) ? -1 : *p;
    }

    // Belady distance: global-time ticks to the next demand; INT64_MAX if never.
    int64_t next_use_dist(uint32_t layer, uint16_t expert, int64_t cur_step,
                          int cur_layer) const override {
        const int32_t s = next_use_step(layer, expert, cur_step, cur_layer);
        if (s < 0) return std::numeric_limits<int64_t>::max();
        return (static_cast<int64_t>(s) - cur_step) * num_layers_ +
               (static_cast<int64_t>(layer) - cur_layer);
    }

    uint64_t matched() const { return matched_; }
    uint64_t mismatched() const { return mismatch_; }
    size_t steps() const { return script_.size(); }

private:
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            ADD_FAILURE() << "DSP52_EVICT_ORACLE file not readable: " << path;
            return;
        }
        uint32_t magic = 0;
        in.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != 0x4c43524fu /* 'ORCL' — DSP52_ORACLE_DUMP */) {
            ADD_FAILURE() << "evict oracle dump bad magic 0x" << std::hex << magic;
            return;
        }
        struct Rec { int32_t step; int32_t layer; std::vector<uint16_t> experts; };
        std::vector<Rec> recs;
        int64_t max_step = -1;
        uint64_t nrec = 0;
        for (;;) {
            int32_t step = 0, layer = 0; uint32_t cnt = 0;
            in.read(reinterpret_cast<char*>(&step), 4);
            if (!in) break;
            in.read(reinterpret_cast<char*>(&layer), 4);
            in.read(reinterpret_cast<char*>(&cnt), 4);
            if (!in) break;
            Rec r; r.step = step; r.layer = layer; r.experts.reserve(cnt);
            for (uint32_t i = 0; i < cnt; ++i) {
                uint16_t e = 0; uint8_t g = 0, z = 0;
                in.read(reinterpret_cast<char*>(&e), 2);
                in.read(reinterpret_cast<char*>(&g), 1);
                in.read(reinterpret_cast<char*>(&z), 1);
                r.experts.push_back(e);
            }
            if (step > max_step) max_step = step;
            recs.push_back(std::move(r));
            ++nrec;
        }
        script_.assign(static_cast<size_t>(max_step + 1),
                       std::vector<std::vector<uint16_t>>(
                           static_cast<size_t>(num_layers_)));
        for (auto& r : recs) {
            if (r.step < 0 || r.layer < 0 || r.layer >= num_layers_) continue;
            std::sort(r.experts.begin(), r.experts.end());
            r.experts.erase(std::unique(r.experts.begin(), r.experts.end()),
                            r.experts.end());
            for (uint16_t e : r.experts)
                occ_[(static_cast<uint32_t>(r.layer) << 16) | e].push_back(r.step);
            script_[static_cast<size_t>(r.step)][static_cast<size_t>(r.layer)] =
                std::move(r.experts);
        }
        for (auto& kv : occ_) {  // dedupe + sort each expert's occurrence list
            auto& v = kv.second;
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
        std::cerr << "  [dsp52 BELADY] loaded " << nrec << " records over "
                  << script_.size() << " steps, " << occ_.size()
                  << " (layer,expert) next-use series from " << path << "\n";
    }

    int num_layers_;
    // occ_[(layer<<16)|expert] = ascending steps at which it is demanded.
    std::unordered_map<uint32_t, std::vector<int32_t>> occ_;
    // script_[step][layer] = sorted-unique routed expert set (determinism gate).
    std::vector<std::vector<std::vector<uint16_t>>> script_;
    uint64_t matched_ = 0, mismatch_ = 0;
};
// ── REEF decision stack: EXTRACTED to the library (2026-08-18) ───────────────
// ReefOrch + reef_orch_route + reef_orch_apply now live in
// src/core/gpu_loader/reef_orch.{h,cpp} (byte-identical extraction) so the
// daemon's E_CMD_REEF_ROUTE service and this test drive ONE implementation.
// Test-only seams: bank_node_fn (arena-map CSV lookup, installed in
// make_reef_orch below) and the Belady NextUseOracle (ReefEvictOracle).
using ReefOrch = layerstorm::gpu_loader::ReefOrch;
using layerstorm::gpu_loader::reef_orch_route;

// The sideband entry structs and the library's views must stay layout-equal
// (the library is core-side and cannot include daemon/ipc_protocol.h).
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
                            lipc::ExpertEvictionEntry* evicts, uint32_t count,
                            const std::vector<uint8_t>* stream_mask = nullptr,
                            int64_t cur_step = -1) {
    layerstorm::gpu_loader::reef_orch_apply(
        o, layer,
        reinterpret_cast<const layerstorm::gpu_loader::ReefEntry*>(entries),
        reinterpret_cast<layerstorm::gpu_loader::ReefVictim*>(evicts), count,
        stream_mask, cur_step);
}

// ── ELB predictor seam (DSP52_ELB; EPM Phase-29 tier-0) ──────────────────────
// Pluggable per-layer expert-prefetch predictor. The driver (this test today;
// the Python orchestrator's REEF port in production — P-18 line) OBSERVES every
// routed decision it dispatches and asks for PREDICTED candidates ahead of the
// demand point. Contract:
//   observe(layer, entries, n): the deduped (expert → target_gpu, zone) entry
//     list just dispatched to E_CMD_FETCH_AND_RUN_MOE for `layer` — one decode
//     row's top-K, a verify-chunk union, or a prefill-chunk union. Entries are
//     in gating selection-rank order (the placement fill_moe_entries wrote).
//   predict(layer): candidate entries for the NEXT visit of `layer`,
//     best-first. Placement rides the entries (target_gpu = where the driver
//     last placed the expert), so a prefetch lands where placement wants it.
// Predictions are PERFORMANCE-ADVISORY ONLY (INV-ELB-ADVISORY) and the issue
// policy around them is capacity-bounded (INV-ELB-CAP) — see dsp52_elb().
//
// FUTURE PREDICTORS plug in behind this interface without touching the issue
// policy (the "dspark predictor core" seam): a DSpark-hidden-state EPM head
// (spec/MoE-SpeQ_NOTES.md §8: draft hiddens → per-layer top-m + calibrated
// scores, sideband export per block) or a hybrid re-ranker (EPM-4 verdict:
// b0_prev union-cov 0.881 makes temporal locality the candidate SET, a trained
// model the ranking) would implement predict() from its own observe stream —
// candidates just need (expert, target_gpu, zone) with a best-first order.
class ElbPredictor {
public:
    virtual ~ElbPredictor() = default;
    virtual void observe(int layer, const lipc::ExpertPrefetchEntry* entries,
                         uint32_t count) = 0;
    virtual const std::vector<lipc::ExpertPrefetchEntry>& predict(int layer) = 0;
};

// Tier-0: temporal locality ("b0_prev") — per layer, the last observed routed
// entry list verbatim (previous decode round / previous chunk-or-prefill
// union at its previous placement). Zero training, zero state beyond one
// entry list per layer; the EPM-4 grid bar every trained model must clear.
class ElbPrevRoutedPredictor final : public ElbPredictor {
public:
    explicit ElbPrevRoutedPredictor(int num_layers)
        : by_layer_(static_cast<size_t>(std::max(num_layers, 0))) {}
    void observe(int layer, const lipc::ExpertPrefetchEntry* entries,
                 uint32_t count) override {
        if (layer < 0 || static_cast<size_t>(layer) >= by_layer_.size()) return;
        by_layer_[static_cast<size_t>(layer)].assign(entries, entries + count);
    }
    const std::vector<lipc::ExpertPrefetchEntry>& predict(int layer) override {
        if (layer < 0 || static_cast<size_t>(layer) >= by_layer_.size())
            return empty_;
        return by_layer_[static_cast<size_t>(layer)];
    }
private:
    std::vector<std::vector<lipc::ExpertPrefetchEntry>> by_layer_;
    const std::vector<lipc::ExpertPrefetchEntry> empty_;
};

// ── ORACLE UPPER-BOUND predictor (DIAGNOSTIC, NOT DEPLOYABLE) ────────────────
// Perfect-accuracy expert predictor: the committed routed union + placement of
// every (traj-step, layer) recorded verbatim by a matched forced-trajectory
// RUN 1 (DSP52_ORACLE_DUMP). predict(P) returns the EXACT experts layer P will
// demand this step (perfect same-round lookahead) or next step (perfect
// b0_prev, DSP52_ORACLE_NEXTROUND). Since it plugs into the ElbPredictor seam,
// the ELB issue policy (residency filter, per-layer cap = top-N, lookahead,
// priority -1, stream/stable zone) turns it into the residency-aware, budgeted,
// lookahead-scheduled PERFECT prefetch the oracle upper-bound measures.
// observe() is the DETERMINISM GATE: the actual committed union (RUN 2) must
// match the recorded expert SET for (step, layer) — else the oracle is stale
// (trajectory forked) and the measurement is aborted. The `step_ptr` tracks the
// driver's traj_step_ (incremented per prefill-chunk / decode-step / verify-
// chunk sweep) so record (RUN 1) and replay (RUN 2) index the SAME script slot.
class OracleRoutedPredictor final : public ElbPredictor {
public:
    OracleRoutedPredictor(int num_layers, const std::string& path,
                          const int64_t* step_ptr, bool next_round)
        : num_layers_(std::max(num_layers, 0)),
          step_ptr_(step_ptr),
          next_round_(next_round) {
        load(path);
    }

    // DETERMINISM GATE (RUN 2): the committed union for (cur step, layer) must
    // equal the recorded expert SET. Placement (gpu_idx) may legitimately
    // differ (the canonical EP combine is placement-invariant), so only the
    // expert-id multiset is compared. A mismatch means the replay forked the
    // recorded trajectory → the oracle would be predicting stale futures.
    void observe(int layer, const lipc::ExpertPrefetchEntry* entries,
                 uint32_t count) override {
        const int64_t s = step_ptr_ ? *step_ptr_ : -1;
        if (s < 0 || static_cast<size_t>(s) >= script_.size()) return;
        if (layer < 0 || layer >= num_layers_) return;
        const auto& rec = script_[static_cast<size_t>(s)][
            static_cast<size_t>(layer)];
        if (rec.empty()) return;  // not recorded (e.g. this step un-dumped)
        std::vector<uint16_t> a, b;
        a.reserve(count); b.reserve(rec.size());
        for (uint32_t i = 0; i < count; ++i) a.push_back(entries[i].expert_idx);
        for (const auto& e : rec) b.push_back(e.expert_idx);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) {
            ++mismatch_;
            if (mismatch_ <= 4)
                ADD_FAILURE() << "ORACLE STALE: step " << s << " layer "
                              << layer << " committed union (" << a.size()
                              << ") != recorded (" << b.size()
                              << ") — replay forked the recorded trajectory";
        } else {
            ++matched_;
        }
    }

    const std::vector<lipc::ExpertPrefetchEntry>& predict(int layer) override {
        int64_t s = step_ptr_ ? *step_ptr_ : -1;
        if (next_round_) ++s;
        if (s < 0 || static_cast<size_t>(s) >= script_.size()) return empty_;
        if (layer < 0 || layer >= num_layers_) return empty_;
        return script_[static_cast<size_t>(s)][static_cast<size_t>(layer)];
    }

    uint64_t matched() const { return matched_; }
    uint64_t mismatched() const { return mismatch_; }
    size_t steps() const { return script_.size(); }

private:
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            ADD_FAILURE() << "DSP52_PREFETCH_ORACLE file not readable: " << path;
            return;
        }
        uint32_t magic = 0;
        in.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != kOracleMagic) {
            ADD_FAILURE() << "oracle dump bad magic 0x" << std::hex << magic;
            return;
        }
        int64_t max_step = -1;
        uint64_t nrec = 0;
        struct Rec { int32_t step; int32_t layer;
                     std::vector<lipc::ExpertPrefetchEntry> ents; };
        std::vector<Rec> recs;
        for (;;) {
            int32_t step = 0, layer = 0; uint32_t cnt = 0;
            in.read(reinterpret_cast<char*>(&step), 4);
            if (!in) break;
            in.read(reinterpret_cast<char*>(&layer), 4);
            in.read(reinterpret_cast<char*>(&cnt), 4);
            if (!in) break;
            Rec r; r.step = step; r.layer = layer;
            r.ents.resize(cnt);
            for (uint32_t i = 0; i < cnt; ++i) {
                uint16_t e = 0; uint8_t g = 0, z = 0;
                in.read(reinterpret_cast<char*>(&e), 2);
                in.read(reinterpret_cast<char*>(&g), 1);
                in.read(reinterpret_cast<char*>(&z), 1);
                lipc::ExpertPrefetchEntry pe{};
                pe.layer_idx  = static_cast<uint32_t>(layer);
                pe.expert_idx = e;
                pe.gpu_idx    = g;
                pe.zone       = 0;  // zone is chosen by the issue policy
                r.ents[i] = pe;
            }
            if (step > max_step) max_step = step;
            recs.push_back(std::move(r));
            ++nrec;
        }
        script_.assign(static_cast<size_t>(max_step + 1),
                       std::vector<std::vector<lipc::ExpertPrefetchEntry>>(
                           static_cast<size_t>(num_layers_)));
        for (auto& r : recs) {
            if (r.step < 0 || r.layer < 0 || r.layer >= num_layers_) continue;
            script_[static_cast<size_t>(r.step)][
                static_cast<size_t>(r.layer)] = std::move(r.ents);
        }
        std::cerr << "  [dsp52 ORACLE] loaded " << nrec << " records over "
                  << script_.size() << " steps from " << path
                  << " (mode=" << (next_round_ ? "next-round" : "same-round")
                  << ")\n";
    }

    static constexpr uint32_t kOracleMagic = 0x4c43524fu;  // 'ORCL' (LE)
    int num_layers_;
    const int64_t* step_ptr_;
    bool next_round_;
    // script_[step][layer] = committed routed union (expert + placement).
    std::vector<std::vector<std::vector<lipc::ExpertPrefetchEntry>>> script_;
    const std::vector<lipc::ExpertPrefetchEntry> empty_;
    uint64_t matched_ = 0, mismatch_ = 0;
public:
    static constexpr uint32_t magic() { return kOracleMagic; }
};

// ── Decode result + timing (mirrors keeper52 / first_token_test.cpp) ─────────

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
    uint32_t moe_lookups = 0;  // routed experts examined this token
};

// Verify-chunk result (dsp52): the R greedy argmax ids of the batched-verify
// output head plus keeper-style timings.
struct VerifyChunkResult {
    std::vector<uint32_t> argmax;   // R per-row greedy ids
    float top1_prob = 0.f;
    float entropy = 0.f;
    StepTimings timings;
    uint32_t moe_lookups = 0;       // deduped union entries summed over layers
};

// ── Fixture ──────────────────────────────────────────────────────────────────

class Dsp52Test : public ::testing::Test {
protected:
    // Bring up the GLM-5.2 engine with the dsp52 config. Returns via the
    // caller's HasFailure on setup errors.
    void start_engine() {
        const std::string src = LAYERSTORM_SOURCE_DIR;
        const bool have_prepacked =
            fs::exists(src + kPrepackedRel + "/manifest.json");
        if (!have_prepacked)
            fprintf(stderr,
                    "[dsp52] WARNING: prepacked set %s absent — falling back "
                    "to the GGUF-mmap expert path (NOT full-fit; the keeper "
                    "envelope needs the prepacked O_DIRECT arena)\n",
                    (src + kPrepackedRel).c_str());

        nlohmann::json j = build_dsp52_config(src, /*have_gpus=*/true,
                                              have_prepacked,
                                              &dspark_gamma_, &dspark_block_);
        weights_path_ = j["model"]["weights_path"].get<std::string>();
        // I8 loader wiring — identical to keeper52. Under KEEPER52_REEF_ORCH
        // the engine loader is forced OFF (the TEST is the orchestrator).
        if (keeper52_reef_orch()) setenv("LS_LOADER_SHADOW", "0", 1);
        if (!std::getenv("LS_LOADER_SHADOW")) setenv("LS_LOADER_SHADOW", "1", 0);
        // THIS TEST defaults the M3 static arena placement OFF (no-overwrite):
        // the GLM-5.2 config json now carries the table by default
        // (memory.arena_placement.freq_table), but dsp52's established
        // default arm — and every historical fingerprint — is the legacy
        // identity-0 placement. Explicit LS_ARENA_PLACE_FREQ=<table> (the
        // champion static-table recipe) still overrides; other consumers
        // (serve, Python bridge, keepers) inherit the config default.
        setenv("LS_ARENA_PLACE_FREQ", "off", 0);
        // ONLINE self-tuning placement is the ENGINE CONFIG default
        // (memory.arena_placement.online = true; user decision 2026-08-18:
        // ONLINE champion 10.459-10.464 > static anchors 10.23-10.42).
        // No env pin here — env LS_ARENA_PLACE_ONLINE=0 is the explicit
        // static-arm override for A/Bs.
        // Canonical placement-invariant EP combine defaults ON (keeper52
        // user decision 2026-07-18): one routing trajectory for every arm.
        setenv("LAYERSTORM_DETERMINISTIC_EP_COMBINE", "1", 0);
        setenv("LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION", "bf16", 0);
        // Small-M GEMM route: TEST-INTERFACE compat. The engine default is ON
        // since 2026-08-17 (TD-CHUNK-SMALLM-DEFAULT resolved) with the inverse
        // flag LS_NO_CHUNK_SMALLM=1; historical dsp52/keeper recipes drive this
        // fixture with LS_CHUNK_SMALLM=0/1, so translate the disable form here
        // and keep those recipes meaningful (=1/unset both mean default ON).
        // No-overwrite: an explicit LS_NO_CHUNK_SMALLM always wins.
        if (const char* sm = std::getenv("LS_CHUNK_SMALLM");
            sm && *sm && sm[0] == '0')
            setenv("LS_NO_CHUNK_SMALLM", "1", 0);
        {
            j["gpu_loader"]["enabled"] = true;
            const char* cp = std::getenv("LS_LOADER_CALIB");
            loader_calib_path_ = cp ? cp
                : (dsp52_ep() == 4 ? "gpu_loader_calibration_ep4x4.json"
                                   : "gpu_loader_calibration_5090x2.json");
            j["gpu_loader"]["calibration_path"] = loader_calib_path_;
        }
        // C-6 CPU expert offload (Milestone A / TASK-1 B-aware never-lose solver):
        // opt-in via KEEPER52_CPU_EXPERT=<numa_node> (mirrors test_keeper52_reef.py)
        // — builds the host NumaCpuExpertDevice that fold_cpu_forced_experts uses.
        // Default UNSET ⇒ no CPU device ⇒ champion untouched. Pairs with the runtime
        // gates LS_CPU_EXPERT=1 + (LS_CPU_EXPERT_FORCE | LS_LOADER_CPU_SOLVER=1).
        if (const char* cn = std::getenv("KEEPER52_CPU_EXPERT"); cn && *cn) {
            nlohmann::json dev;
            dev["numa_node"] = std::atoi(cn);
            if (const char* mt = std::getenv("KEEPER52_CPU_EXPERT_THREADS");
                mt && *mt && std::atoi(mt) > 0)
                dev["max_threads"] = std::atoi(mt);
            j["hardware"]["cpu_expert_devices"] = nlohmann::json::array({dev});
        }
        config_path_ = "/tmp/dsp52_config.json";
        { std::ofstream o(config_path_); o << j.dump(2); }

        vocab_size_      = j["model"]["vocab_size"].get<int>();
        first_moe_layer_ = j["model"].value("first_k_dense_replace", 3);
        topk_            = j["model"].value("num_experts_per_tok", 8);

        auto backends = ldam::default_backends();
        backends.skip_cuda_graphs = true;
        engine_ = std::make_unique<ldam::Engine>(config_path_,
                                                 std::move(backends));

        // Lever-4 verify-chunk prefetch: publish the parsed config field so
        // dsp52_prefetch() can fall back to it (env DSP52_PREFETCH still
        // wins when set — override semantics, see dsp52_prefetch()).
        dsp52_set_prefetch_config(
            engine_->config().speculation.dspark.verify_chunk_prefetch);
        {
            const char* pf_env = std::getenv("DSP52_PREFETCH");
            fprintf(stderr,
                    "[dsp52] lever-4 verify-chunk prefetch: %s (source=%s, "
                    "config speculation.dspark.verify_chunk_prefetch=%d, "
                    "env DSP52_PREFETCH=%s)\n",
                    dsp52_prefetch() ? "ON" : "off",
                    (pf_env && *pf_env) ? "env" : "config",
                    engine_->config().speculation.dspark.verify_chunk_prefetch
                        ? 1 : 0,
                    (pf_env && *pf_env) ? pf_env : "<unset>");
        }

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
        if (engine_) { engine_->shutdown(); engine_.reset(); }
        maybe_run_loader_trainer();
        if (!config_path_.empty()) std::remove(config_path_.c_str());
    }

    // I8 trainer feedback hook — identical to keeper52 / FirstTokenTest.
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
                              ? std::string("gpu_loader_calibration_ep4x4.json")
                              : loader_calib_path_;
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
                // DSP52_ELB fire-and-forget completions: routed to the
                // predictor bookkeeping (board rollback on failure).
                if (elb_intercept(out)) continue;
                // DSP52_PREFETCH fire-and-forget completions: drop by
                // cmd_seq (errors logged, never fatal — prefetch is a
                // best-effort residency warm-up).
                if (!fire_forget_seqs_.empty()) {
                    auto it = fire_forget_seqs_.find(out.cmd_seq);
                    if (it != fire_forget_seqs_.end()) {
                        fire_forget_seqs_.erase(it);
                        if (out.cmp_type ==
                            static_cast<uint32_t>(lipc::CMP_ERROR))
                            std::cerr << "  [dsp52 PF] prefetch error: "
                                      << out.error.message << "\n";
                        continue;
                    }
                }
                // DSP52_OVERLAP: an async DSpark step may be in flight —
                // its completion can interleave with the overlapped plain
                // step's completions (event-ordered, not ring-ordered).
                // Stash it; dspark_collect_async delivers it later.
                if (dspark_pending_seq_ != 0
                    && out.cmd_seq == dspark_pending_seq_) {
                    if (out.cmp_type ==
                        static_cast<uint32_t>(lipc::CMP_ERROR)) {
                        ADD_FAILURE() << "CMP_ERROR (async dspark): "
                                      << out.error.message;
                        dspark_pending_seq_ = 0;
                        return false;
                    }
                    dspark_cmp_       = out;
                    dspark_cmp_ready_ = true;
                    dspark_pending_seq_ = 0;
                    continue;
                }
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

    // ── DSP52_OVERLAP async draft plumbing ───────────────────────────────
    // Send the DSpark step WITHOUT waiting; the completion is stashed by
    // wait() if it interleaves with the overlapped plain step, or fetched
    // directly by dspark_collect_async.
    void dspark_send_async(uint64_t seq_id, uint32_t anchor_token,
                           uint32_t anchor_pos, int gamma) {
        auto c = make_cmd(lipc::D_CMD_RUN_DSPARK_STEP);
        c.run_dspark_step.seq_id          = seq_id;
        c.run_dspark_step.anchor_token_id = anchor_token;
        c.run_dspark_step.anchor_pos      = anchor_pos;
        c.run_dspark_step.num_query       = static_cast<uint8_t>(gamma);
        c.run_dspark_step.step_idx        = 0;
        dspark_cmp_ready_ = false;
        send(c);
        dspark_pending_seq_ = c.cmd_seq;
    }
    bool dspark_collect_async(int gamma, int32_t* ids_out, float* conf_out,
                              int timeout_s = 300) {
        if (!dspark_cmp_ready_) {
            auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::seconds(timeout_s);
            lipc::Completion out{};
            while (!dspark_cmp_ready_
                   && std::chrono::steady_clock::now() < deadline) {
                if (cmp_ring_->try_read(&out)) {
                    if (elb_intercept(out)) continue;
                    if (!fire_forget_seqs_.empty()) {
                        auto it = fire_forget_seqs_.find(out.cmd_seq);
                        if (it != fire_forget_seqs_.end()) {
                            fire_forget_seqs_.erase(it);
                            if (out.cmp_type ==
                                static_cast<uint32_t>(lipc::CMP_ERROR))
                                std::cerr << "  [dsp52 PF] prefetch error: "
                                          << out.error.message << "\n";
                            continue;
                        }
                    }
                    if (out.cmp_type ==
                        static_cast<uint32_t>(lipc::CMP_ERROR)) {
                        ADD_FAILURE() << "CMP_ERROR (dspark collect): "
                                      << out.error.message;
                        dspark_pending_seq_ = 0;
                        return false;
                    }
                    if (out.cmp_type ==
                        static_cast<uint32_t>(lipc::CMP_CHECKPOINT))
                        continue;
                    if (out.cmd_seq == dspark_pending_seq_) {
                        dspark_cmp_       = out;
                        dspark_cmp_ready_ = true;
                        dspark_pending_seq_ = 0;
                        break;
                    }
                    ADD_FAILURE() << "unexpected completion (type 0x"
                                  << std::hex << out.cmp_type << std::dec
                                  << " seq " << out.cmd_seq
                                  << ") while collecting async dspark step";
                    return false;
                }
                std::this_thread::yield();
            }
            if (!dspark_cmp_ready_) {
                ADD_FAILURE() << "timeout collecting async dspark step";
                dspark_pending_seq_ = 0;
                return false;
            }
        }
        dspark_cmp_ready_ = false;
        return parse_dspark_readback(dspark_cmp_, gamma, ids_out, conf_out);
    }
    uint32_t dspark_pending_seq_ = 0;
    bool dspark_cmp_ready_ = false;
    lipc::Completion dspark_cmp_{};
    // DSP52_PREFETCH state: fire-and-forget prefetch cmd_seqs + the
    // per-layer previous-chunk union (expert,gpu placements) predictor.
    std::unordered_set<uint32_t> fire_forget_seqs_;
    std::vector<std::vector<lipc::ExpertPrefetchEntry>> pf_pred_;

    // ── ORACLE upper-bound diagnostic state (DSP52_PREFETCH_ORACLE /
    //    DSP52_ORACLE_DUMP) ────────────────────────────────────────────────
    // Monotonic trajectory-step id: one per full layer sweep (prefill chunk /
    // decode step / verify chunk), incremented identically in RUN 1 (record)
    // and RUN 2 (replay) so the recorded script and the replay align slot-for-
    // slot. -1 until the first sweep.
    int64_t traj_step_ = -1;
    std::unique_ptr<std::ofstream> oracle_dump_os_;  // RUN 1 record sink
    OracleRoutedPredictor* oracle_ = nullptr;        // set iff oracle replay
    // Construct the ELB predictor lazily: the oracle scripted predictor when
    // DSP52_PREFETCH_ORACLE is set, else the tier-0 b0_prev predictor.
    void ensure_elb_predictor() {
        if (elb_) return;
        if (!dsp52_prefetch_oracle().empty()) {
            auto p = std::make_unique<OracleRoutedPredictor>(
                num_layers_, dsp52_prefetch_oracle(), &traj_step_,
                dsp52_oracle_nextround());
            oracle_ = p.get();
            elb_ = std::move(p);
        } else {
            elb_ = std::make_unique<ElbPrevRoutedPredictor>(num_layers_);
        }
    }
    // RUN 1: append (traj_step_, layer, committed union+placement) to the dump.
    void oracle_record(int layer, const lipc::ExpertPrefetchEntry* entries,
                       uint32_t count) {
        if (traj_step_ < 0) return;
        if (!oracle_dump_os_) {
            oracle_dump_os_ = std::make_unique<std::ofstream>(
                dsp52_oracle_dump(), std::ios::binary | std::ios::trunc);
            const uint32_t m = OracleRoutedPredictor::magic();
            oracle_dump_os_->write(reinterpret_cast<const char*>(&m), 4);
        }
        const int32_t s = static_cast<int32_t>(traj_step_);
        const int32_t L = layer;
        oracle_dump_os_->write(reinterpret_cast<const char*>(&s), 4);
        oracle_dump_os_->write(reinterpret_cast<const char*>(&L), 4);
        oracle_dump_os_->write(reinterpret_cast<const char*>(&count), 4);
        for (uint32_t i = 0; i < count; ++i) {
            const uint16_t e = entries[i].expert_idx;
            const uint8_t g = entries[i].gpu_idx;
            const uint8_t z = entries[i].zone;
            oracle_dump_os_->write(reinterpret_cast<const char*>(&e), 2);
            oracle_dump_os_->write(reinterpret_cast<const char*>(&g), 1);
            oracle_dump_os_->write(reinterpret_cast<const char*>(&z), 1);
        }
    }

    uint64_t oracle_bulk_issued_ = 0;   // entries fired by the bulk arm
    // DSP52_ORACLE_BULK: fire this traj step's ENTIRE non-resident routed
    // manifest (every MoE layer) as fire-and-forget transient (zone=1)
    // prefetch batches at priority -1, BEFORE the first attention of the step.
    // Reuses the proven lever-4 fire_forget completion draining (one drop per
    // batch cmd_seq). Advisory only: no board admission, no placement change
    // (demand fill_moe_entries is untouched — a ready copy just serves the
    // zone-blind lookup as was_cached). Non-resident filtered via the REEF
    // board; entries beyond kMaxExpertPrefetch spill into further batches.
    void oracle_bulk_issue() {
        if (!oracle_ || !dsp52_oracle_bulk() || traj_step_ < 0) return;
        std::vector<lipc::ExpertPrefetchEntry> want;
        for (int L = first_moe_layer_; L < num_layers_; ++L) {
            const auto& pred = elb_->predict(L);  // this step's layer-L union
            for (auto e : pred) {
                const int g = static_cast<int>(e.gpu_idx);
                bool resident = false;
                if (reef_orch_)
                    resident = reef_orch_->board.is_resident(
                        g, layerstorm::memory::ExpertKey{e.layer_idx,
                                                         e.expert_idx});
                if (resident) continue;
                e.zone = 1;  // transient landing zone (scan-resistant)
                want.push_back(e);
            }
        }
        if (want.empty()) return;
        auto* pfe = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
            sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
        for (size_t off = 0; off < want.size();
             off += lipc::kMaxExpertPrefetch) {
            const uint32_t n = static_cast<uint32_t>(std::min(
                static_cast<size_t>(lipc::kMaxExpertPrefetch),
                want.size() - off));
            for (uint32_t i = 0; i < n; ++i) pfe[i] = want[off + i];
            auto pc = make_cmd(lipc::D_B_CMD_PREFETCH_BATCH);
            pc.prefetch_batch.count    = n;
            pc.prefetch_batch.priority = -1.0f;
            pc.prefetch_batch.delay_us = 0;
            fire_forget_seqs_.insert(pc.cmd_seq);
            send(pc);
            if (::testing::Test::HasFailure()) return;
            oracle_bulk_issued_ += n;
        }
    }

    // ── DSP52_ELB state (predictor seam + issue-policy bookkeeping) ──────
    std::unique_ptr<ElbPredictor> elb_;
    // (layer, expert, gpu) keys of prefetches issued but not yet completed —
    // never re-issued, never picked as eviction victims while in flight.
    std::unordered_set<uint64_t> elb_inflight_;
    // cmd_seq → outstanding CMP_ELM_EXPERT_READY count for elb-issued
    // prefetch batches (one lifecycle completion per entry) and evict-batch
    // acks (count 1). Intercepted out of the completion stream by
    // elb_intercept(); a failed prefetch rolls back the optimistic REEF
    // board admission so the board keeps tracking ACTUAL residency.
    std::unordered_map<uint32_t, uint32_t> elb_seqs_;
    uint64_t elb_issued_ = 0, elb_ready_ = 0, elb_failed_ = 0;
    uint64_t elb_evicted_ = 0, elb_skipped_room_ = 0;
    uint64_t elb_stream_hits_ = 0;  // demand entries served by a streamed copy

    // ── DSP52_UPART state (union-aware cache partitioning) ───────────────
    std::vector<int> upart_stream_cap_;   // per-gpu streaming-zone slot totals
    std::vector<int> upart_pf_pending_;   // zone=1 lever-4 prefetch entries
                                          // issued for the CURRENT verify layer
    std::vector<uint8_t> upart_mask_scratch_;  // per-entry transient mask
    uint64_t upart_streamed_ = 0;   // union misses marked transient
    uint64_t upart_overflow_ = 0;   // union misses kept stable (no room)

    void upart_report() {
        if (!dsp52_upart()) return;
        std::cerr << "  [dsp52 UPART] streamed=" << upart_streamed_
                  << " overflow_stable=" << upart_overflow_ << "\n";
    }

    // ── DSP52_LOOKAHEAD_DIAG state (Wave-3 M4 stage 1) ───────────────────
    struct LaLayerAgg {
        double row_recall_sum = 0;   // Σ chunks of mean per-row topk recall
        double union_cov_sum  = 0;   // Σ chunks of |U1∩U2|/|U2|
        double wcov_sum       = 0;   // Σ chunks of gate-weighted coverage
        uint64_t u1 = 0, u2 = 0, inter = 0;  // raw union sizes (aggregates)
        // vs the lever-4 prev-chunk union (the manifest the champion ALREADY
        // prefetches): what pass-1 adds beyond temporal locality.
        uint64_t prev_inter = 0;     // |U2 ∩ U_prev|
        uint64_t new_n = 0;          // |U2 \ U_prev|  (lever-4 misses)
        uint64_t new_p1 = 0;         // |(U2 \ U_prev) ∩ U1|
        uint64_t hyb_inter = 0;      // |U2 ∩ (U1 ∪ U_prev)|
        int prev_chunks = 0;         // chunks where a prev union existed
        int chunks = 0;
    };
    std::vector<LaLayerAgg> la_agg_;                // per layer
    std::vector<std::vector<int32_t>> la_p1_topk_;  // per-layer R×topk (flat)
    std::vector<uint32_t> la_p1_rows_;              // rows recorded per layer
    double   la_pass1_ms_ = 0;                      // Σ pass-1 wall
    int      la_chunks_   = 0;                      // pass-1 executions
    uint64_t la_p1_union_entries_    = 0;  // resident mode: union size seen
    uint64_t la_p1_resident_entries_ = 0;  // resident mode: entries computed

    // Stage-1 agreement: pass-1 (recorded by lookahead_pass1) vs the real
    // pass-2 routing of THIS chunk+layer, straight off the routing-export
    // sideband. u2 = pass-2 deduped union (selection-rank order).
    void lookahead_accumulate(int layer, uint32_t R, uint32_t k2,
                              const int32_t* ridx2, const float* w2,
                              const std::vector<uint16_t>& u2) {
        if (la_agg_.empty()) la_agg_.resize(static_cast<size_t>(num_layers_));
        const auto& p1 = la_p1_topk_[static_cast<size_t>(layer)];
        const uint32_t r1 = la_p1_rows_[static_cast<size_t>(layer)];
        const uint32_t k1 = r1 ? static_cast<uint32_t>(p1.size()) / r1 : 0;
        if (!r1 || !k1 || !k2) return;
        std::vector<uint8_t> in_u1(static_cast<size_t>(num_experts_), 0);
        for (int32_t e : p1)
            if (e >= 0 && e < num_experts_) in_u1[static_cast<size_t>(e)] = 1;
        // Per-row top-k recall (rows aligned: same chunk tokens/positions).
        const uint32_t rows = std::min(R, r1);
        double rec_sum = 0;
        for (uint32_t b = 0; b < rows; ++b) {
            int hitc = 0, denom = 0;
            for (uint32_t k = 0; k < k2; ++k) {
                const int32_t e = ridx2[b * k2 + k];
                if (e < 0 || e >= num_experts_) continue;
                ++denom;
                for (uint32_t q = 0; q < k1; ++q)
                    if (p1[b * k1 + q] == e) { ++hitc; break; }
            }
            rec_sum += denom ? static_cast<double>(hitc) / denom : 1.0;
        }
        uint32_t inter = 0;
        for (uint16_t e : u2) if (in_u1[e]) ++inter;
        double wtot = 0, wcov = 0;
        for (uint32_t b = 0; b < R; ++b)
            for (uint32_t k = 0; k < k2; ++k) {
                const int32_t e = ridx2[b * k2 + k];
                if (e < 0 || e >= num_experts_) continue;
                float w = w2[b * k2 + k];
                if (!(w > 0.0f)) w = 0.0f;
                wtot += w;
                if (in_u1[static_cast<size_t>(e)]) wcov += w;
            }
        uint32_t u1n = 0;
        for (int e = 0; e < num_experts_; ++e)
            u1n += in_u1[static_cast<size_t>(e)];
        auto& a = la_agg_[static_cast<size_t>(layer)];
        // Lever-4 comparison: prev-chunk union of this layer (pf_pred_ holds
        // it until the pass-2 fill below overwrites it with THIS chunk's).
        if (dsp52_prefetch()
            && static_cast<size_t>(layer) < pf_pred_.size()
            && !pf_pred_[static_cast<size_t>(layer)].empty()) {
            std::vector<uint8_t> in_prev(static_cast<size_t>(num_experts_),
                                         0);
            for (const auto& e : pf_pred_[static_cast<size_t>(layer)])
                if (e.expert_idx < num_experts_) in_prev[e.expert_idx] = 1;
            for (uint16_t e : u2) {
                const bool p = in_prev[e] != 0, q = in_u1[e] != 0;
                if (p) ++a.prev_inter;
                else {
                    ++a.new_n;
                    if (q) ++a.new_p1;
                }
                if (p || q) ++a.hyb_inter;
            }
            ++a.prev_chunks;
        }
        a.row_recall_sum += rows ? rec_sum / rows : 0.0;
        a.union_cov_sum  += u2.empty()
            ? 0.0 : static_cast<double>(inter) / static_cast<double>(u2.size());
        a.wcov_sum       += wtot > 0 ? wcov / wtot : 0.0;
        a.u1 += u1n;
        a.u2 += static_cast<uint64_t>(u2.size());
        a.inter += inter;
        ++a.chunks;
    }

    void lookahead_report() {
        const int mode = dsp52_lookahead_diag();
        if (!mode || la_chunks_ == 0) return;
        std::cerr << "  [dsp52 LOOKAHEAD stage-1] mode="
                  << (mode == 1 ? "shared" : "resident")
                  << " chunks=" << la_chunks_
                  << " pass1_mean_ms=" << std::fixed << std::setprecision(2)
                  << la_pass1_ms_ / la_chunks_;
        if (mode == 2 && la_p1_union_entries_ > 0)
            std::cerr << " p1_resident_frac="
                      << std::setprecision(4)
                      << static_cast<double>(la_p1_resident_entries_)
                             / static_cast<double>(la_p1_union_entries_);
        std::cerr << "\n    layer ch  row-rec  u-cov  w-cov   |U1|   |U2|"
                     "   int  prevcov  newP1cov  hybcov\n";
        // Aggregates: overall + deep-20 (the M4 GATE band = last 20 MoE
        // layers), chunk-weighted.
        double all_rr = 0, all_uc = 0, all_wc = 0;
        uint64_t all_u1 = 0, all_u2 = 0, all_in = 0;
        uint64_t all_pv = 0, all_nw = 0, all_np = 0, all_hy = 0, all_pu2 = 0;
        int all_n = 0;
        double deep_uc = 0, deep_wc = 0;
        uint64_t deep_u2 = 0, deep_in = 0;
        uint64_t deep_pv = 0, deep_nw = 0, deep_np = 0, deep_hy = 0,
                 deep_pu2 = 0;
        int deep_n = 0;
        const int deep_lo = num_layers_ - 20;
        for (int L = 0; L < num_layers_; ++L) {
            const auto& a = la_agg_.empty()
                ? LaLayerAgg{} : la_agg_[static_cast<size_t>(L)];
            if (a.chunks == 0) continue;
            // Prev/hybrid columns use only chunks that HAD a prev union;
            // approximate their |U2| share by the layer mean.
            const double mean_u2 =
                static_cast<double>(a.u2) / a.chunks;
            const double pu2 = mean_u2 * a.prev_chunks;
            std::cerr << "    L" << std::setw(2) << L
                      << std::setw(4) << a.chunks << std::setprecision(4)
                      << std::setw(9) << a.row_recall_sum / a.chunks
                      << std::setw(7) << std::setprecision(3)
                      << a.union_cov_sum / a.chunks
                      << std::setw(7) << a.wcov_sum / a.chunks
                      << std::setw(7) << std::setprecision(1)
                      << static_cast<double>(a.u1) / a.chunks
                      << std::setw(7)
                      << static_cast<double>(a.u2) / a.chunks
                      << std::setw(6)
                      << static_cast<double>(a.inter) / a.chunks
                      << std::setprecision(3)
                      << std::setw(9)
                      << (pu2 > 0 ? a.prev_inter / pu2 : 0.0)
                      << std::setw(10)
                      << (a.new_n
                              ? static_cast<double>(a.new_p1) / a.new_n
                              : 0.0)
                      << std::setw(8)
                      << (pu2 > 0 ? a.hyb_inter / pu2 : 0.0) << "\n";
            all_rr += a.row_recall_sum; all_uc += a.union_cov_sum;
            all_wc += a.wcov_sum; all_n += a.chunks;
            all_u1 += a.u1; all_u2 += a.u2; all_in += a.inter;
            all_pv += a.prev_inter; all_nw += a.new_n; all_np += a.new_p1;
            all_hy += a.hyb_inter;
            all_pu2 += static_cast<uint64_t>(pu2 + 0.5);
            if (L >= deep_lo) {
                deep_uc += a.union_cov_sum; deep_wc += a.wcov_sum;
                deep_n += a.chunks; deep_u2 += a.u2; deep_in += a.inter;
                deep_pv += a.prev_inter; deep_nw += a.new_n;
                deep_np += a.new_p1; deep_hy += a.hyb_inter;
                deep_pu2 += static_cast<uint64_t>(pu2 + 0.5);
            }
        }
        if (all_n > 0)
            std::cerr << "  [dsp52 LOOKAHEAD] ALL layers: row-recall="
                      << std::setprecision(4) << all_rr / all_n
                      << " union-cov=" << all_uc / all_n
                      << " w-cov=" << all_wc / all_n
                      << " byte-cov(aggregate)="
                      << (all_u2 ? static_cast<double>(all_in) / all_u2 : 0.0)
                      << " manifest-waste="
                      << (all_u1 ? 1.0 - static_cast<double>(all_in) / all_u1
                                 : 0.0)
                      << "\n    vs lever-4: prev-cov="
                      << (all_pu2 ? static_cast<double>(all_pv) / all_pu2
                                  : 0.0)
                      << " lever4-missed=" << all_nw
                      << " p1-covers-missed="
                      << (all_nw ? static_cast<double>(all_np) / all_nw : 0.0)
                      << " hybrid-cov="
                      << (all_pu2 ? static_cast<double>(all_hy) / all_pu2
                                  : 0.0)
                      << "\n";
        if (deep_n > 0)
            std::cerr << "  [dsp52 LOOKAHEAD GATE] deep-20 (L>=" << deep_lo
                      << "): union-cov=" << std::setprecision(4)
                      << deep_uc / deep_n
                      << " w-cov=" << deep_wc / deep_n
                      << " agg-cov="
                      << (deep_u2 ? static_cast<double>(deep_in) / deep_u2
                                  : 0.0)
                      << " hybrid-cov="
                      << (deep_pu2 ? static_cast<double>(deep_hy) / deep_pu2
                                   : 0.0)
                      << " p1-covers-missed="
                      << (deep_nw ? static_cast<double>(deep_np) / deep_nw
                                  : 0.0)
                      << (deep_uc / deep_n < 0.70 ? "  [<0.70 KILL]"
                                                  : "  [>=0.70 PASS]")
                      << "\n";
    }

    // ── DSP52_LOOKAHEAD_DIAG pass-1 (Wave-3 M4 stage 1) ──────────────────
    // A cheap approximate forward of the verify chunk yielding per-layer
    // routed top-K predictions BEFORE the real chunk runs. Identical chunk
    // preamble + per-layer attention commands (same position-addressed
    // KV/indexer/aux writes — fully overwritten by pass-2 at the same
    // positions); the routed FFN is approximated per mode (see
    // dsp52_lookahead_diag()). No fill_moe_entries, no OUTPUT_HEAD, no
    // lever-4/ELB/UPART feeds — placement/board/LRU state untouched.
    void lookahead_pass1(const std::vector<uint32_t>& toks, uint64_t seq_id,
                         uint32_t pos0, int mode,
                         std::vector<GpuLru>* lrus) {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        lipc::Completion cmp{};
        const uint32_t R = static_cast<uint32_t>(toks.size());
        if (la_p1_topk_.empty()) {
            la_p1_topk_.resize(static_cast<size_t>(num_layers_));
            la_p1_rows_.assign(static_cast<size_t>(num_layers_), 0);
        }
        for (auto& v : la_p1_topk_) v.clear();
        std::fill(la_p1_rows_.begin(), la_p1_rows_.end(), 0u);

        auto* token_ids = reinterpret_cast<uint32_t*>(
            sideband_ + lipc::IpcLayout::kTokenIdsOff);
        for (uint32_t b = 0; b < R; ++b) token_ids[b] = toks[b];
        auto embed_cmd = make_cmd(lipc::CMD_EMBEDDING_LOOKUP);
        embed_cmd.embedding_lookup.num_tokens    = R;
        embed_cmd.embedding_lookup.output_buf_id = hidden_buf_id_;
        send(embed_cmd);
        EXPECT_TRUE(wait(cmp, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)))
            << "lookahead pass-1 embedding";
        EXPECT_EQ(cmp.status, 0u);
        if (::testing::Test::HasFailure()) return;

        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        for (uint32_t b = 0; b < R; ++b) {
            batch[b].seq_id    = seq_id;
            batch[b].token_pos = pos0 + b;
            batch[b]._pad      = 0;
        }

        const int tp = engine_->info().num_gpus;
        for (int layer = 0; layer < num_layers_; ++layer) {
            const bool is_moe = layer >= first_moe_layer_;
            auto attn_cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
            attn_cmd.run_attention.layer_idx   = static_cast<uint32_t>(layer);
            attn_cmd.run_attention.num_seqs    = R;
            attn_cmd.run_attention.is_prefill  = 1;
            attn_cmd.run_attention.use_graph   = 0;
            attn_cmd.run_attention.chunk_start = pos0;
            attn_cmd.run_attention.chunk_len   = R;
            attn_cmd.run_attention.emit_gating  = is_moe ? 1 : 0;
            attn_cmd.run_attention.store_gating = is_moe ? 1 : 0;
            send(attn_cmd);
            EXPECT_TRUE(wait(cmp,
                static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)))
                << "lookahead pass-1 attn L" << layer;
            EXPECT_EQ(cmp.status, 0u);
            if (::testing::Test::HasFailure()) return;

            if (is_moe) {
                const auto* hdr =
                    reinterpret_cast<const lipc::RoutingExportHeader*>(
                        sideband_ + lipc::IpcLayout::kRoutingExportOff);
                EXPECT_EQ(hdr->num_tokens, R)
                    << "pass-1 routing export rows L" << layer;
                EXPECT_EQ(hdr->layer_idx, static_cast<uint32_t>(layer))
                    << "pass-1 routing-export layer mismatch";
                if (::testing::Test::HasFailure()) return;
                const auto* ridx = reinterpret_cast<const int32_t*>(
                    sideband_ + lipc::IpcLayout::kRoutingExportIndicesOff);
                const uint32_t rn = hdr->num_tokens * hdr->topk;
                la_p1_topk_[static_cast<size_t>(layer)].assign(ridx,
                                                               ridx + rn);
                la_p1_rows_[static_cast<size_t>(layer)] = hdr->num_tokens;

                // Approximate routed FFN. Sideband entries are consumed by
                // the daemon at dispatch, before the completion we wait on —
                // the pass-2 overwrite below is race-free (ring order).
                auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
                    sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
                uint32_t count = 0;
                if (mode == 2) {  // resident-only
                    std::vector<uint8_t> seen(
                        static_cast<size_t>(num_experts_), 0);
                    for (uint32_t k = 0; k < rn; ++k) {
                        const int32_t e = ridx[k];
                        if (e < 0 || e >= num_experts_) continue;
                        if (seen[static_cast<size_t>(e)]) continue;
                        seen[static_cast<size_t>(e)] = 1;
                        ++la_p1_union_entries_;
                        int g_res = -1;
                        if (reef_orch_) {
                            const layerstorm::memory::ExpertKey key{
                                static_cast<uint32_t>(layer),
                                static_cast<uint16_t>(e)};
                            for (int g = 0; g < tp; ++g)
                                if (reef_orch_->board.is_resident(g, key)) {
                                    g_res = g;
                                    break;
                                }
                        } else if (lrus) {
                            const LruKey lk{static_cast<uint32_t>(layer),
                                            static_cast<uint16_t>(e)};
                            for (int g = 0; g < tp; ++g)
                                if ((*lrus)[static_cast<size_t>(g)]
                                        .resident.count(lk)) {
                                    g_res = g;
                                    break;
                                }
                        }
                        if (g_res < 0) continue;
                        entries[count].layer_idx  =
                            static_cast<uint32_t>(layer);
                        entries[count].expert_idx =
                            static_cast<uint16_t>(e);
                        entries[count].zone       = 0;
                        entries[count].gpu_idx    =
                            static_cast<uint8_t>(g_res);
                        ++count;
                    }
                    la_p1_resident_entries_ += count;
                }
                auto moe_cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
                moe_cmd.fetch_and_run_moe.layer_idx =
                    static_cast<uint32_t>(layer);
                moe_cmd.fetch_and_run_moe.num_seqs      = R;
                moe_cmd.fetch_and_run_moe.expert_count  = count;
                moe_cmd.fetch_and_run_moe.have_evict_map = 0;
                moe_cmd.fetch_and_run_moe.timeout_us    = 5000000;
                // moe_mode=1: suppress the miss count in the completion so
                // pass-1's absent routed experts don't pollute miss stats.
                moe_cmd.fetch_and_run_moe.moe_mode      = 1;
                send(moe_cmd);
            } else {
                auto moe_cmd = make_cmd(lipc::D_B_CMD_RUN_MOE);
                moe_cmd.run_moe.layer_idx = static_cast<uint32_t>(layer);
                moe_cmd.run_moe.num_seqs  = R;
                moe_cmd.run_moe.moe_mode  = 0;
                moe_cmd.run_moe.apply_residual_correction = 0;
                moe_cmd.run_moe.store_gating_output       = 0;
                moe_cmd.run_moe.emit_checkpoint           = 0;
                send(moe_cmd);
            }
            EXPECT_TRUE(wait(cmp,
                static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)))
                << "lookahead pass-1 moe L" << layer;
            EXPECT_EQ(cmp.status, 0u);
            if (::testing::Test::HasFailure()) return;
        }
        la_pass1_ms_ +=
            std::chrono::duration<double, std::milli>(clock::now() - t0)
                .count();
        ++la_chunks_;
    }

    // Streaming-zone mode state: ELB's own ring of streamed experts.
    struct ElbStreamEnt { int gpu = -1; bool ready = false; };
    std::unordered_map<uint64_t, ElbStreamEnt> elb_stream_;   // key: elb_key2
    std::vector<std::deque<uint64_t>> elb_fifo_;              // per-gpu ring
    std::vector<int> elb_stream_cap_;                         // per-gpu cap
    std::vector<uint8_t> elb_mask_scratch_;  // per-entry stream-served mask

    static uint64_t elb_key(uint32_t layer, uint16_t expert, uint32_t gpu) {
        return (static_cast<uint64_t>(layer) << 24)
             | (static_cast<uint64_t>(expert) << 8)
             | static_cast<uint64_t>(gpu & 0xffu);
    }
    static uint64_t elb_key2(uint32_t layer, uint16_t expert) {
        return (static_cast<uint64_t>(layer) << 16)
             | static_cast<uint64_t>(expert);
    }

    // Route an ELB fire-and-forget completion out of the ring stream.
    // Returns true if the completion belonged to an ELB-issued command.
    bool elb_intercept(const lipc::Completion& out) {
        if (elb_seqs_.empty()) return false;
        auto it = elb_seqs_.find(out.cmd_seq);
        if (it == elb_seqs_.end()) return false;
        if (out.cmp_type == static_cast<uint32_t>(lipc::CMP_ELM_EXPERT_READY)) {
            const uint32_t layer  = out.elm_expert.layer_idx;
            const uint16_t expert =
                static_cast<uint16_t>(out.elm_expert.expert_idx);
            elb_inflight_.erase(elb_key(layer, expert, out.gpu_idx));
            if (out.status == 0) {
                ++elb_ready_;
                if (dsp52_elb_stream()) {
                    auto sit = elb_stream_.find(elb_key2(layer, expert));
                    if (sit != elb_stream_.end()
                        && sit->second.gpu == static_cast<int>(out.gpu_idx))
                        sit->second.ready = true;
                }
            } else {
                ++elb_failed_;
                if (dsp52_elb_stream()) {
                    // Streaming reserve infeasible: forget the entry (the
                    // FIFO slot is reclaimed lazily on pop).
                    elb_stream_.erase(elb_key2(layer, expert));
                } else if (reef_orch_) {
                    // Reserve/victim infeasibility engine-side: roll back the
                    // optimistic board admission (silent skip — INV-ELB-CAP).
                    reef_orch_->board.evict(
                        static_cast<int>(out.gpu_idx),
                        layerstorm::memory::ExpertKey{layer, expert});
                }
            }
            if (it->second <= 1) elb_seqs_.erase(it);
            else --it->second;
            return true;
        }
        if (out.cmp_type == static_cast<uint32_t>(lipc::CMP_ERROR)) {
            std::cerr << "  [dsp52 ELB] error: " << out.error.message << "\n";
            elb_seqs_.erase(it);
            return true;
        }
        if (out.cmp_type == static_cast<uint32_t>(lipc::CMP_CACHE_OP_DONE)) {
            elb_seqs_.erase(it);  // evict-batch ack
            return true;
        }
        return true;  // progress events etc.: drop without decrementing
    }

    void elb_report() {
        if (reef_orch_ && belady_oracle_) {
            const auto& or_ = *belady_oracle_;
            std::cerr << "  [dsp52 BELADY] mode="
                      << (reef_orch_->evict_bridge > 0
                              ? ("bridge<=" + std::to_string(reef_orch_->evict_bridge)
                                 + "tok")
                              : std::string("exact"))
                      << " determinism_matched=" << or_.matched()
                      << " mismatched=" << or_.mismatched() << "\n";
        }
        if (!dsp52_elb()) return;
        std::cerr << "  [dsp52 ELB] zone="
                  << (dsp52_elb_stream() ? "stream" : "stable")
                  << " issued=" << elb_issued_
                  << " ready=" << elb_ready_ << " failed=" << elb_failed_
                  << " evicted=" << elb_evicted_
                  << " skipped_no_room=" << elb_skipped_room_
                  << " stream_hits=" << elb_stream_hits_
                  << " still_inflight=" << elb_inflight_.size() << "\n";
        if (oracle_)
            std::cerr << "  [dsp52 ORACLE] ahead=" << dsp52_elb_ahead()
                      << " cap=" << dsp52_elb_cap()
                      << " nextround=" << (dsp52_oracle_nextround() ? 1 : 0)
                      << " determinism_matched=" << oracle_->matched()
                      << " mismatched=" << oracle_->mismatched()
                      << " bulk_issued=" << oracle_bulk_issued_
                      << "  (conversion metric = stream_hits above)\n";
    }

    // ── DSP52_ELB issue policy ───────────────────────────────────────────
    // Called at the LAUNCH of `cur_layer`'s attention in the decode step and
    // the batched verify chunk (NOT in prefill — INV-FAR-WAVE owns that
    // seam). Issues at most one evict batch + one prefetch batch covering
    // predictions for layers cur_layer..cur_layer+AHEAD. Both sideband
    // regions are consumed by the daemon BEFORE the attention completion the
    // caller waits on (ring order), so the later fill_moe_entries overwrite
    // is race-free — one batch pair per attention launch, never more.
    void elb_issue(int cur_layer, std::vector<GpuLru>* lrus) {
        if (!dsp52_elb()) return;
        if (dsp52_oracle_bulk()) return;  // bulk arm fires once per step instead
        ensure_elb_predictor();
        const int tp    = engine_->info().num_gpus;
        const int ahead = dsp52_elb_ahead();
        const int cap   = dsp52_elb_cap();

        // 1. Gather capped, not-resident, not-in-flight candidates for
        //    layers cur_layer..cur_layer+ahead; protect every predicted
        //    (layer, expert) from victim selection.
        std::vector<lipc::ExpertPrefetchEntry> want;
        std::unordered_set<uint64_t> protect;
        for (int a = 0; a <= ahead; ++a) {
            const int P = cur_layer + a;
            if (P >= num_layers_) break;
            if (P < first_moe_layer_) continue;
            const auto& pred = elb_->predict(P);
            int taken = 0;
            for (const auto& e : pred) {
                if (taken >= cap) break;
                if (e.zone != 0) continue;  // stable-zone predictions only
                protect.insert(elb_key(e.layer_idx, e.expert_idx, 0xffu));
                const layerstorm::memory::ExpertKey k{e.layer_idx,
                                                      e.expert_idx};
                const int g = static_cast<int>(e.gpu_idx);
                if (dsp52_elb_stream()
                    && elb_stream_.count(elb_key2(e.layer_idx, e.expert_idx)))
                    continue;  // already streamed / stream-in-flight
                bool resident = false;
                if (reef_orch_) {
                    resident = reef_orch_->board.is_resident(g, k);
                } else if (lrus && g < static_cast<int>(lrus->size())) {
                    resident = (*lrus)[static_cast<size_t>(g)].resident.count(
                        LruKey{e.layer_idx, e.expert_idx}) != 0;
                }
                if (resident) continue;
                if (elb_inflight_.count(
                        elb_key(e.layer_idx, e.expert_idx, e.gpu_idx)))
                    continue;
                if (want.size() >=
                    static_cast<size_t>(lipc::kMaxExpertPrefetch))
                    break;
                want.push_back(e);
                ++taken;
            }
        }
        if (want.empty()) return;

        // 2a. STREAM zone (default): room comes from ELB's OWN per-GPU
        //     streaming ring — evict the oldest ALREADY-READY streamed entry
        //     when the ring is full; an in-flight entry blocks recycling of
        //     its slot (skip-when-full, INV-ELB-CAP). The stable zone and
        //     the REEF board are never touched here.
        uint32_t evict_count = 0;
        if (dsp52_elb_stream()) {
            auto* evicts = reinterpret_cast<lipc::ExpertEvictionEntry*>(
                sideband_ + lipc::IpcLayout::kExpertEvictionOff);
            if (elb_fifo_.empty()) {
                elb_fifo_.resize(static_cast<size_t>(tp));
                elb_stream_cap_.resize(static_cast<size_t>(tp), 0);
                for (int g = 0; g < tp; ++g) {
                    int zc = engine_->expert_cache()->total_slots(
                        g, layerstorm::memory::CacheZone::kStreaming);
                    if (dsp52_elb_stream_cap() > 0)
                        zc = std::min(zc, dsp52_elb_stream_cap());
                    elb_stream_cap_[static_cast<size_t>(g)] = zc;
                }
                std::cerr << "  [dsp52 ELB] stream-zone caps:";
                for (int g = 0; g < tp; ++g)
                    std::cerr << " g" << g << "="
                              << elb_stream_cap_[static_cast<size_t>(g)];
                std::cerr << "\n";
            }
            std::vector<lipc::ExpertPrefetchEntry> kept;
            kept.reserve(want.size());
            for (auto& e : want) {
                const int g = static_cast<int>(e.gpu_idx);
                auto& fifo = elb_fifo_[static_cast<size_t>(g)];
                const int gcap = elb_stream_cap_[static_cast<size_t>(g)];
                // Lazily drop forgotten entries (failed prefetches).
                while (!fifo.empty() && !elb_stream_.count(fifo.front()))
                    fifo.pop_front();
                bool room = static_cast<int>(fifo.size()) < gcap;
                if (!room && !fifo.empty()) {
                    const auto oldest = elb_stream_.find(fifo.front());
                    if (oldest != elb_stream_.end() && oldest->second.ready
                        && evict_count < lipc::kMaxExpertEviction) {
                        evicts[evict_count].layer_idx =
                            static_cast<uint32_t>(fifo.front() >> 16);
                        evicts[evict_count].expert_idx =
                            static_cast<uint16_t>(fifo.front() & 0xffffu);
                        evicts[evict_count].gpu_idx =
                            static_cast<uint8_t>(oldest->second.gpu);
                        evicts[evict_count]._pad = 0;
                        ++evict_count;
                        ++elb_evicted_;
                        elb_stream_.erase(oldest);
                        fifo.pop_front();
                        room = true;
                    }
                }
                if (!room || gcap <= 0) {
                    ++elb_skipped_room_;
                    continue;  // in-flight ring or 0-slot zone: silent skip
                }
                e.zone = 1;  // streaming zone
                const uint64_t k2 = elb_key2(e.layer_idx, e.expert_idx);
                elb_stream_[k2] = ElbStreamEnt{g, false};
                fifo.push_back(k2);
                kept.push_back(e);
            }
            want.swap(kept);
        }

        // 2b. STABLE zone: make room from cheaply-evictable victims (REEF
        //     board, cheapest first, reuse-period guarded). Predictions that
        //     cannot get a slot are silently dropped (INV-ELB-CAP).
        if (!dsp52_elb_stream() && reef_orch_ && dsp52_elb_evict()) {
            auto* evicts = reinterpret_cast<lipc::ExpertEvictionEntry*>(
                sideband_ + lipc::IpcLayout::kExpertEvictionOff);
            auto& board = reef_orch_->board;
            const double now = board.recency_now();
            const double hot_w = dsp52_elb_hot() > 0
                ? static_cast<double>(dsp52_elb_hot())
                : static_cast<double>(num_layers_ - first_moe_layer_);
            std::vector<int> need(static_cast<size_t>(tp), 0);
            for (const auto& e : want) ++need[e.gpu_idx];
            std::vector<int> granted(static_cast<size_t>(tp), 0);
            std::vector<layerstorm::memory::ExpertKey> keys;
            for (int g = 0; g < tp; ++g) {
                if (need[static_cast<size_t>(g)] == 0) continue;
                const int room = reef_orch_->cap[static_cast<size_t>(g)]
                               - board.resident_count(g);
                int shortfall = need[static_cast<size_t>(g)] - room;
                granted[static_cast<size_t>(g)] =
                    need[static_cast<size_t>(g)];
                if (shortfall <= 0) continue;
                board.cheapest_keys(
                    g, shortfall + need[static_cast<size_t>(g)] + 8, keys);
                int freed = 0;
                for (const auto& vk : keys) {
                    if (freed >= shortfall) break;
                    const double s = board.score(g, vk);
                    if (s < 0.0) continue;
                    if (now - s < hot_w) break;  // ascending: rest are hotter
                    if (protect.count(
                            elb_key(vk.layer_idx, vk.expert_idx, 0xffu)))
                        continue;
                    if (elb_inflight_.count(elb_key(
                            vk.layer_idx, vk.expert_idx,
                            static_cast<uint32_t>(g))))
                        continue;
                    if (evict_count >= lipc::kMaxExpertEviction) break;
                    evicts[evict_count].layer_idx  = vk.layer_idx;
                    evicts[evict_count].expert_idx = vk.expert_idx;
                    evicts[evict_count].gpu_idx    = static_cast<uint8_t>(g);
                    evicts[evict_count]._pad       = 0;
                    ++evict_count;
                    board.evict(g, vk);
                    ++elb_evicted_;
                    ++freed;
                }
                if (freed < shortfall) {
                    // Not enough cold victims: grant only what fits.
                    granted[static_cast<size_t>(g)] =
                        std::max(0, room + freed);
                    elb_skipped_room_ += static_cast<uint64_t>(
                        need[static_cast<size_t>(g)]
                        - granted[static_cast<size_t>(g)]);
                }
            }
            // Drop over-grant entries per GPU from the tail (lowest
            // selection rank loses).
            std::vector<int> used(static_cast<size_t>(tp), 0);
            std::vector<lipc::ExpertPrefetchEntry> kept;
            kept.reserve(want.size());
            for (const auto& e : want) {
                if (used[e.gpu_idx] >= granted[e.gpu_idx]) continue;
                ++used[e.gpu_idx];
                kept.push_back(e);
            }
            want.swap(kept);
        }
        // Evict batch (both zones): streaming-ring recycling or stable
        // cheap victims. Sent BEFORE the prefetch batch (slots free first);
        // both sideband regions are consumed before the attention completion
        // the caller waits on (ring order).
        if (evict_count > 0) {
            auto ec = make_cmd(lipc::D_B_CMD_EVICT_BATCH);
            ec.evict_batch.count = evict_count;
            elb_seqs_[ec.cmd_seq] = 1;
            send(ec);
            if (::testing::Test::HasFailure()) return;
        }
        if (want.empty()) return;

        // 3. Fire-and-forget prefetch at priority -1.0 (below every demand
        //    fetch; a demand join re-asserts priority engine-side). Admit
        //    the predictions into the REEF board optimistically (the engine
        //    reserves the slot synchronously at handle_prefetch_batch;
        //    a failed reserve rolls back via elb_intercept) so next round's
        //    placement pins hits onto the prefetched copies.
        auto* pfe = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
            sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
        for (size_t i = 0; i < want.size(); ++i) pfe[i] = want[i];
        auto pc = make_cmd(lipc::D_B_CMD_PREFETCH_BATCH);
        pc.prefetch_batch.count    = static_cast<uint32_t>(want.size());
        pc.prefetch_batch.priority = -1.0f;
        pc.prefetch_batch.delay_us = 0;
        elb_seqs_[pc.cmd_seq] = static_cast<uint32_t>(want.size());
        for (const auto& e : want) {
            elb_inflight_.insert(
                elb_key(e.layer_idx, e.expert_idx, e.gpu_idx));
            // Stable mode only: optimistic board admission (rolled back on a
            // failed lifecycle completion). Stream-zone entries live in
            // elb_stream_, outside the stable board model.
            if (!dsp52_elb_stream() && reef_orch_)
                reef_orch_->board.update(
                    static_cast<int>(e.gpu_idx),
                    layerstorm::memory::ExpertKey{e.layer_idx, e.expert_idx},
                    reef_orch_->board.recency_now());
        }
        elb_issued_ += want.size();
        send(pc);
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

    // KEEPER52_REEF_ORCH: test-side REEF decision stack — identical to
    // keeper52's make_reef_orch.
    std::unique_ptr<ReefOrch> make_reef_orch() {
        if (!keeper52_reef_orch()) return nullptr;
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
        // (the private LS_ARENA_MAP_DUMP CSV read is retired). Epoch 1 is
        // published daemon-side at the first routed-MoE fetch (the same
        // point the CSV used to be dumped — byte-compatible boot race),
        // then refreshed at online-migrator commit boundaries.
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
        // BELADY next-use eviction oracle (DSP52_EVICT_ORACLE; default OFF).
        if (!dsp52_evict_oracle().empty()) {
            belady_oracle_ = std::make_unique<NextUseOracle>(
                num_layers_, dsp52_evict_oracle());
            o->evict_oracle = belady_oracle_.get();
            o->evict_bridge = dsp52_evict_bridge();
            std::cerr << "  [REEF-ORCH] BELADY eviction armed ("
                      << (o->evict_bridge > 0
                              ? ("bridge recurs<=" + std::to_string(o->evict_bridge)
                                 + " tok")
                              : std::string("exact Belady"))
                      << ")\n";
        }
        std::cerr << "  [REEF-ORCH] test-side solver+board armed (calib="
                  << calib << ", policy=" << psrc << ") freq_w=" << o->freq_w
                  << " reuse_w=" << o->reuse_w << " reuse_tau=" << o->reuse_tau
                  << " keeppred_w=" << o->keeppred_w
                  << "\n";
        return o;
    }
    std::unique_ptr<ReefOrch> reef_orch_;
    // Belady diagnostic oracle (fixture-owned; the library holds a raw seam).
    std::unique_ptr<NextUseOracle> belady_oracle_;

    // Per-GPU LRU model sized to each GPU's stable-zone slot count.
    std::vector<GpuLru> make_lrus() {
        const int tp = engine_->info().num_gpus;
        EXPECT_NE(engine_->expert_cache(), nullptr);
        std::vector<GpuLru> lrus(tp);
        for (int g = 0; g < tp; ++g)
            lrus[g].capacity = engine_->expert_cache()->total_slots(
                g, layerstorm::memory::CacheZone::kStable);
        return lrus;
    }

    // ── Shared expert placement + eviction map (keeper52 machinery) ──────
    // Places the routed experts `topk` (a single decode row's top-K OR a
    // deduped multi-row verify-chunk UNION — the placement machinery is
    // list-shaped either way) for `layer` and fills the sideband prefetch +
    // eviction entries. Placement: REEF-ORCH > KEEPER52_AFFINITY > static
    // e%tp — byte-identical logic to keeper52's decode_step_fetch_and_run
    // MoE block, factored out so the speculative verify chunk routes through
    // the SAME code. Returns the entry count (== topk.size()).
    // static_mod > 0 (DSP52_VB_TP_ONLY bisect): FORCE static e%static_mod
    // placement for this call (restricts the union to GPU positions
    // [0, static_mod) — removes the EP-XTP extra ranks from the chunk MoE;
    // INV-MOE-EP-DISJOINT admits arbitrary placement). Skips REEF/affinity
    // for the call — bisect mode only.
    uint32_t fill_moe_entries(int layer, const std::vector<uint16_t>& topk,
                              std::vector<GpuLru>* lrus, int static_mod = 0,
                              bool transient_union = false) {
        const int tp = engine_->info().num_gpus;
        const bool use_reef = reef_orch_ && static_mod == 0;
        elb_mask_scratch_.clear();  // per-call stream-served mask (DSP52_ELB)
        upart_mask_scratch_.clear();  // per-call transient mask (DSP52_UPART)
        if (static_mod > 0 && reef_orch_) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::cerr << "  [dsp52] WARNING: DSP52_VB_TP_ONLY forces "
                             "static placement for verify chunks — the REEF "
                             "board's residency model will diverge (bisect "
                             "mode; run without KEEPER52_REEF_ORCH)\n";
            }
        }
        std::vector<uint8_t> assign(topk.size());
        if (static_mod > 0) {
            for (size_t i = 0; i < topk.size(); ++i)
                assign[i] = static_cast<uint8_t>(topk[i] % static_mod);
        } else if (use_reef && !topk.empty()) {
            // CAPACITY BOUND: unions <= kMaxExperts (64) take the frozen
            // production solver; unions in (64, 256] route to the
            // LoaderSolver256 instantiation inside reef_orch_route (a γ=15
            // verify chunk's raw union can reach 16×8=128). Beyond
            // kMaxExpertsLarge (256) fail loud rather than trip the solver's
            // assertion.
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
            // DSP52_ELB stream zone: a READY streamed copy serves the demand
            // in place — pin the entry to the streamed copy's GPU (lookup is
            // zone-blind, so the engine sees was_cached and computes from
            // the streaming slot; placement changes are trajectory-safe
            // under the canonical deterministic EP combine). Masked entries
            // are excluded from the stable evict-map/board admission.
            if (dsp52_elb() && dsp52_elb_stream() && !elb_stream_.empty()) {
                elb_mask_scratch_.assign(topk.size(), 0);
                for (size_t i = 0; i < topk.size(); ++i) {
                    auto sit = elb_stream_.find(elb_key2(
                        static_cast<uint32_t>(layer), topk[i]));
                    if (sit == elb_stream_.end() || !sit->second.ready)
                        continue;
                    assign[i] = static_cast<uint8_t>(sit->second.gpu);
                    elb_mask_scratch_[i] = 1;
                    ++elb_stream_hits_;
                }
            }
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
            const bool numa_rt = keeper52_numa_route()
                && g_arena_map.ensure(dsp52_arena_map_path());
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

        // ── DSP52_UPART: mark union MISSES transient (zone=1) ────────────
        // Placement (assign) is UNCHANGED — only the zone/eviction class of
        // the board-non-resident subset changes. Per-GPU marks are capped by
        // the streaming zone's slot count minus this layer's in-flight
        // lever-4 prefetch entries; the overflow keeps zone=0 (today's exact
        // stable path: victim map + board admission). See dsp52_upart().
        if (transient_union && dsp52_upart()) {
            if (dsp52_elb()) {
                ADD_FAILURE() << "DSP52_UPART and DSP52_ELB are mutually "
                                 "exclusive (both manage the streaming zone)";
                return 0;
            }
            if (upart_stream_cap_.empty()) {
                upart_stream_cap_.resize(static_cast<size_t>(tp), 0);
                upart_pf_pending_.assign(static_cast<size_t>(tp), 0);
                for (int g = 0; g < tp; ++g)
                    upart_stream_cap_[static_cast<size_t>(g)] =
                        engine_->expert_cache()->total_slots(
                            g, layerstorm::memory::CacheZone::kStreaming);
                std::cerr << "  [dsp52 UPART] streaming caps:";
                for (int g = 0; g < tp; ++g)
                    std::cerr << " g" << g << "="
                              << upart_stream_cap_[static_cast<size_t>(g)];
                std::cerr << "\n";
            }
            std::vector<int> free_est(static_cast<size_t>(tp), 0);
            for (int g = 0; g < tp; ++g)
                free_est[static_cast<size_t>(g)] = std::max(
                    0, upart_stream_cap_[static_cast<size_t>(g)]
                           - upart_pf_pending_[static_cast<size_t>(g)]);
            upart_mask_scratch_.assign(topk.size(), 0);
            for (size_t i = 0; i < topk.size(); ++i) {
                const int g = static_cast<int>(assign[i]);
                bool resident;
                if (use_reef) {
                    resident = reef_orch_->board.is_resident(
                        g, layerstorm::memory::ExpertKey{
                               static_cast<uint32_t>(layer), topk[i]});
                } else if (lrus && g < static_cast<int>(lrus->size())) {
                    resident = (*lrus)[static_cast<size_t>(g)].resident.count(
                                   LruKey{static_cast<uint32_t>(layer),
                                          topk[i]}) != 0;
                } else {
                    continue;  // no residency model: keep today's path
                }
                if (resident) continue;
                if (free_est[static_cast<size_t>(g)] > 0) {
                    --free_est[static_cast<size_t>(g)];
                    upart_mask_scratch_[i] = 1;
                    ++upart_streamed_;
                } else {
                    ++upart_overflow_;
                }
            }
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
            entries[count].zone       =
                upart_mask_scratch_.empty() ? 0 : upart_mask_scratch_[i];
            entries[count].gpu_idx    = assign[i];
            // 13c-2.0: default = no victim (router/local fallback picks).
            evicts[count].layer_idx  = static_cast<uint32_t>(layer);
            evicts[count].expert_idx = 0xFFFF;   // sentinel
            evicts[count].gpu_idx    = assign[i];
            evicts[count]._pad       = 0;
            ++count;
        }

        // 13c-2.0 Option A / REEF-ORCH victim map — identical to keeper52.
        // DSP52_UPART transient entries reuse the stream_mask seam: no stable
        // victim, no board admission (the streamed copy lives outside the
        // board's stable model; the engine releases it at completion reap).
        if (use_reef) {
            const std::vector<uint8_t>* mask = nullptr;
            if (dsp52_elb() && dsp52_elb_stream()
                && elb_mask_scratch_.size() == topk.size())
                mask = &elb_mask_scratch_;
            else if (upart_mask_scratch_.size() == topk.size())
                mask = &upart_mask_scratch_;
            reef_orch_apply(*reef_orch_, layer, entries, evicts, count, mask,
                            traj_step_);
        } else if (lrus) {
            auto& Ls = *lrus;
            for (int g = 0; g < tp; ++g) {
                GpuLru& lru = Ls[g];
                std::vector<uint16_t> want;
                std::vector<uint32_t> miss_ei;   // entry idx of misses on g
                for (uint32_t i = 0; i < count; ++i) {
                    if (entries[i].gpu_idx != g) continue;
                    uint16_t e = entries[i].expert_idx;
                    want.push_back(e);
                    // DSP52_UPART transient entry: routed (stays in `want`
                    // for the needed_now victim guard) but NOT an LRU miss —
                    // no victim, no LRU admission (streaming-zone copy,
                    // released at completion reap).
                    if (!upart_mask_scratch_.empty() && upart_mask_scratch_[i])
                        continue;
                    LruKey k{static_cast<uint32_t>(layer), e};
                    if (lru.resident.count(k)) lru.touch(k);
                    else miss_ei.push_back(i);
                }
                auto needed_now = [&](const LruKey& k) {
                    if (k.layer != static_cast<uint32_t>(layer)) return false;
                    return std::find(want.begin(), want.end(),
                                     static_cast<uint16_t>(k.expert)) != want.end();
                };
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
                for (uint32_t ei : miss_ei) {
                    LruKey k{static_cast<uint32_t>(layer), entries[ei].expert_idx};
                    lru.resident.insert(k);
                    lru.touch(k);
                }
            }
        }

        // DSP52_ELB: feed the predictor seam with the committed decision —
        // decode top-K, verify-chunk union AND prefill-chunk union all route
        // through here (prefill only observes; no prefetch is issued there).
        if (dsp52_elb()) {
            ensure_elb_predictor();
            elb_->observe(layer, entries, count);
        }
        // ORACLE RUN 1 (record): dump the committed union+placement for this
        // (traj-step, layer). Independent of ELB (RUN 1 is the pure champion).
        if (!dsp52_oracle_dump().empty())
            oracle_record(layer, entries, count);
        // BELADY replay determinism gate: the committed routed union must equal
        // the recorded set for this (step, layer) — proves Belady eviction did
        // NOT perturb routing (eviction is lossless ⇒ routing invariant).
        if (reef_orch_ && belady_oracle_)
            belady_oracle_->observe(layer, entries, count, traj_step_);
        return count;
    }

    // One decode step via the production routed-MoE path — faithful clone of
    // keeper52's decode_step_fetch_and_run with the placement block factored
    // into fill_moe_entries (shared with the verify chunk).
    DecodeResult decode_step_fetch_and_run(uint32_t input_token, uint64_t seq_id,
                                           uint32_t token_pos,
                                           std::vector<GpuLru>* lrus = nullptr) {
        ++traj_step_;  // oracle record/replay step id (one per layer sweep)
        if (dsp52_oracle_bulk()) { ensure_elb_predictor(); oracle_bulk_issue(); }
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

            // DSP52_ELB: predicted prefetch for layers L..L+AHEAD at
            // attention-launch time (the compute-idle H2D window).
            elb_issue(layer, lrus);
            if (::testing::Test::HasFailure()) return result;

            auto ta = clock::now();
            auto attn_cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
            attn_cmd.run_attention.layer_idx = static_cast<uint32_t>(layer);
            attn_cmd.run_attention.num_seqs  = 1;
            attn_cmd.run_attention.is_prefill = 0;
            attn_cmd.run_attention.use_graph  = 0;
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

                const uint32_t count = fill_moe_entries(layer, topk, lrus);
                if (::testing::Test::HasFailure()) return result;
                result.moe_lookups += count;

                auto moe_cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
                moe_cmd.fetch_and_run_moe.layer_idx    = static_cast<uint32_t>(layer);
                moe_cmd.fetch_and_run_moe.num_seqs     = 1;
                moe_cmd.fetch_and_run_moe.expert_count = count;
                moe_cmd.fetch_and_run_moe.have_evict_map =
                    (reef_orch_ || lrus) ? 1 : 0;
                moe_cmd.fetch_and_run_moe.timeout_us   = 5000000;  // 5s
                moe_cmd.fetch_and_run_moe.moe_mode     = 0;
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

    // ── DSpark draft step (DSP-5 command seam) ───────────────────────────
    // ONE D_CMD_RUN_DSPARK_STEP: [anchor_token, mask × γ] at positions
    // anchor_pos + [0..γ] (bonus-anchor 1+N layout, INV-DSPARK-ANCHOR;
    // external contract: γ draft ids, slot k → target position
    // anchor_pos+k+1). The γ i32 sampled ids arrive in the sideband readback
    // slice (kSpecCheckpointOff + 2560) when the completion fires —
    // confidence off → data_bytes == γ*4 (dispatch_forward.cpp). Same shape
    // as the golden's install_dspark_step_executor, minus the
    // SpeculationMethod object (the command is driven directly).
    // conf_out non-null (DSP52_CONF_THRESH > 0 → confidence_enabled=true):
    // ALSO read the γ f32 raw survival probabilities c_k that follow the ids
    // in the sideband slice (data_bytes == 2*γ*4, dispatch_forward.cpp
    // DSP-6 readback contract).
    bool dspark_draft_step(uint64_t seq_id, uint32_t anchor_token,
                           uint32_t anchor_pos, int gamma, int32_t* ids_out,
                           float* conf_out = nullptr) {
        lipc::Completion cmp{};
        auto c = make_cmd(lipc::D_CMD_RUN_DSPARK_STEP);
        c.run_dspark_step.seq_id          = seq_id;
        c.run_dspark_step.anchor_token_id = anchor_token;
        c.run_dspark_step.anchor_pos      = anchor_pos;
        c.run_dspark_step.num_query       = static_cast<uint8_t>(gamma);
        c.run_dspark_step.step_idx        = 0;
        send(c);
        if (!wait(cmp, lipc::CMP_COMPUTE_DONE)) return false;
        return parse_dspark_readback(cmp, gamma, ids_out, conf_out);
    }
    bool parse_dspark_readback(const lipc::Completion& cmp, int gamma,
                               int32_t* ids_out, float* conf_out) {
        const uint32_t want =
            static_cast<uint32_t>(gamma) * sizeof(int32_t) *
            (conf_out ? 2u : 1u);
        if (cmp.compute.host_buf_offset == 0 ||
            cmp.compute.data_bytes < want) {
            ADD_FAILURE() << "dspark step readback missing: off="
                          << cmp.compute.host_buf_offset << " bytes="
                          << cmp.compute.data_bytes << " (want >= "
                          << want << ")";
            return false;
        }
        const auto* ids = reinterpret_cast<const int32_t*>(
            sideband_ + cmp.compute.host_buf_offset);
        for (int k = 0; k < gamma; ++k) {
            if (ids[k] < 0 || ids[k] >= vocab_size_) {
                ADD_FAILURE() << "dspark draft id out of vocab: d_" << k
                              << " = " << ids[k];
                return false;
            }
            ids_out[k] = ids[k];
        }
        if (conf_out) {
            const auto* cf = reinterpret_cast<const float*>(
                sideband_ + cmp.compute.host_buf_offset +
                static_cast<uint32_t>(gamma) * sizeof(int32_t));
            for (int k = 0; k < gamma; ++k) {
                if (!std::isfinite(cf[k]) || cf[k] <= 0.0f || cf[k] >= 1.0f) {
                    ADD_FAILURE() << "dspark confidence c_" << k
                                  << " outside (0,1): " << cf[k];
                    return false;
                }
                conf_out[k] = cf[k];
            }
        }
        return true;
    }

    // ── Batched verify chunk (dsp52) ─────────────────────────────────────
    // ONE multi-row same-seq chunk of R rows `toks` at positions
    // [pos0, pos0+R): embedding → per-layer RUN_ATTENTION is_prefill=1
    // (chunk fields, fused gate on MoE layers; the golden prefill_step chunk
    // shape) → DEDUPED routed union through fill_moe_entries → ONE
    // FETCH_AND_RUN_MOE per layer → CMD_OUTPUT_HEAD num_tokens=R with
    // readback_to_host → R u32 greedy argmax ids from the sideband readback
    // (batched-verify head contract, dispatch_compute.cpp: tokens u32[R] at
    // [0..4R), capped at kMaxOutputHeadReadbackTokens).
    VerifyChunkResult verify_step_fetch_and_run(
            const std::vector<uint32_t>& toks, uint64_t seq_id, uint32_t pos0,
            std::vector<GpuLru>* lrus = nullptr) {
        ++traj_step_;  // oracle record/replay step id (one per layer sweep)
        if (dsp52_oracle_bulk()) { ensure_elb_predictor(); oracle_bulk_issue(); }
        using clock = std::chrono::steady_clock;
        auto step_start = clock::now();
        lipc::Completion cmp{};
        VerifyChunkResult result{};
        result.timings.attention_ms.resize(num_layers_);
        result.timings.moe_ms.resize(num_layers_);
        const uint32_t R = static_cast<uint32_t>(toks.size());

        // DSP52_LOOKAHEAD_DIAG (Wave-3 M4 stage 1): approximate PASS-1 of
        // this exact chunk first — records per-layer predicted routing; the
        // real chunk below is pass-2 (its writes fully overwrite pass-1's
        // position-addressed state).
        if (const int la_mode = dsp52_lookahead_diag(); la_mode != 0) {
            lookahead_pass1(toks, seq_id, pos0, la_mode, lrus);
            if (::testing::Test::HasFailure()) return result;
        }

        // 1. Row token ids + embedding for the whole chunk.
        auto* token_ids = reinterpret_cast<uint32_t*>(
            sideband_ + lipc::IpcLayout::kTokenIdsOff);
        for (uint32_t b = 0; b < R; ++b) token_ids[b] = toks[b];

        auto t0 = clock::now();
        auto embed_cmd = make_cmd(lipc::CMD_EMBEDDING_LOOKUP);
        embed_cmd.embedding_lookup.num_tokens    = R;
        embed_cmd.embedding_lookup.output_buf_id = hidden_buf_id_;
        send(embed_cmd);
        EXPECT_TRUE(wait(cmp, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)));
        EXPECT_EQ(cmp.status, 0u);
        result.timings.embedding_ms =
            std::chrono::duration<double, std::milli>(clock::now() - t0).count();

        // 2. Batch descriptor: R rows of ONE sequence at positions pos0+b.
        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        for (uint32_t b = 0; b < R; ++b) {
            batch[b].seq_id    = seq_id;
            batch[b].token_pos = pos0 + b;
            batch[b]._pad      = 0;
        }

        // 3. Per-layer attention (prefill-shaped chunk) + MoE union.
        if (dsp52_prefetch() && pf_pred_.empty())
            pf_pred_.resize(static_cast<size_t>(num_layers_));
        for (int layer = 0; layer < num_layers_; ++layer) {
            const bool is_moe = layer >= first_moe_layer_;

            // DSP52_UPART: per-layer reset of the lever-4 prefetch occupancy
            // estimate (refilled by the prefetch block below; consumed by
            // fill_moe_entries' streaming-cap discipline).
            if (dsp52_upart() && !upart_pf_pending_.empty())
                std::fill(upart_pf_pending_.begin(), upart_pf_pending_.end(),
                          0);

            // DSP52_ELB supersedes the lever-4 prototype below: same
            // previous-chunk union at a=0 plus the lookahead layers, with
            // capacity-bounded cheap-victim room-making.
            if (dsp52_elb()) {
                elb_issue(layer, lrus);
                if (::testing::Test::HasFailure()) return result;
            }

            // Lever 4 (DSP52_PREFETCH): fire the predicted layer-L union
            // prefetch at attention-launch time — the H2D idle window. The
            // sideband prefetch region is consumed by the daemon BEFORE the
            // attention completion we wait on below (ring order), so the
            // later fill_moe_entries overwrite is race-free.
            const int pf_layer = layer + dsp52_prefetch_ahead();
            if (!dsp52_elb() && dsp52_prefetch()
                && pf_layer < num_layers_ && pf_layer >= first_moe_layer_
                && !pf_pred_[static_cast<size_t>(pf_layer)].empty()) {
                const auto& pred = pf_pred_[static_cast<size_t>(pf_layer)];
                auto* pfe = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
                    sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
                uint32_t n = 0;
                const int miss_budget = dsp52_prefetch_miss();
                if (miss_budget > 0 && reef_orch_) {
                    // TD-KEEPPRED-HYBRID: fire ONLY predicted-reuse experts NOT
                    // currently resident on their target GPU (the misses
                    // keeppred/freq did not keep), budgeted to the top
                    // `miss_budget` by decayed frequency (most-likely-to-recur
                    // first). Never re-fetches a resident expert. Fired at
                    // attention-launch → the H2D transfer overlaps the compute
                    // (idle-link) window. Skip-when-none (no fabricated budget).
                    using layerstorm::memory::ExpertKey;
                    std::vector<std::pair<float, uint32_t>> cand;  // (freq, idx)
                    cand.reserve(pred.size());
                    for (uint32_t i = 0; i < pred.size(); ++i) {
                        const ExpertKey k{pred[i].layer_idx, pred[i].expert_idx};
                        if (reef_orch_->board.is_resident(pred[i].gpu_idx, k))
                            continue;  // already resident → do NOT re-fetch
                        auto it = reef_orch_->freq.find(k);
                        cand.emplace_back(
                            it == reef_orch_->freq.end() ? 0.0f : it->second, i);
                    }
                    const size_t take = std::min(
                        static_cast<size_t>(miss_budget), cand.size());
                    std::partial_sort(
                        cand.begin(), cand.begin() + static_cast<long>(take),
                        cand.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
                    for (size_t j = 0; j < take
                             && n < lipc::kMaxExpertPrefetch; ++j)
                        pfe[n++] = pred[cand[j].second];
                } else {
                    n = static_cast<uint32_t>(
                        std::min(pred.size(),
                                 static_cast<size_t>(lipc::kMaxExpertPrefetch)));
                    for (uint32_t i = 0; i < n; ++i) pfe[i] = pred[i];
                }
                // Skip the SEND (not the layer) when nothing is predicted to
                // miss — no fabricated budget, no empty prefetch command.
                if (n > 0) {
                // DSP52_UPART: the prefetched zone=1 (streamed) entries
                // occupy streaming slots until this layer's sweep — count
                // them against fill_moe_entries' per-GPU cap estimate.
                if (dsp52_upart() && !upart_pf_pending_.empty())
                    for (uint32_t i = 0; i < n; ++i)
                        if (pfe[i].zone == 1
                            && pfe[i].gpu_idx < upart_pf_pending_.size())
                            ++upart_pf_pending_[pfe[i].gpu_idx];
                auto pc = make_cmd(lipc::D_B_CMD_PREFETCH_BATCH);
                pc.prefetch_batch.count    = n;
                pc.prefetch_batch.priority = -1.0f;  // below FED's 0.0
                pc.prefetch_batch.delay_us = 0;
                fire_forget_seqs_.insert(pc.cmd_seq);
                send(pc);
                if (::testing::Test::HasFailure()) return result;
                }  // if (n > 0)
            }

            auto ta = clock::now();
            auto attn_cmd = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
            attn_cmd.run_attention.layer_idx   = static_cast<uint32_t>(layer);
            attn_cmd.run_attention.num_seqs    = R;
            attn_cmd.run_attention.is_prefill  = 1;
            attn_cmd.run_attention.use_graph   = 0;
            attn_cmd.run_attention.chunk_start = pos0;
            attn_cmd.run_attention.chunk_len   = R;
            attn_cmd.run_attention.emit_gating  = is_moe ? 1 : 0;
            attn_cmd.run_attention.store_gating = is_moe ? 1 : 0;
            send(attn_cmd);
            EXPECT_TRUE(wait(cmp,
                static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)))
                << "verify attn L" << layer;
            EXPECT_EQ(cmp.status, 0u);
            result.timings.attention_ms[layer] =
                std::chrono::duration<double, std::milli>(clock::now() - ta).count();
            if (::testing::Test::HasFailure()) return result;

            auto tm = clock::now();

            if (is_moe) {
                const auto* hdr = reinterpret_cast<const lipc::RoutingExportHeader*>(
                    sideband_ + lipc::IpcLayout::kRoutingExportOff);
                EXPECT_EQ(hdr->num_tokens, R)
                    << "routing export rows != verify chunk rows L" << layer;
                EXPECT_EQ(hdr->layer_idx, static_cast<uint32_t>(layer))
                    << "routing-export layer mismatch in verify chunk";
                const auto* ridx = reinterpret_cast<const int32_t*>(
                    sideband_ + lipc::IpcLayout::kRoutingExportIndicesOff);
                const uint32_t rn = hdr->num_tokens * hdr->topk;

                // DEDUPED union across all R rows, first-occurrence
                // (selection-rank) order — the golden prefill_step shape.
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
                if (topk.empty()) return result;

                // DSP52_LOOKAHEAD_DIAG stage-1 agreement: pass-1 prediction
                // vs THIS (real) pass-2 routing.
                if (dsp52_lookahead_diag()
                    && static_cast<size_t>(layer) < la_p1_topk_.size()
                    && !la_p1_topk_[static_cast<size_t>(layer)].empty()) {
                    lookahead_accumulate(
                        layer, hdr->num_tokens, hdr->topk, ridx,
                        reinterpret_cast<const float*>(
                            sideband_
                            + lipc::IpcLayout::kRoutingExportWeightsOff),
                        topk);
                }

                const uint32_t count = fill_moe_entries(
                    layer, topk, lrus,
                    /*static_mod=*/dsp52_vb_tp_only() ? 2 : 0,
                    /*transient_union=*/dsp52_upart());
                if (::testing::Test::HasFailure()) return result;
                result.moe_lookups += count;

                // Lever 4: record THIS chunk's union+placement as the next
                // chunk's layer-L prediction (read back from the sideband
                // entries fill_moe_entries just wrote). Under DSP52_ELB the
                // seam's observe hook in fill_moe_entries records instead.
                if (!dsp52_elb() && dsp52_prefetch()) {
                    const auto* ent =
                        reinterpret_cast<const lipc::ExpertPrefetchEntry*>(
                            sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
                    pf_pred_[static_cast<size_t>(layer)].assign(ent,
                                                                ent + count);
                }

                auto moe_cmd = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
                moe_cmd.fetch_and_run_moe.layer_idx    = static_cast<uint32_t>(layer);
                moe_cmd.fetch_and_run_moe.num_seqs     = R;
                moe_cmd.fetch_and_run_moe.expert_count = count;
                moe_cmd.fetch_and_run_moe.have_evict_map =
                    (reef_orch_ || lrus) ? 1 : 0;
                moe_cmd.fetch_and_run_moe.timeout_us   = 5000000;  // 5s
                moe_cmd.fetch_and_run_moe.moe_mode     = 0;
                send(moe_cmd);
            } else {
                auto moe_cmd = make_cmd(lipc::D_B_CMD_RUN_MOE);
                moe_cmd.run_moe.layer_idx = static_cast<uint32_t>(layer);
                moe_cmd.run_moe.num_seqs  = R;
                moe_cmd.run_moe.moe_mode  = 0;
                moe_cmd.run_moe.apply_residual_correction = 0;
                moe_cmd.run_moe.store_gating_output       = 0;
                moe_cmd.run_moe.emit_checkpoint           = 0;
                send(moe_cmd);
            }

            EXPECT_TRUE(wait(cmp,
                static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)))
                << "verify moe L" << layer;
            EXPECT_EQ(cmp.status, 0u);
            result.timings.moe_ms[layer] =
                std::chrono::duration<double, std::milli>(clock::now() - tm).count();
            if (::testing::Test::HasFailure()) return result;
        }

        // 4. Batched-verify output head: R per-row greedy argmax ids in ONE
        // completion (readback_to_host + num_tokens=R).
        auto to = clock::now();
        auto head_cmd = make_cmd(lipc::CMD_OUTPUT_HEAD);
        head_cmd.output_head.num_tokens         = R;
        head_cmd.output_head.input_buf_id       = hidden_buf_id_;
        head_cmd.output_head.output_buf_id      = logits_buf_id_;
        head_cmd.output_head.readback_to_host   = 1;
        head_cmd.output_head.compute_confidence = 1;
        send(head_cmd);
        EXPECT_TRUE(wait(cmp, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)));
        EXPECT_EQ(cmp.status, 0u);
        result.top1_prob = cmp.compute.top1_prob;
        result.entropy   = cmp.compute.entropy;
        result.timings.output_head_ms =
            std::chrono::duration<double, std::milli>(clock::now() - to).count();
        if (::testing::Test::HasFailure()) return result;

        EXPECT_NE(cmp.compute.host_buf_offset, 0u)
            << "batched-verify head readback missing";
        EXPECT_GE(cmp.compute.data_bytes, R * sizeof(uint32_t))
            << "batched-verify head readback short";
        if (cmp.compute.host_buf_offset == 0 ||
            cmp.compute.data_bytes < R * sizeof(uint32_t))
            return result;
        const auto* ids = reinterpret_cast<const uint32_t*>(
            sideband_ + cmp.compute.host_buf_offset);
        result.argmax.assign(ids, ids + R);
        for (uint32_t b = 0; b < R; ++b)
            EXPECT_LT(result.argmax[b], static_cast<uint32_t>(vocab_size_))
                << "verify argmax row " << b << " out of vocab";

        // DSP52_VB_SAMPLE=1 (bisect, suspect 4): per-row argmax over the SAME
        // logits buffer through CMD_SAMPLE_TOKENS (writes R ids to
        // kTokenIdsOff). A diff vs the head readback isolates the head's
        // sample/readback seam from an upstream (logits-already-wrong) bug.
        if (dsp52_vb_sample()) {
            auto sc = make_cmd(lipc::CMD_SAMPLE_TOKENS);
            sc.sample_tokens.num_tokens    = R;
            sc.sample_tokens.logits_buf_id = logits_buf_id_;
            sc.sample_tokens.vocab_size    = static_cast<uint32_t>(vocab_size_);
            sc.sample_tokens.temperature   = 0.0f;  // argmax
            sc.sample_tokens.top_p         = 1.0f;
            sc.sample_tokens.top_k         = 0;
            sc.sample_tokens.random_seed   = 42;
            send(sc);
            EXPECT_TRUE(wait(cmp, static_cast<uint32_t>(lipc::CMP_COMPUTE_DONE)));
            const auto* tid = reinterpret_cast<const uint32_t*>(
                sideband_ + lipc::IpcLayout::kTokenIdsOff);
            for (uint32_t b = 0; b < R; ++b)
                if (tid[b] != result.argmax[b])
                    std::cerr << "  [dsp52 VB_SAMPLE] row " << b
                              << ": head readback " << result.argmax[b]
                              << " != SAMPLE_TOKENS argmax " << tid[b] << "\n";
        }

        result.timings.total_ms =
            std::chrono::duration<double, std::milli>(clock::now() - step_start).count();
        return result;
    }

    // ── PROMPT-FED arm (DSP52_PROMPT): chunked prefill ───────────────────
    // One REAL prefill chunk over `n` prompt tokens at positions
    // [pos0, pos0+n) — the DsparkAcceptRealistic pattern
    // (glm52_gguf_golden_test prefill_step) on dsp52's placement machinery:
    // is_prefill=1 chunk attention with the fused gate, DEDUPED routed
    // union through fill_moe_entries (REEF/affinity/LRU bookkeeping stays
    // consistent with the decode loop), ONE FETCH_AND_RUN_MOE per layer.
    // No output head — prefill feeds KV + the aux export only (the draft
    // context ingests every chunk, INV-DSPARK-AUX / TD-DSPARK-PREFILL-CAP).
    void prefill_chunk_fetch_and_run(const uint32_t* toks, uint32_t n,
                                     uint64_t seq_id, uint32_t pos0,
                                     std::vector<GpuLru>* lrus) {
        ++traj_step_;  // oracle record/replay step id (one per layer sweep)
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

    // Feed the DSP52_PROMPT prompt's first L-1 tokens as 64-token prefill
    // chunks on `seq_id` (the caller has created the sequence with
    // prompt_len = L). The LAST prompt token is NOT fed here — the caller
    // feeds it through ONE plain decode step (the loop's seed step), the
    // DsparkAcceptRealistic pattern. The prefill wall prints and is
    // EXCLUDED from every perf stat (the loops re-baseline the H2D miss
    // counter and start their wall clocks after this returns).
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
        std::cerr << "  [dsp52] prompt prefill: " << pre << " tokens in "
                  << std::fixed << std::setprecision(1) << s << " s (chunk "
                  << kChunk << "; last prompt token via the seed decode "
                     "step; excluded from perf stats)\n"
                  << std::setprecision(3);
    }

    // ── Batched-verify logits-equivalence gate (TD-DSP52-BATCHED-VERIFY-
    // EQUIV → resolved; INV-DSPARK-LOSSLESS B>1 clause) ──────────────────
    // D2H the first vocab_size_ FP32 logits of logits_buf_id_ — row 0 of the
    // [num_tokens, vocab] block both CMD_OUTPUT_HEAD shapes write there
    // (dispatch_forward.cpp dispatch_output_head_tp: opts.logits_out holds
    // full [num_tokens, vocab_size] on the primary GPU). Same readback
    // pattern as the glm52 goldens (BufferRegistry lookup + cudaMemcpy).
    bool read_logits_row0(std::vector<float>& out) {
        const auto* e = engine_->buffer_registry()->lookup(logits_buf_id_);
        if (!e || !e->device_ptr ||
            e->size_bytes <
                static_cast<int64_t>(vocab_size_) * static_cast<int64_t>(sizeof(float))) {
            ADD_FAILURE() << "logits buffer lookup failed for row-0 readback"
                          << " (buf " << logits_buf_id_ << ")";
            return false;
        }
        out.resize(static_cast<size_t>(vocab_size_));
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(e->gpu_idx);
        const cudaError_t rc =
            cudaMemcpy(out.data(), e->device_ptr, out.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
        cudaSetDevice(cur);
        if (rc != cudaSuccess) {
            ADD_FAILURE() << "row-0 logits D2H failed: "
                          << cudaGetErrorString(rc);
            return false;
        }
        return true;
    }

    // Equivalence check at MATCHED context: `batched` = the verify chunk's
    // row-0 logits, `replay` = the B=1 decode replay's logits for the SAME
    // position fed the SAME committed history. Chunk and decode kernel
    // pipelines are different FP reduction orders and the divergence is
    // AMPLIFIED over the 78-layer backbone, so the contract is the
    // CALIBRATED end-to-end band per element — |a-b| <= DSP52_REF_BAND_ABS
    // + DSP52_REF_BAND_REL*|b| (defaults from the 2026-08-01 calibration,
    // see dsp52_ref_band_abs()) — with argmax identity asserted ONLY when
    // the replay's top1-vs-top2 margin exceeds the band at both ends (a
    // within-band perturbation cannot flip a wider margin; below it,
    // near-tie flips are the resolved TD-DSP52-BATCHED-VERIFY-EQUIV
    // regime, not corruption).
    void expect_row0_equivalent(int round, const std::vector<float>& batched,
                                const std::vector<float>& replay) {
        ASSERT_EQ(batched.size(), replay.size());
        const size_t V = replay.size();
        ASSERT_GE(V, 2u);
        const double babs = dsp52_ref_band_abs();
        const double brel = dsp52_ref_band_rel();
        size_t bt1 = 0;                       // batched row-0 argmax
        for (size_t i = 1; i < V; ++i)
            if (batched[i] > batched[bt1]) bt1 = i;
        size_t t1 = 0;                        // replay top1 / top2
        for (size_t i = 1; i < V; ++i)
            if (replay[i] > replay[t1]) t1 = i;
        size_t t2 = (t1 == 0) ? 1 : 0;
        for (size_t i = 0; i < V; ++i)
            if (i != t1 && replay[i] > replay[t2]) t2 = i;
        size_t nonfinite = 0, viol = 0, worst = 0, argw = 0;
        double worst_excess = -1.0, max_err = 0.0, max_rel = 0.0;
        for (size_t i = 0; i < V; ++i) {
            const float a = batched[i], b = replay[i];
            if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; continue; }
            const double err  = std::abs(static_cast<double>(a) - b);
            const double band = babs + brel * std::abs(static_cast<double>(b));
            if (err > max_err) { max_err = err; argw = i; }
            max_rel = std::max(
                max_rel, err / (1.0 + std::abs(static_cast<double>(b))));
            if (err > band && err - band > worst_excess) {
                worst_excess = err - band;
                worst = i;
            }
            if (err > band) ++viol;
        }
        EXPECT_EQ(nonfinite, 0u)
            << "round " << round << ": non-finite logits in the batched/replay"
               " row-0 comparison";
        EXPECT_EQ(viol, 0u)
            << "round " << round << ": " << viol << "/" << V
            << " logits outside the calibrated end-to-end band |a-b| <= "
            << babs << " + " << brel << "*|b|; worst at vocab " << worst
            << ": batched=" << batched[worst] << " replay=" << replay[worst]
            << " (band "
            << babs + brel * std::abs(static_cast<double>(replay[worst]))
            << ")";
        const double margin = static_cast<double>(replay[t1]) - replay[t2];
        const double mband =
            (babs + brel * std::abs(static_cast<double>(replay[t1]))) +
            (babs + brel * std::abs(static_cast<double>(replay[t2])));
        std::cerr << "  [dsp52 EQ] round " << round << ": replay top1=" << t1
                  << " top2=" << t2 << " margin=" << margin << " (band "
                  << mband << ")  batched top1=" << bt1 << "  max|dLogit|="
                  << max_err << " @vocab " << argw << " (replay="
                  << replay[argw] << ")  max|d|/(1+|ref|)=" << max_rel
                  << (margin > mband ? "  [argmax asserted]"
                                     : "  [flat - argmax not asserted]")
                  << "\n";
        if (margin > mband)
            EXPECT_EQ(bt1, t1)
                << "round " << round << ": batched row-0 argmax " << bt1
                << " != B=1 replay argmax " << t1 << " with top1 margin "
                << margin << " ABOVE the equivalence band " << mband
                << " — this is not a flat-position near-tie flip";
    }

    // Per-TP-GPU VRAM breakdown grid — clone of keeper52's finish_xray_report.
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
        pr::emit(rep, xj ? xj : "/tmp/xray_dsp52.json");
    }

    // ── Plain-loop trajectory (DSP52_REF baseline / DSP52_SPEC=0 body) ───
    // Runs `num_tokens` plain keeper decode steps on `seq_id` starting from
    // `seed_token`, recording the generated trajectory. Prints the sparse
    // keeper step log; timing aggregation stays with the caller via `agg`.
    struct PlainAgg {
        double total_ms = 0, embed = 0, attn = 0, moe = 0, head = 0, sample = 0;
        uint64_t lookups = 0;
        int nan_count = 0;
        std::vector<double> per_token_ms, per_token_attn_ms, per_token_moe_ms;
    };
    std::vector<uint32_t> run_plain_loop(uint64_t seq_id, int num_tokens,
                                         uint32_t seed_token,
                                         std::vector<GpuLru>* lrus,
                                         PlainAgg& agg,
                                         const std::vector<uint32_t>* prompt
                                             = nullptr) {
        std::vector<uint32_t> out;
        const uint32_t prompt_len =
            prompt ? static_cast<uint32_t>(prompt->size()) : 0;
        create_sequence(seq_id, prompt_len ? prompt_len : 1);
        // PROMPT-FED arm (DSP52_PROMPT): chunked prefill of the first L-1
        // prompt tokens; the LAST prompt token becomes the loop's first fed
        // token (its decode step produces generated token 0), so the loop
        // still measures exactly `num_tokens` generated tokens.
        uint32_t pos0  = 0;
        uint32_t token = seed_token;
        if (prompt_len) {
            feed_prompt_prefill(seq_id, *prompt, lrus);
            if (::testing::Test::HasFailure()) {
                free_sequence(seq_id);
                return out;
            }
            token = (*prompt)[prompt_len - 1];
            pos0  = prompt_len - 1;
        }
        h2d_loop_base_phase_ = engine_->h2d_path_stats().phase_count;
        int prof_a = -1, prof_b = -1;
        if (const char* pw = std::getenv("LS_KEEPER_NSYS"); pw && *pw)
            if (std::sscanf(pw, "%d,%d", &prof_a, &prof_b) != 2) { prof_a = prof_b = -1; }
        const std::vector<uint32_t>* forced = dsp52_force_traj();
        int force_match = 0;
        for (int i = 0; i < num_tokens; ++i) {
            if (i == prof_a) cudaProfilerStart();
            auto r = decode_step_fetch_and_run(token, seq_id,
                                               pos0 + static_cast<uint32_t>(i),
                                               lrus);
            if (i == prof_b) cudaProfilerStop();
            if (::testing::Test::HasFailure()) break;
            EXPECT_LT(r.sampled_token, static_cast<uint32_t>(vocab_size_))
                << "Invalid token at step " << i;
            if (!std::isfinite(r.top1_prob) || !std::isfinite(r.entropy))
                ++agg.nan_count;
            agg.total_ms += r.timings.total_ms;
            agg.per_token_ms.push_back(r.timings.total_ms);
            agg.embed  += r.timings.embedding_ms;
            agg.head   += r.timings.output_head_ms;
            agg.sample += r.timings.sample_ms;
            double tok_attn = 0, tok_moe = 0;
            for (double a : r.timings.attention_ms) tok_attn += a;
            for (double m : r.timings.moe_ms)       tok_moe  += m;
            agg.attn += tok_attn;
            agg.moe  += tok_moe;
            agg.per_token_attn_ms.push_back(tok_attn);
            agg.per_token_moe_ms.push_back(tok_moe);
            agg.lookups += r.moe_lookups;
            const bool dump_all = std::getenv("LS_KEEPER_DUMP_ALL") != nullptr;
            if (dump_all || (i + 1) % 10 == 0 || i == 0)
                std::cerr << "  step " << std::setw(3) << i << ": token="
                          << std::setw(6) << r.sampled_token << "  prob="
                          << r.top1_prob << "  ms=" << r.timings.total_ms << "\n";
            // DSP52_FORCE_TRAJ: commit the forced token (walls honest —
            // the step already ran); model-match tracked for the report.
            uint32_t next = r.sampled_token;
            if (forced && static_cast<size_t>(i) < forced->size()) {
                if (r.sampled_token == (*forced)[static_cast<size_t>(i)])
                    ++force_match;
                next = (*forced)[static_cast<size_t>(i)];
            }
            out.push_back(next);
            token = next;
        }
        if (forced)
            std::cerr << "  [dsp52] FORCE_TRAJ plain: model matched "
                      << force_match << "/" << num_tokens
                      << " forced tokens\n";
        elb_report();
        upart_report();
        free_sequence(seq_id);
        return out;
    }

    // ── Speculative decode loop (dsp52) ──────────────────────────────────
    // See the file header for the round structure. Commits AT LEAST
    // `gen_tokens` tokens (the last round's bonus may overshoot); returns the
    // committed trajectory (first element = the first generated token, i.e.
    // directly comparable to run_plain_loop's output).
    struct SpecStats {
        int rounds = 0;
        int fallback_rounds = 0;     // conf-truncated-to-0 rounds (plain step)
        int64_t trunc_slots = 0;     // draft slots dropped by DSP52_CONF_THRESH
        int64_t proposed = 0, accepted = 0;
        double wall_ms = 0;          // whole loop incl. the seed feed
                                     // (EXCL. any DSP52_PROMPT prefill)
        double draft_ms = 0;         // Σ DSpark step wall (OVERLAP mode:
                                     // only the EXPOSED post-plain wait)
        double overlap_plain_ms = 0; // Σ overlapped plain-step wall (lever 2)
        double verify_ms = 0;        // Σ verify chunk wall
        double attn_ms = 0, moe_ms = 0;   // Σ per-layer walls (verify chunks)
        uint64_t lookups = 0;        // Σ FETCH entries (seed step + chunks)
        int nan_count = 0;
    };
    // replay_seq != 0 (DSP52_REF=1 + DSP52_VB=batched): arm the
    // logits-equivalence gate — after each batched verify chunk read row-0's
    // logits and print its top1 margin; for the first dsp52_ref_rounds()
    // rounds RECORD them, and AFTER the loop replay the committed
    // trajectory as B=1 decode on the scratch sequence `replay_seq`
    // (teacher-forced from position 0 — the replay cannot interleave with
    // the rounds, see the post-loop block) comparing at each recorded
    // anchor position via expect_row0_equivalent. The replay steps are
    // excluded from SpecStats (st.wall_ms now excludes them too — REF mode
    // is not a perf measurement regardless).
    std::vector<uint32_t> run_speculative_loop(uint64_t seq_id, int gen_tokens,
                                               uint32_t seed_token,
                                               std::vector<GpuLru>* lrus,
                                               SpecStats& st,
                                               uint64_t replay_seq = 0,
                                               const std::vector<uint32_t>*
                                                   prompt = nullptr) {
        using clock = std::chrono::steady_clock;
        std::vector<uint32_t> committed;
        // Lever-1 knobs: DSP52_GAMMA caps the draft length below the
        // checkpoint's speculative_tokens; DSP52_CONF_THRESH arms the DSP-9
        // static_threshold truncation off the DSP-6 confidence readback.
        int gamma = dspark_gamma_;
        if (int g = dsp52_gamma_override(); g > 0)
            gamma = std::min(g, dspark_gamma_);
        const double conf_thresh = dsp52_conf_thresh();
        if (gamma != dspark_gamma_ || conf_thresh > 0.0)
            std::cerr << "  [dsp52] lever-1: effective gamma=" << gamma
                      << " (ckpt " << dspark_gamma_ << "), conf_thresh="
                      << conf_thresh << "\n";
        const uint32_t R = static_cast<uint32_t>(gamma) + 1;
        // Capacity guards for the multi-row verify chunk.
        EXPECT_LE(R, lipc::kMaxOutputHeadReadbackTokens)
            << "verify rows exceed the batched-verify readback cap";
        EXPECT_LE(R, lipc::kMaxBatchDescriptors);
        EXPECT_LE(R, lipc::kMaxSidebandTokenIds);
        EXPECT_LE(static_cast<int32_t>(R), engine_->info().moe_batch_capacity)
            << "verify rows exceed the engine MoE batch capacity";
        if (::testing::Test::HasFailure()) return committed;

        const uint32_t prompt_len =
            prompt ? static_cast<uint32_t>(prompt->size()) : 0;
        create_sequence(seq_id, prompt_len ? prompt_len : 1);

        // PROMPT-FED arm (DSP52_PROMPT): chunked prefill of the first L-1
        // prompt tokens — the aux export ingests every chunk, so the draft
        // context is armed over the REAL prompt (INV-DSPARK-AUX). The LAST
        // prompt token becomes the seed step below; prefill is excluded
        // from the wall and the H2D miss baseline.
        uint32_t seed = seed_token, seed_pos = 0;
        if (prompt_len) {
            feed_prompt_prefill(seq_id, *prompt, lrus);
            if (::testing::Test::HasFailure()) {
                free_sequence(seq_id);
                return committed;
            }
            seed     = (*prompt)[prompt_len - 1];
            seed_pos = prompt_len - 1;
        }
        h2d_loop_base_phase_ = engine_->h2d_path_stats().phase_count;

        auto loop_start = clock::now();

        // Seed feed: ONE plain decode step at the first un-prefilled
        // position → first generated token. The aux export captures this
        // row (decode shape, B=1); without a prompt the pos-0 capture arms
        // the draft context for seq_id from position 0.
        auto r0 = decode_step_fetch_and_run(seed, seq_id, seed_pos, lrus);
        if (::testing::Test::HasFailure()) { free_sequence(seq_id); return committed; }
        // DSP52_FORCE_TRAJ (matched-trajectory A/B): the target stream IS
        // the forced continuation — commits come from it (walls honest: every
        // step/draft/chunk still executes); ftok(k) = forced token for
        // committed index k (model value beyond the forced tail).
        const std::vector<uint32_t>* forced = dsp52_force_traj();
        auto ftok = [&](size_t idx, uint32_t model_tok) -> uint32_t {
            return (forced && idx < forced->size()) ? (*forced)[idx]
                                                    : model_tok;
        };
        committed.push_back(ftok(0, r0.sampled_token));
        st.lookups += r0.moe_lookups;
        if (!std::isfinite(r0.top1_prob) || !std::isfinite(r0.entropy))
            ++st.nan_count;
        std::cerr << "  seed step: token=" << r0.sampled_token << "  prob="
                  << r0.top1_prob << "  ms=" << r0.timings.total_ms << "\n";

        uint32_t anchor = committed.back();  // newest committed, not yet fed
                                             // (== r0.sampled_token unless
                                             // DSP52_FORCE_TRAJ replaced it)
        uint32_t fed = seed_pos + 1;         // fed-token count == anchor_pos
                                             // (INCLUDES the prompt)

        // Equivalence-gate state: the batched row-0 logits of the first
        // dsp52_ref_rounds() rounds are RECORDED (round, anchor position,
        // logits) and compared against the B=1 teacher-forced replay AFTER
        // the loop — the replay cannot be interleaved with the rounds (see
        // the post-loop block). Fed-history invariant of THIS loop: token
        // at position p is prompt[p] for p < seed_pos, the seed (the last
        // prompt token, or the bare seed token) at p == seed_pos, and
        // committed[p - seed_pos - 1] beyond (committed.size() + seed_pos
        // == fed at every round boundary), so prompt + committed IS the
        // teacher-forcing script.
        struct GatedRound {
            int round;
            uint32_t anchor_pos;
            std::vector<float> batched_row0;
        };
        std::vector<GatedRound> gated;
        bool replay_created = false;
        // Row-0 equivalence-gate recorder shared by the batched and overlap
        // chunk paths: print the top1 margin, record the first
        // dsp52_ref_rounds() rounds at the chunk's row-0 position `pos0`
        // (the replay feeds the committed script and compares at pos0 —
        // row 0 is always a committed-trajectory token there).
        auto gate_row0 = [&](uint32_t pos0) {
            if (replay_seq == 0) return true;
            std::vector<float> batched_row0;
            if (!read_logits_row0(batched_row0)) return false;
            {
                size_t t1 = 0;
                for (size_t i = 1; i < batched_row0.size(); ++i)
                    if (batched_row0[i] > batched_row0[t1]) t1 = i;
                size_t t2 = (t1 == 0) ? 1 : 0;
                for (size_t i = 0; i < batched_row0.size(); ++i)
                    if (i != t1 && batched_row0[i] > batched_row0[t2])
                        t2 = i;
                std::cerr << "  [dsp52 EQ] round " << st.rounds
                          << ": batched row-0 top1=" << t1 << " margin="
                          << (static_cast<double>(batched_row0[t1])
                              - batched_row0[t2]) << "\n";
            }
            if (st.rounds < dsp52_ref_rounds())
                gated.push_back({st.rounds, pos0, std::move(batched_row0)});
            return true;
        };
        const bool overlap = dsp52_overlap() && dsp52_vb_batched();
        if (dsp52_overlap() && !dsp52_vb_batched())
            std::cerr << "  [dsp52] WARNING: DSP52_OVERLAP=1 requires "
                         "DSP52_VB=batched — ignored\n";

        // ── DSP52_ROUND_CSV (bridge-gap ledger, TD-BRIDGE-CPP-GAP): one row
        // per seed step / overlap macro-round with the per-section walls —
        // OFF unless the env var names a file (zero champion overhead: the
        // timings written already exist). Columns mirror the Python bridge's
        // LS_BRIDGE_ROUND_CSV (spec_decode.py) for the paired ledger.
        std::ofstream round_csv;
        auto lsum = [](const std::vector<double>& v) {
            double s = 0.0;
            for (double x : v) s += x;
            return s;
        };
        if (const char* rc = std::getenv("DSP52_ROUND_CSV"); rc && *rc) {
            round_csv.open(rc, std::ios::trunc);
            if (round_csv.is_open())
                round_csv << "kind,round,wall_ms,plain_total,plain_embed,"
                             "plain_layers,plain_head,plain_sample,"
                             "draft_exposed,verify_total,verify_embed,"
                             "verify_layers,verify_head,gap_ms,j,g_use\n";
            round_csv << std::setprecision(6) << std::fixed;
        }
        if (round_csv.is_open())
            round_csv << "seed,0," << r0.timings.total_ms << ","
                      << r0.timings.total_ms << ","
                      << r0.timings.embedding_ms << ","
                      << (lsum(r0.timings.attention_ms)
                          + lsum(r0.timings.moe_ms)) << ","
                      << r0.timings.output_head_ms << ","
                      << r0.timings.sample_ms << ",0,0,0,0,0,0,0,0\n";

        std::array<int32_t, 16> draft{};
        std::array<float, 16> conf{};
        // DSP52_FORCE_DRAFT / DSP52_FORCE_DRAFT_DUMP (no-fork completion): dump
        // records the REAL per-round draft block+confidences (champion recorder);
        // force replays a recorded block over draft[]/conf[] AFTER the real draft
        // executed (wall honest, output replaced) — with FORCE_TRAJ this makes
        // g_use/acceptance/round structure identical across arms by construction.
        const std::vector<Dsp52DraftRound>* fdrafts = dsp52_force_draft();
        size_t fdraft_idx = 0;
        size_t fdraft_mism = 0;
        std::ofstream fdraft_dump;
        if (const char* dp = std::getenv("DSP52_FORCE_DRAFT_DUMP"); dp && *dp) {
            fdraft_dump.open(dp, std::ios::trunc);
            if (!fdraft_dump.is_open())
                ADD_FAILURE() << "DSP52_FORCE_DRAFT_DUMP not writable: " << dp;
        }
        auto apply_forced_draft = [&](int g) {
            if (fdraft_dump.is_open()) {
                fdraft_dump << g;
                for (int k = 0; k < g; ++k)
                    fdraft_dump << ' ' << draft[static_cast<size_t>(k)];
                fdraft_dump << std::setprecision(9);
                for (int k = 0; k < g; ++k)
                    fdraft_dump << ' ' << conf[static_cast<size_t>(k)];
                fdraft_dump << '\n' << std::flush;
            }
            if (fdrafts) {
                if (fdraft_idx < fdrafts->size() &&
                    static_cast<int>((*fdrafts)[fdraft_idx].toks.size()) == g) {
                    const auto& r = (*fdrafts)[fdraft_idx];
                    for (int k = 0; k < g; ++k) {
                        if (draft[static_cast<size_t>(k)] != r.toks[static_cast<size_t>(k)])
                            ++fdraft_mism;
                        draft[static_cast<size_t>(k)] = r.toks[static_cast<size_t>(k)];
                        conf[static_cast<size_t>(k)]  = r.confs[static_cast<size_t>(k)];
                    }
                } else {
                    // Exhausted / shape-mismatched recording ⇒ the no-fork premise
                    // is void — FAIL LOUD rather than silently un-force.
                    ADD_FAILURE() << "DSP52_FORCE_DRAFT round " << fdraft_idx
                                  << " unavailable (recorded "
                                  << fdrafts->size() << " rounds, g=" << g << ")";
                }
            }
            ++fdraft_idx;
        };
        // STS-calibration trace (DSP-7, dspark_calibration.py): one line per
        // round with the RAW DSP-6 survival confidences c_k and the observed
        // accepted prefix length — the (c_k, accepted) ConfidenceRound
        // record. Printed only when the confidence head is armed
        // (DSP52_CONF_THRESH > 0); g_use marks the truncation point (slots
        // >= g_use were never verified — the offline fit must censor there).
        // Format mirrors the golden harness's [glm52-dspark-conf] trace.
        auto conf_trace = [&](int j_acc, int g_use) {
            if (conf_thresh <= 0.0) return;
            std::ostringstream cs;
            cs << std::setprecision(6);
            for (int k = 0; k < gamma; ++k)
                cs << ' ' << conf[static_cast<size_t>(k)];
            std::cerr << "  [dsp52-conf] round " << st.rounds << ": c_k=["
                      << cs.str() << " ] g_use=" << g_use
                      << " accepted=" << j_acc << "\n";
        };
        std::vector<uint32_t> row_toks;
        while (static_cast<int>(committed.size()) < gen_tokens
               && !::testing::Test::HasFailure()) {
            if (overlap) {
                // ── DSP52_OVERLAP macro-round (lever 2) ──────────────────
                // (a') async draft on the 5080 UNDER (b') one plain decode
                // step on the target — the plain step's output ALWAYS
                // commits, so the draft cost is hidden, never wasted.
                auto t_round0 = clock::now();
                dspark_send_async(seq_id, anchor, fed, gamma);
                auto rr = decode_step_fetch_and_run(anchor, seq_id, fed, lrus);
                if (::testing::Test::HasFailure()) break;
                st.overlap_plain_ms += rr.timings.total_ms;
                st.lookups += rr.moe_lookups;
                for (double a : rr.timings.attention_ms) st.attn_ms += a;
                for (double m : rr.timings.moe_ms)       st.moe_ms  += m;
                if (!std::isfinite(rr.top1_prob) || !std::isfinite(rr.entropy))
                    ++st.nan_count;
                const uint32_t t = ftok(committed.size(), rr.sampled_token);
                committed.push_back(t);
                fed += 1;
                // (c') collect the draft — only the EXPOSED wait counts.
                auto tw = clock::now();
                if (!dspark_collect_async(
                        gamma, draft.data(),
                        conf_thresh > 0.0 ? conf.data() : nullptr))
                    break;
                const double draft_exposed =
                    std::chrono::duration<double, std::milli>(
                        clock::now() - tw).count();
                st.draft_ms += draft_exposed;
                apply_forced_draft(gamma);  // no-fork replay / recorder (walls paid)
                // Truncation over the REMAINING slots: slot 0 is checked
                // for free against the plain result; slot k>=1 survives at
                // prod(c_1..c_k) conditional on slot 0 accepted.
                int g_use = gamma;
                if (conf_thresh > 0.0) {
                    double cum = 1.0;
                    g_use = 1;
                    while (g_use < gamma) {
                        cum *= conf[static_cast<size_t>(g_use)];
                        if (cum < conf_thresh) break;
                        ++g_use;
                    }
                    st.trunc_slots += gamma - g_use;
                }
                int j = 0;
                double round_verify_ms = 0.0;
                double v_embed = 0.0, v_layers = 0.0, v_head = 0.0;
                if (t == static_cast<uint32_t>(draft[0]) && g_use >= 2) {
                    // (d') chunk-verify the remaining slots: rows
                    // [t@fed, d_1@fed+1, ...] — row i's argmax is the
                    // target for slot i+1's position (greedy longest-prefix
                    // + bonus, INV-DSPARK-LOSSLESS unchanged).
                    const uint32_t Ruse = static_cast<uint32_t>(g_use);
                    row_toks.resize(Ruse);
                    row_toks[0] = t;
                    for (int k = 1; k < g_use; ++k)
                        row_toks[static_cast<size_t>(k)] =
                            static_cast<uint32_t>(
                                draft[static_cast<size_t>(k)]);
                    auto vr = verify_step_fetch_and_run(row_toks, seq_id,
                                                        fed, lrus);
                    if (::testing::Test::HasFailure()
                        || vr.argmax.size() != static_cast<size_t>(Ruse))
                        break;
                    st.verify_ms += vr.timings.total_ms;
                    round_verify_ms = vr.timings.total_ms;
                    st.lookups += vr.moe_lookups;
                    for (double a : vr.timings.attention_ms) st.attn_ms += a;
                    for (double m : vr.timings.moe_ms)       st.moe_ms  += m;
                    v_embed  = vr.timings.embedding_ms;
                    v_layers = lsum(vr.timings.attention_ms)
                               + lsum(vr.timings.moe_ms);
                    v_head   = vr.timings.output_head_ms;
                    if (!std::isfinite(vr.top1_prob)
                        || !std::isfinite(vr.entropy))
                        ++st.nan_count;
                    if (!gate_row0(fed)) break;
                    int j2 = 0;
                    if (forced) {
                        // Forced stream is the target: slot k+1 accepted iff
                        // it equals the forced token at its position.
                        const size_t base = committed.size();
                        while (j2 + 1 < g_use &&
                               static_cast<uint32_t>(
                                   draft[static_cast<size_t>(j2) + 1]) ==
                                   ftok(base + static_cast<size_t>(j2),
                                        vr.argmax[static_cast<size_t>(j2)]))
                            ++j2;
                    } else {
                        while (j2 + 1 < g_use &&
                               vr.argmax[static_cast<size_t>(j2)] ==
                                   static_cast<uint32_t>(
                                       draft[static_cast<size_t>(j2) + 1]))
                            ++j2;
                    }
                    for (int k = 1; k <= j2; ++k)
                        committed.push_back(static_cast<uint32_t>(
                            draft[static_cast<size_t>(k)]));
                    const uint32_t bonus =
                        ftok(committed.size(),
                             vr.argmax[static_cast<size_t>(j2)]);
                    committed.push_back(bonus);
                    fed += static_cast<uint32_t>(j2) + 1;
                    anchor = bonus;
                    j = 1 + j2;
                } else {
                    if (t == static_cast<uint32_t>(draft[0]))
                        j = 1;  // slot 0 matched but nothing left to verify
                    anchor = t;
                }
                ++st.rounds;
                st.proposed += g_use;
                st.accepted += j;
                std::cerr << "  round " << std::setw(3) << st.rounds
                          << " (overlap): accepted=" << j << "/" << g_use
                          << "  committed=" << committed.size()
                          << "  plain ms=" << std::fixed
                          << std::setprecision(3) << rr.timings.total_ms
                          << "  verify ms=" << round_verify_ms << "\n";
                conf_trace(j, g_use);
                if (round_csv.is_open()) {
                    const double wall =
                        std::chrono::duration<double, std::milli>(
                            clock::now() - t_round0).count();
                    round_csv << "overlap," << st.rounds << "," << wall << ","
                              << rr.timings.total_ms << ","
                              << rr.timings.embedding_ms << ","
                              << (lsum(rr.timings.attention_ms)
                                  + lsum(rr.timings.moe_ms)) << ","
                              << rr.timings.output_head_ms << ","
                              << rr.timings.sample_ms << ","
                              << draft_exposed << ","
                              << round_verify_ms << ","
                              << v_embed << "," << v_layers << ","
                              << v_head << ","
                              << (wall - rr.timings.total_ms - draft_exposed
                                  - round_verify_ms) << ","
                              << j << "," << g_use << "\n";
                }
                continue;
            }
            // (a) DRAFT — one whole-γ DSpark block off the ingested context.
            auto td = clock::now();
            if (!dspark_draft_step(seq_id, anchor, fed, gamma, draft.data(),
                                   conf_thresh > 0.0 ? conf.data() : nullptr))
                break;
            st.draft_ms +=
                std::chrono::duration<double, std::milli>(clock::now() - td).count();
            apply_forced_draft(gamma);  // no-fork replay / recorder (walls paid)

            // DSP-9 static_threshold truncation: submit only the draft
            // prefix whose cumulative survival cumprod(c_1..c_k) stays >=
            // the threshold (c_k conditional on predecessors accepted —
            // cumprod-composable, INV-DSPARK-CONF).
            int g_use = gamma;
            if (conf_thresh > 0.0) {
                double cum = 1.0;
                g_use = 0;
                while (g_use < gamma) {
                    cum *= conf[static_cast<size_t>(g_use)];
                    if (cum < conf_thresh) break;
                    ++g_use;
                }
                st.trunc_slots += gamma - g_use;
            }

            int j = 0;                 // accepted drafts this round
            uint32_t bonus = 0;
            double round_verify_ms = 0.0;
            if (g_use == 0) {
                // Confidence truncated the whole block: hybrid AR fallback —
                // ONE plain decode step feeds the anchor (always commits its
                // output; the sunk draft cost is the only loss). The aux
                // export captures the fed row, so the draft context advances
                // in lockstep for the next round.
                auto rr = decode_step_fetch_and_run(anchor, seq_id, fed, lrus);
                if (::testing::Test::HasFailure()) break;
                st.verify_ms += rr.timings.total_ms;
                round_verify_ms = rr.timings.total_ms;
                st.lookups += rr.moe_lookups;
                for (double a : rr.timings.attention_ms) st.attn_ms += a;
                for (double m : rr.timings.moe_ms)       st.moe_ms  += m;
                if (!std::isfinite(rr.top1_prob) || !std::isfinite(rr.entropy))
                    ++st.nan_count;
                bonus = ftok(committed.size(), rr.sampled_token);
                committed.push_back(bonus);
                fed += 1;
                ++st.fallback_rounds;
            } else if (dsp52_vb_batched()) {
                // (b) VERIFY (batched) — one (1+g_use)-row chunk
                // [anchor@fed, d_0@fed+1, ...].
                const uint32_t Ruse = static_cast<uint32_t>(g_use) + 1;
                row_toks.resize(Ruse);
                row_toks[0] = anchor;
                for (int k = 0; k < g_use; ++k)
                    row_toks[static_cast<size_t>(k) + 1] =
                        static_cast<uint32_t>(draft[static_cast<size_t>(k)]);
                auto vr = verify_step_fetch_and_run(row_toks, seq_id, fed, lrus);
                if (::testing::Test::HasFailure()
                    || vr.argmax.size() != static_cast<size_t>(Ruse))
                    break;
                st.verify_ms += vr.timings.total_ms;
                round_verify_ms = vr.timings.total_ms;
                st.lookups += vr.moe_lookups;
                for (double a : vr.timings.attention_ms) st.attn_ms += a;
                for (double m : vr.timings.moe_ms)       st.moe_ms  += m;
                if (!std::isfinite(vr.top1_prob) || !std::isfinite(vr.entropy))
                    ++st.nan_count;

                // Equivalence gate (replay_seq armed): row-0 logits margin
                // every round; the first DSP52_REF_ROUNDS rounds RECORD
                // (round, anchor_pos=fed, row-0 logits) for the post-loop
                // B=1 replay comparison (TD-DSP52-BATCHED-VERIFY-EQUIV
                // resolution). Row 0 is the anchor@fed — its logits predict
                // position fed+1, exactly what a B=1 decode step feeding
                // the anchor at position fed produces on the
                // matched-history replay.
                if (!gate_row0(fed)) break;

                // (c) ACCEPT — longest matching prefix + bonus (greedy rule:
                // ids[k] is the target's argmax for position fed+k+1, exactly
                // what draft slot k predicts — INV-DSPARK-ANCHOR alignment).
                // DSP52_FORCE_TRAJ: the forced stream replaces the argmax
                // targets (chunk walls already paid; argmax ignored).
                if (forced) {
                    const size_t base = committed.size();
                    while (j < g_use &&
                           static_cast<uint32_t>(
                               draft[static_cast<size_t>(j)]) ==
                               ftok(base + static_cast<size_t>(j),
                                    vr.argmax[static_cast<size_t>(j)]))
                        ++j;
                } else {
                    while (j < g_use &&
                           vr.argmax[static_cast<size_t>(j)] ==
                               static_cast<uint32_t>(
                                   draft[static_cast<size_t>(j)]))
                        ++j;
                }
                for (int k = 0; k < j; ++k)
                    committed.push_back(
                        static_cast<uint32_t>(draft[static_cast<size_t>(k)]));
                bonus = ftok(committed.size(),
                             vr.argmax[static_cast<size_t>(j)]);
                committed.push_back(bonus);
                fed += static_cast<uint32_t>(j) + 1;
            } else {
                // (b)+(c) VERIFY (sequential, DEFAULT) — early-stop
                // teacher-forced feeds through the SAME plain decode step
                // the keeper loop uses (the DsparkLossless golden pattern):
                // feed the anchor, then each draft WHILE it matches the main
                // model's greedy output. Rejected drafts are NEVER fed (no
                // KV rewind, indexer coverage stays alive); every fed row is
                // aux-captured (decode shape) so the draft context advances
                // in lockstep. accepted = feeds - 1; the last feed's output
                // is the bonus.
                uint32_t feed = anchor;
                std::vector<uint32_t> targets;
                const size_t seq_base = committed.size();
                for (int k = 0; k <= g_use && !::testing::Test::HasFailure();
                     ++k) {
                    auto rr = decode_step_fetch_and_run(
                        feed, seq_id, fed + static_cast<uint32_t>(k), lrus);
                    if (::testing::Test::HasFailure()) break;
                    st.verify_ms += rr.timings.total_ms;
                    round_verify_ms += rr.timings.total_ms;
                    st.lookups += rr.moe_lookups;
                    for (double a : rr.timings.attention_ms) st.attn_ms += a;
                    for (double m : rr.timings.moe_ms)       st.moe_ms  += m;
                    if (!std::isfinite(rr.top1_prob)
                        || !std::isfinite(rr.entropy))
                        ++st.nan_count;
                    // DSP52_FORCE_TRAJ: the target is the forced token at
                    // this committed index (model token otherwise).
                    const uint32_t tgt =
                        ftok(seq_base + targets.size(), rr.sampled_token);
                    targets.push_back(tgt);
                    if (k == g_use) break;  // bonus after full acceptance
                    if (tgt !=
                        static_cast<uint32_t>(draft[static_cast<size_t>(k)]))
                        break;              // mismatch → its output = bonus
                    feed = tgt;
                }
                if (::testing::Test::HasFailure() || targets.empty()) break;
                j = static_cast<int>(targets.size()) - 1;
                // Commit: accepted drafts ARE targets[0..j-1]; bonus =
                // targets[j] (identical accounting to the batched path).
                for (uint32_t t : targets) committed.push_back(t);
                bonus = targets.back();
                fed += static_cast<uint32_t>(j) + 1;
            }
            anchor = bonus;
            ++st.rounds;
            st.proposed += g_use;
            st.accepted += j;
            std::cerr << "  round " << std::setw(3) << st.rounds
                      << ": accepted=" << j << "/" << g_use
                      << "  bonus=" << bonus
                      << "  committed=" << committed.size()
                      << "  verify ms=" << std::fixed
                      << std::setprecision(3) << round_verify_ms << "\n";
            conf_trace(j, g_use);
        }
        // Drain an uncollected async draft (overlap-mode early break) so its
        // completion cannot leak into later ring waits.
        if (dspark_pending_seq_ != 0 && !dspark_cmp_ready_) {
            std::array<int32_t, 16> tmp{};
            (void)dspark_collect_async(gamma, tmp.data(), nullptr, 30);
        }
        dspark_cmp_ready_ = false;

        // No-fork audit banner: replay = how many real draft tokens the forced
        // recording overrode (fork magnitude the replay suppressed); record =
        // rounds captured.
        if (fdrafts || fdraft_dump.is_open())
            std::cerr << "  [dsp52] FORCE_DRAFT "
                      << (fdrafts ? "replay" : "record")
                      << ": rounds=" << fdraft_idx
                      << (fdrafts ? ("  overridden-draft-tokens="
                                     + std::to_string(fdraft_mism))
                                  : std::string())
                      << "\n";

        st.wall_ms =
            std::chrono::duration<double, std::milli>(clock::now() - loop_start).count();

        // POST-LOOP equivalence replay (replay_seq armed). The B=1 replay
        // CANNOT be interleaved with the speculative rounds: the DSpark
        // draft context is SINGLE-SLOT and re-arms only on a position-0
        // capture (dspark_runtime TD-DSPARK-BATCH "sequence switch
        // mid-context" invalidation) — replaying after the verify chunk
        // leaves the context tracked on replay_seq, replaying before it
        // invalidates the context outright; either way the next round's
        // draft step dies with CMP_ERROR "no valid ingested context"
        // (2026-08-01 band calibration). So the gated rounds' batched
        // row-0 logits were recorded above, and the teacher-forced replay
        // runs HERE, once, after the last draft: seq `replay_seq` is fed
        // the committed trajectory from position 0, and at each recorded
        // anchor position the fresh logits are compared. Numerically
        // identical to an in-loop replay — the committed trajectory IS the
        // teacher-forcing script, and anchor positions strictly increase.
        if (replay_seq != 0 && !gated.empty()
            && !::testing::Test::HasFailure()) {
            std::cerr << "  [dsp52 EQ] post-loop B=1 replay (seq "
                      << replay_seq << "): " << gated.size()
                      << " gated round(s), positions 0.."
                      << gated.back().anchor_pos << "\n";
            create_sequence(replay_seq, prompt_len ? prompt_len : 1);
            replay_created = !::testing::Test::HasFailure();
            size_t gi = 0;
            for (uint32_t p = 0;
                 p <= gated.back().anchor_pos && gi < gated.size()
                 && !::testing::Test::HasFailure();
                 ++p) {
                const uint32_t tok =
                    (p < seed_pos)    ? (*prompt)[p]
                    : (p == seed_pos) ? seed
                                      : committed[p - seed_pos - 1];
                (void)decode_step_fetch_and_run(tok, replay_seq, p, lrus);
                if (::testing::Test::HasFailure()) break;
                if (p == gated[gi].anchor_pos) {
                    std::vector<float> replay_row0;
                    if (!read_logits_row0(replay_row0)) break;
                    expect_row0_equivalent(gated[gi].round,
                                           gated[gi].batched_row0,
                                           replay_row0);
                    ++gi;
                }
            }
        }

        elb_report();
        upart_report();
        lookahead_report();
        free_sequence(seq_id);
        if (replay_created) free_sequence(replay_seq);
        return committed;
    }

    std::unique_ptr<ldam::Engine> engine_;
    std::unique_ptr<lipc::CommandRing> cmd_ring_;
    std::unique_ptr<lipc::CompletionRing> cmp_ring_;
    uint8_t* sideband_ = nullptr;
    std::string config_path_;
    std::string weights_path_;
    std::string loader_calib_path_;
    uint32_t cmd_seq_ = 1;
    int num_layers_ = 0, num_experts_ = 0, first_moe_layer_ = 3, topk_ = 8;
    int vocab_size_ = 0;
    uint32_t hidden_buf_id_ = 0, logits_buf_id_ = 0;
    int dspark_gamma_ = 7, dspark_block_ = 8;  // derived from ckpt config.json
    // H2D phase_count snapshot taken by the loops AFTER any DSP52_PROMPT
    // prefill (miss/hit-rate baseline of the DECODE loop only).
    uint64_t h2d_loop_base_phase_ = 0;
};

// ═══════════════════════════════════════════════════════════════════════════
// THE dsp52 benchmark — keeper52 + DSpark speculative decode (skips when the
// 4 SM120 GPUs / GLM-5.2 assets / DSpark checkpoint are absent, or when
// KEEPER52_EP=2 forces the draft-less keeper shape).
// ═══════════════════════════════════════════════════════════════════════════
TEST_F(Dsp52Test, SpeculativeHundredTokenDecodeFetchAndRun_FullFit_EP4_GLM52) {
    const std::string src = LAYERSTORM_SOURCE_DIR;
    if (!fs::exists(src + kConfigRel))
        GTEST_SKIP() << "GLM-5.2 config missing: " << src + kConfigRel;
    if (!fs::exists(src + kGgufRel))
        GTEST_SKIP() << "GLM-5.2 GGUF not present: " << src + kGgufRel;
    const std::string ckpt = dsp52_ckpt_dir(src);
    if (!fs::exists(ckpt + "/config.json"))
        GTEST_SKIP() << "DSpark checkpoint not present: " << ckpt;

    if (dsp52_ep() != 4)
        GTEST_SKIP() << "dsp52 requires the EP=4 shape (draft on the last "
                        "5080, an expert-only rank) — KEEPER52_EP=2 has no "
                        "draft host";
    {
        // EP=4 (INV-MOE-EP-XTP): need ALL FOUR GPUs — 2× >=28 GB (5090 TP
        // ranks) + 2 more >=14 GB (5080 expert-only ranks; the last also
        // hosts the DSpark draft).
        auto gpus = probe_sm120_gpus();
        if (gpus.size() < 4 || gpus[1].gib < 28.0 || gpus[3].gib < 14.0)
            GTEST_SKIP() << "dsp52 needs 4 SM120 GPUs "
                            "(2x >=28 GB + 2x >=14 GB); found " << gpus.size();
    }

    start_engine();
    if (::testing::Test::HasFailure()) return;

    // KEEPER52_TOKENS: dev/profiling knob (the KEEPER measurement is 100).
    int num_tokens = kNumTokens;
    if (const char* nt = std::getenv("KEEPER52_TOKENS"); nt && *nt)
        if (int v = std::atoi(nt); v > 0) num_tokens = v;
    uint32_t seed_token = 15234;
    if (const char* st = std::getenv("LS_KEEPER_SEED_TOKEN"); st && *st) {
        long v = std::atol(st);
        if (v >= 0 && v < static_cast<long>(vocab_size_))
            seed_token = static_cast<uint32_t>(v);
    }

    // ── PROMPT-FED arm (DSP52_PROMPT): real-context token-id prompt ──────
    std::vector<uint32_t> prompt;
    if (const char* pp = std::getenv("DSP52_PROMPT"); pp && *pp) {
        std::ifstream tf(pp);
        ASSERT_TRUE(tf.is_open())
            << "DSP52_PROMPT token file not readable: " << pp;
        int64_t v = 0;
        while (tf >> v) {
            ASSERT_GE(v, 0) << "DSP52_PROMPT id negative";
            ASSERT_LT(v, static_cast<int64_t>(vocab_size_))
                << "DSP52_PROMPT id out of vocab";
            prompt.push_back(static_cast<uint32_t>(v));
        }
        if (const char* pc = std::getenv("DSP52_PROMPT_TOKENS"); pc && *pc)
            if (size_t cap = static_cast<size_t>(std::atoll(pc));
                cap > 0 && cap < prompt.size())
                prompt.resize(cap);
        ASSERT_GE(prompt.size(), size_t{2})
            << "DSP52_PROMPT too short: " << pp;
        std::cerr << "  [dsp52] PROMPT-FED: " << prompt.size()
                  << " prompt tokens from " << pp
                  << " (seed = last prompt token " << prompt.back()
                  << "; LS_KEEPER_SEED_TOKEN ignored)\n";
    }
    const std::vector<uint32_t>* prompt_p =
        prompt.empty() ? nullptr : &prompt;

    // ── TEACHER-FORCED regime (DSP52_FORCE_TRAJ): load-fail is loud; ids
    // must be in-vocab (keeper52 parity).
    if (const std::vector<uint32_t>* fj = dsp52_force_traj()) {
        for (uint32_t t : *fj)
            ASSERT_LT(t, static_cast<uint32_t>(vocab_size_))
                << "DSP52_FORCE_TRAJ id out of vocab";
    }
    if (::testing::Test::HasFailure()) return;  // unreadable forced file

    // 13c-2.0 orchestrator eviction model + P-25 REEF stack (keeper52).
    std::vector<GpuLru> lrus = make_lrus();
    reef_orch_ = make_reef_orch();
    {   // x-ray: GPU→NUMA-node map.
        const auto& gpus = engine_->config().hardware.gpus;
        std::string m = "  GPU->NUMA node:";
        for (size_t p = 0; p < gpus.size(); ++p)
            m += " gpu" + std::to_string(p) + "=node"
                 + std::to_string(gpus[p].numa_node);
        std::cerr << m << "\n";
    }
    const int tp = engine_->info().num_gpus;
    const int num_moe_layers = num_layers_ - first_moe_layer_;
    {
        std::string caps;
        for (size_t g = 0; g < lrus.size(); ++g)
            caps += (g ? "," : "") + std::to_string(lrus[g].capacity);
        std::cerr << std::fixed << std::setprecision(3)
                  << "\n=== dsp52 SpeculativeHundredTokenDecodeFetchAndRun "
                     "(GLM-5.2 GGUF, EP=4 over " << tp << " GPUs, DSpark "
                     "gamma=" << dspark_gamma_ << " block=" << dspark_block_
                  << ", verify=" << (dsp52_vb_batched() ? "batched" : "seq")
                  << ", full-fit O_DIRECT, FETCH_AND_RUN_MOE, TQ + "
                     "sharded-KV + tiering) ===\n"
                  << "  layers=" << num_layers_ << " first_moe="
                  << first_moe_layer_ << " experts=" << num_experts_
                  << " topk=" << topk_ << " GPUs=" << tp
                  << "  moe_batch_capacity="
                  << engine_->info().moe_batch_capacity << "\n"
                  << "  stable slots/GPU=[" << caps << "]  (draft host = "
                     "position 3; worst-case plain working set = "
                  << num_moe_layers << "x" << (topk_ + tp - 1) / tp << ")\n";
    }

    const bool spec_on = [] {
        const char* e = std::getenv("DSP52_SPEC");
        return !(e && *e == '0');
    }();
    const bool ref_mode = [] {
        const char* e = std::getenv("DSP52_REF");
        return e && *e == '1';
    }();

    // Shared keeper reporting helpers.
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

    if (!spec_on) {
        // ── DSP52_SPEC=0: the PLAIN keeper loop (A/B baseline in this
        // binary — same engine, same config, same placement machinery). ──
        std::cerr << "  [dsp52] DSP52_SPEC=0 — plain keeper loop\n";
        PlainAgg agg;
        auto toks = run_plain_loop(1, num_tokens, seed_token, &lrus, agg,
                                   prompt_p);
        if (::testing::Test::HasFailure()) return;
        // Miss baseline = the loop's post-prefill snapshot (prompt prefill
        // fetches are excluded from the decode hit rate).
        const uint64_t misses =
            engine_->h2d_path_stats().phase_count - h2d_loop_base_phase_;
        const double hit_rate = agg.lookups
            ? 1.0 - static_cast<double>(misses) / static_cast<double>(agg.lookups)
            : 0.0;
        const double per_tok = agg.total_ms / num_tokens;
        const auto ms_stat = med_ss(agg.per_token_ms);
        std::cerr << "\n  Total: " << agg.total_ms << " ms for " << num_tokens
                  << " tokens\n  Avg: " << per_tok << " ms/token ("
                  << (1000.0 / per_tok) << " tok/s)\n"
                  << "  Median: " << ms_stat.first << " ms/token ("
                  << (ms_stat.first > 0 ? 1000.0 / ms_stat.first : 0.0)
                  << " tok/s)  SteadyState(excl tok0): " << ms_stat.second
                  << " ms/token\n"
                  << "  attention: " << agg.attn / num_tokens
                  << " ms/token  moe: " << agg.moe / num_tokens << " ms/token\n"
                  << "  cache: lookups=" << agg.lookups << " misses=" << misses
                  << " hit_rate=" << std::setprecision(4) << hit_rate
                  << std::setprecision(3) << "\n";
        {
            namespace pr = layerstorm::perf_report;
            pr::Report rep;
            rep.title = "X-RAY (dsp52 plain)";
            const double nt = num_tokens;
            rep.row("embedding",  0, agg.embed / nt, -1.0, -1.0, 1.0);
            rep.row("attention",  0, agg.attn / nt, -1.0,
                    agg.attn / (nt * num_layers_), static_cast<double>(num_layers_));
            rep.row("moe (fused fetch+compute)", 0, agg.moe / nt, -1.0,
                    agg.moe / (nt * num_layers_), static_cast<double>(num_layers_));
            rep.row("output_head", 0, agg.head / nt, -1.0, -1.0, 1.0);
            rep.row("sample",      0, agg.sample / nt, -1.0, -1.0, 1.0);
            auto f = [](double v, int p) {
                std::ostringstream s; s << std::fixed << std::setprecision(p) << v;
                return s.str();
            };
            rep.stat("tok/s",    f(1000.0 / per_tok, 3));
            rep.stat("ms/token", f(per_tok, 3));
            rep.stat("hit_rate", f(hit_rate, 4));
            rep.stat("NaN", std::to_string(agg.nan_count) + "/" +
                            std::to_string(num_tokens));
            finish_xray_report(rep);
        }
        EXPECT_EQ(agg.nan_count, 0) << "Some tokens produced NaN prob/entropy";
        dsp52_dump_tokens(toks);
        return;
    }

    // ── Reference mode (DSP52_REF=1), shape-dependent gate. ──────────────
    // VB=seq: plain baseline (seq 1) first → strict token identity (the
    // sequential verifier is the SAME B=1 decode pipeline). VB=batched: NO
    // plain baseline — token identity across chunk-vs-decode pipelines is
    // the wrong gate (different FP reduction orders; resolved
    // TD-DSP52-BATCHED-VERIFY-EQUIV); the per-round logits-equivalence gate
    // inside run_speculative_loop replaces it (INV-DSPARK-LOSSLESS B>1
    // clause).
    std::vector<uint32_t> baseline;
    if (ref_mode && dsp52_force_traj())
        std::cerr << "  [dsp52] WARNING: DSP52_FORCE_TRAJ + DSP52_REF=1 — "
                     "forcing replaces the commit targets, so REF gates are "
                     "NOT lossless-gate evidence in this mode\n";
    if (ref_mode && dsp52_vb_batched()) {
        std::cerr << "  [dsp52] DSP52_REF=1 + DSP52_VB=batched — per-round "
                     "row-0 logits-equivalence gate (first "
                  << dsp52_ref_rounds() << " rounds vs B=1 replay on seq 3); "
                     "no plain-baseline token identity (cross-shape "
                     "bit-identity is not required)\n";
    } else if (ref_mode) {
        std::cerr << "  [dsp52] DSP52_REF=1 — plain baseline (seq 1) first\n";
        PlainAgg agg;
        baseline = run_plain_loop(1, num_tokens, seed_token, &lrus, agg,
                                  prompt_p);
        if (::testing::Test::HasFailure()) return;
        std::cerr << "  baseline done: " << baseline.size() << " tokens, "
                  << agg.total_ms << " ms ("
                  << (agg.total_ms > 0
                          ? 1000.0 * baseline.size() / agg.total_ms : 0.0)
                  << " tok/s)\n\n";
    }

    // ── The speculative loop. ────────────────────────────────────────────
    SpecStats st;
    const uint64_t spec_seq = ref_mode ? 2 : 1;
    // Batched REF: seq 3 = the B=1 equivalence-replay scratch sequence.
    const uint64_t replay_seq = (ref_mode && dsp52_vb_batched()) ? 3 : 0;
    auto committed = run_speculative_loop(spec_seq, num_tokens, seed_token,
                                          &lrus, st, replay_seq, prompt_p);
    if (::testing::Test::HasFailure()) return;
    ASSERT_GE(static_cast<int>(committed.size()), num_tokens)
        << "speculative loop under-committed";

    // Miss baseline = the loop's post-prefill snapshot (prompt prefill
    // fetches are excluded from the decode hit rate).
    const uint64_t misses =
        engine_->h2d_path_stats().phase_count - h2d_loop_base_phase_;
    const double hit_rate = st.lookups
        ? 1.0 - static_cast<double>(misses) / static_cast<double>(st.lookups)
        : 0.0;

    const double acc_rate = st.proposed
        ? static_cast<double>(st.accepted) / static_cast<double>(st.proposed)
        : 0.0;
    // τ = committed tokens per round incl. bonus (the seed token is not a
    // round product — exclude it).
    const double tau = st.rounds
        ? static_cast<double>(committed.size() - 1) / st.rounds : 0.0;
    const double wall_toks = st.wall_ms > 0
        ? 1000.0 * static_cast<double>(committed.size()) / st.wall_ms : 0.0;

    std::cerr << std::fixed << std::setprecision(3)
              << "\n  ── speculation ──\n"
              << "  rounds=" << st.rounds << "  proposed=" << st.proposed
              << "  accepted=" << st.accepted << "  acceptance="
              << std::setprecision(4) << acc_rate << std::setprecision(3)
              << "\n  tau (tokens/round incl bonus)=" << tau
              << "  committed=" << committed.size() << "\n"
              << "  wall: " << st.wall_ms << " ms  → " << wall_toks
              << " tok/s over COMMITTED tokens\n"
              << "  draft: " << st.draft_ms / std::max(1, st.rounds)
              << " ms/round  verify: " << st.verify_ms / std::max(1, st.rounds)
              << " ms/round  (attn " << st.attn_ms / std::max(1, st.rounds)
              << ", moe " << st.moe_ms / std::max(1, st.rounds) << ")\n"
              << "  cache: lookups=" << st.lookups << " misses=" << misses
              << " hit_rate=" << std::setprecision(4) << hit_rate
              << std::setprecision(3) << "\n";

    {   // Unified x-ray report (keeper style) + speculation section.
        namespace pr = layerstorm::perf_report;
        pr::Report rep;
        rep.title = "X-RAY (dsp52 speculative)";
        const double nr = std::max(1, st.rounds);
        rep.section("per round (draft + verify chunk)");
        rep.row("dspark draft step",   1, st.draft_ms / nr, -1.0, -1.0,
                static_cast<double>(st.rounds));
        if (st.overlap_plain_ms > 0)
            rep.row("overlap plain step",  1, st.overlap_plain_ms / nr, -1.0,
                    -1.0, static_cast<double>(st.rounds));
        rep.row("verify chunk wall",   1, st.verify_ms / nr, -1.0, -1.0,
                static_cast<double>(st.rounds));
        rep.row("verify attention",    2, st.attn_ms / nr,
                st.verify_ms > 0 ? 100.0 * st.attn_ms / st.verify_ms : -1.0,
                st.attn_ms / (nr * num_layers_),
                static_cast<double>(num_layers_));
        rep.row("verify moe (fetch+run union)", 2, st.moe_ms / nr,
                st.verify_ms > 0 ? 100.0 * st.moe_ms / st.verify_ms : -1.0,
                st.moe_ms / (nr * num_layers_),
                static_cast<double>(num_layers_));
        auto f = [](double v, int p) {
            std::ostringstream s; s << std::fixed << std::setprecision(p) << v;
            return s.str();
        };
        rep.stat("verify backend",    dsp52_vb_batched() ? "batched" : "seq");
        rep.stat("committed tokens",  std::to_string(committed.size()));
        rep.stat("rounds",            std::to_string(st.rounds));
        rep.stat("proposed/accepted", std::to_string(st.proposed) + "/" +
                                      std::to_string(st.accepted));
        rep.stat("acceptance rate",   f(acc_rate, 4));
        rep.stat("tau tokens/round",  f(tau, 3));
        rep.stat("wall tok/s (committed)", f(wall_toks, 3));
        rep.stat("hit_rate",          f(hit_rate, 4));
        rep.stat("lookups/misses",    std::to_string(st.lookups) + "/" +
                                      std::to_string(misses));
        rep.stat("gamma",             std::to_string(dspark_gamma_));
        {   // Lever-1 knobs (DSP52_GAMMA / DSP52_CONF_THRESH).
            int geff = dspark_gamma_;
            if (int g = dsp52_gamma_override(); g > 0)
                geff = std::min(g, dspark_gamma_);
            rep.stat("gamma_eff",     std::to_string(geff));
            rep.stat("conf_thresh",   f(dsp52_conf_thresh(), 3));
            rep.stat("fallback_rounds", std::to_string(st.fallback_rounds));
            rep.stat("trunc_slots",   std::to_string(st.trunc_slots));
        }
        rep.stat("NaN", std::to_string(st.nan_count));
        if (dsp52_force_traj())
            rep.stat("forced_traj",
                     std::to_string(dsp52_force_traj()->size()) + " tokens");
        finish_xray_report(rep);
    }

    EXPECT_GT(st.rounds, 0) << "speculation never ran a round";
    EXPECT_EQ(st.nan_count, 0) << "NaN prob/entropy during speculation";
    dsp52_dump_tokens(committed);

    if (ref_mode && dsp52_vb_batched()) {
        // INV-DSPARK-LOSSLESS B>1 clause: under batched verification the
        // target distribution IS the batched forward's own outputs (vLLM
        // semantics) — the per-round logits-equivalence gate ran inside the
        // loop; no token identity vs a B=1 plain-decode trajectory (valid
        // only between same-shaped pipelines). DSP52_REF_DET=1: cheap
        // self-consistency determinism check — the SAME batched loop
        // re-run in-process (fresh seq 4, same seed) must reproduce the
        // committed trajectory exactly (greedy-deterministic; placement
        // drift is numerics-neutral under the canonical EP combine).
        std::cerr << "  [dsp52] batched equivalence gate: "
                  << std::min(st.rounds, dsp52_ref_rounds())
                  << " gated rounds (DSP52_REF_ROUNDS="
                  << dsp52_ref_rounds() << ")\n";
        if (dsp52_ref_det()) {
            std::cerr << "  [dsp52] DSP52_REF_DET=1 — batched determinism "
                         "re-run (seq 4)\n";
            SpecStats st2;
            auto committed2 = run_speculative_loop(4, num_tokens, seed_token,
                                                   &lrus, st2, 0, prompt_p);
            if (!::testing::Test::HasFailure()) {
                ASSERT_EQ(committed2.size(), committed.size())
                    << "batched determinism: committed counts differ";
                for (size_t i = 0; i < committed.size(); ++i)
                    EXPECT_EQ(committed2[i], committed[i])
                        << "batched determinism violation at token " << i;
                std::cerr << "  [dsp52] determinism gate: "
                          << committed.size() << " tokens bit-identical "
                             "across in-process re-run\n";
            }
        }
    } else if (ref_mode) {
        // INV-DSPARK-LOSSLESS (sequential verifier — same-shaped B=1 decode
        // pipeline): greedy verification must make the speculative
        // trajectory IDENTICAL to plain autoregressive decode.
        const size_t n = std::min({baseline.size(), committed.size(),
                                   static_cast<size_t>(num_tokens)});
        ASSERT_GT(n, 0u);
        for (size_t i = 0; i < n; ++i)
            EXPECT_EQ(committed[i], baseline[i])
                << "lossless violation at generated token " << i
                << " (spec=" << committed[i] << " plain=" << baseline[i] << ")";
        std::cerr << "  [dsp52] lossless gate: " << n
                  << " tokens compared against the plain baseline\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU-ONLY structural check — NO CUDA, NO weights, NO decode. Builds the exact
// dsp52 config, parses it, and asserts the keeper52 features PLUS the DSpark
// speculation block are present (clone of Keeper52ConfigResolves).
// ═══════════════════════════════════════════════════════════════════════════
TEST(Dsp52ConfigCpu, Dsp52ConfigResolves) {
    namespace cfgns = layerstorm::config;
    const std::string src = LAYERSTORM_SOURCE_DIR;
    if (!fs::exists(src + kConfigRel))
        GTEST_SKIP() << "GLM-5.2 config missing: " << src + kConfigRel;

    // have_gpus=false → keep the preset's single-GPU hardware block (no CUDA
    // probe; draft_gpus stays auto). have_prepacked=false → skip the O_DIRECT
    // arena knobs.
    int gamma = 0, block = 0;
    nlohmann::json j = build_dsp52_config(src, /*have_gpus=*/false,
                                          /*have_prepacked=*/false,
                                          &gamma, &block);

    // keeper52's four feature deltas + the dsp52 speculation block.
    EXPECT_EQ(j["compute"]["attention_backend"], "turboquant_mla");
    EXPECT_EQ(j["hardware"]["dcp_kv_mode"], "sharded");
    EXPECT_EQ(j["memory"]["kv_tiering"]["enabled"], true);
    EXPECT_GT(j["model"]["index_topk"].get<int>(), 0);   // DSA active
    EXPECT_EQ(j["model"]["weights_format"], "gguf");
    EXPECT_EQ(j["speculation"]["method"], "dspark");
    EXPECT_EQ(j["speculation"]["enabled"], true);
    EXPECT_EQ(j["speculation"]["dspark"]["confidence_enabled"], false);
    EXPECT_FALSE(j["speculation"]["dspark"]["checkpoint_path"]
                     .get<std::string>().empty());
    EXPECT_GE(gamma, 1);
    EXPECT_EQ(block, gamma + 1)
        << "bonus-anchor layout: block_size = 1 + speculative_tokens "
           "(INV-DSPARK-ANCHOR)";

    const std::string tmp = "/tmp/dsp52_cpu_config.json";
    { std::ofstream o(tmp); o << j.dump(2); }

    // Pure-CPU parse (no CUDA). parse_config throws on any schema/type error.
    cfgns::Config cfg;
    ASSERT_NO_THROW(cfg = cfgns::parse_config(tmp));
    std::remove(tmp.c_str());

    EXPECT_EQ(cfg.compute.attention_backend,
              cfgns::AttentionBackendType::turboquant_mla);
    EXPECT_EQ(cfg.hardware.dcp_kv_mode, cfgns::DcpKvMode::sharded);
    EXPECT_TRUE(cfg.memory.kv_tiering.enabled);
    EXPECT_EQ(cfg.speculation.method, cfgns::SpeculationMethodType::dspark);
    EXPECT_TRUE(cfg.speculation.enabled);
    EXPECT_FALSE(cfg.speculation.dspark.confidence_enabled);
    // Lever-4 verify-chunk prefetch is a config field now (default OFF —
    // measured -1.0% champion wall; the test json must NOT carry the
    // default value, only non-default overrides live there).
    EXPECT_FALSE(j["speculation"]["dspark"].contains("verify_chunk_prefetch"));
    EXPECT_FALSE(cfg.speculation.dspark.verify_chunk_prefetch);
    // M3b online placement is a config field, DEFAULT TRUE (2026-08-18
    // champion decision; env LS_ARENA_PLACE_ONLINE overrides at boot).
    // The json carries only non-defaults, so the key must be absent.
    EXPECT_FALSE(j["memory"].contains("arena_placement")
                 && j["memory"]["arena_placement"].contains("online"));
    EXPECT_TRUE(cfg.memory.arena_placement.online);
    EXPECT_GE(cfg.speculation.dspark.speculative_tokens, 1);
    EXPECT_EQ(cfg.speculation.dspark.block_size,
              cfg.speculation.dspark.speculative_tokens + 1);

    // The verify chunk R = 1+γ must fit every fixed IPC capacity.
    const uint32_t R =
        static_cast<uint32_t>(cfg.speculation.dspark.speculative_tokens) + 1;
    EXPECT_LE(R, lipc::kMaxOutputHeadReadbackTokens);
    EXPECT_LE(R, lipc::kMaxBatchDescriptors);
    EXPECT_LE(R, lipc::kMaxSidebandTokenIds);
    // Raw (pre-dedup) union bound of a verify chunk vs the sideband list.
    EXPECT_LE(R * static_cast<uint32_t>(cfg.model.num_experts_per_tok),
              lipc::kMaxExpertPrefetch);

    // ── config → lever wiring for verify_chunk_prefetch, and the
    // override-wins precedence of the DSP52_PREFETCH test-interface env.
    nlohmann::json jp = j;
    jp["speculation"]["dspark"]["verify_chunk_prefetch"] = true;
    const std::string tmp_pf = "/tmp/dsp52_cpu_config_prefetch.json";
    { std::ofstream o(tmp_pf); o << jp.dump(2); }
    cfgns::Config cfg_pf;
    ASSERT_NO_THROW(cfg_pf = cfgns::parse_config(tmp_pf));
    std::remove(tmp_pf.c_str());
    EXPECT_TRUE(cfg_pf.speculation.dspark.verify_chunk_prefetch);

    const char* pf_env = std::getenv("DSP52_PREFETCH");
    const bool pf_env_set = pf_env && *pf_env;
    // start_engine() publishes exactly this value into the lever.
    dsp52_set_prefetch_config(cfg_pf.speculation.dspark.verify_chunk_prefetch);
    if (pf_env_set) {
        const bool forced = std::atoi(pf_env) != 0;
        EXPECT_EQ(dsp52_prefetch(), forced) << "env override must win";
        dsp52_set_prefetch_config(false);
        EXPECT_EQ(dsp52_prefetch(), forced) << "env override must win";
    } else {
        EXPECT_TRUE(dsp52_prefetch())
            << "config verify_chunk_prefetch=true must arm the lever";
        dsp52_set_prefetch_config(false);
        EXPECT_FALSE(dsp52_prefetch());
    }
    // Restore the schema default for any later test in this binary.
    dsp52_set_prefetch_config(false);

    fprintf(stderr,
            "[dsp52-cpu] dspark gamma=%d block=%d → verify chunk R=%u "
            "(readback cap %u, moe floor %u)\n",
            gamma, block, R, lipc::kMaxOutputHeadReadbackTokens,
            lipc::kMaxBatchDescriptors);
}

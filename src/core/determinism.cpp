// Determinism superflags — registry + apply.  See determinism.h for the
// two-flag hierarchy (run-to-run determinism < reference-trajectory
// identity) and scratchpad/gf3_speed_saga/DETERMINISM_FLAG.md for the
// design note.  KEEP THIS TABLE IN LOCKSTEP with
// python/orchestrator/determinism.py (tests/unit/test_determinism_superflag.py
// asserts equality of the actionable rows; and its tree scan makes an
// unregistered LS_*/LAYERSTORM_* knob a test failure).
//
// PARSE NOTE for the parity test: one registry entry per line, shaped
//   {"ENV", A::kAxis, C::kClass, <forced|nullptr>, K::kKind, <exact|nullptr>, "evidence"},
// A::kRunToRun rows act under FLAG 1 (and flag 2 via the hierarchy);
// A::kReference rows act under FLAG 2 only.

#include "core/determinism.h"

#include <cstdlib>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace layerstorm {
namespace core {
namespace determinism {

namespace {

using A = Axis;
using C = KnobClass;
using K = ConflictKind;

// Shorthand evidence strings shared by whole families.
constexpr const char* kDiagEv =
    "diagnostic/observability only (dump/trace/probe/csv) — reads state, never feeds compute";
constexpr const char* kPlacementEv =
    "placement/residency channel only — combine is placement-invariant under forced "
    "DET-EP-COMBINE (INV-DRIFT-EPCOMBINE); residual cross-placement depth sensitivity is "
    "TD-MOE-EP-XTP-PLACEMENT-DRIFT (open), same-placement reruns unaffected";
constexpr const char* kTransportEv =
    "byte-preserving transport/memory-domicile knob — moves identical bytes, never changes them";

const std::vector<Knob>& table() {
    static const std::vector<Knob> t = {
        // ── kForceOn: determinism ENFORCERS ─────────────────────────────
        {"LAYERSTORM_DETERMINISTIC_REDUCE", A::kRunToRun, C::kForceOn, "1", K::kIfFalsy, nullptr, "INV-DRIFT-DETREDUCE: legacy atomic softmax-denominator path is warp-scheduling-order FP summation -> NOT bit-reproducible run-to-run; fixed-order path is (config default true; env pins either way)"},
        {"LAYERSTORM_DETERMINISTIC_EP_COMBINE", A::kRunToRun, C::kForceOn, "1", K::kIfFalsy, nullptr, "INV-DRIFT-EPCOMBINE / dossier 4b precondition 1: mode-0 combine follows LIVE expert residency -> greedy output not reproducible run-to-run; canonical fixed-slot combine is bit-placement-invariant by construction (schema default false -> MUST be forced)"},
        {"LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION", A::kReference, C::kForceOn, "bf16", K::kIfExact, "fp32", "bf16 is the canonical champion payload (boot.sh + schema default); fp32 payload is placement-invariant too but produces a DIFFERENT trajectory (different rounding sequence) -> forks the reference"},

        // ── kForceOff: trajectory-forking / run-to-run nondeterministic ─
        {"LS_TQ_SPLITKV", A::kReference, C::kForceOff, "0", K::kIfTruthy, nullptr, "P-29 step 9 (glm53_flash.md:555-560): run-to-run bit-identical WITH ITSELF but 327/6999 TF argmax flips vs reference (FP32 reassociation in the split combine); champion legs fork at both rungs — THE worked example of deterministic-but-not-reference-reproducible. DEFAULT ON since P-29 step 14 (OQ-6 grant): the champion recipe rides the split; the CANONICAL numerics route stays unsplit, so this row keeps forcing 0 — flag 2 reproduces the pre-flip trajectory (proven in vivo, P-29 step 14)"},
        {"LS_SNAPMLA_FP8_DECODE", A::kReference, C::kForceOff, "0", K::kIfTruthy, nullptr, "P-29 step 10 stage (b): different numerics profile vs the snapMLA-exact route — +0.0063 nats, 310/6999 flips (glm53_flash.md:585-586); deterministic run-to-run but forks the snapMLA reference trajectory. DEFAULT ON since P-29 step 14 (OQ-7 grant); the tier's canonical numerics stay exact, so this row keeps forcing 0"},
        {"LS_ORCH_SUBGRID_MIDEDGE", A::kReference, C::kForceOff, "0", K::kIfTruthy, nullptr, "dossier 4b precondition 3 / INV-PREFIX-CACHE-1 SUB-GRID clause: a sub-grid mid-edge hit is deterministically NON-identical to an uncached run; TD-PREFIX-MIDEDGE-SUBGRID-CLASS proved NO sub-grid subclass is identity-safe"},
        {"LS_SPEC_GOVERNOR", A::kRunToRun, C::kForceOff, "0", K::kIfTruthy, nullptr, "dossier 4b precondition 4 / TD-GLM52-CHAMPION-GREEDY-NONDET-RUN-TO-RUN: governor picks round shapes from wall-clock EMAs and near-tie argmax follows the shape -> trajectory follows timing noise; literal \"0\" required (unset means ON)"},

        // ── kPinDefault: trajectory-relevant, canonical == code default ─
        {"LS_PINNED_EXACT_WIDTHS", A::kReference, C::kPinDefault, nullptr, K::kIfFalsy, nullptr, "TD-MOE-EP-XTP-PLACEMENT-DRIFT arm (3): =0 moves the GPU-0 expert-zone carve -> decode residency -> measured trajectory divergence at +16; canonical champion state is the default ON"},
        {"LS_MOE_RESIDENT_OVERLAP", A::kReference, C::kPinDefault, nullptr, K::kIfTruthy, nullptr, "default flipped OFF P-29 step 18 (OQ-10 champion A/B: split costs 1.4-1.8% @8k, OFF byte-identical to the champion trajectory at both rungs); =1 re-arms the split, which carries the 2026-09-03 caveat — INV-MOE-OVERLAP contract (2) x+0 bit-identity FALSIFIED when the split genuinely engages on the tp1+extras shape (+2 divergence, TD-MOE-EP-XTP-PLACEMENT-DRIFT); canonical champion state is the default OFF — arming the split is the risk direction"},
        {"LS_INDEXER_REWIND", A::kReference, C::kPinDefault, nullptr, K::kIfFalsy, nullptr, "=0 drops rewound rows to DENSE indexer steps — a different attention read set (selection change), config default true is the measured champion state (INV-DSA-REWIND; dsp52 A/B bit-identical 107-token trajectory)"},
        {"LS_TP_COMBINE_FP32", A::kReference, C::kPinDefault, nullptr, K::kIfTruthy, nullptr, "=1 changes TP-combine accumulation precision (fp32 partials + one post-sum bf16 round) — a different rounding sequence, not a reorder (dcp_executor.cpp:480); default OFF is the reference"},
        {"LS_CHUNK_SMALLM", A::kReference, C::kPinDefault, nullptr, K::kIfFalsy, nullptr, "explicitly NOT bit-identical across the route flip (dcp_executor.cpp:1077 — different K-reduction structure); default ON (489e0941) is the champion reference"},
        {"LS_NO_CHUNK_SMALLM", A::kReference, C::kPinDefault, nullptr, K::kIfTruthy, nullptr, "kill-switch spelling of LS_CHUNK_SMALLM=0 — same non-bit-identical route flip"},
        {"LS_GG_FORCE", A::kReference, C::kPinDefault, nullptr, K::kIfSet, nullptr, "pins the GGUF grouped-INT strategy: 'a different-but-valid reduction order' (cuda_sm120_expert_device.cu:134); Auto is shape-deterministic and is the reference; also the one documented NULL-B-ptr violation path (INVARIANTS.md:387)"},
        {"LS_NO_PERSLOT_ZEROFIX", A::kReference, C::kPinDefault, nullptr, K::kIfTruthy, nullptr, "disables a CORRECTNESS fix: stale per-slot rows from a prior layer/token enter the canonical combine ('a real corruption of the routed output', moe_driver.cpp:885-896)"},
        {"LS_ORCH_NO_SC", A::kReference, C::kPinDefault, nullptr, K::kIfExact, "1", "=1 returns served prefill to the pre-superchunk trajectory (TD-SERVE-SC-TRAJECTORY accepted-divergence class) — a fork from the recipe's reference"},
        {"LS_ORCH_SC_STRIDE", A::kReference, C::kPinDefault, nullptr, K::kIfSet, nullptr, "MoE batch width = bf16 grouped-GEMM shape class -> trajectory (TD-SERVE-SC-TRAJECTORY); also correctness-critical above single-shot capacity (orchestrator.py:1738-1745); recipe governs, env pin refused"},
        {"LS_ORCH_PREFILL_CHUNK", A::kReference, C::kPinDefault, nullptr, K::kIfSet, nullptr, "prefill chunk rows = same batch-width shape class as SC stride; recipe governs, env pin refused"},
        {"LS_ORCH_FORCE_SPLIT_ACT", A::kReference, C::kPinDefault, nullptr, K::kIfExact, "1", "=1 restores legacy split/ACT arm: static e%N placement + INV-LOADER-ACT-FPDRIFT ('OUTPUT-token-robust but NOT routing-bit-...', INVARIANTS.md:464) — not the reference stack"},
        {"LAYERSTORM_MOE_BIG_VRAM_CAP_MB", A::kReference, C::kPinDefault, nullptr, K::kIfSet, nullptr, "caps BIG-path VRAM -> superchunk capacity/stride clamps -> prefill batch-width shape class; recipe governs"},
        {"LS_MOE_BIG_FIT_HEADROOM_MB", A::kReference, C::kPinDefault, nullptr, K::kIfSet, nullptr, "overrides the chunk-fit working headroom (both device classes) -> realized chunk capacity = EP4 stride clamp -> prefill batch-width shape class (P-30 step 2); recipe (compute.moe_big_fit_headroom*_mb) governs, env is diagnostic"},
        {"LS_KVT_COHORT_ALWAYS", A::kReference, C::kPinDefault, nullptr, K::kIfTruthy, nullptr, "forces the cohort prefill arm; TD-KVT-COHORT-PREFILL-NUMERICS (sub-ulp geometry sensitivity) is OPEN — unproven either way, pinned to the default-auto safe side"},
        {"LS_KVT_COHORT_ROWWISE", A::kReference, C::kPinDefault, nullptr, K::kIfTruthy, nullptr, "forces the per-row arm of the same seam; same open numerics TD, pinned to default-auto"},
        {"LS_CPU_EXPERT", A::kRunToRun, C::kPinDefault, nullptr, K::kIfTruthy, nullptr, "routes expert FFN to the CPU kernel — different compute path (unmerged C-6 line) whose multithreaded reduction order is run-to-run UNPROVEN -> pinned off under BOTH flags (safe side)"},
        {"LS_CPU_EXPERT_FORCE", A::kRunToRun, C::kPinDefault, nullptr, K::kIfTruthy, nullptr, "force-arm of the CPU expert path — same run-to-run-unproven class, pinned under both flags"},
        {"LS_V4_ROW_PREFILL", A::kReference, C::kPinDefault, nullptr, K::kIfTruthy, nullptr, "bisect arm: 'numerics differ at bf16-lsb (split-KV partitioning + reduction shape)' (arch_deepseek_v4.cpp:1140-1141); default batched path is the golden-gated reference"},
        {"LS_V4_NSP", A::kReference, C::kPinDefault, nullptr, K::kIfSet, nullptr, "split-count knob on the V4 CSA decode — FP reassociation class (same family as LS_SNAPMLA_NSP); identity across values unproven -> pinned to default auto"},
        {"LS_CSA_DECODE_LEGACY", A::kReference, C::kPinDefault, nullptr, K::kIfExact, "1", "kernel-variant flip on the CSA decode (splitkv_csa.cu:1387); reduction-shape class, identity across variants unproven -> pinned to default"},
        {"LS_TQ_ROTATE_LEGACY", A::kReference, C::kPinDefault, nullptr, K::kIfExact, "1", "kernel-variant flip on TQ q-rotate (tq_q_rotate.cu:201); unproven either way -> pinned to default"},
        {"LS_TQ_ROTATE_TILED", A::kReference, C::kPinDefault, nullptr, K::kIfExact, "1", "kernel-variant flip on TQ v-rotate-back (tq_v_rotate_back.cu:77); unproven either way -> pinned to default"},
        {"LS_MHC_FORCE_LEGACY", A::kReference, C::kPinDefault, nullptr, K::kIfExact, "1", "kernel-variant flip on the mHC mixes kernel (mhc.cu:378); unproven either way -> pinned to default"},
        {"LS_KDA_DECODE_PIPELINE", A::kReference, C::kPinDefault, nullptr, K::kIfFalsy, nullptr, "kernel-variant flip on KDA decode (kda_decode.cu:192, default ON is the shipped/champion state; TD-KDA-DECODE-PIPELINE resolved with it ON); unproven-identity flip -> pinned to default"},
        {"LS_GGUF_E1_KSPLIT", A::kReference, C::kPinDefault, nullptr, K::kIfFalsy, nullptr, "E1 K-split GEMV (default ON, in the champion since the perf-saga merge; golden-gated) — flipping changes the reduction shape of routed GEMVs; pinned to default"},
        {"LS_GGUF_GROUPED_KSPLIT", A::kReference, C::kPinDefault, nullptr, K::kIfSet, nullptr, "grouped-INT kernel variant selector (gguf_grouped_int.cu:861, default kCompact measured-best, golden 12089) — variant flip = reduction-shape class; pinned to default"},
        {"LS_GGUF_GROUPED_KSPLIT_COMPACT", A::kReference, C::kPinDefault, nullptr, K::kIfSet, nullptr, "tri-state override of the kCompact choice (gguf_grouped_int.cu:930); pinned to default auto"},
        {"LS_GGUF_GROUPED_PIPE", A::kReference, C::kPinDefault, nullptr, K::kIfFalsy, nullptr, "grouped-INT pipeline variant (default ON, golden-gated); variant flip unproven-identity -> pinned to default"},
        {"LS_GGUF_GROUPED_CPASYNC", A::kReference, C::kPinDefault, nullptr, K::kIfFalsy, nullptr, "grouped-INT cp.async staging variant (default ON, golden-gated); pinned to default"},
        {"LS_GGUF_GROUPED_KSC_SMEM", A::kReference, C::kPinDefault, nullptr, K::kIfFalsy, nullptr, "grouped-INT smem variant (default ON, golden-gated); pinned to default"},
        {"LS_CPU_GEMV_SINGLE_ROW", A::kReference, C::kPinDefault, nullptr, K::kIfExact, "1", "CPU expert GEMV row-block vs single-row variant (nvfp4_cpu_kernel.cpp:852); reduction-shape class on the (pinned-off) CPU path; pinned to default"},

        // ── kKeepOn: PROVEN bit/byte-identical — deliberately untouched ─
        {"LS_KDA_PROJ_FUSED", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 15: six same-input KDA projections fused into one quantize + one multi-segment mmvq — per-CTA compute/reduction identical, only the blockIdx->(segment,channel) mapping changes (GgufMmvqMulti unit byte-compare + negative control; identical champion shas in vivo)"},
        {"LS_TQ_SPARSE_SLIM", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 6: bit-identical by construction and on-GPU proof (TqSparseDecodeSlim unit suite + flipped-KV-byte negative control + identical engine text shas); +26.9% @8k — exactly what deterministic mode must NOT give up"},
        {"LS_TQ_DECODE_GRAPH", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "same 4 kernels/args replayed from a per-layer graph; pointer/shape fingerprint forces re-capture on drift (tq_sm120_attention_device.cpp:832-855); champion shas identical"},
        {"LS_DECODE_CHAIN_GRAPH", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 7: byte-identical (all three sha16 groups identical across control, both ON boots and the P-29 step-6 champion; DecodeSpanGraphTest capture/replay memcmp)"},
        {"LS_MOE_FFN_GRAPH", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "INV-0.6(a): validated subgraph, bit-identical replay; default ON since 2026-07-20 and inside every golden since"},
        {"LS_FFN_GRAPH_KCOPY", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 2 (KCOPY): same bytes as memcpy/memset nodes, kernel-node emission only; unit tests vs references + negative controls; byte-identical arc across arms"},
        {"LS_MOE_META_CACHE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 3 (MPOKE): caches/binds IDENTICAL bytes behind fingerprints, self-healing on pointer change; BindCacheBitIdenticalAndSkipsReupload unit proof + identical shas"},
        {"LS_NCCL_FUSE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "bit-identical to two separate group calls — disjoint buffers, no shared reduction (moe_ranks.cpp:504-505)"},
        {"LS_NCCL_GRAPH", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "replays the SAME collectives in the SAME order on the same comm (construction argument; no measured identity proof cited in-tree — default OFF, champion boots leave it to the recipe)"},
        {"LS_MOE_NULL_SKIP_DECODE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "skip vs walking a zero-weight buffer: x+0 exact either way under the pre-zero/re-zero contract (INVARIANTS.md:387 clause 5 documents the contract violation that WOULD break it)"},
        {"LS_MOE_DECODE_WAVE_GATE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "pass-membership only: each permuted row computed by exactly one pass either way (x+0 exact => bit-identical tokens, moe_progressive.cpp:1185-1193)"},
        {"LS_MOE_WAVE_MASK", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "'kill-switch for A/B: bitwise-identical output' (dispatch_detail.h:106) — rows dropped from the GEMM, not summed differently"},
        {"LS_FAR_GATED_FINAL", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "compute identical to the host-poll path (same passes, same kernels, same order); only enqueue time changes; precondition failures fall back byte-identically (moe_progressive.cpp:561-577)"},
        {"LS_FAR_PROLOGUE_PREISSUE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 16: routing-independent MoE prologue (per-rank mHC collapse + ffn RMSNorm + EP-XTP broadcast) pre-issued at FAR attention-dispatch time — same kernels, same inputs, same per-stream order (attn_moe_event is recorded after gating+export, so every read still follows its producer); the finalize skips only the idempotent recompute; any failure falls back byte-identically (identical champion shas in vivo)"},
        {"LS_FAR_PROLOGUE_PRIME", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step-16 bisect sub-knob of LS_FAR_PROLOGUE_PREISSUE (prime half); bit-identical either way — the un-preissued half re-emits at its legacy site"},
        {"LS_FAR_PROLOGUE_BCAST", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step-16 bisect sub-knob of LS_FAR_PROLOGUE_PREISSUE (broadcast half) — since P-29 step 17 DEFAULT ON iff LS_MOE_XTP_BOUNCE is on (OQ-9 resolved: the P-29 step-16 ~6.5% loss was the driver-staged cross-device copies' deferred path when enqueued behind an unfired event; with the pinned-bounce the pre-issue wins +0.7% @8k); OFF under legacy staged copies; bit-identical either way"},
        {"LS_FAR_ISSUE_SLIM", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 17: fused single-pass graph b_ptr staging fill with live-index tracking — the staged BYTES equal the legacy three per-projection walks (same excluded value, same base+offset arithmetic, same TD-91d guard fallback); host-only bookkeeping, no stream content change (moe_driver.cpp fill_graph_b_ptrs_fused)"},
        {"LS_MOE_XTP_BOUNCE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 17: EP-XTP broadcast via explicit pinned-bounce (rank0 D2H -> pinned slot -> per-extra H2D) instead of driver-staged cross-device copies — identical bytes to identical extra-rank destinations; consumers ordered behind their own stream's H2Ds (equivalent fence); decode/small-M only, legacy path above num_tokens 8 (moe_ranks.cpp ep_xtp_broadcast)"},
        {"LS_SPEC_VERIFY_FETCH_HIDE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-32 stage 1: extends the three B=1 fetch-hiding fast paths (prologue pre-issue, resident-overlap, gated-final barrier) to spec_verify R<=8 MoE commands — same passes, same kernels, same order; only enqueue/detection time moves; consulted ONLY on spec_verify commands (speculation.method=mtp arms), so default boots never read it; =0 restores the phase-B exposed-H2D verify byte-identically"},
        {"LS_SPEC_VERIFY_BATCHED", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-32 stage 1: sparse-MLA spec_verify layers run their R rows as ONE batched dispatch — teacher-forced append-then-attend (all rows' K appended before scoring; every row's selection bounded by its own causal prefix, INV-DSA-BATCH discipline), device s_q=R arm executes the exact per-row kernel bodies (grid.y indexes row-local slices; locked by SnapMlaSparseVerify.* memcmp GPU tests) with a per-row batch-of-1 sub-dispatch fallback; consulted ONLY on spec_verify FAR commands; =0 restores the phase-B per-row command loop"},
        {"LS_FAR_GATE_DISPATCH", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 19: force-dispatches the layer's own STAGED demand copies past max_inflight_per_gpu inside the gated-final commit (the 8-slot window was measured 100% stale physically-complete copies) — same copies, same bytes, same priority-queue dispatch order as the flush_staged that would run ~34 us later; enqueue time only; failure falls back byte-identically to fb_not_dispatched (moe_progressive.cpp far_stream_gate_commit)"},
        {"LS_EVENT_POOL", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 19: DeviceBackend event free-list — recycles timing-disabled cudaEvents instead of create/destroy (~495 pairs/token); pooled handles query cudaSuccess (completed or fresh), so reuse is exactly equivalent to a fresh event for query/record/stream_wait; pending-record handles retire lazily and are never recycled early; no stream content change (cuda_sm120_device_backend.cu)"},
        {"LS_FAR_STREAM_GATE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "device-gating of the same kernels; off => no-op, on => enqueue-order/timing only (moe_progressive.cpp:821)"},
        {"LS_MOE_FOLD_VIA_HOST", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "identical fold bytes via pinned host hop instead of D2D; fully stream-ordered (moe_ranks.cpp:1024)"},
        {"LS_MOE_SKIP_MISS_PROBE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "removes a host-side miss-counter D2H + spin-sync; telemetry source changes, compute does not (moe_progressive.cpp:1756)"},
        {"LS_SNAPMLA_BUDGET_STAGING", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 10 stage (a): bit-identical, three independent proofs (unit memcmp incl. poison-holed staging + negative control; tp2 in-vivo shas exact; TF byte-identical on all 6,999 positions)"},
        {"LS_KV_EXPERT_REBALANCE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "44z: text byte-identical across all six boots and 66 runs incl. the capacity-different arm (glm53_flash.md:344); residency channel neutral under DET-EP-COMBINE"},
        {"LS_MTP_PROBE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "P-29 step 11: zero-perturbation proven — target records reproduce the archived TF baseline byte-exactly with the probe interleaved (0/98, then 0/5000 in P-29 step 13); NOTE: still do not arm on production boots (TD-MTP-PROBE-DEFERRED-CONSUMERS)"},
        {"LS_MTP_FORCE_ARM", A::kReference, C::kPinDefault, nullptr, K::kIfSet, nullptr, "P-29 step 13 bisect knob on the (default-OFF) MTP spec arm: forces alternate verify arms ('1row'/'d2_wrong'/'chain_run_d1'); spec serving was proven token-identical WITHOUT these arms forced — forced-arm identity unproven -> safe-side pin to unset"},
        {"LS_MTP_ROW0_PROBE", A::kReference, C::kPinDefault, nullptr, K::kIfSet, nullptr, "P-29 step 13 diagnostic on the MTP spec arm: runs an extra plain step at the named position via anchor+restore; designed non-perturbing but zero-perturbation NOT proven in serving -> safe-side pin to unset"},
        {"LS_SPAN_STATS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "P-29 step 13: span-graph engagement counter logging every 8192 runs (decode_span_graph.cpp) — print-only"},
        {"LS_MTP_ROUND_DEBUG", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "P-29 step 13: per-round debug prints on the MTP spec arm (orchestrator.py) — print-only"},
        {"LS_V4_META_CACHE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "V4 twin of LS_MOE_META_CACHE: caches identical meta bytes; kill switch restores the per-call device kernel (csa_hca_sm120_attention_device.cpp:436)"},
        {"LS_ORCH_NO_MIDEDGE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "=1 disables ALL mid-edge prefix reuse; with sub-grid forced OFF the remaining grid-clamped policy is token-identical either way (INV-PREFIX-CACHE-1) — identity-safe both ways"},
        {"LS_DSPARK_CTX_ROTATE", A::kReference, C::kKeepOn, nullptr, K::kNone, nullptr, "arm-choice only (speculative vs plain admission at long prompts); both arms commit target argmax at B=1 (INV-DSPARK-LOSSLESS) and the governor is forced off"},

        // ── kNeutral: audited — diagnostics / placement / transport ─────
        // Diagnostics & observability (read-only or perf-only):
        {"LS_EVENT_POOL_STATS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "P-29 step 19: event-pool hit/create/destroy counters printed at backend teardown — print-only"},
        {"LS_SPEC_DISPATCH_PROF", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "P-32 stage 1: host section timing (kv_meta/stage/execute) of batched spec_verify dispatches, printed every 512 — print-only"},
        {"LS_PERF_TRACE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_PERF_TRACE_OUT", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_PERF_TRACE_POLLTICKS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_DRIFT_DUMP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_SEAM_DUMP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_SEAM_DUMP_TQ", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_SEAM_DUMP_EXEC", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_SEAM_DUMP_MAXLAYER", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_SEAM_EXEC_FROM", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_SEAM_EXEC_TO", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_EPM_DUMP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_EP_DUP_DUMP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_ARENA_MAP_DUMP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_LOADER_SHADOW_DUMP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_LOADER_SHADOW_LOG", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_REEF_DECISION_DUMP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_REEF_RELOC_TRACE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_FORK_TRACE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_DEBUG_GATING", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_CUDA_TRIPWIRE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_KDA_XRAY", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "contains device_sync() (arch_glm5_next.cpp:273-317) — perf poison, output-neutral; keep OFF for measurements"},
        {"LS_MOE_BIG_XRAY", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_ATTN_CHUNK_PROF", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_DSPARK_PROF", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_MULTINUMA_PROFILE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_TP_FFN_PROBE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_TP_HIDDEN_PROBE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_SPEC_GOV_PROBE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "probe period of the round-shape governor — dead under forced LS_SPEC_GOVERNOR=0"},
        {"LS_SNAPMLA_NSP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "split count of the FP8 snapMLA decode kernel — dead under forced LS_SNAPMLA_FP8_DECODE=0"},
        {"LS_V4_STAGE_SYNC", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "serializes staging for observation (arch_deepseek_v4.cpp:603) — timing only"},
        {"LS_V4_STAGE_TAGS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_V4_STAGE_DUMP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_V4_KVT_STATS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_ORCH_ROUND_CSV", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_BRIDGE_ROUND_CSV", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_BRIDGE_NO_CYTHON", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "python-fallback bridge speaks the identical protocol; perf only"},
        {"LS_FEED_EXPERTSTATS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_DSPARK_AUX_SHIFT", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "diagnostic A/B probe on draft context keying; drafts are lossless-verified so target tokens unchanged (INVARIANTS.md:229: 'acceptance UNCHANGED... diagnostic only')"},
        {"LS_DSPARK_DRAFT_LOWPRI", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "draft stream priority — scheduling only; drafts lossless-verified"},
        {"LS_DSP4_SKIP_MOE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "test-only ablation (deliberately wrong output); not a serving knob — deterministic mode does not certify deliberately-broken configs"},
        {"LS_DSP4_SKIP_ATTN", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "test-only ablation — same class as LS_DSP4_SKIP_MOE"},
        // Placement / residency / eviction (trajectory-neutral given DET-EP-COMBINE):
        {"LS_LOADER_SHADOW", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_POLICY", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_ACT", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_ABLATE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_M2", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_MACH_PROF", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_LOADER_FORCE_IDENTITY", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "mechanism-isolation diagnostic (identity assignment) — placement channel; NOTE presence-only parse: =0 still ENABLES it (dispatch_loader.cpp:109)"},
        {"LS_LOADER_PIN_HITS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_REUSE_W", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_REUSE_TAU", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_FREQ_W", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_FREQ_DECAY", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_KEEPPRED_W", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_EVICT_WEIGHT", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_EVICT_UNIT_US", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_AFFINITY", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_SUM", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_W_RESID", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_W_NUMA", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_W_HOTNESS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_HOTNESS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_HOTNESS_W", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_HOTNESS_TAU", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_HOTNESS_FREQ", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_HOTNESS_FHOT", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_HOTNESS_CLAMP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_RARITY_THRESH", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_TARGET_RARE_ENCOURAGE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_RESID_TAX", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_OFFNODE_PROTECT", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_OFFLOAD_NODE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_HBM_OFFLOAD_BASE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_PLACE_DDR_OFFLOAD_PENALTY", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_CPU_SOLVER", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_CPU_SOLVER_NODE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_CPU_SKIP_B1", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_CPU_P", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_CPU_OVERLAP_TAU", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_CPU_OVERLAP_FLOOR", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_CPU_COST_MULT", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_CPU_A_US", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_CPU_B_US", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_CPU_A_PERTOK_US", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_LOADER_CPU_A_FIXED_US", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_EVICT_DECAY", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "eviction-score decay — residency channel; the historical decay-linked routing fork (DEBUG.md:96) was the placement->bf16-combine drift class killed by forced DET-EP-COMBINE"},
        {"LS_EVICT_LRU_FALLBACK", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_FAR_ENSURE_RESIDENTS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_REEF_ROUTE_CAP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_REEF_BANK_LIVE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_ARENA_PLACE_ONLINE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_ARENA_PLACE_FREQ", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_ARENA_MIGRATE_RATIO", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_ARENA_MIGRATE_MIN_TOTAL", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_ARENA_MIGRATE_MBPS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_ARENA_MIGRATE_MARGIN", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_ARENA_MIGRATE_HALF_LIFE_S", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_ARENA_ATTACH", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kTransportEv},
        {"LS_KVXP_HIGH_FRAC", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_KVXP_LOW_FRAC", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_KVXP_MAX_WASTE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_KVXP_MAX_SLOTS_PER_GRANT", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_KVXP_GRANT_COOLDOWN_TICKS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_KVXP_GRANT_COOLDOWN_MS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_KVXP_TICK_MS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_KVXP_LARGE_PREFILL_TOKENS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        {"LS_KVXP_LARGE_PREFILL_WAIT_MS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kPlacementEv},
        // Transport / memory domicile / IO (byte-preserving):
        {"LS_IPC_PIN", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kTransportEv},
        {"LS_KEEP_HOST_WEIGHTS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kTransportEv},
        {"LS_KDA_STATE_MAPPED", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kTransportEv},
        {"LS_KDA_HOLDER_SPILL", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kTransportEv},
        {"LS_KDA_PREFIX_CKPT", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "P-29 step 24 KDA prefix checkpoints — restore is a whole-slot fp32 byte round-trip and replay from a 64-aligned uniform frontier is bit-identical (INV-KDA-CARRY; golden CkptForkReplay + TF byte-compare gates); changes admission latency, never bytes"},
        {"LS_KVT_VERIFY", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "KV-tiering verification pass — read-only check; tier moves are byte-preserving (INV-KVT contracts)"},
        {"LS_KVT_SLAB_DEMOTE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kTransportEv},
        {"LS_KVT_HOT_WAIT", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kTransportEv},
        {"LS_SEQ_CKPT_PATH", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "checkpoint load path — input-class (changes WHAT is served, not how deterministically)"},
        {"LS_CPU_EXPERT_NODE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "CPU-expert family tuning — dead while LS_CPU_EXPERT is pinned off"},
        {"LS_CPU_EXPERT_SPREAD", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "CPU-expert family tuning — dead while LS_CPU_EXPERT is pinned off"},
        {"LS_CPU_EXPERT_OVERLAP", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "CPU-expert family tuning — dead while LS_CPU_EXPERT is pinned off"},
        {"LS_CPU_EXPERT_EARLY_KICK", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "CPU-expert family tuning — dead while LS_CPU_EXPERT is pinned off"},
        {"LS_CPU_EXPERT_LOSSLESS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "CPU-expert family tuning — dead while LS_CPU_EXPERT is pinned off"},
        {"LS_CPU_EXPERT_RESERVE_CORE", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "CPU-expert family tuning — dead while LS_CPU_EXPERT is pinned off"},
        // Calibration harness (offline; never on a serving boot):
        {"LS_CALIB_H2D_VS_DDR2HBM", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_CALIB_H2D_VS_CPUKERNEL", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_CALIB_DDR2HBM_THREADS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_CALIB_DDR2HBM_SRC", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_CALIB_DDR2HBM_SRC_MB", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_CALIB_DDR2HBM_DST", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_CALIB_DDR2HBM_DST_MB", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_CALIB_CPUKERNEL_WARM_MS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_CALIB_CPUKERNEL_PREPACK", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_CALIB_CPUKERNEL_BIN", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        {"LS_CALIB_CPUKERNEL_ARMS", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, kDiagEv},
        // The superflag itself (read by mode_requested, registered so the
        // tree-scan finds it accounted for):
        {"LS_DETERMINISTIC", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "flag 1 itself (run-to-run determinism)"},
        {"LS_REFERENCE_TRAJECTORY_IDENTITY", A::kReference, C::kNeutral, nullptr, K::kNone, nullptr, "flag 2 itself (reference-trajectory identity; implies flag 1)"},
    };
    return t;
}

bool env_conflicts(const Knob& k, const char* v) {
    if (v == nullptr) return false;
    switch (k.conflict) {
        case K::kNone:     return false;
        case K::kIfTruthy: return v[0] != '\0' && v[0] != '0';
        case K::kIfFalsy:  return v[0] == '0';
        case K::kIfSet:    return v[0] != '\0';
        case K::kIfExact:  return k.conflict_exact != nullptr &&
                                  std::string(v) == k.conflict_exact;
    }
    return false;
}

bool env_conforms_forced(const Knob& k, const char* v) {
    return v != nullptr && k.forced != nullptr && std::string(v) == k.forced;
}

// Env boolean with "set & first char != '0' -> on, '0...' -> off,
// unset/empty -> fall back to config" — the house override-either-way idiom.
bool env_bool_or(const char* env_name, bool config_value,
                 const char* (*get)(const char*)) {
    const char* e = get(env_name);
    if (e && *e) return e[0] != '0';
    return config_value;
}

const char* real_getenv(const char* n) { return std::getenv(n); }

}  // namespace

const std::vector<Knob>& registry() { return table(); }

Mode mode_requested(bool config_deterministic, bool config_reference_identity) {
    const bool ref = env_bool_or("LS_REFERENCE_TRAJECTORY_IDENTITY",
                                 config_reference_identity, &real_getenv);
    const bool det = env_bool_or("LS_DETERMINISTIC", config_deterministic,
                                 &real_getenv);
    if (ref) return Mode::kReferenceIdentity;  // strictly includes run-to-run
    if (det) return Mode::kRunToRun;
    return Mode::kOff;
}

Plan plan(Mode mode, const char* (*get)(const char*)) {
    Plan p;
    if (mode == Mode::kOff) return p;
    const bool reference = (mode == Mode::kReferenceIdentity);
    for (const Knob& k : table()) {
        // kRunToRun rows act under both modes (the hierarchy); kReference
        // rows act only under reference-identity mode.
        const bool row_active = (k.axis == Axis::kRunToRun) || reference;
        if (!row_active) continue;
        const char* v = get(k.env);
        if (env_conflicts(k, v)) {
            std::string line = std::string(k.env) + "=" + (v ? v : "") +
                " pinned by env conflicts with " +
                (k.axis == Axis::kRunToRun
                     ? "run-to-run determinism (LS_DETERMINISTIC)"
                     : "reference-trajectory identity "
                       "(LS_REFERENCE_TRAJECTORY_IDENTITY)") +
                " — " + k.evidence;
            p.conflicts.push_back(std::move(line));
            continue;  // refuse instead of also forcing a conflicting knob
        }
        switch (k.cls) {
            case C::kForceOn:
            case C::kForceOff: {
                Plan::Action a;
                a.env = k.env;
                a.value = k.forced ? k.forced : "";
                a.axis = k.axis;
                a.prior = v ? v : "";
                a.already_conforming = env_conforms_forced(k, v);
                p.forced.push_back(std::move(a));
                break;
            }
            case C::kKeepOn:
                p.kept.push_back(k.env);
                break;
            case C::kPinDefault:
            case C::kNeutral:
                break;
        }
    }
    p.notes.push_back(
        "DET-TOPK-TIES (INV-TOPK-TIE-DET) is a BUILD-time precondition "
        "(deps/LayerStoRmKernels >= 8228e71) — not runtime-verifiable; the gate "
        "suite (LightningTopkLongCtx.ContestedThresholdTiesDeterministicLowestIndex) "
        "is the verifier");
    p.notes.push_back(
        "cross-PLACEMENT bit-identity at depth on EP shapes is NOT guaranteed "
        "(TD-MOE-EP-XTP-PLACEMENT-DRIFT, open); same-config same-placement reruns are");
    p.notes.push_back(
        "sampled decoding (temperature > 0) is reproducible only per decode arm "
        "with a fixed seed; the reference discipline is greedy");
    if (reference)
        p.notes.push_back(
            "'reference' is RELATIVE to this configuration: the canonical "
            "numerics path for the chosen backend/tier/parallelism with no "
            "flag-level shortcut forking it — not a global golden sha");
    return p;
}

void apply_or_throw(bool config_deterministic, bool config_reference_identity) {
    const Mode mode =
        mode_requested(config_deterministic, config_reference_identity);
    if (mode == Mode::kOff) return;

    // The hierarchy is strict: reference identity IMPLIES run-to-run
    // determinism.  Requesting flag 2 while explicitly suppressing flag 1
    // is a contradiction, refused — never silently un-suppressed.
    const char* det_env = std::getenv("LS_DETERMINISTIC");
    if (mode == Mode::kReferenceIdentity && det_env && *det_env &&
        det_env[0] == '0') {
        throw std::runtime_error(
            "[determinism] REFUSING to boot: LS_REFERENCE_TRAJECTORY_IDENTITY "
            "is requested but LS_DETERMINISTIC=0 explicitly suppresses "
            "run-to-run determinism. Reference-trajectory identity IMPLIES "
            "run-to-run determinism (matching a deterministic reference "
            "exactly means being deterministic) — drop one of the two pins.");
    }

    const char* ref_env = std::getenv("LS_REFERENCE_TRAJECTORY_IDENTITY");
    const bool reference = (mode == Mode::kReferenceIdentity);
    spdlog::info(
        "[determinism] mode: {} (compute.deterministic={} LS_DETERMINISTIC={} "
        "| compute.reference_trajectory_identity={} "
        "LS_REFERENCE_TRAJECTORY_IDENTITY={}){}",
        reference ? "REFERENCE-TRAJECTORY IDENTITY (implies run-to-run "
                    "determinism)"
                  : "RUN-TO-RUN DETERMINISM",
        config_deterministic, det_env ? det_env : "<unset>",
        config_reference_identity, ref_env ? ref_env : "<unset>",
        reference ? "" : " — trajectory-forking-but-stable optimizations "
                         "(e.g. LS_TQ_SPLITKV) are permitted under this flag");

    Plan p = plan(mode, &real_getenv);

    if (!p.conflicts.empty()) {
        std::string msg =
            "[determinism] REFUSING to boot: the requested mode was pinned "
            "against by the environment. Unset the pin(s) or drop the flag. "
            "Conflicts:";
        for (const auto& c : p.conflicts) {
            spdlog::error("[determinism] CONFLICT: {}", c);
            msg += "\n  - " + c;
        }
        throw std::runtime_error(msg);
    }

    int forced_count = 0;
    for (const auto& a : p.forced) {
        const char* tag =
            a.axis == Axis::kRunToRun ? "run-to-run" : "reference";
        if (a.already_conforming) {
            spdlog::info("[determinism] {}={} already pinned (conforming, {})",
                         a.env, a.value, tag);
        } else {
            spdlog::info("[determinism] FORCED ({}) {}={} (was {})", tag,
                         a.env, a.value,
                         a.prior.empty() ? "<unset>" : a.prior.c_str());
            ++forced_count;
        }
        ::setenv(a.env.c_str(), a.value.c_str(), 1);
    }
    spdlog::info("[determinism] forced {} knob(s); kept {} proven-bit-identical "
                 "optimization(s) enabled (these are make-it-reproducible "
                 "switches, not make-it-slow switches)",
                 forced_count, p.kept.size());
    for (const auto& n : p.notes) spdlog::info("[determinism] note: {}", n);
}


}  // namespace determinism
}  // namespace core
}  // namespace layerstorm

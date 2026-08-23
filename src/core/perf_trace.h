#pragma once
// Deep fetch-path micro-timing trace (temporary, opt-in via LS_PERF_TRACE env).
//
// One record per stage transition: a lock-free atomic fetch_add into a
// pre-allocated array (no malloc on the hot path) + a 32-byte write. Records
// carry steady_clock ns + a stage tag + correlation IDs (expert key, gpu,
// cmd_seq, transfer token) so the full cross-thread lifecycle of each fetch
// can be reconstructed offline. Dump is deferred to engine shutdown (cold).
//
// Two-level gating:
//   * Compile-time: LAYERSTORM_PERF_TRACE (CMake option, default 1). When 0, the
//     entire facility is compiled out — record()/init()/dump_csv() become inline
//     no-ops with no symbols, no static state, and not even a branch. Set OFF for
//     production builds (`-DLAYERSTORM_PERF_TRACE=OFF`).
//   * Runtime (when compiled in): opt-in via the LS_PERF_TRACE env var. Zero cost
//     beyond a null check unless init()/init_from_env() actually allocated.
#include <cstdint>
#include <cstddef>

#ifndef LAYERSTORM_PERF_TRACE
#define LAYERSTORM_PERF_TRACE 1
#endif

namespace layerstorm::perf_trace {

enum Stage : uint16_t {
    kCmdIssued = 0,    // consumer: prefetch batch cmd written to ring
    kCmdDrained,       // daemon: prefetch batch cmd picked up
    kEnsureEnter,      // daemon: ensure_resident/try_start_h2d entered for an expert
    kResolveExit,      // daemon: resolve_host_source returned
    kEnqueueInline,    // daemon: dispatched inline (memcpy_async issued)
    kEnqueueStaged,    // daemon: staged (queued, not yet dispatched)
    kDispatchStaged,   // daemon: flush/replenish issued the staged memcpy
    kDetected,         // daemon: cudaEventQuery == ready
    kReadyPosted,      // daemon: EXPERT_READY completion written to ring
    kReadyReceived,    // consumer: EXPERT_READY read from ring
    kDmaGpu,           // daemon: pure GPU memcpy time (token = elapsed ns from
                       //         cudaEventElapsedTime(start,end)); not a timestamp
    kPollTick,         // daemon: one transfer-engine poll_completions() entry
    // ── MoE-command (E_CMD_FETCH_AND_RUN_MOE) phase markers; cmd_seq = MoE cmd ──
    kMoeEnter,         // daemon: handle_fetch_and_run_moe entered (command drained)
    kMoeFetchIssued,   // daemon: residency+decider+eviction done, all H2D issued
    kMoeFinalizeEnter, // daemon: quiescent → begin finalize (compute) pass
    kMoeFinalizeExit,  // daemon: finalize kernels enqueued (compute launched)
    kExpertArrived,    // daemon: a routed expert became resident (is_arrived);
                       //         gpu = target GPU, cmd_seq = MoE cmd, ns = arrival.
                       //         Per-GPU last-arrival → fetch wall + fastest/dead.
    kComputeGpu,       // daemon: pure GPU compute time of the finalize MoE chain
                       //         (token = elapsed ns from cudaEventElapsedTime over
                       //         the kExpertFfn stream); not a timestamp. gpu = rank.
    // ── §12h host-dispatch seam decomposition (dispatch_moe_all_ranks). Each
    // stage is recorded TWICE per occurrence: token=0 at segment entry, token=1
    // at exit; pair in order per (stage, gpu, key) offline. key = layer<<16. ──
    kMoeSegRankPre,    // one TP rank's Phase-1 (kPreAllreduce) internal dispatch
    kMoeSegExtra,      // one expert-only rank's internal dispatch (EP-XTP)
    kMoeSegFold,       // the Phase-1c fold loop (all extras → TP staging + add)
    kMoeSegNccl,       // Phase-2 collective enqueue (fused group / graph replay)
    kMoeSegRankPost,   // one TP rank's Phase-3 (kPostAllreduce) internal dispatch
    // ── Bridge-gap ledger markers (TD-BRIDGE-CPP-GAP, 2026-08-23). Generic
    // per-command daemon boundaries + the FAR handler's internal phases so a
    // paired C++-vs-bridge per-section ledger can be reconstructed offline
    // (analysis: scratchpad/gap_ledger.py). key = ipc cmd_type. ──
    kCmdDequeued,      // daemon: command drained from the cmd ring (any type)
    kCmpWritten,       // daemon: CMP_COMPUTE_DONE written to the cmp ring
    kFarAttnDispatched,// daemon FAR: dispatch_fused_attention returned
    kFarAttnReady,     // daemon FAR: attention stream event ready (spin done)
    kFarUnionDone,     // daemon FAR: routing-export dedup done
    kFarRouteDone,     // daemon FAR: reef_orch_route (solver) returned
    kFarSolveDone,     // daemon FAR: placement complete (reef apply / static)
    kNumStages
};

// ── Cold-path aggregation types (always defined; used by the x-ray report) ──
// One reduced phase: how many intervals were observed + their summed duration.
struct StageDelta {
    uint64_t count    = 0;
    double   total_ms = 0.0;
    double   mean_ms() const { return count ? total_ms / count : 0.0; }
};
// Daemon-side fine breakdown, reduced from the event buffer by pairing the
// per-MoE-command phase markers on `cmd_seq` and summing the kDmaGpu elapsed-ns
// tokens. `available` is false when the facility was compiled out or never
// enabled (the report then shows coarse rows only).
struct Aggregate {
    bool       available    = false;
    bool       overflow     = false;
    uint64_t   total_events = 0;
    StageDelta moe_fetch;    // kMoeEnter        → kMoeFetchIssued  (residency+decider+evict+H2D issue)
    StageDelta moe_wait;     // kMoeFetchIssued  → kMoeFinalizeEnter (quiescent: wait for H2D ready)
    StageDelta moe_compute;  // kMoeFinalizeEnter→ kMoeFinalizeExit  (finalize/compute launch)
    StageDelta moe_total;    // kMoeEnter        → kMoeFinalizeExit  (whole MoE command)
    StageDelta dma_gpu;      // Σ kDmaGpu token (pure GPU memcpy elapsed ns)
    // Fetch-wall decomposition (from kExpertArrived per cmd_seq). Per command:
    //   t_last  = max over GPUs of that GPU's LAST arrival; fastest = min over GPUs.
    //   fetch_wall    = t_last  − fetch_issued   (first byte→last byte; the fetch span)
    //   fetch_fastest = fastest − fetch_issued   (until the first GPU finished)
    //   fetch_dead    = t_last  − fastest        (straggler: faster GPU idle = imbalance)
    // fetch_fastest + fetch_dead == fetch_wall. Counted only for commands that fetched.
    StageDelta fetch_wall;
    StageDelta fetch_fastest;
    StageDelta fetch_dead;
    StageDelta compute_gpu;  // Σ kComputeGpu token (pure GPU compute elapsed ns,
                             // the finalize MoE GEMM chain; ≈ per-GPU compute wall)
};

#if LAYERSTORM_PERF_TRACE

// One-time allocation of `capacity` records. No-op if already initialized or
// capacity==0. Call once at engine init. Reads LS_PERF_TRACE if capacity==0.
void init(size_t capacity);
// Convenience: enable from the LS_PERF_TRACE env var (a record count, or "1"
// → default 64M). No-op if unset. kPollTick (per-poll spam) is recorded ONLY if
// LS_PERF_TRACE_POLLTICKS is set — off by default so the fetch/DMA/compute markers
// span the whole run instead of the ~2 tokens the poll spam would fill.
void init_from_env();

bool enabled() noexcept;

// Hot path: one atomic increment + one 32-byte store. `key` packs the expert
// as (layer_idx<<16)|expert_idx; `token` is the TransferToken (0 where N/A).
void record(uint16_t stage, uint16_t gpu, uint32_t cmd_seq,
            uint32_t key, uint64_t token) noexcept;

// Reduce the recorded buffer. Cold (call at shutdown / end of test). Cheap
// relative to a run (one hash-map pass over the events).
Aggregate aggregate();

// Deferred dump (cold): write all recorded events as CSV. Path from
// LS_PERF_TRACE_OUT env or the given default.
void dump_csv(const char* default_path);

#else  // LAYERSTORM_PERF_TRACE == 0 — compiled out: inline no-ops, fully elided.

inline void init(size_t /*capacity*/) {}
inline void init_from_env() {}
inline bool enabled() noexcept { return false; }
inline void record(uint16_t /*stage*/, uint16_t /*gpu*/, uint32_t /*cmd_seq*/,
                   uint32_t /*key*/, uint64_t /*token*/) noexcept {}
inline Aggregate aggregate() { return Aggregate{}; }  // available=false
inline void dump_csv(const char* /*default_path*/) {}

#endif  // LAYERSTORM_PERF_TRACE

}  // namespace layerstorm::perf_trace

#pragma once

// MultiNumaCpuExpertDevice — a CPU expert-FFN device that computes ONE expert's
// FFN across M NUMA nodes in parallel (tensor-parallel WITHIN the expert), to cut
// the 1-token-1-expert latency toward 1/M, while also supporting the grouped
// (multi-token / multi-expert) GEMM variants (C-6).
//
// Motivation: the single-node NumaCpuExpertDevice does 1×1×1 in ~2.0 ms hot on one
// node's 14 physical cores; it is decode(vpermps)-compute-bound, far above the
// memory floor, so the other idle NUMA nodes (Xeon Max 9480 SNC4: 4 nodes × 14
// physical cores) are wasted. This device spreads each expert's three projections
// across M nodes by an OUTPUT-ROW (N) partition so the per-node FLOPs (and decode
// work) drop ~1/M.
//
// Distribution (OUTPUT-N partition, contiguous packed row-slabs — clean to stage):
//   gate GEMM [N=I=2048, K=H=7168]  : node j computes rows [j_slice of I]
//   up   GEMM [N=I=2048, K=H=7168]  : node j computes rows [j_slice of I]
//   SwiGLU(d=I)                     : node j on its own I-slice -> swig[j_slice]
//   ALL-GATHER swig across nodes (full [M_tokens, I] needed by down)
//   down GEMM [N=H=7168, K=I=2048]  : node j computes output rows [j_slice of H]
//   CONCAT per-node down output row-slices -> moe_output (disjoint rows, no reduce)
// Slice boundaries are rounded to multiples of 128 ROWS so the Sm1xx scale-atom
// (128 rows × 4 groups) stays in-bounds when a node de-interleaves its slice
// (INV-CPU-EXP-3). A node computes ONLY from its LOCAL arena-scratch slab — the
// per-expert weight slices are staged on-demand into node_scratch (see PART 1 of
// the arena work: PinnedExpertArena::node_scratch).
//
// Threading: M-1 PERSISTENT cross-node driver threads (spawned once, parked on a
// generation handshake — NOT per call; per-call std::thread create/join across
// NUMA nodes costs more than the compute it saves at ms granularity), plus the
// calling daemon thread as driver 0. Each driver is pinned to its node and drives
// that node's NumaThreadPool over its row-slice. The cross-node join is a
// spin-then-yield wait on a remaining-count (same discipline as ik_barrier), and
// each node's own pool uses its own ik_barrier intra-node. A residency map skips
// re-staging a slice already resident for the same (proj, expert, slice).
//
// This is a CUDA-free / GPU-SDK-free TU exactly like NumaCpuExpertDevice
// (INV-GPU-1, INV-BH-2, layerstorm_no_cuda_check). The `void* stream` arg on every
// ExpertDevice method is IGNORED (synchronous). fp8_grouped_gemm THROWS (routed
// experts are nvfp4-sm1xx).

#include "core/expert_device.h"
#include "compute/cpu/numa_cpu_expert_device.h"  // CpuExpertModelDims

#include <memory>
#include <vector>

namespace layerstorm::memory {
class NumaManager;
class PinnedExpertArena;
}  // namespace layerstorm::memory

namespace layerstorm::compute {

/// Dependencies for a multi-NUMA CPU expert device. All non-owning; the engine
/// owns the NumaManager / PinnedExpertArena and outlives the device.
struct MultiNumaCpuExpertDeps {
    /// The SET of NUMA nodes to spread each expert across (M = nodes.size()).
    /// Order is significant: nodes[0] hosts driver-thread 0 (the caller) and the
    /// assembled result; the output-N partition assigns nodes[j] the j-th slice.
    std::vector<int>           nodes;
    memory::NumaManager*       numa  = nullptr;  ///< node-local alloc / topology
    /// Host arena providing the per-node front-of-slab scratch the device stages
    /// weight slices into (PinnedExpertArena::node_scratch). Optional: when null,
    /// the device device_alloc's a private node-local staging buffer per node.
    memory::PinnedExpertArena* arena = nullptr;
    CpuExpertModelDims         dims{};
    int max_threads_per_node = 0;  ///< <=0 => all physical cores on each node
};

/// Factory: create a MultiNumaCpuExpertDevice for `gpu` (a GpuType::cpu GpuRef;
/// `gpu.id` carries the PRIMARY NUMA node — deps.nodes[0]). The device spreads
/// each expert's projections across `deps.nodes` and assembles the result on the
/// primary node. Throws std::invalid_argument if deps.nodes is empty.
std::unique_ptr<ExpertDevice> make_multi_numa_cpu_expert_device(
    config::GpuRef gpu, MultiNumaCpuExpertDeps deps);

// ── Benchmark instrumentation (stage-vs-compute attribution) ─────────────────
// Process-wide accumulators populated by MultiNumaCpuExpertDevice ONLY when the
// env var LS_MULTINUMA_PROFILE is set (zero overhead otherwise). Summed across all
// node-threads since the last reset: nanoseconds in cross-node weight STAGING vs
// node COMPUTE, and bytes actually staged (re-stages only — residency hits cost 0).
// Used by MultiNumaCpuExpert.ScalingBenchmark to report stage_us separately from
// compute_us per the C-6 deliverable. NOT a production API.
void     multinuma_profile_reset();
uint64_t multinuma_profile_stage_ns();
uint64_t multinuma_profile_compute_ns();
uint64_t multinuma_profile_staged_bytes();

}  // namespace layerstorm::compute

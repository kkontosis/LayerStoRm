#pragma once

// NumaCpuExpertDevice: a CPU expert-FFN device for ONE NUMA node (C-6).
//
// Concrete ExpertDevice (and ONLY ExpertDevice — it does NOT subclass
// DeviceBackend and has nothing to do with attention). It runs the routed
// expert FFN entirely on host cores pinned to one NUMA node, reading its weights
// from the host-resident PinnedExpertArena slabs mbind'd to that node, and
// produces moe_output in host RAM.
//
// This is the realization of the `Ddr5CpuExpertDevice` / `Hbm2CpuExpertDevice`
// placeholders noted in core/expert_device.h:16-17.
//
// The header is CUDA-free and CPU-SDK-free, exactly like null_expert_device.h:
// the whole .cpp compiles with NO GPU SDK and is covered by
// layerstorm_no_cuda_check (INV-GPU-1, INV-BH-2). The `void* stream` argument on
// every ExpertDevice method is IGNORED (synchronous CPU compute), like
// NullExpertDevice.
//
// FFN chain (per nvfp4 routed expert, dims = DeepSeek V3.2):
//   moe_permute -> gate GEMM (N=2048,K=7168) -> up GEMM (N=2048,K=7168)
//     -> SwiGLU (d=moe_intermediate=2048) -> down GEMM (N=7168,K=2048)
//     -> moe_unpermute
// hidden=7168, moe_intermediate=2048, topk=8, n_routed_experts=256.
//
// Threading: fork-join per FFN call via NumaThreadPool (one parallel region),
// threads pinned to this node's physical cores, ik-barrier between
// gate/up -> SwiGLU -> down stages, GEMM work partitioned by output rows (N).
// Thread 0 == the calling daemon thread, which holds the result.
//
// DEFERRED SEAM (the only stub in the shipped device): the produced moe_output
// lives in HOST RAM on thread 0. Crossing it to the TP GPUs (H2D + fold into
// dispatch_moe.cpp allreduce_hidden / dcp_communicator.cpp) is a SEPARATE
// SESSION. See the seam comment in numa_cpu_expert_device.cpp, the
// spec/SPEC_UPDATES.md note, and the spec/TECH_DEBT.md entry.

#include "core/expert_device.h"

#include <memory>

namespace layerstorm::memory {
class NumaManager;
class PinnedExpertArena;
}  // namespace layerstorm::memory

namespace layerstorm::compute {

/// Model dimensions the CPU expert FFN needs (DeepSeek V3.2 routed expert).
/// Passed at construction so the device never reaches into the global Config.
struct CpuExpertModelDims {
    int hidden_dim = 7168;        ///< H
    int moe_intermediate = 2048;  ///< I (per-expert FFN width; NOT dense 18432)
    int topk = 8;                 ///< experts per token
    int n_routed_experts = 256;   ///< total routed experts per layer
};

/// Dependencies for a NUMA CPU expert device. All non-owning; the engine owns
/// the NumaManager and PinnedExpertArena and outlives the device.
struct NumaCpuExpertDeps {
    memory::NumaManager*       numa  = nullptr;  ///< node-local alloc / bind
    // Reserved for the deferred GPU-handoff dispatch wiring (resolving per-expert
    // arena slot pointers): stored but not yet consumed by the current device
    // (which receives weights via scattered B_ptrs). Do not remove the include.
    memory::PinnedExpertArena* arena = nullptr;  ///< host weight slabs (warm tier)
    CpuExpertModelDims         dims{};
    int max_threads = 0;          ///< <=0 => all physical cores on the node
    /// Reserve this many of the node's LEADING physical cores for the daemon /
    /// orchestrator (C-6 QA): the FFN pool skips them (workers start from the
    /// node's SECOND core onward when =1). Default 0 => legacy layout (all cores).
    int reserve_leading_cores = 0;
};

/// Factory: create a NumaCpuExpertDevice for `gpu` (a GpuType::cpu GpuRef whose
/// `id` carries the NUMA node index — see gpu_ref.h). The device pins its work
/// to `gpu.id`'s NUMA node and resolves weights from that node's arena slabs.
std::unique_ptr<ExpertDevice> make_numa_cpu_expert_device(
    config::GpuRef gpu, NumaCpuExpertDeps deps);

}  // namespace layerstorm::compute

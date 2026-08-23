#pragma once

//==============================================================================
// NCCL Group Graph Runner (INV-NCCL-GRAPH)
//
// Captures ONE fixed group of NCCL allreduce ops (all ranks, issued exactly
// like the eager grouped path) into per-rank CUDA graphs, replayed once per
// layer. Removes the eager NCCL enqueue host cost + launch stagger for
// per-layer collectives whose buffers/counts are FIXED across layers and
// decode steps (e.g. the o_proj TP reduce on hidden_out_, the fused MoE
// combine on shared_expert_output + moe_output).
//
// Capture protocol (single orchestrator thread, ThreadLocal capture mode —
// same as DcpAllreduceGraphRunner): begin capture on EVERY rank's stream,
// issue the grouped collectives exactly as the eager path does
// (ncclGroupStart → per-rank ncclAllReduce → ncclGroupEnd), end capture on
// every stream → one graph per rank, instantiated per device. Nothing
// executes during capture, so the caller must REPLAY immediately after a
// successful init to actually perform that step's reduction.
//
// Replay contract: all ranks' graphs must be replayed together, in rank
// order, on the same per-rank streams — exactly like the eager grouped
// launch. Mixing captured replays and eager collectives on the same comms is
// supported by NCCL (graph mixing support).
//
// Failure = fail-open: init() returns false (never throws for capture
// errors), the caller keeps the eager path.
//
// INV-0.6 (refined): bounded fixed-shape subgraph, no transfers.
// Thread safety: NOT thread-safe (INV-3.4.2).
//==============================================================================

#include <cstddef>
#include <vector>

namespace layerstorm::compute {

class NcclGroupGraphRunner {
public:
    struct Op {
        void*  buffer = nullptr;  // in-place allreduce (send == recv)
        size_t count = 0;         // element count
        bool   fp32 = false;      // false = bf16 payload
    };

    NcclGroupGraphRunner() = default;
    ~NcclGroupGraphRunner() { destroy(); }

    NcclGroupGraphRunner(const NcclGroupGraphRunner&) = delete;
    NcclGroupGraphRunner& operator=(const NcclGroupGraphRunner&) = delete;
    NcclGroupGraphRunner(NcclGroupGraphRunner&&) = delete;
    NcclGroupGraphRunner& operator=(NcclGroupGraphRunner&&) = delete;

    /// Capture the grouped collectives into per-rank graphs.
    /// device_ids[r] = CUDA device ordinal (GpuRef.id) of rank r;
    /// comms[r] = ncclComm_t; streams[r] = cudaStream_t (the rank's live
    /// stream — prior pending work is unaffected by capture);
    /// ops[r] = rank r's op list (must be structurally identical across
    /// ranks: same op count/order/dtypes, per NCCL collective matching).
    /// Returns true iff every rank captured + instantiated. On failure all
    /// partial state is torn down and is_captured() stays false.
    bool init(const std::vector<int>& device_ids,
              const std::vector<void*>& comms,
              const std::vector<void*>& streams,
              const std::vector<std::vector<Op>>& ops);

    /// Launch every rank's graph on its stream (rank order, like the eager
    /// grouped launch). Precondition: is_captured().
    void replay(const std::vector<void*>& streams);

    /// Free all CUDA graph resources. Idempotent.
    void destroy();

    bool is_captured() const { return captured_; }

private:
    bool captured_ = false;
    std::vector<int> device_ids_;
    std::vector<void*> execs_;  // cudaGraphExec_t per rank
};

}  // namespace layerstorm::compute

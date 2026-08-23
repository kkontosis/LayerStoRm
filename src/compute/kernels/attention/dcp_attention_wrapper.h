#pragma once

//==============================================================================
// DCP Attention Wrapper
//
// Encapsulates the per-layer DCP correction flow (DCP_GUIDE §5 steps 10-14).
// Called by DcpExecutor (#40) between the attention kernel output and the
// o_proj GEMM result.  Does NOT call the attention backend or o_proj itself.
//
// Two methods:
//   correct_output() — steps 10-12: allgather LSE → correction kernel →
//                      allreduce output
//   reduce_hidden()  — step 14: allreduce o_proj hidden (TP reduction)
//
// When dcp_size == 1: all methods are no-ops.
//
// INV-DCP-1:  backend-agnostic (no KV cache / quant / sparsity references)
// INV-DCP-8:  correct_output and reduce_hidden are separate (HOP-B pipelining)
// INV-DCP-11: global_lse buffers pre-allocated at construction
// INV-3.4.1:  no blocking calls — all work enqueued on streams
// INV-3.4.2:  NOT thread-safe (single-threaded orchestrator)
//==============================================================================

#include <cstddef>
#include <functional>
#include <vector>

#include "core/gpu_ref.h"

namespace layerstorm::compute { class AttentionDevice; }
namespace layerstorm::parallelism { class DcpCommunicator; }

namespace layerstorm::compute {

// -- DCP correction kernel callback ------------------------------------------

/// Type-erased callback for the DCP LSE correction kernel.
/// Concrete implementations construct kernel params and launch.
using LaunchCorrectionFn = std::function<void(
    void* output, const float* lses, float* global_lse,
    int B, int H, int D, int N, int rank, void* stream)>;

/// Real CUDA correction kernel (launch_dcp_lse_correct).
LaunchCorrectionFn cuda_launch_correction();

/// No-op correction (for unit tests without CUDA).
inline LaunchCorrectionFn null_launch_correction() {
    return [](void*, const float*, float*, int, int, int, int, int, void*) {};
}

// -- Configuration -----------------------------------------------------------

struct DcpAttentionWrapperConfig {
    int num_heads_local;          // H_local per GPU (e.g. 64 for TP=2 V3.2)
    int head_dim;                 // D, kv_lora_rank for absorbed MLA (e.g. 512)
    int hidden_size;              // model.hidden_size (e.g. 7168)
    int max_batch_size;           // for global_lse buffer pre-allocation
    // INV-KVS-QAG: head count of the LSE combine (correct_output). 0 (default)
    // = num_heads_local (legacy per-rank combine). Under SHARDED KV the engine
    // sets dcp*num_heads_local == model num_attention_heads: after the Q-head
    // allgather every rank's attention partial covers ALL heads over its local
    // token shard, and the combine merges same-head partials across ranks.
    int combine_num_heads = 0;
    std::vector<config::GpuRef> gpus;  // TP GPU refs, indexed by rank (INV-4.18)
};

// -- DcpAttentionWrapper -----------------------------------------------------

class DcpAttentionWrapper {
public:
    /// @param comm             Non-owning pointer to DcpCommunicator.
    ///                         When nullptr or comm->dcp_size()==1, all methods are no-ops.
    /// @param cfg              Configuration (dimensions, device IDs).
    /// @param attention_devices Per-rank AttentionDevice* for set_device/alloc/free (INV-BH-1).
    /// @param correction_fn    DCP correction kernel launcher.
    DcpAttentionWrapper(
        layerstorm::parallelism::DcpCommunicator* comm,
        DcpAttentionWrapperConfig cfg,
        std::vector<AttentionDevice*> attention_devices,
        LaunchCorrectionFn correction_fn = cuda_launch_correction());

    ~DcpAttentionWrapper();

    // Non-copyable, non-movable (owns device buffers)
    DcpAttentionWrapper(const DcpAttentionWrapper&) = delete;
    DcpAttentionWrapper& operator=(const DcpAttentionWrapper&) = delete;
    DcpAttentionWrapper(DcpAttentionWrapper&&) = delete;
    DcpAttentionWrapper& operator=(DcpAttentionWrapper&&) = delete;

    /// Steps 10-12: allgather LSE + DCP correction kernel + allreduce output.
    /// partial_outputs [dcp_size] corrected IN-PLACE ([B, H_c, D] BF16 with
    /// H_c = combine_num_heads > 0 ? combine_num_heads : num_heads_local;
    /// partial_lses are [B, H_c] FP32). No-op when dcp_size == 1.
    void correct_output(void* const* partial_outputs,
                         const float* const* partial_lses,
                         int batch_size,
                         void* const* streams);

    /// Step 14: allreduce o_proj hidden states (TP reduction).
    /// hiddens [dcp_size] reduced IN-PLACE ([B, hidden_size] BF16).
    /// No-op when dcp_size == 1.
    void reduce_hidden(void* const* hiddens,
                        int batch_size,
                        void* const* streams);

    bool is_active() const;
    int dcp_size() const;

private:
    layerstorm::parallelism::DcpCommunicator* comm_;
    DcpAttentionWrapperConfig cfg_;
    std::vector<AttentionDevice*> attention_devices_;
    LaunchCorrectionFn correction_fn_;
    int dcp_size_ = 1;
    int combine_heads_ = 0;   // resolved combine head count (INV-KVS-QAG)

    // Pre-allocated per-rank: [max_batch_size, combine_heads_] FP32
    std::vector<float*> global_lse_buffers_;
};

}  // namespace layerstorm::compute

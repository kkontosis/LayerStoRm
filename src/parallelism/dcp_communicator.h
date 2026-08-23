#pragma once

//==============================================================================
// DCP Communicator
//
// Owns NCCL communicator handles for the DCP/TP GPU pair and provides typed
// collective operations (allgather, allreduce) for the DCP correction flow.
//
// Lifecycle: created at engine init from parallelism.tensor_parallelism config,
// destroyed at shutdown.  DcpAllreduceGraphRunner (#39) receives ncclComm_t
// via the comm(rank) accessor for CUDA graph capture.
//
// INV-DCP-10: sole owner of NCCL comm handles.
// INV-DCP-11: pre-allocates all recv buffers at init (no hot-path allocation).
// INV-3.4.2:  NOT thread-safe (single-threaded orchestrator).
//
// Communication per layer (DCP_GUIDE §5):
//   Step 7b (sharded KV only, INV-KVS-QAG): allgather Q in the HEAD dim
//            [B, H_local, d_q] BF16 -> rank-major [N, B, H_local, d_q]
//   Step 10: allgather LSE        [B, H] FP32 -> [N, B, H] FP32
//   Step 12: allreduce output     [B, H, D] BF16 SUM
//   Step 14: allreduce hidden     [B, hidden_size] BF16 SUM (o_proj TP)
// H for steps 10/12 is per-call: H_local under the legacy per-rank combine,
// dcp*H_local (= num_heads) under the sharded-KV all-head combine.
//
// Grouped API: all collective methods handle group_begin/end and
// set_device internally via CollectiveBackend + DeviceBackend.
//==============================================================================

#include "core/gpu_ref.h"
#include "parallelism/collective_backend.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace layerstorm::compute { class DeviceBackend; }

namespace layerstorm::parallelism {

// -- DcpCommunicator ---------------------------------------------------------

class DcpCommunicator {
public:
    struct Options {
        int dcp_size = 1;                  // parallelism.tensor_parallelism
        std::vector<compute::DeviceBackend*> device_backends; // one per rank
        int max_batch_size = 64;           // max decode batch
        int num_heads = 128;               // model.num_attention_heads
        int attn_output_dim = 512;         // kv_lora_rank for absorbed MLA
        int hidden_size = 7168;            // model.hidden_size
        CollectiveBackend* collective = nullptr; // nullptr OK for dcp_size==1
    };

    /// @throws std::invalid_argument if dcp_size < 1 or
    ///         device_backends.size() != dcp_size (when dcp_size >= 2).
    explicit DcpCommunicator(Options opts);
    ~DcpCommunicator();

    // Non-copyable, non-movable (owns NCCL comms and device buffers)
    DcpCommunicator(const DcpCommunicator&) = delete;
    DcpCommunicator& operator=(const DcpCommunicator&) = delete;
    DcpCommunicator(DcpCommunicator&&) = delete;
    DcpCommunicator& operator=(DcpCommunicator&&) = delete;

    // -- Comm access (for DcpAllreduceGraphRunner graph capture) -------------

    /// Returns the raw ncclComm_t for a given rank.
    /// @throws std::out_of_range if rank >= dcp_size.
    /// Returns nullptr when dcp_size == 1.
    void* comm(int rank) const;

    // -- Grouped collectives -------------------------------------------------
    // All methods handle ncclGroupStart/End and cudaSetDevice internally.
    // Array parameters are indexed by rank [0, dcp_size).

    /// Allgather LSE: per-rank [B, H] FP32 -> per-rank [N, B, H].
    /// Writes into pre-allocated gathered_lse_buffer(rank).
    /// num_heads: per-head-count of the LSE rows. -1 (default) = the legacy
    /// per-rank H_local; the sharded-KV all-head combine (INV-KVS-QAG) passes
    /// dcp*H_local == num_heads. Must be <= Options::num_heads (buffer bound).
    /// When dcp_size==1: identity copy from local_lses[0] to buffer.
    void allgather_lse(const float* const* local_lses,
                        int batch_size,
                        void* const* streams,
                        int num_heads = -1);

    /// Allreduce corrected output: [B, H, D] BF16 SUM (in-place).
    /// num_heads: -1 (default) = H_local; the sharded-KV all-head combine
    /// passes dcp*H_local == num_heads (see allgather_lse).
    /// When dcp_size==1: no-op.
    void allreduce_output(void* const* outputs,
                           int batch_size,
                           void* const* streams,
                           int num_heads = -1);

    /// INV-KVS-QAG (sharded KV, DCP_GUIDE §5 step 7b): allgather the absorbed
    /// query across DCP ranks in the HEAD dimension. send_bufs[r] is rank r's
    /// [batch_size, H_local, d_q] BF16 query (per_rank_elems = batch_size *
    /// H_local * d_q BF16 elements); recv_bufs[r] receives the rank-major
    /// NCCL concatenation [dcp, batch_size, H_local, d_q] on rank r's GPU.
    /// Because rank s holds global heads [s*H_local, (s+1)*H_local), the
    /// gathered rank-major order IS the global head order; at batch_size==1
    /// the recv buffer is directly [1, dcp*H_local, d_q] (head-major). For
    /// batch_size>1 the caller must rearrange token-major (DcpExecutor does a
    /// per-source-rank strided copy). When dcp_size==1: no-op.
    void allgather_q(const void* const* send_bufs,
                     void* const* recv_bufs,
                     int batch_size,
                     size_t per_rank_elems,
                     void* const* streams);

    /// Allreduce o_proj hidden: [B, hidden_size] BF16 SUM (in-place).
    /// Separate from allreduce_output per INV-DCP-8 (HOP-B pipelining).
    /// When dcp_size==1: no-op.
    /// fp32 (DET-REDUCE Phase 1b): when true the buffers are FP32 and the reduce
    /// runs in fp32 (placement-invariant EP combine for the routed MoE). Default
    /// false = legacy bf16 in-place sum.
    /// rows_per_token (DET-REDUCE Phase 1b canonical): >1 for the per-slot combine
    /// — the buffers are [batch_size, rows_per_token, hidden_size] and the
    /// collective element count is batch_size*rows_per_token*hidden_size, while
    /// the max_batch_size guard still applies to batch_size (the token count).
    void allreduce_hidden(void* const* hiddens,
                           int batch_size,
                           void* const* streams,
                           bool fp32 = false,
                           int rows_per_token = 1);

    /// INV-NCCL-FUSE: fused MoE-combine allreduce — the per-layer shared-expert
    /// partial-sum reduce ([B, hidden] BF16, buffers_a) and the routed
    /// EP-combine reduce (buffers_b: mode-0 [B, hidden] BF16, or the canonical
    /// per-slot payload with b_fp32 / b_rows_per_token as in allreduce_hidden)
    /// issued inside ONE ncclGroup per rank, so NCCL aggregates both ops into
    /// a single device launch per rank instead of two back-to-back grid-1
    /// latency-bound kernels. BIT-IDENTICAL to two separate allreduce_hidden
    /// calls (same per-element rank sums; aggregation changes launch count
    /// only). When dcp_size==1: no-op.
    void allreduce_hidden_fused(void* const* buffers_a,
                                void* const* buffers_b,
                                int batch_size,
                                void* const* streams,
                                bool b_fp32 = false,
                                int b_rows_per_token = 1);

    /// KD-4g: Allgather partial logits for TP-sharded output head.
    /// Per-rank [batch_size, local_vocab_size] FP32 → per-rank [batch_size, vocab_size] FP32.
    /// send_bufs[r] and recv_bufs[r] are device pointers on rank r's GPU.
    /// When dcp_size==1: no-op.
    /// NOTE: For batch_size > 1, gathered layout is rank-major (not token-major).
    /// Correct only when batch_size == 1 (decode). TODO: transpose for prefill.
    void allgather_logits(const float* const* send_bufs,
                          float* const* recv_bufs,
                          int batch_size, int local_vocab_size,
                          void* const* streams);

    /// TD-GLM-INDEXER-LOCAL-MERGE (dcp_indexer_mode=local): allgather the
    /// per-rank shard-local top-k candidate lists before the cross-rank
    /// merge. send_bufs[r] is rank r's packed candidate buffer of
    /// `count_words` 32-bit words ([B*topk int32 local indices]
    /// [B*topk f32 scores]); recv_bufs[r] receives the rank-major NCCL
    /// concatenation [dcp, count_words] on rank r's GPU. The payload is
    /// bit-transparent (gathered as 32-bit words, no arithmetic). Tiny:
    /// 2*B*topk words per rank (32 KB at B=1, topk=2048). When dcp_size==1:
    /// no-op.
    void allgather_indexer_candidates(const void* const* send_bufs,
                                      void* const* recv_bufs,
                                      size_t count_words,
                                      void* const* streams);

    // -- Pre-allocated buffer access -----------------------------------------

    /// Returns pointer to gathered LSE buffer on the given rank's device.
    /// Layout: [dcp_size, max_batch_size, H] FP32 where H is the per-call
    /// num_heads of the preceding allgather_lse. Sized for the worst case
    /// H == Options::num_heads (the sharded-KV all-head combine).
    /// @throws std::out_of_range if rank >= dcp_size.
    /// Returns nullptr when dcp_size == 1.
    float* gathered_lse_buffer(int rank) const;

    // -- Queries -------------------------------------------------------------

    int dcp_size() const { return opts_.dcp_size; }
    int num_heads_local() const { return num_heads_local_; }
    bool is_active() const { return opts_.dcp_size >= 2; }
    const std::vector<config::GpuRef>& gpus() const { return gpus_; }

private:
    Options opts_;
    int num_heads_local_ = 0;
    std::vector<config::GpuRef> gpus_;    // cached from device_backends

    std::vector<void*>  comms_;           // [dcp_size] ncclComm_t handles
    std::vector<float*> lse_buffers_;     // [dcp_size] gathered LSE buffers

    void validate_rank(int rank) const;
};

}  // namespace layerstorm::parallelism

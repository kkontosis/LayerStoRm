#pragma once

//==============================================================================
// DCP Allreduce CUDA Graph Runner
//
// Captures the per-layer DCP correction sequence into a replayable CUDA graph:
//   1. NCCL allgather LSE       -- [B,H] FP32 per rank -> [N,B,H] FP32
//   2. DCP LSE correction kernel -- reweight output by exp(local_lse - global_lse)
//   3. NCCL allreduce output     -- [B,H,D] BF16 SUM across DCP ranks
//
// Usage:
//   DcpAllreduceGraphRunner runner;
//   runner.init(cfg, stream);   // allocate intermediates, capture graph
//   // per decode step (data in DecodeGraphRunner's fixed buffers changes):
//   runner.replay(stream);
//   runner.destroy();
//
// Design: no update() needed -- reads from DecodeGraphRunner's fixed-address
// output buffers (partial_output, partial_lse) whose addresses are stable
// across replays.  Data changes each step, but addresses don't.
//
// INV-DCP-7:  requires dcp_size >= 2.
// INV-DCP-8:  captures only steps 10-12 (DCP correction).  o_proj TP
//             allreduce (step 14) is separate for future HOP-B pipelining.
// INV-DCP-9:  partial_output and partial_lse must be stable fixed-address
//             buffers across replays.
// INV-0.6 (refined): bounded fixed-shape subgraphs only -- never a unified
//            whole-layer graph, never transfers. The decode routed-expert FFN
//            is now a permitted subgraph (flag-gated); see spec/INVARIANTS.md.
//
// Thread safety: NOT thread-safe (INV-3.4.2 -- single-threaded orchestrator).
//==============================================================================

namespace layerstorm::compute {

class DeviceBackend;  // #86b: forward decl for alloc/free abstraction

// -- Configuration -- fixed at graph capture time ----------------------------

struct DcpAllreduceConfig {
    int batch_size;      // B: tokens in this batch (baked into graph grid)
    int num_heads;       // H: local heads per GPU (e.g., 64 for TP=2 on V3.2)
    int head_dim;        // D: kv_lora_rank for absorbed MLA (e.g., 512 for V3.2)
    int dcp_size;        // N: number of DCP ranks (must be >= 2)
    int rank;            // this rank's index [0, N)

    // External NCCL communicator (managed by #40a, passed as void* — cast
    // to ncclComm_t inside the .cpp to keep this header CUDA/NCCL-free).
    void* nccl_comm;

    // External stable pointers (from DecodeGraphRunner's fixed buffers)
    void*  partial_output;   // [B, H, D] BF16 -- corrected in-place
    float* partial_lse;      // [B, H] FP32 -- this rank's LSE from attention
};

// -- Runner ------------------------------------------------------------------

class DcpAllreduceGraphRunner {
public:
    DcpAllreduceGraphRunner() = default;
    ~DcpAllreduceGraphRunner() { destroy(); }

    // Non-copyable, non-movable (owns CUDA resources)
    DcpAllreduceGraphRunner(const DcpAllreduceGraphRunner&) = delete;
    DcpAllreduceGraphRunner& operator=(const DcpAllreduceGraphRunner&) = delete;
    DcpAllreduceGraphRunner(DcpAllreduceGraphRunner&&) = delete;
    DcpAllreduceGraphRunner& operator=(DcpAllreduceGraphRunner&&) = delete;

    /// Allocate intermediate buffers and capture the CUDA graph.
    /// Caller must set cudaSetDevice before calling.
    /// @throws std::invalid_argument if cfg.dcp_size < 2 (INV-DCP-7).
    /// @throws std::runtime_error on CUDA/NCCL failure.
    /// @param dev  DeviceBackend for this GPU (used for buffer alloc/free).
    void init(const DcpAllreduceConfig& cfg, void* stream,
              DeviceBackend* dev);

    /// Replay the captured graph.
    /// Precondition: init() succeeded (is_captured() == true).
    void replay(void* stream);

    /// Free all CUDA resources.  Idempotent.
    void destroy();

    bool is_captured() const { return captured_; }
    const DcpAllreduceConfig& config() const { return cfg_; }

    // Intermediate buffer accessors (for testing/validation)
    float* gathered_lse_ptr() { return buf_gathered_lse_; }
    float* global_lse_ptr()   { return buf_global_lse_; }

private:
    DcpAllreduceConfig cfg_{};
    bool captured_ = false;

    // CUDA graph objects (void* — cast to cudaGraph_t/cudaGraphExec_t in .cpp)
    void* graph_      = nullptr;
    void* graph_exec_ = nullptr;

    // Owned intermediate buffers (fixed-address, allocated once)
    DeviceBackend* dev_ = nullptr;       // #86b: for buffer alloc/free
    float* buf_gathered_lse_ = nullptr;  // [N, B, H] FP32
    float* buf_global_lse_   = nullptr;  // [B, H] FP32

    void allocate_buffers();
    void free_buffers();
    void capture_graph(void* stream);
};

}  // namespace layerstorm::compute

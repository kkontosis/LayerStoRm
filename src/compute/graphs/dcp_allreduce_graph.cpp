#include "compute/graphs/dcp_allreduce_graph.h"

#include <cuda_runtime.h>
#include <nccl.h>

#include "compute/kernels/attention/dcp_lse_correct.h"
#include "core/device_backend.h"

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

// -- Error checking macros (local to this TU) --------------------------------

#define CUDA_GRAPH_CHECK(expr)                                           \
    do {                                                                 \
        cudaError_t err_ = (expr);                                       \
        if (err_ != cudaSuccess) {                                       \
            throw std::runtime_error(                                    \
                std::string("DcpAllreduceGraph CUDA error: ")            \
                + cudaGetErrorString(err_)                               \
                + " at " + __FILE__ + ":" + std::to_string(__LINE__));   \
        }                                                                \
    } while (0)

#define NCCL_GRAPH_CHECK(expr)                                           \
    do {                                                                 \
        ncclResult_t res_ = (expr);                                      \
        if (res_ != ncclSuccess) {                                       \
            throw std::runtime_error(                                    \
                std::string("DcpAllreduceGraph NCCL error: ")            \
                + ncclGetErrorString(res_)                               \
                + " at " + __FILE__ + ":" + std::to_string(__LINE__));   \
        }                                                                \
    } while (0)

// -- init --------------------------------------------------------------------

void DcpAllreduceGraphRunner::init(const DcpAllreduceConfig& cfg,
                                    void* stream,
                                    DeviceBackend* dev) {
    if (cfg.dcp_size < 2) {
        throw std::invalid_argument(
            "DcpAllreduceGraphRunner requires dcp_size >= 2 (INV-DCP-7), got "
            + std::to_string(cfg.dcp_size));
    }
    if (!cfg.nccl_comm) {
        throw std::invalid_argument(
            "DcpAllreduceGraphRunner requires a valid nccl_comm");
    }
    if (!cfg.partial_output || !cfg.partial_lse) {
        throw std::invalid_argument(
            "DcpAllreduceGraphRunner requires non-null partial_output "
            "and partial_lse");
    }
    if (!dev) {
        throw std::invalid_argument(
            "DcpAllreduceGraphRunner requires a valid DeviceBackend");
    }

    cfg_ = cfg;
    dev_ = dev;
    allocate_buffers();
    capture_graph(stream);
    captured_ = true;

    spdlog::debug("DcpAllreduceGraphRunner: captured graph bs={} heads={} "
                  "dim={} dcp_size={} rank={}",
                  cfg_.batch_size, cfg_.num_heads, cfg_.head_dim,
                  cfg_.dcp_size, cfg_.rank);
}

// -- allocate_buffers --------------------------------------------------------

void DcpAllreduceGraphRunner::allocate_buffers() {
    const size_t B = static_cast<size_t>(cfg_.batch_size);
    const size_t H = static_cast<size_t>(cfg_.num_heads);
    const size_t N = static_cast<size_t>(cfg_.dcp_size);

    buf_gathered_lse_ = static_cast<float*>(
        dev_->device_alloc(N * B * H * sizeof(float)));
    buf_global_lse_ = static_cast<float*>(
        dev_->device_alloc(B * H * sizeof(float)));
    if (!buf_gathered_lse_ || !buf_global_lse_) {
        throw std::runtime_error(
            "DcpAllreduceGraph: device_alloc failed for LSE buffers");
    }
}

// -- capture_graph -----------------------------------------------------------

void DcpAllreduceGraphRunner::capture_graph(void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle);
    auto comm   = static_cast<ncclComm_t>(cfg_.nccl_comm);

    const int B = cfg_.batch_size;
    const int H = cfg_.num_heads;
    const int D = cfg_.head_dim;
    const int N = cfg_.dcp_size;

    // Begin capture on the provided stream.
    // ThreadLocal mode: only operations from the calling thread on this stream
    // are captured.  This allows concurrent captures on different GPUs from
    // different threads (production: one process/thread per GPU).
    CUDA_GRAPH_CHECK(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    // Step 1: NCCL allgather LSE
    //   sendbuf: partial_lse [B*H] FP32 on this rank
    //   recvbuf: gathered_lse [N*B*H] FP32 (concatenation of all ranks)
    NCCL_GRAPH_CHECK(ncclAllGather(
        cfg_.partial_lse,       // sendbuf
        buf_gathered_lse_,      // recvbuf
        static_cast<size_t>(B) * H,  // sendcount (per rank)
        ncclFloat32,
        comm,
        stream));

    // Step 2: DCP LSE correction kernel
    //   Reweights this rank's partial output by exp(local_lse - global_lse)
    DcpLseCorrectParams params{};
    params.output     = cfg_.partial_output;
    params.lses       = buf_gathered_lse_;
    params.global_lse = buf_global_lse_;
    params.B    = B;
    params.H    = H;
    params.D    = D;
    params.N    = N;
    params.rank = cfg_.rank;
    // Contiguous layout strides
    params.stride_o_B   = H * D;
    params.stride_o_H   = D;
    params.stride_o_D   = 1;
    params.stride_lse_N = B * H;
    params.stride_lse_B = H;
    params.stride_lse_H = 1;

    run_dcp_lse_correct_kernel(params, stream);

    // Step 3: NCCL allreduce corrected output (in-place SUM)
    //   Sums the corrected [B, H, D] BF16 outputs across all DCP ranks
    NCCL_GRAPH_CHECK(ncclAllReduce(
        cfg_.partial_output,    // sendbuf
        cfg_.partial_output,    // recvbuf (in-place)
        static_cast<size_t>(B) * H * D,
        ncclBfloat16,
        ncclSum,
        comm,
        stream));

    // End capture and instantiate
    cudaGraph_t graph = nullptr;
    CUDA_GRAPH_CHECK(cudaStreamEndCapture(stream, &graph));
    graph_ = graph;

    cudaGraphExec_t exec = nullptr;
    CUDA_GRAPH_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    graph_exec_ = exec;
}

// -- replay ------------------------------------------------------------------

void DcpAllreduceGraphRunner::replay(void* stream) {
    CUDA_GRAPH_CHECK(cudaGraphLaunch(
        static_cast<cudaGraphExec_t>(graph_exec_),
        static_cast<cudaStream_t>(stream)));
}

// -- destroy -----------------------------------------------------------------

void DcpAllreduceGraphRunner::destroy() {
    if (graph_exec_) {
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec_));
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(static_cast<cudaGraph_t>(graph_));
        graph_ = nullptr;
    }
    free_buffers();
    captured_ = false;
}

// -- free_buffers ------------------------------------------------------------

void DcpAllreduceGraphRunner::free_buffers() {
    if (dev_) {
        dev_->device_free(buf_gathered_lse_);
        dev_->device_free(buf_global_lse_);
    }
    buf_gathered_lse_ = nullptr;
    buf_global_lse_ = nullptr;
}

}  // namespace layerstorm::compute

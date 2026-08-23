#include "compute/graphs/routed_ffn_graph.h"

#include <cuda_runtime.h>

#include <spdlog/spdlog.h>

namespace layerstorm::compute {

bool RoutedFfnGraphRunner::begin_capture(void* stream) {
    last_error_ = "";
    auto s = static_cast<cudaStream_t>(stream);
    // ThreadLocal: only ops from the calling thread on this stream are captured,
    // matching the DcpAllreduceGraphRunner convention (one thread per GPU).
    cudaError_t err = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        last_error_ = cudaGetErrorString(err);
        spdlog::error("RoutedFfnGraph: begin_capture failed: {}", last_error_);
        return false;
    }
    return true;
}

bool RoutedFfnGraphRunner::end_capture(void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(s, &graph);
    if (err != cudaSuccess) {
        last_error_ = cudaGetErrorString(err);
        spdlog::error("RoutedFfnGraph: end_capture failed: {}", last_error_);
        // The capture is invalidated; nothing to instantiate.
        return false;
    }

    // Destroy any prior instance (re-capture path).
    if (graph_exec_) {
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec_));
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(static_cast<cudaGraph_t>(graph_));
        graph_ = nullptr;
    }
    graph_ = graph;

    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiateWithFlags(&exec, graph, 0);
    if (err != cudaSuccess) {
        last_error_ = cudaGetErrorString(err);
        spdlog::error("RoutedFfnGraph: instantiate failed: {}", last_error_);
        cudaGraphDestroy(graph);
        graph_ = nullptr;
        return false;
    }
    graph_exec_ = exec;
    captured_ = true;
    return true;
}

bool RoutedFfnGraphRunner::replay(void* stream) {
    if (!graph_exec_) return false;
    cudaError_t err = cudaGraphLaunch(
        static_cast<cudaGraphExec_t>(graph_exec_),
        static_cast<cudaStream_t>(stream));
    if (err != cudaSuccess) {
        last_error_ = cudaGetErrorString(err);
        spdlog::error("RoutedFfnGraph: replay failed: {}", last_error_);
        return false;
    }
    return true;
}

void RoutedFfnGraphRunner::destroy() {
    if (graph_exec_) {
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec_));
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(static_cast<cudaGraph_t>(graph_));
        graph_ = nullptr;
    }
    captured_ = false;
}

}  // namespace layerstorm::compute

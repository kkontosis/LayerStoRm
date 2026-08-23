#include "compute/graphs/nccl_group_graph.h"

#include <cuda_runtime.h>
#include <nccl.h>

#include <spdlog/spdlog.h>

//==============================================================================
// NCCL Group Graph Runner — designated CUDA/NCCL TU (INV-GPU-1; listed in
// CUDA_DESIGNATED_TUS). Capture pattern follows dcp_allreduce_graph.cpp
// (ThreadLocal stream capture of NCCL collectives), extended to capture ALL
// ranks' streams concurrently from the single orchestrator thread so the
// grouped launch is recorded exactly as the eager path issues it.
//==============================================================================

namespace layerstorm::compute {

bool NcclGroupGraphRunner::init(const std::vector<int>& device_ids,
                                const std::vector<void*>& comms,
                                const std::vector<void*>& streams,
                                const std::vector<std::vector<Op>>& ops) {
    destroy();
    const int n = static_cast<int>(device_ids.size());
    if (n < 2 || comms.size() != device_ids.size()
        || streams.size() != device_ids.size()
        || ops.size() != device_ids.size()) {
        return false;
    }
    for (int r = 0; r < n; ++r) {
        if (!comms[r] || !streams[r] || ops[r].empty()) return false;
        for (const auto& op : ops[r])
            if (!op.buffer || op.count == 0) return false;
    }

    device_ids_ = device_ids;
    execs_.assign(static_cast<size_t>(n), nullptr);

    std::vector<cudaGraph_t> graphs(static_cast<size_t>(n), nullptr);
    std::vector<bool> capturing(static_cast<size_t>(n), false);
    bool ok = true;

    // Begin capture on every rank's stream. ThreadLocal: only this thread's
    // subsequent ops on these streams are captured; pending prior work on the
    // streams is unaffected.
    for (int r = 0; r < n && ok; ++r) {
        cudaSetDevice(device_ids_[r]);
        if (cudaStreamBeginCapture(static_cast<cudaStream_t>(streams[r]),
                                   cudaStreamCaptureModeThreadLocal)
            != cudaSuccess) {
            ok = false;
            break;
        }
        capturing[static_cast<size_t>(r)] = true;
    }

    // The grouped collectives — byte-for-byte the eager launch structure.
    if (ok && ncclGroupStart() != ncclSuccess) ok = false;
    if (ok) {
        for (int r = 0; r < n && ok; ++r) {
            cudaSetDevice(device_ids_[r]);
            for (const auto& op : ops[static_cast<size_t>(r)]) {
                if (ncclAllReduce(op.buffer, op.buffer, op.count,
                                  op.fp32 ? ncclFloat32 : ncclBfloat16,
                                  ncclSum,
                                  static_cast<ncclComm_t>(comms[r]),
                                  static_cast<cudaStream_t>(streams[r]))
                    != ncclSuccess) {
                    ok = false;
                    break;
                }
            }
        }
        if (ncclGroupEnd() != ncclSuccess) ok = false;
    }

    // End capture on every stream that started (also on failure — leave the
    // streams clean). A failed/invalidated capture yields an error + null
    // graph; swallow and fail open.
    for (int r = 0; r < n; ++r) {
        if (!capturing[static_cast<size_t>(r)]) continue;
        cudaSetDevice(device_ids_[r]);
        cudaGraph_t g = nullptr;
        const cudaError_t ec = cudaStreamEndCapture(
            static_cast<cudaStream_t>(streams[r]), &g);
        if (ec != cudaSuccess || !g) {
            ok = false;
            (void)cudaGetLastError();
            continue;
        }
        graphs[static_cast<size_t>(r)] = g;
    }

    if (ok) {
        for (int r = 0; r < n && ok; ++r) {
            cudaSetDevice(device_ids_[r]);
            cudaGraphExec_t e = nullptr;
            if (cudaGraphInstantiate(&e, graphs[static_cast<size_t>(r)],
                                     nullptr, nullptr, 0) != cudaSuccess
                || !e) {
                ok = false;
                (void)cudaGetLastError();
                break;
            }
            execs_[static_cast<size_t>(r)] = e;
        }
    }

    for (int r = 0; r < n; ++r) {
        if (graphs[static_cast<size_t>(r)]) {
            cudaSetDevice(device_ids_[r]);
            cudaGraphDestroy(graphs[static_cast<size_t>(r)]);
        }
    }

    if (!ok) {
        spdlog::warn("NcclGroupGraphRunner: capture failed — staying eager");
        destroy();
        (void)cudaGetLastError();
        return false;
    }
    captured_ = true;
    return true;
}

void NcclGroupGraphRunner::replay(const std::vector<void*>& streams) {
    if (!captured_) return;
    for (size_t r = 0; r < execs_.size(); ++r) {
        cudaSetDevice(device_ids_[r]);
        cudaGraphLaunch(static_cast<cudaGraphExec_t>(execs_[r]),
                        static_cast<cudaStream_t>(streams[r]));
    }
}

void NcclGroupGraphRunner::destroy() {
    for (size_t r = 0; r < execs_.size(); ++r) {
        if (execs_[r]) {
            cudaSetDevice(device_ids_[r]);
            cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(execs_[r]));
        }
    }
    execs_.clear();
    device_ids_.clear();
    captured_ = false;
}

}  // namespace layerstorm::compute

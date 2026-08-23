#pragma once

//==============================================================================
// CollectiveBackend — abstract interface for multi-GPU collective communication.
//
// Separate from DeviceBackend because collectives span multiple GPUs (one comm
// handle, multiple ranks).  Concrete implementations: NcclCollectiveBackend
// (real NCCL), NullCollectiveBackend (no-op for unit tests).
//
// Datatype and reduction-op constants match NCCL's integer encoding so callers
// can pass them through without including <nccl.h>:
//   kFloat32 = 7   (ncclFloat32)
//   kBfloat16 = 9  (ncclBfloat16)
//   kSum = 0       (ncclSum)
//==============================================================================

#include <cstddef>
#include <memory>
#include <vector>

namespace layerstorm::parallelism {

// ── Datatype / reduction-op constants (SDK-free) ────────────────────────────
// Canonical values matching NCCL's enum encoding.  Used by DcpCommunicator
// (caller) and NcclCollectiveBackend (implementor) without including <nccl.h>.

static constexpr int kCollFloat32  = 7;   // ncclFloat32
static constexpr int kCollBfloat16 = 9;   // ncclBfloat16
static constexpr int kCollSum      = 0;   // ncclSum

class CollectiveBackend {
public:
    virtual ~CollectiveBackend() = default;

    // ── Communicator lifecycle ──────────────────────────────────────────────

    /// Create one communicator handle per device (identified by CUDA device ID).
    /// Returns opaque handles indexed by rank [0, N).
    virtual std::vector<void*> create_comms(
        const std::vector<int>& device_ids) = 0;

    /// Destroy a single communicator handle.
    virtual void destroy_comm(void* comm) = 0;

    // ── Group bracket ───────────────────────────────────────────────────────

    /// Begin a grouped collective (NCCL: ncclGroupStart).
    virtual void group_begin() = 0;

    /// End a grouped collective (NCCL: ncclGroupEnd).
    virtual void group_end() = 0;

    // ── Collectives ─────────────────────────────────────────────────────────

    /// Allgather: each rank sends sendcount elements, receives N * sendcount.
    /// @param datatype  NCCL datatype integer (e.g. kFloat32 = 7).
    virtual void allgather(const void* sendbuf, void* recvbuf,
                           size_t sendcount, int datatype,
                           void* comm, void* stream) = 0;

    /// Allreduce: element-wise reduction across all ranks.
    /// @param datatype  NCCL datatype integer.
    /// @param op        NCCL reduction op integer (e.g. kSum = 0).
    virtual void allreduce(const void* sendbuf, void* recvbuf,
                           size_t count, int datatype, int op,
                           void* comm, void* stream) = 0;

    // Non-copyable
    CollectiveBackend(const CollectiveBackend&) = delete;
    CollectiveBackend& operator=(const CollectiveBackend&) = delete;

protected:
    CollectiveBackend() = default;
};

}  // namespace layerstorm::parallelism

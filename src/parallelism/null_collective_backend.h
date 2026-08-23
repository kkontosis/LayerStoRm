#pragma once

//==============================================================================
// NullCollectiveBackend — no-op collective backend for unit tests.
//
// Communicators are distinct heap pointers (no NCCL/CUDA required).
// Collectives are no-ops (can't memcpy device pointers without CUDA).
// Memory uses std::malloc/std::free.
//==============================================================================

#include "parallelism/collective_backend.h"

#include <cstdlib>
#include <memory>
#include <vector>

namespace layerstorm::parallelism {

class NullCollectiveBackend : public CollectiveBackend {
public:
    NullCollectiveBackend() = default;
    ~NullCollectiveBackend() override = default;

    std::vector<void*> create_comms(
        const std::vector<int>& device_ids) override {
        std::vector<void*> result;
        result.reserve(device_ids.size());
        for (size_t i = 0; i < device_ids.size(); ++i)
            result.push_back(new int(counter_++));
        return result;
    }

    void destroy_comm(void* comm) override {
        delete static_cast<int*>(comm);
    }

    void group_begin() override {}
    void group_end() override {}

    void allgather(const void* /*sendbuf*/, void* /*recvbuf*/,
                   size_t /*sendcount*/, int /*datatype*/,
                   void* /*comm*/, void* /*stream*/) override {
        // No-op: can't memcpy device pointers without CUDA.
    }

    void allreduce(const void* /*sendbuf*/, void* /*recvbuf*/,
                   size_t /*count*/, int /*datatype*/, int /*op*/,
                   void* /*comm*/, void* /*stream*/) override {
        // No-op: single-rank allreduce is identity.
    }

private:
    int counter_ = 0;
};

/// Factory function (matches project convention).
inline std::unique_ptr<CollectiveBackend> make_null_collective_backend() {
    return std::make_unique<NullCollectiveBackend>();
}

}  // namespace layerstorm::parallelism

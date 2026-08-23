#include "parallelism/nccl_collective_backend.h"

#include <cuda_runtime.h>
#include <nccl.h>

#include <stdexcept>
#include <string>

namespace layerstorm::parallelism {

// -- Error checking macros (local to this TU) --------------------------------

#define NCCL_COLL_CHECK(expr)                                              \
    do {                                                                   \
        ncclResult_t res_ = (expr);                                        \
        if (res_ != ncclSuccess) {                                         \
            throw std::runtime_error(                                      \
                std::string("NcclCollectiveBackend NCCL error: ")          \
                + ncclGetErrorString(res_)                                 \
                + " at " + __FILE__ + ":" + std::to_string(__LINE__));     \
        }                                                                  \
    } while (0)

// -- NcclCollectiveBackend ---------------------------------------------------

class NcclCollectiveBackend final : public CollectiveBackend {
public:
    NcclCollectiveBackend() = default;
    ~NcclCollectiveBackend() override = default;

    std::vector<void*> create_comms(
        const std::vector<int>& device_ids) override {
        int n = static_cast<int>(device_ids.size());
        std::vector<ncclComm_t> comms(n);
        NCCL_COLL_CHECK(ncclCommInitAll(comms.data(), n, device_ids.data()));
        std::vector<void*> result(n);
        for (int i = 0; i < n; ++i)
            result[i] = static_cast<void*>(comms[i]);
        return result;
    }

    void destroy_comm(void* comm) override {
        if (comm) ncclCommDestroy(static_cast<ncclComm_t>(comm));
    }

    void group_begin() override {
        NCCL_COLL_CHECK(ncclGroupStart());
    }

    void group_end() override {
        NCCL_COLL_CHECK(ncclGroupEnd());
    }

    void allgather(const void* sendbuf, void* recvbuf,
                   size_t sendcount, int datatype,
                   void* comm, void* stream) override {
        NCCL_COLL_CHECK(ncclAllGather(
            sendbuf, recvbuf, sendcount,
            static_cast<ncclDataType_t>(datatype),
            static_cast<ncclComm_t>(comm),
            static_cast<cudaStream_t>(stream)));
    }

    void allreduce(const void* sendbuf, void* recvbuf,
                   size_t count, int datatype, int op,
                   void* comm, void* stream) override {
        NCCL_COLL_CHECK(ncclAllReduce(
            sendbuf, recvbuf, count,
            static_cast<ncclDataType_t>(datatype),
            static_cast<ncclRedOp_t>(op),
            static_cast<ncclComm_t>(comm),
            static_cast<cudaStream_t>(stream)));
    }
};

#undef NCCL_COLL_CHECK

// -- Factory -----------------------------------------------------------------

std::unique_ptr<CollectiveBackend> make_nccl_collective_backend() {
    return std::make_unique<NcclCollectiveBackend>();
}

}  // namespace layerstorm::parallelism

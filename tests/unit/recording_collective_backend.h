#pragma once

// Shared test utility: recording collective backend for NCCL call-sequencing tests.
// Logs group_begin/group_end/allgather/allreduce to a shared vector.
// Used by dcp_executor_test.cpp and command_dispatcher_test.cpp.

#include "parallelism/null_collective_backend.h"

#include <string>
#include <vector>

namespace layerstorm::test {

class RecordingCollectiveBackend : public parallelism::NullCollectiveBackend {
public:
    /// @param log  Shared log vector — caller owns lifetime.
    /// @param prefix  String prepended to each log entry (e.g. "nccl_").
    explicit RecordingCollectiveBackend(std::vector<std::string>& log,
                                       std::string prefix = "")
        : log_(log), prefix_(std::move(prefix)) {}

    void group_begin() override { log_.push_back(prefix_ + "group_begin"); }
    void group_end() override { log_.push_back(prefix_ + "group_end"); }

    void allgather(const void*, void*, size_t, int,
                   void*, void*) override {
        log_.push_back(prefix_ + "allgather");
    }

    void allreduce(const void*, void*, size_t, int, int,
                   void*, void*) override {
        log_.push_back(prefix_ + "allreduce");
    }

private:
    std::vector<std::string>& log_;
    std::string prefix_;
};

}  // namespace layerstorm::test

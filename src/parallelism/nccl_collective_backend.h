#pragma once

//==============================================================================
// NcclCollectiveBackend — real NCCL collective backend.
//
// Wraps ncclCommInitAll, ncclGroupStart/End, ncclAllGather, ncclAllReduce.
// All CUDA/NCCL headers are confined to the .cpp file.
//==============================================================================

#include "parallelism/collective_backend.h"

#include <memory>

namespace layerstorm::parallelism {

/// Create a real NCCL-backed CollectiveBackend.
std::unique_ptr<CollectiveBackend> make_nccl_collective_backend();

}  // namespace layerstorm::parallelism

#pragma once

//==============================================================================
// GpuRef — Lightweight GPU reference (INV-4.18)
//
// Dependency-free GPU identifier passed through all internal APIs.
// Contains position index (for internal array indexing), device ID
// (for CUDA calls), and device type (for asymmetric scheduling).
//
// GpuConfig (generated from schema) nests GpuRef as its `ref` field.
// Higher-level structures pass const GpuRef& instead of bare int.
//==============================================================================

#include <cstdint>

namespace layerstorm::config {

/// GPU device type. The first two values match the schema enum for
/// gpu_object.type (x-externalType: true — not schema-generated, so we are
/// free to append extra enumerators the schema does not list).
///
/// `cpu` (C-6, NumaCpuExpertDevice): a CPU NUMA-node "device" that runs the
/// expert FFN on host cores. It has NO CUDA device ordinal — for a cpu GpuRef,
/// `GpuRef::id` is reused to carry the NUMA node index, NOT a cudaSetDevice
/// ordinal (CPU device leaf code never calls cudaSetDevice). `position` keeps
/// its usual meaning (index into config.hardware.gpus[]).
enum class GpuType { rtx5090, rtx5080, cpu };

/// Lightweight GPU reference. No JSON or heavy dependencies.
struct GpuRef {
    int position = 0;       ///< Index in config.hardware.gpus[] (INV-4.18).
    int id = 0;             ///< CUDA device ID (for cudaSetDevice etc.).
    GpuType type = GpuType::rtx5090;

    bool operator==(const GpuRef&) const = default;
};

}  // namespace layerstorm::config

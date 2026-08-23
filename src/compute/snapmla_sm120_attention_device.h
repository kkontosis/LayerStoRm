// SnapMLA SM120 concrete AttentionDevice.
//
// Composes CudaSm120DeviceBackend (hardware ops) with SnapMLA attention
// pipeline dispatch.  Header is CUDA-free — implementation in
// snapmla_sm120_attention_device.cpp.

#pragma once

#include "core/attention_device.h"

#include <memory>

namespace layerstorm::compute {

/// Factory: creates a SnapMlaSm120AttentionDevice for the given GPU.
std::unique_ptr<AttentionDevice> make_snapmla_sm120_attention_device(
    config::GpuRef gpu);

// ── KD-4f-d.1a: SnapMLA-specific free-function accessors ──────────────────
// The class definition is in the .cpp (CUDA-free header design), so callers
// cannot downcast.  These functions bridge the gap: they take AttentionDevice*
// and internally static_cast to SnapMlaSm120AttentionDevice*.  Caller must
// ensure the device was created with make_snapmla_sm120_attention_device().

/// Set model dimensions for prefill param population.  Init-time.
/// sm_scale: model softmax scale ((qk_nope+qk_rope)^-0.5 · yarn mscale², see
/// compute/rope_table.h). 0 → legacy 1/√(d_c+d_rope) fallback (device unit tests).
void snapmla_device_set_model_dims(AttentionDevice* dev,
    int batch_size, int d_c, int d_rope, int h_q, float sm_scale = 0.0f);

/// Allocate prefill scratch buffers (KV staging + index linearization).
/// Must be called after set_model_dims (needs d_c, d_rope, batch_size).
void snapmla_device_set_prefill_scratch(AttentionDevice* dev, int max_kv_tokens);

/// DET-REDUCE: enable the gated deterministic (bit-reproducible) softmax
/// denominator reduction (config compute.deterministic_reduce ||
/// env LAYERSTORM_DETERMINISTIC_REDUCE=1). Init-time. Default off.
void snapmla_device_set_deterministic_reduce(AttentionDevice* dev, bool enable);

}  // namespace layerstorm::compute

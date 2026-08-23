// TurboQuant MLA SM120 concrete AttentionDevice.
//
// Composes CudaSm120DeviceBackend (hardware ops) with TurboQuant attention
// pipeline dispatch.  Owns per-instance TQ state (resources, layer_idx,
// model dims) instead of mutable globals on a const singleton.
//
// Header is CUDA-free — implementation in tq_sm120_attention_device.cpp.

#pragma once

#include "core/attention_device.h"

#include <memory>

namespace layerstorm::compute {

// Forward-declare TqResources (defined in tq_init.h).
struct TqResources;

class TqSm120AttentionDevice;  // forward-declare for non-virtual setters

/// Factory: creates a TqSm120AttentionDevice for the given GPU.
std::unique_ptr<AttentionDevice> make_tq_sm120_attention_device(
    config::GpuRef gpu);

// ── R0H-1d: TQ-specific free-function accessors ────────────────────────────
// The class definition is in the .cpp (CUDA-free header design), so callers
// cannot downcast.  These functions bridge the gap: they take AttentionDevice*
// and internally static_cast to TqSm120AttentionDevice*.  Caller must ensure
// the device was created with make_tq_sm120_attention_device(); UB otherwise.

/// Set TQ resources (codebook + per-layer rotation matrices).  Init-time.
void tq_device_set_resources(AttentionDevice* dev, const TqResources* res);

/// Set model dimensions (batch_size, d_c, d_rope, h_q, s_q).  Init-time.
/// sm_scale: model softmax scale ((qk_nope+qk_rope)^-0.5 · yarn mscale², see
/// compute/rope_table.h). 0 → legacy 1/√(d_c+d_rope) fallback (device unit tests).
void tq_device_set_model_dims(AttentionDevice* dev,
    int batch_size, int d_c, int d_rope, int h_q, int s_q = 1,
    float sm_scale = 0.0f);

/// Allocate prefill scratch buffers for dequant index linearization.
/// Must be called after set_model_dims (needs d_c, d_rope, batch_size).
void tq_device_set_prefill_scratch(AttentionDevice* dev, int max_kv_tokens);

/// DET-REDUCE (TD-TQ-PREFILL-DETREDUCE-WIRING): enable the gated deterministic
/// (bit-reproducible) softmax denominator reduction for this device's prefill
/// kernels (dense + sparse; config compute.deterministic_reduce ||
/// env LAYERSTORM_DETERMINISTIC_REDUCE — same gate as the SnapMLA twin).
void tq_device_set_deterministic_reduce(AttentionDevice* dev, bool enable);


}  // namespace layerstorm::compute

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "model/quantization/gguf_kquant.h"
#include "model/weight_loader/safetensors_reader.h"
#include "model/weight_loader/tensor_id.h"

namespace layerstorm::model {

class ModelConfig;  // forward

// ── RawTensor ────────────────────────────────────────────────────────────────
// A view into mmap'd tensor data. Does not own the memory.

struct RawTensor {
    std::span<const std::byte> data;
    SafetensorsDtype dtype;
    std::vector<int64_t> shape;

    /// GGUF k-quant type, when this tensor came from a GGUF file and is stored
    /// as one of the six supported k-quants. nullopt for safetensors tensors and
    /// for non-quantized GGUF tensors (norms/scalars/embeddings stored float).
    /// Preserved here (and surfaced on WeightBundle) so the GEMM dispatch
    /// (GG-4 attention / GG-5 experts) can pick the right kernel. GG-6 only
    /// loads bytes + records the type; it does not consume it.
    std::optional<GgufKQuantType> gguf_type;

    int64_t numel() const;
};

// ── WeightBundle ─────────────────────────────────────────────────────────────
// A logical weight: the main weight tensor + any auxiliary tensors (scales).
// For NVFP4: weight(U8) + weight_scale(F8_E4M3) + weight_scale_2(F32) + input_scale(F32).
// For FP8/native: just the weight.

struct WeightBundle {
    TensorId id;  // Identity (component, layer, expert). Role is always 'weight'.
    RawTensor weight;
    std::vector<std::pair<TensorRole, RawTensor>> aux;

    /// Owned buffer for packed expert data (when mmap'd tensors must be linearized).
    /// When non-null, packed_slot spans into this buffer.
    std::shared_ptr<std::vector<std::byte>> owned_buf;

    /// Packed slot data: contiguous DMA-ready layout for ExpertCache.
    /// Set by pack_*_expert(). Empty when unpacked or after release.
    /// For FP8 zero-copy: spans mmap directly (owned_buf is null).
    std::span<const std::byte> packed_slot;

    /// Total bytes across weight + all aux tensors.
    int64_t total_bytes() const;

    /// Find an auxiliary tensor by role. Returns nullptr if not present.
    const RawTensor* find_aux(TensorRole role) const;

    /// The main weight's GGUF k-quant type, if it was loaded from a GGUF file
    /// as a supported k-quant (otherwise nullopt). GG-4/GG-5 read this to
    /// dispatch the GGUF GEMM kernel.
    std::optional<GgufKQuantType> gguf_type() const { return weight.gguf_type; }
};

// ── WeightHandler ────────────────────────────────────────────────────────────
// Abstract interface for quantization-specific weight grouping and validation.
// Each quantization format (NVFP4, FP8, native BF16/F32) has its own handler.

class WeightHandler {
public:
    virtual ~WeightHandler() = default;

    /// Human-readable name for this handler (e.g. "nvfp4", "fp8", "native").
    virtual std::string_view name() const = 0;

    /// Which auxiliary TensorRoles does this format expect alongside each weight?
    /// NVFP4 returns {weight_scale, weight_scale_2, input_scale}.
    /// FP8 and native return {}.
    virtual std::vector<TensorRole> expected_aux_roles() const = 0;

    /// Validate that a weight bundle has correct dtypes, shapes, and aux tensors.
    /// Returns empty string on success, or an error description on failure.
    virtual std::string validate(const WeightBundle& bundle,
                                 const ModelConfig& model_cfg) const = 0;
};

/// Select the appropriate WeightHandler based on the main weight's dtype.
/// U8 -> NVFP4, F8_E4M3/F8_E5M2 -> FP8, BF16/F16/F32 -> native.
/// Returns a non-owning pointer to a static handler instance.
const WeightHandler& handler_for_dtype(SafetensorsDtype dtype);

/// Select the handler for a bundle. GGUF k-quants (gguf_type set) are stored as
/// packed bytes (dtype U8 == NVFP4's dtype) but use the GGUF handler; everything
/// else falls back to handler_for_dtype(). Use this from the loader.
const WeightHandler& handler_for_bundle(const WeightBundle& bundle);

}  // namespace layerstorm::model

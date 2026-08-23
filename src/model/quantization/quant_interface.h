#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include "config/config_parser.h"

namespace layerstorm::model {

// ── ExpertShape ──────────────────────────────────────────────────────────────
// POD struct describing a single expert's weight dimensions.

struct ExpertShape {
    int hidden_size;        // e.g. 7168 (V3.2/K2.5), 6144 (GLM-5)
    int intermediate_size;  // moe_intermediate_size, e.g. 2048

    constexpr int64_t gate_params() const {
        return static_cast<int64_t>(hidden_size) * intermediate_size;
    }
    constexpr int64_t up_params() const {
        return static_cast<int64_t>(hidden_size) * intermediate_size;
    }
    constexpr int64_t down_params() const {
        return static_cast<int64_t>(intermediate_size) * hidden_size;
    }
    constexpr int64_t total_params() const {
        return gate_params() + up_params() + down_params();
    }
};

// ── Projection ───────────────────────────────────────────────────────────────
// Identifies a single expert sub-component (gate/up/down projection).

enum class Projection { gate, up, down };

// ── ScaleDtype + MemoryLayout ────────────────────────────────────────────────
// Descriptors for kernel dispatch and memory accounting.

enum class ScaleDtype { ue8m0, fp16, fp32, none };

struct MemoryLayout {
    int bits_per_weight;    // 4 for FP4/INT4, 8 for FP8
    int group_size;         // elements per scale factor (16 NVFP4, 32 INT4, etc.)
    bool has_zero_point;    // true for INT4 asymmetric only
    ScaleDtype scale_dtype; // scale factor type
    int alignment_bytes;    // DMA/tensor core alignment
};

// ── QuantInterface ───────────────────────────────────────────────────────────
// Pure abstract interface for quantization format metadata.
// Concrete implementations: NVFP4 (#9), FP8 (#10).

class QuantInterface {
public:
    virtual ~QuantInterface() = default;

    virtual std::string_view name() const = 0;
    virtual config::WeightQuant weight_quant() const = 0;
    virtual int64_t bytes_per_expert(const ExpertShape& shape) const = 0;
    virtual int64_t bytes_per_projection(const ExpertShape& shape, Projection proj) const = 0;
    virtual double bytes_per_element() const = 0;
    virtual double dequant_flops_per_element() const = 0;
    virtual bool tensor_core_eligible() const = 0;
    virtual MemoryLayout memory_layout() const = 0;

    /// Weight-only bytes per projection (excluding group scales and scalar scales).
    /// Used to compute the boundary between weight data and scale data within
    /// an ExpertCache slot region. NVFP4: ceil(params/2). FP8: params.
    virtual int64_t weight_bytes_per_projection(const ExpertShape& shape,
                                                Projection proj) const = 0;

    /// Bytes per projection that are replicated (not divided by TP).
    /// NVFP4: 8 (weight_scale_2 + input_scale). FP8/native: 0.
    virtual int64_t replicated_bytes_per_projection() const { return 0; }
};

} // namespace layerstorm::model

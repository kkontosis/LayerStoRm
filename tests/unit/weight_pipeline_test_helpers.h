#pragma once

// Shared test helpers for weight pipeline tests (WP-2, WP-3, WP-4).
// Provides synthetic WeightBundle / LoadedModel construction for NVFP4 and FP8.

#include <cstring>
#include <memory>
#include <vector>

#include "model/quantization/gguf_kquant.h"
#include "model/quantization/quant_interface.h"
#include "model/weight_loader/weight_handler.h"
#include "model/weight_loader/weight_loader.h"
#include "config/config_parser.h"

namespace layerstorm::test {

/// Mixin base providing shared helpers for weight pipeline test fixtures.
/// Subclass your gtest fixture from this AND ::testing::Test.
class WeightPipelineHelpers {
protected:
    /// Allocate a buffer of `size` bytes filled with `fill_byte`.
    std::shared_ptr<std::vector<std::byte>> alloc(size_t size, uint8_t fill_byte) {
        auto buf = std::make_shared<std::vector<std::byte>>(
            size, static_cast<std::byte>(fill_byte));
        data_bufs_.push_back(buf);
        return buf;
    }

    /// Create a minimal Config for prepack_experts / ModelConfig.
    config::Config make_config(int n_layers, int n_experts,
                               int hidden, int intermediate,
                               int first_moe_layer) {
        config::Config cfg;
        cfg.model.num_hidden_layers = n_layers;
        cfg.model.hidden_size = hidden;
        cfg.model.moe_intermediate_size = intermediate;
        cfg.model.n_routed_experts = n_experts;
        cfg.model.first_k_dense_replace = first_moe_layer;
        cfg.model.moe_layer_freq = 1;
        cfg.model.weights_path = "/fake/model";
        return cfg;
    }

    /// Create NVFP4 weight bundles (gate/up/down) for one expert.
    /// scale_seed != 0 fills the group-scale bytes with a deterministic LCG
    /// pattern (uniform 0xBB otherwise — layout bugs are invisible to uniform
    /// data). is_* set the per-projection input_scale aux scalars.
    std::vector<model::WeightBundle> make_nvfp4_bundles(
            const model::ExpertShape& shape,
            uint32_t scale_seed = 0,
            float is_gate = 1.0f, float is_up = 1.0f, float is_down = 1.0f) {
        std::vector<model::WeightBundle> bundles;
        uint32_t lcg = scale_seed;

        auto make_proj = [&](model::TensorComponent comp, int64_t params,
                             float is_val) {
            int64_t wb = (params + 1) / 2;
            auto wbuf = alloc(wb, 0xAA);

            int64_t sb = (params + 15) / 16;
            auto sbuf = alloc(sb, 0xBB);
            if (scale_seed != 0) {
                for (int64_t i = 0; i < sb; ++i) {
                    lcg = lcg * 1664525u + 1013904223u;
                    (*sbuf)[static_cast<size_t>(i)] =
                        static_cast<std::byte>((lcg >> 24) & 0xFF);
                }
            }

            auto ws2_buf = alloc(sizeof(float), 0xCC);
            auto is_buf  = alloc(sizeof(float), 0xDD);
            float ws2_val = 1.0f;
            std::memcpy(ws2_buf->data(), &ws2_val, sizeof(float));
            std::memcpy(is_buf->data(), &is_val, sizeof(float));

            model::WeightBundle b;
            b.id.component = comp;
            b.weight.dtype = model::SafetensorsDtype::U8;
            b.weight.data = std::span<const std::byte>(wbuf->data(), wbuf->size());
            b.aux.push_back({model::TensorRole::weight_scale,
                             model::RawTensor{std::span<const std::byte>(sbuf->data(), sbuf->size()),
                                              model::SafetensorsDtype::U8, {sb}}});
            b.aux.push_back({model::TensorRole::weight_scale_2,
                             model::RawTensor{std::span<const std::byte>(ws2_buf->data(), ws2_buf->size()),
                                              model::SafetensorsDtype::F32, {1}}});
            b.aux.push_back({model::TensorRole::input_scale,
                             model::RawTensor{std::span<const std::byte>(is_buf->data(), is_buf->size()),
                                              model::SafetensorsDtype::F32, {1}}});
            return b;
        };

        bundles.push_back(make_proj(model::TensorComponent::gate_proj,
                                    shape.gate_params(), is_gate));
        bundles.push_back(make_proj(model::TensorComponent::up_proj,
                                    shape.up_params(), is_up));
        bundles.push_back(make_proj(model::TensorComponent::down_proj,
                                    shape.down_params(), is_down));
        return bundles;
    }

    /// Create GGUF k-quant weight bundles (gate/up/down) for one expert with
    /// per-projection k-quant types (GG-10 per-layer mixed support). Each
    /// projection's block bytes are filled with the given fill byte so slot
    /// layout / offset bugs are visible in the packed output.
    std::vector<model::WeightBundle> make_gguf_bundles(
            const model::ExpertShape& shape,
            model::GgufKQuantType gate_t, model::GgufKQuantType up_t,
            model::GgufKQuantType down_t,
            uint8_t gate_fill = 0xA0, uint8_t up_fill = 0xB0,
            uint8_t down_fill = 0xC0) {
        std::vector<model::WeightBundle> bundles;

        auto make_proj = [&](model::TensorComponent comp,
                             model::GgufKQuantType t,
                             int64_t out_features, int64_t in_features,
                             uint8_t fill) {
            int64_t bytes = model::gguf::gguf_packed_bytes(
                out_features, in_features, t);
            auto buf = alloc(static_cast<size_t>(bytes), fill);

            model::WeightBundle b;
            b.id.component = comp;
            b.weight.dtype = model::SafetensorsDtype::U8;
            b.weight.gguf_type = t;
            b.weight.data =
                std::span<const std::byte>(buf->data(), buf->size());
            return b;
        };

        bundles.push_back(make_proj(model::TensorComponent::gate_proj, gate_t,
                                    shape.intermediate_size, shape.hidden_size,
                                    gate_fill));
        bundles.push_back(make_proj(model::TensorComponent::up_proj, up_t,
                                    shape.intermediate_size, shape.hidden_size,
                                    up_fill));
        bundles.push_back(make_proj(model::TensorComponent::down_proj, down_t,
                                    shape.hidden_size, shape.intermediate_size,
                                    down_fill));
        return bundles;
    }

    /// Create FP8 weight bundles (gate/up/down) for one expert.
    std::vector<model::WeightBundle> make_fp8_bundles(
            const model::ExpertShape& shape) {
        std::vector<model::WeightBundle> bundles;

        auto make_proj = [&](model::TensorComponent comp, int64_t n, int64_t k) {
            int64_t wb = n * k;
            auto wbuf = alloc(wb, 0xEE);

            int64_t nb = (n + 127) / 128;
            int64_t kb = (k + 127) / 128;
            int64_t sb = nb * kb * static_cast<int64_t>(sizeof(float));
            auto sbuf = alloc(sb, 0x00);
            float one = 1.0f;
            for (int64_t i = 0; i < nb * kb; ++i) {
                std::memcpy(sbuf->data() + i * sizeof(float), &one, sizeof(float));
            }

            model::WeightBundle b;
            b.id.component = comp;
            b.weight.dtype = model::SafetensorsDtype::F8_E4M3;
            b.weight.data = std::span<const std::byte>(wbuf->data(), wbuf->size());
            b.weight.shape = {n, k};
            b.aux.push_back({model::TensorRole::weight_scale,
                             model::RawTensor{std::span<const std::byte>(sbuf->data(), sbuf->size()),
                                              model::SafetensorsDtype::F32, {nb, kb}}});
            return b;
        };

        bundles.push_back(make_proj(model::TensorComponent::gate_proj,
                                    shape.intermediate_size, shape.hidden_size));
        bundles.push_back(make_proj(model::TensorComponent::up_proj,
                                    shape.intermediate_size, shape.hidden_size));
        bundles.push_back(make_proj(model::TensorComponent::down_proj,
                                    shape.hidden_size, shape.intermediate_size));
        return bundles;
    }

    /// Build a LoadedModel with synthetic routed experts.
    model::LoadedModel make_mock_model(int n_layers, int n_experts,
                                       const model::ExpertShape& shape,
                                       int first_moe_layer,
                                       model::SafetensorsDtype dtype) {
        model::LoadedModel model;
        model.layers.resize(n_layers);
        for (int l = 0; l < n_layers; ++l) {
            model.layers[l].layer_idx = l;
            if (l >= first_moe_layer) {
                model.layers[l].routed_experts.resize(n_experts);
                for (int e = 0; e < n_experts; ++e) {
                    if (dtype == model::SafetensorsDtype::U8) {
                        model.layers[l].routed_experts[e] = make_nvfp4_bundles(shape);
                    } else {
                        model.layers[l].routed_experts[e] = make_fp8_bundles(shape);
                    }
                }
            }
        }
        return model;
    }

    std::vector<std::shared_ptr<std::vector<std::byte>>> data_bufs_;
};

}  // namespace layerstorm::test

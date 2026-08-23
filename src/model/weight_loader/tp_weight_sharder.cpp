#include "model/weight_loader/tp_weight_sharder.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <numeric>
#include <stdexcept>

#include "model/model_config.h"
#include "model/quantization/gguf_kquant.h"
#include "model/weight_loader/safetensors_reader.h"

namespace layerstorm::model {

// ── shard_mode_for ──────────────────────────────────────────────────────────

ShardMode shard_mode_for(TensorComponent component) {
    switch (component) {
        case TensorComponent::q_b_proj:
        case TensorComponent::kv_b_proj:
            return ShardMode::kColumnParallel;
        case TensorComponent::o_proj:
            return ShardMode::kRowParallel;
        // V4-2c (TD-V4-TP, 2026-08-21): DeepSeek-V4 grouped o_proj shards
        // BY GROUP. o_proj_a [o_groups*o_lora_rank, group_dim] row-major,
        // groups-major slabs → out-dim (shape[0]) split = rank r gets the
        // contiguous group slabs [r*G/tp, (r+1)*G/tp) — exactly the head
        // spans rank r's column-sharded q_b produces (group g = heads
        // [g*h_q/G, (g+1)*h_q/G)). o_proj_b [hidden, o_groups*o_lora_rank]
        // splits the K dim over the same group slice (row-parallel; the
        // executor allreduces the partial hidden after stage 2). All other
        // V4 components (q_a, kv_a, norms, compressor_*, indexer_*, hc_*,
        // attn_sinks, output_hc) stay REPLICATED — mirrors the ticket-C
        // pinned plan (÷tp only q_b/o_a/o_b) and the sglang V4 TP shape.
        case TensorComponent::o_proj_a:
            return ShardMode::kColumnParallel;
        case TensorComponent::o_proj_b:
            return ShardMode::kRowParallel;
        default:
            return ShardMode::kReplicated;
    }
}

// ── shard_mode_for_ffn ──────────────────────────────────────────────────────

ShardMode shard_mode_for_ffn(TensorComponent component) {
    switch (component) {
        case TensorComponent::gate_proj:
        case TensorComponent::up_proj:
            return ShardMode::kColumnParallel;
        case TensorComponent::down_proj:
            return ShardMode::kRowParallel;
        default:
            return ShardMode::kReplicated;
    }
}

// ── ShardedTensor ───────────────────────────────────────────────────────────

int64_t ShardedTensor::numel() const {
    // Scalar tensors have shape [] (empty) but contain 1 element.
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

int64_t ShardedTensor::size_bytes() const {
    // GGUF k-quant weights are stored as packed U8 bytes whose count is NOT
    // numel*dtype_size (=out*in) but the packed `out*(in/QK)*block_bytes`. The
    // logical shape is [out, in] but the buffer is the packed super-block stream
    // (TD-GGUF-ATTN-TP-SHARD). Report the true packed byte count so the upload
    // plan (gguf_attention_layer_bytes) and the sharded tensor agree.
    if (gguf_type.has_value() && shape.size() >= 2) {
        return gguf::gguf_packed_bytes(shape[0], shape[1], *gguf_type);
    }
    return numel() * static_cast<int64_t>(dtype_size(dtype));
}

// ── ShardedWeightBundle ─────────────────────────────────────────────────────

int64_t ShardedWeightBundle::total_bytes() const {
    int64_t total = weight.size_bytes();
    for (auto& [role, st] : aux) {
        total += st.size_bytes();
    }
    return total;
}

const ShardedTensor* ShardedWeightBundle::find_aux(TensorRole role) const {
    for (auto& [r, st] : aux) {
        if (r == role) return &st;
    }
    return nullptr;
}

// ── TpWeightSharder ─────────────────────────────────────────────────────────

TpWeightSharder::TpWeightSharder(const ModelConfig& model_cfg, int tp_degree)
    : tp_degree_(tp_degree),
      num_heads_(model_cfg.raw().num_attention_heads) {
    if (tp_degree_ < 1) {
        throw std::invalid_argument("tp_degree must be >= 1");
    }
    if (tp_degree_ > 1 && num_heads_ % tp_degree_ != 0) {
        throw std::invalid_argument(
            std::format("num_attention_heads ({}) must be divisible by tp_degree ({})",
                        num_heads_, tp_degree_));
    }
}

// ── wrap_replicated ─────────────────────────────────────────────────────────

ShardedTensor TpWeightSharder::wrap_replicated(const RawTensor& tensor) {
    return ShardedTensor{
        .data = tensor.data,
        .dtype = tensor.dtype,
        .shape = tensor.shape,
        .gguf_type = tensor.gguf_type,
        .owned_buf = nullptr,
    };
}

// ── shard_column_parallel ───────────────────────────────────────────────────
// Split axis 0 (output dim).  In row-major layout, this gives contiguous
// sub-tensors → zero-copy subspan.

ShardedTensor TpWeightSharder::shard_column_parallel(const RawTensor& tensor,
                                                      int rank) const {
    if (tensor.shape.size() < 2) {
        // 1D tensor: replicate (shouldn't happen for projections, but safe)
        return wrap_replicated(tensor);
    }

    int64_t rows = tensor.shape[0];
    int64_t cols = tensor.shape[1];

    if (rows % tp_degree_ != 0) {
        throw std::invalid_argument(std::format(
            "column-parallel shard: output rows ({}) not divisible by tp_degree ({})",
            rows, tp_degree_));
    }
    int64_t rows_per_rank = rows / tp_degree_;

    // GGUF k-quant: the buffer is the packed super-block stream laid out as
    // [out, (in/QK)*block_bytes]. Output rows are independent and contiguous in
    // the packed buffer, so each rank takes a whole-row sub-span — but the row
    // stride is the PACKED row size, not cols*1 (TD-GGUF-ATTN-TP-SHARD).
    if (tensor.gguf_type.has_value()) {
        const auto type = *tensor.gguf_type;
        const int qk = gguf::block_values(type);
        if (cols % qk != 0) {
            throw std::invalid_argument(std::format(
                "column-parallel GGUF shard: in_features ({}) not a multiple of "
                "QK ({}) for the k-quant type", cols, qk));
        }
        const int64_t packed_row_bytes = (cols / qk) * gguf::block_bytes(type);
        const size_t byte_offset =
            static_cast<size_t>(rank * rows_per_rank * packed_row_bytes);
        const size_t byte_count =
            static_cast<size_t>(rows_per_rank * packed_row_bytes);

        std::vector<int64_t> sharded_shape = tensor.shape;
        sharded_shape[0] = rows_per_rank;

        return ShardedTensor{
            .data = tensor.data.subspan(byte_offset, byte_count),
            .dtype = tensor.dtype,
            .shape = std::move(sharded_shape),
            .gguf_type = tensor.gguf_type,
            .owned_buf = nullptr,
        };
    }

    size_t elem_size = dtype_size(tensor.dtype);

    size_t byte_offset = static_cast<size_t>(rank * rows_per_rank * cols) * elem_size;
    size_t byte_count = static_cast<size_t>(rows_per_rank * cols) * elem_size;

    std::vector<int64_t> sharded_shape = tensor.shape;
    sharded_shape[0] = rows_per_rank;

    return ShardedTensor{
        .data = tensor.data.subspan(byte_offset, byte_count),
        .dtype = tensor.dtype,
        .shape = std::move(sharded_shape),
        .owned_buf = nullptr,
    };
}

// ── shard_row_parallel ──────────────────────────────────────────────────────
// Split axis 1 (input dim).  Non-contiguous in row-major → pack into owned buf.

ShardedTensor TpWeightSharder::shard_row_parallel(const RawTensor& tensor,
                                                    int rank) const {
    if (tensor.shape.size() < 2) {
        return wrap_replicated(tensor);
    }

    int64_t rows = tensor.shape[0];
    int64_t cols = tensor.shape[1];

    if (cols % tp_degree_ != 0) {
        throw std::invalid_argument(std::format(
            "row-parallel shard: input cols ({}) not divisible by tp_degree ({})",
            cols, tp_degree_));
    }
    int64_t cols_per_rank = cols / tp_degree_;
    int64_t col_start = rank * cols_per_rank;

    // GGUF k-quant: the input dim K is split across ranks, but a super-block
    // packs QK contiguous K-values, so the split MUST be super-block-aligned —
    // you cannot split a super-block across ranks (TD-GGUF-ATTN-TP-SHARD). Each
    // rank takes (cols/tp/QK) contiguous blocks per row; gather them into a
    // packed owned buffer using PACKED block strides (not dtype_size).
    if (tensor.gguf_type.has_value()) {
        const auto type = *tensor.gguf_type;
        const int qk = gguf::block_values(type);
        const int blk = gguf::block_bytes(type);
        if (cols % qk != 0) {
            throw std::invalid_argument(std::format(
                "row-parallel GGUF shard: in_features ({}) not a multiple of "
                "QK ({}) for the k-quant type", cols, qk));
        }
        if (cols_per_rank % qk != 0) {
            throw std::invalid_argument(std::format(
                "row-parallel GGUF shard: per-rank in_features ({} = {}/{}) not a "
                "multiple of QK ({}) — a {}-value super-block cannot be split "
                "across ranks", cols_per_rank, cols, tp_degree_, qk, qk));
        }

        const int64_t blocks_total = cols / qk;
        const int64_t blocks_per_rank = cols_per_rank / qk;
        const size_t src_row_bytes = static_cast<size_t>(blocks_total) * blk;
        const size_t dst_row_bytes = static_cast<size_t>(blocks_per_rank) * blk;
        const size_t block_byte_offset =
            static_cast<size_t>(rank * blocks_per_rank) * blk;

        auto buf = std::make_shared<std::vector<std::byte>>(
            static_cast<size_t>(rows) * dst_row_bytes);
        const auto* src = tensor.data.data();
        auto* dst = buf->data();
        for (int64_t r = 0; r < rows; ++r) {
            std::memcpy(dst + r * dst_row_bytes,
                        src + r * src_row_bytes + block_byte_offset,
                        dst_row_bytes);
        }

        std::vector<int64_t> sharded_shape = tensor.shape;
        sharded_shape[1] = cols_per_rank;

        return ShardedTensor{
            .data = std::span<const std::byte>(buf->data(), buf->size()),
            .dtype = tensor.dtype,
            .shape = std::move(sharded_shape),
            .gguf_type = tensor.gguf_type,
            .owned_buf = std::move(buf),
        };
    }

    size_t elem_size = dtype_size(tensor.dtype);

    size_t shard_bytes = static_cast<size_t>(rows * cols_per_rank) * elem_size;
    auto buf = std::make_shared<std::vector<std::byte>>(shard_bytes);

    const auto* src = tensor.data.data();
    auto* dst = buf->data();

    size_t src_row_bytes = static_cast<size_t>(cols) * elem_size;
    size_t dst_row_bytes = static_cast<size_t>(cols_per_rank) * elem_size;
    size_t col_byte_offset = static_cast<size_t>(col_start) * elem_size;

    for (int64_t r = 0; r < rows; ++r) {
        std::memcpy(dst + r * dst_row_bytes,
                    src + r * src_row_bytes + col_byte_offset,
                    dst_row_bytes);
    }

    std::vector<int64_t> sharded_shape = tensor.shape;
    sharded_shape[1] = cols_per_rank;

    return ShardedTensor{
        .data = std::span<const std::byte>(buf->data(), buf->size()),
        .dtype = tensor.dtype,
        .shape = std::move(sharded_shape),
        .owned_buf = std::move(buf),
    };
}

// ── shard_scale ─────────────────────────────────────────────────────────────
// Scale tensors follow their parent weight's sharding axis.
// Scalar-like scales (weight_scale_2, input_scale) are always replicated.

ShardedTensor TpWeightSharder::shard_scale(const RawTensor& scale,
                                            TensorRole role,
                                            ShardMode parent_mode,
                                            int rank) const {
    // Scalar-like scales: always replicated
    if (role == TensorRole::weight_scale_2 || role == TensorRole::input_scale) {
        return wrap_replicated(scale);
    }

    // Bias: always replicated
    if (role == TensorRole::bias) {
        return wrap_replicated(scale);
    }

    // 1D or scalar scale: replicate
    if (scale.shape.size() < 2) {
        return wrap_replicated(scale);
    }

    // TODO:DEBT TD-53m: NVFP4 scale sharding for row-parallel may be incorrect (group alignment)
    // 2D scale: follow parent's sharding axis
    switch (parent_mode) {
        case ShardMode::kColumnParallel:
            return shard_column_parallel(scale, rank);
        case ShardMode::kRowParallel:
            return shard_row_parallel(scale, rank);
        case ShardMode::kReplicated:
            return wrap_replicated(scale);
    }
    return wrap_replicated(scale);  // unreachable
}

// ── shard_attention ─────────────────────────────────────────────────────────

ShardedWeightBundle TpWeightSharder::shard_attention(const WeightBundle& bundle,
                                                      int rank) const {
    ShardMode mode = (tp_degree_ <= 1) ? ShardMode::kReplicated
                                       : shard_mode_for(bundle.id.component);

    ShardedTensor sharded_weight;
    switch (mode) {
        case ShardMode::kColumnParallel:
            sharded_weight = shard_column_parallel(bundle.weight, rank);
            break;
        case ShardMode::kRowParallel:
            sharded_weight = shard_row_parallel(bundle.weight, rank);
            break;
        case ShardMode::kReplicated:
            sharded_weight = wrap_replicated(bundle.weight);
            break;
    }

    // Preserve the GGUF k-quant type through sharding (GG-6 → GG-4): sharding
    // changes shape/bytes but not the per-tensor quant type. The dcp_executor
    // reads this off the uploaded attention bundle to pick the GGUF GEMM kernel.
    sharded_weight.gguf_type = bundle.weight.gguf_type;

    // Shard auxiliary tensors (scales) following the parent weight's mode
    std::vector<std::pair<TensorRole, ShardedTensor>> sharded_aux;
    sharded_aux.reserve(bundle.aux.size());
    for (auto& [aux_role, aux_raw] : bundle.aux) {
        sharded_aux.emplace_back(aux_role, shard_scale(aux_raw, aux_role, mode, rank));
    }

    return ShardedWeightBundle{
        .id = bundle.id,
        .weight = std::move(sharded_weight),
        .aux = std::move(sharded_aux),
    };
}

// ── shard_attention_layer ───────────────────────────────────────────────────

std::vector<ShardedWeightBundle> TpWeightSharder::shard_attention_layer(
    const std::vector<WeightBundle>& attention,
    const std::vector<WeightBundle>& indexer,
    int rank) const {

    std::vector<ShardedWeightBundle> result;
    result.reserve(attention.size() + indexer.size());

    for (auto& bundle : attention) {
        result.push_back(shard_attention(bundle, rank));
    }

    // DSA indexer weights are always replicated
    for (auto& bundle : indexer) {
        ShardedWeightBundle sb{
            .id = bundle.id,
            .weight = wrap_replicated(bundle.weight),
            .aux = {},
        };
        sb.aux.reserve(bundle.aux.size());
        for (auto& [aux_role, aux_raw] : bundle.aux) {
            sb.aux.emplace_back(aux_role, wrap_replicated(aux_raw));
        }
        result.push_back(std::move(sb));
    }

    return result;
}

// ── shard_ffn ──────────────────────────────────────────────────────────────

ShardedWeightBundle TpWeightSharder::shard_ffn(const WeightBundle& bundle,
                                                int rank) const {
    ShardMode mode = (tp_degree_ <= 1) ? ShardMode::kReplicated
                                       : shard_mode_for_ffn(bundle.id.component);

    ShardedTensor sharded_weight;
    switch (mode) {
        case ShardMode::kColumnParallel:
            sharded_weight = shard_column_parallel(bundle.weight, rank);
            break;
        case ShardMode::kRowParallel:
            sharded_weight = shard_row_parallel(bundle.weight, rank);
            break;
        case ShardMode::kReplicated:
            sharded_weight = wrap_replicated(bundle.weight);
            break;
    }

    std::vector<std::pair<TensorRole, ShardedTensor>> sharded_aux;
    sharded_aux.reserve(bundle.aux.size());
    for (auto& [aux_role, aux_raw] : bundle.aux) {
        sharded_aux.emplace_back(aux_role, shard_scale(aux_raw, aux_role, mode, rank));
    }

    return ShardedWeightBundle{
        .id = bundle.id,
        .weight = std::move(sharded_weight),
        .aux = std::move(sharded_aux),
    };
}

// ── shard_ffn_layer ────────────────────────────────────────────────────────

std::vector<ShardedWeightBundle> TpWeightSharder::shard_ffn_layer(
    const std::vector<WeightBundle>& bundles, int rank) const {

    std::vector<ShardedWeightBundle> result;
    result.reserve(bundles.size());

    for (auto& bundle : bundles) {
        result.push_back(shard_ffn(bundle, rank));
    }

    return result;
}

}  // namespace layerstorm::model

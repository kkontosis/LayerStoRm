#pragma once

//==============================================================================
// TP Weight Sharder
//
// Stateless post-load transform that splits attention projection weights
// across TP ranks.  Sits between load_weights() (full unsharded mmap) and
// GPU upload.
//
// Sharding modes for attention components:
//   ColumnParallel (split axis 0): q_b_proj, kv_b_proj  — zero-copy subspan
//   RowParallel    (split axis 1): o_proj               — packed copy
//   Replicated     (no split):     q_a_proj, kv_a_proj, norms, DSA indexer
//
// Scale tensors follow their parent weight's sharding axis.
// Scalar scales (weight_scale_2, input_scale) are always replicated.
//==============================================================================

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "model/weight_loader/weight_handler.h"
#include "model/weight_loader/tensor_id.h"

namespace layerstorm::model {

class ModelConfig;

// ── Shard mode classification ───────────────────────────────────────────────

/// How a tensor component is distributed across TP ranks.
enum class ShardMode {
    kReplicated,       ///< Full copy on every rank
    kColumnParallel,   ///< Split axis 0 (output dim). Contiguous in row-major.
    kRowParallel,      ///< Split axis 1 (input dim). Non-contiguous — packed copy.
};

/// Determine the sharding mode for an attention-related TensorComponent.
/// Returns kReplicated for all non-shardable components.
ShardMode shard_mode_for(TensorComponent component);

/// Determine the sharding mode for an FFN-like TensorComponent.
/// gate_proj/up_proj → column-parallel, down_proj → row-parallel, else replicated.
ShardMode shard_mode_for_ffn(TensorComponent component);

// ── ShardedTensor ───────────────────────────────────────────────────────────

/// A tensor view for a single TP rank.  Either a zero-copy sub-span of the
/// original mmap'd data (column-parallel / replicated), or an owned buffer
/// with packed data (row-parallel).
struct ShardedTensor {
    std::span<const std::byte> data;   ///< Points into mmap or owned_buf
    SafetensorsDtype dtype;
    std::vector<int64_t> shape;        ///< Sharded shape

    /// GGUF k-quant type, preserved from the source RawTensor (GG-6). Set only
    /// for GGUF-quantized tensors; nullopt otherwise. GG-4 reads this off the
    /// sharded attention bundle to dispatch the GGUF GEMM kernel per projection.
    std::optional<GgufKQuantType> gguf_type;

    /// Owned buffer for non-contiguous shards (row-parallel).
    /// Empty for zero-copy (column-parallel, replicated).
    std::shared_ptr<std::vector<std::byte>> owned_buf;

    int64_t numel() const;
    int64_t size_bytes() const;
    bool is_owned() const { return owned_buf != nullptr && !owned_buf->empty(); }
};

// ── ShardedWeightBundle ─────────────────────────────────────────────────────

/// A per-rank sharded version of WeightBundle.
struct ShardedWeightBundle {
    TensorId id;
    ShardedTensor weight;
    std::vector<std::pair<TensorRole, ShardedTensor>> aux;

    int64_t total_bytes() const;
    const ShardedTensor* find_aux(TensorRole role) const;
};

// ── TpWeightSharder ─────────────────────────────────────────────────────────

/// Stateless sharder for TP weight distribution.
/// Constructed once with model config; all shard methods are const.
class TpWeightSharder {
public:
    /// @param model_cfg  Model configuration (for num_heads, head dims)
    /// @param tp_degree  Number of TP ranks (1 = no sharding)
    /// @throws std::invalid_argument if num_attention_heads % tp_degree != 0
    TpWeightSharder(const ModelConfig& model_cfg, int tp_degree);

    int tp_degree() const { return tp_degree_; }

    /// Shard a single attention WeightBundle for the given rank.
    /// When tp_degree=1, returns a trivial wrapper (zero-copy of everything).
    ShardedWeightBundle shard_attention(const WeightBundle& bundle, int rank) const;

    /// Shard all attention + indexer bundles for one layer, for one rank.
    /// Returns a vector of ShardedWeightBundle in the same order as the inputs.
    /// Indexer bundles are always replicated.
    std::vector<ShardedWeightBundle> shard_attention_layer(
        const std::vector<WeightBundle>& attention,
        const std::vector<WeightBundle>& indexer,
        int rank) const;

    /// Shard a single FFN-like bundle (gate/up = column-parallel, down = row-parallel).
    ShardedWeightBundle shard_ffn(const WeightBundle& bundle, int rank) const;

    /// Shard all bundles for one FFN-like layer (shared expert or dense FFN).
    std::vector<ShardedWeightBundle> shard_ffn_layer(
        const std::vector<WeightBundle>& bundles, int rank) const;

private:
    ShardedTensor shard_column_parallel(const RawTensor& tensor, int rank) const;
    ShardedTensor shard_row_parallel(const RawTensor& tensor, int rank) const;
    ShardedTensor shard_scale(const RawTensor& scale, TensorRole role,
                              ShardMode parent_mode, int rank) const;
    static ShardedTensor wrap_replicated(const RawTensor& tensor);

    int tp_degree_;
    int num_heads_;
};

}  // namespace layerstorm::model

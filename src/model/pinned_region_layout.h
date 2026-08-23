#pragma once

// Pinned Region Layout
//
// Single authoritative computation of the pinned VRAM region size per TP GPU.
// Consumed by:
//   1. LayerRegistry::estimate_gpu_budgets() — for VRAM allocator sizing
//   2. Engine::upload_pinned_weights() — for verification
//
// Canonical upload order (both layout and upload must follow this exactly):
//   1. Embedding (column-parallel on vocab axis)
//   2. Output head weight + bias (column-parallel on vocab axis)
//   3. Per hidden layer (0..num_hidden_layers-1):
//      a. Attention projections + internal norms (TP-sharded per TpWeightSharder)
//      b. Layer norms: input_layernorm + post_attention_layernorm (replicated, BF16)
//      c. Gating: gate_weight + e_score_correction_bias (replicated)
//      d. Shared expert: gate/up/down (TP-sharded, column/row-parallel)
//      e. Dense FFN: gate/up/down (TP-sharded, column/row-parallel)
//   4. Final norm (replicated, BF16)
//   5. MTP block layers (attention + norms + gating + shared expert + 6 MTP-specific tensors)

#include <cstdint>

#include "config/config_parser.h"
#include "model/model_config.h"  // V4AttentionType (V4-3a)

namespace layerstorm::config {
struct ModelConfig;
}  // namespace layerstorm::config

namespace layerstorm::model {

class ModelConfig;
class QuantInterface;

// ── Shared sizing helpers ───────────────────────────────────────────────────

inline constexpr double kBf16BytesPerElement = 2.0;

struct AttentionDims {
    int64_t q_a_params;
    int64_t q_b_params;
    int64_t kv_a_params;
    int64_t kv_b_params;
    int64_t o_params;
    int64_t q_a_norm_params;
    int64_t kv_a_norm_params;
    int64_t indexer_params;       // total indexer params (projections + norms)
    int64_t indexer_norm_params;  // k_norm weight + k_norm bias (F32, subset of indexer_params)

    // Per-projection input feature count (the GEMM contraction / weight column
    // dim). Needed for exact GGUF k-quant packed-byte sizing (GG-4): the packed
    // formula is out*(in/QK)*block_bytes and requires in % QK == 0. The out dim
    // is params/in. Filled by compute_attn_dims.
    int64_t q_a_in;     // = hidden_size
    int64_t q_b_in;     // = q_lora_rank
    int64_t kv_a_in;    // = hidden_size
    int64_t kv_b_in;    // = kv_lora_rank
    int64_t o_in;       // = num_attention_heads * v_head_dim
};

AttentionDims compute_attn_dims(const config::ModelConfig& m);

/// Compute pinned bytes for one attention layer.
/// Uses WeightQuant to determine per-projection storage dtype:
///   NVFP4: o_proj at NVFP4 (weight+scale+scalars), all others at BF16
///   FP8:   all projections at FP8 (1.0 bpe)
///   BF16:  all projections at BF16 (2.0 bpe)
/// When force_bf16_oproj is true, o_proj is computed as BF16 regardless of wq
/// (V3.2 NVFP4 checkpoints store MTP o_proj as BF16).
int64_t attention_layer_bytes(const AttentionDims& d,
                              config::WeightQuant wq,
                              bool include_kv_b, int tp,
                              bool include_indexer = true,
                              bool force_bf16_oproj = false);

// ── DeepSeek-V4 attention sizing (V4-3a) ────────────────────────────────────
//
// V4 attention weights are BF16/F32-NATIVE in the GGUF (attention requant is
// gated OFF for V4 — dossier ticket B), so sizing is dtype-exact and does NOT
// depend on quantization.weights.  Per-layer anatomy varies with
// compress_ratios[l] (spec/DEEPSEEK4_PLAN.md V4-3a):
//   all layers:  q_a + q_a_norm + q_b + attn_kv (single 512 latent, K==V) +
//                kv_a_norm + attn_sinks + grouped o_proj (o_a, o_b) +
//                mHC hc_attn_* / hc_ffn_* stream weights
//   CSA (r=4):   + compressor (wkv/wgate [hidden, 2·head_dim], APE, norm)
//                + Lightning Indexer (proj, q_b, indexer-compressor)
//   HCA (r=128): + heavy compressor (wkv/wgate [hidden, head_dim], APE, norm)
//   SWA (r=0):   projections + mHC only
// TP: only the head-parallel projections (q_b, o_a, o_b) are divided by tp;
// the actual V4 sharder is V4-2b — revisit the split there if it lands
// differently.  Norms/APE/sinks/mHC are sized at native F32 (upper bound if a
// later upload converts to BF16 — safe for the generic-gguf slot rule, which
// only rejects UNDER-sized slots).
int64_t v4_attention_layer_bytes(const config::ModelConfig& m,
                                 V4AttentionType type, int tp);

/// Model-level output_hc_{fn,base,scale} bytes (F32; mHC head collapse).
int64_t v4_output_hc_bytes(const config::ModelConfig& m);

/// V4 hash-layer token→expert table bytes: tid2eid [num_experts_per_tok,
/// vocab_size] I32 (layers l < num_hash_layers carry it INSTEAD of the
/// exp_probs_b gating bias).
int64_t v4_hash_gating_table_bytes(const config::ModelConfig& m);

/// Per-tensor alignment budget for upload padding (~2000 sub-tensors × 15 bytes each).
inline constexpr int64_t kUploadAlignBudget = 32 * 1024;

// ── PinnedRegionLayout ──────────────────────────────────────────────────────

struct PinnedRegionLayout {
    int64_t total_bytes = 0;

    // Component subtotals (diagnostics/logging)
    int64_t embedding_bytes = 0;
    int64_t output_head_bytes = 0;
    int64_t attention_bytes = 0;
    int64_t layer_norm_bytes = 0;
    int64_t final_norm_bytes = 0;
    int64_t gating_bytes = 0;
    int64_t shared_expert_bytes = 0;
    int64_t dense_ffn_bytes = 0;
    int64_t mtp_bytes = 0;
    int64_t output_hc_bytes = 0;  // V4 mHC head collapse (model-level, F32)
};

/// Compute the pinned region layout for a single TP rank.
/// Uses model dimensions + config to predict exact byte counts
/// that will be uploaded by Engine::upload_pinned_weights().
PinnedRegionLayout compute_pinned_layout(
    const ModelConfig& model_cfg,
    const config::Config& cfg,
    const QuantInterface& expert_quant,
    int tp_degree,
    int rank);

/// Returns false when kv_b_proj is absorbed into the KV cache format
/// (SnapMLA, TurboQuant MLA) and won't appear in the checkpoint.
bool has_kv_b_in_checkpoint(const config::Config& cfg);

/// Returns true only if the model checkpoint includes an lm_head bias tensor.
bool has_output_head_bias(const config::Config& cfg);

}  // namespace layerstorm::model

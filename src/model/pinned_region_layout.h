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
struct GgufNonExpertWidths;  // weight_loader.h (GF3.15)

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

// ── glm5_next attention sizing (GF3.3) ──────────────────────────────────────
//
// GLM-5.3-Flash ships TWO per-layer attention anatomies (spec/GLM-5.3-FLASH-
// MODELINFO.md §3a/§3b/§3c/§3e), and the per-layer bytes must be computed from
// the one the layer actually has — ModelConfig::is_linear_attention_layer(l):
//
//   KDA linear attention (34 layers) — entirely BF16 except A_log/dt_bias
//   (F32); the WHOLE layer sits in the FP8 skip list, so its size does NOT
//   depend on the checkpoint weight quant at all:
//     q/k/v_proj  [H*D, h] BF16 x3        column-parallel (head axis)  ÷tp
//     b_proj      [H, h]   BF16           column-parallel              ÷tp
//     f_a/g_a     [D, h]   BF16 x2        REPLICATED (rank-D bottleneck)
//     f_b/g_b     [H*D, D] BF16 x2        column-parallel              ÷tp
//     q/k/v_conv1d[H*D,1,K] BF16 x3       column-parallel              ÷tp
//     A_log       [H]      F32            column-parallel              ÷tp
//     dt_bias     [H*D]    F32            column-parallel              ÷tp
//     o_norm      [D]      BF16           REPLICATED
//     o_proj      [h, H*D] BF16           row-parallel                 ÷tp
//
//   NoPE sparse MLA (11 layers + the MTP block) — MIXED precision inside one
//   layer: q_a/q_b/kv_a/o_proj are FP8 with F32 [ceil(N/128), ceil(K/128)]
//   blockwise scales, while kv_b_proj, the two layernorms and the ENTIRE
//   indexer (IndexPool compressor included) are BF16 in the skip list.
//
//   mHC (§3e) adds hc_{attn,ffn}_fn [(2+hc)*hc, hc*h] BF16 + _base [(2+hc)*hc]
//   F32 + _scale [3] F32 to every HIDDEN layer (0..44).  NOTE the dtype mix
//   differs from DeepSeek-V4, whose hc set is all-F32 — do not reuse the V4
//   formula.  The MTP block layer 45 has NO hc tensors, and glm5_next has no
//   model-level output_hc set either.
//
// TP: exactly the components the TpWeightSharder splits are divided here (see
// shard_mode_for), scales included — the sharder-vs-sizing byte equality is
// what validate_plan enforces at boot.
//
// GGUF releases (GF3.9) take a separate UPPER-BOUND arm: the unsloth
// GLM-5.3-Flash GGUF ships every attention matrix packed Q8_0 and every
// vector/norm/APE F32 (scratchpad/gf39/SURVEY_BOOT.md §3.7), so matrices are
// sized BF16 (>= any GGUF packing, and exact for a load-time dequant to BF16)
// and F32 tensors are sized F32 — except the four norms validate_plan halves
// by dtype (q_a/kv_a layernorm, indexer k_norm weight+bias), sized BF16 to
// match that correction.  The kpool compressor APE is F32 on the GGUF path
// (BF16 on the native path) because the GGUF ships and uploads it F32.  The
// resulting slack is legal: validate_plan's `gguf_upper_bound` allowance
// covers glm5_next + any gguf quant, and only an UNDER-sized slot is a bug.
//
// GF3.15 (TD-AUTOCONFIG-PINNED-BYTES-UPPER-BOUND) narrows that GGUF arm: when
// `widths` is supplied (a header-only pre-scan of the checkpoint, built once by
// the engine — `gguf_non_expert_widths_from_path`) every PACKED matrix is sized
// at its REAL k-quant width instead of the BF16 bound.  The upper bound was
// never physics, only the absence of the header scan at plan time; on
// GLM-5.3-Flash it cost 5.2 GiB of pinned VRAM on the single TP GPU.  Tensors
// the loader TRANSFORMS at load keep their transformed width and are never
// looked up: the split kv_b halves (assembled to one combined BF16 kv_b_proj),
// the IndexPool compressor gate (dequanted to BF16 for a BF16-only GEMM),
// hc_*_fn (widened to F32 for launch_mhc_pre) and the four dtype-halved norms.
// `widths == nullptr`, or a tensor absent from it (plain-float in the file),
// keeps the previous sizing — still an upper bound, because every remaining
// load-time transform on those (requant_bundle_to_q8_0, F32→BF16 norms) makes
// the uploaded tensor SMALLER.
//
// @throws std::runtime_error for nvfp4 — no such glm5_next artifact path is
//         supported yet (GF3.3).
int64_t glm5_next_attention_layer_bytes(const config::ModelConfig& m,
                                        config::WeightQuant wq,
                                        bool linear_attention,
                                        bool include_hc,
                                        int tp,
                                        const GgufNonExpertWidths* widths = nullptr,
                                        int layer_idx = -1);

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
/// `widths` (GF3.15): the checkpoint's real per-tensor GGUF widths from a header
/// pre-scan; nullptr keeps the pre-load upper-bound sizing.  Only the glm5_next
/// GGUF arm consumes it today — see glm5_next_attention_layer_bytes.
PinnedRegionLayout compute_pinned_layout(
    const ModelConfig& model_cfg,
    const config::Config& cfg,
    const QuantInterface& expert_quant,
    int tp_degree,
    int rank,
    const GgufNonExpertWidths* widths = nullptr);

/// Returns false when kv_b_proj is absorbed into the KV cache format
/// (SnapMLA, TurboQuant MLA) and won't appear in the checkpoint.
bool has_kv_b_in_checkpoint(const config::Config& cfg);

/// Returns true only if the model checkpoint includes an lm_head bias tensor.
bool has_output_head_bias(const config::Config& cfg);

}  // namespace layerstorm::model

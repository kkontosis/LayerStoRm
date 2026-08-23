#include "model/pinned_region_layout.h"

#include <algorithm>

#include "config/config_parser.h"
#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/pinned_upload_plan.h"
#include "model/quantization/gguf_kquant.h"
#include "model/quantization/quant_interface.h"
#include "model/weight_loader/weight_loader.h"  // GgufModelExpertTypes

namespace layerstorm::model {

// ── has_kv_b_in_checkpoint ─────────────────────────────────────────────────

bool has_kv_b_in_checkpoint(const config::Config& cfg) {
    // DeepSeek V3.2 (and other MLA models) ship kv_b_proj in the checkpoint
    // even when the attention backend absorbs it into the KV cache format.
    // The weight must be loaded and pinned so the engine can use it during
    // the first KV write (absorption step).
    // kv_b_proj exists whenever kv_lora_rank > 0 (MLA architecture).
    // V4 (native MQA-over-latent) has NO kv_b anywhere — attn_kv IS the single
    // latent (dossier ticket B) — and its config still carries the inert MLA
    // schema defaults (kv_lora_rank=512), so gate on architecture first.
    if (cfg.model.architecture == config::Architecture::deepseek_v4) return false;
    return cfg.model.kv_lora_rank > 0;
}

bool has_output_head_bias(const config::Config& /*cfg*/) {
    // V3.2 and GLM-5 checkpoints have no lm_head.bias tensor.
    return false;
}

// ── Shared sizing helpers (declared in pinned_region_layout.h) ────────────

// TODO:DEBT TD-53e: Non-MLA attention dims incorrect (assumes MLA projections always)
AttentionDims compute_attn_dims(const config::ModelConfig& m) {
    AttentionDims d{};
    d.q_a_params = static_cast<int64_t>(m.hidden_size) * m.q_lora_rank;
    d.q_b_params = static_cast<int64_t>(m.q_lora_rank) *
                   (m.num_attention_heads * (m.qk_nope_head_dim + m.qk_rope_head_dim));
    d.kv_a_params = static_cast<int64_t>(m.hidden_size) *
                    (m.kv_lora_rank + m.qk_rope_head_dim);
    d.kv_b_params = static_cast<int64_t>(m.kv_lora_rank) *
                    (m.num_attention_heads * (m.qk_nope_head_dim + m.v_head_dim));
    d.o_params = static_cast<int64_t>(m.num_attention_heads) * m.v_head_dim * m.hidden_size;
    d.q_a_norm_params = m.q_lora_rank;
    d.kv_a_norm_params = m.kv_lora_rank;

    // Per-projection input (contraction) dim for GGUF packed-byte sizing.
    d.q_a_in  = m.hidden_size;
    d.q_b_in  = m.q_lora_rank;
    d.kv_a_in = m.hidden_size;
    d.kv_b_in = m.kv_lora_rank;
    d.o_in    = static_cast<int64_t>(m.num_attention_heads) * m.v_head_dim;

    if (m.index_topk > 0) {
        int64_t q_idx_b = static_cast<int64_t>(m.q_lora_rank) *
                          m.index_n_heads * m.index_head_dim;
        int64_t k_idx = static_cast<int64_t>(m.hidden_size) * m.index_head_dim;
        int64_t k_idx_norm = m.index_head_dim;
        int64_t k_idx_norm_bias = m.index_head_dim;
        int64_t weights_proj = static_cast<int64_t>(m.hidden_size) * m.index_n_heads;
        d.indexer_params = q_idx_b + k_idx + k_idx_norm + k_idx_norm_bias + weights_proj;
        d.indexer_norm_params = k_idx_norm + k_idx_norm_bias;
    }

    return d;
}

// NVFP4 byte count for a single projection: weight(FP4) + scale(FP8) + 2 scalars.
static int64_t nvfp4_projection_bytes(int64_t params) {
    int64_t weight = (params + 1) / 2;         // 2 FP4 values per byte
    int64_t scale  = (params + 15) / 16;       // 1 FP8 scale per 16 elements
    return weight + scale + 2 * static_cast<int64_t>(sizeof(float));  // +weight_scale_2 +input_scale
}

// GGUF uniform-variant attention sizing (GG-4 / TD-GGUF-ATTN-UPLOAD-SIZING):
// each projection's packed byte count comes from the exact gguf_packed_bytes
// primitive (out*(in/QK)*block_bytes), NOT the scalar bytes_per_element. Norms
// are F32→BF16 (2 B/elem) regardless. The generic `gguf` enum is per-tensor
// mixed and cannot be sized from the config alone — callers must use the
// per-tensor bundle types (engine pre-scans the file); we reject it loudly.
static int64_t gguf_attention_layer_bytes(const AttentionDims& d,
                                          GgufKQuantType type,
                                          bool include_kv_b, int tp,
                                          bool include_indexer) {
    auto packed = [&](int64_t params, int64_t in_features) -> int64_t {
        const int64_t out_features = in_features > 0 ? params / in_features : 0;
        return gguf::gguf_packed_bytes(out_features, in_features, type);
    };

    // Replicated projections (q_a, kv_a) — full bytes (not divided by TP).
    int64_t replicated = packed(d.q_a_params, d.q_a_in)
                       + packed(d.kv_a_params, d.kv_a_in);

    // Norms: F32→BF16 during upload (TD-73c).
    int64_t norms = (d.q_a_norm_params + d.kv_a_norm_params) * 2;

    // Indexer: projections BF16, norms F32→BF16 (indexer is never GGUF-packed).
    int64_t indexer = 0;
    if (include_indexer && d.indexer_params > 0) {
        int64_t norm_params = d.indexer_norm_params;
        int64_t proj_params = d.indexer_params - norm_params;
        indexer = static_cast<int64_t>(proj_params * kBf16BytesPerElement)
                + norm_params * 2;
    }

    // Sharded projections (q_b, kv_b, o) — TP-divided. The packed bytes scale
    // with the sharded output-row count, so divide the packed total by tp
    // (k-quant rows are independent super-block sequences → exact for the row
    // splits these projections use; over-allocates by < 1 block on odd splits).
    int64_t shardable_qb = packed(d.q_b_params, d.q_b_in) / tp;
    int64_t shardable_kvb = 0;
    if (include_kv_b) shardable_kvb = packed(d.kv_b_params, d.kv_b_in) / tp;
    int64_t o_bytes = packed(d.o_params, d.o_in) / tp;

    int64_t total = replicated + norms + indexer + shardable_qb + shardable_kvb
                  + o_bytes;
    return (total + 15) & ~int64_t{15};
}

int64_t attention_layer_bytes(const AttentionDims& d,
                              config::WeightQuant wq,
                              bool include_kv_b, int tp,
                              bool include_indexer,
                              bool force_bf16_oproj) {
    using WQ = config::WeightQuant;

    // GGUF uniform variants (gguf_qX_k / gguf_q8_0): size each projection from
    // the exact packed-byte primitive. Generic `gguf` is per-tensor mixed —
    // attention projections in such checkpoints are frequently F16/BF16 (only
    // experts low-bit), and we can't see per-tensor types here, so size the
    // slot at BF16 (2 B/elem): a safe UPPER BOUND for every GGUF type (all six
    // k-quants/Q8_0 are < 2 B/elem) so the actual packed upload always fits.
    // The actual byte count copied is the bundle's true packed size.
    if (gguf::is_gguf_weight_quant(wq)) {
        if (wq == WQ::gguf) {
            const double bf16 = kBf16BytesPerElement;
            int64_t replicated = static_cast<int64_t>(
                (d.q_a_params + d.kv_a_params) * bf16);
            int64_t norms = (d.q_a_norm_params + d.kv_a_norm_params) * 2;
            int64_t indexer = 0;
            if (include_indexer && d.indexer_params > 0) {
                int64_t norm_params = d.indexer_norm_params;
                int64_t proj_params = d.indexer_params - norm_params;
                indexer = static_cast<int64_t>(proj_params * bf16)
                        + norm_params * 2;
            }
            int64_t shardable = d.q_b_params;
            if (include_kv_b) shardable += d.kv_b_params;
            int64_t shardable_bytes = static_cast<int64_t>(shardable * bf16) / tp;
            int64_t o_bytes = static_cast<int64_t>(d.o_params * bf16) / tp;
            int64_t total = replicated + norms + indexer + shardable_bytes
                          + o_bytes;
            return (total + 15) & ~int64_t{15};
        }
        return gguf_attention_layer_bytes(
            d, gguf::type_from_weight_quant(wq), include_kv_b, tp,
            include_indexer);
    }

    // For NVFP4 checkpoints: q/kv projections stored as BF16, only o_proj as NVFP4.
    // For FP8/BF16: all projections use the weight quant dtype uniformly.
    const bool nvfp4 = (wq == WQ::nvfp4);
    const double proj_bpe = nvfp4 ? kBf16BytesPerElement
                                  : bytes_per_element(wq);

    // Replicated projections (q_a, kv_a): BF16 for NVFP4, uniform bpe otherwise.
    int64_t replicated = static_cast<int64_t>(
        (d.q_a_params + d.kv_a_params) * proj_bpe);

    // Norm weights: converted F32→BF16 during upload (TD-73c).
    int64_t norms = (d.q_a_norm_params + d.kv_a_norm_params) * 2;

    // Indexer weights: BF16 for projections, F32 for norms (k_norm + k_norm_bias).
    // indexer_params includes both but they use different dtypes. Separate them:
    // k_norm + k_norm_bias = 2 * index_head_dim params at F32.
    // The rest (q_idx_b + k_idx + weights_proj) at BF16.
    int64_t indexer = 0;
    if (include_indexer && d.indexer_params > 0) {
        int64_t norm_params = d.indexer_norm_params;
        int64_t proj_params = d.indexer_params - norm_params;
        indexer = static_cast<int64_t>(proj_params * kBf16BytesPerElement)
                + norm_params * 2;  // BF16 after F32→BF16 conversion (TD-73c)
    }

    // Shardable: q_b, kv_b at proj_bpe; o_proj at weight quant format.
    int64_t shardable_qkv_params = d.q_b_params;
    if (include_kv_b) shardable_qkv_params += d.kv_b_params;
    int64_t shardable_qkv = static_cast<int64_t>(shardable_qkv_params * proj_bpe) / tp;

    // o_proj: divide each component by TP independently (matches sharder behavior).
    // Scalars (weight_scale_2, input_scale) are replicated (not divided by TP).
    int64_t o_bytes;
    if (nvfp4 && !force_bf16_oproj) {
        int64_t o_weight = (d.o_params + 1) / 2;
        int64_t o_scale  = (d.o_params + 15) / 16;
        o_bytes = o_weight / tp + o_scale / tp
                + 2 * static_cast<int64_t>(sizeof(float));
    } else {
        double o_bpe = force_bf16_oproj ? kBf16BytesPerElement : proj_bpe;
        o_bytes = static_cast<int64_t>(d.o_params * o_bpe) / tp;
    }

    int64_t total = replicated + norms + indexer + shardable_qkv + o_bytes;
    // Round up to 16 bytes so the following slot (layer_norm) starts aligned
    // for vectorized kernels (RMSNorm, FP8 quant).
    return (total + 15) & ~int64_t{15};
}

// ── DeepSeek-V4 attention sizing (V4-3a) ───────────────────────────────────
//
// Shapes are the GGUF census ground truth (scratchpad/DS4_DOSSIER.md §0.3):
//   q_a [hidden, q_lora]                q_a_norm [q_lora] F32
//   q_b [q_lora, heads·head_dim]        attn_kv [hidden, head_dim]
//   kv_a_norm [head_dim] F32            attn_sinks [heads] F32
//   o_a [heads/o_groups·head_dim, o_groups·o_lora]   o_b [o_groups·o_lora, hidden]
//   hc_{attn,ffn}_fn [hc·hidden, (2+hc)·hc] F32, base [(2+hc)·hc] F32, scale [3] F32
//   CSA compressor: wkv/wgate [hidden, 2·head_dim], APE [2·head_dim, 4] F32,
//                   norm [head_dim] F32
//   HCA compressor: wkv/wgate [hidden, head_dim], APE [head_dim, 128] F32,
//                   norm [head_dim] F32
//   CSA indexer: proj [hidden, idx_heads], q_b [q_lora, idx_heads·idx_dim],
//                idx-compressor wkv/wgate [hidden, 2·idx_dim],
//                APE [2·idx_dim, 4] F32, norm [idx_dim] F32
int64_t v4_attention_layer_bytes(const config::ModelConfig& m,
                                 V4AttentionType type, int tp) {
    constexpr int64_t kBf16 = 2;
    constexpr int64_t kF32 = 4;
    const int64_t hidden = m.hidden_size;
    const int64_t heads  = m.num_attention_heads;
    const int64_t hd     = m.head_dim;        // 512 (Flash)
    const int64_t qlr    = m.q_lora_rank;     // 1024
    const int64_t olr    = m.o_lora_rank;     // 1024
    const int64_t og     = std::max(1, m.o_groups);
    const int64_t hc     = std::max(1, m.hc_mult);
    const int64_t t      = std::max(1, tp);

    // Replicated projections + norms + sinks.
    int64_t b = 0;
    b += hidden * qlr * kBf16;                 // q_a
    b += qlr * kF32;                           // q_a_norm (weighted, F32)
    b += hidden * hd * kBf16;                  // attn_kv (single latent; K==V)
    b += hd * kF32;                            // kv_a_norm (weighted, F32)
    b += heads * kF32;                         // attn_sinks (per Q head, F32)

    // Head-parallel projections (÷tp — see V4-2b note in the header).
    b += qlr * heads * hd * kBf16 / t;         // q_b
    b += heads * hd * olr * kBf16 / t;         // o_a  (per-group [hd·heads/og, olr])
    b += og * olr * hidden * kBf16 / t;        // o_b

    // mHC stream weights (F32, replicated): one set each for attn + ffn.
    const int64_t hc_mix = (2 + hc) * hc;      // 24 at hc=4
    const int64_t hc_set = hc * hidden * hc_mix + hc_mix + 3;
    b += 2 * hc_set * kF32;

    // Compressor + indexer per layer type.
    if (type == V4AttentionType::kCsa) {
        const int64_t coff = 2;                // prev|cur overlap halves
        b += 2 * hidden * coff * hd * kBf16;   // wkv + wgate
        b += coff * hd * 4 * kF32;             // APE [2·hd, ratio=4]
        b += hd * kF32;                        // compressor norm
        // Lightning Indexer (CSA layers only).
        const int64_t ih = m.index_n_heads;    // 64
        const int64_t id = m.index_head_dim;   // 128
        b += hidden * ih * kBf16;              // indexer proj
        // V4-2c: the indexer q_b is REPLICATED (the lightning top-k needs
        // all 64 index heads' scores on every rank; the sharder replicates
        // the whole indexer bundle and the executor computes the full-head
        // selection per rank). The original ÷tp here under-sized the plan
        // by 8.4 MB × 21 CSA layers at tp=2 and overflowed the pinned
        // region into the next VRAM region at upload.
        b += qlr * ih * id * kBf16;            // indexer q_b (replicated)
        b += 2 * hidden * coff * id * kBf16;   // idx-compressor wkv + wgate
        b += coff * id * 4 * kF32;             // idx-compressor APE [2·id, 4]
        b += id * kF32;                        // idx-compressor norm
    } else if (type == V4AttentionType::kHca) {
        b += 2 * hidden * hd * kBf16;          // wkv + wgate (single half)
        b += hd * 128 * kF32;                  // APE [hd, ratio=128]
        b += hd * kF32;                        // compressor norm
    }
    // kSwa: projections + mHC only.

    // 16-byte round-up so the following slot starts aligned (same rule as
    // attention_layer_bytes).
    return (b + 15) & ~int64_t{15};
}

int64_t v4_output_hc_bytes(const config::ModelConfig& m) {
    constexpr int64_t kF32 = 4;
    const int64_t hc = std::max(1, m.hc_mult);
    // output_hc_fn [hc·hidden, hc] + base [hc] + scale [1], all F32.
    return (hc * static_cast<int64_t>(m.hidden_size) * hc + hc + 1) * kF32;
}

int64_t v4_hash_gating_table_bytes(const config::ModelConfig& m) {
    // ffn_gate_tid2eid [num_experts_per_tok, vocab_size] I32.
    return static_cast<int64_t>(m.num_experts_per_tok) * m.vocab_size * 4;
}

// ── compute_pinned_layout (derived from PinnedUploadPlan) ─────────────────

PinnedRegionLayout compute_pinned_layout(
    const ModelConfig& model_cfg,
    const config::Config& cfg,
    const QuantInterface& expert_quant,
    int tp_degree,
    int rank) {

    // GG-9: the pinned REGION size must be an upper bound over the actual upload.
    // A mixed `gguf` checkpoint's shared experts can be Q8_0 (the largest k-quant)
    // while `expert_quant` carries the routed types (e.g. gate Q6_K) — sizing the
    // region with the routed types under-allocates and the upload overflows into
    // the KV region. Size shared + dense FFN at Q8_0 (the max k-quant ⇒ a safe
    // upper bound for any mix); the actual per-slot upload still uses exact types,
    // so the surplus is harmless trailing slack. (Pre-load, so we cannot read the
    // real per-owner types here — TD-GG9-REGION-Q8-UPPER-BOUND.)
    static const GgufModelExpertTypes kQ8{GgufKQuantType::Q8_0, GgufKQuantType::Q8_0,
                                          GgufKQuantType::Q8_0};
    const bool gguf_generic = cfg.quantization.weights == config::WeightQuant::gguf;
    auto plan = build_upload_plan(model_cfg, cfg, expert_quant, tp_degree, rank,
                                  gguf_generic ? &kQ8 : nullptr,
                                  gguf_generic ? &kQ8 : nullptr);
    const int num_hidden = model_cfg.raw().num_hidden_layers;

    PinnedRegionLayout layout{};
    for (const auto& slot : plan.slots) {
        const bool is_mtp = (slot.layer_idx >= num_hidden);
        switch (slot.component) {
            case PinnedComponent::embedding:
                layout.embedding_bytes += slot.size_bytes; break;
            case PinnedComponent::output_head_weight:
            case PinnedComponent::output_head_bias:
                layout.output_head_bytes += slot.size_bytes; break;
            case PinnedComponent::attention:
                (is_mtp ? layout.mtp_bytes : layout.attention_bytes) += slot.size_bytes; break;
            case PinnedComponent::layer_norm:
                (is_mtp ? layout.mtp_bytes : layout.layer_norm_bytes) += slot.size_bytes; break;
            case PinnedComponent::gating_weight:
            case PinnedComponent::gating_bias:
                (is_mtp ? layout.mtp_bytes : layout.gating_bytes) += slot.size_bytes; break;
            case PinnedComponent::shared_expert_gate:
            case PinnedComponent::shared_expert_up:
            case PinnedComponent::shared_expert_gate_scales:
            case PinnedComponent::shared_expert_up_scales:
            case PinnedComponent::shared_expert_gate_scalar:
            case PinnedComponent::shared_expert_up_scalar:
            case PinnedComponent::shared_expert_down:
                (is_mtp ? layout.mtp_bytes : layout.shared_expert_bytes) += slot.size_bytes;
                break;
            case PinnedComponent::dense_ffn_gate:
            case PinnedComponent::dense_ffn_up:
            case PinnedComponent::dense_ffn_gate_scales:
            case PinnedComponent::dense_ffn_up_scales:
            case PinnedComponent::dense_ffn_gate_scalar:
            case PinnedComponent::dense_ffn_up_scalar:
            case PinnedComponent::dense_ffn_down:
                layout.dense_ffn_bytes += slot.size_bytes; break;
            case PinnedComponent::final_norm:
                layout.final_norm_bytes += slot.size_bytes; break;
            case PinnedComponent::gating_hash_table:
                (is_mtp ? layout.mtp_bytes : layout.gating_bytes) += slot.size_bytes; break;
            case PinnedComponent::output_hc:
                layout.output_hc_bytes += slot.size_bytes; break;
            case PinnedComponent::mtp_embed_tokens:
            case PinnedComponent::mtp_eh_proj:
            case PinnedComponent::mtp_enorm:
            case PinnedComponent::mtp_hnorm:
            case PinnedComponent::mtp_shared_head_weight:
            case PinnedComponent::mtp_shared_head_norm:
                layout.mtp_bytes += slot.size_bytes; break;
        }
    }
    // KD-4f-d.1b: Each sub-tensor is 16-byte aligned during upload to satisfy
    // GEMM kernel alignment requirements. Budget worst-case padding for all
    // upload calls (~2000 sub-tensors × 15 bytes each ≈ 30 KB).
    layout.total_bytes = plan.total_bytes + kUploadAlignBudget;
    return layout;
}

}  // namespace layerstorm::model

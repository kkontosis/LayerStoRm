#include "model/layer_registry.h"

#include <algorithm>

#include <stdexcept>

#include "config/config_resolver.h"
#include "model/pinned_region_layout.h"
#include "model/quantization/gguf_kquant.h"
#include "model/weight_loader/dspark_loader.h"
#include "speculation/dspark_runtime.h"  // DSP-3: runtime scratch sizing

namespace layerstorm::model {

// ── bytes_per_element helpers ───────────────────────────────────────────────

double bytes_per_element(config::AttentionQuant q) {
    switch (q) {
        case config::AttentionQuant::fp8_e4m3:
        case config::AttentionQuant::fp8_e5m2:
            return 1.0;
        case config::AttentionQuant::fp16:
            return 2.0;
    }
    return 1.0;  // unreachable
}

double bytes_per_element(config::GatingQuant q) {
    switch (q) {
        case config::GatingQuant::fp32:
            return 4.0;
        case config::GatingQuant::fp16:
            return 2.0;
    }
    return 4.0;  // unreachable
}

double bytes_per_element(config::WeightQuant q) {
    switch (q) {
        case config::WeightQuant::nvfp4:
            return 0.5625;  // FP4 E2M1 + UE8M0 scales (no per-projection scalars)
        case config::WeightQuant::fp8_e4m3:
        case config::WeightQuant::fp8_e5m2:
            return 1.0;
        case config::WeightQuant::int4_symmetric:
        case config::WeightQuant::int4_asymmetric:
            // TD-61m: real INT4 = 4-bit weights + FP16 scales at
            // group_size=128 → 0.5 + 2/128 = 0.515625 bytes/elem (NOT the
            // NVFP4 0.5625, whose FP8 scales sit at group_size=16).
            return 0.515625;
        // GGUF k-quants: fractional bytes/element from the single-source block
        // table (gguf::bytes_per_element). Exact only when in_features % QK == 0;
        // byte-exact attention/dense GGUF sizing goes through gguf_packed_bytes
        // in attention_layer_bytes (TD-GGUF-ATTN-UPLOAD-SIZING). Q2_K=0.328125,
        // Q3_K=0.4296875, Q4_K=0.5625, Q5_K=0.6875, Q6_K=0.8203125, Q8_0=1.0625.
        case config::WeightQuant::gguf_q2_k:
            return gguf::bytes_per_element(GgufKQuantType::Q2_K);
        case config::WeightQuant::gguf_q3_k:
            return gguf::bytes_per_element(GgufKQuantType::Q3_K);
        case config::WeightQuant::gguf_q4_k:
            return gguf::bytes_per_element(GgufKQuantType::Q4_K);
        case config::WeightQuant::gguf_q5_k:
            return gguf::bytes_per_element(GgufKQuantType::Q5_K);
        case config::WeightQuant::gguf_q6_k:
            return gguf::bytes_per_element(GgufKQuantType::Q6_K);
        case config::WeightQuant::gguf_q8_0:
            return gguf::bytes_per_element(GgufKQuantType::Q8_0);
        case config::WeightQuant::gguf:
            // Generic GGUF is per-tensor mixed — there is no single bytes/element.
            // Callers needing GGUF sizing must use gguf_packed_bytes per tensor.
            throw std::runtime_error(
                "bytes_per_element(WeightQuant::gguf): the generic 'gguf' enum "
                "is per-tensor mixed and has no scalar bytes/element; size each "
                "tensor via gguf::gguf_packed_bytes(out,in,type)");
    }
    return 1.0;  // unreachable
}

// ── LayerInfo ───────────────────────────────────────────────────────────────

int64_t LayerInfo::pinned_bytes() const {
    int64_t total = 0;
    if (attention_pinned) total += attention_bytes;
    if (ffn_pinned) total += ffn_bytes;
    if (gating_pinned) total += gating_bytes;
    if (shared_expert_pinned) total += shared_expert_bytes;
    return total;
}

// ── Pinning helper ──────────────────────────────────────────────────────────

static bool is_pinned(const config::LayerPinSpec& spec, int layer_idx) {
    if (auto* s = std::get_if<std::string>(&spec))
        return *s == "all";
    auto& v = std::get<std::vector<int>>(spec);
    return std::find(v.begin(), v.end(), layer_idx) != v.end();
}

// ── LayerRegistry constructor ───────────────────────────────────────────────

LayerRegistry::LayerRegistry(const ModelConfig& model_cfg,
                             const config::Config& cfg,
                             const QuantInterface& expert_quant)
    : cfg_(&cfg) {
    const auto& m = model_cfg.raw();
    const auto& q = cfg.quantization;
    const auto& pin = cfg.memory.pinned_layers;

    double attn_bpe = bytes_per_element(q.attention_compute);
    double gate_bpe = bytes_per_element(q.gating_compute);

    // ── Attention weight params per layer (MLA projections + norms) ──
    int64_t q_a = static_cast<int64_t>(m.hidden_size) * m.q_lora_rank;
    int64_t q_b = static_cast<int64_t>(m.q_lora_rank) *
                  (m.num_attention_heads * (m.qk_nope_head_dim + m.qk_rope_head_dim));
    int64_t kv_a = static_cast<int64_t>(m.hidden_size) *
                   (m.kv_lora_rank + m.qk_rope_head_dim);
    int64_t kv_b = static_cast<int64_t>(m.kv_lora_rank) *
                   (m.num_attention_heads * (m.qk_nope_head_dim + m.v_head_dim));
    int64_t o = static_cast<int64_t>(m.num_attention_heads) * m.v_head_dim * m.hidden_size;
    int64_t norms = m.q_lora_rank + m.kv_lora_rank +
                    2 * static_cast<int64_t>(m.hidden_size);
    int64_t attn_params = q_a + q_b + kv_a + kv_b + o + norms;
    int64_t attention_bytes_per_layer = static_cast<int64_t>(attn_params * attn_bpe);

    // ── DSA indexer projection weights per layer (only for DSA models) ──
    // V4 guard: has_dsa() is FALSE for V4 even though index_topk > 0 (the V4
    // Lightning Indexer is part of the CSA anatomy, sized inside
    // v4_attention_layer_bytes) — the MLA indexer formula below must not run.
    int64_t indexer_bytes_per_layer = 0;
    if (model_cfg.has_dsa()) {
        int64_t q_idx_b = static_cast<int64_t>(m.q_lora_rank) *
                          m.index_n_heads * m.index_head_dim;
        int64_t k_idx = static_cast<int64_t>(m.hidden_size) * m.index_head_dim;
        int64_t k_idx_norm = m.index_head_dim;
        int64_t k_idx_norm_bias = m.index_head_dim;
        int64_t weights_proj = static_cast<int64_t>(m.hidden_size) * m.index_n_heads;
        int64_t indexer_params = q_idx_b + k_idx + k_idx_norm + k_idx_norm_bias + weights_proj;
        indexer_bytes_per_layer = static_cast<int64_t>(indexer_params * attn_bpe);
    }

    // ── Gating params per MoE layer ──
    int64_t gating_params = static_cast<int64_t>(m.hidden_size) * m.n_routed_experts;
    int64_t gating_bytes_per_layer = static_cast<int64_t>(gating_params * gate_bpe);

    // ── Expert sizes via QuantInterface ──
    ExpertShape moe_shape{m.hidden_size, m.moe_intermediate_size};
    int64_t shared_bytes_per_layer =
        expert_quant.bytes_per_expert(moe_shape) * m.n_shared_experts;
    per_routed_expert_bytes_ = expert_quant.bytes_per_expert(moe_shape);
    total_routed_experts_ = m.n_routed_experts;

    ExpertShape dense_shape{m.hidden_size, m.intermediate_size};
    int64_t dense_ffn_bytes_per_layer = expert_quant.bytes_per_expert(dense_shape);

    // ── Embedding & output head ──
    embedding_bytes_ = static_cast<int64_t>(
        static_cast<int64_t>(m.vocab_size) * m.hidden_size * kEmbeddingBytesPerElement);
    {
        int64_t oh_params = static_cast<int64_t>(m.vocab_size) * m.hidden_size;
        if (has_output_head_bias(cfg)) oh_params += m.hidden_size;
        output_head_bytes_ = oh_params * kEmbeddingBytesPerElement;
    }
    embedding_pinned_ = pin.embedding;
    output_head_pinned_ = pin.output_head;

    // ── Build per-layer info ──
    num_moe_layers_ = model_cfg.num_moe_layers();
    num_dense_layers_ = model_cfg.num_dense_layers();

    layers_.reserve(m.num_hidden_layers);
    for (int l = 0; l < m.num_hidden_layers; ++l) {
        LayerInfo info{};
        info.layer_idx = l;
        info.is_moe = model_cfg.is_moe_layer(l);
        // V4 (V4-3a): per-layer anatomy from compress_ratios — SWA/CSA/HCA
        // sizes differ (compressor, indexer, mHC included; BF16/F32-native,
        // unsharded here: LayerInfo carries full-layer sizes, tp=1).
        info.attention_bytes = model_cfg.is_v4()
            ? v4_attention_layer_bytes(
                  m, model_cfg.attention_type_for_layer(l), /*tp=*/1)
            : attention_bytes_per_layer + indexer_bytes_per_layer;
        info.attention_pinned = is_pinned(pin.attention, l);

        if (info.is_moe) {
            info.ffn_bytes = 0;
            info.gating_bytes = gating_bytes_per_layer;
            // V4 hash layers (l < num_hash_layers) carry the I32
            // token→expert table instead of the exp_probs_b bias.
            if (model_cfg.is_v4() && model_cfg.is_hash_layer(l))
                info.gating_bytes += v4_hash_gating_table_bytes(m);
            info.shared_expert_bytes = shared_bytes_per_layer;
            info.per_routed_expert_bytes = per_routed_expert_bytes_;
            info.ffn_pinned = false;
            info.gating_pinned = is_pinned(pin.gating, l);
            info.shared_expert_pinned = true;  // always pinned per spec S4.1
        } else {
            info.ffn_bytes = dense_ffn_bytes_per_layer;
            info.gating_bytes = 0;
            info.shared_expert_bytes = 0;
            info.per_routed_expert_bytes = 0;
            info.ffn_pinned =
                std::find(pin.dense_ffn_layers.begin(),
                          pin.dense_ffn_layers.end(), l) != pin.dense_ffn_layers.end();
            info.gating_pinned = false;
            info.shared_expert_pinned = false;
        }

        layers_.push_back(info);
    }

    // ── Total pinned bytes (legacy, used for non-TP fallback) ──
    // TODO:DEBT TD-53w: total_pinned_bytes_ is stale dead code (superseded by pinned_layout_)
    total_pinned_bytes_ = 0;
    for (const auto& li : layers_) {
        total_pinned_bytes_ += li.pinned_bytes();
    }
    if (embedding_pinned_) total_pinned_bytes_ += embedding_bytes_;
    if (output_head_pinned_) total_pinned_bytes_ += output_head_bytes_;

    // ── Authoritative pinned layout (replaces old per-component budget math) ──
    int tp = std::max(1, cfg.parallelism.tensor_parallelism);
    pinned_layout_ = compute_pinned_layout(model_cfg, cfg, expert_quant, tp, 0);

    // ── DSpark draft weights + runtime scratch (DSP-2 / DSP-3) ──
    // When speculation.method=dspark the whole draft is pinned on ONE
    // draft GPU (draft_gpus[0] or the first non-TP GPU — see dspark_loader.h),
    // plus the DsparkRuntime device scratch (aux staging, context-KV arena,
    // query buffers — DSP-3). Weights sized EXACTLY from the checkpoint's
    // safetensors header AT the configured upload quantization
    // (draft_weights_quant, TD-DSPARK-DRAFT-QUANT — quantized GEMM operands
    // + scale bytes when != bf16); scratch from config + checkpoint dims
    // (same layout function create() allocates from). Fails closed when the
    // checkpoint is unreadable (the engine cannot run dspark without it
    // anyway).
    if (cfg.speculation.method == config::SpeculationMethodType::dspark) {
        // TD-DSPARK-DRAFT-SHARD: one charge PER DRAFT RANK — the rank's
        // weight SHARD (header-derived, at the configured quant) + its
        // per-rank runtime scratch + the align slack. Single-rank keeps the
        // legacy whole-draft charge on one GPU.
        dspark_rank_gpus_ = resolve_dspark_draft_gpus(cfg);
        const int nr = static_cast<int>(dspark_rank_gpus_.size());
        const auto dck = parse_dspark_checkpoint_config(
            cfg.speculation.dspark.checkpoint_path);
        for (int r = 0; r < nr; ++r) {
            // Qualified: the free function (dspark_loader.h), not the
            // accessor.
            int64_t charge = layerstorm::model::dspark_draft_bytes(
                cfg.speculation.dspark.checkpoint_path,
                cfg.speculation.dspark.draft_weights_quant, r, nr);
            charge +=
                speculation::dspark_runtime_scratch_bytes(cfg, dck, r, nr);
            // The runtime carves weights + scratch out of the pinned region
            // with 256-byte alignment per item — cover the padding.
            charge += speculation::kDsparkArenaAlignSlack;
            dspark_rank_charges_.push_back(charge);
        }
    }
}

// ── estimate_gpu_budgets ────────────────────────────────────────────────────
//
// Delegates to PinnedRegionLayout for the authoritative per-GPU pinned bytes.
// Non-TP GPUs hold only routed experts (in expert cache), zero pinned bytes.

std::vector<LayerRegistry::GpuVramBudget> LayerRegistry::estimate_gpu_budgets() const {
    const auto& hw = cfg_->hardware;
    if (hw.gpus.empty()) return {};

    // Build TP membership set
    std::vector<bool> in_tp(hw.gpus.size(), false);
    for (int idx : hw.tp_array) {
        if (idx >= 0 && idx < static_cast<int>(hw.gpus.size()))
            in_tp[idx] = true;
    }
    int tp_count = static_cast<int>(hw.tp_array.size());

    std::vector<GpuVramBudget> budgets;
    budgets.reserve(hw.gpus.size());

    if (tp_count > 0) {
        for (size_t i = 0; i < hw.gpus.size(); ++i) {
            GpuVramBudget b{};
            b.gpu_id = hw.gpus[i].id;
            b.total_vram_bytes = config::vram_gb_to_bytes(hw.gpus[i].vram_gb);
            b.pinned_bytes = in_tp[i] ? pinned_layout_.total_bytes : 0;
            b.available_for_cache_bytes = b.total_vram_bytes - b.pinned_bytes;
            budgets.push_back(b);
        }
    } else {
        // TODO:DEBT TD-53j: No-TP fallback distributes TP-sharded layout size (wrong total)
        // No TP: distribute pinned bytes across all GPUs weighted by VRAM
        double total_vram_gb = 0.0;
        for (const auto& g : hw.gpus) total_vram_gb += g.vram_gb;

        for (size_t i = 0; i < hw.gpus.size(); ++i) {
            GpuVramBudget b{};
            b.gpu_id = hw.gpus[i].id;
            b.total_vram_bytes = config::vram_gb_to_bytes(hw.gpus[i].vram_gb);
            b.pinned_bytes = (total_vram_gb > 0.0)
                ? static_cast<int64_t>(
                      static_cast<double>(pinned_layout_.total_bytes) *
                      (hw.gpus[i].vram_gb / total_vram_gb))
                : 0;
            b.available_for_cache_bytes = b.total_vram_bytes - b.pinned_bytes;
            budgets.push_back(b);
        }
    }

    // ── DSpark draft weights (DSP-2): pinned per draft rank GPU
    //    (TD-DSPARK-DRAFT-SHARD: one shard charge per rank) ──
    for (size_t r = 0; r < dspark_rank_gpus_.size(); ++r) {
        const int gpu = dspark_rank_gpus_[r];
        const int64_t bytes = dspark_rank_charges_[r];
        if (bytes > 0 && gpu >= 0 && gpu < static_cast<int>(budgets.size())) {
            auto& b = budgets[static_cast<size_t>(gpu)];
            b.pinned_bytes += bytes;
            b.available_for_cache_bytes -= bytes;
        }
    }

    return budgets;
}

}  // namespace layerstorm::model

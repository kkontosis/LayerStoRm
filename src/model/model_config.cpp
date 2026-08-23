#include "model/model_config.h"

#include <algorithm>

namespace layerstorm::model {

ModelConfig::ModelConfig(const config::Config& cfg) : cfg_(cfg.model) {
    compute_layer_counts();
}

ModelConfig::ModelConfig(const config::ModelConfig& model_cfg) : cfg_(model_cfg) {
    compute_layer_counts();
}

bool ModelConfig::is_moe_layer(int layer_idx) const {
    if (layer_idx < cfg_.first_k_dense_replace) return false;
    if (layer_idx >= cfg_.num_hidden_layers) return false;
    if (cfg_.moe_layer_freq <= 0) return false;
    return (layer_idx - cfg_.first_k_dense_replace) % cfg_.moe_layer_freq == 0;
}

bool ModelConfig::has_dsa() const {
    // V4's index_topk > 0 is the Lightning Indexer (part of the CSA pipeline,
    // dispatched inside the attention device) — NOT the standalone DSA indexer.
    if (is_v4()) return false;
    return cfg_.index_topk > 0;
}

bool ModelConfig::is_v4() const {
    return cfg_.architecture == config::Architecture::deepseek_v4;
}

bool ModelConfig::uses_mla() const {
    return !is_v4();
}

bool ModelConfig::has_csa_hca() const {
    return is_v4();
}

V4AttentionType ModelConfig::attention_type_for_layer(int layer_idx) const {
    if (!is_v4()) return V4AttentionType::kSwa;
    if (layer_idx < 0 ||
        layer_idx >= static_cast<int>(cfg_.compress_ratios.size()))
        return V4AttentionType::kSwa;
    switch (cfg_.compress_ratios[static_cast<size_t>(layer_idx)]) {
        case 4:   return V4AttentionType::kCsa;
        case 128: return V4AttentionType::kHca;
        default:  return V4AttentionType::kSwa;
    }
}

bool ModelConfig::layer_uses_compress_rope(int layer_idx) const {
    return is_v4() &&
           attention_type_for_layer(layer_idx) != V4AttentionType::kSwa;
}

bool ModelConfig::has_mhc() const {
    return cfg_.hc_mult > 1;
}

bool ModelConfig::has_grouped_o_proj() const {
    return cfg_.o_groups > 1;
}

bool ModelConfig::is_hash_layer(int layer_idx) const {
    return layer_idx >= 0 && layer_idx < cfg_.num_hash_layers;
}

bool ModelConfig::is_full_index_layer(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= cfg_.num_hidden_layers) return false;
    // No sharing configured (GGUF default / llama.cpp reference): every layer
    // recomputes the indexer.
    if (cfg_.index_topk_freq <= 0) return true;
    if (layer_idx < cfg_.index_skip_topk_offset) return true;  // leading-full
    return (layer_idx - cfg_.index_skip_topk_offset + 1) % cfg_.index_topk_freq == 0;
}

bool ModelConfig::has_mtp() const {
    return cfg_.num_nextn_predict_layers > 0;
}

bool ModelConfig::has_grouped_routing() const {
    return cfg_.n_group > 1;
}

bool ModelConfig::has_vision() const {
    return cfg_.vision.has_value() && cfg_.vision->enabled;
}

int ModelConfig::qk_head_dim() const {
    return cfg_.qk_nope_head_dim + cfg_.qk_rope_head_dim;
}

int ModelConfig::kv_cache_dim() const {
    return cfg_.kv_lora_rank + cfg_.qk_rope_head_dim;
}

void ModelConfig::compute_layer_counts() {
    moe_layer_indices_.clear();
    dense_layer_indices_.clear();
    full_index_layer_mask_.assign(std::max(cfg_.num_hidden_layers, 0), false);
    num_full_index_layers_ = 0;
    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        if (is_moe_layer(l)) {
            moe_layer_indices_.push_back(l);
        } else {
            dense_layer_indices_.push_back(l);
        }
        const bool full = is_full_index_layer(l);
        full_index_layer_mask_[l] = full;
        if (full) ++num_full_index_layers_;
    }
    num_moe_layers_ = static_cast<int>(moe_layer_indices_.size());
    num_dense_layers_ = static_cast<int>(dense_layer_indices_.size());
}

}  // namespace layerstorm::model

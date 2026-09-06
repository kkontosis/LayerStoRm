#include "model/model_config.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <stdexcept>

namespace layerstorm::model {

namespace {
// P-29 step 13 phase B: config-armed MTP expert census (see arm_mtp_experts).
// consulted latches on the FIRST census query so a late arming that would
// flip an already-computed layer table throws instead of splitting the
// process into two censuses.
std::atomic<bool> g_mtp_experts_armed{false};
std::atomic<bool> g_mtp_census_consulted{false};
}  // namespace

ModelConfig::ModelConfig(const config::Config& cfg) : cfg_(cfg.model) {
    // P-29 step 13 phase B: arm the MTP expert census from config before the
    // first layer count is computed. speculation.method == mtp +
    // mtp.enabled on glm5_next means the MTP layer's routed experts are
    // arena tenants for the serving draft path.
    if (cfg.speculation.enabled &&
        cfg.speculation.method == config::SpeculationMethodType::mtp &&
        cfg.speculation.mtp.enabled &&
        cfg.model.architecture == config::Architecture::glm5_next &&
        cfg.model.num_nextn_predict_layers > 0) {
        arm_mtp_experts();
    }
    compute_layer_counts();
}

ModelConfig::ModelConfig(const config::ModelConfig& model_cfg) : cfg_(model_cfg) {
    compute_layer_counts();
}

bool ModelConfig::is_moe_layer(int layer_idx) const {
    if (layer_idx < cfg_.first_k_dense_replace) return false;
    if (layer_idx >= cfg_.num_hidden_layers) {
        // P-29 step 11 / OQ-3 phase A (LS_MTP_PROBE=1): glm5_next MTP/NextN
        // block(s) are full MoE layers (MODELINFO §5: 288 routed + shared +
        // gate). Counting them here makes their routed experts arena
        // tenants (prepack, ELM window, FETCH_AND_RUN) with NO other code
        // knowing about MTP. Default OFF: bound identical to the historical
        // `return false`, so champion layer counts and the 12,096-slot
        // arena identity are untouched.
        return mtp_probe_experts_enabled()
            && cfg_.architecture == config::Architecture::glm5_next
            && layer_idx < cfg_.num_hidden_layers + cfg_.num_nextn_predict_layers
            && cfg_.moe_layer_freq > 0;
    }
    if (cfg_.moe_layer_freq <= 0) return false;
    return (layer_idx - cfg_.first_k_dense_replace) % cfg_.moe_layer_freq == 0;
}

bool ModelConfig::mtp_probe_experts_enabled() {
    static const bool env_on = [] {
        const char* e = std::getenv("LS_MTP_PROBE");
        return e && *e == '1';
    }();
    g_mtp_census_consulted.store(true, std::memory_order_release);
    return env_on || g_mtp_experts_armed.load(std::memory_order_acquire);
}

void ModelConfig::arm_mtp_experts() {
    if (g_mtp_experts_armed.load(std::memory_order_acquire)) return;
    // Arming after a census was consulted (and read false) would split the
    // process into two censuses — refuse loudly. The env latch (already
    // true) is exempt: the effective value does not change.
    const char* e = std::getenv("LS_MTP_PROBE");
    const bool env_on = e && *e == '1';
    if (g_mtp_census_consulted.load(std::memory_order_acquire) && !env_on) {
        throw std::logic_error(
            "ModelConfig::arm_mtp_experts: MTP expert census armed AFTER a "
            "layer census was already consulted — construct the "
            "config::Config-based ModelConfig (or set LS_MTP_PROBE=1) "
            "before any census-dependent table is built");
    }
    g_mtp_experts_armed.store(true, std::memory_order_release);
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

bool ModelConfig::is_glm5_next() const {
    return cfg_.architecture == config::Architecture::glm5_next;
}

bool ModelConfig::is_linear_attention_layer(int layer_idx) const {
    if (!is_glm5_next()) return false;
    if (layer_idx < 0 ||
        layer_idx >= static_cast<int>(cfg_.layer_types.size()))
        return false;
    return cfg_.layer_types[static_cast<size_t>(layer_idx)] ==
           config::LayerAttentionType::linear_attention;
}

bool ModelConfig::has_index_pool() const {
    return is_glm5_next() && cfg_.index_kpool > 1;
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
    // glm5_next (GF3.2): IndexShare does not exist (indexer_types uniformly
    // "full"), and only the sparse-MLA layers CARRY an indexer — KDA linear
    // layers have no indexer at all, so the computing-layer set is exactly
    // the deepseek_sparse_attention layers (GLM-5.3-Flash: 11; MODELINFO
    // §3c). The MTP layer's indexer is outside this per-hidden-layer mask.
    if (is_glm5_next()) return !is_linear_attention_layer(layer_idx);
    // No sharing configured (GGUF default / llama.cpp reference): every layer
    // recomputes the indexer.
    if (cfg_.index_topk_freq <= 0) return true;
    if (layer_idx < cfg_.index_skip_topk_offset) return true;  // leading-full
    return (layer_idx - cfg_.index_skip_topk_offset + 1) % cfg_.index_topk_freq == 0;
}

bool ModelConfig::computes_indexer(int layer_idx) const {
    if (!has_dsa()) return false;
    const int nh = cfg_.num_hidden_layers;
    if (layer_idx < 0 ||
        layer_idx >= nh + cfg_.num_nextn_predict_layers) return false;
    if (layer_idx >= nh) {
        // MTP layer(s). glm5_next: sparse MLA with its OWN indexer tensors
        // (MODELINFO §3c: "the 11 sparse layers + the MTP layer") — MTP
        // iteration 0 computes its top-k, iterations 1+ reuse it
        // (index_share_for_mtp_iteration, GF3.11). Legacy IndexShare MTP is
        // shared by construction and never computes/stores.
        return is_glm5_next();
    }
    if (is_glm5_next()) {
        // No IndexShare and no "∪ layer 0" union: layer 0 is a KDA linear
        // layer with no indexer at all. Computing set == sparse layers.
        return !is_linear_attention_layer(layer_idx);
    }
    return is_full_index_layer(layer_idx) || layer_idx == 0;
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
    // P-29 step 11 (LS_MTP_PROBE): append glm5_next MTP/NextN MoE layer(s) so the
    // MoE-layer set stays contiguous (3..NH+nextn-1) and every consumer that
    // derives from moe_layer_indices()/num_moe_layers() — arena sizing,
    // prepacker, ELM ordinal window — funds them consistently. No-op with
    // the flag off (is_moe_layer(>=NH) is false).
    for (int l = cfg_.num_hidden_layers;
         l < cfg_.num_hidden_layers + cfg_.num_nextn_predict_layers; ++l)
        if (is_moe_layer(l))
            moe_layer_indices_.push_back(l);
    num_moe_layers_ = static_cast<int>(moe_layer_indices_.size());
    num_dense_layers_ = static_cast<int>(dense_layer_indices_.size());
    num_linear_attention_layers_ = 0;
    linear_layer_ordinal_.assign(std::max(cfg_.num_hidden_layers, 0), -1);
    for (int l = 0; l < cfg_.num_hidden_layers; ++l)
        if (is_linear_attention_layer(l))
            linear_layer_ordinal_[l] = num_linear_attention_layers_++;
    // KV-bearing layers: glm5_next linear layers keep no KV cache (per-
    // request recurrent state instead); everywhere else every hidden layer
    // appends KV.
    num_kv_layers_ = cfg_.num_hidden_layers - num_linear_attention_layers_;
}

}  // namespace layerstorm::model

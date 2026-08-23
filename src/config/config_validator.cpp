#include "config_validator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <string>

#include "model/layer_registry.h"
#include "model/model_config.h"
#include "model/quantization/registry.h"

namespace layerstorm::config {

namespace {

// ── Helpers ──────────────────────────────────────────────────────────────────

void error(ValidationResult& r, std::string field, std::string msg) {
    r.errors.push_back({std::move(field), std::move(msg)});
}

void warn(ValidationResult& r, std::string field, std::string msg) {
    r.warnings.push_back({std::move(field), std::move(msg)});
}

/// Resolve pinned layer indices from a LayerPinSpec.
/// Returns all layer indices [0, num_layers) for "all", or the specified list.
std::vector<int> resolve_pin_spec(const LayerPinSpec& spec, int num_layers) {
    if (auto* s = std::get_if<std::string>(&spec)) {
        if (*s == "all") {
            std::vector<int> all(static_cast<size_t>(num_layers));
            std::iota(all.begin(), all.end(), 0);
            return all;
        }
    }
    if (auto* v = std::get_if<std::vector<int>>(&spec)) {
        return *v;
    }
    return {};
}

// ── Validation categories ────────────────────────────────────────────────────

void validate_gpu_config(const Config& cfg, ValidationResult& r) {
    const auto& gpus = cfg.hardware.gpus;

    if (gpus.empty()) {
        error(r, "hardware.gpus", "At least one GPU must be configured");
        return;
    }

    std::set<int> ids;
    for (size_t i = 0; i < gpus.size(); ++i) {
        const auto& g = gpus[i];
        std::string prefix = "hardware.gpus[" + std::to_string(i) + "]";

        if (!ids.insert(g.id).second) {
            error(r, prefix + ".id",
                  "Duplicate GPU id " + std::to_string(g.id));
        }
        if (g.vram_gb < 0.0) {
            error(r, prefix + ".vram_gb", "VRAM must be non-negative");
        }
        if (g.pcie_gen != 0 && (g.pcie_gen < 3 || g.pcie_gen > 6)) {
            warn(r, prefix + ".pcie_gen",
                 "Unusual PCIe generation " + std::to_string(g.pcie_gen));
        }
        if (g.vram_gb > 0.0) {
            double v = g.vram_gb;
            if (g.type == GpuType::rtx5090 && (v < 28.0 || v > 36.0)) {
                warn(r, prefix + ".vram_gb",
                     "RTX 5090 typically has ~31–32 GiB VRAM, got " +
                     std::to_string(v) + " GiB");
            }
            if (g.type == GpuType::rtx5080 && (v < 12.0 || v > 20.0)) {
                warn(r, prefix + ".vram_gb",
                     "RTX 5080 typically has ~15–16 GiB VRAM, got " +
                     std::to_string(v) + " GiB");
            }
        }
        if (g.roles.empty()) {
            warn(r, prefix + ".roles", "GPU has no assigned roles");
        }
    }

    if (cfg.hardware.system_ram_gb <= 0) {
        error(r, "hardware.system_ram_gb", "System RAM must be positive");
    }
}

void validate_tp_array(const Config& cfg, ValidationResult& r) {
    const auto& tp = cfg.hardware.tp_array;
    if (tp.empty()) return;

    const auto& gpus = cfg.hardware.gpus;

    auto find_gpu = [&](int id) -> const GpuConfig* {
        for (const auto& g : gpus) {
            if (g.id == id) return &g;
        }
        return nullptr;
    };

    // Length must be power of 2
    int len = static_cast<int>(tp.size());
    if (len & (len - 1)) {
        error(r, "hardware.tp_array",
              "TP array length must be a power of 2, got " + std::to_string(len));
    }

    // All IDs must be distinct and reference existing GPUs
    std::set<int> seen;
    for (size_t i = 0; i < tp.size(); ++i) {
        std::string idx = "hardware.tp_array[" + std::to_string(i) + "]";
        if (!seen.insert(tp[i]).second) {
            error(r, idx,
                  "Duplicate GPU id " + std::to_string(tp[i]) + " in tp_array");
        }
        const auto* gpu = find_gpu(tp[i]);
        if (!gpu) {
            error(r, idx,
                  "GPU id " + std::to_string(tp[i]) + " not found in hardware.gpus");
        }
        // INV-0.5: TP requires same-type GPUs (currently 5090 only)
        if (gpu && gpu->type != GpuType::rtx5090) {
            error(r, idx,
                  "TP requires RTX 5090 GPUs; GPU " + std::to_string(tp[i]) +
                  " is not an RTX 5090");
        }
    }
}

void validate_tp_mode_requires_tp_array(const Config& cfg, ValidationResult& r) {
    const auto& tp = cfg.memory.tp_mode_per_layer;
    bool needs_tp = (tp.gating > 1 || tp.pinned_dense_ffn > 1 || tp.default_mode > 1);

    if (needs_tp && cfg.hardware.tp_array.empty()) {
        error(r, "memory.tp_mode_per_layer",
              "TP mode > 1 configured but no hardware.tp_array specified");
    }
}

void validate_nvme(const Config& cfg, ValidationResult& r) {
    if (cfg.memory.nvme_tier.enabled) {
        if (cfg.hardware.nvme_paths.empty()) {
            error(r, "hardware.nvme_paths",
                  "NVMe tier is enabled but hardware.nvme_paths is empty");
        }
        if (!cfg.hardware.nvme_capacity_gb) {
            warn(r, "hardware.nvme_capacity_gb",
                 "NVMe tier is enabled but nvme_capacity_gb is not set");
        }
    } else if (!cfg.hardware.nvme_paths.empty()) {
        warn(r, "memory.nvme_tier.enabled",
             "hardware.nvme_paths is set but NVMe tier is disabled");
    }
}

void validate_model_dimensions(const Config& cfg, ValidationResult& r) {
    const auto& m = cfg.model;

    if (m.num_hidden_layers <= 0)
        error(r, "model.num_hidden_layers", "Must be positive");
    if (m.hidden_size <= 0)
        error(r, "model.hidden_size", "Must be positive");
    if (m.num_attention_heads <= 0)
        error(r, "model.num_attention_heads", "Must be positive");
    if (m.num_key_value_heads <= 0)
        error(r, "model.num_key_value_heads", "Must be positive");
    if (m.vocab_size < 0)
        error(r, "model.vocab_size",
              "Must be >= 0 (0 = autodetect from the weights at load)");
    if (m.intermediate_size <= 0)
        error(r, "model.intermediate_size", "Must be positive");
    if (m.max_position_embeddings <= 0)
        error(r, "model.max_position_embeddings", "Must be positive");
    if (m.n_routed_experts <= 0)
        error(r, "model.n_routed_experts", "Must be positive");
    if (m.num_experts_per_tok <= 0)
        error(r, "model.num_experts_per_tok", "Must be positive");

    if (m.num_experts_per_tok > m.n_routed_experts) {
        error(r, "model.num_experts_per_tok",
              "Cannot exceed n_routed_experts (" + std::to_string(m.n_routed_experts) + ")");
    }
    if (m.first_k_dense_replace < 0) {
        error(r, "model.first_k_dense_replace", "Cannot be negative");
    }
    if (m.num_hidden_layers > 0 && m.first_k_dense_replace > m.num_hidden_layers) {
        error(r, "model.first_k_dense_replace",
              "Cannot exceed num_hidden_layers (" + std::to_string(m.num_hidden_layers) + ")");
    }
    if (m.n_group <= 0)
        error(r, "model.n_group", "Must be positive");
    if (m.topk_group <= 0)
        error(r, "model.topk_group", "Must be positive");
    if (m.topk_group > m.n_group) {
        error(r, "model.topk_group",
              "Cannot exceed n_group (" + std::to_string(m.n_group) + ")");
    }
    if (m.n_routed_experts > 0 && m.n_group > 0 && m.n_routed_experts % m.n_group != 0) {
        error(r, "model.n_group",
              "n_routed_experts (" + std::to_string(m.n_routed_experts) +
              ") must be divisible by n_group (" + std::to_string(m.n_group) + ")");
    }
    if (m.moe_layer_freq <= 0) {
        error(r, "model.moe_layer_freq", "Must be positive");
    }
}

void validate_pinned_layers(const Config& cfg, ValidationResult& r) {
    const int n = cfg.model.num_hidden_layers;
    if (n <= 0) return;  // already reported

    auto check_indices = [&](const std::vector<int>& indices, const std::string& field) {
        for (int idx : indices) {
            if (idx < 0 || idx >= n) {
                error(r, field,
                      "Layer index " + std::to_string(idx) +
                      " out of range [0, " + std::to_string(n) + ")");
            }
        }
    };

    auto attn_layers = resolve_pin_spec(cfg.memory.pinned_layers.attention, n);
    check_indices(attn_layers, "memory.pinned_layers.attention");

    check_indices(cfg.memory.pinned_layers.dense_ffn_layers,
                  "memory.pinned_layers.dense_ffn_layers");

    auto gating_layers = resolve_pin_spec(cfg.memory.pinned_layers.gating, n);
    check_indices(gating_layers, "memory.pinned_layers.gating");
}

void validate_eviction_weights(const Config& cfg, ValidationResult& r) {
    const auto& ec = cfg.memory.expert_cache;

    auto check_weight = [&](double w, const char* name) {
        if (w < 0.0) {
            error(r, std::string("memory.expert_cache.") + name,
                  "Eviction weight must be non-negative");
        }
    };
    check_weight(ec.eviction_alpha_recency, "eviction_alpha_recency");
    check_weight(ec.eviction_beta_frequency, "eviction_beta_frequency");
    check_weight(ec.eviction_gamma_routing_weight, "eviction_gamma_routing_weight");
    check_weight(ec.eviction_delta_temporal_autocorr, "eviction_delta_temporal_autocorr");
    check_weight(ec.eviction_epsilon_coactivation, "eviction_epsilon_coactivation");
    check_weight(ec.eviction_zeta_prefetch_score, "eviction_zeta_prefetch_score");

    double sum = ec.eviction_alpha_recency + ec.eviction_beta_frequency +
                 ec.eviction_gamma_routing_weight + ec.eviction_delta_temporal_autocorr +
                 ec.eviction_epsilon_coactivation + ec.eviction_zeta_prefetch_score;
    if (sum <= 0.0) {
        error(r, "memory.expert_cache.eviction_*",
              "Sum of eviction weights must be positive");
    }
}

void validate_performance_objective(const Config& cfg, ValidationResult& r) {
    const auto& po = cfg.orchestrator.performance_objective;
    if (po.latency_weight < 0.0)
        error(r, "orchestrator.performance_objective.latency_weight", "Must be non-negative");
    if (po.throughput_weight < 0.0)
        error(r, "orchestrator.performance_objective.throughput_weight", "Must be non-negative");
    if (po.latency_weight + po.throughput_weight <= 0.0)
        error(r, "orchestrator.performance_objective",
              "latency_weight + throughput_weight must be positive");
}

void validate_vram_budget(const Config& cfg, ValidationResult& r) {
    const auto& m = cfg.model;
    const auto& hw = cfg.hardware;

    if (hw.gpus.empty() || m.hidden_size <= 0 || m.num_hidden_layers <= 0 ||
        m.intermediate_size <= 0 || m.n_routed_experts <= 0 ||
        m.moe_layer_freq <= 0)
        return;

    for (const auto& g : hw.gpus)
        if (g.vram_gb <= 0.0) return;

    // Generic `gguf` weights cannot be sized without the file's per-projection
    // k-quant header scan (the registry sentinel THROWS on expert sizing —
    // TD-GGUF-GENERIC-DEFAULT-MISSIZE). The engine builds the concrete
    // interface at init (engine.cpp Step 4); the pure-config validator skips
    // the budget estimate for this combination instead of throwing.
    if (cfg.model.weights_format == WeightsFormat::gguf &&
        cfg.quantization.weights == WeightQuant::gguf)
        return;

    const auto* quant_fmt = model::find_format(cfg.quantization.weights);
    if (!quant_fmt) return;

    model::LayerRegistry registry = [&]() {
        model::ModelConfig model_cfg(cfg);
        return model::LayerRegistry(model_cfg, cfg, *quant_fmt);
    }();

    auto budgets = registry.estimate_gpu_budgets();
    int64_t expert_bytes = registry.per_routed_expert_bytes();
    int64_t min_expert_cache = static_cast<int64_t>(m.num_experts_per_tok) * expert_bytes;
    constexpr int64_t kSafetyMargin = 512LL * 1024 * 1024;

    for (const auto& b : budgets) {
        int64_t required = b.pinned_bytes + min_expert_cache + kSafetyMargin;
        if (required > b.total_vram_bytes) {
            error(r, "hardware.gpus",
                  "Insufficient VRAM on GPU " + std::to_string(b.gpu_id) +
                  ": pinned layers ~" +
                  std::to_string(b.pinned_bytes / (1024LL * 1024 * 1024)) +
                  " GB, minimum expert cache ~" +
                  std::to_string(min_expert_cache / (1024LL * 1024 * 1024)) +
                  " GB, safety margin ~0.5 GB, but GPU VRAM is " +
                  std::to_string(b.total_vram_bytes / (1024LL * 1024 * 1024)) +
                  " GB");
            break;
        }
    }
}

void validate_speculation_pool(const Config& cfg, ValidationResult& r) {
    if (!cfg.speculation.enabled) return;

    const auto& kv = cfg.memory.kv_cache;
    const auto& serving = cfg.serving;

    // Max speculation depth across all draft methods
    int max_spec_depth = 0;
    if (cfg.speculation.mtp.enabled)
        max_spec_depth = std::max(max_spec_depth, cfg.speculation.mtp.max_depth);
    if (cfg.speculation.prompt_lookup.enabled)
        max_spec_depth = std::max(max_spec_depth, cfg.speculation.prompt_lookup.max_continuation_length);

    if (max_spec_depth <= 0) return;

    // INV-0.7: speculation_pool_pages >= max_concurrent_requests * max_speculation_depth
    // Only checkable when max_pages_per_gpu is an explicit number
    if (auto* pages_ptr = std::get_if<int>(&kv.max_pages_per_gpu)) {
        int pages_per_gpu = *pages_ptr;
        int spec_pages_per_gpu =
            static_cast<int>(static_cast<double>(pages_per_gpu) * kv.speculation_pool_fraction);
        int num_gpus = static_cast<int>(cfg.hardware.gpus.size());
        if (num_gpus <= 0) return;

        // Requests distributed across GPUs; check per-GPU capacity
        int requests_per_gpu = (serving.max_concurrent_requests + num_gpus - 1) / num_gpus;
        int required_per_gpu = requests_per_gpu * max_spec_depth;

        if (spec_pages_per_gpu < required_per_gpu) {
            error(r, "memory.kv_cache",
                  "Speculation pool too small: " + std::to_string(spec_pages_per_gpu) +
                  " pages per GPU, but need at least " + std::to_string(required_per_gpu) +
                  " (ceil(max_concurrent_requests/num_gpus)=" +
                  std::to_string(requests_per_gpu) +
                  " * max_speculation_depth=" + std::to_string(max_spec_depth) +
                  ") [INV-0.7]");
        }
    }
}

void validate_speculation_pool_fraction(const Config& cfg, ValidationResult& r) {
    double f = cfg.memory.kv_cache.speculation_pool_fraction;
    if (f < 0.0 || f > 1.0) {
        error(r, "memory.kv_cache.speculation_pool_fraction",
              "Must be in range [0.0, 1.0]");
    }
    if (cfg.speculation.enabled && f <= 0.0) {
        error(r, "memory.kv_cache.speculation_pool_fraction",
              "Must be positive when speculation is enabled");
    }
}

void validate_dspark(const Config& cfg, ValidationResult& r) {
    // DSP-1 config contract — binds only when the dspark method is selected
    // (the knobs are inert otherwise).
    if (cfg.speculation.method != SpeculationMethodType::dspark) return;
    const auto& d = cfg.speculation.dspark;

    // STS calibration temperatures are per block position: length must equal
    // block_size (γ) when non-empty (empty = identity, no calibration).
    if (!d.sts_temperatures.empty() &&
        static_cast<int>(d.sts_temperatures.size()) != d.block_size) {
        error(r, "speculation.dspark.sts_temperatures",
              "Length must equal block_size (gamma=" +
                  std::to_string(d.block_size) + ") when non-empty, got " +
                  std::to_string(d.sts_temperatures.size()));
    }

    // The scheduler consumes per-position survival probabilities from the
    // trained confidence head — non-off modes require it wired in.
    if (d.scheduler_mode != DsparkSchedulerMode::off && !d.confidence_enabled) {
        error(r, "speculation.dspark.scheduler_mode",
              "scheduler_mode=static_threshold/throughput requires "
              "confidence_enabled=true (the scheduler consumes the trained "
              "confidence head's survival probabilities)");
    }

    // DSP-2 checkpoint-shaped fields (defaults = the shipped RedHatAI
    // GLM-5.2 speculator; the loader cross-validates against the checkpoint's
    // own config.json — these checks catch self-inconsistent configs early).
    if (d.speculative_tokens < 1 || d.speculative_tokens > d.block_size) {
        error(r, "speculation.dspark.speculative_tokens",
              "Must be in [1, block_size] (block " + std::to_string(d.block_size) +
                  " = anchor + up to " + std::to_string(d.block_size - 1) +
                  " speculative tokens), got " +
                  std::to_string(d.speculative_tokens));
    }
    if (d.aux_hidden_state_layer_ids.empty()) {
        error(r, "speculation.dspark.aux_hidden_state_layer_ids",
              "Must be non-empty: the DFlash draft ingests the target's aux "
              "hidden states via the fc fusion");
    }
    for (size_t i = 0; i < d.aux_hidden_state_layer_ids.size(); ++i) {
        int id = d.aux_hidden_state_layer_ids[i];
        if (id < 0 ||
            (cfg.model.num_hidden_layers > 0 && id >= cfg.model.num_hidden_layers)) {
            error(r, "speculation.dspark.aux_hidden_state_layer_ids",
                  "Layer id " + std::to_string(id) +
                      " outside target layer range [0, " +
                      std::to_string(cfg.model.num_hidden_layers) + ")");
        }
        if (i > 0 && d.aux_hidden_state_layer_ids[i] <=
                         d.aux_hidden_state_layer_ids[i - 1]) {
            error(r, "speculation.dspark.aux_hidden_state_layer_ids",
                  "Layer ids must be strictly increasing");
        }
    }
    // mask_token_id indexes the draft's embed_tokens, which is EMBED-vocab
    // sized (>= draft_vocab_size under a reduced-vocab d2t checkpoint —
    // TD-DSPARK-VOCAB-REMAP); the config carries only the draft vocab, so
    // bound against the larger of it and the target model vocab here — the
    // loader enforces the exact checkpoint embed-vocab bound at load.
    const int64_t mask_bound =
        std::max<int64_t>(d.draft_vocab_size, cfg.model.vocab_size);
    if (d.mask_token_id < 0 || d.mask_token_id >= mask_bound) {
        error(r, "speculation.dspark.mask_token_id",
              "Must be in [0, max(draft_vocab_size, model.vocab_size)=" +
                  std::to_string(mask_bound) + "), got " +
                  std::to_string(d.mask_token_id));
    }
    if (d.checkpoint_path.empty()) {
        error(r, "speculation.dspark.checkpoint_path",
              "Must be non-empty when speculation.method=dspark (directory of "
              "the speculators-v0.5 draft checkpoint)");
    }
    for (int pos : d.draft_gpus) {
        if (pos < 0 || pos >= static_cast<int>(cfg.hardware.gpus.size())) {
            error(r, "speculation.dspark.draft_gpus",
                  "GPU position " + std::to_string(pos) +
                      " outside hardware.gpus range [0, " +
                      std::to_string(cfg.hardware.gpus.size()) + ")");
        }
    }
}

void validate_cuda_graphs(const Config& /*cfg*/, ValidationResult& /*r*/) {
    // INV-0.6: capture_expert_ffn is a const (always false) in the schema,
    // so no runtime check is needed. Kept as a placeholder for future constraints.
}

void validate_dcp(const Config& cfg, ValidationResult& r) {
    // dcp_enabled is auto-derived from tensor_parallelism by the resolver;
    // validate that the derived value is consistent with tp_array.
    if (cfg.hardware.dcp_enabled && cfg.hardware.tp_array.size() < 2) {
        error(r, "hardware.dcp_enabled",
              "DCP is enabled (tensor_parallelism >= 2) but tp_array has fewer than 2 GPUs (" +
              std::to_string(cfg.hardware.tp_array.size()) + ")");
    }

    // dcp_indexer_mode "local" is harmless but meaningless for non-DSA models.
    if (cfg.hardware.dcp_indexer_mode == DcpIndexerMode::local && cfg.model.index_topk <= 0) {
        warn(r, "hardware.dcp_indexer_mode",
             "dcp_indexer_mode is 'local' but model.index_topk <= 0 (DSA disabled); setting has no effect");
    }
}

void validate_serving(const Config& cfg, ValidationResult& r) {
    if (cfg.serving.max_concurrent_requests <= 0)
        error(r, "serving.max_concurrent_requests", "Must be positive");
    if (cfg.serving.max_sequence_length <= 0)
        error(r, "serving.max_sequence_length", "Must be positive");
    if (cfg.serving.port < 1 || cfg.serving.port > 65535)
        error(r, "serving.port", "Must be in range [1, 65535]");
}

void validate_stable_zone_fraction(const Config& cfg, ValidationResult& r) {
    double f = cfg.memory.expert_cache.stable_zone_fraction;
    if (f < 0.0 || f > 1.0) {
        error(r, "memory.expert_cache.stable_zone_fraction",
              "Must be in range [0.0, 1.0]");
    }

    for (size_t i = 0; i < cfg.hardware.gpus.size(); ++i) {
        const auto& g = cfg.hardware.gpus[i];
        if (g.vram_allocation_gb) {
            double gf = g.vram_allocation_gb->stable_zone_fraction;
            // sentinel -1.0 means "use global default"; only validate if explicitly set
            if (gf >= 0.0 && gf > 1.0) {
                error(r,
                      "hardware.gpus[" + std::to_string(i) +
                      "].vram_allocation_gb.stable_zone_fraction",
                      "Must be in range [0.0, 1.0]");
            }
        }
    }
}

void validate_attention_backend(const Config& cfg, ValidationResult& r) {
    if (cfg.compute.attention_backend == AttentionBackendType::turboquant_mla) {
        if (cfg.model.kv_lora_rank <= 0) {
            error(r, "compute.attention_backend",
                  "turboquant_mla requires kv_lora_rank > 0 (MLA models only), "
                  "but kv_lora_rank is " + std::to_string(cfg.model.kv_lora_rank));
        }
    }

    const bool is_v4 = cfg.model.architecture == Architecture::deepseek_v4;
    const bool v4_backend =
        cfg.compute.attention_backend == AttentionBackendType::csa_hca ||
        cfg.compute.attention_backend == AttentionBackendType::csa_hca_tq ||
        cfg.compute.attention_backend == AttentionBackendType::csa_hca_tq_mix;
    if (is_v4 && !v4_backend) {
        error(r, "compute.attention_backend",
              "deepseek_v4 requires a V4 attention backend (csa_hca, "
              "csa_hca_tq, or csa_hca_tq_mix) — MLA backends "
              "(snapmla/turboquant_mla) are incompatible with V4 hybrid "
              "CSA/HCA attention");
    }
    if (!is_v4 && v4_backend) {
        error(r, "compute.attention_backend",
              "csa_hca* backends require model.architecture=deepseek_v4");
    }
}

// ── DeepSeek-V4 architecture rules (spec/DEEPSEEK4_PLAN.md V4-1c) ────────────
// V4-4a/b: gating scoring function + SwiGLU clamp support surface.
// Fail-closed: reject combinations no kernel implements.
void validate_gating_activation(const Config& cfg, ValidationResult& r) {
    // `softmax` is a schema enum value with NO gating-kernel implementation
    // anywhere (deps topk_gating supports sigmoid + sqrtsoftplus only).
    if (cfg.model.gating_score_fn == GatingScoreFn::softmax) {
        error(r, "model.gating_score_fn",
              "'softmax' has no gating-kernel implementation (supported: "
              "sigmoid, sqrtsoftplus)");
    }
    if (cfg.model.swiglu_limit < 0.0) {
        error(r, "model.swiglu_limit",
              "Must be >= 0 (0 = no clamp; V4-Flash: 10.0)");
    }
    // The fused SiLU-mul->NVFP4 activation-quant kernel has no swiglu_limit
    // support; NVFP4 expert weights + a clamp would silently skip the clamp
    // on the routed path. (V4 experts are GGUF MXFP4 — not affected.)
    if (cfg.model.swiglu_limit > 0.0 &&
        cfg.quantization.weights == WeightQuant::nvfp4) {
        error(r, "model.swiglu_limit",
              "swiglu_limit > 0 is unsupported with quantization.weights="
              "'nvfp4' (fused SiLU-mul->NVFP4 kernel has no clamp support)");
    }
}

void validate_v4(const Config& cfg, ValidationResult& r) {
    const auto& m = cfg.model;
    const bool is_v4 = m.architecture == Architecture::deepseek_v4;

    if (!is_v4) {
        // V4-only knobs are inert-but-suspicious on other architectures.
        if (!m.compress_ratios.empty()) {
            warn(r, "model.compress_ratios",
                 "compress_ratios is set but architecture is not deepseek_v4; "
                 "ignored");
        }
        return;
    }

    // V4 is native MQA-over-latent: one shared KV head. (The in-tree MLA
    // convention num_key_value_heads == num_attention_heads does NOT apply.)
    if (m.num_key_value_heads != 1) {
        error(r, "model.num_key_value_heads",
              "deepseek_v4 is native MQA-over-latent: num_key_value_heads "
              "must be 1, got " + std::to_string(m.num_key_value_heads));
    }
    if (m.head_dim <= 0) {
        error(r, "model.head_dim",
              "deepseek_v4 requires head_dim > 0 (V4-Flash: 512)");
    } else if (m.head_dim <= m.qk_rope_head_dim) {
        error(r, "model.head_dim",
              "head_dim (" + std::to_string(m.head_dim) +
              ") must exceed qk_rope_head_dim (" +
              std::to_string(m.qk_rope_head_dim) + ")");
    }

    // compress_ratios: >= num_hidden_layers entries in {0,4,128}; the HF/GGUF
    // array may carry trailing MTP-layer zeros (V4-Flash ships 46 = 43 + 3) —
    // extra entries are accepted but must be 0.
    if (static_cast<int>(m.compress_ratios.size()) < m.num_hidden_layers) {
        error(r, "model.compress_ratios",
              "Must have at least num_hidden_layers (" +
              std::to_string(m.num_hidden_layers) + ") entries, got " +
              std::to_string(m.compress_ratios.size()));
    }
    for (size_t i = 0; i < m.compress_ratios.size(); ++i) {
        const int v = m.compress_ratios[i];
        if (v != 0 && v != 4 && v != 128) {
            error(r, "model.compress_ratios",
                  "Entry [" + std::to_string(i) + "] = " + std::to_string(v) +
                  " invalid: values must be in {0, 4, 128}");
        } else if (v != 0 && static_cast<int>(i) >= m.num_hidden_layers) {
            error(r, "model.compress_ratios",
                  "Entry [" + std::to_string(i) + "] = " + std::to_string(v) +
                  " beyond num_hidden_layers (" +
                  std::to_string(m.num_hidden_layers) +
                  ") must be 0 (trailing MTP-layer padding)");
        }
    }

    const bool has_compressed_layers = [&] {
        const int n = std::min<int>(m.num_hidden_layers,
                                    static_cast<int>(m.compress_ratios.size()));
        for (int i = 0; i < n; ++i)
            if (m.compress_ratios[static_cast<size_t>(i)] != 0) return true;
        return false;
    }();
    if (has_compressed_layers && m.compress_rope_theta <= 0.0) {
        error(r, "model.compress_rope_theta",
              "Must be positive when compress_ratios has compressed layers "
              "(V4-Flash: 160000)");
    }
    if (m.sliding_window <= 0) {
        error(r, "model.sliding_window",
              "deepseek_v4 requires sliding_window > 0 (V4-Flash: 128)");
    }

    // Grouped output projection.
    if (m.o_groups > 1) {
        if (m.num_attention_heads % m.o_groups != 0) {
            error(r, "model.o_groups",
                  "Must divide num_attention_heads (" +
                  std::to_string(m.num_attention_heads) + "), got " +
                  std::to_string(m.o_groups));
        }
        if (m.o_lora_rank <= 0) {
            error(r, "model.o_lora_rank",
                  "Must be positive when o_groups > 1 (V4-Flash: 1024)");
        }
    }

    // V4-2c (TD-V4-TP): TP sharding constraints. Rank shards must be
    // group-aligned (rank heads = whole o_proj groups) and per-rank heads
    // must fit the padded decode-kernel tile bound (<= 128; the executor
    // pads sub-64 rank head counts to the kernel's 64-head tile).
    const int tp = cfg.parallelism.tensor_parallelism;
    if (tp > 1) {
        if (m.num_attention_heads % tp != 0) {
            error(r, "parallelism.tensor_parallelism",
                  "deepseek_v4: num_attention_heads (" +
                  std::to_string(m.num_attention_heads) +
                  ") must be divisible by tensor_parallelism (" +
                  std::to_string(tp) + ")");
        }
        if (m.o_groups > 1 && m.o_groups % tp != 0) {
            error(r, "parallelism.tensor_parallelism",
                  "deepseek_v4: o_groups (" + std::to_string(m.o_groups) +
                  ") must be divisible by tensor_parallelism (" +
                  std::to_string(tp) + ") — grouped o_proj shards BY GROUP");
        }
        if (m.o_groups > 1 && m.num_attention_heads % m.o_groups == 0
            && m.num_attention_heads % tp == 0) {
            const int heads_per_group = m.num_attention_heads / m.o_groups;
            if ((m.num_attention_heads / tp) % heads_per_group != 0) {
                error(r, "parallelism.tensor_parallelism",
                      "deepseek_v4: per-rank heads must be a whole number "
                      "of o_proj groups (heads/tp % (heads/o_groups) != 0)");
            }
        }
        if (m.num_attention_heads % tp == 0
            && m.num_attention_heads / tp > 128) {
            error(r, "parallelism.tensor_parallelism",
                  "deepseek_v4: per-rank head count exceeds the csa_fp8 "
                  "decode-kernel tile bound (128)");
        }
    }

    // mHC.
    if (m.hc_mult > 1) {
        if (m.hc_sinkhorn_iters < 1) {
            error(r, "model.hc_sinkhorn_iters",
                  "Must be >= 1 when hc_mult > 1 (V4: 20)");
        }
        if (m.hc_eps <= 0.0) {
            error(r, "model.hc_eps", "Must be positive when hc_mult > 1");
        }
    }

    if (m.num_hash_layers > m.num_hidden_layers) {
        error(r, "model.num_hash_layers",
              "Cannot exceed num_hidden_layers (" +
              std::to_string(m.num_hidden_layers) + ")");
    }
}

// P-24b memory.arena_attach.on_conflict × persist compatibility. persist=true
// declares the holder store authoritative and never-wiped, so only 'fail' is
// coherent with it. Also enforced (throwing) in finalize_config so every
// production load_config path fails at PARSE time, never at attach.
void validate_arena_attach(const Config& cfg, ValidationResult& r) {
    const auto& a = cfg.memory.arena_attach;
    if (!a.persist) return;
    if (a.on_conflict == ArenaOnConflict::kill) {
        error(r, "memory.arena_attach.on_conflict",
              "'kill' contradicts persist=true: persist means the holder "
              "store is NEVER wiped. Use on_conflict='fail' (or omit it); to "
              "wipe + cold-rebuild intentionally, boot once with "
              "persist=false.");
    }
    if (a.on_conflict == ArenaOnConflict::new_arena) {
        error(r, "memory.arena_attach.on_conflict",
              "'new' contradicts persist=true: the deployment declared the "
              "persistent holder store as this engine's arena — silently "
              "serving from a throwaway process-private arena would bypass "
              "it and double-allocate ~500 GB host RAM behind the operator. "
              "Use on_conflict='fail' (or omit it), or set persist=false if "
              "a private arena is really intended.");
    }
}

}  // namespace

ValidationResult validate_config(const Config& cfg) {
    ValidationResult result;

    validate_gpu_config(cfg, result);
    validate_tp_array(cfg, result);
    validate_tp_mode_requires_tp_array(cfg, result);
    validate_dcp(cfg, result);
    validate_nvme(cfg, result);
    validate_model_dimensions(cfg, result);
    validate_pinned_layers(cfg, result);
    validate_eviction_weights(cfg, result);
    validate_performance_objective(cfg, result);
    validate_vram_budget(cfg, result);
    validate_speculation_pool(cfg, result);
    validate_speculation_pool_fraction(cfg, result);
    validate_dspark(cfg, result);
    validate_cuda_graphs(cfg, result);
    validate_serving(cfg, result);
    validate_stable_zone_fraction(cfg, result);
    validate_attention_backend(cfg, result);
    validate_gating_activation(cfg, result);
    validate_v4(cfg, result);
    validate_arena_attach(cfg, result);

    return result;
}

}  // namespace layerstorm::config

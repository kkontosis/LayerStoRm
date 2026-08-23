#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "model/quantization/quant_interface.h"
#include "model/weight_loader/gguf_reader.h"
#include "model/weight_loader/safetensors_reader.h"
#include "model/weight_loader/weight_handler.h"
#include "model/weight_loader/tensor_id.h"

namespace layerstorm::model {

class ModelConfig;
class LayerRegistry;

}  // namespace layerstorm::model

namespace layerstorm::config {

struct Config;

}  // namespace layerstorm::config

namespace layerstorm::model {

// ── LoadedModel ──────────────────────────────────────────────────────────────
// The result of loading a model's weights from safetensors.
// All RawTensor data spans point into mmap'd shard files — the LoadedModel
// owns the SafetensorsReader handles that keep the mmaps alive.

struct LoadedModel {
    /// Weights for one transformer layer.
    struct LayerWeights {
        int layer_idx = -1;

        // Attention projections: q_a, q_a_norm, q_b, kv_a, kv_a_norm, kv_b, o_proj
        std::vector<WeightBundle> attention;

        // DSA indexer tensors (empty if model has no DSA)
        std::vector<WeightBundle> indexer;

        // Layer norms: input_layernorm, post_attention_layernorm
        std::vector<WeightBundle> norms;

        // Gating/router: gate weight + e_score_correction_bias (empty for dense layers)
        std::vector<WeightBundle> gating;

        // Shared expert FFN: gate/up/down (empty for dense layers)
        std::vector<WeightBundle> shared_expert;

        // Dense FFN: gate/up/down (empty for MoE layers)
        std::vector<WeightBundle> dense_ffn;

        // Routed experts: [expert_idx] -> {gate, up, down} bundles
        std::vector<std::vector<WeightBundle>> routed_experts;
    };

    /// All transformer layers [0..num_hidden_layers-1].
    std::vector<LayerWeights> layers;

    /// Model-level weights.
    std::optional<WeightBundle> embedding;
    std::optional<WeightBundle> output_head;
    std::optional<WeightBundle> final_norm;

    /// DeepSeek-V4 model-level mHC output collapse (output_hc_{fn,base,scale});
    /// empty for non-V4 models.
    std::vector<WeightBundle> output_hc;

    /// MTP (Multi-Token Prediction) weights, if present.
    struct MtpWeights {
        int base_layer_idx = -1;  // First MTP layer index (e.g. 61 for V3.2)
        std::vector<WeightBundle> tensors;       // enorm, hnorm, eh_proj, shared_head, embed
        std::vector<LayerWeights> block_layers;  // MTP block decoder layers
    };
    std::optional<MtpWeights> mtp;

    /// Shard file handles. Keep alive for mmap data lifetime.
    std::vector<SafetensorsReader> shards;

    /// GGUF shard handles (when weights_format == gguf). Keep alive for mmap
    /// data lifetime, parallel to `shards` for the safetensors path.
    std::vector<GgufReader> gguf_shards;

    /// Statistics
    int64_t total_weight_bytes = 0;
    int total_tensors_loaded = 0;

    /// P-24b RAM hygiene: after every PINNED weight reached VRAM
    /// (upload_pinned_weights + quantize_attention_weights), drop the host
    /// bytes — free the transform-owned buffers (GG-9 embed/head BF16, GLM-1
    /// combined kv_b, Q8_0 requants) and MADV_DONTNEED every GGUF shard mmap
    /// (clean read-only pages; re-faultable from file). Routed experts are
    /// untouched (separate PrepackedSource path; WP-6 clears them). Bundle
    /// METADATA (ids, shapes, dtypes) survives — only the byte spans are
    /// emptied. Returns the number of owned-heap bytes freed.
    size_t release_pinned_host_bytes();
};

/// Round 2b streaming dequant: BF16-convert an element RANGE of a k-quant
/// (or float) tensor without materializing the full tensor — the upload path
/// streams embed/lm_head chunk-by-chunk straight into pinned staging, so the
/// old ~3.8 GB F32 intermediate and ~1.8 GB BF16 host copies never exist.
/// elem_off/elem_cnt must be multiples of the quant block size (256 for
/// K-quants, 32 for Q8_0; any value for float dtypes). Throws on misalignment.
std::vector<uint16_t> dequant_kquant_range_to_bf16(const RawTensor& t,
                                                   int64_t elem_off,
                                                   int64_t elem_cnt);

/// Load all model weights from safetensors files.
///
/// @param cfg Full system config (for weights_path, weights_format)
/// @param model_cfg Parsed model config (for architecture, dimensions)
/// @param registry Layer registry (for validation of expected components)
/// @return LoadedModel with all weights mapped via mmap
/// @throws std::runtime_error on I/O, format, or validation errors
LoadedModel load_weights(const config::Config& cfg,
                         const ModelConfig& model_cfg,
                         const LayerRegistry& registry,
                         bool skip_routed_experts = false);

/// Load all model weights from a GGUF container (single .gguf file or a
/// directory of split *.gguf files). Parallel to load_weights() but for the
/// GGUF format: parses the container, maps GGUF tensor names to TensorIds,
/// de-stacks the 3D routed-expert tensors into per-expert bundles, and PRESERVES
/// each tensor's GgufKQuantType on its RawTensor (so the GEMM dispatch — GG-4
/// attention / GG-5 experts — can select the right kernel). Dispatched
/// automatically by load_weights() when cfg.model.weights_format == gguf.
LoadedModel load_weights_gguf(const config::Config& cfg,
                              const ModelConfig& model_cfg,
                              const LayerRegistry& registry,
                              bool skip_routed_experts = false);

/// Resolve the ordered .gguf shard files for a weights_path. Accepts a single
/// .gguf file, a split-set member `<stem>-NNNNN-of-MMMMM.gguf` (expanded to all
/// M members in the same directory, ordered 00001..MMMMM), or a directory of
/// *.gguf files (sorted by name). Throws if a referenced split set is
/// incomplete. Exposed for testing.
std::vector<std::filesystem::path> resolve_gguf_files(const std::string& weights_path);

/// TD-VOCAB-AUTODETECT: the weights-derived vocabulary width — the row count
/// of the embedding tensor (output-head fallback), padding included. This is
/// the authoritative logits width. Cheap header-only scan, both formats:
///   safetensors: model.embed_tokens.weight shape[0] (lm_head.weight fallback);
///                sharded models resolve the owning shard via the index.
///   GGUF:        token_embd.weight dims[1] (output.weight fallback) — GGUF
///                dims are reversed (dims[0] = fastest = hidden columns).
/// Throws std::runtime_error when the weights or the tensor cannot be found.
int64_t detect_weights_vocab_rows(const config::Config& cfg);

/// TD-VOCAB-AUTODETECT: resolve cfg.model.vocab_size against the weights.
/// The single cross-check seam (called from Engine::init_modules and
/// Engine::run_prepack BEFORE LayerRegistry / VramAllocator consume the
/// value):
///   - vocab_size == 0 (absent) -> ADOPT the weights-derived width (logged);
///   - vocab_size set and == weights width -> pass;
///   - vocab_size set and != weights width -> FAIL LOUD with both numbers
///     (weights are ground truth; config is explicit intent; disagreement is
///     always a bug — a wrong value silently mis-sizes logits readbacks,
///     sampling, and guided-decode grammar masks).
void resolve_vocab_size(config::Config& cfg);

/// Result of assembling the split MLA up-projection into a combined kv_b_proj
/// (GLM-1). Holds the owned BF16 buffer and the combined row-major shape
/// `[n_head*(qk_nope_head_dim + v_head_dim), kv_lora_rank]`.
struct AssembledKvB {
    std::shared_ptr<std::vector<std::byte>> buf;  // owned BF16 bytes
    std::vector<int64_t> shape;                   // [n_head*(P+V), L]
};

/// Assemble a combined BF16 `kv_b_proj` from llama.cpp's split MLA up-projection
/// tensors (GLM-1). Some GGUFs ship the MLA up-projection PRE-SPLIT (the MLA-
/// optimized llama.cpp layout) instead of the combined `attn_kv_b`:
///   attn_k_b : per head `[kv_lora_rank, qk_nope_head_dim]` = W_UK TRANSPOSED
///   attn_v_b : per head `[v_head_dim, kv_lora_rank]`       = W_UV (as the engine wants)
/// (RawTensor.shape is the GGUF-reversed row-major shape: k_b = [H, L, P],
/// v_b = [H, V, L].) This dequants each tensor per its own dtype/gguf_type to
/// f32 (lossless for Q8_0), transposes attn_k_b per head to `[qk_nope, kv_lora]`,
/// copies attn_v_b per head as-is, and stacks `[W_UK; W_UV]` per head into the
/// combined `[n_head*(qk_nope + v_head_dim), kv_lora]` BF16 layout the absorbed-
/// MLA path (q_absorb W_UK / kv_bv W_UV) consumes — matching ktransformers
/// `cat(attn_k_b.transpose(1,2), attn_v_b, dim=1)`.
/// Throws std::runtime_error on shape mismatch or an unsupported dequant type.
AssembledKvB assemble_split_kv_b_proj(const RawTensor& attn_k_b,
                                      const RawTensor& attn_v_b);

/// Per-projection GGUF k-quant types of the routed experts in a loaded GGUF
/// model. Experts are stacked with ONE ggml_type per projection (uniform across
/// experts AND layers in every real GGUF), so three types describe them all.
struct GgufModelExpertTypes {
    GgufKQuantType gate = GgufKQuantType::Q4_K;
    GgufKQuantType up   = GgufKQuantType::Q4_K;
    GgufKQuantType down = GgufKQuantType::Q4_K;
};

/// Scan a loaded GGUF model's routed experts and return the per-projection
/// k-quant types (from the first MoE layer's first expert). Throws if the model
/// has no routed experts loaded (e.g. skip_routed_experts was set) or if a
/// projection's type is inconsistent across layers/experts.
GgufModelExpertTypes gguf_expert_types_from_model(const LoadedModel& model);

/// Cheap pre-scan: read just the GGUF tensor infos (no data load) at
/// `weights_path` (single .gguf file or directory of split files) and return the
/// routed experts' per-projection k-quant types. Used by the engine to build the
/// expert QuantInterface BEFORE the full weight load (LayerRegistry / VRAM
/// budget need it). Throws if no stacked routed-expert tensors are found.
GgufModelExpertTypes gguf_expert_types_from_path(const std::string& weights_path,
                                                 bool use_mmap = true);

/// GG-9: scan a loaded GGUF model's FFN tensors for the given owner
/// (`routed_expert`, `shared_expert`, or `dense_ffn`) and return the
/// per-projection MAXIMUM-byte k-quant type seen (an upper bound that any tensor
/// of that owner fits). Real "XL"-mix GGUFs (e.g. GLM-4.7-Flash-UD-Q5_K_XL) use
/// DIFFERENT k-quants per layer/owner — shared experts Q8_0, dense Q5_K/Q6_K,
/// routed a per-layer mix — so a single global type under-sizes some slots; this
/// gives each owner its own correct upper bound. Returns nullopt if the owner has
/// no GGUF-typed FFN tensors in the model.
std::optional<GgufModelExpertTypes> gguf_owner_types_from_model(
    const LoadedModel& model, TensorOwner owner);

/// Normalized NVFP4 activation input_scale values for one layer (TRT-LLM
/// semantics): fc31 = max over the layer's experts of max(gate, up) input
/// scales (gate and up share one quantized activation, so their fields MUST
/// be equal in the packed slot); fc2 = max over experts of down input scales.
/// Values <= 0 mean "not computed" — pack falls back to the per-expert
/// bundle values with a local max(gate, up) merge.
struct Nvfp4InputScaleNorm {
    float fc31 = 0.f;
    float fc2 = 0.f;
};

/// Compute the per-layer input_scale normalization over all routed experts of
/// one layer. Warns (once per layer) if any expert's gate and up calibrations
/// diverge — ModelOpt checkpoints carry identical values (TRT-LLM asserts it).
Nvfp4InputScaleNorm compute_nvfp4_input_scale_norm(
    const std::vector<std::vector<WeightBundle>>& layer_experts, int layer_idx);

/// Ensure an expert's weight bundles are packed into a contiguous buffer
/// matching the ExpertCache slot layout. Called lazily from resolve_host_source().
/// No-op if already packed (bundles[0].owned_buf != nullptr).
/// Handles NVFP4 (U8), FP8 E4M3, and FP8 E5M2 dtypes.
/// norm (NVFP4 only): per-layer input_scale normalization to bake into the slot.
void ensure_expert_packed(std::vector<WeightBundle>& bundles,
                          const ExpertShape& shape,
                          const Nvfp4InputScaleNorm* norm = nullptr);

/// Legacy alias for ensure_expert_packed (forwards to it).
void ensure_nvfp4_expert_packed(std::vector<WeightBundle>& bundles,
                                const ExpertShape& shape);

/// Pack FP8 routed expert bundles into a contiguous slot-layout buffer.
/// After packing, bundles[0].packed_slot spans the packed data and
/// bundles[0].owned_buf holds the allocation (null if zero-copy from mmap).
void pack_fp8_expert(std::vector<WeightBundle>& bundles, const ExpertShape& shape);

/// Pack GGUF k-quant routed expert bundles into a contiguous slot-layout buffer:
/// the raw GGUF blocks of gate | up | down concatenated verbatim (k-quant scales
/// and mins pack INSIDE each super-block — no separate scale region, no reformat,
/// replicated == 0). The per-projection sizes come from each bundle's OWN
/// gguf_type (GG-6/GG-10): a per-layer mixed "XL" GGUF packs each layer at its
/// own k-quant sizes, NOT at a global triple. If `expected_types` is non-null,
/// each bundle's k-quant type must match it exactly (strict per-layer validation
/// — used by the prepacker against its per-layer type table).
/// After packing, bundles[0].packed_slot spans the packed data and
/// bundles[0].owned_buf holds the allocation (null if the mmap blocks are already
/// contiguous in gate|up|down order). Returns false (packed_slot untouched) on
/// missing bundle/type, type mismatch, or block-byte mismatch.
bool pack_gguf_expert(std::vector<WeightBundle>& bundles, const ExpertShape& shape,
                      const GgufModelExpertTypes* expected_types = nullptr);

/// Pack NVFP4 routed expert bundles into a contiguous slot-layout buffer.
/// After packing, bundles[0].packed_slot spans the packed data and
/// bundles[0].owned_buf holds the allocation.
/// Group scales are written in the Sm1xx interleaved SFB layout
/// ("nvfp4-sm1xx" — the only packed NVFP4 convention; see
/// nvfp4_sfb_reformat.h) and input_scale scalars are normalized per
/// Nvfp4InputScaleNorm (gate == up always; layer max when norm is provided).
void pack_nvfp4_expert(std::vector<WeightBundle>& bundles, const ExpertShape& shape,
                       const Nvfp4InputScaleNorm* norm = nullptr);

}  // namespace layerstorm::model

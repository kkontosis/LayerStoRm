#pragma once

//==============================================================================
// FP8 quantization skip list (GF3.3 — PLAN.md Phase GF3;
// spec/GLM-5.3-FLASH-MODELINFO.md §7)
//
// Native-FP8 HF checkpoints (DeepSeek/GLM family) are MIXED precision: the
// checkpoint's config.json `quantization_config.modules_to_not_convert`
// enumerates the modules whose tensors stay BF16/F32 inside the otherwise
// FP8-e4m3 block-quantized checkpoint. For GLM-5.3-Flash that is 1509 literal
// entries: whole KDA layers, sparse-layer kv_b_proj, the whole indexer, all
// norms/gates/hc tensors, lm_head/embed, MTP extras, and the entire vision
// tower.
//
// THE NAMESPACE TRAP (MODELINFO §1/§7): the list names MODULE paths of the
// loaded HF model — `model.layers.N.*`, `visual.*`, `model.visual` — while
// the checkpoint's TENSOR paths live under the multimodal wrapper —
// `model.language_model.layers.N.*`, `model.visual.*`. A literal string
// match therefore matches NOTHING and an FP8 converter would silently
// quantize tensors that must stay BF16 (plausible-but-wrong numerics, not a
// crash). This matcher canonicalizes BOTH sides into one namespace before
// matching, and load_weights() additionally fails LOUDLY when a present
// skip list matches zero loaded tensors.
//
// This engine never converts dtypes at load (the safetensors header is the
// dtype authority), so the skip list's role here is a correctness
// cross-check: in an FP8 checkpoint every >=2-D weight-role tensor must be
// FP8 exactly when it is NOT in the skip list. A disagreement means either
// the checkpoint or our reading of it is wrong — both are load-stopping.
//==============================================================================

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace layerstorm::model {

class QuantSkipList {
public:
    /// Parse `<model_dir>/config.json` and build the skip list from
    /// `quantization_config.modules_to_not_convert`. Returns nullopt when the
    /// file, the quantization_config section, or the list is absent (plain
    /// BF16 checkpoints, GGUF dirs, prepacked artifacts). Throws
    /// std::runtime_error only on a malformed present section.
    static std::optional<QuantSkipList> load_from_model_dir(
        const std::filesystem::path& model_dir);

    /// Build directly from entries (tests, callers with their own config
    /// plumbing). `quant_method` as in config.json ("fp8"), may be empty.
    static QuantSkipList from_entries(std::vector<std::string> entries,
                                      std::string quant_method);

    /// True when the TENSOR named `tensor_name` (safetensors tensor path,
    /// role suffix included, either namespace) belongs to a module the
    /// checkpoint keeps unquantized. Matching, on canonicalized names:
    ///   1. exact module match after stripping a trailing role suffix
    ///      (.weight/.weight_scale/.weight_scale_inv/.bias) — and the full
    ///      name as-is (entries like `...gate.e_score_correction_bias` and
    ///      `...proj.bias` carry the suffix themselves);
    ///   2. subtree match: an entry that is a dot-prefix of the name covers
    ///      the whole module subtree (`model.visual` covers every
    ///      `model.visual.*` tensor);
    ///   3. module-class catch-alls: entries with no '.' (e.g. `dt_bias`,
    ///      `weights_proj`, `lm_head`, `visual`) match any name containing
    ///      that path segment.
    bool matches(std::string_view tensor_name) const;

    /// Canonical namespace form used for matching (exposed for tests):
    /// `model.language_model.X` -> `model.X`; `model.visual[...]` ->
    /// `visual[...]`.
    static std::string canonicalize(std::string_view name);

    size_t entry_count() const { return entry_count_; }
    const std::string& quant_method() const { return quant_method_; }

private:
    QuantSkipList() = default;

    std::unordered_set<std::string> exact_;     // canonical dotted entries
    std::unordered_set<std::string> segments_;  // single-segment catch-alls
    size_t entry_count_ = 0;
    std::string quant_method_;
};

}  // namespace layerstorm::model

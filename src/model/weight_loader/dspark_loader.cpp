// DSpark draft-checkpoint weight loader implementation (DSP-2).
// See dspark_loader.h for the checkpoint ground truth and the placement
// decision. Layout/semantics references: the checkpoint's config.py
// (DSparkSpeculatorConfig ⊂ DFlashSpeculatorConfig, speculators v0.5) and
// vLLM ref/vllm/vllm/model_executor/models/qwen3_dspark.py (which skips
// mask_embedding/confidence_head — we deliberately load the confidence head:
// DSP-6 wires it).

#include "model/weight_loader/dspark_loader.h"

#include "config/config_parser.h"
#include "core/device_backend.h"
#include "model/quantization/kgroup_quant.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace layerstorm::model {

namespace {

constexpr const char* kConfigFile = "config.json";
constexpr const char* kWeightsFile = "model.safetensors";
constexpr int64_t kArenaAlign = 256;  // device tensor offset alignment

int64_t align_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

[[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error("dspark_loader: " + what);
}

std::string shape_str(const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(s[i]);
    }
    return out + "]";
}

}  // namespace

// ── parse_dspark_checkpoint_config ──────────────────────────────────────────

DsparkCheckpointConfig parse_dspark_checkpoint_config(
        const std::filesystem::path& checkpoint_dir) {
    // Ticket J: a `.gguf` FILE checkpoint_path selects the V4 dflash arm.
    if (dspark_checkpoint_is_gguf(checkpoint_dir))
        return parse_dspark_gguf_checkpoint_config(checkpoint_dir);
    const auto cfg_path = checkpoint_dir / kConfigFile;
    std::ifstream ifs(cfg_path);
    if (!ifs) fail("cannot open " + cfg_path.string());

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(ifs);
    } catch (const std::exception& e) {
        fail("invalid JSON in " + cfg_path.string() + ": " + e.what());
    }

    if (j.value("speculators_model_type", std::string{}) != "dspark") {
        fail(cfg_path.string() + ": speculators_model_type is '" +
             j.value("speculators_model_type", std::string{"<absent>"}) +
             "', expected 'dspark' (speculators v0.5 DSpark checkpoint)");
    }

    DsparkCheckpointConfig c;
    try {
        c.block_size = j.at("block_size").get<int>();
        c.markov_rank = j.at("markov_rank").get<int>();
        c.markov_head_type = j.at("markov_head_type").get<std::string>();
        c.enable_confidence_head = j.at("enable_confidence_head").get<bool>();
        c.confidence_head_with_markov =
            j.at("confidence_head_with_markov").get<bool>();
        c.draft_vocab_size = j.at("draft_vocab_size").get<int64_t>();
        c.mask_token_id = j.at("mask_token_id").get<int64_t>();
        c.max_anchors = j.at("max_anchors").get<int>();
        c.aux_hidden_state_layer_ids =
            j.at("aux_hidden_state_layer_ids").get<std::vector<int>>();
        c.tie_word_embeddings = j.value("tie_word_embeddings", false);

        // speculators_config.proposal_methods[0].speculative_tokens
        const auto& sc = j.at("speculators_config");
        const auto& pms = sc.at("proposal_methods");
        if (!pms.is_array() || pms.empty())
            fail("speculators_config.proposal_methods is empty");
        c.speculative_tokens = pms[0].at("speculative_tokens").get<int>();

        // DFlash backbone dims (Qwen3 dense transformer).
        const auto& t = j.at("transformer_layer_config");
        c.model_type = t.value("model_type", std::string{});
        c.num_hidden_layers = t.at("num_hidden_layers").get<int>();
        c.hidden_size = t.at("hidden_size").get<int64_t>();
        c.num_attention_heads = t.at("num_attention_heads").get<int>();
        c.num_key_value_heads = t.at("num_key_value_heads").get<int>();
        c.head_dim = t.at("head_dim").get<int64_t>();
        c.intermediate_size = t.at("intermediate_size").get<int64_t>();
        c.rms_norm_eps = t.at("rms_norm_eps").get<double>();
        c.vocab_size = t.at("vocab_size").get<int64_t>();
        // rope_theta lives in rope_parameters (transformers v5) or flat.
        if (t.contains("rope_parameters"))
            c.rope_theta = t.at("rope_parameters").at("rope_theta").get<double>();
        else
            c.rope_theta = t.value("rope_theta", 0.0);
    } catch (const nlohmann::json::exception& e) {
        fail(cfg_path.string() + ": missing/mistyped field: " + e.what());
    }

    if (c.num_hidden_layers <= 0 || c.hidden_size <= 0 ||
        c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
        c.head_dim <= 0 || c.intermediate_size <= 0 ||
        c.draft_vocab_size <= 0 || c.markov_rank <= 0 || c.block_size <= 0) {
        fail(cfg_path.string() + ": non-positive backbone/speculator dimension");
    }
    if (c.aux_hidden_state_layer_ids.empty())
        fail(cfg_path.string() + ": aux_hidden_state_layer_ids is empty");
    // Reduced draft vocab (TD-DSPARK-VOCAB-REMAP): draft_vocab_size may be
    // SMALLER than the transformer/embed vocab (the d2t map covers the gap),
    // never larger (a draft id must map to a target id).
    if (c.draft_vocab_size > c.vocab_size)
        fail(cfg_path.string() + ": draft_vocab_size (" +
             std::to_string(c.draft_vocab_size) +
             ") > transformer vocab_size (" + std::to_string(c.vocab_size) +
             ")");
    // mask_token_id indexes embed_tokens, which is EMBED-vocab sized.
    if (c.mask_token_id < 0 || c.mask_token_id >= c.vocab_size)
        fail(cfg_path.string() + ": mask_token_id " +
             std::to_string(c.mask_token_id) + " outside embed vocab [0, " +
             std::to_string(c.vocab_size) + ")");

    return c;
}

// ── load_dspark_draft ───────────────────────────────────────────────────────

namespace {

/// One expected slot: name → destination RawTensor + expected shape/dtype.
struct Slot {
    RawTensor* dst;
    std::vector<int64_t> shape;
    SafetensorsDtype dtype = SafetensorsDtype::BF16;
};

/// Build the full expected-name → slot map from the checkpoint config.
/// Covers ALL tensors of the speculators v0.5 DSpark format.
std::unordered_map<std::string, Slot> build_slot_map(
        const DsparkCheckpointConfig& c, DsparkDraftWeights& w) {
    const int64_t H = c.hidden_size;
    const int64_t V = c.vocab_size;        // target/embed vocab
    const int64_t Vd = c.draft_vocab_size; // draft vocab (== V for full-vocab)
    const int64_t r = c.markov_rank;
    const int64_t D = c.head_dim;
    const int64_t Q = static_cast<int64_t>(c.num_attention_heads) * D;
    const int64_t KV = static_cast<int64_t>(c.num_key_value_heads) * D;
    const int64_t I = c.intermediate_size;
    const int64_t n_aux =
        static_cast<int64_t>(c.aux_hidden_state_layer_ids.size());

    std::unordered_map<std::string, Slot> slots;
    // embed_tokens / markov_w1 consume TARGET ids (anchor, mask, sampled
    // tokens) -> target/embed vocab.  lm_head / markov_w2 produce DRAFT-
    // space logits/bias -> draft vocab (TD-DSPARK-VOCAB-REMAP).
    slots.emplace("embed_tokens.weight", Slot{&w.embed_tokens, {V, H}});
    slots.emplace("lm_head.weight", Slot{&w.lm_head, {Vd, H}});
    slots.emplace("fc.weight", Slot{&w.fc, {H, n_aux * H}});
    slots.emplace("hidden_norm.weight", Slot{&w.hidden_norm, {H}});
    slots.emplace("norm.weight", Slot{&w.final_norm, {H}});
    slots.emplace("markov_head.markov_w1.weight", Slot{&w.markov_w1, {V, r}});
    slots.emplace("markov_head.markov_w2.weight", Slot{&w.markov_w2, {Vd, r}});
    if (c.enable_confidence_head) {
        const int64_t conf_in = H + (c.confidence_head_with_markov ? r : 0);
        slots.emplace("confidence_head.proj.weight",
                      Slot{&w.confidence_proj_weight, {1, conf_in}});
        slots.emplace("confidence_head.proj.bias",
                      Slot{&w.confidence_proj_bias, {1}});
    }
    if (Vd != V) {
        // Reduced draft vocab: the draft->target id offset map is REQUIRED
        // (vLLM d2t convention: target_id = draft_id + d2t[draft_id]).
        slots.emplace("d2t", Slot{&w.d2t, {Vd}, SafetensorsDtype::I64});
    }

    w.layers.resize(static_cast<size_t>(c.num_hidden_layers));
    for (int l = 0; l < c.num_hidden_layers; ++l) {
        auto& L = w.layers[static_cast<size_t>(l)];
        const std::string p = "layers." + std::to_string(l) + ".";
        slots.emplace(p + "self_attn.q_proj.weight", Slot{&L.q_proj, {Q, H}});
        slots.emplace(p + "self_attn.k_proj.weight", Slot{&L.k_proj, {KV, H}});
        slots.emplace(p + "self_attn.v_proj.weight", Slot{&L.v_proj, {KV, H}});
        slots.emplace(p + "self_attn.o_proj.weight", Slot{&L.o_proj, {H, Q}});
        slots.emplace(p + "self_attn.q_norm.weight", Slot{&L.q_norm, {D}});
        slots.emplace(p + "self_attn.k_norm.weight", Slot{&L.k_norm, {D}});
        slots.emplace(p + "input_layernorm.weight",
                      Slot{&L.input_layernorm, {H}});
        slots.emplace(p + "post_attention_layernorm.weight",
                      Slot{&L.post_attention_layernorm, {H}});
        slots.emplace(p + "mlp.gate_proj.weight", Slot{&L.gate_proj, {I, H}});
        slots.emplace(p + "mlp.up_proj.weight", Slot{&L.up_proj, {I, H}});
        slots.emplace(p + "mlp.down_proj.weight", Slot{&L.down_proj, {H, I}});
    }
    return slots;
}

}  // namespace

DsparkDraftWeights load_dspark_draft(const std::filesystem::path& checkpoint_dir,
                                     bool use_mmap) {
    if (dspark_checkpoint_is_gguf(checkpoint_dir))
        fail("checkpoint_path is a dflash GGUF — the V4 arm needs the TARGET "
             "weights path for the aliased embed/lm_head; call "
             "load_dspark_v4_gguf_draft(checkpoint, model.weights_path) "
             "instead (DsparkRuntime::create does)");
    DsparkDraftWeights w;
    w.ckpt = parse_dspark_checkpoint_config(checkpoint_dir);

    const auto weights_path = checkpoint_dir / kWeightsFile;
    w.shard = SafetensorsReader::open(weights_path, use_mmap);

    auto slots = build_slot_map(w.ckpt, w);

    std::vector<std::string> unmapped;   // in file, no slot
    std::vector<std::string> bad_shape;  // mapped but wrong shape/dtype
    for (const auto& entry : w.shard.entries()) {
        auto it = slots.find(entry.name);
        if (it == slots.end()) {
            unmapped.push_back(entry.name);
            continue;
        }
        Slot& slot = it->second;
        if (slot.dst->data.data() != nullptr) {
            bad_shape.push_back(entry.name + " (duplicate)");
            continue;
        }
        if (entry.dtype != slot.dtype) {
            bad_shape.push_back(entry.name + " (dtype " +
                                std::string(dtype_name(entry.dtype)) +
                                ", expected " +
                                std::string(dtype_name(slot.dtype)) + ")");
            continue;
        }
        if (entry.shape != slot.shape) {
            bad_shape.push_back(entry.name + " (shape " +
                                shape_str(entry.shape) + ", expected " +
                                shape_str(slot.shape) + ")");
            continue;
        }
        slot.dst->data = w.shard.tensor_data(entry);
        slot.dst->dtype = entry.dtype;
        slot.dst->shape = entry.shape;
        ++w.total_tensors_loaded;
        w.total_weight_bytes += static_cast<int64_t>(entry.data_size_bytes);
    }

    std::vector<std::string> missing;  // slot expected, not in file
    for (const auto& [name, slot] : slots)
        if (slot.dst->data.data() == nullptr) missing.push_back(name);

    if (!unmapped.empty() || !missing.empty() || !bad_shape.empty()) {
        std::sort(unmapped.begin(), unmapped.end());
        std::sort(missing.begin(), missing.end());
        std::sort(bad_shape.begin(), bad_shape.end());
        std::string msg = "checkpoint coverage failure (INV-DSPARK-CKPT) in " +
                          weights_path.string() + ":";
        auto append = [&msg](const char* label,
                             const std::vector<std::string>& v) {
            if (v.empty()) return;
            msg += std::string(" ") + label + " [" + std::to_string(v.size()) + "]:";
            for (const auto& n : v) msg += " " + n;
            msg += ";";
        };
        append("unmapped", unmapped);
        append("missing", missing);
        append("shape/dtype mismatch", bad_shape);
        fail(msg);
    }

    // Reduced-vocab d2t sanity (TD-DSPARK-VOCAB-REMAP): every draft id must
    // map into the embed vocab — target_id = draft_id + d2t[draft_id] in
    // [0, vocab_size).  Host-side scan of the mmap'd I64 data, fail closed.
    if (w.d2t.data.data() != nullptr) {
        const auto* off =
            reinterpret_cast<const int64_t*>(w.d2t.data.data());
        for (int64_t i = 0; i < w.ckpt.draft_vocab_size; ++i) {
            const int64_t tgt = i + off[i];
            if (tgt < 0 || tgt >= w.ckpt.vocab_size)
                fail("d2t[" + std::to_string(i) + "] maps draft id to " +
                     std::to_string(tgt) + ", outside target vocab [0, " +
                     std::to_string(w.ckpt.vocab_size) + ") in " +
                     weights_path.string());
        }
    }

    spdlog::info(
        "dspark_loader: loaded {} draft tensors ({} bytes, BF16) from {} — "
        "γ={} r={} V={} backbone={}x{} ({} aux layers)",
        w.total_tensors_loaded, w.total_weight_bytes, weights_path.string(),
        w.ckpt.block_size, w.ckpt.markov_rank, w.ckpt.draft_vocab_size,
        w.ckpt.num_hidden_layers, w.ckpt.hidden_size,
        w.ckpt.aux_hidden_state_layer_ids.size());
    return w;
}

// ── dspark_draft_bytes ──────────────────────────────────────────────────────

bool dspark_tensor_is_gemm_operand(const std::string& name) {
    if (name == "lm_head.weight" || name == "fc.weight") return true;
    // layers.<l>.self_attn.{q,k,v,o}_proj.weight and
    // layers.<l>.mlp.{gate,up,down}_proj.weight — the only "*_proj.weight"
    // names in the speculators v0.5 format (norms end in "_norm.weight" /
    // "layernorm.weight"; confidence is "confidence_head.proj.weight",
    // excluded by the layers. prefix).
    constexpr std::string_view suffix = "_proj.weight";
    return name.rfind("layers.", 0) == 0 && name.size() > suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(),
                        suffix.data()) == 0;
}

DsparkShardKind dspark_tensor_shard_kind(const std::string& name) {
    auto ends_with = [&name](std::string_view suf) {
        return name.size() >= suf.size() &&
               name.compare(name.size() - suf.size(), suf.size(),
                            suf.data()) == 0;
    };
    if (name == "lm_head.weight") return DsparkShardKind::kColParallel;
    if (name.rfind("layers.", 0) == 0) {
        if (ends_with("self_attn.o_proj.weight") ||
            ends_with("mlp.down_proj.weight"))
            return DsparkShardKind::kRowParallel;
        if (ends_with("_proj.weight")) return DsparkShardKind::kColParallel;
        return DsparkShardKind::kReplicated;  // q/k norms + layernorms
    }
    if (name == "norm.weight") return DsparkShardKind::kReplicated;
    // embed_tokens, fc, hidden_norm, markov heads, confidence head, d2t —
    // the single-homed ctx/heads pipeline (rank 0).
    return DsparkShardKind::kPrimaryOnly;
}

std::vector<int64_t> dspark_shard_shape(
        const std::string& name, const std::vector<int64_t>& shape, int rank,
        int num_ranks, config::DsparkDraftWeightsQuant quant) {
    if (num_ranks <= 1) return shape;
    if (rank < 0 || rank >= num_ranks)
        fail("dspark_shard_shape: rank " + std::to_string(rank) +
             " outside [0, " + std::to_string(num_ranks) + ")");
    const auto kind = dspark_tensor_shard_kind(name);
    if (kind == DsparkShardKind::kPrimaryOnly)
        return rank == 0 ? shape : std::vector<int64_t>{};
    if (kind == DsparkShardKind::kReplicated) return shape;
    if (shape.size() != 2)
        fail("dspark_shard_shape: sharded tensor " + name +
             " is not 2-D (" + shape_str(shape) + ")");
    auto sharded = shape;
    const size_t dim = (kind == DsparkShardKind::kColParallel) ? 0 : 1;
    if (shape[dim] % num_ranks != 0)
        fail("dspark_shard_shape: " + name + " dim " + std::to_string(dim) +
             " (" + std::to_string(shape[dim]) + ") not divisible by " +
             std::to_string(num_ranks) + " draft ranks");
    sharded[dim] = shape[dim] / num_ranks;
    // Row-parallel + quant: the shard requantizes over its K-column window
    // with groups anchored at the slice start — the slice boundary must be a
    // group multiple so shard scales coincide with the full tensor's
    // (INV-DSPARK-QUANT group-alignment clause).
    if (kind == DsparkShardKind::kRowParallel &&
        quant != config::DsparkDraftWeightsQuant::bf16 &&
        dspark_tensor_is_gemm_operand(name)) {
        const int64_t group =
            quant == config::DsparkDraftWeightsQuant::fp8_e4m3
                ? kgroup::kFp8GroupSize
                : kgroup::kNvfp4GroupSize;
        if (sharded[1] % group != 0)
            fail("dspark_shard_shape: " + name + " K shard " +
                 std::to_string(sharded[1]) +
                 " not a scale-group multiple (" + std::to_string(group) +
                 ") — quantized row-parallel shard boundaries must be "
                 "group-aligned");
    }
    return sharded;
}

namespace {

/// Device bytes (weights, scales) for one tensor under `quant`. Non-GEMM
/// tensors and the bf16 mode return the raw checkpoint size with no scales.
/// Shared by the upload layout and the budget helper (must never drift).
std::pair<int64_t, int64_t> device_tensor_bytes(
        const std::string& name, const std::vector<int64_t>& shape,
        int64_t raw_bytes, config::DsparkDraftWeightsQuant quant) {
    if (quant == config::DsparkDraftWeightsQuant::bf16 ||
        !dspark_tensor_is_gemm_operand(name) || shape.size() != 2)
        return {raw_bytes, 0};
    const int64_t n = shape[0];
    const int64_t k = shape[1];
    if (quant == config::DsparkDraftWeightsQuant::fp8_e4m3)
        return {kgroup::fp8_weight_bytes(n, k), kgroup::fp8_scale_bytes(n, k)};
    return {kgroup::nvfp4_weight_bytes(n, k),
            kgroup::nvfp4_scale_bytes(n, k)};
}

}  // namespace

int64_t dspark_draft_bytes(const std::filesystem::path& checkpoint_dir,
                           config::DsparkDraftWeightsQuant quant, int rank,
                           int num_ranks) {
    if (dspark_checkpoint_is_gguf(checkpoint_dir)) {
        // Ticket J: the dflash GGUF is pre-quantized (Q8_0 + MXFP4) and
        // single-rank; requant/sharding do not apply.
        if (quant != config::DsparkDraftWeightsQuant::bf16)
            fail("draft_weights_quant must be bf16 for a dflash GGUF "
                 "checkpoint (the artifact is already Q8_0/MXFP4-quantized)");
        if (rank != 0 || num_ranks != 1)
            fail("the V4 dflash draft is single-rank only — configure exactly "
                 "one speculation.dspark.draft_gpus entry");
        return dspark_gguf_draft_bytes(checkpoint_dir);
    }
    const auto weights_path = checkpoint_dir / kWeightsFile;
    int64_t total = 0;
    for (const auto& entry : SafetensorsReader::read_header(weights_path)) {
        const auto shape = dspark_shard_shape(entry.name, entry.shape, rank,
                                              num_ranks, quant);
        if (shape.empty() && !entry.shape.empty())
            continue;  // kPrimaryOnly, rank > 0 — not resident here
        // Shard raw bytes scale with the element count (uniform dtype).
        int64_t raw = static_cast<int64_t>(entry.data_size_bytes);
        int64_t full_elems = 1, shard_elems = 1;
        for (int64_t s : entry.shape) full_elems *= s;
        for (int64_t s : shape) shard_elems *= s;
        if (full_elems > 0 && shard_elems != full_elems)
            raw = raw / full_elems * shard_elems;
        const auto [wb, sb] =
            device_tensor_bytes(entry.name, shape, raw, quant);
        total += wb + sb;
    }
    if (total <= 0)
        fail("no tensors in " + weights_path.string());
    return total;
}

// ── resolve_dspark_draft_gpu ────────────────────────────────────────────────

std::vector<int> resolve_dspark_draft_gpus(const config::Config& cfg) {
    const auto& hw = cfg.hardware;
    const auto& draft_gpus = cfg.speculation.dspark.draft_gpus;
    const int n_gpus = static_cast<int>(hw.gpus.size());
    if (!draft_gpus.empty()) {
        std::vector<int> out;
        out.reserve(draft_gpus.size());
        for (size_t i = 0; i < draft_gpus.size(); ++i) {
            const int pos = draft_gpus[i];
            if (pos < 0 || pos >= n_gpus)
                fail("speculation.dspark.draft_gpus[" + std::to_string(i) +
                     "] = " + std::to_string(pos) + " out of range [0, " +
                     std::to_string(n_gpus) + ")");
            if (std::find(out.begin(), out.end(), pos) != out.end())
                fail("speculation.dspark.draft_gpus has duplicate position " +
                     std::to_string(pos));
            out.push_back(pos);
        }
        return out;
    }
    return {resolve_dspark_draft_gpu(cfg)};
}

int resolve_dspark_draft_gpu(const config::Config& cfg) {
    const auto& hw = cfg.hardware;
    const auto& draft_gpus = cfg.speculation.dspark.draft_gpus;
    const int n_gpus = static_cast<int>(hw.gpus.size());

    if (!draft_gpus.empty()) {
        const int pos = draft_gpus[0];
        if (pos < 0 || pos >= n_gpus)
            fail("speculation.dspark.draft_gpus[0] = " + std::to_string(pos) +
                 " out of range [0, " + std::to_string(n_gpus) + ")");
        return pos;
    }

    // Auto: first GPU NOT in the TP set (secondary GPUs are idle while the
    // TP GPUs run the target).
    std::vector<bool> in_tp(hw.gpus.size(), false);
    for (int idx : hw.tp_array)
        if (idx >= 0 && idx < n_gpus) in_tp[static_cast<size_t>(idx)] = true;
    for (int i = 0; i < n_gpus; ++i)
        if (!in_tp[static_cast<size_t>(i)]) return i;

    fail("no non-TP GPU available for the DSpark draft — set "
         "speculation.dspark.draft_gpus explicitly (all " +
         std::to_string(n_gpus) + " GPUs are in hardware.tp_array)");
}

// ── validate_dspark_config_against_checkpoint ───────────────────────────────

void validate_dspark_config_against_checkpoint(
        const config::DsparkConfig& d, const DsparkCheckpointConfig& c) {
    std::vector<std::string> errs;
    auto check_eq = [&errs](const char* field, auto cfg_v, auto ckpt_v) {
        if (static_cast<int64_t>(cfg_v) != static_cast<int64_t>(ckpt_v))
            errs.push_back(std::string(field) + ": config " +
                           std::to_string(cfg_v) + " != checkpoint " +
                           std::to_string(ckpt_v));
    };
    check_eq("block_size", d.block_size, c.block_size);
    check_eq("markov_rank", d.markov_rank, c.markov_rank);
    check_eq("draft_vocab_size", d.draft_vocab_size, c.draft_vocab_size);
    check_eq("mask_token_id", d.mask_token_id, c.mask_token_id);
    // The dflash GGUF carries no max_anchors key — the config knob is
    // ignored for the V4 arm (drafting is one anchor per RUN_DSPARK_STEP).
    if (!c.is_v4_dflash)
        check_eq("max_anchors", d.max_anchors, c.max_anchors);
    check_eq("speculative_tokens", d.speculative_tokens, c.speculative_tokens);

    if (std::vector<int>(d.aux_hidden_state_layer_ids.begin(),
                         d.aux_hidden_state_layer_ids.end()) !=
        c.aux_hidden_state_layer_ids) {
        errs.push_back("aux_hidden_state_layer_ids: config != checkpoint");
    }

    // head_type markov ↔ checkpoint "vanilla" (gated/rnn map by name).
    const char* want_ckpt_head = nullptr;
    switch (d.head_type) {
    case config::DsparkHeadType::markov: want_ckpt_head = "vanilla"; break;
    case config::DsparkHeadType::gated:  want_ckpt_head = "gated";   break;
    case config::DsparkHeadType::rnn:    want_ckpt_head = "rnn";     break;
    }
    if (want_ckpt_head && c.markov_head_type != want_ckpt_head)
        errs.push_back("head_type: config expects checkpoint markov_head_type '" +
                       std::string(want_ckpt_head) + "', checkpoint has '" +
                       c.markov_head_type + "'");

    if (d.confidence_enabled && !c.enable_confidence_head)
        errs.push_back("confidence_enabled=true but the checkpoint has no "
                       "confidence head (enable_confidence_head=false)");

    if (!errs.empty()) {
        std::string msg = "speculation.dspark config does not match the "
                          "checkpoint:";
        for (const auto& e : errs) msg += " " + e + ";";
        fail(msg);
    }
}

// ── Device placement ────────────────────────────────────────────────────────

DsparkDeviceWeights::~DsparkDeviceWeights() {
    if (backend && arena && owns_arena) backend->device_free(arena);
}

DsparkDeviceWeights::DsparkDeviceWeights(DsparkDeviceWeights&& other) noexcept {
    *this = std::move(other);
}

DsparkDeviceWeights& DsparkDeviceWeights::operator=(
        DsparkDeviceWeights&& other) noexcept {
    if (this == &other) return *this;
    if (backend && arena && owns_arena) backend->device_free(arena);
    backend = other.backend;
    arena = other.arena;
    arena_bytes = other.arena_bytes;
    owns_arena = other.owns_arena;
    embed_tokens = std::move(other.embed_tokens);
    lm_head = std::move(other.lm_head);
    fc = std::move(other.fc);
    hidden_norm = std::move(other.hidden_norm);
    final_norm = std::move(other.final_norm);
    markov_w1 = std::move(other.markov_w1);
    markov_w2 = std::move(other.markov_w2);
    confidence_proj_weight = std::move(other.confidence_proj_weight);
    confidence_proj_bias = std::move(other.confidence_proj_bias);
    d2t = std::move(other.d2t);
    layers = std::move(other.layers);
    v4 = std::move(other.v4);
    total_tensors_uploaded = other.total_tensors_uploaded;
    other.backend = nullptr;
    other.arena = nullptr;
    other.arena_bytes = 0;
    other.owns_arena = true;
    other.total_tensors_uploaded = 0;
    return *this;
}

DsparkDeviceWeights upload_dspark_draft(const DsparkDraftWeights& weights,
                                        compute::DeviceBackend& backend,
                                        void* arena, int64_t arena_capacity,
                                        config::DsparkDraftWeightsQuant quant,
                                        int rank, int num_ranks) {
    // Collect (host tensor, device slot) pairs in a fixed order. `name`
    // drives the requant classification (dspark_tensor_is_gemm_operand) and
    // the shard classification (dspark_tensor_shard_kind).
    struct Slot {
        const RawTensor* src;
        DsparkDeviceTensor* dst;
        std::string name;
    };
    if (num_ranks < 1 || rank < 0 || rank >= num_ranks)
        fail("upload_dspark_draft: bad rank " + std::to_string(rank) + "/" +
             std::to_string(num_ranks));
    if (weights.ckpt.is_v4_dflash) {
        // Ticket J dispatch: the V4 dflash upload (single-rank, native quant).
        if (quant != config::DsparkDraftWeightsQuant::bf16)
            fail("draft_weights_quant must be bf16 for a dflash GGUF "
                 "checkpoint");
        if (num_ranks != 1)
            fail("the V4 dflash draft is single-rank only");
        return upload_dspark_v4_gguf_draft(weights, backend, arena,
                                           arena_capacity);
    }
    std::vector<Slot> pairs;
    DsparkDeviceWeights out;
    out.layers.resize(weights.layers.size());

    auto add = [&pairs](const RawTensor& src, DsparkDeviceTensor& dst,
                        std::string name) {
        if (src.data.data() == nullptr) return;  // disabled optional head
        pairs.push_back({&src, &dst, std::move(name)});
    };
    add(weights.embed_tokens, out.embed_tokens, "embed_tokens.weight");
    add(weights.lm_head, out.lm_head, "lm_head.weight");
    add(weights.fc, out.fc, "fc.weight");
    add(weights.hidden_norm, out.hidden_norm, "hidden_norm.weight");
    add(weights.final_norm, out.final_norm, "norm.weight");
    add(weights.markov_w1, out.markov_w1, "markov_head.markov_w1.weight");
    add(weights.markov_w2, out.markov_w2, "markov_head.markov_w2.weight");
    add(weights.confidence_proj_weight, out.confidence_proj_weight,
        "confidence_head.proj.weight");
    add(weights.confidence_proj_bias, out.confidence_proj_bias,
        "confidence_head.proj.bias");
    add(weights.d2t, out.d2t, "d2t");  // I64; byte-copied like every tensor
    for (size_t l = 0; l < weights.layers.size(); ++l) {
        const auto& src = weights.layers[l];
        auto& dst = out.layers[l];
        const std::string p = "layers." + std::to_string(l) + ".";
        add(src.q_proj, dst.q_proj, p + "self_attn.q_proj.weight");
        add(src.k_proj, dst.k_proj, p + "self_attn.k_proj.weight");
        add(src.v_proj, dst.v_proj, p + "self_attn.v_proj.weight");
        add(src.o_proj, dst.o_proj, p + "self_attn.o_proj.weight");
        add(src.q_norm, dst.q_norm, p + "self_attn.q_norm.weight");
        add(src.k_norm, dst.k_norm, p + "self_attn.k_norm.weight");
        add(src.input_layernorm, dst.input_layernorm,
            p + "input_layernorm.weight");
        add(src.post_attention_layernorm, dst.post_attention_layernorm,
            p + "post_attention_layernorm.weight");
        add(src.gate_proj, dst.gate_proj, p + "mlp.gate_proj.weight");
        add(src.up_proj, dst.up_proj, p + "mlp.up_proj.weight");
        add(src.down_proj, dst.down_proj, p + "mlp.down_proj.weight");
    }

    // Shard views (TD-DSPARK-DRAFT-SHARD): per pair, the logical shard shape
    // + the effective host source. Col-parallel shards are contiguous row
    // slices of the checkpoint mmap; row-parallel shards are host-repacked
    // K-column windows (persisted per tensor until its H2D lands — the
    // synchronous memcpy_h2d below makes a single reusable buffer safe, but
    // repack_bufs keeps this correct even if the copies turn async later).
    struct ShardView {
        bool skip = false;
        const std::byte* src = nullptr;
        int64_t src_bytes = 0;
        std::vector<int64_t> shape;
    };
    std::vector<ShardView> views(pairs.size());
    std::vector<std::vector<std::byte>> repack_bufs;
    for (size_t i = 0; i < pairs.size(); ++i) {
        const auto& p = pairs[i];
        auto& v = views[i];
        v.shape = dspark_shard_shape(p.name, p.src->shape, rank, num_ranks,
                                     quant);
        if (v.shape.empty() && !p.src->shape.empty()) {
            v.skip = true;  // kPrimaryOnly on rank > 0
            continue;
        }
        if (num_ranks == 1 || v.shape == p.src->shape) {
            v.src = p.src->data.data();
            v.src_bytes = static_cast<int64_t>(p.src->data.size());
            continue;
        }
        const auto kind = dspark_tensor_shard_kind(p.name);
        const int64_t n = p.src->shape[0];
        const int64_t k = p.src->shape[1];
        const int64_t elem = static_cast<int64_t>(p.src->data.size()) /
                             (n * k);  // BF16 = 2 for every sharded tensor
        if (kind == DsparkShardKind::kColParallel) {
            const int64_t n_local = v.shape[0];
            v.src = p.src->data.data() +
                    static_cast<int64_t>(rank) * n_local * k * elem;
            v.src_bytes = n_local * k * elem;
        } else {  // kRowParallel: strided K-window — repack host-side
            const int64_t k_local = v.shape[1];
            auto& buf = repack_bufs.emplace_back();
            buf.resize(static_cast<size_t>(n * k_local * elem));
            const auto* src_base = p.src->data.data() +
                                   static_cast<int64_t>(rank) * k_local * elem;
            for (int64_t row = 0; row < n; ++row)
                std::memcpy(buf.data() + row * k_local * elem,
                            src_base + row * k * elem,
                            static_cast<size_t>(k_local * elem));
            v.src = buf.data();
            v.src_bytes = n * k_local * elem;
        }
    }

    // Lay out the arena: 256-byte aligned offsets, per-tensor weight bytes
    // then (quant path) scale bytes as a second aligned entry
    // (TD-DSPARK-DRAFT-QUANT). device_tensor_bytes over the SHARD shape is
    // the same sizing the LayerRegistry budget uses (dspark_draft_bytes) —
    // they cannot drift.
    int64_t offset = 0;
    std::vector<int64_t> w_offsets, s_offsets, w_bytes, s_bytes;
    w_offsets.reserve(pairs.size());
    s_offsets.reserve(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (views[i].skip) {
            w_offsets.push_back(-1);
            w_bytes.push_back(0);
            s_offsets.push_back(-1);
            s_bytes.push_back(0);
            continue;
        }
        const auto [wb, sb] = device_tensor_bytes(
            pairs[i].name, views[i].shape, views[i].src_bytes, quant);
        w_offsets.push_back(offset);
        w_bytes.push_back(wb);
        offset += align_up(wb, kArenaAlign);
        s_offsets.push_back(sb > 0 ? offset : -1);
        s_bytes.push_back(sb);
        if (sb > 0) offset += align_up(sb, kArenaAlign);
    }
    out.arena_bytes = offset;

    backend.set_device();
    if (arena) {
        // DSP-3: caller-provided region (the VramAllocator pinned region the
        // LayerRegistry budget already carved on the draft GPU). No second
        // device_alloc — that would double-book the VRAM.
        if (arena_capacity < out.arena_bytes)
            fail("provided draft arena too small: " +
                 std::to_string(arena_capacity) + " < " +
                 std::to_string(out.arena_bytes) + " bytes");
        out.arena = arena;
        out.owns_arena = false;
    } else {
        out.arena = backend.device_alloc(static_cast<size_t>(out.arena_bytes));
        if (!out.arena)
            fail("device_alloc of " + std::to_string(out.arena_bytes) +
                 " bytes failed on GPU position " +
                 std::to_string(backend.gpu().position) +
                 " (DSpark draft arena)");
        out.owns_arena = true;
    }
    out.backend = &backend;

    auto* base = static_cast<std::byte*>(out.arena);
    int requantized = 0;
    std::vector<uint8_t> qbuf;   // reused host staging (largest: lm_head)
    std::vector<uint8_t> sbuf;
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (views[i].skip) continue;  // kPrimaryOnly, rank > 0
        const auto& p = pairs[i];
        auto* dst = p.dst;
        dst->ptr = base + w_offsets[i];
        dst->bytes = w_bytes[i];
        dst->shape = views[i].shape;
        if (s_bytes[i] == 0) {
            // BF16 verbatim (or non-GEMM tensor): byte-identical legacy path.
            dst->dtype = DsparkWeightDtype::kBF16;
            backend.memcpy_h2d(dst->ptr, views[i].src,
                               static_cast<size_t>(views[i].src_bytes));
        } else {
            // Requant at upload (TD-DSPARK-DRAFT-QUANT): host-side K-group
            // packing (kgroup_quant), then H2D of weights + scales.
            const int64_t n = views[i].shape[0];
            const int64_t k = views[i].shape[1];
            const auto* src_bf16 =
                reinterpret_cast<const uint16_t*>(views[i].src);
            qbuf.resize(static_cast<size_t>(w_bytes[i]));
            sbuf.resize(static_cast<size_t>(s_bytes[i]));
            if (quant == config::DsparkDraftWeightsQuant::fp8_e4m3) {
                dst->dtype = DsparkWeightDtype::kFp8E4M3;
                dst->k_groups = (k + kgroup::kFp8GroupSize - 1) /
                                kgroup::kFp8GroupSize;
                kgroup::quantize_rows_fp8_e4m3(
                    src_bf16, n, k, qbuf.data(),
                    reinterpret_cast<float*>(sbuf.data()));
            } else {
                dst->dtype = DsparkWeightDtype::kNvfp4;
                dst->k_groups = (k + kgroup::kNvfp4GroupSize - 1) /
                                kgroup::kNvfp4GroupSize;
                kgroup::quantize_rows_nvfp4(src_bf16, n, k, qbuf.data(),
                                            sbuf.data());
            }
            dst->scales = base + s_offsets[i];
            dst->scales_bytes = s_bytes[i];
            backend.memcpy_h2d(dst->ptr, qbuf.data(), qbuf.size());
            backend.memcpy_h2d(dst->scales, sbuf.data(), sbuf.size());
            ++requantized;
        }
        ++out.total_tensors_uploaded;
    }
    backend.device_sync();

    spdlog::info(
        "dspark_loader: uploaded {} draft tensors{} ({} arena bytes, {}{}) "
        "onto GPU position {} (cuda id {})",
        out.total_tensors_uploaded,
        num_ranks > 1 ? " [shard rank " + std::to_string(rank) + "/" +
                            std::to_string(num_ranks) + "]"
                      : std::string{},
        out.arena_bytes,
        quant == config::DsparkDraftWeightsQuant::bf16
            ? "BF16"
            : (quant == config::DsparkDraftWeightsQuant::fp8_e4m3
                   ? "FP8-E4M3 GEMM operands"
                   : "NVFP4 GEMM operands"),
        requantized > 0 ? " (" + std::to_string(requantized) +
                              " tensors requantized at upload)"
                        : std::string{},
        backend.gpu().position, backend.gpu().id);
    return out;
}

}  // namespace layerstorm::model

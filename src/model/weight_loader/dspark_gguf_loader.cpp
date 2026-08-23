// DSpark V4 dflash GGUF checkpoint loader (ticket J).
//
// Loads the DeepSeek-V4-Flash dspark speculator GGUF
// (dspark-DeepSeek-V4-Flash-*.gguf, `general.architecture = "dflash"`):
// 3 full V4-shaped SWA-only decoder blocks (latent-MQA attention + sinks +
// grouped o_proj + per-block mHC + MoE: 256 MXFP4 routed experts + Q8_0
// shared expert), model-level fc [H, 3H] target-feature fusion +
// enc.output_norm, ONE vanilla markov head (markov_w1/w2 [V, 256] BF16), ONE
// confidence head (conf_proj [1, H+r] BF16, NO bias), the draft's own
// output_hc_* hc-head + output_norm, and NO embed/lm_head (aliased from the
// TARGET GGUF's token_embd/output — vLLM DSparkDeepseekV4ForCausalLM
// has_own_embed_tokens/lm_head = False).
//
// Semantics references: ref/vllm/vllm/models/deepseek_v4/nvidia/dspark.py
// (draft model + context-KV precompute + weight map),
// ref/vllm/vllm/models/deepseek_v4/nvidia/model.py:1082-1110 (aux capture =
// mhc_post residual .mean over hc streams per target layer),
// ref/llama.cpp/common/speculative.cpp:957 (block_size semantics: query =
// [id_last, <mask> x (block_size-1)]).
//
// Load-time conversions (host): Q8_0 -> BF16 dequant (attention/shexp/fc),
// F32 norms -> BF16 (the BF16 rmsnorm kernel path, matching the V4 target's
// norm upload convention), MXFP4 routed experts packed VERBATIM (consumed
// natively by the GGUF grouped int GEMM), F32 heads (hc/sinks/exp_probs_b/
// output_hc) verbatim. After conversion host bytes == device bytes for every
// tensor, so upload is a plain per-tensor byte copy.

#include "model/weight_loader/dspark_loader.h"

#include "core/bf16_convert.h"
#include "core/device_backend.h"
#include "model/quantization/gguf_kquant.h"
#include "model/weight_loader/weight_loader.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace layerstorm::model {

namespace {

constexpr int64_t kArenaAlign = 256;

int64_t align_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

[[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error("dspark_gguf_loader: " + what);
}

/// Logical row-major shape from a GGUF entry (ne order reversed).
std::vector<int64_t> logical_shape(const GgufTensorEntry& e) {
    std::vector<int64_t> s(e.dims.rbegin(), e.dims.rend());
    return s;
}

int64_t numel(const GgufTensorEntry& e) {
    int64_t n = 1;
    for (int64_t d : e.dims) n *= d;
    return n;
}

int64_t md_i64(const GgufReader& g, const char* key) {
    auto v = g.metadata(key);
    if (!v) fail(g.path().string() + ": missing metadata key " + key);
    switch (v->kind) {
    case GgufMetadataValue::Kind::u64: return static_cast<int64_t>(v->u);
    case GgufMetadataValue::Kind::i64: return v->i;
    default: fail(g.path().string() + ": metadata key " + key +
                  " is not an integer");
    }
}

double md_f64(const GgufReader& g, const char* key) {
    auto v = g.metadata(key);
    if (!v) fail(g.path().string() + ": missing metadata key " + key);
    if (v->kind == GgufMetadataValue::Kind::f64) return v->f;
    if (v->kind == GgufMetadataValue::Kind::u64)
        return static_cast<double>(v->u);
    if (v->kind == GgufMetadataValue::Kind::i64)
        return static_cast<double>(v->i);
    fail(g.path().string() + ": metadata key " + std::string(key) +
         " is not numeric");
}

bool md_bool(const GgufReader& g, const char* key, bool dflt) {
    auto v = g.metadata(key);
    if (!v) return dflt;
    if (v->kind == GgufMetadataValue::Kind::boolean) return v->b;
    return dflt;
}

const GgufTensorEntry* find_entry(const GgufReader& g,
                                  const std::string& name) {
    for (const auto& e : g.entries())
        if (e.name == name) return &e;
    return nullptr;
}

/// Device bytes one draft tensor occupies after load-time conversion.
/// Single home for the sizing rule — dspark_gguf_draft_bytes and the
/// upload layout both use it (they cannot drift).
int64_t v4_device_bytes(const GgufTensorEntry& e, bool to_bf16_norm) {
    const auto t = static_cast<GgmlType>(e.ggml_type);
    switch (t) {
    case GgmlType::MXFP4:
        return static_cast<int64_t>(e.data_size_bytes);  // packed verbatim
    case GgmlType::Q8_0:
        return numel(e) * 2;  // dequant -> BF16
    case GgmlType::F32:
        return to_bf16_norm ? numel(e) * 2
                            : static_cast<int64_t>(e.data_size_bytes);
    case GgmlType::BF16:
        return static_cast<int64_t>(e.data_size_bytes);
    default:
        fail("tensor " + e.name + " has unsupported ggml type " +
             gguf_ggml_type_name(e.ggml_type) +
             " for the dflash draft (expected MXFP4/Q8_0/F32/BF16)");
    }
}

/// True for the F32 tensors the loader converts to BF16 (rmsnorm weights).
bool is_bf16_converted_norm(const std::string& name) {
    auto ends_with = [&name](const char* suf) {
        const size_t n = std::strlen(suf);
        return name.size() >= n &&
               name.compare(name.size() - n, n, suf) == 0;
    };
    if (name == "output_norm.weight" || name == "enc.output_norm.weight")
        return true;
    return ends_with(".attn_norm.weight") || ends_with(".ffn_norm.weight") ||
           ends_with(".attn_q_a_norm.weight") ||
           ends_with(".attn_kv_a_norm.weight");
}

/// The fixed tensor-name enumeration of the dflash draft (model-level then
/// per-layer), shared by the budget and the load coverage check.
std::vector<std::string> v4_expected_names(int num_layers,
                                           bool has_confidence) {
    std::vector<std::string> names = {
        "fc.weight",
        "enc.output_norm.weight",
        "output_norm.weight",
        "output_hc_fn.weight",
        "output_hc_base.weight",
        "output_hc_scale.weight",
        "markov_w1.weight",
        "markov_w2.weight",
    };
    if (has_confidence) names.push_back("conf_proj.weight");
    for (int l = 0; l < num_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        for (const char* s :
             {"attn_norm.weight", "ffn_norm.weight", "attn_q_a.weight",
              "attn_q_a_norm.weight", "attn_q_b.weight", "attn_kv.weight",
              "attn_kv_a_norm.weight", "attn_output_a.weight",
              "attn_output_b.weight", "attn_sinks.weight",
              "hc_attn_fn.weight", "hc_attn_base.weight",
              "hc_attn_scale.weight", "hc_ffn_fn.weight",
              "hc_ffn_base.weight", "hc_ffn_scale.weight",
              "ffn_gate_inp.weight", "exp_probs_b.bias",
              "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
              "ffn_down_shexp.weight", "ffn_gate_exps.weight",
              "ffn_up_exps.weight", "ffn_down_exps.weight"})
            names.push_back(p + s);
    }
    return names;
}

}  // namespace

// ── dspark_checkpoint_is_gguf ───────────────────────────────────────────────

bool dspark_checkpoint_is_gguf(const std::filesystem::path& checkpoint_path) {
    const auto ext = checkpoint_path.extension().string();
    return ext == ".gguf";
}

// ── parse_dspark_gguf_checkpoint_config ─────────────────────────────────────

DsparkCheckpointConfig parse_dspark_gguf_checkpoint_config(
        const std::filesystem::path& gguf_file) {
    auto g = GgufReader::open(gguf_file);
    const std::string src = gguf_file.string();

    auto arch = g.metadata("general.architecture");
    if (!arch || arch->kind != GgufMetadataValue::Kind::str ||
        arch->s != "dflash")
        fail(src + ": general.architecture is not 'dflash' (a dspark "
                   "speculator GGUF)");

    DsparkCheckpointConfig c;
    c.is_v4_dflash = true;
    c.model_type = "dflash";
    c.markov_head_type = "vanilla";
    c.tie_word_embeddings = false;
    c.max_anchors = 0;  // no such key in the GGUF; ignored for this arm

    c.num_hidden_layers = static_cast<int>(md_i64(g, "dflash.block_count"));
    c.hidden_size = md_i64(g, "dflash.embedding_length");
    c.num_attention_heads =
        static_cast<int>(md_i64(g, "dflash.attention.head_count"));
    c.num_key_value_heads =
        static_cast<int>(md_i64(g, "dflash.attention.head_count_kv"));
    c.head_dim = md_i64(g, "dflash.attention.key_length");
    if (md_i64(g, "dflash.attention.value_length") != c.head_dim)
        fail(src + ": attention value_length != key_length (V==K latent "
                   "expected)");
    if (c.num_key_value_heads != 1)
        fail(src + ": head_count_kv != 1 (V4 native latent MQA expected)");
    c.intermediate_size = md_i64(g, "dflash.expert_feed_forward_length");
    c.rms_norm_eps = md_f64(g, "dflash.attention.layer_norm_rms_epsilon");
    c.rope_theta = md_f64(g, "dflash.rope.freq_base");

    // Draft rope decision (recorded in the dossier): the draft blocks are
    // trained as extra V4 layers at compress-ratio 0 => BASE theta, NO yarn
    // (the GGUF's rope.scaling block is a wholesale copy of the target's;
    // ratio-0 layers never apply it — dual-rope rule, ticket D).
    {
        auto ratios = g.metadata_array_i64("dflash.attention.compress_ratios");
        if (!ratios || static_cast<int>(ratios->size()) < c.num_hidden_layers)
            fail(src + ": missing/short dflash.attention.compress_ratios");
        for (int l = 0; l < c.num_hidden_layers; ++l)
            if ((*ratios)[static_cast<size_t>(l)] != 0)
                fail(src + ": compress_ratios[" + std::to_string(l) +
                     "] != 0 — only SWA-only dflash blocks are supported");
    }
    if (md_i64(g, "dflash.hash_layer_count") != 0)
        fail(src + ": hash_layer_count != 0 unsupported for the draft");
    if (md_i64(g, "dflash.expert_gating_func") != 4)
        fail(src + ": expert_gating_func != 4 (sqrtsoftplus) unsupported");
    if (md_i64(g, "dflash.expert_shared_count") != 1)
        fail(src + ": expert_shared_count != 1 unsupported");

    auto& v4 = c.v4;
    v4.q_lora_rank = md_i64(g, "dflash.attention.q_lora_rank");
    v4.o_groups = md_i64(g, "dflash.attention.output_group_count");
    v4.o_lora_rank = md_i64(g, "dflash.attention.output_lora_rank");
    v4.rope_dim = md_i64(g, "dflash.rope.dimension_count");
    v4.sliding_window = md_i64(g, "dflash.attention.sliding_window");
    v4.n_routed_experts = md_i64(g, "dflash.expert_count");
    v4.n_expert_used = md_i64(g, "dflash.expert_used_count");
    v4.moe_intermediate = md_i64(g, "dflash.expert_feed_forward_length");
    v4.routed_scaling = md_f64(g, "dflash.expert_weights_scale");
    v4.norm_topk_prob = md_bool(g, "dflash.expert_weights_norm", true);
    v4.hc_mult =
        static_cast<int>(md_i64(g, "dflash.hyper_connection.count"));
    v4.hc_sinkhorn_iters = static_cast<int>(
        md_i64(g, "dflash.hyper_connection.sinkhorn_iterations"));
    v4.hc_eps = md_f64(g, "dflash.hyper_connection.epsilon");
    {
        // swiglu clamp: per-layer arrays, expected uniform across layers AND
        // across routed/shared (the shipped artifact ships [10.0] x L twice).
        auto ce = g.metadata_array_f64("dflash.swiglu_clamp_exp");
        auto cs = g.metadata_array_f64("dflash.swiglu_clamp_shexp");
        if (!ce || ce->empty() || !cs || cs->empty())
            fail(src + ": missing dflash.swiglu_clamp_exp/shexp arrays");
        v4.swiglu_limit = (*ce)[0];
        for (double x : *ce)
            if (x != v4.swiglu_limit)
                fail(src + ": non-uniform swiglu_clamp_exp unsupported");
        for (double x : *cs)
            if (x != v4.swiglu_limit)
                fail(src + ": swiglu_clamp_shexp differs from clamp_exp — "
                           "unsupported");
    }

    // Speculator fields.
    c.block_size = static_cast<int>(md_i64(g, "dflash.block_size"));
    // llama.cpp common/speculative.cpp:957 — query = [anchor, mask x
    // (block_size-1)]; our runtime's bonus-anchor layout is identical.
    c.speculative_tokens = c.block_size - 1;
    c.mask_token_id = md_i64(g, "tokenizer.ggml.mask_token_id");

    auto layers = g.metadata_array_i64("dflash.target_layers");
    if (!layers || layers->empty())
        fail(src + ": missing dflash.target_layers");
    c.aux_hidden_state_layer_ids.assign(layers->begin(), layers->end());

    // Markov / confidence anatomy from the tensor infos (header-only).
    const auto* w1 = find_entry(g, "markov_w1.weight");
    const auto* w2 = find_entry(g, "markov_w2.weight");
    if (!w1 || !w2 || w1->dims.size() != 2 || w2->dims.size() != 2)
        fail(src + ": missing/malformed markov_w1/markov_w2 tensors");
    c.markov_rank = static_cast<int>(w1->dims[0]);  // ne0 = r
    c.vocab_size = w1->dims[1];                     // ne1 = V
    c.draft_vocab_size = w2->dims[1];               // full-vocab draft
    if (c.markov_rank != static_cast<int>(w2->dims[0]))
        fail(src + ": markov_w1/markov_w2 rank mismatch");
    const auto* conf = find_entry(g, "conf_proj.weight");
    c.enable_confidence_head = conf != nullptr;
    c.confidence_has_bias = false;  // no bias tensor in the GGUF format
    if (conf) {
        const int64_t conf_in = conf->dims[0];  // ne0 = input width
        if (conf_in == c.hidden_size + c.markov_rank)
            c.confidence_head_with_markov = true;
        else if (conf_in == c.hidden_size)
            c.confidence_head_with_markov = false;
        else
            fail(src + ": conf_proj input width " + std::to_string(conf_in) +
                 " matches neither H nor H+r");
    }

    if (c.block_size <= 0 || c.block_size > 16)
        fail(src + ": block_size " + std::to_string(c.block_size) +
             " outside (0, 16]");
    if (c.hidden_size <= 0 || c.num_hidden_layers <= 0 ||
        c.markov_rank <= 0 || c.vocab_size <= 0)
        fail(src + ": non-positive derived dimension");
    if (c.mask_token_id < 0 || c.mask_token_id >= c.vocab_size)
        fail(src + ": mask_token_id outside vocab");
    if (c.draft_vocab_size != c.vocab_size)
        fail(src + ": reduced-vocab dflash drafts unsupported (no d2t in "
                   "the GGUF format)");
    return c;
}

// ── dspark_gguf_draft_bytes ─────────────────────────────────────────────────

int64_t dspark_gguf_draft_bytes(const std::filesystem::path& gguf_file) {
    auto g = GgufReader::open(gguf_file);
    const auto c = parse_dspark_gguf_checkpoint_config(gguf_file);

    int64_t total = 0;
    for (const auto& name :
         v4_expected_names(c.num_hidden_layers, c.enable_confidence_head)) {
        const auto* e = find_entry(g, name);
        if (!e) fail(g.path().string() + ": missing tensor " + name);
        total += align_up(v4_device_bytes(*e, is_bf16_converted_norm(name)),
                          kArenaAlign);
    }
    // Target-aliased embed + lm_head (BF16 [V, H] each).
    const int64_t vh = c.vocab_size * c.hidden_size * 2;
    total += align_up(vh, kArenaAlign) * 2;
    // Per-layer expert B_ptrs device arrays (gate/up/down, [E] void* each).
    total += static_cast<int64_t>(c.num_hidden_layers) * 3 *
             align_up(c.v4.n_routed_experts *
                          static_cast<int64_t>(sizeof(void*)),
                      kArenaAlign);
    return total;
}

// ── load_dspark_v4_gguf_draft ───────────────────────────────────────────────

namespace {

/// Build a converted host tensor: dequant Q8_0 -> BF16, convert F32 norms ->
/// BF16, pass everything else through as an mmap view. Owned buffers land in
/// `owned` (inner heap blocks are pointer-stable across outer growth).
RawTensor make_host_tensor(const GgufReader& g, const GgufTensorEntry& e,
                           bool to_bf16_norm,
                           std::vector<std::vector<std::byte>>& owned) {
    RawTensor t;
    t.shape = logical_shape(e);
    const auto type = static_cast<GgmlType>(e.ggml_type);
    const auto raw = g.tensor_data(e);
    switch (type) {
    case GgmlType::MXFP4: {
        t.data = raw;
        t.dtype = SafetensorsDtype::U8;
        t.gguf_type = GgufKQuantType::MXFP4;
        return t;
    }
    case GgmlType::Q8_0: {
        RawTensor q;
        q.data = raw;
        q.dtype = SafetensorsDtype::U8;
        q.shape = t.shape;
        q.gguf_type = GgufKQuantType::Q8_0;
        const int64_t n = numel(e);
        auto bf16 = dequant_kquant_range_to_bf16(q, 0, n);
        auto& buf = owned.emplace_back();
        buf.resize(static_cast<size_t>(n) * 2);
        std::memcpy(buf.data(), bf16.data(), buf.size());
        t.data = {buf.data(), buf.size()};
        t.dtype = SafetensorsDtype::BF16;
        return t;
    }
    case GgmlType::F32: {
        if (to_bf16_norm) {
            const int64_t n = numel(e);
            auto bf16 = layerstorm::f32_to_bf16(raw.data(),
                                                static_cast<size_t>(n));
            auto& buf = owned.emplace_back();
            buf.resize(static_cast<size_t>(n) * 2);
            std::memcpy(buf.data(), bf16.data(), buf.size());
            t.data = {buf.data(), buf.size()};
            t.dtype = SafetensorsDtype::BF16;
        } else {
            t.data = raw;
            t.dtype = SafetensorsDtype::F32;
        }
        return t;
    }
    case GgmlType::BF16: {
        t.data = raw;
        t.dtype = SafetensorsDtype::BF16;
        return t;
    }
    default:
        fail("tensor " + e.name + " has unsupported ggml type " +
             gguf_ggml_type_name(e.ggml_type));
    }
}

void expect_shape(const RawTensor& t, const std::string& name,
                  std::vector<int64_t> want) {
    if (t.shape != want) {
        std::string got = "[", exp = "[";
        for (size_t i = 0; i < t.shape.size(); ++i)
            got += (i ? "," : "") + std::to_string(t.shape[i]);
        for (size_t i = 0; i < want.size(); ++i)
            exp += (i ? "," : "") + std::to_string(want[i]);
        fail("tensor " + name + " shape " + got + "], expected " + exp + "]");
    }
}

}  // namespace

DsparkDraftWeights load_dspark_v4_gguf_draft(
        const std::filesystem::path& gguf_file,
        const std::string& target_weights_path, bool use_mmap) {
    DsparkDraftWeights w;
    w.ckpt = parse_dspark_gguf_checkpoint_config(gguf_file);
    w.v4 = std::make_unique<DsparkV4HostWeights>();
    auto& v4 = *w.v4;
    v4.draft_gguf = GgufReader::open(gguf_file, use_mmap);
    const auto& g = v4.draft_gguf;
    const auto& c = w.ckpt;
    const int64_t H = c.hidden_size;
    const int64_t E = c.v4.n_routed_experts;
    const int64_t I = c.v4.moe_intermediate;
    const int64_t Q = c.v4.q_lora_rank;
    const int64_t D = c.head_dim;
    const int64_t HQ = c.num_attention_heads;
    const int64_t OG = c.v4.o_groups;
    const int64_t OLR = c.v4.o_lora_rank;
    const int64_t r = c.markov_rank;
    const int hc = c.v4.hc_mult;
    const int64_t mix = static_cast<int64_t>((2 + hc) * hc);
    const int64_t n_aux =
        static_cast<int64_t>(c.aux_hidden_state_layer_ids.size());

    // Strict coverage: every expected slot present, every file tensor mapped.
    const auto expected =
        v4_expected_names(c.num_hidden_layers, c.enable_confidence_head);
    std::unordered_map<std::string, bool> seen;
    for (const auto& n : expected) seen.emplace(n, false);
    std::vector<std::string> unmapped;
    for (const auto& e : g.entries()) {
        auto it = seen.find(e.name);
        if (it == seen.end())
            unmapped.push_back(e.name);
        else
            it->second = true;
    }
    std::vector<std::string> missing;
    for (const auto& [n, ok] : seen)
        if (!ok) missing.push_back(n);
    if (!unmapped.empty() || !missing.empty()) {
        std::sort(unmapped.begin(), unmapped.end());
        std::sort(missing.begin(), missing.end());
        std::string msg = "checkpoint coverage failure (INV-DSPARK-CKPT) in " +
                          gguf_file.string() + ":";
        auto append = [&msg](const char* label,
                             const std::vector<std::string>& v) {
            if (v.empty()) return;
            msg += std::string(" ") + label + " [" +
                   std::to_string(v.size()) + "]:";
            for (const auto& n : v) msg += " " + n;
            msg += ";";
        };
        append("unmapped", unmapped);
        append("missing", missing);
        fail(msg);
    }

    auto load = [&](const std::string& name) -> RawTensor {
        const auto* e = find_entry(g, name);
        // Coverage already verified above.
        RawTensor t =
            make_host_tensor(g, *e, is_bf16_converted_norm(name), v4.owned);
        ++w.total_tensors_loaded;
        w.total_weight_bytes += static_cast<int64_t>(t.data.size());
        return t;
    };

    // Model-level slots (reusing the base struct's members).
    w.fc = load("fc.weight");
    expect_shape(w.fc, "fc.weight", {H, n_aux * H});
    w.hidden_norm = load("enc.output_norm.weight");
    expect_shape(w.hidden_norm, "enc.output_norm.weight", {H});
    w.final_norm = load("output_norm.weight");
    expect_shape(w.final_norm, "output_norm.weight", {H});
    w.markov_w1 = load("markov_w1.weight");
    expect_shape(w.markov_w1, "markov_w1.weight", {c.vocab_size, r});
    w.markov_w2 = load("markov_w2.weight");
    expect_shape(w.markov_w2, "markov_w2.weight", {c.draft_vocab_size, r});
    if (c.enable_confidence_head) {
        w.confidence_proj_weight = load("conf_proj.weight");
        expect_shape(w.confidence_proj_weight, "conf_proj.weight",
                     {1, H + (c.confidence_head_with_markov ? r : 0)});
        // NO bias tensor in the GGUF format (confidence_has_bias = false).
    }
    v4.output_hc_fn = load("output_hc_fn.weight");
    expect_shape(v4.output_hc_fn, "output_hc_fn.weight", {hc, hc * H});
    v4.output_hc_base = load("output_hc_base.weight");
    expect_shape(v4.output_hc_base, "output_hc_base.weight", {hc});
    v4.output_hc_scale = load("output_hc_scale.weight");
    expect_shape(v4.output_hc_scale, "output_hc_scale.weight", {1});

    v4.layers.resize(static_cast<size_t>(c.num_hidden_layers));
    for (int l = 0; l < c.num_hidden_layers; ++l) {
        auto& L = v4.layers[static_cast<size_t>(l)];
        const std::string p = "blk." + std::to_string(l) + ".";
        L.attn_norm = load(p + "attn_norm.weight");
        expect_shape(L.attn_norm, p + "attn_norm", {H});
        L.ffn_norm = load(p + "ffn_norm.weight");
        L.q_a = load(p + "attn_q_a.weight");
        expect_shape(L.q_a, p + "attn_q_a", {Q, H});
        L.q_a_norm = load(p + "attn_q_a_norm.weight");
        expect_shape(L.q_a_norm, p + "attn_q_a_norm", {Q});
        L.q_b = load(p + "attn_q_b.weight");
        expect_shape(L.q_b, p + "attn_q_b", {HQ * D, Q});
        L.kv = load(p + "attn_kv.weight");
        expect_shape(L.kv, p + "attn_kv", {D, H});
        L.kv_norm = load(p + "attn_kv_a_norm.weight");
        expect_shape(L.kv_norm, p + "attn_kv_a_norm", {D});
        L.o_a = load(p + "attn_output_a.weight");
        expect_shape(L.o_a, p + "attn_output_a", {OG * OLR, HQ * D / OG});
        L.o_b = load(p + "attn_output_b.weight");
        expect_shape(L.o_b, p + "attn_output_b", {H, OG * OLR});
        L.sinks = load(p + "attn_sinks.weight");
        expect_shape(L.sinks, p + "attn_sinks", {HQ});
        L.hc_attn_fn = load(p + "hc_attn_fn.weight");
        expect_shape(L.hc_attn_fn, p + "hc_attn_fn", {mix, hc * H});
        L.hc_attn_base = load(p + "hc_attn_base.weight");
        L.hc_attn_scale = load(p + "hc_attn_scale.weight");
        L.hc_ffn_fn = load(p + "hc_ffn_fn.weight");
        expect_shape(L.hc_ffn_fn, p + "hc_ffn_fn", {mix, hc * H});
        L.hc_ffn_base = load(p + "hc_ffn_base.weight");
        L.hc_ffn_scale = load(p + "hc_ffn_scale.weight");
        L.gate_inp = load(p + "ffn_gate_inp.weight");
        expect_shape(L.gate_inp, p + "ffn_gate_inp", {E, H});
        L.exp_probs_b = load(p + "exp_probs_b.bias");
        expect_shape(L.exp_probs_b, p + "exp_probs_b", {E});
        L.shexp_gate = load(p + "ffn_gate_shexp.weight");
        expect_shape(L.shexp_gate, p + "ffn_gate_shexp", {I, H});
        L.shexp_up = load(p + "ffn_up_shexp.weight");
        L.shexp_down = load(p + "ffn_down_shexp.weight");
        expect_shape(L.shexp_down, p + "ffn_down_shexp", {H, I});
        L.exps_gate = load(p + "ffn_gate_exps.weight");
        expect_shape(L.exps_gate, p + "ffn_gate_exps", {E, I, H});
        L.exps_up = load(p + "ffn_up_exps.weight");
        expect_shape(L.exps_up, p + "ffn_up_exps", {E, I, H});
        L.exps_down = load(p + "ffn_down_exps.weight");
        expect_shape(L.exps_down, p + "ffn_down_exps", {E, H, I});
    }

    // Target-aliased embed + lm_head: token_embd/output from the TARGET GGUF
    // shards (the draft shares the target's embedding space — vLLM
    // has_own_embed_tokens = False).
    {
        const auto shards = resolve_gguf_files(target_weights_path);
        bool got_embed = false, got_head = false;
        for (const auto& path : shards) {
            auto reader = GgufReader::open(path, use_mmap);
            const auto* emb = find_entry(reader, "token_embd.weight");
            const auto* head = find_entry(reader, "output.weight");
            if (!emb && !head) continue;
            if (emb && !got_embed) {
                if (static_cast<GgmlType>(emb->ggml_type) != GgmlType::BF16)
                    fail("target token_embd.weight is not BF16");
                w.embed_tokens = make_host_tensor(reader, *emb, false,
                                                  v4.owned);
                expect_shape(w.embed_tokens, "target token_embd",
                             {c.vocab_size, H});
                got_embed = true;
                ++w.total_tensors_loaded;
                w.total_weight_bytes +=
                    static_cast<int64_t>(w.embed_tokens.data.size());
            }
            if (head && !got_head) {
                if (static_cast<GgmlType>(head->ggml_type) != GgmlType::BF16)
                    fail("target output.weight is not BF16");
                w.lm_head = make_host_tensor(reader, *head, false, v4.owned);
                expect_shape(w.lm_head, "target output",
                             {c.draft_vocab_size, H});
                got_head = true;
                ++w.total_tensors_loaded;
                w.total_weight_bytes +=
                    static_cast<int64_t>(w.lm_head.data.size());
            }
            v4.target_shards.push_back(std::move(reader));
            if (got_embed && got_head) break;
        }
        if (!got_embed || !got_head)
            fail("target GGUF (" + target_weights_path +
                 ") is missing token_embd.weight/output.weight for the "
                 "draft's aliased embed/lm_head");
    }

    spdlog::info(
        "dspark_gguf_loader: loaded {} dflash draft tensors ({} bytes) from "
        "{} — γ={} r={} V={} blocks={}x{} aux_layers={} conf={} (no bias)",
        w.total_tensors_loaded, w.total_weight_bytes, gguf_file.string(),
        c.speculative_tokens, c.markov_rank, c.vocab_size,
        c.num_hidden_layers, c.hidden_size,
        c.aux_hidden_state_layer_ids.size(),
        c.enable_confidence_head ? "yes" : "no");
    return w;
}

// ── upload_dspark_v4_gguf_draft ─────────────────────────────────────────────

DsparkDeviceWeights upload_dspark_v4_gguf_draft(
        const DsparkDraftWeights& weights, compute::DeviceBackend& backend,
        void* arena, int64_t arena_capacity) {
    if (!weights.v4 || !weights.ckpt.is_v4_dflash)
        fail("upload_dspark_v4_gguf_draft: weights are not a dflash "
             "checkpoint");
    const auto& c = weights.ckpt;
    const auto& hv = *weights.v4;
    const int64_t E = c.v4.n_routed_experts;
    const int L = c.num_hidden_layers;

    DsparkDeviceWeights out;
    out.v4 = std::make_unique<DsparkV4DeviceWeights>();
    out.v4->layers.resize(static_cast<size_t>(L));

    // (host tensor, device slot) pairs in a fixed order; after the load-time
    // conversions host bytes == device bytes for every tensor, so upload is
    // a plain per-tensor byte copy into the 256-aligned arena.
    struct Slot {
        const RawTensor* src;
        DsparkDeviceTensor* dst;
    };
    std::vector<Slot> pairs;
    auto add = [&pairs](const RawTensor& src, DsparkDeviceTensor& dst) {
        if (src.data.data() == nullptr) return;  // disabled optional head
        pairs.push_back({&src, &dst});
    };
    add(weights.embed_tokens, out.embed_tokens);
    add(weights.lm_head, out.lm_head);
    add(weights.fc, out.fc);
    add(weights.hidden_norm, out.hidden_norm);
    add(weights.final_norm, out.final_norm);
    add(weights.markov_w1, out.markov_w1);
    add(weights.markov_w2, out.markov_w2);
    add(weights.confidence_proj_weight, out.confidence_proj_weight);
    add(hv.output_hc_fn, out.v4->output_hc_fn);
    add(hv.output_hc_base, out.v4->output_hc_base);
    add(hv.output_hc_scale, out.v4->output_hc_scale);
    for (int l = 0; l < L; ++l) {
        const auto& s = hv.layers[static_cast<size_t>(l)];
        auto& d = out.v4->layers[static_cast<size_t>(l)];
        add(s.attn_norm, d.attn_norm);
        add(s.ffn_norm, d.ffn_norm);
        add(s.q_a, d.q_a);
        add(s.q_a_norm, d.q_a_norm);
        add(s.q_b, d.q_b);
        add(s.kv, d.kv);
        add(s.kv_norm, d.kv_norm);
        add(s.o_a, d.o_a);
        add(s.o_b, d.o_b);
        add(s.sinks, d.sinks);
        add(s.hc_attn_fn, d.hc_attn_fn);
        add(s.hc_attn_base, d.hc_attn_base);
        add(s.hc_attn_scale, d.hc_attn_scale);
        add(s.hc_ffn_fn, d.hc_ffn_fn);
        add(s.hc_ffn_base, d.hc_ffn_base);
        add(s.hc_ffn_scale, d.hc_ffn_scale);
        add(s.gate_inp, d.gate_inp);
        add(s.exp_probs_b, d.exp_probs_b);
        add(s.shexp_gate, d.shexp_gate);
        add(s.shexp_up, d.shexp_up);
        add(s.shexp_down, d.shexp_down);
        add(s.exps_gate, d.exps_gate);
        add(s.exps_up, d.exps_up);
        add(s.exps_down, d.exps_down);
    }

    // Arena layout: per-tensor 256-aligned offsets, then the per-layer
    // expert B_ptrs arrays (gate/up/down, [E] void* each).
    int64_t offset = 0;
    std::vector<int64_t> offsets;
    offsets.reserve(pairs.size());
    for (const auto& p : pairs) {
        offsets.push_back(offset);
        offset += align_up(static_cast<int64_t>(p.src->data.size()),
                           kArenaAlign);
    }
    const int64_t ptrs_bytes =
        align_up(E * static_cast<int64_t>(sizeof(void*)), kArenaAlign);
    std::vector<int64_t> ptr_offsets;  // L*3 arrays
    for (int i = 0; i < L * 3; ++i) {
        ptr_offsets.push_back(offset);
        offset += ptrs_bytes;
    }
    out.arena_bytes = offset;

    backend.set_device();
    if (arena) {
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
                 " (V4 dflash draft arena)");
        out.owns_arena = true;
    }
    out.backend = &backend;

    auto* base = static_cast<std::byte*>(out.arena);
    for (size_t i = 0; i < pairs.size(); ++i) {
        auto* dst = pairs[i].dst;
        dst->ptr = base + offsets[i];
        dst->bytes = static_cast<int64_t>(pairs[i].src->data.size());
        dst->shape = pairs[i].src->shape;
        dst->dtype = DsparkWeightDtype::kBF16;  // dtype tag unused by the V4
                                                // forward (no weight_gemm)
        backend.memcpy_h2d(dst->ptr, pairs[i].src->data.data(),
                           pairs[i].src->data.size());
        ++out.total_tensors_uploaded;
    }

    // Expert B_ptrs: per (layer, projection) an [E] device array of pointers
    // into the packed expert-major blocks (per-expert block size =
    // gguf_packed_bytes over the expert's logical [N, K]).
    std::vector<const void*> host_ptrs(static_cast<size_t>(E));
    int pi = 0;
    for (int l = 0; l < L; ++l) {
        auto& d = out.v4->layers[static_cast<size_t>(l)];
        auto fill = [&](const DsparkDeviceTensor& t, void*& dev_arr) {
            // t.shape = [E, N, K] logical.
            const int64_t block = gguf::gguf_packed_bytes(
                t.shape[1], t.shape[2], GgufKQuantType::MXFP4);
            if (block * E != t.bytes)
                fail("expert block size mismatch for layer " +
                     std::to_string(l));
            for (int64_t e = 0; e < E; ++e)
                host_ptrs[static_cast<size_t>(e)] =
                    static_cast<const std::byte*>(t.ptr) + e * block;
            dev_arr = base + ptr_offsets[static_cast<size_t>(pi++)];
            backend.memcpy_h2d(dev_arr, host_ptrs.data(),
                               static_cast<size_t>(E) * sizeof(void*));
        };
        fill(d.exps_gate, d.exps_gate_ptrs);
        fill(d.exps_up, d.exps_up_ptrs);
        fill(d.exps_down, d.exps_down_ptrs);
    }
    backend.device_sync();

    spdlog::info(
        "dspark_gguf_loader: uploaded {} dflash draft tensors ({} arena "
        "bytes; MXFP4 experts native, Q8_0 dequanted to BF16) onto GPU "
        "position {} (cuda id {})",
        out.total_tensors_uploaded, out.arena_bytes,
        backend.gpu().position, backend.gpu().id);
    return out;
}

}  // namespace layerstorm::model

"""Model identity from a bare weights path (AUTO_RUN step 0).

``autoconfigure --model <path>`` must be enough. Everything the solver needs
about WHICH model it is fitting — the recipe ``model`` section, the quant
route, the prepacked store, the tokenizer directory — is derived here from
the weights themselves, CPU-only, header-reads only.

Three sources, in precedence order:
  1. a base recipe's ``model`` section (``--config``; advanced/override);
  2. a HuggingFace ``config.json`` in or beside the weights directory —
     field names already match ours per the CLAUDE.md convention, so the
     mapping is nearly identity;
  3. the GGUF metadata KV block + an ARCHITECTURE PROFILE for the handful of
     fields GGUF has no key for.

The profiles are DATA with citations, in the spirit of engine_constraints.py:
a GGUF cannot express IndexShare geometry (``index_topk_freq`` /
``index_skip_topk_offset``) or the rope interleave flags, and getting them
wrong is not cosmetic — ``index_topk_freq <= 0`` makes the engine treat EVERY
layer as a full indexer layer (model_config.h:41), which is 79 computing
layers instead of GLM-5.2's 21 and a ~4x indexer-K sizing error. Values come
from spec/LLM-MODELS-INFO.md, which reads them off the published HF config.

An architecture with no profile and no HF config is a REFUSAL, not a guess
(AUTOCONFIG §6): the user is told to pass ``--config`` with a model section.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import os
import re
from dataclasses import dataclass

from tokenizer.locator import (TIER_TEST_DATA, TOKENIZER_MARKERS,
                               locate_tokenizer_dir)

from . import gguf_meta
from .explain import Infeasible

_log = logging.getLogger("layerstorm.autoconfig")

# TOKENIZER_MARKERS is re-exported from tokenizer.locator (shared, P-34).


@dataclass(frozen=True)
class ModelSource:
    """Everything the solver needs to know about the model, plus the paths
    the emitted recipe must carry."""
    model_section: dict
    quantization: dict
    weights_path: str          # as written into the recipe
    weights_abs: str
    weights_format: str
    prepacked_dir: str         # "" = none discovered (live prepack / safetensors)
    live_prepack: bool
    tokenizer_path: str        # "auto" or a concrete directory
    display_name: str          # sanitised, drives the default config filename
    source: str                # "hf-config" | "gguf-metadata" | "base-recipe"
    provenance: tuple[str, ...] = ()


# --------------------------------------------------------------- profiles

@dataclass(frozen=True)
class ArchProfile:
    """One GGUF architecture the derivation knows how to read."""
    gguf_arch: str          # general.architecture
    engine_arch: str        # model.architecture (config/schema.json enum)
    constants: dict         # fields GGUF has no key for
    reference: str          # where the constants come from
    kv_heads_from: str = "head_count"   # MLA archs report head_count_kv=1


GATING_FUNC = {1: "softmax", 2: "sigmoid", 4: "sqrtsoftplus"}

ARCH_PROFILES = (
    ArchProfile(
        gguf_arch="glm-dsa",
        engine_arch="glm_moe_dsa",
        # GLM-5/GLM-5.2 class. IndexShare period + leading-full count and the
        # two interleave flags have no GGUF key; without them the engine
        # would run a full indexer on all 79 layers.
        constants={
            "moe_layer_freq": 1,
            "index_topk_freq": 4,
            "index_skip_topk_offset": 3,
            "rope_interleave": True,
            "indexer_rope_interleave": True,
        },
        reference="spec/LLM-MODELS-INFO.md 2b (GLM-5.2 config.json)",
        kv_heads_from="head_count",
    ),
    ArchProfile(
        gguf_arch="glm5next",
        engine_arch="glm5_next",
        # GLM-5.3-Flash class (hybrid KDA + sparse MLA + MoE).  The GGUF
        # publishes nearly everything (incl. the per-layer
        # attention.head_count_kv array that encodes layer_types, and
        # attention.indexer.kpool); these four have NO GGUF key — the
        # MODELINFO §8b cross-check lists them as omitted.  num_heads for
        # the KDA layers equals attention.head_count on this family but
        # GGUF has no kda.head_count key, so the value is asserted by the
        # profile, not read.
        constants={
            "moe_layer_freq": 1,
            "indexer_rope_interleave": True,   # inert at rope dim 0 (NoPE)
            "index_kpool_compress": True,
            "index_kpool_always_select_tail": True,
            "mla_use_nope": True,
        },
        reference="spec/LLM-MODELS-INFO.md §1-2 + "
                  "spec/GLM-5.3-FLASH-MODELINFO.md §3/§8b "
                  "(GLM-5.3-Flash config.json)",
        kv_heads_from="head_count",  # head_count_kv is a PER-LAYER array here
    ),
    ArchProfile(
        gguf_arch="deepseek4",
        engine_arch="deepseek_v4",
        constants={
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
            "num_nextn_predict_layers": 0,
        },
        reference="recipes/deepseek_v4_serve_tp2ep4.json (shipped V4 recipe)",
        kv_heads_from="head_count_kv",
    ),
)


def profile_for(gguf_arch: str):
    return next((p for p in ARCH_PROFILES if p.gguf_arch == gguf_arch), None)


# --------------------------------------------------------------- helpers

def _sanitise(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    return s.lower() or "model"


def _rel_to(path: str, root: str) -> str:
    """Repo-relative when the path lives under the repo (so the emitted
    recipe stays portable like the shipped ones), absolute otherwise."""
    ap, ar = os.path.abspath(path), os.path.abspath(root)
    return os.path.relpath(ap, ar) if ap.startswith(ar + os.sep) else ap


def first_shard(path: str) -> str:
    """The GGUF file to read: a file is itself; a directory yields shard 1."""
    if os.path.isfile(path):
        return path
    cands = sorted(_glob.glob(os.path.join(path, "*.gguf")))
    if not cands:
        raise Infeasible(
            "model-path-has-no-weights", path, path,
            "a .gguf file, a directory of GGUF shards, or a HF model dir",
            "point --model at the model directory or its first .gguf shard")
    firsts = [c for c in cands if "-00001-of-" in os.path.basename(c)]
    return firsts[0] if firsts else cands[0]


def find_tokenizer_dir(weights_abs: str, repo_root: str = "",
                       display_name: str = "") -> str:
    """serve.py resolve_tokenizer_dir("auto") prefers the WEIGHTS dir and
    GGUF-embedded tokenizers are not extracted (TD-SERVE-GGUF-TOKENIZER),
    so a GGUF-only box needs a real path or serve refuses to boot.  The
    search (P-34: ONE precedence, shared with serve via tokenizer.locator):
    weights dir, then name-prefix siblings beside the weights
    (test-data/GLM-5.2-GGUF-Q4_K_XL -> test-data/GLM-5.2), then — FALLBACK
    ONLY, surfaced loudly by probe_model — repo test-data dirs whose
    sanitised name matches the model's display name (P-31: weights living
    OUTSIDE the repo, e.g. /srv/models, have no useful siblings)."""
    return locate_tokenizer_dir(weights_abs, model_name=display_name,
                                repo_root=repo_root)[0]


def find_prepacked_dir(weights_abs: str, repo_root: str) -> str:
    """A prepack manifest records the model it was packed from
    (``source_model_path``) — that is the identity check, not the name.
    Search the weights' parent and the repo's test-data dir."""
    want = os.path.abspath(weights_abs)
    wdir = os.path.dirname(want)
    # the weights dir, its PARENT (where the shipped store lives:
    # test-data/GLM-5.2-prepacked beside test-data/GLM-5.2-GGUF-Q4_K_XL)
    # and the repo test-data dir
    roots = {wdir, os.path.dirname(wdir),
             os.path.join(os.path.abspath(repo_root), "test-data")}
    for root in sorted(roots):
        for cand in sorted(_glob.glob(os.path.join(root, "*prepack*"))):
            man = os.path.join(cand, "manifest.json")
            if not os.path.isfile(man):
                continue
            try:
                with open(man) as f:
                    m = json.load(f)
            except (OSError, ValueError):
                continue
            src = m.get("source_model_path", "")
            if not src:
                continue
            src_abs = src if os.path.isabs(src) else os.path.join(
                os.path.abspath(repo_root), src)
            if os.path.abspath(src_abs) == want:
                return cand
    return ""


def _quantization_for(weights_format: str, hf) -> dict:
    """The quant route. GGUF serves through the int k-quant path (both
    shipped recipes); a safetensors checkpoint's own quant config decides."""
    if weights_format == "gguf":
        return {"weights": "gguf", "gguf_strategy": "int",
                "attention_compute": "fp8_e4m3", "kv_cache": "fp8_e4m3",
                "gating_compute": "fp16"}
    qc = ((hf or {}).get("quantization_config") or {})
    method = str(qc.get("quant_method", "")).lower()
    weights = "nvfp4" if ("nvfp4" in method or "fp4" in method) else (
        "fp8_e4m3" if "fp8" in method else "bf16")
    return {"weights": weights, "attention_compute": "fp8_e4m3",
            "kv_cache": "fp8_e4m3", "gating_compute": "fp16"}


# ------------------------------------------------------------ GGUF route

def _gguf_model_section(kv: dict, prof: ArchProfile):
    a = prof.gguf_arch

    def g(suffix, default=None):
        return kv.get(a + "." + suffix, default)

    prov = []
    nextn = int(g("nextn_predict_layers", prof.constants.get(
        "num_nextn_predict_layers", 0)) or 0)
    blocks = int(g("block_count", 0))
    rope_dim = int(g("rope.dimension_count", 0) or 0)
    vocab = g("vocab_size")
    if not vocab:
        toks = kv.get("tokenizer.ggml.tokens") or {}
        vocab = int(toks.get("_array_len", 0)) if isinstance(toks, dict) else 0
        prov.append("vocab_size from the GGUF token-table length "
                    "(this architecture publishes no vocab_size key)")
    heads = int(g("attention.head_count", 0) or 0)
    kv_heads = heads if prof.kv_heads_from == "head_count" else int(
        g("attention.head_count_kv", 1) or 1)

    m = {
        "architecture": prof.engine_arch,
        # block_count counts the MTP block(s) too; the engine's
        # num_hidden_layers does not (num_nextn_predict_layers is separate).
        "num_hidden_layers": blocks - nextn,
        "hidden_size": int(g("embedding_length", 0) or 0),
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "q_lora_rank": int(g("attention.q_lora_rank", 0) or 0),
        "qk_rope_head_dim": rope_dim,
        "n_routed_experts": int(g("expert_count", 0) or 0),
        "n_shared_experts": int(g("expert_shared_count", 0) or 0),
        "num_experts_per_tok": int(g("expert_used_count", 0) or 0),
        "n_group": int(g("expert_group_count", 1) or 1),
        "topk_group": int(g("expert_group_used_count", 1) or 1),
        "vocab_size": int(vocab or 0),
        "max_position_embeddings": int(g("context_length", 0) or 0),
        "rms_norm_eps": float(g("attention.layer_norm_rms_epsilon", 0.0) or 0.0),
        "num_nextn_predict_layers": nextn,
        "routed_scaling_factor": float(g("expert_weights_scale", 1.0) or 1.0),
        "moe_intermediate_size": int(g("expert_feed_forward_length", 0) or 0),
        "norm_topk_prob": bool(g("expert_weights_norm", True)),
        "gating_score_fn": GATING_FUNC.get(int(g("expert_gating_func", 2) or 2),
                                           "sigmoid"),
        "index_topk": int(g("attention.indexer.top_k", 0) or 0),
        "index_n_heads": int(g("attention.indexer.head_count", 0) or 0),
        "index_head_dim": int(g("attention.indexer.key_length", 0) or 0),
    }
    # rope_theta: emit only when the GGUF actually carries a positive
    # rope.freq_base — a NoPE architecture (glm5_next, rope dim 0) has no
    # rope key at all, and emitting 0.0 violates the schema's
    # positive_float (exclusiveMinimum 0; default 10000).
    _rt = float(g("rope.freq_base", 0.0) or 0.0)
    if _rt > 0:
        m["rope_theta"] = _rt
    if g("leading_dense_block_count") is not None:
        m["first_k_dense_replace"] = int(g("leading_dense_block_count"))
    if g("feed_forward_length") is not None:
        m["intermediate_size"] = int(g("feed_forward_length"))

    if prof.engine_arch == "deepseek_v4":
        # V4: no uniform MLA split; the side-tier geometry is the shape.
        m["head_dim"] = int(g("attention.key_length", 0) or 0)
        m["intermediate_size"] = int(g("expert_feed_forward_length", 0) or 0)
        ratios = list(g("attention.compress_ratios", []) or [])
        m["compress_ratios"] = [int(x) for x in ratios[:m["num_hidden_layers"]]]
        m["compress_rope_theta"] = float(g("attention.compress_rope_freq_base", 0.0) or 0.0)
        m["sliding_window"] = int(g("attention.sliding_window", 0) or 0)
        m["o_groups"] = int(g("attention.output_group_count", 1) or 1)
        m["o_lora_rank"] = int(g("attention.output_lora_rank", 0) or 0)
        m["hc_mult"] = int(g("hyper_connection.count", 1) or 1)
        m["hc_sinkhorn_iters"] = int(g("hyper_connection.sinkhorn_iterations", 0) or 0)
        m["hc_eps"] = float(g("hyper_connection.epsilon", 0.0) or 0.0)
        m["num_hash_layers"] = int(g("hash_layer_count", 0) or 0)
        clamp = g("swiglu_clamp_exp", []) or []
        if clamp:
            m["swiglu_limit"] = float(clamp[0])
        scaling_type = g("rope.scaling.type", "")
        if scaling_type:
            m["rope_scaling"] = {
                "type": str(scaling_type),
                "factor": float(g("rope.scaling.factor", 1.0) or 1.0),
                "original_max_position_embeddings":
                    int(g("rope.scaling.original_context_length", 0) or 0),
                "beta_fast": float(g("rope.scaling.yarn_beta_fast", 0.0) or 0.0),
                "beta_slow": float(g("rope.scaling.yarn_beta_slow", 0.0) or 0.0),
            }
    else:
        # MLA archs: key_length_mla = qk_nope + qk_rope, value_length_mla = v.
        key_mla = int(g("attention.key_length_mla", 0) or 0)
        m["kv_lora_rank"] = int(g("attention.kv_lora_rank", 0) or 0)
        m["qk_nope_head_dim"] = key_mla - rope_dim
        m["v_head_dim"] = int(g("attention.value_length_mla", 0) or 0)

    if prof.engine_arch == "glm5_next":
        # Hybrid geometry (KDA + sparse MLA). layer_types is encoded in the
        # PER-LAYER attention.head_count_kv array (1 = KV-bearing DSA layer,
        # 0 = linear attention); the array covers the MTP block too, so it
        # is truncated to num_hidden_layers.
        kv_arr = g("attention.head_count_kv", []) or []
        if isinstance(kv_arr, (list, tuple)) and kv_arr:
            m["layer_types"] = [
                "deepseek_sparse_attention" if int(x) > 0
                else "linear_attention"
                for x in list(kv_arr)[:m["num_hidden_layers"]]]
            prov.append("layer_types from the per-layer "
                        "attention.head_count_kv array (1 = DSA layer, "
                        "0 = linear attention; MTP entry truncated)")
        m["index_kpool"] = int(g("attention.indexer.kpool", 1) or 1)
        m["linear_attn_config"] = {
            "num_heads": heads,   # no kda.head_count key; == head_count here
            "head_dim": int(g("kda.head_dim", 128) or 128),
            "short_conv_kernel_size": int(g("ssm.conv_kernel", 4) or 4),
            "gate_lower_bound": float(g("kda.gate_lower_bound", -5.0)
                                      or -5.0),
        }
        m["hc_mult"] = int(g("hyper_connection.count", 1) or 1)
        m["hc_sinkhorn_iters"] = int(
            g("hyper_connection.sinkhorn_iterations", 0) or 0)
        m["hc_eps"] = float(g("hyper_connection.epsilon", 0.0) or 0.0)
        clamp = g("swiglu_clamp_exp", []) or []
        if isinstance(clamp, (list, tuple)) and clamp:
            m["swiglu_limit"] = float(clamp[0])

    m.update(prof.constants)
    prov.append("GGUF metadata KV block (" + a + ".*) of shard 1")
    prov.append("architecture profile '" + prof.gguf_arch + "' supplies "
                + str(sorted(prof.constants)) + " — no GGUF key exists for "
                "these (" + prof.reference + ")")
    return m, prov


# -------------------------------------------------------------- HF route

# HF config.json -> recipe model section. Names already match per the
# CLAUDE.md convention; only these differ.
_HF_RENAME = {
    "model_type": "architecture",
    "scoring_func": "gating_score_fn",
}
_HF_DROP = {
    "architectures", "attention_bias", "attention_dropout", "dtype",
    "ep_size", "hidden_act", "initializer_range", "pad_token_id",
    "pretraining_tp", "tie_word_embeddings", "transformers_version",
    "use_cache", "eos_token_id", "bos_token_id", "torch_dtype",
    "quantization_config", "auto_map", "rope_parameters", "topk_method",
    "index_topk_pattern", "indexer_types", "mlp_layer_types",
    "index_share_for_mtp_iteration", "head_dim", "qk_head_dim",
}


def _hf_model_section(hf: dict):
    descended = False
    if isinstance(hf.get("text_config"), dict):
        # Multimodal HF config (e.g. GLM-5.3-Flash): the text tower IS the
        # model we serve; geometry lives in text_config while the engine's
        # architecture name is the top-level family model_type
        # ("glm5_next", not "glm5_next_text").  P-34: a config.json placed
        # beside GGUF weights must yield a correct section.
        tc = dict(hf["text_config"])
        if hf.get("model_type"):
            tc["model_type"] = hf["model_type"]
        hf, descended = tc, True
    m = {}
    for k, v in hf.items():
        if k in _HF_DROP:
            continue
        m[_HF_RENAME.get(k, k)] = v
    rp = hf.get("rope_parameters") or {}
    if "rope_theta" not in m and rp.get("rope_theta") is not None:
        m["rope_theta"] = float(rp["rope_theta"])
    if rp.get("rope_type") and rp["rope_type"] != "default":
        m["rope_scaling"] = {k: v for k, v in rp.items() if k != "rope_theta"}
    if "layer_types" in hf:
        m["layer_types"] = list(hf["layer_types"])
    ordered = {"architecture": m.pop("architecture", "")}
    ordered.update(m)
    return ordered, ["HuggingFace config.json (field names match ours per "
                     "the config convention; only model_type/scoring_func "
                     "are renamed)"
                     + (" — multimodal config: text_config is the model, "
                        "top-level model_type is the architecture"
                        if descended else "")]


# ------------------------------------------------------------------ probe

def probe_model(model_path: str, repo_root: str = ".", *,
                hf_config: str = "", prepacked: str = "",
                tokenizer: str = "", base_model_section=None,
                base_quantization=None) -> ModelSource:
    """Derive a ModelSource from a weights path (AUTO_RUN step 0)."""
    abs_path = os.path.abspath(
        model_path if os.path.isabs(model_path)
        else os.path.join(repo_root, model_path))
    if not os.path.exists(abs_path):
        raise Infeasible("model-path-missing", model_path, model_path,
                         "an existing path", "check --model")

    is_dir = os.path.isdir(abs_path)
    hf_path = hf_config or (os.path.join(abs_path, "config.json") if is_dir else "")
    hf = None
    if hf_path and os.path.isfile(hf_path):
        with open(hf_path) as f:
            hf = json.load(f)

    gguf_here = bool(_glob.glob(os.path.join(abs_path, "*.gguf"))) if is_dir \
        else abs_path.endswith(".gguf")
    prov = []

    def _hf_has_geometry(cfg) -> bool:
        """A config.json beside GGUF weights is only authoritative for the
        model SECTION when it actually carries the geometry — tokenizer
        checkouts often ship a stub (model_type + vocab + token ids, e.g.
        the V4-Flash weights dir).  A stub must not shadow GGUF metadata
        (P-34)."""
        tc = cfg.get("text_config") if isinstance(
            cfg.get("text_config"), dict) else cfg
        return "num_hidden_layers" in tc

    if gguf_here:
        weights_abs = first_shard(abs_path)
        weights_format = "gguf"
        kv = gguf_meta.read_metadata(weights_abs)
        arch = str(kv.get("general.architecture", ""))
        display = str(kv.get("general.name") or kv.get("general.basename")
                      or os.path.basename(os.path.dirname(weights_abs)))
        if base_model_section:
            section, prov = dict(base_model_section), ["base recipe model section"]
            source = "base-recipe"
        elif hf is not None and _hf_has_geometry(hf):
            section, prov = _hf_model_section(hf)
            source = "hf-config"
        else:
            if hf is not None:
                prov.append("config.json beside the weights lacks model "
                            "geometry (stub) — ignored for the model "
                            "section, using GGUF metadata")
            profile = profile_for(arch)
            if profile is None:
                raise Infeasible(
                    "unknown-gguf-architecture", weights_abs,
                    "general.architecture='" + arch + "'",
                    "one of " + ", ".join(p.gguf_arch for p in ARCH_PROFILES)
                    + " (or a HF config.json beside the weights)",
                    "pass --config <recipe with a model section>, or add an "
                    "ArchProfile in python/autoconfig/modelprobe.py")
            section, prov = _gguf_model_section(kv, profile)
            source = "gguf-metadata"
    else:
        if hf is None:
            raise Infeasible(
                "model-path-unrecognised", abs_path, abs_path,
                "GGUF shards or a HF config.json",
                "point --model at a GGUF file/dir or a HuggingFace model dir")
        weights_abs = abs_path
        weights_format = "safetensors"
        display = os.path.basename(abs_path.rstrip("/"))
        if base_model_section:
            section, prov = dict(base_model_section), ["base recipe model section"]
            source = "base-recipe"
        else:
            section, prov = _hf_model_section(hf)
            source = "hf-config"

    weights_rel = _rel_to(weights_abs, repo_root)
    section["weights_path"] = weights_rel
    section["weights_format"] = weights_format
    # canonical field order: identity first, then geometry (readability —
    # the emitted recipe is meant to be diffed against the shipped ones)
    head = {k: section.pop(k) for k in ("architecture", "weights_path",
                                        "weights_format") if k in section}
    section = dict(head, **section)

    prep = prepacked or find_prepacked_dir(weights_abs, repo_root)
    if prep:
        prep = _rel_to(prep, repo_root)
        prov.append("prepacked store " + prep + " (manifest source_model_path "
                    "matches these weights)")
    live_prepack = (not prep) and weights_format == "gguf"

    tok, tok_tier = (tokenizer, "explicit") if tokenizer else \
        locate_tokenizer_dir(weights_abs, model_name=display,
                             repo_root=repo_root)
    tok_out = _rel_to(tok, repo_root) if tok else "auto"
    if tok and tok_tier == TIER_TEST_DATA:
        # P-34: a repo test dir satisfying a production derivation is a
        # works-on-this-checkout-only arrangement — allowed, but LOUD.
        _log.warning(
            "TOKENIZER FALLBACK: no tokenizer files beside the weights "
            "(%s) — using repo test-data at %s. Place the HF tokenizer "
            "files next to the weights (or pass --tokenizer) to make the "
            "recipe portable.", weights_abs, tok)
        prov.append("tokenizer dir " + tok_out + " — FALLBACK from repo "
                    "test-data/ (nothing found beside the weights; put the "
                    "HF tokenizer files next to the weights or pass "
                    "--tokenizer — TD-SERVE-GGUF-TOKENIZER)")
    elif tok:
        prov.append("tokenizer dir " + tok_out + " (found beside the "
                    "weights; GGUF tokenizers are not extracted — "
                    "TD-SERVE-GGUF-TOKENIZER)")

    return ModelSource(
        model_section=section,
        quantization=dict(base_quantization) if base_quantization
        else _quantization_for(weights_format, hf),
        weights_path=weights_rel, weights_abs=weights_abs,
        weights_format=weights_format, prepacked_dir=prep,
        live_prepack=live_prepack, tokenizer_path=tok_out,
        display_name=_sanitise(display), source=source,
        provenance=tuple(prov))


def find_draft_checkpoints(weights_abs: str, repo_root: str) -> list:
    """All `.dspark` speculator CANDIDATES beside the weights or in
    test-data, as recipe-ready paths, nearest root first. Identity is NOT
    checked here — the caller must match each candidate to the model
    (solver.draft_identity_errors) before wiring one in: a `.dspark` merely
    lying near the weights proves nothing about WHOSE speculator it is
    (TD-AUTOCONFIG-DRAFT-IDENTITY)."""
    roots = [os.path.dirname(os.path.abspath(weights_abs)),
             os.path.dirname(os.path.dirname(os.path.abspath(weights_abs))),
             os.path.join(os.path.abspath(repo_root), "test-data")]
    seen: set = set()
    out: list = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        for cand in sorted(_glob.glob(os.path.join(root, "*.dspark"))):
            if os.path.isfile(os.path.join(cand, "config.json")):
                out.append(_rel_to(cand, repo_root))
    return out


def find_draft_checkpoint(weights_abs: str, repo_root: str) -> str:
    """First `.dspark` candidate (see find_draft_checkpoints — identity
    UNCHECKED), "" when none exists (then autoconfig serves without a
    draft — AUTOCONFIG P6)."""
    cands = find_draft_checkpoints(weights_abs, repo_root)
    return cands[0] if cands else ""

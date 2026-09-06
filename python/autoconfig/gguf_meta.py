"""Minimal GGUF metadata reader — header + tensor infos only, CPU-only.

Reads ONLY the GGUF header region (metadata KVs + tensor descriptors), never
tensor data, so an 11-shard 500 GB model costs a few MB of reads. Two users:
  - per-layer expert projection quant types for the GG-9 rule: on a mixed
    "XL" GGUF, bytes_per_expert = per-projection MAX across layers
    (src/model/quantization/gguf_kquant.cpp:196-215, expert_prepacker.cpp:245);
  - the metadata KV block (``read_metadata``), from which
    ``modelprobe.py`` derives a recipe ``model`` section when the user
    points autoconfig at a bare weights path (AUTO_RUN step 0);
  - the engine's PINNED REGION (``pinned_region_layout``, bottom of this
    file): the per-rank VRAM the engine reserves for non-expert weights. It
    is NOT the stored-byte sum — it is a slot-by-slot transcription of
    ``build_upload_plan``/``compute_pinned_layout``, several of whose slots
    are upper bounds, and on the glm5_next GGUF path it needs this module's
    per-tensor width scan (GF3.15).

Format reference: ref/vllm + llama.cpp GGUF v3 spec (magic "GGUF", little-
endian; metadata value types 0-12).
"""

from __future__ import annotations

import glob as _glob
import os
import struct
from dataclasses import dataclass

_MAGIC = b"GGUF"

# gguf_type id -> (block_bytes, block_elems) for the k-quant family we size
# (mirror gguf_kquant.h:59-67 kBlockSpecs + float types).
GGUF_TYPE_BLOCKS = {
    0: (4, 1),      # F32
    1: (2, 1),      # F16
    8: (34, 32),    # Q8_0
    10: (84, 256),  # Q2_K
    11: (110, 256), # Q3_K
    12: (144, 256), # Q4_K
    13: (176, 256), # Q5_K
    14: (210, 256), # Q6_K
    30: (2, 1),     # BF16
    39: (17, 32),   # MXFP4
}

GGUF_TYPE_BLOCKS.update({
    2: (18, 32),    # Q4_0
    3: (20, 32),    # Q4_1
    6: (22, 32),    # Q5_0
    7: (24, 32),    # Q5_1
})

GGUF_TYPE_NAMES = {
    0: "f32", 1: "f16", 8: "q8_0", 10: "q2_k", 11: "q3_k", 12: "q4_k",
    13: "q5_k", 14: "q6_k", 30: "bf16", 39: "mxfp4",
}


@dataclass(frozen=True)
class GgufTensorInfo:
    name: str
    dims: tuple[int, ...]  # ne[0] fastest (contraction dim first)
    gguf_type: int
    offset: int


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


_SCALAR_FMT = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
    5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
    12: ("<d", 8),
}

# GGUF stores f32; a bf16-trained eps arrives as 9.999999747378752e-06.
# 7 significant digits is exactly float32's decimal precision, so this
# recovers the authored value (1e-05) without inventing precision.
_F32_TYPES = (6,)


def _read_value(f, vtype: int, keep_arrays: int = 64):
    """Read one metadata value. Arrays longer than ``keep_arrays`` are
    consumed but returned as their length (token vocabularies are 150k+
    strings and no consumer of this module wants them in memory)."""
    if vtype in _SCALAR_FMT:
        fmt, n = _SCALAR_FMT[vtype]
        v = struct.unpack(fmt, f.read(n))[0]
        if vtype in _F32_TYPES:
            return float(f"{v:.7g}")
        return v
    if vtype == 8:
        return _read_str(f)
    if vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        vals = []
        for i in range(count):
            v = _read_value(f, etype, keep_arrays)
            if i < keep_arrays:
                vals.append(v)
        return vals if count <= keep_arrays else {"_array_len": count}
    raise ValueError(f"GGUF: unknown metadata value type {vtype}")


def _skip_value_elem(f, etype: int) -> None:
    """Consume one ARRAY ELEMENT of type etype (arrays are not nested)."""
    if etype in _SCALAR_FMT:
        f.read(_SCALAR_FMT[etype][1])
    elif etype == 8:
        _read_str(f)
    else:
        raise ValueError(f"GGUF: unsupported array element type {etype}")


def _skip_value(f, vtype: int) -> None:
    """Consume one metadata value without materialising it (fast path for
    read_tensor_infos, which walks past the KV block)."""
    if vtype in _SCALAR_FMT:
        f.read(_SCALAR_FMT[vtype][1])
    elif vtype == 8:  # string
        _read_str(f)
    elif vtype == 9:  # array
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if etype in _SCALAR_FMT:
            f.read(_SCALAR_FMT[etype][1] * count)
        elif etype == 8:
            for _ in range(count):
                _read_str(f)
        else:
            for _ in range(count):
                _skip_value(f, etype)
    else:
        raise ValueError(f"GGUF: unknown metadata value type {vtype}")


def read_metadata(path: str, keep_arrays: int = 64) -> dict:
    """The GGUF metadata KV block of one shard, as a dict.

    Shard 1 of a split set carries the full architecture block (later shards
    carry only split bookkeeping), so callers pass the first shard."""
    out: dict = {}
    with open(path, "rb") as f:
        if f.read(4) != _MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"{path}: GGUF v{version} unsupported")
        struct.unpack("<Q", f.read(8))          # n_tensors
        (n_kv,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if key.startswith("tokenizer.ggml.") and vtype == 9:
                # 150k-entry vocabularies: record the LENGTH (some archs
                # publish no vocab_size key and the token count is the only
                # source) and skip the elements.
                (etype,) = struct.unpack("<I", f.read(4))
                (count,) = struct.unpack("<Q", f.read(8))
                for _ in range(count):
                    _skip_value_elem(f, etype)
                out[key] = {"_array_len": count}
                continue
            out[key] = _read_value(f, vtype, keep_arrays)
    return out


def read_tensor_infos(path: str) -> list[GgufTensorInfo]:
    """Tensor descriptors of one shard (header-only read)."""
    with open(path, "rb") as f:
        if f.read(4) != _MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"{path}: GGUF v{version} unsupported")
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_kv):
            _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            _skip_value(f, vtype)
        infos: list[GgufTensorInfo] = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            gtype, = struct.unpack("<I", f.read(4))
            off, = struct.unpack("<Q", f.read(8))
            infos.append(GgufTensorInfo(name, tuple(int(d) for d in dims), gtype, off))
        return infos


def _shard_paths(first_shard: str) -> list[str]:
    """All shards of a `-NNNNN-of-NNNNN.gguf` set (or the single file)."""
    base = os.path.basename(first_shard)
    if "-of-" not in base:
        return [first_shard]
    prefix = base[: base.rindex("-", 0, base.rindex("-of-"))]
    pattern = os.path.join(os.path.dirname(first_shard), prefix + "-*-of-*.gguf")
    shards = sorted(_glob.glob(pattern))
    return shards or [first_shard]


def packed_bytes(out_dim: int, in_dim: int, gguf_type: int) -> int:
    """gguf_kquant.cpp gguf_packed_bytes(): out * (in/QK) * block_bytes."""
    block_bytes, qk = GGUF_TYPE_BLOCKS[gguf_type]
    if in_dim % qk:
        raise ValueError(f"in_dim {in_dim} not divisible by block {qk}")
    return out_dim * (in_dim // qk) * block_bytes


def expert_slot_bytes_from_gguf(first_shard: str) -> tuple[int, dict]:
    """GG-9 slot size: per-projection MAX across MoE layers.

    Returns (bytes_per_expert, detail) where detail records the winning type
    per projection for the explanation line. Expert tensors are the 3D
    `blk.<l>.ffn_{gate,up,down}_exps.weight` (ne = [in, out, n_experts])."""
    best: dict[str, tuple[int, int]] = {}  # proj -> (bytes, gguf_type)
    for shard in _shard_paths(first_shard):
        for t in read_tensor_infos(shard):
            for proj in ("gate", "up", "down"):
                if t.name.endswith(f"ffn_{proj}_exps.weight"):
                    in_dim, out_dim = t.dims[0], t.dims[1]
                    b = packed_bytes(out_dim, in_dim, t.gguf_type)
                    if proj not in best or b > best[proj][0]:
                        best[proj] = (b, t.gguf_type)
    if len(best) != 3:
        raise ValueError(f"{first_shard}: expert projection tensors not found "
                         f"(got {sorted(best)})")
    total = sum(b for b, _ in best.values())
    detail = {p: {"bytes": b, "type": GGUF_TYPE_NAMES.get(ty, str(ty))}
              for p, (b, ty) in best.items()}
    return total, detail


def expert_slot_bytes_from_manifest(prepacked_dir: str) -> int | None:
    """Prefer the prepack manifest's slot_size_bytes when present — it is the
    engine's own slot unit (engine.cpp: expert_source_()->slot_size_bytes())."""
    import json
    p = os.path.join(prepacked_dir, "manifest.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        m = json.load(f)
    slot = m.get("slot") or {}
    v = slot.get("slot_size_bytes")
    return int(v) if v else None


def non_expert_bytes(first_shard: str) -> tuple[int, int]:
    """Total stored bytes of all NON-expert tensors across shards (the
    pinned-weights raw material: attention, dense FFN, shared expert,
    gating, norms, embedding, output head, indexer).

    Returns (known_bytes, unknown_type_bytes_estimated_as_f16). Under the
    GGUF serving route these upload at stored size (norms F32->BF16 halve,
    a small correction we ignore conservatively)."""
    known = unknown = 0
    for shard in _shard_paths(first_shard):
        for t in read_tensor_infos(shard):
            if "_exps." in t.name:
                continue
            numel = 1
            for d in t.dims:
                numel *= d
            spec = GGUF_TYPE_BLOCKS.get(t.gguf_type)
            if spec is None:
                unknown += numel * 2  # f16-sized upper estimate
            else:
                block_bytes, qk = spec
                known += (numel // qk) * block_bytes
    return known, unknown


# ═══════════════════════════════════════════════════════════════════════════
#  The engine's PINNED REGION, transcribed (TD-AUTOCONFIG-PINNED-BYTES-
#  UPPER-BOUND)
#
#  `non_expert_bytes` above answers "how many bytes of non-expert weight does
#  this checkpoint STORE".  That is NOT what the engine reserves.  The engine
#  reserves a REGION laid out slot by slot by
#  ``build_upload_plan`` (src/model/pinned_upload_plan.cpp:76) and summed by
#  ``compute_pinned_layout`` (src/model/pinned_region_layout.cpp:565), whose
#  slots are sized from the CONFIG (plus, on the glm5_next GGUF path, a header
#  pre-scan of the checkpoint's real k-quant widths) — several of them at an
#  upper bound above the stored size.  On GLM-5.3-Flash the two differ by
#  6.7 GiB, which is enough to make the solver hand out VRAM that does not
#  exist.
#
#  Everything below is a faithful transcription of those two functions; each
#  block cites the C++ site.  Divergence between this file and the engine is a
#  bug HERE — the engine is authoritative (same contract as sizing.py).  The
#  boot-log cross-check in ``enginecheck.py`` is what keeps the two honest.
# ═══════════════════════════════════════════════════════════════════════════

# pinned_region_layout.h:39 / layer_registry.h:25 / pinned_region_layout.h:178.
BF16_BYTES_PER_ELEMENT = 2.0
EMBEDDING_BYTES_PER_ELEMENT = 2.0
UPLOAD_ALIGN_BUDGET = 32 * 1024
_FP8_BLOCK_SCALE_TILE = 128          # fp8.h:17 kBlockScaleTile

# gguf_reader.h:76 is_kquant() -> is_supported_gguf_kquant(): the seven packed
# families.  A plain F32/F16/BF16 tensor is NOT in the engine's width map, so
# its slot keeps the formula's own (BF16 / F32) sizing.
KQUANT_GGUF_TYPES = frozenset({8, 10, 11, 12, 13, 14, 39})

# config::WeightQuant uniform GGUF variants -> gguf type id
# (gguf_kquant.cpp type_from_weight_quant).
GGUF_UNIFORM_WEIGHT_QUANT = {
    "gguf_q2_k": 10, "gguf_q3_k": 11, "gguf_q4_k": 12, "gguf_q5_k": 13,
    "gguf_q6_k": 14, "gguf_q8_0": 8, "gguf_mxfp4": 39,
}
_Q8_0 = 8


class UnmodelledArchitecture(ValueError):
    """This architecture's pinned region is not transcribed here.

    Raised rather than guessed: a wrong pinned figure is the bug this module
    exists to kill, so the caller must fall back to a clearly-labelled
    estimate instead of receiving a number that looks authoritative."""


def align16(n: int) -> int:
    """The 16-byte slot round-up every attention sizer ends with
    (pinned_region_layout.cpp:555)."""
    return (n + 15) & ~15


def is_gguf_weight_quant(wq: str) -> bool:
    """gguf_kquant.cpp is_gguf_weight_quant()."""
    return wq == "gguf" or wq in GGUF_UNIFORM_WEIGHT_QUANT


def weight_bytes_per_element(wq: str) -> float:
    """layer_registry.cpp:55-90 bytes_per_element(config::WeightQuant).

    The generic ``gguf`` sentinel has no single bytes/element (per-tensor
    mixed) — the engine throws there and so do we."""
    if wq == "nvfp4":
        return 0.5625
    if wq in ("fp8_e4m3", "fp8_e5m2"):
        return 1.0
    if wq in ("int4_symmetric", "int4_asymmetric"):
        return 0.515625
    if wq in GGUF_UNIFORM_WEIGHT_QUANT:
        bb, qk = GGUF_TYPE_BLOCKS[GGUF_UNIFORM_WEIGHT_QUANT[wq]]
        return bb / qk
    if wq == "gguf":
        raise ValueError("generic 'gguf' is per-tensor mixed — size it with "
                         "packed_bytes(), not bytes_per_element")
    # config/schema.json quantization.weights is a CLOSED enum; anything else
    # is a value this transcription has never seen, and guessing 2 B/elem for
    # it would produce exactly the authoritative-looking wrong number this
    # module exists to prevent.
    raise UnmodelledArchitecture(
        f"quantization.weights={wq!r} has no transcribed bytes/element "
        "(layer_registry.cpp:55-90)")


def gating_bytes_per_element(gating_quant: str) -> float:
    """layer_registry.cpp:29-37 bytes_per_element(config::GatingQuant)."""
    return 2.0 if gating_quant == "fp16" else 4.0


# ── glm5_next: the checkpoint's real per-tensor widths (GF3.15) ─────────────
#
# The engine's own pre-scan is ``gguf_non_expert_widths_from_path``
# (weight_loader.cpp:2243): walk every shard's headers, keep the k-quant type
# of each NON-expert weight tensor keyed by (layer, TensorComponent), widest
# type wins on duplicates.  This is the same walk, keyed by the same component
# names, so ``glm5_next_attention_layer_bytes`` below can be the engine's
# formula verbatim.
#
# Only tensors the loader uploads PACKED appear here.  The ones it DEQUANTS or
# WIDENS at load keep their transformed width in the formula and are
# deliberately absent (weight_loader.h:230-234):
#   token_embd / output           -> BF16 (streamed dequant)
#   hc_{attn,ffn}_fn              -> F32  (launch_mhc_pre takes F32)
#   indexer_compressor_gate       -> BF16 (the executor GEMM is BF16-only)
#   attn_k_b + attn_v_b           -> one combined BF16 kv_b_proj (GLM-1)
#   *_norm / indexer.k_norm.*     -> BF16 (F32 halved on upload)
_GLM5_PACKED_COMPONENT = {
    # KDA linear-attention layers (MODELINFO §3a)
    "attn_q.weight":              "kda_q_proj",
    "attn_k.weight":              "kda_k_proj",
    "attn_v.weight":              "kda_v_proj",
    "ssm_beta.weight":            "kda_b_proj",
    "ssm_f_a.weight":             "kda_f_a_proj",
    "ssm_g_a.weight":             "kda_g_a_proj",
    "ssm_f_b.weight":             "kda_f_b_proj",
    "ssm_g_b.weight":             "kda_g_b_proj",
    # shared by both anatomies
    "attn_output.weight":         "o_proj",
    # sparse-MLA layers (MODELINFO §3b/§3c)
    "attn_q_a.weight":            "q_a_proj",
    "attn_q_b.weight":            "q_b_proj",
    "attn_kv_a_mqa.weight":       "kv_a_proj_with_mqa",
    "indexer.attn_q_b.weight":    "indexer_wq_b",
    "indexer.attn_k.weight":      "indexer_wk",
}

_BLK_RE = None


def glm5_next_packed_widths(first_shard: str) -> dict:
    """{(layer_idx, component): gguf_type} for the glm5_next attention tensors
    the loader uploads PACKED.

    Mirror of ``gguf_non_expert_widths_from_path`` (weight_loader.cpp:2243)
    narrowed to the components ``glm5_next_attention_layer_bytes`` looks up.
    Widest type wins across shards, exactly as the engine's ``rank()`` does."""
    import re
    global _BLK_RE
    if _BLK_RE is None:
        _BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
    out: dict = {}
    for shard in _shard_paths(first_shard):
        for t in read_tensor_infos(shard):
            if t.gguf_type not in KQUANT_GGUF_TYPES:
                continue          # plain float — the formula's bound stands
            m = _BLK_RE.match(t.name)
            if not m:
                continue
            comp = _GLM5_PACKED_COMPONENT.get(m.group(2))
            if comp is None:
                continue
            key = (int(m.group(1)), comp)
            prev = out.get(key)
            # rank() = gguf_packed_bytes(1, 256, t) — widest type wins.
            if prev is None or _rank_bytes(t.gguf_type) > _rank_bytes(prev):
                out[key] = t.gguf_type
    return out


def _rank_bytes(gguf_type: int) -> int:
    bb, qk = GGUF_TYPE_BLOCKS[gguf_type]
    return (256 // qk) * bb


def _packed_or_bf16(widths, layer_idx: int, component: str,
                    out_dim: int, in_dim: int) -> int:
    """pinned_region_layout.cpp:399-408 ``packed`` lambda: the checkpoint's
    real k-quant width when the pre-scan supplies it, the BF16 upper bound
    otherwise.  ``in_dim`` is the PER-RANK contraction extent — a TP split can
    break ``in % QK == 0``, and the engine then falls back to the bound rather
    than throwing out of a sizing pass."""
    if widths and layer_idx >= 0:
        ty = widths.get((layer_idx, component))
        if ty is not None:
            block_bytes, qk = GGUF_TYPE_BLOCKS[ty]
            if qk > 0 and in_dim % qk == 0:
                return out_dim * (in_dim // qk) * block_bytes
    return int(out_dim * in_dim * BF16_BYTES_PER_ELEMENT)


def _glm5_next_hc_bytes(m: dict) -> int:
    """pinned_region_layout.cpp:324-335 glm5_next_hc_bytes.  hc_*_fn is WIDENED
    to F32 at load (launch_mhc_pre's contract), so the slot carries the
    uploaded F32 width, not the checkpoint's packed one."""
    hc = max(1, int(m.get("hc_mult", 1)))
    hc_mix = (2 + hc) * hc
    fn = hc_mix * (hc * int(m["hidden_size"]))
    one_set = fn * 4 + hc_mix * 4 + 3 * 4
    return 2 * one_set


def glm5_next_attention_layer_bytes(m: dict, wq: str, linear_attention: bool,
                                    include_hc: bool, tp: int,
                                    widths: dict | None = None,
                                    layer_idx: int = -1) -> int:
    """pinned_region_layout.cpp:339-556 glm5_next_attention_layer_bytes.

    ``widths`` is the GF3.15 header pre-scan (``glm5_next_packed_widths``).
    Pass it and the slot is the checkpoint-EXACT size the engine reserves
    today; omit it and this is the pre-GF3.15 BF16 upper bound — 6.5 GiB
    larger on GLM-5.3-Flash, which is what the boot log of a pre-fix engine
    reports."""
    if wq == "nvfp4":
        raise UnmodelledArchitecture(
            "glm5_next + nvfp4 is not a supported artifact path (the engine "
            "throws here too — pinned_region_layout.cpp:352)")
    gguf_ckpt = is_gguf_weight_quant(wq)
    t = max(1, tp)
    hidden = int(m["hidden_size"])
    bf16, f32 = 2, 4
    b = 0

    def packed(component: str, out_dim: int, in_dim: int) -> int:
        return _packed_or_bf16(widths, layer_idx, component, out_dim, in_dim)

    if linear_attention:
        la = m.get("linear_attn_config") or {}
        H = int(la.get("num_heads", 64))
        D = int(la.get("head_dim", 128))
        K = int(la.get("short_conv_kernel_size", 4))
        HD = H * D
        b += packed("kda_q_proj", HD // t, hidden)
        b += packed("kda_k_proj", HD // t, hidden)
        b += packed("kda_v_proj", HD // t, hidden)
        b += packed("kda_b_proj", H // t, hidden)
        b += packed("kda_f_a_proj", D, hidden)      # replicated
        b += packed("kda_g_a_proj", D, hidden)      # replicated
        b += packed("kda_f_b_proj", HD // t, D)
        b += packed("kda_g_b_proj", HD // t, D)
        conv_bpe = f32 if gguf_ckpt else bf16
        o_norm_bpe = f32 if gguf_ckpt else bf16
        b += 3 * (HD // t) * K * conv_bpe           # q/k/v_conv1d [HD, 1, K]
        b += (H // t) * f32                         # A_log
        b += (HD // t) * f32                        # dt_bias
        b += D * o_norm_bpe                         # o_norm (replicated)
        b += packed("o_proj", hidden, HD // t)      # row-parallel
    elif gguf_ckpt:
        q_lora = int(m["q_lora_rank"])
        kv_lora = int(m["kv_lora_rank"])
        qk_head = int(m["qk_nope_head_dim"]) + int(m["qk_rope_head_dim"])
        heads = int(m["num_attention_heads"])
        q_b_out = heads * qk_head
        kv_a_out = kv_lora + int(m["qk_rope_head_dim"])
        kv_b_out = heads * (int(m["qk_nope_head_dim"]) + int(m["v_head_dim"]))
        o_in = heads * int(m["v_head_dim"])
        b += packed("q_a_proj", q_lora, hidden)           # replicated
        b += q_lora * bf16                                # q_a_layernorm F32->BF16
        b += packed("q_b_proj", q_b_out // t, q_lora)
        b += packed("kv_a_proj_with_mqa", kv_a_out, hidden)
        b += kv_lora * bf16                               # kv_a_layernorm F32->BF16
        # kv_b: attn_k_b/attn_v_b are dequanted, transposed and stacked into ONE
        # combined BF16 kv_b_proj at load (GLM-1) — BF16 is the uploaded width.
        b += (kv_b_out // t) * kv_lora * bf16
        b += packed("o_proj", hidden, o_in // t)          # row-parallel
        idx_h = int(m["index_n_heads"])
        idx_d = int(m["index_head_dim"])
        b += packed("indexer_wq_b", idx_h * idx_d, q_lora)
        b += packed("indexer_wk", idx_d, hidden)
        b += 2 * idx_d * bf16                             # k_norm weight + bias
        b += idx_h * hidden * f32                         # weights_proj F32
        if int(m.get("index_kpool", 1)) > 1:
            b += idx_d * hidden * bf16                    # compress_gate -> BF16
            b += int(m["index_kpool"]) * idx_d * f32      # compress_ape F32
    else:
        fp8 = wq in ("fp8_e4m3", "fp8_e5m2")
        proj_bpe = 1.0 if fp8 else weight_bytes_per_element(wq)
        q_lora = int(m["q_lora_rank"])
        kv_lora = int(m["kv_lora_rank"])
        qk_head = int(m["qk_nope_head_dim"]) + int(m["qk_rope_head_dim"])
        heads = int(m["num_attention_heads"])
        q_b_out = heads * qk_head
        kv_a_out = kv_lora + int(m["qk_rope_head_dim"])
        kv_b_out = heads * (int(m["qk_nope_head_dim"]) + int(m["v_head_dim"]))
        o_in = heads * int(m["v_head_dim"])

        def qbytes(n: int, k: int) -> int:
            return int(n * k * proj_bpe)

        def scale_bytes(n_rows: int, k_cols: int, row_tp: int, col_tp: int) -> int:
            """pinned_region_layout.cpp:314-318 glm_fp8_scale_bytes."""
            return (-(-n_rows // _FP8_BLOCK_SCALE_TILE) // row_tp) * \
                   (-(-k_cols // _FP8_BLOCK_SCALE_TILE) // col_tp) * 4

        b += qbytes(q_lora, hidden)
        if fp8:
            b += scale_bytes(q_lora, hidden, 1, 1)
        b += q_lora * bf16
        b += qbytes(q_b_out // t, q_lora)
        if fp8:
            b += scale_bytes(q_b_out, q_lora, t, 1)
        b += qbytes(kv_a_out, hidden)
        if fp8:
            b += scale_bytes(kv_a_out, hidden, 1, 1)
        b += kv_lora * bf16
        b += (kv_b_out // t) * kv_lora * bf16
        b += qbytes(hidden, o_in // t)
        if fp8:
            b += scale_bytes(hidden, o_in, 1, t)
        idx_h = int(m["index_n_heads"])
        idx_d = int(m["index_head_dim"])
        b += idx_h * idx_d * q_lora * bf16
        b += idx_d * hidden * bf16
        b += 2 * idx_d * bf16
        b += idx_h * hidden * bf16
        if int(m.get("index_kpool", 1)) > 1:
            b += idx_d * hidden * bf16
            b += int(m["index_kpool"]) * idx_d * bf16

    if include_hc:
        b += _glm5_next_hc_bytes(m)
    return align16(b)


def compute_attn_dims(m: dict) -> dict:
    """pinned_region_layout.cpp:42-73 compute_attn_dims."""
    hidden = int(m["hidden_size"])
    heads = int(m["num_attention_heads"])
    qk_head = int(m["qk_nope_head_dim"]) + int(m["qk_rope_head_dim"])
    d = {
        "q_a_params": hidden * int(m["q_lora_rank"]),
        "q_b_params": int(m["q_lora_rank"]) * heads * qk_head,
        "kv_a_params": hidden * (int(m["kv_lora_rank"]) + int(m["qk_rope_head_dim"])),
        "kv_b_params": int(m["kv_lora_rank"]) * heads *
                       (int(m["qk_nope_head_dim"]) + int(m["v_head_dim"])),
        "o_params": heads * int(m["v_head_dim"]) * hidden,
        "q_a_norm_params": int(m["q_lora_rank"]),
        "kv_a_norm_params": int(m["kv_lora_rank"]),
        "indexer_params": 0,
        "indexer_norm_params": 0,
        "q_a_in": hidden,
        "q_b_in": int(m["q_lora_rank"]),
        "kv_a_in": hidden,
        "kv_b_in": int(m["kv_lora_rank"]),
        "o_in": heads * int(m["v_head_dim"]),
    }
    if int(m.get("index_topk", 0)) > 0:
        idx_h, idx_d = int(m["index_n_heads"]), int(m["index_head_dim"])
        q_idx_b = int(m["q_lora_rank"]) * idx_h * idx_d
        k_idx = hidden * idx_d
        weights_proj = hidden * idx_h
        d["indexer_params"] = q_idx_b + k_idx + idx_d + idx_d + weights_proj
        d["indexer_norm_params"] = 2 * idx_d
    return d


def attention_layer_bytes(d: dict, wq: str, include_kv_b: bool, tp: int,
                          include_indexer: bool = True,
                          force_bf16_oproj: bool = False) -> int:
    """pinned_region_layout.cpp:126-213 attention_layer_bytes (the non-glm5,
    non-V4 MLA path: DeepSeek V3.2, GLM-5.2)."""
    t = max(1, tp)
    bf16 = BF16_BYTES_PER_ELEMENT

    def indexer_bytes(bpe: float) -> int:
        if not (include_indexer and d["indexer_params"] > 0):
            return 0
        norm = d["indexer_norm_params"]
        return int((d["indexer_params"] - norm) * bpe) + norm * 2

    if is_gguf_weight_quant(wq):
        if wq == "gguf":
            # Generic gguf is per-tensor mixed and unreadable at plan time —
            # the engine sizes the whole slot at the BF16 upper bound.
            replicated = int((d["q_a_params"] + d["kv_a_params"]) * bf16)
            norms = (d["q_a_norm_params"] + d["kv_a_norm_params"]) * 2
            shardable = d["q_b_params"] + (d["kv_b_params"] if include_kv_b else 0)
            total = (replicated + norms + indexer_bytes(bf16)
                     + int(shardable * bf16) // t
                     + int(d["o_params"] * bf16) // t)
            return align16(total)
        # Uniform gguf_qX_k / gguf_q8_0: exact packed bytes per projection
        # (pinned_region_layout.cpp:87-124 gguf_attention_layer_bytes).
        ty = GGUF_UNIFORM_WEIGHT_QUANT[wq]

        def pk(params: int, in_features: int) -> int:
            out_features = params // in_features if in_features > 0 else 0
            return packed_bytes(out_features, in_features, ty)

        replicated = pk(d["q_a_params"], d["q_a_in"]) + pk(d["kv_a_params"], d["kv_a_in"])
        norms = (d["q_a_norm_params"] + d["kv_a_norm_params"]) * 2
        total = (replicated + norms + indexer_bytes(bf16)
                 + pk(d["q_b_params"], d["q_b_in"]) // t
                 + (pk(d["kv_b_params"], d["kv_b_in"]) // t if include_kv_b else 0)
                 + pk(d["o_params"], d["o_in"]) // t)
        return align16(total)

    nvfp4 = (wq == "nvfp4")
    proj_bpe = bf16 if nvfp4 else weight_bytes_per_element(wq)
    replicated = int((d["q_a_params"] + d["kv_a_params"]) * proj_bpe)
    norms = (d["q_a_norm_params"] + d["kv_a_norm_params"]) * 2
    shardable_params = d["q_b_params"] + (d["kv_b_params"] if include_kv_b else 0)
    shardable = int(shardable_params * proj_bpe) // t
    if nvfp4 and not force_bf16_oproj:
        o_bytes = ((d["o_params"] + 1) // 2) // t + ((d["o_params"] + 15) // 16) // t + 8
    else:
        o_bpe = bf16 if force_bf16_oproj else proj_bpe
        o_bytes = int(d["o_params"] * o_bpe) // t
    return align16(replicated + norms + indexer_bytes(bf16) + shardable + o_bytes)


def _ffn_projection_bytes(wq: str, hidden: int, intermediate: int,
                          proj: str, gguf_q8_upper_bound: bool) -> int:
    """pinned_upload_plan.cpp:168-192 ``proj_bytes``.

    gate/up use n=intermediate,k=hidden; down uses n=hidden,k=intermediate
    (GgufQuantInterface::projection_nk).  Under the generic ``gguf`` quant the
    engine sizes shared-expert and dense-FFN at Q8_0 — the widest k-quant —
    because the real per-owner types are not known before the load
    (TD-GG9-REGION-Q8-UPPER-BOUND)."""
    n = hidden if proj == "down" else intermediate
    k = intermediate if proj == "down" else hidden
    if gguf_q8_upper_bound and wq == "gguf":
        return packed_bytes(n, k, _Q8_0)
    if wq in GGUF_UNIFORM_WEIGHT_QUANT:
        return packed_bytes(n, k, GGUF_UNIFORM_WEIGHT_QUANT[wq])
    params = n * k
    if wq == "nvfp4":
        raw = (params + 1) // 2 + (params + 15) // 16 + 8
        return (raw + 127) & ~127            # nvfp4.cpp:25-46 (align 128)
    if wq in ("fp8_e4m3", "fp8_e5m2"):
        nb = -(-n // _FP8_BLOCK_SCALE_TILE)
        kb = -(-k // _FP8_BLOCK_SCALE_TILE)
        return params + nb * kb * 4          # fp8.cpp:25-51
    return int(params * weight_bytes_per_element(wq))


@dataclass(frozen=True)
class PinnedRegionModel:
    """One modelled pinned region, with the honesty metadata a cross-check
    needs: is this figure EXACT against the engine, and if not, which way does
    it err?"""
    total_bytes: int
    by_component: dict
    provenance: str
    exact: bool
    direction: str            # "exact" | "over" | "under"

    @property
    def total_mib(self) -> float:
        return self.total_bytes / (1 << 20)


# The solver's own pinned-layer policy (solver.py:1275).  It is a CONSTANT of
# the derivation, not an input, so the model defaults to it when no recipe
# section is supplied.
def default_pinned_layers(model: dict) -> dict:
    return {"attention": "all", "gating": "all", "embedding": True,
            "output_head": True,
            "dense_ffn_layers": list(range(int(model.get("first_k_dense_replace", 0))))}


def _is_pinned(spec, layer: int) -> bool:
    """pinned_upload_plan.cpp:127-132 ``is_pinned``."""
    if isinstance(spec, str):
        return spec == "all"
    return layer in (spec or ())


def _is_moe_layer(m: dict, layer: int) -> bool:
    """model_config.cpp compute_layer_counts (mirrored by
    ModelShape.is_moe_layer)."""
    fk = int(m.get("first_k_dense_replace", 0))
    if layer < fk:
        return False
    return (layer - fk) % max(1, int(m.get("moe_layer_freq", 1))) == 0


def pinned_region_layout(model: dict, quantization: dict,
                         tensor_parallelism: int = 1,
                         first_shard: str = "",
                         pinned_layers: dict | None = None,
                         tp_mode_per_layer: dict | None = None,
                         attention_arm: str = "auto") -> PinnedRegionModel:
    """The engine's pinned VRAM region for one TP rank, slot for slot.

    Transcription of ``build_upload_plan`` (pinned_upload_plan.cpp:76) summed
    by ``compute_pinned_layout`` (pinned_region_layout.cpp:565), including its
    ``kUploadAlignBudget`` tail.  Divergence between this function and those
    two is a bug HERE — the engine is authoritative.

    ``attention_arm``:
      * ``"auto"``        — what the engine does TODAY: checkpoint-exact
                            attention slots on the glm5_next GGUF path when
                            ``first_shard`` is readable, the formula's own
                            bound everywhere else;
      * ``"upper_bound"`` — force the pre-GF3.15 BF16 bound (what a pre-fix
                            engine's boot log reports).

    Raises ``UnmodelledArchitecture`` for architectures whose slot anatomy is
    not transcribed here (deepseek_v4), rather than returning a number that
    would look authoritative and be wrong."""
    m = model
    arch = str(m.get("architecture", ""))
    if arch == "deepseek_v4":
        raise UnmodelledArchitecture(
            "deepseek_v4 pinned slots (grouped o_proj / per-layer CSA-HCA-SWA "
            "anatomy / hash gating tables / model-level output_hc, "
            "pinned_region_layout.cpp:230-298) are not transcribed here")
    wq = str(quantization.get("weights", ""))
    glm5 = (arch == "glm5_next")
    tp = max(1, int(tensor_parallelism))
    pin = pinned_layers or default_pinned_layers(m)
    tm = tp_mode_per_layer or {}

    def resolve_tp(mode) -> int:
        return int(mode) if mode else tp

    embed_tp = resolve_tp(tm.get("embedding"))
    outhead_tp = resolve_tp(tm.get("output_head"))
    shared_tp = resolve_tp(tm.get("shared_expert"))
    dense_tp = resolve_tp(tm.get("pinned_dense_ffn"))

    widths = None
    arm = "engine formula (config-only)"
    if glm5 and is_gguf_weight_quant(wq) and attention_arm != "upper_bound" \
            and first_shard and os.path.exists(first_shard):
        widths = glm5_next_packed_widths(first_shard)
        arm = "checkpoint-exact attention slots (GF3.15 header pre-scan)"
    elif glm5 and is_gguf_weight_quant(wq):
        arm = "BF16 upper-bound attention slots (pre-GF3.15)"

    hidden = int(m["hidden_size"])
    vocab = int(m["vocab_size"])
    n_hidden = int(m["num_hidden_layers"])
    layer_types = tuple(m.get("layer_types", ()) or ())
    by: dict = {}

    def emit(component: str, size: int) -> None:
        by[component] = by.get(component, 0) + size

    # 1/2. Embedding + output head (pinned_upload_plan.cpp:134-151).  A k-quant
    # embed/lm_head is STREAM-DEQUANTED to BF16 at upload, so 2 B/elem is the
    # exact uploaded width, not slack.  has_output_head_bias() is false
    # unconditionally (pinned_region_layout.cpp:33).
    embed_total = int(vocab * hidden * EMBEDDING_BYTES_PER_ELEMENT)
    if pin.get("embedding", True):
        emit("embedding", embed_total // embed_tp)
    if pin.get("output_head", True):
        emit("output_head_weight", embed_total // outhead_tp)

    # 3. Per hidden layer.
    per_layer_norm = 2 * hidden * 2
    gate_weight = int(hidden * int(m["n_routed_experts"])
                      * gating_bytes_per_element(
                          str(quantization.get("gating_compute", "fp32"))))
    gate_bias = int(m["n_routed_experts"]) * 4
    rep_per_proj = 8 if wq == "nvfp4" else 0
    gguf_generic = (wq == "gguf")

    moe_int = int(m["moe_intermediate_size"])
    dense_int = int(m["intermediate_size"])
    n_shared = int(m.get("n_shared_experts", 0))
    se = {p: _ffn_projection_bytes(wq, hidden, moe_int, p, gguf_generic)
          for p in ("gate", "up", "down")}
    dn = {p: _ffn_projection_bytes(wq, hidden, dense_int, p, gguf_generic)
          for p in ("gate", "up", "down")}
    se_sharded = {p: ((v - rep_per_proj) // shared_tp + rep_per_proj) * n_shared
                  for p, v in se.items()}
    dn_sharded = {p: (v - rep_per_proj) // dense_tp + rep_per_proj
                  for p, v in dn.items()}

    attn_dims = compute_attn_dims(m) if not glm5 else None
    include_kv_b = int(m.get("kv_lora_rank", 0)) > 0     # has_kv_b_in_checkpoint
    if not glm5:
        attn_per_layer = attention_layer_bytes(attn_dims, wq, include_kv_b, tp)

    dense_ffn_layers = pin.get("dense_ffn_layers") or []
    for l in range(n_hidden):
        is_moe = _is_moe_layer(m, l)
        if _is_pinned(pin.get("attention", "all"), l):
            if glm5:
                linear = (l < len(layer_types)
                          and layer_types[l] == "linear_attention")
                emit("attention", glm5_next_attention_layer_bytes(
                    m, wq, linear, True, tp, widths, l))
            else:
                emit("attention", attn_per_layer)
            emit("layer_norm", per_layer_norm)
        if is_moe and _is_pinned(pin.get("gating", "all"), l):
            emit("gating_weight", gate_weight)
            emit("gating_bias", gate_bias)
        if is_moe:
            for p in ("gate", "up", "down"):
                emit("shared_expert_" + p, se_sharded[p])
        elif l in dense_ffn_layers:
            for p in ("gate", "up", "down"):
                emit("dense_ffn_" + p, dn_sharded[p])

    # 4. Final norm (F32->BF16 on upload).
    emit("final_norm", hidden * 2)

    # 5. MTP block layers (pinned_upload_plan.cpp:334-425).  glm5_next emits
    # NO mtp_embed_tokens and NO mtp_shared_head_weight — the block shares
    # model.embed_tokens and lm_head (MODELINFO §5).
    n_mtp = int(m.get("num_nextn_predict_layers", 0))
    if n_mtp > 0:
        mtp_bf16_oproj = (wq == "nvfp4")
        for mi in range(n_mtp):
            ml = n_hidden + mi
            if glm5:
                emit("attention", glm5_next_attention_layer_bytes(
                    m, wq, False, False, tp, widths, ml))
            else:
                emit("attention", attention_layer_bytes(
                    attn_dims, wq, include_kv_b, tp,
                    int(m.get("index_topk", 0)) > 0, mtp_bf16_oproj))
            emit("layer_norm", per_layer_norm)
            emit("gating_weight", gate_weight)
            emit("gating_bias", gate_bias)
            if mtp_bf16_oproj:
                # V3.2 NVFP4 checkpoints store the MTP shared expert as BF16.
                emit("shared_expert_gate", moe_int * hidden * 2 // shared_tp * n_shared)
                emit("shared_expert_up", moe_int * hidden * 2 // shared_tp * n_shared)
                emit("shared_expert_down", hidden * moe_int * 2 // shared_tp * n_shared)
            else:
                for p in ("gate", "up", "down"):
                    emit("shared_expert_" + p, se_sharded[p])
            if not glm5:
                emit("mtp_embed_tokens", vocab * hidden * 2 // embed_tp)
                emit("mtp_shared_head_weight", vocab * hidden * 2 // outhead_tp)
            emit("mtp_shared_head_norm", hidden * 2)
            emit("mtp_eh_proj", hidden * 2 * hidden * 2 // tp)
            emit("mtp_enorm", hidden * 2)
            emit("mtp_hnorm", hidden * 2)

    total = sum(by.values()) + UPLOAD_ALIGN_BUDGET

    # Honesty bookkeeping: which slots are byte-exact against the engine, and
    # which are a bound (and in which direction).
    exact = True
    direction = "exact"
    notes = []
    if gguf_generic:
        # TD-GG9-REGION-Q8-UPPER-BOUND: shared/dense sized at Q8_0.  This
        # matches the ENGINE exactly (which is what the solver must model);
        # it over-reserves against the checkpoint only when the real tensors
        # are narrower than Q8_0.
        notes.append("shared/dense at Q8_0 (the engine's own pre-load bound, "
                     "TD-GG9-REGION-Q8-UPPER-BOUND; exact when those tensors "
                     "really are Q8_0, as on GLM-5.3-Flash)")
    if glm5 and is_gguf_weight_quant(wq) and widths is None:
        exact = False
        direction = "over"
        why = ("forced" if attention_arm == "upper_bound"
               else "the checkpoint was not readable")
        notes.append(f"attention at the BF16 upper bound ({why}) — this is the "
                     "PRE-GF3.15 engine's figure and OVER-estimates a "
                     "post-GF3.15 one by ~6.5 GiB on GLM-5.3-Flash")
    prov = (f"engine pinned region, tp={tp} ({arm}; "
            f"pinned_upload_plan.cpp:76 + pinned_region_layout.cpp:565)")
    if notes:
        prov += " — " + "; ".join(notes)
    return PinnedRegionModel(total_bytes=total, by_component=by,
                             provenance=prov, exact=exact, direction=direction)


def pinned_region_bytes(first_shard: str, model: dict, quantization: dict,
                        tensor_parallelism: int = 1,
                        pinned_layers: dict | None = None,
                        tp_mode_per_layer: dict | None = None,
                        attention_arm: str = "auto") -> tuple:
    """``(bytes, provenance)`` — the engine's pinned region for one TP rank.
    Thin wrapper over :func:`pinned_region_layout`."""
    lay = pinned_region_layout(model, quantization, tensor_parallelism,
                               first_shard, pinned_layers, tp_mode_per_layer,
                               attention_arm)
    return lay.total_bytes, lay.provenance

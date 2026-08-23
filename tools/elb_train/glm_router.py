"""GLM-5.2 router semantics (CPU reference) + standalone router-weight loader.

EPM-2 (Phase 29, spec/MoE-SpeQ_NOTES.md §7): the B1 Tier-0 baseline applies
the FROZEN target routers to raw draft hiddens, so this module replicates the
engine's noaux_tc top-8 gating EXACTLY and loads the router tensors straight
from a safetensors checkpoint without pulling the engine in.

noaux_tc semantics (mirrors the engine's simple_topk_kernel, n_group=1 —
deps/LayerStoRmExpertKernels/csrc/sm120/gating/topk_gating.cu, CPU reference
tests/unit/topk_gating_test.cpp topk_gating_ref):
  1. scores  = sigmoid(logits)          (numerically stable tanh identity)
  2. sel     = scores + e_score_correction_bias   (selection ONLY)
  3. top-8   = argmax-8 of sel, ties broken by LOWER expert index,
               output ranked by sel descending
  4. weights = UNBIASED scores of the selected experts
  5. norm_topk_prob: weights *= routed_scaling_factor / sum(weights)
     (sum > 0 guard; GLM-5.2: routed_scaling_factor = 2.5)

Router tensors in the HF checkpoint (src/model/weight_loader/tensor_id.cpp):
  model.layers.{j}.mlp.gate.weight                  [E=256, H=6144]  (BF16)
  model.layers.{j}.mlp.gate.e_score_correction_bias [E=256]          (F32)
Router projection = plain GEMV: logits = hidden @ weight.T (no bias term in
the projection; the correction bias enters only at selection, step 2).

The safetensors reader here parses only the 8-byte header-length prefix +
JSON header and preads the exact byte ranges of the requested tensors — no
safetensors package, no engine.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from pathlib import Path

import numpy as np

GLM52_TOPK = 8
GLM52_ROUTED_SCALING_FACTOR = 2.5
GLM52_AUX_LAYER_IDS = (8, 23, 39, 55, 70)  # aux_hidden_state_layer_ids


# ── BF16 <-> F32 bit helpers ─────────────────────────────────────────────────

def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """uint16 raw BF16 bits -> float32 (exact: BF16 is truncated F32)."""
    b = np.ascontiguousarray(bits, dtype=np.uint16)
    return (b.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> uint16 BF16 bits, round-to-nearest-even (hardware
    convention; used by the synthetic-corpus writer to mirror the engine's
    BF16 feature dump)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    # RNE: add 0x7FFF + LSB-of-result before truncating.
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    out = (rounded >> 16).astype(np.uint16)
    # NaN must stay NaN (rounding could overflow the exponent of inf/nan).
    nan = np.isnan(u.view(np.float32))
    if np.any(nan):
        out = np.where(nan, np.uint16(0x7FC0), out)
    return out


# ── noaux_tc CPU reference ───────────────────────────────────────────────────

def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """0.5 * tanh(0.5 x) + 0.5 — the engine's stable_sigmoid, bit-for-bit
    the same formulation (topk_gating.cu)."""
    x = np.asarray(x, dtype=np.float32)
    return (0.5 * np.tanh(0.5 * x) + 0.5).astype(np.float32)


def rank_experts(logits: np.ndarray, bias: np.ndarray | None,
                 m: int) -> np.ndarray:
    """Top-m expert ids ranked by the noaux_tc SELECTION score (sigmoid +
    correction bias) descending, ties -> lower expert index.  logits
    [..., E] float; returns int32 [..., m].  m may exceed topk=8 — this is
    the recall@m prediction list (over-predicting is cheap downstream)."""
    logits = np.asarray(logits, dtype=np.float32)
    sel = stable_sigmoid(logits)
    if bias is not None:
        sel = sel + np.asarray(bias, dtype=np.float32)
    # kind='stable' on -sel: equal scores keep ascending index order.
    order = np.argsort(-sel, axis=-1, kind="stable")
    return order[..., :m].astype(np.int32)


def noaux_tc_topk(logits: np.ndarray, bias: np.ndarray | None,
                  topk: int = GLM52_TOPK,
                  routed_scaling_factor: float = GLM52_ROUTED_SCALING_FACTOR,
                  renormalize: bool = True,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """The full engine gating output: (ids int32 [..., topk], weights f32
    [..., topk]).  Selection by biased score, weights from UNBIASED sigmoid
    scores, renormalized to sum == routed_scaling_factor (see module doc)."""
    logits = np.asarray(logits, dtype=np.float32)
    scores = stable_sigmoid(logits)
    ids = rank_experts(logits, bias, topk)
    w = np.take_along_axis(scores, ids, axis=-1).astype(np.float32)
    if renormalize:
        s = w.sum(axis=-1, keepdims=True)
        w = np.where(s > 0.0, w * (routed_scaling_factor / np.where(
            s > 0.0, s, 1.0)), w).astype(np.float32)
    return ids, w


# ── Standalone safetensors router loader ─────────────────────────────────────

_ST_DTYPES = {
    "F32": (np.float32, 4), "F16": (np.float16, 2), "BF16": (np.uint16, 2),
    "F64": (np.float64, 8), "I64": (np.int64, 8), "I32": (np.int32, 4),
    "U16": (np.uint16, 2), "U8": (np.uint8, 1),
    # fp8 returned as RAW e4m3 bits (numpy has no fp8); consumers
    # dequantize via torch .view(float8_e4m3fn) + the *.scale tensors.
    "F8_E4M3": (np.uint8, 1),
}


def _read_st_header(path: Path) -> tuple[dict, int]:
    """(header dict, data-section offset) of a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def read_safetensors_tensor(path: str | Path, name: str,
                            _hdr_cache: dict | None = None) -> np.ndarray:
    """Read ONE tensor from a safetensors file by pread of its byte range.
    BF16 tensors are returned as float32 (exact); F16/F32 as-is."""
    path = Path(path)
    if _hdr_cache is not None and path in _hdr_cache:
        hdr, base = _hdr_cache[path]
    else:
        hdr, base = _read_st_header(path)
        if _hdr_cache is not None:
            _hdr_cache[path] = (hdr, base)
    if name not in hdr:
        raise KeyError(f"{path}: tensor {name!r} not in header")
    meta = hdr[name]
    dtype = meta["data_type"] if "data_type" in meta else meta["dtype"]
    if dtype not in _ST_DTYPES:
        raise ValueError(f"{path}: {name}: unsupported dtype {dtype}")
    np_dtype, itemsize = _ST_DTYPES[dtype]
    shape = tuple(meta["shape"])
    beg, end = meta["data_offsets"]
    count = int(np.prod(shape)) if shape else 1
    if end - beg != count * itemsize:
        raise ValueError(f"{path}: {name}: offsets/shape mismatch")
    with open(path, "rb") as f:
        f.seek(base + beg)
        raw = f.read(end - beg)
    arr = np.frombuffer(raw, np_dtype).reshape(shape)
    if dtype == "BF16":
        arr = bf16_bits_to_f32(arr).reshape(shape)
    return arr


@dataclasses.dataclass
class RouterBank:
    """Per-MoE-layer frozen router replicas.

    moe_layers: int32 [J] absolute layer ids (sorted, matches the shard
                index's `moe_layers`)
    weight:     float32 [J, E, H] gate weights
    bias:       float32 [J, E] e_score_correction_bias (zeros if absent)
    """
    moe_layers: np.ndarray
    weight: np.ndarray
    bias: np.ndarray

    @property
    def n_experts(self) -> int:
        return self.weight.shape[1]

    @property
    def hidden(self) -> int:
        return self.weight.shape[2]

    def logits(self, hiddens: np.ndarray) -> np.ndarray:
        """Router projection for all J layers: hiddens [..., H] float32 ->
        logits [..., J, E].  Raw hiddens in, no normalization (Tier-0
        contract: normalization choices are downstream concerns)."""
        h = np.asarray(hiddens, dtype=np.float32)
        # [..., H] x [J, E, H] -> [..., J, E]
        return np.einsum("...h,jeh->...je", h, self.weight,
                         optimize=True).astype(np.float32)

    def save_npz(self, path: str | Path) -> None:
        np.savez(path, moe_layers=self.moe_layers, weight=self.weight,
                 bias=self.bias)

    @classmethod
    def from_npz(cls, path: str | Path) -> "RouterBank":
        z = np.load(path)
        return cls(moe_layers=z["moe_layers"].astype(np.int32),
                   weight=z["weight"].astype(np.float32),
                   bias=z["bias"].astype(np.float32))

    @classmethod
    def from_checkpoint(cls, model_dir: str | Path,
                        moe_layers: list[int] | np.ndarray,
                        weight_tmpl: str =
                        "model.layers.{j}.mlp.gate.weight",
                        bias_tmpl: str =
                        "model.layers.{j}.mlp.gate.e_score_correction_bias",
                        ) -> "RouterBank":
        """Load gate weight + selection-bias tensors for the given absolute
        layer ids from a HF safetensors checkpoint (single-file or sharded
        with model.safetensors.index.json). Tensor names are templates with
        `{j}` = absolute layer id (defaults: DeepSeek/GLM family); a model
        without a bias tensor gets zeros (KeyError path)."""
        model_dir = Path(model_dir)
        idx_path = model_dir / "model.safetensors.index.json"
        if idx_path.is_file():
            with open(idx_path, encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]

            def shard_of(name: str) -> Path:
                if name not in weight_map:
                    raise KeyError(f"{idx_path}: {name!r} not in weight_map")
                return model_dir / weight_map[name]
        else:
            single = model_dir / "model.safetensors"
            if not single.is_file():
                raise FileNotFoundError(
                    f"{model_dir}: neither model.safetensors.index.json "
                    f"nor model.safetensors")

            def shard_of(name: str) -> Path:  # noqa: ARG001
                return single

        layers = np.asarray(sorted(int(l) for l in moe_layers), np.int32)
        hdr_cache: dict = {}
        weights, biases = [], []
        for j in layers:
            wname = weight_tmpl.format(j=j)
            bname = bias_tmpl.format(j=j)
            w = read_safetensors_tensor(shard_of(wname), wname, hdr_cache)
            weights.append(np.asarray(w, np.float32))
            try:
                b = read_safetensors_tensor(shard_of(bname), bname, hdr_cache)
                biases.append(np.asarray(b, np.float32))
            except KeyError:
                biases.append(np.zeros(w.shape[0], np.float32))
        return cls(moe_layers=layers,
                   weight=np.stack(weights).astype(np.float32),
                   bias=np.stack(biases).astype(np.float32))


def aux_prior_tap(moe_layers: np.ndarray | list[int],
                  aux_layer_ids: tuple[int, ...] = GLM52_AUX_LAYER_IDS,
                  ) -> np.ndarray:
    """The aux-segment PRIOR tap mapping (spec/MoE-SpeQ_NOTES.md §3): draft
    layer i is the natural feature source for target segment
    [aux_i, aux_{i+1}); segment 0 also covers the MoE layers below aux_0
    (GLM-5.2: layers 3..7).  Returns int32 [J] tap index per layer."""
    aux = np.asarray(aux_layer_ids)
    layers = np.asarray(moe_layers)
    taps = np.searchsorted(aux, layers, side="right") - 1
    return np.clip(taps, 0, len(aux) - 1).astype(np.int32)

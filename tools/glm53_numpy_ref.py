#!/usr/bin/env python3
"""NumPy reference forward pass for GLM-5.3-Flash (glm5_next), fp32 throughout.

Debug artifact for the LayerStoRm engine's first boot (PLAN GF3.9): an
independent, dependency-light transcription of the model that the engine's
seam dumps can be diffed against.

Sources (all read, not guessed):
  * vLLM tag ``pr-53906-head`` (ref/vllm):
      - vllm/model_executor/kernels/mhc/torch.py   (mhc_pre_torch/mhc_post_torch)
      - vllm/model_executor/layers/mhc.py          (hc_expand/hc_contract, norm fusion)
      - vllm/models/glm5next/nvidia/model.py       (per-layer mHC flow, MLP, MoE)
      - vllm/models/glm5next/nvidia/attention.py   (MLA scale = qk_head_dim**-0.5)
      - vllm/models/glm5next/nvidia/kda.py         (KDA wiring)
      - vllm/model_executor/layers/activation.py   (SiluAndMulWithClamp)
      - vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py
    Copyright contributors to the vLLM project (Apache-2.0).
  * test-data/kda-goldens/manifest.json + tests/unit/kda_reference.h (KDA scan).
  * src/model/weight_loader/weight_loader.cpp (attn_k_b/attn_v_b orientation).

Usage:
    .venv/bin/python tools/glm53_numpy_ref.py [--gguf DIR] [--skip-goldens]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# Constants (test-data/GLM-5.3-Flash/config.json :: text_config)
# --------------------------------------------------------------------------
HIDDEN = 4096
N_LAYERS = 45              # blk.45 is the MTP layer and is NOT run
VOCAB = 154880
RMS_EPS = 1e-5

HC_MULT = 4                # hc_mult / mhc_num_residual_streams
HC_EPS = 1e-6              # hc_eps (both pre-mix eps and sinkhorn eps)
HC_SINKHORN_ITERS = 20
HC_POST_MULT = 2.0         # mhc_post_mult_value

KDA_HEADS = 64
KDA_HEAD_DIM = 128
KDA_CONV_W = 4
KDA_SCALE = np.float32(0.0883883461356163)   # 128 ** -0.5, pinned by the goldens
KDA_LOWER_BOUND = np.float32(-5.0)
L2NORM_EPS = np.float32(1e-6)

MLA_HEADS = 64
QK_NOPE = 256
V_HEAD_DIM = 256
KV_LORA = 512
Q_LORA = 1536
MLA_SCALE = np.float32((QK_NOPE + 0) ** -0.5)   # qk_head_dim**-0.5, qk_rope_head_dim = 0

SWIGLU_LIMIT = np.float32(10.0)
N_ROUTED = 288
TOP_K = 8
ROUTED_SCALE = np.float32(2.5)
FIRST_K_DENSE = 3
FULL_ATTN_LAYERS = frozenset([3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43])

PROMPT_IDS = [785, 6722, 315, 9621, 374]   # "The capital of France is"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GGUF = "/srv/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q4_K_XL"
GOLDEN_DIR = os.path.join(REPO, "test-data", "kda-goldens")
TOKENIZER_JSON = os.path.join(REPO, "test-data", "GLM-5.3-Flash", "tokenizer.json")


def log(msg: str) -> None:
    print(msg, flush=True)


def rms(x) -> float:
    a = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


# ==========================================================================
# Primitive ops
# ==========================================================================
def rmsnorm(x, w, eps=RMS_EPS):
    """RMSNorm over the last axis (vLLM RMSNorm: fp32 variance, then weight mul)."""
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x / np.sqrt(var + np.float32(eps))).astype(np.float32) * w.astype(np.float32)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def silu(x):
    return (x * sigmoid(x)).astype(np.float32)


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp((x - m).astype(np.float32))
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def linear(x, w):
    """y = x @ w.T with w in row-major numpy order [out, in]."""
    return (x.astype(np.float32) @ w.astype(np.float32).T).astype(np.float32)


def swiglu_clamped(gate, up, limit=SWIGLU_LIMIT):
    """vLLM SiluAndMulWithClamp.forward_native (alpha=1.0, beta=0.0)."""
    g = np.minimum(gate, limit)
    u = np.clip(up, -limit, limit)
    return (g * sigmoid(g) * u).astype(np.float32)


# ==========================================================================
# mHC (multi-head hyper-connections) -- port of vllm mhc/torch.py
# ==========================================================================
def hc_expand(x, n=HC_MULT):
    """[s, hidden] -> [s, n, hidden] by replication."""
    return np.repeat(x[:, None, :], n, axis=1).astype(np.float32)


def hc_contract(x):
    """[s, n, hidden] -> [s, hidden] by averaging."""
    return np.mean(x.astype(np.float32), axis=1, dtype=np.float32)


def mhc_pre(residual, fn, hc_scale, hc_base, norm_weight):
    """mhc_pre_torch + the fused output RMSNorm (MHCPreOp._apply_mhc_norm).

    residual: [s, n, hidden];  fn: [2n + n*n, n*hidden];  hc_scale: [3];
    hc_base: [2n + n*n].
    Returns (post_mix [s, n, 1], comb_mix [s, n, n], layer_input [s, hidden]).
    """
    s, n, hidden = residual.shape
    x = residual.reshape(s, n * hidden).astype(np.float32)

    mixes = (x @ fn.T).astype(np.float32)
    sqrsum = np.sum(x * x, axis=-1, keepdims=True, dtype=np.float32)
    mixes = mixes * (1.0 / np.sqrt(sqrsum / np.float32(n * hidden) + np.float32(RMS_EPS)))
    mixes = mixes.astype(np.float32)

    pre_mix = sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n]) + np.float32(HC_EPS)
    post_mix = sigmoid(mixes[:, n:2 * n] * hc_scale[1] + hc_base[n:2 * n]) * np.float32(HC_POST_MULT)

    comb_logits = (mixes[:, 2 * n:].reshape(s, n, n) * hc_scale[2]
                   + hc_base[2 * n:].reshape(1, n, n)).astype(np.float32)
    comb = softmax(comb_logits, axis=-1) + np.float32(HC_EPS)
    comb = comb / (np.sum(comb, axis=-2, keepdims=True) + np.float32(HC_EPS))
    for _ in range(HC_SINKHORN_ITERS - 1):
        comb = comb / (np.sum(comb, axis=-1, keepdims=True) + np.float32(HC_EPS))
        comb = comb / (np.sum(comb, axis=-2, keepdims=True) + np.float32(HC_EPS))
    comb = comb.astype(np.float32)

    layer_input = np.sum(pre_mix[:, :, None] * residual, axis=1, dtype=np.float32)
    layer_input = rmsnorm(layer_input, norm_weight, RMS_EPS)
    return post_mix[:, :, None].astype(np.float32), comb, layer_input


def mhc_post(x, residual, post_layer_mix, comb_res_mix):
    """out_j = post_mix_j * x + sum_i comb[i][j] * residual_i."""
    mixed = np.einsum("sij,sih->sjh", comb_res_mix, residual, optimize=True).astype(np.float32)
    return (mixed + post_layer_mix * x[:, None, :]).astype(np.float32)


def mhc_fused_post_pre(x, residual, post, comb, fn, hc_scale, hc_base, norm_weight):
    residual = mhc_post(x, residual, post, comb)
    post, comb, layer_input = mhc_pre(residual, fn, hc_scale, hc_base, norm_weight)
    return residual, post, comb, layer_input


# ==========================================================================
# KDA -- see test-data/kda-goldens/manifest.json :: math.order
# ==========================================================================
def kda_conv_silu(x, w, state=None):
    """Depthwise causal conv (width 4, no bias) then SiLU.

    x: [T, C] raw pre-conv activations;  w: [C, 4] (last tap = current token);
    state: [C, 3] oldest-first raw history, or None (zeros).
    Returns (conv_out [T, C] post-SiLU, new_state [C, 3] raw).
    """
    T, C = x.shape
    if state is None:
        state = np.zeros((C, 3), dtype=np.float32)
    hist = np.concatenate([state.T.astype(np.float32), x.astype(np.float32)], axis=0)
    acc = np.zeros((T, C), dtype=np.float32)
    for j in range(KDA_CONV_W):                # accumulate j = 0..3 in order
        acc = (acc + hist[j:j + T] * w[:, j][None, :]).astype(np.float32)
    return silu(acc), hist[T:].T.copy()


def kda_scan(q, k, v, raw_g, g2, beta_raw, a_exp, dt_bias, o_norm_w, S0=None):
    """Recurrent KDA scan.

    q/k/v:     [T, H, D] post-conv, post-SiLU
    raw_g/g2:  [T, H, D];  beta_raw: [T, H];  a_exp: [H] (= exp(A_log))
    dt_bias:   [H, D];     o_norm_w: [D];     S0: [H, D, D] as S[h][v][c]
    Returns (core [T, H, D], onorm [T, H, D], S [H, D, D]).
    """
    T, H, D = q.shape
    S = np.zeros((H, D, D), dtype=np.float32) if S0 is None else S0.astype(np.float32).copy()

    g_log = (KDA_LOWER_BOUND
             * sigmoid(a_exp[None, :, None] * (raw_g + dt_bias[None, :, :]))).astype(np.float32)
    decay = np.exp(g_log.astype(np.float32)).astype(np.float32)

    qn = (q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True, dtype=np.float32) + L2NORM_EPS)).astype(np.float32)
    kn = (k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True, dtype=np.float32) + L2NORM_EPS)).astype(np.float32)
    qn = (qn * KDA_SCALE).astype(np.float32)
    beta = sigmoid(beta_raw)

    core = np.empty((T, H, D), dtype=np.float32)
    for t in range(T):
        S *= decay[t][:, None, :]                                          # decay FIRST
        Sk = np.sum(S * kn[t][:, None, :], axis=-1, dtype=np.float32)
        err = ((v[t] - Sk) * beta[t][:, None]).astype(np.float32)
        S += err[:, :, None] * kn[t][:, None, :]
        core[t] = np.sum(S * qn[t][:, None, :], axis=-1, dtype=np.float32)  # read AFTER write

    denom = np.sqrt(np.mean(core * core, axis=-1, keepdims=True, dtype=np.float32) + np.float32(RMS_EPS))
    onorm = ((core / denom) * o_norm_w[None, None, :] * sigmoid(g2)).astype(np.float32)
    return core, onorm, S


# ==========================================================================
# GGUF weight index (lazy, per-name, across shards)
# ==========================================================================
class GGUFStore:
    def __init__(self, path: str):
        import gguf
        self._gguf = gguf
        files = sorted(glob.glob(os.path.join(path, "*.gguf")))
        if not files:
            raise SystemExit("no .gguf shards under " + path)
        self._readers = {}
        self._index = {}
        for f in files:
            r = gguf.GGUFReader(f, "r")
            self._readers[f] = r
            for t in r.tensors:
                self._index[t.name] = t
        self._cache = {}
        log("[gguf] indexed %d tensors across %d shards" % (len(self._index), len(files)))

    def tensor(self, name):
        t = self._index.get(name)
        if t is None:
            raise KeyError("tensor not in GGUF: " + name)
        return t

    def _dequant(self, raw, tt):
        if raw.dtype == np.float32:
            return raw.astype(np.float32)
        return self._gguf.quants.dequantize(raw, tt).astype(np.float32)

    def get(self, name, cache=True):
        """Full dequantised tensor in row-major numpy order.

        ``ReaderTensor.data`` is already ne-reversed relative to ``.shape``.
        """
        if name in self._cache:
            return self._cache[name]
        t = self.tensor(name)
        a = self._dequant(t.data, t.tensor_type)
        if cache:
            self._cache[name] = a
        return a

    def rows(self, name, idx):
        """Dequantise only the leading-axis slice ``idx`` (blocks are row-local)."""
        t = self.tensor(name)
        return self._dequant(t.data[idx], t.tensor_type)

    def drop(self, prefix):
        for k in [k for k in self._cache if k.startswith(prefix)]:
            del self._cache[k]


# ==========================================================================
# Golden validation (test-data/kda-goldens, case A "synth")
# ==========================================================================
_M64 = (1 << 64) - 1


def _splitmix64_stream(seed, n):
    s = seed & _M64
    out = np.empty(n, dtype=np.uint64)
    for i in range(n):
        s = (s + 0x9E3779B97F4A7C15) & _M64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
        z ^= z >> 31
        out[i] = z
    return out


def _bf16_round(a):
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = (u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) & np.uint64(0xFFFFFFFF)
    r = (r & np.uint64(0xFFFF0000)).astype(np.uint32)
    return r.view(np.float32)


def _prng_tensor(seed, shape, lo, hi, bf16):
    n = int(np.prod(shape))
    r = _splitmix64_stream(seed, n)
    u01 = ((r >> np.uint64(40)).astype(np.float64) * (1.0 / 16777216.0)).astype(np.float32)
    a = (np.float64(lo) + (np.float64(hi) - np.float64(lo)) * u01.astype(np.float64)).astype(np.float32)
    if bf16:
        a = _bf16_round(a)
    return a.reshape(shape)


def validate_kda_goldens():
    """Regenerate the synthetic golden inputs and check core/onorm/state."""
    man_path = os.path.join(GOLDEN_DIR, "manifest.json")
    if not os.path.exists(man_path):
        log("[goldens] manifest.json missing -- skipping validation")
        return False
    man = json.load(open(man_path))
    geo = man["geometry"]
    H, D, T = geo["heads"], geo["head_dim"], geo["T_total"]
    Tp = geo["T_prefill"]

    chk = man["prng_check"]
    got = _splitmix64_stream(101, 4)
    exp = [int(v) for v in chk["splitmix64_seed101_first4_u64_decimal"]]
    assert [int(v) for v in got] == exp, "splitmix64 mismatch: %s vs %s" % (list(got), exp)
    q_first = _prng_tensor(101, (4,), -1.0, 1.0, True)
    assert np.allclose(q_first, np.array(chk["synth_q_raw_seed101_first4_f32_decimal"],
                                         dtype=np.float32)), "prng u01 mismatch: %s" % q_first

    q_raw = _prng_tensor(101, (T, H, D), -1.0, 1.0, True)
    k_raw = _prng_tensor(102, (T, H, D), -1.0, 1.0, True)
    v_raw = _prng_tensor(103, (T, H, D), -1.0, 1.0, True)
    raw_g = _prng_tensor(104, (T, H, D), -3.0, 3.0, True)
    g2 = _prng_tensor(105, (T, H, D), -2.0, 2.0, True)
    beta_raw = _prng_tensor(106, (T, H), -2.0, 2.0, True)
    A_log = _prng_tensor(107, (H,), -1.0, 1.5, False)
    dt_bias = _prng_tensor(108, (H, D), -1.0, 1.0, False)
    wq = _prng_tensor(109, (H * D, 4), -0.5, 0.5, True)
    wk = _prng_tensor(110, (H * D, 4), -0.5, 0.5, True)
    wv = _prng_tensor(111, (H * D, 4), -0.5, 0.5, True)
    o_norm_w = _prng_tensor(112, (D,), 0.5, 1.5, True)
    S0 = _prng_tensor(113, (H, D, D), -1.0, 1.0, False)
    cs_q = _prng_tensor(114, (H * D, 3), -1.0, 1.0, True)
    cs_k = _prng_tensor(115, (H * D, 3), -1.0, 1.0, True)
    cs_v = _prng_tensor(116, (H * D, 3), -1.0, 1.0, True)

    qc, _ = kda_conv_silu(q_raw.reshape(T, H * D), wq, cs_q)
    kc, _ = kda_conv_silu(k_raw.reshape(T, H * D), wk, cs_k)
    vc, _ = kda_conv_silu(v_raw.reshape(T, H * D), wv, cs_v)
    core, onorm, S = kda_scan(
        qc.reshape(T, H, D), kc.reshape(T, H, D), vc.reshape(T, H, D),
        raw_g, g2, beta_raw, np.exp(A_log.astype(np.float32)).astype(np.float32),
        dt_bias, o_norm_w, S0)

    def ref(fn, shape):
        return np.fromfile(os.path.join(GOLDEN_DIR, fn), dtype="<f4").reshape(shape)

    checks = [
        ("prefill_core", core[:Tp], ref("synth_prefill_core.bin", (Tp, H, D))),
        ("prefill_onorm", onorm[:Tp], ref("synth_prefill_onorm.bin", (Tp, H, D))),
        ("decode_core", core[Tp:], ref("synth_decode_core.bin", (T - Tp, H, D))),
        ("decode_onorm", onorm[Tp:], ref("synth_decode_onorm.bin", (T - Tp, H, D))),
        ("state_after_prefill", None, None),
        ("state_after_decode", S, ref("synth_state_after_decode.bin", (H, D, D))),
    ]
    ok = True
    for name, got_a, exp_a in checks:
        if got_a is None:
            continue
        rel = rms(got_a - exp_a) / max(rms(exp_a), 1e-30)
        status = "OK  " if rel < 1e-3 else "FAIL"
        if rel >= 1e-3:
            ok = False
        log("[goldens] %s synth_%-20s rel_rms=%.3e  max_abs_err=%.3e"
            % (status, name, rel, float(np.max(np.abs(got_a - exp_a)))))
    return ok


# ==========================================================================
# Layer forwards
# ==========================================================================
def kda_layer(store, n, x, diag=False):
    """KDA linear-attention layer. x: [s, hidden] (already hc_pre-normed)."""
    p = "blk.%d." % n
    H, D = KDA_HEADS, KDA_HEAD_DIM
    s = x.shape[0]

    q = linear(x, store.get(p + "attn_q.weight"))
    k = linear(x, store.get(p + "attn_k.weight"))
    v = linear(x, store.get(p + "attn_v.weight"))
    beta_raw = linear(x, store.get(p + "ssm_beta.weight"))
    raw_g = linear(linear(x, store.get(p + "ssm_f_a.weight")), store.get(p + "ssm_f_b.weight"))
    g2 = linear(linear(x, store.get(p + "ssm_g_a.weight")), store.get(p + "ssm_g_b.weight"))

    wq = store.get(p + "ssm_conv1d_q.weight").reshape(H * D, KDA_CONV_W)
    wk = store.get(p + "ssm_conv1d_k.weight").reshape(H * D, KDA_CONV_W)
    wv = store.get(p + "ssm_conv1d_v.weight").reshape(H * D, KDA_CONV_W)
    qc, _ = kda_conv_silu(q, wq)
    kc, _ = kda_conv_silu(k, wk)
    vc, _ = kda_conv_silu(v, wv)

    a_exp = (-store.get(p + "ssm_a")).astype(np.float32)   # ssm_a stores -exp(A_log)
    dt_bias = store.get(p + "ssm_dt.bias").reshape(H, D)
    o_norm_w = store.get(p + "ssm_norm.weight")

    core, onorm, _ = kda_scan(qc.reshape(s, H, D), kc.reshape(s, H, D), vc.reshape(s, H, D),
                              raw_g.reshape(s, H, D), g2.reshape(s, H, D), beta_raw,
                              a_exp, dt_bias, o_norm_w)
    if diag:
        log("    [KDA L%d diag, last pos] q_proj=%.6g conv_q=%.6g conv_k=%.6g conv_v=%.6g "
            "core=%.6g onorm=%.6g"
            % (n, rms(q[-1]), rms(qc[-1]), rms(kc[-1]), rms(vc[-1]), rms(core[-1]), rms(onorm[-1])))
    return linear(onorm.reshape(s, H * D), store.get(p + "attn_output.weight"))


def mla_layer(store, n, x, diag=False):
    """NoPE sparse MLA -- dense causal attention (5 tokens << index_topk)."""
    p = "blk.%d." % n
    s = x.shape[0]
    H = MLA_HEADS

    q = linear(x, store.get(p + "attn_q_a.weight"))
    q = rmsnorm(q, store.get(p + "attn_q_a_norm.weight"))
    q = linear(q, store.get(p + "attn_q_b.weight")).reshape(s, H, QK_NOPE)

    kv = linear(x, store.get(p + "attn_kv_a_mqa.weight"))
    c_kv = rmsnorm(kv, store.get(p + "attn_kv_a_norm.weight"))

    k_b = store.get(p + "attn_k_b.weight")     # [H, L=512, P=256]  (W_UK^T per head)
    v_b = store.get(p + "attn_v_b.weight")     # [H, V=256, L=512]  (W_UV  per head)
    k = np.einsum("sl,hlp->shp", c_kv, k_b, optimize=True).astype(np.float32)
    v = np.einsum("sl,hvl->shv", c_kv, v_b, optimize=True).astype(np.float32)

    scores = np.einsum("shd,thd->hst", q, k, optimize=True).astype(np.float32) * MLA_SCALE
    mask = np.triu(np.ones((s, s), dtype=bool), 1)
    scores = np.where(mask[None, :, :], np.float32(-np.inf), scores)
    probs = softmax(scores, axis=-1)
    ctx = np.einsum("hst,thv->shv", probs, v, optimize=True).astype(np.float32)

    out = linear(ctx.reshape(s, H * V_HEAD_DIM), store.get(p + "attn_output.weight"))
    if diag:
        log("    [MLA L%d diag, last pos] q=%.6g c_kv=%.6g ctx=%.6g attn_out=%.6g"
            % (n, rms(q[-1]), rms(c_kv[-1]), rms(ctx[-1]), rms(out[-1])))
    return out


def dense_mlp(store, n, x):
    p = "blk.%d." % n
    gate = linear(x, store.get(p + "ffn_gate.weight"))
    up = linear(x, store.get(p + "ffn_up.weight"))
    return linear(swiglu_clamped(gate, up), store.get(p + "ffn_down.weight"))


def moe_mlp(store, n, x):
    p = "blk.%d." % n
    s = x.shape[0]

    logits = linear(x, store.get(p + "ffn_gate_inp.weight"))
    scores = sigmoid(logits)
    biased = scores + store.get(p + "exp_probs_b.bias")[None, :]
    # n_group == topk_group == 1 -> grouping is a no-op; plain top-k on biased scores.
    topk_ids = np.argsort(-biased, axis=-1, kind="stable")[:, :TOP_K]
    w = np.take_along_axis(scores, topk_ids, axis=-1)      # UNBIASED routing weights
    w = w / np.sum(w, axis=-1, keepdims=True)              # norm_topk_prob
    w = (w * ROUTED_SCALE).astype(np.float32)

    out = np.zeros((s, HIDDEN), dtype=np.float32)
    uniq = sorted(set(int(e) for e in topk_ids.reshape(-1)))
    ge = store.rows(p + "ffn_gate_exps.weight", uniq)      # [E, 2048, 4096]
    ue = store.rows(p + "ffn_up_exps.weight", uniq)
    de = store.rows(p + "ffn_down_exps.weight", uniq)      # [E, 4096, 2048]
    pos = dict((e, i) for i, e in enumerate(uniq))
    for ti in range(s):
        for j in range(TOP_K):
            i = pos[int(topk_ids[ti, j])]
            xt = x[ti:ti + 1]
            h = swiglu_clamped(linear(xt, ge[i]), linear(xt, ue[i]))
            out[ti] += w[ti, j] * linear(h, de[i])[0]
    del ge, ue, de

    # Shared expert: always added, unscaled (apply_routed_scale_to_output=False).
    sh = swiglu_clamped(linear(x, store.get(p + "ffn_gate_shexp.weight")),
                        linear(x, store.get(p + "ffn_up_shexp.weight")))
    out += linear(sh, store.get(p + "ffn_down_shexp.weight"))
    return out


# ==========================================================================
# Full forward
# ==========================================================================
def forward(store, ids):
    te = store.tensor("token_embd.weight")
    h = store._dequant(te.data[list(ids)], te.tensor_type).astype(np.float32)
    log("[embed] rms=%.6g last_pos_rms=%.6g" % (rms(h), rms(h[-1])))

    residual = post = comb = None
    trace = []

    n_run = int(os.environ.get("GLM53_REF_NLAYERS", N_LAYERS))
    for n in range(n_run):
        t0 = time.time()
        p = "blk.%d." % n
        hc_attn = (store.get(p + "hc_attn_fn.weight"), store.get(p + "hc_attn_scale.weight"),
                   store.get(p + "hc_attn_base.weight"))
        attn_norm_w = store.get(p + "attn_norm.weight")

        if post is None:                     # layer 0: standalone hc_pre
            residual = hc_expand(h, HC_MULT)
            post, comb, x = mhc_pre(residual, hc_attn[0], hc_attn[1], hc_attn[2], attn_norm_w)
        else:
            residual, post, comb, x = mhc_fused_post_pre(
                h, residual, post, comb, hc_attn[0], hc_attn[1], hc_attn[2], attn_norm_w)

        is_mla = n in FULL_ATTN_LAYERS
        diag = (n == 0) or (n == 3)
        stages = os.environ.get("GLM53_REF_STAGES") == "1"
        if stages:
            log("  [stage] L%d Ain(res)=%.6f x_attn=%.6f" % (n, rms(residual[-1]), rms(x[-1])))
        x = mla_layer(store, n, x, diag) if is_mla else kda_layer(store, n, x, diag)
        if stages:
            log("  [stage] L%d attn_out=%.6f" % (n, rms(x[-1])))

        hc_ffn = (store.get(p + "hc_ffn_fn.weight"), store.get(p + "hc_ffn_scale.weight"),
                  store.get(p + "hc_ffn_base.weight"))
        residual, post, comb, x = mhc_fused_post_pre(
            x, residual, post, comb, hc_ffn[0], hc_ffn[1], hc_ffn[2],
            store.get(p + "ffn_norm.weight"))
        if stages:
            log("  [stage] L%d Min(res-after-attn-post)=%.6f Mhcx(ffn-x)=%.6f" % (n, rms(residual[-1]), rms(x[-1])))

        x = dense_mlp(store, n, x) if n < FIRST_K_DENSE else moe_mlp(store, n, x)
        if stages:
            log("  [stage] L%d mlp_out=%.6f" % (n, rms(x[-1])))

        # Residual entering the next layer (the deferred hc_post materialised).
        nxt = mhc_post(x, residual, post, comb)
        trace.append(rms(nxt[-1]))
        log("L%-2d rms=%.6f  (%s, %s, %.1fs)"
            % (n, trace[-1], "MLA" if is_mla else "KDA",
               "dense" if n < FIRST_K_DENSE else "moe", time.time() - t0))

        h = hc_contract(nxt) if n == N_LAYERS - 1 else x
        store.drop(p)

    h = rmsnorm(h, store.get("output_norm.weight"))
    log("[final] post-contract post-norm rms=%.6g" % rms(h[-1]))

    # Logits for the last position only, vocab-chunked to bound memory.
    ow = store.tensor("output.weight")
    last = h[-1].astype(np.float32)
    logits = np.empty(VOCAB, dtype=np.float32)
    CH = 16384
    for i in range(0, VOCAB, CH):
        j = min(i + CH, VOCAB)
        logits[i:j] = store._dequant(ow.data[i:j], ow.tensor_type) @ last
    return logits, trace


def load_vocab():
    try:
        tk = json.load(open(TOKENIZER_JSON))
        inv = dict((int(v), k) for k, v in tk["model"]["vocab"].items())
        for a in tk.get("added_tokens", []):
            inv[int(a["id"])] = a["content"]
        return inv
    except Exception as exc:
        log("[tokenizer] unavailable (%s)" % exc)
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--skip-goldens", action="store_true")
    ap.add_argument("--goldens-only", action="store_true")
    args = ap.parse_args()

    if not args.skip_goldens:
        log("=== KDA golden validation (test-data/kda-goldens, case A synth) ===")
        t0 = time.time()
        ok = validate_kda_goldens()
        log("[goldens] %s in %.1fs" % ("VALIDATED" if ok else "MISMATCH", time.time() - t0))
        if not ok:
            log("KDA goldens did not validate -- aborting the full forward.")
            return 1
    if args.goldens_only:
        return 0

    log("=== GLM-5.3-Flash NumPy reference forward ===")
    store = GGUFStore(args.gguf)
    t0 = time.time()
    logits, trace = forward(store, PROMPT_IDS)
    log("[forward] %.1fs" % (time.time() - t0))

    probs = softmax(logits.astype(np.float32))
    order = np.argsort(-logits)
    inv = load_vocab()
    log("=== top-5 next tokens after 'The capital of France is' ===")
    for r, i in enumerate(order[:5]):
        log("  #%d  id=%-7d p=%.6f  logit=%.4f  %r"
            % (r + 1, int(i), float(probs[i]), float(logits[i]), inv.get(int(i), "<?>")))
    log("[check] id 12089 %r: rank=%d p=%.6f"
        % (inv.get(12089, "<?>"), int(np.where(order == 12089)[0][0]) + 1, float(probs[12089])))
    log("[trace] " + " ".join("L%d=%.4f" % (i, v) for i, v in enumerate(trace)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

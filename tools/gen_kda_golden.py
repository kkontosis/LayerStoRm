#!/usr/bin/env python3
"""Generate CPU golden fixtures for Kimi Delta Attention (KDA), GLM-5.3-Flash.

This is a NumPy float32 transcription of the KDA *recurrent* (step-by-step)
path, used to pin a C++ CPU reference implementation.  The math mirrors
vLLM @ pr-53906-head:

  * ``vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`` ->
    ``fused_recurrent_gated_delta_rule_fwd_kernel`` with
    ``IS_KDA=True, COMPUTE_GATE=True, SIGMOID_BETA=True,
    USE_QK_L2NORM_IN_KERNEL=True`` (SAFE_GATE branch), and
  * ``vllm/models/glm5next/nvidia/kda.py`` for the depthwise causal conv +
    gating wiring, plus ``FusedRMSNormGated(activation="sigmoid", eps=1e-5)``.

Two cases are emitted into ``test-data/kda-goldens/``:

  synth_*  everything synthetic (params and activations), non-zero initial
           recurrent state and non-zero initial conv state.
  real_*   real layer-0 parameters fetched by
           ``tools/fetch_glm53_kda_layer0.py`` (heads 0/17/42/63), synthetic
           activations, ZERO initial recurrent and conv state.  raw_g and the
           output gate g2 are produced through the real low-rank f_b / g_b
           matrices.

All inputs come from a deterministic splitmix64 stream (one independent
stream per tensor, seeded per the manifest) so a C++ implementation can
regenerate the exact same inputs without shipping them.

Usage:  python3 tools/gen_kda_golden.py [--out DIR] [--real DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Model / test constants
# ---------------------------------------------------------------------------
D = 128           # head_dim (K == V == D)
H = 4             # heads in the golden
T_PREFILL = 144
T_DECODE = 8
T_TOTAL = T_PREFILL + T_DECODE     # 152
CONV_W = 4        # short_conv_kernel_size
C = H * D         # 512 conv channels

LOWER_BOUND = np.float32(-5.0)     # config linear_attn_config.gate_lower_bound
SCALE = np.float32(1.0 / np.sqrt(np.float64(128.0)))
L2_EPS = np.float32(1e-6)          # inside the sqrt
RMS_EPS = np.float32(1e-5)

REAL_HEADS = [0, 17, 42, 63]
REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"

REAL_NUM_HEADS = 64
REAL_CHANNELS = REAL_NUM_HEADS * D  # 8192

# ---------------------------------------------------------------------------
# splitmix64
# ---------------------------------------------------------------------------
_M64 = 0xFFFFFFFFFFFFFFFF
_GAMMA = 0x9E3779B97F4A7C15
_C1 = 0xBF58476D1CE4E5B9
_C2 = 0x94D049BB133111EB
_INV24 = 1.0 / 16777216.0


def splitmix64_scalar(seed, n):
    """Reference (pure-Python) splitmix64: s += GAMMA, then mix."""
    s = seed & _M64
    out = []
    for _ in range(n):
        s = (s + _GAMMA) & _M64
        z = s
        z = ((z ^ (z >> 30)) * _C1) & _M64
        z = ((z ^ (z >> 27)) * _C2) & _M64
        z = z ^ (z >> 31)
        out.append(z)
    return out


def splitmix64_vec(seed, n):
    """Vectorised splitmix64. s_i is closed form: seed + (i+1)*GAMMA mod 2^64."""
    idx = np.arange(1, n + 1, dtype=np.uint64)
    s = np.uint64(seed) + idx * np.uint64(_GAMMA)
    z = s
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_C1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_C2)
    z = z ^ (z >> np.uint64(31))
    return z


def stream_f32(seed, lo, hi, shape):
    """uniform[lo,hi) float32 tensor filled in flat row-major order."""
    n = 1
    for d in shape:
        n *= int(d)
    r = splitmix64_vec(seed, n)
    u01 = (r >> np.uint64(40)).astype(np.float64) * _INV24
    vals = (float(lo) + (float(hi) - float(lo)) * u01).astype(np.float32)
    return vals.reshape(shape)


# ---------------------------------------------------------------------------
# bfloat16 helpers
# ---------------------------------------------------------------------------
def bf16_round(x):
    """Round float32 -> nearest-even bfloat16 -> back to float32."""
    a = np.ascontiguousarray(x, dtype=np.float32)
    shape = a.shape
    u = a.reshape(-1).view(np.uint32).astype(np.uint64)
    r = (u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) & np.uint64(0xFFFFFFFF)
    r = (r & np.uint64(0xFFFF0000)).astype(np.uint32)
    return r.view(np.float32).reshape(shape).copy()


def load_bf16_bin(path, shape):
    u = np.fromfile(path, dtype="<u2").astype(np.uint32) << np.uint32(16)
    a = u.view(np.float32)
    n = 1
    for d in shape:
        n *= int(d)
    if a.size != n:
        raise RuntimeError("%s: %d bf16 elements, expected %d" % (path, a.size, n))
    return a.reshape(shape).copy()


def load_f32_bin(path, shape):
    a = np.fromfile(path, dtype="<f4")
    n = 1
    for d in shape:
        n *= int(d)
    if a.size != n:
        raise RuntimeError("%s: %d f32 elements, expected %d" % (path, a.size, n))
    return a.reshape(shape).copy()


# ---------------------------------------------------------------------------
# elementwise helpers (float32 throughout)
# ---------------------------------------------------------------------------
def sigmoid_f32(x):
    with np.errstate(over="ignore"):
        return (np.float32(1.0) / (np.float32(1.0) + np.exp(-np.float32(1.0) * x))).astype(
            np.float32
        )


def silu_f32(x):
    return (x * sigmoid_f32(x)).astype(np.float32)


# ---------------------------------------------------------------------------
# depthwise causal conv1d (width 4, no bias, SiLU)
# ---------------------------------------------------------------------------
def conv_forward(x, w, init_state):
    """x [T,C] f32 raw inputs, w [C,4] f32, init_state [C,3] f32 oldest-first.

    Returns (conv_out [T,C] post-SiLU, states) where states is a dict mapping
    a token count -> the conv state [C,3] after that many tokens (raw,
    pre-conv, pre-SiLU inputs, oldest-first).
    """
    T = x.shape[0]
    padded = np.concatenate([np.ascontiguousarray(init_state.T), x], axis=0).astype(np.float32)
    acc = np.zeros((T, x.shape[1]), dtype=np.float32)
    for j in range(CONV_W):
        acc = (acc + padded[j : j + T, :] * w[:, j][None, :]).astype(np.float32)
    return silu_f32(acc), padded


def conv_state_at(padded, ntok):
    """Conv state after ``ntok`` tokens: the last 3 raw inputs, oldest-first."""
    # padded rows are t = -3, -2, -1, 0, 1, ...  -> row index = t + 3
    rows = padded[ntok : ntok + 3, :]      # t = ntok-3, ntok-2, ntok-1
    return np.ascontiguousarray(rows.T).astype(np.float32)


# ---------------------------------------------------------------------------
# KDA recurrent path
# ---------------------------------------------------------------------------
def run_kda(params, acts, S0, conv_init):
    """Run the full T_TOTAL-token recurrence. Returns an output dict."""
    A_log = params["A_log"]           # [H] f32
    dt_bias = params["dt_bias"]       # [H,D] f32
    w_on = params["o_norm_w"]         # [D] f32
    wq, wk, wv = params["wq"], params["wk"], params["wv"]   # [C,4] f32

    q_in = acts["q_raw"].reshape(T_TOTAL, C)
    k_in = acts["k_raw"].reshape(T_TOTAL, C)
    v_in = acts["v_raw"].reshape(T_TOTAL, C)
    raw_g = acts["raw_g"]             # [T,H,D]
    g2_raw = acts["g2_raw"]           # [T,H,D]
    beta_raw = acts["beta_raw"]       # [T,H]

    qc, q_pad = conv_forward(q_in, wq, conv_init["q"])
    kc, k_pad = conv_forward(k_in, wk, conv_init["k"])
    vc, v_pad = conv_forward(v_in, wv, conv_init["v"])
    qc = qc.reshape(T_TOTAL, H, D)
    kc = kc.reshape(T_TOTAL, H, D)
    vc = vc.reshape(T_TOTAL, H, D)

    S = np.array(S0, dtype=np.float32, copy=True)      # [H,V,K]

    core = np.zeros((T_TOTAL, H, D), dtype=np.float32)
    onorm = np.zeros((T_TOTAL, H, D), dtype=np.float32)
    state_after_prefill = None

    a_amp = np.exp(A_log.astype(np.float32)).astype(np.float32)   # [H]

    glog_min = np.float32(np.inf)
    glog_max = np.float32(-np.inf)
    n_at_lb = 0
    n_at_zero = 0

    for t in range(T_TOTAL):
        for h in range(H):
            q = qc[t, h]
            k = kc[t, h]
            v = vc[t, h]

            # 1. L2 normalise (eps INSIDE the sqrt)
            q = (q / np.sqrt(np.sum(q * q, dtype=np.float32) + L2_EPS)).astype(np.float32)
            k = (k / np.sqrt(np.sum(k * k, dtype=np.float32) + L2_EPS)).astype(np.float32)
            # 2. scale
            q = (q * SCALE).astype(np.float32)

            # 3. gate: g_log = LOWER_BOUND / (1 + exp(-(exp(A_log) * (g + bias))))
            gk = (raw_g[t, h] + dt_bias[h]).astype(np.float32)
            with np.errstate(over="ignore"):
                g_log = (
                    LOWER_BOUND / (np.float32(1.0) + np.exp(-(a_amp[h] * gk).astype(np.float32)))
                ).astype(np.float32)
            decay = np.exp(g_log).astype(np.float32)

            glog_min = min(glog_min, g_log.min())
            glog_max = max(glog_max, g_log.max())
            n_at_lb += int(np.count_nonzero(g_log <= LOWER_BOUND))
            n_at_zero += int(np.count_nonzero(g_log >= np.float32(0.0)))

            # 4. decay the state (per key-channel c)
            S[h] *= decay[None, :]
            # 5. delta error
            err = (v - np.sum(S[h] * k[None, :], axis=1, dtype=np.float32)).astype(np.float32)
            # 6. beta
            beta = sigmoid_f32(beta_raw[t, h])
            err = (err * beta).astype(np.float32)
            # 7. rank-1 write
            S[h] += (err[:, None] * k[None, :]).astype(np.float32)
            # 8. read AFTER the write
            co = np.sum(S[h] * q[None, :], axis=1, dtype=np.float32).astype(np.float32)
            core[t, h] = co

            # 9. gated RMSNorm (sigmoid activation)
            ms = (np.sum(co * co, dtype=np.float32) / np.float32(D)).astype(np.float32)
            rms = np.sqrt(ms + RMS_EPS).astype(np.float32)
            onorm[t, h] = (((co / rms) * w_on) * sigmoid_f32(g2_raw[t, h])).astype(np.float32)

        if t == T_PREFILL - 1:
            state_after_prefill = np.array(S, dtype=np.float32, copy=True)

    return {
        "core": core,
        "onorm": onorm,
        "state_after_prefill": state_after_prefill,
        "state_after_decode": np.array(S, dtype=np.float32, copy=True),
        "conv_state_q": conv_state_at(q_pad, T_PREFILL),
        "conv_state_k": conv_state_at(k_pad, T_PREFILL),
        "conv_state_v": conv_state_at(v_pad, T_PREFILL),
        "glog_min": float(glog_min),
        "glog_max": float(glog_max),
        "n_at_lb": n_at_lb,
        "n_at_zero": n_at_zero,
    }


# ---------------------------------------------------------------------------
# Independent recompute of a single (token, head), written from scratch
# ---------------------------------------------------------------------------
def recheck_token(params, acts, S0, conv_init, tt, gh, dt=np.float32):
    """Re-derive core/onorm for token ``tt`` head ``gh`` with a separate loop.

    Deliberately structured differently from ``run_kda``: the conv is evaluated
    per channel with scalar accumulation, and only head ``gh`` is advanced.
    ``dt`` selects float32 (bit-exact cross-check) or float64 (numeric
    cross-check).
    """
    one = dt(1.0)
    lb = dt(LOWER_BOUND)
    l2e = dt(1e-6)
    rmse = dt(1e-5)
    sc = dt(SCALE)

    def sig(x):
        with np.errstate(over="ignore"):
            return (one / (one + np.exp(-x))).astype(dt)

    base = gh * D
    A = dt(params["A_log"][gh])
    amp = dt(np.exp(A))
    bias = params["dt_bias"][gh].astype(dt)
    won = params["o_norm_w"].astype(dt)

    # --- per-channel scalar conv for this head only, tokens 0..tt ----------
    conv = {}
    for tag, xin, wfull, st in (
        ("q", acts["q_raw"], params["wq"], conv_init["q"]),
        ("k", acts["k_raw"], params["wk"], conv_init["k"]),
        ("v", acts["v_raw"], params["wv"], conv_init["v"]),
    ):
        out = np.zeros((tt + 1, D), dtype=dt)
        for d in range(D):
            ch = base + d
            hist = [dt(st[ch, 0]), dt(st[ch, 1]), dt(st[ch, 2])]
            w0, w1, w2, w3 = (dt(wfull[ch, j]) for j in range(CONV_W))
            for t in range(tt + 1):
                xt = dt(xin[t, gh, d])
                acc = dt(0.0)
                acc = dt(acc + dt(hist[0] * w0))
                acc = dt(acc + dt(hist[1] * w1))
                acc = dt(acc + dt(hist[2] * w2))
                acc = dt(acc + dt(xt * w3))
                out[t, d] = dt(acc * sig(acc))
                hist = [hist[1], hist[2], xt]
        conv[tag] = out

    # --- recurrence for this head only ------------------------------------
    S = np.array(S0[gh], dtype=dt, copy=True)
    core = None
    on = None
    for t in range(tt + 1):
        q = conv["q"][t].astype(dt)
        k = conv["k"][t].astype(dt)
        v = conv["v"][t].astype(dt)

        q = (q / np.sqrt(np.sum(q * q, dtype=dt) + l2e)).astype(dt)
        k = (k / np.sqrt(np.sum(k * k, dtype=dt) + l2e)).astype(dt)
        q = (q * sc).astype(dt)

        gk = (acts["raw_g"][t, gh].astype(dt) + bias).astype(dt)
        with np.errstate(over="ignore"):
            g_log = (lb / (one + np.exp(-(amp * gk).astype(dt)))).astype(dt)
        dec = np.exp(g_log).astype(dt)

        for c in range(D):
            S[:, c] = (S[:, c] * dec[c]).astype(dt)

        pred = np.zeros(D, dtype=dt)
        for vv in range(D):
            pred[vv] = np.sum(S[vv] * k, dtype=dt)
        err = (v - pred).astype(dt)
        err = (err * sig(dt(acts["beta_raw"][t, gh]))).astype(dt)
        for vv in range(D):
            S[vv] = (S[vv] + (err[vv] * k)).astype(dt)

        core = np.zeros(D, dtype=dt)
        for vv in range(D):
            core[vv] = np.sum(S[vv] * q, dtype=dt)

        ms = (np.sum(core * core, dtype=dt) / dt(D)).astype(dt)
        rms = np.sqrt(ms + rmse).astype(dt)
        on = (((core / rms) * won) * sig(acts["g2_raw"][t, gh].astype(dt))).astype(dt)

    return core, on


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------
def lowrank_gemm(act, w):
    """out[t] = w @ act[t] for w [N,D], act [T,D] -> [T,N], float32.

    Deliberately avoids BLAS: each output element is an elementwise product
    followed by a numpy pairwise summation, exactly like the two 128-wide
    reductions in the recurrent step. This keeps the fixture reproducible on
    any host regardless of the installed BLAS (a BLAS sgemm and this form
    disagree by ~1 bf16 ULP on a handful of elements).
    """
    T = act.shape[0]
    out = np.empty((T, w.shape[0]), dtype=np.float32)
    for t in range(T):
        out[t] = np.sum(w * act[t][None, :], axis=1, dtype=np.float32)
    return out


def build_case_a():
    params = {
        "A_log": stream_f32(107, -1.0, 1.5, (H,)),
        "dt_bias": stream_f32(108, -1.0, 1.0, (H, D)),
        "wq": bf16_round(stream_f32(109, -0.5, 0.5, (C, CONV_W))),
        "wk": bf16_round(stream_f32(110, -0.5, 0.5, (C, CONV_W))),
        "wv": bf16_round(stream_f32(111, -0.5, 0.5, (C, CONV_W))),
        "o_norm_w": bf16_round(stream_f32(112, 0.5, 1.5, (D,))),
    }
    acts = {
        "q_raw": bf16_round(stream_f32(101, -1.0, 1.0, (T_TOTAL, H, D))),
        "k_raw": bf16_round(stream_f32(102, -1.0, 1.0, (T_TOTAL, H, D))),
        "v_raw": bf16_round(stream_f32(103, -1.0, 1.0, (T_TOTAL, H, D))),
        "raw_g": bf16_round(stream_f32(104, -3.0, 3.0, (T_TOTAL, H, D))),
        "g2_raw": bf16_round(stream_f32(105, -2.0, 2.0, (T_TOTAL, H, D))),
        "beta_raw": bf16_round(stream_f32(106, -2.0, 2.0, (T_TOTAL, H))),
    }
    S0 = stream_f32(113, -1.0, 1.0, (H, D, D))
    conv_init = {
        "q": bf16_round(stream_f32(114, -1.0, 1.0, (C, 3))),
        "k": bf16_round(stream_f32(115, -1.0, 1.0, (C, 3))),
        "v": bf16_round(stream_f32(116, -1.0, 1.0, (C, 3))),
    }
    return params, acts, S0, conv_init


def build_case_b(real_dir):
    def p(n):
        return os.path.join(real_dir, n + ".bin")

    A_log_full = load_f32_bin(p("A_log"), (REAL_NUM_HEADS,))
    dt_full = load_f32_bin(p("dt_bias"), (REAL_CHANNELS,))
    o_norm_w = load_bf16_bin(p("o_norm_weight"), (D,))
    f_b = load_bf16_bin(p("f_b_proj_weight"), (REAL_CHANNELS, D))
    g_b = load_bf16_bin(p("g_b_proj_weight"), (REAL_CHANNELS, D))
    convs = {
        "wq": load_bf16_bin(p("q_conv1d_weight"), (REAL_CHANNELS, 1, CONV_W)).reshape(
            REAL_CHANNELS, CONV_W
        ),
        "wk": load_bf16_bin(p("k_conv1d_weight"), (REAL_CHANNELS, 1, CONV_W)).reshape(
            REAL_CHANNELS, CONV_W
        ),
        "wv": load_bf16_bin(p("v_conv1d_weight"), (REAL_CHANNELS, 1, CONV_W)).reshape(
            REAL_CHANNELS, CONV_W
        ),
    }

    rows = np.concatenate(
        [np.arange(rh * D, (rh + 1) * D, dtype=np.int64) for rh in REAL_HEADS]
    )
    params = {
        "A_log": np.array([A_log_full[rh] for rh in REAL_HEADS], dtype=np.float32),
        "dt_bias": np.stack(
            [dt_full[rh * D : (rh + 1) * D] for rh in REAL_HEADS]
        ).astype(np.float32),
        "wq": np.ascontiguousarray(convs["wq"][rows]).astype(np.float32),
        "wk": np.ascontiguousarray(convs["wk"][rows]).astype(np.float32),
        "wv": np.ascontiguousarray(convs["wv"][rows]).astype(np.float32),
        "o_norm_w": o_norm_w.astype(np.float32),
    }

    # low-rank gating through the REAL f_b / g_b matrices; GEMM output is the
    # model dtype (bf16) in vLLM, so it is rounded here too.
    f_a_act = bf16_round(stream_f32(301, -1.0, 1.0, (T_TOTAL, D)))
    g_a_act = bf16_round(stream_f32(302, -1.0, 1.0, (T_TOTAL, D)))
    raw_g_full = bf16_round(lowrank_gemm(f_a_act, f_b))   # [T,8192]
    g2_full = bf16_round(lowrank_gemm(g_a_act, g_b))      # [T,8192]

    acts = {
        "q_raw": bf16_round(stream_f32(303, -1.0, 1.0, (T_TOTAL, H, D))),
        "k_raw": bf16_round(stream_f32(304, -1.0, 1.0, (T_TOTAL, H, D))),
        "v_raw": bf16_round(stream_f32(305, -1.0, 1.0, (T_TOTAL, H, D))),
        "raw_g": np.ascontiguousarray(raw_g_full[:, rows].reshape(T_TOTAL, H, D)),
        "g2_raw": np.ascontiguousarray(g2_full[:, rows].reshape(T_TOTAL, H, D)),
        "beta_raw": bf16_round(stream_f32(306, -2.0, 2.0, (T_TOTAL, H))),
    }
    S0 = np.zeros((H, D, D), dtype=np.float32)
    conv_init = {
        "q": np.zeros((C, 3), dtype=np.float32),
        "k": np.zeros((C, 3), dtype=np.float32),
        "v": np.zeros((C, 3), dtype=np.float32),
    }
    return params, acts, S0, conv_init


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
def write_bin(out_dir, name, arr):
    path = os.path.join(out_dir, name)
    a = np.ascontiguousarray(arr, dtype="<f4")
    a.tofile(path)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return {
        "file": name,
        "dtype": "float32",
        "byte_order": "little-endian",
        "shape": list(a.shape),
        "byte_size": int(a.nbytes),
        "sha256": digest,
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.astype(np.float64).mean()),
    }


def emit_case(out_dir, prefix, res):
    files = []
    files.append(write_bin(out_dir, prefix + "prefill_core.bin", res["core"][:T_PREFILL]))
    files.append(write_bin(out_dir, prefix + "prefill_onorm.bin", res["onorm"][:T_PREFILL]))
    files.append(write_bin(out_dir, prefix + "state_after_prefill.bin", res["state_after_prefill"]))
    files.append(write_bin(out_dir, prefix + "conv_state_q.bin", res["conv_state_q"]))
    files.append(write_bin(out_dir, prefix + "conv_state_k.bin", res["conv_state_k"]))
    files.append(write_bin(out_dir, prefix + "conv_state_v.bin", res["conv_state_v"]))
    files.append(write_bin(out_dir, prefix + "decode_core.bin", res["core"][T_PREFILL:]))
    files.append(write_bin(out_dir, prefix + "decode_onorm.bin", res["onorm"][T_PREFILL:]))
    files.append(write_bin(out_dir, prefix + "state_after_decode.bin", res["state_after_decode"]))
    return files


def check_finite(tag, res):
    bad = []
    for key in (
        "core",
        "onorm",
        "state_after_prefill",
        "state_after_decode",
        "conv_state_q",
        "conv_state_k",
        "conv_state_v",
    ):
        a = res[key]
        if not np.all(np.isfinite(a)):
            bad.append("%s.%s has %d non-finite" % (tag, key, int((~np.isfinite(a)).sum())))
    if bad:
        raise RuntimeError("non-finite values: " + "; ".join(bad))


# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="generate KDA CPU golden fixtures")
    ap.add_argument("--out", default=os.path.join(here, "test-data", "kda-goldens"))
    ap.add_argument(
        "--real", default=os.path.join(here, "test-data", "GLM-5.3-Flash", "kda-layer0")
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- PRNG self-check ---------------------------------------------------
    for seed in (101, 107, 113, 301, 306):
        ref = splitmix64_scalar(seed, 64)
        vec = splitmix64_vec(seed, 64)
        for i in range(64):
            if int(vec[i]) != ref[i]:
                raise RuntimeError(
                    "splitmix64 vector/scalar mismatch seed=%d i=%d" % (seed, i)
                )
    print("splitmix64: vectorised == scalar reference for all probed seeds")

    prng_u64 = [str(x) for x in splitmix64_scalar(101, 4)]
    q0 = bf16_round(stream_f32(101, -1.0, 1.0, (T_TOTAL, H, D))).reshape(-1)[:4]
    q0_hex = ["0x%08X" % int(x) for x in q0.view(np.uint32)]
    q0_dec = [float(x) for x in q0]
    print("prng_check splitmix64(101)[0:4] =", prng_u64)
    print("prng_check synth q_raw[0:4] hex =", q0_hex)
    print("prng_check synth q_raw[0:4] dec =", q0_dec)

    # ---- build + run -------------------------------------------------------
    cases = []
    pa, aa, sa, ca = build_case_a()
    cases.append(("synth_", "A (synthetic params + activations)", pa, aa, sa, ca))
    pb, ab, sb, cb = build_case_b(args.real)
    cases.append(("real_", "B (real GLM-5.3-Flash layer-0 params)", pb, ab, sb, cb))

    manifest_cases = {}
    all_files = []
    for prefix, label, params, acts, S0, conv_init in cases:
        print("\n=== case %s (prefix %r) ===" % (label, prefix))
        res = run_kda(params, acts, S0, conv_init)
        check_finite(prefix, res)

        print(
            "  g_log range: [%.9g, %.9g]  (LOWER_BOUND=%.1f); at-lower-bound=%d at-zero=%d "
            "of %d gate values"
            % (
                res["glog_min"],
                res["glog_max"],
                float(LOWER_BOUND),
                res["n_at_lb"],
                res["n_at_zero"],
                T_TOTAL * H * D,
            )
        )
        if not (res["glog_min"] >= float(LOWER_BOUND) and res["glog_max"] <= 0.0):
            raise RuntimeError("gate out of [%.1f, 0]" % float(LOWER_BOUND))

        files = emit_case(args.out, prefix, res)
        for f in files:
            print(
                "  %-32s %-18s %9d B  min=%+.6g max=%+.6g mean=%+.6g"
                % (f["file"], str(f["shape"]), f["byte_size"], f["min"], f["max"], f["mean"])
            )
            print("      sha256=%s" % f["sha256"])
        all_files.extend(files)

        # ---- independent recompute of t=100, head 2 ------------------------
        tt, gh = 100, 2
        c32, o32 = recheck_token(params, acts, S0, conv_init, tt, gh, dt=np.float32)
        np.testing.assert_array_equal(
            c32, res["core"][tt, gh], err_msg="%s core t=%d h=%d mismatch" % (prefix, tt, gh)
        )
        np.testing.assert_array_equal(
            o32, res["onorm"][tt, gh], err_msg="%s onorm t=%d h=%d mismatch" % (prefix, tt, gh)
        )
        c64, o64 = recheck_token(params, acts, S0, conv_init, tt, gh, dt=np.float64)
        rel_c = float(
            np.max(np.abs(c64 - c32.astype(np.float64))) / max(np.max(np.abs(c64)), 1e-30)
        )
        rel_o = float(
            np.max(np.abs(o64 - o32.astype(np.float64))) / max(np.max(np.abs(o64)), 1e-30)
        )
        print(
            "  independent recompute t=%d h=%d: f32 loop BIT-EXACT; "
            "f64 rel-dev core=%.3g onorm=%.3g" % (tt, gh, rel_c, rel_o)
        )
        if rel_c > 1e-2 or rel_o > 1e-2:
            raise RuntimeError("float64 cross-check deviates too much")

        manifest_cases[prefix.rstrip("_")] = {
            "prefix": prefix,
            "label": label,
            "gate_log_min": res["glog_min"],
            "gate_log_max": res["glog_max"],
            "gate_values_at_lower_bound": res["n_at_lb"],
            "gate_values_at_zero": res["n_at_zero"],
            "files": files,
        }

    # ---- manifest ----------------------------------------------------------
    manifest = {
        "description": (
            "CPU golden fixtures for Kimi Delta Attention (KDA), the linear-attention "
            "layer of GLM-5.3-Flash. Float32 NumPy transcription of the RECURRENT "
            "(step-by-step) path."
        ),
        "reference": {
            "repo": "vllm",
            "tag": "pr-53906-head",
            "kernel": (
                "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py :: "
                "fused_recurrent_gated_delta_rule_fwd_kernel "
                "(IS_KDA=True, COMPUTE_GATE=True, SIGMOID_BETA=True, "
                "USE_QK_L2NORM_IN_KERNEL=True, SAFE_GATE)"
            ),
            "wiring": "vllm/models/glm5next/nvidia/kda.py",
            "o_norm": "FusedRMSNormGated(hidden=128, activation='sigmoid', eps=1e-5)",
        },
        "geometry": {
            "heads": H,
            "head_dim": D,
            "K": D,
            "V": D,
            "T_total": T_TOTAL,
            "T_prefill": T_PREFILL,
            "T_decode": T_DECODE,
            "conv_width": CONV_W,
            "conv_channels": C,
            "scale": float(SCALE),
            "scale_hex_u32": "0x%08X" % int(np.float32(SCALE).view(np.uint32)),
            "lower_bound": float(LOWER_BOUND),
            "l2norm_eps": 1e-6,
            "l2norm_eps_position": "inside the sqrt: x / sqrt(sum(x*x) + eps)",
            "rmsnorm_eps": 1e-5,
        },
        "math": {
            "order": [
                "depthwise causal conv1d (width 4, no bias) over the RAW q/k/v, then SiLU",
                "L2 normalise q and k (eps inside sqrt); q *= scale",
                "a = exp(A_log[h]); g_log = LOWER_BOUND / (1 + exp(-(a*(raw_g + dt_bias))))",
                "decay = exp(g_log); S[v][c] *= decay[c]  (decay FIRST)",
                "err[v] = v_conv[v] - sum_c S[v][c]*k[c]",
                "err *= sigmoid(beta_raw)",
                "S[v][c] += err[v]*k[c]",
                "core_out[v] = sum_c S[v][c]*q[c]   (read AFTER the write)",
                "rms = sqrt(mean(core_out^2) + 1e-5); "
                "onorm[v] = (core_out[v]/rms) * o_norm_w[v] * sigmoid(g2_raw[v])",
            ],
            "state_layout": "S[h][v][c], v = value/output channel, c = key channel",
            "conv_state_layout": "[channel][3], oldest-first: x[t-3], x[t-2], x[t-1] (RAW, pre-conv, pre-SiLU)",
            "conv_channel_index": "ch = head*128 + dim",
            "accumulation": (
                "plain float32. The two 128-wide reductions are done as an elementwise "
                "product followed by a summation (numpy pairwise), mirroring the Triton "
                "kernel's tl.sum(b_h * b_k[None, :], 1); no BLAS gemv is used, so the "
                "goldens do not depend on the host BLAS build. The conv accumulates "
                "j=0..3 in order. A C++ reference summing sequentially will agree to a "
                "few ULP, not bit-for-bit: compare with a relative tolerance around 1e-5."
            ),
            "conv_output_not_bf16_rounded": True,
            "prefill_decode_split": (
                "one continuous recurrence over 152 tokens; tokens 0..143 land in the "
                "prefill files, tokens 144..151 in the decode files"
            ),
        },
        "caveats": [
            "The o_norm RMS is eps-dominated in both cases: mean(core_out^2) has a median "
            "of ~2.8e-7 (synth) and ~4.7e-9 (real) against rmsnorm_eps=1e-5, so the "
            "1/sqrt(.) is clamped by eps for most tokens. The onorm goldens therefore "
            "exercise the gate and the per-channel weight strongly, but the normalisation "
            "division only weakly. Use *_core.bin (pre-o_norm) as the primary check of the "
            "recurrence, and prefer the state_after_* tensors for end-to-end drift.",
            "core_out is small because q is L2-normalised and then scaled by 1/sqrt(128), "
            "and because the real conv weights are small (std ~0.02) while the synthetic "
            "pre-conv activations are unit-scale; the real engine feeds larger pre-conv "
            "activations out of q/k/v_proj.",
            "The generator is BLAS-free by construction, so regenerating on another host "
            "reproduces these bytes exactly given the same numpy. A C++ reference that sums "
            "sequentially instead of pairwise will differ by a few ULP; compare with a "
            "relative tolerance of about 1e-5, not bit-for-bit.",
            "Conv output is NOT bf16-rounded before entering the recurrence, matching the "
            "spec this fixture pins; the engine's causal_conv1d writes bf16."
        ],
        "prng": {
            "algorithm": "splitmix64",
            "step": "s += 0x9E3779B97F4A7C15; z=s; z=(z^(z>>30))*0xBF58476D1CE4E5B9; "
            "z=(z^(z>>27))*0x94D049BB133111EB; z^=z>>31",
            "u01": "float32((r >> 40) * (1.0/16777216.0))  # 24-bit, exact in f32",
            "uniform": "float32(lo + (hi-lo)*u01), (hi-lo)*u01 evaluated in float64",
            "fill_order": "flat row-major, one independent stream per tensor",
            "bf16_round": (
                "u = f32 bits; r = (u + 0x7FFF + ((u>>16)&1)) & 0xFFFFFFFF; "
                "r &= 0xFFFF0000; reinterpret as f32 (round-to-nearest-even)"
            ),
        },
        "prng_check": {
            "splitmix64_seed101_first4_u64_decimal": prng_u64,
            "synth_q_raw_seed101_first4_f32_hex_u32": q0_hex,
            "synth_q_raw_seed101_first4_f32_decimal": q0_dec,
            "note": "q_raw values are AFTER bf16 rounding",
        },
        "case_A_synth": {
            "prefix": "synth_",
            "all_synthetic": True,
            "initial_recurrent_state": "seed 113, non-zero",
            "initial_conv_state": "seeds 114/115/116, non-zero",
            "streams": [
                {"seed": 101, "tensor": "q_raw (conv input)", "range": [-1.0, 1.0], "shape": [T_TOTAL, H, D], "bf16": True},
                {"seed": 102, "tensor": "k_raw (conv input)", "range": [-1.0, 1.0], "shape": [T_TOTAL, H, D], "bf16": True},
                {"seed": 103, "tensor": "v_raw (conv input)", "range": [-1.0, 1.0], "shape": [T_TOTAL, H, D], "bf16": True},
                {"seed": 104, "tensor": "raw_g", "range": [-3.0, 3.0], "shape": [T_TOTAL, H, D], "bf16": True},
                {"seed": 105, "tensor": "g2_raw", "range": [-2.0, 2.0], "shape": [T_TOTAL, H, D], "bf16": True},
                {"seed": 106, "tensor": "beta_raw", "range": [-2.0, 2.0], "shape": [T_TOTAL, H], "bf16": True},
                {"seed": 107, "tensor": "A_log", "range": [-1.0, 1.5], "shape": [H], "bf16": False},
                {"seed": 108, "tensor": "dt_bias", "range": [-1.0, 1.0], "shape": [H, D], "bf16": False},
                {"seed": 109, "tensor": "q_conv_weight", "range": [-0.5, 0.5], "shape": [C, CONV_W], "bf16": True},
                {"seed": 110, "tensor": "k_conv_weight", "range": [-0.5, 0.5], "shape": [C, CONV_W], "bf16": True},
                {"seed": 111, "tensor": "v_conv_weight", "range": [-0.5, 0.5], "shape": [C, CONV_W], "bf16": True},
                {"seed": 112, "tensor": "o_norm_weight", "range": [0.5, 1.5], "shape": [D], "bf16": True},
                {"seed": 113, "tensor": "initial recurrent state S0 [h][v][c]", "range": [-1.0, 1.0], "shape": [H, D, D], "bf16": False},
                {"seed": 114, "tensor": "initial conv state q", "range": [-1.0, 1.0], "shape": [C, 3], "bf16": True},
                {"seed": 115, "tensor": "initial conv state k", "range": [-1.0, 1.0], "shape": [C, 3], "bf16": True},
                {"seed": 116, "tensor": "initial conv state v", "range": [-1.0, 1.0], "shape": [C, 3], "bf16": True},
            ],
        },
        "case_B_real": {
            "prefix": "real_",
            "real_fixture_dir": "test-data/GLM-5.3-Flash/kda-layer0",
            "revision": REVISION,
            "real_heads": REAL_HEADS,
            "head_mapping": "golden head g <- real head REAL_HEADS[g]",
            "initial_recurrent_state": "ZERO",
            "initial_conv_state": "ZERO",
            "real_params": {
                "A_log": "A_log[REAL_HEADS[g]] (f32)",
                "dt_bias": "dt_bias[rh*128:(rh+1)*128] (f32)",
                "conv_weights": "{q,k,v}_conv1d.weight rows [rh*128,(rh+1)*128), BF16 -> f32",
                "o_norm_weight": "o_norm.weight [128], BF16 -> f32, shared by all heads",
            },
            "gating_through_real_lowrank": {
                "raw_g": "bf16_round(f_a_act[t] @ f_b_real^T)  then rows [rh*128,(rh+1)*128)",
                "g2_raw": "bf16_round(g_a_act[t] @ g_b_real^T)  then rows [rh*128,(rh+1)*128)",
                "gemm_dtype": (
                    "float32 accumulation; result rounded to bf16 to mirror the engine "
                    "dataflow (vLLM's GEMM output is the model dtype). Each output element "
                    "is an elementwise product plus a numpy pairwise sum, not a BLAS sgemm, "
                    "so the fixture is reproducible independently of the host BLAS."
                ),
            },
            "streams": [
                {"seed": 301, "tensor": "f_a_act", "range": [-1.0, 1.0], "shape": [T_TOTAL, D], "bf16": True},
                {"seed": 302, "tensor": "g_a_act", "range": [-1.0, 1.0], "shape": [T_TOTAL, D], "bf16": True},
                {"seed": 303, "tensor": "q_raw (conv input)", "range": [-1.0, 1.0], "shape": [T_TOTAL, H, D], "bf16": True},
                {"seed": 304, "tensor": "k_raw (conv input)", "range": [-1.0, 1.0], "shape": [T_TOTAL, H, D], "bf16": True},
                {"seed": 305, "tensor": "v_raw (conv input)", "range": [-1.0, 1.0], "shape": [T_TOTAL, H, D], "bf16": True},
                {"seed": 306, "tensor": "beta_raw", "range": [-2.0, 2.0], "shape": [T_TOTAL, H], "bf16": True},
            ],
            "notes": (
                "Activations are injected POST-projection by design: no real b_proj / "
                "f_a_proj / g_a_proj / q_proj / k_proj / v_proj GEMMs from hidden states "
                "are exercised here. b_proj, f_a_proj and g_a_proj are still fetched into "
                "the layer-0 fixture for completeness but are unused by this generator; "
                "only f_b_proj, g_b_proj, the three conv1d weights, A_log, dt_bias and "
                "o_norm.weight feed the goldens."
            ),
        },
        "cases": manifest_cases,
    }

    mpath = os.path.join(args.out, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print("\nwrote %d golden bins + manifest -> %s" % (len(all_files), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

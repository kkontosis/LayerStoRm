#!/usr/bin/env python3
"""TD-GOLDEN: layer-level Python reference forward + engine dump comparison.

Computes a DeepSeek-V3.2 FP32 reference forward (math adapted from the
official DeepSeek-V3 reference, ref/DeepSeek-V3/inference/model.py; MIT
License, Copyright (c) 2023 DeepSeek — see THIRD_PARTY_NOTICES.md — naive
non-absorbed MLA, YaRN RoPE, sigmoid+bias group-top-k gating; the V3.2 DSA
indexer is a no-op at prompt length << index_topk and is skipped) on the SAME
safetensors checkpoint the engine loads, and compares it per (position,
layer, stage) against the engine buffers dumped by
RoutedExpertTest.GoldenDumpDeterminism (tests/integration/first_token_test.cpp).

Modes:
  trajectory — full reference forward of the prompt; per-stage cosine vs the
               dumps. Error accumulates across layers; look for the first
               cliff. Also validates the reference itself (must predict
               " Paris" = 11111 at the last position).
  transfer   — per layer L, feed the ENGINE's dumped layer-L inputs through
               the reference layer (KV history rebuilt from those same
               inputs) and compare the layer's outputs. No error
               accumulation: the first stage whose cosine collapses is the
               broken component. Attention and FFN are isolated separately
               (FFN reference consumes the ENGINE's post-attn dump).

Usage:
  .venv/bin/python tools/golden_ref/ref_forward.py \
      --model-dir test-data/DeepSeek-V3.2-NVFP4 \
      --dump-dir /tmp/td_golden_dumps/pass1 --mode transfer
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from safetensors_lite import ShardedSafetensors  # noqa: E402
import nvfp4  # noqa: E402

torch.set_grad_enabled(False)


# ─── dump file readers ──────────────────────────────────────────────────────

def load_bf16(path):
    raw = np.fromfile(path, dtype=np.uint8)
    return torch.from_numpy(raw).view(torch.bfloat16).float()


def load_f32(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).copy())


def load_routing(path):
    raw = open(path, "rb").read()
    num_tokens, topk, layer_idx, _ = struct.unpack("<4I", raw[:16])
    n = num_tokens * topk
    weights = np.frombuffer(raw[16:16 + 4 * n], dtype=np.float32)
    indices = np.frombuffer(raw[16 + 4 * n:16 + 8 * n], dtype=np.int32)
    return layer_idx, topk, weights.copy(), indices.copy()


# ─── metrics ────────────────────────────────────────────────────────────────

def metrics(ref, eng):
    """Returns (cosine, rel_l2, norm_ratio engine/ref)."""
    ref = ref.flatten().double()
    eng = eng.flatten().double()
    nr, ne = ref.norm().item(), eng.norm().item()
    if nr == 0.0 and ne == 0.0:
        return 1.0, 0.0, 1.0
    if nr == 0.0 or ne == 0.0:
        return 0.0, 1.0, float("inf") if nr == 0 else 0.0
    cos = float((ref @ eng) / (nr * ne))
    rel = float((ref - eng).norm() / nr)
    return cos, rel, ne / nr


# ─── reference model ────────────────────────────────────────────────────────

class RefModel:
    def __init__(self, model_dir, yarn=True):
        self.st = ShardedSafetensors(model_dir)
        cfg = json.load(open(Path(model_dir) / "config.json"))
        self.cfg = cfg
        self.eps = cfg["rms_norm_eps"]
        self.H = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.q_lora = cfg["q_lora_rank"]
        self.kv_lora = cfg["kv_lora_rank"]
        self.d_nope = cfg["qk_nope_head_dim"]
        self.d_rope = cfg["qk_rope_head_dim"]
        self.d_v = cfg["v_head_dim"]
        self.qk_dim = self.d_nope + self.d_rope
        self.first_moe = cfg["first_k_dense_replace"]
        self.n_experts = cfg["n_routed_experts"]
        self.topk = cfg["num_experts_per_tok"]
        self.n_group = cfg["n_group"]
        self.topk_group = cfg["topk_group"]
        self.route_scale = cfg["routed_scaling_factor"]
        self.num_layers = cfg["num_hidden_layers"]

        rs = cfg.get("rope_scaling") or {}
        self.yarn = yarn and rs.get("type") == "yarn"
        self.softmax_scale = self.qk_dim ** -0.5
        if self.yarn:
            mscale = 0.1 * rs.get("mscale", 1.0) * math.log(rs["factor"]) + 1.0
            self.softmax_scale *= mscale * mscale
        self._freqs_cache = None
        self._rope_params = rs

    # — weights —

    def w(self, name):
        return self.st.tensor_f32(name)

    def wq4(self, name):
        """Dequant an NVFP4 weight (name without .weight suffix)."""
        return nvfp4.dequant(self.st.tensor(name + ".weight"),
                             self.st.tensor(name + ".weight_scale"),
                             self.st.tensor(name + ".weight_scale_2"))

    def linear_w(self, name):
        """Weight for `name` regardless of stored precision → FP32 [N, K]."""
        dt, _ = self.st.meta(name + ".weight")
        return self.wq4(name) if dt == "U8" else self.w(name + ".weight")

    # — math blocks (DeepSeek-V3 reference math) —

    def rms_norm(self, x, w):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def freqs_cis(self, n_pos):
        """YaRN-corrected complex rotary table [n_pos, d_rope/2].

        Adapted from precompute_freqs_cis in ref/DeepSeek-V3/inference/model.py.
        """
        if self._freqs_cache is not None and self._freqs_cache.shape[0] >= n_pos:
            return self._freqs_cache[:n_pos]
        dim, base = self.d_rope, self.cfg["rope_theta"]
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if self.yarn:
            rs = self._rope_params
            factor = rs["factor"]
            orig = rs["original_max_position_embeddings"]
            beta_fast, beta_slow = rs.get("beta_fast", 32), rs.get("beta_slow", 1)

            def corr_dim(num_rot):
                return (dim * math.log(orig / (num_rot * 2 * math.pi))
                        / (2 * math.log(base)))

            low = max(math.floor(corr_dim(beta_fast)), 0)
            high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, dtype=torch.float32) - low)
                / max(high - low, 0.001), 0, 1)
            smooth = 1 - ramp
            freqs = freqs / factor * (1 - smooth) + freqs * smooth
        t = torch.arange(max(n_pos, 64), dtype=torch.float32)
        table = torch.polar(torch.ones(t.numel(), dim // 2),
                            torch.outer(t, freqs))
        self._freqs_cache = table
        return table[:n_pos]

    @staticmethod
    def rope(x, freqs):
        """x [S, h, d] (d even, consecutive pairs), freqs [S, d/2] complex."""
        s, h, d = x.shape
        xc = torch.view_as_complex(x.float().reshape(s, h, d // 2, 2))
        out = torch.view_as_real(xc * freqs.view(s, 1, d // 2)).reshape(s, h, d)
        return out

    def attn_block(self, layer, x):
        """x [S, H] residual stream entering layer (positions 0..S-1, causal).

        Returns (post_attn [S, H], internals dict).
        """
        p = f"model.layers.{layer}"
        a = f"{p}.self_attn"
        hn = self.rms_norm(x, self.w(f"{p}.input_layernorm.weight"))

        q = self.rms_norm(hn @ self.linear_w(f"{a}.q_a_proj").T,
                          self.w(f"{a}.q_a_layernorm.weight"))
        q = (q @ self.linear_w(f"{a}.q_b_proj").T)
        s = x.shape[0]
        q = q.reshape(s, self.n_heads, self.qk_dim)
        q_nope, q_pe = q.split([self.d_nope, self.d_rope], dim=-1)
        freqs = self.freqs_cis(s)
        q_pe = self.rope(q_pe, freqs)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = hn @ self.linear_w(f"{a}.kv_a_proj_with_mqa").T
        c_kv, k_pe = kv.split([self.kv_lora, self.d_rope], dim=-1)
        k_pe = self.rope(k_pe.unsqueeze(1), freqs)          # [S, 1, d_rope]
        c_kv = self.rms_norm(c_kv, self.w(f"{a}.kv_a_layernorm.weight"))
        kvb = (c_kv @ self.linear_w(f"{a}.kv_b_proj").T)
        kvb = kvb.reshape(s, self.n_heads, self.d_nope + self.d_v)
        k_nope, v = kvb.split([self.d_nope, self.d_v], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, self.n_heads, -1)], dim=-1)

        scores = torch.einsum("shd,thd->sht", q, k) * self.softmax_scale
        mask = torch.full((s, s), float("-inf")).triu_(1)   # causal
        scores = scores + mask.view(s, 1, s)
        scores = scores.softmax(dim=-1, dtype=torch.float32)
        o = torch.einsum("sht,thd->shd", scores, v).reshape(s, -1)
        hidden_out = o @ self.linear_w(f"{a}.o_proj").T     # pre-residual
        return x + hidden_out, {
            "normed_hidden": hn, "q": q, "hidden_out": hidden_out,
        }

    def _mlp(self, prefix, x):
        gate = x @ self.linear_w(f"{prefix}.gate_proj").T
        up = x @ self.linear_w(f"{prefix}.up_proj").T
        return (torch.nn.functional.silu(gate) * up) \
            @ self.linear_w(f"{prefix}.down_proj").T

    def gate_topk(self, layer, hn):
        """V3.2 routing: sigmoid + e_score_correction_bias selection, group
        top-k; combine weights are UNBIASED sigmoid scores, renormalized,
        scaled by routed_scaling_factor. Returns (weights, indices,
        router_logits) for hn [S, H]."""
        p = f"model.layers.{layer}.mlp.gate"
        logits = hn @ self.w(f"{p}.weight").T               # [S, E]
        scores = logits.sigmoid()
        original = scores
        scores = scores + self.w(f"{p}.e_score_correction_bias")
        s = hn.shape[0]
        g = scores.view(s, self.n_group, -1)
        group_scores = g.topk(2, dim=-1)[0].sum(dim=-1)     # bias present
        gidx = group_scores.topk(self.topk_group, dim=-1)[1]
        mask = torch.ones(s, self.n_group, dtype=torch.bool) \
            .scatter_(1, gidx, False)
        g = g.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(g, self.topk, dim=-1)[1]
        weights = original.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices, logits

    def ffn_block(self, layer, x):
        """x [S, H] post-attn residual stream. Returns (post_moe, internals)."""
        p = f"model.layers.{layer}"
        hn = self.rms_norm(x, self.w(f"{p}.post_attention_layernorm.weight"))
        info = {"normalized_hidden": hn}
        if layer < self.first_moe:
            out = self._mlp(f"{p}.mlp", hn)
        else:
            weights, indices, logits = self.gate_topk(layer, hn)
            info.update(routing_weights=weights, routing_indices=indices,
                        router_logits=logits)
            y = torch.zeros_like(hn)
            for e in indices.unique().tolist():
                rows, k = torch.where(indices == e)
                y[rows] += self._mlp(f"{p}.mlp.experts.{e}", hn[rows]) \
                    * weights[rows, k, None]
            y = y + self._mlp(f"{p}.mlp.shared_experts", hn)
            out = y
        return x + out, info

    def embed(self, tokens):
        emb = self.st.tensor("model.embed_tokens.weight")
        return emb[torch.tensor(tokens)].float()

    def head(self, x):
        hn = self.rms_norm(x, self.w("model.norm.weight"))
        return hn @ self.st.tensor_f32("lm_head.weight").T


# ─── comparison drivers ─────────────────────────────────────────────────────

def fmt(vals):
    return "[" + " ".join(f"{v:.3f}" for v in vals) + "]"


class Reporter:
    def __init__(self, threshold):
        self.threshold = threshold
        self.failures = []   # (layer, stage, pos, cos)

    def check(self, layer, stage, pos, ref, eng):
        cos, rel, ratio = metrics(ref, eng)
        if cos < self.threshold:
            self.failures.append((layer, stage, pos, cos, rel, ratio))
        return cos, rel, ratio

    def summary(self):
        if not self.failures:
            print(f"\nALL STAGES HEALTHY (cosine >= {self.threshold})")
            return
        self.failures.sort(key=lambda f: (f[0], f[2]))
        first = self.failures[0]
        print(f"\n{len(self.failures)} stage comparisons below "
              f"threshold {self.threshold}.")
        print(f"FIRST DIVERGENT STAGE: layer={first[0]} stage={first[1]} "
              f"pos={first[2]} cos={first[3]:.4f} rel_l2={first[4]:.4f} "
              f"norm_ratio(engine/ref)={first[5]:.4f}")
        print("worst 15 by layer order:")
        for L, st, p, c, r, nr in self.failures[:15]:
            print(f"  L{L:02d} {st:<18} pos{p} cos={c:+.4f} rel={r:.4f} "
                  f"norm_ratio={nr:.4f}")


def stage_paths(dump, pos, layer):
    base = dump / f"pos{pos}"
    return {
        "post_attn": base / f"L{layer}.post_attn.bin",
        "post_moe":  base / f"L{layer}.post_moe.bin",
        "routing":   base / f"L{layer}.routing.bin",
        "hidden_out": base / f"L{layer}.dcp.hidden_out.rank0.bin",
        "normed_hidden": base / f"L{layer}.dcp.normed_hidden.rank0.bin",
        "moe_normalized": base / f"L{layer}.moe.normalized_hidden.bin",
        "router_logits": base / f"L{layer}.moe.router_logits.bin",
    }


def compare_layer(model, rep, dump, layer, x_attn_in, positions, H,
                  transfer=True):
    """Compare one reference layer against engine dumps.

    x_attn_in: [S, H] inputs to the layer (engine dumps in transfer mode,
    reference trajectory in trajectory mode). In transfer mode the FFN
    consumes the ENGINE's post-attn dumps (isolates FFN from attention
    errors); in trajectory mode it consumes the reference's own post-attn
    (pure trajectory). Returns reference post_moe and prints one line.
    """
    post_attn_ref, ai = model.attn_block(layer, x_attn_in)

    attn_cos, ho_cos = [], []
    eng_post_attn = []
    for i, pos in enumerate(positions):
        sp = stage_paths(dump, pos, layer)
        eng = load_bf16(sp["post_attn"])[:H]
        eng_post_attn.append(eng)
        c, _, _ = rep.check(layer, "attn(post_attn)", pos,
                            post_attn_ref[i], eng)
        attn_cos.append(c)
        if sp["normed_hidden"].exists():
            c, _, _ = rep.check(layer, "attn(input_norm)", pos,
                                ai["normed_hidden"][i],
                                load_bf16(sp["normed_hidden"])[:H])
        if sp["hidden_out"].exists():
            c, _, _ = rep.check(layer, "attn(o_proj_out)", pos,
                                ai["hidden_out"][i],
                                load_bf16(sp["hidden_out"])[:H])
            ho_cos.append(c)

    # FFN stage input: engine post-attn (transfer isolation) or the
    # reference's own post-attn (pure trajectory).
    x_ffn = torch.stack(eng_post_attn) if transfer else post_attn_ref
    post_moe_ref, fi = model.ffn_block(layer, x_ffn)

    ffn_cos, route_ov, rl_cos = [], [], []
    for i, pos in enumerate(positions):
        sp = stage_paths(dump, pos, layer)
        c, _, _ = rep.check(layer, "ffn(post_moe)", pos,
                            post_moe_ref[i], load_bf16(sp["post_moe"])[:H])
        ffn_cos.append(c)
        if sp["moe_normalized"].exists():
            rep.check(layer, "ffn(post_attn_norm)", pos,
                      fi["normalized_hidden"][i],
                      load_bf16(sp["moe_normalized"])[:H])
        if "router_logits" in fi and sp["router_logits"].exists():
            c, _, _ = rep.check(layer, "ffn(router_logits)", pos,
                                fi["router_logits"][i],
                                load_f32(sp["router_logits"]))
            rl_cos.append(c)
        if "routing_indices" in fi and sp["routing"].exists():
            _, topk, _, eng_idx = load_routing(sp["routing"])
            ov = len(set(eng_idx.tolist())
                     & set(fi["routing_indices"][i].tolist()))
            route_ov.append(f"{ov}/{topk}")

    line = (f"L{layer:02d} attn{fmt(attn_cos)} ffn{fmt(ffn_cos)}")
    if ho_cos:
        line += f" o_proj{fmt(ho_cos)}"
    if rl_cos:
        line += f" router{fmt(rl_cos)}"
    if route_ov:
        line += f" topk[{' '.join(route_ov)}]"
    print(line, flush=True)
    return post_moe_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dump-dir", required=True,
                    help="a pass directory, e.g. /tmp/td_golden_dumps/pass1")
    ap.add_argument("--mode", choices=["transfer", "trajectory"],
                    default="transfer")
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.98)
    ap.add_argument("--no-yarn", action="store_true",
                    help="ablation: disable YaRN freq correction + mscale^2")
    args = ap.parse_args()

    dump = Path(args.dump_dir)
    manifest = json.load(open(dump.parent / "manifest.json"))
    tokens = manifest["prompt_tokens"]
    positions = list(range(len(tokens)))
    H = manifest["hidden_size"]

    model = RefModel(args.model_dir, yarn=not args.no_yarn)
    n_layers = manifest["num_layers"]
    if args.max_layers is not None:
        n_layers = min(n_layers, args.max_layers)
    rep = Reporter(args.threshold)

    print(f"mode={args.mode} layers=0..{n_layers - 1} "
          f"positions={positions} yarn={model.yarn} "
          f"softmax_scale={model.softmax_scale:.6f}", flush=True)

    # Stage 0: embedding (token_pos p input is tokens[p]).
    ref_embed = model.embed(tokens)
    emb_cos = []
    for pos in positions:
        eng = load_bf16(dump / f"pos{pos}" / "embed.bin")[:H]
        c, _, _ = rep.check(-1, "embedding", pos, ref_embed[pos], eng)
        emb_cos.append(c)
    print(f"EMB {fmt(emb_cos)}", flush=True)

    x = ref_embed  # trajectory state
    for layer in range(n_layers):
        if args.mode == "transfer":
            # Engine inputs to this layer: post_moe of L-1 (embed for L0).
            x_in = torch.stack([
                load_bf16(dump / f"pos{p}" /
                          ("embed.bin" if layer == 0
                           else f"L{layer - 1}.post_moe.bin"))[:H]
                for p in positions])
        else:
            x_in = x
        x = compare_layer(model, rep, dump, layer, x_in, positions, H,
                          transfer=(args.mode == "transfer"))

    # Output head: transfer comparison on the engine's pre-head hidden.
    if (dump / "pos0" / "logits.bin").exists() and n_layers == manifest["num_layers"]:
        pre = torch.stack([load_bf16(dump / f"pos{p}" / "pre_head.bin")[:H]
                           for p in positions])
        ref_logits = model.head(pre)
        head_cos = []
        for i, pos in enumerate(positions):
            eng_logits = load_f32(dump / f"pos{pos}" / "logits.bin")
            c, _, _ = rep.check(99, "output_head", pos,
                                ref_logits[i], eng_logits)
            head_cos.append(c)
            if pos == positions[-1]:
                ref_arg = int(ref_logits[i].argmax())
                eng_arg = int(eng_logits.argmax())
                print(f"final-pos argmax: ref(on engine hidden)={ref_arg} "
                      f"engine={eng_arg} expected={manifest['expected_token']}")
        print(f"HEAD {fmt(head_cos)}", flush=True)

        if args.mode == "trajectory":
            traj_logits = model.head(x)
            arg = int(traj_logits[-1].argmax())
            top1 = float(traj_logits[-1].softmax(-1).max())
            ok = arg == manifest["expected_token"]
            print(f"REFERENCE VALIDATION: full-trajectory argmax at last pos "
                  f"= {arg} (top1_prob={top1:.3f}) "
                  f"{'== expected — reference VALID' if ok else '!= expected '+str(manifest['expected_token'])+' — INVESTIGATE REFERENCE'}")

    rep.summary()


if __name__ == "__main__":
    main()

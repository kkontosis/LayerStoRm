#!/usr/bin/env python3
"""TD-GOLDEN zoom: forensic sub-stage comparison of layer 0 attention.

Uses the dcp.* stage dumps (pos0-2, rank0) to walk the engine's MLA pipeline
stage by stage against the FP32 reference, then simulates specific bug
candidates for the W_UV (kv_b_v) extraction and o_proj consumption and
correlates each against the ENGINE's hidden_out dump. The candidate with
cosine ~1 identifies the bug.

Engine stage semantics (src/parallelism/dcp_executor.cpp execute paths):
  dcp.normed_hidden  [B,7168]  input_layernorm(h)
  dcp.q_compressed   [B,1536]  q_a_layernorm(q_a_proj(hn))     (post-norm)
  dcp.q_heads        [B,64,192] q_b_proj out, PRE-rope (rank0 = heads 0-63)
  dcp.q_absorbed     [B,64,576] q_nope @ W_UK  ++  rope(q_pe)
  dcp.kv_compressed  [B,576]   [kv_a_layernorm(c_kv), rope(k_pe)]
  dcp.prefill_out    [B,64,512] compressed attention out (post DCP correction)
  dcp.hidden_out     [B,7168]  o_proj out, post TP-allreduce, pre-residual
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from ref_forward import RefModel, load_bf16, metrics  # noqa: E402

torch.set_grad_enabled(False)

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "test-data/DeepSeek-V3.2-NVFP4"
DUMP = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/td_golden_dumps/pass1")
LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 0
POSITIONS = [0, 1, 2]

m = RefModel(MODEL_DIR)
HL = m.n_heads // 2          # rank0 head count under TP=2
P, V, D_c, R = m.d_nope, m.d_v, m.kv_lora, m.d_rope


def rep(name, ref, eng):
    cos, rel, ratio = metrics(ref, eng)
    print(f"  {name:<28} cos={cos:+.4f} rel={rel:.4f} "
          f"norm_ratio(eng/ref)={ratio:.4f}")
    return cos


def dump(pos, name):
    return load_bf16(DUMP / f"pos{pos}" / f"L{LAYER}.dcp.{name}.rank0.bin")


a = f"model.layers.{LAYER}.self_attn"
w_q_a = m.linear_w(f"{a}.q_a_proj")
w_q_b = m.linear_w(f"{a}.q_b_proj")
w_kv_a = m.linear_w(f"{a}.kv_a_proj_with_mqa")
kv_b = m.linear_w(f"{a}.kv_b_proj")           # [128*(P+V), D_c]
w_o = m.linear_w(f"{a}.o_proj")               # [7168, 128*V]
q_a_ln = m.w(f"{a}.q_a_layernorm.weight")
kv_a_ln = m.w(f"{a}.kv_a_layernorm.weight")
in_ln = m.w(f"model.layers.{LAYER}.input_layernorm.weight")

kv_b_h = kv_b.reshape(m.n_heads, P + V, D_c)
W_UK = kv_b_h[:, :P, :]                       # [128, P, D_c]
W_UV = kv_b_h[:, P:, :]                       # [128, V, D_c]

x_in = torch.stack([
    load_bf16(DUMP / f"pos{p}" /
              ("embed.bin" if LAYER == 0 else f"L{LAYER-1}.post_moe.bin"))[:m.H]
    for p in POSITIONS])
S = len(POSITIONS)
freqs = m.freqs_cis(S)

# ── reference sub-stages ────────────────────────────────────────────────────
hn = m.rms_norm(x_in, in_ln)
q_c = m.rms_norm(hn @ w_q_a.T, q_a_ln)
q_heads = (q_c @ w_q_b.T).reshape(S, m.n_heads, P + R)
q_nope, q_pe = q_heads.split([P, R], dim=-1)
q_pe_rot = m.rope(q_pe, freqs)
# q_absorbed = q_nope @ W_UK per head ++ rotated rope
q_abs = torch.cat([
    torch.einsum("shp,hpc->shc", q_nope, W_UK), q_pe_rot], dim=-1)

kv = hn @ w_kv_a.T
c_kv_raw, k_pe = kv.split([D_c, R], dim=-1)
c_kv = m.rms_norm(c_kv_raw, kv_a_ln)
k_pe_rot = m.rope(k_pe.unsqueeze(1), freqs).squeeze(1)
kv_comp_ref = torch.cat([c_kv, k_pe_rot], dim=-1)        # engine layout

# compressed-space attention (what prefill_out should hold)
k_comp = kv_comp_ref                                     # [S, 576]
scores = torch.einsum("shc,tc->sht", q_abs, k_comp) * m.softmax_scale
mask = torch.full((S, S), float("-inf")).triu_(1)
scores = (scores + mask.view(S, 1, S)).softmax(-1, dtype=torch.float32)
comp_out = torch.einsum("sht,tc->shc", scores, c_kv)     # [S, 128, D_c]

v_ref = torch.einsum("shc,hvc->shv", comp_out, W_UV)     # [S, 128, V]
hidden_out_ref = v_ref.reshape(S, -1) @ w_o.T

print(f"=== L{LAYER} sub-stage walk (rank0 = heads 0-{HL-1}) ===")
for i, pos in enumerate(POSITIONS):
    print(f"pos{pos}:")
    rep("normed_hidden", hn[i], dump(pos, "normed_hidden")[:m.H])
    rep("q_compressed(post-norm)", q_c[i], dump(pos, "q_compressed")[:m.q_lora])
    rep("q_heads(pre-rope)", q_heads[i, :HL],
        dump(pos, "q_heads")[:HL * (P + R)].reshape(HL, P + R))
    rep("q_absorbed", q_abs[i, :HL],
        dump(pos, "q_absorbed")[:HL * (D_c + R)].reshape(HL, D_c + R))
    rep("kv_compressed(norm+rope)", kv_comp_ref[i],
        dump(pos, "kv_compressed")[:D_c + R])
    rep("prefill_out(comp attn)", comp_out[i, :HL],
        dump(pos, "prefill_out")[:HL * D_c].reshape(HL, D_c))
    rep("hidden_out(o_proj+AR)", hidden_out_ref[i],
        dump(pos, "hidden_out")[:m.H])

# ── bug-candidate simulation against engine hidden_out ────────────────────
# Uses the ENGINE's prefill_out for heads 0-63; heads 64-127 use the
# reference comp_out (at pos0 all heads are identical = c_kv, so this is
# exact; at pos>0 it is approximate for the rank1 half).
print(f"\n=== candidate simulation vs ENGINE hidden_out ===")
for i, pos in enumerate(POSITIONS):
    eng_comp = dump(pos, "prefill_out")[:HL * D_c].reshape(HL, D_c)
    comp_full = torch.cat([eng_comp, comp_out[i, HL:]], dim=0)  # [128, D_c]
    eng_hidden = dump(pos, "hidden_out")[:m.H]

    def sim(name, v_full):
        h = v_full.reshape(-1) @ w_o.T
        cos, _, ratio = metrics(h, eng_hidden)
        print(f"  pos{pos} {name:<34} cos={cos:+.4f} ratio={ratio:.4f}")

    # C1 correct math on engine comp_out
    sim("C1 correct W_UV", torch.einsum("hc,hvc->hv", comp_full, W_UV))
    # C2 W_UK used instead of W_UV (missing P-row offset)
    sim("C2 W_UK instead of W_UV", torch.einsum("hc,hvc->hv", comp_full, W_UK))
    # C3 head stride bug: rows P + h*V contiguous (stride V*D_c not (P+V)*D_c)
    flat = kv_b.reshape(-1, D_c)
    per_rank = []
    for r0 in range(2):
        local = flat[r0 * (m.n_heads // 2) * (P + V):
                     (r0 + 1) * (m.n_heads // 2) * (P + V)]
        wuv_bug = torch.stack([local[P + h * V: P + (h + 1) * V]
                               for h in range(HL)])
        per_rank.append(wuv_bug)
    wuv_c3 = torch.cat(per_rank, dim=0)                  # [128, V, D_c]
    sim("C3 stride V*D_c (no P+V skip)", torch.einsum("hc,hvc->hv", comp_full, wuv_c3))
    # C4 W_UV transposed (D_c x V read as V x D_c) — dims differ (512 vs 128)
    # only valid square-wise; approximate by transposing the [V, D_c] block
    # in its first VxV corner is meaningless — skip true transpose; instead:
    # C4 o_proj head-interleave swap: act laid out [V, HL] instead of [HL, V]
    v_ok = torch.einsum("hc,hvc->hv", comp_full, W_UV)   # [128, V]
    act_swapped = torch.cat([v_ok[:HL].T.reshape(-1), v_ok[HL:].T.reshape(-1)])
    cos, _, ratio = metrics(act_swapped @ w_o.T, eng_hidden)
    print(f"  pos{pos} {'C4 o_proj act [V,HL] swap':<34} cos={cos:+.4f} ratio={ratio:.4f}")
    # C5 v from UNNORMALIZED c_kv (kv_a_layernorm missing on value path)
    if pos == 0:
        v_unnorm = torch.einsum("c,hvc->hv", c_kv_raw[i], W_UV)
        sim("C5 W_UV @ unnormalized c_kv", v_unnorm.expand_as(v_ok).clone()
            if v_unnorm.dim() == 2 else v_unnorm)
print("done")

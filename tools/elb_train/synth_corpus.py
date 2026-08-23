"""EPM-2 synthetic corpus: raw EPM-1-format dumps with a KNOWN generating
process (Phase 29).  Used by the baseline unit tests and by the synthetic
run of run_baselines.py while the real ≥100k-block corpus is pending
(TD-EPM-CORPUS).

Generating process (per sequence, all deterministic under `seed`):
  - One latent chain PER TAP t ∈ [0, n_taps): z_t[p] evolves over decode
    positions p with AR(1) temporal correlation rho:
        z_t[p] = rho * z_t[p-1] + sqrt(1 - rho^2) * eps
    (unit marginal scale — rho=1 makes routing constant over positions,
    rho=0 makes consecutive positions independent: B0's planted knob).
  - Each MoE layer j is ASSIGNED a planted tap t*(j); its true router
    (W_j [E, H], bias_j) scores the tap's latent:
        logits_j[p] = z_{t*(j)}[p] @ W_j.T
    labels = noaux_tc_topk(logits_j, bias_j)  (the exact engine semantics,
    glm_router.noaux_tc_topk) — EPMR records carry these for EVERY decoded
    position, exactly like the engine dump.
  - Draft features for a block anchored at a: x[k, t] = z_t[a + k] +
    tap_noise * eps, stored as BF16 bits (RNE, like the engine).  So B1
    with the TRUE router at the planted tap achieves recall ~1.0 at
    tap_noise=0 (exactly 1.0: labels and predictions run the same selection
    on the same logits — BF16 feature rounding is the only gap, eliminated
    by construction below at tap_noise=0 where z is drawn pre-rounded from
    BF16-representable values); a wrong/shuffled router or wrong tap
    degrades to chance; B2's planted linear map (h -> h @ W_j.T) is exactly
    recoverable by the ridge probe.
  - Sequential early-stop label coverage: block b accepts a_b draft tokens
    (drawn from `accept_probs` survival), verify feeds positions
    a .. a + a_b (a_b + 1 rows, capped at gamma), the next block anchors at
    a + a_b + 1 — routing records tile the position axis exactly like the
    engine's accepted stream, and label coverage decays with k.

Outputs one run-unit directory: epm_blocks.bin + epm_routing.bin +
manifest.jsonl (same layouts as src/speculation/epm_dump.h /
python/orchestrator/epm_manifest.py), plus the ground truth returned to the
caller (`routers` RouterBank of the TRUE routers, `true_taps` [J]).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import glm_router

EPMB_MAGIC = 0x424D5045
EPMR_MAGIC = 0x524D5045


def _pack_block(seq, blk, anchor, anchor_tok, gamma, n_layers, hidden,
                ids, conf, hiddens_u16) -> bytes:
    out = struct.pack("<IIQIIIIIII", EPMB_MAGIC, 1, seq, blk, anchor,
                      anchor_tok, gamma, n_layers, hidden,
                      1 if conf is not None else 0)
    out += np.asarray(ids, np.int32).tobytes()
    if conf is not None:
        out += np.asarray(conf, np.float32).tobytes()
    out += np.asarray(hiddens_u16, np.uint16).tobytes()
    return out


def _pack_route(seq, pos, layers, logits_f16, top_ids, top_w) -> bytes:
    rows = len(layers)
    n_e = logits_f16.shape[1]
    k = top_ids.shape[1]
    out = struct.pack("<IIQIIII", EPMR_MAGIC, 1, seq, pos, rows, n_e, k)
    for r in range(rows):
        out += struct.pack("<I", int(layers[r]))
        out += np.asarray(logits_f16[r], np.float16).tobytes()
        out += np.asarray(top_ids[r], np.int32).tobytes()
        out += np.asarray(top_w[r], np.float32).tobytes()
    return out


def generate_synthetic_run(
    dump_dir: str | Path,
    *,
    n_seqs: int = 8,
    blocks_per_seq: int = 6,
    gamma: int = 4,
    n_taps: int = 5,
    hidden: int = 24,
    moe_layers: tuple[int, ...] = (3, 4, 5, 10, 11, 12),
    n_experts: int = 32,
    topk: int = 8,
    rho: float = 0.9,
    tap_noise: float = 0.0,
    accept_probs: tuple[float, ...] | None = None,
    routed_scaling_factor: float = 2.5,
    seed: int = 0,
    write_manifest: bool = True,
    feature_map: str = "identity",
    tap_assignment: str = "round_robin",
) -> dict:
    """Write one synthetic run unit; returns the ground truth:
    {"routers": RouterBank (TRUE routers), "true_taps": int32 [J],
     "feature_maps": f32 [n_taps, H, H] or None, "n_blocks": int,
     "moe_layers": [J]}.

    accept_probs[k] = P(draft position k accepted | k-1 accepted); default
    a survival ladder (0.7, 0.5, 0.4, ...) so coverage decays with k.

    feature_map: "identity" (features = the latent the labels score —
    Tier 0 with the true router is perfect) or "linear" (EPM-3's planted
    RECOVERABLE map: features = z @ A_t with A_t a random per-tap
    ORTHOGONAL matrix — Tier 0 degrades to ~chance while a trained linear
    adapter/head can recover A_t^{-1} exactly, and feature norms are
    preserved so normalization behaves identically).

    tap_assignment: "round_robin" (default: t*(j) = j mod n_taps) or
    "prior" (t*(j) = the aux-segment prior mapping — configs that keep
    taps.map="prior" then start from the CORRECT tap).
    """
    if feature_map not in ("identity", "linear"):
        raise ValueError(f"unknown feature_map {feature_map!r}")
    if tap_assignment not in ("round_robin", "prior"):
        raise ValueError(f"unknown tap_assignment {tap_assignment!r}")
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    j = len(moe_layers)
    if accept_probs is None:
        accept_probs = tuple(max(0.1, 0.7 - 0.15 * k)
                             for k in range(gamma - 1))

    # True routers + planted tap per layer (round-robin so every tap owns
    # at least one layer when j >= n_taps).
    weight = (rng.standard_normal((j, n_experts, hidden))
              / np.sqrt(hidden)).astype(np.float32)
    bias = (0.02 * rng.standard_normal((j, n_experts))).astype(np.float32)
    if tap_assignment == "prior":
        true_taps = glm_router.aux_prior_tap(moe_layers).astype(np.int32)
        true_taps = np.minimum(true_taps, n_taps - 1)
    else:
        true_taps = np.array([i % n_taps for i in range(j)], np.int32)
    feature_maps = None
    if feature_map == "linear":
        a = rng.standard_normal((n_taps, hidden, hidden))
        feature_maps = np.stack([np.linalg.qr(a[t])[0]
                                 for t in range(n_taps)]).astype(np.float32)
    bank = glm_router.RouterBank(
        moe_layers=np.asarray(moe_layers, np.int32), weight=weight,
        bias=bias)

    blocks_bin = b""
    routes_bin = b""
    manifest: list[dict] = []
    n_blocks = 0

    for seq in range(1, n_seqs + 1):
        # Latent chains, drawn BF16-representable so that at tap_noise=0 the
        # feature dump is EXACTLY the latent the labels were computed from.
        n_pos = 1 + blocks_per_seq * (gamma + 1) + gamma  # generous
        z = np.empty((n_pos, n_taps, hidden), np.float32)
        z[0] = glm_router.bf16_bits_to_f32(glm_router.f32_to_bf16_bits(
            rng.standard_normal((n_taps, hidden)).astype(np.float32)))
        for p in range(1, n_pos):
            step = (rho * z[p - 1] + np.sqrt(max(0.0, 1.0 - rho * rho))
                    * rng.standard_normal((n_taps, hidden))).astype(
                        np.float32)
            z[p] = glm_router.bf16_bits_to_f32(
                glm_router.f32_to_bf16_bits(step))

        def routing_at(p: int):
            lat = z[p, true_taps]                      # [J, H]
            logits = np.einsum("jh,jeh->je", lat, weight)  # [J, E]
            # Labels go through FP16 exactly like the engine dump; top-8 is
            # computed on the PRE-fp16 logits (the engine selects on the
            # device logits and dumps fp16 copies).
            ids, w = glm_router.noaux_tc_topk(
                logits, bias, topk=topk,
                routed_scaling_factor=routed_scaling_factor)
            return logits.astype(np.float16), ids, w

        # Warm-up decoded position 0 (pre-first-block: gives B0-prev a
        # source at the first block's k=0).
        layers_arr = np.asarray(moe_layers, np.uint32)
        lg, ids, w = routing_at(0)
        routes_bin += _pack_route(seq, 0, layers_arr, lg, ids, w)

        anchor = 1
        for blk in range(blocks_per_seq):
            if anchor + gamma >= n_pos:
                break
            # Draft features: x[k, t] = map_t(z[anchor + k, t]) + noise.
            feats = z[anchor:anchor + gamma].copy()   # [gamma, taps, H]
            if feature_maps is not None:
                feats = np.einsum("kth,thg->ktg", feats, feature_maps,
                                  optimize=True).astype(np.float32)
            if tap_noise > 0.0:
                feats = feats + tap_noise * rng.standard_normal(
                    feats.shape).astype(np.float32)
            feats_u16 = glm_router.f32_to_bf16_bits(feats)

            draft_ids = rng.integers(0, 1000, gamma).astype(np.int32)
            conf = rng.random(gamma).astype(np.float32)
            blocks_bin += _pack_block(seq, blk, anchor, 7, gamma, n_taps,
                                      hidden, draft_ids, conf, feats_u16)
            n_blocks += 1

            # Sequential early-stop: accepted length from the survival
            # ladder; verify feeds positions anchor .. anchor + accepted.
            accepted = 0
            for k in range(gamma - 1):
                if rng.random() < accept_probs[k]:
                    accepted += 1
                else:
                    break
            fed = min(accepted + 1, gamma)
            for k in range(fed):
                lg, ids, w = routing_at(anchor + k)
                routes_bin += _pack_route(seq, anchor + k, layers_arr,
                                          lg, ids, w)
            if write_manifest:
                manifest.append({
                    "seq_id": seq, "anchor_pos": int(anchor),
                    "block_idx": blk, "request_id": 1, "gamma": gamma,
                    "draft_tokens": [int(t) for t in draft_ids],
                    "verify_tokens": [int(t) for t in draft_ids[:fed]],
                    "tokens_match": True, "accepted_len": accepted,
                    "attempted_len": fed,
                    "confidences": [float(c) for c in conf], "ts": 0.0})
            anchor += accepted + 1

    (dump_dir / "epm_blocks.bin").write_bytes(blocks_bin)
    (dump_dir / "epm_routing.bin").write_bytes(routes_bin)
    if write_manifest:
        with open(dump_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
            for e in manifest:
                f.write(json.dumps(e) + "\n")

    return {"routers": bank, "true_taps": true_taps,
            "feature_maps": feature_maps, "n_blocks": n_blocks,
            "moe_layers": list(moe_layers)}

"""EPM-2 zero-training baselines B0/B1/B2 (Phase 29, MoE-SpeQ_NOTES.md §7).

All baselines evaluate through metrics.EvalAccumulator (coverage-aware,
per-(position k, layer j) cells; acceptance-conditioned by construction via
label_mask).  Splits are ALWAYS sequence-level (INV-EPM-DATA) — callers pass
seq_keys from dataset.sequence_split.

B0 — temporal locality.  Predict position k's layer-j top-8 from routing the
    engine already computed, read DIRECTLY from epm_routing.bin (EPMR
    records exist for EVERY decoded position, not just block positions).
    Two variants:
      prev:   source = the previous DECODED position's record
              (anchor_pos + k - 1).  For k >= 1 that source is only
              materialized DURING this block's verify pass — a streaming/
              oracle-leaning bound on temporal locality, not a block-start
              predictor.
      anchor: source = the last record known AT BLOCK START
              (anchor_pos - 1, the previous block's final verified feed) for
              ALL k — the causal block-start variant.
    Prediction width is 8 (the record's top-8) -> recall@16/32 are absent
    (n=0).  Calibration scores: top_w / routed_scaling_factor (the
    renormalized gate weights mapped back to a [0,1] simplex share) — a
    monotone proxy, NOT a probability; ECE for B0 is a diagnostic only.

B1 — Tier-0 frozen-router probe.  Apply the target's frozen router_j
    (glm_router.RouterBank) to RAW draft hiddens (no normalization — Tier-0
    contract), full noaux_tc semantics.  Full tap sweep: recall for every
    (tap l, layer j); best-tap-per-layer selected on the TRAIN split
    (n-weighted recall@8 over positions, ties -> lower tap), reported on
    held-out; the aux-segment prior mapping (glm_router.aux_prior_tap) is
    the comparison arm.  Scores = unbiased sigmoid scores (probability-like
    -> ECE meaningful).

B2 — per-layer full-rank linear-probe skyline.  DIRECT 6144 -> 256-logit
    probe (chosen over 6144 -> 6144 + frozen router: closed-form on the
    exact target quantity, the true upper bound for any linear feature map
    feeding the frozen router — the router itself is linear pre-sigmoid, so
    direct-logit ridge dominates adapter+router in the linear family).
    Closed-form ridge (deterministic, CPU numpy, no torch):
        M_j = argmin ||X_j M - Y_j||^2 + lam ||M||^2   (bias column
        appended to X, unpenalized), Y_j = the TRUE router logits from
    labels_logits (the KL-distillation target EPM-3 will train against).
    Input tap per layer = the B1 train-selected best tap (pass taps=).
    Selection on predicted logits uses the layer's e_score_correction_bias
    (same noaux_tc pipeline as the labels).  Trained on the train split,
    evaluated held-out.
"""

from __future__ import annotations

import numpy as np

from . import dataset, glm_router
from .metrics import EvalAccumulator


# ── shared block-side helpers ────────────────────────────────────────────────

def _block_features_f32(block: dict) -> np.ndarray:
    """[Gb, L, H] float32 raw draft hiddens (exact BF16 expansion)."""
    return glm_router.bf16_bits_to_f32(block["features_bf16"])


def _iter_split(shard_dir, seq_keys):
    return dataset.iterate_blocks(shard_dir, seq_keys=seq_keys)


# ── B0: temporal locality ────────────────────────────────────────────────────

def load_routing_maps(dump_root) -> dict:
    """(run_idx, seq_id, token_pos) -> {layer: (top_ids [K], top_w [K])}
    from every run unit under dump_root.  Run indexing matches
    dataset.assemble_shards (same find_run_units order), so shard blocks'
    (run_idx, seq_id) join cleanly."""
    units = dataset.find_run_units(dump_root)
    if not units:
        raise FileNotFoundError(f"no run units under {dump_root}")
    out: dict = {}
    for run_idx, unit in enumerate(units):
        rpath = unit / "epm_routing.bin"
        if not rpath.is_file():
            continue
        for rr in dataset.read_routing_records(rpath):
            rows = {int(l): (rr.top_ids[i], rr.top_w[i])
                    for i, l in enumerate(rr.layers)}
            out[(run_idx, rr.seq_id, rr.token_pos)] = rows
    return out


def run_b0(shard_dir, dump_root_or_maps, seq_keys, variant: str = "anchor",
           routed_scaling_factor: float = glm_router.
           GLM52_ROUTED_SCALING_FACTOR) -> dict:
    """B0 result dict (metrics.EvalAccumulator.result()).  variant in
    {"prev", "anchor"} — see module docstring for the causality split."""
    if variant not in ("prev", "anchor"):
        raise ValueError(f"unknown B0 variant {variant!r}")
    maps = (dump_root_or_maps if isinstance(dump_root_or_maps, dict)
            else load_routing_maps(dump_root_or_maps))
    index = dataset.load_index(shard_dir)
    moe_layers = np.asarray(index["moe_layers"], np.int32)
    j = len(moe_layers)
    k_top = index["topk"]
    acc = EvalAccumulator(index["max_gamma"], moe_layers, topk=k_top)
    for b in _iter_split(shard_dir, seq_keys):
        gb = b["gamma"]
        pred = np.full((gb, j, k_top), -1, np.int32)
        scores = np.full((gb, j, k_top), np.nan, np.float32)
        for k in range(gb):
            if not b["label_mask"][k]:
                continue
            src = (b["anchor_pos"] - 1 if variant == "anchor"
                   else b["anchor_pos"] + k - 1)
            rows = maps.get((b["run_idx"], b["seq_id"], src))
            if rows is None:
                continue
            for jj, layer in enumerate(moe_layers):
                r = rows.get(int(layer))
                if r is None:
                    continue
                pred[k, jj] = r[0]
                scores[k, jj] = np.clip(
                    r[1] / routed_scaling_factor, 0.0, 1.0)
        acc.add_block(pred, b["labels_top_ids"], b["label_mask"],
                      pred_scores=scores)
    return acc.result()


# ── B1: Tier-0 frozen-router tap sweep ───────────────────────────────────────

def b1_tap_sweep(shard_dir, bank: glm_router.RouterBank, seq_keys,
                 m_rank: int = 32) -> dict:
    """Full n_taps × J sweep on one split.  Returns
    {"per_tap": [result dict per tap], "recall_table": f64 [T, J]
     (n-weighted recall@8 over positions; NaN where n=0),
     "n_table": i64 [T, J]}."""
    index = dataset.load_index(shard_dir)
    moe_layers = np.asarray(index["moe_layers"], np.int32)
    _check_bank(bank, moe_layers, index)
    n_taps = index["n_draft_layers"]
    accs = [EvalAccumulator(index["max_gamma"], moe_layers)
            for _ in range(n_taps)]
    for b in _iter_split(shard_dir, seq_keys):
        feats = _block_features_f32(b)                    # [Gb, T, H]
        for t in range(n_taps):
            logits = bank.logits(feats[:, t])             # [Gb, J, E]
            pred = glm_router.rank_experts(logits, bank.bias, m_rank)
            sc = np.take_along_axis(glm_router.stable_sigmoid(logits),
                                    pred, axis=-1)
            accs[t].add_block(pred, b["labels_top_ids"], b["label_mask"],
                              pred_scores=sc)
    per_tap = [a.result() for a in accs]
    recall_table = np.full((n_taps, len(moe_layers)), np.nan)
    n_table = np.zeros((n_taps, len(moe_layers)), np.int64)
    for t, res in enumerate(per_tap):
        r = res["recall"][8]
        n_j = r["n"].sum(axis=0)
        s_j = np.nansum(np.where(r["n"] > 0, r["mean"] * r["n"], 0.0),
                        axis=0)
        recall_table[t] = np.where(n_j > 0, s_j / np.maximum(n_j, 1),
                                   np.nan)
        n_table[t] = n_j
    return {"per_tap": per_tap, "recall_table": recall_table,
            "n_table": n_table}


def select_best_taps(sweep: dict) -> np.ndarray:
    """argmax tap per layer from a TRAIN-split sweep table (NaN -> tap 0;
    ties -> lower tap via argmax-first-hit)."""
    table = np.where(np.isfinite(sweep["recall_table"]),
                     sweep["recall_table"], -1.0)
    return np.argmax(table, axis=0).astype(np.int32)


def run_b1(shard_dir, bank: glm_router.RouterBank, seq_keys,
           taps: np.ndarray, m_rank: int = 32) -> dict:
    """B1 with a fixed per-layer tap assignment (best-tap or prior)."""
    index = dataset.load_index(shard_dir)
    moe_layers = np.asarray(index["moe_layers"], np.int32)
    _check_bank(bank, moe_layers, index)
    taps = np.asarray(taps, np.int32)
    if len(taps) != len(moe_layers):
        raise ValueError("taps must be per-layer")
    acc = EvalAccumulator(index["max_gamma"], moe_layers)
    for b in _iter_split(shard_dir, seq_keys):
        feats = _block_features_f32(b)                    # [Gb, T, H]
        h = feats[:, taps]                                # [Gb, J, H]
        logits = np.einsum("gjh,jeh->gje", h, bank.weight,
                           optimize=True).astype(np.float32)
        pred = glm_router.rank_experts(logits, bank.bias, m_rank)
        sc = np.take_along_axis(glm_router.stable_sigmoid(logits), pred,
                                axis=-1)
        acc.add_block(pred, b["labels_top_ids"], b["label_mask"],
                      pred_scores=sc)
    return acc.result()


def _check_bank(bank: glm_router.RouterBank, moe_layers: np.ndarray,
                index: dict) -> None:
    if not np.array_equal(bank.moe_layers, moe_layers):
        raise ValueError(
            f"RouterBank layers {bank.moe_layers.tolist()} != shard "
            f"moe_layers {moe_layers.tolist()}")
    if bank.n_experts != index["n_experts"]:
        raise ValueError("RouterBank n_experts != shard n_experts")
    if bank.hidden != index["hidden"]:
        raise ValueError("RouterBank hidden != draft hidden")


# ── B2: per-layer linear-probe skyline ───────────────────────────────────────

def train_b2(shard_dir, seq_keys, taps: np.ndarray,
             ridge_lambda: float = 1e-6) -> dict:
    """Closed-form ridge probe per layer: [x, 1] @ M_j ≈ true router logits.

    Streaming normal equations (no design matrix in memory): per tap t one
    Gram (H+1)², per layer one cross-moment (H+1)×E, f64; Cholesky-free
    np.linalg.solve per tap with all its layers' RHS.  ridge_lambda is
    RELATIVE (scaled by trace(Gram_xx)/H — scale-invariant); the bias
    column is unpenalized.  Deterministic.

    Returns probe {"M": f32 [J, H+1, E], "taps": i32 [J],
    "ridge_lambda": float, "n_train": int}."""
    index = dataset.load_index(shard_dir)
    moe_layers = np.asarray(index["moe_layers"], np.int32)
    j = len(moe_layers)
    h = index["hidden"]
    e = index["n_experts"]
    taps = np.asarray(taps, np.int32)
    if len(taps) != j:
        raise ValueError("taps must be per-layer")
    used_taps = sorted(set(int(t) for t in taps))
    gram = {t: np.zeros((h + 1, h + 1)) for t in used_taps}
    cross = np.zeros((j, h + 1, e))
    n_train = 0
    for b in _iter_split(shard_dir, seq_keys):
        feats = _block_features_f32(b)                    # [Gb, T, H]
        lm = np.asarray(b["label_mask"], bool)
        if not lm.any():
            continue
        rows = np.nonzero(lm)[0]
        n_train += len(rows)
        y = np.asarray(b["labels_logits"], np.float64)    # [Gb, J, E]
        for t in used_taps:
            x = np.concatenate(
                [feats[rows][:, t].astype(np.float64),
                 np.ones((len(rows), 1))], axis=1)        # [n, H+1]
            gram[t] += x.T @ x
            sel = np.nonzero(taps == t)[0]
            cross[sel] += np.einsum("nh,nje->jhe", x, y[rows][:, sel],
                                    optimize=True)
    m_out = np.zeros((j, h + 1, e), np.float32)
    for t in used_taps:
        g = gram[t]
        tr = np.trace(g[:h, :h])
        lam = ridge_lambda * (tr / h if tr > 0 else 1.0)
        reg = np.eye(h + 1) * lam
        reg[h, h] = 0.0                                   # bias unpenalized
        sel = np.nonzero(taps == t)[0]
        rhs = np.concatenate([cross[jj] for jj in sel], axis=1)
        sol = np.linalg.solve(g + reg, rhs)
        for i, jj in enumerate(sel):
            m_out[jj] = sol[:, i * e:(i + 1) * e].astype(np.float32)
    return {"M": m_out, "taps": taps.copy(),
            "ridge_lambda": float(ridge_lambda), "n_train": n_train}


def run_b2(shard_dir, probe: dict, bank: glm_router.RouterBank | None,
           seq_keys, m_rank: int = 32) -> dict:
    """Evaluate the B2 probe.  Selection on predicted logits uses the true
    e_score_correction_bias when a RouterBank is given (same noaux_tc
    pipeline the labels went through); bias=None otherwise."""
    index = dataset.load_index(shard_dir)
    moe_layers = np.asarray(index["moe_layers"], np.int32)
    taps = probe["taps"]
    m_mat = probe["M"]                                    # [J, H+1, E]
    bias = bank.bias if bank is not None else None
    acc = EvalAccumulator(index["max_gamma"], moe_layers)
    for b in _iter_split(shard_dir, seq_keys):
        feats = _block_features_f32(b)                    # [Gb, T, H]
        h = feats[:, taps]                                # [Gb, J, H]
        gb = h.shape[0]
        x = np.concatenate([h, np.ones((gb, len(moe_layers), 1),
                                       np.float32)], axis=-1)
        logits = np.einsum("gjh,jhe->gje", x, m_mat,
                           optimize=True).astype(np.float32)
        pred = glm_router.rank_experts(logits, bias, m_rank)
        sc = np.take_along_axis(glm_router.stable_sigmoid(logits), pred,
                                axis=-1)
        acc.add_block(pred, b["labels_top_ids"], b["label_mask"],
                      pred_scores=sc)
    return acc.result()

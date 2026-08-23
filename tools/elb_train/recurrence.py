"""EPM recurrence relabeling (TD-EPM-DRAFT-FEATURES, Phase 1) — derive the
BELADY-EVICTION training target from the existing EPM-5 corpus WITHOUT
re-collection.

Scientific target (DSP52_BOOST "Oracle Belady-EVICTION" GO / POLICY_LAB
finding #2/#3): per (committed decode position P, MoE layer j, ROUTED expert
e), the binary label

    recur16[P, j, e] = 1  iff  e is routed at layer j at some committed
                              decode position in (P, P + H]   (H = 16 tok)

This is exactly the "recurs <= 16 tok" bridge signal the in-engine
DSP52_EVICT_BRIDGE machinery consumes.  The question this corpus answers:
do the frozen 5-tap DSpark draft hiddens predict recur16 where OCCURRENCE
statistics cannot (finding #3: an online logistic on count/gap features
LOSES to LRU)?

The committed decode trajectory is reconstructed per sequence from the
corpus block tiling (verified: next anchor_pos == anchor_pos +
accepted_len + 1, exact over the corpus).  Block b contributes committed
positions anchor_b + k for k in 0..accepted_len_b (label_mask[k] true);
these tile contiguously and non-overlapping into the sequence timeline.
The MODEL INPUT at a committed position is features_bf16[b, k] (the draft
hidden the corpus already aligns with labels at anchor_pos + k).

Sidecars written beside the shards (idempotent), analogous to
dataset.build_prev_membership_sidecars:

  recur_%05d.npz  per shard, all arrays shape [B, G, J, K]:
    recur16    int8   1/0 recurrence-within-H target; -1 = cell is not a
                      committed+labeled routed slot (pad / speculative
                      tail / uncovered layer) -> excluded from train/eval.
    occ_prev   int8   b0_prev temporal-locality feature: 1 iff e was
                      routed at layer j at the PREVIOUS committed position.
    occ_trail  int8   trailing-window count: # of the last H committed
                      positions (strictly before P) that routed e at j.
    occ_recency int16 committed-position distance to e's most recent PRIOR
                      route at j (1 = previous position; big = long ago;
                      -1 = first-ever / no prior).  The LRU-implicit signal.
    nextdist   int16  committed-position distance to e's NEXT route at j
                      (graded next-use distance for an optional regression
                      head / exact-Belady analog); -1 = never again in seq.

Split honesty (INV-EPM-DATA): recurrence/occurrence are computed per
SEQUENCE from that sequence's own committed timeline only — no cross
-sequence leakage.  Held-out evaluation uses dataset.sequence_split, the
same procedure Phase C used (split_seed 35).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import dataset as ds

RECUR_HORIZON = 16


def _seq_timelines(shard_dir: Path, index: dict):
    """Pass 1: per sequence, the ordered committed-cell timeline.

    Returns seq -> list of committed cells sorted by absolute decode
    position, each a tuple (abs_pos, si, row, k, top_ids[J,K]); plus the
    per-cell coordinate so pass 2 can scatter results back into per-shard
    sidecars.  Only cells with k <= accepted_len AND label_mask[k] are
    committed (the verified prefix that entered the real trajectory)."""
    seqs: dict[int, list] = {}
    for si, sh in enumerate(index["shards"]):
        z = np.load(shard_dir / sh["path"])
        sk = z["seq_key"].astype(np.int64)
        ap = z["anchor_pos"].astype(np.int64)
        al = z["accepted_len"].astype(np.int64)
        lm = z["label_mask"]
        tids = z["labels_top_ids"]      # [B, G, J, K] int32
        b, g = lm.shape
        for r in range(b):
            acc = int(al[r])
            base = int(ap[r])
            for k in range(min(acc + 1, g)):
                if not lm[r, k]:
                    continue
                seqs.setdefault(int(sk[r]), []).append(
                    (base + k, si, r, k, tids[r, k]))
    for s in seqs:
        seqs[s].sort(key=lambda t: t[0])
    return seqs


def build_recurrence_sidecars(shard_dir: str | Path,
                              horizon: int = RECUR_HORIZON) -> dict:
    """Write recur_%05d.npz beside each shard.  Idempotent.  Returns a
    small stats dict (committed cells, positive rate, coverage)."""
    shard_dir = Path(shard_dir)
    index = ds.load_index(shard_dir)
    J = len(index["moe_layers"])
    K = int(index["topk"])
    G = int(index["max_gamma"])
    E = int(index["n_experts"])

    seqs = _seq_timelines(shard_dir, index)

    # Per-shard output arrays, lazily sized to each shard's block count.
    shard_nblocks = [int(sh["num_blocks"]) for sh in index["shards"]]
    out_recur = [np.full((n, G, J, K), -1, np.int8) for n in shard_nblocks]
    out_prev = [np.zeros((n, G, J, K), np.int8) for n in shard_nblocks]
    out_trail = [np.zeros((n, G, J, K), np.int8) for n in shard_nblocks]
    out_recency = [np.full((n, G, J, K), -1, np.int16) for n in shard_nblocks]
    out_next = [np.full((n, G, J, K), -1, np.int16) for n in shard_nblocks]

    n_cells = 0
    n_pos = 0
    for _s, cells in seqs.items():
        T = len(cells)
        if T == 0:
            continue
        # top_ids for the whole timeline: [T, J, K]
        tl = np.stack([c[4] for c in cells], axis=0).astype(np.int64)  # -1 pad
        si_arr = np.array([c[1] for c in cells], np.int64)
        row_arr = np.array([c[2] for c in cells], np.int64)
        k_arr = np.array([c[3] for c in cells], np.int64)
        # Per-sequence result buffers (vectorized fill per layer, one scatter).
        r_recur = np.full((T, J, K), -1, np.int8)
        r_prev = np.zeros((T, J, K), np.int8)
        r_trail = np.zeros((T, J, K), np.int8)
        r_recency = np.full((T, J, K), -1, np.int16)
        r_next = np.full((T, J, K), -1, np.int16)
        t_idx = np.arange(T)
        for j in range(J):
            ids = tl[:, j, :]                       # [T, K]
            slot_valid = ids >= 0                    # [T, K]
            routed = np.zeros((T, E), np.uint8)      # routed[t, e]
            tt, kk = np.nonzero(slot_valid)
            routed[tt, ids[tt, kk]] = 1
            csum = np.zeros((T + 1, E), np.int32)    # prefix sum over positions
            np.cumsum(routed, axis=0, out=csum[1:])
            # prev/next PRIOR occurrence index of each expert (sequential).
            last_idx = np.full(E, -1, np.int64)
            prev_occ = np.empty((T, E), np.int64)
            for t in range(T):
                prev_occ[t] = last_idx
                last_idx[ids[t][slot_valid[t]]] = t
            nxt = np.full(E, T, np.int64)            # T = never
            next_occ = np.empty((T, E), np.int64)
            for t in range(T - 1, -1, -1):
                next_occ[t] = nxt
                nxt[ids[t][slot_valid[t]]] = t
            # Vectorized per-(t,slot) gathers over the K slots at each t.
            e = np.where(slot_valid, ids, 0)         # safe index (masked later)
            hi = np.minimum(t_idx + horizon, T - 1)  # [T]
            lo = np.maximum(t_idx - horizon, 0)       # [T]
            # recurrence within (t, hi]:  csum[hi+1, e] - csum[t+1, e]
            rec = (csum[hi + 1, e.T].T - csum[t_idx + 1, e.T].T) > 0
            trail = csum[t_idx, e.T].T - csum[lo, e.T].T
            po = prev_occ[t_idx, e.T].T               # [T,K]
            no = next_occ[t_idx, e.T].T
            prev1 = np.zeros((T, K), np.uint8)
            prev1[1:] = routed[t_idx[:-1], e[1:].T].T
            r_recur[:, j, :] = np.where(slot_valid, rec.astype(np.int8), -1)
            r_prev[:, j, :] = np.where(slot_valid, prev1, 0).astype(np.int8)
            r_trail[:, j, :] = np.where(
                slot_valid, np.minimum(trail, 127), 0).astype(np.int8)
            rc = np.where(po >= 0, t_idx[:, None] - po, -1)
            r_recency[:, j, :] = np.where(
                slot_valid, np.minimum(rc, 32767), -1).astype(np.int16)
            nd = np.where(no < T, no - t_idx[:, None], -1)
            r_next[:, j, :] = np.where(
                slot_valid, np.minimum(nd, 32767), -1).astype(np.int16)
            n_pos += T if j == 0 else 0
        # One scatter into the per-shard sidecars.
        for t in range(T):
            si, row, k = int(si_arr[t]), int(row_arr[t]), int(k_arr[t])
            out_recur[si][row, k] = r_recur[t]
            out_prev[si][row, k] = r_prev[t]
            out_trail[si][row, k] = r_trail[t]
            out_recency[si][row, k] = r_recency[t]
            out_next[si][row, k] = r_next[t]
        n_cells += T

    n_pad = 0
    n_one = 0
    for si, sh in enumerate(index["shards"]):
        m = out_recur[si] >= 0
        n_pad += int(m.sum())
        n_one += int((out_recur[si] == 1).sum())
        np.savez_compressed(shard_dir / f"recur_{si:05d}.npz",
                            recur16=out_recur[si], occ_prev=out_prev[si],
                            occ_trail=out_trail[si],
                            occ_recency=out_recency[si],
                            nextdist=out_next[si])
    return {
        "shards": len(index["shards"]),
        "sequences": len(seqs),
        "committed_positions": n_pos,
        "labeled_routed_slots": int(n_pad),
        "positive_rate": (n_one / n_pad) if n_pad else 0.0,
        "horizon": horizon,
    }


def load_recur_sidecar(shard_dir: str | Path, si: int) -> dict | None:
    p = Path(shard_dir) / f"recur_{si:05d}.npz"
    if not p.is_file():
        return None
    return dict(np.load(p))

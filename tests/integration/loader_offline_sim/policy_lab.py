#!/usr/bin/env python3
"""Policy lab: max out the SIM-predicted tok/s (min predicted fetch-ms/tok, max
hit rate) on the committed keeper trajectories — CPU only, no GPU.

Decouples the two decisions the loader makes per routed expert:
  EVICTION  (which resident to displace when a cache is full)
  PLACEMENT (which GPU a miss is fetched to; hits always ride their residency)

and evaluates policy combinations against the recorded trajectories, including
clairvoyant (Belady) upper bounds. Primary target: the CANONICAL trace
(deterministic_ep_combine ON ⇒ routing is policy-invariant, so offline results
transfer to the engine modulo state fidelity — INVESTIGATION.md 2026-07-18).

Metrics per run: hit rate, fetches/token, predicted fetch-ms/tok via the
calibration transfer model (trajectory_sim.predict_us — the sim's tok/s proxy).

Usage:
  python3 policy_lab.py --calib ../../assets/gpu_loader_calibration_4gpu_hbm.json \
      [--trace assets/traj_4gpu_canonical.jsonl.gz] [--mode zoo|bounds|train|all]
"""
import argparse
import heapq
import json
import math
import random

from trajectory_sim import load_traj, predict_us, compute_us

CAPS = [203, 203, 443, 443]
INF = 10**18


# ───────────────────────────── trace preparation ─────────────────────────────

def flatten(rows):
    """Per-record: (layer, [(expert_idx, bank), ...]). Key = (layer, expert)."""
    out = []
    for r in rows:
        out.append((r["layer_idx"], [(e["expert_idx"], e["bank"]) for e in r["experts"]]))
    return out


def next_use_tables(recs):
    """For Belady: for each access (record i, key) the index of the NEXT record
    referencing the same key (INF if never again)."""
    occ = {}
    for i, (layer, exps) in enumerate(recs):
        for e, _ in exps:
            occ.setdefault((layer, e), []).append(i)
    nxt = {}
    for key, idxs in occ.items():
        for p, i in enumerate(idxs):
            nxt[(key, i)] = idxs[p + 1] if p + 1 < len(idxs) else INF
    return nxt


# ───────────────────────────── cache substrate ───────────────────────────────

class GpuCache:
    """One GPU's resident set with pluggable victim scoring. Victim = min score;
    score is supplied at touch time (higher = keep longer)."""

    __slots__ = ("cap", "score", "heap", "n")

    def __init__(self, cap):
        self.cap = cap
        self.score = {}          # key -> current score (authoritative)
        self.heap = []           # (score, key) lazy min-heap
        self.n = 0

    def __contains__(self, key):
        return key in self.score

    def touch(self, key, score):
        """Insert or rescore. Returns evicted key or None."""
        if key not in self.score:
            self.n += 1
        self.score[key] = score
        heapq.heappush(self.heap, (score, key))
        if self.n <= self.cap:
            return None
        while True:
            s, k = heapq.heappop(self.heap)
            if k in self.score and self.score[k] == s:
                del self.score[k]
                self.n -= 1
                return k

    def victim_score(self):
        """Current minimum score (the would-be victim), or None if free slots."""
        if self.n < self.cap:
            return None
        while self.heap:
            s, k = self.heap[0]
            if k in self.score and self.score[k] == s:
                return s
            heapq.heappop(self.heap)
        return None


# ───────────────────────────── eviction scorers ──────────────────────────────
# A scorer maps access context -> keep-score (higher = protected longer).
# Context: t (global access counter), ridx (record index), key, state (dict for
# online statistics), nxt (Belady table or None).

def sc_lru(t, ridx, key, state, nxt):
    return t


def sc_belady(t, ridx, key, state, nxt):
    return nxt.get((key, ridx), INF)  # keep-until = next use (farther = evicted first? no:)
    # NOTE: Belady evicts the FARTHEST next use → victim = min score ⇒ score must
    # be NEGATIVE next-use. Handled by the caller wrapper below.


def make_sc_belady():
    def f(t, ridx, key, state, nxt):
        nu = nxt.get((key, ridx), INF)
        return -nu  # farthest next use = smallest score = evicted first
    return f


def make_sc_window(w):
    """Sliding-window LRU: recency, but entries older than w accesses are all
    equally stale (floor) — approximates TTL semantics on the score axis."""
    def f(t, ridx, key, state, nxt):
        return t  # window is enforced by the placement side (victim_age cap)
    return f


def make_sc_ewma(alpha=0.3, default_gap=150.0, overdue_k=1.0):
    """Reuse-distance EWMA (online Belady approximation): victim = farthest
    PREDICTED next use. keep = -effective_next_use; an OVERDUE prediction
    (predicted next use already in the past) must decay toward eviction —
    without this the stale-prediction entries become immortal (the inversion
    that scored 0.24 in the first zoo run)."""
    def f(t, ridx, key, state, nxt):
        gaps = state.setdefault("gap", {})
        last = state.setdefault("last", {})
        if key in last:
            g = ridx - last[key]
            gaps[key] = (1 - alpha) * gaps.get(key, g) + alpha * g
        last[key] = ridx
        pred_next = ridx + gaps.get(key, default_gap)
        return -pred_next
    return f


def make_sc_ewma_live(alpha=0.3, default_gap=150.0):
    """EWMA with LIVE overdue decay: the keep-score is re-evaluated lazily by
    storing predicted-next-use directly; overdue entries fall behind fresh ones
    naturally because their predicted_next is smaller than the current frontier.
    (Same as make_sc_ewma — kept separate for A/B clarity.)"""
    return make_sc_ewma(alpha, default_gap)


def make_sc_freq(decay=0.02, w=200.0):
    """Recency + decayed-frequency hybrid (GDSF-lite): keep = ridx + w*freq_decayed.

    ENGINE UNITS (P-26 sim-model fix, 2026-07-19): the clock is ridx — the
    per-record LAYER-VISIT clock, exactly the EvictScoreBoard's recency
    (advance_recency once per layer; every touch in the visit shares the
    value) — NOT the per-touch counter t. With t the same w bought ~topk (≈8)
    times less protection than the engine's, so the sim only ever saw the
    engine freq curve's UPSLOPE and ranked w=120 above the keeper-measured
    peak w=60 (keeper-confirmed inversion: sim #1 fw=120 → keeper 0.7083 vs
    defaults 0.7327 — assets/policy_params_deployfit_fw120_rejected.json).
    Historical t-domain zoo rows (POLICY_LAB_RESULTS.md freq arms) predate
    this fix and are not reproducible with these units."""
    def f(t, ridx, key, state, nxt):
        fr = state.setdefault("freq", {})
        fr[key] = fr.get(key, 0.0) * math.exp(-decay) + 1.0
        return ridx + w * fr[key]
    return f


def make_sc_lruk(k=2):
    """LRU-K (k=2): keep-score = the K-th most recent reference time (classic
    O'Neil LRU-2). Scan-like keys (single reference) carry score = their only
    (old) touch minus a demotion, so they lose to twice-referenced residents.
    Monotone timeline scores — safe for the lazy heap (no overdue decay needed,
    unlike the EWMA predicted-next-use form, which is NOT lazily heapable)."""
    def f(t, ridx, key, state, nxt):
        hist = state.setdefault("hist", {})
        h = hist.setdefault(key, [])
        h.append(t)
        if len(h) > k:
            del h[0]
        return h[0] if len(h) == k else h[0] - 10**7  # <k refs: demoted epoch
    return f


def make_sc_hotset(gap_thresh=200.0, bonus=2000.0, alpha=0.3):
    """Binary hot-set protection: experts whose EWMA token-gap is short get a
    large fixed recency bonus; scan-like (long-gap) experts compete as plain LRU."""
    def f(t, ridx, key, state, nxt):
        gaps = state.setdefault("gap", {})
        last = state.setdefault("last", {})
        if key in last:
            g = ridx - last[key]
            gaps[key] = (1 - alpha) * gaps.get(key, g) + alpha * g
        last[key] = ridx
        return t + (bonus if gaps.get(key, 1e9) <= gap_thresh else 0.0)
    return f


# ───────────────────────────── placement policies ────────────────────────────
# placement(miss_list, caches, ctx) -> gpu per miss. Hits are already routed.

def place_affinity(miss, caches, ctx):
    """Per-layer load balance, tie-break oldest victim (the keeper router)."""
    out = []
    load = [0] * 4
    for i in miss:
        best, bk = 0, None
        for g in range(4):
            vs = caches[g].victim_score()
            age = -INF if vs is None else vs  # free slot = oldest possible
            k = (load[g], age)
            if bk is None or k < bk:
                best, bk = g, k
        out.append(best)
        load[best] += 1
    return out


def make_place_reuse(w=2000.0, tau=2400.0, calib=None):
    """The landed engine policy: transfer makespan + w/(1+age/tau) victim cost.

    ENGINE UNITS (P-26 sim-model fix, 2026-07-19): age = ridx − victim_score,
    LAYER-VISITS — the engine's `recency_now() − cheapest_score` with τ in
    layer-visit units (τ=300 ≈ 4 tokens). Pair ONLY with a ridx-domain scorer
    (make_sc_freq above; sc_lru/lruk/hotset are t-domain — mixing domains
    makes every age negative → clamps to max reuse cost everywhere)."""
    def f(miss, caches, ctx):
        t = ctx["ridx"]
        sub = [0.0] * 4
        cnt = list(ctx["hit_cnt"])
        pd = []
        for g in range(4):
            vs = caches[g].victim_score()
            if vs is None:
                pd.append(0.0)
            else:
                age = max(0.0, t - vs) if vs > -1e17 else INF  # lru-score domain
                pd.append(w / (1.0 + age / tau) if age < 1e17 else 0.0)
        out = []
        for i, bank in miss:
            best, bt = 0, None
            for g in range(4):
                cell = ctx["calib"]["matrix"][bank][g]
                sx = cell["rate_us"] + cell["lat_us"]
                m = max(sub[g] + sx + compute_us(ctx["calib"]["devices"][g].get("compute", {}), cnt[g] + 1), 0)
                v = m + pd[g]
                if bt is None or v < bt:
                    best, bt = g, v
            out.append(best)
            sub[best] += ctx["calib"]["matrix"][bank][best]["rate_us"] + ctx["calib"]["matrix"][bank][best]["lat_us"]
            cnt[best] += 1
        return out
    return f


def place_belady(miss, caches, ctx):
    """Clairvoyant placement: put each miss on the GPU whose victim's next use
    is FARTHEST (max headroom); free slots first."""
    out = []
    for i in miss:
        best, bk = 0, None
        for g in range(4):
            vs = caches[g].victim_score()
            k = -INF if vs is None else vs  # belady scores are -next_use
            if bk is None or k < bk:
                best, bk = g, k
        out.append(best)
    return out


def make_place_linear(theta, calib):
    """Trainable linear scorer over causal features; miss goes to argmin score.
    Features per (miss, gpu): [subxfer_norm, victim_age_norm, layer_load,
    free_slot, cap_norm, hitcnt_norm]."""
    def f(miss, caches, ctx):
        t = ctx["t"]
        out = []
        load = [0] * 4
        for i, bank in miss:
            best, bt = 0, None
            for g in range(4):
                cell = calib["matrix"][bank][g]
                sx = (cell["rate_us"] + cell["lat_us"]) / 700.0
                vs = caches[g].victim_score()
                free = 1.0 if vs is None else 0.0
                age = 0.0 if vs is None else max(0.0, min((t - vs) / 5000.0, 4.0))
                feats = (sx, age, load[g] / 4.0, free, CAPS[g] / 443.0,
                         ctx["hit_cnt"][g] / 8.0, 1.0)
                v = sum(a * b for a, b in zip(theta, feats))
                if bt is None or v < bt:
                    best, bt = g, v
            out.append(best)
            load[best] += 1
        return out
    return f


# ───────────────────────────── replay engine ─────────────────────────────────

def replay(recs, calib, scorer, placer, nxt=None, caps=CAPS):
    caches = [GpuCache(c) for c in caps]
    state = {}
    t = 0
    hits = lookups = fetches = 0
    pred_total = 0.0
    for ridx, (layer, exps) in enumerate(recs):
        # 1) split hits/misses vs CURRENT state
        hit_gpu = {}
        miss = []
        hit_cnt = [0] * 4
        for e, bank in exps:
            key = (layer, e)
            for g in range(4):
                if key in caches[g]:
                    hit_gpu[(e, bank)] = g
                    hit_cnt[g] += 1
                    break
            else:
                miss.append((e, bank))
        ctx = {"t": t, "calib": calib, "hit_cnt": hit_cnt, "ridx": ridx}
        placement = placer(miss, caches, ctx) if miss else []
        # 2) account + mutate
        placements, banks, misses_mask = [], [], []
        for e, bank in exps:
            lookups += 1
            t += 1
            key = (layer, e)
            if (e, bank) in hit_gpu:
                g = hit_gpu[(e, bank)]
                hits += 1
                caches[g].touch(key, scorer(t, ridx, key, state, nxt))
                placements.append(g); banks.append(bank); misses_mask.append(False)
        for (e, bank), g in zip(miss, placement):
            key = (layer, e)
            fetches += 1
            caches[g].touch(key, scorer(t, ridx, key, state, nxt))
            placements.append(g); banks.append(bank); misses_mask.append(True)
        pred_total += predict_us(calib, placements, banks, misses_mask, 4)
    ntok = len(recs) / 75.0
    return {"hit": hits / max(lookups, 1),
            "fetch_tok": fetches / ntok,
            "pred_ms": pred_total / 1000.0 / ntok}


def pooled_belady(recs, nxt, cap=sum(CAPS)):
    """Placement-free ceiling: ONE pooled cache, clairvoyant eviction."""
    c = GpuCache(cap)
    hits = lookups = 0
    for ridx, (layer, exps) in enumerate(recs):
        for e, _ in exps:
            key = (layer, e)
            lookups += 1
            if key in c:
                hits += 1
            c.touch(key, -nxt.get((key, ridx), INF))
    return hits / lookups


# ───────────────────────────── ES training ───────────────────────────────────

def train_linear(recs, calib, scorer, gens=24, pop=16, sigma=0.35, seed=7):
    rng = random.Random(seed)
    dim = 7
    theta = [1.0, -1.0, 0.5, -1.5, 0.0, 0.0, 0.0]  # sane init: cheap link, old victim, spread, free slots

    def fitness(th):
        r = replay(recs, calib, scorer, make_place_linear(th, calib))
        return -(r["pred_ms"] + (1.0 - r["hit"]) * 120.0)  # pred + hit shaping

    best, best_f = theta[:], fitness(theta)
    for gen in range(gens):
        cands = []
        for _ in range(pop):
            th = [b + rng.gauss(0, sigma) for b in best]
            cands.append((fitness(th), th))
        cands.sort(reverse=True)
        if cands[0][0] > best_f:
            best_f, best = cands[0][0], cands[0][1]
        sigma *= 0.93
        print(f"    gen {gen:02d} best_f={best_f:.2f} theta={[round(x,2) for x in best]}", flush=True)
    return best


# ───────────────────────────── main ──────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="assets/traj_4gpu_canonical.jsonl.gz")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--mode", default="all", choices=["bounds", "zoo", "train", "all"])
    ap.add_argument("--eval-traces", nargs="*", default=[])
    args = ap.parse_args()
    calib = json.load(open(args.calib))
    recs = flatten(load_traj(args.trace))
    nxt = next_use_tables(recs)
    print(f"trace {args.trace}: {len(recs)} records")

    def show(name, r):
        print(f"  {name:34s} hit={r['hit']:.4f} fetch/tok={r['fetch_tok']:.1f} pred={r['pred_ms']:.1f} ms/tok")

    if args.mode in ("bounds", "all"):
        print("── clairvoyant bounds ──")
        print(f"  pooled Belady ({sum(CAPS)} slots)       hit={pooled_belady(recs, nxt):.4f}")
        show("partitioned Belady + belady-place", replay(recs, calib, make_sc_belady(), place_belady, nxt))
        show("partitioned Belady + affinity-place", replay(recs, calib, make_sc_belady(), place_affinity, nxt))
    if args.mode in ("zoo", "all"):
        print("── online policies ──")
        show("LRU + affinity-place (champion)", replay(recs, calib, sc_lru, place_affinity))
        # ridx-domain LRU (make_sc_freq w=0 == the engine board at FREQ_W=0):
        # make_place_reuse ages are layer-visit units now — sc_lru (t-domain)
        # would make every age negative (see the placer's units note).
        show("LRU + reuse-place (engine design)", replay(recs, calib, make_sc_freq(w=0.0), make_place_reuse(calib=calib)))
        for a in (0.2, 0.5):
            show(f"EWMA(a={a}) + affinity-place", replay(recs, calib, make_sc_ewma(a), place_affinity))
        for d, w in ((0.02, 200.0), (0.05, 400.0), (0.01, 100.0)):
            show(f"freq(d={d},w={w}) + affinity-place", replay(recs, calib, make_sc_freq(d, w), place_affinity))
        for gt, b in ((150, 2000), (300, 2000), (150, 8000)):
            show(f"hotset(gt={gt},b={b}) + affinity", replay(recs, calib, make_sc_hotset(gt, b), place_affinity))
    if args.mode in ("train", "all"):
        print("── ES-trained linear placement (LRU eviction) ──")
        theta = train_linear(recs, calib, sc_lru)
        show("trained-linear + LRU", replay(recs, calib, sc_lru, make_place_linear(theta, calib)))
        print(f"  theta = {[round(x, 3) for x in theta]}")
        for t2 in args.eval_traces:
            r2 = flatten(load_traj(t2))
            show(f"trained-linear on {t2.split('/')[-1]}", replay(r2, calib, sc_lru, make_place_linear(theta, calib)))


if __name__ == "__main__":
    main()

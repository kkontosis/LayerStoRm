#!/usr/bin/env python3
"""Trajectory simulator: replay LS_LOADER_SHADOW_DUMP trajectories through a
clean per-GPU LRU under interchangeable PLACEMENT policies, and predict the
per-layer times via the loader cost model (calibration JSON).

Decomposes the ACT-vs-affinity gap the same way INVESTIGATION.md Q3 did for the
2-GPU era, but on the live 4-GPU keeper dumps and INCLUDING the affinity router
itself as a simulated policy:

  * ROUTING-TRACE effect: the two arms' routed expert sequences decorrelate
    (placement changes the EP-combine FP order -> near-tie gating flips ->
    autoregressive cascade), so each trajectory is replayed under EVERY policy —
    comparing a policy across trajectories isolates the trace; comparing
    policies on one trajectory isolates placement.
  * PLACEMENT effect: policies below.
  * EVICTION effect: sim 'executed' vs the engine's real hit rate (the dump's
    cached mask) — a clean-LRU-vs-engine gap is an engine eviction artifact.

Policies:
  executed  : what the run actually did (affinity dump: oj; ACT dump: oj if the
              SIM says hit at oj else the solver's j — mirrors the reroute rule)
  etp       : static e%tp (expert_idx % M)
  affinity  : the keeper52 router re-simulated on SIM cache state — hit -> the
              resident GPU; miss -> argmin per-layer assigned-miss count,
              tie-break by the globally OLDEST evictable victim (pooled LRU)
  solver    : the dump's j[.] verbatim (the solver's choice, hits included)

Predicted time per record (transfer model, serial banks):
  T = max( max_j (sum_{misses on j} (rate+lat)[bank][j] + compute_j(count)),
           max_b sum_{misses from b} egress[b] ) + recon + fixed_overhead
computed from the CALIBRATION JSON and the SIM's cache state (not the dump's
subxfer fields, which reflect the engine's own cache history).

Usage:
  python3 trajectory_sim.py --dump <model.jsonl> [--dump <model2.jsonl> ...] \
      --calib <gpu_loader_calibration.json> [--caps 203,203,443,443]
"""
import argparse
import gzip
import json
from collections import OrderedDict


def load_traj(path):
    rows = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            rows.append(d)
    return rows


class Caches:
    """Per-GPU LRU over (layer, expert) with a global recency tick, so the
    affinity policy can compare victim ages ACROSS GPUs (pooled-LRU emulation —
    mirrors keeper52_test.cpp GpuLru + g_lru_tick)."""

    def __init__(self, caps):
        self.caps = caps
        self.lru = [OrderedDict() for _ in caps]  # key -> global tick
        self.tick = 0

    def hit_gpu(self, key):
        for g, c in enumerate(self.lru):
            if key in c:
                return g
        return -1

    def touch(self, g, key):
        self.tick += 1
        c = self.lru[g]
        if key in c:
            c.move_to_end(key)
        c[key] = self.tick
        evicted = None
        if len(c) > self.caps[g]:
            evicted = c.popitem(last=False)[0]
        return evicted

    def oldest_tick(self, g):
        c = self.lru[g]
        if len(c) < self.caps[g]:
            return -1  # free slot = infinitely old victim (prefer filling)
        return next(iter(c.values()))


def compute_us(cc, c):
    if c <= 0:
        return 0.0
    p = cc.get("P", 1) or 1
    batches = c if p == 1 else 1 + (c - 1) // p
    return cc.get("a_us", 0.0) * c + cc.get("b_us", 0.0) * batches


# Optional M2/M2v2 EXPOSED-WALL params (tools/loader_xray/exposed_model.py):
# v1 {s,o0,oc,cpw,b0} or v2 {form:"v2", s,o0,oc,hsat,cpw,g_c,b0}. When set,
# predict_us returns the exposed critical-path-weighted wall instead of the
# raw transfer window (total R2 0.2->0.74 on the fitted config; cpw/b0 are
# CONFIG-SCOPED, refit per deployment). v2 is the B>1-forward form: batch-step
# compute term kept, SATURATING overlap credit (DETHRONE_AFFINITY_IDEAS reqs).
EXPOSED = None


def set_exposed(params):
    global EXPOSED
    EXPOSED = params


def predict_us(calib, placements, banks, misses, M):
    """One record: placements[i]=gpu, banks[i]=bank slot, misses[i]=bool."""
    sub = [0.0] * M
    cnt = [0] * M
    nmiss = [0] * M
    egress = {}
    for i, g in enumerate(placements):
        cnt[g] += 1
        if misses[i]:
            nmiss[g] += 1
            b = banks[i]
            cell = calib["matrix"][b][g]
            sub[g] += cell["rate_us"] + cell["lat_us"]
            egress[b] = egress.get(b, 0.0) + calib["banks"][b]["egress_us"]
    if EXPOSED is not None:
        import math
        v2 = EXPOSED.get("form") == "v2"
        best = 0.0
        for g in range(M):
            hits_g = cnt[g] - nmiss[g]
            if v2:
                hs = max(EXPOSED["hsat"][g], 1e-6)
                credit = EXPOSED["o0"][g] + EXPOSED["oc"][g] * hs * (
                    1.0 - math.exp(-hits_g / hs))
                dc = calib["devices"][g]["compute"]
                comp = EXPOSED["g_c"] * (
                    dc["a_us"] * cnt[g]
                    + dc["b_us"] * math.ceil(cnt[g] / dc["P"]) * (cnt[g] > 0))
            else:
                credit = EXPOSED["o0"][g] + EXPOSED["oc"][g] * hits_g
                comp = 0.0
            exp_g = max(0.0, EXPOSED["s"][g] * sub[g] - credit)
            best = max(best, EXPOSED["cpw"][g] * (comp + exp_g))
        return best + EXPOSED["b0"]
    makespan, ov, ad = 0.0, 0.0, 0.0
    for g in range(M):
        if cnt[g] == 0:
            continue
        makespan = max(makespan, sub[g] + compute_us(calib["devices"][g].get("compute", {}), cnt[g]))
        ov = max(ov, calib["devices"][g].get("recon_overhead_us", 0.0))
        ad += calib["devices"][g].get("recon_added_us", 0.0)
    bank_floor = max(egress.values(), default=0.0)
    return max(makespan, bank_floor) + ov + ad + calib.get("fixed_overhead_us", 0.0)


def reuse_place_cost(caches, g, w, tau, shape):
    """Per-device victim-protection cost for placing ONE MISS on g: 0 while g has
    free slots, else w * g(age of g's oldest (cheapest) victim). Young victim =
    valuable = expensive to displace; old victim = safe. This is the cross-token
    reuse reward developed here and mirrored into the engine's place term."""
    c = caches.lru[g]
    if len(c) < caches.caps[g]:
        return 0.0
    age = caches.tick - next(iter(c.values()))
    if shape == "exp":
        import math
        return w * math.exp(-age / tau)
    return w / (1.0 + age / tau)  # "inv"


def place_misses_reuse(caches, calib, exps, hit_gpus, M, w, tau, shape, tp_pen=0.0):
    """Engine-faithful miss placement: hits pinned to their resident GPU; misses
    minimize  max_j(sub_j + compute_j(cnt_j)) + sum_i place_dev[j_i]  (the bank
    floor is placement-invariant under serial banks). Exact enumeration for <=3
    misses, greedy marginal-cost placement beyond (engine uses exact B&B; greedy
    is a close stand-in at decode miss counts)."""
    sub = [0.0] * M
    cnt = [0] * M
    place_dev = [reuse_place_cost(caches, g, w, tau, shape) for g in range(M)]
    if tp_pen > 0.0:  # attention/TP ranks (devices 0,1 on the keeper) cost extra to disturb
        place_dev[0] += tp_pen
        place_dev[1] += tp_pen
    miss_idx = []
    out = [0] * len(exps)
    for i, e in enumerate(exps):
        if hit_gpus[i] >= 0:
            out[i] = hit_gpus[i]
            cnt[hit_gpus[i]] += 1
        else:
            miss_idx.append(i)

    def sx(i, g):
        cell = calib["matrix"][exps[i]["bank"]][g]
        return cell["rate_us"] + cell["lat_us"]

    def makespan(sub_, cnt_):
        m = 0.0
        for g in range(M):
            if cnt_[g]:
                m = max(m, sub_[g] + compute_us(calib["devices"][g].get("compute", {}), cnt_[g]))
        return m

    if len(miss_idx) <= 3:  # exact
        best, best_asn = None, None
        n = len(miss_idx)
        for code in range(M ** n):
            s2, c2, pl = sub[:], cnt[:], 0.0
            x = code
            asn = []
            for i in miss_idx:
                g = x % M
                x //= M
                s2[g] += sx(i, g)
                c2[g] += 1
                pl += place_dev[g]
                asn.append(g)
            t = makespan(s2, c2) + pl
            if best is None or t < best - 1e-12:
                best, best_asn = t, asn
        for i, g in zip(miss_idx, best_asn):
            out[i] = g
    else:  # greedy marginal
        for i in miss_idx:
            best_g, best_t = 0, None
            for g in range(M):
                sub[g] += sx(i, g)
                cnt[g] += 1
                t = makespan(sub, cnt) + place_dev[g]
                sub[g] -= sx(i, g)
                cnt[g] -= 1
                if best_t is None or t < best_t - 1e-12:
                    best_g, best_t = g, t
            out[i] = best_g
            sub[best_g] += sx(i, best_g)
            cnt[best_g] += 1
    return out


def replay(rows, caps, calib, policy, mode="act"):
    caches = Caches(caps)
    M = len(caps)
    lookups = hits = 0
    fetches = 0
    dup_fetches = 0          # miss while resident on ANOTHER gpu
    evict_reuse_window = {}  # key -> record index of eviction
    evicted_reused = 0
    pred_total = 0.0
    seen_home = {}           # key -> set of gpus it was ever fetched to
    reuse_cfg = None
    if policy.startswith("reuse"):
        parts = policy.split(":")
        _, shape, w, tau = parts[:4]
        tp_pen = float(parts[4]) if len(parts) > 4 else 0.0
        reuse_cfg = (shape, float(w), float(tau), tp_pen)
    for ridx, r in enumerate(rows):
        exps = r["experts"]
        n = len(exps)
        # 1) decide placement per expert
        place, miss = [0] * n, [False] * n
        layer_miss_cnt = [0] * M
        if reuse_cfg is not None:
            shape, w, tau, tp_pen = reuse_cfg
            hit_gpus = [caches.hit_gpu((r["layer_idx"], e["expert_idx"])) for e in exps]
            place = place_misses_reuse(caches, calib, exps, hit_gpus, M, w, tau, shape, tp_pen)
            for i, e in enumerate(exps):
                miss[i] = (r["layer_idx"], e["expert_idx"]) not in caches.lru[place[i]]
        else:
          for i, e in enumerate(exps):
            key = (r["layer_idx"], e["expert_idx"])
            hg = caches.hit_gpu(key)
            if policy == "solver":
                g = e["j"]
            elif policy == "etp":
                g = e["expert_idx"] % M
            elif policy == "executed":
                # ACT dump: the engine runs at the ORCHESTRATOR target when the
                # expert is resident there, else at the solver's j (verified:
                # cached[oj] OR cached[j] == the engine's reported hit rate).
                # Shadow-only (affinity) dump: oj IS the executed decision.
                oj = e.get("oj", -1)
                if oj < 0:
                    oj = e["expert_idx"] % M
                if mode == "act":
                    g = oj if key in caches.lru[oj] else e["j"]
                else:
                    g = oj
            elif policy == "affinity":
                if hg >= 0:
                    g = hg
                else:
                    best, best_load, best_age = 0, None, None
                    for cand in range(M):
                        load = layer_miss_cnt[cand]
                        age = caches.oldest_tick(cand)
                        k2 = (load, age)
                        if best_load is None or k2 < (best_load, best_age):
                            best, best_load, best_age = cand, load, age
                    g = best
            else:
                raise ValueError(policy)
            place[i] = g
            m = key not in caches.lru[g]
            miss[i] = m
            if m:
                layer_miss_cnt[g] += 1
        # 2) account + mutate cache
        for i, e in enumerate(exps):
            key = (r["layer_idx"], e["expert_idx"])
            lookups += 1
            if not miss[i]:
                hits += 1
                caches.touch(place[i], key)
            else:
                fetches += 1
                if caches.hit_gpu(key) >= 0:
                    dup_fetches += 1
                if key in evict_reuse_window and ridx - evict_reuse_window[key] <= 75 * 4:
                    evicted_reused += 1
                seen_home.setdefault(key, set()).add(place[i])
                ev = caches.touch(place[i], key)
                if ev is not None:
                    evict_reuse_window[ev] = ridx
        pred_total += predict_us(calib, place, [e["bank"] for e in exps], miss, M)
    churn = sum(1 for s in seen_home.values() if len(s) > 1)
    return {
        "hit_rate": hits / max(lookups, 1),
        "fetch_per_tok": fetches / (len(rows) / 75.0),
        "dup_fetch_frac": dup_fetches / max(fetches, 1),
        "evicted_reused_4tok": evicted_reused,
        "multi_home_experts": churn,
        "pred_ms_per_tok": pred_total / 1000.0 / (len(rows) / 75.0),
    }


def engine_hit_rate(rows):
    """The REAL engine's hit rate as recorded in the dump's cached mask at the
    executed device (oj) — the reference the clean-LRU sim is compared against."""
    hits = lookups = 0
    for r in rows:
        for e in r["experts"]:
            lookups += 1
            oj = e.get("oj", -1)
            hit = 0 <= oj < len(e["cached"]) and e["cached"][oj]
            # under ACT a miss was rerouted to j; hit accounting is vs oj target
            hits += 1 if hit else 0
    return hits / max(lookups, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True,
                    help="<path>:<act|aff> — the run mode the dump came from")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--caps", default="203,203,443,443")
    ap.add_argument("--exposed", default=None,
                    help="M2 exposed-wall params JSON (exposed_model.py --out)")
    ap.add_argument("--sweep", action="store_true",
                    help="grid-sweep the reuse place-cost (shape x w x tau)")
    ap.add_argument("--policy", action="append",
                    help="extra policies, e.g. reuse:inv:600:2400")
    args = ap.parse_args()
    caps = [int(x) for x in args.caps.split(",")]
    with open(args.calib) as fh:
        calib = json.load(fh)
    if args.exposed:
        with open(args.exposed) as fh:
            set_exposed(json.load(fh))
    policies = ["executed", "etp", "affinity", "solver"] + (args.policy or [])
    if args.sweep:
        policies = ["affinity", "solver"]
        for shape in ("inv", "exp"):
            for w in (50, 200, 600, 2000, 6000):
                for tau in (600, 2400, 9600):
                    policies.append(f"reuse:{shape}:{w}:{tau}")
    for spec in args.dump:
        dump, _, mode = spec.rpartition(":")
        if mode not in ("act", "aff"):
            dump, mode = spec, "act"
        rows = load_traj(dump)
        print(f"\n=== trajectory {dump} [{mode}] ({len(rows)} records) ===")
        print(f"  engine-recorded hit rate (cached[oj]): {engine_hit_rate(rows):.4f}")
        for policy in policies:
            s = replay(rows, caps, calib, policy, mode)
            print(f"  {policy:20s}: hit={s['hit_rate']:.4f} fetch/tok={s['fetch_per_tok']:.1f} "
                  f"dup={s['dup_fetch_frac']:.3f} evict-reuse<=4tok={s['evicted_reused_4tok']} "
                  f"multi-home={s['multi_home_experts']} pred={s['pred_ms_per_tok']:.1f} ms/tok")


if __name__ == "__main__":
    main()

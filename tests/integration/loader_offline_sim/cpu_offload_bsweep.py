#!/usr/bin/env python3
"""CPU-expert offload B-SWEEP explorer (no-boot strategic proxy).

━━━ RESIDENCY-HONEST UPDATE (2026-08-10, --mode honest is now the default) ━━━
The original sweep (--mode legacy) predicted +3.71% at B=16 but NEVER PRICED the
GPU-residency DISRUPTION: it inserted OFFLOADED experts into the GPU cache (so their
future uses were free hits) AND assumed the host FFN fully hides (overlap→1). Both
were false (MILESTONE_C measured −12.4%). The honest model fixes both:
  * OFFLOAD EXCLUSION: an offloaded expert is host-computed and EXCLUDED from GPU;
    the cache is advanced TOKEN-BY-TOKEN so its next non-offloaded use (median reuse
    = 1 token) MISSES — the forfeited caching the +3.7% sweep skipped.
  * MEASURED OVERLAP (--eta-cap): the host FFN hides only ~0.15-0.3 at B=16 with M3
    residency (small GPU window), not ~1.0. The champion DECIDED with the optimistic
    belief and PAID the exposed cost — reproduced here (naive).
VERDICT (champion regime pin=800 caps=64 eta-cap=0.0): naive (champion) = −11% at
B=16 (≈ measured −12.4%); the residency-priced never-lose solver (priced) and the
residency-safe policy (resaware) DECLINE → +0.00%. NO win survives once residency is
honestly priced — reuse is too short (96% of uses recur within 16 tokens) for a
residency-safe offload population to exist. CPU-offload is residency-capped. NO-GO.

━━━ SHORT-HORIZON CORRECTION (2026-08-10, task_ab_shorthorizon.py) ━━━
The NO-GO above used the 16-tok RECURRENCE rate as the residency window — WRONG.
The residency window is the cache-RETENTION horizon: how long an offload candidate
would ACTUALLY survive on GPU. Measured on the champion cache (pin+LRU): 75 MoE
layers SHARE each e%4 home ⇒ ~150 insertions/tok/home, so the COLD (non-pinned,
offloadable) tail has hit 0.107 and P(hit) 0.30@gap1, 0.08@gap2, ~0@gap>=3 — an
effective retention window of ~1 TOKEN (champion hits come from PINNING, not LRU).
⇒ Run resaware/priced with --resid-window 1 (not 16). The residency-SAFE population
(next-use > 1 tok) is ~30% of uses (excl trace-end), not the 3.7% the 16-tok framing
gave. Result at B=16: resaware/priced turn naive's −11.09% into +0.00% (eta 0.0) /
+0.07% (eta 0.15) / +0.45% (eta 0.30) — i.e. the short-horizon reframe REMOVES the
−11% residency loss (break-even to marginal-positive) and a FREE prev-union signal
captures 100% of the oracle ceiling. But it does NOT open a clear WIN at the measured
overlap: the binding constraint FLIPS from residency to the host↔GPU OVERLAP (eta) —
+0.4..+2.9% needs eta 0.3..1.0. Net: residency-capped → OVERLAP-capped.

━━━ original TASK-3 header ━━━

Extends the loader offline sim with a CPU-expert DEVICE modelled by a CALIBRATED
cost model AND its BATCH-SIZE scaling, then sweeps the effective batch B to find
the CROSSOVER where the never-lose I8 solver starts OFFLOADING (and winning).

WHY B MATTERS (coordinator directive, 2026-08-09): the plain keeper52 harness is
B=1 decode (~5.89 tok/s). The real CHAMPION is dsp52 SPECULATIVE, whose VERIFY
CHUNK processes 1+gamma = 16 tokens at once = a de-facto BATCH OF 16. At B=16 the
GPU compute WINDOW is ~16x bigger, the per-engaged-layer fold overhead + the H2D
fetch AMORTIZE over 16 tokens, and the host FFN runs an M=B GEMM (more efficient
per token). So B=16 is the HEADLINE regime — the one the plain-B=1 harness
structurally cannot show.

MECHANISM (the model, all terms from real measurements):
  Per engaged MoE layer, batch B, with U routed-union experts and m misses:
    all-GPU wall  = max( per_gpu_fetch(m), per_gpu_compute(U,B) )          [4 GPUs]
    offload k cold misses to CPU (never-lose: k=0 always available):
        gpu_wall' = max( per_gpu_fetch(m-k)*contention, per_gpu_compute(U-k,B) )
        cpu_wall  = FOLD + k*host_expert(B)          [host FFN, runs || the GPU]
        layer_wall(k) = max(gpu_wall', cpu_wall)     [CPU overlaps the GPU window]
      pick k* = argmin_k layer_wall(k).  k*=0 <=> no beneficial offload (neutral).
  The WIN appears once B grows the GPU window enough to HIDE the host FFN
  (cpu_wall <= gpu_wall') while the k removed fetches relieve the fetch floor.

CALIBRATION (real numbers):
  * expert_bytes 24.77 MB  -> HOST_WEIGHTREAD 387 us (24.77MB / 64 GB/s DDR),
    B-INVARIANT (read the expert's weights once, apply to all B chunk tokens).
  * H2D fetch 960 us/xfer (engine DMA+detect), B-INVARIANT (weights once/chunk).
  * fold 440 us / engaged-layer, B-INVARIANT (D2H input + H2D output + barriers).
  * host per-expert coordination/staging ~520 us, B-INVARIANT (cross-thread).
  * host + GPU per-token compute slopes (small; decode is weight-read bound).
  Anchors: B=1 all-GPU reproduces the champion (169.8 ms/tok, 5.89 tok/s, hit
  0.426); B=1 forced-6-offload reproduces the measured concT8 (~ -9% net).

Usage:  python3 cpu_offload_bsweep.py [--trace assets/traj_4gpu_canonical.jsonl.gz]
"""
import argparse
import math
import sys
from collections import OrderedDict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from trajectory_sim import load_traj  # noqa: E402

# ── calibrated cost model (us) ───────────────────────────────────────────────
FETCH_US            = 960.0   # H2D per expert xfer (engine DMA+detect); B-invariant
FETCH_CHANNELS      = 4       # 4 GPUs fetch in parallel (DDR contention folded in)
GPU_COMPUTE_PER_TOK = 8.0     # per-expert GPU FFN, per token (decode is fetch-bound; small)
HOST_WEIGHTREAD_US  = 387.0   # 24.77MB / 64 GB/s; B-invariant (weights read once)
HOST_COORD_US       = 520.0   # per-expert host staging/coordination; B-invariant
HOST_COMPUTE_PER_TOK= 14.0    # host FFN per token (14-core node-3); small
FOLD_US             = 440.0   # per-engaged-layer fold overhead; B-invariant
DMA_CONTENTION      = 1.057   # surviving-fetch DDR-staging step when offloading (960->1015)
FIXED_TOK_US        = 84000.0 # per-token non-MoE wall (attention/etc.), calibrated to 5.89 tok/s
CAPS                = [305, 305, 305, 305]  # per-GPU (layer,expert) slots
NLAYERS             = 75

# OVERLAP EFFICIENCY vs B: the host FFN hides under the GPU window only to the
# extent the window EXISTS. At B=1 the fold's D2H drains the GPU first (poor
# overlap — the measured concT8 REGRESSED); as B grows the ~B x bigger GPU
# window (and the graph-hoist mechanism, Task A) lets the host FFN hide. Modeled
# as the fraction of cpu_wall that is OVERLAPPABLE (the rest is exposed serial).
# eta(1)=0.15 (near-serial, matches the B=1 offload-loses reality / Task-2 zero),
# saturating to ~1 by B~16. This is the single most B-sensitive knob (sweep it).
# ETA_CAP caps the fraction of the host FFN that hides under the GPU window. The
# original curve saturated to ~1 by B~16 (the ASSUMPTION behind the +3.7% sweep) —
# MILESTONE_C MEASURED this false: with M3 residency the per-engaged-layer GPU
# window is small, so overlap ON vs serial gained only ~2% (host essentially
# EXPOSED). Set ETA_CAP≈0.15-0.3 to model the measured poor overlap.
ETA_CAP = 1.0


def overlap_eta(B):
    return min(ETA_CAP, 1.0 - 0.85 * math.exp(-(B - 1) / 4.0))


def host_expert(B):
    """Per-expert host FFN wall at batch B: weight-read + coord (B-invariant) +
    compute (∝ B). More efficient PER TOKEN as B grows (fixed cost amortizes)."""
    return HOST_WEIGHTREAD_US + HOST_COORD_US + HOST_COMPUTE_PER_TOK * B


def per_gpu_fetch(m, contention=1.0):
    return math.ceil(m / FETCH_CHANNELS) * FETCH_US * contention


def per_gpu_compute(nexp, B):
    return math.ceil(nexp / FETCH_CHANNELS) * GPU_COMPUTE_PER_TOK * B


def layer_wall_allgpu(U, m, B):
    return max(per_gpu_fetch(m), per_gpu_compute(U, B))


def layer_wall_neverlose(U, m, B, eta=None):
    """Return (best_wall, k*) minimizing the layer wall over k offloaded misses,
    with a B-dependent overlap: the CPU host FFN hides under eta*gpu_wall', the rest
    is EXPOSED (serial). k=0 (all-GPU) is always in the search -> the never-lose
    guarantee. Pass eta= the OPTIMISTIC (uncapped) curve to model the champion's
    DECISION belief; the ACCOUNTING wall uses the real (capped) overlap_eta."""
    base = layer_wall_allgpu(U, m, B)
    if eta is None:
        eta = overlap_eta(B)
    best, best_k = base, 0
    for k in range(1, m + 1):
        gpu2 = max(per_gpu_fetch(m - k, DMA_CONTENTION), per_gpu_compute(U - k, B))
        cpu2 = FOLD_US + k * host_expert(B)
        exposed = max(0.0, cpu2 - eta * gpu2)   # unhidden host wall (serial)
        w = gpu2 + exposed
        if w < best - 1e-9:
            best, best_k = w, k
    return best, best_k


def optimistic_eta(B):
    """The uncapped overlap the +3.7% sweep BELIEVED (saturates to ~1 by B~16) —
    the champion solver's decision model, before MILESTONE_C measured it false."""
    return 1.0 - 0.85 * math.exp(-(B - 1) / 4.0)


# ── batched-union residency (per-GPU LRU over the e%tp home) ──────────────────
class Lru:
    def __init__(self, cap):
        self.cap = cap
        self.od = OrderedDict()

    def hit(self, key):
        if key in self.od:
            self.od.move_to_end(key)
            return True
        return False

    def insert(self, key):
        self.od[key] = 1
        self.od.move_to_end(key)
        while len(self.od) > self.cap:
            self.od.popitem(last=False)

    def evict(self, key):
        """Remove key if resident (used to model CPU-offload residency EXCLUSION:
        an offloaded expert is computed on the host and is NOT placed on any GPU,
        so it must be dropped from the GPU cache — the residency-disruption cost)."""
        self.od.pop(key, None)


# ── residency-disruption cost model (the term the +3.7% sweep NEVER priced) ───
# WHEN an expert is offloaded to the CPU it is computed on the host and EXCLUDED
# from every GPU bitset ⇒ it is not fetched/cached. EACH of its FUTURE GPU-routed
# uses (on a chunk that does NOT also offload it) therefore becomes an ADDED MISS.
# In the champion (no offload) a routed MISS is fetched+CACHED, so its next use is
# a HIT; offloading FORFEITS that caching. The forfeit costs a future miss only if
# the expert would still have been RESIDENT at its next use — i.e. next-use ≤ the
# residency window. Reuse distance (Belady next-use) is the honest driver.
#   RESID_COEFF  : tunable/trainable weight on the priced future-miss term (the
#                  solver's ESTIMATE of the disruption; the accounting below is
#                  ground truth via cache exclusion).
#   RESID_WINDOW : residency horizon in TOKENS. next-use ≤ window ⇒ offloading
#                  forfeits a would-be hit (priced); next-use > window ⇒ the expert
#                  would have been LRU-evicted anyway ⇒ offloading is ~free.
RESID_COEFF  = 1.0
# RESID_WINDOW is the cache-RETENTION horizon (how long an offload candidate would
# survive on GPU), NOT the recurrence rate. Measured cold-tail retention on the
# champion pin+LRU cache ≈ 1 TOKEN (75 layers share each e%4 home ⇒ sub-token LRU
# life; hits come from pinning). Use --resid-window 1 for the champion regime; the
# legacy 16 over-restricts the residency-safe set to ~4% and masks the reframe.
# See task_ab_shorthorizon.py + the SHORT-HORIZON CORRECTION header.
RESID_WINDOW = 16     # tokens; DEFAULT kept at 16 for back-compat — pass 1 (measured)


def build_nextuse(by_layer, ntok):
    """occ[L][e] = sorted token positions routing expert e in layer L. Used to
    answer 'next use of e in layer L strictly after token te' (Belady, tokens)."""
    import bisect
    occ = [dict() for _ in range(NLAYERS)]
    for L in range(NLAYERS):
        for t in range(ntok):
            for e in by_layer[L][t]:
                occ[L].setdefault(e, []).append(t)

    def nextuse_after(L, e, te):
        lst = occ[L].get(e)
        if not lst:
            return 10**9
        i = bisect.bisect_right(lst, te)
        return (lst[i] - te) if i < len(lst) else 10**9
    return nextuse_after


def build_freq(by_layer, ntok):
    """freq[(L,e)] = occurrence count (the champion's 'cold-by-freq' naive signal —
    aggregate frequency, NO reuse-distance awareness)."""
    freq = {}
    for L in range(NLAYERS):
        for t in range(ntok):
            for e in by_layer[L][t]:
                freq[(L, e)] = freq.get((L, e), 0) + 1
    return freq


def warm_caches(caches, by_layer, ntok, passes, M=4):
    """M3-STATIC STEADY-STATE proxy: prime the per-GPU LRU with `passes` all-GPU
    replays of the trace so the HOT union is already resident when we measure. This
    reproduces the champion's regime — a verify chunk's union is mostly HITS, so
    only a FEW cold-tail experts miss ⇒ a SMALL GPU fetch window. In that regime the
    host FFN (weight-read+coord, ~1.1 ms/expert) EXCEEDS the window and is EXPOSED —
    which is exactly why the measured champion offload regressed (MILESTONE_C)."""
    for _ in range(passes):
        for t in range(ntok):
            for layer in range(NLAYERS):
                for e in by_layer[layer][t]:
                    caches[e % M].insert((layer, e))


def build_pinned(freq, npin):
    """M3-STATIC hot set: the top-`npin` (layer,expert) by frequency are PINNED
    resident in GPU HBM (the champion's static placement). Pinned experts are
    always HITS and are never fetched/offloaded — they model the M3-HBM residency
    whose disruption MILESTONE_C blames for the regression. With a hot set pinned,
    a verify-chunk union is MOSTLY hits ⇒ only a few COLD-TAIL misses ⇒ a SMALL GPU
    fetch window that the ~1.1 ms/expert host FFN EXCEEDS (exposed ⇒ offload loses)."""
    if npin <= 0:
        return set()
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:npin]
    return {k for k, _ in top}


def run_B_honest(by_layer, ntok, B, policy, nextuse, freq, M=4,
                 resid_window=None, resid_coeff=None, warm=0, pinned=None):
    """Residency-HONEST offload sim. policy ∈ {allgpu, naive, resaware, priced}.

      allgpu   : no offload (champion baseline; honest cache).
      naive    : offload the wall-optimal COUNT k* of misses REUSE-BLIND (the
                 champion's real never-lose present-wall solver — it minimizes the
                 instantaneous fetch/compute wall and never looks at reuse, so it
                 picks reuse-representative experts, median next-use = 1 token).
                 Honest accounting ⇒ the residency-disruption regression appears.
      naivefreq: offload the k* COLDEST-by-freq misses (aggregate-frequency prior,
                 the 3-task place_cons signal — a WEAK reuse proxy).
      resaware : offload ONLY experts whose next-use > residency window (truly cold
                 ⇒ excluding them costs ~0 future misses).
      priced   : never-lose k* search with the residency future-miss term ADDED to
                 the layer wall (RESID_COEFF × amortized fetch × recurrences-in-win).

    RESIDENCY MODEL (the fix). The chunk union sets the FETCH floor and the offload
    ECONOMICS (host FFN amortized over B), but the GPU cache is advanced TOKEN BY
    TOKEN so the real short reuse-distance (median 1 tok) — the residency the
    champion's hit 0.368 lives in — is preserved. An OFFLOADED expert is computed on
    the host and EXCLUDED from GPU for the whole chunk: its per-token lookups are
    host-served (evicted, not cached), so its NEXT non-offloaded GPU use (typically
    ~1 token later) MISSES. That forfeited caching is the residency-disruption cost
    the +3.7% sweep never priced — it shows up as EXTRA MISSES on all future steps."""
    if resid_window is None:
        resid_window = RESID_WINDOW
    if resid_coeff is None:
        resid_coeff = RESID_COEFF
    if pinned is None:
        pinned = set()
    caches = [Lru(CAPS[g]) for g in range(M)]
    if warm:
        warm_caches(caches, by_layer, ntok, warm, M)
    nchunks = math.ceil(ntok / B)
    tot_all = tot_nl = 0.0
    tot_off = tot_U = tot_m = 0
    gpu_lookups = gpu_hits = 0        # token-level GPU accounting (excludes offloaded)
    engaged_layers = 0
    fetch_amort = FETCH_US / FETCH_CHANNELS   # per-fetch amortized cost (per chunk)
    for c in range(nchunks):
        t0 = c * B
        t1 = min(t0 + B, ntok)
        bsz = t1 - t0
        te = t1 - 1                              # last token of the chunk
        for layer in range(NLAYERS):
            uni = {}
            for t in range(t0, t1):
                for e in by_layer[layer][t]:
                    uni[e] = 1
            U = len(uni)
            # residency at chunk start (drives the fetch floor + offload target set).
            # Pinned (M3-HBM hot) experts are ALWAYS resident and never miss/offload.
            resident = {e for e in uni
                        if (layer, e) in pinned or (layer, e) in caches[e % M].od}
            miss = [e for e in uni if e not in resident]
            m = len(miss)
            # ── decide the offload set K ⊆ union (misses are the cold candidates)
            # The champion DECIDED with the optimistic overlap belief (kstar_opt);
            # the honest never-lose count (kstar_hon) uses the real capped overlap.
            _, kstar_opt = layer_wall_neverlose(U, m, bsz, eta=optimistic_eta(bsz))
            base_wall, kstar_hon = layer_wall_neverlose(U, m, bsz)
            K = set()
            if policy == "allgpu":
                pass
            elif policy == "naive":
                # reuse-BLIND champion: offload the OPTIMISTICALLY-beneficial count,
                # then get ACCOUNTED with the real exposed host FFN below.
                order = sorted(miss)
                K = set(order[:kstar_opt])
            elif policy == "naivefreq":
                order = sorted(miss, key=lambda e: freq.get((layer, e), 0))
                K = set(order[:kstar_opt])
            elif policy == "resaware":
                # HONEST never-lose count, RESTRICTED to residency-safe experts
                # (next-use beyond the window ⇒ excluding them costs ~0 misses).
                elig = [e for e in miss if nextuse(layer, e, te) > resid_window]
                elig.sort(key=lambda e: freq.get((layer, e), 0))
                K = set(elig[:kstar_hon])
            elif policy == "priced":
                eta = overlap_eta(bsz)
                best_w, best_K = layer_wall_allgpu(U, m, bsz), set()
                order = sorted(miss, key=lambda e: freq.get((layer, e), 0))
                for k in range(1, m + 1):
                    cand = order[:k]
                    gpu2 = max(per_gpu_fetch(m - k, DMA_CONTENTION),
                               per_gpu_compute(U - k, bsz))
                    cpu2 = FOLD_US + k * host_expert(bsz)
                    exposed = max(0.0, cpu2 - eta * gpu2)
                    w = gpu2 + exposed
                    pen = 0.0                      # residency future-miss penalty
                    for e in cand:
                        if nextuse(layer, e, te) <= resid_window:
                            pen += resid_coeff * fetch_amort
                    w += pen
                    if w < best_w - 1e-9:
                        best_w, best_K = w, set(cand)
                K = best_K
            else:
                raise ValueError(policy)
            k = len(K)
            # ── layer wall (honest fetch floor: only non-offloaded misses fetch) ─
            m_gpu = m - k                          # K ⊆ miss ⇒ removed from fetch
            eta = overlap_eta(bsz)
            gpu2 = max(per_gpu_fetch(m_gpu, DMA_CONTENTION if k else 1.0),
                       per_gpu_compute(U - k, bsz))
            if k:
                cpu2 = FOLD_US + k * host_expert(bsz)
                exposed = max(0.0, cpu2 - eta * gpu2)
                wall = gpu2 + exposed
            else:
                wall = layer_wall_allgpu(U, m, bsz)
            tot_all += layer_wall_allgpu(U, m, bsz)
            tot_nl += wall
            tot_off += k
            tot_U += U
            tot_m += m
            if k > 0:
                engaged_layers += 1
            # ── HONEST token-level residency advance ─────────────────────────
            for t in range(t0, t1):
                for e in by_layer[layer][t]:
                    key = (layer, e)
                    g = e % M
                    if key in pinned:              # M3-HBM: always resident hit
                        gpu_lookups += 1
                        gpu_hits += 1
                        continue
                    if e in K:                     # offloaded → host-served, excluded
                        caches[g].evict(key)
                        continue
                    gpu_lookups += 1
                    if caches[g].hit(key):
                        gpu_hits += 1
                    else:
                        caches[g].insert(key)
    fixed = FIXED_TOK_US * ntok
    tps_all = ntok / ((tot_all + fixed) / 1e6)
    tps_nl = ntok / ((tot_nl + fixed) / 1e6)
    return {
        "B": B, "policy": policy, "chunks": nchunks,
        "avg_U": tot_U / (nchunks * NLAYERS),
        "avg_m": tot_m / (nchunks * NLAYERS),
        "hit": gpu_hits / max(gpu_lookups, 1),
        "misses": gpu_lookups - gpu_hits,
        "off_total": tot_off,
        "off_per_engaged_layer": tot_off / max(engaged_layers, 1),
        "engaged_frac": engaged_layers / (nchunks * NLAYERS),
        "tps_all": tps_all, "tps_nl": tps_nl,
        "delta_pct": 100.0 * (tps_nl - tps_all) / tps_all,
    }


def run_B(recs_by_layer, ntok, B, M=4):
    """recs_by_layer[layer] = list over tokens of the 8 routed expert_idx. Chunk
    the ntok tokens into groups of B; each chunk-layer routes the UNION of its B
    tokens' experts (the speculative verify-chunk shape). Returns metrics."""
    caches = [Lru(CAPS[g]) for g in range(M)]
    nchunks = math.ceil(ntok / B)
    tot_all = 0.0
    tot_nl = 0.0
    tot_off = 0
    tot_U = 0
    tot_m = 0
    tot_lookup = 0
    tot_hit = 0
    engaged_layers = 0
    for c in range(nchunks):
        t0 = c * B
        t1 = min(t0 + B, ntok)
        bsz = t1 - t0
        for layer in range(NLAYERS):
            # union of the chunk's routed experts for this layer
            uni = {}
            for t in range(t0, t1):
                for e in recs_by_layer[layer][t]:
                    uni[e] = 1
            U = len(uni)
            # residency at e%tp home
            m = 0
            for e in uni:
                g = e % M
                key = (layer, e)
                if caches[g].hit(key):
                    tot_hit += 1
                else:
                    m += 1
            tot_lookup += U
            for e in uni:                       # after solve: all fetched/computed
                caches[e % M].insert((layer, e))
            wall_all = layer_wall_allgpu(U, m, bsz)
            wall_nl, k = layer_wall_neverlose(U, m, bsz)
            tot_all += wall_all
            tot_nl += wall_nl
            tot_off += k
            tot_U += U
            tot_m += m
            if k > 0:
                engaged_layers += 1
    # per-CHUNK token wall (a chunk delivers bsz tokens); tok/s = tokens / walltime
    fixed = FIXED_TOK_US * ntok          # non-MoE wall scales with tokens
    walltime_all = tot_all + fixed
    walltime_nl = tot_nl + fixed
    toks = ntok
    tps_all = toks / (walltime_all / 1e6)
    tps_nl = toks / (walltime_nl / 1e6)
    return {
        "B": B,
        "chunks": nchunks,
        "avg_U": tot_U / (nchunks * NLAYERS),
        "avg_m": tot_m / (nchunks * NLAYERS),
        "hit": tot_hit / max(tot_lookup, 1),
        "off_total": tot_off,
        "off_per_engaged_layer": tot_off / max(engaged_layers, 1),
        "engaged_layers": engaged_layers,
        "engaged_frac": engaged_layers / (nchunks * NLAYERS),
        "host_expert_us": host_expert(B),
        "host_us_per_tok": host_expert(B) / B,
        "fold_frac_of_chunk": FOLD_US / max(tot_nl / (nchunks * NLAYERS), 1e-9),
        "tps_all": tps_all,
        "tps_nl": tps_nl,
        "delta_pct": 100.0 * (tps_nl - tps_all) / tps_all,
    }


def main():
    global HOST_COORD_US, FETCH_US
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="assets/traj_4gpu_canonical.jsonl.gz")
    ap.add_argument("--bs", default="1,2,4,8,16,32")
    ap.add_argument("--mode", default="honest",
                    choices=["honest", "legacy", "both"],
                    help="honest = residency-priced offload (default); legacy = "
                         "the old insert-all sweep that priced NO residency")
    ap.add_argument("--resid-window", type=int, default=RESID_WINDOW)
    ap.add_argument("--resid-coeff", type=float, default=RESID_COEFF)
    ap.add_argument("--warm", type=int, default=0,
                    help="M3-static steady-state: prime the LRU with N all-GPU "
                         "passes before measuring (union mostly resident)")
    ap.add_argument("--caps", default=None,
                    help="override per-GPU cache slots, e.g. 96,96,96,96 (smaller "
                         "= higher pressure = lower hit, to reach the 0.37 regime)")
    ap.add_argument("--pin", type=int, default=0,
                    help="M3-static: pin the top-N (layer,expert) by freq resident "
                         "in HBM (union mostly hits ⇒ small cold-tail fetch window)")
    ap.add_argument("--eta-cap", type=float, default=1.0,
                    help="cap on host-FFN overlap-hide fraction (MEASURED ~0.15-0.3 "
                         "at B=16 with M3 residency; 1.0 = the old optimistic sweep)")
    args = ap.parse_args()
    global ETA_CAP
    ETA_CAP = args.eta_cap
    if args.caps:
        global CAPS
        CAPS = [int(x) for x in args.caps.split(",")]
    rows = load_traj(args.trace)
    # The trace is token-major: NLAYERS consecutive MoE-layer records per token.
    ntok = len(rows) // NLAYERS
    by_layer = [[[] for _ in range(ntok)] for _ in range(NLAYERS)]
    for i, r in enumerate(rows):
        t = i // NLAYERS
        L = i % NLAYERS
        if t < ntok:
            by_layer[L][t] = [e["expert_idx"] for e in r["experts"]]
    print(f"trace {args.trace}: {len(rows)} records, {ntok} tokens, {NLAYERS} layers")
    print(f"cost model: FETCH={FETCH_US}us fold={FOLD_US}us host_expert(B)="
          f"{HOST_WEIGHTREAD_US}+{HOST_COORD_US}+{HOST_COMPUTE_PER_TOK}*B us "
          f"contention={DMA_CONTENTION}")
    bs = [int(x) for x in args.bs.split(",")]

    if args.mode in ("honest", "both"):
        nextuse = build_nextuse(by_layer, ntok)
        freq = build_freq(by_layer, ntok)
        pinned = build_pinned(freq, args.pin)
        print(f"\n=== RESIDENCY-HONEST offload (window={args.resid_window} tok, "
              f"coeff={args.resid_coeff}, pin={args.pin}, warm={args.warm}) ===")
        print("Offloaded experts are EXCLUDED from GPU cache ⇒ future GPU-routed "
              "uses become misses.")
        print("policies: naive=reuse-BLIND present-wall (champion) | "
              "naivefreq=cold-by-freq | resaware=next-use>window | priced=+resid-term")
        hh = ("  B   pol        avg_U  avg_m   hit    misses  off/eng  eng-frac  "
              "tps_all  tps_nl   delta%")
        print(hh)
        print("  " + "-" * (len(hh) - 2))
        for B in bs:
            base = run_B_honest(by_layer, ntok, B, "allgpu", nextuse, freq,
                                resid_window=args.resid_window,
                                resid_coeff=args.resid_coeff, warm=args.warm,
                                pinned=pinned)
            for pol in ("naive", "naivefreq", "resaware", "priced"):
                s = run_B_honest(by_layer, ntok, B, pol, nextuse, freq,
                                 resid_window=args.resid_window,
                                 resid_coeff=args.resid_coeff, warm=args.warm,
                                 pinned=pinned)
                # delta vs the HONEST all-GPU champion baseline (same residency)
                d = 100.0 * (s["tps_nl"] - base["tps_all"]) / base["tps_all"]
                hitd = s["hit"] - base["hit"]
                print(f"  {B:<3d} {pol:<9s}  {s['avg_U']:5.1f}  {s['avg_m']:5.2f}  "
                      f"{s['hit']:.3f}  {s['misses']:6d}  "
                      f"{s['off_per_engaged_layer']:5.2f}   {s['engaged_frac']:.3f}   "
                      f"{base['tps_all']:6.2f}  {s['tps_nl']:6.2f}  {d:+6.2f}"
                      f"  (Δhit {hitd:+.3f})")
            print()
        if args.mode == "honest":
            return

    hdr = ("  B  chunks  avg_U  avg_m   hit   host/tok  off/eng-lyr  eng-frac   "
           "tps_all  tps_nl   delta%")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    bs = [int(x) for x in args.bs.split(",")]
    results = []
    for B in bs:
        s = run_B(by_layer, ntok, B)
        results.append(s)
        print(f"  {s['B']:<3d}{s['chunks']:>6d}  {s['avg_U']:5.1f}  {s['avg_m']:5.2f}  "
              f"{s['hit']:.3f}  {s['host_us_per_tok']:7.0f}   {s['off_per_engaged_layer']:6.2f}    "
              f"{s['engaged_frac']:.3f}   {s['tps_all']:6.2f}  {s['tps_nl']:6.2f}  {s['delta_pct']:+6.2f}")
    print()
    # crossover
    cross = next((s for s in results if s["delta_pct"] > 0.05), None)
    if cross:
        print(f"CROSSOVER: never-lose offload goes NET-POSITIVE at B={cross['B']} "
              f"(+{cross['delta_pct']:.2f}%, {cross['off_per_engaged_layer']:.2f} "
              f"experts/engaged-layer, {cross['engaged_frac']*100:.0f}% of layers engage)")
    else:
        print("CROSSOVER: none in the swept range — never-lose stays k*=0 (neutral).")
    hl = next((s for s in results if s["B"] == 16), None)
    if hl:
        print(f"\nHEADLINE (B=16, dsp52 speculative verify chunk):")
        print(f"  host FFN per-expert = {hl['host_expert_us']:.0f} us for 16 tokens "
              f"= {hl['host_us_per_tok']:.0f} us/token (vs {host_expert(1):.0f} us/token at B=1)")
        print(f"  fold overhead = {FOLD_US:.0f} us = {hl['fold_frac_of_chunk']*100:.1f}% of the "
              f"per-engaged-layer chunk wall (amortized over 16 tokens)")
        print(f"  never-lose offloads {hl['off_per_engaged_layer']:.2f} experts/engaged-layer "
              f"on {hl['engaged_frac']*100:.0f}% of layers")
        print(f"  predicted tok/s: all-GPU {hl['tps_all']:.2f} -> never-lose {hl['tps_nl']:.2f} "
              f"({hl['delta_pct']:+.2f}%)")

    # ── sensitivity of the B=16 headline to the two most-uncertain knobs ──
    base_coord, base_fetch = HOST_COORD_US, FETCH_US
    print("\nSENSITIVITY (B=16 delta%): host-coord x {0.7,1.0,1.4} X fetch x {0.75,1.0}")
    for fm in (0.75, 1.0):
        row = []
        for cm in (0.7, 1.0, 1.4):
            HOST_COORD_US = base_coord * cm
            FETCH_US = base_fetch * fm
            s = run_B(by_layer, ntok, 16)
            row.append(f"{s['delta_pct']:+5.2f}")
        print(f"  fetch x{fm:<4}: coord[.7/1/1.4] = {'  '.join(row)}")
    HOST_COORD_US, FETCH_US = base_coord, base_fetch


if __name__ == "__main__":
    main()

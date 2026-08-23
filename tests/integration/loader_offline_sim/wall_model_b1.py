#!/usr/bin/env python3
"""
Symbolic wall model for the GLM-5.2 494GB champion (B=1 DSpark speculative decode).

Calibrated against measured anchors (spec/cpu_offload_tuning.md §8d/§8e,
spec/reports/DSP52_BOOST.md M6 §3, xray_dspark_r1.log round table, FETCH_XRAY):

  A1  champion wall 11150.5 ms / 107 committed tok = 9.596 tok/s
  A2  plain autoregressive arm 138.5 ms/tok med (7.12-7.22 tok/s)
  A3  chunk fetch: 14976 transfers x 26.34 MiB = 385.3 GiB, sum span 2590.8 ms
      (AGG 148.7 GiB/s), ~fully exposed (M3 span-to-wall conversion ~1:1)
  A4  decode fetch: 100% hidden (daemon exposed 0.01 ms/tok)
  A5  verify wall fit V(g) from 8 measured chunk rounds
  A6  offline residency sim: pooled LRU hit {1052:0.279, 1600:0.381, 2500:0.471},
      Belady {1052:0.510}; compulsory share 0.24
  A7  in-engine Belady +3.0% / bridge16 +4.5%
  A8  contention physics: link 47 GiB/s, DDR node cap 64 GiB/s, GPU:CPU 2.5:1,
      CPU cold kernel 450us DDR / 284us HBM-local, fold 440us/engaged-layer

Round data (xray_dspark_r1.log, FORCE_DRAFT champion, bit-deterministic):
  (g_use, accepted, committed_delta, plain_ms, verify_ms)
"""
import numpy as np

ROUNDS = [  # g_use, accepted, plain_ms, verify_ms
    (2,0,215.075,0),(2,0,209.982,0),(1,0,171.604,0),(2,0,168.759,0),
    (2,0,169.927,0),(2,0,173.963,0),(2,0,165.581,0),(2,0,162.596,0),
    (3,0,168.661,0),(2,0,159.519,0),(2,0,150.133,0),(2,0,161.759,0),
    (2,0,143.008,0),(2,0,137.832,0),(2,0,145.269,0),(2,0,139.945,0),
    (2,0,150.158,0),(2,0,152.920,0),(2,0,152.293,0),(2,0,154.765,0),
    (3,0,148.900,0),(2,0,134.523,0),(2,0,161.785,0),(2,0,150.019,0),
    (2,0,131.838,0),(2,0,148.278,0),(2,0,127.382,0),
    (2,2,125.762,282.697),
    (4,0,176.746,0),(3,0,137.180,0),(3,0,129.961,0),
    (4,4,127.164,377.145),
    (8,8,171.586,567.843),
    (10,10,178.348,675.420),
    (8,8,172.680,638.082),
    (7,0,147.167,0),
    (12,12,163.865,762.338),
    (10,10,168.979,708.962),
    (13,13,171.650,799.697),
]
WALL_MS = 11150.520
COMMITTED = 107
EXPERT_MIB = 26.34
CHUNK_BYTES_GIB = 385.28          # total chunk-regime H2D per run
CHUNK_SPAN_MS = 2590.8            # sum of chunk per-cmd fetch spans
CHUNK_AGG_GIBS = 148.7            # bandwidth-bound aggregate during chunk fetch
DECODE_BYTES_GIB = 272.29         # hidden
CHUNK_FFN_MS = 140.27             # kComputeGpu per chunk sweep (avg, Sec 8e)
CHUNK_ATTN_MS = 70.0              # post-SMALLM chunk attention (M6)
PLAIN_FFN_MS = 29.7               # dspark decode FFN compute per token (Sec 8e)

def base_accounting():
    tot_plain = sum(r[2] for r in ROUNDS)
    tot_verify = sum(r[3] for r in ROUNDS)
    chunk_rounds = [r for r in ROUNDS if r[3] > 0]
    rej = [r for r in ROUNDS if r[3] == 0]
    print(f"rounds={len(ROUNDS)} plain_sum={tot_plain:.0f} verify_sum={tot_verify:.0f} "
          f"sum={tot_plain+tot_verify:.0f} vs wall {WALL_MS:.0f} "
          f"(residual {WALL_MS-tot_plain-tot_verify:.0f} ms)")
    g = np.array([r[0] for r in chunk_rounds]); v = np.array([r[3] for r in chunk_rounds])
    A = np.vstack([np.ones_like(g), g]).T
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    pred = A @ coef
    print(f"V(g) fit: V = {coef[0]:.1f} + {coef[1]:.1f}*g  (resid rms "
          f"{np.sqrt(np.mean((v-pred)**2)):.1f} ms)")
    print(f"reject rounds: n={len(rej)} plain avg {np.mean([r[2] for r in rej]):.1f} ms")
    print(f"chunk rounds: n={len(chunk_rounds)} plain avg "
          f"{np.mean([r[2] for r in chunk_rounds]):.1f} ms, "
          f"tokens from chunks={sum(r[1] for r in chunk_rounds)} (+1 bonus each)")
    return coef

V0, VS = None, None

# ---- offline LRU hit vs cache capacity (pooled decode+chunk, A6) ----
CAP_HIT = [(1052,0.279),(1600,0.381),(2500,0.471)]
def lru_hit(cap):
    caps = np.array([c for c,_ in CAP_HIT]); hits = np.array([h for _,h in CAP_HIT])
    return float(np.interp(cap, caps, hits))
ENGINE_HIT = 0.3675
ENGINE_MISSES = 27175.0

def miss_scale_from_capacity(cap_factor):
    """Relative miss volume when cache slot count scales by cap_factor.
    Uses the offline LRU curve shape, applied at the engine's operating point
    (engine policy ~= LRU+freq; assume equal RELATIVE gains). Compulsory floor 0.24."""
    h0 = lru_hit(1052); h1 = lru_hit(1052*cap_factor)
    rel = (1-h1)/(1-h0)
    # never below compulsory share of current misses
    return max(rel, 0.24)

# ---------------------------------------------------------------- levers
def wall_with(kappa_stream=1.0, resident_lowbit=False, extend_gamma=None,
              p_cont=0.0, offload_k=0, offload_mem='hbm', skip_draft=False,
              belady=None, verbose=False):
    """Recompute the run wall under lever settings.
    kappa_stream: byte ratio of the streamed/miss copy (1.0 = today)
    resident_lowbit: residents also low-bit -> cache slots x(1/kappa), FFN reads scale
    extend_gamma: new gamma ceiling (today 15, block 16); deep-accept rounds extend
    p_cont: per-position continuation (acceptance) prob beyond today's g_use
    offload_k: CPU-computed experts per engaged layer on chunk sweeps
    offload_mem: 'hbm' (bank-disjoint) or 'ddr' (contended)
    belady: None | 'bridge16' -> measured miss reduction applied to chunk bytes
    """
    plain_sum = sum(r[2] for r in ROUNDS)
    if skip_draft:
        # drop draft tax ~10 ms on reject rounds after 3 consecutive rejects (hysteresis)
        consec, saved = 0, 0.0
        for r in ROUNDS:
            if r[3] == 0 and r[1] == 0:
                consec += 1
                if consec > 3: saved += 10.0
            else: consec = 0
        plain_sum -= saved

    # chunk bytes ledger
    chunk_bytes = CHUNK_BYTES_GIB
    miss_rel = 1.0
    if resident_lowbit:
        miss_rel *= miss_scale_from_capacity(1.0/kappa_stream)
    if belady == 'bridge16':
        miss_rel *= (1-0.161)   # measured miss delta
    chunk_bytes *= miss_rel * kappa_stream

    # offload: remove k experts/layer from the H2D burst on the 8 chunk sweeps
    off_ms_penalty = 0.0
    if offload_k:
        misses_per_layer = 14976/8/75.0    # ~25 fetch copies per layer-cmd (all ranks)
        frac = min(offload_k/ misses_per_layer, 0.5)
        removed = chunk_bytes * frac
        if offload_mem == 'ddr':
            # CPU read returns 22.8/26.34 of the bytes onto the same node caps and
            # loses arbitration 2.5:1; net effective byte relief ~ 25% of removed
            removed *= 0.25
        chunk_bytes -= removed
        # fold cost on chunk finalize path: 440us x 75 layers x 8 chunks, ~50% overlapped
        off_ms_penalty = 0.440*75*8*0.5

    new_span = chunk_bytes/CHUNK_AGG_GIBS*1000.0
    span_delta = CHUNK_SPAN_MS - new_span      # converts ~1:1 to wall (M3 evidence)

    # verify compute scales with kappa when residents/stream low-bit (DRAM-bound GEMM);
    # conservative: only half the compute is weight-read-bound
    comp_delta = 0.0
    kc = 0.5 + 0.5*kappa_stream
    if kappa_stream < 1.0:
        comp_delta += (1-kc)*CHUNK_FFN_MS*8
        if resident_lowbit:
            comp_delta += (1-kc)*PLAIN_FFN_MS*len(ROUNDS)   # plain-step FFN on 39 rounds

    # gamma extension: deep-accept rounds (g>=8, fully accepted) extend to new ceiling
    tok = COMMITTED
    ext_cost = 0.0
    if extend_gamma and p_cont > 0:
        for g,acc,_,vms in ROUNDS:
            if vms > 0 and acc == g and g >= 8:
                room = extend_gamma - g
                # expected extra accepted tokens: geometric with continuation p_cont
                e_extra = sum(p_cont**i for i in range(1, room+1))
                tok += e_extra
                # marginal verify cost 48 ms/pos; 62% is fetch (scales with byte
                # ledger), 38% compute (scales with kc)
                ext_cost += 48.0 * e_extra * (0.38*(0.5+0.5*kappa_stream)
                                              + 0.62*miss_rel*kappa_stream)

    wall = 211.0 + plain_sum + sum(r[3] for r in ROUNDS) - span_delta - comp_delta \
           + ext_cost + off_ms_penalty
    tps = tok/ (wall/1000.0)
    if verbose:
        print(f"  chunk bytes {CHUNK_BYTES_GIB:.0f}->{chunk_bytes:.0f} GiB, "
              f"span {CHUNK_SPAN_MS:.0f}->{new_span:.0f} ms, tok {tok:.1f}, "
              f"wall {wall:.0f} ms -> {tps:.2f} tok/s")
    return tps

if __name__ == '__main__':
    print("=== calibration ===")
    base_accounting()
    b = wall_with(verbose=True)
    print(f"baseline model: {b:.3f} tok/s (measured 9.596)\n")

    print("=== single levers ===")
    for name, kw in [
        ("bridge16 eviction (measured ceiling)", dict(belady='bridge16')),
        ("stream-tier kappa=0.80 (Q3_K-class)", dict(kappa_stream=0.80)),
        ("stream-tier kappa=0.61 (Q2_K-class)", dict(kappa_stream=0.61)),
        ("kappa=0.61 + low-bit residents (cache x1.64)", dict(kappa_stream=0.61, resident_lowbit=True)),
        ("offload k=4/layer HBM", dict(offload_k=4)),
        ("offload k=8/layer HBM", dict(offload_k=8)),
        ("offload k=4/layer DDR", dict(offload_k=4, offload_mem='ddr')),
        ("gamma 15->31, p_cont=0.3", dict(extend_gamma=31, p_cont=0.3)),
        ("gamma 15->31, p_cont=0.6", dict(extend_gamma=31, p_cont=0.6)),
        ("gamma 15->31, p_cont=0.9", dict(extend_gamma=31, p_cont=0.9)),
        ("skip-draft hysteresis", dict(skip_draft=True)),
    ]:
        t = wall_with(**kw)
        print(f"  {name:48s} {t:6.2f} tok/s  ({(t/b-1)*100:+.1f}%)")

    print("\n=== stacks ===")
    for name, kw in [
        ("kappa .61 + residents + gamma31 p.6", dict(kappa_stream=0.61, resident_lowbit=True, extend_gamma=31, p_cont=0.6)),
        ("kappa .61 + residents + gamma31 p.6 + offload4 HBM", dict(kappa_stream=0.61, resident_lowbit=True, extend_gamma=31, p_cont=0.6, offload_k=4)),
        ("kappa .61 + residents + gamma31 p.6 + off4 + skip", dict(kappa_stream=0.61, resident_lowbit=True, extend_gamma=31, p_cont=0.6, offload_k=4, skip_draft=True)),
        ("kappa .80 + gamma31 p.6 (conservative)", dict(kappa_stream=0.80, extend_gamma=31, p_cont=0.6)),
    ]:
        t = wall_with(verbose=True, **kw)
        print(f"  {name:52s} {t:6.2f} tok/s  ({(t/b-1)*100:+.1f}%)")

    print("\n=== break-even tables (speculation governor) ===")
    print("fire a g-deep chunk iff P(full-accept) > pi* = V(g)/((g+1)*P_plain):")
    for g in (2,4,8,13,20,31):
        for kap in (1.0, 0.61):
            V = 190+47*g*(0.6+0.4*kap)
            print(f"  g={g:2d} kappa={kap:.2f}: V={V:5.0f} ms  pi*={V/((g+1)*155):.2f}")
    print("extend one position iff p_cont > 47*kb/155 :",
          f"kappa=1: {47/155:.2f}, kappa=.61: {47*0.76/155:.2f}")

    print("\n=== plain-window relocation budget (does the union hide?) ===")
    # plain step ~155 ms; 4 links; shared-node-degraded effective agg ~148.7 GiB/s
    window_gib = 0.155 * 148.7
    for kap, rel in [(1.0,1.0),(0.80,1.0),(0.61,1.0),(0.61,0.83)]:
        per_chunk = CHUNK_BYTES_GIB/8 * kap * rel
        print(f"  kappa={kap:.2f} miss_rel={rel:.2f}: union {per_chunk:.1f} GiB vs "
              f"window {window_gib:.1f} GiB -> "
              f"{'FITS (hideable)' if per_chunk <= window_gib else f'exposes {per_chunk-window_gib:.0f} GiB'}")

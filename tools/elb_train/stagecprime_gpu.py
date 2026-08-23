"""GPU engine for the P11.c' extended-signal mixture (torch, one device).

Generalizes stagec_gpu's dense [J, E] masked-tensor engine (same parity
contract: stable sorts, predict-then-update, identical Hedge/ridge math)
with

  * a FEATURE/MEMBER REGISTRY: base 12 stageb features + optional extras
    (trans2 skip-transitions, cross-layer prev-position co-activation,
    gate-weight dynamics, global frequency, conf interaction, embedding-
    bucket memo, draft router taps),
  * per-position CONTENT inputs (token bucket, tap row, draft conf) and
    per-chunk content maxpools for manifest-time scoring (the draft block
    is known at manifest time),
  * an OPEN CHANNEL: scores ranked over ALL 256 experts (~in_prev), not
    just the trail-64 candidate union — the only route to the NOVEL mass
    (true slots outside the union, which candidate-ranking models cannot
    reach by construction). rls_open is a second ridge trained on all-E
    rows,
  * NOVEL-SLOT accounting: novel = true & outside-union & non-boundary,
    reported per variant next to addressable recovery (boundary
    discipline: coverage wins without addressable/novel backing are
    noise-fitting),
  * x13 CUTTRACK (studies/epm/p11-stagecprime/ARCHITECT_REVIEW.md):
    a SOFT-LABEL + DECISION-ROW closed-form head (soft_head — target
    Phi(sel margin to the top-K cut / online drift-MAD sigma) over
    e not in prev8, own A_dr/b_soft, Hedge member "rls_soft"), damped
    Holt level+trend filters on the observed sel process + a per-layer
    cut filter with a 3-member gain grid (filter_feats — features
    holt_fc/cut_margin/cut_prob/trend_z), and a PAST/Oja rank-r
    common-factor forecast (factor_r — features fac_fc/fac_margin).

With cfg.features == () and cfg.open_ch == False the member set and all
formulas reduce exactly to stagec_gpu's — the x0 parity gate.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from . import stage0
from .stageb import EVICT_PROBE_TOPK, MANIFEST_DS, POOL_KS
from .stageb_features import FEATURES as BASE_FEATURES
from .stageb_features import RECENCY_CAP
from .stagec import (BANK as BASE_BANK, HEDGE_ETA, RLS_DECAY,
                     RLS_LAMBDA, RLS_REFRESH)
from .stagec_gpu import _rank_norm_dense

EXTRA_FEATURES = ("trans2", "xprev_up", "xprev_dn", "w_prev", "w_ema",
                  "gfreq", "conf_trans", "memo", "memo2", "ctx",
                  "tap", "xsame")
# extras that can also enter the Hedge bank as raw score members
EXTRA_MEMBERS = ("trans2", "xprev_up", "xprev_dn", "gfreq", "memo", "tap")
# open-channel standalone diagnostics (content members)
OPEN_DIAG = ("memo", "tap", "gfreq")
BASE_VARIANTS = ("combined", "probe_frozen", "rls_online", "trans_ema")
BASE_F = len(BASE_FEATURES)
W_EMA_HALF_LIFE = 8.0
GFREQ_DECAY = 0.999

# ── x13 CUTTRACK (studies/epm/p11-stagecprime/ARCHITECT_REVIEW.md) ──
# Engine-computed features from the score/cut state-space filters
# (appended AFTER the extras, in this order) and the factor stage.
FILTER_FEATURES = ("holt_fc", "cut_margin", "cut_prob", "trend_z")
FACTOR_FEATURES = ("fac_fc", "fac_margin")
# Damped local-level+trend (Holt; Gardner–McKenzie damping) gain grid
# (a1, a2), run in parallel; per-layer active gain = argmin decayed
# innovation MSE (closed-form discrete selection, no gradients).
HOLT_GAINS = ((0.5, 0.1), (0.25, 0.05), (0.1, 0.01))
HOLT_BETA = 0.98         # trend damping
VAR_DECAY = 0.99         # EW innovation-variance decay (q)
GAIN_MSE_DECAY = 0.99    # per-layer gain-selection MSE decay
WARMUP_POS = 4           # trend pinned to 0 for first positions/seq
SIG_DECAY = 0.99         # per-layer EW drift-MAD decay (soft label)
MAD_TO_SIGMA = 1.4826    # Gaussian-consistent MAD -> sigma scale
SOFT_SIG_FLOOR = 1e-4    # sigma floor: soft label degrades to hard
VAR_FLOOR = 1e-6         # variance floor inside sqrt
CLAMP_Z = 20.0           # standardized-feature clamp
MU_DECAY = 0.99          # factor-stage EW mean decay (centering)
OJA_ETA = 0.05           # Oja/PAST subspace step
QR_EVERY = 256           # V re-orthonormalization period
# x13 revision arms (ARCHITECT_REVIEW.md Part D): the soft label is
# KILLED (population-level ranking bias under heterogeneous margin
# variance: E[Phi((s-c)/sig)|x] = Phi(m/sqrt(sig^2+v)) is not a
# monotone transform of eta = Phi(m/sqrt(v)) when v varies across
# rows). Replacements are hard-label WEIGHTINGS (row selection can
# not bias the target):
MIX_POW = 4.0            # gain-mix inverse-MSE sharpness (inf=argmin)
MIX_EPS = 1e-12
DORMANT_REC = 64.0       # dynamics-feature gate: rows with recency
#                          >= trail-64 horizon get NEUTRAL dynamics
#                          features (novel candidates stay memo-ruled)
# x16 bigram memo (ARCHITECT_REVIEW.md Part B.6 ladder): hashed
# (prev_tok, tok) -> per-(layer, bucket) expert counts — measures
# whether token-SEQUENCE context routes beyond the unigram memo.
# The n-gram-table-before-neural rung: a tiny context encoder earns
# a slot ONLY on the fires-but-sparse outcome.
BIGRAM_BOS = np.int64(1 << 20)   # sentinel prev-token at seq start
# π̂ calibration head (ARCHITECT_CALIBRATION.md): rank-conditional
# decayed reliability tables on w_open — the P2 governor's coverage
# curves C(M), expected misses, and marginal-value dC/dM. Rank
# binning is drift-invariant and perfectly balanced (one observation
# per (layer, rank) per position).
CAL_DECAY = 0.999
# pihat2d (ARCHITECT_CALIBRATION §7, fd3ce3a5 naming): 2-D
# calibration ADJUSTMENT — rank x per-layer quantile-bucketed
# score-gap-to-the-rank-(M-K)-cut. Tables accumulate RAW-decayed
# effective counts (mul rho, add 1) so the stationary per-(j,r)
# total is 1/(1-rho) = 1000 observations and the Laplace pseudo-
# count is meaningful; empty bins reduce exactly to the certified
# 1-D rate.
CAL2D_BUCKETS = 8
CAL2D_MICRO = 128        # per-layer decayed micro-histogram (tanh gap)
CAL2D_ALPHA = 8.0        # Laplace pseudo-observations toward p1d
CAL2D_POOL_M = 32        # the governor pool the cut is taken at


def bigram_lut(tokens: np.ndarray, buckets: int) -> np.ndarray:
    """Per-sequence hashed bigram bucket ids [T] from committed token
    ids [T]. Position 0 pairs with the BOS sentinel. Deterministic
    multiplicative-xor hash in wrapping int64 (numpy semantics);
    numpy mod is nonnegative for a positive modulus."""
    tok = tokens.astype(np.int64)
    prev = np.empty_like(tok)
    prev[0] = BIGRAM_BOS
    prev[1:] = tok[:-1]
    with np.errstate(over="ignore"):
        a = prev * np.int64(1000003) + np.int64(12582917)
        b = tok * np.int64(69069) + np.int64(2038074743)
    return (a ^ b) % np.int64(buckets)


def _phi(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, erf form."""
    return 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def _pool_geometry(sc: torch.Tensor, in_prev: torch.Tensor,
                   k_pool: int = 24):
    """π̂ §7.2 position score-geometry scalars from the OPEN ranking
    of w_open scores (current-position information — what no
    past-only EWMA can contain): T1 = mean_j gap across the pool
    boundary (rank k_pool -> rank k_pool+8), T2 = mean_j spread
    inside the pool (rank 8 -> rank k_pool). Flat profiles (small
    T1) = ambiguous boundary = high expected misses."""
    neg = torch.finfo(torch.float32).min
    so = torch.where(~in_prev, sc, torch.full_like(sc, neg))
    sv = torch.sort(so, dim=1, descending=True).values
    t1 = (sv[:, k_pool - 1] - sv[:, k_pool + 7]).mean()
    t2 = (sv[:, 7] - sv[:, k_pool - 1]).mean()
    return t1, t2


@dataclasses.dataclass
class ExpCfg:
    """One named experiment: which extra features enter the ridge X,
    which extra raw members enter the Hedge bank, whether the open
    (all-E) channel is evaluated."""
    name: str
    features: tuple = ()
    members: tuple = ()
    open_ch: bool = False
    memo_buckets: int = 4096
    memo2_buckets: int = 65536  # x16 bigram memo hash size; dense
    #                             table [J, B2, E] f32 = ~5 GB at 64k
    #                             (131072 = ~10 GB, the pre-registered
    #                             collision-confirmation rung)
    quant_fp8: bool = False     # round-trip persistent state through
    #                             float8_e4m3 at the warm->held boundary
    #                             (deployment-precision simulation)
    recur_head: bool = False    # second closed-form head on the SAME
    #                             open-channel A, target = recur16
    #                             (expert routed again within the next
    #                             16 committed positions; free delayed
    #                             label via a 16-slot feature ring)
    recur_horizons: tuple = ()  # extra recur heads (e.g. (32, 64)) —
    #                             same delayed-ring machinery, one
    #                             shared ring trimmed at max horizon
    manifest_head: bool = False  # w_manifest head on the shared A:
    #                             target = membership of the NEXT
    #                             verify chunk's true union (chunk-
    #                             granularity delayed labels); used as
    #                             the topup scorer in eval
    ctx_d: int = 0              # ctx-encoder size-slope overrides
    ctx_d_out: int = 0          # (0 = module defaults 32/16; the
    #                             frozen PCA-32 features and all else
    #                             stay fixed — capacity slope probe)
    novel_head: bool = False    # shared-A head fit ONLY on rows
    #                             OUTSIDE the trail64 union (and
    #                             outside prev8) — row-masked second
    #                             accumulation, exact solve; consumed
    #                             via split pools (novel_splits)
    novel_splits: tuple = ((20, 4), (18, 6), (16, 8))
    #                             (k_in by w_open among in-union,
    #                             k_out by w_novel among out-of-union)
    #                             at fixed total 24
    head_mask_features: tuple = ()  # per-consumer FEATURE masking:
    #                             these features are EXCLUDED from the
    #                             manifest/recur head fits (second
    #                             masked-accumulation A; masked columns
    #                             solve to exactly 0 -> scoring with
    #                             the full Xa is unchanged). w_open
    #                             keeps the full set.
    evict_scorer: str = "rls_open"   # "rls_open" | "recur16"
    evict_scorers: tuple = ()   # multi-scorer dumps (overrides
    #                             evict_scorer): e.g. ("recur16",
    #                             "recur32", "recur64")
    xsame_lag: int = 0          # ARCHITECT_TRICK.md xsame_Δ: within-
    #                             position layer-lag channel. Feature
    #                             row j reads layer j-Δ of the CURRENT
    #                             position (legal in the PIPELINED
    #                             protocol only — metrics of runs with
    #                             this feature are pool@32-pipelined(Δ)
    #                             and must NEVER be mixed with the
    #                             strict ship bars). Table T_Δ[j,p,e]
    #                             = decayed count of (p in top8(t,j-Δ),
    #                             e in top8(t,j)), trans-parity decay;
    #                             updated AFTER scoring (state <= t-1).
    prune_memo_s: float = 0.0   # x20 Tier-2 prunes (ARCHITECT_
    prune_memo_k: int = 0       # ABLATION §3), applied to the
    prune_memo2_s: float = 0.0  # persistent tables IN PLACE at the
    prune_memo2_k: int = 0      # warm->held boundary (the fp8-cert
    prune_tau: float = 0.0      # pattern): memo/memo2 per-(j,bucket)
    #                             row support-s + top-k stamps;
    #                             prune_tau zeroes pair-table entries
    #                             < tau*row_max (trans/xprev/xsame,
    #                             scale-invariant). Held metrics +
    #                             dumps then measure the PRUNED
    #                             deployment; online updates continue
    #                             post-prune (deployment semantics).
    cal2d: bool = False         # pihat2d 2-D adjustment on the π̂
    #                             head (rank x gap-quantile bucket;
    #                             own tables per stream — stack-
    #                             agnostic, requires cal_head)
    cal_head: bool = False      # π̂ calibration head
    #                             (ARCHITECT_CALIBRATION.md): per-
    #                             layer decayed rank-conditional hit
    #                             rates of the w_open open ranking
    #                             (+ prev8 slots by score) -> C(M) /
    #                             expected-miss / marginal curves for
    #                             the P2 governor. STRICT configs
    #                             only (never on xsame runs — the
    #                             pipelined score stream is a
    #                             different distribution).
    xsame_soft: bool = True     # Q&A addendum tier (b): feature =
    #                             sum_p sel(t,j-Δ,p) * T_Δ[j,p,e] over
    #                             the FULL 256-sel row (soft form,
    #                             strictly more information at equal
    #                             state/gather cost). False = hard
    #                             8-sparse form (top8 of the sel row).
    ablate: tuple = ()          # ARCHITECT_ABLATION.md Tier 1: zero
    #                             these feature columns in X (before
    #                             scoring AND accumulation). Tikhonov
    #                             zero-column lemma => the remaining
    #                             weights are EXACTLY the reduced-
    #                             system solution (exact LOFO, not an
    #                             approximation). Ablated extras skip
    #                             their table builds and exports;
    #                             structural bookkeeping (t16/t64/
    #                             prev8/valid) is untouched; raw Hedge
    #                             members stay real (report-only,
    #                             pre-registered).

    @property
    def horizons(self) -> tuple:
        hs = set(self.recur_horizons)
        if self.recur_head:
            hs.add(16)
        return tuple(sorted(hs))

    @property
    def eff_evict_scorers(self) -> tuple:
        return self.evict_scorers or (self.evict_scorer,)
    attn_mode: str = ""         # "" | "pooled" | "expert_query" — the
    #                             §3-P11(4)(iii) tiny-attention
    #                             challenger (stagecprime_attn), an
    #                             ONLINE residual over the open ridge
    attn_zero_gate: bool = False   # LayerScale gamma=0 output gate:
    #                             init == ridge ranking exactly;
    #                             gamma trajectory logged (capacity-
    #                             value diagnostic)
    attn_hedge: bool = False    # attention output ALSO a Hedge member
    #                             next to rls_open (regret floor)
    soft_head: bool = False     # x13a CUTTRACK: soft-label (Phi of
    #                             sel margin to the top-K cut over the
    #                             online drift-MAD sigma) + DECISION-
    #                             ROW (e not in prev8) ridge head on
    #                             its own A_dr/b_soft; Hedge member
    #                             "rls_soft" (open rank domain)
    filter_feats: bool = False  # x13b: damped Holt level+trend
    #                             filters per (gain, j, e) on sel +
    #                             per-layer cut filter -> features
    #                             holt_fc/cut_margin/cut_prob/trend_z
    factor_r: int = 0           # x13c: PAST/Oja rank-r subspace per
    #                             layer over centered sel -> features
    #                             fac_fc/fac_margin (0 = off)
    band_heads: bool = False    # x13a2 factorial: three HARD-label
    #                             closed-form heads — rls_dr (decision
    #                             rows), rls_drband (decision rows,
    #                             |sel-cut| <= sigma_j excluded),
    #                             rls_band (all rows, band excluded)
    dormancy_gate: bool = False  # x13b2: neutral dynamics features
    #                             (cut_margin/-Z, cut_prob 0, trend_z
    #                             0, fac_margin/-Z) on rows with
    #                             recency >= 64 — memo rules the
    #                             novel class again
    gain_mix: bool = False      # x13b2: inverse-MSE^4 gain MIXTURE
    #                             instead of per-layer argmin (the
    #                             x13d-diag flip churn, closed form)

    def __post_init__(self):
        for f in self.features:
            if f not in EXTRA_FEATURES:
                raise ValueError(f"unknown extra feature {f!r}")
        for m in self.members:
            if m not in EXTRA_MEMBERS or m not in self.features:
                raise ValueError(f"member {m!r} must be an enabled "
                                 f"extra feature")
        if self.soft_head and not self.open_ch:
            raise ValueError("soft_head deploys in the open channel "
                             "— open_ch required")
        if self.factor_r < 0:
            raise ValueError("factor_r must be >= 0")
        if self.factor_r and not self.filter_feats:
            raise ValueError("factor_r needs the cut/variance state "
                             "— filter_feats required")
        if self.band_heads and not self.open_ch:
            raise ValueError("band_heads deploy in the open channel "
                             "— open_ch required")
        if (self.dormancy_gate or self.gain_mix) \
                and not self.filter_feats:
            raise ValueError("dormancy_gate/gain_mix act on the "
                             "filter features — filter_feats required")
        for f in self.ablate:
            if f not in self.feat_names:
                raise ValueError(f"ablate feature {f!r} not in the "
                                 f"feature set")
        if "xsame" in self.features and self.xsame_lag < 1:
            raise ValueError("xsame requires xsame_lag >= 1")
        if self.xsame_lag and "xsame" not in self.features:
            raise ValueError("xsame_lag set without the xsame "
                             "feature")
        if self.cal_head and not self.open_ch:
            raise ValueError("cal_head calibrates the open ranking "
                             "— open_ch required")
        if self.cal2d and not self.cal_head:
            raise ValueError("cal2d is an adjustment ON the π̂ head "
                             "— cal_head required")
        if (self.prune_memo_s or self.prune_memo_k) \
                and "memo" not in self.features:
            raise ValueError("memo prune without the memo feature")
        if (self.prune_memo2_s or self.prune_memo2_k) \
                and "memo2" not in self.features:
            raise ValueError("memo2 prune without the memo2 feature")
        if self.prune_tau < 0 or self.prune_tau >= 1:
            raise ValueError("prune_tau must be in [0, 1)")

    @property
    def has_prune(self) -> bool:
        return bool(self.prune_memo_s or self.prune_memo_k
                    or self.prune_memo2_s or self.prune_memo2_k
                    or self.prune_tau)
        # cal_head + xsame is ALLOWED (ship3 green light, teacher
        # stream-separation rule): the head trains in-run on the
        # run's OWN score stream, so a pipelined run grows pipelined
        # tables by construction. What remains forbidden is REUSING
        # tables across streams — enforced by packaging/consumers
        # (feature_spec labels the stream), not by cfg.

    @property
    def feat_names(self) -> tuple:
        out = tuple(BASE_FEATURES) + tuple(
            f for f in EXTRA_FEATURES if f in self.features)
        if self.filter_feats:
            out = out + FILTER_FEATURES
        if self.factor_r:
            out = out + FACTOR_FEATURES
        return out

    @property
    def member_names(self) -> tuple:
        out = tuple(BASE_BANK) + tuple(
            m for m in EXTRA_MEMBERS if m in self.members)
        if self.open_ch:
            out = out + ("rls_open",)
        if self.soft_head:
            out = out + ("rls_soft",)
        if self.band_heads:
            out = out + ("rls_dr", "rls_drband", "rls_band")
        if self.attn_hedge:
            out = out + ("attn",)
        return out

    @property
    def variant_names(self) -> tuple:
        out = list(BASE_VARIANTS)
        out += [m for m in EXTRA_MEMBERS if m in self.members]
        if self.open_ch:
            out += ["rls_open", "combined_open"]
            out += [f"{m}_open" for m in OPEN_DIAG if m in self.members]
        if self.soft_head:
            out += ["rls_soft"]
        if self.band_heads:
            out += ["rls_dr", "rls_drband", "rls_band"]
        if self.attn_mode:
            out += ["attn_open"]
        return tuple(out)


class CPrimeState:
    """All engine state on one device. Cross-sequence (warm) state:
    trans/trans2/xprev tables, gfreq, memo, ridge stats, Hedge weights;
    everything else resets per sequence."""

    def __init__(self, J: int, E: int, K: int, dev: torch.device,
                 probe_packed: np.ndarray, sigma: np.ndarray,
                 cfg: ExpCfg):
        self.J, self.E, self.K, self.dev = J, E, K, dev
        self.cfg = cfg
        self.feat_names = cfg.feat_names
        self.member_names = cfg.member_names
        self.F = len(self.feat_names)
        f32 = dict(device=dev, dtype=torch.float32)
        if cfg.ablate:
            am = torch.ones(self.F, device=dev, dtype=torch.float32)
            for f in cfg.ablate:
                am[self.feat_names.index(f)] = 0.0
            self.ablate_mask = am    # [F] — exact zero-column LOFO
        else:
            self.ablate_mask = None
        self.t16 = torch.zeros(J, E, **f32)
        self.t64 = torch.zeros(J, E, **f32)
        self.ring16: list = []
        self.ring64: list = []
        self.trans = torch.zeros(J, E, E, **f32)
        self.trans_scale = 1.0
        self.prev8: torch.Tensor | None = None       # [J, K] i64
        self.prev2_8: torch.Tensor | None = None     # [J, K] i64 (t-2)
        self.prev8_mask = torch.zeros(J, E, **f32)
        self.ema2 = torch.zeros(J, E, **f32)
        self.ema8 = torch.zeros(J, E, **f32)
        self.n2 = 0.0
        self.n8 = 0.0
        self.d2 = 0.5 ** (1.0 / 2.0)
        self.d8 = 0.5 ** (1.0 / 8.0)
        self.g_prev = torch.zeros(J, E, **f32)
        self.g_prev2 = torch.zeros(J, E, **f32)
        self.cut_prev = torch.zeros(J, **f32)
        self.last_seen = torch.full((J, E), -1, device=dev,
                                    dtype=torch.int64)
        self.t_idx = 0
        # extras state
        if "trans2" in cfg.features:
            self.trans2 = torch.zeros(J, E, E, **f32)
            self.trans2_scale = 1.0
        if "xsame" in cfg.features and "xsame" not in cfg.ablate:
            self.xsame = torch.zeros(J, E, E, **f32)
            self.xsame_scale = 1.0
        if "xprev_up" in cfg.features and "xprev_up" not in cfg.ablate:
            self.xu = torch.zeros(J, E, E, **f32)
            self.xu_scale = 1.0
        if "xprev_dn" in cfg.features and "xprev_dn" not in cfg.ablate:
            self.xd = torch.zeros(J, E, E, **f32)
            self.xd_scale = 1.0
        if "w_prev" in cfg.features or "w_ema" in cfg.features:
            self.w_prev = torch.zeros(J, E, **f32)
            self.w_ema = torch.zeros(J, E, **f32)
            self.n_w = 0.0
            self.d_w = 0.5 ** (1.0 / W_EMA_HALF_LIFE)
        if "gfreq" in cfg.features:
            self.gfreq = torch.zeros(J, E, **f32)
            self.gfreq_scale = 1.0
        if "memo" in cfg.features:
            self.memo = torch.zeros(J, cfg.memo_buckets, E, **f32)
        if "memo2" in cfg.features:
            self.memo2 = torch.zeros(J, cfg.memo2_buckets, E, **f32)
        # ridge (f64) + hedge
        self.A = torch.zeros(J, self.F + 1, self.F + 1, device=dev,
                             dtype=torch.float64)
        self.bvec = torch.zeros(J, self.F + 1, device=dev,
                                dtype=torch.float64)
        self.wr = torch.zeros(J, self.F + 1, device=dev,
                              dtype=torch.float64)
        if cfg.open_ch:
            self.Ao = torch.zeros_like(self.A)
            self.bo = torch.zeros_like(self.bvec)
            self.wo = torch.zeros_like(self.wr)
        if cfg.horizons or cfg.manifest_head:
            if not cfg.open_ch:
                raise ValueError("shared-A heads require open_ch")
        if cfg.horizons:
            self.brec_h = {h: torch.zeros(J, self.F + 1, device=dev,
                                          dtype=torch.float64)
                           for h in cfg.horizons}
            self.wrec_h = {h: torch.zeros(J, self.F + 1, device=dev,
                                          dtype=torch.float64)
                           for h in cfg.horizons}
            self.rring: list = []       # (Xa f32 [J,E,F+1], pos_idx)
        if cfg.manifest_head:
            self.bman = torch.zeros(J, self.F + 1, device=dev,
                                    dtype=torch.float64)
            self.wman = torch.zeros_like(self.bman)
            self.cbuf: list = []        # current chunk's X (f32)
        if cfg.novel_head:
            if not cfg.open_ch:
                raise ValueError("novel_head requires open_ch")
            self.A_nov = torch.zeros(J, self.F + 1, self.F + 1,
                                     device=dev, dtype=torch.float64)
            self.b_nov = torch.zeros(J, self.F + 1, device=dev,
                                     dtype=torch.float64)
            self.w_nov = torch.zeros_like(self.b_nov)
        if cfg.head_mask_features and (cfg.horizons
                                       or cfg.manifest_head):
            for f in cfg.head_mask_features:
                if f not in self.feat_names:
                    raise ValueError(f"head_mask feature {f!r} not "
                                     f"in the feature set")
            m = torch.ones(self.F + 1, device=dev,
                           dtype=torch.float64)
            for f in cfg.head_mask_features:
                m[self.feat_names.index(f)] = 0.0
            self.head_mask = m          # [F+1], bias kept
            self.A_sub = torch.zeros(J, self.F + 1, self.F + 1,
                                     device=dev, dtype=torch.float64)
        else:
            self.head_mask = None
        # x13 CUTTRACK state (ARCHITECT_REVIEW.md Part B.1 / Part C)
        if cfg.soft_head:
            self.Adr = torch.zeros_like(self.A)       # decision rows
            self.bsoft = torch.zeros_like(self.bvec)
            self.wsoft = torch.zeros_like(self.wr)
            self.sig_drift = torch.zeros(J, **f32)    # EW drift MAD
            self.sig_n = 0.0                          # bias correction
        if cfg.filter_feats:
            ng = len(HOLT_GAINS)
            self.a1 = torch.tensor([g[0] for g in HOLT_GAINS], **f32)
            self.a2 = torch.tensor([g[1] for g in HOLT_GAINS], **f32)
            self.f_lv = torch.zeros(ng, J, E, **f32)  # level
            self.f_tr = torch.zeros(ng, J, E, **f32)  # damped trend
            self.f_qv = torch.zeros(ng, J, E, **f32)  # EW innov var
            self.f_m = torch.zeros(ng, J, **f32)      # gain-sel MSE
            self.c_lv = torch.zeros(ng, J, **f32)     # cut level
            self.c_tr = torch.zeros(ng, J, **f32)
            self.c_qc = torch.zeros(ng, J, **f32)
            self.f_init = False        # per-seq: filters primed?
            self.f_seq_n = 0           # positions since seq start
        if cfg.cal_head:
            # π̂ reliability tables (cross-sequence): open ranks
            # 1..E-K and prev8 slots 1..K by score order
            self.calH = torch.zeros(J, E - K, **f32)
            self.calP = torch.zeros(J, K, **f32)
            self.cal_n = 0.0
        if cfg.cal2d:
            # pihat2d: [J, E, 8] hit/exposure effective counts in
            # RANK order (open ranks 0..E-K-1, then prev slots) +
            # the per-layer decayed gap micro-histogram
            self.calH2 = torch.zeros(J, E, CAL2D_BUCKETS, **f32)
            self.calN2 = torch.zeros(J, E, CAL2D_BUCKETS, **f32)
            self.cal_ghist = torch.zeros(J, CAL2D_MICRO, **f32)
        if cfg.band_heads:
            # three hard-label weighted heads (Part D factorial):
            # DR, DR+band-excluded, all-rows+band-excluded
            self.Ad = torch.zeros_like(self.A)
            self.bd = torch.zeros_like(self.bvec)
            self.wd = torch.zeros_like(self.wr)
            self.Adb = torch.zeros_like(self.A)
            self.bdb = torch.zeros_like(self.bvec)
            self.wdb = torch.zeros_like(self.wr)
            self.Abn = torch.zeros_like(self.A)
            self.bbn = torch.zeros_like(self.bvec)
            self.wbn = torch.zeros_like(self.wr)
        if cfg.factor_r:
            r = cfg.factor_r
            g = torch.random.get_rng_state()
            torch.manual_seed(0)
            v0 = torch.randn(J, E, r)
            torch.random.set_rng_state(g)
            self.V = torch.linalg.qr(v0)[0].to(dev)   # orthonormal
            self.f_mu = torch.zeros(J, E, **f32)      # EW mean (ctr)
            self.mu_n = 0.0
            self.u_lv = torch.zeros(J, r, **f32)      # factor level
            self.u_tr = torch.zeros(J, r, **f32)      # factor trend
            self._qr_since = 0
        self._since = 0
        self.hw = torch.full((J, len(self.member_names)),
                             1.0 / len(self.member_names),
                             device=dev, dtype=torch.float64)
        # frozen probe params (over the BASE 12 features)
        d = BASE_F
        self.mu = torch.as_tensor(probe_packed[:d], **f32)
        self.sd = torch.as_tensor(probe_packed[d:2 * d], **f32)
        self.pw = torch.as_tensor(probe_packed[2 * d:], **f32)
        self.sigma = torch.as_tensor(sigma, **f32)
        self.jj = torch.arange(J, device=dev)
        self.layer_f = (self.jj.float() / max(1, J - 1))[:, None] \
            .expand(J, E)
        self._zeros_je = torch.zeros(J, E, **f32)

    def reset_seq(self):
        self.t16.zero_()
        self.t64.zero_()
        self.ring16.clear()
        self.ring64.clear()
        self.prev8 = None
        self.prev2_8 = None
        self.prev8_mask.zero_()
        self.ema2.zero_()
        self.ema8.zero_()
        self.n2 = self.n8 = 0.0
        self.g_prev.zero_()
        self.g_prev2.zero_()
        self.cut_prev.zero_()
        self.last_seen.fill_(-1)
        self.t_idx = 0
        if hasattr(self, "w_prev"):
            self.w_prev.zero_()
            self.w_ema.zero_()
            self.n_w = 0.0
        if self.cfg.horizons:
            # tail-of-sequence windows are truncated — drop them
            # rather than train on biased short-horizon labels
            self.rring.clear()
        if self.cfg.manifest_head:
            self.cbuf.clear()   # last chunk has no next-chunk label
        if self.cfg.filter_feats:
            # level/trend (and factor scores) track the per-sequence
            # hidden-state trajectory -> re-primed at the first
            # position; variance/gain-MSE/subspace/mean persist as
            # cross-sequence priors
            self.f_init = False
            self.f_seq_n = 0

    # ── features + raw member scores (PRE-update state) ─────────────────

    def _pair_score(self, table: torch.Tensor, scale: float,
                    src8: torch.Tensor | None) -> torch.Tensor:
        if src8 is None:
            return self._zeros_je
        return table[self.jj[:, None], src8].sum(dim=1) * scale

    def features(self, tap_now: torch.Tensor | None = None,
                 conf_now: float = 0.0,
                 memo_now: torch.Tensor | None = None,
                 bucket: int = -1, bucket2: int = -1,
                 ctx_now: torch.Tensor | None = None,
                 sel_now: torch.Tensor | None = None,
                 tops_now: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """(X [J,E,F], valid [J,E] bool, raw member-score dict).
        Content inputs: tap_now [J,E] (router-tap sel), memo_now [J,E]
        (overrides the bucket lookup — the manifest chunk maxpool), or
        bucket >= 0 (per-position token bucket); bucket2 >= 0 is the
        hashed bigram bucket (x16 — manifest calls leave it -1, the
        column is position-level only)."""
        J, E = self.J, self.E
        cfg = self.cfg
        valid = self.t64 > 0
        tp = self._pair_score(self.trans, self.trans_scale, self.prev8)
        rec = torch.where(self.last_seen >= 0,
                          (self.t_idx - self.last_seen).float(),
                          torch.full_like(self.t16, RECENCY_CAP))
        rec = rec.clamp(max=RECENCY_CAP)
        tpm = torch.where(valid, tp, torch.full_like(tp, float("inf")))
        ranks = torch.argsort(
            torch.argsort(-tpm, dim=1, stable=True), dim=1,
            stable=True).float()
        n_inv = (~valid).sum(dim=1, keepdim=True).float()
        nval = valid.sum(dim=1, keepdim=True).float()
        trank = (ranks - n_inv) / (nval - 1).clamp(min=1)
        e2 = self.ema2 / self.n2 if self.n2 > 0 else self.ema2
        e8 = self.ema8 / self.n8 if self.n8 > 0 else self.ema8
        cols = [self.prev8_mask, self.t16, self.t64, rec,
                tp, trank, e2, e8, self.g_prev,
                self.g_prev - self.cut_prev[:, None],
                self.g_prev - self.g_prev2, self.layer_f]
        raw: dict = {}
        for f in EXTRA_FEATURES:
            if f not in cfg.features:
                continue
            if f in cfg.ablate:
                # ablated extra: exact zero column, table lookup (and
                # its build, see __init__/update) skipped entirely
                v = self._zeros_je
                raw[f] = v
                cols.append(v)
                continue
            if f == "trans2":
                v = self._pair_score(self.trans2, self.trans2_scale,
                                     self.prev2_8)
                raw["trans2"] = v
            elif f == "xprev_up":
                # source: layer j+1 at t-1 -> target layer j at t
                if self.prev8 is None:
                    v = self._zeros_je
                else:
                    v = torch.zeros(J, E, device=self.dev)
                    v[:-1] = (self.xu[self.jj[:-1, None],
                                      self.prev8[1:]].sum(dim=1)
                              * self.xu_scale)
                raw["xprev_up"] = v
            elif f == "xprev_dn":
                if self.prev8 is None:
                    v = self._zeros_je
                else:
                    v = torch.zeros(J, E, device=self.dev)
                    v[1:] = (self.xd[self.jj[1:, None],
                                     self.prev8[:-1]].sum(dim=1)
                             * self.xd_scale)
                raw["xprev_dn"] = v
            elif f == "w_prev":
                v = self.w_prev
            elif f == "w_ema":
                v = self.w_ema / self.n_w if self.n_w > 0 else self.w_ema
            elif f == "gfreq":
                g = self.gfreq * self.gfreq_scale
                v = g / (1.0 + g.sum(dim=1, keepdim=True))
                raw["gfreq"] = v
            elif f == "conf_trans":
                v = tp * conf_now
            elif f == "memo":
                if memo_now is not None:
                    v = memo_now
                elif bucket >= 0:
                    row = self.memo[:, bucket, :]
                    v = row / (1.0 + row.sum(dim=1, keepdim=True))
                else:
                    v = self._zeros_je
                raw["memo"] = v
            elif f == "memo2":
                if bucket2 >= 0:
                    row = self.memo2[:, bucket2, :]
                    v = row / (1.0 + row.sum(dim=1, keepdim=True))
                else:
                    v = self._zeros_je
                raw["memo2"] = v
            elif f == "ctx":
                # x17 ctxformer output (engine-supplied, position-
                # level only; manifest calls leave it None -> zeros)
                v = ctx_now if ctx_now is not None else self._zeros_je
                raw["ctx"] = v
            elif f == "tap":
                v = tap_now if tap_now is not None else self._zeros_je
                raw["tap"] = v
            elif f == "xsame":
                # PIPELINED-protocol feature: row j reads the CURRENT
                # position's layer j-Δ (sel_now = sel(t), observed at
                # deploy time when layer j-Δ has run). Table state is
                # <= t-1 (update happens after scoring). Rows j < Δ
                # are zeros. Manifest/chunk queries pass sel_now=None
                # -> zeros (and the feature is head-masked anyway).
                d = cfg.xsame_lag
                if sel_now is None:
                    v = self._zeros_je
                else:
                    v = torch.zeros(J, E, device=self.dev)
                    if cfg.xsame_soft:
                        v[d:] = torch.einsum(
                            "jp,jpe->je", sel_now[:-d],
                            self.xsame[d:]) * self.xsame_scale
                    else:
                        # hard sources = ROUTED top8 of layer j-Δ
                        # (teacher review of 33b3d57e: what deploy
                        # observes is the routing decision, not a
                        # sel re-argsort; ties/numerics can differ)
                        if tops_now is None:
                            v = self._zeros_je
                        else:
                            src8 = tops_now[:-d]
                            v[d:] = (self.xsame[self.jj[d:, None],
                                                src8].sum(dim=1)
                                     * self.xsame_scale)
                raw["xsame"] = v
            cols.append(v)
        if cfg.filter_feats:
            # one-step forecasts; state through t-1
            if cfg.gain_mix:
                # inverse-MSE^4 gain MIXTURE (f64: cold-start values
                # overflow f32 before normalization)
                wg = (self.f_m.double() + MIX_EPS).pow(-MIX_POW)
                wg = (wg / wg.sum(dim=0, keepdim=True)).float()
                lv = torch.einsum("gj,gje->je", wg, self.f_lv)
                tr = torch.einsum("gj,gje->je", wg, self.f_tr)
                qv = torch.einsum("gj,gje->je", wg, self.f_qv)
                ch = torch.einsum(
                    "gj,gj->j", wg,
                    self.c_lv + HOLT_BETA * self.c_tr)     # [J]
                qc = torch.einsum("gj,gj->j", wg, self.c_qc)
            else:
                # per-layer ACTIVE gain (argmin decayed innov MSE)
                gidx = self.f_m.argmin(dim=0)              # [J]
                lv = self.f_lv[gidx, self.jj]              # [J, E]
                tr = self.f_tr[gidx, self.jj]
                qv = self.f_qv[gidx, self.jj]
                ch = (self.c_lv[gidx, self.jj]
                      + HOLT_BETA * self.c_tr[gidx, self.jj])  # [J]
                qc = self.c_qc[gidx, self.jj]
            sh = lv + HOLT_BETA * tr
            den = (qv + qc[:, None] + VAR_FLOOR).sqrt()
            marg = ((sh - ch[:, None]) / den).clamp(-CLAMP_Z, CLAMP_Z)
            prob = _phi(marg)
            tz = (tr / (qv + VAR_FLOOR).sqrt()).clamp(-CLAMP_Z,
                                                      CLAMP_Z)
            if cfg.dormancy_gate:
                # neutral CONSTANT dynamics on dormant rows: no rank
                # noise among novel candidates (memo rules there)
                act = rec < DORMANT_REC
                marg = torch.where(act, marg,
                                   torch.full_like(marg, -CLAMP_Z))
                prob = torch.where(act, prob,
                                   torch.zeros_like(prob))
                tz = torch.where(act, tz, torch.zeros_like(tz))
            cols += [sh, marg, prob, tz]
            if cfg.factor_r:
                muh = (self.f_mu / self.mu_n if self.mu_n > 0
                       else self.f_mu)
                uh = self.u_lv + HOLT_BETA * self.u_tr
                ff = muh + torch.einsum("jer,jr->je", self.V, uh)
                fm = ((ff - ch[:, None]) / den).clamp(-CLAMP_Z,
                                                      CLAMP_Z)
                if cfg.dormancy_gate:
                    fm = torch.where(act, fm,
                                     torch.full_like(fm, -CLAMP_Z))
                cols += [ff, fm]
        X = torch.stack(cols, dim=2)
        if self.ablate_mask is not None:
            # exact LOFO: zero the ablated columns (base features
            # incl. trans/trans_rank) before scoring AND accumulation
            # — every consumer (ridge/open/head A's, recur ring,
            # manifest buffer, evict scorers) sees the masked X.
            X = X * self.ablate_mask
        raw["trans"] = tp
        return X, valid, raw

    def member_scores(self, X: torch.Tensor, valid: torch.Tensor,
                      raw: dict) -> torch.Tensor:
        """[M, J, E] raw member scores in member_names order."""
        z = (X[:, :, :BASE_F] - self.mu) / self.sd
        probe = torch.sigmoid(
            torch.einsum("jef,f->je", z, self.pw[:-1]) + self.pw[-1])
        Xa = torch.cat([X.double(),
                        torch.ones(self.J, self.E, 1, device=self.dev,
                                   dtype=torch.float64)], dim=2)
        rls = torch.einsum("jef,jf->je", Xa, self.wr).float()
        base = {"b0_prev": self.prev8_mask, "trans_ema": raw["trans"],
                "score_ema2": X[:, :, 6], "trail16": self.t16,
                "probe_frozen": probe, "rls_online": rls}
        rows = []
        for m in self.member_names:
            if m in base:
                rows.append(base[m])
            elif m == "rls_open":
                rows.append(torch.einsum("jef,jf->je", Xa,
                                         self.wo).float())
            elif m == "rls_soft":
                rows.append(torch.einsum("jef,jf->je", Xa,
                                         self.wsoft).float())
            elif m == "rls_dr":
                rows.append(torch.einsum("jef,jf->je", Xa,
                                         self.wd).float())
            elif m == "rls_drband":
                rows.append(torch.einsum("jef,jf->je", Xa,
                                         self.wdb).float())
            elif m == "rls_band":
                rows.append(torch.einsum("jef,jf->je", Xa,
                                         self.wbn).float())
            elif m == "attn":
                # placeholder — the engine overwrites this row with
                # the challenger logits (needs the rls_open row first)
                rows.append(torch.zeros_like(self.prev8_mask))
            else:
                rows.append(raw[m])
        return torch.stack(rows)

    def member_masks(self, valid: torch.Tensor) -> torch.Tensor:
        """[M, J, E] bool ranking domain per member: base members are
        candidate-restricted (parity), extras/open rank all experts."""
        rows = []
        allE = torch.ones_like(valid)
        for m in self.member_names:
            rows.append(valid if m in BASE_BANK else allE)
        return torch.stack(rows)

    # ── updates (reveal position t) ─────────────────────────────────────

    def _pair_update(self, table: torch.Tensor, scale_attr: str,
                     src8: torch.Tensor, tgt8: torch.Tensor,
                     lo: int = 0, hi: int | None = None) -> None:
        """table[j, src, tgt] += 1/scale for j in [lo, hi); decays by
        0.99/position via the lazy scale (trans_ema parity)."""
        E = self.E
        scale = getattr(self, scale_attr) * 0.99
        setattr(self, scale_attr, scale)
        w_inc = 1.0 / scale
        jr = self.jj[lo:hi]
        flat = (jr[:, None, None] * E * E
                + src8[:, :, None] * E + tgt8[:, None, :]).reshape(-1)
        table.view(-1).scatter_add_(
            0, flat, torch.full_like(flat, w_inc, dtype=torch.float32))
        if scale < 1e-20:
            table.mul_(scale)
            setattr(self, scale_attr, 1.0)

    def update(self, tops: torch.Tensor, sel: torch.Tensor,
               top_w: torch.Tensor, X: torch.Tensor,
               valid: torch.Tensor, yb: torch.Tensor,
               bucket: int = -1, bucket2: int = -1) -> None:
        J, E, K = self.J, self.E, self.K
        cfg = self.cfg
        ones = torch.ones(J, K, device=self.dev)
        # ridge stats
        vf = valid.double()
        Xa = torch.cat([X.double(),
                        torch.ones(J, E, 1, device=self.dev,
                                   dtype=torch.float64)], dim=2)
        Xv = Xa * vf[:, :, None]
        self.A.mul_(RLS_DECAY).add_(torch.einsum("jef,jeg->jfg", Xv, Xv))
        self.bvec.mul_(RLS_DECAY).add_(
            torch.einsum("jef,je->jf", Xv, yb.double() * vf))
        if cfg.open_ch:
            self.Ao.mul_(RLS_DECAY).add_(
                torch.einsum("jef,jeg->jfg", Xa, Xa))
            self.bo.mul_(RLS_DECAY).add_(
                torch.einsum("jef,je->jf", Xa, yb.double()))
        if cfg.horizons:
            for h in cfg.horizons:
                self.brec_h[h].mul_(RLS_DECAY)
            self.rring.append((Xa.float(), self.t_idx))
        if cfg.manifest_head:
            self.bman.mul_(RLS_DECAY)
        if self.head_mask is not None:
            Xm = Xa * self.head_mask
            self.A_sub.mul_(RLS_DECAY).add_(
                torch.einsum("jef,jeg->jfg", Xm, Xm))
        if cfg.novel_head:
            # ROW-masked accumulation: only e outside the trail64
            # union and outside prev8 (prev8_mask still holds t-1
            # here — the prediction-time semantics)
            out_rows = (~valid & (self.prev8_mask == 0)).double()
            Xn = Xa * out_rows[:, :, None]
            self.A_nov.mul_(RLS_DECAY).add_(
                torch.einsum("jef,jeg->jfg", Xn, Xn))
            self.b_nov.mul_(RLS_DECAY).add_(
                torch.einsum("jef,je->jf", Xn,
                             yb.double() * out_rows))
        # ── x13 CUTTRACK: soft-label head + score/cut filters ──
        if cfg.soft_head or cfg.filter_feats or cfg.band_heads:
            cutk = torch.topk(sel, K, dim=1).values[:, -1]   # [J]
        if cfg.band_heads:
            # hard-label weighted heads; the noise band |sel - cut|
            # <= sigma_j (stage-0 drift MAD — the boundary scale) is
            # EXCLUDED from the *band heads'* fit: pure row
            # selection, the target stays the deployed one
            band = (sel - cutk[:, None]).abs() > self.sigma[:, None]
            drm = self.prev8_mask == 0        # decision rows at t
            yd = yb.double()
            for an, bn, mask in (("Ad", "bd", drm),
                                 ("Adb", "bdb", drm & band),
                                 ("Abn", "bbn", band)):
                Xw = Xa * mask[:, :, None].double()
                getattr(self, an).mul_(RLS_DECAY).add_(
                    torch.einsum("jef,jeg->jfg", Xw, Xw))
                getattr(self, bn).mul_(RLS_DECAY).add_(
                    torch.einsum("jef,je->jf", Xw, yd))
        if cfg.soft_head:
            if self.t_idx >= 1:
                drift = (sel - self.g_prev).abs() \
                    .median(dim=1).values                    # [J]
                self.sig_drift.mul_(SIG_DECAY).add_(
                    drift * MAD_TO_SIGMA, alpha=1 - SIG_DECAY)
                self.sig_n = self.sig_n * SIG_DECAY + (1 - SIG_DECAY)
            if self.sig_n > 0:
                sig = (self.sig_drift / self.sig_n) \
                    .clamp(min=SOFT_SIG_FLOOR)
            else:       # no drift observed yet: hard-label limit
                sig = torch.full_like(self.sig_drift,
                                      SOFT_SIG_FLOOR)
            # Rao-Blackwellized label: Phi of the sel margin to the
            # top-K cut over the drift-MAD scale (drift MAD ~ sqrt(2)
            # x per-position jitter sigma — the review's denominator)
            ysoft = _phi((sel - cutk[:, None]) / sig[:, None])
            # DECISION rows only: the pool at t ranks e not in
            # top8(t-1); prev8_mask still holds t-1 here
            dr = (self.prev8_mask == 0).double()
            Xdr = Xa * dr[:, :, None]
            self.Adr.mul_(RLS_DECAY).add_(
                torch.einsum("jef,jeg->jfg", Xdr, Xdr))
            self.bsoft.mul_(RLS_DECAY).add_(
                torch.einsum("jef,je->jf", Xdr, ysoft.double()))
        if cfg.filter_feats:
            if not self.f_init:
                self.f_lv.copy_(sel[None].expand_as(self.f_lv))
                self.f_tr.zero_()
                self.c_lv.copy_(cutk[None].expand_as(self.c_lv))
                self.c_tr.zero_()
                if cfg.factor_r:
                    muh = (self.f_mu / self.mu_n if self.mu_n > 0
                           else self.f_mu)
                    self.u_lv.copy_(torch.einsum(
                        "jer,je->jr", self.V, sel - muh))
                    self.u_tr.zero_()
                self.f_init = True
            else:
                fc = self.f_lv + HOLT_BETA * self.f_tr
                inn = sel[None] - fc
                self.f_lv.copy_(fc + self.a1[:, None, None] * inn)
                self.f_tr.mul_(HOLT_BETA).add_(
                    self.a2[:, None, None] * inn)
                self.f_qv.mul_(VAR_DECAY).add_(inn * inn,
                                               alpha=1 - VAR_DECAY)
                self.f_m.mul_(GAIN_MSE_DECAY).add_(
                    inn.pow(2).mean(dim=2), alpha=1 - GAIN_MSE_DECAY)
                cc = self.c_lv + HOLT_BETA * self.c_tr
                ic = cutk[None] - cc
                self.c_lv.copy_(cc + self.a1[:, None] * ic)
                self.c_tr.mul_(HOLT_BETA).add_(self.a2[:, None] * ic)
                self.c_qc.mul_(VAR_DECAY).add_(ic * ic,
                                               alpha=1 - VAR_DECAY)
                if cfg.factor_r:
                    muh = (self.f_mu / self.mu_n if self.mu_n > 0
                           else self.f_mu)
                    sc = sel - muh
                    u = torch.einsum("jer,je->jr", self.V, sc)
                    if cfg.gain_mix:
                        wg = (self.f_m.double()
                              + MIX_EPS).pow(-MIX_POW)
                        wg = (wg / wg.sum(dim=0,
                                          keepdim=True)).float()
                        a1g = torch.einsum("gj,g->j", wg,
                                           self.a1)[:, None]
                        a2g = torch.einsum("gj,g->j", wg,
                                           self.a2)[:, None]
                    else:
                        gidx = self.f_m.argmin(dim=0)
                        a1g = self.a1[gidx][:, None]
                        a2g = self.a2[gidx][:, None]
                    fu = self.u_lv + HOLT_BETA * self.u_tr
                    iu = u - fu
                    self.u_lv.copy_(fu + a1g * iu)
                    self.u_tr.mul_(HOLT_BETA).add_(a2g * iu)
                    # Oja step on the PRE-update basis (the one the
                    # factor filter and predict-time features used)
                    uu = torch.einsum("jr,js->jrs", u, u)
                    self.V.add_(OJA_ETA * (
                        torch.einsum("je,jr->jer", sc, u)
                        - torch.einsum("jes,jsr->jer", self.V, uu)))
                    self._qr_since += 1
                    if self._qr_since >= QR_EVERY:
                        self.V.copy_(torch.linalg.qr(self.V)[0])
                        self._qr_since = 0
            if cfg.factor_r:
                self.f_mu.mul_(MU_DECAY).add_(sel,
                                              alpha=1 - MU_DECAY)
                self.mu_n = self.mu_n * MU_DECAY + (1 - MU_DECAY)
            if self.f_seq_n < WARMUP_POS:
                self.f_tr.zero_()
                self.c_tr.zero_()
                if cfg.factor_r:
                    self.u_tr.zero_()
            self.f_seq_n += 1
        self._since += 1
        if self._since >= RLS_REFRESH:
            eye = torch.eye(self.F + 1, device=self.dev,
                            dtype=torch.float64) * RLS_LAMBDA
            self.wr = torch.linalg.solve(
                self.A + eye, self.bvec.unsqueeze(2)).squeeze(2)
            if cfg.open_ch:
                self.wo = torch.linalg.solve(
                    self.Ao + eye, self.bo.unsqueeze(2)).squeeze(2)
            if cfg.horizons or cfg.manifest_head:
                A_heads = self.A_sub if self.head_mask is not None \
                    else self.Ao
            for h in cfg.horizons:
                self.wrec_h[h] = torch.linalg.solve(
                    A_heads + eye,
                    self.brec_h[h].unsqueeze(2)).squeeze(2)
            if cfg.manifest_head:
                self.wman = torch.linalg.solve(
                    A_heads + eye, self.bman.unsqueeze(2)).squeeze(2)
            if cfg.novel_head:
                self.w_nov = torch.linalg.solve(
                    self.A_nov + eye,
                    self.b_nov.unsqueeze(2)).squeeze(2)
            if cfg.soft_head:
                self.wsoft = torch.linalg.solve(
                    self.Adr + eye, self.bsoft.unsqueeze(2)).squeeze(2)
            if cfg.band_heads:
                self.wd = torch.linalg.solve(
                    self.Ad + eye, self.bd.unsqueeze(2)).squeeze(2)
                self.wdb = torch.linalg.solve(
                    self.Adb + eye, self.bdb.unsqueeze(2)).squeeze(2)
                self.wbn = torch.linalg.solve(
                    self.Abn + eye, self.bbn.unsqueeze(2)).squeeze(2)
            self._since = 0
        # pair tables
        if "xsame" in cfg.features and "xsame" not in cfg.ablate:
            # within-position pairs (top8(t, j-Δ) -> top8(t, j)),
            # added AFTER this position was scored — the table seen
            # by features() at t reflects positions <= t-1 only
            d = cfg.xsame_lag
            self._pair_update(self.xsame, "xsame_scale",
                              tops[:-d], tops[d:], d, None)
        if self.prev8 is not None:
            self._pair_update(self.trans, "trans_scale", self.prev8,
                              tops)
            if "xprev_up" in cfg.features \
                    and "xprev_up" not in cfg.ablate:
                self._pair_update(self.xu, "xu_scale",
                                  self.prev8[1:], tops[:-1], 0, J - 1)
            if "xprev_dn" in cfg.features \
                    and "xprev_dn" not in cfg.ablate:
                self._pair_update(self.xd, "xd_scale",
                                  self.prev8[:-1], tops[1:], 1, None)
        if self.prev2_8 is not None and "trans2" in cfg.features:
            self._pair_update(self.trans2, "trans2_scale",
                              self.prev2_8, tops)
        # trailing windows
        for ring, counts, hor in ((self.ring16, self.t16, 16),
                                  (self.ring64, self.t64, 64)):
            if len(ring) >= hor:
                old = ring.pop(0)
                counts.scatter_add_(1, old, -ones)
            ring.append(tops)
            counts.scatter_add_(1, tops, ones)
        # EMAs, prev, scores
        self.ema2.mul_(self.d2).add_(sel, alpha=1 - self.d2)
        self.ema8.mul_(self.d8).add_(sel, alpha=1 - self.d8)
        self.n2 = self.n2 * self.d2 + (1 - self.d2)
        self.n8 = self.n8 * self.d8 + (1 - self.d8)
        # gate-weight dynamics
        if hasattr(self, "w_prev"):
            self.w_prev.zero_().scatter_(1, tops, top_w)
            w_now = torch.zeros(J, E, device=self.dev)
            w_now.scatter_(1, tops, top_w)
            self.w_ema.mul_(self.d_w).add_(w_now, alpha=1 - self.d_w)
            self.n_w = self.n_w * self.d_w + (1 - self.d_w)
        # global frequency (cross-sequence, lazy-decayed)
        if "gfreq" in cfg.features:
            self.gfreq_scale *= GFREQ_DECAY
            inc = torch.full_like(ones, 1.0 / self.gfreq_scale)
            self.gfreq.scatter_add_(1, tops, inc)
            if self.gfreq_scale < 1e-20:
                self.gfreq.mul_(self.gfreq_scale)
                self.gfreq_scale = 1.0
        # embedding-bucket memo (cross-sequence counts)
        if "memo" in cfg.features and bucket >= 0:
            self.memo[:, bucket, :].scatter_add_(1, tops, ones)
        # hashed bigram memo (x16, cross-sequence counts)
        if "memo2" in cfg.features and bucket2 >= 0:
            self.memo2[:, bucket2, :].scatter_add_(1, tops, ones)
        self.prev2_8 = self.prev8
        self.prev8 = tops
        self.prev8_mask.zero_().scatter_(1, tops, 1.0)
        self.g_prev2.copy_(self.g_prev)
        self.g_prev.copy_(sel)
        self.cut_prev.copy_(torch.topk(sel, K, dim=1).values[:, -1])
        self.last_seen.scatter_(
            1, tops, torch.full_like(tops, self.t_idx))
        self.t_idx += 1
        if cfg.horizons:
            # matured windows: position t0's recur-h label is decided
            # once h further positions have been revealed. One shared
            # ring; the entry with t0 = p - h sits at index -(h+1)
            # (this position's entry was appended above at index -1).
            p = self.t_idx - 1
            for h in cfg.horizons:
                if len(self.rring) > h:
                    Xa0, t0 = self.rring[-(h + 1)]
                    assert t0 == p - h
                    y = (self.last_seen > t0).double()
                    Xa0d = Xa0.double()
                    if self.head_mask is not None:
                        Xa0d = Xa0d * self.head_mask
                    self.brec_h[h].add_(
                        torch.einsum("jef,je->jf", Xa0d, y))
            max_h = cfg.horizons[-1]
            while len(self.rring) > max_h:
                self.rring.pop(0)

    @property
    def wrec(self):
        """h=16 recur head (certified evictor scorer / export alias)."""
        return self.wrec_h[16]

    # ── π̂ calibration head (ARCHITECT_CALIBRATION.md) ──────────────────

    def cal_curve(self, m_budgets: list[int]) -> torch.Tensor:
        """Coverage curves Ĉ_j(M) [len(Ms), J]: expected number of
        the K true experts covered by the pool prev8 ∪ top-(M-K).
        Bias-corrected; cold start (n_cal = 0) returns zero coverage
        (maximum risk — the safe direction for a governor)."""
        if self.cal_n <= 0:
            return torch.zeros(len(m_budgets), self.J,
                               device=self.dev)
        base = (self.calP / self.cal_n).sum(dim=1)          # [J]
        cum = torch.cumsum(self.calH / self.cal_n, dim=1)   # [J,E-K]
        rows = []
        for m in m_budgets:
            k = max(0, min(int(m) - self.K, self.E - self.K))
            rows.append(base + (cum[:, k - 1] if k > 0
                                else torch.zeros_like(base)))
        return torch.stack(rows)

    def cal_update(self, sc: torch.Tensor, in_prev: torch.Tensor,
                   yb: torch.Tensor):
        """Decayed rank-conditional reliability update. sc [J,E] =
        w_open scores; in_prev [J,E] bool = top8(t-1); yb [J,E] bool
        = revealed truth. Returns (hits_open [J,E-K], hits_prev
        [J,K]) f32 realized hits in rank order (certification).
        Callers read cal_curve BEFORE this (predict-then-update)."""
        neg = torch.finfo(torch.float32).min
        so = torch.where(~in_prev, sc, torch.full_like(sc, neg))
        oo = torch.argsort(so, dim=1, descending=True, stable=True)
        hits_o = yb.gather(1, oo[:, : self.E - self.K]).float()
        sp = torch.where(in_prev, sc, torch.full_like(sc, neg))
        op = torch.argsort(sp, dim=1, descending=True, stable=True)
        hits_p = yb.gather(1, op[:, : self.K]).float()
        self.calH.mul_(CAL_DECAY).add_(hits_o, alpha=1 - CAL_DECAY)
        self.calP.mul_(CAL_DECAY).add_(hits_p, alpha=1 - CAL_DECAY)
        self.cal_n = self.cal_n * CAL_DECAY + (1 - CAL_DECAY)
        return hits_o, hits_p

    # ── pihat2d adjustment (ARCHITECT_CALIBRATION §7, fd3ce3a5) ────────

    def cal2d_buckets(self, sc: torch.Tensor, in_prev: torch.Tensor):
        """Rank the position and bucket each rank's score-gap-to-the-
        rank-(M-K)-cut by the PER-LAYER decayed gap quantiles (pre-
        update histogram — predict-then-update). Returns (b_all [J,E]
        bucket ids in rank order: open 0..E-K-1 then prev slots,
        micro [J,E] micro-bin ids for the histogram update)."""
        neg = torch.finfo(torch.float32).min
        so = torch.where(~in_prev, sc, torch.full_like(sc, neg))
        oo = torch.argsort(so, dim=1, descending=True, stable=True)
        s_open = so.gather(1, oo)[:, : self.E - self.K]
        cut_r = min(CAL2D_POOL_M - self.K, self.E - self.K)
        cut = s_open[:, cut_r - 1]
        sp_ = torch.where(in_prev, sc, torch.full_like(sc, neg))
        op = torch.argsort(sp_, dim=1, descending=True, stable=True)
        s_prev = sp_.gather(1, op[:, : self.K])
        gaps = torch.cat([s_open, s_prev], dim=1) - cut[:, None]
        x = torch.tanh(gaps)
        micro = ((x + 1.0) * 0.5 * (CAL2D_MICRO - 1)).round() \
            .long().clamp(0, CAL2D_MICRO - 1)
        tot = self.cal_ghist.sum(dim=1, keepdim=True)
        if float(tot.sum()) > 0:
            cdf = torch.cumsum(self.cal_ghist, dim=1) \
                / tot.clamp(min=1e-30)
            qlev = (torch.arange(1, CAL2D_BUCKETS, device=self.dev)
                    .float() / CAL2D_BUCKETS)
            cx = cdf.gather(1, micro)
            b_all = (cx[:, :, None] > qlev).sum(dim=2).long()
        else:
            b_all = torch.zeros_like(micro)
        return b_all, micro

    def cal2d_mhat(self, b_all: torch.Tensor) -> torch.Tensor:
        """Position-conditional M̂(32) from the 2-D tables (pre-
        update). Laplace shrinkage toward the certified 1-D rate:
        p2 = (H2 + a*p1)/(N2 + a) — empty bins reduce to 1-D
        exactly, so the marginal surface cannot degrade."""
        H2 = self.calH2.gather(2, b_all[:, :, None]).squeeze(2)
        N2 = self.calN2.gather(2, b_all[:, :, None]).squeeze(2)
        if self.cal_n > 0:
            p1 = torch.cat([self.calH, self.calP], dim=1) \
                / self.cal_n
        else:
            p1 = torch.zeros(self.J, self.E, device=self.dev)
        p2 = (H2 + CAL2D_ALPHA * p1) / (N2 + CAL2D_ALPHA)
        cut_r = min(CAL2D_POOL_M - self.K, self.E - self.K)
        pool_p = (p2[:, : cut_r].sum(dim=1)
                  + p2[:, self.E - self.K:].sum(dim=1))
        return (self.K - pool_p).sum()

    def cal2d_update(self, hits_o: torch.Tensor,
                     hits_p: torch.Tensor, b_all: torch.Tensor,
                     micro: torch.Tensor) -> None:
        """Scatter this position's realized hits into the (rank,
        gap-bucket) cells (raw-decayed effective counts) and the gap
        micro-histogram. Called AFTER cal2d_buckets/cal2d_mhat."""
        E, B = self.E, CAL2D_BUCKETS
        hits = torch.cat([hits_o, hits_p], dim=1)      # rank order
        flat = (self.jj[:, None] * E * B
                + torch.arange(E, device=self.dev)[None, :] * B
                + b_all).reshape(-1)
        self.calH2.mul_(CAL_DECAY)
        self.calH2.view(-1).scatter_add_(0, flat, hits.reshape(-1))
        self.calN2.mul_(CAL_DECAY)
        self.calN2.view(-1).scatter_add_(
            0, flat, torch.ones_like(flat, dtype=torch.float32))
        gflat = (self.jj[:, None] * CAL2D_MICRO + micro).reshape(-1)
        self.cal_ghist.mul_(CAL_DECAY)
        self.cal_ghist.view(-1).scatter_add_(
            0, gflat, torch.ones_like(gflat, dtype=torch.float32))

    def memo_maxpool(self, buckets: list[int]) -> torch.Tensor:
        """Chunk-content memo score: max over the chunk's token buckets
        of the normalized per-bucket expert distribution. [J, E]."""
        if not buckets:
            return self._zeros_je
        rows = self.memo[:, sorted(set(buckets)), :]        # [J, nb, E]
        rows = rows / (1.0 + rows.sum(dim=2, keepdim=True))
        return rows.max(dim=1).values


# ── fp8 state round-trip (deployment-precision simulation) ───────────────────


def fp8_roundtrip(x: torch.Tensor, dims: tuple) -> tuple[torch.Tensor,
                                                         torch.Tensor]:
    """Quantize x to float8_e4m3 with amax scales over `dims` (the
    reduced dims share one scale) and dequantize back. Returns
    (round-tripped f32 tensor, scales). amax scaling => no clipping at
    e4m3's 448 max by construction; precision cost is the ~3-mantissa-
    bit relative error this study measures."""
    amax = x.abs().amax(dim=dims, keepdim=True).clamp(min=1e-30)
    scale = amax / 448.0
    q = (x / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale
    return q, scale.squeeze()


def prune_state(st: "CPrimeState", cfg: ExpCfg) -> dict:
    """x20 Tier-2 table pruning IN PLACE (warm->held boundary).
    memo/memo2: per-(j,bucket) rows — zero rows with decayed support
    < s, keep top-k experts per surviving row (bucket-chunked; the
    identity-lut memo is ~12 GB). Pair tables (trans/xu/xd/xsame):
    zero entries < tau*row_max per (j,p) row (scale-invariant under
    the lazy decay scale). Returns retained-mass/nnz accounting."""
    info: dict = {}

    def _stamp(name: str, s: float, k: int):
        t = getattr(st, name)                      # [J, B, E]
        B = t.shape[1]
        tot = kept = 0.0
        nz = 0
        for lo in range(0, B, 8192):
            sl = t[:, lo:lo + 8192, :]
            tot += float(sl.sum())
            if s > 0:
                sl[sl.sum(dim=2) < s] = 0.0
            if k and k < sl.shape[2]:
                thr = torch.topk(sl, k, dim=2).values[:, :, -1:]
                sl[sl < thr] = 0.0
                # ties at the threshold could keep > k: cap exactly
                # is unnecessary — ties in decayed counts are rare
                # and keep-mass accounting is what ships
            kept += float(sl.sum())
            nz += int((sl > 0).sum())
        info[name] = {"s": s, "k": k,
                      "mass_kept": kept / max(tot, 1e-30),
                      "nnz": nz}

    if cfg.prune_memo_s or cfg.prune_memo_k:
        _stamp("memo", cfg.prune_memo_s, cfg.prune_memo_k)
    if cfg.prune_memo2_s or cfg.prune_memo2_k:
        _stamp("memo2", cfg.prune_memo2_s, cfg.prune_memo2_k)
    if cfg.prune_tau > 0:
        for name in ("trans", "xu", "xd", "xsame"):
            if not hasattr(st, name):
                continue
            t = getattr(st, name)
            rmax = t.amax(dim=2, keepdim=True)
            m = t < (cfg.prune_tau * rmax)
            tot = float(t.sum())
            t[m] = 0.0
            info[name] = {"tau": cfg.prune_tau,
                          "mass_kept": float(t.sum())
                          / max(tot, 1e-30),
                          "nnz": int((t > 0).sum()),
                          "nnz_frac": float((t > 0).float().mean())}
    return info


def quantize_state_fp8(st: "CPrimeState", ctxs=None) -> dict:
    """Round-trip the PERSISTENT (shipped) state through fp8-e4m3:
    pair tables (stationary units, per-LAYER scales), memo counts
    (per-layer scales), ridge weights (per-layer-ROW scales — feature
    columns span decades, rows share a regime). Online A/b sufficient
    statistics continue in f32 (deployment re-solves locally)."""
    info: dict = {"granularity": {}}

    def _table(name: str, scale_attr: str | None = None):
        t = getattr(st, name)
        if scale_attr is not None:
            t = t * getattr(st, scale_attr)
            setattr(st, scale_attr, 1.0)
        q, s = fp8_roundtrip(t, dims=(1, 2))
        getattr(st, name).copy_(q)
        info["granularity"][name] = "per-layer [J] amax"
        info[f"{name}_amax_max"] = float(s.max() * 448.0)

    if "xsame" in st.cfg.features and "xsame" not in st.cfg.ablate:
        _table("xsame", "xsame_scale")
    if not {"trans", "trans_rank"} <= set(st.cfg.ablate):
        _table("trans", "trans_scale")
    if "xprev_up" in st.cfg.features \
            and "xprev_up" not in st.cfg.ablate:
        _table("xu", "xu_scale")
    if "xprev_dn" in st.cfg.features \
            and "xprev_dn" not in st.cfg.ablate:
        _table("xd", "xd_scale")
    if "memo" in st.cfg.features:
        # [J, B, E] counts (non-negative): per-layer amax over (B, E),
        # round-tripped IN PLACE in bucket chunks — the identity-lut
        # table is ~12 GB and a whole-tensor temp OOMs a 32 GB card.
        amax = st.memo.amax(dim=(1, 2), keepdim=True).clamp(min=1e-30)
        scale = amax / 448.0
        B = st.memo.shape[1]
        for lo in range(0, B, 8192):
            sl = st.memo[:, lo:lo + 8192, :]
            sl.copy_((sl / scale).to(torch.float8_e4m3fn)
                     .to(torch.float32) * scale)
        info["granularity"]["memo"] = "per-layer [J] amax (chunked)"
        info["memo_amax_max"] = float(amax.max())
    if "memo2" in st.cfg.features:
        amax = st.memo2.amax(dim=(1, 2), keepdim=True).clamp(min=1e-30)
        scale = amax / 448.0
        B = st.memo2.shape[1]
        for lo in range(0, B, 8192):
            sl = st.memo2[:, lo:lo + 8192, :]
            sl.copy_((sl / scale).to(torch.float8_e4m3fn)
                     .to(torch.float32) * scale)
        info["granularity"]["memo2"] = "per-layer [J] amax (chunked)"
        info["memo2_amax_max"] = float(amax.max())
    wnames = ("wr", "wo") if st.cfg.open_ch else ("wr",)
    if st.cfg.soft_head:
        wnames = wnames + ("wsoft",)
    if st.cfg.band_heads:
        wnames = wnames + ("wd", "wdb", "wbn")
    for wname in wnames:
        w = getattr(st, wname).float()
        q, _ = fp8_roundtrip(w, dims=(1,))
        setattr(st, wname, q.double())
        info["granularity"][wname] = "per-layer-row [J] amax"
    # x13 persistent filter priors (per-seq level/trend state is NOT
    # shipped; scalar-class rows — c_qc/f_m/sig_drift — stay f32 like
    # the online A/b statistics)
    if st.cfg.filter_feats:
        q, _ = fp8_roundtrip(st.f_qv, dims=(2,))
        st.f_qv.copy_(q)
        info["granularity"]["f_qv"] = "per-(gain,layer) amax"
    if st.cfg.factor_r:
        for nm, dims in (("V", (1, 2)), ("f_mu", (1,))):
            t = getattr(st, nm)
            q, _ = fp8_roundtrip(t, dims=dims)
            t.copy_(q)
            info["granularity"][nm] = "per-layer amax"
    # shared-A heads (per-layer-row like the ridge weights)
    for h in st.cfg.horizons:
        q, _ = fp8_roundtrip(st.wrec_h[h].float(), dims=(1,))
        st.wrec_h[h] = q.double()
        info["granularity"][f"wrec{h}"] = "per-layer-row [J] amax"
    if st.cfg.manifest_head:
        q, _ = fp8_roundtrip(st.wman.float(), dims=(1,))
        st.wman = q.double()
        info["granularity"]["wman"] = "per-layer-row [J] amax"
    if st.cfg.novel_head:
        q, _ = fp8_roundtrip(st.w_nov.float(), dims=(1,))
        st.w_nov = q.double()
        info["granularity"]["w_nov"] = "per-layer-row [J] amax"
    # ctx encoder: learned params + frozen PCA features (per-tensor)
    if ctxs is not None:
        with torch.no_grad():
            for pn, p in ctxs.model.named_parameters():
                q, _ = fp8_roundtrip(p.data, dims=tuple(
                    range(p.dim())))
                p.data.copy_(q)
            q, _ = fp8_roundtrip(
                ctxs.emb, dims=tuple(range(ctxs.emb.dim())))
            ctxs.emb.copy_(q)
        info["granularity"]["ctx_params"] = "per-tensor amax"
        info["granularity"]["ctx_pca"] = "per-tensor amax"
    return info


# ── state export (deployment packaging) ──────────────────────────────────────


def memo_sparse_topk(memo: "torch.Tensor", k: int = 16,
                     chunk: int = 8192):
    """Sparse top-k stamp of the memo table [J, B, E]: buckets with any
    counts -> (seen [S] i64, ids [S, J, k] u8 expert ids, counts
    [S, J, k] u16 integer counts; ids beyond a bucket-layer's support
    are padded with count 0). Chunked over buckets (the identity-lut
    table is ~12 GB)."""
    J, B, E = memo.shape
    if E > 256 or k > 256:
        raise ValueError("u8 expert ids require E <= 256")
    k = min(k, E)
    seen_chunks = []
    for lo in range(0, B, chunk):
        m = memo[:, lo:lo + chunk, :]
        nz = m.abs().sum(dim=(0, 2)) > 0
        seen_chunks.append(torch.nonzero(nz).squeeze(1) + lo)
    seen = torch.cat(seen_chunks)
    S = len(seen)
    ids = np.empty((S, J, k), np.uint8)
    cnt = np.empty((S, J, k), np.uint16)
    for lo in range(0, S, chunk):
        sl = seen[lo:lo + chunk]
        rows = memo[:, sl, :]                       # [J, s, E]
        v, i = torch.topk(rows, k, dim=2)           # [J, s, k]
        ids[lo:lo + chunk] = i.permute(1, 0, 2).cpu().numpy() \
            .astype(np.uint8)
        cnt[lo:lo + chunk] = v.permute(1, 0, 2).cpu().numpy() \
            .clip(0, 65535).astype(np.uint16)
    return seen.cpu().numpy().astype(np.int64), ids, cnt


def export_state(st: CPrimeState, export_dir: Path,
                 ctxs=None) -> dict:
    """Deployment-warm state snapshot at the end of the held pass:
    ridge weights (valid + open channels), pair tables in stationary
    decayed-count units (scale baked), sparse memo stamp; the trained
    ctxformer encoder (state_dict tensors) when present. Per-sequence
    state (b0_prev/trails/EMAs/g_prev) starts empty per request and is
    NOT exported."""
    ed = Path(export_dir)
    ed.mkdir(parents=True, exist_ok=True)
    cfg = st.cfg
    if ctxs is not None:
        sd = {k: v.detach().cpu().numpy()
              for k, v in ctxs.model.state_dict().items()}
        np.savez(ed / "ctx_encoder.npz", **sd)
    files: dict = {}
    ridge = {"w_valid": st.wr.cpu().numpy(),
             "features": np.array(list(st.feat_names) + ["bias"])}
    if cfg.open_ch:
        ridge["w_open"] = st.wo.cpu().numpy()
    for h in cfg.horizons:
        key = "w_recur" if h == 16 else f"w_recur{h}"
        ridge[key] = st.wrec_h[h].cpu().numpy()
    if cfg.manifest_head:
        ridge["w_manifest"] = st.wman.cpu().numpy()
    if cfg.novel_head:
        ridge["w_novel"] = st.w_nov.cpu().numpy()
    if cfg.soft_head:
        ridge["w_soft"] = st.wsoft.cpu().numpy()
    if cfg.band_heads:
        ridge["w_dr"] = st.wd.cpu().numpy()
        ridge["w_drband"] = st.wdb.cpu().numpy()
        ridge["w_band"] = st.wbn.cpu().numpy()
    np.savez(ed / "ridge.npz", **ridge)
    files["ridge"] = "ridge.npz"
    if not {"trans", "trans_rank"} <= set(cfg.ablate):
        np.savez_compressed(
            ed / "trans.npz",
            trans=(st.trans * st.trans_scale).cpu().numpy()
            .astype(np.float32))
        files["trans"] = "trans.npz"
    if "xsame" in cfg.features and "xsame" not in cfg.ablate:
        np.savez_compressed(
            ed / "xsame.npz",
            xsame=(st.xsame * st.xsame_scale).cpu().numpy()
            .astype(np.float32), lag=np.int64(cfg.xsame_lag))
        files["xsame"] = "xsame.npz"
    _xu = "xprev_up" in cfg.features and "xprev_up" not in cfg.ablate
    _xd = "xprev_dn" in cfg.features and "xprev_dn" not in cfg.ablate
    if _xu or _xd:
        xp = {}
        if _xu:
            xp["xu"] = (st.xu * st.xu_scale).cpu().numpy() \
                .astype(np.float32)
        if _xd:
            xp["xd"] = (st.xd * st.xd_scale).cpu().numpy() \
                .astype(np.float32)
        np.savez_compressed(ed / "xprev.npz", **xp)
        files["xprev"] = "xprev.npz"
    if "memo" in cfg.features:
        seen, ids, cnt = memo_sparse_topk(st.memo)
        np.savez_compressed(ed / "memo_sparse.npz", seen=seen,
                            top_ids=ids, top_counts=cnt,
                            buckets=cfg.memo_buckets, topk=16)
        files["memo_sparse"] = "memo_sparse.npz"
        files["memo_seen_tokens"] = int(len(seen))
    if "memo2" in cfg.features:
        # pre-registered deployed form (Part B.6): per-(layer,bucket)
        # rows with support < 8 dropped, k = 8 stamp. Row filter is
        # applied IN PLACE — export runs after all evaluation.
        rowsum = st.memo2.sum(dim=2)                     # [J, B2]
        st.memo2[rowsum < 8.0] = 0.0
        seen2, ids2, cnt2 = memo_sparse_topk(st.memo2, k=8)
        np.savez_compressed(ed / "memo2_sparse.npz", seen=seen2,
                            top_ids=ids2, top_counts=cnt2,
                            buckets=cfg.memo2_buckets, topk=8,
                            min_support=8)
        files["memo2_sparse"] = "memo2_sparse.npz"
        files["memo2_seen_buckets"] = int(len(seen2))
        files["memo2_stamp_bytes"] = int(
            len(seen2) * st.J * 8 * 3 + len(seen2) * 8)
    if cfg.cal_head:
        # π̂ tables, bias-corrected stationary form (f32 — 77 KB,
        # not worth fp8 scale blocks; ARCHITECT_CALIBRATION.md §4)
        n = max(st.cal_n, 1e-12)
        cal2d_extra = ({
            # pihat2d adjustment tables (raw-decayed effective
            # counts + the per-layer gap micro-histogram; consumer
            # applies p2 = (H2 + a*p1)/(N2 + a))
            "H2": st.calH2.cpu().numpy(),
            "N2": st.calN2.cpu().numpy(),
            "ghist": st.cal_ghist.cpu().numpy(),
            "cal2d_alpha": np.float64(CAL2D_ALPHA),
            "cal2d_buckets": np.int64(CAL2D_BUCKETS),
            "cal2d_pool_m": np.int64(CAL2D_POOL_M),
        } if cfg.cal2d else {})
        np.savez(ed / "calibration.npz",
                 H=(st.calH / n).cpu().numpy(),
                 P=(st.calP / n).cpu().numpy(),
                 decay=np.float64(CAL_DECAY),
                 n_cal=np.float64(st.cal_n), **cal2d_extra)
        files["calibration"] = "calibration.npz"
    if cfg.soft_head or cfg.filter_feats:
        filt: dict = {}
        if cfg.soft_head:
            filt["sig_drift"] = st.sig_drift.cpu().numpy()
            filt["sig_n"] = np.float64(st.sig_n)
        if cfg.filter_feats:
            filt["qv"] = st.f_qv.cpu().numpy()
            filt["gain_mse"] = st.f_m.cpu().numpy()
            filt["cut_qc"] = st.c_qc.cpu().numpy()
            filt["gains"] = np.asarray(HOLT_GAINS, np.float32)
        np.savez(ed / "filters.npz", **filt)
        files["filters"] = "filters.npz"
    if cfg.factor_r:
        np.savez(ed / "factor.npz", V=st.V.cpu().numpy(),
                 mu=st.f_mu.cpu().numpy(), mu_n=np.float64(st.mu_n))
        files["factor"] = "factor.npz"
    return files


# ── eval pass ────────────────────────────────────────────────────────────────


def gpu_eval_seqs(work: Path, c: stage0.Corpus, sigma, bias, held_keys,
                  deep_lo: int, deep_hi: int, warm_keys, cfg: ExpCfg,
                  device: str = "cuda:0", tap_provider=None,
                  lut: np.ndarray | None = None,
                  probe_packed: np.ndarray | None = None,
                  export_dir: Path | None = None,
                  evict_keys: set[int] | None = None,
                  attn_lut: np.ndarray | None = None,
                  ctx_emb: np.ndarray | None = None) -> dict:
    dev = torch.device(device)
    if probe_packed is None:
        probe_packed = np.load(work / "probe_full.npy")
    st = CPrimeState(c.J, c.E, c.K, dev, probe_packed, sigma, cfg)
    use_tap = "tap" in cfg.features
    use_memo = "memo" in cfg.features
    use_memo2 = "memo2" in cfg.features
    if use_tap and tap_provider is None:
        raise ValueError("cfg enables 'tap' but no tap_provider given")
    if use_memo and lut is None:
        raise ValueError("cfg enables 'memo' but no bucket lut given")
    evict_keys = evict_keys or set()
    if evict_keys and not cfg.open_ch:
        raise ValueError("evict dump scores with w_open — open_ch "
                         "experiments only")
    attn = None
    if cfg.attn_mode:
        from .stagecprime_attn import AttnState
        if attn_lut is None:
            raise ValueError("cfg.attn_mode set but no attn_lut given")
        if not cfg.open_ch:
            raise ValueError("attn challenger rides the open ridge — "
                             "open_ch required")
        attn = AttnState(c.J, c.E, int(attn_lut.max()) + 1,
                         cfg.attn_mode, dev,
                         zero_gate=cfg.attn_zero_gate)
        print(f"[stagec'] attn challenger {cfg.attn_mode}: "
              f"{attn.model.n_params():,} params "
              f"(zero_gate={cfg.attn_zero_gate}, "
              f"hedge={cfg.attn_hedge})", flush=True)
    ctxs = None
    if "ctx" in cfg.features:
        from .stagecprime_attn import CtxState
        if ctx_emb is None:
            raise ValueError("cfg enables 'ctx' but no ctx_emb given")
        ctx_kw = {}
        if cfg.ctx_d:
            ctx_kw["d"] = cfg.ctx_d
        if cfg.ctx_d_out:
            ctx_kw["d_out"] = cfg.ctx_d_out
        ctxs = CtxState(c.J, c.E, torch.as_tensor(ctx_emb), dev,
                        **ctx_kw)
        print(f"[stagec'] ctxformer: {ctxs.model.n_params():,} "
              f"learned params + frozen [{ctx_emb.shape[0]}, "
              f"{ctx_emb.shape[1]}] token features", flush=True)
    gamma_traj: list = []
    _dump_keys = ("seq", "pos", "layer", "expert", "prob",
                  "label", "occ_prev", "nextdist")
    dumps: dict[str, dict] = {nm: {k: [] for k in _dump_keys}
                              for nm in cfg.eff_evict_scorers}
    dump = dumps[cfg.eff_evict_scorers[0]]   # legacy single-scorer ref
    deep_np = np.array([i for i, l in enumerate(c.moe_layers)
                        if deep_lo <= l <= deep_hi])
    deep = torch.as_tensor(deep_np, device=dev)
    bias_t = torch.as_tensor(bias, device=dev)
    sig_t = st.sigma
    n_ks = len(POOL_KS)
    ks = torch.as_tensor(POOL_KS, device=dev)
    variants = cfg.variant_names
    members = cfg.member_names
    m_idx = {m: i for i, m in enumerate(members)}
    n_var = len(variants)
    n_mem = len(members)

    pool_hit = torch.zeros(n_var, n_ks, device=dev, dtype=torch.int64)
    addr_hit = torch.zeros(n_var, n_ks, device=dev, dtype=torch.int64)
    novel_hit = torch.zeros(n_var, n_ks, device=dev, dtype=torch.int64)
    pool_true = 0
    addr_n = torch.zeros((), device=dev, dtype=torch.int64)
    novel_n = torch.zeros((), device=dev, dtype=torch.int64)
    # x16 support diagnostic: pre-update bigram-row count mass at
    # held positions — the fires-but-sparse gate input
    m2_n = torch.zeros((), device=dev, dtype=torch.int64)
    m2_sum = torch.zeros((), device=dev, dtype=torch.float64)
    m2_lt8 = torch.zeros((), device=dev, dtype=torch.int64)
    # π̂ certification accumulators (held; 4 seq-quartile banks for
    # the drift gate); columns 0..E-K-1 = open ranks, E-K.. = prev
    # slots. Per-position M̂/M kept as 0-dim tensors (no syncs).
    if cfg.cal_head:
        cal_pred = torch.zeros(4, c.J, c.E, device=dev,
                               dtype=torch.float64)
        cal_real = torch.zeros(4, c.J, c.E, device=dev,
                               dtype=torch.float64)
        cal_bn = torch.zeros(4, device=dev, dtype=torch.float64)
        cal_mhat: list = []
        cal_mreal: list = []
        cal_t1: list = []       # §7.2 pool-boundary gap (geometry)
        cal_t2: list = []       # §7.2 in-pool spread
        cal_mhat2: list = []    # pihat2d position-conditional M̂
    man = torch.zeros(len(MANIFEST_DS), 2, device=dev,
                      dtype=torch.float64)
    n_splits = len(cfg.novel_splits) if cfg.novel_head else 0
    split_pool = torch.zeros(max(1, n_splits), device=dev,
                             dtype=torch.int64)
    split_addr = torch.zeros(max(1, n_splits), device=dev,
                             dtype=torch.int64)
    split_novel = torch.zeros(max(1, n_splits), device=dev,
                              dtype=torch.int64)
    man_head = torch.zeros(len(MANIFEST_DS), 2, device=dev,
                           dtype=torch.float64)
    man_head_open = torch.zeros(len(MANIFEST_DS), 2, device=dev,
                                dtype=torch.float64)
    man_open = torch.zeros(len(MANIFEST_DS), 2, device=dev,
                           dtype=torch.float64)
    mf_acc = torch.zeros(2, 3, device=dev, dtype=torch.float64)
    collect = {"on": False}
    prev_u: dict[int, list] = {}
    neg_inf = torch.finfo(torch.float32).min

    def variant_tensors(raws, norms_v, norms_o, comb, comb_o, valid,
                        in_prev, chal=None):
        """([V,J,E] scores, [V,J,E] resid masks) in variant order."""
        rows, masks = [], []
        resid_v = valid & ~in_prev
        resid_o = ~in_prev
        for v in variants:
            if v == "attn_open":
                rows.append(chal)
                masks.append(resid_o)
            elif v == "combined":
                rows.append(comb)
                masks.append(resid_v)
            elif v == "combined_open":
                rows.append(comb_o)
                masks.append(resid_o)
            elif v == "rls_open":
                rows.append(norms_o[m_idx["rls_open"]])
                masks.append(resid_o)
            elif v in ("rls_soft", "rls_dr", "rls_drband",
                       "rls_band"):
                # open rank domain (must precede the endswith check
                # fall-through to the valid channel)
                rows.append(norms_o[m_idx[v]])
                masks.append(resid_o)
            elif v.endswith("_open"):
                rows.append(norms_o[m_idx[v[:-5]]])
                masks.append(resid_o)
            else:
                rows.append(norms_v[m_idx[v]])
                masks.append(resid_v)
        return torch.stack(rows), torch.stack(masks)

    def process_seq(key: int, s: dict, v_view) -> None:
        recs = c.lookup_records(s["run_idx"], s["seq_id"], s["pos"])
        tops_np = np.ascontiguousarray(v_view.top_ids[recs])
        logits_np = v_view.logits(recs)
        topw_np = np.ascontiguousarray(v_view.top_w[recs])
        tops_all = torch.as_tensor(tops_np, device=dev,
                                   dtype=torch.int64)
        topw_all = torch.as_tensor(topw_np, device=dev).float()
        lg = torch.as_tensor(logits_np, device=dev).float()
        sel_all = (0.5 * torch.tanh(0.5 * lg) + 0.5) + bias_t[None]
        conf_all = s["conf"]
        buckets_all = (lut[s["token"]].astype(np.int64) if use_memo
                       else None)
        big_all = (bigram_lut(np.asarray(s["token"]),
                              cfg.memo2_buckets)
                   if use_memo2 else None)
        attn_b_all = (attn_lut[s["token"]].astype(np.int64)
                      if attn is not None else None)
        tap_cell = tap_blk = None
        if use_tap:
            tc, tb = tap_provider.for_seq(key, s)
            tap_cell = torch.as_tensor(tc, device=dev)   # [T,J,E] f16
            tap_blk = torch.as_tensor(tb, device=dev)    # [nb,J,E] f16
        chunk = s["chunk"]
        st.reset_seq()
        if attn is not None:
            attn.reset_seq()
        if ctxs is not None:
            ctxs.reset_seq()
        is_evict = collect["on"] and key in evict_keys \
            and s["recur"] is not None
        prob_rows: dict[str, list] = {nm: [] for nm in
                                      cfg.eff_evict_scorers}
        prob_meta: list = []
        for ci, blk in enumerate(s["blocks"]):
            # ── manifest at chunk start (pre-chunk state) ──
            frecs = c.lookup_records(s["run_idx"], s["seq_id"],
                                     blk["full_pos"])
            ft = torch.as_tensor(
                np.ascontiguousarray(v_view.top_ids[frecs]),
                device=dev, dtype=torch.int64)[:, deep]
            ucur = torch.zeros(len(deep_np), c.E, device=dev,
                               dtype=torch.bool)
            ucur.scatter_(1, ft.permute(1, 0, 2)
                          .reshape(len(deep_np), -1), True)
            if cfg.manifest_head:
                # label the PREVIOUS chunk's buffered rows with THIS
                # chunk's full-layer true union (target: membership of
                # the NEXT verify chunk's union; causal — the union is
                # revealed exactly now)
                ftf = torch.as_tensor(
                    np.ascontiguousarray(v_view.top_ids[frecs]),
                    device=dev, dtype=torch.int64)     # [n, J, K]
                u_full = torch.zeros(c.J, c.E, device=dev,
                                     dtype=torch.bool)
                u_full.scatter_(1, ftf.permute(1, 0, 2)
                                .reshape(c.J, -1), True)
                if st.cbuf:
                    yman = u_full.double()
                    for Xb in st.cbuf:
                        Xab = torch.cat(
                            [Xb.double(),
                             torch.ones(c.J, c.E, 1, device=dev,
                                        dtype=torch.float64)], dim=2)
                        if st.head_mask is not None:
                            Xab = Xab * st.head_mask
                        st.bman.add_(torch.einsum(
                            "jef,je->jf", Xab, yman))
                    st.cbuf.clear()
            state = prev_u.setdefault(key, [None, None])
            u1, u2 = state[0], state[1]
            if collect["on"] and u1 is not None and st.t_idx >= 1:
                un = ucur.sum(dim=1)
                ok = un > 0
                unc = un.clamp(min=1).double()
                cov1 = ((u1 & ucur).sum(dim=1).double() / unc)[ok]
                u12 = (u1 | u2) if u2 is not None else u1
                cov2 = ((u12 & ucur).sum(dim=1).double() / unc)[ok]
                mf_acc[0, 0] += cov1.sum()
                mf_acc[0, 1] += u1[ok].sum()
                mf_acc[0, 2] += ok.sum()
                mf_acc[1, 0] += cov2.sum()
                mf_acc[1, 1] += u12[ok].sum()
                mf_acc[1, 2] += ok.sum()
                # chunk-content inputs (draft block known at manifest)
                tap_m = (tap_blk[ci].float() if use_tap else None)
                if tap_m is not None:
                    tap_m = (0.5 * torch.tanh(0.5 * tap_m) + 0.5) \
                        + bias_t
                memo_m = (st.memo_maxpool(
                    [int(b) for b in lut[blk["full_token"]]])
                    if use_memo else None)
                conf_m = float(np.nan_to_num(
                    blk["full_conf"]).mean()) if len(
                        blk["full_conf"]) else 0.0
                X, valid, raw = st.features(tap_now=tap_m,
                                            conf_now=conf_m,
                                            memo_now=memo_m)
                raw_m = st.member_scores(X, valid, raw)
                norm = _rank_norm_dense(raw_m, valid[None])
                comb = torch.einsum("jn,nje->je", st.hw.float(), norm)
                combd = comb[deep]
                validd = valid[deep]
                resid = validd & ~u1
                scm = torch.where(resid, combd,
                                  torch.full_like(combd, neg_inf))
                order = torch.argsort(scm, dim=1, descending=True,
                                      stable=True)
                for di, dsize in enumerate(MANIFEST_DS):
                    add = order[:, :dsize]
                    m = u1.clone()
                    m.scatter_(1, add, resid.gather(1, add))
                    cov = ((m & ucur).sum(dim=1).double() / unc)[ok]
                    man[di, 0] += cov.sum()
                    man[di, 1] += ok.sum()
                if cfg.manifest_head:
                    # per-consumer scoring: topup ranked by w_manifest
                    Xa_mn = torch.cat(
                        [X.double(),
                         torch.ones(c.J, c.E, 1, device=dev,
                                    dtype=torch.float64)], dim=2)
                    smn = torch.einsum("jef,jf->je", Xa_mn,
                                       st.wman).float()[deep]
                    for resid_m, acc in ((resid, man_head),
                                         (~u1, man_head_open)):
                        scn = torch.where(resid_m, smn,
                                          torch.full_like(smn,
                                                          neg_inf))
                        order_n = torch.argsort(scn, dim=1,
                                                descending=True,
                                                stable=True)
                        for di, dsize in enumerate(MANIFEST_DS):
                            add = order_n[:, :dsize]
                            m = u1.clone()
                            m.scatter_(1, add,
                                       resid_m.gather(1, add))
                            cov = ((m & ucur).sum(dim=1).double()
                                   / unc)[ok]
                            acc[di, 0] += cov.sum()
                            acc[di, 1] += ok.sum()
                if cfg.open_ch:
                    ones_m = torch.ones_like(valid)
                    norm_o = _rank_norm_dense(raw_m, ones_m[None])
                    comb_o = torch.einsum("jn,nje->je",
                                          st.hw.float(), norm_o)[deep]
                    resid_o = ~u1
                    scm_o = torch.where(resid_o, comb_o,
                                        torch.full_like(comb_o,
                                                        neg_inf))
                    order_o = torch.argsort(scm_o, dim=1,
                                            descending=True,
                                            stable=True)
                    for di, dsize in enumerate(MANIFEST_DS):
                        add = order_o[:, :dsize]
                        m = u1.clone()
                        m.scatter_(1, add, resid_o.gather(1, add))
                        cov = ((m & ucur).sum(dim=1).double() / unc)[ok]
                        man_open[di, 0] += cov.sum()
                        man_open[di, 1] += ok.sum()
            state[1] = state[0]
            state[0] = ucur
            # ── positions of this chunk ──
            for t in np.nonzero(chunk == ci)[0]:
                tops = tops_all[t]
                sel = sel_all[t]
                top_w = topw_all[t]
                bucket = int(buckets_all[t]) if use_memo else -1
                bucket2 = int(big_all[t]) if use_memo2 else -1
                tok_id = int(s["token"][t]) if ctxs is not None else -1
                conf_now = float(conf_all[t])
                tap_now = None
                if use_tap:
                    tap_now = (0.5 * torch.tanh(
                        0.5 * tap_cell[t].float()) + 0.5) + bias_t
                if st.t_idx >= 1:
                    ctx_now = (ctxs.predict(tok_id)
                               if ctxs is not None else None)
                    X, valid, raw = st.features(tap_now=tap_now,
                                                conf_now=conf_now,
                                                bucket=bucket,
                                                bucket2=bucket2,
                                                ctx_now=ctx_now,
                                                sel_now=sel,
                                                tops_now=tops)
                    raw_m = st.member_scores(X, valid, raw)
                    chal = None
                    if attn is not None and (cfg.attn_hedge
                                             or collect["on"]):
                        chal = attn.predict(
                            raw_m[m_idx["rls_open"]])
                        if cfg.attn_hedge:
                            raw_m[m_idx["attn"]] = chal
                    mmask = st.member_masks(valid)
                    norm = _rank_norm_dense(raw_m, valid[None])
                    comb = torch.einsum("jn,nje->je", st.hw.float(),
                                        norm)
                    norm_o = comb_o = None
                    if cfg.open_ch:
                        norm_o = _rank_norm_dense(
                            raw_m, torch.ones_like(valid)[None])
                        comb_o = torch.einsum("jn,nje->je",
                                              st.hw.float(), norm_o)
                    yb = torch.zeros(c.J, c.E, device=dev,
                                     dtype=torch.bool)
                    yb.scatter_(1, tops, True)
                    in_prev = st.prev8_mask > 0
                    cut9 = torch.topk(sel, c.K + 1,
                                      dim=1).values[:, -1]
                    bnd = yb & ((sel - cut9[:, None])
                                < sig_t[:, None])
                    addressable = yb & ~in_prev & ~bnd
                    novel = yb & ~valid & ~bnd
                    if collect["on"]:
                        nonlocal pool_true
                        pool_true += c.K * c.J
                        addr_n.add_(addressable.sum())
                        novel_n.add_(novel.sum())
                        if use_memo2:
                            sup = st.memo2[:, bucket2, :].sum(dim=1)
                            m2_n.add_(sup.numel())
                            m2_sum.add_(sup.sum().double())
                            m2_lt8.add_((sup < 8).sum())
                        var_sc, var_mask = variant_tensors(
                            raw_m, norm, norm_o, comb, comb_o,
                            valid, in_prev, chal=chal)
                        scm = torch.where(
                            var_mask, var_sc,
                            torch.full_like(var_sc, neg_inf))
                        order = torch.argsort(scm, dim=2,
                                              descending=True,
                                              stable=True)
                        om = var_mask.gather(2, order)
                        ybx = yb[None].expand(n_var, -1, -1) \
                            .gather(2, order) & om
                        adx = addressable[None].expand(n_var, -1, -1) \
                            .gather(2, order) & om
                        nvx = novel[None].expand(n_var, -1, -1) \
                            .gather(2, order) & om
                        cy = ybx.long().cumsum(dim=2)
                        ca = adx.long().cumsum(dim=2)
                        cn = nvx.long().cumsum(dim=2)
                        prev_hits = (yb & in_prev).sum()
                        pool_hit.add_(cy[:, :, ks - 1].sum(dim=1)
                                      + prev_hits)
                        addr_hit.add_(ca[:, :, ks - 1].sum(dim=1))
                        novel_hit.add_(cn[:, :, ks - 1].sum(dim=1))
                        if cfg.novel_head:
                            # split pools: prev8 ∪ top-k_in by w_open
                            # (in-union) ∪ top-k_out by w_novel
                            # (out-of-union)
                            Xa_v = torch.cat(
                                [X.double(),
                                 torch.ones(c.J, c.E, 1, device=dev,
                                            dtype=torch.float64)],
                                dim=2)
                            snov = torch.einsum(
                                "jef,jf->je", Xa_v, st.w_nov).float()
                            sopen = raw_m[m_idx["rls_open"]]
                            in_m = valid & ~in_prev
                            out_m = ~valid & ~in_prev
                            sin = torch.where(
                                in_m, sopen,
                                torch.full_like(sopen, neg_inf))
                            sout = torch.where(
                                out_m, snov,
                                torch.full_like(snov, neg_inf))
                            for si_, (ki, ko) in enumerate(
                                    cfg.novel_splits):
                                ii = torch.topk(sin, ki,
                                                dim=1).indices
                                oo = torch.topk(sout, ko,
                                                dim=1).indices
                                hi = (yb.gather(1, ii)
                                      & in_m.gather(1, ii)).sum()
                                ho = (yb.gather(1, oo)
                                      & out_m.gather(1, oo)).sum()
                                ai = (addressable.gather(1, ii)
                                      & in_m.gather(1, ii)).sum()
                                nvo = (novel.gather(1, oo)
                                       & out_m.gather(1, oo)).sum()
                                split_pool[si_] += prev_hits + hi + ho
                                split_addr[si_] += ai
                                split_novel[si_] += nvo
                        if is_evict:
                            # protection = prev8 ∪ top-48 of each
                            # evict scorer among valid non-prev
                            # candidates (stagec budget 8 + 48 = 56)
                            Xa_e = None
                            for sc_name in cfg.eff_evict_scorers:
                                if sc_name == "rls_open":
                                    esc = raw_m[m_idx["rls_open"]]
                                elif sc_name.startswith("recur"):
                                    h = int(sc_name[5:])
                                    if Xa_e is None:
                                        Xa_e = torch.cat(
                                            [X.double(),
                                             torch.ones(
                                                 c.J, c.E, 1,
                                                 device=dev,
                                                 dtype=torch
                                                 .float64)], dim=2)
                                    esc = torch.einsum(
                                        "jef,jf->je", Xa_e,
                                        st.wrec_h[h]).float()
                                else:
                                    raise ValueError(sc_name)
                                csc = torch.where(
                                    valid & ~in_prev, esc,
                                    torch.full_like(esc, neg_inf))
                                pidx = torch.topk(
                                    csc, EVICT_PROBE_TOPK,
                                    dim=1).indices
                                prot = torch.zeros_like(yb)
                                prot.scatter_(1, pidx, True)
                                prot &= valid & ~in_prev
                                prot |= in_prev
                                prob_rows[sc_name].append(
                                    prot.gather(1, tops).float())
                            prob_meta.append(int(t))
                    # hedge update
                    tcand = (yb & valid).sum(dim=1)
                    tall = yb.sum(dim=1)
                    m8 = torch.topk(
                        torch.where(mmask, raw_m,
                                    torch.full_like(raw_m, neg_inf)),
                        8, dim=2).indices
                    hits8 = yb[None].expand(n_mem, -1, -1) \
                        .gather(2, m8).sum(dim=2).double()
                    # denom per member: candidate-restricted members
                    # cap at the candidate truth count (parity)
                    den_v = torch.minimum(
                        torch.full_like(tcand, 8),
                        tcand).clamp(min=1).double()
                    den_o = torch.minimum(
                        torch.full_like(tall, 8),
                        tall).clamp(min=1).double()
                    dens = torch.stack(
                        [den_v if m in BASE_BANK else den_o
                         for m in members])
                    losses = 1.0 - hits8 / dens
                    active = (tcand > 0).double()
                    st.hw *= torch.exp(-HEDGE_ETA * losses.t()
                                       * active[:, None])
                    st.hw /= st.hw.sum(dim=1, keepdim=True)
                    if attn is not None:
                        attn.update(raw_m[m_idx["rls_open"]], yb,
                                    int(attn_b_all[t]), tops,
                                    (float(top_w.mean()), conf_now))
                    if cfg.cal_head:
                        # π̂: read curves PRE-update, then update the
                        # reliability tables (trains on warm too;
                        # certification accumulates on held only)
                        sc_cal = raw_m[m_idx["rls_open"]]
                        if collect["on"]:
                            if st.cal_n > 0:
                                ph = st.calH / st.cal_n
                                pp = st.calP / st.cal_n
                            else:
                                ph = torch.zeros_like(st.calH)
                                pp = torch.zeros_like(st.calP)
                            p32 = 32 - c.K
                            c32 = pp.sum(1) + ph[:, :p32].sum(1)
                        b2_all = mi2 = None
                        if cfg.cal2d:
                            # pihat2d: bucket the position pre-update
                            b2_all, mi2 = st.cal2d_buckets(sc_cal,
                                                           in_prev)
                            if collect["on"]:
                                cal_mhat2.append(
                                    st.cal2d_mhat(b2_all))
                        ho, hp = st.cal_update(sc_cal, in_prev, yb)
                        if cfg.cal2d:
                            st.cal2d_update(ho, hp, b2_all, mi2)
                        if collect["on"]:
                            b = collect.get("bank", 0)
                            ek = c.E - c.K
                            cal_pred[b, :, :ek] += ph.double()
                            cal_pred[b, :, ek:] += pp.double()
                            cal_real[b, :, :ek] += ho.double()
                            cal_real[b, :, ek:] += hp.double()
                            cal_bn[b] += 1
                            cal_mhat.append((c.K - c32).sum())
                            cal_mreal.append(
                                (c.K - hp.sum(1)
                                 - ho[:, :p32].sum(1)).sum())
                            t1, t2 = _pool_geometry(sc_cal, in_prev)
                            cal_t1.append(t1)
                            cal_t2.append(t2)
                    if ctxs is not None:
                        # decision rows: prev8_mask still holds t-1
                        ctxs.update(yb, st.prev8_mask == 0, tok_id)
                    st.update(tops, sel, top_w, X, valid, yb, bucket,
                              bucket2)
                    if cfg.manifest_head:
                        # buffer this position's PRE-update features
                        # for next-chunk-union labeling
                        st.cbuf.append(X)
                else:
                    X = torch.zeros(c.J, c.E, st.F, device=dev)
                    valid = torch.zeros(c.J, c.E, device=dev,
                                        dtype=torch.bool)
                    yb = torch.zeros(c.J, c.E, device=dev,
                                     dtype=torch.bool)
                    yb.scatter_(1, tops, True)
                    if attn is not None:      # prime the window ring
                        attn.update(torch.zeros(c.J, c.E, device=dev),
                                    yb, int(attn_b_all[t]), tops,
                                    (float(top_w.mean()), conf_now))
                    if ctxs is not None:      # prime the token ring
                        ctxs.update(yb, st.prev8_mask == 0, tok_id)
                    st.update(tops, sel, top_w, X, valid, yb, bucket,
                              bucket2)
        if is_evict and prob_meta:
            for sc_name, rows in prob_rows.items():
                dm = dumps[sc_name]
                probs = torch.stack(rows).cpu().numpy()  # [Te, J, K]
                for i, t in enumerate(prob_meta):
                    for j in range(c.J):
                        dm["seq"].append(np.full(c.K, key, np.int64))
                        dm["pos"].append(
                            np.full(c.K, s["pos"][t], np.int32))
                        dm["layer"].append(np.full(c.K, j, np.int16))
                        dm["expert"].append(
                            tops_np[t, j].astype(np.int16))
                        dm["prob"].append(
                            probs[i, j].astype(np.float32))
                        dm["label"].append(s["recur"]["label"][t, j])
                        dm["occ_prev"].append(
                            s["recur"]["occ_prev"][t, j])
                        dm["nextdist"].append(
                            s["recur"]["nextdist"][t, j])

    import time
    quant_info = None
    prune_info = None
    for phase, keys in (("warm", warm_keys), ("held", held_keys)):
        if not len(keys):
            continue
        if phase == "held" and cfg.has_prune:
            prune_info = prune_state(st, cfg)
            print(f"[stagec'] x20 prune: {prune_info}", flush=True)
        if phase == "held" and cfg.quant_fp8:
            quant_info = quantize_state_fp8(st, ctxs=ctxs)
            print(f"[stagec'] fp8 state round-trip: {quant_info}",
                  flush=True)
        collect["on"] = phase == "held"
        seqs = stage0.gather_split_cells(
            c, keys, recur_keys=evict_keys if phase == "held"
            else frozenset())
        n_done = 0
        t0 = time.time()
        for run_idx, skeys in stage0._by_run(seqs):
            v = c.view(run_idx)
            for key in skeys:
                if cfg.cal_head and collect["on"]:
                    # π̂ drift gate: seq-quartile bank index
                    collect["bank"] = min(
                        3, (4 * n_done) // max(1, len(seqs)))
                process_seq(key, seqs[key], v)
                n_done += 1
                if n_done % 20 == 0:
                    print(f"[stagec'] {cfg.name} {phase} "
                          f"{n_done}/{len(seqs)} seqs "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    if cfg.attn_zero_gate:
                        g = attn.model.gamma.detach().cpu().numpy()
                        gamma_traj.append(
                            {"phase": phase, "seqs": n_done,
                             "abs_mean": float(np.abs(g).mean()),
                             "abs_max": float(np.abs(g).max())})
            c.drop_run(run_idx)

    torch.cuda.synchronize(dev)
    export_files = None
    if export_dir is not None:
        export_files = export_state(st, export_dir, ctxs=ctxs)
        if ctxs is not None:
            export_files["ctx_encoder"] = "ctx_encoder.npz"
        print(f"[stagec'] exported state: {export_files}", flush=True)
    dump_path = None
    if dump["seq"]:
        if len(cfg.eff_evict_scorers) == 1:
            arrs = {k: np.concatenate(v) for k, v in dump.items()}
            p = work / f"preds_stagecprime_{cfg.name}.npz"
            np.savez_compressed(p, **arrs)
            dump_path = str(p)
            print(f"[stagec'] evict dump: {dump_path} "
                  f"({len(arrs['seq']):,} rows)", flush=True)
        else:
            dump_path = {}
            for sc_name, dm in dumps.items():
                arrs = {k: np.concatenate(v) for k, v in dm.items()}
                p = work / (f"preds_stagecprime_{cfg.name}_"
                            f"{sc_name}.npz")
                np.savez_compressed(p, **arrs)
                dump_path[sc_name] = str(p)
                print(f"[stagec'] evict dump [{sc_name}]: {p} "
                      f"({len(arrs['seq']):,} rows)", flush=True)
    def _cal_metrics():
        """π̂ certification (ARCHITECT_CALIBRATION.md §5): ECE over
        balanced rank bins, deep max bin gap, position-level M̂ vs
        realized misses (bias/MAE/Spearman), seq-quartile drift."""
        if not cfg.cal_head or float(cal_bn.sum()) <= 0:
            return None
        n_all = cal_bn.sum()
        gap = (cal_pred.sum(0) - cal_real.sum(0)).abs() / n_all
        ece_mean = float(gap.mean(dim=1).mean())
        deep_gap = float(gap[deep].max())
        drift = None
        if float(cal_bn[0]) > 0 and float(cal_bn[3]) > 0:
            e0 = float(((cal_pred[0] - cal_real[0]).abs()
                        / cal_bn[0]).mean())
            e3 = float(((cal_pred[3] - cal_real[3]).abs()
                        / cal_bn[3]).mean())
            drift = e3 / max(e0, 1e-9)
        mh = torch.stack(cal_mhat).cpu().numpy()
        mr = torch.stack(cal_mreal).cpu().numpy()
        # §7.2 series dump: the offline decision set for the decay
        # sweep + the position-risk ridge (per-position M-hat, M,
        # and current-position score geometry)
        sp_path = work / f"pihat_series_{cfg.name}.npz"
        extra = ({"mhat2d": torch.stack(cal_mhat2).cpu().numpy()}
                 if cfg.cal2d and cal_mhat2 else {})
        np.savez(sp_path, mhat=mh, mreal=mr,
                 t1=torch.stack(cal_t1).cpu().numpy(),
                 t2=torch.stack(cal_t2).cpu().numpy(), **extra)
        m_bias = float((mh - mr).mean())
        m_mae = float(np.abs(mh - mr).mean())
        sp = None
        if len(mh) > 1:
            def _rk(x):
                r = np.empty(len(x))
                r[np.argsort(x, kind="stable")] = np.arange(len(x))
                return r
            cc = np.corrcoef(_rk(mh), _rk(mr))[0, 1]
            sp = float(cc) if np.isfinite(cc) else None
        out = {
            "ece_mean": ece_mean,
            "ece_deep_max_gap": deep_gap,
            "m_bias": m_bias, "m_mae": m_mae, "m_spearman": sp,
            "drift_ratio": drift,
            "held_positions": int(n_all.item()),
            "series": str(sp_path),
            "gates": {
                "ece": ece_mean <= 0.010,
                "deep_gap": deep_gap <= 0.05,
                "bias": abs(m_bias) <= 0.10,
                "mae": m_mae <= 0.5,
                "spearman": sp is not None and sp >= 0.5,
                "drift": drift is not None and drift <= 2.0,
            }}
        if cfg.cal2d and cal_mhat2:
            # pihat2d certification (§7 pre-registration, total
            # scale): spearman >= 0.5, MAE <= 0.7*std(M) (= beat
            # the constant by >= 30%), |bias| <= 2.0. The 1-D
            # marginal gates above are unchanged by construction
            # (Laplace shrinkage to the 1-D rate; tables separate).
            m2 = torch.stack(cal_mhat2).cpu().numpy()
            b2 = float((m2 - mr).mean())
            mae2 = float(np.abs(m2 - mr).mean())
            sd = float(mr.std())
            mae_c = float(np.abs(mr - mr.mean()).mean())
            cc2 = np.corrcoef(_rk(m2), _rk(mr))[0, 1]
            sp2 = float(cc2) if np.isfinite(cc2) else None
            beat = 1.0 - mae2 / max(mae_c, 1e-9)
            out["pihat2d"] = {
                "m_bias": b2, "m_mae": mae2, "m_spearman": sp2,
                "std_m": sd, "const_mae": mae_c,
                "beat_const_pct": 100 * beat,
                "gates": {
                    "spearman": sp2 is not None and sp2 >= 0.5,
                    "mae_vs_std": mae2 <= 0.7 * sd,
                    "beat_const": beat >= 0.30,
                    "bias": abs(b2) <= 2.0,
                }}
        return out

    ph = pool_hit.cpu().numpy()
    ah = addr_hit.cpu().numpy()
    nh = novel_hit.cpu().numpy()
    mn = man.cpu().numpy()
    mo = man_open.cpu().numpy()
    mf = mf_acc.cpu().numpy()
    ml = c.moe_layers
    deep_mask = (ml >= deep_lo) & (ml <= deep_hi)
    hw = st.hw.cpu().numpy()
    return {
        "variants": list(variants),
        "members": list(members),
        "pool_hit": ph.tolist(),
        "addr_hit": ah.tolist(),
        "novel_hit": nh.tolist(),
        "pool_true": pool_true,
        "addr_n": int(addr_n.item()),
        "novel_n": int(novel_n.item()),
        "pi_hat": _cal_metrics(),
        "memo2_support": (
            {"mean": float(m2_sum.item() / max(1, m2_n.item())),
             "frac_lt8": float(m2_lt8.item() / max(1, m2_n.item()))}
            if use_memo2 else None),
        "man": {str(d): [float(mn[i, 0]), int(mn[i, 1])]
                for i, d in enumerate(MANIFEST_DS)},
        "man_head": {str(d): [float(man_head[i, 0].item()),
                              int(man_head[i, 1].item())]
                     for i, d in enumerate(MANIFEST_DS)}
        if cfg.manifest_head else None,
        "man_head_open": {str(d): [float(man_head_open[i, 0].item()),
                                   int(man_head_open[i, 1].item())]
                          for i, d in enumerate(MANIFEST_DS)}
        if cfg.manifest_head else None,
        "man_open": {str(d): [float(mo[i, 0]), int(mo[i, 1])]
                     for i, d in enumerate(MANIFEST_DS)},
        "man_free": {"prev_union": [float(mf[0, 0]), float(mf[0, 1]),
                                    int(mf[0, 2])],
                     "two_chunk": [float(mf[1, 0]), float(mf[1, 1]),
                                   int(mf[1, 2])]},
        "hedge_w_mean": hw.mean(axis=0).tolist(),
        "hedge_w_deep": hw[deep_mask].mean(axis=0).tolist(),
        "ridge_w": st.wr.cpu().numpy().tolist(),
        "feat_names": list(st.feat_names),
        "export_files": export_files,
        "evict_dump": dump_path,
        "quant_fp8": quant_info,
        "x20_prune": prune_info,
        "novel_splits": {f"{ki}_{ko}": [int(split_pool[i]),
                                        int(split_addr[i]),
                                        int(split_novel[i])]
                         for i, (ki, ko) in
                         enumerate(cfg.novel_splits)}
        if cfg.novel_head else None,
        "gamma_traj": gamma_traj or None,
        "gamma_final": (attn.model.gamma.detach().cpu().numpy()
                        .tolist() if cfg.attn_zero_gate else None),
    }

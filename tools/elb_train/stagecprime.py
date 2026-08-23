"""P11.c' driver: the extended-signal saga (TD-P11-MISSING-SIGNALS;
spec/reports/EXPOSED_BYTE_CALCULUS.md §3-P11 c' backlog).

Each SIGNAL is a named experiment over the stagecprime_gpu engine (the
stagec dense-[J,E] template + feature/member registry + open channel);
every experiment runs warm(train)+held on ONE GPU and is judged on the
same deployed metrics as P11.c (pool coverage, manifest top-up) plus
NOVEL-slot recovery, with boundary discipline (addressable recovery
reported next to coverage).

Steps (state in --workdir, results accumulate in results.json):

  prep-kmeans  GPU k-means over the checkpoint embedding table ->
               token -> bucket lut (embedding-space memo; fixes the
               id-hash collision noise that killed token_memo).
  prep-tap     RouterBank logits over the corpus DRAFT hiddens (aux
               tap mapping) -> per-seq committed-row tap logits +
               per-block maxpool cache (~16 GB, one-time).
  run          --exp <name,...|all>: run experiments on --device.
  verdict      Compare vs the P11.c bars + the x0 parity baseline;
               file the study numbers.

Experiments (see EXPERIMENTS): x0 parity gate, x1 trans2, x2 xprev,
x3 dynamics, x4 embedding memo, x5 draft tap, x6 open channel (novel
mass), x7 final combined bank (winners only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elb_train import glm_router, stage0  # noqa: E402
from elb_train.stageb import MANIFEST_DS, POOL_KS  # noqa: E402
from elb_train.stagecprime_gpu import ExpCfg  # noqa: E402

EXPERIMENTS: dict[str, dict] = {
    "x0_parity": dict(),
    "x1_trans2": dict(features=("trans2",), members=("trans2",)),
    "x2_xprev": dict(features=("xprev_up", "xprev_dn"),
                     members=("xprev_up", "xprev_dn")),
    "x3_dyn": dict(features=("w_prev", "w_ema", "gfreq", "conf_trans"),
                   members=("gfreq",)),
    "x4_memo": dict(features=("memo",), members=("memo",)),
    "x5_tap": dict(features=("tap",), members=("tap",)),
    "x6_open": dict(features=("memo", "tap"), members=("memo", "tap"),
                    open_ch=True),
    # final stacks (composed from the x1-x6 verdicts, LOG.md):
    # x7_final = the keepers (memo + xprev) with the open channel;
    # x8_kitchen retests the marginal signals on top of the keepers.
    "x7_final": dict(features=("xprev_up", "xprev_dn", "memo"),
                     members=("memo",), open_ch=True),
    # memo bucket-size sensitivity around the champion stack
    "x7b_memo8k": dict(features=("xprev_up", "xprev_dn", "memo"),
                       members=("memo",), open_ch=True,
                       memo_buckets=8192),
    "x7c_memo1k": dict(features=("xprev_up", "xprev_dn", "memo"),
                       members=("memo",), open_ch=True,
                       memo_buckets=1024),
    "x7d_memo16k": dict(features=("xprev_up", "xprev_dn", "memo"),
                        members=("memo",), open_ch=True,
                        memo_buckets=16384),
    "x7e_memo32k": dict(features=("xprev_up", "xprev_dn", "memo"),
                        members=("memo",), open_ch=True,
                        memo_buckets=32768),
    "x7f_memo64k": dict(features=("xprev_up", "xprev_dn", "memo"),
                        members=("memo",), open_ch=True,
                        memo_buckets=65536),
    # identity endpoint: per-token memo (B = vocab, identity lut —
    # no k-means; answers semantic-sharing vs pure resolution)
    "x7g_memotok": dict(features=("xprev_up", "xprev_dn", "memo"),
                        members=("memo",), open_ch=True,
                        memo_buckets=154880),
    "x8_kitchen": dict(features=("trans2", "xprev_up", "xprev_dn",
                                 "w_prev", "w_ema", "gfreq",
                                 "conf_trans", "memo", "tap"),
                       members=("memo", "tap"), open_ch=True),
    # champion under fp8-e4m3 persistent state (deployment precision)
    "x9_fp8": dict(features=("xprev_up", "xprev_dn", "memo"),
                   members=("memo",), open_ch=True,
                   memo_buckets=154880, quant_fp8=True),
    # stage-2: recur16 second head on the shared open-channel A —
    # eviction-native target (near-future reuse), pool paths untouched
    "x9b_recur16": dict(features=("xprev_up", "xprev_dn", "memo"),
                        members=("memo",), open_ch=True,
                        memo_buckets=154880, recur_head=True,
                        evict_scorer="recur16"),
    # stage-3: the pre-registered §3-P11(4)(iii) tiny-attention
    # challenger (multi-token window + expert-id embeddings), an
    # online residual over the open ridge; kill terms: beat pool@32
    # 0.7252 AND novel-rec 0.586 at the ≤50 µs rent, else killed.
    "x10_attn": dict(features=("xprev_up", "xprev_dn", "memo"),
                     members=("memo",), open_ch=True,
                     memo_buckets=154880, attn_mode="pooled"),
    "x11_attn_expq": dict(features=("xprev_up", "xprev_dn", "memo"),
                          members=("memo",), open_ch=True,
                          memo_buckets=154880,
                          attn_mode="expert_query"),
    # x12: challenger subsumes c' BY MECHANICS — zero-init LayerScale
    # gate (init == ridge ranking exactly; gamma trajectory = the
    # ledger's capacity-value diagnostic) + the attention output as a
    # Hedge member next to rls_open (regret floor >= best member).
    "x12_attn_floor": dict(features=("xprev_up", "xprev_dn", "memo"),
                           members=("memo",), open_ch=True,
                           memo_buckets=154880,
                           attn_mode="expert_query",
                           attn_zero_gate=True, attn_hedge=True),
    # stage-4: x13 CUTTRACK staged arms (studies/epm/p11-stagecprime/
    # ARCHITECT_REVIEW.md Part B.1 + Part C). Pre-registered kill
    # gates (held, seed 35): PROMOTE a stage iff best-of
    # {rls_soft, rls_open} pool@32 >= 0.7282 (champion +0.3pp) AND
    # novel-rec >= 0.58615 AND addressable-rec >= 0.7390 moving
    # proportionally with coverage (boundary discipline: wins not
    # backed by addressable/novel recovery are noise-fitting -> KILL)
    # AND topup32 >= 0.9282. Parity zone (bars beaten, < +0.3pp):
    # keep x13a only (zero-state label repair); otherwise KILL.
    # x13a keeps F = 15, so its in-run rls_open must reproduce the
    # champion bars BIT-EXACTLY (0.7251875134961241 /
    # 0.5861492837044215) — the parity anchor. x13b/c enlarge F
    # (19 / 21): their rls_open re-derives on the new feature space;
    # x13a's rls_open stays the cross-run control. x13c runs only if
    # the P-B0 probe fires (common innovation variance >= 30%).
    "x13a_soft": dict(features=("xprev_up", "xprev_dn", "memo"),
                      members=("memo",), open_ch=True,
                      memo_buckets=154880, soft_head=True),
    "x13b_cuttrack": dict(features=("xprev_up", "xprev_dn", "memo"),
                          members=("memo",), open_ch=True,
                          memo_buckets=154880, soft_head=True,
                          filter_feats=True),
    "x13c_factor": dict(features=("xprev_up", "xprev_dn", "memo"),
                        members=("memo",), open_ch=True,
                        memo_buckets=154880, soft_head=True,
                        filter_feats=True, factor_r=8),
    # stage-4 REVISIONS (one per arm, ARCHITECT_REVIEW.md Part D).
    # The soft label is KILLED with mechanism (target bias under
    # heterogeneous margin variance — x13a addr-rec −3.7pp, worse
    # with filter features). Revisions use HARD labels only; row
    # WEIGHTING cannot bias the target.
    # x13a2 factorial readout (F=15, in-run rls_open = bit-exact
    # anchor): rls_dr vs rls_open isolates the decision-row
    # restriction; rls_band vs rls_open isolates noise-band row
    # exclusion (|sel-cut| <= sigma_j); rls_drband is the stack.
    # PROMOTE any head iff pool@32 >= 0.7282 AND novel >= 0.58615
    # AND addr-rec-backed; if none beats rls_open, the objective
    # line on F=15 is DEAD (champion objective vindicated).
    "x13a2_band": dict(features=("xprev_up", "xprev_dn", "memo"),
                       members=("memo",), open_ch=True,
                       memo_buckets=154880, band_heads=True),
    # x13b2: keep the champion objective (hard, all rows — rls_open
    # IS the candidate); fix the two measured x13b deficiencies:
    # dormancy-gated dynamics features (novel −0.2pp = forecast
    # noise on recency>=64 rows where memo must rule) + inverse-MSE^4
    # gain MIXTURE (x13d-diag: 436 argmin flips/seq = selection
    # churn). Readout: novel restored >= 0.58615 attributes to the
    # gate; pool beyond 0.72639 attributes to mixing + gated
    # addressable sharpening. Same promote gates.
    "x13b2_gated": dict(features=("xprev_up", "xprev_dn", "memo"),
                        members=("memo",), open_ch=True,
                        memo_buckets=154880, filter_feats=True,
                        dormancy_gate=True, gain_mix=True),
    # x13c2: b2 + the factor stage (P-B0 FIRED: 52-59% common
    # innovation variance, cut-PC1 corr 0.59). Run ONLY after x13c
    # returns and only if x13c rls_open >= x13b rls_open (factor
    # features not harmful); else the c-arm dies with b's verdict.
    "x13c2_factor": dict(features=("xprev_up", "xprev_dn", "memo"),
                         members=("memo",), open_ch=True,
                         memo_buckets=154880, filter_feats=True,
                         dormancy_gate=True, gain_mix=True,
                         factor_r=8),
    # x16 bigram ladder (ARCHITECT_REVIEW.md Part B.6): hashed
    # (prev_tok, tok) -> expert counts as a 16th feature on the
    # champion stack — measures token-SEQUENCE context beyond the
    # unigram memo; the n-gram-table-before-neural rung. Deployed
    # form pre-registered: support >= 8 rows, k = 8 sparse stamp
    # (honest bytes reported by the export as memo2_stamp_bytes).
    # Decision rule (held, seed 35):
    #   FIRES:  rls_open pool@32 >= 0.7282 AND novel >= 0.58615 AND
    #           addr-backed -> context signal real; next rung =
    #           bucket sweep (memo2_buckets=131072, ~10 GB table).
    #   FLAT:   pool@32 < 0.72619 (champion +0.1pp) -> ONE
    #           confirmation at 131072 (collision check, the memo
    #           bucket-curve lesson); if still flat, token-sequence
    #           context is DEAD on this corpus and the tiny-
    #           transformer-as-feature line is CLOSED (its exclusive
    #           food measured absent).
    #   FIRES-BUT-SPARSE: pool@32 in [0.72619, 0.7282) AND
    #           memo2_support.frac_lt8 >= 0.5 -> the pre-registered
    #           trigger for the tiny context encoder (feature-form,
    #           score-space decision-row residual target, Hedge-
    #           floored, 100k-1MB params).
    "x16_bigram": dict(features=("xprev_up", "xprev_dn", "memo",
                                 "memo2"),
                       members=("memo",), open_ch=True,
                       memo_buckets=154880, memo2_buckets=65536),
    # x13c3 REPAIR (completes the 2x2 after the x13b2/c2 verdicts):
    # measured decomposition — factor helps BOTH axes (x13b->x13c:
    # pool +0.12pp, novel +0.13pp); gate+mix adds ~+0.15pp pool but
    # crushes novel −1.25pp in BOTH stacks (the dormancy gate removed
    # real dormant-row dynamics signal — mechanism attribution wrong,
    # owned in Part D). c3 = filter + factor + gain_mix, NO dormancy
    # gate, NO soft head. vs x13c isolates the mix effect; vs x13c2
    # isolates the gate effect. The one live promote path:
    # PROMOTE iff rls_open pool@32 >= 0.7282 AND novel >= 0.58615
    # AND addr-backed AND topup32 >= 0.9282 (conjunctive, unchanged).
    # If it misses, the corpus objective/feature line CONCEDES per
    # the Part D decision rule (c'' recommendation stands).
    "x13c3_repair": dict(features=("xprev_up", "xprev_dn", "memo"),
                         members=("memo",), open_ch=True,
                         memo_buckets=154880, filter_feats=True,
                         gain_mix=True, factor_r=8),
    # x17 ctxformer (user-requested direct experiment; Part B.6
    # strongest form under the x10-x13 lessons): ONE-head transformer
    # over the trailing-8 committed-token window, identity-resolution
    # FROZEN PCA-32 token features (embed_ctx32.npz, lazy prep),
    # per-layer UNTIED expert tables, score-space MSE on DECISION
    # rows w/ per-layer bias (no BCE link), output = ONE c' feature
    # column (~350k learned params; one forward/position, µs-class).
    # Gates: PROMOTE at the standing bars (0.7282 / 0.58615 /
    # addr-backed). Scientific readout vs x16_bigram: table >= x17
    # -> tables win the context class; x17 > x16 both sub-gate ->
    # context real, neural extraction insufficient at this size;
    # x17 promotes -> the capacity ledger is OVERTURNED for the
    # feature-form seat (record it prominently).
    "x17_ctxformer": dict(features=("xprev_up", "xprev_dn", "memo",
                                    "ctx"),
                          members=("memo",), open_ch=True,
                          memo_buckets=154880),
}
EXPERIMENTS["x16m_heads"] = dict(
    # green-lit head batch on the PROMOTED x16 stack: recur16 evictor
    # cert for x16 (vs +10.68%), w_manifest topup head (vs 0.92835),
    # recur32/64 heads with their own sim pairs
    features=("xprev_up", "xprev_dn", "memo", "memo2"),
    members=("memo",), open_ch=True,
    memo_buckets=154880, memo2_buckets=65536,
    recur_horizons=(16, 32, 64), manifest_head=True,
    evict_scorers=("recur16", "recur32", "recur64"))
EXPERIMENTS["x18_hybrid"] = dict(
    # decisive hybrid: bigram AND ctxformer features, per-consumer
    # scoring (pools = rls_open on the full stack; topup =
    # w_manifest). Gates = x16's numbers: pool >= 0.72894, novel >=
    # 0.61530, addr-backed, topup >= 0.92835.
    features=("xprev_up", "xprev_dn", "memo", "memo2", "ctx"),
    members=("memo",), open_ch=True,
    memo_buckets=154880, memo2_buckets=65536,
    recur_horizons=(16,), manifest_head=True)

EXPERIMENTS["x18e_evict"] = dict(
    # x18 evictor certification: recur16-head + rls_open dumps on the
    # hybrid stack (feature-invariance lesson predicts ~+10.6 / ~+7.1)
    **{k: v for k, v in EXPERIMENTS["x18_hybrid"].items()},
    evict_scorers=("recur16", "rls_open"))
EXPERIMENTS["x18m_masked"] = dict(
    # per-consumer FEATURE masking: manifest/recur heads fit on the
    # no-ctx subset (second masked-accumulation A — masked columns
    # solve to exactly 0), w_open keeps the full set. Expectation:
    # pools hold 0.73475-class, topup recovers to >= 0.93269-class.
    **{k: v for k, v in EXPERIMENTS["x18_hybrid"].items()},
    head_mask_features=("ctx",),
    evict_scorers=("recur16",))
EXPERIMENTS["x17m_heads"] = dict(
    # MANDATORY full row: x17 stack + full head family; heads fit on
    # the no-ctx subset (masked-manifest variant — ctx is in this
    # stack, so the mask applies), w_open keeps ctx.
    features=("xprev_up", "xprev_dn", "memo", "ctx"),
    members=("memo",), open_ch=True, memo_buckets=154880,
    recur_horizons=(16,), manifest_head=True,
    head_mask_features=("ctx",),
    evict_scorers=("recur16",))

# ctx-encoder SIZE SLOPE (capacity probe on the WORKING seat): 1.25x
# (d_out 20 -> 436,903 params) and 2.1x (d 48 / d_out 32 -> 737,067)
# vs baseline 350,203; frozen PCA-32 + everything else fixed; ctx
# stays w_open-only (head mask). Gates: the stack's standing bars.
for _nm, _base, _kw in (
        ("x18s1_ctx125", "x18m_masked", dict(ctx_d_out=20)),
        ("x18s2_ctx210", "x18m_masked", dict(ctx_d=48, ctx_d_out=32)),
        ("x17s1_ctx125", "x17m_heads", dict(ctx_d_out=20)),
        ("x17s2_ctx210", "x17m_heads", dict(ctx_d=48, ctx_d_out=32))):
    EXPERIMENTS[_nm] = dict(
        **{k: v for k, v in EXPERIMENTS[_base].items()
           if k != "evict_scorers"}, **_kw)

EXPERIMENTS["x17n_novel"] = dict(
    **{k: v for k, v in EXPERIMENTS["x17m_heads"].items()},
    novel_head=True)
EXPERIMENTS["x18n_novel"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18m_masked"].items()},
    novel_head=True)

# SHIP configurations (E2E green-light, 2026-08-14): novel head
# KILLED on both stacks (open channel is the optimal novel
# allocator); ctx size-slope REAL — 1.25x point promotes on x17
# (0.73174/0.59486/0.80769 vs 0.73092/0.59345/0.80628; 2.1x slopes
# further — logged, not shipped). Compact = x17 stack + 1.25x ctx +
# masked head family; XL = x18 stack (ctx point per its own slope
# verdict), both with w_open/w_valid/w_recur16/w_manifest.
EXPERIMENTS["x17ship"] = dict(
    **{k: v for k, v in EXPERIMENTS["x17m_heads"].items()},
    ctx_d_out=20)
EXPERIMENTS["x17ship_fp8"] = dict(
    **{k: v for k, v in EXPERIMENTS["x17m_heads"].items()
       if k != "evict_scorers"},
    ctx_d_out=20, quant_fp8=True)
EXPERIMENTS["x18ship"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18m_masked"].items()},
    ctx_d_out=20)
EXPERIMENTS["x18ship_fp8"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18m_masked"].items()
       if k != "evict_scorers"},
    ctx_d_out=20, quant_fp8=True)

# SHIP-2 candidates (2026-08-15): the 2.1x ctx slope point (ctx_d=48,
# ctx_d_out=32 -> 737,067-param encoder) as candidate successors to
# both ships. Gates = the CURRENT ship bars, conjunctive; masked heads
# see no ctx, so topup/evict cells must be BIT-IDENTICAL to the 1.25x
# ships (chassis guarantee — any drift is a bug).
EXPERIMENTS["x17ship2"] = dict(
    **{k: v for k, v in EXPERIMENTS["x17m_heads"].items()},
    ctx_d=48, ctx_d_out=32)
EXPERIMENTS["x17ship2_fp8"] = dict(
    **{k: v for k, v in EXPERIMENTS["x17m_heads"].items()
       if k != "evict_scorers"},
    ctx_d=48, ctx_d_out=32, quant_fp8=True)
EXPERIMENTS["x18ship2"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18m_masked"].items()},
    ctx_d=48, ctx_d_out=32)
EXPERIMENTS["x18ship2_fp8"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18m_masked"].items()
       if k != "evict_scorers"},
    ctx_d=48, ctx_d_out=32, quant_fp8=True)

# A2+A3 COMBINED ablation (ARCHITECT_ABLATION.md Tier 1, 2026-08-15
# order): xprev AND trans completely removed from the ridge/heads of
# both frontier champions — exact zero-column LOFO (Tikhonov lemma);
# ablated tables are neither built nor exported (the ~50 MB price:
# xprev 33.0 + trans 16.7). Gates: architect DEAD gates vs the 2.1x
# bars (dpool >= -0.05pp, dnovel >= -0.10pp, daddr >= -0.10pp,
# dtopup >= -0.05pp, devict@1052 >= -0.30pp); any dnovel < -0.3pp
# rejects regardless of bytes.
EXPERIMENTS["x17ship2_a23"] = dict(
    **{k: v for k, v in EXPERIMENTS["x17ship2"].items()},
    ablate=("xprev_up", "xprev_dn", "trans", "trans_rank"))
EXPERIMENTS["x18ship2_a23"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship2"].items()},
    ablate=("xprev_up", "xprev_dn", "trans", "trans_rank"))
# xsame_Δ PIPELINED channel (ARCHITECT_TRICK.md + Q&A addendum,
# 2026-08-15 green light): SOFT form (score-weighted [J,E,E] gather
# over the full 256-sel row of layer j-Δ at the CURRENT position),
# Δ in {1,2,4,8}, base stack x18ship2, xsame head-masked next to ctx
# (heads/evict stay strict-contract; w_open is the pipelined
# consumer). METRIC DISCIPLINE: these runs' pool/novel/addr are
# pool@32-PIPELINED(Δ) — reported under that name, NEVER mixed with
# the strict ship bars. Gates: P1 Δ=1 residual-AUC >= 0.744
# (ceiling diagnostic, probe script); P2 pipelined Δ=8 >= ship
# +1.0pp addressable-backed, monotone in Δ; KILL at Δ=8 < ship
# +0.3pp (closes the corpus line AND collapses the c'' prior).
for _d in (1, 2, 4, 8):
    EXPERIMENTS[f"x18ship2_xs{_d}"] = dict(
        **{k: v for k, v in EXPERIMENTS["x18ship2"].items()
           if k not in ("features", "head_mask_features")},
        features=(*EXPERIMENTS["x18ship2"]["features"], "xsame"),
        head_mask_features=("ctx", "xsame"),
        xsame_lag=_d)

# daylight ladder rung (post-kill-gate, sanctioned by the standing
# hard-multi-if-daylight clause): HARD-single Δ=8 — P1 measured the
# hard Δ=1 feature at record AUC 0.783 while the soft ridge arm
# converted only +0.147pp; this run decides whether the gap is the
# soft FORM (hard jumps) or ridge-level subsumption (hard lands
# ~+0.15pp too -> channel closed, hard-multi not worth building).
EXPERIMENTS["x18ship2_xs8h"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship2_xs8"].items()},
    xsame_soft=False)
# hard-form curve completion (xs8h FIRED P2 at +1.604pp — the soft
# form was the defect, not the channel): Δ=4/2 for the monotone cell
# and the deployment-lead choice; Δ=1 = the in-ridge channel ceiling.
for _d in (1, 2, 4):
    EXPERIMENTS[f"x18ship2_xs{_d}h"] = dict(
        **{k: v for k, v in EXPERIMENTS[f"x18ship2_xs{_d}"].items()},
        xsame_soft=False)

# x18ship3 (USER GREEN LIGHT 2026-08-15): the NEW LEADER — x18ship2
# strict stack + HARD xsame Δ=2 (+2.234pp pipelined pool) + its OWN
# π̂ tables trained on the pipelined stream (teacher stream-
# separation rule; never x19's strict tables). Sidecars (EWMA ρ=0.7
# + x19b-lite geometry ridge) attach at packaging as UNCERTIFIED
# companions fit on this run's own series dump. Expected cells:
# pipelined2 pool/novel/addr bit-exact vs xs2h; topups/evict
# bit-identical vs ship (masked heads); π̂ reliability gates on ITS
# stream; fp8 extended round-trip incl. the xsame table.
EXPERIMENTS["x18ship3"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship2_xs2h"].items()},
    cal_head=True)
EXPERIMENTS["x18ship3_fp8"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship2_xs2h"].items()
       if k != "evict_scorers"},
    cal_head=True, quant_fp8=True)

# pihat2d (§7 pre-registration, fd3ce3a5 naming; user green light
# 2026-08-15): 2-D adjustment on the π̂ head, OWN tables per stream.
# Primary = ship3's pipelined(2) stream (the deployed target);
# comparison = the strict stream. Gates: 1-D marginal/reliability
# unchanged AND spearman >= 0.5 AND beat the 4-float sidecar bar
# (ship3 0.4755 / strict 0.4646) AND beat-const MAE >= 30% (mae <=
# 0.7*std(M)) AND |bias| <= 2.0 total scale. Anchors mandatory.
EXPERIMENTS["x18ship3_p2d"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship3"].items()},
    cal2d=True)
EXPERIMENTS["x18ship2_p2d"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship2"].items()},
    cal_head=True, cal2d=True)

# x20 pruning campaign (ARCHITECT_ABLATION §3 Tier-2, user green
# light 2026-08-16): knee replays on the x18ship3 stack — prunes
# applied at the warm->held boundary (fp8-cert pattern), judged vs
# ship3's bars (0.75923/0.64332/0.85005/0.92997, evict +10.60);
# any novel loss > 0.3pp REJECTED regardless of bytes; combined
# end-state target <= 0.1pp total. Curves: x20_curves.json.
EXPERIMENTS["x20_memo_k8"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship3"].items()},
    prune_memo_k=8)
EXPERIMENTS["x20_memo2_s16"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship3"].items()},
    prune_memo2_s=16.0)
EXPERIMENTS["x20_tau05"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship3"].items()},
    prune_tau=0.05)
# memo k8 knee REJECTED (dnovel -1.293pp absolute reject — the memo
# tail is novel-mass signal; the teacher's k16->8-free prediction is
# falsified). Combined prune = the surviving knees only.
EXPERIMENTS["x18ship3_pruned"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship3"].items()},
    prune_memo2_s=16.0, prune_tau=0.05)
EXPERIMENTS["x18ship3_pruned_fp8"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship3"].items()
       if k != "evict_scorers"},
    prune_memo2_s=16.0, prune_tau=0.05, quant_fp8=True)

# single ablations (2x2 lattice completion on XL; interaction read
# against baseline + combined at zero extra compute)
EXPERIMENTS["x18ship2_a2"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship2"].items()},
    ablate=("xprev_up", "xprev_dn"))
EXPERIMENTS["x18ship2_a3"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship2"].items()},
    ablate=("trans", "trans_rank"))

# ctx slope 3rd point (~4x, 2026-08-15 green-lit probe): ctx_d=64,
# ctx_d_out=56 -> 1,354,931-param encoder (3.87x baseline 350,203).
# Judged vs the 2.1x candidate bars (slope-shape / knee question);
# NOT a ship registration — no publishing.
for _nm, _base in (("x17s3_ctx390", "x17m_heads"),
                   ("x18s3_ctx390", "x18m_masked")):
    EXPERIMENTS[_nm] = dict(
        **{k: v for k, v in EXPERIMENTS[_base].items()},
        ctx_d=64, ctx_d_out=56)
    EXPERIMENTS[_nm + "_fp8"] = dict(
        **{k: v for k, v in EXPERIMENTS[_base].items()
           if k != "evict_scorers"},
        ctx_d=64, ctx_d_out=56, quant_fp8=True)

# π̂ calibration head (ARCHITECT_CALIBRATION.md, architect-
# implemented 2026-08-15): rank-conditional reliability tables on
# w_open riding the strict XL ship stack. Certification gates (all
# must pass to fold π̂ into the ship bundles + the P2 governor
# seam): ece_mean <= 0.010, deep max bin gap <= 0.05, |m_bias| <=
# 0.10 experts/position, m_mae <= 0.5, spearman(M̂, M) >= 0.5,
# drift ratio (last/first held seq-quartile ECE) <= 2.0. Pool/head
# metrics must be BIT-EXACT vs x18ship2 (the head is read-only on
# the score paths — chassis anchor). Strict-contract only (cfg
# validation refuses xsame).
EXPERIMENTS["x19_pihat"] = dict(
    **{k: v for k, v in EXPERIMENTS["x18ship2"].items()},
    cal_head=True)

ATTN_LUT_BUCKETS = 4096

# P11.c bars (studies/epm/p11-stagec, held split seed 35)
BAR_POOL32 = 0.6257576118140112
BAR_TOPUP32 = 0.9282143710220624


# ── prep: embedding k-means lut ──────────────────────────────────────────────

DEFAULT_EMBED_TENSOR = "model.embed_tokens.weight"


def _embed_shard(ckpt: str, tensor: str) -> Path:
    """Safetensors file holding `tensor` (sharded index or single file)."""
    ckpt_dir = Path(ckpt)
    idx_path = ckpt_dir / "model.safetensors.index.json"
    if idx_path.is_file():
        wm = json.load(open(idx_path))["weight_map"]
        if tensor not in wm:
            raise KeyError(f"{idx_path}: {tensor!r} not in weight_map")
        return ckpt_dir / wm[tensor]
    single = ckpt_dir / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"{ckpt_dir}: no safetensors index or "
                                f"single file")
    return single


def step_prep_kmeans(ckpt: str, work: Path, device: str, buckets: int,
                     iters: int = 25, seed: int = 0,
                     embed_tensor: str = DEFAULT_EMBED_TENSOR) -> dict:
    shard_path = _embed_shard(ckpt, embed_tensor)
    hdr, _ = glm_router._read_st_header(shard_path)
    V = int(hdr[embed_tensor]["shape"][0])
    out = work / f"embed_lut_b{buckets}.npz"
    if buckets >= V:
        # identity endpoint: per-token memo, no k-means involved
        np.savez(out, lut=np.arange(V, dtype=np.int32), buckets=buckets)
        info = {"path": str(out), "vocab": V, "buckets": buckets,
                "identity": True}
        print(f"[prep-kmeans] {info}", flush=True)
        return info
    import torch
    dev = torch.device(device)
    emb_np = glm_router.read_safetensors_tensor(shard_path, embed_tensor)
    V, H = emb_np.shape
    x = torch.as_tensor(emb_np, device=dev)
    x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-6)   # cosine space
    g = torch.Generator(device="cpu").manual_seed(seed)
    cent = x[torch.randperm(V, generator=g)[:buckets].to(dev)].clone()
    assign = torch.zeros(V, device=dev, dtype=torch.int64)
    t0 = time.time()
    for it in range(iters):
        # assign (chunked over tokens)
        for lo in range(0, V, 16384):
            hi = min(V, lo + 16384)
            sim = x[lo:hi] @ cent.t()
            assign[lo:hi] = sim.argmax(dim=1)
        # update
        newc = torch.zeros_like(cent)
        newc.index_add_(0, assign, x)
        cnt = torch.bincount(assign, minlength=buckets).float()
        empty = cnt == 0
        n_empty = int(empty.sum())
        newc = newc / cnt.clamp(min=1)[:, None]
        if n_empty:
            ridx = torch.randperm(V, generator=g)[:n_empty].to(dev)
            newc[empty] = x[ridx]
        cent = newc / newc.norm(dim=1, keepdim=True).clamp(min=1e-6)
        if it % 5 == 0 or it == iters - 1:
            occ = int((cnt > 0).sum())
            print(f"[prep-kmeans] iter {it}: occupied {occ}/{buckets} "
                  f"max {int(cnt.max())} ({time.time()-t0:.0f}s)",
                  flush=True)
    lut = assign.cpu().numpy().astype(np.int32)
    out = work / f"embed_lut_b{buckets}.npz"
    np.savez(out, lut=lut, buckets=buckets,
             centroids=cent.half().cpu().numpy())
    cnt = np.bincount(lut, minlength=buckets)
    info = {"path": str(out), "vocab": int(V), "buckets": buckets,
            "occupied": int((cnt > 0).sum()),
            "max_bucket": int(cnt.max()),
            "median_bucket": float(np.median(cnt[cnt > 0])),
            "seconds": round(time.time() - t0)}
    print(f"[prep-kmeans] {info}", flush=True)
    return info


def step_prep_emb32(ckpt: str, work: Path, device: str,
                    embed_tensor: str = DEFAULT_EMBED_TENSOR) -> dict:
    """x17 ctxformer frozen token features: PCA-32 of the model's
    embedding table, per-column standardized, stored f16."""
    import torch

    from elb_train.stagecprime_attn import emb_project
    out = work / "embed_ctx32.npz"
    shard_path = _embed_shard(ckpt, embed_tensor)
    emb_np = glm_router.read_safetensors_tensor(shard_path,
                                                embed_tensor)
    t0 = time.time()
    f = emb_project(torch.as_tensor(emb_np,
                                    device=torch.device(device)))
    np.savez(out, emb=f.half().cpu().numpy())
    info = {"path": str(out), "vocab": int(f.shape[0]),
            "dim": int(f.shape[1]),
            "seconds": round(time.time() - t0)}
    print(f"[prep-emb32] {info}", flush=True)
    return info


# ── prep: draft router-tap cache ─────────────────────────────────────────────

def step_prep_tap(c: stage0.Corpus, bank: glm_router.RouterBank,
                  work: Path, device: str) -> dict:
    """One pass over ALL shards: tap logits for every labeled row
    (draft hidden of its block row, aux tap mapping), stored per seq as
    committed-row logits [Tc, J, E] f16 + per-block maxpool over all
    labeled rows [nb, J, E] f16 (exactly what the engine consumes:
    committed timeline + manifest-time chunk content)."""
    import torch
    dev = torch.device(device)
    cache = work / "tapcache"
    cache.mkdir(exist_ok=True)
    taps = glm_router.aux_prior_tap(c.moe_layers)        # [J] in 0..4
    W = torch.as_tensor(bank.weight, device=dev).to(torch.bfloat16)
    J, E, H = W.shape
    tap_groups = [(g, np.nonzero(taps == g)[0]) for g in
                  sorted(set(taps.tolist()))]
    acc: dict[int, dict[str, list]] = {}
    t0 = time.time()
    for si, sh in enumerate(c.index["shards"]):
        z = np.load(c.shard_dir / sh["path"])
        lm = z["label_mask"]
        accl = z["accepted_len"]
        anchor = z["anchor_pos"].astype(np.int64)
        skey = z["seq_key"].astype(np.int64)
        feats = z["features_bf16"]                       # [B,G,5,H]
        B, G = lm.shape
        bidx, kidx = np.nonzero(lm)
        if not len(bidx):
            continue
        hid = feats[bidx, kidx]                          # [N,5,H] u16
        ht = torch.from_numpy(
            np.ascontiguousarray(hid).view(np.int16)).to(dev) \
            .view(torch.bfloat16)                        # [N,5,H]
        N = len(bidx)
        out = torch.empty(N, J, E, device=dev, dtype=torch.float16)
        for g, jsel in tap_groups:
            Wg = W[jsel].reshape(-1, H)                  # [Jg*E, H]
            lg = ht[:, g, :] @ Wg.t()                    # [N, Jg*E] bf16
            out[:, jsel, :] = lg.reshape(N, len(jsel), E).half()
        # per-block maxpool over labeled rows
        blk_max = torch.full((B, J * E), float("-inf"), device=dev,
                             dtype=torch.float32)
        bi = torch.as_tensor(bidx, device=dev)
        blk_max.scatter_reduce_(
            0, bi[:, None].expand(N, J * E),
            out.reshape(N, J * E).float(), reduce="amax")
        blk_max = blk_max.reshape(B, J, E).half().cpu().numpy()
        out_np = out.cpu().numpy()
        comm = kidx <= accl[bidx]
        for key in np.unique(skey[bidx]):
            a = acc.setdefault(int(key), {"anchor": [], "pos": [],
                                          "tap": [], "blk_anchor": [],
                                          "blk_max": []})
            rows = np.nonzero((skey[bidx] == key) & comm)[0]
            a["anchor"].append(anchor[bidx[rows]])
            a["pos"].append(anchor[bidx[rows]] + kidx[rows])
            a["tap"].append(out_np[rows])
            brows = np.nonzero(skey == key)[0]
            brows = brows[np.isin(brows, np.unique(bidx))]
            a["blk_anchor"].append(anchor[brows])
            a["blk_max"].append(blk_max[brows])
        if si % 10 == 0:
            print(f"[prep-tap] shard {si}/{len(c.index['shards'])} "
                  f"rows {N} ({time.time()-t0:.0f}s)", flush=True)
    n_rows = 0
    for key, a in acc.items():
        an = np.concatenate(a["anchor"])
        po = np.concatenate(a["pos"])
        tp = np.concatenate(a["tap"])
        ba = np.concatenate(a["blk_anchor"])
        bm = np.concatenate(a["blk_max"])
        order = np.lexsort((po, an))
        an, po, tp = an[order], po[order], tp[order]
        # last-wins dedup on (anchor, pos) (assemble_shards convention)
        kk = (an << 32) | po
        keep = np.ones(len(kk), bool)
        keep[:-1] = kk[1:] != kk[:-1]
        an, po, tp = an[keep], po[keep], tp[keep]
        border = np.argsort(ba, kind="stable")
        ba, bm = ba[border], bm[border]
        bkeep = np.ones(len(ba), bool)
        bkeep[:-1] = ba[1:] != ba[:-1]
        np.savez(cache / f"seq_{key}.npz", anchor=an, pos=po, tap=tp,
                 blk_anchor=ba[bkeep], blk_max=bm[bkeep])
        n_rows += len(po)
    info = {"cache": str(cache), "seqs": len(acc),
            "committed_rows": n_rows,
            "seconds": round(time.time() - t0)}
    print(f"[prep-tap] {info}", flush=True)
    return info


class TapProvider:
    """Joins the tap cache to a sequence's committed timeline + block
    list: returns (tap_cells [T, J, E] f16, tap_blocks [nb, J, E] f16)
    aligned with s['pos'] and s['blocks'] (join key (anchor, pos) —
    duplicate positions across speculative re-drafts are disambiguated
    by the block anchor)."""

    def __init__(self, cache: Path):
        self.cache = Path(cache)

    def for_seq(self, key: int, s: dict):
        z = np.load(self.cache / f"seq_{key}.npz")
        fk = (z["anchor"].astype(np.int64) << 32) | z["pos"].astype(
            np.int64)
        anchors = np.asarray([b["anchor"] for b in s["blocks"]],
                             np.int64)
        want = (anchors[s["chunk"]] << 32) | s["pos"].astype(np.int64)
        loc = np.searchsorted(fk, want)
        if np.any(loc >= len(fk)) or np.any(fk[np.minimum(
                loc, len(fk) - 1)] != want):
            raise KeyError(f"tapcache seq {key}: committed row missing")
        ba = z["blk_anchor"].astype(np.int64)
        bloc = np.searchsorted(ba, anchors)
        if np.any(bloc >= len(ba)) or np.any(ba[np.minimum(
                bloc, len(ba) - 1)] != anchors):
            raise KeyError(f"tapcache seq {key}: block missing")
        return z["tap"][loc], z["blk_max"][bloc]


# ── run / finalize ───────────────────────────────────────────────────────────

def finalize(c: stage0.Corpus, p: dict) -> dict:
    variants = p["variants"]
    out: dict = {"pool": {}, "manifest": {}, "hedge": {},
                 "counts": {"pool_true": p["pool_true"],
                            "addr_n": p["addr_n"],
                            "novel_n": p["novel_n"]}}
    ph = np.asarray(p["pool_hit"], np.float64)
    ah = np.asarray(p["addr_hit"], np.float64)
    nh = np.asarray(p["novel_hit"], np.float64)
    for i, v in enumerate(variants):
        out["pool"][v] = {}
        for ki, k in enumerate(POOL_KS):
            out["pool"][v][f"k{k}"] = {
                "coverage": ph[i, ki] / max(1, p["pool_true"]),
                "pool_size": c.K + k,
                "addressable_recovery": ah[i, ki] / max(1, p["addr_n"]),
                "novel_recovery": nh[i, ki] / max(1, p["novel_n"]),
            }
    mf = p["man_free"]
    for alt, (cov, size, n) in mf.items():
        out["manifest"][alt] = {"cov_deep": cov / max(1, n),
                                "avg_size": size / max(1, n)}
    base = out["manifest"]["prev_union"]["avg_size"]
    for d in MANIFEST_DS:
        cov, n = p["man"][str(d)]
        out["manifest"][f"combined_topup{d}"] = {
            "cov_deep": cov / max(1, n), "avg_size": base + d}
        cov_o, n_o = p["man_open"][str(d)]
        if n_o:
            out["manifest"][f"open_topup{d}"] = {
                "cov_deep": cov_o / max(1, n_o), "avg_size": base + d}
        for src, tag in (("man_head", "head_topup"),
                         ("man_head_open", "head_open_topup")):
            if p.get(src):
                cov_h, n_h = p[src][str(d)]
                if n_h:
                    out["manifest"][f"{tag}{d}"] = {
                        "cov_deep": cov_h / max(1, n_h),
                        "avg_size": base + d}
    out["hedge"]["members"] = p["members"]
    out["hedge"]["w_mean"] = p["hedge_w_mean"]
    out["hedge"]["w_deep"] = p["hedge_w_deep"]
    out["feat_names"] = p["feat_names"]
    if p.get("memo2_support") is not None:
        out["memo2_support"] = p["memo2_support"]
    if p.get("pi_hat") is not None:
        out["pi_hat"] = p["pi_hat"]
    if p.get("novel_splits"):
        out["novel_splits"] = {
            sp: {"coverage": v[0] / max(1, p["pool_true"]),
                 "addressable_recovery": v[1] / max(1, p["addr_n"]),
                 "novel_recovery": v[2] / max(1, p["novel_n"])}
            for sp, v in p["novel_splits"].items()}
    if p.get("gamma_final") is not None:
        out["gamma_final"] = p["gamma_final"]
        out["gamma_traj"] = p["gamma_traj"]
    return out


def summarize_experiment(name: str, ev: dict) -> dict:
    pool = ev["pool"]

    def cov(v, k=24):
        return pool[v][f"k{k}"]["coverage"] if v in pool else None

    s = {
        "pool32_combined": cov("combined"),
        "pool32_rls_online": cov("rls_online"),
        "addr_rec32_rls": pool["rls_online"]["k24"][
            "addressable_recovery"],
        "manifest_topup32":
            ev["manifest"]["combined_topup32"]["cov_deep"],
        "hedge_w_deep": dict(zip(ev["hedge"]["members"],
                                 [round(x, 4) for x in
                                  ev["hedge"]["w_deep"]])),
    }
    for v in ("rls_open", "combined_open", "rls_soft", "rls_dr",
              "rls_drband", "rls_band", "attn_open", "memo", "tap",
              "memo_open", "tap_open", "trans2", "xprev_up",
              "xprev_dn", "gfreq"):
        if v in pool:
            s[f"pool32_{v}"] = cov(v)
            s[f"novel_rec32_{v}"] = pool[v]["k24"]["novel_recovery"]
            s[f"addr_rec32_{v}"] = pool[v]["k24"][
                "addressable_recovery"]
    if "open_topup32" in ev["manifest"]:
        s["manifest_open_topup32"] = \
            ev["manifest"]["open_topup32"]["cov_deep"]
    for tag in ("head_topup32", "head_open_topup32"):
        if tag in ev["manifest"]:
            s[f"manifest_{tag}"] = ev["manifest"][tag]["cov_deep"]
    for sp, v in (ev.get("novel_splits") or {}).items():
        s[f"pool32_split{sp}"] = v["coverage"]
        s[f"novel_rec32_split{sp}"] = v["novel_recovery"]
        s[f"addr_rec32_split{sp}"] = v["addressable_recovery"]
    if "memo2_support" in ev:
        s["memo2_support_mean"] = ev["memo2_support"]["mean"]
        s["memo2_frac_lt8"] = ev["memo2_support"]["frac_lt8"]
    if "pi_hat" in ev:
        s["pi_hat"] = ev["pi_hat"]
    if "gamma_final" in ev:
        g = np.abs(np.asarray(ev["gamma_final"]))
        s["gamma_abs_mean"] = float(g.mean())
        s["gamma_abs_max"] = float(g.max())
        s["gamma_argmax_layer"] = int(g.argmax())
    cfgd = ev.get("cfg", {})
    if "xsame" in cfgd.get("features", ()):
        # METRIC DISCIPLINE (ARCHITECT_TRICK.md): xsame runs score
        # pools under the PIPELINED protocol — their pool/novel/addr
        # numbers live under pipelined{lag}_* keys and must never be
        # compared against (or crowned over) the strict bars.
        lag = cfgd.get("xsame_lag", 0)
        s = {(f"pipelined{lag}_{k}"
              if k.startswith(("pool32_", "novel_rec32_",
                               "addr_rec32_")) else k): v
             for k, v in s.items()}
    return s


def run_experiment(name: str, spec: dict, c: stage0.Corpus, sigma,
                   bank, work: Path, a) -> dict:
    from elb_train import stagecprime_gpu
    default_b = int(str(a.memo_buckets).split(",")[0])
    cfg = ExpCfg(name=name, **{"memo_buckets": default_b, **spec})
    tap_provider = None
    lut = None
    if "tap" in cfg.features:
        tap_provider = TapProvider(work / "tapcache")
    if "memo" in cfg.features:
        lut_path = work / f"embed_lut_b{cfg.memo_buckets}.npz"
        if not lut_path.is_file():     # lazy auto-prep (e2e replay)
            step_prep_kmeans(a.router_ckpt, work, a.device,
                             cfg.memo_buckets,
                             embed_tensor=a.embed_tensor)
        lut = np.load(lut_path)["lut"]
    attn_lut = None
    if cfg.attn_mode:
        ap_ = work / f"embed_lut_b{ATTN_LUT_BUCKETS}.npz"
        if not ap_.is_file():
            step_prep_kmeans(a.router_ckpt, work, a.device,
                             ATTN_LUT_BUCKETS,
                             embed_tensor=a.embed_tensor)
        attn_lut = np.load(ap_)["lut"]
    ctx_emb = None
    if "ctx" in cfg.features:
        cp_ = work / "embed_ctx32.npz"
        if not cp_.is_file():     # lazy auto-prep (e2e replay)
            step_prep_emb32(a.router_ckpt, work, a.device,
                            embed_tensor=a.embed_tensor)
        ctx_emb = np.load(cp_)["emb"].astype(np.float32)
    held = sorted(int(s) for s in c.held_keys)
    t0 = time.time()
    # the champion experiment ALWAYS leaves its deployment export in
    # the workdir (stageb leaves probe_*.npy, stagec ridge_evictor.npz)
    want_export = getattr(a, "export", False) \
        or name == getattr(a, "package_exp", "")
    export_dir = (work / "export" / name) if want_export else None
    evict_keys = set(held[: a.evict_max_seqs]) \
        if getattr(a, "evict_max_seqs", 0) else set()
    partial = stagecprime_gpu.gpu_eval_seqs(
        work, c, sigma, bank.bias, held, a.deep_lo, a.deep_hi,
        c.train_keys, cfg, device=a.device, tap_provider=tap_provider,
        lut=lut, export_dir=export_dir, evict_keys=evict_keys,
        attn_lut=attn_lut, ctx_emb=ctx_emb)
    ev = finalize(c, partial)
    if partial.get("evict_dump"):
        ev["evict_dump"] = partial["evict_dump"]
    ev["seconds"] = round(time.time() - t0)
    ev["cfg"] = {"features": list(cfg.features),
                 "members": list(cfg.members), "open_ch": cfg.open_ch,
                 **({"xsame_lag": cfg.xsame_lag}
                    if "xsame" in cfg.features else {})}
    ev["summary"] = summarize_experiment(name, ev)
    print(f"[stagec'] {name} DONE {ev['seconds']}s: "
          f"{json.dumps(ev['summary'], indent=1)}", flush=True)
    return ev


# ── package (named-model export) ─────────────────────────────────────────────

_ST_DTYPE = {np.dtype(np.float32): "F32", np.dtype(np.float64): "F64",
             np.dtype(np.uint8): "U8", np.dtype(np.uint16): "U16",
             np.dtype(np.int32): "I32", np.dtype(np.int64): "I64"}


def write_safetensors(path: Path, tensors: dict[str, np.ndarray],
                      fp8_names: frozenset = frozenset()) -> None:
    """Minimal safetensors writer (mirror of glm_router's reader).
    Tensors in `fp8_names` must be uint8 arrays holding raw
    float8_e4m3 bits — stored with dtype F8_E4M3."""
    import struct
    header: dict = {}
    blobs: list[bytes] = []
    off = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        dt = "F8_E4M3" if name in fp8_names else _ST_DTYPE[arr.dtype]
        if name in fp8_names and arr.dtype != np.uint8:
            raise ValueError(f"{name}: fp8 tensors must be u8 bits")
        blob = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(blob)]}
        off += len(blob)
        blobs.append(blob)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


def _fp8_bits(x: np.ndarray, reduce_dims: tuple
              ) -> tuple[np.ndarray, np.ndarray]:
    """(u8 e4m3 bits, f32 scales) with amax scaling over reduce_dims
    (CPU torch; mirrors stagecprime_gpu.fp8_roundtrip)."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x, np.float32))
    amax = t.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-30)
    scale = amax / 448.0
    bits = (t / scale).to(torch.float8_e4m3fn).view(torch.uint8)
    return bits.numpy(), scale.squeeze().float().numpy()


def build_fp8_bundle(ed: Path, dest: Path) -> dict:
    """Supplementary fp8 bundle model.fp8.safetensors from the f32
    export: e4m3 tables + f32 scales as tensors; ridge weights kept
    f32 (tiny). Scale granularity matches the certified x9_fp8 run:
    per-layer for tables/memo counts, per-layer-row for nothing (ridge
    not quantized here)."""
    tensors: dict = {}
    fp8_names = set()
    rz = np.load(ed / "ridge.npz")
    for k in rz.files:                   # every head, f32 (tiny)
        if k != "features":
            tensors[f"ridge.{k}"] = rz[k].astype(np.float32)
    tz = np.load(ed / "trans.npz")
    bits, sc = _fp8_bits(tz["trans"], (1, 2))
    tensors["trans.fp8"] = bits
    tensors["trans.scale"] = sc
    fp8_names.add("trans.fp8")
    if (ed / "xprev.npz").is_file():
        xz = np.load(ed / "xprev.npz")
        for k in xz.files:
            bits, sc = _fp8_bits(xz[k], (1, 2))
            tensors[f"xprev.{k}.fp8"] = bits
            tensors[f"xprev.{k}.scale"] = sc
            fp8_names.add(f"xprev.{k}.fp8")
    if (ed / "xsame.npz").is_file():
        sz = np.load(ed / "xsame.npz")
        bits, sc = _fp8_bits(sz["xsame"], (1, 2))
        tensors["xsame.fp8"] = bits
        tensors["xsame.scale"] = sc
        tensors["xsame.lag"] = np.atleast_1d(
            sz["lag"]).astype(np.int64)
        fp8_names.add("xsame.fp8")
    cal_src = (dest / "calibration.npz"
               if (dest / "calibration.npz").is_file()
               else ed / "calibration.npz")   # dest copy carries the
    #                                           §7.2 sidecars
    if cal_src.is_file():
        # π̂ tables stay f32 (77 KB — ARCHITECT_CALIBRATION §4);
        # sidecar companions ride along f32 as well
        kz = np.load(cal_src, allow_pickle=True)
        for k in kz.files:
            a = kz[k]
            if a.dtype.kind in "US":
                continue                 # status strings stay in npz
            tensors[f"calibration.{k}"] = np.atleast_1d(
                a.astype(np.float64 if a.dtype.kind == "f"
                         else a.dtype))
    for stem, pref in (("memo_sparse", "memo"),
                       ("memo2_sparse", "memo2")):
        if (ed / f"{stem}.npz").is_file():
            mz = np.load(ed / f"{stem}.npz")
            tensors[f"{pref}.seen"] = mz["seen"].astype(np.int64)
            tensors[f"{pref}.top_ids"] = mz["top_ids"]
            bits, sc = _fp8_bits(
                mz["top_counts"].astype(np.float32), (0, 2))
            tensors[f"{pref}.top_counts.fp8"] = bits
            tensors[f"{pref}.top_counts.scale"] = sc
            fp8_names.add(f"{pref}.top_counts.fp8")
    if (ed / "ctx_encoder.npz").is_file():
        cz = np.load(ed / "ctx_encoder.npz")
        for k in cz.files:
            bits, sc = _fp8_bits(cz[k].astype(np.float32),
                                 tuple(range(cz[k].ndim)))
            tensors[f"ctx.{k}.fp8"] = bits
            tensors[f"ctx.{k}.scale"] = np.atleast_1d(sc)
            fp8_names.add(f"ctx.{k}.fp8")
    pca = ed.parent.parent / "embed_ctx32.npz"
    if (ed / "ctx_encoder.npz").is_file() and pca.is_file():
        pz = np.load(pca)
        feats_p = pz[pz.files[0]].astype(np.float32)
        bits, sc = _fp8_bits(feats_p, (0, 1))
        tensors["ctx.pca_features.fp8"] = bits
        tensors["ctx.pca_features.scale"] = np.atleast_1d(sc)
        fp8_names.add("ctx.pca_features.fp8")
    out = dest / "model.fp8.safetensors"
    write_safetensors(out, tensors, frozenset(fp8_names))
    return {"path": str(out), "tensors": len(tensors),
            "bytes": out.stat().st_size}

FEATURE_RULES = {
    "in_prev": "1 if expert in previous position's top-8 (reset/pos)",
    "trail16": "route count over trailing 16 positions (ring)",
    "trail64": "route count over trailing 64 positions (ring)",
    "recency": "positions since expert last routed, cap 512",
    "trans": "sum over p in top8(t-1) of T[j,p,e]; T decays 0.99/pos "
             "(lazy scale), +1 per (prev,cur) pair",
    "trans_rank": "rank of `trans` among trail-64 candidates / (n-1)",
    "ema2": "bias-corrected EMA of sel scores, half-life 2 positions",
    "ema8": "bias-corrected EMA of sel scores, half-life 8 positions",
    "g_last": "previous position's sel score",
    "margin_last": "g_last - previous rank-8 sel cut",
    "drift1": "g_last - sel score two positions back",
    "layer_f": "layer index / (J-1)",
    "xprev_up": "sum over p in top8(t-1) at layer j+1 of XU[j,p,e]; "
                "XU decays 0.99/pos, +1 per pair (last layer: 0)",
    "xprev_dn": "sum over p in top8(t-1) at layer j-1 of XD[j,p,e]; "
                "XD decays 0.99/pos, +1 per pair (first layer: 0)",
    "memo": "M[j, lut[token_t], :] / (1 + row_sum) — per-token expert "
            "distribution; M = integer counts, +1 per routing, NO "
            "decay, cross-sequence (updates free at every verify)",
}


def _has_recur(ed: Path) -> bool:
    return "w_recur" in np.load(ed / "ridge.npz").files


def _sg(s: dict, cfgd: dict, stem: str):
    """Prefix-aware summary lookup: xsame configs store pool/novel/
    addr under pipelined{lag}_* names (metric quarantine)."""
    if "xsame" in cfgd.get("features", ()):
        v = s.get(f"pipelined{cfgd.get('xsame_lag', 0)}_{stem}")
        if v is not None:
            return v
    return s.get(stem)


def fit_pihat_sidecars(series_path: Path, rho: float = 0.7) -> dict:
    """§7.2 UNCERTIFIED companion sidecars, fit offline on the run's
    OWN pihat series dump (stream-separated by construction): (i)
    EWMA(rho) of realized per-position miss count M; (ii) x19b-lite
    geometry ridge M_hat_cond = w . (1, EWMA[M], T1, T2), decayed
    closed-form fit on the first half of the held series,
    standardization params shipped, spearman on the second half
    recorded for the label. Never part of the certified surface."""
    z = np.load(series_path)
    m = z["mreal"].astype(np.float64)
    t1 = z["t1"].astype(np.float64)
    t2 = z["t2"].astype(np.float64)
    n = len(m)
    half = n // 2
    ew = np.empty(n)
    acc = wsum = 0.0
    for t in range(n):
        ew[t] = acc / wsum if wsum > 0 else 0.0   # strictly past
        acc = rho * acc + (1 - rho) * m[t]
        wsum = rho * wsum + (1 - rho)
    X = np.stack([np.ones(n), ew, t1, t2], 1)
    mu = X[:half].mean(0)
    sd = X[:half].std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xs[:, 0] = 1.0
    wdec = 0.999 ** np.arange(half - 1, -1, -1)
    A = (Xs[:half] * wdec[:, None]).T @ Xs[:half] + 1e-3 * np.eye(4)
    b = (Xs[:half] * wdec[:, None]).T @ m[:half]
    w = np.linalg.solve(A, b)

    def _rank(v):
        o = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[o] = np.arange(len(v))
        return r

    sp = float(np.corrcoef(_rank(Xs[half:] @ w),
                           _rank(m[half:]))[0, 1])
    return {"sidecar_rho": np.float64(rho),
            "sidecar_ridge_w": w,
            "sidecar_ridge_mu": mu,
            "sidecar_ridge_sd": sd,
            "sidecar_ridge_features": np.array(
                ["bias", "ewma_m", "t1", "t2"]),
            "sidecar_spearman_2nd": np.float64(sp),
            "sidecar_status": np.array(
                "UNCERTIFIED-COMPANION (ARCHITECT_CALIBRATION "
                "S7.2) — rides free, never the certified surface")}


def step_package(work: Path, results: dict, dest: Path,
                 exp: str) -> dict:
    """Assemble the named model directory from a --export run's
    artifacts + the lut + generated feature-spec/README."""
    import shutil
    from elb_train.stagec import (HEDGE_ETA, RLS_DECAY, RLS_LAMBDA,
                                  RLS_REFRESH)
    from elb_train.stagecprime_gpu import W_EMA_HALF_LIFE  # noqa: F401
    ed = work / "export" / exp
    if not (ed / "ridge.npz").is_file():
        raise FileNotFoundError(f"{ed}: no export — rerun the "
                                f"experiment with --export first")
    ev = results["experiments"][exp]
    s = ev["summary"]
    cfgd = ev["cfg"]
    if "xsame" in cfgd.get("features", ()):
        # resolve the pipelined{lag}_* quarantined names onto the
        # canonical stems for README/spec formatting — the spec
        # carries the metric_protocol label alongside
        s = {**s, **{stem: _sg(s, cfgd, stem)
                     for stem in ("pool32_rls_open",
                                  "novel_rec32_rls_open",
                                  "addr_rec32_rls_open",
                                  "pool32_rls_online",
                                  "addr_rec32_rls",
                                  "pool32_combined")
                     if _sg(s, cfgd, stem) is not None}}
    buckets = int(np.load(ed / "memo_sparse.npz")["buckets"]) \
        if (ed / "memo_sparse.npz").is_file() else None
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in ("ridge.npz", "trans.npz", "xprev.npz",
              "memo_sparse.npz", "memo2_sparse.npz",
              "ctx_encoder.npz", "xsame.npz", "calibration.npz"):
        if (ed / f).is_file():
            shutil.copy2(ed / f, dest / f)
            copied.append(f)
    if (ed / "ctx_encoder.npz").is_file() \
            and (work / "embed_ctx32.npz").is_file():
        shutil.copy2(work / "embed_ctx32.npz",
                     dest / "embed_ctx32.npz")
        copied.append("embed_ctx32.npz")
    if buckets:
        lut_src = work / f"embed_lut_b{buckets}.npz"
        shutil.copy2(lut_src, dest / "embed_lut.npz")
        copied.append("embed_lut.npz")
    feats = [str(x) for x in
             np.load(ed / "ridge.npz")["features"]]
    spec = {
        "model": dest.name,
        "experiment": exp,
        "config": cfgd,
        "features_in_order": feats,
        "feature_update_rules": {f: FEATURE_RULES[f] for f in feats
                                 if f in FEATURE_RULES},
        "constants": {
            "rls_decay_per_position": RLS_DECAY,
            "rls_lambda": RLS_LAMBDA,
            "rls_refresh_positions": RLS_REFRESH,
            "hedge_eta": HEDGE_ETA,
            "pair_table_decay_per_position": 0.99,
            "recency_cap": 512.0,
            "memo_buckets": buckets,
        },
        "scoring_contract": {
            "sel_space": "stable_sigmoid(router_logits) + "
                         "e_score_correction_bias (noaux_tc)",
            **({"recur16_head": {
                "weights": "ridge.w_recur [J, F+1] — second closed-"
                           "form head SHARING the open-channel A "
                           "(differs only in b)",
                "target": "recur16: expert routed again within the "
                          "next 16 committed positions (free delayed "
                          "label — resolve position t once t+16 is "
                          "revealed, from last_seen)",
                "use": "EVICTION protection scorer: prev8 ∪ top-48 "
                       "of X_aug @ w_recur among resident non-prev "
                       "candidates",
                "online_updates": "b_recur from a 16-slot feature "
                                  "ring (b lags shared A by 16 "
                                  "positions ≈ 0.999^16 scale — "
                                  "negligible); re-solve with the "
                                  "same refresh cadence",
            }} if _has_recur(ed) else {}),
            "deployable_scorer": "w_open (rls_open): score all 256 "
                                 "experts, exclude prev-top8, take "
                                 "top-K into the pool next to prev8",
            "online_updates": "A/b decayed sufficient statistics on "
                              "ALL 256 rows per (position, layer); "
                              "closed-form re-solve every "
                              "rls_refresh_positions; labels free at "
                              "every verify (TD-EPM-REJECTED-LABELS)",
            "per_sequence_state_resets": "prev8/trails/EMAs/g_prev "
                                         "reset per request; memo/"
                                         "trans/xprev/ridge stats are "
                                         "cross-request",
        },
        "bars_held_seed35": {
            "pool32_rls_open": _sg(s, cfgd, "pool32_rls_open"),
            "novel_recovery32": _sg(s, cfgd, "novel_rec32_rls_open"),
            "pool32_rls_valid": _sg(s, cfgd, "pool32_rls_online"),
            "addressable_recovery32": _sg(s, cfgd, "addr_rec32_rls"),
            "manifest_topup32": s.get("manifest_topup32"),
        },
    }
    lag = cfgd.get("xsame_lag", 0)
    if "xsame" in cfgd.get("features", ()):
        # METRIC DISCIPLINE travels with the model: the pool bars
        # above are pool@32-PIPELINED(lag) numbers.
        spec["bars_held_seed35"]["metric_protocol"] = \
            f"pipelined({lag}) — NEVER comparable to strict bars"
        spec["pipelined_protocol"] = {
            "xsame_lag": lag,
            "form": "hard: routed top8 of layer j-lag at the "
                    "CURRENT position (xsame_soft=False)",
            "feature": "xsame.npz [J,E,E] decayed pair table; "
                       "feature row j = sum_p in top8(t, j-lag) "
                       "T[j,p,e]; rows j < lag are zeros",
            "metric_names": f"pipelined{lag}_pool32_* / "
                            f"pipelined{lag}_novel_rec32_* / "
                            f"pipelined{lag}_addr_rec32_*",
            "consumer_contract": "the pool for layer j may be "
                "issued only once layer j-lag's routing has landed "
                "at the current position; the strict heads "
                "(w_recur/w_manifest) are ctx+xsame-masked and "
                "remain strict-contract (bit-identical to the "
                "strict ship)",
        }
    if (dest / "calibration.npz").is_file():
        # packaging debt (ARCHITECT_CALIBRATION §4 + §7 rescope)
        spec["pi_hat"] = {
            "source": "w_open open ranking (E-K ranks) + prev8 "
                      "slots by score"
                      + (f" [PIPELINED({lag}) stream — tables are "
                         f"stream-specific, never reuse on the "
                         f"strict stream]" if lag else ""),
            "tables": "rank-conditional decayed hit rates "
                      "(calibration.npz H, P; stationary "
                      "bias-corrected form H/n_cal)",
            "outputs": "coverage curves C(M), expected misses "
                       "m(M)/M-hat, marginal value dC/dM = "
                       "H_j[M-K] (the byte-calculus quantity)",
            "decay": 0.999,
            "bias_correction": "stationary H/n_cal (decay baked "
                               "out at export)",
            "certified_surface": "per-layer marginal functionals "
                "ONLY (ARCHITECT_CALIBRATION §7 rescope); "
                "position-level M-hat is uncertified-long-run-"
                "average — never per-position admission",
        }
        sp_series = work / f"pihat_series_{exp}.npz"
        if sp_series.is_file():
            side = fit_pihat_sidecars(sp_series)
            cal = dict(np.load(dest / "calibration.npz",
                               allow_pickle=True))
            cal.update(side)
            np.savez(dest / "calibration.npz", **cal)
            spec["pi_hat"]["sidecars"] = (
                "UNCERTIFIED companions fit on this run's own "
                "series (§7.2): EWMA(rho=%.2f) miss-pressure + "
                "x19b-lite geometry ridge (calibration.npz "
                "sidecar_* keys; ridge spearman 2nd-half %.4f)"
                % (float(side["sidecar_rho"]),
                   float(side["sidecar_spearman_2nd"])))
            copied.append("calibration.npz (+sidecars)")
    json.dump(spec, open(dest / "feature_spec.json", "w"), indent=1)
    copied.append("feature_spec.json")
    fp8_info = build_fp8_bundle(ed, dest)
    copied.append("model.fp8.safetensors")
    if not (dest / "README.md").is_file():
        (dest / "README.md").write_text(_model_readme(dest.name, exp,
                                                      s, buckets))
        copied.append("README.md")
    else:
        # supersede flow: existing model dir keeps its history and
        # gains a lineage addendum with the new configuration's bars
        with open(dest / "README.md", "a") as f:
            f.write(f"""

## Lineage addendum ({exp})

This directory is SUPERSEDED-IN-PLACE by the `{exp}` configuration
(same feature chassis, extended head family). New bars (held, seed
35): pool@32 {s.get('pool32_rls_open'):.5f}, novel-rec
{s.get('novel_rec32_rls_open'):.5f}, addr-rec
{s.get('addr_rec32_rls_open', 0) or s.get('addr_rec32_rls', 0):.5f},
manifest head-topup32 {s.get('manifest_head_topup32', 0):.5f}
(open {s.get('manifest_head_open_topup32', 0):.5f}). Weights/tables
in this directory now come from `{exp}`'s export; feature_spec.json
is authoritative for the head contracts.
""")
        copied.append("README.md (lineage addendum)")
    write_safetensors_bundle(dest, spec)
    copied.append("model.safetensors")
    info = {"dest": str(dest), "files": copied,
            "export_files": ev.get("export_files")}
    print(f"[stagec'] packaged: {info}", flush=True)
    return info


_ST_DTYPES = {"float64": "F64", "float32": "F32", "float16": "F16",
              "int64": "I64", "int32": "I32", "uint16": "U16",
              "uint8": "U8"}


def write_safetensors_bundle(model_dir: Path, spec: dict) -> Path:
    """Single-file safetensors bundle of every tensor in the model dir
    (the npz files stay — this is the ecosystem-standard load form,
    readable by glm_router.read_safetensors_tensor / any safetensors
    client). Non-tensor scalars and the feature order go into
    __metadata__ (string map, per the format spec)."""
    model_dir = Path(model_dir)
    tensors: list[tuple[str, np.ndarray]] = []
    meta: dict[str, str] = {}
    for stem, names in (("ridge", None), ("memo_sparse", None),
                        ("memo2_sparse", None), ("ctx_encoder", None),
                        ("embed_ctx32", None),
                        ("xprev", None), ("trans", None),
                        ("xsame", None), ("calibration", None),
                        ("embed_lut", None)):
        if not (model_dir / f"{stem}.npz").is_file():
            continue                     # optional stems per config
        z = np.load(model_dir / f"{stem}.npz", allow_pickle=True)
        for k in z.files:
            a = z[k]
            if a.dtype.kind in "US" or a.ndim == 0:
                meta[f"{stem}.{k}"] = (",".join(map(str, a.tolist()))
                                       if a.ndim else str(a.item()))
            else:
                tensors.append((f"{stem}.{k}",
                                np.ascontiguousarray(a)))
    meta["feature_spec"] = json.dumps(
        {k: spec[k] for k in ("features_in_order", "bars_held_seed35",
                              "experiment", "model") if k in spec})
    header: dict = {"__metadata__": meta}
    off = 0
    for name, a in tensors:
        n = a.nbytes
        header[name] = {"dtype": _ST_DTYPES[str(a.dtype)],
                        "shape": list(a.shape),
                        "data_offsets": [off, off + n]}
        off += n
    hjson = json.dumps(header).encode()
    pad = (8 - len(hjson) % 8) % 8          # spec: 8-byte alignment
    hjson += b" " * pad
    out = model_dir / "model.safetensors"
    with open(out, "wb") as f:
        f.write(len(hjson).to_bytes(8, "little"))
        f.write(hjson)
        for _, a in tensors:
            f.write(a.tobytes())
    return out


def _model_readme(name: str, exp: str, s: dict,
                  buckets: int | None) -> str:
    import subprocess
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        rev = "unknown"
    return f"""# {name}

Online expert-prediction ridge for GLM-5.2 routed-MoE layers — the
P11.c' champion ({exp}: 12 stageb history features + cross-layer
prev-position co-activation + per-token expert memo, open all-256
channel). NOT an inference checkpoint: a µs-class side-model that
predicts each layer's next top-8 experts for prefetch/pool/eviction
decisions.

## Bars set (EPM-5 held split, seed 35 — studies/epm/p11-stagecprime)

- pool@32 coverage (prev8 + top-24): **{s.get('pool32_rls_open'):.4f}**
  (P11.c bar was 0.6258)
- NOVEL-slot recovery (true experts outside the trail-64 union —
  unreachable for history-only models): **{s.get(
      'novel_rec32_rls_open'):.4f}**
- addressable recovery {s.get('addr_rec32_rls'):.4f}; manifest
  topup32 {s.get('manifest_topup32'):.4f} (par with P11.c)

## Files

- `ridge.npz` — `w_open` [75, F+1] f64 (THE deployable scorer,
  all-256-expert channel), `w_valid` (candidate-restricted twin),
  `features` (column order; last = bias)
- `memo_sparse.npz` — per-token expert counts, sparse top-16 stamp:
  `seen` token ids, `top_ids` [S, 75, 16] u8, `top_counts` u16
  (integer counts, no decay; buckets={buckets} = per-token identity)
- `xprev.npz` — `xu`/`xd` [75, 256, 256] f32 cross-layer prev-position
  co-activation tables (stationary decayed-count units, scale baked)
- `trans.npz` — same-layer transition table (feeds trans/trans_rank)
- `embed_lut.npz` — token -> bucket lut (identity for this model)
- `feature_spec.json` — feature order, update rules, constants,
  scoring contract, measured bars

## Load + score (offline reference)

    z = np.load('ridge.npz'); w = z['w_open']          # [75, F+1]
    # per (position, layer): build X [256, F] from the feature spec's
    # update rules, score = X_aug @ w[j]; exclude prev-top8; take
    # top-K into the pool next to prev8.

## Online-update contract

Labels are free at every verify (TD-EPM-REJECTED-LABELS): after each
committed position, update memo/trans/xprev counts and the ridge's
decayed sufficient statistics (decay 0.999/position, lambda 1.0,
closed-form re-solve every 64 positions). Per-sequence state (prev8,
trailing windows, EMAs, g_prev) resets per request; memo/trans/xprev/
ridge stats persist across requests — the model never goes
regime-stale by construction. sel-space = stable_sigmoid(logits) +
e_score_correction_bias (noaux_tc; bias from the serving checkpoint).

## Provenance

Corpus /srv/models/epm5-corpus (105,090 blocks / 388 seqs, split seed
35, 302,176 committed rows); checkpoint
/srv/models/lukealonso/GLM-5.2-NVFP4; repo LayerStoRm3 branch fab4
commit {rev}; saga studies/epm/p11-stagecprime (2026-08-14). Warm
state = train split + held pass of that corpus, accumulated online.

## Integration status

Engine integration is GATED: manifests behind the P9 byte bar
(p >= U/S, post-P1 regime), governor/pool decisions behind P2,
eviction behind the +2% wall bar. This package is the banked,
deployment-ready form those triggers consume.
"""


# ── verdict ──────────────────────────────────────────────────────────────────

def evaluate_verdict(results: dict) -> dict:
    exps = results.get("experiments", {})
    out: dict = {"bars": {"pool32": BAR_POOL32,
                          "topup32": BAR_TOPUP32}}
    x0 = exps.get("x0_parity", {}).get("summary")
    if x0:
        out["parity"] = {
            "pool32_combined": x0["pool32_combined"],
            "delta_vs_bar": x0["pool32_combined"] - BAR_POOL32,
            "ok": abs(x0["pool32_combined"] - BAR_POOL32) <= 2e-3,
        }
    def _is_pipelined(ev):
        return ("xsame" in ev.get("cfg", {}).get("features", ())
                or any(k.startswith("pipelined")
                       for k in ev.get("summary", {})))

    table = {}
    for name, ev in exps.items():
        if _is_pipelined(ev):
            continue                  # own section below — never here
        s = ev.get("summary", {})
        row = {"pool32_combined": s.get("pool32_combined"),
               "pool32_rls_online": s.get("pool32_rls_online"),
               "addr_rec32_rls": s.get("addr_rec32_rls"),
               "manifest_topup32": s.get("manifest_topup32")}
        for k, v in s.items():
            if k.startswith("novel_rec32") or k.endswith("_open") \
                    or k.endswith(("_rls_soft", "_rls_dr",
                                   "_rls_drband", "_rls_band")) \
                    or k == "manifest_open_topup32":
                row[k] = v
        table[name] = row
    out["experiments"] = table
    def _peak(s):
        return max(s.get("pool32_combined") or 0.0,
                   s.get("pool32_rls_open") or 0.0,
                   s.get("pool32_rls_soft") or 0.0,
                   s.get("pool32_rls_dr") or 0.0,
                   s.get("pool32_rls_drband") or 0.0,
                   s.get("pool32_rls_band") or 0.0)

    best = max(((_peak(ev["summary"]), name)
                for name, ev in exps.items()
                if "summary" in ev and not _is_pipelined(ev)),
               default=(None, None))
    out["best"] = {"experiment": best[1], "pool32": best[0]}
    if best[0] is not None:
        out["beats_bar"] = best[0] > BAR_POOL32
        bs = exps[best[1]]["summary"]
        out["best"]["novel_rec32"] = bs.get("novel_rec32_rls_open")
        out["best"]["manifest_topup32"] = bs.get("manifest_topup32")
    # ── pipelined section (ARCHITECT_TRICK.md P2/kill computation):
    # Δ-indexed deltas vs the STRICT same-stack baseline (x18ship2).
    # Never merged with best/beats_bar above.
    base = exps.get("x18ship2", {}).get("summary", {})
    pipe: dict = {}
    for name, ev in exps.items():
        if not _is_pipelined(ev) or "summary" not in ev:
            continue
        s = ev["summary"]
        lag = ev.get("cfg", {}).get("xsame_lag", 0)
        pre = f"pipelined{lag}_"

        def _g(stem, s=s, pre=pre):
            return s.get(pre + stem, s.get(stem))

        row = {"lag": lag,
               "pool32_pipelined": _g("pool32_rls_open"),
               "novel_rec32_pipelined": _g("novel_rec32_rls_open"),
               "addr_rec32_pipelined": _g("addr_rec32_rls_open")}
        if base and row["pool32_pipelined"] is not None:
            row["d_pool_pp"] = 100 * (row["pool32_pipelined"]
                                      - base["pool32_rls_open"])
            row["d_novel_pp"] = 100 * (row["novel_rec32_pipelined"]
                                       - base["novel_rec32_rls_open"])
            row["d_addr_pp"] = 100 * (row["addr_rec32_pipelined"]
                                      - base["addr_rec32_rls_open"])
        pipe[name] = row
    if pipe:
        lags = {r["lag"]: r for r in pipe.values()
                if r.get("d_pool_pp") is not None}
        gates: dict = {"baseline": "x18ship2 (strict bars)"}
        if 8 in lags:
            gates["p2_fires"] = bool(
                lags[8]["d_pool_pp"] >= 1.0
                and lags[8]["d_addr_pp"] > 0)
            gates["kill_fires"] = bool(lags[8]["d_pool_pp"] < 0.3)
        ds = [lags[d]["d_pool_pp"] for d in sorted(lags)]
        if len(ds) >= 2:
            gates["monotone_in_lag"] = bool(
                all(a >= b for a, b in zip(ds, ds[1:])))
        out["pipelined"] = {"experiments": pipe, "gates": gates}
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--routing", required=True)
    ap.add_argument("--split-seed", type=int, default=35)
    ap.add_argument("--held-fraction", type=float, default=0.25)
    ap.add_argument("--do", default="run,verdict",
                    help="comma list: prep-kmeans, prep-tap, run, "
                         "verdict")
    ap.add_argument("--exp", default="all",
                    help="experiment names (comma list) or 'all'")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--router-ckpt", default=stage0.DEFAULT_ROUTER_CKPT)
    ap.add_argument("--embed-tensor", default=DEFAULT_EMBED_TENSOR,
                    help="embedding tensor name in the checkpoint "
                         "(model-specific knob)")
    ap.add_argument("--memo-buckets", default="4096",
                    help="bucket count for --exp runs without a "
                         "per-spec value; prep-kmeans accepts a comma "
                         "list; >= vocab -> identity lut (no k-means)")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip experiments already in results.json "
                         "(pipeline resume)")
    ap.add_argument("--export", action="store_true",
                    help="run step: also export deployment-warm state "
                         "(ridge weights + sparse memo + pair tables) "
                         "to <workdir>/export/<exp>/")
    ap.add_argument("--package-dir", default="",
                    help="package step: destination model directory")
    ap.add_argument("--export-configs", default="",
                    help="run step: comma list of experiment names "
                         "that must end the step WITH deployment "
                         "exports (run+export any that lack one)")
    ap.add_argument("--publish-map", default="",
                    help="package step: comma list exp=dir — publish "
                         "each experiment's export to its model dir "
                         "(existing README gets a lineage addendum)")
    ap.add_argument("--package-exp", default="x9b_recur16",
                    help="the certified multi-head champion: x7g pool "
                         "behavior (bit-exact) + the recur16 eviction "
                         "head")
    ap.add_argument("--evict-max-seqs", type=int, default=0,
                    help="run step: dump w_open evict protection for "
                         "the first N held seqs (stagec schema; "
                         "open-channel experiments only)")
    ap.add_argument("--evict-caps", default="700,1052")
    ap.add_argument("--features", default="",
                    help="custom experiment: comma feature list "
                         "(with --members/--open under --exp custom)")
    ap.add_argument("--members", default="")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--sigma", default="build/recur/p11-stage0/sigma.json")
    ap.add_argument("--deep-lo", type=int, default=58)
    ap.add_argument("--deep-hi", type=int, default=77)
    ap.add_argument("--limit-seqs", type=int, default=0)
    ap.add_argument("--stageb-workdir", default="build/recur/p11-stageb")
    ap.add_argument("--workdir", default="build/recur/p11-stagecprime")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    steps = [s.strip() for s in a.do.split(",") if s.strip()]
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    for fn in ("probe_full.npy", "router_bank.npz"):
        src = Path(a.stageb_workdir) / fn
        dst = work / fn
        if not dst.is_file():
            dst.write_bytes(src.read_bytes())

    c = stage0.Corpus(a.shards, a.routing, a.split_seed,
                      a.held_fraction)
    if a.limit_seqs:
        c.train_keys = c.train_keys[: max(2, a.limit_seqs * 3)]
        c.held_keys = c.held_keys[: a.limit_seqs]
    sigma = np.asarray(json.load(open(a.sigma))["sigma"], np.float32)
    bank = glm_router.RouterBank.from_npz(work / "router_bank.npz")

    rpath = work / "results.json"
    results: dict = json.load(open(rpath)) if rpath.is_file() else {}
    results.setdefault("experiments", {})
    results["config"] = vars(a)

    def _checkpoint():
        # merge-on-write: concurrent experiment processes share this
        # file — take on-disk entries for experiments this process did
        # not (re)run, so parallel single-experiment runs don't clobber
        # each other (in-memory wins for names run here).
        if rpath.is_file():
            try:
                disk = json.load(open(rpath))
                for name, ev in disk.get("experiments", {}).items():
                    results["experiments"].setdefault(name, ev)
                for k in ("prep_kmeans", "prep_tap"):
                    if k in disk and k not in results:
                        results[k] = disk[k]
            except json.JSONDecodeError:
                pass
        json.dump(results, open(rpath, "w"), indent=1)

    if "prep-kmeans" in steps:
        results["prep_kmeans"] = [
            step_prep_kmeans(a.router_ckpt, work, a.device, int(b),
                             embed_tensor=a.embed_tensor)
            for b in str(a.memo_buckets).split(",") if b]
        _checkpoint()
    if "prep-tap" in steps:
        results["prep_tap"] = step_prep_tap(c, bank, work, a.device)
        _checkpoint()
    if "run" in steps:
        if a.exp == "custom":
            todo = {"custom": dict(
                features=tuple(f for f in a.features.split(",") if f),
                members=tuple(m for m in a.members.split(",") if m),
                open_ch=a.open)}
        elif a.exp == "all":
            todo = EXPERIMENTS
        else:
            todo = {n: EXPERIMENTS[n] for n in a.exp.split(",")}
        for name, spec in todo.items():
            if a.skip_done and name in results["experiments"]:
                print(f"[stagec'] {name} already done — skipped "
                      f"(--skip-done)", flush=True)
                continue
            results["experiments"][name] = run_experiment(
                name, spec, c, sigma, bank, work, a)
            _checkpoint()
        for name in [x for x in a.export_configs.split(",") if x]:
            if (work / "export" / name / "ridge.npz").is_file() \
                    and name in results["experiments"]:
                continue                 # export already present
            a.export = True
            results["experiments"][name] = run_experiment(
                name, EXPERIMENTS[name], c, sigma, bank, work, a)
            _checkpoint()
    if "evict" in steps:
        # CPU sims (stage0.run_evict, process-parallel): the standing
        # GPU-only rule is waived per explicit user authorization for
        # the evictor certification pair.
        exp = a.package_exp
        dump = (results["experiments"].get(exp) or {}).get(
            "evict_dump") or str(
            work / f"preds_stagecprime_{exp}.npz")
        if isinstance(dump, dict):     # multi-scorer dumps
            dmap = {f"stagecprime_{exp}_{k}": v
                    for k, v in dump.items()}
        else:
            dmap = {f"stagecprime_{exp}": dump}
        for pth in dmap.values():
            if not Path(pth).is_file():
                raise FileNotFoundError(
                    f"{pth}: no evict dump — rerun with "
                    f"--evict-max-seqs first")
        caps = [int(x) for x in a.evict_caps.split(",")]
        cert = stage0.run_evict(dmap, caps)
        prev = results.get("evict_cert")
        if prev and "rows" in prev:      # accumulate across arms
            cert["rows"] = {**prev["rows"], **cert["rows"]}
        results["evict_cert"] = cert
        _checkpoint()
    if "package" in steps:
        if a.publish_map:
            results["package"] = {}
            for pair in a.publish_map.split(","):
                exp, dest = pair.split("=", 1)
                results["package"][exp] = step_package(
                    work, results, Path(dest), exp)
                _checkpoint()
        else:
            if not a.package_dir:
                raise SystemExit("--package-dir or --publish-map "
                                 "required for package")
            results["package"] = step_package(
                work, results, Path(a.package_dir), a.package_exp)
            _checkpoint()
    if "verdict" in steps:
        # teacher remedy (b): entries recorded under the pre-fix
        # module lack cfg.xsame_lag (gates would collapse lags to 0)
        # — inject the lag from the registry and re-summarize, which
        # also normalizes stored keys to pipelined{lag}_*. Idempotent
        # for post-fix entries.
        for n, ev in results["experiments"].items():
            cfgd = ev.get("cfg", {})
            if "xsame" in cfgd.get("features", ()) \
                    and n in EXPERIMENTS and "pool" in ev:
                cfgd.setdefault(
                    "xsame_lag", EXPERIMENTS[n].get("xsame_lag", 0))
                ev["summary"] = summarize_experiment(n, ev)
        results["verdict"] = evaluate_verdict(results)
        print(json.dumps(results["verdict"], indent=1), flush=True)
        _checkpoint()
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1)
    print(f"[stagec'] wrote {rpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

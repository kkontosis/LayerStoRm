"""Engine-current constraint registry (AUTOCONFIG §4) — DATA, not logic.

User directive (2026-08-30): rules of the form "X and Y are incompatible in
the engine TODAY" must live where a person can find, read, and DELETE them
in one edit when the underlying gate changes — with the ticket whose
resolution removes them, and a marking that they are engine-version facts,
not physics. The solver READS this table; deleting a row re-derives the
config with the rule gone, without touching solver logic.

Physics (per-GPU VRAM, NUMA capacity, the Gen3-x4 disk ceiling, model
geometry) is NOT here — it is baked into sizing.py / hwdetect.py and no
commit deletes it.

Each row: id, statement (present tense, what the engine DOES), ticket
(what would remove or change it), scope (arch filter, "" = all), and kind:
  engine_gate  — a literal code gate; grep the cited site before trusting
  measured     — a measured policy fact with provenance; re-measure to change
  validator    — a config_validator/invariant rule
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintRow:
    id: str
    kind: str          # engine_gate | measured | validator
    statement: str
    ticket: str        # "" when none exists yet
    scope: str = ""    # architecture filter; "" = all archs
    site: str = ""     # code/spec citation


ENGINE_CONSTRAINTS: tuple[ConstraintRow, ...] = (
    ConstraintRow(
        id="v4-no-sharded-kv",
        kind="engine_gate",
        statement="deepseek_v4 rejects dcp_kv_mode=sharded (replicated only).",
        ticket="TD-V4-DCP-KV",
        scope="deepseek_v4",
        site="src/core/memory/vram_allocator.cpp:358-365",
    ),
    ConstraintRow(
        id="v4-max-pages-auto-only",
        kind="engine_gate",
        statement="deepseek_v4 requires memory.kv_cache.max_pages_per_gpu='auto'.",
        ticket="",
        scope="deepseek_v4",
        site="src/core/memory/vram_allocator.cpp:368-373",
    ),
    ConstraintRow(
        id="dspark-ctx-cap",
        kind="engine_gate",
        statement=(
            "The dspark draft context arena is a fixed carve of "
            "draft_context_capacity_tokens (default 8192, ~0.16 GiB/1k "
            "tokens); over-cap prompts route to the plain arm at start and "
            "degrade via INV-SERVE-SPEC-FALLBACK mid-request. Do not size "
            "the draft as if it helps at every context."),
        ticket="TD-DSPARK-CTX-POLICY",
        site="src/speculation/dspark_runtime.cpp:158-238",
    ),
    # (replaced 2026-09-01: row kda-state-fixed-carve described the
    #  dedicated carve that TD-KDA-STATE-MAPPED-SLABS retired when mapped
    #  became the default on 2026-08-31 — the row outliving its ticket is
    #  TD-AUTOCONFIG-STALE-KDA-CARVE-ROW, and the stale model it fed the
    #  solver is TD-AUTOCONFIG-MAXSEQ-IGNORES-MAPPED-KDA. The carve
    #  survives only as the off-path: LS_KDA_STATE_MAPPED=0 /
    #  _internal-kda_state.mapped=false.)
    ConstraintRow(
        id="kda-state-mapped-tenant",
        kind="engine_gate",
        statement=(
            "glm5_next KDA recurrent state is a MAPPED TENANT of the shared "
            "kv_main slab pool (default since 2026-08-31): per-(request, "
            "linear layer) contiguous whole-slab runs claimed at seq_create "
            "— the per-request charge is the SLAB-PADDED one (150.2 "
            "MiB/request on GLM-5.3-Flash vs the 145.6 MiB slot), and the "
            "policy share rides IN kv_main as shared, lendable capacity, "
            "not a dedicated region. Consequence the solver must model: "
            "KV(max_seq) + full-length indexer reservation + mapped state "
            "all draw on the ONE pool, so the pool must hold their SUM for "
            "one max-length request before max_sequence_length may be "
            "advertised (the engine logs its own admissible ceiling at "
            "boot; enginecheck compares). Carve survives as the off-path "
            "(LS_KDA_STATE_MAPPED=0). Deleting this row reverts the solver "
            "to the carve model."),
        ticket="TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA",
        scope="glm5_next",
        site="src/core/memory/vram_allocator.cpp (mapped share + admissible "
             "ceiling); src/core/memory/page_allocator.cpp:150-230",
    ),
    # (deleted 2026-08-30: row kv-pool-counts-all-layers — GF3.9 fixed
    #  TD-KV-POOL-SIZED-OVER-ALL-LAYERS; the sizer now uses num_kv_layers()
    #  + nextn, which modelshape.engine_kv_pool_layers() mirrors. This is
    #  the registry mechanism working as designed: the ticket landed, the
    #  row died, the solver re-derived.)
    ConstraintRow(
        id="glm5next-tiering-default-off",
        kind="measured",
        statement=(
            "glm5_next KV tiering ships default OFF: at 25k its KV+indexer "
            "costs ~0.14 GiB vs GLM-5.2's 1.24 GiB (8.8x less KV/token, 11 "
            "KV-bearing layers), so tiering buys nothing at champion "
            "context. Re-decide on measurement at >=256k. P-31 step 1 "
            "(2026-09-06): OFF stays the DEFAULT CANDIDATE; the E1 lattice "
            "now escalates to ON for CAPACITY when the untiered pool "
            "cannot hold the conc x max_seq ask (conc-ask-exceeds-"
            "untiered-pool; INV-KVT-16 windowed admission) — exactly the "
            ">=256k regime this row deferred. Delete this row and the "
            "solver weighs tiering by bytes again with no escalation "
            "machinery in the way."),
        ticket="",
        scope="glm5_next",
        site="spec/SPEC_UPDATES.md:668 (GF3.9, 2026-08-30)",
    ),
    ConstraintRow(
        id="glm5next-tp1-default",
        kind="measured",
        scope="glm5_next",
        statement=(
            "glm5_next serves at TP=1 by default. The KDA decode kernel is "
            "LATENCY-floored, not width-bound: kda_decode_step 6.07 us at "
            "H=64 (TP1) vs 5.89 us at H=32/rank (TP2) — halving the heads "
            "buys ~nothing — while the bf16 combine costs ~12 us/layer "
            "intrinsic (median) and 45 combines/token make ~0.5-2 ms/token, "
            "i.e. ~0.5-2% of a 10 tok/s serving wall, plus rank-skew "
            "exposure. TP=2 is CORRECT and stays available for VRAM "
            "headroom: take it for CAPACITY, never for speed. EVIDENCE "
            "BOUND: prefill was measured FLAT at ONE 141-token shape only "
            "(superchunk 35.4 s TP1 vs 34.8 s TP2; chunked 40.2 vs 42.9 s) "
            "— that is weak evidence at champion-scale prefill, and a "
            "large-prefill measurement is what would falsify this row. "
            "LARGE-PREFILL ADDENDUM (P-31 step 1, 2026-09-06): P-30 steps "
            "1-4 measured attention as the MAJORITY of the champion-scale "
            "fresh-prefill wall (60.4% at S=2048, ~73% at S=6144, "
            "stride-invariant, measured at tp=2 with heads already "
            "sharded) — head-sharding is the only lever touching that "
            "section, so for deep-context asks (max_sequence_length >= "
            "262144, the tiering row's own re-decide bound) the solver "
            "orders the head-divisible ceiling FIRST and TP=1 stays the "
            "fallback (solver.tp_plan). The tp1 decode default is "
            "unchanged for short/decode-dominated service. This is an "
            "arithmetic consequence of a measured share, not a tp1-vs-tp2 "
            "prefill A/B — such an A/B at >=8k would sharpen or shrink "
            "the 262144 boundary."),
        ticket="",
        site="spec/measurements/glm53_flash.md (GF3.10, 2026-08-30); INV-KDA-TP",
    ),
    ConstraintRow(
        id="local-indexer-prefill-cost",
        kind="measured",
        statement=(
            "dcp_indexer_mode=local is a CAPACITY decision with a known "
            "price, not a free VRAM win. Champion A/B at dcp=2 (sharded KV, "
            "tiering + tiered sparse prefill ON), replicated -> local: "
            "45.15 -> 41.56 tok/s at 8k (-8.0%), 43.88 -> 39.95 at 20k "
            "(-9.0%), 43.85 -> 40.04 at 20k repeat (-8.7%), 42.73 -> 38.82 "
            "at 25k (-9.2%) served prefill; decode in-noise, TOKEN-IDENTICAL "
            "on every leg, indexer_dense_steps=0, degraded=0. PAYOFF: the "
            "indexer-K sizing share halves per rank — 84 slabs (87.1 MiB) -> "
            "42 slabs (43.5 MiB) per rank at max_sequence_length 25600 "
            "(boot-log verified; the share is ceil(max_seq / "
            "indexer_k_page_size) pages x DSA computing layers x slab, "
            "ceil-divided by dcp under local). MECHANISM: under local each "
            "rank selects over its own shard, so every chunk ROW pays a "
            "cross-rank candidate merge, per layer, over B=64 rows — the "
            "cost tracks prefill ROWS, not context bytes, which is why "
            "decode (one row) is unaffected. POLICY: prefer replicated while "
            "its indexer-K bytes are affordable; take local when they are "
            "genuinely contested, and NAME the price when doing so. "
            "EVIDENCE BOUND: ONE box (2x rtx5090 + 2x rtx5080, dcp=2), ONE "
            "model (GLM-5.2 champion, glm_moe_dsa, 21 DSA computing layers, "
            "turboquant_mla), at the 8k/20k/25k ladder, under the §4b "
            "preconditions (deterministic_ep_combine ON, DET-TOPK-TIES "
            "kernel, degraded=0). It does NOT establish the cost on another "
            "topology (dcp>2, other TP widths), on another architecture "
            "(V4's side-tier indexer, IndexPool archs whose pages are "
            "~kpool x cheaper), or at other context lengths — the per-row "
            "mechanism PREDICTS a roughly constant percentage at 100k+ and a "
            "payoff that grows into GiB there, and neither has been "
            "measured. FALSIFIED BY: a ladder at >=100k, at dcp>2, or on "
            "another DSA model landing outside ~5-12%, or a payoff large "
            "enough to flip the trade at champion context. Re-measure with "
            "scratchpad/kvtlocal_ab_gate.py; then edit this row's numbers, "
            "or DELETE it and the solver goes back to taking local wherever "
            "nothing forbids it."),
        ticket="",
        site=("spec/measurements/glm_prefill.md 2026-08-30 (champion A/B, "
              "TD-KVT-LOCAL-INDEXER-UNBLOCK resolution; row added "
              "2026-08-31); evidence scratchpad/kvtlocal_ab_replicated.json "
              "+ scratchpad/kvtlocal_ab_local.json + "
              "scratchpad/kvtlocal_ab_compare.py; INV-KVT-20"),
    ),
    ConstraintRow(
        id="mla-tq-vs-snapmla-accuracy",
        kind="measured",
        statement=(
            "`turboquant_mla` (4-bit TQ_MSE KV codec) is MEASURABLY less "
            "accurate than `snapmla` (fp8 KV rows), and the gap is small "
            "and low-margin-shaped. Teacher-forced exact-NLL A/B on "
            "GLM-5.3-Flash (glm5_next), 7000-token held-out prose corpus "
            "(CALM paper intro, model-own tokenization), one rtx5090, tp=1, "
            "deterministic_ep_combine ON, DET-TOPK-TIES kernel, "
            "degraded=0 on every step: mean NLL 1.8782 (TQ) vs 1.8500 "
            "(snapmla) nats/token — dNLL +0.0282 +- 0.0063 (95% CI, "
            "excludes zero; perplexity 6.542 vs 6.360, +2.9% relative), "
            "top-1 accuracy vs the corpus 56.57% vs 57.17%, cross-arm "
            "argmax agreement 92.31% with the 7.69% flips concentrated at "
            "LOW margin (median top1-top2 gap 0.17 nats at flip positions "
            "vs 1.35 overall — the 'both fluent' TQ character, now a "
            "number), truncated top-16 KL(tq||snapmla) 0.021 nats. The gap "
            "does NOT grow with context up to 7k (dNLL +0.054 at <=1k, "
            "+0.027 at 1-4k, +0.023 at 6.5-7k). PAYOFF of TQ: 0.50x KV "
            "bytes/row (glm5_next NoPE 258 vs 516 B; GLM-5.2 rope geometry "
            "386 vs 644 B, 0.60x). POLICY: the accuracy lever's ladder "
            "orders snapmla above turboquant_mla; `standard` keeps the "
            "family template — snapmla since P-29 step 14's champion "
            "switch (wins BOTH axes on the fixed binary; template synced "
            "P-31 step 1) — `compact` floors at TQ for the 0.50x KV "
            "bytes, and capacity flows through the ordinary E1 fit. "
            "EVIDENCE BOUND: ONE model (GLM-5.3-Flash GGUF), ONE box, ONE "
            "corpus domain (academic prose, in-pretraining-distribution — "
            "the bias is shared by both arms and cancels in the PAIRED "
            "comparison, but absolute ppl is optimistic), context <= 7k, "
            "B=1 decode-shaped teacher forcing on the production DSA "
            "path. It does NOT establish the gap at >=100k context (TQ "
            "error could accumulate with KV volume), on other MLA "
            "geometries, or across corpus domains. FALSIFIED BY: a re-run "
            "whose CI covers zero or flips the sign (delete the row); a "
            ">=100k-context A/B showing a materially larger gap (edit the "
            "numbers — the ORDER stands, the magnitude moves); a "
            "domain sweep reversing the top-1-accuracy order. Re-measure "
            "with scratchpad/accuracy_lever/tf_nll.py + analyze.py; edit "
            "this row's numbers, or DELETE it and the accuracy lever's "
            "floor downgrades to the structural precision ordering "
            "(explain kind heuristic, marked unpriced)."),
        ticket="",
        site=("spec/measurements/glm53_flash.md 2026-09-02 (accuracy-lever "
              "A/B); evidence scratchpad/accuracy_lever/"
              "{tq,snapmla}.tf.jsonl.gz + ANALYSIS.txt + tf_nll.py"),
    ),
    ConstraintRow(
        id="tp-gpus-top-class-only",
        kind="validator",
        statement=(
            "Every TP GPU must be the box's top attention class "
            "(INV-0.5: rtx5090 today); mixed-class TP groups are rejected."),
        ticket="",
        site="src/config/config_validator.cpp:92-131",
    ),
    ConstraintRow(
        id="expert-host-prefix",
        kind="engine_gate",
        statement=(
            "The expert-host set is the PREFIX of hardware.gpus up to the "
            "first entry without resident/expert_streaming roles — the scan "
            "BREAKS, it does not skip. Order attention-only GPUs last."),
        ticket="",
        site="python/bridge/ring_bridge.py:291-297, src/daemon/dispatch_reef.cpp:461-479",
    ),
    ConstraintRow(
        id="sharded-kv-beats-replicated",
        kind="measured",
        scope="glm_moe_dsa",
        statement=(
            "dcp_kv_mode=sharded measured +4.6% e2e over replicated at "
            "dcp=2 on the champion (replicated KV LOSES 4.6%). ARCH-SCOPED "
            "(TD-AUTOCONFIG-GLM5NEXT-TP-SHARDED-KV (c)): measured on the "
            "GLM-5.2 champion only — a measured row must never price an "
            "arch that cannot shard (glm5_next fail-closes sharded KV, "
            "combination row glm5next-no-sharded-kv) or one it was not "
            "measured on; off-scope archs take sharded as an UNPRICED "
            "structural default instead."),
        ticket="",
        site="spec: arena placement config campaign (2026-08)",
    ),
    ConstraintRow(
        id="dspark-sharded-nvfp4-draft",
        kind="measured",
        statement=(
            "A safetensors dspark draft runs best nvfp4-quantized and "
            "sharded across the TP GPUs (tax 10->7.7 ms/step; acceptance UP "
            "under quant). GGUF dflash drafts are bf16 single-rank only."),
        ticket="",
        site="TD-DSPARK-DRAFT-QUANT/TD-DSPARK-DRAFT-SHARD (resolved 2026-08-05)",
    ),
    ConstraintRow(
        id="hbm-fraction-free-06",
        kind="measured",
        statement=(
            "CPU-less HBM bank pinned budget: fraction_free 0.8 was too "
            "tight (node-local OOM, bare 'Killed'); 0.6 works."),
        ticket="",
        site="scratchpad/CAMPAIGN_DOSSIER.md §1",
    ),
    ConstraintRow(
        id="full-calibration-needs-cold-boot",
        kind="engine_gate",
        statement=(
            "calibration_mode=full/loaded-fallback allocates a ~32 GiB "
            "NUMA-bound host footprint EARLY in boot; against a warm "
            "arena-holder store the GPU-local node is already ~90% pinned "
            "and the kernel node-constrained OOM-kills the engine "
            "(CONSTRAINT_MEMORY_POLICY, verified 2026-08-30 boot leg). A "
            "calibrating boot must run COLD (no holder attach) — true by "
            "construction on a new box, but an operator re-calibrating a "
            "warm box must kill the holder first."),
        ticket="",
        site="scratchpad/autoconfig_boot2_selfheal.log + journalctl 2026-08-30",
    ),
    ConstraintRow(
        id="tiering-needs-row-self-contained-format",
        kind="engine_gate",
        statement=(
            "GLM/SnapMLA KV tiering also requires a row-self-contained KV "
            "format: attention_backend snapmla or turboquant_mla "
            "(kSnapMlaFp8 / kTurboQuantMse4)."),
        ticket="",
        site="src/daemon/command_dispatcher.cpp:1361-1366",
    ),
    ConstraintRow(
        id="tiering-chunk-page-divisibility",
        kind="engine_gate",
        statement=(
            "Under sharded KV, tiering requires dcp_chunk_size % "
            "page_size_tokens == 0 (shard_ok geometry)."),
        ticket="",
        site="src/daemon/command_dispatcher.cpp:1367-1370",
    ),
)


def get(constraint_id: str) -> ConstraintRow | None:
    """Row lookup; None when the row has been deleted (rule no longer holds)."""
    for row in ENGINE_CONSTRAINTS:
        if row.id == constraint_id:
            return row
    return None


def active_for(arch: str) -> tuple[ConstraintRow, ...]:
    return tuple(r for r in ENGINE_CONSTRAINTS if not r.scope or r.scope == arch)

# Autoconfig derivation — glm53flash_serve.json

| path | value | kind | because | measurement | refs |
|---|---|---|---|---|---|
| `_internal-prefix_cache.max_entry_tokens` | `1048576` | closed_form | one full request is the largest reusable prefix |  |  |
| `autoconfig.fingerprint` | `hwfp1-a96da7eccdc211ff325b963430d58953` | measured | measured-topology fingerprint over live /proc + /sys + calibration tier matrix |  |  |
| `autoconfig.vram_expert_ratio` | `0.3054` | heuristic | lever unset: expert hosts filled to capacity, TP GPUs take the residual after KV/indexer (the engine's own residual carve) |  |  |
| `compute.attention_backend` | `snapmla` | pinned | PINNED by the user (--pin compute.attention_backend=snapmla): a HARD CONSTRAINT the solver treated as fixed and solved around — told, not derived. It survives every E1 rung; an infeasible fit refuses naming this pin rather than relaxing it |  | TD-AUTOCONFIG-PINNED-CONSTRAINTS; spec/plans/AUTOCONFIG_PINNED_CONSTRAINTS.md; AUTOCONFIG §2.5 |
| `compute.prefill_superchunk_tokens` | `512` | template | superchunk prefill REQUEST cap from the family serving template: GF3.15 measured superchunk FASTER than chunked on glm5_next (141-token arm: 35.9 vs 38.1 s mapped, 36.2 vs 41.3 s carve); the engine derives the effective batch capacity K elastically at init (prefill_moe_big), this caps the request | spec/measurements/glm53_flash.md (GF3.15) | recipes/glm53flash_serve.json; TD-PREFILL-SUPERCHUNK; TD-AUTOCONFIG-NO-SERVING-SURFACE |
| `gpu_loader.calibration_mode` | `loaded` | measured | an accepted calibration exists for this box+model, so the engine loads it instead of re-measuring (schema default; the champion omits the field). It still self-heals: a missing file falls back to a full calibration and writes it | gpu_loader_calibration_glm-5.3-flash_trained.json | engine.cpp:519-568; INV-LOADER-CAL-6 |
| `hardware.dcp_indexer_mode` | `replicated` | measured | replicated is the MEASURED default at dcp=2: `local` would halve the per-rank indexer-K share (1536 slabs / 399.1 MiB -> 768 slabs / 199.5 MiB per max-length sequence) but costs ~8-9% of served prefill (45.15/43.88/43.85/42.73 -> 41.56/39.95/40.04/38.82 tok/s at 8k/20k/20k/25k (-8.0/-9.0/-8.7/-9.2%); decode in-noise, token-identical, tiering composes) because the per-chunk-row cross-rank merge runs per layer over B=64 rows. Those 199.5 MiB/rank are AFFORDABLE here (0.59 GiB slack left on the tightest TP GPU), so the box buys prefill with VRAM it is not using; the E1 fit escalates to `local` on its own when they are contested | champion A/B 2026-08-30, spec/measurements/glm_prefill.md; evidence scratchpad/kvtlocal_ab_{replicated,local}.json | local-indexer-prefill-cost; TD-KVT-LOCAL-INDEXER-UNBLOCK; INV-KVT-20; AUTOCONFIG §5 P5 |
| `hardware.dcp_kv_mode` | `replicated` | engine_gate | combination gate (arch == glm5_next AND tensor_parallelism > 1): glm5_next rejects dcp_kv_mode=sharded at tp>=2: the 34 KDA layers' recurrent state is whole-sequence, so there is no token shard for a rank to own - the engine fail-closes with the fix in the message (GF3.10, mirror of TD-V4-DCP-KV; belt-and-braces refusal in the arch's validate_shape). Replicated KV is the only mode that boots. |  | glm5next-no-sharded-kv; src/daemon/engine.cpp:637-643; TD-AUTOCONFIG-GLM5NEXT-TP-SHARDED-KV |
| `hardware.gpus` | `4 entries` | closed_form | one entry per detected GPU (4 on this box, hwdetect BAR1 scan), placement-ordered; the first 2 (tp_array) carry attention/resident roles, the rest are expert_streaming hosts |  | AUTOCONFIG §5 E1; hwdetect.detect_hardware |
| `hardware.gpus[0].vram_allocation_gb.expert_streaming` | `20.0` | searched | lever vram_expert_ratio: 1166 slots x 18415616 B (GGUF tensor headers, GG-9 per-projection max (down:q6_k,gate:q5_k,up:q5_k)); TP GPU: overflow share after non-TP hosts filled |  | AUTOCONFIG §2.1/§5 E1 |
| `hardware.gpus[0].vram_gb` | `30` | heuristic | physical 32768 MiB (BAR1) x 0.9375 usable fraction (1/16 driver+runtime reserve) | BAR1 span, 0000:6a:00.0 |  |
| `hardware.gpus[1].vram_allocation_gb.expert_streaming` | `20.0` | searched | lever vram_expert_ratio: 1166 slots x 18415616 B (GGUF tensor headers, GG-9 per-projection max (down:q6_k,gate:q5_k,up:q5_k)); TP GPU: overflow share after non-TP hosts filled |  | AUTOCONFIG §2.1/§5 E1 |
| `hardware.gpus[1].vram_gb` | `30` | heuristic | physical 32768 MiB (BAR1) x 0.9375 usable fraction (1/16 driver+runtime reserve) | BAR1 span, 0000:94:00.0 |  |
| `hardware.gpus[2].vram_allocation_gb.expert_streaming` | `11.5` | searched | lever vram_expert_ratio: 670 slots x 18415616 B (GGUF tensor headers, GG-9 per-projection max (down:q6_k,gate:q5_k,up:q5_k)); expert host: capacity after margin 2.25 + overhead 1.25 GiB |  | AUTOCONFIG §2.1/§5 E1 |
| `hardware.gpus[2].vram_gb` | `15` | heuristic | physical 16384 MiB (BAR1) x 0.9375 usable fraction (1/16 driver+runtime reserve) | BAR1 span, 0000:16:00.0 |  |
| `hardware.gpus[3].vram_allocation_gb.expert_streaming` | `11.5` | searched | lever vram_expert_ratio: 670 slots x 18415616 B (GGUF tensor headers, GG-9 per-projection max (down:q6_k,gate:q5_k,up:q5_k)); expert host: capacity after margin 2.25 + overhead 1.25 GiB |  | AUTOCONFIG §2.1/§5 E1 |
| `hardware.gpus[3].vram_gb` | `15` | heuristic | physical 16384 MiB (BAR1) x 0.9375 usable fraction (1/16 driver+runtime reserve) | BAR1 span, 0000:40:00.0 |  |
| `hardware.gpus[].order` | `[2, 3, 0, 1]` | measured | best-first: TP group = first tensor_parallelism entries (engine.cpp:585-592), expert-host prefix scan follows (registry row expert-host-prefix) | gpu_loader calibration (gpu_loader_calibration_glm-5.3-flash_trained.json) | AUTOCONFIG §5 P1; expert-host-prefix |
| `hardware.tp_array` | `[0, 1]` | closed_form | positions of the first tensor_parallelism entries of hardware.gpus (the engine's own TP-group rule; note the id-vs-position validator ambiguity — both shipped recipes use positions) |  | engine.cpp:585-592 |
| `memory.arena_placement` | `{'freq_table': '/srv/models/unsloth/GLM-5.3-Flash-GGUF/UD...` | carried | measured placement table carried into the derivation (auto-run step 4 supplied/reused it, or a base recipe carried it): a per-(layer,expert) demand-fetch fit, trace-fit per model AND box — tools/loader_xray/freq_table.py over an LS_PERF_TRACE dump. Its content hash folds into the ArenaCache store identity, so CHANGING it costs one cold store rebuild |  | memory.arena_placement.freq_table; arena_placement.h; spec/AUTO_RUN.md step 4 |
| `memory.cross_node_spill.nodes` | `[0, 1, 4, 5, 6, 7]` | heuristic | all memory nodes except those local to a TP GPU [2, 3] (TP-local free memory serves KV-tiering host pools and attention D2H staging — D2H must stay NUMA-local) |  | AUTOCONFIG §5 H1; feedback: D2H NUMA-aware |
| `memory.cross_node_spill.per_node(hbm)` | `0.6` | measured | CPU-less HBM banks pin at fraction_free 0.6 — 0.8 measured too tight (node-local OOM with 30 GB free system-wide) |  | hbm-fraction-free-06 |
| `memory.kv_cache.max_pages_per_gpu` | `3072` | heuristic | tiered KV device pool = 2 seq x ceil(hot 2048/page 16) x 12 pool layers, rounded up to 1024 (24 MiB at 8256 B/page); the host cold pool holds the rest (INV-KVT-16 windowed admission) |  | kv_tiering_manager.cpp:120-137; INV-KVT-16 |
| `memory.kv_tiering.enabled` | `True` | pinned | PINNED by the user (--pin memory.kv_tiering.enabled=true): a HARD CONSTRAINT the solver treated as fixed and solved around — told, not derived. It survives every E1 rung; an infeasible fit refuses naming this pin rather than relaxing it |  | TD-AUTOCONFIG-PINNED-CONSTRAINTS; spec/plans/AUTOCONFIG_PINNED_CONSTRAINTS.md; AUTOCONFIG §2.5 |
| `memory.kv_tiering.hot_buffer_slots` | `2048` | measured | hot window = index_topk (champion-measured; engine auto would be 2x); cold pool 6144 pages/rank = 48 MiB pinned host per rank after replica dedup |  | kv_tiering_manager.cpp:120-137 |
| `memory.nvme_tier.enabled` | `False` | closed_form | host arena holds every expert slot in RAM — no NVMe tier needed (nvme gen3 x4 ceiling ~3.4 GB/s; cold boot pays the disk ceiling once, the arena holder makes warm boots ~25-75 s) | nvme gen3 x4 ceiling ~3.4 GB/s | disk-ceiling: hardware Gen3 x4 |
| `memory.pin_host_expert_pool_sizing` | `{'mode': 'fraction_total', 'value': 0.9}` | measured | host arena 223 GB across GPU nodes [0, 2, 3] + spill (capacity ~384 GB); per-NUMA-node capacity is the real ceiling, not RLIMIT_MEMLOCK |  | INV-4.12f; pinned_expert_arena.cpp:315-388 |
| `memory.pinned_layers.dense_ffn_layers` | `[0, 1, 2]` | closed_form | first_k_dense_replace=3 |  |  |
| `orchestrator.max_batch_size` | `512` | template | family serving template: the GF3 live-serving recipe's schedule width for glm5_next — a copied serving-policy knob (§5 M1), not a derivation |  | recipes/glm53flash_serve.json; TD-AUTOCONFIG-NO-SERVING-SURFACE |
| `parallelism.tensor_parallelism` | `2` | pinned | PINNED by the user (--pin parallelism.tensor_parallelism=2): a HARD CONSTRAINT the solver treated as fixed and solved around — told, not derived. It survives every E1 rung; an infeasible fit refuses naming this pin rather than relaxing it |  | TD-AUTOCONFIG-PINNED-CONSTRAINTS; spec/plans/AUTOCONFIG_PINNED_CONSTRAINTS.md; AUTOCONFIG §2.5 |
| `serving.enable_auto_tool_choice` | `True` | carried | serving surface carried from the base recipe — deployment wiring (parsers, tokenizer mode) the solver must never silently drop |  | TD-AUTOCONFIG-NO-SERVING-SURFACE |
| `serving.max_concurrent_requests` | `2` | pinned | PINNED by the user (--pin serving.max_concurrent_requests=2): a HARD CONSTRAINT the solver treated as fixed and solved around — told, not derived. It survives every E1 rung; an infeasible fit refuses naming this pin rather than relaxing it |  | TD-AUTOCONFIG-PINNED-CONSTRAINTS; spec/plans/AUTOCONFIG_PINNED_CONSTRAINTS.md; AUTOCONFIG §2.5 |
| `serving.max_sequence_length` | `1048576` | pinned | PINNED by the user (--pin serving.max_sequence_length=1048576): a HARD CONSTRAINT the solver treated as fixed and solved around — told, not derived. It survives every E1 rung; an infeasible fit refuses naming this pin rather than relaxing it |  | TD-AUTOCONFIG-PINNED-CONSTRAINTS; spec/plans/AUTOCONFIG_PINNED_CONSTRAINTS.md; AUTOCONFIG §2.5 |
| `serving.prefix_cache.max_cached_tokens` | `2097152` | heuristic | active-context lever rounded up to a 64k prefix budget granule |  |  |
| `serving.reasoning_parser` | `glm45` | carried | serving surface carried from the base recipe — deployment wiring (parsers, tokenizer mode) the solver must never silently drop |  | TD-AUTOCONFIG-NO-SERVING-SURFACE |
| `serving.tokenizer_mode` | `glm5_next` | carried | serving surface carried from the base recipe — deployment wiring (parsers, tokenizer mode) the solver must never silently drop |  | TD-AUTOCONFIG-NO-SERVING-SURFACE |
| `serving.tokenizer_path` | `test-data/GLM-5.3-Flash` | closed_form | a resolved tokenizer directory, not 'auto': serve's auto-resolution only searches the WEIGHTS dir and GGUF-embedded tokenizers are not extracted, so a GGUF box needs the path in the recipe or every boot needs --tokenizer-path |  | python/cli/serve.py resolve_tokenizer_dir; TD-SERVE-GGUF-TOKENIZER |
| `serving.tool_call_parser` | `glm47` | carried | serving surface carried from the base recipe — deployment wiring (parsers, tokenizer mode) the solver must never silently drop |  | TD-AUTOCONFIG-NO-SERVING-SURFACE |
| `speculation.dspark` | `None` | searched | omitted with the method decision — no dspark checkpoint discoverable for this model |  | AUTOCONFIG §5 P6 |
| `speculation.enabled` | `False` | closed_form | follows the P6 method decision — the speculation scaffold arms exactly when a sized method is on (the GF3 hand recipe's shape: enabled=false when no method serves; enabled:true beside method:'none' was a template constant masquerading as a decision) |  | AUTOCONFIG §5 P6; TD-AUTOCONFIG-NO-SERVING-SURFACE |
| `speculation.method` | `none` | searched | no dspark checkpoint discoverable for this model |  | AUTOCONFIG §5 P6 |

## Warnings

- pin compute.attention_backend='snapmla' moves the backend off the standard tier's `turboquant_mla` — measured registry row 'mla-tq-vs-snapmla-accuracy' prices the step (teacher-forced dNLL +0.0282 +- 0.0063 nats/token, TQ worse (ppl 6.542 vs 6.360, +2.9%); top-1 acc 56.57% vs 57.17%; argmax agreement 92.31% with flips at low margin (median 0.17 vs 1.35 nats); gap flat in context to 7k) [spec/measurements/glm53_flash.md 2026-09-02 (accuracy-lever A/B); evidence scratchpad/accuracy_lever/{tq,snapmla}.tf.jsonl.gz + ANALYSIS.txt + tf_nll.py] — deliberate override assumed, never silent (TD-AUTOCONFIG-PINNED-CONSTRAINTS (a))
- pin parallelism.tensor_parallelism=2 OVERRIDES measured registry row 'glm5next-tp1-default': glm5_next serves at TP=1 by default (KDA decode is latency-floored; combines cost ~0.5-2% of the decode wall; TP is for CAPACITY, never speed) [spec/measurements/glm53_flash.md (GF3.10, 2026-08-30); INV-KDA-TP] — deliberate override assumed, never silent (TD-AUTOCONFIG-PINNED-CONSTRAINTS (a))
- indexer-K share: VRAM affords 1 concurrent max-length sequence(s), below max_concurrent_requests=2 (399.1 MiB/seq) — S4 elastic: extra demand draws on free KV slabs; concurrent max-length requests churn holder evictions (retryable, TD-INDEXER-POOL-EVICT)
- pin memory.kv_tiering.enabled=true OVERRIDES measured registry row 'glm5next-tiering-default-off': glm5_next KV tiering ships default OFF: at 25k its KV+indexer costs ~0.14 GiB vs GLM-5.2's 1.24 GiB (8.8x less KV/token, 11 KV-bearing layers), so tiering buys nothing at champion context. Re-decide on measurement at >=256k. Delete this row when that measurement lands and the solver will weigh tiering by bytes again. [spec/SPEC_UPDATES.md:668 (GF3.9, 2026-08-30)] — deliberate override assumed, never silent (TD-AUTOCONFIG-PINNED-CONSTRAINTS (a))

## Post-derivation deltas (P-29 recipe promotion, 2026-09-05 — hand-applied, each cited)

The derivation above is the solver's fit for the pinned ask (tp=2 / snapmla /
conc 2 / max_seq 1,048,576 / kv_tiering ON) from the champion base
`scratchpad/kvxp_decisive2/glm53-tp2ep4.ab.json`. Four fields were then
changed by hand, all outside the solver's derived surface or covering a FILED
solver defect:

1. `orchestrator.max_batch_size: 512 -> 128`. The 512 is a `[template]` copy
   from the STALE pre-promotion recipe this file replaces (the solver calls it
   "a copied serving-policy knob, not a derivation"). The champion serves at
   128. At max_seq 1M, B=512 sizes `dev_block_tables` alone at 5.75 GiB/rank
   (46 layers x 512 x 65,536 blocks x 4 B, `command_dispatcher.cpp:1237`) —
   ~9.0 GiB/rank of margin-funded runtime allocations, undeployable. B=128
   needs ~3.15 GiB/rank. Prefill throughput is unaffected (chunk row bound =
   max(max_batch, prefill_superchunk_tokens=512) = 512 either way).
2. `hardware.gpus[0,1].vram_allocation_gb.safety_margin_gb: 3.75` (TP 5090s;
   global stays 2.25). TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH (OPEN, ★HIGH):
   the solver does not charge max_seq-scaled RUNTIME allocations that live
   outside the VramAllocator block and are funded ONLY by the margin.
   Enumerated at 1M/B=128 per TP rank: dev_block_tables 1,472 MiB + snapmla
   prefill_kv_staging 1,024 MiB (max_seq x 1,024 B; `engine.cpp:909`,
   `snapmla_sm120_attention_device.cpp:481`) + KVT union staging ~270 MiB +
   logits/indexer/KDA scratch ~380 MiB = ~3.15 GiB > 2.25. 3.75 GiB is the
   GLM-5.2 1M precedent (`scratchpad/kvxp_fat2/`), leaving ~0.6 GiB for
   unquantified terms (CUTLASS workspaces, graph exec objects, fragmentation).
3. `hardware.gpus[0,1].vram_allocation_gb.expert_streaming: 20.0 -> 18.5`.
   Funds delta 2 out of the solver-awarded carve so the in-block fit keeps its
   0.59 GiB slack (the block is total - margin; the extra 1.5 GiB margin must
   come out of an in-block tenant). Still +4.0 GiB per 5090 over the
   pre-promotion champion's 14.5 (the tiering dividend).
4. `autoconfig.enabled: true -> false`. A production recipe must serve as-is:
   with enabled=true and a foreign output_path plain `serve.py --config`
   REFUSES (TD-AUTOCONFIG-SERVE-SUBSTITUTES-THE-CONFIG), and a
   fingerprint-triggered re-derivation would NOT carry the promotion pins
   (pins are CLI-only by design) — it would derive tp=1/turboquant_mla.
   The fingerprint stays for provenance.

## STATUS: UNVERIFIED — NEVER BOOTED (2026-09-05)

User scope cut ("just set it. then we'll test another time."): this recipe
was derived and committed WITHOUT a boot. No sha check, no 8k band check, no
conc-2 admission test, no tiering-at-depth probe. 1M/conc2 admissibility on
snapmla is UNTESTED — every long-context measurement on file (hot-window-fixed
KV pages, the 443,552 single-request ceiling, 44z 93.8% donation) is TQ-era
or GLM-5.2. glm5_next + kv_tiering has unit coverage (GF3.9 KV-bearing mask)
but has never been booted live. First boot should watch: warm attach adopts
12,096; `Indexer-K ... S4 ELASTIC ... 1 sequence(s) at
max_sequence_length=1048576`; KvTiering init; the champion sha groups
(da30b0d40b3a508f / 1874bd3d2aeb9d1a / 380592b1a8ccb9b0) — tiered_prefill
changes admission windowing, so an identity fork is possible and must be
checked before this recipe is called the champion's equal; 8k decode band
~27.0; and a conc-2 overlap pair. Probe drivers are staged in
scratchpad/gf3_speed_saga/promotion/ (conc2_probe.py, depth24k_probe.py,
legs_promo.py). The no-tiering alternative fit is
scratchpad/gf3_speed_saga/promotion/derive_A_1m_notier.json (expert carve
shed to 8 GiB/5090 instead; single-request ceiling 1,915,360 tokens but two
concurrent 1M requests exceed the pool).

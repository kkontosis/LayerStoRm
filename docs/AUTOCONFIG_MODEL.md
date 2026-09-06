# AUTOCONFIG — hardware-fit config derivation (TD-AUTOCONFIG-HARDWARE-FIT)

Write the objective down, then solve it (prior art:
docs/I8_PLACEMENT_MODEL.md). The code in `python/autoconfig/` is a transcription
of this document; code comments cite sections back here as `AUTOCONFIG §n`.

## 0. The single call (P-31, boot-verified 2026-09-06)

Autoconfig's production form is ONE serve command from the weights alone —
no `--config`, no pins, no hand deltas:

    .venv/bin/python python/cli/serve.py --autoconfig \
        --model /srv/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
        --max-sequence-length 1048576 --max-concurrent 2

This derived, on the reference box, the full glm5_next long-context recipe:
`snapmla` / tp2 / kv_tiering ON (capacity escalation) / `S = N = 2048`
(prefill superchunk = MoE-big chunk, the EP diagonal) / `max_batch_size 128`
/ TP `safety_margin_gb` 4.5 / `moe_big_fit_headroom_mb` 2426 / expert carve
18.0 GiB per 5090 + 11.5 GiB per 5080 — every field explained in the
sidecar, and BOOT-VERIFIED as the first live glm5_next + kv_tiering boot
(P-31 step 1; measured, internal ledger).

**Serving-shape flags become solver pins.** Under `--autoconfig`,
`--max-sequence-length` and `--max-concurrent` are not mere HTTP-surface
overrides: serve.py converts them into HARD pins
(`serving.max_sequence_length=…`, `serving.max_concurrent_requests=…`,
via `autoconfig.pins.parse_pin_args` — §2.5 semantics: the axis is removed
from the search lattice, and an infeasible pin REFUSES naming itself).
As plain ServeOptions overrides they would change the HTTP limits but not
the engine sizing, which reads `serving.*` from the config file.

**The user levers, by surface** (everything else is solved for):

- `serve.py --autoconfig`: `--model` (derive from the weights; measured
  artifacts beside them are reused; the derive is CPU-only),
  `--accuracy compact|standard|high|superior` (§2.4), `--prefer
  speed|balanced|capacity` (§2.3), `--max-sequence-length` /
  `--max-concurrent` (pinned as above), `--autoconfig-redetect`
  (re-derive despite a matching fingerprint), and optionally `--config`
  (a base recipe carrying extras into the derivation).
- `python/cli/autoconfigure.py` (the measured pipeline, §10): the same
  `--prefer` / `--accuracy` plus `--vram-expert-ratio` (§2.1; unset =
  fill every expert host to capacity — P-31 folding (b)),
  `--active-context` (§2.2), `--pin` (repeatable, file or
  `dotted.path=value`, §2.5), `--redetect`, and the artifact-scoped flags
  (`--calibration`, `--trained`, `--placement-table`, `--reset`,
  `--skip-training`, `--skip-placement`, `--name`/`--prefix`, …).

**Measured on the derived recipe** (single legs, not banked bands; P-31
step 1): decode 8k repeat median **24.52 tok/s**; fresh prefill
**159.0 tok/s** at a clean 27k prompt (+34% vs the S=512 hand champion's
118.5 @24k); TTFT after a late (~97%) divergence **26.2 s** (KDA
checkpoint restore); conc-2 serves overlapping request pairs; **1M context
is admitted but unmeasured** — the engine's boot log states windowed
admission (INV-KVT-16) serves past the single-request pool ceiling
(243,968), and no 1M prefill has been run.

**The kv-tiering price is measured** and lives in the
`glm5next-tiering-default-off` registry row: vs the untiered champion,
decode ~−10% @8k (24.4–24.6 vs 27.0–27.4) and ~−13% @24k (22.9–23.1 vs
~26.5–27.2), fresh prefill −16% (159.0 vs 189 untiered @S=2048) — the
tiered_prefill tax. That price buys the 2×1M admissibility. **Untiered
stays the default below the capacity bound**: the E1 lattice escalates
tiering ON only when the untiered pool cannot hold the conc × max_seq ask
(`conc-ask-exceeds-untiered-pool`); a 2×100k ask derives untiered.
Consequently the 200k hand champion (untiered, ~27.0–27.4 tok/s @8k)
remains the FASTER decode arm and coexists with the derived 1M recipe —
autoconfig is the default path, the champion recipe is the tuned-arm
example, and neither replaces the other.

**Why derivation beats hand-editing a recipe.** The hand-promoted
`recipes/glm53flash_serve.json` (P-29) is now BEHIND the derivation: it
keeps S=512 (leaving the measured +54–62% fresh-prefill win of S=2048 on
the table) and it under-reserves the max_seq-scaled post-check
allocations — at 1M its `moe_big_fit_headroom_mb` is silently
under-provisioned, so that recipe would have died at the block-table
alloc instead of booting (P-31 diff: S 512→2048 + derived headroom 2426,
margin 3.75→4.5, expert 18.5→18.0 — the honest price of both; everything
else byte-equal). A hand edit satisfies the fields you look at; the
derivation re-charges every coupled term and explains each one.

**The four P-31 gaps, all CLOSED** (details §5b; ticket tracked in the
internal planning notes, not shipped):

1. TD-AUTOCONFIG-STRIDE-NOT-DERIVED — `prefill_superchunk_tokens` was a
   family-template constant (512); now derived as max-S-that-fits on the
   engine's own per-device-class MoE-big fit arithmetic (speed axis).
2. TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH — max_seq/max_batch-scaled
   runtime allocations (block tables, prefill KV staging, RoPE tables,
   KVT union staging, …) are now charged in `sizing.py`; derived
   long-context recipes boot.
3. TD-AUTOCONFIG-NO-GLM5NEXT-PROFILE — an ArchProfile for GGUF arch
   `glm5next` exists (layer_types from the `attention.head_count_kv`
   array, linear_attn_config, index_kpool, …), so `--model` alone is
   enough for GLM-5.3-class weights.
4. `orchestrator.max_batch_size` — was a stale template copy (512; 5.75
   GiB/rank of block tables at 1M); now derived as the largest schedule
   width whose block tables fit the runtime-scratch budget (128 here).

**Accuracy-ladder default flip (filed user decision, P-31 step 1):**
the `standard` tier now derives **snapmla** for glm5_next (template sync
to the P-29 step-14 champion switch — on the fixed binary snapmla wins
BOTH axes: 24.10 vs 23.74 tok/s 8k repeat median, TF-NLL 1.7897 vs
1.8016). `turboquant_mla` remains serveable as the `compact` tier.
Revert = one line in `templates.py` `FAMILY["glm5_next"]`; the decision
is filed in the internal campaign log (P-31 OPEN QUESTIONS).

## 1. Contract

- **OPT-IN, default OFF.** `autoconfig.enabled: false` in `config/schema.json`.
  A hand-tuned recipe always wins unless the user passes `--autoconfig` (CLI)
  or sets `autoconfig.enabled: true` in the base config.
- **The output is a normal recipe** — a full, schema-valid, diffable JSON the
  user can read and edit, persisted to `autoconfig.output_path` (default:
  `<base-config-stem>.autoconfig.json` next to the base config).
- **Tuned once, not every boot**: the emitted recipe carries the hardware
  fingerprint (§7) inside its own `autoconfig` section. On boot with
  `--autoconfig`, an existing output recipe whose fingerprint matches the
  live box is reused verbatim; re-derivation happens only on fingerprint
  change or explicit `--redetect` (serve.py: `--autoconfig-redetect`).
- **Every derived number is explainable** (§8): the solver emits one line per
  derived field naming the constraint that produced it, extending the boot-log
  pattern of `vram_allocator.cpp` (indexer pool / V4 side tiers).
- **Refuse rather than guess** (§6): infeasible lever values produce a
  refusal naming the binding constraint, never a config that boots into OOM.

## 2. The four human-facing levers

Everything else is solved for; these are the dials a person reasons in.

### 2.1 `vram_expert_ratio` — VRAM-resident expert fraction

**Definition (GLOBAL, counted in SLOTS):**

    vram_expert_ratio = S_vram / S_total
    S_total = n_routed_experts × num_moe_layers            (all expert slots)
    S_vram  = Σ_g floor(expert_zone_bytes(g) / bytes_per_expert)

summed over every expert-hosting GPU g (roles containing `resident` or
`expert_streaming`), where `expert_zone_bytes(g)` = the GPU's total expert
carve (stable + streaming zones) and `bytes_per_expert` is the UNPADDED quant
size the engine divides zones by (`expert_cache.cpp:29-39`), not the 4096-
aligned on-disk stride. Slots, not bytes: slots are the engine's own unit
(`ExpertCache::SlotAllocator`), invariant across quant routes; per-layer
ratios are not meaningful because the VRAM cache is one global slot pool —
per-layer residency is placement's job, not sizing's.

The solver translates the ratio into the per-GPU
`hardware.gpus[].vram_allocation_gb.expert_streaming` overrides (the expert
reserve carved BEFORE KV, `vram_allocator.cpp:860-865`), distributing slots
across expert hosts in proportion to each GPU's free capacity after its
non-expert tenants (pinned weights, dspark charge, KV/indexer on TP GPUs).
Champion-implied value on this box for GLM-5.2: ≈ 0.066 (≈1264 of 19200).

### 2.2 `total_active_context_tokens` — aggregate active context

**Definition:** the total number of tokens, summed over concurrently active
sequences, whose position-indexed state (KV pages + indexer-K + V4 side tiers
+ KDA state slots) the box must be able to hold at once. NOT the per-request
maximum. It decomposes as

    total_active_context = max_concurrent_requests × max_sequence_length

and the solver owns the split (§5 stage K1): `max_sequence_length` defaults to
`min(model.max_position_embeddings, total_active_context)` clamped by the
per-request feasibility ceiling, and `max_concurrent_requests =
max(1, round(total_active_context / max_sequence_length))`. Degradation is
graceful and ordered: first fewer concurrent requests, then a shorter
`max_sequence_length` — never a boot failure (§6). Champion-implied value:
2 × 25600 = 51200.

### 2.3 `prefer` — what the fit gives up FIRST

**Definition:** `speed | balanced | capacity`, default `balanced`. It changes
the ORDER the derivation tries things, and NOTHING else.

The E1 search is a lattice with three axes, each already ordered "free first"
in its own terms:

| axis | free end | priced end | price |
|---|---|---|---|
| indexer mode (P5) | `replicated` | `local` | ~8-9% of served prefill (`local-indexer-prefill-cost`) |
| TP degree (P2) | `tp_plan()[0]` | the head-divisible ceiling | ~0.5-2% of the decode wall (`glm5next-tp1-default`) |
| the user's ask (K1) | whole | concurrency down, then `max_sequence_length` halved, plus the optional draft | the ask itself |

`prefer` decides, for each PRICED axis, whether it is spent BEFORE the user's
ask or after it:

- **`speed`** — spend every free capacity lever before buying any priced
  slowdown. It sheds concurrency and context rather than pay ~8-9% of served
  prefill for `local`. It still takes a priced option when that option is the
  ONLY feasible path.
- **`balanced`** (default) — the shipped mix: the indexer price is spent
  before the ask (it keeps the ask whole for a measured, bounded cost), the TP
  escalation after it.
- **`capacity`** — spend every priced slowdown before touching the requested
  context/concurrency; the draft goes late.

**It is not a weight, a percentage, or an exchange rate**, and must not become
one: tok/s and MiB have no meaningful conversion — 43.5 MiB/rank is worth zero
on a box with slack and everything on a box that would otherwise refuse
(TD-AUTOCONFIG-SPEED-BUDGET records why, and what the successor is once a
third priced option lands). Order, not arithmetic.

**Two properties the implementation owes:**

1. *A preference can never cause a refusal another preference would have
   avoided.* Every preference is a SORT KEY over one lattice built once
   (`Solver._lattice`), so the set of candidate fits is preference-independent
   by construction; feasibility is a property of the box. The refusal itself
   reports the VRAM-cheapest corner's headroom, which is likewise
   order-independent.
2. *The sidecar names the preference exactly when it changed the outcome.*
   The solver re-walks the same (memoised) lattice in `balanced` order and
   compares winners; if they differ it emits an `autoconfig.prefer` row saying
   what it took or declined and at what price, and if they agree it says
   nothing. A knob that silently reorders is worse than no knob; one that
   narrates itself for doing nothing is noise.

### 2.4 `accuracy` — the numerics floor

**Definition:** `compact | standard | high | superior`, default `standard`.
It FLOORS the family's numerics ladder — the ordered list of attention
backends the engine can actually serve for the architecture
(`templates.ACCURACY_LADDER`; the partition is `config_validator.cpp:510-534`:
`{snapmla, turboquant_mla}` for the MLA families, `{csa_hca, csa_hca_tq,
csa_hca_tq_mix}` for V4) — and touches NOTHING else.  It exists because the
solver used to spend the numerics trade silently: the explain row justified
`turboquant_mla` **by bytes per KV row** — a capacity argument for what is an
accuracy decision.

| tier | meaning | glm5_next | mla_dsa | deepseek_v4 |
|---|---|---|---|---|
| `compact` | most KV-byte-efficient serveable backend | `turboquant_mla` | `turboquant_mla` | `csa_hca_tq` |
| `standard` | the family template's champion-proven choice (**default**; was byte-identical to the pre-lever derivation — proven the `prefer` way: git-archive pre-tree, 12 scenarios, full-derivation diff. P-31 step 1 (2026-09-06) moved the glm5_next template to `snapmla`, following the P-29 step-14 champion switch — snapmla wins BOTH axes on the fixed binary — so the NOOP proof is historical for glm5_next) | `snapmla` | `turboquant_mla` | `csa_hca_tq_mix` |
| `high` | most accurate serveable backend (all-FP8 KV path, no 4-bit TQ codec) | `snapmla` | `snapmla` | `csa_hca` |
| `superior` | full-precision KV (bf16 rows, codec `kFull`), no TQ anywhere, everything maximal — **DEFINED, NOT IMPLEMENTED**: the engine builds no `kFull` device arm (`kv_codec.h:12-14`), so the tier REFUSES at derive time naming that fact rather than approximating | refuses | refuses |

**It is a floor over measured options, never a weight** — tok/s and
perplexity have no honest exchange rate (the same argument that rejected a
scalar in TD-AUTOCONFIG-SPEED-BUDGET).  The ladder's ORDER is structural
(each step down quantizes strictly more of the KV path to the 4-bit TQ
codec); its MLA step is additionally MEASURED — registry row
`mla-tq-vs-snapmla-accuracy` carries the teacher-forced NLL/agreement A/B
that prices `turboquant_mla` against `snapmla`, with its evidence bound and
falsifiers.  On V4 only the structural order is known (per-arm goldens are
pass/fail, not a gradient) and the explain says so (`heuristic`, never
`measured`).

Properties the implementation owes (mirroring `prefer`):

1. *`standard` is the shipped derivation, byte for byte* — recipe, explain
   table, warnings, degradations and refusal text
   (an internal cross-tree no-op proof pins the diff; the
   unit suite pins the in-tree property).
2. *The sidecar names the tier exactly when it changed the derived backend*
   (`autoconfig.accuracy` row + a tier-aware `compute.attention_backend`
   row naming BOTH sides: the accuracy basis and the computed KV-bytes
   price); silent otherwise — including when an engine-gate registry row
   has already narrowed the menu to what the tier would have chosen.
3. *Capacity consequences flow through the ordinary E1 fit*: a tier never
   does arithmetic of its own — `high`'s 2.0x KV rows on glm5_next simply
   make the same ask fit half the context, listed under degradations.
4. *A menu row stays data*: re-introducing `glm5next-tq-backend-unwired`
   removes TQ from the menu for every tier; deleting the measured accuracy
   row downgrades the explain basis to `heuristic`/unpriced but keeps the
   structural floor.

### 2.5 `--pin` — hard constraints, not a lever (TD-AUTOCONFIG-PINNED-CONSTRAINTS)

Full design in the internal planning notes (not shipped). A pin is a THIRD
input class: the base config is an IDENTITY source the derivation overwrites
(`base_from_source` starts from `dict(base_config)`); a lever is an ASK the
E1 ladder may degrade; **a pin is neither — a hard constraint that survives
every E1 rung. It REMOVES an axis from the search lattice** (the lattice is
built with that axis collapsed to one point), and when no fit exists WITH
the pin the solver REFUSES naming the pin (`pinned-constraint-binds`, via a
relax-one-pin probe; `pinned-constraints-jointly-bind` when only the whole
set binds) — never quietly relaxes it. A pinned `max_sequence_length` is not
the base recipe's carried starting point: no halving rungs exist for it, and
the mapped-KDA admissibility loop refuses (`pinned-max-seq-not-admissible`,
quoting `kda-state-mapped-tenant` and the pool decomposition) instead of
reducing.

Surface: CLI-only (`--pin file.json` | `--pin dotted.path=value`,
repeatable) — no schema work owed (a pin file is a path argument; unlike
`prefer`/`accuracy`, whose schema properties landed 2026-09-02 and which a
derived recipe now carries exactly when non-default, pins stay CLI-only BY
DESIGN); the emitted recipe never carries pins, and pins
force a re-derivation past the fingerprint-reuse gate (reuse would silently
ignore a hard constraint). Pinnable = the solver's DECISION surface only
(`pins.PINNABLE`): `compute.attention_backend`, `hardware.dcp_indexer_mode`,
`hardware.dcp_kv_mode`, `parallelism.tensor_parallelism`,
`serving.max_sequence_length`, `serving.max_concurrent_requests`,
`speculation.method` (none/dspark), `memory.kv_tiering.enabled`; anything
else refuses at parse time listing the menu (an unmodeled pin would be
either an inert overlay or silently inconsistent with the sizing).

Registry rows split by `kind`: **`measured` rows are overridable, LOUDLY**
(warning quoting pin + row; deleting the row deletes the warning);
**`engine_gate`/`validator` rows and model geometry REFUSE** quoting both.
The sheet marks every pinned field **`pinned`** — told, not derived — and
the path becomes immutable in the ExplanationSet. Composition: `prefer`
sorts the remaining axes (attempt-set identity holds per pin set); a pinned
backend removes the numerics axis from `accuracy`'s scope — `standard`
(the no-preference tier) never conflicts, an explicit tier whose floor picks
a different backend refuses (`pin-conflicts-accuracy-floor`, the `superior`
refuses-rather-than-approximates precedent), agreement is silent. Default
path byte-identity proven the `prefer` way by an internal no-op proof
(12 scenarios, empty diff).

## 3. Inputs — the HardwareDescriptor and ModelShape

All detection is CPU-only (no CUDA calls; INV-GPU-1 untouched). Sources and
the traps each one encodes:

| fact | source | trap encoded |
|---|---|---|
| GPU inventory (model, PCI addr) | `/proc/driver/nvidia/gpus/*/information` | order is lexicographic-by-PCI, not CUDA ordinal; pair with sysfs by bus id |
| GPU VRAM (physical) | PCI BAR1 span size from `/sys/bus/pci/devices/<id>/resource` | requires resizable BAR (holds on this box: 32.0/16.0 GiB exact); fallback table by device name |
| usable `vram_gb` | `floor(physical_GiB − vram_reserve_gib)` (internal knob, default 0.9) | reproduces the champion's declared 30/15 |
| PCIe gen/width | sysfs `max_link_speed` / `max_link_width` | `current_link_speed` shows the idle downclock (2.5 GT/s) — never use it (`config_resolver.cpp` kickstarts the link for the same reason) |
| GPU NUMA node | sysfs `numa_node` | lowercase the bus id (sysfs_pci_id) |
| NUMA topology | `/sys/devices/system/node/node*/` (cpulist, meminfo) | CPU-less HBM banks = has memory + empty cpulist (nodes 4-7 here, 16 GiB each); per-node meminfo has NO MemAvailable line (`numa_manager` reconstructs it) |
| host RAM | `/proc/meminfo` MemTotal | pinnable capacity is per-NUMA-node, not RLIMIT_MEMLOCK (INV-4.12f) |
| NVMe bandwidth ceiling | nvme device's PCIe link via sysfs: gen3 x4 → ~3.3 GB/s | a HARDWARE limit no config routes around (disk-ceiling finding); used to annotate cold-boot expectations, not to size |
| model shape | recipe `model` section, or HF `config.json` (field names already match per convention), or GGUF metadata | expert bytes for mixed “XL” GGUFs = per-projection MAX across layers (GG-9); read from GGUF tensor tables when the weights are present |

ModelShape derives: layers, MoE layers (`first_k_dense_replace`,
`moe_layer_freq`), KV-bearing layers (hybrids: `num_layers −
num_linear_attention_layers`, + MTP layer which IS KV-bearing —
TD-KV-POOL-SIZED-OVER-ALL-LAYERS), indexer geometry incl. computing-layer
census (`index_topk_freq`/`index_skip_topk_offset`: 21 of 79 on GLM-5.2),
IndexPool (`index_kpool`), KDA/linear-attention state geometry, and
`bytes_per_expert` for the quant route.

## 4. Constraint classes — what is baked vs what is data

**Physics (baked into the sizing model, never deleted by a commit):** per-GPU
VRAM; NUMA node capacities incl. HBM banks; host RAM; PCIe/NVMe link physics;
the model's layer/expert/KV geometry; byte-size formulas that follow from
quant formats.

**Engine-current facts (DATA, `python/autoconfig/engine_constraints.py`):**
rules that a future commit deletes. Each registry row carries `id`,
`statement`, `ticket` (the TD whose resolution removes or changes it), and the
solver effect. Deleting a row re-derives the config without touching solver
logic. Selected rows (`engine_constraints.py` is authoritative and has
grown past this table):

| id | statement | ticket |
|---|---|---|
| ~~`local-indexer-disables-tiering`~~ | DELETED 2026-08-30 — the dcp≥2 replicated-only `KvTieringManager` gate was a merge artifact; tiering is indexer-mode-agnostic (INV-KVT-20). Its replacement is the priced row below | TD-KVT-LOCAL-INDEXER-UNBLOCK (resolved) |
| `local-indexer-prefill-cost` | `dcp_indexer_mode: local` halves the per-rank indexer-K share (84→42 slabs, 87.1→43.5 MiB at max_seq 25600) and costs ~8-9% of served prefill (per-chunk-row cross-rank merge, per layer, B=64 rows); decode in-noise, token-identical. The mode is a CAPACITY decision with a known price | — (re-measure to change; evidence bound in the row) |
| `mla-tq-vs-snapmla-accuracy` | `turboquant_mla` is measurably less accurate than `snapmla`: teacher-forced dNLL +0.0282±0.0063 nats/token on glm5_next (ppl 6.542 vs 6.360; flips low-margin; gap flat to 7k ctx) for 0.50x KV bytes/row. Orders the `accuracy` lever's MLA ladder | — (re-measure with the internal teacher-forced NLL harness; evidence bound + falsifiers in the row) |
| `v4-no-sharded-kv` | deepseek_v4 rejects `dcp_kv_mode: sharded` | TD-V4-DCP-KV |
| `dspark-ctx-cap` | draft stops helping above `draft_context_capacity_tokens` (default 8192); over-cap prompts route to the plain arm | TD-DSPARK-CTX-POLICY |
| `glm5next-tiering-default-off` | glm5_next KV tiering ships default OFF (8.8× less KV/token than GLM-5.2 — it buys nothing at champion context); the E1 lattice escalates it ON for CAPACITY when the untiered pool cannot hold the conc × max_seq ask. The price is MEASURED (P-31 step 1): decode ~−10% @8k / −13% @24k, fresh prefill −16% vs untiered | — (measured; numbers live in the row) |
| `glm5next-tp1-default` | glm5_next decodes at TP=1 by default (KDA kernel latency-floored); for deep-context asks ≥ 262144 the TP plan orders the head-divisible ceiling FIRST — attention is the measured majority (60.4–73%) of the champion-scale fresh-prefill wall (P-30) | — (measured) |
| `kda-state-fixed-carve` | glm5_next KDA state is a fixed per-request carve, `slots = max_concurrent + prefix_entries`, not lendable | TD-KDA-STATE-MAPPED-SLABS |
| `kv-pool-counts-all-layers` | the ENGINE sizes the KV pool over ALL layers even on hybrids; the solver must model the engine as it is, while reporting the waste | TD-KV-POOL-SIZED-OVER-ALL-LAYERS |
| `tp-gpus-must-be-flagship` | every TP GPU must be the box's top attention class (INV-0.5: rtx5090) | — (validator rule) |
| `expert-host-prefix` | the expert-host set is the PREFIX of `hardware.gpus` up to the first non-expert-host entry | — (ring_bridge/dispatch_reef scan) |

**Combination-keyed per-model rows (DATA,
`python/autoconfig/model_constraints.json`, loaded by
`combo_constraints.py`):** rules that bind a COMBINATION of choices — the
scope of an `engine_constraints` row is one arch; a combination row's `when`
is a predicate set over a closed key vocabulary (`PREDICATE_KEYS`: `arch`,
`tensor_parallelism`; a future GPU predicate is a new key + its ctx feed,
not a new mechanism). A row is consulted ONLY when every predicate matches
(a tp=1 glm5_next derivation never sees the tp>1 row — nothing is
over-constrained); its `force` names the only value the engine accepts
under the combination. Combination rows are ENGINE-CURRENT FACTS (kind
`engine_gate`/`validator`, never measured/heuristic): a firing row explains
with its own kind `engine_gate` citing the enforcing engine site, a
contradicting `--pin` REFUSES quoting both pins and the row, and deleting
the row re-derives with no code edit. The loader is fail-closed: unknown
predicate keys/operators refuse at import rather than silently un-firing a
gate. Shipped rows:

| id | when ⇒ force | ticket |
|---|---|---|
| `glm5next-no-sharded-kv` | arch == glm5_next AND tensor_parallelism > 1 ⇒ `hardware.dcp_kv_mode: replicated` (KDA recurrent state is whole-sequence — no token shard to own; the engine fail-closes, `engine.cpp:637-643`, GF3.10) | TD-AUTOCONFIG-GLM5NEXT-TP-SHARDED-KV |

**Measured policy (heuristic constants with provenance, in
`_internal-autoconfig`):** sharded-KV wins ≈ +4.6% over replicated at dcp=2
(arena-placement campaign); HBM-node pinned `fraction_free` 0.6 (0.8 measured
too tight — dossier §1); host arena `fraction_total` 0.9; TP-local NUMA nodes
excluded from spill (D2H NUMA-aware rule); `vram_safety_margin_gb` 2.25
(champion-proven; KvTiering device pools allocate OUTSIDE the carve, i.e. out
of this margin — `kv_tiering_manager.cpp:189-245`).

## 5. The derivation ladder

Ordered stages; later stages consume earlier results. Marking:
[C] closed-form, [H] documented heuristic, [S] searched.

- **P1 [C] GPU classes and ordering.** Sort GPUs best-first (VRAM, then
  compute weight table {5090:1.0, 5080:0.5}); `hardware.gpus` is emitted in
  this order because the engine takes the FIRST `tensor_parallelism` entries
  as the TP group and scans the expert-host prefix.
- **P2 [C] TP degree + tp_array.** Largest power-of-2 run of the top class
  (mirrors `auto_detect_tp_array`), capped by head divisibility
  (`num_attention_heads % tp == 0`; per-rank heads ≤ 128; KDA heads % tp;
  o_groups rules for V4). `tp_array` = POSITIONS `[0..tp-1]` (matches both
  shipped recipes and the engine's first-N rule; the id-vs-position validator
  ambiguity is documented in the emitter).
- **P3 [C] roles.** TP GPUs: `[attention, resident, expert_streaming]`;
  non-TP expert hosts: `[expert_streaming]`; all listed before any
  attention-only GPU (expert-host-prefix row).
- **P4 [H] DCP/KV mode.** tp≥2 → `dcp_kv_mode: sharded` unless a registry
  row forbids it: the arch row `v4-no-sharded-kv`, or a combination row
  (`glm5next-no-sharded-kv`: glm5_next at tp>1 runs replicated only —
  explained with kind `engine_gate`, citing `engine.cpp:637`). The +4.6%
  sharded measurement (`sharded-kv-beats-replicated`) is ARCH-SCOPED to
  glm_moe_dsa, where it was measured; off-scope archs take sharded as an
  unpriced structural default (kind `heuristic`). A `dcp_kv_mode` pin
  contradicting a firing combination row refuses (both pinned ⇒ at solve()
  entry, quoting both pins and the row); with tp unpinned the gated tp
  lanes leave the plan loudly (adapt-around) instead.
- **P5 [H→S] Indexer mode — a capacity decision with a known price.** Not a
  legality question any more (the tiering gate fell, INV-KVT-20) and not a
  free VRAM win either: registry row `local-indexer-prefill-cost` prices
  `local` at ~8-9% of served prefill for half the per-rank indexer-K share
  (84→42 slabs, 87.1→43.5 MiB at max_seq 25600 — the solver's own K2 sizing
  reproduces both numbers). So P5 emits an ORDERED CANDIDATE LIST,
  `(replicated, local)`, and E1 takes the first that fits at each rung: the
  fast mode while its bytes are affordable, the cheap mode when they are
  contested. The explanation always says WHY, and when it takes `local` it
  NAMES THE PRICE — the same "capacity, not speed" pattern as the
  `glm5next-tp1-default` TP row. The mode is spent BEFORE the draft and
  before any capacity shed (it keeps the user's ask whole), and deeper rungs
  re-try `replicated` first because a rung that freed GiB may afford it
  again — `prefer` (§2.3) is what moves that boundary. Two rows still narrow
  the menu to `replicated` alone: a
  re-introduced `local-indexer-disables-tiering` (tiering would be lost), and
  dcp<2 (nothing to localise). V4 and non-DSA archs keep the plain
  VRAM-cheaper heuristic — neither measured nor modelled by the price row —
  with the per-model note that `local`'s saving is arch-dependent (IndexPool
  pools are ~kpool× cheaper).
- **K1 [C→S] Active-context split.** As §2.2; feasibility loop (§6) may
  degrade `max_concurrent_requests` first, then `max_sequence_length`.
- **K2 [C] Position-state demand per sequence.** Transcribed engine formulas
  (`python/autoconfig/sizing.py` = `vram_allocator.cpp` in Python):
  KV bytes/token (MLA/TQ/MHA arms), KV pages with the sharded-DCP rank-0
  worst-case cycle math, indexer-K pages over COMPUTING layers with
  IndexShare census and local-mode ceil-divide, S1 slab geometry + S4
  elastic share, V4 side tiers (CSA/HCA/SWA/LID + holder scaling), KDA slots.
- **K3 [H] KV tiering.** ON iff the model has DSA (`index_topk > 0`), the
  KV format is row-self-contained, and the registry rows allow it.
  `hot_buffer_slots = index_topk` (champion-measured; engine auto would be
  2×topk), `host_to_device_ratio` = 8 default, bounded by host pinned budget;
  `tiered_prefill: true` (INV-KVT-16 windowed admission).
- **E1 [S] Expert/KV/draft fit — the one genuinely coupled step.** Deterministic
  search, no randomness: the lattice of §2.3 is built ONCE
  (`tp_idx × idx_idx × (draft × capacity) chain`), sorted by the `prefer` key,
  and walked to the first point that fits.  At each point every GPU's carve is
  computed with the K2 sizes
  and the expert reserve implied by `vram_expert_ratio`; the point fails if any
  carve goes negative or under-floors (`min_expert_cache = num_experts_per_tok ×
  bytes_per_expert` on expert hosts).  The default `balanced` order is the
  shipped ladder:
  (0) escalate the indexer mode to `local` (P5 — it keeps the ask AND the
  draft whole, at a measured ~8-9% of prefill; every rung below is tried at
  both modes, `replicated` first), (1) drop the draft (the user's
  active-context ask outranks the optional accelerator), (2) drop concurrency
  toward 1, (3) shorten max_sequence_length, (4) escalate TP where a second
  candidate exists, (5) refuse (§6).  `prefer=speed` moves both priced axes
  behind the whole capacity ladder; `prefer=capacity` moves both in front of
  it.  A `--pin` REMOVES its axis before any of this: the lattice is built
  with the pinned axis collapsed to one point, so no rung and no preference
  can trade the pinned value away (§2.5).  Expert reserve is then distributed
  across expert hosts proportionally to residual capacity → per-GPU
  `expert_streaming` overrides.  Only the WINNING point is explained — search
  probes emit nothing (`Solver._quiet`), and warnings survive only from the
  attempts the chosen order actually visited.
- **P6 [H] Draft (dspark) decision.** Run a draft iff (a) a `.dspark`
  checkpoint is discoverable for the model, (b) after charging its weight +
  scratch bytes (sized from the checkpoint header, CPU-only; `kv_arena` term
  linear in `draft_context_capacity_tokens`) the E1 fit still holds at the
  requested levers. Placement: sharded across the TP GPUs when the checkpoint
  is safetensors (TD-DSPARK-DRAFT-SHARD measured win), else the first non-TP
  GPU. On a tight box E1's degradation order drops the draft FIRST, before
  any capacity is shed — that VRAM is worth more as experts or KV (the
  ticket's 63-vs-64-slot lesson).
- **H1 [H] Host arena + NUMA plan.** Arena on GPU-attached nodes
  (`fraction_total` 0.9); `cross_node_spill` on all remaining memory nodes
  EXCEPT nodes local to a TP GPU (D2H NUMA-aware rule — reproduces the
  champion's `[0,1,4,5,6,7]` on this box), HBM banks at `fraction_free` 0.6
  (measured; champion's 0.8 is flagged as a known-tight divergence).
- **M1 [C] Mechanical knobs.** Page sizes, chunk sizes, prefix-cache budgets,
  quant route, preload/holder policy: carried from the arch family's proven
  defaults (data-backed template per architecture), each with its provenance
  line. These are [C] copies, not judgements — the solver's value is knowing
  WHICH template applies and when a knob must deviate (e.g. `dcp_chunk_size`
  must be a multiple of `page_size_tokens` for tiering shard_ok;
  `indexer_k_page_size_tokens % index_kpool == 0`).

## 5b. Long-context runtime plan (P-31, 2026-09-06)

Four derivations that close TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH and
TD-AUTOCONFIG-STRIDE-NOT-DERIVED; the sizing lives in
`sizing.runtime_scratch` / `moe_big_transient_bytes` (engine transcriptions
with file:line cites) and the plan in `Solver._longctx_plan`:

- **The block semantics are the constraint.** The VramAllocator claims
  `vram_gb − margin` as ONE physical allocation; every max_seq/max_batch/
  stride-scaled runtime allocation (block tables, attention prefill
  staging, RoPE tables, indexer/KDA executor scratch, KVT union staging,
  logits scratch, MoE persistent set) lives OUTSIDE it, funded only by the
  margin + the physical slack above the declared `vram_gb`. Two phases,
  split at the MoE-big elastic fit check (CommandDispatcher ctor): PREFIT
  allocations reduce the free the check samples; POSTCHECK allocations are
  what the per-class `compute.moe_big_fit_headroom_mb` must reserve room
  for (or the boot dies at the block-table alloc).
- **`orchestrator.max_batch_size`** derives as the largest halving of the
  family cap whose `dev_block_tables` fit `block_table_budget_gb` (1.5):
  512 @1M/page16 would be 5.75 GiB/rank.
- **`compute.prefill_superchunk_tokens` = `compute.moe_big_chunk_tokens`**
  (the EP diagonal, TD-MOE-EP-XTP-WAVES) derives as max-S-that-fits on the
  engine's own per-device-class fit arithmetic (P-30 step 2): the
  expert-only class binds first (no scratch tail); the attention hosts are
  funded by raising the TP margin, capped by `stride_margin_cap_gb`. A
  SPEED-axis spend — never the accuracy ladder. The engine's elastic
  step-down remains the runtime guard (never-OOM, INV-MOE-BIG-ROWS).
- **`compute.moe_big_fit_headroom_mb`** derives as the post-check demand
  × `moe_fit_headroom_cushion` when the stride exceeds the chunk floor.
- **Per-TP-GPU `safety_margin_gb`** derives as base + whatever the above
  needs beyond the modeled free-at-fit (class anchors
  `moe_fit_fixed_{attn,expert}_mib`, calibrated to P-30 step 2's measured
  1668/1685 MiB on this box; heuristic elsewhere — the engine's fit log is
  the truth to reconcile against on a new box).
- **Tiering is an E1 axis on glm5_next** (candidates OFF→ON): the untiered
  branch refuses `conc-ask-exceeds-untiered-pool` when the pool cannot
  hold conc × one request's whole-life demand (§2.2's own definition of
  the ask), and the lattice escalates to tiering (INV-KVT-16 windowed
  admission) before any other axis moves. The escalation's price is
  measured (§0; `glm5next-tiering-default-off` row): ~−10–13% decode,
  −16% fresh prefill — which is exactly why OFF stays the default
  candidate wherever the untiered pool affords the ask.
- **TP plan orders the ceiling first for asks ≥ 262144 tokens** on
  glm5_next: attention is the measured majority of the large-prefill wall
  (P-30 steps 1-4; 60.4-73%, stride-invariant) and head-sharding is the
  only lever that touches it; the tp1 decode default governs short asks.

## 6. Feasibility and refusal

A refusal is a first-class output: `Infeasible(constraint_id, binding_gpu,
requested, affordable, suggestion)`. Triggers include: indexer-K for the
requested active context exceeding any TP GPU's carve (the 1M-on-a-small-box
case); `expert_total < min_expert_cache` (`vram_allocator.cpp:1259-1267`
would throw at boot — we refuse at derivation time instead); KDA slots < 1;
host arena nodes yielding zero slots (INV-4.12f). The refusal names the
constraint in the same vocabulary as the explanation lines, and reports the
largest feasible lever value found by the E1 descent so the user can re-ask —
specifically the VRAM-cheapest corner of the lattice (every priced option
spent, the ask shed to its floor, no draft), so the number does not depend on
the search order and therefore not on `prefer` (§2.3). When pins are present
the refusal must additionally say WHICH pin binds (§2.5): each pin is
relaxed alone and the ones whose relaxation restores a fit are named; only
when the box is infeasible regardless of pins does the ordinary refusal
above stand.

## 7. Fingerprint

Config-level intent, never volatile measurements (INV-ARENA-CACHE-ORDER
lesson; slot counts drift ±1 under `fraction_free`). Fingerprint = SHA-256
over the canonical JSON of: GPU list (name, PCI bus id, physical VRAM GiB,
max PCIe gen/width, NUMA node), NUMA node list (id, MiB total rounded to GiB,
is_hbm), MemTotal rounded to GiB, NVMe (gen, width) list. Free memory,
link training state, clocks, driver versions are EXCLUDED. Stored in the
emitted recipe's `autoconfig.fingerprint`; compared on `--autoconfig` boots.

## 8. Explainability

Every derived field gets an `Explanation{path, value, kind, formula, because,
refs}`; the emitter writes them as (a) `<out>.explain.md` sidecar table and
(b) log lines in the established boot-log voice, e.g.

    autoconfig: expert_streaming 11.5 GiB on GPU 1 (rtx5080) — residual after
    pinned 0.6 + margin 2.25 on a 15 GiB carve; expert host, no KV tenant
    (roles=[expert_streaming]); lever vram_expert_ratio=0.066 satisfied
    globally (1264/19200 slots)

`refs` name spec sections, tickets, or registry row ids — a number no one can
trace to a constraint is a number no one can debug.

## 9. Acceptance

`recipes/glm52_serve_champion.json` is the role model: the solver, given this
box's descriptor + the GLM-5.2 shape + champion-implied levers (0.066, 51200),
must land at or near it. `python/autoconfig/compare.py` diffs field-by-field
and classifies every divergence as one of: `match`, `derived-equal`
(different bytes, same derived meaning), `solver-choice` (explained, with the
registry/measurement that justifies it), or `unexplained` (a solver bug until
proven a discovery). The unit suite pins the classification so a regression
in the solver surfaces as new `unexplained` rows.

## 10. The auto-run pipeline

§1-§9 describe the DERIVATION. This section describes the one command a user
actually runs, which wraps it:

    python python/cli/autoconfigure.py --model <weights>

**Step 0 — probe (`modelprobe.py`).** A model path is enough. The recipe
`model` section, the quant route, the prepacked store, the tokenizer dir and a
`.dspark` draft are all derived from the weights: GGUF metadata KV block, or a
HuggingFace `config.json` when one is present, or (advanced) a base recipe via
`--config`. GGUF cannot express a few fields that are NOT cosmetic —
`index_topk_freq`/`index_skip_topk_offset` (IndexShare geometry: at
`index_topk_freq <= 0` the engine treats every layer as a full indexer layer,
79 instead of 21 on GLM-5.2) and the rope interleave flags — so a per-
architecture PROFILE supplies them, as cited data (internal model reference),
in the spirit of `engine_constraints.py`. An unknown architecture is a
refusal, not a guess. Profiles exist for the bring-up archs including GGUF
`glm5next` (P-31 closed TD-AUTOCONFIG-NO-GLM5NEXT-PROFILE: layer_types from
the `attention.head_count_kv` array, linear_attn_config, index_kpool,
hc_*, swiglu_limit), so `--model` alone is enough for GLM-5.3-class
weights. Prepack discovery is by manifest identity
(`source_model_path`), never by name; the tokenizer must be a real path
because serve's `auto` only searches the weights dir and GGUF-embedded
tokenizers are not extracted (TD-SERVE-GGUF-TOKENIZER).

**Step 1 — calibrate.** GPU capability is MEASURED, never inferred from card
names (§3), which means the artifact must exist. If none is accepted for this
box+model, the pipeline emits a PROVISIONAL recipe (`calibration_mode: full`),
boots it once, and the engine writes the artifact itself
(`load_or_calibrate_with`). The boot is guarded by the registry row
`full-calibration-needs-cold-boot`: a running arena holder is detected and
refused, because the calibration's NUMA-bound footprint node-OOMs against a
warm store.

**Step 2 — derive.** §5, unchanged, now with an accepted calibration.

**Step 3 — train.** The hardware calibration predicts; a real workload
corrects. The pipeline boots the derived recipe with the loader shadow solver
and the perf trace on (`LS_LOADER_SHADOW_DUMP`, `LS_PERF_TRACE_OUT`), decodes
100 tokens through the ordinary serving path, then runs
`tools/loader_xray/trainer_apply.py` to fit predicted-vs-actual and bake
corrected constants into a SEPARATE artifact (`..._trained.json`). The
emitted recipe's `gpu_loader.calibration_path` then names the trained file:
it is the second iteration of the same constants, and serving must use the
better one. The untrained baseline is kept — the trainer needs it as its
starting point on every refit.

**Step 4 — place.** The pinned host arena places expert slots across NUMA
banks by per-(layer, expert) DEMAND-FETCH frequency (M3, `arena_placement.h`).
That table is measured, not derivable — but step 3's decode already emitted
the trace it is fitted from, so step 4 is CPU-only whenever step 3 ran:
`tools/loader_xray/freq_table.py` counts dispatched expert H2D copies per key
(excluding prefill and warm-up sweeps) and the emitted recipe carries the
result in `memory.arena_placement.freq_table`. Three honest limits are stated
in the explanation: the fit is a BOOTSTRAP from one regime (more traces
accumulate into a better table), the M3b online migrator refines it live
either way, and the table's content hash folds into the ArenaCache store
identity, so the first boot with it rebuilds the warm store once.

**The pipeline finishes (user directive, 2026-08-30).** Reusing or supplying an
artifact scopes to THAT step; it never truncates the run. With no trace
available — step 3 reused, supplied, or skipped — step 4 captures one itself
and fits, leaving whatever satisfied step 3 untouched (the capture boot emits
a perf trace only: no shadow dump, no `trainer_apply`), and says so in the
output so a boot that did not retrain never reads like one that did.
`--skip-placement` is the single way to end without a table; `--skip-training`
scopes to step 3 alone.

**Naming and reuse.** One config per model, one artifact SET per model:
`<stem>.autoconfig.json`, `gpu_loader_calibration_<stem>{,_trained}.json` and
`arena_placement_<stem>.csv`,
where `<stem>` defaults to the model's own name and is overridable with
`--name`/`--prefix`. Both measured artifacts are reused when they match this
box AND this model (the engine's own accept predicate: device UUIDs +
compute_N/K), re-measured on `--reset`, and supplied verbatim with
`--calibration`/`--trained` — which is also how a derivation runs with no GPU
time at all. A calibration is not conceptually per-model, but the engine
REJECTS one whose compute dims belong to a different model
(`calibration.py:accept_calibration` mirrors `engine.cpp:519-568`), so in
practice it is model-scoped and lives with the weights.

**What the pipeline does NOT do.** It does not serve (`serve.py --autoconfig`
already exists for derive-then-serve), and it never invents a measured
artifact: each of the three is either measured, reused, supplied, or absent
WITH an explanation naming what replaces it.

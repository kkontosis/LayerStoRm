# Design Revisions

**Date: 2026-08-24.** [`DESIGN.md`](DESIGN.md) is the project's initial
architecture document. The system it describes was largely built — but
implementation experience revised several of its mechanisms, and this file
records the divergences honestly. Read the two together: `DESIGN.md` for the
argument and the mechanisms at depth, this file for what reality changed.

## 1. PreScope (§5) — not used

The one-layer-ahead gating lookahead and its learned (LLaPor-style) extension
were never wired into the production path. In practice, routing is computed
in-line within the fused per-layer flows (the fused forward command and the
superchunk prefill drive), and the measured prefetch levers of this family
*lost* on the DMA-bound fetch wall: the tier-0 prefetch campaign closed
default-OFF, because adding speculative bytes to a bandwidth-bound link hurts
even at ~21% prediction precision.

## 2. MoE-SpeQ (§5, fuser) — not really used

It exists as a trained expert-prediction model line (offline, AUC-gated), but
its integration is deferred; the exact look-ahead variant failed its coverage
gate and was killed. What *is* used from this idea-space is different: during
speculative decoding, the verify batch's **expert union** drives fetching
directly — a deterministic union, not a probabilistic hint.

## 3. The predictor ensemble and prefetch fuser (§2, §5) — absent

PROBE was never built; SP-MoE-as-hint-source, the α/β/γ/δ fusion formula,
`PrefetchHint` records, and Stream 5 (`kPrefetchCompute`) are inert. The
"single decision authority" *principle* held — but the authority is the
REEF/I8 placement solver deciding per fetch command, with **epoch-latched,
NUMA-paired bank inputs** (INV-REEF-BANK: solver inputs are frozen per
placement epoch and refresh only at migrator-commit boundaries) — a mechanism
the initial design lacks entirely.

## 4. Speculation drafters (§11 note) — different than specified

The document lists native-MTP, prompt-lookup, and self-speculative drafts;
none shipped. The implemented system is **DSpark**: an external draft model
with strictly lossless batched/overlap verification and KV rewind. The
models' MTP heads are unused (GLM-5.2 and DeepSeek-V4 both run with
`nextn` disabled). The LoRA-on-MTP expert-prediction extension remains
unbuilt research.

## 5. Performance model (§10) — superseded in one key respect

The decode wall proved to be host-side **completion detection**, not H2D
bandwidth: the GPUs DMA in parallel at full per-link rate, and the single
daemon thread's arrival polling dominated the MoE wall. Aggregate fetch is
bounded by source-NUMA DRAM physics, not by the sum of per-link rates.
Prefill economics changed category with **superchunk prefill** (one
expert-union stream per layer per superchunk, making the prefill wall
essentially length-independent) — a mechanism that post-dates the document.

## 6. §11's not-yet-implemented list — still accurate, with measured caveats

The per-device kernel-strategy solver, predictive H2D backfill, and
pinned-layer rotation remain unimplemented. Backfill's value is now known to
be bounded by the source-DRAM ceiling above, which the design predates.

## 7. Additions not covered by the initial document

TurboQuant and SnapMLA KV codec arms, HiSparse KV tiering, prefix caching
with chain-aware eviction, the persistent arena holder with its reasoned-wipe
contract, guided decoding, DeepSeek-V4 support via the arch-split
attention/MoE drivers, and mini-superchunk served prefill. See the README and
the changelog for these.

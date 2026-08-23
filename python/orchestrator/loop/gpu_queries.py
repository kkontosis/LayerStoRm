"""GPU state queries — read-only snapshots of GPU / cache / expert state.

These helper methods query the StateReader (seqlock-based zero-copy
numpy view into the C++ daemon's published state) and the orchestrator's
internal bookkeeping to produce per-GPU summaries used by Phase 4 (PLAN).

Each method returns a dict or list keyed by GPU position (0..N-1).
All reads are non-blocking.  Exceptions from StateReader (e.g. stale
seqlock, uninitialized snapshot) are caught and produce safe defaults.

Methods here are mixed into OrchestratorLoop via _GpuQueryMixin.

IMPORTANT: Keep this file richly commented.  The orchestrator loop is the
most complex module in the system and every helper must be self-documenting.
Do not remove comments during edits — add more if anything is unclear.
"""

from __future__ import annotations

import numpy as np

from orchestrator.shm_protocol import MAX_EXPERTS
from orchestrator.types import ExpertEvictionInput, ExpertKey, WorkStatus


class _GpuQueryMixin:
    """Mixin: GPU state query helpers for the orchestrator loop."""

    def _compute_queue_depths(self) -> dict[int, int]:
        """Count dispatched (in-flight) work items per GPU.

        Used by the GPU assigner to avoid overloading busy GPUs.
        """
        depths: dict[int, int] = {
            g.position: 0 for g in self._metadata.gpus
        }
        for item in self._work_queue.by_status(WorkStatus.DISPATCHED):
            depths[item.target_gpu] = (
                depths.get(item.target_gpu, 0) + 1
            )
        return depths

    def _compute_vram_free(self) -> dict[int, int]:
        """Compute free VRAM bytes per GPU from the state snapshot.

        Falls back to full VRAM capacity on read errors (e.g. during
        the first cycle before the daemon publishes state).
        """
        result: dict[int, int] = {}
        for g in self._metadata.gpus:
            try:
                used, total = self._state_reader.vram_usage(g.position)
                result[g.position] = max(0, total - used)
            except Exception:
                result[g.position] = g.vram_bytes
        return result

    def _compute_free_slots(self) -> dict[int, tuple[int, int]]:
        """Compute free cache slots per GPU: (stable_free, streaming_free).

        Read from the daemon's cache_fill counters: (stable_used,
        stable_total, streaming_used, streaming_total) per GPU.
        """
        result: dict[int, tuple[int, int]] = {}
        for g in self._metadata.gpus:
            try:
                su, st, ru, rt = self._state_reader.cache_fill(g.position)
                result[g.position] = (st - su, rt - ru)
            except Exception:
                result[g.position] = (0, 0)
        return result

    def _compute_gpu_capacities(self) -> list[int]:
        """Get total stable cache capacity per GPU (for placement optimizer).

        Returns a list indexed by GPU position.  Used by
        PlacementOptimizer.maybe_reoptimize() to decide whether
        re-placement is needed.
        """
        caps = []
        for g in self._metadata.gpus:
            try:
                _, st, _, _ = self._state_reader.cache_fill(g.position)
                caps.append(st)
            except Exception:
                caps.append(0)
        return caps

    def _compute_host_resident_keys(self) -> set[ExpertKey]:
        """Build the set of experts currently warm in host RAM.

        Used by PlacementOptimizer to decide NUMA migrations.

        #91: reads the WHOLE host-resident bitmap in ONE seqlock
        transaction (StateReader.host_warm_bitmap) and decodes it with
        numpy.  The previous per-(layer, expert) is_host_warm() scan was
        num_moe_layers × num_experts separate transactions — ~seconds per
        cycle against a live daemon (each read retries under publish
        churn), the dominant term of the #91 crawl.  It also indexed the
        bitmap by ABSOLUTE layer while the C++ ELM writes MOE-RELATIVE
        indices (expert_lifecycle_manager.cpp publish_host_state) — wrong
        rows AND out-of-slot reads past MAX_MOE_LAYERS.  Returned keys
        carry ABSOLUTE layer_idx (the orchestrator-wide ExpertKey
        convention).
        """
        try:
            bitmap = self._state_reader.host_warm_bitmap()
        except Exception:
            return set()
        # TD-ORCH-PLAN-CRAWL: this runs EVERY Phase-4 PLAN cycle and the
        # full-fit arena keeps ~19k warm bits — decoding them into 19k
        # ExpertKeys per cycle was a dominant crawl term. The bitmap only
        # changes when host arena contents change (rare in steady state):
        # cache the decoded set keyed on the raw bitmap bytes. Callers
        # treat the set as read-only (process_numa_migrations).
        cached = getattr(self, "_host_resident_cache", None)
        if cached is not None and cached[0] == bitmap:
            return cached[1]
        bits = np.unpackbits(
            np.frombuffer(bitmap, dtype=np.uint8), bitorder="little",
        )
        num_moe = self._metadata.num_moe_layers
        num_exp = self._metadata.num_experts
        idxs = np.nonzero(bits)[0]
        moe_layer, expert = np.divmod(idxs, MAX_EXPERTS)
        valid = (moe_layer < num_moe) & (expert < num_exp)
        first = self._first_moe_layer
        result: set[ExpertKey] = {
            ExpertKey(l, e) for l, e in zip(
                (moe_layer[valid] + first).tolist(),
                expert[valid].tolist(),
            )
        }
        self._host_resident_cache = (bitmap, result)
        return result

    def _build_eviction_inputs(self) -> list[ExpertEvictionInput]:
        """Build eviction scoring inputs for all resident experts.

        This is the hot path in Phase 4 (PLAN).  With 64 resident experts
        and mock state reader, the cost is ~141μs — dominated by Python
        loop overhead (dict/set iteration, method calls, dataclass construction).

        Optimization history:
          - Batch stat reader: 1 seqlock transaction instead of N.  Saves real
            time with actual shared memory (N seqlock retries avoided), but no
            measurable difference with mock readers.
          - Numpy zone vectorization: TRIED AND REVERTED.  np.array construction
            overhead exceeded per-element savings at N=64.  Numpy wins at N>500.
          - The remaining cost is Python-object overhead: 64 recommended_zone()
            calls + 64 frozen-dataclass constructions.  The real fix for <100μs
            at N=64 is the skip-when-no-evictions guard (#13c-1 item 2), which
            eliminates this function call entirely on most cycles.

        Fed to TransferScheduler.plan_evictions() via EvictionPolicy scorer.
        """
        # Collect all (gpu_idx, key) pairs across all GPUs
        all_pairs: list[tuple[int, ExpertKey]] = []
        for gpu_idx, keys in self._resident_keys.items():
            for key in keys:
                all_pairs.append((gpu_idx, key))

        if not all_pairs:
            return []

        # Batch-read all expert stats in one seqlock transaction.
        # With real shared memory this avoids N separate seqlock retries.
        # #91: snapshot expert arrays are MOE-RELATIVE-indexed (the C++
        # StatePublisher writes at (layer - first_moe_layer) * kMaxExperts
        # + expert) while ExpertKey carries the ABSOLUTE layer — convert.
        # Out-of-range rows (dense layers, layers >= MAX_MOE_LAYERS) fall
        # back to zero stats inside expert_stats_batch.
        stat_keys = [
            (key.layer_idx - self._first_moe_layer, key.expert_idx)
            for _, key in all_pairs
        ]
        try:
            stats = self._state_reader.expert_stats_batch(stat_keys)
        except Exception:
            stats = [(0.0, 0.0, 0.0, 0.0)] * len(all_pairs)

        # Build inputs with per-expert zone recommendation.
        # At N=64 the per-call overhead of recommended_zone() + dataclass
        # construction is the dominant cost (~2μs/expert).  Numpy vectorization
        # was tried but the array construction overhead made it worse at N<500.
        inputs: list[ExpertEvictionInput] = []
        for (gpu_idx, key), (freq, rec, rw, ac) in zip(all_pairs, stats):
            zone = self._expert_placement.recommended_zone(key, gpu_idx)
            inputs.append(ExpertEvictionInput(
                key=key, zone=zone, gpu_idx=gpu_idx,
                recency=rec, frequency=freq,
                routing_weight=rw, temporal_autocorr=ac,
            ))
        return inputs

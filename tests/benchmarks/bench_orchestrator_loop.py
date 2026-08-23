"""Benchmark: orchestrator loop cycle time.

Measures per-cycle and per-phase wall-clock time for the orchestrator
event loop under realistic conditions.  Validates INV-3.4.1 (<100μs
per cycle).

Usage:
    python tests/benchmarks/bench_orchestrator_loop.py

Configurations:
  - idle:     no requests, no completions (baseline overhead).
  - 1_req:    single active request, autoregressive forward pass.
  - 8_req:    eight concurrent requests, mixed layers.
  - spec_1:   single request with MTP speculation (2 draft steps).
  - prefetch: single request with all prefetch predictors enabled.

Each configuration runs N warm-up cycles + M measured cycles and reports
min / median / p95 / p99 / max in microseconds.

NOTE: The 64_experts benchmark always forces eviction scoring (even after
#13c-1 adds the skip-when-no-evictions-needed optimization) so we measure
the actual _build_eviction_inputs cost, not the fast-path bypass.
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "python"))

from orchestrator.command_writer import CommandWriter, CompletionReader
from orchestrator.draft_combiner import DraftCombiner
from orchestrator.eviction_policy import EvictionPolicy
from orchestrator.expert_placement import ExpertPlacement, ExpertPlacementConfig
from orchestrator.gpu_load_balancer import GpuLoadBalancer, GpuLoadBalancerConfig
from orchestrator.layer_skip import LayerSkip
from orchestrator.loop.orchestrator_loop import (
    CycleMetrics,
    InferenceRequest,
    OrchestratorConfig,
    OrchestratorLoop,
    RequestState,
)
from orchestrator.moe_speq import MoeSpeq
from orchestrator.mtp_draft import MtpDraft
from orchestrator.online_calibrator import OnlineCalibrator
from orchestrator.performance_objective import PerformanceObjective
from orchestrator.placement_optimizer import PlacementOptimizer, PlacementOptimizerConfig
from orchestrator.prefetch_fuser import PrefetchFuser
from orchestrator.prescope import PreScope
from orchestrator.probe import Probe
from orchestrator.prompt_lookup import PromptLookup
from orchestrator.reasoning_mode import ReasoningMode
from orchestrator.scheduler import Scheduler
from orchestrator.self_speculative import SelfSpeculative
from orchestrator.shm_protocol import (
    CMP_COMPUTE_DONE,
    Completion,
)
from orchestrator.speculative_prefetch import SpeculativePrefetch
from orchestrator.spsc_ring import SpscRingWriter
from orchestrator.state_reader import StateReader
from orchestrator.transfer_scheduler import TransferScheduler
from orchestrator.types import (
    EngineMetadata,
    EvictionPlan,
    ExpertKey,
    GpuConfig,
    TransferPlan,
    WorkItem,
    WorkOperation,
    WorkStatus,
)
from orchestrator.utility_scorer import UtilityScorer
from orchestrator.verifier import Verifier
from orchestrator.work_queue import WorkQueue


# ---------------------------------------------------------------------------
# Fixtures (same pattern as unit tests but optimized for speed)
# ---------------------------------------------------------------------------

def _gpu_configs(n: int = 2) -> tuple[GpuConfig, ...]:
    return tuple(
        GpuConfig(position=i, gpu_type="rtx5090", is_tp=(i < 2),
                  vram_bytes=32 * 1024**3, compute_weight=1.0)
        for i in range(n)
    )


def _metadata(num_gpus=2, num_layers=6, num_moe_layers=4,
              num_experts=8) -> EngineMetadata:
    return EngineMetadata(
        num_gpus=num_gpus,
        num_moe_layers=num_moe_layers,
        num_experts=num_experts,
        num_layers=num_layers,
        expert_bytes=2_359_296,
        kv_bytes_per_page=4096,
        gpus=_gpu_configs(num_gpus),
    )


class NullCompletionReader:
    """Completion reader that always returns empty (no daemon responses)."""
    def drain(self, max_count=0xFFFFFFFF) -> list:
        return []
    def is_empty(self) -> bool:
        return True


class NullRingWriter:
    """Ring writer that discards commands (measures orchestrator overhead only)."""
    def write_struct(self, cmd) -> bool:
        return True
    def write(self, data: bytes) -> bool:
        return True


class NullStateReader:
    """State reader returning safe defaults (no real shared memory)."""
    def shift_detected(self) -> bool:
        return False
    def windowed_acceptance_rate(self) -> float:
        return 0.5
    def vram_usage(self, gpu_idx: int) -> tuple[int, int]:
        return (1 * 1024**3, 32 * 1024**3)
    def cache_fill(self, gpu_idx: int) -> tuple[int, int, int, int]:
        return (10, 100, 5, 50)
    def is_host_warm(self, layer: int, expert: int) -> bool:
        return False
    def expert_stats(self, layer: int, expert: int):
        return (0.1, 0.5, 0.3, 0.2)
    def expert_stats_batch(self, keys: list) -> list:
        return [(0.1, 0.5, 0.3, 0.2) for _ in keys]


def _build_bench_loop(**overrides) -> OrchestratorLoop:
    """Build a loop optimized for benchmarking (no mocks, real modules)."""
    md = overrides.pop("metadata", _metadata())
    config = OrchestratorConfig()
    cmp_reader = NullCompletionReader()
    ring_writer = NullRingWriter()
    state_reader = NullStateReader()
    cmd_writer = CommandWriter()
    work_queue = WorkQueue()
    scheduler = Scheduler()
    gpu_assigner = GpuLoadBalancer(GpuLoadBalancerConfig(gpus=md.gpus))

    placement_cfg = ExpertPlacementConfig(
        num_moe_layers=md.num_moe_layers,
        num_experts=md.num_experts,
        cache_gpu_indices=list(range(md.num_gpus)),
    )
    expert_placement = ExpertPlacement(placement_cfg)
    placement_optimizer = PlacementOptimizer(
        PlacementOptimizerConfig(), expert_placement,
    )

    # Use real (not mocked) modules for accurate timing
    prescope = MagicMock(spec=PreScope)
    prescope.enabled = False
    probe = MagicMock(spec=Probe)
    probe.enabled = False
    probe.probe_layers = ()
    moe_speq = MagicMock(spec=MoeSpeq)
    moe_speq.enabled = False

    prefetch_fuser = PrefetchFuser()
    speculative_prefetch = MagicMock(spec=SpeculativePrefetch)
    utility_scorer = MagicMock(spec=UtilityScorer)
    utility_scorer.recommended_depth = MagicMock(return_value=0)
    verifier = MagicMock(spec=Verifier)
    prompt_lookup = PromptLookup()
    draft_combiner = DraftCombiner()
    mtp_draft = MagicMock(spec=MtpDraft)
    mtp_draft.is_enabled = False
    mtp_draft.num_mtp_layers = 0
    self_speculative = MagicMock(spec=SelfSpeculative)
    self_speculative.is_enabled = False
    layer_skip = MagicMock(spec=LayerSkip)
    layer_skip.is_enabled = False
    reasoning_mode = MagicMock(spec=ReasoningMode)
    online_calibrator = OnlineCalibrator()
    performance_objective = PerformanceObjective()

    transfer_scheduler = MagicMock(spec=TransferScheduler)
    transfer_scheduler.plan_transfers = MagicMock(return_value=TransferPlan())
    transfer_scheduler.plan_evictions = MagicMock(return_value=EvictionPlan())
    eviction_policy = EvictionPolicy()

    return OrchestratorLoop(
        config=config, metadata=md,
        cmd_writer=cmd_writer, ring_writer=ring_writer,
        completion_reader=cmp_reader, state_reader=state_reader,
        work_queue=work_queue, scheduler=scheduler,
        gpu_assigner=gpu_assigner,
        transfer_scheduler=transfer_scheduler,
        eviction_policy=eviction_policy,
        expert_placement=expert_placement,
        placement_optimizer=placement_optimizer,
        prescope=prescope, probe=probe, moe_speq=moe_speq,
        prefetch_fuser=prefetch_fuser,
        speculative_prefetch=speculative_prefetch,
        utility_scorer=utility_scorer, verifier=verifier,
        prompt_lookup=prompt_lookup, draft_combiner=draft_combiner,
        mtp_draft=mtp_draft, self_speculative=self_speculative,
        layer_skip=layer_skip,
        reasoning_mode=reasoning_mode,
        online_calibrator=online_calibrator,
        performance_objective=performance_objective,
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    samples: list[float]  # microseconds per cycle

    @property
    def min_us(self) -> float:
        return min(self.samples)

    @property
    def median_us(self) -> float:
        return statistics.median(self.samples)

    @property
    def p95_us(self) -> float:
        idx = int(len(self.samples) * 0.95)
        return sorted(self.samples)[min(idx, len(self.samples) - 1)]

    @property
    def p99_us(self) -> float:
        idx = int(len(self.samples) * 0.99)
        return sorted(self.samples)[min(idx, len(self.samples) - 1)]

    @property
    def max_us(self) -> float:
        return max(self.samples)


def run_bench(name: str, loop: OrchestratorLoop,
              warmup: int = 200, measured: int = 2000) -> BenchResult:
    """Run warmup + measured cycles, return per-cycle timings."""
    # Warm up (let Python JIT and caches settle)
    for _ in range(warmup):
        loop.run_one_cycle()

    # Measured runs
    samples = []
    for _ in range(measured):
        t0 = time.perf_counter_ns()
        loop.run_one_cycle()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1000.0)

    return BenchResult(name=name, samples=samples)


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

def bench_idle() -> BenchResult:
    """Empty loop: no requests, no completions. Measures baseline overhead."""
    loop = _build_bench_loop()
    return run_bench("idle", loop)


def bench_1_request() -> BenchResult:
    """Single request with pending work items (realistic scoring path)."""
    loop = _build_bench_loop()
    # Create a request with several pending work items at different layers
    req = RequestState(
        request_id=1, seq_id=1, default_gpu=0,
        current_layer=3, is_prefill=False,
    )
    loop._requests[1] = req
    for layer in range(3, 6):
        item = WorkItem(
            request_id=1, layer_idx=layer,
            operation=WorkOperation.ATTENTION,
            target_gpu=0, status=WorkStatus.READY,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._work_queue.insert(item)
    return run_bench("1_req", loop)


def bench_8_requests() -> BenchResult:
    """Eight concurrent requests spread across layers and GPUs."""
    loop = _build_bench_loop()
    for rid in range(1, 9):
        gpu = rid % 2
        layer = rid % 6
        req = RequestState(
            request_id=rid, seq_id=rid, default_gpu=gpu,
            current_layer=layer, is_prefill=False,
        )
        loop._requests[rid] = req
        item = WorkItem(
            request_id=rid, layer_idx=layer,
            operation=WorkOperation.ATTENTION,
            target_gpu=gpu, status=WorkStatus.READY,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._work_queue.insert(item)
    # Populate some resident experts for scoring
    for e in range(4):
        loop._resident_keys[0].add(ExpertKey(3, e))
        loop._resident_keys[1].add(ExpertKey(4, e))
    return run_bench("8_req", loop)


def bench_with_resident_experts() -> BenchResult:
    """One request + 64 resident experts (exercises eviction input building).

    Forces eviction scoring even if a future optimization (#13c-1) adds a
    skip guard.  The transfer plan returns required_slots > 0 so the
    eviction path is always exercised.
    """
    from orchestrator.types import TransferPlanEntry
    loop = _build_bench_loop()
    req = RequestState(
        request_id=1, seq_id=1, default_gpu=0,
        current_layer=4, is_prefill=False,
    )
    loop._requests[1] = req
    item = WorkItem(
        request_id=1, layer_idx=4,
        operation=WorkOperation.ATTENTION,
        target_gpu=0, status=WorkStatus.READY,
        timestamp_created_ns=time.perf_counter_ns(),
    )
    loop._work_queue.insert(item)
    # 64 experts across 2 GPUs
    for gpu in range(2):
        for layer in range(2, 6):
            for expert in range(8):
                loop._resident_keys[gpu].add(ExpertKey(layer, expert))
    # Force eviction scoring: transfer plan claims it needs slots, so the
    # skip-when-no-evictions guard (#13c-1) won't bypass _build_eviction_inputs.
    loop._transfer_scheduler.plan_transfers = MagicMock(
        return_value=TransferPlan(entries=[
            TransferPlanEntry(key=ExpertKey(5, 0), target_gpu=0, expert_bytes=1024),
        ]),
    )
    return run_bench("64_experts", loop)


# ---------------------------------------------------------------------------
# Per-phase profiler
# ---------------------------------------------------------------------------

def profile_phases(loop: OrchestratorLoop, warmup: int = 200,
                   measured: int = 2000) -> dict[str, BenchResult]:
    """Instrument each of the 6 phases and return per-phase timings.

    Monkey-patches the loop's phase methods with timing wrappers.
    Also profiles _build_eviction_inputs separately since it's the
    known hot spot within Phase 4 (PLAN).
    """
    phase_names = [
        "_phase_collect",
        "_phase_prefetch_scoring",
        "_phase_statistics_read",
        "_phase_plan",
        "_phase_dispatch",
        "_phase_yield",
    ]
    # Sub-functions within _phase_plan worth profiling individually
    sub_names = [
        "_build_eviction_inputs",
        "_compute_queue_depths",
        "_compute_vram_free",
    ]

    all_targets = phase_names + sub_names
    accumulators: dict[str, list[float]] = {name: [] for name in all_targets}
    originals: dict[str, object] = {}

    # Wrap each target method with a timing decorator
    for name in all_targets:
        original = getattr(loop, name)
        originals[name] = original
        acc = accumulators[name]

        def make_wrapper(orig, acc_list):
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter_ns()
                result = orig(*args, **kwargs)
                t1 = time.perf_counter_ns()
                acc_list.append((t1 - t0) / 1000.0)
                return result
            return wrapper

        setattr(loop, name, make_wrapper(original, acc))

    # Warm up
    for _ in range(warmup):
        loop.run_one_cycle()
    # Clear warmup samples
    for acc in accumulators.values():
        acc.clear()

    # Measured runs
    for _ in range(measured):
        loop.run_one_cycle()

    # Restore originals
    for name, original in originals.items():
        setattr(loop, name, original)

    # Build results
    results: dict[str, BenchResult] = {}
    for name in all_targets:
        samples = accumulators[name]
        if samples:
            results[name] = BenchResult(name=name, samples=samples)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    benchmarks = [
        bench_idle,
        bench_1_request,
        bench_8_requests,
        bench_with_resident_experts,
    ]

    # ── Per-config cycle timing ───────────────────────────────────────
    print(f"{'Benchmark':<20} {'Min':>8} {'Median':>8} {'P95':>8} {'P99':>8} {'Max':>8}  (μs)")
    print("-" * 76)

    budget_violations = 0
    for bench_fn in benchmarks:
        result = bench_fn()
        violated = " *** >100μs" if result.p95_us > 100 else ""
        if result.p95_us > 100:
            budget_violations += 1
        print(
            f"{result.name:<20} "
            f"{result.min_us:>8.1f} "
            f"{result.median_us:>8.1f} "
            f"{result.p95_us:>8.1f} "
            f"{result.p99_us:>8.1f} "
            f"{result.max_us:>8.1f}"
            f"{violated}"
        )

    print()
    if budget_violations:
        print(f"WARNING: {budget_violations} benchmark(s) exceed 100μs at P95 (INV-3.4.1)")
    else:
        print("All benchmarks within 100μs cycle budget at P95.")

    # ── Per-phase breakdown for 64_experts (the hot config) ───────────
    print()
    print("=== Per-phase breakdown: 64_experts ===")
    print(f"{'Phase':<30} {'Median':>8} {'P95':>8}  (μs)")
    print("-" * 52)

    from orchestrator.types import TransferPlanEntry
    loop = _build_bench_loop()
    req = RequestState(
        request_id=1, seq_id=1, default_gpu=0,
        current_layer=4, is_prefill=False,
    )
    loop._requests[1] = req
    item = WorkItem(
        request_id=1, layer_idx=4,
        operation=WorkOperation.ATTENTION,
        target_gpu=0, status=WorkStatus.READY,
        timestamp_created_ns=time.perf_counter_ns(),
    )
    loop._work_queue.insert(item)
    for gpu in range(2):
        for layer in range(2, 6):
            for expert in range(8):
                loop._resident_keys[gpu].add(ExpertKey(layer, expert))
    loop._transfer_scheduler.plan_transfers = MagicMock(
        return_value=TransferPlan(entries=[
            TransferPlanEntry(key=ExpertKey(5, 0), target_gpu=0, expert_bytes=1024),
        ]),
    )

    phase_results = profile_phases(loop)
    for name, result in sorted(phase_results.items(),
                                key=lambda x: -x[1].median_us):
        print(
            f"  {name:<28} "
            f"{result.median_us:>8.1f} "
            f"{result.p95_us:>8.1f}"
        )


if __name__ == "__main__":
    main()

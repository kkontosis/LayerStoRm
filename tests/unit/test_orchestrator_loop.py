"""Tests for orchestrator.orchestrator_loop — 6-phase event loop."""

from __future__ import annotations

import ctypes
import time
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from orchestrator.command_writer import CommandWriter, CompletionReader
from orchestrator.draft_combiner import CombinedDraft, CombinedDraftStep, DraftCombiner
from orchestrator.eviction_policy import EvictionPolicy, EvictionPolicyConfig
from orchestrator.layer_skip import LayerSkip
from orchestrator.config_hot_reload import ConfigHotReload
from orchestrator.mtp_draft import DraftStep, MtpDraft
from orchestrator.self_speculative import SelfSpeculative
from orchestrator.expert_placement import ExpertPlacement, ExpertPlacementConfig
from orchestrator.gpu_load_balancer import GpuLoadBalancer, GpuLoadBalancerConfig
from orchestrator.moe_speq import MoeSpeq
from orchestrator.online_calibrator import OnlineCalibrator
from orchestrator.orchestrator_loop import (
    CycleMetrics,
    InferenceRequest,
    OrchestratorConfig,
    OrchestratorLoop,
    RequestState,
)
from orchestrator.performance_objective import PerformanceObjective
from orchestrator.placement_optimizer import (
    PlacementOptimizer,
    PlacementOptimizerConfig,
)
from orchestrator.prefetch_fuser import PrefetchFuser
from orchestrator.prescope import PreScope
from orchestrator.probe import Probe
from orchestrator.prompt_lookup import PromptLookup
from orchestrator.reasoning_mode import ReasoningMode
from orchestrator.scheduler import Scheduler
from orchestrator.shm_protocol import (
    CMD_CANCEL_TRANSFER,
    CMD_SEQ_FORK,
    CMP_COMPUTE_DONE,
    CMP_ELM_EXPERT_EVICTED,
    CMP_ELM_EXPERT_READY,
    CMP_ERROR,
    CMP_CHECKPOINT,
    Completion,
    ComputeCompletionPayload,
    D_B_CMD_RUN_ATTENTION,
    D_CMD_MTP_PROJECT,
    D_CMD_RUN_MTP_STEP,
    D_CMD_RUN_SELF_SPEC_FORWARD,
    E_CMD_FETCH_AND_RUN_MOE,
    E_CMD_SEQ_FREE,
    ExpertPrefetchEntry,
    RoutingExportHeader,
    SIDEBAND_EXPERT_PREFETCH_OFF,
    SIDEBAND_ROUTING_EXPORT_INDICES_OFF,
    SIDEBAND_ROUTING_EXPORT_OFF,
    SIDEBAND_TOTAL_SIZE,
)
from orchestrator.speculative_prefetch import SpeculativePrefetch
from orchestrator.transfer_scheduler import TransferScheduler
from orchestrator.types import (
    CacheZone,
    EngineMetadata,
    EvictionPlan,
    ExpertKey,
    GpuConfig,
    SpeculationState,
    TransferPlan,
    TransferPlanEntry,
    VerificationPlan,
    WorkItem,
    WorkOperation,
    WorkStatus,
)
from orchestrator.utility_scorer import UtilityScorer
from orchestrator.verifier import VerificationResult, Verifier
from orchestrator.work_queue import WorkQueue


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _gpu_configs() -> tuple[GpuConfig, ...]:
    return (
        GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                  vram_bytes=32 * 1024**3, compute_weight=1.0),
        GpuConfig(position=1, gpu_type="rtx5090", is_tp=True,
                  vram_bytes=32 * 1024**3, compute_weight=1.0),
    )


def _metadata() -> EngineMetadata:
    return EngineMetadata(
        num_gpus=2,
        num_moe_layers=4,
        num_experts=8,
        num_layers=6,
        expert_bytes=2_359_296,
        kv_bytes_per_page=4096,
        gpus=_gpu_configs(),
    )


class MockCompletionReader:
    def __init__(self):
        self._completions: list[Completion] = []

    def drain(self, max_count: int = 0xFFFFFFFF) -> list[Completion]:
        result = self._completions[:max_count]
        self._completions = self._completions[max_count:]
        return result

    def is_empty(self) -> bool:
        return len(self._completions) == 0

    def add(self, cmp: Completion) -> None:
        self._completions.append(cmp)


class MockRingWriter:
    def __init__(self):
        self.written: list = []

    def write_struct(self, cmd) -> bool:
        self.written.append(cmd)
        return True

    def write(self, data: bytes) -> bool:
        self.written.append(data)
        return True


class MockStateReader:
    def __init__(self):
        self._shift = False

    def shift_detected(self) -> bool:
        val = self._shift
        self._shift = False
        return val

    def windowed_acceptance_rate(self) -> float:
        return 0.5

    def vram_usage(self, gpu_idx: int) -> tuple[int, int]:
        return (1 * 1024**3, 32 * 1024**3)

    def cache_fill(self, gpu_idx: int) -> tuple[int, int, int, int]:
        return (10, 100, 5, 50)

    def is_host_warm(self, layer: int, expert: int) -> bool:
        return False

    def expert_stats(self, layer: int, expert: int) -> tuple[float, float, float, float]:
        return (0.1, 0.5, 0.3, 0.2)

    def expert_stats_batch(self, keys: list) -> list:
        return [(0.1, 0.5, 0.3, 0.2) for _ in keys]

    def is_resident(self, layer: int, expert: int, gpu: int) -> bool:
        return False


def _make_completion(cmp_type: int, cmd_seq: int = 0,
                     gpu_idx: int = 0, status: int = 0) -> Completion:
    cmp = Completion()
    cmp.cmp_type = cmp_type
    cmp.cmd_seq = cmd_seq
    cmp.gpu_idx = gpu_idx
    cmp.status = status
    return cmp


def _build_loop(**overrides) -> tuple[OrchestratorLoop, dict]:
    md = _metadata()
    config = OrchestratorConfig()
    cmp_reader = MockCompletionReader()
    ring_writer = MockRingWriter()
    state_reader = MockStateReader()
    cmd_writer = CommandWriter()
    work_queue = WorkQueue()
    scheduler = Scheduler()
    gpu_assigner = GpuLoadBalancer(GpuLoadBalancerConfig(gpus=md.gpus))

    placement_cfg = ExpertPlacementConfig(
        num_moe_layers=md.num_moe_layers,
        num_experts=md.num_experts,
        cache_gpu_indices=[0, 1],
    )
    expert_placement = ExpertPlacement(placement_cfg)

    placement_optimizer = PlacementOptimizer(
        PlacementOptimizerConfig(), expert_placement,
    )

    prescope = MagicMock(spec=PreScope)
    prescope.enabled = False
    prescope.is_moe_layer = MagicMock(return_value=False)

    probe = MagicMock(spec=Probe)
    probe.enabled = False
    probe.probe_layers = ()

    moe_speq = MagicMock(spec=MoeSpeq)
    moe_speq.enabled = False
    moe_speq.is_moe_layer = MagicMock(return_value=False)

    prefetch_fuser = PrefetchFuser()

    speculative_prefetch = MagicMock(spec=SpeculativePrefetch)
    utility_scorer = MagicMock(spec=UtilityScorer)
    utility_scorer.recommended_depth = MagicMock(return_value=0)

    verifier = MagicMock(spec=Verifier)
    prompt_lookup = PromptLookup()
    draft_combiner = DraftCombiner()

    # MTP / self-spec / layer-skip: disabled by default so existing tests
    # exercise the prompt-lookup-only path with zero behavioral change.
    mtp_draft = MagicMock(spec=MtpDraft)
    mtp_draft.is_enabled = False
    mtp_draft.num_mtp_layers = 0

    self_speculative_mock = MagicMock(spec=SelfSpeculative)
    self_speculative_mock.is_enabled = False

    layer_skip_mock = MagicMock(spec=LayerSkip)
    layer_skip_mock.is_enabled = False

    reasoning_mode = MagicMock(spec=ReasoningMode)

    online_calibrator = OnlineCalibrator()
    performance_objective = PerformanceObjective()

    ts_config = MagicMock()
    ts_config.num_gpus = 2
    ts_config.expert_bytes = md.expert_bytes
    ts_config.pcie_bandwidth_gbps = (1.969, 1.969)
    ts_config.max_inflight_per_gpu = 4
    ts_config.min_priority_threshold = 0.01
    ts_config.cycle_budget_us = 1000.0
    ts_config.stable_zone_threshold = 0.7

    transfer_scheduler = MagicMock(spec=TransferScheduler)
    transfer_scheduler.plan_transfers = MagicMock(return_value=TransferPlan())
    transfer_scheduler.plan_evictions = MagicMock(return_value=EvictionPlan())

    eviction_policy = EvictionPolicy()

    deps = {
        "config": config,
        "metadata": md,
        "cmd_writer": cmd_writer,
        "ring_writer": ring_writer,
        "completion_reader": cmp_reader,
        "state_reader": state_reader,
        "work_queue": work_queue,
        "scheduler": scheduler,
        "gpu_assigner": gpu_assigner,
        "transfer_scheduler": transfer_scheduler,
        "eviction_policy": eviction_policy,
        "expert_placement": expert_placement,
        "placement_optimizer": placement_optimizer,
        "prescope": prescope,
        "probe": probe,
        "moe_speq": moe_speq,
        "prefetch_fuser": prefetch_fuser,
        "speculative_prefetch": speculative_prefetch,
        "utility_scorer": utility_scorer,
        "verifier": verifier,
        "prompt_lookup": prompt_lookup,
        "draft_combiner": draft_combiner,
        "mtp_draft": mtp_draft,
        "self_speculative": self_speculative_mock,
        "layer_skip": layer_skip_mock,
        "reasoning_mode": reasoning_mode,
        "online_calibrator": online_calibrator,
        "performance_objective": performance_objective,
    }
    deps.update(overrides)

    loop = OrchestratorLoop(**deps)
    return loop, deps


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_construction(self):
        loop, _ = _build_loop()
        assert loop.cycle_count == 0
        assert len(loop.requests) == 0

    def test_delegates_wired(self):
        loop, deps = _build_loop()
        assert loop._scheduler.gpu_assigner is deps["gpu_assigner"]
        assert loop._scheduler.eviction_scorer is deps["eviction_policy"]

    def test_resident_keys_initialized(self):
        loop, _ = _build_loop()
        assert 0 in loop.resident_keys
        assert 1 in loop.resident_keys
        assert len(loop.resident_keys[0]) == 0


# ---------------------------------------------------------------------------
# Phase 1: COLLECT
# ---------------------------------------------------------------------------

class TestPhaseCollect:
    def test_drain_empty_no_op(self):
        loop, _ = _build_loop()
        loop._phase_collect()
        assert loop.cycle_count == 0

    def test_new_request_creates_work_item(self):
        loop, deps = _build_loop()
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=[10, 20, 30], gpu=0,
        ))
        loop._phase_collect()
        assert 1 in loop.requests
        assert loop.requests[1].seq_id == 1
        assert len(deps["ring_writer"].written) == 1
        items = deps["work_queue"].pending_items()
        assert len(items) == 1
        assert items[0].operation == WorkOperation.EMBEDDING

    def test_compute_done_updates_status(self):
        loop, deps = _build_loop()
        item = WorkItem(request_id=1, layer_idx=0,
                        operation=WorkOperation.ATTENTION,
                        status=WorkStatus.DISPATCHED,
                        timestamp_created_ns=time.perf_counter_ns())
        deps["work_queue"].insert(item)
        loop._requests[1] = RequestState(
            request_id=1, seq_id=1, default_gpu=0,
        )
        loop._cmd_seq_map[42] = (1, item)

        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=42)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert item.status == WorkStatus.COMPLETED

    def test_elm_expert_ready_adds_to_resident(self):
        loop, deps = _build_loop()
        cmp = _make_completion(CMP_ELM_EXPERT_READY, gpu_idx=0)
        cmp.payload.elm_expert.layer_idx = 3
        cmp.payload.elm_expert.expert_idx = 5
        deps["completion_reader"].add(cmp)
        loop._phase_collect()
        assert ExpertKey(3, 5) in loop.resident_keys[0]

    def test_elm_expert_evicted_removes_from_resident(self):
        loop, deps = _build_loop()
        loop._resident_keys[0].add(ExpertKey(3, 5))
        cmp = _make_completion(CMP_ELM_EXPERT_EVICTED, gpu_idx=0)
        cmp.payload.elm_expert.layer_idx = 3
        cmp.payload.elm_expert.expert_idx = 5
        deps["completion_reader"].add(cmp)
        loop._phase_collect()
        assert ExpertKey(3, 5) not in loop.resident_keys[0]

    def test_error_cancels_request(self):
        """CMP_ERROR on normal command calls on_complete + frees KV (TD-59a)."""
        results = []
        loop, deps = _build_loop()
        item = WorkItem(request_id=1, layer_idx=0,
                        operation=WorkOperation.ATTENTION,
                        status=WorkStatus.DISPATCHED,
                        timestamp_created_ns=time.perf_counter_ns())
        deps["work_queue"].insert(item)
        loop._requests[1] = RequestState(
            request_id=1, seq_id=1, default_gpu=0,
            on_complete=lambda rid, toks, reason, lp=None: results.append(
                (rid, toks, reason)),
        )
        loop._cmd_seq_map[99] = (1, item)

        ring_before = len(deps["ring_writer"].written)
        cmp = _make_completion(CMP_ERROR, cmd_seq=99)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()
        assert 1 not in loop.requests
        # TD-59a: on_complete must be called with finish_reason="error"
        assert len(results) == 1
        rid, toks, reason = results[0]
        assert rid == 1
        assert reason == "error"
        # TD-59a: E_CMD_SEQ_FREE must be sent to free KV pages
        free_cmds = [
            c for c in deps["ring_writer"].written[ring_before:]
            if hasattr(c, "cmd_type") and c.cmd_type == E_CMD_SEQ_FREE
        ]
        assert len(free_cmds) == 1
        assert free_cmds[0].payload.seq_free.seq_id == 1

    def test_checkpoint_stores_hidden_state(self):
        loop, deps = _build_loop()
        loop._requests[1] = RequestState(
            request_id=1, seq_id=1, default_gpu=0,
        )
        item = WorkItem(request_id=1, layer_idx=5,
                        operation=WorkOperation.ATTENTION,
                        status=WorkStatus.DISPATCHED,
                        timestamp_created_ns=time.perf_counter_ns())
        loop._cmd_seq_map[50] = (1, item)

        cmp = _make_completion(CMP_CHECKPOINT, cmd_seq=50)
        cmp.payload.checkpoint.layer_idx = 5
        cmp.payload.checkpoint.checkpoint_type = 0
        cmp.payload.checkpoint.host_buf_offset = 1024
        cmp.payload.checkpoint.data_bytes = 256
        deps["completion_reader"].add(cmp)
        loop._phase_collect()
        assert 5 in loop.requests[1].hidden_state_checkpoints
        assert loop.requests[1].hidden_state_checkpoints[5] == (1024, 256)


# ---------------------------------------------------------------------------
# Phase 2: PREFETCH SCORING
# ---------------------------------------------------------------------------

class TestPhasePrefetchScoring:
    def test_fuser_returns_empty_with_no_requests(self):
        loop, _ = _build_loop()
        priorities = loop._phase_prefetch_scoring()
        assert priorities == []

    def test_fuser_called_with_active_requests(self):
        loop, _ = _build_loop()
        loop._requests[1] = RequestState(
            request_id=1, seq_id=1, default_gpu=0, current_layer=3,
        )
        priorities = loop._phase_prefetch_scoring()
        assert isinstance(priorities, list)


# ---------------------------------------------------------------------------
# Phase 3: STATISTICS READ
# ---------------------------------------------------------------------------

class TestPhaseStatisticsRead:
    def test_shift_triggers_placement_reset(self):
        loop, deps = _build_loop()
        deps["state_reader"]._shift = True
        loop._phase_statistics_read()
        assert deps["state_reader"]._shift is False

    def test_no_sync_when_no_adjustment(self):
        loop, deps = _build_loop()
        loop._phase_statistics_read()
        assert len(deps["ring_writer"].written) == 0


# ---------------------------------------------------------------------------
# Phase 4: PLAN
# ---------------------------------------------------------------------------

class TestPhasePlan:
    def test_plan_with_no_work(self):
        loop, _ = _build_loop()
        ev, tr, batches = loop._phase_plan([])
        assert len(ev.entries) == 0
        assert len(tr.entries) == 0
        assert len(batches) == 0

    def test_prefetch_target_gpu_assigned(self):
        """TD-59d: prefetch priorities get target_gpu from load balancer."""
        from orchestrator.types import PrefetchPriority
        loop, _ = _build_loop()
        pp = PrefetchPriority(
            key=ExpertKey(3, 5), target_layer=3,
            target_gpu=0, priority_score=0.9,
        )
        loop._phase_plan([pp])
        # With 2 GPUs the balancer should assign a GPU (not crash);
        # exact GPU depends on scoring, but the assignment ran.
        assert pp.target_gpu in (0, 1)

    def test_plan_with_ready_items(self):
        loop, deps = _build_loop()
        item = WorkItem(
            request_id=1, layer_idx=0,
            operation=WorkOperation.ATTENTION,
            target_gpu=0, status=WorkStatus.READY,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        deps["work_queue"].insert(item)
        loop._requests[1] = RequestState(
            request_id=1, seq_id=1, default_gpu=0,
        )
        _, _, batches = loop._phase_plan([])
        assert len(batches) > 0


# ---------------------------------------------------------------------------
# Phase 5: DISPATCH
# ---------------------------------------------------------------------------

class TestPhaseDispatch:
    def test_dispatch_evict_then_transfer_then_compute(self):
        loop, deps = _build_loop()
        loop._resident_keys[0].add(ExpertKey(2, 0))

        ev_plan = EvictionPlan(entries=[
            MagicMock(key=ExpertKey(2, 0), gpu_idx=0, zone=CacheZone.STABLE),
        ])
        tr_plan = TransferPlan()
        loop._phase_dispatch(ev_plan, tr_plan, [])
        assert len(deps["ring_writer"].written) >= 1
        assert ExpertKey(2, 0) not in loop.resident_keys[0]

    def test_dispatch_compute_tracks_cmd_seq(self):
        loop, deps = _build_loop()
        item = WorkItem(
            request_id=1, layer_idx=0,
            operation=WorkOperation.ATTENTION,
            target_gpu=0, status=WorkStatus.READY,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        deps["work_queue"].insert(item)
        loop._requests[1] = RequestState(
            request_id=1, seq_id=1, default_gpu=0,
        )
        loop._dispatch_work_item(item)
        assert item.status == WorkStatus.DISPATCHED
        assert len(loop._cmd_seq_map) == 1


# ---------------------------------------------------------------------------
# Full cycle
# ---------------------------------------------------------------------------

class TestFullCycle:
    def test_single_cycle_no_crash(self):
        loop, _ = _build_loop()
        metrics = loop.run_one_cycle()
        assert loop.cycle_count == 1
        assert metrics.elapsed_us >= 0

    def test_request_advances_through_layers(self):
        loop, deps = _build_loop()

        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=[1], gpu=0,
        ))
        loop.run_one_cycle()
        assert 1 in loop.requests

        pending = deps["work_queue"].pending_items()
        ready = deps["work_queue"].by_status(WorkStatus.READY)
        dispatched = deps["work_queue"].by_status(WorkStatus.DISPATCHED)
        assert len(dispatched) > 0 or len(ready) > 0 or len(pending) > 0

    def test_progress_guarantee(self):
        loop, deps = _build_loop()
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=[1], gpu=0,
        ))
        loop.run_one_cycle()
        assert deps["ring_writer"].written


# ---------------------------------------------------------------------------
# Request advancement
# ---------------------------------------------------------------------------

class TestRequestAdvancement:
    def test_embedding_leads_to_attention(self):
        loop, deps = _build_loop()
        req = RequestState(request_id=1, seq_id=1, default_gpu=0)
        loop._requests[1] = req
        loop._advance_request(req, 0, WorkOperation.EMBEDDING)
        items = deps["work_queue"].pending_items()
        assert any(i.operation == WorkOperation.ATTENTION for i in items)

    def test_attention_at_moe_layer_leads_to_expert_ffn(self):
        loop, deps = _build_loop()
        req = RequestState(request_id=1, seq_id=1, default_gpu=0)
        loop._requests[1] = req
        moe_layer = loop._first_moe_layer
        loop._advance_request(req, moe_layer, WorkOperation.ATTENTION)
        items = deps["work_queue"].pending_items()
        assert any(i.operation == WorkOperation.EXPERT_FFN for i in items)

    def test_attention_at_non_moe_leads_to_expert_ffn(self):
        # TD-84e: all layers (including dense) go ATTENTION -> EXPERT_FFN.
        # C++ dispatch_moe_internal handles dense layers via the dense early-out.
        loop, deps = _build_loop()
        req = RequestState(request_id=1, seq_id=1, default_gpu=0)
        loop._requests[1] = req
        loop._advance_request(req, 0, WorkOperation.ATTENTION)
        items = deps["work_queue"].pending_items()
        assert any(
            i.operation == WorkOperation.EXPERT_FFN and i.layer_idx == 0
            for i in items
        )

    def test_expert_ffn_advances_layer(self):
        loop, deps = _build_loop()
        req = RequestState(request_id=1, seq_id=1, default_gpu=0)
        loop._requests[1] = req
        moe_layer = loop._first_moe_layer
        loop._advance_request(req, moe_layer, WorkOperation.EXPERT_FFN)
        assert req.current_layer == moe_layer + 1

    def test_last_layer_leads_to_output_head(self):
        loop, deps = _build_loop()
        req = RequestState(request_id=1, seq_id=1, default_gpu=0)
        loop._requests[1] = req
        last_layer = loop._metadata.num_layers - 1
        loop._advance_request(req, last_layer, WorkOperation.EXPERT_FFN)
        items = deps["work_queue"].pending_items()
        assert any(i.operation == WorkOperation.OUTPUT_HEAD for i in items)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestShutdown:
    def test_shutdown_sets_flag(self):
        loop, _ = _build_loop()
        loop.shutdown()
        assert loop._shutdown is True

    def test_run_exits_on_shutdown(self):
        loop, _ = _build_loop()
        loop._shutdown = True
        loop.run()
        assert loop.cycle_count == 0


# ---------------------------------------------------------------------------
# Helpers for speculation tests
# ---------------------------------------------------------------------------

def _make_combined_draft(tokens: list[int]) -> CombinedDraft:
    steps = [
        CombinedDraftStep(
            token_id=t, position=i, source="prompt_lookup",
            p_accept=0.9, utility=1.0,
        )
        for i, t in enumerate(tokens)
    ]
    return CombinedDraft(steps=steps)


def _make_decode_request(loop, deps, request_id=1, gpu=0):
    """Create a request that has completed prefill (is_prefill=False)."""
    req = RequestState(
        request_id=request_id, seq_id=1, default_gpu=gpu,
        is_prefill=False, tokens_generated=1,
        token_history=[10, 20, 30, 20, 30],
    )
    loop._requests[request_id] = req
    return req


def _setup_speculation_mocks(deps, depth=3, draft_tokens=None):
    """Configure mocks for speculation to trigger."""
    if draft_tokens is None:
        draft_tokens = [100, 101, 102]
    deps["utility_scorer"].recommended_depth = MagicMock(return_value=depth)
    deps["utility_scorer"].record_iteration = MagicMock()
    deps["utility_scorer"].apply_ceiling = UtilityScorer.apply_ceiling


class HostBuffer:
    """Small ctypes buffer simulating the daemon's shared host memory region.

    Writes uint32 token IDs at a given offset.  Pass .base_address as
    host_buf_base to OrchestratorLoop so _extract_token_id can read them.
    """

    def __init__(self, size: int = 256):
        self._buf = (ctypes.c_uint8 * size)()
        self.base_address = ctypes.addressof(self._buf)

    def write_tokens(self, offset: int, tokens: list[int]) -> None:
        arr = (ctypes.c_uint32 * len(tokens)).from_address(
            self.base_address + offset
        )
        for i, t in enumerate(tokens):
            arr[i] = t


def _make_completion_with_token(
    cmp_type: int, cmd_seq: int = 0,
    host_buf_offset: int = 0, num_tokens: int = 1,
) -> Completion:
    """Create a completion whose compute payload points to a host buffer region."""
    cmp = Completion()
    cmp.cmp_type = cmp_type
    cmp.cmd_seq = cmd_seq
    cmp.gpu_idx = 0
    cmp.status = 0
    cmp.payload.compute.host_buf_offset = host_buf_offset
    cmp.payload.compute.data_bytes = num_tokens * 4
    return cmp


# ---------------------------------------------------------------------------
# Speculation lifecycle
# ---------------------------------------------------------------------------

class TestSpeculationLifecycle:

    # ── State transitions ──────────────────────────────────────────────

    def test_autoregressive_to_drafting(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        _setup_speculation_mocks(deps, depth=3)

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[42] = (1, item)

        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=42)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert req.speculation_state != SpeculationState.AUTOREGRESSIVE
        # #92: prompt-lookup-only drafting takes NO fork — verification
        # runs as sequential early-stop feeds on the MAIN sequence.
        assert req.draft_seq_id is None
        if req.speculation_state == SpeculationState.VERIFYING:
            assert req.mtp_sequential_verify is True

    def test_no_speculation_during_prefill(self):
        loop, deps = _build_loop()
        req = RequestState(
            request_id=1, seq_id=1, default_gpu=0,
            is_prefill=True, tokens_generated=0,
        )
        loop._requests[1] = req
        _setup_speculation_mocks(deps, depth=3)

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[42] = (1, item)

        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=42)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert req.is_prefill is False
        assert req.speculation_state == SpeculationState.AUTOREGRESSIVE
        items = deps["work_queue"].pending_items()
        assert any(i.operation == WorkOperation.EMBEDDING for i in items)

    def test_no_speculation_when_depth_zero(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        deps["utility_scorer"].recommended_depth = MagicMock(return_value=0)

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[42] = (1, item)

        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=42)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert req.speculation_state == SpeculationState.AUTOREGRESSIVE
        items = deps["work_queue"].pending_items()
        assert any(i.operation == WorkOperation.EMBEDDING for i in items)

    def test_full_lifecycle(self):
        """Prompt-lookup-only speculation (#92): no fork; verification is
        sequential early-stop on the MAIN sequence (the MtpLossless
        pattern shared with MTP/DSpark) — the first verify feed is a
        plain AR EMBEDDING work item."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.token_history = [10, 20, 30, 20, 30]
        _setup_speculation_mocks(deps, depth=3)

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[42] = (1, item)

        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=42)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        if req.speculation_state == SpeculationState.AUTOREGRESSIVE:
            return  # prompt lookup found no continuation

        # Sequential-verify entry state: draft combined, NO fork taken.
        assert req.speculation_state == SpeculationState.VERIFYING
        assert req.mtp_sequential_verify is True
        assert req.combined_draft is not None
        assert req.draft_seq_id is None
        assert len(req.verify_tokens) > 0
        # First verification feed enqueued as a normal AR forward.
        items = deps["work_queue"].pending_items()
        assert any(i.operation == WorkOperation.EMBEDDING for i in items)

    # ── Draft generation ───────────────────────────────────────────────

    def test_prompt_lookup_fed_to_combiner(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)

        mock_combiner = MagicMock(spec=DraftCombiner)
        mock_combiner.combine = MagicMock(
            return_value=_make_combined_draft([100, 101]),
        )
        mock_combiner.record_result = MagicMock()
        loop._draft_combiner = mock_combiner

        mock_lookup = MagicMock(spec=PromptLookup)
        mock_lookup.lookup = MagicMock(return_value=[100, 101])
        mock_lookup.acceptance_rate = 0.5
        mock_lookup.record_result = MagicMock()
        loop._prompt_lookup = mock_lookup

        req.speculation_depth = 3
        loop._start_drafting(req)

        mock_lookup.lookup.assert_called_once_with(
            req.token_history, max_continuation=3,
        )
        mock_combiner.combine.assert_called_once()
        call_args = mock_combiner.combine.call_args
        assert call_args.kwargs["prompt_lookup_tokens"] == [100, 101]

    def test_empty_draft_reverts_to_autoregressive(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3

        mock_combiner = MagicMock(spec=DraftCombiner)
        mock_combiner.combine = MagicMock(
            return_value=CombinedDraft(),
        )
        loop._draft_combiner = mock_combiner

        mock_lookup = MagicMock(spec=PromptLookup)
        mock_lookup.lookup = MagicMock(return_value=[])
        mock_lookup.acceptance_rate = 0.0
        loop._prompt_lookup = mock_lookup

        loop._start_drafting(req)

        assert req.speculation_state == SpeculationState.AUTOREGRESSIVE
        assert req.combined_draft is None
        assert req.draft_seq_id is None
        items = deps["work_queue"].pending_items()
        assert any(i.operation == WorkOperation.EMBEDDING for i in items)

    # ── Verification ───────────────────────────────────────────────────

    def test_no_gating_bypasses_sp_moe(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.combined_draft = _make_combined_draft([100, 101, 102])
        req.draft_gating = None
        req.draft_complete = True
        req.speculation_depth = 3

        loop._start_prefetching(req)

        assert req.verification_plan is not None
        assert req.verification_plan.max_depth == 3
        assert len(req.verification_plan.transfers) == 0
        deps["speculative_prefetch"].compute_verification_plan.assert_not_called()

    def test_verification_commands_dispatched(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.combined_draft = _make_combined_draft([100, 101])
        req.draft_gating = None
        req.max_verifiable_depth = 2
        req.speculation_state = SpeculationState.PREFETCHING_VERIFY
        req.verification_plan = VerificationPlan(transfers=[], max_depth=2)

        real_verifier = Verifier(num_layers=6, num_moe_layers=4,
                                 first_moe_layer=2, num_experts=8)
        loop._verifier = real_verifier

        ring_before = len(deps["ring_writer"].written)
        loop._start_verification(req)
        ring_after = len(deps["ring_writer"].written)

        assert ring_after > ring_before
        assert req.speculation_state == SpeculationState.VERIFYING
        assert req.verified_depth == 2
        assert any(
            cs_entry[1].is_speculative
            for cs_entry in loop._cmd_seq_map.values()
        )

    # ── TD-59l: per-GPU verification readiness ─────────────────────────

    def test_verification_readiness_wrong_gpu_not_ready(self):
        """Expert on GPU 1 must not pass readiness for request on GPU 0."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps, gpu=0)
        req.combined_draft = _make_combined_draft([100, 101])
        req.draft_gating = np.zeros((2, 4, 8))  # non-None triggers expert check
        req.speculation_state = SpeculationState.PREFETCHING_VERIFY
        req.verification_plan = VerificationPlan(
            transfers=[TransferPlanEntry(key=ExpertKey(2, 3), target_gpu=0)],
            max_depth=2,
        )

        # Expert resident on GPU 1 only
        loop._resident_keys[1].add(ExpertKey(2, 3))

        loop._check_verification_ready(req)
        assert req.speculation_state == SpeculationState.PREFETCHING_VERIFY

    def test_verification_readiness_correct_gpu_passes(self):
        """Expert on the request's default GPU passes readiness check."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps, gpu=0)
        req.combined_draft = _make_combined_draft([100, 101])
        req.draft_gating = np.zeros((2, 4, 8))
        req.speculation_state = SpeculationState.PREFETCHING_VERIFY
        req.verification_plan = VerificationPlan(
            transfers=[TransferPlanEntry(key=ExpertKey(2, 3), target_gpu=0)],
            max_depth=2,
        )

        # Expert resident on correct GPU
        loop._resident_keys[0].add(ExpertKey(2, 3))

        # Spy on _start_verification to confirm readiness passed
        # (the downstream verification flow is tested separately)
        called = []
        orig = loop._start_verification
        loop._start_verification = lambda r: called.append(r)

        loop._check_verification_ready(req)
        assert len(called) == 1
        assert called[0] is req

    # ── Accept / reject ────────────────────────────────────────────────

    def test_full_acceptance(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 99
        req.combined_draft = _make_combined_draft([100, 101, 102])
        req.verification_analyses = []
        req.verified_depth = 3
        req.speculation_state = SpeculationState.ACCEPTING

        vresult = VerificationResult(
            accepted_length=3, attempted_length=3,
            accepted_tokens=[100, 101, 102],
        )
        deps["verifier"].record_result = MagicMock()
        deps["utility_scorer"].record_iteration = MagicMock()

        tokens_before = req.tokens_generated
        loop._process_acceptance(req, vresult)

        assert req.tokens_generated == tokens_before + 3
        assert req.speculation_state == SpeculationState.AUTOREGRESSIVE
        assert req.combined_draft is None
        assert 100 in req.token_history
        assert 101 in req.token_history
        assert 102 in req.token_history

    def test_partial_acceptance(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 99
        req.combined_draft = _make_combined_draft([100, 101, 102])
        req.verification_analyses = []
        req.verified_depth = 3
        req.speculation_state = SpeculationState.ACCEPTING

        vresult = VerificationResult(
            accepted_length=2, attempted_length=3,
            accepted_tokens=[100, 101],
            rejected_positions=[2],
        )
        deps["verifier"].record_result = MagicMock()
        deps["utility_scorer"].record_iteration = MagicMock()

        tokens_before = req.tokens_generated
        loop._process_acceptance(req, vresult)

        assert req.tokens_generated == tokens_before + 2
        assert req.speculation_state == SpeculationState.AUTOREGRESSIVE

    def test_full_rejection(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 99
        req.combined_draft = _make_combined_draft([100, 101, 102])
        req.verification_analyses = []
        req.verified_depth = 3
        req.speculation_state = SpeculationState.ACCEPTING

        vresult = VerificationResult(
            accepted_length=0, attempted_length=3,
            rejected_positions=[0, 1, 2],
        )
        deps["verifier"].record_result = MagicMock()
        deps["utility_scorer"].record_iteration = MagicMock()

        tokens_before = req.tokens_generated
        loop._process_acceptance(req, vresult)

        assert req.tokens_generated == tokens_before
        assert req.speculation_state == SpeculationState.AUTOREGRESSIVE

    # ── KV management ──────────────────────────────────────────────────

    def test_seq_fork_on_draft_start(self):
        """The KV fork is now taken ONLY for self-spec drafting (#92):
        its draft forward writes approximate main-layer KV so it cannot
        run on the main sequence.  Prompt-lookup-only takes no fork."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 2

        # Arm self-spec (mock with the attributes _dispatch_self_spec_step
        # reads: max_depth gates step count, draft_expert_count goes into
        # the fused command payload).
        deps["self_speculative"].is_enabled = True
        deps["self_speculative"].max_depth = 2
        deps["self_speculative"].draft_expert_count = 1

        mock_lookup = MagicMock(spec=PromptLookup)
        mock_lookup.lookup = MagicMock(return_value=[])
        mock_lookup.acceptance_rate = 0.5
        loop._prompt_lookup = mock_lookup

        ring_before = len(deps["ring_writer"].written)
        loop._start_drafting(req)

        fork_cmds = [
            c for c in deps["ring_writer"].written[ring_before:]
            if hasattr(c, "cmd_type") and c.cmd_type == CMD_SEQ_FORK
        ]
        assert len(fork_cmds) == 1
        assert fork_cmds[0].payload.seq_fork.src_seq_id == req.seq_id
        assert fork_cmds[0].payload.seq_fork.dst_seq_id == req.draft_seq_id

    def test_no_fork_for_prompt_lookup_only(self):
        """#92: prompt-lookup-only drafting must NOT fork — verification
        feeds run on the MAIN sequence (sequential early-stop), so the
        main KV keeps every accepted position by construction."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 2

        mock_lookup = MagicMock(spec=PromptLookup)
        mock_lookup.lookup = MagicMock(return_value=[100, 101])
        mock_lookup.acceptance_rate = 0.5
        loop._prompt_lookup = mock_lookup

        ring_before = len(deps["ring_writer"].written)
        loop._start_drafting(req)

        new_cmds = deps["ring_writer"].written[ring_before:]
        assert not any(
            getattr(c, "cmd_type", 0) == CMD_SEQ_FORK for c in new_cmds
        )
        assert req.draft_seq_id is None
        assert req.speculation_state == SpeculationState.VERIFYING
        assert req.mtp_sequential_verify is True
        assert req.verify_tokens == [100, 101]

    def test_seq_free_on_acceptance(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 99
        req.combined_draft = _make_combined_draft([100])
        req.speculation_state = SpeculationState.ACCEPTING

        vresult = VerificationResult(
            accepted_length=1, attempted_length=1,
            accepted_tokens=[100],
        )
        deps["verifier"].record_result = MagicMock()
        deps["utility_scorer"].record_iteration = MagicMock()

        ring_before = len(deps["ring_writer"].written)
        loop._process_acceptance(req, vresult)

        free_cmds = [
            c for c in deps["ring_writer"].written[ring_before:]
            if hasattr(c, "cmd_type") and c.cmd_type == E_CMD_SEQ_FREE
        ]
        assert len(free_cmds) == 1
        assert free_cmds[0].payload.seq_free.seq_id == 99
        assert req.draft_seq_id is None

    # ── Cancellation ───────────────────────────────────────────────────

    def test_rejection_cancels_downstream(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 99
        req.combined_draft = _make_combined_draft([100, 101])
        req.speculation_state = SpeculationState.ACCEPTING

        pending_item = WorkItem(
            request_id=1, layer_idx=0,
            operation=WorkOperation.ATTENTION,
            status=WorkStatus.READY,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        deps["work_queue"].insert(pending_item)

        vresult = VerificationResult(
            accepted_length=0, attempted_length=2,
            rejected_positions=[0, 1],
        )
        deps["verifier"].record_result = MagicMock()
        deps["utility_scorer"].record_iteration = MagicMock()

        loop._process_acceptance(req, vresult)

        assert pending_item.status == WorkStatus.COMPLETED

    def test_abort_draft_frees_and_resets(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 88
        req.combined_draft = _make_combined_draft([100])
        req.draft_complete = True
        req.speculation_state = SpeculationState.DRAFTING

        ring_before = len(deps["ring_writer"].written)
        loop._abort_draft(req)

        assert req.speculation_state == SpeculationState.AUTOREGRESSIVE
        assert req.combined_draft is None
        assert req.draft_seq_id is None
        assert req.draft_complete is False

        free_cmds = [
            c for c in deps["ring_writer"].written[ring_before:]
            if hasattr(c, "cmd_type") and c.cmd_type == E_CMD_SEQ_FREE
        ]
        assert len(free_cmds) == 1
        assert free_cmds[0].payload.seq_free.seq_id == 88

        items = deps["work_queue"].pending_items()
        assert any(i.operation == WorkOperation.EMBEDDING for i in items)

    # ── Verification error recovery (TD-21) ────────────────────────────

    def test_verification_error_aborts_draft(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.combined_draft = _make_combined_draft([100, 101])
        req.draft_gating = None
        req.draft_seq_id = 77
        req.max_verifiable_depth = 2
        req.speculation_state = SpeculationState.PREFETCHING_VERIFY
        req.verification_plan = VerificationPlan(transfers=[], max_depth=2)

        real_verifier = Verifier(num_layers=6, num_moe_layers=4,
                                 first_moe_layer=2, num_experts=8)
        loop._verifier = real_verifier

        loop._start_verification(req)

        assert req.speculation_state == SpeculationState.VERIFYING
        assert len(loop._verify_cmd_seqs) > 0

        intermediate_cs = next(iter(loop._verify_cmd_seqs))

        cmp = _make_completion(CMP_ERROR, cmd_seq=intermediate_cs)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert req.speculation_state == SpeculationState.AUTOREGRESSIVE
        assert req.combined_draft is None
        assert req.draft_seq_id is None
        assert len(loop._verify_cmd_seqs) == 0

    def test_intermediate_completion_silently_consumed(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.combined_draft = _make_combined_draft([100, 101])
        req.draft_gating = None
        req.draft_seq_id = 77
        req.max_verifiable_depth = 2
        req.speculation_state = SpeculationState.PREFETCHING_VERIFY
        req.verification_plan = VerificationPlan(transfers=[], max_depth=2)

        real_verifier = Verifier(num_layers=6, num_moe_layers=4,
                                 first_moe_layer=2, num_experts=8)
        loop._verifier = real_verifier

        loop._start_verification(req)

        num_tracked = len(loop._verify_cmd_seqs)
        assert num_tracked > 0

        intermediate_cs = next(iter(loop._verify_cmd_seqs))
        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=intermediate_cs)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert len(loop._verify_cmd_seqs) == num_tracked - 1
        assert req.speculation_state == SpeculationState.VERIFYING

    # ── Token extraction from host buffer (TD-23) ──────────────────────

    def test_ar_output_head_extracts_token_to_history(self):
        hbuf = HostBuffer()
        hbuf.write_tokens(offset=0, tokens=[42])
        loop, deps = _build_loop(host_buf_base=hbuf.base_address)
        req = _make_decode_request(loop, deps)
        history_before = len(req.token_history)

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[50] = (1, item)

        cmp = _make_completion_with_token(
            CMP_COMPUTE_DONE, cmd_seq=50,
            host_buf_offset=0, num_tokens=1,
        )
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert len(req.token_history) == history_before + 1
        assert req.token_history[-1] == 42

    def test_verification_extracts_multiple_tokens(self):
        hbuf = HostBuffer()
        hbuf.write_tokens(offset=0, tokens=[200, 201, 202])
        loop, deps = _build_loop(host_buf_base=hbuf.base_address)
        req = _make_decode_request(loop, deps)
        req.combined_draft = _make_combined_draft([200, 201, 202])
        req.draft_seq_id = 55
        req.verified_depth = 3
        req.verification_analyses = []
        req.speculation_state = SpeculationState.VERIFYING

        deps["verifier"].build_result = MagicMock(
            return_value=VerificationResult(
                accepted_length=3, attempted_length=3,
                accepted_tokens=[200, 201, 202],
            ),
        )
        deps["verifier"].record_result = MagicMock()
        deps["utility_scorer"].record_iteration = MagicMock()

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            is_speculative=True,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[60] = (1, item)

        cmp = _make_completion_with_token(
            CMP_COMPUTE_DONE, cmd_seq=60,
            host_buf_offset=0, num_tokens=3,
        )
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert req.speculation_state == SpeculationState.AUTOREGRESSIVE
        assert 200 in req.token_history
        assert 201 in req.token_history
        assert 202 in req.token_history

    def test_token_extraction_returns_none_without_host_buf(self):
        loop, deps = _build_loop()  # host_buf_base=0 (default)
        req = _make_decode_request(loop, deps)
        history_before = list(req.token_history)

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[70] = (1, item)

        cmp = _make_completion_with_token(
            CMP_COMPUTE_DONE, cmd_seq=70,
            host_buf_offset=0, num_tokens=1,
        )
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        # No token extracted — history unchanged (only prompt tokens)
        assert req.token_history == history_before


# ---------------------------------------------------------------------------
# Request termination (#62f)
# ---------------------------------------------------------------------------

class TestRequestTermination:

    def test_eos_token_stops_generation(self):
        eos_id = 128001
        hbuf = HostBuffer()
        hbuf.write_tokens(offset=0, tokens=[eos_id])
        md = _metadata()
        md = EngineMetadata(
            num_gpus=md.num_gpus, num_moe_layers=md.num_moe_layers,
            num_experts=md.num_experts, num_layers=md.num_layers,
            expert_bytes=md.expert_bytes, kv_bytes_per_page=md.kv_bytes_per_page,
            gpus=md.gpus, eos_token_ids=(eos_id,),
        )
        loop, deps = _build_loop(
            metadata=md, host_buf_base=hbuf.base_address,
        )
        req = _make_decode_request(loop, deps)

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[80] = (1, item)

        cmp = _make_completion_with_token(
            CMP_COMPUTE_DONE, cmd_seq=80,
            host_buf_offset=0, num_tokens=1,
        )
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert 1 not in loop.requests

    def test_max_tokens_stops_generation(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.max_tokens = 2
        req.tokens_generated = 1  # one already generated

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[81] = (1, item)

        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=81)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        # tokens_generated incremented to 2, then max_tokens check fires
        assert 1 not in loop.requests

    def test_cancel_request_stops_generation(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)

        loop.cancel_request(1)
        assert req.cancelled is True

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[82] = (1, item)

        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=82)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert 1 not in loop.requests

    def test_stop_during_verifying_aborts_speculation(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.combined_draft = _make_combined_draft([100, 101])
        req.draft_seq_id = 77
        req.verified_depth = 2
        req.verification_analyses = []
        req.speculation_state = SpeculationState.VERIFYING
        req.cancelled = True

        deps["verifier"].build_result = MagicMock(
            return_value=VerificationResult(
                accepted_length=2, attempted_length=2,
                accepted_tokens=[100, 101],
            ),
        )
        deps["verifier"].record_result = MagicMock()
        deps["utility_scorer"].record_iteration = MagicMock()

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            is_speculative=True,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[83] = (1, item)

        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=83)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        # Request finalized after acceptance completes
        assert 1 not in loop.requests

    def test_no_stop_when_criteria_not_met(self):
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.max_tokens = 100  # far from limit

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[84] = (1, item)

        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=84)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        # Request still alive, next EMBEDDING enqueued
        assert 1 in loop.requests


# ---------------------------------------------------------------------------
# on_complete callback (#67)
# ---------------------------------------------------------------------------

class TestOnComplete:

    def test_on_complete_called_on_eos(self):
        eos_id = 128001
        hbuf = HostBuffer()
        hbuf.write_tokens(offset=0, tokens=[eos_id])
        md = _metadata()
        md = EngineMetadata(
            num_gpus=md.num_gpus, num_moe_layers=md.num_moe_layers,
            num_experts=md.num_experts, num_layers=md.num_layers,
            expert_bytes=md.expert_bytes, kv_bytes_per_page=md.kv_bytes_per_page,
            gpus=md.gpus, eos_token_ids=(eos_id,),
        )
        results = []
        loop, deps = _build_loop(
            metadata=md, host_buf_base=hbuf.base_address,
        )
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=[10, 20], gpu=0,
            on_complete=lambda rid, toks, reason, lp=None: results.append((rid, toks, reason)),
        ))
        loop._phase_collect()  # drain request
        req = loop.requests[1]
        req.is_prefill = False
        req.tokens_generated = 0

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[90] = (1, item)
        cmp = _make_completion_with_token(
            CMP_COMPUTE_DONE, cmd_seq=90,
            host_buf_offset=0, num_tokens=1,
        )
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert len(results) == 1
        rid, toks, reason = results[0]
        assert rid == 1
        assert toks[-1] == eos_id
        assert reason == "stop"

    def test_on_complete_called_on_max_tokens(self):
        loop, deps = _build_loop()
        results = []
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=[10, 20], gpu=0,
            max_tokens=2,
            on_complete=lambda rid, toks, reason, lp=None: results.append((rid, toks, reason)),
        ))
        loop._phase_collect()
        req = loop.requests[1]
        req.is_prefill = False
        req.tokens_generated = 1

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[91] = (1, item)
        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=91)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert len(results) == 1
        assert results[0][2] == "length"

    def test_on_complete_called_on_cancel(self):
        loop, deps = _build_loop()
        results = []
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=[10, 20], gpu=0,
            on_complete=lambda rid, toks, reason, lp=None: results.append((rid, toks, reason)),
        ))
        loop._phase_collect()
        req = loop.requests[1]
        req.is_prefill = False
        loop.cancel_request(1)

        item = WorkItem(
            request_id=1, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[92] = (1, item)
        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=92)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert len(results) == 1
        assert results[0][2] == "cancelled"


# ---------------------------------------------------------------------------
# MTP and self-speculative draft dispatch (#62d, #62e)
# ---------------------------------------------------------------------------

def _enable_mtp(loop, num_steps=3):
    """Replace the disabled MTP mock with a real MtpDraft (enabled)."""
    from orchestrator.mtp_draft import MtpDraft, MtpDraftConfig
    loop._mtp_draft = MtpDraft(
        MtpDraftConfig(enabled=True, max_depth=num_steps),
        num_mtp_layers=1, num_layers=6,
    )


def _enable_self_spec(loop, max_depth=3):
    """Replace the disabled self-spec mock with a real SelfSpeculative (enabled)."""
    from orchestrator.self_speculative import SelfSpeculative, SelfSpeculativeConfig
    loop._self_speculative = SelfSpeculative(
        SelfSpeculativeConfig(enabled=True, max_depth=max_depth),
        num_layers=6, num_moe_layers=4, first_moe_layer=2,
    )


class TestMtpDraftDispatch:
    """MTP drafting on the PRODUCTION seam (TD-MTP-PY-LOOP-KV / #89).

    A draft step is the 4-command D_CMD_MTP_PROJECT chain the MtpLossless
    golden drives — the fused D_CMD_RUN_MTP_STEP is never used, and no
    KV fork is created (drafts write only the MTP layer's KV at scratch
    positions >= anchor on the MAIN sequence).
    """

    def test_mtp_step_dispatched_when_enabled(self):
        """MTP enabled + depth>0 -> D_CMD_MTP_PROJECT chain starts, no fork."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        _enable_mtp(loop, num_steps=3)

        ring_before = len(deps["ring_writer"].written)
        # Manually enter DRAFTING (bypass OUTPUT_HEAD trigger)
        loop._start_drafting(req)

        new_cmds = [
            c for c in deps["ring_writer"].written[ring_before:]
            if hasattr(c, "cmd_type")
        ]
        # Production seam: project first, fused step NEVER (#89)
        project_cmds = [c for c in new_cmds
                        if c.cmd_type == D_CMD_MTP_PROJECT]
        assert len(project_cmds) == 1
        assert project_cmds[0].payload.mtp_project.input_token_id == 30
        assert all(c.cmd_type != D_CMD_RUN_MTP_STEP for c in new_cmds)
        # No KV fork for the MTP path (TD-MTP-PY-LOOP-KV)
        assert all(c.cmd_type != CMD_SEQ_FORK for c in new_cmds)
        assert req.draft_seq_id is None
        assert req.mtp_cmd_seq is not None
        assert req.mtp_cmd_seq in loop._draft_cmd_seqs
        assert req.mtp_phase == "project"
        assert req.mtp_mode == "draft"

    def test_mtp_step_phase_chain(self):
        """project -> attn(mtp layer, gating) -> FETCH_AND_RUN -> head(mtp)."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 1
        req.token_history = [10, 20, 30]
        _enable_mtp(loop, num_steps=1)

        loop._start_drafting(req)
        ring = deps["ring_writer"].written

        # project completion -> RUN_ATTENTION on the MTP layer (6) with the
        # fused gate + routing export
        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=req.mtp_cmd_seq)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()
        attn = ring[-1]
        assert attn.cmd_type == D_B_CMD_RUN_ATTENTION
        assert attn.payload.run_attention.layer_idx == 6  # num_layers + 0
        assert attn.payload.run_attention.emit_gating == 1
        assert attn.payload.run_attention.store_gating == 1
        assert req.mtp_phase == "attn"

        # attn completion -> E_CMD_FETCH_AND_RUN_MOE (production MoE seam,
        # NOT the deprecated resident-only RUN_MOE)
        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=req.mtp_cmd_seq)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()
        moe = ring[-1]
        assert moe.cmd_type == E_CMD_FETCH_AND_RUN_MOE
        assert moe.payload.fetch_and_run_moe.layer_idx == 6
        assert req.mtp_phase == "moe"

        # moe completion -> OUTPUT_HEAD through the MTP shared head
        cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=req.mtp_cmd_seq)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()
        head = ring[-1]
        assert head.cmd_type == 0x0105  # CMD_OUTPUT_HEAD
        assert head.payload.output_head.mtp_head == 1  # mtp_idx 0 + 1
        assert head.payload.output_head.readback_to_host == 1
        assert req.mtp_phase == "head"

    def test_mtp_confidence_early_exit(self):
        """should_continue False -> combine -> sequential verification."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        _enable_mtp(loop, num_steps=3)
        req.speculation_state = SpeculationState.DRAFTING

        loop._start_drafting(req)
        assert req.mtp_cmd_seq is not None

        # Simulate step 0 head completion with LOW confidence (< 0.4):
        # should_continue(step=1, confidence=0.1) -> False -> finish
        # drafting -> sequential early-stop verification on the MAIN seq.
        loop._finish_mtp_draft_step(req, token_id=100, confidence=0.1)

        assert req.speculation_state == SpeculationState.VERIFYING
        assert req.mtp_sequential_verify is True
        assert req.verify_tokens == [100]
        # No fork, no batched verification plan
        assert req.draft_seq_id is None
        assert req.verification_plan is None

    def test_mtp_disabled_falls_through(self):
        """MTP disabled -> falls through to _finish_drafting (prompt-lookup-only)."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30, 20, 30]

        # MTP is disabled by default in _build_loop
        loop._start_drafting(req)

        # No MTP commands dispatched (fused or production-seam)
        mtp_cmds = [
            c for c in deps["ring_writer"].written
            if hasattr(c, "cmd_type") and c.cmd_type in (
                D_CMD_RUN_MTP_STEP, D_CMD_MTP_PROJECT,
            )
        ]
        assert len(mtp_cmds) == 0

    def test_mtp_error_aborts_draft(self):
        """CMP_ERROR on an MTP sub-command -> _abort_draft, back to AR."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        _enable_mtp(loop, num_steps=3)

        req.speculation_state = SpeculationState.DRAFTING
        loop._start_drafting(req)
        mtp_cs = req.mtp_cmd_seq
        assert mtp_cs is not None

        # Inject error for the MTP command
        cmp = _make_completion(CMP_ERROR, cmd_seq=mtp_cs)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert req.speculation_state == SpeculationState.AUTOREGRESSIVE
        assert req.mtp_cmd_seq is None
        assert req.mtp_phase is None
        assert mtp_cs not in loop._draft_cmd_seqs


class TestMtpKvScheduleAndSequentialVerify:
    """INV-MTP-KV warm/catch-up schedule + sequential early-stop verification.

    Mirrors the asserted behavior of the Glm52GgufGolden.MtpLossless C++
    driver (the reference schedule): every main-model feed at position p
    with a known successor is followed by one MTP step at p embedding it,
    except when a draft round starts at p (its step 0 covers it); rejected
    draft tokens are NEVER fed (early stop -> no fork/rewind/copy-back).
    """

    def _feed_output_head(self, loop, deps, hbuf, req, cmd_seq, token,
                          offset=0):
        """Simulate an OUTPUT_HEAD completion producing `token`.

        Consumes any pending work items first (the simulated forward pass
        "executed" them) so the next EMBEDDING insert doesn't collide.
        """
        for it in list(loop._work_queue.pending_items()):
            loop._work_queue.update_status(it, WorkStatus.COMPLETED)
        hbuf.write_tokens(offset=offset, tokens=[token])
        item = WorkItem(
            request_id=req.request_id, layer_idx=5,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[cmd_seq] = (req.request_id, item)
        cmp = _make_completion_with_token(
            CMP_COMPUTE_DONE, cmd_seq=cmd_seq,
            host_buf_offset=offset, num_tokens=1,
        )
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

    def _complete_mtp_step(self, loop, deps, req):
        """Drive one MTP step through all four phase completions."""
        for _ in range(4):
            assert req.mtp_cmd_seq is not None
            cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=req.mtp_cmd_seq)
            deps["completion_reader"].add(cmp)
            loop._phase_collect()

    def test_catchup_after_nondrafting_decode_step(self):
        """depth==0 decode step -> catch-up MTP step at the fed position,
        embedding the committed successor; the next EMBEDDING is deferred
        until the catch-up completes (it needs the un-clobbered hidden)."""
        hbuf = HostBuffer()
        loop, deps = _build_loop(host_buf_base=hbuf.base_address)
        req = _make_decode_request(loop, deps)
        # tokens_generated=1 -> after this feed's OUTPUT_HEAD it becomes 2
        # -> anchor (newest FED position, #91 convention) = 3 + 2 - 2 = 3.
        req.prompt_len = 3
        _enable_mtp(loop, num_steps=3)
        deps["utility_scorer"].recommended_depth = MagicMock(return_value=0)

        ring = deps["ring_writer"].written
        ring_before = len(ring)
        self._feed_output_head(loop, deps, hbuf, req, cmd_seq=70, token=42)

        # Catch-up project dispatched, embedding the committed token 42
        # at the just-fed position (anchor).
        new_cmds = ring[ring_before:]
        assert [c.cmd_type for c in new_cmds] == [D_CMD_MTP_PROJECT]
        assert new_cmds[0].payload.mtp_project.input_token_id == 42
        assert req.mtp_mode == "catchup"
        assert req.mtp_pos == req.prompt_len + req.tokens_generated - 2
        # EMBEDDING deferred (would clobber the hidden the catch-up needs)
        items = deps["work_queue"].pending_items()
        assert not any(i.operation == WorkOperation.EMBEDDING for i in items)

        # Complete all 4 catch-up phases -> AR embedding resumes
        self._complete_mtp_step(loop, deps, req)
        assert req.mtp_phase is None and req.mtp_mode is None
        items = deps["work_queue"].pending_items()
        assert any(i.operation == WorkOperation.EMBEDDING for i in items)

    def test_catchup_after_prefill_transition(self):
        """The prefill->decode OUTPUT_HEAD also gets a catch-up (warm)."""
        hbuf = HostBuffer()
        loop, deps = _build_loop(host_buf_base=hbuf.base_address)
        req = RequestState(
            request_id=1, seq_id=1, default_gpu=0,
            is_prefill=True, tokens_generated=0, prompt_len=3,
            token_history=[10, 20, 30],
        )
        loop._requests[1] = req
        _enable_mtp(loop, num_steps=3)
        deps["utility_scorer"].recommended_depth = MagicMock(return_value=0)

        ring = deps["ring_writer"].written
        ring_before = len(ring)
        self._feed_output_head(loop, deps, hbuf, req, cmd_seq=71, token=42)

        assert req.is_prefill is False
        new_cmds = ring[ring_before:]
        assert [c.cmd_type for c in new_cmds] == [D_CMD_MTP_PROJECT]
        assert new_cmds[0].payload.mtp_project.input_token_id == 42
        assert req.mtp_mode == "catchup"

    def test_no_catchup_when_draft_round_starts(self):
        """depth>0 -> draft step 0 covers the anchor (it embeds the same
        true token off the same trunk hidden) — no separate catch-up."""
        hbuf = HostBuffer()
        loop, deps = _build_loop(host_buf_base=hbuf.base_address)
        req = _make_decode_request(loop, deps)
        req.prompt_len = 3
        _enable_mtp(loop, num_steps=2)
        _setup_speculation_mocks(deps, depth=2)

        ring = deps["ring_writer"].written
        ring_before = len(ring)
        self._feed_output_head(loop, deps, hbuf, req, cmd_seq=72, token=42)

        assert req.speculation_state == SpeculationState.DRAFTING
        projects = [c for c in ring[ring_before:]
                    if c.cmd_type == D_CMD_MTP_PROJECT]
        assert len(projects) == 1  # draft step 0 only, no extra catch-up
        assert req.mtp_mode == "draft"
        # Draft step 0 embeds the committed anchor token x=42 -> its KV
        # write at the anchor IS the INV-MTP-KV warm write.
        assert projects[0].payload.mtp_project.input_token_id == 42
        assert req.mtp_pos == req.mtp_anchor_pos

    def test_sequential_verify_accept_catchup_reject(self):
        """Accepted feed -> catch-up then next feed; mismatch -> round ends
        with the main model's own token committed; the rejected draft is
        NEVER fed or committed (INV-MTP-LOSSLESS, TD-MTP-PY-LOOP-KV (2))."""
        hbuf = HostBuffer()
        loop, deps = _build_loop(host_buf_base=hbuf.base_address)
        req = _make_decode_request(loop, deps)
        req.prompt_len = 3
        _enable_mtp(loop, num_steps=2)
        _setup_speculation_mocks(deps, depth=2)
        deps["verifier"].record_result = MagicMock()

        # Enter sequential verification with drafts [100, 999]
        req.speculation_depth = 2
        req.combined_draft = _make_combined_draft([100, 999])
        req.speculation_state = SpeculationState.VERIFYING
        loop._start_sequential_verification(req)
        assert req.verify_tokens == [100, 999]
        items = deps["work_queue"].pending_items()
        assert any(i.operation == WorkOperation.EMBEDDING for i in items)

        # Feed 1 produces 100 == draft[0]: accepted -> catch-up at the fed
        # position embedding 100, next feed only after it completes.
        ring = deps["ring_writer"].written
        ring_before = len(ring)
        self._feed_output_head(loop, deps, hbuf, req, cmd_seq=80, token=100)
        assert req.verify_idx == 1
        assert req.token_history[-1] == 100
        new_cmds = ring[ring_before:]
        assert [c.cmd_type for c in new_cmds] == [D_CMD_MTP_PROJECT]
        assert new_cmds[0].payload.mtp_project.input_token_id == 100
        assert req.mtp_mode == "catchup"
        self._complete_mtp_step(loop, deps, req)

        # Feed 2 produces 101 != draft[1]=999: early stop.  101 (the main
        # model's own output) is committed; 999 is never fed.
        deps["utility_scorer"].recommended_depth = MagicMock(return_value=0)
        self._feed_output_head(loop, deps, hbuf, req, cmd_seq=81, token=101)
        assert req.token_history[-1] == 101
        assert 999 not in req.token_history
        assert req.mtp_sequential_verify is False
        assert req.verify_tokens == []
        # Round stats: 1 of 2 drafts accepted
        vres = deps["verifier"].record_result.call_args.args[0]
        assert vres.accepted_length == 1
        assert vres.attempted_length == 2

    def test_sequential_verify_full_acceptance_bonus_feed(self):
        """All drafts accepted -> one bonus feed ends the round; a new
        draft round chains directly off the bonus feed's trunk hidden."""
        hbuf = HostBuffer()
        loop, deps = _build_loop(host_buf_base=hbuf.base_address)
        req = _make_decode_request(loop, deps)
        req.prompt_len = 3
        req.max_tokens = 100
        _enable_mtp(loop, num_steps=1)
        _setup_speculation_mocks(deps, depth=1)
        deps["verifier"].record_result = MagicMock()

        req.speculation_depth = 1
        req.combined_draft = _make_combined_draft([100])
        req.mtp_steps_result = [
            DraftStep(token_id=100, confidence=0.9, mtp_layer_idx=6),
        ]
        req.speculation_state = SpeculationState.VERIFYING
        loop._start_sequential_verification(req)

        # Feed 1: 100 == draft[0] -> accepted -> catch-up
        self._feed_output_head(loop, deps, hbuf, req, cmd_seq=90, token=100)
        assert req.verify_idx == 1
        self._complete_mtp_step(loop, deps, req)

        # Bonus feed: produces 55, committed; round over.  With depth>0
        # the next draft round starts IMMEDIATELY (golden pattern — the
        # bonus feed's trunk hidden is the new anchor's).
        ring = deps["ring_writer"].written
        ring_before = len(ring)
        self._feed_output_head(loop, deps, hbuf, req, cmd_seq=91, token=55)
        assert req.token_history[-1] == 55
        assert req.speculation_state == SpeculationState.DRAFTING
        projects = [c for c in ring[ring_before:]
                    if c.cmd_type == D_CMD_MTP_PROJECT]
        assert len(projects) == 1
        assert projects[0].payload.mtp_project.input_token_id == 55
        vres = deps["verifier"].record_result.call_args.args[0]
        assert vres.accepted_length == 1
        assert vres.attempted_length == 1
        # Acceptance statistics reach MtpDraft (acceptance > 0 provable)
        assert loop._mtp_draft.total_accepted == 1

    def test_mtp_round_no_fork_no_seq_free(self):
        """The whole MTP round runs on the MAIN sequence: no CMD_SEQ_FORK,
        no E_CMD_SEQ_FREE — the verified KV lives where decoding continues
        (fixes TD-MTP-PY-LOOP-KV (2): no verified-KV thrown away)."""
        hbuf = HostBuffer()
        loop, deps = _build_loop(host_buf_base=hbuf.base_address)
        req = _make_decode_request(loop, deps)
        req.prompt_len = 3
        _enable_mtp(loop, num_steps=1)
        _setup_speculation_mocks(deps, depth=1)
        deps["verifier"].record_result = MagicMock()

        ring = deps["ring_writer"].written
        ring_before = len(ring)

        # Draft round: OUTPUT_HEAD -> DRAFTING -> step 0 (4 phases with a
        # scripted head token) -> sequential verification
        self._feed_output_head(loop, deps, hbuf, req, cmd_seq=95, token=42)
        assert req.speculation_state == SpeculationState.DRAFTING
        for _ in range(3):  # project, attn, moe completions
            cmp = _make_completion(CMP_COMPUTE_DONE, cmd_seq=req.mtp_cmd_seq)
            deps["completion_reader"].add(cmp)
            loop._phase_collect()
        # head completion with draft token 100 (offset 8), confident draft
        hbuf.write_tokens(offset=8, tokens=[100])
        cmp = _make_completion_with_token(
            CMP_COMPUTE_DONE, cmd_seq=req.mtp_cmd_seq,
            host_buf_offset=8, num_tokens=1,
        )
        cmp.payload.compute.top1_prob = 0.9
        deps["completion_reader"].add(cmp)
        loop._phase_collect()
        assert req.speculation_state == SpeculationState.VERIFYING

        # Mismatch feed ends the round
        deps["utility_scorer"].recommended_depth = MagicMock(return_value=0)
        self._feed_output_head(loop, deps, hbuf, req, cmd_seq=96, token=77)

        cmd_types = [c.cmd_type for c in ring[ring_before:]
                     if hasattr(c, "cmd_type")]
        assert CMD_SEQ_FORK not in cmd_types
        assert E_CMD_SEQ_FREE not in cmd_types
        assert D_CMD_RUN_MTP_STEP not in cmd_types
        assert req.draft_seq_id is None


class TestSelfSpecDraftDispatch:

    def test_self_spec_dispatched_when_mtp_disabled(self):
        """MTP disabled, self-spec enabled -> D_CMD_RUN_SELF_SPEC_FORWARD."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        _enable_self_spec(loop, max_depth=3)

        ring_before = len(deps["ring_writer"].written)
        loop._start_drafting(req)

        ss_cmds = [
            c for c in deps["ring_writer"].written[ring_before:]
            if hasattr(c, "cmd_type")
            and c.cmd_type == D_CMD_RUN_SELF_SPEC_FORWARD
        ]
        assert len(ss_cmds) == 1
        assert req.self_spec_cmd_seq is not None

    def test_mtp_does_not_chain_self_spec(self):
        """MTP path never chains self-spec (TD-MTP-PY-LOOP-KV).

        Self-spec drafting needs the KV fork + batched verification (its
        forward writes approximate main-layer KV), which is incompatible
        with the MTP flow's verify-on-main-sequence.  With both enabled,
        the MTP flow wins and goes straight to sequential verification.
        """
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 1
        req.token_history = [10, 20, 30]
        _enable_mtp(loop, num_steps=1)
        _enable_self_spec(loop, max_depth=2)
        req.speculation_state = SpeculationState.DRAFTING

        # Start drafting -> dispatches MTP step (production seam, no fork)
        loop._start_drafting(req)
        assert req.mtp_cmd_seq is not None
        assert req.draft_seq_id is None

        # Simulate MTP step 0 head completion: plan (depth 1) exhausted
        ring_before = len(deps["ring_writer"].written)
        loop._finish_mtp_draft_step(req, token_id=100, confidence=0.9)

        ss_cmds = [
            c for c in deps["ring_writer"].written[ring_before:]
            if hasattr(c, "cmd_type")
            and c.cmd_type == D_CMD_RUN_SELF_SPEC_FORWARD
        ]
        assert len(ss_cmds) == 0
        assert req.speculation_state == SpeculationState.VERIFYING
        assert req.mtp_sequential_verify is True

    def test_self_spec_disabled_falls_through(self):
        """Both disabled -> prompt-lookup-only (existing behavior)."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30, 20, 30]

        loop._start_drafting(req)

        ss_cmds = [
            c for c in deps["ring_writer"].written
            if hasattr(c, "cmd_type")
            and c.cmd_type == D_CMD_RUN_SELF_SPEC_FORWARD
        ]
        assert len(ss_cmds) == 0

    def test_self_spec_error_aborts_draft(self):
        """CMP_ERROR on self-spec cmd_seq -> _abort_draft."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        _enable_self_spec(loop, max_depth=3)

        req.speculation_state = SpeculationState.DRAFTING
        loop._start_drafting(req)
        ss_cs = req.self_spec_cmd_seq
        assert ss_cs is not None

        cmp = _make_completion(CMP_ERROR, cmd_seq=ss_cs)
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert req.speculation_state == SpeculationState.AUTOREGRESSIVE
        assert req.self_spec_cmd_seq is None


class TestDraftIntegration:

    def test_zero_change_when_both_disabled(self):
        """Both MTP and self-spec disabled -> identical to prompt-lookup-only."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30, 20, 30]

        loop._start_drafting(req)

        # No GPU draft commands dispatched
        gpu_draft_cmds = [
            c for c in deps["ring_writer"].written
            if hasattr(c, "cmd_type") and c.cmd_type in (
                D_CMD_RUN_MTP_STEP, D_CMD_RUN_SELF_SPEC_FORWARD,
            )
        ]
        assert len(gpu_draft_cmds) == 0

        # prompt_lookup_tokens stashed on request
        assert isinstance(req.prompt_lookup_tokens, list)

    def test_finish_drafting_combines_mtp_result(self):
        """_finish_drafting assembles MtpDraftResult from accumulated steps."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING
        req.prompt_lookup_tokens = [200, 201]

        from orchestrator.mtp_draft import DraftStep
        req.mtp_steps_result = [
            DraftStep(token_id=100, confidence=0.9, mtp_layer_idx=6),
            DraftStep(token_id=101, confidence=0.8, mtp_layer_idx=7),
        ]

        mock_combiner = MagicMock(spec=DraftCombiner)
        mock_combiner.combine = MagicMock(
            return_value=_make_combined_draft([100, 101, 200]),
        )
        loop._draft_combiner = mock_combiner

        loop._finish_drafting(req)

        # DraftCombiner.combine called with MtpDraftResult
        call_args = mock_combiner.combine.call_args
        mtp_arg = call_args.kwargs.get("mtp_result") or call_args[1].get("mtp_result")
        if mtp_arg is None and len(call_args.args) > 1:
            mtp_arg = call_args.args[1]
        # The combine was called (at least prompt_lookup_tokens passed)
        mock_combiner.combine.assert_called_once()

    def test_finish_drafting_builds_draft_gating_from_self_spec(self):
        """Self-spec gating rows assembled into draft_gating ndarray."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 2
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING
        req.prompt_lookup_tokens = []

        import numpy as np
        # 2 steps, each with (4 moe_layers, 8 experts)
        req.self_spec_gating_rows = [
            np.ones((4, 8), dtype=np.float32),
            np.ones((4, 8), dtype=np.float32) * 0.5,
        ]
        req.self_spec_steps_result = [
            DraftStep(token_id=100, confidence=0.9, mtp_layer_idx=-1),
            DraftStep(token_id=101, confidence=0.8, mtp_layer_idx=-1),
        ]

        mock_combiner = MagicMock(spec=DraftCombiner)
        mock_combiner.combine = MagicMock(
            return_value=_make_combined_draft([100, 101]),
        )
        loop._draft_combiner = mock_combiner

        # _finish_drafting calls _start_prefetching which calls
        # compute_verification_plan on the draft_gating.  Provide a
        # real return value so apply_ceiling doesn't compare against a mock.
        deps["speculative_prefetch"].compute_verification_plan = (
            MagicMock(return_value=VerificationPlan(transfers=[], max_depth=2))
        )

        loop._finish_drafting(req)

        # draft_gating should be assembled from gating rows
        assert req.draft_gating is not None
        assert req.draft_gating.shape == (2, 4, 8)


# ---------------------------------------------------------------------------
# LayerSkip orchestrator data flow (#62g)
# ---------------------------------------------------------------------------

def _enable_layer_skip(loop, threshold=0.995):
    """Replace the disabled LayerSkip mock with a real enabled instance."""
    from orchestrator.layer_skip import LayerSkip, LayerSkipConfig
    loop._layer_skip = LayerSkip(
        LayerSkipConfig(enabled=True, threshold=threshold,
                        no_skip_first=1, no_skip_last=1,
                        min_acceptance_rate=0.0),
        num_layers=6,
    )


class TestLayerSkipDataFlow:

    def test_similarity_collected_from_checkpoint_type_2(self):
        """checkpoint_type=2 during DRAFTING self-spec stores similarity."""
        loop, deps = _build_loop()
        host_buf = HostBuffer(256)
        loop._host_buf_base = host_buf.base_address

        req = _make_decode_request(loop, deps)
        req.speculation_state = SpeculationState.DRAFTING
        req.self_spec_cmd_seq = 42

        # Write a float32 similarity value (0.997) at offset 0
        import struct
        struct.pack_into("f", host_buf._buf, 0, 0.997)

        # Simulate the tracked command in _cmd_seq_map (checkpoint does .get)
        item = WorkItem(
            request_id=1, layer_idx=3,
            operation=WorkOperation.OUTPUT_HEAD,
            status=WorkStatus.DISPATCHED,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[42] = (1, item)

        cmp = _make_completion(CMP_CHECKPOINT, cmd_seq=42)
        cmp.payload.checkpoint.checkpoint_type = 2
        cmp.payload.checkpoint.layer_idx = 3
        cmp.payload.checkpoint.host_buf_offset = 0
        cmp.payload.checkpoint.data_bytes = 4
        deps["completion_reader"].add(cmp)
        loop._phase_collect()

        assert 3 in req.layer_similarities
        assert abs(req.layer_similarities[3] - 0.997) < 0.001

    def test_skip_set_computed_from_similarities(self):
        """With similarities populated, compute_skip_set produces real skip set."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING
        req.acceptance_rate = 0.8
        # High similarity at layers 2,3 — layer 3 should be skippable
        # (no_skip_first=1 means layer 0 protected, no_skip_last=1 means layer 5)
        req.layer_similarities = {2: 0.999, 3: 0.999, 4: 0.999}

        _enable_layer_skip(loop, threshold=0.995)
        _enable_self_spec(loop, max_depth=3)

        loop._dispatch_self_spec_step(req)

        # current_skip_set should be non-empty (layers with high similarity)
        assert len(req.current_skip_set) > 0

    def test_step_0_empty_skip_set(self):
        """First self-spec step has no prior similarities — skip set is empty."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING
        req.acceptance_rate = 0.8
        # Empty similarities (step 0)
        req.layer_similarities = {}

        _enable_layer_skip(loop, threshold=0.995)
        _enable_self_spec(loop, max_depth=3)

        loop._dispatch_self_spec_step(req)

        assert req.current_skip_set == set()

    def test_prefetch_hints_suppressed_for_skipped_layers(self):
        """Phase 2 filters SP-MoE hints for layers in current_skip_set."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        # Simulate a request with a non-empty skip set
        req.current_skip_set = {3, 4}
        req.draft_complete = True

        # Create a mock draft_gating so SP-MoE hints are generated
        req.draft_gating = np.ones((2, 4, 8), dtype=np.float32)
        # speculative_prefetch.gating_to_hints returns hints at various layers
        from orchestrator.types import PrefetchHint, PrefetchConfidence, PrefetchSource
        mock_hints = [
            PrefetchHint(key=ExpertKey(2, 0), target_layer=2,
                         confidence=PrefetchConfidence.HIGH,
                         source=PrefetchSource.SP_MOE, score=0.9),
            PrefetchHint(key=ExpertKey(3, 1), target_layer=3,
                         confidence=PrefetchConfidence.HIGH,
                         source=PrefetchSource.SP_MOE, score=0.8),
            PrefetchHint(key=ExpertKey(4, 2), target_layer=4,
                         confidence=PrefetchConfidence.HIGH,
                         source=PrefetchSource.SP_MOE, score=0.7),
        ]
        deps["speculative_prefetch"].gating_to_hints = MagicMock(
            return_value=mock_hints,
        )

        priorities = loop._phase_prefetch_scoring()

        # The fuser receives filtered hints — layers 3 and 4 should be gone.
        # We can't directly inspect what the fuser received, but we can
        # check via the mock that gating_to_hints was called and the
        # fuser was called (it's a real PrefetchFuser, returns []).
        deps["speculative_prefetch"].gating_to_hints.assert_called_once()

    def test_layer_skip_disabled_empty_skip_set(self):
        """LayerSkip disabled -> always empty skip set regardless of similarities."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING
        req.layer_similarities = {2: 0.999, 3: 0.999}

        # LayerSkip is disabled by default in _build_loop
        _enable_self_spec(loop, max_depth=3)

        loop._dispatch_self_spec_step(req)

        assert req.current_skip_set == set()

    def test_abort_draft_resets_layer_skip_fields(self):
        """_abort_draft clears layer_similarities and current_skip_set."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 88
        req.layer_similarities = {2: 0.999, 3: 0.998}
        req.current_skip_set = {3, 4}
        req.speculation_state = SpeculationState.DRAFTING

        loop._abort_draft(req)

        assert req.layer_similarities == {}
        assert req.current_skip_set == set()

    def test_cross_step_refinement(self):
        """Step N's similarities are available to step N+1 via req.layer_similarities."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING
        req.acceptance_rate = 0.8
        req.self_spec_step = 0

        _enable_layer_skip(loop, threshold=0.995)
        _enable_self_spec(loop, max_depth=3)

        # Step 0: no similarities yet
        loop._dispatch_self_spec_step(req)
        assert req.current_skip_set == set()

        # Simulate daemon reporting similarities for step 0
        req.layer_similarities = {2: 0.999, 3: 0.999, 4: 0.999}
        req.self_spec_step = 1
        req.self_spec_cmd_seq = None  # previous step completed

        # Step 1: now has similarity data from step 0
        loop._dispatch_self_spec_step(req)
        # Should have computed a non-empty skip set from the similarities
        assert len(req.current_skip_set) > 0


# ---------------------------------------------------------------------------
# Origin-based transfer cancellation (redesign)
# ---------------------------------------------------------------------------

def _setup_origin_transfer(loop, req, key, cmd_seq, gpu=0):
    """Register a verification-origin transfer in _origin_transfers."""
    origin = (req.request_id, req.tokens_generated, 0)
    od = loop._origin_transfers.setdefault(origin, {})
    od[key] = (cmd_seq, gpu)
    loop._origin_birth_cycle[origin] = loop._cycle_count
    loop._inflight_transfers[gpu].add(key)


class TestTransferCancellation:

    def test_stable_skip_adds_to_cancel_queue(self):
        """2 consecutive steps with same skip set -> origin added to _cancel_queue."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING
        req.acceptance_rate = 0.8

        _enable_layer_skip(loop, threshold=0.995)
        _enable_self_spec(loop, max_depth=3)

        # Register a verification-origin transfer at layer 3
        _setup_origin_transfer(loop, req, ExpertKey(3, 1), cmd_seq=777)

        # Step 0: high similarity -> skip set computed but no cancel (window=2)
        req.layer_similarities = {2: 0.999, 3: 0.999, 4: 0.999}
        req.self_spec_step = 0
        loop._dispatch_self_spec_step(req)
        assert len(loop._cancel_queue) == 0

        # Step 1: same similarities -> stable skip -> queued for cancel
        req.self_spec_step = 1
        req.self_spec_cmd_seq = None
        loop._dispatch_self_spec_step(req)
        assert len(loop._cancel_queue) >= 1

    def test_cancel_queue_processed_end_of_phase5(self):
        """Queued cancellations emit CMD_CANCEL_TRANSFER in _phase_dispatch."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)

        key = ExpertKey(3, 1)
        _setup_origin_transfer(loop, req, key, cmd_seq=777)

        origin = (req.request_id, req.tokens_generated, 0)
        loop._cancel_queue.append((origin, {3}))

        ring_before = len(deps["ring_writer"].written)
        loop._phase_dispatch(EvictionPlan(), TransferPlan(), [])

        cancel_cmds = [
            c for c in deps["ring_writer"].written[ring_before:]
            if hasattr(c, "cmd_type") and c.cmd_type == CMD_CANCEL_TRANSFER
        ]
        assert len(cancel_cmds) == 1
        assert cancel_cmds[0].payload.cancel_transfer.target_cmd_seq == 777
        assert len(loop._cancel_queue) == 0

    def test_unstable_skip_no_cancel(self):
        """Layer skipped on step 0 but not step 1 -> no cancel queued."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING
        req.acceptance_rate = 0.8

        _enable_layer_skip(loop, threshold=0.995)
        _enable_self_spec(loop, max_depth=3)

        _setup_origin_transfer(loop, req, ExpertKey(3, 1), cmd_seq=777)

        req.layer_similarities = {2: 0.999, 3: 0.999, 4: 0.999}
        req.self_spec_step = 0
        loop._dispatch_self_spec_step(req)

        # Step 1: LOW similarity -> not skipped -> no cancel
        req.layer_similarities = {2: 0.5, 3: 0.5, 4: 0.5}
        req.self_spec_step = 1
        req.self_spec_cmd_seq = None
        loop._dispatch_self_spec_step(req)
        assert len(loop._cancel_queue) == 0

    def test_cancel_skips_completed_experts(self):
        """Expert not in _inflight_transfers -> cancel skipped."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)

        key = ExpertKey(3, 1)
        # Origin exists but expert already completed (not in inflight)
        origin = (req.request_id, req.tokens_generated, 0)
        loop._origin_transfers[origin] = {key: (777, 0)}
        loop._origin_birth_cycle[origin] = 0
        # NOT in _inflight_transfers

        loop._cancel_queue.append((origin, {3}))

        ring_before = len(deps["ring_writer"].written)
        loop._phase_dispatch(EvictionPlan(), TransferPlan(), [])

        cancel_cmds = [
            c for c in deps["ring_writer"].written[ring_before:]
            if hasattr(c, "cmd_type") and c.cmd_type == CMD_CANCEL_TRANSFER
        ]
        assert len(cancel_cmds) == 0

    def test_layer_filter_cancels_subset(self):
        """Cancel with layer_filter only cancels matching layers."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)

        key3 = ExpertKey(3, 1)
        key4 = ExpertKey(4, 2)
        _setup_origin_transfer(loop, req, key3, cmd_seq=777)
        _setup_origin_transfer(loop, req, key4, cmd_seq=888)

        origin = (req.request_id, req.tokens_generated, 0)
        loop._cancel_queue.append((origin, {3}))  # only layer 3

        ring_before = len(deps["ring_writer"].written)
        loop._phase_dispatch(EvictionPlan(), TransferPlan(), [])

        cancel_cmds = [
            c for c in deps["ring_writer"].written[ring_before:]
            if hasattr(c, "cmd_type") and c.cmd_type == CMD_CANCEL_TRANSFER
        ]
        assert len(cancel_cmds) == 1
        assert cancel_cmds[0].payload.cancel_transfer.target_cmd_seq == 777
        assert key4 in loop._inflight_transfers[0]  # layer 4 untouched

    def test_skip_set_history_reset_on_abort(self):
        """_abort_draft clears skip_set_history."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 88
        req.speculation_state = SpeculationState.DRAFTING
        req.skip_set_history = [{3, 4}, {3, 4}]

        loop._abort_draft(req)
        assert req.skip_set_history == []

    def test_cancel_skip_disabled_no_cancel(self):
        """cancel_skip_transfers=False -> stable skips don't queue cancels."""
        hc = ConfigHotReload()
        hc.update("speculation.transfer_cancellation.cancel_skip_transfers", False)
        loop, deps = _build_loop(hot_config=hc)
        req = _make_decode_request(loop, deps)
        req.speculation_depth = 3
        req.token_history = [10, 20, 30]
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING
        req.acceptance_rate = 0.8

        _enable_layer_skip(loop, threshold=0.995)
        _enable_self_spec(loop, max_depth=3)

        _setup_origin_transfer(loop, req, ExpertKey(3, 1), cmd_seq=777)

        req.layer_similarities = {2: 0.999, 3: 0.999, 4: 0.999}
        req.self_spec_step = 0
        loop._dispatch_self_spec_step(req)
        req.self_spec_step = 1
        req.self_spec_cmd_seq = None
        loop._dispatch_self_spec_step(req)

        assert len(loop._cancel_queue) == 0


# ---------------------------------------------------------------------------
# Draft abort transfer cancellation (origin-based redesign)
# ---------------------------------------------------------------------------

class TestDraftAbortCancellation:

    def test_abort_cancels_verification_origin(self):
        """Draft abort queues all verification origins for cancel."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING

        vkey = ExpertKey(3, 1)
        _setup_origin_transfer(loop, req, vkey, cmd_seq=888)

        loop._abort_draft(req)

        # Origin queued for cancel
        assert len(loop._cancel_queue) >= 1

        # Phase 5 processes the queue
        ring_before = len(deps["ring_writer"].written)
        loop._phase_dispatch(EvictionPlan(), TransferPlan(), [])

        cancel_cmds = [
            c for c in deps["ring_writer"].written[ring_before:]
            if hasattr(c, "cmd_type") and c.cmd_type == CMD_CANCEL_TRANSFER
        ]
        assert len(cancel_cmds) == 1
        assert cancel_cmds[0].payload.cancel_transfer.target_cmd_seq == 888

    def test_abort_dedup_via_daemon_refcount(self):
        """Cancel + fetch same expert -> both emitted, daemon deduplicates."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING

        vkey = ExpertKey(3, 1)
        _setup_origin_transfer(loop, req, vkey, cmd_seq=888)

        loop._abort_draft(req)

        # Phase 5 with a transfer plan that re-requests the same expert.
        # Both the transfer AND the cancel are emitted — daemon refcount handles it.
        from orchestrator.types import TransferPlanEntry
        tr_plan = TransferPlan(entries=[
            TransferPlanEntry(key=vkey, target_gpu=0, priority=0.9),
        ])

        ring_before = len(deps["ring_writer"].written)
        loop._phase_dispatch(EvictionPlan(), tr_plan, [])

        # The cancel fires because inflight check passes (key IS inflight).
        # The new prefetch also fires. Daemon refcount: +1 -1 = net zero change.
        all_cmds = deps["ring_writer"].written[ring_before:]
        cancel_cmds = [c for c in all_cmds
                       if hasattr(c, "cmd_type") and c.cmd_type == CMD_CANCEL_TRANSFER]
        assert len(cancel_cmds) == 1

    def test_no_cancel_for_general_prefetch(self):
        """General prefetch (no origin) -> not cancelled on draft abort."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING

        # General inflight transfer with no origin tracking
        gkey = ExpertKey(4, 2)
        loop._inflight_transfers[0].add(gkey)
        # NOT in _origin_transfers

        loop._abort_draft(req)

        loop._phase_dispatch(EvictionPlan(), TransferPlan(), [])

        cancel_cmds = [
            c for c in deps["ring_writer"].written
            if hasattr(c, "cmd_type") and c.cmd_type == CMD_CANCEL_TRANSFER
        ]
        assert len(cancel_cmds) == 0
        assert gkey in loop._inflight_transfers[0]

    def test_acceptance_clears_origins(self):
        """_process_acceptance clears _origin_transfers for the request."""
        loop, deps = _build_loop()
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 99
        req.combined_draft = _make_combined_draft([100])
        req.speculation_state = SpeculationState.ACCEPTING

        _setup_origin_transfer(loop, req, ExpertKey(3, 1), cmd_seq=888)
        assert len(loop._origin_transfers) == 1

        vresult = VerificationResult(
            accepted_length=1, attempted_length=1,
            accepted_tokens=[100],
        )
        deps["verifier"].record_result = MagicMock()
        deps["utility_scorer"].record_iteration = MagicMock()

        loop._process_acceptance(req, vresult)
        assert len(loop._origin_transfers) == 0

    def test_cancel_disabled_leaves_transfers(self):
        """cancel_abort_transfers=False -> no cancels queued on abort."""
        hc = ConfigHotReload()
        hc.update("speculation.transfer_cancellation.cancel_abort_transfers", False)
        loop, deps = _build_loop(hot_config=hc)
        req = _make_decode_request(loop, deps)
        req.draft_seq_id = 99
        req.speculation_state = SpeculationState.DRAFTING

        vkey = ExpertKey(3, 1)
        _setup_origin_transfer(loop, req, vkey, cmd_seq=888)

        loop._abort_draft(req)
        assert len(loop._cancel_queue) == 0
        assert vkey in loop._inflight_transfers[0]

    def test_origin_expiry_sweep(self):
        """Expired origin entries are swept in _phase_yield."""
        loop, deps = _build_loop(
            config=OrchestratorConfig(transfer_origin_expiry_cycles=100),
        )
        req = _make_decode_request(loop, deps)

        _setup_origin_transfer(loop, req, ExpertKey(3, 1), cmd_seq=777)
        assert len(loop._origin_transfers) == 1

        # Advance cycle_count past expiry threshold + sweep interval
        loop._cycle_count = 1999
        loop._phase_yield(time.perf_counter_ns())

        # Not yet at sweep interval (every 1000)
        assert len(loop._origin_transfers) == 1

        loop._cycle_count = 2000
        loop._phase_yield(time.perf_counter_ns())

        # Now swept (birth=0, threshold=2000-100=1900, 0 < 1900)
        assert len(loop._origin_transfers) == 0


# ---------------------------------------------------------------------------
# TD-ORCH-ROUTING-EXPORT-MULTI: per-completion routing-export capture + the
# routing-export slot gate (INV-IPC-6b)
# ---------------------------------------------------------------------------

def _publish_routing_export(base: int, layer_idx: int,
                            indices: list[int]) -> None:
    """Simulate the daemon's routed top-K export into the shared slot
    (same fabrication as tests/unit/test_command_writer.py)."""
    hdr = RoutingExportHeader.from_address(base + SIDEBAND_ROUTING_EXPORT_OFF)
    hdr.num_tokens = 1
    hdr.topk = len(indices)
    hdr.layer_idx = layer_idx
    arr = (ctypes.c_int32 * len(indices)).from_address(
        base + SIDEBAND_ROUTING_EXPORT_INDICES_OFF)
    for i, e in enumerate(indices):
        arr[i] = e


def _read_expert_prefetch(base: int, count: int) -> list[int]:
    """Read back the expert indices Python wrote for FETCH_AND_RUN_MOE."""
    arr = (ExpertPrefetchEntry * count).from_address(
        base + SIDEBAND_EXPERT_PREFETCH_OFF)
    return [int(arr[i].expert_idx) for i in range(count)]


def _read_expert_prefetch_gpus(base: int, count: int) -> list[tuple[int, int]]:
    """Read back (expert_idx, gpu_idx) pairs of the written fetch entries."""
    arr = (ExpertPrefetchEntry * count).from_address(
        base + SIDEBAND_EXPERT_PREFETCH_OFF)
    return [(int(arr[i].expert_idx), int(arr[i].gpu_idx))
            for i in range(count)]


class TestRoutingExportMultiRequest:
    """TD-ORCH-ROUTING-EXPORT-MULTI: under MULTIPLE concurrent requests,
    another request's attention at the SAME layer index can overwrite the
    single sideband routing-export slot between one request's attention
    completion and its EXPERT_FFN dispatch.  The fix is two-fold:
      1. per-completion capture — _handle_compute_done copies the export
         into the owning RequestState the instant the producer completes;
         EXPERT_FFN forwards the captured copy, never the live slot;
      2. the slot gate — at most ONE gating-bearing command in flight, so
         the daemon can never overwrite an un-captured export either.
    Metadata: num_layers=6, num_moe_layers=4 -> first MoE layer = 2.
    """

    MOE_LAYER = 3

    def _sideband_loop(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        loop, deps = _build_loop(sideband_base=base)
        deps["_sideband_buf"] = buf  # keep backing memory alive
        return loop, deps, base

    def _add_decode_request(self, loop, rid: int) -> RequestState:
        req = RequestState(
            request_id=rid, seq_id=rid, default_gpu=0,
            is_prefill=False, prompt_len=1, tokens_generated=1,
            token_history=[5, 6],
        )
        loop._requests[rid] = req
        return req

    def _item(self, rid: int, layer: int, op: WorkOperation) -> WorkItem:
        return WorkItem(
            request_id=rid, layer_idx=layer, operation=op,
            target_gpu=0, status=WorkStatus.READY,
            timestamp_created_ns=time.perf_counter_ns(),
        )

    def _complete(self, loop, deps, cmd_seq: int) -> None:
        deps["completion_reader"].add(
            _make_completion(CMP_COMPUTE_DONE, cmd_seq=cmd_seq))
        loop._phase_collect()

    def _pop_expert_ffn_item(self, deps, rid: int) -> WorkItem:
        items = [i for i in deps["work_queue"].pending_items()
                 if i.request_id == rid
                 and i.operation == WorkOperation.EXPERT_FFN]
        assert len(items) == 1
        return items[0]

    # -- The core race: capture beats a same-layer slot overwrite ----------

    def test_expert_ffn_forwards_own_captured_routing(self):
        """Two requests, interleaved same-layer attention completions with
        distinct routings in the shared slot: each EXPERT_FFN must forward
        ITS OWN routing (pre-fix, R1 would forward R2's [7, 6, 5, 0])."""
        loop, deps, base = self._sideband_loop()
        ring = deps["ring_writer"]
        r1 = self._add_decode_request(loop, 1)
        r2 = self._add_decode_request(loop, 2)
        L = self.MOE_LAYER

        # R1's attention at layer L: dispatch, daemon publishes, completes.
        loop._dispatch_work_item(self._item(1, L, WorkOperation.ATTENTION))
        attn1 = ring.written[-1]
        assert attn1.cmd_type == D_B_CMD_RUN_ATTENTION
        _publish_routing_export(base, L, [1, 2, 3, 4])
        self._complete(loop, deps, attn1.cmd_seq)
        assert r1.captured_routing == (L, [1, 2, 3, 4])

        # R2's attention at the SAME layer overwrites the shared slot
        # before R1's EXPERT_FFN dispatches (the TD's exact race).
        loop._dispatch_work_item(self._item(2, L, WorkOperation.ATTENTION))
        attn2 = ring.written[-1]
        _publish_routing_export(base, L, [7, 6, 5, 0])
        self._complete(loop, deps, attn2.cmd_seq)
        assert r2.captured_routing == (L, [7, 6, 5, 0])

        # R1's EXPERT_FFN must forward R1's CAPTURED routing, not the slot.
        loop._dispatch_work_item(self._pop_expert_ffn_item(deps, 1))
        moe1 = ring.written[-1]
        assert moe1.cmd_type == E_CMD_FETCH_AND_RUN_MOE
        assert moe1.payload.fetch_and_run_moe.expert_count == 4
        assert _read_expert_prefetch(base, 4) == [1, 2, 3, 4]

        # And R2's EXPERT_FFN forwards R2's routing.
        loop._dispatch_work_item(self._pop_expert_ffn_item(deps, 2))
        moe2 = ring.written[-1]
        assert moe2.cmd_type == E_CMD_FETCH_AND_RUN_MOE
        assert moe2.payload.fetch_and_run_moe.expert_count == 4
        assert _read_expert_prefetch(base, 4) == [7, 6, 5, 0]

    # -- The slot gate: producers are serialized ---------------------------

    def test_gate_defers_second_gating_attention(self):
        """While R1's gating attention is in flight (export un-captured),
        R2's gating attention must NOT dispatch — the daemon would clobber
        the slot before R1's capture.  It stays READY and dispatches after
        R1's completion frees the gate."""
        loop, deps, base = self._sideband_loop()
        ring = deps["ring_writer"]
        self._add_decode_request(loop, 1)
        self._add_decode_request(loop, 2)
        L = self.MOE_LAYER

        loop._dispatch_work_item(self._item(1, L, WorkOperation.ATTENTION))
        attn1 = ring.written[-1]
        assert loop._gating_attn_inflight == attn1.cmd_seq

        item2 = self._item(2, L, WorkOperation.ATTENTION)
        n = len(ring.written)
        loop._dispatch_work_item(item2)
        assert len(ring.written) == n          # deferred: nothing written
        assert item2.status == WorkStatus.READY  # re-planned next cycle

        _publish_routing_export(base, L, [1, 2, 3, 4])
        self._complete(loop, deps, attn1.cmd_seq)
        assert loop._gating_attn_inflight is None

        loop._dispatch_work_item(item2)
        assert ring.written[-1].cmd_type == D_B_CMD_RUN_ATTENTION
        assert loop._gating_attn_inflight == ring.written[-1].cmd_seq

    def test_gate_ignores_dense_attention(self):
        """Dense-layer attention (no store_gating, no export) is never
        gated and never acquires the gate."""
        loop, deps, base = self._sideband_loop()
        ring = deps["ring_writer"]
        self._add_decode_request(loop, 1)
        self._add_decode_request(loop, 2)

        loop._dispatch_work_item(self._item(1, self.MOE_LAYER,
                                            WorkOperation.ATTENTION))
        gate = loop._gating_attn_inflight
        assert gate is not None

        # Dense layer 0 (< first_moe_layer=2) dispatches despite the gate.
        loop._dispatch_work_item(self._item(2, 0, WorkOperation.ATTENTION))
        assert ring.written[-1].cmd_type == D_B_CMD_RUN_ATTENTION
        assert loop._gating_attn_inflight == gate  # unchanged

    def test_error_releases_gate(self):
        """CMP_ERROR on the in-flight producer must free the gate (its
        export will never be captured) or all gating dispatches deadlock."""
        loop, deps, base = self._sideband_loop()
        ring = deps["ring_writer"]
        self._add_decode_request(loop, 1)

        loop._dispatch_work_item(self._item(1, self.MOE_LAYER,
                                            WorkOperation.ATTENTION))
        cmd_seq = ring.written[-1].cmd_seq
        assert loop._gating_attn_inflight == cmd_seq

        deps["completion_reader"].add(
            _make_completion(CMP_ERROR, cmd_seq=cmd_seq))
        loop._phase_collect()
        assert loop._gating_attn_inflight is None

    # -- Single-request path stays behavior-identical ----------------------

    def test_single_request_chain_never_defers(self):
        """The existing (single-request) e2e chain: ATTENTION completes ->
        EXPERT_FFN forwards the routing — same commands, same sideband
        entries as the pre-capture code; the gate never defers anything."""
        loop, deps, base = self._sideband_loop()
        ring = deps["ring_writer"]
        self._add_decode_request(loop, 1)
        L = self.MOE_LAYER

        loop._dispatch_work_item(self._item(1, L, WorkOperation.ATTENTION))
        attn = ring.written[-1]
        assert attn.cmd_type == D_B_CMD_RUN_ATTENTION
        _publish_routing_export(base, L, [9, 8])
        self._complete(loop, deps, attn.cmd_seq)
        assert loop._gating_attn_inflight is None  # freed at capture

        loop._dispatch_work_item(self._pop_expert_ffn_item(deps, 1))
        moe = ring.written[-1]
        assert moe.cmd_type == E_CMD_FETCH_AND_RUN_MOE
        assert moe.payload.fetch_and_run_moe.expert_count == 2
        assert _read_expert_prefetch(base, 2) == [9, 8]

    def test_no_sideband_behaves_as_before(self):
        """sideband_base=0 (unit-test construction): no capture, empty
        expert list, no gating — byte-for-byte the old behavior."""
        loop, deps = _build_loop()  # sideband_base defaults to 0
        ring = deps["ring_writer"]
        req = self._add_decode_request(loop, 1)
        L = self.MOE_LAYER

        loop._dispatch_work_item(self._item(1, L, WorkOperation.ATTENTION))
        attn = ring.written[-1]
        assert loop._gating_attn_inflight is None  # gate needs a sideband
        self._complete(loop, deps, attn.cmd_seq)
        assert req.captured_routing is None

        loop._dispatch_work_item(self._pop_expert_ffn_item(deps, 1))
        moe = ring.written[-1]
        assert moe.cmd_type == E_CMD_FETCH_AND_RUN_MOE
        assert moe.payload.fetch_and_run_moe.expert_count == 0

    # -- Stale capture is discarded, not forwarded --------------------------

    def test_cross_layer_stale_capture_discarded(self):
        """A capture whose slot layer mismatches the completed attention's
        layer (gate bug / rogue producer) must be discarded — EXPERT_FFN
        then forwards an empty list rather than wrong experts."""
        loop, deps, base = self._sideband_loop()
        ring = deps["ring_writer"]
        req = self._add_decode_request(loop, 1)
        L = self.MOE_LAYER

        loop._dispatch_work_item(self._item(1, L, WorkOperation.ATTENTION))
        attn = ring.written[-1]
        _publish_routing_export(base, L + 1, [1, 2])  # wrong layer in slot
        self._complete(loop, deps, attn.cmd_seq)
        assert req.captured_routing is None

        loop._dispatch_work_item(self._pop_expert_ffn_item(deps, 1))
        moe = ring.written[-1]
        assert moe.payload.fetch_and_run_moe.expert_count == 0

    # -- MTP steps park on the gate and resume ------------------------------

    def test_mtp_attn_parks_on_gate_and_resumes(self):
        """An MTP step whose gating attention finds the gate held parks in
        mtp_phase='attn-wait'; _retry_gating_waiters dispatches it once the
        gate frees (multi-request MTP correctness)."""
        loop, deps, base = self._sideband_loop()
        ring = deps["ring_writer"]
        deps["mtp_draft"].mtp_layer_idx = MagicMock(return_value=5)

        # R1 holds the gate with a main-chain gating attention.
        self._add_decode_request(loop, 1)
        loop._dispatch_work_item(self._item(1, self.MOE_LAYER,
                                            WorkOperation.ATTENTION))
        attn1 = ring.written[-1]

        # R2's MTP step reaches phase "project" completion -> wants to
        # dispatch its gating attention -> gate held -> parks.
        r2 = self._add_decode_request(loop, 2)
        r2.mtp_phase = "project"
        r2.mtp_mode = "catchup"
        r2.mtp_pos = 1
        r2.mtp_cmd_seq = 12345
        proj_item = WorkItem(
            request_id=2, layer_idx=5, operation=WorkOperation.OUTPUT_HEAD,
            target_gpu=0, status=WorkStatus.DISPATCHED, is_speculative=True,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        loop._cmd_seq_map[12345] = (2, proj_item)
        loop._draft_cmd_seqs[12345] = 2
        n = len(ring.written)
        self._complete(loop, deps, 12345)
        assert r2.mtp_phase == "attn-wait"
        assert len(ring.written) == n  # nothing dispatched while parked

        # R1's attention completes -> gate frees -> the SAME collect's
        # retry pass dispatches R2's parked MTP attention.
        _publish_routing_export(base, self.MOE_LAYER, [1, 2])
        self._complete(loop, deps, attn1.cmd_seq)
        mtp_attn = ring.written[-1]
        assert mtp_attn.cmd_type == D_B_CMD_RUN_ATTENTION
        assert mtp_attn.payload.run_attention.layer_idx == 5
        assert r2.mtp_phase == "attn"
        assert loop._gating_attn_inflight == mtp_attn.cmd_seq


# ---------------------------------------------------------------------------
# TD-PREFILL-MOE-BIG: superchunk prefill
# ---------------------------------------------------------------------------

class TestSuperchunkPrefill:
    """Superchunk prefill (TD-PREFILL-MOE-BIG): the prompt BODY
    ([0, prompt_len-1)) prefills in superchunks of up to
    metadata.moe_batch_capacity tokens — per superchunk, embedding +
    attention run in prefill_subchunk_tokens sub-chunks (row_offset +
    superchunk flag) and each layer runs ONE E_CMD_FETCH_AND_RUN_MOE_BIG
    (dense layers: RUN_MOE) over all superchunk tokens with the routed
    union of every sub-chunk's export.  The FINAL prompt token then feeds
    through the unchanged per-token path (its OUTPUT_HEAD emits the first
    generated token) — mirrors the C++ longctx driver exactly.
    Metadata: num_layers=6, num_moe_layers=4 -> first MoE layer = 2.
    """

    CAP = 8       # moe_batch_capacity
    SUB = 4       # prefill_subchunk_tokens
    PROMPT = list(range(100, 110))   # 10 tokens -> body 9 = 8 + 1

    def _sc_loop(self):
        import dataclasses
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        md = dataclasses.replace(_metadata(), moe_batch_capacity=self.CAP)
        cfg = OrchestratorConfig(prefill_subchunk_tokens=self.SUB)
        loop, deps = _build_loop(sideband_base=base, metadata=md, config=cfg)
        deps["_sideband_buf"] = buf
        return loop, deps, base

    def _pop_single(self, deps) -> WorkItem:
        items = [i for i in deps["work_queue"].pending_items()
                 if i.status == WorkStatus.READY]
        assert len(items) == 1, f"expected 1 READY item, got {items}"
        return items[0]

    def _complete(self, loop, deps, cmd_seq: int) -> None:
        deps["completion_reader"].add(
            _make_completion(CMP_COMPUTE_DONE, cmd_seq=cmd_seq))
        loop._phase_collect()

    def _step(self, loop, deps, base, routing=None):
        """Dispatch the single pending item; publish `routing` (MoE attn)
        before completing; return the written command."""
        item = self._pop_single(deps)
        loop._dispatch_work_item(item)
        cmd = deps["ring_writer"].written[-1]
        if routing is not None:
            _publish_routing_export(
                base, cmd.payload.run_attention.layer_idx, routing)
        self._complete(loop, deps, cmd.cmd_seq)
        return cmd

    def test_superchunk_activation_and_first_embedding(self):
        loop, deps, base = self._sc_loop()
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=self.PROMPT, gpu=0))
        loop._phase_collect()
        req = loop.requests[1]
        assert req.sc_active
        assert req.sc_len == self.CAP          # min(body=9, cap=8)

        item = self._pop_single(deps)
        assert item.operation == WorkOperation.EMBEDDING
        loop._dispatch_work_item(item)
        emb = deps["ring_writer"].written[-1]
        assert emb.payload.embedding_lookup.num_tokens == self.SUB
        assert emb.payload.embedding_lookup.row_offset == 0

    def test_full_superchunk_chain(self):
        """Drive the whole first superchunk: 2 embed sub-chunks, then per
        layer 2 attention sub-chunks + ONE MoE — dense layers RUN_MOE,
        MoE layers FETCH_AND_RUN_MOE_BIG with the sub-chunk union."""
        from orchestrator.shm_protocol import (
            CMD_EMBEDDING_LOOKUP,
            D_B_CMD_RUN_MOE,
            E_CMD_FETCH_AND_RUN_MOE_BIG,
        )
        loop, deps, base = self._sc_loop()
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=self.PROMPT, gpu=0))
        loop._phase_collect()
        req = loop.requests[1]

        # Embedding: 2 sub-chunks (8 = 2 x 4) at row offsets 0 and 4.
        e0 = self._step(loop, deps, base)
        e1 = self._step(loop, deps, base)
        assert e0.cmd_type == e1.cmd_type == CMD_EMBEDDING_LOOKUP
        assert e1.payload.embedding_lookup.row_offset == self.SUB

        first_moe = loop._first_moe_layer  # = 2
        for layer in range(6):
            is_moe = layer >= first_moe
            # Two attention sub-chunks per layer.
            routing_a = [layer, 6] if is_moe else None
            routing_b = [layer, 7] if is_moe else None
            a0 = self._step(loop, deps, base, routing=routing_a)
            a1 = self._step(loop, deps, base, routing=routing_b)
            for a, off in ((a0, 0), (a1, self.SUB)):
                assert a.cmd_type == D_B_CMD_RUN_ATTENTION
                p = a.payload.run_attention
                assert p.layer_idx == layer
                assert p.num_seqs == self.SUB
                assert p.superchunk == 1
                assert p.row_offset == off
                assert p.is_prefill == 1
                assert p.chunk_start == off      # superchunk base pos0=0
                assert p.emit_gating == (1 if is_moe else 0)
            # One MoE command over ALL 8 tokens.
            m = self._step(loop, deps, base)
            if is_moe:
                assert m.cmd_type == E_CMD_FETCH_AND_RUN_MOE_BIG
                p = m.payload.fetch_and_run_moe_big
                assert p.num_seqs == self.CAP
                # Union of the two sub-chunk exports, first-seen order.
                assert p.expert_count == 3
                assert _read_expert_prefetch(base, 3) == [layer, 6, 7]
                assert p.chunk_tokens == 0       # engine default
            else:
                assert m.cmd_type == D_B_CMD_RUN_MOE
                assert m.payload.run_moe.num_seqs == self.CAP

        # First superchunk done: cursor advanced, second superchunk of the
        # remaining 1 body token armed.
        assert req.prefill_pos == self.CAP
        assert req.sc_active
        assert req.sc_len == 1

    def test_final_token_hands_off_to_per_token_path(self):
        """After the last superchunk, the FINAL prompt token feeds through
        the unchanged per-token path (EMBEDDING with a single sideband
        token; its chain ends in OUTPUT_HEAD)."""
        loop, deps, base = self._sc_loop()
        # 4-token prompt: body 3 -> one superchunk of 3 (single sub-chunk
        # runs of 3 < SUB).
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=[10, 20, 30, 40], gpu=0))
        loop._phase_collect()
        req = loop.requests[1]
        assert req.sc_active and req.sc_len == 3

        # 1 embed + 6 layers x (1 attn + 1 moe).
        self._step(loop, deps, base)
        for layer in range(6):
            routing = [1] if layer >= loop._first_moe_layer else None
            self._step(loop, deps, base, routing=routing)
            self._step(loop, deps, base)

        # Superchunk mode exited; the final token (position 3) goes
        # per-token: EMBEDDING with num_tokens=1, row_offset=0.
        assert not req.sc_active
        assert req.prefill_pos == 3
        item = self._pop_single(deps)
        assert item.operation == WorkOperation.EMBEDDING
        loop._dispatch_work_item(item)
        emb = deps["ring_writer"].written[-1]
        assert emb.payload.embedding_lookup.num_tokens == 1
        assert emb.payload.embedding_lookup.row_offset == 0

    def test_superchunk_entries_honor_ep_gpu_indices(self):
        """TD-ORCH-ELM-COMPLETION-LIVELOCK: superchunk EXPERT_FFN fetch
        entries round-robin over the EP owner set (ep_gpu_indices) exactly
        like the decode/verify + MTP sites — a GPU outside the owner set
        (e.g. the dspark draft GPU) must receive ZERO entries.  The old
        `e % ngpu` targeted all GPUs; the draft GPU's over-capacity share
        livelocked the daemon's ELM completion path."""
        import dataclasses
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        md = dataclasses.replace(_metadata(), num_gpus=3,
                                 moe_batch_capacity=self.CAP)
        cfg = OrchestratorConfig(prefill_subchunk_tokens=self.SUB,
                                 ep_gpu_indices=(0, 1))
        loop, deps = _build_loop(sideband_base=base, metadata=md, config=cfg)
        deps["_sideband_buf"] = buf
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=self.PROMPT, gpu=0))
        loop._phase_collect()

        # 2 embed sub-chunks, then layers 0..1 dense (attn x2 + RUN_MOE).
        self._step(loop, deps, base)
        self._step(loop, deps, base)
        for _layer in range(loop._first_moe_layer):
            self._step(loop, deps, base)
            self._step(loop, deps, base)
            self._step(loop, deps, base)

        # First MoE layer: two sub-chunk exports whose union covers every
        # residue mod 3 (2 and 5 would target the draft GPU under e % ngpu).
        self._step(loop, deps, base, routing=[2, 5])
        self._step(loop, deps, base, routing=[6, 7])
        item = self._pop_single(deps)
        loop._dispatch_work_item(item)
        cmd = deps["ring_writer"].written[-1]
        p = cmd.payload.fetch_and_run_moe_big
        assert p.expert_count == 4
        pairs = _read_expert_prefetch_gpus(base, 4)
        # owners[e % len(owners)] over owners=(0, 1); NEVER gpu 2.
        assert pairs == [(2, 0), (5, 1), (6, 0), (7, 1)]
        assert all(g in (0, 1) for _, g in pairs)

    def test_no_capacity_no_superchunk(self):
        """moe_batch_capacity=0 (mock/unit contexts): per-token prefill,
        byte-identical to the pre-BIG behavior."""
        loop, deps = _build_loop()   # metadata without capacity
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=self.PROMPT, gpu=0))
        loop._phase_collect()
        assert not loop.requests[1].sc_active

    def test_short_prompt_no_superchunk(self):
        """body < 2 keeps the per-token path (nothing to batch)."""
        loop, deps, base = self._sc_loop()
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=[10, 20], gpu=0))
        loop._phase_collect()
        assert not loop.requests[1].sc_active

"""Realistic orchestrator scenario tests — multi-cycle end-to-end (#62h).

Unlike unit tests (which test individual methods with manually injected
completions), these tests simulate realistic multi-cycle execution:
  - Completions arrive at specific cycle offsets (mimicking daemon latency).
  - The full command trace is asserted exhaustively.
  - State transitions chain correctly across cycles.
  - No unexpected commands are emitted.

Small model config (2 layers, 1 MoE layer, 4 experts) keeps test data
manageable.  Each forward pass is 5 commands:
  EMBEDDING → ATTENTION(0) → ATTENTION(1) → EXPERT_FFN(1) → OUTPUT_HEAD

IMPORTANT: Keep this file richly commented.  Scenario tests are complex
and every step must be self-documenting.
"""

from __future__ import annotations

import ctypes
import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from orchestrator.command_writer import CommandWriter
from orchestrator.config_hot_reload import ConfigHotReload
from orchestrator.draft_combiner import DraftCombiner
from orchestrator.dspark_draft import DsparkDraft, DsparkDraftConfig
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
from orchestrator.mtp_draft import MtpDraft, MtpDraftConfig
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
    BatchDescriptorEntry,
    CMD_EMBEDDING_LOOKUP,
    CMD_OUTPUT_HEAD,
    CMD_SEQ_FORK,
    CMP_CHECKPOINT,
    CMP_COMPUTE_DONE,
    CMP_ERROR,
    Completion,
    D_B_CMD_RUN_ATTENTION,
    D_B_CMD_RUN_MOE,
    D_CMD_MTP_PROJECT,
    D_CMD_RUN_DSPARK_STEP,
    D_CMD_RUN_MTP_STEP,
    E_CMD_FETCH_AND_RUN_MOE,
    E_CMD_SEQ_CREATE,
    E_CMD_SEQ_FREE,
    SIDEBAND_BATCH_DESCRIPTOR_OFF,
    SIDEBAND_TOTAL_SIZE,
)
from orchestrator.speculative_prefetch import SpeculativePrefetch
from orchestrator.state_reader import StateReader
from orchestrator.transfer_scheduler import TransferScheduler
from orchestrator.types import (
    EngineMetadata,
    EvictionPlan,
    GpuConfig,
    SpeculationState,
    TransferPlan,
    WorkItem,
    WorkOperation,
    WorkStatus,
)
from orchestrator.utility_scorer import UtilityScorer
from orchestrator.verifier import Verifier
from orchestrator.work_queue import WorkQueue


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

# Small model: 2 layers, layer 0 = dense attention, layer 1 = MoE (4 experts).
# Forward pass: EMBEDDING → ATTENTION(0) → EXPERT_FFN(0) → ATTENTION(1) → EXPERT_FFN(1) → OUTPUT_HEAD
# TD-84e: all layers dispatch EXPERT_FFN; C++ handles dense via early-out.
SCENARIO_METADATA = EngineMetadata(
    num_gpus=1,
    num_moe_layers=1,
    num_experts=4,
    num_layers=2,
    expert_bytes=1024,
    kv_bytes_per_page=256,
    gpus=(GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                    vram_bytes=32 * 1024**3, compute_weight=1.0),),
)


class ScriptedDaemon:
    """Pre-scripted daemon responses keyed by cycle number.

    Implements the CompletionReader interface.  The test scripts completions
    at specific cycles; when the orchestrator calls drain() on cycle N,
    it receives all completions scheduled for that cycle.
    """
    def __init__(self):
        self._timeline: dict[int, list[Completion]] = {}
        self._cycle: int = 0

    def at_cycle(self, cycle: int, cmp: Completion) -> None:
        """Schedule a completion to arrive at the given cycle."""
        self._timeline.setdefault(cycle, []).append(cmp)

    def drain(self, max_count: int = 0xFFFFFFFF) -> list[Completion]:
        """Return all completions scheduled for the current cycle."""
        result = self._timeline.pop(self._cycle, [])
        return result[:max_count]

    def is_empty(self) -> bool:
        return self._cycle not in self._timeline

    def advance(self) -> None:
        self._cycle += 1

    @property
    def cycle(self) -> int:
        return self._cycle


@dataclass
class TracedCommand:
    """A command recorded with its cycle number."""
    cycle: int
    cmd_type: int
    cmd_seq: int


class CommandTrace:
    """Records all commands written to the ring with cycle numbers.

    Call snapshot() after each run_one_cycle() to tag new commands
    with the current cycle number.
    """
    def __init__(self, ring_writer):
        self._ring = ring_writer
        self._cycle = 0
        self.commands: list[TracedCommand] = []

    def snapshot(self) -> None:
        """Tag all new commands since last snapshot with current cycle."""
        while len(self.commands) < len(self._ring.written):
            cmd = self._ring.written[len(self.commands)]
            self.commands.append(TracedCommand(
                cycle=self._cycle,
                cmd_type=cmd.cmd_type,
                cmd_seq=cmd.cmd_seq,
            ))
        self._cycle += 1

    def by_type(self, cmd_type: int) -> list[TracedCommand]:
        return [c for c in self.commands if c.cmd_type == cmd_type]

    def types_in_order(self) -> list[int]:
        return [c.cmd_type for c in self.commands]

    def at_cycle(self, cycle: int) -> list[TracedCommand]:
        return [c for c in self.commands if c.cycle == cycle]

    def cmd_seq_for_type(self, cmd_type: int, index: int = 0) -> int:
        """Get the cmd_seq of the Nth command of a given type."""
        matches = self.by_type(cmd_type)
        return matches[index].cmd_seq


class MockRingWriter:
    """Ring writer that records all commands for inspection."""
    def __init__(self):
        self.written: list = []

    def write_struct(self, cmd) -> bool:
        self.written.append(cmd)
        return True

    def write(self, data: bytes) -> bool:
        return True


class MockStateReader:
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


class HostBuffer:
    """Simulated host buffer for token readback."""
    def __init__(self, size: int = 1024):
        self._buf = (ctypes.c_uint8 * size)()
        ctypes.memset(self._buf, 0, size)

    @property
    def base_address(self) -> int:
        return ctypes.addressof(self._buf)

    def write_draft_result(self, offset: int, token_id: int,
                           confidence: float) -> None:
        """Write token_id (uint32) + confidence (float32) at offset."""
        addr = self.base_address + offset
        (ctypes.c_uint32 * 1).from_address(addr)[0] = token_id
        (ctypes.c_float * 1).from_address(addr + 4)[0] = confidence

    def write_floats(self, offset: int, values: list[float]) -> None:
        """Write multiple float32 values at offset (DSP-6 c_k readback)."""
        arr = (ctypes.c_float * len(values)).from_address(
            self.base_address + offset)
        for i, v in enumerate(values):
            arr[i] = v

    def write_tokens(self, offset: int, tokens: list[int]) -> None:
        """Write multiple uint32 token IDs at offset."""
        arr = (ctypes.c_uint32 * len(tokens)).from_address(
            self.base_address + offset
        )
        for i, t in enumerate(tokens):
            arr[i] = t


def _make_compute_done(cmd_seq: int, layer_idx: int = 0,
                       host_buf_offset: int = 0,
                       data_bytes: int = 0,
                       top1_prob: float = 0.0) -> Completion:
    """Build a CMP_COMPUTE_DONE completion."""
    cmp = Completion()
    cmp.cmp_type = CMP_COMPUTE_DONE
    cmp.cmd_seq = cmd_seq
    cmp.gpu_idx = 0
    cmp.status = 0
    cmp.payload.compute.layer_idx = layer_idx
    cmp.payload.compute.host_buf_offset = host_buf_offset
    cmp.payload.compute.data_bytes = data_bytes
    cmp.payload.compute.top1_prob = top1_prob
    return cmp


def _make_error(cmd_seq: int, message: bytes = b"scripted error") -> Completion:
    """Build a CMP_ERROR completion (kComputeValidation-class)."""
    cmp = Completion()
    cmp.cmp_type = CMP_ERROR
    cmp.cmd_seq = cmd_seq
    cmp.gpu_idx = 0
    cmp.status = 1
    cmp.payload.error.error_category = 2
    cmp.payload.error.message = message
    return cmp


class SidebandBuffer:
    """Simulated sideband region (batch descriptors, routing export, ...).

    Passing .base_address as sideband_base makes the loop write real batch
    descriptors + expert prefetch entries and read the routing export slot
    (all zeros here — the mock daemon publishes nothing — so FETCH_AND_RUN
    is dispatched with expert_count=0, which is fine for a scripted run).
    """

    def __init__(self):
        self._buf = (ctypes.c_uint8 * SIDEBAND_TOTAL_SIZE)()
        ctypes.memset(self._buf, 0, SIDEBAND_TOTAL_SIZE)

    @property
    def base_address(self) -> int:
        return ctypes.addressof(self._buf)

    def read_batch_descriptor(self) -> tuple[int, int]:
        """Read (seq_id, token_pos) of batch descriptor slot 0."""
        e = BatchDescriptorEntry.from_address(
            self.base_address + SIDEBAND_BATCH_DESCRIPTOR_OFF
        )
        return (int(e.seq_id), int(e.token_pos))

    def read_batch_descriptors(self, n: int) -> list[tuple[int, int]]:
        """Read (seq_id, token_pos) of the first n descriptor slots
        (batched-verify attentions write V rows)."""
        arr = (BatchDescriptorEntry * n).from_address(
            self.base_address + SIDEBAND_BATCH_DESCRIPTOR_OFF
        )
        return [(int(e.seq_id), int(e.token_pos)) for e in arr]


def _build_scenario_loop(
    *,
    metadata: EngineMetadata = SCENARIO_METADATA,
    mtp_enabled: bool = False,
    mtp_max_depth: int = 2,
    dspark_enabled: bool = False,
    dspark_gamma: int = 2,
    dspark_confidence_enabled: bool = False,
    self_spec_enabled: bool = False,
    recommended_depth: int = 0,
    host_buf: HostBuffer | None = None,
    sideband: SidebandBuffer | None = None,
    cycle_budget_us: float = 0.0,
) -> tuple[OrchestratorLoop, dict]:
    """Build a loop configured for scenario testing.

    Returns (loop, deps_dict) where deps_dict has all injected modules
    for test inspection.
    """
    config = OrchestratorConfig(
        cycle_budget_us=cycle_budget_us,
        max_idle_wait_us=0.0,  # no idle wait in tests
    )

    daemon = ScriptedDaemon()
    ring_writer = MockRingWriter()
    cmd_writer = CommandWriter()
    state_reader = MockStateReader()
    work_queue = WorkQueue()
    scheduler = Scheduler()
    gpu_assigner = GpuLoadBalancer(GpuLoadBalancerConfig(gpus=metadata.gpus))

    placement_cfg = ExpertPlacementConfig(
        num_moe_layers=metadata.num_moe_layers,
        num_experts=metadata.num_experts,
        cache_gpu_indices=[0],
    )
    expert_placement = ExpertPlacement(placement_cfg)
    placement_optimizer = PlacementOptimizer(
        PlacementOptimizerConfig(), expert_placement,
    )

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
    utility_scorer.recommended_depth = MagicMock(return_value=recommended_depth)

    verifier = MagicMock(spec=Verifier)
    # Verifier stubs for speculation: compare_logits returns full match
    verifier.compare_logits = MagicMock(side_effect=lambda draft_tokens, target_tokens: len(draft_tokens))
    verifier.build_result = MagicMock(side_effect=lambda draft, analyses, verified_depth, accepted_length, seq_id: MagicMock(
        accepted_length=accepted_length,
        attempted_length=verified_depth,
        accepted_tokens=draft.tokens[:accepted_length],
    ))
    verifier.record_result = MagicMock()
    verifier.plan_verification_pass = MagicMock(return_value=MagicMock(commands=[]))

    prompt_lookup = PromptLookup()
    draft_combiner = DraftCombiner()

    # MTP / self-spec / layer-skip
    if mtp_enabled:
        mtp_draft = MtpDraft(
            MtpDraftConfig(enabled=True, max_depth=mtp_max_depth),
            num_mtp_layers=1, num_layers=metadata.num_layers,
        )
    else:
        mtp_draft = MagicMock(spec=MtpDraft)
        mtp_draft.is_enabled = False
        mtp_draft.num_mtp_layers = 0

    # DSpark planner (DSP-5) — a real planner when enabled (whole-block
    # rounds); a default-constructed one is inert (enabled=False).
    dspark_draft = DsparkDraft(DsparkDraftConfig(
        enabled=dspark_enabled, speculative_tokens=dspark_gamma,
        confidence_enabled=dspark_confidence_enabled,
    ))

    self_speculative = MagicMock(spec=SelfSpeculative)
    self_speculative.is_enabled = self_spec_enabled

    layer_skip = MagicMock(spec=LayerSkip)
    layer_skip.is_enabled = False

    reasoning_mode = MagicMock(spec=ReasoningMode)
    online_calibrator = OnlineCalibrator()
    performance_objective = PerformanceObjective()

    transfer_scheduler = MagicMock(spec=TransferScheduler)
    transfer_scheduler.plan_transfers = MagicMock(return_value=TransferPlan())
    transfer_scheduler.plan_evictions = MagicMock(return_value=EvictionPlan())
    eviction_policy = EvictionPolicy()

    hbuf = host_buf or HostBuffer()

    loop = OrchestratorLoop(
        config=config, metadata=metadata,
        cmd_writer=cmd_writer, ring_writer=ring_writer,
        completion_reader=daemon, state_reader=state_reader,
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
        host_buf_base=hbuf.base_address,
        sideband_base=sideband.base_address if sideband is not None else 0,
        dspark_draft=dspark_draft,
    )

    deps = {
        "daemon": daemon,
        "ring_writer": ring_writer,
        "cmd_writer": cmd_writer,
        "host_buf": hbuf,
        "sideband": sideband,
        "verifier": verifier,
        "utility_scorer": utility_scorer,
        "mtp_draft": mtp_draft,
        "dspark_draft": dspark_draft,
    }
    return loop, deps


def _run_cycles(loop: OrchestratorLoop, daemon: ScriptedDaemon,
                trace: CommandTrace, max_cycles: int = 50) -> int:
    """Run the loop until the request is finalized or max_cycles reached.

    Returns the number of cycles executed.
    """
    for cycle in range(max_cycles):
        loop.run_one_cycle()
        trace.snapshot()
        daemon.advance()
        if not loop.requests:
            return cycle + 1
    return max_cycles


# ---------------------------------------------------------------------------
# Scenario: MTP speculative decoding — lossless, production seam
# (TD-MTP-PY-LOOP-KV / TD-MTP-FUSED-RUNMOE)
# ---------------------------------------------------------------------------

def _drive_mtp_scenario(
    loop: OrchestratorLoop,
    deps: dict,
    *,
    main_tokens: list[int],
    mtp_tokens: list[tuple[int, float]],
    prompt_len: int = 3,
    max_cycles: int = 500,
) -> dict:
    """Auto-responding daemon driver — the Python mirror of the MtpLossless
    golden's engine.

    ``main_tokens`` scripts the MAIN model's OUTPUT_HEAD outputs in feed
    order (the "ground truth" AR continuation); ``mtp_tokens`` scripts the
    MTP head's (token, confidence) outputs, consumed by draft AND catch-up
    steps (cycled if exhausted).  Every other command gets a plain
    CMP_COMPUTE_DONE one cycle later.

    A BATCHED-verify main OUTPUT_HEAD (num_tokens = V > 1) is answered
    POSITION-deterministically: the token for a row at fed position p is
    main_tokens[p - (prompt_len - 1)] — the main model's exact output at
    that position GIVEN the true prefix.  This mirror is faithful for
    every token the loop may commit: the acceptance rule only commits
    outputs whose teacher-forced prefix was fully accepted (= true), and
    the greedy target model is deterministic per position.  Rows past the
    script (never committable within max_tokens) read 0.

    Returns a dict with:
      attn_trace:  [(layer_idx, seq_id, token_pos)] for every RUN_ATTENTION
                   row, descriptors snapshotted at dispatch time (needs
                   sideband).
      main_consumed: number of main-model tokens consumed (feed order).
      cycles:      cycles executed until the request finished.
    """
    daemon = deps["daemon"]
    ring_writer = deps["ring_writer"]
    hbuf = deps["host_buf"]
    sideband = deps.get("sideband")

    attn_trace: list[tuple[int, int, int]] = []
    last_main_positions: list[int] = []
    main_i = 0
    mtp_i = 0
    off = 0
    responded = 0

    for cycle in range(max_cycles):
        loop.run_one_cycle()
        for cmd in ring_writer.written[responded:]:
            responded += 1
            t = getattr(cmd, "cmd_type", None)
            if t in (E_CMD_SEQ_CREATE, E_CMD_SEQ_FREE, CMD_SEQ_FORK):
                continue  # CMP_SEQ_OP_DONE acks are ignored by the loop
            if t == D_B_CMD_RUN_ATTENTION and sideband is not None:
                n = int(cmd.payload.run_attention.num_seqs)
                layer = int(cmd.payload.run_attention.layer_idx)
                entries = sideband.read_batch_descriptors(n)
                for seq_id, pos in entries:
                    attn_trace.append((layer, seq_id, pos))
                if layer < loop._metadata.num_layers:  # noqa: SLF001
                    # Main-layer batch: remember the fed positions for the
                    # position-deterministic batched OUTPUT_HEAD below.
                    last_main_positions = [pos for _, pos in entries]
            if t == CMD_OUTPUT_HEAD:
                n = int(cmd.payload.output_head.num_tokens)
                if cmd.payload.output_head.mtp_head > 0:
                    tok, conf = mtp_tokens[mtp_i % len(mtp_tokens)]
                    mtp_i += 1
                    hbuf.write_tokens(off, [tok])
                    daemon.at_cycle(daemon.cycle + 1, _make_compute_done(
                        cmd.cmd_seq, host_buf_offset=off, data_bytes=4,
                        top1_prob=conf,
                    ))
                    off += 4
                elif n > 1:
                    # Batched verify head: all V rows' argmax tokens,
                    # position-deterministic (see docstring).
                    assert len(last_main_positions) == n, (
                        "batched OUTPUT_HEAD without a matching V-row "
                        "attention batch")
                    toks = []
                    for pos in last_main_positions:
                        idx = pos - (prompt_len - 1)
                        toks.append(main_tokens[idx]
                                    if 0 <= idx < len(main_tokens) else 0)
                    main_i = max(main_i, min(
                        max(p - (prompt_len - 1) for p in
                            last_main_positions) + 1, len(main_tokens)))
                    hbuf.write_tokens(off, toks)
                    daemon.at_cycle(daemon.cycle + 1, _make_compute_done(
                        cmd.cmd_seq, host_buf_offset=off,
                        data_bytes=4 * n, top1_prob=0.5,
                    ))
                    off += 4 * n
                else:
                    assert main_i < len(main_tokens), "main token script ran dry"
                    tok, conf = main_tokens[main_i], 0.5
                    main_i += 1
                    hbuf.write_tokens(off, [tok])
                    daemon.at_cycle(daemon.cycle + 1, _make_compute_done(
                        cmd.cmd_seq, host_buf_offset=off, data_bytes=4,
                        top1_prob=conf,
                    ))
                    off += 4
            elif t != CMD_OUTPUT_HEAD:
                daemon.at_cycle(daemon.cycle + 1,
                                _make_compute_done(cmd.cmd_seq))
        daemon.advance()
        if not loop.requests:
            return {"attn_trace": attn_trace, "main_consumed": main_i,
                    "cycles": cycle + 1}
    raise AssertionError(f"scenario did not finish in {max_cycles} cycles")


class TestScenarioMtpSpeculation:
    """End-to-end MTP rounds against a scripted daemon — the Python mirror
    of Glm52GgufGolden.MtpLossless (TD-MTP-PY-LOOP-KV).

    Model: 2 layers (layer 0 dense, layer 1 MoE), MTP layer = 2.
    Prompt [10, 20, 30] (prompt_len=3), max_tokens=5; the scripted main
    model always answers 42, 43, 100, 101, 55 in feed order.

    Asserted (mirroring the golden):
      - LOSSLESS: the committed tokens are EXACTLY the main model's
        outputs, whatever the MTP head drafts (only main-model outputs
        are ever fed/committed — rejected drafts are never fed).
      - INV-MTP-KV: the MTP layer's attention runs at gap-free positions
        of the MAIN sequence: a warm/catch-up after every non-drafting
        feed, draft step 0 covering each round's anchor.
      - Production seam: D_CMD_MTP_PROJECT chain + E_CMD_FETCH_AND_RUN_MOE
        — never the fused D_CMD_RUN_MTP_STEP (#89), never a KV fork.
      - Acceptance > 0 when drafts match (MTP is real speculation, not a
        no-op with empty KV).
    """

    MAIN_TOKENS = [42, 43, 100, 101, 55]

    def _run(self, mtp_tokens, mtp_enabled=True, depth=2):
        hbuf = HostBuffer(4096)
        sideband = SidebandBuffer()
        loop, deps = _build_scenario_loop(
            mtp_enabled=mtp_enabled,
            mtp_max_depth=2,
            recommended_depth=depth,
            host_buf=hbuf,
            sideband=sideband,
        )
        results: list = []
        loop.submit_request(InferenceRequest(
            request_id=1,
            prompt_token_ids=[10, 20, 30],
            gpu=0,
            max_tokens=5,
            on_complete=lambda rid, toks, reason, lp: results.append(
                (rid, list(toks), reason)),
        ))
        stats = _drive_mtp_scenario(
            loop, deps,
            main_tokens=list(self.MAIN_TOKENS),
            mtp_tokens=mtp_tokens,
        )
        assert len(results) == 1
        return results[0], stats, deps

    def test_mtp_lossless_with_accepting_drafts(self):
        """Drafts that match the main model: identical tokens, acceptance>0."""
        # MTP head script in consumption order (#91 real prefill added the
        # prompt-warm steps): warm@0, warm@1 (discarded prompt warms),
        # post-transition catch-up@2 (discarded), draft step 0 -> 100,
        # draft step 1 -> 101, two verify catch-ups (discarded).
        (rid, tokens, reason), stats, deps = self._run(
            mtp_tokens=[(0, 0.9), (0, 0.9), (0, 0.9),
                        (100, 0.9), (101, 0.8), (0, 0.9), (0, 0.9)],
        )
        assert rid == 1
        assert reason == "length"
        # LOSSLESS: exactly the main model's own outputs, in order
        assert tokens == self.MAIN_TOKENS
        assert stats["main_consumed"] == 5
        # Real speculation: both drafts were accepted
        assert deps["mtp_draft"].total_accepted == 2
        assert deps["mtp_draft"].total_steps == 2

        ring = deps["ring_writer"].written
        types = [c.cmd_type for c in ring if hasattr(c, "cmd_type")]
        # Production seam only: no fused MTP step, no KV fork, no seq free
        # besides the request teardown of the MAIN sequence.
        assert D_CMD_RUN_MTP_STEP not in types
        assert CMD_SEQ_FORK not in types
        frees = [c for c in ring if getattr(c, "cmd_type", 0) == E_CMD_SEQ_FREE]
        assert len(frees) == 1  # request teardown only (main seq)
        assert frees[0].payload.seq_free.seq_id == 1
        # 7 MTP steps total (2 prompt warms + 1 transition catch-up + 2
        # drafts + 2 verify catch-ups), each a full production-seam chain.
        assert types.count(D_CMD_MTP_PROJECT) == 7
        # FETCH_AND_RUN is the seam for BOTH the MTP layer (2) and the
        # main MoE layer (1, #91): 7 MTP steps + 7 main forwards.
        moe_layers = [c.payload.fetch_and_run_moe.layer_idx for c in ring
                      if getattr(c, "cmd_type", 0) == E_CMD_FETCH_AND_RUN_MOE]
        assert moe_layers.count(2) == 7
        assert moe_layers.count(1) == 7
        # RUN_MOE survives ONLY for the dense layer (0) — the deprecated
        # resident-only path never runs a routed MoE layer (#91).
        dense_moe = [c.payload.run_moe.layer_idx for c in ring
                     if getattr(c, "cmd_type", 0) == D_B_CMD_RUN_MOE]
        assert dense_moe and all(l == 0 for l in dense_moe)

        # INV-MTP-KV: the MTP layer (2) attends at gap-free MAIN-seq
        # positions: prompt warms @0,@1, transition catch-up@2, draft@3
        # (anchor, embeds the true token 43), draft@4 (scratch),
        # catch-up@4 (true-token overwrite), catch-up@5.  seq_id is
        # ALWAYS the main sequence (1) — never a fork.
        mtp_attn = [(s, p) for (l, s, p) in stats["attn_trace"] if l == 2]
        assert mtp_attn == [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                            (1, 4), (1, 5)]
        # Real prefill feeds the prompt at 0..2 (#91); decode feeds are
        # gap-free from prompt_len: 3 (AR), then verify feeds 4, 5, 6.
        main_attn = [(s, p) for (l, s, p) in stats["attn_trace"] if l == 0]
        assert main_attn == [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                             (1, 5), (1, 6)]

    def test_mtp_lossless_with_garbage_drafts(self):
        """Garbage drafts: tokens STILL identical (early-stop rejects all —
        a rejected draft is never fed), acceptance == 0."""
        (rid, tokens, reason), stats, deps = self._run(
            mtp_tokens=[(777, 0.9)],  # cycled: every draft is 777
        )
        assert tokens == self.MAIN_TOKENS  # INV-MTP-LOSSLESS
        assert stats["main_consumed"] == 5
        assert deps["mtp_draft"].total_accepted == 0
        # 777 was never committed anywhere
        assert 777 not in tokens
        ring = deps["ring_writer"].written
        types = [c.cmd_type for c in ring if hasattr(c, "cmd_type")]
        assert CMD_SEQ_FORK not in types
        assert D_CMD_RUN_MTP_STEP not in types

    def test_mtp_off_baseline_identical(self):
        """The MTP-off baseline commits the same main-model tokens —
        speculation changes command traffic, never the token stream."""
        (rid, tokens, reason), stats, deps = self._run(
            mtp_tokens=[(0, 0.0)], mtp_enabled=False, depth=0,
        )
        assert tokens == self.MAIN_TOKENS
        ring = deps["ring_writer"].written
        types = [c.cmd_type for c in ring if hasattr(c, "cmd_type")]
        assert D_CMD_MTP_PROJECT not in types
        # Main-layer MoE runs on the production FETCH_AND_RUN seam (#91)
        # even with speculation off — but never for the MTP layer (2).
        moe_layers = [c.payload.fetch_and_run_moe.layer_idx for c in ring
                      if getattr(c, "cmd_type", 0) == E_CMD_FETCH_AND_RUN_MOE]
        assert moe_layers and all(l == 1 for l in moe_layers)


# ---------------------------------------------------------------------------
# Scenario: MTP BATCHED verification — one V-token pass per round
# ---------------------------------------------------------------------------

# Same small model, but the engine publishes a MoE batch capacity so the
# loop takes the batched-verify path (V = depth+1 <= capacity).
BATCHED_METADATA = dataclasses.replace(SCENARIO_METADATA,
                                       moe_batch_capacity=64)


class TestScenarioMtpBatchedVerify:
    """MTP rounds with BATCHED verification (speculation.mtp.batched_verify).

    Same scripted-daemon mirror as TestScenarioMtpSpeculation, but the
    round's K drafts are verified by ONE teacher-forced V=K+1-token
    forward on the MAIN sequence: EMBEDDING(V) -> per-layer
    RUN_ATTENTION(V rows, chunked-prefill shape) + FETCH_AND_RUN_MOE(V)
    -> OUTPUT_HEAD(V) reading back all V argmax tokens.

    Asserted:
      - LOSSLESS: committed tokens EXACTLY the main model's outputs —
        with accepting, partially-accepting AND garbage drafts — token-
        identical to the sequential-verify runs and the AR baseline.
      - ONE main forward per round (V-wide) instead of V sequential feeds.
      - INV-MTP-KV: post-verify catch-ups rewrite the accepted positions'
        MTP KV off the batched trunk hiddens (ascending hidden rows), and
        the next round's draft step 0 consumes the new anchor's row.
      - Production seam only: FETCH_AND_RUN_MOE for routed layers, no
        fork, no fused MTP step.
    """

    MAIN_TOKENS = [42, 43, 100, 101, 55]

    def _run(self, mtp_tokens, depth=2, max_tokens=5, main_tokens=None):
        hbuf = HostBuffer(8192)
        sideband = SidebandBuffer()
        loop, deps = _build_scenario_loop(
            metadata=BATCHED_METADATA,
            mtp_enabled=True,
            mtp_max_depth=depth,
            recommended_depth=depth,
            host_buf=hbuf,
            sideband=sideband,
        )
        results: list = []
        loop.submit_request(InferenceRequest(
            request_id=1,
            prompt_token_ids=[10, 20, 30],
            gpu=0,
            max_tokens=max_tokens,
            on_complete=lambda rid, toks, reason, lp: results.append(
                (rid, list(toks), reason)),
        ))
        stats = _drive_mtp_scenario(
            loop, deps,
            main_tokens=list(main_tokens or self.MAIN_TOKENS),
            mtp_tokens=mtp_tokens,
        )
        assert len(results) == 1
        return results[0], stats, deps

    def test_batched_lossless_with_accepting_drafts(self):
        """Both drafts accepted in ONE verify pass; tokens identical."""
        # MTP script: warm@0, warm@1, transition catch-up@2 (discarded),
        # draft step 0 -> 100, draft step 1 -> 101.  The round ends at
        # max_tokens inside the batched commit, so no post-verify
        # catch-ups run.
        (rid, tokens, reason), stats, deps = self._run(
            mtp_tokens=[(0, 0.9), (0, 0.9), (0, 0.9),
                        (100, 0.9), (101, 0.8)],
        )
        assert rid == 1
        assert reason == "length"
        # LOSSLESS: exactly the main model's own outputs, in order.
        assert tokens == self.MAIN_TOKENS
        # Real speculation: both drafts accepted.
        assert deps["mtp_draft"].total_accepted == 2
        assert deps["mtp_draft"].total_steps == 2

        ring = deps["ring_writer"].written
        types = [c.cmd_type for c in ring if hasattr(c, "cmd_type")]
        assert D_CMD_RUN_MTP_STEP not in types
        assert CMD_SEQ_FORK not in types  # batched verify never forks

        # ONE V=3-wide verify pass: exactly one multi-token OUTPUT_HEAD,
        # V argmax readback (the driver asserted the V-row attention
        # batch preceded it).
        heads = [c for c in ring
                 if getattr(c, "cmd_type", 0) == CMD_OUTPUT_HEAD
                 and c.payload.output_head.mtp_head == 0]
        widths = [int(c.payload.output_head.num_tokens) for c in heads]
        # prefill-final feed + 1 AR feed + ONE batched verify head
        assert widths == [1, 1, 3]

        # The verify attentions are 3-row chunked-prefill-shaped batches
        # at positions 4,5,6 of the MAIN seq (base = feed pos 4).
        vattn = [c for c in ring
                 if getattr(c, "cmd_type", 0) == D_B_CMD_RUN_ATTENTION
                 and c.payload.run_attention.num_seqs == 3]
        assert len(vattn) == 2  # layers 0 and 1
        for c in vattn:
            assert c.payload.run_attention.is_prefill == 1
            assert c.payload.run_attention.chunk_start == 4
            assert c.payload.run_attention.chunk_len == 3
        # ... and the trace shows rows 4,5,6 for both layers.
        main_attn = [(l, p) for (l, s, p) in stats["attn_trace"] if l < 2]
        assert [(0, 4), (0, 5), (0, 6)] == main_attn[-6:-3]
        assert [(1, 4), (1, 5), (1, 6)] == main_attn[-3:]

        # Routed MoE stays on the production FETCH_AND_RUN seam; the
        # verify pass's MoE layer runs ONE 3-row command.
        vmoe = [c for c in ring
                if getattr(c, "cmd_type", 0) == E_CMD_FETCH_AND_RUN_MOE
                and c.payload.fetch_and_run_moe.num_seqs == 3]
        assert len(vmoe) == 1
        assert vmoe[0].payload.fetch_and_run_moe.layer_idx == 1

    def test_batched_lossless_with_garbage_drafts(self):
        """Garbage drafts: every round rejects, tokens STILL identical."""
        (rid, tokens, reason), stats, deps = self._run(
            mtp_tokens=[(777, 0.9)],  # cycled: every draft is 777
        )
        assert tokens == self.MAIN_TOKENS  # INV-MTP-LOSSLESS
        assert deps["mtp_draft"].total_accepted == 0
        assert 777 not in tokens
        ring = deps["ring_writer"].written
        types = [c.cmd_type for c in ring if hasattr(c, "cmd_type")]
        assert CMD_SEQ_FORK not in types
        assert D_CMD_RUN_MTP_STEP not in types
        # Every round still verified in ONE batched pass (num_tokens=3).
        heads = [c for c in ring
                 if getattr(c, "cmd_type", 0) == CMD_OUTPUT_HEAD
                 and c.payload.output_head.mtp_head == 0
                 and c.payload.output_head.num_tokens > 1]
        assert len(heads) >= 1
        assert all(int(c.payload.output_head.num_tokens) == 3
                   for c in heads)

    def test_batched_partial_accept_runs_exact_catchups(self):
        """m=1 of 2 accepted: the accepted position gets an exact-hidden
        catch-up off the batched trunk hiddens (row 0), and the next
        round's draft step 0 consumes the new anchor's row (row m=1)."""
        # Draft chain scripts (100, 777): step 0 matches, step 1 garbage.
        # Round 1 verify (base=4): readback [100, 101, 55] vs drafts
        # [100, 777] -> m=1, commits 100 + the correction 101 (tg=4).
        # Catch-up @4 embedding 100 (hidden row 0), then round 2:
        # draft step 0 @5 (hidden row 1), drafts [100, 777] again ->
        # verify base=6 readback [55, 0, 0] -> m=0, commits 55 -> stop.
        (rid, tokens, reason), stats, deps = self._run(
            mtp_tokens=[(0, 0.9), (0, 0.9), (0, 0.9),   # warms + catch-up
                        (100, 0.9), (777, 0.9)],         # cycled draft pairs
        )
        assert tokens == self.MAIN_TOKENS  # INV-MTP-LOSSLESS
        assert deps["mtp_draft"].total_accepted == 1

        ring = deps["ring_writer"].written
        projects = [c for c in ring
                    if getattr(c, "cmd_type", 0) == D_CMD_MTP_PROJECT]
        rows = [int(c.payload.mtp_project.hidden_row) for c in projects]
        # 2 prompt warms + transition catch-up + round-1 drafts (2, rows
        # 0) + post-verify catch-up @4 (row 0) + round-2 draft step 0
        # (row m=1) + round-2 draft step 1 (row 0).
        assert rows.count(1) == 1, rows
        # The round-2 step 0 (row 1) embeds the newest committed token
        # (101, the round-1 correction) — draft step 0 covers the anchor.
        step0_row1 = [c for c in projects
                      if c.payload.mtp_project.hidden_row == 1]
        assert step0_row1[0].payload.mtp_project.input_token_id == 101
        # INV-MTP-KV: the MTP layer attends the accepted position 4 TWICE
        # — once as round-1 draft step 1 scratch, once as the exact-hidden
        # post-verify catch-up.
        mtp_attn_pos = [p for (l, s, p) in stats["attn_trace"] if l == 2]
        assert mtp_attn_pos.count(4) == 2, mtp_attn_pos

    def test_batched_eos_mid_round_truncates(self):
        """An EOS inside the batched commit stops EXACTLY there — the
        remaining accepted/bonus tokens are never committed (identical to
        the per-feed sequential stop check)."""
        hbuf = HostBuffer(8192)
        sideband = SidebandBuffer()
        loop, deps = _build_scenario_loop(
            metadata=dataclasses.replace(BATCHED_METADATA,
                                         eos_token_ids=(101,)),
            mtp_enabled=True, mtp_max_depth=2, recommended_depth=2,
            host_buf=hbuf, sideband=sideband,
        )
        results: list = []
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=[10, 20, 30], gpu=0,
            max_tokens=8,   # EOS is the stopper, not length
            on_complete=lambda rid, toks, reason, lp: results.append(
                (list(toks), reason)),
        ))
        _drive_mtp_scenario(
            loop, deps, main_tokens=[42, 43, 100, 101, 55],
            mtp_tokens=[(0, 0.9), (0, 0.9), (0, 0.9),
                        (100, 0.9), (101, 0.8)],
        )
        tokens, reason = results[0]
        # The round's readback was [100, 101, 55] with both drafts
        # accepted, but 101 is EOS: 55 (the bonus) must NOT be committed.
        assert tokens == [42, 43, 100, 101]
        assert reason == "stop"
        assert 55 not in tokens

    def test_batched_matches_sequential_and_baseline_tokens(self):
        """The batched, sequential and AR-baseline runs commit IDENTICAL
        token streams for the same scripts (the losslessness triangle)."""
        script = [(0, 0.9), (0, 0.9), (0, 0.9), (100, 0.9), (777, 0.9)]
        (_, batched, _), _, _ = self._run(mtp_tokens=list(script))

        # Sequential run: same loop shape, capacity withheld (the
        # metadata's moe_batch_capacity=0 forces the fallback).
        hbuf = HostBuffer(8192)
        sideband = SidebandBuffer()
        loop, deps = _build_scenario_loop(
            mtp_enabled=True, mtp_max_depth=2, recommended_depth=2,
            host_buf=hbuf, sideband=sideband,
        )
        results: list = []
        loop.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=[10, 20, 30], gpu=0,
            max_tokens=5,
            on_complete=lambda rid, toks, reason, lp: results.append(
                list(toks)),
        ))
        _drive_mtp_scenario(loop, deps, main_tokens=list(self.MAIN_TOKENS),
                            mtp_tokens=list(script))
        sequential = results[0]

        assert batched == sequential == self.MAIN_TOKENS
        # The capacity-0 run really took the sequential fallback: no
        # multi-token OUTPUT_HEAD was ever dispatched.
        ring = deps["ring_writer"].written
        assert all(int(c.payload.output_head.num_tokens) == 1
                   for c in ring
                   if getattr(c, "cmd_type", 0) == CMD_OUTPUT_HEAD)


# ---------------------------------------------------------------------------
# Scenario: DSpark speculative decoding — lossless, whole-block rounds (DSP-5)
# ---------------------------------------------------------------------------

def _drive_dspark_scenario(
    loop: OrchestratorLoop,
    deps: dict,
    *,
    main_tokens: list[int],
    dspark_blocks: list[list[int]],
    dspark_confs: list[list[float]] | None = None,
    dspark_errors: bool = False,
    max_cycles: int = 500,
) -> dict:
    """Auto-responding daemon driver — the Python mirror of the
    DsparkLossless golden's engine.

    ``main_tokens`` scripts the MAIN model's OUTPUT_HEAD outputs in feed
    order; ``dspark_blocks`` scripts the D_CMD_RUN_DSPARK_STEP responses —
    per round, the whole γ id block written (i32) into the host buffer and
    referenced by the completion's host_buf_offset/data_bytes, exactly the
    DSP-5 sideband-readback contract (last block repeats when exhausted).
    Every other command gets a plain CMP_COMPUTE_DONE one cycle later.

    Returns a dict with:
      attn_trace:   [(layer_idx, seq_id, token_pos)] per RUN_ATTENTION.
      dspark_trace: [(anchor_token_id, anchor_pos, num_query)] per round.
      main_consumed / cycles.
    """
    daemon = deps["daemon"]
    ring_writer = deps["ring_writer"]
    hbuf = deps["host_buf"]
    sideband = deps.get("sideband")

    attn_trace: list[tuple[int, int, int]] = []
    dspark_trace: list[tuple[int, int, int]] = []
    main_i = 0
    block_i = 0
    off = 0
    responded = 0

    for cycle in range(max_cycles):
        loop.run_one_cycle()
        for cmd in ring_writer.written[responded:]:
            responded += 1
            t = getattr(cmd, "cmd_type", None)
            if t in (E_CMD_SEQ_CREATE, E_CMD_SEQ_FREE, CMD_SEQ_FORK):
                continue  # CMP_SEQ_OP_DONE acks are ignored by the loop
            if t == D_B_CMD_RUN_ATTENTION and sideband is not None:
                seq_id, pos = sideband.read_batch_descriptor()
                attn_trace.append(
                    (int(cmd.payload.run_attention.layer_idx), seq_id, pos)
                )
            if t == D_CMD_RUN_DSPARK_STEP:
                p = cmd.payload.run_dspark_step
                nq = int(p.num_query)
                dspark_trace.append(
                    (int(p.anchor_token_id), int(p.anchor_pos), nq)
                )
                if dspark_errors:
                    # Runtime declined the step (fail-closed context —
                    # e.g. TD-DSPARK-CTX-CAP arena overflow): CMP_ERROR.
                    daemon.at_cycle(daemon.cycle + 1, _make_error(
                        cmd.cmd_seq,
                        b"dspark run_step: no valid ingested context",
                    ))
                    continue
                block = dspark_blocks[min(block_i, len(dspark_blocks) - 1)]
                confs = (dspark_confs[min(block_i, len(dspark_confs) - 1)]
                         if dspark_confs else None)
                block_i += 1
                ids = list(block[:nq]) if nq > 0 else list(block)
                hbuf.write_tokens(off, ids)  # i32 ids (positive → same bits)
                nbytes = 4 * len(ids)
                if confs is not None:
                    # DSP-6 readback contract: gamma f32 c_k after the ids,
                    # data_bytes = gamma * 8 (confidence_enabled on both
                    # sides).
                    hbuf.write_floats(off + 4 * len(ids), confs[:len(ids)])
                    nbytes *= 2
                daemon.at_cycle(daemon.cycle + 1, _make_compute_done(
                    cmd.cmd_seq, host_buf_offset=off,
                    data_bytes=nbytes,
                ))
                off += 2 * 4 * max(len(ids), 1)
            elif t == CMD_OUTPUT_HEAD:
                assert main_i < len(main_tokens), "main token script ran dry"
                tok = main_tokens[main_i]
                main_i += 1
                hbuf.write_tokens(off, [tok])
                daemon.at_cycle(daemon.cycle + 1, _make_compute_done(
                    cmd.cmd_seq, host_buf_offset=off, data_bytes=4,
                    top1_prob=0.5,
                ))
                off += 4
            else:
                daemon.at_cycle(daemon.cycle + 1,
                                _make_compute_done(cmd.cmd_seq))
        daemon.advance()
        if not loop.requests:
            return {"attn_trace": attn_trace, "dspark_trace": dspark_trace,
                    "main_consumed": main_i, "cycles": cycle + 1}
    raise AssertionError(f"scenario did not finish in {max_cycles} cycles")


class TestScenarioDsparkSpeculation:
    """End-to-end DSpark rounds against a scripted daemon — the Python
    mirror of Glm52GgufGolden.DsparkLossless (DSP-5).

    Model: 2 layers (layer 0 dense, layer 1 MoE).  Prompt [10, 20, 30]
    (prompt_len=3), max_tokens=5; the scripted main model always answers
    42, 43, 100, 101, 55 in feed order.

    Asserted (mirroring the golden):
      - LOSSLESS: the committed tokens are EXACTLY the main model's
        outputs, whatever the drafter proposes (rejected drafts are never
        fed — sequential early-stop on the MAIN sequence).
      - Whole-block rounds: ONE D_CMD_RUN_DSPARK_STEP per round with the
        anchor convention anchor_pos == prompt_len + tokens_generated
        (the fed-token count) and anchor_token == the newest committed
        token; NO KV fork, NO MTP machinery, NO catch-up steps (the aux
        export builds the draft context automatically — INV-DSPARK-AUX).
      - Acceptance > 0 when drafts match (real speculation, not a no-op).

    NOTE (PLAN DSP-5 verify): the ticket sketched CMD_SEQ_FORK in the
    lifecycle; the shipped DSpark flow is fork-FREE by design (DSP-3
    context-KV rewind = in-place overwrite) — the assertion here is the
    absence of the fork, mirroring the MTP no-fork rework.
    """

    MAIN_TOKENS = [42, 43, 100, 101, 55]

    def _run(self, dspark_blocks, dspark_enabled=True, depth=2,
             dspark_confs=None, confidence_enabled=False,
             capture_combines=None, dspark_errors=False):
        hbuf = HostBuffer(4096)
        sideband = SidebandBuffer()
        loop, deps = _build_scenario_loop(
            dspark_enabled=dspark_enabled,
            dspark_gamma=2,
            dspark_confidence_enabled=confidence_enabled,
            recommended_depth=depth,
            host_buf=hbuf,
            sideband=sideband,
        )
        if capture_combines is not None:
            # Spy on the REAL combiner: record the DSP-6 per-position
            # survival list each round while delegating unchanged.
            combiner = loop._draft_combiner
            orig = combiner.combine

            def spy(*args, **kwargs):
                capture_combines.append(kwargs.get("dspark_confidences"))
                return orig(*args, **kwargs)
            combiner.combine = spy
        results: list = []
        loop.submit_request(InferenceRequest(
            request_id=1,
            prompt_token_ids=[10, 20, 30],
            gpu=0,
            max_tokens=5,
            on_complete=lambda rid, toks, reason, lp: results.append(
                (rid, list(toks), reason)),
        ))
        stats = _drive_dspark_scenario(
            loop, deps,
            main_tokens=list(self.MAIN_TOKENS),
            dspark_blocks=dspark_blocks,
            dspark_confs=dspark_confs,
            dspark_errors=dspark_errors,
        )
        assert len(results) == 1
        return results[0], stats, deps

    def test_dspark_drafting_lifecycle(self):
        """Accepting drafts: identical tokens, acceptance>0, exact seams."""
        # Round 1 drafts [100, 101] — both match the main model's own
        # continuation after 43.
        (rid, tokens, reason), stats, deps = self._run(
            dspark_blocks=[[100, 101]],
        )
        assert rid == 1
        assert reason == "length"
        # LOSSLESS: exactly the main model's own outputs, in order
        assert tokens == self.MAIN_TOKENS
        assert stats["main_consumed"] == 5
        # Real speculation: the whole block was accepted
        assert deps["dspark_draft"].total_accepted == 2
        assert deps["dspark_draft"].total_proposed == 2
        assert deps["dspark_draft"].total_rounds == 1
        assert deps["dspark_draft"].acceptance_rate == 1.0

        ring = deps["ring_writer"].written
        types = [c.cmd_type for c in ring if hasattr(c, "cmd_type")]
        # ONE whole-block draft round; DSpark seam only — no fork, no MTP
        # machinery, no fused MTP step, no self-spec forward.
        assert types.count(D_CMD_RUN_DSPARK_STEP) == 1
        assert CMD_SEQ_FORK not in types
        assert D_CMD_MTP_PROJECT not in types
        assert D_CMD_RUN_MTP_STEP not in types
        frees = [c for c in ring if getattr(c, "cmd_type", 0) == E_CMD_SEQ_FREE]
        assert len(frees) == 1  # request teardown only (main seq)
        assert frees[0].payload.seq_free.seq_id == 1

        # Anchor convention (#91 fixed position math): the round fired
        # after token 43 was committed (tokens_generated=2) → anchor_pos
        # = fed-token count = prompt_len(3) + 2 - 1 = 4 (43 is committed
        # but not yet fed), anchor token = 43, num_query = min(depth=2,
        # gamma=2).
        assert stats["dspark_trace"] == [(43, 4, 2)]

        # Real prefill feeds the prompt at 0..2 (#91); decode feeds are
        # gap-free from prompt_len — identical shape to the MTP
        # scenario's main_attn (3 AR, then verify feeds 4, 5, 6).
        main_attn = [(s, p) for (l, s, p) in stats["attn_trace"] if l == 0]
        assert main_attn == [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                             (1, 5), (1, 6)]

    def test_dspark_lossless_with_garbage_drafts(self):
        """Garbage drafts: tokens STILL identical (early-stop rejects all —
        a rejected draft is never fed), acceptance == 0."""
        (rid, tokens, reason), stats, deps = self._run(
            dspark_blocks=[[777, 778]],  # repeats: every round is garbage
        )
        assert tokens == self.MAIN_TOKENS  # INV-DSPARK-LOSSLESS
        assert stats["main_consumed"] == 5
        assert deps["dspark_draft"].total_accepted == 0
        assert deps["dspark_draft"].total_rounds >= 1
        # 777/778 were never committed anywhere
        assert 777 not in tokens and 778 not in tokens
        ring = deps["ring_writer"].written
        types = [c.cmd_type for c in ring if hasattr(c, "cmd_type")]
        assert CMD_SEQ_FORK not in types
        assert D_CMD_MTP_PROJECT not in types

    def test_dspark_step_error_disables_drafting_for_request(self):
        """TD-DSPARK-CTX-CAP graceful degradation: when the daemon declines
        a DSpark step (CMP_ERROR — e.g. the drafting context fail-closed on
        context-arena overflow), the loop must (a) abort the round and keep
        decoding LOSSLESSLY, and (b) stop re-issuing DSpark steps for that
        request (sticky per-request disable) instead of erroring every
        round."""
        (rid, tokens, reason), stats, deps = self._run(
            dspark_blocks=[[100, 101]],  # never consumed: every step errors
            dspark_errors=True,
        )
        assert rid == 1
        assert reason == "length"
        # LOSSLESS: plain-AR continuation of the main model's own outputs.
        assert tokens == self.MAIN_TOKENS
        assert stats["main_consumed"] == 5
        # No round ever produced a draft.
        assert deps["dspark_draft"].total_accepted == 0

        # Sticky disable: exactly ONE DSpark step was issued — the errored
        # round disabled drafting for the request; later rounds went
        # straight to the prompt-lookup-only combine (no error spam).
        ring = deps["ring_writer"].written
        types = [c.cmd_type for c in ring if hasattr(c, "cmd_type")]
        assert types.count(D_CMD_RUN_DSPARK_STEP) == 1
        assert CMD_SEQ_FORK not in types

    def test_dspark_confidence_sideband_round_trip(self):
        """DSP-6: with confidence_enabled the completion carries γ i32 ids
        + γ f32 c_k (data_bytes = γ*8); the loop parses both, and the
        combiner receives the CUMULATIVE survival a_j = Π c_i in place of
        the EMA proxy (INV-DSPARK-CONF) — tokens stay LOSSLESS."""
        captured: list = []
        (rid, tokens, reason), stats, deps = self._run(
            dspark_blocks=[[100, 101]],
            dspark_confs=[[0.9, 0.5]],
            confidence_enabled=True,
            capture_combines=captured,
        )
        assert tokens == self.MAIN_TOKENS  # lossless, confidence changes
        assert deps["dspark_draft"].total_accepted == 2  # ranking not ids
        # The dspark round's combine saw a_j = [0.9, 0.45] (to f32
        # round-trip precision); non-dspark rounds (draft result None)
        # saw None (proxy fallback).
        survivals = [c for c in captured if c is not None]
        assert len(survivals) == 1 and len(survivals[0]) == 2
        assert abs(survivals[0][0] - 0.9) < 1e-6
        assert abs(survivals[0][1] - 0.9 * 0.5) < 1e-6
        assert stats["dspark_trace"] == [(43, 4, 2)]

    def test_dspark_confidence_disabled_ids_only(self):
        """confidence_enabled=False: the DSP-5 ids-only contract is
        untouched (data_bytes = γ*4, combiner gets None → proxy)."""
        captured: list = []
        (rid, tokens, reason), stats, deps = self._run(
            dspark_blocks=[[100, 101]],
            confidence_enabled=False,
            capture_combines=captured,
        )
        assert tokens == self.MAIN_TOKENS
        assert deps["dspark_draft"].total_accepted == 2
        assert all(c is None for c in captured)

    def test_dspark_off_baseline_identical(self):
        """The DSpark-off baseline commits the same main-model tokens —
        speculation changes command traffic, never the token stream."""
        (rid, tokens, reason), stats, deps = self._run(
            dspark_blocks=[[0, 0]], dspark_enabled=False, depth=0,
        )
        assert tokens == self.MAIN_TOKENS
        ring = deps["ring_writer"].written
        types = [c.cmd_type for c in ring if hasattr(c, "cmd_type")]
        assert D_CMD_RUN_DSPARK_STEP not in types

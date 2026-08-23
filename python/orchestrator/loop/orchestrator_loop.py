"""Orchestrator main loop — 6-phase non-blocking event loop.

This is the central control file for the Python orchestrator.  It owns the
OrchestratorLoop class definition, constructor, and the six per-cycle phases:

  Phase 1: COLLECT      — drain completions from the C++ daemon + new requests.
  Phase 2: PREFETCH     — score experts for prefetching via PreScope/PROBE/SpeQ.
  Phase 3: STATISTICS   — read workload shift, calibrate draft parameters.
  Phase 4: PLAN         — set speculation depth, score work items, plan transfers.
  Phase 5: DISPATCH     — send eviction/transfer/compute commands to the ring.
  Phase 6: YIELD        — record metrics, idle-wait if no pending work.

All six phases must complete in <100us total (INV-3.4.1).
Single-threaded Python orchestrator (INV-3.4.2).  All GPU ops are async
via the C++ daemon thread.  Communication is lock-free SPSC rings (commands
outbound, completions inbound) + seqlock state snapshot (daemon publishes).

Helper methods are split into mixin classes in neighbouring files:
  - completion_handlers.py: per-CMP_* type handlers (checkpoint, transfer, etc.)
  - speculation.py:         5-state speculation lifecycle (draft/verify/accept)
  - training.py:            online predictor training sample collection
  - gpu_queries.py:         GPU state reads for planning (VRAM, cache, experts)

IMPORTANT: Keep this file richly commented.  The orchestrator loop is the
most complex module in the system and every helper must be self-documenting.
Do not remove comments during edits — add more if anything is unclear.
"""

from __future__ import annotations

import collections
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from orchestrator.verifier import MoeAnalysis

# -- Core IPC + domain types ------------------------------------------------
from orchestrator.command_writer import (
    CommandWriter,
    CompletionReader,
    read_sideband_routing_export,
    write_sideband_batch_descriptors,
    write_sideband_expert_prefetch,
    write_sideband_token_ids,
)
from orchestrator.config_hot_reload import ConfigHotReload
from orchestrator.draft_combiner import CombinedDraft, DraftCombiner
from orchestrator.dspark_draft import DsparkDraft, DsparkDraftResult
from orchestrator.epm_manifest import EpmManifestWriter, effective_epm_dump_dir
from orchestrator.eviction_policy import EvictionPolicy
from orchestrator.expert_placement import ExpertPlacement
from orchestrator.gpu_load_balancer import GpuLoadBalancer, LearnedGpuScorer
from orchestrator.layer_skip import LayerSkip
from orchestrator.moe_speq import MoeSpeq
from orchestrator.mtp_draft import DraftStep, MtpDraft, MtpDraftResult
from orchestrator.self_speculative import SelfSpecDraftResult, SelfSpeculative
from orchestrator.online_calibrator import OnlineCalibrator
from orchestrator.performance_objective import PerformanceObjective
from orchestrator.placement_optimizer import PlacementOptimizer
from orchestrator.prefetch_fuser import PrefetchFuser
from orchestrator.prescope import PreScope
from orchestrator.probe import Probe
from orchestrator.prompt_lookup import PromptLookup
from orchestrator.reasoning_mode import ReasoningMode
from orchestrator.scheduler import GpuAssigner, Scheduler
from orchestrator.shm_protocol import (
    CMP_CACHE_OP_DONE,
    CMP_CANCEL_DONE,
    CMP_CHECKPOINT,
    CMP_COMPUTE_DONE,
    CMP_CONFIG_UPDATE_DONE,
    CMP_ELM_EXPERT_EVICTED,
    CMP_ELM_EXPERT_PROGRESS,
    CMP_ELM_EXPERT_READY,
    CMP_ERROR,
    CMP_EVENT_STATUS,
    CMP_GPU_FATAL,
    CMP_NVME_DONE,
    CMP_SEQ_OP_DONE,
    CMP_TRANSFER_DONE,
    Completion,
)
from orchestrator.speculative_prefetch import SpeculativePrefetch
from orchestrator.spsc_ring import SpscRingWriter
from orchestrator.state_reader import StateReader
from orchestrator.transfer_scheduler import TransferScheduler
from orchestrator.types import (
    CacheZone,
    EngineMetadata,
    EvictionPlan,
    ExpertEvictionInput,
    ExpertKey,
    PrefetchHint,
    PrefetchPriority,
    SpeculationState,
    TransferPlan,
    VerificationPlan,
    WorkItem,
    WorkOperation,
    WorkStatus,
)
from orchestrator.utility_scorer import UtilityScorer
from orchestrator.verifier import CommandDescriptor, VerificationResult, Verifier
from orchestrator.work_queue import WorkQueue

# -- Mixin classes (grouped helper methods) ---------------------------------
from orchestrator.loop.completion_handlers import _CompletionMixin
from orchestrator.loop.gpu_queries import _GpuQueryMixin
from orchestrator.loop.speculation import _SpeculationMixin
from orchestrator.loop.training import _TrainingMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

# Reserved for gating pipeline separation — see TD-29, #13c-6
_MOE_LAYER_OPS = frozenset({WorkOperation.EXPERT_FFN, WorkOperation.GATING})

# Correctness gate for routed experts on the production FETCH_AND_RUN seam
# (#91, golden parity with glm52_gguf_golden_test's decode_step): a missing
# expert degrades like a truncation (wrong MoE output), so allow a generous
# fetch deadline — cold reads come from NVMe (~170 MB/layer for GLM-5.2).
_FETCH_AND_RUN_TIMEOUT_US = 120_000_000  # 120 s


@dataclass(frozen=True)
class OrchestratorConfig:
    """Tuning knobs for the orchestrator event loop."""
    cycle_budget_us: float = 100.0           # enforced in _phase_yield; 0 disables
    max_idle_wait_us: float = 50.0           # max busy-wait in YIELD phase
    emit_progress: bool = False              # reserved; intended to gate CMP_ELM_EXPERT_PROGRESS handling
    hidden_size: int = 7168                  # cross-module: used by residual_correction.py
    max_completions_per_drain: int = 64      # max completions per COLLECT
    transfer_origin_expiry_cycles: int = 10_000  # ~1 sec at 100μs/cycle
    # EP owner set for the FETCH_AND_RUN_MOE round-robin (GPU positions).
    # Empty = all GPUs (the 2-GPU e2e default). MUST exclude GPUs that
    # host no expert cache — e.g. a dspark draft GPU (the C++ golden's
    # ep_gpus_ = tp_array convention): an expert round-robined onto the
    # draft GPU can never become ready, and every MoE layer then burns
    # the full FETCH_AND_RUN timeout (the INV-FAR deadline-burn shape).
    ep_gpu_indices: tuple[int, ...] = ()
    # cancel_abort_transfers, cancel_skip_transfers, cancel_stability_window
    # are x-changeable and live in ConfigHotReload (speculation.transfer_cancellation.*)
    # -- TD-PREFILL-MOE-BIG: superchunk prefill ---------------------------
    # When on AND the engine published a MoE batch capacity > 1
    # (EngineMetadata.moe_batch_capacity), the prompt (all but its final
    # token) is prefilled in SUPERCHUNKS: per superchunk of up to
    # moe_batch_capacity tokens, embedding + attention run in
    # prefill_subchunk_tokens sub-chunks (RUN_ATTENTION row_offset +
    # superchunk flag) and ONE E_CMD_FETCH_AND_RUN_MOE_BIG per layer runs
    # the MoE over the whole superchunk — each routed expert fetched ONCE
    # per layer per superchunk (Order B), chunked GGEMMs + double-buffered
    # waves daemon-side.  The final prompt token then feeds through the
    # unchanged per-token path (its OUTPUT_HEAD yields the first generated
    # token, exactly like the C++ longctx driver's post-prefill decode).
    # Falls back to per-token prefill when MTP prompt-warm is active.
    prefill_superchunk: bool = True
    prefill_subchunk_tokens: int = 64        # attention/embedding sub-chunk


@dataclass
class RequestState:
    """Per-request mutable state tracked by the orchestrator.

    Lifetime: created in _drain_request_queue, removed on completion/error.
    """
    request_id: int
    seq_id: int                              # KV sequence ID (daemon-managed)
    default_gpu: int                         # home GPU for this request
    current_layer: int = 0                   # last completed layer index
    tokens_generated: int = 0                # decode tokens generated so far
    is_prefill: bool = True                  # still in prefill phase?
    prompt_len: int = 0                      # number of prompt tokens
    # Real prefill (#91): index of the prompt token currently being fed.
    # The whole prompt is teacher-forced one position at a time (the GGUF
    # golden's default mode — token i fed at KV position i); intermediate
    # positions skip OUTPUT_HEAD and chain straight to the next EMBEDDING.
    prefill_pos: int = 0
    max_tokens: int = 0                      # 0 = unlimited (set from InferenceRequest)
    cancelled: bool = False                  # set by cancel_request() for external stop
    logprobs: int | None = None              # top-K logprobs requested (None = disabled)
    on_complete: Callable[..., None] | None = None  # (request_id, tokens, finish_reason, logprobs)
    on_token: Callable[..., None] | None = None  # (request_id, token_id, step_logprobs)
    logprob_results: list = field(default_factory=list)  # list[StepLogprobs | None]
    # -- Speculation scaffolding -------------------------------------------
    speculation_depth: int = 0               # target draft depth this cycle
    max_verifiable_depth: int = 0            # ceiling from SP-MoE analysis
    draft_complete: bool = False             # draft forward pass done?
    draft_gating: np.ndarray | None = None   # (depth, moe_layers, experts)
    draft_seq_id: int | None = None          # KV fork for draft tokens
    acceptance_rate: float = 0.5             # EMA acceptance rate
    # -- Checkpoints (host buffer offsets) ---------------------------------
    hidden_state_checkpoints: dict[int, tuple[int, int]] = field(
        default_factory=dict
    )
    gating_output_checkpoints: dict[int, tuple[int, int]] = field(
        default_factory=dict
    )
    # -- Speculation lifecycle state (#62b) ---------------------------------
    speculation_state: SpeculationState = SpeculationState.AUTOREGRESSIVE
    combined_draft: CombinedDraft | None = None
    token_history: list[int] = field(default_factory=list)
    verification_plan: VerificationPlan | None = None
    verification_analyses: list[MoeAnalysis] = field(default_factory=list)
    verified_depth: int = 0
    # -- Speculation timing (TD-22) ----------------------------------------
    last_output_head_ns: int = 0             # perf_counter_ns of last AR OUTPUT_HEAD
    drafting_start_ns: int = 0               # perf_counter_ns when DRAFTING began
    verification_start_ns: int = 0           # perf_counter_ns when VERIFYING began
    # -- MTP draft tracking (#62d, reworked for TD-MTP-PY-LOOP-KV) ----------
    mtp_step: int = 0                        # current MTP step index (0-based)
    mtp_steps_result: list[DraftStep] = field(default_factory=list)
    mtp_cmd_seq: int | None = None           # cmd_seq of awaited MTP sub-command
    prompt_lookup_tokens: list[int] = field(default_factory=list)
    # -- MTP production-seam step sub-state (TD-MTP-PY-LOOP-KV / #89) -------
    # One MTP step = MTP_PROJECT -> RUN_ATTENTION(mtp layer, gating export)
    # -> FETCH_AND_RUN_MOE -> OUTPUT_HEAD(mtp_head).  mtp_phase tracks which
    # completion we're waiting for; mtp_mode distinguishes draft-chain steps
    # ("draft") from KV warm/catch-up steps ("catchup", INV-MTP-KV).
    # "attn-wait" parks the step on the routing-export slot gate: another
    # request's gating-bearing command is in flight, so dispatching the MTP
    # attention now could clobber that request's un-captured export
    # (TD-ORCH-ROUTING-EXPORT-MULTI / INV-IPC-6b).  Re-dispatched by
    # _retry_gating_waiters once the gate frees.
    mtp_phase: str | None = None    # "project"|"attn-wait"|"attn"|"moe"|"head"
    mtp_mode: str | None = None              # "draft" | "catchup"
    mtp_pos: int = 0                         # KV position of in-flight MTP step
    mtp_embed_token: int = 0                 # token embedded by in-flight step
    mtp_anchor_pos: int = 0                  # trunk anchor of current draft round
    # -- Sequential early-stop verification (MtpLossless pattern) -----------
    # Verification feeds run as normal AR forwards on the MAIN sequence; a
    # rejected draft token is NEVER fed, so the main KV needs no fork, no
    # rewind and no copy-back (fixes TD-MTP-PY-LOOP-KV bug (2) for MTP).
    mtp_sequential_verify: bool = False      # VERIFYING uses sequential feeds
    verify_tokens: list[int] = field(default_factory=list)  # pending drafts
    verify_idx: int = 0                      # accepted (matched) feed count
    # -- Batched verification on the MAIN sequence (MTP throughput mode) ----
    # ONE teacher-forced V-token forward (V = K drafts + 1) verifies the
    # whole round: EMBEDDING(V) -> per layer [chunked-prefill-shaped
    # RUN_ATTENTION(V) + FETCH_AND_RUN_MOE(V)] -> OUTPUT_HEAD(V, readback of
    # ALL V argmax tokens).  Feeds land at positions [vb_base_pos,
    # vb_base_pos + V) of the MAIN seq; rejected-position KV is scratch
    # above the committed frontier, overwritten in place by later feeds
    # before it is ever causally visible (same overwrite-above-frontier
    # mechanism as the MTP layer's draft scratch — no fork, no rewind).
    mtp_batched_verify: bool = False         # VERIFYING uses ONE V-token pass
    vb_len: int = 0                          # V = fed tokens this pass
    vb_base_pos: int = 0                     # first feed position
    vb_tokens: list[int] = field(default_factory=list)  # [x, d0..d(K-1)]
    # Post-verify INV-MTP-KV catch-up queue: (pos, embed_token, hidden_row)
    # exact-hidden MTP-KV rewrites for the accepted positions, consuming the
    # batched pass's trunk hiddens (attn_buf rows [0..V); MTP steps write
    # only row 0, so rows are consumed in ascending order).
    mtp_catchup_queue: list[tuple[int, int, int]] = field(default_factory=list)
    # Continuation after the queue drains: "draft" -> chain the next round
    # (its step 0 covers the new anchor), "ar"/None -> next AR embedding.
    mtp_post_verify: str | None = None
    # attn_buf row holding the NEW anchor's trunk hidden for the next
    # round's draft step 0 (row m after accepting m drafts).  0 outside the
    # post-batched-verify window (single-token feeds leave row 0).
    mtp_draft_hidden_row: int = 0
    # -- DSpark draft tracking (DSP-5) ---------------------------------------
    # One DSpark round = ONE D_CMD_RUN_DSPARK_STEP (whole γ block; no fork,
    # no catch-up steps — the aux export builds the draft context KV
    # automatically on every fed target position, INV-DSPARK-AUX).  The γ
    # sampled ids are read from the completion's sideband readback.
    dspark_cmd_seq: int | None = None        # cmd_seq of inflight dspark step
    dspark_result: DsparkDraftResult | None = None  # this round's block
    dspark_anchor_pos: int = 0               # fed-token count at dispatch
    # Sticky per-request kill switch (TD-DSPARK-CTX-CAP graceful
    # degradation): set when a DSpark step CMP_ERRORs — the runtime
    # declined this context (arena overflow / invalidated capture never
    # recovers within one request, positions only grow), so the loop stops
    # re-issuing a step that would error every round.  Decode continues
    # plain-AR + prompt-lookup, lossless.  NOT reset by _abort_draft.
    dspark_draft_disabled: bool = False
    # -- Self-spec draft tracking (#62e) -----------------------------------
    self_spec_step: int = 0                  # current self-spec step index
    self_spec_steps_result: list[DraftStep] = field(default_factory=list)
    self_spec_cmd_seq: int | None = None     # cmd_seq of inflight self-spec step
    self_spec_gating_rows: list[np.ndarray] = field(default_factory=list)
    # Self-spec forward parked on the routing-export slot gate (its
    # store_gating forward publishes the slot per MoE layer) — see
    # mtp_phase "attn-wait" above (TD-ORCH-ROUTING-EXPORT-MULTI).
    self_spec_wait: bool = False
    # -- Routing-export capture (TD-ORCH-ROUTING-EXPORT-MULTI) --------------
    # (layer_idx, indices) of the routed top-K published by this request's
    # most recent gating-bearing RUN_ATTENTION, copied out of the SHARED
    # sideband slot at COMPLETION time (per-completion capture inside
    # _handle_compute_done, INV-IPC-6b).  EXPERT_FFN dispatch forwards THIS
    # captured copy to FETCH_AND_RUN_MOE — never the live slot, which
    # another request's same-layer attention may have overwritten between
    # completion and dispatch.
    captured_routing: tuple[int, list[int]] | None = None
    # -- Superchunk prefill (TD-PREFILL-MOE-BIG) -----------------------------
    # Active while the prompt body ([0, prompt_len-1)) is prefilled in
    # superchunks.  One superchunk = sc_len tokens at prompt positions
    # [prefill_pos, prefill_pos + sc_len): embedding sub-chunks, then per
    # layer K attention sub-chunks (row_offset) + ONE FETCH_AND_RUN_MOE_BIG.
    # The work-item chain reuses EMBEDDING/ATTENTION/EXPERT_FFN operations;
    # sc_* steers _dispatch_work_item/_advance_request to the batched
    # commands.  sc_union accumulates the layer's routed-expert dedup set
    # (first-seen order across sub-chunks = the C++ harness's union order).
    sc_active: bool = False
    sc_len: int = 0                          # tokens in the current superchunk
    sc_off: int = 0                          # sub-chunk row offset in progress
    sc_layer: int = 0                        # layer being swept
    sc_union: list[int] = field(default_factory=list)
    sc_union_seen: set[int] = field(default_factory=set)
    # -- LayerSkip data flow (#62g) ----------------------------------------
    layer_similarities: dict[int, float] = field(default_factory=dict)
    current_skip_set: set[int] = field(default_factory=set)
    # -- Transfer cancellation (#62h-cancel) --------------------------------
    skip_set_history: list[set[int]] = field(default_factory=list)


@dataclass
class CycleMetrics:
    """Per-cycle performance counters returned by run_one_cycle()."""
    elapsed_us: float = 0.0
    completions_processed: int = 0
    commands_dispatched: int = 0


class _ResidentMembershipView:
    """Zero-copy membership view over the per-GPU resident-key sets.

    TD-ORCH-PLAN-CRAWL iter-4: Phase-4 PLAN only membership-tests a few
    required-expert keys per cycle — building the real union set (~1k
    keys in the full-fit deployment) every cycle dominated the wall.
    Duck-types the ``key in resident`` protocol used by
    Scheduler._compute_readiness and the affinity intersection.
    """

    __slots__ = ("_sets",)

    def __init__(self, sets: list[set]) -> None:
        self._sets = sets

    def __contains__(self, key) -> bool:
        for s in self._sets:
            if key in s:
                return True
        return False


@dataclass
class InferenceRequest:
    """External request submitted via submit_request()."""
    request_id: int
    prompt_token_ids: list[int] = field(default_factory=list)
    gpu: int = 0
    max_tokens: int = 0                      # 0 = unlimited
    logprobs: int | None = None              # top-K logprobs requested (None = disabled)
    on_complete: Callable[..., None] | None = None  # (request_id, tokens, finish_reason, logprobs)
    on_token: Callable[..., None] | None = None  # (request_id, token_id, step_logprobs)


# ---------------------------------------------------------------------------
# Main orchestrator class
# ---------------------------------------------------------------------------

class OrchestratorLoop(
    _SpeculationMixin,
    _CompletionMixin,
    _TrainingMixin,
    _GpuQueryMixin,
):
    """6-phase non-blocking orchestrator event loop.

    Not thread-safe.  Single-threaded Python orchestrator (INV-3.4.2).
    All GPU work dispatched async to C++ daemon via SPSC command ring.

    Helper methods are defined in mixin classes (see module docstring).
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        metadata: EngineMetadata,
        cmd_writer: CommandWriter,
        ring_writer: SpscRingWriter,
        completion_reader: CompletionReader,
        state_reader: StateReader,
        work_queue: WorkQueue,
        scheduler: Scheduler,
        gpu_assigner: GpuAssigner,
        transfer_scheduler: TransferScheduler,
        eviction_policy: EvictionPolicy,
        expert_placement: ExpertPlacement,
        placement_optimizer: PlacementOptimizer,
        prescope: PreScope,
        probe: Probe,
        moe_speq: MoeSpeq,
        prefetch_fuser: PrefetchFuser,
        speculative_prefetch: SpeculativePrefetch,
        utility_scorer: UtilityScorer,
        verifier: Verifier,
        prompt_lookup: PromptLookup,
        draft_combiner: DraftCombiner,
        mtp_draft: MtpDraft,
        self_speculative: SelfSpeculative,
        layer_skip: LayerSkip,
        reasoning_mode: ReasoningMode,
        online_calibrator: OnlineCalibrator,
        performance_objective: PerformanceObjective,
        hot_config: ConfigHotReload | None = None,
        host_buf_base: int = 0,
        sideband_base: int = 0,
        dspark_draft: DsparkDraft | None = None,
    ) -> None:
        # -- Module references (injected, never created here) ---------------
        self._config = config
        self._metadata = metadata
        self._cmd_writer = cmd_writer
        self._ring_writer = ring_writer
        self._completion_reader = completion_reader
        self._state_reader = state_reader
        self._work_queue = work_queue
        self._scheduler = scheduler
        self._gpu_assigner = gpu_assigner
        self._transfer_scheduler = transfer_scheduler
        self._eviction_policy = eviction_policy
        self._expert_placement = expert_placement
        self._placement_optimizer = placement_optimizer
        self._prescope = prescope
        self._probe = probe
        self._moe_speq = moe_speq
        self._prefetch_fuser = prefetch_fuser
        self._speculative_prefetch = speculative_prefetch
        self._utility_scorer = utility_scorer
        self._verifier = verifier
        self._prompt_lookup = prompt_lookup
        self._draft_combiner = draft_combiner
        self._mtp_draft = mtp_draft
        # DSpark draft planner (DSP-5) — optional keyword so existing
        # construction sites stay untouched; a default-constructed planner
        # is DISABLED (enabled derives from speculation.method == dspark).
        self._dspark_draft = dspark_draft or DsparkDraft()
        # EPM-1 (Phase 29): block-metadata manifest beside the daemon's raw
        # dumps (epm_blocks.bin / epm_routing.bin). None (default) = OFF —
        # the round-end hook pays a single `is not None` check. Armed only
        # when the dspark planner is enabled AND the dump dir resolves
        # non-empty (DsparkDraftConfig.epm_dump_dir mirroring the engine's
        # speculation.dspark.epm_dump_dir, or the LS_EPM_DUMP env override).
        self._epm_manifest: EpmManifestWriter | None = None
        if self._dspark_draft.is_enabled:
            _epm_dir = effective_epm_dump_dir(self._dspark_draft.epm_dump_dir)
            if _epm_dir:
                _writer = EpmManifestWriter(_epm_dir)
                self._epm_manifest = _writer if _writer.ok else None
        # EP owner set for FETCH_AND_RUN_MOE round-robin (see
        # OrchestratorConfig.ep_gpu_indices). Precomputed once.
        self._ep_gpus: tuple[int, ...] = (
            tuple(int(g) for g in config.ep_gpu_indices)
            or tuple(range(max(metadata.num_gpus, 1))))
        self._self_speculative = self_speculative
        self._layer_skip = layer_skip
        self._reasoning_mode = reasoning_mode
        self._online_calibrator = online_calibrator
        self._performance_objective = performance_objective
        # Live mutable config for x-changeable fields. Falls back to
        # frozen OrchestratorConfig defaults when not provided.
        self._hot_config = hot_config or ConfigHotReload()
        self._host_buf_base = host_buf_base
        self._sideband_base = sideband_base

        # -- Internal state -------------------------------------------------
        self._requests: dict[int, RequestState] = {}
        # Maps cmd_seq -> (request_id, WorkItem) for tracked commands
        self._cmd_seq_map: dict[int, tuple[int, WorkItem]] = {}
        # Maps cmd_seq -> request_id for verification intermediates (TD-21)
        self._verify_cmd_seqs: dict[int, int] = {}
        # Maps cmd_seq -> request_id for inflight draft step commands (#62d/#62e)
        self._draft_cmd_seqs: dict[int, int] = {}
        # Per-GPU sets of resident expert keys
        self._resident_keys: dict[int, set[ExpertKey]] = {
            g.position: set() for g in metadata.gpus
        }
        # Per-GPU sets of in-flight expert transfers
        self._inflight_transfers: dict[int, set[ExpertKey]] = {
            g.position: set() for g in metadata.gpus
        }
        # Origin-keyed transfer tracking (redesign replacing _transfer_cmd_seqs).
        # Maps (request_id, token_offset, draft_step) → {ExpertKey → (cmd_seq, gpu)}.
        # Only verification/draft transfers are tracked; Phase 5 general prefetches
        # are not (Phase 2 hint suppression handles those).  Stale entries expire
        # by cycle count — no per-completion cleanup needed.
        self._origin_transfers: dict[
            tuple[int, int, int], dict[ExpertKey, tuple[int, int]]
        ] = {}
        # Birth cycle per origin for expiry sweep
        self._origin_birth_cycle: dict[tuple[int, int, int], int] = {}
        # Cancellation queue: (origin, layer_filter or None for all).
        # Collected by LayerSkip / _abort_draft during Phase 1, consumed at
        # end of Phase 5 after all fetches.  Daemon ELM refcounting
        # deduplicates against same-cycle fetches automatically.
        self._cancel_queue: list[
            tuple[tuple[int, int, int], set[int] | None]
        ] = []
        self._request_queue: collections.deque[InferenceRequest] = (
            collections.deque()
        )
        self._tokens_processed: int = 0
        # Batched-verify round counter (MTP throughput mode): incremented
        # once per completed batched verification pass — e2e tests assert
        # the batched path actually engaged (vs the sequential fallback).
        self._batched_verify_rounds: int = 0
        # Routing-export slot gate (TD-ORCH-ROUTING-EXPORT-MULTI /
        # INV-IPC-6b): cmd_seq of the ONE in-flight gating-bearing command
        # (RUN_ATTENTION [store_gating] or the self-spec forward) whose
        # routed top-K has not yet been captured out of the shared sideband
        # slot.  While held, no other gating-bearing command may be
        # dispatched — the daemon would overwrite the slot before
        # _handle_compute_done captures it.  Mirrors the single physical
        # slot (IpcLayout::kRoutingExportOff).  Only engages when a
        # sideband mapping exists; with ONE active request the per-request
        # chain is ring-serialized so the gate never defers anything.
        self._gating_attn_inflight: int | None = None
        self._cycle_count: int = 0
        self._shutdown: bool = False
        self._last_metrics = CycleMetrics()
        self._commands_this_cycle: int = 0
        self._completions_this_cycle: int = 0
        self._pending_affinity_request = None
        self._pending_numa_migrations: list = []
        self._next_seq_id: int = 1

        # First MoE layer index (layers before this are dense attention-only)
        self._first_moe_layer = metadata.num_layers - metadata.num_moe_layers

        # Wire scheduler sub-delegates
        self._scheduler.eviction_scorer = eviction_policy
        self._scheduler.transfer_planner = transfer_scheduler
        self._scheduler.gpu_assigner = gpu_assigner

    # -- Properties ---------------------------------------------------------

    @property
    def config(self) -> OrchestratorConfig:
        return self._config

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def requests(self) -> dict[int, RequestState]:
        return self._requests

    @property
    def resident_keys(self) -> dict[int, set[ExpertKey]]:
        return self._resident_keys

    @property
    def last_metrics(self) -> CycleMetrics:
        return self._last_metrics

    def submit_request(self, request: InferenceRequest) -> None:
        """Enqueue an external inference request for the next cycle."""
        self._request_queue.append(request)

    def shutdown(self) -> None:
        """Signal the event loop to exit after the current cycle."""
        self._shutdown = True

    def cancel_request(self, request_id: int) -> None:
        """Mark a request for cancellation (e.g. client disconnect).

        The request will be finalized on the next OUTPUT_HEAD completion
        or at the next _phase_collect cycle.  Safe to call from any thread
        as it only sets a bool flag (Python GIL ensures atomicity).
        """
        req = self._requests.get(request_id)
        if req is not None:
            req.cancelled = True

    def _finalize_request(
        self, req: RequestState, *, finish_reason: str | None = None,
    ) -> None:
        """Terminate a request: abort speculation, free KV, remove from loop.

        Called when any stop criterion fires (EOS, max_tokens, cancel, error).
        *finish_reason* overrides auto-detection when set (e.g. ``"error"``).
        """
        # Abort any in-progress speculation
        if req.speculation_state != SpeculationState.AUTOREGRESSIVE:
            if req.draft_seq_id is not None:
                cmd = self._cmd_writer.e_seq_free(
                    gpu=req.default_gpu, seq_id=req.draft_seq_id,
                )
                self._ring_writer.write_struct(cmd)
                self._commands_this_cycle += 1
            # Clean up verification cmd_seq tracking
            stale = [cs for cs, rid in self._verify_cmd_seqs.items()
                     if rid == req.request_id]
            for cs in stale:
                del self._verify_cmd_seqs[cs]
        # Draft/MTP cmd_seq tracking is swept unconditionally: MTP KV
        # catch-up steps (INV-MTP-KV) run while the request is still in
        # AUTOREGRESSIVE state, so the state check above must not gate this.
        stale_draft = [cs for cs, rid in self._draft_cmd_seqs.items()
                       if rid == req.request_id]
        for cs in stale_draft:
            del self._draft_cmd_seqs[cs]
        # Free the main KV sequence
        cmd = self._cmd_writer.e_seq_free(
            gpu=req.default_gpu, seq_id=req.seq_id,
        )
        self._ring_writer.write_struct(cmd)
        self._commands_this_cycle += 1
        # Cancel all pending work items
        self._work_queue.cancel_by_request(req.request_id)
        # Sweep _cmd_seq_map for this request. Also catches orphaned checkpoint
        # entries where CMP_COMPUTE_DONE never arrived (TD-30).
        stale_tracked = [cs for cs, (rid, _) in self._cmd_seq_map.items()
                         if rid == req.request_id]
        for cs in stale_tracked:
            del self._cmd_seq_map[cs]
        # Deliver result to caller (HTTP server, benchmark harness, etc.)
        if req.on_complete is not None:
            generated = req.token_history[req.prompt_len:]
            if finish_reason is not None:
                reason = finish_reason
            elif req.cancelled:
                reason = "cancelled"
            elif (generated and self._metadata.eos_token_ids
                  and generated[-1] in self._metadata.eos_token_ids):
                reason = "stop"
            elif req.max_tokens > 0 and req.tokens_generated >= req.max_tokens:
                reason = "length"
            else:
                reason = "stop"
            lp = req.logprob_results if req.logprob_results else None
            req.on_complete(req.request_id, generated, reason, lp)
        # Remove from active requests
        self._requests.pop(req.request_id, None)

    # ======================================================================
    # Main loop
    # ======================================================================

    def run(self) -> None:
        """Run the event loop until shutdown() is called."""
        while not self._shutdown:
            self.run_one_cycle()

    def run_one_cycle(self) -> CycleMetrics:
        """Execute one complete 6-phase cycle and return metrics."""
        cycle_start = time.perf_counter_ns()
        self._commands_this_cycle = 0
        self._completions_this_cycle = 0

        # Phase 1: drain completions and new requests
        self._phase_collect()
        # Phase 2: score experts for prefetching
        priorities = self._phase_prefetch_scoring()
        # Phase 3: read statistics and calibrate
        self._phase_statistics_read()
        # Phase 4: plan transfers, evictions, compute batches
        ev_plan, tr_plan, batches = self._phase_plan(priorities)
        # Phase 5: dispatch commands to the ring
        self._phase_dispatch(ev_plan, tr_plan, batches)
        # Phase 6: record metrics, idle-wait
        self._phase_yield(cycle_start)

        self._cycle_count += 1
        return self._last_metrics

    # ======================================================================
    # Phase 1: COLLECT
    # ======================================================================

    def _phase_collect(self) -> None:
        """Drain completion ring and process new inference requests."""
        completions = self._completion_reader.drain(
            max_count=self._config.max_completions_per_drain
        )
        self._completions_this_cycle = len(completions)
        for cmp in completions:
            self._process_completion(cmp)
        # The routing-export slot gate can only free while processing
        # completions above — re-dispatch any speculation step that parked
        # on it (MTP "attn-wait" / self-spec wait, INV-IPC-6b).
        self._retry_gating_waiters()
        self._drain_request_queue()

    def _process_completion(self, cmp: Completion) -> None:
        """Route a completion to the appropriate handler by CMP_* type."""
        ct = cmp.cmp_type

        if ct == CMP_COMPUTE_DONE:
            self._handle_compute_done(cmp)
        elif ct == CMP_CHECKPOINT:
            self._handle_checkpoint(cmp)           # -> completion_handlers.py
        elif ct == CMP_TRANSFER_DONE:
            self._handle_transfer_done(cmp)        # -> completion_handlers.py
        elif ct == CMP_ELM_EXPERT_READY:
            self._handle_elm_expert_ready(cmp)     # -> completion_handlers.py
        elif ct == CMP_ELM_EXPERT_EVICTED:
            self._handle_elm_expert_evicted(cmp)   # -> completion_handlers.py
        elif ct == CMP_ELM_EXPERT_PROGRESS:
            pass  # should check self._config.emit_progress (TD-31)
        elif ct == CMP_SEQ_OP_DONE:
            pass  # seq lifecycle ack, no action needed
        elif ct == CMP_NVME_DONE:
            pass  # NVMe tier ack, no action needed
        elif ct == CMP_CANCEL_DONE:
            self._handle_cancel_done(cmp)          # -> completion_handlers.py
        elif ct == CMP_ERROR:
            self._handle_error(cmp)                # -> completion_handlers.py
        elif ct == CMP_GPU_FATAL:
            self._handle_gpu_fatal(cmp)            # -> completion_handlers.py
        elif ct == CMP_CACHE_OP_DONE:
            pass  # cache ops are fire-and-forget; no orchestrator action needed
        elif ct == CMP_CONFIG_UPDATE_DONE:
            pass  # daemon acknowledged config change; no orchestrator action needed
        elif ct == CMP_EVENT_STATUS:
            pass  # event status ack, no orchestrator action needed
        else:
            logger.warning("unknown completion type 0x%04x", ct)
            pass

    def _handle_compute_done(self, cmp: Completion) -> None:
        """Handle a compute command completion (ATTENTION, MoE, OUTPUT_HEAD).

        Four dispatch paths:
          1. Verification intermediate (in _verify_cmd_seqs): silently consume.
          2. DRAFTING completion (MTP or self-spec step): route to draft handler.
          3. OUTPUT_HEAD: delegate to _handle_output_head_done (speculation.py).
          4. Other ops: feed training samples if EXPERT_FFN, then advance request.
        """
        # Path 1: verification intermediate — just remove tracking entry
        if cmp.cmd_seq in self._verify_cmd_seqs:
            self._verify_cmd_seqs.pop(cmp.cmd_seq)
            return
        # Routing-export slot gate release (TD-ORCH-ROUTING-EXPORT-MULTI):
        # the in-flight slot producer completed.  Release BEFORE any
        # request-existence early-outs so a finalized owner can never leak
        # the gate (deadlocking every other gating dispatch).  The export
        # itself is captured below (Path 4 attention capture) or read
        # directly inside _handle_mtp_completion — both run inside THIS
        # completion's processing, before any other command can be
        # dispatched (single-threaded loop), so the slot is still ours.
        if self._gating_attn_inflight == cmp.cmd_seq:
            self._gating_attn_inflight = None
        # Look up the tracked command
        entry = self._cmd_seq_map.pop(cmp.cmd_seq, None)
        if entry is None:
            return
        request_id, item = entry
        self._work_queue.update_status(item, WorkStatus.COMPLETED)
        req = self._requests.get(request_id)
        if req is None:
            return

        # Path 2a: MTP production-seam sub-command completed.  Routed in ANY
        # speculation state: draft-chain steps run during DRAFTING, but KV
        # warm/catch-up steps (INV-MTP-KV) run during AUTOREGRESSIVE decode
        # and during sequential VERIFYING feeds too.
        if req.mtp_cmd_seq is not None and cmp.cmd_seq == req.mtp_cmd_seq:
            self._draft_cmd_seqs.pop(cmp.cmd_seq, None)
            self._handle_mtp_completion(req, cmp)         # -> speculation.py
            return

        # Path 2a': DSpark whole-block draft step completed (DSP-5).  The γ
        # sampled ids sit in the sideband readback at host_buf_offset.
        if (req.dspark_cmd_seq is not None
                and cmp.cmd_seq == req.dspark_cmd_seq):
            self._draft_cmd_seqs.pop(cmp.cmd_seq, None)
            self._handle_dspark_completion(req, cmp)      # -> speculation.py
            return

        # Path 2b: DRAFTING completion (self-spec step done)
        if req.speculation_state == SpeculationState.DRAFTING and item.is_speculative:
            self._draft_cmd_seqs.pop(cmp.cmd_seq, None)
            if req.self_spec_cmd_seq is not None and cmp.cmd_seq == req.self_spec_cmd_seq:
                self._handle_self_spec_completion(req, cmp)  # -> speculation.py
            return

        # Path 3: OUTPUT_HEAD -> speculation lifecycle entry point
        if item.operation == WorkOperation.OUTPUT_HEAD:
            self._handle_output_head_done(req, cmp)  # -> speculation.py
            return

        # Path 4: non-OUTPUT_HEAD (ATTENTION, EXPERT_FFN, EMBEDDING)
        if item.operation == WorkOperation.EXPERT_FFN:
            self._feed_training_samples(req, item.layer_idx)  # -> training.py

        # Per-completion routing-export capture (TD-ORCH-ROUTING-EXPORT-
        # MULTI): a gating-bearing RUN_ATTENTION just completed for this
        # request — copy its routed top-K out of the shared sideband slot
        # into the RequestState NOW, before any other request's completion
        # can be processed.  EXPERT_FFN dispatch reads this captured copy,
        # never the live slot (INV-IPC-6b).
        if (item.operation == WorkOperation.ATTENTION
                and item.layer_idx >= self._first_moe_layer):
            if req.sc_active:
                # Superchunk prefill (TD-PREFILL-MOE-BIG): each attention
                # sub-chunk exports ITS rows' routed top-K — accumulate the
                # layer-wide dedup UNION (first-seen order) instead of
                # replacing the capture; the layer's single
                # FETCH_AND_RUN_MOE_BIG consumes the union.
                self._accumulate_superchunk_routing(req, item.layer_idx)
            else:
                self._capture_routing_export(req, item.layer_idx)

        self._advance_request(req, item.layer_idx, item.operation)

    def _capture_routing_export(self, req: RequestState, layer: int) -> None:
        """Capture the sideband routing export into the owning RequestState.

        Called from _handle_compute_done the instant the export's producer
        (this request's RUN_ATTENTION [store_gating] at *layer*) completes.
        The dispatch gate (_gating_attn_inflight) guarantees no other
        gating-bearing command was in flight, so the slot content is this
        request's — a layer mismatch here indicates a gate bug or a rogue
        producer and is logged (the stale export is then discarded, which
        degrades that layer's MoE to expert_count=0 rather than silently
        computing another request's experts).
        """
        if not self._sideband_base:
            # Unit-test / no-sideband construction: nothing to capture;
            # EXPERT_FFN dispatch will forward an empty expert list, byte-
            # for-byte what the pre-capture code did.
            req.captured_routing = None
            return
        exp_layer, indices = read_sideband_routing_export(self._sideband_base)
        if exp_layer == layer:
            req.captured_routing = (layer, indices)
        else:
            req.captured_routing = None
            logger.warning(
                "routing export layer mismatch: slot has layer %d, "
                "expected %d (request %d) — discarding stale export",
                exp_layer, layer, req.request_id,
            )

    def _accumulate_superchunk_routing(self, req: RequestState,
                                       layer: int) -> None:
        """Union a superchunk sub-chunk's routing export into sc_union.

        TD-PREFILL-MOE-BIG: mirrors the C++ longctx driver's per-layer
        dedup set — each RUN_ATTENTION sub-chunk exports its own rows'
        routed top-K; the union (first-seen order across sub-chunks) is
        the expert list of the layer's single FETCH_AND_RUN_MOE_BIG.
        Same slot-gate guarantees as _capture_routing_export.
        """
        if not self._sideband_base:
            return
        exp_layer, indices = read_sideband_routing_export(self._sideband_base)
        if exp_layer != layer:
            logger.warning(
                "superchunk routing export layer mismatch: slot has layer "
                "%d, expected %d (request %d) — sub-chunk export dropped",
                exp_layer, layer, req.request_id,
            )
            return
        seen = req.sc_union_seen
        union = req.sc_union
        nexp = self._metadata.num_experts
        for e in indices:
            if 0 <= e < nexp and e not in seen:
                seen.add(e)
                union.append(e)

    def _retry_gating_waiters(self) -> None:
        """Re-dispatch speculation steps parked on the routing-export gate.

        MTP steps park in mtp_phase == "attn-wait" and self-spec steps in
        self_spec_wait when the gate was held at their dispatch point
        (another request's gating-bearing command was in flight).  The
        gate frees only inside completion processing, so _phase_collect
        calls this right after the drain loop.  Each re-dispatch may
        re-acquire the gate, in which case remaining waiters stay parked
        until the next completion (no starvation: the gate frees on every
        producer completion).
        """
        if self._gating_attn_inflight is not None:
            return
        for req in list(self._requests.values()):
            if req.mtp_phase == "attn-wait":
                self._dispatch_mtp_gating_attention(req)  # -> speculation.py
            elif req.self_spec_wait:
                req.self_spec_wait = False
                self._dispatch_self_spec_step(req)        # -> speculation.py
            if self._gating_attn_inflight is not None:
                return

    # -- Request lifecycle --------------------------------------------------

    def _drain_request_queue(self) -> None:
        """Convert queued InferenceRequests into RequestStates + EMBEDDING items.

        For each new request:
          1. Allocate a seq_id and create a RequestState.
          2. Send E_CMD_SEQ_CREATE to the daemon (allocates KV pages).
          3. Insert an EMBEDDING work item to start the forward pass.
        """
        while self._request_queue:
            req = self._request_queue.popleft()
            seq_id = self._next_seq_id
            self._next_seq_id += 1
            gpu = req.gpu

            state = RequestState(
                request_id=req.request_id,
                seq_id=seq_id,
                default_gpu=gpu,
                prompt_len=len(req.prompt_token_ids),
                max_tokens=req.max_tokens,
                logprobs=req.logprobs,
                on_complete=req.on_complete,
                on_token=req.on_token,
                token_history=list(req.prompt_token_ids),
            )
            self._requests[req.request_id] = state

            # TD-PREFILL-MOE-BIG: superchunk prefill over the prompt BODY
            # ([0, prompt_len-1)) when the engine published a batch capacity
            # and MTP prompt-warm is not interleaving per-position steps.
            # The final prompt token always feeds through the per-token path
            # (its OUTPUT_HEAD yields the first generated token).
            body = state.prompt_len - 1
            if (self._config.prefill_superchunk
                    and self._metadata.moe_batch_capacity > 1
                    and body >= 2
                    and not self._mtp_active()):
                state.sc_active = True
                state.sc_len = min(body, self._metadata.moe_batch_capacity)
                state.sc_off = 0
                state.sc_layer = 0

            # Tell daemon to create the KV sequence
            cmd = self._cmd_writer.e_seq_create(
                gpu=gpu, seq_id=seq_id,
                prompt_len=len(req.prompt_token_ids),
            )
            self._ring_writer.write_struct(cmd)
            self._commands_this_cycle += 1

            # Start the forward pass with EMBEDDING
            item = WorkItem(
                request_id=req.request_id,
                layer_idx=0,
                operation=WorkOperation.EMBEDDING,
                target_gpu=gpu,
                status=WorkStatus.READY,
                timestamp_created_ns=time.perf_counter_ns(),
            )
            self._work_queue.insert(item)

    def _advance_request(
        self,
        req: RequestState,
        completed_layer: int,
        completed_op: WorkOperation,
    ) -> None:
        """Create the next work item after a layer operation completes.

        Drives the layer-by-layer forward pass chain:
          EMBEDDING -> ATTENTION(0) -> EXPERT_FFN(0) -> ATTENTION(1) -> ... -> OUTPUT_HEAD

        All layers get EXPERT_FFN after ATTENTION. C++ dispatch_moe_internal
        handles dense layers (< first_k_dense_replace) via the dense FFN
        early-out path, and MoE layers via the routed + shared expert pipeline.
        After the last layer: -> OUTPUT_HEAD.
        """
        rid = req.request_id
        num_layers = self._metadata.num_layers

        # TD-PREFILL-MOE-BIG: superchunk prefill drives its own chain
        # (sub-chunked embedding/attention + one MoE-BIG per layer).
        if req.sc_active:
            self._advance_superchunk(req, completed_op)
            return

        if completed_op == WorkOperation.EMBEDDING:
            # EMBEDDING done -> start ATTENTION at layer 0
            item = WorkItem(
                request_id=rid, layer_idx=0,
                operation=WorkOperation.ATTENTION,
                target_gpu=req.default_gpu,
                status=WorkStatus.READY,
                timestamp_created_ns=time.perf_counter_ns(),
            )
            self._work_queue.insert(item)

        elif completed_op == WorkOperation.ATTENTION:
            # TD-84e: always create EXPERT_FFN after ATTENTION for all layers.
            # Dense layers are handled by C++ dispatch_moe_internal's dense early-out.
            item = WorkItem(
                request_id=rid, layer_idx=completed_layer,
                operation=WorkOperation.EXPERT_FFN,
                target_gpu=req.default_gpu,
                status=WorkStatus.READY,
                timestamp_created_ns=time.perf_counter_ns(),
            )
            self._work_queue.insert(item)

        elif completed_op == WorkOperation.EXPERT_FFN:
            req.current_layer = completed_layer + 1
            if completed_layer + 1 < num_layers:
                # More layers: -> ATTENTION at next layer
                item = WorkItem(
                    request_id=rid, layer_idx=completed_layer + 1,
                    operation=WorkOperation.ATTENTION,
                    target_gpu=req.default_gpu,
                    status=WorkStatus.READY,
                    timestamp_created_ns=time.perf_counter_ns(),
                )
                self._work_queue.insert(item)
            elif req.is_prefill and req.prefill_pos + 1 < req.prompt_len:
                # Real prefill (#91): an INTERMEDIATE prompt position just
                # finished its last layer.  Its logits are unused (the
                # successor is the next prompt token, teacher-forced), so
                # skip OUTPUT_HEAD and chain the next prompt feed directly.
                # With MTP active, first run the prompt-warm MTP step at the
                # just-fed position embedding its true successor (the next
                # prompt token) — INV-MTP-KV, the MtpLossless golden's
                # prompt-warm schedule.  The trunk hidden of this position
                # is still in the attn buffers (the MoE commit left it
                # there), which is exactly what the MTP projection consumes;
                # the next EMBEDDING is deferred until the warm completes.
                fed_pos = req.prefill_pos
                req.prefill_pos += 1
                if (self._mtp_active()
                        and req.prefill_pos < len(req.token_history)):
                    self._begin_mtp_step(
                        req, pos=fed_pos,
                        embed_token=req.token_history[req.prefill_pos],
                        mode="catchup",
                    )
                else:
                    self._enqueue_next_embedding(req)
            else:
                # Last layer of a decode feed (or the FINAL prompt
                # position): -> OUTPUT_HEAD
                item = WorkItem(
                    request_id=rid, layer_idx=completed_layer,
                    operation=WorkOperation.OUTPUT_HEAD,
                    target_gpu=req.default_gpu,
                    status=WorkStatus.READY,
                    timestamp_created_ns=time.perf_counter_ns(),
                )
                self._work_queue.insert(item)

        elif completed_op == WorkOperation.OUTPUT_HEAD:
            # OUTPUT_HEAD handling is in _handle_output_head_done (speculation.py)
            pass

    def _advance_superchunk(self, req: RequestState,
                            completed_op: WorkOperation) -> None:
        """Advance the superchunk-prefill chain (TD-PREFILL-MOE-BIG).

        One superchunk of sc_len tokens at prompt positions
        [prefill_pos, prefill_pos + sc_len):

          EMBEDDING sub-chunks (row_offset 0, sub, 2*sub, ...)
          -> per layer L: ATTENTION sub-chunks (row_offset, superchunk=1,
             routing exports unioned) -> EXPERT_FFN
             (ONE FETCH_AND_RUN_MOE_BIG over all sc_len tokens; dense
             layers: RUN_MOE)
          -> next superchunk | hand the FINAL prompt token to the
             per-token path (its OUTPUT_HEAD emits the first token).
        """
        rid = req.request_id
        sub = max(1, self._config.prefill_subchunk_tokens)
        num_layers = self._metadata.num_layers

        def _item(layer: int, op: WorkOperation) -> None:
            self._work_queue.insert(WorkItem(
                request_id=rid, layer_idx=layer, operation=op,
                target_gpu=req.default_gpu, status=WorkStatus.READY,
                timestamp_created_ns=time.perf_counter_ns(),
            ))

        if completed_op == WorkOperation.EMBEDDING:
            req.sc_off += sub
            if req.sc_off < req.sc_len:
                _item(0, WorkOperation.EMBEDDING)   # next embed sub-chunk
            else:
                req.sc_off = 0
                req.sc_layer = 0
                req.sc_union.clear()
                req.sc_union_seen.clear()
                _item(0, WorkOperation.ATTENTION)   # layer sweep starts

        elif completed_op == WorkOperation.ATTENTION:
            req.sc_off += sub
            if req.sc_off < req.sc_len:
                _item(req.sc_layer, WorkOperation.ATTENTION)
            else:
                _item(req.sc_layer, WorkOperation.EXPERT_FFN)

        elif completed_op == WorkOperation.EXPERT_FFN:
            req.current_layer = req.sc_layer + 1
            req.sc_layer += 1
            req.sc_off = 0
            req.sc_union.clear()
            req.sc_union_seen.clear()
            if req.sc_layer < num_layers:
                _item(req.sc_layer, WorkOperation.ATTENTION)
                return
            # Superchunk complete: advance the prefill cursor.
            req.prefill_pos += req.sc_len
            body = req.prompt_len - 1
            if req.prefill_pos < body:
                # Next superchunk.
                req.sc_len = min(body - req.prefill_pos,
                                 self._metadata.moe_batch_capacity)
                req.sc_off = 0
                req.sc_layer = 0
                _item(0, WorkOperation.EMBEDDING)
            else:
                # Prompt body prefilled — the FINAL prompt token feeds
                # through the unchanged per-token path (mirrors the C++
                # longctx driver's post-prefill decode_step).
                req.sc_active = False
                self._enqueue_next_embedding(req)

    def _enqueue_next_embedding(self, req: RequestState) -> None:
        """Enqueue an EMBEDDING work item for the next autoregressive token."""
        item = WorkItem(
            request_id=req.request_id,
            layer_idx=0,
            operation=WorkOperation.EMBEDDING,
            target_gpu=req.default_gpu,
            status=WorkStatus.READY,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        self._work_queue.insert(item)

    # ======================================================================
    # Phase 2: PREFETCH SCORING
    # ======================================================================

    def _phase_prefetch_scoring(self) -> list[PrefetchPriority]:
        """Collect predictions from all prefetch sources and fuse into priorities.

        Runs PreScope (one-layer-ahead), PROBE (multi-layer MLP), MoE-SpeQ
        (CPU inference), and SP-MoE (from draft gating) on each active request.
        Results are fused by the PrefetchFuser into a single priority list
        consumed by Phase 4's transfer planning.
        """
        prescope_hints: list[PrefetchHint] = []
        probe_hints: list[PrefetchHint] = []
        speq_hints: list[PrefetchHint] = []
        sp_moe_hints: list[PrefetchHint] = []

        for req in self._requests.values():
            layer = req.current_layer

            # PreScope: predict next layer's experts from current gating
            if self._prescope.enabled and self._prescope.is_moe_layer(layer):
                gating_ckpt = req.gating_output_checkpoints.get(layer)
                if gating_ckpt is not None:
                    target = self._prescope.target_layer(layer)
                    if target is not None:
                        scores = self._read_host_buf(gating_ckpt)
                        if scores is not None:
                            prescope_hints.extend(
                                self._prescope.process_gating_output(
                                    target, scores
                                )
                            )

            # PROBE: predict distant-layer experts from probe-point hidden states
            if self._probe.enabled:
                for pp_layer in self._probe.probe_layers:
                    hs_ckpt = req.hidden_state_checkpoints.get(pp_layer)
                    if hs_ckpt is not None and pp_layer <= layer:
                        hs = self._read_host_buf(hs_ckpt)
                        if hs is not None:
                            probe_hints.extend(
                                self._probe.predict(pp_layer, hs)
                            )

            # MoE-SpeQ: CPU-side expert prediction from hidden states
            if self._moe_speq.enabled and self._moe_speq.is_moe_layer(layer):
                hs_ckpt = req.hidden_state_checkpoints.get(layer)
                if hs_ckpt is not None:
                    hs = self._read_host_buf(hs_ckpt)
                    if hs is not None:
                        speq_hints.extend(
                            self._moe_speq.predict(layer, hs)
                        )

            # SP-MoE: expert hints from draft gating (speculative prefetch)
            if req.draft_complete and req.draft_gating is not None:
                sp_moe_hints.extend(
                    self._speculative_prefetch.gating_to_hints(
                        req.draft_gating, self._first_moe_layer,
                    )
                )

        # Suppress prefetch hints for layers in any request's LayerSkip
        # skip set (INV-4.15b).  Skipped layers don't execute MoE, so
        # prefetching their experts wastes PCIe bandwidth.
        all_skipped: set[int] = set()
        for req in self._requests.values():
            if req.current_skip_set:
                all_skipped |= req.current_skip_set
        if all_skipped:
            sp_moe_hints = [
                h for h in sp_moe_hints
                if h.target_layer not in all_skipped
            ]

        # Fuse all sources into a single priority list
        current_layer = min(
            (r.current_layer for r in self._requests.values()), default=0
        )
        return self._prefetch_fuser.fuse(
            prescope_hints, probe_hints, speq_hints, sp_moe_hints,
            current_layer=current_layer,
        )

    # ======================================================================
    # Phase 3: STATISTICS READ
    # ======================================================================

    def _phase_statistics_read(self) -> None:
        """Read workload statistics and adjust calibration parameters.

        Checks for workload shifts (triggers placement re-optimization).
        Updates the online calibrator with the current acceptance rate,
        and if parameters changed, syncs them to the daemon via config_update.
        """
        if self._state_reader.shift_detected():
            self._placement_optimizer.on_workload_shift(
                self._tokens_processed
            )

        acceptance = self._state_reader.windowed_acceptance_rate()
        adjusted = self._online_calibrator.update(
            acceptance, self._tokens_processed
        )
        if adjusted:
            self._sync_draft_params()

    def _sync_draft_params(self) -> None:
        """Push adjusted calibration parameters to the C++ daemon."""
        params = self._online_calibrator.params
        cmd = self._cmd_writer.config_update([
            (0, 0, params.draft_expert_count),
            (0, 1, int(params.layer_skip_enabled)),
        ])
        self._ring_writer.write_struct(cmd)
        self._commands_this_cycle += 1

    # ======================================================================
    # Phase 4: PLAN
    # ======================================================================

    def _phase_plan(
        self, prefetch_priorities: list[PrefetchPriority],
    ) -> tuple[EvictionPlan, TransferPlan, list]:
        """Plan all work for this cycle: speculation depth, GPU assignment,
        transfer/eviction/compute scheduling.

        Steps:
          1. Set speculation_depth for each AUTOREGRESSIVE request.
          2. Re-check PREFETCHING_VERIFY requests (experts may have arrived).
          3. Score and assign pending work items.
          4. Plan transfers (from prefetch priorities) and evictions.
          5. Plan compute batches from READY items.
          6. Optionally re-optimize placement and NUMA migrations.
        """
        now_ns = time.perf_counter_ns()

        # Step 1-2: per-request speculation depth + verification readiness
        for req in self._requests.values():
            if req.speculation_state == SpeculationState.AUTOREGRESSIVE:
                req.speculation_depth = (
                    self._utility_scorer.recommended_depth()
                )
                # If a future MTP/self-spec path produced draft_gating,
                # refine depth via SP-MoE coverage analysis
                if req.draft_complete and req.draft_gating is not None:
                    vplan = (
                        self._speculative_prefetch
                        .compute_verification_plan(
                            req.draft_gating, self._resident_keys,
                            first_moe_layer=self._first_moe_layer,
                        )
                    )
                    req.max_verifiable_depth = vplan.max_depth
                    req.speculation_depth = UtilityScorer.apply_ceiling(
                        req.speculation_depth,
                        req.max_verifiable_depth,
                    )
            elif (
                req.speculation_state
                == SpeculationState.PREFETCHING_VERIFY
            ):
                self._check_verification_ready(req)

        # Step 3: score and assign pending work items
        pending = self._work_queue.pending_items()
        if pending:
            queue_depths = self._compute_queue_depths()       # -> gpu_queries.py
            vram_free = self._compute_vram_free()             # -> gpu_queries.py
            # TD-ORCH-PLAN-CRAWL iter-4: score/_compute_readiness and the
            # iter-3 intersection below only ever MEMBERSHIP-test a
            # handful of required-expert keys — materializing the ~1k-key
            # union set every cycle was 27% of the collector wall. A
            # zero-copy __contains__ view over the per-GPU sets is
            # membership-identical.
            resident_all = _ResidentMembershipView(
                list(self._resident_keys.values()))

            # TD-ORCH-PLAN-CRAWL iter-3: the assigner only ever looks up
            # PENDING items' required_experts in this map (assign/_score
            # use affinity_map.get(key) with key from item.required_experts;
            # recommend_transfer_target is called without a map). Building
            # it over every resident key (~1k entries, EVERY cycle) was
            # 57% of the collector wall. Restricting to required∩resident
            # is lookup-for-lookup identical: the old map held exactly the
            # resident keys, so non-resident required experts missed (got
            # no affinity credit) — the intersection preserves that.
            need: set[ExpertKey] = set()
            for item in pending:
                need.update(item.required_experts)
            affinity_map = self._expert_placement.affinity_map_batch(
                [k for k in need if k in resident_all])
            if hasattr(self._gpu_assigner, "update_state"):
                self._gpu_assigner.update_state(
                    self._resident_keys, queue_depths, vram_free,
                    affinity_map,
                )

            for item in pending:
                item.priority = self._scheduler.score(
                    item, resident_all, now_ns,
                )
                self._work_queue.update_priority(item, item.priority)

            if hasattr(self._gpu_assigner, "assign_batch"):
                self._gpu_assigner.assign_batch(
                    pending, self._resident_keys, queue_depths,
                    vram_free, affinity_map,
                )
            else:
                for item in pending:
                    item.target_gpu = self._gpu_assigner.assign(item)

        # Step 3b: assign target_gpu for prefetch priorities (TD-59d)
        if (prefetch_priorities
                and hasattr(self._gpu_assigner, "recommend_transfer_target")):
            queue_depths_pf = self._compute_queue_depths()
            vram_free_pf = self._compute_vram_free()
            for pp in prefetch_priorities:
                pp.target_gpu = self._gpu_assigner.recommend_transfer_target(
                    pp.key, self._resident_keys, queue_depths_pf, vram_free_pf,
                )

        # Step 4: plan transfers and evictions
        inflight_counts = {
            gpu: len(keys)
            for gpu, keys in self._inflight_transfers.items()
        }
        transfer_plan = self._transfer_scheduler.plan_transfers(
            prefetch_priorities,
            resident_set=self._resident_keys,
            inflight_set=self._inflight_transfers,
            inflight_counts=inflight_counts,
            transfer_latency_us={},
        )

        # TD-ORCH-PLAN-CRAWL (#13c-1 item 2, skip-when-no-evictions):
        # plan_evictions with empty required_slots is a structural no-op,
        # and _build_eviction_inputs scans EVERY resident expert (~1k keys
        # against the full-fit windows) — only pay for it when this
        # cycle's transfer plan actually claims slots.
        required_slots = transfer_plan.required_slots_per_gpu()
        if required_slots:
            eviction_inputs = self._build_eviction_inputs()   # -> gpu_queries.py
            free_slots = self._compute_free_slots()           # -> gpu_queries.py
            eviction_plan = self._transfer_scheduler.plan_evictions(
                required_slots,
                eviction_inputs, self._eviction_policy, free_slots,
            )
        else:
            eviction_plan = EvictionPlan()

        # Step 5: plan compute batches
        ready_items = self._work_queue.by_status(WorkStatus.READY)
        compute_batches = self._scheduler.plan_compute(ready_items)

        # Step 6: placement re-optimization + NUMA migrations
        gpu_caps = self._compute_gpu_capacities()             # -> gpu_queries.py
        hint_req = self._placement_optimizer.maybe_reoptimize(
            shift_detected=False,
            tokens_processed=self._tokens_processed,
            now_s=time.time(),
            gpu_capacities=gpu_caps,
        )
        if hint_req is not None:
            self._pending_affinity_request = hint_req

        host_resident = self._compute_host_resident_keys()    # -> gpu_queries.py
        self._pending_numa_migrations = (
            self._placement_optimizer.process_numa_migrations(host_resident)
        )

        if isinstance(self._gpu_assigner, LearnedGpuScorer):
            self._gpu_assigner.maybe_train()

        return eviction_plan, transfer_plan, compute_batches

    # ======================================================================
    # Phase 5: DISPATCH
    # ======================================================================

    def _phase_dispatch(
        self,
        eviction_plan: EvictionPlan,
        transfer_plan: TransferPlan,
        compute_batches: list,
    ) -> None:
        """Send planned commands to the ring in priority order.

        Order matters: evictions free VRAM -> transfers fill it -> compute uses it.
        NUMA migrations and affinity hints are dispatched last.
        """
        # Evictions first (free VRAM before transfers claim it)
        for ev in eviction_plan.entries:
            cmd = self._cmd_writer.evict_to_host(
                gpu=ev.gpu_idx,
                layer_idx=ev.key.layer_idx,
                expert_idx=ev.key.expert_idx,
            )
            self._ring_writer.write_struct(cmd)
            self._commands_this_cycle += 1
            self._resident_keys.get(ev.gpu_idx, set()).discard(ev.key)

        # Transfers (load experts into freed VRAM)
        for tr in transfer_plan.entries:
            cmd = self._cmd_writer.prefetch_expert(
                gpu=tr.target_gpu,
                layer_idx=tr.key.layer_idx,
                expert_idx=tr.key.expert_idx,
                zone=tr.zone.value if isinstance(tr.zone, CacheZone) else tr.zone,
                priority=tr.priority,
                delay_us=tr.start_delay_us,
            )
            self._ring_writer.write_struct(cmd)
            self._commands_this_cycle += 1
            self._inflight_transfers.setdefault(tr.target_gpu, set()).add(
                tr.key
            )
        # Process cancellation queue AFTER all fetches (origin-based redesign).
        # Daemon ELM refcounting deduplicates: if a fetch and cancel target the
        # same expert this cycle, the interest survives (fetch increments, cancel
        # decrements → net zero if single-interest, positive if multi-interest).
        if self._cancel_queue:
            for origin, layer_filter in self._cancel_queue:
                entries = self._origin_transfers.get(origin, {})
                for key, (cmd_seq, gpu) in list(entries.items()):
                    if (layer_filter is not None
                            and key.layer_idx not in layer_filter):
                        continue
                    inflight = self._inflight_transfers.get(gpu, set())
                    if key not in inflight:
                        continue  # already completed, skip
                    cmd = self._cmd_writer.cancel_transfer(
                        gpu=gpu, target_cmd_seq=cmd_seq,
                    )
                    self._ring_writer.write_struct(cmd)
                    self._commands_this_cycle += 1
                    inflight.discard(key)
            self._cancel_queue.clear()

        # Compute batches (work items with all experts resident)
        for batch in compute_batches:
            for item in batch.items:
                self._dispatch_work_item(item)

        # NUMA migrations (low priority, background)
        for migration in self._pending_numa_migrations:
            cmd = self._cmd_writer.numa_migrate(
                gpu=migration.new_gpu,
                layer=migration.key.layer_idx,
                expert=migration.key.expert_idx,
                target_numa_node=migration.new_numa_node,
            )
            self._ring_writer.write_struct(cmd)
            self._commands_this_cycle += 1

        # Affinity hint requests (triggers daemon-side recompute)
        if self._pending_affinity_request is not None:
            req = self._pending_affinity_request
            cmd = self._cmd_writer.compute_affinity_hints(
                req.num_gpus, req.gpu_capacities,
            )
            self._ring_writer.write_struct(cmd)
            self._commands_this_cycle += 1
            self._pending_affinity_request = None

    def _feed_pos(self, req: RequestState) -> int:
        """KV position of the next (or in-flight) feed for this request.

        Prefill: the prompt index currently being fed (teacher-forced,
        token i at position i — golden parity).  Decode: the fed-token
        count.  The prompt occupies [0, prompt_len) and every committed
        token is fed exactly once with the newest committed token always
        pending, so the next feed lands at prompt_len + tokens_generated
        - 1 (#91 off-by-one fix: the first generated token is fed at
        position prompt_len, exactly where the golden feeds it — the old
        convention skipped a slot by dispatching at prompt_len + 1).
        """
        if req.is_prefill:
            return req.prefill_pos
        return max(req.prompt_len + req.tokens_generated - 1, 0)

    def _dispatch_superchunk_item(self, req: RequestState, item: WorkItem,
                                  gpu: int):
        """Build the next superchunk-prefill command (TD-PREFILL-MOE-BIG).

        Mirrors the C++ longctx driver's prefill_superchunk_step:

          EMBEDDING  — sub-chunk embed into hidden rows [off, off+len)
                       (sideband token ids = the sub-chunk's prompt slice).
          ATTENTION  — sub-chunk RUN_ATTENTION (superchunk=1, row_offset,
                       batch descriptors (seq, pos0+off+b); MoE layers run
                       the fused gate + export their rows, unioned at
                       completion by _accumulate_superchunk_routing).
          EXPERT_FFN — ONE E_CMD_FETCH_AND_RUN_MOE_BIG over all sc_len
                       tokens with the layer's routed-expert union (dense
                       layers: D_B_CMD_RUN_MOE).

        Returns None when the routing-export slot gate is held (the item
        stays READY and re-plans next cycle, like the per-token path).
        """
        op = item.operation
        sub = max(1, self._config.prefill_subchunk_tokens)
        off = req.sc_off
        length = min(sub, req.sc_len - off)
        pos0 = req.prefill_pos            # superchunk base prompt position

        if op == WorkOperation.EMBEDDING:
            if self._sideband_base and req.token_history:
                toks = req.token_history[pos0 + off: pos0 + off + length]
                write_sideband_token_ids(self._sideband_base, toks)
            return self._cmd_writer.embedding_lookup(
                gpu=gpu, num_tokens=length,
                output_buf=self._metadata.hidden_buf_id,
                row_offset=off,
            )

        if op == WorkOperation.ATTENTION:
            layer = req.sc_layer
            is_moe = layer >= self._first_moe_layer
            # Routing-export slot gate (INV-IPC-6b) — same rule as the
            # per-token path: never dispatch a store_gating attention
            # while another export is in flight and uncaptured.
            if (is_moe and self._sideband_base
                    and self._gating_attn_inflight is not None):
                return None
            if self._sideband_base:
                write_sideband_batch_descriptors(
                    self._sideband_base,
                    [(req.seq_id, pos0 + off + b) for b in range(length)])
            cmd = self._cmd_writer.run_attention(
                gpu=gpu, layer_idx=layer, num_seqs=length,
                is_prefill=1,
                chunk_start=pos0 + off, chunk_len=length,
                emit_gating=1 if is_moe else 0,
                store_gating=1 if is_moe else 0,
                superchunk=1, row_offset=off,
            )
            if is_moe and self._sideband_base:
                self._gating_attn_inflight = cmd.cmd_seq
            return cmd

        if op == WorkOperation.EXPERT_FFN:
            layer = req.sc_layer
            if layer >= self._first_moe_layer:
                # Ownership round-robins over the EP owner set
                # (OrchestratorConfig.ep_gpu_indices), exactly like the
                # decode/verify + MTP sites (74b39246).  The old `e % ngpu`
                # here targeted EVERY GPU — in a topology with a draft GPU
                # (no or undersized expert window) the over-capacity share
                # livelocked the daemon's ELM completion path
                # (TD-ORCH-ELM-COMPLETION-LIVELOCK).
                owners = self._ep_gpus
                entries = [
                    (layer, e, CacheZone.STABLE.value,
                     owners[e % len(owners)])
                    for e in req.sc_union
                ]
                if self._sideband_base and entries:
                    write_sideband_expert_prefetch(
                        self._sideband_base, entries)
                return self._cmd_writer.fetch_and_run_moe_big(
                    gpu=gpu, layer_idx=layer, num_seqs=req.sc_len,
                    expert_count=len(entries),
                    timeout_us=_FETCH_AND_RUN_TIMEOUT_US,
                )
            # Dense FFN layers: RUN_MOE over the whole superchunk
            # (dispatch_moe dense early-out; chunked daemon-side too).
            return self._cmd_writer.run_moe(
                gpu=gpu, layer_idx=layer, num_seqs=req.sc_len,
            )

        return None

    def _dispatch_work_item(self, item: WorkItem) -> None:
        """Build and send a single compute command for a work item.

        Maps WorkOperation to the appropriate CommandWriter method and
        tracks the cmd_seq for completion matching.  The per-feed command
        chain mirrors the GGUF golden's decode_step (the production
        engine seam, #91): EMBEDDING → per layer [RUN_ATTENTION(fused
        gate + routing export for MoE layers) → FETCH_AND_RUN_MOE (MoE)
        | RUN_MOE (dense)] → OUTPUT_HEAD(readback).
        """
        op = item.operation
        layer = item.layer_idx
        # Forward-chain commands are issued ONCE on the request's home GPU
        # (the primary rank) — the daemon fans out to all TP ranks
        # internally (DcpExecutor / dispatch_moe_all_ranks), exactly like
        # every C++ driver (gpu_idx=0).  The gpu_assigner's per-item
        # rebalancing applies to transfers/prefetch, not to steps of one
        # TP forward (#91 real-engine convention).
        chain_req = self._requests.get(item.request_id)
        gpu = chain_req.default_gpu if chain_req else item.target_gpu

        # TD-PREFILL-MOE-BIG: superchunk prefill — batched sub-chunk
        # commands (embedding/attention with row_offset; one MoE-BIG per
        # layer) replace the per-token forms while sc_active.
        if chain_req is not None and chain_req.sc_active:
            cmd = self._dispatch_superchunk_item(chain_req, item, gpu)
            if cmd is None:
                return  # gated (routing-export slot busy) — re-plan next cycle
            self._ring_writer.write_struct(cmd)
            self._commands_this_cycle += 1
            self._cmd_seq_map[cmd.cmd_seq] = (item.request_id, item)
            self._work_queue.update_status(item, WorkStatus.DISPATCHED)
            return

        # Batched verification pass (MTP throughput mode): the whole chain
        # (EMBEDDING -> per-layer ATTENTION/EXPERT_FFN -> OUTPUT_HEAD) runs
        # over V = vb_len teacher-forced tokens in one command each.  The
        # work-item chain shape is unchanged — only num_seqs/num_tokens and
        # the sideband inputs widen to V.
        vb = (chain_req is not None and chain_req.mtp_batched_verify
              and chain_req.vb_len > 0)

        if op == WorkOperation.EMBEDDING:
            # Write the input token(s) to the sideband so the daemon's
            # CMD_EMBEDDING_LOOKUP embeds the right token(s).  Prefill feeds
            # the prompt token at prefill_pos (#91 real prefill); decode
            # always feeds the newest committed token (AR continuation and
            # sequential-verify feeds alike: token_history[-1] is exactly
            # the not-yet-fed token); a batched verify pass feeds all V
            # teacher-forced tokens [x, d0..d(K-1)] at once.  Guarded: unit
            # tests run without a sideband mapping (sideband_base=0).
            req = self._requests.get(item.request_id)
            if req is not None and self._sideband_base and req.token_history:
                if vb:
                    write_sideband_token_ids(
                        self._sideband_base, req.vb_tokens)
                else:
                    if (req.is_prefill
                            and req.prefill_pos < len(req.token_history)):
                        feed_tok = req.token_history[req.prefill_pos]
                    else:
                        feed_tok = req.token_history[-1]
                    write_sideband_token_ids(self._sideband_base, [feed_tok])
            cmd = self._cmd_writer.embedding_lookup(
                gpu=gpu, num_tokens=chain_req.vb_len if vb else 1,
                output_buf=self._metadata.hidden_buf_id,
            )
        elif op == WorkOperation.ATTENTION:
            # TD-51cg: write batch descriptor so C++ build_kv_metadata()
            # can resolve seq_id → pages for KV cache addressing.
            # MoE layers run the fused router gate at the end of attention
            # and export the routed top-K to the sideband routing-export
            # slot (F-1/F-3 production seam) — captured at completion time
            # (_capture_routing_export) and forwarded by this layer's
            # FETCH_AND_RUN_MOE dispatch below.
            is_moe = layer >= self._first_moe_layer
            # Routing-export slot gate (TD-ORCH-ROUTING-EXPORT-MULTI /
            # INV-IPC-6b): the slot is a SINGLE shared buffer.  If another
            # request's gating-bearing command is still in flight (its
            # export not yet captured), dispatching this store_gating
            # attention now would let the daemon overwrite that export
            # before its owner captures it.  Defer: the item stays READY
            # and is re-planned next cycle.  Never engages with a single
            # active request — its command chain is strictly serialized
            # (each dispatch waits for the previous completion, which
            # releases the gate).
            if (is_moe and self._sideband_base
                    and self._gating_attn_inflight is not None):
                return
            req = self._requests.get(item.request_id)
            if vb:
                # Batched verify: V rows of the MAIN seq at successive
                # positions — the chunked-prefill shape (teacher-forced
                # batch, k_append before attention, causal windows per
                # row), exactly what real prefill drives at B up to
                # thousands.  The MoE layers' fused gate exports all V
                # rows' routed top-K (publish_routing_export is already
                # multi-token).
                if self._sideband_base:
                    write_sideband_batch_descriptors(
                        self._sideband_base,
                        [(req.seq_id, req.vb_base_pos + i)
                         for i in range(req.vb_len)])
                cmd = self._cmd_writer.run_attention(
                    gpu=gpu, layer_idx=layer, num_seqs=req.vb_len,
                    is_prefill=1,
                    chunk_start=req.vb_base_pos, chunk_len=req.vb_len,
                    emit_gating=1 if is_moe else 0,
                    store_gating=1 if is_moe else 0,
                )
            else:
                if req and self._sideband_base:
                    write_sideband_batch_descriptors(
                        self._sideband_base,
                        [(req.seq_id, self._feed_pos(req))])
                cmd = self._cmd_writer.run_attention(
                    gpu=gpu, layer_idx=layer, num_seqs=1,
                    emit_checkpoint=1,
                    emit_gating=1 if is_moe else 0,
                    store_gating=1 if is_moe else 0,
                )
            if is_moe and self._sideband_base:
                # Hold the gate until _handle_compute_done captures this
                # command's export (or a CMP_ERROR releases it).
                self._gating_attn_inflight = cmd.cmd_seq
        elif op == WorkOperation.EXPERT_FFN:
            if layer >= self._first_moe_layer:
                # Production routed-MoE seam (#91, mirrors the golden and
                # the MTP chain): forward the routing export published by
                # this layer's RUN_ATTENTION to E_CMD_FETCH_AND_RUN_MOE —
                # entries in gating selection-rank order (INV-10c-2), EP
                # split by round-robin expert ownership.  NOT the
                # deprecated resident-only D_B_CMD_RUN_MOE.
                # TD-ORCH-ROUTING-EXPORT-MULTI: read the copy CAPTURED at
                # this layer's attention completion (_capture_routing_
                # export), never the live sideband slot — another request's
                # same-layer attention may have overwritten the slot
                # between completion and this dispatch (the layer check
                # alone cannot catch a same-layer clobber).
                entries: list[tuple[int, int, int, int]] = []
                cap = chain_req.captured_routing if chain_req else None
                if cap is not None and cap[0] == layer:
                    owners = self._ep_gpus
                    # Dedup, first-seen order (INV-10c-2 selection-rank
                    # priority preserved): a multi-token export (batched
                    # verify) repeats experts across rows — the fetch list
                    # is the union, one entry per expert, exactly like the
                    # superchunk path's sc_union.  Single-token exports are
                    # already unique, so this is the identity there.
                    # Ownership round-robins over the EP owner set
                    # (OrchestratorConfig.ep_gpu_indices) — a draft GPU
                    # hosting no expert cache must never own experts
                    # (INV-FAR deadline-burn otherwise).
                    seen: set[int] = set()
                    for e in cap[1]:
                        if e not in seen:
                            seen.add(e)
                            entries.append(
                                (layer, e, CacheZone.STABLE.value,
                                 owners[e % len(owners)]))
                    if self._sideband_base:
                        write_sideband_expert_prefetch(
                            self._sideband_base, entries,
                        )
                cmd = self._cmd_writer.fetch_and_run_moe(
                    gpu=gpu, layer_idx=layer,
                    num_seqs=chain_req.vb_len if vb else 1,
                    expert_count=len(entries),
                    timeout_us=_FETCH_AND_RUN_TIMEOUT_US,
                )
            else:
                # Dense FFN layers (< first_k_dense_replace) stay on
                # RUN_MOE — dispatch_moe_internal's dense early-out.
                cmd = self._cmd_writer.run_moe(
                    gpu=gpu, layer_idx=layer,
                    num_seqs=chain_req.vb_len if vb else 1,
                    store_gating_output=0 if vb else 1,
                    emit_checkpoint=0 if vb else 1,
                )
        elif op == WorkOperation.OUTPUT_HEAD:
            # Batched verify: ONE head over all V rows; the daemon reads
            # back all V argmax tokens (u32 each) so the acceptance rule
            # runs on one completion.
            cmd = self._cmd_writer.output_head(
                gpu=gpu, num_tokens=chain_req.vb_len if vb else 1,
                input_buf=self._metadata.hidden_buf_id,
                output_buf=self._metadata.logits_buf_id,
                readback=True, compute_confidence=True,
            )
        else:
            return

        self._ring_writer.write_struct(cmd)
        self._commands_this_cycle += 1
        self._cmd_seq_map[cmd.cmd_seq] = (item.request_id, item)
        self._work_queue.update_status(item, WorkStatus.DISPATCHED)

    # ======================================================================
    # Phase 6: YIELD
    # ======================================================================

    def _phase_yield(self, cycle_start_ns: int) -> None:
        """Record cycle metrics, enforce cycle budget, optionally idle-wait."""
        elapsed_ns = time.perf_counter_ns() - cycle_start_ns
        elapsed_us = elapsed_ns / 1000.0

        self._last_metrics = CycleMetrics(
            elapsed_us=elapsed_us,
            completions_processed=self._completions_this_cycle,
            commands_dispatched=self._commands_this_cycle,
        )

        # Periodic origin transfer expiry sweep (every 1000 cycles).
        # Stale entries are harmless (inflight check filters them at cancel
        # time) but waste memory if never cleaned.
        if self._cycle_count % 1000 == 0 and self._origin_birth_cycle:
            threshold = (
                self._cycle_count
                - self._config.transfer_origin_expiry_cycles
            )
            expired = [
                o for o, c in self._origin_birth_cycle.items()
                if c < threshold
            ]
            for o in expired:
                self._origin_transfers.pop(o, None)
                self._origin_birth_cycle.pop(o, None)

        # Budget enforcement: if cycle finished early, wait remainder
        # while still checking for completions (responsive wait).
        # cycle_budget_us=0 disables (elapsed always >= 0).
        budget_ns = int(self._config.cycle_budget_us * 1000)
        if elapsed_ns < budget_ns:
            target_ns = cycle_start_ns + budget_ns
            while time.perf_counter_ns() < target_ns:
                if not self._completion_reader.is_empty():
                    break  # completion arrived, start next cycle early

        # Idle wait: if no pending work, spin for incoming completions
        if not self._has_pending_work():
            idle_budget_ns = int(self._config.max_idle_wait_us * 1000)
            idle_target_ns = time.perf_counter_ns() + idle_budget_ns
            while time.perf_counter_ns() < idle_target_ns:
                if not self._completion_reader.is_empty():
                    break

    def _has_pending_work(self) -> bool:
        """Check if there is any work pending (items or queued requests)."""
        return bool(
            self._work_queue.pending_items()
            or self._request_queue
        )

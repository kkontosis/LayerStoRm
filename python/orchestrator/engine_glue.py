"""Real-engine glue (#91 / TD-ORCH-ENGINE-DECODE-E2E).

Builds a fully-wired OrchestratorLoop on top of a LIVE engine started via
the ``layerstorm_engine`` pybind module: the SPSC command/completion rings,
the seqlock StateSnapshot, and the sideband region are attached directly to
the EngineInfo pointers — no further pybind calls on the hot path (the
daemon thread never acquires the GIL).

This is the first construction path where the Python orchestrator drives
the REAL engine end-to-end (previously only mock/scripted-daemon wiring
existed in tests).  Default module configuration is deliberately minimal:

  - Speculation OFF by default (UtilityScorer strategy="fixed",
    static_depth=0) — plain greedy autoregressive decode, the GGUF
    golden's teacher-forced schedule.
  - All prefetch predictors (PreScope/PROBE/MoE-SpeQ) disabled — the
    FETCH_AND_RUN_MOE seam fetches the routed experts just-in-time, which
    is the golden-parity correctness baseline.  Predictive prefetch layers
    on top without changing tokens.
  - host_buf_base == sideband base: every daemon readback offset
    (OUTPUT_HEAD token readback, DSpark draft ids, spec checkpoints) is
    sideband-relative by the IPC contract.

Usage::

    import layerstorm_engine
    from orchestrator.engine_glue import build_engine_loop

    info = layerstorm_engine.start_engine(config_path)   # real backends
    loop = build_engine_loop(info,
                             buffer_ids=layerstorm_engine.query_buffer_ids())
    loop.submit_request(InferenceRequest(...))
    while loop.requests or ...:
        loop.run_one_cycle()
    layerstorm_engine.stop_engine()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from orchestrator.command_writer import CommandWriter, CompletionReader
from orchestrator.config_hot_reload import ConfigHotReload
from orchestrator.draft_combiner import DraftCombiner
from orchestrator.dspark_draft import DsparkDraft, DsparkDraftConfig
from orchestrator.eviction_policy import EvictionPolicy
from orchestrator.expert_placement import ExpertPlacement, ExpertPlacementConfig
from orchestrator.gpu_load_balancer import GpuLoadBalancer, GpuLoadBalancerConfig
from orchestrator.layer_skip import LayerSkip, LayerSkipConfig
from orchestrator.loop.orchestrator_loop import (
    OrchestratorConfig,
    OrchestratorLoop,
)
from orchestrator.moe_speq import MoeSpeq, MoeSpeqConfig
from orchestrator.mtp_draft import MtpDraft, MtpDraftConfig
from orchestrator.online_calibrator import OnlineCalibrator
from orchestrator.performance_objective import PerformanceObjective
from orchestrator.placement_optimizer import (
    PlacementOptimizer,
    PlacementOptimizerConfig,
)
from orchestrator.prefetch_fuser import PrefetchFuser
from orchestrator.prescope import PreScope, PrescopeConfig
from orchestrator.probe import Probe, ProbeConfig
from orchestrator.prompt_lookup import PromptLookup
from orchestrator.reasoning_mode import ReasoningMode
from orchestrator.scheduler import Scheduler
from orchestrator.self_speculative import SelfSpeculative, SelfSpeculativeConfig
from orchestrator.speculative_prefetch import SpeculativePrefetch
from orchestrator.spsc_ring import SpscRingReader, SpscRingWriter
from orchestrator.state_reader import StateReader
from orchestrator.transfer_scheduler import (
    TransferScheduler,
    TransferSchedulerConfig,
)
from orchestrator.types import EngineMetadata, GpuConfig
from orchestrator.utility_scorer import UtilityScorer, UtilityScorerConfig
from orchestrator.verifier import Verifier
from orchestrator.work_queue import WorkQueue


@dataclass
class EngineLoopHandles:
    """Live IPC handles owned by a built loop (for tests / diagnostics)."""
    metadata: EngineMetadata
    cmd_writer: CommandWriter
    ring_writer: SpscRingWriter
    completion_reader: CompletionReader
    state_reader: StateReader
    sideband_base: int


def metadata_from_engine_info(
    info: Any,
    *,
    buffer_ids: Mapping[str, int] | None = None,
    eos_token_ids: tuple[int, ...] = (),
    vocab_size: int = 0,
    think_start_token_id: int = -1,
    think_end_token_id: int = -2,
) -> EngineMetadata:
    """Translate a pybind EngineInfo into the orchestrator's EngineMetadata.

    ``buffer_ids`` is the {name: buf_id} map from
    ``layerstorm_engine.query_buffer_ids()`` — the forward-pass seams need
    the hidden-state attn buffer (EMBEDDING output / OUTPUT_HEAD input)
    and the logits scratch (OUTPUT_HEAD output).  Prefix-matched because
    the registry names carry rank/position suffixes.
    """
    hidden_buf_id = 0
    logits_buf_id = 0
    if buffer_ids:
        for name, bid in buffer_ids.items():
            if name.startswith("hidden_state.attn.rank0"):
                hidden_buf_id = int(bid)
            elif name.startswith("logits_scratch.pos0"):
                logits_buf_id = int(bid)
    gpus = tuple(
        GpuConfig(position=i, gpu_type="gpu", is_tp=True,
                  vram_bytes=1 << 30, compute_weight=1.0)
        for i in range(int(info.num_gpus))
    )
    return EngineMetadata(
        num_gpus=int(info.num_gpus),
        num_moe_layers=int(info.num_moe_layers),
        num_experts=int(info.num_experts),
        num_layers=int(info.num_layers),
        expert_bytes=int(info.expert_bytes),
        kv_bytes_per_page=int(info.kv_bytes_per_page),
        num_expert_devices=int(info.num_expert_devices),
        gpus=gpus,
        eos_token_ids=eos_token_ids,
        think_start_token_id=think_start_token_id,
        think_end_token_id=think_end_token_id,
        hidden_buf_id=hidden_buf_id,
        logits_buf_id=logits_buf_id,
        # TD-VOCAB-AUTODETECT: the engine's resolved width (weights-derived
        # or cross-checked) wins over the caller-supplied config value.
        vocab_size=(int(getattr(info, "vocab_size", 0) or 0) or vocab_size),
        # TD-PREFILL-MOE-BIG: elastic superchunk capacity — enables the
        # loop's superchunk prefill (FETCH_AND_RUN_MOE_BIG per layer).
        moe_batch_capacity=int(getattr(info, "moe_batch_capacity", 0)),
    )


def ep_gpu_indices_from_config(
    config: Mapping[str, Any] | None,
    fallback: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Resolve OrchestratorConfig.ep_gpu_indices from an engine-config dict.

    Schema source: ``_internal-orchestrator.ep_gpu_indices``
    (config/internal-schemas/orchestrator.schema.json).  When the field is
    present and non-empty, it wins; otherwise the DRIVER'S derived value
    (``fallback``, e.g. the TP pair for a dspark topology) is used —
    absent/empty is byte-identical to prior behavior.  HAZARD (INV-FAR
    deadline-burn): the set must exclude GPUs that host no expert cache
    (e.g. a dspark draft GPU) — an expert round-robined onto such a GPU
    never becomes ready and every MoE layer burns the full FETCH_AND_RUN
    fetch deadline.
    """
    section = (config or {}).get("_internal-orchestrator") or {}
    value = section.get("ep_gpu_indices")
    if value:
        return tuple(int(g) for g in value)
    return tuple(int(g) for g in fallback)


def build_engine_loop(
    info: Any,
    *,
    buffer_ids: Mapping[str, int] | None = None,
    eos_token_ids: tuple[int, ...] = (),
    vocab_size: int = 0,
    think_start_token_id: int = -1,
    think_end_token_id: int = -2,
    speculation_depth: int = 0,
    mtp_config: MtpDraftConfig | None = None,
    num_mtp_layers: int = 0,
    dspark_config: DsparkDraftConfig | None = None,
    orchestrator_config: OrchestratorConfig | None = None,
) -> tuple[OrchestratorLoop, EngineLoopHandles]:
    """Build an OrchestratorLoop attached to a live engine's IPC surfaces.

    ``speculation_depth`` 0 (default) = plain greedy AR decode.  > 0 arms
    prompt-lookup speculation (sequential early-stop verify on the main
    sequence); pass ``mtp_config``/``num_mtp_layers`` or ``dspark_config``
    to arm the corresponding drafter (the engine config must match —
    speculation.method selects one drafter).

    ``eos_token_ids``/``vocab_size``/``think_*_token_id`` are
    tokenizer-derived (the engine has no tokenizer) — the serve entrypoint
    (#71, python/cli/serve.py) fills them from the model directory.
    """
    metadata = metadata_from_engine_info(
        info, buffer_ids=buffer_ids, eos_token_ids=eos_token_ids,
        vocab_size=vocab_size,
        think_start_token_id=think_start_token_id,
        think_end_token_id=think_end_token_id,
    )

    ipc_base = int(info.ipc_base)
    sideband_base = ipc_base + int(info.sideband_offset)

    # -- IPC attach (zero-copy over the heap-allocated engine block) ------
    ring_writer = SpscRingWriter(
        ipc_base + int(info.cmd_ring_offset),
        int(info.cmd_ring_slots), int(info.cmd_slot_bytes),
    )
    completion_reader = CompletionReader(SpscRingReader(
        ipc_base + int(info.cmp_ring_offset),
        int(info.cmp_ring_slots), int(info.cmp_slot_bytes),
    ))
    state_reader = StateReader(ipc_base + int(info.state_offset))
    # cmd_seq 0 is reserved-ish (the C++ drivers start at 1); start above 0
    # so completions never carry cmd_seq 0.
    cmd_writer = CommandWriter(initial_seq=1)

    # -- Module wiring (minimal correctness-first defaults) ---------------
    work_queue = WorkQueue()
    scheduler = Scheduler()
    gpu_assigner = GpuLoadBalancer(GpuLoadBalancerConfig(gpus=metadata.gpus))
    transfer_scheduler = TransferScheduler(TransferSchedulerConfig(
        num_gpus=metadata.num_gpus, expert_bytes=metadata.expert_bytes,
    ))
    eviction_policy = EvictionPolicy()
    expert_placement = ExpertPlacement(ExpertPlacementConfig(
        num_moe_layers=metadata.num_moe_layers,
        num_experts=metadata.num_experts,
        cache_gpu_indices=list(range(metadata.num_gpus)),
    ))
    placement_optimizer = PlacementOptimizer(
        PlacementOptimizerConfig(), expert_placement,
    )

    prescope = PreScope(metadata, PrescopeConfig(enabled=False))
    probe = Probe(metadata, ProbeConfig(enabled=False))
    moe_speq = MoeSpeq(metadata, MoeSpeqConfig(enabled=False))
    prefetch_fuser = PrefetchFuser()
    speculative_prefetch = SpeculativePrefetch(
        num_gpus=metadata.num_gpus, expert_bytes=metadata.expert_bytes,
    )

    # Fixed-depth scoring: depth 0 = pure AR (the e2e gate), > 0 arms
    # speculation with a deterministic depth (no cascade exploration —
    # keeps real-engine runs reproducible).
    utility_scorer = UtilityScorer(UtilityScorerConfig(
        strategy="fixed", static_depth=max(0, speculation_depth),
    ))
    first_moe_layer = metadata.num_layers - metadata.num_moe_layers
    verifier = Verifier(
        num_layers=metadata.num_layers,
        num_moe_layers=metadata.num_moe_layers,
        first_moe_layer=first_moe_layer,
        num_experts=metadata.num_experts,
    )
    prompt_lookup = PromptLookup()
    draft_combiner = DraftCombiner()

    mtp_draft = MtpDraft(
        mtp_config or MtpDraftConfig(enabled=False),
        num_mtp_layers=num_mtp_layers,
        num_layers=metadata.num_layers,
    )
    dspark_draft = DsparkDraft(dspark_config or DsparkDraftConfig())
    self_speculative = SelfSpeculative(SelfSpeculativeConfig(enabled=False))
    layer_skip = LayerSkip(LayerSkipConfig(enabled=False))
    reasoning_mode = ReasoningMode()
    online_calibrator = OnlineCalibrator()
    performance_objective = PerformanceObjective()

    loop = OrchestratorLoop(
        config=orchestrator_config or OrchestratorConfig(),
        metadata=metadata,
        cmd_writer=cmd_writer,
        ring_writer=ring_writer,
        completion_reader=completion_reader,
        state_reader=state_reader,
        work_queue=work_queue,
        scheduler=scheduler,
        gpu_assigner=gpu_assigner,
        transfer_scheduler=transfer_scheduler,
        eviction_policy=eviction_policy,
        expert_placement=expert_placement,
        placement_optimizer=placement_optimizer,
        prescope=prescope,
        probe=probe,
        moe_speq=moe_speq,
        prefetch_fuser=prefetch_fuser,
        speculative_prefetch=speculative_prefetch,
        utility_scorer=utility_scorer,
        verifier=verifier,
        prompt_lookup=prompt_lookup,
        draft_combiner=draft_combiner,
        mtp_draft=mtp_draft,
        self_speculative=self_speculative,
        layer_skip=layer_skip,
        reasoning_mode=reasoning_mode,
        online_calibrator=online_calibrator,
        performance_objective=performance_objective,
        hot_config=ConfigHotReload(),
        # IPC contract: every daemon readback offset (OUTPUT_HEAD token
        # readback, DSpark draft ids, spec checkpoints) is relative to the
        # SIDEBAND base — host_buf_base must alias it.
        host_buf_base=sideband_base,
        sideband_base=sideband_base,
        dspark_draft=dspark_draft,
    )
    handles = EngineLoopHandles(
        metadata=metadata,
        cmd_writer=cmd_writer,
        ring_writer=ring_writer,
        completion_reader=completion_reader,
        state_reader=state_reader,
        sideband_base=sideband_base,
    )
    return loop, handles

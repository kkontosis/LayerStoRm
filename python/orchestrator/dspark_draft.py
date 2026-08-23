"""DSpark draft — DFlash-backbone + Markov-head whole-block draft planner.

Orchestrator-side planner for DSpark speculative drafting (DSP-5), the
sibling of mtp_draft.py.  Owns the draft POLICY (γ clamp, anchor position
convention) and acceptance tracking.  Does not dispatch GPU commands
directly — the orchestrator loop (loop/speculation.py) uses this module's
planning output to build the ONE D_CMD_RUN_DSPARK_STEP per round (contrast
MTP's per-token chained steps: DFlash draft latency is ~independent of γ).

Engine contract (DSP-3/DSP-4, mirrors DsparkSpeculationMethod in C++):
  - The draft's context KV is built AUTOMATICALLY from the target's aux
    hiddens on every non-draft target step (INV-DSPARK-AUX) — the loop
    never issues explicit warm/catch-up steps (unlike MTP's INV-MTP-KV
    schedule) and never forks KV.
  - One D_CMD_RUN_DSPARK_STEP = the whole γ block: backbone forward off
    the ingested context at anchor_pos (= fed-token count) + the on-device
    sequential Markov head.  The γ sampled ids are host-visible in the
    sideband readback scratch when the completion fires.
  - Rejected drafts are never fed (sequential early-stop verification), so
    rejection needs no rewind: the next round simply re-drafts at the
    shorter anchor.

Config maps speculation.dspark (schema.json) + _internal-dspark.

Reference: DSpark paper (Alg. 1) + vLLM spec_decode/dspark/speculator.py;
see spec/DSpark_NOTES.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from orchestrator.dspark_calibration import ConfidenceRound, apply_temperature


@dataclass(frozen=True)
class DsparkDraftConfig:
    """Mirrors speculation.dspark (user-facing) + _internal-dspark knobs.

    ``enabled`` is derived from speculation.method == "dspark" (the dspark
    section has no enabled flag — method selection is the switch).
    ``sts_temperatures`` (DSP-7): γ per-position STS temperatures applied
    to the raw c_k BEFORE the cumprod (empty = identity — raw/uncalibrated;
    fitted offline by dspark_calibration.fit_sts_temperatures).
    ``epm_dump_dir`` (EPM-1): training-data dump directory — must mirror
    the engine's speculation.dspark.epm_dump_dir so the Python-side block
    manifest (accepted length, c_k, tokens) lands beside the daemon's raw
    EPMB/EPMR records.  Empty (default) = OFF; the LS_EPM_DUMP env var
    overrides on BOTH sides (epm_manifest.effective_epm_dump_dir).
    """
    enabled: bool = False
    block_size: int = 8              # γ ceiling (checkpoint block size)
    speculative_tokens: int = 7      # γ — tokens proposed per round
    confidence_enabled: bool = False  # DSP-6 trained confidence head
    confidence_threshold: float = 0.5
    scheduler_mode: str = "off"      # off | static_threshold | throughput
    acceptance_ema_alpha: float = 0.3
    sts_temperatures: tuple[float, ...] = ()  # DSP-7 (γ-length or empty)
    confidence_trace_capacity: int = 4096     # DSP-7 fit-trace rounds kept
    epm_dump_dir: str = ""           # EPM-1 manifest dir ("" = off)


@dataclass
class DsparkDraftResult:
    """One whole-γ-block draft round (token ids in position order).

    ``confidences`` (DSP-6, INV-DSPARK-CONF): the trained confidence head's
    raw per-position survival probabilities c_k ∈ (0,1) — the CONDITIONAL
    probability token k survives verification given all predecessors
    accepted, read back from the completion sideband alongside the ids.
    Empty when speculation.dspark.confidence_enabled is off.  Raw
    (pre-STS-calibration — DSP-7 applies per-position temperatures before
    the cumprod).
    """
    tokens: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    source: str = "dspark"

    @property
    def depth(self) -> int:
        return len(self.tokens)


class DsparkDraft:
    """Draft policy + acceptance tracking for the DSpark drafter."""

    def __init__(self, config: DsparkDraftConfig | None = None) -> None:
        self._config = config or DsparkDraftConfig()
        if any(t <= 0.0 for t in self._config.sts_temperatures):
            raise ValueError("speculation.dspark.sts_temperatures must all "
                             "be > 0 (identity = 1.0)")
        self._total_rounds: int = 0
        self._total_proposed: int = 0
        self._total_accepted: int = 0
        self._ema_acceptance_rate: float = 0.0
        self._ema_initialized: bool = False
        self._confidence_trace: deque[ConfidenceRound] = deque(
            maxlen=max(self._config.confidence_trace_capacity, 1))

    # -- Policy ------------------------------------------------------------

    @property
    def gamma(self) -> int:
        """Tokens proposed per round: speculative_tokens clamped to the
        checkpoint block size (validate_dspark enforces the same in C++)."""
        return max(1, min(self._config.speculative_tokens,
                          self._config.block_size))

    def plan_num_query(self, depth: int) -> int:
        """γ for this round, clamped by the utility-recommended depth."""
        return max(0, min(depth, self.gamma))

    def anchor_pos(self, prompt_len: int, tokens_generated: int) -> int:
        """Slot-0 position for the next round == the fed-token count.

        Matches the AR dispatch convention (#91 fixed position math: the
        next feed goes at prompt_len + tokens_generated - 1 because the
        newest committed token is always still pending) AND the runtime's
        ingested context length (one context row per fed position,
        INV-DSPARK-AUX) — the backbone attends over [0, anchor_pos).
        """
        return max(prompt_len + tokens_generated - 1, 0)

    @property
    def confidence_enabled(self) -> bool:
        """The DSP-6 trained confidence head is on (readback contract:
        the completion sideband carries γ f32 c_k after the γ i32 ids)."""
        return self._config.confidence_enabled

    @property
    def epm_dump_dir(self) -> str:
        """EPM-1 configured dump dir ("" = off; env override is applied by
        epm_manifest.effective_epm_dump_dir at the loop, not here)."""
        return self._config.epm_dump_dir

    def confidence_proxy(self) -> float:
        """Per-token p(accept) FALLBACK proxy for the DraftCombiner.

        Used only when the trained confidence head is off (DSP-6) or a
        round carried no confidences — the EMA acceptance rate stands in
        (0.5 before any round completes, the uninformed prior).
        """
        return self._ema_acceptance_rate if self._ema_initialized else 0.5

    def survival_confidences(
            self, result: DsparkDraftResult | None) -> list[float] | None:
        """Cumulative survival a_j = Π_{i≤j} c_i per draft position (DSP-6).

        The trained c_k are CONDITIONAL survival probabilities
        (INV-DSPARK-CONF), so position j's unconditional p(accept) is the
        cumprod over the prefix — monotone non-increasing in j; what the
        DraftCombiner uses as p_accept and what the DSP-9 scheduler will
        consume.  DSP-7: when ``sts_temperatures`` is set, each c_k is
        temperature-scaled on the LOGIT (order-preserving) BEFORE the
        cumprod — calibrated magnitudes; empty temps = raw/uncalibrated.
        None when the head is off or the round carried no usable
        confidences (the caller falls back to confidence_proxy()).
        """
        if (not self._config.confidence_enabled or result is None
                or not result.confidences
                or len(result.confidences) < result.depth):
            return None
        temps = self._config.sts_temperatures
        out: list[float] = []
        a = 1.0
        for k, c in enumerate(result.confidences[:result.depth]):
            c = min(max(c, 0.0), 1.0)
            if k < len(temps) and temps[k] != 1.0:
                c = apply_temperature(c, temps[k])
            a *= c
            out.append(a)
        return out

    @property
    def confidence_trace(self) -> list[ConfidenceRound]:
        """Recorded (c_k, accepted) rounds — the DSP-7 STS fit input
        (bounded by confidence_trace_capacity, oldest dropped)."""
        return list(self._confidence_trace)

    # -- Acceptance tracking -------------------------------------------------

    def record_result(self, result: DsparkDraftResult,
                      num_accepted: int) -> None:
        depth = result.depth
        self._total_rounds += 1
        self._total_proposed += depth
        self._total_accepted += min(max(num_accepted, 0), depth)

        # DSP-7: keep the RAW (c_k, accepted) round for the offline STS
        # fit (labels are the fully-observed prefix acceptance).
        if (self._config.confidence_enabled and depth > 0
                and len(result.confidences) >= depth):
            self._confidence_trace.append(ConfidenceRound(
                confidences=tuple(result.confidences[:depth]),
                accepted=min(max(num_accepted, 0), depth),
            ))

        if depth > 0:
            batch_rate = min(max(num_accepted, 0), depth) / depth
            if not self._ema_initialized:
                self._ema_acceptance_rate = batch_rate
                self._ema_initialized = True
            else:
                a = self._config.acceptance_ema_alpha
                self._ema_acceptance_rate = (
                    a * batch_rate + (1.0 - a) * self._ema_acceptance_rate
                )

    @property
    def acceptance_rate(self) -> float:
        return self._ema_acceptance_rate

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    @property
    def total_rounds(self) -> int:
        return self._total_rounds

    @property
    def total_proposed(self) -> int:
        return self._total_proposed

    @property
    def total_accepted(self) -> int:
        return self._total_accepted

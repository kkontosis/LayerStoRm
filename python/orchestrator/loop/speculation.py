"""Speculation lifecycle — 5-state machine for speculative decoding.

Implements the AUTOREGRESSIVE -> DRAFTING -> PREFETCHING_VERIFY ->
VERIFYING -> ACCEPTING -> AUTOREGRESSIVE cycle within the orchestrator.

State machine (per request):
  - AUTOREGRESSIVE: normal token-at-a-time generation.  OUTPUT_HEAD
    completion triggers a speculation attempt if depth > 0.
  - DRAFTING: generate draft tokens.  Three sources, checked in order:
      1. Prompt lookup (CPU-only, zero GPU cost, same cycle — INV-0.8d).
      2. MTP steps (production-seam D_CMD_MTP_PROJECT chain, multi-cycle;
         see "MTP flow" below — TD-MTP-PY-LOOP-KV / TD-MTP-FUSED-RUNMOE).
      3. Self-spec steps (fused D_CMD_RUN_SELF_SPEC_FORWARD, multi-cycle).
    All sources feed into _finish_drafting → DraftCombiner.combine().
  - PREFETCHING_VERIFY: loading experts needed for verification.  If
    draft_gating is available (from self-spec), uses SP-MoE targeted
    prefetching.  Otherwise instantaneous.  (Skipped by the MTP flow.)
  - VERIFYING: three shapes —
      a. MTP batched flow (mtp_batched_verify, the throughput-bearing
         default): ONE teacher-forced V-token forward on the MAIN
         sequence (V = K drafts + 1) through the work queue — chunked-
         prefill-shaped attention + FETCH_AND_RUN_MOE over the batch +
         a multi-token OUTPUT_HEAD argmax readback; the greedy
         acceptance rule runs on one completion.  Rejected positions'
         main-layer KV is scratch above the committed frontier,
         overwritten in place by later feeds before any causal window
         can reach it — no fork, no rewind (see
         _start_batched_verification).
      b. MTP sequential flow (mtp_sequential_verify, the
         speculation.mtp.batched_verify=false fallback and the DSpark/
         prompt-lookup shape): sequential early-stop teacher-forced
         feeds on the MAIN sequence (MtpLossless pattern); each feed is
         a normal AR forward through the work queue.
      c. Legacy batched flow: full-model verification pass on the KV fork
         dispatched directly to the ring.  Intermediates tracked in
         _verify_cmd_seqs for error recovery (TD-21).  Self-spec only
         (TD-SPEC-VERIFY-FORK-KV).
  - ACCEPTING: compare draft vs verification tokens, accept/reject,
    free draft KV pages, record statistics, return to AUTOREGRESSIVE.
    (Legacy batched flow only; the MTP flow accepts per feed.)

MTP flow (TD-MTP-PY-LOOP-KV — mirrors the Glm52GgufGolden.MtpLossless
C++ driver, the reference schedule for lossless MTP speculation):
  - One MTP step = D_CMD_MTP_PROJECT → RUN_ATTENTION(mtp layer,
    emit/store gating, KV appended at the step's position of the MAIN
    seq) → E_CMD_FETCH_AND_RUN_MOE(mtp layer) → OUTPUT_HEAD(mtp_head).
    This is the production seam — the fused D_CMD_RUN_MTP_STEP is NOT
    used (its internal MoE is resident-only, TD-MTP-FUSED-RUNMOE/#89).
  - INV-MTP-KV: the MTP layer's KV must be gap-free with true-token
    embeddings below the draft anchor.  Every main-model feed at
    position p whose successor token is known is followed by one MTP
    step at p embedding that successor: a CATCH-UP step after each
    non-drafting AR feed and after each accepted verification feed
    (except the round's last — the next round's draft step 0 covers it,
    and it must see the un-clobbered trunk hidden of the anchor).
  - No KV fork: draft MTP steps write only the MTP layer's KV at
    positions >= anchor (scratch above the true-token frontier,
    overwritten by later catch-ups/draft rounds).  Verification feeds
    write main-layer KV for COMMITTED tokens only — a rejected draft
    token is NEVER fed (early stop), so no rewind/copy-back is needed
    and the main sequence keeps the KV of every accepted position.

Key invariants enforced:
  - INV-MTP-KV:  warm/catch-up schedule keeps MTP-layer KV gap-free.
  - INV-MTP-LOSSLESS: only main-model outputs are ever fed/committed —
    speculation changes speed, never tokens.
  - INV-3.4.5: rejected tokens -> cancel downstream + free KV pages.
  - INV-0.8d:  prompt lookup checked first (zero compute cost).
  - INV-4.9b:  draft KV fork freed via E_CMD_SEQ_FREE on completion
    (legacy batched flow).

Methods here are mixed into OrchestratorLoop via _SpeculationMixin.

IMPORTANT: Keep this file richly commented.  The orchestrator loop is the
most complex module in the system and every helper must be self-documenting.
Do not remove comments during edits — add more if anything is unclear.
"""

from __future__ import annotations

import ctypes
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from orchestrator.loop.orchestrator_loop import RequestState

from orchestrator.command_writer import (
    read_sideband_routing_export,
    write_sideband_batch_descriptors,
    write_sideband_expert_prefetch,
)
from orchestrator.dspark_draft import DsparkDraftResult
from orchestrator.mtp_draft import DraftStep, MtpDraftResult
from orchestrator.self_speculative import SelfSpecDraftResult
from orchestrator.shm_protocol import Completion
from orchestrator.types import (
    CacheZone,
    ExpertKey,
    SpeculationState,
    StepLogprobs,
    TokenLogprob,
    VerificationPlan,
    WorkItem,
    WorkOperation,
    WorkStatus,
)
from orchestrator.utility_scorer import UtilityScorer
from orchestrator.verifier import CommandDescriptor, VerificationResult, Verifier


# Correctness gate for the MTP layer's routed experts (golden parity):
# a missing expert degrades the draft like a timeout (wrong draft — still
# lossless, verification rejects it), so allow a generous fetch deadline.
_MTP_FETCH_TIMEOUT_US = 120_000_000  # 120 s


def _encode_skip_mask(skip_set: set[int]) -> tuple[int, int]:
    """Encode a set of layer indices into a 128-bit bitmask (lo, hi)."""
    lo = 0
    hi = 0
    for layer in skip_set:
        if layer < 64:
            lo |= (1 << layer)
        else:
            hi |= (1 << (layer - 64))
    return lo, hi


class _SpeculationMixin:
    """Mixin: speculation lifecycle methods for the orchestrator loop."""

    # ------------------------------------------------------------------
    # OUTPUT_HEAD completion entry point
    # ------------------------------------------------------------------

    def _handle_output_head_done(self, req: RequestState, cmp: Completion) -> None:
        """Route OUTPUT_HEAD completion based on speculation state.

        Checks stop criteria first (EOS, max_tokens, external cancel).
        Then routes by state:
          1. VERIFYING: verification pass result -> accept/reject.
          2. Prefill:   first OUTPUT_HEAD -> transition is_prefill to False.
          3. Decode:    check if speculation should start, else continue AR.
        """
        payload = cmp.payload.compute
        # Try to extract the sampled token ID from the host buffer readback
        # (for a batched verify head this is o1 = the readback's FIRST
        # token — ALWAYS committed: d0 on match, the correction otherwise).
        token_id = self._extract_token_id(payload)
        batched_verify_head = (
            req.speculation_state == SpeculationState.VERIFYING
            and req.mtp_batched_verify
        )
        step_lp: StepLogprobs | None = None
        if token_id is not None:
            req.token_history.append(token_id)
            if req.logprobs is not None:
                # Batched verify readback is V raw u32 tokens — the
                # single-token logprob layout does not apply (bytes 4..8
                # are token o2, not a float): record None like every other
                # speculation-committed token.
                if not batched_verify_head:
                    step_lp = self._extract_logprobs(payload)
                req.logprob_results.append(step_lp)
            if req.on_token is not None:
                req.on_token(req.request_id, token_id, step_lp)

        # Path 1: verification OUTPUT_HEAD completed
        if req.speculation_state == SpeculationState.VERIFYING:
            # MTP batched flow: ONE V-token pass — the readback carries all
            # V argmax tokens; acceptance runs on this single completion.
            if batched_verify_head:
                self._handle_batched_verify_done(req, payload, token_id)
                return
            # MTP flow: each feed is a single-token AR forward on the MAIN
            # sequence — sequential early-stop (MtpLossless pattern).
            if req.mtp_sequential_verify:
                self._handle_sequential_verify_feed(req, token_id)
                return
            # Legacy batched flow (prompt-lookup-only / self-spec on fork)
            vtokens = self._extract_verification_tokens(
                payload, req.verified_depth,
            )
            req.speculation_state = SpeculationState.ACCEPTING
            self._finish_verification(req, vtokens)
            # Check stop criteria after acceptance (tokens may have advanced)
            if self._should_stop(req, token_id):
                self._finalize_request(req)
            return

        # Path 2 & 3: autoregressive OUTPUT_HEAD completed
        req.tokens_generated += 1
        self._tokens_processed += 1
        # Record timestamp for base_time_us computation (TD-22)
        now_ns = time.perf_counter_ns()
        prev_ns = req.last_output_head_ns
        req.last_output_head_ns = now_ns

        # Stop criteria: EOS token, max_tokens, external cancellation
        if self._should_stop(req, token_id):
            self._finalize_request(req)
            return

        # First OUTPUT_HEAD after prefill -> transition to decode mode
        if req.is_prefill:
            req.is_prefill = False
            # INV-MTP-KV catch-up: warm the MTP layer's KV at the last fed
            # position, embedding the just-produced token (its true
            # successor).  The trunk hidden of that position is still in
            # the attn buffers — OUTPUT_HEAD reads, never clobbers.  The
            # next EMBEDDING is deferred until the catch-up completes (it
            # would overwrite the hidden the MTP projection consumes).
            if self._mtp_active() and token_id is not None:
                self._begin_mtp_step(
                    req, pos=self._mtp_anchor_pos(req),
                    embed_token=token_id, mode="catchup",
                )
                return
            self._enqueue_next_embedding(req)
            return

        # Decode token: check if we should speculate
        depth = self._utility_scorer.recommended_depth()
        if depth > 0:
            req.speculation_depth = depth
            req.speculation_state = SpeculationState.DRAFTING
            self._start_drafting(req)
            return

        # depth == 0: continue autoregressive.  With MTP active, first run
        # a catch-up MTP step at the just-fed position (INV-MTP-KV) — when
        # a draft round starts instead, its step 0 covers that position
        # (it embeds the same true token), so no catch-up is needed there.
        if self._mtp_active() and token_id is not None:
            self._begin_mtp_step(
                req, pos=self._mtp_anchor_pos(req),
                embed_token=token_id, mode="catchup",
            )
            return
        self._enqueue_next_embedding(req)

    def _should_stop(self, req: RequestState, token_id: int | None) -> bool:
        """Check the three stop criteria for a request.

        Returns True if generation should terminate:
          1. EOS token detected (token_id in metadata.eos_token_ids).
          2. Max tokens reached (tokens_generated >= max_tokens, when > 0).
          3. External cancellation (cancelled flag set by cancel_request).
        """
        if req.cancelled:
            return True
        if (
            token_id is not None
            and self._metadata.eos_token_ids
            and token_id in self._metadata.eos_token_ids
        ):
            return True
        if req.max_tokens > 0 and req.tokens_generated >= req.max_tokens:
            return True
        return False

    # ------------------------------------------------------------------
    # Token extraction from host buffer
    # ------------------------------------------------------------------

    def _extract_token_id(self, payload) -> int | None:
        """Read a single uint32 token ID from the OUTPUT_HEAD readback buffer."""
        if payload.data_bytes < 4 or self._host_buf_base == 0:
            return None
        addr = self._host_buf_base + payload.host_buf_offset
        return int((ctypes.c_uint32 * 1).from_address(addr)[0])

    def _extract_verification_tokens(
        self, payload, depth: int,
    ) -> list[int]:
        """Read K uint32 token IDs from the verification OUTPUT_HEAD readback."""
        if self._host_buf_base == 0 or payload.data_bytes < depth * 4:
            return []
        addr = self._host_buf_base + payload.host_buf_offset
        arr = (ctypes.c_uint32 * depth).from_address(addr)
        return [int(arr[i]) for i in range(depth)]

    def _extract_logprobs(self, payload) -> StepLogprobs | None:
        """Read logprobs from host buffer when available.

        Readback layout (dynamic — daemon writes what it has):
          [token_id: u32][logprob: f32][num_top: u32]
          [top_token_0: u32, top_logprob_0: f32] ...

        Returns None when daemon didn't provide logprobs (data_bytes <= 4).
        """
        if self._host_buf_base == 0 or payload.data_bytes <= 4:
            return None
        base = self._host_buf_base + payload.host_buf_offset
        token_id = int((ctypes.c_uint32 * 1).from_address(base)[0])
        if payload.data_bytes < 12:
            logprob = float((ctypes.c_float * 1).from_address(base + 4)[0])
            return StepLogprobs(
                token=TokenLogprob(token_id=token_id, logprob=logprob),
                top_logprobs=(),
            )
        logprob = float((ctypes.c_float * 1).from_address(base + 4)[0])
        num_top = int((ctypes.c_uint32 * 1).from_address(base + 8)[0])
        available = (payload.data_bytes - 12) // 8
        num_top = min(num_top, available)
        tops: list[TokenLogprob] = []
        off = base + 12
        for _ in range(num_top):
            tid = int((ctypes.c_uint32 * 1).from_address(off)[0])
            lp = float((ctypes.c_float * 1).from_address(off + 4)[0])
            tops.append(TokenLogprob(token_id=tid, logprob=lp))
            off += 8
        return StepLogprobs(
            token=TokenLogprob(token_id=token_id, logprob=logprob),
            top_logprobs=tuple(tops),
        )

    # ------------------------------------------------------------------
    # Drafting: fork KV, prompt lookup, combine
    # ------------------------------------------------------------------

    def _start_drafting(self, req: RequestState) -> None:
        """Begin draft token generation.

        DSpark flow (DSP-5, no fork, no catch-ups — the aux export builds
        the draft context KV automatically on every fed target position,
        INV-DSPARK-AUX):
          1. Prompt lookup — zero GPU cost, same cycle (INV-0.8d).
          2. ONE D_CMD_RUN_DSPARK_STEP: whole-γ-block backbone + Markov
             head off the ingested context at anchor_pos = fed-token count
             (multi-cycle; ids arrive in the completion's sideband
             readback).
          3. _finish_dspark_drafting — combine, then sequential
             verification (same MtpLossless pattern).
          At most one of MTP / DSpark may be active (speculation.method
          selects one drafter); DSpark is checked first.

        MTP flow (no fork — TD-MTP-PY-LOOP-KV):
          1. Prompt lookup — zero GPU cost, same cycle (INV-0.8d).
          2. MTP draft chain on the MAIN sequence: production-seam steps
             at positions anchor+j (multi-cycle).  Step 0 embeds the
             newest committed token x off the trunk hidden of the anchor
             (which the just-completed OUTPUT_HEAD left in the attn
             buffers), doubling as the anchor's INV-MTP-KV warm write.
          3. _finish_mtp_drafting — combine, then sequential verification.
          Self-spec is NOT chained after MTP here: it needs the KV fork +
          batched verify (its forward writes approximate main-layer KV),
          which is incompatible with verifying on the main sequence.

        Prompt-lookup-only flow (#92): no fork — combine the lookup
        continuation and verify with sequential early-stop feeds on the
        MAIN sequence (same as MTP/DSpark).

        Legacy fork flow (self-spec only):
          1. Fork the main KV sequence for speculative tokens (the
             self-spec forward writes approximate main-layer KV).
          2. Self-spec steps — multi-cycle GPU dispatch.
          3. _finish_drafting — combine all sources, transition to
             PREFETCHING then batched verify on the fork
             (TD-SPEC-VERIFY-FORK-KV: the fork's verified KV is lost).

        GPU steps are multi-cycle: dispatch command, return, wait for
        completion next cycle.  The sub-state is tracked by which
        *_cmd_seq field is non-None on the request.
        """
        req.drafting_start_ns = time.perf_counter_ns()

        # Prompt lookup: zero-cost CPU N-gram matching (checked first, INV-0.8d)
        req.prompt_lookup_tokens = self._prompt_lookup.lookup(
            req.token_history,
            max_continuation=req.speculation_depth,
        )

        # DSpark drafting (DSP-5: one fused whole-block step, no fork)
        if self._dspark_active():
            if req.dspark_draft_disabled:
                # The daemon declined a prior step for this request
                # (TD-DSPARK-CTX-CAP fail-closed context): combine
                # prompt-lookup only — same route as the nothing-to-draft
                # case; decode stays lossless.
                self._finish_dspark_drafting(req)
            else:
                self._dispatch_dspark_step(req)
            return

        # MTP drafting (production seam, main sequence, no fork)
        if self._mtp_active():
            req.mtp_step = 0
            req.mtp_steps_result = []
            req.mtp_anchor_pos = self._mtp_anchor_pos(req)
            self._dispatch_mtp_step(req)
            return

        # Prompt-lookup-only (#92 / TD-SPEC-VERIFY-FORK-KV): NO fork —
        # verify with sequential early-stop feeds on the MAIN sequence
        # (the MtpLossless pattern shared with MTP/DSpark).  The legacy
        # fork + batched verify wrote the accepted positions' KV into the
        # fork and then freed it, losing the verified KV; sequential
        # feeds are normal AR forwards, so the main sequence keeps every
        # accepted position's KV by construction.
        if not self._self_speculative.is_enabled:
            combined = self._draft_combiner.combine(
                prompt_lookup_tokens=req.prompt_lookup_tokens,
                mtp_result=None,
                self_spec_result=None,
                prompt_lookup_acceptance_rate=(
                    self._prompt_lookup.acceptance_rate),
                max_depth=req.speculation_depth,
                max_verifiable_depth=req.speculation_depth,
                expert_coverage_fraction=1.0,
            )
            if combined.depth == 0:
                # Nothing to verify -> back to plain AR (no fork taken).
                self._abort_draft(req)
                return
            req.combined_draft = combined
            req.draft_complete = True
            self._start_sequential_verification(req)
            return

        # Fork KV sequence (self-spec only): the self-spec draft forward
        # writes APPROXIMATE main-layer KV, so it cannot run on the main
        # sequence — draft tokens go into a separate KV fork that is
        # discarded on rejection.  NOTE (TD-SPEC-VERIFY-FORK-KV, narrowed
        # to self-spec): the batched verification below still writes the
        # accepted positions' KV into the fork and frees it — the main
        # sequence never receives that KV.
        req.draft_seq_id = self._next_seq_id
        self._next_seq_id += 1
        cmd = self._cmd_writer.seq_fork(
            gpu=req.default_gpu,
            src_seq_id=req.seq_id,
            dst_seq_id=req.draft_seq_id,
        )
        self._ring_writer.write_struct(cmd)
        self._commands_this_cycle += 1

        # Self-spec drafting: dispatch first step (multi-cycle)
        req.self_spec_step = 0
        req.self_spec_steps_result = []
        req.self_spec_gating_rows = []
        self._dispatch_self_spec_step(req)

    # ------------------------------------------------------------------
    # MTP production-seam steps (TD-MTP-PY-LOOP-KV, TD-MTP-FUSED-RUNMOE)
    # ------------------------------------------------------------------
    #
    # One MTP step is the 4-command chain the MtpLossless golden drives:
    #   D_CMD_MTP_PROJECT (eh_proj of enorm(Emb(embed_token)) ‖ hnorm(prev
    #   hidden in the attn buffers)) → RUN_ATTENTION(mtp layer, fused gate
    #   + routing export, KV appended at `pos` of the MAIN seq) →
    #   E_CMD_FETCH_AND_RUN_MOE(mtp layer, routed experts from the export)
    #   → OUTPUT_HEAD(mtp_head, readback).
    # Each command's completion advances req.mtp_phase.  Two modes:
    #   "draft":   chained proposal steps at anchor+j (DRAFTING).
    #   "catchup": INV-MTP-KV warm step at a just-fed position embedding
    #              its committed successor token; result token discarded.

    def _mtp_active(self) -> bool:
        """MTP drafting is configured and the model ships an MTP layer."""
        return (self._mtp_draft.is_enabled
                and self._mtp_draft.num_mtp_layers > 0)

    def _mtp_anchor_pos(self, req: RequestState) -> int:
        """Trunk position of the newest committed-and-fed token.

        Matches the AR dispatch convention (#91 fixed position math:
        _dispatch_work_item writes the next feed at prompt_len +
        tokens_generated - 1, so the newest FED position is one less
        again — the newest committed token is always still pending).
        The MTP module's sequence is the main sequence shifted by one
        (DeepSeek V3 §2.2): an MTP step at this position embeds the
        token DESTINED for the next position.
        """
        return max(req.prompt_len + req.tokens_generated - 2, 0)

    def _begin_mtp_step(self, req: RequestState, pos: int,
                        embed_token: int, mode: str,
                        hidden_row: int = 0) -> None:
        """Start one production-seam MTP step (project phase).

        Precondition: the attn buffers hold the right prev_hidden — the
        trunk hidden of a just-completed main forward (draft step 0 /
        catch-up) or the MTP output of the previous chained draft step.
        The caller must not dispatch anything that clobbers the hidden
        buffers (e.g. the next EMBEDDING) until this step completes.

        hidden_row selects the attn_buf ROW of prev_hidden: after a batched
        verify pass row j holds the trunk hidden of fed position base+j
        (MTP steps only write row 0, so higher rows survive a sequential
        catch-up chain).  0 = the single-token-feed default.
        """
        step_idx = req.mtp_step if mode == "draft" else 0
        req.mtp_phase = "project"
        req.mtp_mode = mode
        req.mtp_pos = pos
        req.mtp_embed_token = embed_token
        cmd = self._cmd_writer.mtp_project(
            gpu=req.default_gpu,
            mtp_layer_idx=self._mtp_draft.mtp_layer_idx(step_idx),
            input_token_id=embed_token,
            step_idx=step_idx,
            hidden_row=hidden_row,
        )
        self._track_mtp_cmd(req, cmd)

    def _track_mtp_cmd(self, req: RequestState, cmd) -> None:
        """Send an MTP sub-command and register its completion routing.

        All four phases are tracked in _draft_cmd_seqs so a CMP_ERROR on
        any of them aborts the round (_handle_error path 1), and in
        _cmd_seq_map so _handle_compute_done can resolve the request.
        """
        self._ring_writer.write_struct(cmd)
        self._commands_this_cycle += 1
        req.mtp_cmd_seq = cmd.cmd_seq
        item = WorkItem(
            request_id=req.request_id,
            layer_idx=self._mtp_draft.mtp_layer_idx(
                req.mtp_step if req.mtp_mode == "draft" else 0),
            operation=WorkOperation.OUTPUT_HEAD,
            target_gpu=req.default_gpu,
            status=WorkStatus.DISPATCHED,
            is_speculative=True,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        self._cmd_seq_map[cmd.cmd_seq] = (req.request_id, item)
        self._draft_cmd_seqs[cmd.cmd_seq] = req.request_id

    def _dispatch_mtp_gating_attention(self, req: RequestState) -> None:
        """Dispatch the MTP decoder attention (fused gate + routing export).

        Point the MTP layer's KV append at this step's position of the
        MAIN seq (same batch-descriptor contract as any RUN_ATTENTION),
        then run the attention.  Called from _handle_mtp_completion
        (phase "project") when the routing-export slot gate is free, or
        from _retry_gating_waiters for a step parked in "attn-wait".
        Acquires the gate: the export is read back directly inside this
        command's completion handling (_handle_mtp_completion phase
        "attn"), which also releases the gate (INV-IPC-6b).
        """
        step_idx = req.mtp_step if req.mtp_mode == "draft" else 0
        mtp_layer = self._mtp_draft.mtp_layer_idx(step_idx)
        if self._sideband_base:
            write_sideband_batch_descriptors(
                self._sideband_base, [(req.seq_id, req.mtp_pos)],
            )
        cmd = self._cmd_writer.run_attention(
            gpu=req.default_gpu, layer_idx=mtp_layer, num_seqs=1,
            emit_gating=1, store_gating=1,
        )
        req.mtp_phase = "attn"
        if self._sideband_base:
            self._gating_attn_inflight = cmd.cmd_seq
        self._track_mtp_cmd(req, cmd)

    def _dispatch_mtp_step(self, req: RequestState) -> None:
        """Dispatch the next chained MTP DRAFT step (or finish drafting).

        Confidence-gated: after each completed step, should_continue()
        decides whether to chain another.  Step j embeds chain token j
        (x for j=0, previous draft output otherwise) at anchor+j on the
        MAIN sequence — positions >= anchor are the MTP layer's scratch
        region above the true-token frontier (INV-MTP-KV).
        """
        plan = self._mtp_draft.plan_draft_steps(req.speculation_depth)
        if req.mtp_step >= len(plan):
            self._finish_mtp_drafting(req)
            return

        # Chain input: last MTP output, or the newest committed token x
        if req.mtp_steps_result:
            embed = req.mtp_steps_result[-1].token_id
        else:
            embed = req.token_history[-1] if req.token_history else 0

        # Step 0 consumes the NEW anchor's trunk hidden.  After a batched
        # verify pass that hidden sits at attn_buf row m (the accepted
        # count), published via mtp_draft_hidden_row; chained steps (>= 1)
        # consume the previous MTP step's output at row 0, and rounds
        # entered from single-token AR feeds default to row 0.
        row = req.mtp_draft_hidden_row if req.mtp_step == 0 else 0
        if req.mtp_step == 0:
            req.mtp_draft_hidden_row = 0  # one-shot: consume + reset
        self._begin_mtp_step(
            req, pos=req.mtp_anchor_pos + req.mtp_step,
            embed_token=embed, mode="draft", hidden_row=row,
        )

    def _handle_mtp_completion(self, req: RequestState, cmp: Completion) -> None:
        """Advance the MTP step sub-state machine by one completed phase.

        Routes from _handle_compute_done (any speculation state) when
        cmp.cmd_seq matches req.mtp_cmd_seq.
        """
        req.mtp_cmd_seq = None
        phase = req.mtp_phase
        step_idx = req.mtp_step if req.mtp_mode == "draft" else 0
        mtp_layer = self._mtp_draft.mtp_layer_idx(step_idx)
        gpu = req.default_gpu

        if phase == "project":
            # Projection output is in the attn buffers — next up is the
            # MTP decoder attention with the fused gate + routing export
            # (production seam).  That command publishes the SHARED
            # routing-export slot, so it must serialize on the engine-wide
            # slot gate (TD-ORCH-ROUTING-EXPORT-MULTI / INV-IPC-6b): if
            # another request's gating-bearing command is in flight, park
            # this step; _retry_gating_waiters re-dispatches it once the
            # gate frees.  Single-request MTP never parks (its chain is
            # strictly serialized, the gate is always free here).
            if (self._sideband_base
                    and self._gating_attn_inflight is not None):
                req.mtp_phase = "attn-wait"
                return
            self._dispatch_mtp_gating_attention(req)
            return

        if phase == "attn":
            # Routed top-K published to the sideband routing-export slot.
            # Mirror the golden: forward the routed entries (selection-rank
            # order, INV-10c-2) to the production FETCH_AND_RUN_MOE seam —
            # NOT the deprecated resident-only RUN_MOE path (#89).
            entries: list[tuple[int, int, int, int]] = []
            if self._sideband_base:
                exp_layer, indices = read_sideband_routing_export(
                    self._sideband_base,
                )
                if exp_layer == mtp_layer:
                    owners = self._ep_gpus
                    # EP split: each expert fetched/run on its owning GPU
                    # (round-robin ownership over the EP owner set —
                    # OrchestratorConfig.ep_gpu_indices — golden parity;
                    # zone 0 like the golden's entries).
                    entries = [
                        (mtp_layer, e, CacheZone.STABLE.value,
                         owners[e % len(owners)])
                        for e in indices
                    ]
                    write_sideband_expert_prefetch(
                        self._sideband_base, entries,
                    )
            cmd = self._cmd_writer.fetch_and_run_moe(
                gpu=gpu, layer_idx=mtp_layer, num_seqs=1,
                expert_count=len(entries),
                timeout_us=_MTP_FETCH_TIMEOUT_US,
            )
            req.mtp_phase = "moe"
            self._track_mtp_cmd(req, cmd)
            return

        if phase == "moe":
            # MTP hidden state ready — project through the MTP shared head
            # (shared_head.norm + head) with host readback of the argmax
            # token + confidence.
            cmd = self._cmd_writer.output_head(
                gpu=gpu, num_tokens=1,
                input_buf=self._metadata.hidden_buf_id,
                output_buf=self._metadata.logits_buf_id,
                readback=True, compute_confidence=True,
                mtp_head_idx=0,
            )
            req.mtp_phase = "head"
            self._track_mtp_cmd(req, cmd)
            return

        # phase == "head": step complete — extract the draft token
        payload = cmp.payload.compute
        token_id = self._extract_token_id(payload)
        confidence = float(payload.top1_prob)
        req.mtp_phase = None
        mode = req.mtp_mode
        req.mtp_mode = None

        if mode == "draft":
            self._finish_mtp_draft_step(req, token_id, confidence)
        else:
            # Catch-up: the drafted token is discarded — the step exists
            # only for its MTP-layer KV append (INV-MTP-KV).  Resume what
            # was deferred behind it: the next queued post-batched-verify
            # catch-up, the next draft round, the next verification feed
            # (VERIFYING) or the next AR embedding (AUTOREGRESSIVE).
            self._after_mtp_catchup(req)

    def _after_mtp_catchup(self, req: RequestState) -> None:
        """Continue after a completed catch-up MTP step.

        Post-batched-verify rounds queue their exact-hidden catch-ups in
        mtp_catchup_queue (one per accepted position, ascending hidden
        rows) and set mtp_post_verify to the continuation; every other
        catch-up (AR warm, prefill warm, sequential-verify interleave)
        has an empty queue and resumes the deferred embedding — the
        historical behavior, byte-for-byte.
        """
        if req.mtp_catchup_queue:
            pos, tok, row = req.mtp_catchup_queue.pop(0)
            self._begin_mtp_step(
                req, pos=pos, embed_token=tok, mode="catchup",
                hidden_row=row,
            )
            return
        post = req.mtp_post_verify
        req.mtp_post_verify = None
        if post == "draft":
            # Chain the next draft round: its step 0 covers the new
            # anchor's INV-MTP-KV write off the anchor's trunk hidden
            # (attn_buf row m, published via mtp_draft_hidden_row).
            req.speculation_state = SpeculationState.DRAFTING
            self._start_drafting(req)
            return
        self._enqueue_next_embedding(req)

    def _finish_mtp_draft_step(self, req: RequestState,
                               token_id: int | None,
                               confidence: float) -> None:
        """Record a completed draft-chain step and chain or finish."""
        mtp_layer_idx = self._mtp_draft.mtp_layer_idx(req.mtp_step)
        if token_id is None:
            # No host readback available — cannot chain a meaningful draft.
            self._finish_mtp_drafting(req)
            return
        req.mtp_steps_result.append(DraftStep(
            token_id=token_id,
            confidence=confidence,
            mtp_layer_idx=mtp_layer_idx,
        ))
        req.mtp_step += 1

        # Check confidence threshold for early exit
        if self._mtp_draft.should_continue(req.mtp_step, confidence):
            self._dispatch_mtp_step(req)
        else:
            self._finish_mtp_drafting(req)

    def _finish_mtp_drafting(self, req: RequestState) -> None:
        """Combine MTP chain + prompt lookup, then verify sequentially.

        No PREFETCHING_VERIFY phase: the MTP chain exports gating for the
        MTP layer only (no per-layer full-model draft_gating), and the
        sequential feeds fetch their own routed experts through the
        production seam anyway.
        """
        mtp_result = None
        if req.mtp_steps_result:
            mtp_result = MtpDraftResult(steps=list(req.mtp_steps_result))

        combined = self._draft_combiner.combine(
            prompt_lookup_tokens=req.prompt_lookup_tokens,
            mtp_result=mtp_result,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=self._prompt_lookup.acceptance_rate,
            max_depth=req.speculation_depth,
            max_verifiable_depth=req.speculation_depth,
            expert_coverage_fraction=1.0,
        )

        if combined.depth == 0:
            # Nothing to verify -> back to plain AR.  The anchor's
            # INV-MTP-KV write already happened (draft step 0 embeds the
            # true token x), so no catch-up is needed here.
            self._abort_draft(req)
            return

        req.combined_draft = combined
        req.draft_complete = True
        # Batched verify (the throughput-bearing mode): ONE V-token
        # teacher-forced forward on the MAIN sequence replaces the K+1
        # sequential feeds.  Requires the sideband (token ids + batch
        # descriptors + routing export) and a daemon MoE batch capacity
        # covering V rows; falls back to the sequential schedule otherwise
        # (both are lossless — INV-MTP-LOSSLESS).
        v = combined.depth + 1
        if (self._mtp_draft.batched_verify
                and self._sideband_base
                and self._metadata.moe_batch_capacity >= v
                and req.token_history):
            self._start_batched_verification(req)
            return
        self._start_sequential_verification(req)

    def _start_batched_verification(self, req: RequestState) -> None:
        """Verify the round's drafts with ONE batched forward on the MAIN seq.

        The V = K+1 teacher-forced tokens [x, d0..d(K-1)] are fed at
        positions [base, base+V) in one pass shaped exactly like a chunked
        prefill (k_append first, causal window per row): EMBEDDING(V) →
        per-layer RUN_ATTENTION(V, is_prefill/chunk, fused gate + V-row
        routing export on MoE layers) → FETCH_AND_RUN_MOE(V, expert union)
        → OUTPUT_HEAD(V) reading back all V argmax tokens.  The greedy
        acceptance rule then runs on one completion
        (_handle_batched_verify_done).

        KV correctness without a fork: row j's attention covers positions
        [0, base+j] whose keys are exactly the teacher-forced prefix, so
        every output at a position within the accepted prefix is the main
        model's exact AR output (INV-MTP-LOSSLESS).  Rejected positions'
        main-layer KV lies ABOVE the committed frontier and is overwritten
        in place by later feeds before any attention can read it (causal
        windows never reach past the fed frontier) — the same
        overwrite-above-frontier mechanism as the MTP layer's draft
        scratch.  No fork, no rewind, no copy-back.
        """
        req.speculation_state = SpeculationState.VERIFYING
        req.verification_start_ns = time.perf_counter_ns()
        req.mtp_sequential_verify = False
        req.mtp_batched_verify = True
        req.verify_tokens = list(
            req.combined_draft.tokens[:req.speculation_depth],
        )
        req.verify_idx = 0
        req.verified_depth = len(req.verify_tokens)
        # Teacher-forced feed: the newest committed token x, then the drafts.
        req.vb_tokens = [req.token_history[-1]] + list(req.verify_tokens)
        req.vb_len = len(req.vb_tokens)
        req.vb_base_pos = self._feed_pos(req)
        # ONE work-item chain, V-wide at every hop (_dispatch_work_item's
        # vb branches widen num_seqs/num_tokens + the sideband inputs).
        self._enqueue_next_embedding(req)

    def _handle_batched_verify_done(self, req: RequestState,
                                    payload, token_id: int | None) -> None:
        """Apply the greedy acceptance rule to a batched verify readback.

        The OUTPUT_HEAD readback carries all V argmax tokens o1..oV
        (o_{j+1} = the main model's output at fed position base+j).  The
        shared _handle_output_head_done header already committed o1 (it is
        ALWAYS committed: d0 on match, the correction otherwise).  Here:
          1. m = longest matched prefix (drafts[j] == o_{j+1}).
          2. Commit o2..o_{m+1} (m more tokens; per-token stop checks
             mirror the sequential feed loop exactly).
          3. Record round stats (shared with the sequential path).
          4. Queue the INV-MTP-KV exact-hidden catch-ups for the accepted
             positions (rows 0..m-1 of the batched trunk hiddens) and
             chain the next round (draft step 0 off row m) or fall back
             to AR with a final catch-up at the new anchor.
        """
        vtokens = self._extract_verification_tokens(payload, req.vb_len)
        base = req.vb_base_pos
        drafts = list(req.verify_tokens)
        self._batched_verify_rounds += 1

        if not vtokens:
            if token_id is not None:
                # Degraded readback (single token): o1 alone was committed
                # by the header — treat as a zero-accept round.
                vtokens = [token_id]
            else:
                # No readback at all: nothing was committed; the round is
                # abandoned and the next AR feed re-feeds x at base
                # (deterministic, same KV) — lossless, like the sequential
                # None-readback path.
                req.verify_idx = 0
                self._record_sequential_round(req)
                self._enqueue_next_embedding(req)
                return

        # Greedy acceptance: longest prefix with drafts[j] == o_{j+1}.
        # (The same rule as MtpSpeculationMethod::verify / the C++ golden.)
        m = Verifier.compare_logits(
            draft_tokens=drafts[:max(len(vtokens) - 1, 0)],
            target_tokens=vtokens,
        )
        committed = vtokens[:m + 1]

        # Commit committed[0..m] with per-token stop checks — the exact
        # sequential-feed semantics (stop BEFORE counting an acceptance).
        accepted = 0
        stopped = False
        for j, tok in enumerate(committed):
            if j > 0:
                # o1 (j == 0) was already committed by the shared header.
                req.token_history.append(tok)
                if req.logprobs is not None:
                    req.logprob_results.append(None)
                if req.on_token is not None:
                    req.on_token(req.request_id, tok, None)
            req.tokens_generated += 1
            self._tokens_processed += 1
            req.last_output_head_ns = time.perf_counter_ns()
            if self._should_stop(req, tok):
                stopped = True
                break
            if j < m:
                accepted = j + 1

        req.verify_idx = accepted if stopped else m
        self._record_sequential_round(req)
        if stopped:
            self._finalize_request(req)
            return

        # INV-MTP-KV: exact-hidden catch-ups for the accepted positions.
        # Position base+j's trunk hidden is attn_buf row j (the batched
        # pass's rows); its committed successor is committed[j].  MTP steps
        # write only row 0, so the ascending queue consumes intact rows.
        if self._mtp_active():
            req.mtp_catchup_queue = [
                (base + j, committed[j], j) for j in range(m)
            ]

        # Continuation: chain the next round (its draft step 0 covers the
        # new anchor base+m off hidden row m) or return to AR with a final
        # catch-up at the anchor (the sequential schedule's tail rule).
        depth = self._utility_scorer.recommended_depth()
        if depth > 0 and self._mtp_active():
            req.speculation_depth = depth
            req.mtp_post_verify = "draft"
            req.mtp_draft_hidden_row = m
        elif self._mtp_active():
            req.mtp_catchup_queue.append((base + m, committed[m], m))
            req.mtp_post_verify = "ar"
        else:
            req.mtp_post_verify = None

        if req.mtp_catchup_queue or req.mtp_post_verify == "draft":
            # Drive the queue (or go straight to the next round when the
            # queue is empty, m == 0).
            if req.mtp_catchup_queue:
                pos, tok, row = req.mtp_catchup_queue.pop(0)
                self._begin_mtp_step(
                    req, pos=pos, embed_token=tok, mode="catchup",
                    hidden_row=row,
                )
                return
            self._after_mtp_catchup(req)
            return
        self._enqueue_next_embedding(req)

    # ------------------------------------------------------------------
    # DSpark whole-block draft step (DSP-5)
    # ------------------------------------------------------------------
    #
    # One DSpark round is ONE fused D_CMD_RUN_DSPARK_STEP (the DsparkLossless
    # golden's schedule): the DFlash backbone runs the whole γ block off the
    # aux-hidden-ingested context KV and the sequential Markov head samples
    # the γ ids on-device; the completion carries host_buf_offset/data_bytes
    # of the ids in the sideband readback scratch.  No KV fork, no warm or
    # catch-up steps: the aux export captures every fed target position
    # automatically (INV-DSPARK-AUX), so the draft context is always exactly
    # the fed prefix — rejection just re-drafts at the shorter anchor.

    def _dspark_active(self) -> bool:
        """DSpark drafting is configured (speculation.method == dspark).

        Mutually exclusive with the MTP flow by config construction —
        speculation.method selects ONE drafter; the glue must enable at
        most one of the two planners.
        """
        return self._dspark_draft.is_enabled

    def _dispatch_dspark_step(self, req: RequestState) -> None:
        """Dispatch the round's single whole-γ-block DSpark draft step."""
        num_query = self._dspark_draft.plan_num_query(req.speculation_depth)
        if num_query <= 0 or not req.token_history:
            # Nothing to draft — combine whatever prompt lookup found.
            self._finish_dspark_drafting(req)
            return

        # Anchor = the newest committed-but-unfed token; it will be fed at
        # anchor_pos == the fed-token count == the runtime's ingested
        # context length (run_step validates anchor_pos <= ctx_len
        # fail-closed; a CMP_ERROR aborts the round via _handle_error).
        req.dspark_anchor_pos = self._dspark_draft.anchor_pos(
            req.prompt_len, req.tokens_generated,
        )
        cmd = self._cmd_writer.run_dspark_step(
            gpu=req.default_gpu,
            seq_id=req.seq_id,
            anchor_token_id=req.token_history[-1],
            anchor_pos=req.dspark_anchor_pos,
            num_query=num_query,
        )
        self._ring_writer.write_struct(cmd)
        self._commands_this_cycle += 1

        # Track for completion routing (path 2a') + error recovery.
        req.dspark_cmd_seq = cmd.cmd_seq
        item = WorkItem(
            request_id=req.request_id,
            layer_idx=0,
            operation=WorkOperation.OUTPUT_HEAD,
            target_gpu=req.default_gpu,
            status=WorkStatus.DISPATCHED,
            is_speculative=True,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        self._cmd_seq_map[cmd.cmd_seq] = (req.request_id, item)
        self._draft_cmd_seqs[cmd.cmd_seq] = req.request_id

    def _handle_dspark_completion(self, req: RequestState,
                                  cmp: Completion) -> None:
        """Extract the γ sampled draft ids and finish the round's drafting.

        The daemon copied the ids (i32) into the sideband readback scratch
        before recording the completion event (DSP-5 readback contract) —
        payload.host_buf_offset/data_bytes locate them.  DSP-6: with
        speculation.dspark.confidence_enabled the daemon appends the γ raw
        survival probabilities c_k (f32, trained confidence head —
        INV-DSPARK-CONF) after the ids; data_bytes = γ * 8 then (both
        sides derive the split from the shared config).
        """
        req.dspark_cmd_seq = None
        payload = cmp.payload.compute
        tokens: list[int] = []
        confidences: list[float] = []
        words = payload.data_bytes // 4
        conf_on = self._dspark_draft.confidence_enabled
        n = words // 2 if conf_on else words
        if n > 0 and self._host_buf_base != 0:
            addr = self._host_buf_base + payload.host_buf_offset
            arr = (ctypes.c_int32 * n).from_address(addr)
            tokens = [int(arr[i]) for i in range(n)]
            if conf_on:
                carr = (ctypes.c_float * n).from_address(addr + 4 * n)
                confidences = [float(carr[i]) for i in range(n)]
        req.dspark_result = DsparkDraftResult(tokens=tokens,
                                              confidences=confidences)
        self._finish_dspark_drafting(req)

    def _finish_dspark_drafting(self, req: RequestState) -> None:
        """Combine DSpark block + prompt lookup, then verify sequentially.

        No PREFETCHING_VERIFY phase: the draft ran entirely on the draft
        GPU (no per-layer target gating), and the sequential feeds fetch
        their own routed experts through the production seam anyway.
        """
        # DSP-6: per-position cumulative survival a_j = Π c_i from the
        # trained confidence head replaces the EMA-acceptance proxy when
        # confidence_enabled (None → the combiner falls back to the proxy).
        combined = self._draft_combiner.combine(
            prompt_lookup_tokens=req.prompt_lookup_tokens,
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=self._prompt_lookup.acceptance_rate,
            max_depth=req.speculation_depth,
            max_verifiable_depth=req.speculation_depth,
            expert_coverage_fraction=1.0,
            dspark_result=req.dspark_result,
            dspark_confidence=self._dspark_draft.confidence_proxy(),
            dspark_confidences=self._dspark_draft.survival_confidences(
                req.dspark_result),
        )

        if combined.depth == 0:
            # Nothing to verify -> back to plain AR (no fork was taken, so
            # _abort_draft only resets state).
            self._abort_draft(req)
            return

        req.combined_draft = combined
        req.draft_complete = True
        self._start_sequential_verification(req)

    # ------------------------------------------------------------------
    # Sequential early-stop verification (MtpLossless pattern)
    # ------------------------------------------------------------------

    def _start_sequential_verification(self, req: RequestState) -> None:
        """Verify draft tokens with sequential feeds on the MAIN sequence.

        Each feed is a normal AR forward (EMBEDDING → per-layer →
        OUTPUT_HEAD) through the work queue, teacher-forcing the newest
        COMMITTED token.  The feed's output is committed immediately (it
        is the main model's own greedy output — INV-MTP-LOSSLESS) and
        compared against the next pending draft token: on match the loop
        keeps feeding (with an INV-MTP-KV catch-up in between), on
        mismatch it stops — the rejected draft is never fed, so the main
        KV needs no fork, rewind, or copy-back (TD-MTP-PY-LOOP-KV (2)).
        """
        req.speculation_state = SpeculationState.VERIFYING
        req.verification_start_ns = time.perf_counter_ns()
        req.mtp_sequential_verify = True
        req.verify_tokens = list(
            req.combined_draft.tokens[:req.speculation_depth],
        )
        req.verify_idx = 0
        req.verified_depth = len(req.verify_tokens)
        # First feed: x (the anchor's committed successor) at anchor+1
        self._enqueue_next_embedding(req)

    def _handle_sequential_verify_feed(self, req: RequestState,
                                       token_id: int | None) -> None:
        """Process one completed verification feed (its OUTPUT_HEAD).

        The produced token was already committed to token_history by
        _handle_output_head_done's shared header.  Feed count bookkeeping
        happens here (the shared AR path below Path 1 is skipped).
        """
        req.tokens_generated += 1
        self._tokens_processed += 1
        req.last_output_head_ns = time.perf_counter_ns()

        if token_id is None:
            # No readback — cannot compare; end the round conservatively.
            self._record_sequential_round(req)
            self._enqueue_next_embedding(req)
            return

        if self._should_stop(req, token_id):
            self._record_sequential_round(req)
            self._finalize_request(req)
            return

        j = req.verify_idx
        if j < len(req.verify_tokens) and token_id == req.verify_tokens[j]:
            # Draft j accepted.  There is always a next feed (another
            # compare, or the bonus feed after the last draft), so run the
            # INV-MTP-KV catch-up at the just-fed position embedding its
            # committed successor, then feed.  The golden's schedule: a
            # catch-up follows every accepted feed except the round's last.
            req.verify_idx += 1
            if self._mtp_active():
                self._begin_mtp_step(
                    req, pos=self._mtp_anchor_pos(req),
                    embed_token=token_id, mode="catchup",
                )
                return
            self._enqueue_next_embedding(req)
            return

        # Mismatch (token_id is the committed correction) or bonus feed
        # done (j == len(verify_tokens)) — the round is over.  No catch-up
        # at this position: the next round's draft step 0 covers it and
        # needs the trunk hidden still resident in the attn buffers.
        self._record_sequential_round(req)
        self._continue_after_sequential_round(req, token_id)

    def _record_sequential_round(self, req: RequestState) -> None:
        """Record stats for a finished sequential round and reset state."""
        accepted = req.verify_idx
        attempted = len(req.verify_tokens)

        vresult = VerificationResult(
            accepted_length=accepted,
            attempted_length=attempted,
            accepted_tokens=list(req.verify_tokens[:accepted]),
            rejected_positions=list(range(accepted, attempted)),
        )
        self._verifier.record_result(vresult)
        if req.combined_draft is not None:
            self._draft_combiner.record_result(req.combined_draft, accepted)
            self._prompt_lookup.record_result(
                num_proposed=req.combined_draft.depth,
                num_accepted=accepted,
            )
        if req.mtp_steps_result:
            self._mtp_draft.record_result(
                MtpDraftResult(steps=list(req.mtp_steps_result)), accepted,
            )
        if req.dspark_result is not None:
            self._dspark_draft.record_result(req.dspark_result, accepted)
        if attempted > 0:
            req.acceptance_rate = accepted / attempted

        # EPM-1 (Phase 29): append the block's metadata to the dump
        # manifest — the daemon already dumped the raw hiddens/routing;
        # accepted length + the actually-verified token stream live only
        # here. Joined on (seq_id, anchor_pos) with the EPMB record by
        # tools/elb_train/dataset.py. One `is not None` check when off.
        if (self._epm_manifest is not None and req.dspark_result is not None
                and req.dspark_result.tokens):
            self._epm_manifest.append_block(
                seq_id=req.seq_id,
                anchor_pos=req.dspark_anchor_pos,
                request_id=req.request_id,
                draft_tokens=list(req.dspark_result.tokens),
                verify_tokens=list(req.verify_tokens),
                accepted_len=accepted,
                confidences=list(req.dspark_result.confidences),
            )

        # Timing for cascade depth adaptation (TD-22) — same shape as the
        # batched path's accounting in _process_acceptance.
        now_ns = time.perf_counter_ns()
        draft_us = (
            (req.verification_start_ns - req.drafting_start_ns) / 1000.0
            if req.drafting_start_ns > 0 and req.verification_start_ns > 0
            else 0.0
        )
        verify_us = (
            (now_ns - req.verification_start_ns) / 1000.0
            if req.verification_start_ns > 0
            else 0.0
        )
        # NOTE: last_output_head_ns is advanced by every sequential feed,
        # so by round end it is LATER than drafting_start_ns — clamp to 0
        # (the "unknown" convention) instead of feeding a negative base
        # time into depth adaptation.
        base_us = (
            (req.drafting_start_ns - req.last_output_head_ns) / 1000.0
            if req.last_output_head_ns > 0
            and req.drafting_start_ns > req.last_output_head_ns
            else 0.0
        )
        self._utility_scorer.record_iteration(
            depth_used=attempted,
            tokens_accepted=accepted,
            draft_time_us=draft_us,
            verify_time_us=verify_us,
            base_time_us=base_us,
        )

        # Reset round state — back to AUTOREGRESSIVE
        req.speculation_state = SpeculationState.AUTOREGRESSIVE
        req.mtp_sequential_verify = False
        req.mtp_batched_verify = False
        req.vb_len = 0
        req.vb_tokens = []
        req.verify_tokens = []
        req.verify_idx = 0
        req.combined_draft = None
        req.draft_complete = False
        req.verified_depth = 0
        req.mtp_step = 0
        req.mtp_steps_result = []
        req.dspark_result = None
        req.dspark_anchor_pos = 0
        req.prompt_lookup_tokens = []
        req.drafting_start_ns = 0
        req.verification_start_ns = 0

    def _continue_after_sequential_round(self, req: RequestState,
                                         last_token: int) -> None:
        """Round over: chain the next round or return to plain AR.

        The round's last feed left the trunk hidden of the new anchor in
        the attn buffers, so a new draft round can start immediately (its
        step 0 provides the anchor's INV-MTP-KV write).  Without a new
        round, run the catch-up at the anchor instead, then continue AR.
        DSpark chains rounds the same way (its context KV advanced with
        every fed position automatically); it needs no catch-up.
        """
        depth = self._utility_scorer.recommended_depth()
        if depth > 0 and (self._mtp_active() or self._dspark_active()):
            req.speculation_depth = depth
            req.speculation_state = SpeculationState.DRAFTING
            self._start_drafting(req)
            return
        if self._mtp_active():
            self._begin_mtp_step(
                req, pos=self._mtp_anchor_pos(req),
                embed_token=last_token, mode="catchup",
            )
            return
        self._enqueue_next_embedding(req)

    # ------------------------------------------------------------------
    # Self-speculative draft steps (#62e)
    # ------------------------------------------------------------------

    def _dispatch_self_spec_step(self, req: RequestState) -> None:
        """Dispatch the next self-spec forward pass as a fused command.

        Each step is one D_CMD_RUN_SELF_SPEC_FORWARD covering all model
        layers with top-K expert MoE + residual correction.  Gating
        checkpoints arrive as CMP_CHECKPOINT before the final CMP_COMPUTE_DONE.
        """
        if req.self_spec_step >= self._self_speculative.max_depth:
            self._finish_drafting(req)
            return

        # Routing-export slot gate (TD-ORCH-ROUTING-EXPORT-MULTI /
        # INV-IPC-6b): the self-spec forward runs store_gating=1, which
        # publishes the SHARED routing-export slot on every MoE layer of
        # the pass — it must not run while another request's un-captured
        # export is in flight (and, symmetrically, holding the gate keeps
        # other requests' gating attentions from clobbering mid-pass).
        # Park; _retry_gating_waiters re-dispatches when the gate frees.
        if (self._sideband_base
                and self._gating_attn_inflight is not None):
            req.self_spec_wait = True
            return

        # Input token: last self-spec output, or last MTP output, or last AR token
        if req.self_spec_steps_result:
            input_token = req.self_spec_steps_result[-1].token_id
        elif req.mtp_steps_result:
            input_token = req.mtp_steps_result[-1].token_id
        else:
            input_token = req.token_history[-1] if req.token_history else 0

        # Layer skip mask (128-bit: layers 0-63 in lo, 64-127 in hi).
        # Step 0 always has empty similarities (no prior data), so no
        # layers are skipped.  Step 1+ uses similarities from the previous
        # step — conservative, since adjacent tokens have similar layer
        # redundancy patterns (#62g cross-step refinement).
        skip_set: set[int] = set()
        if (self._layer_skip.is_enabled
                and self._layer_skip.should_enable(req.acceptance_rate)
                and req.layer_similarities):
            skip_set = self._layer_skip.compute_skip_set(
                req.layer_similarities,
            )
        req.current_skip_set = skip_set

        # Track skip set history for cancel stability detection (#62h-cancel).
        # Only cancel transfers for layers stably skipped across consecutive
        # steps — a layer skipped once might come back on the next step.
        # When cancel_skip_transfers=False, skip sets still suppress new
        # prefetch hints (Phase 2) but don't cancel in-flight transfers.
        # Read cancellation params from live hot_config (x-changeable via schema)
        hc = self._hot_config
        window = hc.speculation_transfer_cancellation_cancel_stability_window
        req.skip_set_history.append(skip_set)
        if (hc.speculation_transfer_cancellation_cancel_skip_transfers
                and len(req.skip_set_history) >= window):
            stable = req.skip_set_history[-1]
            for i in range(2, window + 1):
                stable = stable & req.skip_set_history[-i]
            if stable:
                self._cancel_transfers_for_skipped_layers(req, stable)

        skip_lo, skip_hi = _encode_skip_mask(skip_set)

        cmd = self._cmd_writer.run_self_spec_forward(
            gpu=req.default_gpu,
            seq_id=req.draft_seq_id,
            input_token_id=input_token,
            draft_expert_count=self._self_speculative.draft_expert_count,
            apply_residual_corr=1,
            store_gating=1,
            step_idx=req.self_spec_step,
            skip_mask_lo=skip_lo,
            skip_mask_hi=skip_hi,
        )
        self._ring_writer.write_struct(cmd)
        self._commands_this_cycle += 1
        # Hold the routing-export slot gate for the whole forward; the
        # final CMP_COMPUTE_DONE releases it (_handle_compute_done).
        if self._sideband_base:
            self._gating_attn_inflight = cmd.cmd_seq

        # Track for completion routing
        req.self_spec_cmd_seq = cmd.cmd_seq
        draft_item = WorkItem(
            request_id=req.request_id,
            layer_idx=self._metadata.num_layers - 1,
            operation=WorkOperation.OUTPUT_HEAD,
            target_gpu=req.default_gpu,
            status=WorkStatus.DISPATCHED,
            is_speculative=True,
            timestamp_created_ns=time.perf_counter_ns(),
        )
        self._cmd_seq_map[cmd.cmd_seq] = (req.request_id, draft_item)
        self._draft_cmd_seqs[cmd.cmd_seq] = req.request_id

    def _handle_self_spec_completion(self, req: RequestState, cmp: Completion) -> None:
        """Process self-spec step completion: extract token+confidence, decide next.

        Routes from _handle_compute_done when req is DRAFTING and
        cmp.cmd_seq matches req.self_spec_cmd_seq.
        """
        token_id, confidence = self._extract_draft_step_result(
            cmp.payload.compute,
        )

        req.self_spec_steps_result.append(DraftStep(
            token_id=token_id,
            confidence=confidence,
            mtp_layer_idx=-1,
        ))
        req.self_spec_step += 1
        req.self_spec_cmd_seq = None

        if self._self_speculative.should_continue(
            req.self_spec_step, confidence,
        ):
            self._dispatch_self_spec_step(req)
        else:
            self._finish_drafting(req)

    # ------------------------------------------------------------------
    # Finish drafting: combine all sources
    # ------------------------------------------------------------------

    def _finish_drafting(self, req: RequestState) -> None:
        """Combine all draft sources and transition to PREFETCHING_VERIFY.

        Called when:
          - Prompt-lookup-only (same cycle, no GPU steps).
          - After MTP steps complete (multi-cycle).
          - After self-spec steps complete (multi-cycle).

        Assembles MtpDraftResult and SelfSpecDraftResult from accumulated
        steps, feeds all three sources to DraftCombiner, builds draft_gating
        from self-spec gating rows, and transitions to _start_prefetching.
        """
        # Build MTP result from accumulated steps
        mtp_result = None
        if req.mtp_steps_result:
            mtp_result = MtpDraftResult(steps=list(req.mtp_steps_result))

        # Build self-spec result from accumulated steps
        self_spec_result = None
        if req.self_spec_steps_result:
            self_spec_result = SelfSpecDraftResult(
                steps=list(req.self_spec_steps_result),
            )

        # Assemble draft_gating from self-spec gating rows
        # Shape: (depth, num_moe_layers, num_experts)
        if req.self_spec_gating_rows:
            req.draft_gating = np.stack(req.self_spec_gating_rows)
        else:
            req.draft_gating = None

        # Combine all draft sources via DraftCombiner
        combined = self._draft_combiner.combine(
            prompt_lookup_tokens=req.prompt_lookup_tokens,
            mtp_result=mtp_result,
            self_spec_result=self_spec_result,
            prompt_lookup_acceptance_rate=self._prompt_lookup.acceptance_rate,
            max_depth=req.speculation_depth,
            max_verifiable_depth=req.speculation_depth,
            expert_coverage_fraction=1.0,
        )

        # No draft tokens produced -> revert to autoregressive
        if combined.depth == 0:
            self._abort_draft(req)
            return

        req.combined_draft = combined
        req.draft_complete = True

        self._start_prefetching(req)

    # ------------------------------------------------------------------
    # Shared draft helpers
    # ------------------------------------------------------------------

    def _extract_draft_step_result(self, payload) -> tuple[int, float]:
        """Read token_id (uint32) + confidence (float32) from draft step readback.

        The fused MTP/self-spec commands write 8 bytes to the host buffer:
        4 bytes token_id + 4 bytes confidence.  Falls back to top1_prob
        from the completion payload if the host buffer isn't available.
        """
        if payload.data_bytes >= 8 and self._host_buf_base != 0:
            addr = self._host_buf_base + payload.host_buf_offset
            token_id = int(
                (ctypes.c_uint32 * 1).from_address(addr)[0]
            )
            confidence = float(
                (ctypes.c_float * 1).from_address(addr + 4)[0]
            )
            return (token_id, confidence)
        # Fallback: use top1_prob from completion, token_id unknown
        return (0, float(payload.top1_prob))

    def _accumulate_draft_gating(self, req: RequestState, cmp: Completion) -> None:
        """Accumulate gating checkpoint from self-spec into per-step buffer.

        Called from _handle_checkpoint (completion_handlers.py) when a
        gating checkpoint arrives during DRAFTING for a self-spec step.
        Builds the (step, moe_layer, expert) gating tensor row by row.
        """
        payload = cmp.payload.checkpoint
        if payload.checkpoint_type != 1:
            return
        gating_data = self._read_host_buf(
            (payload.host_buf_offset, payload.data_bytes),
        )
        if gating_data is None:
            return
        moe_layer_offset = payload.layer_idx - self._first_moe_layer
        if not (0 <= moe_layer_offset < self._metadata.num_moe_layers):
            return
        # Ensure buffer exists for the current step
        while len(req.self_spec_gating_rows) <= req.self_spec_step:
            req.self_spec_gating_rows.append(
                np.zeros(
                    (self._metadata.num_moe_layers,
                     self._metadata.num_experts),
                    dtype=np.float32,
                )
            )
        ne = min(len(gating_data), self._metadata.num_experts)
        req.self_spec_gating_rows[req.self_spec_step][
            moe_layer_offset, :ne
        ] = gating_data[:ne]

    # ------------------------------------------------------------------
    # Draft abort / verification abort
    # ------------------------------------------------------------------

    def _abort_draft(self, req: RequestState) -> None:
        """Cleanly abort a draft attempt: free KV fork, reset state, continue AR."""
        # Queue cancellation for all verification-origin transfers for this
        # request (origin-based redesign).  Phase 5 processes after all fetches;
        # daemon ELM refcounting deduplicates against same-cycle fetches.
        # When cancel_abort_transfers=False, transfers land in cache as warm experts.
        if self._hot_config.speculation_transfer_cancellation_cancel_abort_transfers:
            for origin in list(self._origin_transfers):
                if origin[0] == req.request_id:
                    self._cancel_queue.append((origin, None))

        if req.draft_seq_id is not None:
            cmd = self._cmd_writer.e_seq_free(
                gpu=req.default_gpu, seq_id=req.draft_seq_id,
            )
            self._ring_writer.write_struct(cmd)
            self._commands_this_cycle += 1
            req.draft_seq_id = None
        # Clean up any inflight draft step tracking
        if req.mtp_cmd_seq is not None:
            self._draft_cmd_seqs.pop(req.mtp_cmd_seq, None)
            self._cmd_seq_map.pop(req.mtp_cmd_seq, None)
            req.mtp_cmd_seq = None
        if req.self_spec_cmd_seq is not None:
            self._draft_cmd_seqs.pop(req.self_spec_cmd_seq, None)
            self._cmd_seq_map.pop(req.self_spec_cmd_seq, None)
            req.self_spec_cmd_seq = None
        if req.dspark_cmd_seq is not None:
            self._draft_cmd_seqs.pop(req.dspark_cmd_seq, None)
            self._cmd_seq_map.pop(req.dspark_cmd_seq, None)
            req.dspark_cmd_seq = None
        # Reset all speculation fields
        req.speculation_state = SpeculationState.AUTOREGRESSIVE
        req.combined_draft = None
        req.draft_complete = False
        req.draft_gating = None
        req.verification_plan = None
        req.verification_analyses = []
        req.verified_depth = 0
        req.drafting_start_ns = 0
        req.verification_start_ns = 0
        req.mtp_step = 0
        req.mtp_steps_result = []
        req.mtp_phase = None
        req.mtp_mode = None
        req.mtp_sequential_verify = False
        req.mtp_batched_verify = False
        req.vb_len = 0
        req.vb_tokens = []
        req.mtp_catchup_queue = []
        req.mtp_post_verify = None
        req.mtp_draft_hidden_row = 0
        req.verify_tokens = []
        req.verify_idx = 0
        req.dspark_result = None
        req.dspark_anchor_pos = 0
        req.self_spec_step = 0
        req.self_spec_steps_result = []
        req.self_spec_gating_rows = []
        req.self_spec_wait = False  # un-park from the routing-export gate
        req.prompt_lookup_tokens = []
        req.layer_similarities = {}
        req.current_skip_set = set()
        req.skip_set_history = []
        # Origin entries NOT cleaned here — they may be in _cancel_queue and
        # need to survive until Phase 5 processes them.  Expiry sweep handles
        # stale entries; _process_acceptance handles the success path.
        self._enqueue_next_embedding(req)

    def _abort_verification(self, request_id: int) -> None:
        """Abort an in-progress verification: clean up cmd_seq tracking.

        Called when a verification intermediate command errors (TD-21).
        Removes all _verify_cmd_seqs and _cmd_seq_map entries for this
        request, then delegates to _abort_draft to reset speculation state.
        """
        # Remove all intermediate verification cmd_seqs for this request
        stale = [cs for cs, rid in self._verify_cmd_seqs.items()
                 if rid == request_id]
        for cs in stale:
            del self._verify_cmd_seqs[cs]
        # Remove the tracked OUTPUT_HEAD cmd_seq too
        stale_tracked = [cs for cs, (rid, _) in self._cmd_seq_map.items()
                         if rid == request_id]
        for cs in stale_tracked:
            del self._cmd_seq_map[cs]
        req = self._requests.get(request_id)
        if req is not None:
            self._abort_draft(req)

    # ------------------------------------------------------------------
    # Prefetching: load experts needed for verification
    # ------------------------------------------------------------------

    def _start_prefetching(self, req: RequestState) -> None:
        """Compute verification plan and begin expert prefetching.

        Two paths:
          - draft_gating present (future MTP/self-spec): full SP-MoE analysis,
            dispatch targeted expert transfers.
          - draft_gating None (prompt-lookup-only): skip analysis, set
            max_depth = draft depth, no transfers needed.

        After dispatching transfers (if any), checks if we can immediately
        transition to VERIFYING (all experts already resident).
        """
        req.speculation_state = SpeculationState.PREFETCHING_VERIFY

        if req.draft_gating is not None:
            # MTP/self-spec path: compute which experts verification needs
            vplan = self._speculative_prefetch.compute_verification_plan(
                draft_gating=req.draft_gating,
                resident_experts=self._resident_keys,
                first_moe_layer=self._first_moe_layer,
            )
            req.verification_plan = vplan
            req.max_verifiable_depth = vplan.max_depth
            req.speculation_depth = UtilityScorer.apply_ceiling(
                req.speculation_depth, req.max_verifiable_depth,
            )
            # Dispatch transfers for missing experts
            for tr in vplan.transfers:
                cmd = self._cmd_writer.prefetch_expert(
                    gpu=tr.target_gpu,
                    layer_idx=tr.key.layer_idx,
                    expert_idx=tr.key.expert_idx,
                    zone=(
                        tr.zone.value
                        if isinstance(tr.zone, CacheZone)
                        else tr.zone
                    ),
                    priority=tr.priority,
                )
                self._ring_writer.write_struct(cmd)
                self._commands_this_cycle += 1
                self._inflight_transfers.setdefault(
                    tr.target_gpu, set(),
                ).add(tr.key)
                # Track by causal origin for cancellation (origin-based redesign).
                # origin = (request_id, token_offset, draft_step=0 for verification)
                origin = (req.request_id, req.tokens_generated, 0)
                od = self._origin_transfers.setdefault(origin, {})
                od[tr.key] = (cmd.cmd_seq, tr.target_gpu)
                if origin not in self._origin_birth_cycle:
                    self._origin_birth_cycle[origin] = self._cycle_count
        else:
            # Prompt-lookup-only: no gating data, bypass SP-MoE entirely.
            # Verification runs with moe_mode=0 (use whatever is resident).
            req.verification_plan = VerificationPlan(
                transfers=[],
                max_depth=(
                    req.combined_draft.depth
                    if req.combined_draft
                    else 0
                ),
            )
            req.max_verifiable_depth = req.verification_plan.max_depth

        # May transition immediately if no transfers are needed
        self._check_verification_ready(req)

    def _check_verification_ready(self, req: RequestState) -> None:
        """Check if all verification experts are resident; start if so.

        For prompt-lookup-only (no gating), always ready immediately.
        For MTP/self-spec, checks that every expert in the verification
        plan's transfer list is now in _resident_keys.
        """
        if req.speculation_state != SpeculationState.PREFETCHING_VERIFY:
            return
        if req.verification_plan is None:
            return

        if req.draft_gating is None:
            # No expert requirements for prompt-lookup-only path
            all_ready = True
        else:
            gpu_resident = self._resident_keys.get(req.default_gpu, set())
            required = {tr.key for tr in req.verification_plan.transfers}
            all_ready = required.issubset(gpu_resident)

        if all_ready:
            self._start_verification(req)

    # ------------------------------------------------------------------
    # -- Transfer cancellation for skipped layers (#62h-cancel) -----------

    def _cancel_transfers_for_skipped_layers(
        self, req: RequestState, stable_skips: set[int],
    ) -> None:
        """Queue cancellation for stably-skipped layers (origin-based redesign).

        Adds (origin, skip_layer_set) to _cancel_queue.  Actual CMD_CANCEL_TRANSFER
        emission happens at end of Phase 5, after all fetches, so daemon ELM
        refcounting deduplicates against same-cycle fetches automatically.
        """
        origin = (req.request_id, req.tokens_generated, 0)
        if origin in self._origin_transfers:
            self._cancel_queue.append((origin, stable_skips))

    # ------------------------------------------------------------------
    # Verification: dispatch full-model forward pass
    # ------------------------------------------------------------------

    def _start_verification(self, req: RequestState) -> None:
        """Dispatch the verification forward pass directly to the ring.

        Bypasses the work queue and scheduler — verification is the
        highest-priority operation (experts are already resident).

        Uses Verifier to:
          1. Analyze MoE coverage (if gating data available).
          2. Determine verifiable depth.
          3. Build a VerificationCommandPlan (EMBEDDING + per-layer
             ATTENTION/MoE + OUTPUT_HEAD).

        All intermediate commands are tracked in _verify_cmd_seqs for
        error recovery (TD-21).  Only the final OUTPUT_HEAD is tracked
        in _cmd_seq_map so its completion triggers _handle_output_head_done.
        """
        req.speculation_state = SpeculationState.VERIFYING
        req.verification_start_ns = time.perf_counter_ns()

        # Determine verifiable depth and per-layer moe_modes
        if req.draft_gating is not None and req.combined_draft is not None:
            analyses = self._verifier.analyze_moe_coverage(
                req.combined_draft, req.draft_gating,
                self._resident_keys, self._first_moe_layer,
            )
            verified_depth = self._verifier.compute_verifiable_depth(
                analyses, req.max_verifiable_depth,
            )
        else:
            # No gating: verify all draft positions with moe_mode=0
            analyses = []
            verified_depth = (
                req.combined_draft.depth if req.combined_draft else 0
            )

        req.verification_analyses = analyses
        req.verified_depth = verified_depth

        if verified_depth == 0:
            self._abort_draft(req)
            return

        # Build the command sequence: EMBEDDING, per-layer ATT+MoE, OUTPUT_HEAD
        cmd_plan = self._verifier.plan_verification_pass(
            verified_depth, analyses,
        )

        # Dispatch all commands to the ring and track them
        all_cmd_seqs: list[int] = []
        for desc in cmd_plan.commands:
            cmd = self._build_verification_command(req, desc)
            if cmd is None:
                continue
            self._ring_writer.write_struct(cmd)
            self._commands_this_cycle += 1
            all_cmd_seqs.append(cmd.cmd_seq)

        if all_cmd_seqs:
            # Track intermediates (EMBEDDING, ATTENTION, MoE) for error recovery
            for cs in all_cmd_seqs[:-1]:
                self._verify_cmd_seqs[cs] = req.request_id

            # Track the final OUTPUT_HEAD in _cmd_seq_map so its completion
            # routes through _handle_compute_done -> _handle_output_head_done
            verify_item = WorkItem(
                request_id=req.request_id,
                layer_idx=self._metadata.num_layers - 1,
                operation=WorkOperation.OUTPUT_HEAD,
                target_gpu=req.default_gpu,
                status=WorkStatus.DISPATCHED,
                is_speculative=True,
                timestamp_created_ns=time.perf_counter_ns(),
            )
            self._cmd_seq_map[all_cmd_seqs[-1]] = (
                req.request_id, verify_item,
            )

    def _build_verification_command(self, req: RequestState, desc: CommandDescriptor):
        """Translate a VerificationCommandPlan descriptor into a ring Command."""
        gpu = req.default_gpu
        if desc.cmd_type == "EMBEDDING_LOOKUP":
            return self._cmd_writer.embedding_lookup(
                gpu=gpu, num_tokens=desc.num_seqs,
                output_buf=self._metadata.hidden_buf_id,
            )
        elif desc.cmd_type == "RUN_ATTENTION":
            # TD-51cg: write batch descriptors for verification tokens.
            # Each verification token is at a successive position after the
            # parent's current decode point (= the next feed position,
            # #91 fixed convention: prompt_len + tokens_generated - 1).
            if self._sideband_base and desc.num_seqs > 0:
                seq_id = req.draft_seq_id if req.draft_seq_id is not None else req.seq_id
                base_pos = self._feed_pos(req)
                write_sideband_batch_descriptors(
                    self._sideband_base,
                    [(seq_id, base_pos + i) for i in range(desc.num_seqs)],
                )
            return self._cmd_writer.run_attention(
                gpu=gpu, layer_idx=desc.layer_idx,
                num_seqs=desc.num_seqs,
                emit_checkpoint=desc.emit_checkpoint,
            )
        elif desc.cmd_type == "RUN_MOE":
            return self._cmd_writer.run_moe(
                gpu=gpu, layer_idx=desc.layer_idx,
                num_seqs=desc.num_seqs, moe_mode=desc.moe_mode,
                emit_checkpoint=desc.emit_checkpoint,
            )
        elif desc.cmd_type == "OUTPUT_HEAD":
            return self._cmd_writer.output_head(
                gpu=gpu, num_tokens=desc.num_seqs,
                input_buf=self._metadata.hidden_buf_id,
                output_buf=self._metadata.logits_buf_id,
                readback=bool(desc.readback_to_host),
                compute_confidence=bool(desc.compute_confidence),
            )
        return None

    # ------------------------------------------------------------------
    # Post-verification: compare, accept/reject, record stats
    # ------------------------------------------------------------------

    def _finish_verification(self, req: RequestState, verification_tokens: list[int]) -> None:
        """Compare draft tokens with verification output and process result.

        Uses Verifier.compare_logits (greedy prefix match) to find the
        longest accepted prefix, then builds a VerificationResult and
        delegates to _process_acceptance.
        """
        if req.combined_draft is None:
            self._abort_draft(req)
            return

        accepted_length = Verifier.compare_logits(
            draft_tokens=req.combined_draft.tokens[:req.verified_depth],
            target_tokens=verification_tokens,
        )

        vresult = self._verifier.build_result(
            draft=req.combined_draft,
            analyses=req.verification_analyses,
            verified_depth=req.verified_depth,
            accepted_length=accepted_length,
            seq_id=req.seq_id,
        )

        self._process_acceptance(req, vresult)

    def _process_acceptance(self, req: RequestState, vresult: VerificationResult) -> None:
        """Process accept/reject, free KV, record stats, continue generation.

        Steps:
          1. Free draft KV fork (E_CMD_SEQ_FREE) -- INV-4.9b.
          2. Advance tokens_generated by accepted_length.
          3. On partial rejection: cancel downstream work items (INV-3.4.5).
          4. Record statistics for adaptive depth control.
          5. Reset all speculation fields back to AUTOREGRESSIVE.
          6. Enqueue next EMBEDDING to continue generation.
        """
        # 1. Free draft KV pages
        if req.draft_seq_id is not None:
            cmd = self._cmd_writer.e_seq_free(
                gpu=req.default_gpu, seq_id=req.draft_seq_id,
            )
            self._ring_writer.write_struct(cmd)
            self._commands_this_cycle += 1
            req.draft_seq_id = None

        # 2. Accept tokens
        if vresult.accepted_length > 0:
            req.tokens_generated += vresult.accepted_length
            self._tokens_processed += vresult.accepted_length
            req.token_history.extend(vresult.accepted_tokens)
            if req.logprobs is not None:
                for _ in vresult.accepted_tokens:
                    req.logprob_results.append(None)
            if req.on_token is not None:
                for tok in vresult.accepted_tokens:
                    req.on_token(req.request_id, tok, None)
            if vresult.attempted_length > 0:
                req.acceptance_rate = (
                    vresult.accepted_length / vresult.attempted_length
                )

        # 3. Cancel downstream on rejection (INV-3.4.5)
        if vresult.accepted_length < vresult.attempted_length:
            self._work_queue.cancel_by_request(req.request_id)

        # 4. Record statistics for all involved modules
        self._verifier.record_result(vresult)
        if req.combined_draft is not None:
            self._draft_combiner.record_result(
                req.combined_draft, vresult.accepted_length,
            )
            self._prompt_lookup.record_result(
                num_proposed=req.combined_draft.depth,
                num_accepted=vresult.accepted_length,
            )
        # Compute real timing for cascade depth adaptation (TD-22)
        now_ns = time.perf_counter_ns()
        draft_us = (
            (req.verification_start_ns - req.drafting_start_ns) / 1000.0
            if req.drafting_start_ns > 0 and req.verification_start_ns > 0
            else 0.0
        )
        verify_us = (
            (now_ns - req.verification_start_ns) / 1000.0
            if req.verification_start_ns > 0
            else 0.0
        )
        base_us = (
            (req.drafting_start_ns - req.last_output_head_ns) / 1000.0
            if req.last_output_head_ns > 0 and req.drafting_start_ns > 0
            else 0.0
        )
        self._utility_scorer.record_iteration(
            depth_used=req.verified_depth,
            tokens_accepted=vresult.accepted_length,
            draft_time_us=draft_us,
            verify_time_us=verify_us,
            base_time_us=base_us,
        )

        # 5. Reset speculation state
        req.draft_complete = False
        req.draft_gating = None
        req.combined_draft = None
        req.verification_plan = None
        req.verification_analyses = []
        req.verified_depth = 0
        req.drafting_start_ns = 0
        req.verification_start_ns = 0
        req.mtp_step = 0
        req.mtp_steps_result = []
        req.mtp_cmd_seq = None
        req.mtp_phase = None
        req.mtp_mode = None
        req.mtp_sequential_verify = False
        req.mtp_batched_verify = False
        req.vb_len = 0
        req.vb_tokens = []
        req.mtp_catchup_queue = []
        req.mtp_post_verify = None
        req.mtp_draft_hidden_row = 0
        req.verify_tokens = []
        req.verify_idx = 0
        req.dspark_cmd_seq = None
        req.dspark_result = None
        req.dspark_anchor_pos = 0
        req.self_spec_step = 0
        req.self_spec_steps_result = []
        req.self_spec_cmd_seq = None
        req.self_spec_gating_rows = []
        req.self_spec_wait = False  # un-park from the routing-export gate
        req.prompt_lookup_tokens = []
        req.layer_similarities = {}
        req.current_skip_set = set()
        req.skip_set_history = []
        # Clean up origin entries for this request (don't wait for expiry)
        stale = [o for o in self._origin_transfers if o[0] == req.request_id]
        for o in stale:
            self._origin_transfers.pop(o, None)
            self._origin_birth_cycle.pop(o, None)
        req.speculation_state = SpeculationState.AUTOREGRESSIVE

        # 6. Continue autoregressive from last accepted position
        self._enqueue_next_embedding(req)

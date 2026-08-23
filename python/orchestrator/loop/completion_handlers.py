"""Completion handlers — per-type processing for daemon completions.

Each handler is called from ``_process_completion()`` in the main loop
when a matching ``CMP_*`` type arrives on the completion ring.  Handlers
update internal bookkeeping (resident expert sets, inflight transfers,
cmd_seq tracking) and trigger downstream effects (work item readiness
checks, verification expert readiness checks).

Methods here are mixed into ``OrchestratorLoop`` via ``_CompletionMixin``.

IMPORTANT: Keep this file richly commented.  The orchestrator loop is the
most complex module in the system and every helper must be self-documenting.
Do not remove comments during edits — add more if anything is unclear.
"""

from __future__ import annotations

import logging

from orchestrator.shm_protocol import (
    CHECKPOINT_GATING_OUTPUT,
    CHECKPOINT_HIDDEN_STATE,
    CHECKPOINT_LAYER_SIMILARITY,
    Completion,
)
from orchestrator.types import ExpertKey, SpeculationState, WorkStatus

logger = logging.getLogger(__name__)


class _CompletionMixin:
    """Mixin: completion-type handlers for the orchestrator loop."""

    # -- Checkpoint: hidden-state or gating-output readback from daemon --

    def _handle_checkpoint(self, cmp: Completion) -> None:
        """Store host-buffer offset for a mid-layer checkpoint.

        checkpoint_type 0 = hidden state (used by PROBE/PreScope training).
        checkpoint_type 1 = gating output (used by MoE-SpeQ training + SP-MoE).
        checkpoint_type 2 = layer similarity (daemon-computed cos_sim, #62g).
        During DRAFTING self-spec:
          - type 1 redirected to _accumulate_draft_gating for draft_gating.
          - type 2 stored in req.layer_similarities for LayerSkip.
        """
        # Intentionally NOT popping: a single cmd_seq can emit multiple checkpoints
        # (hidden state + gating at different layers). CMP_COMPUTE_DONE or CMP_ERROR
        # will pop it later. If neither arrives (daemon crash), entry leaks until
        # _finalize_request() sweeps — bounded by request lifetime. See TD-30.
        entry = self._cmd_seq_map.get(cmp.cmd_seq)
        if entry is None:
            return
        request_id, _ = entry
        req = self._requests.get(request_id)
        if req is None:
            return
        payload = cmp.payload.checkpoint

        # During DRAFTING self-spec, redirect gating checkpoints to draft_gating
        if (req.speculation_state == SpeculationState.DRAFTING
                and req.self_spec_cmd_seq is not None
                and payload.checkpoint_type == CHECKPOINT_GATING_OUTPUT):
            self._accumulate_draft_gating(req, cmp)
            return

        # During DRAFTING self-spec, collect per-layer cosine similarity
        # for LayerSkip skip set computation (#62g).  The daemon computes
        # cos_sim(h_before, h_after) per layer and emits it as a 4-byte
        # float32 checkpoint.  Dormant until daemon implements this.
        if (req.speculation_state == SpeculationState.DRAFTING
                and req.self_spec_cmd_seq is not None
                and payload.checkpoint_type == CHECKPOINT_LAYER_SIMILARITY):
            sim = self._read_similarity(payload)
            if sim is not None:
                req.layer_similarities[payload.layer_idx] = sim
            return

        layer = payload.layer_idx
        info = (payload.host_buf_offset, payload.data_bytes)
        if payload.checkpoint_type == CHECKPOINT_HIDDEN_STATE:
            req.hidden_state_checkpoints[layer] = info
        elif payload.checkpoint_type == CHECKPOINT_GATING_OUTPUT:
            req.gating_output_checkpoints[layer] = info

    def _read_similarity(self, payload) -> float | None:
        """Read a single float32 cosine similarity from checkpoint readback.

        Returns None if the host buffer is not available (e.g. unit tests).
        """
        if self._host_buf_base == 0 or payload.data_bytes < 4:
            return None
        import ctypes
        addr = self._host_buf_base + payload.host_buf_offset
        return float((ctypes.c_float * 1).from_address(addr)[0])

    # -- Transfer done: H2D completed outside ELM (legacy path) ----------

    def _handle_transfer_done(self, cmp: Completion) -> None:
        """Mark expert as resident after a non-ELM H2D transfer completes.

        direction 0 = H2D (expert now on GPU).  Also triggers readiness
        checks for work items waiting on this expert and for requests
        in PREFETCHING_VERIFY state.
        """
        payload = cmp.payload.transfer
        key = ExpertKey(payload.layer_idx, payload.expert_idx)
        gpu = cmp.gpu_idx
        self._inflight_transfers.get(gpu, set()).discard(key)
        # No _origin_transfers cleanup here — stale entries expire by cycle
        # count, and the inflight check in Phase 5 cancel processing filters
        # them out.  This avoids the TD-28 collision bug entirely.
        if payload.direction == 0:
            self._resident_keys.setdefault(gpu, set()).add(key)
            self._check_waiting_items_ready()
            self._check_prefetching_requests()

    # -- ELM expert ready: expert reached HOT state via lifecycle mgr ----

    def _handle_elm_expert_ready(self, cmp: Completion) -> None:
        """Mark expert as resident after ELM promotes it to HOT.

        Same downstream effects as _handle_transfer_done (direction=0).
        """
        payload = cmp.payload.elm_expert
        key = ExpertKey(payload.layer_idx, payload.expert_idx)
        gpu = cmp.gpu_idx
        self._inflight_transfers.get(gpu, set()).discard(key)
        self._resident_keys.setdefault(gpu, set()).add(key)
        self._check_waiting_items_ready()
        self._check_prefetching_requests()

    # -- ELM expert evicted: expert removed from GPU cache ---------------

    def _handle_elm_expert_evicted(self, cmp: Completion) -> None:
        """Remove expert from resident set after ELM eviction completes."""
        payload = cmp.payload.elm_expert
        key = ExpertKey(payload.layer_idx, payload.expert_idx)
        gpu = cmp.gpu_idx
        self._resident_keys.get(gpu, set()).discard(key)

    # -- Cancel done: command cancellation acknowledged ------------------

    def _handle_cancel_done(self, cmp: Completion) -> None:
        """Clean up cmd_seq tracking after a command is cancelled."""
        payload = cmp.payload.cancel_result
        if not payload.cancelled:
            logger.warning(
                "cancel_done cmd_seq=%d: cancellation of target_cmd_seq=%d"
                " was NOT acknowledged by daemon",
                cmp.cmd_seq, payload.target_cmd_seq,
            )
        self._cmd_seq_map.pop(cmp.cmd_seq, None)

    # -- Error: command failed -------------------------------------------

    def _handle_error(self, cmp: Completion) -> None:
        """Handle a command failure from the daemon.

        Checks three tracking dicts in order:
          1. Draft step command (MTP/self-spec in _draft_cmd_seqs) — abort
             the draft attempt and return to autoregressive.  The draft KV
             fork is freed, all draft state is reset.
          2. Verification intermediate (in _verify_cmd_seqs) — if the failed
             command is a verification intermediate, abort the entire
             verification for that request instead of leaving it stuck in
             VERIFYING state (TD-21).
          3. Normal tracked command (in _cmd_seq_map) — cancel all work
             items for the owning request and remove it entirely.
        """
        # Log the daemon error details for diagnostics (TD-59f).
        err = cmp.payload.error
        msg = err.message.decode("utf-8", errors="replace").rstrip("\x00")
        logger.error("daemon error cmd_seq=%d cat=%d: %s",
                     cmp.cmd_seq, err.error_category, msg)

        # Routing-export slot gate release (TD-ORCH-ROUTING-EXPORT-MULTI /
        # INV-IPC-6b): if the failed command was the in-flight slot
        # producer, its export will never be captured — release the gate
        # here or every subsequent gating-bearing dispatch deadlocks.
        if self._gating_attn_inflight == cmp.cmd_seq:
            self._gating_attn_inflight = None

        # 1. Draft step error: abort draft, revert to autoregressive
        request_id = self._draft_cmd_seqs.pop(cmp.cmd_seq, None)
        if request_id is not None:
            self._cmd_seq_map.pop(cmp.cmd_seq, None)
            req = self._requests.get(request_id)
            if req is not None:
                # TD-DSPARK-CTX-CAP graceful degradation: a failed DSpark
                # step means the runtime declined this drafting context
                # (context-arena overflow / invalidated capture — fail-
                # closed states that never recover within one request,
                # since fed positions only grow).  Disable further DSpark
                # rounds for THIS request so the loop stops re-issuing a
                # step that errors every round; decode continues plain-AR
                # + prompt-lookup, lossless.
                if (req.dspark_cmd_seq is not None
                        and cmp.cmd_seq == req.dspark_cmd_seq
                        and not req.dspark_draft_disabled):
                    req.dspark_draft_disabled = True
                    logger.warning(
                        "request %d: DSpark step declined by the daemon "
                        "(%s) — drafting disabled for this request, decode "
                        "continues autoregressive", request_id, msg)
                self._abort_draft(req)
            return
        # 2. Verification intermediate error: abort verification + draft
        request_id = self._verify_cmd_seqs.pop(cmp.cmd_seq, None)
        if request_id is not None:
            self._abort_verification(request_id)
            return
        # 3. Normal command error: fatal for the request (TD-59a)
        entry = self._cmd_seq_map.get(cmp.cmd_seq)
        if entry is None:
            return
        request_id, _ = entry
        req = self._requests.get(request_id)
        if req is None:
            self._cmd_seq_map.pop(cmp.cmd_seq, None)
            return
        self._finalize_request(req, finish_reason="error")

    # -- GPU fatal: device-level permanent error -------------------------

    def _handle_gpu_fatal(self, cmp: Completion) -> None:
        """Handle a GPU device-fatal notification (once per GPU).

        The daemon sends CMP_GPU_FATAL when a GPU's CUDA context enters an
        unrecoverable error state.  All pending commands on this GPU will
        also receive individual CMP_ERROR completions.
        """
        gpu = cmp.gpu_idx
        msg = cmp.payload.gpu_fatal.message.decode("utf-8", errors="replace")
        vendor_err = cmp.payload.gpu_fatal.vendor_error_code
        logger.critical("GPU %d device-fatal (vendor err=%d): %s",
                        gpu, vendor_err, msg)
        # TODO: trigger engine shutdown for TP GPUs, graceful degrade for non-TP

    # -- Readiness checks ------------------------------------------------

    def _check_waiting_items_ready(self) -> None:
        """Promote WAITING_TRANSFER work items to READY when experts arrive.

        Scans all WAITING_TRANSFER items and checks their required_experts
        against the current resident set for their target GPU.
        """
        for item in self._work_queue.by_status(WorkStatus.WAITING_TRANSFER):
            gpu = item.target_gpu
            resident = self._resident_keys.get(gpu, set())
            if all(k in resident for k in item.required_experts):
                self._work_queue.update_status(item, WorkStatus.READY)

    def _check_prefetching_requests(self) -> None:
        """Check if any PREFETCHING_VERIFY requests can start verification.

        Called after expert arrivals.  Delegates to _check_verification_ready
        (from _SpeculationMixin) for each qualifying request.
        """
        for req in self._requests.values():
            if req.speculation_state == SpeculationState.PREFETCHING_VERIFY:
                self._check_verification_ready(req)

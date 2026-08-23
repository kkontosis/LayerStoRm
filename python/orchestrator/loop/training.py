"""Training sample collection — feed checkpoint data to online predictors.

After each MoE layer completion, the orchestrator pairs:
  - Hidden-state checkpoints (from earlier ATTENTION emit_checkpoint=1)
  - Gating-output checkpoints (from MoE store_gating_output=1)

These (feature, label) pairs are fed to three online-trained predictors:
  - PROBE:    multi-layer MLP, predicts expert activation from hidden states.
  - MoE-SpeQ: per-source MLP, predicts expert activation from prior MoE output.
  - PreScope: one-layer-ahead gating prediction from previous layer hidden state.

Host buffer reads use ctypes zero-copy from shared memory (the C++ daemon
writes checkpoint data to a pre-allocated host buffer region).

Methods here are mixed into OrchestratorLoop via _TrainingMixin.

IMPORTANT: Keep this file richly commented.  The orchestrator loop is the
most complex module in the system and every helper must be self-documenting.
Do not remove comments during edits — add more if anything is unclear.
"""

from __future__ import annotations

import ctypes

import numpy as np


class _TrainingMixin:
    """Mixin: training sample collection and host buffer access."""

    def _feed_training_samples(self, req, layer_idx: int) -> None:
        """Pair gating output at layer_idx with hidden states for training.

        Called from _handle_compute_done after each EXPERT_FFN completion.
        Feeds samples to PROBE (for each probe point), MoE-SpeQ (for each
        earlier MoE source layer), and PreScope (previous layer).
        """
        gating_ckpt = req.gating_output_checkpoints.get(layer_idx)
        if gating_ckpt is None:
            return

        # PROBE: for each probe point that precedes this layer,
        # pair its hidden state with this layer's gating mask
        if self._probe.enabled:
            for pp_layer in self._probe.probe_layers:
                if pp_layer < layer_idx:
                    hs_ckpt = req.hidden_state_checkpoints.get(pp_layer)
                    if hs_ckpt is not None:
                        hs = self._read_host_buf(hs_ckpt)
                        mask = self._read_gating_mask(gating_ckpt)
                        if hs is not None and mask is not None:
                            self._probe.record_training_sample(
                                pp_layer, hs, layer_idx, mask,
                            )

        # MoE-SpeQ: for each earlier MoE layer, pair its hidden state
        # with this layer's gating mask
        if self._moe_speq.enabled:
            for src_layer in range(self._first_moe_layer, layer_idx):
                if self._moe_speq.is_moe_layer(src_layer):
                    hs_ckpt = req.hidden_state_checkpoints.get(src_layer)
                    if hs_ckpt is not None:
                        hs = self._read_host_buf(hs_ckpt)
                        mask = self._read_gating_mask(gating_ckpt)
                        if hs is not None and mask is not None:
                            self._moe_speq.record_training_sample(
                                src_layer, hs, layer_idx, mask,
                            )

        # PreScope: pair previous layer's hidden state with this gating mask
        if self._prescope.enabled and layer_idx > 0:
            prev_hs = req.hidden_state_checkpoints.get(layer_idx - 1)
            if prev_hs is not None:
                hs = self._read_host_buf(prev_hs)
                mask = self._read_gating_mask(gating_ckpt)
                if hs is not None and mask is not None:
                    self._prescope.record_training_sample(
                        layer_idx - 1, hs, mask,
                    )

    def _read_host_buf(
        self, ckpt: tuple[int, int],
    ) -> np.ndarray | None:
        """Read raw float32 data from the shared host buffer.

        Returns a copied numpy array (the source memory is shared with
        the C++ daemon and may be overwritten next cycle).  Returns None
        if host_buf_base is not set (e.g. in unit tests).
        """
        if self._host_buf_base == 0:
            return None
        offset, nbytes = ckpt
        if nbytes == 0:
            return None
        count = nbytes // 4
        return np.frombuffer(
            (ctypes.c_char * nbytes).from_address(
                self._host_buf_base + offset
            ),
            dtype=np.float32,
            count=count,
        ).copy()

    def _read_gating_mask(
        self, ckpt: tuple[int, int],
    ) -> np.ndarray | None:
        """Read gating scores from host buffer and convert to binary top-K mask.

        Returns a float32 array of shape (num_experts,) with 1.0 at the
        top-K positions and 0.0 elsewhere.  K = min(8, num_experts).
        Used as the label for PROBE/MoE-SpeQ/PreScope training.
        """
        buf = self._read_host_buf(ckpt)
        if buf is None:
            return None
        ne = self._metadata.num_experts
        if len(buf) < ne:
            return None
        scores = buf[:ne]
        mask = np.zeros(ne, dtype=np.float32)
        topk = min(8, ne)
        top_indices = np.argpartition(scores, -topk)[-topk:]
        mask[top_indices] = 1.0
        return mask

"""EPM-1 block-metadata manifest writer (Phase 29, spec/MoE-SpeQ_NOTES.md §7).

The C++ daemon dumps the raw per-block binaries (draft per-backbone-layer
hiddens -> epm_blocks.bin; per-position verify routing -> epm_routing.bin)
but only the ORCHESTRATOR knows the post-verification metadata: the
accepted length, the combined verify tokens actually teacher-forced, and
the calibrated/raw confidence stream.  This module appends one JSONL line
per DSpark block to <dump_dir>/manifest.jsonl at round end
(_record_sequential_round), joined with the daemon records on
(seq_id, anchor_pos) by tools/elb_train/dataset.py.

Manifest entry fields (one JSON object per line):
  seq_id        engine sequence id (u64) — the join key half 1
  anchor_pos    block anchor position — the join key half 2
  block_idx     orchestrator-side per-sequence round counter (convenience;
                the daemon keeps its own equivalent counter in the EPMB
                record — both count successful DSpark steps per sequence)
  request_id    orchestrator request id (provenance)
  gamma         DSpark draft depth this round (len(dspark tokens))
  draft_tokens  the γ DSpark-sampled ids (the positions the EPM predicts)
  verify_tokens the tokens actually teacher-forced by sequential verify
                (the DraftCombiner output — may differ from draft_tokens
                when prompt-lookup won the combine; tokens_match flags it)
  tokens_match  verify_tokens[:gamma] == draft_tokens (label validity for
                positions >= 1; position 0's feed is always the anchor)
  accepted_len  accepted draft prefix length (sequential early-stop)
  attempted_len len(verify_tokens) (<= speculation depth)
  confidences   raw DSP-6 c_k (γ floats; [] when the head is off)
  ts            wall-clock seconds (provenance only)

Zero overhead when disabled: the orchestrator loop holds None instead of a
writer and skips on a single `is not None` check.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def effective_epm_dump_dir(config_dir: str) -> str:
    """Resolve the EPM dump directory exactly like the C++ side
    (speculation::epm_dump_dir): LS_EPM_DUMP env non-empty and not "0"
    overrides; LS_EPM_DUMP=0 forces OFF; otherwise the config value.
    Empty result = dumping disabled.
    """
    env = os.environ.get("LS_EPM_DUMP", "")
    if env:
        return "" if env == "0" else env
    return config_dir or ""


class EpmManifestWriter:
    """Append-mode JSONL manifest for EPM-1 block metadata.

    Fails SOFT on I/O errors (disables itself) — data collection must
    never take the serving loop down.
    """

    def __init__(self, dump_dir: str) -> None:
        self._fp = None
        self._block_counts: dict[int, int] = {}
        try:
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
            self._path = str(Path(dump_dir) / "manifest.jsonl")
            self._fp = open(self._path, "a", encoding="utf-8")
        except OSError:
            self._fp = None

    @property
    def ok(self) -> bool:
        return self._fp is not None

    def append_block(
        self,
        *,
        seq_id: int,
        anchor_pos: int,
        request_id: int,
        draft_tokens: list[int],
        verify_tokens: list[int],
        accepted_len: int,
        confidences: list[float],
    ) -> None:
        """Append one block's metadata line (flushed immediately)."""
        if self._fp is None:
            return
        gamma = len(draft_tokens)
        block_idx = self._block_counts.get(seq_id, 0)
        self._block_counts[seq_id] = block_idx + 1
        entry = {
            "seq_id": int(seq_id),
            "anchor_pos": int(anchor_pos),
            "block_idx": block_idx,
            "request_id": int(request_id),
            "gamma": gamma,
            "draft_tokens": [int(t) for t in draft_tokens],
            "verify_tokens": [int(t) for t in verify_tokens],
            "tokens_match": list(verify_tokens[:gamma]) == list(draft_tokens),
            "accepted_len": int(accepted_len),
            "attempted_len": len(verify_tokens),
            "confidences": [float(c) for c in confidences],
            "ts": time.time(),
        }
        try:
            self._fp.write(json.dumps(entry) + "\n")
            self._fp.flush()
        except OSError:
            try:
                self._fp.close()
            finally:
                self._fp = None

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            finally:
                self._fp = None

"""EPM-1 unit tests: python/orchestrator/epm_manifest.py (JSONL block
manifest + config/env gate resolution — must mirror the C++ side's
speculation::epm_dump_dir semantics exactly)."""

from __future__ import annotations

import json

import pytest

from orchestrator.epm_manifest import EpmManifestWriter, effective_epm_dump_dir


class TestEffectiveDumpDir:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("LS_EPM_DUMP", raising=False)
        assert effective_epm_dump_dir("") == ""

    def test_config_value_used(self, monkeypatch):
        monkeypatch.delenv("LS_EPM_DUMP", raising=False)
        assert effective_epm_dump_dir("/tmp/epm_cfg") == "/tmp/epm_cfg"

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("LS_EPM_DUMP", "/tmp/epm_env")
        assert effective_epm_dump_dir("/tmp/epm_cfg") == "/tmp/epm_env"

    def test_env_zero_forces_off(self, monkeypatch):
        monkeypatch.setenv("LS_EPM_DUMP", "0")
        assert effective_epm_dump_dir("/tmp/epm_cfg") == ""


class TestManifestWriter:
    def test_append_and_round_trip(self, tmp_path):
        w = EpmManifestWriter(str(tmp_path / "dump"))
        assert w.ok
        w.append_block(seq_id=7, anchor_pos=100, request_id=1,
                       draft_tokens=[11, 12, 13],
                       verify_tokens=[11, 12, 13, 14],
                       accepted_len=2, confidences=[0.9, 0.5, 0.25])
        w.append_block(seq_id=7, anchor_pos=103, request_id=1,
                       draft_tokens=[21, 22, 23],
                       verify_tokens=[21, 99],  # combiner diverged
                       accepted_len=1, confidences=[])
        w.append_block(seq_id=8, anchor_pos=5, request_id=2,
                       draft_tokens=[31], verify_tokens=[31],
                       accepted_len=1, confidences=[0.7])
        w.close()

        lines = (tmp_path / "dump" / "manifest.jsonl").read_text().strip() \
            .splitlines()
        entries = [json.loads(ln) for ln in lines]
        assert len(entries) == 3

        e0, e1, e2 = entries
        assert (e0["seq_id"], e0["anchor_pos"]) == (7, 100)
        assert e0["block_idx"] == 0
        assert e0["gamma"] == 3
        assert e0["draft_tokens"] == [11, 12, 13]
        assert e0["verify_tokens"] == [11, 12, 13, 14]
        assert e0["tokens_match"] is True  # verify[:gamma] == draft
        assert e0["accepted_len"] == 2
        assert e0["attempted_len"] == 4
        assert e0["confidences"] == pytest.approx([0.9, 0.5, 0.25])

        # Per-seq block counter + tokens_match=False when the combiner
        # verified a different stream than the dspark draft.
        assert e1["block_idx"] == 1
        assert e1["tokens_match"] is False
        assert e2["block_idx"] == 0  # new sequence restarts the counter
        assert e2["request_id"] == 2

    def test_append_after_close_is_noop(self, tmp_path):
        w = EpmManifestWriter(str(tmp_path))
        w.close()
        w.append_block(seq_id=1, anchor_pos=0, request_id=0,
                       draft_tokens=[1], verify_tokens=[1],
                       accepted_len=1, confidences=[])
        assert not w.ok

    def test_unwritable_dir_fails_soft(self):
        w = EpmManifestWriter("/proc/definitely/not/writable")
        assert not w.ok
        # Must not raise.
        w.append_block(seq_id=1, anchor_pos=0, request_id=0,
                       draft_tokens=[1], verify_tokens=[1],
                       accepted_len=0, confidences=[])

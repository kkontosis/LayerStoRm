"""TD-PREFIX-TIDY-COLD-SPILL — orchestrator-side spill POLICY tests.

The engine byte-move mechanism (spill/unspill, byte cap, refcounts,
byte-exact reload) is covered by tests/unit/kv_tiering_test.cpp
(KvTieringSpill.*).  Here: the orchestrator policy over the scripted
FakeDaemon — config parsing, spill-directory lifecycle (creation,
writability probe + loud disable, stale-file reclamation), the idle
sweep (leaf-only, age-gated, one per pass, background only), the
cap-refusal eviction over the spill directory, and the lost-spill-file
hit fallback (degrades to a MISS, never fails the request).
"""
from __future__ import annotations

import os
import time

import pytest

from test_bridge_ring import FakeDaemon, chain  # noqa: F401
from test_orchestrator_bridge import (_Sink, _finish, _make, _meta,
                                      _serve)

from bridge.ring_bridge import BridgeError
from orchestrator.orchestrator import (InferenceRequest, Orchestrator,
                                       PrefixCacheConfig,
                                       PrefixSpillConfig,
                                       SpeculationConfig)


def _orch_spill(tmp_path, daemon_gamma: int = 5, *, spill_kw=None,
                pc_kw=None):
    bridge, daemon, _ = _make(gamma=daemon_gamma, use_far=True)
    sp = dict(enabled=True, path=str(tmp_path / "spill"),
              max_mib=1024, idle_seconds=0.0)
    sp.update(spill_kw or {})
    orch = Orchestrator(
        bridge, metadata=_meta(),
        speculation=SpeculationConfig(enabled=True, gamma=daemon_gamma,
                                      conf_thresh=0.0),
        prefix_cache=PrefixCacheConfig(**(pc_kw or {})),
        prefix_spill=PrefixSpillConfig(**sp))
    return orch, daemon


def _register_holder(orch, seed, rid=1):
    prompt = chain(seed, 130)  # 2 x 64-token chunks + tail (grid holder)
    sink = _Sink()
    _serve(orch, InferenceRequest(
        request_id=rid, prompt_token_ids=list(prompt), max_tokens=4,
        on_complete=sink.on_complete))
    assert sink.done is not None and sink.done[2] != "error"
    pc = orch.prefix_cache
    assert pc is not None and pc._entries, "holder must be registered"
    return pc, prompt


def test_config_parse_defaults_and_overrides():
    cfg = PrefixSpillConfig.from_config({})
    assert cfg.enabled is True                     # DEFAULT TRUE
    assert cfg.path == "~/.layerstorm/prefix-cache"
    assert cfg.max_mib == 32768                    # mandatory cap default
    assert cfg.idle_seconds == 300.0
    cfg = PrefixSpillConfig.from_config({"_internal-prefix_spill": {
        "enabled": False, "path": "/x", "max_mib": 7, "idle_seconds": 1}})
    assert (cfg.enabled, cfg.path, cfg.max_mib, cfg.idle_seconds) \
        == (False, "/x", 7, 1.0)


def test_spill_dir_created_and_stale_files_reclaimed(tmp_path):
    d = tmp_path / "spill"
    d.mkdir()
    stale = d / "ls-spill-123-seq9.kvspill"
    stale.write_bytes(b"dead")
    keep = d / "unrelated.bin"
    keep.write_bytes(b"live")
    orch, daemon = _orch_spill(tmp_path)
    try:
        assert orch._spill is not None
        assert not stale.exists(), "stale spill files must be reclaimed"
        assert keep.exists(), "non-spill files are never touched"
    finally:
        _finish(daemon)


def test_unwritable_path_disables_loudly_never_fails_requests(tmp_path,
                                                              capsys):
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"")               # a FILE where a dir must go
    orch, daemon = _orch_spill(tmp_path,
                               spill_kw={"path": str(blocker / "sub")})
    try:
        assert orch._spill is None, "feature must disable, not raise"
        assert "DISABLED" in capsys.readouterr().out
        sink = _Sink()
        _serve(orch, InferenceRequest(          # requests still serve
            request_id=3, prompt_token_ids=[7, 8, 9], max_tokens=4,
            on_complete=sink.on_complete))
        assert sink.done is not None and sink.done[2] != "error"
    finally:
        _finish(daemon)


def test_idle_sweep_spills_lru_leaf_once(tmp_path):
    orch, daemon = _orch_spill(tmp_path)
    try:
        pc, _ = _register_holder(orch, 100)
        leaf = [e for e in pc._entries if not e.children][0]
        daemon.spill_answers.append((0, 18))     # engine spilled 18 pages
        orch._spill_last_sweep = 0.0
        orch._spill_sweep()
        assert [s[0] for s in daemon.spills] == [leaf.seq_id]
        assert leaf.spilled is True
        # A second sweep never re-issues for an already-spilled holder.
        orch._spill_last_sweep = 0.0
        orch._spill_sweep()
        assert len(daemon.spills) == 1
    finally:
        _finish(daemon)


def test_sweep_respects_idle_age_and_engine_decline(tmp_path):
    orch, daemon = _orch_spill(tmp_path,
                               spill_kw={"idle_seconds": 3600.0})
    try:
        pc, _ = _register_holder(orch, 200)
        orch._spill_last_sweep = 0.0
        orch._spill_sweep()
        assert daemon.spills == [], "a fresh holder is not idle"
        # Age the holder; the engine's no-tiering arm answers (0, 0) —
        # the holder is marked attempted and never re-issued.
        leaf = [e for e in pc._entries if not e.children][0]
        leaf.wall_touch = time.monotonic() - 7200
        orch._spill_last_sweep = 0.0
        orch._spill_sweep()
        assert len(daemon.spills) == 1
        assert leaf.spilled is False and leaf.spill_attempted is True
        orch._spill_last_sweep = 0.0
        orch._spill_sweep()
        assert len(daemon.spills) == 1
    finally:
        _finish(daemon)


def test_cap_refusal_evicts_lru_spilled_leaf_and_retries(tmp_path):
    orch, daemon = _orch_spill(tmp_path)
    try:
        pc, _ = _register_holder(orch, 300, rid=1)
        _register_holder(orch, 400, rid=2)
        leaves = sorted((e for e in pc._entries if not e.children),
                        key=lambda e: e.last_used)
        assert len(leaves) >= 2
        old, new = leaves[0], leaves[1]
        old.spilled = True                       # already on disk (LRU)
        daemon.spill_answers.append((2, 0))      # cap refusal
        daemon.spill_answers.append((0, 9))      # retry lands
        before = pc.evictions
        orch._spill_holder(new, retry_on_cap=True)
        assert old not in pc._entries, "LRU spilled leaf must be evicted"
        assert pc.evictions == before + 1
        assert new.spilled is True
        assert [s[0] for s in daemon.spills] == [new.seq_id, new.seq_id]
    finally:
        _finish(daemon)


def test_lost_spill_file_hit_degrades_to_miss(tmp_path):
    orch, daemon = _orch_spill(tmp_path)
    try:
        pc, prompt = _register_holder(orch, 500, rid=1)
        leaf = [e for e in pc._entries if not e.children][0]
        leaf.spilled = True
        # The engine-side reload fails (lost file → retryable category);
        # every fork attempt raises until the orchestrator gives up on
        # the holder and serves the request as a MISS.
        real_fork = orch.bridge.fork_sequence

        def failing_fork(*a, **k):
            raise BridgeError("seq_fork: spilled holder reload failed "
                              "(cold pool exhausted or spill file "
                              "unreadable) — evict and retry")

        orch.bridge.fork_sequence = failing_fork
        try:
            sink = _Sink()
            _serve(orch, InferenceRequest(
                request_id=9, prompt_token_ids=list(prompt), max_tokens=4,
                on_complete=sink.on_complete))
            assert sink.done is not None and sink.done[2] != "error", \
                "a lost spill file must NEVER fail the request"
        finally:
            orch.bridge.fork_sequence = real_fork
        assert leaf not in pc._entries, "the dead holder must be evicted"
        assert orch.last_stats.prefix_hit_tokens == 0, "served as a MISS"
    finally:
        _finish(daemon)

"""Tests for orchestrator.engine_glue (#91) — real-daemon loop wiring.

Uses null backends (no CUDA required).  Validates that build_engine_loop
attaches a fully-constructed OrchestratorLoop to a LIVE engine's IPC
surfaces (SPSC rings, StateSnapshot seqlock, sideband) and that empty
cycles run against the real daemon thread without touching the GIL from
the daemon side.  The full compute chain needs CUDA and is covered by
tests/integration/test_orch_engine_decode.py (the #91 e2e gate).
"""

import pathlib

import pytest

import layerstorm_engine
from orchestrator.engine_glue import (
    build_engine_loop,
    ep_gpu_indices_from_config,
    metadata_from_engine_info,
)
from orchestrator.loop.orchestrator_loop import OrchestratorConfig

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_TEST_CONFIG = str(_PROJECT_ROOT / "test-data" / "config" / "valid_deepseek_v3_2.json")


@pytest.fixture
def engine():
    info = layerstorm_engine.start_engine_test(_TEST_CONFIG)
    yield info
    layerstorm_engine.stop_engine()


class TestMetadataFromEngineInfo:
    def test_fields_translated(self, engine):
        buffer_ids = layerstorm_engine.query_buffer_ids()
        md = metadata_from_engine_info(engine, buffer_ids=buffer_ids)
        assert md.num_gpus == engine.num_gpus
        assert md.num_layers == engine.num_layers
        assert md.num_moe_layers == engine.num_moe_layers
        assert md.num_experts == engine.num_experts
        assert md.expert_bytes == engine.expert_bytes
        assert len(md.gpus) == engine.num_gpus

    def test_buffer_id_prefix_match(self, engine):
        # Null backends may not register the hidden/logits buffers — the
        # translation must then default to 0 (unit-safe) without raising.
        md = metadata_from_engine_info(
            engine,
            buffer_ids={"hidden_state.attn.rank0.gpu0": 7,
                        "logits_scratch.pos0": 9,
                        "unrelated": 3},
        )
        assert md.hidden_buf_id == 7
        assert md.logits_buf_id == 9


class TestBuildEngineLoop:
    def test_loop_attaches_and_cycles(self, engine):
        loop, handles = build_engine_loop(
            engine,
            buffer_ids=layerstorm_engine.query_buffer_ids(),
            orchestrator_config=OrchestratorConfig(
                cycle_budget_us=0.0, max_idle_wait_us=0.0,
            ),
        )
        # Live IPC attach: state reader sees the daemon's seqlock snapshot.
        assert handles.state_reader.cycle_count() >= 0
        assert handles.sideband_base == int(engine.ipc_base) + int(
            engine.sideband_offset)
        # host_buf_base aliases the sideband (readback offsets are
        # sideband-relative by the IPC contract).
        assert loop._host_buf_base == handles.sideband_base
        # Empty cycles against the real daemon are safe and cheap.
        for _ in range(3):
            metrics = loop.run_one_cycle()
        assert metrics.commands_dispatched == 0
        assert loop.cycle_count == 3

    def test_default_speculation_off(self, engine):
        loop, _ = build_engine_loop(engine)
        assert loop._utility_scorer.recommended_depth() == 0
        assert loop._mtp_active() is False
        assert loop._dspark_active() is False


class TestEpGpuIndicesFromConfig:
    """_internal-orchestrator.ep_gpu_indices resolution (E2 amendment):
    schema value wins when present; the driver's derived value is the
    fallback; absent/empty is byte-identical to prior behavior."""

    def test_absent_section_uses_fallback(self):
        assert ep_gpu_indices_from_config({}) == ()
        assert ep_gpu_indices_from_config(None) == ()
        assert ep_gpu_indices_from_config({}, fallback=(0, 1)) == (0, 1)

    def test_empty_field_uses_fallback(self):
        cfg = {"_internal-orchestrator": {"ep_gpu_indices": []}}
        assert ep_gpu_indices_from_config(cfg, fallback=(0, 1)) == (0, 1)
        cfg = {"_internal-orchestrator": {}}
        assert ep_gpu_indices_from_config(cfg, fallback=(2,)) == (2,)

    def test_schema_value_wins_over_fallback(self):
        cfg = {"_internal-orchestrator": {"ep_gpu_indices": [0, 1]}}
        assert ep_gpu_indices_from_config(cfg, fallback=(0, 1, 2)) == (0, 1)

    def test_coerced_to_int_tuple(self):
        cfg = {"_internal-orchestrator": {"ep_gpu_indices": [1, 0]}}
        out = ep_gpu_indices_from_config(cfg)
        assert out == (1, 0)
        assert all(isinstance(g, int) for g in out)

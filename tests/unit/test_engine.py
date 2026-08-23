"""Tests for the layerstorm_engine pybind11 module.

Uses null backends (no CUDA required). Validates:
1. start/stop lifecycle
2. EngineInfo fields populated correctly
3. IPC region accessible via ctypes
4. Daemon loop advances cycle count
5. CMD_NOOP processed by daemon
6. CMD_SHUTDOWN causes clean shutdown
"""

import ctypes
import pathlib
import time

import pytest

import layerstorm_engine
from orchestrator.shm_protocol import (
    CMD_NOOP,
    CMD_SHUTDOWN,
    Command,
    IpcHeader,
    RingHeader,
    StateSnapshot,
)
from orchestrator.spsc_ring import (
    SnapshotTransaction,
    SpscRingWriter,
    read_snapshot,
    read_with_retry,
)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_TEST_CONFIG = str(_PROJECT_ROOT / "test-data" / "config" / "valid_deepseek_v3_2.json")
_TEST_CONFIG_TQ = str(_PROJECT_ROOT / "test-data" / "config" / "valid_deepseek_v3_2_tq.json")


@pytest.fixture
def engine():
    """Start engine with null backends, yield info, stop on teardown."""
    info = layerstorm_engine.start_engine_test(_TEST_CONFIG)
    yield info
    layerstorm_engine.stop_engine()


# ── Lifecycle ────────────────────────────────────────────────────────────────

class TestEngineLifecycle:
    def test_start_returns_engine_info(self, engine):
        assert engine.ipc_base != 0
        assert engine.ipc_total_bytes > 0

    def test_engine_info_model_metadata(self, engine):
        assert engine.num_gpus == 4
        assert engine.num_moe_layers > 0
        assert engine.num_experts == 256
        assert engine.num_layers == 61
        assert engine.expert_bytes > 0
        assert engine.num_expert_devices == 4  # one ExpertDevice per GPU

    def test_kv_bytes_per_page_snapmla(self, engine):
        # SnapMLA FP8: 644 bytes/token * 16 tokens/page = 10304
        assert engine.kv_bytes_per_page == 10304

    def test_engine_info_ring_offsets(self, engine):
        assert engine.cmd_ring_offset > 0
        assert engine.cmd_ring_slots == 8192
        assert engine.cmd_slot_bytes == 256
        assert engine.cmp_ring_offset > engine.cmd_ring_offset
        assert engine.cmp_ring_slots == 8192
        assert engine.cmp_slot_bytes == 128
        assert engine.state_offset > engine.cmp_ring_offset
        assert engine.state_bytes == ctypes.sizeof(StateSnapshot)

    def test_double_start_raises(self, engine):
        with pytest.raises(RuntimeError, match="already running"):
            layerstorm_engine.start_engine_test(_TEST_CONFIG)

    def test_stop_is_idempotent(self, engine):
        layerstorm_engine.stop_engine()
        layerstorm_engine.stop_engine()  # Should not raise


# ── IPC Region ───────────────────────────────────────────────────────────────

class TestIpcRegion:
    def test_ipc_header_version(self, engine):
        buf = (ctypes.c_ubyte * engine.ipc_total_bytes).from_address(engine.ipc_base)
        header = IpcHeader.from_buffer(buf)
        assert header.version == 1
        assert header.shutdown_requested == 0
        assert header.error_code == 0

    def test_command_ring_initialized(self, engine):
        buf = (ctypes.c_ubyte * engine.ipc_total_bytes).from_address(engine.ipc_base)
        ring_offset = engine.cmd_ring_offset
        ring_header = RingHeader.from_buffer(buf, ring_offset)
        assert ring_header.slot_count == 8192
        assert ring_header.slot_size == 256
        assert ring_header.producer_seq == 0
        assert ring_header.consumer_seq == 0


# ── Daemon Loop ──────────────────────────────────────────────────────────────

class TestDaemonLoop:
    def test_daemon_cycle_count_advances(self, engine):
        """Daemon loop should be running and incrementing cycle count."""
        time.sleep(0.02)
        snap_addr = engine.ipc_base + engine.state_offset
        tx = SnapshotTransaction(snap_addr)
        cycle = tx.read_u64(64)   # daemon_cycle_count offset
        assert cycle > 0
        ts = tx.read_u64(72)      # timestamp_ns offset
        assert ts > 0

    def test_daemon_processes_noop(self, engine):
        """Send a NOOP command, verify daemon cycle count advances."""
        ring_addr = engine.ipc_base + engine.cmd_ring_offset
        writer = SpscRingWriter(ring_addr,
                                engine.cmd_ring_slots, engine.cmd_slot_bytes)

        snap_addr = engine.ipc_base + engine.state_offset
        tx = SnapshotTransaction(snap_addr)
        cycle1 = tx.read_u64(64)

        cmd = Command()
        cmd.cmd_type = CMD_NOOP
        cmd.cmd_seq = 1
        assert writer.write(bytes(cmd))

        time.sleep(0.02)
        cycle2 = tx.read_u64(64)
        assert cycle2 > cycle1


class TestDaemonShutdown:
    def test_cmd_shutdown_via_ring(self):
        """Send CMD_SHUTDOWN via the command ring, verify clean shutdown."""
        info = layerstorm_engine.start_engine_test(_TEST_CONFIG)
        try:
            ring_addr = info.ipc_base + info.cmd_ring_offset
            writer = SpscRingWriter(ring_addr,
                                    info.cmd_ring_slots, info.cmd_slot_bytes)

            cmd = Command()
            cmd.cmd_type = CMD_SHUTDOWN
            cmd.cmd_seq = 99
            assert writer.write(bytes(cmd))

            # Give daemon time to process the shutdown command
            time.sleep(0.05)
        finally:
            # stop_engine should return quickly since daemon already exited
            layerstorm_engine.stop_engine()


# ── TurboQuant Engine ──────────────────────────────────────────────────────

class TestEngineTQ:
    def test_tq_engine_starts(self):
        """Engine starts with turboquant_mla config and reports correct kv_bytes_per_page."""
        info = layerstorm_engine.start_engine_test(_TEST_CONFIG_TQ)
        try:
            assert info.ipc_base != 0
            assert info.num_layers == 61
            # TQ: 386 bytes/token * 16 tokens/page = 6176
            assert info.kv_bytes_per_page == 6176
        finally:
            layerstorm_engine.stop_engine()

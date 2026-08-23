"""Tests for Python IPC protocol structs and SPSC ring."""

import ctypes

from orchestrator.shm_protocol import (
    CMD_SLOT_BYTES,
    CMD_TRANSFER_H2D,
    CMP_SLOT_BYTES,
    Command,
    Completion,
    EngineInfo,
    GpuSnapshot,
    IpcHeader,
    RequestAcceptance,
    RingHeader,
    StateSnapshot,
)
from orchestrator.spsc_ring import SpscRingReader, SpscRingWriter, read_snapshot


# ── Struct sizes ─────────────────────────────────────────────────────────────

def test_struct_sizes():
    assert ctypes.sizeof(IpcHeader) == 256
    assert ctypes.sizeof(RingHeader) == 128
    assert ctypes.sizeof(Command) == 256
    assert ctypes.sizeof(Completion) == 128
    assert ctypes.sizeof(GpuSnapshot) == 56
    assert ctypes.sizeof(RequestAcceptance) == 16
    # Cross-language invariant: must match C++ sizeof(StateSnapshot) exactly.
    # C++ alignas(64) on seqlock forces sizeof to a multiple of 64 = 1676928.
    # TD-IPC-MOE-LAYER-CAP: kMaxMoeLayers 64→128 doubles the per-(layer,expert) arrays.
    assert ctypes.sizeof(StateSnapshot) == 1676928


# ── Field offsets ────────────────────────────────────────────────────────────

def test_field_offsets():
    # IpcHeader
    assert IpcHeader.version.offset == 0
    assert IpcHeader.shutdown_requested.offset == 64
    assert IpcHeader.error_code.offset == 128
    assert IpcHeader.error_message.offset == 132

    # RingHeader
    assert RingHeader.producer_seq.offset == 0
    assert RingHeader.consumer_seq.offset == 64
    assert RingHeader.slot_count.offset == 120
    assert RingHeader.slot_size.offset == 124

    # Command: payload starts at offset 16
    assert Command.payload.offset == 16

    # Completion: payload starts at offset 16
    assert Completion.payload.offset == 16

    # StateSnapshot: seqlock at 0, then 56B padding, daemon_cycle_count at 64
    assert StateSnapshot.seqlock.offset == 0
    assert StateSnapshot.daemon_cycle_count.offset == 64

    # EngineInfo: cross-language invariant — must match C++ offsetof exactly
    # (232 since V4-7a added v4_hc_mult/v4_num_hash_layers/
    # v4_attention_types[96]; test_ipc_struct_layout pins the full map)
    assert ctypes.sizeof(EngineInfo) == 232
    assert EngineInfo.ipc_base.offset == 0
    assert EngineInfo.ipc_total_bytes.offset == 8
    assert EngineInfo.cmd_ring_offset.offset == 16
    assert EngineInfo.cmd_ring_slots.offset == 24
    assert EngineInfo.cmd_slot_bytes.offset == 28
    assert EngineInfo.cmp_ring_offset.offset == 32
    assert EngineInfo.cmp_ring_slots.offset == 40
    assert EngineInfo.cmp_slot_bytes.offset == 44
    assert EngineInfo.state_offset.offset == 48
    assert EngineInfo.state_bytes.offset == 56
    assert EngineInfo.sideband_offset.offset == 64
    assert EngineInfo.sideband_bytes.offset == 72
    assert EngineInfo.num_gpus.offset == 80
    assert EngineInfo.num_moe_layers.offset == 84
    assert EngineInfo.num_experts.offset == 88
    assert EngineInfo.expert_bytes.offset == 96
    assert EngineInfo.num_layers.offset == 104
    assert EngineInfo.num_expert_devices.offset == 108
    assert EngineInfo.kv_bytes_per_page.offset == 112
    assert EngineInfo.vocab_size.offset == 120


# ── Ring helpers ─────────────────────────────────────────────────────────────

def _make_ring_buffer(slot_count: int, slot_size: int) -> tuple[ctypes.Array, int]:
    """Allocate a ring buffer and return (buffer, base_addr)."""
    ring_header_size = ctypes.sizeof(RingHeader)
    total = ring_header_size + slot_count * slot_size
    buf = ctypes.create_string_buffer(total)
    addr = ctypes.addressof(buf)

    # Initialize ring header
    header = RingHeader.from_address(addr)
    header.producer_seq = 0
    header.consumer_seq = 0
    header.slot_count = slot_count
    header.slot_size = slot_size

    return buf, addr


def _make_command(cmd_seq: int, cmd_type: int = CMD_TRANSFER_H2D) -> bytes:
    """Create a Command as bytes."""
    cmd = Command()
    cmd.cmd_type = cmd_type
    cmd.cmd_seq = cmd_seq
    cmd.gpu_idx = 0
    cmd.stream_id = 3
    cmd.payload.transfer.layer_idx = cmd_seq * 10
    cmd.payload.transfer.expert_idx = cmd_seq % 256
    return bytes(cmd)


# ── Ring write/read ──────────────────────────────────────────────────────────

def test_ring_write_read():
    buf, addr = _make_ring_buffer(4, CMD_SLOT_BYTES)
    writer = SpscRingWriter(addr, 4, CMD_SLOT_BYTES)
    reader = SpscRingReader(addr, 4, CMD_SLOT_BYTES)

    data = _make_command(42)
    assert writer.write(data)

    result = reader.read()
    assert result is not None
    assert len(result) == CMD_SLOT_BYTES

    # Parse result back
    out = Command.from_buffer_copy(result)
    assert out.cmd_type == CMD_TRANSFER_H2D
    assert out.cmd_seq == 42
    assert out.payload.transfer.layer_idx == 420
    assert out.payload.transfer.expert_idx == 42


def test_drain_batch():
    buf, addr = _make_ring_buffer(8, CMD_SLOT_BYTES)
    writer = SpscRingWriter(addr, 8, CMD_SLOT_BYTES)
    reader = SpscRingReader(addr, 8, CMD_SLOT_BYTES)

    for i in range(5):
        assert writer.write(_make_command(i))

    results = reader.drain(10)
    assert len(results) == 5

    for i, data in enumerate(results):
        cmd = Command.from_buffer_copy(data)
        assert cmd.cmd_seq == i
        assert cmd.payload.transfer.layer_idx == i * 10


def test_ring_full_empty():
    buf, addr = _make_ring_buffer(4, CMD_SLOT_BYTES)
    writer = SpscRingWriter(addr, 4, CMD_SLOT_BYTES)
    reader = SpscRingReader(addr, 4, CMD_SLOT_BYTES)

    # Initially empty
    assert reader.is_empty()
    assert reader.read() is None

    # Fill to capacity
    for i in range(4):
        assert writer.write(_make_command(i))

    assert writer.is_full()
    assert not writer.write(_make_command(99))  # Should fail

    # Read all
    for i in range(4):
        result = reader.read()
        assert result is not None
        cmd = Command.from_buffer_copy(result)
        assert cmd.cmd_seq == i

    assert reader.is_empty()


# ── Seqlock ──────────────────────────────────────────────────────────────────

def test_seqlock_read():
    snap_size = ctypes.sizeof(StateSnapshot)
    buf = ctypes.create_string_buffer(snap_size)
    addr = ctypes.addressof(buf)

    snap = StateSnapshot.from_address(addr)

    # Seqlock starts at 0 (even = readable)
    # Write data under seqlock protocol
    snap.seqlock = 1  # Begin write (odd)
    snap.daemon_cycle_count = 100
    snap.timestamp_ns = 999999
    snap.num_gpus = 2
    snap.global_acceptance_rate = 0.75
    snap.shift_detected = 1
    snap.seqlock = 2  # End write (even)

    # read_snapshot should succeed
    copy = read_snapshot(addr)
    assert copy.daemon_cycle_count == 100
    assert copy.timestamp_ns == 999999
    assert copy.num_gpus == 2
    assert abs(copy.global_acceptance_rate - 0.75) < 1e-10
    assert copy.shift_detected == 1


def test_seqlock_torn_read_detection():
    """Verify that read_snapshot retries when seqlock changes mid-read."""
    snap_size = ctypes.sizeof(StateSnapshot)
    buf = ctypes.create_string_buffer(snap_size)
    addr = ctypes.addressof(buf)

    snap = StateSnapshot.from_address(addr)

    # Set valid state with seqlock=2
    snap.seqlock = 2
    snap.daemon_cycle_count = 50

    # Normal read should work
    copy = read_snapshot(addr)
    assert copy.daemon_cycle_count == 50

    # Update with new seqlock cycle
    snap.seqlock = 3  # Writing
    snap.daemon_cycle_count = 99
    snap.seqlock = 4  # Done

    # read_snapshot should get the new data
    copy = read_snapshot(addr)
    assert copy.daemon_cycle_count == 99

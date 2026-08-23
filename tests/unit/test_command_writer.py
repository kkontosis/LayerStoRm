"""Unit tests for command_writer.py.

Pure Python — no engine required.  Builds commands via CommandWriter,
verifies cmd_type, auto-incrementing cmd_seq, and payload fields.
"""

import ctypes

import pytest

from orchestrator.shm_protocol import (
    CMD_ATTENTION_DECODE,
    CMD_ATTENTION_PREFILL,
    CMD_CACHE_DEMOTE,
    CMD_CACHE_EVICT,
    CMD_CACHE_PROMOTE,
    CMD_CACHE_RESERVE,
    CMD_COMPUTE_AFFINITY_HINTS,
    CMD_DCP_CORRECTION,
    B_CMD_NVME_BATCH_READ,
    D_B_CMD_EVICT_BATCH,
    D_B_CMD_PREFETCH_BATCH,
    D_B_CMD_RUN_ATTENTION,
    D_B_CMD_RUN_MOE,
    D_CMD_EVICT_TO_HOST,
    D_CMD_SLOW_EVICT_TO_HOST,
    D_CMD_PREFETCH_EXPERT,
    D_CMD_RUN_ADAPTER_FORWARD,
    D_CMD_RUN_PREFETCH_PROBE,
    D_CMD_STAGE_EXPERT,
    E_CMD_SEQ_CREATE,
    E_CMD_SEQ_FREE,
    CMD_DYNAMIC_FP8_QUANT,
    CMD_EMBEDDING_LOOKUP,
    CMD_EXPERT_FFN,
    CMD_GATING,
    CMD_GRAPH_REPLAY,
    CMD_MOE_PERMUTE,
    CMD_MOE_UNPERMUTE,
    CMD_NCCL_ALLREDUCE,
    CMD_NOOP,
    CMD_NUMA_MIGRATE,
    CMD_OUTPUT_HEAD,
    CMD_SAMPLE_TOKENS,
    CMD_CANCEL_TRANSFER,
    CMD_NVME_EVICT_HOST,
    CMD_NVME_READ,
    CMD_NVME_WRITE,
    CMD_SEQ_CREATE,
    CMD_SEQ_FORK,
    CMD_SEQ_FREE,
    CMD_PRESCOPE_GATING,
    CMD_PROBE_MLP,
    CMD_RECORD_EVENT,
    CMD_RMSNORM,
    CMD_SHUTDOWN,
    CMD_SLOT_BYTES,
    CMD_STREAM_WAIT_EVENT,
    CMD_SWIGLU,
    CMD_TRANSFER_D2H,
    CMD_TRANSFER_H2D,
    CMP_SLOT_BYTES,
    CMP_TRANSFER_DONE,
    Command,
    Completion,
    RingHeader,
    STREAM_ATTENTION,
    STREAM_D2H_TRANSFER,
    STREAM_EXPERT_FFN,
    STREAM_GATING,
    STREAM_H2D_TRANSFER,
    STREAM_PREFETCH_COMPUTE,
    SUB_ALL,
    SUB_GATE,
    ZONE_STABLE,
    ZONE_STREAMING,
)
from orchestrator.spsc_ring import SpscRingReader, SpscRingWriter
from orchestrator.command_writer import (
    CommandWriter,
    CompletionReader,
    parse_completion,
    write_sideband_batch_descriptors,
    write_sideband_token_ids,
)
from orchestrator.shm_protocol import (
    BatchDescriptorEntry,
    E_CMD_FETCH_AND_RUN_MOE,
    ExpertPrefetchEntry,
    MAX_BATCH_DESCRIPTORS,
    MAX_EXPERT_PREFETCH,
    MAX_SIDEBAND_TOKEN_IDS,
    RoutingExportHeader,
    SIDEBAND_BATCH_DESCRIPTOR_OFF,
    SIDEBAND_EXPERT_PREFETCH_OFF,
    SIDEBAND_ROUTING_EXPORT_OFF,
    SIDEBAND_ROUTING_EXPORT_INDICES_OFF,
    SIDEBAND_TOKEN_IDS_OFF,
    SIDEBAND_TOTAL_SIZE,
)
from orchestrator.command_writer import (
    read_sideband_routing_export,
    write_sideband_expert_prefetch,
)

_RING_HEADER_SIZE = ctypes.sizeof(RingHeader)


def _make_ring(slot_count: int, slot_size: int):
    """Allocate a ring buffer.  Returns (buf, addr)."""
    total = _RING_HEADER_SIZE + slot_count * slot_size
    buf = ctypes.create_string_buffer(total)
    addr = ctypes.addressof(buf)
    header = RingHeader.from_address(addr)
    header.producer_seq = 0
    header.consumer_seq = 0
    header.slot_count = slot_count
    header.slot_size = slot_size
    return buf, addr


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence numbering
# ═══════════════════════════════════════════════════════════════════════════════

class TestSequenceNumbers:
    def test_auto_increment(self):
        w = CommandWriter()
        assert w.next_seq == 0
        c0 = w.noop()
        assert c0.cmd_seq == 0
        c1 = w.noop()
        assert c1.cmd_seq == 1
        c2 = w.noop()
        assert c2.cmd_seq == 2
        assert w.next_seq == 3

    def test_custom_initial_seq(self):
        w = CommandWriter(initial_seq=100)
        c = w.noop()
        assert c.cmd_seq == 100
        assert w.next_seq == 101


# ═══════════════════════════════════════════════════════════════════════════════
# Transfer commands
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransferCommands:
    def test_transfer_h2d(self):
        w = CommandWriter()
        cmd = w.transfer_h2d(gpu=1, layer=5, expert=42, zone=ZONE_STABLE,
                              sub_component=SUB_GATE, nbytes=1048576)
        assert cmd.cmd_type == CMD_TRANSFER_H2D
        assert cmd.gpu_idx == 1
        assert cmd.stream_id == STREAM_H2D_TRANSFER
        assert cmd.payload.transfer.layer_idx == 5
        assert cmd.payload.transfer.expert_idx == 42
        assert cmd.payload.transfer.sub_component == SUB_GATE
        assert cmd.payload.transfer.zone == ZONE_STABLE
        assert cmd.payload.transfer.bytes == 1048576

    def test_transfer_d2h(self):
        w = CommandWriter()
        cmd = w.transfer_d2h(gpu=0, layer=3, expert=7, zone=ZONE_STREAMING)
        assert cmd.cmd_type == CMD_TRANSFER_D2H
        assert cmd.stream_id == STREAM_D2H_TRANSFER
        assert cmd.payload.transfer.sub_component == SUB_ALL  # Default

    def test_transfer_defaults(self):
        w = CommandWriter()
        cmd = w.transfer_h2d(gpu=0, layer=0, expert=0, zone=0)
        assert cmd.payload.transfer.sub_component == SUB_ALL
        assert cmd.payload.transfer.bytes == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Cache commands
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheCommands:
    def test_cache_reserve(self):
        w = CommandWriter()
        cmd = w.cache_reserve(gpu=0, layer=10, expert=5, zone=ZONE_STABLE,
                               is_duplicate=True)
        assert cmd.cmd_type == CMD_CACHE_RESERVE
        assert cmd.payload.cache_reserve.layer_idx == 10
        assert cmd.payload.cache_reserve.expert_idx == 5
        assert cmd.payload.cache_reserve.zone == ZONE_STABLE
        assert cmd.payload.cache_reserve.is_duplicate == 1

    def test_cache_evict(self):
        w = CommandWriter()
        cmd = w.cache_evict(gpu=2, layer=3, expert=7)
        assert cmd.cmd_type == CMD_CACHE_EVICT
        assert cmd.gpu_idx == 2
        assert cmd.payload.cache_op.layer_idx == 3
        assert cmd.payload.cache_op.expert_idx == 7

    def test_cache_promote(self):
        w = CommandWriter()
        cmd = w.cache_promote(gpu=0, layer=1, expert=2)
        assert cmd.cmd_type == CMD_CACHE_PROMOTE

    def test_cache_demote(self):
        w = CommandWriter()
        cmd = w.cache_demote(gpu=1, layer=5, expert=10)
        assert cmd.cmd_type == CMD_CACHE_DEMOTE


# ═══════════════════════════════════════════════════════════════════════════════
# Compute commands
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeCommands:
    def test_attention_decode(self):
        w = CommandWriter()
        cmd = w.attention_decode(gpu=0, layer=5, batch_size=8,
                                  hidden_state_buf=1, kv_cache_buf=2,
                                  use_graph=True, is_sparse=True)
        assert cmd.cmd_type == CMD_ATTENTION_DECODE
        assert cmd.stream_id == STREAM_ATTENTION
        assert cmd.payload.attention.layer_idx == 5
        assert cmd.payload.attention.batch_size == 8
        assert cmd.payload.attention.use_graph == 1
        assert cmd.payload.attention.is_sparse == 1
        assert cmd.payload.attention.hidden_state_buf_id == 1
        assert cmd.payload.attention.kv_cache_buf_id == 2

    def test_attention_decode_seqlens_block_table(self):
        w = CommandWriter()
        cmd = w.attention_decode(gpu=0, layer=5, batch_size=8,
                                  hidden_state_buf=1, kv_cache_buf=2,
                                  seqlens_buf=5, block_table_buf=6)
        assert cmd.payload.attention.seqlens_buf_id == 5
        assert cmd.payload.attention.block_table_buf_id == 6

    def test_attention_prefill(self):
        w = CommandWriter()
        cmd = w.attention_prefill(gpu=1, layer=0, batch_size=1,
                                   hidden_state_buf=10, kv_cache_buf=20)
        assert cmd.cmd_type == CMD_ATTENTION_PREFILL
        assert cmd.payload.attention.use_graph == 0
        assert cmd.payload.attention.is_sparse == 0

    def test_attention_prefill_seqlens_block_table(self):
        w = CommandWriter()
        cmd = w.attention_prefill(gpu=0, layer=3, batch_size=4,
                                   hidden_state_buf=1, kv_cache_buf=2,
                                   seqlens_buf=5, block_table_buf=6)
        assert cmd.payload.attention.seqlens_buf_id == 5
        assert cmd.payload.attention.block_table_buf_id == 6

    def test_gating(self):
        w = CommandWriter()
        cmd = w.gating(gpu=0, layer=3, num_tokens=16,
                        input_buf=1, output_weights_buf=2,
                        output_indices_buf=3)
        assert cmd.cmd_type == CMD_GATING
        assert cmd.stream_id == STREAM_GATING
        assert cmd.payload.gating.num_tokens == 16

    def test_expert_ffn(self):
        w = CommandWriter()
        cmd = w.expert_ffn(gpu=0, layer=5, num_experts=8,
                            total_tokens=32, permuted_input_buf=1,
                            output_buf=2, expert_offsets_buf=3)
        assert cmd.cmd_type == CMD_EXPERT_FFN
        assert cmd.stream_id == STREAM_EXPERT_FFN
        assert cmd.payload.expert_ffn.num_experts == 8

    def test_expert_ffn_new_fields(self):
        w = CommandWriter()
        cmd = w.expert_ffn(gpu=0, layer=5, num_experts=8,
                            total_tokens=32, permuted_input_buf=1,
                            output_buf=2, expert_offsets_buf=3,
                            hidden_dim=128, weights_buf=10,
                            workspace_buf=11, quant_mode=1)
        p = cmd.payload.expert_ffn
        assert p.hidden_dim == 128
        assert p.weights_buf_id == 10
        assert p.workspace_buf_id == 11
        assert p.quant_mode == 1

    def test_embedding_lookup(self):
        w = CommandWriter()
        cmd = w.embedding_lookup(gpu=0, num_tokens=4, output_buf=2)
        assert cmd.cmd_type == CMD_EMBEDDING_LOOKUP
        assert cmd.payload.embedding_lookup.num_tokens == 4
        assert cmd.payload.embedding_lookup.output_buf_id == 2

    def test_output_head(self):
        w = CommandWriter()
        cmd = w.output_head(gpu=0, num_tokens=1, input_buf=1,
                             output_buf=2, readback=True)
        assert cmd.cmd_type == CMD_OUTPUT_HEAD
        assert cmd.payload.output_head.readback_to_host == 1

    def test_output_head_no_readback(self):
        w = CommandWriter()
        cmd = w.output_head(gpu=0, num_tokens=1, input_buf=1, output_buf=2)
        assert cmd.payload.output_head.readback_to_host == 0
        assert cmd.payload.output_head.compute_confidence == 0

    def test_output_head_compute_confidence(self):
        w = CommandWriter()
        cmd = w.output_head(gpu=0, num_tokens=1, input_buf=1,
                             output_buf=2, compute_confidence=True)
        assert cmd.payload.output_head.compute_confidence == 1

    def test_sample_tokens(self):
        w = CommandWriter()
        cmd = w.sample_tokens(gpu=0, num_tokens=4, logits_buf=10,
                              vocab_size=129280, temperature=0.8,
                              top_p=0.9, top_k=50, random_seed=42)
        assert cmd.cmd_type == CMD_SAMPLE_TOKENS
        p = cmd.payload.sample_tokens
        assert p.num_tokens == 4
        assert p.logits_buf_id == 10
        assert p.vocab_size == 129280
        assert p.top_k == 50
        assert abs(p.temperature - 0.8) < 1e-6
        assert abs(p.top_p - 0.9) < 1e-6
        assert p.random_seed == 42

    def test_sample_tokens_defaults(self):
        w = CommandWriter()
        cmd = w.sample_tokens(gpu=0, num_tokens=1, logits_buf=1,
                              vocab_size=1024)
        p = cmd.payload.sample_tokens
        assert abs(p.temperature - 1.0) < 1e-6
        assert abs(p.top_p - 1.0) < 1e-6
        assert p.top_k == 0
        assert p.random_seed == 0

    def test_run_attention(self):
        w = CommandWriter()
        cmd = w.run_attention(gpu=1, layer_idx=5, num_seqs=8,
                              is_prefill=1, use_graph=0, is_draft=1,
                              chunk_start=512, chunk_len=256)
        assert cmd.cmd_type == D_B_CMD_RUN_ATTENTION
        assert cmd.gpu_idx == 1
        p = cmd.payload.run_attention
        assert p.layer_idx == 5
        assert p.num_seqs == 8
        assert p.is_prefill == 1
        assert p.use_graph == 0
        assert p.is_draft == 1
        assert p.chunk_start == 512
        assert p.chunk_len == 256

    def test_run_moe(self):
        w = CommandWriter()
        cmd = w.run_moe(gpu=0, layer_idx=42, num_seqs=32,
                        moe_mode=2,
                        apply_residual_correction=1,
                        store_gating_output=1)
        assert cmd.cmd_type == D_B_CMD_RUN_MOE
        p = cmd.payload.run_moe
        assert p.layer_idx == 42
        assert p.num_seqs == 32
        assert p.moe_mode == 2
        assert p.apply_residual_correction == 1
        assert p.store_gating_output == 1

    def test_rmsnorm(self):
        w = CommandWriter()
        cmd = w.rmsnorm(gpu=0, num_tokens=8, input_buf=1,
                         output_buf=2, weight_buf=3, eps=1e-5)
        assert cmd.cmd_type == CMD_RMSNORM
        assert abs(cmd.payload.rmsnorm.eps - 1e-5) < 1e-10

    def test_rmsnorm_hidden_size(self):
        w = CommandWriter()
        cmd = w.rmsnorm(gpu=0, num_tokens=8, input_buf=1,
                         output_buf=2, weight_buf=3, hidden_size=256)
        assert cmd.payload.rmsnorm.hidden_size == 256

    def test_swiglu(self):
        w = CommandWriter()
        cmd = w.swiglu(gpu=0, num_tokens=4, hidden_dim=2048,
                        input_buf=1, output_buf=2)
        assert cmd.cmd_type == CMD_SWIGLU
        assert cmd.payload.swiglu.hidden_dim == 2048

    def test_moe_permute(self):
        w = CommandWriter()
        cmd = w.moe_permute(gpu=0, num_tokens=16, num_experts=8,
                             input_buf=1, output_buf=2,
                             indices_buf=3, offsets_buf=4)
        assert cmd.cmd_type == CMD_MOE_PERMUTE
        assert cmd.payload.moe_permute.offsets_buf_id == 4

    def test_moe_permute_new_fields(self):
        w = CommandWriter()
        cmd = w.moe_permute(gpu=0, num_tokens=16, num_experts=8,
                             input_buf=1, output_buf=2,
                             indices_buf=3, offsets_buf=4,
                             topk=6, hidden_dim=256, workspace_buf=20)
        p = cmd.payload.moe_permute
        assert p.topk == 6
        assert p.hidden_dim == 256
        assert p.workspace_buf_id == 20

    def test_moe_unpermute(self):
        w = CommandWriter()
        cmd = w.moe_unpermute(gpu=0, num_tokens=16, num_experts=8,
                               input_buf=1, output_buf=2, indices_buf=3)
        assert cmd.cmd_type == CMD_MOE_UNPERMUTE

    def test_moe_unpermute_new_fields(self):
        w = CommandWriter()
        cmd = w.moe_unpermute(gpu=0, num_tokens=16, num_experts=8,
                               input_buf=1, output_buf=2, indices_buf=3,
                               topk=6, hidden_dim=256, weights_buf=30)
        p = cmd.payload.moe_unpermute
        assert p.topk == 6
        assert p.hidden_dim == 256
        assert p.weights_buf_id == 30

    def test_dcp_correction(self):
        w = CommandWriter()
        cmd = w.dcp_correction(gpu=0, batch_size=4,
                                input_buf=1, output_buf=2, lse_buf=3)
        assert cmd.cmd_type == CMD_DCP_CORRECTION
        assert cmd.payload.dcp_correction.lse_buf_id == 3

    def test_nccl_allreduce(self):
        w = CommandWriter()
        cmd = w.nccl_allreduce(gpu=0, count=1024, buf_id=5, dtype=2)
        assert cmd.cmd_type == CMD_NCCL_ALLREDUCE
        assert cmd.payload.nccl_allreduce.dtype == 2

    def test_dynamic_fp8_quant(self):
        w = CommandWriter()
        cmd = w.dynamic_fp8_quant(gpu=0, num_tokens=8, hidden_dim=4096,
                                   input_buf=1, output_buf=2, scales_buf=3)
        assert cmd.cmd_type == CMD_DYNAMIC_FP8_QUANT
        assert cmd.payload.dynamic_fp8_quant.hidden_dim == 4096


# ═══════════════════════════════════════════════════════════════════════════════
# Prefetch commands
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrefetchCommands:
    def test_prescope_gating(self):
        w = CommandWriter()
        cmd = w.prescope_gating(gpu=0, target_layer=10, num_tokens=8,
                                 hidden_state_buf=1, output_buf=2)
        assert cmd.cmd_type == CMD_PRESCOPE_GATING
        assert cmd.stream_id == STREAM_PREFETCH_COMPUTE
        assert cmd.payload.prescope.target_layer_idx == 10

    def test_probe_mlp(self):
        w = CommandWriter()
        cmd = w.probe_mlp(gpu=0, probe_point=2, num_tokens=4,
                           hidden_state_buf=1, output_buf=2)
        assert cmd.cmd_type == CMD_PROBE_MLP
        assert cmd.payload.probe.probe_point_idx == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Graph command
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphCommand:
    def test_graph_replay(self):
        w = CommandWriter()
        cmd = w.graph_replay(gpu=0, graph_type=1, batch_size=16)
        assert cmd.cmd_type == CMD_GRAPH_REPLAY
        assert cmd.payload.graph.graph_type == 1
        assert cmd.payload.graph.batch_size == 16


# ═══════════════════════════════════════════════════════════════════════════════
# Sync commands
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyncCommands:
    def test_record_event(self):
        w = CommandWriter()
        cmd = w.record_event(gpu=0, stream=STREAM_ATTENTION, event_id=42)
        assert cmd.cmd_type == CMD_RECORD_EVENT
        assert cmd.payload.event.event_id == 42

    def test_stream_wait_event(self):
        w = CommandWriter()
        cmd = w.stream_wait_event(gpu=1, stream=STREAM_EXPERT_FFN,
                                   event_id=99)
        assert cmd.cmd_type == CMD_STREAM_WAIT_EVENT
        assert cmd.gpu_idx == 1
        assert cmd.stream_id == STREAM_EXPERT_FFN
        assert cmd.payload.event.event_id == 99


# ═══════════════════════════════════════════════════════════════════════════════
# Placement commands
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlacementCommands:
    def test_compute_affinity_hints(self):
        w = CommandWriter()
        caps = [100, 200, 50, 150]
        cmd = w.compute_affinity_hints(num_gpus=4,
                                        gpu_capacity_slots=caps)
        assert cmd.cmd_type == CMD_COMPUTE_AFFINITY_HINTS
        assert cmd.payload.affinity_hints.num_gpus == 4
        for i, cap in enumerate(caps):
            assert cmd.payload.affinity_hints.gpu_capacity_slots[i] == cap

    def test_numa_migrate(self):
        w = CommandWriter()
        cmd = w.numa_migrate(gpu=0, layer=5, expert=42,
                              target_numa_node=1)
        assert cmd.cmd_type == CMD_NUMA_MIGRATE
        assert cmd.payload.numa_migrate.target_numa_node == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence lifecycle commands
# ═══════════════════════════════════════════════════════════════════════════════

class TestSequenceCommands:
    def test_seq_create(self):
        w = CommandWriter()
        cmd = w.seq_create(gpu=0, seq_id=1000, prompt_len=128)
        assert cmd.cmd_type == CMD_SEQ_CREATE
        assert cmd.gpu_idx == 0
        assert cmd.stream_id == 0
        assert cmd.payload.seq_create.seq_id == 1000
        assert cmd.payload.seq_create.prompt_len == 128
        assert cmd.payload.seq_create.pool == 0

    def test_seq_create_speculation_pool(self):
        w = CommandWriter()
        cmd = w.seq_create(gpu=1, seq_id=2000, prompt_len=64, pool=1)
        assert cmd.payload.seq_create.pool == 1
        assert cmd.gpu_idx == 1

    def test_seq_free(self):
        w = CommandWriter()
        cmd = w.seq_free(gpu=1, seq_id=3000)
        assert cmd.cmd_type == CMD_SEQ_FREE
        assert cmd.gpu_idx == 1
        assert cmd.payload.seq_free.seq_id == 3000

    def test_seq_fork(self):
        w = CommandWriter()
        cmd = w.seq_fork(gpu=0, src_seq_id=4000, dst_seq_id=4001)
        assert cmd.cmd_type == CMD_SEQ_FORK
        assert cmd.payload.seq_fork.src_seq_id == 4000
        assert cmd.payload.seq_fork.dst_seq_id == 4001


# ═══════════════════════════════════════════════════════════════════════════════
# NVMe tier + cancel commands
# ═══════════════════════════════════════════════════════════════════════════════

class TestNvmeCommands:
    def test_nvme_read(self):
        w = CommandWriter()
        cmd = w.nvme_read(gpu=1, layer=5, expert=42, gpu_hint=0)
        assert cmd.cmd_type == CMD_NVME_READ
        assert cmd.stream_id == 0
        assert cmd.payload.nvme_read.layer_idx == 5
        assert cmd.payload.nvme_read.expert_idx == 42
        assert cmd.payload.nvme_read.gpu_hint == 0

    def test_nvme_write(self):
        w = CommandWriter()
        cmd = w.nvme_write(gpu=0, layer=3, expert=10)
        assert cmd.cmd_type == CMD_NVME_WRITE
        assert cmd.stream_id == 0
        assert cmd.payload.nvme_write.layer_idx == 3
        assert cmd.payload.nvme_write.expert_idx == 10

    def test_nvme_evict_host(self):
        w = CommandWriter()
        cmd = w.nvme_evict_host(gpu=0, layer=7, expert=99)
        assert cmd.cmd_type == CMD_NVME_EVICT_HOST
        assert cmd.stream_id == 0
        assert cmd.payload.nvme_evict_host.layer_idx == 7
        assert cmd.payload.nvme_evict_host.expert_idx == 99

    def test_cancel_transfer(self):
        w = CommandWriter()
        cmd = w.cancel_transfer(gpu=0, target_cmd_seq=42)
        assert cmd.cmd_type == CMD_CANCEL_TRANSFER
        assert cmd.stream_id == 0
        assert cmd.payload.cancel_transfer.target_cmd_seq == 42


# ═══════════════════════════════════════════════════════════════════════════════
# Lifecycle commands
# ═══════════════════════════════════════════════════════════════════════════════

class TestLifecycleCommands:
    def test_shutdown(self):
        w = CommandWriter()
        cmd = w.shutdown()
        assert cmd.cmd_type == CMD_SHUTDOWN

    def test_noop(self):
        w = CommandWriter()
        cmd = w.noop(gpu=2)
        assert cmd.cmd_type == CMD_NOOP
        assert cmd.gpu_idx == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Completion parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompletionReader:
    def test_parse_completion(self):
        cmp = Completion()
        cmp.cmp_type = CMP_TRANSFER_DONE
        cmp.cmd_seq = 7
        cmp.gpu_idx = 1
        cmp.status = 0
        cmp.payload.transfer.layer_idx = 5
        cmp.payload.transfer.expert_idx = 42

        parsed = parse_completion(bytes(cmp))
        assert parsed.cmp_type == CMP_TRANSFER_DONE
        assert parsed.cmd_seq == 7
        assert parsed.payload.transfer.layer_idx == 5

    def test_completion_reader_read(self):
        buf, addr = _make_ring(8, CMP_SLOT_BYTES)
        writer = SpscRingWriter(addr, 8, CMP_SLOT_BYTES)
        ring_reader = SpscRingReader(addr, 8, CMP_SLOT_BYTES)
        creader = CompletionReader(ring_reader)

        # Empty ring
        assert creader.read() is None
        assert creader.is_empty()

        # Write a completion
        cmp = Completion()
        cmp.cmp_type = CMP_TRANSFER_DONE
        cmp.cmd_seq = 10
        cmp.status = 0
        writer.write(bytes(cmp))

        result = creader.read()
        assert result is not None
        assert result.cmp_type == CMP_TRANSFER_DONE
        assert result.cmd_seq == 10

    def test_completion_reader_drain(self):
        buf, addr = _make_ring(8, CMP_SLOT_BYTES)
        writer = SpscRingWriter(addr, 8, CMP_SLOT_BYTES)
        ring_reader = SpscRingReader(addr, 8, CMP_SLOT_BYTES)
        creader = CompletionReader(ring_reader)

        for i in range(3):
            cmp = Completion()
            cmp.cmd_seq = i
            writer.write(bytes(cmp))

        results = creader.drain(10)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.cmd_seq == i


# ═══════════════════════════════════════════════════════════════════════════════
# Round-trip: build → write → read → parse
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoundTrip:
    def test_command_round_trip(self):
        """Build a command, write to ring, read back, verify all fields."""
        buf, addr = _make_ring(8, CMD_SLOT_BYTES)
        ring_writer = SpscRingWriter(addr, 8, CMD_SLOT_BYTES)
        ring_reader = SpscRingReader(addr, 8, CMD_SLOT_BYTES)

        cw = CommandWriter()
        cmd = cw.transfer_h2d(gpu=1, layer=10, expert=42,
                               zone=ZONE_STABLE, sub_component=SUB_ALL,
                               nbytes=2048)
        assert ring_writer.write_struct(cmd)

        data = ring_reader.read()
        assert data is not None
        parsed = Command.from_buffer_copy(data)
        assert parsed.cmd_type == CMD_TRANSFER_H2D
        assert parsed.cmd_seq == 0
        assert parsed.gpu_idx == 1
        assert parsed.stream_id == STREAM_H2D_TRANSFER
        assert parsed.payload.transfer.layer_idx == 10
        assert parsed.payload.transfer.expert_idx == 42
        assert parsed.payload.transfer.zone == ZONE_STABLE
        assert parsed.payload.transfer.sub_component == SUB_ALL
        assert parsed.payload.transfer.bytes == 2048

    def test_multi_command_round_trip(self):
        """Write several commands, drain, verify sequence."""
        buf, addr = _make_ring(16, CMD_SLOT_BYTES)
        ring_writer = SpscRingWriter(addr, 16, CMD_SLOT_BYTES)
        ring_reader = SpscRingReader(addr, 16, CMD_SLOT_BYTES)

        cw = CommandWriter()
        for i in range(5):
            cmd = cw.noop(gpu=i % 4)
            assert ring_writer.write_struct(cmd)

        results = ring_reader.drain(10)
        assert len(results) == 5
        for i, data in enumerate(results):
            parsed = Command.from_buffer_copy(data)
            assert parsed.cmd_seq == i
            assert parsed.cmd_type == CMD_NOOP


# ── Sideband helpers ────────────────────────────────────────────────────────


class TestWriteSidebandTokenIds:
    def test_write_and_read_back(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        ids = [10, 20, 30, 42]
        write_sideband_token_ids(base, ids)
        arr = (ctypes.c_uint32 * 4).from_address(base + SIDEBAND_TOKEN_IDS_OFF)
        assert list(arr) == [10, 20, 30, 42]

    def test_too_many_tokens_raises(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        with pytest.raises(ValueError, match="Too many token IDs"):
            write_sideband_token_ids(base, list(range(MAX_SIDEBAND_TOKEN_IDS + 1)))

    def test_empty_is_ok(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        write_sideband_token_ids(base, [])  # Should not raise


class TestWriteSidebandBatchDescriptors:
    def test_single_entry(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        write_sideband_batch_descriptors(base, [(1000, 15)])
        arr = (BatchDescriptorEntry * 1).from_address(
            base + SIDEBAND_BATCH_DESCRIPTOR_OFF)
        assert arr[0].seq_id == 1000
        assert arr[0].token_pos == 15
        assert arr[0]._pad == 0

    def test_multiple_entries(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        entries = [(2000, 10), (2001, 50), (2002, 99)]
        write_sideband_batch_descriptors(base, entries)
        arr = (BatchDescriptorEntry * 3).from_address(
            base + SIDEBAND_BATCH_DESCRIPTOR_OFF)
        for i, (sid, tpos) in enumerate(entries):
            assert arr[i].seq_id == sid
            assert arr[i].token_pos == tpos

    def test_too_many_raises(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        entries = [(i, i) for i in range(MAX_BATCH_DESCRIPTORS + 1)]
        with pytest.raises(ValueError, match="Too many batch descriptors"):
            write_sideband_batch_descriptors(base, entries)

    def test_empty_is_ok(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        write_sideband_batch_descriptors(base, [])  # Should not raise


class TestSidebandRoutingExport:
    """F-4 routing-export reader + expert-prefetch writer (TD-MTP-PY-LOOP-KV:
    the production-seam MoE handoff RUN_ATTENTION[store_gating] ->
    E_CMD_FETCH_AND_RUN_MOE the MTP flow drives)."""

    def _publish(self, base, layer_idx, indices, topk):
        """Simulate the daemon's routed top-K export."""
        hdr = RoutingExportHeader.from_address(
            base + SIDEBAND_ROUTING_EXPORT_OFF)
        hdr.num_tokens = 1
        hdr.topk = topk
        hdr.layer_idx = layer_idx
        arr = (ctypes.c_int32 * topk).from_address(
            base + SIDEBAND_ROUTING_EXPORT_INDICES_OFF)
        for i in range(topk):
            arr[i] = indices[i]

    def test_read_back_published_export(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        self._publish(base, layer_idx=78, indices=[7, 3, 250, 0], topk=4)
        layer, indices = read_sideband_routing_export(base)
        assert layer == 78
        assert indices == [7, 3, 250, 0]

    def test_negative_indices_filtered(self):
        """Unrouted slots (< 0) must be dropped (golden parity)."""
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        self._publish(base, layer_idx=5, indices=[2, -1, 9, -1], topk=4)
        layer, indices = read_sideband_routing_export(base)
        assert layer == 5
        assert indices == [2, 9]

    def test_empty_export(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        layer, indices = read_sideband_routing_export(base)
        assert indices == []

    def test_write_expert_prefetch_entries(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        entries = [(78, 7, 0, 0), (78, 3, 0, 1), (78, 250, 1, 0)]
        write_sideband_expert_prefetch(base, entries)
        arr = (ExpertPrefetchEntry * 3).from_address(
            base + SIDEBAND_EXPERT_PREFETCH_OFF)
        for i, (layer, expert, zone, gpu) in enumerate(entries):
            assert arr[i].layer_idx == layer
            assert arr[i].expert_idx == expert
            assert arr[i].zone == zone
            assert arr[i].gpu_idx == gpu

    def test_too_many_prefetch_entries_raises(self):
        buf = ctypes.create_string_buffer(SIDEBAND_TOTAL_SIZE)
        base = ctypes.addressof(buf)
        entries = [(0, i, 0, 0) for i in range(MAX_EXPERT_PREFETCH + 1)]
        with pytest.raises(ValueError, match="Too many expert prefetch"):
            write_sideband_expert_prefetch(base, entries)


# ── IPC-8e fused command writers ──────────────────────────────────────────


class TestFusedCommandWriters:
    def test_prefetch_batch(self):
        w = CommandWriter()
        cmd = w.prefetch_batch(gpu=0, count=5)
        assert cmd.cmd_type == D_B_CMD_PREFETCH_BATCH
        assert cmd.gpu_idx == 0
        assert cmd.payload.prefetch_batch.count == 5

    def test_evict_batch(self):
        w = CommandWriter()
        cmd = w.evict_batch(gpu=1, count=3)
        assert cmd.cmd_type == D_B_CMD_EVICT_BATCH
        assert cmd.gpu_idx == 1
        assert cmd.payload.evict_batch.count == 3

    def test_nvme_batch_read(self):
        w = CommandWriter()
        cmd = w.nvme_batch_read(gpu=0, count=10)
        assert cmd.cmd_type == B_CMD_NVME_BATCH_READ
        assert cmd.payload.nvme_batch_read.count == 10

    def test_prefetch_expert(self):
        w = CommandWriter()
        cmd = w.prefetch_expert(gpu=0, layer_idx=5, expert_idx=42,
                                zone=1, target_gpu=1)
        assert cmd.cmd_type == D_CMD_PREFETCH_EXPERT
        p = cmd.payload.prefetch_expert
        assert p.layer_idx == 5
        assert p.expert_idx == 42
        assert p.zone == 1
        assert p.gpu_idx == 1

    def test_evict_to_host(self):
        w = CommandWriter()
        cmd = w.evict_to_host(gpu=0, layer_idx=3, expert_idx=7)
        assert cmd.cmd_type == D_CMD_EVICT_TO_HOST
        p = cmd.payload.evict_to_host
        assert p.layer_idx == 3
        assert p.expert_idx == 7
        assert p.gpu_idx == 0

    def test_slow_evict_to_host(self):
        w = CommandWriter()
        cmd = w.slow_evict_to_host(gpu=1, layer_idx=5, expert_idx=42)
        assert cmd.cmd_type == D_CMD_SLOW_EVICT_TO_HOST
        p = cmd.payload.slow_evict_to_host
        assert p.layer_idx == 5
        assert p.expert_idx == 42
        assert p.gpu_idx == 1

    def test_stage_expert(self):
        w = CommandWriter()
        cmd = w.stage_expert(gpu=1, layer_idx=10, expert_idx=200,
                             zone=0, target_gpu=1)
        assert cmd.cmd_type == D_CMD_STAGE_EXPERT
        p = cmd.payload.stage_expert
        assert p.layer_idx == 10
        assert p.expert_idx == 200
        assert p.zone == 0
        assert p.gpu_idx == 1

    def test_stage_expert_priority_delay(self):
        w = CommandWriter()
        cmd = w.stage_expert(gpu=0, layer_idx=5, expert_idx=42,
                             priority=0.8, delay_us=100)
        p = cmd.payload.stage_expert
        assert abs(p.priority - 0.8) < 1e-6
        assert p.delay_us == 100

    def test_run_prefetch_probe(self):
        w = CommandWriter()
        cmd = w.run_prefetch_probe(gpu=0, target_layer=15,
                                   num_tokens=64, probe_points=0x05)
        assert cmd.cmd_type == D_CMD_RUN_PREFETCH_PROBE
        p = cmd.payload.run_prefetch_probe
        assert p.target_layer == 15
        assert p.num_tokens == 64
        assert p.probe_points == 0x05

    def test_run_adapter_forward(self):
        w = CommandWriter()
        cmd = w.run_adapter_forward(gpu=0, num_tokens=8,
                                    adapter_weights_buf_id=99)
        assert cmd.cmd_type == D_CMD_RUN_ADAPTER_FORWARD
        p = cmd.payload.run_adapter_forward
        assert p.num_tokens == 8
        assert p.adapter_weights_buf_id == 99

    def test_e_seq_create(self):
        w = CommandWriter()
        cmd = w.e_seq_create(gpu=0, seq_id=5000, prompt_len=100,
                             pool=1)
        assert cmd.cmd_type == E_CMD_SEQ_CREATE
        p = cmd.payload.seq_create
        assert p.seq_id == 5000
        assert p.prompt_len == 100
        assert p.pool == 1

    def test_e_seq_free(self):
        w = CommandWriter()
        cmd = w.e_seq_free(gpu=0, seq_id=5000)
        assert cmd.cmd_type == E_CMD_SEQ_FREE
        assert cmd.payload.seq_free.seq_id == 5000

    def test_fetch_and_run_moe(self):
        """E_CMD_FETCH_AND_RUN_MOE — the production routed-MoE seam (#90)."""
        w = CommandWriter()
        cmd = w.fetch_and_run_moe(gpu=1, layer_idx=78, num_seqs=1,
                                  expert_count=8, timeout_us=120_000_000,
                                  moe_mode=0)
        assert cmd.cmd_type == E_CMD_FETCH_AND_RUN_MOE
        assert cmd.gpu_idx == 1
        p = cmd.payload.fetch_and_run_moe
        assert p.layer_idx == 78
        assert p.num_seqs == 1
        assert p.expert_count == 8
        assert p.timeout_us == 120_000_000
        assert p.moe_mode == 0
        # F-6 decider fields default to 0 -> fetch every listed entry
        assert p.weight_count == 0
        assert p.min_experts == 0
        assert p.max_new_fetches == 0
        assert p.have_evict_map == 0
        assert p.gating_weight_threshold == 0.0

    def test_run_attention_gating_flags(self):
        """F-1/F-3 fused gate + routing export flags (production MoE seam)."""
        w = CommandWriter()
        cmd = w.run_attention(gpu=0, layer_idx=78, num_seqs=1,
                              emit_gating=1, store_gating=1)
        p = cmd.payload.run_attention
        assert p.emit_gating == 1
        assert p.store_gating == 1
        # Default off (byte-identical to pre-F-3 commands)
        cmd2 = w.run_attention(gpu=0, layer_idx=0, num_seqs=1)
        assert cmd2.payload.run_attention.emit_gating == 0
        assert cmd2.payload.run_attention.store_gating == 0

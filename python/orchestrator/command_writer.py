"""Typed command builders and completion reader for the IPC command ring.

CommandWriter builds Command ctypes structs with auto-incrementing sequence
numbers.  Callers send via SpscRingWriter.write_struct(cmd).

CompletionReader wraps SpscRingReader with typed Completion parsing.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence

from orchestrator.shm_protocol import (
    CMD_ATTENTION_DECODE,
    CMD_ATTENTION_PREFILL,
    CMD_CACHE_DEMOTE,
    CMD_CACHE_EVICT,
    CMD_CACHE_PROMOTE,
    CMD_CACHE_RESERVE,
    CMD_COMPUTE_AFFINITY_HINTS,
    CMD_CONFIG_UPDATE,
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
    D_CMD_MTP_PROJECT,
    D_CMD_RUN_DSPARK_STEP,
    D_CMD_RUN_MTP_STEP,
    D_CMD_RUN_SELF_SPEC_FORWARD,
    D_CMD_RUN_PREFETCH_PROBE,
    D_CMD_STAGE_EXPERT,
    E_CMD_SEQ_CREATE,
    E_CMD_SEQ_FREE,
    E_CMD_FETCH_AND_RUN_MOE,
    E_CMD_FETCH_AND_RUN_MOE_BIG,
    CMD_DYNAMIC_FP8_QUANT,
    CMD_EMBEDDING_LOOKUP,
    CMD_EXPERT_FFN,
    CMD_GATING,
    CMD_GRAPH_REPLAY,
    CMD_MOE_PERMUTE,
    CMD_MOE_UNPERMUTE,
    CMD_NCCL_ALLREDUCE,
    CMD_CANCEL_TRANSFER,
    CMD_NOOP,
    CMD_NUMA_MIGRATE,
    CMD_NVME_EVICT_HOST,
    CMD_NVME_READ,
    CMD_NVME_WRITE,
    CMD_SEQ_CREATE,
    CMD_SEQ_FORK,
    CMD_SEQ_FREE,
    CMD_OUTPUT_HEAD,
    CMD_PRESCOPE_GATING,
    CMD_PROBE_MLP,
    CMD_RECORD_EVENT,
    CMD_RMSNORM,
    CMD_SAMPLE_TOKENS,
    CMD_SHUTDOWN,
    CMD_STREAM_WAIT_EVENT,
    CMD_SWIGLU,
    CMD_TRANSFER_D2H,
    CMD_TRANSFER_H2D,
    Command,
    Completion,
    MAX_GPUS,
    BatchDescriptorEntry,
    ExpertPrefetchEntry,
    RoutingExportHeader,
    MAX_BATCH_DESCRIPTORS,
    MAX_EXPERT_PREFETCH,
    MAX_SIDEBAND_TOKEN_IDS,
    SIDEBAND_BATCH_DESCRIPTOR_OFF,
    SIDEBAND_EXPERT_PREFETCH_OFF,
    SIDEBAND_ROUTING_EXPORT_OFF,
    SIDEBAND_ROUTING_EXPORT_INDICES_OFF,
    SIDEBAND_TOKEN_IDS_OFF,
    STREAM_ATTENTION,
    STREAM_D2H_TRANSFER,
    STREAM_EXPERT_FFN,
    STREAM_GATING,
    STREAM_H2D_TRANSFER,
    STREAM_PREFETCH_COMPUTE,
    SUB_ALL,
)
from orchestrator.spsc_ring import SpscRingReader


def parse_completion(data: bytes) -> Completion:
    """Parse raw completion bytes into a typed Completion struct."""
    return Completion.from_buffer_copy(data)


class CompletionReader:
    """Typed wrapper around SpscRingReader for completions."""

    __slots__ = ("_reader",)

    def __init__(self, ring_reader: SpscRingReader) -> None:
        self._reader = ring_reader

    def read(self) -> Completion | None:
        """Read one completion. Returns None if ring is empty."""
        data = self._reader.read()
        return Completion.from_buffer_copy(data) if data else None

    def drain(self, max_count: int = 0xFFFFFFFF) -> list[Completion]:
        """Batch-read and parse up to max_count completions."""
        return [Completion.from_buffer_copy(d)
                for d in self._reader.drain(max_count)]

    def is_empty(self) -> bool:
        return self._reader.is_empty()


class CommandWriter:
    """Typed command builder with auto-incrementing sequence numbers.

    Builds Command structs for each CmdType. Does NOT write to the ring --
    call ``SpscRingWriter.write_struct(cmd)`` to send.

    Usage::

        writer = CommandWriter()
        cmd = writer.transfer_h2d(gpu=0, layer=5, expert=42,
                                   zone=0, nbytes=1048576)
        ring.write_struct(cmd)
    """

    __slots__ = ("_seq",)

    def __init__(self, initial_seq: int = 0) -> None:
        self._seq = initial_seq

    @property
    def next_seq(self) -> int:
        """The sequence number that will be assigned to the next command."""
        return self._seq

    def _next(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    # ── Transfer ──────────────────────────────────────────────────────────

    def transfer_h2d(self, gpu: int, layer: int, expert: int, zone: int,
                     sub_component: int = SUB_ALL, nbytes: int = 0,
                     priority: float = 0.0, delay_us: int = 0) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_TRANSFER_H2D
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = STREAM_H2D_TRANSFER
        cmd.payload.transfer.layer_idx = layer
        cmd.payload.transfer.expert_idx = expert
        cmd.payload.transfer.sub_component = sub_component
        cmd.payload.transfer.zone = zone
        cmd.payload.transfer.bytes = nbytes
        cmd.payload.transfer.priority = priority
        cmd.payload.transfer.delay_us = delay_us
        return cmd

    def transfer_d2h(self, gpu: int, layer: int, expert: int, zone: int,
                     sub_component: int = SUB_ALL, nbytes: int = 0,
                     priority: float = 0.0, delay_us: int = 0) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_TRANSFER_D2H
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = STREAM_D2H_TRANSFER
        cmd.payload.transfer.layer_idx = layer
        cmd.payload.transfer.expert_idx = expert
        cmd.payload.transfer.sub_component = sub_component
        cmd.payload.transfer.zone = zone
        cmd.payload.transfer.bytes = nbytes
        cmd.payload.transfer.priority = priority
        cmd.payload.transfer.delay_us = delay_us
        return cmd

    # ── Cache ─────────────────────────────────────────────────────────────

    def cache_reserve(self, gpu: int, layer: int, expert: int, zone: int,
                      is_duplicate: bool = False) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_CACHE_RESERVE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.cache_reserve.layer_idx = layer
        cmd.payload.cache_reserve.expert_idx = expert
        cmd.payload.cache_reserve.zone = zone
        cmd.payload.cache_reserve.is_duplicate = 1 if is_duplicate else 0
        return cmd

    def cache_evict(self, gpu: int, layer: int, expert: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_CACHE_EVICT
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.cache_op.layer_idx = layer
        cmd.payload.cache_op.expert_idx = expert
        return cmd

    def cache_promote(self, gpu: int, layer: int, expert: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_CACHE_PROMOTE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.cache_op.layer_idx = layer
        cmd.payload.cache_op.expert_idx = expert
        return cmd

    def cache_demote(self, gpu: int, layer: int, expert: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_CACHE_DEMOTE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.cache_op.layer_idx = layer
        cmd.payload.cache_op.expert_idx = expert
        return cmd

    # ── Compute ───────────────────────────────────────────────────────────

    def attention_decode(self, gpu: int, layer: int, batch_size: int,
                         hidden_state_buf: int, kv_cache_buf: int,
                         seqlens_buf: int = 0, block_table_buf: int = 0,
                         stream: int = STREAM_ATTENTION,
                         use_graph: bool = False,
                         is_sparse: bool = False) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_ATTENTION_DECODE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.attention.layer_idx = layer
        cmd.payload.attention.batch_size = batch_size
        cmd.payload.attention.use_graph = 1 if use_graph else 0
        cmd.payload.attention.is_sparse = 1 if is_sparse else 0
        cmd.payload.attention.hidden_state_buf_id = hidden_state_buf
        cmd.payload.attention.kv_cache_buf_id = kv_cache_buf
        cmd.payload.attention.seqlens_buf_id = seqlens_buf
        cmd.payload.attention.block_table_buf_id = block_table_buf
        return cmd

    def attention_prefill(self, gpu: int, layer: int, batch_size: int,
                          hidden_state_buf: int, kv_cache_buf: int,
                          seqlens_buf: int = 0, block_table_buf: int = 0,
                          stream: int = STREAM_ATTENTION,
                          use_graph: bool = False,
                          is_sparse: bool = False) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_ATTENTION_PREFILL
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.attention.layer_idx = layer
        cmd.payload.attention.batch_size = batch_size
        cmd.payload.attention.use_graph = 1 if use_graph else 0
        cmd.payload.attention.is_sparse = 1 if is_sparse else 0
        cmd.payload.attention.hidden_state_buf_id = hidden_state_buf
        cmd.payload.attention.kv_cache_buf_id = kv_cache_buf
        cmd.payload.attention.seqlens_buf_id = seqlens_buf
        cmd.payload.attention.block_table_buf_id = block_table_buf
        return cmd

    def gating(self, gpu: int, layer: int, num_tokens: int,
               input_buf: int, output_weights_buf: int,
               output_indices_buf: int,
               stream: int = STREAM_GATING) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_GATING
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.gating.layer_idx = layer
        cmd.payload.gating.num_tokens = num_tokens
        cmd.payload.gating.input_buf_id = input_buf
        cmd.payload.gating.output_weights_buf_id = output_weights_buf
        cmd.payload.gating.output_indices_buf_id = output_indices_buf
        return cmd

    def expert_ffn(self, gpu: int, layer: int, num_experts: int,
                   total_tokens: int, permuted_input_buf: int,
                   output_buf: int, expert_offsets_buf: int,
                   hidden_dim: int = 0, weights_buf: int = 0,
                   workspace_buf: int = 0, quant_mode: int = 0,
                   k_dim: int = 0, gguf_type: int = 0,
                   stream: int = STREAM_EXPERT_FFN) -> Command:
        # quant_mode: 0=NVFP4, 1=FP8, 2=GGUF. The GGUF path (GG-5) additionally
        # needs k_dim (input dim K) and gguf_type (GgufKQuantType ordinal, uniform
        # across experts per GG-6); weights_buf resolves to the [num_experts]
        # device array of per-expert packed GGUF weight-block pointers.
        cmd = Command()
        cmd.cmd_type = CMD_EXPERT_FFN
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.expert_ffn.layer_idx = layer
        cmd.payload.expert_ffn.num_experts = num_experts
        cmd.payload.expert_ffn.total_tokens = total_tokens
        cmd.payload.expert_ffn.permuted_input_buf_id = permuted_input_buf
        cmd.payload.expert_ffn.output_buf_id = output_buf
        cmd.payload.expert_ffn.expert_offsets_buf_id = expert_offsets_buf
        cmd.payload.expert_ffn.hidden_dim = hidden_dim
        cmd.payload.expert_ffn.weights_buf_id = weights_buf
        cmd.payload.expert_ffn.workspace_buf_id = workspace_buf
        cmd.payload.expert_ffn.k_dim = k_dim
        cmd.payload.expert_ffn.quant_mode = quant_mode
        cmd.payload.expert_ffn.gguf_type = gguf_type
        return cmd

    def embedding_lookup(self, gpu: int, num_tokens: int,
                         output_buf: int,
                         row_offset: int = 0,
                         stream: int = STREAM_ATTENTION) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_EMBEDDING_LOOKUP
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.embedding_lookup.num_tokens = num_tokens
        cmd.payload.embedding_lookup.output_buf_id = output_buf
        # TD-PREFILL-SUPERCHUNK: destination row offset into the hidden
        # buffer — sub-chunk embeds of a superchunk land at disjoint rows.
        cmd.payload.embedding_lookup.row_offset = row_offset
        return cmd

    def output_head(self, gpu: int, num_tokens: int,
                    input_buf: int, output_buf: int,
                    readback: bool = False,
                    compute_confidence: bool = False,
                    num_logprobs: int = 0,
                    mtp_head_idx: int = -1,
                    stream: int = STREAM_ATTENTION) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_OUTPUT_HEAD
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.output_head.num_tokens = num_tokens
        cmd.payload.output_head.input_buf_id = input_buf
        cmd.payload.output_head.output_buf_id = output_buf
        cmd.payload.output_head.readback_to_host = 1 if readback else 0
        cmd.payload.output_head.compute_confidence = 1 if compute_confidence else 0
        cmd.payload.output_head.num_logprobs = min(num_logprobs, 20)
        # #16: >= 0 selects the MTP shared head (wire value = mtp_idx + 1).
        cmd.payload.output_head.mtp_head = mtp_head_idx + 1 if mtp_head_idx >= 0 else 0
        return cmd

    def sample_tokens(self, gpu: int, num_tokens: int, logits_buf: int,
                      vocab_size: int, temperature: float = 1.0,
                      top_p: float = 1.0, top_k: int = 0,
                      random_seed: int = 0,
                      stream: int = STREAM_ATTENTION) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_SAMPLE_TOKENS
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.sample_tokens.num_tokens = num_tokens
        cmd.payload.sample_tokens.logits_buf_id = logits_buf
        cmd.payload.sample_tokens.vocab_size = vocab_size
        cmd.payload.sample_tokens.top_k = top_k
        cmd.payload.sample_tokens.temperature = temperature
        cmd.payload.sample_tokens.top_p = top_p
        cmd.payload.sample_tokens.random_seed = random_seed
        return cmd

    def run_attention(self, gpu: int, layer_idx: int, num_seqs: int,
                      is_prefill: int = 0, use_graph: int = 0,
                      is_draft: int = 0, emit_checkpoint: int = 0,
                      chunk_start: int = 0, chunk_len: int = 0,
                      emit_gating: int = 0, store_gating: int = 0,
                      superchunk: int = 0, row_offset: int = 0,
                      stream: int = STREAM_ATTENTION) -> Command:
        cmd = Command()
        cmd.cmd_type = D_B_CMD_RUN_ATTENTION
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.run_attention.layer_idx = layer_idx
        cmd.payload.run_attention.num_seqs = num_seqs
        cmd.payload.run_attention.is_prefill = is_prefill
        cmd.payload.run_attention.use_graph = use_graph
        cmd.payload.run_attention.is_draft = is_draft
        cmd.payload.run_attention.emit_checkpoint = emit_checkpoint
        cmd.payload.run_attention.chunk_start = chunk_start
        cmd.payload.run_attention.chunk_len = chunk_len
        # F-1/F-3: fused router gate at end of attention + routed top-K
        # export to the sideband routing-export slot (production MoE seam).
        cmd.payload.run_attention.emit_gating = emit_gating
        cmd.payload.run_attention.store_gating = store_gating
        # TD-PREFILL-SUPERCHUNK: sub-chunk launch of a superchunk — hidden
        # rows land at row_offset, coverage guard accepts a covered window.
        cmd.payload.run_attention.superchunk = superchunk
        cmd.payload.run_attention.row_offset = row_offset
        return cmd

    def run_moe(self, gpu: int, layer_idx: int, num_seqs: int,
                moe_mode: int = 0,
                apply_residual_correction: int = 0,
                store_gating_output: int = 0,
                emit_checkpoint: int = 0,
                stream: int = STREAM_EXPERT_FFN) -> Command:
        cmd = Command()
        cmd.cmd_type = D_B_CMD_RUN_MOE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.run_moe.layer_idx = layer_idx
        cmd.payload.run_moe.num_seqs = num_seqs
        cmd.payload.run_moe.moe_mode = moe_mode
        cmd.payload.run_moe.apply_residual_correction = apply_residual_correction
        cmd.payload.run_moe.store_gating_output = store_gating_output
        cmd.payload.run_moe.emit_checkpoint = emit_checkpoint
        return cmd

    # ── Fused commands (IPC-8e) ───────────────────────────────────────────

    def prefetch_batch(self, gpu: int, count: int,
                       priority: float = 0.0, delay_us: int = 0) -> Command:
        cmd = Command()
        cmd.cmd_type = D_B_CMD_PREFETCH_BATCH
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.payload.prefetch_batch.count = count
        cmd.payload.prefetch_batch.priority = priority
        cmd.payload.prefetch_batch.delay_us = delay_us
        return cmd

    def evict_batch(self, gpu: int, count: int) -> Command:
        cmd = Command()
        cmd.cmd_type = D_B_CMD_EVICT_BATCH
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.payload.evict_batch.count = count
        return cmd

    def nvme_batch_read(self, gpu: int, count: int) -> Command:
        cmd = Command()
        cmd.cmd_type = B_CMD_NVME_BATCH_READ
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.payload.nvme_batch_read.count = count
        return cmd

    def prefetch_expert(self, gpu: int, layer_idx: int, expert_idx: int,
                        zone: int = 0, target_gpu: int = 0,
                        priority: float = 0.0, delay_us: int = 0) -> Command:
        cmd = Command()
        cmd.cmd_type = D_CMD_PREFETCH_EXPERT
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.payload.prefetch_expert.layer_idx = layer_idx
        cmd.payload.prefetch_expert.expert_idx = expert_idx
        cmd.payload.prefetch_expert.zone = zone
        cmd.payload.prefetch_expert.gpu_idx = target_gpu
        cmd.payload.prefetch_expert.priority = priority
        cmd.payload.prefetch_expert.delay_us = delay_us
        return cmd

    def evict_to_host(self, gpu: int, layer_idx: int,
                      expert_idx: int) -> Command:
        cmd = Command()
        cmd.cmd_type = D_CMD_EVICT_TO_HOST
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.payload.evict_to_host.layer_idx = layer_idx
        cmd.payload.evict_to_host.expert_idx = expert_idx
        cmd.payload.evict_to_host.gpu_idx = gpu
        return cmd

    def slow_evict_to_host(self, gpu: int, layer_idx: int,
                           expert_idx: int) -> Command:
        cmd = Command()
        cmd.cmd_type = D_CMD_SLOW_EVICT_TO_HOST
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.payload.slow_evict_to_host.layer_idx = layer_idx
        cmd.payload.slow_evict_to_host.expert_idx = expert_idx
        cmd.payload.slow_evict_to_host.gpu_idx = gpu
        return cmd

    def stage_expert(self, gpu: int, layer_idx: int, expert_idx: int,
                     zone: int = 0, target_gpu: int = 0,
                     priority: float = 0.0, delay_us: int = 0) -> Command:
        cmd = Command()
        cmd.cmd_type = D_CMD_STAGE_EXPERT
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.payload.stage_expert.layer_idx = layer_idx
        cmd.payload.stage_expert.expert_idx = expert_idx
        cmd.payload.stage_expert.zone = zone
        cmd.payload.stage_expert.gpu_idx = target_gpu
        cmd.payload.stage_expert.priority = priority
        cmd.payload.stage_expert.delay_us = delay_us
        return cmd

    def run_prefetch_probe(self, gpu: int, target_layer: int,
                           num_tokens: int,
                           probe_points: int = 0x07) -> Command:
        cmd = Command()
        cmd.cmd_type = D_CMD_RUN_PREFETCH_PROBE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.payload.run_prefetch_probe.target_layer = target_layer
        cmd.payload.run_prefetch_probe.num_tokens = num_tokens
        cmd.payload.run_prefetch_probe.probe_points = probe_points
        return cmd

    def run_adapter_forward(self, gpu: int, num_tokens: int,
                            adapter_weights_buf_id: int) -> Command:
        cmd = Command()
        cmd.cmd_type = D_CMD_RUN_ADAPTER_FORWARD
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.payload.run_adapter_forward.num_tokens = num_tokens
        cmd.payload.run_adapter_forward.adapter_weights_buf_id = adapter_weights_buf_id
        return cmd

    # -- MTP / self-spec fused draft commands (#62d, #62e) ----------------

    def run_mtp_step(self, gpu: int, mtp_layer_idx: int,
                     seq_id: int, input_token_id: int,
                     step_idx: int = 0,
                     stream: int = STREAM_ATTENTION) -> Command:
        """Fused MTP draft step: embed → MTP projection → decoder → output head."""
        cmd = Command()
        cmd.cmd_type = D_CMD_RUN_MTP_STEP
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.run_mtp_step.mtp_layer_idx = mtp_layer_idx
        cmd.payload.run_mtp_step.seq_id = seq_id
        cmd.payload.run_mtp_step.input_token_id = input_token_id
        cmd.payload.run_mtp_step.step_idx = step_idx
        return cmd

    def mtp_project(self, gpu: int, mtp_layer_idx: int,
                    input_token_id: int, step_idx: int = 0,
                    hidden_row: int = 0,
                    stream: int = STREAM_ATTENTION) -> Command:
        """#16: MTP projection — eh_proj(enorm(Emb(token)) || hnorm(prev_hidden)).

        Production-seam MTP draft step composition:
          mtp_project -> run_attention(mtp_layer, emit/store gating)
          -> fetch_and_run_moe(mtp_layer) -> output_head(mtp_head_idx)
          -> sample_tokens.
        The caller must have the right prev_hidden resident in the hidden-state
        attn buffers (trunk hidden after a main step; MTP output after a
        chained draft step) and, before run_attention, write the sideband
        batch descriptor for the draft position.

        hidden_row selects the attn_buf ROW holding prev_hidden: after a
        batched verify pass over V rows, row j holds the trunk hidden of
        fed position base+j (MTP steps write only row 0, so a sequential
        catch-up chain can consume rows in ascending order).  Default 0 =
        the historical single-row behavior.
        """
        cmd = Command()
        cmd.cmd_type = D_CMD_MTP_PROJECT
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.mtp_project.mtp_layer_idx = mtp_layer_idx
        cmd.payload.mtp_project.input_token_id = input_token_id
        cmd.payload.mtp_project.step_idx = step_idx
        cmd.payload.mtp_project.hidden_row = hidden_row
        return cmd

    def run_dspark_step(self, gpu: int, seq_id: int,
                        anchor_token_id: int, anchor_pos: int,
                        num_query: int = 0, step_idx: int = 0,
                        stream: int = STREAM_ATTENTION) -> Command:
        """DSP-3/DSP-5: fused DSpark draft step — ONE DFlash-backbone forward
        over the whole gamma block + the chained sequential Markov head.

        Caller contract: the target forward must have run with the dspark
        aux-hidden export armed (speculation.method=dspark) so the draft
        context KV covers positions [0, anchor_pos) of seq_id; the command
        CMP_ERRORs otherwise (fail closed).  anchor_pos = the fed-token
        count (the position the anchor token will be fed at next).  The
        gamma sampled draft ids are read from the sideband readback scratch
        at the completion's host_buf_offset (data_bytes = gamma * 4).
        """
        cmd = Command()
        cmd.cmd_type = D_CMD_RUN_DSPARK_STEP
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.run_dspark_step.seq_id = seq_id
        cmd.payload.run_dspark_step.anchor_token_id = anchor_token_id
        cmd.payload.run_dspark_step.anchor_pos = anchor_pos
        cmd.payload.run_dspark_step.num_query = num_query
        cmd.payload.run_dspark_step.step_idx = step_idx
        return cmd

    def run_self_spec_forward(self, gpu: int, seq_id: int,
                              input_token_id: int,
                              draft_expert_count: int = 1,
                              apply_residual_corr: int = 1,
                              store_gating: int = 1,
                              step_idx: int = 0,
                              skip_mask_lo: int = 0,
                              skip_mask_hi: int = 0,
                              stream: int = STREAM_ATTENTION) -> Command:
        """Fused self-speculative forward pass: all layers, top-K expert MoE."""
        cmd = Command()
        cmd.cmd_type = D_CMD_RUN_SELF_SPEC_FORWARD
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.self_spec_forward.seq_id = seq_id
        cmd.payload.self_spec_forward.input_token_id = input_token_id
        cmd.payload.self_spec_forward.draft_expert_count = draft_expert_count
        cmd.payload.self_spec_forward.apply_residual_corr = apply_residual_corr
        cmd.payload.self_spec_forward.store_gating = store_gating
        cmd.payload.self_spec_forward.step_idx = step_idx
        cmd.payload.self_spec_forward.skip_mask_lo = skip_mask_lo
        cmd.payload.self_spec_forward.skip_mask_hi = skip_mask_hi
        return cmd

    # -- Sequence lifecycle ────────────────────────────────────────────────

    def e_seq_create(self, gpu: int, seq_id: int, prompt_len: int,
                     pool: int = 0) -> Command:
        cmd = Command()
        cmd.cmd_type = E_CMD_SEQ_CREATE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.payload.seq_create.seq_id = seq_id
        cmd.payload.seq_create.prompt_len = prompt_len
        cmd.payload.seq_create.pool = pool
        return cmd

    def e_seq_free(self, gpu: int, seq_id: int) -> Command:
        cmd = Command()
        cmd.cmd_type = E_CMD_SEQ_FREE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.payload.seq_free.seq_id = seq_id
        return cmd

    def fetch_and_run_moe(self, gpu: int, layer_idx: int, num_seqs: int,
                          expert_count: int, timeout_us: int = 0,
                          moe_mode: int = 0,
                          stream: int = STREAM_EXPERT_FFN) -> Command:
        """E_CMD_FETCH_AND_RUN_MOE — the PRODUCTION routed-MoE seam (#90).

        The caller must first write the routed expert list to the sideband
        with write_sideband_expert_prefetch() (entries in gating
        selection-rank order) and the batch descriptor for the token
        positions (same contract as run_attention).  The daemon fetches
        only the missing experts and runs the MoE; a missing expert past
        timeout_us degrades like a truncation (excluded from the run).
        F-6 decider fields are left 0 → fetch every listed entry.
        """
        cmd = Command()
        cmd.cmd_type = E_CMD_FETCH_AND_RUN_MOE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.fetch_and_run_moe.layer_idx = layer_idx
        cmd.payload.fetch_and_run_moe.num_seqs = num_seqs
        cmd.payload.fetch_and_run_moe.expert_count = expert_count
        cmd.payload.fetch_and_run_moe.timeout_us = timeout_us
        cmd.payload.fetch_and_run_moe.moe_mode = moe_mode
        return cmd

    def fetch_and_run_moe_big(self, gpu: int, layer_idx: int, num_seqs: int,
                              expert_count: int, timeout_us: int = 0,
                              moe_mode: int = 0, chunk_tokens: int = 0,
                              stream: int = STREAM_EXPERT_FFN) -> Command:
        """E_CMD_FETCH_AND_RUN_MOE_BIG — big-batch chunked MoE
        (TD-PREFILL-MOE-BIG).

        Same sideband contract as fetch_and_run_moe (expert list +
        batch descriptors written first), extended two ways daemon-side:
        the grouped GEMMs run in ~chunk_tokens chunks with the transient
        scratch reused (never OOM up to EngineInfo.moe_batch_capacity
        tokens), and the rolling expert waves are double-buffered (wave
        i+1's H2D fetches stream while wave i's chunk GEMMs compute).
        chunk_tokens=0 = engine default (compute.moe_big_chunk_tokens).
        """
        cmd = Command()
        cmd.cmd_type = E_CMD_FETCH_AND_RUN_MOE_BIG
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.fetch_and_run_moe_big.layer_idx = layer_idx
        cmd.payload.fetch_and_run_moe_big.num_seqs = num_seqs
        cmd.payload.fetch_and_run_moe_big.expert_count = expert_count
        cmd.payload.fetch_and_run_moe_big.timeout_us = timeout_us
        cmd.payload.fetch_and_run_moe_big.moe_mode = moe_mode
        cmd.payload.fetch_and_run_moe_big.chunk_tokens = chunk_tokens
        return cmd

    def rmsnorm(self, gpu: int, num_tokens: int,
                input_buf: int, output_buf: int, weight_buf: int,
                eps: float = 1e-6, hidden_size: int = 0,
                stream: int = STREAM_ATTENTION) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_RMSNORM
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.rmsnorm.num_tokens = num_tokens
        cmd.payload.rmsnorm.input_buf_id = input_buf
        cmd.payload.rmsnorm.output_buf_id = output_buf
        cmd.payload.rmsnorm.weight_buf_id = weight_buf
        cmd.payload.rmsnorm.eps = eps
        cmd.payload.rmsnorm.hidden_size = hidden_size
        return cmd

    def swiglu(self, gpu: int, num_tokens: int, hidden_dim: int,
               input_buf: int, output_buf: int,
               stream: int = STREAM_EXPERT_FFN) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_SWIGLU
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.swiglu.num_tokens = num_tokens
        cmd.payload.swiglu.hidden_dim = hidden_dim
        cmd.payload.swiglu.input_buf_id = input_buf
        cmd.payload.swiglu.output_buf_id = output_buf
        return cmd

    def moe_permute(self, gpu: int, num_tokens: int, num_experts: int,
                    input_buf: int, output_buf: int,
                    indices_buf: int, offsets_buf: int,
                    topk: int = 0, hidden_dim: int = 0,
                    workspace_buf: int = 0,
                    stream: int = STREAM_EXPERT_FFN) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_MOE_PERMUTE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.moe_permute.num_tokens = num_tokens
        cmd.payload.moe_permute.num_experts = num_experts
        cmd.payload.moe_permute.input_buf_id = input_buf
        cmd.payload.moe_permute.output_buf_id = output_buf
        cmd.payload.moe_permute.indices_buf_id = indices_buf
        cmd.payload.moe_permute.offsets_buf_id = offsets_buf
        cmd.payload.moe_permute.topk = topk
        cmd.payload.moe_permute.hidden_dim = hidden_dim
        cmd.payload.moe_permute.workspace_buf_id = workspace_buf
        return cmd

    def moe_unpermute(self, gpu: int, num_tokens: int, num_experts: int,
                      input_buf: int, output_buf: int, indices_buf: int,
                      topk: int = 0, hidden_dim: int = 0,
                      weights_buf: int = 0,
                      stream: int = STREAM_EXPERT_FFN) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_MOE_UNPERMUTE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.moe_unpermute.num_tokens = num_tokens
        cmd.payload.moe_unpermute.num_experts = num_experts
        cmd.payload.moe_unpermute.input_buf_id = input_buf
        cmd.payload.moe_unpermute.output_buf_id = output_buf
        cmd.payload.moe_unpermute.indices_buf_id = indices_buf
        cmd.payload.moe_unpermute.topk = topk
        cmd.payload.moe_unpermute.hidden_dim = hidden_dim
        cmd.payload.moe_unpermute.weights_buf_id = weights_buf
        return cmd

    def dcp_correction(self, gpu: int, batch_size: int,
                       input_buf: int, output_buf: int, lse_buf: int,
                       stream: int = STREAM_ATTENTION) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_DCP_CORRECTION
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.dcp_correction.batch_size = batch_size
        cmd.payload.dcp_correction.input_buf_id = input_buf
        cmd.payload.dcp_correction.output_buf_id = output_buf
        cmd.payload.dcp_correction.lse_buf_id = lse_buf
        return cmd

    def nccl_allreduce(self, gpu: int, count: int, buf_id: int,
                       dtype: int = 0,
                       stream: int = STREAM_ATTENTION) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_NCCL_ALLREDUCE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.nccl_allreduce.count = count
        cmd.payload.nccl_allreduce.buf_id = buf_id
        cmd.payload.nccl_allreduce.dtype = dtype
        return cmd

    def dynamic_fp8_quant(self, gpu: int, num_tokens: int, hidden_dim: int,
                          input_buf: int, output_buf: int, scales_buf: int,
                          stream: int = STREAM_ATTENTION) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_DYNAMIC_FP8_QUANT
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.dynamic_fp8_quant.num_tokens = num_tokens
        cmd.payload.dynamic_fp8_quant.hidden_dim = hidden_dim
        cmd.payload.dynamic_fp8_quant.input_buf_id = input_buf
        cmd.payload.dynamic_fp8_quant.output_buf_id = output_buf
        cmd.payload.dynamic_fp8_quant.scales_buf_id = scales_buf
        return cmd

    # ── Prefetch compute ──────────────────────────────────────────────────

    def prescope_gating(self, gpu: int, target_layer: int,
                        num_tokens: int, hidden_state_buf: int,
                        output_buf: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_PRESCOPE_GATING
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = STREAM_PREFETCH_COMPUTE
        cmd.payload.prescope.target_layer_idx = target_layer
        cmd.payload.prescope.num_tokens = num_tokens
        cmd.payload.prescope.hidden_state_buf_id = hidden_state_buf
        cmd.payload.prescope.output_buf_id = output_buf
        return cmd

    def probe_mlp(self, gpu: int, probe_point: int, num_tokens: int,
                  hidden_state_buf: int, output_buf: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_PROBE_MLP
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = STREAM_PREFETCH_COMPUTE
        cmd.payload.probe.probe_point_idx = probe_point
        cmd.payload.probe.num_tokens = num_tokens
        cmd.payload.probe.hidden_state_buf_id = hidden_state_buf
        cmd.payload.probe.output_buf_id = output_buf
        return cmd

    # ── CUDA graph ────────────────────────────────────────────────────────

    def graph_replay(self, gpu: int, graph_type: int, batch_size: int,
                     stream: int = STREAM_ATTENTION) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_GRAPH_REPLAY
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.graph.graph_type = graph_type
        cmd.payload.graph.batch_size = batch_size
        return cmd

    # ── Synchronization ───────────────────────────────────────────────────

    def record_event(self, gpu: int, stream: int, event_id: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_RECORD_EVENT
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.event.event_id = event_id
        return cmd

    def stream_wait_event(self, gpu: int, stream: int,
                          event_id: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_STREAM_WAIT_EVENT
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = stream
        cmd.payload.event.event_id = event_id
        return cmd

    # ── Placement ─────────────────────────────────────────────────────────

    def compute_affinity_hints(self, num_gpus: int,
                               gpu_capacity_slots: list[int]) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_COMPUTE_AFFINITY_HINTS
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = 0
        cmd.stream_id = 0
        cmd.payload.affinity_hints.num_gpus = num_gpus
        for i, cap in enumerate(gpu_capacity_slots[:MAX_GPUS]):
            cmd.payload.affinity_hints.gpu_capacity_slots[i] = cap
        return cmd

    def numa_migrate(self, gpu: int, layer: int, expert: int,
                     target_numa_node: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_NUMA_MIGRATE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.numa_migrate.layer_idx = layer
        cmd.payload.numa_migrate.expert_idx = expert
        cmd.payload.numa_migrate.target_numa_node = target_numa_node
        return cmd

    # ── Sequence lifecycle ────────────────────────────────────────────────

    def seq_create(self, gpu: int, seq_id: int, prompt_len: int,
                   pool: int = 0) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_SEQ_CREATE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.seq_create.seq_id = seq_id
        cmd.payload.seq_create.prompt_len = prompt_len
        cmd.payload.seq_create.pool = pool
        return cmd

    def seq_free(self, gpu: int, seq_id: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_SEQ_FREE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.seq_free.seq_id = seq_id
        return cmd

    def seq_fork(self, gpu: int, src_seq_id: int,
                 dst_seq_id: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_SEQ_FORK
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.seq_fork.src_seq_id = src_seq_id
        cmd.payload.seq_fork.dst_seq_id = dst_seq_id
        return cmd

    # ── NVMe tier + cancel ───────────────────────────────────────────────

    def nvme_read(self, gpu: int, layer: int, expert: int,
                  gpu_hint: int = 0) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_NVME_READ
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.nvme_read.layer_idx = layer
        cmd.payload.nvme_read.expert_idx = expert
        cmd.payload.nvme_read.gpu_hint = gpu_hint
        return cmd

    def nvme_write(self, gpu: int, layer: int, expert: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_NVME_WRITE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.nvme_write.layer_idx = layer
        cmd.payload.nvme_write.expert_idx = expert
        return cmd

    def nvme_evict_host(self, gpu: int, layer: int,
                        expert: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_NVME_EVICT_HOST
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.nvme_evict_host.layer_idx = layer
        cmd.payload.nvme_evict_host.expert_idx = expert
        return cmd

    def cancel_transfer(self, gpu: int,
                        target_cmd_seq: int) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_CANCEL_TRANSFER
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        cmd.payload.cancel_transfer.target_cmd_seq = target_cmd_seq
        return cmd

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def config_update(
        self, updates: list[tuple[int, int, int]]
    ) -> Command:
        if len(updates) > 29:
            raise ValueError(f"Too many updates: {len(updates)} > 29")
        cmd = Command()
        cmd.cmd_type = CMD_CONFIG_UPDATE
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = 0
        cmd.stream_id = 0
        cmd.payload.config_update.count = len(updates)
        for i, (fid, vtype, raw) in enumerate(updates):
            cmd.payload.config_update.entries[i].field_id = fid
            cmd.payload.config_update.entries[i].value_type = vtype
            cmd.payload.config_update.entries[i].raw_value = raw
        return cmd

    def shutdown(self, gpu: int = 0) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_SHUTDOWN
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        return cmd

    def noop(self, gpu: int = 0) -> Command:
        cmd = Command()
        cmd.cmd_type = CMD_NOOP
        cmd.cmd_seq = self._next()
        cmd.gpu_idx = gpu
        cmd.stream_id = 0
        return cmd


# ── Sideband helpers ────────────────────────────────────────────────────────


def write_sideband_token_ids(
    sideband_base: int, token_ids: Sequence[int]
) -> None:
    """Write token IDs to the sideband token IDs sub-region.

    Call this before sending CMD_EMBEDDING_LOOKUP so the C++ handler
    can read the token IDs from the sideband region.
    """
    n = len(token_ids)
    if n > MAX_SIDEBAND_TOKEN_IDS:
        raise ValueError(
            f"Too many token IDs: {n} > {MAX_SIDEBAND_TOKEN_IDS}"
        )
    arr_type = ctypes.c_uint32 * n
    arr = arr_type.from_address(sideband_base + SIDEBAND_TOKEN_IDS_OFF)
    for i, tid in enumerate(token_ids):
        arr[i] = tid


def write_sideband_batch_descriptors(
    sideband_base: int,
    entries: Sequence[tuple[int, int]],
) -> None:
    """Write batch descriptor entries to the sideband region.

    Each entry is (seq_id, token_pos).  Call before D_B_CMD_RUN_ATTENTION
    so build_kv_metadata() can read the batch descriptors.
    """
    n = len(entries)
    if n > MAX_BATCH_DESCRIPTORS:
        raise ValueError(
            f"Too many batch descriptors: {n} > {MAX_BATCH_DESCRIPTORS}"
        )
    arr_type = BatchDescriptorEntry * n
    arr = arr_type.from_address(sideband_base + SIDEBAND_BATCH_DESCRIPTOR_OFF)
    for i, (seq_id, token_pos) in enumerate(entries):
        arr[i].seq_id = seq_id
        arr[i].token_pos = token_pos
        arr[i]._pad = 0


def read_sideband_routing_export(
    sideband_base: int,
) -> tuple[int, list[int]]:
    """Read the routed top-K export published by RUN_ATTENTION [store_gating].

    Returns (layer_idx, indices) where indices is the flattened
    num_tokens * topk routed expert index list in gating SELECTION-RANK
    order (negative entries — unrouted slots — are filtered out).  Call
    after the attention completion arrives and before dispatching
    E_CMD_FETCH_AND_RUN_MOE for the same layer.
    """
    hdr = RoutingExportHeader.from_address(
        sideband_base + SIDEBAND_ROUTING_EXPORT_OFF
    )
    n = int(hdr.num_tokens) * int(hdr.topk)
    if n <= 0:
        return (int(hdr.layer_idx), [])
    arr = (ctypes.c_int32 * n).from_address(
        sideband_base + SIDEBAND_ROUTING_EXPORT_INDICES_OFF
    )
    return (int(hdr.layer_idx), [int(arr[i]) for i in range(n) if arr[i] >= 0])


def write_sideband_expert_prefetch(
    sideband_base: int,
    entries: Sequence[tuple[int, int, int, int]],
) -> None:
    """Write the routed expert list consumed by E_CMD_FETCH_AND_RUN_MOE.

    Each entry is (layer_idx, expert_idx, zone, gpu_idx), in gating
    selection-rank order (INV-10c-2: position = model priority for the
    F-6 decider).  Call before sending fetch_and_run_moe.
    """
    n = len(entries)
    if n > MAX_EXPERT_PREFETCH:
        raise ValueError(
            f"Too many expert prefetch entries: {n} > {MAX_EXPERT_PREFETCH}"
        )
    arr_type = ExpertPrefetchEntry * n
    arr = arr_type.from_address(sideband_base + SIDEBAND_EXPERT_PREFETCH_OFF)
    for i, (layer_idx, expert_idx, zone, gpu_idx) in enumerate(entries):
        arr[i].layer_idx = layer_idx
        arr[i].expert_idx = expert_idx
        arr[i].zone = zone
        arr[i].gpu_idx = gpu_idx

# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True, initializedcheck=False
"""Cython hot path for the LayerStoRm ring bridge.

Accelerates the per-command IPC operations that dominate the bridge's
host-side overhead (~320 ring round-trips per decode step at 78 layers):

  ring_write        — SPSC command-ring producer (one memcpy + seq bump)
  cmp_poll          — SPSC completion-ring consumer + typed field extract
                      (no ctypes.from_buffer_copy allocation)
  write_u32         — sideband token-id store
  write_batch_desc  — contiguous single-sequence batch descriptors
  write_moe_entries — ExpertPrefetchEntry[] + aligned ExpertEvictionEntry[]
  routing_union     — routing-export dedup (first-occurrence order)

Memory model: x86-64 TSO — aligned 64-bit loads/stores are atomic and
naturally ordered for this SPSC pattern (same contract as the C++
CommandRing / the pure-Python orchestrator.spsc_ring).  The struct field
offsets baked in here are asserted against the ctypes protocol mirror by
``bridge.ring_bridge._layout_selfcheck`` and against the C++ header by the
pybind ``ipc_struct_layout()`` introspection test.

Layout constants (ipc_protocol.h):
  RingHeader: producer_seq @ 0, consumer_seq @ 64 (both u64), size 128
  Completion (128 B): cmp_type @0, cmd_seq @4, gpu_idx @8, status @12 u32;
    compute payload @16: cmd_type @16, layer_idx @20, host_buf_offset @24,
    data_bytes @28 u32, top1_prob @32, entropy @36 f32
    error payload @16: error_category @16 u32, message @20 char[80]
  BatchDescriptorEntry (16 B): seq_id u64 @0, token_pos u32 @8
  ExpertPrefetchEntry (8 B): layer u32 @0, expert u16 @4, zone u8 @6, gpu u8 @7
  ExpertEvictionEntry (8 B): layer u32 @0, expert u16 @4, gpu u8 @6, pad u8 @7
"""

from libc.string cimport memcpy, memset
from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t, int32_t,
                          uintptr_t)
from posix.time cimport CLOCK_MONOTONIC, clock_gettime, timespec

cdef extern from "sched.h":
    int sched_yield() nogil

cdef extern from *:
    """
    #if defined(__x86_64__)
    #include <immintrin.h>
    #define ls_cpu_pause() _mm_pause()
    #else
    #define ls_cpu_pause() ((void)0)
    #endif
    """
    void ls_cpu_pause() nogil

# Spin-then-yield (the orchestrator wait pattern): hot-spin ~16k pause
# iterations (~50-150 us) so rendezvous-scale completions are detected at
# sub-us latency, THEN fall back to sched_yield + deadline checks for
# ms-scale GPU waits. Yielding every iteration costs ~70 us per rendezvous
# in scheduler wake-up latency — measured as the dominant bridge overhead.
cdef enum:
    # Hot window LONGER than any decode/verify per-layer wait (~1-3 ms):
    # ~1M pause iterations ≈ 10-20 ms, so during decode the completion is
    # ALWAYS detected hot (~sub-us, like the C++ fixture's full-spin
    # waiter) — a 16k window was exhausted before ms-scale GPU work
    # finished, leaving every rendezvous a yield-mode (~10-70us) wake.
    # Yield mode remains for prefill/boot-scale waits (100ms+), where
    # per-completion wake latency is irrelevant and burning the core for
    # minutes would be rude.
    _SPIN_ITERS = 1048576
    _SPIN_CLOCK_MASK = 65535   # deadline check every 64k spins (~1 ms)
    # Yield phase: deadline via clock_gettime (vDSO, ~20ns — but still
    # per-iteration fat next to sched_yield) only every 256th yield;
    # timeout precision degrades by ~O(100us), irrelevant vs 60-300s.
    _YIELD_CLOCK_MASK = 255

CMP_ERROR = 0xEE00

# v2 API: direct-into-ring command packing (no ctypes objects, no bytes
# copies), a GIL-RELEASED completion wait (checkpoint-skip inside), and a
# fused routing-export→dedup→sideband-entries→FETCH-send per MoE layer.
# bridge.ring_bridge falls back to the v1 functions when this constant is
# absent (stale .so).
API_VERSION = 2

cdef uint32_t _CMP_ERROR = 0xEE00
cdef uint32_t _CMP_CHECKPOINT = 0x0200


def ring_write(uintptr_t header_addr, uintptr_t slots_base,
               uint64_t mask, uint64_t slot_count, uint32_t slot_size,
               bytes data):
    """Producer write of one slot. Returns False when the ring is full."""
    cdef uint64_t prod = (<volatile uint64_t*> header_addr)[0]
    cdef uint64_t cons = (<volatile uint64_t*> (header_addr + 64))[0]
    if prod - cons >= slot_count:
        return False
    cdef uintptr_t dest = slots_base + (prod & mask) * slot_size
    memcpy(<void*> dest, <const char*> data, slot_size)
    # Release: TSO store ordering — the data memcpy above is globally
    # visible before this producer_seq store.
    (<volatile uint64_t*> header_addr)[0] = prod + 1
    return True


def cmp_poll(uintptr_t header_addr, uintptr_t slots_base,
             uint64_t mask, uint32_t slot_size):
    """Consumer read of one completion slot.

    Returns None when empty, else the Cmp tuple
    (cmp_type, cmd_seq, gpu_idx, status, cmd_type, layer_idx,
     host_buf_offset, data_bytes, top1_prob, entropy, err_msg).
    """
    cdef uint64_t cons = (<volatile uint64_t*> (header_addr + 64))[0]
    cdef uint64_t prod = (<volatile uint64_t*> header_addr)[0]
    if cons >= prod:
        return None
    cdef uintptr_t src = slots_base + (cons & mask) * slot_size
    cdef uint32_t cmp_type = (<uint32_t*> src)[0]
    cdef uint32_t cmd_seq = (<uint32_t*> (src + 4))[0]
    cdef uint32_t gpu_idx = (<uint32_t*> (src + 8))[0]
    cdef uint32_t status = (<uint32_t*> (src + 12))[0]
    cdef uint32_t cmd_type = 0, layer_idx = 0, hbo = 0, dbytes = 0
    cdef float top1 = 0.0, ent = 0.0
    cdef bytes msg_b
    cdef object err_msg = ""
    if cmp_type == CMP_ERROR:
        # error payload: message char[80] at +20
        msg_b = (<char*> (src + 20))[:80]
        err_msg = msg_b.split(b"\0")[0].decode("utf-8", "replace")
    else:
        cmd_type = (<uint32_t*> (src + 16))[0]
        layer_idx = (<uint32_t*> (src + 20))[0]
        hbo = (<uint32_t*> (src + 24))[0]
        dbytes = (<uint32_t*> (src + 28))[0]
        top1 = (<float*> (src + 32))[0]
        ent = (<float*> (src + 36))[0]
    # Consume AFTER the copy-out (release on TSO).
    (<volatile uint64_t*> (header_addr + 64))[0] = cons + 1
    return (cmp_type, cmd_seq, gpu_idx, status, cmd_type, layer_idx,
            hbo, dbytes, top1, ent, err_msg)


def write_u32(uintptr_t addr, list values):
    cdef uint32_t* p = <uint32_t*> addr
    cdef Py_ssize_t i, n = len(values)
    for i in range(n):
        p[i] = <uint32_t> values[i]


def write_batch_desc(uintptr_t addr, uint64_t seq_id, uint32_t pos0,
                     uint32_t n):
    """n rows of one sequence at contiguous positions pos0+b."""
    cdef uint8_t* base = <uint8_t*> addr
    cdef uint32_t b
    for b in range(n):
        (<uint64_t*> (base + b * 16))[0] = seq_id
        (<uint32_t*> (base + b * 16 + 8))[0] = pos0 + b
        (<uint32_t*> (base + b * 16 + 12))[0] = 0


def write_moe_entries(uintptr_t prefetch_addr, uintptr_t evict_addr,
                      uint32_t layer, list experts, list assigns,
                      list victims):
    """ExpertPrefetchEntry[] (zone=0) + index-aligned ExpertEvictionEntry[].

    victims[i] = (layer, expert, gpu) with expert 0xFFFF = no-victim
    sentinel (13c-2.0 contract)."""
    cdef uint8_t* pfe = <uint8_t*> prefetch_addr
    cdef uint8_t* eve = <uint8_t*> evict_addr
    cdef Py_ssize_t i, n = len(experts)
    cdef tuple v
    for i in range(n):
        (<uint32_t*> (pfe + i * 8))[0] = layer
        (<uint16_t*> (pfe + i * 8 + 4))[0] = <uint16_t> experts[i]
        (pfe + i * 8 + 6)[0] = 0                       # zone
        (pfe + i * 8 + 7)[0] = <uint8_t> assigns[i]    # gpu_idx
        v = victims[i]
        (<uint32_t*> (eve + i * 8))[0] = <uint32_t> v[0]
        (<uint16_t*> (eve + i * 8 + 4))[0] = <uint16_t> v[1]
        (eve + i * 8 + 6)[0] = <uint8_t> v[2]
        (eve + i * 8 + 7)[0] = 0


def routing_union(uintptr_t indices_addr, uint32_t rn,
                  uint32_t num_experts):
    """Dedup the flattened routing export to the first-occurrence
    (selection-rank) expert union."""
    cdef int32_t* idx = <int32_t*> indices_addr
    cdef bytearray seen = bytearray(num_experts)
    cdef uint8_t* sp = seen
    cdef list out = []
    cdef uint32_t k
    cdef int32_t e
    for k in range(rn):
        e = idx[k]
        if e < 0 or <uint32_t> e >= num_experts or sp[e]:
            continue
        sp[e] = 1
        out.append(e)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# v2 hot path
#
# Command layout (256 B): cmd_type @0, cmd_seq @4, gpu_idx @8, stream_id @12
# (all u32); payload @16. Absolute payload offsets used below (base +16):
#   embedding_lookup: num_tokens @16, output_buf_id @20, row_offset @24
#   run_attention: layer @16, num_seqs @20, is_prefill @24, use_graph @25,
#     is_draft @26, emit_checkpoint @27, chunk_start @28, chunk_len @32,
#     emit_gating @36, store_gating @37, superchunk @38, row_offset @40
#   run_moe: layer @16, num_seqs @20, moe_mode @24, apply_residual @25,
#     store_gating @26, emit_checkpoint @27, use_precomputed @28
#   fetch_and_run_moe: layer @16, num_seqs @20, expert_count @24,
#     timeout_us @28, moe_mode @32, weight_count @33, min_experts @34,
#     max_new_fetches @36, have_evict_map @38, gating_weight_threshold @40
#   output_head: num_tokens @16, input_buf @20, output_buf @24,
#     readback @28, compute_confidence @29, num_logprobs @30, mtp_head @31
#   sample_tokens: num_tokens @16, logits_buf @20, vocab @24, top_k @28,
#     temperature @32, top_p @36, random_seed @40
#   seq_create: seq_id @16, prompt_len @24, pool @28;  seq_free: seq_id @16
#   run_dspark_step: seq_id @16, anchor_token @24, anchor_pos @28,
#     num_query @32, step_idx @33
#   prefetch_batch: count @16, priority @20, delay_us @24
# Every offset is asserted against the ctypes protocol mirror at import by
# bridge.ring_bridge._layout_selfcheck (write-through-ctypes probes).
# ═══════════════════════════════════════════════════════════════════════════


cdef inline uintptr_t _claim(uintptr_t hdr, uintptr_t slots, uint64_t mask,
                             uint64_t slot_count, uint32_t slot_size) nogil:
    """Claim + zero the next producer slot; 0 when the ring is full."""
    cdef uint64_t prod = (<volatile uint64_t*> hdr)[0]
    cdef uint64_t cons = (<volatile uint64_t*> (hdr + 64))[0]
    if prod - cons >= slot_count:
        return 0
    cdef uintptr_t dest = slots + (prod & mask) * slot_size
    memset(<void*> dest, 0, slot_size)
    return dest


cdef inline void _publish(uintptr_t hdr) nogil:
    # Release on TSO: the slot writes above are visible before the bump.
    (<volatile uint64_t*> hdr)[0] = (<volatile uint64_t*> hdr)[0] + 1


cdef inline void _hdr4(uintptr_t d, uint32_t cmd_type, uint32_t seq,
                       uint32_t gpu, uint32_t stream) nogil:
    (<uint32_t*> d)[0] = cmd_type
    (<uint32_t*> (d + 4))[0] = seq
    (<uint32_t*> (d + 8))[0] = gpu
    (<uint32_t*> (d + 12))[0] = stream


def send_embed(uintptr_t hdr, uintptr_t slots, uint64_t mask,
               uint64_t count, uint32_t ssz, uint32_t seq,
               uint32_t num_tokens, uint32_t out_buf,
               uint32_t row_offset=0):
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return False
    _hdr4(d, 0x0104, seq, 0, 0)                 # CMD_EMBEDDING_LOOKUP
    (<uint32_t*> (d + 16))[0] = num_tokens
    (<uint32_t*> (d + 20))[0] = out_buf
    (<uint32_t*> (d + 24))[0] = row_offset      # SC superchunk staging row
    _publish(hdr)
    return True


def send_attention(uintptr_t hdr, uintptr_t slots, uint64_t mask,
                   uint64_t count, uint32_t ssz, uint32_t seq,
                   uint32_t layer, uint32_t num_seqs, uint32_t is_prefill,
                   uint32_t chunk_start, uint32_t chunk_len,
                   uint32_t emit_gating, uint32_t superchunk=0,
                   uint32_t row_offset=0):
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return False
    _hdr4(d, 0x0800, seq, 0, 0)                 # D_B_CMD_RUN_ATTENTION
    (<uint32_t*> (d + 16))[0] = layer
    (<uint32_t*> (d + 20))[0] = num_seqs
    (<uint8_t*> (d + 24))[0] = <uint8_t> is_prefill
    (<uint32_t*> (d + 28))[0] = chunk_start
    (<uint32_t*> (d + 32))[0] = chunk_len
    (<uint8_t*> (d + 36))[0] = <uint8_t> emit_gating
    (<uint8_t*> (d + 37))[0] = <uint8_t> emit_gating   # store_gating
    (<uint8_t*> (d + 38))[0] = <uint8_t> superchunk    # SC sub-launch
    (<uint32_t*> (d + 40))[0] = row_offset             # SC staging row
    _publish(hdr)
    return True


def send_dense_moe(uintptr_t hdr, uintptr_t slots, uint64_t mask,
                   uint64_t count, uint32_t ssz, uint32_t seq,
                   uint32_t layer, uint32_t num_seqs):
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return False
    _hdr4(d, 0x0801, seq, 0, 0)                 # D_B_CMD_RUN_MOE
    (<uint32_t*> (d + 16))[0] = layer
    (<uint32_t*> (d + 20))[0] = num_seqs
    _publish(hdr)
    return True


def send_head(uintptr_t hdr, uintptr_t slots, uint64_t mask,
              uint64_t count, uint32_t ssz, uint32_t seq,
              uint32_t num_tokens, uint32_t in_buf, uint32_t out_buf,
              uint32_t readback):
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return False
    _hdr4(d, 0x0105, seq, 0, 0)                 # CMD_OUTPUT_HEAD
    (<uint32_t*> (d + 16))[0] = num_tokens
    (<uint32_t*> (d + 20))[0] = in_buf
    (<uint32_t*> (d + 24))[0] = out_buf
    (<uint8_t*> (d + 28))[0] = <uint8_t> readback
    (<uint8_t*> (d + 29))[0] = 1                # compute_confidence
    _publish(hdr)
    return True


def send_sample(uintptr_t hdr, uintptr_t slots, uint64_t mask,
                uint64_t count, uint32_t ssz, uint32_t seq,
                uint32_t num_tokens, uint32_t logits_buf, uint32_t vocab,
                float temperature=0.0, float top_p=1.0, uint32_t top_k=0,
                uint64_t seed=42):
    """CMD_SAMPLE_TOKENS. Defaults = the champion greedy arm (argmax,
    byte-identical to the historical hardcoded form); the orchestrator's
    sampled path passes real SamplingParams."""
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return False
    _hdr4(d, 0x010D, seq, 0, 0)                 # CMD_SAMPLE_TOKENS
    (<uint32_t*> (d + 16))[0] = num_tokens
    (<uint32_t*> (d + 20))[0] = logits_buf
    (<uint32_t*> (d + 24))[0] = vocab
    (<uint32_t*> (d + 28))[0] = top_k
    (<float*> (d + 32))[0] = temperature
    (<float*> (d + 36))[0] = top_p
    (<uint64_t*> (d + 40))[0] = seed
    _publish(hdr)
    return True


def send_seq_create(uintptr_t hdr, uintptr_t slots, uint64_t mask,
                    uint64_t count, uint32_t ssz, uint32_t seq,
                    uint64_t seq_id, uint32_t prompt_len):
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return False
    _hdr4(d, 0x0600, seq, 0, 0)                 # CMD_SEQ_CREATE
    (<uint64_t*> (d + 16))[0] = seq_id
    (<uint32_t*> (d + 24))[0] = prompt_len
    _publish(hdr)
    return True


def send_seq_free(uintptr_t hdr, uintptr_t slots, uint64_t mask,
                  uint64_t count, uint32_t ssz, uint32_t seq,
                  uint64_t seq_id):
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return False
    _hdr4(d, 0x0601, seq, 0, 0)                 # CMD_SEQ_FREE
    (<uint64_t*> (d + 16))[0] = seq_id
    _publish(hdr)
    return True


def send_dspark(uintptr_t hdr, uintptr_t slots, uint64_t mask,
                uint64_t count, uint32_t ssz, uint32_t seq,
                uint64_t seq_id, uint32_t anchor_token, uint32_t anchor_pos,
                uint32_t gamma):
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return False
    _hdr4(d, 0x0A09, seq, 0, 0)                 # D_CMD_RUN_DSPARK_STEP
    (<uint64_t*> (d + 16))[0] = seq_id
    (<uint32_t*> (d + 24))[0] = anchor_token
    (<uint32_t*> (d + 28))[0] = anchor_pos
    (<uint8_t*> (d + 32))[0] = <uint8_t> gamma
    _publish(hdr)
    return True


def send_prefetch_batch(uintptr_t hdr, uintptr_t slots, uint64_t mask,
                        uint64_t count, uint32_t ssz, uint32_t seq,
                        uint32_t n, float priority):
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return False
    _hdr4(d, 0x0802, seq, 0, 0)                 # D_B_CMD_PREFETCH_BATCH
    (<uint32_t*> (d + 16))[0] = n
    (<float*> (d + 20))[0] = priority
    _publish(hdr)
    return True


def fetch_moe_from_export(uintptr_t hdr, uintptr_t slots, uint64_t mask,
                          uint64_t count, uint32_t ssz, uint32_t seq,
                          uintptr_t routing_hdr, uintptr_t routing_idx,
                          uintptr_t prefetch_addr, uintptr_t evict_addr,
                          uint32_t layer, uint32_t num_seqs,
                          uint32_t num_experts, uint32_t tp,
                          uint32_t timeout_us, uint32_t max_entries):
    """Fused ACT-arm MoE layer issue: validate the routing export, dedup the
    union (first-occurrence order), write the sideband ExpertPrefetchEntry[]
    (zone 0, gpu = e%tp) + sentinel ExpertEvictionEntry[], pack + publish
    E_CMD_FETCH_AND_RUN_MOE (have_evict_map=0).

    Returns (count) >= 1, or a negative code: -1 ring full, -2 export
    row-count mismatch, -3 export layer mismatch, -4 empty union.
    """
    cdef uint32_t rows = (<uint32_t*> routing_hdr)[0]
    cdef uint32_t topk = (<uint32_t*> (routing_hdr + 4))[0]
    cdef uint32_t hdr_layer = (<uint32_t*> (routing_hdr + 8))[0]
    if rows != num_seqs:
        return -2
    if hdr_layer != layer:
        return -3
    cdef int32_t* idx = <int32_t*> routing_idx
    cdef uint8_t* pfe = <uint8_t*> prefetch_addr
    cdef uint8_t* eve = <uint8_t*> evict_addr
    cdef uint8_t seen[256]
    memset(seen, 0, 256)
    cdef uint32_t rn = rows * topk
    cdef uint32_t k, n = 0
    cdef int32_t e
    for k in range(rn):
        e = idx[k]
        if e < 0 or <uint32_t> e >= num_experts or seen[e]:
            continue
        seen[e] = 1
        if n >= max_entries:
            break
        (<uint32_t*> (pfe + n * 8))[0] = layer
        (<uint16_t*> (pfe + n * 8 + 4))[0] = <uint16_t> e
        (pfe + n * 8 + 6)[0] = 0                          # zone stable
        (pfe + n * 8 + 7)[0] = <uint8_t> (e % tp)         # static target
        (<uint32_t*> (eve + n * 8))[0] = layer
        (<uint16_t*> (eve + n * 8 + 4))[0] = 0xFFFF       # no-victim
        (eve + n * 8 + 6)[0] = <uint8_t> (e % tp)
        (eve + n * 8 + 7)[0] = 0
        n += 1
    if n == 0:
        return -4
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return -1
    _hdr4(d, 0x0C02, seq, 0, 0)                 # E_CMD_FETCH_AND_RUN_MOE
    (<uint32_t*> (d + 16))[0] = layer
    (<uint32_t*> (d + 20))[0] = num_seqs
    (<uint32_t*> (d + 24))[0] = n
    (<uint32_t*> (d + 28))[0] = timeout_us
    _publish(hdr)                               # have_evict_map=0 (zeroed)
    return <int> n


def wait_cmp(uintptr_t hdr, uintptr_t slots, uint64_t mask, uint32_t ssz,
             uint32_t expected, uint32_t dspark_seq, double timeout_s):
    """GIL-released completion wait.

    Spins (sched_yield between polls, GIL dropped) until a completion of
    interest arrives; CMP_CHECKPOINT completions are consumed + skipped
    inside the loop. Returns one of:
      ('ok', cmd_seq, status, cmd_type, layer, hbo, dbytes, top1, entropy)
      ('err', cmd_seq, msg)                       CMP_ERROR (any seq)
      ('dspark', <cmp_poll 11-tuple>)             cmd_seq == dspark_seq != 0
      ('other', <cmp_poll 11-tuple>)              unexpected type/seq
      ('timeout',)
    """
    cdef timespec t0, now
    clock_gettime(CLOCK_MONOTONIC, &t0)
    cdef double deadline = t0.tv_sec + t0.tv_nsec * 1e-9 + timeout_s
    cdef uint64_t cons, prod
    cdef uintptr_t src
    cdef uint32_t cmp_type, cmd_seq, gpu_idx, status
    cdef uint32_t cmd_type, layer_idx, hbo, dbytes
    cdef float top1, ent
    cdef bytes msg_b
    cdef int timed_out
    cdef uint32_t spin
    while True:
        timed_out = 0
        with nogil:
            spin = 0
            while True:
                cons = (<volatile uint64_t*> (hdr + 64))[0]
                prod = (<volatile uint64_t*> hdr)[0]
                if cons < prod:
                    break
                spin += 1
                if spin < _SPIN_ITERS:
                    if (spin & _SPIN_CLOCK_MASK) == 0:
                        clock_gettime(CLOCK_MONOTONIC, &now)
                        if now.tv_sec + now.tv_nsec * 1e-9 >= deadline:
                            timed_out = 1
                            break
                    ls_cpu_pause()
                    continue
                if (spin & _YIELD_CLOCK_MASK) == 0:
                    clock_gettime(CLOCK_MONOTONIC, &now)
                    if now.tv_sec + now.tv_nsec * 1e-9 >= deadline:
                        timed_out = 1
                        break
                sched_yield()
        if timed_out:
            return ("timeout",)
        src = slots + (cons & mask) * ssz
        cmp_type = (<uint32_t*> src)[0]
        cmd_seq = (<uint32_t*> (src + 4))[0]
        if cmp_type == _CMP_CHECKPOINT:
            (<volatile uint64_t*> (hdr + 64))[0] = cons + 1
            continue
        if cmp_type == _CMP_ERROR:
            msg_b = (<char*> (src + 20))[:80]
            (<volatile uint64_t*> (hdr + 64))[0] = cons + 1
            return ("err", cmd_seq,
                    msg_b.split(b"\0")[0].decode("utf-8", "replace"))
        gpu_idx = (<uint32_t*> (src + 8))[0]
        status = (<uint32_t*> (src + 12))[0]
        cmd_type = (<uint32_t*> (src + 16))[0]
        layer_idx = (<uint32_t*> (src + 20))[0]
        hbo = (<uint32_t*> (src + 24))[0]
        dbytes = (<uint32_t*> (src + 28))[0]
        top1 = (<float*> (src + 32))[0]
        ent = (<float*> (src + 36))[0]
        (<volatile uint64_t*> (hdr + 64))[0] = cons + 1
        if dspark_seq != 0 and cmd_seq == dspark_seq:
            return ("dspark", (cmp_type, cmd_seq, gpu_idx, status, cmd_type,
                               layer_idx, hbo, dbytes, top1, ent, ""))
        if cmp_type == expected:
            return ("ok", cmd_seq, status, cmd_type, layer_idx, hbo,
                    dbytes, top1, ent)
        return ("other", (cmp_type, cmd_seq, gpu_idx, status, cmd_type,
                          layer_idx, hbo, dbytes, top1, ent, ""))


def reef_route_fetch_from_export(uintptr_t hdr, uintptr_t slots,
                                 uint64_t mask, uint64_t count,
                                 uint32_t ssz, uint32_t seq_route,
                                 uint32_t seq_fetch,
                                 uintptr_t routing_hdr, uintptr_t routing_idx,
                                 uintptr_t prefetch_addr,
                                 uint32_t layer, uint32_t num_seqs,
                                 uint32_t num_experts, uint32_t timeout_us,
                                 uint32_t max_entries):
    """REEF-arm MoE layer issue (E_CMD_REEF_ROUTE contract): validate the
    routing export, dedup the union (first-occurrence order), write the
    sideband ExpertPrefetchEntry[] (zone 0, gpu_idx 0 — REWRITTEN by the
    daemon's ReefOrch service, which also fills the eviction map), then
    publish E_CMD_REEF_ROUTE followed by E_CMD_FETCH_AND_RUN_MOE with
    have_evict_map=1 back-to-back — ring order guarantees the daemon
    solves before the FETCH consumes the same sideband; the caller waits
    for BOTH CMP_COMPUTE_DONEs.

    Returns count >= 1, or negative: -1 ring full, -2 export row-count
    mismatch, -3 export layer mismatch, -4 empty union.
    """
    cdef uint32_t rows = (<uint32_t*> routing_hdr)[0]
    cdef uint32_t topk = (<uint32_t*> (routing_hdr + 4))[0]
    cdef uint32_t hdr_layer = (<uint32_t*> (routing_hdr + 8))[0]
    if rows != num_seqs:
        return -2
    if hdr_layer != layer:
        return -3
    cdef int32_t* idx = <int32_t*> routing_idx
    cdef uint8_t* pfe = <uint8_t*> prefetch_addr
    cdef uint8_t seen[256]
    memset(seen, 0, 256)
    cdef uint32_t rn = rows * topk
    cdef uint32_t k, n = 0
    cdef int32_t e
    for k in range(rn):
        e = idx[k]
        if e < 0 or <uint32_t> e >= num_experts or seen[e]:
            continue
        seen[e] = 1
        if n >= max_entries:
            break
        (<uint32_t*> (pfe + n * 8))[0] = layer
        (<uint16_t*> (pfe + n * 8 + 4))[0] = <uint16_t> e
        (pfe + n * 8 + 6)[0] = 0                          # zone stable
        (pfe + n * 8 + 7)[0] = 0                          # gpu: daemon fills
        n += 1
    if n == 0:
        return -4
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return -1
    _hdr4(d, 0x0C05, seq_route, 0, 0)           # E_CMD_REEF_ROUTE
    (<uint32_t*> (d + 16))[0] = layer
    (<uint32_t*> (d + 20))[0] = n
    _publish(hdr)
    d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return -1
    _hdr4(d, 0x0C02, seq_fetch, 0, 0)           # E_CMD_FETCH_AND_RUN_MOE
    (<uint32_t*> (d + 16))[0] = layer
    (<uint32_t*> (d + 20))[0] = num_seqs
    (<uint32_t*> (d + 24))[0] = n
    (<uint32_t*> (d + 28))[0] = timeout_us
    (<uint8_t*> (d + 38))[0] = 1                # have_evict_map
    _publish(hdr)
    return <int> n


def send_far_layer(uintptr_t hdr, uintptr_t slots, uint64_t mask,
                   uint64_t count, uint32_t ssz, uint32_t seq,
                   uint32_t layer, uint32_t num_seqs, uint32_t chunk_start,
                   uint32_t chunk_len, uint32_t timeout_us,
                   uint32_t is_prefill, uint32_t route_mode):
    """E_CMD_FAR_FORWARD_LAYER: one fused attention + routed FETCH layer
    (route_mode 0 = static e%num_gpus, 1 = ReefOrch service). The single
    CMP_COMPUTE_DONE's data_bytes carries the deduped entry count."""
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return False
    _hdr4(d, 0x0C06, seq, 0, 0)                 # E_CMD_FAR_FORWARD_LAYER
    (<uint32_t*> (d + 16))[0] = layer
    (<uint32_t*> (d + 20))[0] = num_seqs
    (<uint32_t*> (d + 24))[0] = chunk_start
    (<uint32_t*> (d + 28))[0] = chunk_len
    (<uint32_t*> (d + 32))[0] = timeout_us
    (<uint8_t*> (d + 36))[0] = <uint8_t> is_prefill
    (<uint8_t*> (d + 37))[0] = <uint8_t> route_mode
    _publish(hdr)
    return True


def send_seq_fork(uintptr_t hdr, uintptr_t slots, uint64_t mask,
                  uint64_t count, uint32_t ssz, uint32_t seq,
                  uint64_t src_seq_id, uint64_t dst_seq_id):
    """CMD_SEQ_FORK: CoW-fork src's KV (+ DSA indexer-K) pages into dst
    (the prefix-cache / speculative-fork primitive)."""
    cdef uintptr_t d = _claim(hdr, slots, mask, count, ssz)
    if d == 0:
        return False
    _hdr4(d, 0x0602, seq, 0, 0)                 # CMD_SEQ_FORK
    (<uint64_t*> (d + 16))[0] = src_seq_id
    (<uint64_t*> (d + 24))[0] = dst_seq_id
    _publish(hdr)
    return True

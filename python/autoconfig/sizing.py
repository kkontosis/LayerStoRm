"""Engine sizing formulas, transcribed to Python (AUTOCONFIG §5 K2).

Every function is a faithful transcription of the engine's own arithmetic;
each cites the C++ site (file:line at worktree HEAD 809d8fb7). Divergence
between this file and the engine is a bug HERE — the engine is authoritative.

Unit convention: every ``*_gb`` config field is GiB (2^30 bytes) —
config_resolver.h:41-43 ``vram_gb_to_bytes``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .modelshape import ModelShape

GIB = 1 << 30


def align256(n: int) -> int:
    """vram_allocator.cpp:228-232 align_region (kLayoutAlign 256)."""
    return (n + 255) & ~255


def align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


# ---------------------------------------------------------------- KV bytes

def kv_bytes_per_element(kv_quant: str) -> float:
    """vram_allocator.cpp:20-29."""
    return 1.0 if kv_quant in ("fp8_e4m3", "fp8_e5m2") else 2.0


def kv_bytes_per_token(shape: ModelShape, kv_quant: str, attention_backend: str) -> int:
    """vram_allocator.cpp:33-66. V4 has no uniform per-token size (throws)."""
    if shape.is_v4:
        raise ValueError("V4 has no uniform kv_bytes_per_token; use v4 tier sizing")
    bpe = kv_bytes_per_element(kv_quant)
    if shape.kv_lora_rank > 0:  # MLA
        if attention_backend == "turboquant_mla":
            return math.ceil(shape.kv_lora_rank * 0.5 + 2.0
                             + shape.qk_rope_head_dim * 2.0)
        base = shape.kv_lora_rank * bpe + shape.qk_rope_head_dim * 2.0
        if kv_quant in ("fp8_e4m3", "fp8_e5m2"):
            base += 4.0  # per-token f32 dequant scale
        return int(base)
    head_dim = shape.qk_nope_head_dim + shape.qk_rope_head_dim
    return int(2 * shape.num_key_value_heads * head_dim * bpe)


def kv_bytes_per_page(shape: ModelShape, kv_quant: str, attention_backend: str,
                      page_size_tokens: int) -> int:
    """vram_allocator.cpp:70-75."""
    return kv_bytes_per_token(shape, kv_quant, attention_backend) * page_size_tokens


# ---------------------------------------------------------------- Indexer-K

def indexer_k_bytes_per_token(shape: ModelShape, kv_quant: str) -> int:
    """vram_allocator.cpp:79-91: MQA K row + f32 absmax."""
    if shape.index_topk <= 0:
        return 0
    bpe = kv_bytes_per_element(kv_quant)
    return int(1 * shape.index_head_dim * bpe) + 4


def indexer_k_entries_per_page(shape: ModelShape, indexer_k_page_size_tokens: int) -> int:
    """vram_allocator.cpp:93-102."""
    if not shape.has_index_pool:
        return indexer_k_page_size_tokens
    return indexer_k_page_size_tokens // shape.index_kpool


def indexer_k_tail_bytes(shape: ModelShape) -> int:
    """vram_allocator.cpp:104-111: [2, kpool, head_dim] bf16 raw-K + gate."""
    if not shape.has_index_pool:
        return 0
    return 2 * shape.index_kpool * shape.index_head_dim * 2


def indexer_k_bytes_per_page(shape: ModelShape, kv_quant: str,
                             indexer_k_page_size_tokens: int) -> int:
    """vram_allocator.cpp:113-118."""
    return (indexer_k_bytes_per_token(shape, kv_quant)
            * indexer_k_entries_per_page(shape, indexer_k_page_size_tokens)
            + indexer_k_tail_bytes(shape))


def indexer_pages_per_seq(shape: ModelShape, max_sequence_length: int,
                          indexer_k_page_size_tokens: int,
                          dcp_indexer_mode: str, dcp_shard_factor: int) -> int:
    """vram_allocator.cpp:756-782 (+ :565-568 local-mode ceil-divide)."""
    pages = math.ceil(max_sequence_length / indexer_k_page_size_tokens)
    if dcp_indexer_mode == "local" and dcp_shard_factor > 1:
        pages = math.ceil(pages / dcp_shard_factor)
    return pages


# ---------------------------------------------------------------- S1 slabs

@dataclass(frozen=True)
class SlabGeometry:
    pages_per_slab: int
    slab_bytes: int


def slab_geometry(shape: ModelShape, kv_quant: str, attention_backend: str,
                  page_size_tokens: int, indexer_k_page_size_tokens: int) -> SlabGeometry | None:
    """vram_allocator.cpp:389-402: slab = ceil(idx_page/kv_page) kMain pages.
    None when the model has no DSA (no slabbing)."""
    if shape.index_topk <= 0:
        return None
    kvp = kv_bytes_per_page(shape, kv_quant, attention_backend, page_size_tokens)
    idx = indexer_k_bytes_per_page(shape, kv_quant, indexer_k_page_size_tokens)
    pages = math.ceil(idx / kvp)
    return SlabGeometry(pages_per_slab=pages, slab_bytes=pages * kvp)


# ---------------------------------------------------------------- KV pages

def auto_kv_pages(max_sequence_length: int, page_size_tokens: int,
                  kv_shard_factor: int, dcp_chunk_size: int,
                  max_concurrent_requests: int, kv_layers: int,
                  available_for_kv_bytes: int, bytes_per_page: int) -> int:
    """vram_allocator.cpp:281-329 compute_auto_kv_pages, verbatim.

    Sharded DCP KV: rank-0 worst case at chunk granularity (a plain
    needed/dcp floor-divide underprovisions rank 0). The VRAM cap is NOT
    divided in either mode."""
    pages_per_seq = math.ceil(max_sequence_length / page_size_tokens)
    if kv_shard_factor > 1:
        pages_per_chunk = max(1, dcp_chunk_size // page_size_tokens)
        cycle_pages = pages_per_chunk * kv_shard_factor
        full_cycles = pages_per_seq // cycle_pages
        rem = pages_per_seq % cycle_pages
        pages_per_seq = full_cycles * pages_per_chunk + min(rem, pages_per_chunk)
    needed = max_concurrent_requests * pages_per_seq * kv_layers
    if bytes_per_page > 0:
        needed = min(needed, available_for_kv_bytes // bytes_per_page)
    return max(needed, 0)


# ---------------------------------------------------------------- KDA state

def kda_per_layer_bytes(shape: ModelShape, tensor_parallelism: int) -> int:
    """vram_allocator.cpp:236-277: one linear layer's recurrent + conv-ring
    state for one rank. 0 when the model has no linear-attention layers."""
    if shape.num_linear_attention_layers <= 0:
        return 0
    lin = shape.linear_attn
    heads = lin.num_heads if lin else 64
    head_dim = lin.head_dim if lin else 128
    conv = lin.short_conv_kernel_size if lin else 4
    tp = max(1, tensor_parallelism)
    if heads % tp:
        raise ValueError(f"KDA num_heads {heads} % tp {tp} != 0")
    heads_per_rank = heads // tp
    recurrent = heads_per_rank * head_dim * head_dim * 4
    ring = (conv - 1 + 0) * heads_per_rank * head_dim * 4  # spec_columns=0 (GF3.11 seam)
    return recurrent + 3 * ring


def kda_slot_bytes(shape: ModelShape, tensor_parallelism: int) -> int:
    """vram_allocator.cpp:236-277 compute_kda_state_layout.
    0 when the model has no linear-attention layers."""
    n_lin = shape.num_linear_attention_layers
    if n_lin <= 0:
        return 0
    return align256(kda_per_layer_bytes(shape, tensor_parallelism) * n_lin)


def kda_mapped_demand_bytes(shape: ModelShape, tensor_parallelism: int,
                            slab_bytes: int) -> int:
    """Per-request KDA state under MAPPED slabs (TD-KDA-STATE-MAPPED-SLABS).

    The mapped pool claims WHOLE S1 slabs per layer, so the real charge is the
    slab-PADDED one, not ``kda_slot_bytes``: on GLM-5.3-Flash the engine reports
    ``34 linear layers x 17 slabs ... 150.2 MiB/request incl. slab padding vs
    145.6 MiB slot``. Sizing a request at the slot under-charges it by 4.6 MiB
    (TD-AUTOCONFIG-MAXSEQ-IGNORES-MAPPED-KDA)."""
    n_lin = shape.num_linear_attention_layers
    if n_lin <= 0 or slab_bytes <= 0:
        return 0
    per_layer = kda_per_layer_bytes(shape, tensor_parallelism)
    return n_lin * math.ceil(per_layer / slab_bytes) * slab_bytes


def kda_policy_slots(max_concurrent_requests: int, prefix_cache_enabled: bool,
                     prefix_cache_max_entries: int) -> int:
    """vram_allocator.cpp:652-655."""
    slots = max(1, max_concurrent_requests)
    if prefix_cache_enabled:
        slots += prefix_cache_max_entries
    return slots


# ------------------------------------------- single-request admissibility

def single_request_demand_pages(tokens: int, page_size_tokens: int,
                                kv_pool_layers: int,
                                dsa_computing_layers: int,
                                indexer_k_page_size_tokens: int,
                                pages_per_slab: int, state_pages: int,
                                kv_shard_factor: int = 1,
                                dcp_chunk_size: int = 0,
                                idx_shard_factor: int = 1) -> int:
    """One request's WHOLE-LIFE kMain-page demand at admission, mirroring
    the engine's admissible-ceiling arithmetic (vram_allocator.cpp,
    TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA): KV pages + the full-length
    indexer-K reservation (each indexer page costs one slab =
    ``pages_per_slab`` kMain pages on slabbed models; INV-DSA-RESERVE) +
    the mapped KDA state (``state_pages``, see
    :func:`kda_mapped_state_pages`; 0 under the carve off-path)."""
    per_seq = math.ceil(tokens / page_size_tokens)
    if kv_shard_factor > 1:
        per_chunk = max(1, dcp_chunk_size // page_size_tokens)
        cycle = per_chunk * kv_shard_factor
        per_seq = (per_seq // cycle) * per_chunk + min(per_seq % cycle,
                                                       per_chunk)
    demand = per_seq * kv_pool_layers
    if dsa_computing_layers > 0 and indexer_k_page_size_tokens > 0 \
            and pages_per_slab > 0:
        ip = math.ceil(tokens / indexer_k_page_size_tokens)
        if idx_shard_factor > 1:
            ip = math.ceil(ip / idx_shard_factor)
        demand += ip * dsa_computing_layers * pages_per_slab
    return demand + state_pages


def kda_mapped_state_pages(shape: ModelShape, tensor_parallelism: int,
                           slab_bytes: int, pages_per_slab: int) -> int:
    """One request's mapped KDA state, in kMain pages (whole-slab runs;
    0 when the model has no linear-attention layers or is unslabbed)."""
    n_lin = shape.num_linear_attention_layers
    if n_lin <= 0 or slab_bytes <= 0 or pages_per_slab <= 0:
        return 0
    per_layer = kda_per_layer_bytes(shape, tensor_parallelism)
    return n_lin * math.ceil(per_layer / slab_bytes) * pages_per_slab


def admissible_context_tokens(pool_pages: int, page_size_tokens: int,
                              kv_pool_layers: int, dsa_computing_layers: int,
                              indexer_k_page_size_tokens: int,
                              pages_per_slab: int, state_pages: int,
                              kv_shard_factor: int = 1,
                              dcp_chunk_size: int = 0,
                              idx_shard_factor: int = 1) -> int:
    """Largest T with single_request_demand_pages(T) <= pool_pages —
    the same binary search the engine runs at boot and prints as
    'admissible context (single request)' (vram_allocator.cpp,
    TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA)."""
    if pool_pages <= 0 or page_size_tokens <= 0 or kv_pool_layers <= 0:
        return 0

    def demand(t: int) -> int:
        return single_request_demand_pages(
            t, page_size_tokens, kv_pool_layers, dsa_computing_layers,
            indexer_k_page_size_tokens, pages_per_slab, state_pages,
            kv_shard_factor, dcp_chunk_size, idx_shard_factor)

    lo, hi = 0, pool_pages * page_size_tokens + page_size_tokens
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2
        if demand(mid) <= pool_pages:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------- KV tiering

@dataclass(frozen=True)
class KvTieringSizes:
    hot_slots: int
    cold_pool_pages: int          # per rank, after replica dedup
    host_pinned_bytes_per_rank: int  # cold pool only (staging heads excluded)
    device_row_cache_bytes: int   # allocated OUTSIDE the carve (out of margin)


def kv_tiering_sizes(shape: ModelShape, kv_quant: str, attention_backend: str,
                     page_size_tokens: int, hot_buffer_slots: int,
                     host_to_device_ratio: float, kv_layers: int,
                     dcp_size: int, replica_cold_dedup: bool = True,
                     kv_sharded: bool = True) -> KvTieringSizes:
    """kv_tiering_manager.cpp:120-137 (+ device side :189-245, host :262-296).

    host_to_device_ratio = host cold pool depth as a multiple of the
    per-layer hot buffer (rows->pages), per KV layer. The replica-cold-dedup
    divide applies ONLY under REPLICATED KV (INV-KVT-11 — one cold copy per
    demoted page across replicas); SHARDED KV runs per-rank shard tiering
    with a full cold pool per rank (VERIFIED against the 2026-08-30 derived-
    recipe boot: 80896 pages/rank at dcp=2 sharded, vs 40448 predicted by
    the pre-fix dedup-always transcription)."""
    hot = hot_buffer_slots if hot_buffer_slots > 0 else 2 * shape.index_topk
    per_layer_pages = math.ceil(host_to_device_ratio * hot / page_size_tokens)
    cold = per_layer_pages * kv_layers
    if replica_cold_dedup and dcp_size > 1 and not kv_sharded:
        cold = math.ceil(cold / dcp_size)
    row = kv_bytes_per_token(shape, kv_quant, attention_backend)
    block = row * page_size_tokens
    return KvTieringSizes(
        hot_slots=hot,
        cold_pool_pages=cold,
        host_pinned_bytes_per_rank=cold * block,
        device_row_cache_bytes=kv_layers * hot * row,
    )


# ---------------------------------------------------------------- V4 tiers

V4_FP8_ENTRY_BYTES = 1160    # vram_allocator.h:37-83
V4_TQ_ENTRY_BYTES = 644
V4_LOGICAL_BLOCK_TOKENS = 256
V4_CSA_RATIO = 4
V4_HCA_RATIO = 128


def v4_swa_pages_per_layer(attn_type: str, swa_page_tokens: int) -> int:
    """vram_allocator.cpp:215-222."""
    residual = {"csa": V4_CSA_RATIO - 1, "hca": V4_HCA_RATIO - 1}.get(attn_type, 0)
    tokens = swa_page_tokens + residual
    return math.ceil(tokens / swa_page_tokens) + 1


@dataclass(frozen=True)
class V4TierDemand:
    csa_pages: int
    hca_pages: int
    swa_pages: int
    lid_pages: int
    spec_pages: int
    csa_bytes_per_page: int
    hca_bytes_per_page: int
    swa_bytes_per_page: int
    lid_bytes_per_page: int


def v4_tier_demand(shape: ModelShape, attention_backend: str,
                   max_sequence_length: int, max_concurrent_requests: int,
                   prefix_holders: int, holder_tokens: int,
                   indexer_k_page_size_tokens: int,
                   speculation_pool_fraction: float) -> V4TierDemand:
    """vram_allocator.cpp:135-197 geometry + :938-996 page demand.
    CSA pages are NOT holder-scaled (refcount-shared); HCA/SWA/LID are
    (mutate-in-place rings own a complete side-tier set per holder)."""
    if not shape.is_v4:
        raise ValueError("v4_tier_demand on a non-V4 model")
    tq = attention_backend in ("csa_hca_tq", "csa_hca_tq_mix")
    csa_entry = V4_TQ_ENTRY_BYTES if tq else V4_FP8_ENTRY_BYTES
    hca_entry = V4_TQ_ENTRY_BYTES if attention_backend == "csa_hca_tq" else V4_FP8_ENTRY_BYTES
    swa_page_tokens = shape.sliding_window
    layers = list(shape.compress_ratios)
    att = ["csa" if r == 4 else "hca" if r == 128 else "swa" for r in layers]
    n_csa = att.count("csa")
    n_hca = att.count("hca")
    blocks_per_seq = math.ceil(max_sequence_length / V4_LOGICAL_BLOCK_TOKENS)
    holder_blocks = math.ceil(holder_tokens / V4_LOGICAL_BLOCK_TOKENS)
    mr = max_concurrent_requests
    csa_pages = mr * blocks_per_seq * n_csa
    hca_pages = mr * blocks_per_seq * n_hca + prefix_holders * holder_blocks * n_hca
    swa_per_seq = (sum(v4_swa_pages_per_layer(a, swa_page_tokens) for a in att)
                   + shape.num_nextn_predict_layers
                   * v4_swa_pages_per_layer("swa", swa_page_tokens))
    swa_pages = (mr + prefix_holders) * swa_per_seq
    lid_pages = (math.ceil(max_sequence_length / indexer_k_page_size_tokens) * n_csa * mr
                 + math.ceil(holder_tokens / indexer_k_page_size_tokens) * n_csa
                 * prefix_holders)
    spec_pages = int(csa_pages * speculation_pool_fraction)
    idx_entry = shape.index_head_dim + 4
    return V4TierDemand(
        csa_pages=csa_pages, hca_pages=hca_pages, swa_pages=swa_pages,
        lid_pages=lid_pages, spec_pages=spec_pages,
        csa_bytes_per_page=csa_entry * (V4_LOGICAL_BLOCK_TOKENS // V4_CSA_RATIO),
        hca_bytes_per_page=hca_entry * (V4_LOGICAL_BLOCK_TOKENS // V4_HCA_RATIO),
        swa_bytes_per_page=V4_FP8_ENTRY_BYTES * swa_page_tokens,  # SWA always FP8
        lid_bytes_per_page=(indexer_k_page_size_tokens // 4) * idx_entry,
    )


# ---------------------------------------------------------------- Experts

def gguf_packed_bytes_analytic(out_dim: int, in_dim: int, block_bytes: int,
                               block_elems: int) -> int:
    """gguf_kquant.cpp:196-215."""
    return out_dim * (in_dim // block_elems) * block_bytes


def nvfp4_projection_bytes(out_dim: int, in_dim: int) -> int:
    """nvfp4.cpp:25-46: fp4 pairs + UE8M0 scale/16 + 2 f32, align 128."""
    params = out_dim * in_dim
    raw = (params + 1) // 2 + (params + 15) // 16 + 8
    return align_up(raw, 128)


def expert_slots_in_zone(zone_bytes: int, bytes_per_expert: int) -> int:
    """expert_cache.cpp:29-39: slots = floor(zone / bytes_per_expert)."""
    return zone_bytes // bytes_per_expert if bytes_per_expert > 0 else 0


def host_arena_total_bytes(slot_bytes: int, n_routed_experts: int,
                           num_moe_layers: int) -> int:
    """engine.cpp:1681-1687."""
    return slot_bytes * n_routed_experts * num_moe_layers


# ---------------------------------------------------------------- dspark

def dspark_weight_bytes_safetensors(path: str, quant: str, num_ranks: int) -> int:
    """Approximate per-rank device bytes for a safetensors draft checkpoint,
    from the header only (dspark_loader.cpp:410-445 walks the same header).

    Sharding: 2D matmul weights split across ranks; 1D (norms/bias) and
    embedding/lm_head rows replicated conservatively. Quant applies to
    matmul weights only; norms/embeddings stay bf16. This is a sizing
    ESTIMATE for the fit decision, not the loader's exact arithmetic —
    flagged as such in explanations."""
    import json
    import struct as _struct
    with open(path, "rb") as f:
        (hlen,) = _struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta.get("shape", [])
        numel = 1
        for d in shape:
            numel *= int(d)
        two_d = len(shape) >= 2
        keep_bf16 = ("embed" in name) or ("head" in name) or ("norm" in name)
        if two_d and keep_bf16:
            # embeddings / lm_head / norms stay bf16 under weight quant
            total += align_up(numel * 2 // max(1, num_ranks), 256)
            continue
        if two_d and quant == "nvfp4":
            per = (numel + 1) // 2 + (numel + 15) // 16 + 8
            per = align_up(per, 128) // max(1, num_ranks)
        elif two_d and quant == "fp8_e4m3":
            per = numel // max(1, num_ranks)
        else:  # bf16, or non-shardable small tensors
            per = numel * 2
            if two_d:
                per //= max(1, num_ranks)
        total += align_up(per, 256)
    return total


def dspark_scratch_bytes(shape: ModelShape, draft_layers: int,
                         draft_kv_heads: int, draft_head_dim: int,
                         draft_hidden: int, draft_vocab: int,
                         ctx_cap_tokens: int, aux_rows: int,
                         block_size: int, num_ranks: int, rank: int) -> int:
    """dspark_runtime.cpp:158-238 scratch_layout — dominant terms only
    (kv_arena is the context-scaling one: L * 2 * ctx_cap * kv_dim * 2)."""
    kv_dim = draft_kv_heads * draft_head_dim // max(1, num_ranks)
    total = draft_layers * 2 * ctx_cap_tokens * kv_dim * 2  # kv_arena
    if rank == 0:
        total += aux_rows * 5 * draft_hidden * 2            # aux_stage (n_aux=5)
        total += max(aux_rows, ctx_cap_tokens) * draft_hidden * 2  # ctx_hidden
        total += 2 * block_size * draft_vocab * 4           # base+corrected logits
    total += 16 * block_size * draft_hidden * 2             # ~15 block-sized buffers
    return total


# ------------------------------------- runtime device scratch (P-31 gap 2)
#
# TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH: the VramAllocator claims
# ``vram_gb - margin`` as ONE physical allocation, so every allocation below
# lives OUTSIDE the block and is funded only by the margin + the physical
# slack above the declared vram_gb.  Two phases matter, split by WHERE the
# MoE-big elastic fit check runs (CommandDispatcher ctor,
# command_dispatcher.cpp:386-690, engine.cpp:4284):
#   PREFIT    — allocated before the fit check samples cudaMemGetInfo
#               (attention staging, RoPE tables, DcpExecutor indexer/KDA
#               scratch, hidden-state pair buffers);
#   POSTCHECK — allocated after it (block tables, logits scratch, the KVT
#               union staging, the MoE persistent set) — the per-class
#               ``compute.moe_big_fit_headroom_mb`` is what reserves room
#               for these, so at long context the HEADROOM must be derived
#               too or the boot dies at the block-table alloc
#               (command_dispatcher.cpp:1363).

MIB = 1 << 20


@dataclass(frozen=True)
class RuntimeScratch:
    """Per-GPU runtime allocations outside the VramAllocator block, MiB."""
    prefit_bytes: int
    postcheck_bytes: int
    terms: tuple  # (name, bytes, phase, site) — for the explain sheet

    @property
    def prefit_mib(self) -> int:
        return self.prefit_bytes // MIB

    @property
    def postcheck_mib(self) -> int:
        return self.postcheck_bytes // MIB


def _hc_mult(shape: ModelShape) -> int:
    return int((shape.raw or {}).get("hc_mult", 1) or 1)


def attention_staging_bytes(shape: ModelShape, attention_backend: str,
                            max_seq: int) -> int:
    """Prefill KV staging + indices, per TP rank, boot-time.

    snapmla: snapmla_sm120_attention_device.cpp:480-486 (engine.cpp:919);
    turboquant_mla k_out: tq_sm120_attention_device.cpp:139-142
    (engine.cpp:893).  Identical byte formula, mutually exclusive."""
    if attention_backend not in ("snapmla", "turboquant_mla"):
        return 0
    d = shape.kv_lora_rank + shape.qk_rope_head_dim
    return max_seq * d * 2 + max_seq * 4


def rope_table_bytes(shape: ModelShape, max_seq: int) -> int:
    """dcp_executor.cpp:520-533; x2 on V4 (base + compress tables).
    glm5_next is NoPE (qk_rope_head_dim=0) -> 0."""
    b = max_seq * shape.qk_rope_head_dim * 4
    return b * 2 if shape.is_v4 else b


def index_topk_rows(shape: ModelShape) -> int:
    """kv_tiering_manager.h:205-207 / dcp_executor.h:636-638."""
    if shape.index_topk <= 0:
        return 0
    return shape.index_topk + (shape.index_kpool - 1
                               if shape.index_kpool > 1 else 0)


def indexer_executor_scratch_bytes(shape: ModelShape, max_seq: int,
                                   max_batch: int, superchunk: int,
                                   indexer_k_page_size_tokens: int,
                                   sparse_prefill: bool = True) -> int:
    """DcpExecutor indexer buffers, per TP rank, boot-time
    (dcp_executor.cpp:637-716). Dominant terms only; sub-MiB constants
    folded into the fixed overhead anchor."""
    if not shape.has_dsa:
        return 0
    ct = max(max_seq, 4096)
    rows = max(max_batch, superchunk, 1)
    tr = index_topk_rows(shape)
    total = ct * 4                              # indexer_scores_
    total += ct * 4                             # indexer_block_endpoints_
    if sparse_prefill:
        total += max(ct, max_batch * 16384) * 4  # indexer_scores_batched_
        total += max_batch * math.ceil(
            ct / max(indexer_k_page_size_tokens, 64)) * 8  # page table
    total += rows * tr * 4                      # sparse_indices_dev_
    total += rows * 4                           # topk_lengths_dev_
    if shape.index_kpool > 1:
        total += rows * max(shape.index_topk // shape.index_kpool, 1) * 4
    total += (max_batch if sparse_prefill else 1) * tr * 4  # topk scores
    return total


def kda_executor_scratch_bytes(shape: ModelShape, tensor_parallelism: int,
                               max_batch: int, superchunk: int) -> int:
    """DcpExecutor KDA chunk scratch, per TP rank, boot-time
    (dcp_executor.cpp:816-850; kda_chunk.cu:446-451, kC=64, kD=128)."""
    if shape.num_linear_attention_layers <= 0:
        return 0
    lin = shape.linear_attn
    heads = (lin.num_heads if lin else 64) // max(1, tensor_parallelism)
    rows = min(max(max_batch, superchunk, 1), 512)
    ck = heads * 128
    ws = math.ceil(rows / 64) * heads * (6 * 64 * 128 + 64 * 64 + 64) * 4
    return rows * ck * 28 + rows * heads * 2 + rows * 512 + ws


def pair_buffer_bytes(shape: ModelShape, max_batch: int, superchunk: int,
                      expert_only: bool) -> int:
    """Hidden-state pair / fused-MoE hidden buffers, boot-time.
    TP GPUs: attn+moe pair, each rows x hidden x hc_mult x 2
    (engine.cpp:1067-1078); expert-only GPUs: one fused buffer
    rows x hidden x 2 (engine.cpp:1136-1158)."""
    rows = max(max_batch, superchunk, 1)
    if expert_only:
        return rows * shape.hidden_size * 2
    return 2 * rows * shape.hidden_size * _hc_mult(shape) * 2


def block_tables_bytes(shape: ModelShape, max_seq: int, max_batch: int,
                       page_size_tokens: int) -> int:
    """dev_block_tables, per TP rank, POSTCHECK
    (command_dispatcher.cpp:1359): (layers + nextn) x B x pages x 4."""
    layers = shape.num_hidden_layers + shape.num_nextn_predict_layers
    return layers * max_batch * math.ceil(max_seq / page_size_tokens) * 4


def logits_scratch_bytes(shape: ModelShape, max_batch: int, tp: int,
                         rank0: bool) -> int:
    """command_dispatcher.cpp:1181-1255, per TP rank, POSTCHECK."""
    b = max(max_batch, 1)
    total = b * shape.vocab_size * 4              # logits_scratch_
    total += b * shape.hidden_size * 2            # output_norm_scratch_
    if tp >= 2:
        total += b * (shape.vocab_size // tp) * 4  # partial_logits_
        if rank0 and max_batch > 1:
            total += max_batch * shape.vocab_size * 4  # gather (rank 0)
    return total


def kvt_union_staging_bytes(shape: ModelShape, kv_quant: str,
                            attention_backend: str, max_seq: int,
                            max_batch: int, page_size_tokens: int,
                            kv_sharded: bool, dcp_size: int,
                            dcp_chunk_size: int) -> int:
    """KVT union-materialize staging, per KV rank, POSTCHECK
    (command_dispatcher.cpp:1718-1741 + kv_tiering_manager.cpp:236-252).
    Zero when tiering (or tiered sparse prefill) is off."""
    tr = index_topk_rows(shape)
    if tr <= 0 or max_batch <= 1:
        return 0
    local_rows = max_seq
    if kv_sharded and dcp_size >= 2:
        local_rows = max_seq // dcp_size + max(1, dcp_chunk_size)
    rows = min(max_batch * tr, local_rows)
    row = kv_bytes_per_token(shape, kv_quant, attention_backend)
    u_pages = math.ceil(rows / page_size_tokens)
    total = u_pages * page_size_tokens * row      # umat.scratch
    total += rows * row                           # cold_incoming
    total += rows * 8                             # dev_src_ptrs
    total += max_batch * tr * 4                   # dev_uidx
    total += max_batch * u_pages * 4              # dev_union_bt
    return total


def moe_persistent_bytes(shape: ModelShape, batch_capacity_tokens: int) -> int:
    """MoE persistent set, per expert device, POSTCHECK
    (command_dispatcher.cpp:438-449 persist_bytes)."""
    bq = max(1, batch_capacity_tokens)
    exp = bq * shape.num_experts_per_tok
    total = bq * shape.n_routed_experts * 4       # router_logits
    total += exp * 8                              # topk weights + indices
    total += 3 * bq * shape.hidden_size * 2       # moe/normalized/shared out
    return total


def ep_xtp_staging_bytes(shape: ModelShape, batch_capacity_tokens: int,
                         tp: int, det_ep_combine: bool = True) -> int:
    """Canonical per-slot combine staging, per TP rank, POSTCHECK
    (command_dispatcher.cpp:1094)."""
    if tp < 2 or not det_ep_combine:
        return 0
    return (batch_capacity_tokens * shape.num_experts_per_tok
            * shape.hidden_size * 2)


def moe_big_transient_bytes(shape: ModelShape, chunk_tokens: int, tp: int,
                            gguf_weights: bool,
                            det_ep_combine: bool = True) -> int:
    """The MoE-big transient scratch at one chunk size — the engine's
    ``transient_bytes`` verbatim (command_dispatcher.cpp:409-437).
    ~0.48 MiB/token on glm5_next tp2/EP4 (schema x-devDoc: 246 MiB @512)."""
    h = shape.hidden_size
    i = shape.moe_intermediate_size
    e = shape.n_routed_experts
    t = shape.num_experts_per_tok
    i_dense = (shape.intermediate_size // max(1, tp)
               if (shape.first_k_dense_replace > 0
                   and shape.intermediate_size > 0) else 0)
    i_local = i // max(1, tp)
    max_k = max(h, i, i_dense)
    bt = chunk_tokens
    exp = bt * t
    total = exp * h * 2                                  # permuted_input
    total += exp * 2 * 4                                 # maps
    total += max(exp * 2 * i, bt * 2 * i_dense) * 2      # gate_up_output
    total += 2 * max(exp * i, bt * i_dense) * 2          # activation (+split)
    total += exp * h * 2                                 # expert_output
    total += exp * h * 2                                 # moe_wave_accum
    total += exp * 8 * 4                                 # permute workspace
    total += exp * max_k                                 # quant_act
    total += exp * math.ceil(max_k / 128) * 4            # quant_scale
    if gguf_weights:
        total += exp * (max_k // 32) * 36                # Q8_1 workspace
        total += 2 * (exp // 64 + e + 1) * 4
    if det_ep_combine:
        total += bt * t * h * 2                          # per-slot payload
    total += bt * 3 * i_local * 2                        # shared expert
    return total


def runtime_scratch(shape: ModelShape, kv_quant: str, attention_backend: str,
                    *, max_seq: int, max_batch: int, superchunk: int,
                    tp: int, rank0: bool, tiering_on: bool, kv_sharded: bool,
                    page_size_tokens: int, indexer_k_page_size_tokens: int,
                    dcp_chunk_size: int,
                    det_ep_combine: bool = True) -> RuntimeScratch:
    """One TP/attention rank's runtime allocations outside the block,
    split at the MoE-big fit check (see module note above)."""
    terms = []

    def add(name, b, phase, site):
        if b > 0:
            terms.append((name, b, phase, site))
        return b

    pre = add("attention prefill staging",
              attention_staging_bytes(shape, attention_backend, max_seq),
              "prefit", "engine.cpp:893/:919")
    pre += add("rope cos/sin table", rope_table_bytes(shape, max_seq),
               "prefit", "dcp_executor.cpp:520-533")
    pre += add("indexer executor scratch",
               indexer_executor_scratch_bytes(
                   shape, max_seq, max_batch, superchunk,
                   indexer_k_page_size_tokens),
               "prefit", "dcp_executor.cpp:637-716")
    pre += add("kda chunk scratch",
               kda_executor_scratch_bytes(shape, tp, max_batch, superchunk),
               "prefit", "dcp_executor.cpp:816-850")
    pre += add("hidden-state pair buffers",
               pair_buffer_bytes(shape, max_batch, superchunk,
                                 expert_only=False),
               "prefit", "engine.cpp:1067-1078")

    post = add("dev_block_tables",
               block_tables_bytes(shape, max_seq, max_batch,
                                  page_size_tokens),
               "postcheck", "command_dispatcher.cpp:1359")
    post += add("logits scratch",
                logits_scratch_bytes(shape, max_batch, tp, rank0),
                "postcheck", "command_dispatcher.cpp:1181-1255")
    if tiering_on:
        post += add("kvt union staging",
                    kvt_union_staging_bytes(
                        shape, kv_quant, attention_backend, max_seq,
                        max_batch, page_size_tokens, kv_sharded, tp,
                        dcp_chunk_size),
                    "postcheck", "kv_tiering_manager.cpp:236-252")
    post += add("moe persistent set",
                moe_persistent_bytes(shape, max(superchunk, max_batch)),
                "postcheck", "command_dispatcher.cpp:438-449")
    post += add("ep-xtp combine staging",
                ep_xtp_staging_bytes(shape, max(superchunk, max_batch), tp,
                                     det_ep_combine),
                "postcheck", "command_dispatcher.cpp:1094")
    return RuntimeScratch(prefit_bytes=pre, postcheck_bytes=post,
                          terms=tuple(terms))

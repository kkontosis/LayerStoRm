#include "parallelism/kv_bv_dequant_pool.h"

#include "core/attention_device.h"
#include "compute/kernels/smxx/quant/kv_bv_extract_dequant.h"
#include "compute/stream_manager.h"
#include "model/quantization/gguf_kquant.h"  // GG-7b: block_values / type_name (host-only)
#include "parallelism/dcp_executor.h"  // AttentionLayerWeights

#include <spdlog/spdlog.h>
#include <stdexcept>

namespace layerstorm::parallelism {

// GG-7b: the V side of kv_b_proj needs an extract+dequant into the BF16 pool
// whenever the weight is FP8 *or* a packed GGUF k-quant. A GGUF kv_b has no FP8
// scales (kv_b_proj_is_fp8 == false), so gating on FP8 alone would wrongly hand
// out the BF16-direct pointer into packed GGUF bytes (TD-GG7-KVB-V-PATH-GGUF).
static inline bool kv_bv_needs_dequant(const AttentionLayerWeights& w) {
    return w.kv_b_proj_is_fp8 || w.kv_b_is_gguf;
}

bool KvBvDequantPool::needs_dequant(const AttentionLayerWeights& w) {
    return kv_bv_needs_dequant(w);
}

// ── Constructor / Destructor ────────────────────────────────────────────────

KvBvDequantPool::KvBvDequantPool(Options opts) : opts_(std::move(opts)) {
    if (static_cast<int>(opts_.attention_devices.size()) < opts_.dcp_size) {
        throw std::runtime_error(
            "KvBvDequantPool: attention_devices.size() ("
            + std::to_string(opts_.attention_devices.size())
            + ") < dcp_size (" + std::to_string(opts_.dcp_size) + ")");
    }

    const int HL = opts_.num_heads_local;
    const int V  = opts_.v_head_dim;
    const int D_c = opts_.kv_lora_rank;

    // Each slot holds [HL, V, D_c] BF16 per rank.
    slot_bytes_ = static_cast<size_t>(HL) * V * D_c * 2;

    slots_.resize(opts_.num_slots);

    for (int s = 0; s < opts_.num_slots; ++s) {
        auto& slot = slots_[s];
        slot.buffers.resize(opts_.dcp_size, nullptr);
        slot.events.resize(opts_.dcp_size, nullptr);

        for (int r = 0; r < opts_.dcp_size; ++r) {
            auto* attn = opts_.attention_devices[r];
            attn->set_device();
            slot.buffers[r] = attn->device_alloc(slot_bytes_);
            if (!slot.buffers[r]) {
                throw std::runtime_error(
                    "KvBvDequantPool: device_alloc failed for slot " +
                    std::to_string(s) + " rank " + std::to_string(r));
            }
        }

        // Create events via StreamManager (one per rank for async tracking).
        if (opts_.stream_manager) {
            for (int r = 0; r < opts_.dcp_size; ++r) {
                slot.events[r] = opts_.stream_manager->create_event(
                    opts_.attention_devices[r]->gpu().position);
            }
        }
    }

    if (opts_.num_slots == 0) {
        // P-29 step 21 (TD-KVBV-DEQUANT-POOL-INERT-GLM5N): sized to zero by
        // DcpExecutor because no layer's kv_b_proj is FP8/GGUF-packed —
        // acquire() is always BF16-direct; no device buffers are held.
        spdlog::info("KvBvDequantPool: 0 slots (all kv_b_proj BF16-direct) — "
                     "{} MB/rank reclaimed vs the 5-slot carve",
                     5 * slot_bytes_ / (1024 * 1024));
    } else {
        spdlog::info("KvBvDequantPool: {} slots x {} ranks, {} MB each",
                     opts_.num_slots, opts_.dcp_size,
                     slot_bytes_ / (1024 * 1024));
    }
}

KvBvDequantPool::~KvBvDequantPool() {
    for (int s = 0; s < static_cast<int>(slots_.size()); ++s) {
        auto& slot = slots_[s];
        for (int r = 0; r < opts_.dcp_size; ++r) {
            if (slot.buffers[r]) {
                opts_.attention_devices[r]->device_free(slot.buffers[r]);
            }
            if (slot.events[r] && opts_.stream_manager) {
                opts_.stream_manager->destroy_event(
                    slot.events[r],
                    opts_.attention_devices[r]->gpu().position);
            }
        }
    }
}

// ── Slot lookup ─────────────────────────────────────────────────────────────

int KvBvDequantPool::find_slot(int layer_idx) const {
    for (int s = 0; s < static_cast<int>(slots_.size()); ++s) {
        if (slots_[s].layer_idx == layer_idx
            && slots_[s].state != SlotState::kIdle)
            return s;
    }
    return -1;
}

int KvBvDequantPool::find_evictable_slot() const {
    // Prefer kIdle slots first.
    for (int s = kPermanentLayers; s < static_cast<int>(slots_.size()); ++s) {
        if (slots_[s].state == SlotState::kIdle) return s;
    }
    // Then kReady non-permanent slots (oldest = lowest rotating index).
    for (int s = kPermanentLayers; s < static_cast<int>(slots_.size()); ++s) {
        if (slots_[s].state == SlotState::kReady) return s;
    }
    return -1;
}

// ── Dequant launch ──────────────────────────────────────────────────────────

void KvBvDequantPool::launch_dequant_rank(int slot_idx, int rank,
                                           const AttentionLayerWeights& w,
                                           void* stream) {
    compute::KvBvExtractDequantParams p{};
    p.num_heads_local = opts_.num_heads_local;
    p.qk_nope_head_dim = opts_.qk_nope_head_dim;
    p.v_head_dim = opts_.v_head_dim;
    p.kv_lora_rank = opts_.kv_lora_rank;
    p.kv_b_proj = w.kv_b_proj;
    p.output = slots_[slot_idx].buffers[rank];

    if (w.kv_b_is_gguf) {
        // GG-7b: packed GGUF V side → per-element dequant branch. GGUF scales
        // are in-block (pass none); the int gguf_type selects the k-quant
        // policy. Shared L%QK guard: a per-V-row must be whole super-blocks on
        // the kv_lora axis (same constraint q_absorb enforces for W_UK).
        const int qk = model::gguf::block_values(w.kv_b_gguf_type);
        if (opts_.kv_lora_rank % qk != 0) {
            throw std::runtime_error(
                "KvBvDequantPool: GGUF kv_b V-extract requires kv_lora_rank ("
                + std::to_string(opts_.kv_lora_rank)
                + ") % QK (" + std::to_string(qk) + ") == 0 for type "
                + std::string(model::gguf::type_name(w.kv_b_gguf_type))
                + "; the per-element dequant cannot span a partial super-block "
                  "on the kv_lora axis (GG-7b)");
        }
        p.scales = nullptr;
        p.gguf_type = static_cast<int>(w.kv_b_gguf_type);
    } else {
        p.scales = w.kv_b_proj_scales;
        p.gguf_type = -1;
    }

    opts_.attention_devices[rank]->kv_bv_extract_dequant(p, stream);
}

// ── Prime (init-time synchronous dequant for permanent slots) ───────────────

void KvBvDequantPool::prime(int layer_idx,
                             const AttentionLayerWeights* const* weights_by_rank) {
    if (layer_idx >= kPermanentLayers) {
        spdlog::warn("KvBvDequantPool::prime: layer {} >= kPermanentLayers({}), "
                     "ignoring", layer_idx, kPermanentLayers);
        return;
    }

    // P-29 step 21: a zero-slot (inert) pool has nothing to prime. A weight
    // that needs dequant reaching a zero-slot pool is a sizing bug — fail
    // loudly rather than serve garbage.
    if (layer_idx >= static_cast<int>(slots_.size())) {
        if (kv_bv_needs_dequant(*weights_by_rank[0])) {
            throw std::runtime_error(
                "KvBvDequantPool::prime: layer " + std::to_string(layer_idx)
                + " needs dequant but the pool was sized to "
                + std::to_string(slots_.size())
                + " slot(s) — DcpExecutor::set_layer_weights mis-sized the "
                  "pool (TD-KVBV-DEQUANT-POOL-INERT-GLM5N)");
        }
        return;
    }

    auto& slot = slots_[layer_idx];
    slot.layer_idx = layer_idx;

    // Check if BF16 (no dequant needed — permanent slot still reserved but
    // unused). FP8 and GGUF both route through the dequant slots (GG-7b).
    if (!kv_bv_needs_dequant(*weights_by_rank[0])) {
        slot.state = SlotState::kReady;
        return;
    }

    for (int r = 0; r < opts_.dcp_size; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();
        launch_dequant_rank(layer_idx, r, *weights_by_rank[r], nullptr);
        attn->device_sync();  // Synchronous for init.
    }
    slot.state = SlotState::kReady;
    // Synchronous + device_sync: every rank is dequanted and globally ordered.
    slot.launched_ranks = slot.synced_ranks =
        (opts_.dcp_size >= 32) ? ~0u : ((1u << opts_.dcp_size) - 1u);
}

// ── Schedule predictive dequant ─────────────────────────────────────────────

void KvBvDequantPool::schedule_dequant(
    int layer_idx,
    const AttentionLayerWeights* const* weights_by_rank) {

    // Already have a slot? Skip.
    if (find_slot(layer_idx) >= 0) return;

    // BF16 weight? No dequant needed. FP8 and GGUF both dequant (GG-7b).
    if (!kv_bv_needs_dequant(*weights_by_rank[0])) return;

    int slot_idx = find_evictable_slot();
    if (slot_idx < 0) return;  // All slots busy (shouldn't happen with 5 slots).

    auto& slot = slots_[slot_idx];
    slot.layer_idx = layer_idx;
    slot.state = SlotState::kDequanting;
    // All ranks are launched below on their kAsyncDequant streams; no
    // consumer stream is ordered after them yet (acquire() inserts the
    // per-rank stream_wait_event).
    slot.launched_ranks =
        (opts_.dcp_size >= 32) ? ~0u : ((1u << opts_.dcp_size) - 1u);
    slot.synced_ranks = 0;

    for (int r = 0; r < opts_.dcp_size; ++r) {
        auto* attn = opts_.attention_devices[r];
        attn->set_device();

        // Use the async dequant stream.
        void* dequant_stream = opts_.stream_manager
            ? opts_.stream_manager->stream(
                  attn->gpu().position, compute::StreamId::kAsyncDequant)
            : nullptr;

        launch_dequant_rank(slot_idx, r, *weights_by_rank[r], dequant_stream);

        // Record event on dequant stream for later synchronization.
        if (opts_.stream_manager && slot.events[r]) {
            opts_.stream_manager->record_event(
                slot.events[r], attn->gpu().position,
                compute::StreamId::kAsyncDequant);
        }
    }
}

// ── Acquire ─────────────────────────────────────────────────────────────────

KvBvDequantPool::AcquireResult KvBvDequantPool::acquire(
    int layer_idx, int rank,
    const AttentionLayerWeights& weights,
    void* attn_stream) {

    const int HL = opts_.num_heads_local;
    const int P  = opts_.qk_nope_head_dim;
    const int V  = opts_.v_head_dim;
    const int D_c = opts_.kv_lora_rank;

    // BF16 weight: return direct pointer into pinned kv_b_proj.
    // V rows for head h start at row h*(P+V)+P, col 0.
    // Byte offset to first V row: P * D_c * sizeof(BF16).
    // GG-7b: GGUF kv_b is NOT BF16-direct — its packed bytes must be dequanted
    // through a slot (kv_bv_needs_dequant), else the value GEMM reads packed
    // GGUF bytes as BF16 (the TD-GG7-KVB-V-PATH-GGUF bug).
    if (!kv_bv_needs_dequant(weights)) {
        const auto* base = static_cast<const char*>(weights.kv_b_proj);
        const void* v_ptr = base + static_cast<size_t>(P) * D_c * 2;
        return {v_ptr,
                static_cast<int64_t>(P + V) * D_c,  // stride: skip K rows between heads
                true};
    }

    // FP8 or GGUF weight: check for existing slot.
    int slot_idx = find_slot(layer_idx);

    const uint32_t rank_bit = 1u << rank;
    const uint32_t all_ranks =
        (opts_.dcp_size >= 32) ? ~0u : ((1u << opts_.dcp_size) - 1u);

    if (slot_idx >= 0) {
        // P-29 step 21 (TD-KVBV-DEQUANT-POOL-INERT-GLM5N latent bug fix):
        // readiness is PER RANK, not per slot. The old code (a) marked the
        // slot kReady after the synchronous fallback dequanted only the
        // acquiring rank — every other rank then received its buffer with
        // the PREVIOUS layer's weights still in it; and (b) let the first
        // acquiring rank's event wait flip kDequanting→kReady, so later
        // ranks skipped the stream_wait_event on their OWN async dequant.
        auto& slot = slots_[slot_idx];
        if (!(slot.launched_ranks & rank_bit)) {
            // This rank was never dequanted (sync-fallback slot): dequant it
            // now, synchronously on its consumer stream.
            launch_dequant_rank(slot_idx, rank, weights, attn_stream);
            slot.launched_ranks |= rank_bit;
            slot.synced_ranks |= rank_bit;
        } else if (!(slot.synced_ranks & rank_bit)) {
            // Async dequant launched on this rank's kAsyncDequant stream:
            // order this rank's consumer stream after ITS event.
            if (opts_.stream_manager && slot.events[rank]) {
                int gpu_pos = opts_.attention_devices[rank]->gpu().position;
                opts_.stream_manager->wait_event(
                    gpu_pos, compute::StreamId::kAttention, slot.events[rank]);
            }
            slot.synced_ranks |= rank_bit;
        }
        // Evictable (kReady) only once every rank is launched AND ordered.
        slot.state = ((slot.launched_ranks & all_ranks) == all_ranks
                      && (slot.synced_ranks & all_ranks) == all_ranks)
                         ? SlotState::kReady
                         : SlotState::kDequanting;
        return {slot.buffers[rank],
                static_cast<int64_t>(V) * D_c,  // contiguous
                false};
    }

    // No slot — synchronous dequant on attention stream.
    slot_idx = find_evictable_slot();
    if (slot_idx < 0) {
        throw std::runtime_error(
            "KvBvDequantPool: no evictable slot for layer " +
            std::to_string(layer_idx) + " (pool has "
            + std::to_string(slots_.size())
            + " slot(s); 0 means DcpExecutor sized the pool inert for an "
              "all-BF16 kv_b model — a dequant-needing weight here is a "
              "sizing bug, TD-KVBV-DEQUANT-POOL-INERT-GLM5N)");
    }

    auto& slot = slots_[slot_idx];
    slot.layer_idx = layer_idx;
    launch_dequant_rank(slot_idx, rank, weights, attn_stream);
    // Only THIS rank is dequanted; other ranks dequant on their own
    // acquire() (see above). kReady only when every rank is covered.
    slot.launched_ranks = rank_bit;
    slot.synced_ranks = rank_bit;
    slot.state = (rank_bit == all_ranks) ? SlotState::kReady
                                         : SlotState::kDequanting;

    return {slot.buffers[rank],
            static_cast<int64_t>(V) * D_c,
            false};
}

// ── Release ─────────────────────────────────────────────────────────────────

void KvBvDequantPool::release(int layer_idx) {
    // Permanent slots never freed.
    if (layer_idx < kPermanentLayers) return;

    int slot_idx = find_slot(layer_idx);
    if (slot_idx >= 0) {
        // Keep as kReady (data still valid) — slot becomes evictable.
        // The actual eviction happens in find_evictable_slot when needed.
        slots_[slot_idx].state = SlotState::kReady;
    }
}

}  // namespace layerstorm::parallelism

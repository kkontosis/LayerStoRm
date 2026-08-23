#pragma once

//==============================================================================
// KvBvDequantPool — 5-slot BF16 scratch pool for kv_b_v projection weights.
//
// Manages per-layer FP8→BF16 dequantization of the V portion from kv_b_proj
// with predictive pre-dequanting:
//   - Slots 0,1: permanently hold layers 0,1 (always warm)
//   - Slots 2,3,4: rotate for current + lookahead layers
//
// For BF16 weights (no FP8), acquire() returns a direct pointer into the
// pinned kv_b_proj with appropriate stride — no slot consumed.
//
// Design rationale (TD-87a multi-engine analysis):
//   Per-layer dequant avoids ~488 MB/GPU persistent BF16 copy (vllm/sglang
//   approach). Matches DeepSeek V3 reference (dequant per forward call) but
//   with a scratch pool to amortize across layers. TRT-LLM keeps FP8 with
//   block-scaled BMM; we can't do that with cuBLAS (no blockwise scale
//   support in cublasGemmStridedBatchedEx), hence dequant-then-BF16-BMM.
//   Total pool overhead: 160 MB (5 slots x 16 MB x 2 ranks, ~1% pinned).
//
// Scheduling strategy:
//   After execute_attention(layer_N), schedule_dequant(layer_N+2) fires on
//   kAsyncDequant stream (Stream 6).  This overlaps with expert FFN on
//   other GPUs — TP attention GPUs are idle during MoE dispatch.  acquire()
//   on layer_N+1 finds a ready slot (common case), or inserts a
//   stream_wait_event if dequant is still in flight.  Fallback: synchronous
//   dequant on the attention stream if nothing was pre-scheduled.
//
// Thread safety: NOT thread-safe (single-threaded daemon, INV-3.4.2).
//==============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>

namespace layerstorm::compute {
class AttentionDevice;
class StreamManager;
}  // namespace layerstorm::compute

namespace layerstorm::parallelism {

struct AttentionLayerWeights;

class KvBvDequantPool {
public:
    struct Options {
        int dcp_size = 1;
        int num_slots = 5;                ///< 2 permanent (layers 0,1) + 3 rotating
        int num_heads_local = 64;
        int qk_nope_head_dim = 128;
        int v_head_dim = 128;
        int kv_lora_rank = 512;
        std::vector<compute::AttentionDevice*> attention_devices;
        compute::StreamManager* stream_manager = nullptr;
    };

    explicit KvBvDequantPool(Options opts);
    ~KvBvDequantPool();

    // Non-copyable, non-movable (owns device buffers + events).
    KvBvDequantPool(const KvBvDequantPool&) = delete;
    KvBvDequantPool& operator=(const KvBvDequantPool&) = delete;
    KvBvDequantPool(KvBvDequantPool&&) = delete;
    KvBvDequantPool& operator=(KvBvDequantPool&&) = delete;

    /// Pre-dequant layers 0 and 1 synchronously into permanent slots.
    /// Called once at init after all weights are uploaded.
    /// weights_by_rank: [dcp_size] per-rank weight descriptors for the layer.
    void prime(int layer_idx,
               const AttentionLayerWeights* const* weights_by_rank);

    /// Schedule predictive async dequant on the kAsyncDequant stream.
    /// No-op if: layer already has a slot, or weight is BF16.
    void schedule_dequant(int layer_idx,
                          const AttentionLayerWeights* const* weights_by_rank);

    /// Result from acquire(): BF16 weight pointer + stride for batched GEMM.
    struct AcquireResult {
        const void* weight_ptr;    ///< BF16 [HL, V, D_c] (dequanted) or direct into pinned
        int64_t stride_a;          ///< element stride between heads in A
        bool is_direct;            ///< true = pointing into pinned BF16 (no dequant)
    };

    /// Acquire BF16 v_b weight pointer for layer_idx on the given rank.
    ///
    /// - BF16 weight: returns direct pointer into pinned kv_b_proj V rows
    ///   (offset P*D_c*2 bytes, stride (P+V)*D_c elements between heads).
    /// - FP8, slot kReady: returns dequanted buffer immediately.
    /// - FP8, slot kDequanting: inserts stream_wait_event on attn_stream, returns buffer.
    /// - FP8, no slot: dequants synchronously on attn_stream, returns buffer.
    AcquireResult acquire(int layer_idx, int rank,
                          const AttentionLayerWeights& weights,
                          void* attn_stream);

    /// Mark a layer's slot as reusable.  Permanent slots (layers 0,1) are never freed.
    void release(int layer_idx);

private:
    enum class SlotState { kIdle, kDequanting, kReady };
    struct Slot {
        int layer_idx = -1;
        SlotState state = SlotState::kIdle;
        std::vector<void*> buffers;    // [dcp_size] BF16 device ptrs
        std::vector<void*> events;     // [dcp_size] CUDA events for async completion
    };

    /// Find a slot holding layer_idx, or -1.
    int find_slot(int layer_idx) const;

    /// Find an evictable slot (oldest kReady non-permanent, or kIdle).
    int find_evictable_slot() const;

    /// Launch dequant for one rank on the given stream.
    void launch_dequant_rank(int slot_idx, int rank,
                             const AttentionLayerWeights& weights,
                             void* stream);

    static constexpr int kPermanentLayers = 2;  // layers 0, 1

    Options opts_;
    std::vector<Slot> slots_;
    size_t slot_bytes_ = 0;             ///< BF16 bytes per slot per rank
};

}  // namespace layerstorm::parallelism

#pragma once

#include <cstdint>
#include <vector>

#include "config/config_parser.h"
#include "model/layer_registry.h"
#include "model/model_config.h"

namespace layerstorm::compute { class DeviceBackend; }

namespace layerstorm::memory {

// ── KV cache format (informational, for logging/assertions/kernel dispatch) ─

// kV4Fp8 / kV4Tq describe the V4 MAIN-pool (CSA bucket) entry format
// (deps/LayerStoRmKernels csa_fp8/params.h V4CacheLayout: 1160 B; csa_tq:
// 644 B).  Existing non-V4 dispatch gates accept only the first two — V4
// formats are intentionally rejected there (fail-closed until ticket V4-5).
enum class KvCacheFormat { kSnapMlaFp8, kTurboQuantMse4, kV4Fp8, kV4Tq };

// ── DeepSeek-V4 multi-tier KV cache (V4-3b/V4-3c) ───────────────────────────
//
// Three page-size buckets share a unified 256-native-token logical block
// (spec/DEEPSEEK4_PLAN.md V4-3c; deps kernel layouts are authoritative):
//   CSA bucket (Pool::kMain reused): 256/4  = 64 compressed entries / page
//   HCA bucket (Pool::kHca):         256/128 = 2 compressed entries / page
//   SWA bucket (Pool::kSwa):         128 raw tokens / page (sliding window +
//     compressor residual tail, treated as extended window per V4-3c)
// plus the Lightning-Indexer tier (Pool::kIndexerK reused): one entry PER
// COMPRESSED BLOCK (4 native tokens) of index_head_dim FP8 + 4 B F32 scale.
//
// Entry formats: FP8 = 1160 B [K512|scale4|rope128|V512|scale4]; TQ = 644 B.
// SWA is ALWAYS FP8 (all TQ arms).  Backend map: csa_hca → FP8/FP8;
// csa_hca_tq → TQ/TQ; csa_hca_tq_mix → TQ CSA + FP8 HCA (V4-5Mb).

inline constexpr int64_t kV4Fp8EntryBytes = 1160;
inline constexpr int64_t kV4TqEntryBytes = 644;

/// Single authority for V4 compressed-entry bytes by cache format
/// (attention refactor V2 P2 — the codec axis; kv_codec.h re-exposes this
/// for the attention stack). SWA is always FP8 regardless of arm.
inline constexpr int64_t v4_entry_bytes(KvCacheFormat f) {
    return f == KvCacheFormat::kV4Tq ? kV4TqEntryBytes : kV4Fp8EntryBytes;
}
inline constexpr int kV4LogicalBlockTokens = 256;
inline constexpr int kV4CsaRatio = 4;    // validator pins ratios to {0,4,128}
inline constexpr int kV4HcaRatio = 128;

struct V4KvLayout {
    bool enabled = false;

    // Page geometry (fixed by the unified logical block + ratios).
    int logical_block_tokens = kV4LogicalBlockTokens;
    int csa_entries_per_page = kV4LogicalBlockTokens / kV4CsaRatio;   // 64
    int hca_entries_per_page = kV4LogicalBlockTokens / kV4HcaRatio;   // 2
    int swa_page_tokens = 0;          // = model.sliding_window (128)

    // Per-entry bytes by backend arm (SWA always FP8).
    int64_t csa_entry_bytes = 0;
    int64_t hca_entry_bytes = 0;
    int64_t swa_entry_bytes = kV4Fp8EntryBytes;
    KvCacheFormat csa_format = KvCacheFormat::kV4Fp8;
    KvCacheFormat hca_format = KvCacheFormat::kV4Fp8;

    // Derived page bytes.
    int64_t csa_bytes_per_page = 0;   // == VramLayout::kv_bytes_per_page
    int64_t hca_bytes_per_page = 0;
    int64_t swa_bytes_per_page = 0;

    // Lightning-Indexer tier: entry per CSA compressed block
    // (index_head_dim FP8 + 4 B F32 scale); page covers
    // memory.kv_cache.indexer_k_page_size_tokens native tokens.
    int64_t indexer_entry_bytes = 0;
    int64_t indexer_bytes_per_page = 0;  // == VramLayout::indexer_k_bytes_per_page

    // Layer counts from compress_ratios (hidden layers).
    int num_csa_layers = 0;
    int num_hca_layers = 0;
    int num_swa_layers = 0;
    // SWA-bucket KV layers including nextn MTP layers (SWA-only by spec).
    int num_swa_kv_layers = 0;
};

/// Compute the V4 tier/page geometry from config + model (pure math).
/// Precondition: model_cfg.is_v4().
V4KvLayout compute_v4_kv_layout(const config::Config& cfg,
                                const model::ModelConfig& model_cfg);

// ── Per-GPU VRAM budget breakdown ───────────────────────────────────────────

struct GpuVramLayout {
    int gpu_id;
    int64_t total_vram_bytes;
    int64_t safety_margin_bytes;     // 512 MB per GPU (CUDA context overhead)
    int64_t pinned_bytes;            // From LayerRegistry (attention, gating, dense FFN, etc.)
    int64_t kv_main_bytes;           // Main KV cache pool (includes scratch tail)
    int64_t kv_speculation_bytes;    // Speculation KV cache pool
    int64_t expert_stable_bytes;     // Stable expert cache zone (slow eviction)
    int64_t expert_streaming_bytes;  // Streaming expert cache zone (fast turnover)

    // Indexer K cache (DSA only, replicated across TP GPUs even with DCP;
    // V4 reuses it for the Lightning-Indexer tier — CSA layers only)
    int64_t indexer_k_bytes = 0;
    int indexer_k_pages = 0;

    // V4 tier regions (zero / collapsed for non-V4 models, V4-3b).
    // CSA main tier reuses kv_main_bytes/kv_main_pages.
    int64_t kv_hca_bytes = 0;
    int kv_hca_pages = 0;
    int64_t kv_swa_bytes = 0;
    int kv_swa_pages = 0;

    // DCP: KV cache sharding factor. 1 unless hardware.dcp_kv_mode = sharded
    // (then tp_degree on TP GPUs). Replicated KV (default) is 1 even with DCP
    // enabled: each TP GPU claims the same page_idx in lockstep (INV-KV-REP),
    // so per-rank auto sizing must NOT divide (INV-KV-SIZE-SHARD,
    // TD-KV-REP-POOL-HALVED).
    int dcp_kv_shard_factor = 1;

    // Prefill scratch — pre-allocated at tail of KV main
    int64_t prefill_scratch_preallocated_bytes = 0;

    // Streaming zone split: spill (top, repurposable) + prefetch (bottom, permanent)
    int64_t streaming_spill_bytes = 0;
    int64_t streaming_prefetch_bytes = 0;

    int64_t kv_total_bytes() const { return kv_main_bytes + kv_speculation_bytes; }
    int64_t expert_total_bytes() const { return expert_stable_bytes + expert_streaming_bytes; }

    int max_kv_pages;         // Total KV pages (main + speculation)
    int kv_main_pages;
    int kv_speculation_pages;
};

// ── System-wide VRAM layout ─────────────────────────────────────────────────

struct VramLayout {
    std::vector<GpuVramLayout> gpus;
    // Data bytes per KV page in the main/speculation pools.  Uniform across
    // layers for MLA models; for V4 this is the CSA-bucket page size (the
    // main pool IS the CSA bucket — HCA/SWA buckets carry their own page
    // sizes in `v4`).
    int64_t kv_bytes_per_page;
    int64_t indexer_k_bytes_per_page = 0;  // Indexer K page size (0 if unused)
    KvCacheFormat kv_cache_format = KvCacheFormat::kSnapMlaFp8;
    V4KvLayout v4;  // enabled=false for non-V4 models
};

// ── Budget computation (pure math, no CUDA) ─────────────────────────────────

/// KV cache data bytes per token per layer.
/// Dispatches by attention backend: SnapMLA FP8 → 644 B (V3.2), TQ MSE 4-bit → 386 B.
/// V4: throws std::logic_error — there is NO uniform per-token size (per-layer
/// tiers; use compute_v4_kv_layout / v4_kv_bytes_per_token instead).  The V4
/// config carries inert MLA schema defaults (kv_lora_rank=512 etc.), so the
/// SnapMLA branch would otherwise return silent nonsense.
int64_t kv_bytes_per_token(const model::ModelConfig& model_cfg,
                           config::KvCacheQuant kv_quant,
                           config::AttentionBackendType backend);

/// V4 MAIN-tier bytes per NATIVE token for one layer (compressed entry bytes
/// divided by the layer's compress ratio; exact — 1160/4, 644/4, 1160/128 are
/// not integral, so this returns bytes per LOGICAL BLOCK / 256 rounded up via
/// page math in compute_v4_kv_layout; this helper reports entry_bytes and is
/// primarily for diagnostics/tests).  SWA-only layers return 0 (raw tier only).
int64_t v4_main_tier_entry_bytes(const config::Config& cfg,
                                 const model::ModelConfig& model_cfg,
                                 int layer_idx);

/// KV cache data bytes per page (page_size_tokens * kv_bytes_per_token).
int64_t kv_bytes_per_page(const model::ModelConfig& model_cfg,
                          const config::Config& cfg);

/// Indexer K cache bytes per token per layer (0 if no DSA).
/// MQA: 1 K head × index_head_dim × kv_quant_bpe.
int64_t indexer_k_bytes_per_token(const model::ModelConfig& model_cfg,
                                   config::KvCacheQuant kv_quant);

/// Indexer K cache bytes per page (indexer_k_bpt × indexer_k_page_size_tokens).
int64_t indexer_k_bytes_per_page(const model::ModelConfig& model_cfg,
                                  const config::Config& cfg);

/// Dense naive prefill scratch bytes for a given context length.
/// This is the kv_b_proj decompression buffer (one layer at a time, reused):
///   num_attention_heads × (qk_nope_head_dim + v_head_dim) × seq_len × sizeof(BF16)
/// Returns 0 for DSA-only models (they use absorbed sparse, not naive decompression).
int64_t naive_prefill_scratch_bytes(const model::ModelConfig& model_cfg, int seq_len);

/// Compute system-wide VRAM layout (byte budgets only, no allocation).
/// Throws std::runtime_error if any GPU has insufficient VRAM.
VramLayout compute_vram_layout(const config::Config& cfg,
                               const model::LayerRegistry& registry,
                               const model::ModelConfig& model_cfg);

// ── Per-GPU allocated region ────────────────────────────────────────────────

struct GpuRegion {
    config::GpuRef gpu;              // GPU reference (INV-4.18: use .position for indexing, .id for CUDA)
    void* base = nullptr;            // Single allocation (all regions contiguous)
    int64_t allocated_bytes = 0;     // Total allocated (= total_vram - safety_margin)

    // Region pointers: contiguous offsets into base.
    // Layout: pinned | kv_speculation | indexer_k | kv_hca | kv_swa |
    //         kv_main (+scratch tail) | expert_streaming | expert_stable
    // kv_main end == expert_streaming start (contiguity for prefill scratch + spill)
    void* pinned = nullptr;
    void* kv_speculation = nullptr;
    void* indexer_k = nullptr;       // DSA / V4 lightning indexer K cache (collapsed if unused)
    void* kv_hca = nullptr;          // V4 HCA main tier (collapsed for non-V4)
    void* kv_swa = nullptr;          // V4 SWA/raw tier (collapsed for non-V4)
    void* kv_main = nullptr;         // MLA main / V4 CSA tier; incl. scratch at tail
    void* expert_streaming = nullptr;  // Streaming zone: spill (top) + prefetch (bottom)
    void* expert_stable = nullptr;     // Stable zone: last, never touched by prefill
};

// ── VramAllocator (RAII owner of GPU memory) ────────────────────────────────

/// Allocates GPU memory per the computed VramLayout and partitions into typed
/// regions. Owns the memory; frees on destruction.
///
/// Downstream consumers receive region pointers:
///   - Page allocator (#15): kv_main, kv_speculation
///   - Expert cache (#17): expert_stable, expert_streaming
///   - Engine init: pinned (for model weight placement)
class VramAllocator {
public:
    /// Allocate VRAM on all GPUs per the given layout.
    /// @param device_backends  One DeviceBackend* per GPU (indexed by position).
    /// Throws std::runtime_error on allocation failure.
    explicit VramAllocator(VramLayout layout,
                           std::vector<compute::DeviceBackend*> device_backends);
    ~VramAllocator();

    VramAllocator(const VramAllocator&) = delete;
    VramAllocator& operator=(const VramAllocator&) = delete;
    VramAllocator(VramAllocator&&) noexcept;
    VramAllocator& operator=(VramAllocator&&) noexcept;

    const VramLayout& layout() const { return layout_; }
    const GpuRegion& region(int gpu_idx) const { return regions_[gpu_idx]; }
    int gpu_count() const { return static_cast<int>(regions_.size()); }
    bool owns_memory() const { return !regions_.empty(); }

private:
    VramLayout layout_;
    std::vector<GpuRegion> regions_;
    std::vector<compute::DeviceBackend*> device_backends_;

    void allocate_all();
    void free_all();
};

}  // namespace layerstorm::memory

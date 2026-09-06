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

// ── GF3.8: KDA per-request state pool (glm5_next linear-attention layers) ───
//
// Unlike KV, the KDA state is PER REQUEST, not per token: one sequence owns
// one fixed-size SLOT for its whole life — fp32 recurrent state
// [H/tp][D][D] plus THREE fp32 conv rings [K-1+S][H/tp*D] per linear
// layer (GF3.6 pinned exactly this pair as the checkpoint/replay unit,
// INV-KDA-REWIND; the GF3.7 kernels read/write it via
// `base + slot * stride` indirection, which is why the pool must be ONE
// contiguous region with a UNIFORM slot stride — scattered whole-slab
// claims cannot satisfy a single-base kernel launch; see the GF3.8
// amendment in spec/plans/RADIX_SLAB_DESIGN.md §1).
//
// Slot layout (all fp32, offsets from the slot base):
//   for each linear layer l (dense ordinal over the model's linear layers):
//     [recurrent H/tp*D*D | ring_q | ring_k | ring_v]
//   ring_* = (conv_kernel-1 + spec_columns) * (H/tp * D) floats.
// A slot is therefore SELF-CONTAINED: a state CHECKPOINT (GF3.12) is a byte
// copy of the slot and restore is the reverse copy — no other engine state
// participates (the GF3.6 unit, verbatim; fp32 round-trips are exact, so a
// restored slot replays bit-identically, INV-KDA-CARRY).
struct KdaStateLayout {
    bool enabled = false;         ///< model has linear-attention layers
    /// TD-KDA-STATE-MAPPED-SLABS: when true the state pool has NO dedicated
    /// carve — each (request, linear layer) unit is a contiguous run of
    /// whole slabs claimed from the shared kv_main slab region (allocation
    /// unit = ceil(per_layer_bytes / slab_bytes) slabs; the kernels' base +
    /// slot * stride contract holds with stride = slab_bytes and per-layer
    /// slot tables of run-start slab ids). Policy switch, resolved at
    /// sizing: config _internal-kda_state.mapped, env LS_KDA_STATE_MAPPED
    /// overrides either way. Default OFF (the GF3.8 carve stays primary).
    bool mapped = false;
    int num_layers = 0;           ///< linear (KDA) layers, e.g. 34
    int heads_per_rank = 0;       ///< num_heads / tp
    int head_dim = 0;             ///< 128
    int conv_kernel = 4;          ///< short_conv_kernel_size
    int spec_columns = 0;         ///< extra conv ring columns (GF3.11 seam;
                                  ///  0 until a replay-ring design lands)
    int64_t recurrent_bytes_per_layer = 0;  ///< H/tp * D * D * 4
    int64_t ring_bytes_per_layer = 0;       ///< ONE of the q/k/v rings
    int64_t per_layer_bytes = 0;            ///< recurrent + 3 rings
    int64_t slot_bytes = 0;       ///< 256-aligned slot stride (whole request)

    /// Byte offset of linear-layer l's recurrent state inside a slot.
    /// l is the DENSE linear-layer ordinal [0, num_layers), not the model
    /// layer index (GF3.9 owns the model-layer -> ordinal map).
    int64_t recurrent_offset(int l) const { return l * per_layer_bytes; }
    /// Byte offset of linear-layer l's conv ring (which: 0=q, 1=k, 2=v).
    int64_t ring_offset(int l, int which) const {
        return l * per_layer_bytes + recurrent_bytes_per_layer
             + which * ring_bytes_per_layer;
    }
};

/// Compute the KDA state slot geometry (pure math, no CUDA).
/// Disabled (all zeros) when the model has no linear-attention layers.
/// Throws std::invalid_argument when num_heads % tp_degree != 0.
KdaStateLayout compute_kda_state_layout(const model::ModelConfig& model_cfg,
                                        int tp_degree, int spec_columns);

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
    // V4 reuses it for the Lightning-Indexer tier — CSA layers only).
    // S1 slab geometry (TD-INDEXER-POOL-ELASTIC, RADIX_SLAB_DESIGN §1/§5):
    // when VramLayout::slab_bytes > 0 the indexer carve is indexer_k_pages
    // SLABS of slab_bytes each (one indexer page per slab; tail sliver
    // < one kMain page wasted per slab) forming the head of ONE shared
    // region whose tail is the kv_main page span — the indexer/kv boundary
    // is slab-exact (no alignment gap), so a slab is an ALIGNED RUN of
    // flat kMain page indices and (slab, offset) stays representable as an
    // ordinary page index. indexer_k_bytes == indexer_k_pages * slab_bytes.
    int64_t indexer_k_bytes = 0;
    int indexer_k_pages = 0;
    // Alignment shim placed BEFORE the indexer span (never part of slab
    // arithmetic): sized so the span END — the kv_main base — stays
    // kLayoutAlign(256)-aligned even when slab_bytes is not a multiple of
    // 256 (SnapMLA arm: slab ≡ 64 mod 256). Zero on TQ/V4 arms and on
    // non-slabbed models. The indexer span start itself is then only
    // 64-byte aligned, which every indexer consumer tolerates (pointer
    // tables / ≤16 B vector accesses).
    int64_t indexer_k_pad_bytes = 0;
    // S4 (TD-INDEXER-POOL-ELASTIC): on slabbed models the indexer demand is
    // NOT carved — it is folded into kv_main as the shared pool's indexer
    // SHARE (whole slabs, priced at slab_bytes). indexer_k_pages/bytes are
    // then 0; these record the share for boot logs / tests / introspection.
    int indexer_share_slabs = 0;
    int64_t indexer_share_bytes = 0;

    // V4 tier regions (zero / collapsed for non-V4 models, V4-3b).
    // CSA main tier reuses kv_main_bytes/kv_main_pages.
    int64_t kv_hca_bytes = 0;
    int kv_hca_pages = 0;
    int64_t kv_swa_bytes = 0;
    int kv_swa_pages = 0;

    // GF3.8: KDA per-request state pool (glm5_next only; zero/collapsed
    // otherwise). kda_state_slots whole-request slots of
    // VramLayout::kda.slot_bytes each — one slot per live sequence
    // (in-flight requests + prefix holders, which own full copies:
    // INV-PREFIX-CACHE-3 third cost class). The pool is the resource that
    // actually clamps concurrency on this architecture.
    int64_t kda_state_bytes = 0;
    int kda_state_slots = 0;

    // TD-KDA-MAPPED-NONTP-GPUS: does this GPU carry the ATTENTION-SIDE
    // tenants at all? `(!has_tp || in_tp[gpu_i])` — true on every GPU of a
    // tp-less config and on TP-set members; FALSE on an expert-only
    // (non-TP) GPU, which hosts no attention, no KV/indexer pool and no
    // KDA state, so its kMain is legitimately EMPTY. This is the ONE
    // predicate both sides of the allocator must agree on: the sizing
    // (compute_vram_layout) gates every attention-side share on it, and
    // the PageAllocator ctor gates its preconditions on the SAME flag —
    // the divergence between the two is what made mapped KDA state throw
    // `requires slabbed kMain` on expert-only GPUs (a guard meant for
    // UNSLABBED MODELS firing on a NON-PARTICIPATING GPU). Default true:
    // a hand-built single-GPU layout hosts attention.
    bool attention_host = true;

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

    // ── TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA: admissible-context ceiling ──
    // Largest single-request context (tokens, prompt + generation budget)
    // whose WHOLE-LIFE admission demand fits this GPU's kMain pool: KV
    // pages + the full-length indexer-K reservation (INV-DSA-RESERVE
    // claims it at admission) + the mapped KDA state's whole-slab runs.
    // Since TD-KDA-STATE-MAPPED-SLABS these are ALL tenants of the ONE
    // shared pool, so a VRAM-clamped pool can be smaller than one
    // max-length request even though every component "fits" its own share
    // — the engine must not claim to serve max_sequence_length past this
    // number. 0 = not computed (V4 tiers, expert-only GPUs, empty kv_main).
    // The per-request demand components AT cfg max_sequence_length are
    // kept for the boot line and the autoconfig boot-log cross-check row.
    int admissible_ctx_tokens = 0;
    int64_t admission_demand_pages = 0;   // total at max_sequence_length
    int64_t admission_kv_pages = 0;       //   = KV pages
    int64_t admission_indexer_pages = 0;  //   + indexer slabs, in kMain pages
    int64_t admission_state_pages = 0;    //   + mapped KDA state, in kMain pages
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

    // ── S1 shared-region slab geometry (TD-INDEXER-POOL-ELASTIC) ──
    // slab_bytes = ceil(indexer_k_bytes_per_page / kv_bytes_per_page)
    //            * kv_bytes_per_page   (RADIX_SLAB_DESIGN §1 rule 2)
    // Nonzero iff the model carries an indexer-K pool (DSA or V4 LID).
    // A slab is simultaneously (a) the claim unit of the indexer carve —
    // one indexer page per slab, physical stride slab_bytes — and (b) an
    // aligned run of pages_per_slab flat kMain page indices, which is what
    // keeps S2+ an allocator change (block tables / page_ptr / kernel
    // contracts untouched). kMain page counts on slabbed models are
    // quantized DOWN to whole slabs at boot.
    int64_t slab_bytes = 0;
    int pages_per_slab = 0;

    KvCacheFormat kv_cache_format = KvCacheFormat::kSnapMlaFp8;
    V4KvLayout v4;  // enabled=false for non-V4 models
    KdaStateLayout kda;  // GF3.8: enabled=false for non-glm5_next models
};

// ── TD-GLM5-KDA-SLOTS-EXPORT: boot-metadata view of the state pool ──────────
// The orchestrator's admission must SEE state-pool pressure
// (EngineInfo.kda_state_*).  Reduction rule: the BINDING number is the min
// across attention-host GPUs (replicated KV claims slabs in lockstep,
// INV-KDA-TP, so pools stay near-mirrored and a refusal fires when ANY rank
// fails); expert-only GPUs host no state (attention_host false,
// TD-KDA-MAPPED-NONTP-GPUS) and must not dilute the min.  All zero when the
// model has no linear-attention state.
struct KdaStateExport {
    bool mapped = false;          ///< state units live in the SHARED kMain pool
    int slots = 0;                ///< carve mode: dedicated whole-request slots
                                  ///  (min across attention hosts); 0 mapped
    int64_t slot_bytes = 0;       ///< one request's per-rank state bytes
    int64_t pages_per_seq = 0;    ///< mapped: ONE admission's state demand in
                                  ///  kMain pages (whole-slab runs); 0 carved
    int64_t pool_pages = 0;       ///< mapped: shared kMain pool pages (min
                                  ///  across attention hosts); 0 carved
};

/// Pure reduction over a computed VramLayout (no CUDA).
KdaStateExport kda_state_export(const VramLayout& layout);

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
/// NOTE (GF3.5): on IndexPool models (ModelConfig::has_index_pool) this is
/// the per-ENTRY byte count; entries cover index_kpool tokens each, so it is
/// NOT bytes-per-token there — use indexer_k_bytes_per_page for sizing.
int64_t indexer_k_bytes_per_token(const model::ModelConfig& model_cfg,
                                   config::KvCacheQuant kv_quant);

/// GF3.5 (IndexPool): stored entries per indexer-K page. A page always spans
/// indexer_k_page_size_tokens token POSITIONS (owner math, reservation math,
/// coverage and page counts are all stated over positions and unchanged);
/// pooling only divides the stored ROW count: PT/index_kpool pooled entries
/// (PT for legacy kpool<=1 models). PT % index_kpool == 0 is validated.
int indexer_k_entries_per_page(const model::ModelConfig& model_cfg,
                               const config::Config& cfg);

/// GF3.5 (IndexPool): per-page TAIL region bytes — raw bf16 K + raw bf16
/// gate vectors of the in-progress pool ([2, index_kpool, index_head_dim]
/// bf16, vLLM Glm5NextTailCache shape). Present on EVERY pooled page
/// (uniform stride); live only on the frontier page — the in-progress pool
/// always lies inside the frontier page because PT % kpool == 0. Keeping it
/// in-page means holder freeze / CMD_SEQ_HIBERNATE / restore carry the
/// frontier tail state with the page bytes, with zero new machinery
/// (INV-PREFIX-CACHE-3/-4). 0 for legacy models.
int64_t indexer_k_tail_bytes(const model::ModelConfig& model_cfg);

/// Indexer K cache bytes per page. Legacy: indexer_k_bpt × PT. IndexPool
/// (GF3.5): entries × entry_bytes + tail — layout
/// [E × head_dim FP8 | E × 4 B f32 scales | 2 × kpool × head_dim bf16 tail],
/// E = PT / index_kpool. The split fp8|scales shape is the SAME as legacy
/// (scores/append kernels are entry-count-parameterized, not token-count).
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
    // Layout: pinned | kv_speculation | kda_state | kv_hca | kv_swa |
    //         indexer_k | kv_main (+scratch tail) | expert_streaming | expert_stable
    // kv_main end == expert_streaming start (contiguity for prefill scratch + spill).
    // S1 (TD-INDEXER-POOL-ELASTIC): on slabbed models (VramLayout::slab_bytes
    // > 0) [indexer_k, kv_main end) is ONE shared region — the indexer span is
    // indexer_k_pages slabs and kv_main starts EXACTLY at its end (slab-exact
    // boundary, no alignment gap; indexer_k_pad_bytes sits BEFORE the span so
    // kv_main stays 256-aligned). V4 kHca/kSwa keep separate regions
    // (different page geometry, out of S1 scope) and were moved ahead of the
    // shared region to keep it contiguous.
    void* pinned = nullptr;
    void* kv_speculation = nullptr;
    void* kda_state = nullptr;       // GF3.8 KDA per-request state pool (collapsed if unused)
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

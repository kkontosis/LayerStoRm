#include "core/memory/vram_allocator.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>

#include "config/config_resolver.h"
#include "core/device_backend.h"

namespace layerstorm::memory {

// ── KV quant bytes per element ──────────────────────────────────────────────

static double kv_bytes_per_element(config::KvCacheQuant q) {
    switch (q) {
        case config::KvCacheQuant::fp8_e4m3:
        case config::KvCacheQuant::fp8_e5m2:
            return 1.0;
        case config::KvCacheQuant::fp16:
            return 2.0;
    }
    return 1.0;  // unreachable
}

// ── KV bytes per token per layer ────────────────────────────────────────────

int64_t kv_bytes_per_token(const model::ModelConfig& model_cfg,
                           config::KvCacheQuant kv_quant,
                           config::AttentionBackendType backend) {
    // V4 has NO uniform per-token KV size (3-tier per-layer scheme, V4-3b).
    // Its config still carries inert MLA schema defaults (kv_lora_rank=512),
    // so falling through would silently size a SnapMLA cache — fail loud.
    if (model_cfg.is_v4()) {
        throw std::logic_error(
            "kv_bytes_per_token: deepseek_v4 has no uniform per-token KV "
            "size; use compute_v4_kv_layout (V4-3b)");
    }
    const auto& m = model_cfg.raw();
    double bpe = kv_bytes_per_element(kv_quant);

    if (m.kv_lora_rank > 0) {
        if (backend == config::AttentionBackendType::turboquant_mla) {
            // TQ MSE 4-bit: packed 4-bit nope + FP16 L2 norm + BF16 rope
            double base = m.kv_lora_rank * 0.5   // 4 bits per element
                        + 2.0                      // FP16 L2 norm
                        + m.qk_rope_head_dim * 2.0; // BF16 rope
            return static_cast<int64_t>(std::ceil(base));
        }
        // SnapMLA: compressed KV = kv_lora_rank at kv_quant + rope at BF16
        double base = m.kv_lora_rank * bpe + m.qk_rope_head_dim * 2.0;
        if (kv_quant == config::KvCacheQuant::fp8_e4m3 ||
            kv_quant == config::KvCacheQuant::fp8_e5m2) {
            base += 4.0;  // per-token float32 quantization scale (SnapMLA format)
        }
        return static_cast<int64_t>(base);
    }
    // Standard MHA: 2 * num_kv_heads * head_dim * bpe (backend-independent)
    int head_dim = m.qk_nope_head_dim + m.qk_rope_head_dim;
    return static_cast<int64_t>(2.0 * m.num_key_value_heads * head_dim * bpe);
}

// ── KV bytes per page ───────────────────────────────────────────────────────

int64_t kv_bytes_per_page(const model::ModelConfig& model_cfg,
                          const config::Config& cfg) {
    return kv_bytes_per_token(model_cfg, cfg.quantization.kv_cache,
                              cfg.compute.attention_backend) *
           cfg.memory.kv_cache.page_size_tokens;
}

// ── Indexer K cache sizing ──────────────────────────────────────────────────

int64_t indexer_k_bytes_per_token(const model::ModelConfig& model_cfg,
                                   config::KvCacheQuant kv_quant) {
    const auto& m = model_cfg.raw();
    if (m.index_topk <= 0) return 0;  // No DSA

    // MQA: 1 K head × index_head_dim at kv_quant precision, plus the
    // per-token F32 absmax scale (TD-GLM-INDEXER-PAGED: the producer's FP8
    // quant stores one scale per position; page layout is
    // [page_tokens × head_dim K | page_tokens × 4 B scales]).
    double bpe = kv_bytes_per_element(kv_quant);
    return static_cast<int64_t>(1 * m.index_head_dim * bpe) +
           static_cast<int64_t>(sizeof(float));
}

int indexer_k_entries_per_page(const model::ModelConfig& model_cfg,
                               const config::Config& cfg) {
    const int PT = cfg.memory.kv_cache.indexer_k_page_size_tokens;
    if (!model_cfg.has_index_pool()) return PT;
    const int P = model_cfg.raw().index_kpool;
    // PT % P == 0 (validator pool-alignment list) — a pool never crosses an
    // indexer page, so the frontier's in-progress pool lives in the frontier
    // page and entries divide exactly.
    return PT / P;
}

int64_t indexer_k_tail_bytes(const model::ModelConfig& model_cfg) {
    if (!model_cfg.has_index_pool()) return 0;
    const auto& m = model_cfg.raw();
    // [2, kpool, head_dim] bf16: raw post-LayerNorm K (UNrotated,
    // UNquantized) + raw gate vector, one slot per pos % kpool
    // (scratchpad/GF35_KPOOL_REFERENCE.md §2).
    return static_cast<int64_t>(2) * m.index_kpool * m.index_head_dim * 2;
}

int64_t indexer_k_bytes_per_page(const model::ModelConfig& model_cfg,
                                  const config::Config& cfg) {
    return indexer_k_bytes_per_token(model_cfg, cfg.quantization.kv_cache) *
               indexer_k_entries_per_page(model_cfg, cfg) +
           indexer_k_tail_bytes(model_cfg);
}

// ── Naive prefill scratch sizing ─────────────────────────────────────────────

int64_t naive_prefill_scratch_bytes(const model::ModelConfig& model_cfg,
                                     int seq_len) {
    if (seq_len <= 0) return 0;
    const auto& m = model_cfg.raw();
    // kv_b_proj decompression output (one layer at a time, buffer reused):
    // num_attention_heads × (qk_nope_head_dim + v_head_dim) × seq_len × sizeof(BF16)
    return static_cast<int64_t>(m.num_attention_heads) *
           (m.qk_nope_head_dim + m.v_head_dim) *
           seq_len * 2;  // BF16 = 2 bytes
}

// ── DeepSeek-V4 tier geometry (V4-3b) ───────────────────────────────────────

V4KvLayout compute_v4_kv_layout(const config::Config& cfg,
                                const model::ModelConfig& model_cfg) {
    if (!model_cfg.is_v4()) {
        throw std::logic_error("compute_v4_kv_layout: model is not deepseek_v4");
    }
    const auto& m = model_cfg.raw();
    V4KvLayout v4;
    v4.enabled = true;
    v4.swa_page_tokens = m.sliding_window;  // 128 (validator: required > 0)

    // Backend arm → per-tier entry formats (SWA always FP8).
    switch (cfg.compute.attention_backend) {
        case config::AttentionBackendType::csa_hca:
            v4.csa_format = KvCacheFormat::kV4Fp8;
            v4.hca_format = KvCacheFormat::kV4Fp8;
            break;
        case config::AttentionBackendType::csa_hca_tq:
            v4.csa_format = KvCacheFormat::kV4Tq;
            v4.hca_format = KvCacheFormat::kV4Tq;
            break;
        case config::AttentionBackendType::csa_hca_tq_mix:
            // V4-5Mb: TQ on CSA tiers, FP8 on HCA tiers (few entries).
            v4.csa_format = KvCacheFormat::kV4Tq;
            v4.hca_format = KvCacheFormat::kV4Fp8;
            break;
        default:
            throw std::invalid_argument(
                "compute_v4_kv_layout: deepseek_v4 requires a csa_hca* "
                "attention backend");
    }
    v4.csa_entry_bytes = v4_entry_bytes(v4.csa_format);
    v4.hca_entry_bytes = v4_entry_bytes(v4.hca_format);
    v4.swa_entry_bytes = kV4Fp8EntryBytes;

    v4.csa_bytes_per_page = v4.csa_entry_bytes * v4.csa_entries_per_page;
    v4.hca_bytes_per_page = v4.hca_entry_bytes * v4.hca_entries_per_page;
    v4.swa_bytes_per_page = v4.swa_entry_bytes * v4.swa_page_tokens;

    // Lightning-Indexer tier (CSA layers only): one FP8 index_head_dim key +
    // one F32 scale PER COMPRESSED BLOCK (deps lightning_score_mqa.h — MQA
    // cache, no head dim).  Page covers indexer_k_page_size_tokens native
    // tokens → page_tokens / ratio entries.
    v4.indexer_entry_bytes =
        static_cast<int64_t>(m.index_head_dim) + sizeof(float);
    const int idx_page_tokens = cfg.memory.kv_cache.indexer_k_page_size_tokens;
    v4.indexer_bytes_per_page =
        static_cast<int64_t>(idx_page_tokens / kV4CsaRatio) *
        v4.indexer_entry_bytes;

    // Layer census from compress_ratios via the ticket-A dispatch predicates.
    for (int l = 0; l < m.num_hidden_layers; ++l) {
        switch (model_cfg.attention_type_for_layer(l)) {
            case model::V4AttentionType::kCsa: ++v4.num_csa_layers; break;
            case model::V4AttentionType::kHca: ++v4.num_hca_layers; break;
            case model::V4AttentionType::kSwa: ++v4.num_swa_layers; break;
        }
    }
    // MTP layers are SWA-only by spec (compress_ratios[43..]=0) and get raw
    // SWA pages like any other layer (INV-KV-LAYER analogue).
    v4.num_swa_kv_layers = m.num_hidden_layers + m.num_nextn_predict_layers;

    return v4;
}

int64_t v4_main_tier_entry_bytes(const config::Config& cfg,
                                 const model::ModelConfig& model_cfg,
                                 int layer_idx) {
    auto v4 = compute_v4_kv_layout(cfg, model_cfg);
    switch (model_cfg.attention_type_for_layer(layer_idx)) {
        case model::V4AttentionType::kCsa: return v4.csa_entry_bytes;
        case model::V4AttentionType::kHca: return v4.hca_entry_bytes;
        case model::V4AttentionType::kSwa: return 0;  // raw tier only
    }
    return 0;
}

// V4 SWA-bucket pages per (sequence, layer): the raw tier holds the sliding
// window PLUS the compressor residual tail (up to stride−1 not-yet-compressed
// raw tokens, folded into the window per V4-3c), plus one page of ring slack
// so appends never stall on the trailing partial page.
static int v4_swa_pages_per_layer(const V4KvLayout& v4,
                                  model::V4AttentionType type) {
    int residual = 0;
    if (type == model::V4AttentionType::kCsa) residual = kV4CsaRatio - 1;
    if (type == model::V4AttentionType::kHca) residual = kV4HcaRatio - 1;
    const int tokens = v4.swa_page_tokens + residual;
    return (tokens + v4.swa_page_tokens - 1) / v4.swa_page_tokens + 1;
}

// Region boundary alignment (SPEC_UPDATES line 73).
// All region boundaries must be 256-byte aligned for CUDA kernel access patterns.
// Sizes are rounded up inline during layout computation so that contiguous regions
// (kv_main end == streaming start) hold with zero gaps.
static constexpr int64_t kLayoutAlign = 256;

static int64_t align_region(int64_t sz) {
    return (sz + kLayoutAlign - 1) & ~(kLayoutAlign - 1);
}

// ── GF3.8: KDA per-request state slot geometry (pure math) ──────────────────

KdaStateLayout compute_kda_state_layout(const model::ModelConfig& model_cfg,
                                        int tp_degree, int spec_columns) {
    KdaStateLayout k;
    const int n_lin = model_cfg.num_linear_attention_layers();
    if (n_lin <= 0) return k;  // disabled: not a linear-attention model
    const auto& m = model_cfg.raw();
    // GF3.2 schema: model.linear_attn_config carries the KDA geometry
    // (defaults are the GLM-5.3-Flash values: 64 heads x 128, conv kernel 4).
    const config::LinearAttnConfig lin =
        m.linear_attn_config ? *m.linear_attn_config
                             : config::LinearAttnConfig{};
    const int tp = tp_degree > 0 ? tp_degree : 1;
    if (lin.num_heads % tp != 0) {
        throw std::invalid_argument(
            "KDA state layout: linear_attn_config.num_heads ("
            + std::to_string(lin.num_heads)
            + ") is not divisible by tensor_parallelism ("
            + std::to_string(tp) + ")");
    }
    if (spec_columns < 0) {
        throw std::invalid_argument(
            "KDA state layout: spec_columns must be >= 0");
    }
    k.enabled = true;
    k.num_layers = n_lin;
    k.heads_per_rank = lin.num_heads / tp;
    k.head_dim = lin.head_dim;
    k.conv_kernel = lin.short_conv_kernel_size;
    k.spec_columns = spec_columns;
    // All fp32 (the GF3.6/GF3.7 precision contract: the state is the fp32
    // accumulator; a narrower store would break INV-KDA-CARRY bitwise
    // replay).
    k.recurrent_bytes_per_layer =
        static_cast<int64_t>(k.heads_per_rank) * k.head_dim * k.head_dim * 4;
    k.ring_bytes_per_layer =
        static_cast<int64_t>(k.conv_kernel - 1 + k.spec_columns)
        * k.heads_per_rank * k.head_dim * 4;
    k.per_layer_bytes = k.recurrent_bytes_per_layer + 3 * k.ring_bytes_per_layer;
    const int64_t raw_slot = k.per_layer_bytes * n_lin;
    k.slot_bytes = (raw_slot + kLayoutAlign - 1) & ~(kLayoutAlign - 1);
    return k;
}

// ── Auto KV pages calculation ───────────────────────────────────────────────

static int compute_auto_kv_pages(const config::Config& cfg,
                                 int64_t bytes_per_page,
                                 int num_layers,
                                 int64_t available_for_kv,
                                 int kv_shard_factor) {
    // Needed pages: max_concurrent_requests * ceil(max_seq / page_size) * num_layers
    int page_size = cfg.memory.kv_cache.page_size_tokens;
    int max_seq = cfg.serving.max_sequence_length;
    int max_req = cfg.serving.max_concurrent_requests;

    int pages_per_seq = (max_seq + page_size - 1) / page_size;

    // TD-KV-REP-POOL-HALVED: per-rank page demand depends on dcp_kv_mode
    // (INV-KV-SIZE-SHARD).
    //  - replicated (default; kv_shard_factor == 1): every TP GPU claims the
    //    SAME page_idx for every logical page in lockstep (INV-KV-REP,
    //    allocate_main_replicated) — each rank holds the FULL per-sequence
    //    page count. No division.
    //  - sharded (kv_shard_factor == dcp_size): round-robin-by-chunk
    //    ownership (kv_shard_math, INV-4.9e) — size for the worst-case rank
    //    (rank 0), which owns ceil-at-chunk-granularity pages of each
    //    sequence: full_cycles*pages_per_chunk + min(rem, pages_per_chunk).
    //    A plain needed/dcp floor-divide underprovisions rank 0 for odd
    //    counts (same rationale as the indexer-local ceil-divide,
    //    TD-GLM-INDEXER-LOCAL-MERGE).
    if (kv_shard_factor > 1) {
        const int pages_per_chunk =
            std::max(1, cfg.memory.kv_cache.dcp_chunk_size / page_size);
        const int cycle_pages = pages_per_chunk * kv_shard_factor;
        const int full_cycles = pages_per_seq / cycle_pages;
        const int rem = pages_per_seq % cycle_pages;
        pages_per_seq = full_cycles * pages_per_chunk +
                        std::min(rem, pages_per_chunk);
    }

    int64_t needed = static_cast<int64_t>(max_req) * pages_per_seq * num_layers;

    // Cap so kv_bytes fits within available budget. The cap is per-GPU
    // PHYSICAL pages and is applied AFTER the per-rank shard division: a
    // VRAM-bound sharded pool still uses the full available budget (each
    // rank's pages hold only its owned tokens, so a full-VRAM pool per rank
    // is the dcp_size× context capacity Helix sharding exists to provide;
    // dividing the VRAM cap would forfeit it).
    if (bytes_per_page > 0) {
        int64_t max_by_vram = available_for_kv / bytes_per_page;
        if (needed > max_by_vram) needed = max_by_vram;
    }
    return static_cast<int>(std::max(needed, int64_t{0}));
}

// ── compute_vram_layout ─────────────────────────────────────────────────────

VramLayout compute_vram_layout(const config::Config& cfg,
                               const model::LayerRegistry& registry,
                               const model::ModelConfig& model_cfg) {
    VramLayout layout;
    const bool v4 = model_cfg.is_v4();
    if (v4) {
        // V4-3b: 3-tier KV.  Main/spec pools become the CSA bucket; HCA/SWA
        // buckets get their own regions; the indexer_k region is reused for
        // the Lightning-Indexer tier.
        layout.v4 = compute_v4_kv_layout(cfg, model_cfg);
        layout.kv_bytes_per_page = layout.v4.csa_bytes_per_page;
        layout.indexer_k_bytes_per_page = layout.v4.indexer_bytes_per_page;
        layout.kv_cache_format = layout.v4.csa_format;

        // ── V4 compatibility fail-closes (explicit, with TDs) ──
        // TD-V4-KVT RESOLVED (P3, 2026-08-21): memory.kv_tiering under V4
        // routes to the V4 CSA-bucket manager (daemon::V4KvTiering —
        // page-granular demote + selection-driven repromote; HCA/LID/SWA
        // exempt by policy). No layout change needed here: the CSA tier IS
        // kv_main and repromotes ride the unreserved-growth path.
        // V4-2c (2026-08-21): REPLICATED KV at tp >= 2 is supported — kMain
        // (CSA) rides the INV-KV-REP lockstep machinery and the kHca/kSwa/
        // kIndexerK side pools are per-GPU (the dispatcher provisions each
        // rank's pages independently). SHARDED KV stays closed: owner-
        // routing of compressed-entry pages is undesigned (TD-V4-DCP-KV).
        if (cfg.hardware.dcp_enabled &&
            static_cast<int>(cfg.hardware.tp_array.size()) >= 2 &&
            cfg.hardware.dcp_kv_mode == config::DcpKvMode::sharded) {
            throw std::invalid_argument(
                "deepseek_v4: sharded DCP KV is not supported — "
                "owner-routing of compressed-entry pages is undefined — "
                "TD-V4-DCP-KV (replicated KV is the V4 TP mode)");
        }
        // Explicit page counts are per-tier for V4 — a single scalar is
        // ambiguous.  Only auto sizing is supported.
        if (std::get_if<int>(&cfg.memory.kv_cache.max_pages_per_gpu)) {
            throw std::invalid_argument(
                "deepseek_v4: memory.kv_cache.max_pages_per_gpu must be "
                "\"auto\" (V4 KV is 3-tier; a single page count is "
                "ambiguous)");
        }
    } else {
        layout.kv_bytes_per_page = kv_bytes_per_page(model_cfg, cfg);
        layout.indexer_k_bytes_per_page = indexer_k_bytes_per_page(model_cfg, cfg);
        layout.kv_cache_format =
            (cfg.compute.attention_backend == config::AttentionBackendType::turboquant_mla)
            ? KvCacheFormat::kTurboQuantMse4
            : KvCacheFormat::kSnapMlaFp8;
    }

    // ── S1 shared-region slab geometry (TD-INDEXER-POOL-ELASTIC,
    //    RADIX_SLAB_DESIGN §1 rule 2) ──
    // slab = smallest whole-kMain-page multiple that holds one indexer page.
    // The indexer carve becomes indexer_k_pages slabs at stride slab_bytes,
    // slab-exactly abutting the kv_main page span in ONE shared region, so a
    // slab is an aligned run of pages_per_slab flat kMain page indices.
    if (layout.indexer_k_bytes_per_page > 0 && layout.kv_bytes_per_page > 0) {
        layout.pages_per_slab = static_cast<int>(
            (layout.indexer_k_bytes_per_page + layout.kv_bytes_per_page - 1)
            / layout.kv_bytes_per_page);
        layout.slab_bytes =
            static_cast<int64_t>(layout.pages_per_slab) *
            layout.kv_bytes_per_page;
        spdlog::info(
            "VramAllocator: S1 slab geometry — slab {} B = {} kMain pages x "
            "{} B (holds one {} B indexer page, sliver {} B/slab)",
            layout.slab_bytes, layout.pages_per_slab, layout.kv_bytes_per_page,
            layout.indexer_k_bytes_per_page,
            layout.slab_bytes - layout.indexer_k_bytes_per_page);
    }

    auto budgets = registry.estimate_gpu_budgets();
    int num_layers = registry.num_layers();
    // TD-GOLDEN-KV-SPEC: the dispatcher allocates one physical KV page per
    // (logical page, layer) including the MTP layer(s) (INV-KV-LAYER) — KV
    // pool sizing must count them too or full occupancy exhausts the pool.
    // TD-KV-POOL-SIZED-OVER-ALL-LAYERS (GF3.9): count only KV-BEARING
    // layers — glm5_next's 34 KDA linear layers keep no KV pages (sentinel
    // seq_pages_ slots, CommandDispatcher::kmain_layer_), so funding all 45
    // would over-carve the pool ~4x at the expert arena's expense. Uniform-
    // attention models are BYTE-IDENTICAL: num_kv_layers() ==
    // num_hidden_layers == registry.num_layers() there, a pure substitution.
    // The MTP layer is KV-bearing on every arch that reaches this line
    // (glm5_next MTP is sparse MLA — GF3.1 §9.3; V4 takes the v4 branch).
    const int kv_layers = model_cfg.num_kv_layers()
                        + model_cfg.raw().num_nextn_predict_layers;
    int64_t per_expert = registry.per_routed_expert_bytes();
    int experts_per_tok = model_cfg.raw().num_experts_per_tok;
    int64_t min_expert_cache = static_cast<int64_t>(experts_per_tok) * per_expert;

    // If no MoE layers, no expert cache needed
    if (registry.num_moe_layers() == 0) min_expert_cache = 0;

    // Build TP membership — non-TP GPUs don't do attention, skip KV cache.
    const auto& tp_arr = cfg.hardware.tp_array;
    std::vector<bool> in_tp(cfg.hardware.gpus.size(), false);
    for (int idx : tp_arr) {
        if (idx >= 0 && idx < static_cast<int>(cfg.hardware.gpus.size()))
            in_tp[static_cast<size_t>(idx)] = true;
    }
    bool has_tp = !tp_arr.empty();
    int tp_degree = static_cast<int>(tp_arr.size());

    // ── GF3.8: KDA per-request state slot geometry (glm5_next; disabled
    // — all zeros — on every other architecture, so the layout arithmetic
    // below is BYTE-IDENTICAL for GLM-5.2 / V4 / V3.2). spec_columns = 0:
    // GF3.11 owns the conv-ring widening seam (snapshot-per-round was the
    // recommended design; a replay ring would resize the slot at boot).
    layout.kda = compute_kda_state_layout(model_cfg, tp_degree,
                                          /*spec_columns=*/0);
    // TD-KDA-STATE-MAPPED-SLABS: mapped-state policy switch, config-first
    // (_internal-kda_state.mapped, DEFAULT TRUE since the 2026-08-31
    // user-approved promotion), env LS_KDA_STATE_MAPPED overrides either
    // way when set ('1' enables, anything else disables). Mapped means NO
    // dedicated carve below: per-(request, layer) whole-slab runs are
    // claimed from the shared kv_main slab region at admission. The carve
    // stays reachable as the OFF-path — the switch SURVIVES so the A/B is
    // re-runnable after any kernel change. Unslabbed models (no indexer
    // pool => no slab region) cannot map and FALL BACK to the carve with a
    // warning — never a boot failure on the default.
    if (layout.kda.enabled) {
        bool mapped = cfg._internal_kda_state.mapped;
        if (const char* e = std::getenv("LS_KDA_STATE_MAPPED"); e && *e)
            mapped = (e[0] == '1');
        if (mapped && layout.slab_bytes <= 0) {
            spdlog::warn(
                "KDA state: mapped mode requested (default-on) but this "
                "model is UNSLABBED (no indexer pool => no shared slab "
                "region) — falling back to the dedicated carve "
                "(TD-KDA-STATE-MAPPED-SLABS off-path).");
            mapped = false;
        }
        layout.kda.mapped = mapped;
    }

    // TD-72g: vocab_size must be divisible by TP degree for output head sharding.
    // To support indivisible vocab sizes, the fix requires: (1) last rank gets
    // vocab_size - local_vocab * (tp-1) rows in weight upload (engine.cpp),
    // (2) PinnedRegionLayout accounts for the larger last-rank slot,
    // (3) partial_logits_scratch_ sized for the larger last-rank chunk,
    // (4) NCCL allgatherv (variable send counts) instead of allgather,
    // (5) post-allgather transpose handles variable chunk sizes per rank,
    // (6) output head bias sharding matches. Alternative: pad vocab_size to
    // next multiple of tp at config level, zero-fill padding rows on-device
    // after weight upload.
    if (tp_degree >= 2 && model_cfg.raw().vocab_size % tp_degree != 0) {
        throw std::invalid_argument(
            "vocab_size (" + std::to_string(model_cfg.raw().vocab_size) +
            ") is not divisible by tensor_parallelism (" +
            std::to_string(tp_degree) + "). Output head weight sharding "
            "requires even division.");
    }

    // DCP sharding factor
    bool dcp = cfg.hardware.dcp_enabled && tp_degree >= 2;
    int dcp_shard_factor = dcp ? tp_degree : 1;
    // KV pages are per-rank-divided ONLY under sequence-sharded KV
    // (hardware.dcp_kv_mode = sharded, INV-KV-SIZE-SHARD). Under replicated
    // KV (default) each TP GPU claims the same page_idx in lockstep
    // (INV-KV-REP) — full pages per rank, no division.
    bool kv_sharded = dcp &&
        cfg.hardware.dcp_kv_mode == config::DcpKvMode::sharded;

    // DSA model info for indexer K sizing
    bool has_dsa = model_cfg.has_dsa();
    int indexer_k_page_size = cfg.memory.kv_cache.indexer_k_page_size_tokens;

    // Count DSA indexer-COMPUTING layers (IndexShare-aware, INV-KVT-14).
    // The dispatcher provisions Pool::kIndexerK pages only for computing
    // layers — IndexShare full ∪ layer 0, the exact ensure_indexer_pages
    // rule; SHARED layers (including MTP layers, which are shared by
    // construction) reuse the preceding full layer's selection and never
    // allocate an indexer-K page. Sizing the pool for every layer
    // over-provisioned it ~3.8× on IndexShare models (GLM-5.2: 79 vs 21
    // layers = 10.9 GB vs 2.9 GB per TP GPU at max_sequence_length=1M),
    // starving the KV pool + expert cache. Non-IndexShare DSA models
    // (index_topk_freq<=0, e.g. V3.2): every hidden layer is full and MTP
    // layers still never compute — count == num_hidden_layers, unchanged.
    int num_dsa_layers = 0;
    if (has_dsa) {
        const int n_idx_layers = model_cfg.raw().num_hidden_layers
                               + model_cfg.raw().num_nextn_predict_layers;
        for (int l = 0; l < n_idx_layers; ++l)
            if (model_cfg.computes_indexer(l)) ++num_dsa_layers;
    }

    // Prefill scratch pre-allocation: cap by ceiling-based naive scratch need
    int64_t prefill_scratch_configured = config::vram_gb_to_bytes(
        cfg.memory.kv_cache.prefill_scratch_preallocated_gb);

    // Naive context ceiling only applies to non-DSA models. DSA models use
    // absorbed sparse prefill (workspace sized differently), not naive
    // decompression. For non-DSA: if ceiling is set, compute scratch needed
    // for that token count. Scratch beyond that is wasted (absorbed uses zero
    // scratch for contexts > ceiling).
    int64_t prefill_scratch_gb_to_bytes = prefill_scratch_configured;
    // For DSA: no ceiling-based cap on spill zone (use INT64_MAX as sentinel)
    int64_t spill_scratch_needed = INT64_MAX;

    // V4: no MLA kv_b decompression exists — naive-scratch math over the
    // inert MLA defaults would be meaningless.  Keep the configured Tier-1
    // scratch and the DSA-style uncapped spill (V4 prefill scratch demand is
    // ticket V4-5 territory).
    if (!has_dsa && !v4) {
        const auto& ceiling_opt = cfg.memory.kv_cache.naive_prefill_context_ceiling;
        int naive_ceiling_tokens = ceiling_opt.has_value()
            ? ceiling_opt.value()
            : cfg.serving.max_sequence_length;

        // DCP divides per-rank naive scratch demand in BOTH dcp_kv_mode
        // settings (TD-KV-DCP-SCRATCH-CEILING, verified; INV-KV-NAIVE-SCRATCH):
        // naive_prefill_scratch_bytes uses the FULL head count, while the
        // real per-rank decompression demand is
        //   - replicated KV: H/tp heads (kv_b_proj is column-TP-sharded,
        //     INV-MLA-1 — a rank can only decompress its own head shard)
        //     × the FULL sequence (every rank holds all tokens, INV-KV-REP)
        //     = (H × T) / tp;
        //   - sharded KV: ALL H heads (post Q-allgather, INV-KVS-QAG)
        //     × the rank's T/dcp token shard = (H × T) / dcp.
        // Both equal full-H × (T / dcp_shard_factor) while dcp_size ==
        // tp_degree, so the division keys on dcp alone — unlike KV page
        // demand, which divides only under sharded mode (INV-KV-SIZE-SHARD).
        int local_ceiling = dcp ? (naive_ceiling_tokens / dcp_shard_factor)
                                : naive_ceiling_tokens;
        int64_t naive_scratch_needed = naive_prefill_scratch_bytes(
            model_cfg, local_ceiling);

        // Cap pre-allocated scratch to what naive actually needs
        prefill_scratch_gb_to_bytes = std::min(prefill_scratch_configured,
                                                 naive_scratch_needed);
        if (prefill_scratch_gb_to_bytes < 0) prefill_scratch_gb_to_bytes = 0;

        // Remaining scratch need after Tier 1 → caps the spill zone
        spill_scratch_needed = std::max(int64_t{0},
            naive_scratch_needed - prefill_scratch_gb_to_bytes);
    }

    // Streaming spill fraction
    // TD-91e: clamp to [0.0, 1.0] — same pattern as stable_zone_fraction (TD-91c).
    double spill_frac = std::clamp(
        cfg.memory.kv_cache.streaming_spill_fraction, 0.0, 1.0);
    if (spill_frac != cfg.memory.kv_cache.streaming_spill_fraction) {
        spdlog::warn("VramAllocator: streaming_spill_fraction {:.3f} outside "
                     "[0.0, 1.0], clamped to {:.3f}",
                     cfg.memory.kv_cache.streaming_spill_fraction, spill_frac);
    }

    // ── TD-INDEXER-POOL-EVICT / INV-KVT-14b: indexer-K CONCURRENCY ───────
    // How many concurrent sequences the kIndexerK pool covers, decided ONCE
    // for every TP GPU. Two reasons it is not a per-GPU decision:
    //   (a) replicated indexer pages are claimed on every TP GPU in lockstep
    //       for the SAME logical page, so an asymmetric pool has the capacity
    //       of its SMALLEST rank and wastes the difference on the others;
    //   (b) the KV pool is the RESIDUAL of this carve — every indexer byte is
    //       taken 1:1 from KV. On a pinned-weight-heavy rank that residual is
    //       tiny (measured 2026-08-25: GPU 2 had 86 MiB of KV+spec left, so an
    //       unbounded `* max_concurrent_requests` drove kv_main to ZERO pages).
    // So: spend at most a QUARTER of the residual a one-sequence pool would
    // have left for KV, on the tightest rank, and never go below one sequence
    // (the historical size — a max-length sequence must stay serveable).
    int indexer_seqs = 1;
    if (has_dsa && indexer_k_page_size > 0 && num_dsa_layers > 0) {
        int idx_pages_per_seq = (cfg.serving.max_sequence_length
                                 + indexer_k_page_size - 1)
                                / indexer_k_page_size;
        if (cfg.hardware.dcp_indexer_mode == config::DcpIndexerMode::local
            && dcp_shard_factor > 1)
            idx_pages_per_seq =
                (idx_pages_per_seq + dcp_shard_factor - 1) / dcp_shard_factor;
        // S1: the physical cost of an indexer page is one SLAB (page bytes +
        // per-slab sliver) — account the true footprint so the INV-KVT-14b
        // quarter-residual math stays exact under the shared region.
        const int64_t idx_page_stride = layout.slab_bytes > 0
            ? layout.slab_bytes : layout.indexer_k_bytes_per_page;
        const int64_t per_seq_bytes =
            static_cast<int64_t>(idx_pages_per_seq) * num_dsa_layers
            * idx_page_stride;
        const int max_req = std::max(1, cfg.serving.max_concurrent_requests);
        int64_t extra_seqs = -1;   // min over TP GPUs
        for (size_t g = 0; g < budgets.size() && per_seq_bytes > 0; ++g) {
            if (has_tp && !in_tp[g]) continue;
            const auto& hw = cfg.hardware.gpus[g];
            double margin_gb = cfg.memory.vram_safety_margin_gb;
            if (hw.vram_allocation_gb.has_value()
                && hw.vram_allocation_gb->safety_margin_gb >= 0.0)
                margin_gb = hw.vram_allocation_gb->safety_margin_gb;
            int64_t pinned = budgets[g].pinned_bytes;
            if (hw.vram_allocation_gb.has_value()
                && hw.vram_allocation_gb->resident > 0.0)
                pinned = std::max(pinned, config::vram_gb_to_bytes(
                                              hw.vram_allocation_gb->resident));
            int64_t expert_reserve = min_expert_cache;
            if (hw.vram_allocation_gb.has_value()
                && hw.vram_allocation_gb->expert_streaming > 0.0)
                expert_reserve = std::max(
                    expert_reserve,
                    config::vram_gb_to_bytes(
                        hw.vram_allocation_gb->expert_streaming));
            const int64_t residual = budgets[g].total_vram_bytes
                                   - config::vram_gb_to_bytes(margin_gb)
                                   - align_region(pinned)
                                   - expert_reserve
                                   - per_seq_bytes;
            const int64_t afford =
                residual > 0 ? (residual / 4) / per_seq_bytes : 0;
            extra_seqs = extra_seqs < 0 ? afford : std::min(extra_seqs, afford);
        }
        if (extra_seqs < 0) extra_seqs = 0;
        indexer_seqs = static_cast<int>(
            std::min<int64_t>(max_req, 1 + extra_seqs));
        if (indexer_seqs < max_req)
            spdlog::warn(
                "Indexer-K sizing: VRAM affords {} concurrent max-length "
                "sequence(s), below serving.max_concurrent_requests={} "
                "({:.1f} MiB/seq). S4: this is a shared-pool SIZING share, "
                "no longer a fixed carve — shorter requests reserve less "
                "and extra demand may draw on free KV slabs, but concurrent "
                "max-length requests still churn prefix-holder evictions "
                "(retryable, TD-INDEXER-POOL-EVICT); lower "
                "max_sequence_length or free VRAM (pinned weights / "
                "expert_streaming reserve) to raise it.",
                indexer_seqs, max_req,
                static_cast<double>(per_seq_bytes) / (1024.0 * 1024.0));
    }

    // ── V4 prefix-holder side-tier budget (INV-PREFIX-CACHE-3) ──
    // Prefix-cache holder cost is ARCHITECTURE-DEPENDENT. On DSA/GLM a
    // retained holder shares kMain pages by refcount and materializes one
    // CoW indexer frontier page group — holders are nearly free, so the
    // pools above size for max_concurrent_requests only. On V4 the side
    // tiers (kSwa/kHca/kIndexerK-LID) are COPY-ON-FORK (TD-V4-SERVE-PREFIX:
    // mutate-in-place rings, sharing would corrupt the parent), so EVERY
    // retained holder owns a complete side-tier set — a full extra
    // sequence's worth of side pages (~45 MiB/GPU at 32k on the serve
    // profile). Size the V4 side pools for max_concurrent_requests
    // in-flight sequences PLUS the configured holder budget; the
    // evict-retry seam (TD-INDEXER-POOL-EVICT) stays the pressure valve,
    // not the steady-state mechanism. kMain (CSA) is NOT holder-scaled:
    // it is refcount-shared and chain-deduplicated, and its holder
    // pressure stays soft (serving.prefix_cache.max_cached_tokens budget
    // + evict-on-exhaustion).
    const int v4_prefix_holders =
        (v4 && cfg.serving.prefix_cache.enabled)
            ? cfg.serving.prefix_cache.max_entries : 0;

    // ── GF3.8: KDA state pool concurrency policy ──
    // One slot per live sequence. Prefix holders own FULL slot copies (a
    // frozen holder never steps, but the live parent it forked from keeps
    // mutating ITS state, so the copy is unavoidable — INV-PREFIX-CACHE-3
    // third cost class), so the policy target is in-flight requests PLUS
    // the configured holder budget — the INV-KVT-14b boot-sizing contract
    // applied to the pool that actually clamps concurrency here.
    const int kda_policy_slots =
        std::max(1, cfg.serving.max_concurrent_requests)
        + (cfg.serving.prefix_cache.enabled
               ? cfg.serving.prefix_cache.max_entries : 0);
    int v4_holder_tokens = cfg.serving.max_sequence_length;
    if (cfg._internal_prefix_cache.max_entry_tokens > 0)
        v4_holder_tokens = std::min(
            v4_holder_tokens, cfg._internal_prefix_cache.max_entry_tokens);

    for (size_t gpu_i = 0; gpu_i < budgets.size(); ++gpu_i) {
        auto& budget = budgets[gpu_i];
        GpuVramLayout gpu{};
        gpu.gpu_id = budget.gpu_id;
        gpu.total_vram_bytes = budget.total_vram_bytes;
        // TD-KDA-MAPPED-NONTP-GPUS: THE attention-participation predicate,
        // computed ONCE and published in the layout so the PageAllocator
        // gates its preconditions on the very same bit the sizing used.
        // Expert-only (non-TP) GPUs host no attention tenant — no KV pool,
        // no indexer share, no KDA state (mapped or carved) — and so have a
        // legitimately EMPTY kMain.
        gpu.attention_host = (!has_tp || in_tp[gpu_i]);
        // Safety margin: per-GPU vram_allocation_gb.safety_margin_gb override
        // (>= 0 sentinel; -1 = unset) wins over the global
        // memory.vram_safety_margin_gb. The margin is the VRAM left
        // UNALLOCATED (CUDA context, pinned-arena page tables ~3 MiB/GiB,
        // late driver allocations) — a per-GPU shave recovers expert-window
        // bytes on ONE card without growing the physical carve on the others
        // (allocated_bytes = total - margin is device_alloc'd per GPU).
        const auto& hw_gpu = cfg.hardware.gpus[gpu_i];
        double margin_gb = cfg.memory.vram_safety_margin_gb;
        if (hw_gpu.vram_allocation_gb.has_value() &&
            hw_gpu.vram_allocation_gb->safety_margin_gb >= 0.0) {
            margin_gb = hw_gpu.vram_allocation_gb->safety_margin_gb;
            spdlog::info("VramAllocator: GPU {} per-GPU safety margin "
                         "{:.2f} GB (global {:.2f} GB)",
                         gpu_i, margin_gb, cfg.memory.vram_safety_margin_gb);
        }
        gpu.safety_margin_bytes = config::vram_gb_to_bytes(margin_gb);
        gpu.dcp_kv_shard_factor =
            (in_tp[gpu_i] && kv_sharded) ? dcp_shard_factor : 1;

        // Pinned bytes: use per-GPU override if present, else LayerRegistry value.
        // TD-54c: clamp override to at least the computed layout requirement.
        if (hw_gpu.vram_allocation_gb.has_value() &&
            hw_gpu.vram_allocation_gb->resident > 0.0) {
            const int64_t override_bytes = config::vram_gb_to_bytes(
                hw_gpu.vram_allocation_gb->resident);
            if (override_bytes < budget.pinned_bytes) {
                spdlog::warn("VramAllocator: GPU {} resident override ({:.2f} GB = {} B) "
                             "is smaller than computed pinned layout ({} B). "
                             "Clamping to layout minimum.",
                             gpu_i,
                             hw_gpu.vram_allocation_gb->resident,
                             override_bytes, budget.pinned_bytes);
            }
            gpu.pinned_bytes = align_region(
                std::max(override_bytes, budget.pinned_bytes));
        } else {
            gpu.pinned_bytes = align_region(budget.pinned_bytes);
        }

        // Indexer K sizing.
        // S4 (TD-INDEXER-POOL-ELASTIC): on slabbed models there is NO boot
        // carve any more — indexer-K slabs are claimed from the SHARED slab
        // pool on demand (admission reservation, INV-DSA-RESERVE). The
        // demand computed here becomes the shared pool's indexer SHARE
        // (`indexer_share_*`): it is added to the kv_main span so total
        // capacity is byte-identical to the old partition, but the
        // indexer/KV boundary inside it no longer exists. The legacy fixed
        // carve survives only on non-slabbed layouts (no shared region).
        int64_t indexer_share_bytes = 0;   // slab-priced, folded into kv_main
        int indexer_share_slabs = 0;       // (mirrored into gpu.indexer_share_*
                                           //  after the branches below)
        // V4: Lightning-Indexer tier — CSA layers only, per TP GPU, sized for
        // max_concurrent_requests full-length sequences (entries are per
        // compressed block, so pages are cheap: ~270 KB per 8192 tokens).
        if (v4 && gpu.attention_host) {
            const int max_seq = cfg.serving.max_sequence_length;
            const int max_req = cfg.serving.max_concurrent_requests;
            const int pages_per_seq =
                (max_seq + indexer_k_page_size - 1) / indexer_k_page_size;
            // + full LID sets for the prefix-holder budget (COPY-ON-FORK,
            // INV-PREFIX-CACHE-3): holders are capped at v4_holder_tokens.
            const int holder_pages_per_seq =
                (v4_holder_tokens + indexer_k_page_size - 1)
                / indexer_k_page_size;
            gpu.indexer_k_pages =
                pages_per_seq * layout.v4.num_csa_layers * max_req
                + holder_pages_per_seq * layout.v4.num_csa_layers
                      * v4_prefix_holders;
            if (layout.slab_bytes > 0) {
                // S4 elastic: LID demand becomes the shared-pool share.
                indexer_share_slabs = gpu.indexer_k_pages;
                indexer_share_bytes =
                    static_cast<int64_t>(indexer_share_slabs)
                    * layout.slab_bytes;
                gpu.indexer_k_pages = 0;
                gpu.indexer_k_bytes = 0;
                gpu.indexer_k_pad_bytes = 0;
            } else {
                gpu.indexer_k_bytes =
                    static_cast<int64_t>(gpu.indexer_k_pages)
                    * layout.indexer_k_bytes_per_page;
                gpu.indexer_k_pad_bytes =
                    (kLayoutAlign - gpu.indexer_k_bytes % kLayoutAlign)
                    % kLayoutAlign;
            }
        } else
        // DSA TP GPUs only.
        // Replicated mode (default): full context pages per GPU.
        // Local mode with DCP: pages divided by dcp_shard_factor.
        if (has_dsa && gpu.attention_host) {
            int max_seq = cfg.serving.max_sequence_length;
            // TD-INDEXER-POOL-EVICT (INV-KVT-14b): the pool must cover
            // serving.max_concurrent_requests CONCURRENT sequences, exactly
            // like the V4 kIndexerK/LID branch above. Sizing it for ONE
            // sequence made the pool a hard per-sequence wall the moment a
            // second live sequence existed — and serving ALWAYS has more
            // than one: every prefix-cache holder is a live sequence that
            // pins a CoW frontier page GROUP (num_dsa_layers pages). The
            // 2026-08-24 incident: 84 pages/GPU (= 4 × 21 × 1 seq) against
            // 6 live holders wanting 21 each ⇒ ensure_indexer_pages
            // exhausted at a forked 25k prefix ⇒ silent dense downgrade ⇒
            // full cold-page re-promotion ⇒ fail-closed CMP_ERROR.
            // Holders BEYOND this budget stay reclaimable, not fatal: the
            // exhaustion is now a retryable error the orchestrator answers
            // by evicting a holder (TD-INDEXER-POOL-EVICT), so this sizes
            // the WORKING set (in-flight requests), not the cache.
            int pages_per_seq = (max_seq + indexer_k_page_size - 1) /
                                indexer_k_page_size;
            if (cfg.hardware.dcp_indexer_mode == config::DcpIndexerMode::local &&
                dcp_shard_factor > 1) {
                // Round-robin by indexer page (TD-GLM-INDEXER-LOCAL-MERGE):
                // rank 0 owns ceil(pages/dcp) pages — ceil-divide so an odd
                // page count doesn't underprovision the first rank's pool.
                pages_per_seq =
                    (pages_per_seq + dcp_shard_factor - 1) / dcp_shard_factor;
            }
            // `indexer_seqs` is the VRAM-affordable concurrency decided
            // above, uniform across TP GPUs (see the pre-loop block).
            const int seqs = indexer_seqs;
            const int demand_pages =
                static_cast<int>(pages_per_seq) * num_dsa_layers * seqs;
            if (layout.slab_bytes > 0) {
                // S4 elastic (INV-KVT-14b restated): the quarter-residual
                // demand is a shared-pool SIZING SHARE, not a carve — the
                // slabs live in the one shared pool and admission
                // reservation (INV-DSA-RESERVE) is the policy that claims
                // them per request. A request set that needs MORE than the
                // share can take it (up to the pool, KV pressure permitting)
                // instead of hitting the old fixed wall.
                indexer_share_slabs = demand_pages;
                indexer_share_bytes = static_cast<int64_t>(demand_pages)
                                      * layout.slab_bytes;
                gpu.indexer_k_pages = 0;
                gpu.indexer_k_bytes = 0;
                gpu.indexer_k_pad_bytes = 0;
                spdlog::info(
                    "Indexer-K on GPU {} (S4 ELASTIC): no boot carve — "
                    "indexer slabs are claimed from the shared slab pool at "
                    "admission (INV-DSA-RESERVE). Pool sized with an indexer "
                    "share of {} slabs ({:.1f} MiB) = {} pages/seq x {} "
                    "computing layers x {} sequence(s) at "
                    "max_sequence_length={}; demand beyond the share draws "
                    "on free KV slabs, and exhaustion is the RETRYABLE "
                    "admission refusal answered by holder eviction "
                    "(TD-INDEXER-POOL-EVICT).",
                    gpu_i, demand_pages,
                    static_cast<double>(indexer_share_bytes)
                        / (1024.0 * 1024.0),
                    pages_per_seq, num_dsa_layers, seqs, max_seq);
            } else {
                gpu.indexer_k_pages = demand_pages;
                gpu.indexer_k_bytes =
                    static_cast<int64_t>(gpu.indexer_k_pages)
                    * layout.indexer_k_bytes_per_page;
                gpu.indexer_k_pad_bytes =
                    (kLayoutAlign - gpu.indexer_k_bytes % kLayoutAlign)
                    % kLayoutAlign;
                spdlog::info(
                    "Indexer-K pool on GPU {}: {} pages ({:.1f} MiB) = {} "
                    "pages/seq x {} computing layers x {} sequence(s) "
                    "(max_sequence_length={}; unslabbed layout — legacy "
                    "fixed carve). A live sequence beyond this budget — "
                    "prefix-cache holders included — trips a RETRYABLE "
                    "exhaustion answered by holder eviction "
                    "(TD-INDEXER-POOL-EVICT).",
                    gpu_i, gpu.indexer_k_pages,
                    static_cast<double>(gpu.indexer_k_bytes)
                        / (1024.0 * 1024.0),
                    pages_per_seq, num_dsa_layers, seqs, max_seq);
            }
        } else {
            gpu.indexer_k_pages = 0;
            gpu.indexer_k_bytes = 0;
            gpu.indexer_k_pad_bytes = 0;
        }

        gpu.indexer_share_slabs = indexer_share_slabs;
        gpu.indexer_share_bytes = indexer_share_bytes;

        // Available for KV + expert cache (after pinned + safety + indexer_k)
        int64_t available = gpu.total_vram_bytes - gpu.pinned_bytes -
                            gpu.safety_margin_bytes - gpu.indexer_k_bytes -
                            gpu.indexer_k_pad_bytes;

        // TD-MOE-EXPERT-WINDOW: expert-cache reservation carved out BEFORE
        // KV sizing. When the per-GPU vram_allocation_gb.expert_streaming
        // override (TOTAL expert cache, GB) is set, the serving-demand-driven
        // KV pool must leave that much room — previously the auto KV pool
        // consumed everything down to the bare top-K minimum
        // (min_expert_cache = experts_per_tok slots ≈ 211 MiB on GLM-5.2),
        // which (a) let the override overflow the physical block
        // (Σregions > allocation) and (b) squeezed the prefill expert window
        // to ~4 stable slots/GPU → 42 tiny streaming waves per MoE layer.
        int64_t expert_reserve = min_expert_cache;
        if (hw_gpu.vram_allocation_gb.has_value() &&
            hw_gpu.vram_allocation_gb->expert_streaming > 0.0) {
            expert_reserve = std::max(expert_reserve, config::vram_gb_to_bytes(
                hw_gpu.vram_allocation_gb->expert_streaming));
        }

        // ── GF3.8: KDA per-request state pool carve (glm5_next only) ──
        // A dedicated contiguous region: the GF3.7 kernels address the
        // state as base + slot * stride, so the pool must be ONE uniform-
        // stride span (a whole-slab elastic tenant cannot be, RADIX_SLAB_
        // DESIGN §1 GF3.8 amendment). The state does NOT grow with context
        // — capacity is a pure SLOT COUNT, checked once at admission
        // (retryable refusal), never mid-request.
        int64_t kda_mapped_share_bytes = 0;  // TD-KDA-STATE-MAPPED-SLABS
        if (layout.kda.enabled && layout.kda.mapped
            && gpu.attention_host) {
            // TD-KDA-STATE-MAPPED-SLABS: no carve — the state is a mapped
            // tenant of the shared slab region (per-(request, layer)
            // contiguous whole-slab runs, claimed at admission with the
            // same retryable-refusal seam). The former carve bytes stay in
            // `available` and flow into kv_main sizing below — the
            // capacity payoff this switch exists to measure. Requires slab
            // geometry (the run unit IS the slab).
            if (layout.slab_bytes <= 0) {
                // Unreachable: the policy resolution above falls back to
                // the carve on unslabbed models. Defensive only.
                throw std::logic_error(
                    "GPU " + std::to_string(gpu.gpu_id)
                    + ": mapped KDA state reached sizing on an unslabbed "
                      "model — policy fallback failed");
            }
            const int64_t unit_slabs =
                (layout.kda.per_layer_bytes + layout.slab_bytes - 1)
                / layout.slab_bytes;
            // The shared region must CARRY the state demand: KV sizing is
            // demand-capped on many configs, so without this share a
            // mapped boot could admit ZERO sequences. Same policy count
            // as the carve — but as SHARED slab capacity: lent to
            // KV/indexer whenever state demand runs below policy (the
            // ticket's payoff), VRAM-clamped exactly like everything else.
            kda_mapped_share_bytes = kda_policy_slots
                * static_cast<int64_t>(layout.kda.num_layers)
                * unit_slabs * layout.slab_bytes;
            spdlog::info(
                "KDA state pool on GPU {}: MAPPED over the shared slab "
                "region (no dedicated carve; TD-KDA-STATE-MAPPED-SLABS). "
                "Per-request demand = {} linear layers x {} slabs "
                "({:.2f} MiB/layer unit, {:.1f} MiB/request incl. slab "
                "padding vs {:.1f} MiB slot); claimed at seq_create from "
                "the shared free-slab list (contiguous per-layer runs), "
                "freed at seq_free; capacity is DEMAND-sized and shared "
                "with KV/indexer — exhaustion stays the retryable "
                "admission refusal. Tenant segregation ON "
                "(TD-KDA-MAPPED-FRAG b2): state claims scan TOP-DOWN, "
                "KV/indexer pop the free-slab LIFO from the region "
                "bottom, freed state runs return to the free-list "
                "BOTTOM; refusals are counted and classified "
                "contiguity-vs-capacity in the message.",
                gpu_i, layout.kda.num_layers, unit_slabs,
                static_cast<double>(layout.kda.per_layer_bytes)
                    / (1024.0 * 1024.0),
                static_cast<double>(unit_slabs * layout.slab_bytes
                                    * layout.kda.num_layers)
                    / (1024.0 * 1024.0),
                static_cast<double>(layout.kda.slot_bytes)
                    / (1024.0 * 1024.0));
        } else if (layout.kda.enabled && gpu.attention_host) {
            const int64_t resid = available - expert_reserve
                                  - indexer_share_bytes;
            if (resid < layout.kda.slot_bytes) {
                throw std::runtime_error(
                    "GPU " + std::to_string(gpu.gpu_id)
                    + ": insufficient VRAM for even ONE KDA state slot ("
                    + std::to_string(layout.kda.slot_bytes)
                    + " B/request; residual after pinned/margin/expert/"
                      "indexer = " + std::to_string(resid)
                    + " B). A glm5_next engine cannot serve any sequence "
                      "without its recurrent state — reduce pinned layers "
                      "or the expert reserve.");
            }
            // Never let the state pool starve KV/experts: cap at half the
            // residual, floor one slot (a max-length sequence must stay
            // serveable — the INV-KVT-14b clamp style).
            const int64_t afford = std::max<int64_t>(
                1, (resid / 2) / layout.kda.slot_bytes);
            gpu.kda_state_slots = static_cast<int>(
                std::min<int64_t>(kda_policy_slots, afford));
            gpu.kda_state_bytes =
                static_cast<int64_t>(gpu.kda_state_slots)
                * layout.kda.slot_bytes;
            spdlog::info(
                "KDA state pool on GPU {}: DEDICATED CARVE (mapped state "
                "OFF — the TD-KDA-STATE-MAPPED-SLABS A/B off-path, or an "
                "unslabbed model): {} slots x {:.1f} MiB "
                "({:.1f} MiB total) — per-REQUEST recurrent+conv state "
                "({} linear layers x {:.2f} MiB fp32), claimed at "
                "seq_create, freed at seq_free; achievable concurrency = "
                "{} live sequences (in-flight + prefix holders; GF3.12: "
                "hibernated holders spill their slot to host RAM and "
                "return it, so steady-state holders cost ~0 slots — the "
                "+max_entries term is transient/no-hibernate headroom).",
                gpu_i, gpu.kda_state_slots,
                static_cast<double>(layout.kda.slot_bytes) / (1024.0 * 1024.0),
                static_cast<double>(gpu.kda_state_bytes) / (1024.0 * 1024.0),
                layout.kda.num_layers,
                static_cast<double>(layout.kda.per_layer_bytes)
                    / (1024.0 * 1024.0),
                gpu.kda_state_slots);
            if (gpu.kda_state_slots < kda_policy_slots) {
                spdlog::warn(
                    "KDA state pool on GPU {} affords {} slots, below "
                    "serving.max_concurrent_requests={} + "
                    "prefix_cache.max_entries={} = {} (VRAM-clamped; "
                    "INV-KVT-14b/INV-PREFIX-CACHE-3 boot-sizing contract). "
                    "Excess demand trips a RETRYABLE kKvPoolExhausted "
                    "admission refusal answered by holder eviction — expect "
                    "eviction churn; lower max_concurrent_requests/"
                    "max_entries or free VRAM to fix.",
                    gpu_i, gpu.kda_state_slots,
                    cfg.serving.max_concurrent_requests,
                    cfg.serving.prefix_cache.enabled
                        ? cfg.serving.prefix_cache.max_entries : 0,
                    kda_policy_slots);
            }
            available -= gpu.kda_state_bytes;
        }

        // Resolve max_kv_pages — non-TP GPUs get zero (they don't do attention)
        const auto& kv_cfg = cfg.memory.kv_cache;
        int64_t scratch_bytes = 0;
        if (v4) {
            // ── V4-3b: 3-bucket page demand, auto-sized, proportional
            //    scale-down under the VRAM cap ──
            const auto& v4l = layout.v4;
            int64_t csa_pages = 0, hca_pages = 0, swa_pages = 0, spec_pages = 0;
            if (gpu.attention_host) {
                const int max_seq = cfg.serving.max_sequence_length;
                const int max_req = cfg.serving.max_concurrent_requests;
                const int64_t blocks_per_seq =
                    (max_seq + v4l.logical_block_tokens - 1) /
                    v4l.logical_block_tokens;
                // Prefix holders own FULL side-tier sets (copy-on-fork,
                // INV-PREFIX-CACHE-3) capped at v4_holder_tokens; kMain
                // (csa_pages) is refcount-shared and NOT holder-scaled.
                const int64_t holder_blocks =
                    (v4_holder_tokens + v4l.logical_block_tokens - 1) /
                    v4l.logical_block_tokens;
                csa_pages = static_cast<int64_t>(max_req) * blocks_per_seq *
                            v4l.num_csa_layers;
                hca_pages = static_cast<int64_t>(max_req) * blocks_per_seq *
                                v4l.num_hca_layers +
                            static_cast<int64_t>(v4_prefix_holders) *
                                holder_blocks * v4l.num_hca_layers;
                // SWA/raw tier: window + compressor residual per layer
                // (incl. nextn MTP layers, SWA-only).
                int64_t swa_per_seq = 0;
                const auto& m = model_cfg.raw();
                for (int l = 0; l < m.num_hidden_layers; ++l) {
                    swa_per_seq += v4_swa_pages_per_layer(
                        v4l, model_cfg.attention_type_for_layer(l));
                }
                swa_per_seq += static_cast<int64_t>(
                                   m.num_nextn_predict_layers) *
                               v4_swa_pages_per_layer(
                                   v4l, model::V4AttentionType::kSwa);
                swa_pages = static_cast<int64_t>(max_req + v4_prefix_holders)
                            * swa_per_seq;
                // Speculation pool: CSA-page-size sibling of the main pool
                // (the page machinery requires main/spec pages equal-sized).
                spec_pages = static_cast<int64_t>(std::floor(
                    static_cast<double>(csa_pages) *
                    kv_cfg.speculation_pool_fraction));

                // VRAM cap: scale ALL buckets proportionally.
                // S4 elastic: the LID share comes off the top exactly like
                // the former carve did (byte-identical partition), then
                // rejoins the CSA span as whole slabs after quantization.
                int64_t scratch_budget = prefill_scratch_gb_to_bytes;
                int64_t available_for_kv = available - expert_reserve -
                                           scratch_budget -
                                           indexer_share_bytes;
                if (available_for_kv < 0) available_for_kv = 0;
                const int64_t demand =
                    (csa_pages + spec_pages) * v4l.csa_bytes_per_page +
                    hca_pages * v4l.hca_bytes_per_page +
                    swa_pages * v4l.swa_bytes_per_page;
                if (demand > available_for_kv && demand > 0) {
                    const double scale =
                        static_cast<double>(available_for_kv) /
                        static_cast<double>(demand);
                    csa_pages = static_cast<int64_t>(csa_pages * scale);
                    hca_pages = static_cast<int64_t>(hca_pages * scale);
                    swa_pages = static_cast<int64_t>(swa_pages * scale);
                    spec_pages = static_cast<int64_t>(spec_pages * scale);
                    // TD-V4-KMAIN-SIZING fail-loud: a scaled pool cannot
                    // serve serving.max_sequence_length — say so AT BOOT
                    // with the achievable bound instead of a first-request
                    // seq_create surprise.
                    const int64_t serveable_blocks =
                        (max_req > 0 && v4l.num_csa_layers > 0)
                            ? csa_pages / (static_cast<int64_t>(max_req) *
                                           v4l.num_csa_layers)
                            : 0;
                    spdlog::warn(
                        "V4 KV auto-size on GPU {}: demand {:.1f} MiB > "
                        "available {:.1f} MiB — tier pools scaled by {:.3f}. "
                        "kv_main now serves ~{} tokens/seq at "
                        "max_concurrent_requests={} (max_sequence_length={} "
                        "needs {} blocks/seq); longer prompts will fail "
                        "seq_create. Lower max_sequence_length/"
                        "max_concurrent_requests or free VRAM.",
                        gpu_i, static_cast<double>(demand) / (1024.0 * 1024.0),
                        static_cast<double>(available_for_kv)
                            / (1024.0 * 1024.0),
                        scale,
                        serveable_blocks * v4l.logical_block_tokens, max_req,
                        max_seq, blocks_per_seq);
                }

                // S1: quantize the CSA (kMain) page span DOWN to whole slabs
                // so the shared region [indexer slabs | kMain pages] is an
                // exact slab grid (loses < one slab of pages; the bytes fall
                // to scratch/expert below).
                if (layout.pages_per_slab > 1 && csa_pages > 0) {
                    const int64_t drop = csa_pages % layout.pages_per_slab;
                    if (drop > 0) {
                        spdlog::info(
                            "V4 KV auto-size on GPU {}: S1 slab quantization "
                            "drops {} of {} CSA pages (span = whole slabs)",
                            gpu_i, drop, csa_pages);
                        csa_pages -= drop;
                    }
                }
                // S4 elastic: fold the LID share into the CSA (kMain) span
                // as whole slabs — LID tier pages claim whole slabs from
                // this shared span at runtime (ensure_v4_tier_pages).
                if (indexer_share_slabs > 0 && layout.pages_per_slab > 0)
                    csa_pages += static_cast<int64_t>(indexer_share_slabs)
                                 * layout.pages_per_slab;

                // Prefill scratch at the tail of kv_main (CSA region), same
                // placement contract as the MLA path.
                int64_t kv_bytes_now =
                    (csa_pages + spec_pages) * v4l.csa_bytes_per_page +
                    hca_pages * v4l.hca_bytes_per_page +
                    swa_pages * v4l.swa_bytes_per_page;
                int64_t max_scratch = available - kv_bytes_now - expert_reserve;
                if (max_scratch < 0) max_scratch = 0;
                scratch_bytes =
                    std::min(prefill_scratch_gb_to_bytes, max_scratch);
                if (scratch_bytes < 0) scratch_bytes = 0;

                // ── Fail-loud holder accounting (INV-PREFIX-CACHE-3) ──
                // Say AT BOOT how many copy-on-fork prefix holders the
                // side-tier pools actually carry beyond the in-flight
                // working set — a user must not discover a holder-vs-pool
                // mismatch from a mid-session CMP_ERROR (2026-08-26
                // incident: max_entries=8 against pools carved for
                // max_concurrent_requests=2).
                if (v4_prefix_holders > 0) {
                    const int64_t lid_seq_pages =
                        static_cast<int64_t>(
                            (max_seq + indexer_k_page_size - 1)
                            / indexer_k_page_size) * v4l.num_csa_layers;
                    const int64_t lid_holder_pages =
                        static_cast<int64_t>(
                            (v4_holder_tokens + indexer_k_page_size - 1)
                            / indexer_k_page_size) * v4l.num_csa_layers;
                    const int64_t hca_holder_pages =
                        holder_blocks * v4l.num_hca_layers;
                    auto afford = [](int64_t pool, int64_t inflight,
                                     int64_t per_holder) {
                        return per_holder > 0
                                   ? std::max<int64_t>(
                                         0, (pool - inflight) / per_holder)
                                   : std::numeric_limits<int64_t>::max();
                    };
                    // S4 elastic: the LID "pool" is the shared-span share
                    // (the fixed carve is gone on slabbed layouts).
                    const int64_t lid_pool_pages =
                        layout.slab_bytes > 0
                            ? static_cast<int64_t>(indexer_share_slabs)
                            : static_cast<int64_t>(gpu.indexer_k_pages);
                    const int64_t afford_holders = std::min(
                        {afford(lid_pool_pages,
                                static_cast<int64_t>(max_req) * lid_seq_pages,
                                lid_holder_pages),
                         afford(hca_pages,
                                static_cast<int64_t>(max_req) *
                                    blocks_per_seq * v4l.num_hca_layers,
                                hca_holder_pages),
                         afford(swa_pages,
                                static_cast<int64_t>(max_req) * swa_per_seq,
                                swa_per_seq)});
                    const double holder_mib =
                        static_cast<double>(
                            lid_holder_pages * v4l.indexer_bytes_per_page +
                            hca_holder_pages * v4l.hca_bytes_per_page +
                            swa_per_seq * v4l.swa_bytes_per_page) /
                        (1024.0 * 1024.0);
                    spdlog::info(
                        "V4 side-tier pools on GPU {}: {} in-flight + {} "
                        "prefix-holder sequence(s) at {:.1f} MiB/holder "
                        "(LID {} + HCA {} + SWA {} pages; holders are "
                        "COPY-ON-FORK full tier sets, TD-V4-SERVE-PREFIX / "
                        "INV-PREFIX-CACHE-3).",
                        gpu_i, max_req, afford_holders, holder_mib,
                        lid_holder_pages, hca_holder_pages, swa_per_seq);
                    if (afford_holders <
                        static_cast<int64_t>(v4_prefix_holders))
                        spdlog::warn(
                            "V4 side-tier pools on GPU {} afford only {} of "
                            "serving.prefix_cache.max_entries={} prefix "
                            "holders (VRAM cap scaled the pools). Excess "
                            "holders trip RETRYABLE pool exhaustion answered "
                            "by holder eviction (TD-INDEXER-POOL-EVICT) — "
                            "expect eviction churn; lower max_entries/"
                            "max_sequence_length or free VRAM to fix.",
                            gpu_i, afford_holders, v4_prefix_holders);
                }
            }
            gpu.kv_main_bytes = align_region(
                csa_pages * v4l.csa_bytes_per_page + scratch_bytes);
            gpu.kv_speculation_bytes =
                align_region(spec_pages * v4l.csa_bytes_per_page);
            gpu.kv_hca_bytes = align_region(hca_pages * v4l.hca_bytes_per_page);
            gpu.kv_swa_bytes = align_region(swa_pages * v4l.swa_bytes_per_page);
            gpu.prefill_scratch_preallocated_bytes = scratch_bytes;
            gpu.kv_main_pages = static_cast<int>(csa_pages);
            gpu.kv_speculation_pages = static_cast<int>(spec_pages);
            gpu.kv_hca_pages = static_cast<int>(hca_pages);
            gpu.kv_swa_pages = static_cast<int>(swa_pages);
            gpu.max_kv_pages = static_cast<int>(csa_pages + spec_pages);
        }
        int max_kv_pages = 0;
        if (v4) {
            // handled above
        } else if (!gpu.attention_host) {
            // Non-TP GPU: all available VRAM goes to expert cache
            max_kv_pages = 0;
        } else {
            if (auto* val = std::get_if<int>(&kv_cfg.max_pages_per_gpu)) {
                // Explicit count is PHYSICAL pages: one per (sequence,
                // logical page, layer) — operators must include the ×layers
                // factor (INV-KV-LAYER, TD-GOLDEN-KV-SPEC).
                max_kv_pages = *val;
                // DCP: explicit pages are per-GPU (already sharded)
            } else {
                // "auto" mode: compute from serving config, capped by VRAM
                // Reserve the expert-cache reservation + prefill scratch
                // from available (TD-MOE-EXPERT-WINDOW).
                int64_t scratch_budget = gpu.attention_host
                                             ? prefill_scratch_gb_to_bytes : 0;
                // S4 elastic: the indexer share comes off the top exactly
                // like the former carve (byte-identical partition), then
                // rejoins kv_main below.
                int64_t available_for_kv = available - expert_reserve -
                                           scratch_budget -
                                           indexer_share_bytes -
                                           kda_mapped_share_bytes;
                if (available_for_kv < 0) available_for_kv = 0;
                // Per-rank demand division (sharded KV only) happens inside
                // compute_auto_kv_pages, BEFORE the VRAM cap
                // (INV-KV-SIZE-SHARD).
                max_kv_pages = compute_auto_kv_pages(
                    cfg, layout.kv_bytes_per_page, kv_layers, available_for_kv,
                    gpu.dcp_kv_shard_factor);
            }
        }

        if (!v4) {
            // S4 elastic: kv_main absorbs the indexer share — the shared
            // slab pool serves BOTH tenants; admission reservation is what
            // claims indexer slabs out of it (INV-DSA-RESERVE).
            int64_t kv_total = static_cast<int64_t>(max_kv_pages) *
                                   layout.kv_bytes_per_page +
                               indexer_share_bytes +
                               // TD-KDA-STATE-MAPPED-SLABS: the mapped KDA
                               // state share rides in kv_main like the
                               // indexer share — shared slab capacity,
                               // claimed at admission.
                               kda_mapped_share_bytes;

            // Clamp KV total to available budget net of the expert reservation
            // (explicit max_pages_per_gpu conflicting with the expert_streaming
            // override would otherwise overflow the physical block).
            if (kv_total > available - expert_reserve)
                kv_total = available - expert_reserve;
            if (kv_total < 0) kv_total = 0;

            // Prefill scratch: pre-allocated at tail of KV main (only for TP GPUs)
            // Cap scratch so expert cache still gets at least the reservation
            if (gpu.attention_host) {
                int64_t max_scratch = available - kv_total - expert_reserve;
                if (max_scratch < 0) max_scratch = 0;
                scratch_bytes = std::min(prefill_scratch_gb_to_bytes, max_scratch);
                if (scratch_bytes < 0) scratch_bytes = 0;
                kv_total += scratch_bytes;
            }

            // Speculation/main split (before adding scratch). S4: the
            // indexer share is excluded from the spec fraction — it rides
            // in kv_main, where the shared slab pool lives.
            double spec_frac = kv_cfg.speculation_pool_fraction;
            int64_t kv_without_scratch =
                kv_total - scratch_bytes - indexer_share_bytes
                - kda_mapped_share_bytes;
            if (kv_without_scratch < 0) kv_without_scratch = 0;
            gpu.kv_speculation_bytes = align_region(static_cast<int64_t>(
                std::floor(static_cast<double>(kv_without_scratch) * spec_frac)));
            gpu.kv_main_bytes = align_region(kv_total - gpu.kv_speculation_bytes);
            gpu.prefill_scratch_preallocated_bytes = scratch_bytes;
        }

        // Expert cache budget (uses aligned KV sizes, not raw kv_total).
        // V4: the HCA + SWA tier regions are part of the KV carve too.
        const int64_t kv_aligned_total = gpu.kv_speculation_bytes +
                                         gpu.kv_main_bytes +
                                         gpu.kv_hca_bytes + gpu.kv_swa_bytes;
        int64_t expert_total;
        if (hw_gpu.vram_allocation_gb.has_value() &&
            hw_gpu.vram_allocation_gb->expert_streaming > 0.0) {
            // Override: expert_streaming specifies the total expert cache.
            // TD-MOE-EXPERT-WINDOW: clamp to what physically remains after
            // KV (the reservation above sized KV to leave room, but an
            // oversized override must never push Σregions past the block).
            expert_total = std::min(
                config::vram_gb_to_bytes(
                    hw_gpu.vram_allocation_gb->expert_streaming),
                gpu.total_vram_bytes - gpu.pinned_bytes -
                    gpu.indexer_k_bytes - gpu.indexer_k_pad_bytes -
                    gpu.kda_state_bytes -
                    kv_aligned_total - gpu.safety_margin_bytes);
        } else {
            expert_total = gpu.total_vram_bytes - gpu.pinned_bytes -
                           gpu.indexer_k_bytes - gpu.indexer_k_pad_bytes -
                           gpu.kda_state_bytes -
                           kv_aligned_total - gpu.safety_margin_bytes;
        }
        if (expert_total < 0) expert_total = 0;

        // Per-GPU budget table — printed BEFORE the expert-cache validation so
        // a sizing failure always shows the full arithmetic that produced it.
        spdlog::info(
            "VramAllocator: GPU {} (hw id {}, {}) budget: total {:.1f} MiB = "
            "pinned {:.1f} + margin {:.1f} + indexer_k {:.1f} + kv[main {:.1f}"
            " spec {:.1f} hca {:.1f} swa {:.1f} scratch-in-main {:.1f}] + "
            "expert {:.1f} (reserve {:.1f}, min {:.1f})",
            gpu_i, gpu.gpu_id, in_tp[gpu_i] ? "TP" : "expert-only",
            static_cast<double>(gpu.total_vram_bytes) / (1024.0 * 1024.0),
            static_cast<double>(gpu.pinned_bytes) / (1024.0 * 1024.0),
            static_cast<double>(gpu.safety_margin_bytes) / (1024.0 * 1024.0),
            static_cast<double>(gpu.indexer_k_bytes) / (1024.0 * 1024.0),
            static_cast<double>(gpu.kv_main_bytes) / (1024.0 * 1024.0),
            static_cast<double>(gpu.kv_speculation_bytes) / (1024.0 * 1024.0),
            static_cast<double>(gpu.kv_hca_bytes) / (1024.0 * 1024.0),
            static_cast<double>(gpu.kv_swa_bytes) / (1024.0 * 1024.0),
            static_cast<double>(scratch_bytes) / (1024.0 * 1024.0),
            static_cast<double>(expert_total) / (1024.0 * 1024.0),
            static_cast<double>(expert_reserve) / (1024.0 * 1024.0),
            static_cast<double>(min_expert_cache) / (1024.0 * 1024.0));

        // Validate minimum expert cache
        if (min_expert_cache > 0 && expert_total < min_expert_cache) {
            throw std::runtime_error(
                "GPU " + std::to_string(gpu.gpu_id) +
                ": insufficient VRAM for expert cache. Need at least " +
                std::to_string(min_expert_cache) + " bytes (" +
                std::to_string(experts_per_tok) + " active experts), but only " +
                std::to_string(expert_total) + " bytes available. " +
                "Reduce pinned layers or max_sequence_length.");
        }

        // Stable/streaming zone split
        // TD-91c: clamp to [0.0, 1.0] — values outside this range would produce
        // nonsensical streaming/stable splits (negative bytes, silent zone collapse).
        double stable_frac;
        if (hw_gpu.vram_allocation_gb.has_value() &&
            hw_gpu.vram_allocation_gb->stable_zone_fraction >= 0.0) {
            stable_frac = hw_gpu.vram_allocation_gb->stable_zone_fraction;
        } else {
            stable_frac = cfg.memory.expert_cache.stable_zone_fraction;
        }
        if (stable_frac < 0.0 || stable_frac > 1.0) {
            spdlog::warn("VramAllocator: GPU {} stable_zone_fraction {:.3f} outside "
                         "[0.0, 1.0], clamping", gpu_i, stable_frac);
            stable_frac = std::clamp(stable_frac, 0.0, 1.0);
        }
        gpu.expert_streaming_bytes = align_region(
            expert_total - static_cast<int64_t>(
                std::floor(static_cast<double>(expert_total) * stable_frac)));
        gpu.expert_stable_bytes = expert_total - gpu.expert_streaming_bytes;
        if (gpu.expert_stable_bytes < 0) gpu.expert_stable_bytes = 0;

        // Streaming zone sub-split: spill (top) + prefetch (bottom)
        // Spill zone sized by fraction, but capped to what naive actually needs
        // beyond Tier 1. If ceiling is low enough that Tier 1 covers all scratch,
        // spill zone is zero — all streaming goes to expert prefetch.
        int64_t spill_by_fraction = static_cast<int64_t>(
            std::floor(static_cast<double>(gpu.expert_streaming_bytes) * spill_frac));
        gpu.streaming_spill_bytes = std::min(spill_by_fraction, spill_scratch_needed);
        if (gpu.streaming_spill_bytes < 0) gpu.streaming_spill_bytes = 0;
        // Align the sub-split boundary: the prefetch sub-zone base is
        // streaming_base + spill_bytes, and expert slots carved there feed the
        // same quant GEMM kernels as the stable zone (vectorized/int4 loads
        // require aligned pointers). A byte-exact spill_scratch_needed left the
        // prefetch slots misaligned — computing from a prefetch-zone slot
        // faulted with cudaErrorMisalignedAddress (716). Round UP (scratch
        // coverage preserved; the prefetch zone loses < 4 KiB), capped to the
        // streaming region.
        constexpr int64_t kZoneAlign = 4096;
        gpu.streaming_spill_bytes = std::min(
            (gpu.streaming_spill_bytes + kZoneAlign - 1) / kZoneAlign * kZoneAlign,
            gpu.expert_streaming_bytes);
        gpu.streaming_prefetch_bytes = gpu.expert_streaming_bytes -
                                       gpu.streaming_spill_bytes;

        // Page counts (V4 set its per-bucket counts in the branch above)
        if (!v4) {
            gpu.max_kv_pages = max_kv_pages;
            if (layout.kv_bytes_per_page > 0) {
                gpu.kv_speculation_pages = static_cast<int>(
                    gpu.kv_speculation_bytes / layout.kv_bytes_per_page);
                // kv_main_pages: exclude scratch tail from page count
                int64_t kv_main_for_pages = gpu.kv_main_bytes - scratch_bytes;
                gpu.kv_main_pages = static_cast<int>(
                    kv_main_for_pages / layout.kv_bytes_per_page);
                // S1: quantize the kMain page span DOWN to whole slabs so the
                // shared region [indexer slabs | kMain pages] is an exact
                // slab grid (loses < one slab of pages ≈ 1 MiB; the dead
                // bytes stay inside kv_main before the scratch tail).
                if (layout.pages_per_slab > 1 && gpu.kv_main_pages > 0) {
                    const int drop =
                        gpu.kv_main_pages % layout.pages_per_slab;
                    if (drop > 0) {
                        spdlog::info(
                            "KV pool on GPU {}: S1 slab quantization drops "
                            "{} of {} kMain pages (span = whole slabs, {} "
                            "slabs)",
                            gpu_i, drop, gpu.kv_main_pages,
                            (gpu.kv_main_pages - drop)
                                / layout.pages_per_slab);
                        gpu.kv_main_pages -= drop;
                    }
                }
            } else {
                gpu.kv_speculation_pages = 0;
                gpu.kv_main_pages = 0;
            }
        }

        // ── TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA: admissible-context ceiling ──
        // Since TD-KDA-STATE-MAPPED-SLABS the kMain pool serves THREE
        // tenants at admission time — the request's KV pages, its
        // full-length indexer-K reservation (INV-DSA-RESERVE), and its
        // mapped KDA state runs — so a VRAM-clamped pool can be smaller
        // than ONE max-length request's whole-life demand even though the
        // boot "fits". Compute the honest single-request ceiling here, log
        // it (the autoconfig cross-check reads this line back), and WARN
        // when the configured max_sequence_length is not admissible.
        // Non-V4 attention hosts only: V4's demand lives in the side tiers
        // (own arithmetic), expert-only GPUs carry no kMain.
        if (!v4 && gpu.attention_host && gpu.kv_main_pages > 0
            && cfg.serving.max_sequence_length > 0
            && cfg.memory.kv_cache.page_size_tokens > 0) {
            const int adm_page_size = cfg.memory.kv_cache.page_size_tokens;
            const int pps = layout.pages_per_slab;
            const bool adm_idx_local =
                cfg.hardware.dcp_indexer_mode == config::DcpIndexerMode::local
                && dcp_shard_factor > 1;
            const int64_t adm_unit_slabs =
                (layout.kda.enabled && layout.kda.mapped
                 && layout.slab_bytes > 0)
                    ? (layout.kda.per_layer_bytes + layout.slab_bytes - 1)
                          / layout.slab_bytes
                    : 0;
            const int64_t adm_state_pages =
                adm_unit_slabs > 0
                    ? static_cast<int64_t>(layout.kda.num_layers)
                          * adm_unit_slabs * pps
                    : 0;
            const int adm_shard = gpu.dcp_kv_shard_factor;
            const int adm_pages_per_chunk = std::max(
                1, cfg.memory.kv_cache.dcp_chunk_size / adm_page_size);
            // One request's KV pages for a T-token life (worst-rank chunk
            // arithmetic under sharded KV — compute_auto_kv_pages verbatim).
            auto adm_kv_pages = [&](int64_t T) -> int64_t {
                int64_t per_seq = (T + adm_page_size - 1) / adm_page_size;
                if (adm_shard > 1) {
                    const int64_t cycle =
                        static_cast<int64_t>(adm_pages_per_chunk) * adm_shard;
                    per_seq = (per_seq / cycle) * adm_pages_per_chunk
                              + std::min<int64_t>(per_seq % cycle,
                                                  adm_pages_per_chunk);
                }
                return per_seq * kv_layers;
            };
            // Indexer-K reservation, priced in kMain pages (one slab per
            // indexer page on slabbed models; unslabbed models carve a
            // separate kIndexerK pool — zero kMain cost).
            auto adm_idx_pages = [&](int64_t T) -> int64_t {
                if (!has_dsa || indexer_k_page_size <= 0
                    || num_dsa_layers <= 0 || pps <= 0
                    || layout.slab_bytes <= 0)
                    return 0;
                int64_t ip = (T + indexer_k_page_size - 1)
                             / indexer_k_page_size;
                if (adm_idx_local)
                    ip = (ip + dcp_shard_factor - 1) / dcp_shard_factor;
                return ip * num_dsa_layers * pps;
            };
            auto adm_demand = [&](int64_t T) -> int64_t {
                return adm_kv_pages(T) + adm_idx_pages(T) + adm_state_pages;
            };
            const int64_t adm_pool = gpu.kv_main_pages;
            const int64_t adm_max_seq = cfg.serving.max_sequence_length;
            // Largest T with demand(T) <= pool. demand is monotone in T and
            // >= T/page_size, so pool*page_size bounds the search.
            int64_t lo = 0, hi = adm_pool * adm_page_size + adm_page_size;
            while (lo < hi) {
                const int64_t mid = lo + (hi - lo + 1) / 2;
                if (adm_demand(mid) <= adm_pool) lo = mid;
                else hi = mid - 1;
            }
            gpu.admissible_ctx_tokens = static_cast<int>(
                std::min<int64_t>(lo, std::numeric_limits<int>::max()));
            gpu.admission_kv_pages = adm_kv_pages(adm_max_seq);
            gpu.admission_indexer_pages = adm_idx_pages(adm_max_seq);
            gpu.admission_state_pages = adm_state_pages;
            gpu.admission_demand_pages = gpu.admission_kv_pages
                                         + gpu.admission_indexer_pages
                                         + adm_state_pages;
            spdlog::info(
                "VramAllocator: GPU {} admissible context (single request) "
                "= {} tokens — max_sequence_length {} needs {} kMain pages "
                "(KV {} + indexer {} + KDA state {}) vs pool {} pages",
                gpu_i, gpu.admissible_ctx_tokens, adm_max_seq,
                gpu.admission_demand_pages, gpu.admission_kv_pages,
                gpu.admission_indexer_pages, gpu.admission_state_pages,
                adm_pool);
            if (lo < adm_max_seq) {
                // Windowed admission (TD-KVT-ADMISSION-UPFRONT) admits
                // beyond the pool with demote-behind, so the ceiling does
                // not bind there — informational only.
                const bool windowed = cfg.memory.kv_tiering.enabled
                    && cfg.memory.kv_tiering.tiered_prefill
                    && cfg.compute.dsa_sparse_prefill;
                if (windowed) {
                    spdlog::info(
                        "VramAllocator: GPU {}: max_sequence_length {} "
                        "exceeds the single-request pool ceiling {} but "
                        "windowed admission (tiered prefill) serves beyond "
                        "it — ceiling informational only",
                        gpu_i, adm_max_seq, gpu.admissible_ctx_tokens);
                } else {
                    spdlog::warn(
                        "VramAllocator: GPU {}: serving.max_sequence_length "
                        "{} is NOT ADMISSIBLE — one max-length request "
                        "needs {} kMain pages (KV {} + indexer {} + mapped "
                        "KDA state {}) but the pool holds {}; admissible "
                        "ceiling {} tokens. A longer request is refused "
                        "FAST and NON-RETRYABLE at admission (its own "
                        "footprint exceeds the whole pool — holder "
                        "eviction cannot help). Lower "
                        "serving.max_sequence_length, free VRAM, or "
                        "re-derive the recipe "
                        "(TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA).",
                        gpu_i, adm_max_seq, gpu.admission_demand_pages,
                        gpu.admission_kv_pages, gpu.admission_indexer_pages,
                        gpu.admission_state_pages, adm_pool,
                        gpu.admissible_ctx_tokens);
                }
            }
        }

        layout.gpus.push_back(gpu);
    }

    return layout;
}

// ── VramAllocator ───────────────────────────────────────────────────────────

VramAllocator::VramAllocator(VramLayout layout,
                             std::vector<compute::DeviceBackend*> device_backends)
    : layout_(std::move(layout)),
      device_backends_(std::move(device_backends)) {
    if (device_backends_.size() != layout_.gpus.size()) {
        throw std::invalid_argument(
            "VramAllocator: device_backends.size()="
            + std::to_string(device_backends_.size())
            + " must equal layout.gpus.size()="
            + std::to_string(layout_.gpus.size()));
    }
    allocate_all();
}

VramAllocator::~VramAllocator() {
    free_all();
}

VramAllocator::VramAllocator(VramAllocator&& other) noexcept
    : layout_(std::move(other.layout_)),
      regions_(std::move(other.regions_)),
      device_backends_(std::move(other.device_backends_)) {
    other.regions_.clear();
}

VramAllocator& VramAllocator::operator=(VramAllocator&& other) noexcept {
    if (this != &other) {
        free_all();
        layout_ = std::move(other.layout_);
        regions_ = std::move(other.regions_);
        device_backends_ = std::move(other.device_backends_);
        other.regions_.clear();
    }
    return *this;
}

void VramAllocator::allocate_all() {
    regions_.reserve(layout_.gpus.size());

    for (size_t i = 0; i < layout_.gpus.size(); ++i) {
        const auto& gpu = layout_.gpus[i];
        GpuRegion reg;
        reg.gpu.id = gpu.gpu_id;
        reg.gpu.position = static_cast<int>(i);

        // Allocate total_vram - safety_margin as one contiguous block
        reg.allocated_bytes = gpu.total_vram_bytes - gpu.safety_margin_bytes;
        if (reg.allocated_bytes <= 0) {
            throw std::runtime_error(
                "GPU " + std::to_string(gpu.gpu_id) +
                ": nothing to allocate after safety margin");
        }

        device_backends_[i]->set_device();
        reg.base = device_backends_[i]->device_alloc(
            static_cast<size_t>(reg.allocated_bytes));
        if (!reg.base) {
            throw std::runtime_error(
                "GPU " + std::to_string(gpu.gpu_id) +
                ": device_alloc failed for " +
                std::to_string(reg.allocated_bytes) + " bytes");
        }

        // Partition: pinned | kv_speculation | kda_state | kv_hca | kv_swa |
        //            indexer_k | kv_main (+scratch) | expert_streaming | expert_stable
        // kv_main end == expert_streaming start (contiguity for prefill scratch + spill)
        // Each region start aligned to 256 bytes for CUDA kernel alignment requirements
        // (k_append writes float* and bfloat162 to KV cache, needing 4+ byte alignment)
        // EXCEPT the S1 shared-region interior (below).
        constexpr int64_t kRegionAlign = 256;
        auto align_up = [](char* ptr, int64_t align) -> char* {
            auto addr = reinterpret_cast<uintptr_t>(ptr);
            addr = (addr + align - 1) & ~(align - 1);
            return reinterpret_cast<char*>(addr);
        };

        auto* p = static_cast<char*>(reg.base);
        reg.pinned = p;
        p = align_up(p + gpu.pinned_bytes, kRegionAlign);
        reg.kv_speculation = p;
        p = align_up(p + gpu.kv_speculation_bytes, kRegionAlign);
        // GF3.8: KDA per-request state pool (collapsed — zero bytes — on
        // non-glm5_next models, so every pre-GF3.8 layout is byte-identical:
        // p advances by 0 and stays 256-aligned).
        reg.kda_state = p;
        p = align_up(p + gpu.kda_state_bytes, kRegionAlign);
        reg.kv_hca = p;
        p = align_up(p + gpu.kv_hca_bytes, kRegionAlign);
        reg.kv_swa = p;
        p = align_up(p + gpu.kv_swa_bytes, kRegionAlign);
        // ── S1 shared region (TD-INDEXER-POOL-ELASTIC): the pad shim keeps
        // kv_main 256-aligned; the indexer slab span then abuts kv_main
        // SLAB-EXACTLY (no align_up — a slab must be an aligned run of flat
        // kMain page indices). Indexer page base alignment drops to the slab
        // stride's 16+-byte guarantee, which every indexer consumer
        // (pointer tables, FP8 rows + f32 scales) tolerates. ──
        p += gpu.indexer_k_pad_bytes;
        reg.indexer_k = p;
        p += gpu.indexer_k_bytes;
        reg.kv_main = p;
        if (layout_.slab_bytes > 0 && gpu.indexer_k_pages > 0) {
            const bool span_exact =
                static_cast<char*>(reg.kv_main) -
                    static_cast<char*>(reg.indexer_k) ==
                static_cast<int64_t>(gpu.indexer_k_pages) * layout_.slab_bytes;
            const bool kv_aligned =
                reinterpret_cast<uintptr_t>(reg.kv_main) % kRegionAlign == 0;
            if (!span_exact || !kv_aligned) {
                throw std::logic_error(
                    "VramAllocator GPU " + std::to_string(gpu.gpu_id) +
                    ": S1 shared-region geometry broken — indexer span " +
                    std::to_string(gpu.indexer_k_bytes) + " B (" +
                    std::to_string(gpu.indexer_k_pages) + " slabs x " +
                    std::to_string(layout_.slab_bytes) + " B, pad " +
                    std::to_string(gpu.indexer_k_pad_bytes) +
                    " B) must end exactly at a 256-aligned kv_main base");
            }
        }
        p = align_up(p + gpu.kv_main_bytes, kRegionAlign);
        reg.expert_streaming = p;
        p = align_up(p + gpu.expert_streaming_bytes, kRegionAlign);
        reg.expert_stable = p;

        // ── Explicit per-GPU VRAM region breakdown (INV-VRAM: sums to the
        // single contiguous allocation; the rest of the card is CUDA context +
        // safety margin). Bytes → MiB. Sub-regions (scratch/spill/prefetch) are
        // carved from their parent, shown indented. ──
        constexpr double kMiB = 1024.0 * 1024.0;
        const int64_t pinned = gpu.pinned_bytes;
        const int64_t kvspec = gpu.kv_speculation_bytes;
        const int64_t idxk   = gpu.indexer_k_bytes;
        const int64_t kvmain = gpu.kv_main_bytes;
        const int64_t scratch = gpu.prefill_scratch_preallocated_bytes;
        const int64_t estream = gpu.expert_streaming_bytes;
        const int64_t espill  = gpu.streaming_spill_bytes;
        const int64_t eprefetch = gpu.streaming_prefetch_bytes;
        const int64_t estable = gpu.expert_stable_bytes;
        const int64_t kvhca = gpu.kv_hca_bytes;
        const int64_t kvswa = gpu.kv_swa_bytes;
        const int64_t kdast = gpu.kda_state_bytes;
        const int64_t sum = pinned + kvspec + kdast + idxk +
                            gpu.indexer_k_pad_bytes +
                            kvhca + kvswa + kvmain + estream + estable;
        spdlog::info(
            "VramAllocator GPU {}: {:.0f} MiB total, {:.0f} MiB allocated "
            "(safety margin {:.0f} MiB)",
            gpu.gpu_id, gpu.total_vram_bytes / kMiB,
            reg.allocated_bytes / kMiB, gpu.safety_margin_bytes / kMiB);
        spdlog::info(
            "  pinned weights   {:8.1f} MiB  | kv_main       {:8.1f} MiB "
            "({} pages; incl. prefill scratch {:.1f} MiB)",
            pinned / kMiB, kvmain / kMiB, gpu.kv_main_pages, scratch / kMiB);
        spdlog::info(
            "  kv_speculation   {:8.1f} MiB ({} pages) | indexer_k {:8.1f} MiB "
            "({} pages)",
            kvspec / kMiB, gpu.kv_speculation_pages, idxk / kMiB,
            gpu.indexer_k_pages);
        if (layout_.v4.enabled) {
            spdlog::info(
                "  kv_hca (V4)      {:8.1f} MiB ({} pages) | kv_swa (V4) "
                "{:8.1f} MiB ({} pages)",
                kvhca / kMiB, gpu.kv_hca_pages, kvswa / kMiB,
                gpu.kv_swa_pages);
        }
        if (layout_.kda.enabled) {
            spdlog::info(
                "  kda_state (GF3.8){:8.1f} MiB ({} slots x {:.1f} MiB — "
                "per-REQUEST recurrent+conv state; the concurrency clamp "
                "on this architecture)",
                kdast / kMiB, gpu.kda_state_slots,
                layout_.kda.slot_bytes / kMiB);
        }
        spdlog::info(
            "  expert_streaming {:8.1f} MiB (spill {:.1f} + prefetch {:.1f}) "
            "| expert_stable {:8.1f} MiB  ⇒ Σregions {:.1f} MiB",
            estream / kMiB, espill / kMiB, eprefetch / kMiB, estable / kMiB,
            sum / kMiB);
        if (layout_.slab_bytes > 0 && gpu.kv_main_pages > 0) {
            spdlog::info(
                "  S4 shared slab pool: {} slabs ({} pages/slab, slab {} B) "
                "serve KV pages AND indexer-K claims — no fixed indexer "
                "carve; indexer slabs are claimed at admission "
                "(INV-DSA-RESERVE) and returned whole",
                gpu.kv_main_pages / layout_.pages_per_slab,
                layout_.pages_per_slab, layout_.slab_bytes);
        }

        regions_.push_back(reg);
    }
}

void VramAllocator::free_all() {
    for (size_t i = 0; i < regions_.size(); ++i) {
        auto& reg = regions_[i];
        if (reg.base) {
            device_backends_[i]->device_free(reg.base);
            reg.base = nullptr;
        }
    }
    regions_.clear();
}

// ── TD-GLM5-KDA-SLOTS-EXPORT ────────────────────────────────────────────────
KdaStateExport kda_state_export(const VramLayout& layout) {
    KdaStateExport out;
    if (!layout.kda.enabled) return out;
    out.mapped = layout.kda.mapped;
    out.slot_bytes = layout.kda.slot_bytes;
    for (const auto& g : layout.gpus) {
        if (!g.attention_host) continue;  // expert-only host: no state tenant
        if (layout.kda.mapped) {
            if (g.kv_main_pages > 0
                && (out.pool_pages == 0 || g.kv_main_pages < out.pool_pages)) {
                out.pool_pages = g.kv_main_pages;
            }
            // canonical per-request demand from the admissible-ctx pass
            // (identical across attention hosts — model geometry x slab grid)
            if (g.admission_state_pages > 0) {
                out.pages_per_seq = g.admission_state_pages;
            }
        } else if (g.kda_state_slots > 0
                   && (out.slots == 0 || g.kda_state_slots < out.slots)) {
            out.slots = g.kda_state_slots;
        }
    }
    if (out.mapped && out.pages_per_seq == 0 && layout.slab_bytes > 0
        && layout.pages_per_slab > 0) {
        // admissible-ctx pass skipped (e.g. max_sequence_length 0): same
        // arithmetic from the layout geometry.
        const int64_t unit_slabs =
            (layout.kda.per_layer_bytes + layout.slab_bytes - 1)
            / layout.slab_bytes;
        out.pages_per_seq = static_cast<int64_t>(layout.kda.num_layers)
                            * unit_slabs * layout.pages_per_slab;
    }
    return out;
}

}  // namespace layerstorm::memory

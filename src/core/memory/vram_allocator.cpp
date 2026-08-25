#include "core/memory/vram_allocator.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

int64_t indexer_k_bytes_per_page(const model::ModelConfig& model_cfg,
                                  const config::Config& cfg) {
    return indexer_k_bytes_per_token(model_cfg, cfg.quantization.kv_cache) *
           cfg.memory.kv_cache.indexer_k_page_size_tokens;
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

// Region boundary alignment (SPEC_UPDATES line 73).
// All region boundaries must be 256-byte aligned for CUDA kernel access patterns.
// Sizes are rounded up inline during layout computation so that contiguous regions
// (kv_main end == streaming start) hold with zero gaps.
static constexpr int64_t kLayoutAlign = 256;

static int64_t align_region(int64_t sz) {
    return (sz + kLayoutAlign - 1) & ~(kLayoutAlign - 1);
}

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

    auto budgets = registry.estimate_gpu_budgets();
    int num_layers = registry.num_layers();
    // TD-GOLDEN-KV-SPEC: the dispatcher allocates one physical KV page per
    // (logical page, layer) including the MTP layer(s) (INV-KV-LAYER) — KV
    // pool sizing must count them too or full occupancy exhausts the pool.
    const int kv_layers = num_layers + model_cfg.raw().num_nextn_predict_layers;
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
    int64_t indexer_k_bpt = indexer_k_bytes_per_token(model_cfg,
                                                       cfg.quantization.kv_cache);
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
            if (model_cfg.is_full_index_layer(l) || l == 0) ++num_dsa_layers;
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
        const int64_t per_seq_bytes =
            static_cast<int64_t>(idx_pages_per_seq) * num_dsa_layers
            * layout.indexer_k_bytes_per_page;
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
                "Indexer-K pool: VRAM affords {} concurrent sequence(s), "
                "below serving.max_concurrent_requests={} ({:.1f} MiB/seq, "
                "and the KV pool is the residual of this carve). Concurrent "
                "long-context requests churn prefix-holder evictions "
                "(retryable, TD-INDEXER-POOL-EVICT); lower "
                "max_sequence_length or free VRAM (pinned weights / "
                "expert_streaming reserve) to raise it.",
                indexer_seqs, max_req,
                static_cast<double>(per_seq_bytes) / (1024.0 * 1024.0));
    }

    for (size_t gpu_i = 0; gpu_i < budgets.size(); ++gpu_i) {
        auto& budget = budgets[gpu_i];
        GpuVramLayout gpu{};
        gpu.gpu_id = budget.gpu_id;
        gpu.total_vram_bytes = budget.total_vram_bytes;
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

        // Indexer K cache.
        // V4: Lightning-Indexer tier — CSA layers only, per TP GPU, sized for
        // max_concurrent_requests full-length sequences (entries are per
        // compressed block, so pages are cheap: ~270 KB per 8192 tokens).
        if (v4 && (!has_tp || in_tp[gpu_i])) {
            const int max_seq = cfg.serving.max_sequence_length;
            const int max_req = cfg.serving.max_concurrent_requests;
            const int pages_per_seq =
                (max_seq + indexer_k_page_size - 1) / indexer_k_page_size;
            gpu.indexer_k_pages =
                pages_per_seq * layout.v4.num_csa_layers * max_req;
            gpu.indexer_k_bytes = align_region(
                static_cast<int64_t>(gpu.indexer_k_pages) *
                layout.indexer_k_bytes_per_page);
        } else
        // DSA TP GPUs only.
        // Replicated mode (default): full context pages per GPU.
        // Local mode with DCP: pages divided by dcp_shard_factor.
        if (has_dsa && (!has_tp || in_tp[gpu_i])) {
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
            gpu.indexer_k_pages =
                static_cast<int>(pages_per_seq) * num_dsa_layers * seqs;
            gpu.indexer_k_bytes = align_region(
                static_cast<int64_t>(gpu.indexer_k_pages) *
                layout.indexer_k_bytes_per_page);
            spdlog::info(
                "Indexer-K pool on GPU {}: {} pages ({:.1f} MiB) = {} "
                "pages/seq x {} computing layers x {} sequence(s) "
                "(max_sequence_length={}). A live sequence beyond this "
                "budget — prefix-cache holders included — trips a RETRYABLE "
                "exhaustion answered by holder eviction "
                "(TD-INDEXER-POOL-EVICT).",
                gpu_i, gpu.indexer_k_pages,
                static_cast<double>(gpu.indexer_k_bytes) / (1024.0 * 1024.0),
                pages_per_seq, num_dsa_layers, seqs, max_seq);
        } else {
            gpu.indexer_k_pages = 0;
            gpu.indexer_k_bytes = 0;
        }

        // Available for KV + expert cache (after pinned + safety + indexer_k)
        int64_t available = gpu.total_vram_bytes - gpu.pinned_bytes -
                            gpu.safety_margin_bytes - gpu.indexer_k_bytes;

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

        // Resolve max_kv_pages — non-TP GPUs get zero (they don't do attention)
        const auto& kv_cfg = cfg.memory.kv_cache;
        int64_t scratch_bytes = 0;
        if (v4) {
            // ── V4-3b: 3-bucket page demand, auto-sized, proportional
            //    scale-down under the VRAM cap ──
            const auto& v4l = layout.v4;
            int64_t csa_pages = 0, hca_pages = 0, swa_pages = 0, spec_pages = 0;
            if (!has_tp || in_tp[gpu_i]) {
                const int max_seq = cfg.serving.max_sequence_length;
                const int max_req = cfg.serving.max_concurrent_requests;
                const int64_t blocks_per_seq =
                    (max_seq + v4l.logical_block_tokens - 1) /
                    v4l.logical_block_tokens;
                csa_pages = static_cast<int64_t>(max_req) * blocks_per_seq *
                            v4l.num_csa_layers;
                hca_pages = static_cast<int64_t>(max_req) * blocks_per_seq *
                            v4l.num_hca_layers;
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
                swa_pages = static_cast<int64_t>(max_req) * swa_per_seq;
                // Speculation pool: CSA-page-size sibling of the main pool
                // (the page machinery requires main/spec pages equal-sized).
                spec_pages = static_cast<int64_t>(std::floor(
                    static_cast<double>(csa_pages) *
                    kv_cfg.speculation_pool_fraction));

                // VRAM cap: scale ALL buckets proportionally.
                int64_t scratch_budget = prefill_scratch_gb_to_bytes;
                int64_t available_for_kv =
                    available - expert_reserve - scratch_budget;
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
        } else if (has_tp && !in_tp[gpu_i]) {
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
                int64_t scratch_budget = (!has_tp || in_tp[gpu_i])
                                             ? prefill_scratch_gb_to_bytes : 0;
                int64_t available_for_kv = available - expert_reserve - scratch_budget;
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
            int64_t kv_total = static_cast<int64_t>(max_kv_pages) *
                               layout.kv_bytes_per_page;

            // Clamp KV total to available budget net of the expert reservation
            // (explicit max_pages_per_gpu conflicting with the expert_streaming
            // override would otherwise overflow the physical block).
            if (kv_total > available - expert_reserve)
                kv_total = available - expert_reserve;
            if (kv_total < 0) kv_total = 0;

            // Prefill scratch: pre-allocated at tail of KV main (only for TP GPUs)
            // Cap scratch so expert cache still gets at least the reservation
            if (!has_tp || in_tp[gpu_i]) {
                int64_t max_scratch = available - kv_total - expert_reserve;
                if (max_scratch < 0) max_scratch = 0;
                scratch_bytes = std::min(prefill_scratch_gb_to_bytes, max_scratch);
                if (scratch_bytes < 0) scratch_bytes = 0;
                kv_total += scratch_bytes;
            }

            // Speculation/main split (before adding scratch)
            double spec_frac = kv_cfg.speculation_pool_fraction;
            int64_t kv_without_scratch = kv_total - scratch_bytes;
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
                    gpu.indexer_k_bytes - kv_aligned_total -
                    gpu.safety_margin_bytes);
        } else {
            expert_total = gpu.total_vram_bytes - gpu.pinned_bytes -
                           gpu.indexer_k_bytes -
                           kv_aligned_total - gpu.safety_margin_bytes;
        }
        if (expert_total < 0) expert_total = 0;

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
            } else {
                gpu.kv_speculation_pages = 0;
                gpu.kv_main_pages = 0;
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

        // Partition: pinned | kv_speculation | indexer_k | kv_hca | kv_swa | kv_main (+scratch) | expert_streaming | expert_stable
        // kv_main end == expert_streaming start (contiguity for prefill scratch + spill)
        // Each region start aligned to 256 bytes for CUDA kernel alignment requirements
        // (k_append writes float* and bfloat162 to KV cache, needing 4+ byte alignment).
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
        reg.indexer_k = p;
        p = align_up(p + gpu.indexer_k_bytes, kRegionAlign);
        reg.kv_hca = p;
        p = align_up(p + gpu.kv_hca_bytes, kRegionAlign);
        reg.kv_swa = p;
        p = align_up(p + gpu.kv_swa_bytes, kRegionAlign);
        reg.kv_main = p;
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
        const int64_t sum = pinned + kvspec + idxk + kvhca + kvswa + kvmain +
                            estream + estable;
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
        spdlog::info(
            "  expert_streaming {:8.1f} MiB (spill {:.1f} + prefetch {:.1f}) "
            "| expert_stable {:8.1f} MiB  ⇒ Σregions {:.1f} MiB",
            estream / kMiB, espill / kMiB, eprefetch / kMiB, estable / kMiB,
            sum / kMiB);

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

}  // namespace layerstorm::memory

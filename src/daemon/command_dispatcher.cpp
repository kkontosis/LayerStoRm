// Command dispatcher implementation.
//
// See command_dispatcher.h for design overview.

#include "daemon/command_dispatcher.h"
#include "daemon/attention/arch_base.h"  // complete type for arch unique_ptrs (dtor)
#include "daemon/moe/arch_base.h"  // complete type for the MoE arch unique_ptrs (dtor)
#include "daemon/dispatch_detail.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>  // std::getenv (LS_LOADER_SHADOW gate)
#include <cstring>  // std::strcmp (EP combine precision env)
#include <vector>   // TD-PREFILL-NONDET seam_dump_hidden D2H staging

#include <spdlog/spdlog.h>

#include "compute/stream_manager.h"
#include "core/attention_device.h"
#include "core/cuda_hardware_query.h"  // EPM-1: host_unregister_pinned
#include "core/device_backend.h"
#include "core/expert_device.h"
#include "core/memory/expert_cache.h"
#include "core/memory/numa_manager.h"  // EPM-1: NumaBuffer free
#include "core/memory/page_allocator.h"
#include "core/memory/pinned_expert_arena.h"  // reloc-trace sink teardown
#include "speculation/epm_dump.h"      // EPM-1: EpmRoutingDumper (dtor)
#include "model/model_config.h"  // TD-V4-KMAIN-SIZING: attention_type_for_layer
#include "model/quantization/gguf_kquant.h"  // GG-5b: is_gguf_weight_quant (workspace sizing)
#include "daemon/buffer_registry.h"
#include "daemon/kv_tiering_manager.h"
#include "daemon/v4_kv_tiering.h"  // GLM-25k
#include "parallelism/dcp_communicator.h"
#include "parallelism/dcp_executor.h"

// setup_spec_pipeline uses embedding kernel
#include "compute/kernels/embedding/embedding.h"
namespace layerstorm::daemon {

// ── KD-R2: HiddenStatePair ────────────────────────────────────────────────

void HiddenStatePair::commit(size_t bytes, void* stream,
                             compute::DeviceBackend* dev) const {
    if (attn_buf && moe_buf && bytes > 0) {
        dev->memcpy_d2d_async(attn_buf, moe_buf, bytes, stream);
    }
}

// ── Shared micro-helpers ──────────────────────────────────────────────────
// Type conversion helpers (make_key, to_zone, to_stream, to_graph_type,
// sub_component_offset) are now in dispatch_detail.h.

int CommandDispatcher::resolve_pair_idx(uint32_t gpu) const {
    return (gpu < gpu_pos_to_pair_idx_.size())
        ? gpu_pos_to_pair_idx_[gpu] : -1;
}

void* CommandDispatcher::create_and_record_event(int gpu, compute::StreamId sid) {
    void* event = deps_.stream_manager->create_event(gpu);
    deps_.stream_manager->record_event(event, gpu, sid);
    return event;
}

// TD-PREFILL-NONDET diagnostic: env-gated hidden-state stage dump (see
// command_dispatcher.h). Zero work when LS_SEAM_DUMP is unset.
void CommandDispatcher::seam_dump_hidden(uint32_t tag4cc, int layer, int gpu,
                                         const void* dev_buf, int rows,
                                         int hidden) {
    static const char* path = std::getenv("LS_SEAM_DUMP");
    if (!path || !*path) return;
    static int max_layer = [] {
        const char* e = std::getenv("LS_SEAM_DUMP_MAXLAYER");
        return (e && *e) ? std::atoi(e) : 4;
    }();
    if (layer > max_layer) return;
    if (!dev_buf || rows <= 0 || hidden <= 0) return;
    if (gpu < 0 || gpu >= static_cast<int>(deps_.device_backends.size())
        || !deps_.device_backends[gpu])
        return;
    static std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return;
    const size_t bytes = static_cast<size_t>(rows) * hidden * 2;  // BF16
    std::vector<uint8_t> host(bytes);
    auto* be = deps_.device_backends[gpu];
    be->set_device();
    void* stream = deps_.stream_manager
        ? deps_.stream_manager->stream(gpu, compute::StreamId::kAttention)
        : nullptr;
    be->memcpy_d2h_async(host.data(), dev_buf, bytes, stream);
    be->synchronize_device();
    int32_t hdr[5] = {static_cast<int32_t>(tag4cc), layer, gpu, rows, hidden};
    std::fwrite(hdr, sizeof(int32_t), 5, fp);
    std::fwrite(host.data(), 1, bytes, fp);
    std::fflush(fp);
}

void CommandDispatcher::register_pcie_token(uint64_t token, uint32_t cmd_seq) {
    token_to_cmd_seq_[token] = cmd_seq;
    cmd_seq_to_token_[cmd_seq] = {token, false};
}

void CommandDispatcher::register_nvme_token(uint64_t token, uint32_t cmd_seq) {
    nvme_token_to_cmd_seq_[token] = cmd_seq;
    cmd_seq_to_token_[cmd_seq] = {token, true};
}

bool CommandDispatcher::setup_spec_pipeline(
        uint32_t cmd_seq, uint32_t gpu, uint32_t token_id,
        const char* pipeline_name, SpecScratch& ss,
        void*& stream_out, int& pair_idx_out) {
    auto* dev = expert_dev(gpu);
    if (!dev) {
        char msg[64];
        std::snprintf(msg, sizeof(msg), "%s: no expert device", pipeline_name);
        write_error(cmd_seq, gpu, ipc::CmpErrorCategory::kComputeValidation, msg);
        return false;
    }
    dev->set_device();
    stream_out = deps_.stream_manager->stream(
        static_cast<int>(gpu), compute::StreamId::kExpertFfn);

    // Write token_id to sideband for embedding lookup.
    auto* token_ids = reinterpret_cast<int32_t*>(
        deps_.sideband_base + ipc::IpcLayout::kTokenIdsOff);
    token_ids[0] = static_cast<int32_t>(token_id);

    // Embedding lookup → hidden_a.
    const auto& mc = deps_.live_config->model;
    pair_idx_out = resolve_pair_idx(gpu);
    const size_t h_bytes = static_cast<size_t>(mc.hidden_size) * 2;
    bool sharded_embed_done = false;

    // TD-GOLDEN-EMB-OOB: vocab-sharded table (TP) — masked per-rank lookup
    // + allreduce fills EVERY rank's attn_buf; hidden_a is then copied back
    // from the primary rank's attn_buf (replaces lookup + broadcast below).
    if (deps_.cuda_kernels_enabled && embedding_tp_degree() > 1) {
        if (!dispatch_embedding_lookup_sharded(1)) {
            char msg[80];
            std::snprintf(msg, sizeof(msg),
                          "%s: TP-sharded embedding lookup failed", pipeline_name);
            write_error(cmd_seq, gpu,
                        ipc::CmpErrorCategory::kComputeValidation, msg);
            return false;
        }
        if (pair_idx_out >= 0
            && deps_.hidden_state_pairs[pair_idx_out].attn_buf && ss.hidden_a) {
            // kAttention (lookup + allreduce) → kExpertFfn (pipeline) handoff.
            void* evt = deps_.device_backends[gpu]->create_event();
            deps_.stream_manager->record_event(
                evt, static_cast<int>(gpu), compute::StreamId::kAttention);
            deps_.device_backends[gpu]->stream_wait_event(stream_out, evt);
            deps_.device_backends[gpu]->memcpy_d2d_async(
                ss.hidden_a, deps_.hidden_state_pairs[pair_idx_out].attn_buf,
                h_bytes, stream_out);
            deps_.device_backends[gpu]->destroy_event(evt);
        }
        sharded_embed_done = true;
    } else if (gpu < deps_.embedding_table_ptrs.size()
               && deps_.embedding_table_ptrs[gpu]) {
        // TD-50w: H2D copy token IDs to device scratch (KD-4b pattern).
        if (gpu >= embedding_token_scratch_.size()
            || !embedding_token_scratch_[gpu]) {
            char msg[80];
            std::snprintf(msg, sizeof(msg),
                          "%s: no embedding token scratch for gpu", pipeline_name);
            write_error(cmd_seq, gpu,
                        ipc::CmpErrorCategory::kComputeValidation, msg);
            return false;
        }
        auto* device_token_ids = static_cast<int32_t*>(
            embedding_token_scratch_[gpu]);
        deps_.device_backends[gpu]->memcpy_h2d_async(
            device_token_ids, token_ids,
            sizeof(int32_t), stream_out);
        compute::launch_embedding_lookup(
            ss.hidden_a, deps_.embedding_table_ptrs[gpu],
            device_token_ids, 1, mc.vocab_size, mc.hidden_size,
            compute::EmbeddingDtype::kBFloat16, stream_out);
    }

    // Copy embedding output to attn_buf and set up sync event.
    if (!sharded_embed_done && deps_.cuda_kernels_enabled && pair_idx_out >= 0
        && deps_.hidden_state_pairs[pair_idx_out].attn_buf) {
        deps_.device_backends[gpu]->memcpy_d2d_async(
            deps_.hidden_state_pairs[pair_idx_out].attn_buf, ss.hidden_a,
            h_bytes, stream_out);

        // TD-73n: Broadcast embedding to all other TP ranks (same pattern
        // as TD-73i in dispatch_compute.cpp).  Source is the primary rank's
        // attn_buf that was just written above.
        if (deps_.dcp_executor
            && deps_.dcp_executor->dcp_size() > 1
            && deps_.hidden_state_pairs.size() > 1) {
            const void* src =
                deps_.hidden_state_pairs[pair_idx_out].attn_buf;
            void* embed_evt =
                deps_.device_backends[gpu]->create_event();
            deps_.device_backends[gpu]->record_event(
                embed_evt, stream_out);

            for (size_t r = 0;
                 r < deps_.hidden_state_pairs.size(); ++r) {
                if (static_cast<int>(r) == pair_idx_out) continue;
                const auto& dp = deps_.hidden_state_pairs[r];
                if (!dp.attn_buf) continue;
                const int dst_pos = dp.gpu_position;
                void* dst_stream =
                    deps_.stream_manager->stream(
                        dst_pos, compute::StreamId::kAttention);
                deps_.device_backends[dst_pos]->set_device();
                // TD-PREFILL-NONDET: dst rank's prior MoE commit must land
                // before this broadcast overwrites its attn_buf (see
                // dispatch_embedding_lookup_sharded for the race account).
                if (dp.moe_attn_event)
                    deps_.device_backends[dst_pos]->stream_wait_event(
                        dst_stream, dp.moe_attn_event);
                deps_.device_backends[dst_pos]->stream_wait_event(
                    dst_stream, embed_evt);
                deps_.device_backends[dst_pos]->memcpy_async(
                    dp.attn_buf, src, h_bytes, dst_stream);
            }

            // Restore device context.
            deps_.device_backends[gpu]->set_device();
            deps_.device_backends[gpu]->destroy_event(embed_evt);
        }
    }
    if (pair_idx_out >= 0 && deps_.hidden_state_pairs[pair_idx_out].moe_attn_event) {
        deps_.stream_manager->record_event(
            deps_.hidden_state_pairs[pair_idx_out].moe_attn_event,
            static_cast<int>(gpu), compute::StreamId::kExpertFfn);
    }
    return true;
}

// ── GG-S1 Phase 4 (TD-GG5C-SPLIT-BUFFER-ALWAYS-ALLOC) ──────────────────────
// Load-time gate≠up scan that feeds the gguf_gate_up_split scratch sizing.
// CUDA-free (INV-GPU-1): just compares already-populated GGUF k-quant enums.
bool CommandDispatcher::any_dense_shared_gate_ne_up(
    const std::vector<std::vector<Deps::DenseFFNWeights>>& dense_ffn_weight_ptrs,
    const std::vector<std::vector<Deps::SharedExpertWeights>>&
        shared_expert_weight_ptrs) {
    for (const auto& per_gpu : dense_ffn_weight_ptrs)
        for (const auto& dw : per_gpu)
            if (dw.gate_gguf_type != dw.up_gguf_type)
                return true;
    for (const auto& per_gpu : shared_expert_weight_ptrs)
        for (const auto& se : per_gpu)
            if (se.gate_gguf_type != se.up_gguf_type)
                return true;
    return false;
}

// ── Construction / destruction ─────────────────────────────────────────────

CommandDispatcher::CommandDispatcher(Deps deps)
    : deps_(std::move(deps)),
      host_source_deps_{
          .pinned_arena     = deps_.pinned_arena,
          .prepacked_source = deps_.prepacked_source,
          .nvme_tier        = deps_.nvme_tier,
          .packed_cache     = deps_.packed_cache,
          .loaded_model     = deps_.loaded_model,
          .expert_shape     = deps_.expert_shape,
      },
      max_inflight_compute_(deps_.max_inflight_compute)
{
    // I8 GPU-loader config (LS_LOADER_* env gates); see dispatch_loader.cpp.
    init_loader_from_env();

    // DET-REDUCE Phase 1b: placement-invariant fp32 EP combine gate. Read once
    // here (engine init) from config; env LAYERSTORM_DETERMINISTIC_EP_COMBINE
    // overrides EITHER way (mirrors engine.cpp's deterministic_reduce pattern).
    // Default OFF (unlike the perf-neutral attention DET-REDUCE this doubles the
    // routed EP-combine allreduce bytes, so it is opt-in).
    if (deps_.live_config)
        deterministic_ep_combine_ = deps_.live_config->compute.deterministic_ep_combine;
    if (const char* e = std::getenv("LAYERSTORM_DETERMINISTIC_EP_COMBINE"); e && *e)
        deterministic_ep_combine_ = (e[0] != '0');
    // DET-REDUCE Phase 1b BF16-PAYLOAD: per-slot payload precision (fp32|bf16).
    // Config default "fp32" = today's canonical; "bf16" halves the gather bytes.
    if (deps_.live_config)
        ep_combine_bf16_payload_ =
            (deps_.live_config->compute.deterministic_ep_combine_precision ==
             config::DeterministicEpCombinePrecision::bf16);
    if (const char* e = std::getenv("LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION");
        e && *e) {
        if (std::strcmp(e, "bf16") == 0) ep_combine_bf16_payload_ = true;
        else if (std::strcmp(e, "fp32") == 0) ep_combine_bf16_payload_ = false;
    }
    spdlog::info("MoE EP combine: deterministic_ep_combine={} precision={} "
                 "(placement-invariant canonical fixed-slot order)",
                 deterministic_ep_combine_,
                 ep_combine_bf16_payload_ ? "bf16" : "fp32");

    // INV-DSA-REWIND: opt-in overwrite-rewind blessing in the indexer
    // coverage guard (see command_dispatcher.h member doc). Read once here
    // from config (compute.dsa_indexer_rewind); env LS_INDEXER_REWIND
    // overrides EITHER way (set & !='0' -> on, '0' -> off, unset defers to
    // config — the LAYERSTORM_DETERMINISTIC_EP_COMBINE precedence pattern).
    if (deps_.live_config)
        indexer_rewind_ok_ = deps_.live_config->compute.dsa_indexer_rewind;
    if (const char* e = std::getenv("LS_INDEXER_REWIND"); e && *e)
        indexer_rewind_ok_ = (e[0] != '0');
    if (indexer_rewind_ok_)
        spdlog::info("DSA coverage guard: indexer rewind ON "
                     "(compute.dsa_indexer_rewind / LS_INDEXER_REWIND) — "
                     "contiguous same-seq overwrite-rewinds stay "
                     "sparse-eligible (INV-DSA-REWIND)");

    // LS_ATTN_CHUNK_PROF: opt-in chunk-attention dispatch x-ray (diag).
    if (const char* e = std::getenv("LS_ATTN_CHUNK_PROF"); e && *e)
        attn_chunk_prof_ = (e[0] != '0');
    if (attn_chunk_prof_)
        spdlog::info("LS_ATTN_CHUNK_PROF=1 — per-command chunk-attention "
                     "dispatch decomposition enabled (diagnostic)");

    // KD-2: allocate per-GPU sampling scratch buffers.
    for (size_t i = 0; i < deps_.expert_devices.size(); ++i) {
        if (deps_.expert_devices[i]) {
            sampling_scratch_.push_back(
                deps_.expert_devices[i]->device_alloc(
                    ipc::kMaxSidebandTokenIds * sizeof(int32_t)));
        } else {
            sampling_scratch_.push_back(nullptr);
        }
    }

    // KD-2a: allocate per-GPU confidence scratch buffers.
    const size_t conf_bytes = ipc::kMaxBatchDescriptors * sizeof(float);
    for (size_t i = 0; i < deps_.expert_devices.size(); ++i) {
        if (deps_.expert_devices[i]) {
            confidence_top1_scratch_.push_back(
                deps_.expert_devices[i]->device_alloc(conf_bytes));
            confidence_entropy_scratch_.push_back(
                deps_.expert_devices[i]->device_alloc(conf_bytes));
        } else {
            confidence_top1_scratch_.push_back(nullptr);
            confidence_entropy_scratch_.push_back(nullptr);
        }
    }
    confidence_host_staging_.resize(deps_.expert_devices.size());

    // KD-3b: allocate per-GPU MoE pipeline scratch buffers.
    // TD-PREFILL-SUPERCHUNK: legacy default row bound (refined below when the
    // model has MoE dims and a superchunk is configured).
    superchunk_rows_ = deps_.max_batch_size;
    moe_scratch_.resize(deps_.expert_devices.size());
    // TD-PREFILL-FETCH-SEAM-SCALING: per-GPU wave-accumulator touch flags.
    moe_wave_accum_used_.assign(deps_.expert_devices.size(), 0);
    // TD-DECODE-FFN-GRAPH: per-GPU variant-keyed routed-FFN graph runner maps
    // (empty until the first eligible decode dispatch captures a variant —
    // see the member comment for the mixed-GGUF variant-key rationale).
    routed_ffn_graphs_.resize(deps_.expert_devices.size());
    if (deps_.live_config
        && deps_.live_config->model.num_experts_per_tok > 0
        && deps_.live_config->model.hidden_size > 0
        && deps_.live_config->model.moe_intermediate_size > 0
        && deps_.live_config->model.n_routed_experts > 0) {
        const auto& mc = deps_.live_config->model;
        const int T = mc.num_experts_per_tok;
        const int H = mc.hidden_size;
        const int I = mc.moe_intermediate_size;
        const int E = mc.n_routed_experts;

        // ── TD-PREFILL-SUPERCHUNK: effective MoE batch capacity + VRAM
        // fail-safe. The MoE pipeline scratch is sized for
        // max(kMaxBatchDescriptors, superchunk_tokens) tokens so ONE
        // FETCH_AND_RUN_MOE / RUN_MOE can batch a whole superchunk per layer.
        // Before allocating, project the per-GPU scratch bytes (conservative
        // upper bound mirroring the allocations below) against the smallest
        // free VRAM across expert devices; halve the requested superchunk
        // until it fits with headroom (floor = kMaxBatchDescriptors — the
        // legacy capacity — so the feature degrades, never OOMs). ──
        {
            const int tp_est = std::max(
                1, deps_.live_config->parallelism.tensor_parallelism);
            const int I_dense_est =
                (mc.first_k_dense_replace > 0 && mc.intermediate_size > 0)
                    ? mc.intermediate_size / tp_est : 0;
            const int max_K_est = std::max({H, I, I_dense_est});
            const int I_local_est = I / tp_est;
            const bool gguf_est = model::gguf::is_gguf_weight_quant(
                deps_.live_config->quantization.weights);
            // TD-PREFILL-MOE-BIG: the projection is split into TRANSIENT terms
            // (reused per chunk under prefill_moe_big — sized at the chunk
            // capacity Bt) and PERSISTENT terms (span the whole batch — sized
            // at the batch capacity Bq). transient(Bq) + persist(Bq) is the
            // exact legacy TD-PREFILL-SUPERCHUNK formula.
            auto transient_bytes = [&](int Bt) -> size_t {
                const size_t exp = static_cast<size_t>(Bt) * T;
                size_t bytes = 0;
                bytes += exp * H * 2;                                   // permuted_input
                bytes += exp * 2 * sizeof(int32_t);                     // src_to_dest+permuted_idx
                bytes += std::max(exp * 2 * I,
                                  static_cast<size_t>(Bt) * 2 * I_dense_est) * 2;
                // activation_output (+ gguf_gate_up_split worst case)
                bytes += 2 * std::max(exp * I,
                                      static_cast<size_t>(Bt) * I_dense_est) * 2;
                bytes += exp * H * 2;                                   // expert_output
                bytes += exp * H * 2;                                   // moe_wave_accum
                bytes += exp * 8 * sizeof(int32_t);                     // permute workspace
                bytes += exp * max_K_est;                               // quant_act (1 B/elem bound)
                bytes += exp * ((max_K_est + 127) / 128) * sizeof(float);  // quant_scale bound
                if (gguf_est) {                                          // GGUF Q8_1 gemm ws
                    bytes += exp * static_cast<size_t>(max_K_est / 32) * 36
                           + 2 * (exp / 64 + E + 1) * sizeof(int32_t);
                }
                if (deterministic_ep_combine_) {
                    bytes += static_cast<size_t>(Bt) * T * H
                           * (ep_combine_bf16_payload_ ? 2 : 4);
                }
                // shared expert gate_up + activation (I_local)
                bytes += static_cast<size_t>(Bt) * 3 * I_local_est * 2;
                return bytes;
            };
            auto persist_bytes = [&](int Bq) -> size_t {
                const size_t exp = static_cast<size_t>(Bq) * T;
                size_t bytes = 0;
                bytes += static_cast<size_t>(Bq) * E * sizeof(float);   // router_logits
                bytes += exp * (sizeof(float) + sizeof(int32_t));       // topk w+i
                bytes += 3 * static_cast<size_t>(Bq) * H * 2;           // moe_output+normalized+shared_out
                if (deps_.moe_hash_layers > 0)
                    bytes += static_cast<size_t>(Bq) * sizeof(int32_t); // moe_token_ids (V4-4)
                return bytes;
            };
            auto scratch_bytes = [&](int Bq) -> size_t {
                return transient_bytes(Bq) + persist_bytes(Bq);
            };

            // TD-PREFILL-MOE-BIG mode + chunk capacity (config knobs).
            moe_big_enabled_ = deps_.live_config->compute.prefill_moe_big;
            const int cfg_chunk = std::clamp(
                deps_.live_config->compute.moe_big_chunk_tokens, 16, 512);
            // Chunk capacity: at least the decode batch bound so every decode /
            // small-batch dispatch stays on the byte-identical single-shot path.
            const int chunk_cap = std::max(cfg_chunk, deps_.max_batch_size);

            const int floor_cap = static_cast<int>(ipc::kMaxBatchDescriptors);
            int want = std::max(deps_.superchunk_tokens, floor_cap);

            size_t min_free = 0;
            bool have_query = false;
            for (auto* dev : deps_.expert_devices) {
                if (!dev) continue;
                size_t f = 0, t = 0;
                if (dev->device_mem_info(f, t)) {
                    min_free = have_query ? std::min(min_free, f) : f;
                    have_query = true;
                }
            }
            constexpr size_t kHeadroom = 1536ull << 20;  // 1.5 GiB

            if (!moe_big_enabled_) {
                // Legacy TD-PREFILL-SUPERCHUNK fail-safe: transient scratch is
                // batch-sized, so halve the request until the FULL projection
                // fits (never OOM, floor = legacy capacity).
                if (want > floor_cap && have_query) {
                    while (want > floor_cap
                           && scratch_bytes(want) + kHeadroom > min_free) {
                        want = std::max(floor_cap, want / 2);
                    }
                    if (want < deps_.superchunk_tokens) {
                        spdlog::warn(
                            "TD-PREFILL-SUPERCHUNK: requested superchunk "
                            "capacity {} tokens does not fit free VRAM "
                            "(min free {} MiB, projected scratch {} MiB + "
                            "1.5 GiB headroom) — stepped down to {} tokens "
                            "(fail-safe, never OOM)",
                            deps_.superchunk_tokens, min_free >> 20,
                            scratch_bytes(deps_.superchunk_tokens) >> 20,
                            want);
                    }
                }
                moe_batch_capacity_ = want;
                moe_chunk_capacity_ = moe_batch_capacity_;  // chunking off
            } else {
                // TD-PREFILL-MOE-BIG elastic fail-safe: transients are
                // chunk-bounded (fixed), so only the PERSISTENT per-token cost
                // limits the batch capacity — derive it from free VRAM instead
                // of halving. Never OOM by construction; single-token GEMM
                // chunks are the pathological floor of the chunk loop, and the
                // legacy 512-token capacity is the floor of the batch bound.
                if (want > floor_cap && have_query) {
                    // Preserve the config safety margin ON TOP of the working
                    // headroom — the elastic capacity must never eat the VRAM
                    // the margin reserves for page tables / driver overhead.
                    const size_t margin = static_cast<size_t>(
                        deps_.live_config->memory.vram_safety_margin_gb
                        * (1ull << 30));
                    // Scratch homing: when EVERY expert-device GPU has a
                    // prefill-scratch tail big enough for the chunk transients,
                    // they cost zero post-region free VRAM — drop the term.
                    bool tail_covers = !deps_.prefill_scratch_tails.empty();
                    for (size_t gi = 0; tail_covers
                         && gi < deps_.expert_devices.size(); ++gi) {
                        if (!deps_.expert_devices[gi]) continue;
                        const size_t tb =
                            (gi < deps_.prefill_scratch_tails.size())
                                ? deps_.prefill_scratch_tails[gi].second : 0;
                        if (tb < transient_bytes(chunk_cap))
                            tail_covers = false;
                    }
                    // Working headroom ONLY (1.5 GiB) — unlike the engine-
                    // level derivation (pre-arena-registration, must reserve
                    // the margin for page tables/driver still to come), this
                    // re-verify runs at construction time when those costs
                    // are MATERIALIZED and already excluded from min_free.
                    // Re-reserving the config margin here double-counted it
                    // and stepped the capacity back to the 512 floor
                    // (min_free 2768 MiB < margin 3 GiB on the keeper52
                    // shape) after the engine had correctly derived ~13k.
                    (void)margin;
                    const size_t fixed =
                        (tail_covers ? 0 : transient_bytes(chunk_cap))
                        + kHeadroom;
                    while (want > floor_cap
                           && persist_bytes(want) + fixed > min_free) {
                        want = std::max(floor_cap, want - std::max(256, want / 8));
                    }
                    if (want < deps_.superchunk_tokens) {
                        spdlog::warn(
                            "TD-PREFILL-MOE-BIG: requested superchunk capacity "
                            "{} tokens exceeds the elastic bound (min free {} "
                            "MiB, chunk transients {} MiB + 1.5 GiB headroom, "
                            "persistent ~{} B/token) — derived {} tokens "
                            "(elastic, never OOM)",
                            deps_.superchunk_tokens, min_free >> 20,
                            transient_bytes(chunk_cap) >> 20,
                            persist_bytes(1024) / 1024, want);
                    }
                }
                moe_batch_capacity_ = want;
                moe_chunk_capacity_ = std::min(chunk_cap, moe_batch_capacity_);
            }

            superchunk_rows_ = std::max(
                deps_.max_batch_size,
                std::min(deps_.superchunk_tokens, moe_batch_capacity_));
            if (moe_batch_capacity_ > floor_cap || moe_big_enabled_) {
                spdlog::info(
                    "{}: MoE batch capacity {} tokens, single-shot chunk "
                    "capacity {} tokens (transient scratch ~{} MiB/GPU, "
                    "persistent ~{} MiB/GPU)",
                    moe_big_enabled_ ? "TD-PREFILL-MOE-BIG"
                                     : "TD-PREFILL-SUPERCHUNK",
                    moe_batch_capacity_, moe_chunk_capacity_,
                    transient_bytes(moe_chunk_capacity_) >> 20,
                    persist_bytes(moe_batch_capacity_) >> 20);
            }
        }

        const int B = moe_batch_capacity_;
        const int expanded = B * T;
        // TD-PREFILL-MOE-BIG: transient (chunk-reused) buffer row basis. Equals
        // B/expanded when prefill_moe_big is off (legacy byte-identical sizing).
        const int Bt = moe_chunk_capacity_;
        const int expanded_t = Bt * T;

        // TD-PREFILL-MOE-BIG scratch homing: the chunk-bounded TRANSIENT
        // buffers below are carved from the reserved kv_main prefill-scratch
        // tail (Deps::prefill_scratch_tails) via moe_scratch_alloc — zero new
        // VRAM pressure. Only when prefill_moe_big is on; otherwise the spans
        // stay empty and every allocation falls through to device_alloc
        // (byte-identical legacy behaviour).
        moe_tail_span_.assign(deps_.expert_devices.size(), {nullptr, 0});
        moe_tail_used_.assign(deps_.expert_devices.size(), 0);
        if (moe_big_enabled_) {
            for (size_t i = 0; i < deps_.expert_devices.size()
                               && i < deps_.prefill_scratch_tails.size(); ++i) {
                const auto& [ptr, bytes] = deps_.prefill_scratch_tails[i];
                if (ptr && bytes > 0)
                    moe_tail_span_[i] = {static_cast<char*>(ptr), bytes};
            }
        }

        // KD-4f-c: dense FFN intermediate dimension for buffer sizing.
        const int tp = std::max(1, deps_.live_config->parallelism.tensor_parallelism);
        const int I_dense = (mc.first_k_dense_replace > 0 && mc.intermediate_size > 0)
                          ? mc.intermediate_size / tp : 0;

        size_t ws_nvfp4_gu = compute::query_nvfp4_grouped_gemm_workspace_size(
            E, 2 * I, H, compute::GemmOutputDtype::kBFloat16);
        size_t ws_nvfp4_d = compute::query_nvfp4_grouped_gemm_workspace_size(
            E, H, I, compute::GemmOutputDtype::kBFloat16);
        size_t ws_fp8_gu = compute::query_fp8_grouped_gemm_workspace_size(
            E, 2 * I, H, compute::GemmOutputDtype::kBFloat16);
        size_t ws_fp8_d = compute::query_fp8_grouped_gemm_workspace_size(
            E, H, I, compute::GemmOutputDtype::kBFloat16);
        size_t gemm_ws = std::max({ws_nvfp4_gu, ws_nvfp4_d, ws_fp8_gu, ws_fp8_d});

        // KD-4f-c: include dense FFN dimensions in workspace query.
        if (I_dense > 0) {
            size_t ws_dense_nvfp4_gu = compute::query_nvfp4_grouped_gemm_workspace_size(
                1, 2 * I_dense, H, compute::GemmOutputDtype::kBFloat16);
            size_t ws_dense_nvfp4_d = compute::query_nvfp4_grouped_gemm_workspace_size(
                1, H, I_dense, compute::GemmOutputDtype::kBFloat16);
            size_t ws_dense_fp8_gu = compute::query_fp8_grouped_gemm_workspace_size(
                1, 2 * I_dense, H, compute::GemmOutputDtype::kBFloat16);
            size_t ws_dense_fp8_d = compute::query_fp8_grouped_gemm_workspace_size(
                1, H, I_dense, compute::GemmOutputDtype::kBFloat16);
            gemm_ws = std::max({gemm_ws, ws_dense_nvfp4_gu, ws_dense_nvfp4_d,
                                ws_dense_fp8_gu, ws_dense_fp8_d});
        }

        // GG-5b/GG-5d: the GGUF int-strategy grouped GEMM (now device-fused)
        // quantizes ALL permuted activations to Q8_1 ONCE, then — for the mmq
        // sub-path — appends a flattened tile→expert routing map. Mirror the
        // kernel's gguf_grouped_gemm_workspace_bytes(total_tokens, K, num_experts)
        // CUDA-free (its header pulls in <cuda_runtime.h>; can't include here —
        // INV-GPU-1):
        //   q8       = total_tokens · (K/32) · sizeof(block_q8_1)
        //   map_off  = align16(q8)
        //   Tmax     = ceil(total_tokens/64) + num_experts   (tile_m=64 worst case)
        //   ws       = map_off + 2·Tmax·sizeof(int32_t)
        // total_tokens = expanded (routed upper bound; dense/shared B ≤ expanded);
        // num_experts = E (routed; dense/shared use 1, smaller tile map); K = max
        // projection input dim (H for gate/up, I/I_dense for down). The block_q8_1
        // constants (QK8_1=32, BLOCK_Q8_1_SIZE=36 bytes) mirror
        // deps/.../formats/q8_1_format.h (same mirror rationale as
        // TD-GGUF-BLOCK-CONST-MIRROR / TD-GGUF-Q8_1-WS-CONST-MIRROR — the tile-map
        // term EXTENDS that latter debt). The dequant strategy needs 0 bytes; we
        // size for int unconditionally (cheap; covers both strategies at runtime).
        {
            const auto wq_alloc = deps_.live_config->quantization.weights;
            if (model::gguf::is_gguf_weight_quant(wq_alloc)) {
                constexpr int kQk8_1 = 32;
                constexpr size_t kBlockQ8_1Bytes = 36;
                constexpr int kMmqTileM = 64;
                const int K_max = std::max({H, I, I_dense});
                // TD-PREFILL-MOE-BIG: the GEMM workspace is transient (reused
                // per chunk) — size it for the chunk row bound expanded_t
                // (== expanded when prefill_moe_big is off).
                const size_t q8 =
                    static_cast<size_t>(expanded_t) *
                    static_cast<size_t>(K_max / kQk8_1) * kBlockQ8_1Bytes;
                const size_t map_off = (q8 + 15) & ~static_cast<size_t>(15);
                const size_t tmax = static_cast<size_t>(
                    (expanded_t + kMmqTileM - 1) / kMmqTileM) +
                    static_cast<size_t>(E);
                const size_t gguf_ws =
                    map_off + 2ull * tmax * sizeof(int32_t);
                gemm_ws = std::max(gemm_ws, gguf_ws);
            }
        }

        // GG-S1 Phase 4 (TD-GG5C-SPLIT-BUFFER-ALWAYS-ALLOC): scan the per-layer
        // dense/shared GGUF k-quant types — populated by engine.cpp upload_ffn
        // BEFORE this dispatcher is constructed (deps moved in at line ~2355) —
        // to decide whether the dense/shared gate_up SPLIT path can ever fire.
        // ORDERING: this read happens at sizing time and is safe precisely
        // because upload_ffn ran first; reading these vectors any earlier (the
        // GG-S1 deferral trap) would size 0 and the split path would touch an
        // unallocated buffer. All-gate==up models ⇒ false ⇒ buffer gated off.
        const bool any_dense_shared_gate_ne_up_ =
            any_dense_shared_gate_ne_up(deps_.dense_ffn_weight_ptrs,
                                        deps_.shared_expert_weight_ptrs);

        for (size_t i = 0; i < deps_.expert_devices.size(); ++i) {
            auto* dev = deps_.expert_devices[i];
            if (!dev) continue;
            auto& s = moe_scratch_[i];
            s.router_logits     = dev->device_alloc(static_cast<size_t>(B) * E * sizeof(float));
            s.topk_weights      = dev->device_alloc(static_cast<size_t>(expanded) * sizeof(float));
            s.topk_indices      = dev->device_alloc(static_cast<size_t>(expanded) * sizeof(int32_t));
            // V4-4 hash gating: persistent per-step token ids (row-offset
            // aware; written by the embedding handler).
            if (deps_.moe_hash_layers > 0)
                s.moe_token_ids = dev->device_alloc(
                    static_cast<size_t>(B) * sizeof(int32_t));
            // TD-PREFILL-MOE-BIG: TRANSIENT buffers (permute staging, GEMM
            // in/out, wave accumulator, quant) are sized at the CHUNK row bound
            // Bt/expanded_t and reused across chunks by the chunked path;
            // PERSISTENT buffers (routing, moe_output, normalized hidden,
            // shared output) span the full batch capacity B. Bt == B when
            // prefill_moe_big is off (legacy byte-identical sizing).
            // Transients go through moe_scratch_alloc → homed in the idle
            // kv_main prefill-scratch tail when available (device_alloc
            // fallback); persistents always device_alloc (they scale with B).
            s.permuted_input    = moe_scratch_alloc(i, static_cast<size_t>(expanded_t) * H * 2);
            s.expert_offsets    = dev->device_alloc(static_cast<size_t>(E + 1) * sizeof(int32_t));
            s.src_to_dest_map   = moe_scratch_alloc(i, static_cast<size_t>(expanded_t) * sizeof(int32_t));
            s.permuted_idx      = moe_scratch_alloc(i, static_cast<size_t>(expanded_t) * sizeof(int32_t));
            // KD-4f-c: size for max(routed MoE, dense FFN) dimensions.
            const size_t gu_routed = static_cast<size_t>(expanded_t) * 2 * I;
            const size_t gu_dense  = static_cast<size_t>(Bt) * 2 * I_dense;
            s.gate_up_output    = moe_scratch_alloc(i, std::max(gu_routed, gu_dense) * 2);
            const size_t act_routed = static_cast<size_t>(expanded_t) * I;
            const size_t act_dense  = static_cast<size_t>(Bt) * I_dense;
            s.activation_output = moe_scratch_alloc(i, std::max(act_routed, act_dense) * 2);
            s.expert_output     = moe_scratch_alloc(i, static_cast<size_t>(expanded_t) * H * 2);
            s.moe_output        = dev->device_alloc(static_cast<size_t>(B) * H * 2);
            // TD-PREFILL-FETCH-SEAM-SCALING: rolling-wave accumulator, same
            // shape as expert_output ([expanded_t, H] BF16). Waves add their
            // Step-5 rows here; the final pass unpermutes from it. Only used
            // by single-shot (<= Bt tokens) wave commands — the chunked path
            // accumulates into moe_output instead (INV-MOE-BIG-2).
            s.moe_wave_accum    = moe_scratch_alloc(i, static_cast<size_t>(expanded_t) * H * 2);
            // TD-PREFILL-MOE-BIG: chunk-partial unpermute staging for chunked
            // wave passes ([Bt, H] BF16).
            if (moe_big_enabled_) {
                s.big_unperm_tmp = moe_scratch_alloc(
                    i, static_cast<size_t>(Bt) * H * 2);
                // TD-MOE-BIG-GEMM-SWEEP: wave-masked permute scratch — per-
                // chunk masked top-K staging + device per-expert byte mask.
                s.wave_masked_topk = moe_scratch_alloc(
                    i, static_cast<size_t>(expanded_t) * sizeof(int32_t));
                s.wave_expert_mask = dev->device_alloc(static_cast<size_t>(E));
                s.wave_expert_mask_host.assign(static_cast<size_t>(E), 0);
            }
            // DET-REDUCE Phase 1b (canonical): per-slot buffer [Bt, topk, H] for
            // the placement-invariant EP combine. Each of the K expert
            // contributions occupies its own slot; the cross-GPU allreduce gathers
            // slots, then a fixed-order K-reduce → bf16. Only allocated when
            // enabled, and only the active-precision buffer (fp32 = Bt*topk*H*4;
            // bf16 = Bt*topk*H*2, half the bytes). Default off ⇒ both nullptr.
            // TD-PREFILL-MOE-BIG: sized at the single-shot bound Bt — the
            // canonical combine only runs for single-shot dispatches (the
            // chunked path forces the legacy mode-0 combine, INV-MOE-BIG-3).
            if (deterministic_ep_combine_) {
                if (ep_combine_bf16_payload_)
                    s.moe_output_bf16_perslot = moe_scratch_alloc(
                        i, static_cast<size_t>(Bt) * T * H * 2);
                else
                    s.moe_output_fp32 = moe_scratch_alloc(
                        i, static_cast<size_t>(Bt) * T * H * 4);
            }
            s.normalized_hidden = dev->device_alloc(static_cast<size_t>(B) * H * 2);
            // V4-5b mHC FFN-stage scratch (persistent, full batch capacity).
            if (deps_.hc_streams > 1) {
                const size_t hc = static_cast<size_t>(deps_.hc_streams);
                s.hc_x    = dev->device_alloc(static_cast<size_t>(B) * H * 2);
                s.hc_post = dev->device_alloc(static_cast<size_t>(B) * hc * sizeof(float));
                s.hc_comb = dev->device_alloc(static_cast<size_t>(B) * hc * hc * sizeof(float));
            }
            s.permute_workspace = moe_scratch_alloc(i, static_cast<size_t>(expanded_t) * 8 * sizeof(int32_t));
            if (gemm_ws > 0)
                s.gemm_workspace = moe_scratch_alloc(i, gemm_ws);
            s.gemm_workspace_bytes = gemm_ws;

            // KD-3f: grouped GEMM metadata buffers (routed experts).
            s.problem_sizes = dev->device_alloc(static_cast<size_t>(E) * 3 * sizeof(int32_t));
            s.sf_offsets    = dev->device_alloc(static_cast<size_t>(E + 1) * sizeof(int32_t));

            // KD-4f-c2: per-expert B pointer arrays for routed GEMM dispatch.
            const size_t ptr_arr_bytes = static_cast<size_t>(E) * sizeof(void*);
            s.routed_b_ptrs  = dev->device_alloc(ptr_arr_bytes);
            s.routed_sb_ptrs = dev->device_alloc(ptr_arr_bytes);
            s.routed_b_ptrs_host.resize(E, nullptr);
            s.routed_sb_ptrs_host.resize(E, nullptr);

            // GG-5b: 1-element device B_ptrs array for dense/shared GGUF GEMMs.
            if (model::gguf::is_gguf_weight_quant(
                    deps_.live_config->quantization.weights)) {
                s.gguf_single_b_ptr = dev->device_alloc(sizeof(void*));
                // GG-5c: split-path up GEMM output buffer, sized like
                // activation_output ([B, I_dense] dominates [expanded, I]) so it
                // holds the dense/shared up half regardless of intermediate size.
                // GG-S1 Phase 4 (TD-GG5C-SPLIT-BUFFER-ALWAYS-ALLOC): only the
                // dense/shared SPLIT path (gate_gguf_type != up_gguf_type) reads
                // this buffer. Allocate it ONLY when some dense/shared unit is
                // gate≠up; otherwise leave it null and sized 0 (dead VRAM saved
                // on the common all-gate==up case). The split numerics are
                // unaffected — launch_gguf_dense_gate_up asserts up_scratch !=
                // null on the split branch (defensive; impossible once gated).
                if (any_dense_shared_gate_ne_up_) {
                    s.gguf_gate_up_split =
                        moe_scratch_alloc(i, std::max(act_routed, act_dense) * 2);
                }
            }

            // TD-DECODE-FFN-GRAPH (experiment): per-projection device arrays +
            // PINNED host staging for graph-capturable b_ptrs feeds. Only when
            // the FFN graph is enabled (avoids 6 extra device allocs + 6 pinned
            // host allocs per GPU on the default path). Pinned host is required
            // because pageable cudaMemcpyAsync is illegal during graph capture.
            if (moe_ffn_graph_enabled() && i < deps_.device_backends.size()
                && deps_.device_backends[i]) {
                auto* be = deps_.device_backends[i];
                for (int p = 0; p < 3; ++p) {
                    s.g_b_ptrs[p]  = dev->device_alloc(ptr_arr_bytes);
                    s.g_sb_ptrs[p] = dev->device_alloc(ptr_arr_bytes);
                    s.g_b_ptrs_host[p]  = static_cast<const void**>(
                        be->host_alloc_pinned(ptr_arr_bytes));
                    s.g_sb_ptrs_host[p] = static_cast<const void**>(
                        be->host_alloc_pinned(ptr_arr_bytes));
                    // INV-MOE-OVERLAP: separate set for the captured kPartial
                    // (resident-overlap) pass — its graph H2D nodes read the
                    // pinned staging asynchronously while the host refills the
                    // kFinal set for the same layer.
                    s.g_b_ptrs_w[p]  = dev->device_alloc(ptr_arr_bytes);
                    s.g_sb_ptrs_w[p] = dev->device_alloc(ptr_arr_bytes);
                    s.g_b_ptrs_host_w[p]  = static_cast<const void**>(
                        be->host_alloc_pinned(ptr_arr_bytes));
                    s.g_sb_ptrs_host_w[p] = static_cast<const void**>(
                        be->host_alloc_pinned(ptr_arr_bytes));
                }
            }

            // TD-89m: zero weight buffer for missing experts in partial execution.
            // Must be >= max(gate_bytes, up_bytes, down_bytes) so gather_alphas
            // reads at alpha_offset within bounds. Reads produce 0.0f alpha → zero contribution.
            if (deps_.expert_cache) {
                const int64_t max_proj = std::max({deps_.expert_cache->gate_bytes(),
                                                   deps_.expert_cache->up_bytes(),
                                                   deps_.expert_cache->down_bytes()});
                if (max_proj > 0) {
                    s.zero_weight_buf_bytes = static_cast<size_t>(max_proj);
                    s.zero_weight_buf = dev->device_alloc(s.zero_weight_buf_bytes);
                    if (i < deps_.device_backends.size() && deps_.device_backends[i]) {
                        deps_.device_backends[i]->memset_async(
                            s.zero_weight_buf, 0, s.zero_weight_buf_bytes, nullptr);
                    }
                }
            }
            // Host staging for D2H of topk_indices (for miss counting).
            s.topk_indices_host.resize(static_cast<size_t>(B) * T, 0);
            // Resident bitset: 1 bit per expert.
            s.expert_resident_bitset.resize(static_cast<size_t>((E + 7) / 8), 0);

            // TD-81g: NVFP4 alpha buffer — [E] floats for routed, [1] for shared/dense.
            // Allocated here (not in the shared expert block) so models with routed
            // experts but no shared experts still get the buffer.
            s.nvfp4_alpha = dev->device_alloc(static_cast<size_t>(E) * sizeof(float));
            // FP4-ACT-SCALE: per-expert activation input_scales (same shape).
            s.moe_input_scales =
                dev->device_alloc(static_cast<size_t>(E) * sizeof(float));

            // KD-3g: activation quantization scratch buffers.
            // TD-GOLDEN-QSCALE-SIZE: the dense down-GEMM quantizes K = I_dense
            // (intermediate_size/tp), which exceeds both H and I — include it.
            const auto wq = deps_.live_config->quantization.weights;
            const bool use_fp8_alloc = (wq == config::WeightQuant::fp8_e4m3 ||
                                        wq == config::WeightQuant::fp8_e5m2);
            const int max_K = std::max({H, I, I_dense});
            // TD-PREFILL-MOE-BIG: transient (chunk row bound expanded_t).
            if (use_fp8_alloc) {
                s.quant_act_bytes   = static_cast<size_t>(expanded_t) * max_K;
                s.quant_scale_bytes = static_cast<size_t>(expanded_t)
                                    * ((max_K + 127) / 128) * sizeof(float);
            } else {
                s.quant_act_bytes   = static_cast<size_t>(expanded_t) * max_K / 2;
                s.quant_scale_bytes = static_cast<size_t>(std::min(expanded_t, E)) * 128
                                    * ((max_K + 15) / 16);
            }
            s.quant_act   = moe_scratch_alloc(i, s.quant_act_bytes);
            s.quant_scale = moe_scratch_alloc(i, s.quant_scale_bytes);

            // TD-GOLDEN: register MoE intra-layer scratch in the BufferRegistry
            // (registration only — enables layer-level debug readback by name).
            if (deps_.buffer_registry) {
                auto reg = [&](void* ptr, size_t bytes, const char* name) {
                    if (ptr) {
                        deps_.buffer_registry->register_buffer(
                            ptr, static_cast<int64_t>(bytes), static_cast<int>(i),
                            ("moe_scratch." + std::string(name) + ".gpu"
                             + std::to_string(i)).c_str());
                    }
                };
                reg(s.router_logits, static_cast<size_t>(B) * E * sizeof(float),
                    "router_logits");
                reg(s.topk_weights, static_cast<size_t>(expanded) * sizeof(float),
                    "topk_weights");
                reg(s.topk_indices, static_cast<size_t>(expanded) * sizeof(int32_t),
                    "topk_indices");
                reg(s.moe_token_ids, static_cast<size_t>(B) * sizeof(int32_t),
                    "moe_token_ids");
                reg(s.expert_output, static_cast<size_t>(expanded_t) * H * 2,
                    "expert_output");
                reg(s.moe_output, static_cast<size_t>(B) * H * 2, "moe_output");
                reg(s.normalized_hidden, static_cast<size_t>(B) * H * 2,
                    "normalized_hidden");
                reg(s.hc_x, static_cast<size_t>(B) * H * 2, "hc_x");
                reg(s.hc_post,
                    static_cast<size_t>(B) * deps_.hc_streams * sizeof(float),
                    "hc_post");
                reg(s.hc_comb,
                    static_cast<size_t>(B) * deps_.hc_streams * deps_.hc_streams *
                        sizeof(float),
                    "hc_comb");
                reg(s.gate_up_output, std::max(gu_routed, gu_dense) * 2,
                    "gate_up_output");
                reg(s.activation_output, std::max(act_routed, act_dense) * 2,
                    "activation_output");
                reg(s.gguf_gate_up_split, std::max(act_routed, act_dense) * 2,
                    "gguf_gate_up_split");
            }
        }

        // INV-MOE-EP-XTP: expert-only (non-DCP) GPU discovery + per-TP-rank
        // incoming-partial staging. EP degree beyond TP: GPUs that host an
        // ExpertDevice but are not DCP ranks run routed expert subsets whose
        // partials are D2D-folded onto a TP rank before the EP combine
        // (dispatch_moe_ep_extras). Only when CUDA is live and a multi-rank
        // DCP executor exists — otherwise the vector stays empty and every
        // EP-XTP hook is a structural no-op.
        if (deps_.cuda_kernels_enabled && deps_.dcp_executor
            && deps_.dcp_executor->dcp_size() >= 2) {
            const auto& tp_gpus = deps_.dcp_executor->gpus();
            auto is_tp = [&](int pos) {
                for (const auto& g : tp_gpus)
                    if (g.position == pos) return true;
                return false;
            };
            for (size_t i = 0; i < deps_.expert_devices.size(); ++i) {
                if (deps_.expert_devices[i] && !is_tp(static_cast<int>(i)))
                    ep_xtp_gpus_.push_back(static_cast<int>(i));
            }
            if (!ep_xtp_gpus_.empty()) {
                // Staging on each TP rank: max(mode-0 [B,H] bf16, per-slot
                // [Bt,K,H] at the active payload precision).
                size_t staging = static_cast<size_t>(B) * H * 2;
                if (deterministic_ep_combine_) {
                    staging = std::max(staging,
                        static_cast<size_t>(Bt) * T * H
                            * (ep_combine_bf16_payload_ ? 2u : 4u));
                }
                for (const auto& g : tp_gpus) {
                    const int pos = g.position;
                    if (pos < 0
                        || static_cast<size_t>(pos) >= moe_scratch_.size()
                        || !deps_.expert_devices[pos])
                        continue;
                    moe_scratch_[pos].ep_xtp_staging =
                        deps_.expert_devices[pos]->device_alloc(staging);
                }
                spdlog::info("INV-MOE-EP-XTP: {} expert-only GPU(s) beyond the "
                             "{}-rank DCP group — routed EP spans {} GPUs "
                             "(staging {} B per TP rank)",
                             ep_xtp_gpus_.size(), tp_gpus.size(),
                             ep_xtp_gpus_.size() + tp_gpus.size(), staging);
            }
        }
    }

    // KD-3e: allocate shared expert scratch buffers (sized for B tokens, not expanded).
    // KD-3d-fix: uses TP-local intermediate size (column-parallel sharding).
    if (deps_.live_config
        && deps_.live_config->model.n_shared_experts > 0
        && deps_.live_config->model.hidden_size > 0
        && deps_.live_config->model.moe_intermediate_size > 0) {
        const auto& mc = deps_.live_config->model;
        // TD-PREFILL-SUPERCHUNK: shared-expert scratch must cover the same
        // token batch as the routed scratch (Step 7 runs over num_tokens).
        // TD-PREFILL-MOE-BIG: the gate_up/activation staging is TRANSIENT
        // (the chunked path loops the shared-expert GEMMs per chunk) — sized
        // at the chunk bound Bt; only shared_expert_output persists across
        // chunks (row-offset writes). Bt == B when prefill_moe_big is off.
        const int B = moe_batch_capacity_;
        const int Bt = moe_chunk_capacity_;
        const int H = mc.hidden_size;
        const int tp = std::max(1, deps_.live_config->parallelism.tensor_parallelism);
        const int I_local = mc.moe_intermediate_size / tp;
        for (size_t i = 0; i < deps_.expert_devices.size(); ++i) {
            auto* dev = deps_.expert_devices[i];
            if (!dev || i >= moe_scratch_.size()) continue;
            auto& s = moe_scratch_[i];
            s.shared_gate_up_output = moe_scratch_alloc(i, static_cast<size_t>(Bt) * 2 * I_local * 2);
            s.shared_activation     = moe_scratch_alloc(i, static_cast<size_t>(Bt) * I_local * 2);
            s.shared_expert_output  = dev->device_alloc(static_cast<size_t>(B) * H * 2);
            s.shared_expert_offsets  = dev->device_alloc(2 * sizeof(int32_t));
            s.shared_problem_sizes  = dev->device_alloc(3 * sizeof(int32_t));
            s.shared_sf_offsets     = dev->device_alloc(2 * sizeof(int32_t));
            // nvfp4_alpha: normally allocated in the routed expert block above (TD-81g).
            // Fallback for models with shared experts but no routed experts.
            if (!s.nvfp4_alpha) {
                s.nvfp4_alpha = dev->device_alloc(sizeof(float));
            }
            if (!s.moe_input_scales) {
                s.moe_input_scales = dev->device_alloc(sizeof(float));
            }
        }
    }

    // KD-3c: allocate per-GPU speculation pipeline scratch buffers.
    spec_scratch_.resize(deps_.expert_devices.size());
    if (deps_.live_config && deps_.cuda_kernels_enabled) {
        const auto& mc = deps_.live_config->model;
        const int H = mc.hidden_size;
        const int V = mc.vocab_size;
        for (size_t i = 0; i < deps_.expert_devices.size(); ++i) {
            auto* dev = deps_.expert_devices[i];
            if (!dev) continue;
            auto& ss = spec_scratch_[i];
            ss.hidden_a    = dev->device_alloc(static_cast<size_t>(H) * 2);
            ss.hidden_b    = dev->device_alloc(static_cast<size_t>(H) * 2);
            ss.logits      = dev->device_alloc(static_cast<size_t>(V) * sizeof(float));
            ss.cos_sim_out = dev->device_alloc(sizeof(float));
            // Sized for a batched-verify head: up to
            // kMaxOutputHeadReadbackTokens sampled token ids (i32) + the
            // trailing top1_prob (f32).  Single-token pipelines use the
            // first 8 bytes exactly as before.
            ss.readback    = dev->device_alloc(
                (static_cast<size_t>(ipc::kMaxOutputHeadReadbackTokens) + 1)
                * sizeof(int32_t));
            ss.mtp_concat  = dev->device_alloc(static_cast<size_t>(H) * 4);
        }
    }

    // KD-4b: allocate per-rank device scratch for embedding token ID H2D copy.
    for (size_t i = 0; i < deps_.attention_devices.size(); ++i) {
        if (deps_.attention_devices[i]) {
            embedding_token_scratch_.push_back(
                deps_.attention_devices[i]->device_alloc(
                    ipc::kMaxSidebandTokenIds * sizeof(int32_t)));
        } else {
            embedding_token_scratch_.push_back(nullptr);
        }
    }

    // KD-4b: allocate per-rank device scratch for final RMSNorm output.
    if (deps_.live_config && deps_.live_config->model.hidden_size > 0) {
        const size_t norm_buf_size = static_cast<size_t>(deps_.max_batch_size)
                                   * deps_.live_config->model.hidden_size * 2;  // BF16
        for (size_t i = 0; i < deps_.attention_devices.size(); ++i) {
            if (deps_.attention_devices[i]) {
                output_norm_scratch_.push_back(
                    deps_.attention_devices[i]->device_alloc(norm_buf_size));
            } else {
                output_norm_scratch_.push_back(nullptr);
            }
        }
    }

    // KD-4f: allocate per-rank logits scratch for CMD_OUTPUT_HEAD / CMD_SAMPLE_TOKENS.
    if (deps_.live_config && deps_.live_config->model.vocab_size > 0) {
        const size_t logits_bytes = static_cast<size_t>(std::max(deps_.max_batch_size, 1))
                                  * deps_.live_config->model.vocab_size * sizeof(float);
        for (size_t i = 0; i < deps_.attention_devices.size(); ++i) {
            if (deps_.attention_devices[i]) {
                void* buf = deps_.attention_devices[i]->device_alloc(logits_bytes);
                logits_scratch_.push_back(buf);
                if (deps_.buffer_registry) {
                    deps_.buffer_registry->register_buffer(
                        buf, static_cast<int64_t>(logits_bytes),
                        static_cast<int>(i),
                        ("logits_scratch.pos" + std::to_string(i)).c_str());
                }
            } else {
                logits_scratch_.push_back(nullptr);
            }
        }
        // TD-SERVE-NAMED-TOOL-CHOICE / TD-ORCH-SAMPLED-SPEC: pinned host
        // region for the full-logits readback (readback_logits=1 D2H
        // target).  kMaxLogitsReadbackRows rows: row 0 serves the B=1
        // guided-decoding masked sampling (historical single-row layout
        // byte-identical); the speculative sampled/logprobs verify chunk
        // reads the first min(num_tokens, rows) head-batch rows.
        if (!deps_.device_backends.empty() && deps_.device_backends[0]) {
            logits_readback_bytes_ = static_cast<size_t>(
                deps_.live_config->model.vocab_size) * sizeof(float)
                * ipc::kMaxLogitsReadbackRows;
            logits_readback_host_ =
                deps_.device_backends[0]->host_alloc_pinned(
                    logits_readback_bytes_);
        }
    }

    // KD-4g: allocate per-rank partial logits scratch for TP>1 output head.
    if (deps_.live_config && deps_.dcp_communicator
        && deps_.dcp_communicator->is_active()
        && deps_.live_config->model.vocab_size > 0) {
        const int tp = deps_.dcp_communicator->dcp_size();
        const int local_vocab = deps_.live_config->model.vocab_size / tp;
        const size_t partial_bytes = static_cast<size_t>(std::max(deps_.max_batch_size, 1))
                                   * local_vocab * sizeof(float);
        partial_logits_scratch_.resize(deps_.device_backends.size(), nullptr);
        for (size_t i = 0; i < deps_.attention_devices.size(); ++i) {
            if (deps_.attention_devices[i]) {
                partial_logits_scratch_[i] =
                    deps_.attention_devices[i]->device_alloc(partial_bytes);
            }
        }

        // TD-72a: allgather recv scratch for num_tokens>1 transpose.
        // Allocated on the first TP GPU (primary for output head commands).
        if (deps_.max_batch_size > 1 && !deps_.dcp_executor->gpus().empty()) {
            const size_t full_bytes = static_cast<size_t>(deps_.max_batch_size)
                                    * deps_.live_config->model.vocab_size * sizeof(float);
            const int pg = deps_.dcp_executor->gpus()[0].position;
            logits_gather_scratch_gpu_ = static_cast<uint32_t>(pg);
            if (static_cast<size_t>(pg) < deps_.attention_devices.size()
                && deps_.attention_devices[pg]) {
                logits_gather_scratch_ =
                    deps_.attention_devices[pg]->device_alloc(full_bytes);
            }
        }
    }

    // KD-4e1: compute page limits from config (needed for ensure_pages + KV scratch).
    if (deps_.live_config) {
        const int page_size = deps_.kv_page_size > 0 ? deps_.kv_page_size : 64;
        const int max_seq = deps_.live_config->serving.max_sequence_length;
        max_blocks_per_seq_ = (max_seq + page_size - 1) / page_size;
        const int chunk_tok =
            deps_.live_config->memory.kv_cache.page_growth_chunk_tokens;
        chunk_size_pages_ = (chunk_tok + page_size - 1) / page_size;
    }

    // TD-GOLDEN: per-layer KV pages — one physical page per (logical page,
    // layer), including the MTP layer's attention.
    if (deps_.live_config) {
        kv_layers_ = deps_.live_config->model.num_hidden_layers
                   + deps_.live_config->model.num_nextn_predict_layers;
        if (kv_layers_ <= 0) kv_layers_ = 1;

        // TD-V4-KMAIN-SIZING: V4 kMain (CSA bucket) pages are per-CSA-layer
        // only — HCA/SWA(/MTP) layers live in the side pools and get sentinel
        // seq_pages_ slots (see command_dispatcher.h). Mirrors the
        // vram_allocator V4 sizer (csa_pages = blocks × num_csa_layers).
        if (deps_.live_config->model.architecture
                == config::Architecture::deepseek_v4) {
            model::ModelConfig mc(deps_.live_config->model);
            v4_kmain_layer_.assign(static_cast<size_t>(kv_layers_), 0);
            const int hidden = deps_.live_config->model.num_hidden_layers;
            for (int l = 0; l < hidden && l < kv_layers_; ++l) {
                v4_kmain_layer_[static_cast<size_t>(l)] =
                    mc.attention_type_for_layer(l)
                        == model::V4AttentionType::kCsa;
            }
            // MTP (nextn) layers are SWA-only by spec — mask stays 0.
        }
    }

    // KD-4e: allocate per-rank KV cache metadata scratch.
    if (deps_.live_config && deps_.dcp_executor && max_blocks_per_seq_ > 0) {
        const int dcp_size = deps_.dcp_executor->dcp_size();
        // TD-V4-CHUNK-PREFILL (P3 fix): V4 prefill chunks run
        // build_kv_metadata at batch_size == chunk rows (up to the
        // 512-descriptor bound; the executor's row bound is
        // min(max(max_batch, superchunk), 512)) — the scratch MUST cover
        // that or Pass 2's memset/writes run off the vectors AND the
        // device block-table buffer (silent VRAM stomp below 512 rows,
        // segfault at 512). Non-V4 keeps the max_batch sizing.
        const bool scratch_v4 = deps_.live_config->model.architecture
            == config::Architecture::deepseek_v4;
        const int B = scratch_v4
            ? std::max(deps_.max_batch_size,
                       std::min(static_cast<int>(ipc::kMaxBatchDescriptors),
                                std::max(deps_.max_batch_size,
                                         deps_.superchunk_tokens)))
            : deps_.max_batch_size;

        kv_meta_scratch_.resize(dcp_size);
        kv_cache_base_ptrs_.resize(dcp_size, nullptr);

        // TD-GOLDEN-KVMETA-PER-LAYER: block tables / slot mappings hold all
        // kv_layers_ layers (built once per token, layer-offset reads).
        const size_t L = static_cast<size_t>(kv_layers_ > 0 ? kv_layers_ : 1);

        for (int r = 0; r < dcp_size; ++r) {
            auto& m = kv_meta_scratch_[r];

            // Host staging
            m.host_seqlens_k.resize(B, 0);
            m.host_block_tables.resize(
                L * B * max_blocks_per_seq_, 0);
            m.host_slot_mappings.resize(L * B, 0);

            // Device scratch. FAIL-LOUD (1M-cap lesson): dev_block_tables is
            // L × max_batch × max_blocks_per_seq ints — 1.3 GiB/rank at
            // max_sequence_length=1M with B=64 — and a silently-null scratch
            // is consumed as "no block tables" downstream (the sparse
            // consumer skips linearize and attends STALE staging; the dense
            // kernel faults asynchronously) — deterministic wrong tokens
            // with zero errors. An allocation failure here must abort init.
            if (r < static_cast<int>(deps_.attention_devices.size())
                && deps_.attention_devices[r]) {
                auto* dev = deps_.attention_devices[r];
                m.dev_seqlens_k = dev->device_alloc(
                    static_cast<size_t>(B) * sizeof(int));
                m.dev_block_tables = dev->device_alloc(
                    L * B * max_blocks_per_seq_ * sizeof(int));
                m.dev_slot_mappings = dev->device_alloc(
                    L * B * sizeof(int));
                if (!m.dev_seqlens_k || !m.dev_block_tables
                    || !m.dev_slot_mappings) {
                    throw std::runtime_error(
                        "CommandDispatcher: KV metadata device scratch "
                        "allocation failed (rank " + std::to_string(r)
                        + ", block tables "
                        + std::to_string(L * B * max_blocks_per_seq_
                                         * sizeof(int))
                        + " B — scales with orchestrator.max_batch_size × "
                          "serving.max_sequence_length / page_size)");
                }
            }

            // KV cache base pointer for this rank
            if (deps_.page_allocator
                && r < static_cast<int>(deps_.hidden_state_pairs.size())) {
                const int gpu_pos =
                    deps_.hidden_state_pairs[r].gpu_position;
                kv_cache_base_ptrs_[r] =
                    deps_.page_allocator->kv_main_base(gpu_pos);
            }
        }

        // KVS-2: sequence-sharded KV mode (hardware.dcp_kv_mode). Effective
        // only at dcp_size >= 2. Allocates the sharded-only extras: per-rank
        // device GLOBAL seqlens scratch, the global host staging, and one
        // TRASH kMain page per rank (k_append destination for rows the rank
        // does not own — never referenced by any block table).
        kv_sharded_ = dcp_size >= 2
            && deps_.live_config->hardware.dcp_kv_mode
                   == config::DcpKvMode::sharded;
        kv_dcp_chunk_tokens_ =
            deps_.live_config->memory.kv_cache.dcp_chunk_size;
        if (kv_sharded_) {
            host_global_seqlens_.resize(B, 0);
            kv_trash_slot_base_.assign(dcp_size, -1);
            const int page_size = deps_.kv_page_size > 0
                ? deps_.kv_page_size : 64;
            for (int r = 0; r < dcp_size; ++r) {
                auto& m = kv_meta_scratch_[r];
                if (r < static_cast<int>(deps_.attention_devices.size())
                    && deps_.attention_devices[r]) {
                    m.dev_global_seqlens = deps_.attention_devices[r]
                        ->device_alloc(static_cast<size_t>(B) * sizeof(int));
                }
                if (deps_.page_allocator
                    && r < static_cast<int>(deps_.hidden_state_pairs.size())) {
                    const int gpu_pos =
                        deps_.hidden_state_pairs[r].gpu_position;
                    auto h = deps_.page_allocator->allocate(
                        gpu_pos, memory::Pool::kMain);
                    if (h) {
                        kv_trash_pages_.push_back(*h);
                        kv_trash_slot_base_[r] = h->page_idx * page_size;
                    } else {
                        spdlog::error("KVS-2: trash page allocation failed on "
                                      "rank {} — sharded KV dispatch will "
                                      "fail closed", r);
                    }
                }
            }
        }
    }

    // GLM-25k: DSA-guided KV tiering manager (opt-in, feature-OFF default).
    // Gates: config enabled + DSA model + row-self-contained cache format
    // (SnapMLA FP8 / TurboQuant MSE4) + live CUDA + all runtime deps
    // present.  Both KV placement modes are supported: replicated (each rank
    // tiers its own replica) and sharded (TD-KVT-DCP-SHARDED: each rank
    // tiers ITS token shard, INV-KVT-9).  Ineligible configs run the
    // non-tiered path byte-identically.
    // TD-V4-KVT (capability-completion P3): deepseek_v4 routes to the
    // V4 CSA-bucket tiering manager (page-granular demote + selection-
    // driven repromote; HCA/LID/SWA exempt by policy) instead of the
    // SnapMLA-row manager below.
    if (deps_.live_config && deps_.live_config->memory.kv_tiering.enabled
        && deps_.live_config->model.architecture
               == config::Architecture::deepseek_v4) {
        const auto& cfg = *deps_.live_config;
        const int dcp_size = deps_.dcp_executor
            ? deps_.dcp_executor->dcp_size() : 0;
        if (!deps_.cuda_kernels_enabled || !deps_.page_allocator
            || dcp_size < 1 || kv_cache_base_ptrs_.empty()
            || !kv_cache_base_ptrs_[0] || deps_.kv_page_size <= 0) {
            spdlog::warn("V4KvTiering: memory.kv_tiering.enabled but "
                         "prerequisites unmet — tiering DISABLED "
                         "(non-tiered path)");
        } else {
            V4KvTiering::Options vt;
            vt.dcp_size = dcp_size;
            vt.device_backends = deps_.device_backends;
            vt.kv_main_bases = kv_cache_base_ptrs_;
            if (deps_.stream_manager && deps_.dcp_executor) {
                for (const auto& g : deps_.dcp_executor->gpus())
                    vt.attn_streams.push_back(deps_.stream_manager->stream(
                        g.position, compute::StreamId::kAttention));
            }
            // Reuse the GLM seams verbatim (kMain-generic).
            vt.free_page = [this](uint64_t seq, int layer, int logical,
                                  const memory::PageHandle& h) {
                deps_.page_allocator->free(h);
                if (auto* st = find_seq(seq)) {
                    const int L = kv_layers_ > 0 ? kv_layers_ : 1;
                    const size_t idx =
                        static_cast<size_t>(logical) * L + layer;
                    if (idx < st->kv_pages.size()) {
                        st->kv_pages[idx].page_idx = -1;
                        st->kv_pages[idx].gpu_ptr = nullptr;
                    }
                }
                invalidate_kv_meta();
            };
            vt.alloc_page = [this](uint64_t seq, int layer, int logical)
                    -> std::optional<memory::PageHandle> {
                auto* st = find_seq(seq);
                const int L = kv_layers_ > 0 ? kv_layers_ : 1;
                const size_t idx = static_cast<size_t>(logical) * L + layer;
                if (!st || layer < 0 || logical < 0
                    || idx >= st->kv_pages.size())
                    return std::nullopt;
                auto* pa = deps_.page_allocator;
                const auto& dcp = pa->dcp_config();
                const int page_size = deps_.kv_page_size;
                const auto ts = static_cast<uint32_t>(logical)
                              * static_cast<uint32_t>(page_size);
                const bool replicated_kv = dcp.enabled() && !dcp.kv_sharded;
                std::optional<memory::PageHandle> h;
                if (replicated_kv) {
                    h = pa->allocate_replicated(seq, ts,
                                                static_cast<uint32_t>(layer),
                                                memory::Pool::kMain,
                                                /*unreserved=*/true);
                } else {
                    const int gpu = deps_.hidden_state_pairs.empty()
                        ? 0 : deps_.hidden_state_pairs[0].gpu_position;
                    h = pa->allocate_unreserved(gpu, memory::Pool::kMain);
                }
                if (!h) return std::nullopt;
                auto& m = pa->meta(*h);
                m.sequence_id = seq;
                m.token_start = ts;
                m.token_end = ts + static_cast<uint32_t>(page_size);
                m.layer_index = static_cast<uint32_t>(layer);
                st->kv_pages[idx] = *h;
                invalidate_kv_meta();
                return h;
            };
            // V4 kMain stride block == the CSA page byte size
            // (engine sets kv_cache_stride_block = v4l.csa_bytes_per_page).
            vt.page_bytes = deps_.kv_cache_stride_block;
            vt.csa_ratio = memory::kV4CsaRatio;
            vt.page_tokens = deps_.kv_page_size;   // 256-token logical block
            vt.retention_tokens = cfg.memory.kv_tiering.hot_buffer_slots;
            vt.index_topk = cfg.model.index_topk;
            vt.host_to_device_ratio =
                cfg.memory.kv_tiering.host_to_device_ratio;
            vt.csa_layers.assign(
                static_cast<size_t>(cfg.model.num_hidden_layers), 0);
            for (int l = 0; l < cfg.model.num_hidden_layers; ++l) {
                const int ratio =
                    l < static_cast<int>(cfg.model.compress_ratios.size())
                        ? cfg.model.compress_ratios[static_cast<size_t>(l)]
                        : 0;
                if (ratio == memory::kV4CsaRatio)
                    vt.csa_layers[static_cast<size_t>(l)] = 1;
            }
            v4_kv_tiering_ = std::make_unique<V4KvTiering>(std::move(vt));
        }
    } else if (deps_.live_config
               && deps_.live_config->memory.kv_tiering.enabled) {
        const auto& cfg = *deps_.live_config;
        const bool dsa = cfg.model.index_topk > 0;
        const int dcp_size = deps_.dcp_executor
            ? deps_.dcp_executor->dcp_size() : 0;
        const bool indexer_ok = dcp_size <= 1
            || cfg.hardware.dcp_indexer_mode
                   == config::DcpIndexerMode::replicated;
        // TD-KVT-TQ (resolved by audit): both cache formats are row-self-
        // contained, so byte-granular row moves are placement-exact
        // (INV-KVT-1).  SnapMLA FP8: FP8 c_kv | f32 scale | BF16 rope.
        // TurboQuant MSE4: packed 4-bit nope | fp16 norm | BF16 rope, with
        // the rotation Pi + codebook as GLOBAL constants (no per-page/π
        // metadata) — tq_fused_k_append writes and tq_dequant_ckv_indexed
        // reads whole rows at cache_stride_row.  TQ rows (386 B) are not
        // 16 B-aligned: kv_row_copy's byte fallback covers them
        // (KvRowCopyKernel.GatherUnalignedTqGeometryRows).
        const bool fmt_ok = deps_.page_allocator
            && (deps_.page_allocator->kv_cache_format()
                    == memory::KvCacheFormat::kSnapMlaFp8
                || deps_.page_allocator->kv_cache_format()
                    == memory::KvCacheFormat::kTurboQuantMse4);
        // TD-KVT-DCP-SHARDED: sharded KV is a supported tiering mode — each
        // rank demotes ITS token shard to ITS NUMA-local pool and
        // materializes the KVS-4 translated local selection (INV-KVT-9).
        // The chunk geometry must be sane (page-multiple; build_kv_metadata
        // enforces the same rule).
        const bool shard_ok = !kv_sharded_
            || (kv_dcp_chunk_tokens_ > 0 && deps_.kv_page_size > 0
                && kv_dcp_chunk_tokens_ % deps_.kv_page_size == 0);
        if (!dsa || !shard_ok || !indexer_ok || !fmt_ok
            || !deps_.cuda_kernels_enabled || !deps_.stream_manager
            || !deps_.page_allocator || dcp_size < 1
            || kv_cache_base_ptrs_.empty() || !kv_cache_base_ptrs_[0]) {
            spdlog::warn("KvTiering: memory.kv_tiering.enabled but "
                         "prerequisites unmet (dsa={}, shard_geometry_ok={}, "
                         "indexer_replicated={}, row_self_contained_fmt={}, "
                         "cuda={}) — tiering DISABLED (non-tiered path)",
                         dsa, shard_ok, indexer_ok, fmt_ok,
                         deps_.cuda_kernels_enabled);
        } else {
            KvTieringManager::Options topts;
            topts.dcp_size = dcp_size;
            topts.gpus = deps_.dcp_executor->gpus();
            topts.device_backends = deps_.device_backends;
            topts.stream_manager = deps_.stream_manager;
            topts.numa_manager = deps_.numa_manager;
            topts.free_page = [this](uint64_t seq, int layer, int logical,
                                     const memory::PageHandle& h) {
                deps_.page_allocator->free(h);
                // Neutralize the stored handle: bulk frees (seq_free,
                // rollbacks) skip page_idx < 0, and build_kv_metadata's
                // stale block-table entry for a cold page is never read
                // (tiered materialization classifies cold via the manager).
                if (auto* st = find_seq(seq)) {
                    const int L = kv_layers_ > 0 ? kv_layers_ : 1;
                    const size_t idx =
                        static_cast<size_t>(logical) * L + layer;
                    if (idx < st->kv_pages.size()) {
                        st->kv_pages[idx].page_idx = -1;
                        st->kv_pages[idx].gpu_ptr = nullptr;
                    }
                }
            };
            // TD-KVT-SPEC-FORK (re-promotion seam): allocate a fresh kMain
            // VRAM page for (seq, layer, logical) through the SAME growth
            // routing ensure_pages uses — allocate_replicated lockstep under
            // replicated KV (INV-KV-REP: the page_idx is valid against every
            // rank's kv_main base) / owner routing by token position under
            // sharded KV (INV-4.9e) — write it back into seq_pages_
            // (un-neutralize) and poison the kv-meta dirty guard so block
            // tables re-upload.  Unreserved: re-promotion is page growth
            // (INV-4.9f).  nullopt on exhaustion → the manager fails the
            // re-promotion CLOSED.
            topts.alloc_page = [this](uint64_t seq, int layer, int logical)
                    -> std::optional<memory::PageHandle> {
                auto* st = find_seq(seq);
                const int L = kv_layers_ > 0 ? kv_layers_ : 1;
                const size_t idx = static_cast<size_t>(logical) * L + layer;
                if (!st || layer < 0 || logical < 0
                    || idx >= st->kv_pages.size())
                    return std::nullopt;
                auto* pa = deps_.page_allocator;
                const auto& dcp = pa->dcp_config();
                const int page_size =
                    deps_.kv_page_size > 0 ? deps_.kv_page_size : 64;
                const auto ts = static_cast<uint32_t>(logical)
                              * static_cast<uint32_t>(page_size);
                const bool replicated_kv = dcp.enabled() && !dcp.kv_sharded;
                std::optional<memory::PageHandle> h;
                if (replicated_kv) {
                    h = pa->allocate_replicated(seq, ts,
                                                static_cast<uint32_t>(layer),
                                                memory::Pool::kMain,
                                                /*unreserved=*/true);
                } else {
                    const int gpu = dcp.enabled()
                        ? pa->dcp_gpu_for_token(ts)
                        : (deps_.hidden_state_pairs.empty()
                               ? 0
                               : deps_.hidden_state_pairs[0].gpu_position);
                    h = pa->allocate_unreserved(gpu, memory::Pool::kMain);
                }
                if (!h) return std::nullopt;
                auto& m = pa->meta(*h);
                m.sequence_id = seq;
                m.token_start = ts;
                m.token_end = ts + static_cast<uint32_t>(page_size);
                m.layer_index = static_cast<uint32_t>(layer);
                st->kv_pages[idx] = *h;
                invalidate_kv_meta();  // block tables re-upload
                return h;
            };
            topts.kv_main_bases = kv_cache_base_ptrs_;
            topts.stride_block = deps_.kv_cache_stride_block;
            topts.stride_row = deps_.kv_cache_stride_row;
            topts.page_size = deps_.kv_page_size > 0 ? deps_.kv_page_size : 64;
            topts.kv_layers = kv_layers_ > 0 ? kv_layers_ : 1;
            topts.index_topk = cfg.model.index_topk;
            topts.hot_buffer_slots = cfg.memory.kv_tiering.hot_buffer_slots;
            topts.host_to_device_ratio =
                cfg.memory.kv_tiering.host_to_device_ratio;
            // TD-KVT-REPLICA-COLD-DEDUP: single cold copy per page under
            // replicated KV at dcp >= 2 (INV-KVT-11).
            topts.replica_cold_dedup =
                cfg.memory.kv_tiering.replica_cold_dedup;
            // TD-KVT-DCP-SHARDED: per-rank shard tiering under sharded KV.
            topts.kv_sharded = kv_sharded_;
            topts.dcp_chunk_tokens = kv_dcp_chunk_tokens_;
            // TD-KVT-ADMISSION-UPFRONT: size the cohort selection staging
            // for the largest blessable chunk (the dispatcher's tier gate
            // caps chunk rows at max_batch_size — the executor's KVS-4
            // translation buffers share that bound).
            if (cfg.memory.kv_tiering.tiered_prefill
                && cfg.compute.dsa_sparse_prefill) {
                topts.cohort_rows_max = std::max(1, deps_.max_batch_size);
                // TD-KVT-COHORT-BATCHED-MATERIALIZE: union staging capacity
                // = the tight upper bound on a cohort union — no row selects
                // more than index_topk rows AND no selection can exceed the
                // rank-LOCAL prefix (sharded KV: ~max_seq / dcp, plus one
                // ownership chunk of slack for the round-robin remainder).
                {
                    const int64_t max_seq =
                        cfg.serving.max_sequence_length;
                    int64_t local_rows = max_seq;
                    if (kv_sharded_ && dcp_size >= 2) {
                        local_rows = max_seq / dcp_size
                            + std::max(1, kv_dcp_chunk_tokens_);
                    }
                    const int64_t by_topk =
                        static_cast<int64_t>(topts.cohort_rows_max)
                        * std::max(1, cfg.model.index_topk);
                    topts.union_rows_max = static_cast<int>(
                        std::min(by_topk, local_rows));
                }
            }
            // IndexShare full-layer mask (TD-KVT-SYNC/TD-KVT-PREFETCH):
            // mirrors ModelConfig::is_full_index_layer exactly (the same
            // rule engine.cpp feeds DcpExecutor::Options::indexer_full_
            // layers): full ⇔ l < index_skip_topk_offset OR
            // (l - offset + 1) % index_topk_freq == 0; freq <= 0 ⇒ all full
            // (no sharing).  MTP layers (>= num_hidden_layers) are SHARED by
            // construction (beyond the mask).
            {
                const int NH = cfg.model.num_hidden_layers;
                const int freq = cfg.model.index_topk_freq;
                const int off = cfg.model.index_skip_topk_offset;
                std::vector<uint8_t> mask(
                    static_cast<size_t>(std::max(NH, 0)), 1);
                if (freq > 0) {
                    for (int l = 0; l < NH; ++l)
                        mask[static_cast<size_t>(l)] =
                            (l < off || (l - off + 1) % freq == 0) ? 1 : 0;
                }
                topts.indexer_full_layers = std::move(mask);
            }
            kv_tiering_ = std::make_unique<KvTieringManager>(std::move(topts));
        }
    }

    // Configure page headroom reservation in PageAllocator.
    if (deps_.page_allocator && deps_.live_config) {
        const int max_reqs = deps_.live_config->serving.max_concurrent_requests;
        deps_.page_allocator->configure_headroom({
            .max_concurrent_forks     = max_reqs,
            .max_concurrent_sequences = max_reqs,
            .page_growth_chunk_pages  = chunk_size_pages_,
        });
    }

    // KD-R2: build gpu_pos → pair index lookup and attn_bufs_ projection.
    if (deps_.dcp_executor && deps_.stream_manager) {
        const size_t num_gpus = deps_.expert_devices.size();
        gpu_pos_to_pair_idx_.assign(num_gpus, -1);

        for (size_t i = 0; i < deps_.hidden_state_pairs.size(); ++i) {
            auto& pair = deps_.hidden_state_pairs[i];
            const int pos = pair.gpu_position;
            if (pos >= 0 && static_cast<size_t>(pos) < num_gpus) {
                gpu_pos_to_pair_idx_[pos] = static_cast<int>(i);
            }
            if (!pair.attn_moe_event)
                pair.attn_moe_event = deps_.stream_manager->create_event(pos);
            if (!pair.moe_attn_event)
                pair.moe_attn_event = deps_.stream_manager->create_event(pos);
        }

        attn_bufs_.resize(deps_.hidden_state_pairs.size());
        for (size_t i = 0; i < deps_.hidden_state_pairs.size(); ++i)
            attn_bufs_[i] = deps_.hidden_state_pairs[i].attn_buf;
    }

    // INV-5c: per-GPU fatal flag — CMP_GPU_FATAL emitted at most once per GPU
    // from the compute poll path. Transfer poll path has its own flag in DaemonLoop.
    gpu_fatal_emitted_.assign(deps_.device_backends.size(), false);
}

// TD-PREFILL-MOE-BIG scratch homing: bump-allocate `bytes` from this GPU's
// prefill-scratch tail (256-aligned — matches the VramAllocator region
// alignment contract); device_alloc fallback when the tail is absent (non-TP
// GPU, null-backend test, prefill_moe_big off) or exhausted. Never fails
// harder than device_alloc did.
void* CommandDispatcher::moe_scratch_alloc(size_t gpu_idx, size_t bytes) {
    if (gpu_idx < moe_tail_span_.size() && moe_tail_span_[gpu_idx].first
        && bytes > 0) {
        constexpr size_t kAlign = 256;
        const size_t off =
            (moe_tail_used_[gpu_idx] + kAlign - 1) & ~(kAlign - 1);
        if (off + bytes <= moe_tail_span_[gpu_idx].second) {
            moe_tail_used_[gpu_idx] = off + bytes;
            return moe_tail_span_[gpu_idx].first + off;
        }
        static bool warned = false;
        if (!warned) {
            warned = true;
            spdlog::info("TD-PREFILL-MOE-BIG: prefill-scratch tail exhausted "
                         "on gpu {} ({} of {} bytes used) — falling back to "
                         "device_alloc for the remainder",
                         gpu_idx, moe_tail_used_[gpu_idx],
                         moe_tail_span_[gpu_idx].second);
        }
    }
    auto* dev = (gpu_idx < deps_.expert_devices.size())
                    ? deps_.expert_devices[gpu_idx] : nullptr;
    return dev ? dev->device_alloc(bytes) : nullptr;
}

bool CommandDispatcher::moe_ptr_in_tail(size_t gpu_idx, const void* p) const {
    if (!p || gpu_idx >= moe_tail_span_.size()) return false;
    const char* base = moe_tail_span_[gpu_idx].first;
    return base && p >= base && p < base + moe_tail_span_[gpu_idx].second;
}

CommandDispatcher::~CommandDispatcher() {
    // C-6 Milestone C: never leave the async host-FFN worker joinable at teardown.
    join_cpu_fold_worker();

    // LS_REEF_RELOC_TRACE: the arena's location-change sink captures the
    // reef service's dump FILE* — clear it before reef_service_ dies (the
    // arena outlives this dispatcher).
    if (reef_reloc_trace_installed_ && deps_.pinned_arena)
        deps_.pinned_arena->location_change_sink = nullptr;

    // LS_LOADER_MACH_PROF: one-shot machinery cost table (enabled only).
    report_loader_mach_prof();

    // I8b x-ray: flush + close the structured model-record JSONL sink.
    close_loader_dump();

    // §12h Variant-A: release the fold-via-host pinned bounce buffers/events.
    for (size_t g = 0; g < fold_host_staging_.size(); ++g) {
        if (fold_host_staging_[g].first
            && g < deps_.device_backends.size() && deps_.device_backends[g])
            deps_.device_backends[g]->host_free_pinned(
                fold_host_staging_[g].first);
    }
    fold_host_staging_.clear();
    for (auto& [ev, owner] : fold_host_ev_) {
        if (ev && deps_.stream_manager)
            deps_.stream_manager->destroy_event(ev, owner);
    }
    fold_host_ev_.clear();

    // C-6 Milestone A: release the per-GPU CPU-expert fold staging buffers.
    for (size_t g = 0; g < cpu_fold_stage_.size(); ++g) {
        if (cpu_fold_stage_[g] && g < deps_.expert_devices.size()
            && deps_.expert_devices[g])
            deps_.expert_devices[g]->device_free(cpu_fold_stage_[g]);
    }
    cpu_fold_stage_.clear();
    cpu_fold_stage_bytes_.clear();

    // EPM-1: flush the pending routing record + free the NUMA-local D2H
    // staging (unregister before free). The dumper's dtor closes its file.
    epm_routing_dump_.reset();
    if (epm_route_host_) {
        if (epm_route_host_registered_)
            core::host_unregister_pinned(epm_route_host_);
        if (epm_route_host_from_numa_ && deps_.numa_manager) {
            memory::NumaBuffer buf{epm_route_host_, epm_route_host_bytes_,
                                   epm_route_host_node_};
            deps_.numa_manager->free(buf);
        } else if (epm_route_host_gpu_ >= 0 &&
                   epm_route_host_gpu_ <
                       static_cast<int>(deps_.device_backends.size()) &&
                   deps_.device_backends[static_cast<size_t>(
                       epm_route_host_gpu_)]) {
            deps_.device_backends[static_cast<size_t>(epm_route_host_gpu_)]
                ->host_free_pinned(epm_route_host_);
        }
        epm_route_host_ = nullptr;
    }

    // KVS-2: return the per-rank trash pages (sharded KV mode only).
    if (deps_.page_allocator) {
        for (auto& h : kv_trash_pages_) deps_.page_allocator->free(h);
    }
    kv_trash_pages_.clear();

    // TD-EVICT-BOARD-DESYNC: unregister the EvictScoreBoard as the ExpertCache's
    // residency listener before this dispatcher (which owns the board) is torn
    // down — the cache may outlive us, and must not fire into a freed board.
    if (deps_.expert_cache &&
        deps_.expert_cache->residency_listener() ==
            (evict_board_ ? &*evict_board_ : nullptr) &&
        evict_board_)
        deps_.expert_cache->set_residency_listener(nullptr);

    // 13c-2.0: FETCH eviction-map accounting summary (only if the path ran).
    if (far_evict_honored_ || far_evict_rejected_ || far_evict_fallback_) {
        spdlog::info(
            "TD-FAR-EVICT: provided-victim honored={} rejected={} local-fallback={}",
            far_evict_honored_, far_evict_rejected_, far_evict_fallback_);
    }

    // LS_FAR_GATED_FINAL / LS_FAR_STREAM_GATE engagement accounting (only if
    // the gate ever ran): proves per-layer engagement RATE, not just "engaged
    // once" (the one-time ACTIVE log).
    if (gated_final_engaged_ || gated_final_fb_unissued_
        || gated_final_fb_not_transferring_ || gated_final_fb_not_dispatched_
        || gated_final_fb_other_) {
        spdlog::info(
            "FAR gate engagement: engaged={} fb_unissued={} fb_not_transferring={} "
            "fb_not_dispatched={} fb_other={}",
            gated_final_engaged_, gated_final_fb_unissued_,
            gated_final_fb_not_transferring_, gated_final_fb_not_dispatched_,
            gated_final_fb_other_);
    }

    // Destroy pending compute events (including pipeline checkpoint events).
    if (deps_.stream_manager) {
        for (auto& pc : pending_compute_) {
            if (pc.cuda_event) {
                deps_.stream_manager->destroy_event(pc.cuda_event, pc.gpu_idx);
            }
            for (auto& ckpt : pc.pipeline_checkpoints) {
                if (ckpt.cuda_event && !ckpt.emitted) {
                    deps_.stream_manager->destroy_event(ckpt.cuda_event, pc.gpu_idx);
                }
            }
        }
    }
    pending_compute_.clear();

    // Destroy all CUDA events in the pool.
    if (deps_.stream_manager) {
        for (auto& [id, pooled] : event_pool_) {
            if (pooled.handle) {
                deps_.stream_manager->destroy_event(pooled.handle, pooled.gpu_idx);
            }
        }
    }
    event_pool_.clear();

    // KD-R2: destroy sync events owned by hidden state pairs.
    if (deps_.stream_manager) {
        for (auto& pair : deps_.hidden_state_pairs) {
            if (pair.attn_moe_event) deps_.stream_manager->destroy_event(pair.attn_moe_event, pair.gpu_position);
            if (pair.moe_attn_event) deps_.stream_manager->destroy_event(pair.moe_attn_event, pair.gpu_position);
            pair.attn_moe_event = nullptr;
            pair.moe_attn_event = nullptr;
        }
    }

    token_to_cmd_seq_.clear();

    // KD-2: free sampling scratch buffers.
    for (size_t i = 0; i < sampling_scratch_.size(); ++i) {
        if (sampling_scratch_[i] && i < deps_.expert_devices.size()
            && deps_.expert_devices[i]) {
            deps_.expert_devices[i]->device_free(sampling_scratch_[i]);
        }
    }
    sampling_scratch_.clear();

    // Ticket J: free the V4 dflash stream-mean capture staging.
    for (size_t i = 0; i < dspark_mean_scratch_.size(); ++i) {
        if (dspark_mean_scratch_[i] && i < deps_.device_backends.size()
            && deps_.device_backends[i]) {
            deps_.device_backends[i]->set_device();
            deps_.device_backends[i]->device_free(dspark_mean_scratch_[i]);
        }
    }
    dspark_mean_scratch_.clear();

    // KD-2a: free confidence scratch buffers.
    for (size_t i = 0; i < confidence_top1_scratch_.size(); ++i) {
        if (i < deps_.expert_devices.size() && deps_.expert_devices[i]) {
            if (confidence_top1_scratch_[i])
                deps_.expert_devices[i]->device_free(confidence_top1_scratch_[i]);
            if (confidence_entropy_scratch_[i])
                deps_.expert_devices[i]->device_free(confidence_entropy_scratch_[i]);
        }
    }
    confidence_top1_scratch_.clear();
    confidence_entropy_scratch_.clear();
    confidence_host_staging_.clear();

    // KD-3b: free MoE scratch buffers. TD-PREFILL-MOE-BIG: pointers carved
    // from the prefill-scratch tail (moe_scratch_alloc) are NOT device_frees —
    // the tail is a slice of the VramAllocator region and is freed with it.
    for (size_t i = 0; i < moe_scratch_.size(); ++i) {
        if (i >= deps_.expert_devices.size() || !deps_.expert_devices[i]) continue;
        auto* dev = deps_.expert_devices[i];
        auto& s = moe_scratch_[i];
        auto free_scratch = [&](void* p) {
            if (!moe_ptr_in_tail(i, p)) dev->device_free(p);
        };
        dev->device_free(s.router_logits);
        dev->device_free(s.topk_weights);
        dev->device_free(s.topk_indices);
        dev->device_free(s.moe_token_ids);
        free_scratch(s.permuted_input);
        dev->device_free(s.expert_offsets);
        free_scratch(s.src_to_dest_map);
        free_scratch(s.permuted_idx);
        free_scratch(s.gate_up_output);
        free_scratch(s.activation_output);
        free_scratch(s.expert_output);
        dev->device_free(s.moe_output);
        free_scratch(s.moe_wave_accum);
        free_scratch(s.big_unperm_tmp);  // TD-PREFILL-MOE-BIG
        free_scratch(s.wave_masked_topk);       // TD-MOE-BIG-GEMM-SWEEP
        dev->device_free(s.wave_expert_mask);   // TD-MOE-BIG-GEMM-SWEEP
        // DET-REDUCE Phase 1b: per-slot EP-combine buffers (nullptr when off;
        // device_free tolerates nullptr).
        free_scratch(s.moe_output_fp32);
        free_scratch(s.moe_output_bf16_perslot);
        dev->device_free(s.normalized_hidden);
        dev->device_free(s.hc_x);
        dev->device_free(s.hc_post);
        dev->device_free(s.hc_comb);
        free_scratch(s.permute_workspace);
        free_scratch(s.gemm_workspace);
        dev->device_free(s.problem_sizes);
        dev->device_free(s.sf_offsets);
        free_scratch(s.shared_gate_up_output);
        free_scratch(s.shared_activation);
        dev->device_free(s.shared_expert_output);
        dev->device_free(s.shared_expert_offsets);
        dev->device_free(s.shared_problem_sizes);
        dev->device_free(s.shared_sf_offsets);
        dev->device_free(s.nvfp4_alpha);
        dev->device_free(s.moe_input_scales);
        free_scratch(s.quant_act);
        free_scratch(s.quant_scale);
        free_scratch(s.gguf_gate_up_split);  // GG-5c (was leaked pre-BIG)
        dev->device_free(s.gguf_single_b_ptr);  // GG-5b
        dev->device_free(s.zero_weight_buf);
        dev->device_free(s.ep_xtp_staging);  // INV-MOE-EP-XTP
        // TD-DECODE-FFN-GRAPH: free per-projection b_ptrs arrays + pinned host.
        for (int p = 0; p < 3; ++p) {
            dev->device_free(s.g_b_ptrs[p]);
            dev->device_free(s.g_sb_ptrs[p]);
            dev->device_free(s.g_b_ptrs_w[p]);
            dev->device_free(s.g_sb_ptrs_w[p]);
            if (s.g_b_ptrs_host[p])
                deps_.device_backends[i]->host_free_pinned(
                    const_cast<void*>(reinterpret_cast<const void*>(s.g_b_ptrs_host[p])));
            if (s.g_sb_ptrs_host[p])
                deps_.device_backends[i]->host_free_pinned(
                    const_cast<void*>(reinterpret_cast<const void*>(s.g_sb_ptrs_host[p])));
            if (s.g_b_ptrs_host_w[p])
                deps_.device_backends[i]->host_free_pinned(
                    const_cast<void*>(reinterpret_cast<const void*>(s.g_b_ptrs_host_w[p])));
            if (s.g_sb_ptrs_host_w[p])
                deps_.device_backends[i]->host_free_pinned(
                    const_cast<void*>(reinterpret_cast<const void*>(s.g_sb_ptrs_host_w[p])));
        }
    }
    moe_scratch_.clear();

    // TD-DECODE-FFN-GRAPH: destroy captured routed-FFN graphs (must precede
    // device teardown so the graph exec handles are released first).
    for (auto& m : routed_ffn_graphs_)
        for (auto& [key, r] : m)
            if (r) r->destroy();
    routed_ffn_graphs_.clear();

    // KD-3c: free speculation scratch buffers.
    for (size_t i = 0; i < spec_scratch_.size(); ++i) {
        if (i >= deps_.expert_devices.size() || !deps_.expert_devices[i]) continue;
        auto* dev = deps_.expert_devices[i];
        auto& ss = spec_scratch_[i];
        dev->device_free(ss.hidden_a);
        dev->device_free(ss.hidden_b);
        dev->device_free(ss.logits);
        dev->device_free(ss.cos_sim_out);
        dev->device_free(ss.readback);
        dev->device_free(ss.mtp_concat);
    }
    spec_scratch_.clear();

    // KD-4b: free embedding token scratch buffers.
    for (size_t i = 0; i < embedding_token_scratch_.size(); ++i) {
        if (embedding_token_scratch_[i] && i < deps_.attention_devices.size()
            && deps_.attention_devices[i]) {
            deps_.attention_devices[i]->device_free(embedding_token_scratch_[i]);
        }
    }
    embedding_token_scratch_.clear();

    // KD-4b: free output norm scratch buffers.
    for (size_t i = 0; i < output_norm_scratch_.size(); ++i) {
        if (output_norm_scratch_[i] && i < deps_.attention_devices.size()
            && deps_.attention_devices[i]) {
            deps_.attention_devices[i]->device_free(output_norm_scratch_[i]);
        }
    }
    output_norm_scratch_.clear();

    // KD-4f: free logits scratch buffers.
    for (size_t i = 0; i < logits_scratch_.size(); ++i) {
        if (logits_scratch_[i] && i < deps_.attention_devices.size()
            && deps_.attention_devices[i]) {
            deps_.attention_devices[i]->device_free(logits_scratch_[i]);
        }
    }
    logits_scratch_.clear();

    // TD-SERVE-NAMED-TOOL-CHOICE: free the pinned logits readback row.
    if (logits_readback_host_) {
        if (!deps_.device_backends.empty() && deps_.device_backends[0])
            deps_.device_backends[0]->host_free_pinned(logits_readback_host_);
        logits_readback_host_ = nullptr;
        logits_readback_bytes_ = 0;
    }

    // KD-4g: free partial logits scratch buffers.
    for (size_t i = 0; i < partial_logits_scratch_.size(); ++i) {
        if (partial_logits_scratch_[i] && i < deps_.attention_devices.size()
            && deps_.attention_devices[i]) {
            deps_.attention_devices[i]->device_free(partial_logits_scratch_[i]);
        }
    }
    partial_logits_scratch_.clear();

    // TD-72a: free allgather transpose scratch.
    if (logits_gather_scratch_) {
        const auto gp = logits_gather_scratch_gpu_;
        if (gp < deps_.attention_devices.size() && deps_.attention_devices[gp])
            deps_.attention_devices[gp]->device_free(logits_gather_scratch_);
        logits_gather_scratch_ = nullptr;
    }
}

// ── KD-2: device lookup helpers ────────────────────────────────────────────

compute::AttentionDevice* CommandDispatcher::attn_dev(uint32_t gpu_idx) const {
    if (gpu_idx >= deps_.attention_devices.size()) return nullptr;
    return deps_.attention_devices[gpu_idx];
}

compute::ExpertDevice* CommandDispatcher::expert_dev(uint32_t gpu_idx) const {
    if (gpu_idx >= deps_.expert_devices.size()) return nullptr;
    return deps_.expert_devices[gpu_idx];
}

// ── Main dispatch ──────────────────────────────────────────────────────────

void CommandDispatcher::dispatch(const ipc::Command& cmd) {
    // LS_CUDA_TRIPWIRE (diagnostic, default off): peek the per-device sticky
    // CUDA error BEFORE dispatching each command — localizes which earlier
    // (unchecked-async) call left an error that would otherwise surface at an
    // arbitrary later cudaGetLastError (e.g. a kernel-launch check).
    static const bool cuda_tripwire = [] {
        const char* v = std::getenv("LS_CUDA_TRIPWIRE");
        return v && v[0] && v[0] != '0';
    }();
    if (cuda_tripwire && deps_.cuda_kernels_enabled) {
        for (size_t g = 0; g < deps_.device_backends.size(); ++g) {
            if (!deps_.device_backends[g]) continue;
            const int e = deps_.device_backends[g]->peek_last_error();
            if (e != 0)
                spdlog::warn("LS_CUDA_TRIPWIRE: sticky CUDA error {} on gpu "
                             "pos {} BEFORE dispatching cmd 0x{:04x} seq {}",
                             e, g, cmd.cmd_type, cmd.cmd_seq);
        }
    }
    try {
        switch (static_cast<ipc::CmdType>(cmd.cmd_type)) {
            // Transfer
            case ipc::CMD_TRANSFER_H2D:       handle_transfer_h2d(cmd); break;
            case ipc::CMD_TRANSFER_D2H:        handle_transfer_d2h(cmd); break;

            // Expert cache
            case ipc::CMD_CACHE_RESERVE:       handle_cache_reserve(cmd); break;
            case ipc::CMD_CACHE_EVICT:         handle_cache_evict(cmd); break;
            case ipc::CMD_CACHE_PROMOTE:       handle_cache_promote(cmd); break;
            case ipc::CMD_CACHE_DEMOTE:        handle_cache_demote(cmd); break;

            // Graph replay
            case ipc::CMD_GRAPH_REPLAY:        handle_graph_replay(cmd); break;

            // Stream synchronization
            case ipc::CMD_RECORD_EVENT:        handle_record_event(cmd); break;
            case ipc::CMD_STREAM_WAIT_EVENT:   handle_stream_wait_event(cmd); break;

            // Placement
            case ipc::CMD_COMPUTE_AFFINITY_HINTS: handle_compute_affinity_hints(cmd); break;
            case ipc::CMD_NUMA_MIGRATE:            handle_numa_migrate(cmd); break;

            // Sequence lifecycle
            case ipc::CMD_SEQ_CREATE:         handle_seq_create(cmd); break;
            case ipc::CMD_SEQ_FREE:           handle_seq_free(cmd); break;
            case ipc::CMD_SEQ_FORK:           handle_seq_fork(cmd); break;
            case ipc::CMD_SEQ_SNAPSHOT:       handle_seq_snapshot(cmd); break;
            case ipc::CMD_SEQ_RESTORE:        handle_seq_restore(cmd); break;

            // NVMe tier + transfer cancellation
            case ipc::CMD_NVME_READ:          handle_nvme_read(cmd); break;
            case ipc::CMD_NVME_WRITE:         handle_nvme_write(cmd); break;
            case ipc::CMD_NVME_EVICT_HOST:    handle_nvme_evict_host(cmd); break;
            case ipc::CMD_CANCEL_TRANSFER:    handle_cancel_transfer(cmd); break;

            // Fused compute commands (IPC-8d)
            case ipc::D_B_CMD_RUN_ATTENTION:
            case ipc::D_B_CMD_RUN_MOE:
                handle_fused_compute_command(cmd);
                break;

            // Fused batch/sequential commands (IPC-8e)
            case ipc::D_B_CMD_PREFETCH_BATCH:  handle_prefetch_batch(cmd); break;
            case ipc::D_B_CMD_EVICT_BATCH:     handle_evict_batch(cmd); break;
            case ipc::B_CMD_NVME_BATCH_READ:   handle_nvme_batch_read(cmd); break;
            case ipc::D_CMD_PREFETCH_EXPERT:   handle_prefetch_expert(cmd); break;
            case ipc::D_CMD_EVICT_TO_HOST:     handle_evict_to_host(cmd); break;
            case ipc::D_CMD_SLOW_EVICT_TO_HOST: handle_slow_evict_to_host(cmd); break;
            case ipc::D_CMD_STAGE_EXPERT:      handle_stage_expert(cmd); break;
            case ipc::D_CMD_RUN_PREFETCH_PROBE:   handle_run_prefetch_probe(cmd); break;
            case ipc::D_CMD_RUN_ADAPTER_FORWARD:  handle_run_adapter_forward(cmd); break;
            case ipc::D_CMD_RUN_MTP_STEP:         handle_run_mtp_step(cmd); break;
            case ipc::D_CMD_MTP_PROJECT:          handle_mtp_project(cmd); break;
            case ipc::D_CMD_RUN_DSPARK_STEP:      handle_run_dspark_step(cmd); break;
            case ipc::D_CMD_RUN_SELF_SPEC_FORWARD: handle_run_self_spec_forward(cmd); break;
            case ipc::E_CMD_SEQ_CREATE:        handle_e_seq_create(cmd); break;
            case ipc::E_CMD_SEQ_FREE:          handle_e_seq_free(cmd); break;
            case ipc::E_CMD_FETCH_AND_RUN_MOE: handle_fetch_and_run_moe(cmd); break;
            case ipc::E_CMD_FETCH_AND_RUN_MOE_BIG:
                handle_fetch_and_run_moe_big(cmd); break;
            case ipc::E_FORWARD_ONE_LAYER:     handle_forward_one_layer(cmd); break;
            case ipc::E_CMD_REEF_ROUTE:        handle_reef_route(cmd); break;
            case ipc::E_CMD_FAR_FORWARD_LAYER: handle_far_forward_layer(cmd); break;
            case ipc::CMD_CONFIG_UPDATE:       handle_config_update(cmd); break;

            // Compute kernels — require BufferRegistry (IPC-6)
            case ipc::CMD_ATTENTION_DECODE:
            case ipc::CMD_ATTENTION_PREFILL:
            case ipc::CMD_GATING:
            case ipc::CMD_EXPERT_FFN:
            case ipc::CMD_EMBEDDING_LOOKUP:
            case ipc::CMD_OUTPUT_HEAD:
            case ipc::CMD_RMSNORM:
            case ipc::CMD_SWIGLU:
            case ipc::CMD_MOE_PERMUTE:
            case ipc::CMD_MOE_UNPERMUTE:
            case ipc::CMD_DCP_CORRECTION:
            case ipc::CMD_NCCL_ALLREDUCE:
            case ipc::CMD_DYNAMIC_FP8_QUANT:
            case ipc::CMD_PRESCOPE_GATING:
            case ipc::CMD_PROBE_MLP:
            case ipc::CMD_SAMPLE_TOKENS:
                handle_compute_command(cmd);
                break;

            default:
                write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kUnknownCommand, "unknown command type");
                break;
        }
    } catch (const std::exception& e) {
        spdlog::error("CommandDispatcher: exception dispatching 0x{:04X}: {}",
                      cmd.cmd_type, e.what());
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kException, e.what());
    }
}

// ── Transfer token tracking ────────────────────────────────────────────────

uint32_t CommandDispatcher::resolve_cmd_seq(uint64_t transfer_token) const {
    auto it = token_to_cmd_seq_.find(transfer_token);
    return it != token_to_cmd_seq_.end() ? it->second : 0u;
}

void CommandDispatcher::remove_token_mapping(uint64_t transfer_token) {
    auto it = token_to_cmd_seq_.find(transfer_token);
    if (it != token_to_cmd_seq_.end()) {
        cmd_seq_to_token_.erase(it->second);
        token_to_cmd_seq_.erase(it);
    }
}

uint32_t CommandDispatcher::resolve_nvme_cmd_seq(uint64_t nvme_token) const {
    auto it = nvme_token_to_cmd_seq_.find(nvme_token);
    return it != nvme_token_to_cmd_seq_.end() ? it->second : 0u;
}

void CommandDispatcher::remove_nvme_token_mapping(uint64_t nvme_token) {
    auto it = nvme_token_to_cmd_seq_.find(nvme_token);
    if (it != nvme_token_to_cmd_seq_.end()) {
        cmd_seq_to_token_.erase(it->second);
        nvme_token_to_cmd_seq_.erase(it);
    }
}

}  // namespace layerstorm::daemon

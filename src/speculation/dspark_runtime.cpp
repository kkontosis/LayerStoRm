// DSpark DFlash backbone runtime (DSP-3).  See dspark_runtime.h.
//
// CUDA-free TU (INV-GPU-1): all device work routes through DeviceBackend and
// the kernel launcher headers.

#include "speculation/dspark_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "compute/kernels/dspark/dspark_backbone.h"
#include "compute/kernels/elementwise/residual_add.h"
#include "compute/kernels/dspark/dspark_confidence.h"
#include "compute/kernels/dspark/dspark_markov.h"
#include "compute/kernels/dspark/dspark_v4.h"      // ticket J: V4 dflash draft
#include "compute/kernels/attention/v4_prep.h"     // q_prep / sinks / inv rope
#include "compute/kernels/embedding/embedding.h"
#include "compute/kernels/mhc/mhc.h"               // hc_pre/post/head + expand
#include "compute/kernels/moe/router_projection.h"
#include "compute/kernels/norm/rmsnorm.h"
#include "compute/kernels/sm120/gemm/bf16_gemm.h"
#include "compute/kernels/sm120/gemm/wq_gemm.h"
#include "compute/rope_table.h"                    // draft base rope table
#include "core/expert_device.h"                    // V4 MoE on the draft GPU
#include "sm120/gating/topk_gating.h"              // deps ExpertKernels gating
#include "smxx/permute/moe_permute.h"              // permute workspace query
#include "config/config_parser.h"
#include "core/cuda_hardware_query.h"  // EPM-1: host_register_pinned helpers
#include "core/device_backend.h"
#include "core/memory/numa_manager.h"  // EPM-1: NUMA-local D2H staging
#include "speculation/epm_dump.h"      // EPM-1: block record writer

namespace layerstorm::speculation {

namespace {

constexpr int64_t kBf16 = 2;

/// Every device-scratch allocation create() makes, in a fixed order, so the
/// budget helper (dspark_runtime_scratch_bytes) can never drift from the
/// real allocations.
struct ScratchLayout {
    struct Item {
        const char* name;
        int64_t bytes;
    };
    std::vector<Item> items;
    int64_t total = 0;

    void add(const char* name, int64_t bytes) {
        items.push_back({name, bytes});
        total += bytes;
    }
};

/// Ticket J: the V4 dflash draft's scratch (single rank). Fixed order —
/// create() unpacks positionally; the LayerRegistry budget shares this
/// function (dspark_runtime_scratch_bytes) so they cannot drift.
ScratchLayout v4_scratch_layout(const config::DsparkConfig& dc,
                                const model::DsparkCheckpointConfig& ck) {
    const int64_t H = ck.hidden_size;
    const int64_t n_aux =
        static_cast<int64_t>(ck.aux_hidden_state_layer_ids.size());
    const int64_t bs = ck.block_size;
    const int64_t rows = dc.aux_capture_max_rows;
    const int64_t ctx_cap = dc.draft_context_capacity_tokens;
    const int64_t Vd = ck.draft_vocab_size;
    const int64_t r = ck.markov_rank;
    const int64_t L = ck.num_hidden_layers;
    const int64_t D = ck.head_dim;
    const int64_t R = ck.v4.rope_dim;
    const int64_t HQ = ck.num_attention_heads;
    const int64_t E = ck.v4.n_routed_experts;
    const int64_t topk = ck.v4.n_expert_used;
    const int64_t I = ck.v4.moe_intermediate;
    const int64_t hc = ck.v4.hc_mult;
    const int64_t T = bs * topk;  // permuted MoE row bound

    ScratchLayout s;
    // Base-machinery items (shared members; V4 sizes).
    s.add("aux_stage", rows * n_aux * H * kBf16);
    s.add("ctx_hidden", std::max(rows, ctx_cap) * H * kBf16);
    s.add("ctx_normed", rows * H * kBf16);
    s.add("ctx_ktmp", rows * D * kBf16);
    // kv arena: SINGLE-copy roped latents (K == V), row == absolute pos.
    s.add("kv_arena", L * ctx_cap * D * kBf16);
    s.add("q_ids", 64);
    s.add("q_x", bs * hc * H * kBf16);  // hc-wide residual stream
    s.add("q_normed", bs * H * kBf16);
    s.add("hidden_out", bs * H * kBf16);
    s.add("base_logits", bs * Vd * 4);
    s.add("draft_ids", 64);
    s.add("markov_e", bs * r * kBf16);
    s.add("markov_partials",
          static_cast<int64_t>(compute::dspark_markov_num_blocks(Vd)) *
              static_cast<int64_t>(sizeof(compute::DsparkMarkovPartial)));
    s.add("corrected_logits", bs * Vd * 4);
    s.add("conf_out", 64);
    // V4-specific items.
    s.add("v4_rope_table", (ctx_cap + bs) * R * 4);
    s.add("v4_pos", 64);
    s.add("v4_x1", bs * H * kBf16);
    s.add("v4_post", bs * hc * 4);
    s.add("v4_comb", bs * hc * hc * 4);
    s.add("v4_qlat", bs * ck.v4.q_lora_rank * kBf16);
    s.add("v4_qheads", bs * HQ * D * kBf16);
    s.add("v4_qn", bs * HQ * D * kBf16);
    s.add("v4_qr", bs * HQ * R * kBf16);
    s.add("v4_kvlat", bs * D * kBf16);
    s.add("v4_blkkv", bs * D * kBf16);
    s.add("v4_attn", bs * HQ * D * kBf16);
    s.add("v4_lse", bs * HQ * 4);
    s.add("v4_oa", bs * ck.v4.o_groups * ck.v4.o_lora_rank * kBf16);
    s.add("v4_module_out", bs * H * kBf16);
    s.add("v4_logits_e", bs * E * 4);
    s.add("v4_topk_w", bs * topk * 4);
    s.add("v4_topk_idx", bs * topk * 4);
    s.add("v4_perm_in", T * H * kBf16);
    s.add("v4_exp_off", (E + 1) * 4);
    s.add("v4_s2d", T * 4);
    s.add("v4_perm_idx", T * 4);
    s.add("v4_perm_ws",
          static_cast<int64_t>(compute::query_moe_permute_workspace_size(
              static_cast<int>(bs), static_cast<int>(topk),
              static_cast<int>(E))));
    s.add("v4_gate_out", T * I * kBf16);
    s.add("v4_up_out", T * I * kBf16);
    s.add("v4_gu", T * 2 * I * kBf16);
    s.add("v4_act", T * I * kBf16);
    s.add("v4_expert_out", T * H * kBf16);
    s.add("v4_sh_g", bs * I * kBf16);
    s.add("v4_sh_u", bs * I * kBf16);
    s.add("v4_sh_gu", bs * 2 * I * kBf16);
    s.add("v4_sh_out", bs * H * kBf16);
    // GGUF grouped int-GEMM workspace — mirrors the kernel's
    // gguf_grouped_gemm_workspace_bytes CUDA-free (same mirror rationale as
    // command_dispatcher.cpp:601, TD-GGUF-Q8_1-WS-CONST-MIRROR):
    //   q8 = T · (K_max/32) · 36 ; ws = align16(q8) + 2·(⌈T/64⌉+E)·4.
    {
        const int64_t k_max = std::max(H, I);
        const int64_t q8 = T * (k_max / 32) * 36;
        const int64_t map_off = (q8 + 15) & ~int64_t{15};
        const int64_t tmax = (T + 63) / 64 + E;
        s.add("v4_gemm_ws", map_off + 2 * tmax * 4);
    }
    // NO EPM item: the feature dump is unsupported for the V4 dflash draft
    // (the residual is hc-wide; layout differs) — fail-soft disabled.
    return s;
}

ScratchLayout scratch_layout(const config::DsparkConfig& dc,
                             const model::DsparkCheckpointConfig& ck,
                             int rank = 0, int num_ranks = 1) {
    if (ck.is_v4_dflash) return v4_scratch_layout(dc, ck);
    const int64_t H = ck.hidden_size;
    const int64_t n_aux =
        static_cast<int64_t>(ck.aux_hidden_state_layer_ids.size());
    const int64_t nr = std::max(1, num_ranks);
    // Per-rank local dims (TD-DSPARK-DRAFT-SHARD; == the full dims at nr==1;
    // divisibility is enforced by create()/dspark_shard_shape).
    const int64_t q_dim =
        static_cast<int64_t>(ck.num_attention_heads) * ck.head_dim / nr;
    const int64_t kv_dim =
        static_cast<int64_t>(ck.num_key_value_heads) * ck.head_dim / nr;
    const int64_t I = ck.intermediate_size / nr;
    const int64_t L = ck.num_hidden_layers;
    // Logits/bias buffers are DRAFT-vocab sized (== vocab_size for the
    // shipped full-vocab checkpoint; smaller under a reduced-vocab d2t
    // checkpoint — TD-DSPARK-VOCAB-REMAP).
    const int64_t Vd = ck.draft_vocab_size;
    const int64_t Vd_local = Vd / nr;
    const int64_t bs = ck.block_size;
    const int64_t rows = dc.aux_capture_max_rows;
    const int64_t ctx_cap = dc.draft_context_capacity_tokens;

    ScratchLayout s;
    if (rank == 0) {
        s.add("aux_stage", rows * n_aux * H * kBf16);
        // ctx_hidden doubles as the CHUNKED-capture fc accumulator
        // (TD-DSPARK-PREFILL-CAP): target steps larger than the staging fold
        // in per-slot, staging-sized pieces, so it must hold every surviving
        // row of one step — the arena-overflow gate bounds those at ctx_cap.
        s.add("ctx_hidden", std::max(rows, ctx_cap) * H * kBf16);
    }
    s.add("ctx_normed", rows * H * kBf16);
    s.add("ctx_ktmp", rows * kv_dim * kBf16);
    s.add("kv_arena", L * 2 * ctx_cap * kv_dim * kBf16);
    if (rank == 0) s.add("q_ids", 64);  // [block_size <= 16] i32, rounded
    s.add("q_x", bs * H * kBf16);
    s.add("q_normed", bs * H * kBf16);
    s.add("q_tmp", bs * q_dim * kBf16);
    s.add("q_q", bs * q_dim * kBf16);
    s.add("q_k", bs * q_dim * kBf16);
    s.add("q_v", bs * q_dim * kBf16);
    s.add("q_attn", bs * q_dim * kBf16);
    s.add("q_oproj", bs * H * kBf16);
    s.add("q_gate", bs * I * kBf16);
    s.add("q_up", bs * I * kBf16);
    s.add("q_act", bs * I * kBf16);
    s.add("q_mlp", bs * H * kBf16);
    s.add("hidden_out", bs * H * kBf16);
    if (rank == 0) {
        s.add("base_logits", bs * Vd * 4);  // FP32
        // DSP-4 Markov head (INV-DSPARK-MARKOV).  markov_e is the PER-STEP
        // e-chain stash [block_size, r]: slot k = markov_w1[x_{k-1}] — the
        // DSP-4 sequential loop reads/writes it slot-wise and DSP-6's
        // confidence head consumes the whole stash in one kernel.
        const int64_t r = ck.markov_rank;
        s.add("draft_ids", 64);  // [block_size <= 16] i32, rounded
        s.add("markov_e", bs * r * kBf16);
        s.add("markov_partials",
              static_cast<int64_t>(compute::dspark_markov_num_blocks(Vd)) *
                  static_cast<int64_t>(sizeof(compute::DsparkMarkovPartial)));
        s.add("corrected_logits", bs * Vd * 4);  // FP32
        // DSP-6 confidence head (INV-DSPARK-CONF).
        s.add("conf_out", 64);  // [block_size <= 16] FP32, rounded
    }
    // TD-DSPARK-DRAFT-SHARD (nr > 1, every rank): the per-rank lm_head
    // logits shard + the per-(layer, site) peer-partial exchange staging.
    if (nr > 1) {
        s.add("logits_shard", bs * Vd_local * 4);  // FP32
        s.add("peer_stage", L * 2 * bs * H * kBf16);
    }
    // EPM-1 (Phase 29): per-backbone-layer hidden dump staging [bs, L, H]
    // BF16 — allocated ONLY when the dump is armed (epm_dump_dir /
    // LS_EPM_DUMP). Conditional here keeps the LayerRegistry budget helper
    // (dspark_runtime_scratch_bytes) exactly in sync with the real carve.
    // Kept LAST (create() unpacks in layout order).
    if (rank == 0 && !epm_dump_dir(dc).empty())
        s.add("epm_hidden_stage", bs * L * H * kBf16);
    return s;
}

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("dspark_runtime: " + msg);
}

}  // namespace

int64_t dspark_runtime_scratch_bytes(const config::Config& cfg,
                                     const model::DsparkCheckpointConfig& ck,
                                     int rank, int num_ranks) {
    return scratch_layout(cfg.speculation.dspark, ck, rank, num_ranks).total;
}

// ── create ──────────────────────────────────────────────────────────────────

std::unique_ptr<DsparkRuntime> DsparkRuntime::create(
        const config::Config& cfg, std::vector<Rank> ranks,
        parallelism::DcpCommunicator* communicator,
        std::vector<void*> arenas, std::vector<int64_t> arena_bytes,
        memory::NumaManager* numa) {
    const auto& dc = cfg.speculation.dspark;
    if (ranks.empty()) fail("empty draft device set");
    if (ranks.size() > 2)
        fail("draft sharding supports exactly 1 or 2 ranks "
             "(TD-DSPARK-DRAFT-SHARD: the row-parallel combine is a 2-rank "
             "D2D cross-exchange) — configure at most TWO "
             "speculation.dspark.draft_gpus entries");
    for (const auto& r : ranks)
        if (!r.backend || !r.stream) fail("null rank backend/stream");
    if (!arenas.empty() &&
        (arenas.size() != ranks.size() ||
         arena_bytes.size() != ranks.size()))
        fail("arenas/arena_bytes must have one entry per draft rank");

    // Load + cross-validate + upload the draft checkpoint (DSP-2 machinery).
    // Ticket J: a .gguf checkpoint_path selects the V4 dflash arm — its
    // aliased embed/lm_head come from the TARGET GGUF (model.weights_path).
    const bool ckpt_is_gguf =
        model::dspark_checkpoint_is_gguf(dc.checkpoint_path);
    auto host = ckpt_is_gguf
        ? model::load_dspark_v4_gguf_draft(dc.checkpoint_path,
                                           cfg.model.weights_path)
        : model::load_dspark_draft(dc.checkpoint_path);
    model::validate_dspark_config_against_checkpoint(dc, host.ckpt);
    const auto& ck = host.ckpt;
    if (ck.is_v4_dflash && ranks.size() != 1)
        fail("the V4 dflash draft is single-rank only — configure exactly "
             "one speculation.dspark.draft_gpus entry");
    if (!ck.is_v4_dflash &&
        ck.num_attention_heads != ck.num_key_value_heads)
        fail("GQA draft backbones unsupported (num_attention_heads != "
             "num_key_value_heads); the shipped GLM-5.2 speculator is MHA");
    // Reduced draft vocab (TD-DSPARK-VOCAB-REMAP): the loader enforced the
    // d2t map's presence, dtype, shape, and target-range validity; the
    // finalize kernel remaps the draft-space argmax to target ids on-device.
    if (ck.draft_vocab_size != ck.vocab_size &&
        host.d2t.data.data() == nullptr)
        fail("reduced draft vocab without a d2t map (loader contract "
             "violated)");  // unreachable: load_dspark_draft fails first

    auto rt = std::unique_ptr<DsparkRuntime>(new DsparkRuntime());
    rt->ckpt_ = ck;
    rt->ranks_ = std::move(ranks);
    rt->communicator_ = communicator;
    rt->aux_ids_ = ck.aux_hidden_state_layer_ids;
    rt->H_ = static_cast<int>(ck.hidden_size);
    rt->n_heads_ = ck.num_attention_heads;
    rt->head_dim_ = static_cast<int>(ck.head_dim);
    rt->q_dim_ = rt->n_heads_ * rt->head_dim_;
    rt->kv_dim_ = ck.num_key_value_heads * rt->head_dim_;
    rt->I_ = static_cast<int>(ck.intermediate_size);
    rt->L_ = ck.num_hidden_layers;
    rt->block_size_ = ck.block_size;
    rt->spec_tokens_ = ck.speculative_tokens;
    rt->V_ = ck.vocab_size;
    rt->Vd_ = ck.draft_vocab_size;
    rt->mask_token_id_ = static_cast<int>(ck.mask_token_id);
    rt->eps_ = static_cast<float>(ck.rms_norm_eps);
    rt->theta_ = static_cast<float>(ck.rope_theta);
    rt->ctx_cap_ = dc.draft_context_capacity_tokens;
    rt->aux_rows_cap_ = dc.aux_capture_max_rows;
    if (rt->block_size_ > 16) fail("block_size > 16 unsupported");
    if (static_cast<int>(rt->aux_ids_.size()) > 31)
        fail("more than 31 aux layers unsupported");

    // Per-rank local dims (TD-DSPARK-DRAFT-SHARD). The loader's
    // dspark_shard_shape enforces per-tensor divisibility (incl. quant
    // scale-group alignment at the row-parallel split); these are the
    // runtime-facing head/vocab constraints.
    const int nr = static_cast<int>(rt->ranks_.size());
    if (nr > 1) {
        if (rt->n_heads_ % nr != 0)
            fail("num_attention_heads " + std::to_string(rt->n_heads_) +
                 " not divisible by " + std::to_string(nr) +
                 " draft ranks (head-sharded attention)");
        if (rt->I_ % nr != 0 || rt->Vd_ % nr != 0)
            fail("intermediate_size/draft_vocab_size not divisible by the "
                 "draft rank count");
    }
    rt->n_heads_local_ = rt->n_heads_ / nr;
    rt->q_dim_local_ = rt->q_dim_ / nr;
    rt->kv_dim_local_ = rt->kv_dim_ / nr;
    rt->I_local_ = rt->I_ / nr;
    rt->Vd_local_ = rt->Vd_ / nr;

    // Per-rank weight-shard uploads (rank 0 = the whole draft at nr==1).
    auto* backend = rt->ranks_[0].backend;
    rt->weights_ = model::upload_dspark_draft(
        host, *backend, arenas.empty() ? nullptr : arenas[0],
        arenas.empty() ? 0 : arena_bytes[0], dc.draft_weights_quant,
        /*rank=*/0, nr);
    rt->shard_.resize(static_cast<size_t>(nr) - 1);
    for (int r = 1; r < nr; ++r)
        rt->shard_[static_cast<size_t>(r) - 1].weights =
            model::upload_dspark_draft(
                host, *rt->ranks_[static_cast<size_t>(r)].backend,
                arenas.empty() ? nullptr : arenas[static_cast<size_t>(r)],
                arenas.empty() ? 0 : arena_bytes[static_cast<size_t>(r)],
                dc.draft_weights_quant, r, nr);

    // Device scratch: one buffer per layout item (order fixed above), PER
    // RANK. On the engine path the items are CARVED from that rank's arena
    // right after its weights (256-aligned) — the LayerRegistry budget
    // already covers weights + scratch + kDsparkArenaAlignSlack in each
    // draft GPU's pinned region, so no further device_alloc happens. Empty
    // arenas (unit tests): one owned device_alloc per item.
    auto carve_rank = [&](int r, const ScratchLayout& layout,
                          int64_t weights_bytes,
                          std::vector<void*>& owned) -> std::vector<void*> {
        auto* be = rt->ranks_[static_cast<size_t>(r)].backend;
        be->set_device();
        std::vector<void*> ptrs;
        ptrs.reserve(layout.items.size());
        if (!arenas.empty()) {
            void* arena = arenas[static_cast<size_t>(r)];
            const int64_t cap = arena_bytes[static_cast<size_t>(r)];
            auto align256 = [](int64_t v) {
                return (v + 255) & ~int64_t{255};
            };
            int64_t off = align256(weights_bytes);
            for (const auto& it : layout.items) {
                if (off + it.bytes > cap)
                    fail(std::string("draft arena too small for scratch "
                                     "item ") + it.name + " (rank " +
                         std::to_string(r) + "): need " +
                         std::to_string(off + it.bytes) + " > capacity " +
                         std::to_string(cap) +
                         " (LayerRegistry budget must cover weights + "
                         "scratch)");
                ptrs.push_back(static_cast<char*>(arena) + off);
                off = align256(off + it.bytes);
            }
            // owned stays empty: the region belongs to the VramAllocator.
        } else {
            for (const auto& it : layout.items) {
                void* p = be->device_alloc(static_cast<size_t>(it.bytes));
                if (!p) {
                    for (void* q : ptrs) be->device_free(q);
                    fail(std::string("device_alloc failed for ") + it.name +
                         " (" + std::to_string(it.bytes) +
                         " bytes) on draft GPU rank " + std::to_string(r));
                }
                ptrs.push_back(p);
                owned.push_back(p);
            }
        }
        return ptrs;
    };

    const auto layout = scratch_layout(dc, ck, /*rank=*/0, nr);
    auto ptrs = carve_rank(0, layout, rt->weights_.arena_bytes,
                           rt->owned_dev_);
    backend->set_device();
    if (ck.is_v4_dflash) {
        // V4 dflash unpack (v4_scratch_layout order).
        size_t j = 0;
        rt->aux_stage_ = ptrs[j++];
        rt->ctx_hidden_ = ptrs[j++];
        rt->ctx_normed_ = ptrs[j++];
        rt->ctx_ktmp_ = ptrs[j++];
        rt->kv_arena_ = ptrs[j++];
        rt->q_ids_ = ptrs[j++];
        rt->q_x_ = ptrs[j++];
        rt->q_normed_ = ptrs[j++];
        rt->hidden_out_ = ptrs[j++];
        rt->base_logits_ = ptrs[j++];
        rt->draft_ids_ = ptrs[j++];
        rt->markov_e_ = ptrs[j++];
        rt->markov_partials_ = ptrs[j++];
        rt->corrected_logits_ = ptrs[j++];
        rt->conf_out_ = ptrs[j++];
        rt->v4_rope_table_ = ptrs[j++];
        rt->v4_pos_ = ptrs[j++];
        rt->v4_x1_ = ptrs[j++];
        rt->v4_post_ = ptrs[j++];
        rt->v4_comb_ = ptrs[j++];
        rt->v4_qlat_ = ptrs[j++];
        rt->v4_qheads_ = ptrs[j++];
        rt->v4_qn_ = ptrs[j++];
        rt->v4_qr_ = ptrs[j++];
        rt->v4_kvlat_ = ptrs[j++];
        rt->v4_blkkv_ = ptrs[j++];
        rt->v4_attn_ = ptrs[j++];
        rt->v4_lse_ = ptrs[j++];
        rt->v4_oa_ = ptrs[j++];
        rt->v4_module_out_ = ptrs[j++];
        rt->v4_logits_e_ = ptrs[j++];
        rt->v4_topk_w_ = ptrs[j++];
        rt->v4_topk_idx_ = ptrs[j++];
        rt->v4_perm_in_ = ptrs[j++];
        rt->v4_exp_off_ = ptrs[j++];
        rt->v4_s2d_ = ptrs[j++];
        rt->v4_perm_idx_ = ptrs[j++];
        rt->v4_perm_ws_ = ptrs[j++];
        rt->v4_gate_out_ = ptrs[j++];
        rt->v4_up_out_ = ptrs[j++];
        rt->v4_gu_ = ptrs[j++];
        rt->v4_act_ = ptrs[j++];
        rt->v4_expert_out_ = ptrs[j++];
        rt->v4_sh_g_ = ptrs[j++];
        rt->v4_sh_u_ = ptrs[j++];
        rt->v4_sh_gu_ = ptrs[j++];
        rt->v4_sh_out_ = ptrs[j++];
        rt->v4_gemm_ws_ = ptrs[j++];
        rt->v4_gemm_ws_bytes_ = layout.items[j - 1].bytes;

        // Base rope table (theta from the checkpoint, NO yarn — the draft
        // blocks are ratio-0 V4 layers; dossier ticket-J decision).
        {
            const auto table = compute::build_rope_cos_sin_table(
                rt->ctx_cap_ + rt->block_size_,
                static_cast<int>(ck.v4.rope_dim), ck.rope_theta,
                std::nullopt);
            backend->memcpy_h2d(rt->v4_rope_table_, table.data(),
                                table.size() * sizeof(float));
        }
        // MoE runs on an OWNED ExpertDevice for the draft GPU (moe_permute /
        // gguf_grouped_gemm(MXFP4) / fused_swiglu / moe_unpermute).
        rt->v4_expert_dev_ =
            compute::make_cuda_sm120_expert_device(backend->gpu());

        // Events + logging shared with the legacy tail below; the EPM dump
        // is unsupported for the V4 draft (hc-wide residual) — never armed.
        rt->ev_ingest_done_ = backend->create_event();
        rt->ev_step_sync_ = backend->create_event();
        if (!epm_dump_dir(dc).empty())
            spdlog::warn("dspark_runtime: EPM feature dump is unsupported "
                         "for the V4 dflash draft — dump disabled");
        spdlog::info(
            "dspark_runtime: V4 dflash draft armed on GPU position {} — {} "
            "weight bytes + {} scratch bytes (ctx capacity {} tokens, "
            "gamma {} / block {}, window {}, {} MXFP4 experts)",
            backend->gpu().position, rt->weights_.arena_bytes, layout.total,
            rt->ctx_cap_, rt->spec_tokens_, rt->block_size_,
            ck.v4.sliding_window, ck.v4.n_routed_experts);
        return rt;
    }
    size_t i = 0;
    rt->aux_stage_ = ptrs[i++];
    rt->ctx_hidden_ = ptrs[i++];
    rt->ctx_normed_ = ptrs[i++];
    rt->ctx_ktmp_ = ptrs[i++];
    rt->kv_arena_ = ptrs[i++];
    rt->q_ids_ = ptrs[i++];
    rt->q_x_ = ptrs[i++];
    rt->q_normed_ = ptrs[i++];
    rt->q_tmp_ = ptrs[i++];
    rt->q_q_ = ptrs[i++];
    rt->q_k_ = ptrs[i++];
    rt->q_v_ = ptrs[i++];
    rt->q_attn_ = ptrs[i++];
    rt->q_oproj_ = ptrs[i++];
    rt->q_gate_ = ptrs[i++];
    rt->q_up_ = ptrs[i++];
    rt->q_act_ = ptrs[i++];
    rt->q_mlp_ = ptrs[i++];
    rt->hidden_out_ = ptrs[i++];
    rt->base_logits_ = ptrs[i++];
    rt->draft_ids_ = ptrs[i++];
    rt->markov_e_ = ptrs[i++];
    rt->markov_partials_ = ptrs[i++];
    rt->corrected_logits_ = ptrs[i++];
    rt->conf_out_ = ptrs[i++];
    if (nr > 1) {
        rt->logits_shard0_ = ptrs[i++];
        rt->peer_stage0_ = ptrs[i++];
    }

    // Shard ranks' scratch + events (rank >= 1).
    for (int r = 1; r < nr; ++r) {
        auto& sr = rt->shard_[static_cast<size_t>(r) - 1];
        const auto slayout = scratch_layout(dc, ck, r, nr);
        auto sptrs = carve_rank(r, slayout, sr.weights.arena_bytes,
                                sr.owned_dev);
        size_t j = 0;
        sr.ctx_normed = sptrs[j++];
        sr.ctx_ktmp = sptrs[j++];
        sr.kv_arena = sptrs[j++];
        sr.q_x = sptrs[j++];
        sr.q_normed = sptrs[j++];
        sr.q_tmp = sptrs[j++];
        sr.q_q = sptrs[j++];
        sr.q_k = sptrs[j++];
        sr.q_v = sptrs[j++];
        sr.q_attn = sptrs[j++];
        sr.q_oproj = sptrs[j++];
        sr.q_gate = sptrs[j++];
        sr.q_up = sptrs[j++];
        sr.q_act = sptrs[j++];
        sr.q_mlp = sptrs[j++];
        sr.hidden_out = sptrs[j++];
        sr.logits_shard = sptrs[j++];
        sr.peer_stage = sptrs[j++];
        auto* be = rt->ranks_[static_cast<size_t>(r)].backend;
        be->set_device();
        sr.ev_step_sync = be->create_event();
        sr.ev_xfer = be->create_event();
        sr.ev_append = be->create_event();
    }
    if (nr > 1) {
        backend->set_device();
        rt->ev_xfer0_ = backend->create_event();
    }
    backend->set_device();

    // EPM-1 (Phase 29): arm the feature-side dump. The staging item is the
    // LAST layout entry (present iff the dump dir resolved non-empty at
    // scratch_layout time — same cfg + env, so the index math holds).
    const std::string epm_dir = epm_dump_dir(dc);
    if (!epm_dir.empty()) {
        rt->epm_stage_ = ptrs[i++];
        rt->epm_writer_ = std::make_unique<EpmBlockDumper>(epm_dir);
        if (!rt->epm_writer_->ok()) {
            // Fail SOFT: collection must never take the engine down. The
            // staging region stays carved (budget already charged) but the
            // tap goes inert (null stage pointer).
            rt->epm_writer_.reset();
            rt->epm_stage_ = nullptr;
        } else {
            // NUMA-local pinned D2H staging on the DRAFT GPU's home node
            // (P-22 pattern: allocate_for_gpu + host_register_pinned;
            // fallback = backend pinned alloc when no NumaManager (tests)).
            const int64_t hid_cap =
                static_cast<int64_t>(rt->block_size_) * rt->L_ * rt->H_ *
                kBf16;
            rt->epm_hid_cap_bytes_ = hid_cap;
            const size_t host_bytes =
                static_cast<size_t>(hid_cap) + 64 + 64;  // + ids + conf
            if (numa) {
                auto buf = numa->allocate_for_gpu(host_bytes,
                                                  backend->gpu().position);
                rt->numa_ = numa;
                rt->epm_host_ = buf.data;
                rt->epm_host_bytes_ = buf.size;
                rt->epm_host_node_ = buf.numa_node;
                rt->epm_host_from_numa_ = true;
                const int rc =
                    core::host_register_pinned_portable(buf.data, buf.size);
                rt->epm_host_registered_ = (rc == 0);
                if (rc != 0)
                    spdlog::warn("dspark epm_dump: host_register failed "
                                 "(err {}) — staging unpinned (slow but "
                                 "correct)", rc);
            } else {
                rt->epm_host_ = backend->host_alloc_pinned(host_bytes);
                rt->epm_host_bytes_ = host_bytes;
                if (!rt->epm_host_) {
                    rt->epm_writer_.reset();
                    rt->epm_stage_ = nullptr;
                    spdlog::error("dspark epm_dump: pinned staging alloc "
                                  "failed — dump disabled");
                }
            }
            if (rt->epm_writer_)
                spdlog::info("dspark epm_dump: feature dump armed -> {} "
                             "(staging {} B device + {} B host on node {})",
                             epm_dir, hid_cap, rt->epm_host_bytes_,
                             rt->epm_host_node_);
        }
    }

    // ev_capture_done_ is created lazily by the TARGET backend at the first
    // capture (cudaEventRecord needs event+stream on the same device);
    // ev_ingest_done_ records on the draft stream -> draft device event.
    rt->ev_ingest_done_ = backend->create_event();
    rt->ev_step_sync_ = backend->create_event();

    spdlog::info(
        "dspark_runtime: armed on GPU position {} — {} weight bytes + {} "
        "scratch bytes (ctx capacity {} tokens, aux staging {} rows, "
        "gamma {} / block {})",
        backend->gpu().position, rt->weights_.arena_bytes, layout.total,
        rt->ctx_cap_, rt->aux_rows_cap_, rt->spec_tokens_, rt->block_size_);
    return rt;
}

DsparkRuntime::~DsparkRuntime() {
    if (ev_capture_done_ && capture_ev_backend_) {
        capture_ev_backend_->set_device();
        capture_ev_backend_->destroy_event(ev_capture_done_);
    }
    if (ranks_.empty() || !ranks_[0].backend) return;
    // Shard ranks (TD-DSPARK-DRAFT-SHARD): per-rank events + owned scratch
    // on each rank's own device; the shard weights free through their
    // DsparkDeviceWeights members.
    for (size_t r = 1; r <= shard_.size() && r < ranks_.size(); ++r) {
        auto& sr = shard_[r - 1];
        auto* be = ranks_[r].backend;
        if (!be) continue;
        be->set_device();
        if (sr.ev_step_sync) be->destroy_event(sr.ev_step_sync);
        if (sr.ev_xfer) be->destroy_event(sr.ev_xfer);
        if (sr.ev_append) be->destroy_event(sr.ev_append);
        for (void* p : sr.owned_dev) be->device_free(p);
    }
    auto* backend = ranks_[0].backend;
    backend->set_device();
    if (ev_ingest_done_) backend->destroy_event(ev_ingest_done_);
    if (ev_step_sync_) backend->destroy_event(ev_step_sync_);
    if (ev_xfer0_) backend->destroy_event(ev_xfer0_);
    for (void* p : owned_dev_) backend->device_free(p);
    // EPM-1 host staging teardown (unregister before free).
    if (epm_host_) {
        if (epm_host_registered_) core::host_unregister_pinned(epm_host_);
        if (epm_host_from_numa_ && numa_) {
            memory::NumaBuffer buf{epm_host_, epm_host_bytes_,
                                   epm_host_node_};
            numa_->free(buf);
        } else {
            backend->host_free_pinned(epm_host_);
        }
        epm_host_ = nullptr;
    }
}

int DsparkRuntime::draft_gpu_position() const {
    return ranks_[0].backend->gpu().position;
}

const float* DsparkRuntime::base_logits() const {
    return static_cast<const float*>(base_logits_);
}

const void* DsparkRuntime::hidden_out() const { return hidden_out_; }

void* DsparkRuntime::k_base(size_t rank, int layer) const {
    if (ckpt_.is_v4_dflash) {
        // Single-copy roped latents (K == V), row == absolute position.
        const int64_t layer_bytes =
            static_cast<int64_t>(ctx_cap_) * ckpt_.head_dim * kBf16;
        return static_cast<char*>(kv_arena_) + layer * layer_bytes;
    }
    const int64_t layer_bytes =
        int64_t{2} * ctx_cap_ * kv_dim_local_ * kBf16;  // K + V
    void* arena = rank == 0 ? kv_arena_ : shard_[rank - 1].kv_arena;
    return static_cast<char*>(arena) + layer * layer_bytes;
}

void* DsparkRuntime::v_base(size_t rank, int layer) const {
    if (ckpt_.is_v4_dflash) return k_base(rank, layer);  // V == K
    return static_cast<char*>(k_base(rank, layer)) +
           int64_t{1} * ctx_cap_ * kv_dim_local_ * kBf16;
}

// ── TP-seam GEMM helpers ────────────────────────────────────────────────────

void DsparkRuntime::weight_gemm(void* C, const void* A,
                                const model::DsparkDeviceTensor& W, int M,
                                int N, int K, int64_t lda, int64_t ldw,
                                int64_t k_off, int64_t row_off, bool out_fp32,
                                void* stream) const {
    const auto out = out_fp32 ? compute::GemmAccOutDtype::kFloat32
                              : compute::GemmAccOutDtype::kBFloat16;
    switch (W.dtype) {
    case model::DsparkWeightDtype::kBF16: {
        // Legacy path, bit-identical: lda == ldw == K collapses the strided
        // launcher to launch_bf16_gemm_nt (same kernels, same reduction
        // order — bf16_gemm.h contract).
        const auto* w = static_cast<const char*>(W.ptr) +
                        (row_off * ldw + k_off) * kBf16;
        compute::launch_bf16_gemm_nt_strided(
            C, A, w, M, N, K, lda, ldw, compute::GemmInDtype::kBFloat16, out,
            stream);
        return;
    }
    case model::DsparkWeightDtype::kFp8E4M3: {
        const auto* q = static_cast<const char*>(W.ptr) + row_off * ldw;
        const auto* s = static_cast<const char*>(W.scales) +
                        row_off * W.k_groups *
                            static_cast<int64_t>(sizeof(float));
        compute::launch_wq_gemm_nt(C, A, q, s,
                                   compute::WqWeightKind::kFp8E4M3, M, N, K,
                                   lda, ldw, k_off, W.k_groups, out, stream);
        return;
    }
    case model::DsparkWeightDtype::kNvfp4: {
        const auto* q =
            static_cast<const char*>(W.ptr) + row_off * (ldw / 2);
        const auto* s = static_cast<const char*>(W.scales) +
                        row_off * W.k_groups;
        compute::launch_wq_gemm_nt(C, A, q, s, compute::WqWeightKind::kNvfp4,
                                   M, N, K, lda, ldw, k_off, W.k_groups, out,
                                   stream);
        return;
    }
    }
    fail("weight_gemm: unknown weight dtype");
}

void DsparkRuntime::col_parallel_gemm(size_t rank, void* C, const void* A,
                                      const model::DsparkDeviceTensor& W,
                                      int M, int N, int K) const {
    // Column-parallel: rank r owns output rows [r*n_local, (r+1)*n_local) of
    // the logical [N, K] projection.  The loader uploaded each rank's rows
    // as a TIGHT [n_local, K] shard (dspark_tensor_shard_kind kColParallel),
    // so the GEMM runs over the whole shard tensor (row_off 0).  At nr==1
    // the shard IS the full tensor — bit-identical legacy path.
    const int nr = static_cast<int>(ranks_.size());
    const int n_local = N / nr;
    weight_gemm(C, A, W, M, n_local, K, /*lda=*/K, /*ldw=*/K, /*k_off=*/0,
                /*row_off=*/0, /*out_fp32=*/false, ranks_[rank].stream);
}

void DsparkRuntime::allreduce_seam(int layer, int site,
                                   int num_tokens) const {
    // Row-parallel combine point (o_proj partials: site 0 on q_oproj;
    // down_proj partials: site 1 on q_mlp).  nr==1: the single rank already
    // holds the full sum.  nr==2 (TD-DSPARK-DRAFT-SHARD): D2D cross-copy of
    // each rank's [num_tokens, H] BF16 partial into the peer's per-
    // (layer, site) staging slot, then a commutative in-place add on each
    // rank — IEEE addition commutes bitwise at two operands, so both ranks
    // hold the IDENTICAL full sum afterwards (the replicated residual
    // stream stays bit-converged across ranks).  Deliberately NOT NCCL: the
    // target's TP=2 attention collectives run concurrently on the same GPU
    // pair from other streams (see the header block); this is the
    // INV-MOE-EP-XTP D2D staging pattern.  Per-(layer, site) slots make the
    // staging write-once per step — no WAR events needed inside a step (the
    // run_step entry drain fences steps).
    const size_t nr = ranks_.size();
    if (nr == 1) return;
    const int64_t slot_off =
        (static_cast<int64_t>(layer) * 2 + site) *
        static_cast<int64_t>(block_size_) * H_ * kBf16;
    const size_t bytes =
        static_cast<size_t>(num_tokens) * static_cast<size_t>(H_) * kBf16;
    void* partial[2] = {site == 0 ? q_oproj_ : q_mlp_,
                        site == 0 ? shard_[0].q_oproj : shard_[0].q_mlp};
    void* stage[2] = {static_cast<char*>(peer_stage0_) + slot_off,
                      static_cast<char*>(shard_[0].peer_stage) + slot_off};
    void* ev[2] = {ev_xfer0_, shard_[0].ev_xfer};
    // Copy this rank's partial into the PEER's staging (enqueued on the
    // producing rank's stream — ordered after its GEMM), record the rank's
    // xfer event.
    for (size_t r = 0; r < 2; ++r) {
        auto& rk = ranks_[r];
        rk.backend->set_device();
        rk.backend->memcpy_d2d_async(stage[1 - r], partial[r], bytes,
                                     rk.stream);
        rk.backend->record_event(ev[r], rk.stream);
    }
    // Each rank waits the PEER's copy, then folds it in-place.
    for (size_t r = 0; r < 2; ++r) {
        auto& rk = ranks_[r];
        rk.backend->set_device();
        rk.backend->stream_wait_event(rk.stream, ev[1 - r]);
        compute::launch_residual_add(partial[r], stage[r],
                                     num_tokens * H_, rk.stream);
    }
    ranks_[0].backend->set_device();
}

// ── Aux-hidden capture + context ingest ─────────────────────────────────────

int DsparkRuntime::aux_slot_for_layer(uint32_t target_layer) const {
    for (size_t s = 0; s < aux_ids_.size(); ++s)
        if (aux_ids_[s] == static_cast<int>(target_layer))
            return static_cast<int>(s);
    return -1;
}

void DsparkRuntime::invalidate_context(const char* why) {
    if (ctx_valid_)
        spdlog::warn("dspark_runtime: drafting context invalidated — {} "
                     "(target decode unaffected; context re-arms on the next "
                     "position-0 capture)",
                     why);
    ctx_valid_ = false;
    cap_slot_mask_ = 0;
    cap_multi_ = false;
}

bool DsparkRuntime::ensure_capture_event(compute::DeviceBackend& src_backend) {
    // Lazy creation on the SOURCE device: cudaEventRecord requires the
    // event and the stream to live on the same device (the draft-device
    // event recorded on the target stream fails with
    // cudaErrorInvalidResourceHandle — caught live on the 3-GPU golden;
    // the same-GPU unit path cannot expose it).
    if (!ev_capture_done_) {
        ev_capture_done_ = src_backend.create_event();
        capture_ev_backend_ = &src_backend;
    } else if (capture_ev_backend_ != &src_backend) {
        invalidate_context("aux capture source backend changed");
        return false;
    }
    return true;
}

void DsparkRuntime::capture_aux(int slot, const void* target_attn_buf,
                                int rows, uint64_t seq_id, uint32_t start_pos,
                                compute::DeviceBackend& src_backend,
                                void* src_stream) {
    const int n_aux = aux_count();
    if (slot < 0 || slot >= n_aux || rows <= 0 || !target_attn_buf) return;
    if (static_cast<int64_t>(start_pos) + rows > ctx_cap_) {
        invalidate_context(
            "context KV arena overflow (TD-DSPARK-CTX-CAP)");
        return;
    }

    if (slot == 0) {
        // Context continuity: a new sequence may only begin at position 0;
        // rewinds (start_pos <= ctx_len, verification rollback) overwrite in
        // place; gaps kill drafting fail-closed.
        if (seq_id != ctx_seq_id_ || !ctx_valid_) {
            if (start_pos == 0) {
                ctx_seq_id_ = seq_id;
                ctx_len_ = 0;
                ctx_valid_ = true;
                cap_slot_mask_ = 0;
                cap_multi_ = false;
                spdlog::debug("dspark_runtime: context (re)armed for seq {}",
                              seq_id);
            } else if (ctx_valid_ && seq_id != ctx_seq_id_
                       && start_pos <= static_cast<uint32_t>(ctx_len_)) {
                // Prefix-cache fork ADOPTION (serving.prefix_cache): a NEW
                // sequence whose first fed row lands INSIDE the tracked
                // context adopts it — rebind + overwrite from start_pos
                // exactly like a same-seq rewind.  Sound because drafts
                // are ADVISORY (INV-DSPARK-LOSSLESS): the verify pass is
                // exact, so an adopted context can only change ACCEPTANCE,
                // never committed tokens; and on a true prefix-cache hit
                // the retained rows below start_pos ARE the shared prefix
                // (same tokens, same positions), so drafts keep real
                // context. A switch BEYOND the tracked frontier still
                // fails closed below (TD-DSPARK-BATCH).
                ctx_seq_id_ = seq_id;
                cap_slot_mask_ = 0;
                cap_multi_ = false;
                spdlog::debug("dspark_runtime: context ADOPTED by seq {} at "
                              "pos {} (ctx_len {})", seq_id, start_pos,
                              ctx_len_);
            } else if (seq_id != ctx_seq_id_) {
                invalidate_context("sequence switch mid-context "
                                   "(TD-DSPARK-BATCH)");
                return;
            } else {
                return;  // invalid context, not a restart — stay dormant
            }
        }
        // TD-DSPARK-SUPERCHUNK-CAPTURE: a slot-0 window that EXTENDS the
        // open epoch (only slot 0 captured so far, start_pos == its covered
        // end) is a superchunk sub-chunk arriving chunk-major — fold it
        // into the per-slot fc accumulator instead of re-basing the window.
        if (cap_slot_mask_ == 1u && cap_seq_ == seq_id &&
            start_pos == cap_slot_end_[0]) {
            if (!cap_multi_ &&
                !begin_multi_window_epoch(src_backend, src_stream))
                return;
            cap_multi_ = true;
            capture_slot_chunked(0, target_attn_buf, rows,
                                 static_cast<int64_t>(start_pos) -
                                     cap_start_pos_,
                                 src_backend, src_stream);
            if (ctx_valid_)
                cap_slot_end_[0] = start_pos + static_cast<uint32_t>(rows);
            return;
        }
        // Epoch start (a fresh slot-0 window always re-bases the capture
        // window — pre-superchunk behavior; an abandoned earlier epoch is
        // caught by the position-gap/last-slot completeness contracts).
        if (start_pos > static_cast<uint32_t>(ctx_len_)) {
            invalidate_context("position gap in captured hiddens");
            return;
        }
        cap_rows_ = rows;
        cap_start_pos_ = start_pos;
        cap_seq_ = seq_id;
        cap_slot_mask_ = 1u;
        cap_multi_ = false;
        cap_slot_end_.assign(static_cast<size_t>(n_aux), start_pos);
        cap_slot_end_[0] = start_pos + static_cast<uint32_t>(rows);
    } else {
        if (!ctx_valid_) return;
        if (cap_multi_) {
            // Multi-window epoch (superchunk): per-slot contiguous window
            // accumulation.  A slot's FIRST window starts at the epoch base
            // with the previous slot's coverage complete (chunk-major layer
            // order); continuation windows extend the slot's own high-water;
            // no window may overshoot slot 0's coverage (the epoch target).
            const uint32_t target = cap_slot_end_[0];
            const bool first = (cap_slot_mask_ & (1u << slot)) == 0;
            const uint32_t expect =
                first ? cap_start_pos_
                      : cap_slot_end_[static_cast<size_t>(slot)];
            if (cap_seq_ != seq_id || start_pos != expect ||
                static_cast<uint64_t>(start_pos) + rows > target ||
                (first &&
                 cap_slot_end_[static_cast<size_t>(slot) - 1] != target)) {
                invalidate_context("superchunk aux window sequence mismatch "
                                   "across slots");
                return;
            }
            cap_slot_mask_ |= 1u << slot;
            capture_slot_chunked(slot, target_attn_buf, rows,
                                 static_cast<int64_t>(start_pos) -
                                     cap_start_pos_,
                                 src_backend, src_stream);
            if (!ctx_valid_) return;
            cap_slot_end_[static_cast<size_t>(slot)] =
                start_pos + static_cast<uint32_t>(rows);
            if (slot == n_aux - 1 &&
                cap_slot_end_[static_cast<size_t>(slot)] == target) {
                if (cap_slot_mask_ != (1u << n_aux) - 1u) {
                    invalidate_context("incomplete aux slot set at the last "
                                       "aux layer");
                    return;
                }
                finalize_context_chunked(
                    static_cast<int>(target - cap_start_pos_),
                    cap_start_pos_);
                ctx_len_ = static_cast<int>(target);
                cap_slot_mask_ = 0;
                cap_multi_ = false;
                // Restore the caller's (target) device context.
                src_backend.set_device();
            }
            return;
        }
        if (cap_seq_ != seq_id || cap_rows_ != rows ||
            cap_start_pos_ != start_pos || (cap_slot_mask_ & 1u) == 0 ||
            (cap_slot_mask_ & (1u << slot)) != 0) {
            invalidate_context("aux capture step shape mismatch across "
                               "layers");
            return;
        }
        cap_slot_mask_ |= 1u << slot;
        cap_slot_end_[static_cast<size_t>(slot)] =
            start_pos + static_cast<uint32_t>(rows);
    }

    if (rows > aux_rows_cap_) {
        // Chunked capture (TD-DSPARK-PREFILL-CAP): the step exceeds the
        // staging, so this slot's rows fold into the fc accumulator in
        // staging-sized pieces as they stream in — drafting stays armed for
        // whole-prompt prefills / oversized chunks up to the arena capacity.
        capture_slot_chunked(slot, target_attn_buf, rows, /*acc_row_base=*/0,
                             src_backend, src_stream);
        if (!ctx_valid_) return;  // chunked path may have invalidated
        if (slot == n_aux - 1) {
            if (cap_slot_mask_ != (1u << n_aux) - 1u) {
                invalidate_context("incomplete aux slot set at the last aux "
                                   "layer");
                return;
            }
            finalize_context_chunked(rows, start_pos);
            ctx_len_ = static_cast<int>(start_pos) + rows;
            cap_slot_mask_ = 0;
            // Restore the caller's (target) device context: the draft-side
            // work switched to the draft GPU mid-attention-dispatch.
            src_backend.set_device();
        }
        return;
    }

    // Single-shot path (rows fit the staging — bit-identical fast path).
    // Do not clobber the staging buffer while a previous ingest (draft
    // stream) may still read it: the target stream waits the last recorded
    // ingest event (no-op when never recorded).
    if (slot == 0 && ingest_recorded_)
        src_backend.stream_wait_event(src_stream, ev_ingest_done_);

    // Cross-GPU strided copy: [rows, H] tight source rows -> column slot
    // `slot` of the [rows, n_aux*H] staging on the draft GPU.  Enqueued on
    // the TARGET's kAttention stream (ordered after the previous layer's MoE
    // commit wait, so attn_buf holds the input of aux layer `slot`).
    auto* dst = static_cast<char*>(aux_stage_) +
                static_cast<int64_t>(slot) * H_ * kBf16;
    const size_t row_bytes = static_cast<size_t>(H_) * kBf16;
    src_backend.memcpy_2d_async(dst,
                                static_cast<size_t>(n_aux) * row_bytes,
                                target_attn_buf, row_bytes, row_bytes,
                                static_cast<size_t>(rows), src_stream);

    if (slot == n_aux - 1) {
        if (cap_slot_mask_ != (1u << n_aux) - 1u) {
            invalidate_context("incomplete aux slot set at the last aux "
                               "layer");
            return;
        }
        if (!ensure_capture_event(src_backend)) return;
        src_backend.record_event(ev_capture_done_, src_stream);
        ingest_context(rows, start_pos);
        ctx_len_ = static_cast<int>(start_pos) + rows;
        cap_slot_mask_ = 0;
        // Restore the caller's (target) device context: ingest_context
        // switched to the draft GPU mid-attention-dispatch.
        src_backend.set_device();
    }
}

void DsparkRuntime::capture_slot_chunked(int slot,
                                         const void* target_attn_buf,
                                         int rows, int64_t acc_row_base,
                                         compute::DeviceBackend& src_backend,
                                         void* src_stream) {
    // fc distributes over the aux concat: fc(concat_s x_s) = Σ_s fc_s @ x_s
    // with fc_s = fc columns [s*H, (s+1)*H).  Each staging-sized piece of
    // this slot's rows is copied into the slot's staging column, GEMM'd
    // against fc_s (strided operands — no tight repack), and added into the
    // accumulator rows of ctx_hidden_ at acc_row_base (the window's offset
    // from the epoch base — 0 for single-window epochs; superchunk windows
    // land at their absolute-position offset, TD-DSPARK-SUPERCHUNK-CAPTURE).
    // Ping-pong: the target stream reuses
    // the staging only after the draft stream consumed the previous piece
    // (ev_ingest_done_); the draft stream reads a piece only after its copy
    // landed (ev_capture_done_).  Host-ordered enqueues make the two-event
    // reuse safe (each wait binds the event state at call time).
    //
    // Numerics vs the single-shot fc GEMM: the per-slot partials round to
    // BF16 before the adds (n_aux-1 extra BF16 roundings per element) —
    // bounded by the BF16 pipeline noise the context path already carries;
    // the single-shot path is untouched.
    if (!ensure_capture_event(src_backend)) return;

    const int n_aux = aux_count();
    const size_t row_bytes = static_cast<size_t>(H_) * kBf16;
    const int64_t ld = static_cast<int64_t>(n_aux) * H_;  // staging/fc stride
    auto* stage_slot = static_cast<char*>(aux_stage_) +
                       static_cast<int64_t>(slot) * H_ * kBf16;
    const int64_t fc_k_off = static_cast<int64_t>(slot) * H_;  // fc col window
    auto& rk = ranks_[0];

    for (int off = 0; off < rows; off += aux_rows_cap_) {
        const int chunk = std::min(aux_rows_cap_, rows - off);

        // Target side: reuse the staging slot only once the draft consumed
        // the previous piece (or the previous step's ingest).
        if (ingest_recorded_)
            src_backend.stream_wait_event(src_stream, ev_ingest_done_);
        src_backend.memcpy_2d_async(
            stage_slot, static_cast<size_t>(n_aux) * row_bytes,
            static_cast<const char*>(target_attn_buf) +
                static_cast<int64_t>(off) * static_cast<int64_t>(row_bytes),
            row_bytes, row_bytes, static_cast<size_t>(chunk), src_stream);
        src_backend.record_event(ev_capture_done_, src_stream);

        // Draft side: fold the piece into the fc accumulator rows
        // [off, off+chunk).  Slot 0 initializes (plain GEMM store); later
        // slots GEMM into ctx_normed_ (free until the finalize) and add.
        rk.backend->set_device();
        rk.backend->stream_wait_event(rk.stream, ev_capture_done_);
        auto* acc = static_cast<char*>(ctx_hidden_) +
                    (acc_row_base + off) * H_ * kBf16;
        if (slot == 0) {
            weight_gemm(acc, stage_slot, weights_.fc, chunk, H_, H_,
                        /*lda=*/ld, /*ldw=*/ld, fc_k_off, /*row_off=*/0,
                        /*out_fp32=*/false, rk.stream);
        } else {
            weight_gemm(ctx_normed_, stage_slot, weights_.fc, chunk, H_, H_,
                        /*lda=*/ld, /*ldw=*/ld, fc_k_off, /*row_off=*/0,
                        /*out_fp32=*/false, rk.stream);
            compute::launch_residual_add(acc, ctx_normed_, chunk * H_,
                                         rk.stream);
        }
        rk.backend->record_event(ev_ingest_done_, rk.stream);
        ingest_recorded_ = true;
    }
    // Restore the caller's (target) device context — non-final slots return
    // to the attention dispatch from here.
    src_backend.set_device();
}

bool DsparkRuntime::begin_multi_window_epoch(
        compute::DeviceBackend& src_backend, void* src_stream) {
    // TD-DSPARK-SUPERCHUNK-CAPTURE: slot 0's SECOND contiguous window just
    // arrived, so the open epoch becomes multi-window (per-slot fc
    // accumulation).  The FIRST window was handled by the single-window
    // epoch-start path:
    //   (a) rows > staging: capture_slot_chunked already folded it into
    //       ctx_hidden_ rows [0, cap_rows_) — nothing to do;
    //   (b) rows <= staging: its raw rows sit in staging column 0 (the copy
    //       was enqueued on THIS src_stream — the target kAttention stream —
    //       so recording the capture event NOW orders after it) — fold the
    //       slot-0 fc partial into ctx_hidden_ rows [0, cap_rows_).
    if (cap_rows_ > aux_rows_cap_) return true;
    if (!ensure_capture_event(src_backend)) return false;
    src_backend.record_event(ev_capture_done_, src_stream);

    const int n_aux = aux_count();
    const int64_t ld = static_cast<int64_t>(n_aux) * H_;  // staging/fc stride
    auto& rk = ranks_[0];
    rk.backend->set_device();
    rk.backend->stream_wait_event(rk.stream, ev_capture_done_);
    weight_gemm(ctx_hidden_, aux_stage_, weights_.fc, cap_rows_, H_, H_,
                /*lda=*/ld, /*ldw=*/ld, /*k_off=*/0, /*row_off=*/0,
                /*out_fp32=*/false, rk.stream);
    rk.backend->record_event(ev_ingest_done_, rk.stream);
    ingest_recorded_ = true;
    src_backend.set_device();
    return true;
}

void DsparkRuntime::v4_append_context_kv(const void* normed, int rows,
                                         uint32_t start_pos, void* s) {
    // Ticket J (vLLM nvidia/dspark.py precompute_and_store_context_kv):
    // per draft layer, kv = kv_norm_l(wkv_l(main_x)) over the FULL head_dim
    // (V4 norms the pe half too) → rope the tail at the absolute positions
    // → arena rows [start_pos, start_pos + rows). K == V — one copy.
    const int D = static_cast<int>(ckpt_.head_dim);
    const int R = static_cast<int>(ckpt_.v4.rope_dim);
    for (int l = 0; l < L_; ++l) {
        const auto& lw = weights_.v4->layers[static_cast<size_t>(l)];
        compute::launch_bf16_gemm_nt(
            ctx_ktmp_, normed, lw.kv.ptr, rows, D, H_,
            compute::GemmInDtype::kBFloat16,
            compute::GemmAccOutDtype::kBFloat16, s);
        compute::launch_rmsnorm(ctx_ktmp_, ctx_ktmp_, lw.kv_norm.ptr, eps_,
                                rows, D, compute::NormDtype::kBFloat16, s);
        auto* dst = static_cast<char*>(k_base(0, l)) +
                    static_cast<int64_t>(start_pos) * D * kBf16;
        compute::launch_dspark_v4_kv_rope(dst, ctx_ktmp_, v4_rope_table_,
                                          static_cast<int>(start_pos), rows,
                                          D, R, s);
    }
}

void DsparkRuntime::append_context_kv(const void* normed, int rows,
                                      uint32_t start_pos, void* s) {
    if (ckpt_.is_v4_dflash) {
        v4_append_context_kv(normed, rows, start_pos, s);
        return;
    }
    // Per-layer context K/V append at absolute positions (vLLM
    // precompute_and_store_context_kv): K = rope(k_norm(k_proj(x))),
    // V = v_proj(x) raw.  V GEMMs straight into the arena slice (tight
    // rows); K goes GEMM -> per-head norm into the arena -> in-place RoPE.
    //
    // Sharded (nr>1, TD-DSPARK-DRAFT-SHARD): the normed ctx rows (produced
    // on rank 0, `normed` == ctx_normed_) are BROADCAST to each shard
    // rank's replica, then every rank appends ITS head-shard of K/V
    // (kv_dim_local_ columns via its column-parallel k/v_proj shard) into
    // its own arena — attention on rank r reads only rank r's shard.
    // WAR guard: rank 0 re-broadcasts into the replica only after the
    // shard rank's PREVIOUS append finished reading it (ev_append).
    const size_t nr = ranks_.size();
    if (nr > 1) {
        auto& rk0 = ranks_[0];
        rk0.backend->set_device();
        for (size_t r = 1; r < nr; ++r) {
            rk0.backend->stream_wait_event(s, shard_[r - 1].ev_append);
            rk0.backend->memcpy_d2d_async(
                shard_[r - 1].ctx_normed, normed,
                static_cast<size_t>(rows) * static_cast<size_t>(H_) * kBf16,
                s);
        }
        rk0.backend->record_event(ev_xfer0_, s);
        for (size_t r = 1; r < nr; ++r) {
            auto& rk = ranks_[r];
            rk.backend->set_device();
            rk.backend->stream_wait_event(rk.stream, ev_xfer0_);
        }
    }
    const int64_t pos_off =
        static_cast<int64_t>(start_pos) * kv_dim_local_ * kBf16;
    for (size_t r = 0; r < nr; ++r) {
        auto& rk = ranks_[r];
        rk.backend->set_device();
        void* rs = r == 0 ? s : rk.stream;
        const void* rnormed = r == 0 ? normed : shard_[r - 1].ctx_normed;
        void* ktmp = r == 0 ? ctx_ktmp_ : shard_[r - 1].ctx_ktmp;
        for (int l = 0; l < L_; ++l) {
            const auto& lw = rank_weights(r).layers[static_cast<size_t>(l)];
            auto* kdst = static_cast<char*>(k_base(r, l)) + pos_off;
            weight_gemm(ktmp, rnormed, lw.k_proj, rows, kv_dim_local_, H_,
                        /*lda=*/H_, /*ldw=*/H_, 0, 0, /*out_fp32=*/false,
                        rs);
            compute::launch_rmsnorm(kdst, ktmp, lw.k_norm.ptr, eps_,
                                    rows * n_heads_local_, head_dim_,
                                    compute::NormDtype::kBFloat16, rs);
            compute::launch_dspark_rope(kdst, rows, n_heads_local_,
                                        head_dim_,
                                        static_cast<int>(start_pos), theta_,
                                        rs);
            auto* vdst = static_cast<char*>(v_base(r, l)) + pos_off;
            weight_gemm(vdst, rnormed, lw.v_proj, rows, kv_dim_local_, H_,
                        /*lda=*/H_, /*ldw=*/H_, 0, 0, /*out_fp32=*/false,
                        rs);
        }
        if (r > 0)
            rk.backend->record_event(shard_[r - 1].ev_append, rk.stream);
    }
    if (nr > 1) ranks_[0].backend->set_device();
}

void DsparkRuntime::ingest_context(int rows, uint32_t start_pos) {
    auto& rk = ranks_[0];
    rk.backend->set_device();
    void* s = rk.stream;

    // Draft stream consumes the staging only after the target's copies land.
    rk.backend->stream_wait_event(s, ev_capture_done_);

    // fc fusion: [rows, N_aux*H] @ fc[H, N_aux*H]^T -> fused context hidden.
    const int64_t fc_k = static_cast<int64_t>(aux_count()) * H_;
    weight_gemm(ctx_hidden_, aux_stage_, weights_.fc, rows, H_,
                static_cast<int>(fc_k), /*lda=*/fc_k, /*ldw=*/fc_k, 0, 0,
                /*out_fp32=*/false, s);
    compute::launch_rmsnorm(ctx_normed_, ctx_hidden_,
                            weights_.hidden_norm.ptr, eps_, rows, H_,
                            compute::NormDtype::kBFloat16, s);
    append_context_kv(ctx_normed_, rows, start_pos, s);

    rk.backend->record_event(ev_ingest_done_, s);
    ingest_recorded_ = true;
}

void DsparkRuntime::finalize_context_chunked(int rows, uint32_t start_pos) {
    // The fc accumulation over every slot's pieces is already stream-ordered
    // on the draft stream (capture_slot_chunked); hidden_norm + per-layer KV
    // proceed in staging-sized row pieces so ctx_normed_/ctx_ktmp_ keep
    // their aux_rows_cap sizing.  Row-wise ops chunk exactly (positions
    // offset per piece).
    auto& rk = ranks_[0];
    rk.backend->set_device();
    void* s = rk.stream;

    for (int off = 0; off < rows; off += aux_rows_cap_) {
        const int chunk = std::min(aux_rows_cap_, rows - off);
        const auto* acc = static_cast<const char*>(ctx_hidden_) +
                          static_cast<int64_t>(off) * H_ * kBf16;
        compute::launch_rmsnorm(ctx_normed_, acc, weights_.hidden_norm.ptr,
                                eps_, chunk, H_,
                                compute::NormDtype::kBFloat16, s);
        append_context_kv(ctx_normed_, chunk,
                          start_pos + static_cast<uint32_t>(off), s);
    }

    rk.backend->record_event(ev_ingest_done_, s);
    ingest_recorded_ = true;
}

// ── Backbone forward ────────────────────────────────────────────────────────

bool DsparkRuntime::run_step(uint64_t seq_id, uint32_t anchor_token_id,
                             uint32_t anchor_pos, int num_query,
                             std::string* err) {
    auto set_err = [&](const std::string& m) {
        if (err) *err = "dspark run_step: " + m;
        return false;
    };
    if (!ctx_valid_ || seq_id != ctx_seq_id_)
        return set_err("no valid ingested context for seq " +
                       std::to_string(seq_id) + " (tracked seq " +
                       std::to_string(ctx_seq_id_) + ", valid " +
                       std::to_string(ctx_valid_) + ")");
    const int ndraft = num_query > 0 ? num_query : spec_tokens_;
    // Bonus-anchor layout (see header): one extra non-predicting physical
    // row for the anchor; the ndraft mask rows sit AT their predicted
    // positions anchor+1..anchor+ndraft.
    const int nq = ndraft + 1;
    if (nq > block_size_)
        return set_err("query rows " + std::to_string(nq) + " > block_size " +
                       std::to_string(block_size_));
    if (anchor_pos > static_cast<uint32_t>(ctx_len_))
        return set_err("anchor_pos " + std::to_string(anchor_pos) +
                       " beyond ingested context length " +
                       std::to_string(ctx_len_));
    if (static_cast<int64_t>(anchor_token_id) >= V_)
        return set_err("anchor token id out of vocab");

    auto& rk = ranks_[0];
    rk.backend->set_device();
    void* s = rk.stream;

    // LS_DSPARK_PROF=1: split the run_step wall into the entry
    // synchronize_device (drains the draft GPU) vs the enqueue tail —
    // DSP52_BOOST lever-2 profiling (the whole handler runs on the daemon
    // thread, so any CPU-side cost here serializes against target-side
    // command dispatch and cannot be hidden by command overlap).
    static const bool prof = [] {
        const char* e = std::getenv("LS_DSPARK_PROF");
        return e && *e == '1';
    }();
    const auto t_entry = std::chrono::steady_clock::now();

    // Query slot ids: [anchor, mask x (nq-1)] — INV-DSPARK-ANCHOR (every
    // slot is a prediction; slot k's logits sample position anchor_pos+k+1).
    // Drain the DRAFT STREAM before overwriting the persistent host staging
    // / device ids a prior step's pending work may still read. Every
    // consumer of host_ids_/q_ids_/scratch is enqueued on rk.stream, so a
    // stream-scoped drain (event record + host spin) suffices — a
    // device-wide synchronize_device would ALSO serialize against unrelated
    // same-GPU work: kExpertFfn GEMMs on a draft-hosting 5080, or the whole
    // TP pipeline when the quantized draft hosts on a 5090
    // (TD-DSPARK-DRAFT-QUANT placement).
    // Entry drain on EVERY rank's stream (persistent host staging / device
    // ids / peer staging may still be read by the previous step's work).
    rk.backend->record_event(ev_step_sync_, s);
    for (size_t r = 1; r < ranks_.size(); ++r) {
        ranks_[r].backend->set_device();
        ranks_[r].backend->record_event(shard_[r - 1].ev_step_sync,
                                        ranks_[r].stream);
    }
    rk.backend->set_device();
    while (rk.backend->query_event(ev_step_sync_).status ==
           compute::EventStatus::kNotReady) {
    }
    for (size_t r = 1; r < ranks_.size(); ++r) {
        ranks_[r].backend->set_device();
        while (ranks_[r].backend->query_event(shard_[r - 1].ev_step_sync)
                   .status == compute::EventStatus::kNotReady) {
        }
    }
    rk.backend->set_device();
    const auto t_sync = std::chrono::steady_clock::now();
    host_ids_[0] = static_cast<int32_t>(anchor_token_id);
    for (int k = 1; k < nq; ++k)
        host_ids_[static_cast<size_t>(k)] = mask_token_id_;
    rk.backend->memcpy_h2d_async(q_ids_, host_ids_.data(),
                                 static_cast<size_t>(nq) * sizeof(int32_t),
                                 s);

    // ── Ticket J: V4 dflash backbone dispatch ──
    if (ckpt_.is_v4_dflash) {
        std::string verr;
        if (!run_step_v4(anchor_token_id, anchor_pos, nq, &verr))
            return set_err(verr);
        if (prof) {
            const auto t_end = std::chrono::steady_clock::now();
            std::fprintf(
                stderr,
                "[dspark-prof] run_step(v4): sync=%.3f ms enqueue=%.3f ms\n",
                std::chrono::duration<double, std::milli>(t_sync - t_entry)
                    .count(),
                std::chrono::duration<double, std::milli>(t_end - t_sync)
                    .count());
        }
        last_num_query_ = ndraft;
        markov_ran_ = false;
        return true;
    }

    compute::launch_embedding_lookup(
        q_x_, weights_.embed_tokens.ptr, static_cast<const int32_t*>(q_ids_),
        nq, static_cast<int>(V_), H_, compute::EmbeddingDtype::kBFloat16, s);

    // Sharded: broadcast the embeds so every rank runs the REPLICATED
    // residual stream (identical elementwise/norm inputs -> bit-identical
    // replicas; embed_tokens is single-homed on rank 0).
    const size_t nranks = ranks_.size();
    if (nranks > 1) {
        for (size_t r = 1; r < nranks; ++r)
            rk.backend->memcpy_d2d_async(
                shard_[r - 1].q_x, q_x_,
                static_cast<size_t>(nq) * static_cast<size_t>(H_) * kBf16,
                s);
        rk.backend->record_event(ev_xfer0_, s);
        for (size_t r = 1; r < nranks; ++r) {
            ranks_[r].backend->set_device();
            ranks_[r].backend->stream_wait_event(ranks_[r].stream,
                                                 ev_xfer0_);
        }
        rk.backend->set_device();
    }

    // Per-rank buffer selectors (rank 0 = legacy members).
    auto b_qx = [&](size_t r) { return r == 0 ? q_x_ : shard_[r - 1].q_x; };
    auto b_qnormed = [&](size_t r) {
        return r == 0 ? q_normed_ : shard_[r - 1].q_normed;
    };
    auto b_qtmp = [&](size_t r) {
        return r == 0 ? q_tmp_ : shard_[r - 1].q_tmp;
    };
    auto b_qq = [&](size_t r) { return r == 0 ? q_q_ : shard_[r - 1].q_q; };
    auto b_qk = [&](size_t r) { return r == 0 ? q_k_ : shard_[r - 1].q_k; };
    auto b_qv = [&](size_t r) { return r == 0 ? q_v_ : shard_[r - 1].q_v; };
    auto b_qattn = [&](size_t r) {
        return r == 0 ? q_attn_ : shard_[r - 1].q_attn;
    };
    auto b_qoproj = [&](size_t r) {
        return r == 0 ? q_oproj_ : shard_[r - 1].q_oproj;
    };
    auto b_qgate = [&](size_t r) {
        return r == 0 ? q_gate_ : shard_[r - 1].q_gate;
    };
    auto b_qup = [&](size_t r) {
        return r == 0 ? q_up_ : shard_[r - 1].q_up;
    };
    auto b_qact = [&](size_t r) {
        return r == 0 ? q_act_ : shard_[r - 1].q_act;
    };
    auto b_qmlp = [&](size_t r) {
        return r == 0 ? q_mlp_ : shard_[r - 1].q_mlp;
    };
    auto b_hidden = [&](size_t r) {
        return r == 0 ? hidden_out_ : shard_[r - 1].hidden_out;
    };

    // Layer 0 pre-norm (residual = raw embeds in q_x), replicated per rank.
    for (size_t r = 0; r < nranks; ++r) {
        ranks_[r].backend->set_device();
        compute::launch_rmsnorm(
            b_qnormed(r), b_qx(r),
            rank_weights(r).layers[0].input_layernorm.ptr, eps_, nq, H_,
            compute::NormDtype::kBFloat16, ranks_[r].stream);
    }
    rk.backend->set_device();

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    const int base_pos = static_cast<int>(anchor_pos);

    for (int l = 0; l < L_; ++l) {
        for (size_t r = 0; r < nranks; ++r) {
            const auto& lw =
                rank_weights(r).layers[static_cast<size_t>(l)];
            ranks_[r].backend->set_device();
            // Q/K/V column-parallel projections + per-head QK-RMSNorm + RoPE
            // on this rank's head shard.
            col_parallel_gemm(r, b_qtmp(r), b_qnormed(r), lw.q_proj, nq,
                              q_dim_, H_);
            compute::launch_rmsnorm(b_qq(r), b_qtmp(r), lw.q_norm.ptr, eps_,
                                    nq * n_heads_local_, head_dim_,
                                    compute::NormDtype::kBFloat16,
                                    ranks_[r].stream);
            compute::launch_dspark_rope(b_qq(r), nq, n_heads_local_,
                                        head_dim_, base_pos, theta_,
                                        ranks_[r].stream);
            col_parallel_gemm(r, b_qtmp(r), b_qnormed(r), lw.k_proj, nq,
                              kv_dim_, H_);
            compute::launch_rmsnorm(b_qk(r), b_qtmp(r), lw.k_norm.ptr, eps_,
                                    nq * n_heads_local_, head_dim_,
                                    compute::NormDtype::kBFloat16,
                                    ranks_[r].stream);
            compute::launch_dspark_rope(b_qk(r), nq, n_heads_local_,
                                        head_dim_, base_pos, theta_,
                                        ranks_[r].stream);
            col_parallel_gemm(r, b_qv(r), b_qnormed(r), lw.v_proj, nq,
                              kv_dim_, H_);

            // Non-causal block attention over context (< anchor_pos) + block
            // on this rank's head shard of the context-KV arena.
            compute::launch_dspark_block_attention(
                b_qattn(r), b_qq(r),
                anchor_pos > 0 ? k_base(r, l) : nullptr,
                anchor_pos > 0 ? v_base(r, l) : nullptr, b_qk(r), b_qv(r),
                nq, base_pos, n_heads_local_, head_dim_, scale,
                ranks_[r].stream);

            // o_proj row-parallel partial (K-window shard at nr>1; the
            // whole projection at nr==1) — combined at the seam below.
            weight_gemm(b_qoproj(r), b_qattn(r), lw.o_proj, nq, H_,
                        q_dim_local_, /*lda=*/q_dim_local_,
                        /*ldw=*/q_dim_local_, 0, 0, /*out_fp32=*/false,
                        ranks_[r].stream);
        }
        allreduce_seam(l, /*site=*/0, nq);

        // residual += attn; normed = rmsnorm(residual, post_attn_ln) —
        // replicated per rank (identical inputs after the combine).
        for (size_t r = 0; r < nranks; ++r) {
            const auto& lw =
                rank_weights(r).layers[static_cast<size_t>(l)];
            ranks_[r].backend->set_device();
            compute::launch_fused_add_rmsnorm(
                b_qnormed(r), b_qoproj(r), b_qx(r),
                lw.post_attention_layernorm.ptr, eps_, nq, H_,
                compute::NormDtype::kBFloat16, ranks_[r].stream);
        }

        for (size_t r = 0; r < nranks; ++r) {
            const auto& lw =
                rank_weights(r).layers[static_cast<size_t>(l)];
            ranks_[r].backend->set_device();
            col_parallel_gemm(r, b_qgate(r), b_qnormed(r), lw.gate_proj, nq,
                              I_, H_);
            col_parallel_gemm(r, b_qup(r), b_qnormed(r), lw.up_proj, nq, I_,
                              H_);
            compute::launch_dspark_silu_mul(
                b_qact(r), b_qgate(r), b_qup(r),
                static_cast<long long>(nq) * I_local_, ranks_[r].stream);
            // down_proj row-parallel partial + combine seam.
            weight_gemm(b_qmlp(r), b_qact(r), lw.down_proj, nq, H_,
                        I_local_, /*lda=*/I_local_, /*ldw=*/I_local_, 0, 0,
                        /*out_fp32=*/false, ranks_[r].stream);
        }
        allreduce_seam(l, /*site=*/1, nq);

        // residual += mlp; pre-norm for the next layer, or the final norm
        // into hidden_out (what the lm_head + DSP-4/DSP-6 heads consume).
        const bool last = (l + 1 == L_);
        for (size_t r = 0; r < nranks; ++r) {
            const auto& rw = rank_weights(r);
            ranks_[r].backend->set_device();
            compute::launch_fused_add_rmsnorm(
                last ? b_hidden(r) : b_qnormed(r), b_qmlp(r), b_qx(r),
                last ? rw.final_norm.ptr
                     : rw.layers[static_cast<size_t>(l) + 1]
                           .input_layernorm.ptr,
                eps_, nq, H_, compute::NormDtype::kBFloat16,
                ranks_[r].stream);
        }
        rk.backend->set_device();

        // EPM-1 feature tap (Phase 29): q_x_ now holds layer l's POST-MLP
        // residual-stream hidden [nq, H] — strided D2D into column l of the
        // [γ, L, H] dump staging, stream-ordered on the draft stream. One
        // null-check when the dump is off (zero copies, zero allocations).
        if (epm_stage_)
            rk.backend->memcpy_2d_async(
                static_cast<char*>(epm_stage_) +
                    static_cast<int64_t>(l) * H_ * kBf16,
                static_cast<size_t>(L_) * H_ * kBf16, q_x_,
                static_cast<size_t>(H_) * kBf16,
                static_cast<size_t>(H_) * kBf16, static_cast<size_t>(nq), s);
    }

    // Base logits for all gamma positions in ONE head GEMM (FP32 out) —
    // DRAFT-vocab space (== full vocab unless a reduced-vocab checkpoint).
    // BF16 lm_head keeps the legacy launch_output_head (bit-identical);
    // quantized lm_head goes through the fused-dequant GEMM with FP32 out
    // (lm_head has no bias — TD-DSPARK-DRAFT-QUANT).
    if (nranks == 1) {
        if (weights_.lm_head.dtype == model::DsparkWeightDtype::kBF16) {
            compute::launch_output_head(
                static_cast<float*>(base_logits_), hidden_out_,
                weights_.lm_head.ptr,
                /*bias=*/nullptr, nq, static_cast<int>(Vd_), H_,
                compute::EmbeddingDtype::kBFloat16, s);
        } else {
            weight_gemm(base_logits_, hidden_out_, weights_.lm_head, nq,
                        static_cast<int>(Vd_), H_, /*lda=*/H_, /*ldw=*/H_,
                        0, 0, /*out_fp32=*/true, s);
        }
    } else {
        // Sharded lm_head (TD-DSPARK-DRAFT-SHARD): each rank GEMMs its
        // vocab-row shard into a tight [nq, Vd_local] FP32 buffer (full-K
        // reduction — logits bit-equal to the unsharded head GIVEN the same
        // hidden), then 2D-gathers into rank 0's [nq, Vd] base_logits
        // column window; the Markov/confidence chain stays on rank 0.
        for (size_t r = 0; r < nranks; ++r) {
            ranks_[r].backend->set_device();
            void* shard_out = r == 0 ? logits_shard0_
                                     : shard_[r - 1].logits_shard;
            weight_gemm(shard_out, b_hidden(r), rank_weights(r).lm_head, nq,
                        static_cast<int>(Vd_local_), H_, /*lda=*/H_,
                        /*ldw=*/H_, 0, 0, /*out_fp32=*/true,
                        ranks_[r].stream);
            const size_t row_bytes =
                static_cast<size_t>(Vd_local_) * sizeof(float);
            auto* dst = static_cast<char*>(base_logits_) +
                        static_cast<size_t>(r) * row_bytes;
            ranks_[r].backend->memcpy_2d_async(
                dst, static_cast<size_t>(Vd_) * sizeof(float), shard_out,
                row_bytes, row_bytes, static_cast<size_t>(nq),
                ranks_[r].stream);
            if (r > 0)
                ranks_[r].backend->record_event(shard_[r - 1].ev_xfer,
                                                ranks_[r].stream);
        }
        rk.backend->set_device();
        for (size_t r = 1; r < nranks; ++r)
            rk.backend->stream_wait_event(s, shard_[r - 1].ev_xfer);
    }

    if (prof) {
        const auto t_end = std::chrono::steady_clock::now();
        std::fprintf(stderr,
                     "[dspark-prof] run_step: sync=%.3f ms enqueue=%.3f ms\n",
                     std::chrono::duration<double, std::milli>(t_sync - t_entry)
                         .count(),
                     std::chrono::duration<double, std::milli>(t_end - t_sync)
                         .count());
    }
    last_num_query_ = ndraft;
    markov_ran_ = false;  // the e stash belongs to the PREVIOUS block now
    // EPM-1: stash the block identity for epm_write_block_record.
    if (epm_stage_) {
        epm_last_seq_ = seq_id;
        epm_last_anchor_pos_ = anchor_pos;
        epm_last_anchor_token_ = anchor_token_id;
    }
    return true;
}

// ── Ticket J: V4 dflash backbone forward ────────────────────────────────────
// One RUN_DSPARK_STEP over the γ block through 3 V4-shaped SWA-only layers:
// embed(target table) → ×hc repeat → per layer {hc_pre → attn_norm →
// q_a/q_a_norm/q_b → v4_q_prep → wkv/kv_norm/rope → latent-MQA windowed
// non-causal attention (+sinks, +inverse rope) → grouped o_proj → hc_post →
// hc_pre → ffn_norm → MoE(sqrtsoftplus top-6 ×1.5, MXFP4 experts, swiglu
// clamp) + shexp → hc_post} → hc_head(output_hc) → output_norm → target
// lm_head. Math references: ref/vllm deepseek_v4/nvidia/dspark.py +
// ref/llama.cpp deepseek4.cpp (ratio-0 arm) + ticket-G o_proj scheme.

void DsparkRuntime::v4_moe_ffn(int layer, int rows, void* s) {
    const auto& lw = weights_.v4->layers[static_cast<size_t>(layer)];
    auto* dev = v4_expert_dev_.get();
    auto* be = ranks_[0].backend;
    const int E = static_cast<int>(ckpt_.v4.n_routed_experts);
    const int topk = static_cast<int>(ckpt_.v4.n_expert_used);
    const int I = static_cast<int>(ckpt_.v4.moe_intermediate);
    const int T = rows * topk;
    const float limit = static_cast<float>(ckpt_.v4.swiglu_limit);

    // Router + gating (deps ExpertKernels: sqrtsoftplus scoring, exp_probs_b
    // bias affects SELECTION only, unbiased weights renormalized to
    // routed_scaling — the exact target-side V4 gating).
    compute::launch_router_projection(
        static_cast<float*>(v4_logits_e_), q_normed_, lw.gate_inp.ptr, rows,
        E, H_, s);
    compute::TopkGatingParams gp{};
    gp.num_tokens = rows;
    gp.num_experts = E;
    gp.topk = topk;
    gp.n_group = 1;
    gp.topk_group = 1;
    gp.routed_scaling_factor = static_cast<float>(ckpt_.v4.routed_scaling);
    gp.renormalize = ckpt_.v4.norm_topk_prob;
    gp.scoring_func = compute::ScoringFunc::kSqrtSoftplus;
    compute::launch_topk_gating(
        static_cast<float*>(v4_topk_w_), static_cast<int32_t*>(v4_topk_idx_),
        static_cast<const float*>(v4_logits_e_),
        static_cast<const float*>(lw.exp_probs_b.ptr), gp, s);

    // Permute → grouped MXFP4 int GEMMs (native, no dequant) → swiglu clamp
    // → down → weighted unpermute. Mirrors the dispatch_moe GG-S1 sequence.
    dev->moe_permute(v4_perm_in_, static_cast<int32_t*>(v4_exp_off_),
                     static_cast<int32_t*>(v4_s2d_),
                     static_cast<int32_t*>(v4_perm_idx_), q_normed_,
                     static_cast<const int32_t*>(v4_topk_idx_), rows, topk,
                     H_, E, /*elem=*/2, v4_perm_ws_, s);
    auto gguf_gemm = [&](int N, int K, const void* A, void* D_out,
                         void* b_ptrs) {
        compute::GgufGroupedGemmParams p{};
        p.type = static_cast<compute::GgufQuantType>(
            static_cast<int>(model::GgufKQuantType::MXFP4));
        p.strategy = compute::GgufGemmStrategy::int_strategy;
        p.num_experts = E;
        p.N = N;
        p.K = K;
        p.total_tokens = T;
        p.A_base = A;
        p.D_base = D_out;
        p.expert_offsets = static_cast<const int32_t*>(v4_exp_off_);
        p.B_ptrs = static_cast<const void**>(b_ptrs);
        dev->gguf_grouped_gemm(p, v4_gemm_ws_,
                               static_cast<size_t>(v4_gemm_ws_bytes_), s);
    };
    gguf_gemm(I, H_, v4_perm_in_, v4_gate_out_, lw.exps_gate_ptrs);
    gguf_gemm(I, H_, v4_perm_in_, v4_up_out_, lw.exps_up_ptrs);
    const size_t Ib = static_cast<size_t>(I) * kBf16;
    be->memcpy_2d_async(v4_gu_, Ib * 2, v4_gate_out_, Ib, Ib,
                        static_cast<size_t>(T), s);
    be->memcpy_2d_async(static_cast<char*>(v4_gu_) + Ib, Ib * 2, v4_up_out_,
                        Ib, Ib, static_cast<size_t>(T), s);
    compute::FusedSwigluParams sp{};
    sp.num_tokens = T;
    sp.d = I;
    sp.swiglu_limit = limit;
    dev->fused_swiglu(v4_act_, v4_gu_, sp, /*elem=*/2, s);
    gguf_gemm(H_, I, v4_act_, v4_expert_out_, lw.exps_down_ptrs);
    dev->moe_unpermute(v4_module_out_, v4_expert_out_,
                       static_cast<const float*>(v4_topk_w_),
                       static_cast<const int32_t*>(v4_s2d_), rows, topk, H_,
                       /*elem=*/2, s);

    // Shared expert (BF16, same swiglu clamp), added into the module output.
    auto bf16_gemm = [&](void* C, const void* A, const void* W, int M, int N,
                         int K) {
        compute::launch_bf16_gemm_nt(C, A, W, M, N, K,
                                     compute::GemmInDtype::kBFloat16,
                                     compute::GemmAccOutDtype::kBFloat16, s);
    };
    bf16_gemm(v4_sh_g_, q_normed_, lw.shexp_gate.ptr, rows, I, H_);
    bf16_gemm(v4_sh_u_, q_normed_, lw.shexp_up.ptr, rows, I, H_);
    be->memcpy_2d_async(v4_sh_gu_, Ib * 2, v4_sh_g_, Ib, Ib,
                        static_cast<size_t>(rows), s);
    be->memcpy_2d_async(static_cast<char*>(v4_sh_gu_) + Ib, Ib * 2, v4_sh_u_,
                        Ib, Ib, static_cast<size_t>(rows), s);
    sp.num_tokens = rows;
    dev->fused_swiglu(v4_sh_g_, v4_sh_gu_, sp, /*elem=*/2, s);  // reuse as act
    bf16_gemm(v4_sh_out_, v4_sh_g_, lw.shexp_down.ptr, rows, H_, I);
    compute::launch_residual_add(v4_module_out_, v4_sh_out_,
                                 rows * H_, s);
}

bool DsparkRuntime::run_step_v4(uint32_t /*anchor_token_id*/,
                                uint32_t anchor_pos, int nq,
                                std::string* err) {
    auto set_err = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };
    if (!weights_.v4 || !v4_expert_dev_)
        return set_err("V4 dflash draft not armed");
    auto& rk = ranks_[0];
    void* s = rk.stream;
    const auto& wv4 = *weights_.v4;
    const int hc = ckpt_.v4.hc_mult;
    const int D = static_cast<int>(ckpt_.head_dim);
    const int R = static_cast<int>(ckpt_.v4.rope_dim);
    const int HQ = ckpt_.num_attention_heads;
    const int QL = static_cast<int>(ckpt_.v4.q_lora_rank);
    const int OG = static_cast<int>(ckpt_.v4.o_groups);
    const int OLR = static_cast<int>(ckpt_.v4.o_lora_rank);
    const int group_dim = HQ * D / OG;
    const float hc_eps = static_cast<float>(ckpt_.v4.hc_eps);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int window = static_cast<int>(ckpt_.v4.sliding_window);
    const int base = static_cast<int>(anchor_pos);

    // Query positions (device): anchor_pos + [0, nq).
    for (int k = 0; k < nq; ++k)
        host_pos_[static_cast<size_t>(k)] = base + k;
    rk.backend->memcpy_h2d_async(v4_pos_, host_pos_.data(),
                                 static_cast<size_t>(nq) * sizeof(int32_t),
                                 s);
    const auto* d_pos = static_cast<const int*>(v4_pos_);

    auto bf16_gemm = [&](void* C, const void* A, const void* W, int M, int N,
                         int K) {
        compute::launch_bf16_gemm_nt(C, A, W, M, N, K,
                                     compute::GemmInDtype::kBFloat16,
                                     compute::GemmAccOutDtype::kBFloat16, s);
    };
    // Ticket-J bring-up bisect knobs (diagnosis only; draft outputs become
    // garbage but the TARGET must stay bit-identical either way).
    static const bool skip_moe = [] {
        const char* e = std::getenv("LS_DSP4_SKIP_MOE");
        return e && *e == '1';
    }();
    static const bool skip_attn = [] {
        const char* e = std::getenv("LS_DSP4_SKIP_ATTN");
        return e && *e == '1';
    }();

    // Embed (target table) → ×hc repeat into the residual stream.
    compute::launch_embedding_lookup(
        v4_x1_, weights_.embed_tokens.ptr,
        static_cast<const int32_t*>(q_ids_), nq, static_cast<int>(V_), H_,
        compute::EmbeddingDtype::kBFloat16, s);
    compute::launch_hc_expand_repeat(q_x_, v4_x1_, nq, hc, H_, s);

    for (int l = 0; l < L_; ++l) {
        const auto& lw = wv4.layers[static_cast<size_t>(l)];

        // ── Attention module ──
        compute::launch_mhc_pre(v4_x1_, v4_post_, v4_comb_, q_x_,
                                lw.hc_attn_fn.ptr, lw.hc_attn_scale.ptr,
                                lw.hc_attn_base.ptr, eps_, hc_eps,
                                /*post_mult=*/2.0f,
                                ckpt_.v4.hc_sinkhorn_iters, nq, hc, H_, s);
        compute::launch_rmsnorm(q_normed_, v4_x1_, lw.attn_norm.ptr, eps_,
                                nq, H_, compute::NormDtype::kBFloat16, s);
        if (!skip_attn) {
        bf16_gemm(v4_qlat_, q_normed_, lw.q_a.ptr, nq, QL, H_);
        compute::launch_rmsnorm(v4_qlat_, v4_qlat_, lw.q_a_norm.ptr, eps_,
                                nq, QL, compute::NormDtype::kBFloat16, s);
        bf16_gemm(v4_qheads_, v4_qlat_, lw.q_b.ptr, nq, HQ * D, QL);
        compute::launch_v4_q_prep(v4_qn_, v4_qr_, v4_qheads_, d_pos,
                                  v4_rope_table_, eps_, nq, HQ, D, R, s);
        bf16_gemm(v4_kvlat_, q_normed_, lw.kv.ptr, nq, D, H_);
        compute::launch_rmsnorm(v4_kvlat_, v4_kvlat_, lw.kv_norm.ptr, eps_,
                                nq, D, compute::NormDtype::kBFloat16, s);
        compute::launch_dspark_v4_kv_rope(v4_blkkv_, v4_kvlat_,
                                          v4_rope_table_, base, nq, D, R, s);
        compute::launch_dspark_v4_attention(
            v4_attn_, static_cast<float*>(v4_lse_), v4_qn_, v4_qr_,
            base > 0 ? k_base(0, l) : nullptr, v4_blkkv_, nq,
            /*n_ctx=*/base, base, window, HQ, D, R, scale, s);
        compute::launch_v4_attn_sinks(v4_attn_, v4_lse_, lw.sinks.ptr,
                                      /*head_offset=*/0, nq, HQ, D, s);
        compute::launch_v4_out_inverse_rope(v4_attn_, d_pos, v4_rope_table_,
                                            nq, HQ, D, R, s);
        // Grouped o_proj (ticket-G 2-stage scheme, BF16 weights).
        compute::launch_bf16_strided_batched_gemm_nt(
            v4_oa_, lw.o_a.ptr, v4_attn_, /*m=*/OLR, /*n=*/nq,
            /*k=*/group_dim, /*lda=*/group_dim,
            /*strideA=*/static_cast<int64_t>(OLR) * group_dim,
            /*ldb=*/HQ * D, /*strideB=*/group_dim, /*ldc=*/OG * OLR,
            /*strideC=*/OLR, /*batch=*/OG, s);
        bf16_gemm(v4_module_out_, v4_oa_, lw.o_b.ptr, nq, H_, OG * OLR);
        }  // !skip_attn (bisect: module_out holds the previous content)
        compute::launch_mhc_post(q_x_, v4_module_out_, q_x_, v4_post_,
                                 v4_comb_, nq, hc, H_, s);

        // ── FFN module (MoE + shexp) ──
        compute::launch_mhc_pre(v4_x1_, v4_post_, v4_comb_, q_x_,
                                lw.hc_ffn_fn.ptr, lw.hc_ffn_scale.ptr,
                                lw.hc_ffn_base.ptr, eps_, hc_eps, 2.0f,
                                ckpt_.v4.hc_sinkhorn_iters, nq, hc, H_, s);
        compute::launch_rmsnorm(q_normed_, v4_x1_, lw.ffn_norm.ptr, eps_,
                                nq, H_, compute::NormDtype::kBFloat16, s);
        if (!skip_moe) v4_moe_ffn(l, nq, s);
        compute::launch_mhc_post(q_x_, v4_module_out_, q_x_, v4_post_,
                                 v4_comb_, nq, hc, H_, s);
    }

    // hc_head collapse → final norm → target lm_head. The confidence head
    // consumes the POST-final-norm hidden (DeepSpec qwen3 modeling.py:386 —
    // the model output IS norm(hidden); conf + lm_head read the same rows).
    compute::launch_mhc_head(v4_x1_, q_x_, wv4.output_hc_fn.ptr,
                             wv4.output_hc_scale.ptr, wv4.output_hc_base.ptr,
                             eps_, hc_eps, nq, hc, H_, s);
    compute::launch_rmsnorm(hidden_out_, v4_x1_, weights_.final_norm.ptr,
                            eps_, nq, H_, compute::NormDtype::kBFloat16, s);
    compute::launch_output_head(
        static_cast<float*>(base_logits_), hidden_out_,
        weights_.lm_head.ptr, /*bias=*/nullptr, nq, static_cast<int>(Vd_),
        H_, compute::EmbeddingDtype::kBFloat16, s);
    return true;
}

// ── EPM-1 block record dump (Phase 29) ──────────────────────────────────────
// D2H of the per-layer hidden staging + draft ids + (optional) c_k into the
// NUMA-local pinned host buffer, one device sync (dump-ON only — timing is
// irrelevant for collection), then an EPMB append. The record key is
// (seq_id, anchor_pos) — the manifest joins on it; block_idx is a per-seq
// convenience counter.

void DsparkRuntime::epm_write_block_record(bool conf_valid) {
    if (!epm_writer_ || !epm_stage_ || !epm_host_ || last_num_query_ <= 0)
        return;
    auto& rk = ranks_[0];
    rk.backend->set_device();
    void* s = rk.stream;
    const int nq = last_num_query_;
    auto* h = static_cast<char*>(epm_host_);
    const size_t hid_bytes =
        static_cast<size_t>(nq) * L_ * H_ * static_cast<size_t>(kBf16);
    rk.backend->memcpy_d2h_async(h, epm_stage_, hid_bytes, s);
    rk.backend->memcpy_d2h_async(h + epm_hid_cap_bytes_, draft_ids_,
                                 static_cast<size_t>(nq) * sizeof(int32_t),
                                 s);
    const bool conf = conf_valid && conf_out_ != nullptr;
    if (conf)
        rk.backend->memcpy_d2h_async(
            h + epm_hid_cap_bytes_ + 64, conf_out_,
            static_cast<size_t>(nq) * sizeof(float), s);
    rk.backend->synchronize_device();

    if (epm_last_seq_ != epm_ctr_seq_) {
        epm_ctr_seq_ = epm_last_seq_;
        epm_next_block_idx_ = 0;
    }
    epm_writer_->write_block(
        epm_last_seq_, epm_next_block_idx_++, epm_last_anchor_pos_,
        epm_last_anchor_token_, nq, L_, H_,
        reinterpret_cast<const uint16_t*>(h),
        reinterpret_cast<const int32_t*>(h + epm_hid_cap_bytes_),
        conf ? reinterpret_cast<const float*>(h + epm_hid_cap_bytes_ + 64)
             : nullptr);
}

// ── Sequential Markov head (DSP-4, INV-DSPARK-MARKOV) ───────────────────────
// Vanilla low-rank transition bias, per the checkpoint semantics (DeepSpec
// VanillaMarkov / vLLM _sample_sequential):
//   e = markov_w1[x_{k-1}]; logits_k = base_logits[k] + markov_w2 @ e;
//   x_k = argmax(logits_k)  (greedy — the checkpoint's proposal method).
// The gamma-step serial chain runs entirely on the draft stream: the
// finalize kernel of step k writes x_k on-device AND gathers markov_w1[x_k]
// as the next step's e — no host round-trip.

bool DsparkRuntime::run_markov_head(std::string* err) {
    auto set_err = [&](const std::string& m) {
        if (err) *err = "dspark markov_head: " + m;
        return false;
    };
    if (last_num_query_ <= 0)
        return set_err("no run_step outputs to sample from");
    if (ckpt_.markov_head_type != "vanilla")
        return set_err("checkpoint markov_head_type '" +
                       ckpt_.markov_head_type +
                       "' not implemented — only the vanilla low-rank head "
                       "ships (gated/rnn: TD-DSPARK-HEAD-VARIANTS)");
    const int r = ckpt_.markov_rank;
    if (r <= 0 || r > 8192)
        return set_err("markov_rank " + std::to_string(r) +
                       " outside supported range (0, 8192]");

    auto& rk = ranks_[0];
    rk.backend->set_device();
    void* s = rk.stream;
    const int nq = last_num_query_;
    const int nblocks = compute::dspark_markov_num_blocks(Vd_);

    // Per-step e-chain STASH: slot k of markov_e_ [block_size, r] holds
    // markov_w1[x_{k-1}] — step k's bias reads slot k, its finalize gathers
    // slot k+1.  Keeping every step's e (instead of one rolling buffer)
    // costs (block_size-1)*r BF16 and hands DSP-6's confidence head the
    // whole [nq, r] prev-embed matrix for ONE batched kernel.
    auto e_slot = [&](int k) -> void* {
        return static_cast<char*>(markov_e_) +
               static_cast<int64_t>(k) * r * kBf16;
    };

    // e_0 = markov_w1[anchor]: step 0's previous token is the ANCHOR (the
    // last accepted real token — a TARGET-vocab id; markov_w1 is target-
    // vocab), still device-resident at q_ids_[0] from run_step.
    compute::launch_embedding_lookup(
        e_slot(0), weights_.markov_w1.ptr,
        static_cast<const int32_t*>(q_ids_), /*num_tokens=*/1,
        static_cast<int>(V_), r, compute::EmbeddingDtype::kBFloat16, s);

    auto* ids = static_cast<int32_t*>(draft_ids_);
    for (int k = 0; k < nq; ++k) {
        // Bias + argmax run in DRAFT-vocab space (base logits + markov_w2
        // are draft-sized); the finalize maps the draft argmax to a TARGET
        // id via d2t (identity/nullptr for full-vocab checkpoints) before
        // writing ids[k] and gathering the target-vocab markov_w1 row.
        // Bonus-anchor layout: draft slot k's base logits live at physical
        // row k + 1 (row 0 = the discarded bonus-anchor row).
        const auto* base_k = static_cast<const float*>(base_logits_) +
                             static_cast<int64_t>(k + 1) * Vd_;
        auto* corr_k = static_cast<float*>(corrected_logits_) +
                       static_cast<int64_t>(k) * Vd_;
        compute::launch_dspark_markov_bias_argmax(
            corr_k, base_k, weights_.markov_w2.ptr, e_slot(k),
            markov_partials_, static_cast<int>(Vd_), r, s);
        // Writes x_k (target id) and (unless last step) gathers
        // markov_w1[x_k] into slot k+1 for the next step — the serial
        // dependency stays on-device.
        compute::launch_dspark_markov_finalize(
            ids + k, k + 1 < nq ? e_slot(k + 1) : nullptr, markov_partials_,
            nblocks, weights_.markov_w1.ptr, r, weights_.d2t.ptr, s);
    }
    markov_ran_ = true;
    return true;
}

const int32_t* DsparkRuntime::draft_tokens() const {
    return static_cast<const int32_t*>(draft_ids_);
}

const float* DsparkRuntime::corrected_logits() const {
    return static_cast<const float*>(corrected_logits_);
}

// ── Trained confidence head (DSP-6, INV-DSPARK-CONF) ────────────────────────
// c_k = sigmoid(proj · [hidden_k ; markov_w1[x_{k-1}]] + bias), the DeepSpec
// AcceptRatePredictor over the hidden-FIRST concat (qwen3/modeling.py
// predict_confidence_step / eval draft_ops.py _predict_confidence_logits:
// prev_token_ids = [anchor, sampled[:-1]] — exactly the DSP-4 e-chain).
// TRAINED cumprod-composable survival probability — never the output-head
// {top1_prob, entropy} heuristic (compute/kernels/confidence/, IPC-8g).

bool DsparkRuntime::run_confidence_head(std::string* err) {
    auto set_err = [&](const std::string& m) {
        if (err) *err = "dspark confidence_head: " + m;
        return false;
    };
    if (!has_confidence_head())
        return set_err("checkpoint has no confidence head "
                       "(enable_confidence_head=false) — "
                       "speculation.dspark.confidence_enabled requires a "
                       "confidence-trained checkpoint");
    if (last_num_query_ <= 0)
        return set_err("no run_step outputs to score");
    const bool with_markov = ckpt_.confidence_head_with_markov;
    if (with_markov && !markov_ran_)
        return set_err("with_markov head needs the DSP-4 e-chain — call "
                       "run_markov_head first");

    auto& rk = ranks_[0];
    rk.backend->set_device();
    // ONE kernel over the gamma positions, stream-ordered after the Markov
    // chain (whose finalize kernels populated the e stash).
    // Bonus-anchor layout: draft hiddens start at physical row 1.
    compute::launch_dspark_confidence(
        static_cast<float*>(conf_out_),
        static_cast<const char*>(hidden_out_) +
            static_cast<int64_t>(H_) * kBf16,
        with_markov ? markov_e_ : nullptr,
        weights_.confidence_proj_weight.ptr,
        weights_.confidence_proj_bias.ptr, last_num_query_, H_,
        with_markov ? ckpt_.markov_rank : 0, rk.stream);
    return true;
}

const float* DsparkRuntime::confidence() const {
    return static_cast<const float*>(conf_out_);
}

}  // namespace layerstorm::speculation

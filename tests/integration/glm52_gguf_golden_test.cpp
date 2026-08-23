// GLM-25 Phase I: end-to-end golden validation for GLM-5.2 (436 GB GGUF).
//
// Loads the GLM-5.2 Q4_K_XL split GGUF (glm-dsa: MLA-256 + DSA lightning
// indexer + IndexShare + MoE 256-expert top-8, 78 layers) via the production
// engine, teacher-forces "The capital of France is" token by token, and
// checks the greedy next token matches llama.cpp on the identical GGUF.
//
// Validates the full GLM-5.2 stack (split-GGUF loader, MLA-256, MoE 256-way
// gating, production FETCH_AND_RUN seam) for token parity with llama.cpp.
// DSA NOTE: at TP=2 the lightning-indexer producer is gated OFF
// (TD-GLM-INDEXER-DCP: dcp_size==1 only) → attention runs dense, which is
// numerically identical at this context length (top-k would select every
// causal position). The DSA-exercising variant of this golden lands when
// the DCP gate lifts (task: per-rank topk_lengths + dcp_indexer_mode).
//
// MoE runs on the PRODUCTION seam: fused-gate routing export +
// E_CMD_FETCH_AND_RUN_MOE fetches only the routed ≤8 experts per layer
// (~13 GB/token worst case vs 420 GB for whole-layer prefetch).
//
// Run (both RTX 5090s, TP=2/EP=2 keeper shape):
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
//     ./build/tests/integration/glm52_gguf_golden_test
// Strategy override: GG_STRATEGY=int|dequant (default: int only; run dequant
// explicitly when wanted).
//
// Skips gracefully if the GGUF or an SM120+ GPU with enough VRAM is absent.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "daemon/engine.h"
#include "daemon/buffer_registry.h"
#include "daemon/ipc_protocol.h"
#include "core/device_backend.h"         // DSP-3: draft GPU id readback
#include "speculation/dspark_runtime.h"  // DSP-3: aux-export golden
#include "speculation/dspark_speculation_method.h"  // DSP-5: lossless golden
#include "speculation/mtp_speculation_method.h"

namespace lipc = layerstorm::ipc;
namespace ldam = layerstorm::daemon;
namespace fs   = std::filesystem;

// ── Golden constants (llama.cpp ground truth on THIS GGUF, 2026-07-05) ──────
//
// llama-completion (build a410713) -no-cnv raw mode:
//   -m GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf -p "The capital of France is" \
//   -n 1 --temp 0 --top-k 1 --verbose-prompt --seed 42
// --verbose-prompt reported EXACTLY 5 tokens (raw glm-dsa adds NO BOS —
// unlike the GLM-4.7 golden's [gMASK]<sop> prefix) and greedily completed
// " Paris" (id 12089, verified against test-data/GLM-5.2/tokenizer.json).
namespace {
const std::vector<uint32_t> kPromptTokens = {785, 6722, 315, 9621, 374};
constexpr uint32_t kGoldenNext = 12089;  // " Paris"

const char* kConfigRel = "/test-data/config/glm_5_2_gguf.json";
const char* kGgufRel =
    "/test-data/GLM-5.2-GGUF-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf";
}  // namespace

// ── GPU probe ───────────────────────────────────────────────────────────────

// Counts visible SM120+ devices with >= min_gb total VRAM (TP=2 needs two).
static int count_big_sm120(double min_gb) {
    int count = 0, big = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    for (int i = 0; i < count; ++i) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, i)
                != cudaSuccess || major < 12)
            continue;
        size_t freeb = 0, totalb = 0;
        if (cudaSetDevice(i) != cudaSuccess) continue;
        if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) continue;
        if (static_cast<double>(totalb) / (1024.0 * 1024 * 1024) >= min_gb)
            ++big;
    }
    return big;
}

// ── Fixture ──────────────────────────────────────────────────────────────────

class Glm52GgufGolden : public ::testing::Test {
protected:
    void start_engine(const std::string& strategy) {
        const std::string src = LAYERSTORM_SOURCE_DIR;
        const std::string cfg_path = src + kConfigRel;
        const std::string gguf_path = src + kGgufRel;
        ASSERT_TRUE(fs::exists(cfg_path)) << "missing config " << cfg_path;
        ASSERT_TRUE(fs::exists(gguf_path)) << "missing GGUF " << gguf_path;

        std::ifstream f(cfg_path);
        nlohmann::json j = nlohmann::json::parse(f);
        j["model"]["weights_path"] = gguf_path;
        j["quantization"]["gguf_strategy"] = strategy;
        // TP=2 / EP=2 on the two visible 5090s (keeper benchmark shape,
        // spec/BENCHMARK_FULLFIT.md): GLM-5.2's pinned attention does not
        // co-fit with an expert cache on ONE 30 GiB card — TP=2 halves the
        // pinned share per GPU and EP spreads the expert cache. 30 GiB
        // usable per card (a full 31 GiB contiguous block fails after
        // context overhead).
        j["hardware"]["gpus"] = nlohmann::json::array(
            {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 30},
              {"pcie_gen", 5}, {"pcie_width", 16}},
             {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 30},
              {"pcie_gen", 5}, {"pcie_width", 16}}});
        j["hardware"]["tp_array"] = nlohmann::json::array({0, 1});
        j["parallelism"]["tensor_parallelism"] = 2;
        // GLM52_KV_SHARDED=1: sequence-sharded KV (KVS-2/-3, Helix round-robin
        // chunks) — each rank holds only its token shard and the DCP LSE
        // combine (allgather-LSE → lse-correct → output-allreduce) merges the
        // per-rank partials. At this 5-token prompt with dcp_chunk_size=16,
        // rank 1 owns NOTHING — the run exercises the empty-shard path
        // (INV-KVS-EMPTY: lse=+inf → combine weight 0) end to end.
        // ⚠ NOT YET NUMERICALLY VALID: the executor lacks the Q-head
        // allgather (TD-KVS-Q-ALLGATHER / INV-KVS-QAG) — the combine mixes
        // head shards (observed 2026-07-06: argmax 12089 matched but top1
        // inflated 0.4693 → 0.9958 = half the heads garbage). This switch is
        // the harness for that fix; once it lands the greedy token AND top1
        // must match the replicated default.
        if (const char* ks = std::getenv("GLM52_KV_SHARDED"); ks && *ks == '1')
            j["hardware"]["dcp_kv_mode"] = "sharded";

        // GLM52_TQ=1 (TD-KVT-TQ-GOLDEN): TurboQuant MLA 4-bit attention
        // backend. TQ is lossy (packed 4-bit c_kv + FP16 norm + BF16 rope),
        // but at this 5-token parametric prompt the quantization error is
        // far below the golden's margin: the greedy token must stay 12089
        // (TQ smoke gate — the long-context/pruning regime is the long
        // golden's job, glm52_longctx_golden_test GLM52_TQ=1).
        if (const char* tq = std::getenv("GLM52_TQ"); tq && *tq == '1')
            j["compute"]["attention_backend"] = "turboquant_mla";

        // GLM-25k: GLM52_KV_TIERING=<hot_buffer_slots> enables DSA-guided KV
        // tiering with the given per-layer hot buffer (small value = force
        // demotion + cold fetches). Placement-only (INV-KVT-1): the greedy
        // token AND top1 must match the non-tiered golden EXACTLY.
        if (const char* kt = std::getenv("GLM52_KV_TIERING"); kt && *kt) {
            j["memory"]["kv_tiering"] = {
                {"enabled", true},
                {"hot_buffer_slots", std::atoi(kt)},
                {"host_to_device_ratio", 16.0}};
        }

        // GLM52_MTP=1 (#16 / GLM-25g): construct the MtpSpeculationMethod at
        // engine init (speculation.method=mtp).  Decode itself is UNCHANGED
        // (the method only drafts when driven) — the MtpLossless test drives
        // the full draft → verify → accept loop and asserts token identity
        // with plain autoregressive decode (INV-MTP-LOSSLESS).
        if (const char* sm = std::getenv("GLM52_MTP"); sm && *sm == '1') {
            j["speculation"]["method"] = "mtp";
            j["speculation"]["enabled"] = true;      // preset ships it off
            j["speculation"]["mtp"]["enabled"] = true;
        }

        // GLM52_DSPARK_EXPORT=1 (DSP-3): arm the DSpark aux-hidden export —
        // a third (draft) GPU joins the device set (run with
        // CUDA_VISIBLE_DEVICES=2,3,0 so visible ordinal 2 is a 5080), the
        // 7.61 GB draft checkpoint uploads there, and every target attention
        // dispatch of layers {8,23,39,55,70} exports the layer-input hidden
        // to the draft GPU (INV-DSPARK-AUX).  The TARGET must be unchanged:
        // the greedy token stays 12089 (the hook is read-only), and experts
        // stay EP-split across the TWO TP GPUs only (ep_gpus_ below — the
        // draft GPU never hosts experts, keeping placement identical to the
        // 2-GPU baseline).
        // GLM52_DSPARK=1 (DSP-5) arms the same 3-GPU dspark config for the
        // DsparkLossless golden (full draft → verify → accept loop through
        // DsparkSpeculationMethod).
        const char* dspark_export = std::getenv("GLM52_DSPARK_EXPORT");
        const char* dspark_loop   = std::getenv("GLM52_DSPARK");
        // GLM52_DSPARK_REALISTIC=1 (TD-DSPARK-ACCEPT-SHORTPROMPT / DSP-11b):
        // same 3-GPU dspark config, driven by the DsparkAcceptRealistic
        // acceptance measurement over a long realistic prompt.
        const char* dspark_real   = std::getenv("GLM52_DSPARK_REALISTIC");
        const bool dspark_loop_on =
            (dspark_loop && *dspark_loop == '1') ||
            (dspark_real && *dspark_real == '1');
        if ((dspark_export && *dspark_export == '1') || dspark_loop_on) {
            j["hardware"]["gpus"].push_back(
                {{"id", 2}, {"type", "rtx5080"}, {"vram_gb", 15},
                 {"pcie_gen", 5}, {"pcie_width", 16}});
            j["speculation"]["method"] = "dspark";
            j["speculation"]["enabled"] = true;
            // GLM52_DSPARK_CKPT=<dir>: checkpoint override for A/B runs
            // across speculator checkpoints (e.g. /srv/models/dspark/
            // GLM-5.2-speculator.dspark.orig vs glm-5.2-dspark-preview)
            // without re-pointing the test-data symlink.
            std::string dspark_ckpt =
                src + "/test-data/GLM-5.2-speculator.dspark";
            if (const char* ck = std::getenv("GLM52_DSPARK_CKPT"); ck && *ck)
                dspark_ckpt = ck;
            j["speculation"]["dspark"]["checkpoint_path"] = dspark_ckpt;
            j["speculation"]["dspark"]["draft_gpus"] =
                nlohmann::json::array({2});
            // Model-agnostic γ (mirrors the 03a1f755 unit-test fix): the
            // loader HARD cross-validates config block_size/speculative_
            // tokens against the checkpoint, and the symlinked model varies
            // (shipped ckpt 8/7, glm-5.2-dspark-preview 16/15) — derive
            // both from the checkpoint's config.json so engine init matches
            // whatever is symlinked.
            {
                std::ifstream cf(dspark_ckpt + "/config.json");
                if (cf.is_open()) {
                    auto cj = nlohmann::json::parse(cf, nullptr,
                                                    /*allow_exceptions=*/false);
                    if (!cj.is_discarded()) {
                        if (cj.contains("block_size"))
                            j["speculation"]["dspark"]["block_size"] =
                                cj["block_size"];
                        if (cj.contains("speculators_config") &&
                            cj["speculators_config"].contains(
                                "proposal_methods") &&
                            !cj["speculators_config"]["proposal_methods"]
                                 .empty())
                            j["speculation"]["dspark"]["speculative_tokens"] =
                                cj["speculators_config"]["proposal_methods"]
                                  [0]["speculative_tokens"];
                    }
                }
            }
            // TD-DSPARK-PREFILL-CAP e2e: GLM52_DSPARK_AUX_ROWS shrinks the
            // aux capture staging so a prefill chunk (<= the engine's
            // dispatch_attention batch cap of 64) EXCEEDS it and exercises
            // the CHUNKED capture/ingest path on the real 3-GPU engine
            // (the default 2048-row staging cannot be crossed through this
            // harness).  Drafting must stay armed either way.
            if (const char* ar = std::getenv("GLM52_DSPARK_AUX_ROWS"))
                if (int v = std::atoi(ar); v >= 16)
                    j["speculation"]["dspark"]["aux_capture_max_rows"] = v;
            // DSP-6: the lossless loop also exercises the trained
            // confidence head (the checkpoint ships it) — the golden LOGS
            // c_k + cumulative survival per round (informative: raw c_k is
            // uncalibrated until DSP-7).  The export path stays DSP-3-pure.
            if (dspark_loop_on)
                j["speculation"]["dspark"]["confidence_enabled"] = true;
        }

        // GLM52_SPARSE_PREFILL=1 (TD-SPARSE-CHUNK-PREFILL): DSA sparse CHUNK
        // PREFILL — combine with GLM52_PREFILL=1 GLM52_PREFILL_CHUNK=4. At
        // this 5-token prompt every causal position fits in the top-k
        // (index_topk=2048), so sparse ≡ dense: the greedy token must stay
        // 12089 (sanity gate — the pruning regime is the long golden's job).
        if (const char* sp = std::getenv("GLM52_SPARSE_PREFILL");
            sp && *sp == '1')
            j["compute"]["dsa_sparse_prefill"] = true;

        j["memory"]["vram_safety_margin_gb"] = 3.0;
        // The golden needs ~10 tokens; shrink KV growth to 1 page/step.
        // 78 layers × a few logical pages → 2048 physical pages is ample.
        // EXCEPT under GLM52_DSPARK_REALISTIC: that prompt is thousands of
        // tokens — keep the preset's "auto" page budget + default growth
        // (same rationale as glm52_longctx_golden_test).
        if (!(dspark_real && *dspark_real == '1')) {
            j["memory"]["kv_cache"]["max_pages_per_gpu"] = 2048;
            j["memory"]["kv_cache"]["page_growth_chunk_tokens"] = 16;
        }
        // Unlike GLM-4.7 (all experts fit → 0.95 stable), GLM-5.2's routed set
        // is ~420 GB — keep the preset's rolling-LRU zoning; roughly two
        // layers' expert sets (2 × 5.5 GB) stay resident in the ~10 GB zone
        // left after ~16 GB of pinned Q8 attention + embed/head.

        // GLM52_PREPACKED=1: KEEPER PARITY expert source (BENCHMARK_FULLFIT.md,
        // GG-10 prepacked set) — same block as glm52_longctx_golden_test.cpp.
        // Validates the prepacked per-layer-gguf-type runtime path against the
        // short golden (the default GGUF-mmap path stays the baseline).
        if (const char* pp = std::getenv("GLM52_PREPACKED"); pp && *pp == '1') {
            const std::string prepacked = src + "/test-data/GLM-5.2-prepacked";
            if (fs::exists(prepacked + "/manifest.json")) {
                // GLM52_ARENA_FRACTION=<f>: same override as the longctx
                // twin — 0.6 leaves ~26 GB/node slack when another job
                // shares the box (0.76 got this golden OOM-killed on
                // CONSTRAINT_MEMORY_POLICY next to a concurrent sweep).
                double arena_frac = 0.76;
                if (const char* af = std::getenv("GLM52_ARENA_FRACTION");
                    af && *af)
                    if (double v = std::atof(af); v > 0.0 && v <= 1.0)
                        arena_frac = v;
                j["preprocessing"]["prepacked_dir"] = prepacked;
                j["memory"]["preload_expert_buffers"] = true;
                j["memory"]["pin_host_expert_pool"] = true;
                j["memory"]["pin_host_expert_pool_direct_load"] = true;
                j["memory"]["arena_attach"] = {
                    // P-24b: tests default to a process-PRIVATE arena so dev test runs
                    // never attach to / wipe / hold the box's persistent holder store
                    // (configs differ in geometry, so each run would thrash it).
                    // GLM52_ARENA_ATTACH=1 opts the persistence gates back in.
                    {"enabled", [] {
                        const char* aa = std::getenv("GLM52_ARENA_ATTACH");
                        return aa && *aa == '1';
                    }()}};
                j["memory"]["pin_host_expert_pool_preload"] = true;
                j["memory"]["pin_host_expert_pool_direct_o_direct"] = true;
                j["memory"]["pin_host_expert_pool_sizing"] = {
                    {"mode", "fraction_total"}, {"value", arena_frac}};
                j["memory"]["cross_node_spill"] = {
                    {"enabled", true},
                    {"nodes", nlohmann::json::array({
                        {{"node", 0}, {"weight", 1}},
                        {{"node", 1}, {"weight", 1}},
                    })},
                    {"sizing_mode", "fraction_total"},
                    {"sizing_value", arena_frac},
                };
            }
        }

        // Expert placement stays EP-split across the TP GPUs only — a
        // dspark draft GPU appended above must never receive experts (its
        // VRAM carries the draft, and adding it would change EP placement
        // vs the 2-GPU golden baseline).
        ep_gpus_ = static_cast<int>(j["hardware"]["tp_array"].size());

        config_path_ = "/tmp/glm52_golden_config_" + strategy + ".json";
        { std::ofstream o(config_path_); o << j.dump(2); }

        vocab_size_ = j["model"]["vocab_size"].get<int>();
        first_moe_layer_ = j["model"].value("first_k_dense_replace", 3);

        auto backends = ldam::default_backends();
        backends.skip_cuda_graphs = true;
        engine_ = std::make_unique<ldam::Engine>(config_path_,
                                                 std::move(backends));

        auto& info = engine_->info();
        auto* base = reinterpret_cast<uint8_t*>(info.ipc_base);
        cmd_ring_ = std::make_unique<lipc::CommandRing>(
            base + info.cmd_ring_offset);
        cmp_ring_ = std::make_unique<lipc::CompletionRing>(
            base + info.cmp_ring_offset);
        sideband_ = base + info.sideband_offset;
        num_layers_ = info.num_layers;
        num_experts_ = info.num_experts;

        auto* reg = engine_->buffer_registry();
        hidden_buf_id_ = find_buf(reg, "hidden_state.attn.rank0");
        logits_buf_id_ = find_buf(reg, "logits_scratch.pos0");
        ASSERT_NE(hidden_buf_id_, 0u);
        ASSERT_NE(logits_buf_id_, 0u);
    }

    void stop_engine() {
        if (engine_) { engine_->shutdown(); engine_.reset(); }
        if (!config_path_.empty()) std::remove(config_path_.c_str());
    }

    void TearDown() override { stop_engine(); }

    static uint32_t find_buf(const ldam::BufferRegistry* reg,
                             const std::string& prefix) {
        for (const auto& [id, name] : reg->all_named_entries())
            if (name.rfind(prefix, 0) == 0) return id;
        return 0;
    }

    lipc::Command make_cmd(lipc::CmdType type, uint32_t gpu = 0) {
        lipc::Command c{};
        c.cmd_type = static_cast<uint32_t>(type);
        c.cmd_seq = cmd_seq_++;
        c.gpu_idx = gpu;
        c.stream_id = 0;
        return c;
    }
    void send(const lipc::Command& c) {
        ASSERT_TRUE(cmd_ring_->try_write(&c)) << "cmd ring full";
    }
    bool wait(lipc::Completion& out, uint32_t expected, int timeout_s = 300) {
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::seconds(timeout_s);
        while (std::chrono::steady_clock::now() < deadline) {
            if (cmp_ring_->try_read(&out)) {
                if (out.cmp_type == static_cast<uint32_t>(lipc::CMP_ERROR)) {
                    ADD_FAILURE() << "CMP_ERROR: " << out.error.message;
                    return false;
                }
                if (out.cmp_type == static_cast<uint32_t>(lipc::CMP_CHECKPOINT))
                    continue;
                if (out.cmp_type == expected) return true;
            }
            std::this_thread::yield();
        }
        ADD_FAILURE() << "timeout waiting for cmp 0x" << std::hex << expected;
        return false;
    }

    void create_sequence(uint64_t seq_id, uint32_t prompt_len) {
        lipc::Completion cmp{};
        auto c = make_cmd(lipc::CMD_SEQ_CREATE);
        c.seq_create.seq_id = seq_id;
        c.seq_create.prompt_len = prompt_len;
        c.seq_create.pool = 0;
        send(c);
        ASSERT_TRUE(wait(cmp, lipc::CMP_SEQ_OP_DONE));
        ASSERT_EQ(cmp.status, 0u);
    }
    void free_sequence(uint64_t seq_id) {
        lipc::Completion cmp{};
        auto c = make_cmd(lipc::CMD_SEQ_FREE);
        c.seq_free.seq_id = seq_id;
        send(c);
        wait(cmp, lipc::CMP_SEQ_OP_DONE);
    }

    // One teacher-forced step on the PRODUCTION MoE seam (GLM-25 golden runs
    // the path production uses): attention with the fused router gate
    // (emit_gating + store_gating exports the routed top-K to the sideband),
    // then E_CMD_FETCH_AND_RUN_MOE fetches ONLY the routed ≤8 experts
    // (~170 MB/layer vs 5.5 GB for the full 256) and runs the MoE — the exact
    // decode_step_fetch_and_run seam validated by first_token_test.
    uint32_t decode_step(uint32_t input_token, uint64_t seq_id,
                         uint32_t token_pos, float* top1 = nullptr) {
        lipc::Completion cmp{};
        auto* token_ids = reinterpret_cast<uint32_t*>(
            sideband_ + lipc::IpcLayout::kTokenIdsOff);
        token_ids[0] = input_token;

        auto embed = make_cmd(lipc::CMD_EMBEDDING_LOOKUP);
        embed.embedding_lookup.num_tokens = 1;
        embed.embedding_lookup.output_buf_id = hidden_buf_id_;
        send(embed);
        EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE));

        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        batch[0].seq_id = seq_id;
        batch[0].token_pos = token_pos;
        batch[0]._pad = 0;

        for (int layer = 0; layer < num_layers_; ++layer) {
            const bool is_moe = layer >= first_moe_layer_;

            auto a = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
            a.run_attention.layer_idx = static_cast<uint32_t>(layer);
            a.run_attention.num_seqs = 1;
            a.run_attention.is_prefill = 0;
            a.run_attention.use_graph = 0;
            a.run_attention.emit_gating  = is_moe ? 1 : 0;
            a.run_attention.store_gating = is_moe ? 1 : 0;
            send(a);
            EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "attn L" << layer;
            if (::testing::Test::HasFailure()) return 0;

            if (is_moe) {
                // Routed top-K exported by attention's fused gate.
                const auto* hdr =
                    reinterpret_cast<const lipc::RoutingExportHeader*>(
                        sideband_ + lipc::IpcLayout::kRoutingExportOff);
                EXPECT_EQ(hdr->num_tokens, 1u);
                EXPECT_EQ(hdr->layer_idx, static_cast<uint32_t>(layer))
                    << "routing-export layer mismatch";
                const auto* ridx = reinterpret_cast<const int32_t*>(
                    sideband_ + lipc::IpcLayout::kRoutingExportIndicesOff);
                const uint32_t rn = hdr->num_tokens * hdr->topk;

                auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
                    sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
                const int tp = ep_gpus_;  // EP across TP GPUs only (dspark draft GPU excluded)
                uint32_t count = 0;
                for (uint32_t k = 0; k < rn; ++k) {
                    if (ridx[k] < 0) continue;
                    entries[count].layer_idx  = static_cast<uint32_t>(layer);
                    entries[count].expert_idx = static_cast<uint16_t>(ridx[k]);
                    entries[count].zone       = 0;
                    // EP split: each expert fetched/run on its owning GPU.
                    entries[count].gpu_idx =
                        static_cast<uint8_t>(ridx[k] % tp);
                    ++count;
                }
                EXPECT_GT(count, 0u) << "no routed experts exported L" << layer;
                if (count == 0) return 0;

                auto m = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
                m.fetch_and_run_moe.layer_idx    = static_cast<uint32_t>(layer);
                m.fetch_and_run_moe.num_seqs     = 1;
                m.fetch_and_run_moe.expert_count = count;
                // Correctness gate: the routed experts MUST be resident before
                // compute finalizes (a missing expert degrades like a timeout →
                // wrong MoE). Cold reads come from the Gen3-capped NVMe
                // (~170 MB ≈ 60 ms warm cache; allow a generous bound).
                m.fetch_and_run_moe.timeout_us   = 120000000;  // 120 s
                m.fetch_and_run_moe.moe_mode     = 0;
                // F-6 decider fields stay zero-init → fetch every routed entry.
                send(m);
            } else {
                auto m = make_cmd(lipc::D_B_CMD_RUN_MOE);
                m.run_moe.layer_idx = static_cast<uint32_t>(layer);
                m.run_moe.num_seqs = 1;
                m.run_moe.moe_mode = 0;
                m.run_moe.apply_residual_correction = 0;
                m.run_moe.store_gating_output = 0;
                m.run_moe.emit_checkpoint = 0;
                send(m);
            }
            EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE)) << "moe L" << layer;
            if (::testing::Test::HasFailure()) return 0;
        }

        auto head = make_cmd(lipc::CMD_OUTPUT_HEAD);
        head.output_head.num_tokens = 1;
        head.output_head.input_buf_id = hidden_buf_id_;
        head.output_head.output_buf_id = logits_buf_id_;
        head.output_head.compute_confidence = 1;
        send(head);
        EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE));
        if (top1) *top1 = cmp.compute.top1_prob;

        auto s = make_cmd(lipc::CMD_SAMPLE_TOKENS);
        s.sample_tokens.num_tokens = 1;
        s.sample_tokens.logits_buf_id = logits_buf_id_;
        s.sample_tokens.vocab_size = static_cast<uint32_t>(vocab_size_);
        s.sample_tokens.temperature = 0.0f;  // argmax
        s.sample_tokens.top_p = 1.0f;
        s.sample_tokens.top_k = 0;
        s.sample_tokens.random_seed = 42;
        send(s);
        EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE));
        return token_ids[0];
    }

    // #16 / GLM-25g: one MTP draft step on the PRODUCTION seam.
    // Precondition: the hidden-state attn buffers hold prev_hidden — the
    // trunk hidden right after a decode_step (draft step 0 / catch-up) or
    // the MTP layer output right after a previous mtp_step (chained draft).
    // Pipeline: D_CMD_MTP_PROJECT (enorm(Emb(embed_token)) ‖ hnorm(prev) →
    // eh_proj → attn_buf) → RUN_ATTENTION(layer 78, fused gating + routing
    // export, KV appended at `pos` of `seq_id`) → FETCH_AND_RUN_MOE(78) →
    // OUTPUT_HEAD(mtp_head: shared_head.norm) → SAMPLE argmax.
    // Returns false on harness failure; out_token/out_conf get the draft.
    bool mtp_step(uint64_t seq_id, uint32_t pos, uint32_t embed_token,
                  uint32_t* out_token, float* out_conf) {
        lipc::Completion cmp{};
        const uint32_t mtp_layer = static_cast<uint32_t>(num_layers_);

        auto pj = make_cmd(lipc::D_CMD_MTP_PROJECT);
        pj.mtp_project.mtp_layer_idx  = mtp_layer;
        pj.mtp_project.input_token_id = embed_token;
        pj.mtp_project.step_idx       = 0;
        send(pj);
        EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE)) << "mtp project";
        if (::testing::Test::HasFailure()) return false;

        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        batch[0].seq_id = seq_id;
        batch[0].token_pos = pos;
        batch[0]._pad = 0;

        auto a = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
        a.run_attention.layer_idx    = mtp_layer;
        a.run_attention.num_seqs     = 1;
        a.run_attention.is_prefill   = 0;
        a.run_attention.use_graph    = 0;
        a.run_attention.emit_gating  = 1;
        a.run_attention.store_gating = 1;
        send(a);
        EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE)) << "mtp attn L78";
        if (::testing::Test::HasFailure()) return false;

        // Routing export → FETCH_AND_RUN (same as decode_step's MoE branch).
        const auto* hdr = reinterpret_cast<const lipc::RoutingExportHeader*>(
            sideband_ + lipc::IpcLayout::kRoutingExportOff);
        EXPECT_EQ(hdr->num_tokens, 1u);
        EXPECT_EQ(hdr->layer_idx, mtp_layer) << "mtp routing-export mismatch";
        const auto* ridx = reinterpret_cast<const int32_t*>(
            sideband_ + lipc::IpcLayout::kRoutingExportIndicesOff);
        const uint32_t rn = hdr->num_tokens * hdr->topk;
        auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
            sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
        const int tp = ep_gpus_;  // EP across TP GPUs only (dspark draft GPU excluded)
        uint32_t count = 0;
        for (uint32_t k = 0; k < rn; ++k) {
            if (ridx[k] < 0) continue;
            entries[count].layer_idx  = mtp_layer;
            entries[count].expert_idx = static_cast<uint16_t>(ridx[k]);
            entries[count].zone       = 0;
            entries[count].gpu_idx    = static_cast<uint8_t>(ridx[k] % tp);
            ++count;
        }
        EXPECT_GT(count, 0u) << "no routed experts exported L78 (MTP)";
        if (count == 0) return false;

        auto m = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
        m.fetch_and_run_moe.layer_idx    = mtp_layer;
        m.fetch_and_run_moe.num_seqs     = 1;
        m.fetch_and_run_moe.expert_count = count;
        m.fetch_and_run_moe.timeout_us   = 120000000;  // 120 s
        m.fetch_and_run_moe.moe_mode     = 0;
        send(m);
        EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE)) << "mtp moe L78";
        if (::testing::Test::HasFailure()) return false;

        auto head = make_cmd(lipc::CMD_OUTPUT_HEAD);
        head.output_head.num_tokens         = 1;
        head.output_head.input_buf_id       = hidden_buf_id_;
        head.output_head.output_buf_id      = logits_buf_id_;
        head.output_head.compute_confidence = 1;
        head.output_head.mtp_head           = 1;  // MTP module 0 shared head
        send(head);
        EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE)) << "mtp head";
        if (::testing::Test::HasFailure()) return false;
        if (out_conf) *out_conf = cmp.compute.top1_prob;

        auto s = make_cmd(lipc::CMD_SAMPLE_TOKENS);
        s.sample_tokens.num_tokens    = 1;
        s.sample_tokens.logits_buf_id = logits_buf_id_;
        s.sample_tokens.vocab_size    = static_cast<uint32_t>(vocab_size_);
        s.sample_tokens.temperature   = 0.0f;  // argmax draft
        s.sample_tokens.top_p         = 1.0f;
        s.sample_tokens.top_k         = 0;
        s.sample_tokens.random_seed   = 42;
        send(s);
        EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE)) << "mtp sample";
        if (::testing::Test::HasFailure()) return false;

        auto* token_ids = reinterpret_cast<uint32_t*>(
            sideband_ + lipc::IpcLayout::kTokenIdsOff);
        if (out_token) *out_token = token_ids[0];
        return true;
    }

    // TD-GLM-INDEXER-PREFILL: one REAL prefill step over `n` prompt tokens at
    // positions [pos0, pos0+n) — the engine's prefill command path (one batch
    // descriptor per PROMPT POSITION, is_prefill=1 + chunk fields), NOT
    // teacher-forced decode. Attention runs dense prefill; the DSA chunk
    // appender stores the positions' indexer keys so the post-prefill decode
    // is sparse-eligible. MoE runs on the production FETCH_AND_RUN seam with
    // the union of the chunk's routed experts.
    void prefill_step(const uint32_t* toks, uint32_t n, uint64_t seq_id,
                      uint32_t pos0) {
        lipc::Completion cmp{};
        auto* token_ids = reinterpret_cast<uint32_t*>(
            sideband_ + lipc::IpcLayout::kTokenIdsOff);
        for (uint32_t i = 0; i < n; ++i) token_ids[i] = toks[i];

        auto embed = make_cmd(lipc::CMD_EMBEDDING_LOOKUP);
        embed.embedding_lookup.num_tokens = n;
        embed.embedding_lookup.output_buf_id = hidden_buf_id_;
        send(embed);
        EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE));

        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        for (uint32_t b = 0; b < n; ++b) {
            batch[b].seq_id = seq_id;
            batch[b].token_pos = pos0 + b;
            batch[b]._pad = 0;
        }

        for (int layer = 0; layer < num_layers_; ++layer) {
            const bool is_moe = layer >= first_moe_layer_;

            auto a = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
            a.run_attention.layer_idx = static_cast<uint32_t>(layer);
            a.run_attention.num_seqs = n;
            a.run_attention.is_prefill = 1;
            a.run_attention.use_graph = 0;
            a.run_attention.chunk_start = pos0;
            a.run_attention.chunk_len = n;
            a.run_attention.emit_gating  = is_moe ? 1 : 0;
            a.run_attention.store_gating = is_moe ? 1 : 0;
            send(a);
            EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "prefill attn L" << layer;
            if (::testing::Test::HasFailure()) return;

            if (is_moe) {
                const auto* hdr =
                    reinterpret_cast<const lipc::RoutingExportHeader*>(
                        sideband_ + lipc::IpcLayout::kRoutingExportOff);
                EXPECT_EQ(hdr->num_tokens, n);
                EXPECT_EQ(hdr->layer_idx, static_cast<uint32_t>(layer))
                    << "routing-export layer mismatch";
                const auto* ridx = reinterpret_cast<const int32_t*>(
                    sideband_ + lipc::IpcLayout::kRoutingExportIndicesOff);
                const uint32_t rn = hdr->num_tokens * hdr->topk;

                // Union of the chunk's routed experts (dedupe across tokens).
                auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
                    sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
                const int tp = ep_gpus_;  // EP across TP GPUs only (dspark draft GPU excluded)
                std::vector<uint8_t> seen(
                    static_cast<size_t>(num_experts_), 0);
                uint32_t count = 0;
                for (uint32_t k = 0; k < rn; ++k) {
                    if (ridx[k] < 0 || ridx[k] >= num_experts_) continue;
                    if (seen[static_cast<size_t>(ridx[k])]) continue;
                    seen[static_cast<size_t>(ridx[k])] = 1;
                    entries[count].layer_idx  = static_cast<uint32_t>(layer);
                    entries[count].expert_idx = static_cast<uint16_t>(ridx[k]);
                    entries[count].zone       = 0;
                    entries[count].gpu_idx =
                        static_cast<uint8_t>(ridx[k] % tp);
                    ++count;
                }
                EXPECT_GT(count, 0u)
                    << "no routed experts exported L" << layer;
                if (count == 0) return;

                auto m = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
                m.fetch_and_run_moe.layer_idx    = static_cast<uint32_t>(layer);
                m.fetch_and_run_moe.num_seqs     = n;
                m.fetch_and_run_moe.expert_count = count;
                m.fetch_and_run_moe.timeout_us   = 120000000;  // 120 s
                m.fetch_and_run_moe.moe_mode     = 0;
                send(m);
            } else {
                auto m = make_cmd(lipc::D_B_CMD_RUN_MOE);
                m.run_moe.layer_idx = static_cast<uint32_t>(layer);
                m.run_moe.num_seqs = n;
                m.run_moe.moe_mode = 0;
                m.run_moe.apply_residual_correction = 0;
                m.run_moe.store_gating_output = 0;
                m.run_moe.emit_checkpoint = 0;
                send(m);
            }
            EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "prefill moe L" << layer;
            if (::testing::Test::HasFailure()) return;
        }
        fprintf(stderr, "[glm52-golden] prefill chunk [%u, %u) done\n",
                pos0, pos0 + n);
    }

    uint32_t run_prompt(float* top1 = nullptr) {
        create_sequence(1, static_cast<uint32_t>(kPromptTokens.size()));
        uint32_t tok = 0;
        const char* pf = std::getenv("GLM52_PREFILL");
        if (pf && *pf == '1') {
            // GLM52_PREFILL=1: real CHUNKED prefill of the first N−1 prompt
            // tokens, then ONE final teacher-forced decode step, which must
            // be SPARSE-ELIGIBLE (watch for "DSA sparse attention ACTIVE" in
            // the log) and must produce the same greedy token as full-decode
            // mode. Default chunk = 1 token (bit-identical to decode mode:
            // 12089 @ 0.4693). Multi-token chunks (GLM52_PREFILL_CHUNK=4 =
            // the whole prefill in one B=4 chunk) are CORRECT since the
            // TD-PREFILL-CHUNK-ATTN stride fixes (kv_a_norm + k_append over
            // the interleaved [c_kv | k_pe] rows) — 12089 @ 0.4747; the small
            // top1 delta vs chunk=1 is M=1-vs-M>1 GEMM kernel-selection
            // accumulation order, not corruption.
            const uint32_t pre =
                static_cast<uint32_t>(kPromptTokens.size()) - 1;
            uint32_t chunk = 1;
            if (const char* c = std::getenv("GLM52_PREFILL_CHUNK"))
                if (uint32_t v = static_cast<uint32_t>(std::atoi(c)); v > 0)
                    chunk = v;
            for (uint32_t pos = 0; pos < pre && !HasFailure();
                 pos += chunk) {
                const uint32_t len = std::min(chunk, pre - pos);
                prefill_step(kPromptTokens.data() + pos, len, 1, pos);
            }
            if (!HasFailure()) {
                float p = 0.f;
                tok = decode_step(kPromptTokens[pre], 1, pre, &p);
                if (top1) *top1 = p;
                fprintf(stderr,
                        "[glm52-golden] post-prefill decode done (argmax=%u)\n",
                        tok);
            }
        } else {
        for (size_t i = 0; i < kPromptTokens.size(); ++i) {
            float p = 0.f;
            tok = decode_step(kPromptTokens[i], 1, static_cast<uint32_t>(i), &p);
            if (top1) *top1 = p;
            if (::testing::Test::HasFailure()) break;
            fprintf(stderr, "[glm52-golden] step %zu/%zu done (argmax=%u)\n",
                    i + 1, kPromptTokens.size(), tok);
        }
        }
        // GLM52_DUMP_LSE=1: dump both ranks' prefill_lse (last processed
        // layer) to check whether TP ranks compute the same or different
        // heads (DCP LSE-combine semantics probe).
        if (const char* dl = std::getenv("GLM52_DUMP_LSE"); dl && *dl == '1') {
            auto* reg = engine_->buffer_registry();
            for (int r = 0; r < 2; ++r) {
                const uint32_t id =
                    find_buf(reg, "dcp.prefill_lse.rank" + std::to_string(r));
                const auto* e = reg->lookup(id);
                if (!e) { fprintf(stderr, "[lse-dump] rank%d missing\n", r); continue; }
                cudaSetDevice(e->gpu_idx);
                std::vector<float> h(32);
                cudaMemcpy(h.data(), e->device_ptr, 32 * sizeof(float),
                           cudaMemcpyDeviceToHost);
                fprintf(stderr, "[lse-dump] rank%d:", r);
                for (int k = 0; k < 32; ++k) fprintf(stderr, " %.4f", h[k]);
                fprintf(stderr, "\n");
            }
        }
        free_sequence(1);
        return tok;
    }

    void run_golden(const std::string& strategy) {
        if (count_big_sm120(28.0) < 2)
            GTEST_SKIP() << "need TWO SM120+ GPUs with >=28 GB visible (TP=2)";
        const std::string src = LAYERSTORM_SOURCE_DIR;
        if (!fs::exists(src + kGgufRel))
            GTEST_SKIP() << "GLM-5.2 GGUF not present";

        start_engine(strategy);
        if (::testing::Test::HasFailure()) return;

        fprintf(stderr, "[glm52-golden] strategy=%s layers=%d experts=%d\n",
                strategy.c_str(), num_layers_, num_experts_);

        float top1 = 0.f;
        uint32_t tok = run_prompt(&top1);
        if (::testing::Test::HasFailure()) return;

        fprintf(stderr, "[glm52-golden] strategy=%s next_token=%u "
                "top1_prob=%.4f (expected %u)\n",
                strategy.c_str(), tok, top1, kGoldenNext);
        EXPECT_EQ(tok, kGoldenNext)
            << "strategy=" << strategy << " produced token " << tok
            << ", expected " << kGoldenNext;
    }

    // ── DSP-5 shared DSpark harness (DsparkLossless + DsparkAcceptRealistic)
    // Step executor: ONE fused D_CMD_RUN_DSPARK_STEP per round; the γ
    // sampled ids (+ γ raw survival c_k — confidence_enabled is always on
    // for the loop goldens) arrive host-side in the sideband readback
    // scratch when the completion fires (DSP-5/DSP-6 readback contract).
    void install_dspark_step_executor(
            layerstorm::speculation::DsparkSpeculationMethod* method) {
        method->set_step_executor(
            [this](const layerstorm::speculation::DsparkStepRequest& req,
                   layerstorm::speculation::DsparkStepResult& res) {
                lipc::Completion cmp{};
                auto c = make_cmd(lipc::D_CMD_RUN_DSPARK_STEP);
                c.run_dspark_step.seq_id = req.seq_id;
                c.run_dspark_step.anchor_token_id =
                    static_cast<uint32_t>(req.anchor_token);
                c.run_dspark_step.anchor_pos = req.anchor_pos;
                c.run_dspark_step.num_query =
                    static_cast<uint8_t>(req.num_query);
                c.run_dspark_step.step_idx = 0;
                send(c);
                if (!wait(cmp, lipc::CMP_COMPUTE_DONE)) return false;
                // DSP-6 readback contract: confidence_enabled → data_bytes
                // = gamma * 8 (gamma i32 ids, then gamma f32 raw c_k).
                const int words = static_cast<int>(cmp.compute.data_bytes /
                                                   sizeof(int32_t));
                const int n = words / 2;  // confidence_enabled=true
                if (n <= 0 || words != 2 * n ||
                    cmp.compute.host_buf_offset == 0)
                    return false;
                const auto* ids = reinterpret_cast<const int32_t*>(
                    sideband_ + cmp.compute.host_buf_offset);
                const auto* conf = reinterpret_cast<const float*>(
                    sideband_ + cmp.compute.host_buf_offset +
                    static_cast<size_t>(n) * sizeof(int32_t));
                res.count = std::min(
                    n, layerstorm::speculation::DsparkStepResult::kMaxGamma);
                for (int k = 0; k < res.count; ++k)
                    res.token_ids[k] = ids[k];
                for (int k = 0; k < res.count; ++k) {
                    EXPECT_TRUE(std::isfinite(conf[k])) << "c_" << k;
                    EXPECT_GT(conf[k], 0.0f) << "c_" << k;
                    EXPECT_LT(conf[k], 1.0f) << "c_" << k;
                    res.confidence[k] = conf[k];
                }
                res.confidence_count = res.count;
                return true;
            });
    }

    // Draft-round loop (DSP-5): propose one whole-γ block → sequential
    // early-stop teacher-forced verification on the main model (rejected
    // drafts are NEVER fed → no KV rewind; every fed position extends the
    // draft context in lockstep) → method verify/on_accept.  `generated`
    // must already hold the first generated token; `history` = prompt +
    // that token; `anchor_pos` = fed-token count.  Extends `generated`
    // until it holds `gen_tokens` tokens.
    void dspark_draft_rounds(
            layerstorm::speculation::DsparkSpeculationMethod* method,
            uint64_t seq_id, int gen_tokens,
            std::vector<uint32_t>& generated, std::vector<int32_t>& history,
            int anchor_pos) {
        ASSERT_FALSE(generated.empty());
        ASSERT_EQ(history.size(),
                  static_cast<size_t>(anchor_pos) + 1);
        uint32_t x = generated.back();  // newest accepted, not yet fed
        int rounds = 0;
        while (static_cast<int>(generated.size()) < gen_tokens
               && !HasFailure()) {
            // Propose: one whole-γ-block DSpark step off the context.
            std::array<int32_t, 16> draft{};
            std::array<float, 16> probs{};  // DSP-6: raw survival c_k
            layerstorm::speculation::DraftContext dctx;
            dctx.seq_id = seq_id;
            dctx.tokens = history.data();
            dctx.num_tokens = static_cast<int>(history.size());
            dctx.max_tokens = method->max_draft_len();
            layerstorm::speculation::DraftResult dres;
            dres.token_ids = draft.data();
            dres.probs = probs.data();
            dres.capacity = static_cast<int>(draft.size());
            method->propose_draft(dctx, dres);
            ASSERT_FALSE(HasFailure());
            ASSERT_GT(dres.count, 0) << "DSpark proposed no drafts";

            // Sequential early-stop verification: teacher-force x, then
            // each draft WHILE it matches the main model's greedy output.
            std::vector<int32_t> targets;
            uint32_t fed = x;
            int p = anchor_pos;
            for (int j = 0; j <= dres.count && !HasFailure(); ++j) {
                const uint32_t y =
                    decode_step(fed, seq_id, static_cast<uint32_t>(p));
                p += 1;
                targets.push_back(static_cast<int32_t>(y));
                if (j == dres.count) break;  // bonus after full acceptance
                if (static_cast<int32_t>(y) != draft[j]) break;  // y = bonus
                fed = y;
            }
            ASSERT_FALSE(HasFailure());

            // Acceptance rule through the METHOD (the loop's early stop
            // must agree with the greedy rule: accepted = feeds - 1).
            layerstorm::speculation::VerifyContext vctx;
            vctx.seq_id = seq_id;
            vctx.draft_tokens = draft.data();
            vctx.num_draft = dres.count;
            vctx.target_tokens = targets.data();
            layerstorm::speculation::VerifyResult vres;
            method->verify(vctx, vres);
            ASSERT_EQ(vres.accepted_count,
                      static_cast<int>(targets.size()) - 1)
                << "method verify disagrees with sequential verification";
            ASSERT_EQ(vres.bonus_token, targets.back());
            method->on_accept(seq_id, vres.accepted_count);

            // Commit: accepted drafts ARE targets[0..m-1]; bonus =
            // targets[m].
            for (int32_t t : targets) {
                generated.push_back(static_cast<uint32_t>(t));
                history.push_back(t);
            }
            x = static_cast<uint32_t>(targets.back());
            anchor_pos = p;
            // Anchor mirror cross-check: fed count == method expectation.
            EXPECT_EQ(method->expected_anchor_pos(seq_id),
                      static_cast<int64_t>(anchor_pos));
            ++rounds;
            fprintf(stderr,
                    "[glm52-dspark] round %d: proposed=%d accepted=%d "
                    "bonus=%u\n",
                    rounds, dres.count, vres.accepted_count, x);
            // Defect triage (TD-DSPARK-ACCEPT-SHORTPROMPT): draft vs the
            // target's actual greedy tokens, position by position — shows
            // WHETHER the drafter is plausible-but-wrong or context-blind.
            {
                std::string ds, ts;
                const int show = std::min<int>(dres.count, 6);
                for (int k = 0; k < show; ++k)
                    ds += " " + std::to_string(draft[static_cast<size_t>(k)]);
                for (size_t k = 0; k < targets.size() && k < 6; ++k)
                    ts += " " + std::to_string(targets[k]);
                fprintf(stderr,
                        "[glm52-dspark-tok] round %d: draft=[%s ] "
                        "target=[%s ]\n",
                        rounds, ds.c_str(), ts.c_str());
            }
            // DSP-6 (informative — raw c_k is uncalibrated until DSP-7):
            // log the per-position survival c_k + the cumulative survival
            // a_j the DSP-9 scheduler will consume.
            {
                std::string cs, as;
                double a = 1.0;
                for (int k = 0; k < dres.count; ++k) {
                    a *= static_cast<double>(probs[static_cast<size_t>(k)]);
                    char buf[32];
                    snprintf(buf, sizeof buf, " %.4f",
                             probs[static_cast<size_t>(k)]);
                    cs += buf;
                    snprintf(buf, sizeof buf, " %.4f", a);
                    as += buf;
                }
                fprintf(stderr,
                        "[glm52-dspark-conf] round %d: c_k=[%s ] a_j=[%s ] "
                        "accepted=%d\n",
                        rounds, cs.c_str(), as.c_str(), vres.accepted_count);
            }
        }
    }

    std::unique_ptr<ldam::Engine> engine_;
    std::unique_ptr<lipc::CommandRing> cmd_ring_;
    std::unique_ptr<lipc::CompletionRing> cmp_ring_;
    uint8_t* sideband_ = nullptr;
    std::string config_path_;
    uint32_t cmd_seq_ = 1;
    int num_layers_ = 0, num_experts_ = 0, first_moe_layer_ = 3;
    int vocab_size_ = 0;
    uint32_t hidden_buf_id_ = 0, logits_buf_id_ = 0;
    // Experts are EP-split across THIS many leading GPUs (== tp_array size).
    // A dspark draft GPU appended by GLM52_DSPARK_EXPORT never gets experts.
    int ep_gpus_ = 2;
};

// int is the engine default; one full pass cold-streams ~420 GB (hours), so
// dequant runs only on explicit request.
TEST_F(Glm52GgufGolden, IntGolden) {
    const char* s = std::getenv("GG_STRATEGY");
    if (s && std::string(s) != "int") GTEST_SKIP() << "GG_STRATEGY!=int";
    run_golden("int");
}

TEST_F(Glm52GgufGolden, DequantGolden) {
    const char* s = std::getenv("GG_STRATEGY");
    if (!s || std::string(s) != "dequant")
        GTEST_SKIP() << "set GG_STRATEGY=dequant to run";
    run_golden("dequant");
}

// ── #16 / GLM-25g: MTP speculative decoding — LOSSLESS golden ────────────────
// INV-MTP-LOSSLESS: speculative decode must produce IDENTICAL tokens to plain
// autoregressive decode — verification (greedy match against the main model)
// rejects any mismatching draft, so speculation changes SPEED, never tokens.
//
// Two sequences on ONE engine (identical weights/state):
//   seq 1 (baseline): teacher-forced prompt, then kGenTokens plain AR tokens.
//   seq 2 (MTP):      teacher-forced prompt with MTP prompt-warm steps
//     (each prompt position i runs one discarded MTP step embedding
//     prompt[i+1] — populating the MTP layer's KV exactly as vLLM's MTP
//     prefill does), then draft rounds: MtpSpeculationMethod::propose_draft
//     (chained mtp_step's off the frozen trunk) → sequential early-stop
//     teacher-forced verification on the main model (rejected drafts are
//     NEVER fed, so no KV rewind is needed) with interleaved MTP catch-up
//     steps (keeps the MTP layer's KV gap-free) → method verify/on_accept.
// The generated sequences must be IDENTICAL; MTP must accept > 0 drafts
// (else it is a no-op, not speculation).
TEST_F(Glm52GgufGolden, MtpLossless) {
    const char* sm = std::getenv("GLM52_MTP");
    if (!sm || *sm != '1') GTEST_SKIP() << "set GLM52_MTP=1 to run";
    if (count_big_sm120(28.0) < 2)
        GTEST_SKIP() << "need TWO SM120+ GPUs with >=28 GB visible (TP=2)";
    const std::string src = LAYERSTORM_SOURCE_DIR;
    if (!fs::exists(src + kGgufRel))
        GTEST_SKIP() << "GLM-5.2 GGUF not present";

    start_engine("int");
    if (::testing::Test::HasFailure()) return;

    // The factory must have constructed the live MTP method (fail-closed
    // contract: engine init would have thrown on a broken MTP config).
    auto* base_method = engine_->speculation_method();
    ASSERT_NE(base_method, nullptr) << "speculation.method=mtp not live";
    ASSERT_STREQ(base_method->name(), "mtp");
    auto* method =
        static_cast<layerstorm::speculation::MtpSpeculationMethod*>(
            base_method);

    constexpr int kGenTokens = 5;  // tokens generated after the prompt
    const auto prompt_len = static_cast<uint32_t>(kPromptTokens.size());

    // ── Phase A: baseline plain-AR sequence (seq 1) ──────────────────────
    std::vector<uint32_t> baseline;
    {
        create_sequence(1, prompt_len);
        uint32_t tok = 0;
        for (size_t i = 0; i < kPromptTokens.size() && !HasFailure(); ++i)
            tok = decode_step(kPromptTokens[i], 1, static_cast<uint32_t>(i));
        ASSERT_FALSE(HasFailure());
        ASSERT_EQ(tok, kGoldenNext) << "baseline prompt diverged";
        baseline.push_back(tok);
        uint32_t pos = prompt_len;
        while (baseline.size() < kGenTokens && !HasFailure()) {
            tok = decode_step(tok, 1, pos++);
            baseline.push_back(tok);
        }
        free_sequence(1);
        ASSERT_FALSE(HasFailure());
        fprintf(stderr, "[glm52-mtp] baseline:");
        for (uint32_t t : baseline) fprintf(stderr, " %u", t);
        fprintf(stderr, "\n");
    }

    // ── Phase B: MTP speculative sequence (seq 2) ────────────────────────
    create_sequence(2, prompt_len);

    // Teacher-forced prompt + MTP prompt-warm (discarded drafts; appends the
    // MTP layer's KV at positions 0..prompt_len-2 with the true next tokens).
    uint32_t x = 0;
    for (size_t i = 0; i < kPromptTokens.size() && !HasFailure(); ++i) {
        x = decode_step(kPromptTokens[i], 2, static_cast<uint32_t>(i));
        if (HasFailure()) break;
        if (i + 1 < kPromptTokens.size()) {
            uint32_t warm_tok = 0;
            float warm_conf = 0.f;
            ASSERT_TRUE(mtp_step(2, static_cast<uint32_t>(i),
                                 kPromptTokens[i + 1],
                                 &warm_tok, &warm_conf));
        }
    }
    ASSERT_FALSE(HasFailure());
    ASSERT_EQ(x, kGoldenNext)
        << "MTP-enabled prompt diverged from golden (lossless violation)";

    // Draft-round loop.  q = anchor position (trunk position of the last
    // fed token); x = newest accepted token (not yet fed).
    std::vector<uint32_t> generated{x};
    std::vector<int32_t> history(kPromptTokens.begin(), kPromptTokens.end());
    history.push_back(static_cast<int32_t>(x));
    int anchor_pos = static_cast<int>(prompt_len) - 1;

    // Step executor: one production-seam MTP step at anchor + step_idx.
    method->set_step_executor(
        [&](const layerstorm::speculation::MtpStepRequest& req,
            layerstorm::speculation::MtpStepResult& res) {
            uint32_t tok = 0;
            float conf = 0.f;
            if (!mtp_step(req.seq_id,
                          static_cast<uint32_t>(anchor_pos + req.step_idx),
                          static_cast<uint32_t>(req.input_token),
                          &tok, &conf))
                return false;
            res.token_id = static_cast<int32_t>(tok);
            res.confidence = conf;
            return true;
        });

    int rounds = 0;
    while (static_cast<int>(generated.size()) < kGenTokens && !HasFailure()) {
        // Propose: chained MTP steps off the frozen trunk (anchor = trunk
        // hidden of position q, left in attn_buf by the last decode_step).
        std::array<int32_t, 16> draft{};
        std::array<float, 16> probs{};
        layerstorm::speculation::DraftContext dctx;
        dctx.seq_id = 2;
        dctx.tokens = history.data();
        dctx.num_tokens = static_cast<int>(history.size());
        dctx.max_tokens = method->max_draft_len();
        layerstorm::speculation::DraftResult dres;
        dres.token_ids = draft.data();
        dres.probs = probs.data();
        dres.capacity = static_cast<int>(draft.size());
        method->propose_draft(dctx, dres);
        ASSERT_FALSE(HasFailure());
        ASSERT_GT(dres.count, 0) << "MTP proposed no drafts";

        // Sequential early-stop verification: teacher-force x, then each
        // draft WHILE it matches the main model's greedy output.  Rejected
        // drafts are never fed → main KV stays clean without rewind.
        // Interleaved MTP catch-up steps keep the MTP layer's KV gap-free
        // (the embedded token y == the token fed at the next position).
        std::vector<int32_t> targets;
        uint32_t fed = x;
        int p = anchor_pos;
        for (int j = 0; j <= dres.count && !HasFailure(); ++j) {
            p += 1;
            const uint32_t y = decode_step(fed, 2, static_cast<uint32_t>(p));
            targets.push_back(static_cast<int32_t>(y));
            if (j == dres.count) break;  // bonus feed after full acceptance
            if (static_cast<int32_t>(y) != draft[j]) break;  // y = bonus
            fed = y;
            uint32_t cu_tok = 0;
            float cu_conf = 0.f;
            ASSERT_TRUE(mtp_step(2, static_cast<uint32_t>(p), y,
                                 &cu_tok, &cu_conf));
        }
        ASSERT_FALSE(HasFailure());

        // Acceptance rule through the METHOD (the loop's early stop must
        // agree with the greedy rule: accepted = feeds - 1).
        layerstorm::speculation::VerifyContext vctx;
        vctx.seq_id = 2;
        vctx.draft_tokens = draft.data();
        vctx.num_draft = dres.count;
        vctx.target_tokens = targets.data();
        layerstorm::speculation::VerifyResult vres;
        method->verify(vctx, vres);
        ASSERT_EQ(vres.accepted_count,
                  static_cast<int>(targets.size()) - 1)
            << "method verify disagrees with sequential verification";
        ASSERT_EQ(vres.bonus_token, targets.back());
        method->on_accept(2, vres.accepted_count);

        // Commit: accepted drafts ARE targets[0..m-1]; bonus = targets[m].
        for (int32_t t : targets) {
            generated.push_back(static_cast<uint32_t>(t));
            history.push_back(t);
        }
        x = static_cast<uint32_t>(targets.back());
        anchor_pos = p;
        ++rounds;
        fprintf(stderr,
                "[glm52-mtp] round %d: proposed=%d accepted=%d bonus=%u\n",
                rounds, dres.count, vres.accepted_count, x);
    }
    free_sequence(2);
    ASSERT_FALSE(HasFailure());

    // ── LOSSLESS assertion + acceptance sanity ───────────────────────────
    generated.resize(std::min(generated.size(),
                              static_cast<size_t>(kGenTokens)));
    std::vector<uint32_t> base_trunc(
        baseline.begin(), baseline.begin() + generated.size());
    EXPECT_EQ(generated, base_trunc)
        << "INV-MTP-LOSSLESS violated: speculative tokens differ from "
           "plain autoregressive decode";

    const auto& stats = method->acceptance_stats();
    fprintf(stderr,
            "[glm52-mtp] rounds=%llu proposed=%llu accepted=%llu "
            "acceptance_rate=%.3f\n",
            static_cast<unsigned long long>(stats.rounds),
            static_cast<unsigned long long>(stats.tokens_proposed),
            static_cast<unsigned long long>(stats.tokens_accepted),
            method->acceptance_rate());
    EXPECT_GT(stats.tokens_proposed, 0u);
    EXPECT_GT(stats.tokens_accepted, 0u)
        << "MTP accepted nothing — drafting is a no-op, not speculation";
}

// ── DSP-3: DSpark aux-hidden export + backbone smoke on the live target ─────
// (b) The export hook must NOT change the target's output: the same golden
//     prompt with the hook ACTIVE (layers {8,23,39,55,70} exported to the
//     draft GPU every step) must still greedily produce 12089 " Paris".
// (a) The export must track the full prompt (ctx_len == prompt length, one
//     fused context row per position) and ONE D_CMD_RUN_DSPARK_STEP off the
//     ingested context must emit finite base logits [gamma=7, V=154880] on
//     the real draft weights.
//
// Run: GLM52_DSPARK_EXPORT=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
//        CUDA_VISIBLE_DEVICES=2,3,0 ./glm52_gguf_golden_test \
//        --gtest_filter='*DsparkAuxExport*'
// (visible ordinals 0,1 = the 5090 TP pair; ordinal 2 = a 5080 draft GPU).
TEST_F(Glm52GgufGolden, DsparkAuxExportGolden) {
    const char* de = std::getenv("GLM52_DSPARK_EXPORT");
    if (!de || *de != '1')
        GTEST_SKIP() << "set GLM52_DSPARK_EXPORT=1 (CUDA_VISIBLE_DEVICES="
                        "2,3,0: two 5090s + one 5080) to run";
    if (count_big_sm120(28.0) < 2)
        GTEST_SKIP() << "need TWO SM120+ GPUs with >=28 GB visible (TP=2)";
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 3)
        GTEST_SKIP() << "need a THIRD visible GPU for the draft";
    {
        size_t freeb = 0, totalb = 0;
        ASSERT_EQ(cudaSetDevice(2), cudaSuccess);
        ASSERT_EQ(cudaMemGetInfo(&freeb, &totalb), cudaSuccess);
        if (freeb < (size_t{11} << 30))
            GTEST_SKIP() << "draft GPU (visible ordinal 2) has only "
                         << freeb << " B free (need ~11 GB)";
    }
    const std::string src = LAYERSTORM_SOURCE_DIR;
    if (!fs::exists(src + kGgufRel))
        GTEST_SKIP() << "GLM-5.2 GGUF not present";
    if (!fs::exists(src + "/test-data/GLM-5.2-speculator.dspark/"
                          "model.safetensors"))
        GTEST_SKIP() << "DSpark speculator checkpoint not present";

    start_engine("int");
    if (::testing::Test::HasFailure()) return;

    auto* rt = engine_->dspark_runtime();
    ASSERT_NE(rt, nullptr) << "speculation.method=dspark did not arm the "
                              "DsparkRuntime";
    ASSERT_EQ(rt->draft_gpu_position(), 2);

    // (b) Target unchanged with the export hook active.
    float top1 = 0.f;
    const uint32_t tok = run_prompt(&top1);
    if (::testing::Test::HasFailure()) return;
    fprintf(stderr,
            "[glm52-dspark] next_token=%u top1=%.4f (expected %u)\n", tok,
            top1, kGoldenNext);
    EXPECT_EQ(tok, kGoldenNext)
        << "aux-hidden export changed the TARGET output";

    // (a1) The export ingested one fused context row per prompt position.
    EXPECT_TRUE(rt->ctx_valid());
    EXPECT_EQ(rt->ctx_seq_id(), 1u);
    EXPECT_EQ(rt->ctx_len(), static_cast<int>(kPromptTokens.size()));

    // (a2) One backbone step off the ingested context: anchor = the golden
    // token at position 5, gamma = speculative_tokens (7).
    lipc::Completion cmp{};
    auto c = make_cmd(lipc::D_CMD_RUN_DSPARK_STEP);
    c.run_dspark_step.seq_id = 1;
    c.run_dspark_step.anchor_token_id = tok;
    c.run_dspark_step.anchor_pos =
        static_cast<uint32_t>(kPromptTokens.size());
    c.run_dspark_step.num_query = 0;  // config speculative_tokens = 7
    c.run_dspark_step.step_idx = 0;
    send(c);
    ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE));
    ASSERT_EQ(rt->last_num_query(), 7);

    const int64_t V = 154880;
    const int H = 6144;
    ASSERT_EQ(cudaSetDevice(rt->draft_backend()->gpu().id), cudaSuccess);
    std::vector<float> logits(static_cast<size_t>(7) * V);
    ASSERT_EQ(cudaMemcpy(logits.data(), rt->base_logits(),
                         logits.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    double sum_abs = 0.0;
    int argmax0 = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        ASSERT_TRUE(std::isfinite(logits[i])) << "non-finite logit " << i;
        sum_abs += std::abs(logits[i]);
        if (i < static_cast<size_t>(V) && logits[i] > logits[argmax0])
            argmax0 = static_cast<int>(i);
    }
    EXPECT_GT(sum_abs, 0.0) << "all-zero base logits";
    std::vector<uint16_t> hid(static_cast<size_t>(7) * H);
    ASSERT_EQ(cudaMemcpy(hid.data(), rt->hidden_out(), hid.size() * 2,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    fprintf(stderr,
            "[glm52-dspark] backbone step OK: gamma=7 finite logits, "
            "slot-0 argmax=%d\n", argmax0);
}

// ── DSP-5: DSpark speculative decoding — LOSSLESS golden ─────────────────────
// INV-DSPARK-LOSSLESS: DSpark speculative decode must produce IDENTICAL
// tokens to plain autoregressive decode — verification (greedy match against
// the main model, sequential early-stop) rejects any mismatching draft, so
// speculation changes SPEED, never tokens.  The first real acceptance
// measurement of the trained RedHatAI GLM-5.2 speculator inside LayerStoRm.
//
// Two sequences on ONE engine (identical weights/state; the aux-hidden
// export runs automatically on EVERY non-draft target step, so the draft's
// context KV tracks whichever sequence is being decoded — INV-DSPARK-AUX):
//   seq 1 (baseline): teacher-forced prompt, then kDsparkGenTokens plain AR
//     tokens.
//   seq 2 (DSpark):   teacher-forced prompt (context KV fills as a side
//     effect), then draft rounds: DsparkSpeculationMethod::propose_draft
//     (ONE D_CMD_RUN_DSPARK_STEP = backbone + Markov head over the whole
//     γ block; ids read from the sideband readback) → sequential early-stop
//     teacher-forced verification on the main model (rejected drafts are
//     NEVER fed, so no KV rewind is needed and every fed position extends
//     the draft context in lockstep) → method verify/on_accept (anchor
//     mirror advances by accepted+1).
// The generated sequences must be IDENTICAL; acceptance must be > 0 (the
// trained drafter actually accepts — else it is a no-op, not speculation).
//
// Run: GLM52_DSPARK=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
//        CUDA_VISIBLE_DEVICES=2,3,0 ./glm52_gguf_golden_test \
//        --gtest_filter='*DsparkLossless*'
// (visible ordinals 0,1 = the 5090 TP pair; ordinal 2 = a 5080 draft GPU).
TEST_F(Glm52GgufGolden, DsparkLossless) {
    const char* dl = std::getenv("GLM52_DSPARK");
    if (!dl || *dl != '1')
        GTEST_SKIP() << "set GLM52_DSPARK=1 (CUDA_VISIBLE_DEVICES=2,3,0: "
                        "two 5090s + one 5080) to run";
    if (count_big_sm120(28.0) < 2)
        GTEST_SKIP() << "need TWO SM120+ GPUs with >=28 GB visible (TP=2)";
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 3)
        GTEST_SKIP() << "need a THIRD visible GPU for the draft";
    {
        size_t freeb = 0, totalb = 0;
        ASSERT_EQ(cudaSetDevice(2), cudaSuccess);
        ASSERT_EQ(cudaMemGetInfo(&freeb, &totalb), cudaSuccess);
        if (freeb < (size_t{11} << 30))
            GTEST_SKIP() << "draft GPU (visible ordinal 2) has only "
                         << freeb << " B free (need ~11 GB)";
    }
    const std::string src = LAYERSTORM_SOURCE_DIR;
    if (!fs::exists(src + kGgufRel))
        GTEST_SKIP() << "GLM-5.2 GGUF not present";
    if (!fs::exists(src + "/test-data/GLM-5.2-speculator.dspark/"
                          "model.safetensors"))
        GTEST_SKIP() << "DSpark speculator checkpoint not present";

    start_engine("int");
    if (::testing::Test::HasFailure()) return;

    // The factory must have constructed the live DSpark method AND the
    // engine must have armed the runtime (fail-closed contract: engine init
    // would have thrown on a broken dspark config/checkpoint).
    auto* base_method = engine_->speculation_method();
    ASSERT_NE(base_method, nullptr) << "speculation.method=dspark not live";
    ASSERT_STREQ(base_method->name(), "dspark");
    auto* method =
        static_cast<layerstorm::speculation::DsparkSpeculationMethod*>(
            base_method);
    auto* rt = engine_->dspark_runtime();
    ASSERT_NE(rt, nullptr) << "DsparkRuntime not armed";

    constexpr int kDsparkGenTokens = 8;  // tokens generated after the prompt
    const auto prompt_len = static_cast<uint32_t>(kPromptTokens.size());

    // ── Phase A: baseline plain-AR sequence (seq 1) ──────────────────────
    std::vector<uint32_t> baseline;
    {
        create_sequence(1, prompt_len);
        uint32_t tok = 0;
        for (size_t i = 0; i < kPromptTokens.size() && !HasFailure(); ++i)
            tok = decode_step(kPromptTokens[i], 1, static_cast<uint32_t>(i));
        ASSERT_FALSE(HasFailure());
        ASSERT_EQ(tok, kGoldenNext) << "baseline prompt diverged";
        baseline.push_back(tok);
        uint32_t pos = prompt_len;
        while (baseline.size() < kDsparkGenTokens && !HasFailure()) {
            tok = decode_step(tok, 1, pos++);
            baseline.push_back(tok);
        }
        free_sequence(1);
        ASSERT_FALSE(HasFailure());
        fprintf(stderr, "[glm52-dspark] baseline:");
        for (uint32_t t : baseline) fprintf(stderr, " %u", t);
        fprintf(stderr, "\n");
    }

    // ── Phase B: DSpark speculative sequence (seq 2) ─────────────────────
    // Teacher-forced prompt.  No warm steps needed: the aux export captures
    // every fed position automatically, and the pos-0 capture re-arms the
    // tracked context onto seq 2 (INV-DSPARK-AUX).
    create_sequence(2, prompt_len);
    uint32_t x = 0;
    for (size_t i = 0; i < kPromptTokens.size() && !HasFailure(); ++i)
        x = decode_step(kPromptTokens[i], 2, static_cast<uint32_t>(i));
    ASSERT_FALSE(HasFailure());
    ASSERT_EQ(x, kGoldenNext)
        << "DSpark-enabled prompt diverged from golden (lossless violation)";
    ASSERT_TRUE(rt->ctx_valid());
    ASSERT_EQ(rt->ctx_seq_id(), 2u);
    ASSERT_EQ(rt->ctx_len(), static_cast<int>(prompt_len));

    // Step executor + draft-round loop: shared fixture harness (also used
    // by DsparkAcceptRealistic).  x = newest accepted token (not yet fed);
    // the anchor position is len(history)-1 == the fed-token count == the
    // runtime's ingested ctx_len (the method derives it from the history).
    install_dspark_step_executor(method);
    std::vector<uint32_t> generated{x};
    std::vector<int32_t> history(kPromptTokens.begin(), kPromptTokens.end());
    history.push_back(static_cast<int32_t>(x));
    dspark_draft_rounds(method, 2, kDsparkGenTokens, generated, history,
                        static_cast<int>(prompt_len));
    free_sequence(2);
    ASSERT_FALSE(HasFailure());

    // ── LOSSLESS assertion + acceptance sanity ───────────────────────────
    generated.resize(std::min(generated.size(),
                              static_cast<size_t>(kDsparkGenTokens)));
    std::vector<uint32_t> base_trunc(
        baseline.begin(), baseline.begin() + generated.size());
    EXPECT_EQ(generated, base_trunc)
        << "INV-DSPARK-LOSSLESS violated: speculative tokens differ from "
           "plain autoregressive decode";

    const auto& stats = method->acceptance_stats();
    fprintf(stderr,
            "[glm52-dspark] rounds=%llu proposed=%llu accepted=%llu "
            "acceptance_rate=%.3f\n",
            static_cast<unsigned long long>(stats.rounds),
            static_cast<unsigned long long>(stats.tokens_proposed),
            static_cast<unsigned long long>(stats.tokens_accepted),
            method->acceptance_rate());
    EXPECT_GT(stats.tokens_proposed, 0u);
    // Acceptance on this 5-token toy prompt is genuinely OOD-low (nearly
    // no draft context): post-layout-fix the preview ckpt measured 3/75 =
    // 0.040 here vs 0.154 on the 137-tok realistic prompt (pre-fix it was
    // 0/105 — the resolved TD-DSPARK-ACCEPT-SHORTPROMPT misalignment).
    // The hard "drafting actually accepts" gate lives in
    // DsparkAcceptRealistic; here LOSSLESS token identity is the invariant
    // and zero acceptance is only WARNED.
    if (stats.tokens_accepted == 0)
        fprintf(stderr,
                "[glm52-dspark] WARNING: zero accepted drafts on the toy "
                "prompt (OOD-low is expected — post-fix baseline 0.040; "
                "zero may indicate a context/aux regression)\n");
}

// ── TD-DSPARK-ACCEPT-SHORTPROMPT / DSP-11(b): realistic-prompt acceptance ────
// The only prior acceptance measurement was 0.024 on the 5-token toy prompt
// "The capital of France is" — far below the paper's in-distribution τ, and
// it was UNKNOWN whether that is (a) expected OOD behavior on a tiny raw
// prompt (mostly-mask γ blocks over a 5-row context) or (b) a context-quality
// defect (aux fusion / backbone numerics subtly off).  This test MEASURES
// acceptance on a realistic LONG prompt: chunked prefill feeds the prompt
// (the aux export ingests every chunk — INV-DSPARK-AUX; chunks larger than
// aux_capture_max_rows ingest in staging-sized pieces, TD-DSPARK-PREFILL-CAP
// resolved — GLM52_DSPARK_AUX_ROWS shrinks the staging to exercise that
// path), then the same draft → verify → accept loop as
// DsparkLossless generates tokens on BOTH a baseline AR sequence and a
// DSpark sequence.  Lossless (token identity) must hold; the acceptance
// rate printed at the end is the TD's measurement.
//
// RESOLVED (2026-07-31): the answer was a third option — a +1 draft-slot
// misalignment (dense layout on bonus-anchor speculators checkpoints; see
// the TD entry / INV-DSPARK-ANCHOR).  Post-fix baselines on the 137-tok
// simple prompt (GLM52_DSPARK_GEN=64): .orig 43/147=0.293 (tau 3.05),
// preview 44/285=0.154 (tau 3.37 — paper chat range; raw rate is
// gamma-confounded across ckpts, compare tau).  This test remains the
// acceptance gate; GLM52_DSPARK_CKPT=<dir> overrides the checkpoint.
//
// Run (default CUDA ordering puts the 5090s first — see spec/DEBUG.md):
//   GLM52_DSPARK_REALISTIC=1 GLM52_PREPACKED=1 CUDA_VISIBLE_DEVICES=0,1,2 \
//     ./glm52_gguf_golden_test --gtest_filter='*DsparkAcceptRealistic*'
// Knobs: GLM52_DSPARK_PROMPT=<token-id file, one id per whitespace token;
//          default /tmp/glm52_longctx_tokens3.txt = the ~2877-token
//          IMPLEMENTATION_GUIDE excerpt of the longctx golden>
//        GLM52_DSPARK_PROMPT_TOKENS=<cap, 0 = whole file (default)>
//        GLM52_DSPARK_GEN=<generated tokens, default 32>
//        GLM52_PREFILL_CHUNK=<prefill chunk, default 64>
TEST_F(Glm52GgufGolden, DsparkAcceptRealistic) {
    const char* dr = std::getenv("GLM52_DSPARK_REALISTIC");
    if (!dr || *dr != '1')
        GTEST_SKIP() << "set GLM52_DSPARK_REALISTIC=1 (3 GPUs: 5090 TP pair "
                        "+ 5080 draft) to run";
    if (count_big_sm120(28.0) < 2)
        GTEST_SKIP() << "need TWO SM120+ GPUs with >=28 GB visible (TP=2)";
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 3)
        GTEST_SKIP() << "need a THIRD visible GPU for the draft";
    {
        size_t freeb = 0, totalb = 0;
        ASSERT_EQ(cudaSetDevice(2), cudaSuccess);
        ASSERT_EQ(cudaMemGetInfo(&freeb, &totalb), cudaSuccess);
        if (freeb < (size_t{11} << 30))
            GTEST_SKIP() << "draft GPU (visible ordinal 2) has only "
                         << freeb << " B free (need ~11 GB)";
    }
    const std::string src = LAYERSTORM_SOURCE_DIR;
    if (!fs::exists(src + kGgufRel))
        GTEST_SKIP() << "GLM-5.2 GGUF not present";
    {
        std::string ckpt = src + "/test-data/GLM-5.2-speculator.dspark";
        if (const char* ck = std::getenv("GLM52_DSPARK_CKPT"); ck && *ck)
            ckpt = ck;
        if (!fs::exists(ckpt + "/model.safetensors"))
            GTEST_SKIP() << "DSpark speculator checkpoint not present: "
                         << ckpt;
    }

    // ── Prompt: token-id file (realistic long text) ──────────────────────
    std::string tok_path = "/tmp/glm52_longctx_tokens3.txt";
    if (const char* tp = std::getenv("GLM52_DSPARK_PROMPT"); tp && *tp)
        tok_path = tp;
    std::vector<uint32_t> prompt;
    {
        std::ifstream tf(tok_path);
        if (!tf.is_open())
            GTEST_SKIP() << "prompt token file not present: " << tok_path
                         << " (set GLM52_DSPARK_PROMPT)";
        int64_t v = 0;
        while (tf >> v) prompt.push_back(static_cast<uint32_t>(v));
    }
    if (const char* pc = std::getenv("GLM52_DSPARK_PROMPT_TOKENS"); pc && *pc)
        if (size_t cap = static_cast<size_t>(std::atoll(pc));
            cap > 0 && cap < prompt.size())
            prompt.resize(cap);
    ASSERT_GE(prompt.size(), size_t{2}) << "prompt too short: " << tok_path;
    int gen_tokens = 32;
    if (const char* g = std::getenv("GLM52_DSPARK_GEN"); g && *g)
        if (int v = std::atoi(g); v > 0) gen_tokens = v;
    uint32_t chunk = 64;
    if (const char* c = std::getenv("GLM52_PREFILL_CHUNK"); c && *c)
        if (uint32_t v = static_cast<uint32_t>(std::atoi(c)); v > 0)
            chunk = v;
    const auto prompt_len = static_cast<uint32_t>(prompt.size());
    fprintf(stderr,
            "[glm52-dspark-realistic] prompt=%s tokens=%u gen=%d chunk=%u\n",
            tok_path.c_str(), prompt_len, gen_tokens, chunk);

    start_engine("int");
    if (::testing::Test::HasFailure()) return;

    auto* base_method = engine_->speculation_method();
    ASSERT_NE(base_method, nullptr) << "speculation.method=dspark not live";
    ASSERT_STREQ(base_method->name(), "dspark");
    auto* method =
        static_cast<layerstorm::speculation::DsparkSpeculationMethod*>(
            base_method);
    auto* rt = engine_->dspark_runtime();
    ASSERT_NE(rt, nullptr) << "DsparkRuntime not armed";

    // Feed the prompt: chunked prefill of the first N−1 tokens (the aux
    // export captures each chunk into the draft context), then ONE
    // teacher-forced decode step for the last token → first generated
    // token.  Same shape for both sequences.
    auto feed_prompt = [&](uint64_t seq_id) -> uint32_t {
        create_sequence(seq_id, prompt_len);
        const uint32_t pre = prompt_len - 1;
        for (uint32_t pos = 0; pos < pre && !HasFailure(); pos += chunk) {
            const uint32_t len = std::min(chunk, pre - pos);
            prefill_step(prompt.data() + pos, len, seq_id, pos);
        }
        if (HasFailure()) return 0;
        return decode_step(prompt[pre], seq_id, pre);
    };

    // ── Phase A: baseline plain-AR sequence (seq 1) ──────────────────────
    std::vector<uint32_t> baseline;
    {
        const auto t0 = std::chrono::steady_clock::now();
        uint32_t tok = feed_prompt(1);
        ASSERT_FALSE(HasFailure());
        const double prefill_s =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr,
                "[glm52-dspark-realistic] baseline prefill done in %.1fs "
                "(first token %u)\n", prefill_s, tok);
        baseline.push_back(tok);
        uint32_t pos = prompt_len;
        while (static_cast<int>(baseline.size()) < gen_tokens
               && !HasFailure()) {
            tok = decode_step(tok, 1, pos++);
            baseline.push_back(tok);
        }
        free_sequence(1);
        ASSERT_FALSE(HasFailure());
        fprintf(stderr, "[glm52-dspark-realistic] baseline:");
        for (uint32_t t : baseline) fprintf(stderr, " %u", t);
        fprintf(stderr, "\n");
    }

    // ── Phase B: DSpark speculative sequence (seq 2) ─────────────────────
    // The pos-0 prefill capture re-arms the tracked draft context onto
    // seq 2 (INV-DSPARK-AUX); every subsequent chunk/decode extends it.
    uint32_t x = feed_prompt(2);
    ASSERT_FALSE(HasFailure());
    ASSERT_EQ(x, baseline[0])
        << "DSpark-enabled prompt diverged from baseline (lossless "
           "violation at the first token)";
    ASSERT_TRUE(rt->ctx_valid())
        << "draft context invalid after chunked prefill — the aux export "
           "did not ingest the prompt (TD-DSPARK-PREFILL-CAP regression?)";
    ASSERT_EQ(rt->ctx_seq_id(), 2u);
    ASSERT_EQ(rt->ctx_len(), static_cast<int>(prompt_len));

    install_dspark_step_executor(method);
    std::vector<uint32_t> generated{x};
    std::vector<int32_t> history(prompt.begin(), prompt.end());
    history.push_back(static_cast<int32_t>(x));
    dspark_draft_rounds(method, 2, gen_tokens, generated, history,
                        static_cast<int>(prompt_len));
    free_sequence(2);
    ASSERT_FALSE(HasFailure());

    // ── LOSSLESS assertion + THE acceptance measurement ──────────────────
    generated.resize(std::min(generated.size(),
                              static_cast<size_t>(gen_tokens)));
    std::vector<uint32_t> base_trunc(
        baseline.begin(), baseline.begin() + generated.size());
    EXPECT_EQ(generated, base_trunc)
        << "INV-DSPARK-LOSSLESS violated: speculative tokens differ from "
           "plain autoregressive decode";

    const auto& stats = method->acceptance_stats();
    fprintf(stderr,
            "[glm52-dspark-realistic] prompt_tokens=%u gen=%zu rounds=%llu "
            "proposed=%llu accepted=%llu acceptance_rate=%.4f "
            "(post-fix baselines: 0.293/.orig g7, 0.154/preview g15 on the "
            "137-tok simple prompt)\n",
            prompt_len, generated.size(),
            static_cast<unsigned long long>(stats.rounds),
            static_cast<unsigned long long>(stats.tokens_proposed),
            static_cast<unsigned long long>(stats.tokens_accepted),
            method->acceptance_rate());
    EXPECT_GT(stats.tokens_proposed, 0u);
    EXPECT_GT(stats.tokens_accepted, 0u)
        << "DSpark accepted nothing on a realistic prompt — drafting is a "
           "no-op, not speculation";
}

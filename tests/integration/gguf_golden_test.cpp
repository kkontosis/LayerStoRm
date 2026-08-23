// GG-9: end-to-end golden validation for a real GGUF model.
//
// Loads the GLM-4.7-Flash-Lite GGUF (deepseek2 MLA, 47 layers, 64 experts top-4)
// via the production engine, teacher-forces the golden prompt
// "The capital of France is", and checks the greedy next token decodes to
// " Paris" — matching llama.cpp on the identical GGUF.
//
// Run (RTX 5090 only — needs ~24 GB):
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
//     ./build/tests/integration/gguf_golden_test
// Strategy override:  GG_STRATEGY=int|dequant   (default: runs both)
//
// Skips gracefully if the GGUF or an SM120+ GPU with enough VRAM is absent.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

namespace lipc = layerstorm::ipc;
namespace ldam = layerstorm::daemon;
namespace fs   = std::filesystem;

// ── Golden constants (from llama.cpp ground truth + the GLM tokenizer.json) ──
//
// llama.cpp greedy completion of "The capital of France is" → " Paris".
// HF tokenizer (test-data/GLM-4.7-Flash/tokenizer.json):
//   "The capital of France is" → [785, 6722, 315, 9621, 374]
//   "[gMASK]" = 154822, "<sop>" = 154824   (GLM completion prefix)
//   " Paris"  = 12089
namespace {
constexpr uint32_t kTokGmask = 154822;
constexpr uint32_t kTokSop   = 154824;
constexpr uint32_t kTokParis = 12089;
const std::vector<uint32_t> kPromptTokens = {
    kTokGmask, kTokSop, 785, 6722, 315, 9621, 374};

const char* kConfigRel = "/test-data/config/glm_4_7_flash.json";
const char* kGgufRel =
    "/test-data/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q5_K_XL.gguf";
}  // namespace

// ── GPU probe ───────────────────────────────────────────────────────────────

// Returns the first visible SM120+ device with >= min_gb total VRAM, or -1.
static int find_big_sm120(double min_gb) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return -1;
    for (int i = 0; i < count; ++i) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, i)
                != cudaSuccess || major < 12)
            continue;
        size_t freeb = 0, totalb = 0;
        if (cudaSetDevice(i) != cudaSuccess) continue;
        if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) continue;
        if (static_cast<double>(totalb) / (1024.0 * 1024 * 1024) >= min_gb)
            return i;
    }
    return -1;
}

// ── Fixture ──────────────────────────────────────────────────────────────────

class GgufGolden : public ::testing::Test {
protected:
    void start_engine(const std::string& strategy) {
        const std::string src = LAYERSTORM_SOURCE_DIR;
        const std::string cfg_path = src + kConfigRel;
        const std::string gguf_path = src + kGgufRel;
        ASSERT_TRUE(fs::exists(cfg_path)) << "missing config " << cfg_path;
        ASSERT_TRUE(fs::exists(gguf_path)) << "missing GGUF " << gguf_path;

        std::ifstream f(cfg_path);
        nlohmann::json j = nlohmann::json::parse(f);
        // Absolute weights path (test cwd is the build dir).
        j["model"]["weights_path"] = gguf_path;
        j["quantization"]["gguf_strategy"] = strategy;
        // Single visible GPU → device 0.
        // vram_gb drives the allocator's single contiguous cudaMalloc
        // (total − safety_margin). The 5090 reports 32607 MiB total but a single
        // 31 GiB block fails after CUDA-context/handle overhead, so advertise a
        // conservative 30 GiB usable.
        j["hardware"]["gpus"] = nlohmann::json::array(
            {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 30},
              {"pcie_gen", 5}, {"pcie_width", 16}, {"numa_node", 0}}});
        j["hardware"]["tp_array"] = nlohmann::json::array({0});
        j["parallelism"]["tensor_parallelism"] = 1;

        // ── VRAM fit (GG-9): the allocator grabs (total_vram − safety_margin)
        // as ONE contiguous cudaMalloc. On a 32 GiB 5090 the default 1 GB margin
        // leaves a ~33 GB block that fails after CUDA-context/handle overhead, so
        // widen it. Cap the KV cache (the golden needs only ~10 tokens) and make
        // the expert STABLE zone hold all ~20 GB of routed experts resident so a
        // single prefetch keeps them put for the whole decode.
        j["memory"]["vram_safety_margin_gb"] = 3.0;
        // PHYSICAL pages = per (seq, logical page, LAYER): seq_create reserves
        // (1 + page_growth_chunk_tokens/page_size) logical pages × 47 layers.
        // The default chunk (1024 tok → 64 pages) makes that 65×47 ≈ 3055 pages
        // per seq — exhausts the pool. The golden needs only ~16 tokens, so shrink
        // the growth chunk to 1 page; 2048 physical pages is then ample.
        j["memory"]["kv_cache"]["max_pages_per_gpu"] = 2048;
        j["memory"]["kv_cache"]["page_growth_chunk_tokens"] = 16;
        j["memory"]["expert_cache"]["stable_zone_fraction"] = 0.95;

        config_path_ = "/tmp/gguf_golden_config_" + strategy + ".json";
        { std::ofstream o(config_path_); o << j.dump(2); }

        vocab_size_ = j["model"]["vocab_size"].get<int>();
        hidden_size_ = j["model"]["hidden_size"].get<int>();
        first_moe_layer_ = j["model"].value("first_k_dense_replace", 1);

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

    // ── command plumbing ────────────────────────────────────────────
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
    bool wait(lipc::Completion& out, uint32_t expected, int timeout_s = 120) {
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

    // Prefetch one layer's experts onto GPU 0; blocks until all are HOT.
    void prefetch_layer(int layer) {
        auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
            sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
        for (int e = 0; e < num_experts_; ++e) {
            entries[e].layer_idx = static_cast<uint32_t>(layer);
            entries[e].expert_idx = static_cast<uint16_t>(e);
            entries[e].zone = 0;
            entries[e].gpu_idx = 0;
        }
        auto c = make_cmd(lipc::D_B_CMD_PREFETCH_BATCH, 0);
        c.prefetch_batch.count = static_cast<uint32_t>(num_experts_);
        c.prefetch_batch.priority = 1.0f;
        c.prefetch_batch.delay_us = 0;
        send(c);
        int ready = 0;
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::seconds(300);
        while (ready < num_experts_
               && std::chrono::steady_clock::now() < deadline) {
            lipc::Completion cmp{};
            if (cmp_ring_->try_read(&cmp)) {
                if (cmp.cmp_type
                        == static_cast<uint32_t>(lipc::CMP_ELM_EXPERT_READY))
                    ++ready;
                else if (cmp.cmp_type == static_cast<uint32_t>(lipc::CMP_ERROR))
                    FAIL() << "prefetch L" << layer << ": " << cmp.error.message;
            } else {
                std::this_thread::yield();
            }
        }
        ASSERT_EQ(ready, num_experts_)
            << "prefetch L" << layer << " got " << ready << "/" << num_experts_;
    }

    void prefetch_all_experts() {
        for (int l = first_moe_layer_; l < num_layers_; ++l)
            prefetch_layer(l);
    }

    // One teacher-forced step: embed input_token, run all layers, return argmax.
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
            auto a = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
            a.run_attention.layer_idx = static_cast<uint32_t>(layer);
            a.run_attention.num_seqs = 1;
            a.run_attention.is_prefill = 0;
            a.run_attention.use_graph = 0;
            send(a);
            EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "attn L" << layer;
            if (::testing::Test::HasFailure()) return 0;

            // The full routed-expert set (2944 × 8.5 MB ≈ 25 GB) does not fit; make
            // this layer's 64 experts resident just before its MoE. Idempotent —
            // warm (already-resident) experts return ready instantly, so only the
            // first token pays the cold H2D; the LRU evicts older layers.
            if (layer >= first_moe_layer_) {
                prefetch_layer(layer);
                if (::testing::Test::HasFailure()) return 0;
            }

            auto m = make_cmd(lipc::D_B_CMD_RUN_MOE);
            m.run_moe.layer_idx = static_cast<uint32_t>(layer);
            m.run_moe.num_seqs = 1;
            m.run_moe.moe_mode = 0;
            m.run_moe.apply_residual_correction = 0;
            m.run_moe.store_gating_output = 0;
            m.run_moe.emit_checkpoint = 0;
            send(m);
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

    // Teacher-force the whole prompt; return the argmax after the last token.
    uint32_t run_prompt(float* top1 = nullptr) {
        create_sequence(1, static_cast<uint32_t>(kPromptTokens.size()));
        uint32_t tok = 0;
        for (size_t i = 0; i < kPromptTokens.size(); ++i) {
            float p = 0.f;
            tok = decode_step(kPromptTokens[i], 1, static_cast<uint32_t>(i), &p);
            if (top1) *top1 = p;
            if (::testing::Test::HasFailure()) break;
        }
        free_sequence(1);
        return tok;
    }

    void run_golden(const std::string& strategy) {
        const int dev = find_big_sm120(24.0);
        if (dev < 0)
            GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
        const std::string src = LAYERSTORM_SOURCE_DIR;
        if (!fs::exists(src + kGgufRel))
            GTEST_SKIP() << "GGUF model not present";

        start_engine(strategy);
        if (::testing::Test::HasFailure()) return;

        fprintf(stderr, "[golden] strategy=%s layers=%d experts=%d\n",
                strategy.c_str(), num_layers_, num_experts_);

        float top1 = 0.f;
        uint32_t tok = run_prompt(&top1);
        if (::testing::Test::HasFailure()) return;

        fprintf(stderr, "[golden] strategy=%s next_token=%u top1_prob=%.4f "
                "(expected %u=' Paris')\n",
                strategy.c_str(), tok, top1, kTokParis);
        EXPECT_EQ(tok, kTokParis)
            << "strategy=" << strategy << " produced token " << tok
            << ", expected " << kTokParis << " (' Paris')";
    }

    std::unique_ptr<ldam::Engine> engine_;
    std::unique_ptr<lipc::CommandRing> cmd_ring_;
    std::unique_ptr<lipc::CompletionRing> cmp_ring_;
    uint8_t* sideband_ = nullptr;
    std::string config_path_;
    uint32_t cmd_seq_ = 1;
    int num_layers_ = 0, num_experts_ = 0, first_moe_layer_ = 1;
    int vocab_size_ = 0, hidden_size_ = 0;
    uint32_t hidden_buf_id_ = 0, logits_buf_id_ = 0;
};

TEST_F(GgufGolden, DequantParis) {
    const char* s = std::getenv("GG_STRATEGY");
    if (s && std::string(s) != "dequant") GTEST_SKIP() << "GG_STRATEGY!=dequant";
    run_golden("dequant");
}

TEST_F(GgufGolden, IntParis) {
    const char* s = std::getenv("GG_STRATEGY");
    if (s && std::string(s) != "int") GTEST_SKIP() << "GG_STRATEGY!=int";
    run_golden("int");
}

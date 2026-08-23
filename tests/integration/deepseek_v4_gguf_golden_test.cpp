// Ticket H (V4-7b): DeepSeek-V4-Flash first-boot golden + first generation.
//
// Boots the production engine on the 151 GB Unsloth DeepSeek-V4-Flash GGUF
// (5 shards, MXFP4 experts, BF16 attention), teacher-forces golden prompts
// PROMPT-FED (one B==1 decode-shaped step per prompt token — the ticket-H
// reference shape; the chunked-prefill arm below gates the TD-V4-CHUNK-
// PREFILL lift against it), and checks greedy
// next tokens against llama.cpp ground truth minted from the identical GGUF
// (ref/llama.cpp build, --temp 0 --top-k 1, add_bos=false per the GGUF
// tokenizer metadata — no BOS prepended).
//
// MoE runs the PRODUCTION seam: fused-gate routing export (hash-gated tid2eid
// on layers 0-2, sqrtsoftplus top-6 elsewhere) + E_CMD_FETCH_AND_RUN_MOE
// fetching ONLY the routed experts, streamed from the GGUF page cache (no
// pinned host pool, no arena, no holder contact — box rules).
//
// Run (needs one SM120 GPU with >= 24 GB; SLOW cold — experts stream from
// NVMe/page cache; run the binary DIRECTLY, never via ctest [120 s timeout]):
//   V4_GOLDEN=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
//     ./build/tests/integration/deepseek_v4_gguf_golden_test
//
// Env gates: V4_GOLDEN=1 required (skips otherwise); V4_GOLDEN_GEN=N extra
// greedy tokens after the France prompt (default 8).

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
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

namespace lipc = layerstorm::ipc;
namespace ldam = layerstorm::daemon;
namespace fs   = std::filesystem;

// ── Golden constants (llama.cpp ground truth on the identical GGUF) ─────────
// Vocab (GGUF tokenizer.ggml.tokens, GPT-2 byte-level: space = "Ġ"):
//   "ĠParis" = 11111, "ĠTokyo" = 30228, "ĠEinstein" = 43490.
namespace {
constexpr uint32_t kTokParis    = 11111;
constexpr uint32_t kTokTokyo    = 30228;
constexpr uint32_t kTokEinstein = 43490;

// llama.cpp --verbose-prompt tokenizations (add_bos=false, raw prompts;
// minted 2026-08-20, llama-completion -no-cnv --temp 0 --top-k 1 on the
// identical GGUF: "The capital of France is Paris. The capital" /
// "The capital of Japan is Tokyo." /
// "The theory of relativity was developed by Albert Einstein in").
const std::vector<uint32_t> kFrancePrompt   = {671, 6102, 294, 8760, 344};
const std::vector<uint32_t> kJapanPrompt    = {671, 6102, 294, 6310, 344};
const std::vector<uint32_t> kEinsteinPrompt = {671, 6129, 294, 76407, 515,
                                               5873, 513, 26218};

// TD-V4-KVT needle prompt (minted 2026-08-21 from the GGUF-dir HF
// tokenizer): "The secret code is 74123. ..." + ~700 filler tokens +
// "Question: what is the secret code? Answer: The secret code is".
// The needle lives in logical CSA page 0 — exactly the page the
// retention-64 tiering config demotes during prefill.
const std::vector<uint32_t> kNeedlePrompt = {
    671, 8613, 4181, 344, 223, 29573, 1349, 16, 20534, 436, 13254, 16, 455,
    3980, 294, 17117, 57292, 14661, 99703, 14, 26261, 33537, 14, 57883, 14,
    14151, 25371, 305, 5970, 16155, 3653, 16, 30223, 8804, 9428, 418, 15571,
    34688, 305, 29930, 4935, 14, 15579, 9641, 304, 24652, 9551, 305, 28432,
    61490, 16, 28738, 7629, 22861, 9023, 42266, 14, 47992, 14, 22869, 305,
    6968, 3881, 58748, 14, 27760, 10555, 305, 6087, 278, 1656, 25261, 16,
    455, 3980, 294, 17117, 57292, 14661, 99703, 14, 26261, 33537, 14, 57883,
    14, 14151, 25371, 305, 5970, 16155, 3653, 16, 30223, 8804, 9428, 418,
    15571, 34688, 305, 29930, 4935, 14, 15579, 9641, 304, 24652, 9551, 305,
    28432, 61490, 16, 28738, 7629, 22861, 9023, 42266, 14, 47992, 14, 22869,
    305, 6968, 3881, 58748, 14, 27760, 10555, 305, 6087, 278, 1656, 25261,
    16, 455, 3980, 294, 17117, 57292, 14661, 99703, 14, 26261, 33537, 14,
    57883, 14, 14151, 25371, 305, 5970, 16155, 3653, 16, 30223, 8804, 9428,
    418, 15571, 34688, 305, 29930, 4935, 14, 15579, 9641, 304, 24652, 9551,
    305, 28432, 61490, 16, 28738, 7629, 22861, 9023, 42266, 14, 47992, 14,
    22869, 305, 6968, 3881, 58748, 14, 27760, 10555, 305, 6087, 278, 1656,
    25261, 16, 455, 3980, 294, 17117, 57292, 14661, 99703, 14, 26261, 33537,
    14, 57883, 14, 14151, 25371, 305, 5970, 16155, 3653, 16, 30223, 8804,
    9428, 418, 15571, 34688, 305, 29930, 4935, 14, 15579, 9641, 304, 24652,
    9551, 305, 28432, 61490, 16, 28738, 7629, 22861, 9023, 42266, 14, 47992,
    14, 22869, 305, 6968, 3881, 58748, 14, 27760, 10555, 305, 6087, 278,
    1656, 25261, 16, 455, 3980, 294, 17117, 57292, 14661, 99703, 14, 26261,
    33537, 14, 57883, 14, 14151, 25371, 305, 5970, 16155, 3653, 16, 30223,
    8804, 9428, 418, 15571, 34688, 305, 29930, 4935, 14, 15579, 9641, 304,
    24652, 9551, 305, 28432, 61490, 16, 28738, 7629, 22861, 9023, 42266, 14,
    47992, 14, 22869, 305, 6968, 3881, 58748, 14, 27760, 10555, 305, 6087,
    278, 1656, 25261, 16, 455, 3980, 294, 17117, 57292, 14661, 99703, 14,
    26261, 33537, 14, 57883, 14, 14151, 25371, 305, 5970, 16155, 3653, 16,
    30223, 8804, 9428, 418, 15571, 34688, 305, 29930, 4935, 14, 15579, 9641,
    304, 24652, 9551, 305, 28432, 61490, 16, 28738, 7629, 22861, 9023,
    42266, 14, 47992, 14, 22869, 305, 6968, 3881, 58748, 14, 27760, 10555,
    305, 6087, 278, 1656, 25261, 16, 455, 3980, 294, 17117, 57292, 14661,
    99703, 14, 26261, 33537, 14, 57883, 14, 14151, 25371, 305, 5970, 16155,
    3653, 16, 30223, 8804, 9428, 418, 15571, 34688, 305, 29930, 4935, 14,
    15579, 9641, 304, 24652, 9551, 305, 28432, 61490, 16, 28738, 7629,
    22861, 9023, 42266, 14, 47992, 14, 22869, 305, 6968, 3881, 58748, 14,
    27760, 10555, 305, 6087, 278, 1656, 25261, 16, 455, 3980, 294, 17117,
    57292, 14661, 99703, 14, 26261, 33537, 14, 57883, 14, 14151, 25371, 305,
    5970, 16155, 3653, 16, 30223, 8804, 9428, 418, 15571, 34688, 305, 29930,
    4935, 14, 15579, 9641, 304, 24652, 9551, 305, 28432, 61490, 16, 28738,
    7629, 22861, 9023, 42266, 14, 47992, 14, 22869, 305, 6968, 3881, 58748,
    14, 27760, 10555, 305, 6087, 278, 1656, 25261, 16, 455, 3980, 294,
    17117, 57292, 14661, 99703, 14, 26261, 33537, 14, 57883, 14, 14151,
    25371, 305, 5970, 16155, 3653, 16, 30223, 8804, 9428, 418, 15571, 34688,
    305, 29930, 4935, 14, 15579, 9641, 304, 24652, 9551, 305, 28432, 61490,
    16, 28738, 7629, 22861, 9023, 42266, 14, 47992, 14, 22869, 305, 6968,
    3881, 58748, 14, 27760, 10555, 305, 6087, 278, 1656, 25261, 16, 455,
    3980, 294, 17117, 57292, 14661, 99703, 14, 26261, 33537, 14, 57883, 14,
    14151, 25371, 305, 5970, 16155, 3653, 16, 30223, 8804, 9428, 418, 15571,
    34688, 305, 29930, 4935, 14, 15579, 9641, 304, 24652, 9551, 305, 28432,
    61490, 16, 28738, 7629, 22861, 9023, 42266, 14, 47992, 14, 22869, 305,
    6968, 3881, 58748, 14, 27760, 10555, 305, 6087, 278, 1656, 25261, 16,
    455, 3980, 294, 17117, 57292, 14661, 99703, 14, 26261, 33537, 14, 57883,
    14, 14151, 25371, 305, 5970, 16155, 3653, 16, 30223, 8804, 9428, 418,
    15571, 34688, 305, 29930, 4935, 14, 15579, 9641, 304, 24652, 9551, 305,
    28432, 61490, 16, 28738, 7629, 22861, 9023, 42266, 14, 47992, 14, 22869,
    305, 6968, 3881, 58748, 14, 27760, 10555, 305, 6087, 278, 1656, 25261,
    16, 455, 3980, 294, 17117, 57292, 14661, 99703, 14, 26261, 33537, 14,
    57883, 14, 14151, 25371, 305, 5970, 16155, 3653, 16, 30223, 8804, 9428,
    418, 15571, 34688, 305, 29930, 4935, 14, 15579, 9641, 304, 24652, 9551,
    305, 28432, 61490, 16, 28738, 7629, 22861, 9023, 42266, 14, 47992, 14,
    22869, 305, 6968, 3881, 58748, 14, 27760, 10555, 305, 6087, 278, 1656,
    25261, 16, 223, 12742, 28, 1205, 344, 270, 8613, 4181, 33, 9361, 28,
    455, 8613, 4181, 344};
// " 741" + "23" — the code's token ids inside the greedy continuation.
constexpr uint32_t kNeedleTok1 = 29573;
constexpr uint32_t kNeedleTok2 = 1349;

const char* kConfigRel = "/test-data/config/deepseek_v4_flash_gguf.json";
const char* kGgufAbs =
    "/srv/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/"
    "DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf";
}  // namespace

// Returns the first visible SM120+ device with >= min_gb total VRAM, or -1.
static int find_big_sm120(double min_gb) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return -1;
    for (int i = 0; i < count; ++i) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   i) != cudaSuccess || major < 12)
            continue;
        size_t freeb = 0, totalb = 0;
        if (cudaSetDevice(i) != cudaSuccess) continue;
        if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) continue;
        if (static_cast<double>(totalb) / (1024.0 * 1024 * 1024) >= min_gb)
            return i;
    }
    return -1;
}

class DeepseekV4Golden : public ::testing::Test {
protected:
    void start_engine() {
        const std::string src = LAYERSTORM_SOURCE_DIR;
        const std::string cfg_path = src + kConfigRel;
        ASSERT_TRUE(fs::exists(cfg_path)) << "missing config " << cfg_path;
        ASSERT_TRUE(fs::exists(kGgufAbs)) << "missing GGUF " << kGgufAbs;

        std::ifstream f(cfg_path);
        nlohmann::json j = nlohmann::json::parse(f);
        j["model"]["weights_path"] = kGgufAbs;
        // V4-2c (TD-V4-TP): V4_TP=2 boots BOTH visible 5090s in TP=2
        // (replicated KV; q_b head-sharded, grouped o_proj group-sharded +
        // allreduce). Default = the single-GPU ticket-H reference shape.
        // V4_TP_EP=1 additionally spreads routed experts e%2 across the TP
        // GPUs (EP-2) — the EP-combine rounding is a KNOWN trajectory-fork
        // source (ticket J measured it), so the token-identity gates run
        // with experts pinned to GPU 0 unless V4_TP_EP is set.
        tp_ = 1;
        if (const char* t = std::getenv("V4_TP"))
            if (std::atoi(t) == 2) tp_ = 2;
        ep_spread_ = false;
        if (const char* e = std::getenv("V4_TP_EP"))
            ep_spread_ = (*e == '1') && tp_ == 2;
        if (tp_ == 2) {
            j["hardware"]["gpus"] = nlohmann::json::array(
                {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 30},
                  {"pcie_gen", 5}, {"pcie_width", 16}, {"numa_node", 0}},
                 {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 30},
                  {"pcie_gen", 5}, {"pcie_width", 16}, {"numa_node", 1}}});
            j["hardware"]["tp_array"] = nlohmann::json::array({0, 1});
            // V4 TP runs REPLICATED KV (schema default is sharded — the
            // GLM champion mode; V4 sharded KV stays fail-closed,
            // TD-V4-DCP-KV).
            j["hardware"]["dcp_kv_mode"] = "replicated";
            j["parallelism"]["tensor_parallelism"] = 2;
        } else {
            // Single visible GPU → device 0. 30 GiB usable (the 32 GiB 5090
            // cannot serve one contiguous 31 GiB block after ctx overhead).
            j["hardware"]["gpus"] = nlohmann::json::array(
                {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 30},
                  {"pcie_gen", 5}, {"pcie_width", 16}, {"numa_node", 0}}});
            j["hardware"]["tp_array"] = nlohmann::json::array({0});
            j["parallelism"]["tensor_parallelism"] = 1;
        }
        // The shipped config carries the ticket-J speculation block
        // (dflash draft on a second GPU). The goldens are the PLAIN
        // teacher-forced/chunked reference — force speculation fully OFF
        // (method "none": has_dspark() keys on the method, not `enabled`)
        // so the single-GPU hardware patch above stays valid (spec
        // correctness is gated by the ticket-J lossless scripts).
        j["speculation"]["enabled"] = false;
        j["speculation"]["method"] = "none";
        // TD-V4-KVT: the needle test's second boot arms CSA-bucket tiering
        // with a tiny retention (64 tokens) so the ~760-token needle prompt
        // demotes pages 0..1 per CSA layer DURING prefill — the needle
        // itself lives in page 0.
        if (kvt_on_) {
            j["memory"]["kv_tiering"]["enabled"] = true;
            j["memory"]["kv_tiering"]["hot_buffer_slots"] = 64;
        }
        j["memory"]["vram_safety_margin_gb"] = 3.0;
        // Small growth chunk: the goldens need only tens of tokens; V4 kMain
        // logical blocks are 256 tokens (one 64-entry CSA page).
        j["memory"]["kv_cache"]["page_growth_chunk_tokens"] = 16;
        // Shrink the demand-driven V4 KV pools (default max_concurrent 32
        // over-provisions all three side tiers for a one-seq golden).
        j["serving"]["max_concurrent_requests"] = 2;
        // Debug lever: V4_GG_STRATEGY=int|dequant overrides the expert GEMM
        // route (kernel-family bisection).
        if (const char* s = std::getenv("V4_GG_STRATEGY"))
            j["quantization"]["gguf_strategy"] = s;
        // V4-5T (TD-V4-TQ-DEVICE): V4_BACKEND=csa_hca_tq|csa_hca_tq_mix
        // runs the suite on a TQ-coded arm (644-B compressed entries,
        // csa_tq decode + SWA-FP8 LSE merge). The argmax goldens are the
        // TQ-arm plausibility mint; the chunked/superchunk/fork tests are
        // SELF-consistent token-identity gates (teacher-forced reference
        // minted in-run on the same arm). Engagement fingerprint (trap
        // #12): the boot log prints "codec csa=tq4 ..." and the France
        // top1 prob departs from the FP8 arm's value.
        if (const char* b = std::getenv("V4_BACKEND"))
            j["compute"]["attention_backend"] = b;

        config_path_ = "/tmp/v4_golden_config.json";
        { std::ofstream o(config_path_); o << j.dump(2); }

        hidden_size_ = j["model"]["hidden_size"].get<int>();
        first_moe_layer_ = j["model"].value("first_k_dense_replace", 0);

        auto backends = ldam::default_backends();
        backends.skip_cuda_graphs = true;
        const auto t0 = std::chrono::steady_clock::now();
        engine_ = std::make_unique<ldam::Engine>(config_path_,
                                                 std::move(backends));
        const double boot_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[v4-golden] engine boot (151 GB GGUF, streaming): "
                "%.1f s\n", boot_s);

        auto& info = engine_->info();
        auto* base = reinterpret_cast<uint8_t*>(info.ipc_base);
        cmd_ring_ = std::make_unique<lipc::CommandRing>(
            base + info.cmd_ring_offset);
        cmp_ring_ = std::make_unique<lipc::CompletionRing>(
            base + info.cmp_ring_offset);
        sideband_ = base + info.sideband_offset;
        num_layers_ = info.num_layers;
        num_experts_ = info.num_experts;
        vocab_size_ = info.vocab_size;   // autodetect adopts 129280
        EXPECT_EQ(vocab_size_, 129280);
        EXPECT_EQ(info.v4_hc_mult, 4);
        EXPECT_EQ(info.v4_num_hash_layers, 3);

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
    bool wait(lipc::Completion& out, uint32_t expected, int timeout_s = 240) {
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
                if (out.cmp_type
                        == static_cast<uint32_t>(lipc::CMP_ELM_EXPERT_READY))
                    continue;  // FETCH_AND_RUN interior readiness signals
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

    // V4_L0_DUMP=<dir>: after layer-0 attention of the FIRST step, raw-dump
    // named executor buffers for the python bisection reference (in-process
    // read: the engine and test share the CUDA context).
    void maybe_dump_l0(int layer, uint32_t token_pos) {
        static const char* dir = std::getenv("V4_L0_DUMP");
        if (!dir || !*dir || layer != 0 || token_pos != 0) return;
        cudaDeviceSynchronize();
        auto* reg = engine_->buffer_registry();
        const char* names[] = {
            "dcp.normed_hidden.rank0", "dcp.q_compressed.rank0",
            "dcp.q_heads.rank0",       "dcp.kv_compressed.rank0",
            "dcp.v4_attn_out.rank0",   "dcp.v4_attn_lse.rank0",
            "dcp.hidden_out.rank0",    "hc_attn_x.rank0",
            "hc_attn_post.rank0",      "hc_attn_comb.rank0",
            "dcp.v4_q_nope.rank0",     "dcp.v4_q_rope.rank0",
            "vram.kv_swa.gpu0"};
        for (const char* n : names) {
            const uint32_t id = find_buf(reg, n);
            if (!id) { fprintf(stderr, "[l0-dump] missing %s\n", n); continue; }
            const auto* e = reg->lookup(id);
            std::vector<uint8_t> host(static_cast<size_t>(
                std::min<int64_t>(e->size_bytes, 1 << 20)));
            cudaMemcpy(host.data(), e->device_ptr, host.size(),
                       cudaMemcpyDeviceToHost);
            std::string path = std::string(dir) + "/" + n + ".bin";
            FILE* f = fopen(path.c_str(), "wb");
            if (f) { fwrite(host.data(), 1, host.size(), f); fclose(f); }
        }
        fprintf(stderr, "[l0-dump] wrote %s\n", dir);
    }

    // One teacher-forced B==1 step on the production seam: embedding →
    // per layer (RUN_ATTENTION with fused gate + routing export →
    // FETCH_AND_RUN_MOE over the routed experts) → head → greedy sample.
    // Every V4-Flash layer is MoE (first_k_dense_replace = 0).
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
        if (::testing::Test::HasFailure()) return 0;

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
                << "attn L" << layer << " pos " << token_pos;
            if (::testing::Test::HasFailure()) return 0;
            maybe_dump_l0(layer, token_pos);

            if (!is_moe) continue;  // unreachable for V4-Flash (all-MoE)

            const auto* hdr = reinterpret_cast<const lipc::RoutingExportHeader*>(
                sideband_ + lipc::IpcLayout::kRoutingExportOff);
            EXPECT_EQ(hdr->num_tokens, 1u);
            EXPECT_EQ(hdr->layer_idx, static_cast<uint32_t>(layer))
                << "routing-export layer mismatch";
            const auto* ridx = reinterpret_cast<const int32_t*>(
                sideband_ + lipc::IpcLayout::kRoutingExportIndicesOff);
            const uint32_t rn = hdr->num_tokens * hdr->topk;

            auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
                sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
            uint32_t count = 0;
            for (uint32_t k = 0; k < rn; ++k) {
                if (ridx[k] < 0) continue;
                entries[count].layer_idx  = static_cast<uint32_t>(layer);
                entries[count].expert_idx = static_cast<uint16_t>(ridx[k]);
                entries[count].zone       = 0;
                // Identity gates pin experts to GPU 0 (MoE math bit-
                // identical to TP=1); V4_TP_EP=1 = EP-2 measurement arm.
                entries[count].gpu_idx    =
                    ep_spread_ ? static_cast<uint8_t>(ridx[k] % 2) : 0;
                ++count;
            }
            EXPECT_GT(count, 0u) << "no routed experts exported L" << layer;
            if (count == 0) return 0;

            auto m = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
            m.fetch_and_run_moe.layer_idx    = static_cast<uint32_t>(layer);
            m.fetch_and_run_moe.num_seqs     = 1;
            m.fetch_and_run_moe.expert_count = count;
            // Cold reads stream from the GGUF page cache / Gen3-capped NVMe.
            m.fetch_and_run_moe.timeout_us   = 200000000;  // 200 s
            m.fetch_and_run_moe.moe_mode     = 0;
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

    // Teacher-force a prompt (one step per token); return the argmax after
    // the last token. Optionally continue greedy for gen_extra tokens.
    uint32_t run_prompt(uint64_t seq_id, const std::vector<uint32_t>& prompt,
                        int gen_extra, std::vector<uint32_t>* gen_out,
                        float* top1 = nullptr) {
        create_sequence(seq_id, static_cast<uint32_t>(prompt.size()));
        uint32_t tok = 0;
        uint32_t pos = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < prompt.size(); ++i, ++pos) {
            float p = 0.f;
            tok = decode_step(prompt[i], seq_id, pos, &p);
            if (top1) *top1 = p;
            fprintf(stderr, "[v4-golden] pos %u fed %u → argmax %u "
                    "(top1 %.4f)\n", pos, prompt[i], tok, p);
            if (::testing::Test::HasFailure()) break;
        }
        if (gen_out) gen_out->push_back(tok);
        for (int g = 0; g < gen_extra && !::testing::Test::HasFailure();
             ++g, ++pos) {
            tok = decode_step(tok, seq_id, pos, nullptr);
            fprintf(stderr, "[v4-golden] gen +%d → token %u\n", g + 1, tok);
            if (gen_out) gen_out->push_back(tok);
        }
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[v4-golden] seq %lu: %u steps in %.1f s "
                "(%.1f s/token)\n", (unsigned long)seq_id, pos, wall,
                pos ? wall / pos : 0.0);
        free_sequence(seq_id);
        return tok;
    }

    // TD-V4-CHUNK-PREFILL (2026-08-21): one CHUNKED prefill step over n
    // prompt tokens at positions [pos0, pos0+n) — the production prefill
    // command shape (one batch descriptor per prompt position, is_prefill=1
    // + chunk fields; the executor runs its internal per-row loop), with ONE
    // expert-UNION FETCH_AND_RUN_MOE per layer (dedup in first-occurrence
    // order — the bridge's fetch_moe_from_export contract). No output head.
    void prefill_chunk_step(const uint32_t* toks, uint32_t n,
                            uint64_t seq_id, uint32_t pos0) {
        lipc::Completion cmp{};
        auto* token_ids = reinterpret_cast<uint32_t*>(
            sideband_ + lipc::IpcLayout::kTokenIdsOff);
        for (uint32_t i = 0; i < n; ++i) token_ids[i] = toks[i];

        auto embed = make_cmd(lipc::CMD_EMBEDDING_LOOKUP);
        embed.embedding_lookup.num_tokens = n;
        embed.embedding_lookup.output_buf_id = hidden_buf_id_;
        send(embed);
        ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE)) << "chunk embed";

        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        for (uint32_t i = 0; i < n; ++i) {
            batch[i].seq_id = seq_id;
            batch[i].token_pos = pos0 + i;
            batch[i]._pad = 0;
        }

        for (int layer = 0; layer < num_layers_; ++layer) {
            auto a = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
            a.run_attention.layer_idx    = static_cast<uint32_t>(layer);
            a.run_attention.num_seqs     = n;
            a.run_attention.is_prefill   = 1;
            a.run_attention.chunk_start  = pos0;
            a.run_attention.chunk_len    = n;
            a.run_attention.use_graph    = 0;
            a.run_attention.emit_gating  = 1;
            a.run_attention.store_gating = 1;
            send(a);
            ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "prefill attn L" << layer << " pos0 " << pos0;

            const auto* hdr = reinterpret_cast<const lipc::RoutingExportHeader*>(
                sideband_ + lipc::IpcLayout::kRoutingExportOff);
            ASSERT_EQ(hdr->num_tokens, n) << "chunk routing export L" << layer;
            const auto* ridx = reinterpret_cast<const int32_t*>(
                sideband_ + lipc::IpcLayout::kRoutingExportIndicesOff);
            const uint32_t rn = hdr->num_tokens * hdr->topk;

            auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
                sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
            std::vector<uint8_t> seen(static_cast<size_t>(num_experts_), 0);
            uint32_t count = 0;
            for (uint32_t k = 0; k < rn; ++k) {
                if (ridx[k] < 0 || ridx[k] >= num_experts_) continue;
                if (seen[static_cast<size_t>(ridx[k])]) continue;
                seen[static_cast<size_t>(ridx[k])] = 1;
                entries[count].layer_idx  = static_cast<uint32_t>(layer);
                entries[count].expert_idx = static_cast<uint16_t>(ridx[k]);
                entries[count].zone       = 0;
                entries[count].gpu_idx    =
                    ep_spread_ ? static_cast<uint8_t>(ridx[k] % 2) : 0;
                ++count;
            }
            ASSERT_GT(count, 0u) << "no routed experts exported L" << layer;

            auto m = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
            m.fetch_and_run_moe.layer_idx    = static_cast<uint32_t>(layer);
            m.fetch_and_run_moe.num_seqs     = n;
            m.fetch_and_run_moe.expert_count = count;
            m.fetch_and_run_moe.timeout_us   = 200000000;  // 200 s
            m.fetch_and_run_moe.moe_mode     = 0;
            send(m);
            ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "chunk moe L" << layer;
        }
        fprintf(stderr, "[v4-golden] prefill chunk [%u, %u) done\n",
                pos0, pos0 + n);
    }

    // SC (superchunk port): one superchunk over n prompt tokens at positions
    // [pos0, pos0+n), processed LAYER-WISE (the GLM TD-PREFILL-MOE-BIG
    // driver shape): embedding per sub-chunk at row_offset, then per layer K
    // attention sub-chunks (superchunk flag + row_offset; fused gate stores
    // the topk at row offsets, exports unioned here) and ONE
    // E_CMD_FETCH_AND_RUN_MOE_BIG over ALL n rows — one fetch per unique
    // expert per layer per superchunk.
    void prefill_superchunk_step(const uint32_t* toks, uint32_t n,
                                 uint64_t seq_id, uint32_t pos0,
                                 uint32_t sub) {
        lipc::Completion cmp{};
        auto* token_ids = reinterpret_cast<uint32_t*>(
            sideband_ + lipc::IpcLayout::kTokenIdsOff);
        // 1. Embedding sub-chunks into hidden rows [off, off+len).
        for (uint32_t off = 0; off < n; off += sub) {
            const uint32_t len = std::min(sub, n - off);
            for (uint32_t i = 0; i < len; ++i) token_ids[i] = toks[off + i];
            auto embed = make_cmd(lipc::CMD_EMBEDDING_LOOKUP);
            embed.embedding_lookup.num_tokens = len;
            embed.embedding_lookup.output_buf_id = hidden_buf_id_;
            embed.embedding_lookup.row_offset = off;
            send(embed);
            ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "superchunk embed off=" << off;
        }

        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
            sideband_ + lipc::IpcLayout::kExpertPrefetchOff);

        // 2. Layer-wise sweep: K sub-chunk attentions → ONE BIG MoE.
        for (int layer = 0; layer < num_layers_; ++layer) {
            std::vector<uint8_t> seen(static_cast<size_t>(num_experts_), 0);
            uint32_t count = 0;
            for (uint32_t off = 0; off < n; off += sub) {
                const uint32_t len = std::min(sub, n - off);
                for (uint32_t b = 0; b < len; ++b) {
                    batch[b].seq_id = seq_id;
                    batch[b].token_pos = pos0 + off + b;
                    batch[b]._pad = 0;
                }
                auto a = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
                a.run_attention.layer_idx    = static_cast<uint32_t>(layer);
                a.run_attention.num_seqs     = len;
                a.run_attention.is_prefill   = 1;
                a.run_attention.chunk_start  = pos0 + off;
                a.run_attention.chunk_len    = len;
                a.run_attention.use_graph    = 0;
                a.run_attention.emit_gating  = 1;
                a.run_attention.store_gating = 1;
                a.run_attention.superchunk   = 1;
                a.run_attention.row_offset   = off;
                send(a);
                ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                    << "superchunk attn L" << layer << " off=" << off;

                const auto* hdr =
                    reinterpret_cast<const lipc::RoutingExportHeader*>(
                        sideband_ + lipc::IpcLayout::kRoutingExportOff);
                ASSERT_EQ(hdr->num_tokens, len)
                    << "superchunk routing export L" << layer;
                ASSERT_EQ(hdr->layer_idx, static_cast<uint32_t>(layer));
                const auto* ridx = reinterpret_cast<const int32_t*>(
                    sideband_ + lipc::IpcLayout::kRoutingExportIndicesOff);
                const uint32_t rn = hdr->num_tokens * hdr->topk;
                for (uint32_t k = 0; k < rn; ++k) {
                    if (ridx[k] < 0 || ridx[k] >= num_experts_) continue;
                    if (seen[static_cast<size_t>(ridx[k])]) continue;
                    seen[static_cast<size_t>(ridx[k])] = 1;
                    entries[count].layer_idx  = static_cast<uint32_t>(layer);
                    entries[count].expert_idx = static_cast<uint16_t>(ridx[k]);
                    entries[count].zone       = 0;
                    entries[count].gpu_idx    =
                        ep_spread_ ? static_cast<uint8_t>(ridx[k] % 2) : 0;
                    ++count;
                }
            }
            ASSERT_GT(count, 0u) << "no routed experts exported L" << layer;

            auto m = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE_BIG);
            m.fetch_and_run_moe_big.layer_idx    =
                static_cast<uint32_t>(layer);
            m.fetch_and_run_moe_big.num_seqs     = n;
            m.fetch_and_run_moe_big.expert_count = count;
            m.fetch_and_run_moe_big.timeout_us   = 200000000;  // 200 s
            m.fetch_and_run_moe_big.moe_mode     = 0;
            m.fetch_and_run_moe_big.chunk_tokens = 0;  // engine default
            send(m);
            ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "superchunk moe L" << layer;
        }
        fprintf(stderr, "[v4-golden] superchunk [%u, %u) sub=%u done\n",
                pos0, pos0 + n, sub);
    }

    // SC: superchunk-prefill the first N-1 prompt tokens (superchunks up to
    // the engine's moe_batch_capacity, sub-chunks of `sub` rows), then ONE
    // decode step on the last token + optional greedy continuation.
    uint32_t run_prompt_superchunk(uint64_t seq_id,
                                   const std::vector<uint32_t>& prompt,
                                   int gen_extra, uint32_t sub,
                                   std::vector<uint32_t>* gen_out,
                                   float* top1 = nullptr) {
        create_sequence(seq_id, static_cast<uint32_t>(prompt.size()));
        const uint32_t pre = static_cast<uint32_t>(prompt.size()) - 1;
        const uint32_t cap = static_cast<uint32_t>(
            std::max(engine_->info().moe_batch_capacity, 1));
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t pos = 0; pos < pre && !::testing::Test::HasFailure();) {
            const uint32_t len = std::min(cap, pre - pos);
            prefill_superchunk_step(prompt.data() + pos, len, seq_id, pos,
                                    sub);
            pos += len;
        }
        uint32_t tok = 0;
        uint32_t pos = pre;
        if (!::testing::Test::HasFailure()) {
            float p = 0.f;
            tok = decode_step(prompt[pre], seq_id, pos, &p);
            ++pos;
            if (top1) *top1 = p;
            fprintf(stderr, "[v4-golden] post-superchunk decode argmax=%u "
                    "(top1 %.4f)\n", tok, p);
        }
        if (gen_out) gen_out->push_back(tok);
        for (int g = 0; g < gen_extra && !::testing::Test::HasFailure();
             ++g, ++pos) {
            tok = decode_step(tok, seq_id, pos, nullptr);
            fprintf(stderr, "[v4-golden] superchunk-arm gen +%d → token %u\n",
                    g + 1, tok);
            if (gen_out) gen_out->push_back(tok);
        }
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[v4-golden] seq %lu (superchunk prefill): %u tokens "
                "in %.1f s\n", (unsigned long)seq_id, pos, wall);
        free_sequence(seq_id);
        return tok;
    }

    // Chunked-prefill the first N-1 prompt tokens (V4_PREFILL_CHUNK-sized
    // chunks; default = one chunk), then ONE decode step on the last token
    // + optional greedy continuation — the production serving shape.
    uint32_t run_prompt_prefill(uint64_t seq_id,
                                const std::vector<uint32_t>& prompt,
                                int gen_extra, std::vector<uint32_t>* gen_out,
                                float* top1 = nullptr) {
        create_sequence(seq_id, static_cast<uint32_t>(prompt.size()));
        const uint32_t pre = static_cast<uint32_t>(prompt.size()) - 1;
        // One chunk by default, capped at the 512-descriptor/sideband bound.
        uint32_t chunk = std::min<uint32_t>(pre, 512);
        if (const char* c = std::getenv("V4_PREFILL_CHUNK"))
            if (uint32_t v = static_cast<uint32_t>(std::atoi(c)); v > 0)
                chunk = std::min<uint32_t>(v, 512);
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t pos = 0; pos < pre && !::testing::Test::HasFailure();
             pos += chunk) {
            const uint32_t len = std::min(chunk, pre - pos);
            prefill_chunk_step(prompt.data() + pos, len, seq_id, pos);
        }
        uint32_t tok = 0;
        uint32_t pos = pre;
        if (!::testing::Test::HasFailure()) {
            float p = 0.f;
            tok = decode_step(prompt[pre], seq_id, pos, &p);
            ++pos;
            if (top1) *top1 = p;
            fprintf(stderr, "[v4-golden] post-prefill decode argmax=%u "
                    "(top1 %.4f)\n", tok, p);
        }
        if (gen_out) gen_out->push_back(tok);
        for (int g = 0; g < gen_extra && !::testing::Test::HasFailure();
             ++g, ++pos) {
            tok = decode_step(tok, seq_id, pos, nullptr);
            fprintf(stderr, "[v4-golden] prefill-arm gen +%d → token %u\n",
                    g + 1, tok);
            if (gen_out) gen_out->push_back(tok);
        }
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[v4-golden] seq %lu (chunked prefill): %u tokens in "
                "%.1f s\n", (unsigned long)seq_id, pos, wall);
        free_sequence(seq_id);
        return tok;
    }

    static bool gated_off() {
        const char* g = std::getenv("V4_GOLDEN");
        return !(g && std::string(g) == "1");
    }

    std::unique_ptr<ldam::Engine> engine_;
    std::unique_ptr<lipc::CommandRing> cmd_ring_;
    std::unique_ptr<lipc::CompletionRing> cmp_ring_;
    uint8_t* sideband_ = nullptr;
    std::string config_path_;
    uint32_t cmd_seq_ = 1;
    int num_layers_ = 0, num_experts_ = 0, first_moe_layer_ = 0;
    int vocab_size_ = 0, hidden_size_ = 0;
    int tp_ = 1;              // V4_TP=2 → TP=2 boot (V4-2c)
    bool ep_spread_ = false;  // V4_TP_EP=1 → EP-2 expert spread
    bool kvt_on_ = false;     // TD-V4-KVT: arm CSA tiering for this boot
    uint32_t hidden_buf_id_ = 0, logits_buf_id_ = 0;
};

// The correctness golden: "The capital of France is" → " Paris" (11111),
// plus a short greedy continuation (coherence sample for the ledger).
TEST_F(DeepseekV4Golden, FranceParisAndGeneration) {
    if (gated_off()) GTEST_SKIP() << "set V4_GOLDEN=1 to run";
    if (find_big_sm120(24.0) < 0)
        GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "V4 GGUF not present";

    start_engine();
    if (::testing::Test::HasFailure()) return;
    fprintf(stderr, "[v4-golden] layers=%d experts=%d vocab=%d\n",
            num_layers_, num_experts_, vocab_size_);

    int gen_extra = 8;
    if (const char* g = std::getenv("V4_GOLDEN_GEN")) gen_extra = atoi(g);

    float top1 = 0.f;
    std::vector<uint32_t> gen;
    const uint32_t tok = run_prompt(1, kFrancePrompt, gen_extra, &gen, &top1);
    if (::testing::Test::HasFailure()) return;

    fprintf(stderr, "[v4-golden] next_token=%u top1=%.4f expected %u "
            "(' Paris')\n[v4-golden] generation ids:", gen.empty() ? tok
            : gen.front(), top1, kTokParis);
    for (uint32_t t : gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n");
    EXPECT_EQ(gen.empty() ? tok : gen.front(), kTokParis);
}

// Additional vocab goldens minted from the same GGUF via llama.cpp.
TEST_F(DeepseekV4Golden, VocabGoldens) {
    if (gated_off()) GTEST_SKIP() << "set V4_GOLDEN=1 to run";
    if (find_big_sm120(24.0) < 0)
        GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "V4 GGUF not present";

    start_engine();
    if (::testing::Test::HasFailure()) return;

    const uint32_t tokyo = run_prompt(2, kJapanPrompt, 0, nullptr);
    if (::testing::Test::HasFailure()) return;
    EXPECT_EQ(tokyo, kTokTokyo) << "Japan prompt";

    const uint32_t einstein = run_prompt(3, kEinsteinPrompt, 0, nullptr);
    if (::testing::Test::HasFailure()) return;
    EXPECT_EQ(einstein, kTokEinstein) << "Einstein prompt";
}

// TD-V4-CHUNK-PREFILL correctness gate (GLM52_PREFILL pattern): chunked
// prefill-then-decode must be TOKEN-IDENTICAL to the teacher-forced
// per-token reference. One teacher-forced run mints the reference greedy
// trajectory; the prompt is then extended with the first K reference
// tokens so the chunk exceeds the ring-clamped verify bound (8 rows) and
// exercises the snapshot-skip + staging-drain prefill path; the chunked
// arm must reproduce the remaining reference continuation exactly.
// Env: V4_PREFILL=1 to run (with V4_GOLDEN=1); V4_PREFILL_GEN (default 12)
// reference continuation length; V4_PREFILL_CHUNK chunk override.
TEST_F(DeepseekV4Golden, ChunkedPrefillTokenIdentity) {
    if (gated_off()) GTEST_SKIP() << "set V4_GOLDEN=1 to run";
    const char* pf = std::getenv("V4_PREFILL");
    if (!(pf && std::string(pf) == "1"))
        GTEST_SKIP() << "set V4_PREFILL=1 to run the chunked-prefill gate";
    if (find_big_sm120(24.0) < 0)
        GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "V4 GGUF not present";

    start_engine();
    if (::testing::Test::HasFailure()) return;

    int gen_extra = 12;
    if (const char* g = std::getenv("V4_PREFILL_GEN")) gen_extra = atoi(g);
    if (gen_extra < 10) gen_extra = 10;

    // Reference: teacher-forced greedy trajectory (the ticket-H shape).
    std::vector<uint32_t> ref_gen;
    const auto t_ref0 = std::chrono::steady_clock::now();
    run_prompt(21, kFrancePrompt, gen_extra, &ref_gen);
    const double ref_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_ref0).count();
    if (::testing::Test::HasFailure()) return;
    ASSERT_EQ(ref_gen.size(), static_cast<size_t>(gen_extra) + 1);
    EXPECT_EQ(ref_gen.front(), kTokParis);

    // Long prompt = prompt + first K reference tokens → the prefill chunk
    // (len-1 rows) exceeds the 8-row snapshot bound.
    const int K = 8;
    std::vector<uint32_t> long_prompt = kFrancePrompt;
    for (int i = 0; i < K; ++i) long_prompt.push_back(ref_gen[i]);
    // Greedy chain: feeding long_prompt's last token (ref_gen[K-1]) must
    // produce ref_gen[K], then ref_gen[K+1], ...
    const int pf_extra = gen_extra - K;   // >= 2 by the clamp above

    std::vector<uint32_t> pf_gen;
    float pf_top1 = 0.f;
    const auto t_pf0 = std::chrono::steady_clock::now();
    run_prompt_prefill(22, long_prompt, pf_extra, &pf_gen, &pf_top1);
    const double pf_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_pf0).count();
    if (::testing::Test::HasFailure()) return;

    ASSERT_EQ(pf_gen.size(), static_cast<size_t>(pf_extra) + 1);
    fprintf(stderr, "[v4-golden] prefill-identity: ref");
    for (int i = 0; i <= pf_extra; ++i) fprintf(stderr, " %u", ref_gen[K + i]);
    fprintf(stderr, " | chunked");
    for (uint32_t t : pf_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n[v4-golden] walls: teacher-forced %.1f s vs chunked "
            "%.1f s (%zu-token prompt arm)\n", ref_wall, pf_wall,
            long_prompt.size());
    for (int i = 0; i <= pf_extra; ++i)
        EXPECT_EQ(pf_gen[static_cast<size_t>(i)], ref_gen[K + i])
            << "chunked-prefill continuation diverges at +" << i;
}

// TD-V4-SERVE-PREFIX fork gate: CMD_SEQ_FORK of a chunk-prefilled V4
// sequence must clone COMPLETE per-seq state (kMain CoW + side-tier
// copy-on-fork + executor state rings) — BOTH the forked child's and the
// parent's greedy continuations must equal the unforked reference. This is
// the prefix-cache primitive: registration forks request→holder (parent
// continues = the parent leg), a hit forks holder→request (child continues
// = the child leg). Env: V4_FORK=1 (with V4_GOLDEN=1).
TEST_F(DeepseekV4Golden, ForkContinuationTokenIdentity) {
    if (gated_off()) GTEST_SKIP() << "set V4_GOLDEN=1 to run";
    const char* fk = std::getenv("V4_FORK");
    if (!(fk && std::string(fk) == "1"))
        GTEST_SKIP() << "set V4_FORK=1 to run the fork gate";
    if (find_big_sm120(24.0) < 0)
        GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "V4 GGUF not present";

    start_engine();
    if (::testing::Test::HasFailure()) return;

    int gen_extra = 12;
    if (const char* g = std::getenv("V4_FORK_GEN")) gen_extra = atoi(g);
    if (gen_extra < 10) gen_extra = 10;

    // Reference: teacher-forced greedy trajectory, then build the long
    // prompt exactly like the chunked-prefill gate.
    std::vector<uint32_t> ref_gen;
    run_prompt(61, kFrancePrompt, gen_extra, &ref_gen);
    if (::testing::Test::HasFailure()) return;
    ASSERT_EQ(ref_gen.size(), static_cast<size_t>(gen_extra) + 1);
    const int K = 8;
    std::vector<uint32_t> long_prompt = kFrancePrompt;
    for (int i = 0; i < K; ++i) long_prompt.push_back(ref_gen[i]);
    const int fk_extra = gen_extra - K;   // >= 2

    // Chunk-prefill the prompt body on the PARENT, then fork BEFORE any
    // decode (the prefix-holder registration shape).
    const uint64_t parent = 62, child = 63;
    create_sequence(parent, static_cast<uint32_t>(long_prompt.size()));
    const uint32_t pre = static_cast<uint32_t>(long_prompt.size()) - 1;
    prefill_chunk_step(long_prompt.data(), pre, parent, 0);
    if (::testing::Test::HasFailure()) return;
    {
        lipc::Completion cmp{};
        auto c = make_cmd(lipc::CMD_SEQ_FORK);
        c.seq_fork.src_seq_id = parent;
        c.seq_fork.dst_seq_id = child;
        send(c);
        ASSERT_TRUE(wait(cmp, lipc::CMP_SEQ_OP_DONE));
        ASSERT_EQ(cmp.status, 0u) << "CMD_SEQ_FORK failed";
    }

    // CHILD leg (the prefix-cache hit): decode from the fork point.
    std::vector<uint32_t> child_gen;
    {
        uint32_t tok = decode_step(long_prompt[pre], child, pre, nullptr);
        child_gen.push_back(tok);
        uint32_t pos = pre + 1;
        for (int g = 0; g < fk_extra && !::testing::Test::HasFailure();
             ++g, ++pos) {
            tok = decode_step(tok, child, pos, nullptr);
            child_gen.push_back(tok);
        }
    }
    if (::testing::Test::HasFailure()) return;

    // PARENT leg (the request continuing after holder registration).
    std::vector<uint32_t> parent_gen;
    {
        uint32_t tok = decode_step(long_prompt[pre], parent, pre, nullptr);
        parent_gen.push_back(tok);
        uint32_t pos = pre + 1;
        for (int g = 0; g < fk_extra && !::testing::Test::HasFailure();
             ++g, ++pos) {
            tok = decode_step(tok, parent, pos, nullptr);
            parent_gen.push_back(tok);
        }
    }
    free_sequence(child);
    free_sequence(parent);
    if (::testing::Test::HasFailure()) return;

    fprintf(stderr, "[v4-golden] fork-identity: ref");
    for (int i = 0; i <= fk_extra; ++i) fprintf(stderr, " %u", ref_gen[K + i]);
    fprintf(stderr, " | child");
    for (uint32_t t : child_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, " | parent");
    for (uint32_t t : parent_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n");
    ASSERT_EQ(child_gen.size(), static_cast<size_t>(fk_extra) + 1);
    ASSERT_EQ(parent_gen.size(), static_cast<size_t>(fk_extra) + 1);
    for (int i = 0; i <= fk_extra; ++i) {
        EXPECT_EQ(child_gen[static_cast<size_t>(i)], ref_gen[K + i])
            << "forked-child continuation diverges at +" << i;
        EXPECT_EQ(parent_gen[static_cast<size_t>(i)], ref_gen[K + i])
            << "post-fork parent continuation diverges at +" << i;
    }
}

// SC (superchunk port) TOKEN-IDENTITY gate: superchunk-prefilled
// continuation must EXACTLY equal the teacher-forced reference AND the P1
// chunked-prefill reference. The prompt is extended with the first K
// reference tokens so the superchunk body (K+4 rows) spans TWO sub-chunks —
// the first exceeds the snapshot bound (16) and takes the BATCH-shaped
// attention body, the second takes the per-row body — exercising batch
// attention, row_offset placement, layer-sweep window replays and the
// per-layer FETCH_AND_RUN_MOE_BIG union in one gate.
// Env: V4_SUPERCHUNK=1 (with V4_GOLDEN=1); V4_SC_GEN (default 40) reference
// continuation length; V4_SC_SUB (default 24) sub-chunk rows.
TEST_F(DeepseekV4Golden, SuperchunkPrefillTokenIdentity) {
    if (gated_off()) GTEST_SKIP() << "set V4_GOLDEN=1 to run";
    const char* sc = std::getenv("V4_SUPERCHUNK");
    if (!(sc && std::string(sc) == "1"))
        GTEST_SKIP() << "set V4_SUPERCHUNK=1 to run the superchunk gate";
    if (find_big_sm120(24.0) < 0)
        GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "V4 GGUF not present";

    start_engine();
    if (::testing::Test::HasFailure()) return;

    int gen_extra = 40;
    if (const char* g = std::getenv("V4_SC_GEN")) gen_extra = atoi(g);
    if (gen_extra < 26) gen_extra = 26;   // K >= 22 → first sub-chunk > 16
    uint32_t sub = 24;
    if (const char* s = std::getenv("V4_SC_SUB"))
        if (uint32_t v = static_cast<uint32_t>(std::atoi(s)); v > 0)
            sub = std::min<uint32_t>(v, 512);

    // Reference: teacher-forced greedy trajectory.
    std::vector<uint32_t> ref_gen;
    const auto t_ref0 = std::chrono::steady_clock::now();
    run_prompt(41, kFrancePrompt, gen_extra, &ref_gen);
    const double ref_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_ref0).count();
    if (::testing::Test::HasFailure()) return;
    ASSERT_EQ(ref_gen.size(), static_cast<size_t>(gen_extra) + 1);
    EXPECT_EQ(ref_gen.front(), kTokParis);

    const int K = gen_extra - 4;
    std::vector<uint32_t> long_prompt = kFrancePrompt;
    for (int i = 0; i < K; ++i) long_prompt.push_back(ref_gen[i]);
    const int sc_extra = gen_extra - K;   // 4

    // Arm A: P1 chunked prefill (per-command chunk, auto-batch body).
    std::vector<uint32_t> ck_gen;
    const auto t_ck0 = std::chrono::steady_clock::now();
    run_prompt_prefill(42, long_prompt, sc_extra, &ck_gen);
    const double ck_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_ck0).count();
    if (::testing::Test::HasFailure()) return;

    // Arm B: superchunk sweep (sub-chunked attention + MOE_BIG per layer).
    std::vector<uint32_t> sc_gen;
    const auto t_sc0 = std::chrono::steady_clock::now();
    run_prompt_superchunk(43, long_prompt, sc_extra, sub, &sc_gen);
    const double sc_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_sc0).count();
    if (::testing::Test::HasFailure()) return;

    ASSERT_EQ(ck_gen.size(), static_cast<size_t>(sc_extra) + 1);
    ASSERT_EQ(sc_gen.size(), static_cast<size_t>(sc_extra) + 1);
    fprintf(stderr, "[v4-golden] superchunk-identity: ref");
    for (int i = 0; i <= sc_extra; ++i) fprintf(stderr, " %u", ref_gen[K + i]);
    fprintf(stderr, " | chunked");
    for (uint32_t t : ck_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, " | superchunk");
    for (uint32_t t : sc_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n[v4-golden] walls: teacher %.1f s, chunked %.1f s, "
            "superchunk %.1f s (%zu-token prompt arm)\n", ref_wall, ck_wall,
            sc_wall, long_prompt.size());
    for (int i = 0; i <= sc_extra; ++i) {
        EXPECT_EQ(sc_gen[static_cast<size_t>(i)], ref_gen[K + i])
            << "superchunk continuation diverges from teacher-forced at +"
            << i;
        EXPECT_EQ(sc_gen[static_cast<size_t>(i)], ck_gen[static_cast<size_t>(i)])
            << "superchunk continuation diverges from chunked at +" << i;
    }
}

// TD-V4-KVT correctness gate (the project's hard-won lesson: parametric
// goldens MISS mid-context KV corruption — a NEEDLE retrieval must pass
// UNDER TIERING PRESSURE). Two boots in one test:
//   1. tiering OFF — reference: chunked-prefill the ~760-token needle
//      prompt, decode 10 greedy tokens; the continuation should contain
//      the secret-code tokens (" 741" "23" — model-capability sanity).
//   2. tiering ON (retention 64 ⇒ CSA pages 0..1 demote per layer during
//      prefill; the needle is IN page 0) — the continuation must be
//      TOKEN-IDENTICAL to the reference, and the LS_V4_KVT_STATS counters
//      must show real demotions AND repromotes (anti-vacuous, trap #12).
// Coverage honesty: hundreds of tokens (the V4 streaming wall makes 10k+
// impractical); the machinery is page-granular and position-generic, so
// the untested long-context axis is capacity, not addressing.
TEST_F(DeepseekV4Golden, KvTieringNeedleIdentity) {
    if (gated_off()) GTEST_SKIP() << "set V4_GOLDEN=1 to run";
    const char* kv = std::getenv("V4_KVT");
    if (!(kv && std::string(kv) == "1"))
        GTEST_SKIP() << "set V4_KVT=1 to run the tiering needle gate";
    if (find_big_sm120(24.0) < 0)
        GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "V4 GGUF not present";

    const char* stats_path = "/tmp/v4_kvt_stats.txt";
    std::remove(stats_path);
    setenv("LS_V4_KVT_STATS", stats_path, 1);

    // Boot 1: tiering OFF — the reference trajectory.
    kvt_on_ = false;
    start_engine();
    if (::testing::Test::HasFailure()) return;
    std::vector<uint32_t> ref_gen;
    run_prompt_prefill(31, kNeedlePrompt, 10, &ref_gen);
    if (::testing::Test::HasFailure()) return;
    stop_engine();

    fprintf(stderr, "[v4-kvt] reference continuation:");
    for (uint32_t t : ref_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n");
    const bool retrieves =
        std::find(ref_gen.begin(), ref_gen.end(), kNeedleTok1)
            != ref_gen.end()
        && std::find(ref_gen.begin(), ref_gen.end(), kNeedleTok2)
            != ref_gen.end();
    EXPECT_TRUE(retrieves)
        << "reference (untiered) continuation does not contain the secret "
           "code tokens — the retrieval sanity is void (identity gate below "
           "still meaningful)";

    // Boot 2: tiering ON — must be token-identical AND actually tier.
    kvt_on_ = true;
    start_engine();
    kvt_on_ = false;
    if (::testing::Test::HasFailure()) return;
    std::vector<uint32_t> kvt_gen;
    run_prompt_prefill(32, kNeedlePrompt, 10, &kvt_gen);
    if (::testing::Test::HasFailure()) return;
    stop_engine();

    fprintf(stderr, "[v4-kvt] tiered continuation:   ");
    for (uint32_t t : kvt_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n");
    ASSERT_EQ(kvt_gen.size(), ref_gen.size());
    for (size_t i = 0; i < ref_gen.size(); ++i)
        EXPECT_EQ(kvt_gen[i], ref_gen[i])
            << "tiering-on continuation diverges at +" << i;

    // Anti-vacuous: the stats file (manager destructor) must show work.
    std::ifstream sf(stats_path);
    ASSERT_TRUE(sf.good()) << "LS_V4_KVT_STATS file missing — tiering never "
                              "constructed?";
    long long demotes = 0, repromotes = 0;
    std::string line, last;
    while (std::getline(sf, line)) if (!line.empty()) last = line;
    std::sscanf(last.c_str(), "demotes=%lld repromotes=%lld", &demotes,
                &repromotes);
    fprintf(stderr, "[v4-kvt] stats: %s\n", last.c_str());
    EXPECT_GT(demotes, 0) << "no pages demoted — the gate is vacuous";
    EXPECT_GT(repromotes, 0) << "no pages repromoted — selection never "
                                "touched cold pages (vacuous)";
}

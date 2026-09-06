// GF3.9 verify clause: GLM-5.3-Flash HYBRID-execution first-boot gate.
//
// Boots the production engine on the 186 GiB Unsloth GLM-5.3-Flash GGUF
// (6 shards, Q4_K_XL experts) in the ticket reference shape (TP=1, one
// 5090, speculation OFF) and drives the production seam by hand:
// CMD_EMBEDDING_LOOKUP -> per layer D_B_CMD_RUN_ATTENTION (fused gate +
// routing export on the routed layers) -> E_CMD_FETCH_AND_RUN_MOE over the
// exported experts -> CMD_OUTPUT_HEAD -> CMD_SAMPLE_TOKENS at temperature 0.
//
// GLM-5.3-Flash is a HYBRID stack: 45 hidden layers, 34 `linear_attention`
// (KDA, no KV -- per-request recurrent state) interleaved with 11 NoPE
// sparse-MLA layers at depths 3,7,...,43, mHC active on every layer
// (hc_mult=4), `first_k_dense_replace=3` (layers 0-2 dense MLP, 3+ routed).
// That is exactly what these three tests exist to catch: any KDA layer that
// silently computes nothing, or any recurrent-state carry that is not
// invariant to how the prompt was split into launches, shows up as a
// TOKEN divergence between the legs below.
//
// The gate is ARM-AGNOSTIC by construction (the V4 pattern, not the GLM-5.2
// pattern): there is no llama.cpp ground truth for this model, so tests 2
// and 3 mint their reference IN-RUN on the SAME arm, teacher-forced one
// decode-shaped step per prompt token, and compare token-by-token. Test 1
// is only a semantic smoke check.
//
// Run (needs one SM120+ GPU with >= 24 GB; run the binary DIRECTLY, never
// via ctest -- every integration test carries a 120 s ctest TIMEOUT and the
// boot alone takes MINUTES: 186 GiB of GGUF shards are opened, the routed
// set is staged into the pinned arena, and cold expert reads stream from
// the page cache / NVMe):
//   GLM53_GOLDEN=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
//     ./build/tests/integration/glm53flash_gguf_golden_test \
//     --gtest_filter='*Golden'
//   GLM53_GOLDEN=1 GLM53_PREFILL=1    ... --gtest_filter='*ChunkedPrefill*'
//   GLM53_GOLDEN=1 GLM53_SUPERCHUNK=1 ... --gtest_filter='*Superchunk*'
//
// OPS CAVEAT (arena holder): the shipped config pins a large host expert
// pool with cross-node spill. Booting this gate while a warm arena holder
// owns the box's pinned pages will EVICT that warm store -- the holder must
// be stopped (or the box handed over) before a run, and the next champion
// serve pays a cold warm-up.
//
// Env gates:
//   GLM53_GOLDEN=1        required -- every test skips otherwise
//   GLM53_GOLDEN_GEN=N    extra greedy tokens after the France prompt (8)
//   GLM53_PREFILL=1       run ChunkedPrefillTokenIdentity
//   GLM53_PREFILL_GEN=N   reference continuation length (12, clamped >= 10)
//   GLM53_PREFILL_CHUNK=N chunk rows override (default = one chunk, cap 512)
//   GLM53_SUPERCHUNK=1    run SuperchunkPrefillTokenIdentity
//   GLM53_SC_GEN=N        reference continuation length (40, clamped >= 26)
//   GLM53_SC_SUB=N        superchunk sub-chunk rows (default 128 -- MUST stay
//                         a multiple of 64: KDA prefill launches are
//                         bitwise state-carry points on the 64 grid
//                         (INV-KDA-CARRY), so an off-grid sub-chunk makes
//                         the identity gate tolerance-equal, not bitwise)
//   GLM53_GG_STRATEGY=int|dequant  expert-GEMM route override (kernel-family
//                         bisection)
//   GLM53_TQ=1            TD-GLM5-TQ-BACKEND-UNWIRED: boot the TurboQuant
//                         MLA 4-bit backend (258 B NoPE rows vs snapmla's
//                         516 B).  Lossy codec: legs mint their own
//                         references (never diff TQ vs snapmla tokens);
//                         engagement is proven by the kv_bytes_per_page
//                         fingerprint (4128 vs 8256), trap #12.
//   GLM53_TP=2            GF3.10: boot BOTH visible 5090s in TP=2
//                         (replicated KV; KDA tensors head-sharded per
//                         GF3.3, per-rank state slots per GF3.8, one
//                         o_proj allreduce per KDA layer — INV-KDA-TP).
//                         Experts stay pinned to GPU 0 (the V4-2c identity
//                         configuration: expert placement is independent
//                         and EP spread is a known trajectory-fork source,
//                         so the token-identity gates never engage it).
//                         Identity protocol: run each gate once with
//                         GLM53_TP unset (TP=1) and once with GLM53_TP=2
//                         and diff the logged token ids — the INV-V4-TP
//                         gate shape applied to this arch.
//   GLM53_EP=2            TD-GLM53-EP4-DEGENERATE-GENERATION: boot ONE TP
//                         rank (tp_array [0]) plus ONE EXPERT-ONLY 5090
//                         (roles [expert_streaming]) — the EP-beyond-TP
//                         shape autoconfig derives and that nothing else
//                         in this repo exercises. Enables
//                         deterministic_ep_combine so the placement-
//                         invariant per-slot combine makes the identity
//                         BITWISE (INV-MOE-EP-XTP clause 8). Consumed by
//                         EpBeyondTpTokenIdentity, which mints its own
//                         reference on the SAME boot. Like GLM53_TP=2 it
//                         indexes VISIBLE devices, so run it with
//                         CUDA_VISIBLE_DEVICES pointing at the two 5090s.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>   // GF3.12 ckpt gate: memcpy on the header probe
#include <iterator>  // GF3.12 ckpt gate: istreambuf_iterator (slurp)
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

namespace {
// "The capital of France is" under the GLM tokenizer (verified against
// test-data/GLM-5.3-Flash/tokenizer.json; NO BOS prepended -- the same
// tokenization the GLM-5.2 gate uses).
const std::vector<uint32_t> kFrancePrompt = {785, 6722, 315, 9621, 374};
// " Paris" -- the semantic smoke expectation for test 1 only. This is NOT a
// minted golden (no llama.cpp reference exists for GLM-5.3-Flash); it is
// "any correct completion of 'The capital of France is'".
constexpr uint32_t kTokParis = 12089;

// Model shape asserts (spec/GLM-5.3-FLASH-MODELINFO.md).
constexpr int kVocabSize = 154880;
constexpr int kNumLayers = 45;          // MTP block (46th) is NOT walked

const char* kConfigRel = "/test-data/config/glm53_flash_gguf.json";
const char* kGgufAbs =
    "/srv/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/"
    "GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf";
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

// Number of visible SM120+ devices with >= min_gb total VRAM (the EP gate
// needs TWO: one TP rank + one expert-only host).
static int visible_sm120_count(double min_gb) {
    int count = 0, ok = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    for (int i = 0; i < count; ++i) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   i) != cudaSuccess || major < 12)
            continue;
        size_t freeb = 0, totalb = 0;
        if (cudaSetDevice(i) != cudaSuccess) continue;
        if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) continue;
        if (static_cast<double>(totalb) / (1024.0 * 1024 * 1024) >= min_gb)
            ++ok;
    }
    return ok;
}

class Glm53FlashGolden : public ::testing::Test {
protected:
    // NOTE: the engine construction below is deliberately UNBOUNDED in time
    // -- the boot opens 186 GiB of GGUF shards and stages the routed set; it
    // is synchronous here and allowed to take minutes. Only the per-command
    // waits carry a deadline (see wait(), 600 s).
    void start_engine() {
        const std::string src = LAYERSTORM_SOURCE_DIR;
        const std::string cfg_path = src + kConfigRel;
        ASSERT_TRUE(fs::exists(cfg_path)) << "missing config " << cfg_path;
        ASSERT_TRUE(fs::exists(kGgufAbs)) << "missing GGUF " << kGgufAbs;

        std::ifstream f(cfg_path);
        nlohmann::json j = nlohmann::json::parse(f);
        j["model"]["weights_path"] = kGgufAbs;

        // GF3.10 (GLM53_TP=2): boot both visible 5090s in TP=2 with
        // REPLICATED KV (sharded KV is fail-closed for glm5_next -- the
        // KDA recurrent state is whole-sequence). The shipped config is
        // the TP=1 single-5090 reference shape.
        tp_ = 1;
        if (const char* t = std::getenv("GLM53_TP"))
            if (std::atoi(t) == 2) tp_ = 2;
        if (tp_ == 2) {
            j["hardware"]["gpus"] = nlohmann::json::array(
                {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 30},
                  {"pcie_gen", 5}, {"pcie_width", 16}, {"numa_node", 0}},
                 {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 30},
                  {"pcie_gen", 5}, {"pcie_width", 16}, {"numa_node", 1}}});
            j["hardware"]["tp_array"] = nlohmann::json::array({0, 1});
            j["hardware"]["dcp_kv_mode"] = "replicated";
            j["parallelism"]["tensor_parallelism"] = 2;
        }

        // TD-GLM53-EP4-DEGENERATE-GENERATION (GLM53_EP=2): ONE TP rank plus
        // one EXPERT-ONLY GPU — the topology on which the routed EP combine
        // used to drop every expert placed beyond the TP set. Mutually
        // exclusive with GLM53_TP=2 (that arm keeps EP == TP by design).
        ep_gpus_ = {0};
        if (tp_ == 1) {
            if (const char* e = std::getenv("GLM53_EP"))
                if (std::atoi(e) == 2) {
                    j["hardware"]["gpus"] = nlohmann::json::array(
                        {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 30},
                          {"pcie_gen", 5}, {"pcie_width", 16},
                          {"numa_node", 0},
                          {"roles", nlohmann::json::array(
                              {"attention", "resident", "expert_streaming"})}},
                         {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 30},
                          {"pcie_gen", 5}, {"pcie_width", 16},
                          {"numa_node", 1},
                          {"roles", nlohmann::json::array(
                              {"expert_streaming"})}}});
                    j["hardware"]["tp_array"] = nlohmann::json::array({0});
                    // Placement-INVARIANT combine: the per-slot canonical
                    // reduce makes "expert on rank0" and "expert on the extra
                    // host" bit-identical, so the gate below is exact rather
                    // than a valid-trajectory comparison.
                    j["compute"]["deterministic_ep_combine"] = true;
                    ep_gpus_ = {0, 1};
                }
        }

        // Speculation fully OFF -- these are the PLAIN reference legs.
        // (has_dspark() keys on the METHOD, not `enabled`, so clear both
        // when the shipped config carries a method key.)
        j["speculation"]["enabled"] = false;
        if (j["speculation"].contains("method"))
            j["speculation"]["method"] = "none";

        j["memory"]["vram_safety_margin_gb"] = 3.0;
        // Small growth chunk: the gates need only tens of tokens.
        j["memory"]["kv_cache"]["page_growth_chunk_tokens"] = 16;
        // One-seq gates -- do not over-provision the demand-driven pools
        // (and, for glm5_next, the KDA state slots).
        j["serving"]["max_concurrent_requests"] = 2;

        // Debug lever: expert GEMM route (kernel-family bisection).
        if (const char* s = std::getenv("GLM53_GG_STRATEGY"))
            j["quantization"]["gguf_strategy"] = s;

        // GLM53_TQ=1 (TD-GLM5-TQ-BACKEND-UNWIRED): TurboQuant MLA 4-bit
        // backend at the GLM5N NoPE geometry -- 258 B KV rows (256 B packed
        // 4-bit c_kv + 2 B FP16 norm; NO rope tail, qk_rope_head_dim = 0)
        // vs SnapMLA's 516 B.  TQ is LOSSY by design (TQ_MSE codebooks), so
        // a TQ leg must NEVER be diffed token-for-token against a SnapMLA
        // leg: tests 2/3 mint their reference IN-RUN on the SAME arm (the
        // right protocol for a lossy codec), and test 1's " Paris" anchor
        // is codec-independent semantics.  Composes with GLM53_TP=2 /
        // GLM53_EP=2 like the GLM-5.2 GLM52_TQ switch it mirrors
        // (glm52_gguf_golden_test.cpp:144).
        if (const char* tq = std::getenv("GLM53_TQ"); tq && *tq == '1') {
            j["compute"]["attention_backend"] = "turboquant_mla";
            tq_ = true;
        }

        config_path_ = "/tmp/glm53_flash_golden_config.json";
        { std::ofstream o(config_path_); o << j.dump(2); }

        hidden_size_ = j["model"]["hidden_size"].get<int>();
        // Layers [0, first_moe_layer_) are DENSE MLP -- no routing export,
        // no FETCH_AND_RUN_MOE. GLM-5.3-Flash: first_k_dense_replace = 3.
        // This is the ONLY MoE-vs-dense axis; the attention-type axis
        // (KDA vs sparse MLA) is independent of it.
        first_moe_layer_ = j["model"].value("first_k_dense_replace", 0);

        auto backends = ldam::default_backends();
        backends.skip_cuda_graphs = true;
        const auto t0 = std::chrono::steady_clock::now();
        engine_ = std::make_unique<ldam::Engine>(config_path_,
                                                 std::move(backends));
        const double boot_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[glm53-golden] engine boot (186 GiB GGUF): %.1f s\n",
                boot_s);

        auto& info = engine_->info();
        auto* base = reinterpret_cast<uint8_t*>(info.ipc_base);
        cmd_ring_ = std::make_unique<lipc::CommandRing>(
            base + info.cmd_ring_offset);
        cmp_ring_ = std::make_unique<lipc::CompletionRing>(
            base + info.cmp_ring_offset);
        sideband_ = base + info.sideband_offset;
        num_layers_ = info.num_layers;
        num_experts_ = info.num_experts;
        vocab_size_ = info.vocab_size;
        EXPECT_EQ(vocab_size_, kVocabSize);
        EXPECT_EQ(num_layers_, kNumLayers)
            << "glm5_next exports num_hidden_layers (MTP excluded)";
        // Anti-vacuous HYBRID fingerprint (trap #12): the engine must have
        // published the per-layer attention-type map with BOTH kinds --
        // 3 = linear/KDA, 4 = sparse NoPE MLA. If this is all-zero the boot
        // silently degraded to a homogeneous-attention arch and every
        // "hybrid" claim below would be vacuous.
        EXPECT_EQ(static_cast<int>(info.attention_types[0]), 3)
            << "layer 0 must be linear_attention (KDA)";
        EXPECT_EQ(static_cast<int>(info.attention_types[3]), 4)
            << "layer 3 must be sparse NoPE MLA";
        // Anti-vacuous BACKEND fingerprint (trap #12: a "TQ gate" that
        // silently booted the default backend exercises zero TQ code and
        // passes trivially).  The engine-published KV page size IS the row
        // geometry: page_size_tokens = 16, so snapmla 516*16 = 8256 and
        // turboquant_mla 258*16 = 4128 (vram_allocator.cpp
        // kv_bytes_per_token; NoPE => no rope tail).  A silently-ignored
        // GLM53_TQ (or a backend fallback) fails HERE, not in a text diff.
        EXPECT_EQ(info.kv_bytes_per_page, tq_ ? 4128 : 8256)
            << "published KV page bytes do not match the "
            << (tq_ ? "turboquant_mla 258 B/row" : "snapmla 516 B/row")
            << " NoPE geometry -- wrong attention backend booted?";

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
    // 600 s: this boot path is COLDER than the V4 gate's (240 s) -- the
    // first touch of each routed expert streams from the GGUF page cache /
    // Gen3-capped NVMe.
    bool wait(lipc::Completion& out, uint32_t expected, int timeout_s = 600) {
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

    // -- Leg 1: teacher-forced reference -----------------------------------
    // One B==1 decode step on the production seam: embedding -> per layer
    // (RUN_ATTENTION with fused gate + routing export on routed layers ->
    // FETCH_AND_RUN_MOE over the routed experts) -> head -> greedy sample.
    // Layers 0-2 are DENSE (first_k_dense_replace = 3): gate OFF, no MoE.
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
            const auto t_attn0 = std::chrono::steady_clock::now();
            send(a);
            EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "attn L" << layer << " pos " << token_pos;
            note_attn_wall(layer, std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_attn0).count());
            if (::testing::Test::HasFailure()) return 0;

            if (!is_moe) {
                // Dense MLP (layers 0-2): the MoE dispatch is still what
                // runs the FFN AND commits moe_buf -> attn_buf (the
                // attention residual lands in moe_buf; without the commit
                // the next layer reads a STALE attn_buf and the layer
                // contributes exactly nothing — first-boot finding).
                auto m = make_cmd(lipc::D_B_CMD_RUN_MOE);
                m.run_moe.layer_idx = static_cast<uint32_t>(layer);
                m.run_moe.num_seqs = 1;
                m.run_moe.moe_mode = 0;
                m.run_moe.apply_residual_correction = 0;
                m.run_moe.store_gating_output = 0;
                m.run_moe.emit_checkpoint = 0;
                send(m);
                EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                    << "dense mlp L" << layer;
                if (::testing::Test::HasFailure()) return 0;
                continue;
            }

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
                // TP=1 default: all experts on GPU 0. With ep_spread_ (the
                // EP-beyond-TP gate) the routed set round-robins over every
                // expert host, so experts land on GPUs that are NOT DCP ranks
                // and must be folded back by INV-MOE-EP-XTP.
                entries[count].gpu_idx    = ep_spread_
                    ? static_cast<uint32_t>(ep_gpus_[count % ep_gpus_.size()])
                    : 0u;
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
            fprintf(stderr, "[glm53-golden] pos %u fed %u -> argmax %u "
                    "(top1 %.4f)\n", pos, prompt[i], tok, p);
            if (::testing::Test::HasFailure()) break;
        }
        if (gen_out) gen_out->push_back(tok);
        for (int g = 0; g < gen_extra && !::testing::Test::HasFailure();
             ++g, ++pos) {
            // GF3.10: log per-step top1 too — the TP identity analysis needs
            // the greedy margin at any TP=1-vs-TP=2 divergence point.
            float gp = 0.f;
            tok = decode_step(tok, seq_id, pos, &gp);
            fprintf(stderr, "[glm53-golden] gen +%d -> token %u (top1 %.4f)\n",
                    g + 1, tok, gp);
            if (gen_out) gen_out->push_back(tok);
        }
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[glm53-golden] seq %lu: %u steps in %.1f s "
                "(%.1f s/token)\n", (unsigned long)seq_id, pos, wall,
                pos ? wall / pos : 0.0);
        report_attn_wall("teacher-forced");
        free_sequence(seq_id);
        return tok;
    }

    // -- Leg 2: chunked prefill --------------------------------------------
    // One CHUNKED prefill step over n prompt tokens at positions
    // [pos0, pos0+n) -- the production prefill command shape (one batch
    // descriptor per prompt position, is_prefill=1 + chunk fields; the
    // executor runs its internal per-row loop), with ONE expert-UNION
    // FETCH_AND_RUN_MOE per routed layer (dedup in FIRST-OCCURRENCE order --
    // the bridge's fetch_moe_from_export contract). No output head.
    // On the KDA layers this is the state-CARRY path: the recurrent state
    // must advance across the chunk exactly as single-token steps advance it.
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
            const bool is_moe = layer >= first_moe_layer_;

            auto a = make_cmd(lipc::D_B_CMD_RUN_ATTENTION);
            a.run_attention.layer_idx    = static_cast<uint32_t>(layer);
            a.run_attention.num_seqs     = n;
            a.run_attention.is_prefill   = 1;
            a.run_attention.chunk_start  = pos0;
            a.run_attention.chunk_len    = n;
            a.run_attention.use_graph    = 0;
            a.run_attention.emit_gating  = is_moe ? 1 : 0;
            a.run_attention.store_gating = is_moe ? 1 : 0;
            send(a);
            ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "prefill attn L" << layer << " pos0 " << pos0;

            if (!is_moe) {
                // Dense MLP (layers 0-2): the MoE dispatch is still what
                // runs the FFN AND commits moe_buf -> attn_buf (the
                // attention residual lands in moe_buf; without the commit
                // the next layer reads a STALE attn_buf and the layer
                // contributes exactly nothing — first-boot finding).
                auto m = make_cmd(lipc::D_B_CMD_RUN_MOE);
                m.run_moe.layer_idx = static_cast<uint32_t>(layer);
                m.run_moe.num_seqs = n;
                m.run_moe.moe_mode = 0;
                m.run_moe.apply_residual_correction = 0;
                m.run_moe.store_gating_output = 0;
                m.run_moe.emit_checkpoint = 0;
                send(m);
                ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                    << "dense mlp L" << layer;
                continue;
            }

            const auto* hdr = reinterpret_cast<const lipc::RoutingExportHeader*>(
                sideband_ + lipc::IpcLayout::kRoutingExportOff);
            ASSERT_EQ(hdr->num_tokens, n) << "chunk routing export L" << layer;
            ASSERT_EQ(hdr->layer_idx, static_cast<uint32_t>(layer));
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
                // ep_spread_ (EP-beyond-TP gate): the chunk's expert UNION
                // round-robins over every expert host — the batched
                // FETCH_AND_RUN finalize must broadcast + dispatch + fold
                // the extra host's share (INV-MOE-EP-XTP).
                entries[count].gpu_idx    = ep_spread_
                    ? static_cast<uint32_t>(ep_gpus_[count % ep_gpus_.size()])
                    : 0u;
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
        fprintf(stderr, "[glm53-golden] prefill chunk [%u, %u) done\n",
                pos0, pos0 + n);
    }

    // Chunked-prefill the first N-1 prompt tokens (GLM53_PREFILL_CHUNK-sized
    // chunks; default = one chunk), then ONE decode step on the last token
    // + optional greedy continuation -- the production serving shape.
    uint32_t run_prompt_prefill(uint64_t seq_id,
                                const std::vector<uint32_t>& prompt,
                                int gen_extra, std::vector<uint32_t>* gen_out,
                                float* top1 = nullptr) {
        create_sequence(seq_id, static_cast<uint32_t>(prompt.size()));
        const uint32_t pre = static_cast<uint32_t>(prompt.size()) - 1;
        // One chunk by default, capped at the 512-descriptor/sideband bound.
        uint32_t chunk = std::min<uint32_t>(pre, 512);
        if (const char* c = std::getenv("GLM53_PREFILL_CHUNK"))
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
            fprintf(stderr, "[glm53-golden] post-prefill decode argmax=%u "
                    "(top1 %.4f)\n", tok, p);
        }
        if (gen_out) gen_out->push_back(tok);
        for (int g = 0; g < gen_extra && !::testing::Test::HasFailure();
             ++g, ++pos) {
            tok = decode_step(tok, seq_id, pos, nullptr);
            fprintf(stderr, "[glm53-golden] prefill-arm gen +%d -> token %u\n",
                    g + 1, tok);
            if (gen_out) gen_out->push_back(tok);
        }
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[glm53-golden] seq %lu (chunked prefill): %u tokens "
                "in %.1f s\n", (unsigned long)seq_id, pos, wall);
        report_attn_wall("chunked-arm-decode");
        free_sequence(seq_id);
        return tok;
    }

    // -- Leg 3: superchunk -------------------------------------------------
    // One superchunk over n prompt tokens at positions [pos0, pos0+n),
    // processed LAYER-WISE (the TD-PREFILL-MOE-BIG driver shape): embedding
    // per sub-chunk at row_offset, then per layer K attention sub-chunks
    // (superchunk flag + row_offset; the fused gate stores the topk at row
    // offsets, exports unioned here) and ONE E_CMD_FETCH_AND_RUN_MOE_BIG
    // over ALL n rows -- one fetch per unique expert per routed layer.
    // `sub` MUST be a multiple of 64 so every KDA sub-chunk launch starts on
    // the state-carry grid (INV-KDA-CARRY).
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

        // 2. Layer-wise sweep: K sub-chunk attentions -> ONE BIG MoE.
        for (int layer = 0; layer < num_layers_; ++layer) {
            const bool is_moe = layer >= first_moe_layer_;
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
                a.run_attention.emit_gating  = is_moe ? 1 : 0;
                a.run_attention.store_gating = is_moe ? 1 : 0;
                a.run_attention.superchunk   = 1;
                a.run_attention.row_offset   = off;
                send(a);
                ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                    << "superchunk attn L" << layer << " off=" << off;

                if (!is_moe) continue;  // layers 0-2: dense MLP

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
                    // ep_spread_ (EP-beyond-TP gate): the superchunk union
                    // round-robins over every expert host — the MOE_BIG
                    // single-shot dispatch (the production EP4 serving
                    // prefill arm) must fold the extra host's share
                    // (INV-MOE-EP-XTP).
                    entries[count].gpu_idx    = ep_spread_
                        ? static_cast<uint32_t>(
                              ep_gpus_[count % ep_gpus_.size()])
                        : 0u;
                    ++count;
                }
            }
            if (!is_moe) {
                // Dense MLP (layers 0-2): the MoE dispatch is still what
                // runs the FFN AND commits moe_buf -> attn_buf (the
                // attention residual lands in moe_buf; without the commit
                // the next layer reads a STALE attn_buf and the layer
                // contributes exactly nothing — first-boot finding).
                auto m = make_cmd(lipc::D_B_CMD_RUN_MOE);
                m.run_moe.layer_idx = static_cast<uint32_t>(layer);
                m.run_moe.num_seqs = n;
                m.run_moe.moe_mode = 0;
                m.run_moe.apply_residual_correction = 0;
                m.run_moe.store_gating_output = 0;
                m.run_moe.emit_checkpoint = 0;
                send(m);
                ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                    << "superchunk dense mlp L" << layer;
                continue;
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
        fprintf(stderr, "[glm53-golden] superchunk [%u, %u) sub=%u done\n",
                pos0, pos0 + n, sub);
    }

    // Superchunk-prefill the first N-1 prompt tokens (superchunks up to the
    // engine's moe_batch_capacity, sub-chunks of `sub` rows), then ONE
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
            fprintf(stderr, "[glm53-golden] post-superchunk decode argmax=%u "
                    "(top1 %.4f)\n", tok, p);
        }
        if (gen_out) gen_out->push_back(tok);
        for (int g = 0; g < gen_extra && !::testing::Test::HasFailure();
             ++g, ++pos) {
            tok = decode_step(tok, seq_id, pos, nullptr);
            fprintf(stderr, "[glm53-golden] superchunk-arm gen +%d -> token "
                    "%u\n", g + 1, tok);
            if (gen_out) gen_out->push_back(tok);
        }
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[glm53-golden] seq %lu (superchunk prefill): %u "
                "tokens in %.1f s\n", (unsigned long)seq_id, pos, wall);
        report_attn_wall("superchunk-arm-decode");
        free_sequence(seq_id);
        return tok;
    }

    // GF3.12: checkpoint commands. The path rides LS_SEQ_CKPT_PATH (the
    // dispatcher reads it per command — settable in-process per call).
    void seq_snapshot(uint64_t seq_id, uint32_t token_count,
                      const std::string& path) {
        setenv("LS_SEQ_CKPT_PATH", path.c_str(), 1);
        lipc::Completion cmp{};
        auto c = make_cmd(lipc::CMD_SEQ_SNAPSHOT);
        c.seq_ckpt.seq_id = seq_id;
        c.seq_ckpt.token_count = token_count;
        send(c);
        ASSERT_TRUE(wait(cmp, lipc::CMP_SEQ_OP_DONE))
            << "seq_snapshot seq " << seq_id << " @" << token_count;
    }
    void seq_restore(uint64_t seq_id, const std::string& path) {
        setenv("LS_SEQ_CKPT_PATH", path.c_str(), 1);
        lipc::Completion cmp{};
        auto c = make_cmd(lipc::CMD_SEQ_RESTORE);
        c.seq_ckpt.seq_id = seq_id;
        send(c);
        ASSERT_TRUE(wait(cmp, lipc::CMP_SEQ_OP_DONE))
            << "seq_restore seq " << seq_id;
    }
    // P-29 step 24 (LS_KDA_PREFIX_CKPT): host-RAM prefix checkpoint +
    // truncating fork (the divergence-reuse pair the gate below proves).
    void kda_ckpt(uint64_t seq_id, uint32_t pos) {
        lipc::Completion cmp{};
        auto c = make_cmd(lipc::D_CMD_KDA_CKPT);
        c.kda_anchor.seq_id = seq_id;
        c.kda_anchor.pos = pos;
        send(c);
        ASSERT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
            << "kda_ckpt seq " << seq_id << " @" << pos;
        ASSERT_EQ(cmp.status, 0u) << "kda_ckpt skipped (host alloc?)";
        EXPECT_GT(cmp.compute.data_bytes, 0u)
            << "kda_ckpt captured nothing (anti-vacuous)";
    }
    void seq_fork_truncated(uint64_t src, uint64_t dst, uint32_t prefix,
                            bool expect_refusal = false) {
        lipc::Completion cmp{};
        auto c = make_cmd(lipc::CMD_SEQ_FORK);
        c.seq_fork.src_seq_id = src;
        c.seq_fork.dst_seq_id = dst;
        c.seq_fork.prefix_len = prefix;
        c.seq_fork.reserve_tokens = 0;
        send(c);
        if (expect_refusal) {
            // Raw wait: CMP_ERROR is the EXPECTED outcome here.
            auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::seconds(120);
            while (std::chrono::steady_clock::now() < deadline) {
                if (cmp_ring_->try_read(&cmp)) {
                    if (cmp.cmp_type
                            == static_cast<uint32_t>(lipc::CMP_CHECKPOINT))
                        continue;
                    EXPECT_EQ(cmp.cmp_type,
                              static_cast<uint32_t>(lipc::CMP_ERROR))
                        << "truncating fork without a checkpoint must be "
                           "REFUSED (negative control)";
                    return;
                }
                std::this_thread::yield();
            }
            ADD_FAILURE() << "timeout waiting for fork refusal";
            return;
        }
        ASSERT_TRUE(wait(cmp, lipc::CMP_SEQ_OP_DONE))
            << "seq_fork " << src << " -> " << dst << " @" << prefix;
    }
    static std::vector<char> slurp(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        return std::vector<char>((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
    }

    static bool gated_off() {
        const char* g = std::getenv("GLM53_GOLDEN");
        return !(g && std::string(g) == "1");
    }
    static bool env_is(const char* name, const char* val) {
        const char* e = std::getenv(name);
        return e && std::string(e) == val;
    }

    // GF3.10 tax instrument: cumulative wall inside the RUN_ATTENTION
    // command waits of DECODE steps, split by layer type (attention_types:
    // 3 = KDA linear, 4 = sparse MLA). Includes the harness's IPC
    // round-trip per command, which is IDENTICAL across the TP=1/TP=2
    // arms, so the ARM DELTA isolates the per-layer TP cost (for KDA:
    // the GF3.10 o_proj allreduce + the halved-head GEMM/scan work).
    void note_attn_wall(int layer, double seconds) {
        const auto& info = engine_->info();
        if (layer >= 0 && layer < num_layers_
            && static_cast<int>(info.attention_types[layer]) == 3) {
            attn_kda_s_ += seconds; ++attn_kda_n_;
        } else {
            attn_mla_s_ += seconds; ++attn_mla_n_;
        }
    }
    void report_attn_wall(const char* tag) {
        fprintf(stderr, "[glm53-tp-tax] %s tp=%d decode attention walls: "
                "KDA %.3f s / %ld cmds (%.3f ms/cmd), MLA %.3f s / %ld "
                "cmds (%.3f ms/cmd)\n", tag, tp_,
                attn_kda_s_, attn_kda_n_,
                attn_kda_n_ ? attn_kda_s_ * 1e3 / attn_kda_n_ : 0.0,
                attn_mla_s_, attn_mla_n_,
                attn_mla_n_ ? attn_mla_s_ * 1e3 / attn_mla_n_ : 0.0);
        attn_kda_s_ = attn_mla_s_ = 0.0;
        attn_kda_n_ = attn_mla_n_ = 0;
    }

    std::unique_ptr<ldam::Engine> engine_;
    std::unique_ptr<lipc::CommandRing> cmd_ring_;
    std::unique_ptr<lipc::CompletionRing> cmp_ring_;
    uint8_t* sideband_ = nullptr;
    std::string config_path_;
    uint32_t cmd_seq_ = 1;
    int num_layers_ = 0, num_experts_ = 0, first_moe_layer_ = 0;
    int vocab_size_ = 0, hidden_size_ = 0;
    uint32_t hidden_buf_id_ = 0, logits_buf_id_ = 0;
    int tp_ = 1;              // GLM53_TP=2 -> TP=2 boot (GF3.10)
    bool tq_ = false;         // GLM53_TQ=1 -> turboquant_mla backend
    // TD-GLM53-EP4-DEGENERATE-GENERATION: the expert-host set this boot may
    // place routed experts on, and whether the current leg spreads over it.
    std::vector<int> ep_gpus_{0};
    bool ep_spread_ = false;
    double attn_kda_s_ = 0.0, attn_mla_s_ = 0.0;
    long attn_kda_n_ = 0, attn_mla_n_ = 0;
};

// Test 1 -- SEMANTIC SMOKE: the hybrid stack boots and produces a coherent
// completion. "The capital of France is" -> " Paris" (12089).
//
// NOT a minted golden: GLM-5.3-Flash has no llama.cpp ground truth here, so
// this expectation is "any correct completion of 'The capital of France
// is'", chosen because the answer is unambiguous in every arm. It fails
// loudly when hybrid execution is broken enough to garble the output (a
// dead KDA layer, a mis-carried recurrent state, a dropped MoE union) -- it
// is NOT evidence of bitwise numerics. Tests 2 and 3 are the exact gates;
// they mint their reference in-run on the SAME arm.
// Env: GLM53_GOLDEN=1; GLM53_GOLDEN_GEN=N (default 8).
TEST_F(Glm53FlashGolden, Golden) {
    if (gated_off()) GTEST_SKIP() << "set GLM53_GOLDEN=1 to run";
    if (find_big_sm120(24.0) < 0)
        GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "GLM-5.3-Flash GGUF not present";

    start_engine();
    if (::testing::Test::HasFailure()) return;
    fprintf(stderr, "[glm53-golden] layers=%d experts=%d vocab=%d "
            "first_moe_layer=%d\n", num_layers_, num_experts_, vocab_size_,
            first_moe_layer_);

    int gen_extra = 8;
    if (const char* g = std::getenv("GLM53_GOLDEN_GEN")) gen_extra = atoi(g);

    float top1 = 0.f;
    std::vector<uint32_t> gen;
    const uint32_t tok = run_prompt(1, kFrancePrompt, gen_extra, &gen, &top1);
    if (::testing::Test::HasFailure()) return;

    fprintf(stderr, "[glm53-golden] next_token=%u top1=%.4f expected %u "
            "(' Paris')\n[glm53-golden] generation ids:",
            gen.empty() ? tok : gen.front(), top1, kTokParis);
    for (uint32_t t : gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n");
    EXPECT_EQ(gen.empty() ? tok : gen.front(), kTokParis)
        << "semantic smoke only: any CORRECT completion of 'The capital of "
           "France is' is expected here (12089 = ' Paris'); this is not a "
           "minted golden, so a mismatch means the hybrid stack is producing "
           "incoherent text, not that numerics drifted by an ulp";
}

// Test 2 -- CHUNKED-PREFILL TOKEN IDENTITY. Chunked prefill-then-decode must
// be TOKEN-IDENTICAL to the teacher-forced per-token reference. One
// teacher-forced run mints the reference greedy trajectory; the prompt is
// then extended with the first K reference tokens so the prefill chunk
// exceeds the ring-clamped verify bound (8 rows) and exercises the
// snapshot-skip + staging-drain prefill path; the chunked arm must
// reproduce the remaining reference continuation exactly.
//
// For the hybrid stack this is the KDA state-carry gate: the 34 linear
// layers advance a recurrent state, and a multi-token prefill launch must
// leave that state exactly where per-token steps would (INV-KDA-CARRY)
// while the 11 sparse-MLA layers reproduce their KV. A divergence at +0
// means the prefilled state/KV differs from the teacher-forced one.
// Env: GLM53_PREFILL=1 (with GLM53_GOLDEN=1); GLM53_PREFILL_GEN (default 12,
// clamped >= 10); GLM53_PREFILL_CHUNK chunk override.
TEST_F(Glm53FlashGolden, ChunkedPrefillTokenIdentity) {
    if (gated_off()) GTEST_SKIP() << "set GLM53_GOLDEN=1 to run";
    if (!env_is("GLM53_PREFILL", "1"))
        GTEST_SKIP() << "set GLM53_PREFILL=1 to run the chunked-prefill gate";
    if (find_big_sm120(24.0) < 0)
        GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "GLM-5.3-Flash GGUF not present";

    start_engine();
    if (::testing::Test::HasFailure()) return;

    int gen_extra = 12;
    if (const char* g = std::getenv("GLM53_PREFILL_GEN")) gen_extra = atoi(g);
    if (gen_extra < 10) gen_extra = 10;

    // Reference: teacher-forced greedy trajectory, minted in-run on THIS arm.
    std::vector<uint32_t> ref_gen;
    const auto t_ref0 = std::chrono::steady_clock::now();
    run_prompt(21, kFrancePrompt, gen_extra, &ref_gen);
    const double ref_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_ref0).count();
    if (::testing::Test::HasFailure()) return;
    ASSERT_EQ(ref_gen.size(), static_cast<size_t>(gen_extra) + 1);
    EXPECT_EQ(ref_gen.front(), kTokParis) << "semantic smoke (see Golden)";

    // Long prompt = prompt + first K reference tokens -> the prefill chunk
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
    fprintf(stderr, "[glm53-golden] prefill-identity: ref");
    for (int i = 0; i <= pf_extra; ++i) fprintf(stderr, " %u", ref_gen[K + i]);
    fprintf(stderr, " | chunked");
    for (uint32_t t : pf_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n[glm53-golden] walls: teacher-forced %.1f s vs chunked "
            "%.1f s (%zu-token prompt arm, top1 %.4f)\n", ref_wall, pf_wall,
            long_prompt.size(), pf_top1);
    for (int i = 0; i <= pf_extra; ++i)
        EXPECT_EQ(pf_gen[static_cast<size_t>(i)], ref_gen[K + i])
            << "chunked-prefill continuation diverges at +" << i;
}

// Test 3 -- SUPERCHUNK TOKEN IDENTITY: the superchunk-prefilled continuation
// must EXACTLY equal BOTH the teacher-forced reference and the chunked-
// prefill arm. Exercises batch attention with row_offset placement, the
// layer-wise sweep (all K sub-chunks of a layer before the next layer) and
// the per-layer FETCH_AND_RUN_MOE_BIG union in one gate -- and, on the
// hybrid stack, proves the KDA recurrent state carries correctly across the
// layer-wise sweep order, the sharpest reordering the engine performs.
//
// Env: GLM53_SUPERCHUNK=1 (with GLM53_GOLDEN=1); GLM53_SC_GEN (default 40,
// clamped >= 26) reference continuation length; GLM53_SC_SUB (default 128)
// sub-chunk rows -- keep it a MULTIPLE OF 64 (INV-KDA-CARRY: a KDA prefill
// launch that starts off the 64 grid inside a prompt makes the arms
// tolerance-equal rather than bitwise, which this EXPECT_EQ cannot express).
// With the default 40-token reference the superchunk body is a single
// sub-chunk; raise GLM53_SC_GEN past ~132 to force a second, 64-aligned
// sub-chunk boundary.
TEST_F(Glm53FlashGolden, SuperchunkPrefillTokenIdentity) {
    if (gated_off()) GTEST_SKIP() << "set GLM53_GOLDEN=1 to run";
    if (!env_is("GLM53_SUPERCHUNK", "1"))
        GTEST_SKIP() << "set GLM53_SUPERCHUNK=1 to run the superchunk gate";
    if (find_big_sm120(24.0) < 0)
        GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "GLM-5.3-Flash GGUF not present";

    start_engine();
    if (::testing::Test::HasFailure()) return;

    int gen_extra = 40;
    if (const char* g = std::getenv("GLM53_SC_GEN")) gen_extra = atoi(g);
    if (gen_extra < 26) gen_extra = 26;   // K >= 22 -> body > the 16-row bound
    uint32_t sub = 128;                   // multiple of 64 (INV-KDA-CARRY)
    if (const char* s = std::getenv("GLM53_SC_SUB"))
        if (uint32_t v = static_cast<uint32_t>(std::atoi(s)); v > 0)
            sub = std::min<uint32_t>(v, 512);

    // Reference: teacher-forced greedy trajectory, minted in-run.
    std::vector<uint32_t> ref_gen;
    const auto t_ref0 = std::chrono::steady_clock::now();
    run_prompt(41, kFrancePrompt, gen_extra, &ref_gen);
    const double ref_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_ref0).count();
    if (::testing::Test::HasFailure()) return;
    ASSERT_EQ(ref_gen.size(), static_cast<size_t>(gen_extra) + 1);
    EXPECT_EQ(ref_gen.front(), kTokParis) << "semantic smoke (see Golden)";

    const int K = gen_extra - 4;
    std::vector<uint32_t> long_prompt = kFrancePrompt;
    for (int i = 0; i < K; ++i) long_prompt.push_back(ref_gen[i]);
    const int sc_extra = gen_extra - K;   // 4

    // Arm A: chunked prefill (per-command chunk, auto-batch body).
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
    fprintf(stderr, "[glm53-golden] superchunk-identity: ref");
    for (int i = 0; i <= sc_extra; ++i) fprintf(stderr, " %u", ref_gen[K + i]);
    fprintf(stderr, " | chunked");
    for (uint32_t t : ck_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, " | superchunk");
    for (uint32_t t : sc_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n[glm53-golden] walls: teacher %.1f s, chunked %.1f s, "
            "superchunk %.1f s (%zu-token prompt arm, sub=%u)\n", ref_wall,
            ck_wall, sc_wall, long_prompt.size(), sub);
    for (int i = 0; i <= sc_extra; ++i) {
        EXPECT_EQ(sc_gen[static_cast<size_t>(i)], ref_gen[K + i])
            << "superchunk continuation diverges from teacher-forced at +"
            << i;
        EXPECT_EQ(sc_gen[static_cast<size_t>(i)], ck_gen[static_cast<size_t>(i)])
            << "superchunk continuation diverges from chunked at +" << i;
    }
}

// Test 4 -- GF3.12 CHECKPOINT / RESTORE BIT-EXACT CONTINUATION. The PLAN
// verify bar, proven rather than asserted: a glm5_next sequence snapshotted
// mid-run (ckpt format v3: KV pages + indexer-K + the per-rank WHOLE-SLOT
// KDA state + frontier) and restored into a FRESH sequence must continue
// TOKEN-IDENTICALLY to the uninterrupted run -- and, stronger, an
// end-of-run snapshot of the restored sequence must be BYTE-IDENTICAL to
// the uninterrupted run's end-of-run snapshot (same KV bytes, same indexer
// bytes, same fp32 state bytes, same frontier: the whole-slot byte copy
// round-trips exactly, INV-KDA-CARRY / INV-KDA-STATE (f)).
//
// Restart protocol (what an orchestrator would do): the interrupted arm
// records its last sampled token alongside the checkpoint; the restored
// arm feeds that token first, then continues greedy on its own argmaxes.
// Env: GLM53_GOLDEN=1 GLM53_CKPT=1; GLM53_CKPT_GEN=N (default 8).
TEST_F(Glm53FlashGolden, CheckpointRestoreBitExactContinuation) {
    if (gated_off()) GTEST_SKIP() << "set GLM53_GOLDEN=1 to run";
    if (!env_is("GLM53_CKPT", "1"))
        GTEST_SKIP() << "set GLM53_CKPT=1 to run the checkpoint gate";
    if (find_big_sm120(24.0) < 0)
        GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "GLM-5.3-Flash GGUF not present";

    start_engine();
    if (::testing::Test::HasFailure()) return;

    int K = 8;
    if (const char* g = std::getenv("GLM53_CKPT_GEN")) K = atoi(g);
    if (K < 2) K = 2;
    const uint32_t P = static_cast<uint32_t>(kFrancePrompt.size());
    const std::string mid_path = "/tmp/glm53_ckpt_mid.bin";
    const std::string a_end = "/tmp/glm53_ckpt_a_end.bin";
    const std::string b_end = "/tmp/glm53_ckpt_b_end.bin";

    // -- ARM A: uninterrupted reference. Feed the prompt, then K-1 greedy
    // continuation steps (K sampled tokens total; the last is never fed,
    // so the state frontier ends at P+K-1). Snapshot at the end.
    std::vector<uint32_t> a_gen;
    create_sequence(11, P);
    uint32_t tok = 0;
    for (uint32_t i = 0; i < P; ++i)
        tok = decode_step(kFrancePrompt[i], 11, i);
    ASSERT_FALSE(::testing::Test::HasFailure());
    a_gen.push_back(tok);
    for (int g = 1; g < K; ++g) {
        tok = decode_step(tok, 11, P + static_cast<uint32_t>(g) - 1);
        a_gen.push_back(tok);
        ASSERT_FALSE(::testing::Test::HasFailure());
    }
    seq_snapshot(11, P + static_cast<uint32_t>(K) - 1, a_end);
    free_sequence(11);
    ASSERT_FALSE(::testing::Test::HasFailure());
    fprintf(stderr, "[glm53-ckpt] arm A gen:");
    for (uint32_t t : a_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n");

    // The file must actually BE the new format (anti-vacuous): v3.
    {
        auto bytes = slurp(a_end);
        ASSERT_GT(bytes.size(), 8u);
        uint32_t magic = 0, version = 0;
        memcpy(&magic, bytes.data(), 4);
        memcpy(&version, bytes.data() + 4, 4);
        EXPECT_EQ(magic, 0x4B43534CU);
        EXPECT_EQ(version, 3u) << "expected ckpt v3 (replicated KV + KDA "
                                  "state section)";
    }

    // -- ARM B: interrupted at the prompt end. Its own last argmax must
    // match arm A's (same boot, deterministic decode), and is what the
    // restart feeds first.
    create_sequence(12, P);
    uint32_t tok_b = 0;
    for (uint32_t i = 0; i < P; ++i)
        tok_b = decode_step(kFrancePrompt[i], 12, i);
    ASSERT_FALSE(::testing::Test::HasFailure());
    ASSERT_EQ(tok_b, a_gen[0]) << "pre-checkpoint argmax diverged -- the "
                                  "boot is not deterministic; fix that "
                                  "before reading the restore result";
    seq_snapshot(12, P, mid_path);
    free_sequence(12);
    ASSERT_FALSE(::testing::Test::HasFailure());

    // -- Restore into a FRESH sequence and continue.
    create_sequence(13, P);
    seq_restore(13, mid_path);
    ASSERT_FALSE(::testing::Test::HasFailure());
    std::vector<uint32_t> b_gen;
    b_gen.push_back(tok_b);
    tok = tok_b;
    for (int g = 1; g < K; ++g) {
        tok = decode_step(tok, 13, P + static_cast<uint32_t>(g) - 1);
        b_gen.push_back(tok);
        ASSERT_FALSE(::testing::Test::HasFailure());
    }
    seq_snapshot(13, P + static_cast<uint32_t>(K) - 1, b_end);
    fprintf(stderr, "[glm53-ckpt] arm B gen:");
    for (uint32_t t : b_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n");
    free_sequence(13);

    // Token identity of the continuation (INV-PREFIX-CACHE-1 spirit).
    for (int g = 0; g < K; ++g)
        EXPECT_EQ(b_gen[static_cast<size_t>(g)],
                  a_gen[static_cast<size_t>(g)])
            << "restored continuation diverges at +" << g;

    // BYTE identity of the end states: uninterrupted vs
    // snapshot->restore->continue. This is the whole-slot round-trip
    // proof -- KV bytes, indexer bytes, fp32 state bytes and frontier all
    // equal, or the diff offset points at exactly what drifted.
    auto ba = slurp(a_end);
    auto bb = slurp(b_end);
    ASSERT_FALSE(ba.empty());
    ASSERT_EQ(ba.size(), bb.size());
    if (ba != bb) {
        size_t off = 0;
        while (off < ba.size() && ba[off] == bb[off]) ++off;
        ADD_FAILURE() << "end-state checkpoints differ at byte " << off
                      << " of " << ba.size();
    }
    if (!::testing::Test::HasFailure()
        && !env_is("GLM53_CKPT_KEEP", "1")) {
        // GLM53_CKPT_KEEP=1 keeps the files on success — the
        // TD-KDA-STATE-MAPPED-SLABS A/B compares end-state checkpoints
        // ACROSS the carve/mapped modes (they must be byte-identical:
        // the mapping changes where VRAM bytes live, never what the
        // sequence state is).
        std::remove(mid_path.c_str());
        std::remove(a_end.c_str());
        std::remove(b_end.c_str());
    } else if (::testing::Test::HasFailure()) {
        fprintf(stderr, "[glm53-ckpt] FAILURE: files kept for forensics: "
                "%s %s %s\n", mid_path.c_str(), a_end.c_str(),
                b_end.c_str());
    }
}

// Test 4b -- P-29 step 24 (LS_KDA_PREFIX_CKPT): KDA PREFIX-CHECKPOINT
// FORK-REPLAY BIT-EXACT. The GF3.12 divergence-reuse pair proven on the
// real model: a host-RAM checkpoint captured mid-prefill at a 64-aligned
// uniform frontier (D_CMD_KDA_CKPT), consumed by a TRUNCATING fork at
// exactly that position, must replay the tail to a continuation that is
// (a) TOKEN-identical to the uninterrupted run and (b) BYTE-identical in
// its end-of-run ckpt-v3 snapshot (KV bytes + indexer bytes + fp32 state
// bytes + frontier). Identity gate 1 of three (KDA_PREFIX_DESIGN §3.6).
// Negative control: a truncating fork at a NON-checkpoint position is
// refused (the reworked gate never approximates).
// Env: GLM53_GOLDEN=1 GLM53_KDACKPT=1.
TEST_F(Glm53FlashGolden, KdaPrefixCkptForkReplayBitExact) {
    if (gated_off()) GTEST_SKIP() << "set GLM53_GOLDEN=1 to run";
    if (!env_is("GLM53_KDACKPT", "1"))
        GTEST_SKIP() << "set GLM53_KDACKPT=1 to run the prefix-ckpt gate";
    if (find_big_sm120(24.0) < 0)
        GTEST_SKIP() << "no SM120+ GPU with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "GLM-5.3-Flash GGUF not present";

    start_engine();
    if (::testing::Test::HasFailure()) return;

    // Mint a natural >=96-token prompt: the 5-token France prompt + 91
    // greedy continuation tokens (same protocol as the superchunk gate).
    constexpr uint32_t C = 64;    // checkpoint position (64-aligned)
    constexpr uint32_t N = 96;    // full prompt length
    constexpr int G = 6;          // compared continuation length
    std::vector<uint32_t> mint;
    run_prompt(60, kFrancePrompt, static_cast<int>(N)
               - static_cast<int>(kFrancePrompt.size()), &mint);
    if (::testing::Test::HasFailure()) return;
    std::vector<uint32_t> long_prompt = kFrancePrompt;
    for (size_t i = 0; i + kFrancePrompt.size() < N; ++i)
        long_prompt.push_back(mint[i]);
    ASSERT_EQ(long_prompt.size(), static_cast<size_t>(N));

    const std::string a_end = "/tmp/glm53_kdackpt_a_end.bin";
    const std::string b_end = "/tmp/glm53_kdackpt_b_end.bin";

    // -- ARM A: uninterrupted reference. Feed all N, continue G-1 greedy
    // steps (G sampled tokens; last never fed => frontier N+G-1).
    std::vector<uint32_t> a_gen;
    create_sequence(61, N);
    uint32_t tok = 0;
    for (uint32_t i = 0; i < N; ++i)
        tok = decode_step(long_prompt[i], 61, i);
    ASSERT_FALSE(::testing::Test::HasFailure());
    a_gen.push_back(tok);
    for (int g = 1; g < G; ++g) {
        tok = decode_step(tok, 61, N + static_cast<uint32_t>(g) - 1);
        a_gen.push_back(tok);
        ASSERT_FALSE(::testing::Test::HasFailure());
    }
    seq_snapshot(61, N + static_cast<uint32_t>(G) - 1, a_end);
    free_sequence(61);
    ASSERT_FALSE(::testing::Test::HasFailure());
    fprintf(stderr, "[glm53-kdackpt] arm A gen:");
    for (uint32_t t : a_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n");

    // -- ARM B: same prompt on a fresh sequence, checkpoint captured at
    // C mid-feed (uniform frontier between steps), then a TRUNCATING
    // fork at C and a tail replay [C, N) on the CHILD.
    create_sequence(62, N);
    for (uint32_t i = 0; i < C; ++i)
        decode_step(long_prompt[i], 62, i);
    ASSERT_FALSE(::testing::Test::HasFailure());
    kda_ckpt(62, C);
    ASSERT_FALSE(::testing::Test::HasFailure());
    // Parent keeps stepping past the checkpoint (the divergence shape:
    // its state at N is NOT the state at C).
    for (uint32_t i = C; i < N; ++i)
        decode_step(long_prompt[i], 62, i);
    ASSERT_FALSE(::testing::Test::HasFailure());

    // NEGATIVE CONTROL: no checkpoint at 32 => the truncating fork is
    // refused loudly (INV-KDA-ANCHOR discipline).
    seq_fork_truncated(62, 64, /*prefix=*/32, /*expect_refusal=*/true);
    ASSERT_FALSE(::testing::Test::HasFailure());

    // The real fork: child 63 stands at the checkpoint frontier C with
    // the checkpoint state bytes; KV/indexer pages truncate-shared.
    seq_fork_truncated(62, 63, /*prefix=*/C);
    ASSERT_FALSE(::testing::Test::HasFailure());
    free_sequence(62);

    // Replay the tail on the child, then continue greedy.
    std::vector<uint32_t> b_gen;
    uint32_t tok_b = 0;
    for (uint32_t i = C; i < N; ++i)
        tok_b = decode_step(long_prompt[i], 63, i);
    ASSERT_FALSE(::testing::Test::HasFailure());
    b_gen.push_back(tok_b);
    for (int g = 1; g < G; ++g) {
        tok_b = decode_step(tok_b, 63, N + static_cast<uint32_t>(g) - 1);
        b_gen.push_back(tok_b);
        ASSERT_FALSE(::testing::Test::HasFailure());
    }
    seq_snapshot(63, N + static_cast<uint32_t>(G) - 1, b_end);
    free_sequence(63);
    fprintf(stderr, "[glm53-kdackpt] arm B gen:");
    for (uint32_t t : b_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n");

    // (a) TOKEN identity of the continuation.
    for (int g = 0; g < G; ++g)
        EXPECT_EQ(b_gen[static_cast<size_t>(g)],
                  a_gen[static_cast<size_t>(g)])
            << "ckpt-fork replay continuation diverges at +" << g;

    // (b) BYTE identity of the end states (KV + indexer + fp32 state +
    // frontier) — the whole-slot round-trip + INV-KDA-CARRY replay proof.
    auto ba = slurp(a_end);
    auto bb = slurp(b_end);
    ASSERT_FALSE(ba.empty());
    ASSERT_EQ(ba.size(), bb.size());
    if (ba != bb) {
        size_t off = 0;
        while (off < ba.size() && ba[off] == bb[off]) ++off;
        ADD_FAILURE() << "end-state checkpoints differ at byte " << off
                      << " of " << ba.size();
    }
    if (!::testing::Test::HasFailure()) {
        std::remove(a_end.c_str());
        std::remove(b_end.c_str());
    } else {
        fprintf(stderr, "[glm53-kdackpt] FAILURE: files kept: %s %s\n",
                a_end.c_str(), b_end.c_str());
    }
}

// Test 5 -- EP BEYOND TP TOKEN IDENTITY (TD-GLM53-EP4-DEGENERATE-GENERATION).
//
// The shape: ONE TP rank (`tp_array [0]`) plus an EXPERT-ONLY GPU. Nothing
// else in this repo exercises it -- every other EP>1 gate is `tp_array [0,1]`
// -- and the first time it ran in production the routed EP combine SILENTLY
// DROPPED every expert placed on the non-TP host: they were fetched, arrived
// and were marked resident, but no rank dispatched or folded them, so each
// token lost most of its top-K contribution. Nothing reported degraded (the
// experts were not LATE), and the model degenerated into copying its prompt.
//
// The gate mints its reference ON THE SAME BOOT: leg A runs the greedy
// trajectory with every routed expert placed on GPU 0 (the classic
// single-host path); leg B runs the identical prompt with the routed set
// ROUND-ROBINED over {GPU 0, the expert-only GPU}, which is the only
// difference. With `deterministic_ep_combine` on (set by the GLM53_EP=2 boot)
// the per-slot combine is placement-INVARIANT, so the two legs must agree
// BITWISE at the token level -- a single divergence means the extra host's
// partial is missing, doubled, or folded out of order.
//
// TD-MOE-EP-XTP-NO-TP1-COVERAGE: beyond the B=1 decode-shaped dispatch,
// the SAME boot then covers the two production batched shapes on this
// topology as SAME-CODE-PATH placement pairs — leg C runs CHUNKED PREFILL
// (one batched FETCH_AND_RUN_MOE union per layer) all-rank0 AND spread,
// leg D runs SUPERCHUNK prefill (MOE_BIG single-shot, the EP4 champion's
// serving-prefill arm) all-rank0 AND spread; each pair must agree BITWISE
// (placement is the only variable; comparing a batched prefill against the
// teacher-forced trajectory would instead inherit the documented KDA
// scan-vs-recurrent tolerance class, TD-GLM5-PREFILL-RECURRENT-DRIFT).
// Chunked (multi-chunk) MOE_BIG with extra-rank residents stays
// REJECTED-LOUD (TD-MOE-EP-XTP-WAVES) and is out of scope here.
//
// KNOWN RED at HEAD (TD-MOE-EP-XTP-PLACEMENT-DRIFT, 2026-09-03): the A/B
// decode identity is violated at depth — deterministic per placement, but
// the two placements drift apart at a low-margin token (+16 on this prompt
// in every arm; +2 with the decode resident-overlap pass on the
// exact-widths carve — LS_MOE_RESIDENT_OVERLAP=0 removes the +2 layer,
// LS_PINNED_EXACT_WIDTHS=0 moves residency so it never engages). Since
// P-29 step 18 the resident-overlap default is OFF, so the DEFAULT arm here
// behaves like the old =0 arm (first divergence +16, not +2); set
// LS_MOE_RESIDENT_OVERLAP=1 to reproduce the +2 arm. The A/B
// comparison is a SOFT EXPECT so legs C/D always report.
//
// Env: GLM53_GOLDEN=1 GLM53_EP=2 (needs two visible >=24 GB SM120 GPUs);
// GLM53_EP_GEN (default 26 — >= 26 keeps the superchunk body above the
// 16-row bound like the GLM53_SUPERCHUNK gate).
TEST_F(Glm53FlashGolden, EpBeyondTpTokenIdentity) {
    if (gated_off()) GTEST_SKIP() << "set GLM53_GOLDEN=1 to run";
    const char* ep = std::getenv("GLM53_EP");
    if (!ep || std::atoi(ep) != 2)
        GTEST_SKIP() << "set GLM53_EP=2 to run the EP-beyond-TP gate";
    if (std::getenv("GLM53_TP") && std::atoi(std::getenv("GLM53_TP")) == 2)
        GTEST_SKIP() << "GLM53_EP=2 and GLM53_TP=2 are mutually exclusive";
    if (visible_sm120_count(24.0) < 2)
        GTEST_SKIP() << "need two SM120+ GPUs with >=24 GB visible";
    if (!fs::exists(kGgufAbs)) GTEST_SKIP() << "GLM-5.3-Flash GGUF not present";

    start_engine();
    if (::testing::Test::HasFailure()) return;
    ASSERT_EQ(ep_gpus_.size(), 2u) << "GLM53_EP=2 boot did not arm two hosts";

    int gen_extra = 26;
    if (const char* g = std::getenv("GLM53_EP_GEN")) gen_extra = atoi(g);
    if (gen_extra < 5) gen_extra = 5;  // prefill legs replay K = gen-4 > 0

    // Leg A -- reference: every routed expert on the TP rank.
    ep_spread_ = false;
    std::vector<uint32_t> ref_gen;
    run_prompt(51, kFrancePrompt, gen_extra, &ref_gen);
    if (::testing::Test::HasFailure()) return;
    ASSERT_FALSE(ref_gen.empty());
    // The reference must itself be coherent, or the gate would happily
    // compare two garbled trajectories (the exact failure mode that let the
    // production defect through every counter-based check).
    EXPECT_EQ(ref_gen.front(), kTokParis)
        << "EP gate reference leg is not coherent -- fix that before reading "
           "the identity comparison";

    // Leg B -- the routed set spread across the expert-only host.
    ep_spread_ = true;
    std::vector<uint32_t> ep_gen;
    run_prompt(52, kFrancePrompt, gen_extra, &ep_gen);
    ep_spread_ = false;
    if (::testing::Test::HasFailure()) return;

    fprintf(stderr, "[glm53-ep] ref:");
    for (uint32_t t : ref_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n[glm53-ep] ep :");
    for (uint32_t t : ep_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n");

    // ALL engine legs run BEFORE any trajectory comparison: gtest failure
    // state is sticky and the fixture helpers early-out on HasFailure(), so
    // a soft A/B failure would otherwise silently skip the batched legs.
    //
    // Legs C + D -- TD-MOE-EP-XTP-NO-TP1-COVERAGE: the BATCHED shapes on
    // the same boot, asserted as SAME-CODE-PATH placement pairs (rank0 vs
    // spread), so placement is the ONLY variable. Comparing a batched
    // prefill against the teacher-forced trajectory instead would inherit
    // the documented KDA scan-vs-recurrent tolerance class
    // (TD-GLM5-PREFILL-RECURRENT-DRIFT) and assert the wrong thing.
    // Mint the longer prompt from the (coherence-checked) reference.
    const int K = gen_extra - 4;
    std::vector<uint32_t> long_prompt = kFrancePrompt;
    for (int i = 0; i < K; ++i)
        long_prompt.push_back(ref_gen[static_cast<size_t>(i)]);
    const int pf_extra = gen_extra - K;   // 4

    // Leg C -- chunked prefill (batched FETCH_AND_RUN_MOE union per layer):
    // all-rank0 reference vs the union round-robined over both hosts.
    ep_spread_ = false;
    std::vector<uint32_t> ck0_gen;
    run_prompt_prefill(53, long_prompt, pf_extra, &ck0_gen);
    if (::testing::Test::HasFailure()) return;
    ep_spread_ = true;
    std::vector<uint32_t> ck1_gen;
    run_prompt_prefill(54, long_prompt, pf_extra, &ck1_gen);
    ep_spread_ = false;
    if (::testing::Test::HasFailure()) return;

    // Leg D -- superchunk prefill (MOE_BIG single-shot per layer, the EP4
    // champion serving-prefill arm): same placement pair.
    std::vector<uint32_t> sc0_gen;
    run_prompt_superchunk(55, long_prompt, pf_extra, /*sub=*/128, &sc0_gen);
    if (::testing::Test::HasFailure()) return;
    ep_spread_ = true;
    std::vector<uint32_t> sc1_gen;
    run_prompt_superchunk(56, long_prompt, pf_extra, /*sub=*/128, &sc1_gen);
    ep_spread_ = false;
    if (::testing::Test::HasFailure()) return;

    // ── Comparisons (no engine work below this line) ─────────────────────

    // A/B decode identity -- SOFT (EXPECT), so the C/D verdicts always
    // report. KNOWN RED at HEAD (TD-MOE-EP-XTP-PLACEMENT-DRIFT, filed
    // 2026-09-03): the two placements are each bit-STABLE run-to-run but
    // drift apart at a low-margin token (+16 on this prompt in every arm;
    // +2 with the resident-overlap pass on the exact-widths carve). The
    // EP4-fix-era 13-token green was depth-limited, not proof of placement
    // invariance at any depth.
    ASSERT_EQ(ref_gen.size(), ep_gen.size());
    {
        size_t div = 0;
        while (div < ref_gen.size() && ref_gen[div] == ep_gen[div]) ++div;
        EXPECT_EQ(div, ref_gen.size())
            << "EP-beyond-TP decode trajectory diverges at +" << div
            << " (ref " << ref_gen[div] << " vs ep " << ep_gen[div]
            << ") -- the expert-only host's partial is missing, "
               "double-counted, folded out of order, or numerically "
               "placement-sensitive (INV-MOE-EP-XTP; "
               "TD-MOE-EP-XTP-PLACEMENT-DRIFT)";
    }

    ASSERT_EQ(ck0_gen.size(), static_cast<size_t>(pf_extra) + 1);
    ASSERT_EQ(ck1_gen.size(), static_cast<size_t>(pf_extra) + 1);
    ASSERT_EQ(sc0_gen.size(), static_cast<size_t>(pf_extra) + 1);
    ASSERT_EQ(sc1_gen.size(), static_cast<size_t>(pf_extra) + 1);
    fprintf(stderr, "[glm53-ep] chunked  rank0:");
    for (uint32_t t : ck0_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, " | spread:");
    for (uint32_t t : ck1_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n[glm53-ep] superchk rank0:");
    for (uint32_t t : sc0_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, " | spread:");
    for (uint32_t t : sc1_gen) fprintf(stderr, " %u", t);
    fprintf(stderr, "\n");
    for (int i = 0; i <= pf_extra; ++i) {
        EXPECT_EQ(ck1_gen[static_cast<size_t>(i)],
                  ck0_gen[static_cast<size_t>(i)])
            << "EP-spread CHUNKED prefill diverges from its all-rank0 twin "
               "at +" << i << " (INV-MOE-EP-XTP: batched finalize with an "
               "expert-only host)";
        EXPECT_EQ(sc1_gen[static_cast<size_t>(i)],
                  sc0_gen[static_cast<size_t>(i)])
            << "EP-spread SUPERCHUNK (MOE_BIG single-shot) diverges from its "
               "all-rank0 twin at +" << i
            << " (INV-MOE-EP-XTP: the EP4 serving-prefill arm)";
    }
}

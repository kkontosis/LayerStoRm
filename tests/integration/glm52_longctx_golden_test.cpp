// GLM-25 long-context DSA golden: end-to-end validation for GLM-5.2 (436 GB
// GGUF) at sequence lengths BEYOND index_topk (2048).
//
// The short golden (glm52_gguf_golden_test.cpp) runs at 5 tokens where the
// DSA lightning-indexer top-k selects every causal position — sparse and
// dense attention are numerically identical, so the indexer itself is
// unobservable. This test teacher-forces a prompt LONGER than index_topk
// (2048), where the top-k actually PRUNES: sparse ≠ dense becomes observable,
// so token parity with llama.cpp on the identical GGUF validates the whole
// indexer chain — the indexer rope convention, the k_norm LayerNorm, the
// QuaRot-style Hadamard rotation of indexer q/k (TD-GLM-INDEXER-HADAMARD,
// ref/llama.cpp/src/models/deepseek32.cpp:281-286), the FP8 stored-K
// convention, and the causal top-k selection.
//
// MoE runs on the PRODUCTION seam: fused-gate routing export +
// E_CMD_FETCH_AND_RUN_MOE fetches only the routed ≤8 experts per layer.
//
// Inputs come from the environment (skips gracefully when unset):
//   GLM52_LONGCTX_TOKENS = path to a file with the prompt token ids, one
//                          decimal id per line (> 2048 lines for the top-k to
//                          prune), produced by llama.cpp --verbose-prompt on
//                          the identical GGUF;
//   GLM52_LONGCTX_GOLDEN = the expected greedy next-token id (llama.cpp
//                          --temp 0 --top-k 1 ground truth).
//
// Run (both RTX 5090s, TP=2/EP=2 keeper shape):
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
//     GLM52_LONGCTX_TOKENS=/path/tokens.txt GLM52_LONGCTX_GOLDEN=12345 \
//     ./build/tests/integration/glm52_longctx_golden_test
// Strategy override: GG_STRATEGY=int|dequant (default: int only; run dequant
// explicitly when wanted).
//
// Skips gracefully if the GGUF, the token file, the golden id, or an SM120+
// GPU pair with enough VRAM is absent.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <deque>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nlohmann/json.hpp>

#include "core/memory/expert_cache.h"
#include "daemon/engine.h"
#include "daemon/buffer_registry.h"
#include "daemon/ipc_protocol.h"

namespace lipc = layerstorm::ipc;
namespace ldam = layerstorm::daemon;
namespace fs   = std::filesystem;

namespace {
const char* kConfigRel = "/test-data/config/glm_5_2_gguf.json";
const char* kGgufRel =
    "/test-data/GLM-5.2-GGUF-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf";

// Prompt token ids: one decimal id per line (blank lines ignored). Returns an
// empty vector on any read/parse problem — the caller skips.
std::vector<uint32_t> read_token_file(const std::string& path) {
    std::vector<uint32_t> ids;
    std::ifstream f(path);
    if (!f) return ids;
    std::string line;
    while (std::getline(f, line)) {
        // Trim whitespace; skip blank lines.
        const auto b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        const auto e = line.find_last_not_of(" \t\r\n");
        try {
            const unsigned long v = std::stoul(line.substr(b, e - b + 1));
            ids.push_back(static_cast<uint32_t>(v));
        } catch (...) {
            ids.clear();
            return ids;
        }
    }
    return ids;
}
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

// Keeper-style per-GPU LRU model (first_token_test 13c-2.0 Option A): the
// test supplies an eviction victim per overflowing fetch so the daemon's
// arbitrary unranked eviction is replaced by global LRU — the keeper
// benchmark's +44% over arbitrary-eviction FETCH.
struct LruKey {
    uint32_t layer;
    uint16_t expert;
    bool operator==(const LruKey& o) const {
        return layer == o.layer && expert == o.expert;
    }
};
struct LruKeyHash {
    size_t operator()(const LruKey& k) const {
        return (static_cast<size_t>(k.layer) << 16) ^ k.expert;
    }
};
struct GpuLru {
    int capacity = 0;
    std::deque<LruKey> order;                          // front = MRU
    std::unordered_set<LruKey, LruKeyHash> resident;
    void touch(const LruKey& k) {
        for (auto it = order.begin(); it != order.end(); ++it)
            if (*it == k) { order.erase(it); break; }
        order.push_front(k);
    }
};

class Glm52LongCtxGolden : public ::testing::Test {
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
        // TD-MOE-EXPERT-WINDOW: reserve a real expert-cache window per GPU
        // (vram_allocation_gb.expert_streaming = TOTAL expert cache, GB —
        // carved out BEFORE the auto KV pool). The auto KV sizing is
        // serving-demand-driven and otherwise consumes all remaining VRAM,
        // leaving the expert cache at the bare top-K minimum (~211 MiB ≈ 4
        // stable slots/GPU) — the 2876-token prefill then streams each MoE
        // layer's ~253-expert union in ~42 tiny waves of ~6 grouped-GEMM
        // experts (~3.9 TFLOP/s effective). 4.5 GB ≈ 87 stable slots/GPU →
        // 2 fat waves/layer. The 2877-token golden still has >40× KV
        // headroom in the remaining pool. GLM52_EXPERT_CACHE_GB overrides
        // (0 = legacy auto minimum).
        double expert_cache_gb = 4.5;
        if (const char* eg = std::getenv("GLM52_EXPERT_CACHE_GB"); eg && *eg)
            expert_cache_gb = std::atof(eg);
        j["hardware"]["gpus"] = nlohmann::json::array(
            {{{"id", 0}, {"type", "rtx5090"}, {"vram_gb", 30},
              {"pcie_gen", 5}, {"pcie_width", 16}},
             {{"id", 1}, {"type", "rtx5090"}, {"vram_gb", 30},
              {"pcie_gen", 5}, {"pcie_width", 16}}});
        if (expert_cache_gb > 0.0)
            for (auto& g : j["hardware"]["gpus"])
                g["vram_allocation_gb"] =
                    {{"expert_streaming", expert_cache_gb}};
        j["hardware"]["tp_array"] = nlohmann::json::array({0, 1});
        j["parallelism"]["tensor_parallelism"] = 2;
        // Dense A/B control: GLM52_FORCE_DENSE=1 disables DSA outright
        // (index_topk=0 → has_dsa false, attention stays dense) — isolates
        // DSA-selection bugs from generic long-context attention bugs.
        // (This used to flip dcp_indexer_mode=local when local was
        // fail-closed; local is now a REAL sparse mode — see below.)
        if (const char* fd = std::getenv("GLM52_FORCE_DENSE"); fd && *fd == '1')
            j["model"]["index_topk"] = 0;
        // GLM52_INDEXER_LOCAL=1: position-sharded indexer-K + exact cross-
        // rank top-k merge (TD-GLM-INDEXER-LOCAL-MERGE). Shrinks the indexer
        // page to 1024 tokens so the 2877-token prompt splits into REAL
        // shards on both ranks (pages 0,2 → rank 0; page 1 → rank 1 —
        // default 8192 would put the whole prompt on rank 0 and only
        // exercise the empty-shard merge path). Per-position scores are
        // page-size-invariant, so the golden must still match the
        // replicated-indexer result exactly. Snapshots are mode-bound
        // (replicated-shape indexer-K pages): run local from scratch or
        // with a local-mode checkpoint.
        if (const char* il = std::getenv("GLM52_INDEXER_LOCAL"); il && *il == '1') {
            j["hardware"]["dcp_indexer_mode"] = "local";
            j["memory"]["kv_cache"]["indexer_k_page_size_tokens"] = 1024;
        }
        // GLM52_KV_SHARDED=1: sequence-sharded KV (KVS-2/-3) — per-rank token
        // shards + the Q-head-allgather DCP combine (INV-KVS-QAG, KVS-3b).
        // KVS-4: DSA sparse runs under sharding too (replicated indexer →
        // identical global top-k per rank → per-rank GLOBAL→LOCAL translation
        // → sparse partials merged by the QAG combine), so this run must
        // reproduce the replicated-sparse golden exactly. Snapshots are
        // mode-bound: use a mode-specific GLM52_CKPT_PATH (a replicated
        // checkpoint will not restore into a sharded run).
        if (const char* ks = std::getenv("GLM52_KV_SHARDED"); ks && *ks == '1')
            j["hardware"]["dcp_kv_mode"] = "sharded";

        // GLM52_TQ=1 (TD-KVT-TQ-GOLDEN): select the TurboQuant MLA 4-bit
        // attention backend (the second AttentionDevice besides SnapMLA).
        // TQ is a LOSSY backend (TQ_MSE: Pi-rotated c_kv Lloyd-Max-coded to
        // packed 4-bit + FP16 norm, BF16 rope) — the golden pins TQ's OWN
        // reproducible greedy output, NOT bit-equality with SnapMLA. Pass
        // the TQ reference via GLM52_LONGCTX_GOLDEN (tokens3 prompt: 12089
        // " Paris" — TQ reproduces SnapMLA's argmax; top1 recorded in
        // spec/TECH_DEBT.md TD-KVT-TQ-GOLDEN). Checkpoints are BACKEND-
        // bound: TQ KV rows are 386 B packed-4-bit vs SnapMLA's 644 B
        // FP8+scale rows (kv_bytes_per_token, vram_allocator.cpp), and
        // seq_restore's kv_stride_block guard rejects cross-backend files —
        // use a TQ-specific GLM52_CKPT_PATH (glm52-longctx-ckpt-tq.bin).
        // Composes with GLM52_KV_TIERING (INV-KVT-8: TQ rows are self-
        // contained, placement-only tiering ⇒ token AND top1 match the
        // non-tiered TQ golden exactly) and GLM52_SPARSE_PREFILL (TQ sparse
        // chunk prefill).
        if (const char* tq = std::getenv("GLM52_TQ"); tq && *tq == '1')
            j["compute"]["attention_backend"] = "turboquant_mla";

        // GLM-25k: GLM52_KV_TIERING=<hot_buffer_slots> enables DSA-guided KV
        // tiering with the given per-layer hot buffer. A hot buffer smaller
        // than the 2877-token context forces real demotion + cold fetches on
        // the teacher-forcing decode steps. Placement-only (INV-KVT-1): the
        // greedy token AND top1 must match the non-tiered golden EXACTLY.
        // Composes with GLM52_KV_SHARDED=1 (TD-KVT-DCP-SHARDED: each rank
        // tiers its own token shard; must reproduce the sharded golden
        // exactly). Snapshot of a tiered run is supported (TD-KVT-SPEC
        // resolved: demoted pages are captured from the pinned cold pools —
        // the checkpoint is byte-identical to a non-tiered one); restoring
        // FROM any checkpoint restores pages hot and demotion restarts at
        // the first tiered decode step.
        // GLM52_KVT_RATIO=<r> overrides host_to_device_ratio (default 16 —
        // the historical goldens' value, cold capacity 16×hot per layer).
        // The pinned cold pools must cover every token a run can demote:
        // ratio × hot_buffer_slots ≥ context length (further demotions are
        // skipped fail-safe once full — correct but no longer exercising
        // tiering). The 1M capacity smoke uses hot=256, ratio=4160
        // (≈1.065M tokens/layer of cold capacity).
        if (const char* kt = std::getenv("GLM52_KV_TIERING"); kt && *kt) {
            double ratio = 16.0;
            if (const char* kr = std::getenv("GLM52_KVT_RATIO"); kr && *kr)
                if (double v = std::atof(kr); v >= 1.0) ratio = v;
            j["memory"]["kv_tiering"] = {
                {"enabled", true},
                {"hot_buffer_slots", std::atoi(kt)},
                {"host_to_device_ratio", ratio}};
        }

        // GLM-25k 1M capacity smoke: GLM52_MAX_SEQ=<tokens> lifts the
        // serving context cap (preset 32768; schema max 1048576 post-#13).
        // Sizes the indexer-K pool (IndexShare-aware, ~2.9 GB/rank @1M),
        // the SnapMLA prefill KV staging (~1.15 GB/rank @1M) and the rope
        // table (~256 MB/rank @1M) for the cap — combined with
        // GLM52_KV_TIERING + GLM52_KVT_RATIO this proves the engine
        // initializes and allocates at the 1M cap on 2×32 GB.
        if (const char* ms = std::getenv("GLM52_MAX_SEQ"); ms && *ms)
            if (long v = std::atol(ms); v >= 128)
                j["serving"]["max_sequence_length"] = v;

        // GLM52_SPARSE_PREFILL=1 (TD-SPARSE-CHUNK-PREFILL): DSA sparse CHUNK
        // PREFILL attention — combine with GLM52_LONGCTX_PREFILL=<chunk>.
        // Each prefill chunk row then attends only its causal top-k
        // (≤ index_topk = 2048) instead of the dense full prefix; at this
        // 2877-token prompt the selection PRUNES for rows past 2048, so the
        // run validates the sparse-prefill numerics against the DSA
        // reference next token (12089 — the reference, llama.cpp DSA, is
        // sparse end-to-end, prefill included).
        if (const char* sp = std::getenv("GLM52_SPARSE_PREFILL");
            sp && *sp == '1')
            j["compute"]["dsa_sparse_prefill"] = true;
        // GLM52_MAX_BATCH=<n>: overrides orchestrator.max_batch_size
        // (preset 64). The KV-metadata device scratch scales L × max_batch ×
        // (max_seq / page_size) ints — 1.3 GiB/rank at the 1M cap with
        // B=64 — so the 1M capacity smoke runs the honest B==1-decode shape
        // at a small max_batch (8 ⇒ ~166 MiB/rank) instead of failing init.
        // Also caps the GLM52_LONGCTX_PREFILL chunk size.
        if (const char* mb = std::getenv("GLM52_MAX_BATCH"); mb && *mb)
            if (int v = std::atoi(mb); v >= 1)
                j["orchestrator"]["max_batch_size"] = v;
        max_batch_ = j["orchestrator"].value("max_batch_size", 64);

        // GLM52_SUPERCHUNK_K=<K> (TD-PREFILL-SUPERCHUNK): decoupled prefill
        // superchunks — K attention SUB-CHUNKS of GLM52_LONGCTX_PREFILL
        // tokens each (sparse + tiering per sub-chunk UNCHANGED), then ONE
        // FETCH_AND_RUN_MOE over all K×chunk tokens per layer, amortizing the
        // per-layer routed-expert H2D fetch across the whole superchunk.
        // Sets compute.prefill_superchunk_tokens so the engine sizes the MoE
        // scratch + hidden staging (VRAM fail-safe steps it down, never OOM;
        // the driver re-clamps K to EngineInfo.moe_batch_capacity). K=1 runs
        // the superchunk code path at legacy sizes (bit-identity baseline).
        superchunk_k_ = 0;
        if (const char* sk = std::getenv("GLM52_SUPERCHUNK_K"); sk && *sk)
            if (int v = std::atoi(sk); v >= 1)
                superchunk_k_ = v;
        if (superchunk_k_ > 0) {
            int chunk = 64;
            if (const char* pc = std::getenv("GLM52_LONGCTX_PREFILL");
                pc && *pc)
                if (int v = std::atoi(pc); v > 0)
                    chunk = std::min(v, max_batch_);
            j["compute"]["prefill_superchunk_tokens"] =
                superchunk_k_ * chunk;
        }

        j["memory"]["vram_safety_margin_gb"] = 3.0;
        // Unlike the 5-token golden (which shrank KV growth to 1 page/step),
        // this prompt is thousands of tokens: keep the preset's "auto" page
        // budget and default growth chunk so the KV cache can actually grow
        // past index_topk on both TP ranks.
        //
        // KEEPER PARITY (BENCHMARK_FULLFIT.md, GG-10 prepacked set): experts
        // come from test-data/GLM-5.2-prepacked (494 GB, format 9.69.0
        // per-layer gguf types). NUMA-pinned host arenas at 0.76
        // fraction_total (~408 GB of the 537 GB stride-padded set; the rest
        // slab-LRUs via O_DIRECT pread on miss), bulk io_uring O_DIRECT
        // preload (~2.5 min at the 3.3 GB/s Gen3-x4 wall), runtime decode
        // hit-mostly. Mirrors RoutedExpertLruFullFitDirectTest::patch_config.
        {
            // KEEPER52 ARENA PARITY (exact match to keeper52_test::patch_config):
            // fraction_total 0.90 on nodes 0,1 + the CPU-less HBM nodes 4-7 as
            // extra warm capacity (TD-NUMA-HBM-BANKS), so this prefill golden
            // measures on the SAME arena as the keeper benchmark (~500 GB → ~95%
            // of the 530 GB expert set resident, vs the old 0.76/nodes-0,1
            // ~382 GB/72% which thrashed ~28% to disk). GLM52_ARENA_FRACTION=<f>
            // overrides — the 1M capacity smoke, which must co-locate a ~27 GB
            // pinned tiering cold pool on each 5090's home node (nodes 0,1), sets
            // GLM52_ARENA_FRACTION=0.6 for ~26 GB slack (else CONSTRAINT_MEMORY_POLICY
            // OOM-kill; see spec/DEBUG.md box-contention note).
            double arena_frac = 0.90;
            if (const char* af = std::getenv("GLM52_ARENA_FRACTION"); af && *af)
                if (double v = std::atof(af); v > 0.0 && v <= 1.0)
                    arena_frac = v;
            const std::string prepacked = src + "/test-data/GLM-5.2-prepacked";
            if (fs::exists(prepacked + "/manifest.json")) {
                j["preprocessing"]["prepacked_dir"] = prepacked;
                // Gates the PinnedExpertArena build (engine init) — without
                // it there is NO arena tier, and with direct_load the
                // PrepackedSource has no mmap either: every fetch dies with
                // "expert not found" + 120 s timeouts (first-run lesson).
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
                // Exact keeper52 spill block: RAM nodes 0,1 (fraction_total) PLUS
                // the CPU-less HBM nodes 4-7 (per-node fraction_free 0.80 — they
                // are small/partially-occupied), 0 skipped (TD-NUMA-HBM-BANKS).
                j["memory"]["cross_node_spill"] = {
                    {"enabled", true},
                    {"nodes", nlohmann::json::array({
                        {{"node", 0}, {"weight", 1}},
                        {{"node", 1}, {"weight", 1}},
                        {{"node", 4}, {"weight", 1}},
                        {{"node", 5}, {"weight", 1}},
                        {{"node", 6}, {"weight", 1}},
                        {{"node", 7}, {"weight", 1}},
                    })},
                    {"sizing_mode", "fraction_total"},
                    {"sizing_value", arena_frac},
                    {"per_node", nlohmann::json::array({
                        {{"node", 4}, {"mode", "fraction_free"}, {"value", 0.80}},
                        {{"node", 5}, {"mode", "fraction_free"}, {"value", 0.80}},
                        {{"node", 6}, {"mode", "fraction_free"}, {"value", 0.80}},
                        {{"node", 7}, {"mode", "fraction_free"}, {"value", 0.80}},
                    })},
                };
            } else {
                // No prepacked set: fall back to unlimited pinned retention
                // (481-1) so fetches at least stop re-reading the GGUF.
                j["memory"]["host_packed_cache_mb"] = -1;
                j["memory"]["pin_host_expert_pool"] = true;
                j["memory"]["arena_attach"] = {
                    // P-24b: tests default to a process-PRIVATE arena so dev test runs
                    // never attach to / wipe / hold the box's persistent holder store
                    // (configs differ in geometry, so each run would thrash it).
                    // GLM52_ARENA_ATTACH=1 opts the persistence gates back in.
                    {"enabled", [] {
                        const char* aa = std::getenv("GLM52_ARENA_ATTACH");
                        return aa && *aa == '1';
                    }()}};
            }
        }

        config_path_ = "/tmp/glm52_longctx_config_" + strategy + ".json";
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

        // Keeper LRU capacities: per-GPU stable-zone slot counts.
        lrus_.assign(static_cast<size_t>(engine_->info().num_gpus), {});
        for (int g = 0; g < engine_->info().num_gpus; ++g)
            lrus_[g].capacity = engine_->expert_cache()->total_slots(
                g, layerstorm::memory::CacheZone::kStable);

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

    // Keeper LRU (13c-2.0 Option A): attach a global-LRU victim to each miss
    // that would overflow the stable zone; mirrors
    // first_token_test::decode_step_fetch_and_run. Shared by the decode and
    // prefill step drivers (verbatim behavior of the old decode inline).
    void attach_lru_victims(lipc::ExpertPrefetchEntry* entries,
                            lipc::ExpertEvictionEntry* evicts,
                            uint32_t count, int layer, int tp) {
        for (int g = 0; g < tp; ++g) {
            GpuLru& lru = lrus_[g];
            std::vector<uint16_t> want;
            std::vector<uint32_t> miss_ei;
            for (uint32_t i = 0; i < count; ++i) {
                if (entries[i].gpu_idx != g) continue;
                uint16_t e = entries[i].expert_idx;
                want.push_back(e);
                LruKey k{static_cast<uint32_t>(layer), e};
                if (lru.resident.count(k)) lru.touch(k);
                else miss_ei.push_back(i);
            }
            auto needed_now = [&](const LruKey& k) {
                if (k.layer != static_cast<uint32_t>(layer)) return false;
                return std::find(want.begin(), want.end(),
                                 static_cast<uint16_t>(k.expert))
                       != want.end();
            };
            // TD-PREFILL-SUPERCHUNK perf accounting: every miss is one real
            // expert H2D fetch under the keeper-LRU model.
            lru_miss_fetches_ += miss_ei.size();
            int need_room = static_cast<int>(miss_ei.size());
            size_t vi = 0;
            while (static_cast<int>(lru.resident.size()) + need_room
                       > lru.capacity && vi < miss_ei.size()) {
                LruKey victim{}; bool found = false;
                for (auto it = lru.order.rbegin();
                     it != lru.order.rend(); ++it)
                    if (!needed_now(*it)) { victim = *it; found = true; break; }
                if (!found) break;
                uint32_t ei = miss_ei[vi++];
                evicts[ei].layer_idx  = victim.layer;
                evicts[ei].expert_idx = victim.expert;
                evicts[ei].gpu_idx    = static_cast<uint8_t>(g);
                lru.resident.erase(victim);
                for (auto it = lru.order.begin();
                     it != lru.order.end(); ++it)
                    if (*it == victim) { lru.order.erase(it); break; }
            }
            for (uint32_t ei : miss_ei) {
                LruKey k{static_cast<uint32_t>(layer),
                         entries[ei].expert_idx};
                lru.resident.insert(k);
                lru.touch(k);
            }
        }
    }

    // One REAL prefill step over `n` prompt tokens at positions
    // [pos0, pos0+n) — the engine's prefill command path (one batch
    // descriptor per PROMPT POSITION, is_prefill=1 + chunk fields), NOT
    // teacher-forced decode. Attention runs the chunk path: DENSE per-row
    // causal by default, SPARSE chunk-causal under GLM52_SPARSE_PREFILL=1
    // (TD-SPARSE-CHUNK-PREFILL — watch for "DSA sparse CHUNK PREFILL
    // ACTIVE"); the DSA chunk appender stores the positions' indexer keys
    // either way so later steps stay sparse-eligible. MoE runs on the
    // production FETCH_AND_RUN seam with the union of the chunk's routed
    // experts and the same keeper-LRU victims as decode.
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

        const char* pv = std::getenv("GLM52_PREFILL_VERBOSE");
        const bool verbose = pv && *pv == '1';
        // TD-PREFILL-FETCH-SEAM-SCALING x-ray: per-layer attn/moe wall times.
        auto lap_ms = [t = std::chrono::steady_clock::now()]() mutable {
            auto now = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - t).count();
            t = now;
            return ms;
        };
        // GLM52_LONGCTX_PREFILL_LAYERS=<n>: measurement-only cap — run only
        // the first n layers of this chunk (perf probe; combine with
        // GLM52_LONGCTX_PREFILL_LIMIT, which skips the golden check).
        int layer_cap = num_layers_;
        if (const char* ll = std::getenv("GLM52_LONGCTX_PREFILL_LAYERS");
            ll && *ll)
            if (int v = std::atoi(ll); v > 0)
                layer_cap = std::min(layer_cap, v);
        for (int layer = 0; layer < layer_cap; ++layer) {
            const bool is_moe = layer >= first_moe_layer_;

            if (verbose)
                fprintf(stderr, "[glm52-longctx] prefill L%d attn...\n", layer);
            lap_ms();
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
            if (verbose)
                fprintf(stderr,
                        "[glm52-longctx] prefill L%d attn done (%.1f ms), "
                        "moe...\n", layer, lap_ms());

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

                // Union of the chunk's routed experts (dedupe across tokens;
                // ≤ n_routed_experts = 256 = kMaxExpertPrefetch).
                auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
                    sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
                auto* evicts = reinterpret_cast<lipc::ExpertEvictionEntry*>(
                    sideband_ + lipc::IpcLayout::kExpertEvictionOff);
                const int tp = engine_->info().num_gpus;
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
                    evicts[count].layer_idx  = static_cast<uint32_t>(layer);
                    evicts[count].expert_idx = 0xFFFF;   // sentinel: no victim
                    evicts[count].gpu_idx    = entries[count].gpu_idx;
                    evicts[count]._pad       = 0;
                    ++count;
                }
                EXPECT_GT(count, 0u)
                    << "no routed experts exported L" << layer;
                if (count == 0) return;

                attach_lru_victims(entries, evicts, count, layer, tp);
                union_sum_ += count;   // routing-concentration telemetry
                ++moe_layer_calls_;

                auto m = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
                m.fetch_and_run_moe.layer_idx    = static_cast<uint32_t>(layer);
                m.fetch_and_run_moe.num_seqs     = n;
                m.fetch_and_run_moe.expert_count = count;
                m.fetch_and_run_moe.have_evict_map = 1;  // keeper LRU eviction
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
            if (verbose)
                fprintf(stderr,
                        "[glm52-longctx] prefill L%d moe done (%.1f ms)\n",
                        layer, lap_ms());
            if (::testing::Test::HasFailure()) return;
        }
    }

    // TD-PREFILL-SUPERCHUNK: one DECOUPLED superchunk over `n` prompt tokens
    // at positions [pos0, pos0+n), processed LAYER-WISE: per layer, K
    // attention sub-chunks of `sub` tokens each (sparse selection + indexer
    // append + KV exactly as prefill_step — the hidden rows land at
    // row_offset into the engine's superchunk-sized staging), then ONE MoE
    // command over ALL n tokens (FETCH_AND_RUN on the production seam for
    // routed layers; RUN_MOE for dense layers). The per-layer routed union
    // is the DEDUP set — one fetch per unique expert per layer per
    // superchunk instead of per 64-token chunk.
    void prefill_superchunk_step(const uint32_t* toks, uint32_t n,
                                 uint64_t seq_id, uint32_t pos0,
                                 uint32_t sub) {
        lipc::Completion cmp{};
        const char* pv = std::getenv("GLM52_PREFILL_VERBOSE");
        const bool verbose = pv && *pv == '1';
        auto lap_ms = [t = std::chrono::steady_clock::now()]() mutable {
            auto now = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - t).count();
            t = now;
            return ms;
        };

        // 1. Embedding: per sub-chunk, into hidden rows [off, off+len).
        auto* token_ids = reinterpret_cast<uint32_t*>(
            sideband_ + lipc::IpcLayout::kTokenIdsOff);
        for (uint32_t off = 0; off < n; off += sub) {
            const uint32_t len = std::min(sub, n - off);
            for (uint32_t i = 0; i < len; ++i) token_ids[i] = toks[off + i];
            auto embed = make_cmd(lipc::CMD_EMBEDDING_LOOKUP);
            embed.embedding_lookup.num_tokens = len;
            embed.embedding_lookup.output_buf_id = hidden_buf_id_;
            embed.embedding_lookup.row_offset = off;
            send(embed);
            EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                << "superchunk embed off=" << off;
            if (::testing::Test::HasFailure()) return;
        }

        auto* batch = reinterpret_cast<lipc::BatchDescriptorEntry*>(
            sideband_ + lipc::IpcLayout::kBatchDescriptorOff);
        auto* entries = reinterpret_cast<lipc::ExpertPrefetchEntry*>(
            sideband_ + lipc::IpcLayout::kExpertPrefetchOff);
        auto* evicts = reinterpret_cast<lipc::ExpertEvictionEntry*>(
            sideband_ + lipc::IpcLayout::kExpertEvictionOff);
        const int tp = engine_->info().num_gpus;

        int layer_cap = num_layers_;
        if (const char* ll = std::getenv("GLM52_LONGCTX_PREFILL_LAYERS");
            ll && *ll)
            if (int v = std::atoi(ll); v > 0)
                layer_cap = std::min(layer_cap, v);

        // GLM52_CUDA_PROFILER=1: bracket the layer sweep with
        // cudaProfilerStart/Stop so `nsys --capture-range=cudaProfilerApi`
        // captures ONLY the prefill compute (skips the ~3 min arena init).
        const char* prof_env = std::getenv("GLM52_CUDA_PROFILER");
        const bool prof = prof_env && *prof_env == '1';
        if (prof) cudaProfilerStart();

        // 2. Layer-wise sweep: K sub-chunk attentions → one MoE per layer.
        for (int layer = 0; layer < layer_cap; ++layer) {
            const bool is_moe = layer >= first_moe_layer_;
            lap_ms();
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
                a.run_attention.layer_idx = static_cast<uint32_t>(layer);
                a.run_attention.num_seqs = len;
                a.run_attention.is_prefill = 1;
                a.run_attention.use_graph = 0;
                a.run_attention.chunk_start = pos0 + off;
                a.run_attention.chunk_len = len;
                a.run_attention.emit_gating  = is_moe ? 1 : 0;
                a.run_attention.store_gating = is_moe ? 1 : 0;
                a.run_attention.superchunk = 1;
                a.run_attention.row_offset = off;
                send(a);
                EXPECT_TRUE(wait(cmp, lipc::CMP_COMPUTE_DONE))
                    << "superchunk attn L" << layer << " off=" << off;
                if (::testing::Test::HasFailure()) return;

                if (is_moe) {
                    // Accumulate this sub-chunk's routed top-K into the
                    // superchunk-wide UNION (the dedup set).
                    const auto* hdr =
                        reinterpret_cast<const lipc::RoutingExportHeader*>(
                            sideband_ + lipc::IpcLayout::kRoutingExportOff);
                    EXPECT_EQ(hdr->num_tokens, len);
                    EXPECT_EQ(hdr->layer_idx, static_cast<uint32_t>(layer))
                        << "routing-export layer mismatch";
                    const auto* ridx = reinterpret_cast<const int32_t*>(
                        sideband_ + lipc::IpcLayout::kRoutingExportIndicesOff);
                    const uint32_t rn = hdr->num_tokens * hdr->topk;
                    for (uint32_t k = 0; k < rn; ++k) {
                        if (ridx[k] < 0 || ridx[k] >= num_experts_) continue;
                        if (seen[static_cast<size_t>(ridx[k])]) continue;
                        seen[static_cast<size_t>(ridx[k])] = 1;
                        entries[count].layer_idx  =
                            static_cast<uint32_t>(layer);
                        entries[count].expert_idx =
                            static_cast<uint16_t>(ridx[k]);
                        entries[count].zone       = 0;
                        entries[count].gpu_idx =
                            static_cast<uint8_t>(ridx[k] % tp);
                        evicts[count].layer_idx  =
                            static_cast<uint32_t>(layer);
                        evicts[count].expert_idx = 0xFFFF;
                        evicts[count].gpu_idx    = entries[count].gpu_idx;
                        evicts[count]._pad       = 0;
                        ++count;
                    }
                }
            }
            const double attn_ms = lap_ms();

            if (is_moe) {
                EXPECT_GT(count, 0u)
                    << "no routed experts exported L" << layer;
                if (count == 0) return;
                attach_lru_victims(entries, evicts, count, layer, tp);
                // TD-PREFILL-MOE-BIG: the superchunk MoE goes through the BIG
                // command — chunked grouped-GEMM execution (transients bounded
                // at the chunk, elastic capacity) + double-buffered waves
                // (wave i+1's H2D streams while wave i computes). Batches at or
                // below the chunk capacity run the byte-identical single-shot
                // pipeline inside the same handler.
                auto m = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE_BIG);
                m.fetch_and_run_moe_big.layer_idx    =
                    static_cast<uint32_t>(layer);
                m.fetch_and_run_moe_big.num_seqs     = n;
                m.fetch_and_run_moe_big.expert_count = count;
                m.fetch_and_run_moe_big.have_evict_map = 1;
                m.fetch_and_run_moe_big.timeout_us   = 120000000;  // 120 s
                m.fetch_and_run_moe_big.moe_mode     = 0;
                m.fetch_and_run_moe_big.chunk_tokens = 0;  // engine default
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
                << "superchunk moe L" << layer;
            if (::testing::Test::HasFailure()) return;
            const double moe_ms = lap_ms();
            if (is_moe) {
                // Routing-concentration + amortization telemetry: the union
                // is the per-layer fetch set for the WHOLE superchunk.
                union_sum_ += count;
                ++moe_layer_calls_;
                if (verbose)
                    fprintf(stderr,
                            "[glm52-superchunk] L%d n=%u union=%u "
                            "attn=%.1f ms moe=%.1f ms\n",
                            layer, n, count, attn_ms, moe_ms);
            } else if (verbose) {
                fprintf(stderr,
                        "[glm52-superchunk] L%d n=%u dense attn=%.1f ms "
                        "moe=%.1f ms\n", layer, n, attn_ms, moe_ms);
            }
        }
        if (prof) cudaProfilerStop();
    }

    // One teacher-forced step on the PRODUCTION MoE seam (same as the short
    // golden): attention with the fused router gate (emit_gating +
    // store_gating exports the routed top-K to the sideband), then
    // E_CMD_FETCH_AND_RUN_MOE fetches ONLY the routed ≤8 experts and runs
    // the MoE — the exact decode_step_fetch_and_run seam validated by
    // first_token_test.
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
                auto* evicts = reinterpret_cast<lipc::ExpertEvictionEntry*>(
                    sideband_ + lipc::IpcLayout::kExpertEvictionOff);
                const int tp = engine_->info().num_gpus;
                uint32_t count = 0;
                for (uint32_t k = 0; k < rn; ++k) {
                    if (ridx[k] < 0) continue;
                    entries[count].layer_idx  = static_cast<uint32_t>(layer);
                    entries[count].expert_idx = static_cast<uint16_t>(ridx[k]);
                    entries[count].zone       = 0;
                    // EP split: each expert fetched/run on its owning GPU.
                    entries[count].gpu_idx =
                        static_cast<uint8_t>(ridx[k] % tp);
                    evicts[count].layer_idx  = static_cast<uint32_t>(layer);
                    evicts[count].expert_idx = 0xFFFF;   // sentinel: no victim
                    evicts[count].gpu_idx    = entries[count].gpu_idx;
                    evicts[count]._pad       = 0;
                    ++count;
                }
                EXPECT_GT(count, 0u) << "no routed experts exported L" << layer;
                if (count == 0) return 0;

                // Keeper LRU (13c-2.0 Option A): attach a global-LRU victim to
                // each miss that would overflow the stable zone; mirrors
                // first_token_test::decode_step_fetch_and_run.
                attach_lru_victims(entries, evicts, count, layer, tp);

                auto m = make_cmd(lipc::E_CMD_FETCH_AND_RUN_MOE);
                m.fetch_and_run_moe.layer_idx    = static_cast<uint32_t>(layer);
                m.fetch_and_run_moe.num_seqs     = 1;
                m.fetch_and_run_moe.expert_count = count;
                m.fetch_and_run_moe.have_evict_map = 1;  // keeper LRU eviction
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

    // Debug/test checkpoint plumbing (CMD_SEQ_SNAPSHOT/RESTORE): env
    // GLM52_CKPT_PATH names the fixture file, GLM52_CKPT_STEP (default 2000)
    // the position it covers. If the fixture exists, teacher-forcing starts
    // at that step after restoring the KV + indexer-K state; otherwise the
    // run snapshots when it reaches the step, so any later rerun (e.g. after
    // a failure past 2000) restarts mid-prompt in seconds.
    bool seq_ckpt(lipc::CmdType type, uint32_t token_count) {
        lipc::Completion cmp{};
        auto c = make_cmd(type);
        c.seq_ckpt.seq_id = 1;
        c.seq_ckpt.token_count = token_count;
        send(c);
        if (!wait(cmp, lipc::CMP_SEQ_OP_DONE)) return false;
        return cmp.status == 0;
    }

    // Teacher-forces the whole prompt (thousands of steps); progress every
    // 100 steps. The per-command waits above bound each step — no additional
    // per-step timeout is needed.
    uint32_t run_prompt(const std::vector<uint32_t>& prompt,
                        float* top1 = nullptr) {
        const char* ckpt_path = std::getenv("GLM52_CKPT_PATH");
        size_t ckpt_step = 0;
        if (ckpt_path && *ckpt_path) {
            const char* se = std::getenv("GLM52_CKPT_STEP");
            long v = se ? std::strtol(se, nullptr, 10) : 2000;
            if (v > 0 && static_cast<size_t>(v) < prompt.size())
                ckpt_step = static_cast<size_t>(v);
            setenv("LS_SEQ_CKPT_PATH", ckpt_path, 1);  // daemon reads this
        }

        // GLM52_CKPT2_PATH/GLM52_CKPT2_STEP: a SECOND snapshot fixture,
        // SAVED by a decode-mode run that reaches the step (even when the
        // run itself was restored from the primary checkpoint — the
        // primary's `start == 0` save guard doesn't apply). Restore from it
        // by passing it as the primary (GLM52_CKPT_PATH/STEP) next run.
        // Rationale (tiered ≡ non-tiered at 10k): the non-tiered control
        // saves @9500 so the tiered run replays only the final teacher-
        // forced stretch — demotion of the whole restored prefix + the cold
        // needle fetch — instead of hours of identical warm-up.
        size_t ckpt2_step = 0;
        const char* ckpt2_path = std::getenv("GLM52_CKPT2_PATH");
        if (ckpt2_path && *ckpt2_path) {
            if (const char* s2 = std::getenv("GLM52_CKPT2_STEP"); s2 && *s2) {
                long v = std::strtol(s2, nullptr, 10);
                if (v > 0 && static_cast<size_t>(v) < prompt.size())
                    ckpt2_step = static_cast<size_t>(v);
            }
        }

        create_sequence(1, static_cast<uint32_t>(prompt.size()));
        size_t start = 0;
        if (ckpt_step > 0 && fs::exists(ckpt_path)) {
            if (seq_ckpt(lipc::CMD_SEQ_RESTORE,
                         static_cast<uint32_t>(ckpt_step))) {
                start = ckpt_step;
                fprintf(stderr, "[glm52-longctx] restored checkpoint @%zu "
                        "from %s\n", ckpt_step, ckpt_path);
            } else {
                ADD_FAILURE() << "checkpoint restore failed: " << ckpt_path;
                free_sequence(1);
                return 0;
            }
        }

        uint32_t tok = 0;
        // GLM52_LONGCTX_PREFILL=<chunk> (TD-SPARSE-CHUNK-PREFILL harness):
        // REAL chunked prefill of the first N−1 prompt tokens (chunk-size
        // batches, clamped to orchestrator.max_batch_size), then ONE final
        // teacher-forced decode — the long-context analogue of the short
        // golden's GLM52_PREFILL. Default (unset): teacher-forced decode of
        // every position (legacy). Composes with a restored checkpoint
        // (prefill starts at the restored position — coverage is contiguous
        // either way); mid-prefill snapshot SAVING is not supported (decode
        // mode saves as before).
        uint32_t pf_chunk = 0;
        if (const char* pc = std::getenv("GLM52_LONGCTX_PREFILL"); pc && *pc)
            if (int v = std::atoi(pc); v > 0)
                pf_chunk = std::min(static_cast<uint32_t>(v),
                                    static_cast<uint32_t>(max_batch_));
        if (pf_chunk > 0) {
            // GLM52_LONGCTX_PREFILL_LIMIT=<n>: measurement-only early stop
            // after n prefill chunks (TD-PREFILL-FETCH-SEAM-SCALING perf
            // x-ray) — cleanly shuts the engine down (perf_trace CSV + ELM
            // H2D micro logs fire on daemon exit) and SKIPS the final decode
            // + golden assert (the prompt was not fully prefilled).
            uint32_t chunk_limit = 0, chunks_done = 0;
            if (const char* cl = std::getenv("GLM52_LONGCTX_PREFILL_LIMIT");
                cl && *cl)
                if (int v = std::atoi(cl); v > 0)
                    chunk_limit = static_cast<uint32_t>(v);
            const size_t pre = prompt.size() - 1;
            // TD-KVT-PREFILL-REPROMOTE harness:
            // GLM52_DECODE_BEFORE_PREFILL=<n> teacher-forces n decode steps
            // at the restored position FIRST (under GLM52_KV_TIERING those
            // tiered steps demote the restored prefix behind the retention
            // window), then the prefill chunks below append INTO the demoted
            // sequence — with sparse prefill OFF the dense chunks exercise
            // the dispatcher's full cold-page re-promotion gate
            // (repromote_seq(seq, 0) + in-place kv-meta rebuild); the run
            // must still produce the golden token (INV-KVT-1).
            size_t pf_start = start;
            if (const char* db = std::getenv("GLM52_DECODE_BEFORE_PREFILL");
                db && *db) {
                if (long v = std::strtol(db, nullptr, 10); v > 0) {
                    const size_t end_d =
                        std::min(start + static_cast<size_t>(v), pre);
                    for (size_t i = start;
                         i < end_d && !::testing::Test::HasFailure(); ++i) {
                        float p = 0.f;
                        decode_step(prompt[i], 1, static_cast<uint32_t>(i),
                                    &p);
                    }
                    pf_start = end_d;
                    fprintf(stderr,
                            "[glm52-longctx] decode-before-prefill: %zu "
                            "steps done ([%zu, %zu)) — switching to dense "
                            "prefill chunks\n",
                            end_d - start, start, end_d);
                }
            }
            lru_miss_fetches_ = 0;
            union_sum_ = 0;
            moe_layer_calls_ = 0;
            const auto pf_t0 = std::chrono::steady_clock::now();

            // TD-PREFILL-SUPERCHUNK driver: superchunks of K sub-chunks each,
            // clamped to the engine's EFFECTIVE MoE batch capacity (the VRAM
            // fail-safe may have stepped the request down — fail SAFE to a
            // smaller K, never over-issue).
            uint32_t sc_tokens = 0;
            if (superchunk_k_ >= 1) {
                sc_tokens = pf_chunk * static_cast<uint32_t>(superchunk_k_);
                const uint32_t cap = static_cast<uint32_t>(
                    std::max(engine_->info().moe_batch_capacity, 1));
                if (sc_tokens > cap) {
                    const uint32_t k_eff = std::max(1u, cap / pf_chunk);
                    fprintf(stderr,
                            "[glm52-superchunk] K=%d (%u tokens) exceeds "
                            "engine MoE batch capacity %u — clamped to K=%u\n",
                            superchunk_k_, sc_tokens, cap, k_eff);
                    sc_tokens = k_eff * pf_chunk;
                }
                fprintf(stderr,
                        "[glm52-superchunk] ACTIVE: %u tokens/superchunk "
                        "(K=%u sub-chunks of %u)\n",
                        sc_tokens, sc_tokens / pf_chunk, pf_chunk);
            }
            const uint32_t step_tokens = sc_tokens > 0 ? sc_tokens : pf_chunk;
            for (size_t pos = pf_start;
                 pos < pre && !::testing::Test::HasFailure();
                 pos += step_tokens) {
                const uint32_t len = static_cast<uint32_t>(
                    std::min<size_t>(step_tokens, pre - pos));
                if (sc_tokens > 0)
                    prefill_superchunk_step(prompt.data() + pos, len, 1,
                                            static_cast<uint32_t>(pos),
                                            pf_chunk);
                else
                    prefill_step(prompt.data() + pos, len, 1,
                                 static_cast<uint32_t>(pos));
                fprintf(stderr,
                        "[glm52-longctx] prefill %s [%zu, %zu)/%zu done\n",
                        sc_tokens > 0 ? "superchunk" : "chunk",
                        pos, pos + len, pre);
                if (chunk_limit > 0 && ++chunks_done >= chunk_limit) {
                    fprintf(stderr,
                            "[glm52-longctx] prefill chunk limit %u reached — "
                            "stopping early (measurement mode, no golden "
                            "check)\n", chunk_limit);
                    prefill_perf_summary(pf_t0);
                    free_sequence(1);
                    return 0;
                }
            }
            prefill_perf_summary(pf_t0);
            if (!::testing::Test::HasFailure()) {
                float p = 0.f;
                tok = decode_step(prompt[pre], 1,
                                  static_cast<uint32_t>(pre), &p);
                if (top1) *top1 = p;
                fprintf(stderr,
                        "[glm52-longctx] post-prefill decode done "
                        "(argmax=%u)\n", tok);
                // GLM52_FREE_DECODE=<n> (context-ladder measurement): after
                // the golden decode, FREE-RUN n further greedy steps at full
                // context depth (feed the argmax back), timing them — the
                // decode-tok/s-at-depth probe plus a multi-token needle
                // readback (the single golden token only proves the first
                // answer token). Measurement-only: no golden semantics.
                if (const char* fdn = std::getenv("GLM52_FREE_DECODE");
                    fdn && *fdn && !::testing::Test::HasFailure()) {
                    const int n_free = std::atoi(fdn);
                    if (n_free > 0) {
                        uint32_t cur = tok;
                        std::string ids;
                        const auto fd_t0 = std::chrono::steady_clock::now();
                        for (int i = 0;
                             i < n_free && !::testing::Test::HasFailure();
                             ++i) {
                            float fp = 0.f;
                            cur = decode_step(
                                cur, 1,
                                static_cast<uint32_t>(pre + 1 + i), &fp);
                            ids += ' ' + std::to_string(cur);
                        }
                        const double fd_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - fd_t0)
                                                .count();
                        fprintf(stderr,
                                "[glm52-free-decode] %d steps in %.2f s = "
                                "%.3f tok/s at depth %zu; ids:%s\n",
                                n_free, fd_s, n_free / fd_s, pre + 1,
                                ids.c_str());
                    }
                }
            }
            free_sequence(1);
            return tok;
        }
        for (size_t i = start; i < prompt.size(); ++i) {
            if (ckpt_step > 0 && i == ckpt_step && start == 0) {
                if (seq_ckpt(lipc::CMD_SEQ_SNAPSHOT,
                             static_cast<uint32_t>(ckpt_step)))
                    fprintf(stderr, "[glm52-longctx] checkpoint saved @%zu "
                            "to %s\n", ckpt_step, ckpt_path);
                else
                    ADD_FAILURE() << "checkpoint save failed";
                if (::testing::Test::HasFailure()) break;
            }
            // Second fixture: save even mid-restore-run (skip if it already
            // exists — reruns must not clobber the fixture their restore
            // depends on). The daemon reads LS_SEQ_CKPT_PATH per command,
            // so temporarily retarget it, then restore the primary path.
            if (ckpt2_step > 0 && i == ckpt2_step
                && !fs::exists(ckpt2_path)) {
                setenv("LS_SEQ_CKPT_PATH", ckpt2_path, 1);
                if (seq_ckpt(lipc::CMD_SEQ_SNAPSHOT,
                             static_cast<uint32_t>(ckpt2_step)))
                    fprintf(stderr, "[glm52-longctx] checkpoint2 saved @%zu "
                            "to %s\n", ckpt2_step, ckpt2_path);
                else
                    ADD_FAILURE() << "checkpoint2 save failed";
                if (ckpt_path && *ckpt_path)
                    setenv("LS_SEQ_CKPT_PATH", ckpt_path, 1);
                if (::testing::Test::HasFailure()) break;
            }
            float p = 0.f;
            tok = decode_step(prompt[i], 1, static_cast<uint32_t>(i), &p);
            if (top1) *top1 = p;
            if (::testing::Test::HasFailure()) break;
            // GLM52_STEP_PRINT=1: per-step argmax+top1 trace (A/B divergence
            // bisection — teacher forcing feeds identical inputs, so the
            // FIRST divergent step localizes a numeric break to a context
            // length). Default: every 100 steps.
            static const bool step_print = [] {
                const char* sp = std::getenv("GLM52_STEP_PRINT");
                return sp && *sp == '1';
            }();
            if (step_print)
                fprintf(stderr,
                        "[glm52-longctx] step %zu argmax=%u top1=%.4f\n",
                        i + 1, tok, p);
            else if ((i + 1) % 100 == 0 || i + 1 == prompt.size())
                fprintf(stderr,
                        "[glm52-longctx] step %zu/%zu done (argmax=%u)\n",
                        i + 1, prompt.size(), tok);
        }
        free_sequence(1);
        return tok;
    }

    void run_golden(const std::string& strategy) {
        const char* tok_path = std::getenv("GLM52_LONGCTX_TOKENS");
        if (!tok_path || !*tok_path)
            GTEST_SKIP() << "set GLM52_LONGCTX_TOKENS to a prompt token-id "
                            "file (one decimal id per line)";
        const char* golden_env = std::getenv("GLM52_LONGCTX_GOLDEN");
        if (!golden_env || !*golden_env)
            GTEST_SKIP() << "set GLM52_LONGCTX_GOLDEN to the expected greedy "
                            "next-token id";
        if (!fs::exists(tok_path))
            GTEST_SKIP() << "token file not found: " << tok_path;
        const std::vector<uint32_t> prompt = read_token_file(tok_path);
        if (prompt.empty())
            GTEST_SKIP() << "token file empty/unparsable: " << tok_path;
        uint32_t golden = 0;
        try {
            golden = static_cast<uint32_t>(std::stoul(golden_env));
        } catch (...) {
            GTEST_SKIP() << "GLM52_LONGCTX_GOLDEN not a token id: "
                         << golden_env;
        }
        if (prompt.size() <= 2048)
            fprintf(stderr,
                    "[glm52-longctx] WARNING: prompt has %zu tokens <= "
                    "index_topk (2048) — the DSA top-k will not prune\n",
                    prompt.size());

        if (count_big_sm120(28.0) < 2)
            GTEST_SKIP() << "need TWO SM120+ GPUs with >=28 GB visible (TP=2)";
        const std::string src = LAYERSTORM_SOURCE_DIR;
        if (!fs::exists(src + kGgufRel))
            GTEST_SKIP() << "GLM-5.2 GGUF not present";

        start_engine(strategy);
        if (::testing::Test::HasFailure()) return;

        fprintf(stderr,
                "[glm52-longctx] strategy=%s layers=%d experts=%d "
                "prompt_tokens=%zu\n",
                strategy.c_str(), num_layers_, num_experts_, prompt.size());

        float top1 = 0.f;
        uint32_t tok = run_prompt(prompt, &top1);
        if (::testing::Test::HasFailure()) return;

        fprintf(stderr, "[glm52-longctx] strategy=%s next_token=%u "
                "top1_prob=%.4f (expected %u)\n",
                strategy.c_str(), tok, top1, golden);
        if (std::getenv("GLM52_LONGCTX_PREFILL_LIMIT")) return;  // perf probe
        EXPECT_EQ(tok, golden)
            << "strategy=" << strategy << " produced token " << tok
            << ", expected " << golden;
    }

    // Permanent regression for TD-KV-REPLICATED-PAGE-ALIAS (INV-KV-REP).
    // A 327-token prompt embeds an in-context secret code near the start
    // ("The secret code is 4917.") then asks for it back; the model must
    // RETRIEVE it (greedy next token 101474 = "49") at TP=2 REPLICATED KV.
    // This is an IN-CONTEXT recall, NOT a parametric completion — pre-fix, the
    // aliased replicated block table clobbered mid-context KV past chunk 16 so
    // the engine hallucinated (token 18 @0.14) instead of retrieving (101474
    // @0.998). The " Paris" goldens are parametric and structurally cannot
    // catch this class of corruption; this retrieval test can. Tokens live in
    // tests/assets/glm52_needle_retrieval_tokens.txt (llama.cpp-tokenized).
    void run_needle(const std::string& strategy) {
        const std::string path = std::string(LAYERSTORM_SOURCE_DIR)
            + "/tests/assets/glm52_needle_retrieval_tokens.txt";
        if (!fs::exists(path))
            GTEST_SKIP() << "needle asset missing: " << path;
        const std::vector<uint32_t> prompt = read_token_file(path);
        if (prompt.empty())
            GTEST_SKIP() << "needle asset empty/unparsable: " << path;
        constexpr uint32_t kNeedleGolden = 101474;  // "49" of the secret code

        if (count_big_sm120(28.0) < 2)
            GTEST_SKIP() << "need TWO SM120+ GPUs with >=28 GB visible (TP=2)";
        const std::string src = LAYERSTORM_SOURCE_DIR;
        if (!fs::exists(src + kGgufRel))
            GTEST_SKIP() << "GLM-5.2 GGUF not present";

        // Mode: replicated by default; GLM52_KV_SHARDED=1 runs the SAME
        // retrieval under sequence-sharded KV — the needle sits mid-context,
        // so once TD-KVS-Q-ALLGATHER lands, successful recall here proves KV
        // integrity THROUGH the sharded DCP combine (both shards contribute).
        // Until then the sharded run documents the known head-mixing failure.
        const char* ks = std::getenv("GLM52_KV_SHARDED");
        const bool sharded = ks && *ks == '1';
        start_engine(strategy);
        if (::testing::Test::HasFailure()) return;
        fprintf(stderr, "[glm52-needle] strategy=%s prompt_tokens=%zu "
                "(in-context recall @ TP=2 %s)\n",
                strategy.c_str(), prompt.size(),
                sharded ? "SHARDED" : "replicated");

        float top1 = 0.f;
        uint32_t tok = run_prompt(prompt, &top1);
        if (::testing::Test::HasFailure()) return;
        fprintf(stderr, "[glm52-needle] strategy=%s next_token=%u "
                "top1_prob=%.4f (expected %u)\n",
                strategy.c_str(), tok, top1, kNeedleGolden);
        if (std::getenv("GLM52_LONGCTX_PREFILL_LIMIT")) return;  // perf probe
        EXPECT_EQ(tok, kNeedleGolden)
            << "in-context recall failed (TD-KV-REPLICATED-PAGE-ALIAS "
               "regression): produced " << tok << ", expected "
            << kNeedleGolden << " — mid-context KV may be corrupted";
    }

    // TD-PREFILL-SUPERCHUNK: prefill perf/telemetry summary — expert H2D
    // volume (keeper-LRU misses × expert bytes) + routing concentration
    // (mean unique experts per MoE layer call — the dedup/saturation metric).
    void prefill_perf_summary(
        std::chrono::steady_clock::time_point t0) const {
        const double wall_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        const double eb = static_cast<double>(engine_->info().expert_bytes);
        fprintf(stderr,
                "[glm52-prefill-perf] moe_layer_calls=%llu "
                "union_sum=%llu (mean unique experts/layer-call=%.1f) "
                "lru_miss_fetches=%llu expert_h2d=%.2f GB wall=%.1f s\n",
                static_cast<unsigned long long>(moe_layer_calls_),
                static_cast<unsigned long long>(union_sum_),
                moe_layer_calls_ ? static_cast<double>(union_sum_)
                                       / moe_layer_calls_ : 0.0,
                static_cast<unsigned long long>(lru_miss_fetches_),
                lru_miss_fetches_ * eb / 1e9, wall_s);
    }

    std::unique_ptr<ldam::Engine> engine_;
    std::unique_ptr<lipc::CommandRing> cmd_ring_;
    std::unique_ptr<lipc::CompletionRing> cmp_ring_;
    uint8_t* sideband_ = nullptr;
    std::string config_path_;
    uint32_t cmd_seq_ = 1;
    int num_layers_ = 0, num_experts_ = 0, first_moe_layer_ = 3;
    int vocab_size_ = 0;
    int max_batch_ = 64;   // orchestrator.max_batch_size (prefill chunk cap)
    int superchunk_k_ = 0; // TD-PREFILL-SUPERCHUNK: GLM52_SUPERCHUNK_K
    uint64_t lru_miss_fetches_ = 0;  // keeper-LRU misses (= expert H2D fetches)
    uint64_t union_sum_ = 0;         // Σ per-layer routed-union sizes
    uint64_t moe_layer_calls_ = 0;   // FETCH_AND_RUN calls counted in union_sum_
    uint32_t hidden_buf_id_ = 0, logits_buf_id_ = 0;
    std::vector<GpuLru> lrus_;
};

// int is the engine default; one full pass cold-streams ~420 GB (hours), so
// dequant runs only on explicit request.
TEST_F(Glm52LongCtxGolden, IntGolden) {
    const char* s = std::getenv("GG_STRATEGY");
    if (s && std::string(s) != "int") GTEST_SKIP() << "GG_STRATEGY!=int";
    run_golden("int");
}

TEST_F(Glm52LongCtxGolden, DequantGolden) {
    const char* s = std::getenv("GG_STRATEGY");
    if (!s || std::string(s) != "dequant")
        GTEST_SKIP() << "set GG_STRATEGY=dequant to run";
    run_golden("dequant");
}

// In-context recall regression (TD-KV-REPLICATED-PAGE-ALIAS / INV-KV-REP).
// Self-contained (no env vars): runs when two >=28 GB SM120 GPUs + the GGUF
// are present. ~8 min (init + 327-token teacher-forcing, no checkpoint).
TEST_F(Glm52LongCtxGolden, NeedleRetrievalReplicated) {
    const char* s = std::getenv("GG_STRATEGY");
    if (s && std::string(s) != "int") GTEST_SKIP() << "GG_STRATEGY!=int";
    run_needle("int");
}

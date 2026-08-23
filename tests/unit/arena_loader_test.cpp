// ArenaLoader unit tests (J-1).
//
// Exercises the async pinned-arena cold-load worker pool: it copies expert
// packed bytes from a (real, mock-prepacked) PrepackedSource into caller-owned
// destination buffers off the daemon thread, dedups concurrent submits for the
// same key, and reports completions via a pollable queue. No GPU needed — the
// destination buffers here are plain heap memory (the real arena slot is just
// pinned host RAM; ArenaLoader does a memcpy and never touches CUDA).

#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "core/memory/arena_loader.h"
#include "core/memory/eviction_policy.h"
#include "model/model_config.h"
#include "model/quantization/nvfp4.h"
#include "model/weight_pipeline/expert_prepacker.h"
#include "model/weight_pipeline/prepacked_source.h"
#include "model/weight_loader/weight_loader.h"
#include "weight_pipeline_test_helpers.h"

namespace fs = std::filesystem;
using namespace layerstorm::model;
namespace lm = layerstorm::memory;

namespace {
lm::ExpertKey key(uint32_t layer, uint16_t expert) { return {layer, expert}; }
}

class ArenaLoaderTest : public ::testing::Test,
                        public layerstorm::test::WeightPipelineHelpers {
protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "layerstorm_arena_loader_test";
        fs::create_directories(tmp_dir_);
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmp_dir_, ec);
    }

    // Build a small NVFP4 prepacked source.
    std::unique_ptr<PrepackedSource> make_source(int n_layers, int n_experts,
                                                 int hidden, int intermediate,
                                                 int first_moe) {
        ExpertShape shape{hidden, intermediate};
        auto cfg = make_config(n_layers, n_experts, hidden, intermediate, first_moe);
        ModelConfig model_cfg(cfg);
        Nvfp4 quant;
        auto model = make_mock_model(n_layers, n_experts, shape, first_moe,
                                     SafetensorsDtype::U8);
        auto out = tmp_dir_ / "src";
        auto result = prepack_experts(model, model_cfg, quant, cfg, out);
        EXPECT_TRUE(result.error.empty()) << result.error;
        quant_ = std::make_unique<Nvfp4>();
        return std::make_unique<PrepackedSource>(out, *quant_);
    }

    fs::path tmp_dir_;
    std::unique_ptr<Nvfp4> quant_;
};

// All submitted loads complete; each destination equals the source bytes.
TEST_F(ArenaLoaderTest, LoadsAllSubmittedAndMatchesSource) {
    constexpr int kLayers = 4, kFirstMoe = 2, kExperts = 4;
    constexpr int kHidden = 64, kIntermediate = 32;
    auto src = make_source(kLayers, kExperts, kHidden, kIntermediate, kFirstMoe);
    const size_t slot = static_cast<size_t>(src->slot_size_bytes());
    ASSERT_GT(slot, 0u);

    lm::ArenaLoader loader(2);
    EXPECT_GE(loader.num_workers(), 1);

    std::vector<lm::ExpertKey> keys;
    std::vector<std::vector<uint8_t>> dsts;
    for (int e = 0; e < kExperts; ++e) {
        keys.push_back(key(kFirstMoe, static_cast<uint16_t>(e)));
        dsts.emplace_back(slot, 0);
    }
    for (int e = 0; e < kExperts; ++e) {
        EXPECT_TRUE(loader.submit(keys[e], /*gpu=*/e % 2, dsts[e].data(), src.get()));
    }

    // Poll until all complete (bounded).
    std::vector<lm::ArenaLoadCompletion> done;
    for (int i = 0; i < 10000 && done.size() < dsts.size(); ++i) {
        loader.poll_completed(done);
        if (done.size() < dsts.size()) std::this_thread::yield();
    }
    ASSERT_EQ(done.size(), dsts.size());
    for (const auto& c : done) EXPECT_TRUE(c.success);
    EXPECT_EQ(loader.pending_count(), 0u);

    // Each destination must byte-match the prepacked source slot.
    for (int e = 0; e < kExperts; ++e) {
        const void* sp = src->resolve(keys[e]);
        ASSERT_NE(sp, nullptr);
        EXPECT_EQ(std::memcmp(dsts[e].data(), sp, slot), 0)
            << "mismatch expert " << e;
    }
}

// A second submit for an already-in-flight key is rejected (dedup): the slot is
// filled at most once per outstanding load.
TEST_F(ArenaLoaderTest, DedupsConcurrentSubmitForSameKey) {
    auto src = make_source(4, 2, 64, 32, 2);
    const size_t slot = static_cast<size_t>(src->slot_size_bytes());

    lm::ArenaLoader loader(1);  // single worker so the first job stays queued
    auto k = key(2, 0);
    std::vector<uint8_t> d1(slot, 0), d2(slot, 0);

    EXPECT_TRUE(loader.submit(k, 0, d1.data(), src.get()));
    // Immediately resubmit before the worker can finish — must be deduped.
    bool second = loader.submit(k, 0, d2.data(), src.get());
    // It is rejected unless the first already completed; in either case only one
    // completion for k is ever produced.
    std::vector<lm::ArenaLoadCompletion> done;
    for (int i = 0; i < 10000 && done.empty(); ++i) {
        loader.poll_completed(done);
        if (done.empty()) std::this_thread::yield();
    }
    // Drain any stragglers.
    for (int i = 0; i < 100; ++i) { loader.poll_completed(done); std::this_thread::yield(); }

    int k_completions = 0;
    for (const auto& c : done) if (c.key == k) ++k_completions;
    if (!second) {
        EXPECT_EQ(k_completions, 1) << "deduped submit must yield exactly one load";
    } else {
        // Rare race: first finished before the resubmit; then 2 are allowed.
        EXPECT_GE(k_completions, 1);
    }
    EXPECT_EQ(loader.pending_count(), 0u);
}

// Null arguments are rejected; destructor joins cleanly with jobs outstanding.
TEST_F(ArenaLoaderTest, RejectsNullAndDestructorDrains) {
    auto src = make_source(4, 2, 64, 32, 2);
    const size_t slot = static_cast<size_t>(src->slot_size_bytes());
    std::vector<uint8_t> d(slot, 0);

    {
        lm::ArenaLoader loader(2);
        EXPECT_FALSE(loader.submit(key(2, 0), 0, nullptr, src.get()));
        EXPECT_FALSE(loader.submit(key(2, 0), 0, d.data(), nullptr));
        // Submit a real job and let the destructor join with it possibly in flight.
        EXPECT_TRUE(loader.submit(key(2, 1), 0, d.data(), src.get()));
    }  // ~ArenaLoader() must join without hanging or crashing.
    SUCCEED();
}

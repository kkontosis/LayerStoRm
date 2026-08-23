#include "compute/stream_manager.h"

#include <memory>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "core/gpu_ref.h"
#include "core/null_device_backend.h"

namespace lc = layerstorm::compute;

static std::vector<layerstorm::config::GpuRef> make_gpu_refs(int count) {
    std::vector<layerstorm::config::GpuRef> v;
    for (int i = 0; i < count; ++i)
        v.push_back({.position = i, .id = i, .type = layerstorm::config::GpuType::rtx5090});
    return v;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Bundles owned NullDeviceBackend instances with the Options that reference
/// them, keeping pointers valid for the duration of the test.
struct TestContext {
    std::vector<std::unique_ptr<lc::DeviceBackend>> backends;
    lc::StreamManager::Options opts;
};

static TestContext make_test_context(int num_gpus = 2) {
    TestContext ctx;
    auto gpus = make_gpu_refs(num_gpus);
    std::vector<lc::DeviceBackend*> ptrs;
    for (int i = 0; i < num_gpus; ++i) {
        ctx.backends.push_back(lc::make_null_device_backend(gpus[i]));
        ptrs.push_back(ctx.backends.back().get());
    }
    ctx.opts = lc::StreamManager::Options{.device_backends = ptrs};
    return ctx;
}

/// Backend where query_event returns a controllable flag.
class DeferredNullDeviceBackend : public lc::NullDeviceBackend {
public:
    DeferredNullDeviceBackend(layerstorm::config::GpuRef gpu, bool& ready)
        : NullDeviceBackend(std::move(gpu)), event_ready_(ready) {}
    lc::EventQueryResult query_event(void*) override {
        return {event_ready_ ? lc::EventStatus::kReady : lc::EventStatus::kNotReady, 0};
    }
private:
    bool& event_ready_;
};

static constexpr int kNumStreams = static_cast<int>(lc::StreamId::kCount);

// ── Construction tests ──────────────────────────────────────────────────────

TEST(StreamManager, ConstructionValid) {
    auto tc = make_test_context(2);
    lc::StreamManager mgr(tc.opts);
    EXPECT_EQ(mgr.num_gpus(), 2);
}

TEST(StreamManager, ConstructionFourGpus) {
    auto tc = make_test_context(4);
    lc::StreamManager mgr(tc.opts);
    EXPECT_EQ(mgr.num_gpus(), 4);
}

TEST(StreamManager, ConstructionZeroGpuThrows) {
    auto tc = make_test_context(0);
    EXPECT_THROW(lc::StreamManager mgr(tc.opts), std::invalid_argument);
}

TEST(StreamManager, ConstructionNegativeGpuThrows) {
    // make_test_context clamps to 0 backends for negative input
    auto tc = make_test_context(0);
    EXPECT_THROW(lc::StreamManager mgr(tc.opts), std::invalid_argument);
}

// ── Stream access tests ─────────────────────────────────────────────────────

TEST(StreamManager, StreamReturnsNonNull) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    EXPECT_NE(mgr.stream(0, lc::StreamId::kAttention), nullptr);
}

TEST(StreamManager, AllStreamsDistinctPerGpu) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    std::set<void*> ptrs;
    for (int s = 0; s < kNumStreams; ++s) {
        void* p = mgr.stream(0, static_cast<lc::StreamId>(s));
        EXPECT_NE(p, nullptr);
        ptrs.insert(p);
    }
    EXPECT_EQ(ptrs.size(), static_cast<size_t>(kNumStreams));
}

TEST(StreamManager, SameStreamIdDifferentGpuDistinct) {
    auto tc = make_test_context(2);
    lc::StreamManager mgr(tc.opts);
    EXPECT_NE(mgr.stream(0, lc::StreamId::kAttention),
              mgr.stream(1, lc::StreamId::kAttention));
}

TEST(StreamManager, AllStreamsDistinctAcrossGpus) {
    auto tc = make_test_context(4);
    lc::StreamManager mgr(tc.opts);
    std::set<void*> ptrs;
    for (int g = 0; g < 4; ++g) {
        for (int s = 0; s < kNumStreams; ++s) {
            ptrs.insert(mgr.stream(g, static_cast<lc::StreamId>(s)));
        }
    }
    EXPECT_EQ(ptrs.size(), static_cast<size_t>(4 * kNumStreams));
}

TEST(StreamManager, InvalidGpuIdxThrows) {
    auto tc = make_test_context(2);
    lc::StreamManager mgr(tc.opts);
    EXPECT_THROW(mgr.stream(2, lc::StreamId::kAttention), std::out_of_range);
    EXPECT_THROW(mgr.stream(-1, lc::StreamId::kAttention), std::out_of_range);
}

TEST(StreamManager, InvalidStreamIdThrows) {
    auto tc = make_test_context(2);
    lc::StreamManager mgr(tc.opts);
    EXPECT_THROW(mgr.stream(0, static_cast<lc::StreamId>(99)), std::out_of_range);
    EXPECT_THROW(mgr.stream(0, static_cast<lc::StreamId>(-1)), std::out_of_range);
}

// ── Event lifecycle tests ───────────────────────────────────────────────────

TEST(StreamManager, CreateEventReturnsNonNull) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);
    EXPECT_NE(event, nullptr);
    mgr.destroy_event(event, 0);
}

TEST(StreamManager, DestroyEventNoThrow) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);
    EXPECT_NO_THROW(mgr.destroy_event(event, 0));
}

TEST(StreamManager, QueryEventImmediatelyComplete) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);
    EXPECT_EQ(mgr.query_event(event, 0).status, lc::EventStatus::kReady);
    mgr.destroy_event(event, 0);
}

// ── Event record/query tests ────────────────────────────────────────────────

TEST(StreamManager, RecordEventOnStream) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);
    EXPECT_NO_THROW(mgr.record_event(event, 0, lc::StreamId::kAttention));
    EXPECT_EQ(mgr.query_event(event, 0).status, lc::EventStatus::kReady);
    mgr.destroy_event(event, 0);
}

TEST(StreamManager, RecordEventAllStreams) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    for (int s = 0; s < kNumStreams; ++s) {
        void* event = mgr.create_event(0);
        auto sid = static_cast<lc::StreamId>(s);
        EXPECT_NO_THROW(mgr.record_event(event, 0, sid));
        EXPECT_EQ(mgr.query_event(event, 0).status, lc::EventStatus::kReady);
        mgr.destroy_event(event, 0);
    }
}

TEST(StreamManager, RecordEventInvalidGpuThrows) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);
    EXPECT_THROW(mgr.record_event(event, 99, lc::StreamId::kAttention),
                 std::out_of_range);
    mgr.destroy_event(event, 0);
}

TEST(StreamManager, RecordEventInvalidStreamThrows) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);
    EXPECT_THROW(mgr.record_event(event, 0, static_cast<lc::StreamId>(99)),
                 std::out_of_range);
    mgr.destroy_event(event, 0);
}

// ── wait_event tests ────────────────────────────────────────────────────────

TEST(StreamManager, WaitEventValidStreamAndEvent) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);
    mgr.record_event(event, 0, lc::StreamId::kAttention);
    EXPECT_NO_THROW(mgr.wait_event(0, lc::StreamId::kExpertFfn, event));
    mgr.destroy_event(event, 0);
}

TEST(StreamManager, WaitEventInvalidGpuThrows) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);
    EXPECT_THROW(mgr.wait_event(99, lc::StreamId::kAttention, event),
                 std::out_of_range);
    mgr.destroy_event(event, 0);
}

TEST(StreamManager, WaitEventInvalidStreamThrows) {
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);
    EXPECT_THROW(mgr.wait_event(0, static_cast<lc::StreamId>(99), event),
                 std::out_of_range);
    mgr.destroy_event(event, 0);
}

// ── Deferred completion tests ───────────────────────────────────────────────

TEST(StreamManager, DeferredEventNotComplete) {
    bool event_ready = false;
    auto gpus = make_gpu_refs(2);
    std::vector<std::unique_ptr<lc::DeviceBackend>> backends;
    std::vector<lc::DeviceBackend*> ptrs;
    for (int i = 0; i < 2; ++i) {
        backends.push_back(std::make_unique<DeferredNullDeviceBackend>(
            gpus[i], event_ready));
        ptrs.push_back(backends.back().get());
    }

    lc::StreamManager mgr(lc::StreamManager::Options{.device_backends = ptrs});
    void* event = mgr.create_event(0);
    mgr.record_event(event, 0, lc::StreamId::kAttention);

    EXPECT_EQ(mgr.query_event(event, 0).status, lc::EventStatus::kNotReady);

    event_ready = true;
    EXPECT_EQ(mgr.query_event(event, 0).status, lc::EventStatus::kReady);

    mgr.destroy_event(event, 0);
}

// ── Spec synchronization pattern tests ──────────────────────────────────────

TEST(StreamManager, EventAPattern_AttentionToGatingAndPrefetch) {
    // Event A: Attention for layer L complete → triggers gating + PreScope.
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);

    // Record on attention stream.
    mgr.record_event(event, 0, lc::StreamId::kAttention);

    // Gating and prefetch compute wait on the event.
    EXPECT_NO_THROW(mgr.wait_event(0, lc::StreamId::kGating, event));
    EXPECT_NO_THROW(mgr.wait_event(0, lc::StreamId::kPrefetchCompute, event));

    mgr.destroy_event(event, 0);
}

TEST(StreamManager, EventBPattern_TransferToExpertFfn) {
    // Event B: Expert transfer complete → enables expert FFN dispatch.
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);

    mgr.record_event(event, 0, lc::StreamId::kH2dTransfer);
    EXPECT_NO_THROW(mgr.wait_event(0, lc::StreamId::kExpertFfn, event));

    mgr.destroy_event(event, 0);
}

TEST(StreamManager, EventCPattern_ExpertFfnToAttention) {
    // Event C: Expert FFN complete → triggers attention for layer L+1.
    auto tc = make_test_context();
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);

    mgr.record_event(event, 0, lc::StreamId::kExpertFfn);
    EXPECT_NO_THROW(mgr.wait_event(0, lc::StreamId::kAttention, event));

    mgr.destroy_event(event, 0);
}

// ── Multi-GPU tests ─────────────────────────────────────────────────────────

TEST(StreamManager, FourGpuAllStreamsCreated) {
    auto tc = make_test_context(4);
    lc::StreamManager mgr(tc.opts);
    for (int g = 0; g < 4; ++g) {
        for (int s = 0; s < kNumStreams; ++s) {
            EXPECT_NE(mgr.stream(g, static_cast<lc::StreamId>(s)), nullptr);
        }
    }
}

TEST(StreamManager, RecordAndQueryAcrossGpus) {
    auto tc = make_test_context(4);
    lc::StreamManager mgr(tc.opts);
    void* event = mgr.create_event(0);

    // Record on GPU 0 attention, query (GPU-independent for events).
    mgr.record_event(event, 0, lc::StreamId::kAttention);
    EXPECT_EQ(mgr.query_event(event, 0).status, lc::EventStatus::kReady);

    // Wait from a different GPU's stream.
    EXPECT_NO_THROW(mgr.wait_event(2, lc::StreamId::kExpertFfn, event));

    mgr.destroy_event(event, 0);
}

// ── StreamId enum tests ─────────────────────────────────────────────────────

TEST(StreamManager, StreamIdValues) {
    EXPECT_EQ(static_cast<int>(lc::StreamId::kAttention), 0);
    EXPECT_EQ(static_cast<int>(lc::StreamId::kExpertFfn), 1);
    EXPECT_EQ(static_cast<int>(lc::StreamId::kGating), 2);
    EXPECT_EQ(static_cast<int>(lc::StreamId::kH2dTransfer), 3);
    EXPECT_EQ(static_cast<int>(lc::StreamId::kD2hTransfer), 4);
    EXPECT_EQ(static_cast<int>(lc::StreamId::kPrefetchCompute), 5);
    EXPECT_EQ(static_cast<int>(lc::StreamId::kAsyncDequant), 6);
    EXPECT_EQ(static_cast<int>(lc::StreamId::kCount), 7);
}

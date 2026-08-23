#include "core/transfer/transfer_engine.h"

#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "core/device_backend.h"
#include "core/gpu_ref.h"
#include "core/null_device_backend.h"

namespace lt = layerstorm::transfer;
namespace lc = layerstorm::compute;
namespace lmem = layerstorm::memory;

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
    lt::TransferEngine::Options opts;
};

static TestContext make_test_context(int num_gpus = 2) {
    TestContext ctx;
    auto gpus = make_gpu_refs(num_gpus);
    std::vector<lc::DeviceBackend*> ptrs;
    for (int i = 0; i < num_gpus; ++i) {
        ctx.backends.push_back(lc::make_null_device_backend(gpus[i]));
        ptrs.push_back(ctx.backends.back().get());
    }
    ctx.opts.device_backends = ptrs;
    // Gen5 x16 for all GPUs by default
    for (int i = 0; i < num_gpus; ++i) {
        ctx.opts.pcie_info.push_back({5, 16});
    }
    return ctx;
}

/// Backend with directly controllable EventStatus for error-path tests.
class StatusDeviceBackend : public lc::NullDeviceBackend {
public:
    StatusDeviceBackend(layerstorm::config::GpuRef gpu, lc::EventStatus& status)
        : NullDeviceBackend(std::move(gpu)), status_(status) {}
    lc::EventQueryResult query_event(void*) override { return {status_, 0}; }
private:
    lc::EventStatus& status_;
};

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

struct DeferredTestContext {
    std::vector<std::unique_ptr<lc::DeviceBackend>> backends;
    lt::TransferEngine::Options opts;
};

static DeferredTestContext make_deferred_context(bool& event_ready,
                                                  int num_gpus = 1) {
    DeferredTestContext ctx;
    auto gpus = make_gpu_refs(num_gpus);
    std::vector<lc::DeviceBackend*> ptrs;
    for (int i = 0; i < num_gpus; ++i) {
        ctx.backends.push_back(std::make_unique<DeferredNullDeviceBackend>(
            gpus[i], event_ready));
        ptrs.push_back(ctx.backends.back().get());
    }
    ctx.opts.device_backends = ptrs;
    return ctx;
}

static lmem::ExpertKey ek(uint32_t layer, uint16_t expert) {
    return lmem::ExpertKey{layer, expert};
}

// ── Tests ───────────────────────────────────────────────────────────────────

TEST(TransferEngine, ConstructionValid) {
    auto tc = make_test_context(2);
    lt::TransferEngine engine(tc.opts);
    EXPECT_EQ(engine.inflight_count(), 0);
}

TEST(TransferEngine, ConstructionZeroGpuThrows) {
    auto tc = make_test_context(0);
    EXPECT_THROW(lt::TransferEngine engine(tc.opts), std::invalid_argument);
}

TEST(TransferEngine, EnqueueReturnsToken) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> src(1024, 0xAB);
    std::vector<uint8_t> dst(1024, 0);

    auto token = engine.enqueue_h2d(ek(0, 0), 0, dst.data(), src.data(), 1024);
    ASSERT_TRUE(token.has_value());
    EXPECT_GE(*token, 1u);
}

TEST(TransferEngine, NullBackendCopiesData) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> src(1024, 0xAB);
    std::vector<uint8_t> dst(1024, 0);

    engine.enqueue_h2d(ek(0, 0), 0, dst.data(), src.data(), 1024);
    auto completions = engine.poll_completions();

    ASSERT_EQ(completions.size(), 1u);
    EXPECT_TRUE(completions[0].success);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 1024), 0);
}

TEST(TransferEngine, DedupSameKey) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(64);

    auto t1 = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    EXPECT_TRUE(t1.has_value());

    // With null backend, first enqueue completes immediately.
    // But the dedup map is cleared only after poll_completions().
    // Actually with null backend, query_event returns true immediately,
    // but poll hasn't been called yet, so it's still in the dedup map.
    auto t2 = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    EXPECT_FALSE(t2.has_value());

    // After poll, re-enqueue succeeds.
    engine.poll_completions();
    auto t3 = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    EXPECT_TRUE(t3.has_value());
}

TEST(TransferEngine, DifferentKeysNotDeduped) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(64);

    auto t1 = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    auto t2 = engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64);
    EXPECT_TRUE(t1.has_value());
    EXPECT_TRUE(t2.has_value());
    EXPECT_NE(*t1, *t2);
}

TEST(TransferEngine, H2dAndD2hIndependent) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(64);

    auto t1 = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    auto t2 = engine.enqueue_d2h(ek(0, 0), 0, buf.data(), buf.data(), 64);
    EXPECT_TRUE(t1.has_value());
    EXPECT_TRUE(t2.has_value());
    EXPECT_NE(*t1, *t2);

    EXPECT_TRUE(engine.is_inflight_h2d(ek(0, 0), 0));
    EXPECT_TRUE(engine.is_inflight_d2h(ek(0, 0), 0));
}

TEST(TransferEngine, MultiGpuSameKeyDifferentGpu) {
    auto tc = make_test_context(2);
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(64);

    auto t1 = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    auto t2 = engine.enqueue_h2d(ek(0, 0), 1, buf.data(), buf.data(), 64);
    EXPECT_TRUE(t1.has_value());
    EXPECT_TRUE(t2.has_value());
}

TEST(TransferEngine, InflightCountPerGpu) {
    auto tc = make_test_context(2);
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(64);

    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64);
    engine.enqueue_h2d(ek(0, 2), 1, buf.data(), buf.data(), 64);

    EXPECT_EQ(engine.inflight_count(0), 2);
    EXPECT_EQ(engine.inflight_count(1), 1);
    EXPECT_EQ(engine.inflight_count(), 3);
}

TEST(TransferEngine, CompletionDetails) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(128);

    engine.enqueue_h2d(ek(2, 5), 0, buf.data(), buf.data(), 128);
    auto completions = engine.poll_completions();

    ASSERT_EQ(completions.size(), 1u);
    EXPECT_EQ(completions[0].key, ek(2, 5));
    EXPECT_EQ(completions[0].gpu_idx, 0);
    EXPECT_EQ(completions[0].direction, lt::TransferDirection::kH2D);
    EXPECT_EQ(completions[0].bytes, 128);
    EXPECT_TRUE(completions[0].success);
}

TEST(TransferEngine, CallbackFires) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(64);

    bool callback_fired = false;
    lt::TransferToken cb_token = 0;
    bool cb_success = false;

    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 0.0f, 0,
        [&](lt::TransferToken t, bool s) {
            callback_fired = true;
            cb_token = t;
            cb_success = s;
        });

    engine.poll_completions();
    EXPECT_TRUE(callback_fired);
    EXPECT_GE(cb_token, 1u);
    EXPECT_TRUE(cb_success);
}

TEST(TransferEngine, DrainCompletesAll) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(64);

    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64);
    engine.enqueue_d2h(ek(0, 2), 1, buf.data(), buf.data(), 64);

    engine.drain();
    EXPECT_EQ(engine.inflight_count(), 0);
}

TEST(TransferEngine, PcieBandwidthGen5x16) {
    double bw = lt::TransferEngine::pcie_bandwidth_gbps(5, 16);
    EXPECT_NEAR(bw, 63.0, 0.5);
}

TEST(TransferEngine, PcieBandwidthGen4x16) {
    double bw = lt::TransferEngine::pcie_bandwidth_gbps(4, 16);
    EXPECT_NEAR(bw, 31.5, 0.5);
}

TEST(TransferEngine, EstimateTransferUs) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    double us = engine.estimate_transfer_us(0, 1024);
    EXPECT_GT(us, 0.0);
    EXPECT_LT(us, 1.0);  // Sub-microsecond
}

TEST(TransferEngine, DifferentPciePerGpu) {
    auto gpus = make_gpu_refs(2);
    std::vector<std::unique_ptr<lc::DeviceBackend>> backends;
    std::vector<lc::DeviceBackend*> ptrs;
    for (int i = 0; i < 2; ++i) {
        backends.push_back(lc::make_null_device_backend(gpus[i]));
        ptrs.push_back(backends.back().get());
    }
    lt::TransferEngine::Options opts;
    opts.device_backends = ptrs;
    opts.pcie_info = {{5, 16}, {4, 16}};

    lt::TransferEngine engine(std::move(opts));

    int64_t bytes = 1'000'000;
    double us_gen5 = engine.estimate_transfer_us(0, bytes);
    double us_gen4 = engine.estimate_transfer_us(1, bytes);

    // Gen4 is ~half the bandwidth of gen5, so ~double the time.
    EXPECT_NEAR(us_gen4 / us_gen5, 2.0, 0.01);
}

// ── Deferred-completion backend ─────────────────────────────────────────────

TEST(TransferEngine, DeferredCompletionNotReadyUntilEvent) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    EXPECT_EQ(engine.inflight_count(), 1);

    // Event not ready → poll returns nothing.
    auto c1 = engine.poll_completions();
    EXPECT_TRUE(c1.empty());
    EXPECT_EQ(engine.inflight_count(), 1);

    // Now mark ready.
    event_ready = true;
    auto c2 = engine.poll_completions();
    EXPECT_EQ(c2.size(), 1u);
    EXPECT_EQ(engine.inflight_count(), 0);
}

TEST(TransferEngine, DeferredDedupWhileInflight) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    auto t1 = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    EXPECT_TRUE(t1.has_value());

    // Still in-flight → dedup.
    auto t2 = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    EXPECT_FALSE(t2.has_value());

    // Complete, then re-enqueue succeeds.
    event_ready = true;
    engine.poll_completions();

    auto t3 = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    EXPECT_TRUE(t3.has_value());
}

TEST(TransferEngine, MultipleCallbacksFire) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(64);

    int count = 0;
    auto cb = [&](lt::TransferToken, bool) { ++count; };

    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 0.0f, 0, cb);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.0f, 0, cb);
    engine.enqueue_d2h(ek(0, 2), 1, buf.data(), buf.data(), 64, 0.0f, 0, cb);

    engine.poll_completions();
    EXPECT_EQ(count, 3);
}

TEST(TransferEngine, NullCallbackOk) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(64);

    // No callback → should not crash.
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    auto completions = engine.poll_completions();
    EXPECT_EQ(completions.size(), 1u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Cancellation
// ═══════════════════════════════════════════════════════════════════════════

TEST(TransferEngine, Cancel_Success) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> src(64, 0xAA);
    std::vector<uint8_t> dst(64, 0);

    bool callback_fired = false;
    auto token = engine.enqueue_h2d(
        ek(1, 0), 0, dst.data(), src.data(), 64,
        0.0f, 0,
        [&](lt::TransferToken, bool) { callback_fired = true; });
    ASSERT_TRUE(token.has_value());
    EXPECT_EQ(engine.inflight_count(), 1);

    // Cancel before polling.
    auto result = engine.cancel(token.value());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->key, ek(1, 0));
    EXPECT_EQ(result->gpu_idx, 0);
    EXPECT_EQ(result->direction, lt::TransferDirection::kH2D);

    // Transfer is gone.
    EXPECT_EQ(engine.inflight_count(), 0);
    EXPECT_FALSE(engine.is_inflight_h2d(ek(1, 0), 0));

    // Callback should NOT have fired.
    EXPECT_FALSE(callback_fired);

    // Polling should return nothing — transfer was cancelled.
    auto completions = engine.poll_completions();
    EXPECT_TRUE(completions.empty());
}

TEST(TransferEngine, Cancel_NotFound) {
    auto tc = make_test_context();
    lt::TransferEngine engine(tc.opts);

    auto result = engine.cancel(12345);
    EXPECT_FALSE(result.has_value());
}

// ═══════════════════════════════════════════════════════════════════════════
// Priority staging queue (#43b)
// ═══════════════════════════════════════════════════════════════════════════

static TestContext make_staged_context(int num_gpus = 2,
                                        int min_dispatch = 2,
                                        int max_inflight = 8) {
    auto ctx = make_test_context(num_gpus);
    ctx.opts.min_dispatch_per_gpu = min_dispatch;
    ctx.opts.max_inflight_per_gpu = max_inflight;
    return ctx;
}

TEST(TransferEngine, Staging_ImmediateDispatchBelowWatermark) {
    auto tc = make_staged_context(1, 2, 8);
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(64);

    auto t1 = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    auto t2 = engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64);
    EXPECT_TRUE(t1.has_value());
    EXPECT_TRUE(t2.has_value());

    // Both should be in pending (immediately dispatched), not staged.
    EXPECT_EQ(engine.staged_count(0), 0);

    // With null backend, poll completes immediately.
    auto completions = engine.poll_completions();
    EXPECT_EQ(completions.size(), 2u);
}

TEST(TransferEngine, Staging_OverflowGoesToStagingQueue) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.min_dispatch_per_gpu = 2;
    dc.opts.max_inflight_per_gpu = 4;

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    // First 2 are immediate (below watermark).
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64);

    // 3rd goes to staging (above watermark, events not ready).
    engine.enqueue_h2d(ek(0, 2), 0, buf.data(), buf.data(), 64, 0.5f);

    EXPECT_EQ(engine.staged_count(0), 1);

    event_ready = true;  // Allow drain() in destructor to complete.
}

TEST(TransferEngine, Staging_PriorityOrdering) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.min_dispatch_per_gpu = 0;
    dc.opts.max_inflight_per_gpu = 4;

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    // All go to staging (min_dispatch=0).
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 0.1f);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.9f);
    engine.enqueue_h2d(ek(0, 2), 0, buf.data(), buf.data(), 64, 0.5f);
    EXPECT_EQ(engine.staged_count(0), 3);

    // Flush: highest priority first.
    engine.flush_staged();

    // All 3 should now be dispatched (max_inflight=4).
    EXPECT_EQ(engine.staged_count(0), 0);

    // Mark events ready and poll — verify they all complete.
    event_ready = true;
    auto completions = engine.poll_completions();
    EXPECT_EQ(completions.size(), 3u);
}

TEST(TransferEngine, Staging_CompletionDrivenReplenishment) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.min_dispatch_per_gpu = 2;
    dc.opts.max_inflight_per_gpu = 4;

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    // First 2 dispatched immediately.
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 1.0f);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.5f);

    // 3rd and 4th staged.
    engine.enqueue_h2d(ek(0, 2), 0, buf.data(), buf.data(), 64, 0.8f);
    engine.enqueue_h2d(ek(0, 3), 0, buf.data(), buf.data(), 64, 0.3f);
    EXPECT_EQ(engine.staged_count(0), 2);

    // Complete the first 2. Replenishment should auto-promote from staging.
    event_ready = true;
    auto c1 = engine.poll_completions();
    EXPECT_EQ(c1.size(), 2u);

    // After replenishment, staged should have items promoted to pending.
    // With null deferred backend (now event_ready=true), promoted items
    // also complete immediately on next poll.
    EXPECT_LE(engine.staged_count(0), 0);
}

TEST(TransferEngine, Staging_FlushRespectsMaxInflight) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.min_dispatch_per_gpu = 0;
    dc.opts.max_inflight_per_gpu = 2;

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 0.9f);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.8f);
    engine.enqueue_h2d(ek(0, 2), 0, buf.data(), buf.data(), 64, 0.7f);
    EXPECT_EQ(engine.staged_count(0), 3);

    // Flush: only 2 should be dispatched (max_inflight=2).
    engine.flush_staged();
    EXPECT_EQ(engine.staged_count(0), 1);

    event_ready = true;  // Allow drain() in destructor to complete.
}

// ELB demand-join re-assertion: raising a staged transfer's priority reorders
// the staged admission queue (a demand fetch joining a low-priority prefetch
// must dispatch before other staged prefetches).
TEST(TransferEngine, Staging_ReprioritizeStagedItem) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.min_dispatch_per_gpu = 0;   // everything stages
    dc.opts.max_inflight_per_gpu = 1;   // flush admits exactly one

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    // A staged at prefetch priority below B.
    auto ta = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, -1.0f);
    auto tb = engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.0f);
    ASSERT_TRUE(ta.has_value());
    ASSERT_TRUE(tb.has_value());
    EXPECT_EQ(engine.staged_count(0), 2);

    // Demand joins A: raise to demand priority. B keeps 0.0.
    EXPECT_TRUE(engine.reprioritize(*ta,
                                    std::numeric_limits<float>::max()));
    // Lowering (or equal) never applies.
    EXPECT_FALSE(engine.reprioritize(*tb, -2.0f));

    // Flush admits exactly one — it must now be A, not B.
    engine.flush_staged();
    EXPECT_EQ(engine.staged_count(0), 1);
    EXPECT_TRUE(engine.is_dispatched_h2d(ek(0, 0), 0));
    EXPECT_FALSE(engine.is_dispatched_h2d(ek(0, 1), 0));

    // A dispatched transfer can no longer be reprioritized.
    EXPECT_FALSE(engine.reprioritize(*ta, std::numeric_limits<float>::max()));
    // Unknown token.
    EXPECT_FALSE(engine.reprioritize(999999, 1.0f));

    event_ready = true;  // Allow drain() in destructor to complete.
}

TEST(TransferEngine, Staging_CancelStagedItem) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.min_dispatch_per_gpu = 1;
    dc.opts.max_inflight_per_gpu = 4;

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    // First dispatched immediately.
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 1.0f);

    // Second goes to staging.
    auto staged_token = engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.5f);
    ASSERT_TRUE(staged_token.has_value());
    EXPECT_EQ(engine.staged_count(0), 1);

    // Cancel the staged item — O(1), no orphaned DMA.
    auto result = engine.cancel(*staged_token);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->key, ek(0, 1));
    EXPECT_EQ(engine.staged_count(0), 0);

    event_ready = true;  // Allow drain() in destructor to complete.
}

TEST(TransferEngine, Staging_DelayBypassesWatermark) {
    auto tc = make_staged_context(1, 2, 8);
    lt::TransferEngine engine(tc.opts);
    std::vector<uint8_t> buf(64);

    // Even though inflight < min_dispatch, delay_us > 0 → staged.
    auto t = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64,
                                0.5f, 1000000);
    EXPECT_TRUE(t.has_value());
    EXPECT_EQ(engine.staged_count(0), 1);

    // Flush should NOT dispatch it (delay hasn't elapsed).
    engine.flush_staged();
    EXPECT_EQ(engine.staged_count(0), 1);
}

TEST(TransferEngine, Staging_DedupAcrossStaged) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.min_dispatch_per_gpu = 1;
    dc.opts.max_inflight_per_gpu = 4;

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    // First dispatched immediately.
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 1.0f);

    // Second staged.
    auto t1 = engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.5f);
    EXPECT_TRUE(t1.has_value());

    // Same key again → dedup even though it's staged.
    auto t2 = engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.9f);
    EXPECT_FALSE(t2.has_value());

    event_ready = true;  // Allow drain() in destructor to complete.
}

TEST(TransferEngine, Staging_DrainFlushesAll) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.min_dispatch_per_gpu = 1;
    dc.opts.max_inflight_per_gpu = 4;

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 1.0f);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.5f);
    EXPECT_EQ(engine.staged_count(0), 1);

    event_ready = true;
    engine.drain();
    EXPECT_EQ(engine.inflight_count(), 0);
    EXPECT_EQ(engine.staged_count(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Priority bin counts (#43)
// ═══════════════════════════════════════════════════════════════════════════

TEST(TransferEngine, test_bin_counts_empty) {
    auto tc = make_test_context();
    tc.opts.priority_bin_thresholds = {0.3f, 0.7f};
    lt::TransferEngine engine(std::move(tc.opts));

    auto counts = engine.pending_bin_counts(0);
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_EQ(counts[0], 0);
    EXPECT_EQ(counts[1], 0);
    EXPECT_EQ(counts[2], 0);
}

TEST(TransferEngine, test_bin_counts_staged) {
    auto tc = make_test_context();
    tc.opts.priority_bin_thresholds = {0.3f, 0.7f};
    lt::TransferEngine engine(std::move(tc.opts));

    std::vector<uint8_t> buf(64);

    // Force staging via delay_us > 0.
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 0.1f, 1000);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.5f, 1000);
    engine.enqueue_h2d(ek(0, 2), 0, buf.data(), buf.data(), 64, 0.9f, 1000);

    auto counts = engine.pending_bin_counts(0);
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_EQ(counts[0], 1);  // 0.1 < 0.3
    EXPECT_EQ(counts[1], 1);  // 0.3 <= 0.5 < 0.7
    EXPECT_EQ(counts[2], 1);  // 0.9 >= 0.7

    // GPU 1 should be empty.
    auto counts1 = engine.pending_bin_counts(1);
    ASSERT_EQ(counts1.size(), 3u);
    EXPECT_EQ(counts1[0], 0);
    EXPECT_EQ(counts1[1], 0);
    EXPECT_EQ(counts1[2], 0);
}

TEST(TransferEngine, test_bin_counts_inflight) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.priority_bin_thresholds = {0.3f, 0.7f};

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 0.2f);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.8f);

    auto counts = engine.pending_bin_counts(0);
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_EQ(counts[0], 1);  // 0.2 < 0.3
    EXPECT_EQ(counts[1], 0);
    EXPECT_EQ(counts[2], 1);  // 0.8 >= 0.7

    event_ready = true;
    engine.drain();
}

TEST(TransferEngine, test_bin_counts_after_completion) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.priority_bin_thresholds = {0.3f, 0.7f};

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 0.5f);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.9f);

    auto before = engine.pending_bin_counts(0);
    EXPECT_EQ(before[1], 1);
    EXPECT_EQ(before[2], 1);

    event_ready = true;
    engine.poll_completions();

    auto after = engine.pending_bin_counts(0);
    ASSERT_EQ(after.size(), 3u);
    EXPECT_EQ(after[0], 0);
    EXPECT_EQ(after[1], 0);
    EXPECT_EQ(after[2], 0);
}

TEST(TransferEngine, test_bin_counts_mixed) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.min_dispatch_per_gpu = 2;
    dc.opts.max_inflight_per_gpu = 8;
    dc.opts.priority_bin_thresholds = {0.3f, 0.7f};

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    // First 2 go to immediate dispatch (below watermark).
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 0.1f);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.8f);
    // Third goes to staging (at watermark).
    engine.enqueue_h2d(ek(0, 2), 0, buf.data(), buf.data(), 64, 0.5f);

    EXPECT_EQ(engine.staged_count(0), 1);

    auto counts = engine.pending_bin_counts(0);
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_EQ(counts[0], 1);  // 0.1 dispatched
    EXPECT_EQ(counts[1], 1);  // 0.5 staged
    EXPECT_EQ(counts[2], 1);  // 0.8 dispatched

    event_ready = true;
    engine.drain();
}

TEST(TransferEngine, test_bin_counts_custom_thresholds) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.priority_bin_thresholds = {0.25f, 0.5f, 0.75f};

    lt::TransferEngine engine(std::move(dc.opts));
    std::vector<uint8_t> buf(64);

    // Boundary values: exact threshold values land in the higher bin
    // (upper_bound: bin 0 = p < 0.25, bin 1 = [0.25, 0.5), etc.)
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 0.25f);  // bin 1
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64, 0.50f);  // bin 2
    engine.enqueue_h2d(ek(0, 2), 0, buf.data(), buf.data(), 64, 0.75f);  // bin 3
    engine.enqueue_h2d(ek(0, 3), 0, buf.data(), buf.data(), 64, 0.10f);  // bin 0

    auto counts = engine.pending_bin_counts(0);
    ASSERT_EQ(counts.size(), 4u);
    EXPECT_EQ(counts[0], 1);  // 0.10
    EXPECT_EQ(counts[1], 1);  // 0.25
    EXPECT_EQ(counts[2], 1);  // 0.50
    EXPECT_EQ(counts[3], 1);  // 0.75

    event_ready = true;
    engine.drain();
}

TEST(TransferEngine, PollCompletions_GpuFatalError_SetsSuccessFalse) {
    lc::EventStatus event_status = lc::EventStatus::kNotReady;
    auto gpus = make_gpu_refs(1);

    std::vector<std::unique_ptr<lc::DeviceBackend>> backends;
    backends.push_back(std::make_unique<StatusDeviceBackend>(gpus[0], event_status));
    std::vector<lc::DeviceBackend*> ptrs{backends[0].get()};

    lt::TransferEngine::Options opts;
    opts.device_backends = ptrs;
    lt::TransferEngine engine(opts);

    // Enqueue a transfer.
    std::vector<uint8_t> buf(64);
    bool callback_fired = false;
    bool callback_success = true;
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64, 0.5f, 0,
        [&](lt::TransferToken, bool success) {
            callback_fired = true;
            callback_success = success;
        });
    EXPECT_EQ(engine.inflight_count(), 1);

    // Poll with kNotReady — nothing completes.
    auto completions = engine.poll_completions();
    EXPECT_TRUE(completions.empty());
    EXPECT_EQ(engine.inflight_count(), 1);

    // Simulate GPU fatal error.
    event_status = lc::EventStatus::kError;
    completions = engine.poll_completions();
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_FALSE(completions[0].success);
    EXPECT_TRUE(callback_fired);
    EXPECT_FALSE(callback_success);
    EXPECT_EQ(engine.inflight_count(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// WP-7b: Pinned staging pool
// ═══════════════════════════════════════════════════════════════════════════

TEST(TransferEngine, PinnedPool_InitAndQuery) {
    auto tc = make_test_context(1);
    lt::TransferEngine engine(tc.opts);

    EXPECT_EQ(engine.pinned_pool_size(), 0);
    EXPECT_EQ(engine.pinned_pool_free(), 0);

    engine.init_pinned_pool(1024, 4, tc.backends[0].get());

    EXPECT_EQ(engine.pinned_pool_size(), 4);
    EXPECT_EQ(engine.pinned_pool_free(), 4);
}

TEST(TransferEngine, PinnedPool_InitIdempotent) {
    auto tc = make_test_context(1);
    lt::TransferEngine engine(tc.opts);

    engine.init_pinned_pool(1024, 4, tc.backends[0].get());
    engine.init_pinned_pool(2048, 8, tc.backends[0].get());  // Should be no-op.

    EXPECT_EQ(engine.pinned_pool_size(), 4);
}

TEST(TransferEngine, PinnedPool_StagingCopiesData) {
    auto tc = make_test_context(1);
    lt::TransferEngine engine(tc.opts);
    engine.init_pinned_pool(1024, 2, tc.backends[0].get());

    std::vector<uint8_t> src(1024, 0xAB);
    std::vector<uint8_t> dst(1024, 0);

    engine.enqueue_h2d(ek(0, 0), 0, dst.data(), src.data(), 1024,
                       0.0f, 0, {}, /*needs_pinned_staging=*/true);
    auto completions = engine.poll_completions();

    ASSERT_EQ(completions.size(), 1u);
    EXPECT_TRUE(completions[0].success);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 1024), 0);
}

TEST(TransferEngine, PinnedPool_SlotReleasedAfterCompletion) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    lt::TransferEngine engine(std::move(dc.opts));
    engine.init_pinned_pool(64, 2, dc.backends[0].get());

    std::vector<uint8_t> buf(64);
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64,
                       0.0f, 0, {}, true);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64,
                       0.0f, 0, {}, true);

    // Both slots in use.
    EXPECT_EQ(engine.pinned_pool_free(), 0);

    event_ready = true;
    engine.poll_completions();

    // Both released.
    EXPECT_EQ(engine.pinned_pool_free(), 2);
}

TEST(TransferEngine, PinnedPool_ExhaustionFallsBack) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    lt::TransferEngine engine(std::move(dc.opts));
    engine.init_pinned_pool(64, 1, dc.backends[0].get());

    std::vector<uint8_t> src(64, 0xCD);
    std::vector<uint8_t> dst1(64, 0);
    std::vector<uint8_t> dst2(64, 0);

    // Takes the only slot.
    auto t1 = engine.enqueue_h2d(ek(0, 0), 0, dst1.data(), src.data(), 64,
                                  0.0f, 0, {}, true);
    EXPECT_TRUE(t1.has_value());
    EXPECT_EQ(engine.pinned_pool_free(), 0);

    // Pool exhausted — still enqueues successfully (unpinned fallback).
    auto t2 = engine.enqueue_h2d(ek(0, 1), 0, dst2.data(), src.data(), 64,
                                  0.0f, 0, {}, true);
    EXPECT_TRUE(t2.has_value());

    event_ready = true;
    auto completions = engine.poll_completions();
    EXPECT_EQ(completions.size(), 2u);

    // Both succeed — data correct.
    EXPECT_EQ(std::memcmp(dst1.data(), src.data(), 64), 0);
    EXPECT_EQ(std::memcmp(dst2.data(), src.data(), 64), 0);
    EXPECT_EQ(engine.pinned_pool_free(), 1);
}

TEST(TransferEngine, PinnedPool_StagedDispatchAcquiresSlot) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.min_dispatch_per_gpu = 0;  // Force all to staging queue.
    dc.opts.max_inflight_per_gpu = 4;
    lt::TransferEngine engine(std::move(dc.opts));
    engine.init_pinned_pool(64, 2, dc.backends[0].get());

    std::vector<uint8_t> buf(64);
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64,
                       0.5f, 0, {}, true);

    // Slot NOT yet acquired (still in staging queue).
    EXPECT_EQ(engine.pinned_pool_free(), 2);
    EXPECT_EQ(engine.staged_count(0), 1);

    // Flush promotes from staging → dispatched. Acquires slot.
    engine.flush_staged();
    EXPECT_EQ(engine.staged_count(0), 0);
    EXPECT_EQ(engine.pinned_pool_free(), 1);

    // Complete releases slot.
    event_ready = true;
    engine.poll_completions();
    EXPECT_EQ(engine.pinned_pool_free(), 2);
}

TEST(TransferEngine, PinnedPool_CancelReleasesSlotOnPoll) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    lt::TransferEngine engine(std::move(dc.opts));
    engine.init_pinned_pool(64, 2, dc.backends[0].get());

    std::vector<uint8_t> buf(64);
    auto token = engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64,
                                     0.0f, 0, {}, true);
    ASSERT_TRUE(token.has_value());
    EXPECT_EQ(engine.pinned_pool_free(), 1);

    auto result = engine.cancel(*token);
    ASSERT_TRUE(result.has_value());

    // TD-82k/TD-98c: slot stays in-use until DMA finishes (deferred cleanup).
    EXPECT_EQ(engine.pinned_pool_free(), 1);

    // DMA completes → poll_completions drains the cancelled entry.
    event_ready = true;
    auto completions = engine.poll_completions();
    EXPECT_TRUE(completions.empty());  // Cancelled — no completion emitted.
    EXPECT_EQ(engine.pinned_pool_free(), 2);  // Slot released.
}

TEST(TransferEngine, PinnedPool_CancelStagedNoLeak) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    dc.opts.min_dispatch_per_gpu = 1;
    dc.opts.max_inflight_per_gpu = 4;
    lt::TransferEngine engine(std::move(dc.opts));
    engine.init_pinned_pool(64, 2, dc.backends[0].get());

    std::vector<uint8_t> buf(64);

    // First immediate (takes a slot).
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64,
                       1.0f, 0, {}, true);

    // Second goes to staging (no slot acquired yet).
    auto staged_token = engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64,
                                            0.5f, 0, {}, true);
    ASSERT_TRUE(staged_token.has_value());
    EXPECT_EQ(engine.staged_count(0), 1);
    EXPECT_EQ(engine.pinned_pool_free(), 1);  // Only one slot used (immediate).

    // Cancel the staged one — no slot to release.
    auto result = engine.cancel(*staged_token);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(engine.pinned_pool_free(), 1);  // Unchanged.

    event_ready = true;
}

TEST(TransferEngine, PinnedPool_DrainReleasesAll) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    lt::TransferEngine engine(std::move(dc.opts));
    engine.init_pinned_pool(64, 4, dc.backends[0].get());

    std::vector<uint8_t> buf(64);
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64,
                       0.0f, 0, {}, true);
    engine.enqueue_h2d(ek(0, 1), 0, buf.data(), buf.data(), 64,
                       0.0f, 0, {}, true);
    EXPECT_EQ(engine.pinned_pool_free(), 2);

    event_ready = true;
    engine.drain();
    EXPECT_EQ(engine.inflight_count(), 0);
    EXPECT_EQ(engine.pinned_pool_free(), 4);
}

TEST(TransferEngine, PinnedPool_NoPoolStillWorks) {
    auto tc = make_test_context(1);
    lt::TransferEngine engine(tc.opts);

    // No pool initialized — needs_pinned_staging=true should still work (unpinned fallback).
    std::vector<uint8_t> src(64, 0xEF);
    std::vector<uint8_t> dst(64, 0);

    auto token = engine.enqueue_h2d(ek(0, 0), 0, dst.data(), src.data(), 64,
                                     0.0f, 0, {}, true);
    EXPECT_TRUE(token.has_value());

    auto completions = engine.poll_completions();
    ASSERT_EQ(completions.size(), 1u);
    EXPECT_TRUE(completions[0].success);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 64), 0);
}

TEST(TransferEngine, PinnedPool_FalseSkipsPool) {
    bool event_ready = false;
    auto dc = make_deferred_context(event_ready, 1);
    lt::TransferEngine engine(std::move(dc.opts));
    engine.init_pinned_pool(64, 2, dc.backends[0].get());

    std::vector<uint8_t> buf(64);
    engine.enqueue_h2d(ek(0, 0), 0, buf.data(), buf.data(), 64,
                       0.0f, 0, {}, /*needs_pinned_staging=*/false);

    // Pool should be untouched.
    EXPECT_EQ(engine.pinned_pool_free(), 2);

    event_ready = true;
    engine.poll_completions();
    EXPECT_EQ(engine.pinned_pool_free(), 2);
}

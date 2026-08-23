// Unit tests for DaemonLoop (IPC-3).
//
// All tests run without CUDA by using null backends and heap-allocated
// IPC regions, matching the pattern from spsc_ring_test.cpp.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "core/gpu_ref.h"
#include "core/null_device_backend.h"
#include "core/transfer/transfer_engine.h"
#include "daemon/daemon_loop.h"
#include "daemon/ipc_protocol.h"
#include "daemon/spsc_ring.h"

namespace lipc  = layerstorm::ipc;
namespace ltr   = layerstorm::transfer;
namespace ldam  = layerstorm::daemon;
namespace lcomp = layerstorm::compute;

// ── Helpers ────────────────────────────────────────────────────────────────

static constexpr uint32_t kTestSlots = 128;  // Small ring for tests

/// Allocate a zeroed, 64-byte-aligned buffer.
static void* aligned_alloc_zeroed(size_t bytes) {
    void* p = std::aligned_alloc(64, bytes);
    std::memset(p, 0, bytes);
    return p;
}

/// Write a command with the given type and sequence number into the ring.
static bool write_cmd(lipc::CommandRing& ring, lipc::CmdType type,
                      uint32_t seq = 0, uint32_t gpu_idx = 0) {
    lipc::Command cmd{};
    cmd.cmd_type  = static_cast<uint32_t>(type);
    cmd.cmd_seq   = seq;
    cmd.gpu_idx   = gpu_idx;
    cmd.stream_id = 0;
    return ring.try_write(&cmd);
}

/// Read one completion from the ring.  Returns false if empty.
static bool read_cmp(lipc::CompletionRing& ring, lipc::Completion& out) {
    return ring.try_read(&out);
}

// ── Test fixture ───────────────────────────────────────────────────────────

class DaemonLoopTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Allocate full IPC region: header + cmd ring + cmp ring + snapshot
        ipc_bytes_ = lipc::IpcLayout::total_size(kTestSlots, kTestSlots);
        ipc_region_ = static_cast<uint8_t*>(aligned_alloc_zeroed(ipc_bytes_));

        // Init sub-regions
        header_ = reinterpret_cast<lipc::IpcHeader*>(
            ipc_region_ + lipc::IpcLayout::kHeaderOffset);
        header_->version = lipc::kProtocolVersion;
        header_->shutdown_requested = 0;
        header_->error_code = 0;

        void* cmd_ptr = ipc_region_ + lipc::IpcLayout::cmd_ring_offset();
        lipc::CommandRing::init(cmd_ptr, kTestSlots);
        cmd_ring_ = std::make_unique<lipc::CommandRing>(cmd_ptr);

        void* cmp_ptr = ipc_region_ + lipc::IpcLayout::cmp_ring_offset(kTestSlots);
        lipc::CompletionRing::init(cmp_ptr, kTestSlots);
        cmp_ring_ = std::make_unique<lipc::CompletionRing>(cmp_ptr);

        snap_ = reinterpret_cast<lipc::StateSnapshot*>(
            ipc_region_ + lipc::IpcLayout::state_offset(kTestSlots, kTestSlots));
        snap_->seqlock = 0;

        running_.store(true, std::memory_order_relaxed);
    }

    void TearDown() override {
        std::free(ipc_region_);
    }

    /// Build Deps with optional overrides.
    ldam::DaemonLoop::Deps make_deps(
        ltr::TransferEngine* te = nullptr,
        ldam::DaemonLoop::CommandDispatchFn dispatch = {},
        ldam::DaemonLoop::StatePublishFn publish = {})
    {
        return {
            .cmd_ring        = cmd_ring_.get(),
            .cmp_ring        = cmp_ring_.get(),
            .ipc_header      = header_,
            .state_snapshot  = snap_,
            .running         = &running_,
            .transfer_engine = te,
            .dispatch_fn     = std::move(dispatch),
            .publish_fn      = std::move(publish),
        };
    }

    uint8_t*            ipc_region_ = nullptr;
    size_t              ipc_bytes_  = 0;
    lipc::IpcHeader*    header_     = nullptr;
    lipc::StateSnapshot* snap_      = nullptr;
    std::unique_ptr<lipc::CommandRing>    cmd_ring_;
    std::unique_ptr<lipc::CompletionRing> cmp_ring_;
    std::atomic<bool>   running_{false};
};

// ── Tests ──────────────────────────────────────────────────────────────────

TEST_F(DaemonLoopTest, Construction) {
    ldam::DaemonLoop loop(make_deps());
    EXPECT_EQ(loop.cycle_count(), 0u);
}

TEST_F(DaemonLoopTest, EmptyRing_CycleAdvances) {
    ldam::DaemonLoop loop(make_deps());

    EXPECT_TRUE(loop.run_one_cycle());
    EXPECT_EQ(loop.cycle_count(), 1u);
    EXPECT_EQ(snap_->daemon_cycle_count, 1u);
    EXPECT_GT(snap_->timestamp_ns, 0u);
    // Seqlock must be even (readable) after publish
    EXPECT_EQ(snap_->seqlock & 1, 0u);
}

TEST_F(DaemonLoopTest, NoopCommand) {
    ldam::DaemonLoop loop(make_deps());

    write_cmd(*cmd_ring_, lipc::CMD_NOOP, /*seq=*/1);
    EXPECT_TRUE(loop.run_one_cycle());

    // Ring should be drained
    EXPECT_TRUE(cmd_ring_->is_empty());
    // No completion for NOOP
    lipc::Completion cmp{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));
}

TEST_F(DaemonLoopTest, ShutdownCommand) {
    ldam::DaemonLoop loop(make_deps());

    write_cmd(*cmd_ring_, lipc::CMD_SHUTDOWN, /*seq=*/1);
    EXPECT_FALSE(loop.run_one_cycle());
    EXPECT_FALSE(running_.load());
}

TEST_F(DaemonLoopTest, ShutdownViaHeader) {
    ldam::DaemonLoop loop(make_deps());

    header_->shutdown_requested = 1;
    EXPECT_FALSE(loop.run_one_cycle());
    EXPECT_FALSE(running_.load());
}

TEST_F(DaemonLoopTest, UnhandledCommand_NoDispatcher) {
    // With no dispatch_fn, unhandled commands should be drained without crash.
    ldam::DaemonLoop loop(make_deps());

    write_cmd(*cmd_ring_, lipc::CMD_TRANSFER_H2D, /*seq=*/42);
    EXPECT_TRUE(loop.run_one_cycle());
    EXPECT_TRUE(cmd_ring_->is_empty());
}

TEST_F(DaemonLoopTest, CommandForwarding) {
    std::vector<lipc::Command> captured;
    auto dispatch = [&captured](const lipc::Command& cmd) {
        captured.push_back(cmd);
    };

    ldam::DaemonLoop loop(make_deps(nullptr, dispatch));

    write_cmd(*cmd_ring_, lipc::CMD_TRANSFER_H2D, /*seq=*/10, /*gpu_idx=*/2);
    write_cmd(*cmd_ring_, lipc::CMD_CACHE_EVICT, /*seq=*/11, /*gpu_idx=*/0);

    EXPECT_TRUE(loop.run_one_cycle());

    ASSERT_EQ(captured.size(), 2u);
    EXPECT_EQ(captured[0].cmd_type, static_cast<uint32_t>(lipc::CMD_TRANSFER_H2D));
    EXPECT_EQ(captured[0].cmd_seq, 10u);
    EXPECT_EQ(captured[0].gpu_idx, 2u);
    EXPECT_EQ(captured[1].cmd_type, static_cast<uint32_t>(lipc::CMD_CACHE_EVICT));
    EXPECT_EQ(captured[1].cmd_seq, 11u);
}

TEST_F(DaemonLoopTest, TransferCompletion_WritesToRing) {
    // Create a TransferEngine with null device backend (completes immediately on poll)
    layerstorm::config::GpuRef gpu0{0, 0, layerstorm::config::GpuType::rtx5090};
    auto null_backend0 = lcomp::make_null_device_backend(gpu0);
    std::vector<lcomp::DeviceBackend*> backends{null_backend0.get()};
    ltr::TransferEngine::Options opts{
        .device_backends = backends,
        .pcie_info = {{.pcie_gen = 5, .pcie_width = 16}},
    };
    ltr::TransferEngine engine(std::move(opts));

    // Enqueue a transfer — null backend records an event that immediately
    // queries as complete on the next poll_completions() call.
    layerstorm::memory::ExpertKey key{.layer_idx = 3, .expert_idx = 7};
    uint8_t dummy_src[64]{};
    uint8_t dummy_dst[64]{};
    auto token = engine.enqueue_h2d(key, 0, dummy_dst, dummy_src, 64);
    ASSERT_TRUE(token.has_value());

    // Now wire the DaemonLoop with this engine
    ldam::DaemonLoop loop(make_deps(&engine));
    EXPECT_TRUE(loop.run_one_cycle());

    // Should have one transfer completion on the completion ring
    lipc::Completion cmp{};
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_TRANSFER_DONE));
    EXPECT_EQ(cmp.gpu_idx, 0u);
    EXPECT_EQ(cmp.status, 0u);  // success
    EXPECT_EQ(cmp.transfer.layer_idx, 3u);
    EXPECT_EQ(cmp.transfer.expert_idx, 7u);
    EXPECT_EQ(cmp.transfer.direction, 0u);  // H2D = 0
}

/// Backend with controllable EventStatus for error-path tests.
class StatusDeviceBackend : public lcomp::NullDeviceBackend {
public:
    StatusDeviceBackend(layerstorm::config::GpuRef gpu, lcomp::EventStatus& status)
        : NullDeviceBackend(std::move(gpu)), status_(status) {}
    lcomp::EventQueryResult query_event(void*) override { return {status_, 42}; }
private:
    lcomp::EventStatus& status_;
};

TEST_F(DaemonLoopTest, TransferFailure_EmitsCmpGpuFatal) {
    // Verify CMP_GPU_FATAL is emitted when a transfer reports kError.
    lcomp::EventStatus event_status = lcomp::EventStatus::kNotReady;
    layerstorm::config::GpuRef gpu0{0, 0, layerstorm::config::GpuType::rtx5090};

    auto backend = std::make_unique<StatusDeviceBackend>(gpu0, event_status);
    std::vector<lcomp::DeviceBackend*> backends{backend.get()};
    ltr::TransferEngine::Options opts{
        .device_backends = backends,
        .pcie_info = {{.pcie_gen = 5, .pcie_width = 16}},
    };
    ltr::TransferEngine engine(std::move(opts));

    // Enqueue a transfer.
    layerstorm::memory::ExpertKey key{.layer_idx = 5, .expert_idx = 10};
    uint8_t dummy[64]{};
    auto token = engine.enqueue_h2d(key, 0, dummy, dummy, 64);
    ASSERT_TRUE(token.has_value());

    // Wire DaemonLoop.
    ldam::DaemonLoop loop(make_deps(&engine));

    // First cycle: kNotReady — no completions.
    EXPECT_TRUE(loop.run_one_cycle());
    lipc::Completion cmp{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));

    // Simulate GPU fatal.
    event_status = lcomp::EventStatus::kError;
    EXPECT_TRUE(loop.run_one_cycle());

    // First completion: CMP_GPU_FATAL.
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_GPU_FATAL));
    EXPECT_EQ(cmp.gpu_idx, 0u);

    // Second completion: CMP_TRANSFER_DONE with status=1 (failure).
    ASSERT_TRUE(read_cmp(*cmp_ring_, cmp));
    EXPECT_EQ(cmp.cmp_type, static_cast<uint32_t>(lipc::CMP_TRANSFER_DONE));
    EXPECT_EQ(cmp.status, 1u);

    // No more completions.
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));
}

TEST_F(DaemonLoopTest, StatePublishFn_Called) {
    bool called = false;
    auto publish = [&called](lipc::StateSnapshot& snap,
                             ldam::StateTransaction&) {
        called = true;
        snap.shift_detected = 1;
    };

    ldam::DaemonLoop loop(make_deps(nullptr, {}, publish));
    EXPECT_TRUE(loop.run_one_cycle());

    EXPECT_TRUE(called);
    EXPECT_EQ(snap_->shift_detected, 1u);
    // Seqlock must be even (write complete)
    EXPECT_EQ(snap_->seqlock & 1, 0u);
}

TEST_F(DaemonLoopTest, BatchDrainLimit) {
    ldam::DaemonLoop loop(make_deps());

    // Write 100 NOOPs
    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(write_cmd(*cmd_ring_, lipc::CMD_NOOP, static_cast<uint32_t>(i)));
    }
    EXPECT_EQ(cmd_ring_->available(), 100u);

    // First cycle drains at most kMaxCommandsPerCycle (64)
    EXPECT_TRUE(loop.run_one_cycle());
    EXPECT_EQ(cmd_ring_->available(), 36u);

    // Second cycle drains the rest
    EXPECT_TRUE(loop.run_one_cycle());
    EXPECT_EQ(cmd_ring_->available(), 0u);
}

TEST_F(DaemonLoopTest, RunMultipleCycles) {
    ldam::DaemonLoop loop(make_deps());

    // Run on a background thread
    std::thread t([&loop] { loop.run(); });

    // Write a few NOOPs, then shutdown
    for (int i = 0; i < 5; ++i) {
        write_cmd(*cmd_ring_, lipc::CMD_NOOP);
    }

    // Spin until some cycles have run
    while (loop.cycle_count() < 3) {
        std::this_thread::yield();
    }
    EXPECT_GE(loop.cycle_count(), 3u);

    // Shutdown
    write_cmd(*cmd_ring_, lipc::CMD_SHUTDOWN);
    t.join();

    EXPECT_FALSE(running_.load());
    EXPECT_GE(loop.cycle_count(), 3u);
}

TEST_F(DaemonLoopTest, CycleCountMonotonic) {
    ldam::DaemonLoop loop(make_deps());

    for (int i = 1; i <= 10; ++i) {
        EXPECT_TRUE(loop.run_one_cycle());
        EXPECT_EQ(loop.cycle_count(), static_cast<uint64_t>(i));
        EXPECT_EQ(snap_->daemon_cycle_count, static_cast<uint64_t>(i));
    }
}

// #91 / INV-IPC-PUBLISH-THROTTLE: with a publish interval armed, only the
// FIRST cycle publishes (subsequent µs-apart cycles skip the bulk publish
// so the seqlock stays quiet for Python readers); with the default 0 the
// legacy per-cycle publish holds (CycleCountMonotonic above).
TEST_F(DaemonLoopTest, PublishThrottle_SkipsWithinInterval) {
    auto deps = make_deps();
    deps.publish_interval_ns = 60ULL * 1000 * 1000 * 1000;  // 60 s: never re-publish
    ldam::DaemonLoop loop(std::move(deps));

    for (int i = 1; i <= 5; ++i) {
        EXPECT_TRUE(loop.run_one_cycle());
        // Internal counter always advances ...
        EXPECT_EQ(loop.cycle_count(), static_cast<uint64_t>(i));
        // ... but the snapshot was only written by the first publish.
        EXPECT_EQ(snap_->daemon_cycle_count, 1u);
    }
    // Seqlock must be even (readable) — the skip never leaves it odd.
    EXPECT_EQ(snap_->seqlock & 1, 0u);
}

// ── NVMe completion polling (IPC-8a.1) ─────────────────────────────────────

TEST_F(DaemonLoopTest, PollNvmeCompletions_NullTier) {
    // nvme_tier = nullptr (default) → poll_nvme_completions is a no-op.
    auto deps = make_deps();
    ldam::DaemonLoop loop(std::move(deps));

    // Should not crash — null NvmeTier is safely skipped.
    EXPECT_TRUE(loop.run_one_cycle());
    EXPECT_EQ(loop.cycle_count(), 1u);

    // No NVMe completions should appear on the ring.
    lipc::Completion cmp{};
    EXPECT_FALSE(read_cmp(*cmp_ring_, cmp));
}

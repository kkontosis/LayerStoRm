#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "daemon/ipc_protocol.h"
#include "daemon/spsc_ring.h"

namespace lipc = layerstorm::ipc;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Allocate cache-line-aligned buffer.
static std::vector<uint8_t> alloc_aligned(size_t bytes) {
    // Over-allocate and align. Use vector for automatic cleanup.
    std::vector<uint8_t> buf(bytes + 64, 0);
    return buf;
}

/// Return a 64-byte-aligned pointer within the buffer.
static void* align_ptr(std::vector<uint8_t>& buf) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(buf.data());
    uintptr_t aligned = (addr + 63) & ~uintptr_t(63);
    return reinterpret_cast<void*>(aligned);
}

/// Total bytes for a ring region: header + slots.
static size_t ring_region_size(uint32_t slot_count, uint32_t slot_size) {
    return sizeof(lipc::RingHeader) + static_cast<size_t>(slot_count) * slot_size;
}

// ── Struct size tests ───────────────────────────────────────────────────────

TEST(IpcProtocol, StructSizes) {
    EXPECT_EQ(sizeof(lipc::IpcHeader), 256u);
    EXPECT_EQ(sizeof(lipc::RingHeader), 128u);
    EXPECT_EQ(sizeof(lipc::Command), 256u);
    EXPECT_EQ(sizeof(lipc::Completion), 128u);
    EXPECT_EQ(sizeof(lipc::GpuSnapshot), 56u);
    EXPECT_EQ(sizeof(lipc::RequestAcceptance), 16u);
    // Cross-language invariant: Python ctypes must match this exact value.
    EXPECT_EQ(sizeof(lipc::StateSnapshot), 1676928u);  // TD-IPC-MOE-LAYER-CAP: kMaxMoeLayers 64→128 (still a multiple of 64)
}

TEST(IpcProtocol, FieldOffsets) {
    // IpcHeader: shutdown_requested at offset 64, error_code at 128
    EXPECT_EQ(offsetof(lipc::IpcHeader, shutdown_requested), 64u);
    EXPECT_EQ(offsetof(lipc::IpcHeader, error_code), 128u);
    EXPECT_EQ(offsetof(lipc::IpcHeader, error_message), 132u);

    // RingHeader: producer_seq at 0, consumer_seq at 64, slot_count at 120
    EXPECT_EQ(offsetof(lipc::RingHeader, producer_seq), 0u);
    EXPECT_EQ(offsetof(lipc::RingHeader, consumer_seq), 64u);
    EXPECT_EQ(offsetof(lipc::RingHeader, slot_count), 120u);
    EXPECT_EQ(offsetof(lipc::RingHeader, slot_size), 124u);

    // Command payload at offset 16
    EXPECT_EQ(offsetof(lipc::Command, raw), 16u);

    // Completion payload at offset 16
    EXPECT_EQ(offsetof(lipc::Completion, raw), 16u);
}

TEST(IpcProtocol, EngineInfoOffsets) {
    // Cross-language invariant: Python ctypes EngineInfo must match these offsets.
    EXPECT_EQ(sizeof(lipc::EngineInfo), 128u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, ipc_base), 0u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, ipc_total_bytes), 8u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, cmd_ring_offset), 16u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, cmd_ring_slots), 24u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, cmd_slot_bytes), 28u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, cmp_ring_offset), 32u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, cmp_ring_slots), 40u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, cmp_slot_bytes), 44u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, state_offset), 48u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, state_bytes), 56u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, sideband_offset), 64u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, sideband_bytes), 72u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, num_gpus), 80u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, num_moe_layers), 84u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, num_experts), 88u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, expert_bytes), 96u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, num_layers), 104u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, num_expert_devices), 108u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, kv_bytes_per_page), 112u);
    EXPECT_EQ(offsetof(lipc::EngineInfo, vocab_size), 120u);
}

TEST(IpcProtocol, IpcLayoutDefaults) {
    constexpr auto cmd_slots = lipc::kDefaultCmdRingSlots;
    constexpr auto cmp_slots = lipc::kDefaultCmpRingSlots;

    EXPECT_EQ(lipc::IpcLayout::cmd_ring_offset(), 256u);
    EXPECT_EQ(lipc::IpcLayout::cmd_ring_size(cmd_slots),
              128u + 8192u * 256u);
    EXPECT_EQ(lipc::IpcLayout::cmp_ring_offset(cmd_slots),
              256u + 128u + 8192u * 256u);

    auto total = lipc::IpcLayout::total_size(cmd_slots, cmp_slots);
    EXPECT_GT(total, 3u * 1024u * 1024u);  // > 3 MB
}

// ── SPSC ring tests ─────────────────────────────────────────────────────────

class SpscRingTest : public ::testing::Test {
protected:
    static constexpr uint32_t kSlots = 4;
    static constexpr uint32_t kSlotSize = lipc::kCmdSlotBytes;

    void SetUp() override {
        buf_.resize(ring_region_size(kSlots, kSlotSize) + 64, 0);
        region_ = align_ptr(buf_);
        lipc::SpscRing<kSlotSize>::init(region_, kSlots);
        ring_ = std::make_unique<lipc::SpscRing<kSlotSize>>(region_);
    }

    std::vector<uint8_t> buf_;
    void* region_ = nullptr;
    std::unique_ptr<lipc::SpscRing<kSlotSize>> ring_;
};

TEST_F(SpscRingTest, SingleWriteRead) {
    lipc::Command cmd{};
    cmd.cmd_type = lipc::CMD_TRANSFER_H2D;
    cmd.cmd_seq = 42;
    cmd.gpu_idx = 1;
    cmd.stream_id = 3;
    cmd.transfer.layer_idx = 10;
    cmd.transfer.expert_idx = 200;
    cmd.transfer.sub_component = 0x07;
    cmd.transfer.zone = 1;
    cmd.transfer.bytes = 1024 * 1024;

    ASSERT_TRUE(ring_->try_write(&cmd));

    lipc::Command out{};
    ASSERT_TRUE(ring_->try_read(&out));

    EXPECT_EQ(out.cmd_type, lipc::CMD_TRANSFER_H2D);
    EXPECT_EQ(out.cmd_seq, 42u);
    EXPECT_EQ(out.gpu_idx, 1u);
    EXPECT_EQ(out.stream_id, 3u);
    EXPECT_EQ(out.transfer.layer_idx, 10u);
    EXPECT_EQ(out.transfer.expert_idx, 200u);
    EXPECT_EQ(out.transfer.sub_component, 0x07);
    EXPECT_EQ(out.transfer.zone, 1);
    EXPECT_EQ(out.transfer.bytes, 1024 * 1024);
}

TEST_F(SpscRingTest, FillToCapacity) {
    for (uint32_t i = 0; i < kSlots; ++i) {
        lipc::Command cmd{};
        cmd.cmd_seq = i;
        ASSERT_TRUE(ring_->try_write(&cmd)) << "write " << i << " failed";
    }
    EXPECT_TRUE(ring_->is_full());

    // 5th write should fail
    lipc::Command extra{};
    extra.cmd_seq = 999;
    EXPECT_FALSE(ring_->try_write(&extra));

    // Read all 4
    for (uint32_t i = 0; i < kSlots; ++i) {
        lipc::Command out{};
        ASSERT_TRUE(ring_->try_read(&out)) << "read " << i << " failed";
        EXPECT_EQ(out.cmd_seq, i);
    }
    EXPECT_TRUE(ring_->is_empty());
}

TEST_F(SpscRingTest, DrainMultiple) {
    for (uint32_t i = 0; i < 3; ++i) {
        lipc::Command cmd{};
        cmd.cmd_seq = 100 + i;
        ASSERT_TRUE(ring_->try_write(&cmd));
    }

    EXPECT_EQ(ring_->available(), 3u);

    std::vector<uint32_t> seqs;
    uint32_t count = ring_->drain([&](const uint8_t* data) {
        auto* cmd = reinterpret_cast<const lipc::Command*>(data);
        seqs.push_back(cmd->cmd_seq);
    });

    EXPECT_EQ(count, 3u);
    ASSERT_EQ(seqs.size(), 3u);
    EXPECT_EQ(seqs[0], 100u);
    EXPECT_EQ(seqs[1], 101u);
    EXPECT_EQ(seqs[2], 102u);
    EXPECT_TRUE(ring_->is_empty());
}

TEST_F(SpscRingTest, WrapAround) {
    // Fill 4 slots
    for (uint32_t i = 0; i < kSlots; ++i) {
        lipc::Command cmd{};
        cmd.cmd_seq = i;
        ASSERT_TRUE(ring_->try_write(&cmd));
    }

    // Read 2, freeing 2 slots
    for (uint32_t i = 0; i < 2; ++i) {
        lipc::Command out{};
        ASSERT_TRUE(ring_->try_read(&out));
        EXPECT_EQ(out.cmd_seq, i);
    }

    // Write 2 more (wrapping around)
    for (uint32_t i = kSlots; i < kSlots + 2; ++i) {
        lipc::Command cmd{};
        cmd.cmd_seq = i;
        ASSERT_TRUE(ring_->try_write(&cmd)) << "wrap write " << i << " failed";
    }
    EXPECT_TRUE(ring_->is_full());

    // Ring full again — can't write
    lipc::Command extra{};
    EXPECT_FALSE(ring_->try_write(&extra));

    // Read remaining 4
    for (uint32_t i = 2; i < kSlots + 2; ++i) {
        lipc::Command out{};
        ASSERT_TRUE(ring_->try_read(&out)) << "wrap read " << i << " failed";
        EXPECT_EQ(out.cmd_seq, i);
    }
    EXPECT_TRUE(ring_->is_empty());
}

TEST_F(SpscRingTest, EmptyRead) {
    lipc::Command out{};
    EXPECT_FALSE(ring_->try_read(&out));
    EXPECT_TRUE(ring_->is_empty());
    EXPECT_EQ(ring_->available(), 0u);
}

// ── Seqlock tests ───────────────────────────────────────────────────────────

TEST(Seqlock, TornReadDetection) {
    // Allocate aligned StateSnapshot
    auto snap_size = sizeof(lipc::StateSnapshot);
    std::vector<uint8_t> buf(snap_size + 64, 0);
    auto* snap = new (align_ptr(buf)) lipc::StateSnapshot{};

    // Initially seqlock=0 (even), read should succeed
    uint64_t seq = lipc::seqlock_read_begin(*snap);
    EXPECT_EQ(seq, 0u);
    EXPECT_TRUE(lipc::seqlock_read_validate(*snap, seq));

    // Begin write: seqlock becomes odd
    lipc::seqlock_write_begin(*snap);
    // Modify data while "writing"
    snap->daemon_cycle_count = 42;

    // Reader should see odd seqlock — seqlock_read_begin would spin,
    // but we test the raw value instead to avoid infinite loop.
    auto raw_seq = *reinterpret_cast<volatile uint64_t*>(&snap->seqlock);
    EXPECT_TRUE(raw_seq & 1);  // Odd = write in progress

    // End write: seqlock becomes even (2)
    lipc::seqlock_write_end(*snap);
    seq = lipc::seqlock_read_begin(*snap);
    EXPECT_EQ(seq, 2u);
    EXPECT_EQ(snap->daemon_cycle_count, 42u);
    EXPECT_TRUE(lipc::seqlock_read_validate(*snap, seq));
}

TEST(Seqlock, ConsistentRead) {
    auto snap_size = sizeof(lipc::StateSnapshot);
    std::vector<uint8_t> buf(snap_size + 64, 0);
    auto* snap = new (align_ptr(buf)) lipc::StateSnapshot{};

    // Write some data under seqlock protocol
    lipc::seqlock_write_begin(*snap);
    snap->daemon_cycle_count = 100;
    snap->timestamp_ns = 123456789;
    snap->num_gpus = 4;
    snap->gpus[0].vram_used_bytes = 8ULL * 1024 * 1024 * 1024;
    snap->gpus[0].vram_total_bytes = 24ULL * 1024 * 1024 * 1024;
    snap->global_acceptance_rate = 0.85;
    snap->shift_detected = 1;
    lipc::seqlock_write_end(*snap);

    // Read back
    uint64_t seq = lipc::seqlock_read_begin(*snap);
    EXPECT_EQ(snap->daemon_cycle_count, 100u);
    EXPECT_EQ(snap->timestamp_ns, 123456789u);
    EXPECT_EQ(snap->num_gpus, 4u);
    EXPECT_EQ(snap->gpus[0].vram_used_bytes, 8ULL * 1024 * 1024 * 1024);
    EXPECT_DOUBLE_EQ(snap->global_acceptance_rate, 0.85);
    EXPECT_EQ(snap->shift_detected, 1);
    EXPECT_TRUE(lipc::seqlock_read_validate(*snap, seq));
}

// ── Multi-threaded producer/consumer ────────────────────────────────────────

TEST(SpscRingMultiThreaded, ProducerConsumer) {
    static constexpr uint32_t kSlots = 8192;
    static constexpr uint32_t kSlotSize = lipc::kCmdSlotBytes;
    static constexpr uint32_t kNumItems = 100'000;

    size_t region_bytes = ring_region_size(kSlots, kSlotSize);
    std::vector<uint8_t> buf(region_bytes + 64, 0);
    void* region = align_ptr(buf);
    lipc::SpscRing<kSlotSize>::init(region, kSlots);

    std::atomic<bool> done{false};
    std::atomic<uint32_t> items_consumed{0};

    // Producer thread
    std::thread producer([&] {
        lipc::SpscRing<kSlotSize> ring(region);
        for (uint32_t i = 0; i < kNumItems; ++i) {
            lipc::Command cmd{};
            cmd.cmd_type = lipc::CMD_NOOP;
            cmd.cmd_seq = i;
            cmd.transfer.layer_idx = i * 7;  // Canary data
            while (!ring.try_write(&cmd)) {
                // Spin until space available
            }
        }
        done.store(true, std::memory_order_release);
    });

    // Consumer thread (this thread)
    lipc::SpscRing<kSlotSize> ring(region);
    uint32_t expected_seq = 0;
    while (expected_seq < kNumItems) {
        lipc::Command out{};
        if (ring.try_read(&out)) {
            ASSERT_EQ(out.cmd_seq, expected_seq)
                << "Out-of-order at " << expected_seq;
            ASSERT_EQ(out.transfer.layer_idx, expected_seq * 7)
                << "Data corruption at " << expected_seq;
            ++expected_seq;
        }
    }

    producer.join();
    EXPECT_EQ(expected_seq, kNumItems);
    EXPECT_TRUE(ring.is_empty());
}

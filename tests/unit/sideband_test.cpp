// Sideband region layout unit tests (IPC-8b).
//
// Validates entry struct sizes, sub-region offset chaining, total size,
// and write-read round-trips for all 7 sub-regions.

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "daemon/ipc_protocol.h"

namespace ipc = layerstorm::ipc;

// ── Struct sizes ────────────────────────────────────────────────────────────

TEST(SidebandLayout, BatchDescriptorEntrySize) {
    EXPECT_EQ(sizeof(ipc::BatchDescriptorEntry), 16u);
}

TEST(SidebandLayout, ExpertPrefetchEntrySize) {
    EXPECT_EQ(sizeof(ipc::ExpertPrefetchEntry), 8u);
}

TEST(SidebandLayout, ExpertEvictionEntrySize) {
    EXPECT_EQ(sizeof(ipc::ExpertEvictionEntry), 8u);
}

TEST(SidebandLayout, TransferBatchEntrySize) {
    EXPECT_EQ(sizeof(ipc::TransferBatchEntry), 16u);
}

TEST(SidebandLayout, ReserveBatchEntrySize) {
    EXPECT_EQ(sizeof(ipc::ReserveBatchEntry), 8u);
}

TEST(SidebandLayout, NvmeReadEntrySize) {
    EXPECT_EQ(sizeof(ipc::NvmeReadEntry), 8u);
}

// ── Sub-region offset chaining ──────────────────────────────────────────────

TEST(SidebandLayout, SubRegionOffsetsChain) {
    using L = ipc::IpcLayout;

    EXPECT_EQ(L::kBatchDescriptorOff, 0u);
    EXPECT_EQ(L::kExpertPrefetchOff, L::kBatchDescriptorOff + L::kBatchDescriptorSize);
    EXPECT_EQ(L::kExpertEvictionOff, L::kExpertPrefetchOff + L::kExpertPrefetchSize);
    EXPECT_EQ(L::kTransferBatchOff,  L::kExpertEvictionOff + L::kExpertEvictionSize);
    EXPECT_EQ(L::kReserveBatchOff,   L::kTransferBatchOff + L::kTransferBatchSize);
    EXPECT_EQ(L::kNvmeReadOff,       L::kReserveBatchOff + L::kReserveBatchSize);
    EXPECT_EQ(L::kTokenIdsOff,       L::kNvmeReadOff + L::kNvmeReadSize);
    EXPECT_EQ(L::kSpecCheckpointOff, L::kTokenIdsOff + L::kTokenIdsSize);
    // Phase 22: routing-export slot (F-4) then seam-checkpoint slot (F-7),
    // appended after the spec-checkpoint region.
    EXPECT_EQ(L::kRoutingExportOff,  L::kSpecCheckpointOff + L::kSpecCheckpointSize);
    EXPECT_EQ(L::kSeamCheckpointOff, L::kRoutingExportOff + L::kRoutingExportSize);
    EXPECT_EQ(L::kSidebandTotalSize, L::kSeamCheckpointOff + L::kSeamCheckpointSize);
}

TEST(SidebandLayout, SubRegionSizes) {
    using L = ipc::IpcLayout;

    EXPECT_EQ(L::kBatchDescriptorSize, ipc::kMaxBatchDescriptors * sizeof(ipc::BatchDescriptorEntry));
    EXPECT_EQ(L::kExpertPrefetchSize,  ipc::kMaxExpertPrefetch * sizeof(ipc::ExpertPrefetchEntry));
    EXPECT_EQ(L::kExpertEvictionSize,  ipc::kMaxExpertEviction * sizeof(ipc::ExpertEvictionEntry));
    EXPECT_EQ(L::kTransferBatchSize,   ipc::kMaxTransferBatch * sizeof(ipc::TransferBatchEntry));
    EXPECT_EQ(L::kReserveBatchSize,    ipc::kMaxReserveBatch * sizeof(ipc::ReserveBatchEntry));
    EXPECT_EQ(L::kNvmeReadSize,        ipc::kMaxNvmeReadBatch * sizeof(ipc::NvmeReadEntry));
    EXPECT_EQ(L::kTokenIdsSize,        ipc::kMaxSidebandTokenIds * sizeof(uint32_t));
}

TEST(SidebandLayout, TotalSize) {
    // Phase 22 grew the sideband by the routing-export (F-4) + seam-checkpoint
    // (F-7) regions: 26624 (through spec-checkpoint) + 65552 (routing) + 4096
    // (seam) = 96272. Regression guard for accidental layout changes.
    using L = ipc::IpcLayout;
    EXPECT_EQ(L::kRoutingExportSize, 65552u);
    EXPECT_EQ(L::kSeamCheckpointSize, 4096u);
    EXPECT_EQ(L::kSidebandTotalSize, 96272u);
}

TEST(SidebandLayout, TotalSizeIncludesSideband) {
    constexpr uint32_t cmd = ipc::kDefaultCmdRingSlots;
    constexpr uint32_t cmp = ipc::kDefaultCmpRingSlots;
    using L = ipc::IpcLayout;

    EXPECT_EQ(L::total_size(cmd, cmp),
              L::sideband_offset(cmd, cmp) + L::kSidebandTotalSize);
    EXPECT_EQ(L::sideband_offset(cmd, cmp),
              L::state_offset(cmd, cmp) + sizeof(ipc::StateSnapshot));
}

// ── Write-read round trips ──────────────────────────────────────────────────

class SidebandWriteRead : public ::testing::Test {
protected:
    void SetUp() override {
        buf_.resize(ipc::IpcLayout::kSidebandTotalSize, 0);
        base_ = buf_.data();
    }

    uint8_t* base_ = nullptr;
    std::vector<uint8_t> buf_;
};

TEST_F(SidebandWriteRead, BatchDescriptor) {
    auto* entries = reinterpret_cast<ipc::BatchDescriptorEntry*>(
        base_ + ipc::IpcLayout::kBatchDescriptorOff);

    // Write 3 entries.
    entries[0] = {1001, 42, 0};
    entries[1] = {1002, 38, 0};
    entries[2] = {1003, 0, 0};

    // Read back.
    EXPECT_EQ(entries[0].seq_id, 1001u);
    EXPECT_EQ(entries[0].token_pos, 42u);
    EXPECT_EQ(entries[1].seq_id, 1002u);
    EXPECT_EQ(entries[1].token_pos, 38u);
    EXPECT_EQ(entries[2].seq_id, 1003u);
    EXPECT_EQ(entries[2].token_pos, 0u);
}

TEST_F(SidebandWriteRead, ExpertPrefetch) {
    auto* entries = reinterpret_cast<ipc::ExpertPrefetchEntry*>(
        base_ + ipc::IpcLayout::kExpertPrefetchOff);

    entries[0] = {5, 42, 0, 2};
    entries[1] = {10, 100, 1, 3};

    EXPECT_EQ(entries[0].layer_idx, 5u);
    EXPECT_EQ(entries[0].expert_idx, 42u);
    EXPECT_EQ(entries[0].zone, 0u);
    EXPECT_EQ(entries[0].gpu_idx, 2u);
    EXPECT_EQ(entries[1].layer_idx, 10u);
    EXPECT_EQ(entries[1].expert_idx, 100u);
    EXPECT_EQ(entries[1].zone, 1u);
    EXPECT_EQ(entries[1].gpu_idx, 3u);
}

TEST_F(SidebandWriteRead, ExpertEviction) {
    auto* entries = reinterpret_cast<ipc::ExpertEvictionEntry*>(
        base_ + ipc::IpcLayout::kExpertEvictionOff);

    entries[0] = {3, 200, 1, 0};

    EXPECT_EQ(entries[0].layer_idx, 3u);
    EXPECT_EQ(entries[0].expert_idx, 200u);
    EXPECT_EQ(entries[0].gpu_idx, 1u);
}

TEST_F(SidebandWriteRead, TransferBatch) {
    auto* entries = reinterpret_cast<ipc::TransferBatchEntry*>(
        base_ + ipc::IpcLayout::kTransferBatchOff);

    entries[0] = {7, 55, 0x07, 0, 1048576};

    EXPECT_EQ(entries[0].layer_idx, 7u);
    EXPECT_EQ(entries[0].expert_idx, 55u);
    EXPECT_EQ(entries[0].sub_component, 0x07u);
    EXPECT_EQ(entries[0].zone, 0u);
    EXPECT_EQ(entries[0].bytes, 1048576);
}

TEST_F(SidebandWriteRead, ReserveBatch) {
    auto* entries = reinterpret_cast<ipc::ReserveBatchEntry*>(
        base_ + ipc::IpcLayout::kReserveBatchOff);

    entries[0] = {2, 128, 1, 1};

    EXPECT_EQ(entries[0].layer_idx, 2u);
    EXPECT_EQ(entries[0].expert_idx, 128u);
    EXPECT_EQ(entries[0].zone, 1u);
    EXPECT_EQ(entries[0].is_duplicate, 1u);
}

TEST_F(SidebandWriteRead, NvmeRead) {
    auto* entries = reinterpret_cast<ipc::NvmeReadEntry*>(
        base_ + ipc::IpcLayout::kNvmeReadOff);

    entries[0] = {8, 33, 2, 0};

    EXPECT_EQ(entries[0].layer_idx, 8u);
    EXPECT_EQ(entries[0].expert_idx, 33u);
    EXPECT_EQ(entries[0].gpu_hint, 2u);
}

TEST_F(SidebandWriteRead, TokenIds) {
    auto* ids = reinterpret_cast<uint32_t*>(
        base_ + ipc::IpcLayout::kTokenIdsOff);

    ids[0] = 12345;
    ids[1] = 67890;
    ids[511] = 99999;

    EXPECT_EQ(ids[0], 12345u);
    EXPECT_EQ(ids[1], 67890u);
    EXPECT_EQ(ids[511], 99999u);
}

TEST_F(SidebandWriteRead, SubRegionsDontOverlap) {
    // Write distinct patterns to the first byte of each sub-region,
    // then verify they don't interfere.
    base_[ipc::IpcLayout::kBatchDescriptorOff] = 0xAA;
    base_[ipc::IpcLayout::kExpertPrefetchOff]  = 0xBB;
    base_[ipc::IpcLayout::kExpertEvictionOff]  = 0xCC;
    base_[ipc::IpcLayout::kTransferBatchOff]   = 0xDD;
    base_[ipc::IpcLayout::kReserveBatchOff]    = 0xEE;
    base_[ipc::IpcLayout::kNvmeReadOff]        = 0x11;
    base_[ipc::IpcLayout::kTokenIdsOff]        = 0x22;

    EXPECT_EQ(base_[ipc::IpcLayout::kBatchDescriptorOff], 0xAA);
    EXPECT_EQ(base_[ipc::IpcLayout::kExpertPrefetchOff],  0xBB);
    EXPECT_EQ(base_[ipc::IpcLayout::kExpertEvictionOff],  0xCC);
    EXPECT_EQ(base_[ipc::IpcLayout::kTransferBatchOff],   0xDD);
    EXPECT_EQ(base_[ipc::IpcLayout::kReserveBatchOff],    0xEE);
    EXPECT_EQ(base_[ipc::IpcLayout::kNvmeReadOff],        0x11);
    EXPECT_EQ(base_[ipc::IpcLayout::kTokenIdsOff],        0x22);
}

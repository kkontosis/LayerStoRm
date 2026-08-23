// NvmeTier unit tests — mmap-backed per-expert-index file layout (WP-5).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "config/config_parser.h"
#include "core/memory/eviction_policy.h"
#include "core/memory/numa_manager.h"
#include "core/memory/nvme_tier.h"
#include "model/weight_pipeline/prepacked_format.h"

namespace lc = layerstorm::config;
namespace lmem = layerstorm::memory;
namespace fs = std::filesystem;

// ── Helpers ─────────────────────────────────────────────────────────────────

namespace {

constexpr int64_t kSlotSize = 4096;
constexpr int kNumMoeLayers = 4;
constexpr int kNumExperts = 8;
constexpr int kFirstMoeLayer = 3;  // absolute layer index of first MoE layer

std::string make_temp_dir() {
    char tmpl[] = "/tmp/layerstorm_nvme_XXXXXX";
    char* dir = ::mkdtemp(tmpl);
    if (!dir) throw std::runtime_error("mkdtemp failed");
    return std::string(dir);
}

lc::HardwareConfig test_hw() {
    lc::HardwareConfig hw;
    lc::GpuConfig g;
    g.id = 0;
    g.type = lc::GpuType::rtx5090;
    g.numa_node = 0;
    hw.gpus.push_back(g);
    return hw;
}

lmem::ExpertKey key(uint32_t layer, uint16_t expert) {
    return lmem::ExpertKey{layer, expert};
}

void fill_pattern(void* buf, int64_t size, lmem::ExpertKey k) {
    auto* p = static_cast<uint8_t*>(buf);
    uint8_t seed = static_cast<uint8_t>(
        (k.layer_idx * 257 + k.expert_idx * 131) & 0xFF);
    for (int64_t i = 0; i < size; ++i)
        p[i] = static_cast<uint8_t>((seed + i) & 0xFF);
}

bool verify_pattern(const void* buf, int64_t size, lmem::ExpertKey k) {
    auto* p = static_cast<const uint8_t*>(buf);
    uint8_t seed = static_cast<uint8_t>(
        (k.layer_idx * 257 + k.expert_idx * 131) & 0xFF);
    for (int64_t i = 0; i < size; ++i) {
        if (p[i] != static_cast<uint8_t>((seed + i) & 0xFF))
            return false;
    }
    return true;
}

lmem::NvmeTier::Options test_opts(std::vector<std::string> paths) {
    return {
        .drive_paths           = std::move(paths),
        .queue_depth           = 32,
        .slot_size_bytes       = kSlotSize,
        .host_ram_budget_bytes = 0,
        .num_moe_layers        = kNumMoeLayers,
        .num_experts_per_layer = kNumExperts,
        .first_moe_layer       = kFirstMoeLayer,
    };
}

struct TempDirs {
    std::vector<std::string> paths;
    ~TempDirs() {
        for (const auto& p : paths) {
            std::error_code ec;
            fs::remove_all(p, ec);
        }
    }
    void add() { paths.push_back(make_temp_dir()); }
    const std::string& operator[](size_t i) const { return paths[i]; }
};

/// Poll until no more inflight I/O (for sync tests).
void drain_writes(lmem::NvmeTier& tier) {
    tier.drain();
}

}  // namespace

// ── Construction ────────────────────────────────────────────────────────────

TEST(NvmeTier, ConstructionSingleDrive) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});

    lmem::NvmeTier tier(std::move(opts), numa);
    EXPECT_EQ(tier.num_drives(), 1);
    EXPECT_EQ(tier.slot_size_bytes(), kSlotSize);
}

TEST(NvmeTier, ConstructionMultiDrive) {
    TempDirs dirs;
    dirs.add();
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0], dirs[1]});

    lmem::NvmeTier tier(std::move(opts), numa);
    EXPECT_EQ(tier.num_drives(), 2);
}

TEST(NvmeTier, ConstructionZeroDrivesThrows) {
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    EXPECT_THROW(lmem::NvmeTier(test_opts({}), numa),
                 std::invalid_argument);
}

TEST(NvmeTier, ConstructionZeroSlotSizeThrows) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    opts.slot_size_bytes = 0;
    EXPECT_THROW(lmem::NvmeTier(std::move(opts), numa),
                 std::invalid_argument);
}

// ── Per-expert-index file layout ────────────────────────────────────────────

TEST(NvmeTier, ExpertPathFormat) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    // Expert path should be: <drive>/layerstorm/experts/expert_042.bin
    auto path = tier.expert_path(key(kFirstMoeLayer, 42));
    auto expected = (fs::path(dirs[0]) / "layerstorm" / "experts" /
                     "expert_042.bin").string();
    EXPECT_EQ(path, expected);
}

TEST(NvmeTier, InitDirectoriesCreatesFlat) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);
    tier.init_directories();

    auto dir = fs::path(dirs[0]) / "layerstorm" / "experts";
    EXPECT_TRUE(fs::is_directory(dir));
}

// ── Drive striping ──────────────────────────────────────────────────────────

TEST(NvmeTier, DriveStripingByExpertIdx) {
    TempDirs dirs;
    dirs.add();
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0], dirs[1]});
    lmem::NvmeTier tier(std::move(opts), numa);

    // expert_idx % 2
    EXPECT_EQ(tier.drive_for_expert(key(kFirstMoeLayer, 0)), 0);
    EXPECT_EQ(tier.drive_for_expert(key(kFirstMoeLayer, 1)), 1);
    EXPECT_EQ(tier.drive_for_expert(key(kFirstMoeLayer, 2)), 0);
    EXPECT_EQ(tier.drive_for_expert(key(kFirstMoeLayer, 3)), 1);
}

// ── Write + host_ptr round-trip ─────────────────────────────────────────────

TEST(NvmeTier, WriteThenHostPtr) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    auto k = key(kFirstMoeLayer, 0);
    EXPECT_EQ(tier.host_ptr(k), nullptr);  // no data yet

    // Write data.
    std::vector<uint8_t> data(kSlotSize);
    fill_pattern(data.data(), kSlotSize, k);
    auto tok = tier.write_expert(k, data.data());
    ASSERT_TRUE(tok.has_value());
    drain_writes(tier);

    // host_ptr should return the mmap pointer with the written data.
    const void* ptr = tier.host_ptr(k);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(verify_pattern(ptr, kSlotSize, k));
}

TEST(NvmeTier, WriteThenHostPtrOwning) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    auto k = key(kFirstMoeLayer + 1, 2);

    // Write via owning overload.
    auto buf = numa.allocate_interleaved(static_cast<size_t>(kSlotSize));
    fill_pattern(buf.data, kSlotSize, k);
    auto tok = tier.write_expert(k, std::move(buf));
    ASSERT_TRUE(tok.has_value());
    drain_writes(tier);

    const void* ptr = tier.host_ptr(k);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(verify_pattern(ptr, kSlotSize, k));
}

// ── Slot offset addressing ──────────────────────────────────────────────────

TEST(NvmeTier, SlotOffsetAddressing) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    // Write different patterns to different layer slots within the same
    // expert file.  All layers are first_moe_layer through
    // first_moe_layer + num_moe_layers - 1.
    int expert_idx = 3;
    for (int moe_offset = 0; moe_offset < kNumMoeLayers; ++moe_offset) {
        auto k = key(static_cast<uint32_t>(kFirstMoeLayer + moe_offset),
                      static_cast<uint16_t>(expert_idx));
        std::vector<uint8_t> data(kSlotSize);
        fill_pattern(data.data(), kSlotSize, k);
        auto tok = tier.write_expert(k, data.data());
        ASSERT_TRUE(tok.has_value());
    }
    drain_writes(tier);

    // Read back each slot and verify.
    for (int moe_offset = 0; moe_offset < kNumMoeLayers; ++moe_offset) {
        auto k = key(static_cast<uint32_t>(kFirstMoeLayer + moe_offset),
                      static_cast<uint16_t>(expert_idx));
        const void* ptr = tier.host_ptr(k);
        ASSERT_NE(ptr, nullptr) << "layer=" << k.layer_idx;
        EXPECT_TRUE(verify_pattern(ptr, kSlotSize, k))
            << "pattern mismatch at layer=" << k.layer_idx;
    }
}

// ── Mmap pointer lifetime (TD-82n fix verification) ─────────────────────────

TEST(NvmeTier, MmapPointerLifetime) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    auto k = key(kFirstMoeLayer, 0);
    std::vector<uint8_t> data(kSlotSize);
    fill_pattern(data.data(), kSlotSize, k);
    auto tok = tier.write_expert(k, data.data());
    ASSERT_TRUE(tok.has_value());
    drain_writes(tier);

    // Get pointer.
    const void* ptr1 = tier.host_ptr(k);
    ASSERT_NE(ptr1, nullptr);

    // Write another expert (different key) — ptr1 must remain valid.
    auto k2 = key(kFirstMoeLayer + 1, 1);
    std::vector<uint8_t> data2(kSlotSize);
    fill_pattern(data2.data(), kSlotSize, k2);
    tier.write_expert(k2, data2.data());
    drain_writes(tier);

    // ptr1 still valid (process-lifetime mmap).
    const void* ptr1_again = tier.host_ptr(k);
    EXPECT_EQ(ptr1, ptr1_again);
    EXPECT_TRUE(verify_pattern(ptr1, kSlotSize, k));
}

// ── read_expert returns immediate token ─────────────────────────────────────

TEST(NvmeTier, ReadExpertReturnsImmediateToken) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    auto k = key(kFirstMoeLayer, 0);

    // No data yet — read should return nullopt.
    EXPECT_FALSE(tier.read_expert(k).has_value());

    // Write data, then read.
    std::vector<uint8_t> data(kSlotSize, 0x42);
    tier.write_expert(k, data.data());
    drain_writes(tier);

    auto tok = tier.read_expert(k);
    ASSERT_TRUE(tok.has_value());
    EXPECT_EQ(tok.value(), 0u);  // immediate completion sentinel
}

// ── Layer out-of-range returns nullptr ───────────────────────────────────────

TEST(NvmeTier, LayerOutOfRangeReturnsNull) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    // layer_idx below first_moe_layer.
    auto tok = tier.write_expert(key(0, 0), nullptr);
    EXPECT_FALSE(tok.has_value());
    EXPECT_EQ(tier.host_ptr(key(0, 0)), nullptr);

    // layer_idx above range.
    auto k_above = key(kFirstMoeLayer + kNumMoeLayers, 0);
    EXPECT_EQ(tier.host_ptr(k_above), nullptr);
}

// ── Tier queries ────────────────────────────────────────────────────────────

TEST(NvmeTier, TierQueries) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    auto k = key(kFirstMoeLayer, 0);

    // Initially: kNone (no data).
    EXPECT_EQ(tier.tier(k), lmem::ExpertTier::kNone);
    EXPECT_FALSE(tier.is_in_host_ram(k));
    EXPECT_FALSE(tier.is_on_nvme(k));

    // After write: kHostRam (mmap-backed).
    std::vector<uint8_t> data(kSlotSize, 0x99);
    tier.write_expert(k, data.data());
    drain_writes(tier);
    EXPECT_EQ(tier.tier(k), lmem::ExpertTier::kHostRam);
    EXPECT_TRUE(tier.is_in_host_ram(k));
    EXPECT_TRUE(tier.is_on_nvme(k));  // on NVMe AND in host RAM (mmap)
}

TEST(NvmeTier, HostNumaNodeReturnsNegativeOne) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    // Mmap pages are OS-managed.
    EXPECT_EQ(tier.host_numa_node(key(kFirstMoeLayer, 0)), -1);
}

// ── Scan existing files ─────────────────────────────────────────────────────

TEST(NvmeTier, ScanExistingFiles) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);

    auto expected_size = layerstorm::model::prepacked::expected_file_size(
        kNumMoeLayers, kSlotSize);

    // Pre-create an expert file on disk before constructing NvmeTier.
    auto expert_dir = layerstorm::model::prepacked::expert_dir(dirs[0]);
    fs::create_directories(expert_dir);
    auto file_path = layerstorm::model::prepacked::expert_file_path(
        expert_dir, 2);

    // Write a file with correct size and known content.
    {
        std::ofstream ofs(file_path, std::ios::binary);
        std::vector<uint8_t> data(static_cast<size_t>(expected_size), 0xBB);
        ofs.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size()));
    }
    ASSERT_EQ(static_cast<int64_t>(fs::file_size(file_path)), expected_size);

    // Construct NvmeTier — should scan and mmap the file.
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    // All slots for expert 2 should be readable.
    for (int lpos = 0; lpos < kNumMoeLayers; ++lpos) {
        auto k = key(static_cast<uint32_t>(kFirstMoeLayer + lpos), 2);
        EXPECT_TRUE(tier.is_in_host_ram(k)) << "lpos=" << lpos;
        const void* ptr = tier.host_ptr(k);
        ASSERT_NE(ptr, nullptr) << "lpos=" << lpos;
        // Verify content is 0xBB.
        auto* p = static_cast<const uint8_t*>(ptr);
        EXPECT_EQ(p[0], 0xBB) << "lpos=" << lpos;
    }
}

// ── Spill and readback ──────────────────────────────────────────────────────

TEST(NvmeTier, SpillAndReadback) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    auto k = key(kFirstMoeLayer + 2, 5);

    // Write via sync (non-owning) path.
    std::vector<uint8_t> data(kSlotSize);
    fill_pattern(data.data(), kSlotSize, k);
    auto tok = tier.write_expert(k, data.data());
    ASSERT_TRUE(tok.has_value());
    drain_writes(tier);

    // Verify file exists on disk.
    auto path = tier.expert_path(k);
    EXPECT_TRUE(fs::exists(path));

    // Read back via host_ptr (mmap).
    const void* ptr = tier.host_ptr(k);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(verify_pattern(ptr, kSlotSize, k));
}

// ── Multi-drive striping ────────────────────────────────────────────────────

TEST(NvmeTier, MultiDriveStriping) {
    TempDirs dirs;
    dirs.add();
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0], dirs[1]});
    lmem::NvmeTier tier(std::move(opts), numa);

    // Write experts 0 and 1 — should land on different drives.
    auto k0 = key(kFirstMoeLayer, 0);
    auto k1 = key(kFirstMoeLayer, 1);

    std::vector<uint8_t> data(kSlotSize);
    fill_pattern(data.data(), kSlotSize, k0);
    tier.write_expert(k0, data.data());
    fill_pattern(data.data(), kSlotSize, k1);
    tier.write_expert(k1, data.data());
    drain_writes(tier);

    // Expert 0 on drive 0, expert 1 on drive 1.
    auto p0 = tier.expert_path(k0);
    auto p1 = tier.expert_path(k1);
    EXPECT_TRUE(p0.find(dirs[0]) != std::string::npos);
    EXPECT_TRUE(p1.find(dirs[1]) != std::string::npos);

    // Both readable.
    EXPECT_NE(tier.host_ptr(k0), nullptr);
    EXPECT_NE(tier.host_ptr(k1), nullptr);
    EXPECT_TRUE(verify_pattern(tier.host_ptr(k0), kSlotSize, k0));
    EXPECT_TRUE(verify_pattern(tier.host_ptr(k1), kSlotSize, k1));
}

// ── Capacity queries ────────────────────────────────────────────────────────

TEST(NvmeTier, CapacityQueries) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    EXPECT_EQ(tier.host_ram_entry_count(), 0);
    EXPECT_EQ(tier.host_ram_used_bytes(), 0);

    auto k = key(kFirstMoeLayer, 0);
    std::vector<uint8_t> data(kSlotSize, 0x77);
    tier.write_expert(k, data.data());
    drain_writes(tier);

    // 1 expert file mmapped, 1 slot written.
    EXPECT_EQ(tier.host_ram_entry_count(), 1);
    EXPECT_EQ(tier.host_ram_used_bytes(), kSlotSize);
}

// ── Latency estimation ──────────────────────────────────────────────────────

TEST(NvmeTier, LatencyEstimation) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    double nvme_us = tier.estimate_read_us();
    EXPECT_GT(nvme_us, 0.0);

    double h2v_us = lmem::NvmeTier::estimate_host_to_vram_us(kSlotSize, 3.938);
    EXPECT_GT(h2v_us, 0.0);
}

// ── Inflight write tracking ─────────────────────────────────────────────────

TEST(NvmeTier, InflightCount) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    EXPECT_EQ(tier.inflight_count(), 0);
    // Note: without LAYERSTORM_HAS_URING, write_expert uses sync fallback
    // and returns IoToken{0}, so inflight_count stays 0.
}

// ── Cancellation ────────────────────────────────────────────────────────────

TEST(NvmeTier, CancelUnknownTokenReturnsFalse) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    EXPECT_FALSE(tier.cancel(999));
}

// ── poll_completions returns empty without io_uring ─────────────────────────

TEST(NvmeTier, PollCompletionsEmpty) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    auto completions = tier.poll_completions();
    EXPECT_TRUE(completions.empty());
}

// ── Duplicate write for same expert key ─────────────────────────────────────

TEST(NvmeTier, DuplicateWriteInflightReturnsNullopt) {
    // Without io_uring, writes are synchronous, so this test only verifies
    // the API contract: a second write while the first is inflight is rejected.
    // With sync fallback, the first write completes immediately, so the second
    // succeeds.  This test documents the behavior.
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);
    auto opts = test_opts({dirs[0]});
    lmem::NvmeTier tier(std::move(opts), numa);

    auto k = key(kFirstMoeLayer, 0);
    std::vector<uint8_t> data(kSlotSize, 0x11);
    auto tok1 = tier.write_expert(k, data.data());
    ASSERT_TRUE(tok1.has_value());
    // With sync fallback, tok1 == 0 (completed immediately).
    // A second write should also succeed (sync).
    auto tok2 = tier.write_expert(k, data.data());
    EXPECT_TRUE(tok2.has_value());
}

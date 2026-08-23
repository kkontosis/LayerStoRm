/// NVMe tier manual test — exercises real filesystem I/O.
///
/// Build:  cmake --build build --target nvme_tier_manual
/// Run:    ./build/tests/manual/nvme_tier_manual
///
/// Uses mkdtemp temp directories (not real NVMe), but exercises the full
/// admit → evict-to-disk → read-back round-trip with real file I/O.
/// Small data (4 KB per expert) to keep it fast.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/memory/eviction_policy.h"
#include "core/memory/numa_manager.h"
#include "core/memory/nvme_tier.h"

namespace lmem = layerstorm::memory;
namespace lc = layerstorm::config;
namespace fs = std::filesystem;

// ── Helpers ─────────────────────────────────────────────────────────────────

namespace {

constexpr int64_t kExpertBytes = 4096;  // Small, page-aligned.
constexpr int kNumLayers = 4;
constexpr int kNumExperts = 16;

std::string make_temp_dir() {
    char tmpl[] = "/tmp/layerstorm_manual_XXXXXX";
    char* dir = ::mkdtemp(tmpl);
    if (!dir) throw std::runtime_error("mkdtemp failed");
    return std::string(dir);
}

struct TempDirs {
    std::vector<std::string> paths;
    ~TempDirs() {
        for (const auto& p : paths) {
            std::error_code ec;
            fs::remove_all(p, ec);
        }
    }
    void add(int count = 1) {
        for (int i = 0; i < count; ++i)
            paths.push_back(make_temp_dir());
    }
};

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

/// Fill buffer with deterministic pattern derived from expert identity.
void fill_pattern(void* buf, int64_t size, lmem::ExpertKey k) {
    auto* p = static_cast<uint8_t*>(buf);
    uint32_t seed = k.layer_idx * 257u + k.expert_idx * 131u;
    for (int64_t i = 0; i < size; ++i)
        p[i] = static_cast<uint8_t>((seed + static_cast<uint32_t>(i)) & 0xFF);
}

bool verify_pattern(const void* buf, int64_t size, lmem::ExpertKey k) {
    auto* p = static_cast<const uint8_t*>(buf);
    uint32_t seed = k.layer_idx * 257u + k.expert_idx * 131u;
    for (int64_t i = 0; i < size; ++i) {
        if (p[i] != static_cast<uint8_t>((seed + static_cast<uint32_t>(i)) & 0xFF))
            return false;
    }
    return true;
}

lmem::NvmeTier::Options make_opts(std::vector<std::string> paths,
                                   int64_t host_entries) {
    lmem::NvmeTier::Options opts;
    opts.drive_paths = std::move(paths);
    opts.queue_depth = 32;
    opts.expert_bytes = kExpertBytes;
    opts.host_ram_budget_bytes = host_entries * kExpertBytes;
    opts.direct_io = false;  // tmpfs, not real NVMe.
    opts.num_moe_layers = kNumLayers;
    opts.num_experts_per_layer = kNumExperts;
    return opts;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Single-drive round-trip: admit → evict → verify on disk → read back
// ═══════════════════════════════════════════════════════════════════════════

TEST(NvmeTierManual, SingleDriveRoundTrip) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);

    lmem::NvmeTier tier(make_opts({dirs.paths[0]}, /*host_entries=*/10), numa);
    tier.init_directories();

    // Admit 5 experts with distinct patterns.
    for (uint16_t e = 0; e < 5; ++e) {
        auto buf = numa.allocate_interleaved(kExpertBytes);
        fill_pattern(buf.data, kExpertBytes, key(1, e));
        ASSERT_TRUE(tier.admit(key(1, e), buf));
    }
    EXPECT_EQ(tier.host_ram_entry_count(), 5);

    // Evict all 5 to NVMe.
    int evicted = tier.evict_lru(5);
    EXPECT_EQ(evicted, 5);
    EXPECT_EQ(tier.host_ram_entry_count(), 0);

    // Verify files exist on disk and data matches.
    for (uint16_t e = 0; e < 5; ++e) {
        auto path = tier.expert_path(key(1, e));
        ASSERT_TRUE(fs::exists(path)) << "Missing file: " << path;

        auto fsize = fs::file_size(path);
        EXPECT_EQ(static_cast<int64_t>(fsize), kExpertBytes)
            << "Wrong size for " << path;

        std::vector<uint8_t> readback(kExpertBytes);
        std::ifstream ifs(path, std::ios::binary);
        ASSERT_TRUE(ifs.good()) << "Cannot open: " << path;
        ifs.read(reinterpret_cast<char*>(readback.data()), kExpertBytes);
        EXPECT_TRUE(verify_pattern(readback.data(), kExpertBytes, key(1, e)))
            << "Data mismatch for layer=1 expert=" << e;
    }

    // All 5 should report kNvme tier.
    for (uint16_t e = 0; e < 5; ++e) {
        EXPECT_EQ(tier.tier(key(1, e)), lmem::ExpertTier::kNvme);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-drive balanced distribution
// ═══════════════════════════════════════════════════════════════════════════

TEST(NvmeTierManual, MultiDriveBalancedDistribution) {
    TempDirs dirs;
    dirs.add(4);
    auto hw = test_hw();
    lmem::NumaManager numa(hw);

    lmem::NvmeTier tier(make_opts(dirs.paths, /*host_entries=*/200), numa);
    tier.init_directories();

    // Admit 100 experts across 4 layers × 25 experts.
    int total = 0;
    for (uint32_t l = 0; l < 4; ++l) {
        for (uint16_t e = 0; e < 25; ++e) {
            auto buf = numa.allocate_interleaved(kExpertBytes);
            fill_pattern(buf.data, kExpertBytes, key(l, e));
            ASSERT_TRUE(tier.admit(key(l, e), buf));
            total++;
        }
    }
    EXPECT_EQ(total, 100);

    // Evict all to NVMe.
    int evicted = tier.evict_lru(100);
    EXPECT_EQ(evicted, 100);

    // Count files per drive.
    std::vector<int> file_counts(4, 0);
    for (uint32_t l = 0; l < 4; ++l) {
        for (uint16_t e = 0; e < 25; ++e) {
            int drv = tier.drive_for_expert(key(l, e));
            auto path = tier.expert_path(key(l, e));
            if (fs::exists(path)) file_counts[drv]++;
        }
    }

    int total_files = 0;
    for (int i = 0; i < 4; ++i) {
        total_files += file_counts[i];
        // Each drive should get roughly 25 (±50%).
        EXPECT_GT(file_counts[i], 12)
            << "Drive " << i << " has too few files: " << file_counts[i];
        EXPECT_LT(file_counts[i], 40)
            << "Drive " << i << " has too many files: " << file_counts[i];
    }
    EXPECT_EQ(total_files, 100);

    // Verify data integrity for all 100 experts.
    for (uint32_t l = 0; l < 4; ++l) {
        for (uint16_t e = 0; e < 25; ++e) {
            auto path = tier.expert_path(key(l, e));
            std::vector<uint8_t> data(kExpertBytes);
            std::ifstream ifs(path, std::ios::binary);
            ASSERT_TRUE(ifs.good()) << "Cannot open: " << path;
            ifs.read(reinterpret_cast<char*>(data.data()), kExpertBytes);
            EXPECT_TRUE(verify_pattern(data.data(), kExpertBytes, key(l, e)))
                << "Data mismatch: layer=" << l << " expert=" << e;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LRU eviction order under pressure
// ═══════════════════════════════════════════════════════════════════════════

TEST(NvmeTierManual, LruEvictionOrder) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);

    // Budget for only 4 entries.
    lmem::NvmeTier tier(make_opts({dirs.paths[0]}, /*host_entries=*/4), numa);
    tier.init_directories();

    // Admit 4 experts: e0, e1, e2, e3 (in order).
    for (uint16_t e = 0; e < 4; ++e) {
        auto buf = numa.allocate_interleaved(kExpertBytes);
        fill_pattern(buf.data, kExpertBytes, key(0, e));
        ASSERT_TRUE(tier.admit(key(0, e), buf));
    }

    // Touch e0 and e2, making e1 and e3 the oldest.
    tier.touch(key(0, 0));
    tier.touch(key(0, 2));

    // Evict 2 — should evict e1 and e3 (oldest LRU ticks).
    int evicted = tier.evict_lru(2);
    EXPECT_EQ(evicted, 2);

    // e0 and e2 should still be warm.
    EXPECT_TRUE(tier.is_in_host_ram(key(0, 0)));
    EXPECT_TRUE(tier.is_in_host_ram(key(0, 2)));

    // e1 and e3 should be on NVMe.
    EXPECT_TRUE(tier.is_on_nvme(key(0, 1)));
    EXPECT_TRUE(tier.is_on_nvme(key(0, 3)));
    EXPECT_FALSE(tier.is_in_host_ram(key(0, 1)));
    EXPECT_FALSE(tier.is_in_host_ram(key(0, 3)));
}

// ═══════════════════════════════════════════════════════════════════════════
// Evict-then-admit cycle (simulates cache churn)
// ═══════════════════════════════════════════════════════════════════════════

TEST(NvmeTierManual, EvictAdmitCycle) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);

    // Budget for 3 entries.
    lmem::NvmeTier tier(make_opts({dirs.paths[0]}, /*host_entries=*/3), numa);
    tier.init_directories();

    // Fill cache.
    for (uint16_t e = 0; e < 3; ++e) {
        auto buf = numa.allocate_interleaved(kExpertBytes);
        fill_pattern(buf.data, kExpertBytes, key(0, e));
        ASSERT_TRUE(tier.admit(key(0, e), buf));
    }

    // Cache is full — next admit should fail.
    auto buf_overflow = numa.allocate_interleaved(kExpertBytes);
    EXPECT_FALSE(tier.admit(key(0, 10), buf_overflow));
    numa.free(buf_overflow);

    // Evict 1, then admit a new expert.
    tier.evict_lru(1);
    EXPECT_EQ(tier.host_ram_entry_count(), 2);

    auto buf_new = numa.allocate_interleaved(kExpertBytes);
    fill_pattern(buf_new.data, kExpertBytes, key(0, 10));
    EXPECT_TRUE(tier.admit(key(0, 10), buf_new));
    EXPECT_EQ(tier.host_ram_entry_count(), 3);

    // Repeat cycle a few times.
    for (int cycle = 0; cycle < 5; ++cycle) {
        tier.evict_lru(1);
        auto buf = numa.allocate_interleaved(kExpertBytes);
        auto k = key(2, static_cast<uint16_t>(cycle));
        fill_pattern(buf.data, kExpertBytes, k);
        EXPECT_TRUE(tier.admit(k, buf));
    }

    EXPECT_EQ(tier.host_ram_entry_count(), 3);
    // Multiple experts should be on NVMe.
    EXPECT_GT(tier.total_nvme_used_bytes(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Directory structure created correctly across multiple drives
// ═══════════════════════════════════════════════════════════════════════════

TEST(NvmeTierManual, DirectoryStructure) {
    TempDirs dirs;
    dirs.add(3);
    auto hw = test_hw();
    lmem::NumaManager numa(hw);

    auto opts = make_opts(dirs.paths, 10);
    opts.num_moe_layers = 6;
    lmem::NvmeTier tier(opts, numa);
    tier.init_directories();

    // Every drive should have layerstorm/experts/layer_000..layer_005.
    for (const auto& drive_path : dirs.paths) {
        for (int l = 0; l < 6; ++l) {
            std::string layer_name = "layer_";
            if (l < 100) layer_name += "0";
            if (l < 10) layer_name += "0";
            layer_name += std::to_string(l);
            fs::path dir = fs::path(drive_path) / "layerstorm" / "experts" / layer_name;
            EXPECT_TRUE(fs::is_directory(dir)) << "Missing: " << dir.string();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Destructor cleanup — NumaManager bytes return to zero
// ═══════════════════════════════════════════════════════════════════════════

TEST(NvmeTierManual, DestructorCleansUpMemory) {
    TempDirs dirs;
    dirs.add();
    auto hw = test_hw();
    lmem::NumaManager numa(hw);

    size_t baseline = numa.total_allocated_bytes();
    {
        lmem::NvmeTier tier(make_opts({dirs.paths[0]}, 10), numa);
        tier.init_directories();

        for (uint16_t e = 0; e < 5; ++e) {
            auto buf = numa.allocate_interleaved(kExpertBytes);
            fill_pattern(buf.data, kExpertBytes, key(0, e));
            tier.admit(key(0, e), buf);
        }
        EXPECT_EQ(numa.total_allocated_bytes(), baseline + 5 * kExpertBytes);
        // tier goes out of scope — destructor should free all warm cache buffers.
    }
    EXPECT_EQ(numa.total_allocated_bytes(), baseline);
}

// ═══════════════════════════════════════════════════════════════════════════
// io_uring async round-trip (only when LAYERSTORM_HAS_URING is defined)
// ═══════════════════════════════════════════════════════════════════════════

#ifdef LAYERSTORM_HAS_URING

TEST(NvmeTierManual, AsyncRoundTrip) {
    TempDirs dirs;
    dirs.add(2);
    auto hw = test_hw();
    lmem::NumaManager numa(hw);

    lmem::NvmeTier tier(make_opts(dirs.paths, /*host_entries=*/20), numa);
    tier.init_directories();

    // Write 10 experts to NVMe via sync eviction.
    for (uint16_t e = 0; e < 10; ++e) {
        auto buf = numa.allocate_interleaved(kExpertBytes);
        fill_pattern(buf.data, kExpertBytes, key(0, e));
        tier.admit(key(0, e), buf);
    }
    tier.evict_lru(10);

    // All 10 should be kNvme.
    for (uint16_t e = 0; e < 10; ++e)
        EXPECT_EQ(tier.tier(key(0, e)), lmem::ExpertTier::kNvme);

    // Submit async reads for all 10.
    int submitted = 0;
    for (uint16_t e = 0; e < 10; ++e) {
        auto tok = tier.read_expert(key(0, e), 0);
        if (tok.has_value()) submitted++;
    }
    EXPECT_EQ(submitted, 10);

    // Drain all in-flight I/O.
    tier.drain();
    EXPECT_EQ(tier.inflight_count(), 0);

    // All should now be warm, with correct data.
    for (uint16_t e = 0; e < 10; ++e) {
        EXPECT_TRUE(tier.is_in_host_ram(key(0, e)))
            << "Expert " << e << " not in warm cache after drain";
        EXPECT_TRUE(verify_pattern(tier.host_ptr(key(0, e)), kExpertBytes, key(0, e)))
            << "Data mismatch for expert " << e;
    }
}

#endif  // LAYERSTORM_HAS_URING

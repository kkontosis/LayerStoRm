#include <gtest/gtest.h>

#include <dirent.h>
#include <sched.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/memory/numa_manager.h"

namespace lc = layerstorm::config;
namespace lm = layerstorm::memory;

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace {

lc::GpuConfig make_gpu(int id, int numa_node) {
    lc::GpuConfig g;
    g.id = id;
    g.type = lc::GpuType::rtx5090;
    g.vram_gb = 32.0;
    g.numa_node = numa_node;
    return g;
}

// 4 GPUs across 2 NUMA nodes: GPUs 0,1 on node 0; GPUs 2,3 on node 1.
lc::HardwareConfig standard_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 256;
    hw.gpus = {make_gpu(0, 0), make_gpu(1, 0), make_gpu(2, 1), make_gpu(3, 1)};
    hw.tp_array = {0, 1};
    return hw;
}

// All GPUs on same NUMA node.
lc::HardwareConfig single_node_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 128;
    hw.gpus = {make_gpu(0, 0), make_gpu(1, 0)};
    return hw;
}

// Empty GPU list.
lc::HardwareConfig empty_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 64;
    return hw;
}

// Non-contiguous NUMA nodes: 0, 2, 5.
lc::HardwareConfig noncontiguous_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 256;
    hw.gpus = {make_gpu(0, 0), make_gpu(1, 2), make_gpu(2, 5)};
    return hw;
}

size_t page_size() {
    return static_cast<size_t>(sysconf(_SC_PAGESIZE));
}

}  // namespace

// ── Topology ─────────────────────────────────────────────────────────────────

TEST(NumaTopology, StandardTwoNodes) {
    lm::NumaManager mgr(standard_hw());
    EXPECT_EQ(mgr.num_nodes(), 2);
    EXPECT_EQ(mgr.gpu_numa_node(0), 0);
    EXPECT_EQ(mgr.gpu_numa_node(1), 0);
    EXPECT_EQ(mgr.gpu_numa_node(2), 1);
    EXPECT_EQ(mgr.gpu_numa_node(3), 1);
}

TEST(NumaTopology, SingleNode) {
    lm::NumaManager mgr(single_node_hw());
    EXPECT_EQ(mgr.num_nodes(), 1);
    EXPECT_EQ(mgr.gpu_numa_node(0), 0);
    EXPECT_EQ(mgr.gpu_numa_node(1), 0);
}

TEST(NumaTopology, EmptyGpuList) {
    lm::NumaManager mgr(empty_hw());
    EXPECT_EQ(mgr.num_nodes(), 0);
}

TEST(NumaTopology, InvalidGpuId) {
    lm::NumaManager mgr(standard_hw());
    EXPECT_EQ(mgr.gpu_numa_node(-1), -1);
    EXPECT_EQ(mgr.gpu_numa_node(99), -1);
}

TEST(NumaTopology, NonContiguousNodes) {
    lm::NumaManager mgr(noncontiguous_hw());
    EXPECT_EQ(mgr.num_nodes(), 3);
    EXPECT_EQ(mgr.gpu_numa_node(0), 0);
    EXPECT_EQ(mgr.gpu_numa_node(1), 2);
    EXPECT_EQ(mgr.gpu_numa_node(2), 5);
}

TEST(NumaTopology, NumaAvailableMatchesRuntime) {
    lm::NumaManager mgr(standard_hw());
    // On a single-socket machine numa_available() may return false even with
    // libnuma installed (kernel reports no NUMA topology). Either way, no crash.
    (void)mgr.numa_available();
}

// ── Allocation ───────────────────────────────────────────────────────────────

TEST(NumaAlloc, OnNodeReturnsValid) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_on_node(4096, 0);
    EXPECT_NE(buf.data, nullptr);
    EXPECT_GE(buf.size, 4096u);
    EXPECT_EQ(buf.numa_node, 0);
    mgr.free(buf);
}

TEST(NumaAlloc, OnNodePageAligned) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_on_node(4096, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buf.data) % page_size(), 0u);
    EXPECT_EQ(buf.size % page_size(), 0u);
    mgr.free(buf);
}

TEST(NumaAlloc, SizeRoundedUp) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_on_node(100, 0);
    EXPECT_EQ(buf.size, page_size());
    mgr.free(buf);
}

TEST(NumaAlloc, InterleavedReturnsValid) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_interleaved(8192);
    EXPECT_NE(buf.data, nullptr);
    EXPECT_GE(buf.size, 8192u);
    EXPECT_EQ(buf.numa_node, -1);
    mgr.free(buf);
}

TEST(NumaAlloc, ForGpuDelegatesToNode) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_for_gpu(4096, 2);  // GPU 2 is on NUMA node 1
    EXPECT_NE(buf.data, nullptr);
    EXPECT_EQ(buf.numa_node, 1);
    mgr.free(buf);
}

TEST(NumaAlloc, ForGpuInvalidFallsBackToInterleaved) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_for_gpu(4096, 99);
    EXPECT_NE(buf.data, nullptr);
    EXPECT_EQ(buf.numa_node, -1);
    mgr.free(buf);
}

TEST(NumaAlloc, LargeBuffer) {
    lm::NumaManager mgr(standard_hw());
    constexpr size_t expert_size = 18 * 1024 * 1024;  // 18 MB
    auto buf = mgr.allocate_on_node(expert_size, 0);
    EXPECT_NE(buf.data, nullptr);
    EXPECT_GE(buf.size, expert_size);
    // Write and read back to verify accessibility.
    auto* p = static_cast<uint8_t*>(buf.data);
    p[0] = 0xAB;
    p[expert_size - 1] = 0xCD;
    EXPECT_EQ(p[0], 0xAB);
    EXPECT_EQ(p[expert_size - 1], 0xCD);
    mgr.free(buf);
}

TEST(NumaAlloc, WriteReadRoundtrip) {
    lm::NumaManager mgr(standard_hw());
    constexpr size_t sz = 4096;
    auto buf = mgr.allocate_on_node(sz, 0);
    auto* p = static_cast<uint8_t*>(buf.data);
    for (size_t i = 0; i < sz; ++i)
        p[i] = static_cast<uint8_t>(i & 0xFF);
    for (size_t i = 0; i < sz; ++i)
        EXPECT_EQ(p[i], static_cast<uint8_t>(i & 0xFF)) << "mismatch at byte " << i;
    mgr.free(buf);
}

TEST(NumaAlloc, ZeroSizeRoundsToOnePage) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_on_node(0, 0);
    EXPECT_NE(buf.data, nullptr);
    EXPECT_EQ(buf.size, page_size());
    mgr.free(buf);
}

TEST(NumaAlloc, FreeNullIsNoop) {
    lm::NumaManager mgr(standard_hw());
    lm::NumaBuffer buf;  // default-constructed, null
    mgr.free(buf);  // should not crash
    EXPECT_EQ(buf.data, nullptr);
}

TEST(NumaAlloc, FreeResetsBuffer) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_on_node(4096, 0);
    EXPECT_NE(buf.data, nullptr);
    mgr.free(buf);
    EXPECT_EQ(buf.data, nullptr);
    EXPECT_EQ(buf.size, 0u);
    EXPECT_EQ(buf.numa_node, -1);
}

// ── Migration ────────────────────────────────────────────────────────────────

TEST(NumaMigrate, ChangesNodePreservesData) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_on_node(4096, 0);
    auto* p = static_cast<uint8_t*>(buf.data);
    for (size_t i = 0; i < 4096; ++i)
        p[i] = static_cast<uint8_t>(i & 0xFF);

    mgr.migrate(buf, 1);
    EXPECT_EQ(buf.numa_node, 1);
    EXPECT_NE(buf.data, nullptr);
    p = static_cast<uint8_t*>(buf.data);
    for (size_t i = 0; i < 4096; ++i)
        EXPECT_EQ(p[i], static_cast<uint8_t>(i & 0xFF));
    mgr.free(buf);
}

TEST(NumaMigrate, SameNodeIsNoop) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_on_node(4096, 0);
    void* orig_ptr = buf.data;
    mgr.migrate(buf, 0);
    EXPECT_EQ(buf.data, orig_ptr);  // pointer unchanged
    EXPECT_EQ(buf.numa_node, 0);
    mgr.free(buf);
}

TEST(NumaMigrate, NullBufferThrows) {
    lm::NumaManager mgr(standard_hw());
    lm::NumaBuffer buf;
    EXPECT_THROW(mgr.migrate(buf, 1), std::invalid_argument);
}

TEST(NumaMigrate, LargeBufferDataIntact) {
    lm::NumaManager mgr(standard_hw());
    constexpr size_t sz = 18 * 1024 * 1024;  // 18 MB
    auto buf = mgr.allocate_on_node(sz, 0);
    auto* p = static_cast<uint8_t*>(buf.data);
    // Fill with a recognisable pattern.
    for (size_t i = 0; i < sz; i += 4096)
        p[i] = static_cast<uint8_t>((i / 4096) & 0xFF);

    mgr.migrate(buf, 1);
    p = static_cast<uint8_t*>(buf.data);
    for (size_t i = 0; i < sz; i += 4096)
        EXPECT_EQ(p[i], static_cast<uint8_t>((i / 4096) & 0xFF));
    mgr.free(buf);
}

// ── Statistics ───────────────────────────────────────────────────────────────

TEST(NumaStats, InitiallyZero) {
    lm::NumaManager mgr(standard_hw());
    EXPECT_EQ(mgr.total_allocated_bytes(), 0u);
    EXPECT_EQ(mgr.allocated_bytes_on_node(0), 0u);
    EXPECT_EQ(mgr.allocated_bytes_on_node(1), 0u);
    EXPECT_EQ(mgr.allocated_bytes_on_node(-1), 0u);
}

TEST(NumaStats, AfterAllocation) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_on_node(4096, 0);
    EXPECT_GE(mgr.total_allocated_bytes(), 4096u);
    EXPECT_GE(mgr.allocated_bytes_on_node(0), 4096u);
    mgr.free(buf);
}

TEST(NumaStats, AfterFree) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_on_node(4096, 0);
    mgr.free(buf);
    EXPECT_EQ(mgr.total_allocated_bytes(), 0u);
    EXPECT_EQ(mgr.allocated_bytes_on_node(0), 0u);
}

TEST(NumaStats, PerNodeTracking) {
    lm::NumaManager mgr(standard_hw());
    auto b0 = mgr.allocate_on_node(4096, 0);
    auto b1 = mgr.allocate_on_node(8192, 1);
    EXPECT_GE(mgr.allocated_bytes_on_node(0), 4096u);
    EXPECT_GE(mgr.allocated_bytes_on_node(1), 8192u);
    mgr.free(b0);
    mgr.free(b1);
}

TEST(NumaStats, InterleavedTrackedSeparately) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_interleaved(4096);
    EXPECT_GE(mgr.allocated_bytes_on_node(-1), 4096u);
    EXPECT_EQ(mgr.allocated_bytes_on_node(0), 0u);
    mgr.free(buf);
}

TEST(NumaStats, MultipleAllocationsAccumulate) {
    lm::NumaManager mgr(standard_hw());
    auto b1 = mgr.allocate_on_node(4096, 0);
    auto b2 = mgr.allocate_on_node(4096, 0);
    EXPECT_GE(mgr.total_allocated_bytes(), 8192u);
    EXPECT_GE(mgr.allocated_bytes_on_node(0), 8192u);
    mgr.free(b1);
    mgr.free(b2);
}

TEST(NumaStats, MigrateUpdatesPerNode) {
    lm::NumaManager mgr(standard_hw());
    auto buf = mgr.allocate_on_node(4096, 0);
    size_t alloc_size = buf.size;
    EXPECT_EQ(mgr.allocated_bytes_on_node(0), alloc_size);
    EXPECT_EQ(mgr.allocated_bytes_on_node(1), 0u);

    mgr.migrate(buf, 1);
    EXPECT_EQ(mgr.allocated_bytes_on_node(0), 0u);
    EXPECT_EQ(mgr.allocated_bytes_on_node(1), alloc_size);
    EXPECT_EQ(mgr.total_allocated_bytes(), alloc_size);
    mgr.free(buf);
}

TEST(NumaStats, UnknownNodeReturnsZero) {
    lm::NumaManager mgr(standard_hw());
    EXPECT_EQ(mgr.allocated_bytes_on_node(42), 0u);
}

// ── HBM node detection (TD-NUMA-HBM-BANKS / INV-NUMA-HOSTBANK) ───────────────
//
// Ground truth is read INDEPENDENTLY from the kernel's node-state masks
// (/sys/devices/system/node/{has_memory,has_cpu,has_normal_memory}), not through
// libnuma, so the tests cross-check the NumaManager detection path rather than
// mirror it. On boxes without CPU-less memory-only nodes the detection tests
// GTEST_SKIP (the consistency tests still run).

namespace {

// Parse a kernel node-list string ("0-3,5") into a set of node ids.
std::set<int> parse_mask(const std::string& s) {
    std::set<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        const auto dash = tok.find('-');
        const int lo = std::atoi(tok.c_str());
        const int hi = (dash == std::string::npos)
                           ? lo : std::atoi(tok.c_str() + dash + 1);
        for (int n = lo; n <= hi; ++n) out.insert(n);
    }
    return out;
}

std::set<int> read_mask(const char* name) {
    std::ifstream f(std::string("/sys/devices/system/node/") + name);
    std::string line;
    if (!f || !std::getline(f, line)) return {};
    return parse_mask(line);
}

// SLIT distances for `node` (distance to node i at index i), or empty.
std::vector<int> read_distances(int node) {
    std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) +
                    "/distance");
    std::vector<int> d;
    int v;
    while (f >> v) d.push_back(v);
    return d;
}

// HMAT access0 initiator node ids for `node` (empty when no HMAT export).
std::set<int> read_initiators(int node) {
    std::set<int> out;
    const std::string path = "/sys/devices/system/node/node" +
                             std::to_string(node) + "/access0/initiators";
    DIR* dir = ::opendir(path.c_str());
    if (!dir) return out;
    while (struct dirent* de = ::readdir(dir)) {
        int init = -1;
        if (std::sscanf(de->d_name, "node%d", &init) == 1) out.insert(init);
    }
    ::closedir(dir);
    return out;
}

// Expected HBM banks: normal-memory nodes without CPUs.
std::set<int> expected_hbm_nodes() {
    const std::set<int> mem = read_mask("has_normal_memory");
    const std::set<int> cpu = read_mask("has_cpu");
    std::set<int> out;
    for (int n : mem)
        if (!cpu.count(n)) out.insert(n);
    return out;
}

}  // namespace

#ifdef LAYERSTORM_HBM_NODES

TEST(NumaHbm, DetectionMatchesKernelNodeMasks) {
    lm::NumaManager mgr(standard_hw());
    if (!mgr.numa_available()) GTEST_SKIP() << "libnuma unavailable";
    const std::set<int> expected = expected_hbm_nodes();
    if (expected.empty())
        GTEST_SKIP() << "no CPU-less memory-only NUMA nodes on this box";

    std::set<int> detected;
    for (const auto& h : mgr.hbm_nodes()) detected.insert(h.node);
    EXPECT_EQ(detected, expected);
    for (const auto& h : mgr.hbm_nodes()) {
        EXPECT_TRUE(mgr.node_is_hbm(h.node));
        EXPECT_GT(h.total_bytes, 0u) << "node " << h.node;
    }
}

TEST(NumaHbm, CpuAffinityIsNearestCpuNode) {
    lm::NumaManager mgr(standard_hw());
    if (!mgr.numa_available()) GTEST_SKIP() << "libnuma unavailable";
    if (mgr.hbm_nodes().empty())
        GTEST_SKIP() << "no CPU-less memory-only NUMA nodes on this box";
    const std::set<int> cpu_nodes = read_mask("has_cpu");
    ASSERT_FALSE(cpu_nodes.empty());

    for (const auto& h : mgr.hbm_nodes()) {
        const int aff = h.cpu_affinity_node;
        ASSERT_GE(aff, 0) << "node " << h.node;
        EXPECT_TRUE(cpu_nodes.count(aff))
            << "affinity node " << aff << " of HBM node " << h.node
            << " has no CPUs";
        EXPECT_EQ(mgr.hbm_cpu_affinity_node(h.node), aff);

        const std::set<int> initiators = read_initiators(h.node);
        if (!initiators.empty()) {
            // HMAT present: the affinity node must be a declared initiator.
            EXPECT_TRUE(initiators.count(aff))
                << "HBM node " << h.node << ": affinity " << aff
                << " not an access0 initiator";
        } else {
            // No HMAT: must be the min-SLIT-distance CPU node.
            const std::vector<int> dist = read_distances(h.node);
            ASSERT_FALSE(dist.empty());
            int best = -1;
            for (int c : cpu_nodes)
                if (c < static_cast<int>(dist.size()) &&
                    (best < 0 || dist[c] < dist[best]))
                    best = c;
            EXPECT_EQ(aff, best) << "HBM node " << h.node;
        }
    }
}

#else  // !LAYERSTORM_HBM_NODES

TEST(NumaHbm, GateOffYieldsNoHbmNodes) {
    lm::NumaManager mgr(standard_hw());
    EXPECT_TRUE(mgr.hbm_nodes().empty());
    EXPECT_EQ(mgr.all_banks_including_hbm(), mgr.all_memory_nodes());
}

#endif  // LAYERSTORM_HBM_NODES

TEST(NumaHbm, HostBankListsAreConsistent) {
    lm::NumaManager mgr(standard_hw());
    const std::vector<int> ddr = mgr.all_memory_nodes();
    const std::vector<int> all = mgr.all_banks_including_hbm();

    // all_memory_nodes() must never contain a detected HBM node…
    for (int n : ddr) {
        EXPECT_FALSE(mgr.node_is_hbm(n)) << "node " << n;
        EXPECT_EQ(mgr.hbm_cpu_affinity_node(n), -1) << "node " << n;
    }
    // …and the full bank set is exactly the sorted union of DDR + HBM.
    std::set<int> expected(ddr.begin(), ddr.end());
    for (const auto& h : mgr.hbm_nodes()) expected.insert(h.node);
    EXPECT_TRUE(std::is_sorted(all.begin(), all.end()));
    EXPECT_EQ(std::set<int>(all.begin(), all.end()), expected);
    EXPECT_EQ(all.size(), expected.size());  // no duplicates
    // Unknown node queries stay benign.
    EXPECT_FALSE(mgr.node_is_hbm(4242));
    EXPECT_EQ(mgr.hbm_cpu_affinity_node(4242), -1);
}

TEST(NumaHbm, ScopedThreadNodeBindPinsAndRestores) {
    lm::NumaManager mgr(standard_hw());
    cpu_set_t before;
    CPU_ZERO(&before);
    ASSERT_EQ(sched_getaffinity(0, sizeof(before), &before), 0);

    {   // node -1 → no-op scope, affinity untouched inside.
        lm::ScopedThreadNodeBind noop(mgr, -1);
        cpu_set_t inside;
        CPU_ZERO(&inside);
        ASSERT_EQ(sched_getaffinity(0, sizeof(inside), &inside), 0);
        EXPECT_TRUE(CPU_EQUAL(&before, &inside));
    }

    if (mgr.numa_available()) {
        const std::set<int> cpu_nodes = read_mask("has_cpu");
        if (!cpu_nodes.empty()) {
            const int target = *cpu_nodes.begin();
            {
                lm::ScopedThreadNodeBind pin(mgr, target);
                cpu_set_t inside;
                CPU_ZERO(&inside);
                ASSERT_EQ(sched_getaffinity(0, sizeof(inside), &inside), 0);
                EXPECT_GT(CPU_COUNT(&inside), 0);
            }
            cpu_set_t after;
            CPU_ZERO(&after);
            ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
            EXPECT_TRUE(CPU_EQUAL(&before, &after))
                << "affinity not restored after ScopedThreadNodeBind";
        }
    }
}

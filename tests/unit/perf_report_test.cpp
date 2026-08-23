#include <gtest/gtest.h>

#include <string>

#include "core/perf_report.h"
#include "core/perf_trace.h"

namespace pr = layerstorm::perf_report;
namespace pt = layerstorm::perf_trace;

TEST(PerfReport, TableRendersCoarseIndentedSectionStatsAndGrid) {
    pr::Report r;
    r.title = "X-RAY";
    r.row("attention", 0, 35.0, 17.1, 0.58, 61.0);
    r.section("daemon MoE phases");
    r.row("fetch→finalize wait", 1, 120.0, 58.0, 2.07, 58.0);
    r.stat("tok/s", "4.847");
    pr::Grid& g = r.grid("VRAM per TP GPU (MiB)", {"region", "GPU0", "GPU1"});
    g.add({"pinned weights", "13586", "13586"});

    const std::string t = pr::render_table(r);
    EXPECT_NE(t.find("attention"), std::string::npos);
    EXPECT_NE(t.find("  fetch→finalize wait"), std::string::npos);  // indented sub-row
    EXPECT_NE(t.find("── daemon MoE phases ──"), std::string::npos); // section divider
    EXPECT_NE(t.find("── stats ──"), std::string::npos);
    EXPECT_NE(t.find("tok/s"), std::string::npos);
    EXPECT_NE(t.find("VRAM per TP GPU (MiB)"), std::string::npos);
    EXPECT_NE(t.find("pinned weights"), std::string::npos);
}

TEST(PerfReport, JsonHasRowsStatsGridsAndNullBlanks) {
    pr::Report r;
    r.title = "X-RAY";
    r.row("a", 0, 1.0, 2.0, -1.0, 1.0);  // ms_per_op < 0 → null in JSON
    r.stat("k", "v");
    pr::Grid& g = r.grid("G", {"c0", "c1"});
    g.add({"x", "y"});

    const std::string j = pr::render_json(r);
    EXPECT_NE(j.find("\"rows\""), std::string::npos);
    EXPECT_NE(j.find("\"ms_per_op\": null"), std::string::npos);
    EXPECT_NE(j.find("\"stats\""), std::string::npos);
    EXPECT_NE(j.find("\"k\": \"v\""), std::string::npos);
    EXPECT_NE(j.find("\"grids\""), std::string::npos);
    EXPECT_NE(j.find("\"c0\""), std::string::npos);
}

// perf_trace is compiled in by default (LAYERSTORM_PERF_TRACE=ON). aggregate()
// pairs the four MoE phase markers per cmd_seq and sums the kDmaGpu token (an
// elapsed-ns value, so its total is deterministic; the phase deltas use the real
// clock, so only their counts are asserted).
TEST(PerfTrace, AggregatePairsMoePhasesAndSumsDma) {
    pt::init(1u << 16);
    ASSERT_TRUE(pt::enabled());
    // cmd 1001: fetched on 2 GPUs — arrivals between fetch_issued and finalize.
    pt::record(pt::kMoeEnter,         0, 1001, 0, 0);
    pt::record(pt::kMoeFetchIssued,   0, 1001, 0, 0);
    pt::record(pt::kExpertArrived,    0, 1001, 0, 0);  // GPU0 (fastest)
    pt::record(pt::kExpertArrived,    1, 1001, 0, 0);  // GPU1 (straggler, later)
    pt::record(pt::kMoeFinalizeEnter, 0, 1001, 0, 0);
    pt::record(pt::kMoeFinalizeExit,  0, 1001, 0, 0);
    // cmd 1002: no fetches (all hits) → fetch-wall decomposition NOT counted.
    pt::record(pt::kMoeEnter,         0, 1002, 0, 0);
    pt::record(pt::kMoeFetchIssued,   0, 1002, 0, 0);
    pt::record(pt::kMoeFinalizeEnter, 0, 1002, 0, 0);
    pt::record(pt::kMoeFinalizeExit,  0, 1002, 0, 0);
    pt::record(pt::kDmaGpu, 0, 0, 0, /*elapsed ns=*/500000);  // 0.5 ms
    pt::record(pt::kComputeGpu, 0, 0, 0, /*elapsed ns=*/300000);  // 0.3 ms
    pt::record(pt::kComputeGpu, 1, 0, 0, /*elapsed ns=*/200000);  // 0.2 ms

    const pt::Aggregate a = pt::aggregate();
    ASSERT_TRUE(a.available);
    EXPECT_EQ(a.moe_fetch.count, 2u);
    EXPECT_EQ(a.moe_wait.count, 2u);
    EXPECT_EQ(a.moe_compute.count, 2u);
    EXPECT_EQ(a.moe_total.count, 2u);
    EXPECT_GE(a.moe_total.total_ms, 0.0);
    EXPECT_EQ(a.dma_gpu.count, 1u);
    EXPECT_DOUBLE_EQ(a.dma_gpu.total_ms, 0.5);
    EXPECT_EQ(a.compute_gpu.count, 2u);                 // 2 kComputeGpu events
    EXPECT_DOUBLE_EQ(a.compute_gpu.total_ms, 0.5);      // 0.3 + 0.2 ms
    // Fetch-wall decomposition: only cmd 1001 fetched → count 1.
    EXPECT_EQ(a.fetch_wall.count, 1u);
    EXPECT_EQ(a.fetch_fastest.count, 1u);
    EXPECT_EQ(a.fetch_dead.count, 1u);
    EXPECT_GT(a.fetch_wall.total_ms, 0.0);   // arrivals recorded after fetch_issued
    EXPECT_GE(a.fetch_dead.total_ms, 0.0);   // straggler ≥ fastest
    // fastest + dead == wall (the decomposition is exact).
    EXPECT_NEAR(a.fetch_fastest.total_ms + a.fetch_dead.total_ms,
                a.fetch_wall.total_ms, 1e-6);
}

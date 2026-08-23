// REEF orchestrator decision stack (src/core/gpu_loader/reef_orch.h) — CPU
// unit tests for the extracted library: route determinism, hit-pinning to the
// resident device, and the apply victim bookkeeping (13c-2.0 map semantics)
// the test-side implementation guaranteed before extraction.

#include "core/gpu_loader/reef_orch.h"

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace gl = layerstorm::gpu_loader;
using layerstorm::memory::ExpertKey;

namespace {

// Minimal 2-device / 2-bank constants (mirrors gpu_loader_solver_test's
// make_constants shape).
gl::LoaderConstants make_constants(int M) {
    gl::LoaderConstants k;
    k.source       = "reef-test";
    k.expert_bytes = 24772992.0;
    k.num_devices  = M;
    k.num_banks    = M;
    k.ncf          = {0.0, 1.0, 1.2, 1.5};
    for (int d = 0; d < M; ++d) {
        gl::DeviceConstants dc;
        dc.position    = d;
        dc.numa_node   = d;
        dc.xfer_lat_us = 5.0;
        dc.compute     = {150.0, 0.0, 1};
        k.devices.push_back(dc);
    }
    for (int b = 0; b < M; ++b) {
        gl::BankConstants bc;
        bc.node       = b;
        bc.egress_us  = 50.0;
        bc.contention = 1.0;
        k.banks.push_back(bc);
    }
    k.matrix.assign(M, std::vector<gl::TransferCell>(M));
    for (int b = 0; b < M; ++b)
        for (int d = 0; d < M; ++d)
            k.matrix[b][d] = gl::TransferCell{500.0, (b == d ? 1 : 2), 5.0};
    return k;
}

std::vector<uint8_t> route(gl::ReefOrch& o, int layer,
                           const std::vector<uint16_t>& topk) {
    std::vector<uint8_t> assign(topk.size());
    gl::reef_orch_route(o, layer, topk, assign);
    return assign;
}

}  // namespace

TEST(ReefOrchTest, RouteIsDeterministicAndBalances) {
    gl::ReefOrch a(2, {8, 8}, make_constants(2));
    gl::ReefOrch b(2, {8, 8}, make_constants(2));
    const std::vector<uint16_t> topk{0, 1, 2, 3, 4, 5, 6, 7};
    const auto ra = route(a, 3, topk);
    const auto rb = route(b, 3, topk);
    EXPECT_EQ(ra, rb) << "identical state must produce identical placement";
    int c0 = 0;
    for (uint8_t g : ra) c0 += (g == 0);
    EXPECT_GT(c0, 0);
    EXPECT_LT(c0, static_cast<int>(ra.size()))
        << "uncached union must spread across both devices";
}

TEST(ReefOrchTest, HitsRideResidency) {
    gl::ReefOrch o(2, {8, 8}, make_constants(2));
    const std::vector<uint16_t> topk{0, 1, 2, 3};
    // Make expert 2 resident on device 1 (board model).
    o.board.update(1, ExpertKey{3, 2}, o.board.recency_now());
    const auto r = route(o, 3, topk);
    EXPECT_EQ(r[2], 1) << "a resident expert must be pinned to its device";
}

TEST(ReefOrchTest, ApplyFillsVictimMapAndAdmitsMisses) {
    gl::ReefOrch o(2, {2, 2}, make_constants(2));  // tiny caps → eviction
    // Pre-fill device 0 to capacity with old residents (layer 9).
    o.board.advance_recency();
    o.board.update(0, ExpertKey{9, 100}, o.board.recency_now());
    o.board.advance_recency();
    o.board.update(0, ExpertKey{9, 101}, o.board.recency_now());
    // Two layer-3 misses on device 0 → both need victims.
    std::vector<gl::ReefEntry> entries{
        {3, 0, 0, 0},
        {3, 1, 0, 0},
    };
    std::vector<gl::ReefVictim> evicts{
        {3, 0xFFFF, 0, 0},
        {3, 0xFFFF, 0, 0},
    };
    gl::reef_orch_apply(o, 3, entries.data(), evicts.data(), 2);
    // Both sentinels replaced by the two old residents, cheapest-first.
    EXPECT_EQ(evicts[0].layer_idx, 9u);
    EXPECT_EQ(evicts[0].expert_idx, 100);
    EXPECT_EQ(evicts[1].expert_idx, 101);
    EXPECT_EQ(evicts[0].gpu_idx, 0);
    // Board: old residents gone, misses admitted.
    EXPECT_FALSE(o.board.is_resident(0, ExpertKey{9, 100}));
    EXPECT_FALSE(o.board.is_resident(0, ExpertKey{9, 101}));
    EXPECT_TRUE(o.board.is_resident(0, ExpertKey{3, 0}));
    EXPECT_TRUE(o.board.is_resident(0, ExpertKey{3, 1}));
    EXPECT_EQ(o.board.resident_count(0), 2);
}

TEST(ReefOrchTest, ApplyNeverEvictsNeededNow) {
    gl::ReefOrch o(1, {2}, make_constants(1));
    // Device 0 full with layer-3 experts 0 and 1; the new union routes
    // expert 0 again (hit) + expert 5 (miss) — expert 0 is needed_now and
    // must NOT be the victim even if cheapest.
    o.board.advance_recency();
    o.board.update(0, ExpertKey{3, 0}, o.board.recency_now());
    o.board.advance_recency();
    o.board.update(0, ExpertKey{3, 1}, o.board.recency_now());
    std::vector<gl::ReefEntry> entries{
        {3, 0, 0, 0},
        {3, 5, 0, 0},
    };
    std::vector<gl::ReefVictim> evicts{
        {3, 0xFFFF, 0, 0},
        {3, 0xFFFF, 0, 0},
    };
    gl::reef_orch_apply(o, 3, entries.data(), evicts.data(), 2);
    EXPECT_EQ(evicts[0].expert_idx, 0xFFFF) << "hit entry keeps the sentinel";
    EXPECT_EQ(evicts[1].expert_idx, 1) << "victim = the non-needed resident";
    EXPECT_TRUE(o.board.is_resident(0, ExpertKey{3, 0}));
    EXPECT_TRUE(o.board.is_resident(0, ExpertKey{3, 5}));
}

TEST(ReefOrchTest, BankNodeSeamDefaultsToBankZero) {
    gl::ReefOrch o(2, {8, 8}, make_constants(2));
    // Null seam → node -1 → no bank matches → bank 0 (the pre-extraction
    // "no arena map" path). With the seam installed, bank follows the node.
    const std::vector<uint16_t> topk{0, 1};
    (void)route(o, 3, topk);
    EXPECT_EQ(o.req.bank_of[0], 0);
    o.bank_node_fn = [](uint32_t, uint16_t e) { return e == 1 ? 1 : 0; };
    (void)route(o, 3, topk);
    EXPECT_EQ(o.req.bank_of[0], 0);
    EXPECT_EQ(o.req.bank_of[1], 1);
}

TEST(ReefOrchTest, MakeThrowsOnMissingCalibration) {
    EXPECT_THROW(gl::make_reef_orch("/nonexistent/calib.json", 2, {8, 8}),
                 std::exception);
}

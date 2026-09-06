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

// ── TD-MOE-PLACEMENT-CAPACITY-CAP: per-layer route capacity caps ────────────
// The solver must never assign a capped (wave-excluded / EP-XTP) position
// more experts for one layer than its stable zone can physically hold; the
// repair is a post-solve pass, byte-identical when no cap binds.

namespace {

std::vector<int> per_pos_counts(const std::vector<uint8_t>& assign, int M) {
    std::vector<int> cnt(static_cast<size_t>(M), 0);
    for (uint8_t g : assign) {
        EXPECT_LT(g, M);
        ++cnt[g];
    }
    return cnt;
}

}  // namespace

TEST(ReefOrchTest, RouteCapNonBindingIsByteIdentical) {
    // Same state, one with generous caps installed, one without: the repair
    // must be a pure no-op (identical assignments over several rounds).
    gl::ReefOrch a(2, {8, 8}, make_constants(2));
    gl::ReefOrch b(2, {8, 8}, make_constants(2));
    b.route_cap = {8, 8};  // == stable capacity; never binds for n=8 splits
    for (int layer = 3; layer < 6; ++layer) {
        const std::vector<uint16_t> topk{0, 1, 2, 3, 4, 5, 6, 7};
        const auto ra = route(a, layer, topk);
        const auto rb = route(b, layer, topk);
        EXPECT_EQ(ra, rb) << "non-binding caps must not change placement "
                             "(layer " << layer << ")";
    }
    EXPECT_EQ(b.route_cap_moves, 0u);
    EXPECT_EQ(b.route_cap_residual, 0u);
}

TEST(ReefOrchTest, RouteCapRefusesOverCapacityShare) {
    // Device 0's stable zone holds ONE expert and cannot wave-stream
    // (route_cap installed = the measured stable capacity). A 4-expert cold
    // union must NOT hand device 0 more than 1 expert — the synthetic
    // V4-gpu2 shape (63-slot zone, 64-expert cold share) in miniature.
    gl::ReefOrch o(2, {1, 8}, make_constants(2));
    o.route_cap = {1, -1};
    const std::vector<uint16_t> topk{0, 1, 2, 3};
    const auto r = route(o, 3, topk);
    const auto cnt = per_pos_counts(r, 2);
    EXPECT_LE(cnt[0], 1) << "capped position must never exceed its stable "
                            "capacity for one layer's share";
    EXPECT_EQ(cnt[0] + cnt[1], 4) << "every expert stays assigned";
    EXPECT_GT(o.route_cap_moves, 0u) << "the symmetric solve gives device 0 "
                                        ">1 of 4, so the repair must move";
    EXPECT_EQ(o.route_cap_residual, 0u);

    // The board model stays capacity-consistent through apply: no
    // over-admission on the capped device (pre-fix the board overfilled,
    // mirroring the engine's reserve-fail degrade).
    std::vector<gl::ReefEntry> entries(4);
    std::vector<gl::ReefVictim> evicts(4);
    for (int i = 0; i < 4; ++i) {
        entries[static_cast<size_t>(i)] =
            {3, topk[static_cast<size_t>(i)], 0, r[static_cast<size_t>(i)]};
        evicts[static_cast<size_t>(i)] = {3, 0xFFFF, r[static_cast<size_t>(i)], 0};
    }
    gl::reef_orch_apply(o, 3, entries.data(), evicts.data(), 4);
    EXPECT_LE(o.board.resident_count(0), 1);
}

TEST(ReefOrchTest, RouteCapMovesMissesNeverResidentHits) {
    // Device 0 full with this layer's residents (hits). Cap == capacity: the
    // hits must ride residency untouched; every miss lands elsewhere.
    gl::ReefOrch o(2, {2, 8}, make_constants(2));
    o.route_cap = {2, -1};
    o.board.advance_recency();
    o.board.update(0, ExpertKey{3, 0}, o.board.recency_now());
    o.board.update(0, ExpertKey{3, 1}, o.board.recency_now());
    const std::vector<uint16_t> topk{0, 1, 2, 3, 4, 5};
    const auto r = route(o, 3, topk);
    EXPECT_EQ(r[0], 0) << "resident hit keeps its device";
    EXPECT_EQ(r[1], 0) << "resident hit keeps its device";
    const auto cnt = per_pos_counts(r, 2);
    EXPECT_EQ(cnt[0], 2) << "capped device carries exactly its hits";
    EXPECT_EQ(cnt[1], 4) << "all misses re-homed to the headroom device";
    EXPECT_EQ(o.route_cap_residual, 0u);
}

TEST(ReefOrchTest, RouteCapLargeUnionBoundedGreedyPath) {
    // n > kMaxExperts exercises the LoaderSolver256 tier — the production
    // shape of the measured degrade (near-full prefill union on hash
    // layers). The capped position must respect its zone capacity.
    gl::ReefOrch o(2, {40, 512}, make_constants(2));
    o.route_cap = {40, -1};
    std::vector<uint16_t> topk(96);
    for (int i = 0; i < 96; ++i) topk[static_cast<size_t>(i)] =
        static_cast<uint16_t>(i);
    const auto r = route(o, 3, topk);
    const auto cnt = per_pos_counts(r, 2);
    EXPECT_LE(cnt[0], 40);
    EXPECT_EQ(cnt[0] + cnt[1], 96);
    EXPECT_EQ(o.route_cap_residual, 0u);
}

TEST(ReefOrchTest, RouteCapResidualWhenNoHeadroom) {
    // Physically infeasible share (every position capped full): the repair
    // must refuse to livelock — it leaves the residual, counts it, and the
    // route still returns a complete assignment (degrade + serving retry
    // remain the net for this case).
    gl::ReefOrch o(2, {1, 1}, make_constants(2));
    o.route_cap = {1, 1};
    const std::vector<uint16_t> topk{0, 1, 2, 3};
    const auto r = route(o, 3, topk);
    EXPECT_EQ(r.size(), 4u);
    const auto cnt = per_pos_counts(r, 2);
    EXPECT_EQ(cnt[0] + cnt[1], 4);
    EXPECT_GT(o.route_cap_residual, 0u);
}

// ── TD-KVXP-CAPACITY-REPUBLISH: live cap refresh ───────────────────────────

TEST(ReefOrchTest, RefreshCapsUpdatesCapAndRearmsRouteCap) {
    gl::ReefOrch o(2, {8, 8}, make_constants(2));
    // XTP route caps armed on position 1 (the boot arming rule:
    // route_cap[pos] = cap[pos] for XTP positions, -1 elsewhere).
    o.route_cap = {-1, 8};
    gl::reef_orch_refresh_caps(o, {8, 12}, /*xtp_positions=*/{1});
    EXPECT_EQ(o.cap, (std::vector<int>{8, 12}));
    EXPECT_EQ(o.route_cap[0], -1) << "non-XTP position stays uncapped";
    EXPECT_EQ(o.route_cap[1], 12) << "XTP route cap re-armed from the "
                                     "refreshed capacity";
    // Shrink direction (a 44z drain): same rule.
    gl::reef_orch_refresh_caps(o, {8, 5}, {1});
    EXPECT_EQ(o.cap[1], 5);
    EXPECT_EQ(o.route_cap[1], 5);
}

TEST(ReefOrchTest, RefreshCapsLeavesDisarmedRouteCapEmpty) {
    // LS_REEF_ROUTE_CAP=0 diagnostic / no XTP ranks: route_cap is empty and
    // a refresh must not arm it.
    gl::ReefOrch o(2, {8, 8}, make_constants(2));
    ASSERT_TRUE(o.route_cap.empty());
    gl::reef_orch_refresh_caps(o, {8, 12}, {1});
    EXPECT_EQ(o.cap, (std::vector<int>{8, 12}));
    EXPECT_TRUE(o.route_cap.empty()) << "disarmed route caps stay disarmed";
}

TEST(ReefOrchTest, RefreshCapsShapeMismatchIgnored) {
    gl::ReefOrch o(2, {8, 8}, make_constants(2));
    gl::reef_orch_refresh_caps(o, {8, 12, 4}, {});
    EXPECT_EQ(o.cap, (std::vector<int>{8, 8}))
        << "wrong-shaped refresh must never corrupt the model";
}

TEST(ReefOrchTest, RefreshedCapIsConsumedByApply) {
    // The refresh must reach the DECISIONS, not just the stored vector:
    // a full device stops evicting once a grant raises its cap.
    gl::ReefOrch o(1, {2}, make_constants(1));
    o.board.advance_recency();
    o.board.update(0, ExpertKey{9, 100}, o.board.recency_now());
    o.board.advance_recency();
    o.board.update(0, ExpertKey{9, 101}, o.board.recency_now());
    // At cap 2 a new miss needs a victim…
    std::vector<gl::ReefEntry> e1{{3, 0, 0, 0}};
    std::vector<gl::ReefVictim> v1{{3, 0xFFFF, 0, 0}};
    gl::reef_orch_apply(o, 3, e1.data(), v1.data(), 1);
    EXPECT_NE(v1[0].expert_idx, 0xFFFF) << "at the old cap a victim is due";
    // …after a grant-driven refresh to cap 4, the next miss admits free.
    gl::reef_orch_refresh_caps(o, {4}, {});
    std::vector<gl::ReefEntry> e2{{3, 1, 0, 0}};
    std::vector<gl::ReefVictim> v2{{3, 0xFFFF, 0, 0}};
    gl::reef_orch_apply(o, 3, e2.data(), v2.data(), 1);
    EXPECT_EQ(v2[0].expert_idx, 0xFFFF)
        << "granted capacity must be usable without eviction";
    EXPECT_EQ(o.board.resident_count(0), 3);
}

TEST(ReefOrchTest, MakeThrowsOnMissingCalibration) {
    EXPECT_THROW(gl::make_reef_orch("/nonexistent/calib.json", 2, {8, 8}),
                 std::exception);
}

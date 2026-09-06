// TD-GLM5N-ROUTED-EXPERT-ID-TRUNCATION regression tests.
//
// The FAR seam's routed-union builder (dispatch_reef.cpp far_forward_layer,
// extracted to daemon/far_union.h) sized its dedup table at
// ipc::kMaxExperts == 256 and silently skipped expert ids >= 256 while
// glm5_next routes over 288 experts: experts 256-287 were never fetched,
// dispatched, or folded, on every token, with moe_degraded_layers == 0.
// These tests fail on that code: the static_assert fires at compile time
// on kMaxExperts < 288, and HighExpertIdsSurvive fails at runtime if the
// union thins ids >= 256.

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "daemon/far_union.h"
#include "daemon/ipc_protocol.h"

namespace ipc = layerstorm::ipc;
using layerstorm::daemon::build_routed_union;

// glm5_next (GLM-5.3-Flash) ships n_routed_experts = 288. Any IPC expert
// cap below that silently truncates the routed union (~11% routing mass).
static_assert(ipc::kMaxExperts >= 288,
              "ipc::kMaxExperts must cover glm5_next's 288 routed experts "
              "(TD-GLM5N-ROUTED-EXPERT-ID-TRUNCATION)");
// The sideband union capacity must admit a full single-layer union.
static_assert(ipc::kMaxExpertPrefetch >= 288,
              "kMaxExpertPrefetch must admit a full glm5_next routed union");

TEST(FarUnion, HighExpertIdsSurvive) {
    // A routing export whose top-K rows cover every glm5_next expert id,
    // high ids first — the old builder dropped all ids >= 256.
    constexpr int kNumExperts = 288;
    std::vector<int32_t> ridx(kNumExperts);
    for (int i = 0; i < kNumExperts; ++i)
        ridx[i] = kNumExperts - 1 - i;  // 287, 286, ..., 0

    std::vector<uint16_t> out;
    ASSERT_TRUE(build_routed_union(ridx.data(),
                                   static_cast<uint32_t>(ridx.size()),
                                   kNumExperts, out));
    ASSERT_EQ(out.size(), static_cast<size_t>(kNumExperts));
    // First-occurrence order preserved; ids 256-287 all present.
    EXPECT_EQ(out.front(), 287);
    std::vector<uint8_t> present(kNumExperts, 0);
    for (uint16_t e : out) present[e] = 1;
    for (int e = 256; e < kNumExperts; ++e)
        EXPECT_TRUE(present[e]) << "expert " << e << " dropped from union";
}

TEST(FarUnion, DedupAndInvalidSkip) {
    constexpr int kNumExperts = 288;
    const int32_t ridx[] = {5, 5, 287, -1, 288, 300, 287, 0, 256};
    std::vector<uint16_t> out;
    ASSERT_TRUE(build_routed_union(ridx, 9, kNumExperts, out));
    // 5, 287, 0, 256 — deduped, first-occurrence, oob/negative skipped.
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(out[0], 5);
    EXPECT_EQ(out[1], 287);
    EXPECT_EQ(out[2], 0);
    EXPECT_EQ(out[3], 256);
}

TEST(FarUnion, DecodeTopkUnionIsFullTopk) {
    // B=1 decode: 8 distinct routed ids straddling the old 256 boundary
    // must produce a union of exactly 8 (the shadow-dump mean was 7.07 =
    // 8 * 256/288 under the truncating builder).
    const int32_t ridx[] = {12, 250, 255, 256, 257, 270, 286, 287};
    std::vector<uint16_t> out;
    ASSERT_TRUE(build_routed_union(ridx, 8, 288, out));
    EXPECT_EQ(out.size(), 8u);
}

TEST(FarUnion, OverCapRefusesLoudly) {
    // num_experts beyond the IPC cap must refuse (caller writes an error),
    // never silently thin. Engine boot refuses such configs outright.
    std::vector<int32_t> ridx(8);
    std::iota(ridx.begin(), ridx.end(), 0);
    std::vector<uint16_t> out;
    EXPECT_FALSE(build_routed_union(ridx.data(), 8,
                                    ipc::kMaxExperts + 1, out));
    EXPECT_TRUE(out.empty());
}

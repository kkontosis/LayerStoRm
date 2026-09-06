// 44z stage 5 — health tripwire: TD-91d zero-fill guard counters.
//
// The live gate for this tripwire is the §4b degraded-discard rule on a real
// GPU run: a bitset-resident expert whose cache entry is missing at
// pointer-table fill time zeroes its contribution, moe_progressive.cpp diffs
// the counter around the dispatch, and the layer becomes moe_degraded.
//
// NAMED GAP (deliberate, see the commit body): that end-to-end path is NOT
// reachable from the null-backend CommandDispatcher harness in
// command_dispatcher_test.cpp. There, deps_.cuda_kernels_enabled is false, so
// dispatch_moe_internal never populates the residency bitset (total_resident
// stays 0), takes the skip_routed shortcut, and never reaches either
// pointer-table fill — so no guard can fire no matter how the test evicts an
// expert between the bitset snapshot and the dispatch. Building a fake
// resident-but-entryless expert would require a CUDA-backed ExpertCache (and
// edits to files this change may not touch), which the brief excludes.
//
// What IS locked here: the counter bank's existence, its zero default, the
// per-fill warn-once accounting, the all-GPU sum the progressive finalize
// diffs, and the out-of-range-gpu path — an unknown gpu id must never make the
// signal disappear, because a lost fire is a silently wrong token.

#include <gtest/gtest.h>

#include <cstdint>

#include "daemon/moe/moe_internal.h"

namespace ldam = layerstorm::daemon;

namespace {

// A gpu slot no other unit test drives (the null-backend dispatcher harness
// only ever uses gpu 0), so the default-zero assertion is meaningful.
constexpr int kProbeGpu = ldam::kZoneGuardMaxGpus - 1;

}  // namespace

TEST(MoeZoneGuardHealth, CounterDefaultsToZero) {
    EXPECT_EQ(ldam::zone_guard_zero_fills(kProbeGpu), 0)
        << "44z: the TD-91d guard counter must start at 0 — a nonzero default "
           "would degrade every layer on the first dispatch";
}

TEST(MoeZoneGuardHealth, FillSiteCountsEveryFireAndWarnsOnce) {
    const int64_t total_before = ldam::zone_guard_zero_fills_total();
    const int64_t gpu_before   = ldam::zone_guard_zero_fills(kProbeGpu);

    // Two guard fires inside ONE pointer-table fill, exactly as the three fill
    // lambdas drive it: fires_this_fill is the caller's local counter, so only
    // the first fire logs (rate limit) but BOTH are counted.
    int fires_this_fill = 0;
    ldam::zone_guard_zero_fill_hit(/*layer_idx=*/7, /*expert=*/3, kProbeGpu,
                                   fires_this_fill);
    EXPECT_EQ(fires_this_fill, 1) << "first fire in a fill must warn";
    ldam::zone_guard_zero_fill_hit(/*layer_idx=*/7, /*expert=*/4, kProbeGpu,
                                   fires_this_fill);
    EXPECT_EQ(fires_this_fill, 2) << "subsequent fires are counted silently";

    // The fill-done summary is the second half of the rate limit; it must not
    // touch the counters.
    ldam::zone_guard_zero_fill_fill_done(/*layer_idx=*/7, kProbeGpu,
                                         fires_this_fill);

    EXPECT_EQ(ldam::zone_guard_zero_fills(kProbeGpu) - gpu_before, 2);
    EXPECT_EQ(ldam::zone_guard_zero_fills_total() - total_before, 2)
        << "44z: the all-GPU sum is what the progressive finalize diffs around "
           "a dispatch — every fire must reach it";
}

TEST(MoeZoneGuardHealth, OutOfRangeGpuStillReachesTheTotal) {
    const int64_t total_before = ldam::zone_guard_zero_fills_total();

    ldam::zone_guard_note_zero_fill(-1);
    ldam::zone_guard_note_zero_fill(ldam::kZoneGuardMaxGpus);
    ldam::zone_guard_note_zero_fill(ldam::kZoneGuardMaxGpus + 1000);

    EXPECT_EQ(ldam::zone_guard_zero_fills_total() - total_before, 3)
        << "44z: an out-of-range gpu id must fold into the unknown-gpu bucket, "
           "never be dropped — a dropped fire is a silently zeroed expert "
           "contribution that no gate would ever see";
}

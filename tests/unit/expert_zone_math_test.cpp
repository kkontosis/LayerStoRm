#include <gtest/gtest.h>

#include <vector>

#include "core/memory/expert_zone_math.h"

using layerstorm::memory::ExpertZoneGeometry;
using layerstorm::memory::kExpertZoneSlotAlign;

namespace {

/// GLM-5.3-Flash (glm5_next): per_routed_expert_bytes / VramLayout::slab_bytes.
/// Ratio 18415616 / 272448 = 67.5931 — non-integral, hence the general equation.
ExpertZoneGeometry glm5_next() {
    return {
        .expert_slot_bytes = 18415616,
        .slab_bytes = 272448,
        .max_waste = 0.10,
    };
}

/// GLM-5.2 Q4_K_XL. Ratio 27623424 / 1086976 = 25.4131 — also non-integral.
ExpertZoneGeometry glm52() {
    return {
        .expert_slot_bytes = 27623424,
        .slab_bytes = 1086976,
        .max_waste = 0.10,
    };
}

}  // namespace

// ── 1. Served model: GLM-5.3-Flash ───────────────────────────────────────────

TEST(ExpertZoneMath, Glm5NextGeometry) {
    const ExpertZoneGeometry g = glm5_next();
    ASSERT_TRUE(g.valid());

    // 18415616 = 4496 * 4096, so align_up is a no-op and the ticket equation
    // holds verbatim (slot_stride == expert_slot_bytes).
    EXPECT_EQ(g.slot_stride(), 18415616);

    // capacity(67) = 67 * 272448 = 18254016 < 18415616 -> one slot does not fit.
    EXPECT_EQ(g.slots_in(67), 0);
    // capacity(68) = 68 * 272448 = 18526464 >= 18415616 -> exactly one slot.
    EXPECT_EQ(g.slots_in(68), 1);

    // ceil(18415616 / 272448) = ceil(67.5931) = 68.
    EXPECT_EQ(g.slabs_for_slots(1), 68);

    // waste = 18526464 - 18415616 = 110848.
    EXPECT_EQ(g.waste_bytes(68, 1), 110848);
    // 110848 / 18526464 = 0.0059832 -> 0.598%, far inside max_waste = 0.10.
    EXPECT_NEAR(g.waste_frac(68, 1), 0.00598, 1e-5);
    EXPECT_TRUE(g.admissible(68));

    // best_fit over a 500-slab run:
    //   N = floor(500 * 272448 / 18415616) = floor(136224000 / 18415616) = 7
    //   S = ceil(7 * 18415616 / 272448) = ceil(128909312 / 272448)
    //     = ceil(473.15) = 474  (<= 500)
    //   waste_frac(474, 7) = (129140352 - 128909312) / 129140352
    //                      = 231040 / 129140352 = 0.00179 <= 0.10 -> accepted
    //                        on the FIRST iteration of the descent.
    const ExpertZoneGeometry::Fit fit = g.best_fit(500);
    EXPECT_EQ(fit.num_slots, 7);
    EXPECT_EQ(fit.num_slabs, 474);
}

// ── 2. Served model: GLM-5.2 Q4_K_XL ─────────────────────────────────────────

TEST(ExpertZoneMath, Glm52Geometry) {
    const ExpertZoneGeometry g = glm52();
    ASSERT_TRUE(g.valid());

    // 27623424 = 6744 * 4096 -> already aligned.
    EXPECT_EQ(g.slot_stride(), 27623424);

    // ceil(27623424 / 1086976) = ceil(25.4131) = 26.
    EXPECT_EQ(g.slabs_for_slots(1), 26);

    // capacity(26) = 26 * 1086976 = 28261376; waste = 28261376 - 27623424.
    EXPECT_EQ(g.waste_bytes(26, 1), 637952);
    // 637952 / 28261376 = 0.022573 -> 2.257%, still inside max_waste = 0.10.
    EXPECT_NEAR(g.waste_frac(26, 1), 0.02257, 1e-4);
    EXPECT_TRUE(g.admissible(26));
}

// ── 3. The knob genuinely binds ──────────────────────────────────────────────

TEST(ExpertZoneMath, MaxWasteBinds) {
    ExpertZoneGeometry g = glm5_next();
    g.max_waste = 0.001;  // 0.1% — tighter than the 0.598% minimal-S fit.

    EXPECT_FALSE(g.admissible(68));

    // The descent finds nothing: the only candidate (N=1, S=68) wastes 0.598%.
    const ExpertZoneGeometry::Fit fit = g.best_fit(68);
    EXPECT_EQ(fit.num_slots, 0);
    EXPECT_EQ(fit.num_slabs, 0);
}

// ── 4. Alignment pad from a non-4096-aligned region base ─────────────────────

TEST(ExpertZoneMath, BaseMisalignmentPad) {
    const ExpertZoneGeometry g = glm5_next();
    // A granted run's base is only slab-granular (64 B on glm5_next).
    constexpr int64_t kMisalign = 64;

    // 4096 - 64 = 4032 bytes burned before slot 0.
    EXPECT_EQ(g.base_pad(kMisalign), 4032);

    // usable = 18526464 - 4032 = 18522432 >= 18415616 -> the slot still fits.
    EXPECT_EQ(g.slots_in(68, kMisalign), 1);
    // need = 4032 + 18415616 = 18419648; ceil(18419648 / 272448)
    //      = ceil(67.6079) = 68 — the pad does not push it to 69 here.
    EXPECT_EQ(g.slabs_for_slots(1, kMisalign), 68);
}

// ── 5. The descent loop below its first iteration ────────────────────────────

TEST(ExpertZoneMath, GeneralSearchDescends) {
    // Synthetic geometry chosen so ceil() overshoots max_waste at large N —
    // the future-model case the N-search exists for.
    const ExpertZoneGeometry g{
        .expert_slot_bytes = 53248,  // 13 * 4096
        .slab_bytes = 40960,         // 10 * 4096
        .max_waste = 0.05,
    };
    ASSERT_TRUE(g.valid());
    EXPECT_EQ(g.slot_stride(), 53248);

    // N=1: S = ceil(53248 / 40960) = 2; capacity 81920,
    //      waste = 81920 - 53248 = 28672; frac = 0.35 > 0.05 -> inadmissible.
    EXPECT_EQ(g.slabs_for_slots(1), 2);
    EXPECT_DOUBLE_EQ(g.waste_frac(2, 1), 0.35);
    EXPECT_FALSE(g.admissible(2));

    // N=3: S = ceil(3 * 53248 / 40960) = ceil(159744 / 40960) = ceil(3.9) = 4;
    //      capacity 163840, waste = 163840 - 159744 = 4096; frac = 0.025.
    EXPECT_EQ(g.slabs_for_slots(3), 4);
    EXPECT_DOUBLE_EQ(g.waste_frac(4, 3), 0.025);
    EXPECT_TRUE(g.admissible(4));

    // best_fit(7) must walk DOWN three iterations:
    //   slots_in(7) = floor((286720 - 53248) / 53248) + 1 = 4 + 1 = 5
    //   n=5 -> S=ceil(266240/40960)=7, frac = 20480/286720 = 0.0714 -> fail
    //   n=4 -> S=ceil(212992/40960)=6, frac = 32768/245760 = 0.1333 -> fail
    //   n=3 -> S=4,                    frac = 4096/163840  = 0.025  -> pass
    EXPECT_EQ(g.slots_in(7), 5);
    const ExpertZoneGeometry::Fit fit = g.best_fit(7);
    EXPECT_EQ(fit.num_slots, 3);
    EXPECT_EQ(fit.num_slabs, 4);
}

// ── 6. A slot size that is NOT 4096-aligned ──────────────────────────────────

TEST(ExpertZoneMath, UnalignedSlotStride) {
    const ExpertZoneGeometry g{
        .expert_slot_bytes = 13000,  // deliberately not a 4096 multiple
        .slab_bytes = 40960,
        .max_waste = 0.9,
    };
    ASSERT_TRUE(g.valid());

    // align_up(13000, 4096) = 16384 -> stride exceeds the slot size.
    EXPECT_EQ(g.slot_stride(), 16384);
    EXPECT_EQ(kExpertZoneSlotAlign, 4096);

    // floor((40960 - 13000) / 16384) + 1 = floor(1.706) + 1 = 2.
    EXPECT_EQ(g.slots_in(1), 2);

    // Alignment padding counts as waste: 40960 - 2 * 13000 = 14960.
    EXPECT_EQ(g.waste_bytes(1, 2), 14960);
}

// ── 7. best_fit never overruns the free run it was given ─────────────────────

TEST(ExpertZoneMath, FitNeverExceedsRun) {
    const ExpertZoneGeometry g = glm5_next();

    for (const int run : std::vector<int>{1, 67, 68, 100, 1000, 11319}) {
        const ExpertZoneGeometry::Fit fit = g.best_fit(run);
        EXPECT_LE(fit.num_slabs, run) << "run=" << run;
        // {0,0} is the only empty form: slots and slabs agree on emptiness.
        EXPECT_EQ(fit.num_slots == 0, fit.num_slabs == 0) << "run=" << run;
    }
}

// ── 8. grant_slabs: smallest ADMISSIBLE S under the worst-case base pad ──────
//
// A grant is sized before the allocator picks the run, so the pad before
// slot 0 is unknown; slab_bytes is not a 4096-multiple on any served model,
// so it varies per start slab. grant_slabs reasons under base_misalign = 1
// (pad 4095, the largest possible). Monotonicity does the rest: for a fixed
// S a smaller actual pad yields >= slots and <= waste, so admissibility here
// implies admissibility at whatever base comes back.

TEST(ExpertZoneMath, GrantSlabsWorstCasePad) {
    // (a) Served geometry: a slot spans ~67.6 slabs, so 4095 B of pad is
    // < 0.03% of one slot and changes nothing — grant_slabs returns the bare
    // slot fit on its first iteration.
    const ExpertZoneGeometry g = glm5_next();
    EXPECT_EQ(g.slabs_for_slots(8), 541);
    EXPECT_EQ(g.slabs_for_slots(8, /*base_misalign=*/1), 541);
    EXPECT_EQ(g.grant_slabs(8, 100000), 541);
    // 541 slabs = 147,394,368 B hold 8 slots (waste 69,440 B = 0.047%).
    EXPECT_EQ(g.slots_in(541, 1), 8);

    // One slot needs 68 slabs (waste 110,848 B of 18,526,464 = 0.598%).
    EXPECT_EQ(g.grant_slabs(1, 68), 68);
    // 67 slabs cannot hold a slot at all, so no S <= 67 fits the run.
    EXPECT_EQ(g.grant_slabs(1, 67), 0);

    // (b) Small-slot geometry — 3 slots per slab EXACTLY, which is where the
    // waste bound actually bites. Under pad 4095 the last slot of each slab
    // no longer fits, so S must grow until the shortfall amortizes:
    //   S=1: cap    294,912  n=2   waste  98,304 = 33.33%  > 10%
    //   S=2: cap    589,824  n=5   waste  98,304 = 16.67%  > 10%
    //   S=3: cap    884,736  n=8   waste  98,304 = 11.11%  > 10%
    //   S=4: cap  1,179,648  n=11  waste  98,304 =  8.33%  <= 10%  ADMISSIBLE
    // The lost capacity is always the one slot the pad displaces; only the
    // denominator grows, which is why the smallest admissible S exists.
    const ExpertZoneGeometry small{
        .expert_slot_bytes = 98304,
        .slab_bytes = 294912,
        .max_waste = 0.10,
    };
    ASSERT_TRUE(small.valid());
    EXPECT_EQ(small.slot_stride(), 98304);  // already 4096-aligned
    EXPECT_EQ(small.slabs_for_slots(2, /*base_misalign=*/1), 1);
    EXPECT_FALSE(small.admissible(1, 1));
    EXPECT_FALSE(small.admissible(2, 1));
    EXPECT_FALSE(small.admissible(3, 1));
    EXPECT_TRUE(small.admissible(4, 1));
    EXPECT_EQ(small.slots_in(4, 1), 11);
    EXPECT_EQ(small.grant_slabs(2, 45), 4);

    // A run too short to reach the smallest admissible S grants NOTHING —
    // refusing beats handing out a region that breaks the waste bound.
    EXPECT_EQ(small.grant_slabs(2, 3), 0);

    // Degenerate asks are refusals, not crashes.
    EXPECT_EQ(small.grant_slabs(0, 45), 0);
    EXPECT_EQ(small.grant_slabs(2, 0), 0);
}

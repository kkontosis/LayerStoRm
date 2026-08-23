// CPU unit tests for the EP-within-TP residency dedup (INV-MOE-EP-DISJOINT
// enforcement; TD-MOE-EP-COMBINE-RESIDUAL fix, Phase-1c).
//
// dedup_ep_residency() is the SAFETY MECHANISM for the now-UNCONDITIONAL
// (force-ON, no env gate) cross-GPU dedup baked into dispatch_moe_all_ranks. It
// must:
//   (A) be a byte-identical NO-OP on DISJOINT placement (no expert dropped,
//       output == pre-dedup) — so production RUN_MOE / baseline e%tp is
//       unaffected and the dedup costs nothing on the disjoint path.
//   (B) on NON-disjoint placement (a synthetic duplicate) reduce residency to a
//       SINGLE canonical owner = the LOWEST-rank holder, so the EP SUM-combine
//       single-counts each expert (deduped compute-set == single-count
//       reference).
//   (C) handle the edge cases: an expert on ALL GPUs; multiple duplicates within
//       one token's routed top-K; a duplicate that is NOT in the routed top-K
//       (must be left untouched — dedup is over residency, independent of which
//       experts a token routes to).
//
// Pure CPU, no CUDA, no GPU. Synthetic per-GPU bitsets only.
#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "daemon/ep_residency_dedup.h"

namespace {

namespace ld = layerstorm::daemon;

// ── Bitset helpers ──────────────────────────────────────────────────────────
constexpr int kNExperts = 24;  // 3 bytes — spans a byte boundary deliberately.

using Bitset = std::vector<uint8_t>;

Bitset make_bitset() { return Bitset((kNExperts + 7) / 8, 0); }

void set_expert(Bitset& bs, int e) {
    bs[e / 8] |= static_cast<uint8_t>(1u << (e % 8));
}
bool has_expert(const Bitset& bs, int e) {
    return (bs[e / 8] >> (e % 8)) & 1;
}

// Count residency of expert e across all ranks.
int holders(const std::vector<Bitset>& ranks, int e) {
    int n = 0;
    for (const auto& bs : ranks)
        if (has_expert(bs, e)) ++n;
    return n;
}

// Lowest rank holding expert e, or -1.
int lowest_holder(const std::vector<Bitset>& ranks, int e) {
    for (int r = 0; r < static_cast<int>(ranks.size()); ++r)
        if (has_expert(ranks[r], e)) return r;
    return -1;
}

int run_dedup(std::vector<Bitset>& ranks) {
    std::vector<uint8_t*> ptrs;
    ptrs.reserve(ranks.size());
    for (auto& bs : ranks) ptrs.push_back(bs.data());
    return ld::dedup_ep_residency(ptrs.data(), static_cast<int>(ranks.size()),
                                  kNExperts);
}

// ── Single-count reference combine ──────────────────────────────────────────
// Model each expert's contribution c_e as a scalar (placement-independent — the
// I8 §11 proof). The EP SUM-combine sums c_e over EVERY (rank, expert) it is
// resident on. The CORRECT (single-count) total sums each routed expert ONCE.
double combine_over_residency(const std::vector<Bitset>& ranks,
                              const std::vector<int>& routed,
                              const std::vector<double>& c) {
    double total = 0.0;
    for (int e : routed)
        for (const auto& bs : ranks)
            if (has_expert(bs, e)) total += c[e];  // every holder contributes
    return total;
}
double single_count_reference(const std::vector<int>& routed,
                              const std::vector<double>& c) {
    double total = 0.0;
    for (int e : routed) total += c[e];  // each routed expert exactly once
    return total;
}

std::vector<double> make_contributions() {
    std::vector<double> c(kNExperts);
    for (int e = 0; e < kNExperts; ++e) c[e] = 1.0 + 0.5 * e;  // distinct, nonzero
    return c;
}

}  // namespace

// (A) DISJOINT placement → dedup is a byte-identical NO-OP.
TEST(EpResidencyDedup, DisjointIsByteIdenticalNoOp) {
    // 2 ranks, strict e%tp split (the production / baseline disjoint pattern).
    std::vector<Bitset> ranks(2, make_bitset());
    for (int e = 0; e < kNExperts; ++e) set_expert(ranks[e % 2], e);

    const std::vector<Bitset> before = ranks;
    const int dups = run_dedup(ranks);

    EXPECT_EQ(dups, 0);
    EXPECT_EQ(ranks, before);  // byte-identical: nothing dropped, nothing moved.

    // Every expert still resident on exactly one GPU (none lost).
    for (int e = 0; e < kNExperts; ++e) EXPECT_EQ(holders(ranks, e), 1) << e;
}

// (A) Disjoint over MORE than 2 ranks is also a no-op.
TEST(EpResidencyDedup, DisjointFourRanksNoOp) {
    std::vector<Bitset> ranks(4, make_bitset());
    for (int e = 0; e < kNExperts; ++e) set_expert(ranks[e % 4], e);
    const std::vector<Bitset> before = ranks;
    EXPECT_EQ(run_dedup(ranks), 0);
    EXPECT_EQ(ranks, before);
}

// (B) NON-disjoint (one synthetic duplicate) → deduped combine == single-count
//     reference, and the canonical owner is the LOWEST-rank holder.
TEST(EpResidencyDedup, SingleDuplicateCollapsesToLowestRankOwner) {
    const auto c = make_contributions();
    std::vector<Bitset> ranks(2, make_bitset());
    for (int e = 0; e < kNExperts; ++e) set_expert(ranks[e % 2], e);

    // Expert 5 lives on rank 1 (5%2==1). Reroute leaves a stale copy on rank 0
    // too → cross-GPU duplicate.
    set_expert(ranks[0], 5);
    ASSERT_EQ(holders(ranks, 5), 2);

    const std::vector<int> routed = {2, 5, 8, 11};  // 5 is routed (top-K hit)

    // BEFORE dedup the SUM-combine double-counts expert 5.
    EXPECT_DOUBLE_EQ(combine_over_residency(ranks, routed, c),
                     single_count_reference(routed, c) + c[5]);

    const int dups = run_dedup(ranks);
    EXPECT_EQ(dups, 1);

    // Canonical owner = lowest-rank holder (rank 0), the other cleared.
    EXPECT_EQ(holders(ranks, 5), 1);
    EXPECT_EQ(lowest_holder(ranks, 5), 0);
    EXPECT_TRUE(has_expert(ranks[0], 5));
    EXPECT_FALSE(has_expert(ranks[1], 5));

    // AFTER dedup the combine matches the single-count reference exactly.
    EXPECT_DOUBLE_EQ(combine_over_residency(ranks, routed, c),
                     single_count_reference(routed, c));
}

// (C) An expert resident on ALL GPUs collapses to rank 0 only.
TEST(EpResidencyDedup, ExpertOnAllGpus) {
    const auto c = make_contributions();
    std::vector<Bitset> ranks(4, make_bitset());
    // Disjoint base over 4 ranks ...
    for (int e = 0; e < kNExperts; ++e) set_expert(ranks[e % 4], e);
    // ... then make expert 9 resident on ALL four ranks.
    for (auto& bs : ranks) set_expert(bs, 9);
    ASSERT_EQ(holders(ranks, 9), 4);

    const std::vector<int> routed = {9};
    EXPECT_DOUBLE_EQ(combine_over_residency(ranks, routed, c), 4.0 * c[9]);

    EXPECT_EQ(run_dedup(ranks), 1);
    EXPECT_EQ(holders(ranks, 9), 1);
    EXPECT_EQ(lowest_holder(ranks, 9), 0);
    EXPECT_DOUBLE_EQ(combine_over_residency(ranks, routed, c), c[9]);
}

// (C) Multiple duplicates within one token's routed top-K all collapse.
TEST(EpResidencyDedup, MultipleDuplicatesInOneToken) {
    const auto c = make_contributions();
    std::vector<Bitset> ranks(2, make_bitset());
    for (int e = 0; e < kNExperts; ++e) set_expert(ranks[e % 2], e);

    // Three duplicates: 4 (home r0, stale r1), 7 (home r1, stale r0),
    // 20 (home r0, stale r1). All three are in the routed top-K.
    set_expert(ranks[1], 4);
    set_expert(ranks[0], 7);
    set_expert(ranks[1], 20);
    ASSERT_EQ(holders(ranks, 4), 2);
    ASSERT_EQ(holders(ranks, 7), 2);
    ASSERT_EQ(holders(ranks, 20), 2);

    const std::vector<int> routed = {4, 7, 20, 1};
    EXPECT_DOUBLE_EQ(combine_over_residency(ranks, routed, c),
                     single_count_reference(routed, c) + c[4] + c[7] + c[20]);

    EXPECT_EQ(run_dedup(ranks), 3);  // three cross-GPU duplicates found

    EXPECT_EQ(lowest_holder(ranks, 4), 0);   // 4%2==0 owner kept on r0
    EXPECT_EQ(lowest_holder(ranks, 7), 0);   // stale r0 wins as lowest holder
    EXPECT_EQ(lowest_holder(ranks, 20), 0);  // 20%2==0 owner kept on r0
    for (int e : {4, 7, 20}) EXPECT_EQ(holders(ranks, e), 1) << e;

    EXPECT_DOUBLE_EQ(combine_over_residency(ranks, routed, c),
                     single_count_reference(routed, c));
}

// (C) A duplicate NOT in the routed top-K must be deduped too (dedup is over
//     residency, not routing) and must not perturb the routed total.
TEST(EpResidencyDedup, DuplicateOutsideTopKUntouchedInResult) {
    const auto c = make_contributions();
    std::vector<Bitset> ranks(2, make_bitset());
    for (int e = 0; e < kNExperts; ++e) set_expert(ranks[e % 2], e);

    // Expert 13 is duplicated but is NOT routed by this token.
    set_expert(ranks[0], 13);
    ASSERT_EQ(holders(ranks, 13), 2);

    const std::vector<int> routed = {0, 2, 6, 10};  // 13 absent
    const double before_total = combine_over_residency(ranks, routed, c);
    // No routed duplicate → combine already equals the single-count reference.
    EXPECT_DOUBLE_EQ(before_total, single_count_reference(routed, c));

    EXPECT_EQ(run_dedup(ranks), 1);  // the residency duplicate is still resolved

    EXPECT_EQ(holders(ranks, 13), 1);
    EXPECT_EQ(lowest_holder(ranks, 13), 0);
    // The routed total (which never involved 13) is unchanged.
    EXPECT_DOUBLE_EQ(combine_over_residency(ranks, routed, c), before_total);
}

// Null rank pointers are skipped (defensive: absent ranks in dispatch).
TEST(EpResidencyDedup, NullRankPointersSkipped) {
    Bitset r0 = make_bitset();
    Bitset r2 = make_bitset();
    set_expert(r0, 3);
    set_expert(r2, 3);  // duplicate across the two non-null ranks
    uint8_t* ptrs[3] = {r0.data(), nullptr, r2.data()};

    EXPECT_EQ(ld::dedup_ep_residency(ptrs, 3, kNExperts), 1);
    EXPECT_TRUE(has_expert(r0, 3));   // lowest non-null holder kept
    EXPECT_FALSE(has_expert(r2, 3));  // higher holder cleared
}

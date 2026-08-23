// Unit tests for the weighted-sum place_cons cost model (residency reframe).
#include "core/gpu_loader/loader_place_sum.h"

#include <gtest/gtest.h>

#include <vector>

namespace gl = layerstorm::gpu_loader;

// (a) residency baseline: would-miss-anyway (recur 0) is FREE; would-be-HIT
// (recur 1) is priced at the full cold fetch it would cost next round.
TEST(LoaderPlaceSum, ResidencyBaseline) {
  EXPECT_DOUBLE_EQ(gl::residency_baseline(0.0, 900.0), 0.0);    // safe: free
  EXPECT_DOUBLE_EQ(gl::residency_baseline(1.0, 900.0), 900.0);  // recurs: full fetch
  EXPECT_DOUBLE_EQ(gl::residency_baseline(0.5, 900.0), 450.0);  // soft probability
  EXPECT_DOUBLE_EQ(gl::residency_baseline(-1.0, 900.0), 0.0);   // clamp recur >= 0
  EXPECT_DOUBLE_EQ(gl::residency_baseline(2.0, 900.0), 900.0);  // clamp recur <= 1
  EXPECT_DOUBLE_EQ(gl::residency_baseline(1.0, -5.0), 0.0);     // clamp fetch >= 0
}

// The sum composes the constant tax + three weighted factors independently.
TEST(LoaderPlaceSum, WeightedSum) {
  gl::PlaceSumWeights w{/*resid_tax=*/0.0, /*w_resid=*/1.0, /*w_numa=*/2.0,
                        /*w_hotness=*/3.0};
  gl::PlaceSumFactors f{/*resid_baseline=*/100.0, /*numa_penalty=*/10.0,
                        /*hotness=*/5.0};
  // 0 + 1*100 + 2*10 + 3*5 = 135
  EXPECT_DOUBLE_EQ(gl::cpu_place_cost(w, f), 135.0);
}

// The CONSTANT churn tax is the operative residency heuristic: it applies to EVERY
// candidate and does NOT zero for recur=0 — so a would-miss-anyway (recur 0) expert
// is NO LONGER "free" to offload (the fix for the recur=0 churners a prediction-
// weighted term zeroed on). recur=1 additionally carries the protect-term.
TEST(LoaderPlaceSum, DefaultConstantChurnTax) {
  gl::PlaceSumWeights w;  // defaults: resid_tax 960, w_resid 1, numa 0, hotness 0
  gl::PlaceSumFactors recurs{gl::residency_baseline(1.0, 900.0), 50.0, 7.0};
  gl::PlaceSumFactors safe{gl::residency_baseline(0.0, 900.0), 50.0, 7.0};
  EXPECT_DOUBLE_EQ(gl::cpu_place_cost(w, recurs), 960.0 + 900.0);  // tax + protect
  EXPECT_DOUBLE_EQ(gl::cpu_place_cost(w, safe), 960.0);  // recur=0 STILL taxed
}

// All-zero weights ⇒ cost 0 (the raw wall model / byte-identity control arm) —
// includes zeroing the constant tax.
TEST(LoaderPlaceSum, AllZeroWeightsAreInert) {
  gl::PlaceSumWeights w{0.0, 0.0, 0.0, 0.0};
  EXPECT_TRUE(w.all_zero());
  gl::PlaceSumFactors f{900.0, 50.0, 7.0};
  EXPECT_DOUBLE_EQ(gl::cpu_place_cost(w, f), 0.0);
}

// The sum is clamped >= 0: a place cost never REWARDS an offload.
TEST(LoaderPlaceSum, ClampedNonNegative) {
  gl::PlaceSumWeights w{0.0, -1.0, 0.0, 0.0};
  gl::PlaceSumFactors f{100.0, 0.0, 0.0};
  EXPECT_DOUBLE_EQ(gl::cpu_place_cost(w, f), 0.0);  // -100 clamped to 0
}

// (d) NODE-BASED offload bias: HBM-home (M3-hot) pays hbm_offload_base (LOW ⇒
// encourage offload), DDR-home (cold tail) pays ddr_offload_penalty (HIGH ⇒ protect).
// home_hbm selects the column; the two never both apply to one expert.
TEST(LoaderPlaceSum, NodeBasedBias) {
  gl::PlaceSumWeights w{};             // resid_tax 960, w_resid 1, rest defaults
  w.resid_tax = 0.0; w.w_resid = 0.0;  // isolate the node term
  w.hbm_offload_base = 50.0;           // LOW ⇒ HBM-home cheap to offload
  w.ddr_offload_penalty = 100000.0;    // HIGH ⇒ DDR-home protected
  gl::PlaceSumFactors hbm{};  hbm.home_hbm = 1.0;
  gl::PlaceSumFactors ddr{};  ddr.home_hbm = 0.0;
  EXPECT_DOUBLE_EQ(gl::cpu_place_cost(w, hbm), 50.0);      // only the HBM base
  EXPECT_DOUBLE_EQ(gl::cpu_place_cost(w, ddr), 100000.0);  // only the DDR penalty
  // The node bias is ADDITIVE with the other factors.
  gl::PlaceSumWeights w2{};  // defaults + node bias
  w2.hbm_offload_base = 25.0;
  gl::PlaceSumFactors f{gl::residency_baseline(1.0, 800.0), 0.0, 0.0, /*home_hbm=*/1.0};
  // resid_tax 960 + w_resid 1 * 800 + hbm_base 25 = 1785
  EXPECT_DOUBLE_EQ(gl::cpu_place_cost(w2, f), 1785.0);
}

// The node-bias weights participate in all_zero() ⇒ setting ONLY the node term
// still engages the place_sum machinery (not short-circuited as inert).
TEST(LoaderPlaceSum, NodeBiasEngagesAllZero) {
  gl::PlaceSumWeights hbm_only{0.0, 0.0, 0.0, 0.0};
  EXPECT_TRUE(hbm_only.all_zero());
  hbm_only.hbm_offload_base = 10.0;
  EXPECT_FALSE(hbm_only.all_zero());
  gl::PlaceSumWeights ddr_only{0.0, 0.0, 0.0, 0.0};
  ddr_only.ddr_offload_penalty = 10.0;
  EXPECT_FALSE(ddr_only.all_zero());
}

// (e) TARGET-NODE RARE steering: offnode_protect penalizes the COMPLEMENT (protect
// non-target/hot), target_rare_encourage drives the target-rare set toward the 0
// floor. target_rare selects the column.
TEST(LoaderPlaceSum, TargetRareSteering) {
  gl::PlaceSumWeights w{};              // resid_tax 960, w_resid 1, rest 0
  w.w_resid = 0.0;                      // isolate resid_tax + (e)
  w.offnode_protect = 100000.0;         // HIGH ⇒ complement protected (won't offload)
  w.target_rare_encourage = 960.0;      // remove the resid_tax barrier for target-rare
  gl::PlaceSumFactors tr{};   tr.target_rare = 1.0;  // rare expert on the target node
  gl::PlaceSumFactors other{}; other.target_rare = 0.0;
  // target-rare: resid_tax 960 − encourage 960 = 0 ⇒ free to offload.
  EXPECT_DOUBLE_EQ(gl::cpu_place_cost(w, tr), 0.0);
  // complement: resid_tax 960 + offnode_protect 100000 = 100960 ⇒ protected.
  EXPECT_DOUBLE_EQ(gl::cpu_place_cost(w, other), 100960.0);
}

// (e) encourage clamps at the 0 floor — it never REWARDS an offload below free.
TEST(LoaderPlaceSum, TargetRareEncourageClampsAtZero) {
  gl::PlaceSumWeights w{0.0, 0.0, 0.0, 0.0};
  w.resid_tax = 500.0;
  w.target_rare_encourage = 100000.0;   // far exceeds the barrier
  gl::PlaceSumFactors tr{};  tr.target_rare = 1.0;
  EXPECT_DOUBLE_EQ(gl::cpu_place_cost(w, tr), 0.0);  // 500 − 100000 clamped to 0
}

// (e) weights participate in all_zero() ⇒ setting ONLY the (e) term engages the
// machinery and stays inert (byte-identical) when both are 0.
TEST(LoaderPlaceSum, TargetRareEngagesAllZero) {
  gl::PlaceSumWeights protect_only{0.0, 0.0, 0.0, 0.0};
  EXPECT_TRUE(protect_only.all_zero());
  protect_only.offnode_protect = 10.0;
  EXPECT_FALSE(protect_only.all_zero());
  gl::PlaceSumWeights enc_only{0.0, 0.0, 0.0, 0.0};
  enc_only.target_rare_encourage = 10.0;
  EXPECT_FALSE(enc_only.all_zero());
}

// The batch helper fills out[i] for every candidate; all-zero weights short-circuit.
TEST(LoaderPlaceSum, BatchHelper) {
  gl::PlaceSumWeights w{/*resid_tax=*/0.0, /*w_resid=*/1.0, 0.0, 0.0};  // protect only
  std::vector<gl::PlaceSumFactors> factors = {
      {gl::residency_baseline(1.0, 800.0), 0.0, 0.0},  // recurs → 800
      {gl::residency_baseline(0.0, 800.0), 0.0, 0.0},  // safe   → 0
      {gl::residency_baseline(1.0, 300.0), 0.0, 0.0},  // recurs → 300
  };
  std::vector<double> out;
  gl::compute_cpu_place_costs(w, factors, out);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_DOUBLE_EQ(out[0], 800.0);
  EXPECT_DOUBLE_EQ(out[1], 0.0);
  EXPECT_DOUBLE_EQ(out[2], 300.0);

  // The constant tax taxes EVERY candidate (incl. the recur=0 one).
  gl::PlaceSumWeights taxed{/*resid_tax=*/500.0, /*w_resid=*/1.0, 0.0, 0.0};
  gl::compute_cpu_place_costs(taxed, factors, out);
  EXPECT_DOUBLE_EQ(out[0], 1300.0);  // 500 + 800
  EXPECT_DOUBLE_EQ(out[1], 500.0);   // 500 + 0  (recur=0 no longer free)
  EXPECT_DOUBLE_EQ(out[2], 800.0);   // 500 + 300

  gl::PlaceSumWeights zero{0.0, 0.0, 0.0, 0.0};
  gl::compute_cpu_place_costs(zero, factors, out);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_DOUBLE_EQ(out[0], 0.0);
  EXPECT_DOUBLE_EQ(out[1], 0.0);
  EXPECT_DOUBLE_EQ(out[2], 0.0);
}

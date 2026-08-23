// Unit tests for the EPM-0 keeppred eviction-retention bias
// (src/core/gpu_loader/loader_keeppred.h; spec/reports/DSP52_BOOST.md "EPM-0",
// INV-KEEPPRED). The mechanism reorders a board cheapest-first victim candidate
// window so predicted-reuse experts (prev-round-union members) sink toward KEEP.
// Allocation-free / fixed-buffer: caller supplies eff/retained/order arrays.
#include "core/gpu_loader/loader_keeppred.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace {
using layerstorm::gpu_loader::keeppred_victim_order;

// bias == 0 → identity order (the OFF/weight-0 byte-identity contract).
TEST(LoaderKeeppredTest, ZeroBiasIsIdentity) {
  const double eff[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  const uint8_t retained[5] = {1, 0, 1, 0, 1};  // retained ignored at bias 0
  int order[5] = {};
  keeppred_victim_order(eff, retained, 5, 0.0, order);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(order[i], i);
}

// No retained members → identity even at a large bias.
TEST(LoaderKeeppredTest, NoRetainedIsIdentity) {
  const double eff[3] = {5.0, 1.0, 3.0};
  const uint8_t retained[3] = {0, 0, 0};
  int order[3] = {};
  keeppred_victim_order(eff, retained, 3, 1000.0, order);
  EXPECT_EQ(order[0], 0);
  EXPECT_EQ(order[1], 1);
  EXPECT_EQ(order[2], 2);
}

// A retained cheap victim is pushed PAST the un-retained ones (protected).
TEST(LoaderKeeppredTest, RetainedSinksToKeepEnd) {
  const double eff[4] = {1.0, 2.0, 3.0, 4.0};  // idx0 cheapest but retained
  const uint8_t retained[4] = {1, 0, 0, 0};
  int order[4] = {};
  keeppred_victim_order(eff, retained, 4, 10.0, order);  // 1+10=11 > 4
  EXPECT_EQ(order[0], 1);
  EXPECT_EQ(order[1], 2);
  EXPECT_EQ(order[2], 3);
  EXPECT_EQ(order[3], 0);  // retained victim last
}

// BOUNDED weight: uniform retention (all protected) preserves order → cheapest
// still first, still evictable (never fabricates budget / livelocks).
TEST(LoaderKeeppredTest, BoundedBiasStillEvicts) {
  const double eff[3] = {1.0, 2.0, 3.0};
  const uint8_t retained[3] = {1, 1, 1};
  int order[3] = {};
  keeppred_victim_order(eff, retained, 3, 5.0, order);
  EXPECT_EQ(order[0], 0);
  EXPECT_EQ(order[1], 1);
  EXPECT_EQ(order[2], 2);
}

// Small bias only reorders across the gap it can bridge (partial protection).
TEST(LoaderKeeppredTest, SmallBiasPartialReorder) {
  const double eff[3] = {1.0, 2.0, 10.0};
  const uint8_t retained[3] = {1, 0, 0};
  int order[3] = {};
  keeppred_victim_order(eff, retained, 3, 1.5, order);  // idx0 -> 2.5
  EXPECT_EQ(order[0], 1);  // 2.0
  EXPECT_EQ(order[1], 0);  // 2.5
  EXPECT_EQ(order[2], 2);  // 10.0
}

// Stability: equal adjusted scores preserve the input (cheapest-first) order,
// so victim selection stays deterministic (lowest-slot tie-break).
TEST(LoaderKeeppredTest, StableTieBreak) {
  const double eff[4] = {0.0, 1.0, 2.0, 2.0};  // idx1 retained bias1 -> 2.0
  const uint8_t retained[4] = {0, 1, 0, 0};
  int order[4] = {};
  keeppred_victim_order(eff, retained, 4, 1.0, order);
  EXPECT_EQ(order[0], 0);  // 0.0
  EXPECT_EQ(order[1], 1);  // 2.0 (ties idx2,idx3 → stable input order)
  EXPECT_EQ(order[2], 2);
  EXPECT_EQ(order[3], 3);
}

// Degenerate sizes.
TEST(LoaderKeeppredTest, EmptyAndSingle) {
  int order[1] = {-1};
  keeppred_victim_order(nullptr, nullptr, 0, 5.0, order);  // n=0: no-op
  const double eff[1] = {3.0};
  const uint8_t retained[1] = {1};
  keeppred_victim_order(eff, retained, 1, 5.0, order);
  EXPECT_EQ(order[0], 0);
}

// Window-scale: a realistic bounded window with a scattered retained subset —
// all un-retained come first (cheapest order), then retained (cheapest order).
TEST(LoaderKeeppredTest, WindowScaleTwoTierMerge) {
  constexpr int N = 12;
  double eff[N];
  uint8_t retained[N];
  for (int i = 0; i < N; ++i) { eff[i] = static_cast<double>(i); retained[i] = 0; }
  retained[0] = retained[3] = retained[4] = 1;  // cheap ones retained
  int order[N] = {};
  keeppred_victim_order(eff, retained, N, 1000.0, order);  // strong bias
  // Un-retained indices in ascending eff first: 1,2,5,6,7,8,9,10,11
  const int expect_unret[9] = {1, 2, 5, 6, 7, 8, 9, 10, 11};
  for (int i = 0; i < 9; ++i) EXPECT_EQ(order[i], expect_unret[i]) << "i=" << i;
  // Then retained in ascending eff: 0,3,4
  EXPECT_EQ(order[9], 0);
  EXPECT_EQ(order[10], 3);
  EXPECT_EQ(order[11], 4);
}

}  // namespace

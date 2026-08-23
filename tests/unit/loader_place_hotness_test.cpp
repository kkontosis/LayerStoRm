// Unit tests for the hotness-steered place_cons builder (TASK-A2 phase 1).
#include "core/gpu_loader/loader_place_hotness.h"

#include <gtest/gtest.h>

#include <vector>

namespace gl = layerstorm::gpu_loader;

namespace {

gl::SolveRequest make_req(int n, int M) {
  gl::SolveRequest req;
  req.num_devices = M;
  req.num_experts = n;
  req.cached.assign(static_cast<size_t>(n) * M, 0);
  return req;
}

gl::HotnessPlaceSignals make_sig(int M, double w,
                                 std::vector<double> victim_hot) {
  gl::HotnessPlaceSignals sig;
  sig.num_devices = M;
  sig.w = w;
  for (int j = 0; j < M; ++j) {
    sig.valid[static_cast<size_t>(j)] = 1;
    sig.victim_hot[static_cast<size_t>(j)] = victim_hot[static_cast<size_t>(j)];
  }
  return sig;
}

}  // namespace

// The product form: place[i][j] = w * coldness[i] * victim_hot[j].
TEST(LoaderPlaceHotness, ProductForm) {
  const int n = 2, M = 3;
  auto req = make_req(n, M);
  auto sig = make_sig(M, 10.0, {1.0, 0.5, 0.0});
  std::vector<double> cold = {1.0, 0.25};  // expert 0 cold, expert 1 warmish
  gl::add_hotness_place(sig, cold, req);

  ASSERT_EQ(req.place.size(), static_cast<size_t>(n * M));
  // expert 0 (cold=1): 10*1*{1,0.5,0} = {10,5,0}
  EXPECT_DOUBLE_EQ(req.place_at(0, 0), 10.0);
  EXPECT_DOUBLE_EQ(req.place_at(0, 1), 5.0);
  EXPECT_DOUBLE_EQ(req.place_at(0, 2), 0.0);
  // expert 1 (cold=0.25): 10*0.25*{1,0.5,0} = {2.5,1.25,0}
  EXPECT_DOUBLE_EQ(req.place_at(1, 0), 2.5);
  EXPECT_DOUBLE_EQ(req.place_at(1, 1), 1.25);
  EXPECT_DOUBLE_EQ(req.place_at(1, 2), 0.0);
}

// A hot expert (coldness 0) is never penalized, on any device.
TEST(LoaderPlaceHotness, HotExpertZeroCost) {
  const int n = 1, M = 2;
  auto req = make_req(n, M);
  auto sig = make_sig(M, 1000.0, {1.0, 1.0});
  std::vector<double> cold = {0.0};
  gl::add_hotness_place(sig, cold, req);
  EXPECT_DOUBLE_EQ(req.place_at(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(req.place_at(0, 1), 0.0);
}

// A free-slot device (victim_hot 0) never charges the placement — nothing is
// displaced there.
TEST(LoaderPlaceHotness, FreeSlotDeviceZeroCost) {
  const int n = 1, M = 2;
  auto req = make_req(n, M);
  auto sig = make_sig(M, 1000.0, {0.0, 0.8});
  std::vector<double> cold = {1.0};
  gl::add_hotness_place(sig, cold, req);
  EXPECT_DOUBLE_EQ(req.place_at(0, 0), 0.0);     // free slot
  EXPECT_DOUBLE_EQ(req.place_at(0, 1), 800.0);   // hot victim, cold expert
}

// Invalid (unroutable) devices contribute nothing even with a hot victim.
TEST(LoaderPlaceHotness, InvalidDeviceSkipped) {
  const int n = 1, M = 2;
  auto req = make_req(n, M);
  auto sig = make_sig(M, 1.0, {1.0, 1.0});
  sig.valid[1] = 0;
  std::vector<double> cold = {1.0};
  gl::add_hotness_place(sig, cold, req);
  EXPECT_DOUBLE_EQ(req.place_at(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(req.place_at(0, 1), 0.0);
}

// A short coldness span treats the missing tail as fully cold (1.0).
TEST(LoaderPlaceHotness, ShortColdnessTailIsCold) {
  const int n = 2, M = 1;
  auto req = make_req(n, M);
  auto sig = make_sig(M, 4.0, {0.5});
  std::vector<double> cold = {0.0};  // only expert 0 supplied
  gl::add_hotness_place(sig, cold, req);
  EXPECT_DOUBLE_EQ(req.place_at(0, 0), 0.0);   // hot
  EXPECT_DOUBLE_EQ(req.place_at(1, 0), 2.0);   // 4*1.0*0.5 (tail = cold)
}

// The term ADDS onto an existing place row (composes with other place terms).
TEST(LoaderPlaceHotness, AddsOntoExisting) {
  const int n = 1, M = 2;
  auto req = make_req(n, M);
  req.place.assign(static_cast<size_t>(n * M), 0.0);
  req.place[0] = 3.0;
  req.place[1] = 7.0;
  auto sig = make_sig(M, 2.0, {1.0, 0.0});
  std::vector<double> cold = {1.0};
  gl::add_hotness_place(sig, cold, req);
  EXPECT_DOUBLE_EQ(req.place_at(0, 0), 5.0);   // 3 + 2*1*1
  EXPECT_DOUBLE_EQ(req.place_at(0, 1), 7.0);   // 7 + 2*1*0
}

// w == 0 is an exact no-op (default-OFF byte-identity guarantee).
TEST(LoaderPlaceHotness, ZeroScaleNoOp) {
  const int n = 2, M = 2;
  auto req = make_req(n, M);
  auto sig = make_sig(M, 0.0, {1.0, 1.0});
  std::vector<double> cold = {1.0, 1.0};
  gl::add_hotness_place(sig, cold, req);
  EXPECT_TRUE(req.place.empty());  // untouched
}

// Negative w with unclamped solver rewards keeping the hottest resident.
TEST(LoaderPlaceHotness, NegativeScaleRewards) {
  const int n = 1, M = 2;
  auto req = make_req(n, M);
  auto sig = make_sig(M, -5.0, {1.0, 0.0});
  std::vector<double> cold = {1.0};
  gl::add_hotness_place(sig, cold, req);
  EXPECT_DOUBLE_EQ(req.place_at(0, 0), -5.0);
  EXPECT_DOUBLE_EQ(req.place_at(0, 1), 0.0);
}

// reuse_inv_hot01 shape sanity.
TEST(LoaderPlaceHotness, ReuseInvHot01) {
  EXPECT_DOUBLE_EQ(gl::reuse_inv_hot01(0.0, 100.0), 1.0);       // freshest = hot
  EXPECT_DOUBLE_EQ(gl::reuse_inv_hot01(100.0, 100.0), 0.5);     // age == tau
  EXPECT_DOUBLE_EQ(gl::reuse_inv_hot01(-3.0, 100.0), 1.0);      // clamp age>=0
  EXPECT_DOUBLE_EQ(gl::reuse_inv_hot01(1e9, 0.0), 1.0);         // tau<=0 -> inf
  EXPECT_LT(gl::reuse_inv_hot01(400.0, 100.0), 0.25);           // very old = cold
}

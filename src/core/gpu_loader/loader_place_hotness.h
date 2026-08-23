#pragma once
// Hotness-steered place_cons (TASK-A2 phase 1 — activate the dormant place_cons
// solver term from the EvictScoreBoard hotness signal `eff`).
//
// Populates SolveRequest::place so that FETCHING (missing) a COLD expert onto a
// device whose displaced victim is HOT costs more than fetching a hot expert, or
// fetching onto a device with a free/cold victim. The term therefore steers HOT
// residents to STAY on their GPU (they are never the ones displaced) and — when
// a flat-cost CPU device column is present — steers COLD experts toward it.
//
//   place[i][j] (charged ONLY on a MISS — apply() skips place for a hit) =
//       w * coldness[i] * victim_hot[j]
//
//   coldness[i] in [0,1] : 1 = maximally COLD (the routed expert is resident on
//     no device, i.e. a cold-tail miss), 0 = HOT (a warm replica exists and was
//     recently used). coldness = 1 - 1/(1 + age_i/tau), age_i = recency_now -
//     best-resident effective score of expert i (the sim's reuse:inv shape).
//   victim_hot[j] in [0,1] : 1/(1 + victim_age_j/tau), the reuse:inv hotness of
//     device j's cheapest evictable resident (the board heap top). A device with
//     a stable free slot displaces nothing -> victim_hot = 0 -> no penalty.
//
// The PRODUCT is zero whenever the expert is hot OR the device has a cheap/free
// victim, so the term bites ONLY the bad case: a cold thrasher evicting a hot
// resident. req.clamp_place keeps it >= 0 (default); set clamp_place=false with a
// NEGATIVE w to instead REWARD keeping the hottest resident (true negative cost).
//
// Pure and CUDA-free (layerstorm_core / INV-GPU-1). All live-state reads (board
// scores, cache free slots, per-expert residency) happen in the caller
// (dispatch_loader.cpp); this unit is unit-testable with no daemon state.
#include <array>
#include <cstdint>
#include <span>

#include "core/gpu_loader/loader_solver.h"

namespace layerstorm::gpu_loader {

// Per-device victim-hotness signals, frozen for the whole layer solve (matching
// the affinity/reuse terms, whose LRU state does not change while assigning one
// layer's misses). Indexed by solver device j.
struct HotnessPlaceSignals {
  int num_devices = 0;
  double w = 0.0;                                 // scale (us); may be negative
  std::array<uint8_t, kMaxDevices> valid{};       // routable device
  std::array<double, kMaxDevices> victim_hot{};   // 1/(1+victim_age/tau); free=0
};

// ADDS w * coldness[i] * victim_hot[j] into req.place[i*M + j]. req.place is
// resized to N*M (zero-filled) if it does not already carry a same-sized term.
// coldness is read for the first req.num_experts entries (extra ignored; short
// spans treat the missing tail as fully cold = 1.0, the safe conservative default
// that never under-penalizes). Invalid devices (valid[j]==0) contribute nothing.
void add_hotness_place(const HotnessPlaceSignals& sig,
                       std::span<const double> coldness, SolveRequest& req);

// The reuse:inv hotness map hot01(age) = 1/(1 + age/tau), clamped age >= 0.
// tau <= 0 is treated as +inf (hot01 == 1 for any finite age). Shared by the
// caller so the coldness fill and the victim_hot fill use one definition.
inline double reuse_inv_hot01(double age, double tau) {
  if (age < 0.0) age = 0.0;
  if (tau <= 0.0) return 1.0;
  return 1.0 / (1.0 + age / tau);
}

}  // namespace layerstorm::gpu_loader

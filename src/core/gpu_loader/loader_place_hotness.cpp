#include "core/gpu_loader/loader_place_hotness.h"

#include <algorithm>

namespace layerstorm::gpu_loader {

void add_hotness_place(const HotnessPlaceSignals& sig,
                       std::span<const double> coldness, SolveRequest& req) {
  const int M = req.num_devices;
  const int n = req.num_experts;
  if (M <= 0 || n <= 0 || sig.w == 0.0) return;

  const size_t nm = static_cast<size_t>(n) * static_cast<size_t>(M);
  if (req.place.size() != nm) req.place.assign(nm, 0.0);

  const int cn = static_cast<int>(coldness.size());
  for (int i = 0; i < n; ++i) {
    // Missing coldness tail -> treat as fully cold (1.0): conservative, never
    // under-penalizes an unscored expert.
    const double ci = (i < cn) ? coldness[static_cast<size_t>(i)] : 1.0;
    if (ci == 0.0) continue;  // hot expert: no penalty on any device
    const double wci = sig.w * ci;
    double* prow = &req.place[static_cast<size_t>(i) * static_cast<size_t>(M)];
    for (int j = 0; j < M; ++j) {
      if (!sig.valid[static_cast<size_t>(j)]) continue;
      prow[j] += wci * sig.victim_hot[static_cast<size_t>(j)];
    }
  }
}

}  // namespace layerstorm::gpu_loader

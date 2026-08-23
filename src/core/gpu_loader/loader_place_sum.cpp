#include "core/gpu_loader/loader_place_sum.h"

namespace layerstorm::gpu_loader {

void compute_cpu_place_costs(const PlaceSumWeights& w,
                             std::span<const PlaceSumFactors> factors,
                             std::vector<double>& out) {
  const size_t n = factors.size();
  out.assign(n, 0.0);
  if (w.all_zero()) return;  // fast path: no factor engaged ⇒ all-zero (raw wall)
  for (size_t i = 0; i < n; ++i) out[i] = cpu_place_cost(w, factors[i]);
}

}  // namespace layerstorm::gpu_loader

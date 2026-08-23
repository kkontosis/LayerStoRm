#include "core/gpu_loader/loader_place_cons.h"

#include <algorithm>
#include <cassert>

namespace layerstorm::gpu_loader {

PlaceConsTable::PlaceConsTable(int num_experts, int num_gpus)
    : num_experts_(num_experts < 0 ? 0 : num_experts),
      num_gpus_(num_gpus < 0 ? 0 : num_gpus),
      data_(static_cast<size_t>(num_experts_) * static_cast<size_t>(num_gpus_),
            0.0) {}

void PlaceConsTable::set(int expert, int gpu, double v) {
  if (!in_range(expert, gpu)) return;
  data_[idx(expert, gpu)] = v;
}

void PlaceConsTable::add(int expert, int gpu, double dv) {
  if (!in_range(expert, gpu)) return;
  data_[idx(expert, gpu)] += dv;
}

void PlaceConsTable::set_row(int expert, std::span<const double> row_m) {
  if (expert < 0 || expert >= num_experts_) return;
  const int m = std::min(static_cast<int>(row_m.size()), num_gpus_);
  double* dst = &data_[idx(expert, 0)];
  for (int j = 0; j < m; ++j) dst[j] = row_m[j];
}

void PlaceConsTable::clear() { std::fill(data_.begin(), data_.end(), 0.0); }

double PlaceConsTable::at(int expert, int gpu) const {
  if (!in_range(expert, gpu)) return 0.0;
  return data_[idx(expert, gpu)];
}

const double* PlaceConsTable::row(int expert) const {
  if (expert < 0 || expert >= num_experts_) return nullptr;
  return &data_[idx(expert, 0)];
}

void PlaceConsTable::gather(std::span<const int> globals,
                            std::vector<double>& out_local_nm) const {
  const int n = static_cast<int>(globals.size());
  out_local_nm.assign(static_cast<size_t>(n) * static_cast<size_t>(num_gpus_),
                      0.0);
  for (int i = 0; i < n; ++i) {
    const int g = globals[i];
    double* dst = &out_local_nm[static_cast<size_t>(i) *
                                static_cast<size_t>(num_gpus_)];
    if (g < 0 || g >= num_experts_) continue;  // unseen / OOB → neutral zero row
    const double* src = &data_[idx(g, 0)];      // M-contiguous row read
    for (int j = 0; j < num_gpus_; ++j) dst[j] = src[j];
  }
}

}  // namespace layerstorm::gpu_loader

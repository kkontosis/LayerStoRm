#pragma once
// Loader place_cons store (spec/tickets/LOADER_STATS_LOCALITY.md §B).
//
// Per-(expert, device) placement-consequence numbers, RAM-CONTIGUOUS with OUTER
// stride = expert and INNER stride = GPU (the GPU dim is the inner/local one).
// One expert's M consequence values are M contiguous doubles, so gathering one
// routed expert's row touches a single cache line's worth of sequential reads.
//
// HEADLINE (INV-LOADER-PLACECONS-1): the daemon hot-path solve gathers ONLY the
// N routed experts' rows (N*M doubles) via a local->global map — never the
// X-wide table. No min-heap (place_cons needs none — §B.1).
//
// CUDA-free, pure data (layerstorm_core / INV-GPU-1). Lives on the daemon
// top-layer state; updated as expert stats update.
#include <cstddef>
#include <span>
#include <vector>

namespace layerstorm::gpu_loader {

class PlaceConsTable {
 public:
  // num_experts = the flattened expert id space (per-layer X * MoE layers); the
  // flat id is (layer_idx - first_moe_layer) * experts_per_layer + expert_idx,
  // matching ExpertStats::state_index. num_gpus = M (the inner stride).
  PlaceConsTable(int num_experts, int num_gpus);

  int num_experts() const { return num_experts_; }
  int num_gpus() const { return num_gpus_; }

  // ── Updates (as expert stats update) ──────────────────────────────────
  void set(int expert, int gpu, double v);
  void set_row(int expert, std::span<const double> row_m);  // all-M contiguous write
  // Add to an existing value (for incremental future-token reward accumulation).
  void add(int expert, int gpu, double dv);
  // Reset every value to the neutral default (0.0).
  void clear();

  // ── Reads ─────────────────────────────────────────────────────────────
  double at(int expert, int gpu) const;
  const double* row(int expert) const;  // pointer to the M-contiguous row

  // ── HOT PATH: gather the N routed experts' rows into a dense [N*M] block, in
  // LOCAL index order (local i = position in `globals`). Reads exactly N rows
  // (N*M contiguous-per-row doubles) — nothing else. out is resized to N*M and
  // laid out row-major [i*M + j] so SolveRequest.place_at(i,j) reads it directly.
  // A global id < 0 or >= num_experts (unseen / out of range) yields a zero row
  // (neutral place_cons) — never reads out of bounds.
  void gather(std::span<const int> globals, std::vector<double>& out_local_nm) const;

 private:
  size_t idx(int expert, int gpu) const {
    return static_cast<size_t>(expert) * static_cast<size_t>(num_gpus_) +
           static_cast<size_t>(gpu);
  }
  bool in_range(int expert, int gpu) const {
    return expert >= 0 && expert < num_experts_ && gpu >= 0 && gpu < num_gpus_;
  }

  int num_experts_;
  int num_gpus_;
  std::vector<double> data_;  // [num_experts_ * num_gpus_], row-major [expert][gpu]
};

}  // namespace layerstorm::gpu_loader

// Shared deterministic instance generator for the 64-instantiation golden
// regression (GpuLoaderSolver.Solver64GoldenRegression). The goldens hardcoded
// in gpu_loader_solver_test.cpp were captured from the PRE-TEMPLATE solver
// (commit 7398d777, the engine-REEF 0.7328 epoch) by running exactly this
// generator against it — DO NOT CHANGE this file without recapturing (which
// re-baselines the frozen 64-path identity guarantee).
//
// Included by gpu_loader_solver_test.cpp (and by the one-off capture harness).
// Requires: <random>, <vector>, loader_solver.h / loader_constants.h.
namespace golden64 {

struct Instance {
  layerstorm::gpu_loader::LoaderConstants k;
  layerstorm::gpu_loader::SolveRequest req;
};

// Deterministic instance `inst` (independent RNG stream per instance).
// inst 0..3 are hand-picked live shapes (decode pinned-B&B, heterogeneous
// prefix, N=20 prefix, budget-edge B&B); the rest sweep random shapes across
// every reachable 64-solver tier: pinned B&B / unpinned B&B / dp_prefix /
// greedy, with and without the consequence terms (place/evict/recon), bank
// contention, subprep, clamp, m2 and fixed overhead.
inline Instance make_instance(int inst) {
  namespace gl = layerstorm::gpu_loader;
  Instance out;
  std::mt19937_64 rng(0x5EEDF00DULL +
                      static_cast<uint64_t>(inst) * 0x9E3779B97F4A7C15ULL);
  auto uni = [&](double lo, double hi) {
    return std::uniform_real_distribution<double>(lo, hi)(rng);
  };
  auto uniI = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };

  int N, M, B;
  switch (inst) {
    case 0:  N = 8;  M = 4;  B = 4; break;  // decode shape, pinned B&B tier
    case 1:  N = 8;  M = 10; B = 4; break;  // heterogeneous prefix tier
    case 2:  N = 20; M = 10; B = 4; break;  // large-N prefix tier
    case 3:  N = 5;  M = 16; B = 4; break;  // enumeration-budget edge (16^5)
    default: N = uniI(1, 20); M = uniI(1, 10); B = uniI(1, 4); break;
  }
  const bool tie_heavy = (inst % 3 == 0);
  auto rate = [&](double lo, double hi) {
    return tie_heavy ? static_cast<double>(
                           uniI(static_cast<int>(lo), static_cast<int>(hi)))
                     : uni(lo, hi);
  };
  const bool with_consequence = (inst % 2 == 1);
  const bool with_contention  = (inst % 4 == 2);
  const bool with_pins        = (inst % 3 != 2) || inst == 0;
  const bool with_m2          = with_pins && (inst % 5 == 4 || inst == 0);

  gl::LoaderConstants& k = out.k;
  k.source       = "golden64";
  k.expert_bytes = 24772992.0;
  k.num_devices  = M;
  k.num_banks    = B;
  k.ncf          = {0.0, 1.0, 1.2, 1.5};
  if (inst % 2 == 0) k.fixed_overhead_us = rate(0.0, 40.0);
  for (int d = 0; d < M; ++d) {
    gl::DeviceConstants dc;
    dc.position    = d;
    dc.numa_node   = d % B;
    dc.xfer_lat_us = rate(0.0, 10.0);
    dc.compute     = {rate(50.0, 200.0), rate(0.0, 60.0), uniI(1, 4)};
    if (with_consequence) {
      dc.recon_overhead_us = (uniI(0, 1) ? rate(0.0, 30.0) : 0.0);
      dc.recon_added_us    = (uniI(0, 1) ? rate(0.0, 20.0) : 0.0);
    }
    k.devices.push_back(dc);
  }
  for (int b = 0; b < B; ++b) {
    gl::BankConstants bc;
    bc.node       = b;
    bc.egress_us  = rate(0.0, 400.0);
    bc.contention = with_contention ? uni(0.0, 1.0) : 1.0;
    k.banks.push_back(bc);
  }
  k.matrix.assign(B, std::vector<gl::TransferCell>(M));
  for (int b = 0; b < B; ++b)
    for (int d = 0; d < M; ++d)
      k.matrix[b][d] = gl::TransferCell{rate(100.0, 1000.0), (d % B == b ? 1 : 2),
                                        rate(0.0, 10.0)};

  gl::SolveRequest& req = out.req;
  req.num_devices = M;
  req.num_experts = N;
  req.clamp_place = (uniI(0, 1) == 1);
  req.bank_of.resize(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) req.bank_of[static_cast<size_t>(i)] = uniI(0, B - 1);
  req.cached.assign(static_cast<size_t>(N) * M, 0);
  if (inst == 0) {
    for (int i = 0; i < 6; ++i)
      req.cached[static_cast<size_t>(i) * M + (i % M)] = 1;  // 6 hits, 2 misses
  } else {
    for (auto& c : req.cached) c = (uniI(0, 3) == 0) ? 1 : 0;  // ~25% cached
  }
  if (uniI(0, 1)) {
    req.subprep_us.resize(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) req.subprep_us[static_cast<size_t>(i)] = rate(0.0, 50.0);
  }
  if (with_consequence) {
    req.place.assign(static_cast<size_t>(N) * M, 0.0);
    for (auto& p : req.place) p = uni(-30.0, 200.0);
    req.evict_cum.resize(static_cast<size_t>(M));
    for (int d = 0; d < M; ++d) {
      if (uniI(0, 1)) { req.evict_cum[static_cast<size_t>(d)].clear(); continue; }
      const int len = uniI(1, N + 1);
      req.evict_cum[static_cast<size_t>(d)].assign(static_cast<size_t>(len) + 1, 0.0);
      double acc = 0.0;
      for (int n = 1; n <= len; ++n) {
        acc += uni(0.0, 15.0);
        req.evict_cum[static_cast<size_t>(d)][static_cast<size_t>(n)] = acc;
      }
    }
  }
  if (with_pins) {
    // Dispatcher policy: pin every cache hit to its (first) resident device.
    req.pinned.assign(static_cast<size_t>(N), -1);
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        if (req.cached[static_cast<size_t>(i) * M + j]) {
          req.pinned[static_cast<size_t>(i)] = j;
          break;
        }
  }
  if (with_m2) {
    req.m2 = true;
    req.m2_s.resize(static_cast<size_t>(M));
    req.m2_o0.resize(static_cast<size_t>(M));
    req.m2_oc.resize(static_cast<size_t>(M));
    req.m2_hsat.resize(static_cast<size_t>(M));
    req.m2_cpw.resize(static_cast<size_t>(M));
    for (int d = 0; d < M; ++d) {
      req.m2_s[static_cast<size_t>(d)]    = uni(0.5, 1.5);
      req.m2_o0[static_cast<size_t>(d)]   = uni(0.0, 200.0);
      req.m2_oc[static_cast<size_t>(d)]   = uni(0.0, 80.0);
      req.m2_hsat[static_cast<size_t>(d)] = uni(1.0, 12.0);
      req.m2_cpw[static_cast<size_t>(d)]  = uni(1.0, 2.5);
    }
    req.m2_gc = uni(0.0, 1.0);
  }
  return out;
}

inline constexpr int kNumInstances = 48;

}  // namespace golden64

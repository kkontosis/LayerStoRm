#pragma once
// GPU loader calibration: benchmark the transfer/latency/contention constants
// the loader cost model consumes (spec/GPU_LOADER_MODEL.md §6, ticket I8b).
//
// CUDA-free: drives the DeviceBackend interface (timed H2D via create_timing_event
// / record_event / event_elapsed_ms) and NumaManager (NUMA-bound host buffers,
// pinned via host_register_pinned_portable). Raw CUDA stays inside the concrete
// backend's .cu — INV-GPU-1 holds.
//
// Fills: per-device ingest, per-bank egress, the GPU×NUMA transfer matrix,
// per-(bank,device) latency, ncf, same-bank two-device contention, the per-device
// compute curve (compute_j via the ExpertDevice grouped GEMM, TD-LOADER-COMPUTE-PARALLELISM),
// and per-device reconciliation overhead/added time (recon_*). The compute + recon
// passes need an ExpertDevice per GPU; pass them alongside the DeviceBackends (paired
// by GpuRef.position). When absent (transfer-only call), those fields stay at defaults.
#include <string>
#include <vector>

#include "core/gpu_loader/loader_constants.h"

namespace layerstorm::compute { class DeviceBackend; class ExpertDevice; }
namespace layerstorm::memory { class NumaManager; }

namespace layerstorm::gpu_loader {

struct CalibrationConfig {
  double expert_bytes        = 24772992.0;  // transfer unit (V3.2 NVFP4 expert slot)
  int    warmup_iters        = 5;
  int    timed_iters         = 30;          // timed transfers per (bank,device) cell
  double latency_probe_bytes = 4096.0;      // tiny transfer → latency floor
  bool   measure_contention  = true;        // same-bank two-device pass

  // ── Compute-curve pass (compute_j, §2.3) — needs an ExpertDevice per GPU ──
  bool   measure_compute     = false;        // sweep grouped-GEMM expert count → fit (a,b,P)
  // Expert FFN dims for the synthetic grouped GEMM (V3.2 routed gate_up default:
  // N = 2*moe_intermediate_size, K = hidden_size). The curve's a/b/P are dim-
  // dependent — these MUST match the deployed model for the constants to transfer.
  int    compute_N           = 2 * 2048;     // gate_up output dim (2 * moe_intermediate_size)
  int    compute_K           = 7168;         // hidden_size (input dim)
  int    compute_tokens      = 1;            // tokens per expert (decode M==1 hot path)
  // Down-projection dims for the full-chain compute pass (compute_full=true). The
  // real routed FFN is permute → gate_up GEMM (N1=compute_N, K1=compute_K) → SwiGLU
  // → down GEMM (N2=hidden_size, K2=moe_intermediate_size) → unpermute. The down
  // GEMM's N2/K2 differ from gate_up: N2 = K1 (= hidden_size), K2 = N1/2 (= the
  // SwiGLU output = moe_intermediate_size, half of the fused gate_up width). 0 here
  // → DERIVE from compute_N/K (N2=compute_K, K2=compute_N/2); set non-zero (or via
  // env LS_CALIB_DOWN_N / LS_CALIB_DOWN_K) to override for non-standard models.
  // These are NOT persisted to JSON (the gate_up compute_N/K already key the file
  // to the model; the down dims are a deterministic function of them by default).
  int    compute_down_N      = 0;            // down GEMM output dim (0 → derive = compute_K)
  int    compute_down_K      = 0;            // down GEMM input dim  (0 → derive = compute_N/2)
  // Compute-pass mode. false (default) → legacy gate_up-only grouped-GEMM timing
  // (a thin lower bound on the per-layer MoE compute). true → time the FULL routed
  // FFN decode chain (permute → gate_up GEMM → SwiGLU → down GEMM → unpermute) so
  // the fitted (a,b,P) reflect the real ~365 µs/layer count-independent cost, not
  // just the ~38–108 µs gate_up GEMM (INV-LOADER-CAL-5; TD-LOADER-COMPUTE-PARALLELISM).
  bool   compute_full        = true;
  // Expert-count sweep. The fit resolves P only when the SM-saturation knee falls in
  // the INTERIOR of the swept range: if the max count IS the knee, the grid-search
  // pins P to the boundary and the b/P split is soft. We therefore sweep to 64
  // (= loader_solver kMaxExperts), well past the observed ~32 saturation on the 5090s,
  // so P lands in the interior when a real knee exists. The extra 4 timed points
  // (40/48/56/64) cost ~4 grouped-GEMM launches/device — negligible vs the GPU-test
  // budget. See spec/GPU_LOADER_MODEL.md §6 / TD-LOADER-COMPUTE-PARALLELISM.
  std::vector<int> compute_counts{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64};

  // ── Reconciliation pass (recon_overhead/recon_added, §3.4) ──
  bool   measure_recon       = false;        // time the TP reduce/broadcast proxy
  // Routed-output reconciliation payload per token (bytes): hidden_size * dtype.
  // The collective moves this much per participant; recon_added scales with it.
  double recon_payload_bytes = 7168.0 * 2.0; // hidden_size * sizeof(bf16)

  // Total rotating host footprint per bank (bytes). The bench cycles the source
  // across distinct sub-regions of one big NUMA buffer so reads MISS the HBM
  // cache → DDR-bound rates + real contention. 0 (or < expert_bytes) → single
  // buffer (cache-hot, fast). To defeat the cache, set > the HBM working set
  // (~16 GB/cluster on Xeon Max). See spec/GPU_LOADER_MODEL.md §6 / TD.
  double footprint_bytes     = 0.0;
  // NUMA nodes to treat as banks. Empty → use numa.gpu_attached_nodes().
  std::vector<int> bank_nodes;

  // ── HBM banks (TD-NUMA-HBM-BANKS / INV-NUMA-HOSTBANK) ──
  // Append the detected CPU-less HBM / device-memory nodes to the bank set
  // (banks = numa.all_banks_including_hbm() instead of all_memory_nodes()).
  // Their per-bank footprint is capped (below) so the rotating buffer can never
  // over-commit the small node (the original CONSTRAINT_MEMORY_POLICY OOM), and
  // each HBM bank is measured from a thread pinned to its nearest CPU node.
  // No-op when the LAYERSTORM_DETECT_HBM CMake gate is off (hbm_nodes() empty)
  // or the box has no such nodes. cfg.bank_nodes still overrides verbatim.
  bool   include_hbm_banks   = true;
  // Per-HBM-bank footprint cap: min(footprint_bytes, node_free * this frac),
  // floor one expert chunk. DDR banks are NEVER capped (behavior preserved).
  double hbm_footprint_frac  = 0.5;
};

// How the engine obtains the constants (config-driven; see I8b engine hook).
//   kLoaded — load from a JSON path, no benchmarking.
//   kQuick  — single-buffer, few iters: fast, but cache-hot (dev/test/CI).
//   kFull   — large rotating footprint: DDR-bound, representative (production init).
enum class CalibrationMode { kLoaded, kQuick, kFull };

CalibrationMode   parse_calibration_mode(const std::string& s);  // unknown → kQuick
const char*       calibration_mode_name(CalibrationMode);
CalibrationConfig quick_calibration_config();   // footprint 0, warmup 3, timed 20; compute/recon OFF
CalibrationConfig full_calibration_config();    // footprint 32 GiB, warmup 5, timed 30; compute/recon ON
CalibrationConfig config_for_mode(CalibrationMode);  // kLoaded falls back to quick (unused on load)

// Robust fit of the batch-step compute curve compute_j(c) = a*c + b*ceil(c/P) to
// measured samples (counts[k] → us[k]); see loader_solver.h::compute_us. Method:
// grid-search P over [1, max_count]; for each P solve the 2×2 normal equations for
// (a,b) by least squares; keep the P with min SSE (tie → smaller P). a is the
// per-expert (intra-batch) marginal, b the fixed per-batch step. Robust to a single
// flat/noisy point; returns {0,0,1} for empty/degenerate input. CPU-only, pure —
// unit-tested against synthetic curves. `counts` and `us` must be equal length.
ComputeCurve fit_compute_curve(const std::vector<int>& counts,
                               const std::vector<double>& us);

// Config-driven precedence (engine init):
//   kLoaded → load `path`; if it is absent or unreadable, fall back to a FULL
//             calibration and WRITE `path` (self-healing first run).
//   kQuick / kFull → calibrate with the preset; if `path` is non-empty, write it.
// `backends`/`numa` are touched only when a calibration actually runs.
LoaderConstants load_or_calibrate(CalibrationMode mode, const std::string& path,
                                  const std::vector<compute::DeviceBackend*>& backends,
                                  memory::NumaManager& numa,
                                  const std::vector<compute::ExpertDevice*>& experts = {});

// Run the calibration across all devices × banks, returning a filled
// LoaderConstants (source = "calibrated"). `backends[k]` is the DeviceBackend for
// the GPU at position k (0..M-1); they must each expose timing events
// (create_timing_event() != nullptr) — else the corresponding cells are left zero
// and a warning is logged. `experts[k]` (optional, paired by index with backends)
// is the ExpertDevice for the same GPU — required by the compute pass
// (cfg.measure_compute); when empty/null the compute curve is left at defaults.
// The recon pass (cfg.measure_recon) brings up a real NCCL communicator over the
// backends' GPUs (via parallelism::CollectiveBackend) and times the routed-output
// allreduce; it needs >=2 GPUs, else recon_* stay at defaults.
// Allocations are released before return.
LoaderConstants calibrate(const std::vector<compute::DeviceBackend*>& backends,
                          memory::NumaManager& numa,
                          const CalibrationConfig& cfg = {},
                          const std::vector<compute::ExpertDevice*>& experts = {});

}  // namespace layerstorm::gpu_loader

// Integration test: run the GPU loader calibration on real silicon (all visible
// GPUs — set CUDA_VISIBLE_DEVICES to choose them), sanity-check the measured
// constants, exercise the rotating-footprint path, and round-trip through JSON.
// GPU test. Ticket: spec/tickets/I8b_LOADER_CALIBRATION.md.
#include "core/gpu_loader/loader_calibration.h"
#include "core/gpu_loader/loader_constants.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "compute/cuda_sm120_device_backend.h"
#include "config/config_parser.h"
#include "core/cuda_hardware_query.h"
#include "core/device_backend.h"
#include "core/expert_device.h"
#include "core/gpu_ref.h"
#include "core/hardware_detect.h"
#include "core/memory/numa_manager.h"

namespace cfg = layerstorm::config;
namespace gl = layerstorm::gpu_loader;
using layerstorm::compute::DeviceBackend;
using layerstorm::compute::ExpertDevice;
using layerstorm::memory::NumaManager;

namespace {

cfg::GpuType type_from_name(const std::string& name) {
  if (name.find("5080") != std::string::npos) return cfg::GpuType::rtx5080;
  return cfg::GpuType::rtx5090;
}

cfg::HardwareConfig detect_hw() {
  cfg::HardwareConfig hw;
  const int n = layerstorm::core::query_gpu_count();
  for (int id = 0; id < n; ++id) {
    const auto info = layerstorm::core::query_gpu_info(id);
    cfg::GpuConfig g;
    g.numa_node = layerstorm::core::read_numa_node(info.pci_bus_id);
    g.ref = cfg::GpuRef{id, id, type_from_name(info.device_name)};
    hw.gpus.push_back(g);
  }
  return hw;
}

}  // namespace

class GpuLoaderCalibration : public ::testing::Test {
 protected:
  void SetUp() override {
    n_ = layerstorm::core::query_gpu_count();
    if (n_ < 1) GTEST_SKIP() << "no CUDA devices visible";
    hw_   = detect_hw();
    numa_ = std::make_unique<NumaManager>(hw_);
    for (const auto& g : hw_.gpus) {
      owned_.push_back(layerstorm::compute::make_cuda_sm120_device_backend(g.ref));
      backends_.push_back(owned_.back().get());
      owned_experts_.push_back(layerstorm::compute::make_cuda_sm120_expert_device(g.ref));
      experts_.push_back(owned_experts_.back().get());
    }
  }

  void check_structure(const gl::LoaderConstants& c) const {
    EXPECT_EQ(c.num_devices, n_);
    ASSERT_GE(c.num_banks, 1) << "no GPU-attached NUMA banks (numa_node sysfs read?)";
    ASSERT_EQ(static_cast<int>(c.matrix.size()), c.num_banks);
    for (const auto& row : c.matrix) ASSERT_EQ(static_cast<int>(row.size()), c.num_devices);
    for (int b = 0; b < c.num_banks; ++b)
      for (int d = 0; d < c.num_devices; ++d) {
        EXPECT_GT(c.matrix[b][d].rate_us, 0.0) << "bank " << b << " dev " << d;
        EXPECT_GT(c.matrix[b][d].lat_us, 0.0);
        EXPECT_LT(c.matrix[b][d].lat_us, c.matrix[b][d].rate_us);
      }
  }

  int n_ = 0;
  cfg::HardwareConfig hw_;
  std::unique_ptr<NumaManager> numa_;
  std::vector<std::unique_ptr<DeviceBackend>> owned_;
  std::vector<DeviceBackend*> backends_;
  std::vector<std::unique_ptr<ExpertDevice>> owned_experts_;
  std::vector<ExpertDevice*> experts_;
};

// Quick mode (single-buffer, cache-hot): mechanism + JSON round-trip on real data.
TEST_F(GpuLoaderCalibration, QuickCalibrateAndRoundTrip) {
  const gl::LoaderConstants c = gl::calibrate(backends_, *numa_, gl::quick_calibration_config());
  check_structure(c);
  EXPECT_EQ(c.source, "calibrated");
  ASSERT_GE(static_cast<int>(c.ncf.size()), 3);
  EXPECT_GE(c.ncf[2], 0.9) << "ncf2=" << c.ncf[2];

  for (int d = 0; d < c.num_devices; ++d)
    fprintf(stderr, "[calib quick] dev %d node %d  xfer_lat=%.1fus\n", c.devices[d].position,
            c.devices[d].numa_node, c.devices[d].xfer_lat_us);
  for (int b = 0; b < c.num_banks; ++b)
    fprintf(stderr, "[calib quick] bank node %d  egress=%.1fus  contention=%.2f\n", c.banks[b].node,
            c.banks[b].egress_us, c.banks[b].contention);
  fprintf(stderr, "[calib quick] ncf2=%.3f  contention_samples=%zu\n", c.ncf[2], c.contention.size());

  const std::string path = std::string(testing::TempDir()) + "/loader_calib.json";
  gl::save(c, path);
  const gl::LoaderConstants reloaded = gl::load(path);
  EXPECT_EQ(c, reloaded);
  std::remove(path.c_str());
}

// Rotating footprint path runs on hardware (modest 1 GiB to fit the test timeout;
// production "full" uses ~32 GiB). Validates the rotation mechanism + sane rates.
TEST_F(GpuLoaderCalibration, RotatingFootprintRuns) {
  gl::CalibrationConfig ccfg = gl::quick_calibration_config();
  ccfg.footprint_bytes    = 1.0 * 1024 * 1024 * 1024;  // 1 GiB rotation (fast)
  ccfg.measure_contention = false;                      // covered by the quick test
  const gl::LoaderConstants c = gl::calibrate(backends_, *numa_, ccfg);
  check_structure(c);
  for (int b = 0; b < c.num_banks; ++b)
    fprintf(stderr, "[calib rot1G] bank node %d  egress=%.1fus (%.1f GB/s)\n", c.banks[b].node,
            c.banks[b].egress_us,
            c.banks[b].egress_us > 0 ? (c.expert_bytes / 1e9) / (c.banks[b].egress_us / 1e6) : 0.0);
}

// One-off DDR-accurate full calibration. Env-gated (skipped in normal sweeps —
// the ~32-48 GiB pinning blows the 30s ctest timeout); run the binary directly:
//   LS_RUN_FULL_CALIB=1 LS_CALIB_FOOTPRINT_GB=48 LS_CALIB_OUT=<path> \
//   CUDA_VISIBLE_DEVICES=0,1,2,3 ./layerstorm_unit_tests \
//     --gtest_filter='GpuLoaderCalibration.FullCalibrateAndWrite'
TEST_F(GpuLoaderCalibration, FullCalibrateAndWrite) {
  if (!std::getenv("LS_RUN_FULL_CALIB"))
    GTEST_SKIP() << "set LS_RUN_FULL_CALIB=1 to run the one-off DDR calibration";

  gl::CalibrationConfig ccfg = gl::full_calibration_config();
  if (const char* fg = std::getenv("LS_CALIB_FOOTPRINT_GB"))
    ccfg.footprint_bytes = std::atof(fg) * 1024.0 * 1024.0 * 1024.0;
  // Full = everything: DDR transfer + compute curve + recon in one file.
  ccfg.measure_compute = true;
  ccfg.measure_recon   = true;
  if (const char* n = std::getenv("LS_CALIB_N")) ccfg.compute_N = std::atoi(n);
  if (const char* k = std::getenv("LS_CALIB_K")) {
    ccfg.compute_K = std::atoi(k);
    ccfg.recon_payload_bytes = static_cast<double>(ccfg.compute_K) * 2.0;
  }
  // Full routed-FFN chain dims (down GEMM). Default: derive from gate_up
  // (N2=compute_K, K2=compute_N/2). Override for non-standard models.
  if (const char* dn = std::getenv("LS_CALIB_DOWN_N")) ccfg.compute_down_N = std::atoi(dn);
  if (const char* dk = std::getenv("LS_CALIB_DOWN_K")) ccfg.compute_down_K = std::atoi(dk);
  // Compute-pass mode: full chain by default (INV-LOADER-CAL-5). LS_CALIB_GATE_UP_ONLY=1
  // reverts to the legacy gate_up-only grouped-GEMM lower bound.
  if (std::getenv("LS_CALIB_GATE_UP_ONLY")) ccfg.compute_full = false;
  fprintf(stderr, "[full] footprint=%.0f GiB, %d devices, compute N=%d K=%d (mode=%s)\n",
          ccfg.footprint_bytes / (1024.0 * 1024 * 1024), n_, ccfg.compute_N, ccfg.compute_K,
          ccfg.compute_full ? "full-chain" : "gate_up-only");

  const gl::LoaderConstants c = gl::calibrate(backends_, *numa_, ccfg, experts_);
  check_structure(c);
  for (int d = 0; d < c.num_devices; ++d) {
    const auto& cc = c.devices[d].compute;
    fprintf(stderr, "[full] dev %d compute a=%.4f us/expert b=%.3f us/batch P=%d | recon ovh=%.2f added=%.2f us\n",
            c.devices[d].position, cc.a_us, cc.b_us, cc.P,
            c.devices[d].recon_overhead_us, c.devices[d].recon_added_us);
  }

  auto gbps = [&](double us) { return us > 0 ? (c.expert_bytes / 1e9) / (us / 1e6) : 0.0; };
  for (int b = 0; b < c.num_banks; ++b) {
    for (int d = 0; d < c.num_devices; ++d)
      fprintf(stderr, "[full] bank %d → dev %d  tier %d  %.1f us (%.1f GB/s)\n", c.banks[b].node,
              c.devices[d].position, c.matrix[b][d].tier, c.matrix[b][d].rate_us,
              gbps(c.matrix[b][d].rate_us));
    fprintf(stderr, "[full] bank %d egress=%.1f us (%.1f GB/s) contention=%.2f\n", c.banks[b].node,
            c.banks[b].egress_us, gbps(c.banks[b].egress_us), c.banks[b].contention);
  }
  for (const auto& s : c.contention)
    fprintf(stderr, "[full] contention bank %d  dev %d+%d  cfg %d  solo %.1f dual %.1f  factor %.2f\n",
            s.bank, s.device_a, s.device_b, static_cast<int>(s.config), s.solo_us, s.dual_us, s.factor);
  fprintf(stderr, "[full] ncf2=%.3f\n", c.ncf[2]);

  const char* out = std::getenv("LS_CALIB_OUT");
  const std::string path = out ? out : (std::string(testing::TempDir()) + "/gpu_loader_full.json");
  gl::save(c, path);
  fprintf(stderr, "[full] wrote %s\n", path.c_str());
}

// Env-gated diagnostic driver: H2D-vs-CPU-DDR bus-contention variants
// (LS_CALIB_H2D_VS_DDR2HBM=1 and/or LS_CALIB_H2D_VS_CPUKERNEL=1 — see
// loader_calibration.cpp). Lightweight calibration-only path: synthetic pinned
// NUMA buffers + GPUs, no model load. The variant passes run inside calibrate()
// and log "[x2]"/"[x3]" tables; the returned constants are untouched by them.
// Run from the repo root (cpu_kernel_bench + prepack paths are cwd-relative):
//   LS_CALIB_H2D_VS_DDR2HBM=1 LS_CALIB_H2D_VS_CPUKERNEL=1 \
//   LS_CALIB_FOOTPRINT_GB=4 CUDA_DEVICE_ORDER=PCI_BUS_ID \
//   CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/tests/layerstorm_unit_tests \
//     --gtest_filter='GpuLoaderCalibration.H2dContentionVariants'
TEST_F(GpuLoaderCalibration, H2dContentionVariants) {
  if (!std::getenv("LS_CALIB_H2D_VS_DDR2HBM") && !std::getenv("LS_CALIB_H2D_VS_CPUKERNEL"))
    GTEST_SKIP() << "set LS_CALIB_H2D_VS_DDR2HBM=1 / LS_CALIB_H2D_VS_CPUKERNEL=1 to run";

  gl::CalibrationConfig ccfg = gl::quick_calibration_config();
  ccfg.footprint_bytes = 4.0 * 1024 * 1024 * 1024;  // DDR-bound (working set >> LLC)
  if (const char* fg = std::getenv("LS_CALIB_FOOTPRINT_GB"))
    ccfg.footprint_bytes = std::atof(fg) * 1024.0 * 1024.0 * 1024.0;
  ccfg.measure_contention = false;  // GPU-GPU contention covered by the main passes
  const gl::LoaderConstants c = gl::calibrate(backends_, *numa_, ccfg);
  check_structure(c);
}

// One-off compute-curve + reconciliation calibration across all visible GPUs.
// Env-gated (the grouped-GEMM sweep + warmups exceed the 30s ctest timeout); run
// the binary directly. The compute curve is fit at the model's expert FFN dims —
// override via LS_CALIB_N / LS_CALIB_K (defaults: V3.2 gate_up N=4096, K=7168):
//   LS_RUN_COMPUTE_CALIB=1 LS_CALIB_N=4096 LS_CALIB_K=7168 \
//   LS_CALIB_OUT=<adjacent-to-weights>/gpu_loader_calibration.json \
//   CUDA_VISIBLE_DEVICES=0,1,2,3 ./layerstorm_unit_tests \
//     --gtest_filter='GpuLoaderCalibration.ComputeAndReconCalibrate'
TEST_F(GpuLoaderCalibration, ComputeAndReconCalibrate) {
  if (!std::getenv("LS_RUN_COMPUTE_CALIB"))
    GTEST_SKIP() << "set LS_RUN_COMPUTE_CALIB=1 to run the compute+recon calibration";

  gl::CalibrationConfig ccfg = gl::quick_calibration_config();  // skip the heavy footprint pass
  ccfg.measure_compute    = true;
  ccfg.measure_recon      = true;
  ccfg.measure_contention = false;  // transfer side covered elsewhere; focus on compute/recon
  if (const char* n = std::getenv("LS_CALIB_N")) ccfg.compute_N = std::atoi(n);
  if (const char* k = std::getenv("LS_CALIB_K")) {
    ccfg.compute_K = std::atoi(k);
    ccfg.recon_payload_bytes = static_cast<double>(ccfg.compute_K) * 2.0;
  }
  // Full routed-FFN chain down-GEMM dims (default: derive N2=compute_K, K2=compute_N/2).
  if (const char* dn = std::getenv("LS_CALIB_DOWN_N")) ccfg.compute_down_N = std::atoi(dn);
  if (const char* dk = std::getenv("LS_CALIB_DOWN_K")) ccfg.compute_down_K = std::atoi(dk);
  // Full chain (permute→gate_up→SwiGLU→down→unpermute) by default; LS_CALIB_GATE_UP_ONLY=1
  // reverts to the legacy gate_up-only grouped-GEMM lower bound (INV-LOADER-CAL-5).
  if (std::getenv("LS_CALIB_GATE_UP_ONLY")) ccfg.compute_full = false;
  fprintf(stderr, "[compute] N=%d K=%d tokens=%d, %d devices (mode=%s, down N=%d K=%d)\n",
          ccfg.compute_N, ccfg.compute_K, ccfg.compute_tokens, n_,
          ccfg.compute_full ? "full-chain" : "gate_up-only",
          ccfg.compute_down_N > 0 ? ccfg.compute_down_N : ccfg.compute_K,
          ccfg.compute_down_K > 0 ? ccfg.compute_down_K : ccfg.compute_N / 2);

  const gl::LoaderConstants c = gl::calibrate(backends_, *numa_, ccfg, experts_);
  check_structure(c);
  EXPECT_EQ(c.compute_N, ccfg.compute_N);
  EXPECT_EQ(c.compute_K, ccfg.compute_K);

  for (int d = 0; d < c.num_devices; ++d) {
    const auto& cc = c.devices[d].compute;
    fprintf(stderr, "[compute] dev %d  a=%.4f us/expert  b=%.3f us/batch  P=%d  | recon ovh=%.2f "
            "added=%.2f us\n",
            c.devices[d].position, cc.a_us, cc.b_us, cc.P, c.devices[d].recon_overhead_us,
            c.devices[d].recon_added_us);
    EXPECT_GT(cc.a_us, 0.0) << "dev " << d << " compute slope not measured";
    EXPECT_GE(cc.P, 1);
    EXPECT_GE(c.devices[d].recon_overhead_us, 0.0);
    EXPECT_GE(c.devices[d].recon_added_us, 0.0);
  }

  // Recon is now a REAL cross-GPU NCCL allreduce (routed-output reconciliation, 13c-8),
  // not a single-GPU proxy. With >=2 visible GPUs the collective runs and must produce a
  // non-zero fixed floor (recon_overhead) on every participant; the per-device overhead is
  // the same value (the §3.4 max_j rollup). <2 GPUs → no link to cross → left at defaults.
  if (n_ >= 2) {
    for (int d = 0; d < c.num_devices; ++d) {
      EXPECT_GT(c.devices[d].recon_overhead_us, 0.0)
          << "dev " << d << " recon collective floor not measured (NCCL allreduce)";
      EXPECT_DOUBLE_EQ(c.devices[d].recon_overhead_us, c.devices[0].recon_overhead_us)
          << "recon_overhead must be uniform across participants (max_j rollup)";
    }
  }

  const char* out = std::getenv("LS_CALIB_OUT");
  const std::string path =
      out ? out : (std::string(testing::TempDir()) + "/gpu_loader_compute.json");
  gl::save(c, path);
  const gl::LoaderConstants reloaded = gl::load(path);
  EXPECT_EQ(c, reloaded);
  EXPECT_TRUE(gl::compute_dims_match(reloaded, ccfg.compute_N, ccfg.compute_K));
  fprintf(stderr, "[compute] wrote %s\n", path.c_str());
}

// ── CPU-only: full-chain compute-pass config contract (no GPU) ──────────────
// The full routed-FFN chain (INV-LOADER-CAL-5) times permute → gate_up GEMM
// (N1=compute_N, K1=compute_K) → SwiGLU → down GEMM (N2,K2) → unpermute. The down
// dims default to DERIVED from gate_up: N2 = compute_K (= hidden_size), K2 =
// compute_N/2 (= moe_intermediate_size = the SwiGLU output width). This validates
// the load-bearing shape arithmetic the human can't easily check blind on GPU.
TEST(GpuLoaderFullChainConfig, DefaultsAndDerivedDownDims) {
  // full_calibration_config() must default to the full-chain mode + V3.2 gate_up dims.
  const gl::CalibrationConfig full = gl::full_calibration_config();
  EXPECT_TRUE(full.compute_full) << "full mode must time the whole routed-FFN chain by default";
  EXPECT_TRUE(full.measure_compute);
  EXPECT_EQ(full.compute_N, 2 * 2048) << "gate_up N1 = 2*moe_intermediate_size";
  EXPECT_EQ(full.compute_K, 7168) << "gate_up K1 = hidden_size";
  EXPECT_EQ(full.compute_tokens, 1) << "decode M==1 hot path";

  // Derive contract: 0 → N2 = compute_K, K2 = compute_N/2.
  EXPECT_EQ(full.compute_down_N, 0) << "0 sentinel → derive from gate_up";
  EXPECT_EQ(full.compute_down_K, 0);
  const int derived_down_N = (full.compute_down_N > 0) ? full.compute_down_N : full.compute_K;
  const int derived_down_K = (full.compute_down_K > 0) ? full.compute_down_K : full.compute_N / 2;
  EXPECT_EQ(derived_down_N, 7168) << "down N2 = hidden_size";
  EXPECT_EQ(derived_down_K, 2048) << "down K2 = moe_intermediate_size = SwiGLU output width";
  // SwiGLU consumes the fused gate_up output [.,N1] as [.,2*d] → d = N1/2 = K2.
  EXPECT_EQ(full.compute_N / 2, derived_down_K) << "SwiGLU d must equal down K2";

  // gate_up N1 must be even (it is 2*moe_intermediate_size) so SwiGLU d = N1/2 is exact.
  EXPECT_EQ(full.compute_N % 2, 0);
}

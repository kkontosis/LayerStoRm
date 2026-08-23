// Unit tests for the GPU loader constants data model + JSON (CPU-only, no GPU).
// Ticket: spec/tickets/I8b_LOADER_CALIBRATION.md.
#include "core/gpu_loader/loader_constants.h"
#include "core/gpu_loader/loader_calibration.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/gpu_loader/loader_solver.h"  // compute_us (oracle for the fit)

#include "config/config_parser.h"
#include "core/device_backend.h"
#include "core/memory/numa_manager.h"

using namespace layerstorm::gpu_loader;

namespace {

// A representative calibration result: 2x 5090 on nodes 2/3, 4 DDR NUMA banks +
// 1 CPU-less HBM bank (TD-NUMA-HBM-BANKS), numbers anchored to the measured
// envelope (≈557 µs local / 757 µs cross-NUMA, ~0.15 ms/expert compute) — see
// spec/GPU_LOADER_MODEL.md §6.
LoaderConstants make_sample() {
  LoaderConstants c;
  c.version      = kLoaderConstantsVersion;
  c.source       = "calibrated";
  c.expert_bytes = 24772992.0;
  c.num_devices  = 2;
  c.num_banks    = 5;
  c.compute_N    = 4096;   // 2 * moe_intermediate_size (V3.2)
  c.compute_K    = 7168;   // hidden_size
  c.compute_tokens = 1;
  c.recon_payload_bytes = 7168.0 * 2.0;
  c.ncf          = {0.0, 1.0, 1.36, 1.70};  // index 0 unused; tier1=local
  c.devices = {
      DeviceConstants{.position = 0, .numa_node = 2, .name = "RTX 5090", .uuid = "GPU-aaaa0000",
                      .xfer_lat_us = 8.5, .compute = ComputeCurve{0.15, 5.0, 4},
                      .recon_overhead_us = 12.0, .recon_added_us = 3.0},
      DeviceConstants{.position = 1, .numa_node = 3, .name = "RTX 5090", .uuid = "GPU-bbbb1111",
                      .xfer_lat_us = 9.0, .compute = ComputeCurve{0.15, 5.0, 4},
                      .recon_overhead_us = 12.0, .recon_added_us = 3.0},
  };
  c.banks = {
      BankConstants{2, 557.0, 1.00}, BankConstants{3, 557.0, 1.00},
      BankConstants{0, 757.0, 0.90}, BankConstants{1, 757.0, 0.90},
      // A CPU-less HBM bank (TD-NUMA-HBM-BANKS): tagged + CPU-affinity mapped.
      BankConstants{6, 610.0, 0.95, /*is_hbm=*/true, /*cpu_affinity_node=*/2},
  };
  c.matrix.assign(c.num_banks, std::vector<TransferCell>(c.num_devices));
  for (int b = 0; b < c.num_banks; ++b)
    for (int d = 0; d < c.num_devices; ++d)
      c.matrix[b][d] = TransferCell{557.0 + b * 10.0 + d, (b < 2 ? 1 : 2), 110.0};
  c.contention = {
      ContentionSample{2, 0, 1, ContentionConfig::kTwoBound, 557.0, 980.0, 1.76},
      ContentionSample{0, 0, 1, ContentionConfig::kTwoUnbound, 757.0, 1400.0, 1.85},
  };
  return c;
}

}  // namespace

TEST(GpuLoaderConstants, StringRoundTrip) {
  const LoaderConstants a = make_sample();
  const std::string s = to_json_string(a);
  const LoaderConstants b = from_json_string(s);
  EXPECT_EQ(a, b);
}

TEST(GpuLoaderConstants, FileRoundTrip) {
  const LoaderConstants a = make_sample();
  const std::string path = std::string(testing::TempDir()) + "/loader_constants_test.json";
  save(a, path);
  const LoaderConstants b = load(path);
  EXPECT_EQ(a, b);
  std::remove(path.c_str());
}

TEST(GpuLoaderConstants, MatrixDimensionsConsistent) {
  const LoaderConstants a = make_sample();
  ASSERT_EQ(static_cast<int>(a.matrix.size()), a.num_banks);
  for (const auto& row : a.matrix)
    EXPECT_EQ(static_cast<int>(row.size()), a.num_devices);
  ASSERT_EQ(static_cast<int>(a.devices.size()), a.num_devices);
  ASSERT_EQ(static_cast<int>(a.banks.size()), a.num_banks);
}

TEST(GpuLoaderConstants, MissingOptionalFieldsDefault) {
  // A minimal JSON (older/sparse file) must parse with sane defaults, not throw.
  const LoaderConstants c = from_json_string(R"({"version":1,"num_devices":2,"num_banks":4})");
  EXPECT_EQ(c.version, 1);
  EXPECT_EQ(c.num_devices, 2);
  EXPECT_EQ(c.num_banks, 4);
  EXPECT_TRUE(c.devices.empty());
  EXPECT_TRUE(c.contention.empty());
}

TEST(GpuLoaderConstants, PreHbmBankDefaultsToNonHbm) {
  // A pre-HBM bank entry (no is_hbm / cpu_affinity_node keys) must load as a
  // plain DDR bank — false / -1 (backward compatibility, TD-NUMA-HBM-BANKS).
  const LoaderConstants c = from_json_string(
      R"({"version":2,"num_devices":1,"num_banks":1,)"
      R"("banks":[{"node":2,"egress_us":557.0,"contention":1.0}]})");
  ASSERT_EQ(c.banks.size(), 1u);
  EXPECT_FALSE(c.banks[0].is_hbm);
  EXPECT_EQ(c.banks[0].cpu_affinity_node, -1);
}

TEST(GpuLoaderConstants, CalibrationModeParseRoundTrip) {
  EXPECT_EQ(parse_calibration_mode("loaded"), CalibrationMode::kLoaded);
  EXPECT_EQ(parse_calibration_mode("quick"), CalibrationMode::kQuick);
  EXPECT_EQ(parse_calibration_mode("full"), CalibrationMode::kFull);
  EXPECT_EQ(parse_calibration_mode("bogus"), CalibrationMode::kQuick);  // unknown → quick
  EXPECT_STREQ(calibration_mode_name(CalibrationMode::kFull), "full");
}

// load_or_calibrate's LOAD path needs no GPU: an existing file is loaded verbatim
// (backends/numa untouched). Uses an empty NumaManager + empty backends.
TEST(GpuLoaderConstants, LoadOrCalibrateLoadsExistingFile) {
  const LoaderConstants a = make_sample();
  const std::string path = std::string(testing::TempDir()) + "/loader_or_calib.json";
  save(a, path);

  layerstorm::config::HardwareConfig hw;  // empty: no GPUs
  layerstorm::memory::NumaManager numa(hw);
  const std::vector<layerstorm::compute::DeviceBackend*> no_backends;

  const LoaderConstants loaded = load_or_calibrate(CalibrationMode::kLoaded, path, no_backends, numa);
  EXPECT_EQ(a, loaded);
  std::remove(path.c_str());
}

// Missing file in kLoaded mode → falls back to a (here empty, no-GPU) calibration
// and WRITES the file (self-healing first run).
TEST(GpuLoaderConstants, LoadOrCalibrateMissingFileProducesIt) {
  const std::string path = std::string(testing::TempDir()) + "/loader_produced.json";
  std::remove(path.c_str());
  ASSERT_FALSE(std::filesystem::exists(path));

  layerstorm::config::HardwareConfig hw;  // empty → calibrate() does no measurement
  layerstorm::memory::NumaManager numa(hw);
  const std::vector<layerstorm::compute::DeviceBackend*> no_backends;

  const LoaderConstants produced =
      load_or_calibrate(CalibrationMode::kLoaded, path, no_backends, numa);
  EXPECT_EQ(produced.source, "calibrated");
  EXPECT_TRUE(std::filesystem::exists(path)) << "missing-file fallback must write the file";
  std::remove(path.c_str());
}

// ── Compute-curve fit (compute_j, §2.3) ──────────────────────────────────────

// Synthetic t(c) generated from a KNOWN (a,b,P) via the solver's own compute_us;
// fit_compute_curve must recover them exactly (noise-free).
TEST(GpuLoaderComputeFit, RecoversKnownCurveExact) {
  const ComputeCurve truth{0.42, 7.5, 4};  // a=0.42us/expert, b=7.5us/batch, P=4
  const std::vector<int> counts{1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32};
  std::vector<double> us;
  for (int c : counts) us.push_back(compute_us(truth, c));

  const ComputeCurve fit = fit_compute_curve(counts, us);
  EXPECT_EQ(fit.P, truth.P);
  EXPECT_NEAR(fit.a_us, truth.a_us, 1e-6);
  EXPECT_NEAR(fit.b_us, truth.b_us, 1e-6);
  // And it reproduces the curve at every measured point.
  for (size_t i = 0; i < counts.size(); ++i)
    EXPECT_NEAR(compute_us(fit, counts[i]), us[i], 1e-6);
}

// P=1 (pure linear, no batch step) is the degenerate-but-valid case.
TEST(GpuLoaderComputeFit, RecoversLinearP1) {
  const ComputeCurve truth{0.30, 4.0, 1};  // every expert opens a batch → t = (a+b)*c
  const std::vector<int> counts{1, 2, 4, 8, 16};
  std::vector<double> us;
  for (int c : counts) us.push_back(compute_us(truth, c));
  const ComputeCurve fit = fit_compute_curve(counts, us);
  // a+b is identifiable; the (a,b) split for P=1 is not (both scale c), so check the
  // reconstructed curve, not the individual coefficients.
  EXPECT_EQ(fit.P, 1);
  for (size_t i = 0; i < counts.size(); ++i)
    EXPECT_NEAR(compute_us(fit, counts[i]), us[i], 1e-6);
}

// Mild measurement noise → P still recovered, curve within tolerance.
TEST(GpuLoaderComputeFit, RobustToNoise) {
  const ComputeCurve truth{0.50, 10.0, 8};
  const std::vector<int> counts{1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64};
  std::vector<double> us;
  // Deterministic ±2% zig-zag noise (no RNG dependence).
  for (size_t i = 0; i < counts.size(); ++i) {
    const double base = compute_us(truth, counts[i]);
    us.push_back(base * (1.0 + ((i % 2 == 0) ? 0.02 : -0.02)));
  }
  const ComputeCurve fit = fit_compute_curve(counts, us);
  EXPECT_EQ(fit.P, truth.P);
  EXPECT_NEAR(fit.a_us, truth.a_us, 0.15);
  EXPECT_NEAR(fit.b_us, truth.b_us, 3.0);
}

// A real saturation knee in the INTERIOR of the swept range (P*=32, sweep to 64)
// must be recovered — not pinned to the boundary. This is the Task-2 widening: the
// old sweep maxed at 32 (= the knee), so the grid-search couldn't tell P=32 from a
// soft slope; extending past the knee makes P=32 the unique min-SSE solution.
TEST(GpuLoaderComputeFit, RecoversInteriorKnee) {
  const ComputeCurve truth{0.20, 6.0, 32};  // knee at 32; sweep extends to 64
  const std::vector<int> counts{1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64};
  std::vector<double> us;
  for (int c : counts) us.push_back(compute_us(truth, c));
  const ComputeCurve fit = fit_compute_curve(counts, us);
  EXPECT_EQ(fit.P, truth.P) << "knee at P*=32 must be recovered, not the 64 boundary";
  for (size_t i = 0; i < counts.size(); ++i)
    EXPECT_NEAR(compute_us(fit, counts[i]), us[i], 1e-6);
}

// Counterpoint: if the sweep STOPS at the knee (max count == P*), the fit cannot
// distinguish P* from any larger P (all g==1 within range) and pins to the
// boundary. Documents WHY the default sweep must extend past saturation.
TEST(GpuLoaderComputeFit, KneeAtBoundaryIsAmbiguous) {
  const ComputeCurve truth{0.20, 6.0, 32};
  const std::vector<int> counts{1, 2, 4, 8, 16, 24, 32};  // max == knee
  std::vector<double> us;
  for (int c : counts) us.push_back(compute_us(truth, c));
  const ComputeCurve fit = fit_compute_curve(counts, us);
  // The fit still reproduces every measured point (it's a valid fit)...
  for (size_t i = 0; i < counts.size(); ++i)
    EXPECT_NEAR(compute_us(fit, counts[i]), us[i], 1e-6);
  // ...but P need not equal 32 — boundary-pinning is expected here.
  EXPECT_GE(fit.P, 1);
}

TEST(GpuLoaderComputeFit, EmptyInputIsSafe) {
  const ComputeCurve fit = fit_compute_curve({}, {});
  EXPECT_EQ(fit.P, 1);
  EXPECT_EQ(fit.a_us, 0.0);
  EXPECT_EQ(fit.b_us, 0.0);
}

// ── Reconciliation + compute provenance JSON round-trip ──────────────────────

TEST(GpuLoaderConstants, ReconAndComputeFieldsRoundTrip) {
  LoaderConstants a = make_sample();
  a.devices[0].recon_overhead_us = 5.5;
  a.devices[0].recon_added_us    = 1.25;
  a.devices[1].recon_overhead_us = 6.0;
  a.devices[1].recon_added_us    = 1.40;
  a.devices[0].compute = ComputeCurve{0.42, 7.5, 4};
  const LoaderConstants b = from_json_string(to_json_string(a));
  EXPECT_EQ(a, b);
  EXPECT_EQ(b.compute_N, a.compute_N);
  EXPECT_EQ(b.compute_K, a.compute_K);
  EXPECT_DOUBLE_EQ(b.recon_payload_bytes, a.recon_payload_bytes);
  EXPECT_DOUBLE_EQ(b.devices[0].recon_overhead_us, 5.5);
  EXPECT_EQ(b.devices[0].compute, (ComputeCurve{0.42, 7.5, 4}));
}

// ── fixed_overhead_us (I8 trainer feedback) JSON round-trip + default ─────────

TEST(GpuLoaderConstants, FixedOverheadRoundTrip) {
  LoaderConstants a = make_sample();
  a.fixed_overhead_us = 365.5;
  const LoaderConstants b = from_json_string(to_json_string(a));
  EXPECT_EQ(a, b);
  EXPECT_DOUBLE_EQ(b.fixed_overhead_us, 365.5);
}

TEST(GpuLoaderConstants, FixedOverheadDefaultsToZeroForOldFiles) {
  // A file predating the field loads with overhead 0 (backward compatible).
  const LoaderConstants c =
      from_json_string(R"({"version":2,"num_devices":2,"num_banks":4})");
  EXPECT_DOUBLE_EQ(c.fixed_overhead_us, 0.0);
}

// ── Trained place_cons weight JSON round-trip (place_sum_weights, w_numa only) ──
// Only w_numa is trained (residency+hotness are heuristics, not persisted).

TEST(GpuLoaderConstants, PlaceSumWeightsRoundTrip) {
  LoaderConstants a = make_sample();
  a.place_sum_weights.present = true;
  a.place_sum_weights.w_numa  = 600.0;
  const LoaderConstants b = from_json_string(to_json_string(a));
  EXPECT_EQ(a, b);
  EXPECT_TRUE(b.place_sum_weights.present);
  EXPECT_DOUBLE_EQ(b.place_sum_weights.w_numa, 600.0);
}

TEST(GpuLoaderConstants, PlaceSumWeightsAbsentForOldFiles) {
  // A file without the key loads present=false ⇒ engine keeps built-in defaults,
  // and the block is NOT emitted (old files stay clean).
  const LoaderConstants c =
      from_json_string(R"({"version":2,"num_devices":2,"num_banks":4})");
  EXPECT_FALSE(c.place_sum_weights.present);
  EXPECT_EQ(to_json_string(c).find("place_sum_weights"), std::string::npos);
}

// ── Device identity (name + uuid) JSON round-trip + match guard (I8b) ─────────

TEST(GpuLoaderConstants, DeviceIdentityRoundTrip) {
  LoaderConstants a = make_sample();
  a.devices[0].name = "NVIDIA GeForce RTX 5090";
  a.devices[0].uuid = "GPU-12345678-9abc-def0-1234-56789abcdef0";
  a.devices[1].name = "NVIDIA GeForce RTX 5080";
  a.devices[1].uuid = "GPU-aabbccdd-eeff-0011-2233-445566778899";
  const LoaderConstants b = from_json_string(to_json_string(a));
  EXPECT_EQ(a, b);
  EXPECT_EQ(b.devices[0].name, "NVIDIA GeForce RTX 5090");
  EXPECT_EQ(b.devices[0].uuid, "GPU-12345678-9abc-def0-1234-56789abcdef0");
  EXPECT_EQ(b.devices[1].uuid, "GPU-aabbccdd-eeff-0011-2233-445566778899");
}

// A v2 JSON written before the identity fields existed has no name/uuid keys →
// they must deserialize to empty (not throw). Empty uuid = "skip the uuid check".
TEST(GpuLoaderConstants, DeviceIdentityMissingDefaultsEmpty) {
  const char* no_id = R"({
    "version": 2, "num_devices": 1, "num_banks": 1,
    "devices": [{"position": 0, "numa_node": 2, "xfer_lat_us": 8.0}]
  })";
  const LoaderConstants c = from_json_string(no_id);
  ASSERT_EQ(static_cast<int>(c.devices.size()), 1);
  EXPECT_TRUE(c.devices[0].name.empty());
  EXPECT_TRUE(c.devices[0].uuid.empty());
}

// The engine's UUID-match guard logic (synthetic, no GPU): a stored uuid that
// matches the live uuid → accept; a mismatch → reject; an empty stored uuid →
// skip the check (accept old v2 files). Mirrors engine.cpp Step-14c semantics.
TEST(GpuLoaderConstants, UuidMatchGuardLogic) {
  // Replicates the per-device decision used in engine.cpp kLoaded validation.
  auto uuid_ok = [](const std::string& stored, const std::string& live) {
    if (stored.empty()) return true;               // old v2 file → skip identity check
    if (live.empty())   return true;               // can't query live → don't reject
    return stored == live;
  };
  EXPECT_TRUE(uuid_ok("GPU-aaaa", "GPU-aaaa"));     // match → accept
  EXPECT_FALSE(uuid_ok("GPU-aaaa", "GPU-bbbb"));    // mismatch → reject (recalibrate)
  EXPECT_TRUE(uuid_ok("", "GPU-bbbb"));             // old file → skip
  EXPECT_TRUE(uuid_ok("GPU-aaaa", ""));             // live unknown → don't reject
}

// ── Backward-compat: a v1 (transfer-only) JSON must still load with defaults ──

TEST(GpuLoaderConstants, BackwardCompatV1TransferOnly) {
  // No compute/recon provenance, no per-device compute/recon keys → all default.
  const char* v1 = R"({
    "version": 1, "source": "calibrated", "expert_bytes": 24772992.0,
    "num_devices": 1, "num_banks": 1, "ncf": [0.0, 1.0, 1.2],
    "devices": [{"position": 0, "numa_node": 2, "xfer_lat_us": 8.0}],
    "banks": [{"node": 2, "egress_us": 557.0, "contention": 1.0}],
    "matrix": [[{"rate_us": 557.0, "tier": 1, "lat_us": 110.0}]],
    "contention": []
  })";
  const LoaderConstants c = from_json_string(v1);
  EXPECT_EQ(c.version, 1);
  EXPECT_EQ(c.num_devices, 1);
  ASSERT_EQ(static_cast<int>(c.devices.size()), 1);
  // New fields default to "not measured".
  EXPECT_EQ(c.compute_N, 0);
  EXPECT_EQ(c.compute_K, 0);
  EXPECT_DOUBLE_EQ(c.recon_payload_bytes, 0.0);
  EXPECT_EQ(c.devices[0].compute, ComputeCurve{});  // {0,0,1}
  EXPECT_DOUBLE_EQ(c.devices[0].recon_overhead_us, 0.0);
  EXPECT_DOUBLE_EQ(c.devices[0].recon_added_us, 0.0);
}

// The committed reference fixture must still round-trip after the format bump.
TEST(GpuLoaderConstants, CommittedSampleRoundTrips) {
  const std::string path =
      std::string(LAYERSTORM_SOURCE_DIR) + "/tests/assets/gpu_loader_calibration_sample.json";
  if (!std::filesystem::exists(path)) GTEST_SKIP() << "sample fixture not present";
  const LoaderConstants a = load(path);
  const LoaderConstants b = from_json_string(to_json_string(a));
  EXPECT_EQ(a, b);
}

// ── compute_dims_match: weight/config-specificity guard (INV-LOADER-CAL-4) ────

TEST(GpuLoaderConstants, ComputeDimsMatchGuard) {
  LoaderConstants c = make_sample();  // compute_N=4096, compute_K=7168
  EXPECT_TRUE(compute_dims_match(c, 4096, 7168));
  EXPECT_FALSE(compute_dims_match(c, 4096, 6144));   // different hidden → reject
  EXPECT_FALSE(compute_dims_match(c, 6144, 7168));   // different N → reject
  // A transfer-only file (compute_N==0) is compatible with any model — no stale
  // compute numbers to mistrust.
  c.compute_N = 0;
  EXPECT_TRUE(compute_dims_match(c, 1234, 5678));
}

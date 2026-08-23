#pragma once
// GPU loader cost-model constants (spec/GPU_LOADER_MODEL.md §2.2/§2.3/§6).
//
// The benchmarked per-device / per-bank / per-(bank,device) constants the loader
// solver consumes. CUDA-free, pure data + JSON (house style: manual builders,
// see model/weight_pipeline/manifest.cpp). Lives in layerstorm_core; the
// calibration util (loader_calibration) fills these via the DeviceBackend, and
// the engine may calibrate-at-init, re-calibrate at runtime, or load from JSON.
//
// Ticket: spec/tickets/I8b_LOADER_CALIBRATION.md.
#include <cstdint>
#include <string>
#include <vector>

namespace layerstorm::gpu_loader {

// v2 adds the compute-curve / reconciliation measurement metadata (compute_N/K/
// tokens, recon_payload_bytes). v1 files (transfer-only) still load: the new fields
// default and `source` stays whatever the file says. Loaders accept any version
// <= current and validate the embedded dims against the deployed model (the JSON is
// weight/config-specific — INV-LOADER-CAL-4). v2 files may additionally carry the
// per-bank HBM tags `is_hbm` / `cpu_affinity_node` (TD-NUMA-HBM-BANKS) — optional
// keys, absent in pre-HBM files → false / -1, no version bump needed.
inline constexpr int kLoaderConstantsVersion = 2;

// Per-device compute curve: compute_j(c) = a_us*c + b_us*ceil(c/P)  (batch-step,
// TD-LOADER-COMPUTE-PARALLELISM). a = per-expert linear, b = fixed per batch,
// P = batch width (parallel slots: GPU SM-saturation width / CPU threads).
struct ComputeCurve {
  double a_us = 0.0;
  double b_us = 0.0;
  int    P    = 1;
  bool operator==(const ComputeCurve&) const = default;
};

// Per-device (GpuRef position) constants.
struct DeviceConstants {
  int    position    = 0;     // index in hardware.gpus[] (GpuRef.position)
  int    numa_node   = -1;    // this device's NUMA-local node (its "bound" bank)
  std::string name;           // CUDA device name (provenance; empty = old v2 file)
  std::string uuid;           // CUDA device UUID (GPU-xxxxxxxx-...; empty = old v2 file)
  double xfer_lat_us = 0.0;   // pre-DMA enqueue + detection latency (xfer_lat[j])
  ComputeCurve compute{};
  double recon_overhead_us = 0.0;   // recon_overhead[j]
  double recon_added_us    = 0.0;   // recon_added[j]
  bool operator==(const DeviceConstants&) const = default;
};

// Per-NUMA-bank constants (source-side egress; §3.3 bank_egress floor).
struct BankConstants {
  int    node       = -1;     // NUMA node id (the bank id)
  double egress_us  = 0.0;    // numa_speed[b]: solo per-expert egress (shared-channel rate)
  double contention = 1.0;    // numa_contention[b] <= 1 (1 = strict serial sum)
  // CPU-less HBM / device-memory bank (TD-NUMA-HBM-BANKS / INV-NUMA-HOSTBANK):
  // measured with a footprint capped to the node's free memory, from a thread
  // pinned to cpu_affinity_node (the node itself has no CPUs). Optional JSON
  // keys — absent (pre-HBM files) → false / -1, unchanged behavior.
  bool   is_hbm            = false;
  int    cpu_affinity_node = -1;  // nearest CPU node the measurement ran on (-1 = n/a)
  bool operator==(const BankConstants&) const = default;
};

// Per (bank b -> device j) effective transfer cell (the GPU x NUMA matrix).
struct TransferCell {
  double rate_us = 0.0;   // measured per-expert transfer time bank->device (effective, incl. tier)
  int    tier    = 1;     // NUMA-closeness tier (1 = local/bound)
  double lat_us  = 0.0;   // numa_lat[b,j]
  bool operator==(const TransferCell&) const = default;
};

// Same-bank two-device simultaneous-transfer contention measurement (I8b).
// "bound" = the bank is that device's NUMA-local node; "unbound" = remote.
enum class ContentionConfig : uint8_t { kTwoBound = 0, kBoundUnbound = 1, kTwoUnbound = 2 };

struct ContentionSample {
  int  bank     = -1;
  int  device_a = -1;
  int  device_b = -1;
  ContentionConfig config = ContentionConfig::kTwoBound;
  double solo_us = 0.0;   // single device pulling the bank, per-expert
  double dual_us = 0.0;   // both devices pulling the same bank simultaneously, per-expert
  double factor  = 1.0;   // dual_us / solo_us (>1 = contention; ~2 = strict serial sharing)
  bool operator==(const ContentionSample&) const = default;
};

// Trained never-lose place_cons weight — ONLY w_numa (loader_place_sum.h). The two
// PREDICTIVE terms (residency recur, freq/EMA hotness) are HEURISTICS, not trained: a
// weight on a prediction cannot be fixed by training when the prediction is wrong
// (w_resid·recur zeroes on the recur=0 churners — measured −13.7%). w_numa is a
// STATIC home-node FACT, so it is the only trainable place_sum param. Fit black-box
// against the residency-HONEST offline sim (train_place_sum_weights.py — never
// regression, INV-LOADER-OBJECTIVE-MYOPIC) and persisted here so the engine loads it
// from the calibration JSON; env LS_LOADER_PLACE_W_NUMA overrides (env > trained-JSON
// > default). present=false (old files with no "place_sum_weights" key) ⇒ the engine
// keeps its built-in w_numa default, unchanged behavior.
struct PlaceSumWeightsCalib {
  bool   present = false;
  double w_numa  = 0.0;
  bool operator==(const PlaceSumWeightsCalib&) const = default;
};

// The full calibrated constant set.
struct LoaderConstants {
  int         version      = kLoaderConstantsVersion;
  std::string source       = "default";   // "calibrated" | "loaded" | "default"
  double      expert_bytes = 0.0;         // transfer unit size the rates were measured at
  int         num_devices  = 0;
  int         num_banks    = 0;
  // Compute-curve / reconciliation provenance: the model dims the per-device
  // ComputeCurve + recon_* were measured at (0 = not measured). These make the JSON
  // weight/config-specific — a loader must check they match the deployed expert FFN
  // dims (gate_up N = 2*moe_intermediate_size, K = hidden_size) before trusting the
  // compute curves; a mismatch means the file belongs to a different model.
  int         compute_N         = 0;      // grouped-GEMM output dim the curve was fit at
  int         compute_K         = 0;      // grouped-GEMM input dim
  int         compute_tokens    = 0;      // tokens/expert used in the sweep
  double      recon_payload_bytes = 0.0;  // recon proxy payload (hidden_size * dtype)
  // Global fixed per-layer overhead the per-rate model structurally misses (the I8
  // x-ray trainer's additive offsets dev_off+bank_off+recon_off, folded here because a
  // uniform µs cost is NOT a per-device rate). Added to the solver's predicted T
  // (objective_from_sums) AFTER the max(makespan,bank) term. Default 0 → backward
  // compatible: old files (no key) load with no overhead, unchanged behavior.
  double      fixed_overhead_us = 0.0;
  // Trained never-lose place_cons weights (loaded from JSON; absent → built-in
  // defaults). See PlaceSumWeightsCalib. Not measured by calibrate(); injected by
  // the offline place_sum trainer into the calibration JSON.
  PlaceSumWeightsCalib place_sum_weights{};
  std::vector<double>                    ncf;        // ncf[tier]; ncf[1]=1.0 (index 0 unused)
  std::vector<DeviceConstants>           devices;    // [num_devices]
  std::vector<BankConstants>             banks;      // [num_banks]
  std::vector<std::vector<TransferCell>> matrix;     // [bank][device]
  std::vector<ContentionSample>          contention; // measured same-bank two-device configs
  bool operator==(const LoaderConstants&) const = default;
};

// JSON serialization (manual builders; matches model/weight_pipeline/manifest.cpp).
std::string     to_json_string(const LoaderConstants&, int indent = 2);
LoaderConstants from_json_string(const std::string&);
void            save(const LoaderConstants&, const std::string& path);
LoaderConstants load(const std::string& path);

// True iff the file's compute-curve dims match the deployed model's expert FFN dims
// (gate_up N = 2*moe_intermediate_size, K = hidden_size). compute_N==0 means the file
// never measured compute (transfer-only) → compatible (curves are at defaults, no
// stale numbers to mistrust). Used by the engine to reject a loaded calibration whose
// compute curves belong to a different model (INV-LOADER-CAL-4).
bool compute_dims_match(const LoaderConstants&, int model_N, int model_K);

// ── CPU expert offload: CPU as an assignable loader-solver target (Stage 1) ──
//
// Append a CPU expert device to a calibrated LoaderConstants so the solver can
// place routed experts on CPU compute. A CPU expert reads each expert's weights
// from its EXISTING host source bank (SolveRequest::bank_of[i]) — directly, with
// NO PCIe transfer — so:
//   * every appended per-bank transfer cell is 0 (rate + lat): the CPU device's
//     makespan is purely its ComputeCurve (the per-expert CPU FFN time), never a
//     transfer term. This is the "fewer bytes on the wire" win — the CPU-absorbed
//     expert removes its weight-fetch from the binding PCIe bus.
//   * the CPU read is STILL priced as egress on the expert's source bank, because
//     the solver adds banks[b].egress_us for EVERY miss regardless of device.
//     So if that bank is the DDR node a GPU is also pulling, the shared
//     bank_egress floor (bank_egress_factor(contention, g)) prices the DDR
//     contention automatically (the FETCH_XRAY per-node wall); if the expert's
//     weights are staged into an HBM bank (cheap egress), the read is ~free.
// The solver therefore assigns an expert to CPU only when the CPU FFN time it
// adds is cheaper than the GPU (PCIe fetch + GPU compute) it displaces AND doing
// so does not push past the source bank's contention floor — exactly the intended
// "CPU absorbs the tail the GPU would otherwise fetch, off the critical bus."
//
// `compute` is the calibrated per-expert CPU FFN curve. For the expert-per-node
// layout (one expert per NUMA node, computed in parallel) use
// a_us=0, b_us=per_expert_us, P=num_cpu_nodes  →  b_us·ceil(c/P).
// Returns the new device index (== the previous num_devices). num_devices is
// incremented and every matrix bank row gets one appended zero TransferCell.
int append_cpu_expert_device(LoaderConstants& k, int numa_node,
                             const ComputeCurve& compute);

}  // namespace layerstorm::gpu_loader

namespace layerstorm::gpu_loader {

// M2v2 exposed-wall objective params (config-scoped; fit by
// tools/loader_xray/exposed_model.py --form v2 — see SolveRequest::m2 docs).
// Arrays are indexed by solver device index [num_devices].
struct M2Params {
  bool valid = false;
  std::vector<double> s, o0, oc, hsat, cpw;
  double g_c = 1.0;
};

// Load + validate a v2 params JSON ({form:"v2", s[],o0[],oc[],hsat[],cpw[],
// g_c, b0}); b0 is ignored (constant — argmin-irrelevant). Returns
// valid=false (never throws) on missing file, wrong form, or array sizes
// != num_devices.
M2Params load_m2_params(const std::string& path, int num_devices);

}  // namespace layerstorm::gpu_loader

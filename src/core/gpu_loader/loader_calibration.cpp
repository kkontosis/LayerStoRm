#include "core/gpu_loader/loader_calibration.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

#include "core/cuda_hardware_query.h"
#include "core/device_backend.h"
#include "core/expert_device.h"
#include "core/gpu_ref.h"
#include "core/memory/numa_manager.h"
#include "smxx/permute/moe_permute.h"               // full chain: query_moe_permute_workspace_size (CUDA-free decl)
#include "parallelism/collective_backend.h"        // recon: real NCCL allreduce (CUDA-free iface)
#include "parallelism/nccl_collective_backend.h"   // recon: make_nccl_collective_backend()

namespace layerstorm::gpu_loader {
namespace {

using compute::DeviceBackend;
using compute::ExpertDevice;
using memory::NumaBuffer;
using memory::NumaManager;

// A NUMA-bound, portable-pinned host buffer for DMA from a specific bank.
// (Mirrors the P-24 PinnedNodeArena recipe: allocate_on_node → fault → register.)
// Holds a rotating footprint: `chunks` sub-regions of `chunk_bytes` each, so the
// source can cycle across distinct DDR pages and defeat the HBM cache.
struct PinnedNumaBuffer {
  NumaManager* numa = nullptr;
  NumaBuffer   buf{};
  bool         registered = false;

  PinnedNumaBuffer(NumaManager& n, size_t bytes, int node) : numa(&n) {
    buf = numa->allocate_on_node(bytes, node);
    std::memset(buf.data, 0, buf.size);  // fault pages onto this node
    registered = (core::host_register_pinned_portable(buf.data, buf.size) == 0);
  }
  ~PinnedNumaBuffer() {
    if (registered) core::host_unregister_pinned(buf.data);
    if (buf.data) numa->free(buf);
  }
  PinnedNumaBuffer(const PinnedNumaBuffer&) = delete;
  PinnedNumaBuffer& operator=(const PinnedNumaBuffer&) = delete;
  const char* base() const { return static_cast<const char*>(buf.data); }
};

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Median per-transfer time (µs) for H2D of `chunk` bytes, cycling the source
// across `total/chunk` sub-regions of `base` (footprint rotation). Returns 0 if
// timing events are unsupported on this backend.
double measure_h2d_us(DeviceBackend* be, const char* base, size_t total_bytes,
                      size_t chunk, int warmup, int iters) {
  const size_t K = std::max<size_t>(1, total_bytes / chunk);
  void* dst    = be->device_alloc(chunk);
  void* stream = be->create_stream();
  void* ev0    = be->create_timing_event();
  void* ev1    = be->create_timing_event();
  double result = 0.0;
  if (dst && stream && ev0 && ev1) {
    // PCIe link warmup (TD-LOADER-CALIB-WARMUP). An idle PCIe link sits at gen1
    // (2.5 GT/s) under ASPM L1; a fixed tiny warmup (was 3 iters) leaves it only
    // partially trained, systematically UNDER-measuring whichever device idled
    // longest before its turn — observed as dev0 reading ~½ its real H2D
    // bandwidth (29 vs 56 GB/s), a phantom per-device asymmetry the loader solver
    // then "optimizes" by packing the falsely-cheap device, costing tok/s. Drive
    // sustained back-to-back copies (batched to bound queue depth) until a
    // wall-clock floor elapses so the LTSSM completes its speed change to max gen
    // and holds it through the timed window. One-off cost; ~20 ms per cell.
    {
      using clk = std::chrono::steady_clock;
      const auto t0 = clk::now();
      size_t w = 0;
      do {
        for (int j = 0; j < 8; ++j, ++w)
          be->memcpy_h2d_async(dst, base + (w % K) * chunk, chunk, stream);
        be->device_sync();
      } while (std::chrono::duration<double, std::milli>(clk::now() - t0).count() < 20.0);
    }
    for (int i = 0; i < warmup; ++i)
      be->memcpy_h2d_async(dst, base + (static_cast<size_t>(i) % K) * chunk, chunk, stream);
    be->device_sync();
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      const char* src = base + (static_cast<size_t>(i) % K) * chunk;
      be->record_event(ev0, stream);
      be->memcpy_h2d_async(dst, src, chunk, stream);
      be->record_event(ev1, stream);
      be->device_sync();
      const float ms = be->event_elapsed_ms(ev0, ev1);
      if (ms >= 0.0f) samples.push_back(static_cast<double>(ms) * 1000.0);  // ms → µs
    }
    result = median(std::move(samples));
  }
  if (ev0) be->destroy_event(ev0);
  if (ev1) be->destroy_event(ev1);
  if (stream) be->destroy_stream(stream);
  if (dst) be->device_free(dst);
  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// H2D-vs-CPU-DDR bus-contention diagnostic passes (env-gated, DEFAULT OFF).
//
// Quantify how much a CPU-side DDR-read load (the CPU-expert-offload weight
// stream, or a DDR→HBM tier-fill copy) degrades the per-(bank,device) H2D
// fetch rate — the direct measurement behind spec/cpu_offload_tuning.md §8b's
// "CPU offload on DDR occupies the very buses the GPU fetch needs".
//
//  * LS_CALIB_H2D_VS_DDR2HBM=1  — variant 2: re-measure the (bank,device) H2D
//    matrix WHILE an in-process background crew (threads pinned to a source DDR
//    node) runs a sustained rotating memcpy DDR-source → HBM-dest, sweeping the
//    source node (LS_CALIB_DDR2HBM_SRC, default "0,1,2,3"). Reports solo/dual
//    rate + degradation per (bank, ddr2hbm-source-node) and the copier's own
//    achieved GB/s per cell (both directions of the contention).
//  * LS_CALIB_H2D_VS_CPUKERNEL=1 — variant 3: re-measure the matrix WHILE the
//    real cold-streaming CPU-expert kernel harness (scratchpad/cpu_kernel_bench
//    --modes=stream, Q4_K GLM-5.2 weights, working set >> LLC) runs as a child
//    process at LS_CALIB_CPUKERNEL_ARMS (default 10, 20, 40 threads = 1/2/4 DDR
//    nodes x 10 threads, the §8b staircase). The child is spawned per arm and
//    SIGKILLed after the arm's cells (only our own child is ever killed).
//
// Both passes are DIAGNOSTIC ONLY: they log tables (spdlog, grep prefixes
// "[x2]" / "[x3]") and never touch the returned LoaderConstants — the
// production calibration output is byte-identical whether they run or not, and
// with the env vars unset (the default) the code does not execute at all.
// CUDA-free (DeviceBackend timing only) — INV-GPU-1 holds; the child spawn is
// plain POSIX fork/exec.

bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::string(v) != "0";
}

std::vector<int> env_int_list(const char* name, const std::vector<int>& dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  std::vector<int> out;
  const std::string s(v);
  for (size_t p = 0; p < s.size();) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    out.push_back(std::atoi(s.substr(p, q - p).c_str()));
    p = q + 1;
  }
  return out;
}

double env_double(const char* name, double dflt) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::atof(v) : dflt;
}

// Background DDR→HBM copy crew: `nthreads` threads pinned to the source DDR
// node's CPUs, each looping memcpy(dst_chunk, src_chunk, 8 MiB) with the source
// rotating over a footprint >> LLC (so reads genuinely stream from DDR) and the
// dest rotating over an HBM-node buffer. Occupies the source node's DDR read
// bus + the dest node's HBM write bus — the tier-fill / offload-weight-stream
// traffic shape. Byte counter lets the caller compute achieved GB/s per window.
struct BgDdr2HbmCopy {
  NumaManager* numa = nullptr;
  int src_node = -1, dst_node = -1;
  NumaBuffer src{}, dst{};
  std::vector<std::thread> threads;
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> copied{0};
  bool ok = false;

  BgDdr2HbmCopy(NumaManager& n, int s, int d, size_t src_bytes, size_t dst_bytes)
      : numa(&n), src_node(s), dst_node(d) {
    // Cap the dest by the HBM node's free memory (it may be near-full when the
    // arena holder is armed); floor one chunk.
    const size_t dst_free = numa->node_free_bytes(d);
    dst_bytes = std::max<size_t>(kChunk, std::min(dst_bytes, dst_free / 2));
    src = numa->allocate_on_node(src_bytes, s);
    dst = numa->allocate_on_node(dst_bytes, d);
    if (src.data && dst.data) {
      std::memset(src.data, 0x5a, src.size);  // fault pages onto the nodes
      std::memset(dst.data, 0, dst.size);
      ok = true;
    } else {
      spdlog::warn("gpu_loader[x2]: ddr2hbm buffer alloc failed (src node {} {} B, dst node {} {} B)",
                   s, src_bytes, d, dst_bytes);
    }
  }
  ~BgDdr2HbmCopy() {
    halt();
    if (src.data) numa->free(src);
    if (dst.data) numa->free(dst);
  }
  BgDdr2HbmCopy(const BgDdr2HbmCopy&) = delete;
  BgDdr2HbmCopy& operator=(const BgDdr2HbmCopy&) = delete;

  static constexpr size_t kChunk = 8ull << 20;  // 8 MiB per memcpy

  void spawn(int nthreads) {
    if (!ok) return;
    stop.store(false, std::memory_order_release);
    const size_t sk = std::max<size_t>(1, src.size / kChunk);
    const size_t dk = std::max<size_t>(1, dst.size / kChunk);
    for (int t = 0; t < nthreads; ++t) {
      threads.emplace_back([this, t, sk, dk] {
        memory::ScopedThreadNodeBind pin(*numa, src_node);  // read from the local bus
        char*       d = static_cast<char*>(dst.data);
        const char* s = static_cast<const char*>(src.data);
        size_t i = static_cast<size_t>(t) * 7919;  // decorrelate thread phases
        while (!stop.load(std::memory_order_relaxed)) {
          std::memcpy(d + (i % dk) * kChunk, s + (i % sk) * kChunk, kChunk);
          copied.fetch_add(kChunk, std::memory_order_relaxed);
          ++i;
        }
      });
    }
  }
  void halt() {
    stop.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();
    threads.clear();
  }
};

// Child-process runner for the cold CPU-expert kernel harness. stdout+stderr
// land in a named log; halt() SIGKILLs + reaps ONLY this child.
struct BgChildProc {
  pid_t pid = -1;
  bool spawn(const std::string& bin, const std::vector<std::string>& args,
             const std::string& log_path) {
    pid = fork();
    if (pid == 0) {
      const int fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd >= 0) { ::dup2(fd, 1); ::dup2(fd, 2); ::close(fd); }
      std::vector<char*> argv;
      argv.push_back(const_cast<char*>(bin.c_str()));
      for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
      argv.push_back(nullptr);
      ::execv(bin.c_str(), argv.data());
      _exit(127);
    }
    return pid > 0;
  }
  bool alive() const {
    if (pid <= 0) return false;
    int status = 0;
    return ::waitpid(pid, &status, WNOHANG) == 0;
  }
  void halt() {
    if (pid <= 0) return;
    ::kill(pid, SIGKILL);
    int status = 0;
    ::waitpid(pid, &status, 0);
    pid = -1;
  }
  ~BgChildProc() { halt(); }
};

double cell_gbps(double bytes, double us) { return us > 0.0 ? bytes / 1e9 / (us / 1e6) : 0.0; }

// Variant 2: H2D matrix under a swept-source DDR→HBM background copy.
void run_h2d_vs_ddr2hbm_pass(const std::vector<DeviceBackend*>& backends,
                             NumaManager& numa, const CalibrationConfig& cfg,
                             const std::vector<int>& banks,
                             const std::vector<size_t>& bank_totals,
                             const LoaderConstants& c) {
  using clk = std::chrono::steady_clock;
  const std::vector<int> srcs = env_int_list("LS_CALIB_DDR2HBM_SRC", {0, 1, 2, 3});
  const int    nthreads  = static_cast<int>(env_double("LS_CALIB_DDR2HBM_THREADS", 8));
  const size_t src_bytes = static_cast<size_t>(env_double("LS_CALIB_DDR2HBM_SRC_MB", 2048)) << 20;
  const size_t dst_bytes = static_cast<size_t>(env_double("LS_CALIB_DDR2HBM_DST_MB", 1024)) << 20;
  const int    dst_env   = static_cast<int>(env_double("LS_CALIB_DDR2HBM_DST", -1));
  const size_t xfer      = static_cast<size_t>(cfg.expert_bytes);

  spdlog::info("gpu_loader[x2]: H2D-vs-DDR2HBM pass — sources {}, {} copy threads, "
               "src {} MiB rotating, dst {} MiB",
               srcs.size(), nthreads, src_bytes >> 20, dst_bytes >> 20);

  // Dest HBM node per source: env override, else the source node's HBM partner
  // (the HBM bank whose cpu_affinity_node == src), else 4+src.
  auto partner = [&](int s) -> int {
    if (dst_env >= 0) return dst_env;
    for (int b = 0; b < c.num_banks; ++b)
      if (c.banks[b].is_hbm && c.banks[b].cpu_affinity_node == s) return c.banks[b].node;
    return 4 + s;
  };

  // Build all loads once (buffers persist across arms; threads start/stop per arm).
  std::vector<std::unique_ptr<BgDdr2HbmCopy>> loads;
  for (int s : srcs)
    loads.push_back(std::make_unique<BgDdr2HbmCopy>(numa, s, partner(s), src_bytes, dst_bytes));

  // Solo copy rate per load (no H2D running).
  for (auto& L : loads) {
    if (!L->ok) continue;
    L->spawn(nthreads);
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    const uint64_t b0 = L->copied.load();
    const auto     t0 = clk::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    const uint64_t b1 = L->copied.load();
    const double   dt = std::chrono::duration<double>(clk::now() - t0).count();
    L->halt();
    spdlog::info("gpu_loader[x2]: SOLO copy src={} dst={} thr={} rate={:.1f} GB/s",
                 L->src_node, L->dst_node, nthreads,
                 static_cast<double>(b1 - b0) / 1e9 / dt);
  }

  for (int b = 0; b < static_cast<int>(banks.size()); ++b) {
    memory::ScopedThreadNodeBind pin(numa, numa.hbm_cpu_affinity_node(banks[b]));
    PinnedNumaBuffer rbuf(numa, bank_totals[b], banks[b]);
    for (auto& L : loads) {
      if (!L->ok) continue;
      L->spawn(nthreads);
      std::this_thread::sleep_for(std::chrono::milliseconds(200));  // copy steady state
      for (int d = 0; d < c.num_devices; ++d) {
        const uint64_t cb0 = L->copied.load();
        const auto     ct0 = clk::now();
        const double dual = measure_h2d_us(backends[d], rbuf.base(), bank_totals[b], xfer,
                                           cfg.warmup_iters, cfg.timed_iters);
        const double cdt = std::chrono::duration<double>(clk::now() - ct0).count();
        const double co_rate = static_cast<double>(L->copied.load() - cb0) / 1e9 / cdt;
        const double solo = c.matrix[b][d].rate_us;  // same footprint, main pass
        spdlog::info("gpu_loader[x2]: bank={} dev={} src={} solo_us={:.1f} dual_us={:.1f} "
                     "solo={:.1f} dual={:.1f} GB/s degr={:.1f}% copy={:.1f} GB/s",
                     banks[b], c.devices[d].position, L->src_node, solo, dual,
                     cell_gbps(cfg.expert_bytes, solo), cell_gbps(cfg.expert_bytes, dual),
                     (solo > 0.0 && dual > 0.0) ? (dual / solo - 1.0) * 100.0 : 0.0, co_rate);
      }
      L->halt();
    }
  }
}

// Variant 3: H2D matrix under the real cold-streaming CPU-expert kernel.
void run_h2d_vs_cpukernel_pass(const std::vector<DeviceBackend*>& backends,
                               NumaManager& numa, const CalibrationConfig& cfg,
                               const std::vector<int>& banks,
                               const std::vector<size_t>& bank_totals,
                               const LoaderConstants& c) {
  const char* bin_env = std::getenv("LS_CALIB_CPUKERNEL_BIN");
  const std::string bin = bin_env && *bin_env ? bin_env : "scratchpad/cpu_kernel_bench";
  const char* arms_env = std::getenv("LS_CALIB_CPUKERNEL_ARMS");
  // arm = "<nodes-csv>:<tpn>", '|'-separated. Defaults = the §8b staircase at
  // 10 threads/node: 10 thr (1 node), 20 thr (2 nodes), 40 thr (4 nodes).
  const std::string arms_s = arms_env && *arms_env ? arms_env : "1:10|0,1:10|0,1,2,3:10";
  const int warm_ms = static_cast<int>(env_double("LS_CALIB_CPUKERNEL_WARM_MS", 8000));
  const char* pp_env = std::getenv("LS_CALIB_CPUKERNEL_PREPACK");
  const std::string prepack = pp_env && *pp_env ? pp_env : "test-data/GLM-5.2-prepacked";
  const size_t xfer = static_cast<size_t>(cfg.expert_bytes);

  std::vector<std::pair<std::string, std::string>> arms;  // (nodes, tpn)
  for (size_t p = 0; p < arms_s.size();) {
    size_t q = arms_s.find('|', p);
    if (q == std::string::npos) q = arms_s.size();
    const std::string a = arms_s.substr(p, q - p);
    const size_t col = a.find(':');
    if (col != std::string::npos) arms.emplace_back(a.substr(0, col), a.substr(col + 1));
    p = q + 1;
  }
  spdlog::info("gpu_loader[x3]: H2D-vs-CPUKERNEL pass — {} arms via {}", arms.size(), bin);

  for (const auto& [arm_nodes, arm_tpn] : arms) {
    const int n_nodes = 1 + static_cast<int>(std::count(arm_nodes.begin(), arm_nodes.end(), ','));
    const int total_thr = n_nodes * std::atoi(arm_tpn.c_str());
    const std::string log_path =
        "scratchpad/calib_cpukernel_" + std::to_string(total_thr) + "thr.log";
    BgChildProc child;
    if (!child.spawn(bin,
                     {"--nodes=" + arm_nodes, "--tpn=" + arm_tpn, "--mem=ddr",
                      "--modes=stream", "--iters=2000000", "--nexperts=16",
                      "--prepack=" + prepack},
                     log_path)) {
      spdlog::warn("gpu_loader[x3]: spawn failed for arm {}thr; skipping", total_thr);
      continue;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(warm_ms));  // load + warm + stream
    if (!child.alive()) {
      spdlog::warn("gpu_loader[x3]: cpu_kernel_bench exited early (arm {}thr, see {}); skipping",
                   total_thr, log_path);
      child.halt();
      continue;
    }
    for (int b = 0; b < static_cast<int>(banks.size()); ++b) {
      memory::ScopedThreadNodeBind pin(numa, numa.hbm_cpu_affinity_node(banks[b]));
      PinnedNumaBuffer rbuf(numa, bank_totals[b], banks[b]);
      for (int d = 0; d < c.num_devices; ++d) {
        const double dual = measure_h2d_us(backends[d], rbuf.base(), bank_totals[b], xfer,
                                           cfg.warmup_iters, cfg.timed_iters);
        const double solo = c.matrix[b][d].rate_us;
        spdlog::info("gpu_loader[x3]: arm={}thr nodes={} bank={} dev={} solo_us={:.1f} "
                     "dual_us={:.1f} solo={:.1f} dual={:.1f} GB/s degr={:.1f}%",
                     total_thr, arm_nodes, banks[b], c.devices[d].position, solo, dual,
                     cell_gbps(cfg.expert_bytes, solo), cell_gbps(cfg.expert_bytes, dual),
                     (solo > 0.0 && dual > 0.0) ? (dual / solo - 1.0) * 100.0 : 0.0);
      }
    }
    child.halt();
  }
}
// ═══════════════════════════════════════════════════════════════════════════

// ── Compute-curve pass (compute_j, §2.3 / TD-LOADER-COMPUTE-PARALLELISM) ──
//
// Builds synthetic NVFP4 grouped-GEMM buffers for `c` experts (gate_up projection:
// N output, K input, `tokens` rows/expert) and times one launch on the ExpertDevice,
// median over `iters` after `warmup`. All host buffers are plain std::vector copied
// via DeviceBackend::memcpy_h2d (CUDA-free here, per INV-GPU-1/INV-LOADER-CAL-3).
// Layout mirrors tests/unit/grouped_gemm_test.cpp (uniform 0.5 FP4, identity scales)
// — we only care about *timing*, not the numerics, and uniform inputs keep the kernel
// on its normal code path. The ExpertDevice does the launch; the paired DeviceBackend
// supplies the timing events (the two front the same physical GPU, INV-BH-1).
double measure_grouped_gemm_us(ExpertDevice* ed, DeviceBackend* be, int num_experts,
                               int N, int K, int tokens, int warmup, int iters) {
  if (!ed || !be || num_experts <= 0 || N <= 0 || K <= 0 || tokens <= 0) return 0.0;
  constexpr int kGroup = 16;                 // NVFP4 group size
  const int total_M = num_experts * tokens;
  const int half_K  = K / 2;                 // FP4: 2 nibbles/byte

  // Scale-factor layout (CUTLASS Sm1xxBlkScaledConfig padding).
  const int sf_K = ((K / kGroup + 3) / 4) * 4;
  const int sf_N = ((N + 127) / 128) * 128;
  const int sf_rows_per_expert = ((tokens + 127) / 128) * 128;
  // scale_A is read per-expert at sf_offsets[i] = i*sf_rows_per_expert, so it must
  // span ALL experts' padded blocks. NOT pad(total_M,128) (the grouped_gemm_test
  // helper's form) — that only holds when tokens%128==0; decode tokens=1 makes
  // each expert pad to 128 rows and i*128 overruns a pad(num_experts,128) buffer.
  const int sf_M = num_experts * sf_rows_per_expert;

  // Host buffers (uniform fill — 0x44 packs two 0.5 FP4 nibbles; 0x38 = UE4M3 1.0).
  std::vector<uint8_t> h_A(static_cast<size_t>(total_M) * half_K, 0x44);
  std::vector<uint8_t> h_B(static_cast<size_t>(num_experts) * N * half_K, 0x44);
  std::vector<uint8_t> h_sA(static_cast<size_t>(sf_M) * sf_K, 0x38);
  std::vector<uint8_t> h_sB(static_cast<size_t>(num_experts) * sf_N * sf_K, 0x38);
  std::vector<float>   h_alphas(num_experts, 1.0f);
  std::vector<int32_t> h_off(num_experts + 1);
  std::vector<int32_t> h_sf_off(num_experts + 1);
  std::vector<int32_t> h_prob(static_cast<size_t>(num_experts) * 3);
  for (int i = 0; i <= num_experts; ++i) {
    h_off[i]    = i * tokens;
    h_sf_off[i] = i * sf_rows_per_expert;
  }
  for (int i = 0; i < num_experts; ++i) {
    h_prob[i * 3 + 0] = tokens;
    h_prob[i * 3 + 1] = N;
    h_prob[i * 3 + 2] = K;
  }

  const size_t out_bytes = static_cast<size_t>(total_M) * N * sizeof(uint16_t);  // bf16
  const size_t ws_bytes  = compute::query_nvfp4_grouped_gemm_workspace_size(
      num_experts, N, K, compute::GemmOutputDtype::kBFloat16);

  void* d_A   = ed->device_alloc(h_A.size());
  void* d_B   = ed->device_alloc(h_B.size());
  void* d_D   = ed->device_alloc(out_bytes);
  void* d_sA  = ed->device_alloc(h_sA.size());
  void* d_sB  = ed->device_alloc(h_sB.size());
  void* d_al  = ed->device_alloc(h_alphas.size() * sizeof(float));
  void* d_off = ed->device_alloc(h_off.size() * sizeof(int32_t));
  void* d_sfo = ed->device_alloc(h_sf_off.size() * sizeof(int32_t));
  void* d_pr  = ed->device_alloc(h_prob.size() * sizeof(int32_t));
  void* d_ws  = ws_bytes ? ed->device_alloc(ws_bytes) : nullptr;
  void* stream = be->create_stream();
  void* ev0 = be->create_timing_event();
  void* ev1 = be->create_timing_event();

  double result = 0.0;
  const bool ok = d_A && d_B && d_D && d_sA && d_sB && d_al && d_off && d_sfo && d_pr &&
                  (ws_bytes == 0 || d_ws) && stream && ev0 && ev1;
  if (ok) {
    be->memcpy_h2d(d_A, h_A.data(), h_A.size());
    be->memcpy_h2d(d_B, h_B.data(), h_B.size());
    be->memcpy_h2d(d_sA, h_sA.data(), h_sA.size());
    be->memcpy_h2d(d_sB, h_sB.data(), h_sB.size());
    be->memcpy_h2d(d_al, h_alphas.data(), h_alphas.size() * sizeof(float));
    be->memcpy_h2d(d_off, h_off.data(), h_off.size() * sizeof(int32_t));
    be->memcpy_h2d(d_sfo, h_sf_off.data(), h_sf_off.size() * sizeof(int32_t));
    be->memcpy_h2d(d_pr, h_prob.data(), h_prob.size() * sizeof(int32_t));
    be->memset_async(d_D, 0, out_bytes, stream);

    compute::Nvfp4GroupedGemmParams p{};
    p.num_experts    = num_experts;
    p.N              = N;
    p.K              = K;
    p.A_base         = d_A;
    p.B_base         = d_B;
    p.D_base         = d_D;
    p.scale_A_base   = d_sA;
    p.scale_B_base   = d_sB;
    p.alphas         = static_cast<const float*>(d_al);
    p.expert_offsets = static_cast<const int32_t*>(d_off);
    p.sf_offsets     = static_cast<const int32_t*>(d_sfo);
    p.problem_sizes  = static_cast<const int32_t*>(d_pr);
    p.output_dtype   = compute::GemmOutputDtype::kBFloat16;

    for (int i = 0; i < warmup; ++i) ed->nvfp4_grouped_gemm(p, d_ws, ws_bytes, stream);
    be->device_sync();
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      be->record_event(ev0, stream);
      ed->nvfp4_grouped_gemm(p, d_ws, ws_bytes, stream);
      be->record_event(ev1, stream);
      be->device_sync();
      const float ms = be->event_elapsed_ms(ev0, ev1);
      if (ms >= 0.0f) samples.push_back(static_cast<double>(ms) * 1000.0);  // ms → µs
    }
    result = median(std::move(samples));
  }

  if (ev0) be->destroy_event(ev0);
  if (ev1) be->destroy_event(ev1);
  if (stream) be->destroy_stream(stream);
  ed->device_free(d_A);  ed->device_free(d_B);  ed->device_free(d_D);
  ed->device_free(d_sA); ed->device_free(d_sB); ed->device_free(d_al);
  ed->device_free(d_off); ed->device_free(d_sfo); ed->device_free(d_pr);
  if (d_ws) ed->device_free(d_ws);
  return result;
}

// ── Full routed-FFN decode chain (compute_j, full mode) ──────────────────────
//
// measure_grouped_gemm_us above times ONLY the gate_up grouped-GEMM (~38–108 µs),
// but the x-ray on real timings showed the per-layer MoE compute is ≈365 µs and
// count-independent (R²=0): the missing cost is the REST of the chain. This times
// the whole decode chain end-to-end so the fitted (a,b,P) reflect reality:
//
//   moe_permute  →  gate_up GEMM (N1,K1)  →  fused SwiGLU  →  down GEMM (N2,K2)  →  moe_unpermute
//
// mirroring dispatch_moe.cpp::dispatch_moe_internal's run_routed_ffn lambda
// (Step 2 permute → Step 3 gate_up GEMM → Step 4 SwiGLU → Step 5 down GEMM →
// Step 6 unpermute). The chain is driven entirely through the CUDA-free
// ExpertDevice interface (moe_permute / nvfp4_grouped_gemm / fused_swiglu /
// moe_unpermute), so this .cpp stays CUDA-free (INV-GPU-1 / INV-LOADER-CAL-3).
//
// Decode shape: `c` active experts, 1 token each (M_e=1). The permute fans out a
// single hidden row to `c` expert slots (topk = c, one token routed to all c
// distinct experts), giving expanded_tokens = c — the same shape the gate_up /
// down GEMMs see at decode. The two GEMMs reuse measure_grouped_gemm_us's exact
// synthetic NVFP4 buffer + scale_A sizing recipe (sf_M = c * sf_rows_per_expert,
// NOT pad(total_M,128) — decode tokens=1 → each expert pads to 128 rows), once
// per projection with its own N/K + scales + workspace.
//
// GPU-UNTESTED ASSUMPTIONS (flagged for the human GPU run — expect runtime buffer
// bugs like the prior scale_A OOB):
//  (1) topk = num_experts: moe_permute fans the single token to one row per
//      expert. permuted_input must hold expanded_tokens (= num_experts) rows.
//      expert_offsets is WRITTEN by moe_permute (device-computed from topk_indices),
//      so the synthetic gate_up/down GEMMs use our OWN fixed expert_offsets
//      (i*tokens) + problem_sizes, NOT the permute output — the GEMM timing is
//      shape-driven and identical either way; we only need permute to RUN at the
//      right size to charge its cost.
//  (2) fused_swiglu consumes the gate_up GEMM output [expanded_tokens, N1] as
//      [tokens, 2*d] with d = N1/2 (= moe_intermediate_size), producing [tokens, d].
//      N1 (=compute_N) is even (2*moe_intermediate_size) so d = N1/2 is exact.
//  (3) the down GEMM input is fed a FRESH synthetic FP4 buffer (NOT the SwiGLU
//      BF16 output requantized) — the SwiGLU→FP4 requant (launch_fused_swiglu_
//      nvfp4_quant in the real path) is a per-element kernel NOT on the ExpertDevice
//      interface, so its cost (plus activation-quant + gather_alphas + populate_meta,
//      all small per-element/per-expert kernels) is NOT captured here. This is the
//      one documented residual vs the real chain; the dominant grouped-GEMM +
//      permute/unpermute + launch/sync costs ARE captured. See TD-LOADER-COMPUTE-
//      PARALLELISM residual note.
//  (4) moe_unpermute reads expert_output [expanded_tokens, hidden] and topk_weights
//      [tokens, topk] to reduce back to [tokens, hidden]; we feed the down GEMM's
//      BF16 output (N2 = hidden) as its permuted_output and synthetic 1.0 weights.
double measure_full_ffn_us(ExpertDevice* ed, DeviceBackend* be, int num_experts,
                           int N1, int K1, int N2, int K2, int tokens,
                           int warmup, int iters) {
  if (!ed || !be || num_experts <= 0 || N1 <= 0 || K1 <= 0 || N2 <= 0 || K2 <= 0 ||
      tokens <= 0)
    return 0.0;
  constexpr int kGroup = 16;                    // NVFP4 group size
  const int hidden        = K1;                 // permute hidden dim = gate_up K
  const int topk          = num_experts;        // fan the token to all c experts
  const int expanded      = tokens * num_experts;  // permuted rows (decode: = c)
  const int swiglu_d      = N1 / 2;             // SwiGLU intermediate (= moe_intermediate_size)

  // ── Per-GEMM synthetic NVFP4 buffer builder (mirrors measure_grouped_gemm_us). ──
  // Returns false on any device-alloc failure (caller treats as unmeasured).
  struct GemmBufs {
    ExpertDevice* ed = nullptr;
    void* A = nullptr; void* B = nullptr; void* D = nullptr;
    void* sA = nullptr; void* sB = nullptr; void* al = nullptr;
    void* off = nullptr; void* sfo = nullptr; void* pr = nullptr; void* ws = nullptr;
    size_t ws_bytes = 0;
    compute::Nvfp4GroupedGemmParams p{};
    ~GemmBufs() {
      if (!ed) return;
      ed->device_free(A);  ed->device_free(B);  ed->device_free(D);
      ed->device_free(sA); ed->device_free(sB); ed->device_free(al);
      ed->device_free(off); ed->device_free(sfo); ed->device_free(pr);
      if (ws) ed->device_free(ws);
    }
  };

  // Build + upload one grouped-GEMM's buffers for `ne` experts, N, K, M/expert.
  // `out_D` (optional) lets the caller chain the previous projection's output as
  // this GEMM's D target is unused for input — each GEMM writes its own D here
  // (the chain only needs each op to RUN at the right shape for timing).
  auto build_gemm = [&](GemmBufs& g, int ne, int N, int K, int M_per) -> bool {
    g.ed = ed;
    const int total_M = ne * M_per;
    const int half_K  = K / 2;                  // FP4: 2 nibbles/byte
    const int sf_K = ((K / kGroup + 3) / 4) * 4;
    const int sf_N = ((N + 127) / 128) * 128;
    const int sf_rows_per_expert = ((M_per + 127) / 128) * 128;
    const int sf_M = ne * sf_rows_per_expert;   // scale_A spans ALL experts' pads

    std::vector<uint8_t> h_A(static_cast<size_t>(total_M) * half_K, 0x44);
    std::vector<uint8_t> h_B(static_cast<size_t>(ne) * N * half_K, 0x44);
    std::vector<uint8_t> h_sA(static_cast<size_t>(sf_M) * sf_K, 0x38);
    std::vector<uint8_t> h_sB(static_cast<size_t>(ne) * sf_N * sf_K, 0x38);
    std::vector<float>   h_alphas(ne, 1.0f);
    std::vector<int32_t> h_off(ne + 1);
    std::vector<int32_t> h_sf_off(ne + 1);
    std::vector<int32_t> h_prob(static_cast<size_t>(ne) * 3);
    for (int i = 0; i <= ne; ++i) {
      h_off[i]    = i * M_per;
      h_sf_off[i] = i * sf_rows_per_expert;
    }
    for (int i = 0; i < ne; ++i) {
      h_prob[i * 3 + 0] = M_per;
      h_prob[i * 3 + 1] = N;
      h_prob[i * 3 + 2] = K;
    }
    const size_t out_bytes = static_cast<size_t>(total_M) * N * sizeof(uint16_t);  // bf16
    g.ws_bytes = compute::query_nvfp4_grouped_gemm_workspace_size(
        ne, N, K, compute::GemmOutputDtype::kBFloat16);

    g.A   = ed->device_alloc(h_A.size());
    g.B   = ed->device_alloc(h_B.size());
    g.D   = ed->device_alloc(out_bytes);
    g.sA  = ed->device_alloc(h_sA.size());
    g.sB  = ed->device_alloc(h_sB.size());
    g.al  = ed->device_alloc(h_alphas.size() * sizeof(float));
    g.off = ed->device_alloc(h_off.size() * sizeof(int32_t));
    g.sfo = ed->device_alloc(h_sf_off.size() * sizeof(int32_t));
    g.pr  = ed->device_alloc(h_prob.size() * sizeof(int32_t));
    g.ws  = g.ws_bytes ? ed->device_alloc(g.ws_bytes) : nullptr;
    if (!g.A || !g.B || !g.D || !g.sA || !g.sB || !g.al || !g.off || !g.sfo ||
        !g.pr || (g.ws_bytes != 0 && !g.ws))
      return false;

    be->memcpy_h2d(g.A, h_A.data(), h_A.size());
    be->memcpy_h2d(g.B, h_B.data(), h_B.size());
    be->memcpy_h2d(g.sA, h_sA.data(), h_sA.size());
    be->memcpy_h2d(g.sB, h_sB.data(), h_sB.size());
    be->memcpy_h2d(g.al, h_alphas.data(), h_alphas.size() * sizeof(float));
    be->memcpy_h2d(g.off, h_off.data(), h_off.size() * sizeof(int32_t));
    be->memcpy_h2d(g.sfo, h_sf_off.data(), h_sf_off.size() * sizeof(int32_t));
    be->memcpy_h2d(g.pr, h_prob.data(), h_prob.size() * sizeof(int32_t));

    g.p.num_experts    = ne;
    g.p.N              = N;
    g.p.K              = K;
    g.p.A_base         = g.A;
    g.p.B_base         = g.B;
    g.p.D_base         = g.D;
    g.p.scale_A_base   = g.sA;
    g.p.scale_B_base   = g.sB;
    g.p.alphas         = static_cast<const float*>(g.al);
    g.p.expert_offsets = static_cast<const int32_t*>(g.off);
    g.p.sf_offsets     = static_cast<const int32_t*>(g.sfo);
    g.p.problem_sizes  = static_cast<const int32_t*>(g.pr);
    g.p.output_dtype   = compute::GemmOutputDtype::kBFloat16;
    return true;
  };

  GemmBufs gate_up, down;
  if (!build_gemm(gate_up, num_experts, N1, K1, tokens)) return 0.0;
  if (!build_gemm(down,    num_experts, N2, K2, tokens)) return 0.0;

  // ── Permute / SwiGLU / unpermute buffers (BF16, elem_size=2). ──
  // permute: 1 token → topk(=c) experts. hidden_states [tokens, hidden] BF16;
  // permuted_input [expanded, hidden] BF16; topk_indices [tokens, topk] = {0..c-1}.
  const size_t bf16 = sizeof(uint16_t);
  std::vector<uint16_t> h_hidden(static_cast<size_t>(tokens) * hidden, 0x3C00);  // ~1.0 bf16
  std::vector<int32_t>  h_topk_idx(static_cast<size_t>(tokens) * topk);
  std::vector<float>    h_topk_w(static_cast<size_t>(tokens) * topk, 1.0f);
  for (int t = 0; t < tokens; ++t)
    for (int k = 0; k < topk; ++k)
      h_topk_idx[t * topk + k] = k;  // each token routed to all c distinct experts

  // SwiGLU consumes the gate_up GEMM D [expanded, N1] as [expanded, 2*swiglu_d] →
  // [expanded, swiglu_d]. The down GEMM then runs on its own synthetic FP4 input;
  // moe_unpermute reduces the down GEMM D [expanded, N2(=hidden)] → [tokens, hidden].
  const size_t permuted_bytes = static_cast<size_t>(expanded) * hidden * bf16;
  const size_t swiglu_out_bytes = static_cast<size_t>(expanded) * swiglu_d * bf16;
  const size_t unperm_out_bytes = static_cast<size_t>(tokens) * hidden * bf16;
  const size_t permute_ws = compute::query_moe_permute_workspace_size(tokens, topk, num_experts);

  void* d_hidden     = ed->device_alloc(h_hidden.size() * bf16);
  void* d_topk_idx   = ed->device_alloc(h_topk_idx.size() * sizeof(int32_t));
  void* d_topk_w     = ed->device_alloc(h_topk_w.size() * sizeof(float));
  void* d_permuted   = ed->device_alloc(permuted_bytes);
  void* d_swiglu_out = ed->device_alloc(swiglu_out_bytes);
  void* d_unperm_out = ed->device_alloc(unperm_out_bytes);
  void* d_perm_eoff  = ed->device_alloc(static_cast<size_t>(num_experts + 1) * sizeof(int32_t));
  void* d_perm_s2d   = ed->device_alloc(static_cast<size_t>(expanded) * sizeof(int32_t));
  void* d_perm_idx   = ed->device_alloc(static_cast<size_t>(expanded) * sizeof(int32_t));
  void* d_perm_ws    = permute_ws ? ed->device_alloc(permute_ws) : nullptr;

  void* stream = be->create_stream();
  void* ev0 = be->create_timing_event();
  void* ev1 = be->create_timing_event();

  double result = 0.0;
  const bool ok = d_hidden && d_topk_idx && d_topk_w && d_permuted && d_swiglu_out &&
                  d_unperm_out && d_perm_eoff && d_perm_s2d && d_perm_idx &&
                  (permute_ws == 0 || d_perm_ws) && stream && ev0 && ev1;
  if (ok) {
    be->memcpy_h2d(d_hidden, h_hidden.data(), h_hidden.size() * bf16);
    be->memcpy_h2d(d_topk_idx, h_topk_idx.data(), h_topk_idx.size() * sizeof(int32_t));
    be->memcpy_h2d(d_topk_w, h_topk_w.data(), h_topk_w.size() * sizeof(float));

    compute::FusedSwigluParams sw{};
    sw.num_tokens = expanded;
    sw.d          = swiglu_d;

    // One full chain pass (mirrors run_routed_ffn Step 2→6).
    auto run_chain = [&]() {
      ed->moe_permute(d_permuted,
                      static_cast<int32_t*>(d_perm_eoff),
                      static_cast<int32_t*>(d_perm_s2d),
                      static_cast<int32_t*>(d_perm_idx),
                      d_hidden,
                      static_cast<const int32_t*>(d_topk_idx),
                      tokens, topk, hidden, num_experts,
                      /*elem_size_bytes=*/2, d_perm_ws, stream);
      ed->nvfp4_grouped_gemm(gate_up.p, gate_up.ws, gate_up.ws_bytes, stream);
      ed->fused_swiglu(d_swiglu_out, gate_up.D, sw, /*elem_size_bytes=*/2, stream);
      ed->nvfp4_grouped_gemm(down.p, down.ws, down.ws_bytes, stream);
      ed->moe_unpermute(d_unperm_out, down.D,
                        static_cast<const float*>(d_topk_w),
                        static_cast<const int32_t*>(d_perm_s2d),
                        tokens, topk, hidden, /*elem_size_bytes=*/2, stream);
    };

    for (int i = 0; i < warmup; ++i) run_chain();
    be->device_sync();
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      be->record_event(ev0, stream);
      run_chain();
      be->record_event(ev1, stream);
      be->device_sync();
      const float ms = be->event_elapsed_ms(ev0, ev1);
      if (ms >= 0.0f) samples.push_back(static_cast<double>(ms) * 1000.0);  // ms → µs
    }
    result = median(std::move(samples));
  }

  if (ev0) be->destroy_event(ev0);
  if (ev1) be->destroy_event(ev1);
  if (stream) be->destroy_stream(stream);
  ed->device_free(d_hidden);     ed->device_free(d_topk_idx); ed->device_free(d_topk_w);
  ed->device_free(d_permuted);   ed->device_free(d_swiglu_out);
  ed->device_free(d_unperm_out); ed->device_free(d_perm_eoff);
  ed->device_free(d_perm_s2d);   ed->device_free(d_perm_idx);
  if (d_perm_ws) ed->device_free(d_perm_ws);
  return result;
}

// ── Reconciliation pass (recon_overhead/recon_added, §3.4) ──
//
// recon = max_j recon_overhead[j] + Σ_j recon_added[j]: a fixed collective latency
// plus a per-participant bandwidth term. The real engine reconciliation is an NCCL
// allreduce (SUM) of the per-GPU routed `moe_output` across the TP set (13c-8). We
// measure THAT — a true cross-GPU collective over the calibrated device set — not a
// single-GPU proxy.
//
// INV-GPU-1 boundary: we drive the collective through the CUDA-encapsulated
// `parallelism::CollectiveBackend` (the real NCCL backend), whose raw `nccl*`/
// `<cuda_runtime.h>` calls live entirely in nccl_collective_backend.cpp. This .cpp
// stays CUDA/NCCL-free: it sees only `void*` comm/stream/buffer handles plus the
// SDK-free datatype/op constants (kCollBfloat16/kCollSum). Device memory + timing
// events come from each DeviceBackend (per-GPU, INV-BH-1).
//
// `ncclCommInitAll` brings up one communicator per rank inside this single process
// in one call — no per-rank threads, no full engine wiring — which is exactly the
// minimal bring-up the calib harness needs. We hold the GIL of nothing here (pure
// init-time C++); the comms are torn down before return.
//
// ReconResult carries the two timings the §3.4 split needs: a fixed-latency floor
// (tiny payload) and the full-payload allreduce. The caller fits per-device
// overhead/added so the rollup reconstructs the measured full-collective time.
struct ReconResult {
  bool   measured  = false;  // false → no real collective ran (fall back / leave defaults)
  double floor_us  = 0.0;    // fixed collective latency (tiny payload allreduce)
  double full_us   = 0.0;    // allreduce of `payload_bytes` across the whole set
};

// Time a SUM-allreduce of `count_elems` BF16 elements across ALL ranks, median over
// `iters` after `warmup`. Each rank's send/recv buffer lives on its own GPU; rank 0's
// DeviceBackend supplies the timing events (the collective end-to-end latency is the
// same wall-clock for every participant — a single allreduce completes together).
double measure_allreduce_us(parallelism::CollectiveBackend& coll,
                            const std::vector<void*>& comms,
                            const std::vector<DeviceBackend*>& backends,
                            const std::vector<void*>& sendbufs,
                            const std::vector<void*>& recvbufs,
                            const std::vector<void*>& streams,
                            size_t count_elems, int warmup, int iters) {
  const int n = static_cast<int>(backends.size());
  if (n == 0 || count_elems == 0) return 0.0;

  auto run_once = [&]() {
    // One grouped allreduce across all ranks (NCCL requires the group bracket so the
    // per-rank calls are launched as a single collective and do not deadlock).
    coll.group_begin();
    for (int r = 0; r < n; ++r)
      coll.allreduce(sendbufs[r], recvbufs[r], count_elems,
                     parallelism::kCollBfloat16, parallelism::kCollSum,
                     comms[r], streams[r]);
    coll.group_end();
  };

  for (int i = 0; i < warmup; ++i) run_once();
  for (int r = 0; r < n; ++r) backends[r]->device_sync();

  DeviceBackend* timer = backends[0];
  void* ev0 = timer->create_timing_event();
  void* ev1 = timer->create_timing_event();
  double result = 0.0;
  if (ev0 && ev1) {
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      timer->record_event(ev0, streams[0]);
      run_once();
      timer->record_event(ev1, streams[0]);
      for (int r = 0; r < n; ++r) backends[r]->device_sync();
      const float ms = timer->event_elapsed_ms(ev0, ev1);
      if (ms >= 0.0f) samples.push_back(static_cast<double>(ms) * 1000.0);  // ms → µs
    }
    result = median(std::move(samples));
  }
  if (ev0) timer->destroy_event(ev0);
  if (ev1) timer->destroy_event(ev1);
  return result;
}

// Bring up a real NCCL communicator over the calibrated device set and time the
// routed-output allreduce at a tiny floor payload and at `payload_bytes`. Returns
// {measured=false} (caller leaves recon at defaults) when the set is degenerate
// (< 2 GPUs — no cross-GPU collective), or if comm bring-up / buffer allocation
// fails. All comms + device buffers are released before return.
ReconResult measure_recon_collective(const std::vector<DeviceBackend*>& backends,
                                     size_t payload_bytes, int warmup, int iters) {
  ReconResult rr;
  const int n = static_cast<int>(backends.size());
  if (n < 2 || payload_bytes == 0) return rr;  // need ≥2 GPUs to cross a link

  // BF16 routed-output payload (the SUM-allreduce element type, 13c-8).
  const size_t elem_bytes  = sizeof(uint16_t);
  const size_t full_elems  = std::max<size_t>(1, payload_bytes / elem_bytes);
  const size_t floor_elems = 1;  // tiny payload → fixed collective-latency floor
  const size_t buf_bytes   = full_elems * elem_bytes;

  std::unique_ptr<parallelism::CollectiveBackend> coll;
  try {
    coll = parallelism::make_nccl_collective_backend();
  } catch (const std::exception& e) {
    spdlog::warn("gpu_loader: recon NCCL backend unavailable ({}); leaving recon at defaults",
                 e.what());
    return rr;
  }
  if (!coll) return rr;

  std::vector<int> device_ids(n);
  for (int r = 0; r < n; ++r) device_ids[r] = backends[r]->gpu().id;

  std::vector<void*> comms;
  try {
    comms = coll->create_comms(device_ids);  // ncclCommInitAll: one comm per rank
  } catch (const std::exception& e) {
    spdlog::warn("gpu_loader: recon comm bring-up failed ({}); leaving recon at defaults",
                 e.what());
    return rr;
  }

  std::vector<void*> sendbufs(n, nullptr), recvbufs(n, nullptr), streams(n, nullptr);
  bool ok = static_cast<int>(comms.size()) == n;
  for (int r = 0; ok && r < n; ++r) {
    DeviceBackend* be = backends[r];
    sendbufs[r] = be->device_alloc(buf_bytes);
    recvbufs[r] = be->device_alloc(buf_bytes);
    streams[r]  = be->create_stream();
    if (!sendbufs[r] || !recvbufs[r] || !streams[r]) ok = false;
    else be->memset_async(sendbufs[r], 0, buf_bytes, streams[r]);  // defined contents
  }
  for (int r = 0; r < n; ++r) backends[r]->device_sync();

  if (ok) {
    rr.floor_us = measure_allreduce_us(*coll, comms, backends, sendbufs, recvbufs, streams,
                                       floor_elems, warmup, iters);
    rr.full_us  = measure_allreduce_us(*coll, comms, backends, sendbufs, recvbufs, streams,
                                       full_elems, warmup, iters);
    rr.measured = (rr.full_us > 0.0);
  } else {
    spdlog::warn("gpu_loader: recon buffer/stream alloc failed; leaving recon at defaults");
  }

  for (int r = 0; r < n; ++r) {
    DeviceBackend* be = backends[r];
    if (streams[r])  be->destroy_stream(streams[r]);
    if (sendbufs[r]) be->device_free(sendbufs[r]);
    if (recvbufs[r]) be->device_free(recvbufs[r]);
  }
  for (void* cm : comms) coll->destroy_comm(cm);
  return rr;
}

}  // namespace

ComputeCurve fit_compute_curve(const std::vector<int>& counts,
                               const std::vector<double>& us) {
  ComputeCurve best{0.0, 0.0, 1};
  const size_t n = std::min(counts.size(), us.size());
  if (n == 0) return best;

  int max_c = 1;
  for (size_t i = 0; i < n; ++i) max_c = std::max(max_c, counts[i]);

  // For a fixed P, ceil(c/P) is known per sample, so t = a*c + b*g is linear in
  // (a,b); solve the 2×2 normal equations (least squares). Grid-search P, keep the
  // one with the smallest SSE (ties resolved toward the smaller P → simplest model).
  double best_sse = std::numeric_limits<double>::infinity();
  for (int P = 1; P <= max_c; ++P) {
    double Scc = 0, Scg = 0, Sgg = 0, Sct = 0, Sgt = 0;  // normal-equation accumulators
    for (size_t i = 0; i < n; ++i) {
      const double c = static_cast<double>(counts[i]);
      const double g = (counts[i] <= 0) ? 0.0
                       : static_cast<double>(1 + (counts[i] - 1) / P);  // ceil(c/P)
      const double t = us[i];
      Scc += c * c; Scg += c * g; Sgg += g * g; Sct += c * t; Sgt += g * t;
    }
    const double det = Scc * Sgg - Scg * Scg;
    double a, b;
    if (std::abs(det) < 1e-12) {
      // Degenerate (e.g. all c map to one g, or a single point): fit a only, b=0.
      a = (Scc > 0) ? (Sct / Scc) : 0.0;
      b = 0.0;
    } else {
      a = (Sgg * Sct - Scg * Sgt) / det;
      b = (Scc * Sgt - Scg * Sct) / det;
    }
    // Clamp to physically meaningful non-negative costs before scoring.
    if (a < 0.0) a = 0.0;
    if (b < 0.0) b = 0.0;
    double sse = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double c = static_cast<double>(counts[i]);
      const double g = (counts[i] <= 0) ? 0.0
                       : static_cast<double>(1 + (counts[i] - 1) / P);
      const double pred = a * c + b * g;
      const double r = pred - us[i];
      sse += r * r;
    }
    if (sse < best_sse - 1e-9) {  // strict improvement → prefer smaller P on ties
      best_sse = sse;
      best = ComputeCurve{a, b, P};
    }
  }
  return best;
}

CalibrationMode parse_calibration_mode(const std::string& s) {
  if (s == "loaded") return CalibrationMode::kLoaded;
  if (s == "full")   return CalibrationMode::kFull;
  return CalibrationMode::kQuick;  // "quick" or unknown
}

const char* calibration_mode_name(CalibrationMode m) {
  switch (m) {
    case CalibrationMode::kLoaded: return "loaded";
    case CalibrationMode::kQuick:  return "quick";
    case CalibrationMode::kFull:   return "full";
  }
  return "quick";
}

CalibrationConfig quick_calibration_config() {
  CalibrationConfig c;
  c.warmup_iters    = 3;
  c.timed_iters     = 20;
  c.footprint_bytes = 0.0;  // single buffer (cache-hot, fast)
  return c;
}

CalibrationConfig full_calibration_config() {
  CalibrationConfig c;
  c.warmup_iters    = 5;
  c.timed_iters     = 30;
  c.footprint_bytes = 32.0 * 1024.0 * 1024.0 * 1024.0;  // 32 GiB > ~16 GiB HBM cache → DDR
  c.measure_compute = true;   // full run fits the per-device compute curve...
  c.measure_recon   = true;   // ...and the reconciliation overhead/added time
  return c;
}

CalibrationConfig config_for_mode(CalibrationMode m) {
  return m == CalibrationMode::kFull ? full_calibration_config() : quick_calibration_config();
}

LoaderConstants load_or_calibrate(CalibrationMode mode, const std::string& path,
                                  const std::vector<DeviceBackend*>& backends, NumaManager& numa,
                                  const std::vector<ExpertDevice*>& experts) {
  if (mode == CalibrationMode::kLoaded) {
    if (!path.empty() && std::filesystem::exists(path)) {
      try {
        LoaderConstants c = load(path);
        spdlog::info("gpu_loader: loaded constants from {}", path);
        return c;
      } catch (const std::exception& e) {
        spdlog::warn("gpu_loader: failed to load {} ({}); recalibrating (full)", path, e.what());
      }
    } else {
      spdlog::info("gpu_loader: no calibration file at '{}'; running full calibration to produce it",
                   path);
    }
    mode = CalibrationMode::kFull;  // fall through to produce the file
  }
  LoaderConstants c = calibrate(backends, numa, config_for_mode(mode), experts);
  if (!path.empty()) {
    try {
      save(c, path);
      spdlog::info("gpu_loader: wrote {} calibration to {}", calibration_mode_name(mode), path);
    } catch (const std::exception& e) {
      spdlog::warn("gpu_loader: could not write {} ({})", path, e.what());
    }
  }
  return c;
}

LoaderConstants calibrate(const std::vector<DeviceBackend*>& backends,
                          NumaManager& numa, const CalibrationConfig& cfg,
                          const std::vector<ExpertDevice*>& experts) {
  LoaderConstants c;
  c.source       = "calibrated";
  c.expert_bytes = cfg.expert_bytes;
  c.num_devices  = static_cast<int>(backends.size());

  // Banks = ALL host memory nodes (not just GPU-attached) — an expert may source
  // from any NUMA node (the pinned arena spans nodes; spill lands on GPU-less
  // nodes), so every node's egress + per-device cross-NUMA rate + contention must
  // be measured regardless of how many GPUs are visible (cfg.bank_nodes overrides).
  // include_hbm_banks (default on) additionally enrolls the detected CPU-less
  // HBM / device-memory nodes (TD-NUMA-HBM-BANKS): the arena spills experts onto
  // them every run, so the solver's cost model needs their egress rows too. Their
  // footprint is capped to the node's free memory (an uncapped 32 GiB rotating
  // buffer into a 16 GB node is the original CONSTRAINT_MEMORY_POLICY OOM) and
  // they are measured from a thread pinned to their nearest CPU node.
  std::vector<int> banks = cfg.bank_nodes.empty()
      ? (cfg.include_hbm_banks ? numa.all_banks_including_hbm() : numa.all_memory_nodes())
      : cfg.bank_nodes;
  c.num_banks = static_cast<int>(banks.size());

  const size_t xfer_bytes = static_cast<size_t>(cfg.expert_bytes);
  const size_t lat_bytes  = static_cast<size_t>(cfg.latency_probe_bytes);
  // Rotating footprint: round to a whole number of expert chunks, min 1.
  const size_t rot_chunks = (cfg.footprint_bytes >= cfg.expert_bytes)
                                ? static_cast<size_t>(cfg.footprint_bytes / cfg.expert_bytes)
                                : 1;
  const size_t rate_total = rot_chunks * xfer_bytes;

  // Per-bank rotating footprint: DDR banks keep the full configured footprint
  // (path preserved EXACTLY); HBM banks are capped to fit the node —
  // min(rate_total, node_free * hbm_footprint_frac), floor one expert chunk —
  // so the MPOL_BIND'd buffer can never over-commit the small node.
  auto bank_rate_total = [&](int node) -> size_t {
    if (!numa.node_is_hbm(node)) return rate_total;
    const double frac = std::clamp(cfg.hbm_footprint_frac, 0.0, 1.0);
    const size_t cap  = static_cast<size_t>(
        static_cast<double>(numa.node_free_bytes(node)) * frac);
    const size_t chunks = std::max<size_t>(1, std::min(rot_chunks, cap / xfer_bytes));
    const size_t total  = chunks * xfer_bytes;
    if (total < rate_total)
      spdlog::info("gpu_loader: HBM bank node {}: footprint capped {:.1f} -> {:.2f} GiB "
                   "(node free {:.2f} GiB, frac {:.2f}, cpu-affinity node {})",
                   node, static_cast<double>(rate_total) / (1024.0 * 1024.0 * 1024.0),
                   static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0),
                   static_cast<double>(numa.node_free_bytes(node)) /
                       (1024.0 * 1024.0 * 1024.0),
                   frac, numa.hbm_cpu_affinity_node(node));
    return total;
  };
  // Locality node for tiering/contention classes: an HBM bank has no CPUs (and no
  // GPU reports it as its attach node), so a device counts as "local" to it when
  // the device sits on the bank's nearest-CPU node — the same physical
  // socket/cluster the HBM hangs off.
  auto bank_local_node = [&](int node) -> int {
    const int aff = numa.hbm_cpu_affinity_node(node);
    return (aff >= 0) ? aff : node;
  };

  // Per-device identity (numa_node from topology; name+uuid from the CUDA query so
  // the JSON is self-describing — a loader can detect a wrong-machine / reordered
  // file by UUID, INV-LOADER-CAL-6). Compute curve left at defaults.
  c.devices.resize(c.num_devices);
  for (int d = 0; d < c.num_devices; ++d) {
    const int pos = backends[d]->gpu().position;
    c.devices[d].position  = pos;
    c.devices[d].numa_node = numa.gpu_numa_node(pos);
    try {
      const core::GpuHardwareInfo hw = core::query_gpu_info(backends[d]->gpu().id);
      c.devices[d].name = hw.device_name;
      c.devices[d].uuid = hw.uuid;
    } catch (const std::exception& e) {
      spdlog::warn("gpu_loader: query_gpu_info(id={}) failed ({}); device identity left empty",
                   backends[d]->gpu().id, e.what());
    }
  }

  // GPU×NUMA transfer matrix + per-(bank,device) latency. HBM banks: capped
  // footprint + the measuring thread pinned to the bank's nearest CPU node for
  // the buffer alloc/first-touch AND the timing loop (the scope is a no-op for
  // DDR banks — hbm_cpu_affinity_node() is -1 there).
  c.matrix.assign(c.num_banks, std::vector<TransferCell>(c.num_devices));
  for (int b = 0; b < c.num_banks; ++b) {
    const size_t bank_total = bank_rate_total(banks[b]);
    memory::ScopedThreadNodeBind pin(numa, numa.hbm_cpu_affinity_node(banks[b]));
    PinnedNumaBuffer rbuf(numa, bank_total, banks[b]);
    PinnedNumaBuffer lbuf(numa, lat_bytes, banks[b]);
    for (int d = 0; d < c.num_devices; ++d) {
      DeviceBackend* be = backends[d];
      const double rate = measure_h2d_us(be, rbuf.base(), bank_total, xfer_bytes,
                                         cfg.warmup_iters, cfg.timed_iters);
      const double lat  = measure_h2d_us(be, lbuf.base(), lat_bytes, lat_bytes,
                                         cfg.warmup_iters, cfg.timed_iters);
      const bool local  = (c.devices[d].numa_node == bank_local_node(banks[b]));
      c.matrix[b][d] = TransferCell{rate, local ? 1 : 2, lat};
    }
  }

  // Per-device xfer_lat = min latency cell across banks (its local one).
  for (int d = 0; d < c.num_devices; ++d) {
    double lat = 0.0;
    for (int b = 0; b < c.num_banks; ++b) {
      const double l = c.matrix[b][d].lat_us;
      if (l > 0.0 && (lat == 0.0 || l < lat)) lat = l;
    }
    c.devices[d].xfer_lat_us = lat;
  }

  // Per-bank egress = the rate from a device local to the bank (tier 1), else
  // the min rate observed for that bank. Contention defaults to strict serial.
  c.banks.resize(c.num_banks);
  for (int b = 0; b < c.num_banks; ++b) {
    c.banks[b].node              = banks[b];
    c.banks[b].is_hbm            = numa.node_is_hbm(banks[b]);
    c.banks[b].cpu_affinity_node = numa.hbm_cpu_affinity_node(banks[b]);
    // Bank's intrinsic egress = the FASTEST (min-µs) local (tier-1) device rate;
    // fall back to the fastest of any tier. A PCIe-limited GPU local to the bank
    // must NOT define the bank's egress (its own link, not the channel, is the cap).
    double local_best = 0.0, any_best = 0.0;
    for (int d = 0; d < c.num_devices; ++d) {
      const auto& cell = c.matrix[b][d];
      if (cell.rate_us <= 0.0) continue;
      if (any_best == 0.0 || cell.rate_us < any_best) any_best = cell.rate_us;
      if (cell.tier == 1 && (local_best == 0.0 || cell.rate_us < local_best)) local_best = cell.rate_us;
    }
    c.banks[b].egress_us  = (local_best > 0.0) ? local_best : any_best;
    c.banks[b].contention = 1.0;
  }

  // ncf: tier1 = 1.0; tier2 = mean(cross rate) / mean(local rate).
  {
    double loc = 0.0, cross = 0.0; int ln = 0, cn = 0;
    for (int b = 0; b < c.num_banks; ++b)
      for (int d = 0; d < c.num_devices; ++d) {
        const auto& cell = c.matrix[b][d];
        if (cell.rate_us <= 0.0) continue;
        if (cell.tier == 1) { loc += cell.rate_us; ++ln; }
        else                { cross += cell.rate_us; ++cn; }
      }
    const double ncf2 = (ln && cn && loc > 0.0) ? (cross / cn) / (loc / ln) : 1.0;
    c.ncf = {0.0, 1.0, ncf2};
  }

  // Same-bank two-device contention. numa_contention[b] = clamp(factor-1, 0, 1),
  // factor = dual_us / solo_us (≈2 → strict serial sharing → contention 1).
  if (cfg.measure_contention && c.num_devices >= 2) {
    for (int b = 0; b < c.num_banks; ++b) {
      const size_t bank_total = bank_rate_total(banks[b]);
      memory::ScopedThreadNodeBind pin(numa, numa.hbm_cpu_affinity_node(banks[b]));
      PinnedNumaBuffer rbuf(numa, bank_total, banks[b]);
      const char* base = rbuf.base();
      const size_t K = std::max<size_t>(1, bank_total / xfer_bytes);
      std::vector<double> bank_c;
      for (int a = 0; a < c.num_devices; ++a) {
        for (int e = a + 1; e < c.num_devices; ++e) {
          DeviceBackend* ba = backends[a];
          DeviceBackend* bb = backends[e];
          // Disjoint regions: A reads chunks [0,half), B reads [half,K) — they
          // contend on the bank's DDR egress without one warming the HBM cache for
          // the other.
          const size_t half = (K > 1) ? K / 2 : 1;
          const size_t Ka   = half;                       // A: [0, half)
          const size_t Boff = (K > 1) ? half : 0;         // B: [half, K)
          const size_t Kb   = (K > Boff) ? (K - Boff) : 1;
          // Per-region SOLO baselines (same disjoint footprint, no peer) — isolates
          // contention from the cache effect of a smaller footprint.
          const double pa_solo = measure_h2d_us(ba, base, Ka * xfer_bytes, xfer_bytes,
                                                cfg.warmup_iters, cfg.timed_iters);
          const double pb_solo = measure_h2d_us(bb, base + Boff * xfer_bytes, Kb * xfer_bytes,
                                                xfer_bytes, cfg.warmup_iters, cfg.timed_iters);
          if (pa_solo <= 0.0 || pb_solo <= 0.0) continue;
          void* da = ba->device_alloc(xfer_bytes);
          void* db = bb->device_alloc(xfer_bytes);
          void* sa = ba->create_stream();
          void* sb = bb->create_stream();
          void* a0 = ba->create_timing_event();
          void* a1 = ba->create_timing_event();
          void* b0 = bb->create_timing_event();
          void* b1 = bb->create_timing_event();
          if (da && db && sa && sb && a0 && a1 && b0 && b1) {
            const int N = cfg.timed_iters;
            // Both devices read their disjoint regions concurrently (overlap, sync).
            ba->record_event(a0, sa);
            for (int i = 0; i < N; ++i)
              ba->memcpy_h2d_async(da, base + (static_cast<size_t>(i) % Ka) * xfer_bytes, xfer_bytes, sa);
            ba->record_event(a1, sa);
            bb->record_event(b0, sb);
            for (int i = 0; i < N; ++i)
              bb->memcpy_h2d_async(db, base + (Boff + static_cast<size_t>(i) % Kb) * xfer_bytes, xfer_bytes, sb);
            bb->record_event(b1, sb);
            ba->device_sync();
            bb->device_sync();
            const float ma = ba->event_elapsed_ms(a0, a1);
            const float mb = bb->event_elapsed_ms(b0, b1);
            const double pa = (ma >= 0.0f) ? (ma * 1000.0 / N) : pa_solo;
            const double pb = (mb >= 0.0f) ? (mb * 1000.0 / N) : pb_solo;

            const bool a_local = (c.devices[a].numa_node == bank_local_node(banks[b]));
            const bool e_local = (c.devices[e].numa_node == bank_local_node(banks[b]));
            ContentionConfig cc = a_local && e_local ? ContentionConfig::kTwoBound
                                  : (a_local || e_local) ? ContentionConfig::kBoundUnbound
                                                         : ContentionConfig::kTwoUnbound;
            // Worst per-device slowdown vs its own-region solo = the contention factor.
            const double factor = std::max(pa / pa_solo, pb / pb_solo);
            const double solo = std::max(pa_solo, pb_solo);
            const double dual = std::max(pa, pb);
            c.contention.push_back(ContentionSample{banks[b], c.devices[a].position,
                                                    c.devices[e].position, cc, solo, dual, factor});
            bank_c.push_back(std::clamp(factor - 1.0, 0.0, 1.0));
          }
          if (a0) ba->destroy_event(a0);
          if (a1) ba->destroy_event(a1);
          if (b0) bb->destroy_event(b0);
          if (b1) bb->destroy_event(b1);
          if (sa) ba->destroy_stream(sa);
          if (sb) bb->destroy_stream(sb);
          if (da) ba->device_free(da);
          if (db) bb->device_free(db);
        }
      }
      if (!bank_c.empty()) {
        double s = 0.0;
        for (double x : bank_c) s += x;
        c.banks[b].contention = s / static_cast<double>(bank_c.size());
      }
    }
  }

  // ── H2D-vs-CPU-DDR contention diagnostics (env-gated, DEFAULT OFF) ──
  // Diagnostic-only re-measurement of the (bank,device) matrix under CPU-side
  // DDR load; logs tables, never modifies `c` — production output byte-identical.
  if (env_flag("LS_CALIB_H2D_VS_DDR2HBM") || env_flag("LS_CALIB_H2D_VS_CPUKERNEL")) {
    std::vector<size_t> bank_totals(banks.size());
    for (size_t b = 0; b < banks.size(); ++b) bank_totals[b] = bank_rate_total(banks[b]);
    if (env_flag("LS_CALIB_H2D_VS_DDR2HBM"))
      run_h2d_vs_ddr2hbm_pass(backends, numa, cfg, banks, bank_totals, c);
    if (env_flag("LS_CALIB_H2D_VS_CPUKERNEL"))
      run_h2d_vs_cpukernel_pass(backends, numa, cfg, banks, bank_totals, c);
  }

  // ── Compute curve (compute_j, §2.3) — sweep grouped-GEMM expert count, fit (a,b,P) ──
  // The fit is dim-dependent, so the dims that produced it are recorded in the
  // LoaderConstants (compute_*); a loader must validate they match the deployed model
  // (the JSON is weight/config-specific — INV-LOADER-CAL-4).
  if (cfg.measure_compute) {
    c.compute_N      = cfg.compute_N;
    c.compute_K      = cfg.compute_K;
    c.compute_tokens = cfg.compute_tokens;
    // Down-projection dims: explicit override, else derive from gate_up (N2 = K1
    // = hidden_size, K2 = N1/2 = moe_intermediate_size). Only used in full mode.
    // Not persisted to JSON — a deterministic function of compute_N/K by default,
    // which already key the file to the model (INV-LOADER-CAL-4).
    const int down_N = (cfg.compute_down_N > 0) ? cfg.compute_down_N : cfg.compute_K;
    const int down_K = (cfg.compute_down_K > 0) ? cfg.compute_down_K : (cfg.compute_N / 2);
    const bool have_experts = static_cast<int>(experts.size()) == c.num_devices;
    if (!have_experts)
      spdlog::warn("gpu_loader: measure_compute requested but {} expert devices for {} backends; "
                   "leaving compute curves at defaults",
                   experts.size(), c.num_devices);
    if (have_experts)
      spdlog::info("gpu_loader: compute pass mode={} gate_up N={} K={} | down N={} K={} tokens={}",
                   cfg.compute_full ? "full-chain" : "gate_up-only", cfg.compute_N, cfg.compute_K,
                   down_N, down_K, cfg.compute_tokens);
    for (int d = 0; have_experts && d < c.num_devices; ++d) {
      ExpertDevice* ed = experts[d];
      DeviceBackend* be = backends[d];
      if (!ed || !be) continue;
      std::vector<int>    counts;
      std::vector<double> us;
      for (int cc : cfg.compute_counts) {
        // full mode (default): time the WHOLE routed FFN decode chain (permute →
        // gate_up GEMM → SwiGLU → down GEMM → unpermute) so the fit lands the real
        // count-independent ~365 µs/layer (INV-LOADER-CAL-5). gate_up-only mode is
        // the legacy thin lower bound (single grouped GEMM).
        const double t = cfg.compute_full
            ? measure_full_ffn_us(ed, be, cc, cfg.compute_N, cfg.compute_K, down_N, down_K,
                                  cfg.compute_tokens, cfg.warmup_iters, cfg.timed_iters)
            : measure_grouped_gemm_us(ed, be, cc, cfg.compute_N, cfg.compute_K,
                                      cfg.compute_tokens, cfg.warmup_iters, cfg.timed_iters);
        if (t > 0.0) { counts.push_back(cc); us.push_back(t); }
      }
      if (!counts.empty()) {
        c.devices[d].compute = fit_compute_curve(counts, us);
        spdlog::info("gpu_loader: dev {} compute fit a={:.3f}us b={:.3f}us P={} ({} points, {})",
                     c.devices[d].position, c.devices[d].compute.a_us, c.devices[d].compute.b_us,
                     c.devices[d].compute.P, counts.size(),
                     cfg.compute_full ? "full-chain" : "gate_up-only");
      }
    }
  }

  // ── Reconciliation (recon_overhead/recon_added, §3.4) — real NCCL allreduce ──
  // Time the routed-output SUM-allreduce across the whole calibrated set (13c-8) at a
  // tiny floor payload and at recon_payload_bytes, then split into the §3.4 terms so
  // the rollup `max_j overhead[j] + Σ_j added[j]` reconstructs the measured full time:
  //   overhead[j] = floor_us  (equal across j → max = floor_us, the fixed latency)
  //   added[j]    = (full_us - floor_us) / n  (equal share → Σ = the bandwidth term)
  // A single allreduce is one collective spanning all ranks (not per-device), so the
  // floor and the marginal are properties of the whole set; the even split is the
  // model-consistent attribution. When the set has < 2 GPUs (no link to cross) or the
  // collective can't be brought up, recon stays at defaults (rollup contributes 0).
  if (cfg.measure_recon) {
    const size_t payload = static_cast<size_t>(cfg.recon_payload_bytes);
    const ReconResult rr = measure_recon_collective(backends, payload, cfg.warmup_iters,
                                                     cfg.timed_iters);
    if (rr.measured) {
      const double added_each = std::max(0.0, rr.full_us - rr.floor_us)
                                / static_cast<double>(c.num_devices);
      for (int d = 0; d < c.num_devices; ++d) {
        c.devices[d].recon_overhead_us = rr.floor_us;
        c.devices[d].recon_added_us    = added_each;
      }
      spdlog::info("gpu_loader: recon allreduce floor={:.2f}us full={:.2f}us "
                   "(overhead/dev={:.2f} added/dev={:.2f}, {} ranks)",
                   rr.floor_us, rr.full_us, rr.floor_us, added_each, c.num_devices);
    } else {
      spdlog::warn("gpu_loader: recon collective not measured (need >=2 GPUs / NCCL); "
                   "recon_* left at defaults");
    }
    c.recon_payload_bytes = cfg.recon_payload_bytes;
  }

  int hbm_count = 0;
  for (const auto& bk : c.banks) hbm_count += bk.is_hbm ? 1 : 0;
  spdlog::info("gpu_loader: calibrated {} devices x {} banks ({} HBM, footprint={:.1f} GiB, "
               "ncf2={:.3f})",
               c.num_devices, c.num_banks, hbm_count,
               static_cast<double>(rate_total) / (1024.0 * 1024.0 * 1024.0),
               c.ncf.size() > 2 ? c.ncf[2] : 1.0);
  return c;
}

}  // namespace layerstorm::gpu_loader

// Engine module implementation.
//
// Initializes all Phase 1-9 C++ modules, allocates the IPC region,
// spawns a daemon thread (DaemonLoop), and exposes start_engine/stop_engine.
// pybind11 bindings live in python/bindings/engine_pybind.cpp.

#include "daemon/engine.h"

#include <algorithm>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <set>
#include <new>
#include <thread>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include <filesystem>

#include "compute/cuda_sm120_device_backend.h"
#include "core/attention_device.h"
#include "core/bf16_convert.h"
#include "core/cuda_hardware_query.h"
#include "core/device_backend.h"
#include "core/expert_device.h"
#include "core/gpu_loader/loader_calibration.h"
#include "core/null_attention_device.h"
#include "core/null_device_backend.h"
#include "core/null_expert_device.h"
#include "compute/cuda_sm120_expert_device.h"
#include "compute/cpu/numa_cpu_expert_device.h"  // C-6 CPU NUMA expert device
#include "compute/cpu/multi_numa_cpu_expert_device.h"  // TASK-A2 multi-node spread
#include "compute/cpu/numa_thread_pool.h"  // C-6 QA: node_physical_cpus / daemon pin
#include "compute/rope_table.h"
#include "compute/csa_hca_sm120_attention_device.h"
#include "compute/snapmla_sm120_attention_device.h"
#include "compute/tq_sm120_attention_device.h"
#include "compute/tq_init.h"
#include "config/config_parser.h"
#include "config/config_preset.h"
#include "config/config_resolver.h"
#include "config/config_validator.h"
#include "compute/graphs/graph_registry.h"
#include "compute/kernels/attention/dcp_attention_wrapper.h"
#include "compute/stream_manager.h"
#include "core/memory/expert_cache.h"
#include "core/memory/numa_manager.h"
#include "core/perf_trace.h"
#include "core/memory/arena_cache.h"
#include "core/memory/arena_ipc_client.h"
#include "core/memory/arena_loader.h"
#include "core/memory/arena_migrator.h"
#include "core/memory/arena_placement.h"
#include "core/memory/nvme_tier.h"
#include "core/memory/pinned_expert_arena.h"
#include "core/memory/page_allocator.h"
#include "core/memory/vram_allocator.h"
#include "core/statistics/acceptance_tracker.h"
#include "core/statistics/coactivation_graph.h"
#include "core/statistics/expert_stats.h"
#include "core/statistics/workload_detector.h"
#include "core/transfer/transfer_engine.h"
#include "speculation/dspark_runtime.h"       // DSP-3
#include "speculation/speculation_factory.h"
#include "daemon/spsc_ring.h"
#include <array>

#include "model/layer_registry.h"
#include "model/weight_loader/nvfp4_sfb_reformat.h"
#include "model/model_config.h"
#include "model/pinned_upload_plan.h"
#include "model/quantization/gguf_kquant.h"
#include "model/quantization/quant_interface.h"
#include "model/quantization/registry.h"
#include "model/weight_loader/tp_weight_sharder.h"
#include "model/weight_loader/weight_loader.h"
#include "model/weight_pipeline/expert_prepacker.h"
#include "model/weight_pipeline/manifest.h"
#include "model/weight_pipeline/packed_buffer_cache.h"
#include "model/weight_pipeline/prepacked_format.h"
#include "model/weight_pipeline/prepacked_source.h"
#include "parallelism/dcp_communicator.h"
#include "parallelism/dcp_executor.h"
#include "parallelism/nccl_collective_backend.h"
#include "parallelism/null_collective_backend.h"

namespace layerstorm::daemon {

// ── Singleton ───────────────────────────────────────────────────────────────

static std::unique_ptr<Engine> g_engine;
static std::mutex g_engine_mutex;

// ── Backend factories ───────────────────────────────────────────────────────

EngineBackends default_backends() {
    return EngineBackends{
        .skip_hardware_detection = false,
        .skip_weight_loading     = false,
        .skip_cuda_graphs        = false,
    };
}

EngineBackends null_backends() {
    return EngineBackends{
        .skip_hardware_detection = true,
        .skip_weight_loading     = true,
        .skip_cuda_graphs        = true,
    };
}

// ── Helper: build PCIe info vector ──────────────────────────────────────────

static std::vector<transfer::TransferEngine::Options::PcieInfo>
pcie_info(const config::Config& cfg) {
    std::vector<transfer::TransferEngine::Options::PcieInfo> info;
    info.reserve(cfg.hardware.gpus.size());
    for (const auto& g : cfg.hardware.gpus) {
        info.push_back({
            .pcie_gen   = g.pcie_gen > 0 ? g.pcie_gen : 5,
            .pcie_width = g.pcie_width > 0 ? g.pcie_width : 16,
        });
    }
    return info;
}

// ── Engine constructor ──────────────────────────────────────────────────────

Engine::Engine(const std::string& config_path, EngineBackends backends)
    : config_path_(config_path)
    , backends_(std::move(backends))
    , ipc_region_(nullptr, std::free)
{
    init_modules();
    allocate_ipc_region();
    spawn_daemon_thread();
    spdlog::info("Engine initialized: {} GPUs, {} layers, {} MoE layers, {} experts",
                 engine_info_.num_gpus, engine_info_.num_layers,
                 engine_info_.num_moe_layers, engine_info_.num_experts);
}

Engine::~Engine() {
    shutdown();
}

// ── Offline pre-processing (no GPU/IPC init) ───────────────────────────────

model::PrepackResult Engine::run_prepack(
    const std::string& config_path,
    const std::filesystem::path& output_dir)
{
    // Lightweight init: config + model metadata + weight loading only.
    // No resolve_config (CUDA), no VRAM, no IPC, no daemon thread.
    auto cfg = config::load_config(config_path);
    // TD-VOCAB-AUTODETECT: same resolve seam as Engine::init_modules
    // (LayerRegistry below sizes the embedding/output head off vocab_size).
    model::resolve_vocab_size(cfg);
    model::ModelConfig model_cfg(cfg);

    // GGUF: experts are per-projection mixed/uniform and the per-projection
    // k-quant types are data-dependent (read from the file), so the generic
    // `gguf` registry sentinel cannot size them — and LayerRegistry sizes
    // experts at CONSTRUCTION, before any load. Mirror init_modules: pre-scan
    // the file headers and build the mixed GgufQuantInterface up front
    // (TD-GGUF-GENERIC-DEFAULT-MISSIZE resolution, both call sites).
    std::optional<model::GgufQuantInterface> owned_gguf_quant;
    const model::QuantInterface* quant = nullptr;
    if (cfg.model.weights_format == config::WeightsFormat::gguf &&
        cfg.quantization.weights == config::WeightQuant::gguf) {
        auto types = model::gguf_expert_types_from_path(cfg.model.weights_path,
                                                        cfg.model.use_mmap);
        owned_gguf_quant.emplace(
            model::make_gguf_quant(types.gate, types.up, types.down));
        quant = &*owned_gguf_quant;
        spdlog::info("GGUF generic weights: built mixed expert QuantInterface "
                     "(gate={} up={} down={})",
                     model::gguf::type_name(types.gate),
                     model::gguf::type_name(types.up),
                     model::gguf::type_name(types.down));
    } else {
        quant = &model::get_format(cfg.quantization.weights);
    }
    model::LayerRegistry layer_registry(model_cfg, cfg, *quant);

    spdlog::info("Loading weights for pre-processing...");
    auto loaded_model = model::load_weights(cfg, model_cfg, layer_registry);

    spdlog::info("Pre-processing experts to {}...", output_dir.string());
    model::PrepackResult result =
        model::prepack_experts(loaded_model, model_cfg, *quant, cfg,
                               output_dir);

    if (result.error.empty()) {
        spdlog::info("Pre-processing complete: {} written, {} skipped, {:.1f} GB",
                     result.experts_written, result.experts_skipped,
                     result.bytes_written / (1024.0 * 1024.0 * 1024.0));
    } else {
        spdlog::error("Pre-processing failed: {}", result.error);
    }

    return result;
}

// ── Module initialization ───────────────────────────────────────────────────

namespace {
// C-6 Milestone C (node-1 placement, LS_CPU_EXPERT_NODE). The effective NUMA
// node the single-node CPU expert device lives on (threads + scratch + fold
// staging). DEFAULT 1 — the only GPU-free DDR node, so the device's H2D-fold
// staging never contends for a GPU's expert-fetch bandwidth (the +22% B=1
// contention cause). Overridable: LS_CPU_EXPERT_NODE=3 keeps it on the
// GPU-local bank for A/B; any node index is honored. Only ever consulted when a
// CPU expert device is actually being built (offload active) ⇒ CHAMPION-SAFE:
// the pure champion (no cpu_expert_devices) never calls this. The `config_node`
// (from KEEPER52_CPU_EXPERT / hardware.cpu_expert_devices[].numa_node) is used
// only when the env is unset AND non-default is desired — but per the shipped
// architecture the unset default is node 1 regardless of config.
int effective_cpu_expert_node(int config_node) {
    if (const char* e = std::getenv("LS_CPU_EXPERT_NODE"); e && e[0])
        return std::atoi(e);
    (void)config_node;
    return 1;  // shipped default ON: node-1 (GPU-free DDR).
}

// Boot instrumentation (round 2b): seconds since PROCESS start (not engine
// ctor) — /proc/self/stat field 22 (starttime, clock ticks since boot) vs
// /proc/uptime. Attributes the pre-first-log gap (binary load, gtest, CUDA
// driver init) that varied 14-24 s across runs.
double process_age_seconds() {
    FILE* f = std::fopen("/proc/self/stat", "r");
    if (!f) return -1.0;
    char buf[1024];
    const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    buf[n] = 0;
    const char* p = std::strrchr(buf, ')');
    if (!p) return -1.0;
    long long starttime = 0;
    int field = 2;  // after comm: state is field 3
    for (const char* q = p + 2; *q; ++q) {
        if (*q == ' ') continue;
        ++field;
        if (field == 22) { starttime = std::atoll(q); break; }
        while (*q && *q != ' ') ++q;
        if (!*q) break;
    }
    double uptime = 0;
    f = std::fopen("/proc/uptime", "r");
    if (!f) return -1.0;
    if (std::fscanf(f, "%lf", &uptime) != 1) uptime = 0;
    std::fclose(f);
    const double hz = static_cast<double>(sysconf(_SC_CLK_TCK));
    return uptime - static_cast<double>(starttime) / (hz > 0 ? hz : 100.0);
}
}  // namespace

void Engine::init_modules() {
    spdlog::info("[boot] engine init begins at process age {:.1f} s "
                 "(pre-init = binary/gtest/config/CUDA-driver load)",
                 process_age_seconds());
    // Deep fetch-path micro-trace (opt-in via LS_PERF_TRACE; one-time alloc).
    perf_trace::init_from_env();

    // Step 1: Parse config
    cfg_ = std::make_unique<config::Config>(config::load_config(config_path_));

    // Step 1b (TD-VOCAB-AUTODETECT): resolve model.vocab_size against the
    // weights' embedding/output-head row count BEFORE ModelConfig /
    // LayerRegistry / VramAllocator size anything off it. 0/absent = adopt
    // the weights-derived width; explicit-but-different = fail loud (weights
    // are ground truth). Null-backend test engines skip weight access and
    // therefore must carry an explicit vocab_size.
    if (!backends_.skip_weight_loading) {
        model::resolve_vocab_size(*cfg_);
    } else if (cfg_->model.vocab_size <= 0) {
        throw std::runtime_error(
            "model.vocab_size=0 (autodetect) requires weight loading; this "
            "engine was started with skip_weight_loading — set an explicit "
            "vocab_size in the config");
    }

    // Step 2: Hardware detection (fills VRAM/PCIe/NUMA from CUDA/sysfs)
    if (!backends_.skip_hardware_detection) {
        config::resolve_config(*cfg_);
    }

    // Step 3: Model config
    model_cfg_ = std::make_unique<model::ModelConfig>(*cfg_);

    // Step 4: Quant interface from registry. For the generic `gguf` weight type
    // the registry sentinel cannot size experts (GGUF is per-tensor mixed); build
    // an owned GgufQuantInterface from the file's per-projection k-quant types
    // (cheap header pre-scan) and point quant_ at it (TD-GGUF-GENERIC-DEFAULT-
    // MISSIZE resolution). Uniform `gguf_qX_k` resolves through the registry.
    if (cfg_->model.weights_format == config::WeightsFormat::gguf &&
        cfg_->quantization.weights == config::WeightQuant::gguf) {
        auto types = model::gguf_expert_types_from_path(cfg_->model.weights_path,
                                                        cfg_->model.use_mmap);
        owned_gguf_quant_.emplace(
            model::make_gguf_quant(types.gate, types.up, types.down));
        quant_ = &*owned_gguf_quant_;
        spdlog::info("GGUF generic weights: built mixed expert QuantInterface "
                     "(gate={} up={} down={})",
                     model::gguf::type_name(types.gate),
                     model::gguf::type_name(types.up),
                     model::gguf::type_name(types.down));
    } else {
        quant_ = &model::get_format(cfg_->quantization.weights);
    }

    // Step 5: Layer registry
    layer_registry_ = std::make_unique<model::LayerRegistry>(
        *model_cfg_, *cfg_, *quant_);

    // Step 5b: Create per-GPU DeviceBackend instances (#86a)
    device_backends_.clear();
    for (const auto& g : cfg_->hardware.gpus) {
        if (backends_.skip_hardware_detection) {
            device_backends_.push_back(
                compute::make_null_device_backend(g.ref));
        } else {
            device_backends_.push_back(
                compute::make_cuda_sm120_device_backend(g.ref));
        }
    }

    // Build raw-pointer vector for subsystems (#86d)
    std::vector<compute::DeviceBackend*> dev_ptrs;
    dev_ptrs.reserve(device_backends_.size());
    for (auto& db : device_backends_) dev_ptrs.push_back(db.get());

    // Step 6: VRAM allocation
    auto vram_layout = memory::compute_vram_layout(*cfg_, *layer_registry_, *model_cfg_);
    vram_allocator_ = std::make_unique<memory::VramAllocator>(
        std::move(vram_layout), dev_ptrs);

    // Step 7: Page allocator (D2D copy via DeviceBackend; UVA auto-routes)
    page_allocator_ = std::make_unique<memory::PageAllocator>(
        *vram_allocator_, dev_ptrs[0]);

    // Step 8: Expert cache
    const model::ExpertShape expert_shape{
        .hidden_size       = cfg_->model.hidden_size,
        .intermediate_size = cfg_->model.moe_intermediate_size,
    };
    const int64_t expert_bytes = quant_->bytes_per_expert(expert_shape);
    expert_cache_ = std::make_unique<memory::ExpertCache>(
        *vram_allocator_, *cfg_, expert_bytes, *quant_, expert_shape);

    // Step 10: NUMA manager
    numa_manager_ = std::make_unique<memory::NumaManager>(cfg_->hardware);

    // Step 10b: GPU loader transfer-cost calibration runs AFTER the ExpertDevices
    // exist (Step 14b) — the compute-curve pass needs them. See below.

    // Step 11: NVMe tier (optional — requires drives and LAYERSTORM_HAS_URING)
    if (!cfg_->hardware.nvme_paths.empty() && cfg_->memory.nvme_tier.enabled) {
        memory::NvmeTier::Options nvme_opts{
            .drive_paths           = cfg_->hardware.nvme_paths,
            .slot_size_bytes       = expert_bytes,
            .host_ram_budget_bytes =
                static_cast<int64_t>(
                    cfg_->memory.nvme_tier.host_ram_cache_gb.value_or(0)
                    * 1024.0 * 1024.0 * 1024.0),
            .num_moe_layers        = model_cfg_->num_moe_layers(),
            .num_experts_per_layer = cfg_->model.n_routed_experts,
            .first_moe_layer       = cfg_->model.first_k_dense_replace,
        };
        nvme_tier_ = std::make_unique<memory::NvmeTier>(
            std::move(nvme_opts), *numa_manager_);
    }

    // Step 12: Transfer engine (uses DeviceBackend directly, #86c)
    {
        transfer::TransferEngine::Options te_opts{
            .device_backends = dev_ptrs,
            .pcie_info       = pcie_info(*cfg_),
        };
        transfer_engine_ = std::make_unique<transfer::TransferEngine>(
            std::move(te_opts));
    }

    // Step 13: Stream manager (uses DeviceBackend directly, #86c)
    {
        compute::StreamManager::Options sm_opts{
            .device_backends = dev_ptrs,
        };
        stream_manager_ = std::make_unique<compute::StreamManager>(
            std::move(sm_opts));
    }

    // Step 14: Graph registry + buffer registry
    graph_registry_ = std::make_unique<compute::GraphRegistry>();
    buffer_registry_ = std::make_unique<BufferRegistry>();

    // Register VramAllocator regions with buffer registry.
    for (int g = 0; g < vram_allocator_->gpu_count(); ++g) {
        const auto& region = vram_allocator_->region(g);
        const auto& layout = vram_allocator_->layout().gpus[static_cast<size_t>(g)];
        auto gpu_suffix = ".gpu" + std::to_string(g);

        buffer_registry_->register_buffer(
            region.pinned, layout.pinned_bytes, g,
            ("vram.pinned" + gpu_suffix).c_str());
        buffer_registry_->register_buffer(
            region.kv_speculation, layout.kv_speculation_bytes, g,
            ("vram.kv_speculation" + gpu_suffix).c_str());
        if (region.indexer_k && layout.indexer_k_bytes > 0) {
            buffer_registry_->register_buffer(
                region.indexer_k, layout.indexer_k_bytes, g,
                ("vram.indexer_k" + gpu_suffix).c_str());
        }
        // V4-7a (ticket H): V4 tier regions (collapsed / zero for non-V4).
        if (region.kv_hca && layout.kv_hca_bytes > 0) {
            buffer_registry_->register_buffer(
                region.kv_hca, layout.kv_hca_bytes, g,
                ("vram.kv_hca" + gpu_suffix).c_str());
        }
        if (region.kv_swa && layout.kv_swa_bytes > 0) {
            buffer_registry_->register_buffer(
                region.kv_swa, layout.kv_swa_bytes, g,
                ("vram.kv_swa" + gpu_suffix).c_str());
        }
        buffer_registry_->register_buffer(
            region.kv_main, layout.kv_main_bytes, g,
            ("vram.kv_main" + gpu_suffix).c_str());
        buffer_registry_->register_buffer(
            region.expert_streaming, layout.expert_streaming_bytes, g,
            ("vram.expert_streaming" + gpu_suffix).c_str());
        buffer_registry_->register_buffer(
            region.expert_stable, layout.expert_stable_bytes, g,
            ("vram.expert_stable" + gpu_suffix).c_str());
    }
    spdlog::info("Buffer registry: {} entries from VramAllocator",
                 buffer_registry_->size());

    // Step 14b: Create per-GPU ExpertDevice instances (INV-BH-5)
    expert_devices_.clear();
    for (const auto& g : cfg_->hardware.gpus) {
        if (backends_.skip_hardware_detection) {
            expert_devices_.push_back(
                compute::make_null_expert_device(g.ref));
        } else {
            expert_devices_.push_back(
                compute::make_cuda_sm120_expert_device(g.ref));
        }
    }

    // Step 14c: GPU loader cost-constant calibration (I8b). Gated on config (default
    // off) and skipped in null-backend mode. kLoaded loads the JSON (validating its
    // compute dims against this model's expert FFN — the JSON is weight-specific) or
    // self-heals by running a full calibration + writing it. full mode additionally
    // measures the per-device compute curve (needs the ExpertDevices, hence here) and
    // the reconciliation overhead/added time.
    if (cfg_->gpu_loader.enabled && !backends_.skip_hardware_detection) {
        gpu_loader::CalibrationMode mode = gpu_loader::CalibrationMode::kLoaded;
        switch (cfg_->gpu_loader.calibration_mode) {
            case config::CalibrationMode::quick: mode = gpu_loader::CalibrationMode::kQuick; break;
            case config::CalibrationMode::full:  mode = gpu_loader::CalibrationMode::kFull;  break;
            case config::CalibrationMode::loaded:
            default:                             mode = gpu_loader::CalibrationMode::kLoaded; break;
        }
        const int model_N = 2 * cfg_->model.moe_intermediate_size;  // gate_up output dim
        const int model_K = cfg_->model.hidden_size;                // input dim

        // The calibration is weight/config-specific (the compute curve is fit at the
        // model's expert FFN dims), so the JSON lives ADJACENT to the weights: empty →
        // '<weights dir>/gpu_loader_calibration.json'; a bare filename → in the weights
        // dir; an absolute path → verbatim.
        std::filesystem::path weights_dir =
            std::filesystem::path(cfg_->model.weights_path);
        if (!weights_dir.empty() && !std::filesystem::is_directory(weights_dir))
            weights_dir = weights_dir.parent_path();
        std::filesystem::path cal_path(cfg_->gpu_loader.calibration_path);
        if (cal_path.empty())
            cal_path = weights_dir / "gpu_loader_calibration.json";
        else if (cal_path.is_relative())
            cal_path = weights_dir / cal_path;
        const std::string calibration_path = cal_path.string();
        std::vector<compute::ExpertDevice*> expert_ptrs;
        expert_ptrs.reserve(expert_devices_.size());
        for (auto& ed : expert_devices_) expert_ptrs.push_back(ed.get());

        const double recon_payload = static_cast<double>(model_K) * 2.0;  // hidden * bf16

        // kLoaded: prefer the on-disk file, but reject it if its compute curves were
        // measured for a different model (compute_dims_match) OR if a device UUID does
        // not match the live GPU at that position (wrong machine / reordered devices) —
        // then recalibrate. Empty UUIDs (old v2 file) skip the UUID check (INV-LOADER-CAL-6).
        bool loaded_ok = false;
        if (mode == gpu_loader::CalibrationMode::kLoaded &&
            std::filesystem::exists(calibration_path)) {
            try {
                gpu_loader::LoaderConstants disk = gpu_loader::load(calibration_path);
                bool uuid_ok = true;
                for (const auto& dev : disk.devices) {
                    if (dev.uuid.empty()) continue;  // old v2 file → skip identity check
                    if (dev.position < 0 ||
                        dev.position >= static_cast<int>(dev_ptrs.size()) ||
                        dev_ptrs[static_cast<size_t>(dev.position)] == nullptr)
                        continue;  // no live device at this slot → can't compare
                    std::string live_uuid;
                    try {
                        live_uuid = core::query_gpu_info(
                            dev_ptrs[static_cast<size_t>(dev.position)]->gpu().id).uuid;
                    } catch (const std::exception& e) {
                        spdlog::warn("gpu_loader: query_gpu_info(pos={}) failed ({}); "
                                     "skipping UUID check for it", dev.position, e.what());
                        continue;
                    }
                    if (!live_uuid.empty() && live_uuid != dev.uuid) {
                        spdlog::warn("gpu_loader: {} device pos={} UUID {} != live {} "
                                     "(wrong machine / reordered devices); recalibrating",
                                     calibration_path, dev.position, dev.uuid, live_uuid);
                        uuid_ok = false;
                        break;
                    }
                }
                if (uuid_ok && gpu_loader::compute_dims_match(disk, model_N, model_K)) {
                    loader_constants_ = std::move(disk);
                    loaded_ok = true;
                } else if (uuid_ok) {
                    spdlog::warn("gpu_loader: {} compute dims (N={},K={}) != model (N={},K={}); "
                                 "recalibrating",
                                 calibration_path, disk.compute_N, disk.compute_K,
                                 model_N, model_K);
                }
            } catch (const std::exception& e) {
                spdlog::warn("gpu_loader: failed to load {} ({}); recalibrating",
                             calibration_path, e.what());
            }
        }
        if (!loaded_ok) {
            const gpu_loader::CalibrationMode run_mode =
                (mode == gpu_loader::CalibrationMode::kLoaded) ? gpu_loader::CalibrationMode::kFull
                                                              : mode;
            gpu_loader::CalibrationConfig rcfg = gpu_loader::config_for_mode(run_mode);
            rcfg.compute_N = model_N;
            rcfg.compute_K = model_K;
            rcfg.recon_payload_bytes = recon_payload;
            loader_constants_ = gpu_loader::calibrate(dev_ptrs, *numa_manager_, rcfg, expert_ptrs);
            try {
                gpu_loader::save(loader_constants_, calibration_path);
            } catch (const std::exception& e) {
                spdlog::warn("gpu_loader: could not write {} ({})", calibration_path, e.what());
            }
        }
    }

    // Step 15-16: AttentionDevices + DcpExecutor (all TP configs, TD-40h)
    const int tp = std::max(1, cfg_->parallelism.tensor_parallelism);

    // KVS-2/-3: sequence-sharded KV mode (hardware.dcp_kv_mode). Under
    // sharding the executor runs the Q-head-allgather structure (INV-KVS-QAG):
    // attention on EVERY rank covers ALL tp*HL heads over its LOCAL token
    // shard, so the AttentionDevice h_q and the DCP combine head count must
    // be the FULL model head count (not the TP-local shard).
    const bool kv_sharded_mode = (tp >= 2)
        && cfg_->hardware.dcp_kv_mode == config::DcpKvMode::sharded;

    // TP GPUs are the first tp entries in the GPU list
    std::vector<config::GpuRef> tp_gpus;
    std::vector<int> tp_gpu_indices;
    for (int i = 0; i < tp && i < static_cast<int>(cfg_->hardware.gpus.size()); ++i) {
        tp_gpus.push_back(cfg_->hardware.gpus[static_cast<size_t>(i)].ref);
        tp_gpu_indices.push_back(cfg_->hardware.gpus[static_cast<size_t>(i)].ref.position);
    }

    // CollectiveBackend for DCP (NCCL or null)
    if (!backends_.skip_hardware_detection) {
        collective_backend_ = parallelism::make_nccl_collective_backend();
    } else {
        collective_backend_ = parallelism::make_null_collective_backend();
    }

    // TD-PREFILL-MOE-BIG: resolve the ELASTIC superchunk capacity ONCE, before
    // every consumer sized from it (DcpCommunicator batch guard, hidden-state
    // pair buffers, DcpExecutor sparse-index rows, CommandDispatcher scratch).
    // K is DERIVED from free VRAM: with prefill_moe_big the MoE transients are
    // chunk-bounded, so only the per-token PERSISTENT cost (hidden pairs,
    // moe_output/normalized/shared rows, routing, DSA sparse-index rows)
    // limits the capacity. The request (compute.prefill_superchunk_tokens;
    // 0 = derive up to min(max_sequence_length, 32768)) is clamped to what
    // fits with the safety margin + 1.5 GiB headroom preserved — never OOM by
    // construction. The resolved value is written back into the live config so
    // all downstream consumers agree. The CommandDispatcher's construction-time
    // fail-safe re-verifies against the then-current free VRAM and can only
    // step DOWN further. No-op when prefill_moe_big is off (legacy sizing).
    if (cfg_->compute.prefill_moe_big && cfg_->model.n_routed_experts > 0
        && cfg_->model.hidden_size > 0 && cfg_->model.num_experts_per_tok > 0) {
        constexpr int kHardCap = 32768;  // schema bound for the request
        const int floor_rows = static_cast<int>(ipc::kMaxBatchDescriptors);
        const bool explicit_request = cfg_->compute.prefill_superchunk_tokens > 0;
        int request = explicit_request
            ? std::min(cfg_->compute.prefill_superchunk_tokens, kHardCap)
            : std::min(std::max(cfg_->serving.max_sequence_length, floor_rows),
                       kHardCap);
        size_t min_free = 0;
        bool have_query = false;
        for (auto& d : expert_devices_) {
            size_t f = 0, t = 0;
            if (d && d->device_mem_info(f, t)) {
                min_free = have_query ? std::min(min_free, f) : f;
                have_query = true;
            }
        }
        // Validation knob (TD-PREFILL-MOE-BIG NO-OOM bar): cap the free-VRAM
        // the elastic derivation sees, forcing the capacity to shrink as if
        // the GPU were nearly full — proves the never-OOM path end-to-end
        // without needing a physically full card.
        if (const char* e = std::getenv("LAYERSTORM_MOE_BIG_VRAM_CAP_MB");
            e && *e && have_query) {
            const long cap_mb = std::strtol(e, nullptr, 10);
            if (cap_mb > 0) {
                min_free = std::min(min_free,
                                    static_cast<size_t>(cap_mb) << 20);
                spdlog::warn("TD-PREFILL-MOE-BIG: LAYERSTORM_MOE_BIG_VRAM_CAP_MB={} "
                             "— elastic derivation capped at {} MiB free "
                             "(validation knob)", cap_mb, min_free >> 20);
            }
        }
        int rows;
        if (have_query) {
            const auto& m = *cfg_;
            const size_t H = static_cast<size_t>(m.model.hidden_size);
            const size_t E = static_cast<size_t>(m.model.n_routed_experts);
            const size_t T = static_cast<size_t>(m.model.num_experts_per_tok);
            const size_t ITK = m.model.index_topk > 0
                ? static_cast<size_t>(m.model.index_topk) : 0;
            // Per-token persistent VRAM: hidden pair attn+moe bufs (2*H*2) +
            // dispatcher moe_output/normalized/shared_out (3*H*2) + routing
            // (E*4 logits + T*8 topk) + DSA sparse-index rows (ITK*4 + 4).
            const size_t per_tok = 2 * H * 2 + 3 * H * 2 + E * 4 + T * 8
                                 + (ITK ? ITK * 4 + 4 : 0);
            const size_t margin = static_cast<size_t>(
                m.memory.vram_safety_margin_gb * (1ull << 30));
            // Reserve max(margin, 1.5 GiB) — NOT their sum. min_free here
            // ALREADY contains the region-carve margin the VramAllocator
            // left unallocated, and the margin's configured purpose is
            // exactly the future runtime consumers of that pool (host
            // pinned-arena page tables ~3 MiB/GiB, driver/context, late
            // attention scratch). Summing double-reserved ~4.5 GiB on the
            // keeper52 shape (margin 3.0) and clamped the derived capacity
            // to the 512 floor on the very box the elastic K exists for —
            // the legacy halving fail-safe reserved only 1.5 GiB and
            // reached 2048 tokens on less free VRAM.
            const size_t headroom =
                std::max(margin, static_cast<size_t>(1536ull << 20));
            const size_t budget = min_free > headroom ? min_free - headroom : 0;
            rows = static_cast<int>(std::min<size_t>(
                static_cast<size_t>(request),
                std::max<size_t>(static_cast<size_t>(floor_rows),
                                 budget / std::max<size_t>(per_tok, 1))));
        } else {
            // No mem query (null backends): honor an explicit request only —
            // a derived default would inflate every null-backend test's
            // buffers with nothing enforcing the fit.
            rows = explicit_request ? request : floor_rows;
        }
        rows = std::max(rows, floor_rows);
        if (rows != cfg_->compute.prefill_superchunk_tokens) {
            spdlog::info("TD-PREFILL-MOE-BIG: elastic superchunk capacity {} "
                         "tokens (request {}, min free VRAM {} MiB)",
                         rows, cfg_->compute.prefill_superchunk_tokens,
                         have_query ? (min_free >> 20) : 0);
            cfg_->compute.prefill_superchunk_tokens = rows;
        }
    }

    // DCP communicator (TP >= 2 only — communication not needed for single GPU)
    if (tp >= 2) {
        std::vector<compute::DeviceBackend*> tp_dev_ptrs;
        tp_dev_ptrs.reserve(tp);
        for (const auto& g : tp_gpus)
            tp_dev_ptrs.push_back(device_backends_[g.position].get());

        parallelism::DcpCommunicator::Options dcp_opts{
            .dcp_size        = tp,
            .device_backends = std::move(tp_dev_ptrs),
            // TD-PREFILL-SUPERCHUNK: the MoE cross-rank combine + embedding
            // allreduce run over the whole superchunk token batch — raise the
            // communicator's batch guard/LSE sizing to cover it (LSE cost is
            // KB..MB-scale).
            .max_batch_size  = std::max(
                cfg_->orchestrator.max_batch_size,
                cfg_->compute.prefill_superchunk_tokens),
            .num_heads       = cfg_->model.num_attention_heads,
            .attn_output_dim = cfg_->model.kv_lora_rank,  // absorbed MLA: kv_lora_rank, not v_head_dim
            .hidden_size     = cfg_->model.hidden_size,
            .collective      = collective_backend_.get(),
        };
        dcp_communicator_ = std::make_unique<parallelism::DcpCommunicator>(
            std::move(dcp_opts));
    }

    // P-24b lever-1: launch the early arena-attach worker BEFORE the expensive
    // rank init — holder attach, warm validation/adopt and (warm-only)
    // registration then overlap attention-device init + the GGUF load. Joined
    // at the step-18d arena block.
    launch_arena_attach_early_();

    // Create per-GPU AttentionDevice instances (INV-BH-1, INV-BH-7)
    attention_devices_.clear();
    std::vector<compute::AttentionDevice*> attn_ptrs;
    for (int i = 0; i < tp; ++i) {
        if (backends_.skip_hardware_detection) {
            attention_devices_.push_back(
                compute::make_null_attention_device(tp_gpus[static_cast<size_t>(i)]));
        } else {
            attention_devices_.push_back(
                compute::make_attention_device(
                    cfg_->compute.attention_backend,
                    tp_gpus[static_cast<size_t>(i)]));
        }
        attn_ptrs.push_back(attention_devices_.back().get());
    }

    // Attention softmax scale (TD-ROPE): (qk_nope+qk_rope)^-0.5 — the absorbed-space
    // dot product equals the non-absorbed per-head dot product — × yarn mscale²
    // when the context extends beyond the original length (compute/rope_table.h).
    const float attn_sm_scale = compute::rope_softmax_scale(
        cfg_->model.qk_nope_head_dim, cfg_->model.qk_rope_head_dim,
        cfg_->serving.max_sequence_length, cfg_->model.rope_scaling);

    // DET-REDUCE: gated deterministic attention reduction (both backends —
    // TD-TQ-PREFILL-DETREDUCE-WIRING). Config flag OR env
    // LAYERSTORM_DETERMINISTIC_REDUCE overrides the config EITHER way
    // (read once here at init): set & non-"0" → on, "0" → off (the opt-out;
    // config default is now true). Unset → config value. Set false (config or
    // env=0) → byte-identical to the legacy atomic path.
    bool det_reduce = cfg_->compute.deterministic_reduce;
    if (const char* e = std::getenv("LAYERSTORM_DETERMINISTIC_REDUCE"); e && *e)
        det_reduce = (e[0] != '0');

    // Step 14c: TQ resource init (if TurboQuant selected)
    if (cfg_->compute.attention_backend == config::AttentionBackendType::turboquant_mla
        && !backends_.skip_hardware_detection) {
        // INV-KVS-QAG: sharded KV attends ALL heads per rank (post-Q-allgather).
        const int h_q = kv_sharded_mode
            ? cfg_->model.num_attention_heads
            : cfg_->model.num_attention_heads / std::max(tp, 1);
        const int max_kv = cfg_->serving.max_sequence_length;
        tq_resources_.clear();
        tq_resources_.reserve(static_cast<size_t>(tp));
        // Startup lever: the Π/Π^T HOST contents are rank-identical (INV-TQ-4
        // seeds carry no rank) — precompute once, layer-parallel (~1 s), so
        // each rank's init only allocs+uploads (was ~23 s/rank of serial
        // Householder QR, the dominant rank-init cost).
        const auto tq_shared_rotations = compute::precompute_tq_rotations(
            cfg_->model.kv_lora_rank, cfg_->model.num_hidden_layers);
        for (int i = 0; i < tp; ++i) {
            auto* dev = attention_devices_[i].get();
            // INV-TQ-PERRANK: allocate the codebook + Pi on THIS rank's GPU
            // — every rank's kernels dereference these pointers, and a
            // rank-0-only allocation is an illegal access on rank 1+.
            // Contents are rank-identical (INV-TQ-4 deterministic seeds).
            compute::TqInitOptions tq_opts;
            tq_opts.d_c = cfg_->model.kv_lora_rank;
            tq_opts.bits = 4;
            tq_opts.num_layers = cfg_->model.num_hidden_layers;
            tq_opts.codebook_dir = LAYERSTORM_TQ_CODEBOOK_DIR;
            tq_opts.attention_device = dev;
            tq_resources_.push_back(compute::init_tq_resources(
                tq_opts, tq_shared_rotations.get()));
            compute::tq_device_set_resources(dev, tq_resources_.back().get());
            compute::tq_device_set_model_dims(dev,
                cfg_->orchestrator.max_batch_size,
                cfg_->model.kv_lora_rank,
                cfg_->model.qk_rope_head_dim, h_q, /*s_q=*/1, attn_sm_scale);
            compute::tq_device_set_prefill_scratch(dev, max_kv);
            compute::tq_device_set_deterministic_reduce(dev, det_reduce);
        }
        spdlog::info("TQ resources initialized: d_c={}, {} layers, "
                     "{} rank(s) (per-rank codebook+Pi, INV-TQ-PERRANK), "
                     "deterministic_reduce={}",
                     cfg_->model.kv_lora_rank, cfg_->model.num_hidden_layers,
                     tp, det_reduce);
    }

    // Step 14c-snap: SnapMLA prefill scratch init (KD-4f-d.1a)
    if (cfg_->compute.attention_backend == config::AttentionBackendType::snapmla
        && !backends_.skip_hardware_detection) {
        // INV-KVS-QAG: sharded KV attends ALL heads per rank (post-Q-allgather).
        const int h_q = kv_sharded_mode
            ? cfg_->model.num_attention_heads
            : cfg_->model.num_attention_heads / std::max(tp, 1);
        const int max_kv = cfg_->serving.max_sequence_length;
        // DET-REDUCE gate: det_reduce computed once above (shared with the
        // TQ backend block — TD-TQ-PREFILL-DETREDUCE-WIRING).
        for (int i = 0; i < tp; ++i) {
            auto* dev = attention_devices_[i].get();
            compute::snapmla_device_set_model_dims(dev,
                cfg_->orchestrator.max_batch_size,
                cfg_->model.kv_lora_rank,
                cfg_->model.qk_rope_head_dim, h_q, attn_sm_scale);
            compute::snapmla_device_set_prefill_scratch(dev, max_kv);
            compute::snapmla_device_set_deterministic_reduce(dev, det_reduce);
        }
        spdlog::info("SnapMLA prefill scratch initialized: d_c={}, h_q={}, max_kv={}, "
                     "deterministic_reduce={}",
                     cfg_->model.kv_lora_rank, h_q, max_kv, det_reduce);
    }

    // Step 14c-v4: CsaHcaSm120AttentionDevice configuration (V4-5a; V4-5T:
    // the TQ arms configure the SAME arch device with per-tier codecs).
    const bool v4_backend =
        cfg_->compute.attention_backend == config::AttentionBackendType::csa_hca
        || cfg_->compute.attention_backend
               == config::AttentionBackendType::csa_hca_tq
        || cfg_->compute.attention_backend
               == config::AttentionBackendType::csa_hca_tq_mix;
    if (v4_backend && !backends_.skip_hardware_detection) {
        // V4-2c (TD-V4-TP RESOLVED 2026-08-21): tp > 1 runs with q_b
        // head-sharded, o_proj_a group-sharded + o_proj_b row-parallel
        // (executor allreduce), everything else replicated. Per-rank head
        // counts below 64 run the decode kernel PADDED (zero-q pad heads).
        if (kv_sharded_mode) {
            throw std::runtime_error(
                "deepseek_v4: sharded KV mode is fail-closed (TD-V4-DCP-KV)");
        }
        const auto& v4 = vram_allocator_->layout().v4;
        if (!v4.enabled) {
            throw std::runtime_error(
                "deepseek_v4: VramLayout.v4 not enabled — csa_hca requires "
                "the V4 KV layout (V4-3b)");
        }
        const int max_kv = cfg_->serving.max_sequence_length;
        compute::V4DeviceOptions vo;
        vo.max_batch = cfg_->orchestrator.max_batch_size;
        vo.max_attn_rows =
            std::max(512, cfg_->orchestrator.max_batch_size);
        // V4-2c: the deps csa_fp8 decode kernel loads full 64-head Q tiles
        // (h_q must be 64 or 128) — pad sub-tile per-rank head counts to
        // the tile bound. The executor zero-pads the q buffers and runs
        // epilogues over the real heads only; pad-head outputs are never
        // consumed. Validator caps heads/tp at 128.
        {
            const int h_real =
                cfg_->model.num_attention_heads / std::max(tp, 1);
            vo.h_q = h_real <= 64 ? 64 : 128;
            if (h_real > 128) {
                throw std::runtime_error(
                    "deepseek_v4: per-rank head count > 128 exceeds the "
                    "csa_fp8 decode-kernel tile bound");
            }
        }
        vo.head_dim = cfg_->model.head_dim;
        vo.rope_dim = cfg_->model.qk_rope_head_dim;
        // Ticket-D finding: V4 softmax scale is 1/sqrt(head_dim) with the
        // yarn mscale CANCELLED (pure cos/sin tables) — NOT rope_softmax_scale.
        vo.sm_scale = 1.0f / std::sqrt(static_cast<float>(cfg_->model.head_dim));
        vo.rms_eps = static_cast<float>(cfg_->model.rms_norm_eps);
        vo.topk = cfg_->model.index_topk;
        vo.sliding_window = cfg_->model.sliding_window;
        vo.num_sm_parts = 32;  // deps V4K benchmark optimum at B=1
        // Ticket-J S1 bisect knob: LS_V4_NSP overrides the split-KV part
        // count (1 disables the split/combine path entirely).
        if (const char* nsp_env = std::getenv("LS_V4_NSP")) {
            const int nsp = std::atoi(nsp_env);
            if (nsp >= 1) vo.num_sm_parts = nsp;
        }
        // DET-REDUCE (ticket J): the csa_fp8 decode kernel's softmax
        // denominator was the last arrival-order atomicAdd in the V4 arm —
        // wire the same gate the TQ/SnapMLA backends honor (default ON).
        vo.deterministic_reduce = det_reduce;
        vo.csa_entries_per_page = v4.csa_entries_per_page;
        vo.hca_entries_per_page = v4.hca_entries_per_page;
        vo.swa_page_tokens = v4.swa_page_tokens;
        vo.idx_entries_per_page = v4.indexer_entry_bytes > 0
            ? static_cast<int>(v4.indexer_bytes_per_page /
                               v4.indexer_entry_bytes)
            : 0;
        vo.idx_page_bytes = v4.indexer_bytes_per_page;
        vo.index_n_heads = cfg_->model.index_n_heads;
        vo.index_head_dim = cfg_->model.index_head_dim;
        vo.max_index_blocks = (max_kv + memory::kV4CsaRatio - 1) / memory::kV4CsaRatio;
        // V4-5T (TD-V4-TQ-DEVICE): per-tier codecs mirror the allocator's
        // layout formats (the codec-axis authority; SWA always FP8).
        vo.csa_codec = v4.csa_format == memory::KvCacheFormat::kV4Tq
                           ? compute::V4TierCodec::kTq4
                           : compute::V4TierCodec::kFp8;
        vo.hca_codec = v4.hca_format == memory::KvCacheFormat::kV4Tq
                           ? compute::V4TierCodec::kTq4
                           : compute::V4TierCodec::kFp8;
        // Prefill staging bound: visible compressed union (CSA + HCA) + the
        // chunk's raw entries + window carry.
        const int max_staged = max_kv / 4 + max_kv / 128 + vo.max_attn_rows +
                               vo.sliding_window + 64;
        // V4-5T: TQ resources for TQ-coded tiers — the SAME machinery as
        // the turboquant_mla backend (d == head_dim == 512 matches the
        // codebook_d512 artifact; per-layer Π seeded INV-TQ-4). Per-rank
        // codebook+Π allocation (INV-TQ-PERRANK).
        const bool v4_any_tq = vo.csa_codec == compute::V4TierCodec::kTq4
                            || vo.hca_codec == compute::V4TierCodec::kTq4;
        std::shared_ptr<const compute::TqRotationHostData> v4_tq_rot;
        if (v4_any_tq) {
            v4_tq_rot = compute::precompute_tq_rotations(
                cfg_->model.head_dim, cfg_->model.num_hidden_layers);
            tq_resources_.clear();
            tq_resources_.reserve(static_cast<size_t>(tp));
        }
        for (int i = 0; i < tp; ++i) {
            auto* dev = attention_devices_[static_cast<size_t>(i)].get();
            dev->set_device();
            compute::csa_hca_device_configure(dev, vo);
            compute::csa_hca_device_set_scratch(dev, max_staged);
            if (v4_any_tq) {
                compute::TqInitOptions tq_opts;
                tq_opts.d_c = cfg_->model.head_dim;
                tq_opts.bits = 4;
                tq_opts.num_layers = cfg_->model.num_hidden_layers;
                tq_opts.codebook_dir = LAYERSTORM_TQ_CODEBOOK_DIR;
                tq_opts.attention_device = dev;
                tq_resources_.push_back(compute::init_tq_resources(
                    tq_opts, v4_tq_rot.get()));
                compute::csa_hca_device_set_tq(dev,
                                               tq_resources_.back().get());
            }
        }
        spdlog::info(
            "CsaHca V4 device configured: h_q={}, sm_scale=1/sqrt({}), "
            "topk={}, window={}, staged_cap={}, idx_entries/page={}, "
            "codec csa={} hca={}",
            vo.h_q, cfg_->model.head_dim, vo.topk, vo.sliding_window,
            max_staged, vo.idx_entries_per_page,
            vo.csa_codec == compute::V4TierCodec::kTq4 ? "tq4" : "fp8",
            vo.hca_codec == compute::V4TierCodec::kTq4 ? "tq4" : "fp8");
    }

    // TODO:DEBT TD-51d: is_draft D2H readback never wired (no host-side copy enacted)

    // KD-R2: Allocate paired hidden state buffers for TP GPUs.
    // TD-PREFILL-SUPERCHUNK: the pair buffers ARE the superchunk staging —
    // attention sub-launches write disjoint row ranges (row_offset) and one
    // MoE command consumes all rows — so size them for the full superchunk.
    // V4-5b mHC: the residual stream is [rows, hc_mult, hidden] — the pair
    // buffers carry all hc_mult streams (flattened per token).
    const int hc_streams = model_cfg_->has_mhc() ? cfg_->model.hc_mult : 1;
    {
        const size_t buf_size = static_cast<size_t>(std::max(
                                    cfg_->orchestrator.max_batch_size,
                                    cfg_->compute.prefill_superchunk_tokens))
                              * cfg_->model.hidden_size * hc_streams * 2;  // BF16
        hidden_state_pairs_.resize(tp);
        for (int r = 0; r < tp; ++r) {
            auto& pair = hidden_state_pairs_[r];
            pair.attn_buf     = attention_devices_[static_cast<size_t>(r)]->device_alloc(buf_size);
            pair.moe_buf      = expert_devices_[static_cast<size_t>(tp_gpus[r].position)]->device_alloc(buf_size);
            pair.rank         = r;
            pair.gpu_position = tp_gpus[r].position;
            pair.gpu_id       = tp_gpus[r].id;
            // Sync events created by CommandDispatcher constructor.

            // KD-4f: Register attn_buf in BufferRegistry so CMD_EMBEDDING_LOOKUP
            // and CMD_OUTPUT_HEAD can resolve a buf_id to this device memory.
            buffer_registry_->register_buffer(
                pair.attn_buf, static_cast<int64_t>(buf_size),
                pair.gpu_position,
                ("hidden_state.attn.rank" + std::to_string(r)).c_str());
            // TD-GOLDEN: register moe_buf (post-attention residual stream)
            // so layer-level debug readback can resolve it by name.
            buffer_registry_->register_buffer(
                pair.moe_buf, static_cast<int64_t>(buf_size),
                pair.gpu_position,
                ("hidden_state.moe.rank" + std::to_string(r)).c_str());
        }
    }

    // V4-5b mHC: attention-stage hc_pre scratch per TP rank — the collapsed
    // module input x [rows, hidden] BF16 plus the per-token mix coefficients
    // (post [rows, hc] F32, comb [rows, hc*hc] F32) that the attention-side
    // hc_post consumes after o_proj. Rows bound = the pair-buffer row bound.
    if (hc_streams > 1) {
        const size_t rows = static_cast<size_t>(std::max(
            cfg_->orchestrator.max_batch_size,
            cfg_->compute.prefill_superchunk_tokens));
        const size_t x_bytes = rows * cfg_->model.hidden_size * 2;
        const size_t post_bytes = rows * hc_streams * sizeof(float);
        const size_t comb_bytes = rows * hc_streams * hc_streams * sizeof(float);
        hc_attn_x_.resize(tp, nullptr);
        hc_attn_post_.resize(tp, nullptr);
        hc_attn_comb_.resize(tp, nullptr);
        for (int r = 0; r < tp; ++r) {
            auto* dev = attention_devices_[static_cast<size_t>(r)].get();
            hc_attn_x_[r] = dev->device_alloc(x_bytes);
            hc_attn_post_[r] = dev->device_alloc(post_bytes);
            hc_attn_comb_[r] = dev->device_alloc(comb_bytes);
            if (!hc_attn_x_[r] || !hc_attn_post_[r] || !hc_attn_comb_[r]) {
                throw std::runtime_error("engine: mHC attention scratch alloc failed");
            }
            const int pos = tp_gpus[r].position;
            buffer_registry_->register_buffer(
                hc_attn_x_[r], static_cast<int64_t>(x_bytes), pos,
                ("hc_attn_x.rank" + std::to_string(r)).c_str());
            buffer_registry_->register_buffer(
                hc_attn_post_[r], static_cast<int64_t>(post_bytes), pos,
                ("hc_attn_post.rank" + std::to_string(r)).c_str());
            buffer_registry_->register_buffer(
                hc_attn_comb_[r], static_cast<int64_t>(comb_bytes), pos,
                ("hc_attn_comb.rank" + std::to_string(r)).c_str());
        }
    }

    // Allocate MoE hidden state buffers for non-TP GPUs (standalone MoE dispatch).
    // V4-5b mHC note: EP-beyond-TP GPUs receive the COLLAPSED (post-hc_pre)
    // 4096-wide hidden, never the hc-stream residual, so no ×hc_mult here.
    {
        const size_t buf_size = static_cast<size_t>(std::max(
                                    cfg_->orchestrator.max_batch_size,
                                    cfg_->compute.prefill_superchunk_tokens))
                              * cfg_->model.hidden_size * 2;  // BF16
        const auto num_gpus = expert_devices_.size();
        fused_moe_hidden_state_bufs_.resize(num_gpus, nullptr);
        for (size_t i = 0; i < num_gpus; ++i) {
            // Skip TP GPUs — their MoE bufs live in hidden_state_pairs_.
            bool is_tp = false;
            for (int r = 0; r < tp; ++r) {
                if (hidden_state_pairs_[r].gpu_position == static_cast<int>(i)) {
                    is_tp = true;
                    break;
                }
            }
            if (!is_tp) {
                fused_moe_hidden_state_bufs_[i] =
                    expert_devices_[i]->device_alloc(buf_size);
            }
        }
    }

    // TD-74i: DcpAttentionWrapper — DCP correction (steps 10-12) + TP
    // allreduce (step 14).  Active when tp >= 2; no-op for single-GPU.
    if (tp >= 2) {
        const int HL = cfg_->model.num_attention_heads / tp;
        dcp_attention_wrapper_ = std::make_unique<compute::DcpAttentionWrapper>(
            dcp_communicator_.get(),
            compute::DcpAttentionWrapperConfig{
                .num_heads_local = HL,
                .head_dim        = cfg_->model.kv_lora_rank,
                .hidden_size     = cfg_->model.hidden_size,
                .max_batch_size  = cfg_->orchestrator.max_batch_size,
                // INV-KVS-QAG: the LSE combine only runs under sharded KV,
                // where the post-Q-allgather partials cover ALL heads.
                .combine_num_heads = kv_sharded_mode
                    ? cfg_->model.num_attention_heads : 0,
                .gpus            = tp_gpus,
            },
            attn_ptrs);
    }

    // RoPE cos/sin table (TD-ROPE): pure table (no mscale), interleaved-pair
    // frequencies from rope_theta with optional YaRN correction; positions span
    // the serving context. Uploaded per rank by DcpExecutor::allocate_buffers
    // (synchronous in its constructor — the host vector's scope covers it).
    const int rope_max_pos = cfg_->serving.max_sequence_length;
    // V4-4c dual RoPE: V4 builds TWO pure tables — base theta WITHOUT yarn
    // (uncompressed layers run un-yarned, deepseek4.cpp:817-824) + compress
    // theta WITH yarn for compressed layers. Non-V4: single table, unchanged.
    std::vector<float> rope_table;
    std::vector<float> rope_table_compress;
    if (model_cfg_->is_v4()) {
        auto v4_tables = compute::build_v4_rope_tables(
            rope_max_pos, cfg_->model.qk_rope_head_dim,
            cfg_->model.rope_theta, cfg_->model.compress_rope_theta,
            cfg_->model.rope_scaling);
        rope_table = std::move(v4_tables.base);
        rope_table_compress = std::move(v4_tables.compress);
    } else {
        rope_table = compute::build_rope_cos_sin_table(
            rope_max_pos, cfg_->model.qk_rope_head_dim,
            cfg_->model.rope_theta, cfg_->model.rope_scaling);
    }

    // DCP executor (always created — dcp_size=1 is single-GPU passthrough)
    parallelism::DcpExecutor::Options de_opts{
        .dcp_size             = tp,
        .gpus                 = tp_gpus,
        // KVS-2/-3: sequence-sharded KV mode (hardware.dcp_kv_mode). True
        // enables the Q-head-allgather DCP combine path (INV-KVS-QAG;
        // INV-DCP-KVREP: combine ONLY under sharding) and CommandDispatcher's
        // sharded build_kv_metadata branch.
        .dcp_kv_sharded       = kv_sharded_mode,
        // KVS-4: INV-4.9e round-robin chunk size for the GLOBAL→LOCAL
        // sparse-index translation under sharded KV (must match the
        // PageAllocator DcpConfig below).
        .dcp_chunk_tokens     = cfg_->memory.kv_cache.dcp_chunk_size,
        // TD-GLM-INDEXER-LOCAL-MERGE: local (position-sharded) indexer mode —
        // each rank scores its own indexer-K shard, then the executor
        // allgathers per-rank candidates and exactly re-merges the global
        // top-k. Ownership unit = the indexer page (must match the
        // PageAllocator DcpConfig below and the dispatcher's provisioning).
        .indexer_local        =
            (cfg_->hardware.dcp_indexer_mode == config::DcpIndexerMode::local
             && tp >= 2),
        .indexer_k_page_tokens =
            cfg_->memory.kv_cache.indexer_k_page_size_tokens,
        // TD-SPARSE-CHUNK-PREFILL: DSA sparse CHUNK PREFILL attention — a
        // blessed prefill chunk runs the indexer per chunk-query row and
        // attends only its causal top-k (INV-SPARSE-CHUNK-CAUSAL). Default
        // OFF (dense chunks, byte-identical legacy behavior).
        .sparse_prefill       = cfg_->compute.dsa_sparse_prefill,
        .max_batch_size       = cfg_->orchestrator.max_batch_size,
        // TD-PREFILL-SUPERCHUNK: persistent sparse top-k rows for the whole
        // superchunk (IndexShare shared layers consume per sub-chunk).
        .superchunk_tokens    = cfg_->compute.prefill_superchunk_tokens,
        .num_layers           = cfg_->model.num_hidden_layers
                                  + cfg_->model.num_nextn_predict_layers,
        .hidden_size          = cfg_->model.hidden_size,
        .num_attention_heads  = cfg_->model.num_attention_heads,
        .q_lora_rank          = cfg_->model.q_lora_rank,
        // V4-7b: the V4 config silently carries MLA schema defaults for the
        // MLA dims (ticket-A deviation) — override them with the V4-native
        // shapes so the executor's generic buffers size correctly:
        // q_heads_ [mb, h_q·head_dim], kv_compressed_ [mb, head_dim+rope].
        .kv_lora_rank         = model_cfg_->is_v4()
                                    ? cfg_->model.head_dim
                                    : cfg_->model.kv_lora_rank,
        .qk_rope_head_dim    = cfg_->model.qk_rope_head_dim,
        .qk_nope_head_dim    = model_cfg_->is_v4()
                                    ? cfg_->model.head_dim
                                          - cfg_->model.qk_rope_head_dim
                                    : cfg_->model.qk_nope_head_dim,
        .v_head_dim           = model_cfg_->is_v4()
                                    ? cfg_->model.head_dim
                                    : cfg_->model.v_head_dim,
        .rms_norm_eps         = static_cast<float>(cfg_->model.rms_norm_eps),
        .rope_cos_sin_host    = rope_table.data(),
        .rope_max_pos         = rope_max_pos,
        // V4-4c: compress-theta table for compressed layers (null non-V4).
        .rope_cos_sin_compress_host = rope_table_compress.empty()
                                          ? nullptr
                                          : rope_table_compress.data(),
        // V4-5c grouped o_proj (ticket G): dims + oa-scratch enable. Zero
        // (inert) for every non-grouped-o_proj model — no allocation, no
        // behavior change on MLA paths.
        .v4_head_dim          = model_cfg_->has_grouped_o_proj()
                                    ? cfg_->model.head_dim : 0,
        .v4_o_groups          = model_cfg_->has_grouped_o_proj()
                                    ? cfg_->model.o_groups : 0,
        .v4_o_lora_rank       = model_cfg_->has_grouped_o_proj()
                                    ? cfg_->model.o_lora_rank : 0,
        // V4-7b (ticket H): full V4 attention-pipeline config — geometry
        // mirrors VramLayout.v4 + the configured CsaHcaSm120AttentionDevice;
        // tier region bases per TP rank (kMain/CSA base flows per call via
        // AttentionExecParams::kv_cache_ptrs).
        .v4                   = [&]{
            parallelism::DcpExecutor::Options::V4Exec v{};
            const bool v4_be =
                cfg_->compute.attention_backend
                    == config::AttentionBackendType::csa_hca
                || cfg_->compute.attention_backend
                    == config::AttentionBackendType::csa_hca_tq
                || cfg_->compute.attention_backend
                    == config::AttentionBackendType::csa_hca_tq_mix;
            if (model_cfg_->is_v4() && v4_be
                && !backends_.skip_hardware_detection) {
                const auto& v4l = vram_allocator_->layout().v4;
                v.enabled = true;
                // V4-5T: per-tier codec flags (executor compress-insert
                // branch + the device attention comp_tq argument).
                v.csa_tq = v4l.csa_format == memory::KvCacheFormat::kV4Tq;
                v.hca_tq = v4l.hca_format == memory::KvCacheFormat::kV4Tq;
                v.num_layers = cfg_->model.num_hidden_layers;
                v.attn_type.resize(
                    static_cast<size_t>(cfg_->model.num_hidden_layers));
                for (int l = 0; l < cfg_->model.num_hidden_layers; ++l) {
                    switch (model_cfg_->attention_type_for_layer(l)) {
                        case model::V4AttentionType::kSwa:
                            v.attn_type[static_cast<size_t>(l)] = 0; break;
                        case model::V4AttentionType::kCsa:
                            v.attn_type[static_cast<size_t>(l)] = 1; break;
                        case model::V4AttentionType::kHca:
                            v.attn_type[static_cast<size_t>(l)] = 2; break;
                    }
                }
                v.sliding_window = cfg_->model.sliding_window;
                v.csa_ratio = memory::kV4CsaRatio;
                v.hca_ratio = memory::kV4HcaRatio;
                v.csa_entries_per_page = v4l.csa_entries_per_page;
                v.hca_entries_per_page = v4l.hca_entries_per_page;
                v.swa_page_tokens = v4l.swa_page_tokens;
                v.idx_entries_per_page = v4l.indexer_entry_bytes > 0
                    ? static_cast<int>(v4l.indexer_bytes_per_page
                                       / v4l.indexer_entry_bytes)
                    : 0;
                v.idx_page_bytes = v4l.indexer_bytes_per_page;
                v.topk = cfg_->model.index_topk;
                v.max_seq = cfg_->serving.max_sequence_length;
                // Ticket J: dspark speculation needs rewind-lossless ring
                // snapshots + multi-row verify chunks in the executor.
                v.spec_snapshots = speculation::has_dspark(*cfg_);
                for (int i = 0; i < tp; ++i) {
                    const int pos = tp_gpus[static_cast<size_t>(i)].position;
                    const auto& reg = vram_allocator_->region(pos);
                    v.hca_base.push_back(reg.kv_hca);
                    v.swa_base.push_back(reg.kv_swa);
                    v.idx_base.push_back(reg.indexer_k);
                }
            }
            return v;
        }(),
        // V4-7b: V4 carries index_topk=512 for its Lightning indexer, but the
        // DSA machinery (producer arena, indexer-K cache, IndexShare) is
        // MLA-only — has_dsa() is false for V4 (ticket A) and the V4 pipeline
        // does its own lightning selection in execute_attention_v4.
        .has_dsa              = model_cfg_->has_dsa()
                                  && cfg_->model.index_topk > 0,
        .index_topk           = cfg_->model.index_topk,
        .index_n_heads        = cfg_->model.index_n_heads,
        .index_head_dim       = cfg_->model.index_head_dim,
        // IndexShare (GLM-25b): per-layer full/shared mask. Empty (no sharing)
        // unless index_topk_freq configures it — the GLM-5.2 GGUF drops
        // indexer_types, so the preset carries freq=4/offset=3 to reconstruct
        // the trained 21-full-layer pattern.
        .indexer_full_layers  = [&]{
            std::vector<uint8_t> m;
            const auto& mask = model_cfg_->full_index_layer_mask();
            m.reserve(mask.size());
            for (bool full : mask) m.push_back(full ? 1 : 0);
            return m;
        }(),
        // GG-4: GGUF attention GEMM dispatch. Active when the checkpoint is GGUF
        // (weights_format) or the configured weight quant is a gguf* variant.
        // gguf_strategy gates int (mmvq/mmq) vs dequant; gates Q8_1 workspace.
        .gguf_active          =
            cfg_->model.weights_format == config::WeightsFormat::gguf
            || model::gguf::is_gguf_weight_quant(cfg_->quantization.weights),
        .gguf_strategy        = cfg_->quantization.gguf_strategy,
        .communicator         = dcp_communicator_.get(),
        .stream_manager       = stream_manager_.get(),
        .graph_registry       = graph_registry_.get(),
        .dcp_wrapper          = dcp_attention_wrapper_.get(),
        .attention_devices    = attn_ptrs,
        .device_backends      = [&]{
            std::vector<compute::DeviceBackend*> v;
            for (int i = 0; i < tp; ++i)
                v.push_back(device_backends_[static_cast<size_t>(
                    tp_gpus[static_cast<size_t>(i)].position)].get());
            return v;
        }(),
    };
    // INV-KVS-QAG (TD-KVS-Q-ALLGATHER resolved): sharded KV runs the Q-head
    // allgather structure — every rank attends ALL tp*HL heads over its LOCAL
    // token shard, the LSE combine merges same-head partials across ranks,
    // and each rank's kv_bv/o_proj consume its own HL-head combined slice.
    // Nongraph-only (TD-KVS-QAG-GRAPH).
    if (de_opts.dcp_kv_sharded) {
        spdlog::info(
            "dcp_kv_mode=sharded: Q-head-allgather DCP combine active "
            "(INV-KVS-QAG, all {} heads per rank over local shards; "
            "nongraph-only)", cfg_->model.num_attention_heads);
    }
    dcp_executor_ = std::make_unique<parallelism::DcpExecutor>(
        std::move(de_opts));

    // Register DcpExecutor intermediate buffers with buffer registry.
    dcp_executor_->register_buffers(*buffer_registry_);
    spdlog::info("Buffer registry: {} entries after DcpExecutor",
                 buffer_registry_->size());

    // DCP page routing config for PageAllocator (TP >= 2 only)
    if (tp >= 2) {
        // TD-V4-KMAIN-SIZING: V4 kMain pages are LOGICAL-BLOCK granular
        // (256 native tokens = one 64-entry CSA page, ticket-H geometry) —
        // the config page_size_tokens (16) is the non-V4 token-page unit.
        // dcp_chunk_size must stay a page multiple (set_dcp_config assert);
        // V4 sharded KV is fail-closed (TD-V4-DCP-KV), so chunk routing is
        // inert under V4 — round it up to the page size.
        int dcp_page_tokens = cfg_->memory.kv_cache.page_size_tokens;
        int dcp_chunk_tokens = cfg_->memory.kv_cache.dcp_chunk_size;
        if (model_cfg_->is_v4()) {
            dcp_page_tokens = vram_allocator_->layout().v4.logical_block_tokens;
            if (dcp_chunk_tokens % dcp_page_tokens != 0) {
                dcp_chunk_tokens = dcp_page_tokens
                    * ((dcp_chunk_tokens + dcp_page_tokens - 1)
                       / dcp_page_tokens);
            }
        }
        memory::DcpConfig dcp_cfg{
            .dcp_size          = tp,
            .dcp_chunk_size    = dcp_chunk_tokens,
            .page_size_tokens  = dcp_page_tokens,
            .tp_gpu_indices    = tp_gpu_indices,
            .indexer_k_sharded =
                (cfg_->hardware.dcp_indexer_mode == config::DcpIndexerMode::local),
            // TD-GLM-INDEXER-LOCAL-MERGE: local-mode ownership unit —
            // indexer-K pages are rank-routed round-robin by INDEXER PAGE.
            .indexer_k_page_size_tokens =
                cfg_->memory.kv_cache.indexer_k_page_size_tokens,
            // INV-KV-REP: replicated (default) claims each KV page on EVERY
            // TP GPU in lockstep; sharded owner-routes by token position.
            .kv_sharded =
                (cfg_->hardware.dcp_kv_mode == config::DcpKvMode::sharded),
        };
        page_allocator_->set_dcp_config(std::move(dcp_cfg));
    }

    // Step 17: Statistics modules
    const uint32_t num_moe   = static_cast<uint32_t>(model_cfg_->num_moe_layers());
    const uint32_t num_exp   = static_cast<uint32_t>(cfg_->model.n_routed_experts);
    const uint32_t first_moe = static_cast<uint32_t>(cfg_->model.first_k_dense_replace);

    expert_stats_ = std::make_unique<statistics::ExpertStats>(
        statistics::ExpertStats::Options{
            .ewma_alpha      = cfg_->orchestrator.workload_detection.ewma_alpha,
            .num_moe_layers  = num_moe,
            .num_experts     = num_exp,
            .first_moe_layer = first_moe,
        });

    coactivation_graph_ = std::make_unique<statistics::CoactivationGraph>(
        statistics::CoactivationGraph::Options{
            .num_moe_layers       = num_moe,
            .num_experts          = num_exp,
            .first_moe_layer      = first_moe,
            .decay_factor         = cfg_->orchestrator.coactivation_graph.decay_factor,
            .workload_shift_decay = cfg_->orchestrator.coactivation_graph.workload_shift_decay,
        });

    workload_detector_ = std::make_unique<statistics::WorkloadDetector>(
        statistics::WorkloadDetector::Options{
            .num_moe_layers          = num_moe,
            .num_experts             = num_exp,
            .first_moe_layer         = first_moe,
            .token_window_size       = cfg_->orchestrator.workload_detection._advanced.token_window_size,
            .shift_threshold_std_devs = cfg_->orchestrator.workload_detection.shift_threshold_std_devs,
        });

    acceptance_tracker_ = std::make_unique<statistics::AcceptanceTracker>(
        statistics::AcceptanceTracker::Options{});

    // WP-6: Skip routed expert loading when prepacked files are available
    // and legacy_weights is not forced.
    bool skip_routed_experts = false;
    if (!cfg_->preprocessing.legacy_weights) {
        namespace fs = std::filesystem;
        if (!cfg_->preprocessing.prepacked_dir.empty()) {
            auto mf = model::prepacked::manifest_path(
                cfg_->preprocessing.prepacked_dir);
            if (fs::exists(mf)) {
                skip_routed_experts = true;
                spdlog::info("WP-6: Skipping routed expert loading "
                             "(prepacked_dir valid)");
            }
        }
    }

    // Round 2b: the weight load is CPU/mmap-only — release the early

    // arena worker's registration now so it overlaps THIS phase (see

    // ArenaEarlyState::start_register).

    arena_early_.start_register.store(true, std::memory_order_release);



    // Step 18: Load model weights (optional)
    if (!backends_.skip_weight_loading) {
        auto lm = model::load_weights(*cfg_, *model_cfg_, *layer_registry_,
                                      skip_routed_experts);
        loaded_model_ = std::make_unique<model::LoadedModel>(std::move(lm));
        spdlog::info("Loaded {} tensors ({} bytes)",
                     loaded_model_->total_tensors_loaded,
                     loaded_model_->total_weight_bytes);
    }

    // Step 18b: Auto-preprocess expert weights if configured (WP-2)
    if (loaded_model_ && cfg_->preprocessing.auto_preprocess) {
        namespace fs = std::filesystem;

        // Determine target directory: prepacked_dir if set, else auto_preprocess_target.
        std::string target;
        if (!cfg_->preprocessing.prepacked_dir.empty()) {
            target = cfg_->preprocessing.prepacked_dir;
            if (!cfg_->preprocessing.auto_preprocess_target.empty())
                spdlog::warn("Both prepacked_dir and auto_preprocess_target set; "
                             "using prepacked_dir='{}'", target);
        } else if (!cfg_->preprocessing.auto_preprocess_target.empty()) {
            target = cfg_->preprocessing.auto_preprocess_target;
        }

        if (!target.empty()) {
            // Preprocess if no manifest, or if source weights are newer (TD-93c).
            auto manifest_file = model::prepacked::manifest_path(target);
            bool needs_preprocess = !fs::exists(manifest_file);

            if (!needs_preprocess) {
                // Check freshness: compare manifest timestamp against shard mtimes.
                try {
                    auto manifest = model::read_manifest(target);
                    // Compare manifest timestamp against every source shard mtime
                    // (safetensors `shards` OR gguf `gguf_shards`, whichever the
                    // load path filled).
                    auto newer_than_manifest = [&](const fs::path& p) -> bool {
                        std::error_code ec;
                        auto ftime = fs::last_write_time(p, ec);
                        if (ec) return false;
                        auto sys_time = std::chrono::clock_cast<
                            std::chrono::system_clock>(ftime);
                        auto epoch_s = std::chrono::duration_cast<
                            std::chrono::seconds>(
                            sys_time.time_since_epoch()).count();
                        if (epoch_s > manifest.source_freshness_timestamp) {
                            spdlog::info("Source weights newer than pre-processed "
                                         "data ({} mtime {} > manifest {}), "
                                         "re-preprocessing",
                                         p.filename().string(), epoch_s,
                                         manifest.source_freshness_timestamp);
                            return true;
                        }
                        return false;
                    };
                    for (const auto& shard : loaded_model_->shards) {
                        if (newer_than_manifest(shard.path())) { needs_preprocess = true; break; }
                    }
                    if (!needs_preprocess) {
                        for (const auto& shard : loaded_model_->gguf_shards) {
                            if (newer_than_manifest(shard.path())) { needs_preprocess = true; break; }
                        }
                    }
                } catch (const std::exception& e) {
                    spdlog::warn("Failed to read manifest for freshness check: "
                                 "{}, re-preprocessing", e.what());
                    needs_preprocess = true;
                }
            }

            if (needs_preprocess) {
                // TD-97b: routed experts were skipped (WP-6) — can't re-preprocess
                // without them. Use existing (stale) prepacked files instead.
                if (skip_routed_experts) {
                    spdlog::warn("Prepacked files are stale but routed experts "
                                 "were not loaded (WP-6). Using existing files. "
                                 "Set preprocessing.legacy_weights=true to force "
                                 "re-preprocessing.");
                } else {
                    spdlog::info("Auto-preprocessing experts into {}...", target);
                    // quant_ is already a correctly-sized expert interface for
                    // every format including GGUF (generic `gguf` was resolved to
                    // an owned mixed GgufQuantInterface at step 4;
                    // TD-GGUF-GENERIC-DEFAULT-MISSIZE).
                    auto result = model::prepack_experts(
                        *loaded_model_, *model_cfg_, *quant_, *cfg_,
                        fs::path(target));
                    if (!result.error.empty()) {
                        throw std::runtime_error(
                            "Auto-preprocess failed: " + result.error);
                    }
                    spdlog::info("Auto-preprocess complete: {} written, "
                                 "{} skipped",
                                 result.experts_written, result.experts_skipped);
                }
            }
            // Update prepacked_dir so WP-3 (PrepackedSource) can pick it up.
            cfg_->preprocessing.prepacked_dir = target;
        }
    }

    // Step 18c: Create PrepackedSource if prepacked_dir is set and valid (WP-3).
    // WP-6: Skip when legacy_weights forces legacy loading.
    if (!cfg_->preprocessing.prepacked_dir.empty() &&
        !cfg_->preprocessing.legacy_weights) {
        namespace fs = std::filesystem;
        auto manifest_file = model::prepacked::manifest_path(
            cfg_->preprocessing.prepacked_dir);
        if (fs::exists(manifest_file)) {
            try {
                // Stage 2 (TD-J-1-f): mmap-free direct reads when the arena is on
                // and direct_load is enabled (default mirrors preload). The arena
                // is then the deterministic warm tier; cold reads pread from disk
                // (no page cache competing). o_direct bypasses the cache entirely.
                const bool eff_direct =
                    cfg_->memory.pin_host_expert_pool &&
                    cfg_->memory.pin_host_expert_pool_direct_load.value_or(
                        cfg_->memory.pin_host_expert_pool_preload);
                const bool eff_odirect =
                    eff_direct && cfg_->memory.pin_host_expert_pool_direct_o_direct;
                prepacked_source_ = std::make_unique<model::PrepackedSource>(
                    cfg_->preprocessing.prepacked_dir, *quant_,
                    eff_direct, eff_odirect);
                spdlog::info("PrepackedSource: {} expert files from {}{}",
                             prepacked_source_->num_expert_files(),
                             cfg_->preprocessing.prepacked_dir,
                             eff_direct ? (eff_odirect ? " (direct O_DIRECT)"
                                                       : " (direct pread)") : "");
                // WP-7b: Init pinned staging pool for mmap → GPU transfers.
                // Replaces bulk cudaHostRegister (register_for_pinned_dma).
                if (transfer_engine_ && !backends_.skip_hardware_detection) {
                    constexpr int kPinnedPoolCount = 4;
                    transfer_engine_->init_pinned_pool(
                        static_cast<size_t>(
                            layer_registry_->per_routed_expert_bytes()),
                        kPinnedPoolCount, dev_ptrs[0]);
                }
            } catch (const std::exception& e) {
                spdlog::error("PrepackedSource init failed (legacy fallback): {}",
                              e.what());
                prepacked_source_.reset();
            }
        }
    }

    // TD-97c: If routed experts were skipped but PrepackedSource failed to init,
    // there is no expert data from any source — fatal error.
    if (skip_routed_experts && !prepacked_source_) {
        throw std::runtime_error(
            "PrepackedSource init failed but routed experts were not loaded "
            "(WP-6). Set preprocessing.legacy_weights=true or fix prepacked "
            "files in " + cfg_->preprocessing.prepacked_dir);
    }

    // Step 18c2: Wire NvmeTier ↔ PrepackedSource mmap sharing (WP-5).
    if (nvme_tier_ && prepacked_source_) {
        nvme_tier_->attach_prepacked_source(*prepacked_source_);
    }

    // Step 18d: Create PackedBufferCache (WP-4).
    {
        auto cache_mode =
            (prepacked_source_ &&
             cfg_->preprocessing.host_cache_mode == config::HostCacheMode::mmap)
            ? model::PackedBufferCache::Mode::kMmap
            : model::PackedBufferCache::Mode::kExplicit;

        int64_t budget_bytes = static_cast<int64_t>(cfg_->memory.host_packed_cache_mb)
                               * 1024 * 1024;
        if (cfg_->memory.host_packed_cache_mb == -1) budget_bytes = -1;

        packed_cache_ = std::make_unique<model::PackedBufferCache>(
            model::PackedBufferCache::Options{
                .mode = cache_mode,
                .budget_bytes = budget_bytes,
                .slot_size_bytes = layer_registry_->per_routed_expert_bytes(),
                .expert_shape = {cfg_->model.hidden_size,
                                 cfg_->model.moe_intermediate_size},
                // 481-1: explicit-mode retained buffers are NUMA-bound + pinned
                // (opt-in via memory.pin_host_expert_pool — else staging path).
                .numa = cfg_->memory.pin_host_expert_pool ? numa_manager_.get()
                                                          : nullptr,
            });

        // P-24b lever-1 safety: if the early worker ran but the arena gates
        // no longer hold here (e.g. PrepackedSource failed to construct after
        // the worker had already attached), join it and discard its results —
        // the paths below would never consume them. BUSY stays fatal.
        {
            const bool arena_gates_ok =
                cfg_->memory.preload_expert_buffers &&
                cache_mode == model::PackedBufferCache::Mode::kMmap &&
                prepacked_source_ && cfg_->memory.pin_host_expert_pool &&
                numa_manager_ && numa_manager_->numa_available() &&
                !backends_.skip_hardware_detection;
            if (arena_early_.launched && !arena_gates_ok) {
                arena_early_.start_register.store(true, std::memory_order_release);
                if (arena_early_.thread.joinable())
                    arena_early_.thread.join();
                if (arena_early_.fatal)
                    std::rethrow_exception(arena_early_.fatal);
                if (pinned_arena_ || arena_ipc_) {
                    spdlog::warn("arena_attach: early worker results discarded "
                                 "— arena preconditions no longer hold");
                    pinned_arena_.reset();
                    arena_cache_.reset();
                    arena_meta_.reset();
                    arena_ipc_.reset();  // detach; holder keeps its fds
                }
            }
        }

        if (cfg_->memory.preload_expert_buffers) {
            if (cache_mode == model::PackedBufferCache::Mode::kMmap &&
                prepacked_source_) {
                // P-24: when pin_host_expert_pool is on, build ONE pinned,
                // NUMA-bound anonymous arena per GPU-attached node and register
                // it once (Portable) — ~tens of ms/GB vs the 481-1 per-file
                // register's ~5.7 s/GB (TD-100c). The arena is the warm DMA tier;
                // host_source reads prepacked bytes into a pinned slot on miss and
                // DMAs straight from it (no staging). When off: plain madvise +
                // the pinned staging pool (byte-for-byte today's behavior).
                if (cfg_->memory.pin_host_expert_pool && numa_manager_ &&
                    numa_manager_->numa_available() &&
                    !backends_.skip_hardware_detection) {
                    // Size to the full routed-expert set; the constructor caps
                    // each node per pin_host_expert_pool_sizing (fraction of
                    // total/available RAM or an absolute amount, with optional
                    // per-node overrides — INV-4.12f) and slab-manages the rest
                    // with LRU.
                    // Arena slot = the prepacked on-disk STRIDE (padded for
                    // O_DIRECT) so a slot holds a full slot read; the H2D still
                    // copies only the real expert bytes (expert_cache->expert_bytes).
                    // Equals per_routed_expert_bytes for legacy/unpadded data.
                    const size_t slot_bytes = prepacked_source_
                        ? static_cast<size_t>(prepacked_source_->slot_size_bytes())
                        : static_cast<size_t>(
                              layer_registry_->per_routed_expert_bytes());
                    const size_t total_expert_bytes = slot_bytes
                        * static_cast<size_t>(cfg_->model.n_routed_experts)
                        * static_cast<size_t>(model_cfg_->num_moe_layers());
                    // TD-INIT-OVERLAP: when we will bulk-preload, defer the
                    // per-node cudaHostRegister and run it concurrently with the
                    // preload (register = GPU page-tables, preload = NVMe; both
                    // only need the prefaulted pages). Hides the ~21 s register
                    // under the ~100 s preload.
                    const bool will_preload =
                        prepacked_source_ &&
                        cfg_->memory.pin_host_expert_pool_preload;
                    const size_t arena_scratch = static_cast<size_t>(
                        cfg_->memory.pin_host_expert_pool_extra_scratch_bytes);

                    // P-24b: persistent arena — the EARLY worker (launched
                    // before rank init, lever-1) already attached and, on a
                    // warm boot, adopted + registered the stored arena. Join
                    // it here; BUSY is FATAL (a second attached engine would
                    // double host RAM); worker failure / holder-unreachable
                    // fell back to arena_ipc_ == null → classic private arena.
                    arena_early_.start_register.store(
                        true, std::memory_order_release);  // never leave the worker parked
                    if (arena_early_.launched &&
                        arena_early_.thread.joinable())
                        arena_early_.thread.join();
                    if (arena_early_.fatal)
                        std::rethrow_exception(arena_early_.fatal);
                    numa_manager_->set_shared_thp(
                        cfg_->memory.pin_host_expert_pool_thp);
                    memory::ArenaBacking backing{};
                    const bool warm_attach = arena_early_.warm_ready;
                    try {
                        // Identity hashes/node sets were computed by the early
                        // worker (config-stable only — see hash_config); the
                        // warm path is DONE (arena built + registered on the
                        // worker). Verify the manifest stride the worker used
                        // matches the PrepackedSource actually constructed
                        // (an auto-preprocess in between could change it).
                        const uint64_t geom_hash = arena_early_.geom_hash;
                        const uint64_t source_id = arena_early_.source_id;
                        if (arena_ipc_ && arena_early_.slot_bytes != slot_bytes)
                            throw std::runtime_error(
                                "arena_attach: prepacked stride changed during "
                                "init (" +
                                std::to_string(arena_early_.slot_bytes) +
                                " != " + std::to_string(slot_bytes) + ")");
                        if (arena_ipc_ && arena_early_.cold_attached)
                            backing.mode =
                                memory::ArenaBackingMode::kSharedCreate;

                        if (!pinned_arena_) {
                            pinned_arena_ =
                                std::make_unique<memory::PinnedExpertArena>(
                                    *numa_manager_, slot_bytes,
                                    total_expert_bytes,
                                    cfg_->memory.pin_host_expert_pool_sizing,
                                    cfg_->memory.cross_node_spill,
                                    arena_scratch,
                                    /*defer_registration=*/will_preload,
                                    arena_ipc_ ? &backing : nullptr);
                        }
                        spdlog::info("PinnedExpertArena: {} node arena(s), {} "
                                     "slots, {:.1f} GB pinned (P-24{})",
                                     pinned_arena_->num_arenas(),
                                     pinned_arena_->total_slots(),
                                     pinned_arena_->total_pinned_bytes()
                                         / 1073741824.0,
                                     warm_attach
                                         ? "b, ADOPTED"
                                         : (arena_ipc_ ? "b, shared" : ""));

                        // Cold + attached: fresh meta segment + STORE the new
                        // generation with the holder. The geometry recorded is
                        // the arena AS BUILT (the ctor resolved today's free-
                        // RAM budgets) — a later warm attach adopts exactly it.
                        if (arena_ipc_ &&
                            backing.mode ==
                                memory::ArenaBackingMode::kSharedCreate) {
                            std::vector<memory::ArenaCacheNodeGeom> geom;
                            for (const auto& [node, sd] : arena_early_.nodeids)
                                if (const auto* na =
                                        pinned_arena_->node_arena(node))
                                    geom.push_back(
                                        {node,
                                         static_cast<uint64_t>(
                                             na->num_slots())});
                            // TD-ARENA-MIGRATE-EMA-PERSIST: size the fresh
                            // meta segment with the EMA trailer region so a
                            // later online-placement run can persist its
                            // fetch EMA across boots.
                            arena_meta_ =
                                std::make_unique<memory::ArenaMetaSegment>(
                                    memory::ArenaMetaSegment::create(
                                        memory::ArenaCache::
                                            required_bytes_with_ema(
                                                geom,
                                                static_cast<uint32_t>(
                                                    cfg_->model
                                                        .num_hidden_layers),
                                                static_cast<uint32_t>(
                                                    cfg_->model
                                                        .n_routed_experts))));
                            bool stored = arena_meta_->valid() && !geom.empty();
                            if (stored) {
                                arena_cache_ =
                                    std::make_unique<memory::ArenaCache>(
                                        arena_meta_->base(),
                                        arena_meta_->bytes());
                                stored = arena_cache_->format(geom_hash,
                                                              source_id, geom);
                            }
                            if (stored) {
                                std::vector<memory::ArenaIpcSegment> descs;
                                for (const auto& g : geom)
                                    descs.push_back(
                                        {g.node,
                                         pinned_arena_->node_backing_bytes(
                                             g.node),
                                         pinned_arena_->node_backing_fd(
                                             g.node)});
                                stored = arena_ipc_->store(arena_meta_->fd(),
                                                           descs);
                            }
                            if (!stored) {
                                spdlog::warn("arena_attach: STORE failed — "
                                             "persistence disabled this run");
                                arena_cache_.reset();
                                arena_meta_.reset();
                            }
                        }

                        // Wire the persistence meta layer + adopt warm slots.
                        if (arena_cache_) {
                            arena_cache_->set_file_identities(
                                memory::ArenaCache::stat_expert_files(
                                    cfg_->preprocessing.prepacked_dir,
                                    static_cast<uint32_t>(
                                        cfg_->model.n_routed_experts)));
                            pinned_arena_->set_cache(arena_cache_.get());
                            if (warm_attach) {
                                const size_t adopted =
                                    arena_cache_->scan_adoptable(
                                        static_cast<uint32_t>(
                                            cfg_->model.num_hidden_layers),
                                        static_cast<uint32_t>(
                                            cfg_->model.n_routed_experts),
                                        [&](int node, size_t slot,
                                            memory::ExpertKey key) {
                                            pinned_arena_->adopt_ready(
                                                key, node, slot);
                                        });
                                spdlog::info("arena_attach: adopted {} warm "
                                             "slot(s) — preload covers only "
                                             "the gaps", adopted);
                            }
                        }
                    } catch (const std::exception& e) {
                        spdlog::error("PinnedExpertArena build failed (falling "
                                      "back to staging pool): {}", e.what());
                        pinned_arena_.reset();
                        arena_cache_.reset();
                        arena_meta_.reset();
                        arena_ipc_.reset();  // detach; holder keeps its fds
                        for (auto& [n, b] : backing.adopted)
                            numa_manager_->free(b);
                    }
                    // Since prepack format 9.67.0 ("nvfp4-sm1xx"), packed
                    // expert bytes carry Sm1xx-interleaved scales at EVERY
                    // tier (disk/host/VRAM) — the reformat happens once at
                    // PACK time (pack_nvfp4_expert / the prepacker), so the
                    // old per-fill arena post_fill reformat hook is gone
                    // (it cost ~3 ms/slot on the filling thread, +47 s on
                    // full preload — TD-GOLDEN-REFORMAT-COST).
                    // J-1: spin up the async cold-load worker pool alongside the
                    // arena. On an arena miss the daemon submits the file→slot
                    // copy here instead of blocking poll_completions (INV-3.4.2).
                    if (pinned_arena_) {
                        // Stage 3: io_uring backend when requested AND direct mode
                        // is active (needs open fds); else the J-1 worker pool.
                        const bool use_iouring =
                            cfg_->memory.pin_host_expert_pool_iouring &&
                            prepacked_source_ && prepacked_source_->is_direct();
                        arena_loader_ = std::make_unique<memory::ArenaLoader>(
                            0, use_iouring
                                   ? memory::ArenaLoader::Backend::kIoUring
                                   : memory::ArenaLoader::Backend::kWorkerPool);
                        spdlog::info("ArenaLoader: backend={}, {} worker(s)",
                                     arena_loader_->backend() ==
                                             memory::ArenaLoader::Backend::kIoUring
                                         ? "io_uring" : "worker-pool",
                                     arena_loader_->num_workers());
                    }
                    // Warm-by-definition preload: load expert weights into the
                    // arena from the prepacked source at startup, then drop the
                    // prepacked page cache (MADV_DONTNEED) so it doesn't compete
                    // with the pinned arena for RAM. Removes the runtime dependency
                    // on the prepacked cache (a large lazy-filled arena would
                    // otherwise evict it, forcing cold loads to disk).
                    if (pinned_arena_ && prepacked_source_ &&
                        cfg_->memory.pin_host_expert_pool_preload) {
                        const auto t0 = std::chrono::steady_clock::now();
                        // Bulk preload is io_uring's win (deep queue saturates the
                        // NVMe → measured 2.7× vs single-threaded pread), so use it
                        // BY DEFAULT for preload whenever direct mode is active
                        // (needs open fds) — independent of the runtime loader,
                        // which stays worker-pool unless explicitly asked (io_uring
                        // loses the runtime cold-load trickle). Falls back to the
                        // runtime loader without liburing (kIoUring → kWorkerPool).
                        std::unique_ptr<memory::ArenaLoader> preload_loader;
                        memory::ArenaLoader* loader = arena_loader_.get();
                        if (prepacked_source_->is_direct() && arena_loader_ &&
                            arena_loader_->backend() !=
                                memory::ArenaLoader::Backend::kIoUring) {
                            preload_loader = std::make_unique<memory::ArenaLoader>(
                                0, memory::ArenaLoader::Backend::kIoUring);
                            if (preload_loader->backend() ==
                                memory::ArenaLoader::Backend::kIoUring)
                                loader = preload_loader.get();  // else liburing absent
                        }
                        // Arena host placement policy (Wave-2 M3,
                        // arena_placement.h): frequency-aware + HBM-first
                        // slot→node plan for the preload. Resolved from
                        // memory.arena_placement.freq_table with the
                        // LS_ARENA_PLACE_FREQ env override HERE (main
                        // thread, outside any fallback catch) so an
                        // unreadable table fails init loudly. nullptr = the
                        // legacy tiered fill, byte-for-byte.
                        std::unordered_map<memory::ExpertKey, int> place_map;
                        {
                            const auto place_policy =
                                memory::ArenaPlacementPolicy::resolve(
                                    cfg_->memory.arena_placement.freq_table);
                            if (place_policy.enabled) {
                                const auto ftab = memory::ArenaFreqTable::load(
                                    place_policy.freq_path);
                                std::vector<memory::ExpertKey> pkeys;
                                for (uint32_t L = 0; L < static_cast<uint32_t>(
                                         cfg_->model.num_hidden_layers); ++L)
                                    for (uint32_t e = 0; e < static_cast<uint32_t>(
                                             cfg_->model.n_routed_experts); ++e) {
                                        memory::ExpertKey k{
                                            L, static_cast<uint16_t>(e)};
                                        if (prepacked_source_->has(k))
                                            pkeys.push_back(k);
                                    }
                                std::vector<memory::ArenaPlacementNode> pnodes;
                                for (int n : pinned_arena_->arena_nodes()) {
                                    const auto* na =
                                        pinned_arena_->node_arena(n);
                                    pnodes.push_back(
                                        {n, na->num_slots() - na->occupied(),
                                         numa_manager_->node_is_hbm(n)});
                                }
                                place_map = memory::compute_arena_placement(
                                    ftab, pkeys, pnodes);
                                size_t hbm_slots = 0;
                                for (const auto& pn : pnodes)
                                    if (pn.is_hbm) hbm_slots += pn.free_slots;
                                spdlog::info(
                                    "arena placement (LS_ARENA_PLACE_FREQ): "
                                    "ENGAGED — table '{}' ({} keys), {} slots "
                                    "planned over {} node(s) ({} HBM free "
                                    "slots hot-first), identity {:#x}",
                                    place_policy.freq_path, ftab.counts.size(),
                                    place_map.size(), pnodes.size(), hbm_slots,
                                    place_policy.identity);
                            }
                        }
                        // TD-INIT-OVERLAP: run cudaHostRegister (deferred at
                        // construction) on a side thread WHILE this thread drives
                        // the NVMe preload. Disjoint hardware + disjoint arena
                        // state (register touches base/registered, preload the
                        // slot index); register_all() throws-without-freeing so
                        // the concurrent preload's slot pointers stay valid.
                        std::exception_ptr reg_err;
                        std::thread reg_thread([&] {
                            try { pinned_arena_->register_all(); }
                            catch (...) { reg_err = std::current_exception(); }
                        });
                        size_t n = pinned_arena_->preload(
                            *prepacked_source_,
                            static_cast<uint32_t>(cfg_->model.num_hidden_layers),
                            static_cast<uint32_t>(cfg_->model.n_routed_experts),
                            loader,
                            place_map.empty() ? nullptr : &place_map);
                        reg_thread.join();
                        if (reg_err) {
                            std::string msg;
                            try { std::rethrow_exception(reg_err); }
                            catch (const std::exception& e) { msg = e.what(); }
                            spdlog::error("PinnedExpertArena registration failed "
                                          "during overlapped preload (falling back "
                                          "to staging pool): {}", msg);
                            pinned_arena_.reset();
                        } else {
                            const double secs = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
                            const double gb = n * (slot_bytes / 1073741824.0);
                            spdlog::info("PinnedExpertArena: preloaded {} slots "
                                         "({:.1f} GB) in {:.2f} s ({:.2f} GB/s) "
                                         "[register overlapped]",
                                         n, gb, secs, secs > 0 ? gb / secs : 0.0);
                            prepacked_source_->advise_dontneed();
                        }
                        // preload_loader (if any) torn down here — runtime uses
                        // arena_loader_ (worker-pool) for any cold-load misses.
                    }
                    // Arena host placement: LOUD no-op notice when the flag is
                    // set but the preload path (its only application point)
                    // did not run — a silent no-op would poison A/B walls.
                    if (!memory::ArenaPlacementPolicy::resolved_path(
                             cfg_->memory.arena_placement.freq_table)
                             .empty() &&
                        !(pinned_arena_ && prepacked_source_ &&
                          cfg_->memory.pin_host_expert_pool_preload))
                        spdlog::warn(
                            "arena placement table configured "
                            "(memory.arena_placement.freq_table / "
                            "LS_ARENA_PLACE_FREQ) but the arena preload "
                            "path is inactive — placement NOT applied");
                }
                if (!pinned_arena_) {
                    int n = packed_cache_->preload_mmap(*prepacked_source_);
                    spdlog::info("PackedBufferCache: preloaded {} mmap regions "
                                 "(madvise)", n);
                }
            } else if (cache_mode == model::PackedBufferCache::Mode::kExplicit &&
                       loaded_model_) {
                int n = packed_cache_->preload_explicit(
                    *loaded_model_,
                    static_cast<uint32_t>(cfg_->model.first_k_dense_replace),
                    static_cast<uint32_t>(model_cfg_->num_moe_layers()),
                    static_cast<uint32_t>(cfg_->model.n_routed_experts));
                spdlog::info("PackedBufferCache: preloaded {} experts (explicit)",
                             n);
            }
        }

        spdlog::info("PackedBufferCache: mode={}, budget={} MB, slot_size={} bytes",
                     cache_mode == model::PackedBufferCache::Mode::kMmap
                         ? "mmap" : "explicit",
                     cfg_->memory.host_packed_cache_mb,
                     layer_registry_->per_routed_expert_bytes());
    }

    // Step 18d2: Create CPU NUMA expert devices (C-6). One NumaCpuExpertDevice
    // per configured hardware.cpu_expert_devices entry, appended to
    // expert_devices_ AFTER the GPUs (Step 14b created the GPU ExpertDevices;
    // these CPU devices need the PinnedExpertArena built above, so they are
    // created here once the arena exists). The cpu GpuRef's `position` continues
    // after the GPU array; `id` carries the NUMA node (NOT a CUDA ordinal —
    // gpu_ref.h). num_expert_devices auto-tracks via expert_devices_.size().
    // Skipped in null-backend mode (no NUMA enumeration / arena). The host
    // moe_output these produce is folded into the TP all-reduce in a SEPARATE
    // handoff session (see TD-CPU-EXPERT-HANDOFF).
    // TASK-A2: LS_CPU_EXPERT_SPREAD=<n0,n1,...> replaces the per-config single-node
    // CPU devices with ONE MultiNumaCpuExpertDevice that computes each grouped
    // expert-FFN spread across those DDR NUMA nodes (56 physical cores across
    // nodes 0-3 vs 14 on one node), cutting the B=1 host-FFN latency toward 1/M.
    // The fold's cpu_expert_device() picks the first cpu-type device, so a single
    // multi-node device plugs in transparently. HBM nodes (4-7) have NO cpus and
    // must NOT appear here. Unset ⇒ the legacy single-node path below (unchanged).
    std::vector<int> spread_nodes;
    if (const char* sp = std::getenv("LS_CPU_EXPERT_SPREAD"); sp && *sp) {
        std::stringstream ss(sp);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            try { spread_nodes.push_back(std::stoi(tok)); }
            catch (const std::exception&) {}
        }
    }
    if (!backends_.skip_hardware_detection && !spread_nodes.empty() &&
        !cfg_->hardware.cpu_expert_devices.empty()) {
        int mt = 0;
        for (const auto& ced : cfg_->hardware.cpu_expert_devices)
            if (ced.enabled) { mt = ced.max_threads; break; }
        config::GpuRef cpu_ref{};
        cpu_ref.position = static_cast<int>(expert_devices_.size());
        cpu_ref.id = spread_nodes.front();   // primary node (deps.nodes[0]).
        cpu_ref.type = config::GpuType::cpu;
        compute::MultiNumaCpuExpertDeps mdeps{};
        mdeps.nodes = spread_nodes;
        mdeps.numa = numa_manager_.get();
        mdeps.arena = pinned_arena_.get();
        mdeps.dims = compute::CpuExpertModelDims{
            cfg_->model.hidden_size, cfg_->model.moe_intermediate_size,
            cfg_->model.num_experts_per_tok, cfg_->model.n_routed_experts};
        mdeps.max_threads_per_node = mt;
        expert_devices_.push_back(
            compute::make_multi_numa_cpu_expert_device(cpu_ref, std::move(mdeps)));
        std::string ns;
        for (size_t i = 0; i < spread_nodes.size(); ++i)
            ns += (i ? "," : "") + std::to_string(spread_nodes[i]);
        spdlog::warn("MultiNumaCpuExpertDevice: SPREAD across NUMA nodes [{}] "
                     "(position {}, max_threads/node {})",
                     ns, cpu_ref.position, mt);
    } else if (!backends_.skip_hardware_detection &&
        !cfg_->hardware.cpu_expert_devices.empty()) {
        int next_position = static_cast<int>(expert_devices_.size());
        for (const auto& ced : cfg_->hardware.cpu_expert_devices) {
            if (!ced.enabled) continue;
            config::GpuRef cpu_ref{};
            cpu_ref.position = next_position++;
            // C-6 Milestone C: place the device on the effective NUMA node
            // (LS_CPU_EXPERT_NODE, default 1 = GPU-free DDR ⇒ no H2D contention).
            const int cpu_node = effective_cpu_expert_node(ced.numa_node);
            cpu_ref.id = cpu_node;               // NUMA node, not CUDA ordinal.
            cpu_ref.type = config::GpuType::cpu;
            compute::NumaCpuExpertDeps deps{};
            deps.numa = numa_manager_.get();
            deps.arena = pinned_arena_.get();
            deps.dims = compute::CpuExpertModelDims{
                cfg_->model.hidden_size,
                cfg_->model.moe_intermediate_size,
                cfg_->model.num_experts_per_tok,
                cfg_->model.n_routed_experts};
            deps.max_threads = ced.max_threads;
            // C-6 QA (c/d): reserve the node's leading physical core for the
            // daemon/orchestrator (pinned there in spawn_daemon_thread); the FFN
            // pool uses the node's 2nd core onward. DEFAULT ON — this only runs
            // INSIDE CPU-device construction (a CPU device is configured/enabled),
            // so the pure champion with NO cpu_expert_devices never reaches here
            // and stays byte-identical. LS_CPU_EXPERT_RESERVE_CORE=0 overrides OFF
            // (legacy all-cores layout); a value >1 reserves that many cores.
            deps.reserve_leading_cores = 1;
            if (const char* rc = std::getenv("LS_CPU_EXPERT_RESERVE_CORE");
                rc && rc[0]) {
                deps.reserve_leading_cores =
                    (rc[0] == '0' && rc[1] == '\0') ? 0 : std::max(1, std::atoi(rc));
            }
            expert_devices_.push_back(
                compute::make_numa_cpu_expert_device(cpu_ref, deps));
            spdlog::warn("NumaCpuExpertDevice: created on NUMA node {} "
                         "(config numa_node={}, LS_CPU_EXPERT_NODE default 1) "
                         "(position {}, max_threads {})",
                         cpu_node, ced.numa_node, cpu_ref.position,
                         ced.max_threads);
        }
    }

    // Step 18e: Clear routed experts from LoadedModel when PrepackedSource
    // active (WP-6). Handles the auto_preprocess case where routed experts
    // were needed for preprocessing but are now redundant.
    if (prepacked_source_ && loaded_model_ &&
        !cfg_->preprocessing.legacy_weights) {
        size_t cleared = 0;
        for (auto& layer : loaded_model_->layers) {
            if (!layer.routed_experts.empty()) {
                layer.routed_experts.clear();
                layer.routed_experts.shrink_to_fit();
                ++cleared;
            }
        }
        if (cleared > 0)
            spdlog::info("WP-6: Cleared routed experts from {} MoE layers "
                         "(PrepackedSource active)", cleared);
    }

    spdlog::info("[boot] weights loaded at process age {:.1f} s",
                 process_age_seconds());

    // Step 19: Upload pinned weights to GPU VRAM (KD-3d)
    upload_pinned_weights();
    spdlog::info("[boot] pinned upload done at process age {:.1f} s",
                 process_age_seconds());

    // Step 19b: Online BF16→FP8 quantization for attention projections (KD-4f-c3)
    quantize_attention_weights();

    // Step 19c: Wire kv_b_proj metadata and prime dequant pool (TD-87a).
    wire_kv_bv_dequant();

    // Step 19c2 (P-24b RAM hygiene): every pinned weight is now in VRAM —
    // free the transform-owned host buffers (GG-9 embed/head BF16, GLM-1
    // kv_b, requants) and MADV_DONTNEED the GGUF shard mmaps, so ~GBs of
    // host RAM/page cache stop competing with the pinned arena. mmap'd
    // tensors re-fault from file if ever touched again (they aren't: the
    // only post-upload LoadedModel consumer is the routed-experts fallback,
    // which WP-6 already cleared under PrepackedSource). LS_KEEP_HOST_WEIGHTS
    // env keeps everything resident for debugging.
    if (loaded_model_ && !backends_.skip_weight_loading &&
        !std::getenv("LS_KEEP_HOST_WEIGHTS")) {
        const size_t freed = loaded_model_->release_pinned_host_bytes();
        spdlog::info("Released pinned-weight host bytes: {:.1f} GB residual "
                     "owned heap freed (streaming/ring freed the rest inline; "
                     "GGUF page cache intentionally kept — reclaimable)",
                     freed / 1073741824.0);
    }

    // Step 19d: Speculation method seam (SPEC-SCAFFOLD).  DEFAULT
    // speculation.method == none → nothing constructed, zero behavior
    // change.  #16 / GLM-25g drops the concrete MTP method into
    // make_speculation_method; draft/verify dispatch is wired then (the
    // verifier lands on the FETCH_AND_RUN_MOE seam — TD-VERIFY-FETCHSEAM).
    if (cfg_->speculation.method != config::SpeculationMethodType::none) {
        speculation::SpeculationInitContext spec_ctx;
        spec_ctx.config = cfg_.get();
        spec_ctx.attention_devices.reserve(attention_devices_.size());
        for (auto& dev : attention_devices_)
            spec_ctx.attention_devices.push_back(dev.get());
        spec_ctx.expert_devices.reserve(expert_devices_.size());
        for (auto& dev : expert_devices_)
            spec_ctx.expert_devices.push_back(dev.get());
        spec_ctx.loaded_model = loaded_model_.get();
        speculation_method_ = speculation::make_speculation_method(
            cfg_->speculation.method, spec_ctx);
        if (speculation_method_) {
            spdlog::info("Speculation method: {} (max_draft_len={})",
                         speculation_method_->name(),
                         speculation_method_->max_draft_len());
        }
    }

    // Step 19e (DSP-3): DSpark DFlash backbone runtime.  method == dspark
    // loads + uploads the draft checkpoint onto the resolved draft GPU and
    // arms the aux-hidden export (CommandDispatcher Deps.dspark) plus
    // D_CMD_RUN_DSPARK_STEP.  Fails closed (throws) on a missing/mismatched
    // checkpoint or no resolvable draft GPU.  Skipped without CUDA (null
    // backends cannot host the draft; the dspark command then CMP_ERRORs).
    if (speculation::has_dspark(*cfg_) && !backends_.skip_hardware_detection) {
        // TD-DSPARK-DRAFT-SHARD: the draft device SET — one rank (legacy
        // whole-draft placement) or two (TP=2 shard across both 5090s).
        const std::vector<int> draft_gpus =
            model::resolve_dspark_draft_gpus(*cfg_);
        const auto& rank_charges = layer_registry_->dspark_rank_charges();
        if (rank_charges.size() != draft_gpus.size())
            throw std::runtime_error(
                "dspark: LayerRegistry rank charges out of sync with the "
                "draft device set");
        std::vector<speculation::DsparkRuntime::Rank> ranks;
        std::vector<void*> rank_arenas;
        std::vector<int64_t> rank_arena_bytes;
        for (size_t r = 0; r < draft_gpus.size(); ++r) {
            const int draft_gpu = draft_gpus[r];
            if (draft_gpu < 0 ||
                draft_gpu >= static_cast<int>(device_backends_.size()) ||
                !device_backends_[static_cast<size_t>(draft_gpu)] ||
                !stream_manager_) {
                throw std::runtime_error(
                    "dspark: draft GPU " + std::to_string(draft_gpu) +
                    " has no device backend/stream manager");
            }
            // Draft stream: on a NON-TP draft GPU the kAttention stream is
            // free (no target attention runs there), so the draft pipeline
            // uses it and never contends with EP expert GEMMs on
            // kExpertFfn. On a TP draft GPU (TD-DSPARK-DRAFT-QUANT: the
            // quantized draft hosting on a 5090 beside the target weights)
            // kAttention IS the target attention stream — the draft gets
            // its own dedicated stream (one per sharded rank).
            const bool draft_on_tp =
                std::find(cfg_->hardware.tp_array.begin(),
                          cfg_->hardware.tp_array.end(),
                          draft_gpu) != cfg_->hardware.tp_array.end();
            auto* draft_backend =
                device_backends_[static_cast<size_t>(draft_gpu)].get();
            void* draft_stream;
            if (draft_on_tp) {
                draft_backend->set_device();
                // LS_DSPARK_DRAFT_LOWPRI=1 (DSP52_BOOST phase 2, default
                // OFF = byte-identical): create the dedicated draft stream
                // at the device's LOWEST scheduling priority (measured
                // neutral — the tax is DRAM/L2 bandwidth, not dispatch).
                const char* lp = std::getenv("LS_DSPARK_DRAFT_LOWPRI");
                const bool lowpri = lp && *lp == '1';
                void* ds = lowpri
                    ? draft_backend->create_stream_low_priority()
                    : draft_backend->create_stream();
                if (lowpri)
                    spdlog::info("dspark: dedicated draft stream at LOWEST "
                                 "priority (LS_DSPARK_DRAFT_LOWPRI=1)");
                dspark_draft_streams_.push_back({draft_backend, ds});
                draft_stream = ds;
            } else {
                draft_stream = stream_manager_->stream(
                    draft_gpu, compute::StreamId::kAttention);
            }
            ranks.push_back({draft_backend, draft_stream});
            // Weights + scratch are CARVED from the VramAllocator's pinned
            // region on each draft rank GPU — the LayerRegistry budget
            // charged them there (per-rank shard + scratch + align slack),
            // and the VramAllocator already allocated that region: a fresh
            // device_alloc would double-book the VRAM. On a TP draft GPU
            // the pinned region's FRONT holds the target weights
            // (upload_pinned_weights bump-allocates from offset 0; the
            // LayerRegistry charge added the draft on TOP of the layer
            // bytes), so the draft takes the region TAIL: the last
            // rank-charge bytes of the layout. Non-TP draft GPUs keep the
            // legacy offset-0 carve (their pinned region holds only the
            // draft charge).
            const auto& draft_region = vram_allocator_->region(draft_gpu);
            const auto& draft_layout = vram_allocator_->layout().gpus[
                static_cast<size_t>(draft_gpu)];
            int64_t draft_off = 0;
            if (draft_on_tp) {
                const int64_t charge = rank_charges[r];
                if (charge <= 0 || charge > draft_layout.pinned_bytes)
                    throw std::runtime_error(
                        "dspark: draft rank " + std::to_string(r) +
                        " charge " + std::to_string(charge) +
                        " does not fit the TP draft GPU's pinned region (" +
                        std::to_string(draft_layout.pinned_bytes) + " B)");
                // 256-align UP (toward the tail): the arena carve assumes a
                // 256-aligned base (FP32 scale tensors read misaligned
                // otherwise — observed 716 on the first quant GEMM), and
                // the charge's kDsparkArenaAlignSlack (1 MiB) covers the
                // <=255 B capacity loss. Aligning DOWN could collide with
                // the target weights' tail.
                draft_off = (draft_layout.pinned_bytes - charge + 255) &
                            ~int64_t{255};
                spdlog::info(
                    "dspark: TP-GPU draft placement — carving {} B at "
                    "pinned tail offset {} on GPU position {} (rank {}/{}, "
                    "dedicated draft stream)",
                    charge, draft_off, draft_gpu, r, draft_gpus.size());
            }
            rank_arenas.push_back(static_cast<char*>(draft_region.pinned) +
                                  draft_off);
            rank_arena_bytes.push_back(draft_layout.pinned_bytes -
                                       draft_off);
        }
        dspark_runtime_ = speculation::DsparkRuntime::create(
            *cfg_, std::move(ranks), dcp_communicator_.get(),
            std::move(rank_arenas), std::move(rank_arena_bytes),
            numa_manager_.get());  // EPM-1: NUMA-local dump staging
    }

    // Step 20: Capture CUDA graphs (optional)
    if (!backends_.skip_cuda_graphs) {
        // Graph capture deferred to IPC-3/4 when full command dispatch is wired.
        // At this stage graph_registry_ is empty — filled by orchestrator at runtime.
    }
}

// ── Weight upload to VRAM pinned region (KD-3d) ─────────────────���──────────

namespace {

void assign_attn_weight_ptr(parallelism::AttentionLayerWeights& w,
                            model::TensorComponent comp,
                            model::TensorRole role,
                            const void* ptr) {
    using TC = model::TensorComponent;
    using TR = model::TensorRole;
    if (role == TR::weight || role == TR::bias) {
        switch (comp) {
            case TC::q_a_proj:             w.q_a_proj = ptr; break;
            case TC::q_b_proj:             w.q_b_proj = ptr; break;
            case TC::kv_a_proj_with_mqa:   w.kv_a_proj = ptr; break;
            case TC::kv_b_proj:            w.kv_b_proj = ptr; break;
            case TC::o_proj:               w.o_proj = ptr; break;
            case TC::q_a_norm:             w.q_a_norm = ptr; break;
            case TC::kv_a_norm:            w.kv_a_norm = ptr; break;
            case TC::indexer_wq_b:         w.q_idx_b = ptr; break;
            case TC::indexer_wk:           w.k_idx = ptr; break;
            case TC::indexer_k_norm_weight: w.k_idx_norm = ptr; break;
            case TC::indexer_k_norm_bias:   w.k_idx_norm_bias = ptr; break;  // TD-74j
            case TC::indexer_weights_proj:  w.weights_proj = ptr; break;
            // ── DeepSeek-V4 components (V4-5a/5b/5c) ──
            case TC::hc_attn_fn:            w.hc_attn_fn = ptr; break;
            case TC::hc_attn_base:          w.hc_attn_base = ptr; break;
            case TC::hc_attn_scale:         w.hc_attn_scale = ptr; break;
            case TC::hc_ffn_fn:             w.hc_ffn_fn = ptr; break;
            case TC::hc_ffn_base:           w.hc_ffn_base = ptr; break;
            case TC::hc_ffn_scale:          w.hc_ffn_scale = ptr; break;
            case TC::attn_sinks:            w.attn_sinks = ptr; break;
            case TC::o_proj_a:              w.o_proj_a = ptr; break;
            case TC::o_proj_b:              w.o_proj_b = ptr; break;
            case TC::compressor_wkv:        w.compressor_wkv = ptr; break;
            case TC::compressor_wgate:      w.compressor_wgate = ptr; break;
            case TC::compressor_ape:        w.compressor_ape = ptr; break;
            case TC::compressor_norm:       w.compressor_norm = ptr; break;
            case TC::indexer_compressor_wkv:   w.indexer_compressor_wkv = ptr; break;
            case TC::indexer_compressor_wgate: w.indexer_compressor_wgate = ptr; break;
            case TC::indexer_compressor_ape:   w.indexer_compressor_ape = ptr; break;
            case TC::indexer_compressor_norm:  w.indexer_compressor_norm = ptr; break;
            default: break;
        }
    } else if (role == TR::weight_scale) {
        switch (comp) {
            case TC::q_a_proj:           w.q_a_proj_scales = ptr; break;
            case TC::q_b_proj:           w.q_b_proj_scales = ptr; break;
            case TC::kv_a_proj_with_mqa: w.kv_a_proj_scales = ptr; break;
            case TC::kv_b_proj:          w.kv_b_proj_scales = ptr; break;
            case TC::o_proj:             w.o_proj_scales = ptr; break;
            default: break;
        }
    }
}

// TD-GOLDEN bug #6: reformat a 2D NVFP4 weight-scale tensor (UE4M3 bytes,
// row-major [N, K/16]) into the Sm1xx interleaved SFB layout the NVFP4
// grouped GEMM consumes. Returns the reformatted host bytes.
static std::vector<uint8_t> reformat_sfb_host(const model::ShardedTensor& st) {
    std::vector<uint8_t> out(st.data.size());
    model::reformat_nvfp4_sfb(
        out.data(), reinterpret_cast<const uint8_t*>(st.data.data()),
        st.shape[0], st.shape[1]);
    return out;
}

struct FfnWeightPtrs {
    void* gate_up = nullptr;
    void* gate_up_scales = nullptr;
    void* down = nullptr;
    void* down_scales = nullptr;
    float weight_scale_2 = 1.f;       // NVFP4 global scale (from gate_proj bundle)
    float weight_scale_2_down = 1.f;   // NVFP4 global scale (from down_proj bundle)
    // FP4-ACT-SCALE: calibrated activation input_scales. gate/up merged via
    // max (one quantized activation feeds the fused gate+up GEMM).
    float input_scale = 1.f;
    float input_scale_down = 1.f;
    // GG-5c: each projection's OWN GGUF k-quant type (from its source tensor).
    // Sentinel default for non-GGUF weights (never read). gate!=up ⇒ the dense/
    // shared gate_up GEMM must split into two single-type GEMMs (see dispatch).
    model::GgufKQuantType gate_gguf_type = model::GgufKQuantType::Q4_K;
    model::GgufKQuantType up_gguf_type   = model::GgufKQuantType::Q4_K;
    model::GgufKQuantType down_gguf_type = model::GgufKQuantType::Q4_K;
    // V4-7b (ticket H): true iff the projection's source tensor is a packed
    // k-quant. RAW BF16 FFN tensors inside a GGUF checkpoint (V4 shexp) must
    // route through the plain BF16 GEMM, not the packed-block decoder.
    bool gate_is_gguf = false;
    bool up_is_gguf   = false;
    bool down_is_gguf = false;
};

template <typename UploadFn>
FfnWeightPtrs upload_ffn(
    const std::vector<model::ShardedWeightBundle>& sharded,
    bool nvfp4_sub_slots,
    UploadFn& upload_fn) {

    const model::ShardedWeightBundle* gate_bundle = nullptr;
    const model::ShardedWeightBundle* up_bundle = nullptr;
    const model::ShardedWeightBundle* down_bundle = nullptr;
    for (const auto& sb : sharded) {
        if (sb.id.component == model::TensorComponent::gate_proj)
            gate_bundle = &sb;
        else if (sb.id.component == model::TensorComponent::up_proj)
            up_bundle = &sb;
        else if (sb.id.component == model::TensorComponent::down_proj)
            down_bundle = &sb;
    }
    if (!gate_bundle || !up_bundle || !down_bundle) return {};

    // Upload all aux tensors from a bundle, capturing weight_scale and weight_scale_2.
    auto upload_aux = [&](const model::ShardedWeightBundle& bundle,
                          void*& scale_out, float& ws2_out) {
        for (const auto& [role, st] : bundle.aux) {
            if (role == model::TensorRole::weight_scale_2
                && st.data.size() >= sizeof(float))
                std::memcpy(&ws2_out, st.data.data(), sizeof(float));
            void* p;
            if (nvfp4_sub_slots && role == model::TensorRole::weight_scale
                && st.shape.size() == 2) {
                // TD-GOLDEN bug #6: Sm1xx-interleaved SFB for grouped GEMM.
                auto rf = reformat_sfb_host(st);
                p = upload_fn(reinterpret_cast<const std::byte*>(rf.data()),
                              static_cast<int64_t>(rf.size()));
            } else {
                p = upload_fn(st.data.data(), static_cast<int64_t>(st.data.size()));
            }
            if (role == model::TensorRole::weight_scale && !scale_out)
                scale_out = p;
        }
    };

    // TD-52a/52c: assert two pinned pointers are contiguous.
    auto assert_contiguous = [](const void* base, size_t base_size,
                                const void* next, const char* msg) {
        if (base && next &&
            next != static_cast<const std::byte*>(base) + base_size)
            throw std::runtime_error(msg);
    };

    FfnWeightPtrs ptrs;

    // ── Gate + Up (upload order differs by plan layout) ──
    ptrs.gate_up = upload_fn(gate_bundle->weight.data.data(),
                              static_cast<int64_t>(gate_bundle->weight.data.size()));

    if (nvfp4_sub_slots) {
        // Plan layout: [gate_w][up_w][gate_s][up_s][gate_ws2][gate_is][up_ws2][up_is]
        auto* up_w = upload_fn(up_bundle->weight.data.data(),
                               static_cast<int64_t>(up_bundle->weight.data.size()));
        assert_contiguous(ptrs.gate_up, gate_bundle->weight.data.size(), up_w,
                         "upload_ffn: gate+up weight contiguity violation (TD-52a)");

        auto* gate_ws = gate_bundle->find_aux(model::TensorRole::weight_scale);
        auto* up_ws   = up_bundle->find_aux(model::TensorRole::weight_scale);
        // TD-GOLDEN bug #6: upload Sm1xx-interleaved scales (per projection;
        // valid as a unit because I_local % 128 == 0, so the up half starts on
        // a fresh 128-row tile).
        if (gate_ws) {
            auto rf = reformat_sfb_host(*gate_ws);
            ptrs.gate_up_scales = upload_fn(
                reinterpret_cast<const std::byte*>(rf.data()),
                static_cast<int64_t>(rf.size()));
        }
        void* up_ws_ptr = nullptr;
        if (up_ws) {
            auto rf = reformat_sfb_host(*up_ws);
            up_ws_ptr = upload_fn(reinterpret_cast<const std::byte*>(rf.data()),
                                  static_cast<int64_t>(rf.size()));
        }
        if (gate_ws)
            assert_contiguous(ptrs.gate_up_scales, gate_ws->data.size(), up_ws_ptr,
                             "upload_ffn: gate+up scale contiguity violation (TD-52c)");

        // Scalar aux (weight_scale_2, input_scale) per projection.
        auto upload_scalar_aux = [&](const model::ShardedWeightBundle& b,
                                     float* ws2_out, float* is_out) {
            auto* ws2 = b.find_aux(model::TensorRole::weight_scale_2);
            auto* is  = b.find_aux(model::TensorRole::input_scale);
            if (ws2) {
                if (ws2_out && ws2->data.size() >= sizeof(float))
                    std::memcpy(ws2_out, ws2->data.data(), sizeof(float));
                upload_fn(ws2->data.data(), static_cast<int64_t>(ws2->data.size()));
            }
            if (is) {
                if (is_out && is->data.size() >= sizeof(float))
                    std::memcpy(is_out, is->data.data(), sizeof(float));
                upload_fn(is->data.data(), static_cast<int64_t>(is->data.size()));
            }
        };
        float gate_is = 0.f, up_is = 0.f;
        upload_scalar_aux(*gate_bundle, &ptrs.weight_scale_2, &gate_is);
        upload_scalar_aux(*up_bundle, nullptr, &up_is);
        // FP4-ACT-SCALE: gate and up describe the SAME activation tensor —
        // merge via max (ModelOpt writes identical values; TRT-LLM asserts).
        if (gate_is > 0.f && up_is > 0.f &&
            std::fabs(gate_is - up_is) > 1e-6f * std::max(gate_is, up_is)) {
            spdlog::warn("upload_ffn: gate input_scale {} != up input_scale {} "
                         "— using max", gate_is, up_is);
        }
        if (std::max(gate_is, up_is) > 0.f)
            ptrs.input_scale = std::max(gate_is, up_is);
    } else {
        // Plan layout: [gate_w][gate_aux...][up_w][up_aux...]
        upload_aux(*gate_bundle, ptrs.gate_up_scales, ptrs.weight_scale_2);

        auto* up_w = upload_fn(up_bundle->weight.data.data(),
                               static_cast<int64_t>(up_bundle->weight.data.size()));
        assert_contiguous(ptrs.gate_up, gate_bundle->weight.data.size(), up_w,
                         "upload_ffn: gate+up weight contiguity violation — non-nvfp4 (TD-52a)");

        for (const auto& [role, st] : up_bundle->aux)
            upload_fn(st.data.data(), static_cast<int64_t>(st.data.size()));
    }

    // ── Down weight + aux (identical in both paths) ──
    ptrs.down = upload_fn(down_bundle->weight.data.data(),
                           static_cast<int64_t>(down_bundle->weight.data.size()));
    upload_aux(*down_bundle, ptrs.down_scales, ptrs.weight_scale_2_down);
    if (auto* dis = down_bundle->find_aux(model::TensorRole::input_scale);
        dis && dis->data.size() >= sizeof(float)) {
        float v;
        std::memcpy(&v, dis->data.data(), sizeof(float));
        if (v > 0.f) ptrs.input_scale_down = v;
    }

    // GG-5c: carry each projection's OWN GGUF k-quant type from its source
    // tensor (GG-6 set ShardedTensor::gguf_type). On a mixed-quant GGUF the
    // dense/shared ffn_gate/ffn_up/ffn_down can be distinct families — the fused
    // MoE dispatch reads these (not the routed types) and splits gate_up when
    // gate!=up. nullopt (non-GGUF) keeps the sentinel default.
    if (gate_bundle->weight.gguf_type) {
        ptrs.gate_gguf_type = *gate_bundle->weight.gguf_type;
        ptrs.gate_is_gguf = true;
    }
    if (up_bundle->weight.gguf_type) {
        ptrs.up_gguf_type = *up_bundle->weight.gguf_type;
        ptrs.up_is_gguf = true;
    }
    if (down_bundle->weight.gguf_type) {
        ptrs.down_gguf_type = *down_bundle->weight.gguf_type;
        ptrs.down_is_gguf = true;
    }

    return ptrs;
}

}  // anonymous namespace

void Engine::upload_pinned_weights() {
    if (!loaded_model_ || backends_.skip_weight_loading || !vram_allocator_) return;

    const int tp = std::max(1, cfg_->parallelism.tensor_parallelism);
    const int num_layers = static_cast<int>(cfg_->model.num_hidden_layers);
    const int num_mtp = loaded_model_->mtp
        ? static_cast<int>(loaded_model_->mtp->block_layers.size()) : 0;

    // Identify TP GPU positions (first tp GPUs with type rtx5090 or first tp GPUs).
    std::vector<int> tp_positions;
    for (const auto& g : cfg_->hardware.gpus) {
        if (static_cast<int>(tp_positions.size()) >= tp) break;
        if (g.type == config::GpuType::rtx5090 ||
            static_cast<int>(tp_positions.size()) < tp) {
            tp_positions.push_back(g.ref.position);
        }
    }
    if (static_cast<int>(tp_positions.size()) < tp) {
        spdlog::error("upload_pinned_weights: only {} GPUs available for TP={}",
                      tp_positions.size(), tp);
        return;
    }

    model::TpWeightSharder sharder(*model_cfg_, tp);
    // GG-9: for a mixed `gguf` checkpoint, size the shared-expert / dense-FFN
    // pinned slots from THEIR OWN per-projection k-quant types (shared Q8_0, dense
    // Q5_K/Q6_K — differ from the routed `*quant_` types), else the slots are
    // under-sized and the upload overflows the pinned region.
    std::optional<model::GgufModelExpertTypes> shared_gguf_types, dense_gguf_types;
    if (cfg_->model.weights_format == config::WeightsFormat::gguf && loaded_model_) {
        shared_gguf_types = model::gguf_owner_types_from_model(
            *loaded_model_, model::TensorOwner::shared_expert);
        dense_gguf_types = model::gguf_owner_types_from_model(
            *loaded_model_, model::TensorOwner::dense_ffn);
    }
    auto upload_plan = model::build_upload_plan(
        *model_cfg_, *cfg_, *quant_, tp, 0,
        shared_gguf_types ? &*shared_gguf_types : nullptr,
        dense_gguf_types ? &*dense_gguf_types : nullptr);
    model::validate_plan(upload_plan, *loaded_model_, *model_cfg_, *cfg_, *quant_, tp);

    // Resize output vectors.
    // Position-indexed arrays are sized by num_gpus (non-TP slots stay nullptr).
    // Rank-indexed arrays (per_layer_attn_weights_) stay sized by tp.
    const auto num_gpus = static_cast<int>(cfg_->hardware.gpus.size());
    embedding_table_ptrs_.resize(num_gpus, nullptr);
    output_head_weight_ptrs_.resize(num_gpus, nullptr);
    output_head_bias_ptrs_.resize(num_gpus, nullptr);
    final_norm_ptrs_.resize(num_gpus, nullptr);
    output_hc_fn_ptrs_.resize(num_gpus, nullptr);
    output_hc_base_ptrs_.resize(num_gpus, nullptr);
    output_hc_scale_ptrs_.resize(num_gpus, nullptr);
    gating_bias_ptrs_.resize(num_layers + num_mtp, std::vector<const float*>(num_gpus, nullptr));
    hash_gating_table_ptrs_.resize(
        num_layers + num_mtp, std::vector<const int32_t*>(num_gpus, nullptr));
    per_layer_attn_weights_.resize(
        num_layers + num_mtp, std::vector<parallelism::AttentionLayerWeights>(tp));
    router_weight_ptrs_.resize(num_layers + num_mtp, std::vector<const void*>(num_gpus, nullptr));
    shared_expert_weight_ptrs_.resize(
        num_layers + num_mtp,
        std::vector<CommandDispatcher::Deps::SharedExpertWeights>(num_gpus));
    dense_ffn_weight_ptrs_.resize(
        num_layers,
        std::vector<CommandDispatcher::Deps::DenseFFNWeights>(num_gpus));
    // MTP-specific tensors (position-indexed like the arrays above)
    mtp_embed_tokens_ptrs_.resize(num_mtp, std::vector<const void*>(num_gpus, nullptr));
    mtp_shared_head_weight_ptrs_.resize(num_mtp, std::vector<const void*>(num_gpus, nullptr));
    mtp_shared_head_norm_ptrs_.resize(num_mtp, std::vector<const void*>(num_gpus, nullptr));
    mtp_eh_proj_ptrs_.resize(num_mtp, std::vector<const void*>(num_gpus, nullptr));
    mtp_enorm_ptrs_.resize(num_mtp, std::vector<const void*>(num_gpus, nullptr));
    mtp_hnorm_ptrs_.resize(num_mtp, std::vector<const void*>(num_gpus, nullptr));
    mtp_eh_proj_is_gguf_.resize(num_mtp, std::vector<uint8_t>(num_gpus, 0));
    mtp_eh_proj_gguf_type_.resize(
        num_mtp, std::vector<model::GgufKQuantType>(
                     num_gpus, model::GgufKQuantType::Q8_0));

    // Startup lever 3: upload ranks IN PARALLEL (each rank owns its GPU,
    // its VRAM region, and rank-indexed slots of the pointer tables resized
    // above; cudaSetDevice is per-thread). Combined with the pinned staging
    // inside, the 2x ~21 GB pageable upload drops from ~11 s to ~1-2 s.
    // Round 2c dedup: a REPLICATED k-quant embedding is wanted identically
    // by every rank — stream-dequanting it per rank duplicated ~5 s of CPU.
    // Materialize the BF16 once here (parallel_for inside), let every rank
    // upload from the shared buffer (staged), free right after the join.
    // Sharded tensors (lm_head halves) stay per-rank streamed — disjoint
    // ranges, no duplication. Arch-agnostic: host memoize + DeviceBackend.
    std::shared_ptr<std::vector<uint16_t>> embed_bf16_shared;
    if (loaded_model_->embedding &&
        loaded_model_->embedding->weight.gguf_type.has_value()) {
        if (auto* slot = upload_plan.find(model::PinnedComponent::embedding, -1)) {
            const auto& w = loaded_model_->embedding->weight;
            if (slot->size_bytes / 2 == w.numel()) {  // replicated table
                embed_bf16_shared = std::make_shared<std::vector<uint16_t>>(
                    model::dequant_kquant_range_to_bf16(w, 0, w.numel()));
            }
        }
    }

    // Round 2b free-per-tensor ring: once EVERY rank finished a layer's
    // attention upload, that layer's transform-owned host buffers (GLM-1
    // kv_b BF16, Q8_0 requants) are freed immediately — the accumulated
    // owned footprint stays at ~the in-flight window instead of all layers.
    auto layer_upload_left = std::make_unique<std::atomic<int>[]>(
        static_cast<size_t>(num_layers));
    for (int l = 0; l < num_layers; ++l)
        layer_upload_left[l].store(tp, std::memory_order_relaxed);

    auto upload_rank = [&](int rank) {
        // Resolved TD-51l: DeviceBackend::memcpy_h2d calls cudaSetDevice internally
        const int gpu_pos = tp_positions[rank];
        if (gpu_pos < 0 || gpu_pos >= num_gpus) {
            spdlog::error("upload_pinned_weights: tp_positions[{}] = {} out of range "
                          "[0, {})", rank, gpu_pos, num_gpus);
            return;
        }
        auto* pinned_base = static_cast<char*>(vram_allocator_->region(gpu_pos).pinned);
        int64_t offset = 0;

        // KD-4f-d.1b: each upload starts at 16-byte aligned offset so device
        // pointers satisfy GEMM kernel alignment requirements (NVFP4 grouped
        // GEMM scale_B_base, cuBLAS FP8 scale pointers, etc.).
        // Pinned double-buffer staging (startup lever 3): the sources are
        // pageable (mmap spans / transient shard bufs), so a plain cudaMemcpy
        // crawls at ~2 GB/s. Stage chunks through two pinned bounce buffers
        // with async H2D on a dedicated stream — the host memcpy into buffer
        // B overlaps the DMA of buffer A. Falls back to the plain synchronous
        // copy when pinned allocation fails.
        auto* dev = device_backends_[gpu_pos].get();
        constexpr size_t kStageChunk = 64ull << 20;
        void* stage_buf[2] = {dev->host_alloc_pinned(kStageChunk),
                              dev->host_alloc_pinned(kStageChunk)};
        void* stage_evt[2] = {nullptr, nullptr};
        void* stage_stream = nullptr;
        bool stage_used[2] = {false, false};
        int stage_cur = 0;
        const bool staged = stage_buf[0] && stage_buf[1];
        if (staged) {
            stage_stream = dev->create_stream();
            stage_evt[0] = dev->create_event();
            stage_evt[1] = dev->create_event();
        }
        auto stage_wait = [&](int i) {
            if (!stage_used[i]) return;
            while (dev->query_event(stage_evt[i]).status ==
                   compute::EventStatus::kNotReady) {
            }
            stage_used[i] = false;
        };
        auto upload = [&](const std::byte* src, int64_t bytes) -> void* {
            offset = (offset + 15) & ~int64_t{15};
            void* dst = pinned_base + offset;
            if (src && bytes > 0) {
                if (staged && stage_stream) {
                    int64_t done = 0;
                    while (done < bytes) {
                        const size_t n = static_cast<size_t>(
                            std::min<int64_t>(bytes - done,
                                              static_cast<int64_t>(kStageChunk)));
                        stage_wait(stage_cur);
                        std::memcpy(stage_buf[stage_cur], src + done, n);
                        dev->memcpy_h2d_async(static_cast<char*>(dst) + done,
                                              stage_buf[stage_cur], n,
                                              stage_stream);
                        dev->record_event(stage_evt[stage_cur], stage_stream);
                        stage_used[stage_cur] = true;
                        stage_cur ^= 1;
                        done += static_cast<int64_t>(n);
                    }
                } else {
                    dev->memcpy_h2d(dst, src, static_cast<size_t>(bytes));
                }
            }
            offset += bytes;
            return dst;
        };
        auto stage_finish = [&] {
            if (staged) {
                stage_wait(0);
                stage_wait(1);
            }
            if (stage_stream) dev->destroy_stream(stage_stream);
            for (int i = 0; i < 2; ++i) {
                if (stage_evt[i]) dev->destroy_event(stage_evt[i]);
                if (stage_buf[i]) dev->host_free_pinned(stage_buf[i]);
            }
        };

        // Round 2b streaming: dequant a k-quant tensor's element range to
        // BF16 chunk-by-chunk (8 MB) straight into the staged upload — no
        // full-tensor host materialization ever (kills the GG-9 ~3.6 GB
        // owned + ~3.8 GB F32-intermediate peaks). Chunk bytes are 16-byte
        // multiples so the bump allocator inserts no alignment gaps and the
        // destination stays contiguous.
        auto upload_streamed_bf16 = [&](const model::RawTensor& t,
                                        int64_t elem_off,
                                        int64_t elem_cnt) -> void* {
            constexpr int64_t kChunkElems = 4ll << 20;  // 8 MB BF16 per chunk
            void* dst0 = nullptr;
            for (int64_t done = 0; done < elem_cnt; done += kChunkElems) {
                const int64_t nchunk =
                    std::min<int64_t>(kChunkElems, elem_cnt - done);
                const auto bf16 = model::dequant_kquant_range_to_bf16(
                    t, elem_off + done, nchunk);
                void* p = upload(
                    reinterpret_cast<const std::byte*>(bf16.data()),
                    static_cast<int64_t>(bf16.size() * 2));
                if (!dst0) dst0 = p;
            }
            return dst0;
        };

        // Helper: upload attention + norms for one layer (plan-driven).
        auto upload_attention_layer = [&](
            const model::LoadedModel::LayerWeights& layer_weights,
            int attn_idx) {

            auto* attn_slot = upload_plan.find(
                model::PinnedComponent::attention, attn_idx);
            if (!attn_slot) return;

            const int64_t offset_before_attn = offset;

            auto sharded_bundles = sharder.shard_attention_layer(
                layer_weights.attention, layer_weights.indexer, rank);

            auto& attn_w = per_layer_attn_weights_[attn_idx][rank];

            for (const auto& sb : sharded_bundles) {
                // TD-73c: Norm weights/biases are F32 in checkpoint → convert to BF16.
                const bool is_norm =
                    sb.id.component == model::TensorComponent::q_a_norm ||
                    sb.id.component == model::TensorComponent::kv_a_norm ||
                    sb.id.component == model::TensorComponent::indexer_k_norm_weight ||
                    sb.id.component == model::TensorComponent::indexer_k_norm_bias ||
                    // GLM-25a: indexer_weights_proj ships F32 [hidden, n_head]; the
                    // producer consumes it via a small BF16 GEMM, so convert it to
                    // BF16 on upload alongside the norms.
                    sb.id.component == model::TensorComponent::indexer_weights_proj;
                void* ptr;
                if (is_norm && sb.weight.data.size() >= 4) {
                    const size_t n = sb.weight.data.size() / 4;
                    auto bf16 = f32_to_bf16(sb.weight.data.data(), n);
                    ptr = upload(reinterpret_cast<const std::byte*>(bf16.data()),
                                 static_cast<int64_t>(n * 2));
                } else {
                    ptr = upload(sb.weight.data.data(),
                                 static_cast<int64_t>(sb.weight.data.size()));
                }
                assign_attn_weight_ptr(attn_w, sb.id.component,
                                       model::TensorRole::weight, ptr);

                // GG-4/GG-7: thread the per-projection GGUF k-quant type into
                // the attention weights. The four plain projections (q_a/q_b/
                // kv_a/o_proj) route through the GGUF GEMM virtuals; kv_b_proj
                // (GG-7) is consumed in-kernel by q_absorb via its GGUF dequant
                // branch, so its type is threaded onto kv_b_is_gguf/kv_b_gguf_type.
                if (sb.weight.gguf_type) {
                    using TC = model::TensorComponent;
                    const model::GgufKQuantType gt = *sb.weight.gguf_type;
                    switch (sb.id.component) {
                        case TC::q_a_proj:
                            attn_w.q_a_is_gguf = true; attn_w.q_a_gguf_type = gt; break;
                        case TC::q_b_proj:
                            attn_w.q_b_is_gguf = true; attn_w.q_b_gguf_type = gt; break;
                        case TC::kv_a_proj_with_mqa:
                            attn_w.kv_a_is_gguf = true; attn_w.kv_a_gguf_type = gt; break;
                        case TC::kv_b_proj:
                            attn_w.kv_b_is_gguf = true; attn_w.kv_b_gguf_type = gt; break;
                        case TC::o_proj:
                            attn_w.o_proj_is_gguf = true; attn_w.o_proj_gguf_type = gt; break;
                        case TC::indexer_wq_b:  // GLM-25a
                            attn_w.q_idx_b_is_gguf = true; attn_w.q_idx_b_gguf_type = gt; break;
                        case TC::indexer_wk:    // GLM-25a
                            attn_w.k_idx_is_gguf = true; attn_w.k_idx_gguf_type = gt; break;
                        default: break;
                    }
                }

                for (const auto& [role, st] : sb.aux) {
                    // Capture NVFP4 scalars for o_proj from host data before
                    // upload (used at runtime for NVFP4 GEMM alpha).
                    if (sb.id.component == model::TensorComponent::o_proj
                        && st.data.size() >= sizeof(float)) {
                        float val = 0.f;
                        std::memcpy(&val, st.data.data(), sizeof(float));
                        if (role == model::TensorRole::weight_scale_2)
                            attn_w.o_proj_nvfp4_scale_2 = val;
                        else if (role == model::TensorRole::input_scale)
                            attn_w.o_proj_nvfp4_input_scale = val;
                    }
                    void* aux_ptr;
                    if (sb.id.component == model::TensorComponent::o_proj
                        && role == model::TensorRole::weight_scale
                        && st.shape.size() == 2
                        && cfg_->quantization.weights
                               == config::WeightQuant::nvfp4) {
                        // TD-GOLDEN bug #6: o_proj is consumed by the NVFP4
                        // grouped GEMM — upload Sm1xx-interleaved scales.
                        auto rf = reformat_sfb_host(st);
                        aux_ptr = upload(
                            reinterpret_cast<const std::byte*>(rf.data()),
                            static_cast<int64_t>(rf.size()));
                    } else {
                        aux_ptr = upload(st.data.data(),
                                         static_cast<int64_t>(st.data.size()));
                    }
                    assign_attn_weight_ptr(attn_w, sb.id.component, role, aux_ptr);
                }
            }

            // KD-4f-d.1b: per-tensor alignment may cause consumed > slot.
            // Use max() to never rewind into attention data.
            const int64_t slot_end = offset_before_attn + attn_slot->size_bytes;
            offset = std::max(offset, slot_end);

            auto* norm_slot = upload_plan.find(
                model::PinnedComponent::layer_norm, attn_idx);
            if (!norm_slot) return;

            for (const auto& bundle : layer_weights.norms) {
                // TD-73c: Convert F32 checkpoint norms → BF16 for RMSNorm kernel.
                const size_t num_elems = bundle.weight.data.size() / 4;
                auto bf16 = f32_to_bf16(bundle.weight.data.data(), num_elems);
                auto* ptr = upload(reinterpret_cast<const std::byte*>(bf16.data()),
                                   static_cast<int64_t>(num_elems * 2));
                if (bundle.id.component == model::TensorComponent::input_layernorm) {
                    attn_w.input_layernorm = ptr;
                } else if (bundle.id.component ==
                           model::TensorComponent::post_attention_layernorm) {
                    attn_w.post_attention_layernorm = ptr;
                }
            }
        };

        // 1. Embedding table (plan-driven)
        // TD-GOLDEN-EMB-OOB: the slot is vocab/embed_tp rows; with a
        // replicated table (embed_tp=1, slot = full table) rank>0 must read
        // offset 0, not rank*slot — that would run past the host tensor.
        if (auto* slot = upload_plan.find(model::PinnedComponent::embedding, -1);
            slot && loaded_model_->embedding) {
            const auto& w = loaded_model_->embedding->weight;
            if (embed_bf16_shared) {
                // Round 2c: replicated table, dequanted ONCE — plain staged
                // upload of the shared BF16 bytes (identical to the streamed
                // result by DequantRangeEquiv: whole == concat(chunks)).
                embedding_table_ptrs_[gpu_pos] = upload(
                    reinterpret_cast<const std::byte*>(
                        embed_bf16_shared->data()),
                    slot->size_bytes);
            } else if (w.gguf_type.has_value()) {
                // Round 2b: stream k-quant → BF16 (sharded table — element
                // math; host bytes are quant-sized).
                const int64_t total = w.numel();
                const int64_t slot_elems = slot->size_bytes / 2;
                int64_t off = static_cast<int64_t>(rank) * slot_elems;
                if (off + slot_elems > total) off = 0;
                embedding_table_ptrs_[gpu_pos] =
                    upload_streamed_bf16(w, off, slot_elems);
            } else {
                int64_t off = static_cast<int64_t>(rank) * slot->size_bytes;
                if (off + slot->size_bytes > static_cast<int64_t>(w.data.size()))
                    off = 0;  // replicated table — every rank uploads the full copy
                const auto* src = w.data.data() + off;
                auto* ptr = upload(src, slot->size_bytes);
                embedding_table_ptrs_[gpu_pos] = ptr;
            }
        }

        // 2. Output head weight + bias (plan-driven)
        if (auto* slot = upload_plan.find(model::PinnedComponent::output_head_weight, -1);
            slot && loaded_model_->output_head) {
            const auto& w = loaded_model_->output_head->weight;
            void* ptr = nullptr;
            if (w.gguf_type.has_value()) {
                const int64_t slot_elems = slot->size_bytes / 2;
                ptr = upload_streamed_bf16(
                    w, static_cast<int64_t>(rank) * slot_elems, slot_elems);
            } else {
                const auto* src = w.data.data() + rank * slot->size_bytes;
                ptr = upload(src, slot->size_bytes);
            }
            output_head_weight_ptrs_[gpu_pos] = ptr;

            if (auto* bias_slot = upload_plan.find(
                    model::PinnedComponent::output_head_bias, -1)) {
                const auto* bias_tensor = loaded_model_->output_head->find_aux(
                    model::TensorRole::bias);
                if (bias_tensor) {
                    const auto* bias_src =
                        bias_tensor->data.data() + rank * bias_slot->size_bytes;
                    auto* bias_ptr = upload(bias_src, bias_slot->size_bytes);
                    output_head_bias_ptrs_[gpu_pos] =
                        reinterpret_cast<const float*>(bias_ptr);
                }
            }
        }

        // 3. Per hidden layer
        for (int l = 0; l < num_layers; ++l) {
            const auto& layer_info = layer_registry_->layer(l);
            const auto& layer_weights = loaded_model_->layers[l];

            // 3a. Attention + layer norms (plan-driven)
            if (layer_info.attention_pinned) {
                upload_attention_layer(layer_weights, l);
                if (layer_upload_left[l].fetch_sub(
                        1, std::memory_order_acq_rel) == 1) {
                    // Last rank done with this layer — drop its owned bytes.
                    for (auto& b : loaded_model_->layers[l].attention) {
                        if (b.owned_buf) {
                            b.owned_buf.reset();
                            b.weight.data = {};
                        }
                    }
                }
            }

            // 3c. Gating weights (plan-driven, replicated)
            if (layer_info.is_moe && layer_info.gating_pinned) {
                auto* gw_slot = upload_plan.find(
                    model::PinnedComponent::gating_weight, l);
                auto* gb_slot = upload_plan.find(
                    model::PinnedComponent::gating_bias, l);
                // V4-4: hash layers carry the tid2eid I32 table instead of the
                // e_score_correction_bias (plan emits gating_hash_table).
                auto* gh_slot = upload_plan.find(
                    model::PinnedComponent::gating_hash_table, l);
                if (gw_slot) {
                    const int64_t offset_before_gating = offset;

                    for (const auto& bundle : layer_weights.gating) {
                        // GG-9: the router weight (gate_weight) must be BF16 for
                        // launch_router_projection, and the plan sizes its slot at
                        // BF16. GGUF ships ffn_gate_inp.weight as F32 — convert it
                        // here (mirrors the attention-norm F32→BF16 path) so the
                        // upload matches the slot instead of overflowing 2×. The
                        // e_score_correction_bias stays F32 (read as float).
                        void* w_ptr;
                        if (bundle.id.component ==
                                model::TensorComponent::gate_weight &&
                            bundle.weight.dtype == model::SafetensorsDtype::F32 &&
                            bundle.weight.data.size() >= 4) {
                            const size_t n = bundle.weight.data.size() / 4;
                            auto bf16 = f32_to_bf16(bundle.weight.data.data(), n);
                            w_ptr = upload(
                                reinterpret_cast<const std::byte*>(bf16.data()),
                                static_cast<int64_t>(n * 2));
                        } else {
                            w_ptr = upload(bundle.weight.data.data(),
                                static_cast<int64_t>(bundle.weight.data.size()));
                        }

                        for (const auto& [role, aux_t] : bundle.aux) {
                            upload(aux_t.data.data(),
                                   static_cast<int64_t>(aux_t.data.size()));
                        }

                        if (bundle.id.component ==
                            model::TensorComponent::gate_weight) {
                            router_weight_ptrs_[l][gpu_pos] = w_ptr;
                        }
                        if (bundle.id.component ==
                            model::TensorComponent::gate_e_score_correction_bias) {
                            gating_bias_ptrs_[l][gpu_pos] =
                                reinterpret_cast<const float*>(w_ptr);
                        }
                        if (bundle.id.component ==
                            model::TensorComponent::gate_tid2eid) {
                            hash_gating_table_ptrs_[l][gpu_pos] =
                                reinterpret_cast<const int32_t*>(w_ptr);
                        }
                    }

                    const int64_t gating_consumed = offset - offset_before_gating;
                    const int64_t expected = gw_slot->size_bytes
                        + (gb_slot ? gb_slot->size_bytes : 0)
                        + (gh_slot ? gh_slot->size_bytes : 0);
                    if (gating_consumed != expected) {
                        spdlog::error("upload_pinned_weights: gating slot mismatch "
                                      "layer {}: consumed {} vs plan {}",
                                      l, gating_consumed, expected);
                    }
                }
            }

            // 3d. Shared expert FFN (plan-driven, TP-sharded)
            if (layer_info.is_moe && layer_info.shared_expert_pinned) {
                auto sharded_se = sharder.shard_ffn_layer(layer_weights.shared_expert, rank);
                const bool nvfp4_sub = upload_plan.find(
                    model::PinnedComponent::shared_expert_gate_scales, l) != nullptr;
                auto ptrs = upload_ffn(sharded_se, nvfp4_sub, upload);

                // TODO:DEBT TD-53s: n_shared_experts > 1 overwrites single gate_up/down pair
                auto& se_w = shared_expert_weight_ptrs_[l][gpu_pos];
                se_w.gate_up = ptrs.gate_up;
                se_w.gate_up_scales = ptrs.gate_up_scales;
                se_w.down = ptrs.down;
                se_w.down_scales = ptrs.down_scales;
                se_w.weight_scale_2 = ptrs.weight_scale_2;
                se_w.weight_scale_2_down = ptrs.weight_scale_2_down;
                se_w.input_scale = ptrs.input_scale;
                se_w.input_scale_down = ptrs.input_scale_down;
                se_w.alpha = ptrs.weight_scale_2 * ptrs.input_scale;
                se_w.alpha_down = ptrs.weight_scale_2_down * ptrs.input_scale_down;
                // GG-5c: this shared expert's own per-projection GGUF types.
                se_w.gate_gguf_type = ptrs.gate_gguf_type;
                se_w.up_gguf_type   = ptrs.up_gguf_type;
                se_w.down_gguf_type = ptrs.down_gguf_type;
                se_w.gate_is_gguf = ptrs.gate_is_gguf;
                se_w.up_is_gguf   = ptrs.up_is_gguf;
                se_w.down_is_gguf = ptrs.down_is_gguf;
            }

            // 3e. Dense FFN (plan-driven, TP-sharded, non-MoE layers)
            if (!layer_info.is_moe && layer_info.ffn_pinned) {
                if (upload_plan.find(model::PinnedComponent::dense_ffn_gate, l)) {
                    auto sharded_ffn = sharder.shard_ffn_layer(
                        layer_weights.dense_ffn, rank);
                    // TD-62p: use nvfp4_sub_slots when scale sub-slots exist in plan,
                    // so gate+up weights are contiguous (required by grouped GEMM).
                    const bool nvfp4_sub = upload_plan.find(
                        model::PinnedComponent::dense_ffn_gate_scales, l) != nullptr;
                    auto ptrs = upload_ffn(sharded_ffn, nvfp4_sub, upload);

                    auto& dw = dense_ffn_weight_ptrs_[l][gpu_pos];
                    dw.gate_up = ptrs.gate_up;
                    dw.down = ptrs.down;
                    dw.gate_up_scales = ptrs.gate_up_scales;
                    dw.down_scales = ptrs.down_scales;
                    dw.weight_scale_2 = ptrs.weight_scale_2;
                    dw.weight_scale_2_down = ptrs.weight_scale_2_down;
                    dw.input_scale = ptrs.input_scale;
                    dw.input_scale_down = ptrs.input_scale_down;
                    dw.alpha = ptrs.weight_scale_2 * ptrs.input_scale;
                    dw.alpha_down = ptrs.weight_scale_2_down * ptrs.input_scale_down;
                    // GG-5c: this dense FFN's own per-projection GGUF types.
                    dw.gate_gguf_type = ptrs.gate_gguf_type;
                    dw.up_gguf_type   = ptrs.up_gguf_type;
                    dw.down_gguf_type = ptrs.down_gguf_type;
                    dw.gate_is_gguf = ptrs.gate_is_gguf;
                    dw.up_is_gguf   = ptrs.up_is_gguf;
                    dw.down_is_gguf = ptrs.down_is_gguf;
                }
            }
        }

        // 4. Final norm (plan-driven, BF16, replicated)
        if (auto* slot = upload_plan.find(model::PinnedComponent::final_norm, -1);
            slot && loaded_model_->final_norm) {
            const auto& fn = loaded_model_->final_norm.value();
            // TD-73c: F32→BF16 conversion for final_norm.
            const size_t num_elems = fn.weight.data.size() / 4;
            auto bf16 = f32_to_bf16(fn.weight.data.data(), num_elems);
            auto* ptr = upload(reinterpret_cast<const std::byte*>(bf16.data()),
                               static_cast<int64_t>(num_elems * 2));
            final_norm_ptrs_[gpu_pos] = ptr;
        }

        // 4b. output_hc — V4-5b mHC head collapse (F32, replicated).
        // fn [hc*hidden, hc] GGUF ⇒ logical [hc, hc*hidden]; base [hc]; scale [1].
        if (upload_plan.find(model::PinnedComponent::output_hc, -1) &&
            !loaded_model_->output_hc.empty()) {
            for (const auto& bundle : loaded_model_->output_hc) {
                auto* ptr = upload(bundle.weight.data.data(),
                                   static_cast<int64_t>(bundle.weight.data.size()));
                using TC = model::TensorComponent;
                switch (bundle.id.component) {
                    case TC::output_hc_fn:    output_hc_fn_ptrs_[gpu_pos] = ptr; break;
                    case TC::output_hc_base:  output_hc_base_ptrs_[gpu_pos] = ptr; break;
                    case TC::output_hc_scale: output_hc_scale_ptrs_[gpu_pos] = ptr; break;
                    default: break;
                }
            }
        }

        // 5. MTP block layers (plan-driven, full transformer block)
        for (int mi = 0; mi < num_mtp; ++mi) {
            const auto& mtp_layer = loaded_model_->mtp->block_layers[mi];
            int mtp_idx = num_layers + mi;

            // 5a. Attention + layer norms
            upload_attention_layer(mtp_layer, mtp_idx);

            // 5b. Gating weights (replicated, same pattern as regular MoE layers)
            auto* gw_slot = upload_plan.find(
                model::PinnedComponent::gating_weight, mtp_idx);
            auto* gb_slot = upload_plan.find(
                model::PinnedComponent::gating_bias, mtp_idx);
            if (gw_slot) {
                for (const auto& bundle : mtp_layer.gating) {
                    // GG-9 (mirrors the main-layer gating path above): GGUF
                    // ships ffn_gate_inp.weight as F32 but the router needs
                    // BF16 and the plan sizes the slot at gating_compute —
                    // convert when that slot is 2 B/elem or the raw F32
                    // overflows it 2×. Under gating_compute=fp32 (V3.2) the
                    // slot is F32-sized and the legacy raw upload stands.
                    // The e_score_correction_bias stays F32.
                    void* w_ptr;
                    if (bundle.id.component ==
                            model::TensorComponent::gate_weight &&
                        bundle.weight.dtype == model::SafetensorsDtype::F32 &&
                        cfg_->quantization.gating_compute !=
                            config::GatingQuant::fp32 &&
                        bundle.weight.data.size() >= 4) {
                        const size_t n = bundle.weight.data.size() / 4;
                        auto bf16 = f32_to_bf16(bundle.weight.data.data(), n);
                        w_ptr = upload(
                            reinterpret_cast<const std::byte*>(bf16.data()),
                            static_cast<int64_t>(n * 2));
                    } else {
                        w_ptr = upload(bundle.weight.data.data(),
                               static_cast<int64_t>(bundle.weight.data.size()));
                    }
                    for (const auto& [role, aux_t] : bundle.aux) {
                        upload(aux_t.data.data(),
                               static_cast<int64_t>(aux_t.data.size()));
                    }
                    if (bundle.id.component ==
                        model::TensorComponent::gate_weight) {
                        router_weight_ptrs_[mtp_idx][gpu_pos] = w_ptr;
                    }
                    if (bundle.id.component ==
                        model::TensorComponent::gate_e_score_correction_bias) {
                        gating_bias_ptrs_[mtp_idx][gpu_pos] =
                            reinterpret_cast<const float*>(w_ptr);
                    }
                }
            }

            // 5c. Shared expert FFN (TP-sharded)
            auto* se_slot = upload_plan.find(
                model::PinnedComponent::shared_expert_gate, mtp_idx);
            if (se_slot && !mtp_layer.shared_expert.empty()) {
                auto sharded_se = sharder.shard_ffn_layer(
                    mtp_layer.shared_expert, rank);
                const bool nvfp4_sub = upload_plan.find(
                    model::PinnedComponent::shared_expert_gate_scales, mtp_idx) != nullptr;
                auto ptrs = upload_ffn(sharded_se, nvfp4_sub, upload);
                auto& se_w = shared_expert_weight_ptrs_[mtp_idx][gpu_pos];
                se_w.gate_up = ptrs.gate_up;
                se_w.gate_up_scales = ptrs.gate_up_scales;
                se_w.down = ptrs.down;
                se_w.down_scales = ptrs.down_scales;
                se_w.weight_scale_2 = ptrs.weight_scale_2;
                se_w.weight_scale_2_down = ptrs.weight_scale_2_down;
                se_w.input_scale = ptrs.input_scale;
                se_w.input_scale_down = ptrs.input_scale_down;
                se_w.alpha = ptrs.weight_scale_2 * ptrs.input_scale;
                se_w.alpha_down = ptrs.weight_scale_2_down * ptrs.input_scale_down;
                // GG-5c: this MTP shared expert's own per-projection GGUF types.
                se_w.gate_gguf_type = ptrs.gate_gguf_type;
                se_w.up_gguf_type   = ptrs.up_gguf_type;
                se_w.down_gguf_type = ptrs.down_gguf_type;
                se_w.gate_is_gguf = ptrs.gate_is_gguf;
                se_w.up_is_gguf   = ptrs.up_is_gguf;
                se_w.down_is_gguf = ptrs.down_is_gguf;
            }

            // 5d. MTP-specific tensors (from mtp->tensors, TP-sharded where applicable)
            for (const auto& bundle : loaded_model_->mtp->tensors) {
                if (bundle.id.layer_idx != mtp_idx) continue;
                model::PinnedComponent comp;
                bool tp_sharded = false;
                switch (bundle.id.component) {
                    case model::TensorComponent::mtp_embed_tokens:
                        comp = model::PinnedComponent::mtp_embed_tokens;
                        tp_sharded = true; break;
                    case model::TensorComponent::mtp_shared_head_weight:
                        comp = model::PinnedComponent::mtp_shared_head_weight;
                        tp_sharded = true; break;
                    case model::TensorComponent::mtp_shared_head_norm:
                        comp = model::PinnedComponent::mtp_shared_head_norm; break;
                    case model::TensorComponent::mtp_eh_proj:
                        comp = model::PinnedComponent::mtp_eh_proj;
                        tp_sharded = true; break;
                    case model::TensorComponent::mtp_enorm:
                        comp = model::PinnedComponent::mtp_enorm; break;
                    case model::TensorComponent::mtp_hnorm:
                        comp = model::PinnedComponent::mtp_hnorm; break;
                    default: continue;
                }
                auto* slot = upload_plan.find(comp, mtp_idx);
                if (!slot) continue;

                // TD-74b: MTP norm weights need F32→BF16 conversion,
                // matching regular layer norms (TD-73c).
                const bool is_mtp_norm =
                    bundle.id.component == model::TensorComponent::mtp_shared_head_norm ||
                    bundle.id.component == model::TensorComponent::mtp_enorm ||
                    bundle.id.component == model::TensorComponent::mtp_hnorm;

                int64_t src_bytes = static_cast<int64_t>(bundle.weight.data.size());
                int64_t per_rank = tp_sharded ? src_bytes / tp : src_bytes;
                const std::byte* src = bundle.weight.data.data();
                if (tp_sharded) src += per_rank * rank;

                void* ptr;
                if (is_mtp_norm && per_rank >= 4) {
                    const size_t n = static_cast<size_t>(per_rank) / 4;
                    auto bf16 = f32_to_bf16(reinterpret_cast<const float*>(src), n);
                    ptr = upload(reinterpret_cast<const std::byte*>(bf16.data()),
                                 static_cast<int64_t>(n * 2));
                } else {
                    ptr = upload(src, per_rank);
                }

                // Store pointers
                switch (bundle.id.component) {
                    case model::TensorComponent::mtp_embed_tokens:
                        mtp_embed_tokens_ptrs_[mi][gpu_pos] = ptr; break;
                    case model::TensorComponent::mtp_shared_head_weight:
                        mtp_shared_head_weight_ptrs_[mi][gpu_pos] = ptr; break;
                    case model::TensorComponent::mtp_shared_head_norm:
                        mtp_shared_head_norm_ptrs_[mi][gpu_pos] = ptr; break;
                    case model::TensorComponent::mtp_eh_proj:
                        mtp_eh_proj_ptrs_[mi][gpu_pos] = ptr;
                        // #16: GGUF checkpoints upload eh_proj RAW (packed
                        // k-quant, Q8_0 on GLM-5.2) — record the type so the
                        // dispatcher routes the projection GEMM correctly.
                        if (bundle.weight.gguf_type) {
                            mtp_eh_proj_is_gguf_[mi][gpu_pos] = 1;
                            mtp_eh_proj_gguf_type_[mi][gpu_pos] =
                                *bundle.weight.gguf_type;
                        }
                        break;
                    case model::TensorComponent::mtp_enorm:
                        mtp_enorm_ptrs_[mi][gpu_pos] = ptr; break;
                    case model::TensorComponent::mtp_hnorm:
                        mtp_hnorm_ptrs_[mi][gpu_pos] = ptr; break;
                    default: break;
                }
            }
        }

        // Verify total consumed ≤ pinned region (overallocation is benign due
        // to alignment headroom in the layout calculation).
        const int64_t expected = vram_allocator_->layout().gpus[
            static_cast<size_t>(gpu_pos)].pinned_bytes;
        if (offset > expected) {
            // FAIL LOUD (V4-2c hardening): an upload past the pinned region
            // writes into the NEXT VRAM region (kv_speculation) — the boot
            // previously only logged this and served corrupted weights
            // (found as all-NaN layer-42 activations when the plan
            // under-sized the replicated indexer q_b at tp=2).
            throw std::runtime_error(
                "upload_pinned_weights: GPU " + std::to_string(gpu_pos)
                + " pinned-region overflow: consumed " + std::to_string(offset)
                + " > allocated " + std::to_string(expected)
                + " — plan/sharder size mismatch");
        }
            stage_finish();
    };
    {
        std::vector<std::exception_ptr> up_errs(static_cast<size_t>(tp));
        std::vector<std::thread> up_threads;
        up_threads.reserve(static_cast<size_t>(tp));
        for (int rank = 0; rank < tp; ++rank) {
            up_threads.emplace_back([&, rank] {
                try { upload_rank(rank); }
                catch (...) { up_errs[static_cast<size_t>(rank)] = std::current_exception(); }
            });
        }
        for (auto& t : up_threads) t.join();
        for (auto& e : up_errs) if (e) std::rethrow_exception(e);
    }
    embed_bf16_shared.reset();  // round 2c: transient shared table released

    spdlog::info("Uploaded pinned weights to {} TP GPUs ({} layers, {} MTP layers)",
                 tp, num_layers, num_mtp);
}

// ── Online BF16→FP8 attention weight quantization (KD-4f-c3) ───────────────

void Engine::quantize_attention_weights() {
    using WQ = config::WeightQuant;

    // Only NVFP4 checkpoints have BF16 attention projections that need online
    // quantization.  FP8 checkpoints already have the right format.
    if (!cfg_ || cfg_->quantization.weights != WQ::nvfp4) return;
    if (attention_devices_.empty() || per_layer_attn_weights_.empty()) return;
    if (backends_.skip_weight_loading || backends_.skip_hardware_detection) return;

    const int tp = std::max(1, cfg_->parallelism.tensor_parallelism);
    const auto& m = cfg_->model;

    // Projection dimensions (N = output dim / rows, K = input dim / cols).
    // q_a_proj: [q_lora_rank, hidden_size] replicated
    const int q_a_N = m.q_lora_rank;
    const int q_a_K = m.hidden_size;
    // q_b_proj: [H_local*(qk_nope+qk_rope), q_lora_rank] TP-sharded
    const int q_b_N = (m.num_attention_heads / tp) *
                      (m.qk_nope_head_dim + m.qk_rope_head_dim);
    const int q_b_K = m.q_lora_rank;
    // kv_a_proj: [kv_lora_rank+qk_rope, hidden_size] replicated.
    // Rows are zero-padded to a 128 multiple (576→640) so the runtime kv_a
    // GEMM avoids the padded-cooperative fallback; DcpExecutor computes the
    // same kv_a_n_pad_ and issues the GEMM at the padded N.
    const int kv_a_N = m.kv_lora_rank + m.qk_rope_head_dim;
    const int kv_a_N_pad = ((kv_a_N + 127) / 128) * 128;
    const int kv_a_K = m.hidden_size;

    const int num_layers = static_cast<int>(per_layer_attn_weights_.size());
    int count = 0;

    // Helper: quantize one BF16 projection to FP8 and allocate scales.
    //
    // NEVER in place (TD-GOLDEN root cause): the quant kernel re-reads each
    // input element during its write pass, and with output==input the FP8
    // byte written for element e overwrites the BF16 bytes of element e/2 —
    // other warps/blocks may not have read it yet. The block-scheduling-
    // dependent race corrupted roughly the first half of every attention
    // projection, differently on each process launch (flat-logits golden
    // failure; cross-run nondeterminism TD-100d). Instead: quantize into a
    // staging buffer, then D2D-copy the FP8 bytes back over the head of the
    // BF16 slot — steady-state VRAM is unchanged vs the old in-place design
    // (peak +N*K bytes transiently per projection during init).
    //
    // If separate=true, the staging buffer is kept as the weight (for cases
    // where the source memory must not be overwritten, e.g. a temp buffer
    // shared across layers).
    // N_pad > N zero-pads the FP8 weight rows up to N_pad (a 128 multiple) so
    // the runtime fp8_gemm takes the unpadded fast path (kv_a's N=576 would
    // otherwise hit run_padded_cooperative every layer every token: 5 sync
    // cudaMallocs, ~60 strided copies, one host cudaStreamSynchronize).
    // The padded FP8 bytes still fit in the BF16 slot (N_pad ≤ 2N).
    auto quantize_proj = [&](int rank, int N, int K,
                             const void*& weight_ptr,
                             const void*& scale_ptr,
                             bool separate = false,
                             int N_pad = 0) -> bool {
        if (!weight_ptr) return false;
        if (N_pad < N) N_pad = N;

        auto* dev = attention_devices_[rank].get();
        // ceil(N/128) == ceil(N_pad/128) since N_pad = ceil128(N): the scale
        // grid is identical and the pad rows quantize against the last real
        // n-block's scale (they are exact zeros either way).
        const int N_blocks = (N_pad + 127) / 128;
        const int K_blocks = (K + 127) / 128;
        const size_t scale_bytes = static_cast<size_t>(N_blocks) * K_blocks * sizeof(float);

        void* scale_buf = dev->device_alloc(scale_bytes);
        if (!scale_buf) {
            spdlog::error("quantize_attention_weights: device_alloc failed "
                          "for scales (N={}, K={}, rank={})", N, K, rank);
            return false;
        }

        const size_t fp8_bytes = static_cast<size_t>(N_pad) * K;
        if (!separate && fp8_bytes > static_cast<size_t>(N) * K * 2) {
            spdlog::error("quantize_attention_weights: N_pad={} overruns the "
                          "BF16 slot (N={}, K={})", N_pad, N, K);
            dev->device_free(scale_buf);
            return false;
        }
        void* fp8_buf = dev->device_alloc(fp8_bytes);
        if (!fp8_buf) {
            spdlog::error("quantize_attention_weights: device_alloc failed "
                          "(N={}, K={}, rank={})", N, K, rank);
            dev->device_free(scale_buf);
            return false;
        }

        if (N_pad > N) {
            // Zero the pad rows (FP8 0x00 == +0.0) before quantizing the real
            // rows in front of them (null-stream ordered).
            const int gpu_pos = dev->gpu().position;
            device_backends_[gpu_pos]->memset_async(fp8_buf, 0, fp8_bytes,
                                                    nullptr);
        }

        compute::WeightFp8QuantParams params{};
        params.N = N;
        params.K = K;
        params.input = weight_ptr;
        params.output = fp8_buf;
        params.scales = scale_buf;
        dev->weight_quantize_fp8(params, nullptr);

        if (separate) {
            weight_ptr = fp8_buf;
        } else {
            // Copy the FP8 result over the BF16 slot (null-stream ordered
            // after the quant kernel; memcpy_d2d is synchronous), then free
            // the staging buffer.
            const int gpu_pos = dev->gpu().position;
            device_backends_[gpu_pos]->memcpy_d2d(
                const_cast<void*>(weight_ptr), fp8_buf, fp8_bytes);
            dev->device_free(fp8_buf);
        }
        scale_ptr = scale_buf;

        // Track for shutdown cleanup.  Pinned-slot conversions have
        // fp8_weight=nullptr (memory owned by the pinned region).
        quantized_attn_allocs_.push_back(
            {separate ? fp8_buf : nullptr, scale_buf, rank});
        ++count;
        return true;
    };

    const int num_hidden = static_cast<int>(m.num_hidden_layers);

    for (int rank = 0; rank < tp; ++rank) {
        auto* dev = attention_devices_[rank].get();
        dev->set_device();

        // q_a, q_b, kv_a: always BF16 in NVFP4 checkpoints → FP8 (in-place).
        for (int l = 0; l < num_layers; ++l) {
            auto& w = per_layer_attn_weights_[l][rank];
            quantize_proj(rank, q_a_N, q_a_K, w.q_a_proj, w.q_a_proj_scales);
            quantize_proj(rank, q_b_N, q_b_K, w.q_b_proj, w.q_b_proj_scales);
            if (quantize_proj(rank, kv_a_N, kv_a_K, w.kv_a_proj,
                              w.kv_a_proj_scales, /*separate=*/false,
                              kv_a_N_pad)) {
                // Tell the executor the rows are padded: the GEMM runs at
                // kv_a_N_pad on the unpadded fast path.
                w.kv_a_n_padded = kv_a_N_pad;
            }
        }

        // o_proj: regular layers stay NVFP4 (used via NVFP4 grouped GEMM at
        // runtime), MTP layers are BF16 → FP8 in-place.
        for (int l = 0; l < num_layers; ++l) {
            auto& w = per_layer_attn_weights_[l][rank];
            if (!w.o_proj) continue;

            const bool is_mtp = (l >= num_hidden);
            if (!is_mtp) {
                // Regular layer: o_proj is NVFP4 — mark for runtime dispatch.
                // FP4-ACT-SCALE (TRT-LLM scheme): the quantizer divides by the
                // calibrated input_scale (group scale = amax/(6*is), lifting
                // the tiny kv_bv groups out of the UE4M3 2^-9 clamp), so
                // alpha = weight_scale_2 * input_scale multiplies it back.
                const float o_is = (w.o_proj_nvfp4_input_scale > 0.f)
                                       ? w.o_proj_nvfp4_input_scale : 1.0f;
                w.o_proj_is_nvfp4 = true;
                w.o_proj_nvfp4_alpha = w.o_proj_nvfp4_scale_2 * o_is;
                if (w.o_proj_nvfp4_alpha == 0.f) {
                    spdlog::warn("quantize_attention_weights: o_proj alpha=0 "
                                 "for layer {} rank {} (weight_scale_2={} "
                                 "input_scale={})",
                                 l, rank, w.o_proj_nvfp4_scale_2,
                                 w.o_proj_nvfp4_input_scale);
                }
            } else {
                // MTP layer: o_proj is BF16 → FP8 in-place.
                const int o_N = m.hidden_size;
                const int o_K = (m.num_attention_heads / tp) * m.v_head_dim;
                quantize_proj(rank, o_N, o_K, w.o_proj, w.o_proj_scales);
            }
        }

        dev->device_sync();
    }

    spdlog::info("quantize_attention_weights: quantized {} projections "
                 "across {} ranks ({} layers)",
                 count, tp, num_layers);
}

// ── kv_b_v dequant pool wiring (TD-87a) ────────────────────────────────────

void Engine::wire_kv_bv_dequant() {
    if (!dcp_executor_ || per_layer_attn_weights_.empty()) return;

    const int tp = std::max(1, cfg_->parallelism.tensor_parallelism);
    const int num_layers = static_cast<int>(per_layer_attn_weights_.size());

    // Set kv_b_proj_is_fp8 flag: true when scales are present (FP8 checkpoint
    // or post-online-quantization). For NVFP4 checkpoints, kv_b_proj stays
    // BF16 (not quantized in quantize_attention_weights), so flag stays false.
    for (int l = 0; l < num_layers; ++l) {
        for (int r = 0; r < tp; ++r) {
            auto& w = per_layer_attn_weights_[l][r];
            w.kv_b_proj_is_fp8 = (w.kv_b_proj_scales != nullptr);
        }
    }

    // Build per-layer weight pointer array for DcpExecutor's predictive scheduling.
    std::vector<std::vector<const parallelism::AttentionLayerWeights*>> all_weights(
        num_layers, std::vector<const parallelism::AttentionLayerWeights*>(tp));
    for (int l = 0; l < num_layers; ++l) {
        for (int r = 0; r < tp; ++r) {
            all_weights[l][r] = &per_layer_attn_weights_[l][r];
        }
    }

    dcp_executor_->set_layer_weights(std::move(all_weights), num_layers);
    dcp_executor_->prime_dequant_pool();

    spdlog::info("wire_kv_bv_dequant: {} layers, kv_b_proj_is_fp8={}",
                 num_layers,
                 num_layers > 0 && per_layer_attn_weights_[0][0].kv_b_proj_is_fp8);
}

// ── IPC region allocation ───────────────────────────────────────────────────

void Engine::allocate_ipc_region() {
    constexpr uint32_t cmd_slots = ipc::kDefaultCmdRingSlots;
    constexpr uint32_t cmp_slots = ipc::kDefaultCmpRingSlots;

    ipc_total_bytes_ = ipc::IpcLayout::total_size(cmd_slots, cmp_slots);

    // TD-14d: heap-allocate with 64-byte alignment for cache-line isolation,
    // then CUDA-pin it below (compute.ipc_pin, default ON) so sideband D2H
    // readbacks are true-async DMA.
    void* raw = std::aligned_alloc(64, ipc_total_bytes_);
    if (!raw) {
        throw std::runtime_error("Failed to allocate IPC region ("
                                 + std::to_string(ipc_total_bytes_) + " bytes)");
    }
    ipc_region_.reset(static_cast<uint8_t*>(raw));
    std::memset(ipc_region_.get(), 0, ipc_total_bytes_);

    // compute.ipc_pin (DSP52_BOOST lever 2; TD-14d): cudaHostRegister the IPC
    // region so every memcpy_d2h_async into the sideband is a TRUE async DMA.
    // Unpinned, a pageable D2H blocks the DAEMON THREAD until the source
    // stream drains — measured 85 ms per D_CMD_RUN_DSPARK_STEP readback (the
    // whole draft pipeline queued ahead of it), which serializes all command
    // dispatch and defeats draft/target command-level overlap. Registration is
    // one-shot at init (driver-lock trap only bites when concurrent with other
    // CUDA work); failure degrades to the unpinned path, loudly. The region is
    // a private ~4.9 MB aligned_alloc mapping — disjoint from the pinned
    // expert arena and every NUMA DMA buffer, so it cannot partial-overlap
    // another registration (P-24b trap is arena-side only).
    //
    // Config default TRUE; env LS_IPC_PIN overrides EITHER way (set & !='0' →
    // on, '0' → off, unset defers to config — the
    // LAYERSTORM_DETERMINISTIC_EP_COMBINE precedence pattern). Skipped for
    // null/CPU-only backends (no CUDA context to pin against).
    bool ipc_pin = cfg_ ? cfg_->compute.ipc_pin : true;
    if (const char* e = std::getenv("LS_IPC_PIN"); e && *e)
        ipc_pin = (e[0] != '0');
    if (ipc_pin && !backends_.skip_hardware_detection) {
        const int rc =
            core::host_register_pinned_portable(ipc_region_.get(),
                                                ipc_total_bytes_);
        ipc_region_registered_ = (rc == 0);
        if (ipc_region_registered_)
            spdlog::info("IPC region cudaHostRegistered ({} bytes) — sideband "
                         "D2H readbacks are true-async (compute.ipc_pin)",
                         ipc_total_bytes_);
        else
            spdlog::warn("compute.ipc_pin: host_register failed (err {}) — "
                         "sideband D2H readbacks stay pageable/blocking", rc);
    } else {
        spdlog::info("IPC region NOT pinned (compute.ipc_pin={}, {}) — "
                     "sideband D2H readbacks are pageable/blocking",
                     ipc_pin,
                     backends_.skip_hardware_detection ? "null backends"
                                                       : "disabled");
    }

    uint8_t* base = ipc_region_.get();

    // Init IpcHeader
    ipc_header_ = reinterpret_cast<ipc::IpcHeader*>(base + ipc::IpcLayout::kHeaderOffset);
    ipc_header_->version = ipc::kProtocolVersion;
    ipc_header_->shutdown_requested = 0;
    ipc_header_->error_code = 0;

    // Init command ring
    void* cmd_ring_ptr = base + ipc::IpcLayout::cmd_ring_offset();
    ipc::CommandRing::init(cmd_ring_ptr, cmd_slots);
    cmd_ring_ = std::make_unique<ipc::CommandRing>(cmd_ring_ptr);

    // Init completion ring
    void* cmp_ring_ptr = base + ipc::IpcLayout::cmp_ring_offset(cmd_slots);
    ipc::CompletionRing::init(cmp_ring_ptr, cmp_slots);
    cmp_ring_ = std::make_unique<ipc::CompletionRing>(cmp_ring_ptr);

    // Init state snapshot
    state_snapshot_ = reinterpret_cast<ipc::StateSnapshot*>(
        base + ipc::IpcLayout::state_offset(cmd_slots, cmp_slots));
    state_snapshot_->seqlock = 0;  // Even = readable
    state_snapshot_->num_gpus = static_cast<uint32_t>(cfg_->hardware.gpus.size());

    // Populate EngineInfo
    engine_info_.ipc_base        = reinterpret_cast<uintptr_t>(base);
    engine_info_.ipc_total_bytes = ipc_total_bytes_;
    engine_info_.cmd_ring_offset = ipc::IpcLayout::cmd_ring_offset();
    engine_info_.cmd_ring_slots  = cmd_slots;
    engine_info_.cmd_slot_bytes  = ipc::kCmdSlotBytes;
    engine_info_.cmp_ring_offset = ipc::IpcLayout::cmp_ring_offset(cmd_slots);
    engine_info_.cmp_ring_slots  = cmp_slots;
    engine_info_.cmp_slot_bytes  = ipc::kCmpSlotBytes;
    engine_info_.state_offset    = ipc::IpcLayout::state_offset(cmd_slots, cmp_slots);
    engine_info_.state_bytes     = sizeof(ipc::StateSnapshot);
    engine_info_.sideband_offset = ipc::IpcLayout::sideband_offset(cmd_slots, cmp_slots);
    engine_info_.sideband_bytes  = ipc::IpcLayout::kSidebandTotalSize;
    engine_info_.num_gpus        = static_cast<int32_t>(cfg_->hardware.gpus.size());
    engine_info_.num_moe_layers  = static_cast<int32_t>(model_cfg_->num_moe_layers());
    engine_info_.num_experts     = static_cast<int32_t>(cfg_->model.n_routed_experts);
    engine_info_.expert_bytes    = layer_registry_->per_routed_expert_bytes();
    engine_info_.num_layers      = static_cast<int32_t>(cfg_->model.num_hidden_layers);
    engine_info_.num_expert_devices = static_cast<int32_t>(expert_devices_.size());
    engine_info_.kv_bytes_per_page = vram_allocator_
        ? vram_allocator_->layout().kv_bytes_per_page : 0;
    // TD-VOCAB-AUTODETECT: post-resolve value (weights-derived or
    // cross-checked in init_modules Step 1b).
    engine_info_.vocab_size      = static_cast<int32_t>(cfg_->model.vocab_size);

    // V4-7a: DeepSeek-V4 metadata (zero for non-V4 models — engine_info_ is
    // value-initialized). Per-layer attention types come from compress_ratios.
    if (model_cfg_->is_v4()) {
        engine_info_.v4_hc_mult = static_cast<int32_t>(cfg_->model.hc_mult);
        engine_info_.v4_num_hash_layers =
            static_cast<int32_t>(cfg_->model.num_hash_layers);
        const int nl = std::min<int>(cfg_->model.num_hidden_layers,
                                     ipc::EngineInfo::kV4MaxLayers);
        for (int l = 0; l < nl; ++l) {
            switch (model_cfg_->attention_type_for_layer(l)) {
                case model::V4AttentionType::kSwa:
                    engine_info_.v4_attention_types[l] = 0; break;
                case model::V4AttentionType::kCsa:
                    engine_info_.v4_attention_types[l] = 1; break;
                case model::V4AttentionType::kHca:
                    engine_info_.v4_attention_types[l] = 2; break;
            }
        }
    }
}

// ── Daemon thread ───────────────────────────────────────────────────────────

void Engine::spawn_daemon_thread() {
    // ELM-8: Create ExpertLifecycleManager (first instantiation in Engine).
    {
        ExpertLifecycleManager::Deps elm_deps{
            .expert_cache    = expert_cache_.get(),
            .nvme_tier       = nvme_tier_.get(),
            .transfer_engine = transfer_engine_.get(),
            .numa_manager    = numa_manager_.get(),
            .pinned_arena    = pinned_arena_.get(),
            .arena_loader    = arena_loader_.get(),   // J-1 async cold load
            .loaded_model    = loaded_model_.get(),
            .prepacked_source = prepacked_source_.get(),
            .packed_cache    = packed_cache_.get(),
            .snapshot        = state_snapshot_,
            .first_moe_layer = static_cast<uint32_t>(cfg_->model.first_k_dense_replace),
            .num_moe_layers  = static_cast<uint32_t>(model_cfg_->num_moe_layers()),
            .num_experts     = static_cast<uint32_t>(cfg_->model.n_routed_experts),
            .num_gpus        = static_cast<int>(cfg_->hardware.gpus.size()),
            .expert_shape    = model::ExpertShape{cfg_->model.hidden_size,
                                                  cfg_->model.moe_intermediate_size},
        };
        elm_ = std::make_unique<ExpertLifecycleManager>(std::move(elm_deps));
    }

    // KD-2: Build device pointer vectors for CommandDispatcher.
    const auto num_gpus = cfg_->hardware.gpus.size();
    std::vector<compute::AttentionDevice*> attn_ptrs_for_disp(num_gpus, nullptr);
    for (auto& dev : attention_devices_) {
        attn_ptrs_for_disp[dev->gpu().position] = dev.get();
    }
    std::vector<compute::ExpertDevice*> expert_ptrs_for_disp;
    expert_ptrs_for_disp.reserve(expert_devices_.size());
    for (auto& dev : expert_devices_) {
        expert_ptrs_for_disp.push_back(dev.get());
    }

    // GG-5b: resolve the per-projection GGUF k-quant interface for the fused MoE
    // GEMM path (dense/routed/shared). For the generic `gguf` enum quant_ points
    // at the owned mixed interface; for uniform `gguf_qX_k` it points at the
    // registry singleton (also a GgufQuantInterface). nullptr for non-GGUF.
    const model::GgufQuantInterface* gguf_quant_for_disp = nullptr;
    if (model::gguf::is_gguf_weight_quant(cfg_->quantization.weights)) {
        if (owned_gguf_quant_.has_value()) {
            gguf_quant_for_disp = &*owned_gguf_quant_;
        } else {
            gguf_quant_for_disp =
                dynamic_cast<const model::GgufQuantInterface*>(quant_);
        }
    }

    // GG-9: per-layer routed-expert GGUF k-quant types. A mixed "XL" GGUF stacks
    // each layer's experts with ONE type per projection, but that type DIFFERS
    // across layers (e.g. gate Q4_K/Q5_K/Q6_K) — the uniform `gguf_quant` would
    // decode most layers wrong. Capture each layer's own types from the loaded
    // model so dispatch_moe selects the right decode per layer.
    // GG-10: in prepacked mode (WP-6 skip_routed_experts) the routed bundles
    // are NOT loaded, so the types come from the manifest's per-layer block
    // (gguf_types_per_layer) instead — the manifest records the k-quant triple
    // each layer's slots were actually packed at. Without this overlay the
    // entries would stay at their Q4_K defaults and every layer would decode
    // with wrong types AND wrong in-slot up/down offsets.
    std::vector<model::GgufModelExpertTypes> routed_layer_gguf_types;
    if (gguf_quant_for_disp && loaded_model_) {
        const size_t nl = loaded_model_->layers.size();
        routed_layer_gguf_types.resize(nl);
        for (size_t l = 0; l < nl; ++l) {
            for (const auto& experts : loaded_model_->layers[l].routed_experts) {
                for (const auto& b : experts) {
                    auto gt = b.gguf_type();
                    if (!gt) continue;
                    switch (b.id.component) {
                        case model::TensorComponent::gate_proj:
                            routed_layer_gguf_types[l].gate = *gt; break;
                        case model::TensorComponent::up_proj:
                            routed_layer_gguf_types[l].up = *gt; break;
                        case model::TensorComponent::down_proj:
                            routed_layer_gguf_types[l].down = *gt; break;
                        default: break;
                    }
                }
            }
        }
        if (prepacked_source_) {
            for (size_t l = 0; l < nl; ++l) {
                // Only layers with no loaded routed bundles (WP-6 skip) take
                // their types from the manifest; loaded bundles stay the
                // source of truth in legacy/mmap mode.
                if (!loaded_model_->layers[l].routed_experts.empty()) continue;
                auto t = prepacked_source_->gguf_types_for_layer(
                    static_cast<int>(l));
                if (!t) continue;
                routed_layer_gguf_types[l] =
                    model::GgufModelExpertTypes{t->gate, t->up, t->down};
            }
        }
    }

    // TD-PREFILL-MOE-BIG scratch homing: hand the dispatcher the per-GPU
    // prefill-scratch tail spans — the RESERVED tail of each kv_main region
    // (prefill_scratch_preallocated_bytes; TP GPUs only, {nullptr,0}
    // elsewhere). KV pages exclude it (vram_allocator kv_main_for_pages) and
    // no production consumer touches it (the attention devices self-alloc
    // their prefill scratch; ExpertCache spill mode — test-only — uses the
    // ADJACENT streaming spill zone). The dispatcher homes the chunk-bounded
    // MoE transient buffers there under prefill_moe_big.
    std::vector<std::pair<void*, size_t>> prefill_scratch_tails;
    if (vram_allocator_ && vram_allocator_->owns_memory()
        && cfg_->compute.prefill_moe_big) {
        const auto& lay = vram_allocator_->layout();
        for (int g = 0; g < vram_allocator_->gpu_count(); ++g) {
            const auto& region = vram_allocator_->region(g);
            const auto& gl = lay.gpus[static_cast<size_t>(g)];
            const int64_t tail = gl.prefill_scratch_preallocated_bytes;
            if (region.kv_main && tail > 0 && gl.kv_main_bytes >= tail) {
                prefill_scratch_tails.emplace_back(
                    static_cast<char*>(region.kv_main)
                        + (gl.kv_main_bytes - tail),
                    static_cast<size_t>(tail));
            } else {
                prefill_scratch_tails.emplace_back(nullptr, 0);
            }
        }
    }

    // TODO:DEBT TD-14p: No engine-level integration test covering full init→dispatch→shutdown
    // Construct command dispatcher with all module pointers.
    // KD-3a KV strides for the dispatcher. V4 (ticket E): the legacy
    // uniform kv_bytes_per_token THROWS for deepseek_v4 (V4-3b) — feed the
    // kMain(=CSA) bucket geometry from VramLayout.v4 instead. Consumers:
    // AttentionExecParams (V4 attention routing itself is fail-closed until
    // V4-7b), seq snapshot/restore page extents, KV tiering (V4 formats
    // already excluded at the tiering gate).
    int64_t kv_stride_block_deps = 0;
    int kv_stride_row_deps = 0;
    int kv_page_size_deps = cfg_->memory.kv_cache.page_size_tokens;
    if (model_cfg_->is_v4()) {
        const auto& v4l = vram_allocator_->layout().v4;
        kv_stride_block_deps = v4l.csa_bytes_per_page;
        kv_stride_row_deps = static_cast<int>(v4l.csa_entry_bytes);
        // V4-7b (ticket H): the dispatcher's page machinery is TOKEN-granular
        // (seq_pages_ growth, block tables, slot mappings) — one kMain page
        // per LOGICAL BLOCK of 256 native tokens (= exactly one 64-entry CSA
        // page, ticket-C geometry). The earlier csa_entries_per_page value
        // (64) was an entry-unit placeholder that would have grown pages 4×
        // too fast and broken the bt[e/64] entry-slot math.
        kv_page_size_deps = v4l.logical_block_tokens;
    } else {
        const int64_t per_tok = memory::kv_bytes_per_token(
            *model_cfg_, cfg_->quantization.kv_cache,
            cfg_->compute.attention_backend);
        kv_stride_block_deps =
            per_tok * cfg_->memory.kv_cache.page_size_tokens;
        kv_stride_row_deps = static_cast<int>(per_tok);
    }

    CommandDispatcher::Deps disp_deps{
        .cmp_ring           = cmp_ring_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .expert_cache       = expert_cache_.get(),
        .stream_manager     = stream_manager_.get(),
        .graph_registry     = graph_registry_.get(),
        .coactivation_graph = coactivation_graph_.get(),
        .expert_stats       = expert_stats_.get(),
        .nvme_tier          = nvme_tier_.get(),
        .numa_manager       = numa_manager_.get(),
        .pinned_arena       = pinned_arena_.get(),
        .loaded_model       = loaded_model_.get(),
        .prepacked_source   = prepacked_source_.get(),
        .packed_cache       = packed_cache_.get(),
        .expert_shape       = model::ExpertShape{cfg_->model.hidden_size,
                                                 cfg_->model.moe_intermediate_size},
        .dcp_executor       = dcp_executor_.get(),
        .dcp_communicator   = dcp_communicator_.get(),
        .buffer_registry    = buffer_registry_.get(),
        .page_allocator     = page_allocator_.get(),
        .sideband_base      = ipc_region_.get()
                              + ipc::IpcLayout::sideband_offset(
                                    ipc::kDefaultCmdRingSlots,
                                    ipc::kDefaultCmpRingSlots),
        .elm                = elm_.get(),
        .live_config        = cfg_.get(),
        .gguf_quant         = gguf_quant_for_disp,  // GG-5b: per-projection GGUF type
        .routed_layer_gguf_types = std::move(routed_layer_gguf_types),  // GG-9
        .loader_constants   = &loader_constants_,  // I8: shadow/route cost constants
        .attention_devices  = std::move(attn_ptrs_for_disp),
        .expert_devices     = std::move(expert_ptrs_for_disp),
        .device_backends    = [&]{
            std::vector<compute::DeviceBackend*> v;
            v.reserve(device_backends_.size());
            for (auto& db : device_backends_) v.push_back(db.get());
            return v;
        }(),
        .cuda_kernels_enabled = !backends_.skip_hardware_detection,
        .embedding_table_ptrs    = embedding_table_ptrs_,
        .output_head_weight_ptrs = output_head_weight_ptrs_,
        .output_head_bias_ptrs   = output_head_bias_ptrs_,
        .gating_bias_ptrs        = std::move(gating_bias_ptrs_),
        // V4-4: hash-layer routing tables + count (0 disables the branch).
        .hash_gating_table_ptrs  = std::move(hash_gating_table_ptrs_),
        .moe_hash_layers         = model_cfg_->is_v4()
                                       ? cfg_->model.num_hash_layers : 0,
        .kv_cache_stride_block = kv_stride_block_deps,
        .kv_cache_stride_row = kv_stride_row_deps,
        .kv_page_size = kv_page_size_deps,
        .hidden_state_pairs      = hidden_state_pairs_,
        .fused_moe_hidden_states = fused_moe_hidden_state_bufs_,
        .per_layer_attn_weights  = std::move(per_layer_attn_weights_),
        .max_batch_size          = cfg_->orchestrator.max_batch_size,
        .superchunk_tokens       = cfg_->compute.prefill_superchunk_tokens,
        .prefill_scratch_tails   = std::move(prefill_scratch_tails),
        .router_weight_ptrs          = std::move(router_weight_ptrs_),
        .shared_expert_weight_ptrs   = std::move(shared_expert_weight_ptrs_),
        .dense_ffn_weight_ptrs       = std::move(dense_ffn_weight_ptrs_),
        .final_norm_ptrs             = final_norm_ptrs_,
        .hc_streams                  = model_cfg_->has_mhc() ? cfg_->model.hc_mult : 1,
        .output_hc_fn_ptrs           = output_hc_fn_ptrs_,
        .output_hc_base_ptrs         = output_hc_base_ptrs_,
        .output_hc_scale_ptrs        = output_hc_scale_ptrs_,
        .hc_attn_x                   = hc_attn_x_,
        .hc_attn_post                = hc_attn_post_,
        .hc_attn_comb                = hc_attn_comb_,
        .mtp_embed_tokens_ptrs       = std::move(mtp_embed_tokens_ptrs_),
        .mtp_shared_head_weight_ptrs = std::move(mtp_shared_head_weight_ptrs_),
        .mtp_shared_head_norm_ptrs   = std::move(mtp_shared_head_norm_ptrs_),
        .mtp_eh_proj_ptrs            = std::move(mtp_eh_proj_ptrs_),
        .mtp_enorm_ptrs              = std::move(mtp_enorm_ptrs_),
        .mtp_hnorm_ptrs              = std::move(mtp_hnorm_ptrs_),
        .mtp_eh_proj_is_gguf         = std::move(mtp_eh_proj_is_gguf_),
        .mtp_eh_proj_gguf_type       = std::move(mtp_eh_proj_gguf_type_),
        .dspark                      = dspark_runtime_.get(),  // DSP-3
    };
    command_dispatcher_ = std::make_unique<CommandDispatcher>(std::move(disp_deps));
    // TD-PREFILL-SUPERCHUNK: publish the EFFECTIVE MoE batch capacity (the
    // dispatcher's VRAM fail-safe may have stepped the request down).
    engine_info_.moe_batch_capacity = command_dispatcher_->moe_batch_capacity();

    // IPC-5: State publisher
    state_publisher_ = std::make_unique<StatePublisher>(StatePublisher::Deps{
        .expert_stats       = expert_stats_.get(),
        .workload_detector  = workload_detector_.get(),
        .acceptance_tracker = acceptance_tracker_.get(),
        .expert_cache       = expert_cache_.get(),
        .vram_allocator     = vram_allocator_.get(),
        .page_allocator     = page_allocator_.get(),
        .transfer_engine    = transfer_engine_.get(),
        .pending_compute_per_gpu = [this](uint32_t gpu_idx) -> uint32_t {
            return command_dispatcher_->pending_compute_count(gpu_idx);
        },
    });

    // M3b: online self-tuning arena placement. Source of truth = config
    // memory.arena_placement.online (schema default TRUE, 2026-08-18
    // champion decision); env LS_ARENA_PLACE_ONLINE OVERRIDES: non-'0'
    // forces ON, '0' forces OFF, unset defers to the config. Live
    // demand-fetch EMA (ELM fresh-enqueue observer) + background HBM
    // promote/demote migration ticked from the daemon loop. Composes with
    // the static freq table (bootstrap layout, then the online signal
    // takes over). LOUD no-op warn when the arena path is inactive
    // (mirrors the static flag's discipline — a silent no-op poisons A/Bs).
    bool place_online = cfg_->memory.arena_placement.online;
    if (const char* on = std::getenv("LS_ARENA_PLACE_ONLINE"); on && *on)
        place_online = (*on != '0');
    if (place_online) {
        if (pinned_arena_ && elm_) {
            std::vector<int> hbm;
            for (int n : pinned_arena_->arena_nodes())
                if (numa_manager_ && numa_manager_->node_is_hbm(n))
                    hbm.push_back(n);
            if (hbm.empty()) {
                spdlog::warn(
                    "arena_placement.online engaged but no HBM arena nodes exist "
                    "— online placement INACTIVE (nothing to promote into)");
            } else {
                const auto nodes = pinned_arena_->arena_nodes();
                const size_t slot_bytes =
                    pinned_arena_->node_arena(nodes.front())->slot_size();
                const auto mcfg = memory::ArenaMigrator::Config::from_env();
                arena_migrator_ = std::make_unique<memory::ArenaMigrator>(
                    mcfg, *pinned_arena_, hbm,
                    static_cast<uint32_t>(cfg_->model.num_hidden_layers),
                    static_cast<uint32_t>(cfg_->model.n_routed_experts),
                    slot_bytes);
                elm_->set_h2d_enqueue_observer(
                    [m = arena_migrator_.get()](memory::ExpertKey k) {
                        m->on_demand_fetch(k);
                    });
                // TD-ARENA-MIGRATE-EMA-PERSIST: adopt the persisted EMA
                // from the holder-store meta trailer (as-is, warm-up gate
                // restored — see ArenaMigrator::adopt_ema) and arm the
                // periodic persist sink. Both run on the daemon thread
                // (single writer); the final snapshot is taken in
                // shutdown() after the daemon joins.
                if (arena_cache_) {
                    const auto nl = static_cast<uint32_t>(
                        cfg_->model.num_hidden_layers);
                    const auto ne = static_cast<uint32_t>(
                        cfg_->model.n_routed_experts);
                    if (const auto v = arena_cache_->ema_load(nl, ne)) {
                        arena_migrator_->adopt_ema(
                            v->data, static_cast<size_t>(nl) * ne,
                            v->fetches_seen);
                        const auto now_ns = static_cast<uint64_t>(
                            std::chrono::duration_cast<
                                std::chrono::nanoseconds>(
                                std::chrono::system_clock::now()
                                    .time_since_epoch()).count());
                        spdlog::info(
                            "arena_migrate: adopted persisted EMA "
                            "(fetches_seen={:.0f}, ema_total={:.0f}, "
                            "age={:.0f} s)",
                            v->fetches_seen,
                            arena_migrator_->stats().ema_total,
                            now_ns > v->saved_unix_ns
                                ? (now_ns - v->saved_unix_ns) / 1e9
                                : 0.0);
                    }
                    ema_persist_fn_ = [this, nl, ne] {
                        if (arena_cache_ && arena_migrator_)
                            arena_cache_->ema_save(
                                arena_migrator_->freq_table().data(), nl,
                                ne, arena_migrator_->fetches_seen());
                    };
                    arena_migrator_->set_persist_sink(ema_persist_fn_);
                }
                spdlog::info(
                    "arena placement ONLINE (config-default; LS_ARENA_PLACE_ONLINE overrides): ENGAGED "
                    "— {} HBM node(s), budget {:.0f} MB/s, half-life {:.0f} s, "
                    "ratio {}, margin {}, slot {:.1f} MiB",
                    hbm.size(), mcfg.budget_mbps, mcfg.half_life_s, mcfg.ratio,
                    mcfg.margin, slot_bytes / 1048576.0);
            }
        } else {
            spdlog::warn(
                "arena_placement.online engaged but the pinned arena path is "
                "inactive — online placement NOT applied");
        }
    }

    DaemonLoop::Deps deps{
        .cmd_ring        = cmd_ring_.get(),
        .cmp_ring        = cmp_ring_.get(),
        .ipc_header      = ipc_header_,
        .state_snapshot  = state_snapshot_,
        .running         = &running_,
        .transfer_engine = transfer_engine_.get(),
        .nvme_tier       = nvme_tier_.get(),
        .dispatch_fn     = [this](const ipc::Command& cmd) {
            command_dispatcher_->dispatch(cmd);
        },
        .publish_fn      = [this](ipc::StateSnapshot& snap,
                                   daemon::StateTransaction& tx) {
            state_publisher_->publish(snap, tx);
        },
        .cmd_seq_resolver = [this](uint64_t token) {
            return command_dispatcher_->resolve_cmd_seq(token);
        },
        .token_cleanup    = [this](uint64_t token) {
            command_dispatcher_->remove_token_mapping(token);
        },
        .nvme_cmd_seq_resolver = [this](uint64_t token) {
            return command_dispatcher_->resolve_nvme_cmd_seq(token);
        },
        .nvme_token_cleanup    = [this](uint64_t token) {
            command_dispatcher_->remove_nvme_token_mapping(token);
        },
        .elm                   = elm_.get(),
        .writer_yield_interval = cfg_->orchestrator.ipc_transaction.writer_yield_interval,
        // #91 / INV-IPC-PUBLISH-THROTTLE: keep the seqlock quiet between
        // publishes so Python StateReader transactions can validate.
        .publish_interval_ns =
            static_cast<uint64_t>(
                cfg_->orchestrator.ipc_transaction.publish_interval_us)
            * 1000ULL,
        .poll_compute_fn       = [this]() -> uint32_t {
            return command_dispatcher_->poll_compute_completions();
        },
        .pending_compute_count_fn = [this]() -> uint32_t {
            return command_dispatcher_->pending_compute_count();
        },
        // TODO:DEBT TD-13: max_inflight_compute hardcoded to 32, should come from config
        .max_inflight_compute  = 32,
        .advance_progressive_fn = [this]() -> bool {
            return command_dispatcher_->advance_progressive_moe();
        },
        // M3b: arena migrator tick (near-free when idle; nullable).
        .background_fn = arena_migrator_
            ? DaemonLoop::Deps::BackgroundFn(
                  [m = arena_migrator_.get()] { m->tick(); })
            : DaemonLoop::Deps::BackgroundFn{},
    };
    daemon_loop_impl_ = std::make_unique<DaemonLoop>(std::move(deps));

    // Queued FAR/FETCH backpressure: the in-handler progressive-MoE drain
    // needs the loop's completion pump to make arrival progress (arrivals
    // materialize ONLY in pump_completions). Same thread — the handler runs
    // inside the loop's drain phase.
    command_dispatcher_->set_lifecycle_pump(
        [this] { daemon_loop_impl_->pump_completions(); });

    running_.store(true, std::memory_order_release);

    // C-6 QA (d): pin the daemon thread to the LEADING physical core of the
    // CPU-expert node — the exact core the NumaThreadPool holds OUT of the FFN
    // pool (reserve_leading_cores) — so the orchestrator/daemon runs there
    // undisturbed and the host FFN workers (node's 2nd core onward) never steal
    // its cycles. DEFAULT ON, but CHAMPION-SAFE: it engages ONLY when an enabled
    // CPU expert device exists (cpu_node >= 0 ⇒ a NumaThreadPool was built). The
    // pure champion (no cpu_expert_devices) leaves cpu_node = -1 ⇒ daemon UNPINNED
    // and byte-identical. LS_CPU_EXPERT_RESERVE_CORE=0 overrides OFF.
    int daemon_pin_cpu = -1;
    bool reserve_on = true;
    if (const char* rc = std::getenv("LS_CPU_EXPERT_RESERVE_CORE");
        rc && rc[0] == '0' && rc[1] == '\0')
        reserve_on = false;
    if (reserve_on) {
        int cpu_node = -1;
        // Follow the device's EFFECTIVE node (LS_CPU_EXPERT_NODE, default 1) so
        // the reserved/pinned daemon core lives on the same node as the FFN pool.
        for (const auto& ced : cfg_->hardware.cpu_expert_devices)
            if (ced.enabled) { cpu_node = effective_cpu_expert_node(ced.numa_node); break; }
        if (cpu_node >= 0) {
            const auto cpus = compute::cpu::node_physical_cpus(cpu_node);
            if (!cpus.empty()) daemon_pin_cpu = cpus.front();
        }
    }
    daemon_thread_ = std::thread([this, daemon_pin_cpu] {
        if (daemon_pin_cpu >= 0) {
            if (compute::cpu::pin_thread_to_cpu(daemon_pin_cpu))
                spdlog::info("C-6 QA: daemon thread pinned to reserved core cpu {} "
                             "(LS_CPU_EXPERT_RESERVE_CORE)", daemon_pin_cpu);
            else
                spdlog::warn("C-6 QA: daemon pin to cpu {} failed (running unpinned)",
                             daemon_pin_cpu);
        }
        daemon_loop_impl_->run();
    });
}

// ── Shutdown ────────────────────────────────────────────────────────────────

uint64_t Engine::logits_readback_addr() const {
    return command_dispatcher_
        ? reinterpret_cast<uint64_t>(
              command_dispatcher_->logits_readback_host())
        : 0;
}

uint64_t Engine::logits_readback_bytes() const {
    return command_dispatcher_
        ? static_cast<uint64_t>(command_dispatcher_->logits_readback_bytes())
        : 0;
}

ipc::EngineH2dPathStats Engine::h2d_path_stats() const {
    ipc::EngineH2dPathStats out{};
    if (elm_) {
        auto s = elm_->h2d_path_stats();
        out.direct = s.direct;
        out.staged = s.staged;
    }
    if (transfer_engine_) {
        auto p = transfer_engine_->h2d_phase_stats();  // 481-3 sub-phase x-ray
        out.memcpy_ns   = p.memcpy_ns;
        out.dma_wait_ns = p.dma_wait_ns;
        out.phase_count = p.count;
        out.dma_gpu_ns    = p.dma_gpu_ns;     // perf_trace pure-DMA x-ray
        out.dma_gpu_count = p.gpu_count;
    }
    return out;
}

// ── P-24b lever-1: early arena-attach worker ─────────────────────────────────
// Overlaps holder attach + warm adopt + (warm-only) registration with rank
// init and the GGUF load. Gating mirrors the step-18d arena block exactly —
// substituting "manifest.json exists" for "PrepackedSource constructed".
// Cold boots deliberately do NOT register here: registration stays overlapped
// with the NVMe preload at the late block (TD-INIT-OVERLAP).

void Engine::launch_arena_attach_early_() {
    namespace fs = std::filesystem;
    bool attach_enabled = cfg_->memory.arena_attach.enabled;
    if (const char* aa = std::getenv("LS_ARENA_ATTACH"); aa && *aa)
        attach_enabled = (*aa != '0');
    const bool gates =
        attach_enabled &&
        cfg_->memory.preload_expert_buffers &&
        cfg_->preprocessing.host_cache_mode == config::HostCacheMode::mmap &&
        !cfg_->preprocessing.legacy_weights &&
        !cfg_->preprocessing.prepacked_dir.empty() &&
        cfg_->memory.pin_host_expert_pool &&
        numa_manager_ && numa_manager_->numa_available() &&
        !backends_.skip_hardware_detection;
    if (!gates) return;
    std::error_code ec;
    if (!fs::exists(model::prepacked::manifest_path(
            cfg_->preprocessing.prepacked_dir), ec)) return;

    // THP policy must be set before ANY shared allocation/adopt on the worker.
    numa_manager_->set_shared_thp(cfg_->memory.pin_host_expert_pool_thp);

    // Arena host placement identity (memory.arena_placement.freq_table /
    // LS_ARENA_PLACE_FREQ override) — resolved on the MAIN thread so an
    // unreadable table fails engine init LOUDLY (a throw inside the worker
    // would silently degrade to the private-arena fallback).
    arena_early_.placement_id =
        memory::ArenaPlacementPolicy::resolve(
            cfg_->memory.arena_placement.freq_table).identity;

    arena_early_.launched = true;
    arena_early_.thread = std::thread([this] { arena_attach_early_worker_(); });
}

void Engine::arena_attach_early_worker_() {
    memory::ArenaBacking backing{};
    try {
        namespace fs = std::filesystem;
        // Manifest-only parse (cheap; PrepackedSource is built later on the
        // main thread — its slot_size_bytes must equal this stride, verified
        // at join).
        const auto manifest =
            model::read_manifest(cfg_->preprocessing.prepacked_dir);
        arena_early_.slot_bytes = static_cast<size_t>(
            manifest.slot.stride_bytes > 0 ? manifest.slot.stride_bytes
                                           : manifest.slot.slot_size_bytes);
        if (arena_early_.slot_bytes == 0) return;  // malformed → private path

        // Config-stable identity (node set + share degrees from topology +
        // spill config — NO free-RAM budgets; see hash_config).
        const auto& spill = cfg_->memory.cross_node_spill;
        if (spill.enabled)
            for (const auto& sn : spill.nodes)
                if (sn.weight > 0) arena_early_.spill_weight[sn.node] = sn.weight;
        std::set<int> uni;
        for (int n : numa_manager_->gpu_attached_nodes()) {
            uni.insert(n);
            arena_early_.node_share[n] = 0;
        }
        for (const auto& [n, w] : arena_early_.spill_weight) uni.insert(n);
        for (int g = 0; g < numa_manager_->num_gpus(); ++g) {
            auto it = arena_early_.node_share.find(
                numa_manager_->gpu_numa_node(g));
            if (it != arena_early_.node_share.end()) ++it->second;
        }
        std::vector<memory::ArenaCache::NodeIdentity> nodeids;
        for (int n : uni) {
            const int sd = std::max(
                1, arena_early_.node_share.count(n)
                       ? arena_early_.node_share[n] : 1);
            arena_early_.nodeids.push_back({n, sd});
            nodeids.push_back({n, sd});
        }
        arena_early_.geom_hash = memory::ArenaCache::hash_config(
            arena_early_.slot_bytes,
            static_cast<size_t>(
                cfg_->memory.pin_host_expert_pool_extra_scratch_bytes),
            nodeids, cfg_->memory.pin_host_expert_pool_sizing, spill,
            arena_early_.placement_id);
        arena_early_.source_id = memory::ArenaCache::hash_source_id(
            std::filesystem::absolute(
                cfg_->preprocessing.prepacked_dir).string(),
            manifest.format_version,
            static_cast<int64_t>(arena_early_.slot_bytes));

        // Attach (may auto-spawn the holder).
        arena_ipc_ = std::make_unique<memory::ArenaIpcClient>();
        using Outcome = memory::ArenaIpcClient::AttachOutcome;
        const Outcome outcome = arena_ipc_->attach(
            cfg_->memory.arena_attach.socket_path,
            cfg_->memory.arena_attach.holder_binary,
            cfg_->memory.arena_attach.auto_spawn);
        if (outcome == Outcome::kBusy) {
            arena_ipc_.reset();
            arena_early_.fatal = std::make_exception_ptr(std::runtime_error(
                "arena_attach: another LayerStoRm instance is attached to the "
                "arena holder (single-attachment, INV-ARENA-HOLD). Stop it, "
                "use a different memory.arena_attach.socket_path, or set "
                "memory.arena_attach.enabled=false."));
            return;
        }
        if (outcome == Outcome::kError) {
            spdlog::warn("arena_attach: holder unreachable — process-private "
                         "arena this run (no persistence)");
            arena_ipc_.reset();
            return;
        }
        if (outcome == Outcome::kEmpty) {
            arena_early_.cold_attached = true;  // late block creates + STOREs
            return;
        }

        // Warm: validate stored meta (config identity), adopt the STORED
        // geometry, build the arena and REGISTER it here — nothing else to
        // overlap registration with on a warm boot.
        auto segs = arena_ipc_->take_segments();
        arena_meta_ = std::make_unique<memory::ArenaMetaSegment>(
            memory::ArenaMetaSegment::adopt(arena_ipc_->take_meta_fd()));
        const char* why = "";  // first failed validation check (diagnostics)
        bool ok = arena_meta_->valid();
        if (!ok) why = "meta segment invalid";
        if (ok) {
            arena_cache_ = std::make_unique<memory::ArenaCache>(
                arena_meta_->base(), arena_meta_->bytes());
            ok = arena_cache_->open_stored(arena_early_.geom_hash,
                                           arena_early_.source_id);
            if (!ok)
                why = "config identity (geometry/placement/source hash) "
                      "differs from the stored arena";
        }
        std::vector<memory::ArenaCacheNodeGeom> stored;
        if (ok) {
            stored = arena_cache_->stored_geometry();
            ok = stored.size() == arena_early_.nodeids.size() &&
                 segs.size() == stored.size();
            if (!ok) why = "stored node/segment count differs";
        }
        if (ok) {
            // TD-ARENA-MIGRATE-EMA-PERSIST: a pre-trailer store's meta
            // segment has no EMA room — grow it in place (memfd growth
            // persists with the holder's fd; new bytes read zero = trailer
            // invalid = cold EMA this boot, persisted from now on). Growth
            // remaps, so reopen the cache on the new base. Failure only
            // disables EMA persistence — adoption proceeds untouched.
            const size_t want = memory::ArenaCache::required_bytes_with_ema(
                stored,
                static_cast<uint32_t>(cfg_->model.num_hidden_layers),
                static_cast<uint32_t>(cfg_->model.n_routed_experts));
            if (arena_meta_->bytes() < want) {
                if (arena_meta_->ensure_bytes(want)) {
                    arena_cache_ = std::make_unique<memory::ArenaCache>(
                        arena_meta_->base(), arena_meta_->bytes());
                    ok = arena_cache_->open_stored(arena_early_.geom_hash,
                                                   arena_early_.source_id);
                } else {
                    spdlog::warn(
                        "arena_attach: meta-segment grow for the EMA "
                        "trailer failed — EMA persistence disabled this "
                        "store");
                }
            }
        }
        if (ok) {
            const size_t scratch = static_cast<size_t>(
                cfg_->memory.pin_host_expert_pool_extra_scratch_bytes);
            std::unordered_map<int, memory::ArenaIpcSegment> by_node;
            for (const auto& sg : segs) by_node.emplace(sg.numa_node, sg);
            for (const auto& g : stored) {
                auto it = by_node.find(g.node);
                if (it == by_node.end() ||
                    it->second.size_bytes !=
                        memory::PinnedExpertArena::slab_bytes(
                            arena_early_.slot_bytes, g.num_slots, scratch)) {
                    ok = false;
                    why = "per-node segment size differs from expected "
                          "slab bytes";
                    break;
                }
            }
            if (ok) {
                for (const auto& g : stored) {
                    auto& sg = by_node.at(g.node);
                    backing.adopted[g.node] = numa_manager_->adopt_shared(
                        sg.fd, sg.size_bytes, g.node);
                    sg.fd = -1;
                    backing.adopt_plans.push_back(
                        {g.node, arena_early_.spill_weight.count(g.node) != 0,
                         static_cast<size_t>(g.num_slots),
                         std::max(1, arena_early_.node_share.count(g.node)
                                         ? arena_early_.node_share[g.node]
                                         : 1)});
                }
                backing.mode = memory::ArenaBackingMode::kAdopt;
            }
        }
        if (!ok) {
            for (auto& [n, b] : backing.adopted) numa_manager_->free(b);
            backing.adopted.clear();
            backing.adopt_plans.clear();
            arena_cache_.reset();
            arena_meta_.reset();  // closes our meta fd
            for (auto& sg : segs)
                if (sg.fd >= 0) {
                    ::close(sg.fd);
                    sg.fd = -1;
                }
            // memory.arena_attach.on_conflict governs the mismatch reaction
            // (absent = derived from persist: false → kill, true → fail —
            // resolved_arena_on_conflict; the persist × on_conflict
            // compatibility matrix was already enforced at config parse).
            using OnConflict = config::ArenaOnConflict;
            const OnConflict on_conflict =
                config::resolved_arena_on_conflict(cfg_->memory.arena_attach);
            if (on_conflict == OnConflict::fail) {
                // 'fail' (the persist=true default): the store is
                // authoritative — NEVER wipe it, never rebuild over it,
                // never fall back to a second full-size private arena
                // (2026-08-18 incident: a mismatch wipe + OOM-killed
                // rebuild destroyed a 504 GB warm store).
                arena_ipc_.reset();  // detach; the holder retains the store
                arena_early_.fatal = std::make_exception_ptr(
                    std::runtime_error(std::string(
                        "arena_attach: the holder store does not match "
                        "this config (") + why + ") and the resolved "
                        "memory.arena_attach.on_conflict is 'fail' — "
                        "refusing to touch the store. Fix the config to "
                        "match it, run ONE boot with on_conflict='kill' "
                        "(persist=false) to wipe + cold-rebuild "
                        "intentionally, or use on_conflict='new' "
                        "(persist=false) to run on a private arena while "
                        "preserving the store."));
                return;
            }
            if (on_conflict == OnConflict::new_arena) {
                // 'new': PRESERVE the holder store untouched and run THIS
                // engine on a process-private arena for the run — the same
                // machinery as the holder-unreachable fallback (arena_ipc_
                // == null → the late block builds the classic private
                // arena).
                arena_ipc_.reset();  // detach; the holder retains the store
                spdlog::warn(
                    "arena_attach: stored arena is stale/mismatched ({}) "
                    "and on_conflict='new' — holder store PRESERVED "
                    "untouched; PERSISTENCE BYPASSED this run "
                    "(process-private arena; the retained store still "
                    "occupies its host RAM)", why);
                return;
            }
            // 'kill' (the historical persist=false behavior): wipe the
            // holder store and cold-rebuild it in place.
            spdlog::warn("arena_attach: stored arena is stale/mismatched "
                         "({}) — on_conflict='kill': wiping and rebuilding "
                         "cold", why);
            // The holder drops its dups NOW (kWipe, reason
            // kIdentityMismatch + the named failing check) — the old arena's
            // pages must be freed BEFORE the replacement prefault (old+new
            // must never coexist in RAM). The holder honors reasoned wipes
            // and stays up to host the successor store.
            if (!arena_ipc_->wipe(
                    memory::arena_ipc::WipeReason::kIdentityMismatch, why)) {
                // Refusal can only come from a holder that does not speak the
                // reason contract (pre-2026-08-22 binary) or a protocol
                // error. Treat as attach failure and ESCALATE to 'fail'
                // naming the refusal — never silently degrade to 'new' (a
                // surprise second full-size arena beside the retained store).
                arena_ipc_.reset();  // detach; the holder retains the store
                arena_early_.fatal = std::make_exception_ptr(
                    std::runtime_error(std::string(
                        "arena_attach: the holder store does not match "
                        "this config (") + why + ") and "
                        "on_conflict='kill', but the holder REFUSED the "
                        "reasoned wipe (kIdentityMismatch) — escalating to "
                        "'fail': store retained, boot aborted. Restart the "
                        "holder from the current build (an older binary "
                        "without the kWipe reason contract, or one still "
                        "passed the retired --persist flag, refuses every "
                        "wipe), or fix the config to match the store."));
                return;
            }
            arena_early_.cold_attached = true;
            return;
        }
        for (auto& sg : segs)
            if (sg.fd >= 0) ::close(sg.fd);  // all moved into buffers: none

        const size_t total_expert_bytes = arena_early_.slot_bytes
            * static_cast<size_t>(cfg_->model.n_routed_experts)
            * static_cast<size_t>(model_cfg_->num_moe_layers());
        pinned_arena_ = std::make_unique<memory::PinnedExpertArena>(
            *numa_manager_, arena_early_.slot_bytes, total_expert_bytes,
            cfg_->memory.pin_host_expert_pool_sizing,
            cfg_->memory.cross_node_spill,
            static_cast<size_t>(
                cfg_->memory.pin_host_expert_pool_extra_scratch_bytes),
            /*defer_registration=*/true, &backing);
        // Round 2b: cudaHostRegister holds driver locks that stall every
        // OTHER CUDA call — registering during rank init serialized the whole
        // boot (measured: GGUF load only STARTED after register finished).
        // Wait for the main thread to reach its CPU-only phase (the GGUF
        // load) so the 30+ s register overlaps disk/CPU work instead.
        while (!arena_early_.start_register.load(std::memory_order_acquire))
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        pinned_arena_->register_all();  // parallel; resident pages
        arena_early_.warm_ready = true;
        spdlog::info("arena_attach: early worker adopted + registered the "
                     "persistent arena ({} node(s), {:.1f} GB) overlapped "
                     "with rank init", pinned_arena_->num_arenas(),
                     pinned_arena_->total_pinned_bytes() / 1073741824.0);
    } catch (const std::exception& e) {
        for (auto& [n, b] : backing.adopted) numa_manager_->free(b);
        pinned_arena_.reset();
        arena_cache_.reset();
        arena_meta_.reset();
        arena_ipc_.reset();  // detach only — the holder retains its store
        arena_early_.cold_attached = false;
        arena_early_.warm_ready = false;
        if (config::resolved_arena_on_conflict(cfg_->memory.arena_attach) ==
            config::ArenaOnConflict::fail) {
            // 'fail' (the persist=true default): a failed adoption must not
            // degrade into a second full-size private arena beside the
            // retained store (guaranteed OOM on a store-sized box) — fail
            // the boot loud.
            arena_early_.fatal = std::make_exception_ptr(std::runtime_error(
                std::string("arena_attach: warm adoption failed (") +
                e.what() + ") and the resolved memory.arena_attach."
                "on_conflict is 'fail' — the holder store is retained; "
                "refusing the process-private fallback. Fix the failure, or "
                "select on_conflict='new'/'kill' (persist=false)."));
            return;
        }
        // 'kill'/'new' (the historical persist=false behavior): clean up
        // fully; the late block falls back to the classic private arena (or
        // staging pool) exactly as if attach were disabled. No wipe — an
        // adoption exception is not an identity mismatch, so the store is
        // left for the next boot to retry.
        spdlog::error("arena_attach: early worker failed ({}) — falling back "
                      "to a process-private arena", e.what());
    }
}

void Engine::shutdown() {
    // P-24b lever-1: if init aborted between worker launch and join, reap it.
    arena_early_.start_register.store(true, std::memory_order_release);
    if (arena_early_.thread.joinable()) arena_early_.thread.join();
    if (!running_.load(std::memory_order_acquire)) {
        if (daemon_thread_.joinable()) {
            daemon_thread_.join();
        }
        return;
    }

    // Signal shutdown via IpcHeader
    if (ipc_header_) {
        ipc_header_->shutdown_requested = 1;
    }
    running_.store(false, std::memory_order_release);

    // Join daemon thread
    if (daemon_thread_.joinable()) {
        daemon_thread_.join();
    }

    // TD-ARENA-MIGRATE-EMA-PERSIST: final EMA snapshot into the holder-store
    // meta trailer (daemon stopped — single writer; arena_cache_/migrator
    // are still alive here, destroyed with the Engine).
    if (ema_persist_fn_) ema_persist_fn_();

    // J-1: join the async cold-load workers now — the daemon has stopped, so no
    // new submits can arrive. Must happen while prepacked_source_ (mmap source)
    // and pinned_arena_ (slot dest) are still alive, since workers read/write
    // them. (Member destruction order would otherwise free prepacked_source_
    // first.) Destructor drains + joins.
    arena_loader_.reset();

    // KD-4f-d.1b: synchronize all GPUs before freeing any device memory.
    // Pending async work (NCCL, GEMM, D2H copies) must complete first.
    for (auto& dev : device_backends_) {
        if (dev) dev->synchronize_device();
    }

    // SPEC-SCAFFOLD: destroy the speculation method before device teardown —
    // it holds non-owning AttentionDevice/ExpertDevice/LoadedModel pointers
    // and a concrete method may own device allocations freed through them.
    speculation_method_.reset();

    // Destroy DCP components before freeing device memory.
    // Order matters: DcpExecutor → DcpAttentionWrapper → DcpCommunicator
    // (DcpAttentionWrapper holds a raw DcpCommunicator* that must stay valid).
    dcp_executor_.reset();
    dcp_attention_wrapper_.reset();  // TD-74i: frees global_lse_buffers_ on each GPU
    command_dispatcher_.reset();

    // DSP-3: destroy the DSpark runtime after the dispatcher (which holds a
    // raw Deps.dspark pointer) and before device-backend teardown (it frees
    // the draft arena + scratch through its rank backend).
    dspark_runtime_.reset();
    // TD-DSPARK-DRAFT-QUANT/-SHARD: the dedicated TP-GPU draft stream(s)
    // outlive the runtime (its Ranks held them non-owning); destroy now.
    for (auto& [be, ds] : dspark_draft_streams_) {
        if (!be || !ds) continue;
        be->set_device();
        be->destroy_stream(ds);
    }
    dspark_draft_streams_.clear();

    // Sync before NCCL teardown.
    for (auto& dev : device_backends_) {
        if (dev) dev->synchronize_device();
    }

    dcp_communicator_.reset();

    collective_backend_.reset();

    // Re-sync after NCCL teardown to clear any errors it introduced.
    for (auto& dev : device_backends_) {
        if (dev) dev->synchronize_device();
    }

    // Drain in-flight transfers
    if (transfer_engine_) {
        transfer_engine_->drain();
    }

    // Free TQ device memory before attention devices are destroyed
    // (INV-TQ-PERRANK: each rank's resources free on ITS device).
    for (size_t i = 0; i < tq_resources_.size(); ++i) {
        if (tq_resources_[i] && i < attention_devices_.size()
            && attention_devices_[i]) {
            compute::destroy_tq_resources(*tq_resources_[i],
                                          *attention_devices_[i]);
        }
    }
    tq_resources_.clear();

    // KD-R2: Free paired hidden state device buffers before devices are destroyed.
    for (auto& pair : hidden_state_pairs_) {
        if (pair.attn_buf && pair.rank >= 0
            && static_cast<size_t>(pair.rank) < attention_devices_.size()
            && attention_devices_[pair.rank]) {
            attention_devices_[pair.rank]->device_free(pair.attn_buf);
        }
        if (pair.moe_buf && pair.gpu_position >= 0
            && static_cast<size_t>(pair.gpu_position) < expert_devices_.size()
            && expert_devices_[pair.gpu_position]) {
            expert_devices_[pair.gpu_position]->device_free(pair.moe_buf);
        }
    }
    hidden_state_pairs_.clear();

    for (size_t i = 0; i < fused_moe_hidden_state_bufs_.size(); ++i) {
        if (fused_moe_hidden_state_bufs_[i] && i < expert_devices_.size()
            && expert_devices_[i]) {
            expert_devices_[i]->device_free(fused_moe_hidden_state_bufs_[i]);
        }
    }
    fused_moe_hidden_state_bufs_.clear();

    // KD-4f-c3: Free FP8 weight + scale buffers for quantized attention projections.
    for (auto& alloc : quantized_attn_allocs_) {
        if (alloc.rank >= 0
            && static_cast<size_t>(alloc.rank) < attention_devices_.size()
            && attention_devices_[alloc.rank]) {
            attention_devices_[alloc.rank]->device_free(alloc.fp8_weight);
            attention_devices_[alloc.rank]->device_free(alloc.scales);
        }
    }
    quantized_attn_allocs_.clear();

    // compute.ipc_pin: unregister the IPC region before its std::free (all GPU
    // work is drained above; a registered region must never be freed while
    // pinned — cudaHostUnregister sticky-error trap, DEBUG.md).
    if (ipc_region_registered_) {
        core::host_unregister_pinned(ipc_region_.get());
        ipc_region_registered_ = false;
    }

    spdlog::info("Engine shut down");
}

// ── Free functions (singleton API) ──────────────────────────────────────────

ipc::EngineInfo start_engine(const std::string& config_path) {
    return start_engine(config_path, default_backends());
}

ipc::EngineInfo start_engine(const std::string& config_path,
                             EngineBackends backends) {
    std::lock_guard lock(g_engine_mutex);
    if (g_engine) {
        throw std::runtime_error("Engine already running — call stop_engine() first");
    }
    g_engine = std::make_unique<Engine>(config_path, std::move(backends));
    return g_engine->info();
}

void stop_engine() {
    std::lock_guard lock(g_engine_mutex);
    if (g_engine) {
        g_engine->shutdown();
        g_engine.reset();
    }
}

Engine* get_engine() {
    return g_engine.get();
}

}  // namespace layerstorm::daemon

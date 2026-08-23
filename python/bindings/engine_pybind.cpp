// pybind11 bindings for the LayerStoRm engine module.
//
// Extracted from src/daemon/engine.cpp to allow Engine to link into
// layerstorm_core without pulling in pybind11.

#include <pybind11/pybind11.h>

#include "core/memory/expert_cache.h"
#include "daemon/engine.h"
#include "daemon/ipc_protocol.h"
#include "model/weight_pipeline/expert_prepacker.h"

namespace py = pybind11;
using namespace layerstorm;

PYBIND11_MODULE(layerstorm_engine, m) {
    m.doc() = "LayerStoRm C++ engine: start_engine/stop_engine + EngineInfo";

    py::class_<ipc::EngineInfo>(m, "EngineInfo")
        .def_readonly("ipc_base",        &ipc::EngineInfo::ipc_base)
        .def_readonly("ipc_total_bytes", &ipc::EngineInfo::ipc_total_bytes)
        .def_readonly("cmd_ring_offset", &ipc::EngineInfo::cmd_ring_offset)
        .def_readonly("cmd_ring_slots",  &ipc::EngineInfo::cmd_ring_slots)
        .def_readonly("cmd_slot_bytes",  &ipc::EngineInfo::cmd_slot_bytes)
        .def_readonly("cmp_ring_offset", &ipc::EngineInfo::cmp_ring_offset)
        .def_readonly("cmp_ring_slots",  &ipc::EngineInfo::cmp_ring_slots)
        .def_readonly("cmp_slot_bytes",  &ipc::EngineInfo::cmp_slot_bytes)
        .def_readonly("state_offset",    &ipc::EngineInfo::state_offset)
        .def_readonly("state_bytes",     &ipc::EngineInfo::state_bytes)
        .def_readonly("num_gpus",        &ipc::EngineInfo::num_gpus)
        .def_readonly("num_moe_layers",  &ipc::EngineInfo::num_moe_layers)
        .def_readonly("num_experts",     &ipc::EngineInfo::num_experts)
        // TD-PREFILL-SUPERCHUNK: effective MoE token-batch capacity.
        .def_readonly("moe_batch_capacity", &ipc::EngineInfo::moe_batch_capacity)
        .def_readonly("expert_bytes",    &ipc::EngineInfo::expert_bytes)
        .def_readonly("num_layers",      &ipc::EngineInfo::num_layers)
        .def_readonly("num_expert_devices", &ipc::EngineInfo::num_expert_devices)
        .def_readonly("kv_bytes_per_page", &ipc::EngineInfo::kv_bytes_per_page)
        .def_readonly("vocab_size",      &ipc::EngineInfo::vocab_size)
        .def_readonly("sideband_offset", &ipc::EngineInfo::sideband_offset)
        .def_readonly("sideband_bytes",  &ipc::EngineInfo::sideband_bytes)
        // V4-7a: DeepSeek-V4 metadata (zeros / empty for non-V4 models).
        .def_readonly("v4_hc_mult",      &ipc::EngineInfo::v4_hc_mult)
        .def_readonly("v4_num_hash_layers",
                      &ipc::EngineInfo::v4_num_hash_layers)
        .def_property_readonly("v4_attention_types",
            [](const ipc::EngineInfo& i) {
                py::list out;
                const int n = std::min<int>(
                    i.num_layers, ipc::EngineInfo::kV4MaxLayers);
                if (i.v4_hc_mult > 0)
                    for (int l = 0; l < n; ++l)
                        out.append(static_cast<int>(i.v4_attention_types[l]));
                return out;
            });

    m.def("start_engine",
          py::overload_cast<const std::string&>(&daemon::start_engine),
          py::arg("config_path"),
          "Initialize engine with CUDA backends, spawn daemon thread, return EngineInfo");

    m.def("start_engine_test",
          [](const std::string& config_path) {
              return daemon::start_engine(config_path, daemon::null_backends());
          },
          py::arg("config_path"),
          "Initialize engine with null backends for testing without CUDA");

    m.def("stop_engine", &daemon::stop_engine,
          "Shutdown daemon thread and free all resources");

    m.def("prepack_experts",
          [](const std::string& config_path, const std::string& output_dir) {
              model::PrepackResult result;
              {
                  py::gil_scoped_release release;
                  result = daemon::Engine::run_prepack(config_path, output_dir);
              }
              py::dict d;
              d["experts_written"] = result.experts_written;
              d["experts_skipped"] = result.experts_skipped;
              d["bytes_written"] = result.bytes_written;
              d["error"] = result.error;
              return d;
          },
          py::arg("config_path"), py::arg("output_dir"),
          "Offline pre-processing: pack raw safetensors into DMA-ready expert files");

    m.def("query_buffer_ids", []() -> py::dict {
        py::dict d;
        auto* engine = daemon::get_engine();
        if (engine && engine->buffer_registry()) {
            for (const auto& [id, name] : engine->buffer_registry()->all_named_entries()) {
                if (!name.empty()) {
                    d[py::cast(name)] = id;
                }
            }
        }
        return d;
    }, "Return {name: buf_id} mapping for all named buffers");

    // TD-SERVE-NAMED-TOOL-CHOICE (guided decoding): host address of the
    // pinned full-logits readback row (CMD_OUTPUT_HEAD readback_logits=1
    // D2H target).  Single process — the orchestrator reads it directly
    // after the OUTPUT_HEAD completion.  Returns (addr, bytes); (0, 0)
    // when unavailable (engine down / no CUDA / vocab 0).
    m.def("logits_readback_addr", []() -> py::tuple {
        auto* engine = daemon::get_engine();
        if (!engine)
            return py::make_tuple(uint64_t{0}, uint64_t{0});
        return py::make_tuple(engine->logits_readback_addr(),
                              engine->logits_readback_bytes());
    }, "Return (host_addr, bytes) of the guided-decoding logits readback row");

    m.def("ipc_tx_config", []() -> py::dict {
        py::dict d;
        auto* engine = daemon::get_engine();
        if (!engine) return d;
        // Expose IPC transaction config for Python-side retry policy init
        d["reader_max_retries"] = engine->config().orchestrator.ipc_transaction.reader_max_retries;
        d["reader_backoff_stage_size"] = engine->config().orchestrator.ipc_transaction.reader_backoff_stage_size;
        d["reader_stage0_yield_every"] = engine->config().orchestrator.ipc_transaction.reader_stage0_yield_every;
        d["writer_yield_interval"] = engine->config().orchestrator.ipc_transaction.writer_yield_interval;
        return d;
    }, "Return IPC transaction config for Python retry policy");

    // ── IPC layout introspection for cross-language consistency tests ──
    // Each returns {struct_name: {"sizeof": N, "fields": {name: (offset, size)}}}

    // Helper lambda shared by all layout functions.
    auto make_layout = [](std::initializer_list<std::tuple<const char*, size_t, size_t>> fields,
                          size_t total_size) -> py::dict {
        py::dict d;
        d["sizeof"] = total_size;
        py::dict fdict;
        for (auto& [name, offset, size] : fields)
            fdict[name] = py::make_tuple(offset, size);
        d["fields"] = fdict;
        return d;
    };

    // Category 1: IPC envelope structs (IpcHeader, RingHeader, Command, Completion,
    // GpuSnapshot, RequestAcceptance, StateSnapshot, EngineInfo)
    m.def("ipc_struct_layout", [make_layout]() -> py::dict {
        auto make = make_layout;
        py::dict result;

        result["IpcHeader"] = make({
            {"version",            offsetof(ipc::IpcHeader, version),            sizeof(ipc::IpcHeader::version)},
            {"shutdown_requested", offsetof(ipc::IpcHeader, shutdown_requested), sizeof(ipc::IpcHeader::shutdown_requested)},
            {"error_code",         offsetof(ipc::IpcHeader, error_code),         sizeof(ipc::IpcHeader::error_code)},
            {"error_message",      offsetof(ipc::IpcHeader, error_message),      sizeof(ipc::IpcHeader::error_message)},
        }, sizeof(ipc::IpcHeader));

        result["RingHeader"] = make({
            {"producer_seq", offsetof(ipc::RingHeader, producer_seq), sizeof(ipc::RingHeader::producer_seq)},
            {"consumer_seq", offsetof(ipc::RingHeader, consumer_seq), sizeof(ipc::RingHeader::consumer_seq)},
            {"slot_count",   offsetof(ipc::RingHeader, slot_count),   sizeof(ipc::RingHeader::slot_count)},
            {"slot_size",    offsetof(ipc::RingHeader, slot_size),    sizeof(ipc::RingHeader::slot_size)},
        }, sizeof(ipc::RingHeader));

        result["Command"] = make({
            {"cmd_type",  offsetof(ipc::Command, cmd_type),  sizeof(ipc::Command::cmd_type)},
            {"cmd_seq",   offsetof(ipc::Command, cmd_seq),   sizeof(ipc::Command::cmd_seq)},
            {"gpu_idx",   offsetof(ipc::Command, gpu_idx),   sizeof(ipc::Command::gpu_idx)},
            {"stream_id", offsetof(ipc::Command, stream_id), sizeof(ipc::Command::stream_id)},
            {"payload",   offsetof(ipc::Command, raw),       sizeof(ipc::Command::raw)},
        }, sizeof(ipc::Command));

        result["Completion"] = make({
            {"cmp_type", offsetof(ipc::Completion, cmp_type), sizeof(ipc::Completion::cmp_type)},
            {"cmd_seq",  offsetof(ipc::Completion, cmd_seq),  sizeof(ipc::Completion::cmd_seq)},
            {"gpu_idx",  offsetof(ipc::Completion, gpu_idx),  sizeof(ipc::Completion::gpu_idx)},
            {"status",   offsetof(ipc::Completion, status),   sizeof(ipc::Completion::status)},
            {"payload",  offsetof(ipc::Completion, raw),      sizeof(ipc::Completion::raw)},
        }, sizeof(ipc::Completion));

        result["GpuSnapshot"] = make({
            {"vram_used_bytes",        offsetof(ipc::GpuSnapshot, vram_used_bytes),        sizeof(ipc::GpuSnapshot::vram_used_bytes)},
            {"vram_total_bytes",       offsetof(ipc::GpuSnapshot, vram_total_bytes),       sizeof(ipc::GpuSnapshot::vram_total_bytes)},
            {"expert_stable_used",     offsetof(ipc::GpuSnapshot, expert_stable_used),     sizeof(ipc::GpuSnapshot::expert_stable_used)},
            {"expert_stable_total",    offsetof(ipc::GpuSnapshot, expert_stable_total),     sizeof(ipc::GpuSnapshot::expert_stable_total)},
            {"expert_streaming_used",  offsetof(ipc::GpuSnapshot, expert_streaming_used),  sizeof(ipc::GpuSnapshot::expert_streaming_used)},
            {"expert_streaming_total", offsetof(ipc::GpuSnapshot, expert_streaming_total), sizeof(ipc::GpuSnapshot::expert_streaming_total)},
            {"kv_main_free_pages",     offsetof(ipc::GpuSnapshot, kv_main_free_pages),     sizeof(ipc::GpuSnapshot::kv_main_free_pages)},
            {"kv_spec_free_pages",     offsetof(ipc::GpuSnapshot, kv_spec_free_pages),     sizeof(ipc::GpuSnapshot::kv_spec_free_pages)},
            {"inflight_h2d_count",     offsetof(ipc::GpuSnapshot, inflight_h2d_count),     sizeof(ipc::GpuSnapshot::inflight_h2d_count)},
            {"inflight_d2h_count",     offsetof(ipc::GpuSnapshot, inflight_d2h_count),     sizeof(ipc::GpuSnapshot::inflight_d2h_count)},
            {"compute_queue_depth",    offsetof(ipc::GpuSnapshot, compute_queue_depth),    sizeof(ipc::GpuSnapshot::compute_queue_depth)},
            {"prefill_mode",           offsetof(ipc::GpuSnapshot, prefill_mode),           sizeof(ipc::GpuSnapshot::prefill_mode)},
        }, sizeof(ipc::GpuSnapshot));

        result["RequestAcceptance"] = make({
            {"request_id",      offsetof(ipc::RequestAcceptance, request_id),      sizeof(ipc::RequestAcceptance::request_id)},
            {"acceptance_rate", offsetof(ipc::RequestAcceptance, acceptance_rate), sizeof(ipc::RequestAcceptance::acceptance_rate)},
        }, sizeof(ipc::RequestAcceptance));

        result["StateSnapshot"] = make({
            {"seqlock",                    offsetof(ipc::StateSnapshot, seqlock),                    sizeof(ipc::StateSnapshot::seqlock)},
            {"daemon_cycle_count",         offsetof(ipc::StateSnapshot, daemon_cycle_count),         sizeof(ipc::StateSnapshot::daemon_cycle_count)},
            {"timestamp_ns",              offsetof(ipc::StateSnapshot, timestamp_ns),              sizeof(ipc::StateSnapshot::timestamp_ns)},
            {"gpus",                       offsetof(ipc::StateSnapshot, gpus),                       sizeof(ipc::StateSnapshot::gpus)},
            {"num_gpus",                   offsetof(ipc::StateSnapshot, num_gpus),                   sizeof(ipc::StateSnapshot::num_gpus)},
            {"residency_bitmap",           offsetof(ipc::StateSnapshot, residency_bitmap),           sizeof(ipc::StateSnapshot::residency_bitmap)},
            {"expert_gpu_tier",            offsetof(ipc::StateSnapshot, expert_gpu_tier),            sizeof(ipc::StateSnapshot::expert_gpu_tier)},
            {"expert_interest_count",      offsetof(ipc::StateSnapshot, expert_interest_count),      sizeof(ipc::StateSnapshot::expert_interest_count)},
            {"host_resident_bitmap",       offsetof(ipc::StateSnapshot, host_resident_bitmap),       sizeof(ipc::StateSnapshot::host_resident_bitmap)},
            {"host_tier",                  offsetof(ipc::StateSnapshot, host_tier),                  sizeof(ipc::StateSnapshot::host_tier)},
            {"expert_frequency",           offsetof(ipc::StateSnapshot, expert_frequency),           sizeof(ipc::StateSnapshot::expert_frequency)},
            {"expert_recency",             offsetof(ipc::StateSnapshot, expert_recency),             sizeof(ipc::StateSnapshot::expert_recency)},
            {"expert_routing_weight",      offsetof(ipc::StateSnapshot, expert_routing_weight),      sizeof(ipc::StateSnapshot::expert_routing_weight)},
            {"expert_temporal_autocorr",   offsetof(ipc::StateSnapshot, expert_temporal_autocorr),   sizeof(ipc::StateSnapshot::expert_temporal_autocorr)},
            {"global_acceptance_rate",     offsetof(ipc::StateSnapshot, global_acceptance_rate),     sizeof(ipc::StateSnapshot::global_acceptance_rate)},
            {"windowed_acceptance_rate",   offsetof(ipc::StateSnapshot, windowed_acceptance_rate),   sizeof(ipc::StateSnapshot::windowed_acceptance_rate)},
            {"layer_skip_acceptance_rate", offsetof(ipc::StateSnapshot, layer_skip_acceptance_rate), sizeof(ipc::StateSnapshot::layer_skip_acceptance_rate)},
            {"total_verifications",        offsetof(ipc::StateSnapshot, total_verifications),        sizeof(ipc::StateSnapshot::total_verifications)},
            {"total_accepted_tokens",      offsetof(ipc::StateSnapshot, total_accepted_tokens),      sizeof(ipc::StateSnapshot::total_accepted_tokens)},
            {"total_attempted_tokens",     offsetof(ipc::StateSnapshot, total_attempted_tokens),     sizeof(ipc::StateSnapshot::total_attempted_tokens)},
            {"shift_detected",             offsetof(ipc::StateSnapshot, shift_detected),             sizeof(ipc::StateSnapshot::shift_detected)},
            {"total_inflight_transfers",   offsetof(ipc::StateSnapshot, total_inflight_transfers),   sizeof(ipc::StateSnapshot::total_inflight_transfers)},
            {"per_request_acceptance",     offsetof(ipc::StateSnapshot, per_request_acceptance),     sizeof(ipc::StateSnapshot::per_request_acceptance)},
            {"num_tracked_requests",       offsetof(ipc::StateSnapshot, num_tracked_requests),       sizeof(ipc::StateSnapshot::num_tracked_requests)},
            {"expert_host_numa",           offsetof(ipc::StateSnapshot, expert_host_numa),           sizeof(ipc::StateSnapshot::expert_host_numa)},
            {"host_numa_tier",             offsetof(ipc::StateSnapshot, host_numa_tier),             sizeof(ipc::StateSnapshot::host_numa_tier)},
            {"expert_last_change_ns",      offsetof(ipc::StateSnapshot, expert_last_change_ns),      sizeof(ipc::StateSnapshot::expert_last_change_ns)},
        }, sizeof(ipc::StateSnapshot));

        result["EngineInfo"] = make({
            {"ipc_base",           offsetof(ipc::EngineInfo, ipc_base),           sizeof(ipc::EngineInfo::ipc_base)},
            {"ipc_total_bytes",    offsetof(ipc::EngineInfo, ipc_total_bytes),    sizeof(ipc::EngineInfo::ipc_total_bytes)},
            {"cmd_ring_offset",    offsetof(ipc::EngineInfo, cmd_ring_offset),    sizeof(ipc::EngineInfo::cmd_ring_offset)},
            {"cmd_ring_slots",     offsetof(ipc::EngineInfo, cmd_ring_slots),     sizeof(ipc::EngineInfo::cmd_ring_slots)},
            {"cmd_slot_bytes",     offsetof(ipc::EngineInfo, cmd_slot_bytes),     sizeof(ipc::EngineInfo::cmd_slot_bytes)},
            {"cmp_ring_offset",    offsetof(ipc::EngineInfo, cmp_ring_offset),    sizeof(ipc::EngineInfo::cmp_ring_offset)},
            {"cmp_ring_slots",     offsetof(ipc::EngineInfo, cmp_ring_slots),     sizeof(ipc::EngineInfo::cmp_ring_slots)},
            {"cmp_slot_bytes",     offsetof(ipc::EngineInfo, cmp_slot_bytes),     sizeof(ipc::EngineInfo::cmp_slot_bytes)},
            {"state_offset",       offsetof(ipc::EngineInfo, state_offset),       sizeof(ipc::EngineInfo::state_offset)},
            {"state_bytes",        offsetof(ipc::EngineInfo, state_bytes),        sizeof(ipc::EngineInfo::state_bytes)},
            {"sideband_offset",    offsetof(ipc::EngineInfo, sideband_offset),    sizeof(ipc::EngineInfo::sideband_offset)},
            {"sideband_bytes",     offsetof(ipc::EngineInfo, sideband_bytes),     sizeof(ipc::EngineInfo::sideband_bytes)},
            {"num_gpus",           offsetof(ipc::EngineInfo, num_gpus),           sizeof(ipc::EngineInfo::num_gpus)},
            {"num_moe_layers",     offsetof(ipc::EngineInfo, num_moe_layers),     sizeof(ipc::EngineInfo::num_moe_layers)},
            {"num_experts",        offsetof(ipc::EngineInfo, num_experts),        sizeof(ipc::EngineInfo::num_experts)},
            {"moe_batch_capacity", offsetof(ipc::EngineInfo, moe_batch_capacity), sizeof(ipc::EngineInfo::moe_batch_capacity)},
            {"expert_bytes",       offsetof(ipc::EngineInfo, expert_bytes),       sizeof(ipc::EngineInfo::expert_bytes)},
            {"num_layers",         offsetof(ipc::EngineInfo, num_layers),         sizeof(ipc::EngineInfo::num_layers)},
            {"num_expert_devices", offsetof(ipc::EngineInfo, num_expert_devices), sizeof(ipc::EngineInfo::num_expert_devices)},
            {"kv_bytes_per_page",  offsetof(ipc::EngineInfo, kv_bytes_per_page),  sizeof(ipc::EngineInfo::kv_bytes_per_page)},
            {"vocab_size",         offsetof(ipc::EngineInfo, vocab_size),         sizeof(ipc::EngineInfo::vocab_size)},
            {"v4_hc_mult",         offsetof(ipc::EngineInfo, v4_hc_mult),         sizeof(ipc::EngineInfo::v4_hc_mult)},
            {"v4_num_hash_layers", offsetof(ipc::EngineInfo, v4_num_hash_layers), sizeof(ipc::EngineInfo::v4_num_hash_layers)},
            {"v4_attention_types", offsetof(ipc::EngineInfo, v4_attention_types), sizeof(ipc::EngineInfo::v4_attention_types)},
        }, sizeof(ipc::EngineInfo));

        return result;
    }, "C++ IPC envelope struct layouts (IpcHeader..EngineInfo)");

    // Category 2: IPC command + completion payload structs
    // Offsets relative to payload start (not to Command/Completion start).
    m.def("ipc_payload_layout", [make_layout]() -> py::dict {
        auto make = make_layout;
        py::dict result;

        // ── Command payload structs ──
        constexpr size_t CMD_PAY = offsetof(ipc::Command, raw);  // = 16

        result["TransferPayload"] = make({
            {"layer_idx",     offsetof(ipc::Command, transfer.layer_idx) - CMD_PAY, sizeof(ipc::Command::transfer.layer_idx)},
            {"expert_idx",    offsetof(ipc::Command, transfer.expert_idx) - CMD_PAY, sizeof(ipc::Command::transfer.expert_idx)},
            {"sub_component", offsetof(ipc::Command, transfer.sub_component) - CMD_PAY, sizeof(ipc::Command::transfer.sub_component)},
            {"zone",          offsetof(ipc::Command, transfer.zone) - CMD_PAY, sizeof(ipc::Command::transfer.zone)},
            {"bytes",         offsetof(ipc::Command, transfer.bytes) - CMD_PAY, sizeof(ipc::Command::transfer.bytes)},
            {"priority",      offsetof(ipc::Command, transfer.priority) - CMD_PAY, sizeof(ipc::Command::transfer.priority)},
            {"delay_us",      offsetof(ipc::Command, transfer.delay_us) - CMD_PAY, sizeof(ipc::Command::transfer.delay_us)},
        }, sizeof(ipc::Command::transfer));

        result["CacheReservePayload"] = make({
            {"layer_idx",    offsetof(ipc::Command, cache_reserve.layer_idx) - CMD_PAY, sizeof(ipc::Command::cache_reserve.layer_idx)},
            {"expert_idx",   offsetof(ipc::Command, cache_reserve.expert_idx) - CMD_PAY, sizeof(ipc::Command::cache_reserve.expert_idx)},
            {"zone",         offsetof(ipc::Command, cache_reserve.zone) - CMD_PAY, sizeof(ipc::Command::cache_reserve.zone)},
            {"is_duplicate", offsetof(ipc::Command, cache_reserve.is_duplicate) - CMD_PAY, sizeof(ipc::Command::cache_reserve.is_duplicate)},
        }, sizeof(ipc::Command::cache_reserve));

        result["CacheOpPayload"] = make({
            {"layer_idx",  offsetof(ipc::Command, cache_op.layer_idx) - CMD_PAY, sizeof(ipc::Command::cache_op.layer_idx)},
            {"expert_idx", offsetof(ipc::Command, cache_op.expert_idx) - CMD_PAY, sizeof(ipc::Command::cache_op.expert_idx)},
        }, sizeof(ipc::Command::cache_op));

        result["AttentionPayload"] = make({
            {"layer_idx",           offsetof(ipc::Command, attention.layer_idx) - CMD_PAY, sizeof(ipc::Command::attention.layer_idx)},
            {"batch_size",          offsetof(ipc::Command, attention.batch_size) - CMD_PAY, sizeof(ipc::Command::attention.batch_size)},
            {"use_graph",           offsetof(ipc::Command, attention.use_graph) - CMD_PAY, sizeof(ipc::Command::attention.use_graph)},
            {"is_sparse",           offsetof(ipc::Command, attention.is_sparse) - CMD_PAY, sizeof(ipc::Command::attention.is_sparse)},
            {"hidden_state_buf_id", offsetof(ipc::Command, attention.hidden_state_buf_id) - CMD_PAY, sizeof(ipc::Command::attention.hidden_state_buf_id)},
            {"kv_cache_buf_id",     offsetof(ipc::Command, attention.kv_cache_buf_id) - CMD_PAY, sizeof(ipc::Command::attention.kv_cache_buf_id)},
            {"seqlens_buf_id",      offsetof(ipc::Command, attention.seqlens_buf_id) - CMD_PAY, sizeof(ipc::Command::attention.seqlens_buf_id)},
            {"block_table_buf_id",  offsetof(ipc::Command, attention.block_table_buf_id) - CMD_PAY, sizeof(ipc::Command::attention.block_table_buf_id)},
        }, sizeof(ipc::Command::attention));

        result["GatingPayload"] = make({
            {"layer_idx",              offsetof(ipc::Command, gating.layer_idx) - CMD_PAY, sizeof(ipc::Command::gating.layer_idx)},
            {"num_tokens",             offsetof(ipc::Command, gating.num_tokens) - CMD_PAY, sizeof(ipc::Command::gating.num_tokens)},
            {"input_buf_id",           offsetof(ipc::Command, gating.input_buf_id) - CMD_PAY, sizeof(ipc::Command::gating.input_buf_id)},
            {"output_weights_buf_id",  offsetof(ipc::Command, gating.output_weights_buf_id) - CMD_PAY, sizeof(ipc::Command::gating.output_weights_buf_id)},
            {"output_indices_buf_id",  offsetof(ipc::Command, gating.output_indices_buf_id) - CMD_PAY, sizeof(ipc::Command::gating.output_indices_buf_id)},
        }, sizeof(ipc::Command::gating));

        result["ExpertFfnPayload"] = make({
            {"layer_idx",              offsetof(ipc::Command, expert_ffn.layer_idx) - CMD_PAY, sizeof(ipc::Command::expert_ffn.layer_idx)},
            {"num_experts",            offsetof(ipc::Command, expert_ffn.num_experts) - CMD_PAY, sizeof(ipc::Command::expert_ffn.num_experts)},
            {"total_tokens",           offsetof(ipc::Command, expert_ffn.total_tokens) - CMD_PAY, sizeof(ipc::Command::expert_ffn.total_tokens)},
            {"permuted_input_buf_id",  offsetof(ipc::Command, expert_ffn.permuted_input_buf_id) - CMD_PAY, sizeof(ipc::Command::expert_ffn.permuted_input_buf_id)},
            {"output_buf_id",          offsetof(ipc::Command, expert_ffn.output_buf_id) - CMD_PAY, sizeof(ipc::Command::expert_ffn.output_buf_id)},
            {"expert_offsets_buf_id",  offsetof(ipc::Command, expert_ffn.expert_offsets_buf_id) - CMD_PAY, sizeof(ipc::Command::expert_ffn.expert_offsets_buf_id)},
            {"hidden_dim",             offsetof(ipc::Command, expert_ffn.hidden_dim) - CMD_PAY, sizeof(ipc::Command::expert_ffn.hidden_dim)},
            {"weights_buf_id",         offsetof(ipc::Command, expert_ffn.weights_buf_id) - CMD_PAY, sizeof(ipc::Command::expert_ffn.weights_buf_id)},
            {"workspace_buf_id",       offsetof(ipc::Command, expert_ffn.workspace_buf_id) - CMD_PAY, sizeof(ipc::Command::expert_ffn.workspace_buf_id)},
            {"k_dim",                  offsetof(ipc::Command, expert_ffn.k_dim) - CMD_PAY, sizeof(ipc::Command::expert_ffn.k_dim)},
            {"quant_mode",             offsetof(ipc::Command, expert_ffn.quant_mode) - CMD_PAY, sizeof(ipc::Command::expert_ffn.quant_mode)},
            {"gguf_type",              offsetof(ipc::Command, expert_ffn.gguf_type) - CMD_PAY, sizeof(ipc::Command::expert_ffn.gguf_type)},
        }, sizeof(ipc::Command::expert_ffn));

        result["EmbeddingLookupPayload"] = make({
            {"num_tokens",    offsetof(ipc::Command, embedding_lookup.num_tokens) - CMD_PAY, sizeof(ipc::Command::embedding_lookup.num_tokens)},
            {"output_buf_id", offsetof(ipc::Command, embedding_lookup.output_buf_id) - CMD_PAY, sizeof(ipc::Command::embedding_lookup.output_buf_id)},
            {"row_offset",    offsetof(ipc::Command, embedding_lookup.row_offset) - CMD_PAY, sizeof(ipc::Command::embedding_lookup.row_offset)},
        }, sizeof(ipc::Command::embedding_lookup));

        result["OutputHeadPayload"] = make({
            {"num_tokens",          offsetof(ipc::Command, output_head.num_tokens) - CMD_PAY, sizeof(ipc::Command::output_head.num_tokens)},
            {"input_buf_id",        offsetof(ipc::Command, output_head.input_buf_id) - CMD_PAY, sizeof(ipc::Command::output_head.input_buf_id)},
            {"output_buf_id",       offsetof(ipc::Command, output_head.output_buf_id) - CMD_PAY, sizeof(ipc::Command::output_head.output_buf_id)},
            {"readback_to_host",    offsetof(ipc::Command, output_head.readback_to_host) - CMD_PAY, sizeof(ipc::Command::output_head.readback_to_host)},
            {"compute_confidence",  offsetof(ipc::Command, output_head.compute_confidence) - CMD_PAY, sizeof(ipc::Command::output_head.compute_confidence)},
            {"num_logprobs",        offsetof(ipc::Command, output_head.num_logprobs) - CMD_PAY, sizeof(ipc::Command::output_head.num_logprobs)},
            {"mtp_head",            offsetof(ipc::Command, output_head.mtp_head) - CMD_PAY, sizeof(ipc::Command::output_head.mtp_head)},
            {"readback_logits",     offsetof(ipc::Command, output_head.readback_logits) - CMD_PAY, sizeof(ipc::Command::output_head.readback_logits)},
        }, sizeof(ipc::Command::output_head));

        result["SampleTokensPayload"] = make({
            {"num_tokens",    offsetof(ipc::Command, sample_tokens.num_tokens) - CMD_PAY, sizeof(ipc::Command::sample_tokens.num_tokens)},
            {"logits_buf_id", offsetof(ipc::Command, sample_tokens.logits_buf_id) - CMD_PAY, sizeof(ipc::Command::sample_tokens.logits_buf_id)},
            {"vocab_size",    offsetof(ipc::Command, sample_tokens.vocab_size) - CMD_PAY, sizeof(ipc::Command::sample_tokens.vocab_size)},
            {"top_k",         offsetof(ipc::Command, sample_tokens.top_k) - CMD_PAY, sizeof(ipc::Command::sample_tokens.top_k)},
            {"temperature",   offsetof(ipc::Command, sample_tokens.temperature) - CMD_PAY, sizeof(ipc::Command::sample_tokens.temperature)},
            {"top_p",         offsetof(ipc::Command, sample_tokens.top_p) - CMD_PAY, sizeof(ipc::Command::sample_tokens.top_p)},
            {"random_seed",   offsetof(ipc::Command, sample_tokens.random_seed) - CMD_PAY, sizeof(ipc::Command::sample_tokens.random_seed)},
        }, sizeof(ipc::Command::sample_tokens));

        result["RmsnormPayload"] = make({
            {"num_tokens",    offsetof(ipc::Command, rmsnorm.num_tokens) - CMD_PAY, sizeof(ipc::Command::rmsnorm.num_tokens)},
            {"input_buf_id",  offsetof(ipc::Command, rmsnorm.input_buf_id) - CMD_PAY, sizeof(ipc::Command::rmsnorm.input_buf_id)},
            {"output_buf_id", offsetof(ipc::Command, rmsnorm.output_buf_id) - CMD_PAY, sizeof(ipc::Command::rmsnorm.output_buf_id)},
            {"weight_buf_id", offsetof(ipc::Command, rmsnorm.weight_buf_id) - CMD_PAY, sizeof(ipc::Command::rmsnorm.weight_buf_id)},
            {"eps",           offsetof(ipc::Command, rmsnorm.eps) - CMD_PAY, sizeof(ipc::Command::rmsnorm.eps)},
            {"hidden_size",   offsetof(ipc::Command, rmsnorm.hidden_size) - CMD_PAY, sizeof(ipc::Command::rmsnorm.hidden_size)},
        }, sizeof(ipc::Command::rmsnorm));
        result["SwigluPayload"] = make({
            {"num_tokens",    offsetof(ipc::Command, swiglu.num_tokens) - CMD_PAY, sizeof(ipc::Command::swiglu.num_tokens)},
            {"hidden_dim",    offsetof(ipc::Command, swiglu.hidden_dim) - CMD_PAY, sizeof(ipc::Command::swiglu.hidden_dim)},
            {"input_buf_id",  offsetof(ipc::Command, swiglu.input_buf_id) - CMD_PAY, sizeof(ipc::Command::swiglu.input_buf_id)},
            {"output_buf_id", offsetof(ipc::Command, swiglu.output_buf_id) - CMD_PAY, sizeof(ipc::Command::swiglu.output_buf_id)},
        }, sizeof(ipc::Command::swiglu));
        result["MoePermutePayload"] = make({
            {"num_tokens",       offsetof(ipc::Command, moe_permute.num_tokens) - CMD_PAY, sizeof(ipc::Command::moe_permute.num_tokens)},
            {"num_experts",      offsetof(ipc::Command, moe_permute.num_experts) - CMD_PAY, sizeof(ipc::Command::moe_permute.num_experts)},
            {"input_buf_id",     offsetof(ipc::Command, moe_permute.input_buf_id) - CMD_PAY, sizeof(ipc::Command::moe_permute.input_buf_id)},
            {"output_buf_id",    offsetof(ipc::Command, moe_permute.output_buf_id) - CMD_PAY, sizeof(ipc::Command::moe_permute.output_buf_id)},
            {"indices_buf_id",   offsetof(ipc::Command, moe_permute.indices_buf_id) - CMD_PAY, sizeof(ipc::Command::moe_permute.indices_buf_id)},
            {"offsets_buf_id",   offsetof(ipc::Command, moe_permute.offsets_buf_id) - CMD_PAY, sizeof(ipc::Command::moe_permute.offsets_buf_id)},
            {"topk",             offsetof(ipc::Command, moe_permute.topk) - CMD_PAY, sizeof(ipc::Command::moe_permute.topk)},
            {"hidden_dim",       offsetof(ipc::Command, moe_permute.hidden_dim) - CMD_PAY, sizeof(ipc::Command::moe_permute.hidden_dim)},
            {"workspace_buf_id", offsetof(ipc::Command, moe_permute.workspace_buf_id) - CMD_PAY, sizeof(ipc::Command::moe_permute.workspace_buf_id)},
        }, sizeof(ipc::Command::moe_permute));
        result["MoeUnpermutePayload"] = make({
            {"num_tokens",     offsetof(ipc::Command, moe_unpermute.num_tokens) - CMD_PAY, sizeof(ipc::Command::moe_unpermute.num_tokens)},
            {"num_experts",    offsetof(ipc::Command, moe_unpermute.num_experts) - CMD_PAY, sizeof(ipc::Command::moe_unpermute.num_experts)},
            {"input_buf_id",   offsetof(ipc::Command, moe_unpermute.input_buf_id) - CMD_PAY, sizeof(ipc::Command::moe_unpermute.input_buf_id)},
            {"output_buf_id",  offsetof(ipc::Command, moe_unpermute.output_buf_id) - CMD_PAY, sizeof(ipc::Command::moe_unpermute.output_buf_id)},
            {"indices_buf_id", offsetof(ipc::Command, moe_unpermute.indices_buf_id) - CMD_PAY, sizeof(ipc::Command::moe_unpermute.indices_buf_id)},
            {"topk",           offsetof(ipc::Command, moe_unpermute.topk) - CMD_PAY, sizeof(ipc::Command::moe_unpermute.topk)},
            {"hidden_dim",     offsetof(ipc::Command, moe_unpermute.hidden_dim) - CMD_PAY, sizeof(ipc::Command::moe_unpermute.hidden_dim)},
            {"weights_buf_id", offsetof(ipc::Command, moe_unpermute.weights_buf_id) - CMD_PAY, sizeof(ipc::Command::moe_unpermute.weights_buf_id)},
        }, sizeof(ipc::Command::moe_unpermute));
        result["DcpCorrectionPayload"] = make({
            {"batch_size",    offsetof(ipc::Command, dcp_correction.batch_size) - CMD_PAY, sizeof(ipc::Command::dcp_correction.batch_size)},
            {"input_buf_id",  offsetof(ipc::Command, dcp_correction.input_buf_id) - CMD_PAY, sizeof(ipc::Command::dcp_correction.input_buf_id)},
            {"output_buf_id", offsetof(ipc::Command, dcp_correction.output_buf_id) - CMD_PAY, sizeof(ipc::Command::dcp_correction.output_buf_id)},
            {"lse_buf_id",    offsetof(ipc::Command, dcp_correction.lse_buf_id) - CMD_PAY, sizeof(ipc::Command::dcp_correction.lse_buf_id)},
        }, sizeof(ipc::Command::dcp_correction));
        result["NcclAllreducePayload"] = make({
            {"count",  offsetof(ipc::Command, nccl_allreduce.count) - CMD_PAY, sizeof(ipc::Command::nccl_allreduce.count)},
            {"buf_id", offsetof(ipc::Command, nccl_allreduce.buf_id) - CMD_PAY, sizeof(ipc::Command::nccl_allreduce.buf_id)},
            {"dtype",  offsetof(ipc::Command, nccl_allreduce.dtype) - CMD_PAY, sizeof(ipc::Command::nccl_allreduce.dtype)},
        }, sizeof(ipc::Command::nccl_allreduce));
        result["DynamicFp8QuantPayload"] = make({
            {"num_tokens",    offsetof(ipc::Command, dynamic_fp8_quant.num_tokens) - CMD_PAY, sizeof(ipc::Command::dynamic_fp8_quant.num_tokens)},
            {"hidden_dim",    offsetof(ipc::Command, dynamic_fp8_quant.hidden_dim) - CMD_PAY, sizeof(ipc::Command::dynamic_fp8_quant.hidden_dim)},
            {"input_buf_id",  offsetof(ipc::Command, dynamic_fp8_quant.input_buf_id) - CMD_PAY, sizeof(ipc::Command::dynamic_fp8_quant.input_buf_id)},
            {"output_buf_id", offsetof(ipc::Command, dynamic_fp8_quant.output_buf_id) - CMD_PAY, sizeof(ipc::Command::dynamic_fp8_quant.output_buf_id)},
            {"scales_buf_id", offsetof(ipc::Command, dynamic_fp8_quant.scales_buf_id) - CMD_PAY, sizeof(ipc::Command::dynamic_fp8_quant.scales_buf_id)},
        }, sizeof(ipc::Command::dynamic_fp8_quant));
        result["PrescopePayload"] = make({
            {"target_layer_idx",    offsetof(ipc::Command, prescope.target_layer_idx) - CMD_PAY, sizeof(ipc::Command::prescope.target_layer_idx)},
            {"num_tokens",          offsetof(ipc::Command, prescope.num_tokens) - CMD_PAY, sizeof(ipc::Command::prescope.num_tokens)},
            {"hidden_state_buf_id", offsetof(ipc::Command, prescope.hidden_state_buf_id) - CMD_PAY, sizeof(ipc::Command::prescope.hidden_state_buf_id)},
            {"output_buf_id",       offsetof(ipc::Command, prescope.output_buf_id) - CMD_PAY, sizeof(ipc::Command::prescope.output_buf_id)},
        }, sizeof(ipc::Command::prescope));
        result["ProbePayload"] = make({
            {"probe_point_idx",     offsetof(ipc::Command, probe.probe_point_idx) - CMD_PAY, sizeof(ipc::Command::probe.probe_point_idx)},
            {"num_tokens",          offsetof(ipc::Command, probe.num_tokens) - CMD_PAY, sizeof(ipc::Command::probe.num_tokens)},
            {"hidden_state_buf_id", offsetof(ipc::Command, probe.hidden_state_buf_id) - CMD_PAY, sizeof(ipc::Command::probe.hidden_state_buf_id)},
            {"output_buf_id",       offsetof(ipc::Command, probe.output_buf_id) - CMD_PAY, sizeof(ipc::Command::probe.output_buf_id)},
        }, sizeof(ipc::Command::probe));
        result["GraphPayload"] = make({
            {"graph_type", offsetof(ipc::Command, graph.graph_type) - CMD_PAY, sizeof(ipc::Command::graph.graph_type)},
            {"batch_size", offsetof(ipc::Command, graph.batch_size) - CMD_PAY, sizeof(ipc::Command::graph.batch_size)},
        }, sizeof(ipc::Command::graph));
        result["EventPayload"] = make({
            {"event_id", offsetof(ipc::Command, event.event_id) - CMD_PAY, sizeof(ipc::Command::event.event_id)},
        }, sizeof(ipc::Command::event));
        result["AffinityHintsPayload"] = make({
            {"num_gpus",           offsetof(ipc::Command, affinity_hints.num_gpus) - CMD_PAY, sizeof(ipc::Command::affinity_hints.num_gpus)},
            {"gpu_capacity_slots", offsetof(ipc::Command, affinity_hints.gpu_capacity_slots) - CMD_PAY, sizeof(ipc::Command::affinity_hints.gpu_capacity_slots)},
        }, sizeof(ipc::Command::affinity_hints));
        result["NumaMigratePayload"] = make({
            {"layer_idx",        offsetof(ipc::Command, numa_migrate.layer_idx) - CMD_PAY, sizeof(ipc::Command::numa_migrate.layer_idx)},
            {"expert_idx",       offsetof(ipc::Command, numa_migrate.expert_idx) - CMD_PAY, sizeof(ipc::Command::numa_migrate.expert_idx)},
            {"target_numa_node", offsetof(ipc::Command, numa_migrate.target_numa_node) - CMD_PAY, sizeof(ipc::Command::numa_migrate.target_numa_node)},
        }, sizeof(ipc::Command::numa_migrate));
        result["NvmeReadPayload"] = make({
            {"layer_idx",  offsetof(ipc::Command, nvme_read.layer_idx) - CMD_PAY, sizeof(ipc::Command::nvme_read.layer_idx)},
            {"expert_idx", offsetof(ipc::Command, nvme_read.expert_idx) - CMD_PAY, sizeof(ipc::Command::nvme_read.expert_idx)},
            {"gpu_hint",   offsetof(ipc::Command, nvme_read.gpu_hint) - CMD_PAY, sizeof(ipc::Command::nvme_read.gpu_hint)},
        }, sizeof(ipc::Command::nvme_read));
        result["NvmeWritePayload"] = make({
            {"layer_idx",  offsetof(ipc::Command, nvme_write.layer_idx) - CMD_PAY, sizeof(ipc::Command::nvme_write.layer_idx)},
            {"expert_idx", offsetof(ipc::Command, nvme_write.expert_idx) - CMD_PAY, sizeof(ipc::Command::nvme_write.expert_idx)},
        }, sizeof(ipc::Command::nvme_write));
        result["NvmeEvictHostPayload"] = make({
            {"layer_idx",  offsetof(ipc::Command, nvme_evict_host.layer_idx) - CMD_PAY, sizeof(ipc::Command::nvme_evict_host.layer_idx)},
            {"expert_idx", offsetof(ipc::Command, nvme_evict_host.expert_idx) - CMD_PAY, sizeof(ipc::Command::nvme_evict_host.expert_idx)},
        }, sizeof(ipc::Command::nvme_evict_host));
        result["CancelTransferPayload"] = make({
            {"target_cmd_seq", offsetof(ipc::Command, cancel_transfer.target_cmd_seq) - CMD_PAY, sizeof(ipc::Command::cancel_transfer.target_cmd_seq)},
        }, sizeof(ipc::Command::cancel_transfer));
        result["PrefetchBatchPayload"] = make({
            {"count",    offsetof(ipc::Command, prefetch_batch.count) - CMD_PAY, sizeof(ipc::Command::prefetch_batch.count)},
            {"priority", offsetof(ipc::Command, prefetch_batch.priority) - CMD_PAY, sizeof(ipc::Command::prefetch_batch.priority)},
            {"delay_us", offsetof(ipc::Command, prefetch_batch.delay_us) - CMD_PAY, sizeof(ipc::Command::prefetch_batch.delay_us)},
        }, sizeof(ipc::Command::prefetch_batch));
        result["EvictBatchPayload"] = make({
            {"count", offsetof(ipc::Command, evict_batch.count) - CMD_PAY, sizeof(ipc::Command::evict_batch.count)},
        }, sizeof(ipc::Command::evict_batch));
        result["NvmeBatchReadPayload"] = make({
            {"count", offsetof(ipc::Command, nvme_batch_read.count) - CMD_PAY, sizeof(ipc::Command::nvme_batch_read.count)},
        }, sizeof(ipc::Command::nvme_batch_read));
        result["PrefetchExpertPayload"] = make({
            {"layer_idx",  offsetof(ipc::Command, prefetch_expert.layer_idx) - CMD_PAY, sizeof(ipc::Command::prefetch_expert.layer_idx)},
            {"expert_idx", offsetof(ipc::Command, prefetch_expert.expert_idx) - CMD_PAY, sizeof(ipc::Command::prefetch_expert.expert_idx)},
            {"zone",       offsetof(ipc::Command, prefetch_expert.zone) - CMD_PAY, sizeof(ipc::Command::prefetch_expert.zone)},
            {"gpu_idx",    offsetof(ipc::Command, prefetch_expert.gpu_idx) - CMD_PAY, sizeof(ipc::Command::prefetch_expert.gpu_idx)},
            {"priority",   offsetof(ipc::Command, prefetch_expert.priority) - CMD_PAY, sizeof(ipc::Command::prefetch_expert.priority)},
            {"delay_us",   offsetof(ipc::Command, prefetch_expert.delay_us) - CMD_PAY, sizeof(ipc::Command::prefetch_expert.delay_us)},
        }, sizeof(ipc::Command::prefetch_expert));
        result["EvictToHostPayload"] = make({
            {"layer_idx",  offsetof(ipc::Command, evict_to_host.layer_idx) - CMD_PAY, sizeof(ipc::Command::evict_to_host.layer_idx)},
            {"expert_idx", offsetof(ipc::Command, evict_to_host.expert_idx) - CMD_PAY, sizeof(ipc::Command::evict_to_host.expert_idx)},
            {"gpu_idx",    offsetof(ipc::Command, evict_to_host.gpu_idx) - CMD_PAY, sizeof(ipc::Command::evict_to_host.gpu_idx)},
        }, sizeof(ipc::Command::evict_to_host));
        result["StageExpertPayload"] = make({
            {"layer_idx",  offsetof(ipc::Command, stage_expert.layer_idx) - CMD_PAY, sizeof(ipc::Command::stage_expert.layer_idx)},
            {"expert_idx", offsetof(ipc::Command, stage_expert.expert_idx) - CMD_PAY, sizeof(ipc::Command::stage_expert.expert_idx)},
            {"zone",       offsetof(ipc::Command, stage_expert.zone) - CMD_PAY, sizeof(ipc::Command::stage_expert.zone)},
            {"gpu_idx",    offsetof(ipc::Command, stage_expert.gpu_idx) - CMD_PAY, sizeof(ipc::Command::stage_expert.gpu_idx)},
            {"priority",   offsetof(ipc::Command, stage_expert.priority) - CMD_PAY, sizeof(ipc::Command::stage_expert.priority)},
            {"delay_us",   offsetof(ipc::Command, stage_expert.delay_us) - CMD_PAY, sizeof(ipc::Command::stage_expert.delay_us)},
        }, sizeof(ipc::Command::stage_expert));
        result["RunPrefetchProbePayload"] = make({
            {"target_layer", offsetof(ipc::Command, run_prefetch_probe.target_layer) - CMD_PAY, sizeof(ipc::Command::run_prefetch_probe.target_layer)},
            {"num_tokens",   offsetof(ipc::Command, run_prefetch_probe.num_tokens) - CMD_PAY, sizeof(ipc::Command::run_prefetch_probe.num_tokens)},
            {"probe_points", offsetof(ipc::Command, run_prefetch_probe.probe_points) - CMD_PAY, sizeof(ipc::Command::run_prefetch_probe.probe_points)},
        }, sizeof(ipc::Command::run_prefetch_probe));
        result["RunAdapterForwardPayload"] = make({
            {"num_tokens",             offsetof(ipc::Command, run_adapter_forward.num_tokens) - CMD_PAY, sizeof(ipc::Command::run_adapter_forward.num_tokens)},
            {"adapter_weights_buf_id", offsetof(ipc::Command, run_adapter_forward.adapter_weights_buf_id) - CMD_PAY, sizeof(ipc::Command::run_adapter_forward.adapter_weights_buf_id)},
        }, sizeof(ipc::Command::run_adapter_forward));
        result["SlowEvictToHostPayload"] = make({
            {"layer_idx",  offsetof(ipc::Command, slow_evict_to_host.layer_idx) - CMD_PAY, sizeof(ipc::Command::slow_evict_to_host.layer_idx)},
            {"expert_idx", offsetof(ipc::Command, slow_evict_to_host.expert_idx) - CMD_PAY, sizeof(ipc::Command::slow_evict_to_host.expert_idx)},
            {"gpu_idx",    offsetof(ipc::Command, slow_evict_to_host.gpu_idx) - CMD_PAY, sizeof(ipc::Command::slow_evict_to_host.gpu_idx)},
        }, sizeof(ipc::Command::slow_evict_to_host));

        result["RunAttentionPayload"] = make({
            {"layer_idx",       offsetof(ipc::Command, run_attention.layer_idx) - CMD_PAY, sizeof(ipc::Command::run_attention.layer_idx)},
            {"num_seqs",        offsetof(ipc::Command, run_attention.num_seqs) - CMD_PAY, sizeof(ipc::Command::run_attention.num_seqs)},
            {"is_prefill",      offsetof(ipc::Command, run_attention.is_prefill) - CMD_PAY, sizeof(ipc::Command::run_attention.is_prefill)},
            {"use_graph",       offsetof(ipc::Command, run_attention.use_graph) - CMD_PAY, sizeof(ipc::Command::run_attention.use_graph)},
            {"is_draft",        offsetof(ipc::Command, run_attention.is_draft) - CMD_PAY, sizeof(ipc::Command::run_attention.is_draft)},
            {"emit_checkpoint", offsetof(ipc::Command, run_attention.emit_checkpoint) - CMD_PAY, sizeof(ipc::Command::run_attention.emit_checkpoint)},
            {"chunk_start",     offsetof(ipc::Command, run_attention.chunk_start) - CMD_PAY, sizeof(ipc::Command::run_attention.chunk_start)},
            {"chunk_len",       offsetof(ipc::Command, run_attention.chunk_len) - CMD_PAY, sizeof(ipc::Command::run_attention.chunk_len)},
            {"emit_gating",     offsetof(ipc::Command, run_attention.emit_gating) - CMD_PAY, sizeof(ipc::Command::run_attention.emit_gating)},
            {"store_gating",    offsetof(ipc::Command, run_attention.store_gating) - CMD_PAY, sizeof(ipc::Command::run_attention.store_gating)},
            {"superchunk",      offsetof(ipc::Command, run_attention.superchunk) - CMD_PAY, sizeof(ipc::Command::run_attention.superchunk)},
            {"row_offset",      offsetof(ipc::Command, run_attention.row_offset) - CMD_PAY, sizeof(ipc::Command::run_attention.row_offset)},
        }, sizeof(ipc::Command::run_attention));

        result["RunMoePayload"] = make({
            {"layer_idx",                 offsetof(ipc::Command, run_moe.layer_idx) - CMD_PAY, sizeof(ipc::Command::run_moe.layer_idx)},
            {"num_seqs",                  offsetof(ipc::Command, run_moe.num_seqs) - CMD_PAY, sizeof(ipc::Command::run_moe.num_seqs)},
            {"moe_mode",                  offsetof(ipc::Command, run_moe.moe_mode) - CMD_PAY, sizeof(ipc::Command::run_moe.moe_mode)},
            {"apply_residual_correction", offsetof(ipc::Command, run_moe.apply_residual_correction) - CMD_PAY, sizeof(ipc::Command::run_moe.apply_residual_correction)},
            {"store_gating_output",       offsetof(ipc::Command, run_moe.store_gating_output) - CMD_PAY, sizeof(ipc::Command::run_moe.store_gating_output)},
            {"emit_checkpoint",           offsetof(ipc::Command, run_moe.emit_checkpoint) - CMD_PAY, sizeof(ipc::Command::run_moe.emit_checkpoint)},
        }, sizeof(ipc::Command::run_moe));

        result["SeqCreatePayload"] = make({
            {"seq_id",     offsetof(ipc::Command, seq_create.seq_id) - CMD_PAY, sizeof(ipc::Command::seq_create.seq_id)},
            {"prompt_len", offsetof(ipc::Command, seq_create.prompt_len) - CMD_PAY, sizeof(ipc::Command::seq_create.prompt_len)},
            {"pool",       offsetof(ipc::Command, seq_create.pool) - CMD_PAY, sizeof(ipc::Command::seq_create.pool)},
        }, sizeof(ipc::Command::seq_create));

        result["SeqFreePayload"] = make({
            {"seq_id", offsetof(ipc::Command, seq_free.seq_id) - CMD_PAY, sizeof(ipc::Command::seq_free.seq_id)},
        }, sizeof(ipc::Command::seq_free));

        result["SeqForkPayload"] = make({
            {"src_seq_id", offsetof(ipc::Command, seq_fork.src_seq_id) - CMD_PAY, sizeof(ipc::Command::seq_fork.src_seq_id)},
            {"dst_seq_id", offsetof(ipc::Command, seq_fork.dst_seq_id) - CMD_PAY, sizeof(ipc::Command::seq_fork.dst_seq_id)},
        }, sizeof(ipc::Command::seq_fork));

        result["MtpStepPayload"] = make({
            {"mtp_layer_idx",  offsetof(ipc::Command, run_mtp_step.mtp_layer_idx) - CMD_PAY, sizeof(ipc::Command::run_mtp_step.mtp_layer_idx)},
            {"seq_id",         offsetof(ipc::Command, run_mtp_step.seq_id) - CMD_PAY, sizeof(ipc::Command::run_mtp_step.seq_id)},
            {"input_token_id", offsetof(ipc::Command, run_mtp_step.input_token_id) - CMD_PAY, sizeof(ipc::Command::run_mtp_step.input_token_id)},
            {"step_idx",       offsetof(ipc::Command, run_mtp_step.step_idx) - CMD_PAY, sizeof(ipc::Command::run_mtp_step.step_idx)},
        }, sizeof(ipc::Command::run_mtp_step));

        result["SelfSpecForwardPayload"] = make({
            {"seq_id",              offsetof(ipc::Command, self_spec_forward.seq_id) - CMD_PAY, sizeof(ipc::Command::self_spec_forward.seq_id)},
            {"input_token_id",      offsetof(ipc::Command, self_spec_forward.input_token_id) - CMD_PAY, sizeof(ipc::Command::self_spec_forward.input_token_id)},
            {"draft_expert_count",  offsetof(ipc::Command, self_spec_forward.draft_expert_count) - CMD_PAY, sizeof(ipc::Command::self_spec_forward.draft_expert_count)},
            {"apply_residual_corr", offsetof(ipc::Command, self_spec_forward.apply_residual_corr) - CMD_PAY, sizeof(ipc::Command::self_spec_forward.apply_residual_corr)},
            {"store_gating",        offsetof(ipc::Command, self_spec_forward.store_gating) - CMD_PAY, sizeof(ipc::Command::self_spec_forward.store_gating)},
            {"step_idx",            offsetof(ipc::Command, self_spec_forward.step_idx) - CMD_PAY, sizeof(ipc::Command::self_spec_forward.step_idx)},
            {"skip_mask_lo",        offsetof(ipc::Command, self_spec_forward.skip_mask_lo) - CMD_PAY, sizeof(ipc::Command::self_spec_forward.skip_mask_lo)},
            {"skip_mask_hi",        offsetof(ipc::Command, self_spec_forward.skip_mask_hi) - CMD_PAY, sizeof(ipc::Command::self_spec_forward.skip_mask_hi)},
        }, sizeof(ipc::Command::self_spec_forward));

        result["ReefRoutePayload"] = make({
            {"layer_idx",    offsetof(ipc::Command, reef_route.layer_idx) - CMD_PAY, sizeof(ipc::Command::reef_route.layer_idx)},
            {"expert_count", offsetof(ipc::Command, reef_route.expert_count) - CMD_PAY, sizeof(ipc::Command::reef_route.expert_count)},
        }, sizeof(ipc::Command::reef_route));

        result["FarForwardLayerPayload"] = make({
            {"layer_idx",   offsetof(ipc::Command, far_forward_layer.layer_idx) - CMD_PAY, sizeof(ipc::Command::far_forward_layer.layer_idx)},
            {"num_seqs",    offsetof(ipc::Command, far_forward_layer.num_seqs) - CMD_PAY, sizeof(ipc::Command::far_forward_layer.num_seqs)},
            {"chunk_start", offsetof(ipc::Command, far_forward_layer.chunk_start) - CMD_PAY, sizeof(ipc::Command::far_forward_layer.chunk_start)},
            {"chunk_len",   offsetof(ipc::Command, far_forward_layer.chunk_len) - CMD_PAY, sizeof(ipc::Command::far_forward_layer.chunk_len)},
            {"timeout_us",  offsetof(ipc::Command, far_forward_layer.timeout_us) - CMD_PAY, sizeof(ipc::Command::far_forward_layer.timeout_us)},
            {"is_prefill",  offsetof(ipc::Command, far_forward_layer.is_prefill) - CMD_PAY, sizeof(ipc::Command::far_forward_layer.is_prefill)},
            {"route_mode",  offsetof(ipc::Command, far_forward_layer.route_mode) - CMD_PAY, sizeof(ipc::Command::far_forward_layer.route_mode)},
        }, sizeof(ipc::Command::far_forward_layer));

        result["ForwardOneLayerPayload"] = make({
            {"layer_idx",     offsetof(ipc::Command, forward_one_layer.layer_idx) - CMD_PAY, sizeof(ipc::Command::forward_one_layer.layer_idx)},
            {"num_seqs",      offsetof(ipc::Command, forward_one_layer.num_seqs) - CMD_PAY, sizeof(ipc::Command::forward_one_layer.num_seqs)},
            {"topk_override", offsetof(ipc::Command, forward_one_layer.topk_override) - CMD_PAY, sizeof(ipc::Command::forward_one_layer.topk_override)},
            {"is_prefill",    offsetof(ipc::Command, forward_one_layer.is_prefill) - CMD_PAY, sizeof(ipc::Command::forward_one_layer.is_prefill)},
            {"use_graph",     offsetof(ipc::Command, forward_one_layer.use_graph) - CMD_PAY, sizeof(ipc::Command::forward_one_layer.use_graph)},
            {"is_draft",      offsetof(ipc::Command, forward_one_layer.is_draft) - CMD_PAY, sizeof(ipc::Command::forward_one_layer.is_draft)},
            {"store_gating",  offsetof(ipc::Command, forward_one_layer.store_gating) - CMD_PAY, sizeof(ipc::Command::forward_one_layer.store_gating)},
        }, sizeof(ipc::Command::forward_one_layer));

        result["ConfigUpdatePayload"] = make({
            {"count", offsetof(ipc::Command, config_update.count) - CMD_PAY, sizeof(ipc::Command::config_update.count)},
        }, sizeof(ipc::Command::config_update));

        // ── Completion payload structs ──
        constexpr size_t CMP_PAY = offsetof(ipc::Completion, raw);  // = 16

        result["ComputeCompletionPayload"] = make({
            {"cmd_type",        offsetof(ipc::Completion, compute.cmd_type) - CMP_PAY, sizeof(ipc::Completion::compute.cmd_type)},
            {"layer_idx",       offsetof(ipc::Completion, compute.layer_idx) - CMP_PAY, sizeof(ipc::Completion::compute.layer_idx)},
            {"host_buf_offset", offsetof(ipc::Completion, compute.host_buf_offset) - CMP_PAY, sizeof(ipc::Completion::compute.host_buf_offset)},
            {"data_bytes",      offsetof(ipc::Completion, compute.data_bytes) - CMP_PAY, sizeof(ipc::Completion::compute.data_bytes)},
            {"top1_prob",       offsetof(ipc::Completion, compute.top1_prob) - CMP_PAY, sizeof(ipc::Completion::compute.top1_prob)},
            {"entropy",         offsetof(ipc::Completion, compute.entropy) - CMP_PAY, sizeof(ipc::Completion::compute.entropy)},
        }, sizeof(ipc::Completion::compute));

        result["SeqOpCompletionPayload"] = make({
            {"seq_id",     offsetof(ipc::Completion, seq_op.seq_id) - CMP_PAY, sizeof(ipc::Completion::seq_op.seq_id)},
            {"page_count", offsetof(ipc::Completion, seq_op.page_count) - CMP_PAY, sizeof(ipc::Completion::seq_op.page_count)},
        }, sizeof(ipc::Completion::seq_op));

        result["ErrorCompletionPayload"] = make({
            {"error_category", offsetof(ipc::Completion, error.error_category) - CMP_PAY, sizeof(ipc::Completion::error.error_category)},
            {"message",        offsetof(ipc::Completion, error.message) - CMP_PAY, sizeof(ipc::Completion::error.message)},
        }, sizeof(ipc::Completion::error));

        result["TransferCompletionPayload"] = make({
            {"layer_idx",    offsetof(ipc::Completion, transfer.layer_idx) - CMP_PAY, sizeof(ipc::Completion::transfer.layer_idx)},
            {"expert_idx",   offsetof(ipc::Completion, transfer.expert_idx) - CMP_PAY, sizeof(ipc::Completion::transfer.expert_idx)},
            {"direction",    offsetof(ipc::Completion, transfer.direction) - CMP_PAY, sizeof(ipc::Completion::transfer.direction)},
            {"vram_address", offsetof(ipc::Completion, transfer.vram_address) - CMP_PAY, sizeof(ipc::Completion::transfer.vram_address)},
        }, sizeof(ipc::Completion::transfer));
        result["NvmeCompletionPayload"] = make({
            {"layer_idx",  offsetof(ipc::Completion, nvme_op.layer_idx) - CMP_PAY, sizeof(ipc::Completion::nvme_op.layer_idx)},
            {"expert_idx", offsetof(ipc::Completion, nvme_op.expert_idx) - CMP_PAY, sizeof(ipc::Completion::nvme_op.expert_idx)},
            {"op",         offsetof(ipc::Completion, nvme_op.op) - CMP_PAY, sizeof(ipc::Completion::nvme_op.op)},
        }, sizeof(ipc::Completion::nvme_op));
        result["CancelCompletionPayload"] = make({
            {"target_cmd_seq", offsetof(ipc::Completion, cancel_result.target_cmd_seq) - CMP_PAY, sizeof(ipc::Completion::cancel_result.target_cmd_seq)},
            {"cancelled",      offsetof(ipc::Completion, cancel_result.cancelled) - CMP_PAY, sizeof(ipc::Completion::cancel_result.cancelled)},
        }, sizeof(ipc::Completion::cancel_result));
        result["ElmExpertCompletionPayload"] = make({
            {"layer_idx",  offsetof(ipc::Completion, elm_expert.layer_idx) - CMP_PAY, sizeof(ipc::Completion::elm_expert.layer_idx)},
            {"expert_idx", offsetof(ipc::Completion, elm_expert.expert_idx) - CMP_PAY, sizeof(ipc::Completion::elm_expert.expert_idx)},
            {"nvme_us",    offsetof(ipc::Completion, elm_expert.nvme_us) - CMP_PAY, sizeof(ipc::Completion::elm_expert.nvme_us)},
            {"dma_us",     offsetof(ipc::Completion, elm_expert.dma_us) - CMP_PAY, sizeof(ipc::Completion::elm_expert.dma_us)},
            {"total_us",   offsetof(ipc::Completion, elm_expert.total_us) - CMP_PAY, sizeof(ipc::Completion::elm_expert.total_us)},
        }, sizeof(ipc::Completion::elm_expert));
        result["ElmProgressCompletionPayload"] = make({
            {"layer_idx",  offsetof(ipc::Completion, elm_progress.layer_idx) - CMP_PAY, sizeof(ipc::Completion::elm_progress.layer_idx)},
            {"expert_idx", offsetof(ipc::Completion, elm_progress.expert_idx) - CMP_PAY, sizeof(ipc::Completion::elm_progress.expert_idx)},
            {"phase",      offsetof(ipc::Completion, elm_progress.phase) - CMP_PAY, sizeof(ipc::Completion::elm_progress.phase)},
        }, sizeof(ipc::Completion::elm_progress));

        result["CheckpointCompletionPayload"] = make({
            {"cmd_type",        offsetof(ipc::Completion, checkpoint.cmd_type) - CMP_PAY, sizeof(ipc::Completion::checkpoint.cmd_type)},
            {"layer_idx",       offsetof(ipc::Completion, checkpoint.layer_idx) - CMP_PAY, sizeof(ipc::Completion::checkpoint.layer_idx)},
            {"checkpoint_type", offsetof(ipc::Completion, checkpoint.checkpoint_type) - CMP_PAY, sizeof(ipc::Completion::checkpoint.checkpoint_type)},
            {"host_buf_offset", offsetof(ipc::Completion, checkpoint.host_buf_offset) - CMP_PAY, sizeof(ipc::Completion::checkpoint.host_buf_offset)},
            {"data_bytes",      offsetof(ipc::Completion, checkpoint.data_bytes) - CMP_PAY, sizeof(ipc::Completion::checkpoint.data_bytes)},
        }, sizeof(ipc::Completion::checkpoint));

        return result;
    }, "C++ IPC command/completion payload layouts");

    // Category 3: IPC sideband entry structs (IPC-8b).
    m.def("ipc_sideband_layout", [make_layout]() -> py::dict {
        auto make = make_layout;
        py::dict result;

        result["BatchDescriptorEntry"] = make({
            {"seq_id",    offsetof(ipc::BatchDescriptorEntry, seq_id), sizeof(ipc::BatchDescriptorEntry::seq_id)},
            {"token_pos", offsetof(ipc::BatchDescriptorEntry, token_pos), sizeof(ipc::BatchDescriptorEntry::token_pos)},
        }, sizeof(ipc::BatchDescriptorEntry));

        result["ExpertPrefetchEntry"] = make({
            {"layer_idx",  offsetof(ipc::ExpertPrefetchEntry, layer_idx), sizeof(ipc::ExpertPrefetchEntry::layer_idx)},
            {"expert_idx", offsetof(ipc::ExpertPrefetchEntry, expert_idx), sizeof(ipc::ExpertPrefetchEntry::expert_idx)},
            {"zone",       offsetof(ipc::ExpertPrefetchEntry, zone), sizeof(ipc::ExpertPrefetchEntry::zone)},
            {"gpu_idx",    offsetof(ipc::ExpertPrefetchEntry, gpu_idx), sizeof(ipc::ExpertPrefetchEntry::gpu_idx)},
        }, sizeof(ipc::ExpertPrefetchEntry));

        result["ExpertEvictionEntry"] = make({
            {"layer_idx",  offsetof(ipc::ExpertEvictionEntry, layer_idx), sizeof(ipc::ExpertEvictionEntry::layer_idx)},
            {"expert_idx", offsetof(ipc::ExpertEvictionEntry, expert_idx), sizeof(ipc::ExpertEvictionEntry::expert_idx)},
            {"gpu_idx",    offsetof(ipc::ExpertEvictionEntry, gpu_idx), sizeof(ipc::ExpertEvictionEntry::gpu_idx)},
        }, sizeof(ipc::ExpertEvictionEntry));

        result["TransferBatchEntry"] = make({
            {"layer_idx",     offsetof(ipc::TransferBatchEntry, layer_idx), sizeof(ipc::TransferBatchEntry::layer_idx)},
            {"expert_idx",    offsetof(ipc::TransferBatchEntry, expert_idx), sizeof(ipc::TransferBatchEntry::expert_idx)},
            {"sub_component", offsetof(ipc::TransferBatchEntry, sub_component), sizeof(ipc::TransferBatchEntry::sub_component)},
            {"zone",          offsetof(ipc::TransferBatchEntry, zone), sizeof(ipc::TransferBatchEntry::zone)},
            {"bytes",         offsetof(ipc::TransferBatchEntry, bytes), sizeof(ipc::TransferBatchEntry::bytes)},
        }, sizeof(ipc::TransferBatchEntry));

        result["ReserveBatchEntry"] = make({
            {"layer_idx",    offsetof(ipc::ReserveBatchEntry, layer_idx), sizeof(ipc::ReserveBatchEntry::layer_idx)},
            {"expert_idx",   offsetof(ipc::ReserveBatchEntry, expert_idx), sizeof(ipc::ReserveBatchEntry::expert_idx)},
            {"zone",         offsetof(ipc::ReserveBatchEntry, zone), sizeof(ipc::ReserveBatchEntry::zone)},
            {"is_duplicate", offsetof(ipc::ReserveBatchEntry, is_duplicate), sizeof(ipc::ReserveBatchEntry::is_duplicate)},
        }, sizeof(ipc::ReserveBatchEntry));

        result["NvmeReadEntry"] = make({
            {"layer_idx",  offsetof(ipc::NvmeReadEntry, layer_idx), sizeof(ipc::NvmeReadEntry::layer_idx)},
            {"expert_idx", offsetof(ipc::NvmeReadEntry, expert_idx), sizeof(ipc::NvmeReadEntry::expert_idx)},
            {"gpu_hint",   offsetof(ipc::NvmeReadEntry, gpu_hint), sizeof(ipc::NvmeReadEntry::gpu_hint)},
        }, sizeof(ipc::NvmeReadEntry));

        // Sub-region layout metadata
        py::dict regions;
        regions["batch_descriptor"] = py::make_tuple(
            ipc::IpcLayout::kBatchDescriptorOff, ipc::IpcLayout::kBatchDescriptorSize, ipc::kMaxBatchDescriptors);
        regions["expert_prefetch"] = py::make_tuple(
            ipc::IpcLayout::kExpertPrefetchOff, ipc::IpcLayout::kExpertPrefetchSize, ipc::kMaxExpertPrefetch);
        regions["expert_eviction"] = py::make_tuple(
            ipc::IpcLayout::kExpertEvictionOff, ipc::IpcLayout::kExpertEvictionSize, ipc::kMaxExpertEviction);
        regions["transfer_batch"] = py::make_tuple(
            ipc::IpcLayout::kTransferBatchOff, ipc::IpcLayout::kTransferBatchSize, ipc::kMaxTransferBatch);
        regions["reserve_batch"] = py::make_tuple(
            ipc::IpcLayout::kReserveBatchOff, ipc::IpcLayout::kReserveBatchSize, ipc::kMaxReserveBatch);
        regions["nvme_read"] = py::make_tuple(
            ipc::IpcLayout::kNvmeReadOff, ipc::IpcLayout::kNvmeReadSize, ipc::kMaxNvmeReadBatch);
        regions["token_ids"] = py::make_tuple(
            ipc::IpcLayout::kTokenIdsOff, ipc::IpcLayout::kTokenIdsSize, ipc::kMaxSidebandTokenIds);
        result["_regions"] = regions;
        result["_total_size"] = ipc::IpcLayout::kSidebandTotalSize;

        return result;
    }, "C++ IPC sideband struct layouts (IPC-8b)");

    // ── Expert H2D path counters (keeper52 Python port) ──────────────────
    // Cumulative process-wide counters from Engine::h2d_path_stats() — the
    // SAME source the C++ keepers use for their cache hit/miss x-ray
    // (misses = phase_count = expert H2D transfers; callers take deltas
    // around a measured window). Read-only; empty dict when no engine.
    m.def("h2d_path_stats", []() -> py::dict {
        py::dict d;
        auto* engine = daemon::get_engine();
        if (!engine) return d;
        const ipc::EngineH2dPathStats s = engine->h2d_path_stats();
        d["direct"]        = s.direct;
        d["staged"]        = s.staged;
        d["memcpy_ns"]     = s.memcpy_ns;
        d["dma_wait_ns"]   = s.dma_wait_ns;
        d["phase_count"]   = s.phase_count;
        d["dma_gpu_ns"]    = s.dma_gpu_ns;
        d["dma_gpu_count"] = s.dma_gpu_count;
        return d;
    }, "Expert H2D path counters (misses = phase_count); cumulative — take deltas");

    // ── Expert-cache slot capacities (dsp52 bridge Python port) ──────────
    // Per-GPU stable/streaming slot totals+used from the live ExpertCache —
    // the SAME source the C++ keepers use to size the 13c-2.0 test-side LRU
    // eviction model (GpuLru.capacity = total_slots(g, kStable)). Read-only;
    // empty list when no engine / no expert cache.
    m.def("expert_cache_slots", []() -> py::list {
        py::list out;
        auto* engine = daemon::get_engine();
        if (!engine || !engine->expert_cache()) return out;
        const auto* ec = engine->expert_cache();
        const int ng = engine->info().num_gpus;
        using Z = memory::CacheZone;
        for (int g = 0; g < ng; ++g) {
            py::dict d;
            d["stable_total"]    = ec->total_slots(g, Z::kStable);
            d["stable_used"]     = ec->used_slots(g, Z::kStable);
            d["streaming_total"] = ec->total_slots(g, Z::kStreaming);
            d["streaming_used"]  = ec->used_slots(g, Z::kStreaming);
            out.append(d);
        }
        return out;
    }, "Per-GPU expert-cache slot capacities (stable/streaming, total/used)");
}

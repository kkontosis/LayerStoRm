// C-6 CPU expert offload — Milestone A (eager correctness).
// Part of CommandDispatcher — see command_dispatcher.h.
//
// Wires the host NumaCpuExpertDevice into the routed-MoE dispatch behind two
// runtime gates (default OFF ⇒ the champion path is byte-identical):
//   · LS_CPU_EXPERT       — kill-switch (1 = on).
//   · LS_CPU_EXPERT_FORCE — "<layer>:<e0,e1,..>[;<layer>:...]" the forced set of
//                            (layer, expert) the CPU computes (bypasses the
//                            loader solver for the first correctness boot).
//
// A forced (layer,expert) is EXCLUDED from every GPU's resident bitset (never
// fetched / locked / transferred — see the short-circuits in dispatch_moe.cpp),
// computed on host directly from the pinned expert arena (zero H2D of weights),
// and its reduced-bf16 [num_tokens, hidden] moe_output contribution is folded
// into every TP rank's moe_output AFTER the EP combine and BEFORE the Phase-3
// residual add (so all TP ranks stay identical). LOSSLESS parity of the CPU
// kernel vs the GPU GGUF mmvq path is gated separately by LS_CPU_EXPERT_LOSSLESS
// inside NumaCpuExpertDevice::gguf_grouped_gemm.

#include "daemon/command_dispatcher.h"
#include "daemon/dispatch_detail.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "compute/stream_manager.h"
#include "core/device_backend.h"
#include "core/expert_device.h"
#include "compute/kernels/elementwise/residual_add.h"
#include "model/quantization/gguf_kquant.h"
#include "model/quantization/quant_interface.h"

namespace layerstorm::daemon {

// ── Gate resolution ─────────────────────────────────────────────────────────

bool CommandDispatcher::cpu_expert_enabled() {
    if (cpu_expert_state_ < 0) {
        const char* v = std::getenv("LS_CPU_EXPERT");
        cpu_expert_state_ = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    if (cpu_expert_state_ == 0) return false;
    if (!cpu_forced_parsed_) {
        cpu_forced_parsed_ = true;
        // Parse LS_CPU_EXPERT_FORCE = "<layer>:<e0,e1,..>[;<layer>:...]".
        const char* raw = std::getenv("LS_CPU_EXPERT_FORCE");
        if (raw && raw[0]) {
            std::string s(raw);
            size_t i = 0;
            while (i < s.size()) {
                size_t semi = s.find(';', i);
                std::string chunk = s.substr(i, semi - i);
                i = (semi == std::string::npos) ? s.size() : semi + 1;
                size_t colon = chunk.find(':');
                if (colon == std::string::npos || colon == 0) continue;
                uint32_t layer =
                    static_cast<uint32_t>(std::strtoul(chunk.c_str(), nullptr, 10));
                std::string es = chunk.substr(colon + 1);
                auto& vec = cpu_forced_experts_[layer];
                size_t j = 0;
                while (j < es.size()) {
                    size_t comma = es.find(',', j);
                    std::string tok = es.substr(j, comma - j);
                    j = (comma == std::string::npos) ? es.size() : comma + 1;
                    if (tok.empty()) continue;
                    vec.push_back(static_cast<uint16_t>(
                        std::strtoul(tok.c_str(), nullptr, 10)));
                }
            }
        }
        if (!cpu_forced_experts_.empty()) {
            int total = 0;
            for (auto& [l, v] : cpu_forced_experts_) total += static_cast<int>(v.size());
            spdlog::warn("LS_CPU_EXPERT ON: forced CPU expert set = {} (layer,expert) "
                         "pair(s) across {} layer(s) (bypasses the loader solver; "
                         "excluded from every GPU bitset)",
                         total, cpu_forced_experts_.size());
        } else {
            spdlog::warn("LS_CPU_EXPERT=1 but LS_CPU_EXPERT_FORCE is empty — no expert "
                         "will be offloaded to CPU (champion path unchanged)");
        }
    }
    return true;
}

bool CommandDispatcher::cpu_expert_overlap_enabled() {
    // C-6 Milestone C: default ON (host FFN ∥ GPU replay). =0 ⇒ serial/exposed.
    if (cpu_expert_overlap_state_ < 0) {
        const char* v = std::getenv("LS_CPU_EXPERT_OVERLAP");
        cpu_expert_overlap_state_ = (v && v[0] == '0' && v[1] == '\0') ? 0 : 1;
        spdlog::warn("LS_CPU_EXPERT_OVERLAP {}: forced-CPU host FFN {} the GPU "
                     "FFN replay/emit",
                     cpu_expert_overlap_state_ ? "ON (default)" : "OFF",
                     cpu_expert_overlap_state_ ? "OVERLAPS" : "SERIALIZES behind");
    }
    return cpu_expert_overlap_state_ == 1;
}

bool CommandDispatcher::cpu_expert_early_kick_enabled() {
    // C-6 early-kick: default ON (=1). =0 ⇒ Milestone-C late kick (Phase-1).
    if (cpu_expert_early_kick_state_ < 0) {
        const char* v = std::getenv("LS_CPU_EXPERT_EARLY_KICK");
        cpu_expert_early_kick_state_ = (v && v[0] == '0' && v[1] == '\0') ? 0 : 1;
        spdlog::warn("LS_CPU_EXPERT_EARLY_KICK {}: forced-CPU host FFN kicked {}",
                     cpu_expert_early_kick_state_ ? "ON (default)" : "OFF",
                     cpu_expert_early_kick_state_
                         ? "at FETCH-LAUNCH (overlaps the H2D fetch window)"
                         : "at Phase-1 (Milestone-C, overlaps GEMM/allreduce only)");
    }
    return cpu_expert_early_kick_state_ == 1;
}

// C-6 early-kick: mirror the finalize's dispatch_moe_all_ranks fold-format choice
// so the early produce and the finalize fold agree on cpu_fold_moe_o_'s layout.
//   perslot = deterministic_ep_combine_ && ep-within-tp && !dense && !chunked
//   bf16    = ep_combine_bf16_payload_
// ep-within-tp is data-dependent at finalize (per-rank resident bitsets differ),
// but for the only config where deterministic_ep_combine_ is armed — sharded EP,
// where each expert lives on exactly one rank — it is structurally always true on
// a routed (non-dense) layer. A mismatch (replicated-EP fallback) is caught and
// reconciled at finalize (join+re-produce), so this predictor need only be right
// on the common path.
void CommandDispatcher::predict_cpu_fold_format(uint32_t layer_idx, int num_tokens,
                                                bool& perslot, bool& bf16) const {
    bf16 = ep_combine_bf16_payload_;
    perslot = false;
    if (!deps_.live_config) return;
    const int first_k_dense = deps_.live_config->model.first_k_dense_replace;
    const bool is_dense = static_cast<int>(layer_idx) < first_k_dense;
    const bool chunked = deps_.cuda_kernels_enabled
                         && num_tokens > moe_chunk_capacity_;
    perslot = deterministic_ep_combine_ && !is_dense && !chunked;
}

bool CommandDispatcher::cpu_layer_has_forced(uint32_t layer_idx) {
    if (!cpu_expert_enabled()) return false;
    auto it = cpu_forced_experts_.find(layer_idx);
    return it != cpu_forced_experts_.end() && !it->second.empty();
}

bool CommandDispatcher::is_cpu_forced(uint32_t layer_idx, uint16_t expert_idx) {
    if (!cpu_expert_enabled()) return false;
    auto it = cpu_forced_experts_.find(layer_idx);
    if (it == cpu_forced_experts_.end()) return false;
    for (uint16_t e : it->second)
        if (e == expert_idx) return true;
    return false;
}

compute::ExpertDevice* CommandDispatcher::cpu_expert_device() {
    if (!cpu_expert_dev_resolved_) {
        cpu_expert_dev_resolved_ = true;
        for (auto* dev : deps_.expert_devices) {
            if (dev && dev->gpu().type == config::GpuType::cpu) {
                cpu_expert_dev_ = dev;
                break;
            }
        }
        if (!cpu_expert_dev_)
            spdlog::error("LS_CPU_EXPERT ON but NO cpu ExpertDevice built — "
                          "config hardware.cpu_expert_devices[] must be set. "
                          "Forced experts will be DROPPED (routed miss).");
    }
    return cpu_expert_dev_;
}

// ── Graph-hoist overlap: input-ready event ──────────────────────────────────

void CommandDispatcher::record_cpu_input_event(uint32_t gpu, int num_tokens,
                                               uint32_t layer_idx, void* stream) {
    // C-6 Milestone C: record for ALL batch sizes — the input-ready event is what
    // lets the fold's input D2H overlap the FFN work on BOTH the B=1 captured
    // decode replay and the B>1 (verify-chunk) eager emit. Only meaningful when
    // overlap is ON (the serial arm drains the device instead of waiting on it).
    (void)num_tokens;
    if (!deps_.cuda_kernels_enabled || !cpu_expert_overlap_enabled()
        || !cpu_layer_has_forced(layer_idx))
        return;
    if (gpu >= deps_.device_backends.size() || !deps_.device_backends[gpu]
        || !stream)
        return;
    if (cpu_input_ready_event_.size() <= gpu)
        cpu_input_ready_event_.resize(gpu + 1, nullptr);
    if (!cpu_input_ready_event_[gpu])
        cpu_input_ready_event_[gpu] = deps_.device_backends[gpu]->create_event();
    if (cpu_input_ready_event_[gpu])
        deps_.device_backends[gpu]->record_event(cpu_input_ready_event_[gpu],
                                                 stream);
}

// ── Host compute + fold ─────────────────────────────────────────────────────

void CommandDispatcher::join_cpu_fold_worker() {
    if (cpu_fold_worker_.joinable())
        cpu_fold_worker_.join();
    cpu_fold_worker_kicked_ = false;
}

// C-6 Milestone C: KICK the host FFN on a worker thread so it overlaps the GPU
// FFN GEMMs + EP allreduce window (called right after Phase-1 enqueue). No-op
// unless overlap is ON and the layer has forced experts. The produce runs on the
// worker; fold_cpu_forced_experts joins it and does the H2D tail on the daemon.
void CommandDispatcher::start_cpu_forced_experts(
        uint32_t layer_idx, int num_tokens,
        const std::vector<int>& gpu_positions,
        bool perslot, bool bf16_payload) {
    if (!cpu_expert_overlap_enabled()) return;           // serial ⇒ produce at fold
    if (!cpu_layer_has_forced(layer_idx)) return;
    if (!deps_.cuda_kernels_enabled || !deps_.stream_manager || !deps_.live_config)
        return;
    if (gpu_positions.empty() || num_tokens <= 0) return;
    const int rank0 = gpu_positions[0];
    join_cpu_fold_worker();          // never leave a prior worker joinable
    cpu_fold_worker_ok_    = false;
    cpu_fold_worker_layer_ = layer_idx;
    cpu_fold_worker_kicked_ = true;
    cpu_fold_worker_ = std::thread(
        [this, layer_idx, num_tokens, rank0, perslot, bf16_payload] {
            cpu_fold_worker_ok_ =
                cpu_forced_produce(layer_idx, num_tokens, rank0, perslot,
                                   bf16_payload);
        });
}

bool CommandDispatcher::fold_cpu_forced_experts(
        uint32_t layer_idx, int num_tokens,
        const std::vector<int>& gpu_positions,
        bool perslot, bool bf16_payload) {
    if (!cpu_layer_has_forced(layer_idx)) return true;   // fast no-op
    if (!deps_.cuda_kernels_enabled || !deps_.stream_manager || !deps_.live_config)
        return true;
    if (gpu_positions.empty() || num_tokens <= 0) return true;

    const bool overlap = cpu_expert_overlap_enabled();
    if (overlap && cpu_fold_worker_kicked_
        && cpu_fold_worker_layer_ == layer_idx) {
        // OVERLAP: the host FFN was kicked after Phase 1 and ran concurrently
        // with the GPU FFN + allreduce. Join it (a memory barrier ⇒ all produce
        // writes visible), then do only the H2D fold tail.
        join_cpu_fold_worker();
        if (!cpu_fold_worker_ok_) return false;
    } else {
        // SERIAL (overlap OFF) or a non-kicked fold path (non-EP finalize /
        // legacy mode-0): produce synchronously on the daemon thread, then fold.
        if (cpu_fold_worker_kicked_) join_cpu_fold_worker();   // drop a stale kick
        if (!cpu_forced_produce(layer_idx, num_tokens, gpu_positions[0], perslot,
                                bf16_payload))
            return false;
    }
    return cpu_forced_fold_h2d(layer_idx, num_tokens, gpu_positions, perslot,
                               bf16_payload);
}

// ── PRODUCE: D2H input + host FFN chain → cpu_fold_moe_o_ (worker or daemon) ──
bool CommandDispatcher::cpu_forced_produce(
        uint32_t layer_idx, int num_tokens, int rank0,
        bool perslot, bool bf16_payload) {
    cpu_fold_contrib_bytes_ = 0;   // no-op sentinel until moe_o is ready
    cpu_fold_contrib_elems_ = 0;
    cpu_fold_active_ = 0;
    if (!cpu_layer_has_forced(layer_idx)) return true;

    compute::ExpertDevice* cpu = cpu_expert_device();
    if (!cpu) return false;                               // configured on, no device

    const auto& forced = cpu_forced_experts_.at(layer_idx);
    const auto& mc = deps_.live_config->model;
    const int hidden       = mc.hidden_size;
    const int intermediate = mc.moe_intermediate_size;
    const int n_experts    = mc.n_routed_experts;
    const int topk         = mc.num_experts_per_tok;
    const int P            = num_tokens * topk;
    if (num_tokens <= 0 || P <= 0) return true;

    if (rank0 < 0 || static_cast<size_t>(rank0) >= moe_scratch_.size()
        || static_cast<size_t>(rank0) >= deps_.device_backends.size()
        || !deps_.device_backends[rank0])
        return false;
    const auto& s0 = moe_scratch_[rank0];
    if (!s0.normalized_hidden || !s0.topk_indices || !s0.topk_weights) {
        spdlog::error("fold_cpu_forced_experts: rank0 gpu {} scratch not ready "
                      "(layer {})", rank0, layer_idx);
        return false;
    }

    // ── D2H rank0's normalized hidden + routing on a SIDE stream ────────────
    // C-6 Task A (graph-hoist overlap): the norm+router that produce the CPU
    // input were enqueued on rank0's kExpertFfn stream BEFORE the captured FFN
    // replay, and an input-ready event was recorded there (dispatch_moe_internal).
    // Ride the kD2hTransfer stream and wait ONLY on that event — NOT the whole
    // device — so this D2H (and the host FFN below) overlaps the in-flight GPU
    // graph replay instead of draining it. We block ONLY until the tiny input
    // D2H lands (a per-device event, µs), never the replay.
    const bool overlap = cpu_expert_overlap_enabled();
    auto* be0 = deps_.device_backends[rank0];
    be0->set_device();
    void* d2h_stream =
        deps_.stream_manager->stream(rank0, compute::StreamId::kD2hTransfer);
    void* input_ready =
        (static_cast<size_t>(rank0) < cpu_input_ready_event_.size())
            ? cpu_input_ready_event_[rank0] : nullptr;
    // Overlap (default): the side D2H waits ONLY on the input-ready event (norm+
    // router) so the host FFN runs concurrently with the in-flight FFN replay/
    // emit. Serial (LS_CPU_EXPERT_OVERLAP=0): drain rank0's device first — the
    // host FFN is then fully EXPOSED behind the GPU work (the A/B baseline).
    if (overlap) {
        if (input_ready)
            be0->stream_wait_event(d2h_stream, input_ready);
    } else {
        be0->synchronize_device();
    }
    std::vector<uint16_t> host_hidden(static_cast<size_t>(num_tokens) * hidden);
    std::vector<int32_t>  host_idx(P);
    std::vector<float>    host_w(P);
    be0->memcpy_d2h_async(host_hidden.data(), s0.normalized_hidden,
                          host_hidden.size() * sizeof(uint16_t), d2h_stream);
    be0->memcpy_d2h_async(host_idx.data(), s0.topk_indices,
                          host_idx.size() * sizeof(int32_t), d2h_stream);
    be0->memcpy_d2h_async(host_w.data(), s0.topk_weights,
                          host_w.size() * sizeof(float), d2h_stream);
    // Wait for ONLY this input D2H to land (event-scoped, not synchronize_device):
    // in steady state the input-ready event has long fired (Phase 1), so the copy
    // completes in µs and the host FFN below runs concurrently with the replay.
    if (cpu_fold_d2h_done_event_ && cpu_fold_d2h_event_dev_ != rank0) {
        deps_.device_backends[cpu_fold_d2h_event_dev_]
            ->destroy_event(cpu_fold_d2h_done_event_);
        cpu_fold_d2h_done_event_ = nullptr;
    }
    if (!cpu_fold_d2h_done_event_) {
        cpu_fold_d2h_done_event_ = be0->create_event();
        cpu_fold_d2h_event_dev_ = rank0;
    }
    if (cpu_fold_d2h_done_event_) {
        be0->record_event(cpu_fold_d2h_done_event_, d2h_stream);
        while (be0->query_event(cpu_fold_d2h_done_event_).status
               == compute::EventStatus::kNotReady) { /* spin: µs */ }
    } else {
        be0->synchronize_device();   // fallback if event creation failed
    }

    // ── Mask routing to the forced experts only (others → sentinel -1) ──────
    // The CPU permute buckets sentinel (-1 / out-of-range) slots past all real
    // experts and marks src_to_dest_map = -1, so the unpermute drops them: the
    // resulting moe_output is EXACTLY the forced experts' weighted contribution
    // with zeros for every token/slot the forced set does not own — mirroring
    // the GPU EP semantics (excluded experts contribute zero on the GPU).
    std::vector<int32_t> masked_idx(P);
    for (int p = 0; p < P; ++p) {
        const int32_t e = host_idx[p];
        bool keep = false;
        if (e >= 0 && e < n_experts)
            for (uint16_t f : forced)
                if (static_cast<int32_t>(f) == e) { keep = true; break; }
        masked_idx[p] = keep ? e : -1;
    }

    // ── Per-projection GGUF quant types + in-slot offsets (this layer) ──────
    auto to_compute = [](model::GgufKQuantType t) -> compute::GgufQuantType {
        return static_cast<compute::GgufQuantType>(static_cast<int>(t));
    };
    model::GgufKQuantType gate_mt = model::GgufKQuantType::Q4_K;
    model::GgufKQuantType up_mt   = model::GgufKQuantType::Q4_K;
    model::GgufKQuantType down_mt = model::GgufKQuantType::Q4_K;
    if (layer_idx < deps_.routed_layer_gguf_types.size()) {
        gate_mt = deps_.routed_layer_gguf_types[layer_idx].gate;
        up_mt   = deps_.routed_layer_gguf_types[layer_idx].up;
        down_mt = deps_.routed_layer_gguf_types[layer_idx].down;
    } else if (deps_.gguf_quant) {
        gate_mt = deps_.gguf_quant->projection_type(model::Projection::gate);
        up_mt   = deps_.gguf_quant->projection_type(model::Projection::up);
        down_mt = deps_.gguf_quant->projection_type(model::Projection::down);
    }
    const int64_t gate_off = 0;
    const int64_t up_off   =
        model::gguf::gguf_packed_bytes(intermediate, hidden, gate_mt);
    const int64_t down_off =
        up_off + model::gguf::gguf_packed_bytes(intermediate, hidden, up_mt);

    // ── Resolve the forced experts' arena slabs (zero H2D — host RAM) ───────
    // active[i] = the forced expert idx with >=1 routed token; the compacted
    // B_ptrs/offsets mirror the GPU permute's per-expert grouping.
    std::vector<const void*> gate_ptrs, up_ptrs, down_ptrs;
    std::vector<int32_t> active_expert;   // sorted-by-index active forced experts
    // Sort forced experts ascending so the compaction below (which follows the
    // permute's ascending-expert row order) stays consistent.
    std::vector<uint16_t> forced_sorted(forced.begin(), forced.end());
    std::sort(forced_sorted.begin(), forced_sorted.end());
    // Count routed tokens per forced expert to decide "active".
    for (uint16_t e : forced_sorted) {
        bool any = false;
        for (int p = 0; p < P; ++p)
            if (masked_idx[p] == static_cast<int32_t>(e)) { any = true; break; }
        if (!any) continue;
        auto hs = resolve_host_source(layer_idx, e);
        if (!hs.ptr) {
            spdlog::error("fold_cpu_forced_experts: no host arena source for "
                          "L{}E{} — forced CPU expert DROPPED", layer_idx, e);
            continue;
        }
        const auto* base = static_cast<const uint8_t*>(hs.ptr);
        gate_ptrs.push_back(base + gate_off);
        up_ptrs.push_back(base + up_off);
        down_ptrs.push_back(base + down_off);
        active_expert.push_back(e);
    }

    // ── CPU FFN chain (host scratch via the CPU device's node-local alloc) ──
    cpu->set_device();
    auto* perm   = static_cast<uint16_t*>(cpu->device_alloc(sizeof(uint16_t) * P * hidden));
    auto* gate_o = static_cast<uint16_t*>(cpu->device_alloc(sizeof(uint16_t) * P * intermediate));
    auto* up_o   = static_cast<uint16_t*>(cpu->device_alloc(sizeof(uint16_t) * P * intermediate));
    auto* gu     = static_cast<uint16_t*>(cpu->device_alloc(sizeof(uint16_t) * P * 2 * intermediate));
    auto* swig   = static_cast<uint16_t*>(cpu->device_alloc(sizeof(uint16_t) * P * intermediate));
    auto* down_o = static_cast<uint16_t*>(cpu->device_alloc(sizeof(uint16_t) * P * hidden));
    // moe_o holds the fold source: per-slot [B, topk, H] (fp32 or bf16 payload)
    // for the canonical pre-combine fold, else the reduced [B, H] bf16.
    // C-6 Task A: moe_o is PERSISTED across layers and read by the async fold
    // H2D — deferred-free. Wait on the previous fold's per-rank consume events
    // (long done in steady state ⇒ no-op) before reusing it, then grow it.
    const size_t contrib_elems = static_cast<size_t>(perslot ? P : num_tokens)
                                 * hidden;
    const size_t payload_elem_bytes = (perslot && !bf16_payload) ? 4u : 2u;
    const size_t contrib_bytes = contrib_elems * payload_elem_bytes;
    if (cpu_fold_moe_o_inflight_) {
        for (size_t g = 0; g < cpu_fold_consume_event_.size(); ++g) {
            void* ev = cpu_fold_consume_event_[g];
            if (!ev || g >= deps_.device_backends.size() || !deps_.device_backends[g])
                continue;
            auto* be = deps_.device_backends[g];
            be->set_device();
            while (be->query_event(ev).status == compute::EventStatus::kNotReady) {}
        }
        cpu_fold_moe_o_inflight_ = false;
    }
    cpu->set_device();
    if (cpu_fold_moe_o_bytes_ < contrib_bytes) {
        if (cpu_fold_moe_o_) cpu->device_free(cpu_fold_moe_o_);
        cpu_fold_moe_o_ = cpu->device_alloc(contrib_bytes);
        cpu_fold_moe_o_bytes_ = cpu_fold_moe_o_ ? contrib_bytes : 0;
    }
    void* moe_o = cpu_fold_moe_o_;
    auto free_all = [&] {
        cpu->device_free(perm);   cpu->device_free(gate_o); cpu->device_free(up_o);
        cpu->device_free(gu);     cpu->device_free(swig);   cpu->device_free(down_o);
    };
    if (!perm || !gate_o || !up_o || !gu || !swig || !down_o || !moe_o) {
        spdlog::error("fold_cpu_forced_experts: CPU host scratch alloc failed");
        free_all();
        return false;
    }

    std::vector<int32_t> offsets(n_experts + 1, 0), s2d(P, 0), pidx(P, 0);
    cpu->moe_permute(perm, offsets.data(), s2d.data(), pidx.data(),
                     host_hidden.data(), masked_idx.data(),
                     num_tokens, topk, hidden, n_experts, 2, nullptr, nullptr);

    // Compacted active offsets: {0, offsets[e0+1], offsets[e1+1], ...}. Because
    // only the forced (active) experts are non-sentinel, their permuted rows
    // start at physical row 0 in ascending-expert order (offsets[e]=0 for the
    // first active expert), so B_ptrs index A_base from row 0 exactly like the
    // GPU grouped GEMM (matches tests/unit/numa_cpu_expert_test run_ffn_chain).
    std::vector<int32_t> active_offsets;
    active_offsets.reserve(active_expert.size() + 1);
    active_offsets.push_back(0);
    for (int32_t e : active_expert)
        active_offsets.push_back(offsets[e + 1]);
    const int active = static_cast<int>(active_expert.size());

    auto run_gemm = [&](void* D, const void* A, std::vector<const void*>& bp,
                        int N, int K, compute::GgufQuantType type) {
        if (active == 0) return;
        compute::GgufGroupedGemmParams p{};
        p.type = type;
        p.num_experts = active;
        p.N = N;
        p.K = K;
        p.total_tokens = active_offsets.back();
        p.A_base = A;
        p.D_base = D;
        p.expert_offsets = active_offsets.data();
        p.B_ptrs = bp.data();
        cpu->gguf_grouped_gemm(p, nullptr, 0, nullptr);
    };
    // C-6 early-kick η: stamp the host FFN compute interval (begin here — the
    // input D2H above is complete — end after the unpermute below). Read by the
    // daemon after the worker join to compute the fetch-window overlap fraction.
    cpu_hf_start_ns_ = xray_now_ns();
    run_gemm(gate_o, perm, gate_ptrs, intermediate, hidden, to_compute(gate_mt));
    run_gemm(up_o,   perm, up_ptrs,   intermediate, hidden, to_compute(up_mt));
    // Interleave gate|up into the [P, 2*I] layout the device's SwiGLU expects.
    for (int p = 0; p < P; ++p)
        for (int j = 0; j < intermediate; ++j) {
            gu[static_cast<size_t>(p) * 2 * intermediate + j] =
                gate_o[static_cast<size_t>(p) * intermediate + j];
            gu[static_cast<size_t>(p) * 2 * intermediate + intermediate + j] =
                up_o[static_cast<size_t>(p) * intermediate + j];
        }
    // V4-4b: model-wide SwiGLU clamp (llama.cpp DEEPSEEK4; 0.0 = off).
    compute::FusedSwigluParams sp{P, intermediate,
                                  static_cast<float>(mc.swiglu_limit)};
    cpu->fused_swiglu(swig, gu, sp, 2, nullptr);
    run_gemm(down_o, swig, down_ptrs, hidden, intermediate, to_compute(down_mt));

    // Weighted unpermute → host fold source.
    //   perslot: per-slot [B, topk, H] contribution c_k = w_k*expert_out_k in
    //            each slot (dropped slots = 0) — bit-identical to the GPU
    //            per-slot combine, joins the canonical fixed-order reduce.
    //   legacy : reduced bf16 [B, H] (host-summed, folded post-combine).
    // (active==0 ⇒ every src_to_dest_map slot is -1 ⇒ moe_o is all zeros.)
    const compute::MoeCombineMode combine_mode =
        perslot ? (bf16_payload ? compute::MoeCombineMode::kPerSlotBf16
                                : compute::MoeCombineMode::kPerSlotFp32)
                : compute::MoeCombineMode::kReducedBf16;
    cpu->moe_unpermute(moe_o, down_o, host_w.data(), s2d.data(),
                       num_tokens, topk, hidden, 2, nullptr, combine_mode);
    cpu_hf_end_ns_ = xray_now_ns();   // C-6 early-kick η: host FFN compute done

    // moe_o (cpu_fold_moe_o_) is ready. Publish the fold metadata for the daemon
    // H2D pass (visible after the worker join's memory barrier) and free the
    // per-call host scratch (perm..down_o, fully consumed by the unpermute above).
    cpu_fold_active_        = active;
    cpu_fold_contrib_elems_ = contrib_elems;
    cpu_fold_contrib_bytes_ = contrib_bytes;
    cpu_fold_fp32_add_      = perslot && !bf16_payload;
    free_all();
    return true;
}

// ── FOLD H2D: the GPU-ordered tail (daemon thread, kExpertFfn stream) ────────
bool CommandDispatcher::cpu_forced_fold_h2d(
        uint32_t layer_idx, int num_tokens,
        const std::vector<int>& gpu_positions,
        bool perslot, bool bf16_payload) {
    if (cpu_fold_contrib_bytes_ == 0 || !cpu_fold_moe_o_)
        return true;                                     // no-op produce
    const bool overlap        = cpu_expert_overlap_enabled();
    void*  moe_o              = cpu_fold_moe_o_;
    const int    active       = cpu_fold_active_;
    const size_t contrib_elems= cpu_fold_contrib_elems_;
    const size_t contrib_bytes= cpu_fold_contrib_bytes_;
    const bool   fp32_add     = cpu_fold_fp32_add_;
    // ── Fold moe_o into every TP rank's combine buffer (same value on all ranks
    //    so the TP replicas stay identical). Enqueued on each rank's kExpertFfn
    //    stream. Ordering per mode:
    //      perslot=true  → target moe_output_fp32 / moe_output_bf16_perslot,
    //        added AFTER the per-slot gather allreduce (forced slots are 0 on
    //        every GPU ⇒ 0 + c_k = c_k exact) and BEFORE the fixed-order combine
    //        reduce → the forced experts ride the SAME canonical reduce.
    //      perslot=false → target moe_output, added AFTER the EP combine and
    //        BEFORE Phase 3 (legacy reduced-bf16 fold). ──────────────────────
    if (cpu_fold_stage_.size() < deps_.device_backends.size()) {
        cpu_fold_stage_.resize(deps_.device_backends.size(), nullptr);
        cpu_fold_stage_bytes_.resize(deps_.device_backends.size(), 0);
    }
    if (cpu_fold_consume_event_.size() < deps_.device_backends.size())
        cpu_fold_consume_event_.resize(deps_.device_backends.size(), nullptr);
    std::vector<int> folded_gpus;
    bool ok = true;
    for (int g : gpu_positions) {
        void* target = nullptr;
        if (g >= 0 && static_cast<size_t>(g) < moe_scratch_.size()
            && static_cast<size_t>(g) < deps_.device_backends.size()
            && deps_.device_backends[g]) {
            target = perslot
                ? (bf16_payload ? moe_scratch_[g].moe_output_bf16_perslot
                                : moe_scratch_[g].moe_output_fp32)
                : moe_scratch_[g].moe_output;
        }
        if (!target) {
            spdlog::error("fold_cpu_forced_experts: rank gpu {} lacks the {} "
                          "fold target buffer (layer {})", g,
                          perslot ? (bf16_payload ? "bf16-perslot" : "fp32-perslot")
                                  : "moe_output",
                          layer_idx);
            ok = false;
            break;
        }
        auto* be = deps_.device_backends[g];
        be->set_device();
        auto* edev = expert_dev(static_cast<uint32_t>(g));
        if (!edev) { ok = false; break; }
        if (cpu_fold_stage_bytes_[g] < contrib_bytes) {
            if (cpu_fold_stage_[g]) edev->device_free(cpu_fold_stage_[g]);
            cpu_fold_stage_[g] = edev->device_alloc(contrib_bytes);
            cpu_fold_stage_bytes_[g] = cpu_fold_stage_[g] ? contrib_bytes : 0;
        }
        if (!cpu_fold_stage_[g]) {
            spdlog::error("fold_cpu_forced_experts: fold staging alloc failed "
                          "(gpu {}, {} B)", g, contrib_bytes);
            ok = false;
            break;
        }
        void* stream =
            deps_.stream_manager->stream(g, compute::StreamId::kExpertFfn);
        be->memcpy_h2d_async(cpu_fold_stage_[g], moe_o, contrib_bytes, stream);
        if (fp32_add)
            compute::launch_add_inplace_f32(target, cpu_fold_stage_[g],
                                            static_cast<int>(contrib_elems),
                                            stream);
        else
            compute::launch_residual_add(target, cpu_fold_stage_[g],
                                         static_cast<int>(contrib_elems), stream);
        // C-6 Task A: record a consume event AFTER the fold H2D so the persisted
        // moe_o free/reuse is deferred to the next layer (no full-device sync).
        if (!cpu_fold_consume_event_[g])
            cpu_fold_consume_event_[g] = be->create_event();
        if (cpu_fold_consume_event_[g])
            be->record_event(cpu_fold_consume_event_[g], stream);
        folded_gpus.push_back(g);
    }
    // C-6 Task A/Milestone C: OVERLAP ⇒ do NOT synchronize_device here — the host
    // FFN already overlapped the GPU FFN + allreduce window, and moe_o is guarded
    // by the per-rank consume events above (waited at the NEXT layer's produce
    // reuse). SERIAL (overlap OFF) ⇒ drain each folded rank now so the fold is a
    // fully-ordered exposed step (the A/B baseline); moe_o is then free to reuse
    // without an inflight wait. (The per-call host scratch was freed in produce.)
    if (!folded_gpus.empty()) {
        if (overlap) {
            cpu_fold_moe_o_inflight_ = true;
        } else {
            for (int g : folded_gpus) {
                auto* be = deps_.device_backends[g];
                be->set_device();
                be->synchronize_device();
            }
            cpu_fold_moe_o_inflight_ = false;
        }
    }

    if (ok) {
        // Throttled instrumentation: log the first engagement, then every fold
        // that ACTUALLY computed a forced expert on CPU (active>0) up to a few
        // times, plus a running total — so the boot proves real host compute,
        // not just that the hook fired on a step where no forced expert routed.
        // File-static (one CommandDispatcher per process) to avoid a header
        // recompile; pure diagnostics, no behavior effect.
        static uint64_t s_calls = 0, s_total = 0;
        static int s_active_logs = 0;
        s_calls += 1;
        s_total += static_cast<uint64_t>(active);
        // ── C-6 early-kick η: host-FFN-∥-fetch overlap fraction ──────────────
        // The fetch window is [win_start (issue_moe_wave), win_end (finalize entry
        // = experts arrived)] — stamped in dispatch_moe_all_ranks BEFORE Phase-1.
        // Overlap the host FFN interval [hf_start,hf_end] (worker-stamped, visible
        // after join) with it; η = overlap / hf_dur. EARLY kick: host FFN ran
        // during the fetch ⇒ η→1. LATE kick: host FFN spawned at Phase-1 (after
        // win_end) ⇒ hf_start > win_end ⇒ η→0. (Falls back to now if unstamped.)
        double eta = 0.0;
        double hf_ms = 0.0, win_ms = 0.0;
        if (active > 0 && cpu_hf_end_ns_ > cpu_hf_start_ns_
            && cpu_fetch_win_start_ns_ > 0) {
            const uint64_t win_end = (cpu_fetch_win_end_ns_ > cpu_fetch_win_start_ns_)
                ? cpu_fetch_win_end_ns_ : xray_now_ns();
            const uint64_t hf_s = cpu_hf_start_ns_, hf_e = cpu_hf_end_ns_;
            const uint64_t w_s = cpu_fetch_win_start_ns_;
            const uint64_t ov_lo = std::max(hf_s, w_s);
            const uint64_t ov_hi = std::min(hf_e, win_end);
            const double hf_dur = static_cast<double>(hf_e - hf_s);
            const double overlap = (ov_hi > ov_lo)
                ? static_cast<double>(ov_hi - ov_lo) : 0.0;
            eta = hf_dur > 0.0 ? overlap / hf_dur : 0.0;
            hf_ms = hf_dur / 1e6;
            win_ms = static_cast<double>(win_end - w_s) / 1e6;
            cpu_eta_sum_ += eta;
            cpu_eta_n_ += 1;
        }
        if (s_calls == 1
            || (active > 0 && (s_active_logs < 8 || cpu_eta_n_ % 25 == 0))) {
            if (active > 0 && s_active_logs < 8) ++s_active_logs;
            spdlog::warn("CPU-EXPERT FOLD ACTIVE: layer {} computed {} forced "
                         "expert(s) on the host CPU device and folded onto {} TP "
                         "rank(s) [B={}] (calls={}, cum_experts={}) "
                         "ETA_OVERLAP={:.3f} host_ffn_ms={:.3f} fetch_win_ms={:.3f} "
                         "eta_mean={:.3f} eta_n={}",
                         layer_idx, active, folded_gpus.size(), num_tokens,
                         s_calls, s_total, eta, hf_ms, win_ms,
                         cpu_eta_n_ ? cpu_eta_sum_ / static_cast<double>(cpu_eta_n_)
                                    : 0.0,
                         cpu_eta_n_);
        }
    }
    return ok;
}

}  // namespace layerstorm::daemon

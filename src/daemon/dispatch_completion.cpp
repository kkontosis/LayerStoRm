// Completion helpers and async compute event polling.
// Part of CommandDispatcher — see command_dispatcher.h.

#include "daemon/command_dispatcher.h"

#include <cstdio>
#include <cstring>

#include <spdlog/spdlog.h>

#include "compute/stream_manager.h"
#include "core/perf_trace.h"  // kComputeGpu reap (finalize GPU-compute timing)
#include "daemon/spsc_ring.h"

namespace layerstorm::daemon {

// ── Async compute event polling (KD-1) ────────────────────────────────────

uint32_t CommandDispatcher::poll_compute_completions() {
    if (pending_compute_.empty()) return 0;

    uint32_t written = 0;

    for (size_t i = pending_compute_.size(); i > 0; --i) {
        auto& pc = pending_compute_[i - 1];
        bool gpu_error = false;
        int gpu_error_code = 0;

        // KD-3c: process pipeline checkpoints in order before final event.
        if (!pc.pipeline_checkpoints.empty()) {
            bool all_emitted = true;
            for (auto& ckpt : pc.pipeline_checkpoints) {
                if (ckpt.emitted) continue;
                auto [status, err] = deps_.stream_manager->query_event(ckpt.cuda_event, pc.gpu_idx);
                if (status == compute::EventStatus::kError) {
                    gpu_error = true;
                    gpu_error_code = err;
                    break;
                }
                if (status == compute::EventStatus::kNotReady) {
                    all_emitted = false;
                    break;
                }
                write_checkpoint_completion(pc.cmd_type, pc.cmd_seq, pc.gpu_idx,
                                            ckpt.layer_idx, ckpt.checkpoint_type,
                                            ckpt.host_buf_offset, ckpt.data_bytes);
                deps_.stream_manager->destroy_event(ckpt.cuda_event, pc.gpu_idx);
                ckpt.cuda_event = nullptr;
                ckpt.emitted = true;
                ++written;
            }
            if (!gpu_error && !all_emitted) continue;
        }

        if (!gpu_error) {
            auto [status, err] = deps_.stream_manager->query_event(pc.cuda_event, pc.gpu_idx);
            if (status == compute::EventStatus::kError) {
                gpu_error = true;
                gpu_error_code = err;
            } else if (status == compute::EventStatus::kNotReady) {
                continue;
            }
        }

        if (gpu_error) {
            spdlog::error("poll_compute_completions: GPU {} fatal error ({}) "
                          "for cmd_seq={}", pc.gpu_idx, gpu_error_code, pc.cmd_seq);
            // Emit CMP_GPU_FATAL once per GPU.
            if (pc.gpu_idx < gpu_fatal_emitted_.size()
                && !gpu_fatal_emitted_[pc.gpu_idx]) {
                gpu_fatal_emitted_[pc.gpu_idx] = true;
                char fatal_msg[108];
                std::snprintf(fatal_msg, sizeof(fatal_msg),
                              "GPU %u device-fatal (vendor err=%d)",
                              pc.gpu_idx, gpu_error_code);
                write_gpu_fatal(pc.gpu_idx, gpu_error_code, fatal_msg);
            }
            char err_msg[80];
            std::snprintf(err_msg, sizeof(err_msg),
                          "GPU fatal error during compute (vendor err=%d)",
                          gpu_error_code);
            write_error(pc.cmd_seq, pc.gpu_idx, ipc::CmpErrorCategory::kGpuFatal, err_msg);
            // Destroy all events owned by this entry.
            if (pc.cuda_event) {
                deps_.stream_manager->destroy_event(pc.cuda_event, pc.gpu_idx);
            }
            if (pc.compute_t_start && pc.gpu_idx < deps_.device_backends.size()) {
                auto* be = deps_.device_backends[pc.gpu_idx];
                be->destroy_event(pc.compute_t_start);
                if (pc.compute_t_end) be->destroy_event(pc.compute_t_end);
            }
            for (auto& ckpt : pc.pipeline_checkpoints) {
                if (ckpt.cuda_event && !ckpt.emitted) {
                    deps_.stream_manager->destroy_event(ckpt.cuda_event, pc.gpu_idx);
                }
            }
            pending_compute_[i - 1] = std::move(pending_compute_.back());
            pending_compute_.pop_back();
            ++written;
            continue;
        }

        // ── kReady: normal completion ──

        // TODO:DEBT TD-14fb: per-GPU staging clobbered if >1 confidence output_head in-flight on same GPU
        if (pc.has_confidence && pc.gpu_idx < confidence_host_staging_.size()) {
            const auto& staging = confidence_host_staging_[pc.gpu_idx];
            pc.top1_prob = staging.top1_prob;
            pc.entropy = staging.entropy;
        }

        if (pc.checkpoint_data) {
            write_checkpoint_completion(pc.cmd_type, pc.cmd_seq, pc.gpu_idx,
                                        pc.layer_idx,
                                        pc.checkpoint_data->checkpoint_type,
                                        pc.checkpoint_data->host_buf_offset,
                                        pc.checkpoint_data->data_bytes);
        }

        write_compute_completion(pc.cmd_type, pc.cmd_seq, pc.gpu_idx,
                                  pc.layer_idx, /*status=*/0,
                                  pc.host_buf_offset, pc.data_bytes,
                                  pc.top1_prob, pc.entropy,
                                  pc.routed_miss_count);

        // Union-aware cache partitioning: the kExpertFfn event is ready, so
        // the GPU is done reading this command's transient (streaming-zone)
        // union experts — release them (metadata-only; skips locked /
        // interested / mid-transfer entries). 0 for every zone=0-only command.
        if (pc.transient_sweep_mask != 0)
            sweep_transient_streaming(pc.transient_sweep_mask);

        deps_.stream_manager->destroy_event(pc.cuda_event, pc.gpu_idx);

        // perf_trace x-ray: finalize GPU-compute time (kComputeGpu). The timing
        // pair was recorded around the dispatch on the kExpertFfn stream; the
        // completion above guarantees both are done, so elapsed is readable now.
        float finalize_gpu_ms = -1.0f;
        if (pc.compute_t_start && pc.compute_t_end
            && pc.gpu_idx < deps_.device_backends.size()) {
            auto* be = deps_.device_backends[pc.gpu_idx];
            const float ms = be->event_elapsed_ms(pc.compute_t_start, pc.compute_t_end);
            if (ms >= 0.0f) {
                finalize_gpu_ms = ms;
                perf_trace::record(perf_trace::kComputeGpu,
                                   static_cast<uint16_t>(pc.gpu_idx), pc.cmd_seq, 0,
                                   static_cast<uint64_t>(static_cast<double>(ms) * 1e6));
            }
            be->destroy_event(pc.compute_t_start);
            be->destroy_event(pc.compute_t_end);
        }

        // LS_ATTN_CHUNK_PROF: one line per chunk-shaped attention command.
        // host = daemon dispatch wall (pre-executor kv-meta/indexer | executor
        // | post residual+gate+export); gpu_chain = kAttention stream wall
        // from dispatch begin to last enqueued op (timing pair above);
        // tail = dispatch end → completion-detect here.
        if (pc.attn_prof) {
            const auto& ap = *pc.attn_prof;
            const float tail_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - ap.dispatch_end).count();
            spdlog::info("[attn-chunk prof] L{} host={:.3f} (pre={:.3f} "
                         "exec={:.3f} post={:.3f}) gpu_chain={:.3f} "
                         "tail={:.3f}",
                         pc.layer_idx,
                         ap.host_pre_ms + ap.host_exec_ms + ap.host_post_ms,
                         ap.host_pre_ms, ap.host_exec_ms, ap.host_post_ms,
                         finalize_gpu_ms, tail_ms);
        }

        // LS_MOE_BIG_XRAY: one clean line per BIG MoE command — the
        // authoritative BIG-path timing (perf_trace pairing is broken there,
        // spec/bloat/PREFILL_FETCH_OVERHEAD.md §3). host issue/enq = daemon
        // time in issue_moe_wave / run_moe_wave_pass; wait = fetch-arrival
        // detection wall; drain = wave GPU-event wall; gpu waves/finalize =
        // kExpertFfn event elapsed (waves overlap the next wave's H2D).
        if (pc.moe_big_xray) {
            const auto& x = *pc.moe_big_xray;
            spdlog::info("[moe-big xray] L{} B={} experts={} waves={} "
                         "fetches={} | host ms: issue={:.1f} wave_enq={:.1f} | "
                         "wait ms: experts={:.1f} drain={:.1f} | gpu ms: "
                         "waves={:.1f} finalize={:.1f} | wall={:.1f}",
                         pc.layer_idx, x.num_seqs, x.experts, x.waves,
                         x.fetches, x.issue_ms, x.enq_ms, x.wait_ms,
                         x.drain_ms, x.wave_gpu_ms, finalize_gpu_ms,
                         x.wall_ms);
        }

        pending_compute_[i - 1] = std::move(pending_compute_.back());
        pending_compute_.pop_back();
        ++written;
    }

    return written;
}

uint32_t CommandDispatcher::pending_compute_count() const {
    return static_cast<uint32_t>(pending_compute_.size());
}

uint32_t CommandDispatcher::pending_compute_count(uint32_t gpu_idx) const {
    uint32_t count = 0;
    for (const auto& pc : pending_compute_) {
        if (pc.gpu_idx == gpu_idx) ++count;
    }
    return count;
}

// ── Completion helpers ─────────────────────────────────────────────────────

void CommandDispatcher::write_cache_completion(
        uint32_t cmd_seq, uint32_t gpu_idx, uint32_t status) {
    ipc::Completion cmp{};
    cmp.cmp_type = ipc::CMP_CACHE_OP_DONE;
    cmp.cmd_seq  = cmd_seq;
    cmp.gpu_idx  = gpu_idx;
    cmp.status   = status;
    if (!deps_.cmp_ring->try_write(&cmp)) {
        spdlog::error("CommandDispatcher: completion ring full (cache op)");
    }
}

// TODO:DEBT TD-38: completion ring try_write silently drops completions on ring-full
void CommandDispatcher::write_compute_completion(
        uint32_t orig_cmd_type, uint32_t cmd_seq,
        uint32_t gpu_idx, uint32_t layer_idx, uint32_t status,
        uint32_t host_buf_offset, uint32_t data_bytes,
        float top1_prob, float entropy,
        uint8_t routed_miss_count) {
    ipc::Completion cmp{};
    cmp.cmp_type                    = ipc::CMP_COMPUTE_DONE;
    cmp.cmd_seq                     = cmd_seq;
    cmp.gpu_idx                     = gpu_idx;
    cmp.status                      = status;
    cmp.compute.cmd_type            = orig_cmd_type;
    cmp.compute.layer_idx           = layer_idx;
    cmp.compute.host_buf_offset     = host_buf_offset;
    cmp.compute.data_bytes          = data_bytes;
    cmp.compute.top1_prob           = top1_prob;
    cmp.compute.entropy             = entropy;
    cmp.compute.routed_miss_count   = routed_miss_count;
    perf_trace::record(perf_trace::kCmpWritten,
                       static_cast<uint16_t>(gpu_idx), cmd_seq,
                       orig_cmd_type, layer_idx);
    if (!deps_.cmp_ring->try_write(&cmp)) {
        spdlog::error("CommandDispatcher: completion ring full (compute)");
    }
}

void CommandDispatcher::write_checkpoint_completion(
        uint32_t orig_cmd_type, uint32_t cmd_seq,
        uint32_t gpu_idx, uint32_t layer_idx,
        uint8_t checkpoint_type,
        uint32_t host_buf_offset, uint32_t data_bytes) {
    ipc::Completion cmp{};
    cmp.cmp_type                   = ipc::CMP_CHECKPOINT;
    cmp.cmd_seq                    = cmd_seq;
    cmp.gpu_idx                    = gpu_idx;
    cmp.status                     = 0;
    cmp.checkpoint.cmd_type        = orig_cmd_type;
    cmp.checkpoint.layer_idx       = layer_idx;
    cmp.checkpoint.checkpoint_type = checkpoint_type;
    cmp.checkpoint.host_buf_offset = host_buf_offset;
    cmp.checkpoint.data_bytes      = data_bytes;
    if (!deps_.cmp_ring->try_write(&cmp)) {
        spdlog::error("CommandDispatcher: completion ring full (checkpoint)");
    }
}

void CommandDispatcher::write_event_completion(
        uint32_t cmd_seq, uint32_t gpu_idx, uint32_t status) {
    ipc::Completion cmp{};
    cmp.cmp_type = ipc::CMP_EVENT_STATUS;
    cmp.cmd_seq  = cmd_seq;
    cmp.gpu_idx  = gpu_idx;
    cmp.status   = status;
    if (!deps_.cmp_ring->try_write(&cmp)) {
        spdlog::error("CommandDispatcher: completion ring full (event)");
    }
}

void CommandDispatcher::write_error(
        uint32_t cmd_seq, uint32_t gpu_idx,
        ipc::CmpErrorCategory category, const char* msg) {
    ipc::Completion cmp{};
    cmp.cmp_type              = ipc::CMP_ERROR;
    cmp.cmd_seq               = cmd_seq;
    cmp.gpu_idx               = gpu_idx;
    cmp.status                = 1;
    cmp.error.error_category  = static_cast<uint32_t>(category);
    // Use caller-provided message, or default from category enum.
    const char* m = msg ? msg : ipc::default_message(category);
    std::strncpy(cmp.error.message, m, sizeof(cmp.error.message) - 1);
    cmp.error.message[sizeof(cmp.error.message) - 1] = '\0';
    if (!deps_.cmp_ring->try_write(&cmp)) {
        spdlog::error("CommandDispatcher: completion ring full (error: {})", m);
    }
}

void CommandDispatcher::write_gpu_fatal(uint32_t gpu_idx, int vendor_error_code,
                                        const char* msg) {
    ipc::Completion cmp{};
    cmp.cmp_type = ipc::CMP_GPU_FATAL;
    cmp.cmd_seq  = 0;
    cmp.gpu_idx  = gpu_idx;
    cmp.status   = 1;
    cmp.gpu_fatal.vendor_error_code = static_cast<uint32_t>(vendor_error_code);
    std::strncpy(cmp.gpu_fatal.message, msg, sizeof(cmp.gpu_fatal.message) - 1);
    cmp.gpu_fatal.message[sizeof(cmp.gpu_fatal.message) - 1] = '\0';
    if (!deps_.cmp_ring->try_write(&cmp)) {
        spdlog::error("CommandDispatcher: completion ring full (gpu_fatal GPU {})",
                      gpu_idx);
    }
}

void CommandDispatcher::write_nvme_completion(
        uint32_t cmd_seq, uint32_t gpu_idx,
        uint32_t layer_idx, uint16_t expert_idx,
        uint8_t op, uint32_t status) {
    ipc::Completion cmp{};
    cmp.cmp_type            = ipc::CMP_NVME_DONE;
    cmp.cmd_seq             = cmd_seq;
    cmp.gpu_idx             = gpu_idx;
    cmp.status              = status;
    cmp.nvme_op.layer_idx   = layer_idx;
    cmp.nvme_op.expert_idx  = expert_idx;
    cmp.nvme_op.op          = op;
    if (!deps_.cmp_ring->try_write(&cmp)) {
        spdlog::error("CommandDispatcher: completion ring full (nvme op)");
    }
}

void CommandDispatcher::write_cancel_completion(
        uint32_t cmd_seq, uint32_t gpu_idx,
        uint32_t target_cmd_seq, uint8_t cancelled, uint32_t status) {
    ipc::Completion cmp{};
    cmp.cmp_type                     = ipc::CMP_CANCEL_DONE;
    cmp.cmd_seq                      = cmd_seq;
    cmp.gpu_idx                      = gpu_idx;
    cmp.status                       = status;
    cmp.cancel_result.target_cmd_seq = target_cmd_seq;
    cmp.cancel_result.cancelled      = cancelled;
    if (!deps_.cmp_ring->try_write(&cmp)) {
        spdlog::error("CommandDispatcher: completion ring full (cancel)");
    }
}

void CommandDispatcher::write_seq_completion(
        uint32_t cmd_seq, uint32_t gpu_idx,
        uint64_t seq_id, uint32_t page_count, uint32_t status) {
    ipc::Completion cmp{};
    cmp.cmp_type          = ipc::CMP_SEQ_OP_DONE;
    cmp.cmd_seq           = cmd_seq;
    cmp.gpu_idx           = gpu_idx;
    cmp.status            = status;
    cmp.seq_op.seq_id     = seq_id;
    cmp.seq_op.page_count = page_count;
    if (!deps_.cmp_ring->try_write(&cmp)) {
        spdlog::error("CommandDispatcher: completion ring full (seq op)");
    }
}

}  // namespace layerstorm::daemon

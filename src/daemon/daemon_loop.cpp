// Daemon thread main loop implementation.
//
// See daemon_loop.h for design overview and per-cycle phase description.

#include "daemon/daemon_loop.h"
#include "daemon/expert_lifecycle_manager.h"
#include "core/perf_trace.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include <spdlog/spdlog.h>

#include "core/memory/nvme_tier.h"
#include "core/transfer/transfer_engine.h"
#include "daemon/ipc_protocol.h"
#include "daemon/spsc_ring.h"
#include "daemon/state_transaction.h"

namespace layerstorm::daemon {

// ── Construction ───────────────────────────────────────────────────────────

DaemonLoop::DaemonLoop(Deps deps)
    : deps_(std::move(deps))
{
    if (deps_.state_snapshot) {
        state_tx_ = std::make_unique<StateTransaction>(
            *deps_.state_snapshot, deps_.writer_yield_interval);
    }

    // CMP_GPU_FATAL once per GPU — sized to max supported GPUs.
    gpu_fatal_emitted_.assign(ipc::kMaxGpus, false);
}

DaemonLoop::~DaemonLoop() = default;

// ── Public API ─────────────────────────────────────────────────────────────

void DaemonLoop::run() {
    // TEMP DEBUG (TD-100j deep x-ray): time each cycle to expose whether the
    // daemon thread blocks (e.g. on synchronous cold arena load_into preads),
    // which would stall poll_completions and inflate H2D completion-detect.
    auto dbg_now_ns = [] {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    };
    uint64_t busy_ns = 0, cyc_max_ns = 0, timed = 0, slow = 0;
    // Adaptive spin: after a transfer/NVMe completion, keep polling without
    // yielding for up to kSpinBudget cycles so H2D completions are detected
    // with minimal latency during active decode (the per-layer demand fetch is
    // on the critical path). When no completion arrives within the budget
    // (NVMe wait, or idle), degrade to yield() so the daemon doesn't burn a
    // core. kSpinBudget is the tunable sweet spot (cycles, ~µs each).
    constexpr uint64_t kSpinBudget = 1u << 20;  // EXPERIMENT: tune
    uint64_t spin_left = 0;
    uint64_t spun = 0, yielded = 0;  // TD x-ray: spin vs yield mix
    while (true) {
        const uint64_t c0 = dbg_now_ns();
        const bool cont = run_one_cycle();
        const uint64_t dt = dbg_now_ns() - c0;
        busy_ns += dt; ++timed;
        if (dt > cyc_max_ns) cyc_max_ns = dt;
        if (dt > 5'000'000ULL) ++slow;   // >5 ms cycle
        if (!cont) break;
        if (cycle_had_completion_) spin_left = kSpinBudget;  // reset on completion
        if (spin_left > 0) { --spin_left; ++spun; }          // busy-poll, no yield
        else { ++yielded; std::this_thread::yield(); }       // quiet → low CPU
    }
    spdlog::info("[spin x-ray] spun={} yielded={} ({:.1f}% spun)",
                 spun, yielded, 100.0 * spun / std::max<uint64_t>(spun + yielded, 1));
    spdlog::info("Daemon thread exiting after {} cycles", cycle_count_);
    spdlog::info("[TD-100j x-ray] daemon cycles timed={} avg={:.5f} ms "
                 "max={:.3f} ms slow(>5ms)={}",
                 timed, busy_ns / 1e6 / std::max<uint64_t>(timed, 1),
                 cyc_max_ns / 1e6, slow);
    if (deps_.elm) deps_.elm->log_h2d_micro();
    // Deferred (cold) dump of the deep fetch-path micro-trace.
    perf_trace::dump_csv("/tmp/perf_trace.csv");
}

bool DaemonLoop::run_one_cycle() {
    if (!check_shutdown()) return false;

    // KD-1: Backpressure — skip drain if compute pipeline is saturated.
    bool compute_saturated = false;
    if (deps_.pending_compute_count_fn) {
        compute_saturated =
            deps_.pending_compute_count_fn() >= deps_.max_inflight_compute;
    }

    if (!compute_saturated) {
        drain_and_dispatch();
    }

    // KD-1: Poll compute completions (frees backpressure slots).
    if (deps_.poll_compute_fn) {
        deps_.poll_compute_fn();
    }

    pump_completions();

    // #90: advance progressive fetch-and-run MoE state machine.
    if (deps_.advance_progressive_fn) {
        deps_.advance_progressive_fn();
    }

    // M3b: background maintenance (arena migrator tick) — near-free when idle.
    if (deps_.background_fn) {
        deps_.background_fn();
    }

    publish_state();

    return deps_.running->load(std::memory_order_relaxed);
}

// ── Lifecycle/transfer completion pump (ELM-5 single-poll-then-route) ──────
// The one place expert ARRIVALS are materialized (transfer/NVMe completions →
// elm->poll → ExpertCache::mark_all_ready). Shared by run_one_cycle and, via
// CommandDispatcher::set_lifecycle_pump, by the in-handler progressive-MoE
// drain (queued FAR/FETCH backpressure) — calling advance_progressive_moe
// alone can never observe an arrival, so any blocking drain MUST pump this.
// Daemon-thread only.
void DaemonLoop::pump_completions() {
    if (deps_.elm) {
        auto xfer = deps_.transfer_engine
                        ? deps_.transfer_engine->poll_completions()
                        : std::vector<transfer::TransferCompletion>{};
        auto nvme = deps_.nvme_tier
                        ? deps_.nvme_tier->poll_completions()
                        : std::vector<memory::IoCompletion>{};

        // Adaptive spin signal: did anything complete this cycle?
        cycle_had_completion_ = !xfer.empty() || !nvme.empty();

        auto result = deps_.elm->poll(xfer, nvme);
        write_elm_completions(result.lifecycle);
        write_elm_progress(result.progress);
        write_transfer_completions(result.unhandled_transfers);
        write_nvme_completions(result.unhandled_nvme);
    } else {
        poll_transfer_completions();
        poll_nvme_completions();
        cycle_had_completion_ = true;  // legacy path: stay responsive (spin)
    }

    // Flush staged transfers (overflow beyond min-dispatch watermark).
    if (deps_.transfer_engine) {
        deps_.transfer_engine->flush_staged();
    }
}

// ── Phase 1: shutdown check ────────────────────────────────────────────────

bool DaemonLoop::check_shutdown() {
    if (deps_.ipc_header->shutdown_requested) {
        deps_.running->store(false, std::memory_order_relaxed);
        return false;
    }
    return true;
}

// ── Phase 2: drain command ring and dispatch ───────────────────────────────

uint32_t DaemonLoop::drain_and_dispatch() {
    return deps_.cmd_ring->drain(
        [this](const uint8_t* slot) {
            const auto* cmd = reinterpret_cast<const ipc::Command*>(slot);
            switch (static_cast<ipc::CmdType>(cmd->cmd_type)) {
                case ipc::CMD_SHUTDOWN:
                    deps_.running->store(false, std::memory_order_relaxed);
                    break;
                case ipc::CMD_NOOP:
                    break;
                default:
                    perf_trace::record(perf_trace::kCmdDequeued,
                                       static_cast<uint16_t>(cmd->gpu_idx),
                                       static_cast<uint32_t>(cmd->cmd_seq),
                                       cmd->cmd_type, 0);
                    if (deps_.dispatch_fn) {
                        deps_.dispatch_fn(*cmd);
                    } else {
                        spdlog::warn("Daemon: unhandled command type 0x{:04X}",
                                     cmd->cmd_type);
                    }
                    break;
            }
        },
        kMaxCommandsPerCycle);
}

// ── Phase 3: poll GPU transfer completions ─────────────────────────────────

uint32_t DaemonLoop::poll_transfer_completions() {
    if (!deps_.transfer_engine) return 0;

    auto completions = deps_.transfer_engine->poll_completions();
    uint32_t written = 0;

    for (const auto& tc : completions) {
        // Emit CMP_GPU_FATAL once per GPU on transfer failure.
        if (!tc.success && tc.gpu_idx >= 0
            && tc.gpu_idx < static_cast<int>(gpu_fatal_emitted_.size())
            && !gpu_fatal_emitted_[tc.gpu_idx]) {
            gpu_fatal_emitted_[tc.gpu_idx] = true;
            ipc::Completion fatal{};
            fatal.cmp_type = ipc::CMP_GPU_FATAL;
            fatal.cmd_seq  = 0;
            fatal.gpu_idx  = static_cast<uint32_t>(tc.gpu_idx);
            fatal.status   = 1;
            fatal.gpu_fatal.vendor_error_code = 0;
            std::snprintf(fatal.gpu_fatal.message,
                          sizeof(fatal.gpu_fatal.message),
                          "GPU %d device-fatal (detected via transfer failure)",
                          tc.gpu_idx);
            deps_.cmp_ring->try_write(&fatal);
        }

        ipc::Completion cmp{};
        cmp.cmp_type  = ipc::CMP_TRANSFER_DONE;
        cmp.cmd_seq   = deps_.cmd_seq_resolver
                             ? deps_.cmd_seq_resolver(tc.token)
                             : 0u;
        cmp.gpu_idx   = static_cast<uint32_t>(tc.gpu_idx);
        cmp.status    = tc.success ? 0u : 1u;

        cmp.transfer.layer_idx  = tc.key.layer_idx;
        cmp.transfer.expert_idx = tc.key.expert_idx;
        cmp.transfer.direction  = static_cast<uint8_t>(tc.direction);

        if (deps_.cmp_ring->try_write(&cmp)) {
            ++written;
            if (deps_.token_cleanup) {
                deps_.token_cleanup(tc.token);
            }
        } else {
            spdlog::error(
                "Daemon: completion ring full, dropping transfer completion "
                "L{}E{} gpu={}", tc.key.layer_idx, tc.key.expert_idx, tc.gpu_idx);
        }
    }

    return written;
}

// ── Phase 3b: poll NVMe tier completions ──────────────────────────────────

uint32_t DaemonLoop::poll_nvme_completions() {
    if (!deps_.nvme_tier) return 0;

    auto completions = deps_.nvme_tier->poll_completions();
    uint32_t written = 0;

    for (const auto& nc : completions) {
        ipc::Completion cmp{};
        cmp.cmp_type  = ipc::CMP_NVME_DONE;
        cmp.cmd_seq   = deps_.nvme_cmd_seq_resolver
                             ? deps_.nvme_cmd_seq_resolver(nc.token)
                             : 0u;
        cmp.gpu_idx   = 0;  // NVMe ops are CPU-side; gpu_idx not meaningful
        cmp.status    = nc.success ? 0u : 1u;

        cmp.nvme_op.layer_idx   = nc.key.layer_idx;
        cmp.nvme_op.expert_idx  = nc.key.expert_idx;
        cmp.nvme_op.op          = static_cast<uint8_t>(nc.op);

        if (deps_.cmp_ring->try_write(&cmp)) {
            ++written;
            if (deps_.nvme_token_cleanup) {
                deps_.nvme_token_cleanup(nc.token);
            }
        } else {
            spdlog::error(
                "Daemon: completion ring full, dropping NVMe completion "
                "L{}E{}", nc.key.layer_idx, nc.key.expert_idx);
        }
    }

    return written;
}

// ── ELM-5: write helpers for routed completions ──────────────────────────

void DaemonLoop::write_transfer_completions(
        std::span<const transfer::TransferCompletion> completions) {
    for (const auto& tc : completions) {
        ipc::Completion cmp{};
        cmp.cmp_type  = ipc::CMP_TRANSFER_DONE;
        cmp.cmd_seq   = deps_.cmd_seq_resolver
                             ? deps_.cmd_seq_resolver(tc.token)
                             : 0u;
        cmp.gpu_idx   = static_cast<uint32_t>(tc.gpu_idx);
        cmp.status    = tc.success ? 0u : 1u;

        cmp.transfer.layer_idx  = tc.key.layer_idx;
        cmp.transfer.expert_idx = tc.key.expert_idx;
        cmp.transfer.direction  = static_cast<uint8_t>(tc.direction);

        if (deps_.cmp_ring->try_write(&cmp)) {
            if (deps_.token_cleanup) deps_.token_cleanup(tc.token);
        } else {
            spdlog::error("Daemon: completion ring full, dropping transfer L{}E{} gpu={}",
                          tc.key.layer_idx, tc.key.expert_idx, tc.gpu_idx);
        }
    }
}

void DaemonLoop::write_nvme_completions(
        std::span<const memory::IoCompletion> completions) {
    for (const auto& nc : completions) {
        ipc::Completion cmp{};
        cmp.cmp_type  = ipc::CMP_NVME_DONE;
        cmp.cmd_seq   = deps_.nvme_cmd_seq_resolver
                             ? deps_.nvme_cmd_seq_resolver(nc.token)
                             : 0u;
        cmp.gpu_idx   = 0;
        cmp.status    = nc.success ? 0u : 1u;

        cmp.nvme_op.layer_idx   = nc.key.layer_idx;
        cmp.nvme_op.expert_idx  = nc.key.expert_idx;
        cmp.nvme_op.op          = static_cast<uint8_t>(nc.op);

        if (deps_.cmp_ring->try_write(&cmp)) {
            if (deps_.nvme_token_cleanup) deps_.nvme_token_cleanup(nc.token);
        } else {
            spdlog::error("Daemon: completion ring full, dropping NVMe L{}E{}",
                          nc.key.layer_idx, nc.key.expert_idx);
        }
    }
}

void DaemonLoop::write_elm_completions(
        std::span<const LifecycleCompletion> completions) {
    for (const auto& lc : completions) {
        ipc::Completion cmp{};
        cmp.cmp_type  = lc.is_eviction ? ipc::CMP_ELM_EXPERT_EVICTED
                                       : ipc::CMP_ELM_EXPERT_READY;
        cmp.cmd_seq   = lc.cmd_seq;
        cmp.gpu_idx   = static_cast<uint32_t>(lc.gpu_idx);
        cmp.status    = lc.success ? 0u : 1u;

        cmp.elm_expert.layer_idx  = lc.key.layer_idx;
        cmp.elm_expert.expert_idx = lc.key.expert_idx;
        cmp.elm_expert.nvme_us    = lc.nvme_us;
        cmp.elm_expert.dma_us     = lc.dma_us;
        cmp.elm_expert.total_us   = lc.total_us;

        if (!deps_.cmp_ring->try_write(&cmp)) {
            // Throttled (TD-ORCH-ELM-COMPLETION-LIVELOCK): first drop + every
            // 100000th — a producer-side retry storm once wrote ~1 GB/35 s of
            // this line.
            if (elm_drop_count_++ % 100000 == 0) {
                spdlog::error("Daemon: completion ring full, dropping ELM "
                              "L{}E{} gpu={} (drop #{})",
                              lc.key.layer_idx, lc.key.expert_idx, lc.gpu_idx,
                              elm_drop_count_);
            }
        }
    }
}

void DaemonLoop::write_elm_progress(
        std::span<const ProgressEvent> events) {
    for (const auto& ev : events) {
        ipc::Completion cmp{};
        cmp.cmp_type = ipc::CMP_ELM_EXPERT_PROGRESS;
        cmp.cmd_seq  = ev.cmd_seq;
        cmp.gpu_idx  = static_cast<uint32_t>(ev.gpu_idx);
        cmp.status   = 0;

        cmp.elm_progress.layer_idx  = ev.key.layer_idx;
        cmp.elm_progress.expert_idx = ev.key.expert_idx;
        cmp.elm_progress.phase      = ev.phase;

        if (!deps_.cmp_ring->try_write(&cmp)) {
            // Throttled like the ELM completion drop above.
            if (elm_progress_drop_count_++ % 100000 == 0) {
                spdlog::error("Daemon: completion ring full, dropping ELM "
                              "progress L{}E{} gpu={} (drop #{})",
                              ev.key.layer_idx, ev.key.expert_idx, ev.gpu_idx,
                              elm_progress_drop_count_);
            }
        }
    }
}

// ── Phase 4: publish state snapshot via transactions ──────────────────────
// Each logical update group is a separate transaction so the seqlock is
// released between groups — gives Python readers even-seqlock windows.
// See spec/IPC_TX.md.

void DaemonLoop::publish_state() {
    ++cycle_count_;

    // #91 / INV-IPC-PUBLISH-THROTTLE: skip the publish when the last one
    // finished less than publish_interval_ns ago.  The interval is measured
    // from publish END (the expert-stats group alone iterates
    // num_moe_layers × 256 keys, ~0.5 ms — measuring from START would
    // re-trigger immediately and keep the seqlock churning).  Python
    // seqlock readers need the writer quiet most of the time; a few-ms
    // stale snapshot is fine for the orchestrator's PLAN phase (ELM
    // residency bits are event-driven micro-transactions, NOT throttled —
    // INV-ELM-7).
    if (deps_.publish_interval_ns > 0 && last_publish_end_ns_ > 0) {
        const auto now = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        if (now - last_publish_end_ns_ < deps_.publish_interval_ns) return;
    }

    // Core fields: cycle count + timestamp (small, fast transaction)
    {
        StateTransactionGuard tx(*state_tx_);
        deps_.state_snapshot->daemon_cycle_count = cycle_count_;
        deps_.state_snapshot->timestamp_ns = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }

    // Module-specific fields: each sub-publish opens its own transactions
    if (deps_.publish_fn) {
        deps_.publish_fn(*deps_.state_snapshot, *state_tx_);
    }

    last_publish_end_ns_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

}  // namespace layerstorm::daemon

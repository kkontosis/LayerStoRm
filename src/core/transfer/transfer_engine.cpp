#include "core/transfer/transfer_engine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include <spdlog/spdlog.h>

#ifdef LAYERSTORM_HAS_NUMA
#include <numaif.h>
#endif

#include "core/device_backend.h"
#include "core/perf_trace.h"

namespace {
// perf_trace source-NUMA x-ray: NUMA node backing a host pointer (the pinned
// arena slot a transfer DMAs from). Profiling-only; one get_mempolicy syscall
// per transfer, called only when tracing. -1 if unknown / NUMA unavailable.
inline int numa_node_of(const void* p) {
#ifdef LAYERSTORM_HAS_NUMA
    int node = -1;
    if (get_mempolicy(&node, nullptr, 0, const_cast<void*>(p),
                      MPOL_F_NODE | MPOL_F_ADDR) != 0)
        return -1;
    return node;
#else
    (void)p;
    return -1;
#endif
}
}  // namespace

namespace { inline uint32_t pt_key(const layerstorm::memory::ExpertKey& k) {
    return (k.layer_idx << 16) | (k.expert_idx & 0xffffu); } }

namespace layerstorm::transfer {

// ── Construction / destruction ──────────────────────────────────────────────

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 481-3 H2D x-ray: nanosecond steady clock for sub-phase timing.
static uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

TransferEngine::TransferEngine(Options opts)
    : device_backends_(std::move(opts.device_backends)),
      min_dispatch_per_gpu_(opts.min_dispatch_per_gpu),
      max_inflight_per_gpu_(opts.max_inflight_per_gpu)
{
    if (device_backends_.empty()) {
        throw std::invalid_argument("TransferEngine: device_backends must not be empty");
    }
    if (!std::is_sorted(opts.priority_bin_thresholds.begin(),
                        opts.priority_bin_thresholds.end())) {
        throw std::invalid_argument(
            "TransferEngine: priority_bin_thresholds must be sorted ascending");
    }
    priority_bin_thresholds_ = std::move(opts.priority_bin_thresholds);

    const int num_gpus = static_cast<int>(device_backends_.size());
    const int num_bins = static_cast<int>(priority_bin_thresholds_.size()) + 1;
    gpus_.resize(num_gpus);
    staged_.resize(num_gpus);
    staged_bin_counts_.assign(num_gpus, std::vector<int>(num_bins, 0));
    for (int i = 0; i < num_gpus; ++i) {
        auto& g = gpus_[i];
        g.h2d_stream = device_backends_[i]->create_stream();
        g.d2h_stream = device_backends_[i]->create_stream();

        if (i < static_cast<int>(opts.pcie_info.size())) {
            g.pcie_gen = opts.pcie_info[i].pcie_gen;
            g.pcie_width = opts.pcie_info[i].pcie_width;
        }
    }

    spdlog::debug("TransferEngine: initialized with {} GPUs, min_dispatch={}, max_inflight={}",
                  num_gpus, min_dispatch_per_gpu_, max_inflight_per_gpu_);
}

TransferEngine::~TransferEngine() {
    drain();
    const int num_gpus = static_cast<int>(gpus_.size());
    for (int i = 0; i < num_gpus; ++i) {
        device_backends_[i]->destroy_stream(gpus_[i].h2d_stream);
        device_backends_[i]->destroy_stream(gpus_[i].d2h_stream);
    }
    // Free pinned staging pool (WP-7b).
    if (pinned_pool_backend_) {
        for (auto& slot : pinned_pool_) {
            if (slot.data) pinned_pool_backend_->host_free_pinned(slot.data);
        }
    }
}

// ── Bin lookup ──────────────────────────────────────────────────────────────

int TransferEngine::bin_for(float priority) const {
    auto it = std::upper_bound(priority_bin_thresholds_.begin(),
                               priority_bin_thresholds_.end(), priority);
    return static_cast<int>(it - priority_bin_thresholds_.begin());
}

// ── Enqueue ─────────────────────────────────────────────────────────────────

std::optional<TransferToken> TransferEngine::enqueue_h2d(
    memory::ExpertKey key, int gpu_idx,
    void* dst, const void* src, int64_t bytes,
    float priority, int delay_us,
    TransferCallback callback,
    bool needs_pinned_staging) {
    return enqueue_impl(TransferDirection::kH2D, key, gpu_idx, dst, src, bytes,
                        priority, delay_us, std::move(callback),
                        needs_pinned_staging);
}

std::optional<TransferToken> TransferEngine::enqueue_d2h(
    memory::ExpertKey key, int gpu_idx,
    void* dst, const void* src, int64_t bytes,
    float priority, int delay_us,
    TransferCallback callback) {
    return enqueue_impl(TransferDirection::kD2H, key, gpu_idx, dst, src, bytes,
                        priority, delay_us, std::move(callback));
}

std::optional<TransferToken> TransferEngine::enqueue_impl(
    TransferDirection dir, memory::ExpertKey key, int gpu_idx,
    void* dst, const void* src, int64_t bytes,
    float priority, int delay_us,
    TransferCallback callback,
    bool needs_pinned_staging) {

    InFlightKey ifk{key, gpu_idx, dir};

    // Dedup: reject if already in-flight or staged.
    if (inflight_.contains(ifk)) {
        return std::nullopt;
    }

    TransferToken token = next_token_++;

    // Hot path: immediate dispatch when below watermark and no delay.
    if (delay_us <= 0 && inflight_count(gpu_idx) < min_dispatch_per_gpu_) {
        // WP-7b: acquire pinned staging slot for mmap sources.
        int slot_idx = -1;
        const void* dma_src = src;
        if (needs_pinned_staging && dir == TransferDirection::kH2D) {
            slot_idx = acquire_pinned_slot();
            if (slot_idx >= 0) {
                auto& slot = pinned_pool_[slot_idx];
                if (static_cast<size_t>(bytes) <= slot.size) {
                    const uint64_t t0 = now_ns();
                    std::memcpy(slot.data, src, static_cast<size_t>(bytes));
                    h2d_phase_stats_.memcpy_ns += now_ns() - t0;  // 481-3 x-ray
                    dma_src = slot.data;
                } else {
                    release_pinned_slot(slot_idx);
                    slot_idx = -1;
                }
            }
            // Pool exhausted or oversized: graceful fallback to unpinned src.
        }

        inflight_[ifk] = token;

        auto& g = gpus_[gpu_idx];
        void* stream = (dir == TransferDirection::kH2D) ? g.h2d_stream : g.d2h_stream;

        // perf_trace pure-DMA x-ray: bracket the memcpy with timing events so
        // cudaEventElapsedTime gives the GPU memcpy time alone (H2D only).
        void* t_start = nullptr;
        void* t_end   = nullptr;
        const bool pt_dma = perf_trace::enabled()
                            && dir == TransferDirection::kH2D;
        void* event = device_backends_[gpu_idx]->create_event();
        if (pt_dma) {
            t_start = device_backends_[gpu_idx]->create_timing_event();
            if (t_start) device_backends_[gpu_idx]->record_event(t_start, stream);
        }
        device_backends_[gpu_idx]->memcpy_async(dst, dma_src, static_cast<size_t>(bytes), stream);
        if (t_start) {
            t_end = device_backends_[gpu_idx]->create_timing_event();
            if (t_end) device_backends_[gpu_idx]->record_event(t_end, stream);
        }
        device_backends_[gpu_idx]->record_event(event, stream);

        pending_[token] = PendingTransfer{
            token, key, gpu_idx, dir, bytes, priority, event,
            std::move(callback), slot_idx};
        pending_[token].dma_enqueue_ns = now_ns();  // 481-3 x-ray
        pending_[token].t_start = t_start;
        pending_[token].t_end   = t_end;
        if (pt_dma) pending_[token].src_numa = numa_node_of(dma_src);
        // cmd_seq field carries the in-flight queue depth at enqueue (backlog proxy).
        perf_trace::record(perf_trace::kEnqueueInline,
                           static_cast<uint16_t>(gpu_idx),
                           static_cast<uint32_t>(pending_.size()),
                           pt_key(key), token);
        return token;
    }

    // Stage for later dispatch.
    inflight_[ifk] = token;
    staged_[gpu_idx].push(StagedTransfer{
        token, key, gpu_idx, dir, dst, src, bytes,
        priority, delay_us, now_us(), std::move(callback),
        needs_pinned_staging});
    staged_token_gpu_[token] = gpu_idx;
    staged_bin_counts_[gpu_idx][bin_for(priority)]++;
    perf_trace::record(perf_trace::kEnqueueStaged,
                       static_cast<uint16_t>(gpu_idx), 0, pt_key(key), token);
    return token;
}

void TransferEngine::dispatch_staged(StagedTransfer& st) {
    // WP-7b: acquire pinned staging slot for mmap sources.
    int slot_idx = -1;
    const void* dma_src = st.src;
    if (st.needs_pinned_staging && st.direction == TransferDirection::kH2D) {
        slot_idx = acquire_pinned_slot();
        if (slot_idx >= 0) {
            auto& slot = pinned_pool_[slot_idx];
            if (static_cast<size_t>(st.bytes) <= slot.size) {
                const uint64_t t0 = now_ns();
                std::memcpy(slot.data, st.src, static_cast<size_t>(st.bytes));
                h2d_phase_stats_.memcpy_ns += now_ns() - t0;  // 481-3 x-ray
                dma_src = slot.data;
            } else {
                release_pinned_slot(slot_idx);
                slot_idx = -1;
            }
        }
    }

    auto& g = gpus_[st.gpu_idx];
    void* stream = (st.direction == TransferDirection::kH2D)
                       ? g.h2d_stream : g.d2h_stream;

    // perf_trace pure-DMA x-ray: bracket the memcpy with timing events (H2D only).
    void* t_start = nullptr;
    void* t_end   = nullptr;
    const bool pt_dma = perf_trace::enabled()
                        && st.direction == TransferDirection::kH2D;
    void* event = device_backends_[st.gpu_idx]->create_event();
    if (pt_dma) {
        t_start = device_backends_[st.gpu_idx]->create_timing_event();
        if (t_start) device_backends_[st.gpu_idx]->record_event(t_start, stream);
    }
    device_backends_[st.gpu_idx]->memcpy_async(st.dst, dma_src, static_cast<size_t>(st.bytes), stream);
    if (t_start) {
        t_end = device_backends_[st.gpu_idx]->create_timing_event();
        if (t_end) device_backends_[st.gpu_idx]->record_event(t_end, stream);
    }
    device_backends_[st.gpu_idx]->record_event(event, stream);
    perf_trace::record(perf_trace::kDispatchStaged,
                       static_cast<uint16_t>(st.gpu_idx),
                       static_cast<uint32_t>(pending_.size()), pt_key(st.key),
                       st.token);

    staged_bin_counts_[st.gpu_idx][bin_for(st.priority)]--;
    pending_[st.token] = PendingTransfer{
        st.token, st.key, st.gpu_idx, st.direction,
        st.bytes, st.priority, event, std::move(st.callback), slot_idx};
    pending_[st.token].dma_enqueue_ns = now_ns();  // 481-3 x-ray
    pending_[st.token].t_start = t_start;
    pending_[st.token].t_end   = t_end;
    if (pt_dma) pending_[st.token].src_numa = numa_node_of(dma_src);
    staged_token_gpu_.erase(st.token);
}

void TransferEngine::replenish_gpu(int gpu_idx) {
    auto& q = staged_[gpu_idx];
    int inflight = 0;
    for (auto& [ifk, _] : inflight_) {
        if (ifk.gpu_idx == gpu_idx && !staged_token_gpu_.contains(_))
            ++inflight;
    }

    int64_t current = now_us();
    while (inflight < min_dispatch_per_gpu_ && !q.empty()) {
        auto st = std::move(const_cast<StagedTransfer&>(q.top()));
        q.pop();
        if (st.delay_us > 0) {
            int64_t elapsed = current - st.enqueue_time_us;
            if (elapsed < st.delay_us) {
                staged_[gpu_idx].push(std::move(st));
                break;
            }
        }
        dispatch_staged(st);
        ++inflight;
    }
}

// ── Staging queue flush ────────────────────────────────────────────────────

void TransferEngine::flush_staged() {
    int64_t current = now_us();
    int num_gpus = static_cast<int>(staged_.size());

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        auto& q = staged_[gpu];
        // Count actually dispatched (non-staged) inflight for this GPU.
        int inflight = 0;
        for (auto& [ifk, tok] : inflight_) {
            if (ifk.gpu_idx == gpu && !staged_token_gpu_.contains(tok))
                ++inflight;
        }

        while (inflight < max_inflight_per_gpu_ && !q.empty()) {
            auto st = std::move(const_cast<StagedTransfer&>(q.top()));
            q.pop();
            if (st.delay_us > 0) {
                int64_t elapsed = current - st.enqueue_time_us;
                if (elapsed < st.delay_us) {
                    staged_[gpu].push(std::move(st));
                    break;
                }
            }
            dispatch_staged(st);
            ++inflight;
        }
    }
}

// ── Completion polling ──────────────────────────────────────────────────────

std::vector<TransferCompletion> TransferEngine::poll_completions() {
    std::vector<TransferCompletion> completed;
    // perf_trace: one tick per poll cycle → reconstructs the daemon's poll
    // cadence offline (bounds detection lag; the long DMA+detect tail lives
    // either here, as poll-interval, or in stream backlog — kDmaGpu separates them).
    perf_trace::record(perf_trace::kPollTick, 0, 0, 0, 0);

    // Collect completed tokens first to avoid iterator invalidation.
    struct DoneToken { TransferToken token; bool success; };
    std::vector<DoneToken> done_tokens;
    for (auto& [token, pt] : pending_) {
        auto [status, err] = device_backends_[pt.gpu_idx]->query_event(pt.event);
        if (status == compute::EventStatus::kReady) {
            done_tokens.push_back({token, true});
        } else if (status == compute::EventStatus::kError) {
            spdlog::error("TransferEngine: GPU {} fatal error ({}) for "
                          "transfer token={}", pt.gpu_idx, err, token);
            done_tokens.push_back({token, false});
        }
    }

    // Track which GPUs had completions for replenishment.
    std::vector<bool> gpu_completed(gpus_.size(), false);

    for (auto& [token, success] : done_tokens) {
        auto it = pending_.find(token);
        auto& pt = it->second;

        // Deferred cancel cleanup (TD-82k/TD-98c): DMA finished, safe to
        // release resources.  No callback, no completion record, no replenish.
        if (pt.cancelled) {
            release_pinned_slot(pt.pinned_slot_idx);
            device_backends_[pt.gpu_idx]->destroy_event(pt.event);
            if (pt.t_start) device_backends_[pt.gpu_idx]->destroy_event(pt.t_start);
            if (pt.t_end)   device_backends_[pt.gpu_idx]->destroy_event(pt.t_end);
            pending_.erase(it);
            continue;
        }

        // 481-3 H2D x-ray: DMA-enqueue → completion-detected latency.
        if (success && pt.direction == TransferDirection::kH2D
            && pt.dma_enqueue_ns != 0) {
            h2d_phase_stats_.dma_wait_ns += now_ns() - pt.dma_enqueue_ns;
            ++h2d_phase_stats_.count;
        }
        // perf_trace pure-DMA x-ray: GPU-measured memcpy time (start→end timing
        // events). Isolates real DMA from host detection/queue lag. Events are
        // complete here (the detection event fired after t_end on the same stream).
        if (pt.t_start && pt.t_end) {
            float ms = device_backends_[pt.gpu_idx]->event_elapsed_ms(
                pt.t_start, pt.t_end);
            if (ms >= 0.0f) {
                uint64_t gpu_ns = static_cast<uint64_t>(ms * 1e6);
                h2d_phase_stats_.dma_gpu_ns += gpu_ns;
                ++h2d_phase_stats_.gpu_count;
                // cmd_seq carries the source NUMA node (+1 so 0 stays "unknown")
                // → offline local/cross split: gpu's node vs source node.
                perf_trace::record(perf_trace::kDmaGpu,
                                   static_cast<uint16_t>(pt.gpu_idx),
                                   static_cast<uint32_t>(pt.src_numa + 1),
                                   pt_key(pt.key), gpu_ns);
            }
            device_backends_[pt.gpu_idx]->destroy_event(pt.t_start);
            device_backends_[pt.gpu_idx]->destroy_event(pt.t_end);
            pt.t_start = pt.t_end = nullptr;
        }
        if (pt.direction == TransferDirection::kH2D)
            perf_trace::record(perf_trace::kDetected,
                               static_cast<uint16_t>(pt.gpu_idx), 0,
                               pt_key(pt.key), token);

        // Fire callback.
        if (pt.callback) {
            pt.callback(pt.token, success);
        }

        // Build completion record.
        completed.push_back(TransferCompletion{
            pt.token, pt.key, pt.gpu_idx, pt.direction, pt.bytes, success});

        gpu_completed[pt.gpu_idx] = true;

        // Release pinned staging buffer (WP-7b).
        release_pinned_slot(pt.pinned_slot_idx);

        // Remove from dedup map.
        InFlightKey ifk{pt.key, pt.gpu_idx, pt.direction};
        inflight_.erase(ifk);

        // Destroy event and remove pending entry.
        device_backends_[pt.gpu_idx]->destroy_event(pt.event);
        pending_.erase(it);
    }

    // Completion-driven replenishment: maintain min-dispatch watermark.
    for (int gpu = 0; gpu < static_cast<int>(gpus_.size()); ++gpu) {
        if (gpu_completed[gpu]) {
            replenish_gpu(gpu);
        }
    }

    return completed;
}

// ── Dedup queries ───────────────────────────────────────────────────────────

bool TransferEngine::is_inflight_h2d(memory::ExpertKey key, int gpu_idx) const {
    return inflight_.contains(InFlightKey{key, gpu_idx, TransferDirection::kH2D});
}

bool TransferEngine::is_inflight_d2h(memory::ExpertKey key, int gpu_idx) const {
    return inflight_.contains(InFlightKey{key, gpu_idx, TransferDirection::kD2H});
}

bool TransferEngine::is_dispatched_h2d(memory::ExpertKey key, int gpu_idx) const {
    // TD-FAR-STREAM-GATE: in-flight AND not sitting in the staged queue —
    // i.e. the memcpy_async is already enqueued on the h2d stream, so a
    // barrier event recorded on that stream fences it.
    auto it = inflight_.find(InFlightKey{key, gpu_idx, TransferDirection::kH2D});
    if (it == inflight_.end()) return false;
    return !staged_token_gpu_.contains(it->second);
}

// ── H2D stream barrier (TD-FAR-STREAM-GATE) ─────────────────────────────────

void* TransferEngine::record_h2d_barrier(int gpu_idx) {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size())) return nullptr;
    // The event is recorded at the CURRENT tail of the h2d stream: it captures
    // every copy dispatched so far and nothing enqueued later. Caller destroys
    // it (device backend of the same GPU) once the stream-wait is enqueued.
    void* ev = device_backends_[gpu_idx]->create_event();
    if (ev)
        device_backends_[gpu_idx]->record_event(ev, gpus_[gpu_idx].h2d_stream);
    return ev;
}

int TransferEngine::inflight_count(int gpu_idx) const {
    int count = 0;
    for (auto& [ifk, _] : inflight_) {
        if (ifk.gpu_idx == gpu_idx) ++count;
    }
    return count;
}

int TransferEngine::inflight_count() const {
    return static_cast<int>(inflight_.size());
}

int TransferEngine::inflight_h2d_count(int gpu_idx) const {
    int count = 0;
    for (auto& [ifk, _] : inflight_) {
        if (ifk.gpu_idx == gpu_idx && ifk.dir == TransferDirection::kH2D) ++count;
    }
    return count;
}

int TransferEngine::inflight_d2h_count(int gpu_idx) const {
    int count = 0;
    for (auto& [ifk, _] : inflight_) {
        if (ifk.gpu_idx == gpu_idx && ifk.dir == TransferDirection::kD2H) ++count;
    }
    return count;
}

// ── Bandwidth estimation ────────────────────────────────────────────────────

double TransferEngine::pcie_bandwidth_gbps(int pcie_gen, int pcie_width) {
    // Per-lane throughput in GB/s (after encoding overhead).
    // PCIe 3.0: 8 GT/s * 128/130 encoding ≈ 0.985 GB/s/lane
    // PCIe 4.0: 16 GT/s * 128/130 encoding ≈ 1.969 GB/s/lane
    // PCIe 5.0: 32 GT/s * 128/130 encoding ≈ 3.938 GB/s/lane
    // PCIe 6.0: 64 GT/s * 128/130 encoding ≈ 7.877 GB/s/lane
    double per_lane = 0.0;
    switch (pcie_gen) {
        case 3: per_lane = 0.985; break;
        case 4: per_lane = 1.969; break;
        case 5: per_lane = 3.938; break;
        case 6: per_lane = 7.877; break;
        default: per_lane = 3.938; break;  // Default to gen5
    }
    return per_lane * pcie_width;
}

double TransferEngine::estimate_transfer_us(int gpu_idx, int64_t bytes) const {
    auto& g = gpus_[gpu_idx];
    double bw = pcie_bandwidth_gbps(g.pcie_gen, g.pcie_width);
    // bandwidth is in GB/s = 1e9 bytes/s. Result in microseconds.
    // us = bytes / (bw * 1e9) * 1e6 = bytes / (bw * 1e3)
    return static_cast<double>(bytes) / (bw * 1e3);
}

// ── Staging queries ────────────────────────────────────────────────────────

int TransferEngine::staged_count(int gpu_idx) const {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(staged_.size())) return 0;
    return static_cast<int>(staged_[gpu_idx].size());
}

int TransferEngine::staged_count() const {
    return static_cast<int>(staged_token_gpu_.size());
}

// ── Priority bin counts ─────────────────────────────────────────────────────

std::vector<int> TransferEngine::pending_bin_counts(int gpu_idx) const {
    const int num_bins = static_cast<int>(priority_bin_thresholds_.size()) + 1;
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(staged_bin_counts_.size())) {
        return std::vector<int>(num_bins, 0);
    }

    // Start with pre-tracked staged counts.
    std::vector<int> counts = staged_bin_counts_[gpu_idx];

    // Add dispatched (in pending_ map).
    for (auto& [token, pt] : pending_) {
        if (pt.gpu_idx == gpu_idx) {
            counts[bin_for(pt.priority)]++;
        }
    }

    return counts;
}

// ── Cancellation ────────────────────────────────────────────────────────────

std::optional<CancelledTransfer> TransferEngine::cancel(TransferToken token) {
    // Check staged queue first — O(1) metadata removal, no orphaned DMA.
    auto staged_it = staged_token_gpu_.find(token);
    if (staged_it != staged_token_gpu_.end()) {
        int gpu_idx = staged_it->second;
        staged_token_gpu_.erase(staged_it);

        // Rebuild the priority queue without the cancelled token.
        auto& q = staged_[gpu_idx];
        std::priority_queue<StagedTransfer> new_q;
        CancelledTransfer result{};
        bool found = false;
        while (!q.empty()) {
            auto st = std::move(const_cast<StagedTransfer&>(q.top()));
            q.pop();
            if (st.token == token && !found) {
                result = CancelledTransfer{st.key, st.gpu_idx, st.direction};
                InFlightKey ifk{st.key, st.gpu_idx, st.direction};
                inflight_.erase(ifk);
                staged_bin_counts_[gpu_idx][bin_for(st.priority)]--;
                found = true;
            } else {
                new_q.push(std::move(st));
            }
        }
        q = std::move(new_q);
        if (found) return result;
    }

    // Check in-flight (already dispatched to CUDA stream).
    auto it = pending_.find(token);
    if (it == pending_.end()) return std::nullopt;

    auto& pt = it->second;
    if (pt.cancelled) return std::nullopt;  // Already cancelled.

    CancelledTransfer result{pt.key, pt.gpu_idx, pt.direction};

    // Mark cancelled — DMA may still be in-flight on the stream.
    // Leave pending entry, event, staging slot, and callback intact so
    // poll_completions() can drain them safely after the DMA finishes.
    // Keeping the callback alive preserves captured host buffer references
    // until the DMA completes (fixes TD-82k + TD-98c).
    pt.cancelled = true;

    InFlightKey ifk{pt.key, pt.gpu_idx, pt.direction};
    inflight_.erase(ifk);

    return result;
}

// ── Priority re-assertion (ELB: demand joins a staged prefetch) ─────────────

bool TransferEngine::reprioritize(TransferToken token, float priority) {
    auto staged_it = staged_token_gpu_.find(token);
    if (staged_it == staged_token_gpu_.end()) return false;  // dispatched/unknown
    const int gpu_idx = staged_it->second;

    // Rebuild the owning GPU's priority queue with the raised priority.
    auto& q = staged_[gpu_idx];
    std::priority_queue<StagedTransfer> new_q;
    bool updated = false;
    while (!q.empty()) {
        auto st = std::move(const_cast<StagedTransfer&>(q.top()));
        q.pop();
        if (st.token == token && priority > st.priority) {
            staged_bin_counts_[gpu_idx][bin_for(st.priority)]--;
            staged_bin_counts_[gpu_idx][bin_for(priority)]++;
            st.priority = priority;
            updated = true;
        }
        new_q.push(std::move(st));
    }
    q = std::move(new_q);
    return updated;
}

// ── Drain ───────────────────────────────────────────────────────────────────

void TransferEngine::drain() {
    // Flush all staged items first (ignore delays).
    for (int gpu = 0; gpu < static_cast<int>(staged_.size()); ++gpu) {
        auto& q = staged_[gpu];
        while (!q.empty()) {
            auto st = std::move(const_cast<StagedTransfer&>(q.top()));
            q.pop();
            dispatch_staged(st);
        }
    }
    while (!pending_.empty()) {
        poll_completions();
    }
}

// ── Pinned staging pool (WP-7b) ────────────────────────────────────────────

void TransferEngine::init_pinned_pool(size_t buffer_size, int count,
                                       compute::DeviceBackend* backend) {
    if (!pinned_pool_.empty() || count <= 0 || buffer_size == 0 || !backend)
        return;

    pinned_pool_backend_ = backend;
    pinned_pool_.resize(count);
    int allocated = 0;
    for (int i = 0; i < count; ++i) {
        void* ptr = backend->host_alloc_pinned(buffer_size);
        if (ptr) {
            pinned_pool_[i] = {ptr, buffer_size, false};
            ++allocated;
        } else {
            spdlog::warn("TransferEngine: pinned staging alloc {} of {} failed",
                         i, count);
        }
    }
    spdlog::info("TransferEngine: pinned staging pool: {} of {} slots, "
                 "{} bytes each ({} MB total)",
                 allocated, count, buffer_size,
                 (static_cast<int64_t>(allocated) * buffer_size) / (1024 * 1024));
}

int TransferEngine::acquire_pinned_slot() {
    for (int i = 0; i < static_cast<int>(pinned_pool_.size()); ++i) {
        if (!pinned_pool_[i].in_use && pinned_pool_[i].data) {
            pinned_pool_[i].in_use = true;
            return i;
        }
    }
    return -1;
}

void TransferEngine::release_pinned_slot(int idx) {
    if (idx >= 0 && idx < static_cast<int>(pinned_pool_.size())) {
        pinned_pool_[idx].in_use = false;
    }
}

int TransferEngine::pinned_pool_free() const {
    int count = 0;
    for (auto& s : pinned_pool_) {
        if (!s.in_use && s.data) ++count;
    }
    return count;
}

int TransferEngine::pinned_pool_size() const {
    return static_cast<int>(pinned_pool_.size());
}

}  // namespace layerstorm::transfer

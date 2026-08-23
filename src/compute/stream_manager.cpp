#include "compute/stream_manager.h"

#include <stdexcept>

#include <spdlog/spdlog.h>

namespace layerstorm::compute {

// ── Construction / destruction ──────────────────────────────────────────────

StreamManager::StreamManager(Options opts)
    : device_backends_(std::move(opts.device_backends))
{
    if (device_backends_.empty()) {
        throw std::invalid_argument("StreamManager: device_backends must not be empty");
    }

    const int num_gpus = static_cast<int>(device_backends_.size());
    gpus_.resize(num_gpus);
    for (int g = 0; g < num_gpus; ++g) {
        for (int s = 0; s < kStreamsPerGpu; ++s) {
            gpus_[g].streams[s] = device_backends_[g]->create_stream();
        }
    }

    spdlog::debug("StreamManager: initialized {} GPUs, {} streams each",
                  num_gpus, kStreamsPerGpu);
}

StreamManager::~StreamManager() {
    const int num_gpus = static_cast<int>(gpus_.size());
    for (int g = 0; g < num_gpus; ++g) {
        for (int s = 0; s < kStreamsPerGpu; ++s) {
            device_backends_[g]->destroy_stream(gpus_[g].streams[s]);
        }
    }
}

// ── Stream access ───────────────────────────────────────────────────────────

void* StreamManager::get_stream(int gpu_idx, StreamId id) const {
    int sid = static_cast<int>(id);
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(gpus_.size())) {
        throw std::out_of_range(
            "StreamManager: invalid gpu_idx " + std::to_string(gpu_idx));
    }
    if (sid < 0 || sid >= kStreamsPerGpu) {
        throw std::out_of_range(
            "StreamManager: invalid StreamId " + std::to_string(sid));
    }
    return gpus_[gpu_idx].streams[sid];
}

void* StreamManager::stream(int gpu_idx, StreamId id) const {
    return get_stream(gpu_idx, id);
}

// ── Event lifecycle ─────────────────────────────────────────────────────────

void* StreamManager::create_event(int gpu_idx) {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(device_backends_.size()))
        throw std::out_of_range(
            "StreamManager::create_event: invalid gpu_idx " + std::to_string(gpu_idx));
    return device_backends_[gpu_idx]->create_event();
}

void StreamManager::destroy_event(void* event, int gpu_idx) {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(device_backends_.size())) {
        spdlog::warn("StreamManager::destroy_event: invalid gpu_idx {} (max {}), event leaked",
                     gpu_idx, static_cast<int>(device_backends_.size()) - 1);
        return;  // graceful no-op (destructor safety)
    }
    device_backends_[gpu_idx]->destroy_event(event);
}

// ── Event operations ────────────────────────────────────────────────────────

void StreamManager::record_event(void* event, int gpu_idx, StreamId stream_id) {
    void* s = get_stream(gpu_idx, stream_id);
    device_backends_[gpu_idx]->record_event(event, s);
}

EventQueryResult StreamManager::query_event(void* event, int gpu_idx) const {
    if (gpu_idx < 0 || gpu_idx >= static_cast<int>(device_backends_.size()))
        return {EventStatus::kError, -1};
    return device_backends_[gpu_idx]->query_event(event);
}

void StreamManager::wait_event(int gpu_idx, StreamId stream_id, void* event) {
    void* s = get_stream(gpu_idx, stream_id);
    device_backends_[gpu_idx]->stream_wait_event(s, event);
}

// ── Data transfer ──────────────────────────────────────────────────────────

void StreamManager::memcpy_d2h_async(void* host_dst, const void* device_src,
                                      size_t bytes, int gpu_idx,
                                      StreamId stream_id) {
    void* s = get_stream(gpu_idx, stream_id);
    device_backends_[gpu_idx]->memcpy_d2h_async(host_dst, device_src, bytes, s);
}

}  // namespace layerstorm::compute

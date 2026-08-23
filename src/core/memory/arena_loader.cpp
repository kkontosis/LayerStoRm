// ArenaLoader (J-1 + Stage 3) — async pinned-arena slot filler. See arena_loader.h.

#include "core/memory/arena_loader.h"

#include <algorithm>

#include <spdlog/spdlog.h>

#include "model/weight_pipeline/prepacked_source.h"

#ifdef LAYERSTORM_HAS_URING
#include <liburing.h>
#endif

namespace layerstorm::memory {

ArenaLoader::ArenaLoader(int num_workers, Backend backend, int queue_depth)
    : backend_(backend) {
#ifdef LAYERSTORM_HAS_URING
    if (backend_ == Backend::kIoUring) {
        if (queue_depth <= 0) queue_depth = 64;
        ring_ = new ::io_uring{};
        int rc = io_uring_queue_init(static_cast<unsigned>(queue_depth), ring_, 0);
        if (rc < 0) {
            spdlog::warn("ArenaLoader: io_uring_queue_init failed ({}); falling "
                         "back to worker pool", rc);
            delete ring_;
            ring_ = nullptr;
            backend_ = Backend::kWorkerPool;
        } else {
            spdlog::info("ArenaLoader: io_uring backend, queue_depth={} (Stage 3)",
                         queue_depth);
            return;  // no worker threads in io_uring mode
        }
    }
#else
    if (backend_ == Backend::kIoUring) {
        spdlog::warn("ArenaLoader: io_uring requested but built without liburing "
                     "(LAYERSTORM_HAS_URING) — using worker pool");
        backend_ = Backend::kWorkerPool;
    }
#endif

    // Worker-pool backend (J-1).
    if (num_workers <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        num_workers = static_cast<int>(std::min(4u, hw == 0 ? 1u : hw));
    }
    num_workers = std::max(1, num_workers);
    workers_.reserve(static_cast<size_t>(num_workers));
    for (int i = 0; i < num_workers; ++i)
        workers_.emplace_back([this] { worker_main(); });
}

ArenaLoader::~ArenaLoader() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_)
        if (t.joinable()) t.join();
#ifdef LAYERSTORM_HAS_URING
    if (ring_) {
        io_uring_queue_exit(ring_);
        delete ring_;
        ring_ = nullptr;
    }
#endif
}

bool ArenaLoader::submit(ExpertKey key, int gpu, void* dst,
                         const model::PrepackedSource* src) {
    if (!dst || !src) return false;

#ifdef LAYERSTORM_HAS_URING
    if (backend_ == Backend::kIoUring) {
        std::lock_guard<std::mutex> lk(mu_);
        if (inflight_.count(key)) return false;        // dedup (as worker pool)
        int fd = -1;
        int64_t off = 0, len = 0;
        if (!src->read_descriptor(key, fd, off, len)) return false;  // need direct fd
        io_uring_sqe* sqe = io_uring_get_sqe(ring_);
        if (!sqe) return false;                         // ring full → caller retries
        const uint64_t token = next_token_++;
        io_uring_prep_read(sqe, fd, dst, static_cast<unsigned>(len),
                           static_cast<__u64>(off));
        io_uring_sqe_set_data64(sqe, token);
        if (io_uring_submit(ring_) < 0) return false;
        uring_inflight_[token] = UringPending{key, gpu, dst, len};
        inflight_.insert(key);
        return true;
    }
#endif

    {
        std::lock_guard<std::mutex> lk(mu_);
        // Dedup: a slot is filled at most once per outstanding load. The arena
        // keeps the slot reserved + `loading`, so resolve() returns "pending"
        // and the daemon never resubmits for the same key.
        if (inflight_.count(key)) return false;
        inflight_.insert(key);
        jobs_.push(Job{key, gpu, dst, src});
    }
    cv_.notify_one();
    return true;
}

void ArenaLoader::worker_main() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
            if (stop_ && jobs_.empty()) return;
            job = jobs_.front();
            jobs_.pop();
        }

        // The expensive part: file→pinned-slot read (mmap memcpy or pread).
        // Runs the synchronous load_into() OFF the daemon thread.
        const bool ok = job.src->load_into(job.key, job.dst);

        {
            std::lock_guard<std::mutex> lk(mu_);
            done_.push_back(ArenaLoadCompletion{job.key, job.gpu, job.dst, ok});
            inflight_.erase(job.key);
        }
    }
}

void ArenaLoader::poll_completed(std::vector<ArenaLoadCompletion>& out) {
    std::lock_guard<std::mutex> lk(mu_);

#ifdef LAYERSTORM_HAS_URING
    if (backend_ == Backend::kIoUring) {
        if (uring_inflight_.empty()) return;
        // Flush kernel completions (non-blocking), then drain the CQE ring.
        io_uring_cqe* cqe = nullptr;
        __kernel_timespec ts{0, 0};
        io_uring_wait_cqe_timeout(ring_, &cqe, &ts);
        while (io_uring_peek_cqe(ring_, &cqe) == 0) {
            const uint64_t token = io_uring_cqe_get_data64(cqe);
            auto it = uring_inflight_.find(token);
            if (it != uring_inflight_.end()) {
                const UringPending& p = it->second;
                const bool ok = (cqe->res >= 0 &&
                                 static_cast<int64_t>(cqe->res) == p.len);
                out.push_back(ArenaLoadCompletion{p.key, p.gpu, p.dst, ok});
                inflight_.erase(p.key);
                uring_inflight_.erase(it);
            }
            io_uring_cqe_seen(ring_, cqe);
        }
        return;
    }
#endif

    if (done_.empty()) return;
    out.insert(out.end(), done_.begin(), done_.end());
    done_.clear();
}

void ArenaLoader::wait_completed(std::vector<ArenaLoadCompletion>& out) {
#ifdef LAYERSTORM_HAS_URING
    if (backend_ == Backend::kIoUring) {
        std::lock_guard<std::mutex> lk(mu_);
        if (uring_inflight_.empty()) return;   // nothing to wait for
        // Block for the first completion (bulk preload: no daemon loop to keep
        // alive, so a real wait beats spinning), then drain the rest non-blocking.
        io_uring_cqe* cqe = nullptr;
        if (io_uring_wait_cqe(ring_, &cqe) == 0 && cqe) {
            const uint64_t token = io_uring_cqe_get_data64(cqe);
            auto it = uring_inflight_.find(token);
            if (it != uring_inflight_.end()) {
                const UringPending& p = it->second;
                const bool ok = (cqe->res >= 0 &&
                                 static_cast<int64_t>(cqe->res) == p.len);
                out.push_back(ArenaLoadCompletion{p.key, p.gpu, p.dst, ok});
                inflight_.erase(p.key);
                uring_inflight_.erase(it);
            }
            io_uring_cqe_seen(ring_, cqe);
        }
        while (io_uring_peek_cqe(ring_, &cqe) == 0) {
            const uint64_t token = io_uring_cqe_get_data64(cqe);
            auto it = uring_inflight_.find(token);
            if (it != uring_inflight_.end()) {
                const UringPending& p = it->second;
                const bool ok = (cqe->res >= 0 &&
                                 static_cast<int64_t>(cqe->res) == p.len);
                out.push_back(ArenaLoadCompletion{p.key, p.gpu, p.dst, ok});
                inflight_.erase(p.key);
                uring_inflight_.erase(it);
            }
            io_uring_cqe_seen(ring_, cqe);
        }
        return;
    }
#endif
    // Worker-pool / non-URING: degrade to a non-blocking drain (preload only takes
    // the async path under kIoUring, so this branch is not used by preload).
    poll_completed(out);
}

bool ArenaLoader::busy(ExpertKey key) const {
    std::lock_guard<std::mutex> lk(mu_);
    return inflight_.count(key) != 0;
}

size_t ArenaLoader::pending_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return inflight_.size();
}

}  // namespace layerstorm::memory

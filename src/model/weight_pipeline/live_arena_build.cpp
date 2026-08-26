#include "model/weight_pipeline/live_arena_build.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

#include "core/memory/numa_manager.h"
#include "core/memory/pinned_expert_arena.h"
#include "model/weight_pipeline/live_gguf_source.h"

namespace layerstorm::model {

LiveBuildStats live_prepack_build(
        memory::PinnedExpertArena& arena, memory::NumaManager& numa,
        const LiveGgufExpertSource& src, uint32_t num_layers,
        uint32_t num_experts,
        const std::unordered_map<memory::ExpertKey, int>* placement,
        int threads) {
    using memory::PinnedExpertArena;
    const auto t0 = std::chrono::steady_clock::now();

    LiveBuildStats stats;
    PinnedExpertArena::PreloadPlanStats plan_stats;
    const std::vector<PinnedExpertArena::PreloadAssignment> plan =
        arena.plan_preload(src, num_layers, num_experts, placement,
                           &plan_stats);
    stats.adopted = plan_stats.already;
    stats.skipped_full = plan_stats.skipped_full;
    stats.placed = plan_stats.placed;
    stats.place_fallback = plan_stats.place_fallback;
    if (plan.empty()) {
        stats.seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        spdlog::info("live prepack: nothing to build ({} adopted warm, {} "
                     "skipped — arenas full)", stats.adopted,
                     stats.skipped_full);
        return stats;
    }

    // Group plan indices by destination node (workers pin to their node so
    // slot writes and staging first-touch are NUMA-local).
    std::unordered_map<int, std::vector<size_t>> by_node;
    for (size_t i = 0; i < plan.size(); ++i)
        by_node[plan[i].node].push_back(i);

    if (threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        threads = static_cast<int>(std::min(hw == 0 ? 1u : hw, 32u));
    }
    threads = std::max(threads, static_cast<int>(by_node.size()));

    // Completion channel: workers push (plan index, ok); the init thread
    // drains and mark_ready()s — the arena/ArenaCache stay single-writer.
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::pair<size_t, bool>> done;
    done.reserve(1024);

    struct NodeWork {
        int node = -1;
        const std::vector<size_t>* idxs = nullptr;
        std::atomic<size_t> cursor{0};
    };
    std::vector<std::unique_ptr<NodeWork>> works;
    for (auto& [node, idxs] : by_node) {
        auto w = std::make_unique<NodeWork>();
        w->node = node;
        w->idxs = &idxs;
        works.push_back(std::move(w));
    }

    // Distribute workers proportionally to each node's slot count (>= 1).
    std::vector<int> workers_per(works.size(), 1);
    {
        int remaining = threads - static_cast<int>(works.size());
        if (remaining > 0) {
            for (size_t w = 0; w < works.size(); ++w) {
                const double share =
                    static_cast<double>(works[w]->idxs->size()) /
                    static_cast<double>(plan.size());
                workers_per[w] += static_cast<int>(share * remaining + 0.5);
            }
        }
    }

    std::vector<std::thread> pool;
    int total_workers = 0;
    for (size_t w = 0; w < works.size(); ++w) {
        for (int k = 0; k < workers_per[w]; ++k) {
            ++total_workers;
            pool.emplace_back([&, w] {
                NodeWork& nw = *works[w];
                numa.pin_current_thread_to_node(nw.node);
                for (;;) {
                    const size_t c =
                        nw.cursor.fetch_add(1, std::memory_order_relaxed);
                    if (c >= nw.idxs->size()) break;
                    const size_t pi = (*nw.idxs)[c];
                    const auto& a = plan[pi];
                    const bool ok = src.load_into(a.key, a.slot);
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        done.emplace_back(pi, ok);
                    }
                    cv.notify_one();
                }
            });
        }
    }
    spdlog::info("live prepack: building {} slot(s) over {} node(s) with {} "
                 "worker(s) ({} adopted warm, {} skipped-full{})",
                 plan.size(), works.size(), total_workers, stats.adopted,
                 stats.skipped_full,
                 placement ? ", placement ENGAGED" : "");

    // Drain + commit on the init thread.
    size_t committed = 0;
    std::vector<std::pair<size_t, bool>> batch;
    while (committed < plan.size()) {
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return !done.empty(); });
            batch.swap(done);
        }
        for (const auto& [pi, ok] : batch) {
            if (ok) {
                arena.mark_ready(plan[pi].key);
                ++stats.filled;
            } else {
                ++stats.failed;  // slot stays reserved-empty (reloadable)
            }
            ++committed;
        }
        batch.clear();
    }
    for (auto& t : pool) t.join();

    stats.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    stats.gigabytes = static_cast<double>(stats.filled) *
                      static_cast<double>(src.slot_size_bytes()) / 1073741824.0;
    spdlog::info("live prepack: filled {} slot(s) ({:.1f} GB) in {:.2f} s "
                 "({:.2f} GB/s), {} failed, {} adopted warm, {} skipped-full",
                 stats.filled, stats.gigabytes, stats.seconds,
                 stats.seconds > 0 ? stats.gigabytes / stats.seconds : 0.0,
                 stats.failed, stats.adopted, stats.skipped_full);
    if (placement)
        spdlog::info("live prepack: host placement — {} slots "
                     "placement-directed, {} fell back to tiered fill",
                     stats.placed, stats.place_fallback);
    return stats;
}

}  // namespace layerstorm::model

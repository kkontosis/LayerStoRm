#include "core/perf_trace.h"

#if LAYERSTORM_PERF_TRACE  // compiled out entirely when the gate is off

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace layerstorm::perf_trace {
namespace {

struct Ev {
    uint64_t ns;
    uint16_t stage;
    uint16_t gpu;
    uint32_t cmd_seq;
    uint32_t key;
    uint32_t _pad;
    uint64_t token;
};  // 32 bytes

Ev*                 g_buf = nullptr;
size_t              g_cap = 0;
std::atomic<size_t> g_idx{0};
bool                g_polltick = false;  // kPollTick recorded only if LS_PERF_TRACE_POLLTICKS set

inline uint64_t now_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace

void init(size_t capacity) {
    if (g_buf || capacity == 0) return;
    g_buf = new Ev[capacity];     // one-time, at init (NOT the hot path)
    g_cap = capacity;
    g_idx.store(0, std::memory_order_relaxed);
    std::fprintf(stderr, "[perf_trace] enabled, capacity=%zu (%zu MB)\n",
                 capacity, capacity * sizeof(Ev) / (1024 * 1024));
}

void init_from_env() {
    const char* e = std::getenv("LS_PERF_TRACE");
    if (!e || !*e) return;
    size_t cap = std::strtoull(e, nullptr, 10);
    if (cap <= 1) cap = 64u << 20;  // "1" → 64M records default (full keeper52 100-tok run)
    // kPollTick (one event per transfer-engine poll — millions/s during MoE wait)
    // is the ring-filler and is NOT needed by aggregate(). OFF unless opted in, so
    // the meaningful fetch/DMA/compute markers span the WHOLE run, not ~2 tokens.
    const char* pt = std::getenv("LS_PERF_TRACE_POLLTICKS");
    g_polltick = pt && *pt && *pt != '0';
    init(cap);
}

bool enabled() noexcept { return g_buf != nullptr; }

void record(uint16_t stage, uint16_t gpu, uint32_t cmd_seq,
            uint32_t key, uint64_t token) noexcept {
    if (!g_buf) return;
    if (stage == kPollTick && !g_polltick) return;  // per-poll spam: opt-in only
    size_t i = g_idx.fetch_add(1, std::memory_order_relaxed);
    if (i >= g_cap) return;         // drop overflow (deterministic, no realloc)
    g_buf[i] = Ev{now_ns(), stage, gpu, cmd_seq, key, 0, token};
}

Aggregate aggregate() {
    Aggregate a;
    if (!g_buf) return a;  // available=false
    size_t n = g_idx.load(std::memory_order_relaxed);
    a.overflow = n > g_cap;
    if (a.overflow) n = g_cap;
    a.available    = true;
    a.total_events = n;

    // Pair the four MoE phase markers per cmd_seq. Slots: 0=enter, 1=fetch_issued,
    // 2=finalize_enter, 3=finalize_exit (ns; 0 = unseen). Each MoE command's
    // cmd_seq is unique, so an entry is completed exactly once (at finalize_exit).
    constexpr int kAggMaxGpus = 16;
    std::unordered_map<uint32_t, std::array<uint64_t, 4>> moe;
    // Per-cmd_seq per-GPU LAST arrival ns (0 = none), from kExpertArrived.
    std::unordered_map<uint32_t, std::array<uint64_t, kAggMaxGpus>> arr;
    auto add = [](StageDelta& d, uint64_t a_ns, uint64_t b_ns) {
        if (a_ns && b_ns && b_ns >= a_ns) { ++d.count; d.total_ms += (b_ns - a_ns) / 1e6; }
    };
    for (size_t i = 0; i < n; ++i) {
        const Ev& e = g_buf[i];
        switch (e.stage) {
            case kMoeEnter:         moe[e.cmd_seq][0] = e.ns; break;
            case kMoeFetchIssued:   moe[e.cmd_seq][1] = e.ns; break;
            case kMoeFinalizeEnter: moe[e.cmd_seq][2] = e.ns; break;
            case kExpertArrived: {
                if (e.gpu < kAggMaxGpus) {
                    auto& g = arr[e.cmd_seq];   // value-initialized to all-zero
                    if (e.ns > g[e.gpu]) g[e.gpu] = e.ns;
                }
                break;
            }
            case kMoeFinalizeExit: {
                auto it = moe.find(e.cmd_seq);
                if (it != moe.end()) {
                    const auto& t = it->second;
                    add(a.moe_fetch,   t[0], t[1]);
                    add(a.moe_wait,    t[1], t[2]);
                    add(a.moe_compute, t[2], e.ns);
                    add(a.moe_total,   t[0], e.ns);
                    // Fetch-wall decomposition: pair the per-GPU last-arrivals with
                    // this cmd's fetch_issued (t[1]). t_last = max, fastest = min over
                    // GPUs that actually fetched (arrival > t[1]).
                    auto ai = arr.find(e.cmd_seq);
                    if (ai != arr.end() && t[1]) {
                        uint64_t t_last = 0, fastest = 0;
                        for (uint64_t v : ai->second) {
                            if (v <= t[1]) continue;       // GPU fetched nothing this cmd
                            if (v > t_last) t_last = v;
                            if (fastest == 0 || v < fastest) fastest = v;
                        }
                        if (t_last) {                       // ≥1 GPU fetched
                            add(a.fetch_wall,    t[1],    t_last);
                            add(a.fetch_fastest, t[1],    fastest);
                            add(a.fetch_dead,    fastest, t_last);
                        }
                    }
                    if (ai != arr.end()) arr.erase(ai);
                    moe.erase(it);
                }
                break;
            }
            case kDmaGpu:
                // token IS the elapsed ns of the pure GPU memcpy (not a timestamp).
                ++a.dma_gpu.count;
                a.dma_gpu.total_ms += e.token / 1e6;
                break;
            case kComputeGpu:
                // token IS the elapsed ns of the finalize MoE GEMM chain (GPU time).
                ++a.compute_gpu.count;
                a.compute_gpu.total_ms += e.token / 1e6;
                break;
            default: break;
        }
    }
    return a;
}

void dump_csv(const char* default_path) {
    if (!g_buf) return;
    const char* out = std::getenv("LS_PERF_TRACE_OUT");
    const char* path = (out && *out) ? out : default_path;
    size_t n = g_idx.load(std::memory_order_relaxed);
    bool overflow = n > g_cap;
    if (overflow) n = g_cap;
    FILE* f = std::fopen(path, "w");
    if (!f) {
        std::fprintf(stderr, "[perf_trace] FAILED to open %s\n", path);
        return;
    }
    std::fprintf(f, "ns,stage,gpu,cmd_seq,key,token\n");
    for (size_t i = 0; i < n; ++i) {
        const Ev& e = g_buf[i];
        std::fprintf(f, "%llu,%u,%u,%u,%u,%llu\n",
                     static_cast<unsigned long long>(e.ns), e.stage, e.gpu,
                     e.cmd_seq, e.key, static_cast<unsigned long long>(e.token));
    }
    std::fclose(f);
    std::fprintf(stderr, "[perf_trace] dumped %zu events to %s%s\n", n, path,
                 overflow ? " (OVERFLOW — raise LS_PERF_TRACE)" : "");
}

}  // namespace layerstorm::perf_trace

#endif  // LAYERSTORM_PERF_TRACE

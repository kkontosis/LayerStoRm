// H2D micro-benchmark — find the fastest absolute way to move an expert-sized
// buffer from a (NUMA-placed) host RAM pool into GPU VRAM, ready to compute.
//
// This is an INDEPENDENT feasibility probe for LayerStoRm expert prefetch.
// It is intentionally not wired into the engine; it measures the physical
// ceiling and the overhead above it so we know whether "copy an expert and run
// it in microseconds, bound only by PCIe" is achievable on this box.
//
// Axes measured (see --help):
//   - latency floor: fixed per-transfer overhead (tiny copies)
//   - size sweep: achieved GB/s vs size, on the GPU's local NUMA node
//   - pin strategy: pageable vs cudaHostAlloc vs write-combined vs
//                   numa_alloc_onnode + cudaHostRegister (the real 500GB path)
//   - mechanism: plain memcpyAsync vs CUDA graph replay vs multi-stream
//   - sync mode: stream sync vs event sync vs spin-poll
//   - NUMA matrix: every (source node x target GPU) — local vs cross-NUMA
//   - contention: two GPUs pulling from one shared node (share_degree=2)
//
// Build: see CMakeLists.txt in this directory.

#include <cuda_runtime.h>
#include <numa.h>
#include <numaif.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// ----------------------------------------------------------------------------
// Error handling
// ----------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                        \
    do {                                                                       \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,\
                         cudaGetErrorString(_e));                             \
            std::exit(1);                                                     \
        }                                                                    \
    } while (0)

using Clock = std::chrono::steady_clock;
static double now_us() {
    return std::chrono::duration<double, std::micro>(
               Clock::now().time_since_epoch())
        .count();
}

// ----------------------------------------------------------------------------
// Stats
// ----------------------------------------------------------------------------
struct Stat {
    double p50 = 0, p99 = 0, mn = 0, mx = 0, mean = 0;
};
static Stat summarize(std::vector<double>& v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto pick = [&](double q) {
        size_t i = (size_t)(q * (v.size() - 1) + 0.5);
        return v[i];
    };
    s.p50 = pick(0.50);
    s.p99 = pick(0.99);
    s.mn = v.front();
    s.mx = v.back();
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / v.size();
    return s;
}
static double gbps(size_t bytes, double us) {
    // GB/s as 1e9 bytes/s (decimal), matching nvidia conventions.
    return (double)bytes / (us * 1e3);
}

// ----------------------------------------------------------------------------
// Topology discovery
// ----------------------------------------------------------------------------
struct GpuInfo {
    int id;
    std::string name;
    std::string bdf;     // 0000:16:00.0
    int numa_node;       // NUMA node the GPU is attached to (-1 unknown)
    std::string link_now;
    std::string link_max;
};

static std::string read_sysfs(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return {};
    char buf[128] = {0};
    if (!std::fgets(buf, sizeof(buf), f)) { std::fclose(f); return {}; }
    std::fclose(f);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

static std::vector<GpuInfo> discover_gpus() {
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    std::vector<GpuInfo> out;
    for (int i = 0; i < n; ++i) {
        GpuInfo g;
        g.id = i;
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, i));
        g.name = p.name;
        char bdf[32];
        CUDA_CHECK(cudaDeviceGetPCIBusId(bdf, sizeof(bdf), i));
        // cudaDeviceGetPCIBusId returns e.g. "0000:16:00.0" (lowercase hex)
        std::string b(bdf);
        for (auto& c : b) c = (char)std::tolower((unsigned char)c);
        g.bdf = b;
        std::string base = "/sys/bus/pci/devices/" + b + "/";
        std::string nn = read_sysfs(base + "numa_node");
        g.numa_node = nn.empty() ? -1 : std::atoi(nn.c_str());
        g.link_now = read_sysfs(base + "current_link_speed");
        g.link_max = read_sysfs(base + "max_link_speed");
        out.push_back(g);
    }
    return out;
}

// ----------------------------------------------------------------------------
// Host buffer strategies
// ----------------------------------------------------------------------------
enum class Pin { Pageable, HostAlloc, WriteCombined, NumaRegister };
static const char* pin_name(Pin p) {
    switch (p) {
        case Pin::Pageable: return "pageable(malloc)";
        case Pin::HostAlloc: return "cudaHostAlloc";
        case Pin::WriteCombined: return "cudaHostAlloc+WC";
        case Pin::NumaRegister: return "numa_onnode+register";
    }
    return "?";
}

struct HostBuf {
    void* ptr = nullptr;
    size_t bytes = 0;
    Pin pin = Pin::Pageable;
    int node = -1;
    bool registered = false;
};

// Allocate a host buffer of `bytes` with the given strategy. For NumaRegister,
// memory is bound to `node` (numa_alloc_onnode) and page-locked via
// cudaHostRegister — this is the realistic path for a multi-hundred-GB pool we
// cannot cudaHostAlloc directly.
static HostBuf alloc_host(size_t bytes, Pin pin, int node) {
    HostBuf h;
    h.bytes = bytes;
    h.pin = pin;
    h.node = node;
    switch (pin) {
        case Pin::Pageable:
            h.ptr = std::malloc(bytes);
            break;
        case Pin::HostAlloc:
            CUDA_CHECK(cudaHostAlloc(&h.ptr, bytes, cudaHostAllocDefault));
            break;
        case Pin::WriteCombined:
            CUDA_CHECK(cudaHostAlloc(&h.ptr, bytes, cudaHostAllocWriteCombined));
            break;
        case Pin::NumaRegister:
            h.ptr = numa_alloc_onnode(bytes, node);
            if (!h.ptr) {
                std::fprintf(stderr, "numa_alloc_onnode(%zu, %d) failed\n",
                             bytes, node);
                std::exit(1);
            }
            std::memset(h.ptr, 1, bytes);
            CUDA_CHECK(cudaHostRegister(h.ptr, bytes, cudaHostRegisterDefault));
            h.registered = true;
            break;
    }
    if (pin != Pin::NumaRegister) std::memset(h.ptr, 1, bytes);
    return h;
}
static void free_host(HostBuf& h) {
    if (!h.ptr) return;
    switch (h.pin) {
        case Pin::Pageable:
            std::free(h.ptr);
            break;
        case Pin::HostAlloc:
        case Pin::WriteCombined:
            CUDA_CHECK(cudaFreeHost(h.ptr));
            break;
        case Pin::NumaRegister:
            if (h.registered) CUDA_CHECK(cudaHostUnregister(h.ptr));
            numa_free(h.ptr, h.bytes);
            break;
    }
    h.ptr = nullptr;
}

// Verify the pages actually landed on `node` (sanity for NUMA placement).
static int actual_node_of(void* ptr) {
    int status = -1;
    void* pages[1] = {ptr};
    if (move_pages(0, 1, pages, nullptr, &status, 0) != 0) return -1;
    return status;
}

// ----------------------------------------------------------------------------
// Core measurement: CPU-observed latency of {issue copy; wait until on GPU}.
// This is the metric that matters for "copy an expert then run it now".
// ----------------------------------------------------------------------------
struct CopyResult {
    Stat cpu_us;     // wall time issue->ready (what the orchestrator feels)
    Stat gpu_us;     // pure DMA time measured by CUDA events
};

static CopyResult bench_copy(int gpu, void* dst, const void* src, size_t bytes,
                             cudaStream_t stream, int iters, int warmup) {
    CUDA_CHECK(cudaSetDevice(gpu));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::vector<double> cpu_t, gpu_t;
    cpu_t.reserve(iters);
    gpu_t.reserve(iters);
    for (int i = 0; i < warmup + iters; ++i) {
        double t0 = now_us();
        CUDA_CHECK(cudaEventRecord(a, stream));
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaEventRecord(b, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double t1 = now_us();
        if (i >= warmup) {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            cpu_t.push_back(t1 - t0);
            gpu_t.push_back(ms * 1e3);
        }
    }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    CopyResult r;
    r.cpu_us = summarize(cpu_t);
    r.gpu_us = summarize(gpu_t);
    return r;
}

// ----------------------------------------------------------------------------
// Test: latency floor (tiny copies) — the irreducible per-transfer overhead.
// ----------------------------------------------------------------------------
static void test_latency_floor(const GpuInfo& g) {
    int node = g.numa_node < 0 ? 0 : g.numa_node;
    numa_run_on_node(node);
    std::printf("\n== Latency floor (GPU %d '%s', src node %d, local) ==\n",
                g.id, g.name.c_str(), node);
    std::printf("  %-10s %12s %12s %12s\n", "size", "cpu_p50(us)",
                "cpu_p99(us)", "gpu_p50(us)");
    size_t sizes[] = {256, 1024, 4096, 16384, 65536, 262144};
    size_t maxb = 262144;
    HostBuf h = alloc_host(maxb, Pin::NumaRegister, node);
    CUDA_CHECK(cudaSetDevice(g.id));
    void* dst;
    CUDA_CHECK(cudaMalloc(&dst, maxb));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    for (size_t b : sizes) {
        CopyResult r = bench_copy(g.id, dst, h.ptr, b, s, 3000, 200);
        std::printf("  %-10zu %12.2f %12.2f %12.2f\n", b, r.cpu_us.p50,
                    r.cpu_us.p99, r.gpu_us.p50);
    }
    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(dst));
    free_host(h);
}

// ----------------------------------------------------------------------------
// Test: size sweep on the local node — achieved GB/s vs the PCIe ceiling.
// Samples the PCIe link speed during the sustained run (checks retrain).
// ----------------------------------------------------------------------------
static void test_size_sweep(const GpuInfo& g) {
    int node = g.numa_node < 0 ? 0 : g.numa_node;
    numa_run_on_node(node);
    std::printf("\n== Size sweep (GPU %d, src node %d local) — link max %s ==\n",
                g.id, node, g.link_max.c_str());
    std::printf("  %-12s %12s %12s %12s %12s\n", "size(MB)", "cpu_p50(us)",
                "gpu_p50(us)", "cpu_GB/s", "gpu_GB/s");
    double sizes_mb[] = {0.25, 1, 2, 4, 8, 16, 24.77, 32, 64};
    std::string base = "/sys/bus/pci/devices/" + g.bdf + "/";
    for (double mb : sizes_mb) {
        size_t b = (size_t)(mb * 1024 * 1024);
        HostBuf h = alloc_host(b, Pin::NumaRegister, node);
        CUDA_CHECK(cudaSetDevice(g.id));
        void* dst;
        CUDA_CHECK(cudaMalloc(&dst, b));
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        int iters = b < (4 << 20) ? 1000 : 300;
        CopyResult r = bench_copy(g.id, dst, h.ptr, b, s, iters, 50);
        std::string link = read_sysfs(base + "current_link_speed");
        std::printf("  %-12.2f %12.2f %12.2f %12.1f %12.1f   link=%s\n", mb,
                    r.cpu_us.p50, r.gpu_us.p50, gbps(b, r.cpu_us.p50),
                    gbps(b, r.gpu_us.p50), link.c_str());
        CUDA_CHECK(cudaStreamDestroy(s));
        CUDA_CHECK(cudaFree(dst));
        free_host(h);
    }
}

// ----------------------------------------------------------------------------
// Test: pin strategy at a couple of representative sizes.
// ----------------------------------------------------------------------------
static void test_pin_strategy(const GpuInfo& g) {
    int node = g.numa_node < 0 ? 0 : g.numa_node;
    numa_run_on_node(node);
    std::printf("\n== Pin strategy (GPU %d, node %d) ==\n", g.id, node);
    std::printf("  %-22s %-10s %12s %12s %10s\n", "strategy", "size", "cpu_p50",
                "gpu_GB/s", "on_node");
    size_t sizes[] = {(size_t)(4 << 20), (size_t)(24.77 * 1024 * 1024)};
    Pin pins[] = {Pin::Pageable, Pin::HostAlloc, Pin::WriteCombined,
                  Pin::NumaRegister};
    for (size_t b : sizes) {
        for (Pin pin : pins) {
            HostBuf h = alloc_host(b, pin, node);
            int an = actual_node_of(h.ptr);
            CUDA_CHECK(cudaSetDevice(g.id));
            void* dst;
            CUDA_CHECK(cudaMalloc(&dst, b));
            cudaStream_t s;
            CUDA_CHECK(cudaStreamCreate(&s));
            CopyResult r = bench_copy(g.id, dst, h.ptr, b, s, 200, 30);
            std::printf("  %-22s %-10.1f %12.2f %12.1f %10d\n", pin_name(pin),
                        b / 1048576.0, r.cpu_us.p50, gbps(b, r.gpu_us.p50), an);
            CUDA_CHECK(cudaStreamDestroy(s));
            CUDA_CHECK(cudaFree(dst));
            free_host(h);
        }
    }
}

// ----------------------------------------------------------------------------
// Test: mechanism — plain async, CUDA graph replay, multi-stream concurrency.
// ----------------------------------------------------------------------------
static void test_mechanism(const GpuInfo& g, size_t b) {
    int node = g.numa_node < 0 ? 0 : g.numa_node;
    numa_run_on_node(node);
    std::printf("\n== Mechanism (GPU %d, node %d, %.2f MB) ==\n", g.id, node,
                b / 1048576.0);
    CUDA_CHECK(cudaSetDevice(g.id));

    // (1) plain
    {
        HostBuf h = alloc_host(b, Pin::NumaRegister, node);
        void* dst;
        CUDA_CHECK(cudaMalloc(&dst, b));
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        CopyResult r = bench_copy(g.id, dst, h.ptr, b, s, 500, 50);
        std::printf("  %-26s cpu_p50=%8.2f us  gpu_GB/s=%7.1f\n",
                    "plain memcpyAsync", r.cpu_us.p50, gbps(b, r.gpu_us.p50));
        CUDA_CHECK(cudaStreamDestroy(s));
        CUDA_CHECK(cudaFree(dst));
        free_host(h);
    }

    // (2) CUDA graph replay (capture one copy, replay) — removes per-launch CPU
    // cost; relevant when the orchestrator fires the same prefetch shape often.
    {
        HostBuf h = alloc_host(b, Pin::NumaRegister, node);
        void* dst;
        CUDA_CHECK(cudaMalloc(&dst, b));
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        CUDA_CHECK(cudaMemcpyAsync(dst, h.ptr, b, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaStreamEndCapture(s, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        std::vector<double> cpu;
        for (int i = 0; i < 550; ++i) {
            double t0 = now_us();
            CUDA_CHECK(cudaGraphLaunch(exec, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            double t1 = now_us();
            if (i >= 50) cpu.push_back(t1 - t0);
        }
        Stat st = summarize(cpu);
        std::printf("  %-26s cpu_p50=%8.2f us  gpu_GB/s=%7.1f\n",
                    "CUDA graph replay", st.p50, gbps(b, st.p50));
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(s));
        CUDA_CHECK(cudaFree(dst));
        free_host(h);
    }

    // (3) multi-stream concurrency — K experts in flight at once. Reports
    // aggregate GB/s (does overlap reach the link ceiling?).
    for (int K : {2, 4, 8}) {
        std::vector<HostBuf> hs(K);
        std::vector<void*> dst(K);
        std::vector<cudaStream_t> st(K);
        for (int k = 0; k < K; ++k) {
            hs[k] = alloc_host(b, Pin::NumaRegister, node);
            CUDA_CHECK(cudaMalloc(&dst[k], b));
            CUDA_CHECK(cudaStreamCreate(&st[k]));
        }
        std::vector<double> agg;
        for (int it = 0; it < 250; ++it) {
            double t0 = now_us();
            for (int k = 0; k < K; ++k)
                CUDA_CHECK(cudaMemcpyAsync(dst[k], hs[k].ptr, b,
                                           cudaMemcpyHostToDevice, st[k]));
            for (int k = 0; k < K; ++k) CUDA_CHECK(cudaStreamSynchronize(st[k]));
            double t1 = now_us();
            if (it >= 30) agg.push_back(t1 - t0);
        }
        Stat s = summarize(agg);
        char label[64];
        std::snprintf(label, sizeof(label), "multi-stream x%d", K);
        std::printf("  %-26s cpu_p50=%8.2f us  agg_GB/s=%7.1f  (per=%5.1f)\n",
                    label, s.p50, gbps((size_t)b * K, s.p50),
                    gbps(b, s.p50));
        for (int k = 0; k < K; ++k) {
            CUDA_CHECK(cudaStreamDestroy(st[k]));
            CUDA_CHECK(cudaFree(dst[k]));
            free_host(hs[k]);
        }
    }
}

// ----------------------------------------------------------------------------
// Test: sync mode — blocking stream sync vs event sync vs spin-poll.
// ----------------------------------------------------------------------------
static void test_sync_mode(const GpuInfo& g, size_t b) {
    int node = g.numa_node < 0 ? 0 : g.numa_node;
    numa_run_on_node(node);
    std::printf("\n== Sync mode (GPU %d, node %d, %.2f MB) ==\n", g.id, node,
                b / 1048576.0);
    CUDA_CHECK(cudaSetDevice(g.id));
    HostBuf h = alloc_host(b, Pin::NumaRegister, node);
    void* dst;
    CUDA_CHECK(cudaMalloc(&dst, b));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    cudaEvent_t done;
    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));

    auto run = [&](const char* name, int mode) {
        std::vector<double> cpu;
        for (int i = 0; i < 550; ++i) {
            double t0 = now_us();
            CUDA_CHECK(cudaMemcpyAsync(dst, h.ptr, b, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaEventRecord(done, s));
            if (mode == 0) {
                CUDA_CHECK(cudaStreamSynchronize(s));
            } else if (mode == 1) {
                CUDA_CHECK(cudaEventSynchronize(done));
            } else {
                while (cudaEventQuery(done) == cudaErrorNotReady) {
                }
            }
            double t1 = now_us();
            if (i >= 50) cpu.push_back(t1 - t0);
        }
        Stat st = summarize(cpu);
        std::printf("  %-22s cpu_p50=%8.2f us  cpu_p99=%8.2f us\n", name, st.p50,
                    st.p99);
    };
    run("cudaStreamSynchronize", 0);
    run("cudaEventSynchronize", 1);
    run("spin cudaEventQuery", 2);

    CUDA_CHECK(cudaEventDestroy(done));
    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(dst));
    free_host(h);
}

// ----------------------------------------------------------------------------
// Test: NUMA matrix — every (source node x target GPU). The core P-22 result.
// ----------------------------------------------------------------------------
static void test_numa_matrix(const std::vector<GpuInfo>& gpus, size_t b) {
    int nodes = numa_max_node() + 1;
    std::printf("\n== NUMA matrix: gpu_GB/s [src node -> GPU], %.2f MB ==\n",
                b / 1048576.0);
    std::printf("  (local cells marked *; GPU-less source nodes are pure cross-NUMA)\n");
    std::printf("  %-10s", "src\\gpu");
    for (auto& g : gpus) std::printf(" %10s", ("GPU" + std::to_string(g.id)).c_str());
    std::printf("\n");
    for (int src = 0; src < nodes; ++src) {
        if (numa_bitmask_isbitset(numa_get_mems_allowed(), src) == 0) continue;
        std::printf("  node%-6d", src);
        HostBuf h = alloc_host(b, Pin::NumaRegister, src);
        for (auto& g : gpus) {
            CUDA_CHECK(cudaSetDevice(g.id));
            void* dst;
            CUDA_CHECK(cudaMalloc(&dst, b));
            cudaStream_t s;
            CUDA_CHECK(cudaStreamCreate(&s));
            CopyResult r = bench_copy(g.id, dst, h.ptr, b, s, 200, 30);
            bool local = (g.numa_node == src);
            char cell[32];
            std::snprintf(cell, sizeof(cell), "%.1f%s", gbps(b, r.gpu_us.p50),
                          local ? "*" : "");
            std::printf(" %10s", cell);
            CUDA_CHECK(cudaStreamDestroy(s));
            CUDA_CHECK(cudaFree(dst));
        }
        std::printf("\n");
        free_host(h);
    }
}

// ----------------------------------------------------------------------------
// Test: contention — two GPUs pulling from one shared node concurrently
// (share_degree=2, e.g. node 2 -> GPU1+GPU2), vs each alone, vs two GPUs from
// independent nodes. Quantifies the shared-node memory-controller penalty.
// ----------------------------------------------------------------------------
static double bench_pair(int gpuA, void* srcA, int gpuB, void* srcB, size_t b,
                         int iters) {
    // Each GPU driven by its own thread; measure wall time for both to finish.
    std::atomic<double> outA{0}, outB{0};
    auto worker = [&](int gpu, void* src, std::atomic<double>& out) {
        CUDA_CHECK(cudaSetDevice(gpu));
        void* dst;
        CUDA_CHECK(cudaMalloc(&dst, b));
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        // warmup
        for (int i = 0; i < 20; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, b, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
        }
        double t0 = now_us();
        for (int i = 0; i < iters; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, b, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
        }
        double t1 = now_us();
        out = (t1 - t0) / iters;
        CUDA_CHECK(cudaStreamDestroy(s));
        CUDA_CHECK(cudaFree(dst));
    };
    std::thread tA(worker, gpuA, srcA, std::ref(outA));
    std::thread tB(worker, gpuB, srcB, std::ref(outB));
    tA.join();
    tB.join();
    // aggregate GB/s = 2 buffers / max(perA, perB)
    double per = std::max(outA.load(), outB.load());
    return gbps(b * 2, per);
}

static void test_contention(const std::vector<GpuInfo>& gpus, size_t b) {
    std::printf("\n== Contention (%.2f MB) ==\n", b / 1048576.0);
    // Find the shared node (>=2 GPUs) and two GPUs on it.
    int nodes = numa_max_node() + 1;
    int shared_node = -1, gA = -1, gB = -1;
    for (int nd = 0; nd < nodes; ++nd) {
        std::vector<int> on;
        for (auto& g : gpus)
            if (g.numa_node == nd) on.push_back(g.id);
        if ((int)on.size() >= 2) {
            shared_node = nd;
            gA = on[0];
            gB = on[1];
            break;
        }
    }
    if (shared_node >= 0) {
        HostBuf h = alloc_host(b, Pin::NumaRegister, shared_node);
        // single GPU alone from shared node
        {
            numa_run_on_node(shared_node);
            CUDA_CHECK(cudaSetDevice(gA));
            void* dst;
            CUDA_CHECK(cudaMalloc(&dst, b));
            cudaStream_t s;
            CUDA_CHECK(cudaStreamCreate(&s));
            CopyResult r = bench_copy(gA, dst, h.ptr, b, s, 200, 30);
            std::printf("  GPU%d alone   <- node%d : %7.1f GB/s\n", gA,
                        shared_node, gbps(b, r.gpu_us.p50));
            CUDA_CHECK(cudaStreamDestroy(s));
            CUDA_CHECK(cudaFree(dst));
        }
        double pair = bench_pair(gA, h.ptr, gB, h.ptr, b, 200);
        std::printf("  GPU%d+GPU%d    <- node%d (SHARED) : %7.1f GB/s aggregate\n",
                    gA, gB, shared_node, pair);
        free_host(h);
    } else {
        std::printf("  (no node hosts >=2 GPUs; skipping shared-node case)\n");
    }

    // Two GPUs on independent nodes, each from its own local node — the
    // best-case aggregate (no shared memory controller).
    int iA = -1, iB = -1;
    for (size_t i = 0; i < gpus.size(); ++i)
        for (size_t j = i + 1; j < gpus.size(); ++j)
            if (gpus[i].numa_node != gpus[j].numa_node && gpus[i].numa_node >= 0 &&
                gpus[j].numa_node >= 0) {
                iA = (int)i;
                iB = (int)j;
            }
    if (iA >= 0) {
        HostBuf hA = alloc_host(b, Pin::NumaRegister, gpus[iA].numa_node);
        HostBuf hB = alloc_host(b, Pin::NumaRegister, gpus[iB].numa_node);
        double pair = bench_pair(gpus[iA].id, hA.ptr, gpus[iB].id, hB.ptr, b, 200);
        std::printf("  GPU%d+GPU%d    <- nodes %d,%d (INDEPENDENT) : %7.1f GB/s aggregate\n",
                    gpus[iA].id, gpus[iB].id, gpus[iA].numa_node,
                    gpus[iB].numa_node, pair);
        free_host(hA);
        free_host(hB);
    }
}

// ----------------------------------------------------------------------------
// Test: pin stress — how much host RAM can we actually page-lock?
// Registers numa-bound chunks round-robin across GPU-attached nodes via
// cudaHostRegister (portable), ramping toward a target. Answers the central
// feasibility question: does cudaHostRegister count against RLIMIT_MEMLOCK
// (~63 GB here), or does the driver bypass it (→ 300 GB pool is fine)?
// ----------------------------------------------------------------------------
static void test_pin_stress(const std::vector<GpuInfo>& gpus, double target_gb,
                            double chunk_gb) {
    // GPU-attached nodes, in order, deduped.
    std::vector<int> nodes;
    for (auto& g : gpus) {
        if (g.numa_node < 0) continue;
        if (std::find(nodes.begin(), nodes.end(), g.numa_node) == nodes.end())
            nodes.push_back(g.numa_node);
    }
    if (nodes.empty()) nodes.push_back(0);

    size_t chunk = (size_t)(chunk_gb * (1ull << 30));
    size_t target = (size_t)(target_gb * (1ull << 30));
    std::printf("\n== Pin stress: target %.0f GB in %.1f GB chunks across nodes",
                target_gb, chunk_gb);
    std::printf(" {");
    for (size_t i = 0; i < nodes.size(); ++i)
        std::printf("%s%d", i ? "," : "", nodes[i]);
    std::printf("} ==\n");
    std::printf("  RLIMIT_MEMLOCK soft/hard ~63 GB on this box; watching for the cliff.\n");

    struct Chunk { void* p; size_t b; };
    std::vector<Chunk> chunks;
    size_t pinned = 0;
    bool failed = false;
    int ci = 0;
    while (pinned < target) {
        size_t b = std::min(chunk, target - pinned);
        int node = nodes[ci % nodes.size()];
        void* p = numa_alloc_onnode(b, node);
        if (!p) {
            std::printf("  numa_alloc_onnode(%.1f GB, node %d) FAILED at %.1f GB pinned\n",
                        b / 1073741824.0, node, pinned / 1073741824.0);
            failed = true;
            break;
        }
        std::memset(p, 1, b);  // fault pages onto the node
        double t0 = now_us();
        cudaError_t e = cudaHostRegister(p, b, cudaHostRegisterPortable);
        double t1 = now_us();
        if (e != cudaSuccess) {
            std::printf("  cudaHostRegister FAILED at cumulative %.1f GB: %s\n",
                        pinned / 1073741824.0, cudaGetErrorString(e));
            std::printf("  (chunk node %d, %.1f GB; register attempt took %.1f ms)\n",
                        node, b / 1073741824.0, (t1 - t0) / 1e3);
            numa_free(p, b);
            failed = true;
            cudaGetLastError();  // clear sticky error
            break;
        }
        chunks.push_back({p, b});
        pinned += b;
        std::printf("  +%.1f GB on node %d -> %.1f GB pinned  (register %.1f ms, %.1f GB/s)\n",
                    b / 1073741824.0, node, pinned / 1073741824.0,
                    (t1 - t0) / 1e3, gbps(b, t1 - t0));
        ++ci;
    }

    if (!failed) {
        std::printf("  SUCCESS: pinned %.1f GB total (no RLIMIT cliff hit)\n",
                    pinned / 1073741824.0);
        // Confirm a chunk well past the 63 GB mark is DMA-usable.
        if (!chunks.empty() && pinned > (70ull << 30)) {
            CUDA_CHECK(cudaSetDevice(gpus[0].id));
            void* dst;
            size_t tb = std::min((size_t)(24.77 * 1048576), chunks.back().b);
            CUDA_CHECK(cudaMalloc(&dst, tb));
            cudaStream_t s;
            CUDA_CHECK(cudaStreamCreate(&s));
            CopyResult r = bench_copy(gpus[0].id, dst,
                                      chunks.back().p, tb, s, 50, 10);
            std::printf("  DMA from last chunk (past 63 GB): %.1f GB/s — usable\n",
                        gbps(tb, r.gpu_us.p50));
            CUDA_CHECK(cudaStreamDestroy(s));
            CUDA_CHECK(cudaFree(dst));
        }
    }
    for (auto& c : chunks) {
        cudaHostUnregister(c.p);
        numa_free(c.p, c.b);
    }
    cudaGetLastError();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Test: REPLAY a recorded keeper transfer sequence in a bare harness.
//
// Reads tests/assets/fetch_transfer_log.csv (seq,gpu,src_numa,bytes,keeper_us) —
// the exact ordered H2D sequence the FETCH keeper issued — and re-runs ONLY the
// transfers: no daemon, no compute, no poll. Per source node we register ONE
// large region (size = region_gb, swept) to reproduce the 126 GB arena's
// IOMMU/IOTLB footprint; each transfer DMAs `bytes` from a scattered offset
// within its node's region to a VRAM dest on the recorded GPU. Each transfer is
// timed SOLO by CUDA events (pure GPU DMA — excludes CPU launch/"wall" time and
// any cross-transfer overlap), then averaged per class (local vs cross-NUMA).
//
// If the bare replay reproduces the keeper's 44.5/32.7 GB/s → the slowdown is
// intrinsic to the transfer pattern (arena footprint / IOMMU), NOT the daemon
// environment. If it gives ~56 → the daemon environment is the cause.
// ----------------------------------------------------------------------------
struct RTrec { int gpu; int src_numa; size_t bytes; };

static void test_replay(const std::vector<GpuInfo>& gpus, const char* csv,
                        double region_gb) {
    std::FILE* f = std::fopen(csv, "r");
    if (!f) { std::printf("\n[replay] cannot open %s\n", csv); return; }
    char line[256];
    std::vector<RTrec> recs;
    if (!std::fgets(line, sizeof(line), f)) { std::fclose(f); return; }  // header
    while (std::fgets(line, sizeof(line), f)) {
        int seq, g, src; long long b;
        if (std::sscanf(line, "%d,%d,%d,%lld", &seq, &g, &src, &b) >= 4)
            recs.push_back({g, src, (size_t)b});
    }
    std::fclose(f);
    if (recs.empty()) { std::printf("\n[replay] empty csv\n"); return; }
    const size_t bytes = recs[0].bytes;
    const size_t region = (size_t)(region_gb * (1ull << 30));
    const size_t nslots = region / bytes > 0 ? region / bytes : 1;

    // Per-node large registered region (only nodes referenced by the trace).
    HostBuf node_buf[16]; bool node_used[16] = {false};
    for (auto& r : recs)
        if (r.src_numa >= 0 && r.src_numa < 16) node_used[r.src_numa] = true;
    for (int nd = 0; nd < 16; ++nd)
        if (node_used[nd]) node_buf[nd] = alloc_host(region, Pin::NumaRegister, nd);

    // Per-GPU VRAM dest + stream + a reusable timing event pair.
    void* dst[16] = {nullptr}; cudaStream_t strm[16] = {nullptr};
    cudaEvent_t evS[16], evE[16];
    bool gpu_used[16] = {false};
    for (auto& r : recs) if (r.gpu >= 0 && r.gpu < 16) gpu_used[r.gpu] = true;
    for (int gi = 0; gi < (int)gpus.size() && gi < 16; ++gi) {
        if (!gpu_used[gi]) continue;
        CUDA_CHECK(cudaSetDevice(gpus[gi].id));
        CUDA_CHECK(cudaMalloc(&dst[gi], bytes));
        CUDA_CHECK(cudaStreamCreate(&strm[gi]));
        CUDA_CHECK(cudaEventCreate(&evS[gi]));
        CUDA_CHECK(cudaEventCreate(&evE[gi]));
    }

    std::printf("\n== REPLAY %zu transfers, region=%.1f GB/node (%zu slots), "
                "%.2f MB/xfer — pure GPU DMA, solo-timed ==\n",
                recs.size(), region_gb, nslots, bytes / 1048576.0);

    std::vector<double> us_local, us_cross;
    for (size_t i = 0; i < recs.size(); ++i) {
        const RTrec& r = recs[i];
        if (r.gpu < 0 || r.gpu >= (int)gpus.size() || !node_used[r.src_numa])
            continue;
        // Scattered offset (Knuth multiplicative hash) → maximize distinct pages.
        size_t slot = ((size_t)(i * 2654435761u)) % nslots;
        char* src = (char*)node_buf[r.src_numa].ptr + slot * bytes;
        int gi = r.gpu;
        CUDA_CHECK(cudaSetDevice(gpus[gi].id));
        CUDA_CHECK(cudaEventRecord(evS[gi], strm[gi]));
        CUDA_CHECK(cudaMemcpyAsync(dst[gi], src, bytes, cudaMemcpyHostToDevice,
                                   strm[gi]));
        CUDA_CHECK(cudaEventRecord(evE[gi], strm[gi]));
        CUDA_CHECK(cudaEventSynchronize(evE[gi]));
        float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, evS[gi], evE[gi]));
        double us = ms * 1e3;
        bool local = (gpus[gi].numa_node == r.src_numa);
        (local ? us_local : us_cross).push_back(us);
    }

    auto report = [&](const char* lbl, std::vector<double>& v) {
        if (v.empty()) { std::printf("  %-22s (none)\n", lbl); return; }
        double mean = 0; for (double x : v) mean += x; mean /= v.size();
        std::printf("  %-22s n=%6zu  mean=%7.1f us  %5.1f GB/s\n",
                    lbl, v.size(), mean, gbps(bytes, mean));
    };
    report("LOCAL (src==gpu node)", us_local);
    report("CROSS-NUMA",            us_cross);

    for (int nd = 0; nd < 16; ++nd) if (node_used[nd]) free_host(node_buf[nd]);
    for (int gi = 0; gi < 16; ++gi) if (gpu_used[gi] && dst[gi]) {
        cudaSetDevice(gpus[gi].id);
        cudaFree(dst[gi]); cudaStreamDestroy(strm[gi]);
        cudaEventDestroy(evS[gi]); cudaEventDestroy(evE[gi]);
    }
}

int main(int argc, char** argv) {
    if (numa_available() < 0) {
        std::fprintf(stderr, "libnuma not available\n");
        return 1;
    }
    std::string mode = argc > 1 ? argv[1] : "all";
    auto gpus = discover_gpus();

    std::printf("=== H2D benchmark ===\n");
    std::printf("NUMA nodes: %d\n", numa_max_node() + 1);
    for (auto& g : gpus)
        std::printf("GPU %d: %-22s bdf=%s node=%d link=%s / max=%s\n", g.id,
                    g.name.c_str(), g.bdf.c_str(), g.numa_node,
                    g.link_now.c_str(), g.link_max.c_str());

    const size_t expert = (size_t)(24.77 * 1024 * 1024);  // V3.2 packed slot
    const size_t few_mb = 4 << 20;

    int primary = 0;  // GPU 0 for single-GPU tests
    if (mode == "all" || mode == "env") {
        // env already printed
    }
    if (mode == "all" || mode == "latency")
        test_latency_floor(gpus[primary]);
    if (mode == "all" || mode == "sweep")
        test_size_sweep(gpus[primary]);
    if (mode == "all" || mode == "pin")
        test_pin_strategy(gpus[primary]);
    if (mode == "all" || mode == "mech") {
        test_mechanism(gpus[primary], few_mb);
        test_mechanism(gpus[primary], expert);
    }
    if (mode == "all" || mode == "sync")
        test_sync_mode(gpus[primary], expert);
    if (mode == "all" || mode == "numa")
        test_numa_matrix(gpus, expert);
    if (mode == "all" || mode == "contention")
        test_contention(gpus, expert);
    if (mode == "pinstress") {
        double total_gb = argc > 2 ? std::atof(argv[2]) : 100.0;
        double chunk_gb = argc > 3 ? std::atof(argv[3]) : 4.0;
        test_pin_stress(gpus, total_gb, chunk_gb);
    }
    if (mode == "replay") {
        const char* csv = argc > 2 ? argv[2]
            : "../../tests/assets/fetch_transfer_log.csv";
        // Region-size sweep: small (IOTLB-hot, bench-like) → large (arena-like).
        // Find where bandwidth falls from ~56 toward the keeper's 44.5.
        if (argc > 3) {
            test_replay(gpus, csv, std::atof(argv[3]));
        } else {
            for (double rg : {0.25, 1.0, 4.0, 16.0, 48.0})
                test_replay(gpus, csv, rg);
        }
    }

    std::printf("\n=== done ===\n");
    return 0;
}

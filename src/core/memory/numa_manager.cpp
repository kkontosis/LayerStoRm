#include "core/memory/numa_manager.h"

#include <dirent.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

#ifdef LAYERSTORM_HAS_NUMA
#include <numa.h>
#include <numaif.h>
#endif

namespace layerstorm::memory {

#if defined(LAYERSTORM_HAS_NUMA) && defined(LAYERSTORM_HBM_NODES)
namespace {

// Parse a kernel node-list string ("0-3,5,7-8\n") into a set of node ids.
std::set<int> parse_node_list(const char* s) {
    std::set<int> out;
    if (!s) return out;
    const char* p = s;
    while (*p) {
        char* end = nullptr;
        long lo = std::strtol(p, &end, 10);
        if (end == p) break;  // no digits → done (trailing '\n' etc.)
        long hi = lo;
        p = end;
        if (*p == '-') {
            ++p;
            hi = std::strtol(p, &end, 10);
            if (end == p) break;
            p = end;
        }
        for (long n = lo; n <= hi; ++n) out.insert(static_cast<int>(n));
        if (*p == ',') ++p;
        else break;
    }
    return out;
}

// Read a /sys/devices/system/node/<mask> node-state mask (e.g. "has_normal_memory").
// Returns the empty set if the file is unreadable (caller treats as "unknown").
std::set<int> read_node_state_mask(const char* mask) {
    std::string path = std::string("/sys/devices/system/node/") + mask;
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return {};
    char line[256] = {0};
    const char* got = std::fgets(line, sizeof(line), f);
    std::fclose(f);
    return parse_node_list(got);
}

// HMAT best-performance initiator for memory node `n`: the CPU-ful node listed
// under /sys/devices/system/node/nodeN/access0/initiators/ (access class 0 =
// best-performing initiators per the kernel's HMAT export). Multiple CPU
// initiators → the one at min numa_distance. Returns -1 when the directory is
// absent (no HMAT — pre-ACPI-6.2 firmware) or lists no CPU-ful node.
int hmat_initiator_node(int n, const std::set<int>& cpu_nodes) {
    std::string path = "/sys/devices/system/node/node" + std::to_string(n) +
                       "/access0/initiators";
    DIR* dir = ::opendir(path.c_str());
    if (!dir) return -1;
    int best = -1;
    int best_dist = std::numeric_limits<int>::max();
    while (struct dirent* de = ::readdir(dir)) {
        int init = -1;
        if (std::sscanf(de->d_name, "node%d", &init) != 1) continue;
        if (!cpu_nodes.count(init)) continue;  // initiator must have CPUs to pin to
        const int d = ::numa_distance(n, init);
        if (best < 0 || (d > 0 && d < best_dist)) {
            best = init;
            best_dist = (d > 0) ? d : best_dist;
        }
    }
    ::closedir(dir);
    return best;
}

}  // namespace
#endif  // LAYERSTORM_HAS_NUMA && LAYERSTORM_HBM_NODES

// ── Construction / destruction ───────────────────────────────────────────────

NumaManager::NumaManager(const config::HardwareConfig& hw)
    : page_size_(static_cast<size_t>(sysconf(_SC_PAGESIZE))) {
    // Build GPU -> NUMA node mapping.
    gpu_to_numa_.reserve(hw.gpus.size());
    std::set<int> nodes;
    for (const auto& g : hw.gpus) {
        gpu_to_numa_.push_back(g.numa_node);
        if (g.numa_node >= 0)
            nodes.insert(g.numa_node);
    }
    all_nodes_.assign(nodes.begin(), nodes.end());

    // Runtime NUMA availability check.
#ifdef LAYERSTORM_HAS_NUMA
    numa_available_ = (::numa_available() >= 0);
#else
    numa_available_ = false;
#endif

    // Pre-populate per-node counters (including -1 for interleaved).
    per_node_allocated_[-1].store(0);
    for (int n : all_nodes_)
        per_node_allocated_[n].store(0);

    // CPU-less HBM / device-memory node detection (TD-NUMA-HBM-BANKS).
    detect_hbm_nodes();
    for (const auto& h : hbm_nodes_)
        per_node_allocated_[h.node].store(0);

    spdlog::info("NumaManager: {} NUMA node(s), {} GPU(s), {} HBM node(s), libnuma {}",
                 all_nodes_.size(), gpu_to_numa_.size(), hbm_nodes_.size(),
                 numa_available_ ? "available" : "unavailable");
}

NumaManager::~NumaManager() {
    size_t remaining = total_allocated_.load(std::memory_order_relaxed);
    if (remaining > 0)
        spdlog::warn("NumaManager destroyed with {} bytes still allocated", remaining);
    assert(remaining == 0 && "NumaManager: leaked allocations");
}

// ── Topology queries ─────────────────────────────────────────────────────────

int NumaManager::num_nodes() const {
    return static_cast<int>(all_nodes_.size());
}

std::vector<int> NumaManager::all_memory_nodes() const {
#ifdef LAYERSTORM_HAS_NUMA
    if (numa_available_) {
        std::vector<int> nodes;
        const int maxn = ::numa_max_node();
        struct bitmask* cpus = numa_allocate_cpumask();
        for (int n = 0; n <= maxn; ++n) {
            if (::numa_node_size64(n, nullptr) <= 0)  // node must have memory
                continue;
            // CPU-less memory-only nodes (HBM / device memory onlined as small
            // hotplug NUMA nodes, e.g. four 16 GB nodes 4-7 next to the four
            // 128 GB host nodes on the dev box) are kept OUT of this list —
            // not because they are unbindable (they are: a strict MPOL_BIND
            // fills a full 16 GiB node with no OOM), but because they need
            // capacity-capped allocations (a default-sized host buffer, e.g.
            // the calibration's 32 GiB rotating footprint, over-commits the
            // 16 GB node → kernel CONSTRAINT_MEMORY_POLICY OOM) and explicit
            // CPU pinning. Callers wanting them use hbm_nodes() /
            // all_banks_including_hbm() and honor those constraints
            // (INV-NUMA-HOSTBANK, TD-NUMA-HBM-BANKS).
            if (cpus) {
                if (::numa_node_to_cpus(n, cpus) == 0 &&
                    ::numa_bitmask_weight(cpus) == 0)
                    continue;  // memory-only node → exposed via hbm_nodes()
            }
            nodes.push_back(n);
        }
        if (cpus) numa_free_cpumask(cpus);
        if (!nodes.empty()) return nodes;
    }
#endif
    return all_nodes_;  // fallback: GPU-attached nodes
}

void NumaManager::detect_hbm_nodes() {
#if defined(LAYERSTORM_HAS_NUMA) && defined(LAYERSTORM_HBM_NODES)
    if (!numa_available_) return;
    const int maxn = ::numa_max_node();

    // Partition nodes-with-memory into CPU-ful vs memory-only.
    std::set<int> cpu_nodes;
    std::vector<int> memory_only;
    struct bitmask* cpus = numa_allocate_cpumask();
    for (int n = 0; n <= maxn; ++n) {
        if (::numa_node_size64(n, nullptr) <= 0) continue;
        bool has_cpu = true;  // unknown cpumask → conservatively treat as CPU-ful
        if (cpus && ::numa_node_to_cpus(n, cpus) == 0)
            has_cpu = (::numa_bitmask_weight(cpus) > 0);
        if (has_cpu) cpu_nodes.insert(n);
        else memory_only.push_back(n);
    }
    if (cpus) numa_free_cpumask(cpus);
    if (memory_only.empty() || cpu_nodes.empty()) return;

    // Robustness gate: the node must be in the kernel's N_NORMAL_MEMORY mask —
    // normal, allocatable, bindable RAM. A non-coherent device aperture or a
    // not-yet-onlined region would not be listed. (An unreadable mask file →
    // empty set → no node qualifies: fail-safe to the DDR-only behavior.)
    const std::set<int> normal = read_node_state_mask("has_normal_memory");

    for (int n : memory_only) {
        if (!normal.count(n)) {
            spdlog::info("NumaManager: memory-only node {} not in has_normal_memory; "
                         "not usable as an HBM bank", n);
            continue;
        }
        // Nearest CPU node: prefer the HMAT access0 initiator (the platform's
        // own best-initiator declaration), fall back to min SLIT distance.
        int aff = hmat_initiator_node(n, cpu_nodes);
        if (aff < 0) {
            int best_dist = std::numeric_limits<int>::max();
            for (int c : cpu_nodes) {
                const int d = ::numa_distance(n, c);
                if (d > 0 && d < best_dist) { best_dist = d; aff = c; }
            }
        }
        if (aff < 0) {
            spdlog::warn("NumaManager: memory-only node {} has no resolvable CPU "
                         "affinity node; not usable as an HBM bank", n);
            continue;
        }
        hbm_nodes_.push_back(HbmNodeInfo{n, aff, node_total_bytes(n)});
        spdlog::info("NumaManager: HBM node {} detected ({:.1f} GiB, cpu-affinity "
                     "node {})", n,
                     static_cast<double>(hbm_nodes_.back().total_bytes) /
                         (1024.0 * 1024.0 * 1024.0),
                     aff);
    }
#endif
}

bool NumaManager::node_is_hbm(int node) const {
    for (const auto& h : hbm_nodes_)
        if (h.node == node) return true;
    return false;
}

int NumaManager::hbm_cpu_affinity_node(int node) const {
    for (const auto& h : hbm_nodes_)
        if (h.node == node) return h.cpu_affinity_node;
    return -1;
}

std::vector<int> NumaManager::all_banks_including_hbm() const {
    std::vector<int> banks = all_memory_nodes();
    for (const auto& h : hbm_nodes_)
        banks.push_back(h.node);
    std::sort(banks.begin(), banks.end());
    banks.erase(std::unique(banks.begin(), banks.end()), banks.end());
    return banks;
}

int NumaManager::gpu_numa_node(int gpu_position) const {
    if (gpu_position < 0 || gpu_position >= static_cast<int>(gpu_to_numa_.size()))
        return -1;
    return gpu_to_numa_[gpu_position];
}

bool NumaManager::numa_available() const {
    return numa_available_;
}

int NumaManager::expert_home_node(uint32_t expert_idx) const {
    if (all_nodes_.empty()) return -1;
    // Round-robin over GPU-attached nodes (balanced per node, P-22).
    return all_nodes_[expert_idx % all_nodes_.size()];
}

size_t NumaManager::node_meminfo_kb(int node, const char* key) const {
    if (node < 0 || !key) return 0;
    // /sys/devices/system/node/nodeN/meminfo lines: "Node N <Key>:  NNN kB".
    // Match the full key followed by ':' so e.g. "Active(file)" does not also
    // match "Active:" (or vice-versa) — substrings of distinct fields collide.
    std::string path = "/sys/devices/system/node/node" + std::to_string(node) +
                       "/meminfo";
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    const size_t keylen = std::strlen(key);
    while (std::fgets(line, sizeof(line), f)) {
        const char* p = std::strstr(line, key);
        if (p && p[keylen] == ':') {
            kb = std::strtoull(p + keylen + 1, nullptr, 10);
            break;
        }
    }
    std::fclose(f);
    return kb;
}

size_t NumaManager::node_free_bytes(int node) const {
    return node_meminfo_kb(node, "MemFree") * 1024;
}

size_t NumaManager::node_total_bytes(int node) const {
    return node_meminfo_kb(node, "MemTotal") * 1024;
}

size_t NumaManager::node_available_bytes(int node) const {
    // Per-node meminfo has no MemAvailable line (that is /proc-only), so
    // approximate it: free + reclaimable page cache (file-backed Active/Inactive)
    // + reclaimable slab. This is the honest "could be used" figure — MemFree
    // alone undercounts badly when page cache is large (INV-4.12f).
    return (node_meminfo_kb(node, "MemFree") +
            node_meminfo_kb(node, "Active(file)") +
            node_meminfo_kb(node, "Inactive(file)") +
            node_meminfo_kb(node, "SReclaimable")) * 1024;
}

void NumaManager::bind_range(void* ptr, size_t size, int numa_node) {
    if (!ptr || size == 0 || numa_node < 0) return;
    apply_mbind_bind(ptr, size, numa_node);
}

void NumaManager::pin_current_thread_to_node([[maybe_unused]] int numa_node) {
#ifdef LAYERSTORM_HAS_NUMA
    if (!numa_available_ || numa_node < 0) return;
    numa_run_on_node(numa_node);
#endif
}

// ── Internal helpers ─────────────────────────────────────────────────────────

size_t NumaManager::align_to_page(size_t size) const {
    if (size == 0)
        size = 1;  // at least one byte → one page
    return (size + page_size_ - 1) & ~(page_size_ - 1);
}

void* NumaManager::raw_mmap(size_t aligned_size) {
    void* ptr = ::mmap(nullptr, aligned_size,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS,
                       -1, 0);
    if (ptr == MAP_FAILED)
        throw std::bad_alloc();
    return ptr;
}

void NumaManager::raw_munmap(void* ptr, size_t size) {
    ::munmap(ptr, size);
}

void NumaManager::apply_mbind_bind([[maybe_unused]] void* ptr,
                                   [[maybe_unused]] size_t size,
                                   [[maybe_unused]] int numa_node) {
#ifdef LAYERSTORM_HAS_NUMA
    if (!numa_available_)
        return;
    struct bitmask* mask = numa_bitmask_alloc(static_cast<unsigned>(numa_max_node() + 1));
    numa_bitmask_setbit(mask, static_cast<unsigned>(numa_node));
    int rc = mbind(ptr, size, MPOL_BIND,
                   mask->maskp, mask->size + 1,
                   MPOL_MF_MOVE);
    numa_bitmask_free(mask);
    if (rc != 0)
        spdlog::warn("mbind(MPOL_BIND, node={}) failed: {}", numa_node, strerror(errno));
#endif
}

void NumaManager::apply_mbind_interleave([[maybe_unused]] void* ptr,
                                         [[maybe_unused]] size_t size) {
#ifdef LAYERSTORM_HAS_NUMA
    if (!numa_available_)
        return;
    struct bitmask* mask = numa_bitmask_alloc(static_cast<unsigned>(numa_max_node() + 1));
    for (int n : all_nodes_)
        numa_bitmask_setbit(mask, static_cast<unsigned>(n));
    int rc = mbind(ptr, size, MPOL_INTERLEAVE,
                   mask->maskp, mask->size + 1,
                   MPOL_MF_MOVE);
    numa_bitmask_free(mask);
    if (rc != 0)
        spdlog::warn("mbind(MPOL_INTERLEAVE) failed: {}", strerror(errno));
#endif
}

// ── Allocation ───────────────────────────────────────────────────────────────

NumaBuffer NumaManager::allocate_on_node(size_t size, int numa_node) {
    size_t aligned = align_to_page(size);
    void* ptr = raw_mmap(aligned);
    apply_mbind_bind(ptr, aligned, numa_node);

    total_allocated_.fetch_add(aligned, std::memory_order_relaxed);
    auto it = per_node_allocated_.find(numa_node);
    if (it != per_node_allocated_.end())
        it->second.fetch_add(aligned, std::memory_order_relaxed);
    else
        per_node_allocated_[-1].fetch_add(aligned, std::memory_order_relaxed);

    return NumaBuffer{ptr, aligned, numa_node};
}

NumaBuffer NumaManager::allocate_interleaved(size_t size) {
    size_t aligned = align_to_page(size);
    void* ptr = raw_mmap(aligned);
    apply_mbind_interleave(ptr, aligned);

    total_allocated_.fetch_add(aligned, std::memory_order_relaxed);
    per_node_allocated_[-1].fetch_add(aligned, std::memory_order_relaxed);

    return NumaBuffer{ptr, aligned, -1};
}

NumaBuffer NumaManager::allocate_for_gpu(size_t size, int gpu_position) {
    int node = gpu_numa_node(gpu_position);
    if (node < 0)
        return allocate_interleaved(size);
    return allocate_on_node(size, node);
}

// ── Shared (memfd-backed) allocation — persistent arena, P-24b ──────────────

bool NumaManager::shmem_thp_available() {
    FILE* f = std::fopen(
        "/sys/kernel/mm/transparent_hugepage/shmem_enabled", "r");
    if (!f) return false;
    char line[256] = {0};
    const char* got = std::fgets(line, sizeof(line), f);
    std::fclose(f);
    if (!got) return false;
    // The ACTIVE mode is bracketed, e.g. "always within_size advise [never] deny".
    const char* sel = std::strchr(line, '[');
    if (!sel) return false;
    return std::strncmp(sel, "[never]", 7) != 0 &&
           std::strncmp(sel, "[deny]", 6) != 0;
}

namespace {

/// mmap a MAP_SHARED fd at a 2 MB-ALIGNED address (reserve-and-trim): shmem
/// THP only PMD-maps when the virtual range is huge-page aligned — a plain
/// mmap(nullptr) comes back 4K-aligned and silently defeats MADV_HUGEPAGE
/// (measured: ShmemHugePages stayed 0). Returns MAP_FAILED on failure.
void* mmap_shared_aligned(size_t bytes, int fd) {
    constexpr size_t kHuge = 2UL << 20;
    const size_t over = bytes + kHuge;
    void* raw = ::mmap(nullptr, over, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) return MAP_FAILED;
    const uintptr_t base = reinterpret_cast<uintptr_t>(raw);
    const uintptr_t aligned = (base + kHuge - 1) & ~(kHuge - 1);
    void* ptr = ::mmap(reinterpret_cast<void*>(aligned), bytes,
                       PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
    if (ptr == MAP_FAILED) {
        ::munmap(raw, over);
        return MAP_FAILED;
    }
    // Trim the unused reservation edges (the MAP_FIXED window replaced the
    // middle of the PROT_NONE reservation).
    if (aligned > base) ::munmap(raw, aligned - base);
    const uintptr_t end = aligned + bytes;
    const uintptr_t raw_end = base + over;
    if (raw_end > end) ::munmap(reinterpret_cast<void*>(end), raw_end - end);
    return ptr;
}

/// MADV_HUGEPAGE with one-time fallback logging (P-24b THP; must run BEFORE
/// first touch — already-faulted 4K pages are not collapsed here).
void advise_hugepage(void* ptr, size_t size, bool thp_requested) {
    if (!thp_requested) return;
    // Some launchers (systemd units, sandboxes, wrappers) set
    // PR_SET_THP_DISABLE, inherited by every child — it silently defeats
    // MADV_HUGEPAGE even with shmem_enabled=advise (measured: an entire dev
    // session had it set; AnonHugePages/ShmemHugePages pinned at 0). The
    // config explicitly asked for THP, so clear the inherited flag once.
    static const bool thp_prctl_cleared = [] {
        if (::prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0) == 1) {
            if (::prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0) == 0) {
                spdlog::info("arena THP: cleared inherited PR_SET_THP_DISABLE");
                return true;
            }
            spdlog::warn("arena THP: PR_SET_THP_DISABLE inherited and could "
                         "not be cleared — huge pages unavailable");
            return false;
        }
        return true;
    }();
    (void)thp_prctl_cleared;
    static const bool available = NumaManager::shmem_thp_available();
    static bool warned = false;
    if (!available) {
        if (!warned) {
            spdlog::warn("arena THP requested but shmem_enabled disallows "
                         "madvise (see /sys/kernel/mm/transparent_hugepage/"
                         "shmem_enabled) — falling back to 4K pages");
            warned = true;
        }
        return;
    }
    if (::madvise(ptr, size, MADV_HUGEPAGE) != 0)
        spdlog::warn("madvise(MADV_HUGEPAGE, {} bytes) failed: {}", size,
                     strerror(errno));
}
}  // namespace

NumaBuffer NumaManager::allocate_on_node_shared(size_t size, int numa_node,
                                                const char* name) {
    const size_t aligned = align_to_page(size);
    const int fd = ::memfd_create(name ? name : "ls-arena", MFD_CLOEXEC);
    if (fd < 0) {
        spdlog::error("memfd_create({}) failed: {}", name ? name : "ls-arena",
                      strerror(errno));
        throw std::bad_alloc();
    }
    if (::ftruncate(fd, static_cast<off_t>(aligned)) != 0) {
        spdlog::error("ftruncate(memfd, {} bytes) failed: {}", aligned,
                      strerror(errno));
        ::close(fd);
        throw std::bad_alloc();
    }
    void* ptr = mmap_shared_aligned(aligned, fd);
    if (ptr == MAP_FAILED) {
        ::close(fd);
        throw std::bad_alloc();
    }
    // Same first-touch discipline as allocate_on_node: the policy is set now,
    // the tmpfs pages land on `numa_node` when the caller's prefault writes
    // fault them in. THP advice must precede that first touch (and the
    // mapping is 2 MB-aligned so PMD mappings are actually possible).
    advise_hugepage(ptr, aligned, shared_thp_);
    apply_mbind_bind(ptr, aligned, numa_node);

    total_allocated_.fetch_add(aligned, std::memory_order_relaxed);
    auto it = per_node_allocated_.find(numa_node);
    if (it != per_node_allocated_.end())
        it->second.fetch_add(aligned, std::memory_order_relaxed);
    else
        per_node_allocated_[-1].fetch_add(aligned, std::memory_order_relaxed);

    return NumaBuffer{ptr, aligned, numa_node, fd};
}

NumaBuffer NumaManager::adopt_shared(int fd, size_t size, int numa_node) {
    if (fd < 0 || size == 0)
        throw std::runtime_error("NumaManager::adopt_shared: bad fd/size");
    const size_t aligned = align_to_page(size);
    struct stat st{};
    if (::fstat(fd, &st) != 0 ||
        static_cast<size_t>(st.st_size) != aligned) {
        throw std::runtime_error(
            "NumaManager::adopt_shared: segment size mismatch (fstat " +
            std::to_string(st.st_size) + " vs expected " +
            std::to_string(aligned) + ")");
    }
    void* ptr = mmap_shared_aligned(aligned, fd);
    if (ptr == MAP_FAILED) {
        spdlog::error("adopt_shared: mmap({} bytes) failed: {}", aligned,
                      strerror(errno));
        throw std::bad_alloc();
    }
    // Re-assert the bind policy: no-op for the already-placed resident pages,
    // and it governs any pages faulted through THIS mapping later. THP advice
    // likewise applies only to future faults (a pre-THP store stays 4K until
    // a cold rebuild; khugepaged may collapse opportunistically).
    advise_hugepage(ptr, aligned, shared_thp_);
    apply_mbind_bind(ptr, aligned, numa_node);

    total_allocated_.fetch_add(aligned, std::memory_order_relaxed);
    auto it = per_node_allocated_.find(numa_node);
    if (it != per_node_allocated_.end())
        it->second.fetch_add(aligned, std::memory_order_relaxed);
    else
        per_node_allocated_[-1].fetch_add(aligned, std::memory_order_relaxed);

    return NumaBuffer{ptr, aligned, numa_node, fd};
}

// ── Deallocation ─────────────────────────────────────────────────────────────

void NumaManager::free(NumaBuffer& buf) {
    if (!buf.data)
        return;

    raw_munmap(buf.data, buf.size);
    if (buf.fd >= 0)
        ::close(buf.fd);  // shared backing (P-24b); pages live on in holder dups

    total_allocated_.fetch_sub(buf.size, std::memory_order_relaxed);
    auto it = per_node_allocated_.find(buf.numa_node);
    if (it != per_node_allocated_.end())
        it->second.fetch_sub(buf.size, std::memory_order_relaxed);
    else
        per_node_allocated_[-1].fetch_sub(buf.size, std::memory_order_relaxed);

    buf.data = nullptr;
    buf.size = 0;
    buf.numa_node = -1;
    buf.fd = -1;
}

// ── Migration ────────────────────────────────────────────────────────────────

void NumaManager::migrate(NumaBuffer& buf, int new_numa_node) {
    if (!buf.data)
        throw std::invalid_argument("NumaManager::migrate: null buffer");
    if (buf.numa_node == new_numa_node)
        return;

    NumaBuffer new_buf = allocate_on_node(buf.size, new_numa_node);
    std::memcpy(new_buf.data, buf.data, buf.size);
    free(buf);
    buf = new_buf;
}

// ── Statistics ───────────────────────────────────────────────────────────────

size_t NumaManager::total_allocated_bytes() const {
    return total_allocated_.load(std::memory_order_relaxed);
}

size_t NumaManager::allocated_bytes_on_node(int node) const {
    auto it = per_node_allocated_.find(node);
    if (it == per_node_allocated_.end())
        return 0;
    return it->second.load(std::memory_order_relaxed);
}

// ── ScopedThreadNodeBind ─────────────────────────────────────────────────────

ScopedThreadNodeBind::ScopedThreadNodeBind(NumaManager& numa, int numa_node) {
    if (numa_node < 0 || !numa.numa_available())
        return;  // no-op scope (non-HBM path stays untouched)
    CPU_ZERO(&saved_);
    if (sched_getaffinity(0, sizeof(saved_), &saved_) != 0) {
        spdlog::warn("ScopedThreadNodeBind: sched_getaffinity failed ({}); "
                     "not pinning to node {}", strerror(errno), numa_node);
        return;
    }
    numa.pin_current_thread_to_node(numa_node);
    active_ = true;
}

ScopedThreadNodeBind::~ScopedThreadNodeBind() {
    if (active_ && sched_setaffinity(0, sizeof(saved_), &saved_) != 0)
        spdlog::warn("ScopedThreadNodeBind: affinity restore failed ({})",
                     strerror(errno));
}

}  // namespace layerstorm::memory

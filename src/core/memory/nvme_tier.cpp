// NvmeTier — mmap-backed NVMe tier with per-expert-index file layout (WP-5).

#include "core/memory/nvme_tier.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <spdlog/spdlog.h>

#ifdef LAYERSTORM_HAS_URING
#include <liburing.h>
#endif

#include "model/weight_pipeline/prepacked_format.h"
#include "model/weight_pipeline/prepacked_source.h"

namespace layerstorm::memory {

namespace fs = std::filesystem;

// ── Constructor / Destructor ────────────────────────────────────────────────

NvmeTier::NvmeTier(Options opts, NumaManager& numa)
    : opts_(std::move(opts)), numa_(numa) {
    if (opts_.drive_paths.empty())
        throw std::invalid_argument("NvmeTier: drive_paths must not be empty");
    if (opts_.slot_size_bytes <= 0)
        throw std::invalid_argument("NvmeTier: slot_size_bytes must be > 0");
    if (opts_.num_moe_layers <= 0)
        throw std::invalid_argument("NvmeTier: num_moe_layers must be > 0");
    if (opts_.num_experts_per_layer <= 0)
        throw std::invalid_argument(
            "NvmeTier: num_experts_per_layer must be > 0");

    drives_.resize(opts_.drive_paths.size());
    for (size_t i = 0; i < opts_.drive_paths.size(); ++i) {
        drives_[i].path = opts_.drive_paths[i];

#ifdef LAYERSTORM_HAS_URING
        int ret = io_uring_queue_init(
            static_cast<unsigned>(opts_.queue_depth), &drives_[i].ring, 0);
        if (ret < 0) {
            for (size_t j = 0; j < i; ++j) {
                if (drives_[j].ring_initialized)
                    io_uring_queue_exit(&drives_[j].ring);
            }
            throw std::runtime_error(
                "NvmeTier: io_uring_queue_init failed for drive " +
                opts_.drive_paths[i] + ": " + std::strerror(-ret));
        }
        drives_[i].ring_initialized = true;
        drives_[i].num_inflight = 0;
#endif
    }

    // Initialize per-expert mmap regions and slot tracking.
    mmap_regions_.resize(static_cast<size_t>(opts_.num_experts_per_layer));
    slot_written_.resize(
        static_cast<size_t>(opts_.num_experts_per_layer),
        std::vector<bool>(static_cast<size_t>(opts_.num_moe_layers), false));

    // Scan and mmap any existing expert files.
    scan_and_mmap_existing_files();

    spdlog::info("NvmeTier: {} drive(s), {} bytes/slot, {} MoE layers, "
                 "{} experts, first_moe_layer={}",
                 num_drives(), opts_.slot_size_bytes, opts_.num_moe_layers,
                 opts_.num_experts_per_layer, opts_.first_moe_layer);
}

NvmeTier::~NvmeTier() {
    drain();

    // Free any deferred cancelled write buffers (defensive — drain() should
    // have emptied this, but guard against drain() failure).
    for (auto& [tok, cw] : cancelled_writes_) {
        if (cw.buf.data) numa_.free(cw.buf);
    }
    cancelled_writes_.clear();

    for (auto& region : mmap_regions_) {
        if (region.base && !region.borrowed) {
            ::munmap(region.base, region.size);
        }
        if (region.fd >= 0 && !region.borrowed) {
            ::close(region.fd);
        }
    }

#ifdef LAYERSTORM_HAS_URING
    for (auto& d : drives_) {
        if (d.ring_initialized)
            io_uring_queue_exit(&d.ring);
    }
#endif
}

// ── Startup / Shutdown ──────────────────────────────────────────────────────

void NvmeTier::init_directories() {
    for (const auto& d : drives_) {
        auto dir = model::prepacked::expert_dir(d.path);
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            spdlog::warn("NvmeTier: failed to create {}: {}",
                         dir.string(), ec.message());
            continue;
        }

        // Format-version marker: scan_and_mmap_existing_files() trusts any
        // right-sized pre-existing file, but slot CONTENT conventions change
        // across format versions (9.67.0 stores NVFP4 scales Sm1xx-interleaved
        // — old raw-scale cache files would be served as valid and produce
        // quietly-wrong GEMMs). On mismatch, invalidate the cache.
        const auto marker = dir / "FORMAT_VERSION";
        std::string on_disk;
        {
            std::ifstream f(marker);
            if (f) std::getline(f, on_disk);
        }
        if (on_disk != model::prepacked::kFormatVersion) {
            int removed = 0;
            for (const auto& entry : fs::directory_iterator(dir, ec)) {
                if (ec) break;
                if (entry.path().extension() == ".bin") {
                    fs::remove(entry.path(), ec);
                    if (!ec) ++removed;
                }
            }
            if (removed > 0 || !on_disk.empty()) {
                spdlog::warn("NvmeTier: cache format '{}' != engine '{}' — "
                             "invalidated {} cached expert file(s) in {}",
                             on_disk.empty() ? "<none>" : on_disk,
                             model::prepacked::kFormatVersion, removed,
                             dir.string());
            }
            std::ofstream f(marker, std::ios::trunc);
            if (f) f << model::prepacked::kFormatVersion << '\n';
        }
    }
}

void NvmeTier::drain() {
#ifdef LAYERSTORM_HAS_URING
    while (!pending_.empty() || !cancelled_writes_.empty()) {
        for (auto& drive : drives_) {
            if (drive.num_inflight == 0) continue;
            struct io_uring_cqe* cqe = nullptr;
            int ret = io_uring_wait_cqe(&drive.ring, &cqe);
            if (ret < 0) break;

            auto token = static_cast<IoToken>(cqe->user_data);
            auto it = pending_.find(token);
            if (it != pending_.end()) {
                auto& op = it->second;
                if (cqe->res >= 0) {
                    int lpos = layer_pos(op.key);
                    if (lpos >= 0) {
                        auto eidx = static_cast<size_t>(op.key.expert_idx);
                        auto lp = static_cast<size_t>(lpos);
                        if (!slot_written_[eidx][lp])
                            nvme_used_bytes_ += opts_.slot_size_bytes;
                        slot_written_[eidx][lp] = true;
                    }
                }
                if (op.write_buf.data) numa_.free(op.write_buf);
                if (op.fd >= 0) ::close(op.fd);
                inflight_keys_.erase(op.key);
                pending_.erase(it);
                drive.num_inflight--;
            } else {
                // CQE for a cancelled write.
                auto cit = cancelled_writes_.find(token);
                if (cit != cancelled_writes_.end()) {
                    if (cit->second.buf.data) numa_.free(cit->second.buf);
                    cancelled_writes_.erase(cit);
                }
                drive.num_inflight--;
            }
            io_uring_cqe_seen(&drive.ring, cqe);
        }
    }
#endif
}

void NvmeTier::attach_prepacked_source(
        const model::PrepackedSource& source) {
    auto regions = source.mmap_regions();

    for (size_t eidx = 0; eidx < regions.size() &&
                           eidx < mmap_regions_.size(); ++eidx) {
        const auto& src = regions[eidx];
        if (!src.base) continue;

        // Check if this expert file would live on one of our NVMe drives
        // by comparing the prepacked directory's expert file path with
        // our drive mount points.
        // Since PrepackedSource covers all experts from a single directory,
        // and it has already been validated, we borrow unconditionally.
        // The caller (engine.cpp) is responsible for only calling this when
        // the prepacked_dir is on an NVMe drive.
        auto& region = mmap_regions_[eidx];
        if (region.base) continue;  // already mapped (e.g. from scan)

        region.base = const_cast<void*>(
            static_cast<const void*>(src.base));
        region.size = src.size;
        region.fd = -1;          // borrowed — no fd for pwrite
        region.borrowed = true;

        // Mark all slots as written.
        auto& slots = slot_written_[eidx];
        std::fill(slots.begin(), slots.end(), true);
    }

    spdlog::info("NvmeTier: attached PrepackedSource ({} expert files)",
                 regions.size());
}

// ── Write: Host RAM → NVMe ──────────────────────────────────────────────────

std::optional<IoToken> NvmeTier::write_expert(ExpertKey key,
                                               const void* host_ptr) {
    if (!host_ptr) return std::nullopt;
    if (inflight_keys_.contains(key)) return std::nullopt;

    int lpos = layer_pos(key);
    if (lpos < 0) return std::nullopt;

    ensure_mmap(static_cast<int>(key.expert_idx));

#ifdef LAYERSTORM_HAS_URING
    int drv = drive_for_expert(key);
    auto& drive = drives_[drv];
    if (drive.num_inflight >= opts_.queue_depth) return std::nullopt;

    // Use the mmap region's fd for pwrite at the correct slot offset.
    auto& region = mmap_regions_[key.expert_idx];
    if (region.fd < 0) return std::nullopt;  // borrowed region — can't pwrite

    int64_t offset = model::prepacked::slot_offset(
        lpos, opts_.slot_size_bytes);

    struct io_uring_sqe* sqe = io_uring_get_sqe(&drive.ring);
    if (!sqe) return std::nullopt;

    IoToken token = next_token_++;
    io_uring_prep_write(sqe, region.fd, host_ptr,
                        static_cast<unsigned>(opts_.slot_size_bytes),
                        static_cast<__u64>(offset));
    io_uring_sqe_set_data64(sqe, token);
    io_uring_submit(&drive.ring);

    PendingOp op;
    op.token = token;
    op.key = key;
    op.drive_idx = drv;
    // Non-owning write — fd is shared from mmap_regions_, don't close per-op.
    op.fd = -1;
    pending_[token] = std::move(op);
    inflight_keys_[key] = token;
    drive.num_inflight++;

    return token;
#else
    write_expert_sync(key, host_ptr);
    return IoToken{0};
#endif
}

std::optional<IoToken> NvmeTier::write_expert(ExpertKey key, NumaBuffer buf) {
    if (!buf.data) return std::nullopt;
    if (inflight_keys_.contains(key)) {
        numa_.free(buf);
        return std::nullopt;
    }

    int lpos = layer_pos(key);
    if (lpos < 0) {
        numa_.free(buf);
        return std::nullopt;
    }

    ensure_mmap(static_cast<int>(key.expert_idx));

#ifdef LAYERSTORM_HAS_URING
    int drv = drive_for_expert(key);
    auto& drive = drives_[drv];

    auto& region = mmap_regions_[key.expert_idx];
    if (drive.num_inflight < opts_.queue_depth && region.fd >= 0) {
        int64_t offset = model::prepacked::slot_offset(
            lpos, opts_.slot_size_bytes);

        struct io_uring_sqe* sqe = io_uring_get_sqe(&drive.ring);
        if (sqe) {
            IoToken token = next_token_++;
            io_uring_prep_write(sqe, region.fd, buf.data,
                                static_cast<unsigned>(opts_.slot_size_bytes),
                                static_cast<__u64>(offset));
            io_uring_sqe_set_data64(sqe, token);
            io_uring_submit(&drive.ring);

            PendingOp op;
            op.token = token;
            op.key = key;
            op.drive_idx = drv;
            op.write_buf = std::move(buf);
            op.fd = -1;  // shared fd from mmap_regions_
            pending_[token] = std::move(op);
            inflight_keys_[key] = token;
            drive.num_inflight++;
            return token;
        }
    }
#endif
    // Async submission failed or io_uring unavailable — sync fallback.
    write_expert_sync(key, buf.data);
    numa_.free(buf);
    return IoToken{0};
}

// ── Read: NVMe → Host RAM (mmap + madvise) ─────────────────────────────────

std::optional<IoToken> NvmeTier::read_expert(ExpertKey key, int /*gpu_hint*/) {
    int lpos = layer_pos(key);
    if (lpos < 0) return std::nullopt;

    auto eidx = static_cast<size_t>(key.expert_idx);
    if (eidx >= mmap_regions_.size()) return std::nullopt;

    auto& region = mmap_regions_[eidx];
    if (!region.base) return std::nullopt;
    if (!slot_written_[eidx][static_cast<size_t>(lpos)]) return std::nullopt;

    // Already available via mmap — issue prefetch hint.
    auto* slot_ptr = static_cast<char*>(region.base) +
                     model::prepacked::slot_offset(lpos, opts_.slot_size_bytes);
    ::madvise(slot_ptr, static_cast<size_t>(opts_.slot_size_bytes),
              MADV_WILLNEED);

    // Return immediate completion sentinel.
    return IoToken{0};
}

// ── Completion polling ──────────────────────────────────────────────────────

std::vector<IoCompletion> NvmeTier::poll_completions() {
    std::vector<IoCompletion> completions;

#ifdef LAYERSTORM_HAS_URING
    for (auto& drive : drives_) {
        if (drive.num_inflight == 0) continue;

        // Flush kernel-side completions (non-blocking).
        {
            struct io_uring_cqe* flush_cqe = nullptr;
            struct __kernel_timespec ts = {0, 0};
            io_uring_wait_cqe_timeout(&drive.ring, &flush_cqe, &ts);
        }

        struct io_uring_cqe* cqe = nullptr;
        while (io_uring_peek_cqe(&drive.ring, &cqe) == 0) {
            auto token = static_cast<IoToken>(cqe->user_data);
            auto it = pending_.find(token);

            if (it != pending_.end()) {
                auto& op = it->second;
                bool success = (cqe->res >= 0 &&
                                cqe->res >= opts_.slot_size_bytes);

                IoCompletion comp;
                comp.token = token;
                comp.key = op.key;
                comp.success = success;
                comp.error_code = success ? 0 : (cqe->res < 0 ? -cqe->res : 0);
                comp.op = IoCompletion::Op::kWrite;

                if (success) {
                    int lpos = layer_pos(op.key);
                    if (lpos >= 0) {
                        auto eidx = static_cast<size_t>(op.key.expert_idx);
                        auto lp = static_cast<size_t>(lpos);
                        if (!slot_written_[eidx][lp])
                            nvme_used_bytes_ += opts_.slot_size_bytes;
                        slot_written_[eidx][lp] = true;
                    }
                }

                if (op.write_buf.data) numa_.free(op.write_buf);
                // op.fd is -1 (shared from mmap_regions_), don't close.
                inflight_keys_.erase(op.key);
                pending_.erase(it);
                drive.num_inflight--;
                completions.push_back(comp);
            } else {
                // CQE for a cancelled write — free the deferred buffer.
                auto cit = cancelled_writes_.find(token);
                if (cit != cancelled_writes_.end()) {
                    if (cit->second.buf.data) numa_.free(cit->second.buf);
                    cancelled_writes_.erase(cit);
                }
                drive.num_inflight--;
            }

            io_uring_cqe_seen(&drive.ring, cqe);
        }
    }
#endif

    return completions;
}

// ── Cancellation ────────────────────────────────────────────────────────────

bool NvmeTier::cancel(IoToken token) {
    auto it = pending_.find(token);
    if (it == pending_.end()) return false;

    auto op = std::move(it->second);
    pending_.erase(it);
    inflight_keys_.erase(op.key);

    // Defer buffer free until the CQE arrives — the kernel may still be
    // reading from write_buf for the pwrite.  io_uring guarantees a CQE
    // for every submitted SQE, so the buffer will be freed in
    // poll_completions() or drain().  Do NOT decrement num_inflight here;
    // let the CQE arrival handle it.
    cancelled_writes_[token] = {std::move(op.write_buf), op.drive_idx};

#ifdef LAYERSTORM_HAS_URING
    if (op.drive_idx >= 0 &&
        op.drive_idx < static_cast<int>(drives_.size())) {
        auto& drive = drives_[op.drive_idx];
        struct io_uring_sqe* sqe = io_uring_get_sqe(&drive.ring);
        if (sqe) {
            io_uring_prep_cancel64(sqe, token, 0);
            io_uring_sqe_set_data64(sqe, 0);
            io_uring_submit(&drive.ring);
        }
    }
#endif

    return true;
}

// ── Host pointer access ─────────────────────────────────────────────────────

const void* NvmeTier::host_ptr(ExpertKey key) const {
    int lpos = layer_pos(key);
    if (lpos < 0) return nullptr;

    auto eidx = static_cast<size_t>(key.expert_idx);
    if (eidx >= mmap_regions_.size()) return nullptr;

    const auto& region = mmap_regions_[eidx];
    if (!region.base) return nullptr;
    if (!slot_written_[eidx][static_cast<size_t>(lpos)]) return nullptr;

    return static_cast<const char*>(region.base) +
           model::prepacked::slot_offset(lpos, opts_.slot_size_bytes);
}

// ── Tier queries ────────────────────────────────────────────────────────────

ExpertTier NvmeTier::tier(ExpertKey key) const {
    if (inflight_keys_.contains(key)) return ExpertTier::kInflight;

    int lpos = layer_pos(key);
    if (lpos < 0) return ExpertTier::kNone;

    auto eidx = static_cast<size_t>(key.expert_idx);
    if (eidx >= mmap_regions_.size()) return ExpertTier::kNone;

    if (mmap_regions_[eidx].base &&
        slot_written_[eidx][static_cast<size_t>(lpos)])
        return ExpertTier::kHostRam;

    return ExpertTier::kNone;
}

bool NvmeTier::is_on_nvme(ExpertKey key) const {
    // With mmap, "on NVMe" and "in host RAM" are synonymous — the mmap
    // is backed by the NVMe file.  Return true if the slot has data.
    return tier(key) == ExpertTier::kHostRam;
}

bool NvmeTier::is_in_host_ram(ExpertKey key) const {
    return tier(key) == ExpertTier::kHostRam;
}

bool NvmeTier::is_inflight(ExpertKey key) const {
    return inflight_keys_.contains(key);
}

// ── NUMA queries ────────────────────────────────────────────────────────────

int NvmeTier::host_numa_node(ExpertKey /*key*/) const {
    // Mmap pages are OS-managed — no fixed NUMA affinity.
    return -1;
}

// ── Latency estimation ──────────────────────────────────────────────────────

double NvmeTier::estimate_read_us() const {
    // Conservative: 5 GB/s sequential NVMe read.
    constexpr double kNvmeBwBytesPerUs = 5.0e9 / 1.0e6;  // 5000 bytes/us
    return static_cast<double>(opts_.slot_size_bytes) / kNvmeBwBytesPerUs;
}

double NvmeTier::estimate_host_to_vram_us(int64_t slot_size_bytes,
                                           double pcie_bw_gbps) {
    double bw_bytes_per_us = pcie_bw_gbps * 1.0e9 / 1.0e6;
    return static_cast<double>(slot_size_bytes) / bw_bytes_per_us;
}

// ── Capacity queries ────────────────────────────────────────────────────────

int NvmeTier::num_drives() const {
    return static_cast<int>(drives_.size());
}

int64_t NvmeTier::slot_size_bytes() const { return opts_.slot_size_bytes; }

int64_t NvmeTier::total_nvme_capacity_bytes() const {
    return nvme_used_bytes_;
}

int64_t NvmeTier::total_nvme_used_bytes() const {
    return nvme_used_bytes_;
}

int NvmeTier::host_ram_entry_count() const {
    int count = 0;
    for (const auto& region : mmap_regions_) {
        if (region.base) ++count;
    }
    return count;
}

int64_t NvmeTier::host_ram_used_bytes() const {
    int64_t total = 0;
    for (size_t eidx = 0; eidx < slot_written_.size(); ++eidx) {
        for (bool written : slot_written_[eidx]) {
            if (written) total += opts_.slot_size_bytes;
        }
    }
    return total;
}

int64_t NvmeTier::host_ram_budget_bytes() const {
    return opts_.host_ram_budget_bytes;
}

int NvmeTier::inflight_count() const {
    return static_cast<int>(pending_.size());
}

// ── Drive assignment ────────────────────────────────────────────────────────

int NvmeTier::drive_for_expert(ExpertKey key) const {
    return static_cast<int>(
        static_cast<size_t>(key.expert_idx) %
        static_cast<size_t>(num_drives()));
}

std::string NvmeTier::expert_path(ExpertKey key) const {
    return expert_path_for(static_cast<int>(key.expert_idx));
}

// ── Private helpers ─────────────────────────────────────────────────────────

int NvmeTier::layer_pos(ExpertKey key) const {
    int abs_layer = static_cast<int>(key.layer_idx);
    int pos = abs_layer - opts_.first_moe_layer;
    if (pos < 0 || pos >= opts_.num_moe_layers) return -1;
    return pos;
}

void NvmeTier::ensure_mmap(int expert_idx) {
    auto eidx = static_cast<size_t>(expert_idx);
    if (eidx >= mmap_regions_.size()) return;
    if (mmap_regions_[eidx].base) return;  // already mapped

    std::string path = expert_path_for(expert_idx);

    // Ensure parent directory exists.
    {
        std::error_code ec;
        fs::create_directories(fs::path(path).parent_path(), ec);
    }

    auto expected = model::prepacked::expected_file_size(
        opts_.num_moe_layers, opts_.slot_size_bytes);

    // Open or create the file.
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP);
    if (fd < 0) {
        spdlog::warn("NvmeTier: open({}) failed: {}", path,
                     std::strerror(errno));
        return;
    }

    // Check current size.
    struct stat st{};
    if (::fstat(fd, &st) < 0) {
        ::close(fd);
        return;
    }

    // Pre-size the file if needed (fallocate avoids fragmentation).
    if (st.st_size < expected) {
#if defined(__linux__)
        int ret = ::fallocate(fd, 0, 0, expected);
        if (ret < 0) {
            // fallocate not supported on this filesystem — fall back to ftruncate.
            if (::ftruncate(fd, expected) < 0) {
                spdlog::warn("NvmeTier: ftruncate({}) failed: {}",
                             path, std::strerror(errno));
                ::close(fd);
                return;
            }
        }
#else
        if (::ftruncate(fd, expected) < 0) {
            spdlog::warn("NvmeTier: ftruncate({}) failed: {}",
                         path, std::strerror(errno));
            ::close(fd);
            return;
        }
#endif
    }

    void* base = ::mmap(nullptr, static_cast<size_t>(expected),
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        spdlog::warn("NvmeTier: mmap({}) failed: {}", path,
                     std::strerror(errno));
        ::close(fd);
        return;
    }

    // Random access pattern (any layer per cycle).
    ::madvise(base, static_cast<size_t>(expected), MADV_RANDOM);

    mmap_regions_[eidx] = {base, static_cast<size_t>(expected), fd, false};
}

void NvmeTier::scan_and_mmap_existing_files() {
    auto expected = model::prepacked::expected_file_size(
        opts_.num_moe_layers, opts_.slot_size_bytes);

    for (int eidx = 0; eidx < opts_.num_experts_per_layer; ++eidx) {
        if (mmap_regions_[static_cast<size_t>(eidx)].base) continue;

        std::string path = expert_path_for(eidx);
        std::error_code ec;
        if (!fs::exists(path, ec)) continue;

        auto file_size = fs::file_size(path, ec);
        if (ec || static_cast<int64_t>(file_size) != expected) {
            if (!ec) {
                spdlog::warn("NvmeTier: {} size {} != expected {} — skipping",
                             path, file_size, expected);
            }
            continue;
        }

        int fd = ::open(path.c_str(), O_RDWR, 0);
        if (fd < 0) continue;

        void* base = ::mmap(nullptr, static_cast<size_t>(expected),
                            PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            ::close(fd);
            continue;
        }

        ::madvise(base, static_cast<size_t>(expected), MADV_RANDOM);

        mmap_regions_[static_cast<size_t>(eidx)] = {
            base, static_cast<size_t>(expected), fd, false};

        // Mark all slots as written (pre-existing file is fully populated).
        auto& slots = slot_written_[static_cast<size_t>(eidx)];
        std::fill(slots.begin(), slots.end(), true);
        nvme_used_bytes_ += expected;

        spdlog::debug("NvmeTier: scanned and mmapped existing file {}", path);
    }
}

std::string NvmeTier::expert_path_for(int expert_idx) const {
    int drv = static_cast<int>(
        static_cast<size_t>(expert_idx) %
        static_cast<size_t>(num_drives()));
    return (model::prepacked::expert_dir(drives_[drv].path) /
            model::prepacked::expert_filename(expert_idx)).string();
}

void NvmeTier::write_expert_sync(ExpertKey key, const void* data) {
    int lpos = layer_pos(key);
    if (lpos < 0) return;

    ensure_mmap(static_cast<int>(key.expert_idx));

    auto eidx = static_cast<size_t>(key.expert_idx);
    auto& region = mmap_regions_[eidx];
    if (!region.base || region.fd < 0) return;

    int64_t offset = model::prepacked::slot_offset(
        lpos, opts_.slot_size_bytes);

    auto written = ::pwrite(region.fd, data,
                            static_cast<size_t>(opts_.slot_size_bytes),
                            offset);

    if (written == opts_.slot_size_bytes) {
        if (!slot_written_[eidx][static_cast<size_t>(lpos)])
            nvme_used_bytes_ += opts_.slot_size_bytes;
        slot_written_[eidx][static_cast<size_t>(lpos)] = true;
    } else {
        spdlog::warn("NvmeTier: sync pwrite for layer={} expert={} wrote "
                     "{}/{} bytes",
                     key.layer_idx, key.expert_idx, written,
                     opts_.slot_size_bytes);
    }
}

}  // namespace layerstorm::memory

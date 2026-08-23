// PrepackedSource — mmap-based reader for pre-processed expert files (WP-3).

#include "model/weight_pipeline/prepacked_source.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

#include "core/cuda_hardware_query.h"
#include "model/weight_pipeline/prepacked_format.h"

// O_DIRECT is a GNU extension; if the platform lacks it, treat as 0 (the
// requested O_DIRECT mode then degrades to buffered open, handled at ctor).
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

namespace layerstorm::model {

PrepackedSource::PrepackedSource(const std::filesystem::path& prepacked_dir,
                                 const QuantInterface& quant,
                                 bool direct_io, bool o_direct)
    : direct_io_(direct_io), o_direct_(o_direct) {
    namespace fs = std::filesystem;

    // 1. Read and validate manifest.
    manifest_ = read_manifest(prepacked_dir);
    auto vr = verify_manifest(manifest_, quant);
    if (!vr.ok) {
        throw std::runtime_error(
            "PrepackedSource manifest verification failed: " + vr.error);
    }

    // 2. Validate expert file count.
    if (manifest_.n_expert_files != manifest_.n_routed_experts) {
        throw std::runtime_error(
            "PrepackedSource: n_expert_files (" +
            std::to_string(manifest_.n_expert_files) +
            ") != n_routed_experts (" +
            std::to_string(manifest_.n_routed_experts) + ")");
    }

    // Use the on-disk STRIDE (padded, kSlotAlignBytes-aligned) for offsets, read
    // length, file-size and arena slot sizing. The padded tail is dead bytes; the
    // H2D copies only the real expert bytes (expert_cache->expert_bytes()).
    // stride_bytes is mandatory (verify_manifest rejects legacy unpadded data).
    slot_size_bytes_ = manifest_.slot.stride_bytes;
    const auto expected_size = prepacked::expected_file_size(
        manifest_.moe_layers.count, slot_size_bytes_);

    // Stage 2: O_DIRECT needs 4096-aligned slot size (offset = layer_pos*slot_size
    // and the read length are then aligned, as are the page-aligned arena slots).
    // Fall back to buffered pread+fadvise if the slot size is not aligned, or if
    // the platform lacks O_DIRECT.
    if (o_direct_ && (O_DIRECT == 0 || slot_size_bytes_ % 4096 != 0)) {
        spdlog::warn("PrepackedSource: O_DIRECT unavailable/slot_size {} not "
                     "4096-aligned — using buffered pread + fadvise(DONTNEED)",
                     slot_size_bytes_);
        o_direct_ = false;
    }

    // 3. Mmap each expert file.
    mmaps_.resize(static_cast<size_t>(manifest_.n_expert_files));

    for (int i = 0; i < manifest_.n_expert_files; ++i) {
        auto path = prepacked::expert_file_path(prepacked_dir, i);

        if (!fs::exists(path)) {
            throw std::runtime_error(
                "PrepackedSource: missing expert file: " + path.string());
        }

        // Validate file size.
        std::error_code ec;
        auto file_size = fs::file_size(path, ec);
        if (ec) {
            throw std::runtime_error(
                "PrepackedSource: cannot stat " + path.string() +
                ": " + ec.message());
        }
        if (static_cast<int64_t>(file_size) != expected_size) {
            throw std::runtime_error(
                "PrepackedSource: " + path.string() +
                " size " + std::to_string(file_size) +
                " != expected " + std::to_string(expected_size) +
                " (corrupt or partial write?)");
        }

        int oflags = O_RDONLY;
        if (direct_io_ && o_direct_) oflags |= O_DIRECT;
        int fd = ::open(path.c_str(), oflags);
        if (fd < 0) {
            throw std::runtime_error(
                "PrepackedSource: open() failed for " + path.string() +
                ": " + std::strerror(errno));
        }

        if (direct_io_) {
            // Stage 2: mmap-free. Keep the fd open; slots are read via pread so
            // the page cache never holds a second copy of the arena's data.
            mmaps_[static_cast<size_t>(i)] = {
                nullptr, static_cast<size_t>(file_size), false, -1, fd};
            continue;
        }

        // COW-writable mapping (MAP_PRIVATE): reads are zero-copy from the page
        // cache (no write ever happens), but PROT_WRITE is required for
        // cudaHostRegister to page-lock it for full-bandwidth DMA — a read-only
        // (PROT_READ) mapping cannot be registered (cudaErrorInvalidValue). 481-1.
        void* base = ::mmap(nullptr, static_cast<size_t>(file_size),
                            PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        ::close(fd);

        if (base == MAP_FAILED) {
            throw std::runtime_error(
                "PrepackedSource: mmap() failed for " + path.string() +
                ": " + std::strerror(errno));
        }

        // Expert access is random (any layer per cycle), not sequential.
        ::madvise(base, static_cast<size_t>(file_size), MADV_RANDOM);

        mmaps_[static_cast<size_t>(i)] = {
            base, static_cast<size_t>(file_size), false, -1, -1};
    }

    spdlog::debug("PrepackedSource: mmapped {} expert files from {}, "
                  "slot_size={} bytes, {} MoE layers",
                  manifest_.n_expert_files, prepacked_dir.string(),
                  slot_size_bytes_, manifest_.moe_layers.count);
}

PrepackedSource::~PrepackedSource() {
    unregister_pinned_dma();
    for (auto& region : mmaps_) {
        if (region.base) {
            ::munmap(region.base, region.size);
            region.base = nullptr;
        }
        if (region.fd >= 0) {       // Stage 2 direct mode: close the kept fd.
            ::close(region.fd);
            region.fd = -1;
        }
    }
}

const void* PrepackedSource::resolve(memory::ExpertKey key) const {
    if (key.expert_idx >= mmaps_.size()) return nullptr;
    const auto& region = mmaps_[key.expert_idx];
    if (!region.base) return nullptr;   // direct_io mode: no mmap, use pread_into
    int layer_pos = manifest_.moe_layers.layer_position(
        static_cast<int>(key.layer_idx));
    if (layer_pos < 0) return nullptr;
    return static_cast<const char*>(region.base)
           + prepacked::slot_offset(layer_pos, slot_size_bytes_);
}

bool PrepackedSource::load_into(memory::ExpertKey key, void* dst) const {
    if (direct_io_) return pread_into(key, dst);   // Stage 2: mmap-free path
    const void* src = resolve(key);
    if (!src || !dst) return false;
    std::memcpy(dst, src, static_cast<size_t>(slot_size_bytes_));
    return true;
}

bool PrepackedSource::pread_into(memory::ExpertKey key, void* dst) const {
    if (!dst || key.expert_idx >= mmaps_.size()) return false;
    int layer_pos = manifest_.moe_layers.layer_position(
        static_cast<int>(key.layer_idx));
    if (layer_pos < 0) return false;
    const MmapRegion& region = mmaps_[key.expert_idx];
    if (region.fd < 0) return false;
    const off_t off = static_cast<off_t>(
        prepacked::slot_offset(layer_pos, slot_size_bytes_));
    const size_t len = static_cast<size_t>(slot_size_bytes_);
    char* out = static_cast<char*>(dst);
    size_t done = 0;
    while (done < len) {
        ssize_t r = ::pread(region.fd, out + done, len - done,
                            off + static_cast<off_t>(done));
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) break;  // unexpected EOF
        done += static_cast<size_t>(r);
    }
    if (done != len) return false;
    // Buffered mode: drop the just-read pages so the page cache never holds a
    // second copy of arena data (O_DIRECT already bypasses the cache).
    if (!o_direct_)
        ::posix_fadvise(region.fd, off, static_cast<off_t>(len),
                        POSIX_FADV_DONTNEED);
    return true;
}

bool PrepackedSource::read_descriptor(memory::ExpertKey key, int& fd,
                                      int64_t& offset, int64_t& length) const {
    if (!direct_io_ || key.expert_idx >= mmaps_.size()) return false;
    int layer_pos = manifest_.moe_layers.layer_position(
        static_cast<int>(key.layer_idx));
    if (layer_pos < 0) return false;
    const MmapRegion& region = mmaps_[key.expert_idx];
    if (region.fd < 0) return false;
    fd = region.fd;
    offset = prepacked::slot_offset(layer_pos, slot_size_bytes_);
    length = slot_size_bytes_;   // the on-disk stride (aligned)
    return true;
}

bool PrepackedSource::has(memory::ExpertKey key) const {
    if (key.expert_idx >= mmaps_.size()) return false;
    return manifest_.moe_layers.layer_position(
        static_cast<int>(key.layer_idx)) >= 0;
}

std::optional<GgufExpertTypes> PrepackedSource::gguf_types_for_layer(
        int layer_idx) const {
    if (manifest_.gguf_types_per_layer.empty()) return std::nullopt;
    const int layer_pos = manifest_.moe_layers.layer_position(layer_idx);
    if (layer_pos < 0 ||
        static_cast<size_t>(layer_pos) >= manifest_.gguf_types_per_layer.size())
        return std::nullopt;
    return manifest_.gguf_types_per_layer[static_cast<size_t>(layer_pos)];
}

int PrepackedSource::register_for_pinned_dma() {
    int registered = 0;
    for (auto& region : mmaps_) {
        if (region.pinned || !region.base) continue;
        int err = core::host_register_pinned(region.base, region.size);
        if (err == 0) {
            region.pinned = true;
            ++registered;
        } else {
            spdlog::warn("PrepackedSource: cudaHostRegister failed for "
                         "region at {:p} ({} bytes), error {} — DMA will "
                         "use unpinned path",
                         region.base, region.size, err);
        }
    }
    return registered;
}

bool PrepackedSource::is_pinned(memory::ExpertKey key) const {
    if (key.expert_idx >= mmaps_.size()) return false;
    return mmaps_[key.expert_idx].pinned;
}

void PrepackedSource::advise_dontneed() const {
    size_t dropped = 0, bytes = 0;
    for (const auto& region : mmaps_) {
        if (region.pinned || !region.base || region.size == 0) continue;
        if (::madvise(region.base, region.size, MADV_DONTNEED) == 0) {
            ++dropped;
            bytes += region.size;
        }
    }
    spdlog::info("PrepackedSource: MADV_DONTNEED on {} regions ({:.1f} GB) — "
                 "dropped prepacked page cache (re-faults from file on access)",
                 dropped, bytes / 1073741824.0);
}

void PrepackedSource::unregister_pinned_dma() {
    for (auto& region : mmaps_) {
        if (!region.pinned || !region.base) continue;
        core::host_unregister_pinned(region.base);
        region.pinned = false;
    }
}

}  // namespace layerstorm::model

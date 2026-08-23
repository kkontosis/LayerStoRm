#include "model/weight_loader/safetensors_reader.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace layerstorm::model {

// ── SafetensorsDtype helpers ─────────────────────────────────────────────────

size_t dtype_size(SafetensorsDtype dtype) {
    switch (dtype) {
        case SafetensorsDtype::F64:      return 8;
        case SafetensorsDtype::F32:      return 4;
        case SafetensorsDtype::F16:      return 2;
        case SafetensorsDtype::BF16:     return 2;
        case SafetensorsDtype::F8_E4M3:  return 1;
        case SafetensorsDtype::F8_E5M2:  return 1;
        case SafetensorsDtype::U8:       return 1;
        case SafetensorsDtype::U16:      return 2;
        case SafetensorsDtype::I32:      return 4;
        case SafetensorsDtype::I64:      return 8;
        case SafetensorsDtype::BOOL:     return 1;
    }
    return 0;
}

std::optional<SafetensorsDtype> parse_dtype(std::string_view s) {
    if (s == "F64")      return SafetensorsDtype::F64;
    if (s == "F32")      return SafetensorsDtype::F32;
    if (s == "F16")      return SafetensorsDtype::F16;
    if (s == "BF16")     return SafetensorsDtype::BF16;
    if (s == "F8_E4M3")  return SafetensorsDtype::F8_E4M3;
    if (s == "F8_E5M2")  return SafetensorsDtype::F8_E5M2;
    if (s == "U8")       return SafetensorsDtype::U8;
    if (s == "U16")      return SafetensorsDtype::U16;
    if (s == "I32")      return SafetensorsDtype::I32;
    if (s == "I64")      return SafetensorsDtype::I64;
    if (s == "BOOL")     return SafetensorsDtype::BOOL;
    return std::nullopt;
}

std::string_view dtype_name(SafetensorsDtype dtype) {
    switch (dtype) {
        case SafetensorsDtype::F64:      return "F64";
        case SafetensorsDtype::F32:      return "F32";
        case SafetensorsDtype::F16:      return "F16";
        case SafetensorsDtype::BF16:     return "BF16";
        case SafetensorsDtype::F8_E4M3:  return "F8_E4M3";
        case SafetensorsDtype::F8_E5M2:  return "F8_E5M2";
        case SafetensorsDtype::U8:       return "U8";
        case SafetensorsDtype::U16:      return "U16";
        case SafetensorsDtype::I32:      return "I32";
        case SafetensorsDtype::I64:      return "I64";
        case SafetensorsDtype::BOOL:     return "BOOL";
    }
    return "unknown";
}

// ── Header parsing (shared by open() and read_header()) ──────────────────────

namespace {

struct ParsedHeader {
    std::vector<TensorEntry> entries;
    uint64_t header_size;
};

ParsedHeader parse_safetensors_header(int fd, const std::filesystem::path& path) {
    // Read 8-byte header size (little-endian u64)
    uint64_t header_size = 0;
    if (::pread(fd, &header_size, 8, 0) != 8) {
        throw std::runtime_error("Failed to read safetensors header size from " + path.string());
    }

    // Sanity check: header shouldn't be larger than 100MB
    if (header_size > 100 * 1024 * 1024) {
        throw std::runtime_error("Safetensors header too large (" +
                                 std::to_string(header_size) + " bytes) in " + path.string());
    }

    // Read JSON header
    std::string header_json(header_size, '\0');
    ssize_t n_read = ::pread(fd, header_json.data(), header_size, 8);
    if (n_read < 0 || static_cast<uint64_t>(n_read) != header_size) {
        throw std::runtime_error("Failed to read safetensors JSON header from " + path.string());
    }

    auto header = nlohmann::json::parse(header_json);

    std::vector<TensorEntry> entries;
    entries.reserve(header.size());

    for (auto& [name, info] : header.items()) {
        if (name == "__metadata__") continue;

        auto dtype_str = info.at("dtype").get<std::string>();
        auto dtype_opt = parse_dtype(dtype_str);
        if (!dtype_opt) {
            spdlog::warn("Unrecognized dtype '{}' for tensor '{}' in {}, skipping",
                         dtype_str, name, path.string());
            continue;
        }

        auto& shape_arr = info.at("shape");
        std::vector<int64_t> shape;
        shape.reserve(shape_arr.size());
        for (auto& dim : shape_arr) {
            shape.push_back(dim.get<int64_t>());
        }

        auto& offsets = info.at("data_offsets");
        size_t start = offsets.at(0).get<size_t>();
        size_t end = offsets.at(1).get<size_t>();

        entries.push_back(TensorEntry{
            .name = name,
            .dtype = *dtype_opt,
            .shape = std::move(shape),
            .data_offset = start,
            .data_size_bytes = end - start,
        });
    }

    return ParsedHeader{std::move(entries), header_size};
}

}  // namespace

// ── SafetensorsReader ────────────────────────────────────────────────────────

SafetensorsReader::~SafetensorsReader() {
    close();
}

SafetensorsReader::SafetensorsReader(SafetensorsReader&& other) noexcept
    : path_(std::move(other.path_)),
      entries_(std::move(other.entries_)),
      use_mmap_(other.use_mmap_),
      mapped_data_(other.mapped_data_),
      mapped_size_(other.mapped_size_),
      data_region_offset_(other.data_region_offset_) {
    other.mapped_data_ = nullptr;
    other.mapped_size_ = 0;
}

SafetensorsReader& SafetensorsReader::operator=(SafetensorsReader&& other) noexcept {
    if (this != &other) {
        close();
        path_ = std::move(other.path_);
        entries_ = std::move(other.entries_);
        use_mmap_ = other.use_mmap_;
        mapped_data_ = other.mapped_data_;
        mapped_size_ = other.mapped_size_;
        data_region_offset_ = other.data_region_offset_;
        other.mapped_data_ = nullptr;
        other.mapped_size_ = 0;
    }
    return *this;
}

void SafetensorsReader::close() {
    if (mapped_data_) {
        if (use_mmap_) {
            ::munmap(mapped_data_, mapped_size_);
        } else {
            delete[] static_cast<std::byte*>(mapped_data_);
        }
        mapped_data_ = nullptr;
        mapped_size_ = 0;
    }
}

SafetensorsReader SafetensorsReader::open(const std::filesystem::path& path, bool use_mmap) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Cannot open safetensors file " + path.string() +
                                 ": " + std::strerror(errno));
    }

    // Get file size
    struct stat st {};
    if (::fstat(fd, &st) < 0) {
        ::close(fd);
        throw std::runtime_error("Cannot stat safetensors file " + path.string() +
                                 ": " + std::strerror(errno));
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    // Parse header
    auto parsed = parse_safetensors_header(fd, path);
    size_t data_start = 8 + parsed.header_size;

    SafetensorsReader reader;
    reader.path_ = path;
    reader.entries_ = std::move(parsed.entries);
    reader.use_mmap_ = use_mmap;
    reader.data_region_offset_ = data_start;

    if (use_mmap) {
        void* mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);

        if (mapped == MAP_FAILED) {
            throw std::runtime_error("mmap failed for " + path.string() +
                                     ": " + std::strerror(errno));
        }

        ::madvise(static_cast<char*>(mapped) + data_start,
                  file_size - data_start, MADV_SEQUENTIAL);

        reader.mapped_data_ = mapped;
        reader.mapped_size_ = file_size;
    } else {
        // Read the entire file into a heap buffer
        auto* buf = new std::byte[file_size];
        size_t total_read = 0;
        while (total_read < file_size) {
            ssize_t n = ::pread(fd, buf + total_read, file_size - total_read,
                                static_cast<off_t>(total_read));
            if (n <= 0) {
                delete[] buf;
                ::close(fd);
                throw std::runtime_error("Failed to read safetensors file " + path.string() +
                                         ": " + std::strerror(errno));
            }
            total_read += static_cast<size_t>(n);
        }
        ::close(fd);

        reader.mapped_data_ = buf;
        reader.mapped_size_ = file_size;
    }

    return reader;
}

std::vector<TensorEntry> SafetensorsReader::read_header(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Cannot open safetensors file " + path.string() +
                                 ": " + std::strerror(errno));
    }
    auto parsed = parse_safetensors_header(fd, path);
    ::close(fd);
    return std::move(parsed.entries);
}

std::span<const std::byte> SafetensorsReader::tensor_data(const TensorEntry& entry) const {
    if (!mapped_data_) {
        throw std::runtime_error("SafetensorsReader not open, cannot read tensor data");
    }
    auto* base = static_cast<const std::byte*>(mapped_data_);
    return {base + data_region_offset_ + entry.data_offset, entry.data_size_bytes};
}

// ── ShardIndex ───────────────────────────────────────────────────────────────

ShardIndex read_shard_index(const std::filesystem::path& model_dir) {
    auto index_path = model_dir / "model.safetensors.index.json";

    if (std::filesystem::exists(index_path)) {
        // Sharded model
        std::ifstream ifs(index_path);
        if (!ifs) {
            throw std::runtime_error("Cannot open shard index " + index_path.string());
        }
        auto idx = nlohmann::json::parse(ifs);

        ShardIndex result;
        auto& wm = idx.at("weight_map");
        result.tensor_to_shard.reserve(wm.size());

        std::set<std::string> seen_shards;
        for (auto& [tensor_name, shard_file] : wm.items()) {
            auto shard = shard_file.get<std::string>();
            result.tensor_to_shard.emplace_back(tensor_name, shard);
            if (seen_shards.insert(shard).second) {
                result.shard_files.push_back(shard);
            }
        }

        std::sort(result.shard_files.begin(), result.shard_files.end());
        spdlog::info("Found sharded model: {} shards, {} tensors in index",
                     result.shard_files.size(), result.tensor_to_shard.size());
        return result;
    }

    // Single-file model
    auto single_path = model_dir / "model.safetensors";
    if (std::filesystem::exists(single_path)) {
        ShardIndex result;
        result.shard_files.push_back("model.safetensors");
        spdlog::info("Found single-file model: {}", single_path.string());
        return result;
    }

    throw std::runtime_error("No safetensors files found in " + model_dir.string() +
                             " (looked for model.safetensors.index.json and model.safetensors)");
}

}  // namespace layerstorm::model

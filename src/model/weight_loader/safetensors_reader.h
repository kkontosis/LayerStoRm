#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace layerstorm::model {

// ── SafetensorsDtype ─────────────────────────────────────────────────────────
// Data types found in safetensors files.

enum class SafetensorsDtype {
    F64,
    F32,
    F16,
    BF16,
    F8_E4M3,
    F8_E5M2,
    U8,
    U16,
    I32,
    I64,
    BOOL,
};

/// Bytes per element for a safetensors dtype.
size_t dtype_size(SafetensorsDtype dtype);

/// Parse a dtype string from safetensors JSON header.
/// Returns nullopt if unrecognized.
std::optional<SafetensorsDtype> parse_dtype(std::string_view s);

/// String name for a dtype (for logging).
std::string_view dtype_name(SafetensorsDtype dtype);

// ── TensorEntry ──────────────────────────────────────────────────────────────
// Metadata for one tensor in a safetensors file, parsed from the JSON header.

struct TensorEntry {
    std::string name;
    SafetensorsDtype dtype;
    std::vector<int64_t> shape;
    size_t data_offset;      // Byte offset from start of data region
    size_t data_size_bytes;  // Total bytes of this tensor's data
};

// ── SafetensorsReader ────────────────────────────────────────────────────────
// Reads a single safetensors file. Supports two modes:
//   - mmap (default): zero-copy access via memory-mapped file
//   - pread: data read into heap-allocated buffers (no mmap)

class SafetensorsReader {
public:
    SafetensorsReader() = default;
    ~SafetensorsReader();

    // Move-only (owns mmap or heap buffer)
    SafetensorsReader(SafetensorsReader&& other) noexcept;
    SafetensorsReader& operator=(SafetensorsReader&& other) noexcept;
    SafetensorsReader(const SafetensorsReader&) = delete;
    SafetensorsReader& operator=(const SafetensorsReader&) = delete;

    /// Open a single safetensors shard file.
    /// When use_mmap=true (default), uses mmap for zero-copy access.
    /// When use_mmap=false, reads the data region into a heap buffer via pread.
    /// Throws std::runtime_error on failure.
    static SafetensorsReader open(const std::filesystem::path& path, bool use_mmap = true);

    /// Parse only the JSON header without loading the data region.
    /// Useful for scanning tensor names without loading data.
    static std::vector<TensorEntry> read_header(const std::filesystem::path& path);

    /// All tensor entries in this file.
    const std::vector<TensorEntry>& entries() const { return entries_; }

    /// Access a tensor's raw data (zero-copy if mmap, from heap buffer otherwise).
    std::span<const std::byte> tensor_data(const TensorEntry& entry) const;

    /// Path of the opened file (for error messages).
    const std::filesystem::path& path() const { return path_; }

    /// Whether this reader has been opened.
    bool is_open() const { return mapped_data_ != nullptr; }

    /// Whether this reader was opened with mmap.
    bool is_mmap() const { return use_mmap_; }

private:
    std::filesystem::path path_;
    std::vector<TensorEntry> entries_;

    bool use_mmap_ = true;
    void* mapped_data_ = nullptr;
    size_t mapped_size_ = 0;
    size_t data_region_offset_ = 0;  // Byte offset where tensor data starts

    void close();
};

// ── ShardIndex ───────────────────────────────────────────────────────────────
// Parsed model.safetensors.index.json for sharded models.

struct ShardIndex {
    // tensor_name -> shard filename (relative to model directory)
    std::vector<std::pair<std::string, std::string>> tensor_to_shard;

    // Unique shard filenames in order
    std::vector<std::string> shard_files;
};

/// Read the shard index from a model directory.
/// Looks for model.safetensors.index.json. If not found, looks for a single
/// model.safetensors file. Throws on error.
ShardIndex read_shard_index(const std::filesystem::path& model_dir);

}  // namespace layerstorm::model

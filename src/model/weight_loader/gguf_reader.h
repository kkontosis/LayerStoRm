#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "model/quantization/gguf_kquant.h"
#include "model/weight_loader/safetensors_reader.h"

// ── Minimal, host-only GGUF container reader ─────────────────────────────────
//
// Parses a single `.gguf` file (magic + metadata KV + tensor infos + aligned
// data blob) so the weight loader can map GGUF tensors into the same
// `LoadedModel` bundles the safetensors path uses. This file is PURE host C++ —
// it MUST NOT include any CUDA/kernel header (INV-GPU-1). The byte format is
// taken from ref/llama.cpp/ggml/src/gguf.cpp + ggml/include/gguf.h; we do not
// vendor ggml.
//
// GGUF container layout (version 3):
//   "GGUF" magic (4) | version u32 | n_tensors i64 | n_kv i64
//   for each KV: key(str) | value_type u32 | value (array or scalar)
//   for each tensor: name(str) | n_dims u32 | dims[n_dims] u64 | ggml_type u32 | offset u64
//   <pad to general.alignment> | tensor data blob
//   str = len u64 + raw bytes (no NUL). Each tensor.offset is relative to the
//   blob start and is already alignment-padded.

namespace layerstorm::model {

// ── ggml_type subset ─────────────────────────────────────────────────────────
// The ggml_type enum values we recognize (mirrors ref/llama.cpp ggml.h). Only
// the six supported k-quants map to a GgufKQuantType; F32/F16/BF16 are kept as
// non-quantized passthrough (norms, scalars, embeddings may be stored float).

enum class GgmlType : int32_t {
    F32  = 0,
    F16  = 1,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    I32  = 26,  // integer passthrough (V4 ffn_gate_tid2eid routing table)
    BF16 = 30,
    MXFP4 = 39,  // OCP MX e2m1 + E8M0 scale (V4 QAT routed experts)
};

/// True if `ggml_type` is one of the six GGUF k-quants we support.
bool is_supported_gguf_kquant(int32_t ggml_type);

/// Map a raw ggml_type to GgufKQuantType. Throws for unsupported/non-k-quant.
GgufKQuantType gguf_kquant_from_ggml(int32_t ggml_type);

/// Map a non-quantized ggml_type (F32/F16/BF16/I32) to the engine
/// SafetensorsDtype. Throws for quantized or unknown types.
SafetensorsDtype gguf_float_dtype(int32_t ggml_type);

/// Human-readable name for a ggml_type (for logging/errors).
std::string gguf_ggml_type_name(int32_t ggml_type);

// ── GgufTensorEntry ──────────────────────────────────────────────────────────
// Metadata for one tensor parsed from the GGUF tensor-info section.

struct GgufTensorEntry {
    std::string name;
    int32_t ggml_type = 0;        ///< raw ggml_type from the file
    std::vector<int64_t> dims;    ///< GGUF order: dims[0] is fastest (columns)
    uint64_t data_offset = 0;     ///< byte offset within the data blob
    uint64_t data_size_bytes = 0; ///< total packed bytes of this tensor

    /// True if this tensor is one of the six supported GGUF k-quants.
    bool is_kquant() const { return is_supported_gguf_kquant(ggml_type); }

    /// The k-quant type (only valid when is_kquant()).
    GgufKQuantType kquant_type() const { return gguf_kquant_from_ggml(ggml_type); }
};

// ── GgufMetadataValue ────────────────────────────────────────────────────────
// A scalar metadata value (the loader only needs scalars: alignment, counts).
// Arrays are parsed and skipped (their element bytes are consumed) but not
// retained — none of the loader's routing depends on array metadata.

struct GgufMetadataValue {
    enum class Kind { u64, i64, f64, str, boolean } kind;
    uint64_t u = 0;
    int64_t  i = 0;
    double   f = 0.0;
    std::string s;
    bool     b = false;
};

// ── GgufReader ───────────────────────────────────────────────────────────────
// Reads a single GGUF file. mmap (default, zero-copy) or pread (heap buffer),
// mirroring SafetensorsReader.

class GgufReader {
public:
    GgufReader() = default;
    ~GgufReader();

    GgufReader(GgufReader&& other) noexcept;
    GgufReader& operator=(GgufReader&& other) noexcept;
    GgufReader(const GgufReader&) = delete;
    GgufReader& operator=(const GgufReader&) = delete;

    /// Open a single .gguf file. Throws std::runtime_error on failure.
    static GgufReader open(const std::filesystem::path& path, bool use_mmap = true);

    /// Parse a GGUF blob already in memory (used by tests building synthetic
    /// fixtures). The returned reader does NOT own `blob`; the caller must keep
    /// it alive for the reader's lifetime. Throws on malformed input.
    static GgufReader from_buffer(std::span<const std::byte> blob,
                                  std::string source_name = "<buffer>");

    /// All tensor entries.
    const std::vector<GgufTensorEntry>& entries() const { return entries_; }

    /// Zero-copy span of a tensor's packed data (from mmap/heap/buffer).
    std::span<const std::byte> tensor_data(const GgufTensorEntry& entry) const;

    /// MADV_DONTNEED the whole mmap'd shard — drops its (clean, read-only)
    /// page cache once every needed tensor has been uploaded to VRAM, so the
    /// GGUF no longer competes with the pinned arena for RAM. Pages re-fault
    /// from the file on any later access. No-op in pread/heap mode.
    void advise_dontneed() const;

    /// A scalar metadata value by key, or nullopt if absent/non-scalar.
    std::optional<GgufMetadataValue> metadata(std::string_view key) const;

    /// A SMALL numeric metadata array by key (integer element types →
    /// int64, float32/float64 → double), or nullopt if absent, non-array,
    /// non-numeric, or longer than kSmallArrayCap elements (huge arrays —
    /// tokenizer tables — are parsed-and-skipped as before, never copied).
    /// Ticket J: the dflash draft GGUF routes on `dflash.target_layers`
    /// and the swiglu-clamp arrays.
    static constexpr uint64_t kSmallArrayCap = 512;
    std::optional<std::vector<int64_t>> metadata_array_i64(
        std::string_view key) const;
    std::optional<std::vector<double>> metadata_array_f64(
        std::string_view key) const;

    /// The data-blob alignment (general.alignment, default 32).
    uint64_t alignment() const { return alignment_; }

    /// The opened file path (for error messages).
    const std::filesystem::path& path() const { return path_; }

    bool is_open() const { return data_ != nullptr; }

private:
    std::filesystem::path path_;
    std::vector<GgufTensorEntry> entries_;
    std::vector<std::pair<std::string, GgufMetadataValue>> metadata_;
    // Small numeric arrays retained at parse (<= kSmallArrayCap elements);
    // integer element types land in int_arrays_, float types in f64_arrays_.
    std::vector<std::pair<std::string, std::vector<int64_t>>> int_arrays_;
    std::vector<std::pair<std::string, std::vector<double>>> f64_arrays_;
    uint64_t alignment_ = 32;
    uint64_t data_blob_offset_ = 0;  ///< absolute offset of the data blob

    // Backing storage: either an mmap region or a heap buffer (pread) or a
    // borrowed external buffer (from_buffer; owns_ == false, mmap'd == false).
    const std::byte* data_ = nullptr;
    size_t data_size_ = 0;
    bool mmapped_ = false;
    bool owns_ = false;  // true: data_ is a heap allocation we must delete[]

    void parse(std::span<const std::byte> blob);
    void close();
};

}  // namespace layerstorm::model

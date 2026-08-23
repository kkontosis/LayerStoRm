#include "model/weight_loader/gguf_reader.h"

#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace layerstorm::model {

// ── ggml_type mapping ────────────────────────────────────────────────────────

bool is_supported_gguf_kquant(int32_t ggml_type) {
    switch (static_cast<GgmlType>(ggml_type)) {
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
        case GgmlType::Q6_K:
        case GgmlType::Q8_0:
        case GgmlType::MXFP4:
            return true;
        default:
            return false;
    }
}

GgufKQuantType gguf_kquant_from_ggml(int32_t ggml_type) {
    switch (static_cast<GgmlType>(ggml_type)) {
        case GgmlType::Q2_K: return GgufKQuantType::Q2_K;
        case GgmlType::Q3_K: return GgufKQuantType::Q3_K;
        case GgmlType::Q4_K: return GgufKQuantType::Q4_K;
        case GgmlType::Q5_K: return GgufKQuantType::Q5_K;
        case GgmlType::Q6_K: return GgufKQuantType::Q6_K;
        case GgmlType::Q8_0: return GgufKQuantType::Q8_0;
        case GgmlType::MXFP4: return GgufKQuantType::MXFP4;
        default:
            throw std::runtime_error(
                "gguf_kquant_from_ggml: ggml_type " + gguf_ggml_type_name(ggml_type)
                + " is not a supported GGUF k-quant (engine supports "
                  "Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/MXFP4)");
    }
}

SafetensorsDtype gguf_float_dtype(int32_t ggml_type) {
    switch (static_cast<GgmlType>(ggml_type)) {
        case GgmlType::F32:  return SafetensorsDtype::F32;
        case GgmlType::F16:  return SafetensorsDtype::F16;
        case GgmlType::BF16: return SafetensorsDtype::BF16;
        case GgmlType::I32:  return SafetensorsDtype::I32;
        default:
            throw std::runtime_error(
                "gguf_float_dtype: ggml_type " + gguf_ggml_type_name(ggml_type)
                + " is not a non-quantized passthrough type (F32/F16/BF16/I32)");
    }
}

std::string gguf_ggml_type_name(int32_t ggml_type) {
    switch (static_cast<GgmlType>(ggml_type)) {
        case GgmlType::F32:  return "F32";
        case GgmlType::F16:  return "F16";
        case GgmlType::Q8_0: return "Q8_0";
        case GgmlType::Q2_K: return "Q2_K";
        case GgmlType::Q3_K: return "Q3_K";
        case GgmlType::Q4_K: return "Q4_K";
        case GgmlType::Q5_K: return "Q5_K";
        case GgmlType::Q6_K: return "Q6_K";
        case GgmlType::I32:  return "I32";
        case GgmlType::MXFP4: return "MXFP4";
        case GgmlType::BF16: return "BF16";
        default: break;
    }
    return "ggml_type(" + std::to_string(ggml_type) + ")";
}

// ── Byte-cursor reader (little-endian, mirrors ref gguf.cpp::gguf_reader) ────

namespace {

constexpr uint64_t kGgufMaxStringLength = 1ull << 20;  // 1 MiB sanity cap

class Cursor {
public:
    Cursor(std::span<const std::byte> blob, std::string_view src)
        : data_(blob.data()), size_(blob.size()), src_(src) {}

    uint64_t pos() const { return pos_; }

    void need(uint64_t n) const {
        if (pos_ + n > size_ || pos_ + n < pos_) {
            throw std::runtime_error("GgufReader(" + std::string(src_)
                + "): truncated file (need " + std::to_string(n)
                + " bytes at offset " + std::to_string(pos_) + " of "
                + std::to_string(size_) + ")");
        }
    }

    template <typename T>
    T read_pod() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    std::string read_string() {
        uint64_t len = read_pod<uint64_t>();
        if (len > kGgufMaxStringLength) {
            throw std::runtime_error("GgufReader(" + std::string(src_)
                + "): string length " + std::to_string(len) + " exceeds cap");
        }
        need(len);
        std::string s(reinterpret_cast<const char*>(data_ + pos_),
                      static_cast<size_t>(len));
        pos_ += len;
        return s;
    }

    void skip(uint64_t n) {
        need(n);
        pos_ += n;
    }

    void seek(uint64_t absolute) {
        if (absolute > size_) {
            throw std::runtime_error("GgufReader(" + std::string(src_)
                + "): seek past end (" + std::to_string(absolute) + " > "
                + std::to_string(size_) + ")");
        }
        pos_ = absolute;
    }

private:
    const std::byte* data_;
    uint64_t size_;
    uint64_t pos_ = 0;
    std::string_view src_;
};

// GGUF KV value types (ref gguf.h::gguf_type).
enum class GgufKvType : int32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
    FLOAT64 = 12,
};

// Size in bytes of a fixed-width GGUF scalar type (0 for STRING/ARRAY).
uint64_t kv_scalar_size(GgufKvType t) {
    switch (t) {
        case GgufKvType::UINT8:   case GgufKvType::INT8:  case GgufKvType::BOOL:    return 1;
        case GgufKvType::UINT16:  case GgufKvType::INT16:                            return 2;
        case GgufKvType::UINT32:  case GgufKvType::INT32: case GgufKvType::FLOAT32:  return 4;
        case GgufKvType::UINT64:  case GgufKvType::INT64: case GgufKvType::FLOAT64:  return 8;
        case GgufKvType::STRING:  case GgufKvType::ARRAY:                            return 0;
    }
    return 0;
}

// Read one scalar KV value into a GgufMetadataValue. Strings are captured;
// numeric types are widened into u/i/f as appropriate.
GgufMetadataValue read_scalar_value(Cursor& cur, GgufKvType t, std::string_view src) {
    GgufMetadataValue v{};
    switch (t) {
        case GgufKvType::UINT8:   v.kind = GgufMetadataValue::Kind::u64; v.u = cur.read_pod<uint8_t>();  break;
        case GgufKvType::UINT16:  v.kind = GgufMetadataValue::Kind::u64; v.u = cur.read_pod<uint16_t>(); break;
        case GgufKvType::UINT32:  v.kind = GgufMetadataValue::Kind::u64; v.u = cur.read_pod<uint32_t>(); break;
        case GgufKvType::UINT64:  v.kind = GgufMetadataValue::Kind::u64; v.u = cur.read_pod<uint64_t>(); break;
        case GgufKvType::INT8:    v.kind = GgufMetadataValue::Kind::i64; v.i = cur.read_pod<int8_t>();   break;
        case GgufKvType::INT16:   v.kind = GgufMetadataValue::Kind::i64; v.i = cur.read_pod<int16_t>();  break;
        case GgufKvType::INT32:   v.kind = GgufMetadataValue::Kind::i64; v.i = cur.read_pod<int32_t>();  break;
        case GgufKvType::INT64:   v.kind = GgufMetadataValue::Kind::i64; v.i = cur.read_pod<int64_t>();  break;
        case GgufKvType::FLOAT32: v.kind = GgufMetadataValue::Kind::f64; v.f = cur.read_pod<float>();    break;
        case GgufKvType::FLOAT64: v.kind = GgufMetadataValue::Kind::f64; v.f = cur.read_pod<double>();   break;
        case GgufKvType::BOOL:    v.kind = GgufMetadataValue::Kind::boolean; v.b = (cur.read_pod<int8_t>() != 0); break;
        case GgufKvType::STRING:  v.kind = GgufMetadataValue::Kind::str; v.s = cur.read_string(); break;
        case GgufKvType::ARRAY:
            throw std::runtime_error("GgufReader(" + std::string(src)
                + "): nested array KV value not supported");
    }
    return v;
}

// Round `x` up to the next multiple of `align` (align is a power of 2 > 0).
uint64_t align_up(uint64_t x, uint64_t align) {
    if (align == 0) return x;
    return (x + align - 1) / align * align;
}

}  // namespace

// ── GgufReader ───────────────────────────────────────────────────────────────

GgufReader::~GgufReader() { close(); }

void GgufReader::close() {
    if (mmapped_ && data_) {
        ::munmap(const_cast<std::byte*>(data_), data_size_);
    } else if (owns_ && data_) {
        delete[] data_;
    }
    data_ = nullptr;
    data_size_ = 0;
    mmapped_ = false;
    owns_ = false;
}

GgufReader::GgufReader(GgufReader&& o) noexcept
    : path_(std::move(o.path_)),
      entries_(std::move(o.entries_)),
      metadata_(std::move(o.metadata_)),
      int_arrays_(std::move(o.int_arrays_)),
      f64_arrays_(std::move(o.f64_arrays_)),
      alignment_(o.alignment_),
      data_blob_offset_(o.data_blob_offset_),
      data_(o.data_),
      data_size_(o.data_size_),
      mmapped_(o.mmapped_),
      owns_(o.owns_) {
    o.data_ = nullptr;
    o.data_size_ = 0;
    o.mmapped_ = false;
    o.owns_ = false;
}

GgufReader& GgufReader::operator=(GgufReader&& o) noexcept {
    if (this != &o) {
        close();
        path_ = std::move(o.path_);
        entries_ = std::move(o.entries_);
        metadata_ = std::move(o.metadata_);
        int_arrays_ = std::move(o.int_arrays_);
        f64_arrays_ = std::move(o.f64_arrays_);
        alignment_ = o.alignment_;
        data_blob_offset_ = o.data_blob_offset_;
        data_ = o.data_;
        data_size_ = o.data_size_;
        mmapped_ = o.mmapped_;
        owns_ = o.owns_;
        o.data_ = nullptr;
        o.data_size_ = 0;
        o.mmapped_ = false;
        o.owns_ = false;
    }
    return *this;
}

GgufReader GgufReader::open(const std::filesystem::path& path, bool use_mmap) {
    GgufReader r;
    r.path_ = path;

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("GgufReader: cannot open " + path.string()
                                 + ": " + std::strerror(errno));
    }

    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        throw std::runtime_error("GgufReader: cannot stat " + path.string());
    }
    size_t fsize = static_cast<size_t>(st.st_size);

    if (use_mmap) {
        void* m = ::mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) {
            throw std::runtime_error("GgufReader: mmap failed for " + path.string()
                                     + ": " + std::strerror(errno));
        }
        r.data_ = static_cast<const std::byte*>(m);
        r.data_size_ = fsize;
        r.mmapped_ = true;
    } else {
        auto* buf = new std::byte[fsize];
        size_t total = 0;
        while (total < fsize) {
            ssize_t n = ::pread(fd, buf + total, fsize - total,
                                static_cast<off_t>(total));
            if (n <= 0) {
                delete[] buf;
                ::close(fd);
                throw std::runtime_error("GgufReader: pread failed for "
                                         + path.string());
            }
            total += static_cast<size_t>(n);
        }
        ::close(fd);
        r.data_ = buf;
        r.data_size_ = fsize;
        r.owns_ = true;
    }

    try {
        r.parse(std::span<const std::byte>(r.data_, r.data_size_));
    } catch (...) {
        r.close();
        throw;
    }
    return r;
}

GgufReader GgufReader::from_buffer(std::span<const std::byte> blob,
                                   std::string source_name) {
    GgufReader r;
    r.path_ = std::move(source_name);
    r.data_ = blob.data();
    r.data_size_ = blob.size();
    r.mmapped_ = false;
    r.owns_ = false;  // borrowed
    r.parse(blob);
    return r;
}

void GgufReader::parse(std::span<const std::byte> blob) {
    const std::string src = path_.string();
    Cursor cur(blob, src);

    // 1. Magic.
    char magic[4];
    for (int i = 0; i < 4; ++i) magic[i] = static_cast<char>(cur.read_pod<uint8_t>());
    if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
        throw std::runtime_error("GgufReader(" + src + "): bad magic (not a GGUF file)");
    }

    // 2. Version.
    uint32_t version = cur.read_pod<uint32_t>();
    if (version != 2 && version != 3) {
        throw std::runtime_error("GgufReader(" + src + "): unsupported GGUF version "
                                 + std::to_string(version) + " (engine supports 2/3)");
    }

    // 3/4. Tensor + KV counts (i64 in v3).
    int64_t n_tensors = cur.read_pod<int64_t>();
    int64_t n_kv = cur.read_pod<int64_t>();
    if (n_tensors < 0 || n_kv < 0) {
        throw std::runtime_error("GgufReader(" + src + "): negative tensor/KV count");
    }

    // 5. KV metadata.
    metadata_.reserve(static_cast<size_t>(n_kv));
    for (int64_t i = 0; i < n_kv; ++i) {
        std::string key = cur.read_string();
        auto vtype = static_cast<GgufKvType>(cur.read_pod<int32_t>());

        if (vtype == GgufKvType::ARRAY) {
            // Array: element type + count + elements. Huge/string arrays are
            // consumed-but-skipped (cursor alignment only); SMALL numeric
            // arrays (<= kSmallArrayCap elements) are RETAINED — ticket J:
            // the dflash draft GGUF routes on `dflash.target_layers` and the
            // swiglu-clamp arrays.
            auto elem_t = static_cast<GgufKvType>(cur.read_pod<int32_t>());
            uint64_t n = cur.read_pod<uint64_t>();
            if (elem_t == GgufKvType::STRING) {
                for (uint64_t e = 0; e < n; ++e) (void)cur.read_string();
            } else if (elem_t == GgufKvType::ARRAY) {
                throw std::runtime_error("GgufReader(" + src
                    + "): nested array element type not supported (key '" + key + "')");
            } else if (n <= kSmallArrayCap &&
                       (elem_t == GgufKvType::FLOAT32 ||
                        elem_t == GgufKvType::FLOAT64)) {
                std::vector<double> vals;
                vals.reserve(static_cast<size_t>(n));
                for (uint64_t e = 0; e < n; ++e)
                    vals.push_back(elem_t == GgufKvType::FLOAT32
                                       ? static_cast<double>(cur.read_pod<float>())
                                       : cur.read_pod<double>());
                f64_arrays_.emplace_back(std::move(key), std::move(vals));
            } else if (n <= kSmallArrayCap && elem_t != GgufKvType::BOOL) {
                std::vector<int64_t> vals;
                vals.reserve(static_cast<size_t>(n));
                for (uint64_t e = 0; e < n; ++e) {
                    switch (elem_t) {
                        case GgufKvType::UINT8:  vals.push_back(cur.read_pod<uint8_t>());  break;
                        case GgufKvType::INT8:   vals.push_back(cur.read_pod<int8_t>());   break;
                        case GgufKvType::UINT16: vals.push_back(cur.read_pod<uint16_t>()); break;
                        case GgufKvType::INT16:  vals.push_back(cur.read_pod<int16_t>());  break;
                        case GgufKvType::UINT32: vals.push_back(cur.read_pod<uint32_t>()); break;
                        case GgufKvType::INT32:  vals.push_back(cur.read_pod<int32_t>());  break;
                        case GgufKvType::UINT64: vals.push_back(static_cast<int64_t>(cur.read_pod<uint64_t>())); break;
                        case GgufKvType::INT64:  vals.push_back(cur.read_pod<int64_t>());  break;
                        default:
                            throw std::runtime_error("GgufReader(" + src
                                + "): unexpected array element type (key '" + key + "')");
                    }
                }
                int_arrays_.emplace_back(std::move(key), std::move(vals));
            } else {
                uint64_t sz = kv_scalar_size(elem_t);
                cur.skip(sz * n);
            }
            continue;
        }

        metadata_.emplace_back(std::move(key), read_scalar_value(cur, vtype, src));
    }

    // Resolve general.alignment (default 32).
    alignment_ = 32;
    for (const auto& [k, v] : metadata_) {
        if (k == "general.alignment") {
            if (v.kind == GgufMetadataValue::Kind::u64 && v.u > 0) {
                alignment_ = v.u;
            } else if (v.kind == GgufMetadataValue::Kind::i64 && v.i > 0) {
                alignment_ = static_cast<uint64_t>(v.i);
            }
            break;
        }
    }

    // 6. Tensor infos.
    entries_.reserve(static_cast<size_t>(n_tensors));
    for (int64_t i = 0; i < n_tensors; ++i) {
        GgufTensorEntry e;
        e.name = cur.read_string();
        uint32_t n_dims = cur.read_pod<uint32_t>();
        if (n_dims == 0 || n_dims > 4) {
            throw std::runtime_error("GgufReader(" + src + "): tensor '" + e.name
                + "' has invalid n_dims " + std::to_string(n_dims));
        }
        e.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            e.dims[d] = cur.read_pod<int64_t>();
            if (e.dims[d] <= 0) {
                throw std::runtime_error("GgufReader(" + src + "): tensor '" + e.name
                    + "' has non-positive dim");
            }
        }
        e.ggml_type = cur.read_pod<int32_t>();
        e.data_offset = cur.read_pod<uint64_t>();
        entries_.push_back(std::move(e));
    }

    // 7. The data blob begins at the first alignment boundary after the tensor
    // infos. Each tensor.offset is relative to the blob start.
    data_blob_offset_ = align_up(cur.pos(), alignment_);

    // Compute each tensor's packed byte size and validate it fits in the file.
    for (auto& e : entries_) {
        int64_t numel = 1;
        for (auto d : e.dims) numel *= d;

        if (e.is_kquant()) {
            // Packed k-quant bytes = numel/QK * block_bytes. numel must be a
            // multiple of QK (the leading dim — columns — is divisible by QK).
            GgufKQuantType kt = e.kquant_type();
            int qk = gguf::block_values(kt);
            if (numel % qk != 0) {
                throw std::runtime_error("GgufReader(" + src + "): tensor '" + e.name
                    + "' element count " + std::to_string(numel)
                    + " not a multiple of QK " + std::to_string(qk));
            }
            e.data_size_bytes = static_cast<uint64_t>(numel / qk)
                              * static_cast<uint64_t>(gguf::block_bytes(kt));
        } else {
            // Non-quantized: numel * element bytes.
            SafetensorsDtype dt = gguf_float_dtype(e.ggml_type);
            e.data_size_bytes = static_cast<uint64_t>(numel) * dtype_size(dt);
        }

        uint64_t end = data_blob_offset_ + e.data_offset + e.data_size_bytes;
        if (end > data_size_ || end < e.data_offset) {
            throw std::runtime_error("GgufReader(" + src + "): tensor '" + e.name
                + "' data [" + std::to_string(data_blob_offset_ + e.data_offset)
                + ", " + std::to_string(end) + ") exceeds file size "
                + std::to_string(data_size_));
        }
    }

    spdlog::debug("GgufReader({}): {} tensors, {} KV, alignment {}, blob @ {}",
                  src, entries_.size(), metadata_.size(), alignment_,
                  data_blob_offset_);
}

void GgufReader::advise_dontneed() const {
    if (!mmapped_ || !data_ || data_size_ == 0) return;
    ::madvise(const_cast<std::byte*>(data_), data_size_, MADV_DONTNEED);
}

std::span<const std::byte> GgufReader::tensor_data(const GgufTensorEntry& entry) const {
    uint64_t start = data_blob_offset_ + entry.data_offset;
    return std::span<const std::byte>(data_ + start,
                                      static_cast<size_t>(entry.data_size_bytes));
}

std::optional<GgufMetadataValue> GgufReader::metadata(std::string_view key) const {
    for (const auto& [k, v] : metadata_) {
        if (k == key) return v;
    }
    return std::nullopt;
}

std::optional<std::vector<int64_t>> GgufReader::metadata_array_i64(
        std::string_view key) const {
    for (const auto& [k, v] : int_arrays_) {
        if (k == key) return v;
    }
    return std::nullopt;
}

std::optional<std::vector<double>> GgufReader::metadata_array_f64(
        std::string_view key) const {
    for (const auto& [k, v] : f64_arrays_) {
        if (k == key) return v;
    }
    return std::nullopt;
}

}  // namespace layerstorm::model

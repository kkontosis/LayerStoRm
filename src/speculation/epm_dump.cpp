// EPM-1 dump writers. See epm_dump.h for record layouts. CUDA-free TU
// (INV-GPU-1): pure host-side serialization; the D2H copies happen at the
// capture sites through DeviceBackend.

#include "speculation/epm_dump.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <spdlog/spdlog.h>

#include "config/config_parser.h"

namespace layerstorm::speculation {

namespace {

constexpr uint32_t kEpmbMagic = 0x424D5045u;  // "EPMB" little-endian
constexpr uint32_t kEpmrMagic = 0x524D5045u;  // "EPMR" little-endian
constexpr uint32_t kEpmVersion = 1;

std::FILE* open_append(const std::string& dir, const char* name) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        spdlog::error("epm_dump: cannot create dump dir '{}': {} — dumping "
                      "disabled", dir, ec.message());
        return nullptr;
    }
    const std::string path = dir + "/" + name;
    std::FILE* fp = std::fopen(path.c_str(), "ab");
    if (!fp)
        spdlog::error("epm_dump: cannot open '{}' for append — dumping "
                      "disabled", path);
    return fp;
}

template <typename T>
void put(std::string& out, const T& v) {
    out.append(reinterpret_cast<const char*>(&v), sizeof(T));
}

}  // namespace

std::string epm_dump_dir(const config::DsparkConfig& dc) {
    const char* env = std::getenv("LS_EPM_DUMP");
    if (env && env[0]) {
        if (env[0] == '0' && env[1] == '\0') return {};  // forced OFF
        return env;
    }
    return dc.epm_dump_dir;
}

uint16_t epm_f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
    const uint32_t abs = x & 0x7FFFFFFFu;
    if (abs >= 0x7F800000u) {  // inf / nan (nan keeps a quiet mantissa bit)
        const uint16_t mant = (abs > 0x7F800000u) ? 0x200 : 0;
        return static_cast<uint16_t>(sign | 0x7C00u | mant);
    }
    const int e = static_cast<int>(abs >> 23) - 127;  // unbiased exponent
    const uint32_t m = abs & 0x7FFFFFu;
    if (e >= 16)  // >= 2^16 -> inf (max fp16 = 65504)
        return static_cast<uint16_t>(sign | 0x7C00u);
    if (e >= -14) {  // normal fp16 range; RNE with carry into exponent/inf
        uint32_t hm = m >> 13;
        const uint32_t rem = m & 0x1FFFu;
        uint16_t h = static_cast<uint16_t>(
            sign | (static_cast<uint32_t>(e + 15) << 10) | hm);
        if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) ++h;
        return h;
    }
    if (e >= -25) {  // subnormal fp16; RNE on the full dropped tail
        const uint32_t full = m | 0x800000u;  // implicit leading 1
        const int drop = 13 + (-14 - e);      // 14..24 bits dropped
        uint32_t hm = full >> drop;
        const uint32_t rem = full & ((1u << drop) - 1u);
        const uint32_t halfway = 1u << (drop - 1);
        uint16_t h = static_cast<uint16_t>(sign | hm);
        if (rem > halfway || (rem == halfway && (hm & 1u))) ++h;
        return h;
    }
    return sign;  // underflow to signed zero
}

// ── EpmBlockDumper ───────────────────────────────────────────────────────────

EpmBlockDumper::EpmBlockDumper(const std::string& dir) {
    fp_ = open_append(dir, "epm_blocks.bin");
    if (fp_)
        spdlog::info("epm_dump: block dumper armed -> {}/epm_blocks.bin",
                     dir);
}

EpmBlockDumper::~EpmBlockDumper() {
    if (fp_) std::fclose(fp_);
}

void EpmBlockDumper::write_block(uint64_t seq_id, uint32_t block_idx,
                                 uint32_t anchor_pos, uint32_t anchor_token,
                                 int gamma, int n_layers, int hidden,
                                 const uint16_t* hiddens_bf16,
                                 const int32_t* draft_ids, const float* conf) {
    if (!fp_ || gamma <= 0 || n_layers <= 0 || hidden <= 0 || !hiddens_bf16 ||
        !draft_ids)
        return;
    std::string rec;
    const size_t n_hid = static_cast<size_t>(gamma) *
                         static_cast<size_t>(n_layers) *
                         static_cast<size_t>(hidden);
    rec.reserve(48 + static_cast<size_t>(gamma) * 8 + n_hid * 2);
    put(rec, kEpmbMagic);
    put(rec, kEpmVersion);
    put(rec, seq_id);
    put(rec, block_idx);
    put(rec, anchor_pos);
    put(rec, anchor_token);
    put(rec, static_cast<uint32_t>(gamma));
    put(rec, static_cast<uint32_t>(n_layers));
    put(rec, static_cast<uint32_t>(hidden));
    put(rec, static_cast<uint32_t>(conf ? 1 : 0));
    rec.append(reinterpret_cast<const char*>(draft_ids),
               static_cast<size_t>(gamma) * sizeof(int32_t));
    if (conf)
        rec.append(reinterpret_cast<const char*>(conf),
                   static_cast<size_t>(gamma) * sizeof(float));
    rec.append(reinterpret_cast<const char*>(hiddens_bf16), n_hid * 2);
    if (std::fwrite(rec.data(), 1, rec.size(), fp_) != rec.size()) {
        spdlog::error("epm_dump: block record write failed — dumping "
                      "disabled");
        std::fclose(fp_);
        fp_ = nullptr;
        return;
    }
    std::fflush(fp_);
}

// ── EpmRoutingDumper ─────────────────────────────────────────────────────────

EpmRoutingDumper::EpmRoutingDumper(const std::string& dir, int n_experts,
                                   int topk)
    : n_experts_(n_experts), topk_(topk) {
    fp_ = open_append(dir, "epm_routing.bin");
    if (fp_)
        spdlog::info("epm_dump: routing dumper armed -> {}/epm_routing.bin "
                     "({} experts, top-{})", dir, n_experts, topk);
}

EpmRoutingDumper::~EpmRoutingDumper() {
    flush_pending();
    if (fp_) std::fclose(fp_);
}

void EpmRoutingDumper::flush_pending() {
    if (!pending_ || !fp_) {
        pending_ = false;
        return;
    }
    std::string hdr;
    hdr.reserve(28);
    put(hdr, kEpmrMagic);
    put(hdr, kEpmVersion);
    put(hdr, cur_seq_);
    put(hdr, cur_pos_);
    put(hdr, n_rows_);
    put(hdr, static_cast<uint32_t>(n_experts_));
    put(hdr, static_cast<uint32_t>(topk_));
    if (std::fwrite(hdr.data(), 1, hdr.size(), fp_) != hdr.size() ||
        std::fwrite(body_.data(), 1, body_.size(), fp_) != body_.size()) {
        spdlog::error("epm_dump: routing record write failed — dumping "
                      "disabled");
        std::fclose(fp_);
        fp_ = nullptr;
    } else {
        std::fflush(fp_);
    }
    pending_ = false;
    n_rows_ = 0;
    body_.clear();
}

void EpmRoutingDumper::add_row(uint64_t seq_id, uint32_t token_pos,
                               uint32_t layer_idx, const float* logits_f32,
                               const int32_t* topk_ids, const float* topk_w,
                               bool last_layer) {
    if (!fp_ || !logits_f32 || !topk_ids || !topk_w) return;
    if (pending_ && (cur_seq_ != seq_id || cur_pos_ != token_pos))
        flush_pending();
    if (!pending_) {
        pending_ = true;
        cur_seq_ = seq_id;
        cur_pos_ = token_pos;
        n_rows_ = 0;
        body_.clear();
    }
    put(body_, layer_idx);
    for (int e = 0; e < n_experts_; ++e)
        put(body_, epm_f32_to_f16(logits_f32[e]));
    body_.append(reinterpret_cast<const char*>(topk_ids),
                 static_cast<size_t>(topk_) * sizeof(int32_t));
    body_.append(reinterpret_cast<const char*>(topk_w),
                 static_cast<size_t>(topk_) * sizeof(float));
    ++n_rows_;
    if (last_layer) flush_pending();
}

}  // namespace layerstorm::speculation

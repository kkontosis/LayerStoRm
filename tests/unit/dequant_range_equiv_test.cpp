// Round 2b: bit-equality proof for the streamed dequant path.
// concat(dequant_kquant_range_to_bf16 over chunks) MUST equal the whole-tensor
// dequant for every supported quant type — this is what makes the streamed
// embed/lm_head upload byte-identical to the old eager path (block-local
// kernels + block-aligned ranges). Host-only, no CUDA.

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <vector>

#include "model/weight_loader/weight_handler.h"
#include "model/weight_loader/weight_loader.h"

namespace lm = layerstorm::model;

namespace {

struct QuantSpec {
    lm::GgufKQuantType type;
    int block_elems;
    int block_bytes;
};

lm::RawTensor make_tensor(const QuantSpec& q, int64_t elems,
                          std::vector<std::byte>& storage) {
    const int64_t nb = elems / q.block_elems;
    storage.resize(static_cast<size_t>(nb * q.block_bytes));
    std::mt19937 rng(1234);
    for (auto& b : storage)
        b = static_cast<std::byte>(rng() & 0xFF);
    // Keep the fp16 scale fields finite-ish: zero the exponent-heavy byte of
    // each block's leading scale so dequant math stays in normal float range
    // (bit-equality holds regardless, but avoid inf-noise in the test).
    for (int64_t b = 0; b < nb; ++b)
        storage[static_cast<size_t>(b * q.block_bytes + 1)] &= std::byte{0x3C};
    lm::RawTensor t;
    t.shape = {elems};
    t.dtype = lm::SafetensorsDtype::F32;  // ignored for k-quant
    t.gguf_type = q.type;
    t.data = std::span<const std::byte>(storage.data(), storage.size());
    return t;
}

}  // namespace

TEST(DequantRangeEquiv, ChunkedEqualsWholeForAllKQuants) {
    const QuantSpec specs[] = {
        {lm::GgufKQuantType::Q8_0, 32, 34},
        {lm::GgufKQuantType::Q4_K, 256, 144},
        {lm::GgufKQuantType::Q5_K, 256, 176},
        {lm::GgufKQuantType::Q6_K, 256, 210},
    };
    constexpr int64_t kElems = 256 * 4 * 37;  // several blocks, odd multiple
    for (const auto& q : specs) {
        std::vector<std::byte> storage;
        const lm::RawTensor t = make_tensor(q, kElems, storage);

        // Whole-tensor reference (one range covering everything).
        const auto whole = lm::dequant_kquant_range_to_bf16(t, 0, kElems);
        ASSERT_EQ(whole.size(), static_cast<size_t>(kElems));

        // Chunked at a block-aligned, non-divisor stride + a sharded offset.
        const int64_t chunk_sizes[] = {q.block_elems * 3ll,
                                       q.block_elems * 7ll, kElems / 2};
        for (int64_t chunk : chunk_sizes) {
            std::vector<uint16_t> cat;
            cat.reserve(static_cast<size_t>(kElems));
            for (int64_t off = 0; off < kElems; off += chunk) {
                const int64_t n = std::min(chunk, kElems - off);
                const auto part = lm::dequant_kquant_range_to_bf16(t, off, n);
                cat.insert(cat.end(), part.begin(), part.end());
            }
            ASSERT_EQ(cat.size(), whole.size());
            EXPECT_EQ(0, std::memcmp(cat.data(), whole.data(),
                                     cat.size() * sizeof(uint16_t)))
                << "type=" << static_cast<int>(q.type) << " chunk=" << chunk;
        }

        // Rank-sharded halves (the upload's actual access pattern).
        const int64_t half = kElems / 2;
        const auto lo = lm::dequant_kquant_range_to_bf16(t, 0, half);
        const auto hi = lm::dequant_kquant_range_to_bf16(t, half, half);
        EXPECT_EQ(0, std::memcmp(lo.data(), whole.data(),
                                 static_cast<size_t>(half) * 2));
        EXPECT_EQ(0, std::memcmp(hi.data(), whole.data() + half,
                                 static_cast<size_t>(half) * 2));
    }
}

TEST(DequantRangeEquiv, MisalignedRangeThrows) {
    std::vector<std::byte> storage;
    const lm::RawTensor t = make_tensor({lm::GgufKQuantType::Q4_K, 256, 144},
                                        256 * 8, storage);
    EXPECT_THROW(lm::dequant_kquant_range_to_bf16(t, 128, 256),
                 std::runtime_error);
    EXPECT_THROW(lm::dequant_kquant_range_to_bf16(t, 0, 100),
                 std::runtime_error);
}

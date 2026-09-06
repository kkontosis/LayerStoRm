// GF3.4 — NoPE sparse MLA (ModelType::GLM5N, GLM-5.3-Flash): the D_QK 512
// geometry twins of the 576 attention suites.
//
//   GLM5N  D_QK = 512 = 512 FP8 NOPE + 0 ROPE, d_v = 512
//          SnapMLA cache row = [512 FP8 | 4 B f32 scale] = 516 B (the V32 644 B
//          row minus its 128 B BF16 rope tail); TQ row 258 B.
//
// The bar is the one the 576 tests already set, leg for leg:
//   - prep       (mla_attention_test.cu): k_append -> dequant round-trip and
//                fused_q_quant, here sharpened to EXACT — the stored FP8 byte
//                must be a nearest-representable quantization of v/scale, the
//                scale exactly amax/448, and the dequantized BF16 bit-identical
//                to bf16_rne(fp8(stored) * stored_scale). Plus the two NoPE-only
//                contracts: the cache write footprint is exactly 516 B (no rope
//                tail spills into the next row) and fused_q_quant at
//                d_qk == d_nope == 512 never touches its rope output.
//   - decode     (decode_lse_units_test.cu): sparse + dense FP8 decode, no-split
//                and multi-split + mla_combine, CPU reference computed from the
//                EXACT stored values (read-back FP8 cache bytes + quantized Q),
//                LSE in NATURAL log units, planted high-logit token as the
//                split-weighting discriminator. Q carries NO rope half:
//                params.q_rope is nullptr and the reference has no rope term.
//   - prefill    (prefill_{sparse,dense}_causal_test.cu): batched chunk-causal
//                call vs a CPU reference over exactly the selected+bounded
//                indices (bf16 tolerance) + the DET-REDUCE bit-identity legs
//                (deterministic_reduce=true -> run-to-run and per-row-oracle
//                memcmp). Staged KV rows are 512-wide BF16 latents: with
//                d_v == d_qk == 512 the V columns ARE the K columns.
//   - dispatch   (CPU only): the MODEL1-style geometry refusals — d_qk 512 with
//                d_v 448 must throw for both prefill kernels (Traits<512> would
//                read 512 V columns out of a 448-wide V), and an unknown
//                (d_qk, d_nope) pair must throw for dense FP8 decode.
//   - TurboQuant (TD-GLM5-TQ-BACKEND-UNWIRED, section 7 below): the same three
//                legs for the TQ backend, whose NoPE cache row is 258 B =
//                [256 B packed 4-bit c_kv | 2 B FP16 norm] — the 386 B row every
//                other TQ suite exercises minus its 128 B BF16 rope tail.

#include "compute/kernels/attention/mla_attention.h"
#include "compute/kernels/attention/prep_params.h"
#include "compute/prefill_params.h"

// TurboQuant backend (section 7): dispatch wrappers, prep param structs and the
// host-side codebook / rotation-matrix helpers the TQ suites already use.
#include "compute/kernels/attention/tq_mla_attention.h"
#include "compute/kernels/sm120/attention/tq_prep_params.h"
#include "compute/tq_init.h"

#include "smxx/get_mla_metadata.h"
#include "smxx/mla_combine.h"
#include "sm120/decode/sparse_fp8/params.h"
#include "sm120/decode/dense_fp8/params.h"
#include "sm120/decode/tq_sparse/params.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lc = layerstorm::compute;

namespace {

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        ASSERT_EQ(_err, cudaSuccess)                                           \
            << "CUDA error: " << cudaGetErrorString(_err);                     \
    } while (0)

// ── NoPE geometry ───────────────────────────────────────────────────────────
constexpr int kDC = 512;              // kv_lora_rank = d_nope = d_v
constexpr int kDR = 0;                // qk_rope_head_dim — NoPE: no rope block
constexpr int kDQK = kDC + kDR;       // 512
constexpr int kRowBytes = kDC + 4 + kDR * 2;   // 516 = [512 fp8 | f32 scale]
constexpr int kPage = 64;

static_assert(kDQK == 512, "GLM5N absorbed QK dim");
static_assert(kRowBytes == 516, "GLM5N SnapMLA cache row (no rope tail)");

constexpr float kFp8Max = 448.0f;

// The project compiles CUDA with --use_fast_math (CMakeLists.txt), which implies
// -prec-div=false: the kernels' `amax / 448.0f` is an APPROXIMATE division,
// documented to 2 ULP, so the stored scale may sit one ULP off the IEEE-rounded
// host value. Only the scale's VALUE is held to that band — every exactness
// claim below (nearest-FP8 code, bit-exact dequant) is computed from the
// READ-BACK scale, so it is unaffected.
constexpr float kFastDivUlp = 2.0f * 1.1920929e-7f;  // 2 ULP, relative

// ── Host numeric helpers ────────────────────────────────────────────────────

// Round-to-nearest-even float -> bf16 (matches __float2bfloat16_rn for finite
// inputs); the dequant bit-equality checks depend on this being exact.
__nv_bfloat16 float_to_bf16(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    const uint32_t lsb = (f >> 16) & 1;
    f += 0x7FFF + lsb;
    const uint16_t bits = static_cast<uint16_t>(f >> 16);
    __nv_bfloat16 b;
    std::memcpy(&b, &bits, sizeof(b));
    return b;
}

float bf16_to_float(__nv_bfloat16 b) {
    uint16_t bits;
    std::memcpy(&bits, &b, sizeof(bits));
    const uint32_t f = static_cast<uint32_t>(bits) << 16;
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}

uint16_t bf16_bits(__nv_bfloat16 b) {
    uint16_t bits;
    std::memcpy(&bits, &b, sizeof(bits));
    return bits;
}

float bf16r(float f) { return __bfloat162float(__float2bfloat16(f)); }

// Host FP8 e4m3 decode (bit-exact) — same table as decode_lse_units_test.
float fp8_e4m3_to_float(uint8_t b) {
    const int sign = (b >> 7) & 1;
    const int exp = (b >> 3) & 0xF;
    const int man = b & 0x7;
    float v;
    if (exp == 0) {
        v = std::ldexp(static_cast<float>(man) / 8.0f, -6);  // subnormal
    } else if (exp == 0xF && man == 0x7) {
        v = std::nanf("");
    } else {
        v = std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
    }
    return sign ? -v : v;
}

bool fp8_e4m3_is_nan(uint8_t b) {
    return ((b >> 3) & 0xF) == 0xF && (b & 0x7) == 0x7;
}

// Is `stored` A nearest-representable e4m3 encoding of `target`? Brute force
// over the 254 finite codes — this asserts the quantizer's VALUE choice without
// re-deriving its tie-breaking rule (either of two equidistant codes passes).
::testing::AssertionResult IsNearestFp8(float target, uint8_t stored) {
    if (fp8_e4m3_is_nan(stored))
        return ::testing::AssertionFailure() << "stored code is NaN";
    const float d = std::fabs(fp8_e4m3_to_float(stored) - target);
    for (int c = 0; c < 256; ++c) {
        const uint8_t b = static_cast<uint8_t>(c);
        if (fp8_e4m3_is_nan(b)) continue;
        const float dc = std::fabs(fp8_e4m3_to_float(b) - target);
        if (dc < d)
            return ::testing::AssertionFailure()
                << "target " << target << " stored code " << static_cast<int>(stored)
                << " (" << fp8_e4m3_to_float(stored) << ", err " << d
                << ") but code " << c << " (" << fp8_e4m3_to_float(b)
                << ") is closer (err " << dc << ")";
    }
    return ::testing::AssertionSuccess();
}

}  // namespace

//==============================================================================
// 1. Prep round-trip at the NoPE geometry
//==============================================================================

// launch_fused_k_append with d_rope = 0 writing 516 B rows, then
// launch_dequant_ckv_indexed back to BF16. Asserts, per token:
//   (a) scale is EXACTLY amax(c_kv)/448 (the amax reduction is order-free);
//   (b) every stored FP8 byte is a nearest-representable code for
//       clamp(c_kv * (1/scale)) — the kernel's own inv_scale arithmetic;
//   (c) the dequantized BF16 is BIT-IDENTICAL to bf16_rne(fp8(byte) * scale);
//   (d) NO ROPE TAIL: tokens are appended to EVEN slots only, so with a 516 B
//       row stride a stray rope write would land in the following row — every
//       odd row must still hold its 0xCC poison, byte for byte.
TEST(NopeMlaPrep, KAppendDequantRoundTripExactAtNoPe) {
    REQUIRES_GPU();

    const int num_tokens = 8;
    const int num_pages = 1;

    std::mt19937 rng(7001);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> ckv_f(static_cast<size_t>(num_tokens) * kDC);
    std::vector<__nv_bfloat16> ckv_b(ckv_f.size());
    for (size_t i = 0; i < ckv_f.size(); ++i) {
        ckv_b[i] = float_to_bf16(dist(rng));
        ckv_f[i] = bf16_to_float(ckv_b[i]);  // the value the kernel actually sees
    }

    // EVEN slots only: rows 1,3,5,... stay poisoned unless something writes past
    // byte 516 of the preceding row.
    std::vector<int> slots(num_tokens);
    for (int i = 0; i < num_tokens; ++i) slots[i] = 2 * i;

    __nv_bfloat16* d_ckv = nullptr;
    __nv_fp8_e4m3* d_cache = nullptr;
    int* d_slots = nullptr;
    int* d_indices = nullptr;
    __nv_bfloat16* d_kout = nullptr;
    const int64_t cache_bytes =
        static_cast<int64_t>(num_pages) * kPage * kRowBytes;

    CUDA_CHECK(cudaMalloc(&d_ckv, ckv_b.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_cache, cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_slots, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_indices, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_kout,
                          static_cast<size_t>(num_tokens) * kDQK * 2));
    CUDA_CHECK(cudaMemset(d_cache, 0xCC, cache_bytes));
    CUDA_CHECK(cudaMemcpy(d_ckv, ckv_b.data(), ckv_b.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slots.data(), num_tokens * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, slots.data(), num_tokens * sizeof(int),
                          cudaMemcpyHostToDevice));

    sm120::prep::FusedKAppendParams kp{};
    kp.c_kv = d_ckv;
    kp.k_rope = nullptr;   // NoPE: d_rope == 0, the rope loop never executes
    kp.kv_cache = d_cache;
    kp.cache_stride_block = static_cast<int64_t>(kPage) * kRowBytes;
    kp.cache_stride_row = kRowBytes;
    kp.slot_mapping = d_slots;
    kp.num_tokens = num_tokens;
    kp.d_c = kDC;
    kp.d_rope = kDR;
    kp.page_size = kPage;
    lc::launch_fused_k_append(kp, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    sm120::prep::DequantCKVIndexedParams dq{};
    dq.kv_cache = d_cache;
    dq.cache_stride_block = static_cast<int64_t>(kPage) * kRowBytes;
    dq.cache_stride_row = kRowBytes;
    dq.page_size = kPage;
    dq.indices = d_indices;
    dq.num_fetch = num_tokens;
    dq.k_out = d_kout;
    dq.d_c = kDC;
    dq.d_rope = kDR;
    lc::launch_dequant_ckv_indexed(dq, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> cache_h(cache_bytes);
    std::vector<__nv_bfloat16> kout_h(static_cast<size_t>(num_tokens) * kDQK);
    CUDA_CHECK(cudaMemcpy(cache_h.data(), d_cache, cache_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kout_h.data(), d_kout, kout_h.size() * 2,
                          cudaMemcpyDeviceToHost));

    for (int t = 0; t < num_tokens; ++t) {
        const uint8_t* row = cache_h.data() +
            static_cast<size_t>(slots[t]) * kRowBytes;
        const float* src = ckv_f.data() + static_cast<size_t>(t) * kDC;

        // (a) scale == amax / 448 exactly.
        float amax = 0.0f;
        for (int d = 0; d < kDC; ++d) amax = std::fmax(amax, std::fabs(src[d]));
        const float want_scale = amax / kFp8Max;
        float scale = 0.0f;
        std::memcpy(&scale, row + kDC, 4);
        ASSERT_NEAR(scale, want_scale, want_scale * kFastDivUlp)
            << "token " << t << " scale";
        ASSERT_GT(scale, 0.0f) << "token " << t << " degenerate scale";
        const float inv_scale = 1.0f / scale;

        for (int d = 0; d < kDC; ++d) {
            // (b) nearest-representable FP8 of the kernel's own target value.
            const float target =
                std::fmax(-kFp8Max, std::fmin(kFp8Max, src[d] * inv_scale));
            EXPECT_TRUE(IsNearestFp8(target, row[d]))
                << "token " << t << " dim " << d;
            // (c) dequant is bit-exact vs the stored byte + stored scale.
            const __nv_bfloat16 want =
                float_to_bf16(fp8_e4m3_to_float(row[d]) * scale);
            ASSERT_EQ(bf16_bits(kout_h[static_cast<size_t>(t) * kDQK + d]),
                      bf16_bits(want))
                << "token " << t << " dim " << d
                << ": dequant not bit-equal to bf16_rne(fp8(stored) * scale)";
        }
    }

    // (d) 516 B footprint: odd rows untouched (no BF16 rope tail was written).
    for (int slot = 1; slot < 2 * num_tokens; slot += 2) {
        const uint8_t* row = cache_h.data() +
            static_cast<size_t>(slot) * kRowBytes;
        for (int b = 0; b < kRowBytes; ++b)
            ASSERT_EQ(row[b], 0xCC)
                << "row " << slot << " byte " << b << " was written — the NoPE "
                << "cache row must be exactly " << kRowBytes << " B with no "
                << "rope tail";
    }

    cudaFree(d_ckv);
    cudaFree(d_cache);
    cudaFree(d_slots);
    cudaFree(d_indices);
    cudaFree(d_kout);
}

// fused_q_quant at d_qk == d_nope == 512 (d_rope = 0): the FP8 nope + per-head
// scales obey the same exactness discipline as the K side, and the rope output
// must be TOUCHED NOT AT ALL — a poisoned canary buffer is passed as
// q_rope_bf16 and must come back byte-identical.
TEST(NopeMlaPrep, FusedQQuantNoPeLeavesRopeOutputUntouched) {
    REQUIRES_GPU();

    const int s_q = 4, h_q = 8;
    const int rows = s_q * h_q;

    std::mt19937 rng(7002);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q_f(static_cast<size_t>(rows) * kDQK);
    std::vector<__nv_bfloat16> q_b(q_f.size());
    for (size_t i = 0; i < q_f.size(); ++i) {
        q_b[i] = float_to_bf16(dist(rng));
        q_f[i] = bf16_to_float(q_b[i]);
    }

    // Canary sized as if a 64-wide rope half existed: 0xA5 everywhere.
    const size_t canary_bytes = static_cast<size_t>(rows) * 64 * 2;

    __nv_bfloat16* d_q = nullptr;
    __nv_fp8_e4m3* d_nope = nullptr;
    __nv_bfloat16* d_rope_canary = nullptr;
    float* d_scales = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, q_b.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_nope, static_cast<size_t>(rows) * kDC));
    CUDA_CHECK(cudaMalloc(&d_rope_canary, canary_bytes));
    CUDA_CHECK(cudaMalloc(&d_scales, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, q_b.data(), q_b.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_rope_canary, 0xA5, canary_bytes));

    sm120::prep::FusedQQuantParams qp{};
    qp.q_bf16 = d_q;
    qp.q_nope_fp8 = d_nope;
    qp.q_rope_bf16 = d_rope_canary;
    qp.q_scales = d_scales;
    qp.s_q = s_q;
    qp.h_q = h_q;
    qp.d_qk = kDQK;
    qp.d_nope = kDC;
    lc::launch_fused_q_quant(qp, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> nope_h(static_cast<size_t>(rows) * kDC);
    std::vector<float> scales_h(rows);
    std::vector<uint8_t> canary_h(canary_bytes);
    CUDA_CHECK(cudaMemcpy(nope_h.data(), d_nope, nope_h.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scales_h.data(), d_scales, rows * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(canary_h.data(), d_rope_canary, canary_bytes,
                          cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < canary_h.size(); ++i)
        ASSERT_EQ(canary_h[i], 0xA5)
            << "fused_q_quant wrote rope output byte " << i
            << " at d_rope == 0";

    int mismatch_count = 0;
    for (int r = 0; r < rows; ++r) {
        const float* src = q_f.data() + static_cast<size_t>(r) * kDQK;
        float amax = 0.0f;
        for (int d = 0; d < kDC; ++d) amax = std::fmax(amax, std::fabs(src[d]));
        const float want_scale = amax / kFp8Max;
        ASSERT_NEAR(scales_h[r], want_scale, want_scale * kFastDivUlp)
            << "head " << r << " scale";
        ASSERT_GT(scales_h[r], 0.0f) << "head " << r << " degenerate scale";
        ASSERT_FALSE(std::isnan(scales_h[r])) << "head " << r << " NaN scale";
        const float inv_scale = 1.0f / scales_h[r];
        for (int d = 0; d < kDC; ++d) {
            const uint8_t code = nope_h[static_cast<size_t>(r) * kDC + d];
            const float target =
                std::fmax(-kFp8Max, std::fmin(kFp8Max, src[d] * inv_scale));
            EXPECT_TRUE(IsNearestFp8(target, code)) << "head " << r
                                                    << " dim " << d;
            // Same round-trip statistic the 576 test asserts on its nope half.
            const float reconstructed = fp8_e4m3_to_float(code) * scales_h[r];
            if (std::fabs(src[d]) > 0.01f &&
                std::fabs(reconstructed - src[d]) / std::fabs(src[d]) > 0.15f)
                ++mismatch_count;
        }
    }
    const float mismatch_frac =
        static_cast<float>(mismatch_count) / (rows * kDC);
    EXPECT_LT(mismatch_frac, 0.05f)
        << mismatch_count << " of " << (rows * kDC)
        << " elements have >15% relative error";

    cudaFree(d_q);
    cudaFree(d_nope);
    cudaFree(d_rope_canary);
    cudaFree(d_scales);
}

//==============================================================================
// 2/3. FP8 decode (sparse + dense) at GLM5N
//==============================================================================

namespace nope_decode {

constexpr int kHQ = 64;   // heads = a full BLOCK_SIZE_M tile

// Effective per-token K/V values as the kernel sees them, parsed from the
// read-back cache bytes. NoPE: there is no rope member at all.
struct TokenRow {
    std::vector<float> nope_fp8;   // [512] raw fp8-decoded (NOT scaled)
    float k_scale;
};

struct DecodeRig {
    void* d_cache = nullptr;
    __nv_fp8_e4m3* d_q_nope = nullptr;
    float* d_q_scales = nullptr;
    int* d_indices = nullptr;
    int* d_block_table = nullptr;
    int* d_seqlens = nullptr;
    __nv_bfloat16* d_out = nullptr;
    float* d_lse = nullptr;
    float* d_lse_accum = nullptr;
    float* d_o_accum = nullptr;
    int* d_sched_meta = nullptr;
    int* d_num_splits = nullptr;

    int num_tokens = 0;
    int num_pages = 0;
    int max_sm_parts = 0;

    std::vector<TokenRow> rows;       // [N]
    std::vector<float> q_nope_dec;    // [h, 512] fp8-decoded
    std::vector<float> q_scales;      // [h]

    ~DecodeRig() {
        for (void* p : {d_cache, (void*)d_q_nope, (void*)d_q_scales,
                        (void*)d_indices, (void*)d_block_table,
                        (void*)d_seqlens, (void*)d_out, (void*)d_lse,
                        (void*)d_lse_accum, (void*)d_o_accum,
                        (void*)d_sched_meta, (void*)d_num_splits})
            if (p) cudaFree(p);
    }
};

// Build the 516 B SnapMLA FP8 cache via the production k_append kernel (rope
// pointer nullptr, d_rope 0), quantize Q via fused_q_quant at d_qk == d_nope,
// and read back every stored value for the CPU reference. plant_token >= 0
// aligns that token's key with the mean query direction (high logit deep in the
// sequence — one split's max then dominates the combine).
void build_rig(DecodeRig& rig, int num_tokens, int max_sm_parts,
               int plant_token, unsigned seed) {
    rig.num_tokens = num_tokens;
    rig.num_pages = (num_tokens + kPage - 1) / kPage;
    rig.max_sm_parts = max_sm_parts;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> qf(static_cast<size_t>(kHQ) * kDQK);
    std::vector<float> ckv(static_cast<size_t>(num_tokens) * kDC);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : ckv) v = bf16r(dist(rng));

    if (plant_token >= 0) {
        // Logit ~10 NATURAL units above the noise floor (the 576 twin's gain
        // over its 576 aligned dims; here the same gain over 512 nope dims with
        // sm_scale 1/sqrt(512) lands in the same place). A mixed-unit lse_accum
        // would under-weight that split by ~(e/2)^10.
        constexpr float kPlantGain = 110.0f;
        for (int d = 0; d < kDC; ++d) {
            float acc = 0.f;
            for (int h = 0; h < kHQ; ++h)
                acc += qf[static_cast<size_t>(h) * kDQK + d];
            ckv[static_cast<size_t>(plant_token) * kDC + d] =
                bf16r(acc / kHQ * kPlantGain);
        }
    }

    // ── K append (production prep kernel, NoPE) ──
    std::vector<__nv_bfloat16> ckv_b(ckv.size());
    for (size_t i = 0; i < ckv.size(); ++i) ckv_b[i] = __float2bfloat16(ckv[i]);
    std::vector<int> slots(num_tokens);
    for (int i = 0; i < num_tokens; ++i) slots[i] = i;

    __nv_bfloat16* d_ckv = nullptr;
    int* d_slots = nullptr;
    const int64_t cache_bytes =
        static_cast<int64_t>(rig.num_pages) * kPage * kRowBytes;
    CUDA_CHECK(cudaMalloc(&rig.d_cache, cache_bytes));
    CUDA_CHECK(cudaMemset(rig.d_cache, 0, cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_ckv, ckv_b.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_slots, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ckv, ckv_b.data(), ckv_b.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slots.data(), num_tokens * sizeof(int),
                          cudaMemcpyHostToDevice));

    sm120::prep::FusedKAppendParams kp{};
    kp.c_kv = d_ckv;
    kp.k_rope = nullptr;
    kp.kv_cache = static_cast<__nv_fp8_e4m3*>(rig.d_cache);
    kp.cache_stride_block = static_cast<int64_t>(kPage) * kRowBytes;
    kp.cache_stride_row = kRowBytes;
    kp.slot_mapping = d_slots;
    kp.num_tokens = num_tokens;
    kp.d_c = kDC;
    kp.d_rope = kDR;
    kp.page_size = kPage;
    lc::launch_fused_k_append(kp, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_ckv);
    cudaFree(d_slots);

    // ── Q quantization (production prep kernel, rope output canary) ──
    std::vector<__nv_bfloat16> qb(qf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_rope_canary = nullptr;
    const size_t canary_bytes = static_cast<size_t>(kHQ) * 64 * 2;
    CUDA_CHECK(cudaMalloc(&d_q, qb.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_rope_canary, canary_bytes));
    CUDA_CHECK(cudaMemset(d_rope_canary, 0xA5, canary_bytes));
    CUDA_CHECK(cudaMemcpy(d_q, qb.data(), qb.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&rig.d_q_nope, static_cast<size_t>(kHQ) * kDC));
    CUDA_CHECK(cudaMalloc(&rig.d_q_scales, kHQ * sizeof(float)));

    sm120::prep::FusedQQuantParams qp{};
    qp.q_bf16 = d_q;
    qp.q_nope_fp8 = rig.d_q_nope;
    qp.q_rope_bf16 = d_rope_canary;
    qp.q_scales = rig.d_q_scales;
    qp.s_q = 1;
    qp.h_q = kHQ;
    qp.d_qk = kDQK;
    qp.d_nope = kDC;
    lc::launch_fused_q_quant(qp, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> canary_h(canary_bytes);
    CUDA_CHECK(cudaMemcpy(canary_h.data(), d_rope_canary, canary_bytes,
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < canary_h.size(); ++i)
        ASSERT_EQ(canary_h[i], 0xA5) << "q_quant touched rope output byte " << i;
    cudaFree(d_q);
    cudaFree(d_rope_canary);

    // ── Read back EXACT stored values for the CPU reference ──
    std::vector<uint8_t> cache_h(cache_bytes);
    CUDA_CHECK(cudaMemcpy(cache_h.data(), rig.d_cache, cache_bytes,
                          cudaMemcpyDeviceToHost));
    rig.rows.resize(num_tokens);
    for (int t = 0; t < num_tokens; ++t) {
        const uint8_t* row = cache_h.data() +
            static_cast<int64_t>(t / kPage) * kPage * kRowBytes +
            static_cast<int64_t>(t % kPage) * kRowBytes;
        auto& tr = rig.rows[t];
        tr.nope_fp8.resize(kDC);
        for (int d = 0; d < kDC; ++d) tr.nope_fp8[d] = fp8_e4m3_to_float(row[d]);
        std::memcpy(&tr.k_scale, row + kDC, 4);
    }

    std::vector<uint8_t> qn(static_cast<size_t>(kHQ) * kDC);
    rig.q_scales.resize(kHQ);
    CUDA_CHECK(cudaMemcpy(qn.data(), rig.d_q_nope, qn.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rig.q_scales.data(), rig.d_q_scales,
                          kHQ * sizeof(float), cudaMemcpyDeviceToHost));
    rig.q_nope_dec.resize(qn.size());
    for (size_t i = 0; i < qn.size(); ++i)
        rig.q_nope_dec[i] = fp8_e4m3_to_float(qn[i]);

    // ── Common decode I/O buffers ──
    std::vector<int> indices(num_tokens);
    for (int i = 0; i < num_tokens; ++i) indices[i] = i;
    CUDA_CHECK(cudaMalloc(&rig.d_indices, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(rig.d_indices, indices.data(),
                          num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> btable(rig.num_pages);
    for (int i = 0; i < rig.num_pages; ++i) btable[i] = i;
    CUDA_CHECK(cudaMalloc(&rig.d_block_table, rig.num_pages * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(rig.d_block_table, btable.data(),
                          rig.num_pages * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&rig.d_seqlens, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(rig.d_seqlens, &num_tokens, sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&rig.d_out, static_cast<size_t>(kHQ) * kDC * 2));
    CUDA_CHECK(cudaMalloc(&rig.d_lse, kHQ * sizeof(float)));
    const int max_splits = max_sm_parts + 2;
    CUDA_CHECK(cudaMalloc(&rig.d_lse_accum,
                          static_cast<size_t>(max_splits) * kHQ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rig.d_o_accum, static_cast<size_t>(max_splits) *
                          kHQ * kDC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rig.d_sched_meta, max_sm_parts * 8 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&rig.d_num_splits, 2 * sizeof(int)));
}

// CPU reference from the exact stored values. NO ROPE TERM:
//   score_t(h) = [ sum_d qn(h,d)*kn(t,d) ] * q_scale(h) * k_scale(t) * sm_scale
//   out(h,d)   = softmax-weighted sum of v_eff(t,d) = kn(t,d)*k_scale(t)
void cpu_reference(const DecodeRig& rig, float sm_scale,
                   std::vector<float>& out, std::vector<float>& lse) {
    const int N = rig.num_tokens;
    out.assign(static_cast<size_t>(kHQ) * kDC, 0.0f);
    lse.assign(kHQ, 0.0f);
    std::vector<double> scores(N);
    for (int h = 0; h < kHQ; ++h) {
        const float* qn = rig.q_nope_dec.data() + static_cast<size_t>(h) * kDC;
        double m = -1e300;
        for (int t = 0; t < N; ++t) {
            const auto& tr = rig.rows[t];
            double dot = 0.0;
            for (int d = 0; d < kDC; ++d)
                dot += static_cast<double>(qn[d]) * tr.nope_fp8[d];
            scores[t] = dot * rig.q_scales[h] * tr.k_scale * sm_scale;
            m = std::max(m, scores[t]);
        }
        double l = 0.0;
        for (int t = 0; t < N; ++t) {
            scores[t] = std::exp(scores[t] - m);
            l += scores[t];
        }
        lse[h] = static_cast<float>(m + std::log(l));
        float* orow = out.data() + static_cast<size_t>(h) * kDC;
        for (int t = 0; t < N; ++t) {
            const auto& tr = rig.rows[t];
            const double p = scores[t] / l;
            for (int d = 0; d < kDC; ++d)
                orow[d] += static_cast<float>(p * tr.nope_fp8[d] * tr.k_scale);
        }
    }
}

void fill_metadata(DecodeRig& rig, int num_sm_parts, bool sparse_topk) {
    GetMlaMetadataParams mp{};
    mp.seqlens_k_ptr = rig.d_seqlens;
    mp.tile_scheduler_metadata_ptr = rig.d_sched_meta;
    mp.num_splits_ptr = rig.d_num_splits;
    mp.batch_size = 1;
    mp.block_size_n = 64;
    mp.fixed_overhead_num_blocks = 1;
    mp.num_sm_parts = num_sm_parts;
    mp.topk = sparse_topk ? rig.num_tokens : -1;
    lc::launch_get_mla_metadata(mp, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

int read_num_splits(DecodeRig& rig) {
    int ns[2] = {0, 0};
    cudaMemcpy(ns, rig.d_num_splits, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(ns[0], 0);
    return ns[1];
}

void fill_sparse_params(sm120::decode::sparse_fp8::SparseAttnDecodeParams& p,
                        DecodeRig& rig, float sm_scale, int num_sm_parts) {
    std::memset(&p, 0, sizeof(p));
    p.b = 1; p.s_q = 1; p.h_q = kHQ; p.h_kv = 1;
    p.d_qk = kDQK; p.d_v = kDC;
    p.sm_scale = sm_scale;
    p.sm_scale_div_log2 = sm_scale / std::log(2.0f);
    p.num_blocks = 0;
    p.page_block_size = kPage;
    p.topk = rig.num_tokens;
    p.model_type = sm120::sparse::ModelType::GLM5N;
    p.q = reinterpret_cast<cutlass::bfloat16_t*>(rig.d_q_nope);
    p.q_rope = nullptr;    // NoPE: no rope half at all
    p.q_scales = rig.d_q_scales;
    p.kv = static_cast<cutlass::bfloat16_t*>(rig.d_cache);
    p.indices = rig.d_indices;
    p.topk_length = nullptr;
    p.attn_sink = nullptr;
    p.lse = rig.d_lse;
    p.out = reinterpret_cast<cutlass::bfloat16_t*>(rig.d_out);
    p.stride_q_b = kHQ * kDC;
    p.stride_q_s_q = kHQ * kDC;
    p.stride_q_h_q = kDC;
    p.stride_kv_block = kPage * kRowBytes;
    p.stride_kv_row = kRowBytes;
    p.stride_indices_b = rig.num_tokens;
    p.stride_indices_s_q = rig.num_tokens;
    p.stride_lse_b = kHQ;
    p.stride_lse_s_q = kHQ;
    p.stride_o_b = kHQ * kDC;
    p.stride_o_s_q = kHQ * kDC;
    p.stride_o_h_q = kDC;
    p.lse_accum = rig.d_lse_accum;
    p.o_accum = rig.d_o_accum;
    p.stride_lse_accum_split = kHQ;
    p.stride_lse_accum_s_q = kHQ;
    p.stride_o_accum_split = kHQ * kDC;
    p.stride_o_accum_s_q = kHQ * kDC;
    p.stride_o_accum_h_q = kDC;
    p.tile_scheduler_metadata_ptr =
        reinterpret_cast<sm120::decode::sparse_fp8::DecodingSchedMeta*>(
            rig.d_sched_meta);
    p.num_splits_ptr = rig.d_num_splits;
    p.num_sm_parts = num_sm_parts;
    p.stream = nullptr;
}

void fill_dense_params(sm120::decode::dense_fp8::DenseAttnDecodeParams& p,
                       DecodeRig& rig, float sm_scale, int num_sm_parts) {
    std::memset(&p, 0, sizeof(p));
    p.b = 1; p.s_q = 1; p.h_q = kHQ; p.h_kv = 1;
    // (d_qk, d_nope) == (512, 512) is the GLM5N key in launch_decode_dense_fp8.
    p.d_qk = kDQK; p.d_v = kDC; p.d_nope = kDC;
    p.sm_scale = sm_scale;
    p.q_nope = reinterpret_cast<cutlass::bfloat16_t*>(rig.d_q_nope);
    p.q_rope = nullptr;    // NoPE
    p.q_scales = rig.d_q_scales;
    p.kv_cache = static_cast<cutlass::bfloat16_t*>(rig.d_cache);
    p.stride_kv_block = kPage * kRowBytes;
    p.stride_kv_row = kRowBytes;
    p.block_table = rig.d_block_table;
    p.block_table_batch_stride = rig.num_pages;
    p.page_block_size = kPage;
    p.seqlens_k = rig.d_seqlens;
    p.out = reinterpret_cast<cutlass::bfloat16_t*>(rig.d_out);
    p.lse = rig.d_lse;
    p.stride_o_b = kHQ * kDC;
    p.stride_o_s_q = kHQ * kDC;
    p.stride_o_h_q = kDC;
    p.stride_lse_b = kHQ;
    p.stride_lse_s_q = kHQ;
    p.lse_accum = rig.d_lse_accum;
    p.o_accum = rig.d_o_accum;
    p.stride_lse_accum_split = kHQ;
    p.stride_lse_accum_s_q = kHQ;
    p.stride_o_accum_split = kHQ * kDC;
    p.stride_o_accum_s_q = kHQ * kDC;
    p.stride_o_accum_h_q = kDC;
    p.tile_scheduler_metadata_ptr =
        reinterpret_cast<sm120::decode::sparse_fp8::DecodingSchedMeta*>(
            rig.d_sched_meta);
    p.num_splits_ptr = rig.d_num_splits;
    p.num_sm_parts = num_sm_parts;
    p.deterministic_reduce = false;
    p.stream = nullptr;
}

void run_combine(DecodeRig& rig, int num_sm_parts) {
    MlaCombineParams c{};
    std::memset(&c, 0, sizeof(c));
    c.b = 1;
    c.h_q = kHQ;
    c.h_k = 1;
    c.q_seq_per_hk = kHQ;
    c.d_v = kDC;
    c.o_ptr = rig.d_out;
    c.softmax_lse_ptr = rig.d_lse;
    c.o_batch_stride = kHQ * kDC;
    c.o_head_stride = kDC;
    c.o_row_stride = kDC;
    c.num_splits_ptr = rig.d_num_splits;
    c.num_sm_parts = num_sm_parts;
    c.softmax_lseaccum_ptr = rig.d_lse_accum;
    c.oaccum_ptr = rig.d_o_accum;
    lc::launch_mla_combine(c, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

void check_result(DecodeRig& rig, const std::vector<float>& ref_out,
                  const std::vector<float>& ref_lse, const char* tag) {
    std::vector<__nv_bfloat16> outb(static_cast<size_t>(kHQ) * kDC);
    std::vector<float> lse(kHQ);
    ASSERT_EQ(cudaMemcpy(outb.data(), rig.d_out, outb.size() * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(lse.data(), rig.d_lse, kHQ * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    // LSE in NATURAL units (the DCP cross-rank combine contract). Same bar as
    // the 576 twin: a x ln2 / +log(P_scale) regression is orders larger.
    for (int h = 0; h < kHQ; ++h)
        EXPECT_NEAR(lse[h], ref_lse[h], 5e-2f) << tag << " lse[" << h << "]";

    for (size_t i = 0; i < outb.size(); ++i) {
        const double got = __bfloat162float(outb[i]);
        const double want = ref_out[i];
        ASSERT_FALSE(std::isnan(got)) << tag << " NaN out[" << i << "]";
        ASSERT_NEAR(got, want, 0.10 + 0.10 * std::abs(want))
            << tag << " out[" << i << "]";
    }
}

}  // namespace nope_decode

// ── sparse_fp8 decode (GLM5N) ───────────────────────────────────────────────

TEST(NopeMlaDecode, SparseNoSplitLseNatural) {
    REQUIRES_GPU();
    nope_decode::DecodeRig rig;
    nope_decode::build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
                           /*plant_token=*/-1, /*seed=*/5101);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    std::vector<float> ref_out, ref_lse;
    nope_decode::cpu_reference(rig, sm_scale, ref_out, ref_lse);

    nope_decode::fill_metadata(rig, /*num_sm_parts=*/1, /*sparse_topk=*/true);
    ASSERT_EQ(nope_decode::read_num_splits(rig), 1)
        << "expected the no-split path";

    sm120::decode::sparse_fp8::SparseAttnDecodeParams p{};
    nope_decode::fill_sparse_params(p, rig, sm_scale, /*num_sm_parts=*/1);
    lc::launch_decode_sparse_fp8(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    nope_decode::check_result(rig, ref_out, ref_lse, "nope-sparse-nosplit");
}

TEST(NopeMlaDecode, SparseMultiSplitCombineMatchesReference) {
    REQUIRES_GPU();
    nope_decode::DecodeRig rig;
    // Planted high-logit token deep in the sequence: one split's max dominates
    // — the discriminator for a mixed-unit lse_accum (weights ~ 2^lse_accum).
    nope_decode::build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
                           /*plant_token=*/2040, /*seed=*/5102);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    std::vector<float> ref_out, ref_lse;
    nope_decode::cpu_reference(rig, sm_scale, ref_out, ref_lse);

    nope_decode::fill_metadata(rig, /*num_sm_parts=*/8, /*sparse_topk=*/true);
    ASSERT_GT(nope_decode::read_num_splits(rig), 1)
        << "test requires a genuine multi-split schedule";

    sm120::decode::sparse_fp8::SparseAttnDecodeParams p{};
    nope_decode::fill_sparse_params(p, rig, sm_scale, /*num_sm_parts=*/8);
    lc::launch_decode_sparse_fp8(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    nope_decode::run_combine(rig, /*num_sm_parts=*/8);
    nope_decode::check_result(rig, ref_out, ref_lse, "nope-sparse-split");
}

// ── dense_fp8 decode (GLM5N, keyed by (d_qk, d_nope) == (512, 512)) ─────────

TEST(NopeMlaDecode, DenseNoSplitLseNatural) {
    REQUIRES_GPU();
    nope_decode::DecodeRig rig;
    nope_decode::build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
                           /*plant_token=*/-1, /*seed=*/5103);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    std::vector<float> ref_out, ref_lse;
    nope_decode::cpu_reference(rig, sm_scale, ref_out, ref_lse);

    nope_decode::fill_metadata(rig, /*num_sm_parts=*/1, /*sparse_topk=*/false);
    ASSERT_EQ(nope_decode::read_num_splits(rig), 1)
        << "expected the no-split path";

    sm120::decode::dense_fp8::DenseAttnDecodeParams p{};
    nope_decode::fill_dense_params(p, rig, sm_scale, /*num_sm_parts=*/1);
    lc::launch_decode_dense_fp8(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    nope_decode::check_result(rig, ref_out, ref_lse, "nope-dense-nosplit");
}

TEST(NopeMlaDecode, DenseMultiSplitCombineMatchesReference) {
    REQUIRES_GPU();
    nope_decode::DecodeRig rig;
    nope_decode::build_rig(rig, /*num_tokens=*/2048, /*max_sm_parts=*/8,
                           /*plant_token=*/2040, /*seed=*/5104);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    std::vector<float> ref_out, ref_lse;
    nope_decode::cpu_reference(rig, sm_scale, ref_out, ref_lse);

    nope_decode::fill_metadata(rig, /*num_sm_parts=*/8, /*sparse_topk=*/false);
    ASSERT_GT(nope_decode::read_num_splits(rig), 1)
        << "test requires a genuine multi-split schedule";

    sm120::decode::dense_fp8::DenseAttnDecodeParams p{};
    nope_decode::fill_dense_params(p, rig, sm_scale, /*num_sm_parts=*/8);
    lc::launch_decode_dense_fp8(p);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    nope_decode::run_combine(rig, /*num_sm_parts=*/8);
    nope_decode::check_result(rig, ref_out, ref_lse, "nope-dense-split");
}

//==============================================================================
// 4/5. Absorbed BF16 prefill (<512> instantiations) at GLM5N
//==============================================================================

namespace nope_prefill {

constexpr int kHQ = 32;   // heads per rank

struct Ref {
    std::vector<float> out;  // [s_q, h_q, d_v]
    std::vector<float> lse;  // [s_q, h_q]  (+inf for empty rows)
};

// Staged NoPE KV row: 512 BF16 latent values. d_v == d_qk == 512, so the V
// columns ARE the K columns — the reference reads the SAME row for both.
const float* kv_row(const std::vector<float>& kv, int t) {
    return kv.data() + static_cast<size_t>(t) * kDQK;
}

// CPU reference: query row sq attends exactly {indices[sq][j] : j <
// topk_len[sq], 0 <= indices[sq][j] < bounds[sq]} (chunk-causal sparse).
Ref cpu_sparse_causal_reference(const std::vector<float>& q,
                                const std::vector<float>& kv,
                                const std::vector<int>& indices,
                                const std::vector<int>& topk_len,
                                const std::vector<int>& bounds,
                                int s_q, int topk, float sm_scale) {
    Ref r;
    r.out.assign(static_cast<size_t>(s_q) * kHQ * kDC, 0.0f);
    r.lse.assign(static_cast<size_t>(s_q) * kHQ, 0.0f);
    for (int sq = 0; sq < s_q; ++sq) {
        std::vector<int> sel;
        for (int j = 0; j < topk_len[sq]; ++j) {
            const int t = indices[static_cast<size_t>(sq) * topk + j];
            if (t >= 0 && t < bounds[sq]) sel.push_back(t);
        }
        for (int h = 0; h < kHQ; ++h) {
            const float* qrow =
                q.data() + (static_cast<size_t>(sq) * kHQ + h) * kDQK;
            if (sel.empty()) {
                r.lse[static_cast<size_t>(sq) * kHQ + h] = INFINITY;
                continue;  // out stays 0
            }
            std::vector<double> scores(sel.size());
            double m = -1e300;
            for (size_t i = 0; i < sel.size(); ++i) {
                const float* krow = kv_row(kv, sel[i]);
                double dot = 0.0;
                for (int d = 0; d < kDQK; ++d)
                    dot += static_cast<double>(qrow[d]) * krow[d];
                scores[i] = dot * sm_scale;
                m = std::max(m, scores[i]);
            }
            double l = 0.0;
            for (auto& s : scores) { s = std::exp(s - m); l += s; }
            float* orow =
                r.out.data() + (static_cast<size_t>(sq) * kHQ + h) * kDC;
            for (size_t i = 0; i < sel.size(); ++i) {
                const float* vrow = kv_row(kv, sel[i]);   // V == K columns
                const double p = scores[i] / l;
                for (int d = 0; d < kDC; ++d)
                    orow[d] += static_cast<float>(p * vrow[d]);
            }
            r.lse[static_cast<size_t>(sq) * kHQ + h] =
                static_cast<float>(m + std::log(l));
        }
    }
    return r;
}

// CPU reference: query row sq attends staged rows [0, bounds[sq]) (dense
// causal-chunk semantics).
Ref cpu_dense_causal_reference(const std::vector<float>& q,
                               const std::vector<float>& kv,
                               const std::vector<int>& bounds,
                               int s_q, float sm_scale) {
    Ref r;
    r.out.assign(static_cast<size_t>(s_q) * kHQ * kDC, 0.0f);
    r.lse.assign(static_cast<size_t>(s_q) * kHQ, 0.0f);
    for (int sq = 0; sq < s_q; ++sq) {
        const int len = bounds[sq];
        std::vector<double> scores(len);
        for (int h = 0; h < kHQ; ++h) {
            const float* qrow =
                q.data() + (static_cast<size_t>(sq) * kHQ + h) * kDQK;
            double m = -1e300;
            for (int t = 0; t < len; ++t) {
                const float* krow = kv_row(kv, t);
                double dot = 0.0;
                for (int d = 0; d < kDQK; ++d)
                    dot += static_cast<double>(qrow[d]) * krow[d];
                scores[t] = dot * sm_scale;
                m = std::max(m, scores[t]);
            }
            double l = 0.0;
            for (int t = 0; t < len; ++t) {
                scores[t] = std::exp(scores[t] - m);
                l += scores[t];
            }
            float* orow =
                r.out.data() + (static_cast<size_t>(sq) * kHQ + h) * kDC;
            for (int t = 0; t < len; ++t) {
                const float* vrow = kv_row(kv, t);         // V == K columns
                const double p = scores[t] / l;
                for (int d = 0; d < kDC; ++d)
                    orow[d] += static_cast<float>(p * vrow[d]);
            }
            r.lse[static_cast<size_t>(sq) * kHQ + h] =
                static_cast<float>(m + std::log(l));
        }
    }
    return r;
}

struct DeviceRun {
    std::vector<uint16_t> out;   // raw bf16 bits [s_q, h_q, d_v]
    std::vector<float> lse;      // [s_q, h_q]
    std::vector<float> maxl;     // [s_q, h_q]  (dense only)
};

enum class Mode { kBatchedCausal, kBatchedFlat, kPerRow };

lc::PrefillDims nope_dims() {
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    // d_rope = 0 => d_qk = d_c = 512, d_v = 512, sm_scale = 1/sqrt(512).
    return lc::PrefillDims{kDC, kDR, kHQ, num_sm, 0.0f};
}

DeviceRun run_sparse(const std::vector<__nv_bfloat16>& qb,
                     const std::vector<__nv_bfloat16>& kvb,
                     const std::vector<int>& indices,
                     const std::vector<int>& topk_len,
                     const std::vector<int>& bounds,
                     int s_q, int s_kv, int topk, Mode mode,
                     bool deterministic) {
    void *dq = nullptr, *dkv = nullptr, *dout = nullptr;
    float* dlse = nullptr;
    int *dind = nullptr, *dtkl = nullptr, *dbounds = nullptr;
    const size_t out_elems = static_cast<size_t>(s_q) * kHQ * kDC;
    const size_t sh = static_cast<size_t>(s_q) * kHQ;
    EXPECT_EQ(cudaMalloc(&dq, qb.size() * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dkv, kvb.size() * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dout, out_elems * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dlse, sh * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dind, indices.size() * sizeof(int)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dtkl, s_q * sizeof(int)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dbounds, s_q * sizeof(int)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dq, qb.data(), qb.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dkv, kvb.data(), kvb.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dind, indices.data(), indices.size() * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dtkl, topk_len.data(), s_q * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dbounds, bounds.data(), s_q * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);
    // Poison outputs so "kernel wrote nothing" cannot pass bit-equality.
    EXPECT_EQ(cudaMemset(dout, 0xA5, out_elems * 2), cudaSuccess);
    EXPECT_EQ(cudaMemset(dlse, 0xA5, sh * sizeof(float)), cudaSuccess);

    const lc::PrefillDims dims = nope_dims();

    if (mode == Mode::kPerRow) {
        for (int b = 0; b < s_q; ++b) {
            SparseAttnFwdParams p{};
            lc::populate_sparse_prefill_params(
                p, dims,
                static_cast<const __nv_bfloat16*>(dq)
                    + static_cast<size_t>(b) * kHQ * kDQK,
                dkv,
                static_cast<const int*>(dind) + static_cast<size_t>(b) * topk,
                static_cast<const int*>(dtkl) + b, topk,
                /*batch_size=*/1, /*seq_len_kv=*/bounds[b],
                static_cast<__nv_bfloat16*>(dout)
                    + static_cast<size_t>(b) * kHQ * kDC,
                dlse + static_cast<size_t>(b) * kHQ, nullptr);
            EXPECT_EQ(p.d_qk, kDQK);
            EXPECT_EQ(p.d_v, kDC);
            p.deterministic_reduce = deterministic;
            lc::launch_prefill_sparse(p);
        }
    } else {
        SparseAttnFwdParams p{};
        lc::populate_sparse_prefill_params(p, dims, dq, dkv, dind, dtkl, topk,
                                           s_q, s_kv, dout, dlse, nullptr);
        EXPECT_EQ(p.d_qk, kDQK);
        EXPECT_EQ(p.d_v, kDC);
        p.deterministic_reduce = deterministic;
        if (mode == Mode::kBatchedCausal) p.s_kv_per_row = dbounds;
        lc::launch_prefill_sparse(p);
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    DeviceRun r;
    r.out.resize(out_elems);
    r.lse.resize(sh);
    EXPECT_EQ(cudaMemcpy(r.out.data(), dout, out_elems * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.lse.data(), dlse, sh * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dq); cudaFree(dkv); cudaFree(dout);
    cudaFree(dlse); cudaFree(dind); cudaFree(dtkl); cudaFree(dbounds);
    return r;
}

DeviceRun run_dense(const std::vector<__nv_bfloat16>& qb,
                    const std::vector<__nv_bfloat16>& kvb,
                    const std::vector<int>& bounds,
                    int s_q, int s_kv, bool causal, bool per_row,
                    bool deterministic) {
    void *dq = nullptr, *dkv = nullptr, *dout = nullptr;
    float *dlse = nullptr, *dmaxl = nullptr;
    int* dbounds = nullptr;
    const size_t out_elems = static_cast<size_t>(s_q) * kHQ * kDC;
    const size_t sh = static_cast<size_t>(s_q) * kHQ;
    EXPECT_EQ(cudaMalloc(&dq, qb.size() * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dkv, kvb.size() * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dout, out_elems * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dlse, sh * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dmaxl, sh * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dbounds, s_q * sizeof(int)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dq, qb.data(), qb.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dkv, kvb.data(), kvb.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dbounds, bounds.data(), s_q * sizeof(int),
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemset(dout, 0xA5, out_elems * 2), cudaSuccess);
    EXPECT_EQ(cudaMemset(dlse, 0xA5, sh * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMemset(dmaxl, 0xA5, sh * sizeof(float)), cudaSuccess);

    const lc::PrefillDims dims = nope_dims();

    if (per_row) {
        for (int b = 0; b < s_q; ++b) {
            sm120::prefill::dense::head64::DenseAttnFwdParams p{};
            lc::populate_dense_prefill_params(
                p, dims,
                static_cast<const __nv_bfloat16*>(dq)
                    + static_cast<size_t>(b) * kHQ * kDQK,
                dkv, /*batch_size=*/1, /*seq_len_kv=*/bounds[b],
                static_cast<__nv_bfloat16*>(dout)
                    + static_cast<size_t>(b) * kHQ * kDC,
                dlse + static_cast<size_t>(b) * kHQ, nullptr);
            EXPECT_EQ(p.d_qk, kDQK);
            EXPECT_EQ(p.d_v, kDC);
            p.max_logits = dmaxl + static_cast<size_t>(b) * kHQ;
            p.deterministic_reduce = deterministic;
            lc::launch_prefill_dense(p);
        }
    } else {
        sm120::prefill::dense::head64::DenseAttnFwdParams p{};
        lc::populate_dense_prefill_params(p, dims, dq, dkv, s_q, s_kv,
                                          dout, dlse, nullptr);
        EXPECT_EQ(p.d_qk, kDQK);
        EXPECT_EQ(p.d_v, kDC);
        p.max_logits = dmaxl;
        p.deterministic_reduce = deterministic;
        if (causal) p.s_kv_per_row = dbounds;
        lc::launch_prefill_dense(p);
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    DeviceRun r;
    r.out.resize(out_elems);
    r.lse.resize(sh);
    r.maxl.resize(sh);
    EXPECT_EQ(cudaMemcpy(r.out.data(), dout, out_elems * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.lse.data(), dlse, sh * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.maxl.data(), dmaxl, sh * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dq); cudaFree(dkv); cudaFree(dout);
    cudaFree(dlse); cudaFree(dmaxl); cudaFree(dbounds);
    return r;
}

// Build a row's index list: `n_causal` unique ascending indices < bound plus
// `n_future` unique ascending indices in [bound, s_kv) (which the kernel must
// mask), then -1 padding.
void fill_row(std::vector<int>& indices, int row, int topk, int bound,
              int s_kv, int n_causal, int n_future, std::mt19937& rng,
              int* topk_len_out) {
    std::vector<int> pool_c(bound), pool_f(s_kv - bound);
    for (int i = 0; i < bound; ++i) pool_c[i] = i;
    for (int i = 0; i < s_kv - bound; ++i) pool_f[i] = bound + i;
    std::shuffle(pool_c.begin(), pool_c.end(), rng);
    std::shuffle(pool_f.begin(), pool_f.end(), rng);
    std::vector<int> sel;
    for (int i = 0; i < n_causal && i < static_cast<int>(pool_c.size()); ++i)
        sel.push_back(pool_c[i]);
    for (int i = 0; i < n_future && i < static_cast<int>(pool_f.size()); ++i)
        sel.push_back(pool_f[i]);
    std::sort(sel.begin(), sel.end());
    *topk_len_out = static_cast<int>(sel.size());
    for (int j = 0; j < topk; ++j)
        indices[static_cast<size_t>(row) * topk + j] =
            j < static_cast<int>(sel.size()) ? sel[j] : -1;
}

void compare_runs_bitexact(const DeviceRun& a, const DeviceRun& b,
                           const char* tag) {
    ASSERT_EQ(a.out.size(), b.out.size());
    ASSERT_EQ(a.lse.size(), b.lse.size());
    EXPECT_EQ(std::memcmp(a.out.data(), b.out.data(), a.out.size() * 2), 0)
        << tag << " out differs (bitwise)";
    EXPECT_EQ(std::memcmp(a.lse.data(), b.lse.data(),
                          a.lse.size() * sizeof(float)), 0)
        << tag << " lse differs (bitwise)";
}

// Legacy atomicAdd denominator: same mask + block sequence => only cross-warp
// jitter may differ (~1 ULP in fp32; <=1 step in bf16).
void compare_runs(const DeviceRun& a, const DeviceRun& b, const char* tag) {
    for (size_t i = 0; i < a.lse.size(); ++i) {
        if (std::isinf(a.lse[i]) || std::isinf(b.lse[i])) {
            EXPECT_EQ(a.lse[i], b.lse[i]) << tag << " lse[" << i << "]";
        } else {
            EXPECT_NEAR(a.lse[i], b.lse[i],
                        1e-5f * (1.0f + std::abs(b.lse[i])))
                << tag << " lse[" << i << "]";
        }
    }
    for (size_t i = 0; i < a.out.size(); ++i) {
        __nv_bfloat16 ba, bb;
        std::memcpy(&ba, &a.out[i], 2);
        std::memcpy(&bb, &b.out[i], 2);
        const float fa = __bfloat162float(ba), fb = __bfloat162float(bb);
        ASSERT_FALSE(std::isnan(fa) || std::isnan(fb))
            << tag << " NaN out[" << i << "]";
        ASSERT_NEAR(fa, fb, 1e-2f * (1.0f + std::abs(fb)))
            << tag << " out[" << i << "]";
    }
}

void check_vs_reference(const DeviceRun& run, const Ref& ref, int s_kv) {
    for (size_t i = 0; i < run.lse.size(); ++i) {
        if (std::isinf(ref.lse[i])) {
            EXPECT_TRUE(std::isinf(run.lse[i]) && run.lse[i] > 0.0f)
                << "empty row lse[" << i << "]=" << run.lse[i] << " (want +inf)";
        } else {
            EXPECT_NEAR(run.lse[i], ref.lse[i], 5e-2f)
                << "lse[" << i << "] s_kv=" << s_kv;
        }
    }
    for (size_t i = 0; i < run.out.size(); ++i) {
        __nv_bfloat16 b;
        std::memcpy(&b, &run.out[i], 2);
        const double got = __bfloat162float(b);
        const double want = ref.out[i];
        ASSERT_FALSE(std::isnan(got)) << "NaN out[" << i << "]";
        ASSERT_NEAR(got, want, 0.05 + 0.05 * std::abs(want))
            << "out[" << i << "] s_kv=" << s_kv;
    }
}

// deterministic == false: legacy atomicAdd denominator — cross-run comparisons
// use ~1-ULP tolerances. deterministic == true (DET-REDUCE): every cross-run
// comparison is memcmp bit-exact, PLUS a run-to-run repeat of the identical
// batched call must be byte-identical.
void sparse_case(int s_q, int s_kv, int topk,
                 const std::vector<int>& bounds,
                 const std::vector<int>& n_causal,
                 const std::vector<int>& n_future,
                 uint32_t seed, bool deterministic) {
    ASSERT_EQ(static_cast<int>(bounds.size()), s_q);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> qf(static_cast<size_t>(s_q) * kHQ * kDQK);
    std::vector<float> kvf(static_cast<size_t>(s_kv) * kDQK);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : kvf) v = bf16r(dist(rng));
    std::vector<__nv_bfloat16> qb(qf.size()), kvb(kvf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    for (size_t i = 0; i < kvf.size(); ++i) kvb[i] = __float2bfloat16(kvf[i]);

    std::vector<int> indices(static_cast<size_t>(s_q) * topk, -1);
    std::vector<int> topk_len(s_q, 0);
    for (int b = 0; b < s_q; ++b)
        fill_row(indices, b, topk, bounds[b], s_kv, n_causal[b], n_future[b],
                 rng, &topk_len[b]);

    const auto batched = run_sparse(qb, kvb, indices, topk_len, bounds,
                                    s_q, s_kv, topk, Mode::kBatchedCausal,
                                    deterministic);

    // 0. DET-REDUCE run-to-run bit-reproducibility.
    if (deterministic) {
        const auto rerun = run_sparse(qb, kvb, indices, topk_len, bounds,
                                      s_q, s_kv, topk, Mode::kBatchedCausal,
                                      /*deterministic=*/true);
        compare_runs_bitexact(batched, rerun, "run-to-run (deterministic)");
    }

    // 1. Equivalence vs the per-row batch-of-1 oracle.
    const auto oracle = run_sparse(qb, kvb, indices, topk_len, bounds,
                                   s_q, s_kv, topk, Mode::kPerRow,
                                   deterministic);
    ASSERT_EQ(batched.out.size(), oracle.out.size());
    if (deterministic)
        compare_runs_bitexact(batched, oracle, "chunk-causal vs per-row oracle");
    else
        compare_runs(batched, oracle, "chunk-causal vs per-row oracle");

    // 2. CPU sparse-causal reference (bf16 tolerances).
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    check_vs_reference(batched,
                       cpu_sparse_causal_reference(qf, kvf, indices, topk_len,
                                                   bounds, s_q, topk, sm_scale),
                       s_kv);

    // 3. nullptr s_kv_per_row == legacy flat mask at full bounds.
    {
        const std::vector<int> full(s_q, s_kv);
        const auto causal_full = run_sparse(qb, kvb, indices, topk_len, full,
                                            s_q, s_kv, topk,
                                            Mode::kBatchedCausal, deterministic);
        const auto flat = run_sparse(qb, kvb, indices, topk_len, full,
                                     s_q, s_kv, topk, Mode::kBatchedFlat,
                                     deterministic);
        if (deterministic)
            compare_runs_bitexact(causal_full, flat, "full-bound causal vs flat");
        else
            compare_runs(causal_full, flat, "full-bound causal vs flat");
    }
}

void dense_case(int s_q, int s_kv, const std::vector<int>& bounds,
                uint32_t seed, bool deterministic, bool bit_equal) {
    ASSERT_EQ(static_cast<int>(bounds.size()), s_q);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> qf(static_cast<size_t>(s_q) * kHQ * kDQK);
    std::vector<float> kvf(static_cast<size_t>(s_kv) * kDQK);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : kvf) v = bf16r(dist(rng));
    std::vector<__nv_bfloat16> qb(qf.size()), kvb(kvf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    for (size_t i = 0; i < kvf.size(); ++i) kvb[i] = __float2bfloat16(kvf[i]);

    const auto batched = run_dense(qb, kvb, bounds, s_q, s_kv,
                                   /*causal=*/true, /*per_row=*/false,
                                   deterministic);

    // 1. Bit-equality vs the per-row-loop oracle (deterministic reduce only).
    if (bit_equal) {
        const auto oracle = run_dense(qb, kvb, bounds, s_q, s_kv,
                                      /*causal=*/true, /*per_row=*/true,
                                      deterministic);
        ASSERT_EQ(batched.out.size(), oracle.out.size());
        EXPECT_EQ(std::memcmp(batched.out.data(), oracle.out.data(),
                              batched.out.size() * 2), 0)
            << "batched-causal out != per-row out (bitwise), s_kv=" << s_kv;
        EXPECT_EQ(std::memcmp(batched.lse.data(), oracle.lse.data(),
                              batched.lse.size() * sizeof(float)), 0)
            << "batched-causal lse != per-row lse (bitwise), s_kv=" << s_kv;
        EXPECT_EQ(std::memcmp(batched.maxl.data(), oracle.maxl.data(),
                              batched.maxl.size() * sizeof(float)), 0)
            << "batched-causal max_logits != per-row (bitwise), s_kv=" << s_kv;
    }

    // 2. CPU causal reference (bf16 tolerances).
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    check_vs_reference(batched,
                       cpu_dense_causal_reference(qf, kvf, bounds, s_q, sm_scale),
                       s_kv);

    // 3. nullptr s_kv_per_row == legacy flat mask at full bounds.
    if (bit_equal) {
        const std::vector<int> full(s_q, s_kv);
        const auto causal_full = run_dense(qb, kvb, full, s_q, s_kv,
                                           /*causal=*/true, /*per_row=*/false,
                                           deterministic);
        const auto flat = run_dense(qb, kvb, full, s_q, s_kv,
                                    /*causal=*/false, /*per_row=*/false,
                                    deterministic);
        EXPECT_EQ(std::memcmp(causal_full.out.data(), flat.out.data(),
                              flat.out.size() * 2), 0)
            << "full-bound causal != flat (bitwise), s_kv=" << s_kv;
        EXPECT_EQ(std::memcmp(causal_full.lse.data(), flat.lse.data(),
                              flat.lse.size() * sizeof(float)), 0)
            << "full-bound causal lse != flat (bitwise), s_kv=" << s_kv;
    }
}

}  // namespace nope_prefill

// ── sparse prefill <512> ────────────────────────────────────────────────────

TEST(NopeMlaPrefill, SparseChunkCausalMatchesCpuReference) {
    REQUIRES_GPU();
    // 4 chunk rows (bounds ascending) with index rows that deliberately include
    // FUTURE positions (>= bound) — the kernel's causal half must mask them.
    nope_prefill::sparse_case(/*s_q=*/4, /*s_kv=*/48, /*topk=*/16,
                              /*bounds=*/{1, 2, 24, 48},
                              /*n_causal=*/{1, 2, 10, 16},
                              /*n_future=*/{4, 4, 6, 0}, /*seed=*/6101,
                              /*deterministic=*/false);
}

TEST(NopeMlaPrefill, SparseBlockBoundaryDetReduceBitReproducible) {
    REQUIRES_GPU();
    // topk = 96 straddles the B_TOPK=64 gather-block edge; DET-REDUCE => every
    // cross-run comparison is memcmp bit-exact.
    nope_prefill::sparse_case(/*s_q=*/4, /*s_kv=*/300, /*topk=*/96,
                              /*bounds=*/{63, 64, 150, 300},
                              /*n_causal=*/{40, 64, 90, 96},
                              /*n_future=*/{30, 20, 6, 0}, /*seed=*/6102,
                              /*deterministic=*/true);
}

TEST(NopeMlaPrefill, SparseMidContextPruningDetReduce) {
    REQUIRES_GPU();
    // The production sparse-chunk-prefill shape: rows deep into a longer
    // context where the top-k PRUNES (topk < bound), multi-k_block rescale.
    nope_prefill::sparse_case(/*s_q=*/6, /*s_kv=*/512, /*topk=*/64,
                              /*bounds=*/{507, 508, 509, 510, 511, 512},
                              /*n_causal=*/{64, 64, 64, 64, 64, 64},
                              /*n_future=*/{0, 0, 0, 0, 0, 0}, /*seed=*/6103,
                              /*deterministic=*/true);
}

// ── dense prefill <512> ─────────────────────────────────────────────────────

TEST(NopeMlaPrefill, DenseChunkCausalBitEqualsPerRowOracle) {
    REQUIRES_GPU();
    // 4 consecutive positions of a fresh prompt, bounds {1,2,3,4}.
    nope_prefill::dense_case(/*s_q=*/4, /*s_kv=*/4, {1, 2, 3, 4},
                             /*seed=*/6201, /*deterministic=*/true,
                             /*bit_equal=*/true);
}

TEST(NopeMlaPrefill, DenseBlockBoundaryPartialBlocks) {
    REQUIRES_GPU();
    // Bounds straddling the block edge: partial-last-block mask on both sides.
    nope_prefill::dense_case(/*s_q=*/4, /*s_kv=*/65, {63, 64, 65, 1},
                             /*seed=*/6202, /*deterministic=*/true,
                             /*bit_equal=*/true);
}

TEST(NopeMlaPrefill, DenseMultiBlockNonDeterministicReducePath) {
    REQUIRES_GPU();
    // Multi-k_block online-softmax rescale on the legacy atomicAdd denominator
    // path — CPU tolerance only (that path is not bit-reproducible).
    nope_prefill::dense_case(/*s_q=*/3, /*s_kv=*/130, {7, 64, 130},
                             /*seed=*/6203, /*deterministic=*/false,
                             /*bit_equal=*/false);
}

//==============================================================================
// 6. Geometry dispatch refusals (CPU only — every arm throws before any launch)
//==============================================================================

namespace {

// Non-null placeholder pointers: the null-pointer guards in launch_prefill_dense
// run BEFORE the geometry check, and the geometry check throws before any kernel
// launch, so these are never dereferenced.
cutlass::bfloat16_t* fake_ptr() {
    return reinterpret_cast<cutlass::bfloat16_t*>(0x1000);
}

}  // namespace

// Traits<512> contracts QK over columns [0, 512) and reads V over [0, D_V=512)
// of the SAME staged rows. That is correct only when d_v == 512 (GLM5N NoPE).
// A MODEL1-style d_qk=512 / d_v=448 row would be read as a 512-wide V and
// silently corrupt the output, so both prefill dispatches must refuse it.
TEST(NopeMlaDispatch, SparsePrefillRefusesDqk512WithDv448) {
    SparseAttnFwdParams p{};
    p.s_q = 1; p.s_kv = 1; p.h_q = 1; p.h_kv = 1;
    p.d_qk = 512; p.d_v = 448; p.topk = 1;
    p.q = fake_ptr();
    p.kv = fake_ptr();
    EXPECT_THROW(lc::launch_prefill_sparse(p), std::runtime_error);
}

TEST(NopeMlaDispatch, DensePrefillRefusesDqk512WithDv448) {
    sm120::prefill::dense::head64::DenseAttnFwdParams p{};
    p.s_q = 1; p.s_kv = 1; p.h_q = 1; p.h_kv = 1;
    p.d_qk = 512; p.d_v = 448;
    p.q = fake_ptr();
    p.kv = fake_ptr();
    EXPECT_THROW(lc::launch_prefill_dense(p), std::runtime_error);
}

TEST(NopeMlaDispatch, PrefillRefusesUnknownDqk) {
    sm120::prefill::dense::head64::DenseAttnFwdParams p{};
    p.s_q = 1; p.s_kv = 1; p.h_q = 1; p.h_kv = 1;
    p.d_qk = 640; p.d_v = 512;
    p.q = fake_ptr();
    p.kv = fake_ptr();
    EXPECT_THROW(lc::launch_prefill_dense(p), std::runtime_error);

    SparseAttnFwdParams sp{};
    sp.d_qk = 640; sp.d_v = 512;
    sp.q = fake_ptr();
    sp.kv = fake_ptr();
    EXPECT_THROW(lc::launch_prefill_sparse(sp), std::runtime_error);
}

// launch_decode_dense_fp8 keys on BOTH dims — d_nope alone is ambiguous (V32
// 576/512 and GLM5N 512/512 share d_nope 512). Anything outside
// {576/512, 512/512, 512/448} must throw rather than pick an arm.
TEST(NopeMlaDispatch, DecodeDenseFp8RefusesUnknownGeometry) {
    sm120::decode::dense_fp8::DenseAttnDecodeParams p{};
    p.b = 1; p.s_q = 1; p.h_q = 1; p.h_kv = 1;
    p.d_v = 512;

    p.d_qk = 513; p.d_nope = 512;
    EXPECT_THROW(lc::launch_decode_dense_fp8(p), std::runtime_error);

    p.d_qk = 512; p.d_nope = 384;
    EXPECT_THROW(lc::launch_decode_dense_fp8(p), std::runtime_error);

    p.d_qk = 576; p.d_nope = 448;
    EXPECT_THROW(lc::launch_decode_dense_fp8(p), std::runtime_error);
}

//==============================================================================
// 7. TurboQuant (TQ) at the GLM5N NoPE geometry — TD-GLM5-TQ-BACKEND-UNWIRED
//
// Every TQ unit test in the tree runs at d_rope = 64, where the cache row is
// 386 B = [256 B packed 4-bit c_kv | 2 B FP16 norm | 128 B BF16 rope]
// (tq_mla_attention_test.cu, tq_chunk_prefill_equiv_test.cu,
// tq_prefill_detreduce_test.cu, tq_lse_units_test.cu). The GLM5N row has no
// rope tail at all: 258 B = [256 B packed | 2 B FP16 norm]. The tests below are
// the first to write and read that row.
//
// Kernel truth being pinned:
//   tq_fused_k_append.cu — the norm store and the rope copy are done by CTA
//       (token, split 0) as `rope_out = cache_row + packed_bytes + 2` followed
//       by `for (d = tid; d < d_rope; d += blockDim.x)`. At d_rope = 0 that
//       loop makes ZERO trips, so the written row must end at byte 258 and
//       nothing may spill into the row padding or into the next row.
//   splitkv_mla.cu (tq_sparse) — the rope tail is read at packed_bytes + 2 and
//       EVERY rope operation is guarded by `lane * 2 < d_rope`, so at
//       d_rope = 0 the q_rope pointer must never be dereferenced and the score
//       must be the NOPE term alone.
//
// Suite names deliberately reuse the file's existing NopeMla{Prep,Decode,
// Dispatch} suites so the GPU legs land in the serialized `gpu` ctest bucket
// (tests/unit/CMakeLists.txt GPU_FILTER) and the CPU-only refusals stay in the
// parallel bucket, exactly like their FP8 twins above.
//==============================================================================

#ifdef LAYERSTORM_SOURCE_DIR

namespace nope_tq {

constexpr int kTqPacked   = kDC / 2;                    // 256 packed nibble bytes
constexpr int kTqRowBytes = kTqPacked + 2 + kDR * 2;    // 258 — no rope tail
constexpr int kNumCentroids = 16;

static_assert(kTqRowBytes == 258, "GLM5N TQ cache row = 256 packed + 2 B norm");
static_assert(kDC / 2 + 2 + 64 * 2 == 386,
              "the d_rope = 64 TQ row every other TQ suite exercises");

std::string codebook_path() {
    return std::string(LAYERSTORM_SOURCE_DIR) +
           "/config/tq_codebooks/codebook_d512_b4.json";
}

float cosine(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

// Codebook + rotation matrix on the device, built exactly as init_tq_resources
// builds them for layer 0 (INV-TQ-4: seed 42 + layer * 7).
struct TqCtx {
    lc::TqCodebook cb;
    std::vector<float> Pi;              // [kDC * kDC] row-major
    float* d_Pi = nullptr;
    float* d_centroids = nullptr;
    float* d_boundaries = nullptr;
    ~TqCtx() {
        if (d_Pi) cudaFree(d_Pi);
        if (d_centroids) cudaFree(d_centroids);
        if (d_boundaries) cudaFree(d_boundaries);
    }
};

void build_ctx(TqCtx& c) {
    c.cb = lc::load_codebook(codebook_path());
    ASSERT_EQ(c.cb.d, kDC);
    ASSERT_EQ(c.cb.n_clusters, kNumCentroids);
    ASSERT_EQ(static_cast<int>(c.cb.interior_boundaries.size()),
              kNumCentroids - 1);

    c.Pi.resize(static_cast<size_t>(kDC) * kDC);
    lc::generate_rotation_matrix_cpu(c.Pi.data(), kDC, /*seed=*/42);
    ASSERT_LT(lc::verify_orthogonality(c.Pi.data(), kDC), 1e-4f);

    CUDA_CHECK(cudaMalloc(&c.d_Pi, c.Pi.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.d_centroids, kNumCentroids * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.d_boundaries,
                          c.cb.interior_boundaries.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(c.d_Pi, c.Pi.data(), c.Pi.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.d_centroids, c.cb.centroids.data(),
                          kNumCentroids * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.d_boundaries, c.cb.interior_boundaries.data(),
                          c.cb.interior_boundaries.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// Fill a TqFusedKAppendParams for the NoPE geometry (d_rope = 0, no rope src).
void fill_k_append(sm120::prep::TqFusedKAppendParams& p, const TqCtx& ctx,
                   const __nv_bfloat16* d_ckv, uint8_t* d_cache,
                   int64_t page_bytes, int row_stride, const int* d_slots,
                   int num_tokens) {
    p.c_kv = d_ckv;
    p.k_rope = nullptr;   // NoPE: d_rope == 0, the rope copy loop never runs
    p.kv_cache = d_cache;
    p.cache_stride_block = page_bytes;
    p.cache_stride_row = row_stride;
    p.slot_mapping = d_slots;
    p.Pi = ctx.d_Pi;
    p.centroids = ctx.d_centroids;
    p.decision_boundaries = ctx.d_boundaries;
    p.num_tokens = num_tokens;
    p.d_c = kDC;
    p.d_rope = kDR;
    p.page_size = kPage;
    p.num_centroids = ctx.cb.n_clusters;
}

}  // namespace nope_tq

// ── TqNoPe 1: k_append row footprint ────────────────────────────────────────
//
// The write footprint is measured, not assumed: the SAME append is run twice
// into caches pre-filled with DIFFERENT sentinels (0xCC / 0x33). A byte the
// kernel wrote is equal in both runs (the append is deterministic — the
// per-coordinate rotation dots are bit-identical regardless of the NSPLIT grid,
// see tq_fused_k_append.cu); a byte it did NOT write still holds 0xCC in one
// and 0x33 in the other. So `equal` <=> `written`, exactly.
//
//   arm A (padded row stride 320): proves nothing lands in bytes [258, 320) of
//          the row — i.e. no rope tail is emitted INSIDE the row stride.
//   arm B (production row stride 258, EVEN slots only): with rows packed at
//          258 B a stray tail would land in the following row, so every odd row
//          must still hold its pre-fill, byte for byte. Same construction as
//          NopeMlaPrep.KAppendDequantRoundTripExactAtNoPe leg (d) above.
TEST(NopeMlaPrep, TqNoPeKAppendRowFootprintIs258Bytes) {
    REQUIRES_GPU();

    nope_tq::TqCtx ctx;
    nope_tq::build_ctx(ctx);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    // (c) row-stride math, asserted rather than assumed.
    ASSERT_EQ(nope_tq::kTqRowBytes, kDC / 2 + 2 + kDR * 2);
    ASSERT_EQ(nope_tq::kTqRowBytes, 258);

    const int num_tokens = 8;
    std::mt19937 rng(7301);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> ckv_f(static_cast<size_t>(num_tokens) * kDC);
    std::vector<__nv_bfloat16> ckv_b(ckv_f.size());
    for (size_t i = 0; i < ckv_f.size(); ++i) {
        ckv_b[i] = float_to_bf16(dist(rng));
        ckv_f[i] = bf16_to_float(ckv_b[i]);  // the value the kernel actually sees
    }

    __nv_bfloat16* d_ckv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ckv, ckv_b.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_ckv, ckv_b.data(), ckv_b.size() * 2,
                          cudaMemcpyHostToDevice));

    auto append = [&](uint8_t sentinel, int row_stride,
                      const std::vector<int>& slots) {
        const int64_t page_bytes = static_cast<int64_t>(kPage) * row_stride;
        uint8_t* d_cache = nullptr;
        int* d_slots = nullptr;
        EXPECT_EQ(cudaMalloc(&d_cache, page_bytes), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_slots, slots.size() * sizeof(int)), cudaSuccess);
        EXPECT_EQ(cudaMemset(d_cache, sentinel, page_bytes), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_slots, slots.data(), slots.size() * sizeof(int),
                             cudaMemcpyHostToDevice), cudaSuccess);

        sm120::prep::TqFusedKAppendParams p{};
        nope_tq::fill_k_append(p, ctx, d_ckv, d_cache, page_bytes, row_stride,
                               d_slots, static_cast<int>(slots.size()));
        lc::launch_tq_k_append(p, nullptr);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<uint8_t> host(static_cast<size_t>(page_bytes));
        EXPECT_EQ(cudaMemcpy(host.data(), d_cache, page_bytes,
                             cudaMemcpyDeviceToHost), cudaSuccess);
        cudaFree(d_cache);
        cudaFree(d_slots);
        return host;
    };

    std::vector<int> tight_slots(num_tokens), even_slots(num_tokens);
    for (int i = 0; i < num_tokens; ++i) { tight_slots[i] = i; even_slots[i] = 2 * i; }

    // ── arm A: padded row stride, contiguous slots ──
    const int kPadded = nope_tq::kTqRowBytes + 62;   // 320
    const std::vector<uint8_t> pa = append(0xCC, kPadded, tight_slots);
    const std::vector<uint8_t> pb = append(0x33, kPadded, tight_slots);
    ASSERT_EQ(pa.size(), pb.size());

    for (int t = 0; t < num_tokens; ++t) {
        const size_t off = static_cast<size_t>(tight_slots[t]) * kPadded;
        // (a) all 258 bytes were written — see the two-sentinel oracle above.
        ASSERT_EQ(std::memcmp(pa.data() + off, pb.data() + off,
                              nope_tq::kTqRowBytes), 0)
            << "row " << t << ": a byte inside the 258 B TQ row kept its "
            << "pre-fill — the NoPE row was not fully written";
        // (b) nothing past byte 258 within the row stride.
        for (int by = nope_tq::kTqRowBytes; by < kPadded; ++by) {
            ASSERT_EQ(pa[off + by], 0xCC)
                << "row " << t << " byte " << by << " written past the 258 B "
                << "NoPE row (a rope tail would start exactly here)";
            ASSERT_EQ(pb[off + by], 0x33)
                << "row " << t << " byte " << by << " written past the 258 B "
                << "NoPE row (a rope tail would start exactly here)";
        }
    }
    for (int slot = num_tokens; slot < kPage; ++slot) {
        const size_t off = static_cast<size_t>(slot) * kPadded;
        for (int by = 0; by < kPadded; ++by) {
            ASSERT_EQ(pa[off + by], 0xCC) << "unaddressed row " << slot
                                          << " byte " << by << " was written";
            ASSERT_EQ(pb[off + by], 0x33) << "unaddressed row " << slot
                                          << " byte " << by << " was written";
        }
    }

    // ── arm B: production 258 B stride, EVEN slots ──
    const std::vector<uint8_t> ta = append(0xCC, nope_tq::kTqRowBytes, even_slots);
    const std::vector<uint8_t> tb = append(0x33, nope_tq::kTqRowBytes, even_slots);
    ASSERT_EQ(ta.size(), tb.size());

    for (int t = 0; t < num_tokens; ++t) {
        const size_t off =
            static_cast<size_t>(even_slots[t]) * nope_tq::kTqRowBytes;
        ASSERT_EQ(std::memcmp(ta.data() + off, tb.data() + off,
                              nope_tq::kTqRowBytes), 0)
            << "row " << even_slots[t] << ": 258 B row not fully written";
    }
    for (int slot = 1; slot < 2 * num_tokens; slot += 2) {
        const size_t off = static_cast<size_t>(slot) * nope_tq::kTqRowBytes;
        for (int by = 0; by < nope_tq::kTqRowBytes; ++by) {
            ASSERT_EQ(ta[off + by], 0xCC)
                << "row " << slot << " byte " << by << " was written — with a "
                << "258 B stride a rope tail spills into the NEXT row";
            ASSERT_EQ(tb[off + by], 0x33)
                << "row " << slot << " byte " << by << " was written — with a "
                << "258 B stride a rope tail spills into the NEXT row";
        }
    }

    // (a cont.) the 2 B at offset 256 really are the FP16 ||c_kv||, so the
    // "written" region carries the row's semantics and not just some bytes.
    // FP16 has an 11-bit significand (relative resolution 2^-11 = 4.9e-4); the
    // 1e-3 relative band covers that rounding plus the kernel's fp32 tree-order
    // sum of squares vs this host double reduction.
    for (int t = 0; t < num_tokens; ++t) {
        const size_t off =
            static_cast<size_t>(even_slots[t]) * nope_tq::kTqRowBytes;
        __half stored = __float2half(0.0f);
        std::memcpy(&stored, ta.data() + off + nope_tq::kTqPacked, 2);
        double sq = 0.0;
        for (int d = 0; d < kDC; ++d) {
            const double v = ckv_f[static_cast<size_t>(t) * kDC + d];
            sq += v * v;
        }
        const float want = static_cast<float>(std::sqrt(sq));
        ASSERT_GT(want, 0.0f);
        EXPECT_NEAR(__half2float(stored), want, 1e-3f * want)
            << "token " << t << " FP16 norm at row offset "
            << nope_tq::kTqPacked;
    }

    cudaFree(d_ckv);
}

// ── TqNoPe 2: k_append -> dequant round-trip ────────────────────────────────
//
// Tolerance: per-token NOPE cosine > 0.90 — the SAME bound
// TqMlaAttentionPrep.KAppendDequantRoundTrip (tq_mla_attention_test.cu) holds
// the identical 4-bit codec to at d_rope = 64. TQ is lossy (16 Lloyd-Max
// centroids per rotated coordinate), so this is a direction bound, not an
// equality one. The d_rope = 64 twin additionally checks its BF16 rope
// passthrough; at d_rope = 0 there is nothing to pass through, so that leg is
// replaced by an output canary: the dequant writes [num_fetch, d_c + d_rope]
// = a TIGHT 512-wide row here, and must not run off the end of it.
TEST(NopeMlaPrep, TqNoPeKAppendDequantRoundTrip) {
    REQUIRES_GPU();

    nope_tq::TqCtx ctx;
    nope_tq::build_ctx(ctx);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    const int num_tokens = 8;
    const int64_t page_bytes =
        static_cast<int64_t>(kPage) * nope_tq::kTqRowBytes;

    std::mt19937 rng(7302);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> ckv_f(static_cast<size_t>(num_tokens) * kDC);
    std::vector<__nv_bfloat16> ckv_b(ckv_f.size());
    for (size_t i = 0; i < ckv_f.size(); ++i) {
        ckv_b[i] = float_to_bf16(dist(rng));
        ckv_f[i] = bf16_to_float(ckv_b[i]);
    }
    std::vector<int> slots(num_tokens);
    for (int i = 0; i < num_tokens; ++i) slots[i] = i;

    __nv_bfloat16* d_ckv = nullptr;
    uint8_t* d_cache = nullptr;
    int* d_slots = nullptr;
    int* d_indices = nullptr;
    __nv_bfloat16* d_out = nullptr;

    // The dequant output row stride is (d_c + d_rope) == 512 here. Allocate 64
    // BF16 elements of slack past the tight region and poison everything: a
    // 386 B-shaped kernel writing a 64-wide rope tail would land in the slack.
    const size_t out_elems = static_cast<size_t>(num_tokens) * (kDC + kDR);
    const size_t slack = 64;

    CUDA_CHECK(cudaMalloc(&d_ckv, ckv_b.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_cache, page_bytes));
    CUDA_CHECK(cudaMalloc(&d_slots, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_indices, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, (out_elems + slack) * 2));
    CUDA_CHECK(cudaMemset(d_cache, 0, page_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0xA5, (out_elems + slack) * 2));
    CUDA_CHECK(cudaMemcpy(d_ckv, ckv_b.data(), ckv_b.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slots.data(), num_tokens * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, slots.data(), num_tokens * sizeof(int),
                          cudaMemcpyHostToDevice));

    sm120::prep::TqFusedKAppendParams ka{};
    nope_tq::fill_k_append(ka, ctx, d_ckv, d_cache, page_bytes,
                           nope_tq::kTqRowBytes, d_slots, num_tokens);
    lc::launch_tq_k_append(ka, nullptr);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    sm120::prep::TqDequantCKVIndexedParams dq{};
    dq.kv_cache = d_cache;
    dq.cache_stride_block = page_bytes;
    dq.cache_stride_row = nope_tq::kTqRowBytes;
    dq.page_size = kPage;
    dq.indices = d_indices;
    dq.num_fetch = num_tokens;
    dq.k_out = d_out;
    dq.Pi = ctx.d_Pi;
    dq.centroids = ctx.d_centroids;
    dq.d_c = kDC;
    dq.d_rope = kDR;
    lc::launch_tq_dequant_ckv_indexed(dq, nullptr);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> out_bits(out_elems + slack);
    CUDA_CHECK(cudaMemcpy(out_bits.data(), d_out, (out_elems + slack) * 2,
                          cudaMemcpyDeviceToHost));

    // No overrun past the tight [num_fetch, 512] output.
    for (size_t i = out_elems; i < out_elems + slack; ++i)
        ASSERT_EQ(out_bits[i], static_cast<uint16_t>(0xA5A5))
            << "tq_dequant_ckv_indexed wrote element " << i
            << " past the tight " << (kDC + kDR) << "-wide NoPE output row";

    for (int t = 0; t < num_tokens; ++t) {
        std::vector<float> orig(kDC), recon(kDC);
        for (int d = 0; d < kDC; ++d) {
            orig[d] = ckv_f[static_cast<size_t>(t) * kDC + d];
            __nv_bfloat16 b;
            std::memcpy(&b, &out_bits[static_cast<size_t>(t) * (kDC + kDR) + d], 2);
            recon[d] = bf16_to_float(b);
        }
        const float cos = nope_tq::cosine(orig.data(), recon.data(), kDC);
        // Mirrors TqMlaAttentionPrep.KAppendDequantRoundTrip's 0.90 bound for
        // the same codec at d_rope = 64.
        EXPECT_GT(cos, 0.90f)
            << "token " << t << " NoPE round-trip cosine too low: " << cos;
        // TQ stores the L2 norm exactly (FP16) and quantizes only the
        // direction, so the reconstruction must also keep the SCALE — a
        // dropped/mis-offset norm read (the 258 B row moves it nowhere, but a
        // 386 B-shaped reader would find rope bytes here) shows up as a norm
        // ratio far from 1 even while the cosine stays high.
        double no = 0.0, nr = 0.0;
        for (int d = 0; d < kDC; ++d) {
            no += static_cast<double>(orig[d]) * orig[d];
            nr += static_cast<double>(recon[d]) * recon[d];
        }
        const double ratio = std::sqrt(nr) / std::sqrt(no);
        EXPECT_NEAR(ratio, 1.0, 0.05)
            << "token " << t << " reconstruction/original norm ratio " << ratio;
    }

    cudaFree(d_ckv);
    cudaFree(d_cache);
    cudaFree(d_slots);
    cudaFree(d_indices);
    cudaFree(d_out);
}

// ── TqNoPe 3: sparse decode vs a CPU reference over the DEQUANTIZED rows ────
//
// Chain: tq_k_append (d_rope = 0) -> tq_q_rotate -> tq_sparse decode. The
// reference is built from the values the kernel actually consumes — the packed
// nibbles and FP16 norms READ BACK from the 258 B cache rows, and the FP32
// q_rot read back from the rotate — so the 4-bit codec is exact to its own
// dequant and only fp32-vs-double accumulation separates GPU from CPU. Same
// construction and the same reference formula as TqLseUnits.cpu_reference
// (tq_lse_units_test.cu), minus the rope term:
//   score_t(h) = (sum_d q_rot(h,d) * cent(code(t,d))) * norm(t) * sm_scale
//   out(h,d)   = sum_t (p_t / L) * norm(t) * cent(code(t,d))   [rotated space]
//   lse(h)     = M + log(L)                                    [NATURAL units]
TEST(NopeMlaDecode, TqNoPeSparseDecodeMatchesDequantReference) {
    REQUIRES_GPU();

    nope_tq::TqCtx ctx;
    nope_tq::build_ctx(ctx);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    const int h_q = 8;
    const int num_kv = 200;                                  // cache slots 0..199
    const int num_pages = (num_kv + kPage - 1) / kPage;      // 4
    const int topk = 256;                                    // 4 gather blocks
    const int n_valid = 176;                                 // -> block 3 is ALL -1
    const int64_t page_bytes =
        static_cast<int64_t>(kPage) * nope_tq::kTqRowBytes;
    const int64_t cache_bytes = static_cast<int64_t>(num_pages) * page_bytes;
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));

    std::mt19937 rng(7303);
    std::normal_distribution<float> kdist(0.0f, 0.5f);
    // Q is scaled up so the softmax has real dynamic range at the production
    // sm_scale = 1/sqrt(512): score std ~ |q| * |y_hat| * norm * sm_scale /
    // sqrt(d_c), which is ~O(1) natural unit here instead of ~0.05.
    std::normal_distribution<float> qdist(0.0f, 2.0f);

    std::vector<__nv_bfloat16> ckv_b(static_cast<size_t>(num_kv) * kDC);
    for (auto& v : ckv_b) v = float_to_bf16(kdist(rng));
    // One extra head row of slack so the production-form q_rope pointer
    // (q + d_c, row stride d_qk) stays inside the allocation for the LAST head
    // even though d_rope == 0 means it is never dereferenced.
    std::vector<__nv_bfloat16> q_b(static_cast<size_t>(h_q + 1) * kDQK);
    for (auto& v : q_b) v = float_to_bf16(qdist(rng));

    std::vector<int> slots(num_kv);
    for (int i = 0; i < num_kv; ++i) slots[i] = i;

    // Sparse selection: n_valid distinct ascending slots, then -1 padding.
    std::vector<int> pool(num_kv);
    for (int i = 0; i < num_kv; ++i) pool[i] = i;
    std::shuffle(pool.begin(), pool.end(), rng);
    std::vector<int> sel(pool.begin(), pool.begin() + n_valid);
    std::sort(sel.begin(), sel.end());
    std::vector<int> indices(topk, -1);
    for (int i = 0; i < n_valid; ++i) indices[i] = sel[i];

    __nv_bfloat16* d_ckv = nullptr;
    __nv_bfloat16* d_q = nullptr;
    uint8_t* d_cache = nullptr;
    int* d_slots = nullptr;
    int* d_indices = nullptr;
    float* d_q_rot = nullptr;
    float* d_out = nullptr;
    float* d_lse = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ckv, ckv_b.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_q, q_b.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_cache, cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_slots, num_kv * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_indices, topk * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_q_rot, static_cast<size_t>(h_q) * kDC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(h_q) * kDC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lse, h_q * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_cache, 0, cache_bytes));
    CUDA_CHECK(cudaMemcpy(d_ckv, ckv_b.data(), ckv_b.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, q_b.data(), q_b.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, slots.data(), num_kv * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, indices.data(), topk * sizeof(int),
                          cudaMemcpyHostToDevice));

    // ── 1. Build the 258 B TQ cache with the production prep kernel ──
    sm120::prep::TqFusedKAppendParams ka{};
    nope_tq::fill_k_append(ka, ctx, d_ckv, d_cache, page_bytes,
                           nope_tq::kTqRowBytes, d_slots, num_kv);
    lc::launch_tq_k_append(ka, nullptr);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── 2. Pre-rotate Q (strided over the interleaved [nope|rope] head rows,
    //       exactly as TqSm120AttentionDevice does; at d_rope = 0 the stride
    //       d_qk degenerates to d_c) ──
    sm120::prep::TqQRotateParams qr{};
    qr.q_nope = d_q;
    qr.Pi = ctx.d_Pi;
    qr.q_rot = d_q_rot;
    qr.batch_heads = h_q;
    qr.d_c = kDC;
    qr.q_row_stride = kDQK;
    lc::launch_tq_q_rotate(qr, nullptr);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── 3. Read back exactly what the decode kernel will consume ──
    std::vector<uint8_t> cache_h(static_cast<size_t>(cache_bytes));
    std::vector<float> q_rot_h(static_cast<size_t>(h_q) * kDC);
    CUDA_CHECK(cudaMemcpy(cache_h.data(), d_cache, cache_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(q_rot_h.data(), d_q_rot,
                          q_rot_h.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<std::vector<uint8_t>> codes(num_kv);
    std::vector<float> norms(num_kv);
    for (int t = 0; t < num_kv; ++t) {
        const uint8_t* row = cache_h.data() +
            static_cast<int64_t>(t / kPage) * page_bytes +
            static_cast<int64_t>(t % kPage) * nope_tq::kTqRowBytes;
        codes[t].assign(row, row + nope_tq::kTqPacked);
        __half hn = __float2half(0.0f);
        std::memcpy(&hn, row + nope_tq::kTqPacked, 2);
        norms[t] = __half2float(hn);
        ASSERT_GT(norms[t], 0.0f) << "token " << t << " degenerate TQ norm";
    }

    // ── 4. CPU reference over the dequantized rows (double softmax) ──
    const std::vector<float>& cent = ctx.cb.centroids;
    std::vector<float> ref_out(static_cast<size_t>(h_q) * kDC, 0.0f);
    std::vector<float> ref_lse(h_q, 0.0f);
    for (int h = 0; h < h_q; ++h) {
        const float* qn = q_rot_h.data() + static_cast<size_t>(h) * kDC;
        std::vector<double> scores(n_valid);
        double m = -1e300;
        for (int i = 0; i < n_valid; ++i) {
            const int t = sel[i];
            double nope = 0.0;
            for (int bidx = 0; bidx < nope_tq::kTqPacked; ++bidx) {
                const uint8_t byte = codes[t][bidx];
                nope += static_cast<double>(qn[bidx * 2]) * cent[byte & 0x0F] +
                        static_cast<double>(qn[bidx * 2 + 1]) *
                            cent[(byte >> 4) & 0x0F];
            }
            scores[i] = nope * norms[t] * sm_scale;
            m = std::max(m, scores[i]);
        }
        double l = 0.0;
        for (int i = 0; i < n_valid; ++i) {
            scores[i] = std::exp(scores[i] - m);
            l += scores[i];
        }
        ref_lse[h] = static_cast<float>(m + std::log(l));
        float* orow = ref_out.data() + static_cast<size_t>(h) * kDC;
        for (int i = 0; i < n_valid; ++i) {
            const int t = sel[i];
            const double w = scores[i] / l * norms[t];
            for (int bidx = 0; bidx < nope_tq::kTqPacked; ++bidx) {
                const uint8_t byte = codes[t][bidx];
                orow[bidx * 2] +=
                    static_cast<float>(w * cent[byte & 0x0F]);
                orow[bidx * 2 + 1] +=
                    static_cast<float>(w * cent[(byte >> 4) & 0x0F]);
            }
        }
    }

    // ── 5. Run the sparse decode twice: q_rope = nullptr, and the production
    //       form (q + d_c with row stride d_qk). At d_rope = 0 every rope read
    //       is masked off by `lane * 2 < d_rope`, so the two MUST agree bit for
    //       bit — the decode-side twin of the "rope output untouched" canary in
    //       NopeMlaPrep.FusedQQuantNoPeLeavesRopeOutputUntouched. ──
    auto run_decode = [&](const __nv_bfloat16* q_rope, int rope_row_stride) {
        sm120::decode::tq_sparse::TqSparseDecodeParams p{};
        std::memset(&p, 0, sizeof(p));
        p.b = 1; p.s_q = 1; p.h_q = h_q; p.h_kv = 1;
        p.d_c = kDC; p.d_rope = kDR;
        p.sm_scale = sm_scale;
        p.q_rot = d_q_rot;
        p.q_rope = q_rope;
        p.q_rope_row_stride = rope_row_stride;
        p.kv_cache = d_cache;
        p.cache_stride_block = page_bytes;
        p.cache_stride_row = nope_tq::kTqRowBytes;
        p.page_block_size = kPage;
        p.indices = d_indices;
        p.topk = topk;
        p.stride_indices_b = 0;
        p.stride_indices_s_q = 0;
        p.centroids = ctx.d_centroids;
        p.out = d_out;
        p.lse = d_lse;
        p.stride_o_b = h_q * kDC;
        p.stride_o_s_q = h_q * kDC;
        p.stride_o_h_q = kDC;
        p.stride_lse_b = h_q;
        p.stride_lse_s_q = h_q;
        p.stream = nullptr;

        EXPECT_EQ(cudaMemset(d_out, 0xA5,
                             static_cast<size_t>(h_q) * kDC * sizeof(float)),
                  cudaSuccess);
        EXPECT_EQ(cudaMemset(d_lse, 0xA5, h_q * sizeof(float)), cudaSuccess);
        lc::launch_decode_sparse_tq(p);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::pair<std::vector<float>, std::vector<float>> r;
        r.first.resize(static_cast<size_t>(h_q) * kDC);
        r.second.resize(h_q);
        EXPECT_EQ(cudaMemcpy(r.first.data(), d_out,
                             r.first.size() * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(r.second.data(), d_lse, h_q * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        return r;
    };

    const auto no_rope = run_decode(nullptr, 0);
    const auto prod_rope = run_decode(d_q + kDC, kDQK);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    EXPECT_EQ(std::memcmp(no_rope.first.data(), prod_rope.first.data(),
                          no_rope.first.size() * sizeof(float)), 0)
        << "q_rope = nullptr and the production q_rope pointer disagree at "
        << "d_rope == 0 — the rope tail is being read";
    EXPECT_EQ(std::memcmp(no_rope.second.data(), prod_rope.second.data(),
                          no_rope.second.size() * sizeof(float)), 0)
        << "q_rope = nullptr and the production q_rope pointer disagree on LSE "
        << "at d_rope == 0";

    // ── 6. Compare against the reference ──
    for (int h = 0; h < h_q; ++h) {
        // Mirrors TqLseUnits.check_result (tq_lse_units_test.cu): the caches
        // are exact to the stored codes/norms, so the LSE is fp32-exact and is
        // held to 2e-3 NATURAL units — a base-2 unit regression is a factor
        // ln2 on an LSE of ~5, three orders larger.
        EXPECT_NEAR(no_rope.second[h], ref_lse[h], 2e-3f)
            << "tq-nope-sparse lse[" << h << "]";
        // fp32 kernel output vs a double CPU reference over the SAME stored
        // values: TqMlaAttentionCombine.CombineF32RoundTrip holds that pairing
        // to cosine > 0.99999; relaxed one digit here for the 176-term online
        // softmax with block rescaling.
        const float cos = nope_tq::cosine(&no_rope.first[h * kDC],
                                          &ref_out[h * kDC], kDC);
        EXPECT_GT(cos, 0.9999f)
            << "head " << h << " rotated-space cosine too low: " << cos;
    }
    for (size_t i = 0; i < no_rope.first.size(); ++i) {
        const double got = no_rope.first[i];
        const double want = ref_out[i];
        ASSERT_FALSE(std::isnan(got)) << "NaN out[" << i << "]";
        // Mirrors TqLseUnits.check_result's `1e-2 + 5e-3 * |want|`, with the
        // ABSOLUTE floor tightened 10x to 1e-3. That floor there absorbs a bf16
        // P@V accumulate; the tq_sparse PV loop is pure FP32
        // (`acc += w * centroids[i]`, splitkv_mla.cu) and this reference is
        // exact to the stored codes, so only fp32-vs-double accumulation
        // separates them (~1e-6 absolute at these magnitudes). 1e-3 keeps
        // orders of margin while staying well below the typical |out| (~0.05),
        // so the assertion is not vacuous.
        ASSERT_NEAR(got, want, 1e-3 + 5e-3 * std::abs(want))
            << "tq-nope-sparse out[" << i << "]";
    }

    cudaFree(d_ckv);
    cudaFree(d_q);
    cudaFree(d_cache);
    cudaFree(d_slots);
    cudaFree(d_indices);
    cudaFree(d_q_rot);
    cudaFree(d_out);
    cudaFree(d_lse);
}

#endif  // LAYERSTORM_SOURCE_DIR

// ── TqNoPe 4: TQ geometry dispatch refusals (CPU only) ──────────────────────
//
// The TQ analogues of NopeMlaDispatch.{Sparse,Dense}PrefillRefusesDqk512WithDv448
// and PrefillRefusesUnknownDqk. TQ prefill is a COMPOSITE (dequant the TQ cache
// to BF16, then run the shared absorbed prefill), and the dequant step is
// skipped at num_fetch == 0 (the legal empty-DCP-shard guard, KVS-3), so the
// geometry check in launch_prefill_{dense,sparse}_tq throws before any kernel
// launch — this stays CPU-only like its FP8 twins.
//
// There is deliberately no decode analogue: launch_decode_{dense,sparse}_tq are
// straight pass-throughs to the submodule run_ functions with no geometry key
// to refuse (tq_mla_attention.cu), so inventing an expectation there would be a
// test of nothing.
TEST(NopeMlaDispatch, TqNoPePrefillRefusesDqk512WithDv448) {
    sm120::prep::TqDequantCKVIndexedParams dq{};
    dq.num_fetch = 0;   // empty shard: the dequant launch is skipped entirely

    sm120::prefill::dense::head64::DenseAttnFwdParams dp{};
    dp.s_q = 1; dp.s_kv = 1; dp.h_q = 1; dp.h_kv = 1;
    dp.d_qk = 512; dp.d_v = 448;
    dp.q = fake_ptr();
    dp.kv = fake_ptr();
    EXPECT_THROW(lc::launch_prefill_dense_tq(dq, dp, nullptr),
                 std::runtime_error);

    SparseAttnFwdParams sp{};
    sp.s_q = 1; sp.s_kv = 1; sp.h_q = 1; sp.h_kv = 1;
    sp.d_qk = 512; sp.d_v = 448; sp.topk = 1;
    sp.q = fake_ptr();
    sp.kv = fake_ptr();
    EXPECT_THROW(lc::launch_prefill_sparse_tq(dq, sp, nullptr),
                 std::runtime_error);
}

TEST(NopeMlaDispatch, TqNoPePrefillRefusesUnknownDqk) {
    sm120::prep::TqDequantCKVIndexedParams dq{};
    dq.num_fetch = 0;

    sm120::prefill::dense::head64::DenseAttnFwdParams dp{};
    dp.s_q = 1; dp.s_kv = 1; dp.h_q = 1; dp.h_kv = 1;
    dp.d_qk = 640; dp.d_v = 512;
    dp.q = fake_ptr();
    dp.kv = fake_ptr();
    EXPECT_THROW(lc::launch_prefill_dense_tq(dq, dp, nullptr),
                 std::runtime_error);

    SparseAttnFwdParams sp{};
    sp.s_q = 1; sp.s_kv = 1; sp.h_q = 1; sp.h_kv = 1;
    sp.d_qk = 640; sp.d_v = 512; sp.topk = 1;
    sp.q = fake_ptr();
    sp.kv = fake_ptr();
    EXPECT_THROW(lc::launch_prefill_sparse_tq(dq, sp, nullptr),
                 std::runtime_error);
}

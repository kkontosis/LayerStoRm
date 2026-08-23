// K-grouped weight requantization helpers (TD-DSPARK-DRAFT-QUANT).
//
// BF16 [N, K] row-major GEMM operands are requantized AT UPLOAD into one of
// two storage formats consumed by the fused dequant GEMM
// (compute/kernels/sm120/gemm/wq_gemm.h):
//
//   * FP8-E4M3 + per-128-column-group FP32 scales:
//       q      [N, K]              1 byte/elem
//       scales [N, ceil(K/128)]    FP32; w = decode_e4m3(q) * scale
//     Group scale = amax(group) / 448 (E4M3 max finite), 1.0 for all-zero
//     groups. Encoding is round-to-nearest-even on the E4M3 grid,
//     saturating to +-448.
//
//   * NVFP4 (E2M1, group 16, UE8M0 scales — the model/quantization/nvfp4.h
//     format):
//       q      [N, ceil(K/2)]      2 elems/byte, LOW nibble = even k
//                                  (rows packed independently: row n starts
//                                  at byte n*ceil(K/2))
//       scales [N, ceil(K/16)]     UE8M0; w = decode_e2m1(nibble) *
//                                  decode_ue8m0(scale)
//     Group scale exponent = ceil(log2(amax/6)) (power-of-two scale so
//     every element lands on the E2M1 grid after division), byte 127 (1.0)
//     for all-zero groups. Element encoding is round-to-nearest on the
//     E2M1 magnitude grid, ties to the even-mantissa neighbor.
//
// Scale-group boundaries are ALWAYS multiples of the group size from the
// START of each row, so a GEMM over a column window [k_off, k_off+K) of a
// wider stored row (the DSpark chunked-fc slot GEMMs) indexes scales by the
// GLOBAL column (k_off + k) — no re-quantization per window, any k_off.
//
// CPU-side by design: requant runs once at draft upload (init path); the
// hot path is the fused GPU dequant in the K-loop (no BF16 scratch).

#pragma once

#include <cstdint>

namespace layerstorm::model::kgroup {

inline constexpr int kFp8GroupSize = 128;
inline constexpr int kNvfp4GroupSize = 16;  // == nvfp4::kGroupSize

// ── Scalar encode/decode (exact contracts for tests + packers) ─────────────

/// Round-to-nearest-even encode to FP8 E4M3fn, saturating to +-448.
/// NaN -> 0x7F. +-0 preserved (sign bit kept).
uint8_t encode_fp8_e4m3(float x);

/// Nearest E2M1 code for x (4-bit: [sign:1][magnitude index:3], magnitudes
/// {0, .5, 1, 1.5, 2, 3, 4, 6}). Ties round to the even-mantissa neighbor;
/// |x| > 6 saturates to 6.
uint8_t encode_e2m1(float x);

// ── Sizing (shared by the loader layout and the LayerRegistry budget —
//    they must never drift) ────────────────────────────────────────────────

inline int64_t fp8_weight_bytes(int64_t n, int64_t k) { return n * k; }
inline int64_t fp8_scale_bytes(int64_t n, int64_t k) {
    return n * ((k + kFp8GroupSize - 1) / kFp8GroupSize) *
           static_cast<int64_t>(sizeof(float));
}
inline int64_t nvfp4_weight_bytes(int64_t n, int64_t k) {
    return n * ((k + 1) / 2);
}
inline int64_t nvfp4_scale_bytes(int64_t n, int64_t k) {
    return n * ((k + kNvfp4GroupSize - 1) / kNvfp4GroupSize);
}

// ── Row-major matrix packers (multi-threaded over rows) ────────────────────

/// BF16 [n, k] -> FP8-E4M3 q [n, k] + FP32 scales [n, ceil(k/128)].
void quantize_rows_fp8_e4m3(const uint16_t* bf16, int64_t n, int64_t k,
                            uint8_t* q, float* scales);

/// BF16 [n, k] -> NVFP4 packed q [n, ceil(k/2)] + UE8M0 scales
/// [n, ceil(k/16)]. Low nibble = even column.
void quantize_rows_nvfp4(const uint16_t* bf16, int64_t n, int64_t k,
                         uint8_t* q, uint8_t* scales);

// ── CPU dequant references (test validation; mirror the GPU loaders) ───────

/// w[n, k] float from the FP8 packing above.
void dequantize_rows_fp8_e4m3(const uint8_t* q, const float* scales,
                              int64_t n, int64_t k, float* out);

/// w[n, k] float from the NVFP4 packing above.
void dequantize_rows_nvfp4(const uint8_t* q, const uint8_t* scales,
                           int64_t n, int64_t k, float* out);

}  // namespace layerstorm::model::kgroup

#pragma once

// CPU NVFP4 grouped GEMM for the MoE expert FFN (C-6, C-7).
//
// Strategy (locked by the C-6 design): DEQUANT-WEIGHTS-ON-THE-FLY. The packed
// "nvfp4-sm1xx" on-disk weights are dequantized to FP32 per output tile as the
// GEMM consumes them; activations stay BF16; the dot product accumulates in
// FP32; the output is written BF16. We IGNORE input_scale and activation
// quantization entirely — only the weight scales (E4M3 group scale *
// weight_scale_2) participate. This intentionally does NOT match the GPU
// CUTLASS NVFP4 path bit-for-bit (which also quantizes activations); validate
// instead against tools/golden_ref/nvfp4.py and the CPU reference helpers in
// tests/unit/nvfp4_gemm_test.cpp (dequant-only weight path).
//
// On-disk packed per-PROJECTION block layout ("nvfp4-sm1xx", one projection of
// P = N*K weight elements; three projections — gate, up, down — are concatenated
// into one 4096-strided expert slot in the prepacked file):
//   [ FP4 weights : (P+1)/2 bytes, 2 vals/byte
//                   (low nibble = even k, high nibble = odd k) ]
//   [ E4M3 group scales : (P+15)/16 bytes, group_size=16, stored Sm1xx-
//                         INTERLEAVED (must de-interleave; see scale_byte_index) ]
//   [ padding ]
//   [ f32 weight_scale_2 at proj_end - 8 ]
//   [ f32 input_scale    at proj_end - 4 ]   (IGNORED on this CPU path)
//
// Tensor geometry (row-major [N, K]):
//   gate / up : weight [N = I = 2048, K = H = 7168], scales [I, H/16]
//   down      : weight [N = H = 7168, K = I = 2048], scales [H, I/16]
//
// Dequant of one weight element (logical row n in [0,N), col k in [0,K)):
//   nibble = FP4 byte (k/2) ; low nibble if k even, high if k odd
//   w = E2M1_TABLE[nibble]
//       * fp8_e4m3_decode( group_scale_byte(n, k/16) )   // 0x38 == 1.0
//       * weight_scale_2
// Use src/model/quantization/fp8.{h,cpp} fp8_e4m3::decode for E4M3 (NOT
// nvfp4::dequantize, which is the UE8M0/no-global-scale path — wrong here).
//
// CPU-only: no CUDA / GPU SDK include. Covered by layerstorm_no_cuda_check.

#include <cstddef>
#include <cstdint>

namespace layerstorm::compute::cpu {

// ── E2M1 (FP4) dequant table ────────────────────────────────────────────────
// 16 signed values. bit3 = sign, bits2:0 = magnitude index.
// {0,0.5,1,1.5,2,3,4,6, -0,-0.5,-1,-1.5,-2,-3,-4,-6}. Defined in the .cpp.
extern const float kE2M1Table[16];

// ── Sm1xx scale de-interleave ───────────────────────────────────────────────

/// Byte offset, within a projection's Sm1xx-INTERLEAVED E4M3 group-scale region,
/// of the scale for logical (row n, group g) where g = k / 16 and G = K / 16.
/// Inverse of model/weight_loader/nvfp4_sfb_reformat.h reformat_nvfp4_sfb:
///   atom = 128 rows x 4 groups = 512 bytes
///   byte = ((n / 128) * ceil(G / 4) + (g / 4)) * 512
///        + (n % 32) * 16 + ((n % 128) / 32) * 4 + (g % 4)
/// IMPL: nvfp4 kernel agent. Constexpr-friendly; pure index math, no I/O.
int64_t scale_byte_index(int64_t n, int64_t g, int64_t groups);

/// Read the f32 weight_scale_2 for a projection whose packed block starts at
/// `proj_base` and spans `proj_bytes` (== 4096/3-style stride per projection;
/// the caller passes the exact projection span). Reads proj_base[proj_bytes-8].
/// IMPL: nvfp4 kernel agent.
float read_weight_scale_2(const uint8_t* proj_base, size_t proj_bytes);

// ── Packed-projection view ──────────────────────────────────────────────────

/// Describes one packed projection inside an expert slot. All offsets are byte
/// offsets from `base`. The kernel agent computes fp4_base / scale_base from the
/// geometry (fp4 region first, then scales) — these fields are the agreed
/// contract so the assemble agent can build them from the arena slot + proj idx.
struct PackedProjection {
    const uint8_t* base = nullptr;  ///< start of this projection's packed block
    size_t proj_bytes = 0;          ///< byte span of this projection (incl. tail scalars)
    int N = 0;                      ///< output rows (gate/up: 2048; down: 7168)
    int K = 0;                      ///< input cols  (gate/up: 7168; down: 2048)
};

/// Dequantize a single weight element w[n, k] from a packed projection.
/// Returns E2M1_TABLE[nibble] * E4M3_decode(group_scale) * weight_scale_2.
/// `weight_scale_2` is hoisted by the caller (read once via read_weight_scale_2)
/// to avoid re-reading per element. IMPL: nvfp4 kernel agent.
float dequant_weight(const PackedProjection& proj, int64_t n, int64_t k,
                     float weight_scale_2);

// ── CPU grouped GEMM (single projection across experts) ─────────────────────

/// Per-expert weight source for the grouped GEMM. One PackedProjection per
/// active expert group (scattered arena slots — NOT a contiguous base+stride).
struct CpuNvfp4ExpertWeights {
    const PackedProjection* projections = nullptr;  ///< [num_experts]
    int num_experts = 0;
};

/// CPU NVFP4 grouped GEMM for ONE projection (gate, up, or down).
///
/// Computes, per expert e in [0, num_experts):
///   for m in [expert_offsets[e], expert_offsets[e+1]):     // permuted tokens
///     for n in [0, N):
///       D[m, n] = sum_k A[m, k] * dequant_weight(W_e, n, k)   // FP32 accumulate
///
/// Layout:
///   A          : [total_tokens, K] BF16, row-major (permuted activations)
///   weights[e] : PackedProjection for expert e (this projection)
///   D          : [total_tokens, N] BF16, row-major (output)
///   expert_offsets : [num_experts + 1] cumulative permuted-token counts
///
/// N and K are the SAME for all experts in a call (== weights[e].N / .K). The
/// `pool` partitions the N output rows across threads; thread 0 (caller) holds
/// the result. When `pool` is null, runs single-threaded on the calling thread.
/// `elem_size_bytes` selects the A/D dtype (2 == BF16; the only supported value
/// for now — assert). IMPL: nvfp4 kernel agent.
///
/// Forward-declared pool type to keep this header free of the pool header's
/// thread/atomic includes for callers that only need the dequant helpers.
class NumaThreadPool;  // fwd (defined in numa_thread_pool.h)

void cpu_nvfp4_grouped_gemm(
    void* D,                            ///< [total_tokens, N] BF16 out
    const void* A,                      ///< [total_tokens, K] BF16 in
    const CpuNvfp4ExpertWeights& weights,
    const int32_t* expert_offsets,      ///< [num_experts + 1]
    int N, int K,
    int elem_size_bytes,                ///< 2 (BF16)
    NumaThreadPool* pool);              ///< nullptr => serial

// ── Bit-compatible activation quantization (LOSSLESS PARITY path) ────────────
//
// The default cpu_nvfp4_grouped_gemm keeps activations in full BF16 (MORE
// accurate than the GPU, but DIFFERENT). The GPU CUTLASS expert path instead
// quantizes activations to NVFP4 first (bf16_to_nvfp4_grouped: A_base is
// [tokens, K/2] FP4 + UE4M3 per-16-group scales), so its routed-expert output
// carries the activation-quantization loss. For the champion's DSpark LOSSLESS
// verify + the TQ golden to hold with CPU-computed experts, the CPU must consume
// the SAME lossy activations. These two entry points do that.

/// Quantize ONE activation row [K] BF16 → NVFP4 → back to FP32, bit-matching the
/// GPU bf16_to_nvfp4_grouped_detail.cuh arithmetic:
///   per 16-group g: amax = max_k|x|; is = (input_scale>0?input_scale:1);
///   scale_repr = ue4m3_decode(ue4m3_encode(amax/(6·is)));  denom = scale_repr·is;
///   a_deq[k] = E2M1_TABLE[ quantize_to_fp4(x/denom) ] · scale_repr · is.
/// (denom==0 → a_deq[k]=0.) `input_scale <= 0` means pure dynamic scaling (is=1).
/// `a_deq_out` length >= K. Reused across all N output rows (n-independent).
void nvfp4_quantize_activation_row(const uint16_t* a_bf16, int K,
                                   float input_scale, float* a_deq_out);

/// Bit-compatible NVFP4 grouped GEMM: identical to cpu_nvfp4_grouped_gemm except
/// each activation row is first NVFP4-quantized (nvfp4_quantize_activation_row)
/// so the CPU consumes the SAME quantized operands the GPU does. Weights use the
/// SAME dequant as the lossy path (already golden-ref-exact). The result matches
/// the GPU's quantized arithmetic to within FP32 accumulation-order round-off.
///   input_scales : [num_experts] per-expert TRT-LLM input_scale (null => all 1.0).
/// Same layout / threading contract as cpu_nvfp4_grouped_gemm.
void cpu_nvfp4_grouped_gemm_actquant(
    void* D,                            ///< [total_tokens, N] BF16 out
    const void* A,                      ///< [total_tokens, K] BF16 in
    const CpuNvfp4ExpertWeights& weights,
    const float* input_scales,          ///< [num_experts] or nullptr (=> 1.0)
    const int32_t* expert_offsets,      ///< [num_experts + 1]
    int N, int K,
    int elem_size_bytes,                ///< 2 (BF16)
    NumaThreadPool* pool);              ///< nullptr => serial

// ── M=1 GEMV path selection (ik-technique vs single-row) ─────────────────────
//
// The M=1 fast path of cpu_nvfp4_grouped_gemm fuses E2M1-decode -> FMA without
// materializing the FP32 weight row. Two strategies:
//   * kFusedSingleRow — one output row at a time, ONE FP32 accumulator chain
//                       (the original Change-2 code; latency-bound on the
//                       vpermps -> cvt -> FMA dependency per K-group).
//   * kFusedRowBlock  — the ik_llama mul_mat technique ported to NVFP4: process
//                       several output rows at once, each with its OWN
//                       independent accumulator, reusing the loaded activation
//                       group across rows so the decode->FMA latency is hidden
//                       across the row block (instruction-level parallelism).
// Default is chosen at process start: kFusedRowBlock unless the environment
// variable LS_CPU_GEMV_SINGLE_ROW is set (=1) — see select_m1_gemv_path().
enum class M1GemvPath { kFusedSingleRow, kFusedRowBlock };

/// Process-wide M=1 GEMV path. Read once from the environment on first use.
/// LS_CPU_GEMV_SINGLE_ROW=1 forces the original single-row path (for A/B + the
/// V2 attribution measurement); otherwise the ik-technique row-block path is
/// used. Thread-safe (function-local static init).
M1GemvPath select_m1_gemv_path();

/// Test/measurement hook: override the M=1 GEMV path for the calling process.
/// Pass kFusedSingleRow or kFusedRowBlock. Used by the attribution benchmark to
/// flip V2 (fused) variants without re-spawning. NOT for production callers.
void set_m1_gemv_path_for_testing(M1GemvPath path);

// ── Microbenchmark / attribution hooks (C-6 measurement deliverable) ─────────
//
// These expose the two halves of the fused M=1 GEMV (decode vs dot) and the
// three historical M=1 variants (V0 scalar-decode+materialize, V1 vpermps-
// decode+materialize, V2 vpermps-decode+fused) so the test can attribute the
// per-projection time. All operate on ONE projection / ONE activation row and
// return the FP32 dot (or write the decoded row), matching the kernel numerics
// bit-for-bit. IMPL: nvfp4 kernel agent (measurement only).

/// (C.1 decode-only) Decode ALL N*K weights of `proj` into `out` (FP32, length
/// >= N*K), one full output row after another, with NO dot/FMA. Uses the vpermps
/// vectorized decode (decode_group16) — the same decode the fused path runs.
void bench_decode_full_expert(const PackedProjection& proj, float* out);

/// (C.1 gemv-only) M=1 dot of a BF16 activation row `a` (length K) against a
/// PRE-DECODED FP32 weight matrix `w_pre` ([N, K] row-major, e.g. produced by
/// bench_decode_full_expert) — NO decode. Writes N FP32 results into `out`.
void bench_gemv_predecoded(const float* w_pre, const uint16_t* a, int N, int K,
                           float* out);

/// (C.2 V0) Scalar-decode + materialize: decode one output row with the SCALAR
/// E2M1 path (no vpermps) into a scratch row, then dot. Returns the FP32 dot for
/// output row n. This is the pre-optimization baseline decode.
float bench_m1_v0_scalar_materialize(const PackedProjection& proj, int64_t n,
                                     float weight_scale_2, const uint16_t* a,
                                     float* row_scratch);

/// (C.2 V1) vpermps-decode + materialize: decode one output row with the
/// vectorized decode_group16 into a scratch row, then dot. Returns the FP32 dot
/// for output row n. (Change-1 only: vpermps decode, but still materializes.)
float bench_m1_v1_vec_materialize(const PackedProjection& proj, int64_t n,
                                  float weight_scale_2, const uint16_t* a,
                                  float* row_scratch);

/// (C.2 V2) vpermps-decode + fused single-row GEMV: the current Change-2 path,
/// no materialization. Returns the FP32 dot for output row n.
float bench_m1_v2_fused(const PackedProjection& proj, int64_t n,
                        float weight_scale_2, const uint16_t* a);

}  // namespace layerstorm::compute::cpu

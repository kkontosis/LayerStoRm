#pragma once
// kpool_reference.h — CPU reference for the GLM-5.3-Flash (`glm5_next`)
// IndexPool ("kpool") indexer-K compression + pooled selection, PLAN.md GF3.5.
//
// ── What this is ─────────────────────────────────────────────────────────────
// The SPECIFICATION ARTIFACT for the kpool leg of the glm5_next indexer. The
// SM120 kernels in deps/LayerStoRmKernels/csrc/sm120/indexer/kpool_compress.cu
// are judged against the numbers this file produces
// (tests/unit/indexer_kpool_test.cu). It is deliberately naive and heavily
// documented: correctness and legibility over speed — plain fp32 loops, no
// CUDA, no intrinsics, nothing that needs a GPU to evaluate.
//
// Transcribed from scratchpad/GF35_KPOOL_REFERENCE.md, which is itself the
// authoritative transcription of
//   * vLLM tag `pr-53906-head`:
//       vllm/models/glm5next/nvidia/ops/kpool_compress.py
//         (_kpool_softmax_rotate_write_cache_kernel, expand_pools_and_append_tail)
//       vllm/model_executor/layers/sparse_attn_indexer_kpool.py
//   * SGLang tag `pr-36507-head`:
//       python/sglang/srt/models/glm5_next.py (kpool ops)
// (vLLM: Apache-2.0, Copyright contributors to the vLLM project; SGLang:
//  Apache-2.0, Copyright 2023-2024 SGLang Team — see THIRD_PARTY_NOTICES.md.)
// Section references below (§1, §2, §4, §7) are sections of
// scratchpad/GF35_KPOOL_REFERENCE.md.
//
// ── The chain, per pooled entry (D = index_head_dim = 128, P = index_kpool) ──
// A pooled entry stands for P consecutive token positions. Inputs are the P
// post-LayerNorm indexer keys k[j][:] (bf16) and the P per-dim gate vectors
// gate[j][:] (bf16), plus the learned absolute-position embedding ape[j][:]
// (fp32, indexed by the LOGICAL intra-pool slot j). Then (§1c/§1d/§1e, and the
// precision table §7):
//
//   s[j][d]  = fp32(gate[j][d]) + ape[j][d]        APE added BEFORE the max
//   m[d]     = max_j s[j][d]                       TWO-PASS, not online
//   p[j][d]  = exp(s[j][d] - m[d])                 natural exp, fp32
//   Z[d]     = sum_j p[j][d]
//   acc[d]   = sum_j p[j][d] * fp32(k[j][d])
//   x[d]     = bf16(acc[d] / Z[d])                 MANDATORY bf16 round-trip
//   y        = (1/sqrt(D)) * H_D * x               FWHT, fp32 butterflies,
//                                                  Sylvester/natural order
//   y[d]     = bf16(y[d])                          second bf16 round-trip
//   absmax   = max(max_d |y[d]|, 1e-4)             epsilon FLOOR
//   scale    = exp2(ceil(log2(absmax / 448)))      ue8m0, power of two
//   q[d]     = e4m3fn_rtne(clamp(y[d]/scale, -448, +448))
//
// The RAW per-token key is never rotated and never stored as an entry; the
// frontier's incomplete pool lives in the page tail as raw bf16 and is ALWAYS
// appended to the selection, never scored (§2).
//
// ── Deliberate GPU/CPU divergences the tests must band ───────────────────────
// The kernels are compiled with `--use_fast_math` (top-level CMakeLists.txt),
// so on the device `expf` is the fast approximation (~2^-21 relative) and
// `log2f` likewise. Consequences:
//   * a pooling weight p[j][d] can differ from this reference in the last few
//     fp32 ULP. The mandatory bf16 round-trip of acc/Z absorbs essentially all
//     of it (bf16 keeps 8 mantissa bits), and the final e4m3 quantization (3
//     mantissa bits) absorbs the rest — so the QUANTIZED BYTES agree exactly
//     for well-conditioned inputs, which is what the tests assert.
//   * `scale` is a power of two derived through ceil(log2(...)) — a STEP
//     function of absmax. `log2f` itself is exact at powers of two even under
//     fast math (verified on SM120), so the step only moves when absmax itself
//     lands ON a power-of-two multiple of 448 and one bf16 ULP of drift in the
//     absmax dimension pushes it across. That costs nothing numerically: e4m3
//     is a floating-point grid, so a one-step scale change shifts every
//     quantized exponent by one and the DEQUANTIZED entry is unchanged. Tests
//     therefore assert the dequantized entry always, and scale/bytes exactly
//     only when absmax is off that boundary.
// The 1/sqrt(D) factor uses the literal constant 0.08838834764831845f at
// D == 128 (the production geometry) exactly as the kernel does; for other D
// the kernel uses `rsqrtf`, which fast-math makes approximate — hence the
// tests use D == 128.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace kpool_ref {

// ── bit-exact scalar helpers ────────────────────────────────────────────────

/// Round-trip a float through bfloat16 with round-to-nearest-even, exactly as
/// `__float2bfloat16_rn` does: add the rounding bias (0x7FFF plus the retained
/// LSB) and truncate the low 16 mantissa bits. Inf/NaN pass through unchanged.
inline float bf16_round(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    if (((u >> 23) & 0xFFu) == 0xFFu) return x;  // inf / NaN: no rounding
    const uint32_t lsb = (u >> 16) & 1u;
    u = (u + 0x7FFFu + lsb) & 0xFFFF0000u;
    float r;
    std::memcpy(&r, &u, sizeof(r));
    return r;
}

/// Decode one float8_e4m3fn byte (4-bit exponent, bias 7; 3-bit mantissa; NO
/// infinities — 0x7F/0xFF are NaN; max finite magnitude 448).
inline float e4m3_to_f32(uint8_t b) {
    const float sign = (b & 0x80u) ? -1.0f : 1.0f;
    const int exp_field = (b >> 3) & 0x0F;
    const int mant = b & 0x07;
    if (exp_field == 0) {
        // Subnormal: value = mant * 2^-9 (step = min-normal 2^-6 / 8).
        return sign * static_cast<float>(mant) * 0.001953125f;
    }
    if (exp_field == 0x0F && mant == 0x07) {
        return sign * std::numeric_limits<float>::quiet_NaN();
    }
    return sign * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp_field - 7);
}

/// Encode a float as float8_e4m3fn, round-to-nearest-even, SATURATING (the
/// `__NV_SATFINITE` behaviour of the `__nv_fp8_e4m3(float)` constructor and of
/// `cvt.rn.satfinite.e4m3x2.f32`). Callers pre-clamp to +-448 (§1e), so the
/// saturation branch is a belt-and-braces guard, not the normal path.
inline uint8_t f32_to_e4m3_rtne(float v) {
    const uint32_t sign = std::signbit(v) ? 0x80u : 0x00u;
    const float a = std::fabs(v);
    if (!(a > 0.0f)) return static_cast<uint8_t>(sign);   // +-0 (NaN not expected)
    if (a >= 448.0f) return static_cast<uint8_t>(sign | 0x7Eu);  // saturate

    int fe = 0;
    std::frexp(a, &fe);          // a = m * 2^fe, m in [0.5, 1)
    int E = fe - 1;              // floor(log2(a))
    int step_exp;
    bool subnormal = false;
    if (E < -6) {                // below the min normal 2^-6 -> subnormal grid
        subnormal = true;
        step_exp = -9;
    } else {
        step_exp = E - 3;        // 3 mantissa bits
    }
    const float step = std::ldexp(1.0f, step_exp);
    // nearbyint honours the ambient FE_TONEAREST rounding mode = ties-to-even.
    long qi = static_cast<long>(std::nearbyint(a / step));

    if (subnormal) {
        if (qi <= 7) return static_cast<uint8_t>(sign | static_cast<uint32_t>(qi));
        // Carried up into the smallest normal (exponent field 1, mantissa 0).
        return static_cast<uint8_t>(sign | (1u << 3));
    }
    if (qi >= 16) { qi = 8; ++E; }   // mantissa carry into the next binade
    if (E > 8 || (E == 8 && qi > 14)) return static_cast<uint8_t>(sign | 0x7Eu);
    const uint32_t exp_field = static_cast<uint32_t>(E + 7);
    const uint32_t mant = static_cast<uint32_t>(qi - 8);
    return static_cast<uint8_t>(sign | (exp_field << 3) | mant);
}

/// Order-preserving float -> uint32 map used by the radix top-k
/// (lightning_topk.cu `float_to_sortable`). Reproduced here so the reference
/// selection breaks ties on EXACTLY the kernel's total order (which, unlike
/// `<` on float, distinguishes -0.0 from +0.0).
inline uint32_t sortable_key(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// ── K-side chain (§1) ───────────────────────────────────────────────────────

struct PooledEntry {
    std::vector<uint8_t> bytes;    ///< [D] float8_e4m3fn
    std::vector<float> rotated;    ///< [D] post-FWHT, post-second-bf16 values
    float scale = 0.0f;            ///< ue8m0 power-of-two per-entry scale
    float absmax = 0.0f;           ///< max_d |rotated[d]|, BEFORE the 1e-4 floor
};

/// Compose ONE pooled entry from P slot rows.
/// @param k    [P][D] post-LayerNorm keys, already bf16-representable floats
/// @param gate [P][D] raw gate vectors, already bf16-representable floats
/// @param ape  [P][D] fp32 learned intra-pool position embedding
inline PooledEntry ref_kpool_compose(const float* k, const float* gate,
                                     const float* ape, int D, int P) {
    PooledEntry out;
    out.bytes.assign(static_cast<size_t>(D), 0u);
    out.rotated.assign(static_cast<size_t>(D), 0.0f);

    // (§1c) per-dim TWO-PASS softmax over the P slots, APE added BEFORE the max.
    std::vector<float> x(static_cast<size_t>(D), 0.0f);
    std::vector<float> s(static_cast<size_t>(P), 0.0f);
    for (int d = 0; d < D; ++d) {
        float m = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < P; ++j) {
            s[static_cast<size_t>(j)] = gate[static_cast<size_t>(j) * D + d]
                                        + ape[static_cast<size_t>(j) * D + d];
            m = std::fmax(m, s[static_cast<size_t>(j)]);
        }
        float Z = 0.0f, acc = 0.0f;
        for (int j = 0; j < P; ++j) {
            const float p = std::exp(s[static_cast<size_t>(j)] - m);
            Z += p;
            acc += p * k[static_cast<size_t>(j) * D + d];
        }
        // MANDATORY bf16 round-trip between pooling and rotation.
        x[static_cast<size_t>(d)] = bf16_round(acc / Z);
    }

    // (§1d) FWHT, Sylvester/natural order, XOR butterflies, fp32. Butterfly
    // orientation matches the kernel: the LOW index of each pair gets a+b, the
    // HIGH index (the one carrying the stride bit) gets low - high.
    std::vector<float> tmp(static_cast<size_t>(D), 0.0f);
    for (int stride = 1; stride < D; stride <<= 1) {
        for (int d = 0; d < D; ++d) {
            const float a = x[static_cast<size_t>(d)];
            const float b = x[static_cast<size_t>(d ^ stride)];
            tmp[static_cast<size_t>(d)] = (d & stride) ? (b - a) : (a + b);
        }
        x.swap(tmp);
    }
    // Single 1/sqrt(D) at the end. D == 128 uses the literal constant so the
    // result is bit-identical to vLLM/SGLang (and to the kernel).
    const float inv = (D == 128)
                          ? 0.08838834764831845f
                          : static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
    for (int d = 0; d < D; ++d) {
        x[static_cast<size_t>(d)] = bf16_round(x[static_cast<size_t>(d)] * inv);
    }

    // (§1e) per-entry ue8m0 scale with the 1e-4 absmax FLOOR, +-448 clamp, RTNE.
    float amax = 0.0f;
    for (int d = 0; d < D; ++d) {
        amax = std::fmax(amax, std::fabs(x[static_cast<size_t>(d)]));
    }
    out.absmax = amax;
    const float absmax = std::fmax(amax, 1e-4f);
    const float scale = std::exp2(std::ceil(std::log2(absmax / 448.0f)));
    out.scale = scale;
    for (int d = 0; d < D; ++d) {
        const float q =
            std::fmin(std::fmax(x[static_cast<size_t>(d)] / scale, -448.0f), 448.0f);
        out.bytes[static_cast<size_t>(d)] = f32_to_e4m3_rtne(q);
        out.rotated[static_cast<size_t>(d)] = x[static_cast<size_t>(d)];
    }
    return out;
}

// ── Scoring (§3) — the UNCHANGED lightning MQA math, in the pooled domain ────

/// logits[n] = sum_h relu(sum_d q[h][d] * dequant(entry[n][d])) * weights[h],
/// dequant(entry[n][d]) = e4m3_to_f32(bytes) * scales[n]. Naive fp32; the
/// kernel's reduction tree differs in association, so callers compare with a
/// relative tolerance, never bitwise.
/// @param q       [H][D] query rows (bf16-representable floats)
/// @param weights [H] fp32 head weights (MAY be negative — no activation)
/// @param entries [N][D] float8_e4m3fn bytes
/// @param scales  [N]
inline std::vector<float> ref_kpool_score(const float* q, const float* weights,
                                          const uint8_t* entries, const float* scales,
                                          int N, int H, int D) {
    std::vector<float> out(static_cast<size_t>(N), 0.0f);
    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int h = 0; h < H; ++h) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q[static_cast<size_t>(h) * D + d]
                       * e4m3_to_f32(entries[static_cast<size_t>(n) * D + d]);
            }
            dot *= scales[static_cast<size_t>(n)];
            acc += std::fmax(dot, 0.0f) * weights[static_cast<size_t>(h)];
        }
        out[static_cast<size_t>(n)] = acc;
    }
    return out;
}

// ── Selection (§4) ──────────────────────────────────────────────────────────

struct TopkPools {
    std::vector<int> ids;   ///< [budget] ASCENDING pool ids, -1 padded
    int eff = 0;            ///< number of valid ids
};

/// Causal top-`budget` over pooled entries, mirroring lightning_topk's
/// deterministic contract:
///   * eligibility: pool i is causal iff its ENDPOINT i*P + P-1 <= qpos
///     (floor-causality — the in-progress incomplete pool has no entry);
///   * selection: highest scores first, ties broken toward the LOWER index
///     (DET-TOPK-TIES) on the kernel's own sortable-key total order;
///   * output: the selected SET sorted ASCENDING by index (the kernel's
///     bitonic index sort), -1 padded to `budget`.
inline TopkPools ref_topk_pools(const float* scores, int n, int budget,
                                int qpos, int P) {
    TopkPools out;
    out.ids.assign(static_cast<size_t>(budget), -1);
    std::vector<int> eligible;
    for (int i = 0; i < n; ++i) {
        if (i * P + P - 1 <= qpos) eligible.push_back(i);
    }
    std::stable_sort(eligible.begin(), eligible.end(), [&](int a, int b) {
        const uint32_t ka = sortable_key(scores[static_cast<size_t>(a)]);
        const uint32_t kb = sortable_key(scores[static_cast<size_t>(b)]);
        if (ka != kb) return ka > kb;   // higher score first
        return a < b;                   // tie -> lower index wins
    });
    const int eff = std::min<int>(budget, static_cast<int>(eligible.size()));
    eligible.resize(static_cast<size_t>(eff));
    std::sort(eligible.begin(), eligible.end());
    for (int i = 0; i < eff; ++i) {
        out.ids[static_cast<size_t>(i)] = eligible[static_cast<size_t>(i)];
    }
    out.eff = eff;
    return out;
}

// ── Expansion + always-selected tail (§4) ───────────────────────────────────

struct Expansion {
    std::vector<int> cols;  ///< [out_cols] token indices, -1 padded
    int count = 0;          ///< eff*P + tail — the TOKEN count (not pools)
};

/// Expand `eff` ascending pool ids into token rows and append the tail.
///   cols [0, eff*P)             : pool_ids[c/P]*P + c%P  (-1 if the id is -1)
///   cols [eff*P, eff*P + tail)  : tail_start + (c - eff*P),
///                                 tail_start = (seq_len/P)*P
///   remaining                   : -1
/// Selected pools and the tail are DISJOINT by floor-causality, so no dedup.
inline Expansion ref_kpool_expand(const int* pool_ids, int eff, int seq_len,
                                  int P, int out_cols) {
    Expansion out;
    out.cols.assign(static_cast<size_t>(out_cols), -1);
    const int tail_start = (seq_len / P) * P;
    const int tail = seq_len - tail_start;
    out.count = eff * P + tail;
    for (int c = 0; c < out_cols; ++c) {
        if (c < eff * P) {
            const int pid = pool_ids[static_cast<size_t>(c / P)];
            out.cols[static_cast<size_t>(c)] = pid >= 0 ? pid * P + c % P : -1;
        } else if (c < out.count) {
            out.cols[static_cast<size_t>(c)] = tail_start + (c - eff * P);
        }
    }
    return out;
}

}  // namespace kpool_ref

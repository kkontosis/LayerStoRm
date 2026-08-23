// CPU MoE non-GEMM kernels for the expert FFN (C-8 SwiGLU, C-9 permute/unpermute).
//
// Implements everything declared in cpu_moe_kernels.h:
//   * cpu_swiglu       — SiLU(gate)*up, both interleaved [N,2d] and separate
//                        gate/up [N,d] layouts; FP32 math, BF16 out; AVX-512
//                        vectorized SiLU where available, scalar fallback.
//   * cpu_moe_permute  — build (expert_id, token) pairs, stable-sort by expert,
//                        scatter rows into expert-contiguous order, fill
//                        expert_offsets / src_to_dest_map / permuted_idx.
//   * cpu_moe_unpermute— gather permuted rows back per token, FP32 weighted
//                        accumulate (topk_weights), BF16 store; zero output first.
//
// dtype-generic over elem_size_bytes (2 == BF16; asserted — the only dtype the
// expert FFN uses today). CPU-only TU — NO CUDA/GPU SDK header (INV-GPU-1,
// enforced by layerstorm_no_cuda_check).

#include "compute/cpu/cpu_moe_kernels.h"

#include "compute/cpu/numa_thread_pool.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace layerstorm::compute::cpu {

namespace {

// ── BF16 <-> FP32 scalar conversion ─────────────────────────────────────────
// Convention matches src/core/bf16_convert.h and tests/unit/nvfp4_gemm_test.cpp:
// decode = bits << 16 reinterpreted; encode = round-to-nearest-even.

inline float bf16_to_f32(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // Round-to-nearest-even (matches f32_to_bf16 in core/bf16_convert.h).
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

// Split a [0, total) range into `n_threads` contiguous slices and return the
// half-open [begin, end) owned by `thread_id`. Balanced (remainder spread over
// the first slices).
inline void partition_range(int total, int thread_id, int n_threads, int& begin,
                            int& end) {
    int base = total / n_threads;
    int rem = total % n_threads;
    int b = thread_id * base + std::min(thread_id, rem);
    int e = b + base + (thread_id < rem ? 1 : 0);
    begin = b;
    end = e;
}

// SiLU matching the GPU `fused_swiglu` kernel EXACTLY (TD-CPU-EXPERT-SWIGLU-
// PARITY): the device's dominant (ELEMS_PER_VEC=8) path computes
//   silu_g = gf * (0.5f * tanhf(0.5f * gf) + 0.5f)
// — the *tanh* sigmoid form (deps/LayerStoRmExpertKernels/.../fused_swiglu.cu),
// NOT x/(1+exp(-x)). Reproducing that exact expression in fp32 with host tanhf
// makes cpu_swiglu bit-parity with the GPU up to the tanhf-implementation ULP,
// which the bf16 store (7-bit mantissa) absorbs on all but rare rounding-
// boundary straddles. The prior AVX512 Cephes-exp SiLU used a different
// algebraic form AND a different transcendental, which is what produced the
// ~1-ULP down_o drift and the argmax flip at decode token 22. SwiGLU is ~0.6%
// of the host expert-FFN cost (NumaCpuExpert.SingleExpertBreakdown), so the
// scalar host tanhf carries no meaningful wall cost.
inline float silu_gpu(float x) {
    return x * (0.5f * std::tanh(0.5f * x) + 0.5f);
}

#if defined(__AVX512F__)
// Load 16 BF16 (uint16) lanes -> __m512 f32 (bits << 16).
inline __m512 load_bf16_16(const uint16_t* p) {
    __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    __m512i w = _mm512_cvtepu16_epi32(h);
    w = _mm512_slli_epi32(w, 16);
    return _mm512_castsi512_ps(w);
}

// Store __m512 f32 -> 16 BF16 (uint16) lanes, round-to-nearest-even.
inline void store_bf16_16(uint16_t* p, __m512 f) {
    __m512i bits = _mm512_castps_si512(f);
    // RNE bias: bits + 0x7FFF + ((bits>>16)&1)
    __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    bits = _mm512_add_epi32(bits, _mm512_set1_epi32(0x7FFF));
    bits = _mm512_add_epi32(bits, lsb);
    bits = _mm512_srli_epi32(bits, 16);
    __m256i h = _mm512_cvtepi32_epi16(bits);  // truncate low 16 bits per lane
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), h);
}
#endif  // __AVX512F__

// One token row of SwiGLU: out[j] = SiLU(gate[j]) * up[j], j in [0, d).
// Scalar host tanhf per element, matching the GPU fused_swiglu formula exactly
// (silu_gpu) for maximal bit-parity — the vectorized Cephes-exp SiLU was the
// TD-CPU-EXPERT-SWIGLU-PARITY drift source and is intentionally removed (SwiGLU
// is ~0.6% of the host FFN, so there is no measurable throughput cost).
inline void swiglu_row(uint16_t* out, const uint16_t* gate, const uint16_t* up,
                       int d, float swiglu_limit) {
    const bool do_clamp = swiglu_limit > 0.0f;
    for (int j = 0; j < d; ++j) {
        float g = bf16_to_f32(gate[j]);
        float u = bf16_to_f32(up[j]);
        if (do_clamp) {
            // llama.cpp DEEPSEEK4 clamp (V4-4b), same fminf/fmaxf order as
            // the GPU kernel for bit parity.
            g = std::fmin(g, swiglu_limit);
            u = std::fmin(std::fmax(u, -swiglu_limit), swiglu_limit);
        }
        out[j] = f32_to_bf16(silu_gpu(g) * u);
    }
}

}  // namespace

// ── C-8: SwiGLU ──────────────────────────────────────────────────────────────

void cpu_swiglu(void* output, const void* input, const void* up_in,
                int num_tokens, int d, bool gate_up_interleaved,
                int elem_size_bytes, NumaThreadPool* pool,
                float swiglu_limit) {
    assert(elem_size_bytes == 2 && "cpu_swiglu: only BF16 (elem_size_bytes==2)");
    (void)elem_size_bytes;
    if (num_tokens <= 0 || d <= 0) {
        return;
    }

    auto* out = static_cast<uint16_t*>(output);
    const auto* in = static_cast<const uint16_t*>(input);

    // Per-token gate/up base pointers. Interleaved: input is [num_tokens, 2*d]
    // with gate = cols[0:d], up = cols[d:2d] (up_in ignored). Separate: input is
    // the gate buffer [num_tokens, d] and up_in the up buffer [num_tokens, d].
    const int gate_stride = gate_up_interleaved ? 2 * d : d;
    const uint16_t* up_base =
        gate_up_interleaved ? nullptr : static_cast<const uint16_t*>(up_in);

    auto run_rows = [&](int t_begin, int t_end) {
        for (int t = t_begin; t < t_end; ++t) {
            const uint16_t* gate = in + static_cast<size_t>(t) * gate_stride;
            const uint16_t* up = gate_up_interleaved
                                     ? gate + d
                                     : up_base + static_cast<size_t>(t) * d;
            uint16_t* o = out + static_cast<size_t>(t) * d;
            swiglu_row(o, gate, up, d, swiglu_limit);
        }
    };

    if (pool == nullptr || pool->num_threads() <= 1) {
        run_rows(0, num_tokens);
        return;
    }
    // Embarrassingly parallel over tokens — no intra-region barrier needed.
    pool->parallel_for([&](int thread_id, int n_threads, CpuBarrierState&) {
        int b, e;
        partition_range(num_tokens, thread_id, n_threads, b, e);
        run_rows(b, e);
    });
}

// ── C-9: permute ─────────────────────────────────────────────────────────────

void cpu_moe_permute(void* permuted_input, int32_t* expert_offsets,
                     int32_t* src_to_dest_map, int32_t* permuted_idx,
                     const void* hidden_states, const int32_t* topk_indices,
                     int num_tokens, int topk, int hidden_dim, int num_experts,
                     int elem_size_bytes, NumaThreadPool* pool) {
    assert(elem_size_bytes == 2 && "cpu_moe_permute: only BF16 (elem_size_bytes==2)");
    (void)elem_size_bytes;

    const int total = num_tokens * topk;

    // expert_offsets[0..num_experts]; always defined even for empty inputs.
    for (int e = 0; e <= num_experts; ++e) {
        expert_offsets[e] = 0;
    }
    if (total <= 0 || num_tokens <= 0 || topk <= 0) {
        return;
    }

    // 1) Histogram routed pairs per expert. Sentinel expert -1 (a dropped /
    //    padded slot) maps to bucket `num_experts` (placed AFTER all real
    //    experts, so it never lands in any expert's GEMM slice).
    //    expert id of pair p = topk_indices[p]; routed_expert = clamp to
    //    [0,num_experts] with -1 -> num_experts.
    std::vector<int32_t> pair_expert(total);
    std::vector<int32_t> counts(num_experts + 1, 0);
    for (int p = 0; p < total; ++p) {
        int32_t e = topk_indices[p];
        int32_t bucket = (e < 0 || e >= num_experts) ? num_experts : e;
        pair_expert[p] = bucket;
        ++counts[bucket];
    }

    // 2) Prefix sum -> destination cursor per bucket. expert_offsets is the
    //    cumulative count over REAL experts only ([num_experts+1] entries); the
    //    sentinel bucket's rows live past expert_offsets[num_experts] and are not
    //    consumed by the grouped GEMM (which iterates experts 0..num_experts-1).
    std::vector<int32_t> cursor(num_experts + 1, 0);
    int32_t running = 0;
    for (int e = 0; e <= num_experts; ++e) {
        cursor[e] = running;       // start row of bucket e in permuted order
        if (e < num_experts) {
            expert_offsets[e] = running;
        }
        running += counts[e];
    }
    expert_offsets[num_experts] = running - counts[num_experts];  // = total - sentinels

    // 3) Stable scatter: iterate pairs in original order so within an expert the
    //    rows stay token-ascending (stable). dest = cursor[bucket]++.
    //    permuted_idx[dest] = source token index t (for all rows, incl. sentinel,
    //    so permuted_input is fully described). src_to_dest_map[p] is the gather
    //    index used by unpermute: -1 for a SENTINEL (dropped) pair so unpermute
    //    excludes it from the weighted reduce (the router would zero its weight,
    //    but a -1 dest is unambiguous and never reads the sentinel garbage row).
    std::vector<int32_t> dest_of_pair(total);
    for (int p = 0; p < total; ++p) {
        int32_t bucket = pair_expert[p];
        int32_t dest = cursor[bucket]++;
        dest_of_pair[p] = dest;  // physical row to scatter into (always valid)
        src_to_dest_map[p] = (bucket == num_experts) ? -1 : dest;
        permuted_idx[dest] = p / topk;  // source token index
    }

    // 4) Scatter the activation rows into expert-contiguous order. Each original
    //    token row is replicated to `topk` permuted rows (one per routing slot).
    //    Parallelize the row copies across the pool (independent destinations).
    const auto* hs = static_cast<const uint16_t*>(hidden_states);
    auto* pin = static_cast<uint16_t*>(permuted_input);
    const size_t row_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);

    auto scatter_rows = [&](int p_begin, int p_end) {
        for (int p = p_begin; p < p_end; ++p) {
            int t = p / topk;
            int32_t dest = dest_of_pair[p];
            std::memcpy(pin + static_cast<size_t>(dest) * hidden_dim,
                        hs + static_cast<size_t>(t) * hidden_dim, row_bytes);
        }
    };

    if (pool == nullptr || pool->num_threads() <= 1 || total < 64) {
        scatter_rows(0, total);
        return;
    }
    pool->parallel_for([&](int thread_id, int n_threads, CpuBarrierState&) {
        int b, e;
        partition_range(total, thread_id, n_threads, b, e);
        scatter_rows(b, e);
    });
}

// ── C-9: unpermute ───────────────────────────────────────────────────────────

void cpu_moe_unpermute(void* output, const void* permuted_output,
                       const float* topk_weights, const int32_t* src_to_dest_map,
                       int num_tokens, int topk, int hidden_dim,
                       int elem_size_bytes, NumaThreadPool* pool) {
    assert(elem_size_bytes == 2 && "cpu_moe_unpermute: only BF16 (elem_size_bytes==2)");
    (void)elem_size_bytes;
    if (num_tokens <= 0 || hidden_dim <= 0 || topk <= 0) {
        return;
    }

    auto* out = static_cast<uint16_t*>(output);
    const auto* perm = static_cast<const uint16_t*>(permuted_output);

    // One token: FP32 weighted accumulate over its topk permuted rows, BF16 store.
    // output[t, j] = sum_k topk_weights[t*topk+k] * permuted_output[dest[t*topk+k], j]
    auto run_tokens = [&](int t_begin, int t_end) {
        std::vector<float> acc(hidden_dim);
        for (int t = t_begin; t < t_end; ++t) {
            std::fill(acc.begin(), acc.end(), 0.0f);  // zero output first
            for (int k = 0; k < topk; ++k) {
                int idx = t * topk + k;
                float w = topk_weights[idx];
                int32_t dest = src_to_dest_map[idx];
                if (dest < 0) {
                    continue;  // dropped/padded slot
                }
                const uint16_t* row = perm + static_cast<size_t>(dest) * hidden_dim;
                int j = 0;
#if defined(__AVX512F__)
                __m512 wv = _mm512_set1_ps(w);
                for (; j + 16 <= hidden_dim; j += 16) {
                    __m512 a = _mm512_loadu_ps(acc.data() + j);
                    __m512 r = load_bf16_16(row + j);
                    a = _mm512_fmadd_ps(wv, r, a);
                    _mm512_storeu_ps(acc.data() + j, a);
                }
#endif
                for (; j < hidden_dim; ++j) {
                    acc[j] += w * bf16_to_f32(row[j]);
                }
            }
            uint16_t* o = out + static_cast<size_t>(t) * hidden_dim;
            int j = 0;
#if defined(__AVX512F__)
            for (; j + 16 <= hidden_dim; j += 16) {
                store_bf16_16(o + j, _mm512_loadu_ps(acc.data() + j));
            }
#endif
            for (; j < hidden_dim; ++j) {
                o[j] = f32_to_bf16(acc[j]);
            }
        }
    };

    if (pool == nullptr || pool->num_threads() <= 1) {
        run_tokens(0, num_tokens);
        return;
    }
    pool->parallel_for([&](int thread_id, int n_threads, CpuBarrierState&) {
        int b, e;
        partition_range(num_tokens, thread_id, n_threads, b, e);
        run_tokens(b, e);
    });
}

void cpu_moe_unpermute_perslot(void* perslot_output, const void* permuted_output,
                               const float* topk_weights,
                               const int32_t* src_to_dest_map,
                               int num_tokens, int topk, int hidden_dim,
                               bool fp32_output, NumaThreadPool* pool) {
    if (num_tokens <= 0 || hidden_dim <= 0 || topk <= 0) {
        return;
    }

    const auto* perm = static_cast<const uint16_t*>(permuted_output);
    auto* out_f32 = static_cast<float*>(perslot_output);
    auto* out_bf16 = static_cast<uint16_t*>(perslot_output);

    // Per original token, emit K contiguous slot rows. Slot k holds the single
    // per-slot contribution c_k = w_k * expert_out_k (NO cross-slot accumulate),
    // exactly matching the GPU finalize_moe_routing_bf16_to_{fp32,bf16}_perslot
    // kernels: a single fp32 multiply, optionally rounded to bf16 once. A dropped
    // slot (dest < 0) writes an all-zero row (0 + x = x is bit-exact in the later
    // gather/reduce). Placement-invariance comes from the fixed slot ORDER.
    auto run_tokens = [&](int t_begin, int t_end) {
        for (int t = t_begin; t < t_end; ++t) {
            for (int k = 0; k < topk; ++k) {
                const int idx = t * topk + k;
                const size_t row = static_cast<size_t>(idx) * hidden_dim;
                const int32_t dest = src_to_dest_map[idx];
                if (dest < 0) {
                    if (fp32_output)
                        std::fill(out_f32 + row, out_f32 + row + hidden_dim, 0.0f);
                    else
                        std::fill(out_bf16 + row, out_bf16 + row + hidden_dim,
                                  static_cast<uint16_t>(0));
                    continue;
                }
                const float w = topk_weights[idx];
                const uint16_t* src = perm + static_cast<size_t>(dest) * hidden_dim;
                if (fp32_output) {
                    for (int j = 0; j < hidden_dim; ++j)
                        out_f32[row + j] = w * bf16_to_f32(src[j]);
                } else {
                    for (int j = 0; j < hidden_dim; ++j)
                        out_bf16[row + j] = f32_to_bf16(w * bf16_to_f32(src[j]));
                }
            }
        }
    };

    if (pool == nullptr || pool->num_threads() <= 1) {
        run_tokens(0, num_tokens);
        return;
    }
    pool->parallel_for([&](int thread_id, int n_threads, CpuBarrierState&) {
        int b, e;
        partition_range(num_tokens, thread_id, n_threads, b, e);
        run_tokens(b, e);
    });
}

}  // namespace layerstorm::compute::cpu

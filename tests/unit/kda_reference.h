// kda_reference.h — CPU reference implementation of Kimi Delta Attention (KDA)
// for GLM-5.3-Flash (`glm5_next`), PLAN.md GF3.6.
//
// ── What this is ─────────────────────────────────────────────────────────────
// The SPECIFICATION ARTIFACT for the KDA leg of the glm5_next integration.
// GF3.7 (SM120 kernels) and GF3.8 (state pool) are judged against the numbers
// this file produces. It is deliberately naive and heavily documented:
// correctness and legibility over speed. Everything is plain loops in fp32
// (the property tests use a double instantiation of the same template).
//
// KDA = channel-wise (diagonal) gated delta rule (Kimi Linear, arXiv:2510.26692)
// with a bounded-sigmoid "safe gate", a width-4 depthwise causal short conv on
// q/k/v, per-head scalar write strength beta, and a gated RMSNorm on the output.
//
// Transcribed from (and cross-checked against) three independent references:
//  * vLLM tag `pr-53906-head` (ref/vllm):
//      vllm/models/glm5next/nvidia/kda.py            (layer wiring)
//      vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
//          kernel `fused_recurrent_gated_delta_rule_fwd_kernel`
//          (IS_KDA, COMPUTE_GATE, SIGMOID_BETA, USE_QK_L2NORM_IN_KERNEL)
//      vllm/third_party/flash_linear_attention/ops/kda.py
//          (chunked path: `chunk_kda_with_fused_gate` and its sub-kernels,
//           gate `kda_gate_fwd_kernel` USE_LOWER_BOUND branch)
//      vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
//          (inter-chunk state recurrence)
//    The FLA-derived files carry: Copyright (c) 2023-2025, Songlin Yang,
//    Yu Zhang (MIT) — see spec/NOTICES.md.
//  * SGLang tag `pr-36507-head` (ref/sglang):
//      python/sglang/kernels/ops/attention/fla/kda.py (same gate formula).
//  * TensorRT-LLM (ref/TensorRT-LLM):
//      cpp/tensorrt_llm/kernels/kdaDecode/kdaDecode.cu (native-CUDA decode;
//      same decay `expf(lower_bound * sigmoid(exp(A_log) * g_raw))`).
//
// ── The math, per head (D = head_dim = 128; K-axis = V-axis = D) ────────────
// State: S ∈ R^{V x K}, layout S[v][c] (v = value channel, c = key channel).
// Per token, in this exact order (fused_recurrent.py:133-175):
//   1. q,k,v are the post-conv activations (see conv below), fp32.
//   2. L2 norm:   q <- q / sqrt(sum(q^2) + 1e-6)   (eps INSIDE the sqrt);
//                 k likewise.
//   3. Scale:     q <- q * D^{-1/2}.
//   4. Gate (channel-wise over the KEY axis; "safe gate"):
//        g_log[c] = lower_bound * sigmoid(exp(A_log_h) * (raw_g[c] + dt_bias_h[c]))
//      with lower_bound = -5.0, so g_log ∈ (-5, 0) and the per-channel decay
//        decay[c] = exp(g_log[c]) ∈ (e^-5, 1).
//      DIRECTION (easy to get backwards): raw gate -> -inf gives sigmoid -> 0,
//      g_log -> 0, decay -> 1 = REMEMBER; raw gate -> +inf gives decay -> e^-5
//      = maximal forgetting. A_log only sets the sigmoid steepness per head.
//   5. Decay:     S[v][c] *= decay[c]              (every v row, per column c).
//   6. Delta:     err[v]  = v[v] - sum_c S[v][c] * k[c]
//   7. Write str: err    *= sigmoid(beta_raw_h)    (per-head scalar beta).
//   8. Rank-1:    S[v][c] += err[v] * k[c]
//   9. Output:    core[v] = sum_c S[v][c] * q[c]   (AFTER the write: a token
//                 reads its own contribution).
//  10. Gated RMSNorm (o_norm; eps 1e-5, sigmoid activation; kda.py:248 +
//      FusedRMSNormGated in ops/kda.py):
//        out[v] = core[v] / sqrt(mean(core^2) + 1e-5) * w_onorm[v]
//                 * sigmoid(g2_raw[v])
// The engine wrapping (not part of the scan, covered by GF3.3/GF3.9):
//   q/k/v = {q,k,v}_proj(hidden);  raw_g = f_b_proj(f_a_proj(hidden));
//   g2 = g_b_proj(g_a_proj(hidden));  beta_raw = b_proj(hidden);
//   final out = o_proj(concat heads).
//
// ── Short conv (depthwise, causal, width 4, SiLU, NO bias) ──────────────────
// Channel ch = h*D + d, C = H*D channels, separate weights for q, k, v
// (checkpoint `{q,k,v}_conv1d.weight` [C,1,4]; last tap = current token):
//   conv_out[ch][t] = silu( sum_{j=0..3} w[ch][j] * x[ch][t-3+j] )
// x[<0] comes from the carried conv state (the last W-1 = 3 RAW inputs,
// oldest first). The conv runs BEFORE the L2 norm; its state stores PRE-conv,
// PRE-SiLU inputs. There is no conv bias tensor in the checkpoint.
//
// ── Chunked prefill scan (chunk = 64; UT/WY form) ───────────────────────────
// Mathematically identical to running step 1-9 token by token; the chunked
// form exists so prefill can use GEMMs. Per chunk of length L, per head
// (kda.py `chunk_kda_with_fused_gate` decomposition):
//   gcum[i][c]   = inclusive prefix sum of g_log within the chunk
//   A[i][j]      = beta_i * sum_c kn_i[c] kn_j[c] e^{gcum_i[c]-gcum_j[c]}, i>j
//   Aqk[i][j]    = sum_c qn_i[c] kn_j[c] e^{gcum_i[c]-gcum_j[c]},         i>=j
//   M            = (I + A)^{-1}            (unit lower triangular; exact)
//   w_i[c]       = sum_j M[i][j] beta_j kn_j[c] e^{gcum_j[c]}
//   u_i[v]       = sum_j M[i][j] beta_j v_j[v]
//   v_new_i[v]   = u_i[v] - sum_c w_i[c] S0[v][c]     (S0 = state at chunk start)
//   core_i[v]    = sum_c qn_i[c] e^{gcum_i[c]} S0[v][c] + sum_{j<=i} Aqk[i][j] v_new_j[v]
//   S1[v][c]     = S0[v][c] e^{gcum_{L-1}[c]}
//                  + sum_i kn_i[c] e^{gcum_{L-1}[c]-gcum_i[c]} v_new_i[v]
// (qn/kn are the l2-normed vectors, qn pre-scaled by D^{-1/2}. The Triton
// kernels evaluate the exponentials in base-2 with a 1/ln2 rescale and use
// running anchors for stability; that is an implementation detail, not
// semantics — this reference uses natural exp directly in fp32.)
//
// ── Why the state has NO POSITION AXIS (GF3.2 capability answer) ────────────
// Every KV-style structure in this engine is position-addressed: rewinding n
// tokens = dropping n rows. The KDA state is a single D x D matrix per head
// into which every token's rank-1 write has been folded after channel-wise
// decay. Two consequences, which tests in kda_reference_test.cpp demonstrate:
//   (a) There is no operation on S_t that yields S_{t-n} without replaying the
//       inputs: the state is 128x128 floats per head regardless of how many
//       tokens it absorbed — the per-token information is not there.
//   (b) Even WITH all token inputs, backward reconstruction is exponentially
//       ill-conditioned: one step is algebraically invertible
//       (err = (v - S_t k)/(1 - beta * k.k); S' = S_t - beta err k^T;
//        S_{t-1} = S' * e^{-g}), but e^{-g} amplifies error by up to e^5 ~ 148
//       per channel per step and 1/(1-beta) diverges as beta -> 1. Inverting n
//       steps multiplies these factors — see kda_ref::inverse_step_head and
//       the RewindIsNotPositionAddressable test, which measures the blow-up.
// Hence `lossy_position_indexed_state() == true` for glm5_next: rewind is
// anchor-and-replay (GF3.11), prefix reuse exists only at explicit state
// checkpoints (GF3.12), and truncating forks are rejected. Forward
// checkpoint/replay is EXACT: capturing (S, conv state) at any token boundary
// and resuming reproduces the uninterrupted run bit-for-bit in this reference
// (chunk boundaries are just the natural place the engine will do it).
//
// ── Precision policy ────────────────────────────────────────────────────────
// All accumulation fp32 (double in the property tests). GF3.7 kernels MUST
// keep the state and the scan arithmetic in fp32: llama.cpp #27754 needed
// NVIDIA_TF32_OVERRIDE=0 — TF32 contraction inside the scan is a known
// accuracy trap. The GPU gate must run TF32 both ways (PLAN GF3.6/GF3.7).
// Inputs in the golden fixtures are bf16-rounded so GPU kernels consuming
// bf16 activations can hit the same numbers.

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace layerstorm::kda_ref {

// ── Small utilities ─────────────────────────────────────────────────────────

template <typename Real>
inline Real sigmoid(Real x) {
    return Real(1) / (Real(1) + std::exp(-x));
}

template <typename Real>
inline Real silu(Real x) {
    return x * sigmoid(x);
}

/// Round an fp32 value to the nearest bfloat16 (round-to-nearest-even) and
/// back. Fixture inputs are pushed through this so the reference consumes
/// exactly what a bf16 engine path would.
inline float bf16_round(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float y;
    std::memcpy(&y, &u, 4);
    return y;
}

/// Decode a raw bfloat16 (as stored in the checkpoint / fixtures) to fp32.
inline float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float y;
    std::memcpy(&y, &u, 4);
    return y;
}

/// splitmix64 — the deterministic stream both the golden generator
/// (tools/gen_kda_golden.py) and the C++ tests use to synthesize inputs.
/// Kept here so the two implementations cannot drift apart silently.
inline uint64_t splitmix64(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// Uniform [0,1) with 24-bit granularity — every value is exact in fp32, so
/// the Python generator and this code produce bit-identical streams.
inline float splitmix_u01(uint64_t r) {
    return static_cast<float>((r >> 40) * (1.0 / 16777216.0));
}

/// Uniform [lo,hi) draw from a splitmix64 stream (exact-in-fp32 lattice).
inline float splitmix_uniform(uint64_t& s, float lo, float hi) {
    return static_cast<float>(lo + (double(hi) - double(lo)) * splitmix_u01(splitmix64(s)));
}

// ── Geometry / hyper-parameters ─────────────────────────────────────────────

struct KdaParams {
    int num_heads = 64;         // linear_attn_config.num_heads
    int head_dim = 128;         // linear_attn_config.head_dim (= K = V)
    float lower_bound = -5.0f;  // linear_attn_config.gate_lower_bound
    int conv_width = 4;         // linear_attn_config.short_conv_kernel_size
    float l2_eps = 1e-6f;       // q/k L2-norm epsilon (inside the sqrt)
    float onorm_eps = 1e-5f;    // o_norm gated-RMSNorm epsilon

    float scale() const { return 1.0f / std::sqrt(static_cast<float>(head_dim)); }
    int channels() const { return num_heads * head_dim; }
};

/// Per-request recurrent state for one layer: S[h][v][c], fp32.
/// This plus the conv state below is the ENTIRE sequence memory of a KDA
/// layer — ~4 MiB at H=64, D=128 — regardless of sequence length.
struct KdaState {
    std::vector<float> s;  // [H][D][D], S[h][v][c]

    static KdaState zeros(const KdaParams& p) {
        KdaState st;
        st.s.assign(static_cast<size_t>(p.num_heads) * p.head_dim * p.head_dim, 0.0f);
        return st;
    }
    float* head(const KdaParams& p, int h) {
        return s.data() + static_cast<size_t>(h) * p.head_dim * p.head_dim;
    }
    const float* head(const KdaParams& p, int h) const {
        return s.data() + static_cast<size_t>(h) * p.head_dim * p.head_dim;
    }
};

/// Carried conv state for ONE of the three convs (q, k or v):
/// the last (conv_width-1) RAW inputs per channel, oldest first.
struct ConvState {
    std::vector<float> ring;  // [C][W-1]

    static ConvState zeros(const KdaParams& p) {
        ConvState cs;
        cs.ring.assign(static_cast<size_t>(p.channels()) * (p.conv_width - 1), 0.0f);
        return cs;
    }
};

// ── Gate ────────────────────────────────────────────────────────────────────

/// Safe-gate log-decay for one head and one token:
///   g_log[c] = lower_bound * sigmoid(exp(a_log_h) * (raw_g[c] + dt_bias_h[c]))
/// raw_g / dt_bias_h are the per-head slices ([D]).
template <typename Real>
inline void safe_gate_log_decay(const Real* raw_g, const Real* dt_bias_h, Real a_log_h,
                                Real lower_bound, int d, Real* g_log_out) {
    const Real a = std::exp(a_log_h);
    for (int c = 0; c < d; ++c) {
        g_log_out[c] = lower_bound * sigmoid(a * (raw_g[c] + dt_bias_h[c]));
    }
}

// ── Short conv ──────────────────────────────────────────────────────────────

/// One decode step of the depthwise causal conv + SiLU for all C channels.
/// `x` [C] raw inputs; `w` [C][W]; state advanced in place; `out` [C].
inline void conv_step(const KdaParams& p, const float* x, const float* w, ConvState& cs,
                      float* out) {
    const int W = p.conv_width;
    const int C = p.channels();
    for (int ch = 0; ch < C; ++ch) {
        float* ring = cs.ring.data() + static_cast<size_t>(ch) * (W - 1);
        const float* wc = w + static_cast<size_t>(ch) * W;
        float acc = 0.0f;
        for (int j = 0; j < W - 1; ++j) acc += wc[j] * ring[j];
        acc += wc[W - 1] * x[ch];
        out[ch] = silu(acc);
        for (int j = 0; j + 1 < W - 1; ++j) ring[j] = ring[j + 1];
        ring[W - 2] = x[ch];
    }
}

/// Prefill conv over T tokens. `x` [T][C] raw inputs, `out` [T][C].
/// Equivalent to T calls of conv_step (and tested to be).
inline void conv_prefill(const KdaParams& p, const float* x, int t_len, const float* w,
                         ConvState& cs, float* out) {
    for (int t = 0; t < t_len; ++t) {
        conv_step(p, x + static_cast<size_t>(t) * p.channels(), w, cs,
                  out + static_cast<size_t>(t) * p.channels());
    }
}

// ── Recurrent path (decode; also the arbiter for the chunked path) ──────────

/// One token, one head. `s` is S[v][c] for this head ([D][D], mutated).
/// q/k/v are POST-conv activations [D]; raw_g [D]; dt_bias_h [D].
/// Emits the pre-o_norm core output [D].
template <typename Real>
inline void recurrent_step_head(int d, Real lower_bound, Real l2_eps, Real scale,
                                const Real* q_in, const Real* k_in, const Real* v_in,
                                const Real* raw_g, Real beta_raw, Real a_log_h,
                                const Real* dt_bias_h, Real* s, Real* core_out) {
    std::vector<Real> q(d), k(d), g_log(d);
    // 2. L2 norm (eps inside the sqrt), 3. scale on q.
    Real qs = 0, ks = 0;
    for (int c = 0; c < d; ++c) qs += q_in[c] * q_in[c];
    for (int c = 0; c < d; ++c) ks += k_in[c] * k_in[c];
    const Real qr = Real(1) / std::sqrt(qs + l2_eps);
    const Real kr = Real(1) / std::sqrt(ks + l2_eps);
    for (int c = 0; c < d; ++c) q[c] = q_in[c] * qr * scale;
    for (int c = 0; c < d; ++c) k[c] = k_in[c] * kr;
    // 4. gate
    safe_gate_log_decay(raw_g, dt_bias_h, a_log_h, lower_bound, d, g_log.data());
    // 5. decay, 6. delta readout, 7. beta, 8. rank-1 write, 9. output.
    const Real beta = sigmoid(beta_raw);
    for (int v = 0; v < d; ++v) {
        Real* row = s + static_cast<size_t>(v) * d;
        Real dot = 0;
        for (int c = 0; c < d; ++c) {
            row[c] *= std::exp(g_log[c]);
            dot += row[c] * k[c];
        }
        const Real err = (v_in[v] - dot) * beta;
        Real o = 0;
        for (int c = 0; c < d; ++c) {
            row[c] += err * k[c];
            o += row[c] * q[c];
        }
        core_out[v] = o;
    }
}

/// One token, all heads. Inputs laid out [H][D] (q/k/v/raw_g), [H] beta.
inline void recurrent_step(const KdaParams& p, const float* q, const float* k, const float* v,
                           const float* raw_g, const float* beta_raw, const float* a_log,
                           const float* dt_bias, KdaState& st, float* core_out) {
    for (int h = 0; h < p.num_heads; ++h) {
        const size_t off = static_cast<size_t>(h) * p.head_dim;
        recurrent_step_head<float>(p.head_dim, p.lower_bound, p.l2_eps, p.scale(), q + off,
                                   k + off, v + off, raw_g + off, beta_raw[h], a_log[h],
                                   dt_bias + off, st.head(p, h), core_out + off);
    }
}

/// T-token scan via the recurrence (the "obviously right" path).
/// All activations [T][H][D] (beta [T][H]); core_out [T][H][D].
inline void recurrent_scan(const KdaParams& p, const float* q, const float* k, const float* v,
                           const float* raw_g, const float* beta_raw, const float* a_log,
                           const float* dt_bias, int t_len, KdaState& st, float* core_out) {
    const size_t tok = static_cast<size_t>(p.num_heads) * p.head_dim;
    for (int t = 0; t < t_len; ++t) {
        recurrent_step(p, q + t * tok, k + t * tok, v + t * tok, raw_g + t * tok,
                       beta_raw + static_cast<size_t>(t) * p.num_heads, a_log, dt_bias, st,
                       core_out + t * tok);
    }
}

// ── Chunked-parallel prefill scan ───────────────────────────────────────────

inline constexpr int kChunkSize = 64;  // FLA_CHUNK_SIZE (ops/utils.py)

/// Chunk-parallel scan, mathematically ≡ recurrent_scan (the agreement gate).
/// Same signature as recurrent_scan; processes ceil(T/64) chunks. The state
/// after every chunk equals the recurrent state at that boundary — chunk
/// boundaries are the natural state-carry / checkpoint points (GF3.9, GF3.12).
inline void chunked_scan(const KdaParams& p, const float* q, const float* k, const float* v,
                         const float* raw_g, const float* beta_raw, const float* a_log,
                         const float* dt_bias, int t_len, KdaState& st, float* core_out,
                         int chunk_size = kChunkSize) {
    const int d = p.head_dim;
    const size_t tok = static_cast<size_t>(p.num_heads) * d;
    const float scale = p.scale();

    // Scratch (per chunk, per head).
    const int bt = chunk_size;
    std::vector<float> qn(static_cast<size_t>(bt) * d), kn(static_cast<size_t>(bt) * d);
    std::vector<float> gcum(static_cast<size_t>(bt) * d), beta(bt);
    std::vector<float> a_mat(static_cast<size_t>(bt) * bt), aqk(static_cast<size_t>(bt) * bt);
    std::vector<float> m_inv(static_cast<size_t>(bt) * bt);
    std::vector<float> w_mat(static_cast<size_t>(bt) * d), u_mat(static_cast<size_t>(bt) * d);
    std::vector<float> v_new(static_cast<size_t>(bt) * d);

    for (int t0 = 0; t0 < t_len; t0 += chunk_size) {
        const int len = std::min(chunk_size, t_len - t0);
        for (int h = 0; h < p.num_heads; ++h) {
            float* s = st.head(p, h);
            const size_t hoff = static_cast<size_t>(h) * d;

            // Per-token preprocessing: l2norm, scale, gate + inclusive cumsum.
            for (int i = 0; i < len; ++i) {
                const size_t in_off = static_cast<size_t>(t0 + i) * tok + hoff;
                const float* qi = q + in_off;
                const float* ki = k + in_off;
                float qs = 0, ks = 0;
                for (int c = 0; c < d; ++c) qs += qi[c] * qi[c];
                for (int c = 0; c < d; ++c) ks += ki[c] * ki[c];
                const float qr = 1.0f / std::sqrt(qs + p.l2_eps);
                const float kr = 1.0f / std::sqrt(ks + p.l2_eps);
                for (int c = 0; c < d; ++c) qn[i * d + c] = qi[c] * qr * scale;
                for (int c = 0; c < d; ++c) kn[i * d + c] = ki[c] * kr;
                safe_gate_log_decay(raw_g + in_off, dt_bias + hoff, a_log[h], p.lower_bound,
                                    d, gcum.data() + static_cast<size_t>(i) * d);
                if (i > 0) {
                    for (int c = 0; c < d; ++c) gcum[i * d + c] += gcum[(i - 1) * d + c];
                }
                beta[i] = sigmoid(beta_raw[static_cast<size_t>(t0 + i) * p.num_heads + h]);
            }

            // A (strictly lower, beta-scaled KK^T with relative decay) and
            // Aqk (lower incl. diagonal, decayed QK^T; q carries the scale).
            for (int i = 0; i < len; ++i) {
                for (int j = 0; j < len; ++j) {
                    float acc_a = 0, acc_qk = 0;
                    if (j <= i) {
                        for (int c = 0; c < d; ++c) {
                            const float decay = std::exp(gcum[i * d + c] - gcum[j * d + c]);
                            const float kj = kn[j * d + c] * decay;
                            if (j < i) acc_a += kn[i * d + c] * kj;
                            acc_qk += qn[i * d + c] * kj;
                        }
                    }
                    a_mat[i * bt + j] = (j < i) ? beta[i] * acc_a : 0.0f;
                    aqk[i * bt + j] = (j <= i) ? acc_qk : 0.0f;
                }
            }

            // M = (I + A)^{-1} by forward substitution (A strictly lower =>
            // I + A is unit lower triangular, inverse is exact).
            for (int i = 0; i < len; ++i) {
                for (int j = 0; j <= i; ++j) {
                    float x = (i == j) ? 1.0f : 0.0f;
                    for (int l = j; l < i; ++l) x -= a_mat[i * bt + l] * m_inv[l * bt + j];
                    m_inv[i * bt + j] = x;
                }
            }

            // w = M (beta k e^{gcum}); u = M (beta v).
            for (int i = 0; i < len; ++i) {
                for (int c = 0; c < d; ++c) w_mat[i * d + c] = 0.0f;
                for (int c = 0; c < d; ++c) u_mat[i * d + c] = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    const float mij = m_inv[i * bt + j];
                    if (mij == 0.0f) continue;
                    const float bj = beta[j];
                    const float* vj = v + static_cast<size_t>(t0 + j) * tok + hoff;
                    for (int c = 0; c < d; ++c) {
                        w_mat[i * d + c] +=
                            mij * bj * kn[j * d + c] * std::exp(gcum[j * d + c]);
                        u_mat[i * d + c] += mij * bj * vj[c];
                    }
                }
            }

            // v_new = u - w S0^T ; core = q e^{gcum} S0^T + tril(Aqk) v_new.
            for (int i = 0; i < len; ++i) {
                float* vni = v_new.data() + static_cast<size_t>(i) * d;
                float* out = core_out + static_cast<size_t>(t0 + i) * tok + hoff;
                for (int vv = 0; vv < d; ++vv) {
                    const float* row = s + static_cast<size_t>(vv) * d;
                    float acc_w = 0, acc_q = 0;
                    for (int c = 0; c < d; ++c) {
                        acc_w += w_mat[i * d + c] * row[c];
                        acc_q += qn[i * d + c] * std::exp(gcum[i * d + c]) * row[c];
                    }
                    vni[vv] = u_mat[i * d + vv] - acc_w;
                    out[vv] = acc_q;  // + intra-chunk term below
                }
            }
            for (int i = 0; i < len; ++i) {
                float* out = core_out + static_cast<size_t>(t0 + i) * tok + hoff;
                for (int j = 0; j <= i; ++j) {
                    const float aij = aqk[i * bt + j];
                    const float* vnj = v_new.data() + static_cast<size_t>(j) * d;
                    for (int vv = 0; vv < d; ++vv) out[vv] += aij * vnj[vv];
                }
            }

            // State to end-of-chunk:
            //   S1 = S0 e^{gcum_last} + sum_i (kn_i e^{gcum_last - gcum_i}) v_new_i^T
            const float* g_last = gcum.data() + static_cast<size_t>(len - 1) * d;
            for (int vv = 0; vv < d; ++vv) {
                float* row = s + static_cast<size_t>(vv) * d;
                for (int c = 0; c < d; ++c) row[c] *= std::exp(g_last[c]);
            }
            for (int i = 0; i < len; ++i) {
                const float* vni = v_new.data() + static_cast<size_t>(i) * d;
                for (int vv = 0; vv < d; ++vv) {
                    float* row = s + static_cast<size_t>(vv) * d;
                    const float vnv = vni[vv];
                    for (int c = 0; c < d; ++c) {
                        row[c] += kn[i * d + c] * std::exp(g_last[c] - gcum[i * d + c]) * vnv;
                    }
                }
            }
        }
    }
}

// ── Output norm ─────────────────────────────────────────────────────────────

/// Gated RMSNorm over head_dim with sigmoid activation (o_norm):
///   out[v] = core[v]/sqrt(mean(core^2)+eps) * w[v] * sigmoid(g2_raw[v])
/// One head; w is the shared [D] o_norm.weight.
inline void gated_rmsnorm_head(int d, float eps, const float* core, const float* g2_raw,
                               const float* w, float* out) {
    float ss = 0;
    for (int v = 0; v < d; ++v) ss += core[v] * core[v];
    const float r = 1.0f / std::sqrt(ss / d + eps);
    for (int v = 0; v < d; ++v) out[v] = core[v] * r * w[v] * sigmoid(g2_raw[v]);
}

/// All heads, T tokens: core/g2/out [T][H][D].
inline void gated_rmsnorm(const KdaParams& p, const float* core, const float* g2_raw,
                          const float* w, int t_len, float* out) {
    const size_t tok = static_cast<size_t>(p.num_heads) * p.head_dim;
    for (int t = 0; t < t_len; ++t) {
        for (int h = 0; h < p.num_heads; ++h) {
            const size_t off = t * tok + static_cast<size_t>(h) * p.head_dim;
            gated_rmsnorm_head(p.head_dim, p.onorm_eps, core + off, g2_raw + off, w,
                               out + off);
        }
    }
}

// ── Full scan drivers (conv → scan → o_norm) ────────────────────────────────

enum class ScanPath { kRecurrent, kChunked };

/// End-to-end KDA scan (everything between the projections and o_proj):
/// raw q/k/v [T][H*D] → conv+SiLU → l2norm/scale/gate/delta scan → o_norm.
/// Mutates the recurrent state and the three conv states in place.
inline void kda_scan(const KdaParams& p, ScanPath path, const float* q_raw, const float* k_raw,
                     const float* v_raw, const float* raw_g, const float* g2_raw,
                     const float* beta_raw, const float* a_log, const float* dt_bias,
                     const float* conv_wq, const float* conv_wk, const float* conv_wv,
                     const float* onorm_w, int t_len, KdaState& st, ConvState& cs_q,
                     ConvState& cs_k, ConvState& cs_v, float* core_out, float* onorm_out) {
    const size_t n = static_cast<size_t>(t_len) * p.channels();
    std::vector<float> qc(n), kc(n), vc(n);
    conv_prefill(p, q_raw, t_len, conv_wq, cs_q, qc.data());
    conv_prefill(p, k_raw, t_len, conv_wk, cs_k, kc.data());
    conv_prefill(p, v_raw, t_len, conv_wv, cs_v, vc.data());
    if (path == ScanPath::kRecurrent) {
        recurrent_scan(p, qc.data(), kc.data(), vc.data(), raw_g, beta_raw, a_log, dt_bias,
                       t_len, st, core_out);
    } else {
        chunked_scan(p, qc.data(), kc.data(), vc.data(), raw_g, beta_raw, a_log, dt_bias,
                     t_len, st, core_out);
    }
    gated_rmsnorm(p, core_out, g2_raw, onorm_w, t_len, onorm_out);
}

// ── Rewind demonstration (documented as ill-conditioned; test-only) ─────────

/// Algebraic inverse of ONE recurrent step, given the state AFTER the step and
/// the token's (post-conv) inputs. Exists to DEMONSTRATE that KDA rewind is
/// not viable: correct in exact arithmetic, exponentially error-amplifying in
/// floating point (each channel multiplies error by e^{-g_log} <= e^5, and the
/// delta inversion by 1/(1-beta k.k)). Never use as an engine mechanism —
/// INV-KDA-REWIND territory: anchor-and-replay only.
template <typename Real>
inline void inverse_step_head(int d, Real lower_bound, Real l2_eps, const Real* k_in,
                              const Real* v_in, const Real* raw_g, Real beta_raw,
                              Real a_log_h, const Real* dt_bias_h,
                              Real* s /* in: S_t, out: S_{t-1} */) {
    std::vector<Real> k(d), g_log(d);
    Real ks = 0;
    for (int c = 0; c < d; ++c) ks += k_in[c] * k_in[c];
    const Real kr = Real(1) / std::sqrt(ks + l2_eps);
    for (int c = 0; c < d; ++c) k[c] = k_in[c] * kr;
    safe_gate_log_decay(raw_g, dt_bias_h, a_log_h, lower_bound, d, g_log.data());
    const Real beta = sigmoid(beta_raw);
    Real kk = 0;
    for (int c = 0; c < d; ++c) kk += k[c] * k[c];
    for (int v = 0; v < d; ++v) {
        Real* row = s + static_cast<size_t>(v) * d;
        Real dot = 0;
        for (int c = 0; c < d; ++c) dot += row[c] * k[c];
        const Real err = (v_in[v] - dot) / (Real(1) - beta * kk);  // forward err pre-beta
        for (int c = 0; c < d; ++c) {
            row[c] = (row[c] - beta * err * k[c]) * std::exp(-g_log[c]);
        }
    }
}

}  // namespace layerstorm::kda_ref

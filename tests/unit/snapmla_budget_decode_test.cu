// TD-SNAPMLA-ATCTX-DECODE-PATH — budget-staged sparse decode at the GLM5N NoPE
// geometry (d_qk 512 = 512 nope + 0 rope, d_v 512, 516 B cache row
// [512 FP8 | 4 B f32 scale]): the gather-mode CKV dequant (stage a) and the
// hardened sparse FP8 split-KV decode kernel (stage b).
//
// The bar, leg for leg:
//
//   (A) GATHER STAGING IS A SUBSET, NOT A REWRITE. Every SELECTED row is
//       memcmp-identical to what the legacy full-range dequant put there, and
//       nothing else is touched at all — the 0xA5 poison survives on every
//       unselected row, every -1 entry, every out-of-range entry and every
//       entry past *gather_count. Both kernels compile the SAME
//       `dequant_ckv_row` body in the same TU, so bit-identity per row is by
//       construction; what this leg actually pins is the LAUNCHER + the
//       gather predicate (g < min(*gather_count, max_gather) &&
//       0 <= row < row_bound) and the scatter address k_out[row].
//
//   (B) READ-SET PROOF. The consumer whose sValid mask mirrors that predicate
//       term for term — the sparse prefill kernel — must produce
//       memcmp-identical out AND lse over poison-holed gather staging and
//       over fully-populated staging. That is the only thing that makes
//       "unselected rows stay UNWRITTEN" safe. Index shape is the production
//       one: topk 2051 (NOT a multiple of B_TOPK 64), ascending distinct
//       token positions, one out-of-range index, -1 tail padding, and the
//       SAME device buffer serving as both the top-k index row and
//       gather_rows, with topk_length == gather_count.
//
//   (C) NEGATIVE CONTROL for (B): flipping ONE FP8 byte of ONE selected row
//       must change the output — otherwise the memcmp in (B) is vacuous.
//       Restoring the byte restores bit-identity.
//
//   (D) RESCALE-GUARD REGRESSION. A top-k padded to 2112 (33 blocks) with
//       only ~100 valid indices leaves 31 fully-masked 64-row blocks — the
//       routine shape whenever context < index_topk. Before the guard, a
//       fully-masked block's row max came back as the masking constant and
//       `exp2f((old_max - MAX_INIT_VAL_MASK) * LOG2E)` evaluated to +inf,
//       poisoning sL and rO for every subsequent block: out/lse came back
//       inf/NaN. Now every value must be FINITE and equal to the CPU
//       reference over the valid rows, at nsp=34 (one block per split, so
//       whole splits are fully masked), nsp=4, and nsp=1 (no split) — with
//       the split arms agreeing closely with the no-split walk.
//
//   (E) DET-REDUCE BIT-IDENTITY. deterministic_reduce=true, the same
//       kernel+combine chain run twice with differently poisoned
//       o_accum/lse_accum/out/lse, must be byte-equal. Plus a negative
//       control that the comparison harness can see a difference at all.
//
//   (F) DET-REDUCE IS ORDER-ONLY. det vs legacy atomicAdd: max |delta|
//       measured and bounded, on both the no-split and the split path.
//
//   (G) h_q 32 — the tp=2 per-rank shape — drives the h_q < BLOCK_SIZE_M tail
//       of the same NUM_HEADS=64 instantiation, full pipeline vs the CPU
//       reference at the NopeMlaDecode tolerance.
//
// Reference values are computed in double precision from the EXACT stored
// bytes (read-back FP8 cache payload + stored scale + quantized Q), exactly as
// nope_mla_attention_test.cu does — the tolerance below is that suite's
// (out: 0.10 + 0.10*|ref|, lse: 5e-2 natural units).
//
// ── KNOWN RED: SnapmlaBudgetDecodeGather.GatherStagingSelectedRowsBitIdentical
// The engine wrapper layerstorm::compute::launch_dequant_ckv_indexed
// (src/compute/kernels/sm120/attention/snapmla_prep.cu:55-61) launches
// `dequant_ckv_fused_indexed_kernel<<<params.num_fetch, 128>>>` DIRECTLY — it
// never calls sm120::prep::run_dequant_ckv_fused_indexed, which is the only
// place that branches on gather_rows. So the gather kernel is unreachable from
// the engine: the production call site
// (src/compute/snapmla_sm120_attention_device.cpp:354-366) fills gather_rows /
// gather_count / max_gather / row_bound and then gets the full-range kernel
// over all seq_len_kv rows. Not a correctness fault (full staging is a superset
// of gather staging), but budget-bound staging is a silent NO-OP — the whole
// point of stage (a), killing the O(ctx) staging law, does not happen.
// The kernel itself is fine: the KernelEntry twin of the same test body, which
// calls run_dequant_ckv_fused_indexed, is GREEN.

#include "compute/kernels/attention/mla_attention.h"
#include "compute/kernels/attention/prep_params.h"
#include "compute/prefill_params.h"

#include "smxx/get_mla_metadata.h"
#include "smxx/mla_combine.h"
#include "sm120/decode/sparse_fp8/params.h"
#include "sm120/prep/dequant_ckv_indexed.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace lc = layerstorm::compute;

namespace {

#define ASSERT_CUDA(expr)                                                     \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        ASSERT_EQ(_e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_e);\
    } while (0)

// ── GLM5N geometry ──────────────────────────────────────────────────────────
constexpr int kDC = 512;                       // kv_lora_rank = d_nope = d_v
constexpr int kDR = 0;                         // NoPE: no rope block
constexpr int kDQK = kDC + kDR;                // 512
constexpr int kRowBytes = kDC + 4 + kDR * 2;   // 516 = [512 fp8 | f32 scale]
constexpr int kPage = 64;
constexpr int kTopkBlock = 64;                 // T::TOPK_BLOCK_SIZE / B_TOPK

static_assert(kRowBytes == 516, "GLM5N SnapMLA cache row (no rope tail)");

// NopeMlaDecode tolerance, verbatim.
constexpr double kOutAbs = 0.10, kOutRel = 0.10;
constexpr float kLseAbs = 5e-2f;

// Device allocations owned for the lifetime of a test.
struct Arena {
    std::vector<void*> ptrs;
    void* alloc(size_t bytes) {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        ptrs.push_back(p);
        return p;
    }
    template <typename T>
    T* allocT(size_t count) {
        return static_cast<T*>(alloc(count * sizeof(T)));
    }
    ~Arena() {
        for (void* p : ptrs) cudaFree(p);
    }
    Arena() = default;
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
};

// Host FP8 e4m3 decode (bit-exact) — same table as nope_mla_attention_test.
float fp8_e4m3_to_float(uint8_t b) {
    const int sign = (b >> 7) & 1;
    const int exp = (b >> 3) & 0xF;
    const int man = b & 0x7;
    float v;
    if (exp == 0) {
        v = std::ldexp(static_cast<float>(man) / 8.0f, -6);
    } else if (exp == 0xF && man == 0x7) {
        v = std::nanf("");
    } else {
        v = std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
    }
    return sign ? -v : v;
}

float bf16r(float f) { return __bfloat162float(__float2bfloat16(f)); }

// One bf16 ULP at |v|. bf16 carries an 8-bit significand (1 implicit + 7
// explicit), so for a in [2^k, 2^(k+1)) the last place is 2^(k-7); frexp
// returns e = k+1, hence 2^(e-8).
double bf16_ulp(double v) {
    const double a = std::fabs(v);
    if (!(a > 0.0)) return std::ldexp(1.0, -133);
    int e = 0;
    std::frexp(a, &e);
    return std::ldexp(1.0, e - 8);
}

//==============================================================================
// Shared GLM5N rig: paged FP8 cache built by the PRODUCTION prep kernels
//==============================================================================

// The cache is built through launch_fused_k_append (d_rope 0, k_rope nullptr)
// onto a SHUFFLED slot mapping — position p lives at cache slot pos2slot[p],
// which is exactly the linearized position->slot table the engine feeds the
// gather dequant as `indices`. Q is quantized through launch_fused_q_quant.
// Every stored value is read back so the CPU reference sees the exact bytes.
struct Rig {
    Arena arena;

    int num_tokens = 0;      // logical token POSITIONS
    int num_slots = 0;       // cache slots (>= num_tokens; the tail is unused)
    int num_pages = 0;
    int h_q = 0;

    void* d_cache = nullptr;
    int* d_pos2slot = nullptr;          // [num_tokens] position -> cache slot
    __nv_fp8_e4m3* d_q_nope = nullptr;  // [h_q, 512]
    float* d_q_scales = nullptr;        // [h_q]

    std::vector<int> pos2slot;
    std::vector<float> rows_fp8;   // [num_tokens * 512] raw fp8-decoded (unscaled)
    std::vector<float> k_scale;    // [num_tokens]
    std::vector<float> q_nope_dec; // [h_q * 512] raw fp8-decoded
    std::vector<float> q_scales;   // [h_q]

    int64_t cache_bytes() const {
        return static_cast<int64_t>(num_pages) * kPage * kRowBytes;
    }
    int64_t slot_byte_offset(int slot) const {
        return static_cast<int64_t>(slot / kPage) * kPage * kRowBytes +
               static_cast<int64_t>(slot % kPage) * kRowBytes;
    }
};

// plant_pos >= 0 aligns that POSITION's key with the mean query direction (a
// logit far above the noise floor), so its softmax weight is O(1) for every
// head — the discriminator for split weighting and the lever that makes the
// one-byte negative control (C) visible in bf16 output.
void build_rig(Rig& rig, int num_tokens, int h_q, unsigned seed,
               int plant_pos = -1) {
    rig.num_tokens = num_tokens;
    rig.h_q = h_q;
    rig.num_pages = (num_tokens + kPage - 1) / kPage;
    rig.num_slots = rig.num_pages * kPage;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> qf(static_cast<size_t>(h_q) * kDQK);
    std::vector<float> ckv(static_cast<size_t>(num_tokens) * kDC);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : ckv) v = bf16r(dist(rng));

    if (plant_pos >= 0) {
        constexpr float kPlantGain = 110.0f;
        for (int d = 0; d < kDC; ++d) {
            float acc = 0.f;
            for (int h = 0; h < h_q; ++h)
                acc += qf[static_cast<size_t>(h) * kDQK + d];
            ckv[static_cast<size_t>(plant_pos) * kDC + d] =
                bf16r(acc / h_q * kPlantGain);
        }
    }

    // Nontrivial paged mapping: position p -> a shuffled slot.
    rig.pos2slot.resize(num_tokens);
    {
        std::vector<int> all(rig.num_slots);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        for (int p = 0; p < num_tokens; ++p) rig.pos2slot[p] = all[p];
    }

    // ── K append (production prep kernel, NoPE) ──
    std::vector<__nv_bfloat16> ckv_b(ckv.size());
    for (size_t i = 0; i < ckv.size(); ++i) ckv_b[i] = __float2bfloat16(ckv[i]);

    rig.d_cache = rig.arena.alloc(rig.cache_bytes());
    ASSERT_NE(rig.d_cache, nullptr) << "cache alloc failed";
    ASSERT_CUDA(cudaMemset(rig.d_cache, 0, rig.cache_bytes()));

    Arena scratch;
    auto* d_ckv = scratch.allocT<__nv_bfloat16>(ckv_b.size());
    auto* d_slots = scratch.allocT<int>(num_tokens);
    ASSERT_NE(d_ckv, nullptr);
    ASSERT_NE(d_slots, nullptr);
    ASSERT_CUDA(cudaMemcpy(d_ckv, ckv_b.data(), ckv_b.size() * 2,
                           cudaMemcpyHostToDevice));
    ASSERT_CUDA(cudaMemcpy(d_slots, rig.pos2slot.data(),
                           num_tokens * sizeof(int), cudaMemcpyHostToDevice));

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
    ASSERT_CUDA(cudaDeviceSynchronize());

    // ── Q quantization (production prep kernel) ──
    std::vector<__nv_bfloat16> qb(qf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    auto* d_q = scratch.allocT<__nv_bfloat16>(qb.size());
    auto* d_rope_canary = scratch.allocT<__nv_bfloat16>(
        static_cast<size_t>(h_q) * 64);
    ASSERT_NE(d_q, nullptr);
    ASSERT_NE(d_rope_canary, nullptr);
    ASSERT_CUDA(cudaMemcpy(d_q, qb.data(), qb.size() * 2,
                           cudaMemcpyHostToDevice));
    ASSERT_CUDA(cudaMemset(d_rope_canary, 0xA5,
                           static_cast<size_t>(h_q) * 64 * 2));

    rig.d_q_nope = static_cast<__nv_fp8_e4m3*>(
        rig.arena.alloc(static_cast<size_t>(h_q) * kDC));
    rig.d_q_scales = rig.arena.allocT<float>(h_q);
    ASSERT_NE(rig.d_q_nope, nullptr);
    ASSERT_NE(rig.d_q_scales, nullptr);

    sm120::prep::FusedQQuantParams qp{};
    qp.q_bf16 = d_q;
    qp.q_nope_fp8 = rig.d_q_nope;
    qp.q_rope_bf16 = d_rope_canary;
    qp.q_scales = rig.d_q_scales;
    qp.s_q = 1;
    qp.h_q = h_q;
    qp.d_qk = kDQK;
    qp.d_nope = kDC;
    lc::launch_fused_q_quant(qp, nullptr);
    ASSERT_CUDA(cudaDeviceSynchronize());

    // ── Read back the EXACT stored values ──
    std::vector<uint8_t> cache_h(rig.cache_bytes());
    ASSERT_CUDA(cudaMemcpy(cache_h.data(), rig.d_cache, rig.cache_bytes(),
                           cudaMemcpyDeviceToHost));
    rig.rows_fp8.resize(static_cast<size_t>(num_tokens) * kDC);
    rig.k_scale.resize(num_tokens);
    for (int p = 0; p < num_tokens; ++p) {
        const uint8_t* row = cache_h.data() + rig.slot_byte_offset(rig.pos2slot[p]);
        for (int d = 0; d < kDC; ++d)
            rig.rows_fp8[static_cast<size_t>(p) * kDC + d] =
                fp8_e4m3_to_float(row[d]);
        std::memcpy(&rig.k_scale[p], row + kDC, 4);
    }

    std::vector<uint8_t> qn(static_cast<size_t>(h_q) * kDC);
    rig.q_scales.resize(h_q);
    ASSERT_CUDA(cudaMemcpy(qn.data(), rig.d_q_nope, qn.size(),
                           cudaMemcpyDeviceToHost));
    ASSERT_CUDA(cudaMemcpy(rig.q_scales.data(), rig.d_q_scales,
                           h_q * sizeof(float), cudaMemcpyDeviceToHost));
    rig.q_nope_dec.resize(qn.size());
    for (size_t i = 0; i < qn.size(); ++i)
        rig.q_nope_dec[i] = fp8_e4m3_to_float(qn[i]);

    rig.d_pos2slot = rig.arena.allocT<int>(num_tokens);
    ASSERT_NE(rig.d_pos2slot, nullptr);
    ASSERT_CUDA(cudaMemcpy(rig.d_pos2slot, rig.pos2slot.data(),
                           num_tokens * sizeof(int), cudaMemcpyHostToDevice));
}

// CPU reference over EXACTLY the given token positions (double precision, no
// rope term):
//   score(h,t) = [sum_d qn(h,d)*kn(t,d)] * q_scale(h) * k_scale(t) * sm_scale
//   out(h,d)   = softmax-weighted sum of kn(t,d)*k_scale(t)
void cpu_reference(const Rig& rig, const std::vector<int>& sel, float sm_scale,
                   std::vector<float>& out, std::vector<float>& lse) {
    const int H = rig.h_q;
    out.assign(static_cast<size_t>(H) * kDC, 0.0f);
    lse.assign(H, 0.0f);
    std::vector<double> sc(sel.size());
    for (int h = 0; h < H; ++h) {
        const float* qn = rig.q_nope_dec.data() + static_cast<size_t>(h) * kDC;
        double m = -1e300;
        for (size_t i = 0; i < sel.size(); ++i) {
            const float* kn =
                rig.rows_fp8.data() + static_cast<size_t>(sel[i]) * kDC;
            double dot = 0.0;
            for (int d = 0; d < kDC; ++d)
                dot += static_cast<double>(qn[d]) * kn[d];
            sc[i] = dot * rig.q_scales[h] * rig.k_scale[sel[i]] * sm_scale;
            m = std::max(m, sc[i]);
        }
        double l = 0.0;
        for (auto& s : sc) { s = std::exp(s - m); l += s; }
        lse[h] = static_cast<float>(m + std::log(l));
        float* orow = out.data() + static_cast<size_t>(h) * kDC;
        for (size_t i = 0; i < sel.size(); ++i) {
            const float* kn =
                rig.rows_fp8.data() + static_cast<size_t>(sel[i]) * kDC;
            const double p = sc[i] / l;
            const float ks = rig.k_scale[sel[i]];
            for (int d = 0; d < kDC; ++d)
                orow[d] += static_cast<float>(p * kn[d] * ks);
        }
    }
}

// Sorted, distinct token positions in [0, n).
std::vector<int> pick_positions(int count, int n, std::mt19937& rng) {
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(count);
    std::sort(all.begin(), all.end());
    return all;
}

//==============================================================================
// Decode driver (sparse FP8 split-KV + mla_combine)
//==============================================================================

struct DecodeIo {
    Arena arena;
    __nv_bfloat16* d_out = nullptr;
    float* d_lse = nullptr;
    float* d_lse_accum = nullptr;
    float* d_o_accum = nullptr;
    int* d_sched_meta = nullptr;
    int* d_num_splits = nullptr;
    int h_q = 0, max_parts = 0;
};

void build_decode_io(DecodeIo& io, int h_q, int max_parts) {
    io.h_q = h_q;
    io.max_parts = max_parts;
    const int max_splits = max_parts + 2;
    io.d_out = static_cast<__nv_bfloat16*>(
        io.arena.alloc(static_cast<size_t>(h_q) * kDC * 2));
    io.d_lse = io.arena.allocT<float>(h_q);
    io.d_lse_accum = io.arena.allocT<float>(
        static_cast<size_t>(max_splits) * h_q);
    io.d_o_accum = io.arena.allocT<float>(
        static_cast<size_t>(max_splits) * h_q * kDC);
    io.d_sched_meta = io.arena.allocT<int>(static_cast<size_t>(max_parts) * 8);
    io.d_num_splits = io.arena.allocT<int>(2);
    ASSERT_NE(io.d_out, nullptr);
    ASSERT_NE(io.d_lse, nullptr);
    ASSERT_NE(io.d_lse_accum, nullptr);
    ASSERT_NE(io.d_o_accum, nullptr);
    ASSERT_NE(io.d_sched_meta, nullptr);
    ASSERT_NE(io.d_num_splits, nullptr);
}

// topk mode (seqlens_k_ptr nullptr, block_size_n 64, fixed_overhead 1).
int fill_metadata(DecodeIo& io, int topk, int num_sm_parts) {
    GetMlaMetadataParams mp{};
    mp.seqlens_k_ptr = nullptr;
    mp.tile_scheduler_metadata_ptr = io.d_sched_meta;
    mp.num_splits_ptr = io.d_num_splits;
    mp.batch_size = 1;
    mp.block_size_n = kTopkBlock;
    mp.fixed_overhead_num_blocks = 1;
    mp.num_sm_parts = num_sm_parts;
    mp.topk = topk;
    lc::launch_get_mla_metadata(mp, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    int ns[2] = {-1, -1};
    EXPECT_EQ(cudaMemcpy(ns, io.d_num_splits, 2 * sizeof(int),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(ns[0], 0);
    return ns[1];
}

struct DecodeResult {
    std::vector<uint16_t> out;  // raw bf16 bits [h_q, 512]
    std::vector<float> lse;     // [h_q]
    int num_splits = 0;
};

// One full decode: metadata -> sparse FP8 split-KV kernel -> (combine if
// split). `poison` seeds the byte pattern written into every output and
// accumulator buffer beforehand, so "the kernel wrote nothing" can never pass
// a bit-equality check.
DecodeResult run_decode(const Rig& rig, DecodeIo& io, const int* d_indices,
                        int topk, int indices_stride, int num_sm_parts,
                        bool deterministic, int poison) {
    DecodeResult r;
    const int h_q = rig.h_q;
    const int max_splits = io.max_parts + 2;

    EXPECT_EQ(cudaMemset(io.d_out, poison,
                         static_cast<size_t>(h_q) * kDC * 2), cudaSuccess);
    EXPECT_EQ(cudaMemset(io.d_lse, poison, h_q * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMemset(io.d_lse_accum, poison,
                         static_cast<size_t>(max_splits) * h_q * sizeof(float)),
              cudaSuccess);
    EXPECT_EQ(cudaMemset(io.d_o_accum, poison,
                         static_cast<size_t>(max_splits) * h_q * kDC *
                             sizeof(float)), cudaSuccess);

    r.num_splits = fill_metadata(io, topk, num_sm_parts);

    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    sm120::decode::sparse_fp8::SparseAttnDecodeParams p{};
    std::memset(&p, 0, sizeof(p));
    p.b = 1; p.s_q = 1; p.h_q = h_q; p.h_kv = 1;
    p.d_qk = kDQK; p.d_v = kDC;
    p.sm_scale = sm_scale;
    p.sm_scale_div_log2 = sm_scale / std::log(2.0f);
    p.num_blocks = 0;
    p.page_block_size = kPage;
    p.topk = topk;
    p.model_type = sm120::sparse::ModelType::GLM5N;
    p.q = reinterpret_cast<cutlass::bfloat16_t*>(rig.d_q_nope);
    p.q_rope = nullptr;
    p.q_scales = rig.d_q_scales;
    p.kv = static_cast<cutlass::bfloat16_t*>(rig.d_cache);
    p.indices = const_cast<int*>(d_indices);
    p.topk_length = nullptr;   // GLM5N drives a FLAT top-k of exactly p.topk
    p.attn_sink = nullptr;
    p.lse = io.d_lse;
    p.out = reinterpret_cast<cutlass::bfloat16_t*>(io.d_out);
    p.stride_q_b = h_q * kDC;
    p.stride_q_s_q = h_q * kDC;
    p.stride_q_h_q = kDC;
    p.stride_kv_block = kPage * kRowBytes;
    p.stride_kv_row = kRowBytes;
    p.stride_indices_b = indices_stride;
    p.stride_indices_s_q = indices_stride;
    p.stride_lse_b = h_q;
    p.stride_lse_s_q = h_q;
    p.stride_o_b = h_q * kDC;
    p.stride_o_s_q = h_q * kDC;
    p.stride_o_h_q = kDC;
    p.lse_accum = io.d_lse_accum;
    p.o_accum = io.d_o_accum;
    p.stride_lse_accum_split = h_q;
    p.stride_lse_accum_s_q = h_q;
    p.stride_o_accum_split = h_q * kDC;
    p.stride_o_accum_s_q = h_q * kDC;
    p.stride_o_accum_h_q = kDC;
    p.tile_scheduler_metadata_ptr =
        reinterpret_cast<sm120::decode::sparse_fp8::DecodingSchedMeta*>(
            io.d_sched_meta);
    p.num_splits_ptr = io.d_num_splits;
    p.num_sm_parts = num_sm_parts;
    p.deterministic_reduce = deterministic;
    p.stream = nullptr;

    lc::launch_decode_sparse_fp8(p);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    if (r.num_splits > 1) {
        MlaCombineParams c{};
        std::memset(&c, 0, sizeof(c));
        c.b = 1;
        c.h_q = h_q;
        c.h_k = 1;
        c.q_seq_per_hk = h_q;
        c.d_v = kDC;
        c.o_ptr = io.d_out;
        c.softmax_lse_ptr = io.d_lse;
        c.o_batch_stride = h_q * kDC;
        c.o_head_stride = kDC;
        c.o_row_stride = kDC;
        c.num_splits_ptr = io.d_num_splits;
        c.num_sm_parts = num_sm_parts;
        c.softmax_lseaccum_ptr = io.d_lse_accum;
        c.oaccum_ptr = io.d_o_accum;
        lc::launch_mla_combine(c, nullptr);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    r.out.resize(static_cast<size_t>(h_q) * kDC);
    r.lse.resize(h_q);
    EXPECT_EQ(cudaMemcpy(r.out.data(), io.d_out, r.out.size() * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.lse.data(), io.d_lse, h_q * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    return r;
}

float bits_to_float(uint16_t bits) {
    __nv_bfloat16 b;
    std::memcpy(&b, &bits, 2);
    return __bfloat162float(b);
}

void check_vs_reference(const DecodeResult& run, const std::vector<float>& ref_out,
                        const std::vector<float>& ref_lse, const char* tag) {
    ASSERT_EQ(run.out.size(), ref_out.size());
    ASSERT_EQ(run.lse.size(), ref_lse.size());
    for (size_t h = 0; h < run.lse.size(); ++h) {
        ASSERT_TRUE(std::isfinite(run.lse[h]))
            << tag << " non-finite lse[" << h << "] = " << run.lse[h];
        EXPECT_NEAR(run.lse[h], ref_lse[h], kLseAbs) << tag << " lse[" << h << "]";
    }
    for (size_t i = 0; i < run.out.size(); ++i) {
        const double got = bits_to_float(run.out[i]);
        const double want = ref_out[i];
        ASSERT_TRUE(std::isfinite(got))
            << tag << " non-finite out[" << i << "] = " << got;
        ASSERT_NEAR(got, want, kOutAbs + kOutRel * std::fabs(want))
            << tag << " out[" << i << "]";
    }
}

// Max |delta| between two decode results, in bf16-decoded space for out.
struct Delta { double out = 0.0; double lse = 0.0; int out_ulp = 0; };

Delta max_delta(const DecodeResult& a, const DecodeResult& b) {
    Delta d;
    for (size_t i = 0; i < a.out.size(); ++i) {
        const double fa = bits_to_float(a.out[i]), fb = bits_to_float(b.out[i]);
        d.out = std::max(d.out, std::fabs(fa - fb));
        if (a.out[i] != b.out[i]) {
            // bf16 codes are monotonic within a sign; report the raw code gap
            // for same-sign values, else just flag it as large.
            const int gap = std::abs(static_cast<int>(a.out[i]) -
                                     static_cast<int>(b.out[i]));
            d.out_ulp = std::max(d.out_ulp, gap);
        }
    }
    for (size_t i = 0; i < a.lse.size(); ++i)
        d.lse = std::max(d.lse, static_cast<double>(std::fabs(a.lse[i] - b.lse[i])));
    return d;
}

bool same_bits(const DecodeResult& a, const DecodeResult& b) {
    return a.out.size() == b.out.size() && a.lse.size() == b.lse.size() &&
           std::memcmp(a.out.data(), b.out.data(), a.out.size() * 2) == 0 &&
           std::memcmp(a.lse.data(), b.lse.data(),
                       a.lse.size() * sizeof(float)) == 0;
}

// Overwrite one FP8 payload byte of the cache row holding token `pos`.
uint8_t poke_cache_byte(const Rig& rig, int pos, int dim, uint8_t value) {
    const int64_t off = rig.slot_byte_offset(rig.pos2slot[pos]) + dim;
    uint8_t old = 0;
    EXPECT_EQ(cudaMemcpy(&old, static_cast<uint8_t*>(rig.d_cache) + off, 1,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(static_cast<uint8_t*>(rig.d_cache) + off, &value, 1,
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return old;
}

//==============================================================================
// Gather staging driver
//==============================================================================

// Fill the shared DequantCKVIndexedParams skeleton (position -> slot table +
// paged cache geometry). Gather fields are left null by default = legacy
// full-range behavior.
sm120::prep::DequantCKVIndexedParams stage_params(const Rig& rig,
                                                  __nv_bfloat16* k_out) {
    sm120::prep::DequantCKVIndexedParams dq{};
    dq.kv_cache = static_cast<const __nv_fp8_e4m3*>(rig.d_cache);
    dq.cache_stride_block = static_cast<int64_t>(kPage) * kRowBytes;
    dq.cache_stride_row = kRowBytes;
    dq.page_size = kPage;
    dq.indices = rig.d_pos2slot;
    dq.num_fetch = rig.num_tokens;
    dq.k_out = k_out;
    dq.d_c = kDC;
    dq.d_rope = kDR;
    return dq;
}

// The two entry points into the gather launcher. The engine wrapper
// (layerstorm::compute::launch_dequant_ckv_indexed) is the PRODUCTION path —
// snapmla_sm120_attention_device.cpp:366 calls it with the gather fields set.
// The kernel-library entry (sm120::prep::run_dequant_ckv_fused_indexed) is the
// launcher the gather kernel actually lives behind. Running (A) through BOTH
// attributes any failure to exactly one of them.
enum class DequantEntry { kEngineWrapper, kKernelLauncher };

void dispatch_dequant(const sm120::prep::DequantCKVIndexedParams& p,
                      DequantEntry entry) {
    if (entry == DequantEntry::kEngineWrapper)
        lc::launch_dequant_ckv_indexed(p, nullptr);
    else
        sm120::prep::run_dequant_ckv_fused_indexed(p, nullptr);
}

}  // namespace

//==============================================================================
// (A) Gather staging writes exactly the selected rows, bit-identically
//==============================================================================

namespace {

void gather_selected_rows_body(DequantEntry entry, unsigned seed) {
    constexpr int kTokens = 2048;
    constexpr int kSelected = 200;
    constexpr int kPad = 32;          // entries PAST *gather_count

    Rig rig;
    build_rig(rig, kTokens, /*h_q=*/64, seed);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    std::mt19937 rng(seed ^ 0x11u);
    // 200 selected positions + a few -1 / out-of-range entries INSIDE the
    // counted range, then padding entries (real positions!) PAST the count
    // that must stay unwritten.
    std::vector<int> picked = pick_positions(kSelected + kPad, kTokens, rng);
    std::vector<int> selected(picked.begin(), picked.begin() + kSelected);
    std::vector<int> beyond_count(picked.begin() + kSelected, picked.end());

    std::vector<int> gather(selected);
    gather.push_back(-1);
    gather.push_back(-1);
    gather.push_back(kTokens);          // out of range (== row_bound)
    gather.push_back(kTokens + 7);      // out of range
    gather.push_back(-1);
    const int gather_count = static_cast<int>(gather.size());
    gather.insert(gather.end(), beyond_count.begin(), beyond_count.end());
    const int max_gather = static_cast<int>(gather.size());

    Arena a;
    const size_t stage_rows = kTokens;
    const size_t stage_bytes = stage_rows * kDQK * 2;
    auto* d_stage_full = a.allocT<__nv_bfloat16>(stage_rows * kDQK);
    auto* d_stage_gather = a.allocT<__nv_bfloat16>(stage_rows * kDQK);
    auto* d_gather_rows = a.allocT<int>(gather.size());
    auto* d_gather_count = a.allocT<int>(1);
    ASSERT_NE(d_stage_full, nullptr);
    ASSERT_NE(d_stage_gather, nullptr);
    ASSERT_NE(d_gather_rows, nullptr);
    ASSERT_NE(d_gather_count, nullptr);
    ASSERT_CUDA(cudaMemcpy(d_gather_rows, gather.data(),
                           gather.size() * sizeof(int), cudaMemcpyHostToDevice));
    ASSERT_CUDA(cudaMemcpy(d_gather_count, &gather_count, sizeof(int),
                           cudaMemcpyHostToDevice));

    // Arm 1: legacy full-range dequant into staging A.
    ASSERT_CUDA(cudaMemset(d_stage_full, 0x5A, stage_bytes));
    sm120::prep::DequantCKVIndexedParams full = stage_params(rig, d_stage_full);
    dispatch_dequant(full, entry);
    ASSERT_CUDA(cudaDeviceSynchronize());

    // Arm 2: gather dequant into 0xA5-poisoned staging B.
    ASSERT_CUDA(cudaMemset(d_stage_gather, 0xA5, stage_bytes));
    // NOTE: num_fetch stays at seq_len_kv, exactly as the production call site
    // leaves it (snapmla_sm120_attention_device.cpp:340) — gather mode must
    // IGNORE it. If the launcher falls through to the full-range kernel, the
    // poison assertions below fire.
    sm120::prep::DequantCKVIndexedParams gp = stage_params(rig, d_stage_gather);
    gp.gather_rows = d_gather_rows;
    gp.gather_count = d_gather_count;
    gp.max_gather = max_gather;
    gp.row_bound = kTokens;
    dispatch_dequant(gp, entry);
    ASSERT_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> hfull(stage_bytes), hgather(stage_bytes);
    ASSERT_CUDA(cudaMemcpy(hfull.data(), d_stage_full, stage_bytes,
                           cudaMemcpyDeviceToHost));
    ASSERT_CUDA(cudaMemcpy(hgather.data(), d_stage_gather, stage_bytes,
                           cudaMemcpyDeviceToHost));

    std::vector<bool> is_selected(kTokens, false);
    for (int p : selected) is_selected[p] = true;

    const size_t row_bytes = static_cast<size_t>(kDQK) * 2;
    int checked_sel = 0, checked_poison = 0;
    for (int p = 0; p < kTokens; ++p) {
        const uint8_t* fa = hfull.data() + static_cast<size_t>(p) * row_bytes;
        const uint8_t* fb = hgather.data() + static_cast<size_t>(p) * row_bytes;
        if (is_selected[p]) {
            ASSERT_EQ(std::memcmp(fa, fb, row_bytes), 0)
                << "selected row " << p << " differs between full-range and "
                << "gather staging (dequant_ckv_row must be shared verbatim)";
            ++checked_sel;
        } else {
            for (size_t b = 0; b < row_bytes; ++b)
                ASSERT_EQ(fb[b], 0xA5)
                    << "gather kernel wrote unselected row " << p << " byte " << b;
            ++checked_poison;
        }
    }
    EXPECT_EQ(checked_sel, kSelected);
    EXPECT_EQ(checked_poison, kTokens - kSelected);

    // The full-range arm really did populate everything (so the memcmp above
    // was against real data, not poison-vs-poison).
    for (int p : selected) {
        const uint8_t* fa = hfull.data() + static_cast<size_t>(p) * row_bytes;
        bool all_poison = true;
        for (size_t b = 0; b < row_bytes && all_poison; ++b)
            if (fa[b] != 0x5A) all_poison = false;
        ASSERT_FALSE(all_poison) << "full-range staging row " << p
                                 << " was never written";
    }
}

}  // namespace

// Kernel-library launcher: run_dequant_ckv_fused_indexed branches on
// gather_rows. This is the leg that pins the GATHER KERNEL + its predicate.
TEST(SnapmlaBudgetDecodeGather, GatherStagingSelectedRowsBitIdenticalKernelEntry) {
    REQUIRES_GPU();
    gather_selected_rows_body(DequantEntry::kKernelLauncher, 0x5A170001u);
}

// PRODUCTION path. KNOWN RED — see the TD note at the top of this file:
// layerstorm::compute::launch_dequant_ckv_indexed
// (src/compute/kernels/sm120/attention/snapmla_prep.cu:55) never calls
// run_dequant_ckv_fused_indexed, so it ignores gather_rows entirely and
// budget-bound staging silently degrades to full-context staging.
TEST(SnapmlaBudgetDecodeGather, GatherStagingSelectedRowsBitIdentical) {
    REQUIRES_GPU();
    gather_selected_rows_body(DequantEntry::kEngineWrapper, 0x5A170001u);
}

//==============================================================================
// (B) Read-set proof through the sparse prefill consumer
//==============================================================================

namespace {

// Production index shape: the SAME device buffer is the top-k index row and
// gather_rows; topk_length == gather_count.
struct BudgetIndexSet {
    std::vector<int> host;       // [topk_alloc] positions, -1 padded
    std::vector<int> valid;      // sorted distinct in-range positions
    int count = 0;               // topk_length / gather_count
    int topk = 0;                // logical topk fed to the kernel
    int topk_alloc = 0;          // ceil(topk / 64) * 64 ints allocated
};

BudgetIndexSet make_budget_indices(int n_valid, int num_tokens, int topk,
                                   std::mt19937& rng) {
    BudgetIndexSet s;
    s.topk = topk;
    s.topk_alloc = ((topk + kTopkBlock - 1) / kTopkBlock) * kTopkBlock;
    s.valid = pick_positions(n_valid, num_tokens, rng);
    s.host.assign(s.topk_alloc, -1);
    for (int i = 0; i < n_valid; ++i) s.host[i] = s.valid[i];
    // One out-of-range index, ascending-last: masked by the consumer
    // (idx < s_kv) and skipped by the gather (row < row_bound).
    s.host[n_valid] = num_tokens + 7;
    s.count = n_valid + 1;
    return s;
}

struct PrefillRun {
    std::vector<uint16_t> out;
    std::vector<float> lse;
};

// One sparse prefill call (s_q = 1, deterministic_reduce = true) over `d_kv`.
PrefillRun run_sparse_prefill(int h_q, __nv_bfloat16* d_kv, int s_kv,
                              const __nv_bfloat16* d_q, const int* d_indices,
                              const int* d_topk_length, int topk) {
    Arena a;
    const size_t out_elems = static_cast<size_t>(h_q) * kDC;
    auto* d_out = a.allocT<__nv_bfloat16>(out_elems);
    auto* d_lse = a.allocT<float>(h_q);
    EXPECT_NE(d_out, nullptr);
    EXPECT_NE(d_lse, nullptr);
    EXPECT_EQ(cudaMemset(d_out, 0xA5, out_elems * 2), cudaSuccess);
    EXPECT_EQ(cudaMemset(d_lse, 0xA5, h_q * sizeof(float)), cudaSuccess);

    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    const lc::PrefillDims dims{kDC, kDR, h_q, num_sm, 0.0f};

    SparseAttnFwdParams p{};
    lc::populate_sparse_prefill_params(p, dims, d_q, d_kv, d_indices,
                                       d_topk_length, topk, /*batch_size=*/1,
                                       /*seq_len_kv=*/s_kv, d_out, d_lse,
                                       nullptr);
    EXPECT_EQ(p.d_qk, kDQK);
    EXPECT_EQ(p.d_v, kDC);
    p.deterministic_reduce = true;
    lc::launch_prefill_sparse(p);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    PrefillRun r;
    r.out.resize(out_elems);
    r.lse.resize(h_q);
    EXPECT_EQ(cudaMemcpy(r.out.data(), d_out, out_elems * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.lse.data(), d_lse, h_q * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    return r;
}

bool prefill_same_bits(const PrefillRun& a, const PrefillRun& b) {
    return std::memcmp(a.out.data(), b.out.data(), a.out.size() * 2) == 0 &&
           std::memcmp(a.lse.data(), b.lse.data(),
                       a.lse.size() * sizeof(float)) == 0;
}

// Shared (B)/(C) fixture body: builds the rig, the shared index buffer, both
// staging arms and the BF16 query.
struct BudgetStagingCase {
    Rig rig;
    Arena a;
    BudgetIndexSet idx;
    __nv_bfloat16* d_stage_full = nullptr;
    __nv_bfloat16* d_stage_gather = nullptr;
    __nv_bfloat16* d_q = nullptr;
    int* d_indices = nullptr;
    int* d_count = nullptr;
    int h_q = 64;
    int tokens = 0;
    size_t stage_bytes = 0;
};

void build_staging_case(BudgetStagingCase& c, int tokens, int n_valid,
                        int topk, unsigned seed, int plant_pos_hint) {
    c.tokens = tokens;
    std::mt19937 rng(seed ^ 0x9E37u);
    c.idx = make_budget_indices(n_valid, tokens, topk, rng);
    // Plant the dominant token ON a selected position (the (C) lever).
    const int plant_pos = c.idx.valid[plant_pos_hint % n_valid];
    build_rig(c.rig, tokens, c.h_q, seed, plant_pos);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    // Staging has 64 spare rows so an out-of-range index can never fault even
    // if a future kernel dropped its mask.
    const size_t stage_rows = static_cast<size_t>(tokens) + 64;
    c.stage_bytes = stage_rows * kDQK * 2;
    c.d_stage_full = c.a.allocT<__nv_bfloat16>(stage_rows * kDQK);
    c.d_stage_gather = c.a.allocT<__nv_bfloat16>(stage_rows * kDQK);
    c.d_q = c.a.allocT<__nv_bfloat16>(static_cast<size_t>(c.h_q) * kDQK);
    c.d_indices = c.a.allocT<int>(c.idx.topk_alloc);
    c.d_count = c.a.allocT<int>(1);
    ASSERT_NE(c.d_stage_full, nullptr);
    ASSERT_NE(c.d_stage_gather, nullptr);
    ASSERT_NE(c.d_q, nullptr);
    ASSERT_NE(c.d_indices, nullptr);
    ASSERT_NE(c.d_count, nullptr);

    std::vector<__nv_bfloat16> qh(static_cast<size_t>(c.h_q) * kDQK);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    for (auto& v : qh) v = __float2bfloat16(dist(rng));
    ASSERT_CUDA(cudaMemcpy(c.d_q, qh.data(), qh.size() * 2,
                           cudaMemcpyHostToDevice));
    ASSERT_CUDA(cudaMemcpy(c.d_indices, c.idx.host.data(),
                           c.idx.topk_alloc * sizeof(int),
                           cudaMemcpyHostToDevice));
    ASSERT_CUDA(cudaMemcpy(c.d_count, &c.idx.count, sizeof(int),
                           cudaMemcpyHostToDevice));
}

// Re-stage both arms from the CURRENT cache contents.
void restage(BudgetStagingCase& c) {
    ASSERT_CUDA(cudaMemset(c.d_stage_full, 0x5A, c.stage_bytes));
    sm120::prep::DequantCKVIndexedParams full =
        stage_params(c.rig, c.d_stage_full);
    lc::launch_dequant_ckv_indexed(full, nullptr);

    ASSERT_CUDA(cudaMemset(c.d_stage_gather, 0xA5, c.stage_bytes));
    sm120::prep::DequantCKVIndexedParams gp =
        stage_params(c.rig, c.d_stage_gather);
    gp.gather_rows = c.d_indices;      // SAME buffer as the top-k index row
    gp.gather_count = c.d_count;       // == topk_length
    gp.max_gather = c.idx.topk_alloc;
    gp.row_bound = c.tokens;
    // Kernel-library entry ON PURPOSE: the read-set proof needs staging that
    // really does have poison holes in it. Routing this through the engine
    // wrapper today would silently full-stage (see the TD note at the top of
    // this file) and (B)/(C) would degenerate into comparing full staging with
    // itself. Flip to dispatch_dequant(gp, DequantEntry::kEngineWrapper) once
    // snapmla_prep.cu:55 routes through run_dequant_ckv_fused_indexed.
    dispatch_dequant(gp, DequantEntry::kKernelLauncher);
    ASSERT_CUDA(cudaDeviceSynchronize());
}

}  // namespace

TEST(SnapmlaBudgetDecodeGather, GatherStagingSparseKernelOutputBitIdentical) {
    REQUIRES_GPU();

    BudgetStagingCase c;
    // topk 2051 is the production shape: NOT a multiple of B_TOPK 64, so the
    // last index block is partially -1 padding.
    build_staging_case(c, /*tokens=*/2048, /*n_valid=*/200, /*topk=*/2051,
                       /*seed=*/0x5A17'0002u, /*plant_pos_hint=*/37);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_LT(c.idx.count, c.idx.topk) << "test needs tail -1 padding";

    restage(c);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    Arena a;
    auto* d_tkl = a.allocT<int>(1);
    ASSERT_NE(d_tkl, nullptr);
    ASSERT_CUDA(cudaMemcpy(d_tkl, &c.idx.count, sizeof(int),
                           cudaMemcpyHostToDevice));

    const PrefillRun ref = run_sparse_prefill(c.h_q, c.d_stage_full, c.tokens,
                                              c.d_q, c.d_indices, d_tkl,
                                              c.idx.topk);
    const PrefillRun got = run_sparse_prefill(c.h_q, c.d_stage_gather, c.tokens,
                                              c.d_q, c.d_indices, d_tkl,
                                              c.idx.topk);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    // Sanity: the run is not degenerate (all-poison output would compare equal
    // trivially).
    int finite_out = 0;
    for (uint16_t b : ref.out)
        if (std::isfinite(bits_to_float(b)) && bits_to_float(b) != 0.0f)
            ++finite_out;
    ASSERT_GT(finite_out, static_cast<int>(ref.out.size()) / 2)
        << "reference prefill output looks degenerate";
    for (float l : ref.lse) ASSERT_TRUE(std::isfinite(l)) << "reference lse " << l;

    EXPECT_EQ(std::memcmp(ref.out.data(), got.out.data(), ref.out.size() * 2), 0)
        << "sparse prefill output differs between full staging and gather "
        << "staging — the kernel read a row the gather did not write";
    EXPECT_EQ(std::memcmp(ref.lse.data(), got.lse.data(),
                          ref.lse.size() * sizeof(float)), 0)
        << "sparse prefill lse differs between full staging and gather staging";
}

//==============================================================================
// (C) Negative control — the (B) comparison is not vacuous
//==============================================================================

TEST(SnapmlaBudgetDecodeGather, GatherNegativeControl) {
    REQUIRES_GPU();

    BudgetStagingCase c;
    build_staging_case(c, /*tokens=*/2048, /*n_valid=*/200, /*topk=*/2051,
                       /*seed=*/0x5A17'0003u, /*plant_pos_hint=*/37);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    Arena a;
    auto* d_tkl = a.allocT<int>(1);
    ASSERT_NE(d_tkl, nullptr);
    ASSERT_CUDA(cudaMemcpy(d_tkl, &c.idx.count, sizeof(int),
                           cudaMemcpyHostToDevice));

    restage(c);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const PrefillRun base = run_sparse_prefill(c.h_q, c.d_stage_gather, c.tokens,
                                               c.d_q, c.d_indices, d_tkl,
                                               c.idx.topk);

    // Flip one FP8 byte of the DOMINANT selected row (the planted token, whose
    // softmax weight is O(1) for every head — a byte flipped in a negligibly
    // weighted row would be invisible in bf16 output and would make this
    // control meaningless).
    const int flip_pos = c.idx.valid[37 % 200];
    const uint8_t old = poke_cache_byte(c.rig, flip_pos, /*dim=*/0, 0x40);
    const uint8_t used = (old == 0x40) ? 0x50 : 0x40;
    if (used != 0x40) poke_cache_byte(c.rig, flip_pos, 0, used);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    restage(c);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const PrefillRun flipped = run_sparse_prefill(c.h_q, c.d_stage_gather,
                                                  c.tokens, c.d_q, c.d_indices,
                                                  d_tkl, c.idx.topk);
    EXPECT_FALSE(prefill_same_bits(base, flipped))
        << "flipping one FP8 byte of a selected row did not change the sparse "
        << "prefill output — the bit-identity check in "
        << "GatherStagingSparseKernelOutputBitIdentical is vacuous";

    // Restore and confirm the comparison snaps back to bit-identical.
    poke_cache_byte(c.rig, flip_pos, /*dim=*/0, old);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    restage(c);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const PrefillRun restored = run_sparse_prefill(c.h_q, c.d_stage_gather,
                                                   c.tokens, c.d_q, c.d_indices,
                                                   d_tkl, c.idx.topk);
    EXPECT_TRUE(prefill_same_bits(base, restored))
        << "restoring the flipped byte did not restore the output";
}

//==============================================================================
// (D) Rescale-guard regression: fully-masked top-k blocks stay finite
//==============================================================================

TEST(SnapmlaBudgetDecodeSparseFp8, Fp8SparseDecodeShortTopkLengthFinite) {
    REQUIRES_GPU();

    constexpr int kTokens = 2048;
    constexpr int kValid = 100;
    constexpr int kTopk = 2112;   // 33 blocks of 64 — 31 of them fully masked
    static_assert(kTopk % kTopkBlock == 0, "GLM5N drives a flat topk");

    Rig rig;
    // Plant a dominant token in the valid prefix: split weighting matters, and
    // a mis-weighted split shows up immediately.
    build_rig(rig, kTokens, /*h_q=*/64, /*seed=*/0x5A17'0004u,
              /*plant_pos=*/1234);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    std::mt19937 rng(0x5A17'0044u);
    std::vector<int> sel = pick_positions(kValid - 1, kTokens, rng);
    if (std::find(sel.begin(), sel.end(), 1234) == sel.end()) {
        sel.push_back(1234);
        std::sort(sel.begin(), sel.end());
    } else {
        sel = pick_positions(kValid, kTokens, rng);
    }
    const int n_valid = static_cast<int>(sel.size());

    // indices[g] = cache SLOT of position sel[g] for g < n_valid, -1 beyond.
    std::vector<int> host_idx(kTopk, -1);
    for (int g = 0; g < n_valid; ++g) host_idx[g] = rig.pos2slot[sel[g]];

    Arena a;
    auto* d_idx = a.allocT<int>(kTopk);
    ASSERT_NE(d_idx, nullptr);
    ASSERT_CUDA(cudaMemcpy(d_idx, host_idx.data(), kTopk * sizeof(int),
                           cudaMemcpyHostToDevice));

    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    std::vector<float> ref_out, ref_lse;
    cpu_reference(rig, sel, sm_scale, ref_out, ref_lse);

    DecodeIo io;
    build_decode_io(io, rig.h_q, /*max_parts=*/40);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    // nsp = 1: the no-split walk — every fully-masked block sits inside ONE
    // partition, so the rescale guard is exercised block-to-block.
    const DecodeResult nosplit =
        run_decode(rig, io, d_idx, kTopk, kTopk, /*num_sm_parts=*/1,
                   /*deterministic=*/true, /*poison=*/0xCC);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_EQ(nosplit.num_splits, 1) << "expected the no-split schedule";
    check_vs_reference(nosplit, ref_out, ref_lse, "short-topk nsp=1");

    // nsp = 34: one 64-row block per split, so 31 WHOLE splits are fully
    // masked (sL == 0 -> lse_accum -inf) and must not poison the combine.
    const DecodeResult split34 =
        run_decode(rig, io, d_idx, kTopk, kTopk, /*num_sm_parts=*/34,
                   /*deterministic=*/true, /*poison=*/0x3C);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_GT(split34.num_splits, 1) << "expected a genuine multi-split schedule";
    check_vs_reference(split34, ref_out, ref_lse, "short-topk nsp=34");

    // nsp = 4: fully-masked blocks INSIDE a split that also holds valid rows.
    const DecodeResult split4 =
        run_decode(rig, io, d_idx, kTopk, kTopk, /*num_sm_parts=*/4,
                   /*deterministic=*/true, /*poison=*/0x77);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_GT(split4.num_splits, 1) << "expected a genuine multi-split schedule";
    check_vs_reference(split4, ref_out, ref_lse, "short-topk nsp=4");

    // Split vs no-split: not bit-equal (the combine reorders the FP32 merge),
    // but they must agree to well within a bf16 step.
    double worst = 0.0, worst_ratio = 0.0;
    for (const auto* arm : {&split34, &split4}) {
        for (size_t i = 0; i < arm->out.size(); ++i) {
            const double got = bits_to_float(arm->out[i]);
            const double want = bits_to_float(nosplit.out[i]);
            const double d = std::fabs(got - want);
            worst = std::max(worst, d);
            // 1e-3 absolute, widened by one bf16 ULP: a single last-place
            // rounding step at |out| ~ 0.5 is already 2e-3, and the merge order
            // difference legitimately lands on rounding boundaries.
            const double tol = 1e-3 + bf16_ulp(want);
            worst_ratio = std::max(worst_ratio, d / tol);
            ASSERT_LE(d, tol) << "split vs no-split out[" << i << "] got "
                              << got << " want " << want;
        }
    }
    RecordProperty("split_vs_nosplit_max_abs", std::to_string(worst));
    RecordProperty("split_vs_nosplit_max_tol_ratio", std::to_string(worst_ratio));
    // Observed: max_abs is exactly ONE bf16 code (3.125e-02 at |out| ~ 4.7),
    // i.e. the split merge order only ever moves the last place.
    std::printf("[ D ] short-topk split-vs-nosplit out max_abs = %.3e "
                "(%.3f x tolerance)\n", worst, worst_ratio);
}

//==============================================================================
// (E) DET-REDUCE bit-identity
//==============================================================================

TEST(SnapmlaBudgetDecodeSparseFp8, Fp8SparseDecodeDeterministicBitIdentical) {
    REQUIRES_GPU();

    constexpr int kTokens = 2048;
    constexpr int kTopk = 2112;
    constexpr int kValid = 100;

    Rig rig;
    build_rig(rig, kTokens, /*h_q=*/64, /*seed=*/0x5A17'0005u,
              /*plant_pos=*/2001);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    std::mt19937 rng(0x5A17'0055u);
    std::vector<int> sel = pick_positions(kValid, kTokens, rng);
    if (std::find(sel.begin(), sel.end(), 2001) == sel.end()) {
        sel.back() = 2001;
        std::sort(sel.begin(), sel.end());
    }
    std::vector<int> host_idx(kTopk, -1);
    for (size_t g = 0; g < sel.size(); ++g) host_idx[g] = rig.pos2slot[sel[g]];

    Arena a;
    auto* d_idx = a.allocT<int>(kTopk);
    ASSERT_NE(d_idx, nullptr);
    ASSERT_CUDA(cudaMemcpy(d_idx, host_idx.data(), kTopk * sizeof(int),
                           cudaMemcpyHostToDevice));

    DecodeIo io;
    build_decode_io(io, rig.h_q, /*max_parts=*/16);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    // Poison the accumulators and outputs DIFFERENTLY between the two runs:
    // a kernel that leaves a lane unwritten cannot pass this.
    const DecodeResult r1 = run_decode(rig, io, d_idx, kTopk, kTopk,
                                       /*num_sm_parts=*/8, /*deterministic=*/true,
                                       /*poison=*/0x00);
    const DecodeResult r2 = run_decode(rig, io, d_idx, kTopk, kTopk,
                                       /*num_sm_parts=*/8, /*deterministic=*/true,
                                       /*poison=*/0xFF);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_GT(r1.num_splits, 1) << "test requires a genuine multi-split schedule";
    for (float l : r1.lse) ASSERT_TRUE(std::isfinite(l)) << "lse " << l;
    EXPECT_TRUE(same_bits(r1, r2))
        << "deterministic_reduce=true is not run-to-run bit-identical";

    // Also the no-split path (sL feeds the epilogue directly there).
    const DecodeResult n1 = run_decode(rig, io, d_idx, kTopk, kTopk,
                                       /*num_sm_parts=*/1, /*deterministic=*/true,
                                       /*poison=*/0x11);
    const DecodeResult n2 = run_decode(rig, io, d_idx, kTopk, kTopk,
                                       /*num_sm_parts=*/1, /*deterministic=*/true,
                                       /*poison=*/0xEE);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_TRUE(same_bits(n1, n2))
        << "deterministic_reduce=true no-split path is not bit-identical";

    // Negative control: the harness CAN see a difference. Flip one FP8 byte of
    // the dominant (planted) row and require the same comparison to fail.
    const uint8_t old = poke_cache_byte(rig, 2001, /*dim=*/0, 0x40);
    if (old == 0x40) poke_cache_byte(rig, 2001, /*dim=*/0, 0x50);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const DecodeResult r3 = run_decode(rig, io, d_idx, kTopk, kTopk,
                                       /*num_sm_parts=*/8, /*deterministic=*/true,
                                       /*poison=*/0x00);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_FALSE(same_bits(r1, r3))
        << "flipping one attended FP8 byte did not change the output — the "
        << "bit-identity comparison above is vacuous";
    poke_cache_byte(rig, 2001, /*dim=*/0, old);
}

//==============================================================================
// (F) DET-REDUCE vs legacy atomicAdd: order-only difference
//==============================================================================

TEST(SnapmlaBudgetDecodeSparseFp8, Fp8SparseDecodeDetVsLegacyClose) {
    REQUIRES_GPU();

    constexpr int kTokens = 2048;
    constexpr int kTopk = 2048;   // 32 full blocks, every index valid

    Rig rig;
    build_rig(rig, kTokens, /*h_q=*/64, /*seed=*/0x5A17'0006u,
              /*plant_pos=*/2040);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    std::vector<int> sel(kTokens);
    std::iota(sel.begin(), sel.end(), 0);
    std::vector<int> host_idx(kTopk);
    for (int g = 0; g < kTopk; ++g) host_idx[g] = rig.pos2slot[g];

    Arena a;
    auto* d_idx = a.allocT<int>(kTopk);
    ASSERT_NE(d_idx, nullptr);
    ASSERT_CUDA(cudaMemcpy(d_idx, host_idx.data(), kTopk * sizeof(int),
                           cudaMemcpyHostToDevice));

    DecodeIo io;
    build_decode_io(io, rig.h_q, /*max_parts=*/16);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    for (int nsp : {1, 8}) {
        const DecodeResult legacy =
            run_decode(rig, io, d_idx, kTopk, kTopk, nsp,
                       /*deterministic=*/false, /*poison=*/0x22);
        const DecodeResult det =
            run_decode(rig, io, d_idx, kTopk, kTopk, nsp,
                       /*deterministic=*/true, /*poison=*/0x33);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        const Delta d = max_delta(legacy, det);
        const std::string tag = "nsp" + std::to_string(nsp);
        RecordProperty("det_vs_legacy_out_max_abs_" + tag,
                       std::to_string(d.out));
        RecordProperty("det_vs_legacy_lse_max_abs_" + tag,
                       std::to_string(d.lse));
        RecordProperty("det_vs_legacy_out_max_bf16_codes_" + tag,
                       std::to_string(d.out_ulp));
        std::printf("[ F ] nsp=%d det-vs-legacy: out max_abs=%.3e "
                    "(max %d bf16 codes), lse max_abs=%.3e\n",
                    nsp, d.out, d.out_ulp, d.lse);
        EXPECT_LE(d.out, 2e-3)
            << "det vs legacy out delta (nsp=" << nsp
            << ") exceeds the order-only bound";
        EXPECT_LE(d.lse, 1e-4)
            << "det vs legacy lse delta (nsp=" << nsp
            << ") exceeds the order-only bound";
    }
}

//==============================================================================
// (G) h_q 32 — the tp=2 per-rank shape
//==============================================================================

TEST(SnapmlaBudgetDecodeSparseFp8, Fp8SparseDecodeH32Heads) {
    REQUIRES_GPU();

    constexpr int kTokens = 2048;
    constexpr int kTopk = 2048;
    constexpr int kHQ32 = 32;   // < BLOCK_SIZE_M 64: the head-tail path

    Rig rig;
    build_rig(rig, kTokens, kHQ32, /*seed=*/0x5A17'0007u, /*plant_pos=*/2040);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_EQ(rig.h_q, kHQ32);

    std::vector<int> sel(kTokens);
    std::iota(sel.begin(), sel.end(), 0);
    std::vector<int> host_idx(kTopk);
    for (int g = 0; g < kTopk; ++g) host_idx[g] = rig.pos2slot[g];

    Arena a;
    auto* d_idx = a.allocT<int>(kTopk);
    ASSERT_NE(d_idx, nullptr);
    ASSERT_CUDA(cudaMemcpy(d_idx, host_idx.data(), kTopk * sizeof(int),
                           cudaMemcpyHostToDevice));

    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    std::vector<float> ref_out, ref_lse;
    cpu_reference(rig, sel, sm_scale, ref_out, ref_lse);

    DecodeIo io;
    build_decode_io(io, kHQ32, /*max_parts=*/16);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    const DecodeResult nosplit =
        run_decode(rig, io, d_idx, kTopk, kTopk, /*num_sm_parts=*/1,
                   /*deterministic=*/true, /*poison=*/0x44);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_EQ(nosplit.num_splits, 1);
    check_vs_reference(nosplit, ref_out, ref_lse, "h32 nsp=1");

    const DecodeResult split =
        run_decode(rig, io, d_idx, kTopk, kTopk, /*num_sm_parts=*/8,
                   /*deterministic=*/true, /*poison=*/0x55);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_GT(split.num_splits, 1);
    check_vs_reference(split, ref_out, ref_lse, "h32 nsp=8");

    // A short-top-k arm at h_q 32 too: the rescale guard and the head tail on
    // the same call.
    constexpr int kTopkShort = 2112;
    std::vector<int> short_sel = sel;
    short_sel.resize(96);
    std::vector<int> short_idx(kTopkShort, -1);
    for (size_t g = 0; g < short_sel.size(); ++g)
        short_idx[g] = rig.pos2slot[short_sel[g]];
    auto* d_short = a.allocT<int>(kTopkShort);
    ASSERT_NE(d_short, nullptr);
    ASSERT_CUDA(cudaMemcpy(d_short, short_idx.data(), kTopkShort * sizeof(int),
                           cudaMemcpyHostToDevice));
    std::vector<float> sref_out, sref_lse;
    cpu_reference(rig, short_sel, sm_scale, sref_out, sref_lse);
    const DecodeResult short_run =
        run_decode(rig, io, d_short, kTopkShort, kTopkShort,
                   /*num_sm_parts=*/8, /*deterministic=*/true, /*poison=*/0x66);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    check_vs_reference(short_run, sref_out, sref_lse, "h32 short-topk nsp=8");
}

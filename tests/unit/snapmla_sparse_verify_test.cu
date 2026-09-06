// P-32 stage 1 — batched spec_verify sparse decode (s_q = R) row isolation.
//
// THE CLAIM THIS FILE LOCKS: one s_q = R launch of the SnapMLA FP8 split-KV
// sparse decode chain is BIT-IDENTICAL, row for row, to R separate s_q = 1
// launches of the same chain. The kernel's grid.y axis indexes row-local
// Q / indices / out / accum slices (splitkv_mla.cu:184, 242-252, 627-669) and
// no per-row compute path touches another row's data, so batching R
// teacher-forced verify rows of ONE sequence must be a pure scheduling change,
// never a numerics change.
//
// Everything below drives the RAW kernel chain the production call site drives
// — src/compute/snapmla_sm120_attention_device.cpp
// SnapMlaSm120AttentionDevice::fp8_sparse_decode (search "P-32 stage 1") —
// not the device class:
//
//   launch_tq_sparse_translate_indices (once per row, each with its OWN causal
//     bound and its OWN topk_length)
//   launch_fused_q_quant               (s_q = R over the [R, h_q, d_qk] block)
//   launch_decode_sparse_fp8           (ONE launch, b = 1, s_q = R)
//   launch_mla_combine                 (q_seq_per_hk = R * h_q, when split)
//
// EVERY stride below is copied from that production function verbatim; the
// point of the test is to pin the production geometry, so a stride that drifts
// there must break this file.
//
// The legs:
//
//   (1) ROW ISOLATION, R = 3, h_q = 64. Arm A = one s_q=3 launch; arm B = three
//       s_q=1 launches into separate buffers. memcmp of out AND lse, per row,
//       must be byte-equal — at num_sm_parts 1 (no split, kernel epilogue
//       writes out/lse directly), 8 (genuine split + combine) and 34 (the
//       PRODUCTION default nsp = topk_pad/64 + 1, one 64-row index block per
//       split). The two arms are poisoned with DIFFERENT byte patterns
//       beforehand, so "the kernel wrote nothing" cannot pass.
//
//   (2) PER-ROW CAUSAL BOUNDS BITE. The rows carry ascending bounds
//       (2046/2047/2048 = consecutive MTP-verify positions) and their index
//       lists deliberately contain positions 2043..2047, so row 0 masks
//       {2046, 2047}, row 1 masks {2047} and row 2 masks nothing. If the
//       batched launch had collapsed to one shared bound, the per-row memcmp
//       would fail.
//
//       The selections are ~1500 entries wide ON PURPOSE. With a short
//       selection (a few hundred) EVERY valid index lands in the FIRST split,
//       every other split comes back fully masked (sL == 0 -> lse_accum -inf),
//       and mla_combine skips its o_accum reads entirely — which makes the
//       o_accum split stride unobservable. Measured: at nsp=8 (7 splits, 5
//       index blocks each) a 250-entry selection leaves only split 0
//       contributing, and a deliberately corrupted stride_o_accum_split
//       survives. ~1500 entries spread the contributing rows over ~24 index
//       blocks, so several splits carry real work at every nsp below.
//
//   (3) NEGATIVE CONTROL. The planted dominant token sits in the selections of
//       rows >= 1 and is EXCLUDED from row 0's. Flipping one FP8 byte of that
//       cache row and rerunning arm A alone must change rows >= 1 and leave
//       row 0 byte-identical — which proves both that the memcmp in (1) is not
//       vacuous and that a row only ever reads its own selection.
//
//   (4) R = 6, h_q = 32 — the tp=2 per-rank shape at a deeper verify batch,
//       h_q < BLOCK_SIZE_M 64 (the head-tail path) crossed with s_q > 1. Same
//       memcmp bar, same negative control.
//
// deterministic_reduce = true throughout: without it the cross-warp atomicAdd
// order is not reproducible and byte-equality is not even a well-posed
// question (leg (E) of snapmla_budget_decode_test.cu).

#include "compute/kernels/attention/mla_attention.h"
#include "compute/kernels/attention/prep_params.h"
#include "compute/kernels/attention/tq_mla_attention.h"

#include "smxx/get_mla_metadata.h"
#include "smxx/mla_combine.h"
#include "sm120/decode/sparse_fp8/params.h"

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
#include <set>
#include <string>
#include <vector>

namespace lc = layerstorm::compute;

namespace {

#define ASSERT_CUDA(expr)                                                     \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        ASSERT_EQ(_e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_e);\
    } while (0)

// ── GLM5N geometry (identical to snapmla_budget_decode_test.cu) ─────────────
constexpr int kDC = 512;                       // kv_lora_rank = d_nope = d_v
constexpr int kDR = 0;                         // NoPE: no rope block
constexpr int kDQK = kDC + kDR;                // 512
constexpr int kRowBytes = kDC + 4 + kDR * 2;   // 516 = [512 fp8 | f32 scale]
constexpr int kPage = 64;
constexpr int kTopkBlock = 64;                 // T::TOPK_BLOCK_SIZE / B_TOPK

static_assert(kRowBytes == 516, "GLM5N SnapMLA cache row (no rope tail)");

// Production top-k shape: 2051 is NOT a multiple of 64, so the kernel walks a
// 64-aligned 2112 with the tail permanently -1 (the persistent fill in
// ensure_decode_scratch).
constexpr int kTopk = 2051;
constexpr int kTopkPad = ((kTopk + kTopkBlock - 1) / kTopkBlock) * kTopkBlock;
static_assert(kTopkPad == 2112, "production 64-aligned top-k walk");

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

float bf16r(float f) { return __bfloat162float(__float2bfloat16(f)); }

float bits_to_float(uint16_t bits) {
    __nv_bfloat16 b;
    std::memcpy(&b, &bits, 2);
    return __bfloat162float(b);
}

//==============================================================================
// Rig: paged FP8 KV cache + a [R, h_q, d_qk] BF16 query block
//==============================================================================

struct VerifyRig {
    Arena arena;

    int num_tokens = 0;   // logical token POSITIONS
    int num_slots = 0;    // cache slots (>= num_tokens)
    int num_pages = 0;
    int rows = 0;         // R (s_q)
    int h_q = 0;

    void* d_cache = nullptr;
    int* d_pos2slot = nullptr;        // [num_tokens] position -> cache slot
    __nv_bfloat16* d_q = nullptr;     // [rows, h_q, d_qk] BF16

    std::vector<int> pos2slot;

    int64_t cache_bytes() const {
        return static_cast<int64_t>(num_pages) * kPage * kRowBytes;
    }
    int64_t slot_byte_offset(int slot) const {
        return static_cast<int64_t>(slot / kPage) * kPage * kRowBytes +
               static_cast<int64_t>(slot % kPage) * kRowBytes;
    }
    const __nv_bfloat16* q_row(int r) const {
        return d_q + static_cast<size_t>(r) * h_q * kDQK;
    }
};

// plant_pos >= 0 aligns that POSITION's key with the mean query direction over
// ALL rows and heads, so its softmax weight is O(1) for every (row, head) that
// selects it — the lever that makes the one-byte negative control visible in
// bf16 output.
void build_verify_rig(VerifyRig& rig, int num_tokens, int rows, int h_q,
                      unsigned seed, int plant_pos) {
    rig.num_tokens = num_tokens;
    rig.rows = rows;
    rig.h_q = h_q;
    rig.num_pages = (num_tokens + kPage - 1) / kPage;
    rig.num_slots = rig.num_pages * kPage;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    const size_t q_elems = static_cast<size_t>(rows) * h_q * kDQK;
    std::vector<float> qf(q_elems);
    std::vector<float> ckv(static_cast<size_t>(num_tokens) * kDC);
    for (auto& v : qf) v = bf16r(dist(rng));
    for (auto& v : ckv) v = bf16r(dist(rng));

    if (plant_pos >= 0) {
        constexpr float kPlantGain = 110.0f;
        for (int d = 0; d < kDC; ++d) {
            float acc = 0.f;
            for (int rh = 0; rh < rows * h_q; ++rh)
                acc += qf[static_cast<size_t>(rh) * kDQK + d];
            // Normalized by h_q (not rows*h_q) on purpose: the per-row logit
            // then keeps the same O(10) dominance the 1-row rig has.
            ckv[static_cast<size_t>(plant_pos) * kDC + d] =
                bf16r(acc / h_q * kPlantGain);
        }
    }

    // Nontrivial paged mapping: position p -> a shuffled slot. This is exactly
    // the linearized position->slot table the engine hands the translate kernel
    // as `lin_slots` (prefill_indices_scratch_).
    rig.pos2slot.resize(num_tokens);
    {
        std::vector<int> all(rig.num_slots);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        for (int p = 0; p < num_tokens; ++p) rig.pos2slot[p] = all[p];
    }

    // ── K append through the production prep kernel (NoPE, no rope leg) ──
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

    // ── The BF16 query block, [rows, h_q, d_qk] — the exact layout
    // fp8_sparse_decode hands launch_fused_q_quant with s_q = rows. ──
    std::vector<__nv_bfloat16> qb(qf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = __float2bfloat16(qf[i]);
    rig.d_q = rig.arena.allocT<__nv_bfloat16>(q_elems);
    ASSERT_NE(rig.d_q, nullptr);
    ASSERT_CUDA(cudaMemcpy(rig.d_q, qb.data(), qb.size() * 2,
                           cudaMemcpyHostToDevice));

    rig.d_pos2slot = rig.arena.allocT<int>(num_tokens);
    ASSERT_NE(rig.d_pos2slot, nullptr);
    ASSERT_CUDA(cudaMemcpy(rig.d_pos2slot, rig.pos2slot.data(),
                           num_tokens * sizeof(int), cudaMemcpyHostToDevice));
}

// Overwrite one FP8 payload byte of the cache row holding token `pos`.
uint8_t poke_cache_byte(const VerifyRig& rig, int pos, int dim, uint8_t value) {
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
// Per-row DSA selections
//==============================================================================

// Sorted, distinct positions in [0, num_tokens): `forced` always present,
// `exclude` never present, padded out to n_valid with random picks.
std::vector<int> row_positions(int n_valid, int num_tokens,
                               const std::vector<int>& forced, int exclude,
                               std::mt19937& rng) {
    std::set<int> s;
    for (int f : forced) {
        EXPECT_NE(f, exclude) << "forced position collides with the exclusion";
        s.insert(f);
    }
    std::vector<int> all(num_tokens);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    for (int v : all) {
        if (static_cast<int>(s.size()) >= n_valid) break;
        if (v == exclude) continue;
        s.insert(v);
    }
    return std::vector<int>(s.begin(), s.end());
}

// The per-row index rows the engine hands fp8_sparse_decode: a [rows, topk]
// block of TOKEN POSITIONS, ascending, -1 tail padded, with ONE out-of-range
// entry inside the counted prefix (the translate maps it to -1 via
// `t < bound`), plus per-row topk_lengths.
struct SelectionSet {
    std::vector<int> host;           // [rows * kTopk]
    std::vector<int> lengths;        // [rows]
    std::vector<std::vector<int>> valid;  // per-row in-range positions
};

SelectionSet make_selections(int rows, int num_tokens,
                             const std::vector<int>& bounds,
                             const std::vector<int>& forced, int plant_pos,
                             int n_valid_base, std::mt19937& rng) {
    SelectionSet s;
    s.host.assign(static_cast<size_t>(rows) * kTopk, -1);
    s.lengths.assign(rows, 0);
    s.valid.resize(rows);
    for (int r = 0; r < rows; ++r) {
        // Row 0 EXCLUDES the planted dominant token; every other row selects
        // it. That asymmetry is the negative control in leg (3).
        std::vector<int> f = forced;
        if (r > 0 && plant_pos >= 0) f.push_back(plant_pos);
        const int exclude = (r == 0) ? plant_pos : -1;
        std::vector<int> pos =
            row_positions(n_valid_base + 2 * r, num_tokens, f, exclude, rng);
        s.valid[r] = pos;
        int* row = s.host.data() + static_cast<size_t>(r) * kTopk;
        for (size_t i = 0; i < pos.size(); ++i) row[i] = pos[i];
        // One out-of-range entry inside the counted prefix.
        row[pos.size()] = num_tokens + 7;
        s.lengths[r] = static_cast<int>(pos.size()) + 1;
        EXPECT_LT(s.lengths[r], kTopk) << "row " << r << " needs -1 tail padding";
        EXPECT_LE(bounds[r], num_tokens);
    }
    return s;
}

//==============================================================================
// The decode chain (raw kernels, production strides)
//==============================================================================

struct DecodeBufs {
    Arena a;
    int rows = 0, h_q = 0, max_parts = 0, max_splits = 0;
    int* d_idx = nullptr;                 // [rows, kTopkPad] pool slots
    __nv_fp8_e4m3* d_q_fp8 = nullptr;     // [rows, h_q, d_c]
    float* d_q_scales = nullptr;          // [rows, h_q]
    __nv_bfloat16* d_out = nullptr;       // [rows, h_q, d_v]
    float* d_lse = nullptr;               // [rows, h_q]
    float* d_o_accum = nullptr;           // [max_splits, rows, h_q, d_v]
    float* d_lse_accum = nullptr;         // [max_splits, rows, h_q]
    int* d_meta = nullptr;
    int* d_splits = nullptr;
};

void build_bufs(DecodeBufs& b, int rows, int h_q, int max_parts) {
    b.rows = rows;
    b.h_q = h_q;
    b.max_parts = max_parts;
    b.max_splits = max_parts + 2;
    const size_t rh = static_cast<size_t>(rows) * h_q;
    b.d_idx = b.a.allocT<int>(static_cast<size_t>(rows) * kTopkPad);
    b.d_q_fp8 = static_cast<__nv_fp8_e4m3*>(b.a.alloc(rh * kDC));
    b.d_q_scales = b.a.allocT<float>(rh);
    b.d_out = b.a.allocT<__nv_bfloat16>(rh * kDC);
    b.d_lse = b.a.allocT<float>(rh);
    b.d_o_accum = b.a.allocT<float>(static_cast<size_t>(b.max_splits) * rh * kDC);
    b.d_lse_accum = b.a.allocT<float>(static_cast<size_t>(b.max_splits) * rh);
    b.d_meta = b.a.allocT<int>(static_cast<size_t>(max_parts) * 8);
    b.d_splits = b.a.allocT<int>(2);
    ASSERT_NE(b.d_idx, nullptr);
    ASSERT_NE(b.d_q_fp8, nullptr);
    ASSERT_NE(b.d_q_scales, nullptr);
    ASSERT_NE(b.d_out, nullptr);
    ASSERT_NE(b.d_lse, nullptr);
    ASSERT_NE(b.d_o_accum, nullptr);
    ASSERT_NE(b.d_lse_accum, nullptr);
    ASSERT_NE(b.d_meta, nullptr);
    ASSERT_NE(b.d_splits, nullptr);
    // One-time -1 fill: translate writes [0, kTopk) of every row each call; the
    // 64-alignment pad [kTopk, kTopkPad) must stay -1 forever (production
    // ensure_decode_scratch does exactly this memset once).
    ASSERT_CUDA(cudaMemset(b.d_idx, 0xFF,
                           static_cast<size_t>(rows) * kTopkPad * sizeof(int)));
}

struct RunOut {
    std::vector<uint16_t> out;   // raw bf16 bits [rows, h_q, d_v]
    std::vector<float> lse;      // [rows, h_q]
    int total_splits = 0;
    int rows = 0, h_q = 0;

    const uint16_t* out_row(int r) const {
        return out.data() + static_cast<size_t>(r) * h_q * kDC;
    }
    const float* lse_row(int r) const {
        return lse.data() + static_cast<size_t>(r) * h_q;
    }
};

// Split-KV schedule. b = 1 in BOTH arms (the batch axis carries one sequence;
// the verify rows ride grid.y), so the metadata is bit-identical between arms
// by construction — exactly the production situation.
int fill_metadata(DecodeBufs& b, int num_sm_parts) {
    GetMlaMetadataParams mp{};
    mp.seqlens_k_ptr = nullptr;
    mp.tile_scheduler_metadata_ptr = b.d_meta;
    mp.num_splits_ptr = b.d_splits;
    mp.batch_size = 1;
    mp.block_size_n = kTopkBlock;
    mp.fixed_overhead_num_blocks = 1;
    mp.num_sm_parts = num_sm_parts;
    mp.topk = kTopkPad;   // production: p.topk == topk_pad, 64-aligned walk
    lc::launch_get_mla_metadata(mp, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    int ns[2] = {-1, -1};
    EXPECT_EQ(cudaMemcpy(ns, b.d_splits, 2 * sizeof(int),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(ns[0], 0);
    return ns[1];
}

// The whole chain, at s_q = b.rows. Mirrors
// SnapMlaSm120AttentionDevice::fp8_sparse_decode step for step and stride for
// stride; `poison` seeds every output and accumulator byte beforehand.
RunOut run_chain(const VerifyRig& rig, DecodeBufs& b,
                 const __nv_bfloat16* d_q, const int* d_sel,
                 const int* d_lengths, const int* host_bounds,
                 int seq_len_kv, int num_sm_parts, int poison) {
    RunOut r;
    r.rows = b.rows;
    r.h_q = b.h_q;
    const int rows = b.rows, h_q = b.h_q;
    const size_t rh = static_cast<size_t>(rows) * h_q;

    EXPECT_EQ(cudaMemset(b.d_out, poison, rh * kDC * 2), cudaSuccess);
    EXPECT_EQ(cudaMemset(b.d_lse, poison, rh * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMemset(b.d_lse_accum, poison,
                         static_cast<size_t>(b.max_splits) * rh * sizeof(float)),
              cudaSuccess);
    EXPECT_EQ(cudaMemset(b.d_o_accum, poison,
                         static_cast<size_t>(b.max_splits) * rh * kDC *
                             sizeof(float)), cudaSuccess);
    // Q scratch is poisoned too: a quant that skipped a row would then show up
    // as garbage rather than as a stale-but-correct value.
    EXPECT_EQ(cudaMemset(b.d_q_fp8, poison, rh * kDC), cudaSuccess);
    EXPECT_EQ(cudaMemset(b.d_q_scales, poison, rh * sizeof(float)), cudaSuccess);

    r.total_splits = fill_metadata(b, num_sm_parts);

    // 1. Translate DSA token positions -> pool slots, per row, each with its
    //    OWN causal bound and its OWN topk_length.
    for (int row = 0; row < rows; ++row) {
        lc::launch_tq_sparse_translate_indices(
            d_sel + static_cast<size_t>(row) * kTopk, rig.d_pos2slot,
            d_lengths ? d_lengths + row : nullptr, kTopk,
            host_bounds ? host_bounds[row] : seq_len_kv,
            /*seq_len_dev=*/nullptr,
            b.d_idx + static_cast<size_t>(row) * kTopkPad, nullptr);
    }

    // 2. Quantize Q: BF16 -> FP8 NOPE + per-head scales, s_q = rows.
    sm120::prep::FusedQQuantParams qq{};
    qq.q_bf16 = d_q;
    qq.q_nope_fp8 = b.d_q_fp8;
    qq.q_rope_bf16 = nullptr;
    qq.q_scales = b.d_q_scales;
    qq.s_q = rows;
    qq.h_q = h_q;
    qq.d_qk = kDQK;
    qq.d_nope = kDC;
    lc::launch_fused_q_quant(qq, nullptr);

    // 3. The split-KV sparse FP8 kernel.
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kDQK));
    sm120::decode::sparse_fp8::SparseAttnDecodeParams p{};
    std::memset(&p, 0, sizeof(p));
    p.b = 1;
    p.s_q = rows;
    p.h_q = h_q;
    p.h_kv = 1;
    p.d_qk = kDQK;
    p.d_v = kDC;
    p.sm_scale = sm_scale;
    p.sm_scale_div_log2 = sm_scale * 1.44269504088896340736f;
    p.num_blocks = 0;
    p.page_block_size = kPage;
    p.topk = kTopkPad;
    p.model_type = sm120::decode::sparse_fp8::ModelType::GLM5N;
    p.q = reinterpret_cast<cutlass::bfloat16_t*>(b.d_q_fp8);   // FP8 bytes
    p.q_rope = nullptr;
    p.q_scales = b.d_q_scales;
    p.kv = static_cast<cutlass::bfloat16_t*>(rig.d_cache);     // FP8 bytes
    p.indices = b.d_idx;
    p.topk_length = nullptr;   // flat top-k; the -1 mask governs
    p.attn_sink = nullptr;
    p.lse = b.d_lse;
    p.out = reinterpret_cast<cutlass::bfloat16_t*>(b.d_out);
    p.stride_q_b = p.stride_q_s_q = h_q * kDQK;
    p.stride_q_h_q = kDQK;
    p.stride_kv_block = kPage * kRowBytes;
    p.stride_kv_row = kRowBytes;
    p.stride_indices_b = p.stride_indices_s_q = kTopkPad;
    p.stride_lse_b = p.stride_lse_s_q = h_q;
    p.stride_o_b = p.stride_o_s_q = h_q * kDC;
    p.stride_o_h_q = kDC;
    p.lse_accum = b.d_lse_accum;
    p.o_accum = b.d_o_accum;
    p.stride_lse_accum_split = rows * h_q;      // num_q_seqs = s_q * h_q
    p.stride_lse_accum_s_q = h_q;
    p.stride_o_accum_split = rows * h_q * kDC;
    p.stride_o_accum_s_q = h_q * kDC;
    p.stride_o_accum_h_q = kDC;
    p.tile_scheduler_metadata_ptr =
        reinterpret_cast<sm120::decode::sparse_fp8::DecodingSchedMeta*>(b.d_meta);
    p.num_splits_ptr = b.d_splits;
    p.num_sm_parts = num_sm_parts;
    p.deterministic_reduce = true;
    p.stream = nullptr;

    lc::launch_decode_sparse_fp8(p);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 4. Merge split partials.
    if (r.total_splits > 1) {
        MlaCombineParams c{};
        std::memset(&c, 0, sizeof(c));
        c.b = 1;
        c.h_q = h_q;
        c.h_k = 1;
        c.q_seq_per_hk = rows * h_q;   // h_q / h_k * s_q
        c.d_v = kDC;
        c.o_ptr = b.d_out;
        c.softmax_lse_ptr = b.d_lse;
        c.o_batch_stride = static_cast<int64_t>(h_q) * kDC;
        c.o_row_stride = kDC;
        c.o_head_stride = kDC;
        c.num_splits_ptr = b.d_splits;
        c.num_sm_parts = num_sm_parts;
        c.softmax_lseaccum_ptr = b.d_lse_accum;
        c.oaccum_ptr = b.d_o_accum;
        lc::launch_mla_combine(c, nullptr);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    r.out.resize(rh * kDC);
    r.lse.resize(rh);
    EXPECT_EQ(cudaMemcpy(r.out.data(), b.d_out, r.out.size() * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.lse.data(), b.d_lse, rh * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    return r;
}

bool row_bits_equal(const RunOut& a, int ra, const RunOut& bb, int rb) {
    const size_t on = static_cast<size_t>(a.h_q) * kDC;
    return std::memcmp(a.out_row(ra), bb.out_row(rb), on * 2) == 0 &&
           std::memcmp(a.lse_row(ra), bb.lse_row(rb),
                       static_cast<size_t>(a.h_q) * sizeof(float)) == 0;
}

// A run whose output is all-poison / all-zero would make every memcmp pass
// trivially. Guard against it once per arm.
void expect_row_nondegenerate(const RunOut& r, int row, const char* tag) {
    const uint16_t* o = r.out_row(row);
    int good = 0;
    for (int i = 0; i < r.h_q * kDC; ++i) {
        const float v = bits_to_float(o[i]);
        ASSERT_TRUE(std::isfinite(v)) << tag << " row " << row
                                      << " non-finite out[" << i << "]";
        if (v != 0.0f) ++good;
    }
    ASSERT_GT(good, r.h_q * kDC / 2)
        << tag << " row " << row << " output looks degenerate";
    const float* l = r.lse_row(row);
    for (int h = 0; h < r.h_q; ++h)
        ASSERT_TRUE(std::isfinite(l[h])) << tag << " row " << row
                                         << " non-finite lse[" << h << "]";
}

//==============================================================================
// Shared case body
//==============================================================================

struct VerifyCase {
    VerifyRig rig;
    Arena a;
    SelectionSet sel;
    std::vector<int> bounds;
    int* d_sel = nullptr;        // [rows, kTopk]
    int* d_lengths = nullptr;    // [rows]
    int rows = 0, h_q = 0, tokens = 0, plant_pos = 0;
};

void build_case(VerifyCase& c, int rows, int h_q, int tokens, int plant_pos,
                const std::vector<int>& bounds,
                const std::vector<int>& forced, int n_valid_base,
                unsigned seed) {
    c.rows = rows;
    c.h_q = h_q;
    c.tokens = tokens;
    c.plant_pos = plant_pos;
    c.bounds = bounds;
    ASSERT_EQ(static_cast<int>(bounds.size()), rows);

    build_verify_rig(c.rig, tokens, rows, h_q, seed, plant_pos);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    std::mt19937 rng(seed ^ 0x9E37u);
    c.sel = make_selections(rows, tokens, bounds, forced, plant_pos,
                            n_valid_base, rng);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    c.d_sel = c.a.allocT<int>(static_cast<size_t>(rows) * kTopk);
    c.d_lengths = c.a.allocT<int>(rows);
    ASSERT_NE(c.d_sel, nullptr);
    ASSERT_NE(c.d_lengths, nullptr);
    ASSERT_CUDA(cudaMemcpy(c.d_sel, c.sel.host.data(),
                           c.sel.host.size() * sizeof(int),
                           cudaMemcpyHostToDevice));
    ASSERT_CUDA(cudaMemcpy(c.d_lengths, c.sel.lengths.data(),
                           rows * sizeof(int), cudaMemcpyHostToDevice));

    // The bounds must actually bite: at least one selected position of row 0
    // has to sit at or past row 0's causal bound, otherwise leg (2) is inert.
    int masked_row0 = 0;
    for (int p : c.sel.valid[0]) if (p >= bounds[0]) ++masked_row0;
    ASSERT_GT(masked_row0, 0) << "row 0's causal bound masks nothing";
    // Row 0 must NOT select the planted token (the leg-(3) asymmetry).
    ASSERT_EQ(std::find(c.sel.valid[0].begin(), c.sel.valid[0].end(), plant_pos),
              c.sel.valid[0].end());
    for (int r = 1; r < rows; ++r)
        ASSERT_NE(std::find(c.sel.valid[r].begin(), c.sel.valid[r].end(),
                            plant_pos), c.sel.valid[r].end())
            << "row " << r << " must select the planted token";
}

// Arm A (one s_q=R launch) vs arm B (R s_q=1 launches), byte for byte.
// Returns the split count the schedule produced (-1 if the run bailed early),
// so the caller can PROVE the split/no-split legs were really taken.
int run_row_isolation(VerifyCase& c, int num_sm_parts) {
    const std::string tag = "R" + std::to_string(c.rows) + " h" +
                            std::to_string(c.h_q) + " nsp" +
                            std::to_string(num_sm_parts);

    DecodeBufs ba;
    build_bufs(ba, c.rows, c.h_q, num_sm_parts);
    if (::testing::Test::HasFatalFailure()) return -1;
    const RunOut A = run_chain(c.rig, ba, c.rig.d_q, c.d_sel, c.d_lengths,
                               c.bounds.data(), c.tokens, num_sm_parts,
                               /*poison=*/0x3C);
    if (::testing::Test::HasFatalFailure()) return -1;

    for (int r = 0; r < c.rows; ++r) {
        // Arm B: the SAME chain at s_q = 1, its own buffers, its own bound,
        // its own topk_length, poisoned with a DIFFERENT pattern.
        DecodeBufs bb;
        build_bufs(bb, /*rows=*/1, c.h_q, num_sm_parts);
        if (::testing::Test::HasFatalFailure()) return -1;
        const int bound = c.bounds[r];
        const RunOut B = run_chain(
            c.rig, bb, c.rig.q_row(r),
            c.d_sel + static_cast<size_t>(r) * kTopk, c.d_lengths + r,
            &bound, c.tokens, num_sm_parts, /*poison=*/0xC3);
        if (::testing::Test::HasFatalFailure()) return -1;
        EXPECT_EQ(A.total_splits, B.total_splits)
            << tag << ": the split schedule must not depend on s_q";
        expect_row_nondegenerate(A, r, (tag + " armA").c_str());
        expect_row_nondegenerate(B, 0, (tag + " armB").c_str());
        if (::testing::Test::HasFatalFailure()) return -1;

        const size_t on = static_cast<size_t>(c.h_q) * kDC;
        EXPECT_EQ(std::memcmp(A.out_row(r), B.out_row(0), on * 2), 0)
            << tag << ": batched row " << r << " OUT differs from its own "
            << "s_q=1 launch — the s_q=R generalization is not row-local";
        EXPECT_EQ(std::memcmp(A.lse_row(r), B.lse_row(0),
                              static_cast<size_t>(c.h_q) * sizeof(float)), 0)
            << tag << ": batched row " << r << " LSE differs from its own "
            << "s_q=1 launch";
    }

    // Rows must not be copies of each other (per-row bounds + per-row
    // selections + per-row Q); otherwise the memcmps above prove nothing about
    // row indexing.
    for (int r = 1; r < c.rows; ++r)
        EXPECT_FALSE(row_bits_equal(A, 0, A, r))
            << tag << ": rows 0 and " << r << " came back identical — the "
            << "batched launch is not reading row-local Q/indices";
    return A.total_splits;
}

// Flip one FP8 byte of the planted (dominant) cache row: every row that
// SELECTS it must change, and row 0 — which does not — must not.
void run_negative_control(VerifyCase& c, int num_sm_parts) {
    DecodeBufs ba;
    build_bufs(ba, c.rows, c.h_q, num_sm_parts);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const RunOut base = run_chain(c.rig, ba, c.rig.d_q, c.d_sel, c.d_lengths,
                                  c.bounds.data(), c.tokens, num_sm_parts,
                                  /*poison=*/0x3C);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_GT(base.total_splits, 1)
        << "the negative control must run on the split + combine path";

    const uint8_t old = poke_cache_byte(c.rig, c.plant_pos, /*dim=*/0, 0x40);
    const uint8_t used = (old == 0x40) ? 0x50 : 0x40;
    if (used != 0x40) poke_cache_byte(c.rig, c.plant_pos, 0, used);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    const RunOut flipped = run_chain(c.rig, ba, c.rig.d_q, c.d_sel, c.d_lengths,
                                     c.bounds.data(), c.tokens, num_sm_parts,
                                     /*poison=*/0x3C);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    EXPECT_TRUE(row_bits_equal(base, 0, flipped, 0))
        << "row 0 does not select the flipped token, yet its output moved — "
        << "the batched launch is reading another row's selection";
    for (int r = 1; r < c.rows; ++r)
        EXPECT_FALSE(row_bits_equal(base, r, flipped, r))
            << "flipping one FP8 byte of the dominant selected row left row "
            << r << " unchanged — the bit-identity comparisons are vacuous";

    poke_cache_byte(c.rig, c.plant_pos, /*dim=*/0, old);
}

}  // namespace

//==============================================================================
// (1)(2)(3) R = 3, h_q = 64 — the MTP-verify batch of the full-head rank
//==============================================================================

namespace {

// 2048 tokens; consecutive verify positions -> ascending causal bounds; the
// forced positions straddle those bounds so each row masks a different tail.
constexpr int kTokens1 = 2048;
const std::vector<int> kBounds1 = {2046, 2047, 2048};
const std::vector<int> kForced1 = {2043, 2044, 2045, 2046, 2047};
constexpr int kPlant1 = 977;

}  // namespace

TEST(SnapMlaSparseVerifyR3, BatchedRowsBitIdenticalToPerRowLaunches) {
    REQUIRES_GPU();

    VerifyCase c;
    build_case(c, /*rows=*/3, /*h_q=*/64, kTokens1, kPlant1, kBounds1, kForced1,
               /*n_valid_base=*/1500, /*seed=*/0x9210'0001u);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    // nsp 1  : no split at all — the kernel epilogue writes out/lse directly.
    // nsp 8  : a genuine multi-split schedule + mla_combine.
    // nsp 34 : the PRODUCTION default (topk_pad / 64 + 1) — one 64-row index
    //          block per split, so whole splits are fully masked.
    for (int nsp : {1, 8, 34}) {
        const int splits = run_row_isolation(c, nsp);
        ASSERT_FALSE(::testing::Test::HasFatalFailure())
            << "row isolation failed at nsp=" << nsp;
        // The legs must really be the legs they claim to be: nsp=1 is the
        // no-split epilogue (o_accum untouched), nsp>1 must produce a genuine
        // multi-split schedule so the accum strides and mla_combine are on the
        // hook. Without this, a scheduler change could quietly turn the split
        // legs into no-split repeats and the file would still be green.
        if (nsp == 1) {
            EXPECT_EQ(splits, 1) << "expected the no-split walk at nsp=1";
        } else {
            EXPECT_GT(splits, 1)
                << "nsp=" << nsp << " did not produce a multi-split schedule — "
                << "the o_accum/lse_accum strides and mla_combine are untested";
        }
        std::printf("[ P-32 ] %s nsp=%d -> %d split(s)\n",
                    ::testing::UnitTest::GetInstance()->current_test_info()->name(),
                    nsp, splits);
    }
}

TEST(SnapMlaSparseVerifyR3, NegativeControlOnlyTheSelectingRowsMove) {
    REQUIRES_GPU();

    VerifyCase c;
    build_case(c, /*rows=*/3, /*h_q=*/64, kTokens1, kPlant1, kBounds1, kForced1,
               /*n_valid_base=*/1500, /*seed=*/0x9210'0002u);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    run_negative_control(c, /*num_sm_parts=*/8);
}

//==============================================================================
// (4) R = 6, h_q = 32 — the tp=2 per-rank shape at a deeper verify batch
//==============================================================================

namespace {

constexpr int kTokens2 = 2048;
const std::vector<int> kBounds2 = {2043, 2044, 2045, 2046, 2047, 2048};
const std::vector<int> kForced2 = {2042, 2043, 2044, 2045, 2046, 2047};
constexpr int kPlant2 = 1301;

}  // namespace

TEST(SnapMlaSparseVerifyR6H32, BatchedRowsBitIdenticalToPerRowLaunches) {
    REQUIRES_GPU();

    VerifyCase c;
    build_case(c, /*rows=*/6, /*h_q=*/32, kTokens2, kPlant2, kBounds2, kForced2,
               /*n_valid_base=*/1500, /*seed=*/0x9210'0003u);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    for (int nsp : {1, 8, 34}) {
        const int splits = run_row_isolation(c, nsp);
        ASSERT_FALSE(::testing::Test::HasFatalFailure())
            << "row isolation failed at nsp=" << nsp;
        // The legs must really be the legs they claim to be: nsp=1 is the
        // no-split epilogue (o_accum untouched), nsp>1 must produce a genuine
        // multi-split schedule so the accum strides and mla_combine are on the
        // hook. Without this, a scheduler change could quietly turn the split
        // legs into no-split repeats and the file would still be green.
        if (nsp == 1) {
            EXPECT_EQ(splits, 1) << "expected the no-split walk at nsp=1";
        } else {
            EXPECT_GT(splits, 1)
                << "nsp=" << nsp << " did not produce a multi-split schedule — "
                << "the o_accum/lse_accum strides and mla_combine are untested";
        }
        std::printf("[ P-32 ] %s nsp=%d -> %d split(s)\n",
                    ::testing::UnitTest::GetInstance()->current_test_info()->name(),
                    nsp, splits);
    }
}

TEST(SnapMlaSparseVerifyR6H32, NegativeControlOnlyTheSelectingRowsMove) {
    REQUIRES_GPU();

    VerifyCase c;
    build_case(c, /*rows=*/6, /*h_q=*/32, kTokens2, kPlant2, kBounds2, kForced2,
               /*n_valid_base=*/1500, /*seed=*/0x9210'0004u);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    run_negative_control(c, /*num_sm_parts=*/8);
}

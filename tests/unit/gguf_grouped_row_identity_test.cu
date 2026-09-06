// P-29 step 13 — PER-ROW M-INDEPENDENCE of the routed-expert GGUF grouped GEMM.
//
// The speculative-verify pass batches M = 3 draft rows through the MoE in ONE
// grouped call instead of three M = 1 calls. Token identity of the verify pass
// rests on a BITWISE claim: row r's output at M = 3 must be byte-identical to
// the output of the same row run alone at M = 1, because the per-row FP
// reduction order in these kernels does not depend on M.
//
// Why the claim is not free (i.e. what this test would catch):
//   * launch_gguf_grouped_int picks mmvq vs mmq from the OVERALL token count
//     (avg_m > 8 -> mmq), so M can change the kernel outright;
//   * inside the default compact mmvq layout the row tile (MT = 4 cp.async /
//     register pipe, MT = 1 vs MT = 8 on the E == 1 k-split fast path) is also
//     chosen by total_tokens — an M-dependent DISPATCH whose kernels merely
//     claim identical per-row accumulate order;
//   * the Q8_1 activation quantization is done ONCE for all rows.
// Any regression that made a row's accumulation order depend on its neighbours
// (a cross-row reduction, a split-K rebalance by M, a shared-scale quant)
// breaks the verify pass silently. This test pins it byte-exactly.
//
// Coverage (Q4_K weights, the GLM routed-expert format; K = 512 = 2 Q4_K
// superblocks, N = 256):
//   1. Uniform routing  — 3 tokens x 3 experts (M_e = 3 vs M_e = 1).
//   2. Mixed routing    — realistic verify layout, ragged M_e = {3, 2, 1}
//                         incl. an EMPTY expert segment in the M = 1 runs.
//   3. Single expert    — E == 1, which takes the e1-ksplit fast path whose
//                         MT is selected by total_tokens == 1 vs > 1.
//   4. NEGATIVE CONTROL — perturbing ONE activation element of ONE row must
//                         change that row's bytes and leave its neighbours
//                         byte-identical (proves the comparison can fail and
//                         that rows are genuinely independent).
//
// GPU-required (REQUIRES_GPU); SKIPPED headless.

#include "sm120/gemm/gguf/gguf_grouped_int.h"
#include "sm120/gemm/gguf/gguf_grouped_gemm.h"   // workspace_bytes, block geometry
#include "sm120/gemm/gguf/gguf_dequant_gemm.h"   // gguf_block_bytes/values

#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ik = layerstorm::compute::cpu::ik;
using namespace layerstorm::compute;

namespace {

// P-29 step 13 follow-up: REAL routed-expert shapes (glm5_next gate/up:
// K=hidden=4096, N=moe_intermediate=2048) — the serving fork reproduced
// only at depth, so the identity gate must run the real K/N.
constexpr int kN = 2048;
constexpr int kK = 4096;  // 16 Q4_K superblocks (QK = 256)

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                         cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

// Raw BF16 bit pattern of one output element — comparisons are on BYTES, not
// on the decoded float (so a NaN payload difference would also be caught).
uint16_t bits(__nv_bfloat16 v) {
    uint16_t b;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

// A fixed set of Q4_K-packed experts resident on the device, reused across
// every launch of one test so the weights are bit-identical between arms.
class Experts {
public:
    Experts(int num_experts, uint32_t seed) : num_experts_(num_experts) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> wd(-0.6f, 0.6f);
        const size_t row_bytes = ik::weight_row_bytes(ik::GgufType::q4_k, kK);
        EXPECT_EQ(static_cast<int>(row_bytes),
                  (kK / gguf_block_values(GgufType::Q4_K)) *
                      gguf_block_bytes(GgufType::Q4_K));
        std::vector<void*> hptrs(num_experts_);
        for (int e = 0; e < num_experts_; ++e) {
            std::vector<uint8_t> packed(static_cast<size_t>(kN) * row_bytes);
            for (int n = 0; n < kN; ++n) {
                std::vector<float> wr(kK);
                for (int k = 0; k < kK; ++k) wr[k] = wd(rng);
                ik::quantize_weight(ik::GgufType::q4_k, wr.data(),
                                    packed.data() + static_cast<size_t>(n) * row_bytes,
                                    kK);
            }
            hptrs[e] = upload(packed);
        }
        owned_ = hptrs;
        d_ptrs_ = upload(hptrs);
    }
    ~Experts() {
        for (void* p : owned_) cudaFree(p);
        cudaFree(d_ptrs_);
    }
    const void* const* device_ptrs() const {
        return const_cast<const void**>(d_ptrs_);
    }
    int size() const { return num_experts_; }

private:
    int num_experts_;
    std::vector<void*> owned_;
    void** d_ptrs_ = nullptr;
};

// One grouped launch. `A` is the PERMUTED activation block [total, K] (rows
// already grouped by expert), `offsets` the [E + 1] cumulative row counts.
// Returns D [total, N] as raw BF16 bits.
std::vector<uint16_t> run_grouped(const Experts& experts,
                                  const std::vector<__nv_bfloat16>& A,
                                  const std::vector<int32_t>& offsets,
                                  GgufGroupedIntForce force) {
    const int E = experts.size();
    const int total = offsets.back();
    EXPECT_EQ(static_cast<int>(A.size()), total * kK);

    __nv_bfloat16* dA = upload(A);
    int32_t* dOff = upload(offsets);

    // Sentinel-initialize D: an untouched row would show up as the sentinel in
    // both arms and silently "match", so make the sentinel distinctive and
    // assert below that no output row is left at it.
    const __nv_bfloat16 sentinel = __float2bfloat16(-12345.0f);
    std::vector<__nv_bfloat16> D_init(static_cast<size_t>(total) * kN, sentinel);
    __nv_bfloat16* dD = upload(D_init);

    const size_t ws_bytes = gguf_grouped_gemm_workspace_bytes(total, kK, E);
    void* ws = nullptr;
    EXPECT_EQ(cudaMalloc(&ws, ws_bytes), cudaSuccess);

    launch_gguf_grouped_int(GgufType::Q4_K, E, kN, kK, total, dA, dD, dOff,
                            experts.device_ptrs(), ws, ws_bytes, force,
                            /*stream=*/0);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> D(static_cast<size_t>(total) * kN);
    EXPECT_EQ(cudaMemcpy(D.data(), dD, D.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(dA); cudaFree(dOff); cudaFree(dD); cudaFree(ws);

    std::vector<uint16_t> out(D.size());
    bool all_sentinel = !D.empty();
    for (size_t i = 0; i < D.size(); ++i) {
        out[i] = bits(D[i]);
        if (out[i] != bits(sentinel)) all_sentinel = false;
    }
    EXPECT_FALSE(all_sentinel) << "grouped launch wrote nothing (total="
                               << total << ")";
    return out;
}

// Permuted-layout builder. `routing[t]` lists the experts token t is sent to.
// Rows are grouped by expert (the engine's permuted MoE layout).
struct Permuted {
    std::vector<__nv_bfloat16> A;     // [total, K]
    std::vector<int32_t> offsets;     // [E + 1]
    // row_of[t][e] = permuted row index of (token t, expert e), or -1.
    std::vector<std::vector<int>> row_of;
};

Permuted permute(const std::vector<std::vector<__nv_bfloat16>>& tokens,
                 const std::vector<std::vector<int>>& routing, int E) {
    const int T = static_cast<int>(tokens.size());
    Permuted p;
    p.offsets.assign(E + 1, 0);
    p.row_of.assign(T, std::vector<int>(E, -1));
    int row = 0;
    for (int e = 0; e < E; ++e) {
        p.offsets[e] = row;
        for (int t = 0; t < T; ++t) {
            bool hit = false;
            for (int x : routing[t]) hit |= (x == e);
            if (!hit) continue;
            p.row_of[t][e] = row++;
            p.A.insert(p.A.end(), tokens[t].begin(), tokens[t].end());
        }
    }
    p.offsets[E] = row;
    return p;
}

std::vector<std::vector<__nv_bfloat16>> make_tokens(int T, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ad(-1.0f, 1.0f);
    std::vector<std::vector<__nv_bfloat16>> tok(T,
        std::vector<__nv_bfloat16>(kK));
    for (auto& row : tok)
        for (auto& v : row) v = __float2bfloat16(ad(rng));
    return tok;
}

// Core assertion: batching every token in ONE grouped call must reproduce, byte
// for byte, the rows produced by running each token ALONE.
void assert_row_identity(int E, const std::vector<std::vector<int>>& routing,
                         uint32_t seed, GgufGroupedIntForce force) {
    const int T = static_cast<int>(routing.size());
    Experts experts(E, seed ^ 0x5eedu);
    auto tokens = make_tokens(T, seed);

    const Permuted batched = permute(tokens, routing, E);
    const std::vector<uint16_t> D_batched =
        run_grouped(experts, batched.A, batched.offsets, force);

    for (int t = 0; t < T; ++t) {
        SCOPED_TRACE("token " + std::to_string(t));
        const Permuted solo =
            permute({tokens[t]}, {routing[t]}, E);
        const std::vector<uint16_t> D_solo =
            run_grouped(experts, solo.A, solo.offsets, force);
        for (int e = 0; e < E; ++e) {
            const int rb = batched.row_of[t][e];
            const int rs = solo.row_of[0][e];
            ASSERT_EQ(rb < 0, rs < 0) << "routing mismatch, expert " << e;
            if (rb < 0) continue;
            for (int n = 0; n < kN; ++n) {
                ASSERT_EQ(D_batched[static_cast<size_t>(rb) * kN + n],
                          D_solo[static_cast<size_t>(rs) * kN + n])
                    << "expert " << e << " col " << n
                    << ": batched row " << rb << " != solo row " << rs;
            }
        }
    }
}

class GgufGroupedRowIdentity : public ::testing::Test {
protected:
    void SetUp() override {
        REQUIRES_GPU();
        if (!ik::gguf_supported(ik::GgufType::q4_k))
            GTEST_SKIP() << "ik q4_k packer unsupported in this build";
    }
};

// 1. Uniform routing: the M = 3 verify batch, every row on every expert.
TEST_F(GgufGroupedRowIdentity, Q4K_UniformRouting_M3) {
    assert_row_identity(/*E=*/3, {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}, 0xA11C,
                        GgufGroupedIntForce::Auto);
}

// 2. Mixed routing: ragged M_e = {3, 2, 1}; the solo runs carry EMPTY expert
//    segments, the batched run does not.
TEST_F(GgufGroupedRowIdentity, Q4K_MixedRouting_M3) {
    assert_row_identity(/*E=*/3, {{0, 1}, {0, 2}, {0, 1}}, 0xB22D,
                        GgufGroupedIntForce::Auto);
}

// 3. E == 1 (dense/shared FFN shape): the e1-ksplit fast path selects its row
//    tile from total_tokens == 1 vs > 1 — the sharpest M-dependent dispatch.
TEST_F(GgufGroupedRowIdentity, Q4K_SingleExpert_M3) {
    assert_row_identity(/*E=*/1, {{0}, {0}, {0}}, 0xC33E,
                        GgufGroupedIntForce::Auto);
}

// 4. Forced-mmvq arm: pins the decode kernel family even if the mmvq/mmq
//    cut-over ever moves, so the contract keeps being tested where GF3 uses it.
TEST_F(GgufGroupedRowIdentity, Q4K_ForcedMmvq_M3) {
    assert_row_identity(/*E=*/2, {{0, 1}, {0, 1}, {0, 1}}, 0xD44F,
                        GgufGroupedIntForce::Mmvq);
}

// 5. NEGATIVE CONTROL. Perturbing ONE activation element of the middle row
//    must (a) change that row's output bytes — proving the byte comparison can
//    fail — and (b) leave both neighbour rows byte-identical, proving the rows
//    are independent rather than the test being insensitive.
TEST_F(GgufGroupedRowIdentity, Q4K_NegativeControl_PerturbOneRow) {
    constexpr int E = 3;
    const std::vector<std::vector<int>> routing = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
    Experts experts(E, 0xE55Au);
    auto tokens = make_tokens(3, 0xE55Au);

    const Permuted base = permute(tokens, routing, E);
    const std::vector<uint16_t> D0 =
        run_grouped(experts, base.A, base.offsets, GgufGroupedIntForce::Auto);

    // Perturb one element of token 1 (a real BF16 step, not a no-op round).
    auto perturbed = tokens;
    const float old_v = __bfloat162float(perturbed[1][7]);
    perturbed[1][7] = __float2bfloat16(old_v + 0.5f);
    ASSERT_NE(bits(perturbed[1][7]), bits(tokens[1][7]));

    const Permuted pert = permute(perturbed, routing, E);
    const std::vector<uint16_t> D1 =
        run_grouped(experts, pert.A, pert.offsets, GgufGroupedIntForce::Auto);

    for (int e = 0; e < E; ++e) {
        SCOPED_TRACE("expert " + std::to_string(e));
        // (a) the perturbed row MUST differ somewhere.
        const int rp = base.row_of[1][e];
        bool differs = false;
        for (int n = 0; n < kN && !differs; ++n)
            differs = D0[static_cast<size_t>(rp) * kN + n] !=
                      D1[static_cast<size_t>(rp) * kN + n];
        EXPECT_TRUE(differs)
            << "negative control is vacuous: perturbing an input left the "
               "output of its own row unchanged";
        // (b) untouched rows MUST be byte-identical.
        for (int t : {0, 2}) {
            const int r = base.row_of[t][e];
            for (int n = 0; n < kN; ++n) {
                ASSERT_EQ(D0[static_cast<size_t>(r) * kN + n],
                          D1[static_cast<size_t>(r) * kN + n])
                    << "row " << r << " (token " << t << ") changed when a "
                       "DIFFERENT row's input was perturbed";
            }
        }
    }
}

}  // namespace

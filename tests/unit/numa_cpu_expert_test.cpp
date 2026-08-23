// Unit tests for the CPU expert FFN (C-6, NumaCpuExpertDevice).
//
// CPU-only suite — NO CUDA / GPU SDK. Suite name "NumaCpuExpert" is deliberately
// NOT in tests/unit/CMakeLists.txt _GPU_FILTER, so it runs in the parallel CPU
// pass with no GPU. Auto-discovered by file(GLOB ... *_test.cpp).
//
// Coverage:
//   * parse_cpulist / node_physical_cpus  — range expansion + SMT-sibling drop.
//   * ik barrier                          — N-thread rendezvous correctness.
//   * scale_byte_index / dequant_weight   — vs nvfp4_sfb_reformat.h forward map
//                                           + a fresh scalar E2M1*E4M3*ws2 ref.
//   * cpu_nvfp4_grouped_gemm              — vs scalar dequant-weight matmul.
//   * cpu_swiglu / cpu_moe_permute / cpu_moe_unpermute — vs scalar references.
//   * NumaCpuExpertDevice end-to-end FFN  — permute -> gate -> up -> swiglu ->
//                                           down -> unpermute vs scalar golden.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "compute/cpu/cpu_gguf_gemm.h"
#include "compute/cpu/cpu_moe_kernels.h"
#include "compute/cpu/ik_barrier.h"
#include "compute/cpu/ik_vendor/ik_gguf_gemm.h"
#include "compute/cpu/multi_numa_cpu_expert_device.h"
#include "compute/cpu/numa_cpu_expert_device.h"
#include "compute/cpu/numa_thread_pool.h"
#include "compute/cpu/nvfp4_cpu_kernel.h"
#include "model/quantization/fp8.h"
#include "model/weight_loader/nvfp4_sfb_reformat.h"

using namespace layerstorm;
using namespace layerstorm::compute;
using namespace layerstorm::compute::cpu;

namespace {

// ── BF16 helpers (match src/core/bf16_convert.h) ─────────────────────────────
uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}
float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ── E2M1 quantize (nearest of the 16 codepoints) ─────────────────────────────
constexpr float kMag[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
uint8_t quantize_e2m1(float v) {
    float a = std::fabs(v);
    int best = 0;
    float bd = std::fabs(a - kMag[0]);
    for (int i = 1; i < 8; ++i) {
        float d = std::fabs(a - kMag[i]);
        if (d < bd) { bd = d; best = i; }
    }
    uint8_t sign = (v < 0.0f) ? 1 : 0;
    return static_cast<uint8_t>((sign << 3) | best);
}

// E4M3 encode of 1.0 is 0x38 (per fp8_e4m3::decode(0x38) == 1.0).
constexpr uint8_t kE4M3_One = 0x38;

// Mirror of NumaCpuExpertDevice's nvfp4_proj_bytes / nvfp4.cpp.
int64_t proj_bytes(int N, int K) {
    int64_t params = static_cast<int64_t>(N) * K;
    int64_t wb = (params + 1) / 2;
    int64_t sb = (params + 15) / 16;
    int64_t raw = wb + sb + 2 * static_cast<int64_t>(sizeof(float));
    return (raw + 127) & ~static_cast<int64_t>(127);
}

// Build a packed nvfp4-sm1xx projection block from a dense FP32 weight [N,K]
// (row-major). Quantizes each elem to E2M1, uses a UNIFORM E4M3 group scale of
// 1.0 (0x38) and weight_scale_2 = ws2 — so dequant(n,k) == E2M1(w)*ws2 exactly,
// giving a clean scalar reference. Scales are written in the Sm1xx interleaved
// layout via reformat_nvfp4_sfb (so scale_byte_index must invert it).
std::vector<uint8_t> pack_proj(const std::vector<float>& W, int N, int K,
                               float ws2) {
    const int64_t pb = proj_bytes(N, K);
    std::vector<uint8_t> buf(static_cast<size_t>(pb), 0);
    // FP4 region: (P+1)/2 bytes, low nibble even-k / high nibble odd-k.
    const int64_t params = static_cast<int64_t>(N) * K;
    for (int64_t i = 0; i < params; ++i) {
        uint8_t nib = quantize_e2m1(W[static_cast<size_t>(i)]);
        int64_t byte = i / 2;
        if (i % 2 == 0) buf[static_cast<size_t>(byte)] |= (nib & 0x0F);
        else            buf[static_cast<size_t>(byte)] |= (nib << 4);
    }
    // E4M3 group scales: [N, G] row-major all = 1.0, then reformat to Sm1xx.
    const int G = K / 16;
    std::vector<uint8_t> raw(static_cast<size_t>(N) * G, kE4M3_One);
    const int64_t wb = (params + 1) / 2;
    layerstorm::model::reformat_nvfp4_sfb(buf.data() + wb, raw.data(), N, G);
    // Tail scalars: weight_scale_2 at pb-8, input_scale at pb-4.
    std::memcpy(buf.data() + pb - 8, &ws2, 4);
    float input_scale = 1.0f;
    std::memcpy(buf.data() + pb - 4, &input_scale, 4);
    return buf;
}

// Build a packed nvfp4-sm1xx projection with NON-UNIFORM per-(row,group) E4M3
// group scales (random valid finite UE4M3 bytes), so the Sm1xx scale-slab slicing
// in the multi-node stage_proj_slice is actually EXERCISED — a uniform-scale pack
// (pack_proj) decodes every group to 1.0 and therefore masks any scale-slab offset
// bug at large K. The single-node device reads these same packed bytes, so it is
// the ground-truth reference for the multi-node parity check.
std::vector<uint8_t> pack_proj_varscale(const std::vector<float>& W, int N, int K,
                                        float ws2, std::mt19937& rng) {
    const int64_t pb = proj_bytes(N, K);
    std::vector<uint8_t> buf(static_cast<size_t>(pb), 0);
    const int64_t params = static_cast<int64_t>(N) * K;
    for (int64_t i = 0; i < params; ++i) {
        uint8_t nib = quantize_e2m1(W[static_cast<size_t>(i)]);
        int64_t byte = i / 2;
        if (i % 2 == 0) buf[static_cast<size_t>(byte)] |= (nib & 0x0F);
        else            buf[static_cast<size_t>(byte)] |= (nib << 4);
    }
    // Random valid finite UE4M3 group scales in [0x30, 0x42] (~0.5 .. ~2.25),
    // one per (row, group), then reformat to the Sm1xx interleaved layout.
    const int G = K / 16;
    std::vector<uint8_t> raw(static_cast<size_t>(N) * G);
    std::uniform_int_distribution<int> sd(0x30, 0x42);
    for (auto& b : raw) b = static_cast<uint8_t>(sd(rng));
    const int64_t wb = (params + 1) / 2;
    layerstorm::model::reformat_nvfp4_sfb(buf.data() + wb, raw.data(), N, G);
    std::memcpy(buf.data() + pb - 8, &ws2, 4);
    float input_scale = 1.0f;
    std::memcpy(buf.data() + pb - 4, &input_scale, 4);
    return buf;
}

// Scalar dequant of a packed projection (independent of the kernel).
float ref_dequant(const std::vector<uint8_t>& buf, int N, int K, int n, int k,
                  float ws2) {
    (void)N;
    int64_t i = static_cast<int64_t>(n) * K + k;
    uint8_t byte = buf[static_cast<size_t>(i / 2)];
    uint8_t nib = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
    const float mag = kMag[nib & 0x07];
    const float val = (nib & 0x08) ? -mag : mag;
    return val * 1.0f /* E4M3 group scale (0x38) */ * ws2;
}

}  // namespace

// ── parse_cpulist ────────────────────────────────────────────────────────────
TEST(NumaCpuExpert, ParseCpulist) {
    EXPECT_EQ(parse_cpulist("0-3,8,10-11"),
              (std::vector<int>{0, 1, 2, 3, 8, 10, 11}));
    EXPECT_EQ(parse_cpulist("5"), (std::vector<int>{5}));
    EXPECT_EQ(parse_cpulist(""), (std::vector<int>{}));
    EXPECT_EQ(parse_cpulist("2-2"), (std::vector<int>{2}));
    // dedup + sort tolerance
    EXPECT_EQ(parse_cpulist("3,1,2"), (std::vector<int>{1, 2, 3}));
}

// ── node_physical_cpus smoke (soft-pass on no-NUMA / no-sysfs) ───────────────
TEST(NumaCpuExpert, NodePhysicalCpusSmoke) {
    auto cpus = node_physical_cpus(0);
    if (cpus.empty()) {
        GTEST_SKIP() << "node 0 has no enumerable cpulist (no NUMA / no sysfs)";
    }
    // Ascending + unique (one logical CPU per physical core).
    for (size_t i = 1; i < cpus.size(); ++i) EXPECT_LT(cpus[i - 1], cpus[i]);
    // Pinning the calling thread to the first core must succeed or soft-fail.
    bool pinned = pin_thread_to_cpu(cpus.front());
    SUCCEED() << "physical cores=" << cpus.size()
              << " pin=" << (pinned ? "ok" : "failed(soft)");
}

// ── reserved leading core (C-6 QA c/d): bit-identical output, N-1 threads ─────
// LS_CPU_EXPERT_RESERVE_CORE holds the node's FIRST physical core out of the FFN
// pool for the daemon/orchestrator. This asserts the reserved-core layout (1) uses
// exactly one fewer participant, (2) exposes the node's leading core as reserved,
// (3) never pins a worker/leader onto it, and (4) produces the EXACT same output
// as the full-core pool — the FFN partitions by output ROW, so each row is written
// by exactly one participant and the result is participant-count-invariant.
TEST(NumaCpuExpert, ReservedLeadingCoreBitIdentical) {
    const auto phys = node_physical_cpus(0);
    if (phys.size() < 2) {
        GTEST_SKIP() << "node 0 has < 2 physical cores (no NUMA / no sysfs)";
    }

    constexpr int N = 100003;  // rows; prime so striping leaves a ragged tail
    // Row-partitioned independent write — the exact FFN discipline (partition by
    // output row N). out[i] is produced by exactly ONE participant regardless of
    // how many there are ⇒ the array is participant-count-invariant (bit-equal).
    auto run_rows = [&](NumaThreadPool& pool, std::vector<double>& out) {
        out.assign(N, -1.0);
        pool.parallel_for([&](int tid, int n, CpuBarrierState&) {
            for (int i = tid; i < N; i += n)
                out[static_cast<size_t>(i)] =
                    std::sin(i * 0.0011) * 1.5 - std::cos(i * 0.0007);
        });
    };

    NumaThreadPool full(/*numa_node=*/0, /*max_threads=*/0,
                        /*reserve_leading_cores=*/0);
    NumaThreadPool reserved(/*numa_node=*/0, /*max_threads=*/0,
                            /*reserve_leading_cores=*/1);

    // (1) exactly one fewer participant.
    EXPECT_EQ(reserved.num_threads(), full.num_threads() - 1);
    // (2) the node's leading physical core is the reserved one; full reserves none.
    EXPECT_TRUE(full.reserved_cpus().empty());
    ASSERT_EQ(reserved.reserved_cpus().size(), 1u);
    EXPECT_EQ(reserved.reserved_cpus().front(), phys.front());

    // (4) bit-identical output between the two layouts (memcmp-equal doubles).
    std::vector<double> out_full, out_reserved;
    run_rows(full, out_full);
    run_rows(reserved, out_reserved);
    ASSERT_EQ(out_full.size(), out_reserved.size());
    EXPECT_EQ(std::memcmp(out_full.data(), out_reserved.data(),
                          out_full.size() * sizeof(double)),
              0)
        << "reserved-core pool produced a different result than the full pool";

    // Reserving MORE cores than exist keeps at least one usable (clamp), and a
    // huge reserve collapses toward a single participant but still computes.
    NumaThreadPool clamp(/*numa_node=*/0, /*max_threads=*/0,
                         /*reserve_leading_cores=*/100000);
    EXPECT_GE(clamp.num_threads(), 1);
    std::vector<double> out_clamp;
    run_rows(clamp, out_clamp);
    EXPECT_EQ(std::memcmp(out_full.data(), out_clamp.data(),
                          out_full.size() * sizeof(double)),
              0);
}

// ── ik barrier: N-thread rendezvous, monotonic passed counter ────────────────
TEST(NumaCpuExpert, IkBarrierRendezvous) {
    constexpr int kThreads = 8;
    constexpr int kRounds = 50;
    CpuBarrierState st;
    st.n_threads = kThreads;
    std::atomic<int> stage_count{0};   // counts threads that crossed each round
    std::atomic<int> max_seen{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t) {
        ts.emplace_back([&] {
            for (int r = 0; r < kRounds; ++r) {
                // Before crossing round r, no thread may have entered round r+1:
                // assert all threads are within one round of each other.
                stage_count.fetch_add(1, std::memory_order_relaxed);
                barrier(st);
                int sc = stage_count.load(std::memory_order_relaxed);
                int prev = max_seen.load(std::memory_order_relaxed);
                while (sc > prev &&
                       !max_seen.compare_exchange_weak(prev, sc)) {}
            }
        });
    }
    for (auto& th : ts) th.join();
    // Every thread crossed every round: total entries == threads*rounds.
    EXPECT_EQ(stage_count.load(), kThreads * kRounds);
    // n_barrier resets to 0 after each round (sense-reversing two-counter).
    EXPECT_EQ(st.n_barrier.load(), 0);
    EXPECT_GE(st.n_barrier_passed.load(), kRounds);
}

// ── scale_byte_index inverts reformat_nvfp4_sfb ──────────────────────────────
TEST(NumaCpuExpert, ScaleByteIndexInvertsReformat) {
    const int N = 256, G = 128;  // K=2048
    std::vector<uint8_t> raw(static_cast<size_t>(N) * G);
    std::mt19937 rng(123);
    for (auto& b : raw) b = static_cast<uint8_t>(rng() & 0xFF);
    std::vector<uint8_t> interleaved(static_cast<size_t>(N) * G, 0);
    layerstorm::model::reformat_nvfp4_sfb(interleaved.data(), raw.data(), N, G);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            int64_t idx = scale_byte_index(n, g, G);
            ASSERT_GE(idx, 0);
            ASSERT_LT(idx, static_cast<int64_t>(interleaved.size()));
            EXPECT_EQ(interleaved[static_cast<size_t>(idx)],
                      raw[static_cast<size_t>(n) * G + g])
                << "n=" << n << " g=" << g;
        }
    }
}

// ── dequant_weight vs scalar reference ───────────────────────────────────────
TEST(NumaCpuExpert, DequantWeightMatchesScalar) {
    // Dims must honour the Sm1xx scale-atom alignment (N % 128 == 0, K % 64 == 0
    // so G = K/16 % 4 == 0): the interleaved scale region is then exactly N*G
    // bytes (== the kernel's (P+15)/16 sizing). This is the real engine
    // invariant (every projection is 2048/7168-rowed). Smaller N would make
    // reformat_nvfp4_sfb write n_pad*g_pad > N*G bytes (heap overflow).
    const int N = 128, K = 64;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-6.f, 6.f);
    std::vector<float> W(static_cast<size_t>(N) * K);
    for (auto& w : W) w = dist(rng);
    const float ws2 = 1.5f;
    auto buf = pack_proj(W, N, K, ws2);
    PackedProjection proj{buf.data(), static_cast<size_t>(proj_bytes(N, K)), N, K};
    EXPECT_FLOAT_EQ(read_weight_scale_2(buf.data(), proj.proj_bytes), ws2);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            EXPECT_FLOAT_EQ(dequant_weight(proj, n, k, ws2),
                            ref_dequant(buf, N, K, n, k, ws2))
                << "n=" << n << " k=" << k;
}

// ── cpu_nvfp4_grouped_gemm vs scalar dequant-weight matmul ───────────────────
TEST(NumaCpuExpert, Nvfp4GroupedGemmMatchesScalar) {
    const int N = 128, K = 64, M = 3;  // 3 permuted tokens, one expert
                                        // (N%128==0, K%64==0: Sm1xx-aligned)
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> wd(-4.f, 4.f), ad(-1.f, 1.f);
    std::vector<float> W(static_cast<size_t>(N) * K);
    for (auto& w : W) w = wd(rng);
    const float ws2 = 0.75f;
    auto buf = pack_proj(W, N, K, ws2);
    PackedProjection proj{buf.data(), static_cast<size_t>(proj_bytes(N, K)), N, K};
    CpuNvfp4ExpertWeights weights{&proj, 1};

    std::vector<uint16_t> A(static_cast<size_t>(M) * K);
    std::vector<float> Af(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        Af[i] = ad(rng);
        A[i] = f32_to_bf16(Af[i]);
        Af[i] = bf16_to_f32(A[i]);  // use the rounded value in the reference
    }
    std::vector<int32_t> offsets{0, M};
    std::vector<uint16_t> D(static_cast<size_t>(M) * N, 0);

    cpu_nvfp4_grouped_gemm(D.data(), A.data(), weights, offsets.data(), N, K,
                           /*elem_size_bytes=*/2, /*pool=*/nullptr);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += Af[static_cast<size_t>(m) * K + k] *
                       ref_dequant(buf, N, K, n, k, ws2);
            float got = bf16_to_f32(D[static_cast<size_t>(m) * N + n]);
            EXPECT_NEAR(got, acc, std::fabs(acc) * 0.02f + 0.05f)
                << "m=" << m << " n=" << n;
        }
    }
}

// ── M=1 fused decode->FMA GEMV path vs scalar + vs materialize path ──────────
//
// At exactly one token per expert, grouped_gemm_rows takes the fused
// decode->FMA path that never materializes the FP32 weight row. K=128 spans
// EIGHT 16-element groups (so the per-group scale + group striding is exercised,
// not just a single group). We check the M=1 result against (a) an independent
// scalar dequant-matmul reference and (b) the M>1 materialize-then-dot path fed
// the identical activation row — the two must agree bit-for-bit.
TEST(NumaCpuExpert, FusedGemvM1MatchesScalar) {
    const int N = 128, K = 128;  // N%128==0, K%64==0; 8 groups per row
    std::mt19937 rng(0xF11ED);
    std::uniform_real_distribution<float> wd(-4.f, 4.f), ad(-1.f, 1.f);
    std::vector<float> W(static_cast<size_t>(N) * K);
    for (auto& w : W) w = wd(rng);
    const float ws2 = 0.8125f;
    auto buf = pack_proj(W, N, K, ws2);
    PackedProjection proj{buf.data(), static_cast<size_t>(proj_bytes(N, K)), N, K};
    CpuNvfp4ExpertWeights weights{&proj, 1};

    std::vector<uint16_t> a(static_cast<size_t>(K));
    std::vector<float> af(K);
    for (int k = 0; k < K; ++k) {
        af[k] = ad(rng);
        a[k] = f32_to_bf16(af[k]);
        af[k] = bf16_to_f32(a[k]);
    }

    // (1) M=1 -> fused path.
    std::vector<int32_t> off1{0, 1};
    std::vector<uint16_t> D1(static_cast<size_t>(N), 0);
    cpu_nvfp4_grouped_gemm(D1.data(), a.data(), weights, off1.data(), N, K, 2,
                           nullptr);

    // (2) M=2 (token row duplicated) -> materialize-then-dot path.
    std::vector<uint16_t> a2(static_cast<size_t>(2) * K);
    for (int k = 0; k < K; ++k) { a2[k] = a[k]; a2[K + k] = a[k]; }
    std::vector<int32_t> off2{0, 2};
    std::vector<uint16_t> D2(static_cast<size_t>(2) * N, 0);
    cpu_nvfp4_grouped_gemm(D2.data(), a2.data(), weights, off2.data(), N, K, 2,
                           nullptr);

    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;  // independent scalar reference
        for (int k = 0; k < K; ++k)
            acc += af[k] * ref_dequant(buf, N, K, n, k, ws2);
        const float got = bf16_to_f32(D1[n]);
        EXPECT_NEAR(got, acc, std::fabs(acc) * 0.02f + 0.05f) << "scalar n=" << n;
        // Fused (M=1) must match materialize path (M=2) bit-for-bit.
        EXPECT_EQ(D1[n], D2[n]) << "fused vs materialize n=" << n;
    }
}

// ── cpu_swiglu vs scalar reference (interleaved gate|up) ─────────────────────
TEST(NumaCpuExpert, SwigluMatchesScalar) {
    const int T = 4, d = 16;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(-3.f, 3.f);
    std::vector<uint16_t> in(static_cast<size_t>(T) * 2 * d);
    for (auto& x : in) x = f32_to_bf16(dist(rng));
    std::vector<uint16_t> out(static_cast<size_t>(T) * d, 0);
    cpu_swiglu(out.data(), in.data(), /*up=*/nullptr, T, d,
               /*gate_up_interleaved=*/true, /*elem_size_bytes=*/2, nullptr);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < d; ++j) {
            float g = bf16_to_f32(in[static_cast<size_t>(t) * 2 * d + j]);
            float u = bf16_to_f32(in[static_cast<size_t>(t) * 2 * d + d + j]);
            float ref = (g / (1.0f + std::exp(-g))) * u;
            EXPECT_NEAR(bf16_to_f32(out[static_cast<size_t>(t) * d + j]), ref,
                        std::fabs(ref) * 0.01f + 0.02f);
        }
    }
}

// ── cpu_swiglu V4-4b clamp (llama.cpp DEEPSEEK4 semantics) ──────────────────
// out = SiLU(min(gate, L)) * clamp(up, -L, +L); gate lower end NOT clamped.
TEST(NumaCpuExpert, SwigluLimitClampMatchesReference) {
    const int T = 3, d = 32;
    const float L = 10.0f;
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> dist(-30.f, 30.f);  // exceeds ±L
    std::vector<uint16_t> in(static_cast<size_t>(T) * 2 * d);
    for (auto& x : in) x = f32_to_bf16(dist(rng));
    std::vector<uint16_t> out(static_cast<size_t>(T) * d, 0);
    cpu_swiglu(out.data(), in.data(), /*up=*/nullptr, T, d,
               /*gate_up_interleaved=*/true, /*elem_size_bytes=*/2, nullptr,
               /*swiglu_limit=*/L);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < d; ++j) {
            float g = bf16_to_f32(in[static_cast<size_t>(t) * 2 * d + j]);
            float u = bf16_to_f32(in[static_cast<size_t>(t) * 2 * d + d + j]);
            g = std::fmin(g, L);                       // max clamp only
            u = std::fmin(std::fmax(u, -L), L);        // two-sided
            const float ref = (g / (1.0f + std::exp(-g))) * u;
            EXPECT_NEAR(bf16_to_f32(out[static_cast<size_t>(t) * d + j]), ref,
                        std::fabs(ref) * 0.01f + 0.02f)
                << "t=" << t << " j=" << j;
        }
    }
    // limit=0 must remain bit-identical to the legacy no-limit call.
    std::vector<uint16_t> a(static_cast<size_t>(T) * d, 0),
        b(static_cast<size_t>(T) * d, 0);
    cpu_swiglu(a.data(), in.data(), nullptr, T, d, true, 2, nullptr);
    cpu_swiglu(b.data(), in.data(), nullptr, T, d, true, 2, nullptr, 0.0f);
    EXPECT_EQ(a, b);
}

// ── cpu_moe_permute / cpu_moe_unpermute round-trip ───────────────────────────
TEST(NumaCpuExpert, PermuteUnpermuteRoundTrip) {
    const int T = 4, topk = 2, H = 8, E = 3;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
    for (auto& x : hidden) x = f32_to_bf16(dist(rng));
    // Each token routes to topk distinct experts.
    std::vector<int32_t> topk_idx{0, 1,  1, 2,  0, 2,  2, 0};
    std::vector<float> topk_w{0.5f, 0.5f, 0.3f, 0.7f, 0.6f, 0.4f, 0.2f, 0.8f};

    std::vector<uint16_t> perm(static_cast<size_t>(T) * topk * H, 0);
    std::vector<int32_t> offsets(E + 1, 0);
    std::vector<int32_t> s2d(static_cast<size_t>(T) * topk, 0);
    std::vector<int32_t> pidx(static_cast<size_t>(T) * topk, 0);
    cpu_moe_permute(perm.data(), offsets.data(), s2d.data(), pidx.data(),
                    hidden.data(), topk_idx.data(), T, topk, H, E,
                    /*elem_size_bytes=*/2, nullptr);
    // offsets monotonic, total == T*topk.
    for (int e = 0; e < E; ++e) EXPECT_LE(offsets[e], offsets[e + 1]);
    EXPECT_EQ(offsets[E], T * topk);
    // Each permuted row equals its source token's hidden vector.
    for (int p = 0; p < T * topk; ++p) {
        int src = pidx[p];
        for (int j = 0; j < H; ++j)
            EXPECT_EQ(perm[static_cast<size_t>(p) * H + j],
                      hidden[static_cast<size_t>(src) * H + j]);
    }
    // Unpermute with identity expert output == permuted input, weighted sum
    // must reconstruct sum_k w * hidden[t].
    std::vector<uint16_t> out(static_cast<size_t>(T) * H, 0);
    cpu_moe_unpermute(out.data(), perm.data(), topk_w.data(), s2d.data(), T,
                      topk, H, /*elem_size_bytes=*/2, nullptr);
    for (int t = 0; t < T; ++t) {
        float wsum = 0.0f;
        for (int k = 0; k < topk; ++k) wsum += topk_w[t * topk + k];
        for (int j = 0; j < H; ++j) {
            float ref = wsum * bf16_to_f32(hidden[static_cast<size_t>(t) * H + j]);
            EXPECT_NEAR(bf16_to_f32(out[static_cast<size_t>(t) * H + j]), ref,
                        std::fabs(ref) * 0.02f + 0.02f);
        }
    }
}

// ── cpu_moe_unpermute_perslot: bit-exact per-slot combine (C-6 CPU offload) ───
//
// Validates the host per-slot unpermute used by the canonical (bit-exact) forced
// CPU-expert fold. The per-slot value must be a SINGLE fp32 multiply
// c_k = w_k * bf16_to_f32(expert_out_k) — for the fp32 payload stored raw, for
// the bf16 payload rounded once via f32_to_bf16 — mirroring the GPU kernels
// finalize_moe_routing_bf16_to_{fp32,bf16}_perslot. Dropped slots (masked expert
// ⇒ src_to_dest_map = -1) must write an all-zero row (so 0 + c = c is exact in
// the later gather/reduce). One token's second expert is masked (-1) to exercise
// the drop path.
TEST(NumaCpuExpert, UnpermutePerSlotBitExact) {
    const int T = 3, topk = 2, H = 10, E = 4;
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
    for (auto& x : hidden) x = f32_to_bf16(dist(rng));
    // token1 slot1 masked (-1) ⇒ dropped; the rest route to real experts.
    std::vector<int32_t> topk_idx{0, 2,  1, -1,  3, 0};
    std::vector<float>   topk_w{0.4f, 0.6f, 0.7f, 0.3f, 0.25f, 0.75f};

    std::vector<uint16_t> perm(static_cast<size_t>(T) * topk * H, 0);
    std::vector<int32_t> offsets(E + 1, 0);
    std::vector<int32_t> s2d(static_cast<size_t>(T) * topk, 0);
    std::vector<int32_t> pidx(static_cast<size_t>(T) * topk, 0);
    cpu_moe_permute(perm.data(), offsets.data(), s2d.data(), pidx.data(),
                    hidden.data(), topk_idx.data(), T, topk, H, E,
                    /*elem_size_bytes=*/2, nullptr);

    // Reference: exactly the GPU per-slot formula (single multiply).
    auto ref_slot = [&](int idx, int j) -> float {
        int32_t dest = s2d[idx];
        if (dest < 0) return 0.0f;
        return topk_w[idx] * bf16_to_f32(perm[static_cast<size_t>(dest) * H + j]);
    };

    // fp32 payload.
    std::vector<float> out_f32(static_cast<size_t>(T) * topk * H, -1.f);
    cpu_moe_unpermute_perslot(out_f32.data(), perm.data(), topk_w.data(),
                              s2d.data(), T, topk, H, /*fp32_output=*/true,
                              nullptr);
    // bf16 payload.
    std::vector<uint16_t> out_bf16(static_cast<size_t>(T) * topk * H, 0xFFFF);
    cpu_moe_unpermute_perslot(out_bf16.data(), perm.data(), topk_w.data(),
                              s2d.data(), T, topk, H, /*fp32_output=*/false,
                              nullptr);
    for (int idx = 0; idx < T * topk; ++idx) {
        for (int j = 0; j < H; ++j) {
            const float ref = ref_slot(idx, j);
            const size_t o = static_cast<size_t>(idx) * H + j;
            EXPECT_EQ(out_f32[o], ref) << "fp32 slot " << idx << " col " << j;
            EXPECT_EQ(out_bf16[o], f32_to_bf16(ref))
                << "bf16 slot " << idx << " col " << j;
        }
    }
    // The masked slot (token1 slot1 = idx 3) is an all-zero row in both.
    for (int j = 0; j < H; ++j) {
        EXPECT_EQ(out_f32[static_cast<size_t>(3) * H + j], 0.0f);
        EXPECT_EQ(out_bf16[static_cast<size_t>(3) * H + j], 0u);
    }
}

// ── AVX-512 vectorized path vs independent scalar reference ──────────────────
//
// The hand-written AVX-512 paths in nvfp4_cpu_kernel.cpp (dot_bf16_f32) and
// cpu_moe_kernels.cpp (silu512_ps, unpermute FMA loop) are #if'd on __AVX512F__,
// which is only defined when these TUs are compiled with -march=native (i.e. in
// LAYERSTORM_FAST_CPU_SOURCES). This test deliberately picks dims that *enter*
// the vector body AND leave a non-16-aligned scalar tail, so a mismatch between
// the vector body and the scalar remainder is caught:
//   * nvfp4 gemm:  K=64  -> two 32-wide AVX iterations (K>=32 requirement).
//   * swiglu:      d=24  -> one 16-wide AVX iteration + 8 scalar-tail lanes.
//   * unpermute:   H=40, topk=3 -> two 16-wide AVX FMA iters + 8 scalar lanes,
//                  with topk>1 so the weighted accumulate is genuinely additive.
// On a host without AVX-512 the same code exercises the scalar fallback, so this
// test is a correctness check on whichever path the compiler selected.
TEST(NumaCpuExpert, VectorizedPathMatchesScalar) {
    std::mt19937 rng(20260618);
    std::uniform_real_distribution<float> wd(-4.f, 4.f), ad(-1.f, 1.f);

    // ---- nvfp4 grouped GEMM, K=64 (AVX 32-wide body), 2 experts, 5 tokens ----
    {
        const int N = 128, K = 64;  // Sm1xx-aligned (N%128==0, K%64==0)
        const int E = 2;
        std::vector<float> W0(static_cast<size_t>(N) * K), W1(W0.size());
        for (auto& w : W0) w = wd(rng);
        for (auto& w : W1) w = wd(rng);
        const float ws2 = 0.875f;
        auto b0 = pack_proj(W0, N, K, ws2);
        auto b1 = pack_proj(W1, N, K, ws2);
        PackedProjection p0{b0.data(), static_cast<size_t>(proj_bytes(N, K)), N, K};
        PackedProjection p1{b1.data(), static_cast<size_t>(proj_bytes(N, K)), N, K};
        PackedProjection projs[2]{p0, p1};
        CpuNvfp4ExpertWeights weights{projs, E};

        // Expert 0 gets 3 tokens, expert 1 gets 2 tokens (expert-contiguous A).
        const int M = 5;
        std::vector<int32_t> offsets{0, 3, 5};
        std::vector<uint16_t> A(static_cast<size_t>(M) * K);
        std::vector<float> Af(A.size());
        for (size_t i = 0; i < A.size(); ++i) {
            Af[i] = ad(rng);
            A[i] = f32_to_bf16(Af[i]);
            Af[i] = bf16_to_f32(A[i]);
        }
        std::vector<uint16_t> D(static_cast<size_t>(M) * N, 0);
        cpu_nvfp4_grouped_gemm(D.data(), A.data(), weights, offsets.data(), N, K,
                               /*elem_size_bytes=*/2, /*pool=*/nullptr);

        const std::vector<uint8_t>* bufs[2]{&b0, &b1};
        for (int e = 0; e < E; ++e) {
            for (int m = offsets[e]; m < offsets[e + 1]; ++m) {
                for (int n = 0; n < N; ++n) {
                    float acc = 0.0f;  // independent scalar dequant-matmul ref
                    for (int k = 0; k < K; ++k)
                        acc += Af[static_cast<size_t>(m) * K + k] *
                               ref_dequant(*bufs[e], N, K, n, k, ws2);
                    float got = bf16_to_f32(D[static_cast<size_t>(m) * N + n]);
                    EXPECT_NEAR(got, acc, std::fabs(acc) * 0.02f + 0.05f)
                        << "gemm e=" << e << " m=" << m << " n=" << n;
                }
            }
        }
    }

    // ---- swiglu, d=24 (16-wide AVX iter + 8 scalar tail) ----
    {
        const int T = 3, d = 24;
        std::vector<uint16_t> in(static_cast<size_t>(T) * 2 * d);
        for (auto& x : in) x = f32_to_bf16(ad(rng) * 3.0f);
        std::vector<uint16_t> out(static_cast<size_t>(T) * d, 0);
        cpu_swiglu(out.data(), in.data(), /*up=*/nullptr, T, d,
                   /*gate_up_interleaved=*/true, /*elem_size_bytes=*/2, nullptr);
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < d; ++j) {
                float g = bf16_to_f32(in[static_cast<size_t>(t) * 2 * d + j]);
                float u = bf16_to_f32(in[static_cast<size_t>(t) * 2 * d + d + j]);
                float ref = (g / (1.0f + std::exp(-g))) * u;  // scalar SiLU*up
                EXPECT_NEAR(bf16_to_f32(out[static_cast<size_t>(t) * d + j]), ref,
                            std::fabs(ref) * 0.01f + 0.02f)
                    << "swiglu t=" << t << " j=" << j;
            }
        }
    }

    // ---- unpermute, H=40, topk=3 (two 16-wide AVX iters + 8 scalar tail) ----
    {
        const int T = 4, topk = 3, H = 40;
        // permuted_output rows in arbitrary order; src_to_dest_map picks them.
        const int P = T * topk;
        std::vector<uint16_t> perm(static_cast<size_t>(P) * H);
        std::vector<float> perm_f(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            perm_f[i] = ad(rng) * 2.0f;
            perm[i] = f32_to_bf16(perm_f[i]);
            perm_f[i] = bf16_to_f32(perm[i]);
        }
        // Each token's topk slots map to distinct permuted rows; include one
        // dropped (-1) slot to exercise the skip branch.
        std::vector<int32_t> s2d(P);
        for (int i = 0; i < P; ++i) s2d[i] = (P - 1) - i;  // reverse mapping
        s2d[2] = -1;                                       // token0, k2 dropped
        std::vector<float> w(P);
        for (auto& x : w) x = std::fabs(ad(rng)) + 0.1f;

        std::vector<uint16_t> out(static_cast<size_t>(T) * H, 0);
        cpu_moe_unpermute(out.data(), perm.data(), w.data(), s2d.data(), T, topk,
                          H, /*elem_size_bytes=*/2, nullptr);

        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < H; ++j) {
                float ref = 0.0f;  // independent scalar weighted accumulate
                for (int k = 0; k < topk; ++k) {
                    int idx = t * topk + k;
                    int32_t dest = s2d[idx];
                    if (dest < 0) continue;
                    ref += w[idx] * perm_f[static_cast<size_t>(dest) * H + j];
                }
                float got = bf16_to_f32(out[static_cast<size_t>(t) * H + j]);
                EXPECT_NEAR(got, ref, std::fabs(ref) * 0.02f + 0.02f)
                    << "unpermute t=" << t << " j=" << j;
            }
        }
    }
}

// ── End-to-end single-expert FFN through NumaCpuExpertDevice ─────────────────
//
// One token routed to one expert. Builds gate/up/down packed projections,
// drives the device's 9 virtuals exactly as the dispatcher would
// (permute -> gate -> up -> swiglu -> down -> unpermute), and compares the host
// moe_output to a scalar golden computed with the same dequant-weight policy.
TEST(NumaCpuExpert, DeviceEndToEndFfn) {
    // Sm1xx scale atoms require N % 128 == 0 and K % 64 == 0 for every
    // projection. Gate/up are [I, H] and down is [H, I], so both H and I must be
    // multiples of 128 (and of 64 as K). Use the smallest valid square dims.
    const int H = 128;  // hidden_dim
    const int I = 128;  // moe_intermediate
    const int T = 1, topk = 1, E = 1;

    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> wd(-2.f, 2.f), ad(-1.f, 1.f);

    auto rand_w = [&](int N, int K) {
        std::vector<float> w(static_cast<size_t>(N) * K);
        for (auto& x : w) x = wd(rng);
        return w;
    };
    std::vector<float> Wg = rand_w(I, H), Wu = rand_w(I, H), Wd = rand_w(H, I);
    const float ws2 = 1.0f;
    auto gate_buf = pack_proj(Wg, I, H, ws2);
    auto up_buf   = pack_proj(Wu, I, H, ws2);
    auto down_buf = pack_proj(Wd, H, I, ws2);

    // Input hidden [T,H] BF16.
    std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
    std::vector<float> hidden_f(hidden.size());
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden_f[i] = ad(rng);
        hidden[i] = f32_to_bf16(hidden_f[i]);
        hidden_f[i] = bf16_to_f32(hidden[i]);
    }

    // Build the device with no NumaManager (device_alloc => malloc) on node 0.
    config::GpuRef ref{};
    ref.position = 0; ref.id = 0; ref.type = config::GpuType::cpu;
    NumaCpuExpertDeps deps{};
    deps.dims = CpuExpertModelDims{H, I, topk, E};
    deps.max_threads = 2;
    auto dev = make_numa_cpu_expert_device(ref, deps);
    dev->set_device();

    auto* perm   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * topk * H));
    auto* gate_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * topk * I));
    auto* up_o   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * topk * I));
    // SwiGLU needs interleaved gate|up [T*topk, 2I].
    auto* gu     = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * topk * 2 * I));
    auto* swig   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * topk * I));
    auto* down_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * topk * H));
    auto* moe_o  = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * H));
    ASSERT_NE(perm, nullptr);
    ASSERT_NE(moe_o, nullptr);

    std::vector<int32_t> topk_idx{0};
    std::vector<float> topk_w{1.0f};
    std::vector<int32_t> offsets(E + 1, 0), s2d(T * topk, 0), pidx(T * topk, 0);

    // Step 1: permute.
    dev->moe_permute(perm, offsets.data(), s2d.data(), pidx.data(),
                     hidden.data(), topk_idx.data(), T, topk, H, E, 2,
                     nullptr, nullptr);
    ASSERT_EQ(offsets[E], T * topk);

    // Helper to drive nvfp4_grouped_gemm via the generic param struct.
    auto run_gemm = [&](void* D, const void* A, const uint8_t* w, int N, int K) {
        const void* bptr = w;
        Nvfp4GroupedGemmParams p{};
        p.num_experts = E; p.N = N; p.K = K;
        p.A_base = A; p.D_base = D;
        p.expert_offsets = offsets.data();
        p.B_ptrs = &bptr;
        p.output_dtype = GemmOutputDtype::kBFloat16;
        dev->nvfp4_grouped_gemm(p, nullptr, 0, nullptr);
    };

    // Step 2/3: gate + up GEMM (N=I, K=H).
    run_gemm(gate_o, perm, gate_buf.data(), I, H);
    run_gemm(up_o,   perm, up_buf.data(),   I, H);

    // Interleave gate|up into gu for the device's fused_swiglu contract.
    for (int t = 0; t < T * topk; ++t) {
        for (int j = 0; j < I; ++j) {
            gu[static_cast<size_t>(t) * 2 * I + j] = gate_o[static_cast<size_t>(t) * I + j];
            gu[static_cast<size_t>(t) * 2 * I + I + j] = up_o[static_cast<size_t>(t) * I + j];
        }
    }
    FusedSwigluParams sp{T * topk, I};
    dev->fused_swiglu(swig, gu, sp, 2, nullptr);

    // Step 5: down GEMM (N=H, K=I).
    run_gemm(down_o, swig, down_buf.data(), H, I);

    // Step 6: unpermute.
    dev->moe_unpermute(moe_o, down_o, topk_w.data(), s2d.data(), T, topk, H, 2,
                       nullptr);

    // ── Scalar golden ────────────────────────────────────────────────────────
    auto dequant_mm = [&](const std::vector<uint8_t>& wbuf, int N, int K,
                          const std::vector<float>& a) {
        std::vector<float> o(static_cast<size_t>(N), 0.0f);
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += a[static_cast<size_t>(k)] * ref_dequant(wbuf, N, K, n, k, ws2);
            o[static_cast<size_t>(n)] = bf16_to_f32(f32_to_bf16(acc));
        }
        return o;
    };
    std::vector<float> in_f(hidden_f.begin(), hidden_f.begin() + H);
    auto g = dequant_mm(gate_buf, I, H, in_f);
    auto u = dequant_mm(up_buf, I, H, in_f);
    std::vector<float> sw(static_cast<size_t>(I));
    for (int j = 0; j < I; ++j) {
        float gv = g[static_cast<size_t>(j)], uv = u[static_cast<size_t>(j)];
        sw[static_cast<size_t>(j)] =
            bf16_to_f32(f32_to_bf16((gv / (1.0f + std::exp(-gv))) * uv));
    }
    auto d = dequant_mm(down_buf, H, I, sw);

    for (int j = 0; j < H; ++j) {
        float ref = bf16_to_f32(f32_to_bf16(1.0f * d[static_cast<size_t>(j)]));
        float got = bf16_to_f32(moe_o[static_cast<size_t>(j)]);
        EXPECT_NEAR(got, ref, std::fabs(ref) * 0.05f + 0.1f) << "j=" << j;
    }

    dev->device_free(perm); dev->device_free(gate_o); dev->device_free(up_o);
    dev->device_free(gu); dev->device_free(swig); dev->device_free(down_o);
    dev->device_free(moe_o);
}

// ── Throughput benchmark (prints to stderr; convention: gtest-as-benchmark) ──
//
// Drives the FULL CPU expert-FFN chain (permute -> gate -> up -> swiglu -> down
// -> unpermute) through NumaCpuExpertDevice on REALISTIC routed-expert dims
// (DeepSeek V3.2: hidden=7168, moe_intermediate=2048, topk=8) and times it for a
// sweep of token counts. It is dequant-bandwidth-bound: each output row reads a
// full packed nvfp4 projection, so the dominant cost is streaming/dequantizing
// the expert weights. The test ASSERTS only basic sanity (finite output, tok/s
// > 0); the printed numbers are the point. Follows the same convention as
// GpuLoaderSolver.N8M10ExampleVsGreedyAndBenchmark (warmup + timed loop +
// fprintf to stderr).
TEST(NumaCpuExpert, FfnThroughputBenchmark) {
    // Realistic routed-expert dims.
    const int H = 7168;   // hidden_dim
    const int I = 2048;   // moe_intermediate
    const int topk = 8;
    const int E = 16;     // distinct experts available for routing

    // ── Pack one set of expert weights (gate [I,H], up [I,H], down [H,I]) per
    // expert, with random-but-valid sm1xx-packed nvfp4 + non-trivial ws2. Reuse
    // the file's pack_proj helper (do NOT rewrite packing). ─────────────────────
    std::mt19937 rng(0xC6FFEE);
    std::uniform_real_distribution<float> wd(-2.f, 2.f);
    auto rand_w = [&](int N, int K) {
        std::vector<float> w(static_cast<size_t>(N) * K);
        for (auto& x : w) x = wd(rng);
        return w;
    };

    const int64_t gate_pb = proj_bytes(I, H);  // == up_pb
    const int64_t down_pb = proj_bytes(H, I);
    const int64_t per_expert_bytes = 2 * gate_pb + down_pb;  // gate+up+down

    std::vector<std::vector<uint8_t>> gate_w(E), up_w(E), down_w(E);
    for (int e = 0; e < E; ++e) {
        // Non-trivial weight_scale_2 per expert so dequant isn't a uniform 1.0.
        const float ws2_g = 0.75f + 0.05f * static_cast<float>(e % 5);
        const float ws2_u = 0.80f + 0.03f * static_cast<float>(e % 7);
        const float ws2_d = 0.90f + 0.02f * static_cast<float>(e % 3);
        gate_w[e] = pack_proj(rand_w(I, H), I, H, ws2_g);
        up_w[e]   = pack_proj(rand_w(I, H), I, H, ws2_u);
        down_w[e] = pack_proj(rand_w(H, I), H, I, ws2_d);
    }

    // ── Build the device (node 0, no NumaManager => malloc scratch). ────────────
    config::GpuRef ref{};
    ref.position = 0; ref.id = 0; ref.type = config::GpuType::cpu;
    NumaCpuExpertDeps deps{};
    deps.dims = CpuExpertModelDims{H, I, topk, E};
    deps.max_threads = 0;  // <=0 => all physical cores on the node
    auto dev = make_numa_cpu_expert_device(ref, deps);
    dev->set_device();

    // Learn the thread count the device's pool would actually use, and whether
    // work pinned to physical cores of node 0 or fell back to unpinned. The
    // device owns a private pool, so we construct an identical one to query it
    // (same node + max_threads => same n_threads()).
    cpu::NumaThreadPool probe_pool(/*numa_node=*/0, deps.max_threads);
    const int threads_used = probe_pool.num_threads();
    const auto node0_cores = cpu::node_physical_cpus(0);
    const bool pinned = !node0_cores.empty();
    const unsigned hw_cores = std::thread::hardware_concurrency();
#if defined(__AVX512F__)
    const bool avx512 = true;
#else
    const bool avx512 = false;
#endif

    fprintf(stderr,
            "\n==== NumaCpuExpert.FfnThroughputBenchmark ====\n"
            "  dims: H=%d I=%d topk=%d  experts_available=%d\n"
            "  per-expert packed bytes (gate+up+down) = %lld  "
            "(gate=%lld up=%lld down=%lld)\n"
            "  host: hardware_concurrency=%u  node0_physical_cores=%zu  "
            "threads_used=%d  pinned=%s  AVX-512_compiled=%s\n",
            H, I, topk, E,
            static_cast<long long>(per_expert_bytes),
            static_cast<long long>(gate_pb),
            static_cast<long long>(gate_pb),
            static_cast<long long>(down_pb),
            hw_cores, node0_cores.size(), threads_used,
            pinned ? "yes" : "no(unpinned-fallback)",
            avx512 ? "yes" : "no");

    const int T_sweep[] = {1, 8, 32};
    for (int T : T_sweep) {
        // Route T tokens across E experts, topk distinct experts each.
        std::vector<int32_t> topk_idx(static_cast<size_t>(T) * topk);
        std::vector<float> topk_w(static_cast<size_t>(T) * topk);
        std::mt19937 rrng(0xBEEF + static_cast<unsigned>(T));
        for (int t = 0; t < T; ++t) {
            // Pick topk distinct experts (Fisher-Yates partial shuffle).
            std::vector<int> pool(static_cast<size_t>(E));
            for (int e = 0; e < E; ++e) pool[static_cast<size_t>(e)] = e;
            for (int k = 0; k < topk; ++k) {
                std::uniform_int_distribution<int> pick(k, E - 1);
                std::swap(pool[static_cast<size_t>(k)],
                          pool[static_cast<size_t>(pick(rrng))]);
                topk_idx[static_cast<size_t>(t) * topk + k] =
                    pool[static_cast<size_t>(k)];
            }
            // Routing weights ~ softmax-ish positive; normalized to sum 1.
            float wsum = 0.0f;
            for (int k = 0; k < topk; ++k) {
                float v = std::fabs(wd(rrng)) + 0.05f;
                topk_w[static_cast<size_t>(t) * topk + k] = v;
                wsum += v;
            }
            for (int k = 0; k < topk; ++k)
                topk_w[static_cast<size_t>(t) * topk + k] /= wsum;
        }

        // Input hidden [T,H] BF16.
        std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
        std::uniform_real_distribution<float> ad(-1.f, 1.f);
        for (auto& x : hidden) x = f32_to_bf16(ad(rng));

        // Scratch (device-allocated, node-local when a NumaManager is present).
        const int P = T * topk;  // permuted rows
        auto* perm   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * H));
        auto* gate_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
        auto* up_o   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
        auto* gu     = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * 2 * I));
        auto* swig   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
        auto* down_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * H));
        auto* moe_o  = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * H));
        ASSERT_NE(perm, nullptr);
        ASSERT_NE(moe_o, nullptr);

        std::vector<int32_t> offsets(E + 1, 0), s2d(P, 0), pidx(P, 0);

        // One full FFN chain. The grouped GEMMs receive ONLY the active experts
        // (those with >=1 token) as a compacted B_ptrs/offsets list — exactly how
        // the dispatcher drives the device. We (re)derive the active-expert
        // compaction from the permute offsets each iteration.
        std::vector<const void*> gate_ptrs, up_ptrs, down_ptrs;
        std::vector<int32_t> active_offsets;
        int active_experts = 0;

        auto build_active = [&]() {
            gate_ptrs.clear(); up_ptrs.clear(); down_ptrs.clear();
            active_offsets.clear();
            active_offsets.push_back(0);
            for (int e = 0; e < E; ++e) {
                const int m0 = offsets[e], m1 = offsets[e + 1];
                if (m1 <= m0) continue;
                gate_ptrs.push_back(gate_w[e].data());
                up_ptrs.push_back(up_w[e].data());
                down_ptrs.push_back(down_w[e].data());
                active_offsets.push_back(m1);  // offsets are already cumulative
            }
            active_experts = static_cast<int>(gate_ptrs.size());
        };

        auto run_gemm = [&](void* D, const void* A,
                            std::vector<const void*>& bptrs, int N, int K) {
            Nvfp4GroupedGemmParams p{};
            p.num_experts = active_experts; p.N = N; p.K = K;
            p.A_base = A; p.D_base = D;
            p.expert_offsets = active_offsets.data();
            p.B_ptrs = bptrs.data();
            p.output_dtype = GemmOutputDtype::kBFloat16;
            dev->nvfp4_grouped_gemm(p, nullptr, 0, nullptr);
        };

        auto run_chain = [&]() {
            dev->moe_permute(perm, offsets.data(), s2d.data(), pidx.data(),
                             hidden.data(), topk_idx.data(), T, topk, H, E, 2,
                             nullptr, nullptr);
            build_active();
            run_gemm(gate_o, perm, gate_ptrs, I, H);
            run_gemm(up_o,   perm, up_ptrs,   I, H);
            for (int p = 0; p < P; ++p) {
                for (int j = 0; j < I; ++j) {
                    gu[static_cast<size_t>(p) * 2 * I + j] =
                        gate_o[static_cast<size_t>(p) * I + j];
                    gu[static_cast<size_t>(p) * 2 * I + I + j] =
                        up_o[static_cast<size_t>(p) * I + j];
                }
            }
            FusedSwigluParams sp{P, I};
            dev->fused_swiglu(swig, gu, sp, 2, nullptr);
            run_gemm(down_o, swig, down_ptrs, H, I);
            dev->moe_unpermute(moe_o, down_o, topk_w.data(), s2d.data(), T, topk,
                               H, 2, nullptr);
        };

        // Warmup (also establishes active_experts + sanity-checks the output).
        run_chain();
        ASSERT_EQ(offsets[E], P);
        ASSERT_GT(active_experts, 0);
        bool all_finite = true;
        for (int j = 0; j < T * H; ++j)
            if (!std::isfinite(bf16_to_f32(moe_o[j]))) { all_finite = false; break; }
        ASSERT_TRUE(all_finite) << "moe_output not finite (T=" << T << ")";

        // Pick N so the timed region runs ~200ms; cap to keep < ~30s wall. First
        // measure a single iteration to size the loop.
        auto t_probe0 = std::chrono::steady_clock::now();
        run_chain();
        auto t_probe1 = std::chrono::steady_clock::now();
        const double one_us =
            std::chrono::duration<double, std::micro>(t_probe1 - t_probe0).count();
        const double target_us = 200000.0;  // ~200 ms
        int iters = static_cast<int>(target_us / std::max(one_us, 1.0));
        if (iters < 1) iters = 1;
        // Cap so worst case stays under ~30s.
        const int max_iters = static_cast<int>(30e6 / std::max(one_us, 1.0));
        if (iters > max_iters) iters = std::max(1, max_iters);

        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) run_chain();
        auto t1 = std::chrono::steady_clock::now();
        const double total_us =
            std::chrono::duration<double, std::micro>(t1 - t0).count();

        const double us_per_iter = total_us / iters;
        const double us_per_token = us_per_iter / T;
        const double tokens_per_sec = 1e6 / us_per_token;

        // Weight bytes dequanted per input token = topk experts * per-expert
        // (gate+up+down) packed bytes. (Per iteration, the GEMMs read exactly
        // P=T*topk permuted rows worth of per-expert weights once.)
        const double wbytes_per_token =
            static_cast<double>(topk) * static_cast<double>(per_expert_bytes);
        const double wbytes_per_iter = wbytes_per_token * T;
        const double gbps = (wbytes_per_iter / (us_per_iter * 1e-6)) / 1e9;

        ASSERT_GT(tokens_per_sec, 0.0);

        fprintf(stderr,
                "  T=%-3d  active_experts=%-3d  iters=%-6d  "
                "us/iter=%9.1f  us/token=%9.2f  tok/s=%9.1f  "
                "dequant=%6.2f GB/s\n",
                T, active_experts, iters, us_per_iter, us_per_token,
                tokens_per_sec, gbps);

        dev->device_free(perm);   dev->device_free(gate_o);
        dev->device_free(up_o);   dev->device_free(gu);
        dev->device_free(swig);   dev->device_free(down_o);
        dev->device_free(moe_o);
    }
    fprintf(stderr, "==============================================\n");
}

// ── Single-expert (topk sweep) breakdown at T=1 ──────────────────────────────
//
// Isolates the 1-layer x 1-token x 1-expert unit. Part A sweeps topk={1,2,4,8}
// at T=1 and reports us/iter + tok/s, so the marginal per-expert cost and the
// fixed (permute/unpermute/fork-join) intercept can be read off the topk1 vs
// topk8 delta. Part B, for the topk=1/T=1 case, times EACH stage of the chain
// separately (moe_permute, gate gemm, up gemm, gate/up->gu interleave,
// fused_swiglu, down gemm, moe_unpermute) and prints us + % of total — this
// shows whether the three nvfp4 GEMMs dominate (dequant-bound, scales with
// weight-element count) vs the permute/unpermute/overhead.
//
// Uses the same realistic routed-expert dims, the same pack_proj helper, and
// the same chrono approach as FfnThroughputBenchmark.
TEST(NumaCpuExpert, SingleExpertBreakdown) {
    const int H = 7168;   // hidden_dim
    const int I = 2048;   // moe_intermediate
    const int E = 16;     // distinct experts available for routing
    const int T = 1;

    std::mt19937 rng(0x51A2E);
    std::uniform_real_distribution<float> wd(-2.f, 2.f);
    auto rand_w = [&](int N, int K) {
        std::vector<float> w(static_cast<size_t>(N) * K);
        for (auto& x : w) x = wd(rng);
        return w;
    };

    const int64_t gate_pb = proj_bytes(I, H);  // == up_pb
    const int64_t down_pb = proj_bytes(H, I);
    const int64_t per_expert_bytes = 2 * gate_pb + down_pb;  // gate+up+down

    // Pack E experts' gate/up/down projections (same recipe as the throughput
    // benchmark, non-trivial per-expert ws2).
    std::vector<std::vector<uint8_t>> gate_w(E), up_w(E), down_w(E);
    for (int e = 0; e < E; ++e) {
        const float ws2_g = 0.75f + 0.05f * static_cast<float>(e % 5);
        const float ws2_u = 0.80f + 0.03f * static_cast<float>(e % 7);
        const float ws2_d = 0.90f + 0.02f * static_cast<float>(e % 3);
        gate_w[e] = pack_proj(rand_w(I, H), I, H, ws2_g);
        up_w[e]   = pack_proj(rand_w(I, H), I, H, ws2_u);
        down_w[e] = pack_proj(rand_w(H, I), H, I, ws2_d);
    }

    // Device: node 0, all physical cores (max_threads<=0).
    config::GpuRef ref{};
    ref.position = 0; ref.id = 0; ref.type = config::GpuType::cpu;
    NumaCpuExpertDeps deps{};
    deps.dims = CpuExpertModelDims{H, I, /*topk=*/8, E};
    deps.max_threads = 0;
    auto dev = make_numa_cpu_expert_device(ref, deps);
    dev->set_device();

    cpu::NumaThreadPool probe_pool(/*numa_node=*/0, deps.max_threads);
    const int threads_used = probe_pool.num_threads();
    const auto node0_cores = cpu::node_physical_cpus(0);
    const bool pinned = !node0_cores.empty();
    const unsigned hw_cores = std::thread::hardware_concurrency();
#if defined(__AVX512F__)
    const bool avx512 = true;
#else
    const bool avx512 = false;
#endif

    fprintf(stderr,
            "\n==== NumaCpuExpert.SingleExpertBreakdown ====\n"
            "  dims: H=%d I=%d  experts_available=%d  T=%d\n"
            "  per-expert packed bytes (gate+up+down) = %lld  "
            "(gate=%lld up=%lld down=%lld)\n"
            "  host: hardware_concurrency=%u  node0_physical_cores=%zu  "
            "threads_used=%d  pinned=%s  AVX-512_compiled=%s\n",
            H, I, E, T,
            static_cast<long long>(per_expert_bytes),
            static_cast<long long>(gate_pb),
            static_cast<long long>(gate_pb),
            static_cast<long long>(down_pb),
            hw_cores, node0_cores.size(), threads_used,
            pinned ? "yes" : "no(unpinned-fallback)",
            avx512 ? "yes" : "no");

    // ── Part A: topk sweep at T=1 ─────────────────────────────────────────────
    fprintf(stderr, "  -- topk sweep (T=1, full chain) --\n");
    const int topk_sweep[] = {1, 2, 4, 8};
    double us_topk1 = 0.0, us_topk8 = 0.0;
    for (int topk : topk_sweep) {
        // Route the single token to `topk` distinct experts.
        std::vector<int32_t> topk_idx(static_cast<size_t>(topk));
        std::vector<float> topk_w(static_cast<size_t>(topk));
        {
            std::vector<int> pool(static_cast<size_t>(E));
            for (int e = 0; e < E; ++e) pool[static_cast<size_t>(e)] = e;
            std::mt19937 rrng(0xC0DE + static_cast<unsigned>(topk));
            float wsum = 0.0f;
            for (int k = 0; k < topk; ++k) {
                std::uniform_int_distribution<int> pick(k, E - 1);
                std::swap(pool[static_cast<size_t>(k)],
                          pool[static_cast<size_t>(pick(rrng))]);
                topk_idx[static_cast<size_t>(k)] = pool[static_cast<size_t>(k)];
                float v = std::fabs(wd(rrng)) + 0.05f;
                topk_w[static_cast<size_t>(k)] = v;
                wsum += v;
            }
            for (int k = 0; k < topk; ++k) topk_w[static_cast<size_t>(k)] /= wsum;
        }

        const int P = T * topk;  // permuted rows == topk (T=1)
        std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
        std::uniform_real_distribution<float> ad(-1.f, 1.f);
        for (auto& x : hidden) x = f32_to_bf16(ad(rng));

        auto* perm   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * H));
        auto* gate_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
        auto* up_o   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
        auto* gu     = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * 2 * I));
        auto* swig   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
        auto* down_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * H));
        auto* moe_o  = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * H));
        ASSERT_NE(perm, nullptr);
        ASSERT_NE(moe_o, nullptr);

        std::vector<int32_t> offsets(E + 1, 0), s2d(P, 0), pidx(P, 0);
        std::vector<const void*> gate_ptrs, up_ptrs, down_ptrs;
        std::vector<int32_t> active_offsets;
        int active_experts = 0;

        auto build_active = [&]() {
            gate_ptrs.clear(); up_ptrs.clear(); down_ptrs.clear();
            active_offsets.clear(); active_offsets.push_back(0);
            for (int e = 0; e < E; ++e) {
                const int m0 = offsets[e], m1 = offsets[e + 1];
                if (m1 <= m0) continue;
                gate_ptrs.push_back(gate_w[e].data());
                up_ptrs.push_back(up_w[e].data());
                down_ptrs.push_back(down_w[e].data());
                active_offsets.push_back(m1);
            }
            active_experts = static_cast<int>(gate_ptrs.size());
        };
        auto run_gemm = [&](void* D, const void* A,
                            std::vector<const void*>& bptrs, int N, int K) {
            Nvfp4GroupedGemmParams p{};
            p.num_experts = active_experts; p.N = N; p.K = K;
            p.A_base = A; p.D_base = D;
            p.expert_offsets = active_offsets.data();
            p.B_ptrs = bptrs.data();
            p.output_dtype = GemmOutputDtype::kBFloat16;
            dev->nvfp4_grouped_gemm(p, nullptr, 0, nullptr);
        };
        auto run_chain = [&]() {
            dev->moe_permute(perm, offsets.data(), s2d.data(), pidx.data(),
                             hidden.data(), topk_idx.data(), T, topk, H, E, 2,
                             nullptr, nullptr);
            build_active();
            run_gemm(gate_o, perm, gate_ptrs, I, H);
            run_gemm(up_o,   perm, up_ptrs,   I, H);
            for (int p = 0; p < P; ++p)
                for (int j = 0; j < I; ++j) {
                    gu[static_cast<size_t>(p) * 2 * I + j] =
                        gate_o[static_cast<size_t>(p) * I + j];
                    gu[static_cast<size_t>(p) * 2 * I + I + j] =
                        up_o[static_cast<size_t>(p) * I + j];
                }
            FusedSwigluParams sp{P, I};
            dev->fused_swiglu(swig, gu, sp, 2, nullptr);
            run_gemm(down_o, swig, down_ptrs, H, I);
            dev->moe_unpermute(moe_o, down_o, topk_w.data(), s2d.data(), T, topk,
                               H, 2, nullptr);
        };

        run_chain();  // warmup + establish active_experts
        ASSERT_EQ(offsets[E], P);
        ASSERT_EQ(active_experts, topk);

        auto t_probe0 = std::chrono::steady_clock::now();
        run_chain();
        auto t_probe1 = std::chrono::steady_clock::now();
        const double one_us =
            std::chrono::duration<double, std::micro>(t_probe1 - t_probe0).count();
        int iters = static_cast<int>(200000.0 / std::max(one_us, 1.0));
        if (iters < 1) iters = 1;
        const int max_iters = static_cast<int>(30e6 / std::max(one_us, 1.0));
        if (iters > max_iters) iters = std::max(1, max_iters);

        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) run_chain();
        auto t1 = std::chrono::steady_clock::now();
        const double us_per_iter =
            std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        const double tokens_per_sec = 1e6 / (us_per_iter / T);
        ASSERT_GT(tokens_per_sec, 0.0);

        if (topk == 1) us_topk1 = us_per_iter;
        if (topk == 8) us_topk8 = us_per_iter;

        fprintf(stderr,
                "  topk=%-2d  active_experts=%-2d  iters=%-6d  "
                "us/iter=%9.2f  tok/s=%9.1f\n",
                topk, active_experts, iters, us_per_iter, tokens_per_sec);

        dev->device_free(perm);   dev->device_free(gate_o);
        dev->device_free(up_o);   dev->device_free(gu);
        dev->device_free(swig);   dev->device_free(down_o);
        dev->device_free(moe_o);
    }
    if (us_topk1 > 0.0 && us_topk8 > 0.0) {
        const double marginal = (us_topk8 - us_topk1) / 7.0;
        const double fixed = us_topk1 - marginal;  // intercept at topk=0
        fprintf(stderr,
                "  marginal per-expert (us) ~= (topk8 - topk1)/7 = %.2f   "
                "fixed-overhead intercept (us) ~= topk1 - marginal = %.2f\n",
                marginal, fixed);
    }

    // ── Part B: per-stage breakdown at topk=1, T=1 ────────────────────────────
    fprintf(stderr, "  -- per-stage breakdown (topk=1, T=1) --\n");
    {
        const int topk = 1;
        const int P = T * topk;  // == 1
        std::vector<int32_t> topk_idx{0};
        std::vector<float> topk_w{1.0f};
        std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
        std::uniform_real_distribution<float> ad(-1.f, 1.f);
        for (auto& x : hidden) x = f32_to_bf16(ad(rng));

        auto* perm   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * H));
        auto* gate_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
        auto* up_o   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
        auto* gu     = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * 2 * I));
        auto* swig   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
        auto* down_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * H));
        auto* moe_o  = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * H));
        ASSERT_NE(perm, nullptr);
        ASSERT_NE(moe_o, nullptr);

        std::vector<int32_t> offsets(E + 1, 0), s2d(P, 0), pidx(P, 0);

        // Build the single-active-expert GEMM args once (expert 0).
        std::vector<int32_t> active_offsets{0, P};
        const void* gate_ptr = gate_w[0].data();
        const void* up_ptr   = up_w[0].data();
        const void* down_ptr = down_w[0].data();
        auto run_gemm = [&](void* D, const void* A, const void* bptr,
                            int N, int K) {
            Nvfp4GroupedGemmParams p{};
            p.num_experts = 1; p.N = N; p.K = K;
            p.A_base = A; p.D_base = D;
            p.expert_offsets = active_offsets.data();
            p.B_ptrs = &bptr;
            p.output_dtype = GemmOutputDtype::kBFloat16;
            dev->nvfp4_grouped_gemm(p, nullptr, 0, nullptr);
        };
        auto stage_permute = [&]() {
            dev->moe_permute(perm, offsets.data(), s2d.data(), pidx.data(),
                             hidden.data(), topk_idx.data(), T, topk, H, E, 2,
                             nullptr, nullptr);
        };
        auto stage_gate = [&]() { run_gemm(gate_o, perm, gate_ptr, I, H); };
        auto stage_up   = [&]() { run_gemm(up_o,   perm, up_ptr,   I, H); };
        auto stage_interleave = [&]() {
            for (int p = 0; p < P; ++p)
                for (int j = 0; j < I; ++j) {
                    gu[static_cast<size_t>(p) * 2 * I + j] =
                        gate_o[static_cast<size_t>(p) * I + j];
                    gu[static_cast<size_t>(p) * 2 * I + I + j] =
                        up_o[static_cast<size_t>(p) * I + j];
                }
        };
        auto stage_swiglu = [&]() {
            FusedSwigluParams sp{P, I};
            dev->fused_swiglu(swig, gu, sp, 2, nullptr);
        };
        auto stage_down = [&]() { run_gemm(down_o, swig, down_ptr, H, I); };
        auto stage_unpermute = [&]() {
            dev->moe_unpermute(moe_o, down_o, topk_w.data(), s2d.data(), T, topk,
                               H, 2, nullptr);
        };

        // Warmup: run the full chain once so buffers are touched and routing set.
        stage_permute(); stage_gate(); stage_up(); stage_interleave();
        stage_swiglu(); stage_down(); stage_unpermute();
        ASSERT_EQ(offsets[E], P);

        // Time each stage over enough iters to be stable. Stages are
        // independent given the warmed buffers; we re-run upstream-free so each
        // measures only its own kernel.
        const int kIters = 2000;
        auto time_stage = [&](auto&& fn) {
            // warmup
            for (int i = 0; i < 50; ++i) fn();
            auto a = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) fn();
            auto b = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::micro>(b - a).count() /
                   kIters;
        };

        const double t_permute    = time_stage(stage_permute);
        const double t_gate       = time_stage(stage_gate);
        const double t_up         = time_stage(stage_up);
        const double t_interleave = time_stage(stage_interleave);
        const double t_swiglu     = time_stage(stage_swiglu);
        const double t_down       = time_stage(stage_down);
        const double t_unpermute  = time_stage(stage_unpermute);

        const double total = t_permute + t_gate + t_up + t_interleave +
                             t_swiglu + t_down + t_unpermute;
        ASSERT_GT(total, 0.0);

        struct Row { const char* name; double us; };
        const Row rows[] = {
            {"moe_permute",          t_permute},
            {"gate nvfp4_gemm",      t_gate},
            {"up   nvfp4_gemm",      t_up},
            {"gate/up->gu copy",     t_interleave},
            {"fused_swiglu",         t_swiglu},
            {"down nvfp4_gemm",      t_down},
            {"moe_unpermute",        t_unpermute},
        };
        for (const auto& r : rows)
            fprintf(stderr, "  %-20s  us=%9.3f  %6.2f%%\n",
                    r.name, r.us, 100.0 * r.us / total);
        fprintf(stderr, "  %-20s  us=%9.3f  %6.2f%%\n", "TOTAL (sum)", total,
                100.0);
        const double gemm_us = t_gate + t_up + t_down;
        fprintf(stderr,
                "  gate+up+down GEMMs = %.3f us (%.2f%% of total); "
                "permute+unpermute = %.3f us (%.2f%%)\n",
                gemm_us, 100.0 * gemm_us / total,
                t_permute + t_unpermute,
                100.0 * (t_permute + t_unpermute) / total);

        dev->device_free(perm);   dev->device_free(gate_o);
        dev->device_free(up_o);   dev->device_free(gu);
        dev->device_free(swig);   dev->device_free(down_o);
        dev->device_free(moe_o);
    }
    fprintf(stderr, "==============================================\n");
}

// ── M=1 decode/gemv split + V0/V1/V2 attribution microbenchmarks ─────────────
//
// Single-threaded, ONE expert's gate projection (N=I=2048, K=H=7168 — the real
// engine dims). Answers the user's deliverable C:
//   C.1  decode-only vs gemv-only vs fused-total (per full projection, us).
//   C.2  V0 (scalar decode + materialize) -> V1 (vpermps decode + materialize)
//        -> V2 (vpermps decode + fused) attribution (us + x), isolating how much
//        of the historical M=1 speedup came from the vpermps decode vs the fused
//        GEMV.
// Also does a clean SINGLE-THREADED A/B of the ik-technique row-block GEMV vs the
// single-row fused GEMV over a whole projection (no thread/bandwidth confound).
//
// gtest-as-benchmark convention: prints to stderr, asserts only sanity.
TEST(NumaCpuExpert, M1DecodeGemvMicrobench) {
    const int N = 2048;  // == I (gate/up output rows)
    const int K = 7168;  // == H (gate/up input cols)

    std::mt19937 rng(0xA11CE);
    std::uniform_real_distribution<float> wd(-2.f, 2.f), ad(-1.f, 1.f);
    std::vector<float> W(static_cast<size_t>(N) * K);
    for (auto& w : W) w = wd(rng);
    const float ws2 = 0.8125f;
    auto buf = pack_proj(W, N, K, ws2);
    PackedProjection proj{buf.data(), static_cast<size_t>(proj_bytes(N, K)), N, K};
    CpuNvfp4ExpertWeights weights{&proj, 1};

    std::vector<uint16_t> a(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) a[k] = f32_to_bf16(ad(rng));

    // Decoded-weight scratch for the predecoded GEMV + materialize variants.
    std::vector<float> w_pre(static_cast<size_t>(N) * K);
    std::vector<float> row_scratch(static_cast<size_t>(K));
    std::vector<float> out(static_cast<size_t>(N));

#if defined(__AVX512F__)
    const bool avx512 = true;
#else
    const bool avx512 = false;
#endif
    fprintf(stderr,
            "\n==== NumaCpuExpert.M1DecodeGemvMicrobench ====\n"
            "  one gate projection: N=%d K=%d (single-threaded)  AVX-512=%s\n",
            N, K, avx512 ? "yes" : "no");

    auto time_us = [](int iters, auto&& fn) {
        for (int i = 0; i < 3; ++i) fn();  // warmup
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) fn();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    };

    // ── C.1: decode-only / gemv-only / fused-total ────────────────────────────
    volatile float sink = 0.0f;
    const int kIters = 200;

    // (i) decode-only: decode the full projection's N*K weights, NO dot/FMA.
    const double us_decode = time_us(kIters, [&] {
        bench_decode_full_expert(proj, w_pre.data());
        sink += w_pre[0];
    });
    // (ii) gemv-only: M=1 dot over PRE-DECODED FP32 weights (no decode).
    bench_decode_full_expert(proj, w_pre.data());  // ensure populated
    const double us_gemv = time_us(kIters, [&] {
        bench_gemv_predecoded(w_pre.data(), a.data(), N, K, out.data());
        sink += out[0];
    });
    // (iii) fused total: the production M=1 path (single-threaded, current = the
    // selected path; force row-block + single-row both below in the A/B).
    set_m1_gemv_path_for_testing(M1GemvPath::kFusedSingleRow);
    std::vector<int32_t> off{0, 1};
    std::vector<uint16_t> D(static_cast<size_t>(N), 0);
    const double us_fused_single = time_us(kIters, [&] {
        cpu_nvfp4_grouped_gemm(D.data(), a.data(), weights, off.data(), N, K, 2,
                               nullptr);
        sink += bf16_to_f32(D[0]);
    });
    set_m1_gemv_path_for_testing(M1GemvPath::kFusedRowBlock);
    const double us_fused_block = time_us(kIters, [&] {
        cpu_nvfp4_grouped_gemm(D.data(), a.data(), weights, off.data(), N, K, 2,
                               nullptr);
        sink += bf16_to_f32(D[0]);
    });

    fprintf(stderr,
            "  -- C.1 decode vs gemv split (per full N=%d projection) --\n"
            "  decode-only         us=%9.2f\n"
            "  gemv-only (predec)  us=%9.2f\n"
            "  decode+gemv (sum)   us=%9.2f\n"
            "  fused single-row    us=%9.2f\n"
            "  fused row-block(ik) us=%9.2f\n",
            N, us_decode, us_gemv, us_decode + us_gemv, us_fused_single,
            us_fused_block);

    // ── A/B decision input: single-threaded fused single-row vs ik row-block ──
    fprintf(stderr,
            "  -- A/B (single-threaded, whole projection) --\n"
            "  single-row=%9.2f us   row-block(ik)=%9.2f us   speedup=%.3fx\n",
            us_fused_single, us_fused_block, us_fused_single / us_fused_block);

    // ── C.2: V0 / V1 / V2 attribution (per full projection, all N rows) ───────
    const double us_v0 = time_us(kIters, [&] {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n)
            acc += bench_m1_v0_scalar_materialize(proj, n, ws2, a.data(),
                                                  row_scratch.data());
        sink += acc;
    });
    const double us_v1 = time_us(kIters, [&] {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n)
            acc += bench_m1_v1_vec_materialize(proj, n, ws2, a.data(),
                                               row_scratch.data());
        sink += acc;
    });
    const double us_v2 = time_us(kIters, [&] {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n)
            acc += bench_m1_v2_fused(proj, n, ws2, a.data());
        sink += acc;
    });

    fprintf(stderr,
            "  -- C.2 V0/V1/V2 attribution (per full N=%d projection) --\n"
            "  V0 scalar-decode + materialize  us=%9.2f  (baseline)\n"
            "  V1 vpermps-decode + materialize us=%9.2f  V0->V1 %.3fx (-%.1f us)\n"
            "  V2 vpermps-decode + fused GEMV  us=%9.2f  V1->V2 %.3fx (-%.1f us)\n"
            "  total V0->V2 %.3fx (-%.1f us)\n",
            N, us_v0,
            us_v1, us_v0 / us_v1, us_v0 - us_v1,
            us_v2, us_v1 / us_v2, us_v1 - us_v2,
            us_v0 / us_v2, us_v0 - us_v2);
    fprintf(stderr, "  (sink=%g)\n", static_cast<float>(sink));
    fprintf(stderr, "==============================================\n");

    // Restore the process default (row-block) and assert basic sanity.
    set_m1_gemv_path_for_testing(M1GemvPath::kFusedRowBlock);
    ASSERT_GT(us_decode, 0.0);
    ASSERT_GT(us_gemv, 0.0);
    ASSERT_GT(us_v0, 0.0);
    ASSERT_GT(us_v2, 0.0);
}

// ── MultiNuma helpers: drive a full FFN chain through any ExpertDevice ───────
namespace {

// Packed expert weight set for one expert (gate [I,H], up [I,H], down [H,I]).
struct ExpertWeights {
    std::vector<uint8_t> gate, up, down;
};

// Build E experts' packed projections with non-trivial per-expert ws2.
std::vector<ExpertWeights> build_experts(int E, int H, int I, std::mt19937& rng) {
    std::uniform_real_distribution<float> wd(-2.f, 2.f);
    auto rand_w = [&](int N, int K) {
        std::vector<float> w(static_cast<size_t>(N) * K);
        for (auto& x : w) x = wd(rng);
        return w;
    };
    std::vector<ExpertWeights> out(E);
    for (int e = 0; e < E; ++e) {
        const float ws2_g = 0.75f + 0.05f * static_cast<float>(e % 5);
        const float ws2_u = 0.80f + 0.03f * static_cast<float>(e % 7);
        const float ws2_d = 0.90f + 0.02f * static_cast<float>(e % 3);
        out[e].gate = pack_proj(rand_w(I, H), I, H, ws2_g);
        out[e].up   = pack_proj(rand_w(I, H), I, H, ws2_u);
        out[e].down = pack_proj(rand_w(H, I), H, I, ws2_d);
    }
    return out;
}

// Run permute -> gate -> up -> swiglu -> down -> unpermute through `dev`, writing
// moe_output (BF16 [T,H]) into `moe_out`. T tokens, topk experts each, routing in
// topk_idx / topk_w. Scratch is device_alloc'd and freed. Returns nothing.
void run_ffn_chain(ExpertDevice* dev, int H, int I, int T, int topk, int E,
                   const std::vector<ExpertWeights>& experts,
                   const std::vector<uint16_t>& hidden,
                   const std::vector<int32_t>& topk_idx,
                   const std::vector<float>& topk_w,
                   std::vector<uint16_t>& moe_out) {
    const int P = T * topk;
    auto* perm   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * H));
    auto* gate_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
    auto* up_o   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
    auto* gu     = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * 2 * I));
    auto* swig   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
    auto* down_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * H));
    auto* moe_o  = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * H));

    std::vector<int32_t> offsets(E + 1, 0), s2d(P, 0), pidx(P, 0);
    dev->moe_permute(perm, offsets.data(), s2d.data(), pidx.data(), hidden.data(),
                     topk_idx.data(), T, topk, H, E, 2, nullptr, nullptr);

    // Compact active experts (those with >=1 token) into B_ptrs/offsets.
    std::vector<const void*> gate_ptrs, up_ptrs, down_ptrs;
    std::vector<int32_t> active_offsets{0};
    for (int e = 0; e < E; ++e) {
        if (offsets[e + 1] <= offsets[e]) continue;
        gate_ptrs.push_back(experts[e].gate.data());
        up_ptrs.push_back(experts[e].up.data());
        down_ptrs.push_back(experts[e].down.data());
        active_offsets.push_back(offsets[e + 1]);
    }
    const int active = static_cast<int>(gate_ptrs.size());
    auto run_gemm = [&](void* D, const void* A, std::vector<const void*>& bp,
                        int N, int K) {
        Nvfp4GroupedGemmParams p{};
        p.num_experts = active; p.N = N; p.K = K;
        p.A_base = A; p.D_base = D;
        p.expert_offsets = active_offsets.data();
        p.B_ptrs = bp.data();
        p.output_dtype = GemmOutputDtype::kBFloat16;
        dev->nvfp4_grouped_gemm(p, nullptr, 0, nullptr);
    };
    run_gemm(gate_o, perm, gate_ptrs, I, H);
    run_gemm(up_o,   perm, up_ptrs,   I, H);
    for (int p = 0; p < P; ++p)
        for (int j = 0; j < I; ++j) {
            gu[static_cast<size_t>(p) * 2 * I + j]     = gate_o[static_cast<size_t>(p) * I + j];
            gu[static_cast<size_t>(p) * 2 * I + I + j] = up_o[static_cast<size_t>(p) * I + j];
        }
    FusedSwigluParams sp{P, I};
    dev->fused_swiglu(swig, gu, sp, 2, nullptr);
    run_gemm(down_o, swig, down_ptrs, H, I);
    dev->moe_unpermute(moe_o, down_o, topk_w.data(), s2d.data(), T, topk, H, 2,
                       nullptr);

    moe_out.assign(moe_o, moe_o + static_cast<size_t>(T) * H);
    dev->device_free(perm);   dev->device_free(gate_o); dev->device_free(up_o);
    dev->device_free(gu);     dev->device_free(swig);   dev->device_free(down_o);
    dev->device_free(moe_o);
}

// Usable NUMA nodes the host actually exposes (those with enumerable physical
// cores), capped at `cap`. Falls back to {0} if sysfs yields nothing.
std::vector<int> usable_nodes(int cap) {
    std::vector<int> nodes;
    for (int n = 0; n < 16 && static_cast<int>(nodes.size()) < cap; ++n) {
        if (!cpu::node_physical_cpus(n).empty()) nodes.push_back(n);
    }
    if (nodes.empty()) nodes.push_back(0);
    return nodes;
}

}  // namespace

// ── Correctness gate: distributed result == single-node result ───────────────
//
// The MultiNumaCpuExpertDevice (M=2 and M=4, or however many nodes the host has)
// must produce the SAME moe_output as the single-node NumaCpuExpertDevice for the
// same expert(s)+input, within BF16 tolerance (the output-N partition only changes
// summation ORDER within a projection negligibly — the down-proj rows are disjoint
// per node so there is no cross-node accumulation at all). Gates timing.
TEST(MultiNumaCpuExpert, MatchesSingleNode) {
    const int H = 256, I = 256;  // 128-aligned (N%128==0, K%64==0) per INV-CPU-EXP-3
    const int E = 6, topk = 3;
    const int T_sweep[] = {1, 8};

    std::mt19937 rng(0x31A11);
    auto experts = build_experts(E, H, I, rng);

    // Single-node reference device on node 0.
    config::GpuRef ref0{}; ref0.position = 0; ref0.id = 0; ref0.type = config::GpuType::cpu;
    NumaCpuExpertDeps sdeps{}; sdeps.dims = CpuExpertModelDims{H, I, topk, E};
    sdeps.max_threads = 0;
    auto single = make_numa_cpu_expert_device(ref0, sdeps);
    single->set_device();

    for (int M : {2, 4}) {
        auto nodes = usable_nodes(M);
        const int m_actual = static_cast<int>(nodes.size());

        config::GpuRef refm{}; refm.position = 0; refm.id = nodes[0];
        refm.type = config::GpuType::cpu;
        MultiNumaCpuExpertDeps mdeps{};
        mdeps.nodes = nodes;
        mdeps.dims = CpuExpertModelDims{H, I, topk, E};
        mdeps.max_threads_per_node = 0;
        auto multi = make_multi_numa_cpu_expert_device(refm, mdeps);
        multi->set_device();

        for (int T : T_sweep) {
            // Random routing: T tokens, topk distinct experts each.
            std::vector<int32_t> topk_idx(static_cast<size_t>(T) * topk);
            std::vector<float> topk_w(static_cast<size_t>(T) * topk);
            std::mt19937 rr(0xBEE + static_cast<unsigned>(T * 31 + M));
            for (int t = 0; t < T; ++t) {
                std::vector<int> pool(E);
                for (int e = 0; e < E; ++e) pool[e] = e;
                float wsum = 0.f;
                for (int k = 0; k < topk; ++k) {
                    std::uniform_int_distribution<int> pick(k, E - 1);
                    std::swap(pool[k], pool[pick(rr)]);
                    topk_idx[t * topk + k] = pool[k];
                    float v = std::fabs(std::uniform_real_distribution<float>(-2, 2)(rr)) + 0.05f;
                    topk_w[t * topk + k] = v; wsum += v;
                }
                for (int k = 0; k < topk; ++k) topk_w[t * topk + k] /= wsum;
            }
            std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
            std::uniform_real_distribution<float> ad(-1.f, 1.f);
            for (auto& x : hidden) x = f32_to_bf16(ad(rng));

            std::vector<uint16_t> out_single, out_multi;
            run_ffn_chain(single.get(), H, I, T, topk, E, experts, hidden,
                          topk_idx, topk_w, out_single);
            run_ffn_chain(multi.get(), H, I, T, topk, E, experts, hidden,
                          topk_idx, topk_w, out_multi);

            ASSERT_EQ(out_single.size(), out_multi.size());
            for (size_t i = 0; i < out_single.size(); ++i) {
                float a = bf16_to_f32(out_single[i]);
                float b = bf16_to_f32(out_multi[i]);
                EXPECT_NEAR(b, a, std::fabs(a) * 0.02f + 0.05f)
                    << "M=" << M << "(actual " << m_actual << ") T=" << T
                    << " i=" << i;
            }
        }
        fprintf(stderr,
                "MultiNumaCpuExpert.MatchesSingleNode: M=%d requested, %d node(s) "
                "used (%s) — match within BF16 tol\n",
                M, m_actual,
                [&] { std::string s; for (int n : nodes) s += std::to_string(n) + " "; return s; }().c_str());
    }
}

// ── Scaling benchmark: 1-token-1-expert hot unit at M=1/2/4 nodes ────────────
//
// Times the realistic routed-expert FFN (H=7168, I=2048) for the 1×1×1 hot unit
// at M=1, M=2, M=4 nodes (or however many the host exposes). Reports per-config
// compute us, the cross-node weight-staging us (separately — staging is the
// cross-node cost the speedup must overcome), total us, speedup vs M=1, and the
// achieved fraction of the ideal 1/M. Also a T=8 grouped sanity row. Prints host
// context. gtest-as-benchmark: asserts only sanity.
TEST(MultiNumaCpuExpert, ScalingBenchmark) {
    const int H = 7168, I = 2048, E = 8;

    std::mt19937 rng(0x5CA1E);
    auto experts = build_experts(E, H, I, rng);

    const int64_t gate_pb = proj_bytes(I, H);
    const int64_t down_pb = proj_bytes(H, I);
    const int64_t per_expert_bytes = 2 * gate_pb + down_pb;

    const auto all_nodes = usable_nodes(4);
    const int max_M = static_cast<int>(all_nodes.size());
    const auto n0_cores = cpu::node_physical_cpus(all_nodes.empty() ? 0 : all_nodes[0]);
#if defined(__AVX512F__)
    const bool avx512 = true;
#else
    const bool avx512 = false;
#endif
    fprintf(stderr,
            "\n==== MultiNumaCpuExpert.ScalingBenchmark ====\n"
            "  dims: H=%d I=%d  per-expert packed bytes=%lld (gate=%lld down=%lld)\n"
            "  host: usable_nodes=%d  cores/node(node0)=%zu  AVX-512=%s\n"
            "  ideal: latency -> 1/M as N-rows split across M nodes\n",
            H, I, static_cast<long long>(per_expert_bytes),
            static_cast<long long>(gate_pb), static_cast<long long>(down_pb),
            max_M, n0_cores.size(), avx512 ? "yes" : "no");

    setenv("LS_MULTINUMA_PROFILE", "1", 1);  // enable stage/compute attribution
    auto time_chain = [&](ExpertDevice* dev, int T, int topk, int Eactive,
                          double& staged_mb, double& stage_us, double& compute_us) {
        // Routing.
        std::vector<int32_t> topk_idx(static_cast<size_t>(T) * topk);
        std::vector<float> topk_w(static_cast<size_t>(T) * topk);
        std::mt19937 rr(0xC0FFE + static_cast<unsigned>(T * 7 + topk));
        for (int t = 0; t < T; ++t) {
            std::vector<int> pool(Eactive);
            for (int e = 0; e < Eactive; ++e) pool[e] = e;
            float wsum = 0.f;
            for (int k = 0; k < topk; ++k) {
                std::uniform_int_distribution<int> pick(k, Eactive - 1);
                std::swap(pool[k], pool[pick(rr)]);
                topk_idx[t * topk + k] = pool[k];
                float v = std::fabs(std::uniform_real_distribution<float>(-2, 2)(rr)) + 0.05f;
                topk_w[t * topk + k] = v; wsum += v;
            }
            for (int k = 0; k < topk; ++k) topk_w[t * topk + k] /= wsum;
        }
        std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
        std::uniform_real_distribution<float> ad(-1.f, 1.f);
        for (auto& x : hidden) x = f32_to_bf16(ad(rng));
        std::vector<uint16_t> out;

        // Staged bytes/call = topk experts * (gate+up+down) packed bytes (every
        // node stages its N-slice of each active expert; summed over nodes == one
        // full per-expert set since the slices partition N).
        staged_mb = static_cast<double>(topk) *
                    static_cast<double>(per_expert_bytes) / 1048576.0;

        // Warmup (also populates the stage residency so steady-state skips re-copy).
        run_ffn_chain(dev, H, I, T, topk, Eactive, experts, hidden, topk_idx,
                      topk_w, out);
        bool finite = true;
        for (float v : {bf16_to_f32(out[0]), bf16_to_f32(out.back())})
            if (!std::isfinite(v)) finite = false;
        EXPECT_TRUE(finite);

        auto t_probe0 = std::chrono::steady_clock::now();
        run_ffn_chain(dev, H, I, T, topk, Eactive, experts, hidden, topk_idx,
                      topk_w, out);
        auto t_probe1 = std::chrono::steady_clock::now();
        const double one_us =
            std::chrono::duration<double, std::micro>(t_probe1 - t_probe0).count();
        int iters = static_cast<int>(150000.0 / std::max(one_us, 1.0));
        iters = std::max(3, std::min(iters, static_cast<int>(20e6 / std::max(one_us, 1.0))));

        // Steady-state timing with stage/compute attribution. Profile counters
        // accumulate across ALL node-threads, so per-iter stage/compute us =
        // (total ns / iters) / 1000; this is summed node-time (parallel), reported
        // as the cross-node staging cost vs the per-node compute cost.
        multinuma_profile_reset();
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it)
            run_ffn_chain(dev, H, I, T, topk, Eactive, experts, hidden, topk_idx,
                          topk_w, out);
        auto t1 = std::chrono::steady_clock::now();
        stage_us = (multinuma_profile_stage_ns() / 1e3) / iters;
        compute_us = (multinuma_profile_compute_ns() / 1e3) / iters;
        // staged_mb reported is the ACTUAL re-staged bytes/iter (0 at steady state
        // when residency holds; nonzero only when slices change between calls).
        staged_mb = (static_cast<double>(multinuma_profile_staged_bytes()) / iters)
                    / 1048576.0;
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    };

    // ── Apples-to-apples baseline: the DEDICATED single-node device on node 0 ─
    // This is the real reference the per-node MultiNuma compute must match. We
    // time the SAME 1×1×1 hot unit through NumaCpuExpertDevice (full node-0
    // pool) so the speedup column below is measured against the best single-node
    // path, not against MultiNuma's own M=1 (which carries the cross-node
    // scaffold). Reported as the "M=0" / "single-node" row.
    double single_us = 0.0;
    {
        config::GpuRef ref0{}; ref0.position = 0; ref0.id = all_nodes.empty() ? 0 : all_nodes[0];
        ref0.type = config::GpuType::cpu;
        NumaCpuExpertDeps sdeps{}; sdeps.dims = CpuExpertModelDims{H, I, 1, E};
        sdeps.max_threads = 0;
        auto sdev = make_numa_cpu_expert_device(ref0, sdeps);
        sdev->set_device();
        double smb = 0.0, sst = 0.0, sc = 0.0;
        // Best-of-3 (min) to reject scheduler noise on the shared box.
        single_us = 1e18;
        for (int r = 0; r < 3; ++r)
            single_us = std::min(single_us,
                                 time_chain(sdev.get(), 1, 1, E, smb, sst, sc));
    }

    // ── 1×1×1 hot unit: M = 1, 2, 4 (capped at host node count) ──────────────
    fprintf(stderr, "  -- 1-token-1-expert hot unit (T=1, topk=1) --\n");
    fprintf(stderr, "  single-node NumaCpuExpert (node0, all cores): total_us=%.1f"
                    "  <- apples-to-apples baseline\n", single_us);
    fprintf(stderr, "  %-4s %-7s %-10s %-10s %-11s %-10s %-9s %-8s\n",
            "M", "nodes", "compute_us", "stage_us", "total_us", "restage_MB",
            "speedup", "%ideal");
    double base_us = 0.0;
    for (int M = 1; M <= max_M; M *= 2) {
        auto nodes = usable_nodes(M);
        const int m_actual = static_cast<int>(nodes.size());
        config::GpuRef refm{}; refm.position = 0; refm.id = nodes[0];
        refm.type = config::GpuType::cpu;
        MultiNumaCpuExpertDeps mdeps{};
        mdeps.nodes = nodes;
        mdeps.dims = CpuExpertModelDims{H, I, 1, E};
        mdeps.max_threads_per_node = 0;
        auto dev = make_multi_numa_cpu_expert_device(refm, mdeps);
        dev->set_device();

        double staged_mb = 0.0, stage_us = 0.0, compute_us = 0.0;
        // Best-of-3 (min total) to reject scheduler noise; keep the stage/compute
        // attribution from the fastest run.
        double total_us = 1e18;
        for (int r = 0; r < 3; ++r) {
            double smb = 0.0, sst = 0.0, sc = 0.0;
            const double t = time_chain(dev.get(), /*T=*/1, /*topk=*/1, E,
                                        smb, sst, sc);
            if (t < total_us) {
                total_us = t; staged_mb = smb; stage_us = sst; compute_us = sc;
            }
        }
        if (M == 1) base_us = single_us > 0.0 ? single_us : total_us;
        // speedup is vs the DEDICATED single-node device (apples-to-apples), so
        // %ideal reflects the real latency win, not MultiNuma-vs-its-own-M=1.
        const double speedup = base_us / total_us;
        const double pct_ideal = 100.0 * speedup / static_cast<double>(m_actual);
        fprintf(stderr, "  %-4d %-7d %-10.1f %-10.1f %-11.1f %-10.2f %-9.3f %-8.1f\n",
                M, m_actual, compute_us, stage_us, total_us, staged_mb, speedup,
                pct_ideal);
        ASSERT_GT(total_us, 0.0);
    }

    // ── Multibatch grouped sanity (T=8, topk=2) at full M ────────────────────
    fprintf(stderr, "  -- multibatch grouped sanity (T=8, topk=2, full M) --\n");
    {
        auto nodes = usable_nodes(max_M);
        config::GpuRef refm{}; refm.position = 0; refm.id = nodes[0];
        refm.type = config::GpuType::cpu;
        MultiNumaCpuExpertDeps mdeps{};
        mdeps.nodes = nodes;
        mdeps.dims = CpuExpertModelDims{H, I, 2, E};
        mdeps.max_threads_per_node = 0;
        auto dev = make_multi_numa_cpu_expert_device(refm, mdeps);
        dev->set_device();
        double staged_mb = 0.0, stage_us = 0.0, compute_us = 0.0;
        const double total_us = time_chain(dev.get(), /*T=*/8, /*topk=*/2, E,
                                           staged_mb, stage_us, compute_us);
        fprintf(stderr, "  M=%d  T=8 topk=2  total_us=%.1f  us/token=%.1f  "
                "compute_us=%.1f  stage_us=%.1f  restage_MB=%.2f\n",
                static_cast<int>(nodes.size()), total_us, total_us / 8.0,
                compute_us, stage_us, staged_mb);
        ASSERT_GT(total_us, 0.0);
    }
    fprintf(stderr, "=============================================\n");
}

// ── Multibatch throughput benchmark: cross product of T and M nodes ──────────
//
// Mirrors NumaCpuExpert.FfnThroughputBenchmark (single-node) but for the
// MultiNumaCpuExpertDevice across node counts. Drives the FULL routed-expert FFN
// chain (permute -> gate -> up -> swiglu -> down -> unpermute) on realistic dims
// (DeepSeek V3.2: hidden=7168, moe_intermediate=2048, topk=8, ~16 experts) and
// reports a throughput table over the cross product of T={1,8,32} and M∈{1,2,4}
// nodes (M clamped to the host's usable NUMA node count). For EACH (T,M): warm
// up so the per-node stage residency is hot (restage ~0 at steady state), then
// chrono-time the chain to ~200ms and report us/iter, us/token, tok/s, plus the
// staged (full per-iter weight set) and restage (actually re-copied at steady
// state) MB. The DEDICATED single-node NumaCpuExpertDevice is reported as the
// apples-to-apples baseline row at each T (the same comparison style as the
// single-node bench). gtest-as-benchmark: asserts only sanity (finite output,
// tok/s>0). Set LS_MULTINUMA_PROFILE=1 to attribute restage bytes.
//
// Reuses the file's pack_proj / build_experts / run_ffn_chain / usable_nodes
// helpers and the multinuma_profile_* counters — does NOT rewrite packing or the
// chain driver.
TEST(MultiNumaCpuExpert, MultibatchThroughputBenchmark) {
    // Realistic routed-expert dims (128-aligned: H%128==0, I%128==0).
    const int H = 7168, I = 2048, topk = 8, E = 16;

    std::mt19937 rng(0xBA7C4u);  // deterministic seed
    auto experts = build_experts(E, H, I, rng);

    const int64_t gate_pb = proj_bytes(I, H);  // == up_pb
    const int64_t down_pb = proj_bytes(H, I);
    const int64_t per_expert_bytes = 2 * gate_pb + down_pb;  // gate+up+down

    const auto all_nodes = usable_nodes(4);
    const int max_M = static_cast<int>(all_nodes.size());
    const auto n0_cores =
        cpu::node_physical_cpus(all_nodes.empty() ? 0 : all_nodes[0]);
#if defined(__AVX512F__)
    const bool avx512 = true;
#else
    const bool avx512 = false;
#endif

    fprintf(stderr,
            "\n==== MultiNumaCpuExpert.MultibatchThroughputBenchmark ====\n"
            "  dims: H=%d I=%d topk=%d  experts_available=%d\n"
            "  per-expert packed bytes (gate+up+down) = %lld  "
            "(gate=%lld up=%lld down=%lld)\n"
            "  host: usable_nodes=%d  cores/node(node0)=%zu  AVX-512_compiled=%s\n"
            "  M nodes split each expert's N-rows across M NUMA nodes; T tokens "
            "fan into topk experts.\n",
            H, I, topk, E,
            static_cast<long long>(per_expert_bytes),
            static_cast<long long>(gate_pb), static_cast<long long>(gate_pb),
            static_cast<long long>(down_pb),
            max_M, n0_cores.size(), avx512 ? "yes" : "no");
    if (max_M < 4)
        fprintf(stderr,
                "  NOTE: host exposes only %d usable NUMA node(s); M sweep "
                "clamped to {%s}.\n",
                max_M, [&] {
                    std::string s;
                    for (int m = 1; m <= max_M; m *= 2)
                        s += std::to_string(m) + (m * 2 <= max_M ? "," : "");
                    return s;
                }().c_str());

    setenv("LS_MULTINUMA_PROFILE", "1", 1);  // enable restage-bytes attribution

    // Times the full chain through `dev` at (T, topk, Eactive); returns us/iter
    // and (via out-params) the steady-state restaged MB/iter. Warms up first so
    // residency is hot. Best-of-3 (min) to reject scheduler noise on a shared box.
    auto time_chain = [&](ExpertDevice* dev, int T, double& restage_mb_out) {
        // Routing: T tokens, topk distinct experts each.
        std::vector<int32_t> topk_idx(static_cast<size_t>(T) * topk);
        std::vector<float> topk_w(static_cast<size_t>(T) * topk);
        std::mt19937 rr(0xC0FFE + static_cast<unsigned>(T * 7 + topk));
        for (int t = 0; t < T; ++t) {
            std::vector<int> pool(E);
            for (int e = 0; e < E; ++e) pool[e] = e;
            float wsum = 0.f;
            for (int k = 0; k < topk; ++k) {
                std::uniform_int_distribution<int> pick(k, E - 1);
                std::swap(pool[k], pool[pick(rr)]);
                topk_idx[t * topk + k] = pool[k];
                float v =
                    std::fabs(std::uniform_real_distribution<float>(-2, 2)(rr)) +
                    0.05f;
                topk_w[t * topk + k] = v; wsum += v;
            }
            for (int k = 0; k < topk; ++k) topk_w[t * topk + k] /= wsum;
        }
        std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
        std::uniform_real_distribution<float> ad(-1.f, 1.f);
        for (auto& x : hidden) x = f32_to_bf16(ad(rng));
        std::vector<uint16_t> out;

        // Warmup (populates stage residency; sanity-checks finiteness).
        run_ffn_chain(dev, H, I, T, topk, E, experts, hidden, topk_idx, topk_w,
                      out);
        bool finite = true;
        for (float v : {bf16_to_f32(out.front()), bf16_to_f32(out.back())})
            if (!std::isfinite(v)) finite = false;
        EXPECT_TRUE(finite) << "moe_output not finite (T=" << T << ")";

        // Size the loop to ~200ms; cap so worst case stays < ~30s.
        auto p0 = std::chrono::steady_clock::now();
        run_ffn_chain(dev, H, I, T, topk, E, experts, hidden, topk_idx, topk_w,
                      out);
        auto p1 = std::chrono::steady_clock::now();
        const double one_us =
            std::chrono::duration<double, std::micro>(p1 - p0).count();
        int iters = static_cast<int>(200000.0 / std::max(one_us, 1.0));
        iters = std::max(1, iters);
        const int max_iters = static_cast<int>(30e6 / std::max(one_us, 1.0));
        if (iters > max_iters) iters = std::max(1, max_iters);

        double best_us = 1e18, best_restage_mb = 0.0;
        for (int r = 0; r < 3; ++r) {
            multinuma_profile_reset();
            auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; ++it)
                run_ffn_chain(dev, H, I, T, topk, E, experts, hidden, topk_idx,
                              topk_w, out);
            auto t1 = std::chrono::steady_clock::now();
            const double us =
                std::chrono::duration<double, std::micro>(t1 - t0).count() /
                iters;
            if (us < best_us) {
                best_us = us;
                best_restage_mb =
                    (static_cast<double>(multinuma_profile_staged_bytes()) /
                     iters) / 1048576.0;
            }
        }
        restage_mb_out = best_restage_mb;
        return best_us;
    };

    // Full per-iter weight set staged = topk experts * (gate+up+down) bytes.
    const double staged_mb_per_iter =
        static_cast<double>(topk) * static_cast<double>(per_expert_bytes) /
        1048576.0;

    const int T_sweep[] = {1, 8, 32};
    fprintf(stderr,
            "  staged/iter (full weight set read) = %.2f MB  (topk*%lld)\n",
            staged_mb_per_iter, static_cast<long long>(per_expert_bytes));
    fprintf(stderr,
            "  %-9s %-4s %-7s %-9s %-10s %-11s %-9s %-9s %-9s\n",
            "device", "M", "nodes", "us/iter", "us/token", "tok/s", "staged_MB",
            "restage_MB", "speedup");

    // ── Per-T: single-node baseline + the M sweep ─────────────────────────────
    for (int T : T_sweep) {
        // Apples-to-apples baseline: the dedicated single-node NumaCpuExpert.
        double base_us = 0.0;
        {
            config::GpuRef ref0{};
            ref0.position = 0; ref0.id = all_nodes.empty() ? 0 : all_nodes[0];
            ref0.type = config::GpuType::cpu;
            NumaCpuExpertDeps sdeps{};
            sdeps.dims = CpuExpertModelDims{H, I, topk, E};
            sdeps.max_threads = 0;
            auto sdev = make_numa_cpu_expert_device(ref0, sdeps);
            sdev->set_device();
            double restage_mb = 0.0;
            base_us = time_chain(sdev.get(), T, restage_mb);
            const double us_tok = base_us / T;
            ASSERT_GT(1e6 / us_tok, 0.0);
            fprintf(stderr,
                    "  %-9s %-4s %-7d %-9.1f %-10.2f %-11.1f %-9.2f %-9.2f %-9s\n",
                    "single", "-", 1, base_us, us_tok, 1e6 / us_tok,
                    staged_mb_per_iter, restage_mb, "1.000(ref)");
        }

        // MultiNuma at M = 1, 2, 4 (capped at host node count). M is fixed at
        // construction, so build a fresh device per node count.
        for (int M = 1; M <= max_M; M *= 2) {
            auto nodes = usable_nodes(M);
            const int m_actual = static_cast<int>(nodes.size());
            config::GpuRef refm{};
            refm.position = 0; refm.id = nodes[0]; refm.type = config::GpuType::cpu;
            MultiNumaCpuExpertDeps mdeps{};
            mdeps.nodes = nodes;
            mdeps.dims = CpuExpertModelDims{H, I, topk, E};
            mdeps.max_threads_per_node = 0;
            auto dev = make_multi_numa_cpu_expert_device(refm, mdeps);
            dev->set_device();

            double restage_mb = 0.0;
            const double us = time_chain(dev.get(), T, restage_mb);
            const double us_tok = us / T;
            const double tok_s = 1e6 / us_tok;
            const double speedup = base_us / us;
            ASSERT_GT(tok_s, 0.0);
            fprintf(stderr,
                    "  %-9s %-4d %-7d %-9.1f %-10.2f %-11.1f %-9.2f %-9.2f %-9.3f\n",
                    "multi", M, m_actual, us, us_tok, tok_s, staged_mb_per_iter,
                    restage_mb, speedup);
        }
    }
    fprintf(stderr, "==========================================================\n");
}

// ── GGUF multi-node helpers (TD-IK-MULTINODE) ────────────────────────────────
namespace {

// Packed GGUF expert weight set for one expert (gate [I,H], up [I,H], down [H,I])
// at a given GgufType. Rows are self-contained packed weight rows.
struct GgufExpertWeights {
    std::vector<uint8_t> gate, up, down;
};

// Pack a dense FP32 [N,K] weight as N packed GGUF rows at `t`.
std::vector<uint8_t> pack_gguf(ik::GgufType t, const std::vector<float>& W,
                               int N, int K) {
    const size_t rb = ik::weight_row_bytes(t, K);
    std::vector<uint8_t> packed(static_cast<size_t>(N) * rb);
    for (int n = 0; n < N; ++n)
        ik::quantize_weight(t, W.data() + static_cast<size_t>(n) * K,
                            packed.data() + static_cast<size_t>(n) * rb, K);
    return packed;
}

std::vector<GgufExpertWeights> build_gguf_experts(ik::GgufType t, int E, int H,
                                                  int I, std::mt19937& rng) {
    std::uniform_real_distribution<float> wd(-0.6f, 0.6f);
    auto rand_w = [&](int N, int K) {
        std::vector<float> w(static_cast<size_t>(N) * K);
        for (auto& x : w) x = wd(rng);
        return w;
    };
    std::vector<GgufExpertWeights> out(E);
    for (int e = 0; e < E; ++e) {
        out[e].gate = pack_gguf(t, rand_w(I, H), I, H);
        out[e].up   = pack_gguf(t, rand_w(I, H), I, H);
        out[e].down = pack_gguf(t, rand_w(H, I), H, I);
    }
    return out;
}

// GGUF version of run_ffn_chain: permute -> gate -> up -> swiglu -> down ->
// unpermute through `dev` using gguf_grouped_gemm.
void run_gguf_ffn_chain(ExpertDevice* dev, GgufQuantType qt, int H, int I, int T,
                        int topk, int E,
                        const std::vector<GgufExpertWeights>& experts,
                        const std::vector<uint16_t>& hidden,
                        const std::vector<int32_t>& topk_idx,
                        const std::vector<float>& topk_w,
                        std::vector<uint16_t>& moe_out) {
    const int P = T * topk;
    auto* perm   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * H));
    auto* gate_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
    auto* up_o   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
    auto* gu     = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * 2 * I));
    auto* swig   = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * I));
    auto* down_o = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * P * H));
    auto* moe_o  = static_cast<uint16_t*>(dev->device_alloc(sizeof(uint16_t) * T * H));

    std::vector<int32_t> offsets(E + 1, 0), s2d(P, 0), pidx(P, 0);
    dev->moe_permute(perm, offsets.data(), s2d.data(), pidx.data(), hidden.data(),
                     topk_idx.data(), T, topk, H, E, 2, nullptr, nullptr);

    std::vector<const void*> gate_ptrs, up_ptrs, down_ptrs;
    std::vector<int32_t> active_offsets{0};
    for (int e = 0; e < E; ++e) {
        if (offsets[e + 1] <= offsets[e]) continue;
        gate_ptrs.push_back(experts[e].gate.data());
        up_ptrs.push_back(experts[e].up.data());
        down_ptrs.push_back(experts[e].down.data());
        active_offsets.push_back(offsets[e + 1]);
    }
    const int active = static_cast<int>(gate_ptrs.size());
    auto run_gemm = [&](void* D, const void* A, std::vector<const void*>& bp,
                        int N, int K) {
        GgufGroupedGemmParams p{};
        p.type = qt; p.num_experts = active; p.N = N; p.K = K;
        p.A_base = A; p.D_base = D;
        p.expert_offsets = active_offsets.data();
        p.B_ptrs = bp.data();
        dev->gguf_grouped_gemm(p, nullptr, 0, nullptr);
    };
    run_gemm(gate_o, perm, gate_ptrs, I, H);
    run_gemm(up_o,   perm, up_ptrs,   I, H);
    for (int p = 0; p < P; ++p)
        for (int j = 0; j < I; ++j) {
            gu[static_cast<size_t>(p) * 2 * I + j]     = gate_o[static_cast<size_t>(p) * I + j];
            gu[static_cast<size_t>(p) * 2 * I + I + j] = up_o[static_cast<size_t>(p) * I + j];
        }
    FusedSwigluParams sp{P, I};
    dev->fused_swiglu(swig, gu, sp, 2, nullptr);
    run_gemm(down_o, swig, down_ptrs, H, I);
    dev->moe_unpermute(moe_o, down_o, topk_w.data(), s2d.data(), T, topk, H, 2,
                       nullptr);

    moe_out.assign(moe_o, moe_o + static_cast<size_t>(T) * H);
    dev->device_free(perm);   dev->device_free(gate_o); dev->device_free(up_o);
    dev->device_free(gu);     dev->device_free(swig);   dev->device_free(down_o);
    dev->device_free(moe_o);
}

}  // namespace

// ── Correctness gate: distributed GGUF == single-node GGUF, BIT-EXACT ─────────
//
// The GGUF multi-node path partitions the N OUTPUT ROWS across M nodes; each
// output element is a complete dot computed entirely on ONE node (no cross-node
// accumulation, no summation reorder), so the distributed result must be
// BIT-EXACT to the single-node NumaCpuExpertDevice GGUF output (INV-CPU-EXP-GGUF-MN).
// We assert exact uint16_t (BF16-bitpattern) equality of the full FFN moe_output.
TEST(MultiNumaCpuExpert, GgufMatchesSingleNode) {
    const int H = 256, I = 256;  // K%256==0 keeps K-quants legal too
    const int E = 6, topk = 3;
    const int T_sweep[] = {1, 8};

    struct Case { ik::GgufType ik_t; GgufQuantType qt; const char* name; };
    // GG-5: GgufQuantType is now the canonical {Q2_K..Q8_0} set (no q5_0). The
    // ik-supported families on the CPU path are Q4_K/Q5_K/Q6_K/Q8_0. q5_0 is no
    // longer reachable through the public device enum, so it is exercised only
    // through the ik vendor directly elsewhere, not here.
    std::vector<Case> cases = {
        {ik::GgufType::q8_0, GgufQuantType::Q8_0, "Q8_0"},
    };
#if defined(LS_IK_HAVE_KQUANTS)
    cases.push_back({ik::GgufType::q4_k, GgufQuantType::Q4_K, "Q4_K"});
    cases.push_back({ik::GgufType::q6_k, GgufQuantType::Q6_K, "Q6_K"});
#endif

    for (const auto& c : cases) {
        if (!ik::gguf_supported(c.ik_t)) {
            fprintf(stderr, "  GgufMatchesSingleNode: %s unsupported — skipped\n",
                    c.name);
            continue;
        }
        std::mt19937 rng(0x6CF + static_cast<unsigned>(c.qt) * 101u);
        auto experts = build_gguf_experts(c.ik_t, E, H, I, rng);

        config::GpuRef ref0{}; ref0.position = 0; ref0.id = 0;
        ref0.type = config::GpuType::cpu;
        NumaCpuExpertDeps sdeps{}; sdeps.dims = CpuExpertModelDims{H, I, topk, E};
        sdeps.max_threads = 0;
        auto single = make_numa_cpu_expert_device(ref0, sdeps);
        single->set_device();

        for (int M : {2, 4}) {
            auto nodes = usable_nodes(M);
            const int m_actual = static_cast<int>(nodes.size());
            config::GpuRef refm{}; refm.position = 0; refm.id = nodes[0];
            refm.type = config::GpuType::cpu;
            MultiNumaCpuExpertDeps mdeps{};
            mdeps.nodes = nodes;
            mdeps.dims = CpuExpertModelDims{H, I, topk, E};
            mdeps.max_threads_per_node = 0;
            auto multi = make_multi_numa_cpu_expert_device(refm, mdeps);
            multi->set_device();

            for (int T : T_sweep) {
                std::vector<int32_t> topk_idx(static_cast<size_t>(T) * topk);
                std::vector<float> topk_w(static_cast<size_t>(T) * topk);
                std::mt19937 rr(0xBEE + static_cast<unsigned>(T * 31 + M));
                for (int t = 0; t < T; ++t) {
                    std::vector<int> pool(E);
                    for (int e = 0; e < E; ++e) pool[e] = e;
                    float wsum = 0.f;
                    for (int k = 0; k < topk; ++k) {
                        std::uniform_int_distribution<int> pick(k, E - 1);
                        std::swap(pool[k], pool[pick(rr)]);
                        topk_idx[t * topk + k] = pool[k];
                        float v = std::fabs(std::uniform_real_distribution<float>(-2, 2)(rr)) + 0.05f;
                        topk_w[t * topk + k] = v; wsum += v;
                    }
                    for (int k = 0; k < topk; ++k) topk_w[t * topk + k] /= wsum;
                }
                std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
                std::uniform_real_distribution<float> ad(-1.f, 1.f);
                for (auto& x : hidden) x = f32_to_bf16(ad(rng));

                std::vector<uint16_t> out_single, out_multi;
                run_gguf_ffn_chain(single.get(), c.qt, H, I, T, topk, E, experts,
                                   hidden, topk_idx, topk_w, out_single);
                run_gguf_ffn_chain(multi.get(), c.qt, H, I, T, topk, E, experts,
                                   hidden, topk_idx, topk_w, out_multi);

                ASSERT_EQ(out_single.size(), out_multi.size());
                // BIT-EXACT: identical BF16 bit patterns (row-disjoint partition).
                for (size_t i = 0; i < out_single.size(); ++i)
                    ASSERT_EQ(out_multi[i], out_single[i])
                        << c.name << " M=" << M << "(actual " << m_actual
                        << ") T=" << T << " i=" << i;
            }
            fprintf(stderr,
                    "  GgufMatchesSingleNode: %s M=%d (%d node(s)) — BIT-EXACT\n",
                    c.name, M, m_actual);
        }
    }
}

// ── ENGINE-DIMS parity under active-set churn (TD-MULTINUMA-RESIDENCY) ────────
//
// REPRODUCES the engine corruption the isolation tests missed. The multi-node
// device stages each expert's per-node weight slice into a REUSED node-local
// buffer at offset e*slice_bytes and caches occupancy. The engine holds the token
// count CONSTANT across decode steps (B=1), so the staging buffer never grows —
// while the active-expert set / order CHURNS every step. A cache keyed
// key->offset (the pre-fix scheme) then reports a stale expert "resident" at an
// offset another expert has since overwritten, and SKIPS staging → an expert is
// computed against ANOTHER expert's weights → corrupted FFN output (the engine's
// forced-traj acceptance 1/100 + halved misses).
//
// This test runs a SEQUENCE of grouped GEMMs at REAL GLM-5.2 expert dims
// (H=6144, I=2048; K=6144/2048 both large) with a FIXED token count but a
// CYCLING active-expert set (A,B,A,B,A) that lands DIFFERENT experts at the SAME
// staging offset — exactly the collision. NVFP4 weights use NON-UNIFORM per-group
// Sm1xx scales (pack_proj_varscale) so the scale-slab slice math at large K is
// exercised too (uniform scales would decode every group to 1.0 and hide it).
// The single-node device (reads arena weights directly, no staging) is the
// reference. Pre-fix: diverges by call 3+. Post-fix: matches every call.
TEST(MultiNumaCpuExpert, EngineDimsParityChurn) {
    const int H = 6144, I = 2048;   // real GLM-5.2 expert dims
    const int E = 16, topk = 1, T = 8;   // P=8 tokens, active=2/call (fixed size)

    std::mt19937 rng(0xE9D1);
    auto rand_w = [&](int N, int K) {
        std::uniform_real_distribution<float> wd(-2.f, 2.f);
        std::vector<float> w(static_cast<size_t>(N) * K);
        for (auto& x : w) x = wd(rng);
        return w;
    };

    // NVFP4 experts with per-(row,group) varying scales.
    std::vector<ExpertWeights> nv(E);
    for (int e = 0; e < E; ++e) {
        const float ws2_g = 0.75f + 0.05f * static_cast<float>(e % 5);
        const float ws2_u = 0.80f + 0.03f * static_cast<float>(e % 7);
        const float ws2_d = 0.90f + 0.02f * static_cast<float>(e % 3);
        nv[e].gate = pack_proj_varscale(rand_w(I, H), I, H, ws2_g, rng);
        nv[e].up   = pack_proj_varscale(rand_w(I, H), I, H, ws2_u, rng);
        nv[e].down = pack_proj_varscale(rand_w(H, I), H, I, ws2_d, rng);
    }

    // GGUF experts (engine's actual fold path). Q8_0 always available; add Q4_K
    // when the k-quant vendor is built.
    struct GgufCase { ik::GgufType t; GgufQuantType qt; const char* name; };
    std::vector<GgufCase> gcases = {{ik::GgufType::q8_0, GgufQuantType::Q8_0, "Q8_0"}};
#if defined(LS_IK_HAVE_KQUANTS)
    if (ik::gguf_supported(ik::GgufType::q4_k))
        gcases.push_back({ik::GgufType::q4_k, GgufQuantType::Q4_K, "Q4_K"});
#endif
    std::vector<std::vector<GgufExpertWeights>> gexp(gcases.size());
    for (size_t ci = 0; ci < gcases.size(); ++ci) {
        gexp[ci].resize(E);
        for (int e = 0; e < E; ++e) {
            gexp[ci][e].gate = pack_gguf(gcases[ci].t, rand_w(I, H), I, H);
            gexp[ci][e].up   = pack_gguf(gcases[ci].t, rand_w(I, H), I, H);
            gexp[ci][e].down = pack_gguf(gcases[ci].t, rand_w(H, I), H, I);
        }
    }

    // Two active-expert sets, ascending order. Each call routes all P tokens
    // between exactly its 2 experts (active=2, total_tokens=P — SAME every call,
    // so the device's staging buffer never grows / never self-clears the cache).
    const int setA[2] = {2, 11};
    const int setB[2] = {5, 9};
    auto make_routing = [&](const int (&set)[2], std::vector<int32_t>& idx,
                            std::vector<float>& w) {
        idx.assign(static_cast<size_t>(T) * topk, 0);
        w.assign(static_cast<size_t>(T) * topk, 1.0f);
        for (int t = 0; t < T; ++t) idx[t] = set[(t < T / 2) ? 0 : 1];
    };

    std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
    std::uniform_real_distribution<float> ad(-1.f, 1.f);
    for (auto& x : hidden) x = f32_to_bf16(ad(rng));

    // Cycling sequence A,B,A,B,A — the repeats (call >=3) are where a stale
    // key->offset cache would fire the corrupting skip.
    const std::vector<int> seq = {0, 1, 0, 1, 0};

    for (int M : {2, 4}) {
        auto nodes = usable_nodes(M);
        const int m_actual = static_cast<int>(nodes.size());

        // --- NVFP4 path (tolerance parity; corruption yields gross errors) ---
        {
            config::GpuRef ref0{}; ref0.position = 0; ref0.id = 0;
            ref0.type = config::GpuType::cpu;
            NumaCpuExpertDeps sd{}; sd.dims = CpuExpertModelDims{H, I, topk, E};
            sd.max_threads = 0;
            auto single = make_numa_cpu_expert_device(ref0, sd);
            single->set_device();
            config::GpuRef refm{}; refm.position = 0; refm.id = nodes[0];
            refm.type = config::GpuType::cpu;
            MultiNumaCpuExpertDeps md{}; md.nodes = nodes;
            md.dims = CpuExpertModelDims{H, I, topk, E};
            md.max_threads_per_node = 0;
            auto multi = make_multi_numa_cpu_expert_device(refm, md);
            multi->set_device();
            for (size_t s = 0; s < seq.size(); ++s) {
                std::vector<int32_t> idx; std::vector<float> w;
                make_routing(seq[s] ? setB : setA, idx, w);
                std::vector<uint16_t> os, om;
                run_ffn_chain(single.get(), H, I, T, topk, E, nv, hidden, idx, w, os);
                run_ffn_chain(multi.get(),  H, I, T, topk, E, nv, hidden, idx, w, om);
                ASSERT_EQ(os.size(), om.size());
                for (size_t i = 0; i < os.size(); ++i) {
                    float a = bf16_to_f32(os[i]), b = bf16_to_f32(om[i]);
                    ASSERT_NEAR(b, a, std::fabs(a) * 0.02f + 0.05f)
                        << "NVFP4 M=" << m_actual << " call=" << s << " i=" << i;
                }
            }
        }

        // --- GGUF path (engine's actual fold; BIT-EXACT parity) ---
        for (size_t ci = 0; ci < gcases.size(); ++ci) {
            if (!ik::gguf_supported(gcases[ci].t)) continue;
            config::GpuRef ref0{}; ref0.position = 0; ref0.id = 0;
            ref0.type = config::GpuType::cpu;
            NumaCpuExpertDeps sd{}; sd.dims = CpuExpertModelDims{H, I, topk, E};
            sd.max_threads = 0;
            auto single = make_numa_cpu_expert_device(ref0, sd);
            single->set_device();
            config::GpuRef refm{}; refm.position = 0; refm.id = nodes[0];
            refm.type = config::GpuType::cpu;
            MultiNumaCpuExpertDeps md{}; md.nodes = nodes;
            md.dims = CpuExpertModelDims{H, I, topk, E};
            md.max_threads_per_node = 0;
            auto multi = make_multi_numa_cpu_expert_device(refm, md);
            multi->set_device();
            for (size_t s = 0; s < seq.size(); ++s) {
                std::vector<int32_t> idx; std::vector<float> w;
                make_routing(seq[s] ? setB : setA, idx, w);
                std::vector<uint16_t> os, om;
                run_gguf_ffn_chain(single.get(), gcases[ci].qt, H, I, T, topk, E,
                                   gexp[ci], hidden, idx, w, os);
                run_gguf_ffn_chain(multi.get(), gcases[ci].qt, H, I, T, topk, E,
                                   gexp[ci], hidden, idx, w, om);
                ASSERT_EQ(os.size(), om.size());
                for (size_t i = 0; i < os.size(); ++i)
                    ASSERT_EQ(om[i], os[i])
                        << "GGUF " << gcases[ci].name << " M=" << m_actual
                        << " call=" << s << " i=" << i;
            }
        }
        fprintf(stderr,
                "MultiNumaCpuExpert.EngineDimsParityChurn: M=%d (%d node(s)) — "
                "NVFP4+GGUF parity held across A,B,A,B,A churn at GLM dims\n",
                M, m_actual);
    }
}

// ── GGUF scaling benchmark: 1-token-1-expert hot unit at M=1/2/4 nodes ────────
//
// Mirrors MultiNumaCpuExpert.ScalingBenchmark but for the GGUF path (Q4_K + Q8_0).
// Times the realistic routed-expert FFN (H=7168, I=2048) for the 1×1×1 hot unit
// at M=1, 2, 4 nodes, reporting compute/stage us, total us, speedup vs the
// dedicated single-node device, and %ideal. Plus a T=8 grouped multibatch row.
// gtest-as-benchmark: asserts only sanity.
TEST(MultiNumaCpuExpert, GgufScalingBenchmark) {
    const int H = 7168, I = 2048, E = 8;

    struct Case { ik::GgufType ik_t; GgufQuantType qt; const char* name; };
    std::vector<Case> cases = {
        {ik::GgufType::q8_0, GgufQuantType::Q8_0, "Q8_0"},
    };
#if defined(LS_IK_HAVE_KQUANTS)
    cases.insert(cases.begin(), {ik::GgufType::q4_k, GgufQuantType::Q4_K, "Q4_K"});
#endif

    const auto all_nodes = usable_nodes(4);
    const int max_M = static_cast<int>(all_nodes.size());
    const auto n0_cores =
        cpu::node_physical_cpus(all_nodes.empty() ? 0 : all_nodes[0]);
#if defined(__AVX512F__)
    const bool avx512 = true;
#else
    const bool avx512 = false;
#endif

    setenv("LS_MULTINUMA_PROFILE", "1", 1);

    fprintf(stderr,
            "\n==== MultiNumaCpuExpert.GgufScalingBenchmark ====\n"
            "  dims: H=%d I=%d  host: usable_nodes=%d cores/node(node0)=%zu "
            "AVX-512=%s\n"
            "  ideal: latency -> 1/M as N output-rows split across M nodes\n",
            H, I, max_M, n0_cores.size(), avx512 ? "yes" : "no");

    auto time_chain = [&](ExpertDevice* dev, GgufQuantType qt,
                          const std::vector<GgufExpertWeights>& experts, int T,
                          int topk, int Eactive, double& stage_us,
                          double& compute_us) {
        std::vector<int32_t> topk_idx(static_cast<size_t>(T) * topk);
        std::vector<float> topk_w(static_cast<size_t>(T) * topk);
        std::mt19937 rr(0xC0FFE + static_cast<unsigned>(T * 7 + topk));
        for (int t = 0; t < T; ++t) {
            std::vector<int> pool(Eactive);
            for (int e = 0; e < Eactive; ++e) pool[e] = e;
            float wsum = 0.f;
            for (int k = 0; k < topk; ++k) {
                std::uniform_int_distribution<int> pick(k, Eactive - 1);
                std::swap(pool[k], pool[pick(rr)]);
                topk_idx[t * topk + k] = pool[k];
                float v = std::fabs(std::uniform_real_distribution<float>(-2, 2)(rr)) + 0.05f;
                topk_w[t * topk + k] = v; wsum += v;
            }
            for (int k = 0; k < topk; ++k) topk_w[t * topk + k] /= wsum;
        }
        std::mt19937 hrng(0x1234u);
        std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
        std::uniform_real_distribution<float> ad(-1.f, 1.f);
        for (auto& x : hidden) x = f32_to_bf16(ad(hrng));
        std::vector<uint16_t> out;

        run_gguf_ffn_chain(dev, qt, H, I, T, topk, Eactive, experts, hidden,
                           topk_idx, topk_w, out);
        bool finite = true;
        for (float v : {bf16_to_f32(out[0]), bf16_to_f32(out.back())})
            if (!std::isfinite(v)) finite = false;
        EXPECT_TRUE(finite);

        auto t_probe0 = std::chrono::steady_clock::now();
        run_gguf_ffn_chain(dev, qt, H, I, T, topk, Eactive, experts, hidden,
                           topk_idx, topk_w, out);
        auto t_probe1 = std::chrono::steady_clock::now();
        const double one_us =
            std::chrono::duration<double, std::micro>(t_probe1 - t_probe0).count();
        int iters = static_cast<int>(150000.0 / std::max(one_us, 1.0));
        iters = std::max(3, std::min(iters, static_cast<int>(20e6 / std::max(one_us, 1.0))));

        multinuma_profile_reset();
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it)
            run_gguf_ffn_chain(dev, qt, H, I, T, topk, Eactive, experts, hidden,
                               topk_idx, topk_w, out);
        auto t1 = std::chrono::steady_clock::now();
        stage_us = (multinuma_profile_stage_ns() / 1e3) / iters;
        compute_us = (multinuma_profile_compute_ns() / 1e3) / iters;
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    };

    for (const auto& c : cases) {
        if (!ik::gguf_supported(c.ik_t)) {
            fprintf(stderr, "  %s: unsupported on this build/host — skipped\n",
                    c.name);
            continue;
        }
        std::mt19937 rng(0x5CA1E + static_cast<unsigned>(c.qt));
        auto experts = build_gguf_experts(c.ik_t, E, H, I, rng);

        // Dedicated single-node baseline (apples-to-apples).
        double single_us = 1e18;
        {
            config::GpuRef ref0{}; ref0.position = 0;
            ref0.id = all_nodes.empty() ? 0 : all_nodes[0];
            ref0.type = config::GpuType::cpu;
            NumaCpuExpertDeps sdeps{}; sdeps.dims = CpuExpertModelDims{H, I, 1, E};
            sdeps.max_threads = 0;
            auto sdev = make_numa_cpu_expert_device(ref0, sdeps);
            sdev->set_device();
            double sst = 0.0, sc = 0.0;
            for (int r = 0; r < 3; ++r)
                single_us = std::min(single_us,
                    time_chain(sdev.get(), c.qt, experts, 1, 1, E, sst, sc));
        }

        fprintf(stderr, "  -- %s 1-token-1-expert hot unit (T=1, topk=1) --\n",
                c.name);
        fprintf(stderr, "  single-node NumaCpuExpert (node0): total_us=%.1f"
                        "  <- baseline\n", single_us);
        fprintf(stderr, "  %-4s %-7s %-10s %-10s %-11s %-9s %-8s\n",
                "M", "nodes", "compute_us", "stage_us", "total_us", "speedup",
                "%ideal");
        for (int M = 1; M <= max_M; M *= 2) {
            auto nodes = usable_nodes(M);
            const int m_actual = static_cast<int>(nodes.size());
            config::GpuRef refm{}; refm.position = 0; refm.id = nodes[0];
            refm.type = config::GpuType::cpu;
            MultiNumaCpuExpertDeps mdeps{};
            mdeps.nodes = nodes;
            mdeps.dims = CpuExpertModelDims{H, I, 1, E};
            mdeps.max_threads_per_node = 0;
            auto dev = make_multi_numa_cpu_expert_device(refm, mdeps);
            dev->set_device();

            double total_us = 1e18, stage_us = 0.0, compute_us = 0.0;
            for (int r = 0; r < 3; ++r) {
                double sst = 0.0, sc = 0.0;
                const double t = time_chain(dev.get(), c.qt, experts, 1, 1, E,
                                            sst, sc);
                if (t < total_us) { total_us = t; stage_us = sst; compute_us = sc; }
            }
            const double speedup = single_us / total_us;
            const double pct_ideal = 100.0 * speedup / static_cast<double>(m_actual);
            fprintf(stderr, "  %-4d %-7d %-10.1f %-10.1f %-11.1f %-9.3f %-8.1f\n",
                    M, m_actual, compute_us, stage_us, total_us, speedup, pct_ideal);
            ASSERT_GT(total_us, 0.0);
        }

        // Multibatch grouped row (T=8, topk=2, full M).
        {
            auto nodes = usable_nodes(max_M);
            config::GpuRef refm{}; refm.position = 0; refm.id = nodes[0];
            refm.type = config::GpuType::cpu;
            MultiNumaCpuExpertDeps mdeps{};
            mdeps.nodes = nodes;
            mdeps.dims = CpuExpertModelDims{H, I, 2, E};
            mdeps.max_threads_per_node = 0;
            auto dev = make_multi_numa_cpu_expert_device(refm, mdeps);
            dev->set_device();
            double stage_us = 0.0, compute_us = 0.0;
            const double total_us = time_chain(dev.get(), c.qt, experts, 8, 2, E,
                                               stage_us, compute_us);
            fprintf(stderr, "  %s M=%d  T=8 topk=2  total_us=%.1f  us/token=%.1f  "
                    "compute_us=%.1f  stage_us=%.1f\n", c.name,
                    static_cast<int>(nodes.size()), total_us, total_us / 8.0,
                    compute_us, stage_us);
            ASSERT_GT(total_us, 0.0);
        }
    }
    fprintf(stderr, "=============================================\n");
}

// ── fp8_grouped_gemm virtual throws (routed experts are nvfp4-sm1xx) ─────────
TEST(NumaCpuExpert, Fp8GroupedGemmThrows) {
    config::GpuRef ref{};
    ref.position = 0; ref.id = 0; ref.type = config::GpuType::cpu;
    NumaCpuExpertDeps deps{};
    deps.max_threads = 1;
    auto dev = make_numa_cpu_expert_device(ref, deps);
    Fp8GroupedGemmParams p{};
    p.num_experts = 1; p.N = 16; p.K = 16;
    EXPECT_THROW(dev->fp8_grouped_gemm(p, nullptr, 0, nullptr),
                 std::runtime_error);
}

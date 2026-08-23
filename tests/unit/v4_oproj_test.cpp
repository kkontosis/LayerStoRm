// V4-5c grouped o_proj tests (ticket G, resolves TD-V4-OPROJ).
//
// DcpExecutor::execute_v4_grouped_oproj — the DeepSeek-V4 2-stage grouped
// low-rank output projection (deepseek4.cpp:1066-1074; NO base+LoRA sum):
//   stage 1: per-group batched GEMM (o_proj_a slabs), stage 2: one shared
//   GEMM (o_proj_b), both on batched_gemm_bf16 (BF16 in, FP32 accumulate).
//
// Group A: GPU goldens vs a CPU double reference on synthetic BF16 tensors
//          (real V4-Flash shapes + a small odd shape; the real-GGUF tensors
//          would hard-depend on /srv paths, so goldens are synthetic).
// Group B: fail-loud contract tests (null devices, no GPU needed).

#include "parallelism/dcp_executor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "compute/csa_hca_sm120_attention_device.h"
#include "core/gpu_ref.h"
#include "core/null_attention_device.h"
#include "../gpu_test_utils.h"

namespace lp = layerstorm::parallelism;
namespace lc = layerstorm::compute;

namespace {

// ── Helpers ─────────────────────────────────────────────────────────────────

float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& v) {
    std::vector<__nv_bfloat16> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = __float2bfloat16_rn(v[i]);
    return out;
}

std::vector<float> random_vec(size_t n, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = bf16_round(dist(rng) * scale);
    return v;
}

struct DeviceBuf {
    void* p = nullptr;
    explicit DeviceBuf(size_t bytes) { cudaMalloc(&p, bytes); }
    ~DeviceBuf() { cudaFree(p); }
};

std::vector<layerstorm::config::GpuRef> gpu_refs(int count) {
    std::vector<layerstorm::config::GpuRef> v;
    for (int i = 0; i < count; ++i)
        v.push_back({.position = i, .id = i,
                     .type = layerstorm::config::GpuType::rtx5090});
    return v;
}

// Options with tiny MLA dims (keeps the executor's MLA scratch small) and
// the V4 grouped-o_proj fields under test.
lp::DcpExecutor::Options v4_opts(std::vector<lc::AttentionDevice*> devs,
                                 int heads, int hidden, int head_dim,
                                 int groups, int olr, int max_batch,
                                 int superchunk = 0) {
    lp::DcpExecutor::Options o{};
    o.dcp_size = static_cast<int>(devs.size());
    o.gpus = gpu_refs(o.dcp_size);
    o.max_batch_size = max_batch;
    o.superchunk_tokens = superchunk;
    o.hidden_size = hidden;
    o.num_attention_heads = heads;
    o.q_lora_rank = 32;
    o.kv_lora_rank = 32;
    o.qk_rope_head_dim = 16;
    o.qk_nope_head_dim = 32;
    o.v_head_dim = 32;
    o.rms_norm_eps = 1e-6f;
    o.v4_head_dim = head_dim;
    o.v4_o_groups = groups;
    o.v4_o_lora_rank = olr;
    o.attention_devices = std::move(devs);
    return o;
}

// CPU double reference of the 2-stage factorization. Stage-1 output is
// rounded to BF16 (the GPU stores oa as BF16 between the stages).
//   x [rows, h_q*head_dim], a [groups*olr, group_dim] (groups-major slabs),
//   b [hidden, groups*olr] — all row-major.
std::vector<double> ref_grouped_oproj(const std::vector<float>& x,
                                      const std::vector<float>& a,
                                      const std::vector<float>& b, int rows,
                                      int h_q, int head_dim, int groups,
                                      int olr, int hidden) {
    const int width = h_q * head_dim;
    const int group_dim = (h_q / groups) * head_dim;
    const int oa_width = groups * olr;
    std::vector<double> oa(static_cast<size_t>(rows) * oa_width, 0.0);
    for (int t = 0; t < rows; ++t) {
        for (int g = 0; g < groups; ++g) {
            for (int m = 0; m < olr; ++m) {
                double acc = 0.0;
                const size_t arow =
                    (static_cast<size_t>(g) * olr + m) * group_dim;
                const size_t xrow = static_cast<size_t>(t) * width
                                  + static_cast<size_t>(g) * group_dim;
                for (int k = 0; k < group_dim; ++k)
                    acc += double(a[arow + k]) * double(x[xrow + k]);
                oa[static_cast<size_t>(t) * oa_width + g * olr + m] =
                    bf16_round(static_cast<float>(acc));
            }
        }
    }
    std::vector<double> out(static_cast<size_t>(rows) * hidden, 0.0);
    for (int t = 0; t < rows; ++t) {
        for (int h = 0; h < hidden; ++h) {
            double acc = 0.0;
            const size_t brow = static_cast<size_t>(h) * oa_width;
            for (int j = 0; j < oa_width; ++j)
                acc += double(b[brow + j])
                     * oa[static_cast<size_t>(t) * oa_width + j];
            out[static_cast<size_t>(t) * hidden + h] = acc;
        }
    }
    return out;
}

// Run the executor path on the GPU and compare against the reference.
void run_golden(int rows, int h_q, int head_dim, int groups, int olr,
                int hidden, int max_batch, int superchunk, uint32_t seed) {
    const int width = h_q * head_dim;
    const int oa_width = groups * olr;
    const int group_dim = (h_q / groups) * head_dim;

    auto dev = lc::make_csa_hca_sm120_attention_device(
        layerstorm::config::GpuRef{.position = 0, .id = 0,
                                   .type = layerstorm::config::GpuType::rtx5090});
    dev->set_device();

    lp::DcpExecutor exec(v4_opts({dev.get()}, h_q, hidden, head_dim, groups,
                                 olr, max_batch, superchunk));

    auto x = random_vec(static_cast<size_t>(rows) * width, seed, 0.5f);
    auto a = random_vec(static_cast<size_t>(oa_width) * group_dim, seed + 1,
                        0.05f);
    auto b = random_vec(static_cast<size_t>(hidden) * oa_width, seed + 2,
                        0.05f);
    auto x_bf = to_bf16(x);
    auto a_bf = to_bf16(a);
    auto b_bf = to_bf16(b);

    DeviceBuf d_x(x_bf.size() * 2), d_a(a_bf.size() * 2), d_b(b_bf.size() * 2);
    DeviceBuf d_out(static_cast<size_t>(rows) * hidden * 2);
    ASSERT_TRUE(d_x.p && d_a.p && d_b.p && d_out.p);
    ASSERT_EQ(cudaMemcpy(d_x.p, x_bf.data(), x_bf.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a.p, a_bf.data(), a_bf.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b.p, b_bf.data(), b_bf.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);

    lp::AttentionLayerWeights w{};
    w.o_proj_a = d_a.p;
    w.o_proj_b = d_b.p;

    exec.execute_v4_grouped_oproj(0, w, d_x.p, rows, d_out.p);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> out_bf(static_cast<size_t>(rows) * hidden);
    ASSERT_EQ(cudaMemcpy(out_bf.data(), d_out.p, out_bf.size() * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);

    auto ref = ref_grouped_oproj(x, a, b, rows, h_q, head_dim, groups, olr,
                                 hidden);

    // BF16 inputs/outputs + FP32 accumulation vs double reference: per-element
    // relative error is bounded by the BF16 output rounding (~0.4%) plus
    // accumulation-order noise. Compare against the reference magnitude scale.
    double ref_scale = 0.0;
    for (double v : ref) ref_scale = std::max(ref_scale, std::abs(v));
    ASSERT_GT(ref_scale, 0.0);
    double max_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_err = std::max(
            max_err, std::abs(__bfloat162float(out_bf[i]) - ref[i]));
    }
    EXPECT_LT(max_err / ref_scale, 1.5e-2)
        << "rows=" << rows << " h_q=" << h_q << " groups=" << groups;
}

std::vector<lc::AttentionDevice*> null_devices(
    std::vector<std::unique_ptr<lc::AttentionDevice>>& store, int count) {
    store.clear();
    std::vector<lc::AttentionDevice*> ptrs;
    for (int i = 0; i < count; ++i) {
        store.push_back(lc::make_null_attention_device(
            layerstorm::config::GpuRef{
                .position = i, .id = i,
                .type = layerstorm::config::GpuType::rtx5090}));
        ptrs.push_back(store.back().get());
    }
    return ptrs;
}

}  // namespace

// ============================================================================
// Group A: GPU goldens (CPU double reference, synthetic BF16 tensors)
// ============================================================================

// Small odd shape: exercises rows 1 / mid / prefill-rows-over-max_batch
// (superchunk bound) and a non-Flash group split.
TEST(V4Oproj, SmallShapeGoldenRows) {
    REQUIRES_GPU();
    // h_q 8, head_dim 32, groups 4 → group_dim 64; olr 16 → oa_width 64.
    run_golden(/*rows=*/1, /*h_q=*/8, /*head_dim=*/32, /*groups=*/4,
               /*olr=*/16, /*hidden=*/128, /*max_batch=*/4,
               /*superchunk=*/0, /*seed=*/101);
    run_golden(/*rows=*/4, 8, 32, 4, 16, 128, 4, 0, 202);
    // Prefill-style: rows 33 > max_batch 4 flows through the superchunk bound.
    run_golden(/*rows=*/33, 8, 32, 4, 16, 128, 4, /*superchunk=*/48, 303);
}

// Real DeepSeek-V4-Flash shape: h_q 64, head_dim 512, 8 groups, rank 1024,
// hidden 4096 (o_proj_a [8192, 4096], o_proj_b [4096, 8192]).
TEST(V4Oproj, FlashShapeGolden) {
    REQUIRES_GPU();
    run_golden(/*rows=*/1, /*h_q=*/64, /*head_dim=*/512, /*groups=*/8,
               /*olr=*/1024, /*hidden=*/4096, /*max_batch=*/4,
               /*superchunk=*/0, /*seed=*/404);
    run_golden(/*rows=*/4, 64, 512, 8, 1024, 4096, 4, 0, 505);
}

// The grouping must be REAL: a single flat [8192, 32768] GEMM over the whole
// attention output would mix groups. Verify that permuting head spans ACROSS
// group boundaries changes the result (i.e. group g only sees its own heads).
TEST(V4Oproj, GroupIsolation) {
    REQUIRES_GPU();
    const int rows = 2, h_q = 8, head_dim = 32, groups = 4, olr = 16,
              hidden = 128;
    const int width = h_q * head_dim, group_dim = (h_q / groups) * head_dim;
    const int oa_width = groups * olr;

    auto dev = lc::make_csa_hca_sm120_attention_device(
        layerstorm::config::GpuRef{.position = 0, .id = 0,
                                   .type = layerstorm::config::GpuType::rtx5090});
    dev->set_device();
    lp::DcpExecutor exec(
        v4_opts({dev.get()}, h_q, hidden, head_dim, groups, olr, 4));

    auto x = random_vec(static_cast<size_t>(rows) * width, 606, 0.5f);
    // Swap the group-0 and group-1 spans of row 0.
    auto x_sw = x;
    for (int k = 0; k < group_dim; ++k)
        std::swap(x_sw[k], x_sw[static_cast<size_t>(group_dim) + k]);
    auto a = random_vec(static_cast<size_t>(oa_width) * group_dim, 607, 0.05f);
    auto b = random_vec(static_cast<size_t>(hidden) * oa_width, 608, 0.05f);

    auto ref = ref_grouped_oproj(x, a, b, rows, h_q, head_dim, groups, olr,
                                 hidden);
    auto ref_sw = ref_grouped_oproj(x_sw, a, b, rows, h_q, head_dim, groups,
                                    olr, hidden);
    double diff = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(hidden); ++i)
        diff = std::max(diff, std::abs(ref[i] - ref_sw[i]));
    ASSERT_GT(diff, 1e-3) << "reference not group-sensitive — test is vacuous";

    auto x_bf = to_bf16(x_sw);
    auto a_bf = to_bf16(a);
    auto b_bf = to_bf16(b);
    DeviceBuf d_x(x_bf.size() * 2), d_a(a_bf.size() * 2), d_b(b_bf.size() * 2);
    DeviceBuf d_out(static_cast<size_t>(rows) * hidden * 2);
    ASSERT_EQ(cudaMemcpy(d_x.p, x_bf.data(), x_bf.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a.p, a_bf.data(), a_bf.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b.p, b_bf.data(), b_bf.size() * 2,
                         cudaMemcpyHostToDevice), cudaSuccess);
    lp::AttentionLayerWeights w{};
    w.o_proj_a = d_a.p;
    w.o_proj_b = d_b.p;
    exec.execute_v4_grouped_oproj(0, w, d_x.p, rows, d_out.p);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__nv_bfloat16> out_bf(static_cast<size_t>(rows) * hidden);
    ASSERT_EQ(cudaMemcpy(out_bf.data(), d_out.p, out_bf.size() * 2,
                         cudaMemcpyDeviceToHost), cudaSuccess);

    // GPU(x_swapped) must match ref(x_swapped), NOT ref(x).
    double err_sw = 0.0, err_orig = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(hidden); ++i) {
        const double v = __bfloat162float(out_bf[i]);
        err_sw = std::max(err_sw, std::abs(v - ref_sw[i]));
        err_orig = std::max(err_orig, std::abs(v - ref[i]));
    }
    double scale = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(hidden); ++i)
        scale = std::max(scale, std::abs(ref_sw[i]));
    EXPECT_LT(err_sw / scale, 1.5e-2);
    EXPECT_GT(err_orig, err_sw * 4.0)
        << "GPU output insensitive to cross-group swap — grouping broken";
}

// ============================================================================
// Group B: fail-loud contract (null devices, no GPU)
// ============================================================================

TEST(V4Oproj, UnconfiguredThrows) {
    std::vector<std::unique_ptr<lc::AttentionDevice>> store;
    auto opts = v4_opts(null_devices(store, 1), 8, 128, /*head_dim=*/0,
                        /*groups=*/0, /*olr=*/0, 4);
    lp::DcpExecutor exec(opts);
    lp::AttentionLayerWeights w{};
    int dummy = 0;
    EXPECT_THROW(exec.execute_v4_grouped_oproj(0, w, &dummy, 1, &dummy),
                 std::runtime_error);
}

// V4-2c (TD-V4-TP resolved): dcp_size 2 runs GROUP-LOCAL stages — rank r's
// o_proj_a carries its group slabs (column shard), its o_proj_b the matching
// K-slice (row shard), and the two PARTIAL hidden outputs must SUM to the
// TP=1 result (the production path allreduces them).
TEST(V4Oproj, Tp2GroupLocalPartialsSumToTp1) {
    REQUIRES_GPU();
    int gpu_count = 0;
    if (cudaGetDeviceCount(&gpu_count) != cudaSuccess || gpu_count < 2)
        GTEST_SKIP() << "needs 2 visible GPUs";
    const int rows = 2, h_q = 8, head_dim = 32, groups = 4, olr = 16,
              hidden = 128, tp = 2;
    const int width = h_q * head_dim, group_dim = (h_q / groups) * head_dim;
    const int oa_width = groups * olr;
    const int hl = h_q / tp, gl = groups / tp;   // per-rank heads / groups
    const int oa_local = gl * olr;

    std::vector<std::unique_ptr<lc::AttentionDevice>> store;
    std::vector<lc::AttentionDevice*> devs;
    for (int r = 0; r < tp; ++r) {
        store.push_back(lc::make_csa_hca_sm120_attention_device(
            layerstorm::config::GpuRef{
                .position = r, .id = r,
                .type = layerstorm::config::GpuType::rtx5090}));
        devs.push_back(store.back().get());
    }
    lp::DcpExecutor exec(
        v4_opts(devs, h_q, hidden, head_dim, groups, olr, 4));

    auto x = random_vec(static_cast<size_t>(rows) * width, 909, 0.5f);
    auto a = random_vec(static_cast<size_t>(oa_width) * group_dim, 910,
                        0.05f);
    auto b = random_vec(static_cast<size_t>(hidden) * oa_width, 911, 0.05f);
    auto ref = ref_grouped_oproj(x, a, b, rows, h_q, head_dim, groups, olr,
                                 hidden);

    std::vector<float> out_sum(static_cast<size_t>(rows) * hidden, 0.0f);
    for (int r = 0; r < tp; ++r) {
        // Rank r inputs: attn_out head slice [rows, hl*head_dim] (heads
        // [hl*r, hl*(r+1))), o_proj_a group slabs [gl*olr, group_dim],
        // o_proj_b K-slice [hidden, oa_local].
        std::vector<float> xr(static_cast<size_t>(rows) * hl * head_dim);
        for (int t = 0; t < rows; ++t)
            std::copy(x.begin() + static_cast<size_t>(t) * width
                          + static_cast<size_t>(r) * hl * head_dim,
                      x.begin() + static_cast<size_t>(t) * width
                          + static_cast<size_t>(r + 1) * hl * head_dim,
                      xr.begin() + static_cast<size_t>(t) * hl * head_dim);
        std::vector<float> ar(
            a.begin() + static_cast<size_t>(r) * oa_local * group_dim,
            a.begin() + static_cast<size_t>(r + 1) * oa_local * group_dim);
        std::vector<float> br(static_cast<size_t>(hidden) * oa_local);
        for (int h = 0; h < hidden; ++h)
            std::copy(b.begin() + static_cast<size_t>(h) * oa_width
                          + static_cast<size_t>(r) * oa_local,
                      b.begin() + static_cast<size_t>(h) * oa_width
                          + static_cast<size_t>(r + 1) * oa_local,
                      br.begin() + static_cast<size_t>(h) * oa_local);

        auto x_bf = to_bf16(xr);
        auto a_bf = to_bf16(ar);
        auto b_bf = to_bf16(br);
        devs[r]->set_device();
        DeviceBuf d_x(x_bf.size() * 2), d_a(a_bf.size() * 2),
            d_b(b_bf.size() * 2);
        DeviceBuf d_out(static_cast<size_t>(rows) * hidden * 2);
        ASSERT_EQ(cudaMemcpy(d_x.p, x_bf.data(), x_bf.size() * 2,
                             cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_a.p, a_bf.data(), a_bf.size() * 2,
                             cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_b.p, b_bf.data(), b_bf.size() * 2,
                             cudaMemcpyHostToDevice), cudaSuccess);
        lp::AttentionLayerWeights w{};
        w.o_proj_a = d_a.p;
        w.o_proj_b = d_b.p;
        exec.execute_v4_grouped_oproj(r, w, d_x.p, rows, d_out.p);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<__nv_bfloat16> ob(static_cast<size_t>(rows) * hidden);
        ASSERT_EQ(cudaMemcpy(ob.data(), d_out.p, ob.size() * 2,
                             cudaMemcpyDeviceToHost), cudaSuccess);
        for (size_t i = 0; i < ob.size(); ++i)
            out_sum[i] += __bfloat162float(ob[i]);
    }

    double scale = 0.0;
    for (double v : ref) scale = std::max(scale, std::abs(v));
    ASSERT_GT(scale, 0.0);
    double max_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
        max_err = std::max(max_err, std::abs(out_sum[i] - ref[i]));
    // Partial BF16 outputs summed in f32 vs the TP=1 double reference —
    // one extra bf16 rounding per partial vs the fused TP=1 path.
    EXPECT_LT(max_err / scale, 2.5e-2);
}

TEST(V4Oproj, RowsOverBoundThrows) {
    std::vector<std::unique_ptr<lc::AttentionDevice>> store;
    auto opts = v4_opts(null_devices(store, 1), 8, 128, 32, 4, 16,
                        /*max_batch=*/4, /*superchunk=*/48);
    lp::DcpExecutor exec(opts);
    lp::AttentionLayerWeights w{};
    int dummy = 0;
    // Bound is max(max_batch, superchunk) = 48.
    EXPECT_THROW(exec.execute_v4_grouped_oproj(0, w, &dummy, 49, &dummy),
                 std::runtime_error);
    EXPECT_THROW(exec.execute_v4_grouped_oproj(0, w, &dummy, 0, &dummy),
                 std::runtime_error);
}

TEST(V4Oproj, NullWeightsThrow) {
    std::vector<std::unique_ptr<lc::AttentionDevice>> store;
    auto opts = v4_opts(null_devices(store, 1), 8, 128, 32, 4, 16, 4);
    lp::DcpExecutor exec(opts);
    lp::AttentionLayerWeights w{};  // o_proj_a/b null
    int dummy = 0;
    EXPECT_THROW(exec.execute_v4_grouped_oproj(0, w, &dummy, 1, &dummy),
                 std::runtime_error);
}

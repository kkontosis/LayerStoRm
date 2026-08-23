// GG-5b: fused-MoE GGUF dispatch param-build (CPU-only, CUDA-free).
//
// The fused production MoE path (dispatch_moe_internal) routes all six GEMM
// sites through daemon::launch_grouped_gemm (dispatch_detail.h). GG-5b added a
// GGUF branch there that builds compute::GgufGroupedGemmParams + calls
// dev->gguf_grouped_gemm instead of the FP8/NVFP4 GEMM. This test pins that
// branch: it drives launch_grouped_gemm with use_gguf and captures the params
// on a recording ExpertDevice, asserting the dense (num_experts==1, 1-element
// B_ptrs) and routed (num_experts==E, scattered B_ptrs) param builds carry the
// right per-projection GgufQuantType, strategy, BF16 A_base, and B_ptrs.
//
// Full numerical e2e (a real GGUF model through the daemon) is GG-8 — that needs
// a GPU + a checkpoint. Here we test what is reachable CUDA-free: that the gate
// is taken (use_gguf) and the kernel param struct is built correctly.

#include "daemon/dispatch_detail.h"
#include "core/expert_device.h"
#include "core/null_device_backend.h"
#include "core/gpu_ref.h"
#include "model/quantization/gguf_kquant.h"
#include "config/config_parser.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace lc = layerstorm::compute;
namespace lm = layerstorm::model;
namespace ld = layerstorm::daemon;
namespace lcfg = layerstorm::config;

namespace {

// Recording ExpertDevice: captures the last gguf_grouped_gemm call's params.
// Subclasses the abstract ExpertDevice directly (NullExpertDevice is `final`).
// All non-GEMM ops are no-ops; the three GEMM ops count their invocations so we
// can assert use_gguf routes to gguf_grouped_gemm and never the FP8/NVFP4 path.
class RecordingExpertDevice final : public lc::ExpertDevice {
public:
    explicit RecordingExpertDevice(lcfg::GpuRef gpu) : gpu_(gpu) {}

    void set_device() override {}
    const lcfg::GpuRef& gpu() const override { return gpu_; }

    // GG-5c: one record per GGUF GEMM, capturing the per-call N/type and the
    // first (and only, for dense/shared) B pointer actually bound into the
    // device B_ptrs array at call time — so a test can pin the fused-vs-split
    // GEMM count, per-projection types, and the split's up-block offset.
    struct GgufCall {
        int num_experts;
        int N;
        int K;
        lc::GgufQuantType type;
        const void* b0;
    };
    std::vector<GgufCall> history;

    void gguf_grouped_gemm(const lc::GgufGroupedGemmParams& p,
                           void* ws, size_t ws_bytes, void* /*stream*/) override {
        last = p;
        last_ws = ws;
        last_ws_bytes = ws_bytes;
        ++gguf_calls;
        const void* b0 = p.B_ptrs ? p.B_ptrs[0] : nullptr;
        history.push_back({p.num_experts, p.N, p.K, p.type, b0});
    }
    void fp8_grouped_gemm(const lc::Fp8GroupedGemmParams&,
                          void*, size_t, void*) override { ++fp8_calls; }
    void nvfp4_grouped_gemm(const lc::Nvfp4GroupedGemmParams&,
                            void*, size_t, void*) override { ++nvfp4_calls; }
    void fused_swiglu(void*, const void*, const lc::FusedSwigluParams&,
                      int, void*) override {}
    void moe_permute(void*, int32_t*, int32_t*, int32_t*,
                     const void*, const int32_t*,
                     int, int, int, int, int, void*, void*) override {}
    void moe_unpermute(void*, const void*, const float*, const int32_t*,
                       int, int, int, int, void*,
                       lc::MoeCombineMode = lc::MoeCombineMode::kReducedBf16)
                       override {}
    void* device_alloc(size_t bytes) override { return std::malloc(bytes); }
    void  device_free(void* ptr) override { std::free(ptr); }

    lc::GgufGroupedGemmParams last{};
    void* last_ws = nullptr;
    size_t last_ws_bytes = 0;
    int gguf_calls = 0;
    int fp8_calls = 0;
    int nvfp4_calls = 0;

private:
    lcfg::GpuRef gpu_;
};

lcfg::GpuRef make_gpu() {
    lcfg::GpuRef g{};
    g.position = 0;
    return g;
}

}  // namespace

// The dispatcher's quant gate uses is_gguf_weight_quant() to take the GGUF path
// (and to NOT abort). Pin that the generic sentinel + all six uniform variants
// are recognized, and FP8/NVFP4 are not.
TEST(GgufFusedMoeGate, IsGgufWeightQuant) {
    using WQ = lcfg::WeightQuant;
    EXPECT_TRUE(lm::gguf::is_gguf_weight_quant(WQ::gguf));
    EXPECT_TRUE(lm::gguf::is_gguf_weight_quant(WQ::gguf_q2_k));
    EXPECT_TRUE(lm::gguf::is_gguf_weight_quant(WQ::gguf_q3_k));
    EXPECT_TRUE(lm::gguf::is_gguf_weight_quant(WQ::gguf_q4_k));
    EXPECT_TRUE(lm::gguf::is_gguf_weight_quant(WQ::gguf_q5_k));
    EXPECT_TRUE(lm::gguf::is_gguf_weight_quant(WQ::gguf_q6_k));
    EXPECT_TRUE(lm::gguf::is_gguf_weight_quant(WQ::gguf_q8_0));
    EXPECT_FALSE(lm::gguf::is_gguf_weight_quant(WQ::fp8_e4m3));
    EXPECT_FALSE(lm::gguf::is_gguf_weight_quant(WQ::nvfp4));
}

// The per-projection type cast the dispatcher uses: GgufQuantInterface returns a
// model::GgufKQuantType whose ordinal matches compute::GgufQuantType, so the
// engine→kernel mapping is a by-value cast (mixed gate/up/down preserved).
TEST(GgufFusedMoeGate, ProjectionTypeCastMatches) {
    auto qi = lm::make_gguf_quant(lm::GgufKQuantType::Q4_K,
                                  lm::GgufKQuantType::Q6_K,
                                  lm::GgufKQuantType::Q8_0);
    auto cast = [](lm::GgufKQuantType t) {
        return static_cast<lc::GgufQuantType>(static_cast<int>(t));
    };
    EXPECT_EQ(cast(qi.projection_type(lm::Projection::gate)), lc::GgufQuantType::Q4_K);
    EXPECT_EQ(cast(qi.projection_type(lm::Projection::up)),   lc::GgufQuantType::Q6_K);
    EXPECT_EQ(cast(qi.projection_type(lm::Projection::down)), lc::GgufQuantType::Q8_0);
}

// Dense/shared GGUF GEMM: num_experts==1, BF16 A_base fed directly, single
// weight wrapped in a 1-element B_ptrs array, no FP8/NVFP4 GEMM taken.
TEST(GgufFusedMoeDispatch, DenseSharedSingleExpertParams) {
    RecordingExpertDevice dev{make_gpu()};

    // Stand-in device buffers (pointers only — NullExpertDevice never reads them).
    int bf16_activation = 0;     // BF16 A_base (norm_input / shared_activation)
    int output = 0;              // D_base
    void* single_b_ptr_array = std::malloc(sizeof(void*));  // [1] void*
    int32_t expert_offsets[2] = {0, 4};
    int gemm_ws = 0;

    const int H = 2048, I2 = 1024;  // N = 2*intermediate_local for gate+up

    ld::GroupedGemmArgs g{/*use_fp8=*/false, /*num_experts=*/1, /*N=*/I2, /*K=*/H,
        /*A_base=*/&bf16_activation, /*B_base=*/nullptr, /*D_base=*/&output,
        /*scale_A_base=*/nullptr, /*scale_B_base=*/nullptr, /*alphas=*/nullptr,
        /*expert_offsets=*/expert_offsets, /*sf_offsets=*/nullptr,
        /*problem_sizes=*/nullptr, /*dev=*/&dev,
        /*gemm_workspace=*/&gemm_ws, /*gemm_workspace_bytes=*/64, /*stream=*/nullptr};
    g.use_gguf = true;
    g.gguf_type = lc::GgufQuantType::Q4_K;
    g.gguf_strategy = lc::GgufGemmStrategy::int_strategy;
    g.B_ptrs = static_cast<const void**>(single_b_ptr_array);

    ld::launch_grouped_gemm(g);

    EXPECT_EQ(dev.gguf_calls, 1);
    EXPECT_EQ(dev.fp8_calls, 0);
    EXPECT_EQ(dev.nvfp4_calls, 0);
    EXPECT_EQ(dev.last.num_experts, 1);
    EXPECT_EQ(dev.last.N, I2);
    EXPECT_EQ(dev.last.K, H);
    EXPECT_EQ(dev.last.type, lc::GgufQuantType::Q4_K);
    EXPECT_EQ(dev.last.strategy, lc::GgufGemmStrategy::int_strategy);
    // BF16 activation fed directly as A_base (no FP8/NVFP4 pre-quant).
    EXPECT_EQ(dev.last.A_base, &bf16_activation);
    EXPECT_EQ(dev.last.D_base, &output);
    EXPECT_EQ(dev.last.expert_offsets, expert_offsets);
    EXPECT_EQ(dev.last.B_ptrs, static_cast<const void**>(single_b_ptr_array));
    EXPECT_EQ(dev.last_ws, &gemm_ws);
    EXPECT_EQ(dev.last_ws_bytes, 64u);

    std::free(single_b_ptr_array);
}

// Routed GGUF GEMM: num_experts==E, scattered per-expert B_ptrs reused from the
// cache fill, dequant strategy threaded, per-projection down-type Q6_K.
TEST(GgufFusedMoeDispatch, RoutedScatteredBptrsParams) {
    RecordingExpertDevice dev{make_gpu()};

    int permuted_input = 0;  // BF16 A_base
    int expert_output = 0;
    const int E = 8;
    std::vector<const void*> b_ptrs(E, nullptr);  // routed_b_ptrs device array stand-in
    std::vector<int32_t> offsets(E + 1, 0);
    int gemm_ws = 0;

    const int H = 2048, I = 1024;  // down: N=H, K=I

    ld::GroupedGemmArgs g{/*use_fp8=*/false, /*num_experts=*/E, /*N=*/H, /*K=*/I,
        /*A_base=*/&permuted_input, /*B_base=*/nullptr, /*D_base=*/&expert_output,
        /*scale_A_base=*/nullptr, /*scale_B_base=*/nullptr, /*alphas=*/nullptr,
        /*expert_offsets=*/offsets.data(), /*sf_offsets=*/nullptr,
        /*problem_sizes=*/nullptr, /*dev=*/&dev,
        /*gemm_workspace=*/&gemm_ws, /*gemm_workspace_bytes=*/128, /*stream=*/nullptr};
    g.B_ptrs = b_ptrs.data();
    // scale_B_ptrs intentionally left null for GGUF (intra-block scales).
    g.use_gguf = true;
    g.gguf_type = lc::GgufQuantType::Q6_K;
    g.gguf_strategy = lc::GgufGemmStrategy::dequant;

    ld::launch_grouped_gemm(g);

    EXPECT_EQ(dev.gguf_calls, 1);
    EXPECT_EQ(dev.fp8_calls, 0);
    EXPECT_EQ(dev.nvfp4_calls, 0);
    EXPECT_EQ(dev.last.num_experts, E);
    EXPECT_EQ(dev.last.N, H);
    EXPECT_EQ(dev.last.K, I);
    EXPECT_EQ(dev.last.type, lc::GgufQuantType::Q6_K);
    EXPECT_EQ(dev.last.strategy, lc::GgufGemmStrategy::dequant);
    EXPECT_EQ(dev.last.A_base, &permuted_input);
    EXPECT_EQ(dev.last.B_ptrs, b_ptrs.data());
    EXPECT_EQ(dev.last.expert_offsets, offsets.data());
}

// GG-5c: build a dense/shared gate_up GgufDenseGateUpArgs against a fake weight
// base + the real launch_gguf_dense_gate_up helper, driving a RecordingExpert
// device (GEMMs) and a NullDeviceBackend (the B-pointer H2D actually copies, so
// the bound pointer is observable per call).
namespace {
struct GateUpHarness {
    RecordingExpertDevice dev{make_gpu()};
    layerstorm::compute::NullDeviceBackend gpu_dev{make_gpu()};
    const void* gate_up_weight = reinterpret_cast<const void*>(0x10000);
    void* single_b_ptr = std::malloc(sizeof(void*));
    int32_t expert_offsets[2] = {0, 4};
    // Real-sized BF16 scratch: the split path's NullDeviceBackend memcpy_2d_async
    // actually interleaves gate_scratch+up_scratch ([B,I]) into gate_up_output
    // ([B,2*I]), so these must hold the test dims (B=4, I=1024).
    std::vector<uint8_t> gate_up_output = std::vector<uint8_t>(4 * 2 * 1024 * 2, 0);
    std::vector<uint8_t> gate_scratch = std::vector<uint8_t>(4 * 1024 * 2, 0);
    std::vector<uint8_t> up_scratch = std::vector<uint8_t>(4 * 1024 * 2, 0);
    int gemm_ws = 0;
    ~GateUpHarness() { std::free(single_b_ptr); }

    ld::GgufDenseGateUpArgs make(int num_tokens, int I, int H,
                                 lc::GgufQuantType gate, lc::GgufQuantType up,
                                 lm::GgufKQuantType gate_kq) {
        ld::GgufDenseGateUpArgs a{};
        a.num_tokens = num_tokens;
        a.intermediate_local = I;
        a.hidden = H;
        a.a_base = gate_up_output.data();  // any BF16 input pointer
        a.gate_up_output = gate_up_output.data();
        a.gate_scratch = gate_scratch.data();
        a.up_scratch = up_scratch.data();
        a.gate_type = gate;
        a.up_type = up;
        a.up_block_offset = lm::gguf::gguf_packed_bytes(I, H, gate_kq);
        a.strategy = lc::GgufGemmStrategy::int_strategy;
        a.gate_up_weight = gate_up_weight;
        a.single_b_ptr = single_b_ptr;
        a.expert_offsets = expert_offsets;
        a.dev = &dev;
        a.gpu_dev = &gpu_dev;
        a.gemm_workspace = &gemm_ws;
        a.gemm_workspace_bytes = 64;
        a.stream = nullptr;
        return a;
    }
};
}  // namespace

// Fused: gate_type == up_type ⇒ ONE [2*I, H] GEMM with that shared type, fed the
// gate‖up base pointer directly. This is the fast common path.
TEST(GgufFusedMoeGateUp, FusedWhenGateEqualsUp) {
    GateUpHarness h;
    const int I = 1024, H = 2048;
    auto a = h.make(/*num_tokens=*/4, I, H,
                    lc::GgufQuantType::Q4_K, lc::GgufQuantType::Q4_K,
                    lm::GgufKQuantType::Q4_K);
    const int n = ld::launch_gguf_dense_gate_up(a);

    EXPECT_EQ(n, 1);
    ASSERT_EQ(h.dev.history.size(), 1u);
    EXPECT_EQ(h.dev.history[0].num_experts, 1);
    EXPECT_EQ(h.dev.history[0].N, 2 * I);            // fused width
    EXPECT_EQ(h.dev.history[0].K, H);
    EXPECT_EQ(h.dev.history[0].type, lc::GgufQuantType::Q4_K);
    EXPECT_EQ(h.dev.history[0].b0, h.gate_up_weight);  // base, no offset
    EXPECT_EQ(h.dev.last.D_base, h.gate_up_output.data());
}

// Split: gate_type != up_type ⇒ TWO [I, H] GEMMs — gate (gate_type, base) then up
// (up_type, base + gguf_packed_bytes(I,H,gate)). Each carries its OWN type; the up
// block lives immediately after the gate block. Mirrors the routed gate/up split.
TEST(GgufFusedMoeGateUp, SplitWhenGateDiffersFromUp) {
    GateUpHarness h;
    const int I = 1024, H = 2048;
    // Dense/shared gate=Q4_K, up=Q6_K — distinct from any routed type. The helper
    // takes these directly (decoupled from Deps::gguf_quant), so this pins that
    // the dense/shared OWN types drive the GEMMs.
    auto a = h.make(/*num_tokens=*/4, I, H,
                    lc::GgufQuantType::Q4_K, lc::GgufQuantType::Q6_K,
                    lm::GgufKQuantType::Q4_K);
    const int n = ld::launch_gguf_dense_gate_up(a);

    EXPECT_EQ(n, 2);
    ASSERT_EQ(h.dev.history.size(), 2u);

    // GEMM 0 = gate: N=I, gate_type, at the weight base.
    EXPECT_EQ(h.dev.history[0].N, I);
    EXPECT_EQ(h.dev.history[0].K, H);
    EXPECT_EQ(h.dev.history[0].type, lc::GgufQuantType::Q4_K);
    EXPECT_EQ(h.dev.history[0].b0, h.gate_up_weight);

    // GEMM 1 = up: N=I, up_type, at base + gate-block packed bytes.
    const int64_t gate_bytes =
        lm::gguf::gguf_packed_bytes(I, H, lm::GgufKQuantType::Q4_K);
    EXPECT_EQ(h.dev.history[1].N, I);
    EXPECT_EQ(h.dev.history[1].K, H);
    EXPECT_EQ(h.dev.history[1].type, lc::GgufQuantType::Q6_K);
    EXPECT_EQ(h.dev.history[1].b0,
              static_cast<const std::byte*>(h.gate_up_weight) + gate_bytes);

    EXPECT_EQ(h.dev.fp8_calls, 0);
    EXPECT_EQ(h.dev.nvfp4_calls, 0);
}

// Sanity: with use_gguf false the helper still routes to the FP8/NVFP4 GEMM,
// never the GGUF one (the GGUF branch must be opt-in).
TEST(GgufFusedMoeDispatch, NonGgufDoesNotTakeGgufBranch) {
    RecordingExpertDevice dev{make_gpu()};
    int a = 0, d = 0;
    int32_t offsets[2] = {0, 1};
    int ws = 0;
    ld::GroupedGemmArgs g{/*use_fp8=*/true, 1, 64, 64,
        &a, &a, &d, &a, &a, nullptr, offsets, nullptr, nullptr, &dev,
        &ws, 0, nullptr};
    ld::launch_grouped_gemm(g);
    EXPECT_EQ(dev.gguf_calls, 0);
    EXPECT_EQ(dev.fp8_calls, 1);
}

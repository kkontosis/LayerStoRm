// Unit tests for KvBvDequantPool slot lifecycle (CPU-only, NullAttentionDevice).

#include <gtest/gtest.h>

#include "core/gpu_ref.h"
#include "core/null_attention_device.h"
#include "parallelism/dcp_executor.h"       // AttentionLayerWeights
#include "parallelism/kv_bv_dequant_pool.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace lp = layerstorm::parallelism;
namespace lc = layerstorm::compute;
namespace cfg = layerstorm::config;

// ── Test fixture ────────────────────────────────────────────────────────────

class KvBvDequantPoolTest : public ::testing::Test {
protected:
    // Small dims for fast CPU-only testing.
    static constexpr int HL  = 2;
    static constexpr int P   = 128;
    static constexpr int V   = 128;
    static constexpr int D_c = 512;

    // Bytes for a fake kv_b_proj: [HL*(P+V), D_c] — 1 byte per FP8 element.
    static constexpr size_t kKvBProjBytes = HL * (P + V) * D_c;

    // Bytes for BF16 kv_b_proj: 2 bytes per element.
    static constexpr size_t kKvBProjBf16Bytes = HL * (P + V) * D_c * 2;

    void SetUp() override {
        cfg::GpuRef gpu{0, 0, cfg::GpuType::rtx5090};
        attn_device_ = lc::make_null_attention_device(gpu);

        // Allocate dummy kv_b_proj buffers (FP8 and BF16 variants).
        fp8_buf_ = std::malloc(kKvBProjBytes);
        std::memset(fp8_buf_, 0x42, kKvBProjBytes);

        bf16_buf_ = std::malloc(kKvBProjBf16Bytes);
        std::memset(bf16_buf_, 0x11, kKvBProjBf16Bytes);

        // Dummy FP8 scale buffer.
        scales_buf_ = std::malloc(1024 * sizeof(float));
    }

    void TearDown() override {
        std::free(fp8_buf_);
        std::free(bf16_buf_);
        std::free(scales_buf_);
    }

    lp::KvBvDequantPool::Options make_opts() {
        lp::KvBvDequantPool::Options opts;
        opts.dcp_size = 1;
        opts.num_slots = 5;
        opts.num_heads_local = HL;
        opts.qk_nope_head_dim = P;
        opts.v_head_dim = V;
        opts.kv_lora_rank = D_c;
        opts.attention_devices = {attn_device_.get()};
        opts.stream_manager = nullptr;
        return opts;
    }

    lp::AttentionLayerWeights make_fp8_weights() {
        lp::AttentionLayerWeights w{};
        w.kv_b_proj = fp8_buf_;
        w.kv_b_proj_scales = scales_buf_;
        w.kv_b_proj_is_fp8 = true;
        return w;
    }

    lp::AttentionLayerWeights make_bf16_weights() {
        lp::AttentionLayerWeights w{};
        w.kv_b_proj = bf16_buf_;
        w.kv_b_proj_scales = nullptr;
        w.kv_b_proj_is_fp8 = false;
        return w;
    }

    // GG-7b: a GGUF kv_b has NO FP8 scales (kv_b_proj_is_fp8 == false) but MUST
    // still route through a dequant slot (never the BF16-direct pointer into
    // packed bytes). Q4_K (QK=256) divides D_c=512 → passes the L%QK guard.
    lp::AttentionLayerWeights make_gguf_weights() {
        lp::AttentionLayerWeights w{};
        w.kv_b_proj = fp8_buf_;            // reuse a 1-byte-per-elem packed buffer
        w.kv_b_proj_scales = nullptr;      // GGUF scales are in-block
        w.kv_b_proj_is_fp8 = false;        // the crux: NOT flagged fp8
        w.kv_b_is_gguf = true;
        w.kv_b_gguf_type = layerstorm::model::GgufKQuantType::Q4_K;
        return w;
    }

    std::unique_ptr<lc::AttentionDevice> attn_device_;
    void* fp8_buf_ = nullptr;
    void* bf16_buf_ = nullptr;
    void* scales_buf_ = nullptr;
};

// ── Tests ───────────────────────────────────────────────────────────────────

TEST_F(KvBvDequantPoolTest, Construction) {
    // Pool creates 5 slots, destructor frees without crash.
    lp::KvBvDequantPool pool(make_opts());
}

TEST_F(KvBvDequantPoolTest, BF16DirectPath) {
    lp::KvBvDequantPool pool(make_opts());

    auto w = make_bf16_weights();
    auto result = pool.acquire(5, 0, w, nullptr);

    EXPECT_TRUE(result.is_direct);
    // Stride should be (P+V)*D_c elements between heads.
    EXPECT_EQ(result.stride_a, static_cast<int64_t>(P + V) * D_c);

    // weight_ptr should be offset by P*D_c*sizeof(BF16) from kv_b_proj base.
    const auto* base = static_cast<const char*>(bf16_buf_);
    const auto* expected = base + static_cast<size_t>(P) * D_c * 2;
    EXPECT_EQ(result.weight_ptr, static_cast<const void*>(expected));
}

// GG-7b crux: a GGUF kv_b (no FP8 scales) must NOT return the BF16-direct
// pointer into packed GGUF bytes — it must dequant into a slot
// (is_direct == false), else the batched value GEMM reads packed bytes as BF16
// (TD-GG7-KVB-V-PATH-GGUF). This is the value-side mirror of the W_UK fix.
TEST_F(KvBvDequantPoolTest, GgufRoutesThroughDequantSlotNotDirect) {
    lp::KvBvDequantPool pool(make_opts());

    auto w = make_gguf_weights();
    auto result = pool.acquire(5, 0, w, nullptr);

    EXPECT_FALSE(result.is_direct)
        << "GGUF kv_b must dequant through a slot, not return the packed-direct "
           "pointer";
    // Contiguous dequanted output stride V*D_c (same as the FP8 dequant slot).
    EXPECT_EQ(result.stride_a, static_cast<int64_t>(V) * D_c);
    EXPECT_NE(result.weight_ptr, nullptr);

    // And the pointer is a pool slot buffer, NOT the packed kv_b_proj region.
    const auto* base = static_cast<const char*>(fp8_buf_);
    const auto* packed_direct = base + static_cast<size_t>(P) * D_c * 2;
    EXPECT_NE(result.weight_ptr, static_cast<const void*>(packed_direct));
}

// GG-7b: GGUF also routes through prime()/schedule_dequant() like FP8 (not the
// BF16 no-op short-circuit).
TEST_F(KvBvDequantPoolTest, GgufPrimeAndScheduleConsumeSlots) {
    lp::KvBvDequantPool pool(make_opts());

    auto w = make_gguf_weights();
    const lp::AttentionLayerWeights* w_ptr = &w;

    pool.prime(0, &w_ptr);
    auto r0 = pool.acquire(0, 0, w, nullptr);
    EXPECT_FALSE(r0.is_direct);
    EXPECT_NE(r0.weight_ptr, nullptr);

    pool.schedule_dequant(5, &w_ptr);
    auto r5 = pool.acquire(5, 0, w, nullptr);
    EXPECT_FALSE(r5.is_direct);
    EXPECT_NE(r5.weight_ptr, nullptr);
    EXPECT_NE(r0.weight_ptr, r5.weight_ptr);
}

TEST_F(KvBvDequantPoolTest, FP8AcquireNoPreSchedule) {
    lp::KvBvDequantPool pool(make_opts());

    auto w = make_fp8_weights();
    auto result = pool.acquire(5, 0, w, nullptr);

    EXPECT_FALSE(result.is_direct);
    // Stride is V*D_c for contiguous dequanted output.
    EXPECT_EQ(result.stride_a, static_cast<int64_t>(V) * D_c);
    EXPECT_NE(result.weight_ptr, nullptr);
}

TEST_F(KvBvDequantPoolTest, PrimePermanentSlots) {
    lp::KvBvDequantPool pool(make_opts());

    auto w = make_fp8_weights();
    const lp::AttentionLayerWeights* w_ptr = &w;

    pool.prime(0, &w_ptr);
    pool.prime(1, &w_ptr);

    // Acquiring primed layers should return ready slots.
    auto r0 = pool.acquire(0, 0, w, nullptr);
    EXPECT_FALSE(r0.is_direct);
    EXPECT_NE(r0.weight_ptr, nullptr);

    auto r1 = pool.acquire(1, 0, w, nullptr);
    EXPECT_FALSE(r1.is_direct);
    EXPECT_NE(r1.weight_ptr, nullptr);

    // Different slots.
    EXPECT_NE(r0.weight_ptr, r1.weight_ptr);
}

TEST_F(KvBvDequantPoolTest, ScheduleThenAcquire) {
    lp::KvBvDequantPool pool(make_opts());

    auto w = make_fp8_weights();
    const lp::AttentionLayerWeights* w_ptr = &w;

    pool.schedule_dequant(5, &w_ptr);

    auto result = pool.acquire(5, 0, w, nullptr);
    EXPECT_FALSE(result.is_direct);
    EXPECT_NE(result.weight_ptr, nullptr);
    EXPECT_EQ(result.stride_a, static_cast<int64_t>(V) * D_c);
}

TEST_F(KvBvDequantPoolTest, EvictionOrder) {
    lp::KvBvDequantPool pool(make_opts());

    auto w = make_fp8_weights();
    const lp::AttentionLayerWeights* w_ptr = &w;

    // Prime permanent slots 0,1.
    pool.prime(0, &w_ptr);
    pool.prime(1, &w_ptr);

    // Fill all 3 rotating slots with layers 2,3,4.
    // Must acquire each to transition from kDequanting → kReady (evictable).
    pool.schedule_dequant(2, &w_ptr);
    pool.acquire(2, 0, w, nullptr);
    pool.schedule_dequant(3, &w_ptr);
    pool.acquire(3, 0, w, nullptr);
    pool.schedule_dequant(4, &w_ptr);
    pool.acquire(4, 0, w, nullptr);

    // All 3 rotating slots now kReady. Schedule layer 5 — evicts oldest (layer 2).
    pool.schedule_dequant(5, &w_ptr);

    auto r5 = pool.acquire(5, 0, w, nullptr);
    EXPECT_FALSE(r5.is_direct);
    EXPECT_NE(r5.weight_ptr, nullptr);

    // Layer 2 was evicted — acquire triggers synchronous dequant into an evictable slot.
    auto r2 = pool.acquire(2, 0, w, nullptr);
    EXPECT_FALSE(r2.is_direct);
    EXPECT_NE(r2.weight_ptr, nullptr);
}

TEST_F(KvBvDequantPoolTest, PermanentSlotsProtected) {
    lp::KvBvDequantPool pool(make_opts());

    auto w = make_fp8_weights();
    const lp::AttentionLayerWeights* w_ptr = &w;

    // Prime permanent slots.
    pool.prime(0, &w_ptr);
    pool.prime(1, &w_ptr);

    auto r0_before = pool.acquire(0, 0, w, nullptr);

    // Fill all rotating slots and force eviction.
    pool.schedule_dequant(10, &w_ptr);
    pool.schedule_dequant(11, &w_ptr);
    pool.schedule_dequant(12, &w_ptr);
    pool.schedule_dequant(13, &w_ptr);  // triggers eviction of a rotating slot

    // Permanent slots still accessible with same pointer.
    auto r0_after = pool.acquire(0, 0, w, nullptr);
    EXPECT_EQ(r0_before.weight_ptr, r0_after.weight_ptr);

    auto r1 = pool.acquire(1, 0, w, nullptr);
    EXPECT_NE(r1.weight_ptr, nullptr);
}

TEST_F(KvBvDequantPoolTest, ReleaseMakesSlotEvictable) {
    lp::KvBvDequantPool pool(make_opts());

    auto w = make_fp8_weights();
    const lp::AttentionLayerWeights* w_ptr = &w;

    pool.prime(0, &w_ptr);
    pool.prime(1, &w_ptr);

    // Schedule and acquire layer 5.
    pool.schedule_dequant(5, &w_ptr);
    auto r5 = pool.acquire(5, 0, w, nullptr);
    pool.release(5);

    // Fill remaining rotating slots.
    pool.schedule_dequant(6, &w_ptr);
    pool.schedule_dequant(7, &w_ptr);

    // Schedule layer 8 — should evict the released layer 5's slot.
    pool.schedule_dequant(8, &w_ptr);
    auto r8 = pool.acquire(8, 0, w, nullptr);
    EXPECT_FALSE(r8.is_direct);
    EXPECT_NE(r8.weight_ptr, nullptr);
}

TEST_F(KvBvDequantPoolTest, BF16ScheduleNoop) {
    lp::KvBvDequantPool pool(make_opts());

    auto w_bf16 = make_bf16_weights();
    const lp::AttentionLayerWeights* w_ptr = &w_bf16;

    // Scheduling BF16 weights should be a no-op (no slot consumed).
    pool.schedule_dequant(5, &w_ptr);
    pool.schedule_dequant(6, &w_ptr);
    pool.schedule_dequant(7, &w_ptr);

    // All rotating slots should still be free for FP8 use.
    auto w_fp8 = make_fp8_weights();
    const lp::AttentionLayerWeights* fp8_ptr = &w_fp8;

    pool.schedule_dequant(10, &fp8_ptr);
    pool.schedule_dequant(11, &fp8_ptr);
    pool.schedule_dequant(12, &fp8_ptr);

    // All 3 should succeed without eviction issues.
    auto r10 = pool.acquire(10, 0, w_fp8, nullptr);
    auto r11 = pool.acquire(11, 0, w_fp8, nullptr);
    auto r12 = pool.acquire(12, 0, w_fp8, nullptr);

    EXPECT_NE(r10.weight_ptr, nullptr);
    EXPECT_NE(r11.weight_ptr, nullptr);
    EXPECT_NE(r12.weight_ptr, nullptr);

    // All three should be different buffers.
    EXPECT_NE(r10.weight_ptr, r11.weight_ptr);
    EXPECT_NE(r11.weight_ptr, r12.weight_ptr);
    EXPECT_NE(r10.weight_ptr, r12.weight_ptr);
}

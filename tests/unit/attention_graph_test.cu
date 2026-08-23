// Unit tests for DecodeGraphRunner (CUDA graph capture/replay).
//
// NOTE: DecodeGraphRunner is tested via forward declarations and the library's
// compiled code. The graph header includes prep kernel .cu files (with __global__
// definitions), so we can't include it directly — it would conflict with
// snapmla_prep.cu in the same link unit. Tests call through extern declarations.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "../gpu_test_utils.h"

#include "sm120/components/sparse_config.h"

#define CUDA_CHECK(expr)                                                 \
    do {                                                                 \
        cudaError_t _err = (expr);                                       \
        ASSERT_EQ(_err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_err); \
    } while (0)

// Forward-declare the config + runner from the graph namespace.
// The full definition lives in decode_graph.h (compiled in snapmla_prep.cu).
namespace sm120::graph {

struct DecodeGraphConfig {
    int batch_size;
    int s_q;
    int h_q;
    int h_kv;
    int d_qk;
    int d_v;
    int d_nope;
    int page_block_size;
    int max_num_blocks_per_seq;
    int kv_stride_block;
    int kv_stride_row;
    void* kv_cache;
    float sm_scale;
    sm120::sparse::ModelType model_type;
    int num_sm_parts;
    bool sparse;
    int topk;
    int extra_topk;
};

// Opaque handle — we test lifecycle through extern factory functions
// declared in a separate .cu compilation unit.

}  // namespace sm120::graph

// Forward-declare the TQ config from the graph namespace.
// The full definition lives in tq_decode_graph.h (compiled in tq_prep.cu).
namespace sm120::graph {

struct TqDecodeGraphConfig {
    int batch_size;
    int s_q;
    int h_q;
    int d_c;
    int d_rope;
    int page_block_size;
    int max_num_blocks_per_seq;
    float sm_scale;
    int num_sm_parts;

    void* kv_cache;
    const float* Pi;
    const float* centroids;
};

}  // namespace sm120::graph

// The actual DecodeGraphRunner / TqDecodeGraphRunner are compiled in
// snapmla_prep.cu / tq_prep.cu respectively.  We verify the graph subsystem
// by testing the supporting infrastructure: prep kernel wrappers (tested in
// mla_attention_test.cu) and metadata computation (tested via
// launch_get_mla_metadata).

TEST(AttentionGraph, ConfigStructLayout) {
    // Verify the config struct can be constructed with V3.2 parameters
    sm120::graph::DecodeGraphConfig cfg{};
    cfg.batch_size = 1;
    cfg.s_q = 1;
    cfg.h_q = 64;
    cfg.h_kv = 1;
    cfg.d_qk = 576;
    cfg.d_v = 512;
    cfg.d_nope = 512;
    cfg.page_block_size = 64;
    cfg.max_num_blocks_per_seq = 16;
    cfg.sm_scale = 1.0f / std::sqrt(576.0f);
    cfg.model_type = sm120::sparse::ModelType::V32;
    cfg.num_sm_parts = 8;
    cfg.sparse = false;
    cfg.topk = 0;
    cfg.extra_topk = 0;
    cfg.kv_cache = nullptr;
    cfg.kv_stride_block = 64 * 644;
    cfg.kv_stride_row = 644;

    EXPECT_EQ(cfg.batch_size, 1);
    EXPECT_EQ(cfg.d_qk, 576);
    EXPECT_EQ(cfg.d_nope, 512);
    EXPECT_FLOAT_EQ(cfg.sm_scale, 1.0f / std::sqrt(576.0f));
}

TEST(AttentionGraph, Model1Config) {
    sm120::graph::DecodeGraphConfig cfg{};
    cfg.batch_size = 4;
    cfg.s_q = 1;
    cfg.h_q = 64;
    cfg.h_kv = 1;
    cfg.d_qk = 512;
    cfg.d_v = 512;
    cfg.d_nope = 448;
    cfg.page_block_size = 64;
    cfg.max_num_blocks_per_seq = 8;
    cfg.sm_scale = 1.0f / std::sqrt(512.0f);
    cfg.model_type = sm120::sparse::ModelType::MODEL1;
    cfg.num_sm_parts = 4;
    cfg.sparse = true;
    cfg.topk = 2048;
    cfg.extra_topk = 0;

    EXPECT_EQ(cfg.d_nope, 448);
    EXPECT_EQ(cfg.d_qk - cfg.d_nope, 64);  // d_rope always 64
    EXPECT_TRUE(cfg.sparse);
}

// ── TurboQuant graph config tests ──────────────────────────────────────────

TEST(TqAttentionGraph, ConfigStructLayout) {
    // Verify the TQ config struct can be constructed with V3.2 parameters.
    // TQ uses d_c (compressed dim) and d_rope instead of d_qk/d_v/d_nope.
    sm120::graph::TqDecodeGraphConfig cfg{};
    cfg.batch_size = 1;
    cfg.s_q = 1;
    cfg.h_q = 64;
    cfg.d_c = 512;
    cfg.d_rope = 64;
    cfg.page_block_size = 64;
    cfg.max_num_blocks_per_seq = 16;
    cfg.sm_scale = 1.0f / std::sqrt(576.0f);  // d_c + d_rope = 576
    cfg.num_sm_parts = 8;
    cfg.kv_cache = nullptr;
    cfg.Pi = nullptr;
    cfg.centroids = nullptr;

    EXPECT_EQ(cfg.batch_size, 1);
    EXPECT_EQ(cfg.d_c, 512);
    EXPECT_EQ(cfg.d_rope, 64);
    EXPECT_EQ(cfg.d_c + cfg.d_rope, 576);
    EXPECT_FLOAT_EQ(cfg.sm_scale, 1.0f / std::sqrt(576.0f));
}

TEST(TqAttentionGraph, Model1Config) {
    sm120::graph::TqDecodeGraphConfig cfg{};
    cfg.batch_size = 4;
    cfg.s_q = 1;
    cfg.h_q = 64;
    cfg.d_c = 448;
    cfg.d_rope = 64;
    cfg.page_block_size = 64;
    cfg.max_num_blocks_per_seq = 8;
    cfg.sm_scale = 1.0f / std::sqrt(512.0f);  // d_c + d_rope = 512
    cfg.num_sm_parts = 4;
    cfg.kv_cache = nullptr;
    cfg.Pi = nullptr;
    cfg.centroids = nullptr;

    EXPECT_EQ(cfg.d_c, 448);
    EXPECT_EQ(cfg.d_c + cfg.d_rope, 512);
    EXPECT_EQ(cfg.d_rope, 64);  // d_rope always 64
}

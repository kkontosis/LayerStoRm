#pragma once
// FP8 → BF16 strided dequantization for kv_b_proj V portion.
//
// Extracts the value-head rows from the full kv_b_proj weight matrix
// (which contains both K_nope and V rows interleaved per head), and
// dequantizes them from FP8 E4M3 with blockwise float32 scales to BF16.
//
// Input layout:  [HL*(qk_nope+v_head_dim), kv_lora_rank] FP8 row-major
//   Per head h, rows [h*(P+V)+P .. h*(P+V)+P+V-1] are the V portion,
//   where P=qk_nope_head_dim, V=v_head_dim.
//
// Output layout: [HL, v_head_dim, kv_lora_rank] BF16 contiguous row-major
//   Head h's V block is output rows [h*V .. (h+1)*V-1].
//
// Scale layout:  [ceil(kv_lora_rank/128), ceil(N_total/128)] float32, K-major
//   where N_total = HL*(P+V).  For V row at source index
//   src_row = h*(P+V) + P + v_row, its N-block is src_row/128.
//   When P==V==128 this simplifies to N-block = 2*h+1 (exact block
//   alignment), but the kernel handles arbitrary P, V generically.
//
// Architecture-agnostic (no SM120-specific ops).

#include <cstddef>

namespace layerstorm::compute {

struct KvBvExtractDequantParams {
    int num_heads_local;      ///< HL: heads per TP rank
    int qk_nope_head_dim;     ///< P: nope head dim (128 for V3.2)
    int v_head_dim;           ///< V: value head dim (128 for V3.2)
    int kv_lora_rank;         ///< D_c: compressed KV dim (512 for V3.2)
    const void* kv_b_proj;    ///< [HL*(P+V), D_c] FP8 E4M3 or packed GGUF row-major
    const void* scales;       ///< [ceil(D_c/128), ceil(HL*(P+V)/128)] float32 (FP8 only)
    void* output;             ///< [HL, V, D_c] BF16 contiguous row-major

    // GG-7b: GGUF W_UV value-side path. -1 = not GGUF (use the FP8/BF16 branch
    // above); otherwise a model::GgufKQuantType value cast to int (0=Q2_K …
    // 5=Q8_0). When set, kv_b_proj is a packed GGUF k-quant — each V row is
    // [(D_c/QK)*block_bytes] packed bytes — and the kernel dequants the V rows
    // per element into BF16 (dequant-only; bit-equal to a load-time dequant, the
    // same philosophy as q_absorb's GGUF W_UK branch). `scales` is ignored (GGUF
    // scales are in-block; the caller passes nullptr). Requires D_c % QK == 0.
    int gguf_type = -1;
};

/// Launch FP8 V-extract dequantization kernel.
///
///   stream: cudaStream_t (as void*), or nullptr for default stream.
///
/// Total output elements: HL * V * D_c.
/// Thread grid: one block per (head, v_row) pair = HL * V blocks.
void launch_kv_bv_extract_dequant(const KvBvExtractDequantParams& params,
                                   void* stream);

}  // namespace layerstorm::compute

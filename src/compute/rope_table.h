// RoPE cos/sin table + softmax-scale computation for DeepSeek MLA (host-side, CUDA-free).
//
// Convention (ref/DeepSeek-V3 inference/model.py precompute_freqs_cis + apply_rotary_emb):
// interleaved adjacent-pair rotation — frequency i of d_rope/2 applies to dims (2i, 2i+1).
// The table holds PURE cos/sin (no mscale): YaRN enters only through frequency correction;
// the YaRN mscale factor multiplies the softmax scale (model.py:434-437), applied exactly once.
//
// Table layout matches the rope_rotate / q_absorb kernels: [max_pos][d_rope] float32,
// per position d_rope/2 cos values then d_rope/2 sin values.

#pragma once

#include "config/config_parser.h"

#include <vector>

namespace layerstorm::compute {

/// Build the [max_pos][d_rope] cos|sin table. Applies the YaRN frequency correction
/// (correction range + linear ramp, DeepSeek reference) when `scaling` is a yarn config
/// with factor + original_max_position_embeddings and max_pos exceeds the original length.
std::vector<float> build_rope_cos_sin_table(
    int max_pos, int d_rope, double rope_theta,
    const std::optional<config::RopeScalingConfig>& scaling);

/// V4-4c dual RoPE tables (DeepSeek V4).
/// Rule (ref/llama.cpp deepseek4.cpp:817-824): UNCOMPRESSED layers
/// (compress_ratios[l] == 0) rotate with the base theta and NO yarn
/// (freq_scale 1, ext_factor 0); COMPRESSED layers rotate with
/// compress_rope_theta (160000) WITH the full yarn frequency correction.
/// Both tables are PURE cos/sin: llama.cpp's dsv4_rope_attn_factor
/// = 1/(1 + 0.1·ln(1/freq_scale)) exactly cancels ggml's internal yarn
/// mscale, so no mscale enters the tables — and none enters the V4 softmax
/// scale either (V4 scale = 1/sqrt(head_dim), NOT rope_softmax_scale below).
struct V4RopeTables {
    std::vector<float> base;      ///< [max_pos][d_rope] — uncompressed layers
    std::vector<float> compress;  ///< [max_pos][d_rope] — compressed layers
};
V4RopeTables build_v4_rope_tables(
    int max_pos, int d_rope, double rope_theta, double compress_rope_theta,
    const std::optional<config::RopeScalingConfig>& scaling);

/// Attention softmax scale: (qk_nope + qk_rope)^-0.5 — the absorbed-space dot product
/// equals the original per-head dot product, so the NON-absorbed head dim sets the scale
/// (DeepSeek model.py:434, vLLM deepseek_v2). Under YaRN with context beyond the original
/// length: scale *= mscale^2 with mscale = 0.1·mscale_all_dim·ln(factor) + 1.
float rope_softmax_scale(
    int qk_nope_head_dim, int qk_rope_head_dim, int max_seq_len,
    const std::optional<config::RopeScalingConfig>& scaling);

}  // namespace layerstorm::compute

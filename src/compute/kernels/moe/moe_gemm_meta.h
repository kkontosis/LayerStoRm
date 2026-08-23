// Populate grouped GEMM metadata (problem_sizes, sf_offsets) from expert_offsets.
// GPU kernel — derives M per expert on-device without host sync.

#pragma once

#include <cstdint>

namespace layerstorm::compute {

// Populate problem_sizes and optionally sf_offsets from device-resident expert_offsets.
//   problem_sizes: [num_experts * 3] INT32 output — {M_e, N, K} per expert
//   sf_offsets:    [num_experts + 1] INT32 output — cumulative 128-row-aligned scale offsets
//                  May be nullptr (skipped for FP8 path).
//   expert_offsets: [num_experts + 1] INT32 input — cumulative token counts (device)
void launch_populate_gemm_meta(
    int32_t* problem_sizes,
    int32_t* sf_offsets,
    const int32_t* expert_offsets,
    int N, int K, int num_experts,
    void* stream /*cudaStream_t*/);

/// TD-MOE-BIG-GEMM-SWEEP (wave-masked permute): mask a top-K expert-index
/// array by a per-expert byte mask.
///   masked_indices[i] = (0 <= topk_indices[i] < num_experts
///                        && expert_mask[topk_indices[i]]) ? topk_indices[i] : -1
/// The -1 sentinel is moe_permute's adaptive-gating drop: masked entries are
/// remapped to num_experts inside the permute, radix-sort PAST
/// expert_offsets[num_experts], and are therefore excluded from EVERY grouped-
/// GEMM expert group (zero mmq tiles, mmvq early-return) — a rolling-wave pass
/// only pays for its own experts' rows instead of the full-batch zero-weight
/// sweep. Out-of-range / already-sentinel inputs stay -1.
void launch_mask_topk_indices(
    int32_t* masked_indices,        ///< [count] output
    const int32_t* topk_indices,    ///< [count] input (num_tokens * topk entries)
    const uint8_t* expert_mask,     ///< [num_experts] 0/1 device byte mask
    int count,
    int num_experts,
    void* stream /*cudaStream_t*/);

/// Gather per-expert NVFP4 alpha (weight_scale_2) from scattered B pointers.
/// Each expert's weight_scale_2 float is at b_ptrs[e] + alpha_offset bytes.
/// For any projection, alpha_offset = projection_total_bytes - 8
/// (weight_scale_2 sits 8 bytes before the end of each projection block).
void launch_gather_alphas(
    float* alphas_out,               ///< [num_experts] output — contiguous device array
    const void* const* b_ptrs,       ///< [num_experts] device — per-expert B base pointers
    int64_t alpha_offset,            ///< byte offset from B pointer to weight_scale_2
    int num_experts,
    void* stream /*cudaStream_t*/);

/// Gather per-expert NVFP4 alpha AND input_scale from scattered B pointers
/// (TRT-LLM activation-scale scheme): reads ws2 at b_ptrs[e] + ws2_offset and
/// is at b_ptrs[e] + is_offset (the slot's last 8 bytes: ws2 at -8, is at -4),
/// guards is <= 0 → 1.0 (missing experts point at a zero buffer), and writes
///   alphas_out[e]       = ws2 * is
///   input_scales_out[e] = is   (when non-null; feeds bf16_to_nvfp4_grouped)
/// The is values are pack-time normalized (gate == up == layer fc31 max), so
/// each projection's own slot value is consistent with the shared activation.
void launch_gather_alphas_scaled(
    float* alphas_out,               ///< [num_experts] output device array
    float* input_scales_out,         ///< [num_experts] output, may be nullptr
    const void* const* b_ptrs,       ///< [num_experts] device B base pointers
    int64_t ws2_offset,              ///< byte offset to weight_scale_2
    int64_t is_offset,               ///< byte offset to input_scale
    int num_experts,
    void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute

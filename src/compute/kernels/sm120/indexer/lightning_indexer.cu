// Single-TU driver for the DSA lightning-indexer kernels (GLM-25a).
//
// Includes the vendored kernel .cu files (which define their __global__ +
// run_* host functions inline) and exposes host-callable launch wrappers, the
// same pattern as snapmla_prep.cu. The deps/LayerStoRmKernels/csrc include root
// is already on layerstorm_cuda_kernels, so "sm120/indexer/..." resolves.

#include <stdexcept>
#include <string>

#include "sm120/indexer/lightning_score.cu"
#include "sm120/indexer/lightning_score_mqa.cu"
#include "sm120/indexer/lightning_topk.cu"
#include "sm120/indexer/topk_merge.cu"

#include "compute/kernels/sm120/indexer/lightning_indexer.h"

namespace layerstorm::compute {

void launch_lightning_score(const sm120::indexer::LightningScoreParams& params,
                            cudaStream_t stream) {
    sm120::indexer::run_lightning_score(params, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_lightning_score failed: ") + cudaGetErrorString(err));
    }
}

void launch_lightning_score_mqa(const sm120::indexer::LightningScoreMqaParams& params,
                                cudaStream_t stream) {
    // Kernel contract: 4 threads/head each cover head_dim/4 dims with 4-byte
    // vectorized FP8 loads (head_dim % 16 == 0), and score_proj smem is fixed
    // at 64 heads. Violations would corrupt silently — reject here.
    if (params.index_head_dim % 16 != 0 || params.index_head_dim <= 0) {
        throw std::runtime_error(
            "launch_lightning_score_mqa: index_head_dim must be a positive "
            "multiple of 16, got " + std::to_string(params.index_head_dim));
    }
    if (params.index_n_heads <= 0 || params.index_n_heads > 64) {
        throw std::runtime_error(
            "launch_lightning_score_mqa: index_n_heads must be in 1..64, got "
            + std::to_string(params.index_n_heads));
    }
    sm120::indexer::run_lightning_score_mqa(params, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_lightning_score_mqa failed: ") + cudaGetErrorString(err));
    }
}

void launch_lightning_topk(const sm120::indexer::LightningTopkParams& params,
                           cudaStream_t stream) {
    sm120::indexer::run_lightning_topk(params, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_lightning_topk failed: ") + cudaGetErrorString(err));
    }
}

// TD-SPARSE-PREFILL-SCORE-BATCH: batched multi-row variants — same contracts
// as the single-query wrappers above, one launch over all rows.
void launch_lightning_score_mqa_batched(
    const sm120::indexer::LightningScoreMqaBatchedParams& params,
    cudaStream_t stream) {
    // Same kernel contract as launch_lightning_score_mqa (the per-block body
    // is shared): head_dim multiple of 16, heads in 1..64.
    if (params.index_head_dim % 16 != 0 || params.index_head_dim <= 0) {
        throw std::runtime_error(
            "launch_lightning_score_mqa_batched: index_head_dim must be a "
            "positive multiple of 16, got "
            + std::to_string(params.index_head_dim));
    }
    if (params.index_n_heads <= 0 || params.index_n_heads > 64) {
        throw std::runtime_error(
            "launch_lightning_score_mqa_batched: index_n_heads must be in "
            "1..64, got " + std::to_string(params.index_n_heads));
    }
    if (params.k_page_table && params.page_tokens <= 0) {
        throw std::runtime_error(
            "launch_lightning_score_mqa_batched: paged mode requires "
            "page_tokens > 0, got " + std::to_string(params.page_tokens));
    }
    if (params.scores_stride < params.max_num_blocks) {
        throw std::runtime_error(
            "launch_lightning_score_mqa_batched: scores_stride "
            + std::to_string(params.scores_stride) + " < max_num_blocks "
            + std::to_string(params.max_num_blocks));
    }
    sm120::indexer::run_lightning_score_mqa_batched(params, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_lightning_score_mqa_batched failed: ")
            + cudaGetErrorString(err));
    }
}

void launch_lightning_topk_batched(
    const sm120::indexer::LightningTopkBatchedParams& params,
    cudaStream_t stream) {
    sm120::indexer::run_lightning_topk_batched(params, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_lightning_topk_batched failed: ")
            + cudaGetErrorString(err));
    }
}

void launch_lightning_topk_merge(
    const sm120::indexer::LightningTopkMergeParams& params,
    cudaStream_t stream) {
    sm120::indexer::run_lightning_topk_merge(params, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_lightning_topk_merge failed: ")
            + cudaGetErrorString(err));
    }
}

}  // namespace layerstorm::compute

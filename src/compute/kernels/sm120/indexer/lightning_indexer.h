#pragma once

// Host-callable launch wrappers for the DSA lightning-indexer producer kernels
// vendored in deps/LayerStoRmKernels (csrc/sm120/indexer/lightning_{score,topk}.cu).
// These compute the sparse-attention block selection:
//   scores[n] = Σ_h ReLU(q_proj[h]·K[n,h]) · score_proj[h]      (lightning_score)
//   top-k of scores with causality → sorted block indices        (lightning_topk)
//
// GLM-25a: wired into produce_sparse_indices() (src/compute/indexer_produce.cpp).

#include <cuda_runtime.h>

#include "sm120/indexer/lightning_score.h"
#include "sm120/indexer/lightning_score_mqa.h"
#include "sm120/indexer/lightning_topk.h"
#include "sm120/indexer/topk_merge.h"

namespace layerstorm::compute {

/// Score all resident indexer-K blocks for one query (per-head K layout). Throws
/// on launch error. Retained for the V3.2/V4 per-head-K reference path.
void launch_lightning_score(const sm120::indexer::LightningScoreParams& params,
                            cudaStream_t stream);

/// Score all resident indexer-K blocks for one query — MQA variant with a single
/// shared K per block ([num_blocks, index_head_dim]) broadcast over query heads.
/// Used by DeepSeek-V3.2 / GLM-5.2 (wk → single head). Throws on launch error.
void launch_lightning_score_mqa(const sm120::indexer::LightningScoreMqaParams& params,
                                cudaStream_t stream);

/// TD-SPARSE-PREFILL-SCORE-BATCH: MQA scoring of num_rows queries in ONE
/// launch, each row bounded by its own row_num_blocks[r] (contiguous K or a
/// per-row device page table). Bit-identical to per-row
/// launch_lightning_score_mqa calls. Throws on launch error.
void launch_lightning_score_mqa_batched(
    const sm120::indexer::LightningScoreMqaBatchedParams& params,
    cudaStream_t stream);

/// Select the causal top-k blocks for one query. Throws on launch error.
void launch_lightning_topk(const sm120::indexer::LightningTopkParams& params,
                           cudaStream_t stream);

/// TD-SPARSE-PREFILL-SCORE-BATCH: causal top-k of num_rows rows in ONE
/// launch (one CTA per row, per-row bound + cutoff arrays). Bit-identical to
/// per-row launch_lightning_topk calls. Throws on launch error.
void launch_lightning_topk_batched(
    const sm120::indexer::LightningTopkBatchedParams& params,
    cudaStream_t stream);

/// TD-GLM-INDEXER-LOCAL-MERGE: exact cross-rank merge of per-rank shard-local
/// top-k candidate lists (dcp_indexer_mode=local) — scatter candidates back to
/// global positions over a -inf-filled full-length scratch, then re-run the
/// standard radix top-k. Throws on launch error.
void launch_lightning_topk_merge(
    const sm120::indexer::LightningTopkMergeParams& params,
    cudaStream_t stream);

}  // namespace layerstorm::compute

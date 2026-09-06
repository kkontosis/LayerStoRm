#pragma once

// GF3.5: IndexPool kpool-compression launch wrappers (host-callable).
// Kernels live in deps/LayerStoRmKernels/csrc/sm120/indexer/kpool_compress.cu
// (see its header for the layout + numerics contract and vLLM/SGLang
// attribution); this wrapper TU includes them once, the same single-TU
// pattern as lightning_indexer.cu. The scoring / top-k / merge-reselect
// kernels are the UNCHANGED lightning kernels — pooled callers pass
// entry-domain counts (page_entries, pooled endpoints, pool budget).

#include <cuda_runtime.h>

#include "sm120/indexer/kpool_compress.h"

namespace layerstorm::compute {

void launch_kpool_append_decode(const sm120::indexer::KpoolAppendDecodeParams& p,
                                cudaStream_t stream);
void launch_kpool_chunk_append(const sm120::indexer::KpoolChunkAppendParams& p,
                               cudaStream_t stream);
void launch_kpool_expand(const sm120::indexer::KpoolExpandParams& p,
                         cudaStream_t stream);
void launch_kpool_cand_scatter(const sm120::indexer::KpoolCandScatterParams& p,
                               cudaStream_t stream);

}  // namespace layerstorm::compute

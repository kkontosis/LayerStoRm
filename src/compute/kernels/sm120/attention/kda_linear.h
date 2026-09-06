#pragma once

// GF3.7: KDA (Kimi Delta Attention) SM120 launch wrappers (host-callable).
// Kernels live in deps/LayerStoRmKernels/csrc/sm120/linear/ (see kda_scan.h
// for the API, layout, precision contract and attribution); this wrapper TU
// includes them once — the indexer_kpool.cu / lightning_indexer.cu single-TU
// pattern. Wrappers validate geometry and THROW std::runtime_error on
// misuse (head_dim is compile-fixed at 128; t_len/batch/workspace checks) and
// on launch errors. Wiring into an AttentionDevice / ArchGlm5Next execution
// hook is deliberately NOT here — that is GF3.9's join (the arch hooks stay
// loud kComputeValidation stubs until then).

#include <cuda_runtime.h>

#include "sm120/linear/kda_scan.h"

namespace layerstorm::compute {

/// fp32 workspace bytes launch_kda_chunked_scan requires.
size_t kda_prefill_workspace_bytes(int t_len, int num_heads);

void launch_kda_conv_prefill(const sm120::linear::KdaConvPrefillParams& p,
                             cudaStream_t stream);
void launch_kda_conv_decode(const sm120::linear::KdaConvDecodeParams& p,
                            cudaStream_t stream);
void launch_kda_chunked_scan(const sm120::linear::KdaChunkedScanParams& p,
                             cudaStream_t stream);
void launch_kda_gated_rmsnorm(const sm120::linear::KdaGatedRmsNormParams& p,
                              cudaStream_t stream);
void launch_kda_decode_step(const sm120::linear::KdaDecodeStepParams& p,
                            cudaStream_t stream);

}  // namespace layerstorm::compute

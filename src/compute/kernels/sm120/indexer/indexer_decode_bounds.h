#pragma once

// P-29 step 7 (INV-0.6(b) indexer span): derive the B=1 decode indexer
// scoring bounds ON DEVICE from the live seqlens_k entry, so the batched
// score/top-k launches are CUDA-graph-capturable (the bounds change every
// token; reading them at execution time keeps one capture valid across
// tokens). Writes exactly the two integers the host loop computes:
//   out2[0] = entries = seqlen[0] / kpool   (kpool > 1: settled pools)
//                     = seqlen[0]           (kpool == 1: token entries)
//   out2[1] = seqlen[0] - 1                 (causal cutoff, token position)
//
// CUDA-free header (INV-GPU-1): callable from non-CUDA TUs with the current
// device set to the stream's device.

namespace layerstorm::compute {

void launch_indexer_decode_bounds(const void* seqlen /* device const int* */,
                                  int kpool,
                                  void* out2 /* device int[2] */,
                                  void* stream /* cudaStream_t */);

}  // namespace layerstorm::compute

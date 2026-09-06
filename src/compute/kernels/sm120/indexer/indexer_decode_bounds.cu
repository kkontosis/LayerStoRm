// P-29 step 7 (INV-0.6(b) indexer span): device-side B=1 decode bounds.
// See indexer_decode_bounds.h for the contract.

#include "indexer_decode_bounds.h"

#include <cuda_runtime.h>

namespace layerstorm::compute {

namespace {

__global__ void indexer_decode_bounds_kernel(const int* __restrict__ seqlen,
                                             int kpool,
                                             int* __restrict__ out2) {
    const int len = seqlen[0];
    out2[0] = kpool > 1 ? len / kpool : len;
    out2[1] = len - 1;
}

}  // namespace

void launch_indexer_decode_bounds(const void* seqlen, int kpool, void* out2,
                                  void* stream) {
    indexer_decode_bounds_kernel<<<1, 1, 0,
                                   static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int*>(seqlen), kpool, static_cast<int*>(out2));
}

}  // namespace layerstorm::compute

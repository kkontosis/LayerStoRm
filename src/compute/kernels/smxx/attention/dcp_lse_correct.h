#pragma once

//==============================================================================
// DCP LSE Correction — namespace-forwarding header
//
// Re-exports the DCP LSE correction kernel from deps/LayerStoRmKernels
// into the layerstorm::compute namespace.
//==============================================================================

#include "smxx/dcp_lse_correct.h"

namespace layerstorm::compute {

// Corrects attention outputs when KV cache is sequence-sharded across
// DCP ranks. Each GPU produces partial output + LSE; after allgather of
// LSEs, this kernel reweights by exp(local_lse - global_lse).
inline void launch_dcp_lse_correct(const DcpLseCorrectParams& params,
                                   cudaStream_t stream) {
    run_dcp_lse_correct_kernel(params, stream);
}

}  // namespace layerstorm::compute

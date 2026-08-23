// Single-TU driver for TQ prep + graph kernels.
//
// Includes the TQ prep kernel .cu files (which define __global__ functions
// inline) and the TQ decode graph runner, avoiding duplicate symbol issues.
// Provides host-callable launch wrappers for the prep kernels.
//
// Mirrors snapmla_prep.cu for TurboQuant MLA.

#include "sm120/prep/tq_fused_k_append.cu"
#include "sm120/prep/tq_dequant_ckv_indexed.cu"
#include "sm120/prep/tq_q_rotate.cu"
#include "sm120/prep/tq_v_rotate_back.cu"
#include "sm120/graph/tq_decode_graph.cu"

namespace layerstorm::compute {

void launch_tq_k_append(const sm120::prep::TqFusedKAppendParams& params,
                         cudaStream_t stream) {
    sm120::prep::run_tq_fused_k_append(params, stream);
}

void launch_tq_dequant_ckv_indexed(const sm120::prep::TqDequantCKVIndexedParams& params,
                                    cudaStream_t stream) {
    sm120::prep::run_tq_dequant_ckv_indexed(params, stream);
}

void launch_tq_q_rotate(const sm120::prep::TqQRotateParams& params,
                         cudaStream_t stream) {
    sm120::prep::run_tq_q_rotate(params, stream);
}

void launch_tq_v_rotate_back(const sm120::prep::TqVRotateBackParams& params,
                              cudaStream_t stream) {
    sm120::prep::run_tq_v_rotate_back(params, stream);
}

}  // namespace layerstorm::compute

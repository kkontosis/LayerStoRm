// Single-TU driver for SnapMLA prep + graph kernels.
//
// Includes the prep kernel .cu files (which define __global__ functions inline)
// and the decode graph runner, avoiding duplicate symbol issues.
// Provides host-callable launch wrappers for the prep kernels.

#include "sm120/prep/fused_q_quant.cu"
#include "sm120/prep/q_absorb.cu"
#include "sm120/prep/rope_rotate.cu"
#include "sm120/prep/fused_k_append.cu"
#include "sm120/prep/dequant_ckv_indexed.cu"
#include "sm120/graph/decode_graph.cu"

namespace layerstorm::compute {

void launch_fused_q_quant(const sm120::prep::FusedQQuantParams& params,
                          cudaStream_t stream) {
    int grid = params.s_q * params.h_q;
    sm120::prep::fused_q_quant_kernel<<<grid, 256, 0, stream>>>(params);
}

void launch_q_absorb(const sm120::prep::QAbsorbParams& params,
                     cudaStream_t stream) {
    sm120::prep::run_q_absorb(params, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_q_absorb failed: ") + cudaGetErrorString(err));
    }
}

void launch_rope_rotate(const sm120::prep::RopeRotateParams& params,
                        cudaStream_t stream) {
    sm120::prep::run_rope_rotate(params, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_rope_rotate failed: ") + cudaGetErrorString(err));
    }
}

void launch_fused_k_append(const sm120::prep::FusedKAppendParams& params,
                           cudaStream_t stream) {
    if (!params.kv_cache || !params.slot_mapping) {
        throw std::runtime_error("launch_fused_k_append: null kv_cache or slot_mapping");
    }
    sm120::prep::fused_k_append_kernel<<<params.num_tokens, 256, 0, stream>>>(params);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_fused_k_append failed: ") + cudaGetErrorString(err));
    }
}

void launch_dequant_ckv_indexed(const sm120::prep::DequantCKVIndexedParams& params,
                                cudaStream_t stream) {
    // KVS-3 (INV-KVS-EMPTY): num_fetch == 0 is a legal empty DCP shard —
    // grid.x = 0 would be an invalid launch config. Nothing to dequant.
    if (params.num_fetch <= 0) return;
    sm120::prep::dequant_ckv_fused_indexed_kernel<<<params.num_fetch, 128, 0, stream>>>(params);
}

}  // namespace layerstorm::compute

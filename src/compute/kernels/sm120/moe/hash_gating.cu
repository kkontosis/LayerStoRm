// Hash-layer MoE gating kernel (DeepSeek V4) — see hash_gating.h for the
// reference semantics (llama.cpp deepseek4.cpp hash routing + build_moe_ffn).
//
// Scoring functions copied verbatim from deps/LayerStoRmExpertKernels
// csrc/sm120/gating/topk_gating.cu (stable_sigmoid / stable_sqrtsoftplus) so
// hash-layer weights are bit-identical to what the learned-gating kernel
// computes for the same logits.

#include "compute/kernels/moe/hash_gating.h"

#include <cuda_runtime.h>

namespace {

using layerstorm::compute::ScoringFunc;

/// Numerically stable sigmoid via tanh identity (== deps topk_gating.cu).
__device__ __forceinline__ float stable_sigmoid(float x) {
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

/// sqrt(softplus(x)); x > 20 guard avoids exp overflow (== deps topk_gating.cu).
__device__ __forceinline__ float stable_sqrtsoftplus(float x) {
    float sp = (x > 20.0f) ? x + logf(1.0f + expf(-x))
                           : logf(1.0f + expf(x));
    return sqrtf(sp);
}

__device__ __forceinline__ float apply_scoring(float logit, int scoring_func) {
    if (scoring_func == static_cast<int>(ScoringFunc::kSigmoid))
        return stable_sigmoid(logit);
    return stable_sqrtsoftplus(logit);
}

// One warp (block) per token. Lane k < topk owns routing slot k:
//   id_k = tid2eid[token_id * topk + k]   (table order = selection rank)
//   w_k  = scoring_fn(logits[token, id_k])
// then a warp reduction renormalizes to routed_scaling_factor (matching
// warp_topk_select in deps topk_gating.cu: no renorm ⇒ raw unbiased scores).
// Out-of-range token id or table entry ⇒ id -1 (permute drop sentinel), w 0.
__global__ void hash_gating_kernel(
    float* __restrict__ topk_weights, int32_t* __restrict__ topk_indices,
    const float* __restrict__ logits, const int32_t* __restrict__ tid2eid,
    const int32_t* __restrict__ token_ids,
    int num_experts, int topk, int vocab_size,
    float routed_scaling_factor, bool renormalize, int scoring_func) {

    const int token = blockIdx.x;
    const int lane = threadIdx.x;

    const int32_t tid = token_ids[token];
    const bool tid_ok = (tid >= 0 && tid < vocab_size);

    float w = 0.f;
    int32_t expert = -1;
    if (lane < topk && tid_ok) {
        const int32_t e =
            tid2eid[static_cast<int64_t>(tid) * topk + lane];
        if (e >= 0 && e < num_experts) {
            expert = e;
            w = apply_scoring(
                logits[static_cast<int64_t>(token) * num_experts + e],
                scoring_func);
        }
    }

    if (renormalize) {
        // Warp-sum of the topk weights (lanes >= topk contribute 0).
        float sum = w;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (sum > 0.f)
            w *= routed_scaling_factor / sum;
    }

    if (lane < topk) {
        topk_weights[static_cast<int64_t>(token) * topk + lane] = w;
        topk_indices[static_cast<int64_t>(token) * topk + lane] = expert;
    }
}

}  // anonymous namespace

namespace layerstorm::compute {

void launch_hash_gating(float* topk_weights, int32_t* topk_indices,
                        const float* logits, const int32_t* tid2eid,
                        const int32_t* token_ids,
                        const HashGatingParams& params,
                        void* stream) {
    if (params.num_tokens <= 0 || params.topk <= 0) return;
    // One warp per token; topk must fit in a warp (V4: 6).
    if (params.topk > 32) return;  // unreachable by config validation

    hash_gating_kernel<<<params.num_tokens, 32, 0,
                         static_cast<cudaStream_t>(stream)>>>(
        topk_weights, topk_indices, logits, tid2eid, token_ids,
        params.num_experts, params.topk, params.vocab_size,
        params.routed_scaling_factor, params.renormalize,
        static_cast<int>(params.scoring_func));
}

}  // namespace layerstorm::compute

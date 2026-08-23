// DeepSeek-V4 TQ codec prep kernels (V4-5T, TD-V4-TQ-DEVICE) — SM120.
//
// Single-TU driver for the deps V4 TQ append kernel (the tq_prep.cu /
// snapmla_prep.cu pattern): the deps .cu is included verbatim and the
// engine-facing wrapper below adapts compute::V4TqAppendArgs (v4_prep.h)
// to the deps params. The csa_tq DECODE kernel is NOT included here — it
// is compiled in the tq_mla_kernels OBJECT lib (splitkv_csa_tq.cu) and the
// V4 device calls run_csa_tq_decode directly.

#include "compute/kernels/attention/v4_prep.h"

#include "sm120/prep/v4_tq_k_append.cu"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace layerstorm::compute {

void launch_v4_tq_entry_append(const V4TqAppendArgs& args, void* stream) {
    if (args.num_tokens <= 0) return;
    if (!args.rows || !args.rope_rows || !args.kv_cache || !args.slots ||
        !args.Pi || !args.centroids || !args.boundaries) {
        throw std::runtime_error(
            "v4_tq_entry_append: null argument (rows/rope/cache/slots/"
            "Pi/centroids/boundaries all required)");
    }
    sm120::prep::V4TqKAppendParams p{};
    p.k_nope = static_cast<const __nv_bfloat16*>(args.rows);
    p.k_rope = static_cast<const __nv_bfloat16*>(args.rope_rows);
    // V == K: the compressed entry duplicates the pooled vector into both
    // halves, mirroring the FP8 write (v4_write_fp8_entry).
    p.v_nope = static_cast<const __nv_bfloat16*>(args.rows);
    p.kv_cache = static_cast<uint8_t*>(args.kv_cache);
    p.slot_mapping = args.slots;
    p.Pi = args.Pi;
    p.centroids = args.centroids;
    p.decision_boundaries = args.boundaries;
    p.num_tokens = args.num_tokens;
    p.head_dim = args.head_dim;
    p.qk_rope_head_dim = args.rope_dim;
    p.num_centroids = args.num_centroids;
    sm120::prep::run_v4_tq_k_append(p, static_cast<cudaStream_t>(stream));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("v4_tq_entry_append launch failed: ") +
            cudaGetErrorString(err));
    }
}

}  // namespace layerstorm::compute

// MLA Attention dispatch wrappers — routes to SnapMLA kernel instantiations.

#include "compute/kernels/attention/mla_attention.h"

// Decode param + launch declarations
#include "sm120/decode/dense_fp8/params.h"
#include "sm120/decode/sparse_fp8/params.h"

// Prefill param + launch declarations
#include "sm120/prefill/dense/fwd/head64/phase1.h"
#include "sm120/prefill/sparse/fwd/head64/phase1.h"

// SMXX
#include "smxx/mla_combine.h"
#include "smxx/get_mla_metadata.h"

namespace layerstorm::compute {

using sm120::sparse::ModelType;

// ── Decode Dense FP8 ────────────────────────────────────────────────────────

void launch_decode_dense_fp8(
    const sm120::decode::dense_fp8::DenseAttnDecodeParams& params) {
    namespace ns = sm120::decode::dense_fp8;
    // DET-REDUCE: select the deterministic (bit-reproducible) instantiation when
    // requested; default false path is byte-identical to the legacy atomic path.
    if (params.deterministic_reduce) {
        if (params.d_nope == 512)
            ns::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::V32, 64, true>(params);
        else
            ns::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::MODEL1, 64, true>(params);
    } else {
        if (params.d_nope == 512)
            ns::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::V32, 64, false>(params);
        else
            ns::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::MODEL1, 64, false>(params);
    }
}

// ── Decode Sparse FP8 ───────────────────────────────────────────────────────

void launch_decode_sparse_fp8(
    const sm120::decode::sparse_fp8::SparseAttnDecodeParams& params) {
    namespace ns = sm120::decode::sparse_fp8;
    if (params.model_type == ModelType::V32)
        ns::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, 64>(params);
    else
        ns::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 64>(params);
}

// ── Prefill Dense ───────────────────────────────────────────────────────────

void launch_prefill_dense(
    const sm120::prefill::dense::head64::DenseAttnFwdParams& params) {
    using namespace sm120::prefill::dense::head64;
    if (!params.q) {
        throw std::runtime_error("launch_prefill_dense: null Q pointer");
    }
    if (!params.kv) {
        throw std::runtime_error(
            "launch_prefill_dense: null KV pointer — KV cache not wired into "
            "DenseAttnFwdParams (s_q=" + std::to_string(params.s_q) +
            " s_kv=" + std::to_string(params.s_kv) + ")");
    }
    // Only V3.2 (D_QK=576) has a dense prefill instantiation.
    // MODEL1 uses sparse prefill exclusively.
    // DET-REDUCE: pick the deterministic instantiation when requested.
    if (params.deterministic_reduce)
        run_dense_fwd_phase1_kernel<576, true>(params);
    else
        run_dense_fwd_phase1_kernel<576, false>(params);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_prefill_dense failed: ") + cudaGetErrorString(err));
    }
}

// ── Prefill Sparse ──────────────────────────────────────────────────────────

void launch_prefill_sparse(const SparseAttnFwdParams& params) {
    using namespace sm120::prefill::sparse::head64;
    // DET-REDUCE (TD-SPARSE-PREFILL-DETREDUCE): pick the deterministic
    // (bit-reproducible fixed-order denominator) instantiation when requested;
    // default false path is byte-identical to the legacy atomic path.
    if (params.deterministic_reduce) {
        if (params.d_qk == 576)
            run_fwd_phase1_kernel<576, true>(params);
        else
            run_fwd_phase1_kernel<512, true>(params);
    } else {
        if (params.d_qk == 576)
            run_fwd_phase1_kernel<576, false>(params);
        else
            run_fwd_phase1_kernel<512, false>(params);
    }
}

// ── Split-KV Combine ────────────────────────────────────────────────────────

void launch_mla_combine(MlaCombineParams& params, cudaStream_t stream) {
    run_mla_combine_kernel<cutlass::bfloat16_t>(params, stream);
}

// ── Split-KV Metadata ───────────────────────────────────────────────────────

void launch_get_mla_metadata(GetMlaMetadataParams& params, cudaStream_t stream) {
    run_get_mla_metadata_kernel(params, stream);
}

}  // namespace layerstorm::compute

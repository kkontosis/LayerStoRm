// MLA Attention dispatch wrappers — routes to SnapMLA kernel instantiations.

#include "compute/kernels/attention/mla_attention.h"

#include <stdexcept>
#include <string>

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
    // GEOMETRY KEY: d_nope alone is AMBIGUOUS — V32 (576 = 512+64) and GLM5N
    // (512 = 512+0) both have d_nope == 512. Key on the total QK dim too.
    ModelType model_type;
    if (params.d_qk == 576 && params.d_nope == 512)
        model_type = ModelType::V32;
    else if (params.d_qk == 512 && params.d_nope == 512)
        model_type = ModelType::GLM5N;
    else if (params.d_qk == 512 && params.d_nope == 448)
        model_type = ModelType::MODEL1;
    else
        throw std::runtime_error(
            "launch_decode_dense_fp8: no dense FP8 decode instantiation for "
            "d_qk=" + std::to_string(params.d_qk) +
            " d_nope=" + std::to_string(params.d_nope) +
            " d_v=" + std::to_string(params.d_v) +
            " (have V32 576/512, GLM5N 512/512, MODEL1 512/448)");
    // DET-REDUCE: select the deterministic (bit-reproducible) instantiation when
    // requested; default false path is byte-identical to the legacy atomic path.
    if (params.deterministic_reduce) {
        if (model_type == ModelType::V32)
            ns::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::V32, 64, true>(params);
        else if (model_type == ModelType::GLM5N)
            ns::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::GLM5N, 64, true>(params);
        else
            ns::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::MODEL1, 64, true>(params);
    } else {
        if (model_type == ModelType::V32)
            ns::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::V32, 64, false>(params);
        else if (model_type == ModelType::GLM5N)
            ns::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::GLM5N, 64, false>(params);
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
    else if (params.model_type == ModelType::GLM5N)
        ns::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::GLM5N, 64>(params);
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
    // GEOMETRY: <576> is the V3.2/GLM-5.2 absorbed geometry (512 nope + 64
    // rope); <512> is the GLM5N NoPE geometry, where the staged KV row IS the
    // 512-wide latent. Traits<512> contracts QK over columns [0, 512) and reads
    // V over [0, D_V = 512) of the SAME rows, which is correct only when
    // d_v == 512. A MODEL1-style d_qk=512 / d_v=448 caller would silently read
    // 512 V columns out of a 448-wide V (the 64 rope dims plus 0 padding) and
    // corrupt the output, so refuse it loudly instead of dispatching.
    if (!(params.d_qk == 576 || (params.d_qk == 512 && params.d_v == 512))) {
        throw std::runtime_error(
            "launch_prefill_dense: no dense prefill instantiation for d_qk=" +
            std::to_string(params.d_qk) + " d_v=" + std::to_string(params.d_v) +
            " (have <576> for V3.2/GLM-5.2 and <512> for the GLM5N NoPE "
            "geometry, which REQUIRES d_v == 512; a d_qk=512/d_v=448 "
            "MODEL1-style row would be read as a 512-wide V and silently "
            "corrupt the output)");
    }
    // DET-REDUCE: pick the deterministic instantiation when requested.
    if (params.deterministic_reduce) {
        if (params.d_qk == 576)
            run_dense_fwd_phase1_kernel<576, true>(params);
        else
            run_dense_fwd_phase1_kernel<512, true>(params);
    } else {
        if (params.d_qk == 576)
            run_dense_fwd_phase1_kernel<576, false>(params);
        else
            run_dense_fwd_phase1_kernel<512, false>(params);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_prefill_dense failed: ") + cudaGetErrorString(err));
    }
}

// ── Prefill Sparse ──────────────────────────────────────────────────────────

void launch_prefill_sparse(const SparseAttnFwdParams& params) {
    using namespace sm120::prefill::sparse::head64;
    // GEOMETRY: <576> is the V3.2/GLM-5.2 absorbed geometry; <512> is the GLM5N
    // NoPE geometry. Traits<512> reads V over columns [0, D_V = 512) of the same
    // staged KV rows it contracts QK over, so it is correct only when the row IS
    // the 512-wide latent (d_v == 512). Refuse a MODEL1-style 448 nope + 64 rope
    // row (d_qk 512, d_v 448) loudly — it would be silently wrong.
    if (!(params.d_qk == 576 || (params.d_qk == 512 && params.d_v == 512))) {
        throw std::runtime_error(
            "launch_prefill_sparse: no sparse prefill instantiation for d_qk=" +
            std::to_string(params.d_qk) + " d_v=" + std::to_string(params.d_v) +
            " (have <576> for V3.2/GLM-5.2 and <512> for the GLM5N NoPE "
            "geometry, which REQUIRES d_v == 512)");
    }
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

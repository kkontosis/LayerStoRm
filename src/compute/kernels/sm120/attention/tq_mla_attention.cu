// TQ MLA Attention dispatch wrappers — routes to TurboQuant kernel functions.
//
// Decode: calls submodule run_tq_dense_decode / run_tq_sparse_decode
//         (resolved at link time from tq_mla_kernels OBJECT library).
// Prefill: composite — dequant TQ cache to BF16, then shared absorbed prefill.
// Combine/metadata: shared kernels re-exported for TQ self-containment.

#include "compute/kernels/attention/tq_mla_attention.h"

// TQ decode param + run_ declarations
#include "sm120/decode/tq_dense/params.h"
#include "sm120/decode/tq_sparse/params.h"

// Shared prefill — same BF16 kernels used by SnapMLA
#include "sm120/prefill/dense/fwd/head64/phase1.h"
#include "sm120/prefill/sparse/fwd/head64/phase1.h"

// TQ prep param header (for dequant in prefill composition)
#include "sm120/prep/tq_dequant_ckv_indexed.h"

// SMXX shared (arch-generic)
#include "smxx/mla_combine.h"
#include "smxx/get_mla_metadata.h"

namespace layerstorm::compute {

// ── TQ Decode Dense ────────────────────────────────────────────────────────

void launch_decode_dense_tq(
    const sm120::decode::tq_dense::TqDenseDecodeParams& params) {
    sm120::decode::tq_dense::run_tq_dense_decode(params);
}

// ── TQ Decode Sparse ───────────────────────────────────────────────────────

void launch_decode_sparse_tq(
    const sm120::decode::tq_sparse::TqSparseDecodeParams& params) {
    sm120::decode::tq_sparse::run_tq_sparse_decode(params);
}

// ── TQ sparse-decode index translation (TD-TQ-SPARSE-DECODE-UNWIRED) ───────
// Compose the DSA top-k selection (positions in the linearized sequence
// prefix) with the block-table linearization (position → pool slot index),
// producing the pool-slot index array the TQ sparse decode kernel consumes.
// Semantics mirror the phase1 sparse kernel's validity rule exactly:
// valid ⇔ i < topk_length[0] ∧ idx ≥ 0 ∧ idx < seq_len_kv; else -1.
namespace {
__global__ void tq_sparse_translate_indices_kernel(
    const int* __restrict__ sparse_indices,
    const int* __restrict__ lin_slots,
    const int* __restrict__ topk_length,
    int topk, int seq_len_kv, const int* __restrict__ seq_len_dev,
    int* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= topk) return;
    const int limit = topk_length ? min(__ldg(topk_length), topk) : topk;
    // Causal bound read at execution time when seq_len_dev is set — keeps the
    // launch graph-replayable across tokens (host seq_len_kv would be baked
    // into the captured node).
    const int bound = seq_len_dev ? __ldg(seq_len_dev) : seq_len_kv;
    int t = (i < limit) ? __ldg(sparse_indices + i) : -1;
    out[i] = (t >= 0 && t < bound) ? __ldg(lin_slots + t) : -1;
}
}  // namespace

void launch_tq_sparse_translate_indices(
    const int* sparse_indices, const int* lin_slots, const int* topk_length,
    int topk, int seq_len_kv, const int* seq_len_dev, int* out,
    cudaStream_t stream) {
    const int threads = 256;
    tq_sparse_translate_indices_kernel<<<(topk + threads - 1) / threads,
                                         threads, 0, stream>>>(
        sparse_indices, lin_slots, topk_length, topk, seq_len_kv, seq_len_dev,
        out);
}

// ── TQ Prefill Dense (composite) ───────────────────────────────────────────

void launch_prefill_dense_tq(
    const sm120::prep::TqDequantCKVIndexedParams& dequant_params,
    const sm120::prefill::dense::head64::DenseAttnFwdParams& prefill_params,
    cudaStream_t stream) {
    // Step 1: Dequant TQ cache → BF16. num_fetch == 0 is a legal empty DCP
    // shard (KVS-3, INV-KVS-EMPTY) — skip the zero-grid launch; the prefill
    // kernel below handles s_kv == 0 (zero output, lse = +inf).
    if (dequant_params.num_fetch > 0)
        sm120::prep::run_tq_dequant_ckv_indexed(dequant_params, stream);
    // Step 2: Standard BF16 absorbed dense prefill
    // DET-REDUCE: pick the deterministic instantiation when requested.
    using namespace sm120::prefill::dense::head64;
    if (prefill_params.deterministic_reduce)
        run_dense_fwd_phase1_kernel<576, true>(prefill_params);
    else
        run_dense_fwd_phase1_kernel<576, false>(prefill_params);
}

// ── TQ Prefill Sparse (composite) ──────────────────────────────────────────

void launch_prefill_sparse_tq(
    const sm120::prep::TqDequantCKVIndexedParams& dequant_params,
    const SparseAttnFwdParams& prefill_params,
    cudaStream_t stream) {
    // Step 1: Dequant TQ cache → BF16 (empty-shard guard: see dense twin).
    if (dequant_params.num_fetch > 0)
        sm120::prep::run_tq_dequant_ckv_indexed(dequant_params, stream);
    // Step 2: Standard BF16 absorbed sparse prefill
    // DET-REDUCE (TD-SPARSE-PREFILL-DETREDUCE): pick the deterministic
    // instantiation when requested (same runtime gate as the dense twin).
    using namespace sm120::prefill::sparse::head64;
    if (prefill_params.deterministic_reduce) {
        if (prefill_params.d_qk == 576)
            run_fwd_phase1_kernel<576, true>(prefill_params);
        else
            run_fwd_phase1_kernel<512, true>(prefill_params);
    } else {
        if (prefill_params.d_qk == 576)
            run_fwd_phase1_kernel<576, false>(prefill_params);
        else
            run_fwd_phase1_kernel<512, false>(prefill_params);
    }
}

// ── Split-KV Combine (FP32 for TQ rotated space) ──────────────────────────

void launch_mla_combine_f32(MlaCombineParams& params, cudaStream_t stream) {
    run_mla_combine_kernel<float>(params, stream);
}

// ── Split-KV Metadata (shared, re-exported) ────────────────────────────────

void launch_get_mla_metadata_tq(GetMlaMetadataParams& params, cudaStream_t stream) {
    run_get_mla_metadata_kernel(params, stream);
}

}  // namespace layerstorm::compute

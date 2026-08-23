// Embedding lookup and output head (LM head) CUDA kernels for LayerStoRm.
// Adapted from TRT-LLM best practice for SM120 (GeForce Blackwell).
//
// Embedding lookup: custom gather kernel with vectorized memory access.
// Output head: self-contained BF16/FP16/FP32 GEMM (bf16_gemm.cu) with FP32
// accumulation — replaces the former cuBLAS GEMM (Item 5: drop libcublas).

#include "compute/kernels/embedding/embedding.h"
#include "compute/kernels/sm120/gemm/bf16_gemm.h"
#include "compute/kernels/smxx/norm/vec_types.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

namespace layerstorm::compute {

// ── Embedding lookup kernel ─────────────────────────────────────────────────
// One block per token. Threads cooperate to copy one row via vectorized I/O.

template <typename scalar_t, int VEC_SIZE>
__global__ void embedding_lookup_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ table,
    const int32_t* __restrict__ token_ids, int num_tokens, int vocab_size,
    int hidden_size, int vocab_offset, int local_vocab) {

    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int32_t tok_id = token_ids[token_idx];
    scalar_t* dst_row =
        out + static_cast<int64_t>(token_idx) * hidden_size;

    // TD-GOLDEN-EMB-OOB: the table holds only this rank's vocab shard
    // [vocab_offset, vocab_offset + local_vocab) — index it with the LOCAL
    // row. Out-of-bounds (vs full vocab) or out-of-shard: zero the row
    // (shard misses are summed away by the caller's allreduce).
    const int32_t row = tok_id - vocab_offset;
    if (tok_id < 0 || tok_id >= vocab_size || row < 0 || row >= local_vocab) {
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
            dst_row[i] = static_cast<scalar_t>(0);
        return;
    }

    const scalar_t* src_row =
        table + static_cast<int64_t>(row) * hidden_size;

    using vec_t = vec_n_t<scalar_t, VEC_SIZE>;
    const int num_vec = hidden_size / VEC_SIZE;
    const auto* v_src = reinterpret_cast<const vec_t*>(src_row);
    auto* v_dst = reinterpret_cast<vec_t*>(dst_row);

    for (int i = threadIdx.x; i < num_vec; i += blockDim.x)
        v_dst[i] = v_src[i];

    // Handle tail elements (when hidden_size is not a multiple of VEC_SIZE).
    const int tail_start = num_vec * VEC_SIZE;
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x)
        dst_row[i] = src_row[i];
}

// ── Dispatch helpers (embedding lookup) ─────────────────────────────────────

template <typename scalar_t>
static void dispatch_embedding_lookup(void* out, const void* table,
                                      const int32_t* token_ids,
                                      int num_tokens, int vocab_size,
                                      int hidden_size, cudaStream_t stream,
                                      int vocab_offset, int local_vocab) {
    if (num_tokens <= 0 || hidden_size <= 0) return;

    constexpr int max_vec = 16 / static_cast<int>(sizeof(scalar_t));
    const int vec_size = std::gcd(max_vec, hidden_size);

    const int max_block_size = (num_tokens < 256) ? 1024 : 256;
    const int block_size =
        std::min(std::max(hidden_size / vec_size, 1), max_block_size);
    const dim3 grid(num_tokens);
    const dim3 block(block_size);

    auto* typed_out = static_cast<scalar_t*>(out);
    auto* typed_table = static_cast<const scalar_t*>(table);

#define LAUNCH_EMBED(VS)                                                     \
    embedding_lookup_kernel<scalar_t, VS>                                    \
        <<<grid, block, 0, stream>>>(typed_out, typed_table, token_ids,      \
                                     num_tokens, vocab_size, hidden_size,    \
                                     vocab_offset, local_vocab)

    if constexpr (max_vec >= 8) {
        if (vec_size >= 8) { LAUNCH_EMBED(8); return; }
    }
    if constexpr (max_vec >= 4) {
        if (vec_size >= 4) { LAUNCH_EMBED(4); return; }
    }
    if (vec_size >= 2) { LAUNCH_EMBED(2); return; }
    LAUNCH_EMBED(1);

#undef LAUNCH_EMBED
}

// ── Public: launch_embedding_lookup ─────────────────────────────────────────

void launch_embedding_lookup(void* out, const void* embedding_table,
                             const int32_t* token_ids, int num_tokens,
                             int vocab_size, int hidden_size,
                             EmbeddingDtype table_dtype, void* stream,
                             int vocab_offset, int local_vocab) {
    if (num_tokens <= 0) return;
    if (hidden_size <= 0)
        throw std::invalid_argument("launch_embedding_lookup: hidden_size must be > 0");
    if (vocab_size <= 0)
        throw std::invalid_argument("launch_embedding_lookup: vocab_size must be > 0");
    // TD-GOLDEN-EMB-OOB: local_vocab == -1 → full table (offset must be 0).
    if (local_vocab < 0) {
        if (vocab_offset != 0)
            throw std::invalid_argument(
                "launch_embedding_lookup: vocab_offset requires local_vocab");
        local_vocab = vocab_size;
    }
    if (vocab_offset < 0 || vocab_offset + local_vocab > vocab_size)
        throw std::invalid_argument(
            "launch_embedding_lookup: shard [offset, offset+local) exceeds vocab");

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    switch (table_dtype) {
        case EmbeddingDtype::kFloat32:
            dispatch_embedding_lookup<float>(out, embedding_table, token_ids,
                                             num_tokens, vocab_size,
                                             hidden_size, cuda_stream,
                                             vocab_offset, local_vocab);
            break;
        case EmbeddingDtype::kBFloat16:
            dispatch_embedding_lookup<__nv_bfloat16>(
                out, embedding_table, token_ids, num_tokens, vocab_size,
                hidden_size, cuda_stream, vocab_offset, local_vocab);
            break;
        case EmbeddingDtype::kFloat16:
            dispatch_embedding_lookup<__half>(out, embedding_table, token_ids,
                                              num_tokens, vocab_size,
                                              hidden_size, cuda_stream,
                                              vocab_offset, local_vocab);
            break;
        default:
            throw std::invalid_argument(
                "launch_embedding_lookup: unsupported dtype");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_embedding_lookup failed: ") +
            cudaGetErrorString(err));
    }
}

// ── Bias addition kernel ────────────────────────────────────────────────────
// logits[t * vocab_size + v] += bias[v]

__global__ void add_bias_kernel(float* __restrict__ logits,
                                const float* __restrict__ bias,
                                int total_elements, int vocab_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    logits[idx] += bias[idx % vocab_size];
}

// ── Weight dtype → GEMM input dtype helper ──────────────────────────────────

static GemmInDtype to_gemm_in_dtype(EmbeddingDtype dtype) {
    switch (dtype) {
        case EmbeddingDtype::kFloat32:  return GemmInDtype::kFloat32;
        case EmbeddingDtype::kBFloat16: return GemmInDtype::kBFloat16;
        case EmbeddingDtype::kFloat16:  return GemmInDtype::kFloat16;
    }
    throw std::invalid_argument("launch_output_head: unsupported weight dtype");
}

// ── Public: launch_output_head ──────────────────────────────────────────────
// logits = hidden_states @ weight^T [+ bias]
//   hidden_states A = [num_tokens, hidden_size] row-major
//   weight        W = [vocab_size, hidden_size] row-major
//   logits        C = [num_tokens, vocab_size]  FP32
// This is exactly the row-major NT GEMM C[M,N] = A[M,K] @ W[N,K]^T with
// M = num_tokens, N = vocab_size, K = hidden_size. FP32 accumulation matches
// the prior CUBLAS_COMPUTE_32F numerics. Graph-capturable (no cuBLAS handle).

void launch_output_head(float* logits, const void* hidden_states,
                        const void* weight, const float* bias,
                        int num_tokens, int vocab_size, int hidden_size,
                        EmbeddingDtype weight_dtype, void* stream) {
    if (num_tokens <= 0) return;
    if (vocab_size <= 0)
        throw std::invalid_argument("launch_output_head: vocab_size must be > 0");
    if (hidden_size <= 0)
        throw std::invalid_argument("launch_output_head: hidden_size must be > 0");

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    launch_bf16_gemm_nt(
        logits, hidden_states, weight,
        num_tokens, vocab_size, hidden_size,
        to_gemm_in_dtype(weight_dtype), GemmAccOutDtype::kFloat32,
        cuda_stream);

    // Optional bias addition.
    if (bias != nullptr) {
        const int total = num_tokens * vocab_size;
        constexpr int kBlockSize = 256;
        const int grid_size = (total + kBlockSize - 1) / kBlockSize;
        add_bias_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
            logits, bias, total, vocab_size);
    }
}

}  // namespace layerstorm::compute

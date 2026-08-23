// Embedding lookup and output head (LM head) CUDA kernels for LayerStoRm.
// Embedding lookup: custom gather kernel. Output head: self-contained BF16
// GEMM (bf16_gemm.cu), FP32 accumulation, no cuBLAS.

#pragma once

#include <cstdint>

namespace layerstorm::compute {

// Runtime data type selection for embedding/output-head weights.
enum class EmbeddingDtype { kFloat32, kBFloat16, kFloat16 };

// Token embedding lookup: out[t,:] = table[token_ids[t] - vocab_offset,:]
//   embedding_table: [local_vocab, hidden_size], row-major
//   token_ids:       [num_tokens], values in [0, vocab_size)
//   out:             [num_tokens, hidden_size], same dtype as table
//
// TD-GOLDEN-EMB-OOB: vocab-sharded tables (TP) hold rows
// [vocab_offset, vocab_offset + local_vocab). Tokens outside this rank's
// shard produce a ZERO row — the caller allreduce-sums across ranks to
// reconstruct the full embedding. local_vocab == -1 means the table is the
// full vocab (vocab_offset must be 0): the original single-table behavior.
// Out-of-bounds token IDs (vs full vocab) also produce a zero row.
void launch_embedding_lookup(void* out, const void* embedding_table,
                             const int32_t* token_ids, int num_tokens,
                             int vocab_size, int hidden_size,
                             EmbeddingDtype table_dtype,
                             void* stream /*cudaStream_t*/,
                             int vocab_offset = 0, int local_vocab = -1);

// Output head (LM head): logits = hidden_states @ weight^T [+ bias]
//   hidden_states: [num_tokens, hidden_size], in weight_dtype
//   weight:        [vocab_size, hidden_size], in weight_dtype
//   bias:          [vocab_size] FP32, or nullptr if absent
//   logits:        [num_tokens, vocab_size], always FP32
void launch_output_head(float* logits, const void* hidden_states,
                        const void* weight, const float* bias,
                        int num_tokens, int vocab_size, int hidden_size,
                        EmbeddingDtype weight_dtype,
                        void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute

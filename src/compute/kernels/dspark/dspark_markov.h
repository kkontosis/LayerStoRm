// DSpark sequential Markov head CUDA kernel interfaces (DSP-4).
//
// The suffix-decay mitigation (INV-DSPARK-MARKOV): a low-rank first-order
// transition bias B = W1·W2ᵀ conditions every draft position on its
// ACTUALLY-SAMPLED predecessor.  Vanilla head semantics (authoritative refs:
// ref/DeepSpec deepspec/modeling/dspark/markov_head.py `VanillaMarkov` +
// ref/vllm qwen3_dspark.py `DSparkMarkovHead` / dspark/speculator.py
// `_sample_sequential`; checkpoint markov_head_type "vanilla"):
//
//   e        = markov_w1[x_{k-1}]        BF16 [r]   (nn.Embedding(V, r) row
//                                                    of the PREVIOUS token)
//   bias     = markov_w2 @ e             FP32 [V]   (nn.Linear(r, V) — weight
//                                                    STORED [V, r], so
//                                                    bias_v = w2[v,:]·e)
//   logits_k = base_logits[k] + bias     FP32 [V]   (no scaling anywhere)
//   x_k      = argmax(logits_k)          greedy proposal; FIRST index on
//                                        exact ties (torch.argmax)
//
// Step 0's previous token is the ANCHOR (the last accepted real token);
// each step's sampled x_k feeds the next step's lookup.  The gamma-step
// serial loop stays ON-DEVICE: per step, kernel 1 (bias + add + per-block
// argmax partials) then kernel 2 (final argmax reduce -> writes x_k AND
// gathers markov_w1[x_k] for the next step) — no host round-trip anywhere.
//
// Deterministic by construction: fixed lane-strided dot + butterfly reduce
// order, fixed partial-reduce order, explicit lowest-index tie-break.

#pragma once

#include <cstdint>

namespace layerstorm::compute {

/// Per-block argmax partial (kernel 1 -> kernel 2 scratch element).
struct DsparkMarkovPartial {
    float val;
    int32_t idx;
};

/// Vocab rows covered by one bias/argmax block (scratch sizing contract —
/// the runtime allocates dspark_markov_num_blocks(V) partials).
inline constexpr int kDsparkMarkovRowsPerBlock = 64;

inline int dspark_markov_num_blocks(int64_t vocab_size) {
    return static_cast<int>((vocab_size + kDsparkMarkovRowsPerBlock - 1) /
                            kDsparkMarkovRowsPerBlock);
}

/// Kernel 1 of one Markov step: corrected[v] = base[v] + w2[v,:]·e for all
/// v in [0, vocab), plus per-block argmax partials over corrected.
///   corrected: [vocab] FP32 out (kept for DSP-5/DSP-6 consumers)
///   base:      [vocab] FP32 (one row of run_step's base_logits)
///   w2:        [vocab, rank] BF16 (markov_w2.weight, Linear(r->V) storage)
///   e:         [rank] BF16 (markov_w1 row of the previous token)
///   partials:  [dspark_markov_num_blocks(vocab)] DsparkMarkovPartial scratch
void launch_dspark_markov_bias_argmax(float* corrected, const float* base,
                                      const void* w2, const void* e,
                                      void* partials, int vocab, int rank,
                                      void* stream /*cudaStream_t*/);

/// Kernel 2 of one Markov step: reduce the partials to the global argmax
/// (lowest index on exact ties), map it to the TARGET vocab, write the
/// mapped id to token_out, and — unless e_next is nullptr (last step) —
/// gather markov_w1[mapped] into e_next for the NEXT step's bias.  Single
/// block; the sequential dependency chains entirely on the stream.
///
/// Reduced draft vocab (TD-DSPARK-VOCAB-REMAP, vLLM d2t semantics —
/// qwen3_dflash.py `map_draft_to_target`): the argmax is a DRAFT-vocab id;
/// target_id = draft_id + d2t[draft_id].  d2t == nullptr is the full-vocab
/// identity (target_id = draft_id).  markov_w1 is ALWAYS target-vocab
/// (it embeds previously SAMPLED tokens, which are target ids), so the
/// e-gather uses the mapped id.
///   token_out: &draft_ids[k] (device, int32) — TARGET-vocab id
///   e_next:    [rank] BF16 out, or nullptr on the final step
///   w1:        [target_vocab, rank] BF16 (markov_w1.weight)
///   d2t:       [draft_vocab] I64 offsets (device), or nullptr (identity)
void launch_dspark_markov_finalize(int32_t* token_out, void* e_next,
                                   const void* partials, int num_blocks,
                                   const void* w1, int rank,
                                   const void* d2t,
                                   void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute

// DeepSeek-V4 attention prep/compress kernels (V4-5a) — CUDA-free launchers
// (INV-GPU-1: callable from non-designated TUs).
//
// Model reference: ref/llama.cpp/src/models/deepseek4.cpp
//   build_attention (:796) q/kv prep, build_overlap_compressed_kv_from_state
//   (:440, CSA/LID), build_hca_compressed_kv_from_state (:382), inverse rope
//   (:1061), sinks via build_attn_mha.
//
// V4 entry mapping (duplicated-rope layout, dossier E+F checkpoint §4): the
// deps decode/prefill kernels score q_nope·k_nope(512) + q_rope·k_rope(64)
// and output softmax·v_nope(512). We store
//   k_nope := [448-dim nope | 64-dim ROPED pe]  (the full model kv vector)
//   k_rope := the roped pe again (64, duplicate)
//   v_nope := same 512 vector as k_nope        (model V == K)
// and query-side q_nope := [448 | 0×64], q_rope := roped q_pe. The score then
// equals the model's 512-dim dot (the k_nope[448:512] columns are zeroed by
// q) and V is exact. sm_scale = 1/sqrt(head_dim=512), NO yarn mscale
// (ticket-D finding).
//
// RoPE convention: interleaved consecutive pairs (x[2i], x[2i+1]) — llama.cpp
// LLAMA_ROPE_TYPE_NORM for LLM_ARCH_DEEPSEEK4 — with the ticket-D cos|sin
// half-row tables: row = [cos_0..cos_{r/2-1} | sin_0..sin_{r/2-1}], row
// stride = rope_dim floats, frequency index == pair index.
//
// FP8 cache entries are the deps V4CacheLayout (1160 B): [k_nope 512 FP8 |
// k_scale f32 | k_rope 64 BF16 | v_nope 512 FP8 | v_scale f32]; scale =
// amax/448, dequant = fp8·scale (byte-compatible with deps v4_fp8_k_append).

#pragma once

#include <cstdint>

namespace layerstorm::compute {

// ── Q prep ───────────────────────────────────────────────────────────────
// Per (token, head): parameterless RMS over the head_dim vector (pre-rope,
// deepseek4.cpp:834), then split [head_dim-rope_dim | rope_dim], rope the pe
// part at positions[token], and emit
//   q_nope[t,h,:] = [normed 448 | 0×64]   (head_dim wide)
//   q_rope[t,h,:] = roped pe              (rope_dim wide)
// q_in is [num_tokens, h_q, head_dim] BF16 (contiguous rows per head).
void launch_v4_q_prep(
    void* q_nope_out,
    void* q_rope_out,
    const void* q_in,
    const int* positions,      // [num_tokens] device
    const void* cos_sin,       // [max_pos, rope_dim] f32 cos|sin half rows
    float rms_eps,
    int num_tokens,
    int h_q,
    int head_dim,              // 512
    int rope_dim,              // 64
    void* stream /*cudaStream_t*/);

// ── Raw (SWA-tier) KV append ─────────────────────────────────────────────
// Input kv rows are the post-kv_norm latent [num_tokens, head_dim] BF16
// (448 nope | 64 un-roped pe). Ropes the pe at positions[t], assembles the
// duplicated-rope 512 vector and writes the 1160-B FP8 entry at
// slots[t] into kv_cache (flat slot indexing, deps convention).
void launch_v4_raw_kv_append(
    const void* kv_in,
    const int* positions,      // [num_tokens] device
    const int* slots,          // [num_tokens] device — flat SWA-tier slots
    void* kv_cache,
    const void* cos_sin,
    int num_tokens,
    int head_dim,              // 512
    int rope_dim,              // 64
    void* stream /*cudaStream_t*/);

// ── Compressor state ring write ──────────────────────────────────────────
// Scatter [num_tokens, state_dim] rows into ring rows (positions[t] mod
// ring_capacity). Used for kv_state / score_state / indexer states.
void launch_v4_state_ring_write(
    void* ring,                // [ring_capacity, state_dim] BF16
    const void* src,           // [num_tokens, state_dim] BF16
    const int* positions,      // [num_tokens] device — global token positions
    int ring_capacity,
    int state_dim,
    int num_tokens,
    void* stream /*cudaStream_t*/);

// ── Model-faithful overlap/HCA compression + insert ──────────────────────
// llama.cpp build_overlap_compressed_kv_from_state / build_hca_...:
// For compressed block j (first_block <= j < first_block+num_blocks):
//   window rows w = 0..W-1 (W = 2*stride overlap, stride HCA):
//     overlap: w < stride → token t = (j-1)*stride + w, PREV half
//              (state[0:D]); else t = j*stride + (w-stride), CUR half
//              (state[D:2D]).  t < 0 → excluded (-inf score).
//     hca:     t = j*stride + w, single-half state [0:D].
//   score[w][c] = score_state_row(t)[half·D + c] + ape[t % stride][half·D + c]
//   per-CHANNEL softmax over w → weighted sum of kv_state halves → [D]
//   → weighted RMS norm (norm_w [D] F32, rms_eps)
//   → split [D-rope | rope], rope pe at position (j+1)*stride - 1
//   → duplicated-rope output.
// Output modes:
//   kFp8Entry:     1160-B entry at slots[j - first_block] in kv_cache.
//   kIndexerPaged: FP8 row + f32 scale into the lightning-indexer page
//                  layout ([page_tokens rows × D FP8 | page_tokens f32
//                  scales] per page of page_bytes): slot s → page s /
//                  page_tokens, row s % page_tokens. Scale = amax/448.
//   kBf16Rows:     TQ codec staging (V4-5T): the compressed roped vector
//                  as BF16 rows — bf16_rows[j - first_block][0..D) plus
//                  the roped tail duplicated to bf16_rope_rows
//                  [j - first_block][0..rope_dim) (the same duplicated-
//                  rope content the FP8 entry carries). The caller then
//                  quantizes+packs via launch_v4_tq_entry_append (deps
//                  v4_tq_k_append: L2-normalize → Π-rotate → 4-bit pack,
//                  644-B entries, V == K like the FP8 write).
struct V4CompressArgs {
    const void* kv_state = nullptr;     // [ring_capacity, state_dim] BF16
    const void* score_state = nullptr;  // [ring_capacity, state_dim] BF16
    int ring_capacity = 0;
    int state_dim = 0;                  // 2*D (overlap) or D (hca)
    bool overlap = true;                // CSA/LID overlap vs HCA single-half
    int stride = 0;                     // 4 (CSA/LID) or 128 (HCA)
    const void* ape = nullptr;          // [stride, state_dim] F32
    const void* norm_w = nullptr;       // [D] F32
    const void* cos_sin = nullptr;      // compress-theta table
    float rms_eps = 1e-6f;
    int D = 0;                          // 512 main, 128 indexer
    int rope_dim = 64;
    int first_block = 0;
    int num_blocks = 0;
    const int* slots = nullptr;         // [num_blocks] device

    enum class Out { kFp8Entry, kIndexerPaged, kBf16Rows };
    Out out_mode = Out::kFp8Entry;
    void* kv_cache = nullptr;           // kFp8Entry: tier base
    void* idx_pages = nullptr;          // kIndexerPaged: region base
    int idx_page_tokens = 0;            //   rows per page
    int64_t idx_page_bytes = 0;         //   bytes per page
    void* bf16_rows = nullptr;          // kBf16Rows: [num_blocks, D] BF16
    void* bf16_rope_rows = nullptr;     // kBf16Rows: [num_blocks, rope_dim]
};
void launch_v4_compress_insert(const V4CompressArgs& args,
                               void* stream /*cudaStream_t*/);

// ── V4 TQ entry append (V4-5T, TD-V4-TQ-DEVICE) ──────────────────────────
// Quantize+pack compressed BF16 vectors into 644-B V4 TQ entries at
// slots[t] of a TQ-format tier (deps sm120/prep/v4_tq_k_append: per row
// L2-normalize → Π-rotate → Lloyd-Max 4-bit pack; K_ROPE stored BF16
// as-is; V input == K input mirrors the FP8 duplicated write).
struct V4TqAppendArgs {
    const void* rows = nullptr;        // [num_tokens, head_dim] BF16
    const void* rope_rows = nullptr;   // [num_tokens, rope_dim] BF16
    void* kv_cache = nullptr;          // TQ tier base (644-B entries)
    const int* slots = nullptr;        // [num_tokens] physical entry slots
    int num_tokens = 0;
    int head_dim = 512;
    int rope_dim = 64;
    const float* Pi = nullptr;         // [head_dim, head_dim] device
    const float* centroids = nullptr;  // [num_centroids] device
    const float* boundaries = nullptr; // [num_centroids - 1] device
    int num_centroids = 16;
};
void launch_v4_tq_entry_append(const V4TqAppendArgs& args,
                               void* stream /*cudaStream_t*/);

// ── Two-way LSE-weighted partial-attention merge (V4-5T) ─────────────────
// out_a/lse_a := merge((out_a, lse_a), (out_b, lse_b)) — softmax partials
// over DISJOINT key sets in the SAME value space, LSEs in NATURAL log
// units. Per (row, head): m = max(la, lb); wa = exp(la-m); wb = exp(lb-m);
// out = (wa·oa + wb·ob) / (wa+wb); lse = m + log(wa+wb). A side with
// lse = -inf (no keys) contributes weight 0. Used to fold the csa_tq
// compressed-tier partial into the FP8 SWA-only partial (the deps csa_tq
// kernel carries no SWA support by design — its header names this merge).
struct V4LseMerge2Args {
    void* out_a = nullptr;         // [rows, heads, head_dim] BF16 (in/out)
    float* lse_a = nullptr;        // [rows, heads] F32 (in/out)
    const void* out_b = nullptr;   // [rows, heads, head_dim] BF16
    const float* lse_b = nullptr;  // [rows, heads] F32
    int rows = 0;
    int heads = 0;
    int head_dim = 0;
};
void launch_v4_lse_merge2(const V4LseMerge2Args& args,
                          void* stream /*cudaStream_t*/);

// ── Entry gather (prefill staging) ───────────────────────────────────────
// Copy fixed-size cache entries (entry_bytes each) at src physical slots
// into contiguous staging rows [dst_row_offset, dst_row_offset+count).
void launch_v4_entry_gather(
    void* dst,                 // staging base (entry_bytes-stride rows)
    const void* src_cache,     // tier base
    const int* slots,          // [count] device — physical entry slots
    int count,
    int entry_bytes,           // 1160
    int dst_row_offset,
    void* stream /*cudaStream_t*/);

// ── Logical→physical slot translation (V4-7b, trap #11) ──────────────────
// out[i] = (in < 0 || in >= n_valid) ? -1
//        : page_table[in / entries_per_page] * entries_per_page
//          + in % entries_per_page
// where in = logical_in ? logical_in[i] : i (identity/iota mode — used to
// materialize the HCA dense visibility as physical entry slots). page_table
// is a DEVICE array of pool-relative physical page ids covering logical
// pages [0, ceil(n_valid / entries_per_page)). Writes count outputs; the
// caller sizes count to the -1-padded 64-multiple bound the attention
// kernel expects (iota mode pads i >= n_valid to -1; lightning's own -1
// padding passes through).
void launch_v4_slot_translate(
    int* out,
    const int* logical_in,     // nullable → iota
    const int* page_table,     // [n_pages] device
    int entries_per_page,
    int n_valid,               // logical entries in existence (causal bound)
    int count,                 // outputs written (>= n_valid in iota mode)
    void* stream /*cudaStream_t*/);

// ── Batched-prefill per-row logical index build (superchunk port) ────────
// out[i, j] (i < rows, j < topk): row_num_blocks[i] <= topk (or null
// lightning_in) → per-row IOTA (j < row_num_blocks[i] ? j : -1 — the
// deterministic all-visible selection, ticket-J rule applied per row);
// else the row's lightning top-k passes through (-1 padding preserved).
// In-place merge (out == lightning_in) is legal. Output is LOGICAL block
// ids — feed launch_v4_slot_translate (count = rows*topk) after.
void launch_v4_prefill_indices(
    int* out,                  // [rows, topk] device
    const int* lightning_in,   // [rows, topk] device, nullable
    const int* row_num_blocks, // [rows] device — per-row visible blocks
    int rows,
    int topk,
    void* stream /*cudaStream_t*/);

// ── Batched-prefill per-row SWA block table (superchunk port) ────────────
// Builds the [rows, window] per-token index list consumed by the decode
// kernel at swa_page_block_size == 1 over the chunk's raw-entry staging
// (rows [0, w_pref) = ring prefix for positions [p0-w_pref, p0); rows
// [w_pref, w_pref+R) = the chunk's entries at position p0+i). Row i, slot
// j < swa_len[i]: staging row of position (p0+i) - swa_len[i] + 1 + j
// (ascending chronological window); -1 beyond swa_len[i].
void launch_v4_prefill_swa_bt(
    int* bt,                   // [rows, window] device
    const int* swa_len,        // [rows] device — min(pos+1, window)
    int rows,
    int window,
    int w_pref,
    void* stream /*cudaStream_t*/);

// ── Attention-sink post-epilogue ─────────────────────────────────────────
// Exact sink fold on the FINAL (out, natural-unit LSE) of an attention op:
//   out[t,h,:] *= sigmoid(lse[t,h] − sinks[h + head_offset])
//   lse[t,h]    = logaddexp(lse[t,h], sinks[h + head_offset])
// Softmax-denominator identity: out·L/(L+e^s) with L = e^lse. Applies after
// split-KV combine / chunk LSE-merge (never per split).
void launch_v4_attn_sinks(
    void* out,                 // [num_tokens, h_q, d_v] BF16, in-place
    void* lse,                 // [num_tokens, h_q] f32 natural units, in-place
    const void* sinks,         // [num_heads_total] f32
    int head_offset,           // rank's first head (TP)
    int num_tokens,
    int h_q,
    int d_v,
    void* stream /*cudaStream_t*/);

// ── Output inverse RoPE ──────────────────────────────────────────────────
// In-place on the attention output [num_tokens, h_q, head_dim] BF16: the
// last rope_dim dims of every head get the INVERSE rotation at
// positions[token] (ggml_rope_ext_back, deepseek4.cpp:1061).
void launch_v4_out_inverse_rope(
    void* out,
    const int* positions,      // [num_tokens] device
    const void* cos_sin,       // per-layer table (base or compress)
    int num_tokens,
    int h_q,
    int head_dim,              // 512
    int rope_dim,              // 64
    void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute

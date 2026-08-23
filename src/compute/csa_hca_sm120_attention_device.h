// DeepSeek-V4 CSA/HCA/SWA hybrid AttentionDevice (V4-5a/5d/5e) — public seam.
//
// CUDA-free header (INV-GPU-1): the concrete class lives in the .cpp
// (designated CUDA TU); configuration + per-op calls go through free-function
// bridges that static_cast the AttentionDevice* down (SnapMLA/TQ pattern).
//
// The device wraps the DONE deps/LayerStoRmKernels V4K kernels via their C++
// entry points: run_csa_fp8_decode_kernel<64|128> (CSA sparse / HCA dense /
// SWA-only decode are ONE kernel parameterized by topk + indices),
// get_mla_metadata + mla_combine (split-KV), lightning score/topk (paged
// indexer tier), and CsaFp8DecodeGraphRunner (V4-5e). Sink folding and the
// output inverse RoPE (V4-5d) are engine-side post-epilogues
// (compute/kernels/attention/v4_prep.h).
//
// PREFILL runs the same decode kernel with one batch row per query: the
// per-row sparse indices express BOTH the per-query compressed visibility
// (CSA top-k / HCA prefix) AND the per-query raw sliding window, over a
// gathered contiguous entry staging (v4_entry_gather). Key order is
// irrelevant to attention; -1 indices are masked by the kernel.
//
// Physical addressing contract: all indices/slots passed to the device are
// PHYSICAL entry slots in the respective tier (page_idx * entries_per_page +
// entry_in_page); logical→physical translation is the caller's (V4-7b).

#pragma once

#include <cstdint>
#include <memory>

#include "core/attention_device.h"
#include "compute/kernels/attention/v4_prep.h"  // V4CompressArgs (TQ bridge)

namespace layerstorm::compute {

struct TqResources;  // tq_init.h — codebook + per-layer Π (V4-5T)

/// Factory: every csa_hca* backend arm (csa_hca FP8, csa_hca_tq,
/// csa_hca_tq_mix). The ARCH device is one class; the per-tier CODEC
/// (fp8 vs tq4) is baked at configure() time (INV-BH-7:
/// construction/init-baked, no per-call registry dispatch) — the
/// arch × codec composition of the attention refactor V2 (TD-V4-TQ-DEVICE).
std::unique_ptr<AttentionDevice> make_csa_hca_sm120_attention_device(
    config::GpuRef gpu);

/// Per-tier KV codec of the compressed tiers (SWA is always FP8).
enum class V4TierCodec { kFp8, kTq4 };

// ── Configuration (once, at engine init) ─────────────────────────────────

struct V4DeviceOptions {
    int max_batch = 0;          ///< decode rows bound (graph batch sizes)
    int max_attn_rows = 512;    ///< max rows per (prefill) attention call
    int h_q = 0;                ///< query heads on this rank
    int head_dim = 512;
    int rope_dim = 64;
    float sm_scale = 0.0f;      ///< 1/sqrt(head_dim) — NO yarn mscale
    float rms_eps = 1e-6f;
    int topk = 512;             ///< CSA lightning top-k (multiple of 64)
    int sliding_window = 128;
    int num_sm_parts = 32;      ///< split-KV partitions (decode)
    /// DET-REDUCE: bit-reproducible softmax-denominator reduction in the
    /// deps csa_fp8 decode kernel (fixed-order cross-warp combine instead
    /// of arrival-order atomicAdd). Wired from compute.deterministic_reduce
    /// / LAYERSTORM_DETERMINISTIC_REDUCE like the TQ/SnapMLA backends.
    bool deterministic_reduce = false;

    /// V4-5T (TD-V4-TQ-DEVICE): per-tier compressed-KV codec. kTq4 tiers
    /// store 644-B TQ entries (deps csa_tq family) and decode via the
    /// two-pass compressed(TQ) + SWA(FP8) LSE merge; requires
    /// csa_hca_device_set_tq. csa_hca → fp8/fp8; csa_hca_tq → tq/tq;
    /// csa_hca_tq_mix → tq(CSA)/fp8(HCA). SWA is ALWAYS FP8.
    V4TierCodec csa_codec = V4TierCodec::kFp8;
    V4TierCodec hca_codec = V4TierCodec::kFp8;

    // Tier geometry (VramLayout::v4). Entry bytes are the deps 1160-B format.
    int csa_entries_per_page = 64;
    int hca_entries_per_page = 2;
    int swa_page_tokens = 128;

    // Lightning-indexer tier paging: entries (compressed blocks) per page +
    // page byte stride ([entries × index_head_dim FP8 | entries × f32]).
    int idx_entries_per_page = 0;
    int64_t idx_page_bytes = 0;
    int index_n_heads = 64;
    int index_head_dim = 128;
    int max_index_blocks = 0;   ///< scores scratch bound (per row)

    // Per-layer rope table selection is the caller's; the device receives
    // the table PER CALL (positions index ticket-D cos|sin half rows).
};

void csa_hca_device_configure(AttentionDevice* dev, const V4DeviceOptions& o);

/// Allocate/resize the attention workspaces: out/lse, split-KV accumulators
/// (max_attn_rows + num_sm_parts + 1 accum rows), scheduler metadata, and the
/// entry-gather staging for max_staged_entries 1160-B rows.
void csa_hca_device_set_scratch(AttentionDevice* dev, int max_staged_entries);

// ── Attention (decode + prefill-as-decode) ───────────────────────────────

struct V4AttentionArgs {
    int rows = 0;               ///< batch rows (decode: B; prefill: chunk len)
    const void* q_nope = nullptr;  ///< [rows, h_q, head_dim] BF16 ([448|0])
    const void* q_rope = nullptr;  ///< [rows, h_q, rope_dim] BF16 (roped)

    // Compressed tier (null cache + topk 0 → SWA-only).
    const void* comp_cache = nullptr;   ///< tier base (CSA or HCA)
    const int* sparse_indices = nullptr;///< [rows, topk] device, -1 padded
    int topk = 0;                       ///< multiple of 64 (or 0)

    // Raw SWA tier (decode path; prefill folds the window into the indices).
    const void* swa_cache = nullptr;
    const int* swa_block_table = nullptr; ///< [rows, stride] physical pages
    int swa_block_table_stride = 0;
    const int* swa_seqlens = nullptr;     ///< [rows] device (0 → no SWA pass)
    /// Per-call SWA page granularity override (0 → options swa_page_tokens).
    /// The deps loader resolves entries per token as
    /// bt[row*stride + tpos/pbs]*pbs + tpos%pbs — batched prefill passes 1
    /// so the block table becomes a per-token index list over the chunk's
    /// gathered/appended raw-entry staging (superchunk port).
    int swa_page_block_size = 0;

    int num_sm_parts = 0;       ///< 0 → options default

    /// V4-5T: the compressed tier of THIS call is TQ-coded (the caller
    /// resolves per attention type: CSA → options csa_codec, HCA →
    /// hca_codec). The device then runs the two-pass decode: csa_tq over
    /// comp_cache (q Π-rotated FP32 → FP32 rotated out → Π^T rotate-back)
    /// + the FP8 SWA-only arm, folded by the natural-units LSE merge
    /// (launch_v4_lse_merge2) BEFORE the shared sinks/inverse-rope
    /// epilogues. Requires layer_idx (per-layer Π) when set.
    bool comp_tq = false;
    int layer_idx = -1;

    // Post-epilogues (V4-5a sinks + V4-5d inverse rope).
    const void* sinks = nullptr;      ///< [heads_total] F32 (null → skip)
    int sink_head_offset = 0;
    /// V4-2c TP padding: number of REAL heads in the (possibly padded)
    /// h_q-tile buffers. 0 → options h_q. Epilogues (sink fold + inverse
    /// rope) run over the real heads only — pad-head rows are zero-q
    /// garbage that is never consumed. Valid only with rows == 1 when
    /// num_heads_real < h_q (head-major addressing; no token stride).
    int num_heads_real = 0;
    const int* positions = nullptr;   ///< [rows] device — query positions
    const void* rope_table = nullptr; ///< per-layer table (inverse rope)

    // Outputs (caller buffers): out [rows, h_q, head_dim] BF16,
    // lse [rows, h_q] F32 natural units (post-sink when sinks given).
    void* out = nullptr;
    float* lse = nullptr;

    void* stream = nullptr;  ///< cudaStream_t
};

void csa_hca_device_attention(AttentionDevice* dev, const V4AttentionArgs& a);

// ── V4-5T TQ codec bridges (TD-V4-TQ-DEVICE) ─────────────────────────────

/// Hand the device the TQ resources (codebook + per-layer Π/Π^T; d must
/// equal head_dim = 512). Init-time; required before any TQ-tier call.
void csa_hca_device_set_tq(AttentionDevice* dev, const TqResources* res);

/// TQ-tier compress-insert: run the V4 compressor (V4CompressArgs; the
/// out_mode/out-pointer fields of `ca` are ignored) into device-owned BF16
/// staging rows, then quantize+pack 644-B TQ entries at ca.slots of
/// ca.kv_cache (deps v4_tq_k_append; V == K duplicated like the FP8
/// write). layer_idx selects Π.
void csa_hca_device_tq_compress_insert(AttentionDevice* dev,
                                       const V4CompressArgs& ca,
                                       int layer_idx, void* stream);

// ── Entry gather (prefill staging) ───────────────────────────────────────
// Copy 1160-B entries at src physical slots into contiguous staging rows
// [0, count). Returns the staging base via the pointer argument.
struct V4EntryGatherArgs {
    const void* src_cache = nullptr;
    const int* slots = nullptr;   ///< [count] device — physical entry slots
    int count = 0;
    int dst_row_offset = 0;       ///< staging row to start writing at
};
void csa_hca_device_gather_entries(AttentionDevice* dev,
                                   const V4EntryGatherArgs& a,
                                   void** staging_base_out, void* stream);

// ── Lightning indexer selection (CSA layers) ─────────────────────────────
struct V4LightningArgs {
    int rows = 0;
    const void* q_proj = nullptr;      ///< [rows, n_heads*head_dim] BF16 roped
    const void* score_w = nullptr;     ///< [rows, n_heads] F32 (pre-scaled)
    const int* row_num_blocks = nullptr; ///< [rows] device — visible blocks
    const void* const* k_page_table = nullptr; ///< [rows, page_table_stride]
    int page_table_stride = 0;
    const int* block_endpoints = nullptr; ///< [max_blocks] device (causality)
    const int* query_positions = nullptr; ///< [rows] device (cutoffs)
    int topk = 0;                       ///< selected blocks per row
    int* indices_out = nullptr;         ///< [rows, topk] device (LOGICAL ids,
                                        ///< ascending, -1 padded; row stride
                                        ///< is exactly topk)
};
void csa_hca_device_lightning_select(AttentionDevice* dev,
                                     const V4LightningArgs& a, void* stream);

// ── Decode graph runners (V4-5e) ─────────────────────────────────────────
// One runner per (attention type arm, batch size); attend-only capture
// (decode + combine). Sinks + inverse rope run post-replay via
// csa_hca_device_graph_epilogue. Metadata is filled once at init (constant
// for the fixed topk).
struct V4GraphInitArgs {
    int batch = 0;
    int topk = 0;               ///< 0 for the SWA-only arm
    const void* comp_cache = nullptr;  ///< stable tier base (dummy for SWA)
    const void* swa_cache = nullptr;
    int max_swa_blocks = 0;
    int num_sm_parts = 1;
};
/// Returns an opaque runner handle (device-owned; freed with the device).
void* csa_hca_device_graph_init(AttentionDevice* dev, const V4GraphInitArgs& a);
void csa_hca_device_graph_update(AttentionDevice* dev, void* runner,
                                 const void* q_nope, const void* q_rope,
                                 const int* sparse_indices,
                                 const int* swa_block_table,
                                 const int* swa_seqlens, void* stream);
void csa_hca_device_graph_replay(AttentionDevice* dev, void* runner,
                                 void* stream);
/// out/lse pointers of the runner's device buffers.
void csa_hca_device_graph_outputs(AttentionDevice* dev, void* runner,
                                  void** out, float** lse);
/// Post-replay epilogue on the runner outputs: sinks + inverse rope
/// (same semantics as the V4AttentionArgs fields).
void csa_hca_device_graph_epilogue(AttentionDevice* dev, void* runner,
                                   int rows, const void* sinks,
                                   int sink_head_offset, const int* positions,
                                   const void* rope_table, void* stream);

}  // namespace layerstorm::compute

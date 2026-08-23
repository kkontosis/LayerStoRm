// EPM-1 (Phase 29): raw training-data dump writers for the Expert-Routing
// Prediction Model corpus (spec/MoE-SpeQ_NOTES.md §7, PLAN.md EPM-1).
//
// Two record streams, both little-endian binary, appended to files inside
// the configured dump directory (speculation.dspark.epm_dump_dir, env
// override LS_EPM_DUMP):
//
//   epm_blocks.bin  — one EPMB record per successful D_CMD_RUN_DSPARK_STEP:
//     the draft's per-backbone-layer residual-stream hiddens for the gamma
//     query positions (feature side, decision (B): exact-parity engine
//     capture, no torch DFlash port), plus the sampled draft ids and the
//     raw DSP-6 c_k when the confidence head ran.
//       u32 magic 'EPMB' (0x424D5045 LE)   u32 version (1)
//       u64 seq_id                          u32 block_idx (per-seq counter)
//       u32 anchor_pos    u32 anchor_token  u32 gamma
//       u32 n_layers      u32 hidden        u32 has_conf (0/1)
//       i32 draft_ids[gamma]
//       f32 conf[gamma]                     (present iff has_conf)
//       u16 hiddens[gamma * n_layers * hidden]   (BF16 bits, [γ, L, H])
//
//   epm_routing.bin — one EPMR record per (seq, decode position): the
//     verify/AR feed's routing at EVERY routed layer, captured at the F-1
//     attention-end gate (the production routing producer whose
//     topk_indices/topk_weights buffers the FETCH_AND_RUN seam consumes) —
//     label side, decision (A): full 256 pre-top-k router logits FP16 +
//     top-8 ids/weights.
//       u32 magic 'EPMR' (0x524D5045 LE)   u32 version (1)
//       u64 seq_id                          u32 token_pos
//       u32 n_rows        u32 n_experts     u32 topk
//       n_rows x { u32 layer_idx
//                  u16 logits_fp16[n_experts]
//                  i32 topk_ids[topk]
//                  f32 topk_w[topk] }
//
// The Python orchestrator appends the per-block metadata (accepted length,
// verify tokens, c_k, ...) to manifest.jsonl in the same directory;
// tools/elb_train/dataset.py joins records and manifest on
// (seq_id, anchor_pos) and assembles training shards (INV-EPM-DATA:
// sequence-level splits only).
//
// Threading: daemon thread only (INV-3.4.2) — no locking. Files are opened
// in append mode so successive engine runs accumulate into one corpus dir.
// All writers fail SOFT (disabled + spdlog error) on I/O failure: data
// collection must never take the engine down.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace layerstorm::config {
struct DsparkConfig;
}

namespace layerstorm::speculation {

/// Effective EPM dump directory: LS_EPM_DUMP env (non-empty and not "0")
/// overrides; LS_EPM_DUMP=0 forces OFF; otherwise dc.epm_dump_dir.
/// Empty string = dumping disabled. Reads the env on every call (cheap;
/// callers cache the resulting writer, not this).
std::string epm_dump_dir(const config::DsparkConfig& dc);

/// Scalar FP32 -> IEEE FP16 conversion (round-to-nearest-even, handles
/// inf/nan/subnormals) for the router-logit label dump (decision (A)).
uint16_t epm_f32_to_f16(float f);

/// Appends EPMB block records to <dir>/epm_blocks.bin. Owned by
/// DsparkRuntime (constructed only when dumping is enabled).
class EpmBlockDumper {
public:
    explicit EpmBlockDumper(const std::string& dir);
    ~EpmBlockDumper();
    EpmBlockDumper(const EpmBlockDumper&) = delete;
    EpmBlockDumper& operator=(const EpmBlockDumper&) = delete;

    bool ok() const { return fp_ != nullptr; }

    /// Write one block record. `hiddens_bf16` is host-resident
    /// [gamma, n_layers, hidden] BF16 bits; `conf` may be null (has_conf=0).
    void write_block(uint64_t seq_id, uint32_t block_idx, uint32_t anchor_pos,
                     uint32_t anchor_token, int gamma, int n_layers,
                     int hidden, const uint16_t* hiddens_bf16,
                     const int32_t* draft_ids, const float* conf);

private:
    std::FILE* fp_ = nullptr;
};

/// Accumulates per-layer routing rows for one (seq, position) and appends
/// finished EPMR records to <dir>/epm_routing.bin. Rows arrive layer by
/// layer from the attention-end gate; the pending record is flushed when
/// (seq, pos) changes, when `last_layer` is passed, and on destruction.
/// Owned by CommandDispatcher (constructed only when dumping is enabled).
class EpmRoutingDumper {
public:
    explicit EpmRoutingDumper(const std::string& dir, int n_experts, int topk);
    ~EpmRoutingDumper();
    EpmRoutingDumper(const EpmRoutingDumper&) = delete;
    EpmRoutingDumper& operator=(const EpmRoutingDumper&) = delete;

    bool ok() const { return fp_ != nullptr; }

    /// Append one routed layer's row (host-resident buffers). Converts the
    /// FP32 logits to FP16 into the pending record. `last_layer` = this is
    /// the model's last routed layer -> flush the record immediately.
    void add_row(uint64_t seq_id, uint32_t token_pos, uint32_t layer_idx,
                 const float* logits_f32, const int32_t* topk_ids,
                 const float* topk_w, bool last_layer);

    /// Write the pending record (no-op when nothing is pending).
    void flush_pending();

private:
    std::FILE* fp_ = nullptr;
    int n_experts_ = 0;
    int topk_ = 0;

    // Pending record state (one in flight — decode is single-sequence,
    // TD-DSPARK-BATCH; a (seq,pos) change flushes before accumulating).
    bool pending_ = false;
    uint64_t cur_seq_ = 0;
    uint32_t cur_pos_ = 0;
    uint32_t n_rows_ = 0;
    std::string body_;  ///< serialized rows of the pending record
};

}  // namespace layerstorm::speculation

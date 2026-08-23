// KV codec axis for the attention stack (attention refactor V2 P2).
//
// A "codec" is the KV storage format participant of an attention arm —
// orthogonal to the model ARCH (arch_base.h). The allocator's
// memory::KvCacheFormat is the precedent: memory already carries this axis
// (per-tier formats + entry bytes in memory::V4KvLayout, sizing in
// compute_v4_kv_layout). This header is the attention-stack view of that
// same authority — entry accounting and the arm composition map. Kernels
// stay per-geometry in the devices/deps; this layer never launches.
//
// Codec kinds and where each participant lives today:
//   kFull — full-precision storage. NOT built as a device arm anywhere (FP8
//           is DS4's native trained format; mission note: a BF16 arm must
//           be a trivial registration under this axis if ever wanted).
//   kFp8  — MLA: SnapMlaSm120AttentionDevice (fused_k_append / dequant
//           staging); V4: CsaHcaSm120AttentionDevice + the executor's
//           compress-insert path (1160-B entries, csa_fp8 decode kernels).
//   kTq4  — MLA: TqSm120AttentionDevice (Π-rotate + 4-bit pack append,
//           q_rotate → packed decode → v_rotate_back); V4 (P3,
//           TD-V4-TQ-DEVICE): deps csa_tq family (v4_tq_k_append 644-B
//           entries, splitkv_csa_tq decode, v4_tq_dequant_indexed,
//           CsaTqDecodeGraphRunner).
//
// Backend-arm composition (arch × codec — factory map, config
// compute.attention_backend):
//   snapmla        = MLA × kFp8      turboquant_mla = MLA × kTq4
//   csa_hca        = V4 × kFp8/kFp8  csa_hca_tq     = V4 × kTq4/kTq4
//   csa_hca_tq_mix = V4 × kTq4(CSA)/kFp8(HCA)      SWA tier: ALWAYS kFp8.

#pragma once

#include "core/memory/page_allocator.h"  // Pool + V4KvLayout + formats

namespace layerstorm::daemon {

enum class KvCodecKind { kFull, kFp8, kTq4 };

inline constexpr KvCodecKind codec_kind(memory::KvCacheFormat f) {
    switch (f) {
        case memory::KvCacheFormat::kTurboQuantMse4:
        case memory::KvCacheFormat::kV4Tq:
            return KvCodecKind::kTq4;
        case memory::KvCacheFormat::kSnapMlaFp8:
        case memory::KvCacheFormat::kV4Fp8:
        default:
            return KvCodecKind::kFp8;
    }
}

inline constexpr const char* codec_name(KvCodecKind k) {
    switch (k) {
        case KvCodecKind::kFull: return "full";
        case KvCodecKind::kTq4:  return "tq4";
        case KvCodecKind::kFp8:
        default:                 return "fp8";
    }
}

/// Per-tier V4 entry bytes for a side-tier pool, from the allocator's
/// layout (the single sizing authority): kSwa → swa (always FP8), kHca →
/// hca (arm-dependent), kIndexerK → the Lightning-Indexer tier entry
/// (index_head_dim FP8 + 4 B F32 scale — codec-independent). The CSA tier
/// lives in Pool::kMain (kv_bytes_per_page); not addressed per-entry here.
inline int64_t v4_tier_entry_bytes(const memory::V4KvLayout& v4,
                                   memory::Pool pool) {
    switch (pool) {
        case memory::Pool::kSwa:      return v4.swa_entry_bytes;
        case memory::Pool::kHca:      return v4.hca_entry_bytes;
        case memory::Pool::kIndexerK: return v4.indexer_entry_bytes;
        default:                      return 0;
    }
}

}  // namespace layerstorm::daemon

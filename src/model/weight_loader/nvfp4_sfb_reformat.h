#pragma once
// TD-GOLDEN bug #6: NVFP4 grouped-GEMM weight scales (scale_B) must be in the
// CUTLASS Sm1xx interleaved SFB layout, NOT raw row-major. The kernel binds
// scale_B via Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB; feeding raw
// [N, K/16] row-major scales mis-scales every 16-group (measured grouped-GEMM
// cosine 0.77 vs 1.00000 after reformatting — see GemmKernels NOTICE.md
// "NVFP4 Scale Layout (Sm1xx Interleaved)").
//
// Layout (atom = 128 N-rows x 4 K-groups, 512 bytes):
//   tile      = (n / 128) * ceil(G / 4) + (g / 4)
//   within    = (n % 32) * 16 + ((n % 128) / 32) * 4 + (g % 4)
//   offset    = tile * 512 + within
// For N % 128 == 0 and G % 4 == 0 (every projection in this engine) the
// interleaved buffer is exactly N * G bytes — same size as raw, so the
// reformat can replace the bytes in place via a scratch copy.
//
// Applied exactly ONCE, at PACK time (pack_nvfp4_expert + the prepacker) and
// at init-time pinned uploads of raw checkpoint tensors (reformat_sfb_host in
// engine.cpp). Since format 9.67.0 / quant "nvfp4-sm1xx", packed expert bytes
// are interleaved at EVERY tier (disk, host, VRAM) — there is no runtime
// reformat, and no host source may serve raw scales for cache slots.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace layerstorm::model {

/// Reformat one scale matrix from row-major [N, G] (G = K/16, UE4M3 bytes)
/// into the Sm1xx interleaved SFB layout. dst must hold
/// ceil(N/128)*128 * ceil(G/4)*4 bytes (== N*G when aligned). dst != src.
inline void reformat_nvfp4_sfb(uint8_t* dst, const uint8_t* src,
                               int64_t n_rows, int64_t groups) {
    const int64_t g_pad = ((groups + 3) / 4) * 4;
    const int64_t n_tiles_g = g_pad / 4;
    // Zero padding region first (aligned shapes make this a no-op copy-wise,
    // but keep it correct for ragged callers).
    const int64_t n_pad = ((n_rows + 127) / 128) * 128;
    std::memset(dst, 0, static_cast<size_t>(n_pad) * g_pad);
    for (int64_t n = 0; n < n_rows; ++n) {
        const int64_t tile_n = n / 128;
        const int64_t r0 = (n % 32) * 16;
        const int64_t r1 = ((n % 128) / 32) * 4;
        const uint8_t* srow = src + n * groups;
        for (int64_t g = 0; g < groups; ++g) {
            const int64_t tile = tile_n * n_tiles_g + (g / 4);
            dst[tile * 512 + r0 + r1 + (g % 4)] = srow[g];
        }
    }
}

/// In-place variant via scratch (sizes equal when N%128==0 && G%4==0).
inline void reformat_nvfp4_sfb_inplace(uint8_t* buf, int64_t n_rows,
                                       int64_t groups,
                                       std::vector<uint8_t>& scratch) {
    const size_t bytes = static_cast<size_t>(n_rows) * groups;
    if (scratch.size() < bytes) scratch.resize(bytes);
    reformat_nvfp4_sfb(scratch.data(), buf, n_rows, groups);
    std::memcpy(buf, scratch.data(), bytes);
}

}  // namespace layerstorm::model

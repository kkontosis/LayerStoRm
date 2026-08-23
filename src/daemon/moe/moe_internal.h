// Internal helpers shared by the src/daemon/moe/ TUs (driver + arch files).
// Not part of any public interface. Precedent: parallelism/dcp_executor_internal.h.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/device_backend.h"

namespace layerstorm::daemon {

// ── TD-DRIFT-ROOTCAUSE: DIAGNOSTIC-ONLY routing dump (off by default) ──────────
// Gated on LS_DRIFT_DUMP=<path>. When set, dumps a binary record per
// (sequence, layer, gpu) immediately after the top-K gate runs: the full
// pre-argmax router logit vector + the selected top-K expert ids + their gating
// weights. Two runs (same config twice, OR decay-on vs decay-off) can then be
// diffed at the logit level to locate the FIRST divergence and classify it
// (ULP-scale near-tie argmax flip vs large/garbage). UNSET = zero work, zero
// production impact. When SET it adds a D2H + full device sync, which DOES change
// timing — used only to compare dump-on vs dump-on, and (deliberately) as the
// race probe: forcing this serialization must not change the dumped routing if
// the pipeline is race-free. Record layout (little-endian):
//   int32 hdr[6] = {seq, layer_idx, gpu, num_tokens, n_experts, topk}
//   float logits[num_tokens*n_experts]
//   int32 idx   [num_tokens*topk]
//   float w     [num_tokens*topk]
// (Moved verbatim from dispatch_moe.cpp's anonymous namespace; `inline` so the
// gating hooks in arch_mla_moe.cpp / arch_deepseek_v4_moe.cpp share it.)
inline void drift_dump_routing(compute::DeviceBackend* dev_be, void* stream,
                               const void* router_logits_dev,
                               const void* topk_indices_dev,
                               const void* topk_weights_dev,
                               int num_tokens, int n_experts, int topk,
                               uint32_t layer_idx, int gpu) {
    static const char* path = std::getenv("LS_DRIFT_DUMP");
    if (!path || !*path) return;
    if (!dev_be || !router_logits_dev || !topk_indices_dev || !topk_weights_dev)
        return;
    if (num_tokens <= 0 || n_experts <= 0 || topk <= 0) return;
    static std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return;
    static std::atomic<uint64_t> seq{0};
    const size_t nlog = static_cast<size_t>(num_tokens) * n_experts;
    const size_t nk   = static_cast<size_t>(num_tokens) * topk;
    std::vector<float>   logits(nlog);
    std::vector<int32_t> idx(nk);
    std::vector<float>   w(nk);
    dev_be->set_device();
    dev_be->memcpy_d2h_async(logits.data(), router_logits_dev,
                             nlog * sizeof(float), stream);
    dev_be->memcpy_d2h_async(idx.data(), topk_indices_dev,
                             nk * sizeof(int32_t), stream);
    dev_be->memcpy_d2h_async(w.data(), topk_weights_dev,
                             nk * sizeof(float), stream);
    dev_be->synchronize_device();
    const uint64_t s = seq.fetch_add(1);
    int32_t hdr[6] = {static_cast<int32_t>(s), static_cast<int32_t>(layer_idx),
                      gpu, num_tokens, n_experts, topk};
    std::fwrite(hdr, sizeof(int32_t), 6, fp);
    std::fwrite(logits.data(), sizeof(float), nlog, fp);
    std::fwrite(idx.data(), sizeof(int32_t), nk, fp);
    std::fwrite(w.data(), sizeof(float), nk, fp);
    std::fflush(fp);
}

}  // namespace layerstorm::daemon

// hc-stream MEAN reduction (ticket J) — CUDA-free launcher (INV-GPU-1).
//
// The V4 dflash dspark draft consumes the target's aux hidden states as the
// MEAN over the hc_mult residual streams (ref/vllm deepseek_v4/nvidia/
// model.py:1101 — `mhc_post(...).mean(dim=1)` per captured layer), NOT the
// flattened hc*hidden row. This kernel produces that representation on the
// TARGET GPU before the cross-GPU aux capture copy.

#pragma once

namespace layerstorm::compute {

//   in:  [rows, hc * hidden] BF16 (the committed hc-wide residual rows)
//   out: [rows, hidden] BF16 — out[r, d] = mean_s in[r, s * hidden + d]
// FP32 accumulation, fixed stream order (deterministic).
void launch_hc_stream_mean(void* out, const void* in, int rows, int hc,
                           int hidden, void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute

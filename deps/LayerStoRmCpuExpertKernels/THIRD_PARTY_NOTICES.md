# Third-Party Notices — LayerStoRmCpuExpertKernels

LayerStoRmCpuExpertKernels is licensed under the Apache License 2.0 (see
`LICENSE.md`). Portions of this repository are copied from or derived from
the third-party projects listed below.

---

## ik_llama.cpp / llama.cpp / ggml

- Upstream: https://github.com/ikawrakow/ik_llama.cpp (a fork of
  https://github.com/ggerganov/llama.cpp / https://github.com/ggml-org/ggml)
- License: MIT —
  Copyright (c) 2023-2024 The ggml authors,
  Copyright (c) 2023-2024 The llama.cpp authors,
  Copyright (c) 2024-2025 The ik_llama.cpp authors
- What was copied / derived:
  - `csrc/compute/cpu/ik_vendor/` — a minimal **verbatim** closure of
    ik_llama.cpp (commit d47f484) covering the CPU GGUF quantized-GEMM
    kernels (Q8_0/Q5_0 legacy + Q4_K/Q5_K/Q6_K K-quants). The full MIT
    license text with the upstream copyright lines is in
    `csrc/compute/cpu/ik_vendor/LICENSE.ik`.
  - `csrc/compute/cpu/ik_barrier.h` — verbatim adaptation of
    `ggml_barrier_impl` (full MIT notice in the file header).
  - `csrc/compute/cpu/gguf_lossless.{h,cpp}` — bit-compatible K-quant × Q8_1
    integer-dot arithmetic reproducing ggml's `vec_dot_q4_K_q8_1` /
    `get_scale_min_k4` numerics.
  - `csrc/compute/cpu/nvfp4_cpu_kernel.cpp` — multi-row GEMV technique
    borrowed from ik_llama.cpp `iqk_gemm_floats.cpp`.
  - `csrc/compute/cpu/cpu_moe_kernels.{h,cpp}` — DeepSeek-V4 SwiGLU clamp
    semantics from llama.cpp `LLM_ARCH_DEEPSEEK4` / `ggml_swiglu_split`.

The MIT License text (from `ik_vendor/LICENSE.ik`, applying to all the
material above together with the copyright lines listed):

```
MIT License

Copyright (c) 2023-2024 The ggml authors (https://github.com/ggml-org/ggml/blob/master/AUTHORS)
Copyright (c) 2023-2024 The llama.cpp authors (https://github.com/ggml-org/llama.cpp/blob/master/AUTHORS)
Copyright (c) 2024-2025 The ik_llama.cpp authors (https://github.com/ikawrakow/ik_llama.cpp/blob/main/AUTHORS)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Apache License 2.0

This repository is licensed under the MIT License — see `LICENSE.md`
(Copyright 2026 Kimon Kontosis). It applies to the first-party code in this
repository; the `ik_vendor/` closure and the other MIT-derived material above
remain under the MIT License with the upstream copyrights.

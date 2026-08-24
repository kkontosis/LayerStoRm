# Third-Party Notices — LayerStoRm

LayerStoRm is licensed under the MIT License (see `LICENSE.md`).
Portions of this repository are derived from, adapted from, or reference the
third-party projects listed below. This file collects the required upstream
attributions and license notices. The dependency repositories
(`deps/LayerStoRmKernels`, `deps/LayerStoRmExpertKernels`,
`deps/LayerStoRmGemmKernels`, `deps/LayerStoRmCpuExpertKernels`) carry their
own `THIRD_PARTY_NOTICES.md` files for the material vendored there.

Where a section says "see MIT License text below", the full license text in
the appendix applies together with that section's copyright line(s).

---

## FlashMLA

- Upstream: https://github.com/deepseek-ai/FlashMLA (and the SM120 fork
  https://github.com/IISuperluminaLII/FlashMLA_Windows_Linux_sm120)
- License: MIT — Copyright (c) 2025 DeepSeek (see MIT License text below)
- What was derived: the attention-device drivers in `src/compute/` and
  `src/daemon/attention/` mirror FlashMLA's scheduling and parameter
  conventions. The SM120 MLA split-KV decode / combine / metadata kernels
  derived from FlashMLA live in `deps/LayerStoRmKernels` (see that repo's
  notices file). No FlashMLA sources are vendored or compiled in this
  repository.

## DeepSeek-V3 / DeepSeek-V3.2-Exp reference code

- Upstream: https://github.com/deepseek-ai/DeepSeek-V3 (MIT, Copyright (c)
  2023 DeepSeek — code license; the model weights carry a separate DeepSeek
  license) and https://github.com/deepseek-ai/DeepSeek-V3.2-Exp (MIT,
  Copyright (c) 2025 DeepSeek).
- What was derived: reference forward math in `tools/golden_ref/ref_forward.py`
  (naive MLA, YaRN RoPE, gating), RoPE/architecture conventions in
  `src/compute/rope_table.*` and the attention/MoE arch hooks, and the model
  configuration JSON copies under `test-data/DeepSeek-V3.2/`.
- See MIT License text below.

## llama.cpp / ggml

- Upstream: https://github.com/ggerganov/llama.cpp
- License: MIT — Copyright (c) 2023-2026 The ggml authors (see MIT License
  text below)
- What was derived: the host MXFP4 dequant port in
  `src/model/quantization/gguf_kquant.{h,cpp}` (`dequantize_row_mxfp4`,
  `ggml_e8m0_to_fp32_half`, the FP4 `kValues` table); the GGUF container
  byte-format and `ggml_type` constants in
  `src/model/weight_loader/gguf_reader.*`; DeepSeek-V4 graph semantics
  (swiglu clamp, hash gating, mHC, Hadamard indexer rotation) referenced from
  `src/compute/kernels/**` and `src/daemon/**`; ggml-derived reference dequant
  logic in `tools/scout/*_standalone.cpp`; `tools/mint_v4_hf_tokenizer.py`
  uses llama.cpp's `gguf-py` at runtime (not vendored).

## ik_llama.cpp

- Upstream: https://github.com/ikawrakow/ik_llama.cpp
- License: MIT —
  Copyright (c) 2023-2024 The ggml authors,
  Copyright (c) 2023-2024 The llama.cpp authors,
  Copyright (c) 2024-2025 The ik_llama.cpp authors
  (see MIT License text below)
- What was derived: the vendored CPU GGUF GEMM closure in
  `deps/LayerStoRmCpuExpertKernels/csrc/compute/cpu/ik_vendor/` (verbatim
  copies, commit d47f484; full license in that directory's `LICENSE.ik`),
  the `ik_barrier.h` thread barrier, and the multi-row GEMV technique in
  `nvfp4_cpu_kernel.cpp`.

## vLLM

- Upstream: https://github.com/vllm-project/vllm
- License: Apache-2.0 — Copyright contributors to the vLLM project
- What was derived (ported "lean", i.e. restructured without the vLLM class
  hierarchy): the tool-call / reasoning parsers and registry semantics in
  `python/server/tool_parsers.py`, `python/server/reasoning_parsers.py`,
  `python/tokenizer/tool_call_parser.py`; guided-decoding xgrammar usage
  patterns in `python/server/guided.py`; the streamed-error SSE shape and
  API-parity semantics in `python/server/http_server.py` and
  `python/cli/serve.py`; the rejection-sampler acceptance rule in
  `python/orchestrator/orchestrator.py`; the n-gram prompt-lookup proposer in
  `python/orchestrator/prompt_lookup.py`; CUDA vector-type helpers in
  `src/compute/kernels/smxx/norm/vec_types.cuh`; parts of the norm/gating/
  sampling kernels (see per-file headers).
- vLLM does not ship a NOTICE file; its LICENSE is the stock Apache-2.0 text
  (reproduced in this repository as `LICENSE.md`).

## SGLang

- Upstream: https://github.com/sgl-project/sglang
- License: Apache-2.0 — Copyright 2023-2024 SGLang Team
- What was derived: DeepSeek-V4 sparse-attention indexer semantics and
  adaptive speculation parameter heuristics referenced from
  `python/orchestrator/utility_scorer.py` and the attention kernels in
  `deps/LayerStoRmKernels` (see that repo's notices file). The SnapMLA-style
  FP8 KV-cache conventions follow the SGLang lineage ("SGLang-FluentLLM"
  reference implementation).

## NVIDIA TensorRT-LLM

- Upstream: https://github.com/NVIDIA/TensorRT-LLM
- License: Apache-2.0 — Copyright (c) 2011-2025 NVIDIA CORPORATION &
  AFFILIATES. All rights reserved.
- What was derived: RMSNorm kernel structure
  (`src/compute/kernels/sm120/norm/rmsnorm.cu`), top-k routing/sampling
  packed-score technique (`src/compute/kernels/sm120/sampling/sampling.cu`),
  and SM120 kernel best practices (see per-file headers).
- TensorRT-LLM ships no Apache-2.0 NOTICE file at its repository root, so
  there are no NOTICE contents to reproduce under Apache-2.0 §4(d).

## NVIDIA CUTLASS

- Upstream: https://github.com/NVIDIA/cutlass (consumed as the
  `3rd-party/cutlass` git submodule; not vendored in this tree)
- License: BSD-3-Clause — Copyright (c) 2017 - 2026 NVIDIA CORPORATION &
  AFFILIATES. All rights reserved. (full text below)
- What is used: CUTLASS/CuTe headers are a build dependency of the CUDA
  kernels (MMA atoms, tensor layouts). Binaries built from this repository
  incorporate CUTLASS header code; the BSD-3-Clause notice below applies to
  such binaries. Note: the `python/CuTeDSL` directory of upstream CUTLASS is
  under a separate NVIDIA EULA; LayerStoRm does not use it.

## TurboQuant (paper)

- Paper: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni,
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
  arXiv:2504.19874.
- The TurboQuant 4-bit KV-cache codec kernels in this project are an
  independent implementation of the paper's method. The Lloyd-Max codebooks
  shipped in `config/tq_codebooks/*.json` are numeric quantizer constants
  (centroids/boundaries of the optimal scalar quantizer for the rotated
  unit-sphere coordinate distribution). No code from the GPL-3.0 third-party
  reference implementation (https://github.com/0xSero/turboquant) is
  included in this repository.

## Model configuration and tokenizer test data (`test-data/`)

Small model-metadata files (config.json, generation_config.json,
tokenizer_config.json, tokenizer.json, chat templates, safetensors index
files) are included for tests and were obtained from the following model
repositories:

- `test-data/DeepSeek-V3.2/` — https://huggingface.co/deepseek-ai
  (MIT, Copyright (c) 2025 DeepSeek)
- `test-data/GLM-5.2/` — https://huggingface.co/zai-org (MIT,
  Copyright (c) 2026 Zhipu AI; license copy in `test-data/GLM-5.2/LICENSE`)
- `test-data/GLM-5/`, `test-data/GLM-4.7-Flash/` — https://huggingface.co/zai-org
  (MIT, Zhipu AI)
- `test-data/Kimi-K2.5/` — https://huggingface.co/moonshotai (Moonshot AI —
  K2-family models are published under Moonshot's Modified MIT License; see
  the upstream model repository for its terms)

The `.eval_results/*.yaml` files record published benchmark scores with their
source URLs. No model weights are distributed in this repository.

---

## Appendix A — MIT License text

The following license text applies to the MIT-licensed material identified
above, together with the copyright lines given in each section:

```
MIT License

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

## Appendix B — BSD-3-Clause (NVIDIA CUTLASS)

```
Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Appendix C — Apache License 2.0

This repository is licensed under the MIT License — see `LICENSE.md`.
It applies both to LayerStoRm itself (Copyright 2026 Kimon Kontosis) and to
the Apache-2.0-licensed upstream material identified above (vLLM, SGLang,
NVIDIA TensorRT-LLM).

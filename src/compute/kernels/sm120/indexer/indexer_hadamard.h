#pragma once

// TD-GLM-INDEXER-HADAMARD: orthonormal Walsh-Hadamard rotation of DSA
// lightning-indexer q/k rows (QuaRot-style).
//
// llama.cpp applies a fixed Hadamard rotation to BOTH indexer q and k before
// FP8-quantizing/storing k (ref/llama.cpp/src/models/deepseek32.cpp:281-286;
// matrix from ref/llama.cpp/src/llama-kv-cache.cpp:20-58 ggml_gen_hadamard;
// MIT License, Copyright (c) 2023-2026 The ggml authors — see
// THIRD_PARTY_NOTICES.md).
// The matrix is the Sylvester-construction Hadamard H_n seeded with
// H[0][0] = 1/sqrt(n), i.e. H = H_sylvester / sqrt(n): orthonormal, symmetric,
// an involution (H^2 == I). Because H is orthogonal and applied to both
// operands, the indexer score q·k is mathematically unchanged; its only effect
// is reducing the FP8 quantization error of the stored key.
//
// This kernel computes the equivalent in-place Fast Walsh-Hadamard Transform
// (natural/Hadamard order — exactly the Sylvester H, no permutation) scaled by
// 1/sqrt(dim), over `rows` BF16 rows of width `dim` (power of two; dim=128 is
// the GLM-5.2 index_head_dim case). Butterflies accumulate in FP32.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace layerstorm::compute {

// In-place FWHT of `rows` BF16 rows of width `dim` (power of two, 2..4096),
// scaled by 1/sqrt(dim). x is a device pointer to [rows, dim] BF16.
void launch_indexer_hadamard(void* x_bf16, int rows, int dim,
                             cudaStream_t stream);

}  // namespace layerstorm::compute

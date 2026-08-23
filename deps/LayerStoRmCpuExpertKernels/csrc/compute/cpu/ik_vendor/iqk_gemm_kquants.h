// Copied verbatim from ik_llama.cpp (commit d47f484), MIT License.
// Source: https://github.com/ikawrakow/ik_llama.cpp  (see LICENSE.ik)
// Part of the minimal CPU GGUF GEMM closure vendored for LayerStoRm C-6.
// DO NOT EDIT — keep byte-for-byte with upstream. Build glue lives outside.
#pragma once

#include "iqk_common.h"

#ifdef IQK_IMPLEMENT

#include <array>

bool iqk_set_kernels_kquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);

void iqk_gemm_q8kv_fa(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step);

bool iqk_convert_kquants_q8X_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);

#endif

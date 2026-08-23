#pragma once

// Re-exported parameter structs for SnapMLA prep kernels.
// Now that the submodule (18514ce) has proper .h headers, we just include them.

#include <sm120/prep/fused_q_quant.h>
#include <sm120/prep/fused_k_append.h>
#include <sm120/prep/dequant_ckv_indexed.h>

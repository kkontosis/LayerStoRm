# Extending the ik GGUF CPU vendor (TD-IK-STUBBED-CLOSURE)

This dir is a **minimal verbatim closure** of ik_llama.cpp (commit `d47f484`, MIT —
see `LICENSE.ik`) covering only the GGUF quant families the CPU expert path
currently exposes: **Q8_0, Q5_0** (legacy) and **Q4_K, Q5_K, Q6_K** (K-quants).
The full ik dispatcher + every quant family + the repack closure (~1.3M lines) is
deliberately NOT vendored. This guide makes adding a new family mechanical.

## Architecture (what's here)

| Piece | File | Role |
|---|---|---|
| GEMM family (legacy) | `iqk_gemm_legacy_quants.{cpp,h}` | verbatim ik — Q4_0/Q5_0/Q8_0 int8 GEMM |
| GEMM family (K-quant) | `iqk_gemm_kquants.{cpp,h}` | verbatim ik — Q4_K/Q5_K/Q6_K int8 GEMM |
| Activation quantizer | `ik_q8_quantize.cpp` | verbatim ik `quantize_row_q8_2_x4` (the x86 activation for ALL wired families) |
| Weight ref (de)quant | `ik_qk_quantize.cpp` | verbatim ik K-quant ref (de)quant + helpers (test fixtures) |
| Weight ref (de)quant | `ik_gguf_gemm.cpp` (anon ns) | verbatim ik Q8_0/Q5_0 ref (de)quant |
| ggml glue | `ik_ggml_support.cpp` | `ggml_table_f32_f16` + `ggml_abort` + `ggml_bf16_to_fp32` |
| Router | `ik_iqk_dispatch.cpp` | `ls_iqk_mul_mat` → `MulMat::prepare` tries `iqk_set_kernels_legacy_quants` then `iqk_set_kernels_kquants` |
| Bridge (public) | `ik_gguf_gemm.{h,cpp}` | `GgufType` enum + type/byte maps + `quantize_activations` + `gguf_gemm_one` |
| Engine enum | `src/core/expert_device.h` | `GgufQuantType` + `GgufGroupedGemmParams` (the ExpertDevice contract) |
| Device dispatch | `src/compute/cpu/{numa,multi_numa}_cpu_expert_device.cpp` | `GgufQuantType` → `ik::GgufType` switch |
| Build | `deps/LayerStoRmCpuExpertKernels/CMakeLists.txt` | `CPU_EXPERT_IK_SOURCES` + `LS_IK_HAVE_KQUANTS` gate |
| Tests | `tests/unit/{gguf_gemm_test,quant_comparison_test}.cpp` | accuracy vs FP32-of-dequant + head-to-head |

**ACTIVATION FORMAT (x86_64) — read this first.** Every currently-wired weight
(legacy + plain K-quant) consumes **`GGML_TYPE_Q8_2_X4`** activations, NOT Q8_K.
The activation type is declared by each family's `iqk_set_kernels_*` via
`expected_type_B` (`iqk_gemm_legacy_quants.cpp:2092`, `iqk_gemm_kquants.cpp:2681`
— the non-`_R4`/`_R8` branch). `HAVE_FANCY_SIMD` only switches the int8 dot
instruction, not the activation layout. Q8_K is the activation for the `_R4`/`_R8`
repacked GEMMs only, which we do NOT expose. (aarch64 differs — Q8_0_X4/Q8_1_X4,
unvendored: TD-IK-AARCH64-ACTQUANT.) Always re-read `expected_type_B` for the new
family rather than assuming.

## Recipe — add GGUF family `X`

1. **Find the ik GEMM family** for `X` in `ref/ik_llama.cpp/ggml/src/iqk/`:
   `iqk_gemm_iquants.cpp` (IQ2/IQ3/IQ4…), `iqk_gemm_iqk_quants.cpp` (IQK),
   `iqk_gemm_ktquants.cpp` (KT/trellis), `iqk_gemm_1bit.cpp` (1.5/2-bit ternary).
   Note its `iqk_set_kernels_<family>()` and read `X`'s `expected_type_B`
   (the activation format) + `GGML_TYPE_<X>`.
2. **Vendor the family** `iqk_gemm_<family>.{cpp,h}` BYTE-FOR-BYTE into this dir
   with the standard per-file MIT header (copy an existing one). Add the `.cpp` to
   `CPU_EXPERT_IK_SOURCES` in the dep `CMakeLists.txt` (gate behind a
   `LS_IK_HAVE_<FAMILY>` `if(EXISTS …)` like `iqk_gemm_kquants.cpp` if optional).
3. **Activation quantizer**: if `expected_type_B` is `Q8_2_X4` it's already vendored
   (`ik_q8_quantize.cpp`); else vendor `quantize_row_<acttype>` verbatim and wire it
   in `quantize_activations()`.
4. **Weight ref (de)quant** for `X` (`quantize_row_<X>_ref` / `dequantize_row_<X>`):
   vendor verbatim into `ik_qk_quantize.cpp` (or alongside Q8_0 in `ik_gguf_gemm.cpp`)
   — needed by the tests + the FP32-reference fallback, NOT the hot path.
5. **Router** (`ik_iqk_dispatch.cpp`, `MulMat::prepare`): add
   `if (iqk_set_kernels_<family>(ne00, typeA, typeB, mm.funcs, mm.func16)) return true;`
   and `#include "iqk_gemm_<family>.h"`.
6. **Bridge** (`ik_gguf_gemm.{h,cpp}`): add `GgufType::x`; wire `gguf_supported`,
   `weight_ggml_type`, `weight_row_bytes`, `activation_ggml_type` (return the
   `expected_type_B` you read in step 1), `quantize_weight`/`dequantize_weight`.
7. **Engine contract**: add `GgufQuantType::x` (`src/core/expert_device.h`) and the
   `GgufQuantType → ik::GgufType` case in BOTH `numa_cpu_expert_device.cpp` and
   `multi_numa_cpu_expert_device.cpp` (`to_ik_gguf` / the inline switch).
8. **Tests**: `GgufGemm.X_MultiExpert` + `X_Supported` (mirror Q4_K), and an `X` row
   in `QuantComparison.Nvfp4VsGgufHeadToHead`. Tolerances scale with bit-width.
9. **Build + verify**: `cmake --build build --target layerstorm_unit_tests`;
   `GgufGemm.*`/`QuantComparison.*` green; `layerstorm_no_cuda_check` green
   (the family TU must be CPU-only / CUDA-free, INV-GPU-1).

## Invariants to keep
- ik files BYTE-FOR-BYTE verbatim (MIT header + `LICENSE.ik`); LayerStoRm glue
  (`ik_gguf_gemm.*`, `ik_iqk_dispatch.cpp`) is clearly marked non-verbatim.
- CPU-only / CUDA-free; built for the host ISA via `LAYERSTORM_ARCH` (one target,
  by design — see the DECISION note in `spec/TECH_DEBT.md`).

# CMakeLists.txt — LayerStoRmCpuExpertKernels
#
# CPU-only expert FFN compute kernels for LayerStoRm: the NVFP4 grouped GEMM,
# the SwiGLU + permute/unpermute MoE kernels, the threaded GGUF (ik_llama) GEMM
# driver + the verbatim ik_vendor quant kernels, and the NUMA thread pool /
# ik barrier threading infrastructure those kernels fork-join over.
#
# THIS PACKAGE IS THE SOLE POINT OF CPU-ARCH DIFFERENTIATION (INV-CPU-KERN-ARCH).
# All __AVX512F__ / __AVX2__ / VNNI intrinsic code and all arch compile-flags
# live here. The engine's CPU expert *devices* (NumaCpuExpertDevice,
# MultiNumaCpuExpertDevice) stay in the engine and are arch-agnostic: they
# include this package's public headers and call its kernel functions.
#
# Provides the `layerstorm_cpu_expert_kernels` OBJECT library for consumption
# by LayerStoRm3, mirroring the LayerStoRmGemmKernels add_subdirectory pattern:
#
#   set(CPU_EXPERT_DIR "${CMAKE_SOURCE_DIR}/deps/LayerStoRmCpuExpertKernels")
#   add_subdirectory(${CPU_EXPERT_DIR} ${CMAKE_BINARY_DIR}/cpu_expert_kernels)
#
#   target_include_directories(my_target PRIVATE ${CPU_EXPERT_DIR}/csrc)
#   target_link_libraries(my_target PUBLIC $<TARGET_OBJECTS:layerstorm_cpu_expert_kernels>)
#
# Headers are self-contained under csrc/ with the engine-relative layout
# (e.g. include "compute/cpu/nvfp4_cpu_kernel.h"), so consumers add csrc/ to
# their include path and include only what they use.
#
# The single engine symbol the NVFP4 kernel references —
# layerstorm::model::fp8_e4m3::decode(uint8_t) — is declared by the
# self-contained header compute/cpu/fp8_e4m3_decode.h and DEFINED by the engine
# (src/model/quantization/fp8.cpp), resolved at the consumer's final link. This
# OBJECT library therefore builds standalone with an undefined-but-declared
# reference (no engine headers needed).
#
# CPU-only / CUDA-free (INV-GPU-1): nothing here includes <cuda_runtime.h>.

cmake_minimum_required(VERSION 3.18)
project(LayerStoRmCpuExpertKernels LANGUAGES CXX)

# ── Arch ownership ───────────────────────────────────────────────────────────
# This package is the ONLY place CPU archs are differentiated. The kernel TUs'
# hand-written AVX-512 / AVX2 / VNNI paths are #if'd on __AVX512F__ / __AVX2__,
# which the compiler only defines when handed the matching -m flags. The ik
# vendored TUs gate their in-file SIMD branches the same way.
#
# LAYERSTORMCPUEXPERTKERNELS_ARCH selects the ISA target for ALL kernel TUs:
#   native   (default) → -march=native   (this host's full ISA; AVX-512/VNNI on
#                                          capable hosts — matches the engine's
#                                          LAYERSTORM_FAST_CPU_FLAGS default)
#   portable           → -O3 only        (baseline x86-64; scalar fallbacks run)
#   <other>            → -march=<other>   (an explicit ISA, e.g. x86-64-v3, or a
#                                          flag list — passed through verbatim)
#
# The default INHERITS the engine-wide LAYERSTORM_ARCH (set by the top-level
# LayerStoRm CMakeLists, shared with the gpu_loader solver + other hot paths) when
# present; falls back to "native" on a standalone build with no global set. An
# explicit -DLAYERSTORMCPUEXPERTKERNELS_ARCH=... still overrides (cache var wins).
# Building for ONE target is deliberate (not debt); runtime multi-ISA dispatch is
# out of scope by design.
if(DEFINED LAYERSTORM_ARCH)
  set(_cpukern_arch_default "${LAYERSTORM_ARCH}")
else()
  set(_cpukern_arch_default "native")
endif()
set(LAYERSTORMCPUEXPERTKERNELS_ARCH "${_cpukern_arch_default}" CACHE STRING
    "CPU ISA target for the CPU expert kernel TUs: native | portable | <flag> (inherits LAYERSTORM_ARCH)")

if(LAYERSTORMCPUEXPERTKERNELS_ARCH STREQUAL "native")
  set(CPU_EXPERT_ARCH_FLAGS -march=native)
elseif(LAYERSTORMCPUEXPERTKERNELS_ARCH STREQUAL "portable")
  set(CPU_EXPERT_ARCH_FLAGS)        # -O3 only; no -march, scalar fallbacks
else()
  # Pass-through: an explicit -march value (e.g. "x86-64-v3") or flag.
  set(CPU_EXPERT_ARCH_FLAGS -march=${LAYERSTORMCPUEXPERTKERNELS_ARCH})
endif()

# Tuned opt level shared by every kernel TU (the engine forced -O3 on these).
set(CPU_EXPERT_OPT_FLAGS -O3)

set(CPU_EXPERT_CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(CPU_EXPERT_CPU_DIR  "${CPU_EXPERT_CSRC_DIR}/compute/cpu")
set(CPU_EXPERT_IK_DIR   "${CPU_EXPERT_CPU_DIR}/ik_vendor")

# ── Source files ─────────────────────────────────────────────────────────────
# Pure compute kernels + their threading infra (NUMA pool / ik barrier).
set(CPU_EXPERT_KERNEL_SOURCES
  ${CPU_EXPERT_CPU_DIR}/nvfp4_cpu_kernel.cpp     # NVFP4 grouped GEMM (decode + GEMV)
  ${CPU_EXPERT_CPU_DIR}/cpu_moe_kernels.cpp      # SwiGLU + permute/unpermute
  ${CPU_EXPERT_CPU_DIR}/cpu_gguf_gemm.cpp        # threaded GGUF GEMM driver glue
  ${CPU_EXPERT_CPU_DIR}/gguf_lossless.cpp        # bit-compat Q4_K x Q8_1 (GPU-mmvq parity)
  ${CPU_EXPERT_CPU_DIR}/numa_thread_pool.cpp     # fork-join NUMA pool (sysfs/pthread)
)

# Vendored ik_llama CPU GGUF GEMM TUs (kept BYTE-FOR-BYTE verbatim, MIT — see
# csrc/compute/cpu/ik_vendor/LICENSE.ik). Their in-file AVX2/AVX512/VNNI/BF16
# #ifdef branches are built for the host arch via CPU_EXPERT_ARCH_FLAGS.
set(CPU_EXPERT_IK_SOURCES
  ${CPU_EXPERT_IK_DIR}/iqk_gemm_legacy_quants.cpp  # Q8_0/Q4_0/Q5_0 (verbatim ik)
  ${CPU_EXPERT_IK_DIR}/ik_q8_quantize.cpp          # Q8_x act quant (verbatim ik extract)
  ${CPU_EXPERT_IK_DIR}/ik_ggml_support.cpp         # fp16 table + ggml_abort + bf16
  ${CPU_EXPERT_IK_DIR}/ik_iqk_dispatch.cpp         # thin MulMat driver/router
  ${CPU_EXPERT_IK_DIR}/ik_gguf_gemm.cpp            # LayerStoRm bridge
)
# Q4_K/Q6_K (PART 3): present => define LS_IK_HAVE_KQUANTS so the bridge wires
# q4_k / q6_k and the Q8_K activation quantizer compiles.
if(EXISTS "${CPU_EXPERT_IK_DIR}/iqk_gemm_kquants.cpp")
  list(APPEND CPU_EXPERT_IK_SOURCES
    ${CPU_EXPERT_IK_DIR}/iqk_gemm_kquants.cpp      # Q4_K/Q6_K (verbatim ik)
    ${CPU_EXPERT_IK_DIR}/ik_qk_quantize.cpp)       # Q8_K act quant + K-quant ref pack
  set(CPU_EXPERT_HAVE_KQUANTS ON)
endif()

set(CPU_EXPERT_ALL_SOURCES
  ${CPU_EXPERT_KERNEL_SOURCES}
  ${CPU_EXPERT_IK_SOURCES})

# ── OBJECT library ───────────────────────────────────────────────────────────
add_library(layerstorm_cpu_expert_kernels OBJECT ${CPU_EXPERT_ALL_SOURCES})

target_compile_features(layerstorm_cpu_expert_kernels PRIVATE cxx_std_17)

# Arch + opt flags applied to EVERY kernel TU (this is the arch-ownership seam).
target_compile_options(layerstorm_cpu_expert_kernels PRIVATE
  ${CPU_EXPERT_OPT_FLAGS} ${CPU_EXPERT_ARCH_FLAGS})

# ik vendored TUs pull ggml-common.h, which only includes <cuda_fp16.h> under
# GGML_COMMON_DECL_CUDA (never defined); GGML_COMMON_IMPL_CPP picks the C++ impl.
set_source_files_properties(${CPU_EXPERT_IK_SOURCES} PROPERTIES
  COMPILE_DEFINITIONS "GGML_COMMON_IMPL_CPP")

# GPU-mmvq bit-parity kernel: pin FP contraction OFF so the scalar reduction path
# uses SEPARATE rounded mul+add (matching the explicit AVX512 intrinsics) — with
# the default -ffp-contract=fast GCC would fuse the scalar A*dotQ + B*sumqx into an
# FMA, diverging from the vectorized path (VNNI-vs-scalar bit-identity test). The
# reduction ORDER (the GPU 32-lane warp tree) is reproduced explicitly in-code.
set_source_files_properties(${CPU_EXPERT_CPU_DIR}/gguf_lossless.cpp PROPERTIES
  COMPILE_OPTIONS "-ffp-contract=off")

# Public include root: consumers (engine devices, tests) include
# "compute/cpu/<header>.h" off csrc/. PUBLIC so $<TARGET_OBJECTS> consumers and
# add_subdirectory parents inherit it.
target_include_directories(layerstorm_cpu_expert_kernels PUBLIC
  ${CPU_EXPERT_CSRC_DIR})

# LS_IK_HAVE_KQUANTS must reach consumers too (the bridge header gates q4_k/q6_k
# enum wiring on it), so make it PUBLIC.
if(CPU_EXPERT_HAVE_KQUANTS)
  target_compile_definitions(layerstorm_cpu_expert_kernels PUBLIC LS_IK_HAVE_KQUANTS)
endif()

# Threading (pthread); the NUMA pool uses pthread affinity + sysfs (NO libnuma).
find_package(Threads REQUIRED)
target_link_libraries(layerstorm_cpu_expert_kernels PUBLIC Threads::Threads)

# ── Export variables for parent projects ─────────────────────────────────────
# Only when consumed via add_subdirectory (a parent scope exists); skipped on a
# standalone top-level build.
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(CPU_EXPERT_KERNELS_INCLUDE_DIR  ${CPU_EXPERT_CSRC_DIR}     PARENT_SCOPE)
  set(CPU_EXPERT_KERNELS_ALL_SOURCES  ${CPU_EXPERT_ALL_SOURCES} PARENT_SCOPE)
  set(CPU_EXPERT_KERNELS_IK_SOURCES   ${CPU_EXPERT_IK_SOURCES}  PARENT_SCOPE)
  set(CPU_EXPERT_KERNELS_HAVE_KQUANTS ${CPU_EXPERT_HAVE_KQUANTS} PARENT_SCOPE)
  # The exact arch/opt flag lists, so the parent can mirror them on test TUs that
  # inline the kernel (e.g. the multi-arch NVFP4 probe) or on no-CUDA checks.
  set(CPU_EXPERT_KERNELS_ARCH_FLAGS   ${CPU_EXPERT_ARCH_FLAGS}  PARENT_SCOPE)
  set(CPU_EXPERT_KERNELS_OPT_FLAGS    ${CPU_EXPERT_OPT_FLAGS}   PARENT_SCOPE)
endif()

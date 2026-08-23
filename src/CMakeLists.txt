# ── SnapMLA kernels (OBJECT library) ──────────────────────────────────────────
# SM120 attention kernels from deps/LayerStoRmKernels.
# Decode + prefill instantiations (SM120) and arch-generic smxx utilities.

set(SNAPMLA_DIR "${CMAKE_SOURCE_DIR}/deps/LayerStoRmKernels/csrc")

set(SNAPMLA_SOURCES
  # sm120 decode (dense + sparse FP8)
  ${SNAPMLA_DIR}/sm120/decode/dense_fp8/instantiations/model1_persistent_h64.cu
  ${SNAPMLA_DIR}/sm120/decode/dense_fp8/instantiations/v32_persistent_h64.cu
  ${SNAPMLA_DIR}/sm120/decode/sparse_fp8/instantiations/model1_persistent_h64.cu
  ${SNAPMLA_DIR}/sm120/decode/sparse_fp8/instantiations/v32_persistent_h64.cu

  # sm120 prefill (dense + sparse absorbed BF16)
  ${SNAPMLA_DIR}/sm120/prefill/dense/fwd/head64/instantiations/phase1_k576.cu
  ${SNAPMLA_DIR}/sm120/prefill/sparse/fwd/head64/instantiations/phase1_k512.cu
  ${SNAPMLA_DIR}/sm120/prefill/sparse/fwd/head64/instantiations/phase1_k576.cu
  # fwd_for_small_topk + csa_fp8 prefill: not on the engine dispatch path
  # (regular dense/sparse decode + prefill cover production), but built so the
  # LSE-unit contract is testable (KVS-1 / TD-LSE-UNITS-DECODE; see
  # tests/unit/prefill_lse_units_test.cu). The former params conflict is
  # resolved: decode mode now takes the SnapMLA SparseAttnDecodeParams
  # (SmallTopkArgT unification in fwd_for_small_topk/head64/phase1.h).
  ${SNAPMLA_DIR}/sm120/prefill/sparse/fwd_for_small_topk/head64/instantiations/phase1_prefill_k512.cu
  ${SNAPMLA_DIR}/sm120/prefill/sparse/fwd_for_small_topk/head64/instantiations/phase1_decode_k512.cu
  ${SNAPMLA_DIR}/sm120/prefill/csa_fp8/instantiations/phase1_k576.cu

  # V4-5a: CSA FP8 decode (the one kernel behind CSA sparse / HCA dense /
  # SWA-only V4 decode+prefill; CsaHcaSm120AttentionDevice drives it).
  ${SNAPMLA_DIR}/sm120/decode/csa_fp8/instantiations/v4_h64.cu
  ${SNAPMLA_DIR}/sm120/decode/csa_fp8/instantiations/v4_h128.cu

  # smxx (arch-generic: combine, metadata, DCP LSE correction)
  ${SNAPMLA_DIR}/smxx/mla_combine.cu
  ${SNAPMLA_DIR}/smxx/get_mla_metadata.cu
  ${SNAPMLA_DIR}/smxx/dcp_lse_correct.cu
)

add_library(snapmla_kernels OBJECT ${SNAPMLA_SOURCES})

set_target_properties(snapmla_kernels PROPERTIES CUDA_ARCHITECTURES OFF)

target_compile_options(snapmla_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -gencode=arch=compute_120,code=sm_120
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    --ptxas-options=-v,--register-usage-level=10
  >
)

target_include_directories(snapmla_kernels PRIVATE
  ${SNAPMLA_DIR}
  # GG-7: q_absorb.cu's GGUF W_UK branch includes formats/gguf_dequant_one.h
  # (+ the per-type k-quant format headers) from the GemmKernels package.
  ${CMAKE_SOURCE_DIR}/deps/LayerStoRmGemmKernels/csrc
  ${cutlass_SOURCE_DIR}/include
  ${cutlass_SOURCE_DIR}/tools/util/include
)

target_link_libraries(snapmla_kernels PRIVATE CUDA::cudart)

# ── TQ MLA kernels (OBJECT library) ──────────────────────────────────────────
# SM120 TQ decode kernels from deps/LayerStoRmKernels.
# Prep + graph kernels are included via single-TU driver in layerstorm_cuda_kernels.

set(TQ_MLA_SOURCES
  ${SNAPMLA_DIR}/sm120/decode/tq_dense/splitkv_mla.cu
  ${SNAPMLA_DIR}/sm120/decode/tq_sparse/splitkv_mla.cu
  # csa_tq: V4 CSA TQ sparse decode. Not on the engine dispatch path, but built
  # so the split-KV LSE-unit contract is testable (tests/unit/tq_lse_units_test.cu).
  ${SNAPMLA_DIR}/sm120/decode/csa_tq/splitkv_csa_tq.cu
)

add_library(tq_mla_kernels OBJECT ${TQ_MLA_SOURCES})

set_target_properties(tq_mla_kernels PROPERTIES CUDA_ARCHITECTURES OFF)

target_compile_options(tq_mla_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -gencode=arch=compute_120,code=sm_120
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    --ptxas-options=-v,--register-usage-level=10
  >
)

target_include_directories(tq_mla_kernels PRIVATE
  ${SNAPMLA_DIR}
  ${cutlass_SOURCE_DIR}/include
  ${cutlass_SOURCE_DIR}/tools/util/include
)

target_link_libraries(tq_mla_kernels PRIVATE CUDA::cudart)

# ── GEMM kernels (OBJECT library, from deps/LayerStoRmGemmKernels) ───────────
set(GEMM_DIR "${CMAKE_SOURCE_DIR}/deps/LayerStoRmGemmKernels")
add_subdirectory(${GEMM_DIR} ${CMAKE_BINARY_DIR}/gemm_kernels)

# ── Expert kernels (OBJECT library, from deps/LayerStoRmExpertKernels) ───────
set(EXPERT_DIR "${CMAKE_SOURCE_DIR}/deps/LayerStoRmExpertKernels/csrc")

set(EXPERT_KERNEL_SOURCES
  ${EXPERT_DIR}/sm120/gating/topk_gating.cu
  ${EXPERT_DIR}/smxx/activation/fused_swiglu.cu
  ${EXPERT_DIR}/smxx/permute/moe_permute.cu
  ${EXPERT_DIR}/smxx/gating/adaptive_gating.cu
)

add_library(expert_kernels OBJECT ${EXPERT_KERNEL_SOURCES})
set_target_properties(expert_kernels PROPERTIES CUDA_ARCHITECTURES "120f")
target_include_directories(expert_kernels PRIVATE
  ${EXPERT_DIR}
  ${GEMM_DIR}/csrc
  ${cutlass_SOURCE_DIR}/include
  ${cutlass_SOURCE_DIR}/tools/util/include
)
target_link_libraries(expert_kernels PRIVATE CUDA::cudart)

# ── LayerStoRm CUDA kernels (OBJECT library) ─────────────────────────────────
# Our own .cu files from compute/kernels/{sm120,smxx,*}/.
# Binding headers stay in domain dirs (norm/, etc.);
# implementations live under sm120/ (arch-specific) or smxx/ (generic).
# GEMM and expert kernels now come from submodules above.

file(GLOB_RECURSE LAYERSTORM_CUDA_SOURCES CONFIGURE_DEPENDS compute/kernels/**/*.cu)

# Only create the object library if there are sources (GLOB may be empty early on)
if(LAYERSTORM_CUDA_SOURCES)
  add_library(layerstorm_cuda_kernels OBJECT ${LAYERSTORM_CUDA_SOURCES})
  set_target_properties(layerstorm_cuda_kernels PROPERTIES CUDA_ARCHITECTURES "120f")
  target_include_directories(layerstorm_cuda_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/deps/LayerStoRmKernels/csrc
    ${GEMM_DIR}/csrc
    ${EXPERT_DIR}
  )
  target_link_libraries(layerstorm_cuda_kernels PRIVATE cutlass_headers CUDA::cudart)
else()
  # Placeholder until kernel sources are added
  add_library(layerstorm_cuda_kernels INTERFACE)
endif()

# ── Config codegen ───────────────────────────────────────────────────────────
# Runs tools/gen_config.mjs (Node.js) before compilation to produce:
#   src/config/config_parser.h + .cpp  (from schema.json)
#   config/config-example.json         (example with --example flag)
#
# The `generate_config` target can also be run manually:
#   cmake --build build --target generate_config

find_program(NODE_EXECUTABLE node REQUIRED
  DOC "Node.js interpreter for config code generation")

set(GEN_CONFIG_SCRIPT  "${CMAKE_SOURCE_DIR}/tools/gen_config.mjs")
set(GEN_CONFIG_SCHEMA  "${CMAKE_SOURCE_DIR}/config/schema.json")
set(GEN_CONFIG_HEADER  "${CMAKE_CURRENT_SOURCE_DIR}/config/config_parser.h")
set(GEN_CONFIG_SOURCE  "${CMAKE_CURRENT_SOURCE_DIR}/config/config_parser.cpp")
set(GEN_CONFIG_EXAMPLE "${CMAKE_SOURCE_DIR}/config/config-example.json")

add_custom_command(
  OUTPUT  "${GEN_CONFIG_HEADER}" "${GEN_CONFIG_SOURCE}"
  COMMAND "${NODE_EXECUTABLE}" "${GEN_CONFIG_SCRIPT}"
  DEPENDS "${GEN_CONFIG_SCHEMA}" "${GEN_CONFIG_SCRIPT}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Generating config_parser.h + config_parser.cpp from schema.json"
  VERBATIM
)

add_custom_command(
  OUTPUT  "${GEN_CONFIG_EXAMPLE}"
  COMMAND "${NODE_EXECUTABLE}" "${GEN_CONFIG_SCRIPT}" --example
  DEPENDS "${GEN_CONFIG_SCHEMA}" "${GEN_CONFIG_SCRIPT}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Generating config-example.json from schema.json"
  VERBATIM
)

add_custom_target(generate_config
  DEPENDS "${GEN_CONFIG_HEADER}" "${GEN_CONFIG_SOURCE}" "${GEN_CONFIG_EXAMPLE}"
  COMMENT "Run config codegen: parser (config_parser.h/.cpp) + example config"
)

# Mark generated files so CMake knows they won't exist until build time
set_source_files_properties(
  "${GEN_CONFIG_HEADER}" "${GEN_CONFIG_SOURCE}"
  PROPERTIES GENERATED TRUE
)

# ── layerstorm_core (STATIC library) ─────────────────────────────────────────
# NOTE: file(GLOB ... CONFIGURE_DEPENDS) per subdirectory. CONFIGURE_DEPENDS makes
# the build re-glob and auto-reconfigure when source files are added/removed, so a
# new .cpp is picked up without a manual `cmake -S . -B build` (avoids a confusing
# undefined-symbol link error when a new file isn't yet in the source list).

file(GLOB CORE_SOURCES CONFIGURE_DEPENDS core/*.cpp)
file(GLOB CORE_MEMORY_SOURCES CONFIGURE_DEPENDS core/memory/*.cpp)
file(GLOB CORE_TRANSFER_SOURCES CONFIGURE_DEPENDS core/transfer/*.cpp)
file(GLOB CORE_STATISTICS_SOURCES CONFIGURE_DEPENDS core/statistics/*.cpp)
file(GLOB CORE_ORCHESTRATOR_SOURCES CONFIGURE_DEPENDS core/orchestrator/*.cpp)
# core/gpu_loader: solver + constants + affinity/place objectives + the P-26
# policy-weights artifact loader (loader_policy.cpp). All CUDA-free, all picked
# up by the glob and covered by layerstorm_no_cuda_check (INV-GPU-1).
file(GLOB CORE_GPU_LOADER_SOURCES CONFIGURE_DEPENDS core/gpu_loader/*.cpp)
file(GLOB COMPUTE_SOURCES CONFIGURE_DEPENDS compute/*.cpp)
file(GLOB COMPUTE_GRAPHS_SOURCES CONFIGURE_DEPENDS compute/graphs/*.cpp)
# C-6 CPU expert kernels (NumaCpuExpertDevice + threadpool + nvfp4/moe kernels).
# CPU-only, GPU-SDK-free TUs — covered by layerstorm_no_cuda_check (they derive
# from LAYERSTORM_CORE_SOURCES and are NOT in CUDA_DESIGNATED_TUS).
file(GLOB COMPUTE_CPU_SOURCES CONFIGURE_DEPENDS compute/cpu/*.cpp)
file(GLOB PREFETCH_SOURCES CONFIGURE_DEPENDS prefetch/*.cpp)
file(GLOB SPECULATION_SOURCES CONFIGURE_DEPENDS speculation/*.cpp)
file(GLOB PARALLELISM_SOURCES CONFIGURE_DEPENDS parallelism/*.cpp)
file(GLOB MODEL_SOURCES CONFIGURE_DEPENDS model/*.cpp)
file(GLOB MODEL_QUANT_SOURCES CONFIGURE_DEPENDS model/quantization/*.cpp)
file(GLOB MODEL_WEIGHT_LOADER_SOURCES CONFIGURE_DEPENDS model/weight_loader/*.cpp)
file(GLOB MODEL_WEIGHT_PIPELINE_SOURCES CONFIGURE_DEPENDS model/weight_pipeline/*.cpp)
file(GLOB CONFIG_SOURCES CONFIGURE_DEPENDS config/*.cpp)
file(GLOB DAEMON_SOURCES CONFIGURE_DEPENDS daemon/*.cpp)
# Attention refactor V2: phase/arch-split attention TUs (driver + kv metadata
# + page provisioning + codec axis + per-model arch pipelines). CUDA-free
# (INV-GPU-1) and covered by layerstorm_no_cuda_check — EXCEPT combine.cpp
# (the DCP attention combine, a designated CUDA TU in CUDA_DESIGNATED_TUS).
file(GLOB DAEMON_ATTENTION_SOURCES CONFIGURE_DEPENDS daemon/attention/*.cpp)
# MoE by-model split (TD-MOE-BY-MODEL-SPLIT): job-specialty/arch-split MoE TUs
# (driver + rank/EP orchestration + progressive FETCH_AND_RUN machinery +
# chunked big-batch sibling + per-model arch hooks). All CUDA-free (INV-GPU-1)
# and covered by layerstorm_no_cuda_check.
file(GLOB DAEMON_MOE_SOURCES CONFIGURE_DEPENDS daemon/moe/*.cpp)
# engine.cpp pybind11 bindings extracted to python/bindings/engine_pybind.cpp;
# Engine class itself is now part of layerstorm_core.

# Ensure the generated config_parser.cpp is always present in the source list
# regardless of whether the glob found it (it won't on a fresh checkout).
list(APPEND CONFIG_SOURCES "${GEN_CONFIG_SOURCE}")
list(REMOVE_DUPLICATES CONFIG_SOURCES)

set(LAYERSTORM_CORE_SOURCES
  ${CORE_SOURCES}
  ${CORE_MEMORY_SOURCES}
  ${CORE_TRANSFER_SOURCES}
  ${CORE_STATISTICS_SOURCES}
  ${CORE_ORCHESTRATOR_SOURCES}
  ${CORE_GPU_LOADER_SOURCES}
  ${COMPUTE_SOURCES}
  ${COMPUTE_GRAPHS_SOURCES}
  ${COMPUTE_CPU_SOURCES}
  ${PREFETCH_SOURCES}
  ${SPECULATION_SOURCES}
  ${PARALLELISM_SOURCES}
  ${MODEL_SOURCES}
  ${MODEL_QUANT_SOURCES}
  ${MODEL_WEIGHT_LOADER_SOURCES}
  ${MODEL_WEIGHT_PIPELINE_SOURCES}
  ${CONFIG_SOURCES}
  ${DAEMON_SOURCES}
  ${DAEMON_ATTENTION_SOURCES}
  ${DAEMON_MOE_SOURCES}
)

if(LAYERSTORM_CORE_SOURCES)
  add_library(layerstorm_core STATIC ${LAYERSTORM_CORE_SOURCES})
else()
  # Bootstrap: no .cpp files exist yet. Create a stub static lib.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/stub.cpp" "// layerstorm_core stub\n")
  add_library(layerstorm_core STATIC "${CMAKE_CURRENT_BINARY_DIR}/stub.cpp")
endif()

# ── Fast CPU paths ───────────────────────────────────────────────────────────
# Force tuned flags on latency-critical host "kernels" regardless of the global
# build type (Debug / RelWithDebInfo), so the daemon hot path is never shipped at
# -O0/-O2. Per-source COMPILE_OPTIONS append after the build-type flags, so these
# trailing flags win (the -O here overrides the build-type -O; -g is preserved).
#
# LAYERSTORM_FAST_CPU_FLAGS is a ';'-separated list, overridable at configure time,
# e.g. -DLAYERSTORM_FAST_CPU_FLAGS="-O2" or "-O3;-march=native;-funroll-loops".
# (Shared with the perf-sensitive unit tests in tests/unit/CMakeLists.txt.)
# Force -O3 + the arch flag on latency-critical host kernels regardless of the
# global build type. The arch flag is derived from the engine-wide LAYERSTORM_ARCH
# (native → -march=native; portable → none; <flag> → -march=<flag>), so one knob
# controls every CPU hot path. -march=native makes O3's vectorization a WIN on the
# small device/bank reductions (AVX-512 masked ops vs baseline-SSE remainder
# overhead; measured ~5.3 vs ~5.7 us/solve). Kept LOCAL to the hot kernels on
# purpose: a build-wide -march was measurably SLOWER (likely AVX-512 frequency-
# license downclocking when 512-bit ops are sprinkled binary-wide) — concentrating
# it to the tiny hot TU keeps 512-bit usage contained.
# An explicit -DLAYERSTORM_FAST_CPU_FLAGS=... still overrides (cache var wins).
set(_ls_fast_flags "-O3")
if(LAYERSTORM_ARCH STREQUAL "native")
  list(APPEND _ls_fast_flags "-march=native")
elseif(NOT LAYERSTORM_ARCH STREQUAL "portable")
  list(APPEND _ls_fast_flags "-march=${LAYERSTORM_ARCH}")
endif()
set(LAYERSTORM_FAST_CPU_FLAGS "${_ls_fast_flags}" CACHE STRING
    "Compile flags forced on latency-critical host kernels (LAYERSTORM_FAST_CPU_SOURCES); default derives from LAYERSTORM_ARCH")
set(LAYERSTORM_FAST_CPU_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/core/gpu_loader/loader_solver.cpp  # I8 loader solver
  ${CMAKE_CURRENT_SOURCE_DIR}/core/gpu_loader/loader_evict_scores.cpp # I8 evict-score board (hot path)
  ${CMAKE_CURRENT_SOURCE_DIR}/core/gpu_loader/loader_affinity_place.cpp # P-25 affinity-expressing objective
  ${CMAKE_CURRENT_SOURCE_DIR}/core/gpu_loader/loader_place_hotness.cpp # TASK-A2 hotness-steered place_cons
  ${CMAKE_CURRENT_SOURCE_DIR}/core/gpu_loader/loader_place_sum.cpp    # residency-reframe weighted-sum place_cons
  ${CMAKE_CURRENT_SOURCE_DIR}/core/gpu_loader/loader_place_cons.cpp   # I8 place_cons table (hot path)
  # CPU expert COMPUTE KERNELS (NVFP4 GEMM, SwiGLU/permute, GGUF GEMM, NUMA pool,
  # ik_vendor) moved to deps/LayerStoRmCpuExpertKernels — that package now OWNS
  # all CPU-arch differentiation and applies -O3 + the arch flags itself
  # (LAYERSTORMCPUEXPERTKERNELS_ARCH, default native). See add_subdirectory below.
  # The CPU expert DEVICES stay here and are arch-agnostic (only _mm_pause,
  # SSE2 baseline) — they keep -O3 for the orchestration loop. -march on them is
  # a behavioral no-op (no AVX intrinsics) but harmless; kept for parity.
  ${CMAKE_CURRENT_SOURCE_DIR}/compute/cpu/numa_cpu_expert_device.cpp # device orchestration
  ${CMAKE_CURRENT_SOURCE_DIR}/compute/cpu/multi_numa_cpu_expert_device.cpp # multi-NUMA device
)
if(LAYERSTORM_FAST_CPU_SOURCES)
  set_source_files_properties(${LAYERSTORM_FAST_CPU_SOURCES}
    PROPERTIES COMPILE_OPTIONS "${LAYERSTORM_FAST_CPU_FLAGS}")
endif()

# ── CPU expert kernels (OBJECT library, from deps/LayerStoRmCpuExpertKernels) ─
# Self-contained CPU-only package owning ALL CPU-arch differentiation: the NVFP4
# grouped GEMM, SwiGLU/permute MoE kernels, the threaded GGUF GEMM driver + the
# verbatim ik_vendor quant kernels, and the NUMA thread pool / ik barrier. It
# applies -O3 + the arch flags itself (LAYERSTORMCPUEXPERTKERNELS_ARCH default
# native), so the engine just links it. Provides `layerstorm_cpu_expert_kernels`
# (OBJECT) and PUBLIC include dir csrc/. Mirrors the LayerStoRmGemmKernels
# add_subdirectory consumption pattern.
# NO CUDA in any of it (INV-GPU-1, INV-CPU-KERN-ARCH).
set(CPU_EXPERT_DIR "${CMAKE_SOURCE_DIR}/deps/LayerStoRmCpuExpertKernels")
add_subdirectory(${CPU_EXPERT_DIR} ${CMAKE_BINARY_DIR}/cpu_expert_kernels)
# CPU_EXPERT_KERNELS_* vars exported to PARENT_SCOPE by the subdir CMakeLists:
#   CPU_EXPERT_KERNELS_INCLUDE_DIR, _ALL_SOURCES, _IK_SOURCES, _HAVE_KQUANTS,
#   _ARCH_FLAGS, _OPT_FLAGS.

# Codegen must complete before any source in layerstorm_core is compiled
add_dependencies(layerstorm_core generate_config)
target_link_libraries(layerstorm_core PUBLIC $<TARGET_OBJECTS:layerstorm_cpu_expert_kernels>)
# Public headers (ik_gguf_gemm.h, nvfp4_cpu_kernel.h, …) are included from the
# compute/cpu device TUs off the dep's csrc/ root.
target_include_directories(layerstorm_core PUBLIC
  ${CPU_EXPERT_KERNELS_INCLUDE_DIR})
# Object-lib usage requirements don't flow through $<TARGET_OBJECTS>; propagate
# LS_IK_HAVE_KQUANTS to core consumers (tests) so the GGUF Q4_K/Q6_K paths build.
if(CPU_EXPERT_KERNELS_HAVE_KQUANTS)
  target_compile_definitions(layerstorm_core PUBLIC LS_IK_HAVE_KQUANTS)
endif()

# Preset directory compile-time path
target_compile_definitions(layerstorm_core PRIVATE
  LAYERSTORM_PRESET_DIR="${CMAKE_SOURCE_DIR}/config/presets"
  LAYERSTORM_TQ_CODEBOOK_DIR="${CMAKE_SOURCE_DIR}/config/tq_codebooks"
)

# Validate preset files against schema at build time
file(GLOB PRESET_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/config/presets/*.json")
if(PRESET_FILES)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/presets_validated.stamp"
    COMMAND "${NODE_EXECUTABLE}" "${GEN_CONFIG_SCRIPT}" --validate-presets
    COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/presets_validated.stamp"
    DEPENDS ${PRESET_FILES} "${GEN_CONFIG_SCHEMA}" "${GEN_CONFIG_SCRIPT}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Validating preset files against schema"
    VERBATIM
  )
  add_custom_target(validate_presets
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/presets_validated.stamp"
  )
  add_dependencies(layerstorm_core validate_presets)
endif()

target_include_directories(layerstorm_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/deps/LayerStoRmKernels/csrc
  ${CMAKE_SOURCE_DIR}/deps/LayerStoRmGemmKernels/csrc
  ${CMAKE_SOURCE_DIR}/deps/LayerStoRmExpertKernels/csrc
)

# Link OBJECT libraries
target_link_libraries(layerstorm_core PUBLIC
  $<TARGET_OBJECTS:snapmla_kernels>
  $<TARGET_OBJECTS:tq_mla_kernels>
  $<TARGET_OBJECTS:gemm_kernels>
  $<TARGET_OBJECTS:expert_kernels>
)
if(TARGET layerstorm_cuda_kernels AND NOT layerstorm_cuda_kernels_is_interface)
  get_target_property(_lck_type layerstorm_cuda_kernels TYPE)
  if(_lck_type STREQUAL "OBJECT_LIBRARY")
    target_link_libraries(layerstorm_core PUBLIC
      $<TARGET_OBJECTS:layerstorm_cuda_kernels>
    )
  endif()
endif()

# Link dependencies
target_link_libraries(layerstorm_core PUBLIC
  CUDA::cudart
  CUDA::cuda_driver
  NCCL::NCCL
  nlohmann_json::nlohmann_json
  spdlog::spdlog
  Threads::Threads
  cutlass_headers
)

if(LAYERSTORM_USE_NUMA AND Numa_FOUND)
  target_link_libraries(layerstorm_core PUBLIC Numa::Numa)
endif()

if(LAYERSTORM_USE_NVME AND Uring_FOUND)
  target_link_libraries(layerstorm_core PUBLIC Uring::Uring)
endif()

# ── No-CUDA build enforcement (#86e, INV-GPU-1) ──────────────────────────────
# Compile-only target verifying non-kernel TUs build without CUDA SDK headers.
# Designated CUDA TUs (attention device impls, DCP allreduce graph,
# DCP attention wrapper, NCCL backend) are excluded — they legitimately cast
# void* to cudaStream_t or call NCCL/CUDA graph APIs.
#
# If any listed .cpp file transitively includes <cuda_runtime.h>, compilation
# fails because CUDA include paths are omitted from this target.
#
# Run manually:  cmake --build build --target layerstorm_no_cuda_check

set(CUDA_DESIGNATED_TUS
  ${CMAKE_CURRENT_SOURCE_DIR}/compute/snapmla_sm120_attention_device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compute/tq_sm120_attention_device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compute/csa_hca_sm120_attention_device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/daemon/attention/combine.cpp  # DCP attention combine (moved from compute/dcp_attention_wrapper.cpp)
  ${CMAKE_CURRENT_SOURCE_DIR}/compute/graphs/dcp_allreduce_graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compute/graphs/nccl_group_graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compute/graphs/routed_ffn_graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parallelism/nccl_collective_backend.cpp
)

set(NO_CUDA_CHECK_SOURCES ${LAYERSTORM_CORE_SOURCES})
list(REMOVE_ITEM NO_CUDA_CHECK_SOURCES ${CUDA_DESIGNATED_TUS})
# The CPU expert kernel TUs (NVFP4 GEMM, SwiGLU/permute, GGUF GEMM, NUMA pool +
# verbatim ik_vendor) now live in deps/LayerStoRmCpuExpertKernels. They are
# CPU-only and must build with NO CUDA SDK on the include path (INV-GPU-1).
# Compiling the dep's sources here (against the dep's csrc/ include root) proves
# it. The dep itself already enforces CPU-only via its own object lib; this is
# the redundant no-CUDA-include-path verification.
list(APPEND NO_CUDA_CHECK_SOURCES ${CPU_EXPERT_KERNELS_ALL_SOURCES})

add_library(layerstorm_no_cuda_check OBJECT EXCLUDE_FROM_ALL
  ${NO_CUDA_CHECK_SOURCES})
add_dependencies(layerstorm_no_cuda_check generate_config)

# All CPU expert kernel TUs need -O3 + the arch flags (their SIMD is gated on
# __AVX2__/__AVX512F__); the ik vendored subset additionally needs
# GGML_COMMON_IMPL_CPP. The dep owns these; we mirror its exported flag lists.
set_source_files_properties(${CPU_EXPERT_KERNELS_ALL_SOURCES} PROPERTIES
  COMPILE_OPTIONS "${CPU_EXPERT_KERNELS_OPT_FLAGS};${CPU_EXPERT_KERNELS_ARCH_FLAGS}")
set_source_files_properties(${CPU_EXPERT_KERNELS_IK_SOURCES} PROPERTIES
  COMPILE_DEFINITIONS "GGML_COMMON_IMPL_CPP")

target_include_directories(layerstorm_no_cuda_check PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CPU_EXPERT_KERNELS_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/deps/LayerStoRmKernels/csrc
  ${CMAKE_SOURCE_DIR}/deps/LayerStoRmGemmKernels/csrc
  ${CMAKE_SOURCE_DIR}/deps/LayerStoRmExpertKernels/csrc
)
if(CPU_EXPERT_KERNELS_HAVE_KQUANTS)
  target_compile_definitions(layerstorm_no_cuda_check PRIVATE LS_IK_HAVE_KQUANTS)
endif()

target_link_libraries(layerstorm_no_cuda_check PRIVATE
  nlohmann_json::nlohmann_json
  spdlog::spdlog
  Threads::Threads
  cutlass_headers
)

if(LAYERSTORM_USE_NUMA AND Numa_FOUND)
  target_link_libraries(layerstorm_no_cuda_check PRIVATE Numa::Numa)
endif()

if(LAYERSTORM_USE_NVME AND Uring_FOUND)
  target_link_libraries(layerstorm_no_cuda_check PRIVATE Uring::Uring)
endif()

target_compile_definitions(layerstorm_no_cuda_check PRIVATE
  LAYERSTORM_PRESET_DIR="${CMAKE_SOURCE_DIR}/config/presets"
  LAYERSTORM_TQ_CODEBOOK_DIR="${CMAKE_SOURCE_DIR}/config/tq_codebooks"
)

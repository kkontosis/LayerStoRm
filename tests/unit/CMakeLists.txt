# ── Unit tests ────────────────────────────────────────────────────────────────
# All *_test.cpp files compiled into a single executable.

file(GLOB UNIT_TEST_SOURCES CONFIGURE_DEPENDS *_test.cpp)
file(GLOB UNIT_TEST_CU_SOURCES CONFIGURE_DEPENDS *_test.cu)
list(APPEND UNIT_TEST_SOURCES ${UNIT_TEST_CU_SOURCES})

# PART-1 multi-arch NVFP4 probe TUs (one per arch tier). NOT *_test.cpp so they
# are compiled here explicitly with forced per-arch flags — this is how all three
# tiers (AVX-512 / AVX2 / scalar) get BUILT and compared on the same host.
set(NVFP4_ARCH_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4_arch_avx512.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4_arch_avx2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4_arch_scalar.cpp)
list(APPEND UNIT_TEST_SOURCES ${NVFP4_ARCH_SOURCES})

if(UNIT_TEST_SOURCES)
  add_executable(layerstorm_unit_tests ${UNIT_TEST_SOURCES})

  # Force the arch tier of each probe TU regardless of the host. -O3 throughout
  # so the bench reflects shipped speed. avx512 mirrors -march=native; avx2 kills
  # AVX-512 so __AVX512F__ is undefined; scalar kills both AVX2 and AVX-512.
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4_arch_avx512.cpp
    PROPERTIES COMPILE_OPTIONS "-O3;-march=native")
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4_arch_avx2.cpp
    PROPERTIES COMPILE_OPTIONS "-O3;-mavx2;-mfma;-mno-avx512f")
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4_arch_scalar.cpp
    PROPERTIES COMPILE_OPTIONS "-O3;-mno-avx2;-mno-avx512f")

  # Fast CPU path: force the same tuned flags as LAYERSTORM_FAST_CPU_SOURCES (src/)
  # on perf-sensitive tests so their benchmarks reflect the shipped kernel speed.
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_loader_solver_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/numa_cpu_expert_test.cpp
    PROPERTIES COMPILE_OPTIONS "${LAYERSTORM_FAST_CPU_FLAGS}")
  target_link_libraries(layerstorm_unit_tests PRIVATE
    layerstorm_core
    GTest::gtest_main
  )
  set_target_properties(layerstorm_unit_tests PROPERTIES CUDA_ARCHITECTURES "120f")
  target_compile_definitions(layerstorm_unit_tests PRIVATE
    LAYERSTORM_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    LAYERSTORM_RECORDING_BACKEND
  )
  include(GoogleTest)

  # GPU test suites: real NCCL multi-GPU + single-GPU CUDA kernels.
  # Serialized via RESOURCE_LOCK to prevent NCCL deadlocks from parallel
  # ctest launching multiple processes on the same GPU pair.
  # 30s timeout catches hangs quickly instead of blocking ctest.
  set(_GPU_FILTER
    "DcpCommunicatorMultiGpu.*"
    ":DcpExecutorMultiGpu.*"
    ":DcpAttentionWrapperMultiGpu.*"
    ":DcpAllreduceGraphMultiGpu.*"
    ":AdaptiveGatingTest.*"
    ":TopkGatingTest.*"
    ":RMSNormTest.*"
    ":EmbeddingTest.*"
    ":Fp8GemmTest.*"
    ":Nvfp4GemmTest.*"
    ":CudaSm120DeviceBackend.*"
    ":TqMlaAttentionPrep.*"
    ":TqMlaAttentionDecode.*"
    ":TqMlaAttentionCombine.*"
    ":TqChunkPrefillEquiv.*"
    ":SnapMlaSm120AttentionDevice.*"
    ":TqSm120AttentionDevice.*"
    ":CudaSm120ExpertDevice.*"
    ":GgufGroupedGemmGpu.*"
    ":GgufDenseGateUpSplitGpu.*"
    ":SamplingKernelTest.*"
    ":ConfidenceKernelTest.*"
    ":WeightFp8QuantTest.*"
    ":KvBvExtractDequantTest.*"
    ":KvBvExtractDequantGgufGpu.*"
    ":QAbsorbGgufGpu.*"
    ":PinnedExpertArenaTest.*"
    ":GpuLoaderCalibration.*"
  )
  string(CONCAT GPU_FILTER ${_GPU_FILTER})

  gtest_discover_tests(layerstorm_unit_tests
    TEST_SUFFIX "_gpu"
    TEST_FILTER "${GPU_FILTER}"
    PROPERTIES
      RESOURCE_LOCK "gpu"
      TIMEOUT 30
  )

  # Everything else: CPU-only tests, run in parallel, 60s timeout.
  gtest_discover_tests(layerstorm_unit_tests
    TEST_SUFFIX "_cpu"
    TEST_FILTER "-${GPU_FILTER}"
    PROPERTIES
      TIMEOUT 60
  )
endif()

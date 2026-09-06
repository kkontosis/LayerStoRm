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
    # P-29 step 13: per-row M-independence of the routed-expert grouped GGUF
    # GEMM (gguf_grouped_row_identity_test.cu) — the verify batch's bitwise
    # contract.
    ":GgufGroupedRowIdentity.*"
    # P-29 step 2: graph-node byte-mover kernels (graph_node_ops_test.cu).
    ":GraphNodeOps.*"
    # P-29 step 6: slim sparse-decode bit-identity (tq_sparse_decode_slim_test.cu).
    ":TqSparseDecodeSlim.*"
    # P-29 step 9: split-KV sparse-decode (tq_sparse_decode_splitkv_test.cu).
    ":TqSparseDecodeSplitKv.*"
    ":GgufDenseGateUpSplitGpu.*"
    ":SamplingKernelTest.*"
    ":ConfidenceKernelTest.*"
    ":WeightFp8QuantTest.*"
    # P-30 step 5: activation-quant rows > 65535 grid.y ceiling
    # (TD-MOE-BIG-CHUNK-ROWS-KERNEL-CEILING, quant_rows_grid_ceiling_test.cu).
    ":QuantRowsGridCeiling.*"
    ":KvBvExtractDequantTest.*"
    ":KvBvExtractDequantGgufGpu.*"
    ":QAbsorbGgufGpu.*"
    ":PinnedExpertArenaTest.*"
    ":GpuLoaderCalibration.*"
    # 2026-08-27 (TD-UNIT-SUITE-NO-GREEN-BASELINE): every suite containing
    # REQUIRES_GPU tests belongs here — suites below ran in the PARALLEL cpu
    # bucket while touching the real GPU, producing roaming -j flakes
    # (allocation failures under concurrent VRAM/stream pressure).
    ":KvRowCopyKernel.*"
    ":KvTieringManager.*"
    ":KvTieringManagerReplicated.*"
    ":KvTieringManagerSharded.*"
    ":CsaHcaDevice.*"
    ":DcpAttentionWrapper.*"
    ":DcpCommunicator.*"
    ":DcpExecutor.*"
    ":DcpLseCorrect.*"
    ":DsparkV4Kernel.*"
    ":ExpertDeviceTest.*"
    ":AttentionDeviceTest.*"
    ":FusedSwigluTest.*"
    ":ConfigValidatorGpu.*"
    ":ConfigResolverGpu.*"
    ":Fp8GroupedGemmTest.*"
    ":Nvfp4GroupedGemmTest.*"
    ":HashGatingTest.*"
    ":MhcKernel.*"
    ":MoePermuteTest.*"
    ":V4Oproj.*"
    ":V4Prep.*"
    ":Bf16GemmSmallM.*"
    ":DecodeLseUnits.*"
    ":IndexerChunkAppend.*"
    ":IndexerChunkTopk.*"
    ":IndexerHadamard.*"
    ":IndexerKpool.*"
    ":IndexerPagedScore.*"
    ":IndexerPrep.*"
    ":IndexerShardTranslate.*"
    # GF3.7 KDA (glm5_next) kernel suites — kda_kernels_test.cu; KdaDispatch is
    # CPU-only refusals, parallel bucket.
    ":KdaChunkCarry.*"
    ":KdaChunkScan.*"
    ":KdaConvKernel.*"
    ":KdaDecodeKernel.*"
    ":KdaGoldenReal.*"
    ":KdaGoldenSynth.*"
    ":KdaPrefillDecodeContinuity.*"
    ":LightningIndexerMqa.*"
    ":LightningTopkLongCtx.*"
    ":LightningTopkMerge.*"
    ":MlaAttention.*"
    ":MlaAttentionPrep.*"
    ":MoeBigChunkedIdentity.*"
    # GF3.4 NoPE (GLM5N, D_QK 512) MLA suites — nope_mla_attention_test.cu.
    # NopeMlaDispatch is CPU-only (geometry refusals throw before any launch)
    # and deliberately stays in the parallel bucket.
    ":NopeMlaDecode.*"
    ":NopeMlaPrefill.*"
    ":NopeMlaPrep.*"
    # TD-SNAPMLA-ATCTX-DECODE-PATH budget-staged sparse decode suites —
    # snapmla_budget_decode_test.cu (gather staging + hardened sparse FP8
    # split-KV decode at the GLM5N NoPE geometry).
    ":SnapmlaBudgetDecodeGather.*"
    ":SnapmlaBudgetDecodeSparseFp8.*"
    # P-32 stage 1 batched spec_verify sparse decode (s_q = R) row-isolation
    # suites — snapmla_sparse_verify_test.cu (one s_q=R launch vs R s_q=1
    # launches, byte for byte, at the production strides).
    ":SnapMlaSparseVerifyR3.*"
    ":SnapMlaSparseVerifyR6H32.*"
    ":PrefillDenseCausal.*"
    ":PrefillDenseLongCtx.*"
    ":PrefillLseUnits.*"
    ":PrefillSparseCausal.*"
    ":PrefillSparseDetReduce.*"
    ":TqLseUnits.*"
    ":TqPrefillDetReduce.*"
  )
  string(CONCAT GPU_FILTER ${_GPU_FILTER})

  gtest_discover_tests(layerstorm_unit_tests
    TEST_SUFFIX "_gpu"
    TEST_FILTER "${GPU_FILTER}"
    PROPERTIES
      RESOURCE_LOCK "gpu"
      TIMEOUT 30
  )

  # Suites that allocate large NUMA/pinned host arenas. They pass alone but
  # fail spuriously under `ctest -j` when run CONCURRENTLY with each other:
  # the box's HBM NUMA nodes are 16 GiB each, so parallel arena-sized
  # allocations hit node-local exhaustion while system RAM is plentiful
  # (TD-UNIT-SUITE-NO-GREEN-BASELINE, campaign dossier section 1). A shared
  # RESOURCE_LOCK serializes them among themselves; they still run in
  # parallel with the rest of the CPU bucket.
  set(_HOSTARENA_FILTER
    "ArenaMigratorTest.*"
    ":ArenaLoaderTest.*"
    ":ExpertPrepackerTest.*"
    ":PrepackedSourceTest.*"
    ":LivePrepackTest.*"
    ":ValidatePlan.*"
    ":MultiNumaCpuExpert.*"
  )
  string(CONCAT HOSTARENA_FILTER ${_HOSTARENA_FILTER})

  # 180s: MultiNumaCpuExpert.EngineDimsParityChurn takes ~53s ALONE — the
  # shared 60s ceiling made it a guaranteed flake the moment anything else
  # competed for memory bandwidth.
  # PROCESSORS 8: a big-memory test consumes every ctest -j8 slot, so it runs
  # with the machine to itself (ValidatePlan builds a ~22 GB synthetic model;
  # with the 504 GB arena holder resident the box has ~60 GB free — parallel
  # neighbours turn into bare OOM SIGKILLs, seen as "Subprocess killed").
  gtest_discover_tests(layerstorm_unit_tests
    TEST_SUFFIX "_cpu"
    TEST_FILTER "${HOSTARENA_FILTER}"
    PROPERTIES
      RESOURCE_LOCK "host_numa_arena"
      PROCESSORS 8
      TIMEOUT 180
  )

  # Opt-in real-hardware arms (spec/TESTING.md "Opt-in test arms"): they
  # GTEST_SKIP in milliseconds unless their env var is set, so they cost the
  # default suite nothing — but WHEN enabled they do the real thing (the loader
  # calibration probes every NUMA/HBM bank with a 32 GiB rotating pinned
  # footprint, for minutes). They therefore get the big-host-memory treatment
  # plus an hour-scale timeout; a 60 s ceiling would make the documented opt-in
  # invocation report a timeout instead of a result.
  set(_OPTIN_FILTER
    "GpuLoaderCalibrationSlow.*"
  )
  string(CONCAT OPTIN_FILTER ${_OPTIN_FILTER})

  gtest_discover_tests(layerstorm_unit_tests
    TEST_SUFFIX "_cpu"
    TEST_FILTER "${OPTIN_FILTER}"
    PROPERTIES
      RESOURCE_LOCK "host_numa_arena"
      PROCESSORS 8
      TIMEOUT 3600
  )

  # Everything else: CPU-only tests, run in parallel, 60s timeout.
  gtest_discover_tests(layerstorm_unit_tests
    TEST_SUFFIX "_cpu"
    TEST_FILTER "-${GPU_FILTER}:${HOSTARENA_FILTER}:${OPTIN_FILTER}"
    PROPERTIES
      TIMEOUT 60
  )
endif()

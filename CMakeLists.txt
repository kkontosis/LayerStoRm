cmake_minimum_required(VERSION 3.25)

# MUST be set BEFORE project()/CUDA enablement. project() initializes
# CMAKE_CUDA_ARCHITECTURES to CMake's auto-detected default (e.g. 75) and writes
# THAT to the cache when the variable is unset at that point. Setting it only
# afterwards (a normal variable) shadows it for targets but leaves a stale,
# misleading 75 in CMakeCache.txt, regenerated on every fresh configure / new
# worktree. Setting it here makes the cache authoritative — a fresh configure is
# 120f everywhere without needing -DCMAKE_CUDA_ARCHITECTURES. (RTX 5090 = SM_120.)
set(CMAKE_CUDA_ARCHITECTURES "120f")

project(layerstorm LANGUAGES CXX CUDA)

# ── Standards ─────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# When building Python modules (shared libs), all linked code needs -fPIC.
if(LAYERSTORM_BUILD_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# ── Options ───────────────────────────────────────────────────────────────────
option(LAYERSTORM_BUILD_TESTS      "Build unit/integration tests"   ON)
option(LAYERSTORM_BUILD_PYTHON     "Build pybind11 Python module"   ON)
option(LAYERSTORM_BUILD_BENCHMARKS "Build benchmark executables"    ON)
option(LAYERSTORM_USE_NUMA         "Enable libnuma support"         ON)
# TD-NUMA-HBM-BANKS (#111): detect CPU-less HBM/device-memory NUMA nodes
# (sysfs HMAT + libnuma) and expose them as loader-calibration banks with a
# footprint capped to the node's free memory. Detection is read-only sysfs —
# harmless — so it defaults ON; requires LAYERSTORM_USE_NUMA. OFF restores the
# DDR-only (CPU-backed nodes only) bank enumeration.
option(LAYERSTORM_DETECT_HBM       "Detect CPU-less HBM NUMA nodes as banks" ON)
option(LAYERSTORM_USE_NVME         "Enable io_uring NVMe tier"      ON)
# perf_trace fetch-path micro-timing. ON: compiled in, still runtime-gated by the
# LS_PERF_TRACE env var (zero cost unless enabled). OFF: compiled out entirely
# (record()/init()/dump_csv() become inline no-ops — no symbols, no static state,
# not even a branch) for production builds.
option(LAYERSTORM_PERF_TRACE       "Compile in perf_trace instrumentation"  ON)
# TD-101: AddressSanitizer build for host-heap triage (separate build dir;
# CUDA kernels stay uninstrumented). Run tests with
# ASAN_OPTIONS=protect_shadow_gap=0 (required with CUDA).
option(LAYERSTORM_ENABLE_ASAN      "Build host code with AddressSanitizer" OFF)
if(LAYERSTORM_ENABLE_ASAN)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>
                      $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
                      $<$<COMPILE_LANGUAGE:CXX>:-g>)
  add_link_options(-fsanitize=address)
endif()

# ── Global CPU ISA target for hot-path kernels ───────────────────────────────
# One engine-wide arch knob for every hand-vectorized CPU hot path (the CPU
# expert kernels in deps/LayerStoRmCpuExpertKernels, the gpu_loader solver, and
# any future hot path). Per-component arch options (e.g.
# LAYERSTORMCPUEXPERTKERNELS_ARCH, LAYERSTORM_FAST_CPU_FLAGS) INHERIT this value
# when not explicitly overridden. The engine is intentionally compiled for ONE
# target — runtime multi-ISA dispatch is out of scope by design, not debt.
#   native   (default) → -march=native   (this build host's full ISA)
#   portable           → -O3 only        (baseline x86-64; scalar fallbacks)
#   <other>            → -march=<other>   (explicit ISA, e.g. x86-64-v3)
set(LAYERSTORM_ARCH "native" CACHE STRING
    "Global CPU ISA target for LayerStoRm hot-path kernels: native | portable | <flag>")

set(LAYERSTORM_CUTLASS_SRC_DIR   "${CMAKE_SOURCE_DIR}/3rd-party/cutlass" CACHE PATH "CUTLASS source directory")
set(LAYERSTORM_NVCC_THREADS      "8" CACHE STRING "Parallel NVCC compilation threads")

# ── Required: CUDA Toolkit ────────────────────────────────────────────────────
find_package(CUDAToolkit 12.8 REQUIRED)
find_package(Threads REQUIRED)

# ── Required: CUTLASS (header-only) ──────────────────────────────────────────
include(FetchContent)

if(LAYERSTORM_CUTLASS_SRC_DIR)
  FetchContent_Declare(cutlass
    SOURCE_DIR "${LAYERSTORM_CUTLASS_SRC_DIR}"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
else()
  FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        v4.4.2
    GIT_PROGRESS   TRUE
    GIT_SHALLOW    TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
endif()

# Populate only — do NOT add_subdirectory (avoids hundreds of CUTLASS targets)
FetchContent_Populate(cutlass)
message(STATUS "CUTLASS available at ${cutlass_SOURCE_DIR}")

add_library(cutlass_headers INTERFACE)
target_include_directories(cutlass_headers INTERFACE
  "${cutlass_SOURCE_DIR}/include"
  "${cutlass_SOURCE_DIR}/tools/util/include"
)

# ── Required: NCCL ───────────────────────────────────────────────────────────
find_package(NCCL 2.20 REQUIRED)

# ── Required: nlohmann_json ──────────────────────────────────────────────────
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found locally, fetching...")
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# ── Required: spdlog ─────────────────────────────────────────────────────────
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
  message(STATUS "spdlog not found locally, fetching...")
  FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.14.1
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(spdlog)
endif()

# ── Optional: pybind11 (only if building Python bindings) ────────────────────
if(LAYERSTORM_BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
endif()

# ── Optional: GTest (only if building tests) ─────────────────────────────────
if(LAYERSTORM_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE
  )
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# ── Optional: libnuma ────────────────────────────────────────────────────────
if(LAYERSTORM_USE_NUMA)
  find_package(Numa)
  if(Numa_FOUND)
    add_compile_definitions(LAYERSTORM_HAS_NUMA)
    message(STATUS "libnuma enabled")
    if(LAYERSTORM_DETECT_HBM)
      add_compile_definitions(LAYERSTORM_HBM_NODES)
      message(STATUS "HBM NUMA-node detection enabled (LAYERSTORM_DETECT_HBM)")
    endif()
  else()
    message(WARNING "libnuma requested but not found; disabling NUMA support")
    set(LAYERSTORM_USE_NUMA OFF)
  endif()
endif()

# ── Optional: liburing ───────────────────────────────────────────────────────
if(LAYERSTORM_USE_NVME)
  find_package(Uring)
  if(Uring_FOUND)
    add_compile_definitions(LAYERSTORM_HAS_URING)
    message(STATUS "liburing enabled")
  else()
    message(WARNING "liburing requested but not found; disabling NVMe tier")
    set(LAYERSTORM_USE_NVME OFF)
  endif()
endif()

# perf_trace compile-time gate. Defined globally (all TUs that include
# perf_trace.h must agree — the disabled branch is inline, so a per-TU mismatch
# would be an ODR violation). Header defaults to 1 when undefined (ON).
if(NOT LAYERSTORM_PERF_TRACE)
  add_compile_definitions(LAYERSTORM_PERF_TRACE=0)
  message(STATUS "perf_trace instrumentation compiled OUT")
endif()

# ── Global CUDA flags ────────────────────────────────────────────────────────
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
  $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
  $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF2_OPERATORS__>
  $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
  $<$<COMPILE_LANGUAGE:CUDA>:--threads=${LAYERSTORM_NVCC_THREADS}>
)

# ── Subdirectories ───────────────────────────────────────────────────────────
add_subdirectory(src)

if(LAYERSTORM_BUILD_PYTHON)
  add_subdirectory(python)
endif()

add_subdirectory(tools)

# Trap #11 guard: LS_BUILD_GATES env (read at CONFIGURE time, re-evaluated on
# every reconfigure — NOT cached) puts the manually-run gate executables
# (tests/integration/* + tests/manual/*) into the DEFAULT build target, so a
# plain `cmake --build build` can never leave a stale gate binary behind.
# Unit tests stay out (costly). Arm:   LS_BUILD_GATES=1 cmake -B build
#                              Disarm: cmake -B build   (env unset or "0")
if(DEFINED ENV{LS_BUILD_GATES} AND NOT "$ENV{LS_BUILD_GATES}" STREQUAL ""
   AND NOT "$ENV{LS_BUILD_GATES}" STREQUAL "0")
  set(LAYERSTORM_BUILD_GATES ON)
  message(STATUS "LS_BUILD_GATES: integration + manual gate executables are in "
                 "the DEFAULT build (Trap #11 guard)")
else()
  set(LAYERSTORM_BUILD_GATES OFF)
endif()

if(LAYERSTORM_BUILD_TESTS OR LAYERSTORM_BUILD_BENCHMARKS)
  # enable_testing() must be in the top-level CMakeLists so ctest finds
  # CTestTestfile.cmake in the build root.
  enable_testing()
  add_subdirectory(tests)
endif()

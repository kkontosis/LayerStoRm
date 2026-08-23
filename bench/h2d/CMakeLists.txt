# Standalone H2D micro-benchmark — independent of the main LayerStoRm build.
# Configure:  cmake -S bench/h2d -B bench/h2d/build -DCMAKE_BUILD_TYPE=Release
# Build:      cmake --build bench/h2d/build -j
# Run:        ./bench/h2d/build/h2d_bench [all|latency|sweep|pin|mech|sync|numa|contention]
cmake_minimum_required(VERSION 3.25)
project(h2d_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# RTX 5090 / 5080 are SM_120.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_library(NUMA_LIB numa REQUIRED)

add_executable(h2d_bench h2d_bench.cu)
target_link_libraries(h2d_bench PRIVATE ${NUMA_LIB})
target_compile_options(h2d_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>)

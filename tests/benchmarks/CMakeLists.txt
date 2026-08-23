# ── Benchmarks ────────────────────────────────────────────────────────────────
# Each *_benchmark.cpp file becomes its own executable.

file(GLOB BENCHMARK_SOURCES *_benchmark.cpp)

foreach(bench_source ${BENCHMARK_SOURCES})
  get_filename_component(bench_name ${bench_source} NAME_WE)
  add_executable(${bench_name} ${bench_source})
  target_link_libraries(${bench_name} PRIVATE layerstorm_core)
endforeach()

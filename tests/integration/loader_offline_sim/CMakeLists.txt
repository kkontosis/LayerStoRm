# ── CPU-only loader offline-simulation integration test ──────────────────────
# GPU-free, weights-free: replays each variant's recorded routed trace through
# the REAL engine eviction — a CPU-constructible memory::ExpertCache (NullBackends,
# no CUDA calls), the gpu_loader::EvictScoreBoard as its ResidencyListener, the
# SHARED gpu_loader::apply_far_evictions victim selector (the SAME function the
# daemon calls), the REAL gpu_loader::LoaderSolver placement + T(j) cost model.
#
# Because it now drives the real engine classes (ExpertCache / EvictScoreBoard /
# ExpertCache eviction path) rather than the two stand-alone loader TUs, it links
# layerstorm_core (which is CUDA-built but makes ZERO CUDA calls here: the
# NullDeviceBackend allocs via std::aligned_alloc and every compute op is a
# no-op). The test runs on a box with no live GPU — it is the single-source-of-
# truth cross-check that the sim's hit-rate == the engine's hit-rate.
#
# Build:  cmake --build build --target loader_offline_sim_test -j
# Run:    ctest --test-dir build -R loader_offline_sim --output-on-failure
add_executable(loader_offline_sim_test EXCLUDE_FROM_ALL
  loader_offline_sim_test.cpp
  offline_sim.cpp
)
target_link_libraries(loader_offline_sim_test PRIVATE
  layerstorm_core
  GTest::gtest_main
  nlohmann_json::nlohmann_json
)
target_include_directories(loader_offline_sim_test PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(loader_offline_sim_test PRIVATE
  LAYERSTORM_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

add_test(NAME loader_offline_sim_test COMMAND loader_offline_sim_test)
set_tests_properties(loader_offline_sim_test PROPERTIES
  LABELS "integration;cpu"
  TIMEOUT 120
)

if(TARGET layerstorm_integration_tests)
  add_dependencies(layerstorm_integration_tests loader_offline_sim_test)
endif()

# ── DRIFT-PROBE: resolve the LS_EVICT_DECAY "no-op" contradiction ─────────────
# Decision-stream diff (decay-on vs decay-off on the fixed trace), the decay
# mechanism on the REAL EvictScoreBoard, the clean-LRU ceiling gap + bounded-
# recency prototype, and CPU determinism. CPU-only, same link set.
add_executable(decay_drift_probe_test EXCLUDE_FROM_ALL
  decay_drift_probe_test.cpp
  offline_sim.cpp
)
target_link_libraries(decay_drift_probe_test PRIVATE
  layerstorm_core
  GTest::gtest_main
  nlohmann_json::nlohmann_json
)
target_include_directories(decay_drift_probe_test PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(decay_drift_probe_test PRIVATE
  LAYERSTORM_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
add_test(NAME decay_drift_probe_test COMMAND decay_drift_probe_test)
set_tests_properties(decay_drift_probe_test PROPERTIES
  LABELS "integration;cpu"
  TIMEOUT 180
)
if(TARGET layerstorm_integration_tests)
  add_dependencies(layerstorm_integration_tests decay_drift_probe_test)
endif()

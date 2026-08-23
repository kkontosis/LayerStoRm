# ── Integration tests ─────────────────────────────────────────────────────────
# Hardware-dependent end-to-end tests. Each *_test.cpp is its own executable.
#
# EXCLUDE_FROM_ALL: not built by default — requires GPU + model weights.
#
# Build all:   cmake --build build --target layerstorm_integration_tests
# Build one:   cmake --build build --target first_token_test
# Run all:     ctest --test-dir build -L integration --output-on-failure
# Run one:     ./build/tests/integration/first_token_test [--gtest_filter=...]

file(GLOB INTEGRATION_TEST_SOURCES CONFIGURE_DEPENDS *_test.cpp)

set(_integ_targets "")

# Trap #11 guard: LS_BUILD_GATES (root CMakeLists) drops EXCLUDE_FROM_ALL so
# these gate executables rebuild on every default build.
set(_integ_excl EXCLUDE_FROM_ALL)
if(LAYERSTORM_BUILD_GATES)
  set(_integ_excl "")
endif()

foreach(test_source ${INTEGRATION_TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME_WE)

  add_executable(${test_name} ${_integ_excl} ${test_source})
  # --whole-archive ensures quant self-registration static constructors are
  # included even when no symbol is directly referenced from the test.
  target_link_libraries(${test_name} PRIVATE
    -Wl,--whole-archive layerstorm_core -Wl,--no-whole-archive
    GTest::gtest_main
    CUDA::cudart
  )
  target_include_directories(${test_name} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
  )
  target_compile_definitions(${test_name} PRIVATE
    LAYERSTORM_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )

  add_test(NAME ${test_name} COMMAND ${test_name})
  set_tests_properties(${test_name} PROPERTIES
    LABELS "integration"
    RESOURCE_LOCK "gpu"
    TIMEOUT 120
  )

  list(APPEND _integ_targets ${test_name})
endforeach()

if(_integ_targets)
  add_custom_target(layerstorm_integration_tests
    DEPENDS ${_integ_targets}
  )
endif()

# CPU-only, GPU-free loader offline-simulation test (its own subdir target).
add_subdirectory(loader_offline_sim)

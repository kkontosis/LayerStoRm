# ── Smoke tests ───────────────────────────────────────────────────────────────
# Hardware-dependent tests. Each *_smoke.cpp is its own executable.
#
# EXCLUDE_FROM_ALL: never built by default — not included in `cmake --build build`,
# `layerstorm_unit_tests`, or `layerstorm_integration_tests`. Must be built
# explicitly by name.
#
# Build all:   cmake --build build --target layerstorm_smoke_tests
# Build one:   cmake --build build --target gpu_topology_smoke
# Run all:     ctest --test-dir build -L smoke --output-on-failure
# Run one:     ./build/tests/smoke/gpu_topology_smoke [--gtest_filter=...]

file(GLOB SMOKE_SOURCES *_smoke.cpp *_smoke.cu)

set(_smoke_targets "")

foreach(_src ${SMOKE_SOURCES})
    get_filename_component(_name ${_src} NAME_WE)

    add_executable(${_name} EXCLUDE_FROM_ALL ${_src})
    target_link_libraries(${_name} PRIVATE
        layerstorm_core
        GTest::gtest_main
        CUDA::cudart
    )
    target_include_directories(${_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/src
    )

    # Register with ctest under the "smoke" label.
    # Build the target before running: cmake --build build --target <name>
    add_test(NAME ${_name} COMMAND ${_name})
    set_tests_properties(${_name} PROPERTIES
        LABELS "smoke"
        RESOURCE_LOCK "gpu"   # serialize GPU smoke tests to avoid VRAM contention
    )

    list(APPEND _smoke_targets ${_name})
endforeach()

# Umbrella target: build every smoke binary in one command.
if(_smoke_targets)
    # add_custom_target is excluded from ALL by default (no ALL keyword).
    add_custom_target(layerstorm_smoke_tests
        DEPENDS ${_smoke_targets}
    )
endif()

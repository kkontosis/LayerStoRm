# ── Manual tests ──────────────────────────────────────────────────────────────
# Tests that exercise real hardware or filesystem I/O. Each *_manual.cpp is
# its own executable.
#
# EXCLUDE_FROM_ALL: never built by default — not included in `cmake --build build`,
# `layerstorm_unit_tests`, or `layerstorm_integration_tests`. Must be built
# explicitly by name.
#
# Build all:   cmake --build build --target layerstorm_manual_tests
# Build one:   cmake --build build --target nvme_tier_manual
# Run all:     ctest --test-dir build -L manual --output-on-failure
# Run one:     ./build/tests/manual/nvme_tier_manual [--gtest_filter=...]

file(GLOB MANUAL_SOURCES *_manual.cpp)

set(_manual_targets "")

# Trap #11 guard: LS_BUILD_GATES (root CMakeLists) drops EXCLUDE_FROM_ALL.
set(_manual_excl EXCLUDE_FROM_ALL)
if(LAYERSTORM_BUILD_GATES)
    set(_manual_excl "")
endif()
# Bitrotted vs the current NvmeTier API (TD-NVME-MANUAL-BITROT): stays out of
# the LS_BUILD_GATES default build until repaired (still buildable by name,
# where it fails loudly).
set(_gate_skip nvme_tier_manual)

foreach(_src ${MANUAL_SOURCES})
    get_filename_component(_name ${_src} NAME_WE)

    set(_this_excl ${_manual_excl})
    if(_name IN_LIST _gate_skip)
        set(_this_excl EXCLUDE_FROM_ALL)
    endif()
    add_executable(${_name} ${_this_excl} ${_src})
    target_link_libraries(${_name} PRIVATE
        layerstorm_core
        GTest::gtest_main
    )
    target_include_directories(${_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/src
    )

    # Register with ctest under the "manual" label.
    # Build the target before running: cmake --build build --target <name>
    add_test(NAME ${_name} COMMAND ${_name})
    set_tests_properties(${_name} PROPERTIES LABELS "manual")

    list(APPEND _manual_targets ${_name})
endforeach()

# Umbrella target: build every manual binary in one command.
if(_manual_targets)
    add_custom_target(layerstorm_manual_tests
        DEPENDS ${_manual_targets}
    )
endif()

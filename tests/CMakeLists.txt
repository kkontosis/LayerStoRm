# ── Test orchestrator ─────────────────────────────────────────────────────────

if(LAYERSTORM_BUILD_TESTS)
  add_subdirectory(unit)
  add_subdirectory(smoke)
  add_subdirectory(integration)
  add_subdirectory(manual)
endif()

if(LAYERSTORM_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ── _layerstorm pybind11 module ────────────────────────────────────────────────

file(GLOB PYBIND_SOURCES bindings/*.cpp)
# engine_pybind.cpp defines its own PYBIND11_MODULE — built as layerstorm_engine below.
list(FILTER PYBIND_SOURCES EXCLUDE REGEX ".*engine_pybind\\.cpp$")

if(PYBIND_SOURCES)
  pybind11_add_module(_layerstorm ${PYBIND_SOURCES})
else()
  # Placeholder until pybind sources are added
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/stub_pybind.cpp"
    "#include <pybind11/pybind11.h>\nPYBIND11_MODULE(_layerstorm, m) { m.doc() = \"layerstorm stub\"; }\n")
  pybind11_add_module(_layerstorm "${CMAKE_CURRENT_BINARY_DIR}/stub_pybind.cpp")
endif()

target_link_libraries(_layerstorm PRIVATE layerstorm_core)

# ── layerstorm_engine pybind11 module ─────────────────────────────────────────
# Daemon engine: start_engine(), stop_engine(), EngineInfo.
# pybind11 bindings live here; Engine class itself is in layerstorm_core.

pybind11_add_module(layerstorm_engine
  bindings/engine_pybind.cpp
)

target_include_directories(layerstorm_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

# --whole-archive ensures static initializers (quant self-registration, etc.)
# are included even when nothing in the bindings directly references them.
target_link_libraries(layerstorm_engine PRIVATE
  -Wl,--whole-archive layerstorm_core -Wl,--no-whole-archive
)

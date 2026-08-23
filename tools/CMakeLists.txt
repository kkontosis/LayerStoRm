# ── Offline tools ──────────────────────────────────────────────────────────────
# Standalone CLI utilities. Each *.cpp is its own executable.
#
# EXCLUDE_FROM_ALL: not built by default — must be built explicitly by name.
#
# Build:  cmake --build build --target prepack_experts
# Run:    ./build/tools/prepack_experts <config.json> <output_dir>

add_executable(prepack_experts EXCLUDE_FROM_ALL prepack_experts.cpp)
target_link_libraries(prepack_experts PRIVATE
    -Wl,--whole-archive layerstorm_core -Wl,--no-whole-archive
)
target_include_directories(prepack_experts PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# ── layerstorm_arena_holder (P-24b) ────────────────────────────────────────────
# Persistent-arena fd keeper: CUDA-free, links libc ONLY (INV-ARENA-HOLD — no
# layerstorm_core, no CUDA, no spdlog). Part of the DEFAULT build so the
# engine's arena_attach auto-spawn can find a fresh binary next to the build.
add_executable(layerstorm_arena_holder arena_holder.cpp)
target_include_directories(layerstorm_arena_holder PRIVATE
    ${CMAKE_SOURCE_DIR}/src   # arena_ipc_protocol.h (header-only, libc-only)
)

# ── layerstorm_arena_ctl ───────────────────────────────────────────────────────
# Operator CLI for the holder (status / kOperatorRequest wipe). Same libc-only
# stance as the holder — speaks the wire protocol directly, links nothing.
# Part of the DEFAULT build: it is an ops primitive that must always match the
# holder binary next to it.
add_executable(layerstorm_arena_ctl arena_ctl.cpp)
target_include_directories(layerstorm_arena_ctl PRIVATE
    ${CMAKE_SOURCE_DIR}/src   # arena_ipc_protocol.h (header-only, libc-only)
)

# ── reef_sim (TD-BRIDGE-CPP-GAP Q1) ────────────────────────────────────────────
# Offline REEF bank-seam replay: links the REAL ReefOrch decision stack and
# replays recorded engine traces (LS_REEF_DECISION_DUMP + LS_REEF_RELOC_TRACE)
# under the four bank-input policies. CPU-only.
add_executable(reef_sim EXCLUDE_FROM_ALL reef_sim/reef_sim.cpp)
target_link_libraries(reef_sim PRIVATE
    -Wl,--whole-archive layerstorm_core -Wl,--no-whole-archive
)
target_include_directories(reef_sim PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

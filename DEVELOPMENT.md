# Development

Environment setup (venv before CMake, Node.js, non-default CUDA) is covered in
the README's [Building](README.md#building) section. This file assumes it is
done.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

Useful environment for a working session:

```sh
source .venv/bin/activate                 # or pass -DPYTHON_EXECUTABLE explicitly
export CMAKE_EXPORT_COMPILE_COMMANDS=ON   # compile_commands.json for clangd / IDEs
export LS_BUILD_GATES=1                   # see below
```

`LS_BUILD_GATES` is read at **configure** time: with it set, the integration
and manual gate executables are added to the build (they are excluded by
default because they are costly to compile). Re-run `cmake -B build` after
changing it.

Python dependency sets: `requirements.txt` (build + serve),
`requirements-dev.txt` (pytest, the end-to-end HTTP client, and `jsonschema`
for `tests/validate_schema.sh`),
`requirements-tools.txt` (torch/LightGBM for the offline tooling under
`tools/`).

The command ring's Cython hot path is part of the standard install (see the
README). Re-run it after touching the ring protocol — a stale `_fastbridge`
keeps serving the old layout:

```sh
.venv/bin/python python/bridge/build_fastbridge.py
```

`bridge.ring_bridge.fastbridge_active()` reports whether the compiled path or
the pure-ctypes fallback is live.

Common flags:

| Flag | Default | Purpose |
|---|---|---|
| `LAYERSTORM_BUILD_TESTS` | ON | Unit/integration tests |
| `LAYERSTORM_BUILD_PYTHON` | ON | pybind11 module (requires pybind11) |
| `LAYERSTORM_BUILD_BENCHMARKS` | ON | Benchmark executables |
| `LAYERSTORM_USE_NUMA` | ON | libnuma support |
| `LAYERSTORM_USE_NVME` | OFF | io_uring NVMe tier |

Minimal build (no Python, no benchmarks):

```sh
cmake -S . -B build -DLAYERSTORM_BUILD_PYTHON=OFF -DLAYERSTORM_BUILD_BENCHMARKS=OFF
cmake --build build -j$(nproc)
```

## Tests

```sh
./build/tests/unit/layerstorm_unit_tests
```

Filter to a specific test suite:

```sh
./build/tests/unit/layerstorm_unit_tests --gtest_filter=ConfigParser.*
```

## Requirements

- CMake 3.25+
- CUDA 12.8+ (13.1 confirmed working)
- GCC with C++20 support
- NCCL 2.20+
- pybind11 (only if `LAYERSTORM_BUILD_PYTHON=ON`)

nlohmann_json and spdlog are fetched automatically if not found locally.

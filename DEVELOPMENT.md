# Development

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

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

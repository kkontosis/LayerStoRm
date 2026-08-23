#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

/// Skip the current gtest if no CUDA device is available.
/// Use this at the top of any unit test that calls CUDA directly.
/// On a machine with GPUs the test runs normally; on headless CI it is SKIPPED.
#define REQUIRES_GPU()                                              \
    do {                                                            \
        int _gpu_count = 0;                                         \
        cudaError_t _err = cudaGetDeviceCount(&_gpu_count);         \
        if (_err != cudaSuccess || _gpu_count == 0) {               \
            GTEST_SKIP() << "No CUDA GPU available — skipping";     \
        }                                                           \
    } while (0)

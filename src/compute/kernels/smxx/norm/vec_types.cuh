// Portable CUDA vector types and helpers for LayerStoRm norm kernels.
// Adapted from vLLM (type_convert.cuh, vectorization.cuh, cub_helpers.h;
// Apache-2.0, Copyright contributors to the vLLM project —
// see THIRD_PARTY_NOTICES.md).
// No PyTorch/ATen dependency.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cuda/std/functional>

namespace layerstorm::compute {

// ── CUB reduction op ────────────────────────────────────────────────────────

using CubAddOp = cuda::std::plus<>;

// ── Aligned vector struct ───────────────────────────────────────────────────

template <typename scalar_t, int N>
struct __align__(N * sizeof(scalar_t)) vec_n_t {
    scalar_t val[N];
};

// ── Type conversion helpers ─────────────────────────────────────────────────

template <typename T>
struct TypeConvert {
    static constexpr bool exists = false;
};

template <>
struct TypeConvert<float> {
    static constexpr bool exists = true;
    using cuda_type = float;
    using packed_type = float2;

    __device__ static __forceinline__ float convert(cuda_type x) { return x; }
    __device__ static __forceinline__ float2 convert(packed_type x) { return x; }
    __device__ static __forceinline__ cuda_type convert_from(float x) { return x; }
    __device__ static __forceinline__ packed_type convert_from(float2 x) {
        return x;
    }
};

template <>
struct TypeConvert<__half> {
    static constexpr bool exists = true;
    using cuda_type = __half;
    using packed_type = __half2;

    __device__ static __forceinline__ float convert(cuda_type x) {
        return __half2float(x);
    }
    __device__ static __forceinline__ float2 convert(packed_type x) {
        return __half22float2(x);
    }
    __device__ static __forceinline__ cuda_type convert_from(float x) {
        return __float2half_rn(x);
    }
    __device__ static __forceinline__ packed_type convert_from(float2 x) {
        return __float22half2_rn(x);
    }
};

template <>
struct TypeConvert<__nv_bfloat16> {
    static constexpr bool exists = true;
    using cuda_type = __nv_bfloat16;
    using packed_type = __nv_bfloat162;

    __device__ static __forceinline__ float convert(cuda_type x) {
        return __bfloat162float(x);
    }
    __device__ static __forceinline__ float2 convert(packed_type x) {
        return __bfloat1622float2(x);
    }
    __device__ static __forceinline__ cuda_type convert_from(float x) {
        return __float2bfloat16(x);
    }
    __device__ static __forceinline__ packed_type convert_from(float2 x) {
        return __float22bfloat162_rn(x);
    }
};

// ── f16Vec: vectorized FP16/BF16 operations ─────────────────────────────────
// Adapted from vLLM _f16Vec. 16-byte aligned for 128-bit global memory ops.

template <typename scalar_t, int width>
struct alignas(16) f16Vec {
    static_assert(width > 0 && (width & (width - 1)) == 0,
                  "width must be a positive power of 2");
    using Converter = TypeConvert<scalar_t>;
    using T1 = typename Converter::cuda_type;
    using T2 = typename Converter::packed_type;
    T1 data[width];

    __device__ f16Vec& operator+=(const f16Vec& other) {
        if constexpr (width % 2 == 0) {
#pragma unroll
            for (int i = 0; i < width; i += 2) {
                if constexpr (std::is_same_v<T2, float2>) {
                    data[i] += other.data[i];
                    data[i + 1] += other.data[i + 1];
                } else {
                    T2 temp{data[i], data[i + 1]};
                    temp += T2{other.data[i], other.data[i + 1]};
                    data[i] = temp.x;
                    data[i + 1] = temp.y;
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < width; ++i) data[i] += other.data[i];
        }
        return *this;
    }

    __device__ f16Vec& operator*=(const f16Vec& other) {
        if constexpr (width % 2 == 0) {
#pragma unroll
            for (int i = 0; i < width; i += 2) {
                if constexpr (std::is_same_v<T2, float2>) {
                    data[i] *= other.data[i];
                    data[i + 1] *= other.data[i + 1];
                } else {
                    T2 temp{data[i], data[i + 1]};
                    temp *= T2{other.data[i], other.data[i + 1]};
                    data[i] = temp.x;
                    data[i + 1] = temp.y;
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < width; ++i) data[i] *= other.data[i];
        }
        return *this;
    }

    __device__ f16Vec& operator*=(const float scale) {
        if constexpr (width % 2 == 0) {
#pragma unroll
            for (int i = 0; i < width; i += 2) {
                float2 temp_f =
                    Converter::convert(T2{data[i], data[i + 1]});
                temp_f.x *= scale;
                temp_f.y *= scale;
                T2 temp = Converter::convert_from(temp_f);
                data[i] = temp.x;
                data[i + 1] = temp.y;
            }
        } else {
#pragma unroll
            for (int i = 0; i < width; ++i) {
                float temp = Converter::convert(data[i]) * scale;
                data[i] = Converter::convert_from(temp);
            }
        }
        return *this;
    }

    __device__ float sum_squares() const {
        float result = 0.0f;
        if constexpr (width % 2 == 0) {
#pragma unroll
            for (int i = 0; i < width; i += 2) {
                float2 z = Converter::convert(T2{data[i], data[i + 1]});
                result += z.x * z.x + z.y * z.y;
            }
        } else {
#pragma unroll
            for (int i = 0; i < width; ++i) {
                float x = Converter::convert(data[i]);
                result += x * x;
            }
        }
        return result;
    }
};

// ── Vectorized read helper ──────────────────────────────────────────────────
// Read-only iteration with alignment-aware vectorized access.
// Adapted from vLLM vectorize_read_with_alignment.

template <int VEC_SIZE, typename InT, typename VecOp, typename ScaOp>
__device__ inline void vectorize_read_aligned(const InT* in, int len, int tid,
                                              int stride, VecOp&& vec_op,
                                              ScaOp&& scalar_op) {
    static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                  "VEC_SIZE must be a positive power-of-two");
    constexpr int WIDTH = VEC_SIZE * static_cast<int>(sizeof(InT));
    auto addr = reinterpret_cast<uintptr_t>(in);

    bool can_vec =
        ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
    if (can_vec) {
        int num_vec = len / VEC_SIZE;
        using vin_t = vec_n_t<InT, VEC_SIZE>;
        auto* v_in = reinterpret_cast<const vin_t*>(in);
        for (int i = tid; i < num_vec; i += stride) {
            vin_t tmp = v_in[i];
            vec_op(tmp);
        }
        return;
    }

    // Handle misaligned prefix
    int misalignment = static_cast<int>(addr & (WIDTH - 1));
    int alignment_bytes = WIDTH - misalignment;
    int prefix_elems = (alignment_bytes & (WIDTH - 1)) / static_cast<int>(sizeof(InT));
    if (prefix_elems > len) prefix_elems = len;

    for (int i = tid; i < prefix_elems; i += stride) {
        scalar_op(in[i]);
    }

    const InT* aligned_in = in + prefix_elems;
    int remaining = len - prefix_elems;
    int num_vec = remaining / VEC_SIZE;
    using vin_t = vec_n_t<InT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<const vin_t*>(aligned_in);

    for (int i = tid; i < num_vec; i += stride) {
        vec_op(v_in[i]);
    }

    int tail_start = num_vec * VEC_SIZE;
    for (int i = tid + tail_start; i < remaining; i += stride) {
        scalar_op(aligned_in[i]);
    }
}

}  // namespace layerstorm::compute

// Fused weight-dequant dense GEMM kernels for SM120 (TD-DSPARK-DRAFT-QUANT).
// See wq_gemm.h for the contract. Kernel structure mirrors bf16_gemm.cu
// (GEMV / multi-row GEMV / register-tiled GEMM, FP32 accumulation); the
// weight load is a policy object that dequantizes FP8-E4M3 or NVFP4 in the
// K-loop. Decode semantics are bit-matched to the CPU references in
// model/quantization/{fp8,nvfp4,kgroup_quant}.cpp.

#include "compute/kernels/sm120/gemm/wq_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

constexpr int WARP_SIZE = 32;

__forceinline__ __device__ float bf16_load(const __nv_bfloat16* p, int64_t i) {
    return __bfloat162float(p[i]);
}

__forceinline__ __device__ void store_out(float* p, float v) { *p = v; }
__forceinline__ __device__ void store_out(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}

// ── Weight-load policies ────────────────────────────────────────────────────
// load(n, k) returns the dequantized element at global column (k_off + k) of
// row n. Scale groups are anchored at the row start (kgroup_quant.h layout).

struct Fp8Loader {
    const uint8_t* __restrict__ q;
    const float* __restrict__ s;
    int64_t ldw, lds, k_off;

    __forceinline__ __device__ float load(int n, int k) const {
        const int64_t kk = k_off + k;
        const uint8_t b = q[static_cast<int64_t>(n) * ldw + kk];
        // E4M3fn decode (matches fp8_e4m3::decode; encoder never emits NaN).
        const int exp = (b >> 3) & 0x0F;
        const int man = b & 0x07;
        float v = (exp == 0)
                      ? ldexpf(static_cast<float>(man), -9)
                      : ldexpf(static_cast<float>(8 + man), exp - 10);
        v = (b & 0x80) ? -v : v;
        return v * s[static_cast<int64_t>(n) * lds + (kk >> 7)];
    }
};

__constant__ float kE2M1Mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

struct Nvfp4Loader {
    const uint8_t* __restrict__ q;
    const uint8_t* __restrict__ s;
    int64_t row_bytes;  // ldw / 2 (ldw even)
    int64_t lds, k_off;

    __forceinline__ __device__ float load(int n, int k) const {
        const int64_t kk = k_off + k;
        const uint8_t byte =
            q[static_cast<int64_t>(n) * row_bytes + (kk >> 1)];
        const uint8_t nib = (kk & 1) ? (byte >> 4) : (byte & 0x0F);
        const float mag = kE2M1Mag[nib & 7];
        const float v = (nib & 8) ? -mag : mag;
        // UE8M0: 2^(byte-127) via exponent-field placement (byte 0 -> 0.0f;
        // the packer clamps to [1, 254]).
        const uint32_t sb =
            static_cast<uint32_t>(s[static_cast<int64_t>(n) * lds + (kk >> 4)])
            << 23;
        return v * __uint_as_float(sb);
    }
};

// ── M == 1 GEMV: one warp per output column, lanes stride K ─────────────────

constexpr int GEMV_WARPS_PER_BLOCK = 8;
constexpr int GEMV_BLOCK = GEMV_WARPS_PER_BLOCK * WARP_SIZE;  // 256

template <typename WL, typename OutT>
__global__ void __launch_bounds__(GEMV_BLOCK)
wq_gemv_nt_kernel(OutT* __restrict__ C, const __nv_bfloat16* __restrict__ A,
                  WL wl, int N, int K) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n = blockIdx.x * GEMV_WARPS_PER_BLOCK + warp;
    if (n >= N) return;

    float acc = 0.0f;
    for (int k = lane; k < K; k += WARP_SIZE)
        acc = fmaf(bf16_load(A, k), wl.load(n, k), acc);

    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) store_out(&C[n], acc);
}

// ── Small-M multi-row GEMV (1 < M <= 32): amortize the W-row dequant over an
//    MT-row register tile (the LS_CHUNK_SMALLM pattern — always on here: the
//    quant path is new, there is no bit-compat baseline to preserve) ─────────

constexpr int MROWS_MT = 8;
constexpr int MROWS_MAX_M = 32;

template <typename WL, typename OutT>
__global__ void __launch_bounds__(GEMV_BLOCK)
wq_gemv_nt_mrows_kernel(OutT* __restrict__ C,
                        const __nv_bfloat16* __restrict__ A, WL wl,
                        int M, int N, int K, int64_t lda) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n = blockIdx.x * GEMV_WARPS_PER_BLOCK + warp;
    if (n >= N) return;

    for (int m0 = 0; m0 < M; m0 += MROWS_MT) {
        const int mt = min(M - m0, MROWS_MT);
        float acc[MROWS_MT];
        #pragma unroll
        for (int j = 0; j < MROWS_MT; ++j) acc[j] = 0.0f;

        for (int k = lane; k < K; k += WARP_SIZE) {
            const float w = wl.load(n, k);
            #pragma unroll
            for (int j = 0; j < MROWS_MT; ++j)
                if (j < mt)
                    acc[j] = fmaf(
                        bf16_load(A + static_cast<int64_t>(m0 + j) * lda, k),
                        w, acc[j]);
        }

        #pragma unroll
        for (int j = 0; j < MROWS_MT; ++j) {
            if (j < mt) {
                float v = acc[j];
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, off);
                if (lane == 0)
                    store_out(&C[static_cast<int64_t>(m0 + j) * N + n], v);
            }
        }
    }
}

// ── M > 32 register-tiled GEMM (bf16_gemm.cu tile geometry) ─────────────────

constexpr int BM = 16, BN = 16, TM = 4, TN = 4;
constexpr int TILE_M = BM * TM;  // 64
constexpr int TILE_N = BN * TN;  // 64
constexpr int TK = 16;

template <typename WL, typename OutT>
__global__ void __launch_bounds__(BM * BN)
wq_gemm_nt_kernel(OutT* __restrict__ C, const __nv_bfloat16* __restrict__ A,
                  WL wl, int M, int N, int K, int64_t lda) {
    __shared__ float sA[TILE_M][TK];
    __shared__ float sW[TILE_N][TK];

    const int tid = threadIdx.y * BN + threadIdx.x;
    const int threads = BM * BN;
    const int row0 = blockIdx.y * TILE_M;
    const int col0 = blockIdx.x * TILE_N;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int idx = tid; idx < TILE_M * TK; idx += threads) {
            const int r = idx / TK;
            const int c = idx % TK;
            const int gr = row0 + r;
            const int gk = k0 + c;
            sA[r][c] = (gr < M && gk < K)
                ? bf16_load(A + static_cast<int64_t>(gr) * lda, gk) : 0.0f;
        }
        for (int idx = tid; idx < TILE_N * TK; idx += threads) {
            const int r = idx / TK;
            const int c = idx % TK;
            const int gn = col0 + r;
            const int gk = k0 + c;
            sW[r][c] = (gn < N && gk < K) ? wl.load(gn, gk) : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            float aReg[TM], wReg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                aReg[i] = sA[threadIdx.y * TM + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                wReg[j] = sW[threadIdx.x * TN + j][kk];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(aReg[i], wReg[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = row0 + threadIdx.y * TM + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = col0 + threadIdx.x * TN + j;
            if (gc >= N) continue;
            store_out(&C[static_cast<int64_t>(gr) * N + gc], acc[i][j]);
        }
    }
}

// ── Dispatch ────────────────────────────────────────────────────────────────

template <typename WL, typename OutT>
void dispatch_wq(void* C, const void* A, WL wl, int M, int N, int K,
                 int64_t lda, cudaStream_t stream) {
    auto* c = static_cast<OutT*>(C);
    const auto* a = static_cast<const __nv_bfloat16*>(A);
    if (M == 1) {
        dim3 grid((N + GEMV_WARPS_PER_BLOCK - 1) / GEMV_WARPS_PER_BLOCK);
        wq_gemv_nt_kernel<WL, OutT>
            <<<grid, GEMV_BLOCK, 0, stream>>>(c, a, wl, N, K);
    } else if (M <= MROWS_MAX_M) {
        dim3 grid((N + GEMV_WARPS_PER_BLOCK - 1) / GEMV_WARPS_PER_BLOCK);
        wq_gemv_nt_mrows_kernel<WL, OutT>
            <<<grid, GEMV_BLOCK, 0, stream>>>(c, a, wl, M, N, K, lda);
    } else {
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        dim3 block(BN, BM);
        wq_gemm_nt_kernel<WL, OutT>
            <<<grid, block, 0, stream>>>(c, a, wl, M, N, K, lda);
    }
}

template <typename WL>
void dispatch_wq_out(void* C, const void* A, WL wl, int M, int N, int K,
                     int64_t lda, GemmAccOutDtype out_dtype,
                     cudaStream_t stream) {
    switch (out_dtype) {
        case GemmAccOutDtype::kFloat32:
            dispatch_wq<WL, float>(C, A, wl, M, N, K, lda, stream);
            break;
        case GemmAccOutDtype::kBFloat16:
            dispatch_wq<WL, __nv_bfloat16>(C, A, wl, M, N, K, lda, stream);
            break;
        default:
            throw std::runtime_error(
                "launch_wq_gemm_nt: unsupported output dtype");
    }
}

}  // namespace

void launch_wq_gemm_nt(void* C, const void* A, const void* Wq,
                       const void* scales, WqWeightKind kind,
                       int M, int N, int K,
                       int64_t lda, int64_t ldw, int64_t k_off, int64_t lds,
                       GemmAccOutDtype out_dtype, void* stream) {
    if (M <= 0) return;
    if (N <= 0 || K <= 0)
        throw std::invalid_argument("launch_wq_gemm_nt: N and K must be > 0");
    if (lda < K)
        throw std::invalid_argument("launch_wq_gemm_nt: lda must be >= K");
    if (k_off < 0 || ldw < k_off + K)
        throw std::invalid_argument(
            "launch_wq_gemm_nt: ldw must cover k_off + K");
    if (!Wq || !scales)
        throw std::invalid_argument(
            "launch_wq_gemm_nt: null weight/scales pointer");

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    switch (kind) {
        case WqWeightKind::kFp8E4M3: {
            Fp8Loader wl{static_cast<const uint8_t*>(Wq),
                         static_cast<const float*>(scales), ldw, lds, k_off};
            dispatch_wq_out(C, A, wl, M, N, K, lda, out_dtype, cuda_stream);
            break;
        }
        case WqWeightKind::kNvfp4: {
            if (ldw % 2 != 0)
                throw std::invalid_argument(
                    "launch_wq_gemm_nt: NVFP4 ldw must be even");
            Nvfp4Loader wl{static_cast<const uint8_t*>(Wq),
                           static_cast<const uint8_t*>(scales), ldw / 2, lds,
                           k_off};
            dispatch_wq_out(C, A, wl, M, N, K, lda, out_dtype, cuda_stream);
            break;
        }
        default:
            throw std::runtime_error("launch_wq_gemm_nt: unknown weight kind");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_wq_gemm_nt failed: ") +
                                 cudaGetErrorString(err));
}

}  // namespace layerstorm::compute

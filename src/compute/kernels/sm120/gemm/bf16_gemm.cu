// Self-contained BF16/FP16/FP32 dense GEMM kernels for SM120 — cuBLAS-free.
// See bf16_gemm.h for the contract. All paths accumulate in FP32 to match the
// cublasGemmEx CUBLAS_COMPUTE_32F numerics they replace.

#include "compute/kernels/sm120/gemm/bf16_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>   // LS_CHUNK_SMALLM gate (read once)
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

constexpr int WARP_SIZE = 32;

// ── Element load → float (type-specialized) ─────────────────────────────────

__forceinline__ __device__ float load_f(const float* p, int i)        { return p[i]; }
__forceinline__ __device__ float load_f(const __nv_bfloat16* p, int i) { return __bfloat162float(p[i]); }
__forceinline__ __device__ float load_f(const __half* p, int i)        { return __half2float(p[i]); }

// ── Output store from float (type-specialized) ──────────────────────────────

__forceinline__ __device__ void store_f(float* p, float v)        { *p = v; }
__forceinline__ __device__ void store_f(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__forceinline__ __device__ void store_f(__half* p, float v)        { *p = __float2half_rn(v); }

// ── M == 1 GEMV: C[0, n] = sum_k A[k] * W[n, k] ─────────────────────────────
// One warp per output column n. Lanes stride over K; FP32 accumulate, warp
// reduce. Bandwidth-bound on W [N, K]. WARPS_PER_BLOCK output rows per block.

constexpr int GEMV_WARPS_PER_BLOCK = 8;
constexpr int GEMV_BLOCK = GEMV_WARPS_PER_BLOCK * WARP_SIZE;  // 256

template <typename InT, typename OutT>
__global__ void __launch_bounds__(GEMV_BLOCK)
bf16_gemv_nt_kernel(OutT* __restrict__ C, const InT* __restrict__ A,
                    const InT* __restrict__ W, int N, int K, int64_t ldw) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n = blockIdx.x * GEMV_WARPS_PER_BLOCK + warp;
    if (n >= N) return;

    const InT* Wrow = W + static_cast<size_t>(n) * ldw;

    float acc = 0.0f;
    for (int k = lane; k < K; k += WARP_SIZE)
        acc = fmaf(load_f(A, k), load_f(Wrow, k), acc);

    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) store_f(&C[n], acc);
}

// ── Small-M multi-row GEMV: C[M, N] = A[M, K] @ W[N, K]^T, 1 < M ≤ 32 ───────
// DEFAULT ON since 2026-08-17 (TD-CHUNK-SMALLM-DEFAULT resolved; inverse flag
// LS_NO_CHUNK_SMALLM=1 disables): the tiled kernel below launches only
// ⌈N/64⌉·⌈M/64⌉ CTAs — for the speculative-verify chunk shapes (M = chunk
// rows ≤ 32, N = n_experts/small projections) that is 1-4 CTAs on a 170-SM
// GPU and was measured at ~1.3 ms for the fused-gate router projection
// (M=16, N≈256, K=6144) — ~40% of the whole per-layer verify-chunk attention
// wall. This kernel keeps the M == 1 GEMV structure — one warp per output
// column, lanes stride K, shfl reduce — and amortizes the W-row read over an
// MT-row register tile (the gguf_mmvq pattern). Per output row the FP32
// accumulation order is IDENTICAL to the M == 1 GEMV kernel (lane-strided K,
// shfl_down reduce), i.e. bit-equal to calling that kernel once per row —
// but NOT bit-equal to the tiled kernel it replaces, hence the escape hatch.

constexpr int GEMV_MROWS_MT = 8;  // activation rows per register pass

template <typename InT, typename OutT>
__global__ void __launch_bounds__(GEMV_BLOCK)
bf16_gemv_nt_mrows_kernel(OutT* __restrict__ C, const InT* __restrict__ A,
                          const InT* __restrict__ W, int M, int N, int K,
                          int64_t lda, int64_t ldw) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n = blockIdx.x * GEMV_WARPS_PER_BLOCK + warp;
    if (n >= N) return;

    const InT* Wrow = W + static_cast<size_t>(n) * ldw;

    for (int m0 = 0; m0 < M; m0 += GEMV_MROWS_MT) {
        const int mt = min(M - m0, GEMV_MROWS_MT);
        float acc[GEMV_MROWS_MT];
        #pragma unroll
        for (int j = 0; j < GEMV_MROWS_MT; ++j) acc[j] = 0.0f;

        for (int k = lane; k < K; k += WARP_SIZE) {
            const float w = load_f(Wrow, k);
            #pragma unroll
            for (int j = 0; j < GEMV_MROWS_MT; ++j) {
                if (j < mt) {
                    acc[j] = fmaf(
                        load_f(A + static_cast<size_t>(m0 + j) * lda, k), w,
                        acc[j]);
                }
            }
        }

        #pragma unroll
        for (int j = 0; j < GEMV_MROWS_MT; ++j) {
            if (j < mt) {
                float v = acc[j];
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, off);
                if (lane == 0)
                    store_f(&C[static_cast<size_t>(m0 + j) * N + n], v);
            }
        }
    }
}

// Route 1 < M <= kSmallMMax through the multi-row GEMV. DEFAULT ON (read
// once). Escape hatch: LS_NO_CHUNK_SMALLM=1 (any nonzero) restores the tiled
// kernel for every M > 1 — the byte-identical pre-2026-08-17 behaviour.
// Legacy test/recipe interface: an explicit LS_CHUNK_SMALLM=0 is still
// honoured as a disable (LS_CHUNK_SMALLM=1/unset now both mean ON, so old
// recipes that set it keep working unchanged). LS_NO_CHUNK_SMALLM wins if
// both are set. Same rule in dcp_executor.cpp route_gguf_gemm (mmvq
// crossover) — keep the two in lockstep.
constexpr int GEMV_MROWS_MAX_M = 32;
bool small_m_gemv_enabled() {
    static const bool on = [] {
        const char* no = std::getenv("LS_NO_CHUNK_SMALLM");
        if (no && *no && no[0] != '0') return false;
        const char* e = std::getenv("LS_CHUNK_SMALLM");
        if (e && *e && e[0] == '0') return false;
        return true;
    }();
    return on;
}

// ── M > 1 tiled GEMM: C[M, N] = A[M, K] @ W[N, K]^T ─────────────────────────
// Register-blocked: each thread owns a TM×TN micro-tile of C. Block computes a
// (TM*BM)×(TN*BN) output tile; K is streamed in TK chunks staged through shared
// memory. FP32 accumulate. Generic over input/output element type.

constexpr int BM = 16;   // threads in M
constexpr int BN = 16;   // threads in N
constexpr int TM = 4;    // C rows per thread
constexpr int TN = 4;    // C cols per thread
constexpr int TILE_M = BM * TM;  // 64
constexpr int TILE_N = BN * TN;  // 64
constexpr int TK = 16;           // K chunk staged in shared

template <typename InT, typename OutT>
__global__ void __launch_bounds__(BM * BN)
bf16_gemm_nt_kernel(OutT* __restrict__ C, const InT* __restrict__ A,
                    const InT* __restrict__ W, int M, int N, int K,
                    int64_t lda, int64_t ldw) {
    __shared__ float sA[TILE_M][TK];
    __shared__ float sW[TILE_N][TK];

    const int tid = threadIdx.y * BN + threadIdx.x;  // 0..BM*BN-1
    const int threads = BM * BN;

    const int row0 = blockIdx.y * TILE_M;  // first C row of this block
    const int col0 = blockIdx.x * TILE_N;  // first C col of this block

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        // Stage A tile [TILE_M, TK] and W tile [TILE_N, TK] into shared (FP32).
        for (int idx = tid; idx < TILE_M * TK; idx += threads) {
            const int r = idx / TK;
            const int c = idx % TK;
            const int gr = row0 + r;
            const int gk = k0 + c;
            sA[r][c] = (gr < M && gk < K)
                ? load_f(A + static_cast<size_t>(gr) * lda, gk) : 0.0f;
        }
        for (int idx = tid; idx < TILE_N * TK; idx += threads) {
            const int r = idx / TK;
            const int c = idx % TK;
            const int gn = col0 + r;
            const int gk = k0 + c;
            sW[r][c] = (gn < N && gk < K)
                ? load_f(W + static_cast<size_t>(gn) * ldw, gk) : 0.0f;
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
            store_f(&C[static_cast<size_t>(gr) * N + gc], acc[i][j]);
        }
    }
}

// ── Strided batched GEMM (kv_b_v) ───────────────────────────────────────────
// Per batch b: C[n*ldc + m] = sum_k A[m*lda + k] * B[n*ldb + k], with per-batch
// base offsets strideA/strideB/strideC. FP32 accumulate. The decode-relevant
// shapes are m=128..1024, n=batch(=1 at decode), k=512..6144.
//
// SUPEROPT (fab2, 2026-07-12): the previous layout — ONE block of m threads
// per (n, batch), each thread serially walking all k — launched a single CTA
// at decode (n=1, batch=1), leaving 169/170 SMs idle and running ~95 µs/call
// (≈9.4 ms/token/rank across the indexer + kv_b_v call sites: the #2 GPU
// consumer after the GGUF attention mmvq). Now: one WARP per output row, lanes
// stride k (coalesced on both A row and B col), warp shuffle-reduce, 8 rows
// per CTA → grid (⌈m/8⌉, n, batch) — m=128 gives 16 CTAs instead of 1.
// NUMERICS: FP32 accumulation ORDER changes (lane-strided + shuffle reduce vs
// one thread's serial k walk) — still run-to-run deterministic for a fixed
// shape, but not bit-identical to the old kernel (goldens are argmax/tolerance
// -based and unaffected). The m<=1024 launcher limit is gone (grid.x covers
// any m).

constexpr int SBGEMV_WARPS_PER_BLOCK = 8;
constexpr int SBGEMV_BLOCK = SBGEMV_WARPS_PER_BLOCK * WARP_SIZE;  // 256

template <typename T>
__global__ void __launch_bounds__(SBGEMV_BLOCK)
bf16_strided_batched_gemv_kernel(
    T* __restrict__ C, const T* __restrict__ A, const T* __restrict__ B,
    int m, int n, int k,
    int lda, int64_t strideA,
    int ldb, int64_t strideB,
    int ldc, int64_t strideC) {

    // Grid: x = row tile (SBGEMV_WARPS_PER_BLOCK rows), y = n col, z = batch.
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = blockIdx.x * SBGEMV_WARPS_PER_BLOCK + warp;  // m index
    const int col  = blockIdx.y;                                  // n index
    const int b    = blockIdx.z;                                  // batch index
    if (row >= m || col >= n) return;

    const T* Ab = A + b * strideA;       // [m, k] row-major, lda=k
    const T* Bb = B + b * strideB;       // [n, k] with leading dim ldb
    T* Cb       = C + b * strideC;       // [n, m] with leading dim ldc

    const T* Arow = Ab + static_cast<size_t>(row) * lda;
    const T* Bcol = Bb + static_cast<size_t>(col) * ldb;

    float acc = 0.0f;
    for (int kk = lane; kk < k; kk += WARP_SIZE)
        acc = fmaf(load_f(Arow, kk), load_f(Bcol, kk), acc);

    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0)
        store_f(&Cb[static_cast<size_t>(col) * ldc + row], acc);
}

// ── Dispatch helpers ────────────────────────────────────────────────────────

template <typename InT, typename OutT>
void dispatch_gemm_nt(void* C, const void* A, const void* W,
                      int M, int N, int K, int64_t lda, int64_t ldw,
                      cudaStream_t stream) {
    auto* c = static_cast<OutT*>(C);
    auto* a = static_cast<const InT*>(A);
    auto* w = static_cast<const InT*>(W);
    if (M == 1) {
        dim3 grid((N + GEMV_WARPS_PER_BLOCK - 1) / GEMV_WARPS_PER_BLOCK);
        dim3 block(GEMV_BLOCK);
        bf16_gemv_nt_kernel<InT, OutT><<<grid, block, 0, stream>>>(c, a, w, N, K, ldw);
    } else if (M <= GEMV_MROWS_MAX_M && small_m_gemv_enabled()) {
        // Small-M multi-row GEMV, default ON (see kernel comment).
        dim3 grid((N + GEMV_WARPS_PER_BLOCK - 1) / GEMV_WARPS_PER_BLOCK);
        dim3 block(GEMV_BLOCK);
        bf16_gemv_nt_mrows_kernel<InT, OutT><<<grid, block, 0, stream>>>(
            c, a, w, M, N, K, lda, ldw);
    } else {
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        dim3 block(BN, BM);
        bf16_gemm_nt_kernel<InT, OutT><<<grid, block, 0, stream>>>(c, a, w, M, N, K, lda, ldw);
    }
}

template <typename OutT>
void dispatch_gemm_nt_out(void* C, const void* A, const void* W,
                          int M, int N, int K, int64_t lda, int64_t ldw,
                          GemmInDtype in_dtype, cudaStream_t stream) {
    switch (in_dtype) {
        case GemmInDtype::kFloat32:
            dispatch_gemm_nt<float, OutT>(C, A, W, M, N, K, lda, ldw, stream); break;
        case GemmInDtype::kBFloat16:
            dispatch_gemm_nt<__nv_bfloat16, OutT>(C, A, W, M, N, K, lda, ldw, stream); break;
        case GemmInDtype::kFloat16:
            dispatch_gemm_nt<__half, OutT>(C, A, W, M, N, K, lda, ldw, stream); break;
        default:
            throw std::runtime_error("launch_bf16_gemm_nt: unsupported input dtype");
    }
}

void gemm_nt_strided_impl(void* C, const void* A, const void* W,
                          int M, int N, int K, int64_t lda, int64_t ldw,
                          GemmInDtype in_dtype, GemmAccOutDtype out_dtype,
                          void* stream, const char* who) {
    if (M <= 0) return;
    if (N <= 0)
        throw std::invalid_argument(std::string(who) + ": N must be > 0");
    if (K <= 0)
        throw std::invalid_argument(std::string(who) + ": K must be > 0");
    if (lda < K || ldw < K)
        throw std::invalid_argument(std::string(who) +
                                    ": leading dims must be >= K");

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    switch (out_dtype) {
        case GemmAccOutDtype::kFloat32:
            dispatch_gemm_nt_out<float>(C, A, W, M, N, K, lda, ldw, in_dtype, cuda_stream); break;
        case GemmAccOutDtype::kBFloat16:
            dispatch_gemm_nt_out<__nv_bfloat16>(C, A, W, M, N, K, lda, ldw, in_dtype, cuda_stream); break;
        case GemmAccOutDtype::kFloat16:
            dispatch_gemm_nt_out<__half>(C, A, W, M, N, K, lda, ldw, in_dtype, cuda_stream); break;
        default:
            throw std::runtime_error(std::string(who) + ": unsupported output dtype");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(who) + " failed: " +
                                 cudaGetErrorString(err));
}

}  // namespace

void launch_bf16_gemm_nt(void* C, const void* A, const void* W,
                         int M, int N, int K,
                         GemmInDtype in_dtype, GemmAccOutDtype out_dtype,
                         void* stream) {
    gemm_nt_strided_impl(C, A, W, M, N, K, /*lda=*/K, /*ldw=*/K, in_dtype,
                         out_dtype, stream, "launch_bf16_gemm_nt");
}

void launch_bf16_gemm_nt_strided(void* C, const void* A, const void* W,
                                 int M, int N, int K, int64_t lda, int64_t ldw,
                                 GemmInDtype in_dtype,
                                 GemmAccOutDtype out_dtype, void* stream) {
    gemm_nt_strided_impl(C, A, W, M, N, K, lda, ldw, in_dtype, out_dtype,
                         stream, "launch_bf16_gemm_nt_strided");
}

void launch_bf16_strided_batched_gemm_nt(
    void* C, const void* A, const void* B,
    int m, int n, int k,
    int lda, int64_t strideA,
    int ldb, int64_t strideB,
    int ldc, int64_t strideC,
    int batch_count,
    void* stream) {
    if (m <= 0 || n <= 0 || k <= 0 || batch_count <= 0) return;

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    // One warp per output row; lanes stride k (see the kernel comment).
    dim3 grid((m + SBGEMV_WARPS_PER_BLOCK - 1) / SBGEMV_WARPS_PER_BLOCK,
              n, batch_count);
    dim3 block(SBGEMV_BLOCK);
    bf16_strided_batched_gemv_kernel<__nv_bfloat16><<<grid, block, 0, cuda_stream>>>(
        static_cast<__nv_bfloat16*>(C),
        static_cast<const __nv_bfloat16*>(A),
        static_cast<const __nv_bfloat16*>(B),
        m, n, k, lda, strideA, ldb, strideB, ldc, strideC);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("launch_bf16_strided_batched_gemm_nt failed: ") +
            cudaGetErrorString(err));
}

}  // namespace layerstorm::compute

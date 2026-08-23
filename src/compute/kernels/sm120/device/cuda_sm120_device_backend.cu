// CUDA SM120 DeviceBackend implementation.
//
// Method bodies extracted from cuda_dcp_executor_backend() lambdas
// in compute/kernels/smxx/parallelism/dcp_executor_cuda.cu.

#include "compute/cuda_sm120_device_backend.h"
#include "core/attention_device.h"
#include "compute/kernels/smxx/quant/kv_bv_extract_dequant.h"
#include "compute/kernels/attention/mla_attention.h"
#include "sm120/prep/q_absorb.h"
#include "sm120/prep/rope_rotate.h"
#include "compute/kernels/sm120/indexer/indexer_hadamard.h"
#include "compute/kernels/sm120/indexer/indexer_prep.h"
#include "compute/kernels/sm120/indexer/indexer_shard_translate.h"
#include "compute/kernels/sm120/indexer/lightning_indexer.h"

#include "sm120/gemm/fp8/fp8_gemm.h"
#include "sm120/gemm/grouped_gemm.h"
#include "sm120/gemm/gguf/gguf_mmvq.h"          // launch_gguf_mmvq, workspace
#include "sm120/gemm/gguf/mmq_mma/mmq_mma.h"     // launch_gguf_mmq_mma (fast int8 TC)
#include "sm120/gemm/gguf/gguf_dequant_gemm.h"   // launch_gguf_dequant_gemm, GgufType
#include "compute/kernels/sm120/gemm/bf16_gemm.h"
#include "compute/kernels/norm/rmsnorm.h"
#include "smxx/quant/dynamic_fp8_quant.h"
#include "smxx/quant/weight_fp8_quant.h"
#include "smxx/quant/nvfp4_dequant_bf16.h"
#include "smxx/quant/bf16_to_nvfp4_grouped.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

CudaSm120DeviceBackend::CudaSm120DeviceBackend(config::GpuRef gpu)
    : gpu_(gpu) {}

CudaSm120DeviceBackend::~CudaSm120DeviceBackend() = default;

void CudaSm120DeviceBackend::set_device() {
    cudaError_t err = cudaSetDevice(gpu_.id);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaSetDevice failed for GPU ") +
            std::to_string(gpu_.id) + ": " + cudaGetErrorString(err));
    }
}

void CudaSm120DeviceBackend::synchronize_device() {
    cudaSetDevice(gpu_.id);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // Log but don't throw — we're in shutdown, need to continue cleanup.
        // Note: cudaErrorIllegalAddress (700) is non-recoverable without
        // cudaDeviceReset, but we can't reset during shutdown (other GPUs
        // still have valid state).  The error is surfaced here for debugging.
        // INV-GPU-2: .cu files must not link spdlog — use fprintf for error logging.
        fprintf(stderr, "synchronize_device GPU %d: %s (%d)\n",
                gpu_.id, cudaGetErrorString(err), static_cast<int>(err));
    }
    cudaGetLastError();  // clear last error for subsequent CUDA calls
}

int CudaSm120DeviceBackend::peek_last_error() {
    cudaSetDevice(gpu_.id);
    return static_cast<int>(cudaPeekAtLastError());
}

const config::GpuRef& CudaSm120DeviceBackend::gpu() const {
    return gpu_;
}

void CudaSm120DeviceBackend::gemm(const Fp8GemmParams& params,
                                   void* workspace, void* stream) {
    launch_fp8_gemm(params, workspace, stream);
}

// ── GGUF linear GEMMs (attention projections, GG-4) ─────────────────────────
//
// The engine-side GgufGemmParams is CUDA-free (model::GgufKQuantType, void*
// pointers). Convert it here to the kernel's GgufMmvqParams/GgufDequantGemmParams
// and map model::GgufKQuantType → compute::GgufType. Both enums share the SAME
// value order (Q2_K=0 .. Q8_0=5; see gguf_kquant.h vs gguf_dequant_gemm.h), so
// the cast is value-preserving (asserted via a static_cast static_assert below
// would require the enum in scope — verified by GG-4 routing tests instead).

namespace {

GgufType to_kernel_gguf_type(model::GgufKQuantType t) {
    return static_cast<GgufType>(static_cast<int>(t));
}

GgufMmvqParams to_mmvq_params(const GgufGemmParams& p) {
    GgufMmvqParams k{};
    k.M = p.M;
    k.N = p.N;
    k.K = p.K;
    k.A = static_cast<const __nv_bfloat16*>(p.A);
    k.B = p.B;
    k.C = static_cast<__nv_bfloat16*>(p.C);
    k.type = to_kernel_gguf_type(p.type);
    return k;
}

}  // namespace

void CudaSm120DeviceBackend::gguf_mmvq(const GgufGemmParams& params,
                                        void* q8_1_workspace, void* stream) {
    launch_gguf_mmvq(to_mmvq_params(params), q8_1_workspace,
                     static_cast<cudaStream_t>(stream));
}

void CudaSm120DeviceBackend::gguf_mmq(const GgufGemmParams& params,
                                       void* q8_1_workspace, void* stream) {
    launch_gguf_mmq_mma(to_mmvq_params(params), q8_1_workspace,
                        static_cast<cudaStream_t>(stream));
}

void CudaSm120DeviceBackend::gguf_dequant_gemm(const GgufGemmParams& params,
                                                void* stream) {
    GgufDequantGemmParams k{};
    k.M = params.M;
    k.N = params.N;
    k.K = params.K;
    k.A = static_cast<const __nv_bfloat16*>(params.A);
    k.B = params.B;
    k.C = static_cast<__nv_bfloat16*>(params.C);
    k.type = to_kernel_gguf_type(params.type);
    launch_gguf_dequant_gemm(k, static_cast<cudaStream_t>(stream));
}

void CudaSm120DeviceBackend::rmsnorm(void* out, const void* input,
                                      const void* weight, float eps,
                                      int num_tokens, int hidden_size,
                                      int row_stride, void* stream) {
    launch_rmsnorm_strided(out, input, weight, eps, num_tokens, hidden_size,
                           row_stride, NormDtype::kBFloat16, stream);
}

void CudaSm120DeviceBackend::quantize_fp8(const DynamicFp8QuantParams& params,
                                            void* stream) {
    launch_dynamic_fp8_quant(params, stream);
}

void CudaSm120DeviceBackend::weight_quantize_fp8(const WeightFp8QuantParams& params,
                                                   void* stream) {
    launch_weight_fp8_quant(params, stream);
}

void CudaSm120DeviceBackend::nvfp4_dequant_bf16(const Nvfp4DequantBf16Params& params,
                                                  void* stream) {
    launch_nvfp4_dequant_bf16(params, stream);
}

void CudaSm120DeviceBackend::nvfp4_grouped_gemm(
    const Nvfp4GroupedGemmParams& params,
    void* workspace, size_t workspace_bytes,
    void* stream) {
    launch_nvfp4_grouped_gemm(params, workspace, workspace_bytes, stream);
}

void CudaSm120DeviceBackend::bf16_to_nvfp4_grouped(
    const Bf16ToNvfp4GroupedParams& params, void* stream) {
    launch_bf16_to_nvfp4_grouped(params, stream);
}

// ── FP8 V-extract dequant (kv_b_v) ─────────────────────────────────────────

void CudaSm120DeviceBackend::kv_bv_extract_dequant(
    const KvBvExtractDequantParams& params, void* stream) {
    launch_kv_bv_extract_dequant(params, stream);
}

// ── MLA query W_UK absorption (q_absorb) ───────────────────────────────────

void CudaSm120DeviceBackend::absorb_q(const QAbsorbParams& p, void* stream) {
    sm120::prep::QAbsorbParams kp;
    kp.q_heads      = static_cast<const __nv_bfloat16*>(p.q_heads);
    kp.w_uk         = p.kv_b_proj;
    kp.w_uk_scales  = static_cast<const float*>(p.kv_b_proj_scales);
    kp.q_absorbed   = static_cast<__nv_bfloat16*>(p.q_absorbed);
    kp.s_q          = p.s_q;
    kp.h_q          = p.h_q;
    kp.d_nope_in    = p.d_nope_in;
    kp.d_c          = p.d_c;
    kp.d_rope       = p.d_rope;
    kp.d_v          = p.d_v;
    kp.weight_is_fp8 = p.weight_is_fp8;
    // GG-7: GGUF W_UK dequant-only branch. -1 leaves the BF16/FP8 path unchanged.
    // The kernel ignores w_uk_scales for GGUF (scales are in-block); the caller
    // passes kv_b_proj_scales == nullptr for the GGUF path.
    kp.gguf_type    = p.gguf_type;
    kp.apply_rope   = p.apply_rope;
    kp.seqlens_k    = static_cast<const int*>(p.seqlens_k);
    kp.cos_sin      = static_cast<const float*>(p.cos_sin);
    kp.max_pos      = p.max_pos;
    launch_q_absorb(kp, static_cast<cudaStream_t>(stream));
}

// ── In-place strided RoPE rotation (k_pe) ──────────────────────────────────

void CudaSm120DeviceBackend::rope_rotate(const RopeRotateParams& p, void* stream) {
    sm120::prep::RopeRotateParams kp;
    kp.x              = static_cast<__nv_bfloat16*>(p.x);
    kp.seqlens_k      = static_cast<const int*>(p.seqlens_k);
    kp.cos_sin        = static_cast<const float*>(p.cos_sin);
    kp.num_tokens     = p.num_tokens;
    kp.rows_per_token = p.rows_per_token;
    kp.row_stride     = p.row_stride;
    kp.d_rope         = p.d_rope;
    kp.max_pos        = p.max_pos;
    launch_rope_rotate(kp, static_cast<cudaStream_t>(stream));
}

// ── DSA lightning-indexer producer ops (GLM-25a) ───────────────────────────

void CudaSm120DeviceBackend::indexer_layernorm_bias(
    void* x, const void* weight, const void* bias, int num_rows, int dim,
    float eps, void* stream) {
    layerstorm::compute::IndexerLayerNormParams p{};
    p.x = static_cast<__nv_bfloat16*>(x);
    p.weight = static_cast<const __nv_bfloat16*>(weight);
    p.bias = static_cast<const __nv_bfloat16*>(bias);
    p.num_rows = num_rows; p.dim = dim; p.eps = eps;
    layerstorm::compute::launch_indexer_layernorm_bias(
        p, static_cast<cudaStream_t>(stream));
}

void CudaSm120DeviceBackend::indexer_hadamard(
    void* x, int rows, int dim, void* stream) {
    layerstorm::compute::launch_indexer_hadamard(
        x, rows, dim, static_cast<cudaStream_t>(stream));
}

void CudaSm120DeviceBackend::indexer_k_quant_append(
    const void* k_in, const void* slot_mapping, void* k_cache, void* k_scales,
    int num_tokens, int index_head_dim, int slot_bias, void* stream) {
    layerstorm::compute::IndexerKQuantAppendParams p{};
    p.k_in = static_cast<const __nv_bfloat16*>(k_in);
    p.slot_mapping = static_cast<const int*>(slot_mapping);
    p.k_cache = static_cast<__nv_fp8_e4m3*>(k_cache);
    p.k_scales = static_cast<float*>(k_scales);
    p.num_tokens = num_tokens; p.index_head_dim = index_head_dim;
    p.slot_bias = slot_bias;
    layerstorm::compute::launch_indexer_k_quant_append(
        p, static_cast<cudaStream_t>(stream));
}

void CudaSm120DeviceBackend::indexer_scale_weights(
    const void* in, void* out, int num_rows, int n, float scale, void* stream) {
    layerstorm::compute::IndexerScaleWeightsParams p{};
    p.in = static_cast<const __nv_bfloat16*>(in);
    p.out = static_cast<float*>(out);
    p.num_rows = num_rows; p.n = n; p.scale = scale;
    layerstorm::compute::launch_indexer_scale_weights(
        p, static_cast<cudaStream_t>(stream));
}

void CudaSm120DeviceBackend::indexer_score_topk(
    const IndexerScoreTopkArgs& a, void* stream) {
    auto cs = static_cast<cudaStream_t>(stream);
    const int NIH = a.n_heads, IHD = a.head_dim, ITK = a.topk, NB = a.num_blocks;
    for (int t = 0; t < a.num_tokens; ++t) {
        const __nv_bfloat16* q = static_cast<const __nv_bfloat16*>(a.q_all)
                                 + static_cast<size_t>(t) * NIH * IHD;
        const float* w = static_cast<const float*>(a.score_proj_all)
                         + static_cast<size_t>(t) * NIH;

        if (a.k_pages && a.page_tokens > 0) {
            // TD-GLM-INDEXER-PAGED: coarse pool pages — one score launch per
            // page over its contiguous K range; scales sit at the page tail
            // (base + page_tokens*head_dim). Top-k below is unchanged: it
            // reads the full [NB] scores_scratch.
            for (int p = 0; p * a.page_tokens < NB && p < a.num_k_pages; ++p) {
                const auto* base =
                    static_cast<const __nv_fp8_e4m3*>(a.k_pages[p]);
                sm120::indexer::LightningScoreMqaParams sp{};
                sp.q_proj = q;
                sp.indexer_k_cache = base;
                sp.k_scales = reinterpret_cast<const float*>(
                    base + static_cast<size_t>(a.page_tokens) * IHD);
                sp.score_proj = w;
                sp.scores_out = static_cast<float*>(a.scores_scratch)
                                + static_cast<size_t>(p) * a.page_tokens;
                sp.num_blocks = std::min(a.page_tokens, NB - p * a.page_tokens);
                sp.index_n_heads = NIH; sp.index_head_dim = IHD;
                layerstorm::compute::launch_lightning_score_mqa(sp, cs);
            }
        } else {
            sm120::indexer::LightningScoreMqaParams sp{};
            sp.q_proj = q;
            sp.indexer_k_cache = static_cast<const __nv_fp8_e4m3*>(a.k_cache);
            sp.k_scales = static_cast<const float*>(a.k_scales);
            sp.score_proj = w;
            sp.scores_out = static_cast<float*>(a.scores_scratch);
            sp.num_blocks = NB; sp.index_n_heads = NIH; sp.index_head_dim = IHD;
            layerstorm::compute::launch_lightning_score_mqa(sp, cs);
        }

        sm120::indexer::LightningTopkParams tp{};
        tp.scores = static_cast<const float*>(a.scores_scratch);
        tp.block_endpoints = static_cast<const int*>(a.block_endpoints);
        tp.output_indices = static_cast<int*>(a.sparse_indices_out)
                            + static_cast<size_t>(t) * ITK;
        tp.output_scores = static_cast<float*>(a.topk_scores_scratch);
        tp.effective_k_out = static_cast<int*>(a.topk_lengths_out) + t;
        tp.num_blocks = NB; tp.topk = ITK;
        tp.query_position = a.query_position_base + t;
        layerstorm::compute::launch_lightning_topk(tp, cs);
    }
}

// TD-SPARSE-PREFILL-SCORE-BATCH: batch the retired per-row loop above into
// ONE score launch + ONE top-k launch covering all rows, each row with its
// own device-array bound (row_num_blocks) and cutoff (row_query_position).
// Bit-identical to per-row indexer_score_topk calls: the batched kernels run
// the exact single-query device bodies per (row, block) / per row CTA.
void CudaSm120DeviceBackend::indexer_score_topk_batched(
    const IndexerScoreTopkBatchedArgs& a, void* stream) {
    auto cs = static_cast<cudaStream_t>(stream);

    if (a.max_num_blocks > 0) {
        sm120::indexer::LightningScoreMqaBatchedParams sp{};
        sp.q_all = static_cast<const __nv_bfloat16*>(a.q_all);
        sp.score_proj_all = static_cast<const float*>(a.score_proj_all);
        sp.row_num_blocks = static_cast<const int*>(a.row_num_blocks);
        sp.k_cache = static_cast<const __nv_fp8_e4m3*>(a.k_cache);
        sp.k_scales = static_cast<const float*>(a.k_scales);
        sp.k_page_table = static_cast<const void* const*>(a.k_page_table);
        sp.page_table_stride = a.page_table_stride;
        sp.page_tokens = a.page_tokens;
        sp.scores_out = static_cast<float*>(a.scores_scratch);
        sp.scores_stride = a.scores_stride;
        sp.num_rows = a.num_rows;
        sp.max_num_blocks = a.max_num_blocks;
        sp.index_n_heads = a.n_heads;
        sp.index_head_dim = a.head_dim;
        layerstorm::compute::launch_lightning_score_mqa_batched(sp, cs);
    }

    sm120::indexer::LightningTopkBatchedParams tp{};
    tp.scores = static_cast<const float*>(a.scores_scratch);
    tp.block_endpoints = static_cast<const int*>(a.block_endpoints);
    tp.row_num_blocks = static_cast<const int*>(a.row_num_blocks);
    tp.row_query_position = static_cast<const int*>(a.row_query_position);
    tp.output_indices = static_cast<int*>(a.sparse_indices_out);
    tp.output_scores = static_cast<float*>(a.topk_scores_out);
    tp.effective_k_out = static_cast<int*>(a.topk_lengths_out);
    tp.scores_stride = a.scores_stride;
    tp.num_rows = a.num_rows;
    tp.topk = a.topk;
    layerstorm::compute::launch_lightning_topk_batched(tp, cs);
}

void CudaSm120DeviceBackend::indexer_shard_translate(
    const void* global_indices, const void* global_lengths,
    void* local_indices, void* local_lengths,
    int num_tokens, int topk, int chunk_tokens, int dcp_size, int rank,
    void* stream) {
    layerstorm::compute::IndexerShardTranslateParams p{};
    p.global_indices = static_cast<const int*>(global_indices);
    p.global_lengths = static_cast<const int*>(global_lengths);
    p.local_indices  = static_cast<int*>(local_indices);
    p.local_lengths  = static_cast<int*>(local_lengths);
    p.num_tokens = num_tokens;
    p.topk = topk;
    p.chunk_tokens = chunk_tokens;
    p.dcp_size = dcp_size;
    p.rank = rank;
    layerstorm::compute::launch_indexer_shard_translate(
        p, static_cast<cudaStream_t>(stream));
}

void CudaSm120DeviceBackend::indexer_topk_merge(
    const IndexerTopkMergeArgs& a, void* stream) {
    sm120::indexer::LightningTopkMergeParams p{};
    p.gathered = a.gathered;
    p.seg_words = a.seg_words;
    p.batch = a.batch;
    p.token = a.token;
    p.scores_scratch = static_cast<float*>(a.scores_scratch);
    p.block_endpoints = static_cast<const int*>(a.block_endpoints);
    p.output_indices = static_cast<int*>(a.indices_out);
    p.output_scores = static_cast<float*>(a.topk_scores_scratch);
    p.effective_k_out = static_cast<int*>(a.length_out);
    p.num_blocks = a.num_blocks;
    p.topk = a.topk;
    p.query_position = a.query_position;
    p.dcp_size = a.dcp_size;
    p.page_tokens = a.page_tokens;
    layerstorm::compute::launch_lightning_topk_merge(
        p, static_cast<cudaStream_t>(stream));
}

// ── BF16 strided batched GEMM (kv_b_v projection) ──────────────────────────
// Was cublasGemmStridedBatchedEx; now a self-contained FP32-accumulate kernel
// (bf16_gemm.cu) — same per-head NT contraction, no cuBLAS handle, fully
// graph-capturable. Per batch: C[n*ldc+m] = sum_k A[m*lda+k] * B[n*ldb+k].

void CudaSm120DeviceBackend::batched_gemm_bf16(
    const StridedBatchedGemmBf16Params& params, void* stream) {

    if (params.m <= 0 || params.n <= 0 || params.k <= 0 || params.batch_count <= 0)
        return;

    launch_bf16_strided_batched_gemm_nt(
        params.C, params.A, params.B,
        params.m, params.n, params.k,
        params.lda, params.strideA,
        params.ldb, params.strideB,
        params.ldc, params.strideC,
        params.batch_count,
        stream);
}

// TODO:DEBT TD-56c: cudaMalloc return code ignored — nullptr on OOM, callers assume non-null
void* CudaSm120DeviceBackend::device_alloc(size_t bytes) {
    void* ptr = nullptr;
    cudaSetDevice(gpu_.id);
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void CudaSm120DeviceBackend::device_free(void* ptr) {
    if (ptr) {
        cudaSetDevice(gpu_.id);
        cudaFree(ptr);
    }
}

void* CudaSm120DeviceBackend::host_alloc_pinned(size_t bytes) {
    void* ptr = nullptr;
    cudaSetDevice(gpu_.id);
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        cudaGetLastError();  // Clear error state
        return nullptr;
    }
    return ptr;
}

void CudaSm120DeviceBackend::host_free_pinned(void* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

void CudaSm120DeviceBackend::device_sync() {
    cudaSetDevice(gpu_.id);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaDeviceSynchronize failed on GPU ") +
            std::to_string(gpu_.id) + ": " + cudaGetErrorString(err));
    }
}

// ── Stream lifecycle ────────────────────────────────────────────────────────

void* CudaSm120DeviceBackend::create_stream() {
    cudaSetDevice(gpu_.id);
    cudaStream_t stream{};
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaStreamCreate failed on GPU ") +
            std::to_string(gpu_.id) + ": " + cudaGetErrorString(err));
    }
    return static_cast<void*>(stream);
}

void* CudaSm120DeviceBackend::create_stream_low_priority() {
    cudaSetDevice(gpu_.id);
    int least = 0, greatest = 0;  // numerically: least = LOWEST priority
    cudaError_t err = cudaDeviceGetStreamPriorityRange(&least, &greatest);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaDeviceGetStreamPriorityRange failed on GPU ") +
            std::to_string(gpu_.id) + ": " + cudaGetErrorString(err));
    }
    cudaStream_t stream{};
    err = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, least);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaStreamCreateWithPriority failed on GPU ") +
            std::to_string(gpu_.id) + ": " + cudaGetErrorString(err));
    }
    return static_cast<void*>(stream);
}

void CudaSm120DeviceBackend::destroy_stream(void* stream) {
    if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

// ── Event lifecycle ─────────────────────────────────────────────────────────

void* CudaSm120DeviceBackend::create_event() {
    cudaSetDevice(gpu_.id);
    cudaEvent_t event{};
    cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaEventCreate failed on GPU ") +
            std::to_string(gpu_.id) + ": " + cudaGetErrorString(err));
    }
    return static_cast<void*>(event);
}

void* CudaSm120DeviceBackend::create_timing_event() {
    // Profiling-only (perf_trace pure-DMA x-ray): timing-ENABLED event so
    // cudaEventElapsedTime works. Returns nullptr on failure (caller skips).
    cudaSetDevice(gpu_.id);
    cudaEvent_t event{};
    if (cudaEventCreate(&event) != cudaSuccess) return nullptr;
    return static_cast<void*>(event);
}

float CudaSm120DeviceBackend::event_elapsed_ms(void* start, void* end) {
    if (!start || !end) return -1.0f;
    float ms = -1.0f;
    if (cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(start),
                             static_cast<cudaEvent_t>(end)) != cudaSuccess)
        return -1.0f;
    return ms;
}

void CudaSm120DeviceBackend::destroy_event(void* event) {
    if (event) cudaEventDestroy(static_cast<cudaEvent_t>(event));
}

void CudaSm120DeviceBackend::record_event(void* event, void* stream) {
    cudaEventRecord(static_cast<cudaEvent_t>(event),
                    static_cast<cudaStream_t>(stream));
}

EventQueryResult CudaSm120DeviceBackend::query_event(void* event) {
    cudaError_t err = cudaEventQuery(static_cast<cudaEvent_t>(event));
    if (err == cudaSuccess) return {EventStatus::kReady, 0};
    if (err == cudaErrorNotReady) return {EventStatus::kNotReady, 0};
    return {EventStatus::kError, static_cast<int>(err)};
}

void CudaSm120DeviceBackend::stream_wait_event(void* stream, void* event) {
    cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),
                        static_cast<cudaEvent_t>(event), 0);
}

// ── Async data transfer ─────────────────────────────────────────────────────

void CudaSm120DeviceBackend::memcpy_h2d_async(void* dst, const void* src,
                                                size_t bytes, void* stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                       static_cast<cudaStream_t>(stream));
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpyAsync H2D failed on GPU ") +
            std::to_string(gpu_.id) + " (" + std::to_string(bytes) +
            " bytes): " + cudaGetErrorString(err));
    }
}

void CudaSm120DeviceBackend::memcpy_d2h_async(void* dst, const void* src,
                                                size_t bytes, void* stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(stream));
}

void CudaSm120DeviceBackend::memcpy_d2d_async(void* dst, const void* src,
                                                size_t bytes, void* stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                    static_cast<cudaStream_t>(stream));
}

// ── Async memset ────────────────────────────────────────────────────────────

void CudaSm120DeviceBackend::memset_async(void* dst, int value,
                                            size_t bytes, void* stream) {
    cudaMemsetAsync(dst, value, bytes, static_cast<cudaStream_t>(stream));
}

// ── Synchronous data transfer ───────────────────────────────────────────────

void CudaSm120DeviceBackend::memcpy_h2d(void* dst, const void* src,
                                          size_t bytes) {
    cudaSetDevice(gpu_.id);
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpy H2D failed on GPU ") +
            std::to_string(gpu_.id) + ": " + cudaGetErrorString(err));
    }
}

void CudaSm120DeviceBackend::memcpy_d2d(void* dst, const void* src,
                                          size_t bytes) {
    cudaSetDevice(gpu_.id);
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpy D2D failed on GPU ") +
            std::to_string(gpu_.id) + ": " + cudaGetErrorString(err));
    }
}

// ── Direction-agnostic async (cudaMemcpyDefault) ────────────────────────────

void CudaSm120DeviceBackend::memcpy_async(void* dst, const void* src,
                                            size_t bytes, void* stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault,
                    static_cast<cudaStream_t>(stream));
}

// ── 2D async memcpy (cudaMemcpy2DAsync) ────────────────────────────────────

void CudaSm120DeviceBackend::memcpy_2d_async(void* dst, size_t dpitch,
                                              const void* src, size_t spitch,
                                              size_t width, size_t height,
                                              void* stream) {
    // cudaMemcpyDefault (UVA-inferred): identical to DeviceToDevice for
    // same-device copies, and additionally legal for CROSS-device device
    // pointers (DSP-3 aux-hidden export: target GPU attn_buf -> draft GPU
    // staging; routed peer-to-peer or through host when P2P is unavailable).
    auto err = cudaMemcpy2DAsync(dst, dpitch, src, spitch,
                                  width, height,
                                  cudaMemcpyDefault,
                                  static_cast<cudaStream_t>(stream));
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "cudaMemcpy2DAsync failed on GPU " +
            std::to_string(gpu_.id) + ": " + cudaGetErrorString(err));
    }
}

// ── Factory ─────────────────────────────────────────────────────────────────

std::unique_ptr<DeviceBackend> make_cuda_sm120_device_backend(config::GpuRef gpu) {
    return std::make_unique<CudaSm120DeviceBackend>(std::move(gpu));
}

}  // namespace layerstorm::compute

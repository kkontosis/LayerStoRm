// CudaSm120DeviceBackend: concrete DeviceBackend for SM120 (RTX 5090/5080).
//
// Header is CUDA-free — implementation in
// compute/kernels/sm120/device/cuda_sm120_device_backend.cu.

#pragma once

#include "core/device_backend.h"

#include <memory>

namespace layerstorm::compute {

// Forward declarations (defined in attention_device.h / kv_bv_extract_dequant.h).
struct StridedBatchedGemmBf16Params;
struct KvBvExtractDequantParams;
struct QAbsorbParams;
struct RopeRotateParams;
struct IndexerScoreTopkArgs;
struct IndexerScoreTopkBatchedArgs;
struct IndexerTopkMergeArgs;
struct GgufGemmParams;

class CudaSm120DeviceBackend final : public DeviceBackend {
public:
    explicit CudaSm120DeviceBackend(config::GpuRef gpu);
    ~CudaSm120DeviceBackend() override;

    void set_device() override;
    void synchronize_device() override;
    int  peek_last_error() override;
    const config::GpuRef& gpu() const override;

    void gemm(const Fp8GemmParams& params,
              void* workspace, void* stream) override;
    void rmsnorm(void* out, const void* input,
                 const void* weight, float eps,
                 int num_tokens, int hidden_size, int row_stride,
                 void* stream) override;
    void quantize_fp8(const DynamicFp8QuantParams& params,
                      void* stream) override;
    void weight_quantize_fp8(const WeightFp8QuantParams& params,
                              void* stream) override;
    void nvfp4_dequant_bf16(const Nvfp4DequantBf16Params& params,
                             void* stream) override;
    void nvfp4_grouped_gemm(
        const Nvfp4GroupedGemmParams& params,
        void* workspace, size_t workspace_bytes,
        void* stream) override;
    void bf16_to_nvfp4_grouped(const Bf16ToNvfp4GroupedParams& params,
                                void* stream) override;

    void kv_bv_extract_dequant(const KvBvExtractDequantParams& params,
                                void* stream);
    void batched_gemm_bf16(const StridedBatchedGemmBf16Params& params,
                            void* stream);
    void absorb_q(const QAbsorbParams& params, void* stream);
    void rope_rotate(const RopeRotateParams& params, void* stream);

    // GLM-25a: DSA lightning-indexer producer device ops.
    void indexer_layernorm_bias(void* x, const void* weight, const void* bias,
                                int num_rows, int dim, float eps, void* stream);
    void indexer_hadamard(void* x, int rows, int dim, void* stream);
    void indexer_k_quant_append(const void* k_in, const void* slot_mapping,
                                void* k_cache, void* k_scales, int num_tokens,
                                int index_head_dim, int slot_bias, void* stream);
    void indexer_scale_weights(const void* in, void* out, int num_rows, int n,
                               float scale, void* stream);
    void indexer_score_topk(const IndexerScoreTopkArgs& args, void* stream);
    // TD-SPARSE-PREFILL-SCORE-BATCH: multi-row score + top-k in ONE launch
    // pair (see AttentionDevice::indexer_score_topk_batched).
    void indexer_score_topk_batched(const IndexerScoreTopkBatchedArgs& args,
                                    void* stream);
    // KVS-4: GLOBAL→LOCAL sparse-index translation for sequence-sharded KV
    // (see AttentionDevice::indexer_shard_translate for the contract).
    void indexer_shard_translate(const void* global_indices,
                                 const void* global_lengths,
                                 void* local_indices, void* local_lengths,
                                 int num_tokens, int topk, int chunk_tokens,
                                 int dcp_size, int rank, void* stream);
    // TD-GLM-INDEXER-LOCAL-MERGE: exact cross-rank top-k merge for
    // dcp_indexer_mode=local (see AttentionDevice::indexer_topk_merge).
    void indexer_topk_merge(const IndexerTopkMergeArgs& args, void* stream);

    // GGUF linear GEMMs (attention projections, GG-4). Concrete-only methods
    // (not in the DeviceBackend interface) — called via the composed device_ by
    // the concrete AttentionDevices, which expose them through the
    // AttentionDevice::gguf_* virtuals.
    void gguf_mmvq(const GgufGemmParams& params,
                   void* q8_1_workspace, void* stream);
    void gguf_mmq(const GgufGemmParams& params,
                  void* q8_1_workspace, void* stream);
    void gguf_dequant_gemm(const GgufGemmParams& params, void* stream);

    void* device_alloc(size_t bytes) override;
    void  device_free(void* ptr) override;
    void* host_alloc_pinned(size_t bytes) override;
    void  host_free_pinned(void* ptr) override;
    void  device_sync() override;

    void* create_stream() override;
    void* create_stream_low_priority() override;
    void  destroy_stream(void* stream) override;

    void* create_event() override;
    void* create_timing_event() override;
    float event_elapsed_ms(void* start, void* end) override;
    void  destroy_event(void* event) override;
    void  record_event(void* event, void* stream) override;
    EventQueryResult query_event(void* event) override;
    void  stream_wait_event(void* stream, void* event) override;

    void memcpy_h2d_async(void* dst, const void* src,
                          size_t bytes, void* stream) override;
    void memcpy_d2h_async(void* dst, const void* src,
                          size_t bytes, void* stream) override;
    void memcpy_d2d_async(void* dst, const void* src,
                          size_t bytes, void* stream) override;

    void memcpy_h2d(void* dst, const void* src, size_t bytes) override;
    void memcpy_d2d(void* dst, const void* src, size_t bytes) override;

    void memset_async(void* dst, int value, size_t bytes, void* stream) override;

    void memcpy_async(void* dst, const void* src,
                      size_t bytes, void* stream) override;
    void memcpy_2d_async(void* dst, size_t dpitch,
                         const void* src, size_t spitch,
                         size_t width, size_t height,
                         void* stream) override;

private:
    config::GpuRef gpu_;
};

/// Factory: creates a CudaSm120DeviceBackend.
std::unique_ptr<DeviceBackend> make_cuda_sm120_device_backend(config::GpuRef gpu);

}  // namespace layerstorm::compute

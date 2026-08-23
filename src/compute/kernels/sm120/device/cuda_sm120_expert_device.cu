// CudaSm120ExpertDevice: concrete ExpertDevice for SM120.
//
// Delegates all methods to existing kernel launcher free functions.
// No new CUDA code — pure delegation layer.

#include "compute/cuda_sm120_expert_device.h"
#include "sm120/gemm/grouped_gemm.h"
#include "sm120/gemm/gguf/gguf_grouped_gemm.h"
#include "smxx/activation/fused_swiglu.h"
#include "smxx/permute/moe_permute.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace layerstorm::compute {

class CudaSm120ExpertDevice final : public ExpertDevice {
public:
    explicit CudaSm120ExpertDevice(config::GpuRef gpu) : gpu_(gpu) {}

    // ── Device selection + identity ─────────────────────────────────────────

    void set_device() override {
        cudaSetDevice(gpu_.id);
    }

    const config::GpuRef& gpu() const override { return gpu_; }

    // ── Grouped GEMM ───────────���────────────────────────────────────────────

    void nvfp4_grouped_gemm(
        const Nvfp4GroupedGemmParams& params,
        void* workspace, size_t workspace_bytes,
        void* stream) override
    {
        launch_nvfp4_grouped_gemm(params, workspace, workspace_bytes, stream);
    }

    void fp8_grouped_gemm(
        const Fp8GroupedGemmParams& params,
        void* workspace, size_t workspace_bytes,
        void* stream) override
    {
        launch_fp8_grouped_gemm(params, workspace, workspace_bytes, stream);
    }

    // GGUF grouped GEMM (quant_mode==2). Bridges the engine's POD params to the
    // kernel-side GgufGroupedGemmParams: the engine GgufQuantType and the kernel
    // GgufType share the SAME canonical value order (Q2_K=0 .. Q8_0=5), and the
    // engine GgufGemmStrategy mirrors the kernel GgufGroupedStrategy, so both map
    // by value. The kernel runs a per-expert dispatch loop over the scattered
    // ExpertCache B_ptrs (mmvq / mmq_mma / dequant by strategy + per-expert M).
    void gguf_grouped_gemm(
        const GgufGroupedGemmParams& params,
        void* workspace, size_t workspace_bytes,
        void* stream) override
    {
        compute::GgufGroupedGemmKernelParams kp{};
        kp.type        = static_cast<GgufType>(static_cast<int>(params.type));
        kp.strategy    = (params.strategy == GgufGemmStrategy::dequant)
                             ? GgufGroupedStrategy::Dequant
                             : GgufGroupedStrategy::Int;
        kp.num_experts = params.num_experts;
        kp.N           = params.N;
        kp.K           = params.K;
        kp.A_base      = static_cast<const __nv_bfloat16*>(params.A_base);
        kp.D_base      = static_cast<__nv_bfloat16*>(params.D_base);
        kp.expert_offsets = params.expert_offsets;
        kp.B_ptrs      = params.B_ptrs;
        // GG-5d: device-fused routing — total_tokens (== expert_offsets[num_experts],
        // host-known at the call site) drives the on-device Q8_1 quant + mmq tile-map,
        // so the int path needs NO host D2H of the offsets (CUDA-graph capturable;
        // resolves TD-GG5-GROUPED-HOST-SYNC).
        //
        // TD-PREFILL-SUPERCHUNK bit-identity A/B knob: LS_GG_FORCE=mmvq|mmq
        // pins the int strategy's kernel pick (default Auto: avg_m =
        // total_tokens/num_experts > 8 → mmq). The Auto pick is BATCH-SIZE
        // dependent, so a superchunk that grows the MoE token batch can flip
        // mmvq→mmq (a different-but-valid reduction order). Pinning the pick
        // isolates the superchunk staging/batching numerics from that kernel
        // flip when validating K>1 ≡ K=1 bit-identity. Diagnostic only —
        // default (unset) is byte-identical to the legacy Auto behavior.
        static const GgufGroupedIntForce gg_force = [] {
            const char* v = std::getenv("LS_GG_FORCE");
            if (v && std::strcmp(v, "mmvq") == 0) return GgufGroupedIntForce::Mmvq;
            if (v && std::strcmp(v, "mmq") == 0)  return GgufGroupedIntForce::Mmq;
            return GgufGroupedIntForce::Auto;
        }();
        if (gg_force != GgufGroupedIntForce::Auto
            && kp.strategy == GgufGroupedStrategy::Int) {
            launch_gguf_grouped_gemm_int_forced(
                kp, params.total_tokens, workspace, workspace_bytes,
                gg_force, static_cast<cudaStream_t>(stream));
            return;
        }
        launch_gguf_grouped_gemm(kp, params.total_tokens, workspace, workspace_bytes,
                                 static_cast<cudaStream_t>(stream));
    }

    // ── Activation ────���─────────────────────────────────────────────────────

    void fused_swiglu(
        void* output, const void* input,
        const FusedSwigluParams& params,
        int elem_size_bytes, void* stream) override
    {
        launch_fused_swiglu(output, input, params, elem_size_bytes, stream);
    }

    // ── Token permutation ───────��───────────────────────────────��───────────

    void moe_permute(
        void* permuted_input, int32_t* expert_offsets,
        int32_t* src_to_dest_map, int32_t* permuted_idx,
        const void* hidden_states, const int32_t* topk_indices,
        int num_tokens, int topk, int hidden_dim,
        int num_experts, int elem_size_bytes,
        void* workspace, void* stream) override
    {
        launch_moe_permute(permuted_input, expert_offsets,
                           src_to_dest_map, permuted_idx,
                           hidden_states, topk_indices,
                           num_tokens, topk, hidden_dim,
                           num_experts, elem_size_bytes,
                           workspace, stream);
    }

    void moe_unpermute(
        void* output, const void* permuted_output,
        const float* topk_weights, const int32_t* src_to_dest_map,
        int num_tokens, int topk, int hidden_dim,
        int elem_size_bytes, void* stream,
        MoeCombineMode combine_mode = MoeCombineMode::kReducedBf16) override
    {
        switch (combine_mode) {
            case MoeCombineMode::kPerSlotBf16:
                launch_moe_unpermute_bf16_perslot(
                    output, permuted_output, topk_weights, src_to_dest_map,
                    num_tokens, topk, hidden_dim, elem_size_bytes, stream);
                return;
            case MoeCombineMode::kPerSlotFp32:
            case MoeCombineMode::kReducedBf16:
                launch_moe_unpermute(
                    output, permuted_output, topk_weights, src_to_dest_map,
                    num_tokens, topk, hidden_dim, elem_size_bytes, stream,
                    /*fp32_output=*/combine_mode == MoeCombineMode::kPerSlotFp32);
                return;
        }
    }

    // ── Device memory ───────────────────────────────────────────────────────

    void* device_alloc(size_t bytes) override {
        void* ptr = nullptr;
        cudaSetDevice(gpu_.id);
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) return nullptr;
        return ptr;
    }

    void device_free(void* ptr) override {
        if (ptr) {
            cudaSetDevice(gpu_.id);
            cudaFree(ptr);
        }
    }

    // TD-PREFILL-SUPERCHUNK: free/total VRAM for the MoE-scratch fail-safe.
    bool device_mem_info(size_t& free_bytes, size_t& total_bytes) override {
        cudaSetDevice(gpu_.id);
        return cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
    }

private:
    config::GpuRef gpu_;
};

// ── Factory ──���──────────────────────���───────────────────────────────────────

std::unique_ptr<ExpertDevice> make_cuda_sm120_expert_device(
        config::GpuRef gpu) {
    return std::make_unique<CudaSm120ExpertDevice>(std::move(gpu));
}

}  // namespace layerstorm::compute

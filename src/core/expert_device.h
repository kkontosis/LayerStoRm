// Abstract interface for per-GPU expert FFN device.
//
// ExpertDevice encapsulates all MoE expert FFN compute: grouped GEMM
// (NVFP4 / FP8), SwiGLU activation, and token permutation/unpermutation.
//
// Every GPU with expert cache has its own instance (INV-BH-1).
// Expert-streaming GPUs get ExpertDevice only; TP GPUs get both
// AttentionDevice and ExpertDevice (INV-BH-5, INV-BH-6).
//
// The abstract interface uses void* streams and generic POD param structs —
// no CUDA/HIP/ROCm types — compilable without any GPU SDK (INV-BH-2).
//
// Concrete implementations:
//   CudaSm120ExpertDevice  — SM120 (RTX 5090/5080) via CUTLASS
//   NullExpertDevice        — no-op for unit tests
//   (future) Ddr5CpuExpertDevice  — x86-64 DDR5 direct-RAM compute (Phase 21, C-3)
//   (future) Hbm2CpuExpertDevice  — x86-64 HBM2e cached device (Phase 21, C-4)

#pragma once

#include "core/gpu_ref.h"
#include "sm120/gemm/grouped_gemm.h"
#include "smxx/activation/fused_swiglu.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace layerstorm::compute {

// ── MoE EP-combine mode (DET-REDUCE Phase 1b) ────────────────────────────────
//
// Selects how moe_unpermute writes its result for the cross-GPU EP combine.
//   kReducedBf16  legacy default: K-way weighted sum, bf16 in / bf16 out into
//                 a [num_tokens, hidden] buffer (placement-DEPENDENT — the
//                 partial-sum bf16 round is a routing-drift FP site).
//   kPerSlotFp32  canonical placement-INVARIANT, FP32 payload: each of the K
//                 contributions c_k = w_k*expert_out_k written to its own fp32
//                 slot [num_tokens, topk, hidden] (no cross-slot sum); the
//                 cross-GPU fp32 allreduce GATHERS slots, then a fixed-order
//                 K-reduce → bf16 once.
//   kPerSlotBf16  same canonical fixed-slot order, BF16 payload (half the gather
//                 bytes): per-slot c_k rounded to bf16 once into a [num_tokens,
//                 topk, hidden] bf16 buffer; bf16 gather (exact, 0+x=x) then a
//                 fixed-order fp32-accumulate K-reduce → bf16 once. Matches the
//                 vLLM/llama.cpp convention (fp32 math, 16-bit payload).
// Invariance comes from the fixed slot ORDER, not the payload dtype.
enum class MoeCombineMode : int {
    kReducedBf16 = 0,
    kPerSlotFp32 = 1,
    kPerSlotBf16 = 2,
};

// ── GGUF grouped GEMM (CPU expert path; ik_llama kernels) ────────────────────
//
// One type-tagged method (cleaner than one virtual per GGUF type). The CPU
// devices (NumaCpuExpertDevice / MultiNumaCpuExpertDevice) implement it via the
// vendored ik kernels (compute/cpu/ik_vendor); GPU / Null / Recording devices
// default to throwing (the interface stays satisfiable + usable for the future).

/// GGUF weight quantization tag. CANONICAL set — value order matches the kernel
/// compute::GgufType (gguf_dequant_gemm.h) AND model::GgufKQuantType
/// (gguf_kquant.h): {Q2_K=0, Q3_K=1, Q4_K=2, Q5_K=3, Q6_K=4, Q8_0=5}. The
/// engine→kernel boundary (CudaSm120ExpertDevice::gguf_grouped_gemm) casts by
/// value. The CPU ik bridge (to_ik_gguf) maps the 4 ik-supported families
/// (Q4_K/Q5_K/Q6_K/Q8_0) and throws on Q2_K/Q3_K (no ik vendor support — those
/// are GPU-only; the CPU expert path never receives them). The legacy vestigial
/// q5_0 (never produced by the GGUF reader, GG-6) was dropped in GG-5.
enum class GgufQuantType {
    Q2_K = 0,  ///< K-quant 2-bit (super-block 256). GPU-only (no ik CPU path).
    Q3_K = 1,  ///< K-quant 3-bit (super-block 256). GPU-only (no ik CPU path).
    Q4_K = 2,  ///< K-quant 4-bit (super-block 256). Activations -> Q8_K (CPU) / Q8_1 (GPU).
    Q5_K = 3,  ///< K-quant 5-bit (super-block 256). Activations -> Q8_K (CPU) / Q8_1 (GPU).
    Q6_K = 4,  ///< K-quant 6-bit (super-block 256). Activations -> Q8_K (CPU) / Q8_1 (GPU).
    Q8_0 = 5,  ///< legacy 8-bit (block 32). Activations -> Q8_2_X4 (CPU) / Q8_1 (GPU).
};

/// Parameters for a GGUF grouped GEMM over experts (one projection: gate, up, or
/// down). Like Nvfp4GroupedGemmParams but with packed GGUF weights + a type tag.
/// CPU path: A is BF16 (elem_size 2) permuted activations; B_ptrs[e] points at
/// expert e's packed GGUF weight block ([N, K] row-major at `type` precision);
/// D is BF16 output. expert_offsets are cumulative permuted-token counts.
/// GGUF GEMM compute strategy. GPU-only distinction: the CPU expert path always
/// dequantizes weights on the fly (lossless activations) and ignores this field.
enum class GgufGemmStrategy {
    int_strategy = 0,  ///< quantize activation to Q8_1; integer dp4a / int8 tensor-core
    dequant      = 1,  ///< dequantize weight to BF16 then GEMM (lossless activation)
};

struct GgufGroupedGemmParams {
    GgufQuantType type;            ///< weight quant family
    GgufGemmStrategy strategy =    ///< GPU compute strategy (CPU path ignores)
        GgufGemmStrategy::int_strategy;
    int num_experts;               ///< active expert groups this call
    int N;                         ///< output rows (gate/up: I; down: H)
    int K;                         ///< input cols  (gate/up: H; down: I)

    /// Permuted row count (== expert_offsets[num_experts]); a host-known
    /// capture-time constant. The GPU device-fused int kernel needs it to size
    /// the one-shot Q8_1 activation quant + the mmq tile-map WITHOUT a host D2H
    /// of the offsets (TD-GG5-GROUPED-HOST-SYNC: graph-capturable). The CPU path
    /// reads expert_offsets[num_experts] itself and ignores this field.
    int total_tokens = 0;

    const void* A_base;            ///< [total_tokens, K] BF16 activations
    void*       D_base;            ///< [total_tokens, N] BF16 output

    const int32_t* expert_offsets; ///< [num_experts + 1] cumulative token counts
    const void** B_ptrs;           ///< [num_experts] packed GGUF weight blocks
};

class ExpertDevice {
public:
    virtual ~ExpertDevice() = default;

    // ── Device selection ────────────────────────────────────────────────────

    /// Activate this device's GPU as the current device.
    virtual void set_device() = 0;

    // ── GPU identity ────────────────────────────────────────────────────────

    /// Returns the GpuRef for this device's GPU.
    virtual const config::GpuRef& gpu() const = 0;

    // ── Grouped GEMM (MoE expert FFN) ───────────────────────────────────────

    /// NVFP4 grouped GEMM for gate+up or down projection.
    virtual void nvfp4_grouped_gemm(
        const Nvfp4GroupedGemmParams& params,
        void* workspace, size_t workspace_bytes,
        void* stream) = 0;

    /// FP8 grouped GEMM for gate+up or down projection.
    virtual void fp8_grouped_gemm(
        const Fp8GroupedGemmParams& params,
        void* workspace, size_t workspace_bytes,
        void* stream) = 0;

    /// GGUF (ik_llama) grouped GEMM for gate+up or down projection. Type-tagged
    /// (Q8_0 / Q5_0 / Q4_K / Q5_K / Q6_K). Implemented by the CPU expert devices via the
    /// vendored ik kernels; GPU / Null / Recording devices throw (the GGUF CPU
    /// path is not a GPU compute path). Default-throws here so existing devices
    /// need not override until they support it.
    virtual void gguf_grouped_gemm(
        const GgufGroupedGemmParams& params,
        void* workspace, size_t workspace_bytes,
        void* stream);

    // ── Activation ──────────────────────────────────────────────────────────

    /// Fused SwiGLU: output = SiLU(gate) * up.
    virtual void fused_swiglu(
        void* output, const void* input,
        const FusedSwigluParams& params,
        int elem_size_bytes, void* stream) = 0;

    // ── Token permutation ───────────────────────────────────────────────────

    /// Sort tokens by expert, expand rows for multi-expert routing.
    virtual void moe_permute(
        void* permuted_input, int32_t* expert_offsets,
        int32_t* src_to_dest_map, int32_t* permuted_idx,
        const void* hidden_states, const int32_t* topk_indices,
        int num_tokens, int topk, int hidden_dim,
        int num_experts, int elem_size_bytes,
        void* workspace, void* stream) = 0;

    /// Reduce expert outputs back to original token order with routing weights.
    /// combine_mode (DET-REDUCE Phase 1b) selects the output form for the cross-GPU
    /// EP combine (see MoeCombineMode): kReducedBf16 (default, legacy bf16 K-way
    /// sum → [num_tokens, hidden]); kPerSlotFp32 (`output` is an FP32 per-slot
    /// buffer [num_tokens, topk, hidden]); kPerSlotBf16 (BF16 per-slot buffer,
    /// half the gather bytes). Both per-slot modes write each contribution to its
    /// own slot (no cross-slot sum) for a placement-invariant fixed-order reduce.
    virtual void moe_unpermute(
        void* output, const void* permuted_output,
        const float* topk_weights, const int32_t* src_to_dest_map,
        int num_tokens, int topk, int hidden_dim,
        int elem_size_bytes, void* stream,
        MoeCombineMode combine_mode = MoeCombineMode::kReducedBf16) = 0;

    // ── Device memory ───────────────────────────────────────────────────────

    /// Allocate device memory.  Returns nullptr on failure.
    virtual void* device_alloc(size_t bytes) = 0;

    /// Free device memory.  Must tolerate nullptr.
    virtual void device_free(void* ptr) = 0;

    /// TD-PREFILL-SUPERCHUNK: free/total device memory query for the MoE
    /// scratch VRAM fail-safe. Returns false when unsupported (Null/CPU
    /// devices) — callers must then skip the budget check (fail-open to the
    /// requested size; device_alloc still returns nullptr on a real OOM).
    virtual bool device_mem_info(size_t& free_bytes, size_t& total_bytes) {
        free_bytes = 0;
        total_bytes = 0;
        return false;
    }

    // Non-copyable (polymorphic base).
    ExpertDevice(const ExpertDevice&) = delete;
    ExpertDevice& operator=(const ExpertDevice&) = delete;

protected:
    ExpertDevice() = default;
};

// ── Factory ─────────────────────────────────────────────────────────────────

/// Create a CudaSm120ExpertDevice for the given GPU.
std::unique_ptr<ExpertDevice> make_cuda_sm120_expert_device(
    config::GpuRef gpu);

/// Create a NullExpertDevice for unit tests.
std::unique_ptr<ExpertDevice> make_null_expert_device(
    config::GpuRef gpu);

// NumaCpuExpertDevice factory (C-6): declared in
// compute/cpu/numa_cpu_expert_device.h, which also defines its NumaCpuExpertDeps
// (it needs NumaManager / PinnedExpertArena / model-dims args this CUDA-free,
// memory-layer-free header must not pull in). Include that header to build one.
//   std::unique_ptr<ExpertDevice> make_numa_cpu_expert_device(
//       config::GpuRef gpu, NumaCpuExpertDeps deps);
//
// MultiNumaCpuExpertDevice factory (C-6): one expert's FFN spread across M NUMA
// nodes (tensor-parallel within the expert). Declared in
// compute/cpu/multi_numa_cpu_expert_device.h (defines MultiNumaCpuExpertDeps with
// the node SET + arena/numa/dims). Include that header to build one.
//   std::unique_ptr<ExpertDevice> make_multi_numa_cpu_expert_device(
//       config::GpuRef gpu, MultiNumaCpuExpertDeps deps);

}  // namespace layerstorm::compute

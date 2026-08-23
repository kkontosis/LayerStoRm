// IMPL: assemble agent.
//
// Implement NumaCpuExpertDevice (declared in numa_cpu_expert_device.h) and the
// make_numa_cpu_expert_device factory. The class is `final : public
// ExpertDevice` and lives entirely in this .cpp (like CudaSm120ExpertDevice's
// class living in its .cu) — the header exposes only the factory + deps structs.
//
// Wire the 9 ExpertDevice pure virtuals to the CPU kernels:
//   set_device()            -> pin the CALLING thread to this node (e.g. via
//                              NumaThreadPool core enumeration / set_mempolicy
//                              preferred node). No GPU.
//   gpu()                   -> return the stored GpuRef (GpuType::cpu).
//   nvfp4_grouped_gemm()    -> build CpuNvfp4ExpertWeights (one PackedProjection
//                              per active expert, resolved from the arena slot
//                              for this node) and call cpu_nvfp4_grouped_gemm in
//                              ONE fork-join region on the NumaThreadPool.
//   fp8_grouped_gemm()      -> NOT a CPU path in C-6 (NVFP4 weights only).
//                              Throw std::runtime_error("FP8 grouped GEMM not
//                              supported on NumaCpuExpertDevice") — the routed
//                              experts are nvfp4-sm1xx on disk. (No stub-silent
//                              no-op: that would corrupt output silently.)
//   fused_swiglu()          -> cpu_swiglu (d = params.d).
//   moe_permute()           -> cpu_moe_permute.
//   moe_unpermute()         -> cpu_moe_unpermute (this writes the host
//                              moe_output — see DEFERRED SEAM below).
//   device_alloc()/free()   -> NumaManager node-local alloc/free on this node
//                              (NumaBuffer kept in a side table keyed by ptr so
//                              free() can return the owning handle), so all
//                              scratch (permuted buffers, gate/up/down temporaries)
//                              is node-local for the pinned threads.
//
// The full FFN chain (permute -> gate GEMM -> up GEMM -> SwiGLU -> down GEMM ->
// unpermute) with the single parallel region and ik-barrier between
// gate/up -> SwiGLU -> down stages is orchestrated here; the per-stage GEMMs and
// element-wise kernels come from nvfp4_cpu_kernel.h / cpu_moe_kernels.h. The
// device owns one NumaThreadPool (built for gpu.id's NUMA node) reused across
// calls.
//
// Weight source: for each active (layer, expert), resolve the node-local arena
// slot pointer (memory::PinnedExpertArena::resolve / arena_for_gpu /
// node_arena(N)->resolve); slice it into the three packed projections (gate, up,
// down are concatenated, expert slots 4096-strided). Full tiered fallback
// (SSD -> arena) is transfer::resolve_expert_host_source(key, deps, target) —
// but the device READS already-resident node-local slabs (consumer node == this
// CPU group's node); the engine guarantees residency before dispatch.
//
// ── DEFERRED SEAM (the ONLY allowed stub — do NOT implement here) ─────────────
// moe_unpermute writes the reduced moe_output into HOST RAM on thread 0. On the
// GPU path that host/device output is folded into the TP all-reduce at
// dispatch_moe.cpp (allreduce_hidden, ~lines 1338/1373) via
// DcpCommunicator::allreduce_hidden (dcp_communicator.cpp:193). For the CPU
// device, that host moe_output must be H2D-staged to each TP GPU's hidden buffer
// (NUMA-aware: source the staging copy from this device's node) and then folded
// into the SAME NCCL all-reduce collective. That cross-device handoff is a
// SEPARATE SESSION and is intentionally left unimplemented. The host moe_output
// buffer + its layout/ownership is the seam: this device produces it; the
// handoff session consumes it. See spec/SPEC_UPDATES.md (MoE / backend-hierarchy)
// and spec/TECH_DEBT.md for the exact fold-in contract.
// ─────────────────────────────────────────────────────────────────────────────
//
// CPU-only TU — NO CUDA/GPU SDK header (INV-GPU-1, INV-BH-2,
// layerstorm_no_cuda_check). Compiles with no GPU SDK, like NullExpertDevice.

#include "compute/cpu/numa_cpu_expert_device.h"

#include "compute/cpu/cpu_gguf_gemm.h"
#include "compute/cpu/gguf_lossless.h"
#include "compute/cpu/cpu_moe_kernels.h"
#include "compute/cpu/numa_thread_pool.h"
#include "compute/cpu/nvfp4_cpu_kernel.h"
#include "core/memory/numa_manager.h"
#include "core/memory/pinned_expert_arena.h"
#include "model/quantization/fp8.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace layerstorm::compute {

namespace {

// ── nvfp4-sm1xx packed projection byte span ──────────────────────────────────
//
// Mirrors model/quantization/nvfp4.cpp Nvfp4::bytes_per_projection: the packed
// projection layout is [FP4 weights | E4M3 group scales | PADDING |
// weight_scale_2 (f32) | input_scale (f32)], 128-byte aligned. The CPU kernel's
// read_weight_scale_2(base, proj_bytes) reads base[proj_bytes - 8], so the span
// we compute here MUST place weight_scale_2 at proj_bytes-8 / input_scale at
// proj_bytes-4 — identical to the prepacked on-disk per-projection block.
constexpr int64_t kProjAlign = 128;

int64_t nvfp4_proj_bytes(int N, int K) {
    const int64_t params = static_cast<int64_t>(N) * static_cast<int64_t>(K);
    const int64_t weight_bytes = (params + 1) / 2;    // 2 FP4 vals/byte
    const int64_t scale_bytes = (params + 15) / 16;   // 1 E4M3 byte / 16 elems
    const int64_t raw = weight_bytes + scale_bytes +
                        2 * static_cast<int64_t>(sizeof(float));
    return (raw + kProjAlign - 1) & ~(kProjAlign - 1);
}

// ── FP8 E4M3 CPU grouped GEMM (mirror of the NVFP4 path) ─────────────────────
//
// Routed experts in C-6 are nvfp4-sm1xx only, so the ExpertDevice virtual
// fp8_grouped_gemm() THROWS (see NumaCpuExpertDevice::fp8_grouped_gemm). This
// helper exists so the FP8 dequant grouped GEMM is fully implemented (not a
// stub) for the shared-expert / future FP8-weights path, mirroring the GPU
// blockwise-scaled FP8 grouped GEMM (deps grouped_gemm.h Fp8GroupedGemmParams):
//   A : [total_tokens, K] BF16 activations (row-major)
//   B_e : [N, K] FP8 E4M3 weights (row-major per expert), via B_ptrs[e]
//   scale_B_e : f32 [ceil(K/128), ceil(N/128)] blockwise scales, via scale_B_ptrs[e]
//   D : [total_tokens, N] BF16 output (row-major)
// Dequant w[n,k] = fp8_e4m3::decode(B_e[n*K+k]) * scale_B_e[ (k/128)*nblk_n
//                  + (n/128) ]. Activations stay BF16, accumulate FP32, store
// BF16 (same numeric policy as the NVFP4 CPU path). Single fork-join region,
// partitions N output rows across threads (thread 0 == caller holds result).
constexpr int kFp8BlockTile = 128;  // model/quantization fp8::kBlockScaleTile

inline float bf16_to_f32(uint16_t b) {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    bits += 0x7FFF + ((bits >> 16) & 1);  // round-to-nearest-even
    return static_cast<uint16_t>(bits >> 16);
}

void fp8_grouped_gemm_one_row_range(
    uint16_t* D, const uint16_t* A,
    const void* const* B_ptrs, const void* const* scale_B_ptrs,
    const int32_t* expert_offsets, int num_experts,
    int N, int K, int n0, int n1) {
    const int nblk_n = (N + kFp8BlockTile - 1) / kFp8BlockTile;
    for (int e = 0; e < num_experts; ++e) {
        const int m0 = expert_offsets[e];
        const int m1 = expert_offsets[e + 1];
        if (m1 <= m0) continue;
        const auto* B = static_cast<const uint8_t*>(B_ptrs[e]);
        const auto* SB = static_cast<const float*>(scale_B_ptrs[e]);
        for (int n = n0; n < n1; ++n) {
            const uint8_t* Brow = B + static_cast<size_t>(n) * K;
            const int nb = n / kFp8BlockTile;
            for (int m = m0; m < m1; ++m) {
                const uint16_t* Arow = A + static_cast<size_t>(m) * K;
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const float scale =
                        SB[static_cast<size_t>(k / kFp8BlockTile) * nblk_n + nb];
                    acc += bf16_to_f32(Arow[k]) *
                           (model::fp8_e4m3::decode(Brow[k]) * scale);
                }
                D[static_cast<size_t>(m) * N + n] = f32_to_bf16(acc);
            }
        }
    }
}

// ik bridge GgufType <- ExpertDevice GgufQuantType (canonical set, GG-5).
// Q2_K/Q3_K have no ik vendor support (GPU-only families); they throw. The CPU
// expert path never receives them.
cpu::ik::GgufType to_ik_gguf(GgufQuantType t) {
    switch (t) {
        case GgufQuantType::Q4_K: return cpu::ik::GgufType::q4_k;
        case GgufQuantType::Q5_K: return cpu::ik::GgufType::q5_k;
        case GgufQuantType::Q6_K: return cpu::ik::GgufType::q6_k;
        case GgufQuantType::Q8_0: return cpu::ik::GgufType::q8_0;
        case GgufQuantType::Q2_K:
        case GgufQuantType::Q3_K:
            throw std::runtime_error(
                "GGUF CPU expert path: Q2_K/Q3_K not supported by the ik vendor "
                "kernels (GPU-only families)");
    }
    throw std::runtime_error("to_ik_gguf: unknown GgufQuantType");
}

}  // namespace

// ── NumaCpuExpertDevice ──────────────────────────────────────────────────────

// Concrete CPU expert-FFN device for one NUMA node (C-6). final : ExpertDevice
// only — no DeviceBackend, no attention. Class lives entirely in this TU (like
// CudaSm120ExpertDevice's class in its .cu); the header exposes only the deps
// structs + factory. CPU-only: no GPU SDK include (INV-GPU-1 / INV-BH-2).
class NumaCpuExpertDevice final : public ExpertDevice {
public:
    NumaCpuExpertDevice(config::GpuRef gpu, NumaCpuExpertDeps deps)
        : gpu_(gpu),
          deps_(deps),
          // For a cpu GpuRef, GpuRef::id carries the NUMA node index (gpu_ref.h),
          // NOT a CUDA ordinal. The pool pins to this node's physical cores;
          // reserve_leading_cores holds the node's leading core(s) out for the
          // daemon/orchestrator (C-6 QA, default 0 => all cores as before).
          pool_(gpu.id, deps.max_threads, deps.reserve_leading_cores) {}

    ~NumaCpuExpertDevice() override {
        // Free any device_alloc'd buffers still outstanding (defensive).
        std::lock_guard<std::mutex> lock(alloc_mu_);
        for (auto& [ptr, buf] : allocs_) {
            if (deps_.numa) {
                memory::NumaBuffer b = buf;
                deps_.numa->free(b);
            } else {
                std::free(ptr);
            }
        }
        allocs_.clear();
    }

    // ── Device selection + identity ─────────────────────────────────────────

    /// Pin the CALLING (daemon) thread to this device's NUMA node and prefer
    /// node-local allocation for first-touched pages. No CUDA. Best-effort: if
    /// the node has no enumerable physical cores (no NUMA / sysfs), this is a
    /// no-op and work falls back to an unpinned single thread (pool collapses to
    /// 1). The per-call parallel_for re-pins thread 0 for the duration of each
    /// region and restores affinity on return, so this set_device() pin is only
    /// the steady-state hint between calls.
    void set_device() override {
        const auto cpus = cpu::node_physical_cpus(gpu_.id);
        if (!cpus.empty()) cpu::pin_thread_to_cpu(cpus.front());
    }

    const config::GpuRef& gpu() const override { return gpu_; }

    // ── Grouped GEMM ────────────────────────────────────────────────────────

    /// NVFP4 grouped GEMM for ONE projection (gate, up, or down). Bridges the
    /// generic Nvfp4GroupedGemmParams to the CPU dequant-on-the-fly kernel:
    ///   * params.A_base : [total_tokens, K] BF16 permuted activations.
    ///   * params.B_ptrs[e] : base of expert e's packed nvfp4-sm1xx projection
    ///                        block (the per-expert arena slot + projection
    ///                        offset). REQUIRED (the CPU path has no base+stride
    ///                        weight layout — scattered arena slots only).
    ///   * params.D_base : [total_tokens, N] BF16 output.
    ///   * params.expert_offsets / N / K as on the GPU path.
    /// scale_B_base / scale_B_ptrs / alphas / sf_offsets are IGNORED: the CPU
    /// path dequantizes weights in full (E2M1 * E4M3 group scale * weight_scale_2)
    /// and reads weight_scale_2 from the projection tail itself. activation
    /// quantization / input_scale are ignored (BF16 activations, FP32 accumulate).
    /// Runs as ONE fork-join region partitioning N output rows across the pool's
    /// physical cores; thread 0 (caller) holds the result. `stream` ignored
    /// (synchronous), like NullExpertDevice.
    void nvfp4_grouped_gemm(const Nvfp4GroupedGemmParams& params,
                            void* /*workspace*/, size_t /*workspace_bytes*/,
                            void* /*stream*/) override {
        if (!params.B_ptrs) {
            throw std::runtime_error(
                "NumaCpuExpertDevice::nvfp4_grouped_gemm requires per-expert "
                "B_ptrs (scattered arena slots; no base+stride layout on CPU)");
        }
        const int N = params.N;
        const int K = params.K;
        const int num_experts = params.num_experts;
        const int64_t proj_bytes = nvfp4_proj_bytes(N, K);

        // Build one PackedProjection per active expert from the per-expert
        // projection base pointers. The fp4 region is first, scales follow, and
        // weight_scale_2 sits at base[proj_bytes-8] (read inside the kernel).
        std::vector<cpu::PackedProjection> projs(static_cast<size_t>(num_experts));
        for (int e = 0; e < num_experts; ++e) {
            projs[static_cast<size_t>(e)] = cpu::PackedProjection{
                static_cast<const uint8_t*>(params.B_ptrs[e]),
                static_cast<size_t>(proj_bytes), N, K};
        }
        cpu::CpuNvfp4ExpertWeights weights{projs.data(), num_experts};

        // LOSSLESS PARITY mode (LS_CPU_EXPERT_LOSSLESS=1): quantize activations to
        // NVFP4 exactly like the GPU CUTLASS path (bf16_to_nvfp4_grouped) so a
        // CPU-computed expert produces the SAME lossy output as a GPU-computed one
        // — required for the DSpark lossless verify + the TQ golden to hold with
        // CPU offload on. The per-expert input_scale sits at proj_bytes-4 (the GPU
        // reads the same scalar via gather_nvfp4_alpha_and_input_scale). Default
        // (unset) keeps the faster BF16-activation path (byte-identical to prior
        // behavior) for uses that don't need GPU parity.
        static const bool lossless = [] {
            const char* v = std::getenv("LS_CPU_EXPERT_LOSSLESS");
            return v && v[0] == '1';
        }();
        if (lossless) {
            std::vector<float> input_scales(static_cast<size_t>(num_experts), 1.0f);
            for (int e = 0; e < num_experts; ++e) {
                float is = 1.0f;
                std::memcpy(&is,
                            static_cast<const uint8_t*>(params.B_ptrs[e]) + proj_bytes - 4,
                            sizeof(float));
                input_scales[static_cast<size_t>(e)] = is;
            }
            cpu_nvfp4_grouped_gemm_actquant(
                params.D_base, params.A_base, weights, input_scales.data(),
                params.expert_offsets, N, K, /*elem_size_bytes=*/2, &pool_);
            return;
        }

        cpu_nvfp4_grouped_gemm(
            params.D_base, params.A_base, weights,
            params.expert_offsets, N, K,
            /*elem_size_bytes=*/2, &pool_);
    }

    /// FP8 grouped GEMM is NOT a CPU path in C-6: routed experts are
    /// nvfp4-sm1xx on disk, never FP8. A silent no-op would corrupt MoE output,
    /// so this THROWS. (A fully-implemented CPU FP8 dequant grouped GEMM exists
    /// as the file-local fp8_grouped_gemm_one_row_range helper for the future
    /// FP8-weights path; it is intentionally not reachable through this virtual
    /// until an FP8 routed-expert config exists.)
    void fp8_grouped_gemm(const Fp8GroupedGemmParams& params,
                          void* /*workspace*/, size_t /*workspace_bytes*/,
                          void* /*stream*/) override {
        if (params.num_experts == 0) return;  // nothing to do (degenerate)
        throw std::runtime_error(
            "FP8 grouped GEMM not supported on NumaCpuExpertDevice "
            "(routed experts are nvfp4-sm1xx; no FP8 weights on this path)");
    }

    /// GGUF grouped GEMM (ik_llama kernels) for ONE projection. Quantizes the
    /// BF16 permuted activations to the ik activation format, runs the verbatim
    /// ik kernel per expert (partitioned by output rows across the pool), writes
    /// BF16 output. `params.B_ptrs[e]` is expert e's packed GGUF weight block.
    /// `stream`/`workspace` ignored (synchronous CPU), like the NVFP4 path.
    void gguf_grouped_gemm(const GgufGroupedGemmParams& params,
                           void* /*workspace*/, size_t /*workspace_bytes*/,
                           void* /*stream*/) override {
        if (params.num_experts == 0) return;
        if (!params.B_ptrs) {
            throw std::runtime_error(
                "NumaCpuExpertDevice::gguf_grouped_gemm requires per-expert "
                "B_ptrs (packed GGUF weight blocks)");
        }
        // LOSSLESS PARITY mode (LS_CPU_EXPERT_LOSSLESS=1): for Q4_K (the production
        // TARGET gate/up quant) use the bit-compatible Q8_1 path that reproduces the
        // GPU GGUF mmvq numerics (Q8_1 activation quant + Q4_K int dot), so a
        // CPU-computed Q4_K expert matches the GPU GGUF expert to FP32 accumulation
        // ORDER — the precondition for CPU offload to hold the TQ golden (12089) on
        // the GGUF target. The default ik path (Q8_2_X4 acts) is faster but not
        // GPU-bit-compatible. Q5_K/Q6_K (down proj) still ride the ik path until
        // their lossless variants land (same technique).
        static const bool lossless = [] {
            const char* v = std::getenv("LS_CPU_EXPERT_LOSSLESS");
            return v && v[0] == '1';
        }();
        if (lossless && (params.type == GgufQuantType::Q4_K ||
                         params.type == GgufQuantType::Q5_K ||
                         params.type == GgufQuantType::Q6_K)) {
            const cpu::KQuantLossless kt =
                params.type == GgufQuantType::Q4_K ? cpu::KQuantLossless::Q4_K
                : params.type == GgufQuantType::Q5_K ? cpu::KQuantLossless::Q5_K
                                                     : cpu::KQuantLossless::Q6_K;
            cpu::cpu_gguf_grouped_gemm_kq_lossless(
                kt, static_cast<uint16_t*>(params.D_base),
                static_cast<const uint16_t*>(params.A_base),
                params.B_ptrs, params.expert_offsets, params.num_experts,
                params.N, params.K, &pool_);
            return;
        }

        const cpu::ik::GgufType type = to_ik_gguf(params.type);
        cpu::cpu_gguf_grouped_gemm(
            static_cast<uint16_t*>(params.D_base),
            static_cast<const uint16_t*>(params.A_base),
            type, params.B_ptrs, params.expert_offsets, params.num_experts,
            params.N, params.K, &pool_);
    }

    // ── Activation ──────────────────────────────────────────────────────────

    /// Fused SwiGLU: output = SiLU(gate) * up. The C-6 CPU chain runs SEPARATE
    /// gate/up GEMMs, so `input` is the gate buffer and `up`-half is the second
    /// half of `input` ONLY on the interleaved GPU layout. Here we follow the
    /// GPU virtual's contract (input = [num_tokens, 2*d] gate|up) — matching
    /// FusedSwigluParams{num_tokens, d} — and read gate/up interleaved. (A
    /// separate-buffer SwiGLU is available via cpu_swiglu's gate_up_interleaved
    /// flag for callers that hold gate/up in distinct buffers.)
    void fused_swiglu(void* output, const void* input,
                      const FusedSwigluParams& params,
                      int elem_size_bytes, void* /*stream*/) override {
        cpu::cpu_swiglu(output, input, /*up=*/nullptr,
                        params.num_tokens, params.d,
                        /*gate_up_interleaved=*/true,
                        elem_size_bytes, &pool_, params.swiglu_limit);
    }

    // ── Token permutation ───────────────────────────────────────────────────

    void moe_permute(void* permuted_input, int32_t* expert_offsets,
                     int32_t* src_to_dest_map, int32_t* permuted_idx,
                     const void* hidden_states, const int32_t* topk_indices,
                     int num_tokens, int topk, int hidden_dim,
                     int num_experts, int elem_size_bytes,
                     void* /*workspace*/, void* /*stream*/) override {
        cpu::cpu_moe_permute(permuted_input, expert_offsets, src_to_dest_map,
                             permuted_idx, hidden_states, topk_indices,
                             num_tokens, topk, hidden_dim, num_experts,
                             elem_size_bytes, &pool_);
    }

    void moe_unpermute(void* output, const void* permuted_output,
                       const float* topk_weights, const int32_t* src_to_dest_map,
                       int num_tokens, int topk, int hidden_dim,
                       int elem_size_bytes, void* /*stream*/,
                       MoeCombineMode combine_mode =
                           MoeCombineMode::kReducedBf16) override {
        // C-6 CPU expert offload (canonical per-slot fold): the forced CPU
        // experts join the SAME placement-invariant per-slot EP combine as the
        // GPU experts. kPerSlotFp32/kPerSlotBf16 emit [num_tokens, topk, hidden]
        // per-slot rows (bit-identical to the GPU finalize_moe_routing_*_perslot
        // kernels); kReducedBf16 is the legacy host-reduced [num_tokens, hidden].
        if (combine_mode == MoeCombineMode::kPerSlotFp32
            || combine_mode == MoeCombineMode::kPerSlotBf16) {
            cpu::cpu_moe_unpermute_perslot(
                output, permuted_output, topk_weights, src_to_dest_map,
                num_tokens, topk, hidden_dim,
                /*fp32_output=*/combine_mode == MoeCombineMode::kPerSlotFp32,
                &pool_);
            return;
        }
        cpu::cpu_moe_unpermute(output, permuted_output, topk_weights,
                               src_to_dest_map, num_tokens, topk, hidden_dim,
                               elem_size_bytes, &pool_);
        // ── DEFERRED SEAM (the ONLY allowed stub — do NOT implement here) ────
        // `output` now holds the reduced moe_output for this layer in HOST RAM,
        // node-local to this device. On the GPU ExpertDevice path the equivalent
        // buffer is folded into the TP all-reduce at dispatch_moe.cpp (~lines
        // 1338/1373, allreduce_hidden) via DcpCommunicator::allreduce_hidden
        // (dcp_communicator.cpp:193). For the CPU device, this host moe_output
        // must be H2D-staged onto each TP GPU's hidden buffer — NUMA-aware: the
        // staging copy is sourced from THIS device's node (gpu_.id) so the H2D is
        // local-bandwidth — and then folded into the SAME NCCL all-reduce
        // collective. That cross-device handoff (host moe_output -> TP-GPU H2D ->
        // NCCL fold-in) is a SEPARATE SESSION and is intentionally left
        // unimplemented. The host moe_output buffer + its layout/ownership IS the
        // seam: this device produces it on thread 0; the handoff session consumes
        // it. See spec/SPEC_UPDATES.md (MoE / backend-hierarchy) and
        // spec/TECH_DEBT.md (TD-CPU-EXPERT-HANDOFF) for the exact fold-in contract.
        // ────────────────────────────────────────────────────────────────────
    }

    // ── Device memory (NUMA-node-local) ─────────────────────────────────────

    /// Allocate node-local host memory for this device's NUMA node (so scratch
    /// — permuted buffers, gate/up/down temporaries — is local to the pinned
    /// threads). Goes through NumaManager::allocate_on_node (MPOL_BIND); falls
    /// back to plain malloc when no NumaManager is configured. The owning
    /// NumaBuffer handle is kept in a ptr-keyed side table so device_free can
    /// return it. Returns nullptr on failure (never throws).
    void* device_alloc(size_t bytes) override {
        if (bytes == 0) return nullptr;
        if (deps_.numa) {
            try {
                memory::NumaBuffer buf =
                    deps_.numa->allocate_on_node(bytes, gpu_.id);
                if (buf.data) {
                    std::lock_guard<std::mutex> lock(alloc_mu_);
                    allocs_.emplace(buf.data, buf);
                }
                return buf.data;
            } catch (const std::exception&) {
                return nullptr;  // contract: nullptr on failure.
            }
        }
        return std::malloc(bytes);
    }

    /// Free a buffer from device_alloc. Tolerates nullptr.
    void device_free(void* ptr) override {
        if (!ptr) return;
        if (deps_.numa) {
            memory::NumaBuffer buf{};
            {
                std::lock_guard<std::mutex> lock(alloc_mu_);
                auto it = allocs_.find(ptr);
                if (it == allocs_.end()) {
                    // Not ours (or already freed) — do not pass a foreign ptr to
                    // NumaManager::free / std::free. Defensive no-op.
                    return;
                }
                buf = it->second;
                allocs_.erase(it);
            }
            deps_.numa->free(buf);
        } else {
            std::free(ptr);
        }
    }

private:
    config::GpuRef gpu_;
    NumaCpuExpertDeps deps_;
    cpu::NumaThreadPool pool_;

    // ptr -> owning NumaBuffer for device_free (NumaManager::free needs the
    // handle, not just the pointer). Guarded: completion callbacks on other
    // threads never alloc/free here, but the daemon thread is the only mutator;
    // the mutex is cheap insurance + matches PinnedExpertArena's threading note.
    std::mutex alloc_mu_;
    std::unordered_map<void*, memory::NumaBuffer> allocs_;
};

std::unique_ptr<ExpertDevice> make_numa_cpu_expert_device(
    config::GpuRef gpu, NumaCpuExpertDeps deps) {
    return std::make_unique<NumaCpuExpertDevice>(gpu, deps);
}

}  // namespace layerstorm::compute

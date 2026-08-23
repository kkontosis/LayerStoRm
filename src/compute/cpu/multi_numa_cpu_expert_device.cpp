// MultiNumaCpuExpertDevice — ONE expert's FFN spread across M NUMA nodes (C-6).
//
// Implements multi_numa_cpu_expert_device.h. The class is `final : public
// ExpertDevice` and lives entirely in this TU (like the other CPU devices); the
// header exposes only the factory + deps struct.
//
// DISTRIBUTION (per projection GEMM call): the N output rows are partitioned into
// contiguous 128-row-aligned slices, one slice per chosen NUMA node. M driver
// threads (one per node, pinned to the node) each:
//   1. STAGE its N-slice of every active expert's packed nvfp4-sm1xx projection
//      (FP4 row-slab + the matching Sm1xx scale-row-slab + the per-proj tail
//      scalars) into that node's LOCAL scratch (PinnedExpertArena::node_scratch,
//      or a private node-local device_alloc fallback). 128-row alignment keeps the
//      scale-atom de-interleave in-bounds for the slice read as a standalone
//      projection (INV-CPU-EXP-3). A small residency map skips re-staging a slice
//      already resident for the same (proj, expert, slice) key.
//   2. COMPUTE its slice from the LOCAL staged slab via the existing
//      cpu_nvfp4_grouped_gemm over the node's own NumaThreadPool, writing the
//      slice's N-columns of the shared D output (disjoint columns → no reduction;
//      the concat / all-gather is implicit in writing the shared D).
// Because each GEMM writes the FULL [tokens, N] (all nodes' N-columns), SwiGLU and
// the down GEMM read the fully-materialized intermediate — the PART-2 "all-gather
// the SwiGLU intermediate" is satisfied by D being shared host memory all nodes
// wrote into; an explicit barrier across node-threads gates the join before the
// orchestrator's next stage reads it.
//
// GROUPED case (num_experts>=1, M_tokens>=1): each node loops the experts over its
// own N-slice (the staged slab carries every active expert's slice), so multibatch
// / multi-expert works without special-casing 1×1×1.
//
// CPU-only TU — NO CUDA/GPU SDK header (INV-GPU-1, INV-BH-2,
// layerstorm_no_cuda_check). `void* stream` ignored (synchronous).

#include "compute/cpu/multi_numa_cpu_expert_device.h"

#include "compute/cpu/cpu_gguf_gemm.h"
#include "compute/cpu/cpu_moe_kernels.h"
#include "compute/cpu/ik_barrier.h"
#include "compute/cpu/numa_thread_pool.h"
#include "compute/cpu/nvfp4_cpu_kernel.h"
#include "core/memory/numa_manager.h"
#include "core/memory/pinned_expert_arena.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sched.h>
#if defined(__SSE3__)
#include <immintrin.h>
#endif

namespace layerstorm::compute {

// ── Instrumentation (benchmark attribution: stage vs compute) ────────────────
// Accumulate nanoseconds spent in cross-node weight STAGING (the memcpy of each
// node's N-slice into local scratch) vs node COMPUTE (cpu_nvfp4_grouped_gemm) and
// bytes actually staged. Only enabled when LS_MULTINUMA_PROFILE is set — zero
// overhead otherwise (one relaxed atomic load on a cold flag). The benchmark reads
// these to report stage_us separately from compute_us per the C-6 deliverable.
namespace {
std::atomic<uint64_t> g_stage_ns{0};
std::atomic<uint64_t> g_compute_ns{0};
std::atomic<uint64_t> g_staged_bytes{0};
bool profile_enabled() {
    static const bool on = std::getenv("LS_MULTINUMA_PROFILE") != nullptr;
    return on;
}
}  // namespace

void multinuma_profile_reset() {
    g_stage_ns.store(0); g_compute_ns.store(0); g_staged_bytes.store(0);
}
uint64_t multinuma_profile_stage_ns()    { return g_stage_ns.load(); }
uint64_t multinuma_profile_compute_ns()  { return g_compute_ns.load(); }
uint64_t multinuma_profile_staged_bytes(){ return g_staged_bytes.load(); }

namespace {

constexpr int64_t kProjAlign = 128;        ///< per-projection 128-byte alignment
constexpr int64_t kScaleRowAtom = 128;     ///< Sm1xx scale atom = 128 rows
constexpr int64_t kScaleAtomBytes = 512;   ///< 128 rows × 4 groups, 1 byte each
constexpr int64_t kGroupCols = 4;          ///< groups per scale atom

// Packed nvfp4-sm1xx projection byte span (mirrors nvfp4_proj_bytes in
// numa_cpu_expert_device.cpp / model/quantization/nvfp4.cpp): FP4 weights | E4M3
// group scales | padding | weight_scale_2 (f32) | input_scale (f32), 128-aligned.
int64_t nvfp4_proj_bytes(int64_t N, int64_t K) {
    const int64_t params = N * K;
    const int64_t weight_bytes = (params + 1) / 2;
    const int64_t scale_bytes = (params + 15) / 16;
    const int64_t raw = weight_bytes + scale_bytes + 2 * static_cast<int64_t>(sizeof(float));
    return (raw + kProjAlign - 1) & ~(kProjAlign - 1);
}

// Partition N output rows into M contiguous 128-row-aligned slices. slice[j] is
// [n_begin[j], n_end[j]); boundaries are multiples of 128 except the final n_end ==
// N (N is a multiple of 128 for every real projection, so all are 128-aligned).
// Balanced in 128-row blocks (remainder spread over the first nodes). A node may
// get an EMPTY slice when N has fewer than M 128-row blocks (e.g. tiny test dims);
// such a node does no work.
void partition_rows_128(int N, int M, std::vector<int>& n_begin,
                        std::vector<int>& n_end) {
    n_begin.assign(static_cast<size_t>(M), 0);
    n_end.assign(static_cast<size_t>(M), 0);
    const int atom = static_cast<int>(kScaleRowAtom);
    // Number of whole 128-row blocks; a ragged tail (N % 128 != 0, only in tiny
    // tests) is appended to the last non-empty slice so no rows are dropped.
    const int blocks = N / atom;
    const int tail = N - blocks * atom;
    const int base = blocks / M;
    const int rem = blocks % M;
    int cur = 0;
    int last_nonempty = -1;
    for (int j = 0; j < M; ++j) {
        const int b = base + (j < rem ? 1 : 0);
        n_begin[static_cast<size_t>(j)] = cur;
        cur += b * atom;
        n_end[static_cast<size_t>(j)] = cur;
        if (b > 0) last_nonempty = j;
    }
    if (tail > 0) {
        // Hand the ragged tail to the last non-empty slice (or node 0 if N<128).
        const int j = last_nonempty >= 0 ? last_nonempty : 0;
        n_end[static_cast<size_t>(j)] = N;
        // Shift any slices after j (all empty) to start at N.
        for (int k = j + 1; k < M; ++k) {
            n_begin[static_cast<size_t>(k)] = N;
            n_end[static_cast<size_t>(k)] = N;
        }
    } else if (last_nonempty >= 0 && n_end[static_cast<size_t>(last_nonempty)] != N) {
        n_end[static_cast<size_t>(last_nonempty)] = N;
    }
}

// Stage the contiguous N-row slice [n0, n1) of a packed projection `src`
// (proj_bytes span, N rows, K cols) into `dst` as a SELF-CONTAINED packed
// projection of (n1-n0) rows. n0/n1 MUST be multiples of 128 (or n1==N). Returns
// the staged slice's proj_bytes. Layout of dst (== nvfp4_proj_bytes(Nj,K)):
//   [ FP4 row-slab : rows [n0,n1) of src's FP4 region — a contiguous byte range ]
//   [ E4M3 scale row-slab : rows [n0,n1) of src's Sm1xx-interleaved scales —
//       contiguous because a 128-row-aligned slice maps to whole 512-byte atoms ]
//   [ tail scalars copied from src (weight_scale_2 @ -8, input_scale @ -4) ]
int64_t stage_proj_slice(uint8_t* dst, const uint8_t* src, int64_t src_proj_bytes,
                         int N, int K, int n0, int n1) {
    (void)N;
    const int64_t Nj = n1 - n0;
    const int64_t Kk = K;
    const int64_t G = Kk / 16;                       // groups per row
    const int64_t src_params = static_cast<int64_t>(N) * Kk;
    const int64_t src_fp4_bytes = (src_params + 1) / 2;

    const int64_t dst_pb = nvfp4_proj_bytes(Nj, Kk);
    const int64_t dst_params = Nj * Kk;
    const int64_t dst_fp4_bytes = (dst_params + 1) / 2;

    // 1) FP4 row-slab: src rows [n0,n1) are a contiguous byte range (K even).
    const int64_t fp4_off = (static_cast<int64_t>(n0) * Kk) / 2;
    const int64_t fp4_len = (Nj * Kk) / 2;
    std::memcpy(dst, src + fp4_off, static_cast<size_t>(fp4_len));

    // 2) Scale row-slab: the Sm1xx scale region of src starts at src_fp4_bytes.
    //    scale_byte_index tile = (n/128)*ceil(G/4) + (g/4); for a 128-aligned
    //    slice rows [n0,n1) occupy whole atoms tiles
    //    [(n0/128)*ceil(G/4), (n1/128)*ceil(G/4)) × 512 bytes — contiguous, and
    //    it starts at byte (n0/128)*ceil(G/4)*512 within the scale region. The
    //    destination scale region begins at dst_fp4_bytes (proj-internal offset).
    const int64_t nblk_g = (G + kGroupCols - 1) / kGroupCols;  // ceil(G/4)
    const int64_t src_scale_off = src_fp4_bytes +
                                  (static_cast<int64_t>(n0) / kScaleRowAtom) *
                                      nblk_g * kScaleAtomBytes;
    const int64_t scale_len = (Nj / kScaleRowAtom) * nblk_g * kScaleAtomBytes;
    // Guard the ragged-tail (Nj not a multiple of 128, tiny-dim tests only): copy
    // the exact (Nj*G) bytes the kernel will read instead of whole atoms.
    const int64_t scale_copy =
        (Nj % kScaleRowAtom == 0) ? scale_len : (Nj * G);
    std::memcpy(dst + dst_fp4_bytes, src + src_scale_off,
                static_cast<size_t>(scale_copy));

    // 3) Tail scalars (weight_scale_2 @ pb-8, input_scale @ pb-4) — same for the
    //    whole projection; copy verbatim so read_weight_scale_2 finds them.
    std::memcpy(dst + dst_pb - 8, src + src_proj_bytes - 8, 8);
    return dst_pb;
}

// Partition N output rows into M contiguous balanced slices with NO alignment
// constraint (used by the GGUF path — each output row is a self-contained packed
// weight row, so any row is a valid slice boundary; INV-CPU-EXP-GGUF-MN). The
// remainder is spread over the first nodes. A node may get an EMPTY slice when
// N < M (tiny test dims); such a node does no work.
void partition_rows_balanced(int N, int M, std::vector<int>& n_begin,
                             std::vector<int>& n_end) {
    n_begin.assign(static_cast<size_t>(M), 0);
    n_end.assign(static_cast<size_t>(M), 0);
    const int base = N / M;
    const int rem = N % M;
    int cur = 0;
    for (int j = 0; j < M; ++j) {
        const int b = base + (j < rem ? 1 : 0);
        n_begin[static_cast<size_t>(j)] = cur;
        cur += b;
        n_end[static_cast<size_t>(j)] = cur;
    }
}

// Map a public GgufQuantType to the ik vendor enum (canonical set, GG-5).
// Q2_K/Q3_K have no ik vendor support (GPU-only); they throw.
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

// ── MultiNumaCpuExpertDevice ──────────────────────────────────────────────────

class MultiNumaCpuExpertDevice final : public ExpertDevice {
public:
    MultiNumaCpuExpertDevice(config::GpuRef gpu, MultiNumaCpuExpertDeps deps)
        : gpu_(gpu), deps_(std::move(deps)) {
        if (deps_.nodes.empty())
            throw std::invalid_argument(
                "MultiNumaCpuExpertDevice: deps.nodes must be non-empty");
        M_ = static_cast<int>(deps_.nodes.size());
        // One persistent NumaThreadPool per chosen node, reused across calls.
        pools_.reserve(static_cast<size_t>(M_));
        for (int j = 0; j < M_; ++j) {
            pools_.push_back(std::make_unique<cpu::NumaThreadPool>(
                deps_.nodes[static_cast<size_t>(j)], deps_.max_threads_per_node));
        }
        // Private staging buffers per node when no arena scratch is provided
        // (one per node, grown on demand). Kept node-local via device_alloc.
        stage_bufs_.assign(static_cast<size_t>(M_), nullptr);
        stage_caps_.assign(static_cast<size_t>(M_), 0);
        stage_base_.assign(static_cast<size_t>(M_), nullptr);
        occupant_.resize(static_cast<size_t>(M_));
        n_begin_.assign(static_cast<size_t>(M_), 0);
        n_end_.assign(static_cast<size_t>(M_), 0);

        // PERSISTENT cross-node driver threads (j = 1..M-1), spawned ONCE and
        // parked on a generation handoff — NOT spawned per GEMM call. Per-call
        // std::thread create/join across NUMA nodes cost more than the compute it
        // saves at ms granularity; parking them amortizes that to zero. Driver 0 is
        // the calling (daemon) thread, which runs its slice inline. Mirrors the
        // NumaThreadPool generation-counter handshake (one level up: across nodes).
        if (M_ > 1) {
            drivers_.reserve(static_cast<size_t>(M_ - 1));
            for (int j = 1; j < M_; ++j)
                drivers_.emplace_back([this, j]() { driver_main(j); });
        }
    }

    ~MultiNumaCpuExpertDevice() override {
        // Stop the persistent drivers: bump generation with shutdown set, join.
        shutdown_.store(true, std::memory_order_release);
        job_gen_.fetch_add(1, std::memory_order_acq_rel);
        for (auto& t : drivers_) if (t.joinable()) t.join();
        for (int j = 0; j < M_; ++j) {
            if (stage_bufs_[static_cast<size_t>(j)])
                free_on_node(deps_.nodes[static_cast<size_t>(j)],
                             stage_bufs_[static_cast<size_t>(j)]);
        }
        if (quant_act_buf_) free_on_node(deps_.nodes.front(), quant_act_buf_);
        std::lock_guard<std::mutex> lock(alloc_mu_);
        for (auto& [ptr, buf] : allocs_) {
            if (deps_.numa) { memory::NumaBuffer b = buf; deps_.numa->free(b); }
            else std::free(ptr);
        }
        allocs_.clear();
    }

    // ── Device selection + identity ─────────────────────────────────────────

    /// Pin the calling (daemon) thread to the PRIMARY node (deps.nodes[0]); the
    /// per-call worker threads re-pin per node. Best-effort no-op without sysfs.
    void set_device() override {
        const auto cpus = cpu::node_physical_cpus(deps_.nodes.front());
        if (!cpus.empty()) cpu::pin_thread_to_cpu(cpus.front());
    }

    const config::GpuRef& gpu() const override { return gpu_; }

    // ── Grouped GEMM (distributed across M nodes by output-N partition) ───────

    void nvfp4_grouped_gemm(const Nvfp4GroupedGemmParams& params,
                            void* /*workspace*/, size_t /*workspace_bytes*/,
                            void* /*stream*/) override {
        if (params.num_experts == 0 || params.N == 0) return;
        if (!params.B_ptrs)
            throw std::runtime_error(
                "MultiNumaCpuExpertDevice::nvfp4_grouped_gemm requires per-expert "
                "B_ptrs (scattered packed projections; no base+stride on CPU)");

        const int N = params.N;
        const int64_t src_pb = nvfp4_proj_bytes(N, params.K);

        // Output-N partition into M contiguous 128-row slices (job state read by
        // the persistent drivers).
        partition_rows_128(N, M_, n_begin_, n_end_);

        job_kind_ = JobKind::kNvfp4;
        job_params_ = &params;
        job_Du_ = static_cast<uint16_t*>(params.D_base);
        job_Au_ = static_cast<const uint16_t*>(params.A_base);
        job_src_pb_ = src_pb;

        if (M_ == 1) { run_slice(0); return; }  // no drivers; inline

        // Publish the job to the parked drivers (release the params above) and
        // run slice 0 inline. Drivers decrement jobs_remaining_ when done.
        jobs_remaining_.store(M_ - 1, std::memory_order_release);
        job_gen_.fetch_add(1, std::memory_order_acq_rel);

        run_slice(0);

        // Wait for the M-1 drivers — spin-then-yield (same discipline as the
        // NumaThreadPool / ik_barrier waits). This is the cross-node join; the
        // orchestrator's next stage reads the full D only after it returns.
        for (int spin = 0;
             jobs_remaining_.load(std::memory_order_acquire) != 0; ++spin) {
            if (spin >= 4096) { spin = 0; sched_yield(); }
#if defined(__SSE3__)
            else _mm_pause();
#endif
        }
    }

    /// FP8 grouped GEMM is NOT a CPU path in C-6 (routed experts are nvfp4-sm1xx).
    void fp8_grouped_gemm(const Fp8GroupedGemmParams& params,
                          void* /*workspace*/, size_t /*workspace_bytes*/,
                          void* /*stream*/) override {
        if (params.num_experts == 0) return;
        throw std::runtime_error(
            "FP8 grouped GEMM not supported on MultiNumaCpuExpertDevice "
            "(routed experts are nvfp4-sm1xx; no FP8 weights on this path)");
    }

    /// GGUF grouped GEMM (ik_llama kernels) for ONE projection, DISTRIBUTED across
    /// the M NUMA nodes by an OUTPUT-ROW (N) partition — mirrors the NVFP4 path but
    /// simpler: each GGUF output row is a self-contained packed weight row, so a
    /// node's N-slice is a contiguous range of packed weight rows (no sub-row atom
    /// alignment). Activations are quantized ONCE and the small Q8_x buffer is
    /// shared to all nodes; each node stages its contiguous packed-weight-row slice
    /// into LOCAL scratch and runs gguf_gemm_one over it, writing its disjoint
    /// output-row slice of the shared D. BIT-EXACT to the single-node path: every
    /// output element is a complete dot computed entirely on ONE node, no
    /// cross-node accumulation / reorder (TD-IK-MULTINODE, INV-CPU-EXP-GGUF-MN).
    void gguf_grouped_gemm(const GgufGroupedGemmParams& params,
                           void* /*workspace*/, size_t /*workspace_bytes*/,
                           void* /*stream*/) override {
        if (params.num_experts == 0 || params.N == 0) return;
        if (!params.B_ptrs)
            throw std::runtime_error(
                "MultiNumaCpuExpertDevice::gguf_grouped_gemm requires per-expert "
                "B_ptrs (packed GGUF weight blocks)");
        const cpu::ik::GgufType type = to_ik_gguf(params.type);

        // Preserve the gguf_supported(t) fallback: if unsupported on this
        // build/host, the shared driver throws exactly as the single-node path
        // does (the slice driver would too). Quantizing once requires support; so
        // delegate the unsupported case to the (throwing) shared single-node
        // driver on the primary pool to keep behavior identical.
        if (!cpu::ik::gguf_supported(type)) {
            cpu::cpu_gguf_grouped_gemm(
                static_cast<uint16_t*>(params.D_base),
                static_cast<const uint16_t*>(params.A_base),
                type, params.B_ptrs, params.expert_offsets, params.num_experts,
                params.N, params.K, pools_[0].get());
            return;
        }

        const int N = params.N;
        const int K = params.K;
        const int E = params.num_experts;
        const int total_tokens = params.expert_offsets[E];
        if (total_tokens == 0) return;

        // 1) Quantize the activations ONCE into the shared (primary-node) buffer;
        //    every node's GEMM reads from it (small: [total_tokens, K] in Q8_x).
        const size_t act_rb = cpu::ik::activation_row_bytes(type, K);
        const int64_t qa_bytes =
            static_cast<int64_t>(total_tokens) * static_cast<int64_t>(act_rb);
        uint8_t* qa = acquire_quant_act(qa_bytes);
        const bool prof = profile_enabled();
        auto t_q0 = std::chrono::steady_clock::now();
        cpu::cpu_gguf_quantize_activations(
            static_cast<const uint16_t*>(params.A_base), type, total_tokens, K, qa);
        if (prof)
            g_compute_ns.fetch_add(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t_q0).count()),
                std::memory_order_relaxed);

        // 2) Balanced (unaligned) output-row partition across the M nodes.
        partition_rows_balanced(N, M_, n_begin_, n_end_);

        // 3) Publish the GGUF job + run.
        job_kind_ = JobKind::kGguf;
        job_gguf_params_ = &params;
        job_gguf_type_ = type;
        job_gguf_qa_ = qa;
        job_gguf_act_rb_ = act_rb;

        if (M_ == 1) { run_slice(0); return; }
        jobs_remaining_.store(M_ - 1, std::memory_order_release);
        job_gen_.fetch_add(1, std::memory_order_acq_rel);
        run_slice(0);
        for (int spin = 0;
             jobs_remaining_.load(std::memory_order_acquire) != 0; ++spin) {
            if (spin >= 4096) { spin = 0; sched_yield(); }
#if defined(__SSE3__)
            else _mm_pause();
#endif
        }
    }

    // ── Activation / permutation (cheap; run on the primary node's pool) ─────

    void fused_swiglu(void* output, const void* input,
                      const FusedSwigluParams& params,
                      int elem_size_bytes, void* /*stream*/) override {
        cpu::cpu_swiglu(output, input, /*up=*/nullptr, params.num_tokens, params.d,
                        /*gate_up_interleaved=*/true, elem_size_bytes,
                        pools_[0].get(), params.swiglu_limit);
    }

    void moe_permute(void* permuted_input, int32_t* expert_offsets,
                     int32_t* src_to_dest_map, int32_t* permuted_idx,
                     const void* hidden_states, const int32_t* topk_indices,
                     int num_tokens, int topk, int hidden_dim,
                     int num_experts, int elem_size_bytes,
                     void* /*workspace*/, void* /*stream*/) override {
        cpu::cpu_moe_permute(permuted_input, expert_offsets, src_to_dest_map,
                             permuted_idx, hidden_states, topk_indices, num_tokens,
                             topk, hidden_dim, num_experts, elem_size_bytes,
                             pools_[0].get());
    }

    void moe_unpermute(void* output, const void* permuted_output,
                       const float* topk_weights, const int32_t* src_to_dest_map,
                       int num_tokens, int topk, int hidden_dim,
                       int elem_size_bytes, void* /*stream*/,
                       MoeCombineMode combine_mode =
                           MoeCombineMode::kReducedBf16) override {
        // C-6 CPU expert offload (canonical per-slot fold): forced CPU experts
        // join the SAME placement-invariant per-slot EP combine as GPU experts.
        // kPerSlotFp32/kPerSlotBf16 emit [num_tokens, topk, hidden] per-slot rows
        // (bit-identical to the GPU finalize_moe_routing_*_perslot kernels);
        // kReducedBf16 is the legacy host-reduced [num_tokens, hidden].
        if (combine_mode == MoeCombineMode::kPerSlotFp32
            || combine_mode == MoeCombineMode::kPerSlotBf16) {
            cpu::cpu_moe_unpermute_perslot(
                output, permuted_output, topk_weights, src_to_dest_map,
                num_tokens, topk, hidden_dim,
                /*fp32_output=*/combine_mode == MoeCombineMode::kPerSlotFp32,
                pools_[0].get());
            return;
        }
        cpu::cpu_moe_unpermute(output, permuted_output, topk_weights,
                               src_to_dest_map, num_tokens, topk, hidden_dim,
                               elem_size_bytes, pools_[0].get());
        // DEFERRED SEAM (same as NumaCpuExpertDevice): `output` holds the reduced
        // moe_output in HOST RAM on the primary node. Crossing it to the TP GPUs +
        // folding into the routed all-reduce (dispatch_moe.cpp allreduce_hidden) is
        // a SEPARATE SESSION (TD-CPU-EXPERT-HANDOFF). Left unimplemented here.
    }

    // ── Device memory (primary-node-local) ──────────────────────────────────

    void* device_alloc(size_t bytes) override {
        return alloc_on_node(deps_.nodes.front(), bytes);
    }
    void device_free(void* ptr) override {
        free_on_node(deps_.nodes.front(), ptr);
    }

private:
    // Per-node staging OCCUPANCY: maps a byte OFFSET in that node's staging arena
    // -> the slice key CURRENTLY resident at that offset, so re-staging is skipped
    // ONLY when the identical (proj, expert, slice) is still physically present.
    //
    // BUG FIX (TD-MULTINUMA-RESIDENCY): the prior cache keyed key -> offset, but a
    // given physical offset (e * slice_bytes) is OVERWRITTEN by whichever expert
    // occupies slot `e` this call, and the active-expert set / order changes every
    // step at B=1. A stale key -> offset entry then reported "resident" and SKIPPED
    // staging while the buffer actually held a DIFFERENT expert's weights (staged
    // by an intervening projection or step), computing an expert against another
    // expert's weights → corrupted FFN output → wrong routing downstream (measured:
    // forced-traj acceptance 1/100, misses halved). The unit tests missed it because
    // their two token counts force a buffer GROW between calls, which clears the map;
    // the engine holds token count constant so the buffer never grows and the stale
    // hits accumulate. Keying by offset (occupancy) makes a hit iff the buffer truly
    // still holds that key: correct under ANY active-set churn. Cleared on a grow.
    struct SliceKey {
        const void* src;  ///< source packed-projection base (identity of expert/proj)
        int n0, n1;       ///< row slice
        bool operator==(const SliceKey& o) const {
            return src == o.src && n0 == o.n0 && n1 == o.n1;
        }
    };

    // Persistent driver j (j = 1..M-1): park on the job generation, run slice j of
    // the published GEMM, signal done. Pinned to node j once. Mirrors
    // NumaThreadPool::worker_main one level up (across NUMA nodes).
    void driver_main(int j) {
        const int node = deps_.nodes[static_cast<size_t>(j)];
        const auto cpus = cpu::node_physical_cpus(node);
        if (!cpus.empty()) cpu::pin_thread_to_cpu(cpus.front());
        uint64_t last = 0;
        for (;;) {
            for (int spin = 0;; ++spin) {
                const uint64_t g = job_gen_.load(std::memory_order_acquire);
                if (g != last) { last = g; break; }
                if (spin >= 4096) { spin = 0; sched_yield(); }
#if defined(__SSE3__)
                else _mm_pause();
#endif
            }
            if (shutdown_.load(std::memory_order_acquire)) return;
            run_slice(j);
            jobs_remaining_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    // Run this device's published GEMM job for node-slice j (called inline for
    // j==0, by driver_main for j>=1). Reads the job_* fields published before the
    // generation bump. Dispatches by job kind (NVFP4 vs GGUF).
    void run_slice(int j) {
        const int nb = n_begin_[static_cast<size_t>(j)];
        const int ne = n_end_[static_cast<size_t>(j)];
        if (ne <= nb) return;  // empty slice (N < M) — nothing for this node
        if (job_kind_ == JobKind::kGguf) {
            compute_gguf_slice(j, *job_gguf_params_, nb, ne);
            return;
        }
        const Nvfp4GroupedGemmParams& p = *job_params_;
        compute_slice(j, p, job_Du_, job_Au_, p.N, p.K, p.num_experts,
                      job_src_pb_, nb, ne);
    }

    // Compute node j's N-slice [nb, ne) of all E experts: stage each expert's
    // slice into node j's LOCAL scratch, then run cpu_nvfp4_grouped_gemm over that
    // slab writing only columns [nb, ne) of the shared D.
    void compute_slice(int j, const Nvfp4GroupedGemmParams& params,
                       uint16_t* Du, const uint16_t* Au, int N, int K, int E,
                       int64_t src_pb, int nb, int ne) {
        const int Nj = ne - nb;
        const int64_t slice_pb = nvfp4_proj_bytes(Nj, K);

        // Acquire this node's staging region: arena scratch (preferred — pinned,
        // mbind'd) or a private node-local buffer grown on demand. Need E slices
        // back-to-back (each slice_pb, 128-aligned) PLUS a tiny sub-D output
        // [tokens_total, Nj] BF16 so the kernel writes a contiguous slice output
        // we then scatter into D's columns.
        const int total_tokens = params.expert_offsets[E];
        const int64_t weights_bytes =
            static_cast<int64_t>(E) * slice_pb;
        const int64_t subD_bytes =
            static_cast<int64_t>(total_tokens) * Nj * static_cast<int64_t>(sizeof(uint16_t));
        const int64_t need = weights_bytes + subD_bytes;

        uint8_t* base = acquire_stage(j, need);

        const bool prof = profile_enabled();
        auto t_stage0 = std::chrono::steady_clock::now();

        // Stage each expert's slice (skip if resident at the same offset).
        std::vector<cpu::PackedProjection> projs(static_cast<size_t>(E));
        auto& occ = occupant_[static_cast<size_t>(j)];
        uint64_t staged = 0;
        for (int e = 0; e < E; ++e) {
            const int m0 = params.expert_offsets[e];
            const int m1 = params.expert_offsets[e + 1];
            const size_t off = static_cast<size_t>(e) * static_cast<size_t>(slice_pb);
            uint8_t* dst = base + off;
            const auto* src = static_cast<const uint8_t*>(params.B_ptrs[e]);
            if (m1 > m0) {  // only stage experts that route tokens this call
                // Stage iff this offset does NOT already physically hold this exact
                // (src, slice) — occupancy keyed by offset, correct under churn.
                SliceKey key{src, nb, ne};
                auto it = occ.find(off);
                if (it == occ.end() || !(it->second == key)) {
                    stage_proj_slice(dst, src, src_pb, N, K, nb, ne);
                    occ[off] = key;
                    staged += static_cast<uint64_t>(slice_pb);
                }
            }
            projs[static_cast<size_t>(e)] =
                cpu::PackedProjection{dst, static_cast<size_t>(slice_pb), Nj, K};
        }
        cpu::CpuNvfp4ExpertWeights weights{projs.data(), E};

        auto t_stage1 = std::chrono::steady_clock::now();

        // Compute the slice into the node-local sub-D [total_tokens, Nj], then
        // scatter its columns into the shared D [total_tokens, N] at offset nb.
        uint16_t* subD = reinterpret_cast<uint16_t*>(base + weights_bytes);
        cpu_nvfp4_grouped_gemm(subD, Au, weights, params.expert_offsets, Nj, K,
                               /*elem_size_bytes=*/2, pools_[static_cast<size_t>(j)].get());
        if (prof) {
            auto t_compute1 = std::chrono::steady_clock::now();
            g_stage_ns.fetch_add(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_stage1 - t_stage0).count()), std::memory_order_relaxed);
            g_compute_ns.fetch_add(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_compute1 - t_stage1).count()), std::memory_order_relaxed);
            g_staged_bytes.fetch_add(staged, std::memory_order_relaxed);
        }
        for (int m = 0; m < total_tokens; ++m) {
            std::memcpy(Du + static_cast<size_t>(m) * N + nb,
                        subD + static_cast<size_t>(m) * Nj,
                        static_cast<size_t>(Nj) * sizeof(uint16_t));
        }
    }

    // Compute node j's N-slice [nb, ne) of a GGUF grouped GEMM: stage each
    // expert's contiguous packed-weight-row slice into node j's LOCAL scratch
    // (rows are self-contained — a plain contiguous byte range, no atom align),
    // then run the slice GEMM over the SHARED pre-quantized activations writing
    // only output columns [nb, ne) of the shared D. Disjoint columns -> no reduce.
    void compute_gguf_slice(int j, const GgufGroupedGemmParams& params,
                            int nb, int ne) {
        const int N = params.N;
        const int K = params.K;
        const int E = params.num_experts;
        const int nslice = ne - nb;
        const size_t wrow = cpu::ik::weight_row_bytes(job_gguf_type_, K);
        const int64_t slice_bytes = static_cast<int64_t>(nslice) * static_cast<int64_t>(wrow);

        // Node-local staging: E expert slabs back-to-back, each `slice_bytes`.
        const int64_t need = static_cast<int64_t>(E) * slice_bytes;
        uint8_t* base = acquire_stage(j, need);

        const bool prof = profile_enabled();
        auto t_stage0 = std::chrono::steady_clock::now();

        std::vector<const void*> slabs(static_cast<size_t>(E), nullptr);
        auto& occ = occupant_[static_cast<size_t>(j)];
        uint64_t staged = 0;
        for (int e = 0; e < E; ++e) {
            const int m0 = params.expert_offsets[e];
            const int m1 = params.expert_offsets[e + 1];
            const size_t off = static_cast<size_t>(e) * static_cast<size_t>(slice_bytes);
            uint8_t* dst = base + off;
            const auto* src = static_cast<const uint8_t*>(params.B_ptrs[e]);
            if (m1 > m0) {  // only stage experts that route tokens this call
                // Stage iff this offset does NOT already physically hold this exact
                // (src, slice) — occupancy keyed by offset, correct under churn.
                SliceKey key{src, nb, ne};
                auto it = occ.find(off);
                if (it == occ.end() || !(it->second == key)) {
                    std::memcpy(dst, src + static_cast<size_t>(nb) * wrow,
                                static_cast<size_t>(slice_bytes));
                    occ[off] = key;
                    staged += static_cast<uint64_t>(slice_bytes);
                }
            }
            slabs[static_cast<size_t>(e)] = dst;
        }

        auto t_stage1 = std::chrono::steady_clock::now();

        cpu::cpu_gguf_grouped_gemm_slice(
            static_cast<uint16_t*>(params.D_base), job_gguf_qa_, job_gguf_type_,
            slabs.data(), params.expert_offsets, E, N, K, nb, nslice,
            pools_[static_cast<size_t>(j)].get());

        if (prof) {
            auto t_compute1 = std::chrono::steady_clock::now();
            g_stage_ns.fetch_add(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_stage1 - t_stage0).count()), std::memory_order_relaxed);
            g_compute_ns.fetch_add(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_compute1 - t_stage1).count()), std::memory_order_relaxed);
            g_staged_bytes.fetch_add(staged, std::memory_order_relaxed);
        }
    }

    // Acquire >= `bytes` of shared (primary-node) scratch for the quantized
    // activations, grown on demand. Shared read-only across all node GEMMs.
    uint8_t* acquire_quant_act(int64_t bytes) {
        if (quant_act_cap_ < bytes) {
            if (quant_act_buf_)
                free_on_node(deps_.nodes.front(), quant_act_buf_);
            const int64_t cap = bytes + bytes / 4 + 4096;
            quant_act_buf_ = static_cast<uint8_t*>(
                alloc_on_node(deps_.nodes.front(), static_cast<size_t>(cap)));
            quant_act_cap_ = cap;
        }
        return quant_act_buf_;
    }

    // Acquire >= `bytes` of node-local staging for node j. Prefer the arena's
    // front-of-slab scratch (pinned + mbind'd) when it is large enough; else a
    // private node-local buffer grown on demand (a grow clears the occupancy map).
    // Whenever the returned BASE pointer changes (grow, or switching between arena
    // scratch and the private buffer), the occupancy map is invalidated — its
    // offsets are only meaningful relative to a fixed base.
    uint8_t* acquire_stage(int j, int64_t bytes) {
        const int node = deps_.nodes[static_cast<size_t>(j)];
        uint8_t* base = nullptr;
        if (deps_.arena) {
            void* sc = deps_.arena->node_scratch(node);
            const size_t sb = deps_.arena->node_scratch_bytes(node);
            if (sc && sb >= static_cast<size_t>(bytes))
                base = static_cast<uint8_t*>(sc);
        }
        if (!base) {
            if (stage_caps_[static_cast<size_t>(j)] < bytes) {
                if (stage_bufs_[static_cast<size_t>(j)])
                    free_on_node(node, stage_bufs_[static_cast<size_t>(j)]);
                // Grow with headroom to amortize re-staging across token counts.
                const int64_t cap = bytes + bytes / 4 + 4096;
                stage_bufs_[static_cast<size_t>(j)] = static_cast<uint8_t*>(
                    alloc_on_node(node, static_cast<size_t>(cap)));
                stage_caps_[static_cast<size_t>(j)] = cap;
            }
            base = stage_bufs_[static_cast<size_t>(j)];
        }
        if (base != stage_base_[static_cast<size_t>(j)]) {
            occupant_[static_cast<size_t>(j)].clear();  // base changed → offsets moot
            stage_base_[static_cast<size_t>(j)] = base;
        }
        return base;
    }

    // ── node-local raw alloc/free (NumaManager when present, else malloc) ─────

    void* alloc_on_node(int node, size_t bytes) {
        if (bytes == 0) return nullptr;
        if (deps_.numa) {
            try {
                memory::NumaBuffer buf = deps_.numa->allocate_on_node(bytes, node);
                if (buf.data) {
                    std::lock_guard<std::mutex> lock(alloc_mu_);
                    allocs_.emplace(buf.data, buf);
                }
                return buf.data;
            } catch (const std::exception&) { return nullptr; }
        }
        return std::malloc(bytes);
    }

    void free_on_node(int /*node*/, void* ptr) {
        if (!ptr) return;
        if (deps_.numa) {
            memory::NumaBuffer buf{};
            {
                std::lock_guard<std::mutex> lock(alloc_mu_);
                auto it = allocs_.find(ptr);
                if (it == allocs_.end()) return;  // not ours / already freed
                buf = it->second;
                allocs_.erase(it);
            }
            deps_.numa->free(buf);
        } else {
            std::free(ptr);
        }
    }

    config::GpuRef gpu_;
    MultiNumaCpuExpertDeps deps_;
    int M_ = 1;
    std::vector<std::unique_ptr<cpu::NumaThreadPool>> pools_;  ///< one per node

    // Private node-local staging (used when arena scratch is absent / too small).
    std::vector<uint8_t*> stage_bufs_;
    std::vector<int64_t>  stage_caps_;
    std::vector<uint8_t*> stage_base_;   ///< last base acquire_stage returned per node
    // Per node: OFFSET -> slice key currently staged at that offset (occupancy).
    std::vector<std::unordered_map<size_t, SliceKey>> occupant_;

    // Persistent cross-node driver threads (j = 1..M-1) + generation handshake.
    std::vector<std::thread> drivers_;
    std::atomic<uint64_t> job_gen_{0};          ///< bumped to dispatch a job
    std::atomic<int>      jobs_remaining_{0};    ///< drivers still running the job
    std::atomic<bool>     shutdown_{false};

    // Published job state (written before the job_gen_ bump; read by run_slice).
    enum class JobKind { kNvfp4, kGguf };
    JobKind job_kind_ = JobKind::kNvfp4;
    const Nvfp4GroupedGemmParams* job_params_ = nullptr;
    uint16_t* job_Du_ = nullptr;
    const uint16_t* job_Au_ = nullptr;
    int64_t   job_src_pb_ = 0;
    std::vector<int> n_begin_, n_end_;           ///< per-node row slice [nb,ne)

    // GGUF job state (written before the job_gen_ bump; read by compute_gguf_slice).
    const GgufGroupedGemmParams* job_gguf_params_ = nullptr;
    cpu::ik::GgufType job_gguf_type_ = cpu::ik::GgufType::q8_0;
    const uint8_t* job_gguf_qa_ = nullptr;       ///< shared pre-quantized acts
    size_t job_gguf_act_rb_ = 0;
    // Shared (primary-node) quantized-activation buffer, grown on demand.
    uint8_t* quant_act_buf_ = nullptr;
    int64_t  quant_act_cap_ = 0;

    std::mutex alloc_mu_;
    std::unordered_map<void*, memory::NumaBuffer> allocs_;
};

std::unique_ptr<ExpertDevice> make_multi_numa_cpu_expert_device(
    config::GpuRef gpu, MultiNumaCpuExpertDeps deps) {
    return std::make_unique<MultiNumaCpuExpertDevice>(gpu, std::move(deps));
}

}  // namespace layerstorm::compute

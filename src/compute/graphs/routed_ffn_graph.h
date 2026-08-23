#pragma once

//==============================================================================
// Routed Expert FFN CUDA Graph Runner  (TD-DECODE-FFN-GRAPH, EXPERIMENT)
//
// Captures the per-GPU routed-expert FFN compute chain of the DECODE path into
// a replayable CUDA graph:
//   Step 2  permute            (moe_permute)
//   Step 3  gate grouped GEMM  (+ populate_meta, gather_alphas, activation_quant)
//   Step 3b up   grouped GEMM
//   Step 4  fused SwiGLU + NVFP4 quant
//   Step 5  down grouped GEMM
//   Step 6  unpermute          (moe_unpermute)
//
// This DELIBERATELY violates INV-0.6 ("graphs never capture expert FFN"). It is
// sanctioned ONLY for this experiment to measure whether removing host launch
// overhead helps the decode path (the GEMMs are GPU prologue-floor bound, so the
// expected win is host-launch overhead only).
//
// Decode batch=1 makes the shape FIXED: top_k experts each see M_e=1 token,
// expert_offsets/problem_sizes/sf_offsets are constant across replays. The only
// data that changes per step is the per-expert routing (topk_indices/weights)
// and the resident-expert B/scale-B pointers — fed through FIXED-ADDRESS
// indirection buffers (vLLM pattern): the device pointer arrays live at stable
// addresses, the host source buffers are pinned and refilled before each replay,
// and the captured H2D copies push the fresh routing into the graph.
//
// Non-resident-on-this-GPU experts point their B_ptr at the existing zero buffer
// (zero contribution → the routed allreduce sums to the identical result).
//
// Capture-safety note: all CUTLASS grouped-GEMM host setup (get_workspace_size /
// can_implement / initialize) is read-only CUDA queries + stream-ordered
// workspace init given a PRE-ALLOCATED workspace, and CUB radix sort in permute
// uses a pre-provided workspace — so only device kernel launches are captured.
// The b_ptrs H2D copies source from PINNED host staging (pageable H2D would force
// a synchronous staging copy that is illegal during capture).
//
// Thread safety: NOT thread-safe (INV-3.4.2 — single-threaded orchestrator).
//==============================================================================

#include <cstdint>

namespace layerstorm::compute {

// Opaque capture/replay helper. All raw CUDA lives in the .cpp (CUDA TU). The
// caller (dispatch_moe_internal, a non-CUDA .cpp) drives capture by bracketing
// the existing Step 2..6 kernel launches between begin_capture() and
// end_capture(): the launches issued on `stream` in between are recorded into a
// graph and instantiated. Subsequent calls replay() the instantiated graph.
class RoutedFfnGraphRunner {
public:
    RoutedFfnGraphRunner() = default;
    ~RoutedFfnGraphRunner() { destroy(); }

    RoutedFfnGraphRunner(const RoutedFfnGraphRunner&) = delete;
    RoutedFfnGraphRunner& operator=(const RoutedFfnGraphRunner&) = delete;
    RoutedFfnGraphRunner(RoutedFfnGraphRunner&&) = delete;
    RoutedFfnGraphRunner& operator=(RoutedFfnGraphRunner&&) = delete;

    /// GG-S1 pre-capture warmup hook (TD-GG5-MMQ-CAPTURE-WARMUP, engine side).
    /// The home for any one-time, NON-capturable device setup the captured
    /// sequence needs before begin_capture() — specifically the mmq
    /// cudaFuncSetAttribute(max dynamic shared mem) that the GGUF int kernel
    /// fires LAZILY on its first large-M (mmq) launch. That runtime API is
    /// ILLEGAL inside a stream capture, so it must be issued here, ahead of the
    /// capture, not discovered mid-sequence.
    ///
    /// ENGINE-SIDE NO-OP today, deliberately: the decode FFN graph only ever
    /// hits mmvq (decode M_e=1 ⇒ avg_m≪8 ⇒ the host pick never selects mmq), so
    /// there is nothing to warm for the current captured path; and the
    /// GemmKernels warmup ENTRY POINT (gguf_grouped_gemm_warmup(type, tile_cfg))
    /// does not exist yet — that remains a GemmKernels-repo follow-up. The hook
    /// is wired now so the seam is in place the moment a prefill / large-M GGUF
    /// GEMM is ever captured. Always returns true (nothing to do == success).
    bool warmup(void* /*stream*/) { return true; }

    /// Begin capture on `stream` (ThreadLocal mode). Returns true on success.
    /// On failure (e.g. capture not supported), logs and returns false; the
    /// caller should then run the chain live (no graph).
    bool begin_capture(void* stream);

    /// End capture on `stream`, instantiate the captured graph. Returns true on
    /// success. On failure the captured ops were NOT executed; the caller must
    /// re-run the chain live to preserve correctness (see captured_executed()).
    bool end_capture(void* stream);

    /// Replay the instantiated graph on `stream`. Precondition: is_captured().
    bool replay(void* stream);

    /// Free CUDA graph resources. Idempotent.
    void destroy();

    bool is_captured() const { return captured_; }

    /// Last begin/end capture error string (empty if none).
    const char* last_error() const { return last_error_; }

private:
    bool   captured_   = false;
    void*  graph_      = nullptr;  // cudaGraph_t
    void*  graph_exec_ = nullptr;  // cudaGraphExec_t
    const char* last_error_ = "";
};

}  // namespace layerstorm::compute

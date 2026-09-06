#pragma once

//==============================================================================
// Decode span-graph runner — INV-0.6(b) extension (P-29 step 7)
//
// Generalizes the per-layer TQ decode-chain graph pattern
// (tq_sm120_attention_device.cpp:559-625) into a keyed, fingerprinted
// capture/replay cache that any B=1 decode-path launch SPAN can use:
//
//   span_graphs.run(stream, key, fp, [&]{ ...existing launches... });
//
// Semantics per (key, fingerprint):
//   1st sighting of key     : eager (warmup — lazy allocs / func attrs fire
//                             outside capture, TD-GG5-MMQ-CAPTURE-WARMUP class)
//   2nd sighting            : ThreadLocal stream-capture of the emit lambda,
//                             instantiate, LAUNCH the new graph (the captured
//                             work executes via the replay — capture alone
//                             executes nothing)
//   later, fp match         : cudaGraphLaunch (one host call replaces N)
//   later, fp drift         : new capture into a small per-key variant cache
//                             (cap kVariantCap). At cap, a NEW fingerprint
//                             evicts the least-recently-replayed variant —
//                             but only on its SECOND consecutive sighting
//                             (one eager run of hysteresis), so ≥5 fps
//                             ALTERNATING per run degrade to eager with no
//                             capture churn, while traffic that MOVES to a
//                             new shape (e.g. a long-lived server whose
//                             kIndexer need_pages bucket grows) recovers
//                             replay instead of ratcheting to permanent
//                             eager (P-29 step 21,
//                             TD-SPAN-INDEXER-VARIANT-RATCHET)
//   any capture failure     : re-run emit eagerly (correctness first), mark
//                             the key dead, warn once — fail-safe, the token
//                             is never lost
//
// Requirements on the emit lambda (the span contract):
//   - launches only onto `stream` (single-stream span; cross-stream events,
//     NCCL, cudaGraphLaunch, allocation, D2H-consumed-by-host all stay OUT)
//   - every pointer/scalar baked into its launch params is listed in the
//     fingerprint; per-token variation flows through DEVICE BUFFER CONTENT
//     at stable addresses (the INV-0.6(b) device-read discipline)
//   - host side effects are idempotent (a failed capture re-runs it)
//
// Numerics: bit-identical by construction — the graph records the very same
// kernels in the very same stream order with the very same parameters; only
// the submission mechanism changes. LS_DECODE_CHAIN_GRAPH=0 restores the
// eager path byte-for-byte.
//
// INV-GPU-1: all raw CUDA lives in decode_span_graph.cpp (a designated CUDA
// TU); this header is CUDA-free so non-CUDA TUs (dcp_executor, arch_*.cpp,
// attention_driver) can own and drive the runner.
//
// Thread safety: NOT thread-safe (INV-3.4.2 — the single daemon submit
// thread owns all spans).
//==============================================================================

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace layerstorm::compute {

class DecodeSpanGraphs {
public:
    DecodeSpanGraphs();
    ~DecodeSpanGraphs();
    DecodeSpanGraphs(const DecodeSpanGraphs&) = delete;
    DecodeSpanGraphs& operator=(const DecodeSpanGraphs&) = delete;

    /// Span identity: chain kind (high bits) | layer | rank. Helpers below.
    enum Chain : uint32_t {
        kKdaLayer   = 1,  // KDA per-layer decode chain (arch_glm5_next)
        kMlaPrefix  = 2,  // MLA common prefix (arch_mla execute_common_prefix)
        kGate       = 3,  // fused gate: mhc_pre(ffn)+norm+router+topk (driver)
        kMhcPreAttn = 4,  // attention-stage mHC collapse pair (driver)
        kIndexer    = 5,  // DSA indexer produce chain (arch_mla)
    };
    static constexpr uint32_t make_key(Chain c, int layer, int rank) {
        return (static_cast<uint32_t>(c) << 24)
             | ((static_cast<uint32_t>(layer) & 0xFFFF) << 8)
             | (static_cast<uint32_t>(rank) & 0xFF);
    }

    /// Fingerprint of everything baked into the span's launch params.
    struct Fp {
        static constexpr int kMaxPtrs = 40;
        static constexpr int kMaxScalars = 8;
        const void* p[kMaxPtrs] = {};
        int64_t s[kMaxScalars] = {};
        int np = 0, ns = 0;
        void add(const void* ptr) { if (np < kMaxPtrs) p[np++] = ptr; }
        void add_s(int64_t v) { if (ns < kMaxScalars) s[ns++] = v; }
        bool operator==(const Fp& o) const;
    };

    /// Master switch: LS_DECODE_CHAIN_GRAPH (default ON, =0 restores the
    /// eager path) AND an explicit arm() from the engine. Arming exists
    /// because unit-test harnesses run this code with NULL device backends
    /// whose "streams" are heap ints — a stream-capture call on one is a
    /// segfault, so capture must be opt-in from a site that KNOWS real CUDA
    /// is live (CommandDispatcher, deps.cuda_kernels_enabled).
    bool enabled() const { return enabled_ && armed_ && bypass_ == 0; }
    void arm(bool on) { armed_ = on; }

    /// P-29 step 13 phase B: speculative-verify rows run the SAME chains at
    /// row_offset>0 pointer shapes — letting them capture would spend the
    /// kVariantCap variant budget on transient row shapes and tip hot keys
    /// to permanent eager. The dispatcher brackets verify-row dispatches
    /// with this RAII bypass: run() executes emit() eagerly (counted in
    /// stats().eager), captures nothing, replays nothing.
    class ScopedBypass {
    public:
        explicit ScopedBypass(DecodeSpanGraphs& g) : g_(g) { ++g_.bypass_; }
        ~ScopedBypass() { --g_.bypass_; }
        ScopedBypass(const ScopedBypass&) = delete;
        ScopedBypass& operator=(const ScopedBypass&) = delete;
    private:
        DecodeSpanGraphs& g_;
    };

    /// Execute the span (see file header for the state machine).
    void run(void* stream, uint32_t key, const Fp& fp,
             const std::function<void()>& emit);

    /// Introspection for tests / engagement proof.
    struct Stats {
        uint64_t eager = 0;      // spans run eagerly (warmup/dead/cap)
        uint64_t captured = 0;   // graphs captured+instantiated
        uint64_t replayed = 0;   // cudaGraphLaunch replays
        uint64_t failed = 0;     // capture failures (fail-safe eager)
        uint64_t evicted = 0;    // LRU variant evictions at kVariantCap (P-29 step 21)
    };
    const Stats& stats() const { return stats_; }
    size_t num_keys() const { return keys_.size(); }

private:
    struct Entry {
        void* exec = nullptr;  // cudaGraphExec_t
        Fp fp;
        uint64_t last_use = 0; // clock_ stamp of last replay (LRU eviction)
    };
    struct KeyState {
        int runs = 0;
        bool dead = false;
        std::vector<Entry> variants;
        // P-29 step 21 (TD-SPAN-INDEXER-VARIANT-RATCHET) hysteresis: a new
        // fp at kVariantCap is remembered here on its first miss; only a
        // consecutive second miss of the SAME fp evicts the LRU variant.
        Fp pending_fp;
        bool has_pending = false;
    };
    static constexpr int kVariantCap = 4;

    std::unordered_map<uint32_t, KeyState> keys_;
    uint64_t clock_ = 0;  ///< monotonic replay/capture stamp for LRU
    Stats stats_;
    bool enabled_ = false;
    bool armed_ = false;
    int bypass_ = 0;  ///< P-29 step 13: ScopedBypass depth (0 = normal)
};

}  // namespace layerstorm::compute

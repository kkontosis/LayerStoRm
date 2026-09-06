//==============================================================================
// Decode span-graph runner (INV-0.6(b) extension) — implementation.
//
// Designated CUDA TU (INV-GPU-1): the ONLY file in this feature that touches
// the CUDA runtime. See decode_span_graph.h for the contract.
//==============================================================================

#include "compute/graphs/decode_span_graph.h"

#include <atomic>
#include <cstdlib>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <cstring>

namespace layerstorm::compute {

bool DecodeSpanGraphs::Fp::operator==(const Fp& o) const {
    if (np != o.np || ns != o.ns) return false;
    for (int i = 0; i < np; ++i)
        if (p[i] != o.p[i]) return false;
    for (int i = 0; i < ns; ++i)
        if (s[i] != o.s[i]) return false;
    return true;
}

DecodeSpanGraphs::DecodeSpanGraphs() {
    // House-style kill switch: LS_DECODE_CHAIN_GRAPH=0 restores the eager
    // path byte-for-byte (default ON, P-29 step 7).
    const char* v = std::getenv("LS_DECODE_CHAIN_GRAPH");
    enabled_ = !(v && *v && v[0] == '0');
}

DecodeSpanGraphs::~DecodeSpanGraphs() {
    for (auto& [key, ks] : keys_) {
        for (auto& e : ks.variants) {
            if (e.exec)
                cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(e.exec));
        }
    }
    keys_.clear();
}

void DecodeSpanGraphs::run(void* stream, uint32_t key, const Fp& fp,
                           const std::function<void()>& emit) {
    auto s = static_cast<cudaStream_t>(stream);
    // P-29 step 13 diagnostics: LS_SPAN_STATS=1 logs engagement every 8192 runs.
    static const bool span_stats = [] {
        const char* e = std::getenv("LS_SPAN_STATS");
        return e && *e == '1';
    }();
    if (span_stats) {
        static std::atomic<uint64_t> calls{0};
        if ((calls.fetch_add(1) & 8191) == 8191)
            spdlog::info("[span-stats] keys={} eager={} captured={} "
                         "replayed={} failed={} evicted={}",
                         keys_.size(), stats_.eager, stats_.captured,
                         stats_.replayed, stats_.failed, stats_.evicted);
    }
    if (!enabled_ || bypass_ != 0) {
        // P-29 step 13: ScopedBypass (speculative-verify rows) — eager, no
        // capture, no replay, no variant-budget spend.
        if (bypass_ != 0) ++stats_.eager;
        emit();
        return;
    }
    auto& ks = keys_[key];

    // Replay on fingerprint hit.
    if (!ks.dead) {
        for (auto& e : ks.variants) {
            if (e.fp == fp) {
                cudaError_t err = cudaGraphLaunch(
                    static_cast<cudaGraphExec_t>(e.exec), s);
                if (err == cudaSuccess) {
                    ++stats_.replayed;
                    e.last_use = ++clock_;      // LRU stamp (P-29 step 21)
                    ks.has_pending = false;     // miss streak broken
                    return;
                }
                // Replay failure: fail safe — eager this step, key dead.
                spdlog::warn("DecodeSpanGraphs: replay failed for key {:#x} "
                             "({}) — span falls back EAGER permanently",
                             key, cudaGetErrorString(err));
                ks.dead = true;
                ++stats_.failed;
                emit();
                ++stats_.eager;
                return;
            }
        }
    }

    ++ks.runs;
    bool may_capture = !ks.dead && ks.runs >= 2;
    if (may_capture && static_cast<int>(ks.variants.size()) >= kVariantCap) {
        // P-29 step 21 (TD-SPAN-INDEXER-VARIANT-RATCHET): the cap used to be
        // a one-way ratchet — variants were never evicted, so a long-lived
        // server whose fingerprints outgrew the cache (e.g. kIndexer
        // need_pages crossing >4 distinct 8192-token buckets) fell to
        // PERMANENT eager on every affected key. Now a new fp at cap evicts
        // the least-recently-replayed variant — but only on its SECOND
        // consecutive miss (one eager run of hysteresis): fps that merely
        // ALTERNATE never see two consecutive misses and stay eager with no
        // capture churn; traffic that MOVES to a new shape recaptures on
        // its second token. Identity-safe: replay-vs-eager changes only the
        // submission mechanism (see file header).
        if (ks.has_pending && ks.pending_fp == fp) {
            size_t lru = 0;
            for (size_t i = 1; i < ks.variants.size(); ++i)
                if (ks.variants[i].last_use < ks.variants[lru].last_use)
                    lru = i;
            // Safe while in flight: cudaGraphExecDestroy defers destruction
            // until outstanding launches of this exec complete.
            cudaGraphExecDestroy(
                static_cast<cudaGraphExec_t>(ks.variants[lru].exec));
            ks.variants.erase(ks.variants.begin()
                              + static_cast<std::ptrdiff_t>(lru));
            ks.has_pending = false;
            ++stats_.evicted;
        } else {
            ks.pending_fp = fp;
            ks.has_pending = true;
            may_capture = false;
        }
    }
    if (!may_capture) {
        emit();
        ++stats_.eager;
        return;
    }

    // Capture attempt (ThreadLocal: other streams keep working; unsafe APIs
    // from this thread would invalidate the capture, which we detect below).
    cudaError_t err = cudaStreamBeginCapture(
        s, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        spdlog::warn("DecodeSpanGraphs: begin-capture failed for key {:#x} "
                     "({}) — span falls back EAGER permanently",
                     key, cudaGetErrorString(err));
        ks.dead = true;
        ++stats_.failed;
        emit();
        ++stats_.eager;
        return;
    }
    try {
        emit();  // recorded, NOT executed
    } catch (...) {
        // Abort the capture so the stream is not left in capture mode, then
        // rethrow — the span's own error handling owns the failure.
        cudaGraph_t aborted = nullptr;
        (void)cudaStreamEndCapture(s, &aborted);
        if (aborted) cudaGraphDestroy(aborted);
        (void)cudaGetLastError();
        ks.dead = true;
        ++stats_.failed;
        throw;
    }
    cudaGraph_t graph = nullptr;
    err = cudaStreamEndCapture(s, &graph);
    if (err != cudaSuccess || !graph) {
        // The recorded work never ran — re-run it eagerly (emit must be
        // host-idempotent, see the span contract).
        spdlog::warn("DecodeSpanGraphs: end-capture failed for key {:#x} "
                     "({}) — span falls back EAGER permanently",
                     key, cudaGetErrorString(err));
        if (graph) cudaGraphDestroy(graph);
        // Clear the sticky capture-invalidation error state, if any.
        (void)cudaGetLastError();
        ks.dead = true;
        ++stats_.failed;
        emit();
        ++stats_.eager;
        return;
    }
    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiateWithFlags(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess || !exec) {
        spdlog::warn("DecodeSpanGraphs: instantiate failed for key {:#x} "
                     "({}) — span falls back EAGER permanently",
                     key, cudaGetErrorString(err));
        ks.dead = true;
        ++stats_.failed;
        emit();
        ++stats_.eager;
        return;
    }
    // Launch the fresh graph — this executes the captured work for THIS step.
    err = cudaGraphLaunch(exec, s);
    if (err != cudaSuccess) {
        spdlog::warn("DecodeSpanGraphs: first launch failed for key {:#x} "
                     "({}) — span falls back EAGER permanently",
                     key, cudaGetErrorString(err));
        cudaGraphExecDestroy(exec);
        ks.dead = true;
        ++stats_.failed;
        emit();
        ++stats_.eager;
        return;
    }
    Entry e;
    e.exec = exec;
    e.fp = fp;
    e.last_use = ++clock_;
    ks.variants.push_back(e);
    ks.has_pending = false;
    ++stats_.captured;
}

}  // namespace layerstorm::compute

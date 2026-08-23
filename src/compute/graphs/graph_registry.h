#pragma once

//==============================================================================
// CUDA Graph Registry
//
// Lookup table for captured CUDA graph runners, keyed by (type, gpu_idx,
// batch_size).  The registry owns all inserted entries and calls their
// destroy callbacks on removal or destruction.
//
// The registry does NOT capture or replay graphs — concrete runners
// (DecodeGraphRunner, DcpAllreduceGraphRunner, …) handle their own
// capture/replay.  The orchestrator queries the registry to obtain the
// appropriate runner, then calls update()/replay() directly.
//
// Thread safety: NOT thread-safe (INV-3.4.2 — single-threaded orchestrator).
// INV-0.6 (refined): bounded fixed-shape subgraphs only — never a unified
// whole-layer graph, never transfers. Permitted: attention(+gate), and the
// decode routed-expert FFN (permute→gate/up→SwiGLU→down→unpermute, flag-gated
// LS_MOE_FFN_GRAPH; allreduce tail stays live). See spec/INVARIANTS.md.
//==============================================================================

#include <any>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace layerstorm::compute {

// ── Graph types ─────────────────────────────────────────────────────────────

/// Enumeration of CUDA graph types managed by the registry.
/// INV-0.6 (refined): bounded fixed-shape subgraphs only (attention, decode
/// routed-expert FFN) — never a unified whole-layer graph, never transfers.
enum class GraphType : uint8_t {
    kAttentionDecode = 0,   // DecodeGraphRunner (#38)
    kDcpAllreduce    = 1,   // DcpAllreduceGraphRunner (#39)
    kCount           = 2
};

/// Human-readable name for GraphType (for logging/diagnostics).
const char* graph_type_name(GraphType type);

// ── Composite key ───────────────────────────────────────────────────────────

/// Lookup key for the graph registry.
/// batch_size is part of the key because CUDA graph grid dimensions are
/// baked at capture time (DecodeGraphConfig::batch_size).
struct GraphKey {
    GraphType type{};
    int       gpu_idx{};
    int       batch_size{};

    bool operator==(const GraphKey&) const = default;
};

// ── Graph entry ─────────────────────────────────────────────────────────────

/// Type-erased storage for a captured CUDA graph runner.
/// The registry owns the runner and will call `destroy` on clear/destruction.
struct GraphEntry {
    /// Type-erased runner (e.g. DecodeGraphRunner*).
    /// Retrieved via std::any_cast<ConcreteType*>(entry.runner).
    std::any runner;

    /// Destroy callback: clean up CUDA resources and free the runner.
    /// Called by the registry during remove() and clear().
    using DestroyFn = std::function<void(std::any&)>;
    DestroyFn destroy;
};

// ── GraphRegistry ───────────────────────────────────────────────────────────

class GraphRegistry {
public:
    GraphRegistry() = default;
    ~GraphRegistry();

    // Non-copyable, non-movable (owns CUDA resources via callbacks)
    GraphRegistry(const GraphRegistry&) = delete;
    GraphRegistry& operator=(const GraphRegistry&) = delete;
    GraphRegistry(GraphRegistry&&) = delete;
    GraphRegistry& operator=(GraphRegistry&&) = delete;

    // ── Insert ──────────────────────────────────────────────────────────

    /// Insert a graph runner into the registry.
    /// @throws std::runtime_error if key already exists.
    void insert(GraphKey key, GraphEntry entry);

    // ── Lookup ──────────────────────────────────────────────────────────

    /// Find a graph runner by key. Returns nullptr if not found.
    GraphEntry* find(const GraphKey& key);
    const GraphEntry* find(const GraphKey& key) const;

    /// Get a graph runner by key.
    /// @throws std::runtime_error if not found.
    GraphEntry& get(const GraphKey& key);
    const GraphEntry& get(const GraphKey& key) const;

    /// Check if a key exists.
    bool contains(const GraphKey& key) const;

    // ── Typed convenience accessors ─────────────────────────────────────

    /// Find and cast a runner to the concrete type. Returns nullptr if key
    /// not found. Throws std::bad_any_cast if the stored type doesn't match.
    template <typename T>
    T* find_as(const GraphKey& key) {
        auto* entry = find(key);
        if (!entry) return nullptr;
        return std::any_cast<T*>(entry->runner);
    }

    template <typename T>
    const T* find_as(const GraphKey& key) const {
        const auto* entry = find(key);
        if (!entry) return nullptr;
        return std::any_cast<T*>(entry->runner);
    }

    // ── Removal ─────────────────────────────────────────────────────────

    /// Remove and destroy a single graph runner.
    /// No-op if key doesn't exist.
    void remove(const GraphKey& key);

    /// Remove and destroy all graph runners.
    void clear();

    // ── Queries ─────────────────────────────────────────────────────────

    size_t size() const;
    bool empty() const;

    /// Return all keys in the registry.
    std::vector<GraphKey> keys() const;

    /// Count entries of a specific graph type.
    size_t count_by_type(GraphType type) const;

    /// Return all keys matching a specific graph type.
    std::vector<GraphKey> keys_by_type(GraphType type) const;

private:
    struct GraphKeyHash {
        size_t operator()(const GraphKey& k) const noexcept {
            return std::hash<uint32_t>{}(
                (static_cast<uint32_t>(k.type) << 24) |
                (static_cast<uint32_t>(k.gpu_idx & 0xFF) << 16) |
                (static_cast<uint32_t>(k.batch_size & 0xFFFF)));
        }
    };

    std::unordered_map<GraphKey, GraphEntry, GraphKeyHash> entries_;
};

}  // namespace layerstorm::compute

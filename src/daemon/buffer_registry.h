#pragma once

// Buffer registry: maps integer buffer IDs to GPU device pointers.
//
// Populated at engine init from VramAllocator regions, DcpExecutor
// intermediates, and workspace buffers.  CommandDispatcher resolves
// buf_ids from command payloads before dispatching to kernel launchers.
//
// Thread safety: NOT thread-safe.  Called exclusively from daemon thread
// (INV-3.4.2) or during single-threaded init.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace layerstorm::daemon {

/// Sentinel: no buffer.
static constexpr uint32_t kInvalidBufId = 0;

/// Metadata for a registered GPU buffer.
struct BufferEntry {
    void*   device_ptr = nullptr;
    int64_t size_bytes = 0;
    int     gpu_idx    = -1;
};

/// Maps uint32_t buf_id → (device_ptr, size_bytes, gpu_idx).
///
/// IDs are monotonically assigned starting at 1 (0 = kInvalidBufId).
/// Deregistered IDs are never reused within the same registry lifetime.
class BufferRegistry {
public:
    BufferRegistry() = default;
    ~BufferRegistry() = default;

    BufferRegistry(const BufferRegistry&) = delete;
    BufferRegistry& operator=(const BufferRegistry&) = delete;
    BufferRegistry(BufferRegistry&&) = delete;
    BufferRegistry& operator=(BufferRegistry&&) = delete;

    // ── Registration ──────────────────────────────────────────────

    /// Register a buffer and return its assigned buf_id.
    /// @param name  Optional debug name (for logging/queries).
    uint32_t register_buffer(void* device_ptr, int64_t size_bytes,
                             int gpu_idx, const char* name = nullptr);

    /// Remove a buffer by ID.  No-op if ID not found.
    void deregister(uint32_t buf_id);

    // ── Lookup ────────────────────────────────────────────────────

    /// Full entry lookup.  Returns nullptr if not found.
    const BufferEntry* lookup(uint32_t buf_id) const;

    /// Shorthand: returns device_ptr, or nullptr if not found.
    void* resolve(uint32_t buf_id) const;

    /// Returns device_ptr if found AND size_bytes >= required_bytes.
    /// Otherwise returns nullptr.
    void* resolve_checked(uint32_t buf_id, int64_t required_bytes) const;

    // ── Queries ───────────────────────────────────────────────────

    size_t size() const { return entries_.size(); }
    bool empty() const { return entries_.empty(); }
    bool contains(uint32_t buf_id) const { return entries_.count(buf_id) != 0; }

    /// Return all (buf_id, name) pairs.  For Python query_buffer_ids().
    std::vector<std::pair<uint32_t, std::string>> all_named_entries() const;

    /// Clear all entries.  For testing only.
    void clear();

private:
    uint32_t next_id_ = 1;  // 0 = kInvalidBufId
    std::unordered_map<uint32_t, BufferEntry> entries_;
    std::unordered_map<uint32_t, std::string> names_;
};

}  // namespace layerstorm::daemon

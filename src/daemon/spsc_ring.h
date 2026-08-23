#pragma once

// Lock-free single-producer single-consumer ring buffer.
//
// Operates on raw memory: a RingHeader followed by slot_count * SlotSize bytes.
// The ring is heap-allocated (not POSIX shm) — both threads share the same
// address space.
//
// Memory ordering: std::atomic_thread_fence with release/acquire semantics.
// On x86-64, aligned 64-bit loads/stores are naturally atomic; fences compile
// to compiler barriers (no hardware barrier needed under TSO).

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>

#include "daemon/ipc_protocol.h"

namespace layerstorm::ipc {

template <uint32_t SlotSize>
class SpscRing {
public:
    /// Attach to an already-initialized ring region.
    explicit SpscRing(void* region)
        : header_(static_cast<RingHeader*>(region))
        , slots_(static_cast<uint8_t*>(region) + sizeof(RingHeader)) {}

    /// Initialize a ring region. Must be called once before use.
    static void init(void* region, uint32_t slot_count) {
        auto* h = static_cast<RingHeader*>(region);
        h->producer_seq = 0;
        h->consumer_seq = 0;
        h->slot_count   = slot_count;
        h->slot_size    = SlotSize;
        std::memset(static_cast<uint8_t*>(region) + sizeof(RingHeader),
                    0, static_cast<size_t>(slot_count) * SlotSize);
    }

    // ── Producer API ────────────────────────────────────────────────────────

    /// Try to write one slot. Returns false if ring is full.
    bool try_write(const void* data) {
        uint64_t prod = header_->producer_seq;
        uint64_t cons = load_consumer_seq();
        if (prod - cons >= header_->slot_count) return false;

        uint32_t idx = static_cast<uint32_t>(prod & (header_->slot_count - 1));
        std::memcpy(slots_ + static_cast<size_t>(idx) * SlotSize, data, SlotSize);

        std::atomic_thread_fence(std::memory_order_release);
        header_->producer_seq = prod + 1;
        return true;
    }

    // ── Consumer API ────────────────────────────────────────────────────────

    /// Try to read one slot. Returns false if ring is empty.
    bool try_read(void* data) {
        uint64_t cons = header_->consumer_seq;
        uint64_t prod = load_producer_seq();
        std::atomic_thread_fence(std::memory_order_acquire);
        if (cons >= prod) return false;

        uint32_t idx = static_cast<uint32_t>(cons & (header_->slot_count - 1));
        std::memcpy(data, slots_ + static_cast<size_t>(idx) * SlotSize, SlotSize);

        std::atomic_thread_fence(std::memory_order_release);
        header_->consumer_seq = cons + 1;
        return true;
    }

    /// Batch-read up to max_count slots. Callback receives const uint8_t* per slot.
    template <typename Callback>
    uint32_t drain(Callback&& cb, uint32_t max_count = UINT32_MAX) {
        uint64_t cons = header_->consumer_seq;
        uint64_t prod = load_producer_seq();
        std::atomic_thread_fence(std::memory_order_acquire);

        uint32_t avail = static_cast<uint32_t>(
            std::min(static_cast<uint64_t>(max_count), prod - cons));

        for (uint32_t i = 0; i < avail; ++i) {
            uint32_t idx = static_cast<uint32_t>(
                (cons + i) & (header_->slot_count - 1));
            cb(slots_ + static_cast<size_t>(idx) * SlotSize);
        }

        if (avail > 0) {
            std::atomic_thread_fence(std::memory_order_release);
            header_->consumer_seq = cons + avail;
        }
        return avail;
    }

    // ── Queries ─────────────────────────────────────────────────────────────

    bool is_empty() const {
        return load_producer_seq() == header_->consumer_seq;
    }

    bool is_full() const {
        return header_->producer_seq - load_consumer_seq() >= header_->slot_count;
    }

    uint32_t available() const {
        uint64_t prod = load_producer_seq();
        std::atomic_thread_fence(std::memory_order_acquire);
        return static_cast<uint32_t>(prod - header_->consumer_seq);
    }

    uint32_t slot_count() const { return header_->slot_count; }

private:
    RingHeader* header_;
    uint8_t*    slots_;

    uint64_t load_producer_seq() const {
        return *reinterpret_cast<const volatile uint64_t*>(&header_->producer_seq);
    }

    uint64_t load_consumer_seq() const {
        return *reinterpret_cast<const volatile uint64_t*>(&header_->consumer_seq);
    }
};

// ── Type aliases ────────────────────────────────────────────────────────────

using CommandRing    = SpscRing<kCmdSlotBytes>;
using CompletionRing = SpscRing<kCmpSlotBytes>;

}  // namespace layerstorm::ipc

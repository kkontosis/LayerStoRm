// Stackable write transaction for StateSnapshot seqlock.
//
// Each logical update group (GPU snapshots, residency, expert stats, etc.)
// opens and closes a transaction.  Only the outermost begin/end pair touches
// the seqlock — inner (nested) transactions are no-ops.  This supports
// separation of concerns: a high-level update can wrap a transaction around
// several lower-level updates that each open their own transactions.
//
// Yield policy: every kYieldInterval completed outermost transactions, the
// writer yields briefly so Python readers can observe an even seqlock.
//
// Single-writer invariant: only the daemon thread may call begin()/end().
// See spec/IPC_TX.md for the full design.

#pragma once

#include "daemon/ipc_protocol.h"

#include <sched.h>

namespace layerstorm::daemon {

class StateTransaction {
public:
    /// @param snap  StateSnapshot to write under seqlock.
    /// @param yield_interval  Yield every N completed outermost transactions
    ///                        (from config orchestrator.ipc_transaction.writer_yield_interval).
    explicit StateTransaction(ipc::StateSnapshot& snap, int yield_interval = 16)
        : snap_(snap), yield_interval_(yield_interval) {}

    /// Increment nesting counter.  Acquires seqlock on 0→1 transition.
    void begin() {
        if (counter_++ == 0) {
            ipc::seqlock_write_begin(snap_);
        }
    }

    /// Decrement nesting counter.  Releases seqlock on 1→0 transition.
    /// Yields every yield_interval_ completed outermost transactions.
    void end() {
        if (--counter_ == 0) {
            ipc::seqlock_write_end(snap_);
            if (++completions_ % yield_interval_ == 0) {
                sched_yield();  // ~500ns — no kernel timer overhead
            }
        }
    }

    /// Current nesting depth (0 = no active transaction).
    int depth() const { return counter_; }

    /// Total completed outermost transactions since construction.
    int completions() const { return completions_; }

private:
    ipc::StateSnapshot& snap_;
    int counter_        = 0;
    int completions_    = 0;
    int yield_interval_ = 16;
};

/// RAII guard: calls begin() on construction, end() on destruction.
class StateTransactionGuard {
public:
    explicit StateTransactionGuard(StateTransaction& tx) : tx_(tx) {
        tx_.begin();
    }
    ~StateTransactionGuard() {
        tx_.end();
    }

    StateTransactionGuard(const StateTransactionGuard&) = delete;
    StateTransactionGuard& operator=(const StateTransactionGuard&) = delete;

private:
    StateTransaction& tx_;
};

}  // namespace layerstorm::daemon

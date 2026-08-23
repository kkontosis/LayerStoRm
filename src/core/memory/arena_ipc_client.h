#pragma once

// ArenaIpcClient (P-24b) — engine-side attachment to the persistent-arena
// holder (`layerstorm_arena_holder`, INV-ARENA-HOLD). Connects over a
// SOCK_SEQPACKET unix socket, optionally auto-spawning the holder when absent,
// and exchanges memfd segment fds via SCM_RIGHTS (arena_ipc_protocol.h).
//
// The ATTACHMENT is the open socket: it is held for the client's lifetime, so
// closing (dtor / detach()) — or the engine dying — releases the holder back
// to accepting attaches while it keeps the stored fds (and thus the tmpfs
// pages) alive. CUDA-free (covered by layerstorm_no_cuda_check, INV-GPU-1).
//
// Thread safety: none — init/shutdown path only, single-threaded.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/memory/arena_ipc_protocol.h"  // WipeReason (libc-only header)

namespace layerstorm::memory {

/// One received/stored node arena segment. The client owns received fds until
/// they are handed off (e.g. into a NumaBuffer via NumaManager::adopt_shared).
struct ArenaIpcSegment {
    int    numa_node = -1;
    size_t size_bytes = 0;
    int    fd = -1;
};

class ArenaIpcClient {
public:
    enum class AttachOutcome {
        kEmpty,   ///< attached; holder has no stored arena (cold boot)
        kWarm,    ///< attached; segments() + meta_fd() carry the stored arena
        kBusy,    ///< another instance is attached — NOT attached
        kError,   ///< connect/spawn/protocol failure — NOT attached
    };

    ArenaIpcClient() = default;
    /// Closes the socket (= detach) and any segment fds still owned.
    ~ArenaIpcClient();

    ArenaIpcClient(const ArenaIpcClient&) = delete;
    ArenaIpcClient& operator=(const ArenaIpcClient&) = delete;

    /// Connect to `socket_path` ('@' prefix = abstract namespace) and attach.
    /// When the connect fails and `auto_spawn` is set, spawns the holder
    /// (`holder_binary`, or a built-in candidate search when empty) in --daemon
    /// mode and retries for ~5 s. On kWarm the stored segments/meta fds are
    /// received and owned by this client until released.
    AttachOutcome attach(const std::string& socket_path,
                         const std::string& holder_binary,
                         bool auto_spawn);

    /// Store a (new) arena generation with the holder: `meta_fd` + `segments`
    /// (fds are dup'd by the kernel on send — the caller KEEPS its fds). The
    /// holder closes any previously stored generation. Requires an attached
    /// client. Returns false on send/reply failure.
    bool store(int meta_fd, const std::vector<ArenaIpcSegment>& segments);

    /// Tell the holder to drop its stored generation NOW (fds closed → the old
    /// arena's tmpfs pages are freed). Sent on geometry/source mismatch BEFORE
    /// building the replacement arena, so old + new never coexist in RAM (a
    /// full-fit shape change would otherwise OOM the strict mbind prefault).
    /// Requires an attached client; returns false on send/reply failure.
    ///
    /// `reason` is MANDATORY and part of the wire contract: the holder honors
    /// kIdentityMismatch / kOperatorRequest and REFUSES kUnspecified (which is
    /// also what a pre-reason client's payload-less kWipe looks like). `detail`
    /// is free-form diagnostics for the holder log (e.g. the named failing
    /// identity check); it is truncated to arena_ipc::kMaxWipeDetail bytes.
    bool wipe(arena_ipc::WipeReason reason, const std::string& detail = {});

    /// Close the attachment (idempotent). The holder keeps the stored fds.
    void detach();

    bool attached() const { return sock_ >= 0; }

    /// Warm-attach results (valid after attach() returned kWarm). Ownership of
    /// the fds stays here until released via take_segments()/take_meta_fd().
    const std::vector<ArenaIpcSegment>& segments() const { return segments_; }
    int meta_fd() const { return meta_fd_; }

    /// Transfer ownership of the received fds to the caller.
    std::vector<ArenaIpcSegment> take_segments();
    int take_meta_fd();

private:
    void close_owned_fds();

    int sock_ = -1;
    int meta_fd_ = -1;
    std::vector<ArenaIpcSegment> segments_;
};

}  // namespace layerstorm::memory

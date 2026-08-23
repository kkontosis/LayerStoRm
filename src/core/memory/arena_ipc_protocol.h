#pragma once

// Arena-holder IPC wire protocol (P-24b) — shared by the CUDA-free
// `layerstorm_arena_holder` binary (tools/arena_holder.cpp), the engine-side
// ArenaIpcClient, and the tests. Transport: SOCK_SEQPACKET AF_UNIX socket
// (message boundaries preserved); memfd file descriptors move via SCM_RIGHTS
// ancillary data alongside each message.
//
// INV-ARENA-HOLD: the holder is a dumb fd-keeper. It never maps, reads, or
// interprets the segments it keeps alive — the geometry payload is an OPAQUE
// blob it stores and echoes back. All NUMA/slot logic lives in the engine.
// The ATTACHMENT is the socket connection itself: one connection may be
// attached at a time (later connects get kBusy); EOF — clean close OR the
// attached process being killed — detaches automatically.
//
// Flow:
//   client → kAttach            → AttachReply{kEmpty | kWarm(+payload+fds) | kBusy}
//   client → kStore(+payload+fds) → StoreReply   (replaces any stored set; the
//                                                 holder closes the old fds)
//   client → kWipe(+WipeRequest) → StoreReply    (drops the stored set NOW —
//                                                 sent on geometry/source
//                                                 mismatch BEFORE the engine
//                                                 builds the replacement, so
//                                                 old+new arenas never coexist
//                                                 in RAM)
//   client → kDetach or close   → holder returns to accepting attaches
//
// WIPE REASONS (INV-ARENA-HOLD refinement): a kWipe must state WHY. The holder
// HONORS reasoned wipes — kIdentityMismatch (the attaching engine found the
// store incompatible with its config: the incompatible store "kills itself"
// through the discovering client while the holder process stays up to host the
// successor store) and kOperatorRequest (a human/CLI asked for it) — and
// REFUSES kUnspecified with kError. A payload-less or short kWipe reads as
// kUnspecified, so legacy/rogue clients that predate this field can never drop
// a store. Disaster-class protection lives ENGINE-side
// (memory.arena_attach.persist=true ⇒ that engine never sends kWipe at all and
// fails loud on mismatch), not as a holder-side veto.
//
// fd order in kStore / warm AttachReply ancillary data:
//   fds[0]           = ArenaCache meta segment
//   fds[1..num_segs] = node arena segments, in SegmentDesc payload order.
//
// This header is CUDA-free and libc-only (INV-GPU-1-compatible; the holder
// links nothing but libc).

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace layerstorm::memory::arena_ipc {

inline constexpr uint32_t kMagic = 0x4C534148;  // "LSAH"
inline constexpr uint32_t kProtocolVersion = 1;

/// Max node arena segments per store (dev boxes have ≤ ~16 NUMA banks; SCM
/// fits 253 fds/message — 64 is a generous ceiling).
inline constexpr uint32_t kMaxSegments = 64;

enum class MsgType : uint32_t { kAttach = 1, kStore = 2, kDetach = 3,
                                kWipe = 4 };
enum class AttachStatus : uint32_t { kEmpty = 0, kWarm = 1, kBusy = 2 };
enum class StoreStatus : uint32_t { kOk = 0, kError = 1 };

/// Why a client wants the stored generation dropped. 0 (= a payload-less kWipe
/// from a pre-reason client) is REFUSED by the holder; the two stated reasons
/// are honored. Never renumber — this is wire state shared with older binaries.
enum class WipeReason : uint32_t {
    kUnspecified = 0,      ///< legacy/rogue: holder replies kError, store kept
    kIdentityMismatch = 1, ///< attaching engine: store incompatible with its config
    kOperatorRequest = 2,  ///< explicit human/CLI request (layerstorm_arena_ctl)
};

/// Max bytes of the human-readable wipe detail (NUL-padded, not required to be
/// NUL-terminated by a sender — readers must bound their scan by the array).
inline constexpr size_t kMaxWipeDetail = 256;

/// kWipe payload. Fixed size: `payload_bytes` must equal sizeof(WipeRequest) or
/// the holder treats the message as kUnspecified (and refuses it).
struct WipeRequest {
    uint32_t reason = 0;                 ///< WipeReason
    char detail[kMaxWipeDetail] = {};    ///< e.g. the named failing identity check
};

/// Human-readable reason name for logs (never null).
inline const char* wipe_reason_name(uint32_t reason) {
    switch (static_cast<WipeReason>(reason)) {
        case WipeReason::kIdentityMismatch: return "identity_mismatch";
        case WipeReason::kOperatorRequest:  return "operator_request";
        case WipeReason::kUnspecified:      return "unspecified";
    }
    return "unknown";
}

/// True for the reasons a holder honors.
inline bool wipe_reason_honored(uint32_t reason) {
    return static_cast<WipeReason>(reason) == WipeReason::kIdentityMismatch ||
           static_cast<WipeReason>(reason) == WipeReason::kOperatorRequest;
}

/// Client → holder message header; `payload_bytes` of opaque payload follow
/// in the SAME datagram (SEQPACKET keeps them one message).
struct MsgHeader {
    uint32_t magic = kMagic;
    uint32_t version = kProtocolVersion;
    uint32_t type = 0;           ///< MsgType
    uint32_t payload_bytes = 0;  ///< bytes after this header in the message
};

/// Holder → client reply to kAttach. On kWarm the stored payload follows and
/// `num_fds` fds ride the ancillary data ([meta, segments...]).
struct AttachReply {
    uint32_t magic = kMagic;
    uint32_t status = 0;         ///< AttachStatus
    uint32_t num_fds = 0;
    uint32_t payload_bytes = 0;
};

/// Holder → client reply to kStore.
struct StoreReply {
    uint32_t magic = kMagic;
    uint32_t status = 0;         ///< StoreStatus
};

/// One node arena segment (the kStore payload is an array of these; the holder
/// stores it opaquely). Segment byte size is authoritative here AND recoverable
/// via fstat(fd) — the client validates both agree on warm attach.
struct SegmentDesc {
    int32_t  numa_node = -1;
    uint32_t reserved = 0;
    uint64_t size_bytes = 0;
};

// ── Socket-address helper ────────────────────────────────────────────────────

/// Fill a sockaddr_un from a path. A leading '@' selects the Linux abstract
/// namespace (no filesystem entry; name length limit applies). Returns the
/// sockaddr length to pass to bind/connect, or 0 if the path is too long.
inline socklen_t make_sockaddr(const char* path, sockaddr_un* addr) {
    std::memset(addr, 0, sizeof(*addr));
    addr->sun_family = AF_UNIX;
    const size_t len = std::strlen(path);
    if (len == 0 || len >= sizeof(addr->sun_path)) return 0;
    if (path[0] == '@') {
        addr->sun_path[0] = '\0';
        std::memcpy(addr->sun_path + 1, path + 1, len - 1);
        return static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + len);
    }
    std::memcpy(addr->sun_path, path, len);
    return static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + len + 1);
}

// ── SCM_RIGHTS send/recv helpers ─────────────────────────────────────────────

/// Send one message (`buf`,`len`) with `nfds` fds in SCM_RIGHTS ancillary data.
/// Returns bytes sent, or -1 (errno set). Retries EINTR.
inline ssize_t send_with_fds(int sock, const void* buf, size_t len,
                             const int* fds, size_t nfds) {
    iovec iov{const_cast<void*>(buf), len};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * (kMaxSegments + 1))];
    if (nfds > 0) {
        if (nfds > kMaxSegments + 1) { errno = EINVAL; return -1; }
        std::memset(control, 0, sizeof(control));
        msg.msg_control = control;
        msg.msg_controllen = CMSG_SPACE(sizeof(int) * nfds);
        cmsghdr* cm = CMSG_FIRSTHDR(&msg);
        cm->cmsg_level = SOL_SOCKET;
        cm->cmsg_type = SCM_RIGHTS;
        cm->cmsg_len = CMSG_LEN(sizeof(int) * nfds);
        std::memcpy(CMSG_DATA(cm), fds, sizeof(int) * nfds);
    }
    ssize_t n;
    do { n = ::sendmsg(sock, &msg, MSG_NOSIGNAL); } while (n < 0 && errno == EINTR);
    return n;
}

/// Receive one message into (`buf`,`len`), collecting up to `max_fds` fds from
/// SCM_RIGHTS into `fds`; `*got_fds` is set to the count received. Returns
/// bytes received (0 = EOF), or -1 (errno set). Retries EINTR. Any fds beyond
/// max_fds are closed (never leaked).
inline ssize_t recv_with_fds(int sock, void* buf, size_t len,
                             int* fds, size_t max_fds, size_t* got_fds) {
    if (got_fds) *got_fds = 0;
    iovec iov{buf, len};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * (kMaxSegments + 1))];
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    ssize_t n;
    do { n = ::recvmsg(sock, &msg, MSG_CMSG_CLOEXEC); } while (n < 0 && errno == EINTR);
    if (n <= 0) return n;
    size_t count = 0;
    for (cmsghdr* cm = CMSG_FIRSTHDR(&msg); cm; cm = CMSG_NXTHDR(&msg, cm)) {
        if (cm->cmsg_level != SOL_SOCKET || cm->cmsg_type != SCM_RIGHTS) continue;
        const size_t nfd = (cm->cmsg_len - CMSG_LEN(0)) / sizeof(int);
        const int* p = reinterpret_cast<const int*>(CMSG_DATA(cm));
        for (size_t i = 0; i < nfd; ++i) {
            if (fds && count < max_fds) fds[count++] = p[i];
            else ::close(p[i]);  // over max: close, don't leak
        }
    }
    if (got_fds) *got_fds = count;
    return n;
}

}  // namespace layerstorm::memory::arena_ipc

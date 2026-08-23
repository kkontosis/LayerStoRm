// layerstorm_arena_holder — persistent-arena fd keeper (P-24b, INV-ARENA-HOLD).
//
// A tiny CUDA-free daemon that keeps the engine's pinned-expert-arena RAM alive
// across engine runs. The engine creates the arena as memfd segments
// (MAP_SHARED tmpfs pages, NUMA-placed by the engine's own prefault) and hands
// the fds here via kStore; holding the fds keeps the physical pages — and the
// preloaded expert bytes in them — alive after the engine exits. On the next
// run the engine re-attaches, receives the same fds, maps + cudaHostRegisters
// them, and skips the NVMe preload for still-valid content (ArenaCache).
//
// INV-ARENA-HOLD: this process NEVER maps, reads, or interprets the segments —
// it stores opaque fds + an opaque geometry blob and echoes them back. Exactly
// one connection may be attached at a time (the attachment IS the connection;
// EOF = auto-detach, later connects while attached get kBusy). Links libc only.
//
// Usage: layerstorm_arena_holder [--socket <path|@abstract>] [--daemon]
//   default socket: @layerstorm-arena
//
// WIPE POLICY (2026-08-22): the holder HONORS every REASONED kWipe
// (kIdentityMismatch — the attaching engine found the store incompatible with
// its config, so the incompatible store kills itself through the discovering
// client; kOperatorRequest — a human asked via layerstorm_arena_ctl) and
// REFUSES kUnspecified (a payload-less kWipe from a pre-reason client) with
// kError. The holder process itself SURVIVES a wipe and hosts the successor
// store, which is how a box switches the model it serves without a holder
// restart. The old `--persist` holder-side veto is RETIRED: disaster-class
// protection is engine-side (memory.arena_attach.persist=true ⇒ that engine
// never sends kWipe and fails loud on mismatch). ACKNOWLEDGED TRADE: a
// reasoned identity-mismatch wipe is now honored even against a precious
// store, so a production deployment MUST set persist=true.
//
// Build: cmake --build build --target layerstorm_arena_holder

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/memory/arena_ipc_protocol.h"

namespace ipc = layerstorm::memory::arena_ipc;

namespace {

struct StoredSet {
    std::vector<int> fds;              // [meta, segments...]
    std::vector<char> payload;         // opaque SegmentDesc blob
    void clear() {
        for (int fd : fds) ::close(fd);
        fds.clear();
        payload.clear();
    }
    bool empty() const { return fds.empty(); }
};

void logf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[arena_holder] ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::fflush(stderr);
}

/// Reply kBusy on an accepted connection. The caller closes it AFTER this —
/// crucially we only refuse in response to a READ kAttach message, never at
/// accept time: closing a conn with an unread inbound message races the
/// client's recv into an EOF instead of the busy reply.
void reply_busy(int fd) {
    ipc::AttachReply r;
    r.status = static_cast<uint32_t>(ipc::AttachStatus::kBusy);
    ipc::send_with_fds(fd, &r, sizeof(r), nullptr, 0);
}

/// Double-fork detach for --daemon mode. Parent exits only after the child has
/// bound the socket (signalled over a pipe), so a spawning engine can connect
/// immediately after the parent returns.
int daemonize_prepare(int ready_pipe[2]) {
    if (::pipe(ready_pipe) != 0) return -1;
    pid_t pid = ::fork();
    if (pid < 0) return -1;
    if (pid > 0) {
        // Parent: wait for the grandchild's ready byte, then exit.
        ::close(ready_pipe[1]);
        char b = 0;
        ssize_t n;
        do { n = ::read(ready_pipe[0], &b, 1); } while (n < 0 && errno == EINTR);
        ::close(ready_pipe[0]);
        std::exit(n == 1 && b == 'R' ? 0 : 1);
    }
    ::setsid();
    pid = ::fork();
    if (pid < 0) std::exit(1);
    if (pid > 0) std::exit(0);
    ::close(ready_pipe[0]);
    // Grandchild continues; caller signals readiness after bind+listen.
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // Hygiene: drop every inherited fd ≥ 3 FIRST. The engine posix_spawns us
    // mid-init with its non-CLOEXEC fds attached (e.g. ~256 open prepacked
    // expert-file fds in direct mode) — a keeper must hold ONLY its own store.
    if (::close_range(3, ~0U, 0) != 0)
        for (int fd = 3; fd < 1024; ++fd) ::close(fd);

    const char* socket_path = "@layerstorm-arena";
    bool daemon_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (std::strcmp(argv[i], "--daemon") == 0) {
            daemon_mode = true;
        } else {
            if (std::strcmp(argv[i], "--persist") == 0)
                std::fprintf(stderr,
                             "[arena_holder] --persist was RETIRED "
                             "(2026-08-22): reasoned kWipe messages are always "
                             "honored; store protection is engine-side "
                             "(memory.arena_attach.persist=true).\n");
            std::fprintf(stderr,
                         "usage: %s [--socket <path|@abstract>] [--daemon]\n",
                         argv[0]);
            return 2;
        }
    }

    ::signal(SIGPIPE, SIG_IGN);

    int ready_pipe[2] = {-1, -1};
    if (daemon_mode && daemonize_prepare(ready_pipe) != 0) {
        logf("daemonize failed: %s", std::strerror(errno));
        return 1;
    }

    sockaddr_un addr;
    const socklen_t alen = ipc::make_sockaddr(socket_path, &addr);
    if (alen == 0) {
        logf("socket path too long: %s", socket_path);
        return 1;
    }
    const int srv = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (srv < 0) {
        logf("socket(): %s", std::strerror(errno));
        return 1;
    }
    // Filesystem sockets: remove a stale path from a previous (dead) holder.
    // A LIVE holder still bound there would lose its name — probe by connect.
    if (socket_path[0] != '@') {
        const int probe = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
        if (probe >= 0) {
            if (::connect(probe, reinterpret_cast<sockaddr*>(&addr), alen) == 0) {
                ::close(probe);
                ::close(srv);
                logf("another holder is already serving %s", socket_path);
                return 1;
            }
            ::close(probe);
        }
        ::unlink(socket_path);
    }
    if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), alen) != 0) {
        // Abstract namespace: EADDRINUSE = a live holder already owns the name.
        logf("bind(%s): %s", socket_path, std::strerror(errno));
        ::close(srv);
        return 1;
    }
    if (::listen(srv, 8) != 0) {
        logf("listen(): %s", std::strerror(errno));
        ::close(srv);
        return 1;
    }
    logf("ready: pid=%d socket=%s", ::getpid(), socket_path);
    if (daemon_mode) {
        char b = 'R';
        [[maybe_unused]] ssize_t w = ::write(ready_pipe[1], &b, 1);
        ::close(ready_pipe[1]);
        // Detach stdio (log target gone once the launching terminal closes).
        const int devnull = ::open("/dev/null", O_RDWR);
        if (devnull >= 0) {
            ::dup2(devnull, STDIN_FILENO);
            ::dup2(devnull, STDOUT_FILENO);
            ::dup2(devnull, STDERR_FILENO);
            if (devnull > STDERR_FILENO) ::close(devnull);
        }
    }

    StoredSet stored;
    int attached = -1;                // fd of the attached connection (-1 = none)
    std::vector<int> conns;           // all live connections (incl. attached)
    constexpr size_t kMaxConns = 16;  // accept cap (excess closed unserved)

    // Message buffer: header + the largest payload (SegmentDesc array / wipe).
    constexpr size_t kBufBytes =
        sizeof(ipc::MsgHeader) + ipc::kMaxSegments * sizeof(ipc::SegmentDesc);
    static_assert(kBufBytes >= sizeof(ipc::MsgHeader) + sizeof(ipc::WipeRequest),
                  "message buffer must hold a full WipeRequest payload");
    std::vector<char> buf(kBufBytes);
    int fds[ipc::kMaxSegments + 1];

    auto handle_msg = [&](int conn) -> bool {  // false = close this conn
        size_t got_fds = 0;
        const ssize_t n = ipc::recv_with_fds(conn, buf.data(), buf.size(),
                                             fds, ipc::kMaxSegments + 1, &got_fds);
        if (n <= 0) return false;  // EOF or error → detach/close
        auto close_received = [&] { for (size_t i = 0; i < got_fds; ++i) ::close(fds[i]); };
        if (static_cast<size_t>(n) < sizeof(ipc::MsgHeader)) { close_received(); return false; }
        ipc::MsgHeader hdr;
        std::memcpy(&hdr, buf.data(), sizeof(hdr));
        if (hdr.magic != ipc::kMagic || hdr.version != ipc::kProtocolVersion) {
            logf("bad message (magic/version) — closing connection");
            close_received();
            return false;
        }
        switch (static_cast<ipc::MsgType>(hdr.type)) {
            case ipc::MsgType::kAttach: {
                close_received();  // attach carries no fds
                if (attached >= 0 && attached != conn) {
                    // Single-attachment: reply busy to the READ attach message;
                    // the caller closes this conn on `false`.
                    reply_busy(conn);
                    logf("refused attach (busy)");
                    return false;
                }
                attached = conn;
                ipc::AttachReply r;
                if (stored.empty()) {
                    r.status = static_cast<uint32_t>(ipc::AttachStatus::kEmpty);
                    ipc::send_with_fds(conn, &r, sizeof(r), nullptr, 0);
                    logf("attached (empty store)");
                } else {
                    r.status = static_cast<uint32_t>(ipc::AttachStatus::kWarm);
                    r.num_fds = static_cast<uint32_t>(stored.fds.size());
                    r.payload_bytes = static_cast<uint32_t>(stored.payload.size());
                    std::vector<char> out(sizeof(r) + stored.payload.size());
                    std::memcpy(out.data(), &r, sizeof(r));
                    std::memcpy(out.data() + sizeof(r), stored.payload.data(),
                                stored.payload.size());
                    ipc::send_with_fds(conn, out.data(), out.size(),
                                       stored.fds.data(), stored.fds.size());
                    logf("attached (warm: %zu fds, %zu payload bytes)",
                         stored.fds.size(), stored.payload.size());
                }
                return true;
            }
            case ipc::MsgType::kStore: {
                ipc::StoreReply r;
                const size_t payload = hdr.payload_bytes;
                const bool sane =
                    conn == attached && got_fds >= 1 &&
                    got_fds <= ipc::kMaxSegments + 1 &&
                    payload == (got_fds - 1) * sizeof(ipc::SegmentDesc) &&
                    sizeof(ipc::MsgHeader) + payload == static_cast<size_t>(n);
                if (!sane) {
                    close_received();
                    r.status = static_cast<uint32_t>(ipc::StoreStatus::kError);
                    ipc::send_with_fds(conn, &r, sizeof(r), nullptr, 0);
                    logf("rejected store (%zu fds, %zu payload, attached=%d)",
                         got_fds, payload, conn == attached);
                    return conn == attached;
                }
                stored.clear();  // replace: close previous generation's fds
                stored.fds.assign(fds, fds + got_fds);
                stored.payload.assign(buf.data() + sizeof(ipc::MsgHeader),
                                      buf.data() + sizeof(ipc::MsgHeader) + payload);
                r.status = static_cast<uint32_t>(ipc::StoreStatus::kOk);
                ipc::send_with_fds(conn, &r, sizeof(r), nullptr, 0);
                logf("stored %zu fds (%zu segment(s))", got_fds, got_fds - 1);
                return true;
            }
            case ipc::MsgType::kWipe: {
                close_received();  // wipe carries no fds
                ipc::StoreReply r;
                if (conn != attached) {
                    r.status = static_cast<uint32_t>(ipc::StoreStatus::kError);
                    ipc::send_with_fds(conn, &r, sizeof(r), nullptr, 0);
                    logf("refused wipe (not the attached connection): store "
                         "retained (%zu fds)", stored.fds.size());
                    return true;
                }
                // Reason: a payload-less / short / oversized kWipe (any
                // pre-reason client) reads as kUnspecified and is REFUSED.
                ipc::WipeRequest req;
                if (hdr.payload_bytes == sizeof(ipc::WipeRequest) &&
                    sizeof(ipc::MsgHeader) + hdr.payload_bytes ==
                        static_cast<size_t>(n))
                    std::memcpy(&req, buf.data() + sizeof(ipc::MsgHeader),
                                sizeof(req));
                char detail[ipc::kMaxWipeDetail + 1];
                std::memcpy(detail, req.detail, ipc::kMaxWipeDetail);
                detail[ipc::kMaxWipeDetail] = '\0';
                if (!ipc::wipe_reason_honored(req.reason)) {
                    r.status = static_cast<uint32_t>(ipc::StoreStatus::kError);
                    ipc::send_with_fds(conn, &r, sizeof(r), nullptr, 0);
                    logf("REFUSED wipe (reason=%s): a wipe must state an "
                         "honored reason — store retained (%zu fds)",
                         ipc::wipe_reason_name(req.reason), stored.fds.size());
                    return true;
                }
                const size_t had = stored.fds.size();
                stored.clear();  // free the old generation's pages NOW
                r.status = static_cast<uint32_t>(ipc::StoreStatus::kOk);
                ipc::send_with_fds(conn, &r, sizeof(r), nullptr, 0);
                logf("wiped stored generation (%zu fds) — reason=%s%s%s; "
                     "holder stays up for the successor store",
                     had, ipc::wipe_reason_name(req.reason),
                     detail[0] ? ": " : "", detail);
                return true;
            }
            case ipc::MsgType::kDetach:
                close_received();
                return false;
            default:
                close_received();
                logf("unknown message type %u — closing connection", hdr.type);
                return false;
        }
    };

    for (;;) {
        std::vector<pollfd> pfd;
        pfd.push_back({srv, POLLIN, 0});
        for (int c : conns) pfd.push_back({c, POLLIN, 0});
        if (::poll(pfd.data(), pfd.size(), -1) < 0) {
            if (errno == EINTR) continue;
            logf("poll(): %s", std::strerror(errno));
            break;
        }
        if (pfd[0].revents & POLLIN) {
            const int conn = ::accept4(srv, nullptr, nullptr, SOCK_CLOEXEC);
            if (conn >= 0) {
                if (conns.size() >= kMaxConns) ::close(conn);
                else conns.push_back(conn);
            }
        }
        for (size_t i = 1; i < pfd.size(); ++i) {
            if (!(pfd[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            const int fd = pfd[i].fd;
            if (handle_msg(fd)) continue;
            ::close(fd);
            conns.erase(std::find(conns.begin(), conns.end(), fd));
            if (fd == attached) {
                attached = -1;
                logf("detached (fds retained: %zu)", stored.fds.size());
            }
        }
    }
    stored.clear();
    for (int c : conns) ::close(c);
    ::close(srv);
    return 0;
}

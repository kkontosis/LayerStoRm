// layerstorm_arena_ctl — operator CLI for the persistent-arena holder.
//
// The holder (`layerstorm_arena_holder`, INV-ARENA-HOLD) only ever acts on
// client messages: it has no self-service commands, so dropping a store that
// belongs to a model the box no longer serves needs a client to ask. This is
// that client — the explicit OPERATOR route (WipeReason::kOperatorRequest),
// distinct from the engine's kIdentityMismatch wipe.
//
// Typical use (switching the box to a different model's arena):
//   1. stop every engine attached to the holder (the attachment is the socket
//      connection — a live engine makes this tool report BUSY),
//   2. ./build/tools/layerstorm_arena_ctl --wipe --detail "switching to V4",
//   3. boot the new model's engine: it attaches to an EMPTY holder and cold-
//      builds its own store (allowed even under memory.arena_attach.persist).
//
// Default action is --status (attach, report, detach) — read-only.
//
// CUDA-free and libc-only, like the holder itself: it speaks
// arena_ipc_protocol.h directly rather than linking layerstorm_core
// (INV-GPU-1-compatible; nothing here maps or interprets a segment).
//
// Build: cmake --build build --target layerstorm_arena_ctl

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/memory/arena_ipc_protocol.h"

namespace ipc = layerstorm::memory::arena_ipc;

namespace {

int usage(const char* argv0, int rc) {
    std::fprintf(rc == 0 ? stdout : stderr,
                 "usage: %s [--socket <path|@abstract>] [--status | --wipe] "
                 "[--detail <text>]\n"
                 "  --status   attach, report the store state, detach "
                 "(default)\n"
                 "  --wipe     drop the stored generation NOW "
                 "(WipeReason::kOperatorRequest)\n"
                 "  --detail   free-form note recorded in the holder log\n"
                 "  default socket: @layerstorm-arena\n",
                 argv0);
    return rc;
}

/// Connect + kAttach. Returns the socket (attached) or -1; `status` receives
/// the AttachStatus on a well-formed reply. Any received fds are closed
/// immediately — this tool never maps a segment, it only reports counts.
int attach(const std::string& socket_path, uint32_t* status,
           uint32_t* num_fds) {
    sockaddr_un addr;
    const socklen_t alen = ipc::make_sockaddr(socket_path.c_str(), &addr);
    if (alen == 0) {
        std::fprintf(stderr, "socket path too long: %s\n", socket_path.c_str());
        return -1;
    }
    const int s = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (s < 0) {
        std::perror("socket");
        return -1;
    }
    if (::connect(s, reinterpret_cast<sockaddr*>(&addr), alen) != 0) {
        std::fprintf(stderr, "cannot reach a holder on %s: %s\n",
                     socket_path.c_str(), std::strerror(errno));
        ::close(s);
        return -1;
    }
    ipc::MsgHeader hdr;
    hdr.type = static_cast<uint32_t>(ipc::MsgType::kAttach);
    if (ipc::send_with_fds(s, &hdr, sizeof(hdr), nullptr, 0) < 0) {
        std::perror("sendmsg(kAttach)");
        ::close(s);
        return -1;
    }
    std::vector<char> buf(sizeof(ipc::AttachReply) +
                          ipc::kMaxSegments * sizeof(ipc::SegmentDesc));
    int fds[ipc::kMaxSegments + 1];
    size_t got_fds = 0;
    const ssize_t n = ipc::recv_with_fds(s, buf.data(), buf.size(), fds,
                                         ipc::kMaxSegments + 1, &got_fds);
    for (size_t i = 0; i < got_fds; ++i) ::close(fds[i]);  // never mapped here
    if (n < static_cast<ssize_t>(sizeof(ipc::AttachReply))) {
        std::fprintf(stderr, "short attach reply\n");
        ::close(s);
        return -1;
    }
    ipc::AttachReply reply;
    std::memcpy(&reply, buf.data(), sizeof(reply));
    if (reply.magic != ipc::kMagic) {
        std::fprintf(stderr, "bad attach reply magic\n");
        ::close(s);
        return -1;
    }
    *status = reply.status;
    *num_fds = reply.num_fds;
    if (reply.status == static_cast<uint32_t>(ipc::AttachStatus::kBusy)) {
        ::close(s);
        return -1;
    }
    return s;
}

void detach(int s) {
    ipc::MsgHeader hdr;
    hdr.type = static_cast<uint32_t>(ipc::MsgType::kDetach);
    ipc::send_with_fds(s, &hdr, sizeof(hdr), nullptr, 0);  // best-effort
    ::close(s);
}

}  // namespace

int main(int argc, char** argv) {
    std::string socket_path = "@layerstorm-arena";
    std::string detail;
    bool do_wipe = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (std::strcmp(argv[i], "--detail") == 0 && i + 1 < argc) {
            detail = argv[++i];
        } else if (std::strcmp(argv[i], "--wipe") == 0) {
            do_wipe = true;
        } else if (std::strcmp(argv[i], "--status") == 0) {
            do_wipe = false;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            return usage(argv[0], 0);
        } else {
            return usage(argv[0], 2);
        }
    }

    uint32_t status = 0, num_fds = 0;
    const int s = attach(socket_path, &status, &num_fds);
    if (s < 0) {
        if (status == static_cast<uint32_t>(ipc::AttachStatus::kBusy))
            std::fprintf(stderr,
                         "holder on %s is BUSY — an engine is attached. Stop "
                         "it first (the attachment IS its socket connection).\n",
                         socket_path.c_str());
        return 1;
    }
    const bool warm =
        status == static_cast<uint32_t>(ipc::AttachStatus::kWarm);
    const size_t segs = warm && num_fds >= 1 ? num_fds - 1 : 0;
    std::printf("holder %s: store=%s%s\n", socket_path.c_str(),
                warm ? "WARM" : "EMPTY",
                warm ? (" (" + std::to_string(segs) + " segment(s))").c_str()
                     : "");

    int rc = 0;
    if (do_wipe) {
        char msg[sizeof(ipc::MsgHeader) + sizeof(ipc::WipeRequest)];
        ipc::MsgHeader hdr;
        hdr.type = static_cast<uint32_t>(ipc::MsgType::kWipe);
        hdr.payload_bytes = static_cast<uint32_t>(sizeof(ipc::WipeRequest));
        ipc::WipeRequest req;
        req.reason = static_cast<uint32_t>(ipc::WipeReason::kOperatorRequest);
        std::memcpy(req.detail, detail.data(),
                    detail.size() < ipc::kMaxWipeDetail ? detail.size()
                                                        : ipc::kMaxWipeDetail);
        std::memcpy(msg, &hdr, sizeof(hdr));
        std::memcpy(msg + sizeof(hdr), &req, sizeof(req));
        ipc::StoreReply reply{};
        if (ipc::send_with_fds(s, msg, sizeof(msg), nullptr, 0) < 0) {
            std::perror("sendmsg(kWipe)");
            rc = 1;
        } else if (ipc::recv_with_fds(s, &reply, sizeof(reply), nullptr, 0,
                                      nullptr) !=
                       static_cast<ssize_t>(sizeof(reply)) ||
                   reply.magic != ipc::kMagic ||
                   reply.status !=
                       static_cast<uint32_t>(ipc::StoreStatus::kOk)) {
            std::fprintf(stderr,
                         "holder REFUSED the operator wipe — store retained "
                         "(an older holder binary predates the kWipe reason "
                         "contract; restart it from the current build)\n");
            rc = 1;
        } else {
            std::printf("wiped (reason=operator_request%s%s) — the holder "
                        "stays up for the successor store\n",
                        detail.empty() ? "" : ": ", detail.c_str());
        }
    }
    detach(s);
    return rc;
}

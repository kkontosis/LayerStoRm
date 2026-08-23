// ArenaIpcClient (P-24b) — engine-side attachment to layerstorm_arena_holder.

#include "core/memory/arena_ipc_client.h"

#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>

#include <spdlog/spdlog.h>

#include "core/memory/arena_ipc_protocol.h"

extern char** environ;

namespace layerstorm::memory {

namespace ipc = arena_ipc;

namespace {

int try_connect(const std::string& socket_path) {
    sockaddr_un addr;
    const socklen_t alen = ipc::make_sockaddr(socket_path.c_str(), &addr);
    if (alen == 0) {
        spdlog::error("ArenaIpcClient: socket path too long: {}", socket_path);
        return -1;
    }
    const int s = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (s < 0) return -1;
    if (::connect(s, reinterpret_cast<sockaddr*>(&addr), alen) != 0) {
        ::close(s);
        return -1;
    }
    return s;
}

bool executable_exists(const std::string& p) {
    return !p.empty() && ::access(p.c_str(), X_OK) == 0;
}

/// Spawn the holder detached (--daemon double-forks; the spawned parent exits
/// once the daemon has bound the socket, so we reap it here and can connect
/// immediately). Returns true if the spawn command itself succeeded.
bool spawn_holder(const std::string& binary, const std::string& socket_path) {
    std::vector<char*> argv;
    std::string b = binary, sflag = "--socket", spath = socket_path,
                dflag = "--daemon";
    argv = {b.data(), sflag.data(), spath.data(), dflag.data(), nullptr};
    pid_t pid = -1;
    // posix_spawnp: PATH lookup for bare names, direct exec for paths.
    const int rc = ::posix_spawnp(&pid, binary.c_str(), nullptr, nullptr,
                                  argv.data(), environ);
    if (rc != 0) {
        spdlog::warn("ArenaIpcClient: spawn '{}' failed: {}", binary,
                     std::strerror(rc));
        return false;
    }
    int status = 0;
    ::waitpid(pid, &status, 0);  // parent exits fast (daemon ready or failed)
    const bool ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (!ok)
        spdlog::warn("ArenaIpcClient: holder '{}' exited with status {}",
                     binary, WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    return ok;
}

/// Candidate holder binaries when the config gives none: alongside the build
/// tree (dev runs launch from the repo root), then PATH.
std::vector<std::string> holder_candidates(const std::string& configured) {
    if (!configured.empty()) return {configured};
    std::vector<std::string> c;
    c.push_back("build/tools/layerstorm_arena_holder");
    c.push_back("layerstorm_arena_holder");  // PATH (posix_spawnp)
    return c;
}

}  // namespace

ArenaIpcClient::~ArenaIpcClient() {
    detach();
    close_owned_fds();
}

void ArenaIpcClient::close_owned_fds() {
    if (meta_fd_ >= 0) ::close(meta_fd_);
    meta_fd_ = -1;
    for (auto& s : segments_)
        if (s.fd >= 0) ::close(s.fd);
    segments_.clear();
}

bool ArenaIpcClient::wipe(ipc::WipeReason reason, const std::string& detail) {
    if (sock_ < 0) return false;
    // Header + WipeRequest in ONE datagram (SEQPACKET keeps them together).
    char msg[sizeof(ipc::MsgHeader) + sizeof(ipc::WipeRequest)];
    ipc::MsgHeader hdr;
    hdr.type = static_cast<uint32_t>(ipc::MsgType::kWipe);
    hdr.payload_bytes = static_cast<uint32_t>(sizeof(ipc::WipeRequest));
    ipc::WipeRequest req;
    req.reason = static_cast<uint32_t>(reason);
    std::memcpy(req.detail, detail.data(),
                std::min(detail.size(), ipc::kMaxWipeDetail));
    std::memcpy(msg, &hdr, sizeof(hdr));
    std::memcpy(msg + sizeof(hdr), &req, sizeof(req));
    if (ipc::send_with_fds(sock_, msg, sizeof(msg), nullptr, 0) < 0)
        return false;
    ipc::StoreReply reply{};
    const ssize_t n =
        ipc::recv_with_fds(sock_, &reply, sizeof(reply), nullptr, 0, nullptr);
    const bool ok = n == static_cast<ssize_t>(sizeof(reply)) &&
                    reply.magic == ipc::kMagic &&
                    reply.status == static_cast<uint32_t>(ipc::StoreStatus::kOk);
    if (!ok)
        spdlog::warn("ArenaIpcClient: wipe (reason={}) failed/rejected",
                     ipc::wipe_reason_name(static_cast<uint32_t>(reason)));
    return ok;
}

void ArenaIpcClient::detach() {
    if (sock_ < 0) return;
    ipc::MsgHeader hdr;
    hdr.type = static_cast<uint32_t>(ipc::MsgType::kDetach);
    ipc::send_with_fds(sock_, &hdr, sizeof(hdr), nullptr, 0);  // best-effort
    ::close(sock_);
    sock_ = -1;
}

ArenaIpcClient::AttachOutcome ArenaIpcClient::attach(
        const std::string& socket_path, const std::string& holder_binary,
        bool auto_spawn) {
    if (sock_ >= 0) return AttachOutcome::kError;  // already attached
    close_owned_fds();  // re-attach after detach: drop any prior warm fds

    int s = try_connect(socket_path);
    if (s < 0 && auto_spawn) {
        bool spawned = false;
        for (const auto& cand : holder_candidates(holder_binary)) {
            // Direct paths: check existence first to keep the log clean; bare
            // names go straight to posix_spawnp's PATH lookup.
            if (cand.find('/') != std::string::npos && !executable_exists(cand))
                continue;
            if (spawn_holder(cand, socket_path)) {
                spdlog::info("ArenaIpcClient: spawned holder '{}' on {}", cand,
                             socket_path);
                spawned = true;
                break;
            }
        }
        if (spawned) {
            // The --daemon parent returns only after bind+listen, so one round
            // of short retries covers scheduler jitter.
            for (int i = 0; i < 50 && s < 0; ++i) {
                s = try_connect(socket_path);
                if (s < 0)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
    if (s < 0) {
        spdlog::warn("ArenaIpcClient: cannot reach holder on {} ({})",
                     socket_path, std::strerror(errno));
        return AttachOutcome::kError;
    }

    ipc::MsgHeader hdr;
    hdr.type = static_cast<uint32_t>(ipc::MsgType::kAttach);
    if (ipc::send_with_fds(s, &hdr, sizeof(hdr), nullptr, 0) < 0) {
        ::close(s);
        return AttachOutcome::kError;
    }

    std::vector<char> buf(sizeof(ipc::AttachReply) +
                          ipc::kMaxSegments * sizeof(ipc::SegmentDesc));
    int fds[ipc::kMaxSegments + 1];
    size_t got_fds = 0;
    const ssize_t n = ipc::recv_with_fds(s, buf.data(), buf.size(), fds,
                                         ipc::kMaxSegments + 1, &got_fds);
    auto fail = [&](const char* why) {
        for (size_t i = 0; i < got_fds; ++i) ::close(fds[i]);
        ::close(s);
        spdlog::warn("ArenaIpcClient: attach failed — {}", why);
        return AttachOutcome::kError;
    };
    if (n < static_cast<ssize_t>(sizeof(ipc::AttachReply)))
        return fail("short reply");
    ipc::AttachReply reply;
    std::memcpy(&reply, buf.data(), sizeof(reply));
    if (reply.magic != ipc::kMagic) return fail("bad reply magic");

    switch (static_cast<ipc::AttachStatus>(reply.status)) {
        case ipc::AttachStatus::kBusy:
            for (size_t i = 0; i < got_fds; ++i) ::close(fds[i]);
            ::close(s);
            return AttachOutcome::kBusy;
        case ipc::AttachStatus::kEmpty:
            for (size_t i = 0; i < got_fds; ++i) ::close(fds[i]);
            sock_ = s;
            return AttachOutcome::kEmpty;
        case ipc::AttachStatus::kWarm: {
            const size_t num_segs = (reply.num_fds >= 1) ? reply.num_fds - 1 : 0;
            if (got_fds != reply.num_fds || num_segs == 0 ||
                reply.payload_bytes != num_segs * sizeof(ipc::SegmentDesc) ||
                static_cast<size_t>(n) !=
                    sizeof(ipc::AttachReply) + reply.payload_bytes)
                return fail("inconsistent warm reply");
            meta_fd_ = fds[0];
            segments_.resize(num_segs);
            for (size_t i = 0; i < num_segs; ++i) {
                ipc::SegmentDesc d;
                std::memcpy(&d, buf.data() + sizeof(ipc::AttachReply) +
                                    i * sizeof(ipc::SegmentDesc), sizeof(d));
                // Belt & braces: the memfd's real size must match the stored
                // geometry (a truncated/corrupt store must not adopt).
                struct stat st{};
                if (::fstat(fds[1 + i], &st) != 0 ||
                    static_cast<uint64_t>(st.st_size) != d.size_bytes) {
                    meta_fd_ = -1;  // fds closed below via segments_/fail
                    segments_.clear();
                    return fail("segment size mismatch vs fstat");
                }
                segments_[i] = ArenaIpcSegment{d.numa_node,
                                               static_cast<size_t>(d.size_bytes),
                                               fds[1 + i]};
            }
            sock_ = s;
            return AttachOutcome::kWarm;
        }
        default:
            return fail("unknown attach status");
    }
}

bool ArenaIpcClient::store(int meta_fd,
                           const std::vector<ArenaIpcSegment>& segments) {
    if (sock_ < 0 || meta_fd < 0 || segments.empty() ||
        segments.size() > ipc::kMaxSegments)
        return false;

    std::vector<char> msg(sizeof(ipc::MsgHeader) +
                          segments.size() * sizeof(ipc::SegmentDesc));
    ipc::MsgHeader hdr;
    hdr.type = static_cast<uint32_t>(ipc::MsgType::kStore);
    hdr.payload_bytes =
        static_cast<uint32_t>(segments.size() * sizeof(ipc::SegmentDesc));
    std::memcpy(msg.data(), &hdr, sizeof(hdr));
    std::vector<int> fds;
    fds.reserve(1 + segments.size());
    fds.push_back(meta_fd);
    for (size_t i = 0; i < segments.size(); ++i) {
        ipc::SegmentDesc d{segments[i].numa_node, 0,
                           static_cast<uint64_t>(segments[i].size_bytes)};
        std::memcpy(msg.data() + sizeof(hdr) + i * sizeof(d), &d, sizeof(d));
        fds.push_back(segments[i].fd);
    }
    if (ipc::send_with_fds(sock_, msg.data(), msg.size(), fds.data(),
                           fds.size()) < 0) {
        spdlog::warn("ArenaIpcClient: store send failed: {}",
                     std::strerror(errno));
        return false;
    }
    ipc::StoreReply reply{};
    const ssize_t n =
        ipc::recv_with_fds(sock_, &reply, sizeof(reply), nullptr, 0, nullptr);
    const bool ok = n == static_cast<ssize_t>(sizeof(reply)) &&
                    reply.magic == ipc::kMagic &&
                    reply.status == static_cast<uint32_t>(ipc::StoreStatus::kOk);
    if (!ok) spdlog::warn("ArenaIpcClient: holder rejected store");
    return ok;
}

std::vector<ArenaIpcSegment> ArenaIpcClient::take_segments() {
    return std::exchange(segments_, {});
}

int ArenaIpcClient::take_meta_fd() {
    return std::exchange(meta_fd_, -1);
}

}  // namespace layerstorm::memory

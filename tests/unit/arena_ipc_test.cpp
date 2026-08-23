// Unit tests for the arena-holder IPC building blocks (P-24b):
//   - arena_ipc_protocol.h SCM_RIGHTS send/recv over a SOCK_SEQPACKET pair,
//   - abstract-namespace sockaddr construction,
//   - NumaManager shared (memfd) allocation + adopt roundtrip — the mechanism
//     that lets arena RAM survive a process via holder-held fd dups.
// All in-process, no CUDA, no real holder binary (that is
// tests/manual/arena_holder_manual.cpp).

#include <gtest/gtest.h>

#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <string>

#include "core/memory/arena_ipc_protocol.h"
#include "core/memory/numa_manager.h"

namespace lc = layerstorm::config;
namespace lm = layerstorm::memory;
namespace ipc = layerstorm::memory::arena_ipc;

namespace {

lc::GpuConfig make_gpu(int id, int numa_node) {
    lc::GpuConfig g;
    g.id = id;
    g.type = lc::GpuType::rtx5090;
    g.vram_gb = 32.0;
    g.numa_node = numa_node;
    return g;
}

lc::HardwareConfig single_node_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 128;
    hw.gpus = {make_gpu(0, 0), make_gpu(1, 0)};
    return hw;
}

int make_pattern_memfd(size_t bytes, unsigned char seed) {
    int fd = ::memfd_create("test-seg", MFD_CLOEXEC);
    EXPECT_GE(fd, 0);
    EXPECT_EQ(::ftruncate(fd, static_cast<off_t>(bytes)), 0);
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    EXPECT_NE(p, MAP_FAILED);
    std::memset(p, seed, bytes);
    ::munmap(p, bytes);
    return fd;
}

}  // namespace

TEST(ArenaIpcProtocol, MakeSockaddrFilesystemAndAbstract) {
    sockaddr_un addr{};
    // Filesystem path: sun_path is the NUL-terminated path.
    socklen_t len = ipc::make_sockaddr("/tmp/x.sock", &addr);
    ASSERT_GT(len, 0u);
    EXPECT_STREQ(addr.sun_path, "/tmp/x.sock");
    // Abstract: leading '@' becomes a leading NUL.
    len = ipc::make_sockaddr("@abstract-name", &addr);
    ASSERT_GT(len, 0u);
    EXPECT_EQ(addr.sun_path[0], '\0');
    EXPECT_EQ(std::memcmp(addr.sun_path + 1, "abstract-name", 13), 0);
    // Over-long path is rejected.
    std::string longpath(200, 'a');
    EXPECT_EQ(ipc::make_sockaddr(longpath.c_str(), &addr), 0u);
}

TEST(ArenaIpcProtocol, SendRecvWithFdsRoundtrip) {
    int sv[2];
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv), 0);

    constexpr size_t kBytes = 64 * 1024;
    const int seg_fd = make_pattern_memfd(kBytes, 0x5A);

    ipc::MsgHeader hdr;
    hdr.type = static_cast<uint32_t>(ipc::MsgType::kStore);
    ASSERT_GT(ipc::send_with_fds(sv[0], &hdr, sizeof(hdr), &seg_fd, 1), 0);

    ipc::MsgHeader got{};
    int fds[4] = {-1, -1, -1, -1};
    size_t nfds = 0;
    const ssize_t n = ipc::recv_with_fds(sv[1], &got, sizeof(got), fds, 4, &nfds);
    ASSERT_EQ(n, static_cast<ssize_t>(sizeof(got)));
    EXPECT_EQ(got.magic, ipc::kMagic);
    EXPECT_EQ(got.type, static_cast<uint32_t>(ipc::MsgType::kStore));
    ASSERT_EQ(nfds, 1u);
    ASSERT_GE(fds[0], 0);
    EXPECT_NE(fds[0], seg_fd);  // a kernel dup, not the same fd number required
                                // (may coincide; content check is the real test)

    // The received fd maps to the SAME pages: the pattern is visible.
    void* p = ::mmap(nullptr, kBytes, PROT_READ, MAP_SHARED, fds[0], 0);
    ASSERT_NE(p, MAP_FAILED);
    EXPECT_EQ(static_cast<unsigned char*>(p)[0], 0x5A);
    EXPECT_EQ(static_cast<unsigned char*>(p)[kBytes - 1], 0x5A);
    ::munmap(p, kBytes);

    ::close(fds[0]);
    ::close(seg_fd);
    ::close(sv[0]);
    ::close(sv[1]);
}

TEST(ArenaIpcProtocol, RecvEofReturnsZero) {
    int sv[2];
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv), 0);
    ::close(sv[0]);
    char buf[8];
    size_t nfds = 123;
    EXPECT_EQ(ipc::recv_with_fds(sv[1], buf, sizeof(buf), nullptr, 0, &nfds), 0);
    EXPECT_EQ(nfds, 0u);
    ::close(sv[1]);
}

// ── NumaManager shared allocation (the persistence substrate) ────────────────

TEST(NumaSharedAlloc, SharedAllocSurvivesFreeViaDupAndAdopt) {
    lm::NumaManager numa(single_node_hw());
    constexpr size_t kBytes = 256 * 1024;

    lm::NumaBuffer buf = numa.allocate_on_node_shared(kBytes, 0, "test-arena");
    ASSERT_NE(buf.data, nullptr);
    ASSERT_GE(buf.fd, 0);
    EXPECT_EQ(buf.numa_node, 0);
    std::memset(buf.data, 0xAB, buf.size);

    // "Holder": a dup of the fd keeps the tmpfs pages alive past free().
    const int keeper = ::dup(buf.fd);
    ASSERT_GE(keeper, 0);
    const size_t size = buf.size;
    numa.free(buf);
    EXPECT_EQ(buf.data, nullptr);
    EXPECT_EQ(buf.fd, -1);
    EXPECT_EQ(numa.total_allocated_bytes(), 0u);

    // "Next run": adopt the kept fd — the content is still there.
    lm::NumaBuffer back = numa.adopt_shared(keeper, size, 0);
    ASSERT_NE(back.data, nullptr);
    EXPECT_EQ(back.size, size);
    EXPECT_EQ(static_cast<unsigned char*>(back.data)[0], 0xAB);
    EXPECT_EQ(static_cast<unsigned char*>(back.data)[size - 1], 0xAB);
    numa.free(back);
    EXPECT_EQ(numa.total_allocated_bytes(), 0u);
}

TEST(NumaSharedAlloc, AdoptSizeMismatchThrows) {
    lm::NumaManager numa(single_node_hw());
    lm::NumaBuffer buf = numa.allocate_on_node_shared(64 * 1024, 0);
    const int keeper = ::dup(buf.fd);
    const size_t size = buf.size;
    numa.free(buf);
    EXPECT_THROW(numa.adopt_shared(keeper, size * 2, 0), std::runtime_error);
    ::close(keeper);  // adopt_shared does not consume the fd on validation failure
}

// ── kWipe reason contract (2026-08-22) ───────────────────────────────────────
// Wire-level checks of the reason field: only the two stated reasons are
// honored, a payload-less (pre-contract) kWipe decodes as kUnspecified, and the
// fixed WipeRequest payload round-trips over a SEQPACKET pair intact.

TEST(ArenaIpcWipeReason, OnlyStatedReasonsAreHonored) {
    EXPECT_FALSE(ipc::wipe_reason_honored(
        static_cast<uint32_t>(ipc::WipeReason::kUnspecified)));
    EXPECT_TRUE(ipc::wipe_reason_honored(
        static_cast<uint32_t>(ipc::WipeReason::kIdentityMismatch)));
    EXPECT_TRUE(ipc::wipe_reason_honored(
        static_cast<uint32_t>(ipc::WipeReason::kOperatorRequest)));
    EXPECT_FALSE(ipc::wipe_reason_honored(9999u));  // unknown → refused
    EXPECT_STREQ(ipc::wipe_reason_name(0), "unspecified");
    EXPECT_STREQ(ipc::wipe_reason_name(1), "identity_mismatch");
    EXPECT_STREQ(ipc::wipe_reason_name(2), "operator_request");
    EXPECT_STREQ(ipc::wipe_reason_name(7), "unknown");
    // Wire numbering is frozen — older binaries decode these values.
    EXPECT_EQ(static_cast<uint32_t>(ipc::WipeReason::kUnspecified), 0u);
    EXPECT_EQ(static_cast<uint32_t>(ipc::WipeReason::kIdentityMismatch), 1u);
    EXPECT_EQ(static_cast<uint32_t>(ipc::WipeReason::kOperatorRequest), 2u);
}

TEST(ArenaIpcWipeReason, PayloadRoundTripAndLegacyDecodesUnspecified) {
    int sv[2];
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sv), 0);

    // Reasoned wipe: header + fixed payload in ONE datagram.
    char msg[sizeof(ipc::MsgHeader) + sizeof(ipc::WipeRequest)];
    ipc::MsgHeader hdr;
    hdr.type = static_cast<uint32_t>(ipc::MsgType::kWipe);
    hdr.payload_bytes = static_cast<uint32_t>(sizeof(ipc::WipeRequest));
    ipc::WipeRequest req;
    req.reason = static_cast<uint32_t>(ipc::WipeReason::kIdentityMismatch);
    const char* detail = "config identity (geometry/placement/source hash)";
    std::memcpy(req.detail, detail, std::strlen(detail));
    std::memcpy(msg, &hdr, sizeof(hdr));
    std::memcpy(msg + sizeof(hdr), &req, sizeof(req));
    ASSERT_GT(ipc::send_with_fds(sv[0], msg, sizeof(msg), nullptr, 0), 0);

    char buf[sizeof(ipc::MsgHeader) +
             ipc::kMaxSegments * sizeof(ipc::SegmentDesc)];
    size_t got = 0;
    ssize_t n = ipc::recv_with_fds(sv[1], buf, sizeof(buf), nullptr, 0, &got);
    ASSERT_EQ(n, static_cast<ssize_t>(sizeof(msg)));
    EXPECT_EQ(got, 0u);
    ipc::MsgHeader rhdr;
    std::memcpy(&rhdr, buf, sizeof(rhdr));
    ASSERT_EQ(rhdr.payload_bytes, sizeof(ipc::WipeRequest));
    ipc::WipeRequest rreq;
    std::memcpy(&rreq, buf + sizeof(rhdr), sizeof(rreq));
    EXPECT_TRUE(ipc::wipe_reason_honored(rreq.reason));
    EXPECT_EQ(std::string(rreq.detail, std::strlen(detail)), detail);

    // Pre-contract client: payload-less kWipe → the holder's parse leaves the
    // default-constructed request, i.e. kUnspecified → refused.
    ipc::MsgHeader bare;
    bare.type = static_cast<uint32_t>(ipc::MsgType::kWipe);
    ASSERT_GT(ipc::send_with_fds(sv[0], &bare, sizeof(bare), nullptr, 0), 0);
    n = ipc::recv_with_fds(sv[1], buf, sizeof(buf), nullptr, 0, &got);
    ASSERT_EQ(n, static_cast<ssize_t>(sizeof(bare)));
    std::memcpy(&rhdr, buf, sizeof(rhdr));
    ipc::WipeRequest defaulted;
    if (rhdr.payload_bytes == sizeof(ipc::WipeRequest) &&
        sizeof(ipc::MsgHeader) + rhdr.payload_bytes == static_cast<size_t>(n))
        std::memcpy(&defaulted, buf + sizeof(rhdr), sizeof(defaulted));
    EXPECT_EQ(defaulted.reason,
              static_cast<uint32_t>(ipc::WipeReason::kUnspecified));
    EXPECT_FALSE(ipc::wipe_reason_honored(defaulted.reason));

    ::close(sv[0]);
    ::close(sv[1]);
}

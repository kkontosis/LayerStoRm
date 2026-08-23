// Manual multi-process test for the persistent-arena holder (P-24b).
//
// Spawns the REAL layerstorm_arena_holder binary and exercises the full
// attach/store/detach lifecycle across processes:
//   - cold attach (kEmpty) → create shared segments → STORE → detach,
//   - warm re-attach (kWarm) → adopt → preserved bytes verified,
//   - single-attachment: a second attach while attached → kBusy,
//   - SIGKILL the attached process → holder auto-detaches (EOF) → re-attach OK,
//   - kWipe reason contract: kUnspecified REFUSED (store survives) while
//     kIdentityMismatch / kOperatorRequest are HONORED and the holder process
//     stays up to host the successor store,
//   - the retired --persist flag makes the holder exit 2 (never a silent veto).
//
// Holder binary: $LS_ARENA_HOLDER_BIN, else build/tools/layerstorm_arena_holder
// (run from the repo root). Unique per-run ABSTRACT socket name — no /tmp
// paths, no cross-shard collisions (TD-102 lesson).
//
// Build: cmake --build build --target arena_holder_manual
// Run:   ./build/tests/manual/arena_holder_manual

#include <gtest/gtest.h>

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "core/memory/arena_ipc_client.h"
#include "core/memory/numa_manager.h"

namespace lc = layerstorm::config;
namespace lm = layerstorm::memory;

namespace {

using Outcome = lm::ArenaIpcClient::AttachOutcome;
using Reason = lm::arena_ipc::WipeReason;

std::string holder_binary() {
    if (const char* env = std::getenv("LS_ARENA_HOLDER_BIN")) return env;
    return "build/tools/layerstorm_arena_holder";
}

std::string unique_socket() {
    return "@ls-arena-manual-" + std::to_string(::getpid());
}

lc::HardwareConfig single_node_hw() {
    lc::HardwareConfig hw;
    hw.system_ram_gb = 128;
    lc::GpuConfig g;
    g.id = 0;
    g.type = lc::GpuType::rtx5090;
    g.vram_gb = 32.0;
    g.numa_node = 0;
    hw.gpus = {g};
    return hw;
}

/// Attach with a short retry (covers holder startup and the instant between a
/// SIGKILLed client's EOF and the holder reaping it).
Outcome attach_retry(lm::ArenaIpcClient& c, const std::string& sock,
                     int tries = 50) {
    Outcome o = Outcome::kError;
    for (int i = 0; i < tries; ++i) {
        o = c.attach(sock, /*holder_binary=*/"", /*auto_spawn=*/false);
        if (o == Outcome::kEmpty || o == Outcome::kWarm) return o;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return o;
}

class ArenaHolderManual : public ::testing::Test {
protected:
    void SetUp() override {
        sock_ = unique_socket();
        holder_pid_ = ::fork();
        ASSERT_GE(holder_pid_, 0);
        if (holder_pid_ == 0) {
            ::execl(holder_binary().c_str(), holder_binary().c_str(),
                    "--socket", sock_.c_str(), (char*)nullptr);
            _exit(127);  // exec failed
        }
    }
    void TearDown() override {
        if (holder_pid_ > 0) {
            ::kill(holder_pid_, SIGTERM);
            int status = 0;
            ::waitpid(holder_pid_, &status, 0);
        }
    }
    std::string sock_;
    pid_t holder_pid_ = -1;
};

}  // namespace

TEST_F(ArenaHolderManual, FullLifecycleColdWarmBusyKill) {
    lm::NumaManager numa(single_node_hw());
    constexpr size_t kBytes = 4 << 20;

    // ── Cold attach: empty holder, create + fill + STORE ─────────────────────
    {
        lm::ArenaIpcClient a;
        ASSERT_EQ(attach_retry(a, sock_), Outcome::kEmpty)
            << "holder did not come up (binary: " << holder_binary() << ")";

        lm::NumaBuffer seg = numa.allocate_on_node_shared(kBytes, 0);
        std::memset(seg.data, 0xE7, seg.size);
        const int meta_fd = ::memfd_create("manual-meta", MFD_CLOEXEC);
        ASSERT_GE(meta_fd, 0);
        ASSERT_EQ(::ftruncate(meta_fd, 4096), 0);

        ASSERT_TRUE(a.store(meta_fd, {{0, seg.size, seg.fd}}));

        // ── Single attachment: B is refused while A holds the socket ─────────
        lm::ArenaIpcClient b;
        EXPECT_EQ(b.attach(sock_, "", false), Outcome::kBusy);

        ::close(meta_fd);
        numa.free(seg);  // our mapping/fd die; the holder's dups keep the pages
    }  // A detaches (socket close)

    // ── Warm re-attach: stored fds come back, bytes preserved ────────────────
    size_t warm_size = 0;
    {
        lm::ArenaIpcClient c;
        ASSERT_EQ(attach_retry(c, sock_), Outcome::kWarm);
        ASSERT_EQ(c.segments().size(), 1u);
        EXPECT_EQ(c.segments()[0].numa_node, 0);
        warm_size = c.segments()[0].size_bytes;
        ASSERT_GE(c.meta_fd(), 0);

        auto segs = c.take_segments();
        lm::NumaBuffer back = numa.adopt_shared(segs[0].fd, warm_size, 0);
        EXPECT_EQ(static_cast<unsigned char*>(back.data)[0], 0xE7);
        EXPECT_EQ(static_cast<unsigned char*>(back.data)[warm_size - 1], 0xE7);
        numa.free(back);
    }

    // ── SIGKILL an attached process: holder must auto-detach on EOF ──────────
    pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        lm::ArenaIpcClient c;
        if (attach_retry(c, sock_) != Outcome::kWarm) _exit(1);
        ::pause();  // hold the attachment until killed
        _exit(0);
    }
    // Wait until the child is actually attached (holder replies BUSY to us).
    {
        lm::ArenaIpcClient probe;
        Outcome o = Outcome::kError;
        for (int i = 0; i < 100; ++i) {
            o = probe.attach(sock_, "", false);
            if (o == Outcome::kBusy) break;
            if (o == Outcome::kWarm || o == Outcome::kEmpty) probe.detach();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ASSERT_EQ(o, Outcome::kBusy);
    }
    ::kill(child, SIGKILL);
    int status = 0;
    ::waitpid(child, &status, 0);

    // Holder saw EOF → a fresh attach succeeds and the data is STILL there.
    {
        lm::ArenaIpcClient d;
        ASSERT_EQ(attach_retry(d, sock_), Outcome::kWarm);
        auto segs = d.take_segments();
        ASSERT_EQ(segs.size(), 1u);
        lm::NumaBuffer back = numa.adopt_shared(segs[0].fd, warm_size, 0);
        EXPECT_EQ(static_cast<unsigned char*>(back.data)[123], 0xE7);
        numa.free(back);
    }
}

TEST_F(ArenaHolderManual, StoreReplacesPreviousGeneration) {
    lm::NumaManager numa(single_node_hw());
    constexpr size_t kBytes = 1 << 20;
    {
        lm::ArenaIpcClient a;
        ASSERT_EQ(attach_retry(a, sock_), Outcome::kEmpty);
        lm::NumaBuffer s1 = numa.allocate_on_node_shared(kBytes, 0);
        std::memset(s1.data, 0x01, s1.size);
        const int m1 = ::memfd_create("m1", MFD_CLOEXEC);
        ASSERT_EQ(::ftruncate(m1, 4096), 0);
        ASSERT_TRUE(a.store(m1, {{0, s1.size, s1.fd}}));
        // Second generation (e.g. geometry wipe): replaces the first.
        lm::NumaBuffer s2 = numa.allocate_on_node_shared(kBytes * 2, 0);
        std::memset(s2.data, 0x02, s2.size);
        const int m2 = ::memfd_create("m2", MFD_CLOEXEC);
        ASSERT_EQ(::ftruncate(m2, 4096), 0);
        ASSERT_TRUE(a.store(m2, {{0, s2.size, s2.fd}}));
        ::close(m1);
        ::close(m2);
        numa.free(s1);
        numa.free(s2);
    }
    lm::ArenaIpcClient c;
    ASSERT_EQ(attach_retry(c, sock_), Outcome::kWarm);
    ASSERT_EQ(c.segments().size(), 1u);
    auto segs = c.take_segments();
    lm::NumaBuffer back = numa.adopt_shared(segs[0].fd, segs[0].size_bytes, 0);
    EXPECT_EQ(back.size, numa.total_allocated_bytes());
    EXPECT_EQ(static_cast<unsigned char*>(back.data)[0], 0x02);
    numa.free(back);

    // kWipe: the holder drops its generation immediately (shape-mismatch
    // path — old arena freed BEFORE the engine builds the replacement)…
    ASSERT_TRUE(c.wipe(Reason::kIdentityMismatch, "config identity differs"));
    c.detach();
    // …so the next attach is EMPTY, not warm.
    lm::ArenaIpcClient d;
    ASSERT_EQ(attach_retry(d, sock_), Outcome::kEmpty);
}

TEST_F(ArenaHolderManual, WipeRequiresAttachment) {
    lm::ArenaIpcClient a;
    ASSERT_EQ(attach_retry(a, sock_), Outcome::kEmpty);
    // Empty store: a reasoned wipe is a valid no-op.
    EXPECT_TRUE(a.wipe(Reason::kOperatorRequest, "no-op"));
    lm::ArenaIpcClient never_attached;
    EXPECT_FALSE(never_attached.wipe(Reason::kOperatorRequest));
}

// ── kWipe reason contract (2026-08-22) ───────────────────────────────────────
// The holder HONORS reasoned wipes (the incompatible store kills itself through
// the discovering client while the holder process survives to host the
// successor store) and REFUSES kUnspecified — which is also exactly what a
// pre-reason client's payload-less kWipe looks like on the wire.
TEST_F(ArenaHolderManual, UnspecifiedWipeRefusedStoreSurvives) {
    lm::NumaManager numa(single_node_hw());
    constexpr size_t kBytes = 1 << 20;
    {
        lm::ArenaIpcClient a;
        ASSERT_EQ(attach_retry(a, sock_), Outcome::kEmpty);
        lm::NumaBuffer seg = numa.allocate_on_node_shared(kBytes, 0);
        std::memset(seg.data, 0x5A, seg.size);
        const int meta_fd = ::memfd_create("m-unspec", MFD_CLOEXEC);
        ASSERT_GE(meta_fd, 0);
        ASSERT_EQ(::ftruncate(meta_fd, 4096), 0);
        ASSERT_TRUE(a.store(meta_fd, {{0, seg.size, seg.fd}}));

        // Refused: the client stays attached and the store is intact.
        EXPECT_FALSE(a.wipe(Reason::kUnspecified, "legacy client"));
        ::close(meta_fd);
        numa.free(seg);
    }
    lm::ArenaIpcClient b;
    ASSERT_EQ(attach_retry(b, sock_), Outcome::kWarm);
    ASSERT_EQ(b.segments().size(), 1u);
    auto segs = b.take_segments();
    lm::NumaBuffer back = numa.adopt_shared(segs[0].fd, segs[0].size_bytes, 0);
    EXPECT_EQ(static_cast<unsigned char*>(back.data)[0], 0x5A);
    numa.free(back);

    // …and an OPERATOR wipe on the very same holder IS honored: the process
    // stays up and hosts a fresh (successor) store afterwards.
    ASSERT_TRUE(b.wipe(Reason::kOperatorRequest, "switching models"));
    b.detach();
    lm::ArenaIpcClient c;
    ASSERT_EQ(attach_retry(c, sock_), Outcome::kEmpty);
    lm::NumaBuffer succ = numa.allocate_on_node_shared(kBytes, 0);
    std::memset(succ.data, 0xC3, succ.size);
    const int m2 = ::memfd_create("m-succ", MFD_CLOEXEC);
    ASSERT_GE(m2, 0);
    ASSERT_EQ(::ftruncate(m2, 4096), 0);
    ASSERT_TRUE(c.store(m2, {{0, succ.size, succ.fd}}));
    ::close(m2);
    numa.free(succ);
    c.detach();
    lm::ArenaIpcClient d;
    ASSERT_EQ(attach_retry(d, sock_), Outcome::kWarm);
    auto s2 = d.take_segments();
    ASSERT_EQ(s2.size(), 1u);
    lm::NumaBuffer back2 = numa.adopt_shared(s2[0].fd, s2[0].size_bytes, 0);
    EXPECT_EQ(static_cast<unsigned char*>(back2.data)[0], 0xC3);
    numa.free(back2);
}

// The RETIRED --persist flag must not silently produce a wipe-refusing holder:
// the binary rejects it (exit 2) rather than starting with the old veto.
TEST_F(ArenaHolderManual, PersistFlagRetired) {
    pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) { ::dup2(devnull, 2); ::close(devnull); }
        ::execl(holder_binary().c_str(), holder_binary().c_str(),
                "--socket", "@ls-arena-retired-flag", "--persist",
                (char*)nullptr);
        _exit(127);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 2);
}

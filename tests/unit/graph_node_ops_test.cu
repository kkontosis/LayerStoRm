// GPU tests for the graph-node byte-mover kernels (P-29 step 2).
//
// These three launchers exist ONLY to replace memcpy/memset NODES inside a
// captured decode graph with kernel nodes (~1.5 us -> ~0.1 us of driver host
// time per node at replay). Their entire contract is therefore: the bytes they
// produce are IDENTICAL to the CUDA runtime call they displace. Every case
// below runs the launcher and the reference (cudaMemcpyAsync /
// cudaMemcpy2DAsync / cudaMemsetAsync) on the same inputs and memcmps the
// results byte-for-byte, plus guard regions to prove nothing outside the
// declared range is touched.
//
// Footprint is deliberately tiny (a few hundred KB on device 0) so this suite
// can run next to a live engine holding the rest of VRAM.

#include "../gpu_test_utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "compute/kernels/elementwise/graph_node_ops.h"

namespace lcomp = layerstorm::compute;

namespace {

#define ASSERT_CUDA(expr)                                                  \
    do {                                                                   \
        cudaError_t _e = (expr);                                           \
        ASSERT_EQ(cudaSuccess, _e) << #expr << " -> " << cudaGetErrorString(_e); \
    } while (0)

/// RAII device allocation.
struct DevBuf {
    void* p = nullptr;
    size_t bytes = 0;
    DevBuf() = default;
    explicit DevBuf(size_t n) : bytes(n) {
        EXPECT_EQ(cudaSuccess, cudaMalloc(&p, n));
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; }
    ~DevBuf() { if (p) cudaFree(p); }
    unsigned char* u8() const { return static_cast<unsigned char*>(p); }
};

/// RAII pinned-host allocation via cudaHostAlloc(cudaHostAllocDefault) — the
/// exact staging flavour the production graph path dereferences from device
/// through UVA.
struct PinnedBuf {
    void* p = nullptr;
    PinnedBuf() = default;
    explicit PinnedBuf(size_t n) {
        EXPECT_EQ(cudaSuccess, cudaHostAlloc(&p, n, cudaHostAllocDefault));
    }
    PinnedBuf(const PinnedBuf&) = delete;
    PinnedBuf& operator=(const PinnedBuf&) = delete;
    PinnedBuf(PinnedBuf&& o) noexcept : p(o.p) { o.p = nullptr; }
    ~PinnedBuf() { if (p) cudaFreeHost(p); }
};

std::vector<unsigned char> download(const void* dptr, size_t bytes) {
    std::vector<unsigned char> h(bytes, 0x5A);
    EXPECT_EQ(cudaSuccess,
              cudaMemcpy(h.data(), dptr, bytes, cudaMemcpyDeviceToHost));
    return h;
}

/// Byte-exact comparison with a precise first-mismatch report.
void expect_bytes_equal(const std::vector<unsigned char>& got,
                        const std::vector<unsigned char>& want,
                        const std::string& what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    if (std::memcmp(got.data(), want.data(), got.size()) == 0) return;
    size_t first = 0;
    while (first < got.size() && got[first] == want[first]) ++first;
    size_t ndiff = 0;
    for (size_t i = 0; i < got.size(); ++i) ndiff += (got[i] != want[i]);
    ADD_FAILURE() << what << ": " << ndiff << " / " << got.size()
                  << " bytes differ; first mismatch at byte " << first
                  << " observed=0x" << std::hex << unsigned(got[first])
                  << " expected=0x" << unsigned(want[first]) << std::dec;
}

class GraphNodeOps : public ::testing::Test {
  protected:
    cudaStream_t stream_ = nullptr;
    void SetUp() override {
        REQUIRES_GPU();
        ASSERT_CUDA(cudaSetDevice(0));
        ASSERT_CUDA(cudaStreamCreate(&stream_));
    }
    void TearDown() override {
        if (stream_) cudaStreamDestroy(stream_);
    }
};

// ── 1. copy_ptr_arrays vs cudaMemcpyAsync ─────────────────────────────────
//
// n_arrays in {1,3,6} x n_ptrs in {288, 173}. Sources are pinned host buffers
// read from the device through UVA (production path). Each destination array
// carries a 64-byte 0xCD guard tail so an over-write past n_ptrs entries is
// caught.

void run_copy_ptr_arrays_case(cudaStream_t stream, int n_arrays, int n_ptrs) {
    SCOPED_TRACE("copy_ptr_arrays n_arrays=" + std::to_string(n_arrays) +
                 " n_ptrs=" + std::to_string(n_ptrs));
    const size_t payload = static_cast<size_t>(n_ptrs) * 8;
    constexpr size_t kGuard = 64;

    std::mt19937_64 rng(0xC0FFEEULL + n_arrays * 1000 + n_ptrs);
    std::vector<PinnedBuf> srcs;
    std::vector<DevBuf> dst_k, dst_r;
    srcs.reserve(n_arrays);
    for (int a = 0; a < n_arrays; ++a) {
        srcs.emplace_back(payload);
        ASSERT_NE(nullptr, srcs.back().p);
        auto* h = static_cast<uint64_t*>(srcs.back().p);
        for (int i = 0; i < n_ptrs; ++i) h[i] = rng();
        dst_k.emplace_back(payload + kGuard);
        dst_r.emplace_back(payload + kGuard);
        ASSERT_NE(nullptr, dst_k.back().p);
        ASSERT_NE(nullptr, dst_r.back().p);
        ASSERT_CUDA(cudaMemset(dst_k.back().p, 0xCD, payload + kGuard));
        ASSERT_CUDA(cudaMemset(dst_r.back().p, 0xCD, payload + kGuard));
    }

    // Kernel path: one launch for all arrays.
    lcomp::PtrArrayCopyDesc descs[lcomp::kMaxPtrCopyArrays]{};
    for (int a = 0; a < n_arrays; ++a) {
        descs[a].src = static_cast<const void* const*>(srcs[a].p);
        descs[a].dst = dst_k[a].p;
    }
    lcomp::launch_copy_ptr_arrays(descs, n_arrays, n_ptrs, stream);

    // Reference: the per-array H2D memcpy nodes it replaces.
    for (int a = 0; a < n_arrays; ++a) {
        ASSERT_CUDA(cudaMemcpyAsync(dst_r[a].p, srcs[a].p, payload,
                                    cudaMemcpyHostToDevice, stream));
    }
    ASSERT_CUDA(cudaStreamSynchronize(stream));
    ASSERT_CUDA(cudaGetLastError());

    for (int a = 0; a < n_arrays; ++a) {
        expect_bytes_equal(download(dst_k[a].p, payload + kGuard),
                           download(dst_r[a].p, payload + kGuard),
                           "array " + std::to_string(a) + " (payload+guard)");
    }
}

TEST_F(GraphNodeOps, CopyPtrArraysByteIdenticalToMemcpy) {
    for (int n_arrays : {1, 3, 6}) {
        for (int n_ptrs : {288, 173}) {
            run_copy_ptr_arrays_case(stream_, n_arrays, n_ptrs);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// ── 2. interleave_rows vs the two cudaMemcpy2DAsync nodes ─────────────────

void run_interleave_case(cudaStream_t stream, int rows, size_t row_bytes) {
    SCOPED_TRACE("interleave rows=" + std::to_string(rows) +
                 " row_bytes=" + std::to_string(row_bytes));
    const size_t src_bytes = static_cast<size_t>(rows) * row_bytes;
    const size_t dst_bytes = 2 * src_bytes;

    DevBuf src_a(src_bytes), src_b(src_bytes);
    DevBuf dst_k(dst_bytes), dst_r(dst_bytes);
    ASSERT_NE(nullptr, src_a.p);
    ASSERT_NE(nullptr, dst_k.p);

    // Distinct, position-dependent content so a wrong row/half is visible.
    std::vector<unsigned char> ha(src_bytes), hb(src_bytes);
    std::mt19937 rng(0xBEEF ^ (rows * 131 + static_cast<int>(row_bytes)));
    for (size_t i = 0; i < src_bytes; ++i) ha[i] = static_cast<unsigned char>(rng());
    for (size_t i = 0; i < src_bytes; ++i) hb[i] = static_cast<unsigned char>(rng());
    ASSERT_CUDA(cudaMemcpy(src_a.p, ha.data(), src_bytes, cudaMemcpyHostToDevice));
    ASSERT_CUDA(cudaMemcpy(src_b.p, hb.data(), src_bytes, cudaMemcpyHostToDevice));
    // Pre-fill both destinations identically with a non-zero pattern so any
    // byte the kernel fails to write shows up as a difference vs the memcpy2D
    // reference only if the reference wrote it (and vice versa).
    ASSERT_CUDA(cudaMemset(dst_k.p, 0xAB, dst_bytes));
    ASSERT_CUDA(cudaMemset(dst_r.p, 0xAB, dst_bytes));

    lcomp::launch_interleave_rows(dst_k.p, src_a.p, src_b.p, rows, row_bytes,
                                  stream);

    ASSERT_CUDA(cudaMemcpy2DAsync(dst_r.p, 2 * row_bytes, src_a.p, row_bytes,
                                  row_bytes, rows, cudaMemcpyDeviceToDevice,
                                  stream));
    ASSERT_CUDA(cudaMemcpy2DAsync(dst_r.u8() + row_bytes, 2 * row_bytes,
                                  src_b.p, row_bytes, row_bytes, rows,
                                  cudaMemcpyDeviceToDevice, stream));
    ASSERT_CUDA(cudaStreamSynchronize(stream));
    ASSERT_CUDA(cudaGetLastError());

    const auto got = download(dst_k.p, dst_bytes);
    const auto want = download(dst_r.p, dst_bytes);
    expect_bytes_equal(got, want, "interleaved dst");

    // Independent host-side expectation (guards against BOTH paths agreeing on
    // something wrong, e.g. if the memcpy2D reference were mis-parameterized).
    std::vector<unsigned char> expect(dst_bytes);
    for (int r = 0; r < rows; ++r) {
        std::memcpy(expect.data() + static_cast<size_t>(r) * 2 * row_bytes,
                    ha.data() + static_cast<size_t>(r) * row_bytes, row_bytes);
        std::memcpy(expect.data() + static_cast<size_t>(r) * 2 * row_bytes + row_bytes,
                    hb.data() + static_cast<size_t>(r) * row_bytes, row_bytes);
    }
    expect_bytes_equal(got, expect, "interleaved dst vs host reference");
}

TEST_F(GraphNodeOps, InterleaveRowsByteIdenticalToMemcpy2D) {
    run_interleave_case(stream_, 8, 4096);    // production shape
    if (HasFatalFailure()) return;
    run_interleave_case(stream_, 1, 4096);    // single row
    if (HasFatalFailure()) return;
    run_interleave_case(stream_, 129, 1024);  // more rows, odd count
    if (HasFatalFailure()) return;
    run_interleave_case(stream_, 64, 16);     // minimal 16-byte row
}

// ── 3. zero_fill vs cudaMemsetAsync ───────────────────────────────────────
//
// Guarded layout: [64 guard][offset pad][bytes][64 guard]. The whole buffer is
// pre-filled 0xAB; only [bytes) may become zero. Compared both against
// cudaMemsetAsync on an identical buffer and against an explicit host
// expectation.

void run_zero_fill_case(cudaStream_t stream, size_t bytes, size_t offset) {
    SCOPED_TRACE("zero_fill bytes=" + std::to_string(bytes) +
                 " offset=" + std::to_string(offset));
    constexpr size_t kGuard = 64;
    const size_t total = kGuard + offset + bytes + kGuard;

    DevBuf bk(total), br(total);
    ASSERT_NE(nullptr, bk.p);
    ASSERT_NE(nullptr, br.p);
    ASSERT_CUDA(cudaMemset(bk.p, 0xAB, total));
    ASSERT_CUDA(cudaMemset(br.p, 0xAB, total));

    // cudaMalloc is 256-byte aligned, so (guard + offset) sets the low 4
    // address bits exactly to `offset` — this is the unaligned-pointer arm.
    unsigned char* pk = bk.u8() + kGuard + offset;
    unsigned char* pr = br.u8() + kGuard + offset;
    ASSERT_EQ(offset,
              reinterpret_cast<uintptr_t>(pk) & 15ULL) << "test setup: base not 16B-aligned";

    lcomp::launch_zero_fill(pk, bytes, stream);
    ASSERT_CUDA(cudaMemsetAsync(pr, 0, bytes, stream));
    ASSERT_CUDA(cudaStreamSynchronize(stream));
    ASSERT_CUDA(cudaGetLastError());

    const auto got = download(bk.p, total);
    const auto want = download(br.p, total);
    expect_bytes_equal(got, want, "zero_fill buffer (incl. guards)");

    std::vector<unsigned char> expect(total, 0xAB);
    std::memset(expect.data() + kGuard + offset, 0, bytes);
    expect_bytes_equal(got, expect, "zero_fill buffer vs host reference");
}

TEST_F(GraphNodeOps, ZeroFillByteIdenticalToMemset) {
    const size_t sizes[] = {1, 15, 16, 17, 2304, 32768, 65536};
    for (size_t bytes : sizes) {
        run_zero_fill_case(stream_, bytes, 0);
        if (HasFatalFailure()) return;
    }
}

TEST_F(GraphNodeOps, ZeroFillUnalignedPointerByteIdenticalToMemset) {
    const size_t sizes[] = {1, 15, 16, 17, 2304, 32768, 65536};
    for (size_t bytes : sizes) {
        for (size_t offset = 1; offset <= 15; ++offset) {
            run_zero_fill_case(stream_, bytes, offset);
            if (HasFatalFailure()) return;
        }
    }
}

}  // namespace

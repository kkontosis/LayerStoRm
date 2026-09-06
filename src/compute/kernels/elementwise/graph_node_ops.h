// Kernel-node replacements for memcpy/memset nodes inside captured decode
// graphs (P-29 step 2 — KCOPY, 2026-09-03).
//
// Why: a cudaGraphLaunch replay pays ~1.5 us of host time PER memcpy/memset
// NODE (driver-side node handling at every launch), vs ~0.1 us for a kernel
// node. The routed-FFN decode graph carried 3 H2D + 2 D2D memcpy nodes and
// 3-4 memset nodes per replay (~8.6 us med replay measured at production
// pace); emitting the same byte movements as kernels cuts the replay toward
// the pure-kernel-graph floor (~2 us). Each launcher below produces results
// BYTE-IDENTICAL to the memcpy/memset it replaces.
//
// The pointer-array copy reads pinned host memory directly from the device
// (UVA dereference of cudaHostAlloc memory) — same execution-time-read
// semantics as the captured H2D node it replaces.

#pragma once

#include <cstddef>
#include <cstdint>

namespace layerstorm::compute {

// Copy up to kMaxPtrCopyArrays pointer arrays (n_ptrs 8-byte entries each)
// from pinned-host (device-dereferenceable via UVA) staging into device
// arrays, in ONE kernel launch. Replaces per-projection H2D memcpy nodes in
// the captured routed-FFN graph. Byte-identical to the memcpys it replaces.
inline constexpr int kMaxPtrCopyArrays = 6;

struct PtrArrayCopyDesc {
    const void* const* src;  // pinned host staging (UVA device-readable)
    void* dst;               // device pointer array
};

void launch_copy_ptr_arrays(
    const PtrArrayCopyDesc* descs,  // host array, copied by value into params
    int n_arrays,                   // <= kMaxPtrCopyArrays
    int n_ptrs,                     // entries per array
    void* stream /*cudaStream_t*/);

// Interleave two row-major sources into dst with pitch 2*row_bytes:
//   dst[r*2*row_bytes         .. +row_bytes) = src_a[r*row_bytes ..)
//   dst[r*2*row_bytes+row_bytes .. +row_bytes) = src_b[r*row_bytes ..)
// Replaces the two strided memcpy2D (gate/up -> gate_up interleave) nodes.
// Requires row_bytes % 16 == 0 and 16-byte-aligned pointers (callers keep
// the memcpy2D fallback otherwise). Byte-identical to the memcpy2D pair.
void launch_interleave_rows(
    void* dst,
    const void* src_a,
    const void* src_b,
    int rows,
    size_t row_bytes,
    void* stream /*cudaStream_t*/);

// Zero-fill `bytes` bytes at ptr. Replaces cudaMemsetAsync(ptr, 0, bytes)
// with a kernel node. Handles arbitrary sizes/alignment (vectorized body +
// byte tail). Byte-identical to the memset it replaces.
void launch_zero_fill(
    void* ptr,
    size_t bytes,
    void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute

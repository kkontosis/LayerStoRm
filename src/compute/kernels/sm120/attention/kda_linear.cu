// GF3.7: single-TU driver for the KDA (Kimi Delta Attention) SM120 kernels
// (see kda_linear.h). Includes the vendored kernel .cu files once and exposes
// throwing launch wrappers — the indexer_kpool.cu / lightning_indexer.cu
// pattern. No pre-existing kernel TU is touched (byte-identity by
// construction, measured by the GF3.7 SASS diff).

#include <stdexcept>
#include <string>

#include "sm120/linear/kda_conv.cu"
#include "sm120/linear/kda_decode.cu"
#include "sm120/linear/kda_chunk.cu"

#include "compute/kernels/sm120/attention/kda_linear.h"

namespace layerstorm::compute {

namespace {

namespace sl = sm120::linear;

void check(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(err));
    }
}

void require(bool ok, const char* what, const std::string& msg) {
    if (!ok) throw std::runtime_error(std::string(what) + ": " + msg);
}

void validate_heads(int num_heads, const char* what) {
    require(num_heads > 0 && num_heads <= 1024, what,
            "num_heads must be in 1..1024, got " + std::to_string(num_heads));
}

}  // namespace

size_t kda_prefill_workspace_bytes(int t_len, int num_heads) {
    return sl::kda_prefill_workspace_bytes(t_len, num_heads);
}

void launch_kda_conv_prefill(const sl::KdaConvPrefillParams& p, cudaStream_t stream) {
    const char* what = "launch_kda_conv_prefill";
    require(p.t_len > 0, what, "t_len must be positive, got " + std::to_string(p.t_len));
    require(p.channels > 0 && p.channels % sl::kKdaHeadDim == 0, what,
            "channels must be a positive multiple of 128, got " + std::to_string(p.channels));
    sl::run_kda_conv_prefill(p, stream);
    check(what);
}

void launch_kda_conv_decode(const sl::KdaConvDecodeParams& p, cudaStream_t stream) {
    const char* what = "launch_kda_conv_decode";
    require(p.batch > 0, what, "batch must be positive, got " + std::to_string(p.batch));
    require(p.channels > 0 && p.channels % sl::kKdaHeadDim == 0, what,
            "channels must be a positive multiple of 128, got " + std::to_string(p.channels));
    // TD-KDA-STATE-MAPPED-SLABS: with a real slot table the stride is a
    // UNIT stride (slab_bytes/4 under mapped — slot ids are spaced so
    // slot * stride lands on non-overlapping per-layer runs); only the
    // identity mapping (slots == nullptr, slot b = b) requires the stride
    // to cover the whole [3][C] ring span.
    require(p.slots != nullptr || p.ring_slot_stride >= 3ll * p.channels,
            what,
            "ring_slot_stride must cover [3][C] floats when slots is null, "
            "got " + std::to_string(p.ring_slot_stride));
    require(p.ring_slot_stride > 0, what,
            "ring_slot_stride must be positive, got " +
                std::to_string(p.ring_slot_stride));
    sl::run_kda_conv_decode(p, stream);
    check(what);
}

void launch_kda_chunked_scan(const sl::KdaChunkedScanParams& p, cudaStream_t stream) {
    const char* what = "launch_kda_chunked_scan";
    require(p.t_len > 0, what, "t_len must be positive, got " + std::to_string(p.t_len));
    validate_heads(p.num_heads, what);
    const size_t need = sl::kda_prefill_workspace_bytes(p.t_len, p.num_heads);
    require(p.workspace != nullptr, what, "workspace is null");
    require(p.workspace_bytes >= need, what,
            "workspace too small: need " + std::to_string(need) + " B, got " +
                std::to_string(p.workspace_bytes));
    sl::run_kda_chunked_scan(p, stream);
    check(what);
}

void launch_kda_gated_rmsnorm(const sl::KdaGatedRmsNormParams& p, cudaStream_t stream) {
    const char* what = "launch_kda_gated_rmsnorm";
    require(p.t_len > 0, what, "t_len must be positive, got " + std::to_string(p.t_len));
    validate_heads(p.num_heads, what);
    require(p.out_bf16 != nullptr || p.out_f32 != nullptr, what,
            "at least one of out_bf16 / out_f32 must be set");
    sl::run_kda_gated_rmsnorm(p, stream);
    check(what);
}

void launch_kda_decode_step(const sl::KdaDecodeStepParams& p, cudaStream_t stream) {
    const char* what = "launch_kda_decode_step";
    require(p.batch > 0, what, "batch must be positive, got " + std::to_string(p.batch));
    validate_heads(p.num_heads, what);
    // TD-KDA-STATE-MAPPED-SLABS: same rule as the conv rings — a slot
    // table legitimizes a unit stride smaller than the full state span.
    require(p.slots != nullptr ||
                p.state_slot_stride >=
                    static_cast<int64_t>(p.num_heads) * sl::kKdaHeadDim *
                        sl::kKdaHeadDim,
            what,
            "state_slot_stride must cover [H][128][128] floats when slots "
            "is null, got " + std::to_string(p.state_slot_stride));
    require(p.state_slot_stride > 0, what,
            "state_slot_stride must be positive, got " +
                std::to_string(p.state_slot_stride));
    sl::run_kda_decode_step(p, stream);
    check(what);
}

}  // namespace layerstorm::compute

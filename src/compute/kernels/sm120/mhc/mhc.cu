// Single-TU driver for the DeepSeek-V4 mHC kernels (V4-5b).
//
// Includes the vendored kernel .cu from deps/LayerStoRmKernels (which defines
// the __global__ kernels + run_* host functions inline) and exposes the
// CUDA-free launch wrappers declared in compute/kernels/mhc/mhc.h — the same
// pattern as lightning_indexer.cu / snapmla_prep.cu. Also implements the
// embedding repeat-expansion kernel (engine-local, no deps counterpart).

#include "smxx/mhc.cu"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "compute/kernels/mhc/mhc.h"

namespace layerstorm::compute {

namespace {

void check_launch(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " launch failed: " +
                                 cudaGetErrorString(err));
    }
}

__global__ void hc_expand_repeat_kernel(
    __nv_bfloat16* __restrict__ residual_out,
    const __nv_bfloat16* __restrict__ x_in,
    int rows, int hc, int hidden) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* src = x_in + static_cast<int64_t>(row) * hidden;
    __nv_bfloat16* dst = residual_out + static_cast<int64_t>(row) * hc * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const __nv_bfloat16 v = src[i];
        for (int s = 0; s < hc; ++s) dst[s * hidden + i] = v;
    }
}

}  // namespace

void launch_mhc_pre(void* x_out, void* post_out, void* comb_out,
                    const void* residual, const void* fn, const void* scale,
                    const void* base, float rms_eps, float hc_eps,
                    float post_mult, int sinkhorn_iters, int rows, int hc,
                    int hidden, void* stream) {
    smxx::mhc::MhcPreParams p{};
    p.residual = static_cast<const __nv_bfloat16*>(residual);
    p.residual_row_stride = static_cast<int64_t>(hc) * hidden;
    p.fn = static_cast<const float*>(fn);
    p.hc_scale = static_cast<const float*>(scale);
    p.hc_base = static_cast<const float*>(base);
    p.rms_eps = rms_eps;
    p.hc_eps = hc_eps;
    p.post_mult = post_mult;
    p.sinkhorn_iters = sinkhorn_iters;
    p.post_out = static_cast<float*>(post_out);
    p.comb_out = static_cast<float*>(comb_out);
    p.x_out = static_cast<__nv_bfloat16*>(x_out);
    p.x_out_row_stride = hidden;
    p.num_tokens = rows;
    p.hc = hc;
    p.hidden = hidden;
    smxx::mhc::run_mhc_pre(p, static_cast<cudaStream_t>(stream));
    check_launch("mhc_pre");
}

void launch_mhc_post(void* residual_out, const void* y, const void* residual,
                     const void* post, const void* comb, int rows, int hc,
                     int hidden, void* stream) {
    smxx::mhc::MhcPostParams p{};
    p.y = static_cast<const __nv_bfloat16*>(y);
    p.y_row_stride = hidden;
    p.residual = static_cast<const __nv_bfloat16*>(residual);
    p.residual_row_stride = static_cast<int64_t>(hc) * hidden;
    p.post = static_cast<const float*>(post);
    p.comb = static_cast<const float*>(comb);
    p.residual_out = static_cast<__nv_bfloat16*>(residual_out);
    p.residual_out_row_stride = static_cast<int64_t>(hc) * hidden;
    p.num_tokens = rows;
    p.hc = hc;
    p.hidden = hidden;
    smxx::mhc::run_mhc_post(p, static_cast<cudaStream_t>(stream));
    check_launch("mhc_post");
}

void launch_mhc_head(void* x_out, const void* residual, const void* fn,
                     const void* scale, const void* base, float rms_eps,
                     float hc_eps, int rows, int hc, int hidden, void* stream) {
    smxx::mhc::MhcHeadParams p{};
    p.residual = static_cast<const __nv_bfloat16*>(residual);
    p.residual_row_stride = static_cast<int64_t>(hc) * hidden;
    p.fn = static_cast<const float*>(fn);
    p.hc_scale = static_cast<const float*>(scale);
    p.hc_base = static_cast<const float*>(base);
    p.rms_eps = rms_eps;
    p.hc_eps = hc_eps;
    p.x_out = static_cast<__nv_bfloat16*>(x_out);
    p.x_out_row_stride = hidden;
    p.num_tokens = rows;
    p.hc = hc;
    p.hidden = hidden;
    smxx::mhc::run_mhc_head(p, static_cast<cudaStream_t>(stream));
    check_launch("mhc_head");
}

void launch_hc_expand_repeat(void* residual_out, const void* x_in, int rows,
                             int hc, int hidden, void* stream) {
    if (rows <= 0) return;
    hc_expand_repeat_kernel<<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(residual_out),
        static_cast<const __nv_bfloat16*>(x_in), rows, hc, hidden);
    check_launch("hc_expand_repeat");
}

}  // namespace layerstorm::compute

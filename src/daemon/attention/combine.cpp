// Attention combine phase (attention refactor V2 P2 home): the DCP
// LSE-correction / same-head cross-rank combine (INV-KVS-QAG) behind
// compute/kernels/attention/dcp_attention_wrapper.h. Moved verbatim from
// src/compute/dcp_attention_wrapper.cpp — a designated CUDA TU
// (CUDA_DESIGNATED_TUS; deliberate INV-GPU-1 allowlist path update).
#include "compute/kernels/attention/dcp_attention_wrapper.h"

#include "compute/kernels/attention/dcp_lse_correct.h"
#include "core/attention_device.h"
#include "parallelism/dcp_communicator.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <cstdio>    // TD-PREFILL-NONDET seam_exec_dump
#include <cstdlib>   // TD-PREFILL-NONDET getenv gate
#include <stdexcept>
#include <vector>    // TD-PREFILL-NONDET D2H staging

namespace layerstorm::compute {

namespace {
// TD-PREFILL-NONDET diagnostic (LS_SEAM_DUMP_EXEC=<path>, off by default):
// dump a device buffer at a named DCP-combine stage. `inv` is the
// correct_output invocation ordinal (== layer ordinal within the run:
// 78 layers per prefill chunk, so chunk 2 layer 0 is inv 78); records are
// emitted only for inv in [LS_SEAM_EXEC_FROM, LS_SEAM_EXEC_TO). Record (LE):
// int32 hdr[5]={tag4cc,inv,rank,1,bytes/2}; raw payload. Diagnosis-only —
// the D2H + stream sync serializes the combine when enabled.
void seam_exec_dump(uint32_t tag, int inv, int rank, const void* dev,
                    size_t bytes, void* stream) {
    static const char* path = std::getenv("LS_SEAM_DUMP_EXEC");
    if (!path || !*path || !dev || !bytes) return;
    static const int from = [] {
        const char* e = std::getenv("LS_SEAM_EXEC_FROM");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    static const int to = [] {
        const char* e = std::getenv("LS_SEAM_EXEC_TO");
        return (e && *e) ? std::atoi(e) : (1 << 30);
    }();
    if (inv < from || inv >= to) return;
    static std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return;
    std::vector<uint8_t> host(bytes);
    cudaMemcpyAsync(host.data(), dev, bytes, cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(stream));
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    int32_t hdr[5] = {static_cast<int32_t>(tag), inv, rank, 1,
                      static_cast<int32_t>(bytes / 2)};
    std::fwrite(hdr, sizeof(int32_t), 5, fp);
    std::fwrite(host.data(), 1, bytes, fp);
    std::fflush(fp);
}
}  // namespace

// -- cuda_launch_correction --------------------------------------------------

LaunchCorrectionFn cuda_launch_correction() {
    return [](void* output, const float* lses, float* global_lse,
              int B, int H, int D, int N, int rank, void* stream) {
        DcpLseCorrectParams params{};
        params.output     = output;
        params.lses       = lses;
        params.global_lse = global_lse;
        params.B    = B;
        params.H    = H;
        params.D    = D;
        params.N    = N;
        params.rank = rank;
        params.stride_o_B   = H * D;
        params.stride_o_H   = D;
        params.stride_o_D   = 1;
        params.stride_lse_N = B * H;
        params.stride_lse_B = H;
        params.stride_lse_H = 1;
        run_dcp_lse_correct_kernel(params,
                                    static_cast<cudaStream_t>(stream));
    };
}

// -- DcpAttentionWrapper -----------------------------------------------------

DcpAttentionWrapper::DcpAttentionWrapper(
    layerstorm::parallelism::DcpCommunicator* comm,
    DcpAttentionWrapperConfig cfg,
    std::vector<AttentionDevice*> attention_devices,
    LaunchCorrectionFn correction_fn)
    : comm_(comm),
      cfg_(std::move(cfg)),
      attention_devices_(std::move(attention_devices)),
      correction_fn_(std::move(correction_fn)),
      dcp_size_(comm ? comm->dcp_size() : 1),
      combine_heads_(cfg_.combine_num_heads > 0 ? cfg_.combine_num_heads
                                                : cfg_.num_heads_local) {

    if (!is_active()) {
        spdlog::debug("DcpAttentionWrapper: inactive (dcp_size={})", dcp_size_);
        return;
    }

    // TD-87c: Validate sizes match DcpExecutor's pattern.
    if (static_cast<int>(attention_devices_.size()) < dcp_size_) {
        throw std::invalid_argument(
            "DcpAttentionWrapper: attention_devices.size() (" +
            std::to_string(attention_devices_.size()) +
            ") must be >= dcp_size (" + std::to_string(dcp_size_) + ")");
    }
    if (static_cast<int>(cfg_.gpus.size()) < dcp_size_) {
        throw std::invalid_argument(
            "DcpAttentionWrapper: cfg.gpus.size() (" +
            std::to_string(cfg_.gpus.size()) +
            ") must be >= dcp_size (" + std::to_string(dcp_size_) + ")");
    }

    // Pre-allocate global_lse buffers (INV-DCP-11). Sized for the combine
    // head count: H_local (legacy) or dcp*H_local (sharded KV, INV-KVS-QAG).
    const size_t lse_bytes = static_cast<size_t>(cfg_.max_batch_size)
                           * combine_heads_ * sizeof(float);

    global_lse_buffers_.resize(dcp_size_);
    for (int r = 0; r < dcp_size_; ++r) {
        attention_devices_[r]->set_device();
        global_lse_buffers_[r] = static_cast<float*>(
            attention_devices_[r]->device_alloc(lse_bytes));
        // TD-91a: fail-fast on allocation failure rather than passing nullptr
        // to allgather_lse / correction kernel → GPU crash.
        if (!global_lse_buffers_[r]) {
            throw std::runtime_error(
                "DcpAttentionWrapper: device_alloc failed for global_lse rank " +
                std::to_string(r) + " (" + std::to_string(lse_bytes) + " bytes)");
        }
    }

    spdlog::debug("DcpAttentionWrapper: active dcp_size={} heads_local={} "
                  "combine_heads={} head_dim={} hidden={} max_batch={} "
                  "lse_buf={}B/gpu",
                  dcp_size_, cfg_.num_heads_local, combine_heads_,
                  cfg_.head_dim, cfg_.hidden_size, cfg_.max_batch_size,
                  lse_bytes);
}

DcpAttentionWrapper::~DcpAttentionWrapper() {
    for (int r = 0; r < static_cast<int>(global_lse_buffers_.size()); ++r) {
        if (global_lse_buffers_[r]) {
            attention_devices_[r]->set_device();
            attention_devices_[r]->device_free(global_lse_buffers_[r]);
        }
    }
    global_lse_buffers_.clear();
}

// -- correct_output ----------------------------------------------------------

void DcpAttentionWrapper::correct_output(
    void* const* partial_outputs,
    const float* const* partial_lses,
    int batch_size,
    void* const* streams) {

    if (!is_active()) return;

    // Head count of the combine (INV-KVS-QAG): combine_heads_ is dcp*H_local
    // under sharded KV (post-Q-allgather all-head partials), H_local legacy.

    // TD-PREFILL-NONDET diagnostic staging (env-gated, zero work off).
    static int seam_inv_counter = 0;  // single daemon thread
    const int inv = seam_inv_counter++;
    const size_t seam_out_bytes = static_cast<size_t>(batch_size)
                                * combine_heads_ * cfg_.head_dim * 2;  // BF16
    const size_t seam_lse_bytes = static_cast<size_t>(batch_size)
                                * combine_heads_ * sizeof(float);
    for (int r = 0; r < dcp_size_; ++r) {
        seam_exec_dump(0x20746150u /*'Pat '*/, inv, r, partial_outputs[r],
                       seam_out_bytes, streams[r]);
        seam_exec_dump(0x2069734cu /*'Lsi '*/, inv, r, partial_lses[r],
                       seam_lse_bytes, streams[r]);
    }

    // Step 10: allgather LSE (grouped: ncclGroupStart/End internal)
    comm_->allgather_lse(partial_lses, batch_size, streams, combine_heads_);

    // Step 11: per-rank DCP correction kernel
    for (int r = 0; r < dcp_size_; ++r) {
        attention_devices_[r]->set_device();
        correction_fn_(
            partial_outputs[r],
            comm_->gathered_lse_buffer(r),
            global_lse_buffers_[r],
            batch_size, combine_heads_, cfg_.head_dim,
            dcp_size_, r, streams[r]);
    }

    for (int r = 0; r < dcp_size_; ++r)
        seam_exec_dump(0x20726f43u /*'Cor '*/, inv, r, partial_outputs[r],
                       seam_out_bytes, streams[r]);

    // Step 12: allreduce corrected output (grouped)
    comm_->allreduce_output(partial_outputs, batch_size, streams,
                            combine_heads_);

    for (int r = 0; r < dcp_size_; ++r)
        seam_exec_dump(0x20626d43u /*'Cmb '*/, inv, r, partial_outputs[r],
                       seam_out_bytes, streams[r]);
}

// -- reduce_hidden -----------------------------------------------------------

void DcpAttentionWrapper::reduce_hidden(
    void* const* hiddens,
    int batch_size,
    void* const* streams) {

    if (!is_active()) return;

    // Step 14: allreduce o_proj hidden (TP reduction)
    comm_->allreduce_hidden(hiddens, batch_size, streams);
}

// -- queries -----------------------------------------------------------------

bool DcpAttentionWrapper::is_active() const {
    return dcp_size_ >= 2;
}

int DcpAttentionWrapper::dcp_size() const {
    return dcp_size_;
}

}  // namespace layerstorm::compute

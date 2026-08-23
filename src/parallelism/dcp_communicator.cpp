#include "parallelism/dcp_communicator.h"

#include <spdlog/spdlog.h>

#include "core/device_backend.h"

namespace layerstorm::parallelism {

// Datatype/op constants from collective_backend.h (SDK-free).
using parallelism::kCollFloat32;
using parallelism::kCollBfloat16;
using parallelism::kCollSum;

// -- DcpCommunicator ---------------------------------------------------------

DcpCommunicator::DcpCommunicator(Options opts) : opts_(std::move(opts)) {
    if (opts_.dcp_size < 1) {
        throw std::invalid_argument(
            "DcpCommunicator: dcp_size must be >= 1, got "
            + std::to_string(opts_.dcp_size));
    }

    num_heads_local_ = opts_.num_heads / std::max(opts_.dcp_size, 1);

    // Build cached gpus_ vector from device_backends
    gpus_.reserve(opts_.device_backends.size());
    for (auto* dev : opts_.device_backends)
        gpus_.push_back(dev->gpu());

    if (opts_.dcp_size == 1) {
        // Passthrough mode: no NCCL, no buffers.
        spdlog::debug("DcpCommunicator: dcp_size=1, passthrough mode");
        return;
    }

    // Validate device_backends
    if (static_cast<int>(opts_.device_backends.size()) != opts_.dcp_size) {
        throw std::invalid_argument(
            "DcpCommunicator: device_backends.size()="
            + std::to_string(opts_.device_backends.size())
            + " must equal dcp_size=" + std::to_string(opts_.dcp_size));
    }
    if (!opts_.collective) {
        throw std::invalid_argument(
            "DcpCommunicator: collective must be non-null for dcp_size >= 2");
    }

    // Create collective communicators (backend needs device IDs)
    std::vector<int> device_ids;
    device_ids.reserve(opts_.dcp_size);
    for (auto* dev : opts_.device_backends) device_ids.push_back(dev->gpu().id);
    comms_ = opts_.collective->create_comms(device_ids);

    // Pre-allocate gathered LSE buffers (INV-DCP-11). Sized for the WORST
    // case head count: the sharded-KV all-head combine (INV-KVS-QAG) gathers
    // [dcp, B, dcp*H_local] == [dcp, B, num_heads] — a superset of the legacy
    // per-rank [dcp, B, H_local]. Tiny either way (KB-scale).
    const size_t lse_bytes = static_cast<size_t>(opts_.dcp_size)
                           * opts_.max_batch_size
                           * opts_.num_heads
                           * sizeof(float);

    lse_buffers_.resize(opts_.dcp_size);
    for (int r = 0; r < opts_.dcp_size; ++r) {
        opts_.device_backends[r]->set_device();
        lse_buffers_[r] = static_cast<float*>(
            opts_.device_backends[r]->device_alloc(lse_bytes));
        if (!lse_buffers_[r]) {
            throw std::runtime_error(
                "DcpCommunicator: device_alloc failed for LSE buffer rank "
                + std::to_string(r) + " (" + std::to_string(lse_bytes) + " bytes)");
        }
    }

    spdlog::debug("DcpCommunicator: dcp_size={} devices=[{}{}] "
                  "heads_local={} lse_buf={}B/gpu",
                  opts_.dcp_size,
                  gpus_[0].id,
                  opts_.dcp_size > 1
                      ? "," + std::to_string(gpus_[1].id) : "",
                  num_heads_local_, lse_bytes);
}

DcpCommunicator::~DcpCommunicator() {
    // Free LSE buffers
    for (int r = 0; r < static_cast<int>(lse_buffers_.size()); ++r) {
        if (lse_buffers_[r]) {
            opts_.device_backends[r]->set_device();
            opts_.device_backends[r]->device_free(lse_buffers_[r]);
        }
    }
    lse_buffers_.clear();

    // Destroy collective comms. TD-100b: set each comm's device current before
    // ncclCommDestroy — destroying on the wrong active device leaves NCCL/CUDA
    // state half-torn-down, which poisons a second Engine created in the same
    // process (the "Re-sync after NCCL teardown" note in Engine::shutdown). The
    // LSE-free loop above already set_device's per rank; the comm loop must too.
    if (opts_.collective) {
        for (int r = 0; r < static_cast<int>(comms_.size()); ++r) {
            if (r < static_cast<int>(opts_.device_backends.size())
                && opts_.device_backends[r])
                opts_.device_backends[r]->set_device();
            opts_.collective->destroy_comm(comms_[r]);
        }
    }
    comms_.clear();
}

// -- validate_rank -----------------------------------------------------------

void DcpCommunicator::validate_rank(int rank) const {
    if (rank < 0 || rank >= opts_.dcp_size) {
        throw std::out_of_range(
            "DcpCommunicator: rank=" + std::to_string(rank)
            + " out of range [0, " + std::to_string(opts_.dcp_size) + ")");
    }
}

// -- comm --------------------------------------------------------------------

void* DcpCommunicator::comm(int rank) const {
    if (!is_active()) return nullptr;
    validate_rank(rank);
    return comms_[rank];
}

// -- gathered_lse_buffer -----------------------------------------------------

float* DcpCommunicator::gathered_lse_buffer(int rank) const {
    if (!is_active()) return nullptr;
    validate_rank(rank);
    return lse_buffers_[rank];
}

// -- allgather_lse -----------------------------------------------------------

void DcpCommunicator::allgather_lse(const float* const* local_lses,
                                      int batch_size,
                                      void* const* streams,
                                      int num_heads) {
    if (batch_size > opts_.max_batch_size) {
        throw std::invalid_argument(
            "DcpCommunicator::allgather_lse: batch_size="
            + std::to_string(batch_size) + " > max_batch_size="
            + std::to_string(opts_.max_batch_size));
    }
    const int H = num_heads > 0 ? num_heads : num_heads_local_;
    if (H > opts_.num_heads) {
        throw std::invalid_argument(
            "DcpCommunicator::allgather_lse: num_heads=" + std::to_string(H)
            + " > Options::num_heads=" + std::to_string(opts_.num_heads)
            + " (gathered LSE buffer bound)");
    }

    if (!is_active()) {
        // dcp_size=1: identity copy (local LSE -> gathered buffer)
        // Not possible with null backend (no device memcpy).
        // In production (cuda backend), use cudaMemcpyAsync.
        return;
    }

    const size_t sendcount = static_cast<size_t>(batch_size) * H;

    opts_.collective->group_begin();
    for (int r = 0; r < opts_.dcp_size; ++r) {
        opts_.device_backends[r]->set_device();
        opts_.collective->allgather(
            local_lses[r], lse_buffers_[r], sendcount,
            kCollFloat32, comms_[r], streams[r]);
    }
    opts_.collective->group_end();
}

// -- allreduce_output --------------------------------------------------------

void DcpCommunicator::allreduce_output(void* const* outputs,
                                         int batch_size,
                                         void* const* streams,
                                         int num_heads) {
    // TD-74u: validate before is_active check (match allgather_lse ordering).
    if (batch_size > opts_.max_batch_size) {
        throw std::invalid_argument(
            "DcpCommunicator::allreduce_output: batch_size="
            + std::to_string(batch_size) + " > max_batch_size="
            + std::to_string(opts_.max_batch_size));
    }
    const int H = num_heads > 0 ? num_heads : num_heads_local_;
    if (H > opts_.num_heads) {
        throw std::invalid_argument(
            "DcpCommunicator::allreduce_output: num_heads=" + std::to_string(H)
            + " > Options::num_heads=" + std::to_string(opts_.num_heads));
    }
    if (!is_active()) return;  // dcp_size=1: no-op

    const size_t count = static_cast<size_t>(batch_size)
                       * H * opts_.attn_output_dim;

    opts_.collective->group_begin();
    for (int r = 0; r < opts_.dcp_size; ++r) {
        opts_.device_backends[r]->set_device();
        opts_.collective->allreduce(
            outputs[r], outputs[r], count,
            kCollBfloat16, kCollSum, comms_[r], streams[r]);
    }
    opts_.collective->group_end();
}

// -- allgather_q ---------------------------------------------------------------
//
// INV-KVS-QAG (sharded KV): gather the absorbed query across DCP ranks in the
// HEAD dimension before attention, so every rank attends ALL dcp*H_local heads
// over its LOCAL token shard (vLLM mla_attention.py all_gather(dim=head)).
// NCCL allgather concatenates rank-major, and rank s holds global heads
// [s*H_local, (s+1)*H_local) — the concatenation IS the global head order.

void DcpCommunicator::allgather_q(const void* const* send_bufs,
                                    void* const* recv_bufs,
                                    int batch_size,
                                    size_t per_rank_elems,
                                    void* const* streams) {
    if (batch_size > opts_.max_batch_size) {
        throw std::invalid_argument(
            "DcpCommunicator::allgather_q: batch_size="
            + std::to_string(batch_size) + " > max_batch_size="
            + std::to_string(opts_.max_batch_size));
    }
    if (!is_active()) return;  // dcp_size=1: single rank already has all heads

    opts_.collective->group_begin();
    for (int r = 0; r < opts_.dcp_size; ++r) {
        opts_.device_backends[r]->set_device();
        opts_.collective->allgather(
            send_bufs[r], recv_bufs[r], per_rank_elems,
            kCollBfloat16, comms_[r], streams[r]);
    }
    opts_.collective->group_end();
}

// -- allreduce_hidden --------------------------------------------------------

void DcpCommunicator::allreduce_hidden(void* const* hiddens,
                                         int batch_size,
                                         void* const* streams,
                                         bool fp32,
                                         int rows_per_token) {
    // TD-74u: validate before is_active check (match allgather_lse ordering).
    // Guard applies to the TOKEN count; the per-slot combine expands by
    // rows_per_token (>1) only the element count, not the token batch.
    if (batch_size > opts_.max_batch_size) {
        throw std::invalid_argument(
            "DcpCommunicator::allreduce_hidden: batch_size="
            + std::to_string(batch_size) + " > max_batch_size="
            + std::to_string(opts_.max_batch_size));
    }
    if (!is_active()) return;  // dcp_size=1: no-op

    // Element count is identical for bf16 and fp32 (count = elements, not bytes).
    const size_t count = static_cast<size_t>(batch_size)
                       * static_cast<size_t>(rows_per_token) * opts_.hidden_size;
    const auto dtype = fp32 ? kCollFloat32 : kCollBfloat16;

    opts_.collective->group_begin();
    for (int r = 0; r < opts_.dcp_size; ++r) {
        opts_.device_backends[r]->set_device();
        opts_.collective->allreduce(
            hiddens[r], hiddens[r], count,
            dtype, kCollSum, comms_[r], streams[r]);
    }
    opts_.collective->group_end();
}

// -- allreduce_hidden_fused ---------------------------------------------------
// INV-NCCL-FUSE: shared-expert reduce + routed EP-combine reduce in ONE nccl
// group per rank -> one aggregated device launch instead of two tiny ones.
// BIT-IDENTICAL to two separate allreduce_hidden calls (same per-element rank
// sums; aggregation changes launch count only).

void DcpCommunicator::allreduce_hidden_fused(void* const* buffers_a,
                                             void* const* buffers_b,
                                             int batch_size,
                                             void* const* streams,
                                             bool b_fp32,
                                             int b_rows_per_token) {
    if (batch_size > opts_.max_batch_size) {
        throw std::invalid_argument(
            "DcpCommunicator::allreduce_hidden_fused: batch_size="
            + std::to_string(batch_size) + " > max_batch_size="
            + std::to_string(opts_.max_batch_size));
    }
    if (!is_active()) return;  // dcp_size=1: no-op

    const size_t count_a = static_cast<size_t>(batch_size) * opts_.hidden_size;
    const size_t count_b = static_cast<size_t>(batch_size)
                         * static_cast<size_t>(b_rows_per_token)
                         * opts_.hidden_size;
    const auto dtype_b = b_fp32 ? kCollFloat32 : kCollBfloat16;

    opts_.collective->group_begin();
    for (int r = 0; r < opts_.dcp_size; ++r) {
        opts_.device_backends[r]->set_device();
        opts_.collective->allreduce(
            buffers_a[r], buffers_a[r], count_a,
            kCollBfloat16, kCollSum, comms_[r], streams[r]);
        opts_.collective->allreduce(
            buffers_b[r], buffers_b[r], count_b,
            dtype_b, kCollSum, comms_[r], streams[r]);
    }
    opts_.collective->group_end();
}

// -- allgather_indexer_candidates ---------------------------------------------
//
// TD-GLM-INDEXER-LOCAL-MERGE: gather the per-rank shard-local top-k candidate
// lists (packed int32 indices + f32 scores) rank-major before the cross-rank
// exact merge. 32-bit words, bit-transparent (kCollFloat32 carries the int32
// half unchanged — allgather does no arithmetic).

void DcpCommunicator::allgather_indexer_candidates(
    const void* const* send_bufs, void* const* recv_bufs,
    size_t count_words, void* const* streams) {
    if (!is_active()) return;  // dcp_size=1: single rank already has all
    if (count_words == 0) return;

    opts_.collective->group_begin();
    for (int r = 0; r < opts_.dcp_size; ++r) {
        opts_.device_backends[r]->set_device();
        opts_.collective->allgather(
            send_bufs[r], recv_bufs[r], count_words,
            kCollFloat32, comms_[r], streams[r]);
    }
    opts_.collective->group_end();
}

// -- allgather_logits --------------------------------------------------------

void DcpCommunicator::allgather_logits(const float* const* send_bufs,
                                         float* const* recv_bufs,
                                         int batch_size, int local_vocab_size,
                                         void* const* streams) {
    // TD-74w: validate before is_active check (match allgather_lse ordering).
    if (batch_size > opts_.max_batch_size) {
        throw std::invalid_argument(
            "DcpCommunicator::allgather_logits: batch_size="
            + std::to_string(batch_size) + " > max_batch_size="
            + std::to_string(opts_.max_batch_size));
    }
    if (!is_active()) return;  // dcp_size=1: no-op

    const size_t sendcount =
        static_cast<size_t>(batch_size) * local_vocab_size;

    opts_.collective->group_begin();
    for (int r = 0; r < opts_.dcp_size; ++r) {
        opts_.device_backends[r]->set_device();
        opts_.collective->allgather(
            send_bufs[r], recv_bufs[r], sendcount,
            kCollFloat32, comms_[r], streams[r]);
    }
    opts_.collective->group_end();
}

}  // namespace layerstorm::parallelism

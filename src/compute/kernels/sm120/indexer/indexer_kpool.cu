// GF3.5: single-TU driver for the IndexPool kpool kernels (see
// indexer_kpool.h). Includes the vendored kernel .cu once and exposes
// throwing launch wrappers — the lightning_indexer.cu pattern. Existing
// lightning kernel TUs are untouched (byte-identity by construction; the
// pooled merge composes scatter → launch_lightning_topk → expand at the
// device-backend level instead of editing topk_merge.cu).

#include <stdexcept>
#include <string>

#include "sm120/indexer/kpool_compress.cu"

#include "compute/kernels/sm120/indexer/indexer_kpool.h"

namespace layerstorm::compute {

namespace {
void check(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " failed: "
                                 + cudaGetErrorString(err));
    }
}
void validate_pool_geometry(int kpool, int head_dim, int page_entries,
                            const char* what) {
    if (kpool < 2 || kpool > 8) {
        throw std::runtime_error(std::string(what)
            + ": kpool must be in 2..8, got " + std::to_string(kpool));
    }
    if (head_dim < 2 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0) {
        throw std::runtime_error(std::string(what)
            + ": head_dim must be a power of two in 2..1024, got "
            + std::to_string(head_dim));
    }
    if (page_entries <= 0) {
        throw std::runtime_error(std::string(what)
            + ": page_entries must be positive, got "
            + std::to_string(page_entries));
    }
}
}  // namespace

void launch_kpool_append_decode(const sm120::indexer::KpoolAppendDecodeParams& p,
                                cudaStream_t stream) {
    validate_pool_geometry(p.kpool, p.head_dim, p.page_entries,
                           "launch_kpool_append_decode");
    if (p.page_table) {
        // P-29 step 7 device-indexed arm: page + offset are resolved from
        // device state at execution time — pos_in_page is ignored; validate
        // the indirection inputs instead.
        if (!p.seqlen || p.page_tokens <= 0
            || p.page_tokens != p.page_entries * p.kpool) {
            throw std::runtime_error(
                "launch_kpool_append_decode: device-indexed arm needs seqlen "
                "and page_tokens == page_entries * kpool (got "
                + std::to_string(p.page_tokens) + " vs "
                + std::to_string(p.page_entries) + " x "
                + std::to_string(p.kpool) + ")");
        }
    } else if (p.pos_in_page < 0
               || p.pos_in_page / p.kpool >= p.page_entries) {
        throw std::runtime_error(
            "launch_kpool_append_decode: pos_in_page "
            + std::to_string(p.pos_in_page) + " outside page ("
            + std::to_string(p.page_entries) + " entries x kpool)");
    }
    sm120::indexer::run_kpool_append_decode(p, stream);
    check("launch_kpool_append_decode");
}

void launch_kpool_chunk_append(const sm120::indexer::KpoolChunkAppendParams& p,
                               cudaStream_t stream) {
    validate_pool_geometry(p.kpool, p.head_dim, p.page_entries,
                           "launch_kpool_chunk_append");
    // A mid-pool chunk START is legal: the kernel assembles the leading
    // pool from the page tail (SGLang n_from_tail shape) — the vLLM
    // reference silently DROPS such pools (GF35_KPOOL_REFERENCE §5). The
    // COVERAGE guard (not this launcher) enforces when a tail-continuing
    // start is semantically valid (frontier continuation or a blessed
    // rewind whose tail slots are intact).
    if (p.pos0_in_page < 0 || p.num_rows < 0
        || (p.pos0_in_page + p.num_rows + p.kpool - 1) / p.kpool
               > p.page_entries) {
        throw std::runtime_error(
            "launch_kpool_chunk_append: rows overflow page ("
            + std::to_string(p.num_rows) + " rows at pos0 "
            + std::to_string(p.pos0_in_page) + ")");
    }
    sm120::indexer::run_kpool_chunk_append(p, stream);
    check("launch_kpool_chunk_append");
}

void launch_kpool_expand(const sm120::indexer::KpoolExpandParams& p,
                         cudaStream_t stream) {
    if (p.kpool < 2 || p.out_cols > p.out_stride || p.num_rows < 0) {
        throw std::runtime_error("launch_kpool_expand: bad geometry");
    }
    sm120::indexer::run_kpool_expand(p, stream);
    check("launch_kpool_expand");
}

void launch_kpool_cand_scatter(const sm120::indexer::KpoolCandScatterParams& p,
                               cudaStream_t stream) {
    if (p.page_entries <= 0 || p.dcp_size < 1 || p.cand_count < 0
        || p.cand_count > p.cand_stride) {
        throw std::runtime_error("launch_kpool_cand_scatter: bad geometry");
    }
    sm120::indexer::run_kpool_cand_scatter(p, stream);
    check("launch_kpool_cand_scatter");
}

}  // namespace layerstorm::compute

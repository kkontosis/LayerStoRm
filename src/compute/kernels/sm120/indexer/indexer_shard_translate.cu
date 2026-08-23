// KVS-4: GLOBAL→LOCAL DSA sparse-index translation for sequence-sharded KV.
// See indexer_shard_translate.h for the contract and the shard math it must
// mirror (kv_shard_math.h / PageAllocator::dcp_rank_for_token, INV-4.9e).

#include "compute/kernels/sm120/indexer/indexer_shard_translate.h"

#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

constexpr int kWarp = 32;

// One warp per batch row: stream the (ascending, −1-padded) global top-k in
// 32-entry waves, ballot-compact the entries this rank owns, rewrite each as
// its LOCAL slot index. `base` (uniform across the warp) is the running
// output cursor — order is preserved, so the output stays sorted ascending.
__global__ void indexer_shard_translate_kernel(
    const int* __restrict__ global_indices,
    const int* __restrict__ global_lengths,
    int* __restrict__ local_indices,
    int* __restrict__ local_lengths,
    int topk, int chunk_tokens, int dcp_size, int rank)
{
    const int b    = blockIdx.x;
    const int lane = threadIdx.x;
    const int* gin = global_indices + static_cast<size_t>(b) * topk;
    int* lout      = local_indices + static_cast<size_t>(b) * topk;

    int glen = global_lengths[b];
    glen = glen < 0 ? 0 : (glen > topk ? topk : glen);

    const int cycle = chunk_tokens * dcp_size;
    int base = 0;
    for (int start = 0; start < glen; start += kWarp) {
        const int i = start + lane;
        const int g = (i < glen) ? __ldg(gin + i) : -1;
        const bool own =
            (g >= 0) && ((g / chunk_tokens) % dcp_size == rank);
        const unsigned mask = __ballot_sync(0xffffffffu, own);
        if (own) {
            const int off = __popc(mask & ((1u << lane) - 1u));
            lout[base + off] =
                (g / cycle) * chunk_tokens + (g % chunk_tokens);
        }
        base += __popc(mask);
    }
    for (int i = base + lane; i < topk; i += kWarp)
        lout[i] = -1;
    if (lane == 0) local_lengths[b] = base;
}

}  // namespace

void launch_indexer_shard_translate(const IndexerShardTranslateParams& p,
                                    cudaStream_t stream) {
    if (p.num_tokens <= 0) return;
    // Misconfiguration here would silently select WRONG KV rows — reject.
    if (p.topk <= 0 || p.chunk_tokens <= 0 || p.dcp_size < 2
        || p.rank < 0 || p.rank >= p.dcp_size) {
        throw std::runtime_error(
            "launch_indexer_shard_translate: invalid shard config (topk="
            + std::to_string(p.topk) + " chunk_tokens="
            + std::to_string(p.chunk_tokens) + " dcp_size="
            + std::to_string(p.dcp_size) + " rank="
            + std::to_string(p.rank) + ")");
    }
    if (!p.global_indices || !p.global_lengths || !p.local_indices
        || !p.local_lengths) {
        throw std::runtime_error(
            "launch_indexer_shard_translate: null buffer");
    }
    indexer_shard_translate_kernel<<<p.num_tokens, kWarp, 0, stream>>>(
        p.global_indices, p.global_lengths, p.local_indices, p.local_lengths,
        p.topk, p.chunk_tokens, p.dcp_size, p.rank);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_indexer_shard_translate failed: ")
            + cudaGetErrorString(err));
    }
}

}  // namespace layerstorm::compute

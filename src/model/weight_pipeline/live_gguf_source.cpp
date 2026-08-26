#include "model/weight_pipeline/live_gguf_source.h"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

#include "model/model_config.h"
#include "model/quantization/gguf_kquant.h"
#include "model/quantization/quant_interface.h"
#include "model/weight_loader/gguf_reader.h"
#include "model/weight_loader/tensor_id.h"
#include "model/weight_pipeline/prepacked_format.h"

#ifdef LAYERSTORM_HAS_URING
#include <liburing.h>
#endif

namespace layerstorm::model {

namespace {

constexpr int64_t kAlign = prepacked::kSlotAlignBytes;  // 4096

int64_t align_down(int64_t x) { return x / kAlign * kAlign; }
int64_t align_up(int64_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

/// Per-thread scratch: a 4096-aligned bounce buffer (O_DIRECT landing zone)
/// + a contiguous assembly buffer (gate|up|down at the layer's real sizes —
/// pack_gguf_expert then takes its zero-copy contiguity branch, so the
/// transform validates against the SAME code the offline prepacker runs) +
/// an optional per-thread io_uring for batched O_DIRECT reads.
struct ThreadScratch {
    std::byte* bounce = nullptr;
    size_t bounce_cap = 0;
    std::vector<std::byte> assembly;
#ifdef LAYERSTORM_HAS_URING
    ::io_uring ring{};
    bool ring_ok = false;
    bool ring_tried = false;
#endif

    std::byte* ensure_bounce(size_t cap) {
        if (bounce_cap >= cap) return bounce;
        std::free(bounce);
        // aligned_alloc requires size % alignment == 0.
        const size_t rounded = (cap + kAlign - 1) / kAlign * kAlign;
        bounce = static_cast<std::byte*>(
            std::aligned_alloc(static_cast<size_t>(kAlign), rounded));
        bounce_cap = bounce ? rounded : 0;
        return bounce;
    }

#ifdef LAYERSTORM_HAS_URING
    ::io_uring* ensure_ring() {
        if (!ring_tried) {
            ring_tried = true;
            ring_ok = io_uring_queue_init(8, &ring, 0) == 0;
        }
        return ring_ok ? &ring : nullptr;
    }
#endif

    ~ThreadScratch() {
        std::free(bounce);
#ifdef LAYERSTORM_HAS_URING
        if (ring_ok) io_uring_queue_exit(&ring);
#endif
    }
};

ThreadScratch& tls_scratch() {
    thread_local ThreadScratch s;
    return s;
}

/// Buffered exact read: pread [off, off+len) into dst. Loops on short reads.
bool pread_exact(int fd, int64_t off, int64_t len, std::byte* dst) {
    int64_t got = 0;
    while (got < len) {
        ssize_t r = ::pread(fd, dst + got, static_cast<size_t>(len - got),
                            static_cast<off_t>(off + got));
        if (r <= 0) return false;
        got += r;
    }
    return true;
}

}  // namespace

// ── Construction ────────────────────────────────────────────────────────────

LiveGgufExpertSource::LiveGgufExpertSource(const std::vector<GgufReader>& shards,
                                           const ModelConfig& model_cfg,
                                           const QuantInterface& quant,
                                           int n_routed_experts,
                                           bool o_direct) {
    const auto& raw = model_cfg.raw();
    shape_ = ExpertShape{raw.hidden_size, raw.moe_intermediate_size};
    n_experts_ = n_routed_experts;
    slot_size_bytes_raw_ = quant.bytes_per_expert(shape_);
    slot_stride_bytes_ = prepacked::aligned_slot_stride(slot_size_bytes_raw_);
    layers_.resize(static_cast<size_t>(raw.num_hidden_layers));

    // 1. Scan every shard's tensor directory for the stacked routed-expert
    // projections (the SAME name mapping + de-stack geometry the GGUF weight
    // loader uses — destack_expert_tensor).
    for (size_t si = 0; si < shards.size(); ++si) {
        const GgufReader& r = shards[si];
        shard_paths_.push_back(r.path());
        for (const auto& e : r.entries()) {
            auto id = parse_gguf_name(e.name);
            if (!id || id->owner != TensorOwner::routed_expert) continue;
            int proj = -1;
            switch (id->component) {
                case TensorComponent::gate_proj: proj = 0; break;
                case TensorComponent::up_proj:   proj = 1; break;
                case TensorComponent::down_proj: proj = 2; break;
                default: break;
            }
            if (proj < 0) continue;  // aux routed tensors (e.g. routing tables)
            if (e.dims.size() != 3) {
                throw std::runtime_error(
                    "live prepack: stacked expert tensor '" + e.name +
                    "' expected 3 dims {in,out,n_experts}, got " +
                    std::to_string(e.dims.size()));
            }
            if (!e.is_kquant()) {
                throw std::runtime_error(
                    "live prepack: routed expert tensor '" + e.name +
                    "' is not a supported GGUF k-quant (type " +
                    gguf_ggml_type_name(e.ggml_type) + ") — live prepack "
                    "covers GGUF k-quant experts only");
            }
            const int64_t in_features = e.dims[0];
            const int64_t out_features = e.dims[1];
            const int64_t n_exp = e.dims[2];
            if (n_exp != n_routed_experts) {
                throw std::runtime_error(
                    "live prepack: '" + e.name + "' has " +
                    std::to_string(n_exp) + " experts but config has " +
                    std::to_string(n_routed_experts));
            }
            const GgufKQuantType kt = e.kquant_type();
            const int64_t per_expert =
                gguf::gguf_packed_bytes(out_features, in_features, kt);
            if (per_expert * n_exp != static_cast<int64_t>(e.data_size_bytes)) {
                throw std::runtime_error(
                    "live prepack: '" + e.name + "' size " +
                    std::to_string(e.data_size_bytes) + " != per_expert " +
                    std::to_string(per_expert) + " x " + std::to_string(n_exp));
            }
            if (id->layer_idx < 0 ||
                id->layer_idx >= static_cast<int>(layers_.size())) {
                throw std::runtime_error(
                    "live prepack: '" + e.name + "' layer index out of range");
            }
            LayerInfo& li = layers_[static_cast<size_t>(id->layer_idx)];
            li.proj[static_cast<size_t>(proj)] = ProjSource{
                static_cast<int>(si),
                static_cast<int64_t>(r.data_blob_offset() + e.data_offset),
                per_expert, kt};
        }
    }

    // 2. Validate: every MoE layer must have a complete gate/up/down triple
    // whose packed total fits the (global-MAX) slot — the same GG-9/GG-10
    // checks prepack_experts runs.
    for (int layer_idx : model_cfg.moe_layer_indices()) {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(layers_.size())) {
            throw std::runtime_error(
                "live prepack: MoE layer index " + std::to_string(layer_idx) +
                " out of range");
        }
        LayerInfo& li = layers_[static_cast<size_t>(layer_idx)];
        for (int p = 0; p < 3; ++p) {
            if (li.proj[static_cast<size_t>(p)].shard_idx < 0) {
                throw std::runtime_error(
                    "live prepack: layer " + std::to_string(layer_idx) +
                    " is missing stacked routed-expert projection " +
                    std::to_string(p) + " (gate=0/up=1/down=2) in the GGUF");
            }
        }
        li.is_moe = true;
        li.types = GgufModelExpertTypes{li.proj[0].kquant, li.proj[1].kquant,
                                        li.proj[2].kquant};
        // Cross-check the de-stacked per-expert sizes against the type-derived
        // projection sizes pack_gguf_expert will enforce.
        const GgufQuantInterface lq =
            make_gguf_quant(li.types.gate, li.types.up, li.types.down);
        const int64_t gate_b = lq.bytes_per_projection(shape_, Projection::gate);
        const int64_t up_b   = lq.bytes_per_projection(shape_, Projection::up);
        const int64_t down_b = lq.bytes_per_projection(shape_, Projection::down);
        if (gate_b != li.proj[0].per_expert_bytes ||
            up_b != li.proj[1].per_expert_bytes ||
            down_b != li.proj[2].per_expert_bytes) {
            throw std::runtime_error(
                "live prepack: layer " + std::to_string(layer_idx) +
                " per-expert projection bytes disagree with the k-quant "
                "type-derived sizes (config shape vs GGUF dims mismatch?)");
        }
        li.packed_total = gate_b + up_b + down_b;
        if (li.packed_total <= 0 || li.packed_total > slot_size_bytes_raw_) {
            throw std::runtime_error(
                "live prepack: layer " + std::to_string(layer_idx) +
                " packed total " + std::to_string(li.packed_total) +
                " exceeds slot size " + std::to_string(slot_size_bytes_raw_) +
                " (global gguf_types must be the per-projection MAX)");
        }
    }

    // 3. Open the per-shard data fds (separate from the header mmaps).
    // O_DIRECT when requested; per-file fallback to buffered when the
    // filesystem refuses it (e.g. tmpfs).
    fds_.resize(shard_paths_.size());
    int n_direct = 0;
    for (size_t si = 0; si < shard_paths_.size(); ++si) {
        const std::string p = shard_paths_[si].string();
        int fd = -1;
        bool direct = false;
        if (o_direct) {
            fd = ::open(p.c_str(), O_RDONLY | O_DIRECT);
            direct = fd >= 0;
        }
        if (fd < 0) fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("live prepack: cannot open shard " + p);
        }
        const off_t sz = ::lseek(fd, 0, SEEK_END);
        fds_[si] = ShardFd{fd, direct, static_cast<int64_t>(sz)};
        if (direct) ++n_direct;
    }
    spdlog::info("LiveGgufExpertSource: {} shard(s), {} MoE layer(s), {} "
                 "experts, slot {} B (stride {} B), O_DIRECT on {}/{} shard(s)",
                 shard_paths_.size(), model_cfg.moe_layer_indices().size(),
                 n_experts_, slot_size_bytes_raw_, slot_stride_bytes_,
                 n_direct, shard_paths_.size());
}

LiveGgufExpertSource::~LiveGgufExpertSource() {
    for (auto& sf : fds_)
        if (sf.fd >= 0) ::close(sf.fd);
}

// ── Queries ─────────────────────────────────────────────────────────────────

bool LiveGgufExpertSource::has(memory::ExpertKey key) const {
    if (key.layer_idx >= layers_.size()) return false;
    if (key.expert_idx >= n_experts_) return false;
    return layers_[key.layer_idx].is_moe;
}

std::optional<GgufExpertTypes> LiveGgufExpertSource::gguf_types_for_layer(
        int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers_.size()))
        return std::nullopt;
    const LayerInfo& li = layers_[static_cast<size_t>(layer_idx)];
    if (!li.is_moe) return std::nullopt;
    return GgufExpertTypes{li.types.gate, li.types.up, li.types.down};
}

const std::array<LiveGgufExpertSource::ProjSource, 3>*
LiveGgufExpertSource::layer_sources(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers_.size()))
        return nullptr;
    const LayerInfo& li = layers_[static_cast<size_t>(layer_idx)];
    return li.is_moe ? &li.proj : nullptr;
}

int64_t LiveGgufExpertSource::layer_packed_bytes(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers_.size()))
        return 0;
    return layers_[static_cast<size_t>(layer_idx)].packed_total;
}

bool LiveGgufExpertSource::shard_is_o_direct(int shard_idx) const {
    return shard_idx >= 0 && shard_idx < static_cast<int>(fds_.size()) &&
           fds_[static_cast<size_t>(shard_idx)].o_direct;
}

int LiveGgufExpertSource::shard_fd(int shard_idx) const {
    if (shard_idx < 0 || shard_idx >= static_cast<int>(fds_.size())) return -1;
    return fds_[static_cast<size_t>(shard_idx)].fd;
}

int64_t LiveGgufExpertSource::shard_file_size(int shard_idx) const {
    if (shard_idx < 0 || shard_idx >= static_cast<int>(fds_.size())) return 0;
    return fds_[static_cast<size_t>(shard_idx)].file_size;
}

// ── load_into: read + transform + pad one slot ──────────────────────────────

bool LiveGgufExpertSource::load_into(memory::ExpertKey key, void* dst) const {
    if (!has(key)) return false;
    const LayerInfo& li = layers_[key.layer_idx];
    ThreadScratch& sc = tls_scratch();

    // Per-projection exact ranges + assembly offsets (gate|up|down
    // back-to-back at the layer's real sizes).
    struct Range {
        int fd;
        bool o_direct;
        int64_t off;    // exact file offset of this expert's block bytes
        int64_t len;    // exact byte length
        int64_t asm_off;
    };
    std::array<Range, 3> ranges;
    int64_t asm_off = 0;
    size_t bounce_need = 0;
    for (int p = 0; p < 3; ++p) {
        const ProjSource& ps = li.proj[static_cast<size_t>(p)];
        const ShardFd& sf = fds_[static_cast<size_t>(ps.shard_idx)];
        const int64_t off =
            ps.abs_offset + static_cast<int64_t>(key.expert_idx) *
                                ps.per_expert_bytes;
        ranges[static_cast<size_t>(p)] =
            Range{sf.fd, sf.o_direct, off, ps.per_expert_bytes, asm_off};
        asm_off += ps.per_expert_bytes;
        if (sf.o_direct) {
            bounce_need += static_cast<size_t>(
                align_up(off + ps.per_expert_bytes) - align_down(off));
        }
    }
    if (sc.assembly.size() < static_cast<size_t>(li.packed_total))
        sc.assembly.resize(static_cast<size_t>(li.packed_total));
    std::byte* assembly = sc.assembly.data();
    std::byte* bounce = nullptr;
    if (bounce_need > 0) {
        bounce = sc.ensure_bounce(bounce_need);
        if (!bounce) return false;  // aligned_alloc failed
    }

    // Aligned O_DIRECT sub-reads land in bounce partitions; buffered reads go
    // straight into the assembly. Each entry: which projection, aligned span.
    struct DirectRead {
        int idx;            // ranges[] index
        int64_t a_off;      // aligned file offset
        int64_t a_len;      // aligned read length (may cross EOF: short ok)
        std::byte* buf;     // 4096-aligned bounce partition
        bool done = false;
    };
    std::array<DirectRead, 3> dreads;
    int n_direct = 0;
    size_t bounce_cursor = 0;
    for (int p = 0; p < 3; ++p) {
        const Range& rg = ranges[static_cast<size_t>(p)];
        if (!rg.o_direct) continue;
        const int64_t a_off = align_down(rg.off);
        const int64_t a_len = align_up(rg.off + rg.len) - a_off;
        dreads[static_cast<size_t>(n_direct++)] =
            DirectRead{p, a_off, a_len, bounce + bounce_cursor};
        bounce_cursor += static_cast<size_t>(a_len);
    }

#ifdef LAYERSTORM_HAS_URING
    // io_uring batch: submit all O_DIRECT reads of this slot together (the
    // worker-pool fan-out provides the queue depth across slots).
    if (n_direct > 1) {
        if (::io_uring* ring = sc.ensure_ring()) {
            int submitted = 0;
            for (int i = 0; i < n_direct; ++i) {
                DirectRead& dr = dreads[static_cast<size_t>(i)];
                io_uring_sqe* sqe = io_uring_get_sqe(ring);
                if (!sqe) break;
                io_uring_prep_read(
                    sqe, ranges[static_cast<size_t>(dr.idx)].fd, dr.buf,
                    static_cast<unsigned>(dr.a_len),
                    static_cast<uint64_t>(dr.a_off));
                io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(i));
                ++submitted;
            }
            if (submitted > 0 &&
                io_uring_submit_and_wait(ring, static_cast<unsigned>(
                                                   submitted)) >= 0) {
                for (int c = 0; c < submitted; ++c) {
                    io_uring_cqe* cqe = nullptr;
                    if (io_uring_wait_cqe(ring, &cqe) != 0) break;
                    const auto i = static_cast<size_t>(
                        io_uring_cqe_get_data64(cqe));
                    const int res = cqe->res;
                    io_uring_cqe_seen(ring, cqe);
                    if (i < dreads.size()) {
                        DirectRead& dr = dreads[i];
                        const Range& rg = ranges[static_cast<size_t>(dr.idx)];
                        // Complete iff the read covered the exact slice
                        // (short reads at EOF may still cover it).
                        if (res > 0 &&
                            dr.a_off + res >= rg.off + rg.len) {
                            std::memcpy(assembly + rg.asm_off,
                                        dr.buf + (rg.off - dr.a_off),
                                        static_cast<size_t>(rg.len));
                            dr.done = true;
                        }
                    }
                }
            }
        }
    }
#endif

    // Fallback / remaining O_DIRECT reads: synchronous aligned pread loops.
    for (int i = 0; i < n_direct; ++i) {
        DirectRead& dr = dreads[static_cast<size_t>(i)];
        if (dr.done) continue;
        const Range& rg = ranges[static_cast<size_t>(dr.idx)];
        int64_t got = 0;
        const int64_t want = rg.off + rg.len - dr.a_off;  // cover the slice
        bool ok = true;
        while (got < want) {
            ssize_t r = ::pread(rg.fd, dr.buf + got,
                                static_cast<size_t>(dr.a_len - got),
                                static_cast<off_t>(dr.a_off + got));
            if (r <= 0) { ok = false; break; }
            got += r;
        }
        if (!ok) {
            spdlog::error("live prepack: O_DIRECT read failed for ({}, {}) "
                          "proj {}", key.layer_idx, key.expert_idx, dr.idx);
            return false;
        }
        std::memcpy(assembly + rg.asm_off, dr.buf + (rg.off - dr.a_off),
                    static_cast<size_t>(rg.len));
        dr.done = true;
    }

    // Buffered projections: pread straight into the assembly, then drop the
    // page cache for the range (Stage-2 semantics — the arena is the only
    // warm tier; the source cache must not compete with it for RAM).
    for (int p = 0; p < 3; ++p) {
        const Range& rg = ranges[static_cast<size_t>(p)];
        if (rg.o_direct) continue;
        if (!pread_exact(rg.fd, rg.off, rg.len, assembly + rg.asm_off)) {
            spdlog::error("live prepack: pread failed for ({}, {}) proj {}",
                          key.layer_idx, key.expert_idx, p);
            return false;
        }
        ::posix_fadvise(rg.fd, static_cast<off_t>(align_down(rg.off)),
                        static_cast<off_t>(align_up(rg.off + rg.len) -
                                           align_down(rg.off)),
                        POSIX_FADV_DONTNEED);
    }

    // Transform: the EXACT prepack-tool code path (pack_gguf_expert with the
    // strict per-layer type table, GG-10). The assembly is contiguous
    // gate|up|down, so the zero-copy branch fires and packed_slot spans it.
    std::vector<WeightBundle> bundles(3);
    static constexpr std::array<TensorComponent, 3> kComp{
        TensorComponent::gate_proj, TensorComponent::up_proj,
        TensorComponent::down_proj};
    for (int p = 0; p < 3; ++p) {
        const ProjSource& ps = li.proj[static_cast<size_t>(p)];
        WeightBundle& b = bundles[static_cast<size_t>(p)];
        b.id = TensorId{kComp[static_cast<size_t>(p)], TensorRole::weight,
                        TensorOwner::routed_expert,
                        static_cast<int>(key.layer_idx),
                        static_cast<int>(key.expert_idx)};
        b.weight.data = std::span<const std::byte>(
            assembly + ranges[static_cast<size_t>(p)].asm_off,
            static_cast<size_t>(ps.per_expert_bytes));
        b.weight.dtype = SafetensorsDtype::U8;
        b.weight.gguf_type = ps.kquant;
        const bool is_down = p == 2;
        b.weight.shape = {is_down ? shape_.hidden_size
                                  : shape_.intermediate_size,
                          is_down ? shape_.intermediate_size
                                  : shape_.hidden_size};
    }
    if (!pack_gguf_expert(bundles, shape_, &li.types)) {
        spdlog::error("live prepack: pack_gguf_expert failed for ({}, {})",
                      key.layer_idx, key.expert_idx);
        return false;
    }
    const auto& slot = bundles[0].packed_slot;
    if (static_cast<int64_t>(slot.size()) != li.packed_total) {
        spdlog::error("live prepack: packed size {} != expected {} for "
                      "({}, {})", slot.size(), li.packed_total,
                      key.layer_idx, key.expert_idx);
        return false;
    }

    // Commit: content + zero tail to the aligned stride — byte-identical to
    // the prepacked on-disk slot (content + pad, INV-LIVE-PREPACK-IDENTITY).
    std::memcpy(dst, slot.data(), slot.size());
    std::memset(static_cast<std::byte*>(dst) + slot.size(), 0,
                static_cast<size_t>(slot_stride_bytes_) - slot.size());
    return true;
}

}  // namespace layerstorm::model

// ArenaCache (P-24b) — persistence meta layer for the pinned expert arena.

#include "core/memory/arena_cache.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>

#include <spdlog/spdlog.h>

#include "model/weight_pipeline/prepacked_format.h"

namespace layerstorm::memory {

namespace {

// FNV-1a 64-bit — deterministic, dependency-free. The hash is an index only
// (full-tuple verify on every hit), so FNV's quality is ample here.
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

uint64_t fnv1a(const void* data, size_t len, uint64_t h = kFnvOffset) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h;
}

uint64_t fnv1a_u64(uint64_t v, uint64_t h) { return fnv1a(&v, sizeof(v), h); }

uint64_t packed_key(ExpertKey key) {
    return (static_cast<uint64_t>(key.layer_idx) << 16) | key.expert_idx;
}

}  // namespace

// ── Static helpers ───────────────────────────────────────────────────────────

size_t ArenaCache::required_bytes(const std::vector<ArenaCacheNodeGeom>& geom) {
    size_t records = 0;
    for (const auto& g : geom) records += g.num_slots;
    return sizeof(Header) + geom.size() * sizeof(NodeSpan) +
           records * sizeof(SlotRecord);
}

namespace {
constexpr size_t kEmaAlign = 64;
size_t align_ema(size_t v) { return (v + kEmaAlign - 1) & ~(kEmaAlign - 1); }
}  // namespace

size_t ArenaCache::ema_trailer_bytes(uint32_t num_layers,
                                     uint32_t num_experts) {
    return align_ema(sizeof(EmaTrailer)) +
           static_cast<size_t>(num_layers) * num_experts * sizeof(float);
}

size_t ArenaCache::required_bytes_with_ema(
        const std::vector<ArenaCacheNodeGeom>& geom, uint32_t num_layers,
        uint32_t num_experts) {
    return align_ema(required_bytes(geom)) +
           ema_trailer_bytes(num_layers, num_experts);
}

uint64_t ArenaCache::hash_geometry(size_t slot_size_bytes, size_t scratch_bytes,
                                   const std::vector<ArenaCacheNodeGeom>& geom) {
    uint64_t h = kFnvOffset;
    h = fnv1a_u64(slot_size_bytes, h);
    h = fnv1a_u64(scratch_bytes, h);
    for (const auto& g : geom) {
        h = fnv1a_u64(static_cast<uint64_t>(static_cast<int64_t>(g.node)), h);
        h = fnv1a_u64(g.num_slots, h);
    }
    return h;
}

uint64_t ArenaCache::hash_config(
        size_t slot_size_bytes, size_t scratch_bytes,
        const std::vector<NodeIdentity>& nodes,
        const config::PinHostExpertPoolSizingConfig& sizing,
        const config::CrossNodeSpillConfig& spill,
        uint64_t placement_id) {
    uint64_t h = kFnvOffset;
    h = fnv1a_u64(slot_size_bytes, h);
    h = fnv1a_u64(scratch_bytes, h);
    for (const auto& n : nodes) {
        h = fnv1a_u64(static_cast<uint64_t>(static_cast<int64_t>(n.node)), h);
        h = fnv1a_u64(static_cast<uint64_t>(n.share_degree), h);
    }
    // Sizing knobs: any change to how the arena WOULD be sized is a config
    // change (wipe + cold rebuild at the new sizing). The RESOLVED slot
    // counts are excluded on purpose (free-RAM-volatile).
    auto hash_mode_value = [&](config::HostPoolSizingMode m, double v) {
        h = fnv1a_u64(static_cast<uint64_t>(m), h);
        h = fnv1a(&v, sizeof(v), h);
    };
    hash_mode_value(sizing.mode, sizing.value);
    for (const auto& pn : sizing.per_node) {
        h = fnv1a_u64(static_cast<uint64_t>(static_cast<int64_t>(pn.node)), h);
        hash_mode_value(pn.mode, pn.value);
    }
    h = fnv1a_u64(spill.enabled ? 1 : 0, h);
    hash_mode_value(spill.sizing_mode, spill.sizing_value);
    for (const auto& sn : spill.nodes) {
        h = fnv1a_u64(static_cast<uint64_t>(static_cast<int64_t>(sn.node)), h);
        h = fnv1a_u64(static_cast<uint64_t>(sn.weight), h);
    }
    for (const auto& pn : spill.per_node) {
        h = fnv1a_u64(static_cast<uint64_t>(static_cast<int64_t>(pn.node)), h);
        hash_mode_value(pn.mode, pn.value);
    }
    // Arena host placement policy (arena_placement.h): folded ONLY when
    // active so pre-placement stores keep their exact historical hash.
    if (placement_id != 0) {
        static constexpr char kPlaceTag[] = "placement";
        h = fnv1a(kPlaceTag, sizeof(kPlaceTag) - 1, h);
        h = fnv1a_u64(placement_id, h);
    }
    return h;
}

uint64_t ArenaCache::hash_source_id(const std::string& prepacked_dir_abs,
                                    const std::string& manifest_format_version,
                                    int64_t slot_stride_bytes) {
    uint64_t h = fnv1a(prepacked_dir_abs.data(), prepacked_dir_abs.size());
    h = fnv1a(manifest_format_version.data(), manifest_format_version.size(), h);
    h = fnv1a_u64(static_cast<uint64_t>(slot_stride_bytes), h);
    h = fnv1a_u64(kSchemaVersion, h);
    return h;
}

std::vector<ExpertFileIdentity> ArenaCache::stat_expert_files(
        const std::string& prepacked_dir, uint32_t num_experts) {
    std::vector<ExpertFileIdentity> out(num_experts);
    for (uint32_t e = 0; e < num_experts; ++e) {
        const auto path = model::prepacked::expert_file_path(
            prepacked_dir, static_cast<int>(e));
        struct stat st{};
        if (::stat(path.c_str(), &st) == 0 && st.st_size > 0) {
            out[e].mtime_ns =
                static_cast<uint64_t>(st.st_mtim.tv_sec) * 1000000000ULL +
                static_cast<uint64_t>(st.st_mtim.tv_nsec);
            out[e].size_bytes = static_cast<uint64_t>(st.st_size);
        }
        // else: invalid identity — records for this expert never match/adopt.
    }
    return out;
}

// ── ArenaMetaSegment ─────────────────────────────────────────────────────────

ArenaMetaSegment ArenaMetaSegment::create(size_t bytes) {
    ArenaMetaSegment seg;
    if (bytes == 0) return seg;
    const size_t page = 4096;
    const size_t aligned = (bytes + page - 1) & ~(page - 1);
    const int fd = ::memfd_create("ls-arena-meta", MFD_CLOEXEC);
    if (fd < 0) {
        spdlog::error("ArenaMetaSegment: memfd_create failed: {}",
                      std::strerror(errno));
        return seg;
    }
    if (::ftruncate(fd, static_cast<off_t>(aligned)) != 0) {
        spdlog::error("ArenaMetaSegment: ftruncate({}) failed: {}", aligned,
                      std::strerror(errno));
        ::close(fd);
        return seg;
    }
    void* p = ::mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, 0);
    if (p == MAP_FAILED) {
        spdlog::error("ArenaMetaSegment: mmap failed: {}", std::strerror(errno));
        ::close(fd);
        return seg;
    }
    seg.base_ = p;
    seg.bytes_ = aligned;
    seg.fd_ = fd;
    return seg;
}

ArenaMetaSegment ArenaMetaSegment::adopt(int fd) {
    ArenaMetaSegment seg;
    if (fd < 0) return seg;
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        spdlog::warn("ArenaMetaSegment: adopt fstat failed: {}",
                     std::strerror(errno));
        ::close(fd);
        return seg;
    }
    void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size),
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        spdlog::warn("ArenaMetaSegment: adopt mmap failed: {}",
                     std::strerror(errno));
        ::close(fd);
        return seg;
    }
    seg.base_ = p;
    seg.bytes_ = static_cast<size_t>(st.st_size);
    seg.fd_ = fd;
    return seg;
}

bool ArenaMetaSegment::ensure_bytes(size_t min_bytes) {
    if (!valid()) return false;
    if (bytes_ >= min_bytes) return true;
    const size_t page = 4096;
    const size_t aligned = (min_bytes + page - 1) & ~(page - 1);
    if (::ftruncate(fd_, static_cast<off_t>(aligned)) != 0) {
        spdlog::warn("ArenaMetaSegment: grow ftruncate({}) failed: {}",
                     aligned, std::strerror(errno));
        return false;
    }
    // Map the NEW size first; only release the old mapping on success (a
    // failure must leave the segment usable at its old size).
    void* p = ::mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd_, 0);
    if (p == MAP_FAILED) {
        spdlog::warn("ArenaMetaSegment: grow mmap failed: {}",
                     std::strerror(errno));
        return false;
    }
    ::munmap(base_, bytes_);
    base_ = p;
    bytes_ = aligned;
    return true;
}

ArenaMetaSegment::~ArenaMetaSegment() {
    if (base_) ::munmap(base_, bytes_);
    if (fd_ >= 0) ::close(fd_);
}

ArenaMetaSegment::ArenaMetaSegment(ArenaMetaSegment&& o) noexcept
    : base_(o.base_), bytes_(o.bytes_), fd_(o.fd_) {
    o.base_ = nullptr;
    o.bytes_ = 0;
    o.fd_ = -1;
}

ArenaMetaSegment& ArenaMetaSegment::operator=(ArenaMetaSegment&& o) noexcept {
    if (this != &o) {
        if (base_) ::munmap(base_, bytes_);
        if (fd_ >= 0) ::close(fd_);
        base_ = o.base_;
        bytes_ = o.bytes_;
        fd_ = o.fd_;
        o.base_ = nullptr;
        o.bytes_ = 0;
        o.fd_ = -1;
    }
    return *this;
}

// ── Construction / layout ────────────────────────────────────────────────────

ArenaCache::ArenaCache(void* base, size_t bytes) : base_(base), bytes_(bytes) {}

bool ArenaCache::format(uint64_t geometry_hash, uint64_t source_id,
                        const std::vector<ArenaCacheNodeGeom>& geom) {
    if (!base_ || bytes_ < required_bytes(geom)) return false;
    std::memset(base_, 0, required_bytes(geom));
    header_ = static_cast<Header*>(base_);
    header_->magic = kMagicExpected;
    header_->schema_version = kSchemaVersion;
    header_->num_nodes = static_cast<uint32_t>(geom.size());
    header_->geometry_hash = geometry_hash;
    header_->source_id = source_id;
    spans_ = reinterpret_cast<NodeSpan*>(header_ + 1);
    uint64_t first = 0;
    for (size_t i = 0; i < geom.size(); ++i) {
        spans_[i].node = geom[i].node;
        spans_[i].first_record = first;
        spans_[i].num_slots = geom[i].num_slots;
        first += geom[i].num_slots;
    }
    header_->record_count = first;
    records_ = reinterpret_cast<SlotRecord*>(spans_ + geom.size());
    span_by_node_.clear();
    for (size_t i = 0; i < geom.size(); ++i)
        span_by_node_[spans_[i].node] = &spans_[i];
    index_.clear();
    return true;
}

bool ArenaCache::open_stored(uint64_t geometry_hash, uint64_t source_id) {
    if (!base_ || bytes_ < sizeof(Header)) return false;
    header_ = static_cast<Header*>(base_);
    if (header_->magic != kMagicExpected ||
        header_->schema_version != kSchemaVersion) {
        spdlog::info("ArenaCache: meta magic/schema mismatch — wiping");
        return false;
    }
    if (header_->geometry_hash != geometry_hash) {
        spdlog::info("ArenaCache: arena config changed (hash {:x} != {:x}) — "
                     "wiping", header_->geometry_hash, geometry_hash);
        return false;
    }
    if (header_->source_id != source_id) {
        spdlog::info("ArenaCache: source changed (id {:x} != {:x}) — wiping",
                     header_->source_id, source_id);
        return false;
    }
    // Span/record SELF-consistency: contiguous spans, counts add up, and the
    // whole table fits the mapped segment. The stored per-node slot counts are
    // AUTHORITATIVE (they may differ from what today's free RAM would plan —
    // adoption allocates nothing, so the stored geometry always fits).
    const uint32_t n = header_->num_nodes;
    if (n == 0 ||
        bytes_ < sizeof(Header) + n * sizeof(NodeSpan)) {
        spdlog::info("ArenaCache: bad node count — wiping");
        return false;
    }
    spans_ = reinterpret_cast<NodeSpan*>(header_ + 1);
    uint64_t first = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (spans_[i].node < 0 || spans_[i].first_record != first ||
            spans_[i].num_slots == 0) {
            spdlog::info("ArenaCache: inconsistent node spans — wiping");
            return false;
        }
        first += spans_[i].num_slots;
    }
    if (header_->record_count != first ||
        bytes_ < sizeof(Header) + n * sizeof(NodeSpan) +
                     first * sizeof(SlotRecord)) {
        spdlog::info("ArenaCache: record count / size mismatch — wiping");
        return false;
    }
    records_ = reinterpret_cast<SlotRecord*>(spans_ + n);
    span_by_node_.clear();
    for (uint32_t i = 0; i < n; ++i) {
        if (span_by_node_.count(spans_[i].node)) {
            spdlog::info("ArenaCache: duplicate node span — wiping");
            span_by_node_.clear();
            return false;
        }
        span_by_node_[spans_[i].node] = &spans_[i];
    }
    index_.clear();  // rebuilt by scan_adoptable
    return true;
}

std::vector<ArenaCacheNodeGeom> ArenaCache::stored_geometry() const {
    std::vector<ArenaCacheNodeGeom> out;
    if (!header_ || !spans_) return out;
    out.reserve(header_->num_nodes);
    for (uint32_t i = 0; i < header_->num_nodes; ++i)
        out.push_back({spans_[i].node, spans_[i].num_slots});
    return out;
}

bool ArenaCache::validate(uint64_t geometry_hash, uint64_t source_id,
                          const std::vector<ArenaCacheNodeGeom>& geom) {
    if (!open_stored(geometry_hash, source_id)) return false;
    // Exact-geometry comparison on top of open_stored (tests + callers that
    // know the expected plan).
    const auto stored = stored_geometry();
    if (stored.size() != geom.size()) {
        spdlog::info("ArenaCache: node count mismatch — wiping");
        return false;
    }
    for (size_t i = 0; i < geom.size(); ++i) {
        if (stored[i].node != geom[i].node ||
            stored[i].num_slots != geom[i].num_slots) {
            spdlog::info("ArenaCache: node span mismatch — wiping");
            return false;
        }
    }
    return true;
}

void ArenaCache::set_file_identities(std::vector<ExpertFileIdentity> idents) {
    file_ident_ = std::move(idents);
}

// ── Record addressing ────────────────────────────────────────────────────────

ArenaCache::SlotRecord* ArenaCache::record(int node, size_t slot) {
    auto it = span_by_node_.find(node);
    if (it == span_by_node_.end() || slot >= it->second->num_slots)
        return nullptr;
    return &records_[it->second->first_record + slot];
}

const ArenaCache::SlotRecord* ArenaCache::record(int node, size_t slot) const {
    auto it = span_by_node_.find(node);
    if (it == span_by_node_.end() || slot >= it->second->num_slots)
        return nullptr;
    return &records_[it->second->first_record + slot];
}

uint64_t ArenaCache::key_hash(ExpertKey key) const {
    const ExpertFileIdentity ident =
        key.expert_idx < file_ident_.size() ? file_ident_[key.expert_idx]
                                            : ExpertFileIdentity{};
    uint64_t h = kFnvOffset;
    h = fnv1a_u64(header_ ? header_->source_id : 0, h);
    h = fnv1a_u64(ident.mtime_ns, h);
    h = fnv1a_u64(ident.size_bytes, h);
    h = fnv1a_u64(packed_key(key), h);
    return h;
}

bool ArenaCache::identity_matches(const SlotRecord& r) const {
    if (r.expert_idx >= file_ident_.size()) return false;
    const ExpertFileIdentity& cur = file_ident_[r.expert_idx];
    return cur.valid() && cur.mtime_ns == r.file_mtime_ns &&
           cur.size_bytes == r.file_size;
}

void ArenaCache::index_erase(ExpertKey key, int node, size_t slot) {
    auto it = index_.find(packed_key(key));
    if (it == index_.end()) return;
    auto sp = span_by_node_.find(node);
    if (sp != span_by_node_.end() &&
        it->second == sp->second->first_record + slot)
        index_.erase(it);
}

// ── Record ops (INV-ARENA-CACHE-ORDER) ───────────────────────────────────────

void ArenaCache::on_reuse(int node, size_t slot) {
    SlotRecord* r = record(node, slot);
    if (!r) return;
    if (r->state != kEmpty) {
        index_erase(ExpertKey{r->layer_idx, r->expert_idx}, node, slot);
        // Order: state first (kills the record for any crash-time reader),
        // then a release fence before the caller writes slot bytes.
        r->state = kEmpty;
        std::atomic_thread_fence(std::memory_order_release);
    }
}

void ArenaCache::on_load_start(int node, size_t slot, ExpertKey key) {
    SlotRecord* r = record(node, slot);
    if (!r) return;
    const ExpertFileIdentity ident =
        key.expert_idx < file_ident_.size() ? file_ident_[key.expert_idx]
                                            : ExpertFileIdentity{};
    r->state = kLoading;   // not adoptable until kActive
    r->layer_idx = key.layer_idx;
    r->expert_idx = key.expert_idx;
    r->file_mtime_ns = ident.mtime_ns;
    r->file_size = ident.size_bytes;
    r->key_hash = key_hash(key);
}

void ArenaCache::on_load_ready(int node, size_t slot, ExpertKey key) {
    SlotRecord* r = record(node, slot);
    if (!r) return;
    const ExpertFileIdentity ident =
        key.expert_idx < file_ident_.size() ? file_ident_[key.expert_idx]
                                            : ExpertFileIdentity{};
    r->layer_idx = key.layer_idx;
    r->expert_idx = key.expert_idx;
    r->file_mtime_ns = ident.mtime_ns;
    r->file_size = ident.size_bytes;
    r->key_hash = key_hash(key);
    // The slot's bytes are complete before this call; fence so the record
    // fields land before the state flips ACTIVE (kill-safe publish order).
    std::atomic_thread_fence(std::memory_order_release);
    r->state = kActive;
    auto sp = span_by_node_.find(node);
    if (sp != span_by_node_.end())
        index_[packed_key(key)] = sp->second->first_record + slot;
}

// ── Queries ──────────────────────────────────────────────────────────────────

std::optional<ArenaCache::Hit> ArenaCache::lookup(ExpertKey key) const {
    auto it = index_.find(packed_key(key));
    if (it == index_.end()) return std::nullopt;
    const uint64_t flat = it->second;
    for (const auto& [node, span] : span_by_node_) {
        if (flat < span->first_record ||
            flat >= span->first_record + span->num_slots)
            continue;
        const SlotRecord& r = records_[flat];
        // Full-tuple verify — the hash/index is never trusted alone.
        if (r.state == kActive && r.layer_idx == key.layer_idx &&
            r.expert_idx == key.expert_idx && identity_matches(r) &&
            r.key_hash == key_hash(key))
            return Hit{node, static_cast<size_t>(flat - span->first_record)};
        return std::nullopt;
    }
    return std::nullopt;
}

size_t ArenaCache::scan_adoptable(
        uint32_t num_layers, uint32_t num_experts,
        const std::function<void(int, size_t, ExpertKey)>& fn) {
    if (!header_ || !records_) return 0;
    size_t adoptable = 0, stale = 0, interrupted = 0;
    index_.clear();
    for (uint32_t s = 0; s < header_->num_nodes; ++s) {
        const NodeSpan& span = spans_[s];
        for (uint64_t i = 0; i < span.num_slots; ++i) {
            SlotRecord& r = records_[span.first_record + i];
            if (r.state == kEmpty) continue;
            const ExpertKey key{r.layer_idx, r.expert_idx};
            const bool in_range =
                r.layer_idx < num_layers && r.expert_idx < num_experts;
            if (r.state == kActive && in_range && identity_matches(r) &&
                r.key_hash == key_hash(key) &&
                index_.find(packed_key(key)) == index_.end()) {
                index_[packed_key(key)] = span.first_record + i;
                if (fn) fn(span.node, static_cast<size_t>(i), key);
                ++adoptable;
            } else {
                (r.state == kLoading ? interrupted : stale)++;
                r.state = kEmpty;  // stale identity / dupe / interrupted load
            }
        }
    }
    std::atomic_thread_fence(std::memory_order_release);
    spdlog::info("ArenaCache: scan — {} adoptable, {} stale, {} interrupted "
                 "(of {} records)", adoptable, stale, interrupted,
                 header_->record_count);
    return adoptable;
}

// ── EMA persistence trailer (TD-ARENA-MIGRATE-EMA-PERSIST) ──────────────────

ArenaCache::EmaTrailer* ArenaCache::ema_trailer(uint32_t num_layers,
                                                uint32_t num_experts) const {
    if (!base_ || !header_ || !spans_) return nullptr;
    const std::vector<ArenaCacheNodeGeom> geom = stored_geometry();
    const size_t off = align_ema(required_bytes(geom));
    if (bytes_ < off + ema_trailer_bytes(num_layers, num_experts))
        return nullptr;  // pre-trailer store: no room — never a wipe
    return reinterpret_cast<EmaTrailer*>(static_cast<char*>(base_) + off);
}

std::optional<ArenaCache::EmaView> ArenaCache::ema_load(
        uint32_t num_layers, uint32_t num_experts) const {
    const EmaTrailer* t = ema_trailer(num_layers, num_experts);
    if (!t || t->magic != kEmaMagic || t->version != kEmaVersion ||
        t->state != 1 || t->num_layers != num_layers ||
        t->num_experts != num_experts)
        return std::nullopt;  // absent/interrupted/mismatched → start cold
    EmaView v;
    v.data = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(t) + align_ema(sizeof(EmaTrailer)));
    v.num_layers = t->num_layers;
    v.num_experts = t->num_experts;
    v.fetches_seen = t->fetches_seen;
    v.saved_unix_ns = t->saved_unix_ns;
    return v;
}

bool ArenaCache::ema_save(const float* data, uint32_t num_layers,
                          uint32_t num_experts, double fetches_seen) {
    EmaTrailer* t = ema_trailer(num_layers, num_experts);
    if (!t || !data) return false;
    // INV-ARENA-CACHE-ORDER discipline: invalidate BEFORE the first payload
    // byte; re-arm only AFTER the payload fully landed.
    t->state = 0;
    std::atomic_thread_fence(std::memory_order_release);
    t->magic = kEmaMagic;
    t->version = kEmaVersion;
    t->num_layers = num_layers;
    t->num_experts = num_experts;
    t->saved_unix_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    t->fetches_seen = fetches_seen;
    t->reserved = 0;
    std::memcpy(reinterpret_cast<char*>(t) + align_ema(sizeof(EmaTrailer)),
                data,
                static_cast<size_t>(num_layers) * num_experts * sizeof(float));
    std::atomic_thread_fence(std::memory_order_release);
    t->state = 1;
    return true;
}

}  // namespace layerstorm::memory

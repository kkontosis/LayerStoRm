// reef_sim — offline REEF bank-seam replay (TD-BRIDGE-CPP-GAP Q1 sim).
//
// Links the REAL ReefOrch decision stack (core/gpu_loader/reef_orch.{h,cpp} —
// the exact solver + board + bank-input assembly BOTH production consumers
// drive) and replays a RECORDED engine trace through it under one of four
// bank-input policies, so the Q1 default-flip decision is made on the real
// engine's decision behavior without GPU experimentation:
//
//   live-raw     per-solve live location_node RAW nodes (the daemon service's
//                pre-flip default — must BIT-REPRODUCE the recorded stream:
//                sim validation).
//   live-paired  per-solve live nodes with HBM→CPU-affinity pairing.
//   frozen       construction-time paired snapshot, never refreshed (the
//                measured 10.15-10.56 LS_REEF_BANK_SNAPSHOT winner).
//   epoch        paired snapshot refreshed ONLY at online-migrator COMMIT
//                boundaries (the proposed engine paradigm, INV-REEF-BANK).
//
// TRACE (from ONE instrumented engine run): the LS_REEF_DECISION_DUMP file
// with LS_REEF_RELOC_TRACE=1 armed —
//   H <tp> <cap0> ... <capN-1>              solver caps header
//   R <layer> <n> | e:assign:bank:pinned…   one line per solve (union in
//                                           i-order; assign/bank recorded)
//   A <layer> | vl:ve:g…                    apply line (chosen victims)
//   M <solve> <kind> <layer> <expert> <old> <new>
//                                           arena location change after the
//                                           <solve>-th solve; kind 'C' =
//                                           migrator commit flip, 'R'/'E' =
//                                           reserve/evict.
// MAP: the boot-time LS_ARENA_MAP_DUMP CSV ("e,layer,expert,paired,raw" +
// "g,gpu,node" rows) — initial locations + the raw→paired node table.
//
// Usage:
//   reef_sim --trace <dump> --map <csv> --calib <calibration.json> \
//            --policy live-raw|live-paired|frozen|epoch --out <decision-dump>
// Policy env (LS_LOADER_POLICY / overrides / LS_LOADER_KEEPPRED_W) is read by
// make_reef_orch exactly as in the engine — run under the same environment as
// the recorded run. Emits the replayed decision stream (R/A lines, identical
// format) to --out and a summary (solves, entries, predicted misses, epoch
// refreshes) to stdout.
//
// CUDA-free (INV-GPU-1): pure CPU replay.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/gpu_loader/reef_orch.h"
#include "core/memory/eviction_policy.h"  // ExpertKey

namespace gl = layerstorm::gpu_loader;
using layerstorm::memory::ExpertKey;

namespace {

uint32_t kid(uint32_t layer, uint16_t expert) {
    return (layer << 16) | expert;
}

struct RelocEvent {
    uint64_t solve;  // applies AFTER this many solves completed
    char kind;       // 'C' commit, 'R' reserve, 'E' evict
    uint32_t layer;
    uint16_t expert;
    int old_node;
    int new_node;
};

struct Solve {
    int layer;
    std::vector<uint16_t> topk;  // i-order union (as recorded)
    std::vector<uint8_t> rec_assign;
    std::vector<int> rec_bank;
};

struct Trace {
    int tp = 0;
    std::vector<int> caps;
    std::vector<Solve> solves;
    std::vector<RelocEvent> events;  // in recorded (true) order
};

bool parse_trace(const std::string& path, Trace& t) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == 'H') {
            std::istringstream is(line.substr(1));
            is >> t.tp;
            int c;
            while (is >> c) t.caps.push_back(c);
        } else if (line[0] == 'R') {
            Solve s;
            int n = 0;
            const char* p = line.c_str() + 1;
            char* end = nullptr;
            s.layer = static_cast<int>(std::strtol(p, &end, 10));
            n = static_cast<int>(std::strtol(end, &end, 10));
            const char* bar = std::strchr(end, '|');
            if (!bar) return false;
            p = bar + 1;
            for (int i = 0; i < n; ++i) {
                long e = std::strtol(p, &end, 10);
                if (end == p || *end != ':') return false;
                p = end + 1;
                long a = std::strtol(p, &end, 10);
                if (*end != ':') return false;
                p = end + 1;
                long b = std::strtol(p, &end, 10);
                if (*end != ':') return false;
                p = end + 1;
                std::strtol(p, &end, 10);  // pinned (recomputed on replay)
                p = end;
                s.topk.push_back(static_cast<uint16_t>(e));
                s.rec_assign.push_back(static_cast<uint8_t>(a));
                s.rec_bank.push_back(static_cast<int>(b));
            }
            t.solves.push_back(std::move(s));
        } else if (line[0] == 'M') {
            RelocEvent ev{};
            char kind_buf[8] = {0};
            unsigned long long sc = 0;
            unsigned layer = 0, expert = 0;
            if (std::sscanf(line.c_str(), "M %llu %1s %u %u %d %d", &sc,
                            kind_buf, &layer, &expert, &ev.old_node,
                            &ev.new_node) != 6)
                return false;
            ev.solve = sc;
            ev.kind = kind_buf[0];
            ev.layer = layer;
            ev.expert = static_cast<uint16_t>(expert);
            t.events.push_back(ev);
        }
        // 'A' lines: recorded victims — recomputed on replay, skip.
    }
    return !t.solves.empty();
}

struct ArenaModel {
    std::unordered_map<uint32_t, int> init_raw;     // key → boot raw node
    std::unordered_map<uint32_t, int> cur_raw;      // key → live raw node
    std::unordered_map<int, int> pair_of;           // raw node → paired node
    int paired(int raw) const {
        auto it = pair_of.find(raw);
        return it == pair_of.end() ? raw : it->second;
    }
};

bool parse_map(const std::string& path, ArenaModel& m) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        int l, e, paired, raw;
        if (std::sscanf(line.c_str(), "e,%d,%d,%d,%d", &l, &e, &paired,
                        &raw) == 4) {
            const uint32_t k = kid(static_cast<uint32_t>(l),
                                   static_cast<uint16_t>(e));
            m.init_raw[k] = raw;
            m.cur_raw[k] = raw;
            m.pair_of[raw] = paired;
        }
    }
    return !m.init_raw.empty();
}

enum class Policy { kLiveRaw, kLivePaired, kFrozen, kEpoch };

}  // namespace

int main(int argc, char** argv) {
    std::string trace_path, map_path, calib_path, out_path, policy_str;
    // --race-solves K: the first K solves see NO bank input (-1 → bank 0) —
    // models the retired boot-CSV race (the CSV/first snapshot lands at the
    // first routed-MoE fetch, AFTER the first solve). Old-CSV-path oracle =
    // --policy frozen --race-solves 1 (with the migrator off).
    uint64_t race_solves = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--trace") trace_path = argv[i + 1];
        else if (a == "--map") map_path = argv[i + 1];
        else if (a == "--calib") calib_path = argv[i + 1];
        else if (a == "--policy") policy_str = argv[i + 1];
        else if (a == "--out") out_path = argv[i + 1];
        else if (a == "--race-solves")
            race_solves = std::strtoull(argv[i + 1], nullptr, 10);
        else {
            std::fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    Policy policy;
    if (policy_str == "live-raw") policy = Policy::kLiveRaw;
    else if (policy_str == "live-paired") policy = Policy::kLivePaired;
    else if (policy_str == "frozen") policy = Policy::kFrozen;
    else if (policy_str == "epoch") policy = Policy::kEpoch;
    else {
        std::fprintf(stderr,
                     "usage: reef_sim --trace T --map M --calib C --policy "
                     "live-raw|live-paired|frozen|epoch --out O\n");
        return 2;
    }

    Trace trace;
    if (!parse_trace(trace_path, trace)) {
        std::fprintf(stderr, "reef_sim: cannot parse trace %s\n",
                     trace_path.c_str());
        return 1;
    }
    ArenaModel arena;
    if (!parse_map(map_path, arena)) {
        std::fprintf(stderr, "reef_sim: cannot parse map %s\n",
                     map_path.c_str());
        return 1;
    }
    if (trace.tp == 0 || trace.caps.empty()) {
        std::fprintf(stderr,
                     "reef_sim: trace has no H caps header (record with "
                     "LS_REEF_RELOC_TRACE=1)\n");
        return 1;
    }

    std::string psrc;
    std::unique_ptr<gl::ReefOrch> orch;
    try {
        orch = gl::make_reef_orch(calib_path, trace.tp, trace.caps, &psrc);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "reef_sim: %s\n", e.what());
        return 1;
    }
    // The engine arms the dump via env; the sim owns its own output file.
    if (orch->decision_dump) std::fclose(orch->decision_dump);
    orch->decision_dump = out_path.empty() ? nullptr
                                           : std::fopen(out_path.c_str(), "w");
    if (!out_path.empty() && !orch->decision_dump) {
        std::fprintf(stderr, "reef_sim: cannot open --out %s\n",
                     out_path.c_str());
        return 1;
    }

    // Frozen / epoch snapshots (paired) — construction time = boot map.
    std::unordered_map<uint32_t, int> snap;
    snap.reserve(arena.init_raw.size());
    for (const auto& [k, raw] : arena.init_raw) snap[k] = arena.paired(raw);

    const ArenaModel* am = &arena;
    const std::unordered_map<uint32_t, int>* snap_p = &snap;
    switch (policy) {
        case Policy::kLiveRaw:
            orch->bank_node_fn = [am](uint32_t l, uint16_t e) -> int {
                auto it = am->cur_raw.find(kid(l, e));
                return it == am->cur_raw.end() ? -1 : it->second;
            };
            break;
        case Policy::kLivePaired:
            orch->bank_node_fn = [am](uint32_t l, uint16_t e) -> int {
                auto it = am->cur_raw.find(kid(l, e));
                return it == am->cur_raw.end() ? -1
                                               : am->paired(it->second);
            };
            break;
        case Policy::kFrozen:
        case Policy::kEpoch:
            orch->bank_node_fn = [snap_p](uint32_t l, uint16_t e) -> int {
                auto it = snap_p->find(kid(l, e));
                return it == snap_p->end() ? -1 : it->second;
            };
            break;
    }

    // Replay: apply relocation events between solves; count divergence vs
    // the recorded stream and predicted misses (board non-residency at the
    // assigned device — exactly reef_orch_apply's admission criterion).
    size_t ev_i = 0;
    uint64_t entries_total = 0, predicted_misses = 0, assign_diverged = 0;
    uint64_t bank_diverged = 0;
    uint64_t first_divergent_solve = 0, epoch_refreshes = 0, commits_seen = 0;
    uint64_t misses_pre_first_commit = 0, misses_post_first_commit = 0;
    uint64_t entries_pre_first_commit = 0, entries_post_first_commit = 0;
    std::vector<uint8_t> assign;
    std::vector<gl::ReefEntry> entries;
    std::vector<gl::ReefVictim> evicts;

    for (size_t si = 0; si < trace.solves.size(); ++si) {
        // Events stamped `solve == k` happened after the k-th solve.
        while (ev_i < trace.events.size()
               && trace.events[ev_i].solve <= si) {
            const RelocEvent& ev = trace.events[ev_i++];
            const uint32_t k = kid(ev.layer, ev.expert);
            if (ev.kind == 'E') arena.cur_raw.erase(k);
            else arena.cur_raw[k] = ev.new_node;
            if (ev.new_node >= 0 && !arena.pair_of.count(ev.new_node))
                arena.pair_of[ev.new_node] = ev.new_node;  // unseen: identity
            if (ev.kind == 'C') {
                ++commits_seen;
                if (policy == Policy::kEpoch) {
                    // Epoch boundary: re-derive the paired snapshot from the
                    // reconstructed live state (migrator-commit refresh).
                    snap.clear();
                    for (const auto& [kk, raw] : arena.cur_raw)
                        snap[kk] = arena.paired(raw);
                    ++epoch_refreshes;
                }
            }
        }

        const Solve& s = trace.solves[si];
        const uint32_t n = static_cast<uint32_t>(s.topk.size());
        assign.assign(n, 0);
        // Boot race window: bank inputs absent for the first K solves.
        auto saved_fn = orch->bank_node_fn;
        if (si < race_solves) orch->bank_node_fn = nullptr;
        gl::reef_orch_route(*orch, s.layer, s.topk, assign);
        if (si < race_solves) orch->bank_node_fn = std::move(saved_fn);
        entries.resize(n);
        evicts.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            entries[i] = {static_cast<uint32_t>(s.layer), s.topk[i], 0,
                          assign[i]};
            evicts[i] = {static_cast<uint32_t>(s.layer), 0xFFFF, assign[i], 0};
        }
        uint64_t solve_misses = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const ExpertKey key{static_cast<uint32_t>(s.layer), s.topk[i]};
            if (!orch->board.is_resident(assign[i], key)) ++solve_misses;
            if (assign[i] != s.rec_assign[i] && !assign_diverged)
                first_divergent_solve = si + 1;
            if (assign[i] != s.rec_assign[i]) ++assign_diverged;
            if (orch->req.bank_of[i] != s.rec_bank[i]) ++bank_diverged;
        }
        predicted_misses += solve_misses;
        entries_total += n;
        if (commits_seen == 0) {
            misses_pre_first_commit += solve_misses;
            entries_pre_first_commit += n;
        } else {
            misses_post_first_commit += solve_misses;
            entries_post_first_commit += n;
        }
        gl::reef_orch_apply(*orch, s.layer, entries.data(), evicts.data(), n);
    }

    std::printf(
        "reef_sim policy=%s calib_policy=%s\n"
        "  solves=%zu entries=%llu events=%zu commits=%llu "
        "epoch_refreshes=%llu\n"
        "  predicted_misses=%llu (pre_first_commit=%llu/%llu "
        "post_first_commit=%llu/%llu)\n"
        "  assign_diverged_vs_recorded=%llu first_divergent_solve=%llu "
        "bank_diverged_vs_recorded=%llu\n",
        policy_str.c_str(), psrc.c_str(), trace.solves.size(),
        static_cast<unsigned long long>(entries_total), trace.events.size(),
        static_cast<unsigned long long>(commits_seen),
        static_cast<unsigned long long>(epoch_refreshes),
        static_cast<unsigned long long>(predicted_misses),
        static_cast<unsigned long long>(misses_pre_first_commit),
        static_cast<unsigned long long>(entries_pre_first_commit),
        static_cast<unsigned long long>(misses_post_first_commit),
        static_cast<unsigned long long>(entries_post_first_commit),
        static_cast<unsigned long long>(assign_diverged),
        static_cast<unsigned long long>(first_divergent_solve),
        static_cast<unsigned long long>(bank_diverged));
    if (orch->decision_dump) std::fclose(orch->decision_dump);
    orch->decision_dump = nullptr;
    return 0;
}

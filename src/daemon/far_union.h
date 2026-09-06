#pragma once

// Deduped first-occurrence routed-expert union over a routing export
// (the FAR seam's union builder, extracted from dispatch_reef.cpp so the
// expert-id capacity contract is unit-testable).
//
// TD-GLM5N-ROUTED-EXPERT-ID-TRUNCATION (RESOLVED): the in-line predecessor
// of this function sized its dedup table at ipc::kMaxExperts == 256 and
// SILENTLY skipped any expert id >= 256 while glm5_next routes over 288
// experts — experts 256-287 were never fetched, dispatched, or folded, on
// every token, with moe_degraded_layers staying 0. The table is now sized
// kMaxExperts (>= 288, boot-refused otherwise) and an out-of-range
// num_experts is a loud error at the call site, never a silent thinning.

#include <cstdint>
#include <vector>

#include "ipc_protocol.h"

namespace layerstorm::daemon {

// Builds the deduped first-occurrence union of routed expert ids from the
// routing-export indices `ridx[0..rn)`. Ids outside [0, num_experts) are
// skipped (padding / invalid slots). The union is capped at
// ipc::kMaxExpertPrefetch entries (sideband capacity).
//
// Returns false iff num_experts exceeds ipc::kMaxExperts — the caller must
// fail loudly (Engine::init_modules refuses such configs at boot, so this
// is a defense-in-depth seam guard, not a reachable branch).
inline bool build_routed_union(const int32_t* ridx, uint32_t rn,
                               int num_experts,
                               std::vector<uint16_t>& out) {
    if (num_experts > ipc::kMaxExperts) return false;
    uint8_t seen[ipc::kMaxExperts] = {0};
    for (uint32_t k = 0; k < rn; ++k) {
        const int32_t e = ridx[k];
        if (e < 0 || e >= num_experts || seen[e]) continue;
        seen[e] = 1;
        if (out.size() >= ipc::kMaxExpertPrefetch) break;
        out.push_back(static_cast<uint16_t>(e));
    }
    return true;
}

}  // namespace layerstorm::daemon

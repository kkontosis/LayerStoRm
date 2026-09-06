#pragma once

// ── 44z KV <-> expert zone rebalancing: the waste equation ──────────────────
// (spec/tickets/44z_KV_EXPERT_ZONE_REBALANCE.md §3 — user-required, GENERAL)
//
// Both inputs are MODEL-DERIVED and must never be hardcoded:
//   expert_slot_bytes = LayerRegistry::per_routed_expert_bytes()
//                       (18,415,616 B on GLM-5.3-Flash, 27,623,424 B on
//                        GLM-5.2 Q4_K_XL — GG-9 per-projection max),
//   slab_bytes        = VramLayout::slab_bytes
//                       (272,448 B glm5_next, 1,086,976 B GLM-5.2).
// The ratio is NOT integral on any served model (67.5931 / 25.4131), so the
// equation stays general; measured 2026-09-02, the minimal-S fit is always
// far inside max_waste = 0.10 (0.598% / 2.257%), so the N-search below
// degenerates to its first iteration on today's models — kept because it is
// ten lines and correct for a future model where ceil() overshoots the bound.
//
// For a region of S slabs holding N expert slots:
//   capacity(S)   = S * slab_bytes
//   N(S)          = floor((capacity(S) - base_pad) / slot_stride)
//   waste(S)      = capacity(S) - N(S) * expert_slot_bytes
//   waste_frac(S) = waste(S) / capacity(S)
// admissible iff N(S) >= 1 and waste_frac(S) <= max_waste.
//
// slot_stride / base_pad generalize the ticket equation for device
// alignment (the kZoneAlign lesson, vram_allocator.cpp:1408-1411: a
// byte-exact zone split once faulted with cudaErrorMisalignedAddress 716):
// a granted slab run's base is only guaranteed slab-granular (64 B on
// glm5_next), so slot 0 is placed at align_up(base, kExpertZoneSlotAlign)
// and slots stride by align_up(expert_slot_bytes, kExpertZoneSlotAlign).
// On every served model expert_slot_bytes is already a 4096-multiple, so
// slot_stride == expert_slot_bytes and the ticket equation holds verbatim;
// the pad/stride terms exist so a model where that fails wastes bytes
// instead of faulting. Alignment padding counts as WASTE (capacity the
// region spends on not holding experts), which keeps waste_frac honest.

#include <cstdint>

namespace layerstorm::memory {

/// Device alignment for expert slot base addresses inside a granted region.
/// Mirrors VramAllocator's kZoneAlign (the 716-misalignment lesson).
inline constexpr int64_t kExpertZoneSlotAlign = 4096;

inline constexpr int64_t align_up_i64(int64_t v, int64_t a) {
    return ((v + a - 1) / a) * a;
}

/// Model-derived geometry + the config knob. Construct once per boot.
struct ExpertZoneGeometry {
    int64_t expert_slot_bytes = 0;  ///< per_routed_expert_bytes (model-derived)
    int64_t slab_bytes = 0;         ///< VramLayout::slab_bytes (model-derived)
    double max_waste = 0.10;        ///< config knob (44z §3, default 0.10)

    /// Stride between consecutive slot bases inside a granted region.
    int64_t slot_stride() const {
        return align_up_i64(expert_slot_bytes, kExpertZoneSlotAlign);
    }

    bool valid() const {
        return expert_slot_bytes > 0 && slab_bytes > 0
            && max_waste >= 0.0 && max_waste <= 1.0;
    }

    int64_t capacity_bytes(int num_slabs) const {
        return static_cast<int64_t>(num_slabs) * slab_bytes;
    }

    /// Alignment pad consumed before slot 0 for a region whose device base
    /// has the given misalignment (base % kExpertZoneSlotAlign).
    int64_t base_pad(int64_t base_misalign) const {
        return base_misalign == 0
            ? 0 : kExpertZoneSlotAlign - base_misalign;
    }

    /// N(S): expert slots a region of S slabs holds.
    int slots_in(int num_slabs, int64_t base_misalign = 0) const {
        const int64_t usable =
            capacity_bytes(num_slabs) - base_pad(base_misalign);
        if (usable < expert_slot_bytes) return 0;
        // The LAST slot needs only expert_slot_bytes, not a full stride.
        return static_cast<int>(
            (usable - expert_slot_bytes) / slot_stride() + 1);
    }

    /// Smallest S with slots_in(S) >= N (the "prefer the smallest S" rule —
    /// grants stay aligned to real demand rather than swallowing the pool).
    int slabs_for_slots(int num_slots, int64_t base_misalign = 0) const {
        if (num_slots <= 0) return 0;
        const int64_t need = base_pad(base_misalign)
            + static_cast<int64_t>(num_slots - 1) * slot_stride()
            + expert_slot_bytes;
        return static_cast<int>((need + slab_bytes - 1) / slab_bytes);
    }

    int64_t waste_bytes(int num_slabs, int num_slots) const {
        return capacity_bytes(num_slabs)
            - static_cast<int64_t>(num_slots) * expert_slot_bytes;
    }

    double waste_frac(int num_slabs, int num_slots) const {
        const int64_t cap = capacity_bytes(num_slabs);
        return cap > 0
            ? static_cast<double>(waste_bytes(num_slabs, num_slots))
                  / static_cast<double>(cap)
            : 1.0;
    }

    /// A region of S slabs holding its natural N is admissible iff N >= 1
    /// and waste_frac <= max_waste (44z §3).
    bool admissible(int num_slabs, int64_t base_misalign = 0) const {
        const int n = slots_in(num_slabs, base_misalign);
        return n >= 1 && waste_frac(num_slabs, n) <= max_waste;
    }

    /// The general fit: given a contiguous free run of `run_slabs`, return
    /// the largest admissible (N, S) with S <= run_slabs, preferring the
    /// smallest S for that N. {0, 0} when nothing admissible fits.
    struct Fit {
        int num_slots = 0;
        int num_slabs = 0;
    };
    Fit best_fit(int run_slabs, int64_t base_misalign = 0) const {
        for (int n = slots_in(run_slabs, base_misalign); n >= 1; --n) {
            const int s = slabs_for_slots(n, base_misalign);
            if (s <= run_slabs && waste_frac(s, n) <= max_waste)
                return Fit{n, s};
        }
        return Fit{};
    }

    /// Smallest S <= run_slabs that (i) holds at least ask_slots under the
    /// WORST-CASE base pad and (ii) is admissible (waste_frac <= max_waste)
    /// under that worst-case pad. 0 when no such S fits the run.
    ///
    /// This is the ticket's "prefer the smallest S meeting the bound" made
    /// operational for a CALLER THAT CANNOT SEE THE BASE YET: a grant is
    /// sized BEFORE the allocator picks the run, and slab_bytes is not a
    /// 4096-multiple on any served model (272,448 and 1,086,976), so the pad
    /// before slot 0 varies with the start slab. Sizing against pad 0 and
    /// discovering the real pad after the claim is what lets a small grant
    /// land over max_waste.
    ///
    /// WORST-CASE-PAD MONOTONICITY is what makes one pre-claim check
    /// sufficient. For a fixed S, slots_in is non-increasing in the pad and
    /// waste is non-decreasing in it (a bigger pad only removes usable
    /// bytes). base_misalign = 1 gives base_pad = 4095, the largest pad any
    /// base can have. So a smaller ACTUAL pad yields >= slots and <= waste:
    /// admissibility under pad 4095 IMPLIES admissibility at whatever base
    /// the allocator returns. The caller may then fill the claimed slabs to
    /// slots_in(S, actual_misalign) and stay inside the bound by
    /// construction.
    ///
    /// On the served models the worst-case pad costs < 0.03% of a slot, so
    /// this returns slabs_for_slots(ask_slots) on the first iteration and is
    /// behavior-neutral; the search only bites where a slot is SMALL
    /// relative to a slab.
    int grant_slabs(int ask_slots, int run_slabs) const {
        if (ask_slots < 1 || run_slabs < 1) return 0;
        for (int s = slabs_for_slots(ask_slots, /*base_misalign=*/1);
             s <= run_slabs; ++s)
            if (admissible(s, /*base_misalign=*/1)) return s;
        return 0;
    }
};

}  // namespace layerstorm::memory

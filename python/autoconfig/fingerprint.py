"""Hardware fingerprint (AUTOCONFIG §7).

Measurement-derived topology, never volatile magnitudes: the INV-ARENA-
CACHE-ORDER lesson is that resolved/measured QUANTITIES drift run to run
(fraction_free slot counts ±1) and must not enter an identity hash. What
goes in is the stable topology a re-cabling changes:
- per-GPU: UUID, PCI bus id, NUMA node, max PCIe gen/width (sysfs: catches
  a card moved to a Gen4 or x4 slot), physical VRAM MiB;
- NUMA node list (id, GiB-rounded size, is_hbm);
- host MemTotal rounded to GiB; NVMe (gen, width) list;
- when a calibration artifact is accepted: its per-position (uuid, numa)
  and the [bank][device] TIER matrix — measured locality topology.
Rates (rate_us, egress_us, contention) are deliberately EXCLUDED: they are
magnitudes with jitter, and a fingerprint that flaps forces re-derivation
on every boot.
"""

from __future__ import annotations

import hashlib
import json

from .calibration import CalibrationView
from .hwdetect import HardwareDescriptor


def fingerprint_dict(hw: HardwareDescriptor, cal: CalibrationView | None = None) -> dict:
    d = {
        "gpus": [
            {
                "uuid": g.uuid, "pci": g.pci_bus_id, "numa": g.numa_node,
                "pcie_gen_max": g.pcie_gen_max, "pcie_width_max": g.pcie_width_max,
                "vram_mib": g.vram_mib,
            }
            for g in hw.gpus
        ],
        "numa_nodes": [
            {"node": n.node, "gib": round(n.mem_total_kib / (1 << 20)),
             "hbm": n.is_hbm_bank}
            for n in hw.numa_nodes
        ],
        "mem_total_gib": round(hw.mem_total_kib / (1 << 20)),
        "nvmes": [{"gen": n.pcie_gen, "width": n.pcie_width} for n in hw.nvmes],
    }
    if cal is not None:
        d["calibration"] = {
            "devices": [{"uuid": c.uuid, "numa": c.numa_node} for c in cal.devices],
            "banks": [b.node for b in cal.banks],
            "tier_matrix": [list(r) for r in cal.tier_matrix],
        }
    return d


def fingerprint(hw: HardwareDescriptor, cal: CalibrationView | None = None) -> str:
    js = json.dumps(fingerprint_dict(hw, cal), sort_keys=True, separators=(",", ":"))
    return "hwfp1-" + hashlib.sha256(js.encode()).hexdigest()[:32]

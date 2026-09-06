"""GPU speed capabilities from the engine's own calibration (AUTOCONFIG §3/§4).

User directive (2026-08-30): boxes get re-carded and re-wired — PCIe4 vs
PCIe5, mixed dies, different slots. GPU SPEED therefore comes from the
engine's EXISTING gpu_loader calibration artifact (measured on THIS box),
never from a card-name-keyed table. The artifact shape is
``loader_constants.h:100-121`` (version 2); canonical example
``test-data/gpu_loader_calibration_ep4x4.json``. The engine's
``load_or_calibrate_with`` (engine.cpp:559) is the measuring/self-healing
step: when autoconfig finds no acceptable artifact it emits a config whose
``gpu_loader`` section makes the FIRST BOOT measure and persist one, and
marks every speed-derived choice provisional until then.

The accept predicate here mirrors engine.cpp:519-568 (INV-LOADER-CAL-6):
per-position device UUID must match the live GPU (prefix match — stored
UUIDs are truncated), and compute dims must match the model
(N = 2*moe_intermediate_size, K = hidden_size). Rejection reasons are
returned, not swallowed — they become explanation lines.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .hwdetect import HardwareDescriptor


@dataclass(frozen=True)
class CalDevice:
    position: int
    numa_node: int
    name: str
    uuid: str
    xfer_lat_us: float
    compute_a_us: float
    compute_b_us: float
    compute_p: int

    def tokens_per_us_at(self, experts: int) -> float:
        """Inverse of loader_solver.h compute_us(): a + b*ceil(c/P)."""
        import math
        us = self.compute_a_us + self.compute_b_us * math.ceil(experts / max(1, self.compute_p))
        return experts / us if us > 0 else 0.0


@dataclass(frozen=True)
class CalBank:
    node: int
    egress_us: float
    contention: float
    is_hbm: bool = False


@dataclass(frozen=True)
class CalibrationView:
    path: str
    source: str
    expert_bytes: float
    compute_n: int
    compute_k: int
    devices: tuple[CalDevice, ...]
    banks: tuple[CalBank, ...]
    tier_matrix: tuple[tuple[int, ...], ...]   # [bank][device], stable topology
    rate_matrix: tuple[tuple[float, ...], ...]  # [bank][device] rate_us (volatile)

    def device_speed_rank(self) -> list[int]:
        """Positions sorted fastest-first by measured routed-FFN throughput
        at a full batch (P experts) — the measured replacement for the
        rtx5090/rtx5080 name table."""
        return sorted(
            (d.position for d in self.devices),
            key=lambda p: -next(dv for dv in self.devices
                                if dv.position == p).tokens_per_us_at(
                                    next(dv for dv in self.devices
                                         if dv.position == p).compute_p),
        )

    def h2d_gib_per_s(self, position: int) -> float:
        """Best measured H2D rate into this device across banks, GiB/s
        (rate_us is per expert_bytes transfer)."""
        best_us = min(row[position] for row in self.rate_matrix if row[position] > 0)
        return (self.expert_bytes / (best_us * 1e-6)) / (1 << 30)


def load_calibration(path: str) -> CalibrationView:
    with open(path) as f:
        d = json.load(f)
    devices = tuple(
        CalDevice(
            position=int(dev["position"]), numa_node=int(dev["numa_node"]),
            name=str(dev.get("name", "")), uuid=str(dev.get("uuid", "")),
            xfer_lat_us=float(dev.get("xfer_lat_us", 0.0)),
            compute_a_us=float((dev.get("compute") or {}).get("a_us", 0.0)),
            compute_b_us=float((dev.get("compute") or {}).get("b_us", 0.0)),
            compute_p=int((dev.get("compute") or {}).get("P", 1)),
        )
        for dev in d.get("devices", [])
    )
    banks = tuple(
        CalBank(node=int(b["node"]), egress_us=float(b.get("egress_us", 0.0)),
                contention=float(b.get("contention", 0.0)),
                is_hbm=bool(b.get("is_hbm", False)))
        for b in d.get("banks", [])
    )
    matrix = d.get("matrix", [])
    return CalibrationView(
        path=path, source=str(d.get("source", "")),
        expert_bytes=float(d.get("expert_bytes", 0.0)),
        compute_n=int(d.get("compute_N", 0)), compute_k=int(d.get("compute_K", 0)),
        devices=devices, banks=banks,
        tier_matrix=tuple(tuple(int(c["tier"]) for c in row) for row in matrix),
        rate_matrix=tuple(tuple(float(c["rate_us"]) for c in row) for row in matrix),
    )


def accept_calibration(cal: CalibrationView, hw: HardwareDescriptor,
                       gpu_order: list[int], model_hidden: int,
                       model_moe_intermediate: int) -> list[str]:
    """Mirror engine.cpp:519-568 AcceptFn. Returns [] when accepted, else
    human-readable rejection reasons.

    gpu_order: hardware.gpus emission order as detection ordinals — cal
    position i corresponds to gpu_order[i] (positions are hardware.gpus
    indices, GpuRef contract)."""
    reasons: list[str] = []
    if len(cal.devices) != len(gpu_order):
        reasons.append(f"device count {len(cal.devices)} != live expert-GPU count {len(gpu_order)}")
        return reasons
    by_ordinal = {g.ordinal: g for g in hw.gpus}
    for dev in cal.devices:
        if dev.position >= len(gpu_order):
            reasons.append(f"position {dev.position} out of range")
            continue
        live = by_ordinal.get(gpu_order[dev.position])
        if live is None:
            reasons.append(f"no live GPU for position {dev.position}")
            continue
        if dev.uuid and live.uuid and not live.uuid.startswith(dev.uuid[:36]):
            reasons.append(
                f"position {dev.position}: stored UUID {dev.uuid[:20]}… is not live "
                f"GPU {live.uuid[:20]}… (wrong machine / reordered devices, INV-LOADER-CAL-6)")
    n, k = 2 * model_moe_intermediate, model_hidden
    if cal.compute_n and cal.compute_n != n:
        reasons.append(f"compute_N {cal.compute_n} != model 2*moe_intermediate {n} (wrong model)")
    if cal.compute_k and cal.compute_k != k:
        reasons.append(f"compute_K {cal.compute_k} != model hidden_size {k} (wrong model)")
    return reasons


def resolve_calibration_path(config_value: str, weights_path: str) -> str:
    """engine.cpp:500-512: empty -> <weights dir>/gpu_loader_calibration.json;
    ONLY absolute paths are verbatim — every relative path (bare filename OR
    with directories) joins the weights dir. VERIFIED against a live boot
    2026-08-30: 'scratchpad/x.json' resolved to
    'test-data/<weights>/scratchpad/x.json' (the earlier transcription
    treated relative-with-dir as verbatim — wrong)."""
    wdir = os.path.dirname(weights_path)
    if not config_value:
        return os.path.join(wdir, "gpu_loader_calibration.json")
    if os.path.isabs(config_value):
        return config_value
    return os.path.join(wdir, config_value)

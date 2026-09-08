"""Hardware detection for autoconfig (AUTOCONFIG §3).

CPU-only by construction: the hardware sweep is read from /proc and /sys. No
CUDA — this module must be safe to run while another process owns the GPUs
(INV-GPU-1 stays untouched; we never open a device or create a context).

The ONE exception is the VRAM fallback chain below: when the PCI BAR1 aperture
does not map the framebuffer we shell out to a read-only
`nvidia-smi --query-gpu` (NVML) query. That opens no CUDA context, takes no
device ownership, and does not disturb a live holder, so it is safe in exactly
the situations the /proc + /sys sweep is. It is never on the primary path.

Traps encoded here (each has bitten this project — AUTOCONFIG §3 table):
- /proc/driver/nvidia/gpus/* order is lexicographic by PCI bus id, NOT the
  CUDA ordinal. CUDA ordinals under CUDA_DEVICE_ORDER=PCI_BUS_ID coincide
  with PCI order, which is what the serving stack pins, so we adopt PCI
  order as the ordinal and record it explicitly.
- current_link_speed shows the idle downclock (2.5 GT/s on an idle Gen5
  link); only max_link_speed is trustworthy without kickstarting the link
  (config_resolver.cpp kickstarts; we cannot, GPU-free).
- Physical VRAM is read from the PCI BAR1 span, which maps the whole
  framebuffer only while resizable BAR is ON. With ReBAR OFF the span is a
  256 MiB window that says nothing about framebuffer size, so VRAM resolves
  through three tiers, in order:
      1. "bar1"  — the BAR1 span (primary; every carve constant downstream is
                   calibrated against this number, so it must stay first).
      2. "nvml"  — a read-only nvidia-smi query, keyed by PCI bus id.
      3. "table" — GPU_VRAM_FALLBACK_MIB, keyed by the detected model.
  A model that matches no table row and has neither BAR1 nor NVML REFUSES
  rather than guessing: detect_gpu_type() collapses every unrecognised name
  onto its smallest-card fallback, so a silent table hit plans a small carve
  for a large board with no diagnostic at all.
- CPU-less HBM NUMA banks (memory > 0, empty cpulist) must be enumerated:
  a node-local OOM fires with 30 GB free system-wide if they are treated
  like DDR nodes.
- Per-node meminfo has no MemAvailable line; node "available" must be
  reconstructed (MemFree + FilePages-ish) — we record MemTotal/MemFree and
  leave policy to the solver.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from dataclasses import dataclass, field

from .explain import Infeasible

# Physical VRAM fallback of LAST resort, when neither BAR1 nor NVML answers
# (MiB, driver-reported physical sizes; the *usable* carve is derived by the
# solver, AUTOCONFIG §3). These rows are transcribed verbatim from
# `nvidia-smi --query-gpu=memory.total`, so tier 2 and tier 3 speak the same
# units. Note they are NOT the BAR1 numbers: a 32 GiB board apertures 32768
# MiB but reports 32607 MiB of framebuffer (the driver/ECC carve-out).
GPU_VRAM_FALLBACK_MIB = {
    "rtx5090": 32607,
    "rtx5080": 16303,
}

# Mirrors config_resolver.cpp normalized_speed() / types.py
# DEFAULT_COMPUTE_WEIGHTS (rtx5090: 170 SMs, rtx5080: 84 SMs).
COMPUTE_WEIGHT = {
    "rtx5090": 1.0,
    "rtx5080": 0.494,
}

_LINK_SPEED_TO_GEN = {
    "2.5": 1,
    "5.0": 2,
    "8.0": 3,
    "16.0": 4,
    "32.0": 5,
    "64.0": 6,
}


def detect_gpu_type(name: str) -> str:
    """Mirror src/core/hardware_detect.cpp detect_gpu_type(): the config
    enum only has rtx5090/rtx5080 today; unknown falls back to rtx5080."""
    if "5090" in name:
        return "rtx5090"
    if "5080" in name:
        return "rtx5080"
    return "rtx5080"


def gpu_name_is_known(name: str) -> bool:
    """True when detect_gpu_type() RECOGNISED the model, rather than taking
    its unrecognised-name fallback.

    The two are indistinguishable from the return value alone — both yield
    "rtx5080" — which is exactly why an unknown card used to silently inherit
    the 5080 VRAM row. Any caller that treats the detected type as a fact
    about the hardware (rather than a config enum to fill) must gate on this.
    """
    return "5090" in name or "5080" in name


def normalize_pci_bus_id(bus_id: str) -> str:
    """Normalise a PCI bus id to the lowercase sysfs form ("0000:16:00.0").

    nvidia-smi prints an 8-hex-digit domain ("00000000:16:00.0") while sysfs
    and /proc/driver/nvidia use 4, so the two sources do not join without
    this. Anything that does not look like domain:bus:dev.fn is returned
    lowercased and otherwise untouched.
    """
    b = bus_id.strip().lower()
    parts = b.split(":")
    if len(parts) != 3:
        return b
    parts[0] = parts[0][-4:].rjust(4, "0")
    return ":".join(parts)


def query_nvml_vram_mib(timeout_s: float = 10.0) -> dict[str, int]:
    """Physical VRAM in MiB per PCI bus id, via a read-only NVML query.

    `nvidia-smi --query-gpu` reads driver state; it opens no CUDA context and
    takes no device ownership, so it is safe to run against GPUs another
    process is serving on. EVERY failure mode — nvidia-smi absent, driver
    error, timeout, garbled row — degrades to an empty map so the caller
    falls through to the name table rather than raising. The values are the
    same quantity as GPU_VRAM_FALLBACK_MIB, not the BAR1 aperture.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}
    out: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            mib = int(parts[1])
        except ValueError:
            continue
        if mib > 0:
            out[normalize_pci_bus_id(parts[0])] = mib
    return out


@dataclass(frozen=True)
class GpuInfo:
    ordinal: int              # PCI-bus-id order == CUDA ordinal under PCI_BUS_ID
    pci_bus_id: str           # lowercase sysfs form, e.g. "0000:6a:00.0"
    name: str                 # driver model string
    gpu_type: str             # rtx5090 / rtx5080 (config enum)
    uuid: str
    vram_mib: int             # physical (BAR1 span, NVML, or fallback table)
    vram_source: str          # "bar1" | "nvml" | "table"
    pcie_gen_max: int
    pcie_width_max: int
    numa_node: int

    @property
    def compute_weight(self) -> float:
        return COMPUTE_WEIGHT.get(self.gpu_type, 0.494)


@dataclass(frozen=True)
class NumaNodeInfo:
    node: int
    mem_total_kib: int
    mem_free_kib: int
    has_cpus: bool

    @property
    def is_hbm_bank(self) -> bool:
        """CPU-less memory bank (nodes 4-7 on the dev box: 16 GiB each)."""
        return (not self.has_cpus) and self.mem_total_kib > 0


@dataclass(frozen=True)
class NvmeInfo:
    name: str
    model: str
    pcie_gen: int
    pcie_width: int

    @property
    def bandwidth_ceiling_gbps(self) -> float:
        """Approximate practical ceiling in GB/s (protocol-overhead-adjusted).

        Per-lane payload GB/s by gen (approx measured practice, not raw GT/s):
        gen3 ~0.85, gen4 ~1.7, gen5 ~3.4. The dev box's Gen3 x4 990 PRO
        measures ~3.3 GB/s — a HARDWARE limit no config routes around
        (spec: disk-ceiling finding)."""
        per_lane = {1: 0.21, 2: 0.42, 3: 0.85, 4: 1.7, 5: 3.4, 6: 6.8}
        return per_lane.get(self.pcie_gen, 0.0) * self.pcie_width


@dataclass(frozen=True)
class HardwareDescriptor:
    gpus: tuple[GpuInfo, ...]
    numa_nodes: tuple[NumaNodeInfo, ...]
    mem_total_kib: int
    nvmes: tuple[NvmeInfo, ...] = field(default_factory=tuple)

    @property
    def mem_total_gib(self) -> float:
        return self.mem_total_kib / (1024.0 * 1024.0)

    def gpu_attached_nodes(self) -> tuple[int, ...]:
        return tuple(sorted({g.numa_node for g in self.gpus if g.numa_node >= 0}))

    def hbm_nodes(self) -> tuple[int, ...]:
        return tuple(n.node for n in self.numa_nodes if n.is_hbm_bank)

    def to_json(self) -> dict:
        return {
            "gpus": [g.__dict__ for g in self.gpus],
            "numa_nodes": [n.__dict__ for n in self.numa_nodes],
            "mem_total_kib": self.mem_total_kib,
            "nvmes": [n.__dict__ for n in self.nvmes],
        }

    @classmethod
    def from_json(cls, d: dict) -> "HardwareDescriptor":
        return cls(
            gpus=tuple(GpuInfo(**g) for g in d.get("gpus", [])),
            numa_nodes=tuple(NumaNodeInfo(**n) for n in d.get("numa_nodes", [])),
            mem_total_kib=int(d.get("mem_total_kib", 0)),
            nvmes=tuple(NvmeInfo(**n) for n in d.get("nvmes", [])),
        )


def _read(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _parse_link_gen(s: str | None) -> int:
    """'32.0 GT/s PCIe' -> 5 (mirror parse_pcie_gen). 0 if unrecognised."""
    if not s:
        return 0
    m = re.match(r"\s*([0-9]+\.[0-9]+)\s*GT/s", s)
    if not m:
        return 0
    return _LINK_SPEED_TO_GEN.get(m.group(1), 0)


def _parse_int(s: str | None, default: int = 0) -> int:
    if s is None:
        return default
    try:
        return int(s.strip())
    except ValueError:
        return default


def _bar1_span_bytes(resource_text: str | None) -> int:
    """BAR1 span from a sysfs `resource` file (line 1: start end flags)."""
    if not resource_text:
        return 0
    lines = resource_text.splitlines()
    if len(lines) < 2:
        return 0
    parts = lines[1].split()
    if len(parts) < 2:
        return 0
    try:
        start, end = int(parts[0], 16), int(parts[1], 16)
    except ValueError:
        return 0
    if end <= start:
        return 0
    return end - start + 1


def detect_gpus(proc_root: str = "/proc", sys_root: str = "/sys",
                nvml_vram: dict[str, int] | None = None) -> tuple[GpuInfo, ...]:
    """Enumerate NVIDIA GPUs from /proc/driver/nvidia + PCI sysfs.

    `nvml_vram` is the tier-2 VRAM map (bus id -> MiB). None means "query it
    live, and only if some GPU actually needs it" — pass a dict to keep a
    caller hermetic ({} to assert the no-NVML path).

    Raises Infeasible when a GPU's physical VRAM cannot be established from
    any tier; see the module docstring for why that is a refusal rather than
    a guess.
    """
    gpu_dirs = sorted(glob.glob(os.path.join(proc_root, "driver/nvidia/gpus/*")))
    gpus: list[GpuInfo] = []
    for ordinal, d in enumerate(gpu_dirs):
        bus_id = os.path.basename(d).lower()
        info = _read(os.path.join(d, "information")) or ""
        name = ""
        uuid = ""
        for line in info.splitlines():
            if line.startswith("Model:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("GPU UUID:"):
                uuid = line.split(":", 1)[1].strip()
        gpu_type = detect_gpu_type(name)
        pci_dir = os.path.join(sys_root, "bus/pci/devices", bus_id)
        numa_node = _parse_int(_read(os.path.join(pci_dir, "numa_node")), -1)
        gen_max = _parse_link_gen(_read(os.path.join(pci_dir, "max_link_speed")))
        width_max = _parse_int(_read(os.path.join(pci_dir, "max_link_width")), 0)
        bar1 = _bar1_span_bytes(_read(os.path.join(pci_dir, "resource")))
        if bar1 >= 4 << 30:  # a resizable BAR1 mapping the whole framebuffer
            vram_mib = bar1 // (1 << 20)
            vram_source = "bar1"
        else:
            # ReBAR off: the span is a 256 MiB window, not the framebuffer.
            if nvml_vram is None:
                nvml_vram = query_nvml_vram_mib()  # once, memoised per sweep
            vram_mib = nvml_vram.get(bus_id, 0)
            vram_source = "nvml"
            if vram_mib <= 0:
                if not gpu_name_is_known(name):
                    raise Infeasible(
                        "gpu-vram-unknown",
                        f"gpu {ordinal} ({name or 'unnamed model'}) at {bus_id}",
                        "the physical VRAM size",
                        "BAR1 does not map the framebuffer (resizable BAR is "
                        "off), NVML did not answer, and the model matches no "
                        "GPU_VRAM_FALLBACK_MIB row",
                        "enable Resizable BAR in the system firmware, or make "
                        "nvidia-smi runnable, or add this model to "
                        "GPU_VRAM_FALLBACK_MIB in hwdetect.py — falling back "
                        "to the table here would size the whole recipe for a "
                        "different board.")
                vram_mib = GPU_VRAM_FALLBACK_MIB[gpu_type]
                vram_source = "table"
        gpus.append(GpuInfo(
            ordinal=ordinal, pci_bus_id=bus_id, name=name, gpu_type=gpu_type,
            uuid=uuid, vram_mib=vram_mib, vram_source=vram_source,
            pcie_gen_max=gen_max, pcie_width_max=width_max, numa_node=numa_node,
        ))
    return tuple(gpus)


def detect_numa(sys_root: str = "/sys") -> tuple[NumaNodeInfo, ...]:
    nodes: list[NumaNodeInfo] = []
    for d in sorted(glob.glob(os.path.join(sys_root, "devices/system/node/node[0-9]*")),
                    key=lambda p: int(re.search(r"node(\d+)$", p).group(1))):
        node = int(re.search(r"node(\d+)$", d).group(1))
        cpulist = (_read(os.path.join(d, "cpulist")) or "").strip()
        meminfo = _read(os.path.join(d, "meminfo")) or ""
        total_kib = free_kib = 0
        for line in meminfo.splitlines():
            if "MemTotal:" in line:
                total_kib = int(line.split()[-2])
            elif "MemFree:" in line:
                free_kib = int(line.split()[-2])
        nodes.append(NumaNodeInfo(node=node, mem_total_kib=total_kib,
                                  mem_free_kib=free_kib, has_cpus=bool(cpulist)))
    return tuple(nodes)


def detect_host_ram_kib(proc_root: str = "/proc") -> int:
    meminfo = _read(os.path.join(proc_root, "meminfo")) or ""
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return 0


def detect_nvmes(sys_root: str = "/sys") -> tuple[NvmeInfo, ...]:
    nvmes: list[NvmeInfo] = []
    for d in sorted(glob.glob(os.path.join(sys_root, "class/nvme/nvme[0-9]*"))):
        name = os.path.basename(d)
        model = (_read(os.path.join(d, "model")) or "").strip()
        dev = os.path.join(d, "device")
        # follow to the PCI function; in a fake tree `device` is a plain dir
        pci = os.path.realpath(dev) if os.path.exists(dev) else dev
        gen = _parse_link_gen(_read(os.path.join(pci, "current_link_speed")))
        width = _parse_int(_read(os.path.join(pci, "current_link_width")), 0)
        nvmes.append(NvmeInfo(name=name, model=model, pcie_gen=gen, pcie_width=width))
    return tuple(nvmes)


def detect_hardware(proc_root: str = "/proc", sys_root: str = "/sys",
                    nvml_vram: dict[str, int] | None = None) -> HardwareDescriptor:
    """Full hardware sweep. Roots and the tier-2 VRAM map are injectable for
    tests; see detect_gpus() for the VRAM fallback contract."""
    return HardwareDescriptor(
        gpus=detect_gpus(proc_root, sys_root, nvml_vram),
        numa_nodes=detect_numa(sys_root),
        mem_total_kib=detect_host_ram_kib(proc_root),
        nvmes=detect_nvmes(sys_root),
    )

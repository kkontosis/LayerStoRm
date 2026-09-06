"""Hardware detection for autoconfig (AUTOCONFIG §3).

CPU-only by construction: everything is read from /proc and /sys. No CUDA,
no nvidia-smi, no NVML — this module must be safe to run while another
process owns the GPUs (INV-GPU-1 stays untouched; we never open a device).

Traps encoded here (each has bitten this project — AUTOCONFIG §3 table):
- /proc/driver/nvidia/gpus/* order is lexicographic by PCI bus id, NOT the
  CUDA ordinal. CUDA ordinals under CUDA_DEVICE_ORDER=PCI_BUS_ID coincide
  with PCI order, which is what the serving stack pins, so we adopt PCI
  order as the ordinal and record it explicitly.
- current_link_speed shows the idle downclock (2.5 GT/s on an idle Gen5
  link); only max_link_speed is trustworthy without kickstarting the link
  (config_resolver.cpp kickstarts; we cannot, GPU-free).
- Physical VRAM is read from the PCI BAR1 span (resizable BAR maps the whole
  framebuffer: 32.0/16.0 GiB exact on this box); a name-keyed fallback table
  covers boxes without resizable BAR.
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
from dataclasses import dataclass, field

# Physical VRAM fallback when BAR1 is not resizable (MiB, driver-reported
# physical sizes; the *usable* carve is derived by the solver, AUTOCONFIG §3).
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


@dataclass(frozen=True)
class GpuInfo:
    ordinal: int              # PCI-bus-id order == CUDA ordinal under PCI_BUS_ID
    pci_bus_id: str           # lowercase sysfs form, e.g. "0000:6a:00.0"
    name: str                 # driver model string
    gpu_type: str             # rtx5090 / rtx5080 (config enum)
    uuid: str
    vram_mib: int             # physical (BAR1 span or fallback table)
    vram_source: str          # "bar1" | "table"
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


def detect_gpus(proc_root: str = "/proc", sys_root: str = "/sys") -> tuple[GpuInfo, ...]:
    """Enumerate NVIDIA GPUs from /proc/driver/nvidia + PCI sysfs. CPU-only."""
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
            vram_mib = GPU_VRAM_FALLBACK_MIB.get(gpu_type, 0)
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


def detect_hardware(proc_root: str = "/proc", sys_root: str = "/sys") -> HardwareDescriptor:
    """Full CPU-only hardware sweep. Roots are injectable for tests."""
    return HardwareDescriptor(
        gpus=detect_gpus(proc_root, sys_root),
        numa_nodes=detect_numa(sys_root),
        mem_total_kib=detect_host_ram_kib(proc_root),
        nvmes=detect_nvmes(sys_root),
    )

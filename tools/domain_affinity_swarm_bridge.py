#!/usr/bin/env python3
"""I8 domain-affinity queue scheduler and optional federated swarm bridge.

Implements the exact per-layer cost model from ``docs/I8_PLACEMENT_MODEL.md``:

    T(j[.]) = prep
            + max(device_makespan, bank_egress)
            + recon
            + place_cons_total
            + evict_cons_total

and quantifies how reordering or routing incoming ``/v1/chat/completions``
requests by semantic domain affinity increases GPU expert cache residency
``cached_gpu[i, j]``, reducing uncached fetches ``n_j``, convex eviction cost
``evict_cons_total``, and ``max(device_makespan, bank_egress)``.

Also provides a zero-dependency helper to verify the run-to-run determinism
preconditions documented in ``tools/serve_determinism_probe.py``
(``LAYERSTORM_DETERMINISTIC_EP_COMBINE=1`` and ``LS_SPEC_GOVERNOR=0``) when
connecting a ``python/cli/serve.py`` instance to an upstream OpenAI-compatible
router (such as Project HiveMind).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1. Exact I8 Placement Cost Model Evaluator (docs/I8_PLACEMENT_MODEL.md)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class I8HardwareProfile:
    """Per-box calibrated constants for the I8 placement model.

    Mirrors the symbols in ``docs/I8_PLACEMENT_MODEL.md``:
      - ``num_devices`` (M): participating GPU count
      - ``num_banks`` (B): host NUMA memory banks
      - ``xfer_speed`` / ``xfer_lat``: per-device PCIe ingest cost parameters
      - ``numa_speed`` / ``numa_lat``: per-bank egress cost parameters
      - ``bank_contention`` (c_b in [0, 1]): bank serial/parallel contention
      - ``ncf``: NUMA-to-GPU tier multiplier matrix [bank][device]
      - ``a_j``, ``b_j``, ``P_j``: batched grouped-GEMM compute parameters
      - ``recon_overhead``, ``recon_added``: TP collective reconciliation parameters
      - ``evict_cons``: convex per-device eviction penalty table [device][u] (1-indexed rank u)
    """

    num_devices: int = 4
    num_banks: int = 2
    xfer_speed: Tuple[float, ...] = (18.0, 18.0, 22.0, 22.0)   # us per expert (e.g. 2x 5090 + 2x 5080)
    xfer_lat: Tuple[float, ...] = (2.5, 2.5, 3.0, 3.0)
    numa_speed: Tuple[float, ...] = (14.0, 14.0)
    numa_lat: Tuple[float, ...] = (1.5, 1.5)
    bank_contention: Tuple[float, ...] = (0.35, 0.35)
    ncf: Tuple[Tuple[float, ...], ...] = (
        (1.0, 1.0, 1.15, 1.15),
        (1.15, 1.15, 1.0, 1.0),
    )
    a_j: Tuple[float, ...] = (6.0, 6.0, 8.5, 8.5)
    b_j: Tuple[float, ...] = (12.0, 12.0, 15.0, 15.0)
    P_j: Tuple[int, ...] = (4, 4, 4, 4)
    recon_overhead: Tuple[float, ...] = (4.0, 4.0, 4.5, 4.5)
    recon_added: Tuple[float, ...] = (1.2, 1.2, 1.2, 1.2)
    evict_cons_base: float = 3.0


@dataclass(frozen=True)
class I8LayerBreakdown:
    """Term-by-term breakdown of ``T(j[.])`` for a single MoE layer."""

    prep: float
    device_makespan: float
    bank_egress: float
    recon: float
    place_cons_total: float
    evict_cons_total: float
    uncached_fetches: int
    cached_hits: int

    @property
    def total_cost(self) -> float:
        return (
            self.prep
            + max(self.device_makespan, self.bank_egress)
            + self.recon
            + self.place_cons_total
            + self.evict_cons_total
        )


def compute_device_gemm_cost(c_j: int, a_j: float, b_j: float, p_j: int) -> float:
    """``compute_j(c) = a_j * c + b_j * ceil(c / P_j)`` (I8_PLACEMENT_MODEL.md)."""
    if c_j <= 0:
        return 0.0
    batches = math.ceil(c_j / max(1, p_j))
    return a_j * c_j + b_j * batches


def evaluate_i8_layer_cost(
    activated_experts: Sequence[int],
    assignment: Sequence[int],
    cached_gpu: Mapping[Tuple[int, int], bool],
    expert_numa_bank: Mapping[int, int],
    hw: I8HardwareProfile,
    place_cons: Optional[Mapping[Tuple[int, int], float]] = None,
    subprep_per_expert: Optional[Mapping[int, float]] = None,
) -> I8LayerBreakdown:
    """Evaluate ``T(j[.])`` exactly as defined in ``docs/I8_PLACEMENT_MODEL.md``.

    Args:
        activated_experts: Length-N sequence of expert IDs selected by the router.
        assignment: Length-N sequence where ``assignment[k]`` in ``0..M-1`` is the
            GPU device on which ``activated_experts[k]`` is fetched and computed.
        cached_gpu: Mapping ``(expert_id, device_id) -> bool`` indicating whether
            expert ``i`` is already resident in GPU ``j``'s cache.
        expert_numa_bank: Mapping ``expert_id -> bank_id`` in ``0..B-1``.
        hw: Calibrated ``I8HardwareProfile`` constants.
        place_cons: Optional ``(expert_id, device_id) -> float`` future-token
            placement consequence (may be negative for hot experts).
        subprep_per_expert: Optional ``expert_id -> float`` NVMe-to-RAM staging time.
    """
    if len(activated_experts) != len(assignment):
        raise ValueError("activated_experts and assignment must have identical length")

    prep = 0.0
    if subprep_per_expert:
        prep = sum(float(subprep_per_expert.get(exp_id, 0.0)) for exp_id in activated_experts)

    # Group assigned experts by device j
    device_groups: Dict[int, List[int]] = {j: [] for j in range(hw.num_devices)}
    for exp_id, dev_j in zip(activated_experts, assignment):
        device_groups[dev_j].append(exp_id)

    participating = [j for j, grp in device_groups.items() if grp]
    if not participating:
        return I8LayerBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    # 1. Device makespan: R_dest_j = sum_{i in G_j} subxfer(i) + compute_j(c_j)
    r_dest: List[float] = []
    uncached_per_device: Dict[int, int] = {j: 0 for j in range(hw.num_devices)}
    cached_hits = 0

    for j in participating:
        grp = device_groups[j]
        xfer_sum = 0.0
        for exp_id in grp:
            if cached_gpu.get((exp_id, j), False):
                cached_hits += 1
            else:
                uncached_per_device[j] += 1
                bank_b = expert_numa_bank.get(exp_id, exp_id % hw.num_banks)
                ncf_mult = hw.ncf[bank_b][j]
                xfer_sum += ncf_mult * hw.xfer_speed[j] + hw.xfer_lat[j]
        comp_j = compute_device_gemm_cost(len(grp), hw.a_j[j], hw.b_j[j], hw.P_j[j])
        r_dest.append(xfer_sum + comp_j)

    device_makespan = max(r_dest)

    # 2. Bank egress: max_b raw_sum_b * (c_b + (1 - c_b) / g_b)
    raw_sum_bank: Dict[int, float] = {b: 0.0 for b in range(hw.num_banks)}
    devices_per_bank: Dict[int, set[int]] = {b: set() for b in range(hw.num_banks)}

    for exp_id, dev_j in zip(activated_experts, assignment):
        if not cached_gpu.get((exp_id, dev_j), False):
            bank_b = expert_numa_bank.get(exp_id, exp_id % hw.num_banks)
            ncf_mult = hw.ncf[bank_b][dev_j]
            raw_sum_bank[bank_b] += ncf_mult * hw.numa_speed[bank_b] + hw.numa_lat[bank_b]
            devices_per_bank[bank_b].add(dev_j)

    bank_egress = 0.0
    for b in range(hw.num_banks):
        g_b = len(devices_per_bank[b])
        if g_b > 0 and raw_sum_bank[b] > 0.0:
            c_b = hw.bank_contention[b]
            floor_b = raw_sum_bank[b] * (c_b + (1.0 - c_b) / float(g_b))
            if floor_b > bank_egress:
                bank_egress = floor_b

    # 3. Reconciliation: max_{j in P} recon_overhead[j] + sum_{j in P} recon_added[j]
    recon = max(hw.recon_overhead[j] for j in participating) + sum(
        hw.recon_added[j] for j in participating
    )

    # 4. Consequences: place_cons_total + convex evict_cons_total
    place_cons_total = 0.0
    if place_cons:
        for exp_id, dev_j in zip(activated_experts, assignment):
            if not cached_gpu.get((exp_id, dev_j), False):
                place_cons_total += float(place_cons.get((exp_id, dev_j), 0.0))

    # evict_cons[j, u] is non-decreasing in rank u (convex cumulative sum in n_j)
    evict_cons_total = 0.0
    for j in participating:
        n_j = uncached_per_device[j]
        for u in range(1, n_j + 1):
            evict_cons_total += hw.evict_cons_base * float(u)

    total_uncached = sum(uncached_per_device.values())
    return I8LayerBreakdown(
        prep=prep,
        device_makespan=device_makespan,
        bank_egress=bank_egress,
        recon=recon,
        place_cons_total=place_cons_total,
        evict_cons_total=evict_cons_total,
        uncached_fetches=total_uncached,
        cached_hits=cached_hits,
    )


def solve_i8_greedy_assignment(
    activated_experts: Sequence[int],
    cached_gpu: Mapping[Tuple[int, int], bool],
    expert_numa_bank: Mapping[int, int],
    hw: I8HardwareProfile,
) -> Tuple[List[int], I8LayerBreakdown]:
    """Deterministic lowest-index tie-broken assignment minimizing I8 layer cost."""
    n = len(activated_experts)
    # Exact enumeration when M^N <= 4096 (e.g. M=4, N<=6)
    if hw.num_devices ** n <= 4096:
        best_assign: Optional[List[int]] = None
        best_breakdown: Optional[I8LayerBreakdown] = None
        total_combos = hw.num_devices ** n
        for idx in range(total_combos):
            cand: List[int] = []
            rem = idx
            for _ in range(n):
                cand.append(rem % hw.num_devices)
                rem //= hw.num_devices
            brk = evaluate_i8_layer_cost(activated_experts, cand, cached_gpu, expert_numa_bank, hw)
            if best_breakdown is None or brk.total_cost < best_breakdown.total_cost - 1e-9:
                best_assign = cand
                best_breakdown = brk
        assert best_assign is not None and best_breakdown is not None
        return best_assign, best_breakdown

    # Greedy prefix fallback for larger N
    current: List[int] = []
    for k, exp_id in enumerate(activated_experts):
        best_j = 0
        best_cost = float("inf")
        for j in range(hw.num_devices):
            trial = current + [j]
            brk = evaluate_i8_layer_cost(
                activated_experts[: k + 1], trial, cached_gpu, expert_numa_bank, hw
            )
            if brk.total_cost < best_cost - 1e-9:
                best_cost = brk.total_cost
                best_j = j
        current.append(best_j)
    final_brk = evaluate_i8_layer_cost(activated_experts, current, cached_gpu, expert_numa_bank, hw)
    return current, final_brk


# ---------------------------------------------------------------------------
# 2. Semantic Domain-Affinity Classifier & Queue Reordering
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: Mapping[str, Tuple[str, ...]] = {
    "systems_cpp_cuda": (
        "cuda", "nvcc", "pcie", "numa", "kernel", "warp", "gemm", "fp8", "nvfp4",
        "c++", "cmake", "io_uring", "nccl", "sm120", "gguf",
    ),
    "formal_math_lean4": (
        "lean 4", "lean4", "theorem", "lemma", "mathlib", "byzantine", "proof",
        "banach", "eigenvalue", "convex", "makespan", "laplacian",
    ),
    "python_ml": (
        "pytorch", "tensor", "autograd", "qlora", "safetensors", "tokenizer",
        "vllm", "fastapi", "uvicorn", "transformer",
    ),
    "security_audit": (
        "cve", "ssrf", "injection", "sanitizer", "fuzz", "seccomp", "audit",
        "multi-krum", "sandbox", "vulnerability",
    ),
    "scientific_sim": (
        "hamiltonian", "rk4", "pde", "navier-stokes", "spectroscopy", "exoplanet",
        "lattice", "molecular", "symplectic",
    ),
}


def classify_prompt_domain(prompt: str) -> str:
    """Classify a prompt into an expert-affinity domain using deterministic keyword scoring."""
    lower = prompt.lower()
    best_domain = "general"
    best_hits = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > best_hits:
            best_hits = hits
            best_domain = domain
    return best_domain


@dataclass(frozen=True)
class QueuedChatRequest:
    """A pending OpenAI-compatible chat request waiting for dispatch to LayerStoRm."""

    request_id: str
    prompt: str
    domain: str = field(default="")

    def resolved_domain(self) -> str:
        return self.domain if self.domain else classify_prompt_domain(self.prompt)


def reorder_queue_by_domain_affinity(
    requests: Sequence[QueuedChatRequest],
    preferred_domains: Sequence[str] = (),
) -> List[QueuedChatRequest]:
    """Group requests by semantic domain while preserving FIFO order within each domain.

    Domains listed in ``preferred_domains`` (matching the node's hot expert cache)
    are scheduled first, followed by remaining domains ordered by first appearance.
    """
    buckets: Dict[str, List[QueuedChatRequest]] = {}
    appearance_order: List[str] = []
    for req in requests:
        dom = req.resolved_domain()
        if dom not in buckets:
            buckets[dom] = []
            appearance_order.append(dom)
        buckets[dom].append(req)

    ordered_domains: List[str] = []
    for dom in preferred_domains:
        if dom in buckets and dom not in ordered_domains:
            ordered_domains.append(dom)
    for dom in appearance_order:
        if dom not in ordered_domains:
            ordered_domains.append(dom)

    out: List[QueuedChatRequest] = []
    for dom in ordered_domains:
        out.extend(buckets[dom])
    return out


def simulate_queue_i8_cost(
    requests: Sequence[QueuedChatRequest],
    hw: Optional[I8HardwareProfile] = None,
    experts_per_domain: int = 6,
) -> Dict[str, float]:
    """Simulate cumulative I8 placement cost across a sequence of requests.

    Each domain activates a characteristic subset of ``experts_per_domain``
    experts. After each request step, the GPU cache retains the just-assigned
    ``(expert_id, device_j)`` pairs on each device up to per-device cache capacity.
    """
    hw_prof = hw or I8HardwareProfile()
    domain_list = list(DOMAIN_KEYWORDS.keys()) + ["general"]
    domain_experts: Dict[str, Tuple[int, ...]] = {
        dom: tuple(d_idx * 10 + k for k in range(experts_per_domain))
        for d_idx, dom in enumerate(domain_list)
    }
    expert_numa_bank: Dict[int, int] = {
        exp_id: exp_id % hw_prof.num_banks
        for exps in domain_experts.values()
        for exp_id in exps
    }

    cached_gpu: Dict[Tuple[int, int], bool] = {}
    total_cost = 0.0
    total_evict = 0.0
    total_xfer_floor = 0.0
    total_hits = 0
    total_uncached = 0

    for req in requests:
        dom = req.resolved_domain()
        active_exps = domain_experts.get(dom, domain_experts["general"])
        assign, brk = solve_i8_greedy_assignment(active_exps, cached_gpu, expert_numa_bank, hw_prof)
        total_cost += brk.total_cost
        total_evict += brk.evict_cons_total
        total_xfer_floor += max(brk.device_makespan, brk.bank_egress)
        total_hits += brk.cached_hits
        total_uncached += brk.uncached_fetches

        # Update cache residency to reflect the experts currently loaded on each GPU
        cached_gpu = {(exp_id, dev_j): True for exp_id, dev_j in zip(active_exps, assign)}

    total_activations = max(1, total_hits + total_uncached)
    return {
        "num_requests": float(len(requests)),
        "total_i8_cost_us": round(total_cost, 3),
        "total_xfer_floor_us": round(total_xfer_floor, 3),
        "total_evict_cost_us": round(total_evict, 3),
        "cache_hit_rate": round(total_hits / float(total_activations), 4),
        "uncached_fetches": float(total_uncached),
    }


# ---------------------------------------------------------------------------
# 3. Determinism Environment Check & Optional Federated Swarm Bridge
# ---------------------------------------------------------------------------

def verify_determinism_env_contract(environ: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    """Check the run-to-run determinism contract from ``tools/serve_determinism_probe.py``."""
    env = environ if environ is not None else os.environ
    det_ep = env.get("LAYERSTORM_DETERMINISTIC_EP_COMBINE", "0") == "1"
    spec_gov_off = env.get("LS_SPEC_GOVERNOR", "0") == "0"
    pci_bus_order = env.get("CUDA_DEVICE_ORDER", "") == "PCI_BUS_ID"
    return {
        "deterministic_ep_combine": det_ep,
        "spec_governor_disabled": spec_gov_off,
        "pci_bus_id_order": pci_bus_order,
        "verifiable_ready": det_ep and spec_gov_off,
    }


def probe_local_layerstorm(base_url: str, timeout_s: float = 3.0) -> Dict[str, object]:
    """Query a local LayerStoRm ``python/cli/serve.py`` HTTP endpoint (``/health`` & ``/v1/models``)."""
    clean = base_url.rstrip("/")
    result: Dict[str, object] = {"base_url": clean, "healthy": False, "models": []}
    try:
        with urllib.request.urlopen(f"{clean}/health", timeout=timeout_s) as resp:
            result["healthy"] = resp.status == 200
        with urllib.request.urlopen(f"{clean}/v1/models", timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            data = payload.get("data", [])
            result["models"] = [str(m.get("id", "")) for m in data if isinstance(m, dict)]
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        result["error"] = str(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LayerStoRm I8 domain-affinity simulator and optional federated swarm bridge."
    )
    parser.add_argument(
        "--simulate-i8",
        action="store_true",
        help="Run the offline I8 placement cost comparison (FIFO vs domain-affinity routing).",
    )
    parser.add_argument(
        "--layerstorm-url",
        default="http://127.0.0.1:8000",
        help="Local LayerStoRm python/cli/serve.py base URL.",
    )
    parser.add_argument(
        "--model-name",
        default="glm-5.3-flash",
        help="Served model identifier (e.g. glm-5.3-flash, glm-5.2, deepseek-v4-flash).",
    )
    parser.add_argument(
        "--preferred-domains",
        default="systems_cpp_cuda,formal_math_lean4",
        help="Comma-separated domain tags cached on this LayerStoRm node.",
    )
    args = parser.parse_args()

    if args.simulate_i8:
        sample_queue = [
            QueuedChatRequest("r1", "Write a CUDA SM120 grouped GEMM kernel with PCIe prefetch"),
            QueuedChatRequest("r2", "Prove Banach fixed-point contraction in Lean 4 with Mathlib"),
            QueuedChatRequest("r3", "Audit this FastAPI endpoint for SSRF and SQL injection"),
            QueuedChatRequest("r4", "Optimize NUMA bank egress contention with io_uring and libnuma"),
            QueuedChatRequest("r5", "Formalize Multi-Krum Byzantine resilience theorem in Lean 4"),
            QueuedChatRequest("r6", "Run a seccomp syscall filter audit on untrusted sandbox code"),
            QueuedChatRequest("r7", "Tune NVFP4 KV cache quantization on RTX 5090 PCIe 5.0"),
            QueuedChatRequest("r8", "Prove spectral gap bound for discrete Hodge Laplacian in Lean 4"),
        ]
        pref = [d.strip() for d in args.preferred_domains.split(",") if d.strip()]
        reordered = reorder_queue_by_domain_affinity(sample_queue, preferred_domains=pref)
        fifo_stats = simulate_queue_i8_cost(sample_queue)
        aff_stats = simulate_queue_i8_cost(reordered)
        reduction_pct = (
            100.0
            * (fifo_stats["total_i8_cost_us"] - aff_stats["total_i8_cost_us"])
            / max(1e-9, fifo_stats["total_i8_cost_us"])
        )
        report = {
            "fifo_queue": fifo_stats,
            "domain_affinity_queue": aff_stats,
            "i8_total_cost_reduction_pct": round(reduction_pct, 2),
        }
        print(json.dumps(report, indent=2))
        return 0

    env_status = verify_determinism_env_contract()
    probe = probe_local_layerstorm(args.layerstorm_url)
    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "preferred_domains": [d.strip() for d in args.preferred_domains.split(",") if d.strip()],
                "determinism_contract": env_status,
                "layerstorm_probe": probe,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

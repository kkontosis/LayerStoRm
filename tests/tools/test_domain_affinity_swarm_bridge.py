"""Unit tests for tools/domain_affinity_swarm_bridge.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.domain_affinity_swarm_bridge import (  # noqa: E402
    I8HardwareProfile,
    QueuedChatRequest,
    classify_prompt_domain,
    compute_device_gemm_cost,
    evaluate_i8_layer_cost,
    reorder_queue_by_domain_affinity,
    simulate_queue_i8_cost,
     verify_determinism_env_contract,
)


def test_compute_device_gemm_batch_step_structure() -> None:
    """Verify compute_j(c) = a_j*c + b_j*ceil(c/P_j) steps at multiples of P_j."""
    assert compute_device_gemm_cost(0, a_j=6.0, b_j=12.0, p_j=4) == 0.0
    # 1..4 experts fit in 1 batch (+12.0 fixed overhead)
    assert compute_device_gemm_cost(4, a_j=6.0, b_j=12.0, p_j=4) == 4 * 6.0 + 12.0
    # 5th expert opens a 2nd batch (+24.0 fixed overhead)
    assert compute_device_gemm_cost(5, a_j=6.0, b_j=12.0, p_j=4) == 5 * 6.0 + 24.0


def test_i8_cached_gpu_eliminates_subxfer_egress_and_eviction() -> None:
    """Cached experts (cached_gpu[i, j] = True) incur zero transfer and zero eviction."""
    hw = I8HardwareProfile()
    experts = (0, 1, 2, 3)
    assign = (0, 0, 1, 1)
    banks = {0: 0, 1: 0, 2: 1, 3: 1}

    uncached = evaluate_i8_layer_cost(experts, assign, cached_gpu={}, expert_numa_bank=banks, hw=hw)
    fully_cached = evaluate_i8_layer_cost(
        experts,
        assign,
        cached_gpu={(0, 0): True, (1, 0): True, (2, 1): True, (3, 1): True},
        expert_numa_bank=banks,
        hw=hw,
    )

    assert uncached.uncached_fetches == 4
    assert fully_cached.uncached_fetches == 0
    assert fully_cached.cached_hits == 4
    assert fully_cached.bank_egress == 0.0
    assert fully_cached.evict_cons_total == 0.0
    assert fully_cached.total_cost < uncached.total_cost


def test_domain_affinity_reordering_lowers_cumulative_i8_cost() -> None:
    """Grouping interleaved requests by domain increases cache hits and lowers I8 cost."""
    interleaved = [
        QueuedChatRequest("1", "Optimize CUDA SM120 grouped GEMM kernel over PCIe"),
        QueuedChatRequest("2", "Prove Banach contraction theorem in Lean 4 with Mathlib"),
        QueuedChatRequest("3", "Tune NUMA bank egress with libnuma and nvcc"),
        QueuedChatRequest("4", "Formalize Multi-Krum Byzantine proof in Lean 4"),
        QueuedChatRequest("5", "Debug FP8 and NVFP4 GGUF kernel on RTX 5090"),
        QueuedChatRequest("6", "Prove Hodge Laplacian spectral gap lemma in Lean 4"),
    ]
    reordered = reorder_queue_by_domain_affinity(
        interleaved, preferred_domains=("systems_cpp_cuda", "formal_math_lean4")
    )

    fifo_metrics = simulate_queue_i8_cost(interleaved)
    affinity_metrics = simulate_queue_i8_cost(reordered)

    assert classify_prompt_domain(interleaved[0].prompt) == "systems_cpp_cuda"
    assert classify_prompt_domain(interleaved[1].prompt) == "formal_math_lean4"
    assert affinity_metrics["cache_hit_rate"] > fifo_metrics["cache_hit_rate"]
    assert affinity_metrics["total_evict_cost_us"] < fifo_metrics["total_evict_cost_us"]
    assert affinity_metrics["total_i8_cost_us"] < fifo_metrics["total_i8_cost_us"]


def test_verify_determinism_env_contract() -> None:
    """Verify detection of LAYERSTORM_DETERMINISTIC_EP_COMBINE=1 and LS_SPEC_GOVERNOR=0."""
    ready = verify_determinism_env_contract(
        {
            "LAYERSTORM_DETERMINISTIC_EP_COMBINE": "1",
            "LS_SPEC_GOVERNOR": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
    )
    assert ready["verifiable_ready"] is True

    not_ready = verify_determinism_env_contract(
        {"LAYERSTORM_DETERMINISTIC_EP_COMBINE": "0", "LS_SPEC_GOVERNOR": "1"}
    )
    assert not_ready["verifiable_ready"] is False

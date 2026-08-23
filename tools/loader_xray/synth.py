"""Synthetic data generator for the loader x-ray (goal 6 — CPU-only validation).

Generates a paired (model JSONL, perf_trace CSV) where the "real" timings are
produced from the model inputs by a KNOWN ground-truth law with injected biases:

  * a per-device transfer SCALE (e.g. device 1's PCIe link is 1.4x slower than
    the calibration assumed),
  * a per-device per-expert COMPUTE cost (us/expert) the dump doesn't carry,
  * a BANK-CONTENTION factor: when g devices pull the same bank, the bank wall is
    draw * (c_true + (1-c_true)/g) — the §3.3 open question. c_true < 1 means the
    channel shares better than the strict-serial sum the current model assumes.
  * additive noise.

The model JSONL is exactly what the C++ LS_LOADER_SHADOW_DUMP would write (same
schema). The perf_trace CSV is exactly the LS_PERF_TRACE_OUT format, with the
MoE-phase markers + kDetected + kDmaGpu stages laid out so the joiner recovers
total/transfer_wall/compute/per-device metrics. The whole point: run join ->
analyze -> train on this and assert the framework RECOVERS c_true and the device
scales.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# perf_trace stage ids (mirror joiner.STAGE).
S_DETECTED = 7
S_DMA = 10
S_ENTER = 12
S_FETCH_ISSUED = 13
S_FIN_ENTER = 14
S_FIN_EXIT = 15


@dataclass
class GroundTruth:
    M: int = 2
    B: int = 2
    # per-device transfer scale applied to subxfer when producing the real wall.
    dev_xfer_scale: List[float] = field(default_factory=lambda: [1.0, 1.4])
    # per-device per-expert real compute cost (us/expert).
    dev_compute_a: List[float] = field(default_factory=lambda: [120.0, 150.0])
    # per-bank solo egress (us/expert) — the source-side rate. Made LARGE so the
    # source-side bank channel is the binding wall: that is what makes the
    # contention coefficient identifiable from the transfer wall (if the device
    # roofline always bound, c would be unobservable).
    bank_egress: List[float] = field(default_factory=lambda: [557.0, 760.0])
    # per-device base subxfer rate (us/expert) the calibration "knows" — kept SMALL
    # relative to bank egress so the bank floor dominates the device roofline.
    dev_subxfer: List[float] = field(default_factory=lambda: [90.0, 90.0])
    contention_c: float = 0.55  # TRUE shared-channel coefficient (<1 = sub-serial)
    recon_us: float = 40.0
    prep_us: float = 0.0
    noise_us: float = 4.0  # gaussian std on each real metric
    seed: int = 1234


def _gen_request(rng: random.Random, gt: GroundTruth, N: int) -> dict:
    """Generate one synthetic solve request + a *plausible* balanced assignment."""
    experts = []
    # round-robin assignment to keep devices roughly balanced (mirrors the e%tp baseline).
    for i in range(N):
        bank = rng.randrange(gt.B)
        j = i % gt.M
        cached = [0] * gt.M  # all uncached (cold) — exercises the transfer terms
        subxfer = [gt.dev_subxfer[d] for d in range(gt.M)]
        experts.append({
            "bank": bank,
            "j": j,
            # Stage-1: the executed e%tp index (oj) and expert_idx, as the C++ dump now
            # emits. Here the synthetic assignment IS e%tp (j==i%M), so oj==j (the
            # synth has no solver/executed divergence) but the fields exercise the
            # offline _assignment_for('executed') path + schema.
            "oj": j,
            "expert_idx": i,
            "egress_us": gt.bank_egress[bank],
            "cached": cached,
            "subxfer_us": subxfer,
        })
    return {"N": N, "M": gt.M, "B": gt.B, "experts": experts}


def _model_predicted(req: dict, gt: GroundTruth) -> dict:
    """What the CURRENT (uncorrected) model would predict — fills the dump's
    predicted sub-terms exactly as the C++ solver would (barrier roofline, strict
    serial bank sum, NO compute since dump has none, recon constant)."""
    M, B = req["M"], req["B"]
    dev = [0.0] * M
    for e in req["experts"]:
        j = e["j"]
        if not e["cached"][j]:
            dev[j] += e["subxfer_us"][j]
    device_makespan = max(dev) if dev else 0.0
    bank = [0.0] * B
    for e in req["experts"]:
        if not e["cached"][e["j"]]:
            bank[e["bank"]] += e["egress_us"]
    bank_egress = max(bank) if bank else 0.0
    recon = gt.recon_us
    prep = gt.prep_us
    total = prep + max(device_makespan, bank_egress) + recon
    return {
        "device_makespan_us": device_makespan,
        "bank_egress_us": bank_egress,
        "recon_us": recon,
        "place_us": 0.0,
        "evict_us": 0.0,
        "prep_us": prep,
        "predicted_us": total,
    }


def _real_timings(req: dict, gt: GroundTruth, rng: random.Random) -> Tuple[dict, dict]:
    """Produce the TRUE real timings via the injected ground-truth law.
    Returns (real_metrics, per_device_walls)."""
    M, B = req["M"], req["B"]
    # per-device real transfer wall = scale_d * sum(subxfer on d).
    dev_xfer = [0.0] * M
    dev_count = [0] * M
    for e in req["experts"]:
        j = e["j"]
        if not e["cached"][j]:
            dev_xfer[j] += gt.dev_xfer_scale[j] * e["subxfer_us"][j]
        dev_count[j] += 1
    # per-bank real egress wall under the TRUE contention law.
    # contention degree g_b = # distinct devices fetching from bank b.
    bank_draw = [0.0] * B
    bank_devs: List[set] = [set() for _ in range(B)]
    for e in req["experts"]:
        if not e["cached"][e["j"]]:
            bank_draw[e["bank"]] += e["egress_us"]
            bank_devs[e["bank"]].add(e["j"])
    bank_wall = 0.0
    for b in range(B):
        g = max(1, len(bank_devs[b]))
        eff = bank_draw[b] * (gt.contention_c + (1.0 - gt.contention_c) / g)
        bank_wall = max(bank_wall, eff)
    # real transfer wall = max over (per-device transfer roofline straggler, bank wall).
    dev_straggler = max(dev_xfer) if dev_xfer else 0.0
    transfer_wall = max(dev_straggler, bank_wall) + rng.gauss(0, gt.noise_us)
    transfer_wall = max(1.0, transfer_wall)
    # real compute = straggler device's per-expert compute * its count.
    compute = max(
        gt.dev_compute_a[d] * dev_count[d] for d in range(M)
    ) + rng.gauss(0, gt.noise_us)
    compute = max(1.0, compute)
    recon = gt.recon_us + rng.gauss(0, gt.noise_us / 2)
    total = gt.prep_us + transfer_wall + compute + recon
    # per-device real fetch walls (for per-device fits): scale_d * sum subxfer_d.
    per_dev = {d: max(0.5, dev_xfer[d] + rng.gauss(0, gt.noise_us / 2)) for d in range(M) if dev_count[d]}
    real = {
        "total_us": total,
        "transfer_wall_us": transfer_wall,
        "compute_us": compute,
        "recon_us": recon,
        "dma_total_us": sum(dev_xfer),
        "n_transfers": sum(1 for e in req["experts"] if not e["cached"][e["j"]]),
    }
    return real, per_dev


def generate(
    n_layers: int = 200,
    N_range: Tuple[int, int] = (4, 8),
    gt: Optional[GroundTruth] = None,
    transfer_cmd_seq: Optional[int] = None,
) -> Tuple[List[str], List[str], GroundTruth]:
    """Generate (model_jsonl_lines, trace_csv_lines, ground_truth).

    cmd_seq is a unique per-record id; perf_trace markers reference it. Returns the
    raw text lines so the caller can write files or feed the parsers in-memory.

    ``transfer_cmd_seq``: when set, the kDetected/kDmaGpu (transfer-stage) rows are
    written with THIS cmd_seq instead of the real one — reproducing the §5 KNOWN GAP
    (the C++ transfer engine stamps cmd_seq=0 on those rows). Pass 0 to exercise the
    Stage-3 interval re-key (the MoE markers still carry the real cmd_seq).
    """
    gt = gt or GroundTruth()
    rng = random.Random(gt.seed)
    model_lines: List[str] = []
    trace_lines: List[str] = ["ns,stage,gpu,cmd_seq,key,token"]
    clock = 0  # monotonically advancing ns clock shared across cmds (steady_clock-like)

    for layer in range(n_layers):
        cmd_seq = layer + 1
        N = rng.randint(*N_range)
        req = _gen_request(rng, gt, N)
        pred = _model_predicted(req, gt)
        real, per_dev = _real_timings(req, gt, rng)

        # ── model JSONL record (matches C++ LS_LOADER_SHADOW_DUMP schema) ──
        dev_counts = [0] * req["M"]
        for e in req["experts"]:
            dev_counts[e["j"]] += 1
        rec = {
            "cmd_seq": cmd_seq,
            "layer_idx": layer,
            "N": N,
            "M": req["M"],
            "B": req["B"],
            "experts": req["experts"],
            "j": [e["j"] for e in req["experts"]],
            "dev_counts": dev_counts,
            **pred,
            "exact": True,
        }
        model_lines.append(json.dumps(rec))

        # ── perf_trace CSV markers reconstructing the real timings ──
        # Layout: kMoeEnter -> kMoeFetchIssued (after a small prep gap) ->
        #   per-transfer kDetected at transfer_wall end (+ kDmaGpu pure dma) ->
        #   kMoeFinalizeEnter -> kMoeFinalizeExit (compute window). The joiner
        #   derives total = exit-enter, transfer_wall = last_detect-fetch_issued,
        #   compute = exit-finalize_enter.
        enter = clock
        prep_gap = int(gt.prep_us * 1000)
        fetch_issued = enter + prep_gap + 200  # +200ns issue overhead
        # transfer-stage rows use transfer_cmd_seq when set (the §5 gap), else real.
        tcs = transfer_cmd_seq if transfer_cmd_seq is not None else cmd_seq
        trace_lines.append(f"{enter},{S_ENTER},0,{cmd_seq},{layer},0")
        trace_lines.append(f"{fetch_issued},{S_FETCH_ISSUED},0,{cmd_seq},{layer},0")
        # per-device detect walls: place each device's last kDetected at its real wall.
        last_detect = fetch_issued
        for d, wall in per_dev.items():
            det_ns = fetch_issued + int(wall * 1000)
            trace_lines.append(f"{det_ns},{S_DETECTED},{d},{tcs},{layer},0")
            last_detect = max(last_detect, det_ns)
        # ensure the overall last_detect reflects transfer_wall (bank may dominate).
        tw_ns = fetch_issued + int(real["transfer_wall_us"] * 1000)
        if tw_ns > last_detect:
            # an extra detect representing the bank-bound straggler transfer.
            straggler_dev = max(per_dev, key=per_dev.get) if per_dev else 0
            trace_lines.append(f"{tw_ns},{S_DETECTED},{straggler_dev},{tcs},{layer},0")
            last_detect = tw_ns
        # pure-GPU DMA proxy records (token = elapsed ns).
        for d in per_dev:
            trace_lines.append(
                f"{last_detect},{S_DMA},{d},{tcs},{layer},{int(per_dev[d]*1000)}"
            )
        finalize_enter = last_detect + 100
        # recon has no dedicated perf_trace marker today (folded into the finalize/
        # compute window). The finalize window therefore spans compute + recon, so
        # the joiner's compute_us = compute + recon (one combined "post-fetch" cost).
        finalize_exit = finalize_enter + int((real["compute_us"] + real["recon_us"]) * 1000)
        trace_lines.append(f"{finalize_enter},{S_FIN_ENTER},0,{cmd_seq},{layer},0")
        trace_lines.append(f"{finalize_exit},{S_FIN_EXIT},0,{cmd_seq},{layer},0")
        clock = finalize_exit + 1000  # gap before next cmd

    return model_lines, trace_lines, gt


def write_files(model_path: str, trace_path: str, **kw) -> GroundTruth:
    ml, tl, gt = generate(**kw)
    with open(model_path, "w") as fh:
        fh.write("\n".join(ml) + "\n")
    with open(trace_path, "w") as fh:
        fh.write("\n".join(tl) + "\n")
    return gt


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="Generate synthetic loader x-ray data.")
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--trace-out", required=True)
    ap.add_argument("--layers", type=int, default=200)
    args = ap.parse_args()
    gt = write_files(args.model_out, args.trace_out, n_layers=args.layers)
    print(f"wrote synthetic data; ground-truth contention_c={gt.contention_c} "
          f"dev_xfer_scale={gt.dev_xfer_scale} dev_compute_a={gt.dev_compute_a}")

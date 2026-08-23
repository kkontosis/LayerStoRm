"""Offline joiner: merge the model JSONL dump (LS_LOADER_SHADOW_DUMP) with the
perf_trace CSV (LS_PERF_TRACE_OUT) into a unified per-(cmd_seq,layer) dataset.

JOIN KEY
--------
The model record is keyed by ``(cmd_seq, layer_idx)``. In perf_trace the MoE-phase
markers carry ``cmd_seq`` = the MoE command's cmd_seq, and ``key`` packs the
expert as ``(layer_idx<<16)|expert_idx`` (so ``key>>16`` == layer_idx for the
MoE-phase records, whose ``key`` field is set to ``p.layer_idx``). We therefore
join on ``cmd_seq`` and cross-check ``layer_idx``.

PERF_TRACE STAGE -> REAL METRIC MAPPING
---------------------------------------
The CSV has columns: ns, stage, gpu, cmd_seq, key, token. Stage enum (perf_trace.h):

    kMoeEnter         = handle_fetch_and_run_moe entered (command drained)
    kMoeFetchIssued   = residency+decider+eviction done, all H2D issued
    kMoeFinalizeEnter = quiescent -> begin finalize (compute) pass
    kMoeFinalizeExit  = finalize kernels enqueued (compute launched)
    kDetected         = a transfer's cudaEventQuery == ready (per transfer)
    kDmaGpu           = pure GPU memcpy time; `token` field = elapsed NANOSECONDS
                        from cudaEventElapsedTime (NOT a timestamp)

Per (cmd_seq, layer) we derive these REAL metrics (all microseconds):

    total_us        = ns[kMoeFinalizeExit] - ns[kMoeEnter]
                      (full daemon-side MoE-layer wall: enter -> compute launched)
    transfer_wall_us= ns[last kDetected for this cmd_seq] - ns[kMoeFetchIssued]
                      (issue all H2D -> last transfer detected ready; the real
                       analogue of max(device_makespan, bank_egress) — the fetch wall)
    compute_us      = ns[kMoeFinalizeExit] - ns[kMoeFinalizeEnter]
                      (finalize/compute enqueue window; real analogue of compute share)
    recon_us        = (reconciliation is folded into finalize; not separately marked
                       today) -> left None unless a future marker provides it. The
                       analyzer treats None as "no ground truth for this term".
    dma_total_us    = sum of kDmaGpu token (ns) for this cmd_seq, /1000
                      (pure GPU DMA time, contention-free sum)
    n_transfers     = count of kDetected for this cmd_seq
    per_device_detect_us[gpu] = (last kDetected ns on that gpu) - kMoeFetchIssued ns
                      (real per-device fetch wall — the makespan straggler view)

Rows missing kMoeEnter or kMoeFinalizeExit are dropped (incomplete) and reported.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Stage enum values — must track perf_trace.h order.
STAGE = {
    "kCmdIssued": 0, "kCmdDrained": 1, "kEnsureEnter": 2, "kResolveExit": 3,
    "kEnqueueInline": 4, "kEnqueueStaged": 5, "kDispatchStaged": 6, "kDetected": 7,
    "kReadyPosted": 8, "kReadyReceived": 9, "kDmaGpu": 10, "kPollTick": 11,
    "kMoeEnter": 12, "kMoeFetchIssued": 13, "kMoeFinalizeEnter": 14,
    "kMoeFinalizeExit": 15,
}


@dataclass
class TraceAgg:
    enter_ns: Optional[int] = None
    fetch_issued_ns: Optional[int] = None
    finalize_enter_ns: Optional[int] = None
    finalize_exit_ns: Optional[int] = None
    last_detect_ns: Optional[int] = None
    detect_count: int = 0
    dma_ns_sum: int = 0
    last_detect_per_gpu: Dict[int, int] = field(default_factory=dict)
    layer_idx: Optional[int] = None  # from kMoeEnter key field


def parse_trace_csv(path: str) -> Dict[int, TraceAgg]:
    """Aggregate the perf_trace CSV by cmd_seq."""
    aggs: Dict[int, TraceAgg] = defaultdict(TraceAgg)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            ns = int(r["ns"])
            stage = int(r["stage"])
            gpu = int(r["gpu"])
            cmd_seq = int(r["cmd_seq"])
            key = int(r["key"])
            token = int(r["token"])
            a = aggs[cmd_seq]
            if stage == STAGE["kMoeEnter"]:
                a.enter_ns = ns
                a.layer_idx = key  # MoE markers set key = layer_idx
            elif stage == STAGE["kMoeFetchIssued"]:
                a.fetch_issued_ns = ns
            elif stage == STAGE["kMoeFinalizeEnter"]:
                a.finalize_enter_ns = ns
            elif stage == STAGE["kMoeFinalizeExit"]:
                a.finalize_exit_ns = ns
            elif stage == STAGE["kDetected"]:
                a.detect_count += 1
                if a.last_detect_ns is None or ns > a.last_detect_ns:
                    a.last_detect_ns = ns
                prev = a.last_detect_per_gpu.get(gpu)
                if prev is None or ns > prev:
                    a.last_detect_per_gpu[gpu] = ns
            elif stage == STAGE["kDmaGpu"]:
                a.dma_ns_sum += token  # token = elapsed ns (not a timestamp)
    return aggs


def rekey_transfer_rows(path: str) -> Dict[int, dict]:
    """Stage-3 (§5 KNOWN GAP): re-key the transfer-stage perf_trace rows to the real
    MoE ``cmd_seq``.

    The transfer engine records ``kDetected`` / ``kDmaGpu`` / ``kDispatchStaged`` with
    ``cmd_seq`` UNAVAILABLE at that layer (kDetected: 0; kDmaGpu: overloaded to carry
    src_numa+1; kDispatchStaged: pending-count) — so ``transfer_wall`` could not be
    joined per layer (the original analysis floored it). The daemon processes **one
    progressive MoE at a time** (``handle_fetch_and_run_moe`` rejects a second while one
    is active), so each MoE command owns a **non-overlapping** wall-clock window
    ``[kMoeEnter, kMoeFinalizeExit]``. We therefore re-key each transfer row to the MoE
    ``cmd_seq`` whose window contains the transfer's ``ns`` — a sound interval-join,
    no hot-path change needed.

    Returns ``{cmd_seq: {transfer_wall_us, n_transfers, dma_total_us,
    per_device_detect_us, fetch_issued_ns}}`` for the MoE cmds that own ≥1 transfer.
    A transfer outside every window (a prefetch outside any MoE) is dropped (counted).
    """
    # Pass 1: per-cmd MoE windows + the boundary markers we need for the wall.
    aggs = parse_trace_csv(path)
    windows = []  # (enter_ns, exit_ns, cmd_seq, fetch_issued_ns)
    for cmd_seq, a in aggs.items():
        if a.enter_ns is not None and a.finalize_exit_ns is not None:
            windows.append((a.enter_ns, a.finalize_exit_ns, cmd_seq, a.fetch_issued_ns))
    windows.sort()
    starts = [w[0] for w in windows]

    def owner(ns: int):
        import bisect
        i = bisect.bisect_right(starts, ns) - 1
        if 0 <= i < len(windows) and windows[i][0] <= ns <= windows[i][1]:
            return windows[i]
        return None

    out: Dict[int, dict] = {}
    n_orphan = 0

    def slot(cmd_seq, fetch_issued_ns):
        d = out.setdefault(cmd_seq, {
            "last_detect_ns": None, "detect_count": 0, "dma_ns_sum": 0,
            "last_detect_per_gpu": {}, "fetch_issued_ns": fetch_issued_ns})
        return d

    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            stage = int(r["stage"])
            if stage not in (STAGE["kDetected"], STAGE["kDmaGpu"]):
                continue
            ns = int(r["ns"]); gpu = int(r["gpu"]); token = int(r["token"])
            w = owner(ns)
            if w is None:
                n_orphan += 1
                continue
            d = slot(w[2], w[3])
            if stage == STAGE["kDetected"]:
                d["detect_count"] += 1
                if d["last_detect_ns"] is None or ns > d["last_detect_ns"]:
                    d["last_detect_ns"] = ns
                prev = d["last_detect_per_gpu"].get(gpu)
                if prev is None or ns > prev:
                    d["last_detect_per_gpu"][gpu] = ns
            elif stage == STAGE["kDmaGpu"]:
                d["dma_ns_sum"] += token
    # Reduce to the per-cmd metrics.
    res: Dict[int, dict] = {}
    for cmd_seq, d in out.items():
        fi = d["fetch_issued_ns"]
        tw = None
        per_dev = {}
        if fi is not None and d["last_detect_ns"] is not None:
            tw = (d["last_detect_ns"] - fi) / 1000.0
            for gpu, ns in d["last_detect_per_gpu"].items():
                per_dev[gpu] = (ns - fi) / 1000.0
        res[cmd_seq] = {
            "transfer_wall_us": tw,
            "n_transfers": d["detect_count"],
            "dma_total_us": d["dma_ns_sum"] / 1000.0,
            "per_device_detect_us": per_dev,
            "_n_orphan_transfers": n_orphan,
        }
    return res


def parse_model_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def join(model_jsonl: str, trace_csv: str, rekey_transfers: bool = True):
    """Return (rows, stats). Each row = model record + 'real' sub-dict.

    ``rekey_transfers`` (default True): apply the Stage-3 interval re-key so transfer
    rows (whose ``cmd_seq`` is 0/overloaded) contribute ``transfer_wall_us`` /
    ``per_device_detect_us`` / ``dma_total_us`` per layer. When the transfer rows DO
    carry a real cmd_seq (synthetic data, or a future C++ stamp) the direct aggregate
    already has them, so the re-key only *fills* what the direct join left None.
    """
    model = parse_model_jsonl(model_jsonl)
    trace = parse_trace_csv(trace_csv)
    rekey = rekey_transfer_rows(trace_csv) if rekey_transfers else {}
    rows: List[dict] = []
    n_dropped = 0
    n_layer_mismatch = 0
    n_rekeyed = 0
    for m in model:
        cmd_seq = int(m["cmd_seq"])
        a = trace.get(cmd_seq)
        if a is None or a.enter_ns is None or a.finalize_exit_ns is None:
            n_dropped += 1
            continue
        if a.layer_idx is not None and int(a.layer_idx) != int(m["layer_idx"]):
            n_layer_mismatch += 1  # report but keep (cmd_seq is the strong key)
        real: Dict[str, Optional[float]] = {}
        real["total_us"] = (a.finalize_exit_ns - a.enter_ns) / 1000.0
        if a.fetch_issued_ns is not None and a.last_detect_ns is not None:
            real["transfer_wall_us"] = (a.last_detect_ns - a.fetch_issued_ns) / 1000.0
        else:
            real["transfer_wall_us"] = None
        if a.finalize_enter_ns is not None:
            real["compute_us"] = (a.finalize_exit_ns - a.finalize_enter_ns) / 1000.0
        else:
            real["compute_us"] = None
        real["recon_us"] = None  # no dedicated marker yet (folded into finalize)
        real["dma_total_us"] = a.dma_ns_sum / 1000.0
        real["n_transfers"] = a.detect_count
        per_dev = {}
        if a.fetch_issued_ns is not None:
            for gpu, ns in a.last_detect_per_gpu.items():
                per_dev[gpu] = (ns - a.fetch_issued_ns) / 1000.0
        real["per_device_detect_us"] = per_dev
        # Stage-3: fill transfer-derived metrics from the interval re-key when the
        # direct join left them empty (the common case: kDetected has cmd_seq=0).
        rk = rekey.get(cmd_seq)
        if rk is not None:
            used = False
            if real.get("transfer_wall_us") is None and rk.get("transfer_wall_us") is not None:
                real["transfer_wall_us"] = rk["transfer_wall_us"]; used = True
            if not real.get("per_device_detect_us") and rk.get("per_device_detect_us"):
                real["per_device_detect_us"] = rk["per_device_detect_us"]; used = True
            if not real.get("dma_total_us") and rk.get("dma_total_us"):
                real["dma_total_us"] = rk["dma_total_us"]; used = True
            if not real.get("n_transfers") and rk.get("n_transfers"):
                real["n_transfers"] = rk["n_transfers"]; used = True
            if used:
                n_rekeyed += 1
        row = dict(m)
        row["real"] = real
        rows.append(row)
    stats = {
        "n_model_records": len(model),
        "n_joined": len(rows),
        "n_dropped_incomplete": n_dropped,
        "n_layer_mismatch": n_layer_mismatch,
        "n_trace_cmds": len(trace),
        "n_rekeyed_transfers": n_rekeyed,
    }
    return rows, stats


def write_joined(rows: List[dict], out_path: str) -> None:
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="Join loader model JSONL with perf_trace CSV.")
    ap.add_argument("--model", required=True, help="LS_LOADER_SHADOW_DUMP JSONL path")
    ap.add_argument("--trace", required=True, help="LS_PERF_TRACE_OUT CSV path")
    ap.add_argument("--out", required=True, help="output joined JSONL")
    args = ap.parse_args()
    rows, stats = join(args.model, args.trace)
    write_joined(rows, args.out)
    print(json.dumps(stats, indent=2))

"""Vectorized memmap reader for fixed-geometry EPMR routing streams (P11
Stage-0, spec/reports/EXPOSED_BYTE_CALCULUS.md §3-P11(6)).

dataset.read_routing_records() is a per-row Python loop over a whole-file
read_bytes() — fine for one test file, unusable over the EPM-5 corpus's 78
run_XXX.bin files (13 GB). The corpus streams are uniform (every record:
n_rows == J, n_experts == E, topk == K — verified over the corpus), so a
run file is exactly N fixed-stride records and maps onto one numpy
structured dtype: zero-copy column views over an np.memmap.

Geometry is taken from record 0 and then ENFORCED over every record
(magic/version/rows/E/K vectorized comparison; per-record layer order
spot-checked first + last). Any deviation raises — there is deliberately
no slow-path fallback here; a mixed-geometry stream must go through
dataset.read_routing_records() where per-record shapes are honored.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterator

import numpy as np

from .dataset import EPMR_MAGIC, EPM_VERSION, _ROUTE_HDR


def _record_dtype(rows: int, n_experts: int, topk: int) -> np.dtype:
    """Packed structured dtype of one EPMR record (matches
    src/speculation/epm_dump.h byte-for-byte; numpy packs fields
    consecutively, no padding — asserted against the header struct)."""
    row_dt = np.dtype([("layer", "<u4"),
                       ("logits", "<f2", (n_experts,)),
                       ("ids", "<i4", (topk,)),
                       ("w", "<f4", (topk,))])
    rec_dt = np.dtype([("magic", "<u4"), ("ver", "<u4"), ("seq", "<u8"),
                       ("pos", "<u4"), ("rows", "<u4"), ("ne", "<u4"),
                       ("k", "<u4"), ("rowdata", row_dt, (rows,))])
    expect = _ROUTE_HDR.size + rows * (4 + 2 * n_experts + 8 * topk)
    if rec_dt.itemsize != expect:
        raise AssertionError(f"EPMR dtype itemsize {rec_dt.itemsize} != "
                             f"expected {expect} (packing changed?)")
    return rec_dt


@dataclasses.dataclass
class RunView:
    """Zero-copy columns over one memmapped EPMR run file.

    seq/pos are per-record; top_ids/top_w are [N, rows, K] views; full
    logits are materialized on demand (they are ~93% of the bytes — a
    caller that only needs top-8 never touches them)."""
    path: Path
    recs: np.ndarray            # memmap viewed as the record dtype [N]
    layers: np.ndarray          # int32 [rows] — layer order (uniform)

    @property
    def n_records(self) -> int:
        return len(self.recs)

    @property
    def seq(self) -> np.ndarray:        # uint64 [N]
        return self.recs["seq"]

    @property
    def pos(self) -> np.ndarray:        # uint32 [N]
        return self.recs["pos"]

    @property
    def top_ids(self) -> np.ndarray:    # int32 [N, rows, K]
        return self.recs["rowdata"]["ids"]

    @property
    def top_w(self) -> np.ndarray:      # float32 [N, rows, K]
        return self.recs["rowdata"]["w"]

    def logits(self, sel) -> np.ndarray:
        """Full pre-top-k router logits for the selected records:
        float16 [len(sel), rows, E] (copied out of the memmap)."""
        return np.ascontiguousarray(self.recs["rowdata"]["logits"][sel])


def open_run(path: str | Path) -> RunView:
    """Memmap one epm_routing.bin / run_XXX.bin with uniform geometry."""
    path = Path(path)
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    if len(raw) < _ROUTE_HDR.size:
        raise ValueError(f"{path}: too small for an EPMR header")
    magic, ver, _seq, _pos, rows, n_e, k = _ROUTE_HDR.unpack_from(
        raw[:_ROUTE_HDR.size].tobytes())
    if magic != EPMR_MAGIC or ver != EPM_VERSION:
        raise ValueError(f"{path}: bad EPMR header (magic {magic:#x}, "
                         f"version {ver})")
    if rows == 0 or n_e == 0 or k == 0:
        raise ValueError(f"{path}: degenerate geometry rows={rows} "
                         f"E={n_e} K={k}")
    rec_dt = _record_dtype(rows, n_e, k)
    if len(raw) % rec_dt.itemsize:
        raise ValueError(f"{path}: size {len(raw)} not a multiple of the "
                         f"record size {rec_dt.itemsize} (mixed geometry "
                         f"or truncated tail — use "
                         f"dataset.read_routing_records)")
    recs = raw.view(rec_dt)
    # Uniform-geometry enforcement over EVERY record (vectorized, cheap).
    for field, want in (("magic", EPMR_MAGIC), ("ver", EPM_VERSION),
                        ("rows", rows), ("ne", n_e), ("k", k)):
        bad = np.nonzero(recs[field] != want)[0]
        if bad.size:
            raise ValueError(f"{path}: record {bad[0]}: {field} = "
                             f"{recs[field][bad[0]]} != {want} (mixed "
                             f"geometry — use "
                             f"dataset.read_routing_records)")
    layers = np.asarray(recs["rowdata"]["layer"][0], np.int64).astype(np.int32)
    for probe in (len(recs) - 1,):
        if not np.array_equal(recs["rowdata"]["layer"][probe].astype(np.int32),
                              layers):
            raise ValueError(f"{path}: record {probe}: layer order differs "
                             f"from record 0")
    return RunView(path=path, recs=recs, layers=layers)


def iter_sequences(view: RunView) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (seq_id, record_indices) per sequence, indices sorted by
    token_pos with LAST-wins dedup on duplicate positions (mirrors
    assemble_shards' dict-overwrite join). Sequences yield in ascending
    seq_id order."""
    seq = view.seq
    pos = view.pos
    order = np.lexsort((np.arange(len(seq)), pos, seq))  # stable in file order
    seq_o = seq[order]
    bounds = np.nonzero(np.diff(seq_o))[0] + 1
    for chunk in np.split(order, bounds):
        p = pos[chunk]
        # last occurrence wins per position: chunk is pos-sorted with file
        # order as tiebreak, so keep the LAST entry of each equal-pos run.
        keep = np.ones(len(chunk), bool)
        keep[:-1] = p[1:] != p[:-1]
        yield int(seq[chunk[0]]), chunk[keep]

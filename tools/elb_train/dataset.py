"""EPM-1 dataset assembly: raw engine dumps -> training shards (Phase 29).

Pipeline (spec/MoE-SpeQ_NOTES.md §7 EPM-1):

  engine (dump-ON) writes, per dump directory ("run unit"):
    epm_blocks.bin    EPMB records — draft per-backbone-layer residual
                      hiddens [gamma, L, H] BF16 + draft ids + raw c_k,
                      one per DSpark block (src/speculation/epm_dump.h).
    epm_routing.bin   EPMR records — per decode position, every routed
                      layer's 256 pre-top-k router logits FP16 + top-8
                      ids/weights (the verify/AR feed's routing).
    manifest.jsonl    orchestrator block metadata (accepted length, the
                      actually-verified tokens, c_k, provenance) —
                      python/orchestrator/epm_manifest.py.

  assemble_shards() joins the three on (seq_id, anchor_pos): block k's
  label rows are the routing records at positions anchor_pos + k for
  k in [0, gamma). Under sequential early-stop verification only the
  verified prefix (accepted + 1 feeds, capped at gamma) has labels —
  label_mask marks which positions are complete. Output: shard_%05d.npz
  + shards_index.json in the output directory.

Corpus discipline: engine seq_ids restart per run, so ONE dump directory
must hold ONE engine run. Point assemble_shards at a corpus ROOT holding
several run directories (any subdirectory containing epm_blocks.bin is a
run unit; the root itself counts if it has one) — blocks get globally
unique sequence keys `run_idx << 48 | seq_id`.

Shard arrays (B blocks; G = max gamma; L draft layers; H hidden;
J routed layers; E experts; K top-k):
  features_bf16   uint16  [B, G, L, H]   raw BF16 bits (torch:
                                         .view(torch.bfloat16) — numpy has
                                         no bf16 dtype; kept bit-exact)
  feature_mask    bool    [B, G]         k < gamma(block)
  labels_logits   float16 [B, G, J, E]   0 where label_mask is False
  labels_top_ids  int32   [B, G, J, K]   -1 pad
  labels_top_w    float32 [B, G, J, K]   0 pad
  label_mask      bool    [B, G]         position has ALL J routed layers
  seq_key         uint64  [B]            run-unique sequence key (split key)
  seq_id          uint64  [B]            raw engine seq id
  run_idx         int32   [B]
  block_idx       uint32  [B]            daemon per-seq counter
  anchor_pos      uint32  [B]
  anchor_token    uint32  [B]
  gamma           int32   [B]
  accepted_len    int32   [B]            -1 = no manifest entry joined
  tokens_match    bool    [B]            manifest verify==draft prefix
  draft_tokens    int32   [B, G]         -1 pad
  confidences     float32 [B, G]         NaN pad / NaN when head off
  moe_layers      int32   [J]            absolute routed layer ids

INV-EPM-DATA: train/held-out splits are by SEQUENCE (seq_key), never by
block — adjacent blocks of one sequence share context and leak. Use
sequence_split() and filter with iterate_blocks(..., seq_keys=...).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

EPMB_MAGIC = 0x424D5045  # "EPMB" little-endian
EPMR_MAGIC = 0x524D5045  # "EPMR" little-endian
EPM_VERSION = 1

_BLOCK_HDR = struct.Struct("<IIQIIIIIII")  # magic, ver, seq, blk, anchor,
#                                            anchor_tok, gamma, L, H, has_conf
_ROUTE_HDR = struct.Struct("<IIQIIII")     # magic, ver, seq, pos, rows, E, K


# ── Raw record readers ───────────────────────────────────────────────────────

@dataclasses.dataclass
class BlockRecord:
    seq_id: int
    block_idx: int
    anchor_pos: int
    anchor_token: int
    gamma: int
    n_layers: int
    hidden: int
    draft_ids: np.ndarray        # int32 [gamma]
    confidences: np.ndarray | None  # float32 [gamma] or None
    hiddens_bf16: np.ndarray     # uint16 [gamma, n_layers, hidden]


@dataclasses.dataclass
class RoutingRecord:
    seq_id: int
    token_pos: int
    n_experts: int
    topk: int
    layers: np.ndarray           # int32 [rows]
    logits: np.ndarray           # float16 [rows, n_experts]
    top_ids: np.ndarray          # int32 [rows, topk]
    top_w: np.ndarray            # float32 [rows, topk]


def read_block_records(path: str | Path) -> Iterator[BlockRecord]:
    """Iterate EPMB records from an epm_blocks.bin file."""
    data = Path(path).read_bytes()
    off = 0
    while off + _BLOCK_HDR.size <= len(data):
        (magic, ver, seq, blk, anchor, anchor_tok, gamma, n_layers, hidden,
         has_conf) = _BLOCK_HDR.unpack_from(data, off)
        if magic != EPMB_MAGIC or ver != EPM_VERSION:
            raise ValueError(
                f"{path}: bad EPMB record at offset {off} "
                f"(magic {magic:#x}, version {ver})")
        off += _BLOCK_HDR.size
        n_hid = gamma * n_layers * hidden
        body = gamma * 4 + (gamma * 4 if has_conf else 0) + n_hid * 2
        if off + body > len(data):
            raise ValueError(f"{path}: truncated EPMB record at offset "
                             f"{off - _BLOCK_HDR.size}")
        ids = np.frombuffer(data, np.int32, gamma, off).copy()
        off += gamma * 4
        conf = None
        if has_conf:
            conf = np.frombuffer(data, np.float32, gamma, off).copy()
            off += gamma * 4
        hid = np.frombuffer(data, np.uint16, n_hid, off).copy()
        off += n_hid * 2
        yield BlockRecord(seq, blk, anchor, anchor_tok, gamma, n_layers,
                          hidden, ids, conf,
                          hid.reshape(gamma, n_layers, hidden))
    if off != len(data):
        raise ValueError(f"{path}: {len(data) - off} trailing bytes "
                         f"(truncated record?)")


def read_routing_records(path: str | Path) -> Iterator[RoutingRecord]:
    """Iterate EPMR records from an epm_routing.bin file."""
    data = Path(path).read_bytes()
    off = 0
    while off + _ROUTE_HDR.size <= len(data):
        magic, ver, seq, pos, rows, n_e, k = _ROUTE_HDR.unpack_from(data, off)
        if magic != EPMR_MAGIC or ver != EPM_VERSION:
            raise ValueError(
                f"{path}: bad EPMR record at offset {off} "
                f"(magic {magic:#x}, version {ver})")
        off += _ROUTE_HDR.size
        if off + rows * (4 + n_e * 2 + k * 8) > len(data):
            raise ValueError(f"{path}: truncated EPMR record at offset "
                             f"{off - _ROUTE_HDR.size}")
        layers = np.empty(rows, np.int32)
        logits = np.empty((rows, n_e), np.float16)
        top_ids = np.empty((rows, k), np.int32)
        top_w = np.empty((rows, k), np.float32)
        for r in range(rows):
            layers[r] = struct.unpack_from("<I", data, off)[0]
            off += 4
            logits[r] = np.frombuffer(data, np.float16, n_e, off)
            off += n_e * 2
            top_ids[r] = np.frombuffer(data, np.int32, k, off)
            off += k * 4
            top_w[r] = np.frombuffer(data, np.float32, k, off)
            off += k * 4
        yield RoutingRecord(seq, pos, n_e, k, layers, logits, top_ids, top_w)
    if off != len(data):
        raise ValueError(f"{path}: {len(data) - off} trailing bytes "
                         f"(truncated record?)")


def read_manifest(path: str | Path) -> list[dict]:
    """Read manifest.jsonl (one dict per line; blank lines skipped)."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ── Run-unit discovery + shard assembly ──────────────────────────────────────

def find_run_units(root: str | Path) -> list[Path]:
    """Dump run units under `root`: any directory (root included) that
    directly contains epm_blocks.bin. Sorted for deterministic run_idx."""
    root = Path(root)
    units = []
    if (root / "epm_blocks.bin").is_file():
        units.append(root)
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if (sub / "epm_blocks.bin").is_file():
            units.append(sub)
    return units


def make_seq_key(run_idx: int, seq_id: int) -> int:
    """Globally-unique sequence key across run units (INV-EPM-DATA split
    key). Engine seq ids stay well below 2**48 in practice; fail loudly
    if not rather than silently colliding."""
    if seq_id >= (1 << 48):
        raise ValueError(f"seq_id {seq_id} >= 2**48 — seq_key packing "
                         f"would collide")
    return (run_idx << 48) | seq_id


def assemble_shards(dump_root: str | Path, out_dir: str | Path,
                    blocks_per_shard: int = 1024, *,
                    run_idx_offset: int = 0,
                    append: bool = False,
                    store_logits: bool = True,
                    compress: bool = False,
                    block_filter=None) -> dict:
    """Join raw dumps + manifest into shard_%05d.npz files + an index.

    Returns the index dict (also written to <out_dir>/shards_index.json):
      {"shards": [{"path", "num_blocks", "seq_keys": [...]}, ...],
       "moe_layers": [...], "n_experts": E, "topk": K,
       "n_draft_layers": L, "hidden": H, "max_gamma": G,
       "num_blocks", "num_sequences", "runs": [unit paths],
       "logits_stored": bool}

    EPM-5 incremental-assembly options (disk-bounded corpora, see
    studies/epm/epm5-corpus/PROGRESS.md Phase A):

    - ``run_idx_offset``: added to each unit's local run index for the
      seq_key packing — lets per-batch assemblies into ONE ``out_dir``
      keep globally-unique sequence keys (INV-EPM-DATA). Callers must
      pass non-overlapping offsets (a duplicate offset against an
      appended index raises).
    - ``append``: merge into an existing ``out_dir`` index — geometry
      (dims, gamma, moe_layers, topk, logits_stored) must match; shard
      numbering continues; num_blocks/num_sequences/runs accumulate.
    - ``store_logits=False``: leave labels_logits all-zero (full shape,
      so every loader keeps working) and stamp ``logits_stored: false``
      in the index. BCE targets come from labels_top_ids; only the
      KL-distillation loss needs the raw logits (train.py fails closed
      on the flag). Pair with ``compress`` — zero arrays deflate away.
    - ``compress``: np.savez_compressed instead of np.savez.
    - ``block_filter``: optional ``(seq_id, anchor_pos) -> bool``; blocks
      for which it returns False are DROPPED (routing/manifest for the
      dropped positions simply never join). Used to truncate degenerate
      greedy-repetition tails per sequence (collect_epm5: forced-length
      generation on bounded-answer prompts loops after the natural end;
      the clean prefix is kept, the poisoned tail excluded). Dropped
      blocks are not counted; the dedup check still runs before the
      filter (a duplicate is an error regardless of the filter).
    """
    dump_root = Path(dump_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    units = find_run_units(dump_root)
    if not units:
        raise FileNotFoundError(f"no run units (epm_blocks.bin) under "
                                f"{dump_root}")

    prev: dict | None = None
    if append:
        idx_path = out_dir / "shards_index.json"
        if idx_path.is_file():
            with open(idx_path, encoding="utf-8") as f:
                prev = json.load(f)
            prev_offsets = {r[1] for r in prev.get("run_units", [])}
            new_offsets = {run_idx_offset + i for i in range(len(units))}
            if prev_offsets & new_offsets:
                raise ValueError(
                    f"append: run_idx range {sorted(new_offsets)} collides "
                    f"with already-assembled offsets — seq_keys would "
                    f"collide (INV-EPM-DATA)")

    # Load everything, keyed per run unit.
    blocks: list[tuple[int, BlockRecord]] = []   # (run_idx, record)
    routing: dict[tuple[int, int, int], RoutingRecord] = {}
    manifest: dict[tuple[int, int, int], dict] = {}
    moe_layers: set[int] = set()
    n_experts = topk = None
    for local_idx, unit in enumerate(units):
        run_idx = run_idx_offset + local_idx
        seen_keys = set()
        for br in read_block_records(unit / "epm_blocks.bin"):
            key = (br.seq_id, br.anchor_pos)
            if key in seen_keys:
                raise ValueError(
                    f"{unit}: duplicate block (seq {br.seq_id}, anchor "
                    f"{br.anchor_pos}) — one run per dump dir (seq ids "
                    f"restart per engine run)")
            seen_keys.add(key)
            if block_filter is not None and not block_filter(
                    br.seq_id, br.anchor_pos):
                continue    # truncated (degenerate-tail) block — dropped
            blocks.append((run_idx, br))
        rpath = unit / "epm_routing.bin"
        if rpath.is_file():
            for rr in read_routing_records(rpath):
                routing[(run_idx, rr.seq_id, rr.token_pos)] = rr
                moe_layers.update(int(l) for l in rr.layers)
                if n_experts is None:
                    n_experts, topk = rr.n_experts, rr.topk
                elif (n_experts, topk) != (rr.n_experts, rr.topk):
                    raise ValueError(f"{unit}: inconsistent routing dims")
        mpath = unit / "manifest.jsonl"
        if mpath.is_file():
            for e in read_manifest(mpath):
                manifest[(run_idx, e["seq_id"], e["anchor_pos"])] = e

    if not blocks:
        raise ValueError(f"no EPMB blocks under {dump_root}")
    layers_sorted = np.array(sorted(moe_layers), np.int32)
    j = len(layers_sorted)
    layer_slot = {int(l): i for i, l in enumerate(layers_sorted)}
    if n_experts is None:  # blocks without any routing dump
        n_experts, topk = 0, 0
    max_gamma = max(br.gamma for _, br in blocks)
    n_layers = blocks[0][1].n_layers
    hidden = blocks[0][1].hidden

    index: dict = {
        "shards": [], "moe_layers": [int(l) for l in layers_sorted],
        "n_experts": int(n_experts), "topk": int(topk),
        "n_draft_layers": int(n_layers), "hidden": int(hidden),
        "max_gamma": int(max_gamma), "num_blocks": len(blocks),
        "num_sequences": len({make_seq_key(r, b.seq_id)
                              for r, b in blocks}),
        "runs": [str(u) for u in units],
        "run_units": [[str(u), run_idx_offset + i]
                      for i, u in enumerate(units)],
        "logits_stored": bool(store_logits),
    }
    shard_number_base = 0
    if prev is not None:
        # Geometry must match exactly — a mixed-geometry corpus would
        # silently mis-join labels (INV-EPM-DATA sibling hazard).
        for field_name in ("moe_layers", "n_experts", "topk",
                          "n_draft_layers", "hidden", "max_gamma",
                          "logits_stored"):
            if prev.get(field_name) != index[field_name]:
                raise ValueError(
                    f"append: index field {field_name!r} mismatch "
                    f"(existing {prev.get(field_name)!r} vs new "
                    f"{index[field_name]!r})")
        shard_number_base = len(prev["shards"])
        index["shards"] = list(prev["shards"])
        index["runs"] = list(prev["runs"]) + index["runs"]
        index["run_units"] = (list(prev.get("run_units", []))
                              + index["run_units"])
        index["num_blocks"] += prev["num_blocks"]

    for shard_i in range(0, len(blocks), blocks_per_shard):
        chunk = blocks[shard_i:shard_i + blocks_per_shard]
        b = len(chunk)
        g = max_gamma
        feats = np.zeros((b, g, n_layers, hidden), np.uint16)
        fmask = np.zeros((b, g), bool)
        logits = np.zeros((b, g, j, n_experts), np.float16)
        tids = np.full((b, g, j, topk), -1, np.int32)
        tw = np.zeros((b, g, j, topk), np.float32)
        lmask = np.zeros((b, g), bool)
        seq_key = np.zeros(b, np.uint64)
        seq_id = np.zeros(b, np.uint64)
        run_arr = np.zeros(b, np.int32)
        blk_idx = np.zeros(b, np.uint32)
        anchor = np.zeros(b, np.uint32)
        anchor_tok = np.zeros(b, np.uint32)
        gamma_arr = np.zeros(b, np.int32)
        accepted = np.full(b, -1, np.int32)
        match = np.zeros(b, bool)
        dtok = np.full((b, g), -1, np.int32)
        conf = np.full((b, g), np.nan, np.float32)

        for i, (run_idx, br) in enumerate(chunk):
            if (br.n_layers, br.hidden) != (n_layers, hidden):
                raise ValueError("inconsistent draft dims across blocks")
            seq_key[i] = make_seq_key(run_idx, br.seq_id)
            seq_id[i] = br.seq_id
            run_arr[i] = run_idx
            blk_idx[i] = br.block_idx
            anchor[i] = br.anchor_pos
            anchor_tok[i] = br.anchor_token
            gamma_arr[i] = br.gamma
            feats[i, :br.gamma] = br.hiddens_bf16
            fmask[i, :br.gamma] = True
            dtok[i, :br.gamma] = br.draft_ids
            if br.confidences is not None:
                conf[i, :br.gamma] = br.confidences
            me = manifest.get((run_idx, br.seq_id, br.anchor_pos))
            if me is not None:
                accepted[i] = me["accepted_len"]
                match[i] = bool(me["tokens_match"])
            for k in range(br.gamma):
                rr = routing.get((run_idx, br.seq_id, br.anchor_pos + k))
                if rr is None:
                    continue
                slots = [layer_slot[int(l)] for l in rr.layers]
                if store_logits:
                    logits[i, k, slots] = rr.logits
                tids[i, k, slots] = rr.top_ids
                tw[i, k, slots] = rr.top_w
                lmask[i, k] = len(set(slots)) == j  # ALL routed layers
        shard_num = shard_number_base + shard_i // blocks_per_shard
        name = f"shard_{shard_num:05d}.npz"
        savez = np.savez_compressed if compress else np.savez
        savez(out_dir / name,
                 features_bf16=feats, feature_mask=fmask,
                 labels_logits=logits, labels_top_ids=tids,
                 labels_top_w=tw, label_mask=lmask,
                 seq_key=seq_key, seq_id=seq_id, run_idx=run_arr,
                 block_idx=blk_idx, anchor_pos=anchor,
                 anchor_token=anchor_tok, gamma=gamma_arr,
                 accepted_len=accepted, tokens_match=match,
                 draft_tokens=dtok, confidences=conf,
                 moe_layers=layers_sorted)
        index["shards"].append({
            "path": name, "num_blocks": b,
            "seq_keys": sorted({int(s) for s in seq_key}),
        })

    # num_sequences over the FULL (possibly appended) corpus — per-shard
    # seq_keys lists are authoritative and offsets guarantee uniqueness.
    index["num_sequences"] = len({
        int(s) for sh in index["shards"] for s in sh["seq_keys"]
    })
    with open(out_dir / "shards_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    return index


def build_prev_membership_sidecars(shard_dir: str | Path) -> dict:
    """Write ``prev_top_ids_%05d.npy`` beside each shard — the b0_prev
    temporal-locality candidate set per block position (EPM-5 hybrid arm,
    studies/epm/epm5-corpus; MoE-SpeQ_NOTES §5c / DECISION.md 'EPM as a
    re-ranker ON TOP of temporal locality').

    prev_top_ids[b, k, j] = the top-K routed experts at the PREVIOUS
    decoded position (global pos anchor_pos+k-1) for layer j:
      - k >= 1: the block's OWN labelled position k-1 (same routing the
        engine computed one step earlier), when label_mask[k-1];
      - k == 0: the sequence-previous block's LAST labelled position
        (its accepted tail) — the causal block-start source.
    -1 pad where the source position is unavailable (sequence start,
    unlabelled predecessor). Identical in content to the baselines.py B0
    'prev' variant (routing records ARE the labels at those positions),
    reconstructed from the shards so no raw epm_routing.bin is needed.

    Idempotent; returns {'shards': N, 'blocks': M, 'k0_covered': frac}.
    """
    shard_dir = Path(shard_dir)
    index = load_index(shard_dir)
    j = len(index["moe_layers"])
    k = int(index["topk"])
    g = int(index["max_gamma"])

    # Pass 1: per-block coordinates + each block's labelled tail [J, K].
    # tail = labels_top_ids at the highest k' with label_mask true.
    order: dict[int, list[tuple[int, int, int]]] = {}  # seq -> [(anchor,sh,row)]
    tails: dict[tuple[int, int], np.ndarray] = {}      # (sh,row) -> [J,K] or None
    for si, sh in enumerate(index["shards"]):
        z = np.load(shard_dir / sh["path"])
        sk = z["seq_key"].astype(np.int64)
        ap = z["anchor_pos"].astype(np.int64)
        lm = z["label_mask"]
        tids = z["labels_top_ids"]
        for r in range(len(sk)):
            order.setdefault(int(sk[r]), []).append((int(ap[r]), si, r))
            labelled = np.nonzero(lm[r])[0]
            tails[(si, r)] = (tids[r, int(labelled[-1])]
                              if labelled.size else None)

    # k=0 source per block = previous (by anchor_pos) block's tail.
    k0_src: dict[tuple[int, int], np.ndarray] = {}
    for sk, blocks in order.items():
        blocks.sort()
        prev_tail = None
        for _ap, si, r in blocks:
            if prev_tail is not None:
                k0_src[(si, r)] = prev_tail
            t = tails[(si, r)]
            if t is not None:
                prev_tail = t

    # Pass 2: assemble per shard (own labels shift for k>=1, join for k=0).
    n_blocks = 0
    k0_cov = 0
    for si, sh in enumerate(index["shards"]):
        z = np.load(shard_dir / sh["path"])
        tids = z["labels_top_ids"]
        lm = z["label_mask"]
        b = tids.shape[0]
        prev = np.full((b, g, j, k), -1, np.int32)
        for r in range(b):
            for kk in range(1, g):
                if lm[r, kk - 1]:
                    prev[r, kk] = tids[r, kk - 1]
            src = k0_src.get((si, r))
            if src is not None:
                prev[r, 0] = src
                k0_cov += 1
            n_blocks += 1
        out = shard_dir / f"prev_top_ids_{si:05d}.npy"
        np.save(out, prev)
    return {"shards": len(index["shards"]), "blocks": n_blocks,
            "k0_covered": (k0_cov / n_blocks) if n_blocks else 0.0}


# ── Iteration + splits + stats ───────────────────────────────────────────────

def load_index(shard_dir: str | Path) -> dict:
    with open(Path(shard_dir) / "shards_index.json", encoding="utf-8") as f:
        return json.load(f)


def sequence_split(seq_keys: list[int], held_out_fraction: float = 0.1,
                   seed: int = 0) -> tuple[list[int], list[int]]:
    """Deterministic SEQUENCE-level train/held-out split (INV-EPM-DATA:
    never split by block/position — adjacent blocks of one sequence share
    context and leak). Hash-bucketed (sha256 of "seed:seq_key"), so
    membership is stable under corpus growth. Returns (train, held_out)."""
    if not 0.0 <= held_out_fraction <= 1.0:
        raise ValueError("held_out_fraction must be in [0, 1]")
    train, held = [], []
    for sk in sorted(set(int(s) for s in seq_keys)):
        h = hashlib.sha256(f"{seed}:{sk}".encode()).digest()
        bucket = int.from_bytes(h[:8], "little") / 2.0**64
        (held if bucket < held_out_fraction else train).append(sk)
    return train, held


def iterate_blocks(shard_dir: str | Path,
                   seq_keys: set[int] | list[int] | None = None,
                   ) -> Iterator[dict]:
    """Training-side iterator: yields one dict per block (all shard arrays
    sliced to the block, gamma-trimmed on axis 0 where applicable). Pass
    `seq_keys` (from sequence_split) to restrict to one split — the
    INV-EPM-DATA-compliant way to build train/held-out streams."""
    shard_dir = Path(shard_dir)
    index = load_index(shard_dir)
    keys = None if seq_keys is None else {int(s) for s in seq_keys}
    for si, sh in enumerate(index["shards"]):
        if keys is not None and not keys.intersection(sh["seq_keys"]):
            continue
        z = np.load(shard_dir / sh["path"])
        side_path = shard_dir / f"prev_top_ids_{si:05d}.npy"
        prev_arr = np.load(side_path) if side_path.is_file() else None
        sel = (np.ones(len(z["seq_key"]), bool) if keys is None else
               np.isin(z["seq_key"].astype(np.int64), list(keys)))
        for i in np.nonzero(sel)[0]:
            gm = int(z["gamma"][i])
            blk = {
                "seq_key": int(z["seq_key"][i]),
                "seq_id": int(z["seq_id"][i]),
                "run_idx": int(z["run_idx"][i]),
                "block_idx": int(z["block_idx"][i]),
                "anchor_pos": int(z["anchor_pos"][i]),
                "anchor_token": int(z["anchor_token"][i]),
                "gamma": gm,
                "accepted_len": int(z["accepted_len"][i]),
                "tokens_match": bool(z["tokens_match"][i]),
                "features_bf16": z["features_bf16"][i, :gm],
                "labels_logits": z["labels_logits"][i, :gm],
                "labels_top_ids": z["labels_top_ids"][i, :gm],
                "labels_top_w": z["labels_top_w"][i, :gm],
                "label_mask": z["label_mask"][i, :gm],
                "draft_tokens": z["draft_tokens"][i, :gm],
                "confidences": z["confidences"][i, :gm],
                "moe_layers": z["moe_layers"],
            }
            if prev_arr is not None:
                blk["prev_top_ids"] = prev_arr[i, :gm]
            yield blk


def dataset_stats(shard_dir: str | Path) -> dict:
    """Corpus statistics over an assembled shard directory: block/sequence
    counts, per-position label coverage (the sequential-early-stop reality:
    only the verified prefix has labels), accepted-length histogram,
    on-disk bytes. The EPM-1 'dataset stats note'."""
    shard_dir = Path(shard_dir)
    index = load_index(shard_dir)
    g = index["max_gamma"]
    labeled = np.zeros(g, np.int64)
    present = np.zeros(g, np.int64)
    acc_hist: dict[int, int] = {}
    n_match = n_manifest = 0
    total = 0
    for sh in index["shards"]:
        z = np.load(shard_dir / sh["path"])
        total += int(sh["num_blocks"])
        labeled += z["label_mask"].sum(axis=0).astype(np.int64)
        present += z["feature_mask"].sum(axis=0).astype(np.int64)
        for a in z["accepted_len"]:
            a = int(a)
            if a >= 0:
                n_manifest += 1
                acc_hist[a] = acc_hist.get(a, 0) + 1
        n_match += int(z["tokens_match"].sum())
    bytes_on_disk = sum((shard_dir / sh["path"]).stat().st_size
                        for sh in index["shards"])
    return {
        "num_blocks": total,
        "num_sequences": index["num_sequences"],
        "num_shards": len(index["shards"]),
        "max_gamma": g,
        "moe_layers": index["moe_layers"],
        "n_experts": index["n_experts"],
        "topk": index["topk"],
        "position_feature_counts": present.tolist(),
        "position_label_counts": labeled.tolist(),
        "position_label_coverage": [
            (labeled[k] / present[k] if present[k] else 0.0)
            for k in range(g)],
        "accepted_len_histogram": {str(k): v
                                   for k, v in sorted(acc_hist.items())},
        "manifest_joined": n_manifest,
        "tokens_match": n_match,
        "bytes_on_disk": bytes_on_disk,
    }

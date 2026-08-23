"""EPM-1 unit tests: tools/elb_train/dataset.py (writer/reader round trip,
manifest join, sequence-level split — INV-EPM-DATA, stats).

Synthetic EPMB/EPMR streams are packed here with the exact little-endian
layouts of src/speculation/epm_dump.h (the C++ side's byte-for-byte
round trip against the real capture path is EpmDump.* in
tests/unit/epm_dump_test.cpp; these tests own the assembly layer).
"""

from __future__ import annotations

import json
import pathlib
import struct
import sys

import numpy as np
import pytest

_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "tools"))

from elb_train import dataset  # noqa: E402

EPMB = 0x424D5045
EPMR = 0x524D5045


# ── Synthetic raw-record packers (mirror src/speculation/epm_dump.h) ─────────

def pack_block(seq, blk, anchor, anchor_tok, gamma, n_layers, hidden,
               ids, conf, hiddens_u16):
    out = struct.pack("<IIQIIIIIII", EPMB, 1, seq, blk, anchor, anchor_tok,
                      gamma, n_layers, hidden, 1 if conf is not None else 0)
    out += np.asarray(ids, np.int32).tobytes()
    if conf is not None:
        out += np.asarray(conf, np.float32).tobytes()
    out += np.asarray(hiddens_u16, np.uint16).tobytes()
    return out


def pack_route(seq, pos, rows):
    """rows: list of (layer, logits_f16, ids, w)."""
    n_e = len(rows[0][1])
    k = len(rows[0][2])
    out = struct.pack("<IIQIIII", EPMR, 1, seq, pos, len(rows), n_e, k)
    for layer, logits, ids, w in rows:
        out += struct.pack("<I", layer)
        out += np.asarray(logits, np.float16).tobytes()
        out += np.asarray(ids, np.int32).tobytes()
        out += np.asarray(w, np.float32).tobytes()
    return out


def make_dump(tmp_path, *, seqs=(1, 2), blocks_per_seq=3, gamma=4,
              n_layers=5, hidden=16, moe_layers=(3, 4, 5), n_experts=32,
              topk=8, label_positions=None, rng_seed=0, with_manifest=True):
    """Write a synthetic run unit (epm_blocks.bin/epm_routing.bin/
    manifest.jsonl). Returns the ground truth as dicts for assertions.
    `label_positions(seq, blk)` -> how many block positions get routing
    records (the sequential-early-stop reality); default = all."""
    rng = np.random.default_rng(rng_seed)
    truth = {"blocks": [], "routes": {}, "manifest": []}
    blocks_bin = b""
    routes_bin = b""
    for seq in seqs:
        for blk in range(blocks_per_seq):
            anchor = 10 + blk * (gamma + 1)
            ids = rng.integers(0, 1000, gamma).astype(np.int32)
            conf = rng.random(gamma).astype(np.float32)
            hid = rng.integers(0, 2**16, (gamma, n_layers, hidden)) \
                     .astype(np.uint16)
            blocks_bin += pack_block(seq, blk, anchor, 7, gamma, n_layers,
                                     hidden, ids, conf, hid)
            truth["blocks"].append(dict(seq=seq, blk=blk, anchor=anchor,
                                        ids=ids, conf=conf, hid=hid))
            n_lab = (gamma if label_positions is None
                     else label_positions(seq, blk))
            for k in range(n_lab):
                rows = []
                for layer in moe_layers:
                    lg = rng.standard_normal(n_experts).astype(np.float16)
                    tid = rng.choice(n_experts, topk, replace=False) \
                             .astype(np.int32)
                    tw = rng.random(topk).astype(np.float32)
                    rows.append((layer, lg, tid, tw))
                routes_bin += pack_route(seq, anchor + k, rows)
                truth["routes"][(seq, anchor + k)] = rows
            if with_manifest:
                entry = {"seq_id": seq, "anchor_pos": anchor,
                         "block_idx": blk, "request_id": 1,
                         "gamma": gamma,
                         "draft_tokens": [int(t) for t in ids],
                         "verify_tokens": [int(t) for t in ids],
                         "tokens_match": True,
                         "accepted_len": min(blk, gamma),
                         "attempted_len": gamma,
                         "confidences": [float(c) for c in conf],
                         "ts": 0.0}
                truth["manifest"].append(entry)
    (tmp_path / "epm_blocks.bin").write_bytes(blocks_bin)
    (tmp_path / "epm_routing.bin").write_bytes(routes_bin)
    if with_manifest:
        with open(tmp_path / "manifest.jsonl", "w") as f:
            for e in truth["manifest"]:
                f.write(json.dumps(e) + "\n")
    return truth


# ── Raw readers ──────────────────────────────────────────────────────────────

class TestRawReaders:
    def test_block_reader_round_trip(self, tmp_path):
        truth = make_dump(tmp_path, seqs=(5,), blocks_per_seq=2)
        recs = list(dataset.read_block_records(tmp_path / "epm_blocks.bin"))
        assert len(recs) == 2
        for rec, t in zip(recs, truth["blocks"]):
            assert rec.seq_id == t["seq"]
            assert rec.block_idx == t["blk"]
            assert rec.anchor_pos == t["anchor"]
            assert rec.anchor_token == 7
            np.testing.assert_array_equal(rec.draft_ids, t["ids"])
            np.testing.assert_array_equal(rec.confidences, t["conf"])
            np.testing.assert_array_equal(rec.hiddens_bf16, t["hid"])

    def test_block_reader_no_confidence(self, tmp_path):
        blob = pack_block(1, 0, 10, 7, 2, 3, 4,
                          np.arange(2, dtype=np.int32), None,
                          np.arange(2 * 3 * 4, dtype=np.uint16))
        (tmp_path / "b.bin").write_bytes(blob)
        recs = list(dataset.read_block_records(tmp_path / "b.bin"))
        assert len(recs) == 1
        assert recs[0].confidences is None
        assert recs[0].hiddens_bf16.shape == (2, 3, 4)

    def test_routing_reader_round_trip(self, tmp_path):
        truth = make_dump(tmp_path, seqs=(1,), blocks_per_seq=1)
        recs = list(
            dataset.read_routing_records(tmp_path / "epm_routing.bin"))
        assert len(recs) == len(truth["routes"])
        for rec in recs:
            rows = truth["routes"][(rec.seq_id, rec.token_pos)]
            np.testing.assert_array_equal(
                rec.layers, [r[0] for r in rows])
            np.testing.assert_array_equal(
                rec.logits, np.stack([r[1] for r in rows]))
            np.testing.assert_array_equal(
                rec.top_ids, np.stack([r[2] for r in rows]))
            np.testing.assert_array_equal(
                rec.top_w, np.stack([r[3] for r in rows]))

    def test_truncated_stream_raises(self, tmp_path):
        truth_blob = pack_block(1, 0, 10, 7, 2, 3, 4,
                                np.arange(2, dtype=np.int32), None,
                                np.arange(24, dtype=np.uint16))
        (tmp_path / "b.bin").write_bytes(truth_blob[:-3])
        with pytest.raises(ValueError, match="trailing|truncated"):
            list(dataset.read_block_records(tmp_path / "b.bin"))

    def test_bad_magic_raises(self, tmp_path):
        (tmp_path / "b.bin").write_bytes(b"\x00" * 64)
        with pytest.raises(ValueError, match="bad EPMB"):
            list(dataset.read_block_records(tmp_path / "b.bin"))


# ── Shard assembly + manifest join ───────────────────────────────────────────

class TestAssembly:
    def test_shard_round_trip_and_manifest_join(self, tmp_path):
        dump = tmp_path / "dump"
        dump.mkdir()
        # Sequential-early-stop reality: block b of each seq has labels for
        # min(b + 1, gamma) positions only.
        truth = make_dump(dump, seqs=(1, 2), blocks_per_seq=3, gamma=4,
                          label_positions=lambda s, b: min(b + 1, 4))
        out = tmp_path / "shards"
        index = dataset.assemble_shards(dump, out, blocks_per_shard=4)
        assert index["num_blocks"] == 6
        assert index["num_sequences"] == 2
        assert index["moe_layers"] == [3, 4, 5]
        assert index["n_experts"] == 32 and index["topk"] == 8
        assert len(index["shards"]) == 2  # 4 + 2

        blocks = list(dataset.iterate_blocks(out))
        assert len(blocks) == 6
        by_key = {(b["seq_id"], b["anchor_pos"]): b for b in blocks}
        for t in truth["blocks"]:
            b = by_key[(t["seq"], t["anchor"])]
            np.testing.assert_array_equal(b["features_bf16"], t["hid"])
            np.testing.assert_array_equal(b["draft_tokens"], t["ids"])
            np.testing.assert_array_equal(b["confidences"], t["conf"])
            # Manifest joined: accepted_len == min(blk, gamma) per make_dump.
            assert b["accepted_len"] == min(t["blk"], 4)
            assert b["tokens_match"] is True
            # Label mask matches the simulated verified prefix.
            expect_mask = [k < min(t["blk"] + 1, 4) for k in range(4)]
            assert b["label_mask"].tolist() == expect_mask
            # Labeled positions carry the exact routing rows.
            for k in range(4):
                if not expect_mask[k]:
                    continue
                rows = truth["routes"][(t["seq"], t["anchor"] + k)]
                for j, (layer, lg, tid, tw) in enumerate(rows):
                    np.testing.assert_array_equal(
                        b["labels_logits"][k, j], lg)
                    np.testing.assert_array_equal(
                        b["labels_top_ids"][k, j], tid)
                    np.testing.assert_array_equal(b["labels_top_w"][k, j], tw)

    def test_incomplete_layer_set_not_labeled(self, tmp_path):
        dump = tmp_path / "dump"
        dump.mkdir()
        make_dump(dump, seqs=(1,), blocks_per_seq=1, gamma=2)
        # Append a routing record missing layer 5 at a position of a new
        # block -> that position must NOT be label-masked.
        rng = np.random.default_rng(1)
        rows = [(l, rng.standard_normal(32).astype(np.float16),
                 np.arange(8, dtype=np.int32),
                 np.ones(8, np.float32)) for l in (3, 4)]  # no layer 5
        with open(dump / "epm_routing.bin", "ab") as f:
            f.write(pack_route(1, 100, rows))
        with open(dump / "epm_blocks.bin", "ab") as f:
            f.write(pack_block(1, 1, 100, 7, 2, 5, 16,
                               np.zeros(2, np.int32), None,
                               np.zeros(2 * 5 * 16, np.uint16)))
        out = tmp_path / "shards"
        dataset.assemble_shards(dump, out)
        blocks = {(b["seq_id"], b["anchor_pos"]): b
                  for b in dataset.iterate_blocks(out)}
        b = blocks[(1, 100)]
        assert not b["label_mask"][0]  # incomplete layer set
        assert not b["label_mask"][1]  # no record at all

    def test_multi_run_units_get_distinct_seq_keys(self, tmp_path):
        for run in ("run_a", "run_b"):
            d = tmp_path / run
            d.mkdir()
            make_dump(d, seqs=(1,), blocks_per_seq=1)
        out = tmp_path / "shards"
        index = dataset.assemble_shards(tmp_path, out)
        assert index["num_blocks"] == 2
        # Same raw seq_id 1 in both runs -> two distinct sequence keys.
        assert index["num_sequences"] == 2
        keys = {b["seq_key"] for b in dataset.iterate_blocks(out)}
        assert len(keys) == 2

    def test_duplicate_block_in_one_run_raises(self, tmp_path):
        dump = tmp_path / "dump"
        dump.mkdir()
        make_dump(dump, seqs=(1,), blocks_per_seq=1)
        blob = (dump / "epm_blocks.bin").read_bytes()
        (dump / "epm_blocks.bin").write_bytes(blob + blob)  # same (seq,anchor)
        with pytest.raises(ValueError, match="duplicate block"):
            dataset.assemble_shards(dump, tmp_path / "shards")

    def test_missing_dump_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            dataset.assemble_shards(tmp_path, tmp_path / "shards")


# ── Sequence-level split (INV-EPM-DATA) ──────────────────────────────────────

class TestSequenceSplit:
    def test_split_is_by_sequence_and_deterministic(self, tmp_path):
        dump = tmp_path / "dump"
        dump.mkdir()
        make_dump(dump, seqs=tuple(range(1, 41)), blocks_per_seq=3)
        out = tmp_path / "shards"
        index = dataset.assemble_shards(dump, out, blocks_per_shard=16)
        all_keys = sorted({k for sh in index["shards"]
                           for k in sh["seq_keys"]})
        train, held = dataset.sequence_split(all_keys, 0.25, seed=3)
        assert sorted(train + held) == all_keys
        assert held  # a 25% split of 40 sequences is nonempty w.h.p.
        # Determinism.
        train2, held2 = dataset.sequence_split(all_keys, 0.25, seed=3)
        assert (train, held) == (train2, held2)
        # Different seed -> (almost surely) different membership.
        _, held3 = dataset.sequence_split(all_keys, 0.25, seed=4)
        assert held3 != held

        # INV-EPM-DATA: every BLOCK of a sequence lands on that sequence's
        # side — no block-level leakage between the two streams.
        train_blocks = list(dataset.iterate_blocks(out, seq_keys=train))
        held_blocks = list(dataset.iterate_blocks(out, seq_keys=held))
        assert len(train_blocks) + len(held_blocks) == index["num_blocks"]
        assert {b["seq_key"] for b in train_blocks}.isdisjoint(
            {b["seq_key"] for b in held_blocks})
        for b in train_blocks:
            assert b["seq_key"] in set(train)
        for b in held_blocks:
            assert b["seq_key"] in set(held)

    def test_split_stable_under_corpus_growth(self, tmp_path):
        # Hash-bucketed membership: adding sequences never flips existing
        # ones (collection can continue while training starts).
        keys = list(range(1, 101))
        train_a, held_a = dataset.sequence_split(keys, 0.2, seed=1)
        train_b, held_b = dataset.sequence_split(keys + [500, 501], 0.2,
                                                 seed=1)
        assert set(held_a).issubset(set(held_b) | set(train_b))
        assert set(held_a) == set(held_b) - {500, 501}

    def test_bad_fraction_raises(self):
        with pytest.raises(ValueError):
            dataset.sequence_split([1, 2], 1.5)


# ── Stats ────────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_counts_and_coverage(self, tmp_path):
        dump = tmp_path / "dump"
        dump.mkdir()
        make_dump(dump, seqs=(1, 2), blocks_per_seq=3, gamma=4,
                  label_positions=lambda s, b: min(b + 1, 4))
        out = tmp_path / "shards"
        dataset.assemble_shards(dump, out, blocks_per_shard=4)
        st = dataset.dataset_stats(out)
        assert st["num_blocks"] == 6
        assert st["num_sequences"] == 2
        assert st["max_gamma"] == 4
        assert st["position_feature_counts"] == [6, 6, 6, 6]
        # label_positions min(b+1, 4) over b in {0,1,2}, twice (two seqs):
        # position 0 labeled in all 6 blocks; pos 1 in 4; pos 2 in 2; pos 3
        # never (accepted never reached it) — the early-stop shape.
        assert st["position_label_counts"] == [6, 4, 2, 0]
        assert st["position_label_coverage"][0] == 1.0
        assert st["position_label_coverage"][3] == 0.0
        # accepted_len histogram: blk 0,1,2 -> accepted 0,1,2 per seq.
        assert st["accepted_len_histogram"] == {"0": 2, "1": 2, "2": 2}
        assert st["manifest_joined"] == 6
        assert st["tokens_match"] == 6
        assert st["bytes_on_disk"] > 0


# ── Split-seed selection procedure (E4 addition, ratified 2026-08-02) ────────

class TestSelectSplitSeed:
    def test_reproduces_procedure_and_constraints(self, tmp_path):
        from elb_train import select_split_seed
        dump = tmp_path / "dump"
        dump.mkdir()
        make_dump(dump, seqs=tuple(range(1, 10)), blocks_per_seq=3)
        out = tmp_path / "shards"
        dataset.assemble_shards(dump, out, blocks_per_shard=8)

        chosen, rows = select_split_seed.sweep(
            out, held_out_fraction=0.25, min_held_sequences=3, n_seeds=30)
        assert chosen is not None
        by_seed = {r["seed"]: r for r in rows}
        # The chosen seed satisfies the eligibility constraints ...
        assert by_seed[chosen]["eligible"]
        assert by_seed[chosen]["held_seqs"] >= 3
        assert by_seed[chosen]["train_seqs"] >= 1
        # ... and no eligible seed sits strictly closer to the target
        # block fraction (tie-break: lowest seed wins).
        d_chosen = abs(by_seed[chosen]["held_block_fraction"] - 0.25)
        for r in rows:
            if not r["eligible"]:
                continue
            d = abs(r["held_block_fraction"] - 0.25)
            assert d > d_chosen or (d == d_chosen and r["seed"] >= chosen)
        # Deterministic re-run.
        chosen2, _ = select_split_seed.sweep(
            out, held_out_fraction=0.25, min_held_sequences=3, n_seeds=30)
        assert chosen2 == chosen

    def test_too_small_corpus_returns_none(self, tmp_path):
        from elb_train import select_split_seed
        dump = tmp_path / "dump"
        dump.mkdir()
        make_dump(dump, seqs=(1, 2), blocks_per_seq=2)
        out = tmp_path / "shards"
        dataset.assemble_shards(dump, out)
        chosen, rows = select_split_seed.sweep(
            out, held_out_fraction=0.25, min_held_sequences=3, n_seeds=10)
        assert chosen is None
        assert all(not r["eligible"] for r in rows)


# ── EPM-5 incremental assembly (append / run_idx_offset / store_logits /
#    compress — disk-bounded corpus, studies/epm/epm5-corpus) ────────────────

class TestIncrementalAssembly:
    def _one_batch(self, tmp_path, name, seqs, seed):
        dump = tmp_path / name
        dump.mkdir()
        make_dump(dump, seqs=seqs, blocks_per_seq=3, gamma=4, rng_seed=seed)
        return dump

    def test_append_accumulates_with_unique_seq_keys(self, tmp_path):
        out = tmp_path / "shards"
        d1 = self._one_batch(tmp_path, "b1", (1, 2), 0)
        d2 = self._one_batch(tmp_path, "b2", (1, 2), 1)  # same engine ids
        i1 = dataset.assemble_shards(d1, out, blocks_per_shard=4,
                                     run_idx_offset=0)
        i2 = dataset.assemble_shards(d2, out, blocks_per_shard=4,
                                     run_idx_offset=1, append=True)
        assert i2["num_blocks"] == i1["num_blocks"] * 2 == 12
        # Engine seq ids repeat across runs; offsets keep keys unique.
        assert i2["num_sequences"] == 4
        keys = {k for sh in i2["shards"] for k in sh["seq_keys"]}
        assert len(keys) == 4
        # Shard numbering continues; every referenced file exists.
        names = [sh["path"] for sh in i2["shards"]]
        assert names == sorted(names) and len(names) == len(set(names))
        for sh in i2["shards"]:
            assert (out / sh["path"]).is_file()
        blocks = list(dataset.iterate_blocks(out))
        assert len(blocks) == 12

    def test_append_offset_collision_raises(self, tmp_path):
        out = tmp_path / "shards"
        d1 = self._one_batch(tmp_path, "b1", (1,), 0)
        d2 = self._one_batch(tmp_path, "b2", (1,), 1)
        dataset.assemble_shards(d1, out, run_idx_offset=3)
        with pytest.raises(ValueError, match="collides"):
            dataset.assemble_shards(d2, out, run_idx_offset=3, append=True)

    def test_append_geometry_mismatch_raises(self, tmp_path):
        out = tmp_path / "shards"
        d1 = self._one_batch(tmp_path, "b1", (1,), 0)
        dataset.assemble_shards(d1, out, run_idx_offset=0)
        d2 = tmp_path / "b2"
        d2.mkdir()
        make_dump(d2, seqs=(1,), blocks_per_seq=3, gamma=4, rng_seed=1,
                  n_experts=16)  # different expert count
        with pytest.raises(ValueError, match="mismatch"):
            dataset.assemble_shards(d2, out, run_idx_offset=1, append=True)

    def test_store_logits_false_zeroes_logits_keeps_top_ids(self, tmp_path):
        out = tmp_path / "shards"
        dump = self._one_batch(tmp_path, "b1", (1, 2), 0)
        idx = dataset.assemble_shards(dump, out, store_logits=False,
                                      compress=True)
        assert idx["logits_stored"] is False
        blocks = list(dataset.iterate_blocks(out))
        assert blocks
        for b in blocks:
            assert not np.any(b["labels_logits"])          # all zero
            assert np.any(b["labels_top_ids"] >= 0)        # labels intact
            assert b["labels_logits"].shape[-1] == idx["n_experts"]

    def test_append_logits_flag_mismatch_raises(self, tmp_path):
        out = tmp_path / "shards"
        d1 = self._one_batch(tmp_path, "b1", (1,), 0)
        d2 = self._one_batch(tmp_path, "b2", (1,), 1)
        dataset.assemble_shards(d1, out, run_idx_offset=0,
                                store_logits=False)
        with pytest.raises(ValueError, match="logits_stored"):
            dataset.assemble_shards(d2, out, run_idx_offset=1, append=True,
                                    store_logits=True)

    def test_compressed_shards_round_trip(self, tmp_path):
        out_c = tmp_path / "shards_c"
        out_p = tmp_path / "shards_p"
        dump = self._one_batch(tmp_path, "b1", (1, 2), 0)
        dataset.assemble_shards(dump, out_c, compress=True)
        dataset.assemble_shards(dump, out_p, compress=False)
        bc = {(b["seq_id"], b["anchor_pos"]): b
              for b in dataset.iterate_blocks(out_c)}
        for b in dataset.iterate_blocks(out_p):
            c = bc[(b["seq_id"], b["anchor_pos"])]
            np.testing.assert_array_equal(b["features_bf16"],
                                          c["features_bf16"])
            np.testing.assert_array_equal(b["labels_top_ids"],
                                          c["labels_top_ids"])


class TestPrevMembershipSidecar:
    def test_prev_reconstructs_b0_prev_semantics(self, tmp_path):
        # Fully-labelled blocks: prev[k>=1] must equal own labels[k-1];
        # prev[0] must equal the sequence-previous block's last label row.
        dump = tmp_path / "dump"
        dump.mkdir()
        make_dump(dump, seqs=(1,), blocks_per_seq=3, gamma=4)
        out = tmp_path / "shards"
        dataset.assemble_shards(dump, out, blocks_per_shard=8)
        info = dataset.build_prev_membership_sidecars(out)
        assert info["blocks"] == 3
        z = np.load(out / "shard_00000.npz")
        prev = np.load(out / "prev_top_ids_00000.npy")
        tids, lm, ap = z["labels_top_ids"], z["label_mask"], z["anchor_pos"]
        order = np.argsort(ap)                             # by decode order
        for pos, r in enumerate(order):
            for k in range(1, prev.shape[1]):
                if lm[r, k - 1]:
                    np.testing.assert_array_equal(prev[r, k], tids[r, k - 1])
            if pos == 0:
                assert np.all(prev[r, 0] == -1)           # no predecessor
            else:
                prev_block = order[pos - 1]
                last = np.nonzero(lm[prev_block])[0][-1]
                np.testing.assert_array_equal(
                    prev[r, 0], tids[prev_block, last])

    def test_sidecar_present_flows_through_iterate_and_store(self, tmp_path):
        dump = tmp_path / "dump"
        dump.mkdir()
        make_dump(dump, seqs=(1, 2), blocks_per_seq=3, gamma=4)
        out = tmp_path / "shards"
        dataset.assemble_shards(dump, out)
        # before building: no prev_top_ids in the block dicts
        b0 = next(dataset.iterate_blocks(out))
        assert "prev_top_ids" not in b0
        dataset.build_prev_membership_sidecars(out)
        b1 = next(dataset.iterate_blocks(out))
        assert "prev_top_ids" in b1
        assert b1["prev_top_ids"].shape[0] == b1["gamma"]


class TestBlockFilterTruncation:
    def test_block_filter_drops_tail_blocks(self, tmp_path):
        dump = tmp_path / "dump"
        dump.mkdir()
        # one seq, blocks at anchors 10, 15, 20 (gamma=4 -> +5 each).
        make_dump(dump, seqs=(1,), blocks_per_seq=3, gamma=4)
        out = tmp_path / "shards"
        # drop the block at anchor >= 20 (the last one)
        idx = dataset.assemble_shards(
            dump, out, block_filter=lambda sid, ap: ap < 20)
        assert idx["num_blocks"] == 2
        anchors = sorted(b["anchor_pos"] for b in dataset.iterate_blocks(out))
        assert max(anchors) < 20

    def test_filter_none_keeps_all(self, tmp_path):
        dump = tmp_path / "dump"
        dump.mkdir()
        make_dump(dump, seqs=(1, 2), blocks_per_seq=3, gamma=4)
        out = tmp_path / "shards"
        idx = dataset.assemble_shards(dump, out, block_filter=None)
        assert idx["num_blocks"] == 6

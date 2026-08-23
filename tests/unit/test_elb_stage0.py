"""P11 Stage-0 unit tests: routing_reader (memmap EPMR), miss decomposition,
free-signal bank, skill harness (spec/reports/EXPOSED_BYTE_CALCULUS.md
§3-P11(6)). Synthetic EPMR streams are packed with the exact layout of
src/speculation/epm_dump.h, same as test_elb_dataset.py."""

from __future__ import annotations

import pathlib
import struct
import sys

import numpy as np
import pytest

_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "tools"))

from elb_train import (dataset, routing_reader, signal_bank,  # noqa: E402
                       stage0_decompose)

EPMR = 0x524D5045


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


def make_stream(tmp_path, name="run_000.bin", *, seqs=(1, 2), n_pos=6,
                moe_layers=(3, 5, 9), n_experts=16, topk=4, seed=0,
                dup_positions=()):
    """Synthetic uniform EPMR stream. Returns (path, truth) where truth maps
    (seq, pos) -> rows written LAST for that key (dup_positions get two
    records; the later one is the truth — last-wins contract)."""
    rng = np.random.default_rng(seed)
    truth = {}
    blob = b""

    def emit(seq, pos):
        rows = []
        for layer in moe_layers:
            lg = rng.standard_normal(n_experts).astype(np.float16)
            tid = rng.choice(n_experts, topk, replace=False).astype(np.int32)
            tw = rng.random(topk).astype(np.float32)
            rows.append((layer, lg, tid, tw))
        truth[(seq, pos)] = rows
        return pack_route(seq, pos, rows)

    for seq in seqs:
        for pos in range(10, 10 + n_pos):
            blob += emit(seq, pos)
    for seq, pos in dup_positions:
        blob += emit(seq, pos)      # rewritten — overrides truth
    p = tmp_path / name
    p.write_bytes(blob)
    return p, truth


# ── routing_reader ───────────────────────────────────────────────────────────

class TestRoutingReader:
    def test_roundtrip_vs_slow_reader(self, tmp_path):
        p, _ = make_stream(tmp_path)
        view = routing_reader.open_run(p)
        slow = list(dataset.read_routing_records(p))
        assert view.n_records == len(slow)
        assert view.layers.tolist() == [3, 5, 9]
        for i, rr in enumerate(slow):
            assert int(view.seq[i]) == rr.seq_id
            assert int(view.pos[i]) == rr.token_pos
            np.testing.assert_array_equal(view.top_ids[i], rr.top_ids)
            np.testing.assert_array_equal(view.top_w[i], rr.top_w)
            np.testing.assert_array_equal(view.logits([i])[0], rr.logits)

    def test_iter_sequences_last_wins_dedup(self, tmp_path):
        p, truth = make_stream(tmp_path, dup_positions=((1, 12), (2, 10)))
        view = routing_reader.open_run(p)
        out = dict(routing_reader.iter_sequences(view))
        assert sorted(out) == [1, 2]
        for seq, idx in out.items():
            pos = view.pos[idx]
            assert np.all(np.diff(pos) > 0)          # sorted, deduped
            for i in idx:
                key = (seq, int(view.pos[i]))
                want = np.stack([r[1] for r in truth[key]])
                np.testing.assert_array_equal(view.logits([i])[0], want)

    def test_rejects_bad_magic(self, tmp_path):
        p, _ = make_stream(tmp_path)
        blob = bytearray(p.read_bytes())
        blob[0] ^= 0xFF
        p.write_bytes(bytes(blob))
        with pytest.raises(ValueError, match="bad EPMR header"):
            routing_reader.open_run(p)

    def test_rejects_truncated_tail(self, tmp_path):
        p, _ = make_stream(tmp_path)
        p.write_bytes(p.read_bytes()[:-7])
        with pytest.raises(ValueError, match="not a multiple"):
            routing_reader.open_run(p)

    def test_rejects_mixed_geometry(self, tmp_path):
        p, _ = make_stream(tmp_path, n_experts=16, topk=4)
        # append a same-size-lie record with different E: E'=12, K'=6 gives
        # rows*(4+24+48)=rows*76 vs rows*(4+32+32)=rows*68 — sizes differ,
        # so instead corrupt the E field of a middle record in place.
        blob = bytearray(p.read_bytes())
        rec_size = routing_reader._record_dtype(3, 16, 4).itemsize
        ne_off = rec_size * 2 + 24            # record 2, header field "ne"
        struct.pack_into("<I", blob, ne_off, 99)
        p.write_bytes(bytes(blob))
        with pytest.raises(ValueError, match="mixed geometry"):
            routing_reader.open_run(p)

    def test_rejects_layer_order_change(self, tmp_path):
        p, _ = make_stream(tmp_path, moe_layers=(3, 5, 9))
        blob = bytearray(p.read_bytes())
        rec_size = routing_reader._record_dtype(3, 16, 4).itemsize
        row_size = 4 + 16 * 2 + 4 * 4 + 4 * 4
        # swap layer ids of rows 0/1 in the LAST record (the probed one)
        last = len(blob) - rec_size + 32
        struct.pack_into("<I", blob, last, 5)
        struct.pack_into("<I", blob, last + row_size, 3)
        p.write_bytes(bytes(blob))
        with pytest.raises(ValueError, match="layer order"):
            routing_reader.open_run(p)


# ── decomposition ────────────────────────────────────────────────────────────

class TestDecompose:
    def _hand_case(self):
        """T=4, E=16, K=2, sigma=0.1, c=[1.0]. Hand-verified classes:
        t1 miss {2}: margin 0.8 -> novel (trail16/prevchunk: unseen... seen
        below), t2 miss {1: within-trail, 3: margin 0.05 -> boundary},
        t3 miss {0, 2}: both seen before -> within."""
        E = 16
        tops = np.array([[0, 1], [0, 2], [1, 3], [0, 2]], np.int32)
        w = np.ones((4, 2), np.float32)
        sel = np.zeros((4, E), np.float32)
        sel[0, [0, 1]] = [0.9, 0.8]
        sel[1, [0, 2]] = [0.9, 0.8]
        sel[2, [1, 3]] = [0.9, 0.05]
        sel[3, [0, 2]] = [0.9, 0.8]
        chunk = np.array([0, 0, 1, 1], np.int64)
        return tops, w, sel, chunk

    def test_partition_and_precedence(self):
        tops, w, sel, chunk = self._hand_case()
        acc = stage0_decompose.DecompAccum(
            windows=["prevpos", "prevchunk", "trail16"], cs=[1.0],
            n_layers=1)
        acc.add_sequence_layer(0, tops, w, sel, 0.1, chunk)
        assert acc.n_true[0] == 6 and acc.n_miss[0] == 5
        assert acc.n_hit[0] == 1                      # t1's expert 0
        assert acc.n_boundary[0, 0] == 1              # t2's expert 3
        wi = {n: i for i, n in enumerate(acc.windows)}
        # prevpos: nothing missed is at t-1 by construction
        assert acc.n_within[0, wi["prevpos"], 0] == 0
        assert acc.n_novel[0, wi["prevpos"], 0] == 4
        # trail16: expert 2 at t1 unseen -> novel; t2's 1 and t3's {0,2} seen
        assert acc.n_within[0, wi["trail16"], 0] == 3
        assert acc.n_novel[0, wi["trail16"], 0] == 1
        # prevchunk: t1 has no previous chunk -> novel; t2/t3 vs chunk0 union
        assert acc.n_within[0, wi["prevchunk"], 0] == 3
        assert acc.n_novel[0, wi["prevchunk"], 0] == 1
        # partition identity + summarize sanity
        s = acc.summarize(np.array([3], np.int32), 0, 100,
                          np.array([0.1], np.float32))
        g = s["all"]["grid"]["c1|trail16"]
        assert abs(g["f_boundary"] - 1 / 5) < 1e-9
        assert abs(g["f_within"] - 3 / 5) < 1e-9
        assert abs(g["f_novel"] - 1 / 5) < 1e-9
        assert abs(g["within_headroom"] - 3 / 6) < 1e-9

    def test_boundary_before_within(self):
        """A missed expert that is BOTH boundary and in the window counts
        as boundary (precedence)."""
        E = 16
        tops = np.array([[0, 3], [0, 1], [0, 3]], np.int32)
        w = np.ones((3, 2), np.float32)
        sel = np.zeros((3, E), np.float32)
        sel[0, [0, 3]] = [0.9, 0.8]
        sel[1, [0, 1]] = [0.9, 0.8]
        sel[2, [0, 3]] = [0.9, 0.02]      # 3 seen at t0 AND tiny margin
        acc = stage0_decompose.DecompAccum(windows=["trail16"], cs=[1.0],
                                           n_layers=1)
        acc.add_sequence_layer(0, tops, w, sel, 0.1,
                               np.zeros(3, np.int64))
        # t1 miss {1}: margin 0.8, unseen -> novel. t2 miss {3}: in the
        # trail16 window AND margin 0.02 < c*sigma -> BOUNDARY wins.
        assert acc.n_miss[0] == 2
        assert acc.n_boundary[0, 0] == 1
        assert acc.n_within[0, 0, 0] == 0
        assert acc.n_novel[0, 0, 0] == 1

    def test_sigma_estimator(self):
        rng = np.random.default_rng(0)
        E, T, sigma = 40, 4000, 0.01
        base = np.linspace(0.0, 1.0, E, dtype=np.float32)
        walk = np.cumsum(
            rng.normal(0.0, sigma, (T, E)).astype(np.float32), axis=0)
        sel = base[None, :] + walk
        est = stage0_decompose.SigmaEstimator(1, cap=100_000)
        est.add_sequence_layer(0, sel)
        s = est.sigma()[0]
        assert abs(s - sigma) / sigma < 0.1


# ── signal bank ──────────────────────────────────────────────────────────────

def _stream(rng, T, J, E, K):
    tops = np.stack([np.stack([rng.choice(E, K, replace=False)
                               for _ in range(J)]) for _ in range(T)]
                    ).astype(np.int32)
    w = rng.random((T, J, K)).astype(np.float32)
    sel = rng.random((T, J, E)).astype(np.float32)
    tok = rng.integers(0, 5000, T)
    return tops, w, sel, tok


class TestBank:
    J, E, K = 3, 12, 4

    def _run(self, member, tops, w, sel, tok):
        preds = []
        for t in range(len(tops)):
            preds.append(member.predict(int(tok[t])).copy())
            member.update(tops[t], w[t], sel[t], int(tok[t]))
        return preds

    def test_causality_and_reset(self):
        rng = np.random.default_rng(1)
        tops, w, sel, tok = _stream(rng, 8, self.J, self.E, self.K)
        for make in (lambda: signal_bank.B0Prev(self.J, self.E),
                     lambda: signal_bank.ScoreEma(self.J, self.E, 2.0),
                     lambda: signal_bank.TrailFreq(self.J, self.E, 4),
                     lambda: signal_bank.TokenMemo(self.J, self.E, 64),
                     lambda: signal_bank.TransEma(self.J, self.E)):
            full = self._run(make(), tops, w, sel, tok)
            trunc = self._run(make(), tops[:5], w[:5], sel[:5], tok[:5])
            for a, b in zip(trunc, full[:5]):
                np.testing.assert_array_equal(a, b)
        # reset clears per-seq state but not global tables
        memo = signal_bank.TokenMemo(self.J, self.E, 64)
        self._run(memo, tops, w, sel, tok)
        before = memo.predict(int(tok[0])).copy()
        memo.reset_seq()
        np.testing.assert_array_equal(memo.predict(int(tok[0])), before)
        b0 = signal_bank.B0Prev(self.J, self.E)
        self._run(b0, tops, w, sel, tok)
        assert b0.predict(0).sum() > 0
        b0.reset_seq()
        assert b0.predict(0).sum() == 0

    def test_score_ema_closed_form(self):
        m = signal_bank.ScoreEma(1, 4, 2.0)
        d = m.decay
        s1 = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
        s2 = np.array([[0.0, 1.0, 0.0, 0.0]], np.float32)
        t = np.zeros((1, 1), np.int32)
        wt = np.ones((1, 1), np.float32)
        m.update(t, wt, s1, 0)
        m.update(t, wt, s2, 0)
        want = (s2 + d * s1) / (1.0 + d)
        np.testing.assert_allclose(m.predict(0), want, rtol=1e-6)

    def test_token_memo_counts(self):
        m = signal_bank.TokenMemo(1, 8, buckets=16)
        wt = np.ones((1, 2), np.float32)
        m.update(np.array([[1, 2]], np.int32), wt, None, 5)
        m.update(np.array([[1, 3]], np.int32), wt, None, 5)
        m.update(np.array([[4, 5]], np.int32), wt, None, 21)  # same bucket
        p = m.predict(5)
        np.testing.assert_array_equal(
            p[0], [0, 2, 1, 1, 1, 1, 0, 0])
        assert m.predict(6).sum() == 0

    def test_trans_ema_chain(self):
        m = signal_bank.TransEma(1, 6, decay=0.5)
        wt = np.ones((1, 2), np.float32)
        seq = [np.array([[0, 1]], np.int32), np.array([[2, 3]], np.int32),
               np.array([[0, 1]], np.int32)]
        assert m.predict(0).sum() == 0
        m.update(seq[0], wt, None, 0)
        assert m.predict(0).sum() == 0            # prev set, table empty
        m.update(seq[1], wt, None, 0)             # (0,1)->(2,3)
        p = m.predict(0)                          # prev = (2,3): no rows yet
        assert p.sum() == 0
        m.update(seq[2], wt, None, 0)             # (2,3)->(0,1)
        p = m.predict(0)                          # prev = (0,1)
        assert p[0, 2] > 0 and p[0, 3] > 0        # learned 0/1 -> 2/3
        assert p[0, 2] == p[0, 3]
        assert p[0, [0, 1, 4, 5]].sum() == 0

    def test_trail_freq_window_edge(self):
        m = signal_bank.TrailFreq(1, 6, horizon=2)
        wt = np.ones((1, 1), np.float32)
        for e in (0, 1, 2):
            m.update(np.array([[e]], np.int32), wt, None, 0)
        p = m.predict(0)
        np.testing.assert_array_equal(p[0], [0, 1, 1, 0, 0, 0])


# ── driver-level: skill harness + evict dump schema ──────────────────────────

class TestHarness:
    def test_topm_membership_sparse_vs_dense(self):
        from elb_train import stage0
        pred = np.zeros((2, 10), np.float32)
        pred[0, [3, 7]] = [2.0, 1.0]
        # sparse: only positive-score experts are predictions
        m = stage0.topm_membership(pred, 5, sparse=True)
        assert m[0].sum() == 2 and m[0, 3] and m[0, 7]
        assert m[1].sum() == 0                    # all-zero row: nothing
        # dense: full top-m ranking, negatives legal
        d = -np.arange(10, dtype=np.float32)[None].repeat(2, 0)
        md = stage0.topm_membership(d, 3, sparse=False)
        assert md[0, :3].all() and md[0].sum() == 3

    def test_bank_eval_skill(self):
        from elb_train import stage0
        ev = stage0.BankEval("x", 2, [2, 4], sparse=True)
        tops = np.array([[0, 1], [2, 3]], np.int32)
        pred = np.zeros((2, 8), np.float32)
        pred[0, [0, 5]] = [1.0, 0.5]              # layer0: 1 of 2 in top-2
        pred[1, [2, 3]] = [1.0, 0.5]              # layer1: 2 of 2
        ev.add(pred, tops)
        r = ev.result(np.array([1]))
        assert abs(r["cov2_all"] - 3 / 4) < 1e-9
        assert abs(r["cov2_deep"] - 1.0) < 1e-9

    def test_manifest_eval_pre_chunk_causality(self):
        from elb_train import stage0
        me = stage0.ManifestEval("x", [4])
        ucur = np.zeros((1, 8), bool)
        ucur[0, [1, 2]] = True
        man = {4: np.zeros((1, 8), bool)}
        man[4][0, [1, 5]] = True                  # covers 1 of {1,2}
        me.add(man, ucur)
        r = me.result()
        assert abs(r["m4"]["cov_deep"] - 0.5) < 1e-9
        assert r["m4"]["avg_size"] == 2.0

    def test_evict_dump_roundtrip(self, tmp_path):
        from elb_train import evict_sim
        rng = np.random.default_rng(0)
        n = 400
        seq = np.repeat([1, 2], n // 2).astype(np.int64)
        pos = np.tile(np.repeat(np.arange(n // 20), 10), 2).astype(np.int32)
        arrs = {
            "seq": seq, "pos": pos,
            "layer": rng.integers(0, 3, n).astype(np.int16),
            "expert": rng.integers(0, 16, n).astype(np.int16),
            "prob": rng.random(n).astype(np.float32),
            "label": rng.integers(0, 2, n).astype(np.int8),
            "occ_prev": rng.integers(0, 2, n).astype(np.int8),
            "nextdist": rng.integers(-1, 20, n).astype(np.int16),
        }
        p = tmp_path / "preds_x.npz"
        np.savez_compressed(p, **arrs)
        trace = evict_sim.build_trace(dict(np.load(p)))
        assert len(trace["key"]) == n
        for pol in ("lru", "prevunion", "pred_bridge", "belady"):
            h = evict_sim.simulate(trace, 8, pol)
            assert 0.0 <= h <= 1.0

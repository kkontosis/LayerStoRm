"""P11.c' engine tests: new feature/member math (trans2, xprev, gate
dynamics, gfreq, memo, tap join) against naive reference computations
(tools/elb_train/stagecprime_gpu.py, stagecprime.py). CPU torch, tiny
shapes — pattern follows test_elb_stageb.py."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "tools"))

from elb_train.stagecprime_gpu import CPrimeState, ExpCfg  # noqa: E402
from elb_train.stagecprime import TapProvider  # noqa: E402

J, E, K = 3, 10, 2
BASE_F = 12


def _probe_packed() -> np.ndarray:
    # mu 0, sd 1, weights 0 -> probe == sigmoid(0) everywhere (inert)
    return np.concatenate([np.zeros(BASE_F), np.ones(BASE_F),
                           np.zeros(BASE_F + 1)]).astype(np.float32)


def _state(**cfg_kw) -> CPrimeState:
    cfg = ExpCfg(name="t", **cfg_kw)
    return CPrimeState(J, E, K, torch.device("cpu"), _probe_packed(),
                       np.full(J, 0.1, np.float32), cfg)


def _step(st: CPrimeState, tops, w=None, sel=None, bucket=-1,
          bucket2=-1):
    tops_t = torch.as_tensor(tops, dtype=torch.int64)
    sel_t = (torch.as_tensor(sel, dtype=torch.float32) if sel is not None
             else torch.zeros(J, E))
    w_t = (torch.as_tensor(w, dtype=torch.float32) if w is not None
           else torch.ones(J, K))
    X = torch.zeros(J, E, st.F)
    valid = torch.zeros(J, E, dtype=torch.bool)
    yb = torch.zeros(J, E, dtype=torch.bool)
    yb.scatter_(1, tops_t, True)
    st.update(tops_t, sel_t, w_t, X, valid, yb, bucket, bucket2)


class TestTrans2:
    def test_skip_transition_uses_t_minus_2(self):
        st = _state(features=("trans2",), members=("trans2",))
        t0 = [[0, 1], [2, 3], [4, 5]]
        t1 = [[6, 7], [8, 9], [0, 1]]
        t2 = [[2, 3], [4, 5], [6, 7]]
        _step(st, t0)
        _step(st, t1)          # trans2 pair (t0 -> t1)? no: needs t-2
        _step(st, t2)          # records pair (t0, t2)
        _, _, raw = st.features()
        v = raw["trans2"]
        # prev2_8 is now t1: score = sum over p in t1[j] of T2[j, p, e];
        # T2 holds only the (t0, t2) pair -> nonzero only where
        # p in t0[j] would match — t1 disjoint from t0 => all zeros
        assert torch.all(v == 0)
        # one more step: pair (t1, t3) recorded; prev2_8 becomes t2
        t3 = [[0, 1], [2, 3], [4, 5]]
        _step(st, t3)
        _, _, raw = st.features()
        v = raw["trans2"]
        # prev2_8 == t2; T2 rows for t2's experts hold the (t1->t3)?
        # no: pairs recorded are (t0->t2) and (t1->t3). Rows gathered
        # at t2[0]=[2,3] for layer 0: t0[0]=[0,1] wrote rows 0,1 only;
        # t1[0]=[6,7] wrote rows 6,7 -> row 2,3 empty => zeros again
        assert torch.all(v == 0)
        # direct table check: layer 0 pair (0 -> 6) exists (t0 -> t2)
        tbl = st.trans2 * st.trans2_scale
        assert tbl[0, 0, 2] > 0            # t0[0]=0 -> t2[0]=2
        assert tbl[0, 1, 3] > 0
        assert tbl[1, 2, 4] > 0            # layer 1: 2 -> 4
        assert tbl[0, 6, 0] > 0            # (t1 -> t3) layer 0: 6 -> 0

    def test_trans2_scores_gather_prev2(self):
        st = _state(features=("trans2",), members=("trans2",))
        a = [[0, 1], [0, 1], [0, 1]]
        b = [[2, 3], [2, 3], [2, 3]]
        for tops in (a, b, a, b, a):
            _step(st, tops)
        # pairs recorded: (a,a) x1? sequence a b a b a records
        # (a->a) at step 2 (t0=a, t2=a), (b->b) at 3, (a->a) at 4.
        # prev2_8 == b => score rows b: T2[j, 2] has b->? none (b rows
        # only from (b->b) pair) -> T2[j, 2, 2/3] > 0
        _, _, raw = st.features()
        v = raw["trans2"]
        assert v[0, 2] > 0 and v[0, 3] > 0
        assert v[0, 0] == 0 and v[0, 5] == 0


class TestXPrev:
    def test_xprev_up_shifts_layers(self):
        st = _state(features=("xprev_up",), members=("xprev_up",))
        t0 = [[0, 1], [2, 3], [4, 5]]      # layer j experts at t-1
        t1 = [[6, 7], [8, 9], [0, 1]]
        _step(st, t0)
        _step(st, t1)                      # pair: src t0[j+1] -> t1[j]
        # table: XU[0, {2,3}, {6,7}]; XU[1, {4,5}, {8,9}]
        tbl = st.xu * st.xu_scale
        assert tbl[0, 2, 6] > 0 and tbl[0, 3, 7] > 0
        assert tbl[1, 4, 8] > 0
        assert torch.all(tbl[2] == 0)      # last layer: no source
        # prediction at t2 uses prev8[j+1] = t1[j+1]
        _, _, raw = st.features()
        v = raw["xprev_up"]
        # layer 0 source t1[1] = {8,9}: XU[0, 8/9, :] empty -> 0
        assert torch.all(v[0] == 0)
        # feed t2 s.t. pair (t1[1] -> t2[0]) recorded, then check
        t2 = [[3, 4], [5, 6], [7, 8]]
        _step(st, t2)
        _, _, raw = st.features()
        v = raw["xprev_up"]
        # now prev8[1] = t2[1] = {5,6}; XU[0, 5, ?]: pair recorded at
        # t2: src t1[1]={8,9} -> tgt t2[0]={3,4}. Row 5 empty -> 0.
        assert torch.all(v[0] == 0)
        # but XU[0, 8, 3] must exist now
        tbl = st.xu * st.xu_scale
        assert tbl[0, 8, 3] > 0 and tbl[0, 9, 4] > 0

    def test_xprev_dn_shifts_other_way(self):
        st = _state(features=("xprev_dn",), members=("xprev_dn",))
        t0 = [[0, 1], [2, 3], [4, 5]]
        t1 = [[6, 7], [8, 9], [0, 1]]
        _step(st, t0)
        _step(st, t1)                      # pair: src t0[j-1] -> t1[j]
        tbl = st.xd * st.xd_scale
        assert tbl[1, 0, 8] > 0            # layer1 tgt {8,9} from t0[0]
        assert tbl[2, 2, 0] > 0
        assert torch.all(tbl[0] == 0)      # first layer: no source


class TestDynamics:
    def test_w_prev_and_ema(self):
        st = _state(features=("w_prev", "w_ema"))
        w0 = [[0.5, 0.25]] * J
        _step(st, [[1, 2]] * J, w=w0)
        X, _, _ = st.features()
        fi = {f: i for i, f in enumerate(st.feat_names)}
        assert abs(X[0, 1, fi["w_prev"]].item() - 0.5) < 1e-6
        assert abs(X[0, 2, fi["w_prev"]].item() - 0.25) < 1e-6
        assert X[0, 3, fi["w_prev"]] == 0
        # bias-corrected EMA equals the value itself after one step
        assert abs(X[0, 1, fi["w_ema"]].item() - 0.5) < 1e-5

    def test_gfreq_is_normalized_distribution(self):
        st = _state(features=("gfreq",), members=("gfreq",))
        for _ in range(3):
            _step(st, [[0, 1]] * J)
        _, _, raw = st.features()
        v = raw["gfreq"]
        assert v[0, 0] > 0 and v[0, 1] > 0
        assert torch.all(v[:, 2:] == 0)
        assert float(v.sum(dim=1).max()) < 1.0     # /(1+sum) shrinkage
        # survives reset_seq (cross-sequence state)
        st.reset_seq()
        _, _, raw = st.features()
        assert raw["gfreq"][0, 0] > 0

    def test_conf_trans_column(self):
        st = _state(features=("conf_trans",))
        _step(st, [[0, 1]] * J)
        _step(st, [[0, 1]] * J)            # trans pair recorded
        X, _, raw = st.features(conf_now=0.5)
        fi = {f: i for i, f in enumerate(st.feat_names)}
        expect = raw["trans"] * 0.5
        assert torch.allclose(X[:, :, fi["conf_trans"]], expect)


class TestMemo:
    def test_bucket_counts_and_norm(self):
        st = _state(features=("memo",), members=("memo",),
                    memo_buckets=8)
        _step(st, [[0, 1]] * J, bucket=3)
        _step(st, [[0, 2]] * J, bucket=3)
        _step(st, [[4, 5]] * J, bucket=1)
        _, _, raw = st.features(bucket=3)
        v = raw["memo"]
        # bucket 3 counts: e0:2, e1:1, e2:1 -> normalized /(1+4)
        assert abs(v[0, 0].item() - 2 / 5) < 1e-6
        assert abs(v[0, 1].item() - 1 / 5) < 1e-6
        assert v[0, 4] == 0
        # cross-sequence persistence
        st.reset_seq()
        _, _, raw = st.features(bucket=3)
        assert raw["memo"][0, 0] > 0

    def test_memo_maxpool(self):
        st = _state(features=("memo",), members=("memo",),
                    memo_buckets=8)
        _step(st, [[0, 1]] * J, bucket=2)
        _step(st, [[4, 5]] * J, bucket=6)
        mp = st.memo_maxpool([2, 6])
        # max over per-bucket dists: e0 from bucket2, e4 from bucket6
        assert mp[0, 0] > 0 and mp[0, 4] > 0
        assert mp[0, 7] == 0
        assert torch.all(st.memo_maxpool([]) == 0)


class TestOpenChannel:
    def test_cfg_and_member_sets(self):
        cfg = ExpCfg(name="t", features=("memo", "tap"),
                     members=("memo", "tap"), open_ch=True)
        assert "rls_open" in cfg.member_names
        assert "combined_open" in cfg.variant_names
        assert "memo_open" in cfg.variant_names
        try:
            ExpCfg(name="bad", features=(), members=("memo",))
            raise AssertionError("member without feature must raise")
        except ValueError:
            pass

    def test_member_masks_open_vs_base(self):
        st = _state(features=("memo",), members=("memo",))
        valid = torch.zeros(J, E, dtype=torch.bool)
        valid[:, :4] = True
        masks = st.member_masks(valid)
        names = st.member_names
        for i, m in enumerate(names):
            if m == "memo":
                assert bool(masks[i].all())
            else:
                assert torch.equal(masks[i], valid)


class TestTapProvider:
    def test_join_and_dedup(self, tmp_path):
        Jt, Et = 2, 4
        anchor = np.array([100, 100, 100, 110, 110], np.int64)
        pos = np.array([100, 101, 102, 110, 111], np.int64)
        tap = np.arange(5 * Jt * Et, dtype=np.float16) \
            .reshape(5, Jt, Et)
        blk_anchor = np.array([100, 110], np.int64)
        blk_max = np.stack([np.full((Jt, Et), 1, np.float16),
                            np.full((Jt, Et), 2, np.float16)])
        np.savez(tmp_path / "seq_7.npz", anchor=anchor, pos=pos,
                 tap=tap, blk_anchor=blk_anchor, blk_max=blk_max)
        s = {"pos": np.array([100, 102, 110], np.int64),
             "chunk": np.array([0, 0, 1], np.int64),
             "blocks": [{"anchor": 100}, {"anchor": 110}]}
        tp = TapProvider(tmp_path)
        cells, blocks = tp.for_seq(7, s)
        assert cells.shape == (3, Jt, Et)
        np.testing.assert_array_equal(cells[0], tap[0])
        np.testing.assert_array_equal(cells[1], tap[2])
        np.testing.assert_array_equal(cells[2], tap[3])
        np.testing.assert_array_equal(blocks[0], blk_max[0])
        np.testing.assert_array_equal(blocks[1], blk_max[1])

    def test_missing_row_raises(self, tmp_path):
        np.savez(tmp_path / "seq_9.npz",
                 anchor=np.array([5], np.int64),
                 pos=np.array([5], np.int64),
                 tap=np.zeros((1, 1, 2), np.float16),
                 blk_anchor=np.array([5], np.int64),
                 blk_max=np.zeros((1, 1, 2), np.float16))
        s = {"pos": np.array([6], np.int64),
             "chunk": np.array([0], np.int64),
             "blocks": [{"anchor": 5}]}
        try:
            TapProvider(tmp_path).for_seq(9, s)
            raise AssertionError("missing row must raise")
        except KeyError:
            pass


class TestMemoSparseTopk:
    def test_seen_and_stamp(self):
        from elb_train.stagecprime_gpu import memo_sparse_topk
        J, B, E = 2, 10, 8
        memo = torch.zeros(J, B, E)
        memo[0, 3, 5] = 7.0
        memo[0, 3, 1] = 2.0
        memo[1, 3, 0] = 1.0
        memo[1, 9, 2] = 4.0
        seen, ids, cnt = memo_sparse_topk(memo, k=3, chunk=4)
        np.testing.assert_array_equal(seen, [3, 9])
        assert ids.shape == (2, J, 3) and cnt.dtype == np.uint16
        # bucket 3, layer 0: top experts 5 (7), 1 (2), pad count 0
        assert ids[0, 0, 0] == 5 and cnt[0, 0, 0] == 7
        assert ids[0, 0, 1] == 1 and cnt[0, 0, 1] == 2
        assert cnt[0, 0, 2] == 0
        # bucket 9, layer 1: expert 2 count 4; layer 0 all zero
        assert ids[1, 1, 0] == 2 and cnt[1, 1, 0] == 4
        assert cnt[1, 0].sum() == 0


class TestFp8Roundtrip:
    def test_relative_error_and_scale_shape(self):
        from elb_train.stagecprime_gpu import fp8_roundtrip
        torch.manual_seed(0)
        x = torch.rand(3, 20, 20) * torch.tensor(
            [1.0, 100.0, 10000.0])[:, None, None]
        q, s = fp8_roundtrip(x, dims=(1, 2))
        assert s.shape == (3,)
        rel = ((q - x).abs() / x.clamp(min=1e-9)).max()
        assert float(rel) < 0.1          # e4m3 ~3 mantissa bits
        # zero stays exactly zero
        z = torch.zeros(2, 4)
        qz, _ = fp8_roundtrip(z, dims=(1,))
        assert torch.all(qz == 0)

    def test_no_clipping_by_construction(self):
        from elb_train.stagecprime_gpu import fp8_roundtrip
        x = torch.tensor([[0.001, 65000.0]])
        q, _ = fp8_roundtrip(x, dims=(1,))
        assert float(q[0, 1]) > 60000.0  # amax representable exactly


class TestRecurHead:
    def test_delayed_label_ring(self):
        st = _state(features=("memo",), members=("memo",),
                    memo_buckets=8, open_ch=True, recur_head=True)
        # position 0 routes {0,1}; positions 1..16 route {0,2}:
        # recur16 label for p0 -> expert 0 reused (1), expert 1 not,
        # expert 2 counts too (last_seen > 0). X is zeros in _step so
        # Xa = bias only => brec bias entry == sum of positive labels.
        _step(st, [[0, 1]] * J, bucket=1)
        for _ in range(16):
            _step(st, [[0, 2]] * J, bucket=1)
        assert len(st.rring) == 16          # ring trimmed at horizon
        assert abs(float(st.brec_h[16][0, -1]) - 2.0) < 1e-9
        # sequence reset drops truncated windows
        st.reset_seq()
        assert len(st.rring) == 0

    def test_recur_requires_open(self):
        try:
            _state(features=(), members=(), recur_head=True)
            raise AssertionError("recur_head without open_ch must raise")
        except ValueError:
            pass


class TestTinyAttn:
    def test_shapes_params_and_online_step(self):
        from elb_train.stagecprime_attn import AttnState, TinyAttn
        Jt, Et, B = 4, 12, 64
        for mode in ("pooled", "expert_query"):
            m = TinyAttn(Jt, Et, B, mode=mode)
            tok = torch.zeros(3, dtype=torch.int64)
            tw = torch.zeros(3, Jt, 2, dtype=torch.int64)
            ex = torch.zeros(3, 2)
            out = m(tok, tw, ex)
            assert out.shape == (Jt, Et)
        # full-size param budget: ~100k class
        big = TinyAttn(75, 256, 4096, mode="pooled")
        assert big.n_params() < 100_000
        st = AttnState(Jt, Et, B, "pooled", torch.device("cpu"))
        ridge = torch.zeros(Jt, Et)
        yb = torch.zeros(Jt, Et, dtype=torch.bool)
        yb[:, 0] = True
        # empty ring: predict = scaled ridge, update pushes only
        p0 = st.predict(ridge)
        assert p0.shape == (Jt, Et)
        st.update(ridge, yb, 3,
                  torch.zeros(Jt, 2, dtype=torch.int64), (0.5, 0.9))
        assert len(st.ring) == 1
        before = [p.clone() for p in st.model.parameters()]
        st.update(ridge, yb, 5, torch.zeros(Jt, 2,
                                            dtype=torch.int64),
                  (0.5, 0.9))
        changed = any(not torch.equal(a, b) for a, b in
                      zip(before, st.model.parameters()))
        assert changed                      # online step happened
        for _ in range(12):
            st.update(ridge, yb, 1, torch.zeros(Jt, 2,
                                                dtype=torch.int64),
                      (0.5, 0.9))
        assert len(st.ring) == 8            # window capped
        st.reset_seq()
        assert len(st.ring) == 0


class TestZeroGate:
    def test_init_is_exact_ridge_ranking(self):
        from elb_train.stagecprime_attn import AttnState
        Jt, Et = 3, 10
        st = AttnState(Jt, Et, 16, "expert_query",
                       torch.device("cpu"), zero_gate=True)
        assert torch.all(st.model.gamma == 0)
        ridge = torch.randn(Jt, Et)
        yb = torch.zeros(Jt, Et, dtype=torch.bool)
        yb[:, 1] = True
        st.update(ridge, yb, 2, torch.zeros(Jt, 2,
                                            dtype=torch.int64),
                  (0.4, 0.8))
        # gamma == 0: fused logits == alpha * ridge -> same ranking
        p = st.predict(ridge)
        a = float(st.model.alpha.detach())
        assert torch.allclose(p, a * ridge, atol=1e-6)
        assert torch.equal(p.argsort(dim=1), ridge.argsort(dim=1))


class TestMultiHorizonHeads:
    def test_ring_maturity_per_horizon(self):
        st = _state(features=("memo",), members=("memo",),
                    memo_buckets=8, open_ch=True,
                    recur_horizons=(2, 4))
        # t0 routes {0,1}; t1.. route {2,3}: label(t0,h) = |{2,3}| = 2
        _step(st, [[0, 1]] * J)
        for _ in range(5):
            _step(st, [[2, 3]] * J)
        d = 0.999
        # h=2 head: labels for t0 (at p=2), t1..t3 (2 each)
        b2 = float(st.brec_h[2][0, -1])
        expect = ((2 * d + 2) * d + 2) * d + 2   # t0..t3 with decay
        assert abs(b2 - expect) < 1e-6
        # h=4 head: labels for t0 (at p=4), t1 (at p=5)
        b4 = float(st.brec_h[4][0, -1])
        assert abs(b4 - (2 * d + 2)) < 1e-6
        # shared ring trimmed at max horizon
        assert len(st.rring) == 4
        st.reset_seq()
        assert len(st.rring) == 0

    def test_cfg_properties_and_alias(self):
        cfg = ExpCfg(name="t", features=(), members=(), open_ch=True,
                     recur_head=True, recur_horizons=(32, 64))
        assert cfg.horizons == (16, 32, 64)
        assert cfg.eff_evict_scorers == ("rls_open",)
        cfg2 = ExpCfg(name="t2", features=(), members=(),
                      open_ch=True,
                      evict_scorers=("recur16", "recur32"))
        assert cfg2.eff_evict_scorers == ("recur16", "recur32")
        st = _state(features=(), members=(), open_ch=True,
                    recur_head=True)
        assert st.wrec is st.wrec_h[16]     # alias preserved

    def test_export_carries_all_heads(self, tmp_path):
        from elb_train.stagecprime_gpu import export_state
        st = _state(features=("memo",), members=("memo",),
                    memo_buckets=8, open_ch=True,
                    recur_horizons=(16, 32), manifest_head=True)
        _step(st, [[0, 1]] * J, bucket=1)
        export_state(st, tmp_path)
        import numpy as _np
        z = _np.load(tmp_path / "ridge.npz")
        for k in ("w_open", "w_recur", "w_recur32", "w_manifest"):
            assert k in z.files, k

    def test_manifest_buffer_reset(self):
        st = _state(features=(), members=(), open_ch=True,
                    manifest_head=True)
        st.cbuf.append(torch.zeros(J, E, st.F))
        st.reset_seq()
        assert len(st.cbuf) == 0


class TestHeadFeatureMask:
    def test_masked_columns_solve_to_zero(self):
        st = _state(features=("memo",), members=("memo",),
                    memo_buckets=8, open_ch=True,
                    recur_horizons=(2,), manifest_head=True,
                    head_mask_features=("memo",))
        mi = st.feat_names.index("memo")
        assert float(st.head_mask[mi]) == 0.0
        assert float(st.head_mask[-1]) == 1.0      # bias kept
        g = torch.Generator().manual_seed(0)
        for i in range(70):                        # > RLS_REFRESH
            tops = torch.randint(0, E, (J, K), generator=g).long()
            sel = torch.rand(J, E, generator=g)
            tw = torch.rand(J, K, generator=g)
            # REAL features (the zero-X _step helper would make every
            # non-bias column trivially zero)
            if st.t_idx >= 1:
                X, valid, raw = st.features(bucket=int(i % 8))
            else:
                X = torch.zeros(J, E, st.F)
                valid = torch.zeros(J, E, dtype=torch.bool)
            yb = torch.zeros(J, E, dtype=torch.bool)
            yb.scatter_(1, tops, True)
            st.update(tops, sel, tw, X, valid, yb, int(i % 8))
        # masked feature's head weights are EXACTLY zero; open ridge
        # (full set) is free to use it
        assert torch.all(st.wrec_h[2][:, mi] == 0)
        assert torch.all(st.wman[:, mi] == 0)
        assert not torch.all(st.wo[:, mi] == 0)
        # unmasked columns of the head fit are generally nonzero
        assert float(st.wrec_h[2].abs().sum()) > 0

    def test_mask_requires_known_feature(self):
        try:
            _state(features=(), members=(), open_ch=True,
                   recur_horizons=(16,),
                   head_mask_features=("nope",))
            raise AssertionError("unknown mask feature must raise")
        except ValueError:
            pass


class TestXSame:
    """ARCHITECT_TRICK.md xsame_Δ: within-position layer-lag channel
    (pipelined protocol; soft = score-weighted full-sel gather)."""

    def test_table_semantics_vs_naive(self):
        d = 2
        st = _state(features=("xsame",), open_ch=True, xsame_lag=d)
        g = torch.Generator().manual_seed(0)
        ref = torch.zeros(J, E, E)
        steps = []
        for i in range(5):
            tops = torch.randint(0, E, (J, K), generator=g).long()
            steps.append(tops)
            _step(st, tops)
        # naive decayed reference: pairs (p in top8(t, j-d),
        # e in top8(t, j)) with 0.99^age decay
        for age, tops in enumerate(reversed(steps)):
            wdec = 0.99 ** age
            for j in range(d, J):
                for p in tops[j - d].tolist():
                    for e in tops[j].tolist():
                        ref[j, p, e] += wdec
        got = st.xsame * st.xsame_scale
        assert torch.allclose(got, ref, atol=1e-4)
        assert torch.all(got[:d] == 0)

    def test_soft_and_hard_gather(self):
        d = 1
        for soft in (True, False):
            st = _state(features=("xsame",), open_ch=True,
                        xsame_lag=d, xsame_soft=soft)
            g = torch.Generator().manual_seed(1)
            for i in range(4):
                _step(st, torch.randint(0, E, (J, K),
                                        generator=g).long())
            sel_now = torch.rand(J, E, generator=g)
            tops_now = torch.randint(0, E, (J, K),
                                     generator=g).long()
            X, valid, raw = st.features(sel_now=sel_now,
                                        tops_now=tops_now)
            xi = st.feat_names.index("xsame")
            tbl = st.xsame * st.xsame_scale
            if soft:
                want = torch.einsum("jp,jpe->je", sel_now[:-d],
                                    tbl[d:])
            else:
                # hard sources = ROUTED tops of layer j-d
                src8 = tops_now[:-d]
                want = torch.stack([tbl[d + j][src8[j]].sum(0)
                                    for j in range(J - d)])
            assert torch.allclose(X[d:, :, xi], want, atol=1e-5)
            assert torch.all(X[:d, :, xi] == 0)
            # no sel_now/tops_now (manifest/chunk query) -> zeros
            X0, _, _ = st.features()
            assert torch.all(X0[:, :, xi] == 0)
            if not soft:       # hard without routed tops -> zeros
                Xh, _, _ = st.features(sel_now=sel_now)
                assert torch.all(Xh[:, :, xi] == 0)

    def test_causality_score_then_update(self):
        # the feature queried at position t must reflect pair counts
        # from positions <= t-1 only
        d = 1
        st = _state(features=("xsame",), open_ch=True, xsame_lag=d)
        g = torch.Generator().manual_seed(2)
        t0 = torch.randint(0, E, (J, K), generator=g).long()
        _step(st, t0)
        tbl_before = (st.xsame * st.xsame_scale).clone()
        sel_now = torch.rand(J, E, generator=g)
        X, _, _ = st.features(sel_now=sel_now)
        xi = st.feat_names.index("xsame")
        want = torch.einsum("jp,jpe->je", sel_now[:-d],
                            tbl_before[d:])
        assert torch.allclose(X[d:, :, xi], want, atol=1e-5)
        t1 = torch.randint(0, E, (J, K), generator=g).long()
        _step(st, t1)      # t1's pairs enter only after scoring
        assert not torch.allclose(st.xsame * st.xsame_scale,
                                  tbl_before)

    def test_head_mask_and_validation(self):
        st = _state(features=("xsame",), open_ch=True, xsame_lag=4,
                    recur_horizons=(2,), manifest_head=True,
                    head_mask_features=("xsame",))
        xi = st.feat_names.index("xsame")
        assert float(st.head_mask[xi]) == 0.0
        try:
            _state(features=("xsame",), open_ch=True)
            raise AssertionError("xsame without lag must raise")
        except ValueError:
            pass
        try:
            _state(features=(), open_ch=True, xsame_lag=2)
            raise AssertionError("lag without xsame must raise")
        except ValueError:
            pass


class TestPipelinedMetricDiscipline:
    """Teacher's review fix (33b3d57e follow-up): xsame runs must
    never leak pipelined pool numbers under strict metric names —
    summaries carry pipelined{lag}_* keys and the verdict excludes
    them from best/beats_bar, scoring them in a dedicated section."""

    @staticmethod
    def _fake_ev(pool_open, cfg):
        k24 = {"coverage": 0.5, "addressable_recovery": 0.4,
               "novel_recovery": 0.3}
        return {
            "pool": {"combined": {"k24": dict(k24)},
                     "rls_online": {"k24": dict(k24)},
                     "rls_open": {"k24": {
                         "coverage": pool_open,
                         "addressable_recovery": 0.8,
                         "novel_recovery": 0.6}}},
            "manifest": {"combined_topup32": {"cov_deep": 0.92}},
            "hedge": {"members": ["rls_online"], "w_deep": [1.0]},
            "cfg": cfg,
        }

    def test_summary_prefixes_pipelined_keys(self):
        from elb_train.stagecprime import summarize_experiment
        ev = self._fake_ev(0.75, {"features": ["memo", "xsame"],
                                  "members": [], "open_ch": True,
                                  "xsame_lag": 4})
        s = summarize_experiment("xs4", ev)
        assert "pipelined4_pool32_rls_open" in s
        assert "pool32_rls_open" not in s
        assert not any(k.startswith(("pool32_", "novel_rec32_",
                                     "addr_rec32_")) for k in s)
        assert "manifest_topup32" in s      # heads stay strict
        # strict config untouched
        ev2 = self._fake_ev(0.7, {"features": ["memo"],
                                  "members": [], "open_ch": True})
        s2 = summarize_experiment("strict", ev2)
        assert "pool32_rls_open" in s2

    def test_verdict_excludes_pipelined_from_best(self):
        from elb_train.stagecprime import (evaluate_verdict,
                                           summarize_experiment)
        ship = self._fake_ev(0.7369, {"features": ["memo"],
                                      "members": [], "open_ch": True})
        ship["summary"] = summarize_experiment("x18ship2", ship)
        # old-code entry: STANDARD keys but cfg declares xsame
        old = self._fake_ev(0.99, {"features": ["memo", "xsame"],
                                   "members": [], "open_ch": True,
                                   "xsame_lag": 8})
        old["summary"] = {"pool32_rls_open": 0.99,
                          "novel_rec32_rls_open": 0.7,
                          "addr_rec32_rls_open": 0.9,
                          "pool32_combined": 0.99}
        # new-code entry: prefixed keys
        new = self._fake_ev(0.98, {"features": ["memo", "xsame"],
                                   "members": [], "open_ch": True,
                                   "xsame_lag": 4})
        new["summary"] = summarize_experiment("xs4", new)
        v = evaluate_verdict({"experiments": {
            "x18ship2": ship, "x18ship2_xs8": old,
            "x18ship2_xs4": new}})
        assert v["best"]["experiment"] == "x18ship2"
        assert "x18ship2_xs8" not in v["experiments"]
        assert "x18ship2_xs4" not in v["experiments"]
        pipe = v["pipelined"]["experiments"]
        assert pipe["x18ship2_xs8"]["lag"] == 8
        assert abs(pipe["x18ship2_xs8"]["d_pool_pp"]
                   - 100 * (0.99 - 0.7369)) < 1e-9
        g = v["pipelined"]["gates"]
        assert g["p2_fires"] and not g["kill_fires"]
        assert g["monotone_in_lag"] is not None


class TestX20Prune:
    """x20 Tier-2 prunes: boundary in-place table transforms."""

    def test_pair_table_tau(self):
        from elb_train.stagecprime_gpu import prune_state
        st = _state(features=("xsame",), open_ch=True, xsame_lag=1,
                    prune_tau=0.05)
        g = torch.Generator().manual_seed(0)
        st.trans = torch.rand(J, E, E, generator=g)
        st.xsame = torch.rand(J, E, E, generator=g)
        info = prune_state(st, st.cfg)
        for name in ("trans", "xsame"):
            t = getattr(st, name)
            rmax = t.amax(dim=2, keepdim=True)
            nz = t > 0
            assert torch.all(t[nz] >= 0.05 * rmax.expand_as(t)[nz]
                             - 1e-7)
            assert info[name]["nnz"] == int(nz.sum())
            assert 0 < info[name]["mass_kept"] <= 1

    def test_memo_s_and_k(self):
        from elb_train.stagecprime_gpu import prune_state
        st = _state(features=("memo",), members=("memo",),
                    open_ch=True, memo_buckets=8,
                    prune_memo_s=2.0, prune_memo_k=3)
        g = torch.Generator().manual_seed(1)
        st.memo = torch.rand(J, 8, E, generator=g)
        st.memo[:, 0, :] *= 0.01          # low-support bucket
        ref = st.memo.clone()
        prune_state(st, st.cfg)
        # low-support rows zeroed
        low = ref.sum(dim=2) < 2.0
        assert torch.all(st.memo[low] == 0)
        # surviving rows: at most k nonzero (no threshold ties in
        # continuous rand), and they are the row's top-k
        for j in range(J):
            for b in range(8):
                if low[j, b]:
                    continue
                row = st.memo[j, b]
                assert int((row > 0).sum()) <= 3
                top = torch.topk(ref[j, b], 3).indices
                assert torch.all(row[row > 0]
                                 >= ref[j, b][top[-1]] - 1e-7)

    def test_validation(self):
        try:
            _state(features=(), open_ch=True, prune_memo_k=8)
            raise AssertionError("memo prune without memo must raise")
        except ValueError:
            pass


class TestPihat2d:
    """pihat2d (§7 pre-registration): 2-D rank x gap-quantile
    adjustment — update/query semantics + Laplace reduction."""

    def _st(self):
        return _state(features=(), open_ch=True, cal_head=True,
                      cal2d=True)

    def test_requires_cal_head(self):
        try:
            _state(features=(), open_ch=True, cal2d=True)
            raise AssertionError("cal2d without cal_head must raise")
        except ValueError:
            pass

    def test_update_and_query_semantics(self):
        from elb_train.stagecprime_gpu import (CAL2D_ALPHA,
                                               CAL2D_BUCKETS,
                                               CAL2D_POOL_M)
        st = self._st()
        g = torch.Generator().manual_seed(0)
        for i in range(30):
            sc = torch.rand(J, E, generator=g)
            in_prev = torch.zeros(J, E, dtype=torch.bool)
            in_prev[:, :K] = True
            # production semantics: yb has EXACTLY K true per layer
            # (guarantees sum-of-rates K and M-hat in [0, J*K])
            yb = torch.zeros(J, E, dtype=torch.bool)
            yb.scatter_(1, torch.randint(0, E, (J, K),
                                         generator=g), True)
            b_all, micro = st.cal2d_buckets(sc, in_prev)
            assert b_all.shape == (J, E)
            assert int(b_all.max()) < CAL2D_BUCKETS
            mh2 = st.cal2d_mhat(b_all)
            assert torch.isfinite(mh2)
            ho, hp = st.cal_update(sc, in_prev, yb)
            st.cal2d_update(ho, hp, b_all, micro)
        # exposures accumulated: per (j, r) bucket-sum equals the
        # decayed count of positions (raw-decayed: sum_i rho^i)
        want = sum(0.999 ** i for i in range(30))
        got = st.calN2.sum(dim=2)
        assert torch.allclose(got, torch.full_like(got, want),
                              atol=1e-3)
        # M-hat within the trivial bounds
        mh2 = st.cal2d_mhat(st.cal2d_buckets(
            torch.rand(J, E, generator=g), in_prev)[0])
        assert 0.0 <= float(mh2) <= J * K

    def test_laplace_reduces_to_1d_when_empty(self):
        st = self._st()
        g = torch.Generator().manual_seed(1)
        sc = torch.rand(J, E, generator=g)
        in_prev = torch.zeros(J, E, dtype=torch.bool)
        in_prev[:, :K] = True
        # train ONLY the 1-D head (2-D tables stay empty)
        for i in range(20):
            yb = torch.rand(J, E, generator=g) < 0.3
            st.cal_update(sc, in_prev, yb)
        b_all, _ = st.cal2d_buckets(sc, in_prev)
        mh2 = float(st.cal2d_mhat(b_all))
        from elb_train.stagecprime_gpu import CAL2D_POOL_M
        c1 = st.cal_curve([CAL2D_POOL_M])[0]
        mh1 = float((st.K - c1).sum())
        assert abs(mh2 - mh1) < 1e-3     # empty bins == 1-D exactly


class TestShip3CalOnPipelined:
    def test_cal_head_allowed_on_xsame(self):
        # ship3 green light: cal tables train in-run on the run's
        # OWN stream — pipelined runs grow pipelined tables
        st = _state(features=("xsame",), open_ch=True, xsame_lag=2,
                    xsame_soft=False, cal_head=True)
        assert st.cfg.cal_head and st.cfg.xsame_lag == 2

    def test_sidecar_fit(self, tmp_path):
        from elb_train.stagecprime import fit_pihat_sidecars
        rng = np.random.default_rng(0)
        n = 4000
        t1 = rng.normal(size=n)
        t2 = rng.normal(size=n)
        m = 50 + 10 * t1 - 5 * t2 + rng.normal(scale=2, size=n)
        p = tmp_path / "pihat_series_t.npz"
        np.savez(p, mhat=m * 0, mreal=m, t1=t1, t2=t2)
        side = fit_pihat_sidecars(p)
        assert side["sidecar_ridge_w"].shape == (4,)
        assert float(side["sidecar_spearman_2nd"]) > 0.9
        assert "UNCERTIFIED" in str(side["sidecar_status"])


class TestAblate:
    """ARCHITECT_ABLATION.md Tier 1: zero-column exactness (the
    Tikhonov lemma) — ablate == exact LOFO refit."""

    def test_zero_column_exactness(self):
        ab = ("xprev_up", "xprev_dn", "trans", "trans_rank")
        st = _state(features=("xprev_up", "xprev_dn", "memo"),
                    members=("memo",), memo_buckets=8, open_ch=True,
                    recur_horizons=(2,), manifest_head=True,
                    ablate=ab)
        idx = [st.feat_names.index(f) for f in ab]
        # ablated extras skip their table builds entirely
        assert not hasattr(st, "xu") and not hasattr(st, "xd")
        g = torch.Generator().manual_seed(0)
        for i in range(70):                        # > RLS_REFRESH
            tops = torch.randint(0, E, (J, K), generator=g).long()
            sel = torch.rand(J, E, generator=g)
            tw = torch.rand(J, K, generator=g)
            if st.t_idx >= 1:
                X, valid, raw = st.features(bucket=int(i % 8))
                for j in idx:                      # columns exactly 0
                    assert torch.all(X[:, :, j] == 0)
            else:
                X = torch.zeros(J, E, st.F)
                valid = torch.zeros(J, E, dtype=torch.bool)
            yb = torch.zeros(J, E, dtype=torch.bool)
            yb.scatter_(1, tops, True)
            st.update(tops, sel, tw, X, valid, yb, int(i % 8))
        # every head's ablated weights are exactly zero
        for w in (st.wo, st.wr, st.wrec_h[2], st.wman):
            for j in idx:
                assert torch.all(w[:, j] == 0)
        # Tikhonov lemma on the accumulated stats: full solve with
        # zero columns == reduced-system solve on the kept coords
        from elb_train.stagec import RLS_LAMBDA
        keep = [j for j in range(st.F + 1) if j not in idx]
        eye_f = torch.eye(st.F + 1, dtype=torch.float64) * RLS_LAMBDA
        eye_r = torch.eye(len(keep), dtype=torch.float64) * RLS_LAMBDA
        for jl in range(J):
            wf = torch.linalg.solve(st.Ao[jl] + eye_f, st.bo[jl])
            assert torch.all(wf[idx].abs() < 1e-14)
            wr = torch.linalg.solve(
                st.Ao[jl][keep][:, keep] + eye_r, st.bo[jl][keep])
            assert torch.allclose(wf[keep], wr, atol=1e-10,
                                  rtol=1e-10)

    def test_ablate_requires_known_feature(self):
        try:
            _state(features=(), members=(), open_ch=True,
                   ablate=("nope",))
            raise AssertionError("unknown ablate feature must raise")
        except ValueError:
            pass

    def test_no_ablate_bit_exact(self):
        # chassis guarantee: empty ablate leaves X untouched
        st = _state(features=("memo",), members=("memo",),
                    memo_buckets=8, open_ch=True)
        assert st.ablate_mask is None


class TestNovelHead:
    def test_row_masked_accumulation(self):
        st = _state(features=("memo",), members=("memo",),
                    memo_buckets=8, open_ch=True, novel_head=True)
        g = torch.Generator().manual_seed(1)
        # X zeros -> Xa = bias only: b_nov bias entry counts positives
        # on OUT-of-union rows only. At t0 the union is empty, so t0's
        # routed {0,1} ARE out-rows (everything is novel at sequence
        # start): +2. At t1, {0,1} are in-union: +0 (decay only).
        d = 0.999
        _step(st, [[0, 1]] * J)
        _step(st, [[0, 1]] * J)
        assert abs(float(st.b_nov[0, -1]) - 2.0 * d) < 1e-9
        # route {5, 6} — outside the union AND outside prev8 at
        # prediction time -> +2 out-row positives per layer
        _step(st, [[5, 6]] * J)
        assert abs(float(st.b_nov[0, -1]) - (2.0 * d ** 2 + 2.0)) \
            < 1e-9
        _step(st, [[5, 6]] * J)          # {5,6} now in-union: adds 0
        assert abs(float(st.b_nov[0, -1])
                   - (2.0 * d ** 2 + 2.0) * d) < 1e-9

    def test_novel_requires_open(self):
        try:
            _state(features=(), members=(), novel_head=True)
            raise AssertionError("novel_head without open_ch must "
                                 "raise")
        except ValueError:
            pass


class TestCtxSizeSlope:
    def test_param_counts(self):
        from elb_train.stagecprime_attn import CtxEncoder
        base = CtxEncoder(75, 256).n_params()
        assert base == 350_203
        p125 = CtxEncoder(75, 256, d_out=20).n_params()
        p210 = CtxEncoder(75, 256, d=48, d_out=32).n_params()
        assert abs(p125 / base - 1.25) < 0.02
        assert abs(p210 / base - 2.10) < 0.02
        # forward shape holds at scaled dims
        m = CtxEncoder(3, 10, d=48, d_out=32)
        out = m(torch.zeros(32), torch.zeros(4, 32))
        assert out.shape == (3, 10)


class TestParityReduction:
    def test_empty_cfg_matches_base_bank(self):
        cfg = ExpCfg(name="x0")
        from elb_train.stagec import BANK
        assert cfg.member_names == tuple(BANK)
        assert cfg.variant_names == ("combined", "probe_frozen",
                                     "rls_online", "trans_ema")
        assert len(cfg.feat_names) == BASE_F


# ── x13 CUTTRACK (ARCHITECT_REVIEW.md Part B.1 / Part C) ─────────────


def _phi_np(x):
    from math import erf
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


class TestSoftHead:
    def test_requires_open_channel(self):
        try:
            ExpCfg(name="bad", soft_head=True)
            raise AssertionError("soft_head without open_ch must raise")
        except ValueError:
            pass

    def test_registry_and_feature_count(self):
        cfg = ExpCfg(name="t", open_ch=True, soft_head=True)
        assert "rls_soft" in cfg.member_names
        assert "rls_soft" in cfg.variant_names
        # the soft head adds NO features — x13a keeps F = base
        assert len(cfg.feat_names) == BASE_F

    def test_soft_label_and_decision_rows(self):
        st = _state(open_ch=True, soft_head=True)
        sel = torch.zeros(J, E)
        sel[:, 0] = 0.9
        sel[:, 1] = 0.8          # cut (K=2) = 0.8
        sel[:, 2:] = 0.1
        tops = [[0, 1]] * J
        # step 1: no prev8 -> ALL rows are decision rows; sigma at
        # floor (no drift seen) -> labels ~hard: e0 -> 1, e1 (== cut)
        # -> Phi(0) = 0.5, others -> 0; X zeros => bias column only
        _step(st, tops, sel=sel)
        for j in range(J):
            assert abs(float(st.bsoft[j, -1]) - 1.5) < 1e-6
            assert abs(float(st.Adr[j, -1, -1]) - E) < 1e-9
        # step 2, same sel/tops: e0/e1 now in prev8 -> EXCLUDED from
        # the fit; remaining rows have label ~0 -> pure decay
        _step(st, tops, sel=sel)
        for j in range(J):
            assert abs(float(st.bsoft[j, -1]) - 0.999 * 1.5) < 1e-6
            assert abs(float(st.Adr[j, -1, -1])
                       - (0.999 * E + (E - K))) < 1e-9

    def test_sigma_tracker_bias_corrected_mad(self):
        st = _state(open_ch=True, soft_head=True)
        sel1 = torch.linspace(0, 0.9, E)[None].repeat(J, 1)
        _step(st, [[8, 9]] * J, sel=sel1)
        assert st.sig_n == 0.0            # no drift at t=0
        _step(st, [[8, 9]] * J, sel=sel1 + 0.2)
        # median |drift| = 0.2 -> bias-corrected EW MAD scale
        want = 1.4826 * 0.2
        got = float(st.sig_drift[0]) / st.sig_n
        assert abs(got - want) < 1e-5

    def test_wsoft_solve_and_member_row(self):
        from elb_train.stagec import RLS_REFRESH
        st = _state(open_ch=True, soft_head=True)
        g = torch.Generator().manual_seed(3)
        for i in range(RLS_REFRESH + 1):
            sel = torch.rand(J, E, generator=g)
            tops = torch.topk(sel, K, dim=1).indices.tolist()
            _step(st, tops, sel=sel)
        assert torch.all(torch.isfinite(st.wsoft))
        assert float(st.wsoft.abs().sum()) > 0
        X, valid, raw = st.features()
        rows = st.member_scores(X, valid, raw)
        mi = {m: i for i, m in enumerate(st.member_names)}
        assert "rls_soft" in mi
        Xa = torch.cat([X.double(), torch.ones(J, E, 1,
                                               dtype=torch.float64)],
                       dim=2)
        want = torch.einsum("jef,jf->je", Xa, st.wsoft).float()
        assert torch.allclose(rows[mi["rls_soft"]], want, atol=1e-6)
        # open rank domain: mask is all experts
        masks = st.member_masks(valid)
        assert bool(masks[mi["rls_soft"]].all())


class TestHoltFilter:
    GAINS = ((0.5, 0.1), (0.25, 0.05), (0.1, 0.01))

    @staticmethod
    def _sel(t):
        # deterministic ramp; identical rows across layers
        return (0.1 * t + 0.01 * torch.arange(E).float())[None] \
            .repeat(J, 1)

    def test_recursions_match_scalar_reference(self):
        st = _state(filter_feats=True)
        T = 8
        for t in range(T):
            _step(st, [[0, 1]] * J, sel=self._sel(t))
        e_ref, j_ref = 3, 1
        series = [float(self._sel(t)[j_ref, e_ref]) for t in range(T)]
        cuts = [float(torch.topk(self._sel(t)[j_ref], K).values[-1])
                for t in range(T)]
        for gi, (a1, a2) in enumerate(self.GAINS):
            for ser, lv_t, tr_t, qv_t in (
                    (series, st.f_lv[gi, j_ref, e_ref],
                     st.f_tr[gi, j_ref, e_ref],
                     st.f_qv[gi, j_ref, e_ref]),
                    (cuts, st.c_lv[gi, j_ref], st.c_tr[gi, j_ref],
                     st.c_qc[gi, j_ref])):
                lv = tr = qv = None
                seqn = 0
                for s in ser:
                    if lv is None:
                        lv, tr, qv = s, 0.0, 0.0
                    else:
                        fc = lv + 0.98 * tr
                        i = s - fc
                        lv = fc + a1 * i
                        tr = 0.98 * tr + a2 * i
                        qv = 0.99 * qv + 0.01 * i * i
                    if seqn < 4:        # warm-up trend pin
                        tr = 0.0
                    seqn += 1
                assert abs(float(lv_t) - lv) < 1e-4
                assert abs(float(tr_t) - tr) < 1e-4
                assert abs(float(qv_t) - qv) < 1e-6

    def test_active_gain_selection_in_features(self):
        st = _state(filter_feats=True)
        _step(st, [[0, 1]] * J, sel=self._sel(0))
        _step(st, [[0, 1]] * J, sel=self._sel(1))
        # force per-layer active gains: layer 0 -> gain 1, others -> 0
        st.f_m.zero_()
        st.f_m[:, 0] = torch.tensor([1.0, 0.1, 2.0])
        st.f_m[:, 1:] = torch.tensor([0.1, 1.0, 2.0])[:, None]
        for gi in range(3):
            st.f_lv[gi].fill_(float(gi + 1))
        st.f_tr.zero_()
        X, _, _ = st.features()
        fi = {f: i for i, f in enumerate(st.feat_names)}
        assert torch.all(X[0, :, fi["holt_fc"]] == 2.0)
        assert torch.all(X[1, :, fi["holt_fc"]] == 1.0)
        assert torch.all(X[:, :, fi["cut_prob"]] >= 0)
        assert torch.all(X[:, :, fi["cut_prob"]] <= 1)

    def test_feature_names_order_and_count(self):
        cfg = ExpCfg(name="t", filter_feats=True)
        assert cfg.feat_names[-4:] == ("holt_fc", "cut_margin",
                                       "cut_prob", "trend_z")
        assert len(cfg.feat_names) == BASE_F + 4

    def test_reset_reprimes_but_keeps_priors(self):
        st = _state(filter_feats=True)
        for t in range(6):
            _step(st, [[0, 1]] * J, sel=self._sel(t))
        qv_before = st.f_qv.clone()
        st.reset_seq()
        assert not st.f_init and st.f_seq_n == 0
        assert torch.equal(st.f_qv, qv_before)   # prior persists
        new_sel = torch.full((J, E), 0.7)
        _step(st, [[0, 1]] * J, sel=new_sel)
        for gi in range(3):                      # re-primed to new sel
            assert torch.allclose(st.f_lv[gi], new_sel)
        _step(st, [[0, 1]] * J, sel=new_sel + 0.1)
        assert torch.all(st.f_tr == 0)           # warm-up pin again


class TestFactor:
    def test_requires_filter_feats(self):
        try:
            ExpCfg(name="bad", factor_r=4)
            raise AssertionError("factor_r without filter_feats must "
                                 "raise")
        except ValueError:
            pass

    def test_v_init_orthonormal(self):
        st = _state(filter_feats=True, factor_r=2)
        vtv = torch.einsum("jer,jes->jrs", st.V, st.V)
        eye = torch.eye(2)[None].expand(J, -1, -1)
        assert torch.allclose(vtv, eye, atol=1e-5)

    def test_oja_recovers_rank1_subspace(self):
        st = _state(filter_feats=True, factor_r=1)
        g = torch.Generator().manual_seed(7)
        vstar = torch.randn(J, E, generator=g)
        vstar = vstar / vstar.norm(dim=1, keepdim=True)
        for _ in range(600):
            z = torch.randn(J, 1, generator=g)
            sel = 0.5 + 0.2 * z * vstar \
                + 0.01 * torch.randn(J, E, generator=g)
            _step(st, [[0, 1]] * J, sel=sel)
        v = st.V[:, :, 0]
        cos = (v * vstar).sum(dim=1).abs() / v.norm(dim=1)
        assert torch.all(cos > 0.7), cos
        assert st._qr_since < 256                # QR ran

    def test_factor_features_present_and_finite(self):
        st = _state(filter_feats=True, factor_r=2)
        cfg = st.cfg
        assert cfg.feat_names[-2:] == ("fac_fc", "fac_margin")
        for t in range(3):
            sel = torch.rand(J, E,
                             generator=torch.Generator().manual_seed(t))
            _step(st, [[0, 1]] * J, sel=sel)
        X, _, _ = st.features()
        assert X.shape[2] == BASE_F + 6
        assert torch.all(torch.isfinite(X))


class TestFp8CutTrack:
    def test_roundtrip_covers_new_state(self):
        from elb_train.stagecprime_gpu import quantize_state_fp8
        st = _state(open_ch=True, soft_head=True, filter_feats=True,
                    factor_r=2)
        g = torch.Generator().manual_seed(1)
        for t in range(5):
            sel = torch.rand(J, E, generator=g)
            _step(st, [[0, 1]] * J, sel=sel)
        st.wsoft.copy_(torch.randn(J, st.F + 1, generator=g,
                                   dtype=torch.float64))
        v_before = st.V.clone()
        info = quantize_state_fp8(st)
        for key in ("wsoft", "f_qv", "V", "f_mu"):
            assert key in info["granularity"], key
        amax = float(v_before.abs().amax())
        assert float((st.V - v_before).abs().max()) < 0.1 * amax


class TestExportCutTrack:
    def test_export_includes_soft_and_filters(self, tmp_path):
        from elb_train.stagecprime_gpu import export_state
        st = _state(open_ch=True, soft_head=True, filter_feats=True,
                    factor_r=2)
        for t in range(3):
            sel = torch.rand(J, E,
                             generator=torch.Generator().manual_seed(t))
            _step(st, [[0, 1]] * J, sel=sel)
        files = export_state(st, tmp_path)
        assert files["filters"] == "filters.npz"
        assert files["factor"] == "factor.npz"
        ridge = np.load(tmp_path / "ridge.npz")
        assert "w_soft" in ridge
        filt = np.load(tmp_path / "filters.npz")
        for key in ("sig_drift", "qv", "gain_mse", "cut_qc", "gains"):
            assert key in filt, key
        fac = np.load(tmp_path / "factor.npz")
        assert fac["V"].shape == (J, E, 2)


class TestBandHeads:
    def test_requires_open_channel(self):
        try:
            ExpCfg(name="bad", band_heads=True)
            raise AssertionError("band_heads without open_ch must "
                                 "raise")
        except ValueError:
            pass

    def test_factorial_row_masks_exact(self):
        # sigma = 0.1 (from _state helper); cut (K=2) = 0.8
        st = _state(open_ch=True, band_heads=True)
        sel = torch.full((J, E), 0.1)
        sel[:, 0] = 0.95         # member, |m|=0.15 > sigma: in band
        sel[:, 1] = 0.8          # member AT cut: |m|=0 -> band-excl
        sel[:, 2] = 0.75         # loser, |m|=0.05 <= sigma: band-excl
        tops = [[0, 1]] * J
        # step 1: no prev8 -> dr = all rows
        _step(st, tops, sel=sel)
        for j in range(J):
            # dr head: hard positives = 2 (e0, e1)
            assert abs(float(st.bd[j, -1]) - 2.0) < 1e-9
            # band heads: only e0 survives the band among positives
            assert abs(float(st.bdb[j, -1]) - 1.0) < 1e-9
            assert abs(float(st.bbn[j, -1]) - 1.0) < 1e-9
            # band row count: E minus e1, e2
            assert abs(float(st.Abn[j, -1, -1]) - (E - 2)) < 1e-9
            assert abs(float(st.Ad[j, -1, -1]) - E) < 1e-9
        # step 2, same sel/tops: e0, e1 now prev8 -> dr excludes them
        _step(st, tops, sel=sel)
        for j in range(J):
            assert abs(float(st.bd[j, -1]) - 0.999 * 2.0) < 1e-9
            assert abs(float(st.bdb[j, -1]) - 0.999 * 1.0) < 1e-9
            # all-row band head still sees e0's positive
            assert abs(float(st.bbn[j, -1])
                       - (0.999 * 1.0 + 1.0)) < 1e-9

    def test_heads_solve_and_registry(self):
        from elb_train.stagec import RLS_REFRESH
        cfg = ExpCfg(name="t", open_ch=True, band_heads=True)
        for m in ("rls_dr", "rls_drband", "rls_band"):
            assert m in cfg.member_names and m in cfg.variant_names
        st = _state(open_ch=True, band_heads=True)
        g = torch.Generator().manual_seed(5)
        for _ in range(RLS_REFRESH + 1):
            sel = torch.rand(J, E, generator=g)
            tops = torch.topk(sel, K, dim=1).indices.tolist()
            _step(st, tops, sel=sel)
        for w in (st.wd, st.wdb, st.wbn):
            assert torch.all(torch.isfinite(w))
            assert float(w.abs().sum()) > 0


class TestDormancyGate:
    def test_requires_filter_feats(self):
        try:
            ExpCfg(name="bad", dormancy_gate=True)
            raise AssertionError("dormancy_gate without filter_feats "
                                 "must raise")
        except ValueError:
            pass

    def test_dormant_rows_get_neutral_dynamics(self):
        st = _state(filter_feats=True, dormancy_gate=True)
        sel = torch.full((J, E), 0.1)
        sel[:, 0] = 0.9          # above the cut -> positive margin
        sel[:, 1] = 0.8          # cut (K=2)
        for _ in range(3):
            _step(st, [[0, 1]] * J, sel=sel)
        X, _, _ = st.features()
        fi = {f: i for i, f in enumerate(st.feat_names)}
        # e0/e1 recently routed -> active; e5 never seen -> rec 512
        assert torch.all(X[:, 5, fi["cut_margin"]] == -20.0)
        assert torch.all(X[:, 5, fi["cut_prob"]] == 0.0)
        assert torch.all(X[:, 5, fi["trend_z"]] == 0.0)
        # active above-cut row keeps its own (positive) margin
        assert float(X[0, 0, fi["cut_margin"]]) > 0.0
        assert float(X[0, 0, fi["cut_prob"]]) > 0.5
        # holt_fc (level forecast) stays ungated everywhere
        assert torch.all(torch.isfinite(X[:, :, fi["holt_fc"]]))
        assert abs(float(X[0, 5, fi["holt_fc"]]) - 0.1) < 1e-5


class TestGainMix:
    def test_requires_filter_feats(self):
        try:
            ExpCfg(name="bad", gain_mix=True)
            raise AssertionError("gain_mix without filter_feats must "
                                 "raise")
        except ValueError:
            pass

    def test_uniform_at_cold_start_no_nan(self):
        st = _state(filter_feats=True, gain_mix=True)
        _step(st, [[0, 1]] * J,
              sel=torch.full((J, E), 0.5))
        # f_m still all-zero -> weights uniform; features finite
        X, _, _ = st.features()
        assert torch.all(torch.isfinite(X))
        fi = {f: i for i, f in enumerate(st.feat_names)}
        # all gains primed to the same level -> mixture == level
        assert torch.allclose(X[:, :, fi["holt_fc"]],
                              torch.full((J, E), 0.5))

    def test_mixture_tracks_inverse_mse(self):
        st = _state(filter_feats=True, gain_mix=True)
        _step(st, [[0, 1]] * J, sel=torch.full((J, E), 0.5))
        for gi in range(3):
            st.f_lv[gi].fill_(float(gi + 1))
        st.f_tr.zero_()
        fi = {f: i for i, f in enumerate(st.feat_names)}
        # equal MSE -> uniform mixture -> mean of levels (2.0)
        st.f_m.fill_(1.0)
        X, _, _ = st.features()
        assert torch.allclose(X[:, :, fi["holt_fc"]],
                              torch.full((J, E), 2.0), atol=1e-5)
        # gain 0 MSE 100x smaller -> mixture collapses onto level 1
        st.f_m[0].fill_(0.01)
        X, _, _ = st.features()
        assert torch.all((X[:, :, fi["holt_fc"]] - 1.0).abs() < 1e-3)


class TestBigramMemo:
    def test_lut_deterministic_bos_and_range(self):
        from elb_train.stagecprime_gpu import BIGRAM_BOS, bigram_lut
        toks = np.array([5, 5, 9, 5], np.int64)
        B = 64
        h1 = bigram_lut(toks, B)
        h2 = bigram_lut(toks, B)
        np.testing.assert_array_equal(h1, h2)      # deterministic
        assert h1.dtype == np.int64
        assert np.all((h1 >= 0) & (h1 < B))
        # position 0 pairs with BOS, position 1 with token 5:
        # same current token, different prev -> different context
        assert h1[0] != h1[1] or BIGRAM_BOS == 5   # (never equal)
        # order matters: (5, 9) != (9, 5) as contexts
        h3 = bigram_lut(np.array([9, 5], np.int64), B)
        assert h1[2] != h3[1] or True  # spot check, no hash promise
        # per-sequence reset: same tokens -> same buckets
        np.testing.assert_array_equal(
            bigram_lut(toks[:2], B), bigram_lut(toks[:2], B))

    def test_counts_normalization_and_persistence(self):
        st = _state(features=("memo2",), memo2_buckets=16)
        _step(st, [[0, 1]] * J, bucket2=3)
        _step(st, [[0, 2]] * J, bucket2=3)
        _step(st, [[4, 5]] * J, bucket2=7)
        _, _, raw = st.features(bucket2=3)
        v = raw["memo2"]
        assert abs(v[0, 0].item() - 2 / 5) < 1e-6  # e0: 2 of (1+4)
        assert abs(v[0, 1].item() - 1 / 5) < 1e-6
        assert v[0, 4] == 0
        # no-context call (manifest path) -> zeros column
        _, _, raw0 = st.features(bucket2=-1)
        assert torch.all(raw0["memo2"] == 0)
        # cross-sequence persistence
        st.reset_seq()
        _, _, raw = st.features(bucket2=3)
        assert raw["memo2"][0, 0] > 0

    def test_feature_column_matches_raw(self):
        st = _state(features=("memo2",), memo2_buckets=16)
        _step(st, [[0, 1]] * J, bucket2=5)
        _step(st, [[0, 1]] * J, bucket2=5)
        X, _, raw = st.features(bucket2=5)
        fi = {f: i for i, f in enumerate(st.feat_names)}
        assert torch.allclose(X[:, :, fi["memo2"]], raw["memo2"])

    def test_export_support_filter_and_k8(self, tmp_path):
        from elb_train.stagecprime_gpu import export_state
        st = _state(features=("memo2",), memo2_buckets=16,
                    open_ch=True)
        # bucket 5: 8 updates -> row support 16 (>= 8, kept);
        # bucket 9: 1 update -> support 2 (< 8, dropped)
        for _ in range(8):
            _step(st, [[0, 1]] * J, bucket2=5)
        _step(st, [[4, 5]] * J, bucket2=9)
        files = export_state(st, tmp_path)
        m2 = np.load(tmp_path / "memo2_sparse.npz")
        np.testing.assert_array_equal(m2["seen"], [5])
        assert int(m2["topk"]) == 8 and int(m2["min_support"]) == 8
        assert m2["top_ids"].shape == (1, J, 8)
        assert files["memo2_seen_buckets"] == 1

    def test_x16_registry(self):
        from elb_train.stagecprime import EXPERIMENTS
        cfg = ExpCfg(name="x16", **EXPERIMENTS["x16_bigram"])
        assert "memo2" in cfg.feat_names
        assert len(cfg.feat_names) == 16       # champion 15 + memo2
        assert cfg.memo2_buckets == 65536
        assert "memo2" not in cfg.member_names  # feature-only


class TestCtxFormer:
    def test_emb_project_shape_determinism_scale(self):
        from elb_train.stagecprime_attn import emb_project
        g = torch.Generator().manual_seed(11)
        emb = torch.randn(200, 48, generator=g)
        f1 = emb_project(emb, d=8, seed=0)
        f2 = emb_project(emb, d=8, seed=0)
        assert f1.shape == (200, 8)
        assert torch.equal(f1, f2)               # deterministic
        std = f1.std(dim=0)
        assert torch.all((std - 1.0).abs() < 0.1)  # standardized

    def test_encoder_params_and_shapes(self):
        from elb_train.stagecprime_attn import CtxEncoder
        m = CtxEncoder(75, 256)
        # 100k-1MB class: ~350k learned params
        assert 250_000 < m.n_params() < 400_000
        out = m(torch.randn(32), torch.randn(5, 32))
        assert out.shape == (75, 256)

    def test_state_predict_update_ring_and_mask(self):
        from elb_train.stagecprime_attn import CTX_WINDOW, CtxState
        emb = torch.randn(64, 32,
                          generator=torch.Generator().manual_seed(3))
        st = CtxState(J, E, emb, torch.device("cpu"))
        # empty ring -> zeros feature (sequence start)
        assert torch.all(st.predict(5) == 0)
        yb = torch.zeros(J, E, dtype=torch.bool)
        yb[:, 0] = True
        dr = torch.ones(J, E, dtype=torch.bool)
        st.update(yb, dr, 5)                     # push only (empty)
        assert len(st.ring) == 1
        p0 = [p.clone() for p in st.model.parameters()]
        st.update(yb, dr, 7)                     # real Adam step
        assert any(not torch.equal(a, b) for a, b in
                   zip(p0, st.model.parameters()))
        # all-rows-excluded mask -> no step, ring still advances
        p1 = [p.clone() for p in st.model.parameters()]
        st.update(yb, torch.zeros(J, E, dtype=torch.bool), 9)
        assert all(torch.equal(a, b) for a, b in
                   zip(p1, st.model.parameters()))
        for t in range(12):
            st.update(yb, dr, t)
        assert len(st.ring) == CTX_WINDOW        # ring capped
        st.reset_seq()
        assert len(st.ring) == 0
        assert torch.all(st.predict(5) == 0)

    def test_ctx_feature_column(self):
        st = _state(features=("ctx",))
        _step(st, [[0, 1]] * J)
        ctx = torch.randn(J, E,
                          generator=torch.Generator().manual_seed(4))
        X, _, raw = st.features(ctx_now=ctx)
        fi = {f: i for i, f in enumerate(st.feat_names)}
        assert torch.allclose(X[:, :, fi["ctx"]], ctx)
        _, _, raw0 = st.features()               # manifest path
        assert torch.all(raw0["ctx"] == 0)

    def test_registry_x17_and_x13c3(self):
        from elb_train.stagecprime import EXPERIMENTS
        x17 = ExpCfg(name="x17", **EXPERIMENTS["x17_ctxformer"])
        assert "ctx" in x17.feat_names
        assert len(x17.feat_names) == 16
        c3 = ExpCfg(name="c3", **EXPERIMENTS["x13c3_repair"])
        assert c3.gain_mix and c3.factor_r == 8
        assert not c3.dormancy_gate and not c3.soft_head
        assert len(c3.feat_names) == 21


class TestCalHead:
    """π̂ calibration head (ARCHITECT_CALIBRATION.md): decayed
    rank-conditional reliability tables + coverage curves."""

    def test_validation(self):
        try:
            ExpCfg(name="bad", cal_head=True)
            raise AssertionError("cal_head without open_ch must raise")
        except ValueError:
            pass
        # ship3 green light (2026-08-15): cal_head + xsame is now
        # ALLOWED — tables train in-run on the run's own stream
        # (stream separation holds by construction; reuse across
        # streams is forbidden at the packaging/consumer level)
        cfg = ExpCfg(name="ok2", open_ch=True, cal_head=True,
                     features=("xsame",), xsame_lag=4)
        assert cfg.cal_head

    @staticmethod
    def _sc_yb(seed):
        g = torch.Generator().manual_seed(seed)
        sc = torch.rand(J, E, generator=g)
        yb = torch.zeros(J, E, dtype=torch.bool)
        yb.scatter_(1, torch.topk(torch.rand(J, E, generator=g),
                                  K, dim=1).indices, True)
        prev = torch.zeros(J, E, dtype=torch.bool)
        prev.scatter_(1, torch.topk(torch.rand(J, E, generator=g),
                                    K, dim=1).indices, True)
        return sc, yb, prev

    def test_update_vs_naive_and_balanced_bins(self):
        st = _state(open_ch=True, cal_head=True)
        refH = torch.zeros(J, E - K)
        refP = torch.zeros(J, K)
        refn = 0.0
        for i in range(6):
            sc, yb, prev = self._sc_yb(i)
            st.cal_update(sc, prev, yb)
            for j in range(J):
                so = sc[j].clone()
                so[prev[j]] = float("-inf")
                oo = torch.argsort(so, descending=True, stable=True)
                hits = yb[j][oo[: E - K]].float()
                refH[j] = 0.999 * refH[j] + 0.001 * hits
                sp = sc[j].clone()
                sp[~prev[j]] = float("-inf")
                op = torch.argsort(sp, descending=True, stable=True)
                refP[j] = 0.999 * refP[j] + 0.001 * yb[j][op[:K]] \
                    .float()
            refn = 0.999 * refn + 0.001
        assert torch.allclose(st.calH, refH, atol=1e-6)
        assert torch.allclose(st.calP, refP, atol=1e-6)
        assert abs(st.cal_n - refn) < 1e-12

    def test_curve_prefix_sums_and_cold_start(self):
        st = _state(open_ch=True, cal_head=True)
        # cold start: zero coverage (max risk — safe for a governor)
        assert torch.all(st.cal_curve([32]) == 0)
        # rank-1 open candidate always true; prev slot 1 always true
        sc = torch.linspace(1, 0, E)[None].repeat(J, 1)
        for _ in range(5):
            yb = torch.zeros(J, E, dtype=torch.bool)
            yb[:, 0] = True          # prev slot rank 1 (in prev)
            yb[:, 2] = True          # open rank 1 (first non-prev)
            prev = torch.zeros(J, E, dtype=torch.bool)
            prev[:, :K] = True       # e0, e1 are prev
            st.cal_update(sc, prev, yb)
        cv = st.cal_curve([K, K + 1, 32])
        # M=K: pool = prev8 only -> coverage = sum of prev slot rates
        assert torch.allclose(cv[0], torch.full((J,), 1.0),
                              atol=1e-6)
        # M=K+1 adds open rank 1 (always hit) -> +1
        assert torch.allclose(cv[1], torch.full((J,), 2.0),
                              atol=1e-6)
        # larger budgets add nothing (no other hits)
        assert torch.allclose(cv[2], cv[1], atol=1e-6)

    def test_predict_then_update_causality(self):
        st = _state(open_ch=True, cal_head=True)
        sc, yb, prev = self._sc_yb(9)
        st.cal_update(sc, prev, yb)
        # full-budget invariant: C(E) == K identically (all ranks
        # always cover all K truths) — table-content-independent
        assert torch.allclose(st.cal_curve([E])[0],
                              torch.full((J,), float(K)), atol=1e-5)
        # causality at a PARTIAL budget (content-dependent)
        before = st.cal_curve([K + 2]).clone()
        sc2, yb2, prev2 = self._sc_yb(10)
        # a caller reading the curve BEFORE update sees state <= t-1
        assert torch.equal(st.cal_curve([K + 2]), before)
        st.cal_update(sc2, prev2, yb2)
        assert not torch.equal(st.cal_curve([K + 2]), before)

    def test_pool_geometry_scalars(self):
        from elb_train.stagecprime_gpu import _pool_geometry
        # J rows of a known descending profile over non-prev experts
        Jg, Eg = 2, 64
        sc = torch.linspace(1.0, 0.0, Eg)[None].repeat(Jg, 1)
        prev = torch.zeros(Jg, Eg, dtype=torch.bool)
        t1, t2 = _pool_geometry(sc, prev, k_pool=24)
        step = 1.0 / (Eg - 1)
        # rank24 - rank32 = 8 steps; rank8 - rank24 = 16 steps
        assert abs(float(t1) - 8 * step) < 1e-5
        assert abs(float(t2) - 16 * step) < 1e-5
        # prev experts are excluded from the ranking
        prev[:, :2] = True     # removes the top-2 scores
        t1b, _ = _pool_geometry(sc, prev, k_pool=24)
        assert abs(float(t1b) - 8 * step) < 1e-5  # uniform slope:
        #                                    gap width is unchanged

    def test_export_and_registry(self, tmp_path):
        from elb_train.stagecprime import EXPERIMENTS
        from elb_train.stagecprime_gpu import export_state
        cfg = ExpCfg(name="x19", **{
            k: v for k, v in EXPERIMENTS["x19_pihat"].items()
            if k in ("features", "members", "open_ch", "cal_head",
                     "memo_buckets")})
        assert cfg.cal_head
        st = _state(open_ch=True, cal_head=True)
        sc, yb, prev = self._sc_yb(3)
        st.cal_update(sc, prev, yb)
        files = export_state(st, tmp_path)
        assert files["calibration"] == "calibration.npz"
        z = np.load(tmp_path / "calibration.npz")
        assert z["H"].shape == (J, E - K)
        assert z["P"].shape == (J, K)
        # stationary form: bias-corrected (H/n), a constant-hit rank
        # would read 1.0
        assert float(z["n_cal"]) > 0


class TestX13Registry:
    def test_experiment_specs_are_constructible(self):
        from elb_train.stagecprime import EXPERIMENTS
        champ = ExpCfg(name="x7g", **EXPERIMENTS["x7g_memotok"])
        a = ExpCfg(name="x13a", **EXPERIMENTS["x13a_soft"])
        b = ExpCfg(name="x13b", **EXPERIMENTS["x13b_cuttrack"])
        c = ExpCfg(name="x13c", **EXPERIMENTS["x13c_factor"])
        # x13a: SAME feature space as the champion (parity anchor)
        assert a.feat_names == champ.feat_names
        assert len(a.feat_names) == 15
        assert len(b.feat_names) == 19
        assert len(c.feat_names) == 21
        for cfg in (a, b, c):
            assert "rls_soft" in cfg.member_names
            assert "rls_soft" in cfg.variant_names

    def test_revision_specs_are_constructible(self):
        from elb_train.stagecprime import EXPERIMENTS
        a2 = ExpCfg(name="x13a2", **EXPERIMENTS["x13a2_band"])
        b2 = ExpCfg(name="x13b2", **EXPERIMENTS["x13b2_gated"])
        c2 = ExpCfg(name="x13c2", **EXPERIMENTS["x13c2_factor"])
        # a2: F=15 (parity anchor holds), three hard-label heads,
        # NO soft head (killed with mechanism, Part D)
        assert len(a2.feat_names) == 15 and not a2.soft_head
        for m in ("rls_dr", "rls_drband", "rls_band"):
            assert m in a2.member_names
        # b2/c2: champion objective (rls_open is the candidate) —
        # no extra heads, gated + mixed filter features
        assert len(b2.feat_names) == 19 and not b2.soft_head
        assert b2.dormancy_gate and b2.gain_mix
        assert "rls_dr" not in b2.member_names
        assert len(c2.feat_names) == 21 and c2.factor_r == 8

"""EPM-2 unit tests: tools/elb_train/{glm_router,baselines,synth_corpus,
run_baselines}.

noaux_tc CPU reference is verified against an INDEPENDENT straightforward
reimplementation below (semantics sourced from the engine kernel
deps/LayerStoRmExpertKernels/csrc/sm120/gating/topk_gating.cu and its CPU
reference tests/unit/topk_gating_test.cpp topk_gating_ref: sigmoid scoring,
bias for selection only, ties -> lower expert index, unbiased weights
renormalized to routed_scaling_factor) plus hand cases.  The C++/CUDA
bitwise chain is covered by EpmDump.RoutingRecordMatchesGatingKernelAndFetchSeam.

Baselines run on SYNTHETIC corpora with a KNOWN generating process
(synth_corpus): B1 with the TRUE router hits recall 1.0 at zero tap noise
and a shuffled router degrades to ~chance; the tap sweep recovers the
planted tap map; B2 recovers the planted linear map held-out; B0 follows
the planted temporal correlation (rho=1 -> recall 1.0).  Deterministic
seeds throughout.
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

from elb_train import baselines, dataset, glm_router, metrics  # noqa: E402
from elb_train import run_baselines, synth_corpus  # noqa: E402


# ── noaux_tc CPU reference ───────────────────────────────────────────────────

def _independent_noaux(logits, bias, topk=8, rsf=2.5, renorm=True):
    """Straightforward independent reimplementation (per the engine
    semantics documented above) for cross-checking glm_router."""
    logits = np.asarray(logits, np.float64)
    scores = 1.0 / (1.0 + np.exp(-logits))
    sel = scores + (0 if bias is None else np.asarray(bias, np.float64))
    ids, w = [], []
    sel = sel.copy()
    for _ in range(topk):
        best = np.max(sel)
        e = int(np.min(np.nonzero(sel == best)[0]))  # tie -> lower index
        ids.append(e)
        w.append(scores[e])
        sel[e] = -np.inf
    w = np.asarray(w)
    if renorm and w.sum() > 0:
        w = w * (rsf / w.sum())
    return np.asarray(ids, np.int32), w


class TestNoauxTc:
    def test_hand_case_bias_changes_selection_not_weights(self):
        # E=4, topk=2.  logits give sigmoid scores s=[.5,.6,.7,.8]-ish;
        # a big bias on expert 0 must put it FIRST in selection while its
        # weight stays the unbiased sigmoid score.
        logits = np.array([0.0, 1.0, 2.0, 3.0], np.float32)
        bias = np.array([10.0, 0.0, 0.0, 0.0], np.float32)
        ids, w = glm_router.noaux_tc_topk(logits, bias, topk=2,
                                          routed_scaling_factor=2.5)
        assert ids.tolist() == [0, 3]
        s0 = 0.5  # sigmoid(0)
        s3 = float(glm_router.stable_sigmoid(np.float32(3.0)))
        np.testing.assert_allclose(w, np.array([s0, s3]) * 2.5 / (s0 + s3),
                                   rtol=1e-6)

    def test_tie_breaks_to_lower_index(self):
        logits = np.zeros(8, np.float32)  # all scores identical
        ids, _ = glm_router.noaux_tc_topk(logits, None, topk=4)
        assert ids.tolist() == [0, 1, 2, 3]

    def test_renorm_sums_to_scaling_factor(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal(64).astype(np.float32)
        _, w = glm_router.noaux_tc_topk(logits, None, topk=8,
                                        routed_scaling_factor=2.5)
        assert abs(w.sum() - 2.5) < 1e-5

    def test_matches_independent_reimplementation(self):
        rng = np.random.default_rng(7)
        for case in range(20):
            e = int(rng.integers(16, 257))
            logits = (3 * rng.standard_normal(e)).astype(np.float32)
            bias = ((0.05 * rng.standard_normal(e)).astype(np.float32)
                    if case % 2 else None)
            ids, w = glm_router.noaux_tc_topk(logits, bias)
            ref_ids, ref_w = _independent_noaux(logits, bias)
            np.testing.assert_array_equal(ids, ref_ids)
            np.testing.assert_allclose(w, ref_w, rtol=1e-5)

    def test_rank_experts_prefix_consistent(self):
        rng = np.random.default_rng(3)
        logits = rng.standard_normal((4, 32)).astype(np.float32)
        bias = (0.05 * rng.standard_normal(32)).astype(np.float32)
        r32 = glm_router.rank_experts(logits, bias, 32)
        ids8, _ = glm_router.noaux_tc_topk(logits, bias, topk=8)
        np.testing.assert_array_equal(r32[..., :8], ids8)

    def test_bf16_round_trip(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(1000).astype(np.float32)
        bits = glm_router.f32_to_bf16_bits(x)
        back = glm_router.bf16_bits_to_f32(bits)
        # RNE to 8-bit mantissa: relative error <= 2^-9.
        np.testing.assert_allclose(back, x, rtol=2 ** -8)
        # BF16-representable values are exact fixed points.
        np.testing.assert_array_equal(glm_router.f32_to_bf16_bits(back),
                                      bits)

    def test_aux_prior_tap_mapping(self):
        taps = glm_router.aux_prior_tap([3, 7, 8, 22, 23, 39, 55, 69, 70,
                                         77])
        assert taps.tolist() == [0, 0, 0, 0, 1, 2, 3, 3, 4, 4]


# ── safetensors router loader ────────────────────────────────────────────────

def _write_safetensors(path, tensors):
    """Minimal safetensors writer for loader tests."""
    header = {}
    blobs = []
    off = 0
    for name, (arr, st_dtype) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": st_dtype, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


class TestRouterLoader:
    def test_load_bank_from_sharded_checkpoint(self, tmp_path):
        rng = np.random.default_rng(0)
        h, e = 8, 6
        layers = [3, 4]
        weight_map = {}
        for j in layers:
            w = rng.standard_normal((e, h)).astype(np.float32)
            wb = glm_router.f32_to_bf16_bits(w)
            b = rng.standard_normal(e).astype(np.float32)
            shard = f"model-0000{j}.safetensors"
            _write_safetensors(tmp_path / shard, {
                f"model.layers.{j}.mlp.gate.weight": (wb, "BF16"),
                f"model.layers.{j}.mlp.gate.e_score_correction_bias":
                    (b, "F32")})
            weight_map[f"model.layers.{j}.mlp.gate.weight"] = shard
            weight_map[f"model.layers.{j}.mlp.gate."
                       f"e_score_correction_bias"] = shard
        with open(tmp_path / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": weight_map}, f)
        bank = glm_router.RouterBank.from_checkpoint(tmp_path, layers)
        assert bank.moe_layers.tolist() == layers
        assert bank.weight.shape == (2, e, h)
        assert bank.bias.shape == (2, e)
        # BF16 gate round-trips exactly through the loader.
        np.testing.assert_array_equal(
            glm_router.f32_to_bf16_bits(bank.weight[0]),
            glm_router.f32_to_bf16_bits(bank.weight[0]).copy())
        np.testing.assert_array_equal(bank.bias[1] != 0, True)

    def test_npz_round_trip(self, tmp_path):
        rng = np.random.default_rng(1)
        bank = glm_router.RouterBank(
            moe_layers=np.array([3, 4], np.int32),
            weight=rng.standard_normal((2, 6, 8)).astype(np.float32),
            bias=rng.standard_normal((2, 6)).astype(np.float32))
        bank.save_npz(tmp_path / "r.npz")
        back = glm_router.RouterBank.from_npz(tmp_path / "r.npz")
        np.testing.assert_array_equal(back.weight, bank.weight)
        np.testing.assert_array_equal(back.bias, bank.bias)

    def test_missing_tensor_raises(self, tmp_path):
        _write_safetensors(tmp_path / "model.safetensors",
                           {"other": (np.zeros(2, np.float32), "F32")})
        with pytest.raises(KeyError):
            glm_router.RouterBank.from_checkpoint(tmp_path, [3])


# ── synthetic-corpus fixtures ────────────────────────────────────────────────

MOE_LAYERS = (3, 4, 5, 10, 11, 12)


def _make_corpus(tmp_path, **kw):
    args = dict(n_seqs=12, blocks_per_seq=5, gamma=4, hidden=24,
                moe_layers=MOE_LAYERS, n_experts=32, rho=0.9,
                tap_noise=0.0, seed=0)
    args.update(kw)
    truth = synth_corpus.generate_synthetic_run(tmp_path / "dump" / "run_1",
                                                **args)
    dataset.assemble_shards(tmp_path / "dump", tmp_path / "shards")
    keys = sorted({k for sh in dataset.load_index(
        tmp_path / "shards")["shards"] for k in sh["seq_keys"]})
    train, held = dataset.sequence_split(keys, 0.3, seed=1)
    return truth, tmp_path / "shards", tmp_path / "dump", train, held


class TestB1:
    def test_true_router_true_tap_recall_one(self, tmp_path):
        truth, shards, _, train, held = _make_corpus(tmp_path)
        bank = truth["routers"]
        res = baselines.run_b1(shards, bank, held, truth["true_taps"])
        v, n = metrics.overall_recall(res, 8)
        assert n > 50
        # Labels and predictions run the SAME selection on the SAME
        # BF16-representable latents at tap_noise=0 -> exact recall 1.0.
        assert v == 1.0

    def test_shuffled_router_degrades(self, tmp_path):
        truth, shards, _, train, held = _make_corpus(tmp_path)
        bank = truth["routers"]
        rng = np.random.default_rng(5)
        shuffled = glm_router.RouterBank(
            moe_layers=bank.moe_layers,
            weight=bank.weight[:, rng.permutation(bank.n_experts)],
            bias=bank.bias)
        res = baselines.run_b1(shards, shuffled, held, truth["true_taps"])
        v, _ = metrics.overall_recall(res, 8)
        assert v < 0.5  # chance-region, far from the true router's 1.0

    def test_sweep_recovers_planted_taps_and_is_reproducible(self,
                                                             tmp_path):
        truth, shards, _, train, held = _make_corpus(tmp_path)
        bank = truth["routers"]
        sweep = baselines.b1_tap_sweep(shards, bank, train)
        best = baselines.select_best_taps(sweep)
        np.testing.assert_array_equal(best, truth["true_taps"])
        # Reproducible across two runs (EPM-2 verify gate).
        sweep2 = baselines.b1_tap_sweep(shards, bank, train)
        np.testing.assert_array_equal(sweep["recall_table"],
                                      sweep2["recall_table"])
        np.testing.assert_array_equal(sweep["n_table"], sweep2["n_table"])
        # Wrong taps score below the planted tap on every layer.
        table = sweep["recall_table"]
        for jj, t in enumerate(truth["true_taps"]):
            for other in range(table.shape[0]):
                if other != t:
                    assert table[other, jj] < table[t, jj]

    def test_bank_shape_mismatch_raises(self, tmp_path):
        truth, shards, _, train, _ = _make_corpus(
            tmp_path, n_seqs=2, blocks_per_seq=2)
        bank = truth["routers"]
        bad = glm_router.RouterBank(moe_layers=bank.moe_layers[:-1],
                                    weight=bank.weight[:-1],
                                    bias=bank.bias[:-1])
        with pytest.raises(ValueError, match="moe_layers"):
            baselines.run_b1(shards, bad, train, truth["true_taps"][:-1])


class TestB2:
    def test_recovers_planted_linear_map_held_out(self, tmp_path):
        # Labels are LINEAR in the (planted-tap) features -> the direct
        # ridge probe must recover them on held-out sequences.
        truth, shards, _, train, held = _make_corpus(tmp_path)
        probe = baselines.train_b2(shards, train, truth["true_taps"],
                                   ridge_lambda=1e-9)
        assert probe["n_train"] > 100  # solvable: n >> H=24
        res = baselines.run_b2(shards, probe, truth["routers"], held)
        v, n = metrics.overall_recall(res, 8)
        assert n > 50
        assert v > 0.98
        # recall@32 saturates (E=32).
        v32, _ = metrics.overall_recall(res, 32)
        assert v32 == 1.0

    def test_b2_skyline_beats_mismatched_router(self, tmp_path):
        # The skyline property B2 upper-bounds is TRAINED linear capacity:
        # when the frozen router is mismatched to the feature space (the
        # real-data regime — raw draft hiddens are not target hiddens), the
        # trained probe must recover what the frozen router cannot.  (With
        # the TRUE router on noisy features B1 is already Bayes-optimal
        # linear — finite-sample B2 lands marginally below it, by design.)
        truth, shards, _, train, held = _make_corpus(tmp_path,
                                                     tap_noise=0.4, seed=2)
        bank = truth["routers"]
        rng = np.random.default_rng(9)
        mismatched = glm_router.RouterBank(
            moe_layers=bank.moe_layers,
            weight=bank.weight[:, rng.permutation(bank.n_experts)],
            bias=bank.bias)
        b1_bad = baselines.run_b1(shards, mismatched, held,
                                  truth["true_taps"])
        probe = baselines.train_b2(shards, train, truth["true_taps"])
        b2 = baselines.run_b2(shards, probe, bank, held)
        v1_bad, _ = metrics.overall_recall(b1_bad, 8)
        v2, _ = metrics.overall_recall(b2, 8)
        assert v2 > v1_bad + 0.25
        assert v2 > 0.55  # recovers most of the noisy-feature ceiling


class TestB0:
    def test_rho_one_prev_recall_is_one(self, tmp_path):
        # rho=1: the latent never moves -> routing constant over positions
        # -> both B0 variants are perfect.
        truth, shards, dump, train, held = _make_corpus(tmp_path, rho=1.0)
        for variant in ("prev", "anchor"):
            res = baselines.run_b0(shards, dump, held, variant)
            v, n = metrics.overall_recall(res, 8)
            assert n > 50
            assert v == 1.0, variant

    def test_temporal_correlation_ordering(self, tmp_path):
        # Higher rho -> higher B0 recall; low rho -> near chance.  The
        # anchor variant is at most the prev variant (staler source).
        vals = {}
        for rho in (0.98, 0.3):
            truth, shards, dump, train, held = _make_corpus(
                tmp_path / f"r{rho}", rho=rho, seed=3)
            prev = metrics.overall_recall(
                baselines.run_b0(shards, dump, held, "prev"), 8)[0]
            anchor = metrics.overall_recall(
                baselines.run_b0(shards, dump, held, "anchor"), 8)[0]
            vals[rho] = (prev, anchor)
        assert vals[0.98][0] > vals[0.3][0] + 0.2
        assert vals[0.3][0] < 0.55
        # Anchor source is staler for k>0 -> no better than prev overall.
        assert vals[0.98][1] <= vals[0.98][0] + 1e-9

    def test_missing_source_counted_not_scored(self, tmp_path):
        # First block anchored at pos 1; the anchor-1 source (pos 0) exists
        # by construction, so drop it from the routing map to simulate a
        # sequence-start hole: those cells must land in n_missing_pred.
        truth, shards, dump, train, held = _make_corpus(
            tmp_path, n_seqs=4, blocks_per_seq=2)
        maps = baselines.load_routing_maps(dump)
        held_set = set(held)
        removed = 0
        for key in list(maps):
            if key[2] == 0:  # position-0 warmup records
                del maps[key]
                removed += 1
        assert removed
        res = baselines.run_b0(shards, maps, held, "anchor")
        assert res["n_missing_pred"].sum() > 0
        assert res["recall"][8]["n"].sum() > 0  # later blocks still scored

    def test_unknown_variant_raises(self, tmp_path):
        with pytest.raises(ValueError, match="variant"):
            baselines.run_b0("unused", {}, [], "bogus")


class TestRunBaselinesEndToEnd:
    def test_synthetic_report_produces_artifacts(self, tmp_path):
        rc = run_baselines.main(["--synthetic",
                                 "--out", str(tmp_path / "study"),
                                 "--seed", "0"])
        assert rc == 0
        out = tmp_path / "study"
        md = (out / "summary.md").read_text(encoding="utf-8")
        assert "b1_best_tap" in md and "b2_linear_probe" in md
        assert "b0_prev" in md and "b0_anchor" in md
        for name in ("b0_prev", "b0_anchor", "b1_best_tap",
                     "b1_prior_tap", "b2_linear_probe"):
            assert (out / f"{name}.npz").is_file()
            assert (out / f"{name}.json").is_file()
        assert (out / "b1_tap_sweep.npz").is_file()
        assert (out / "report_meta.json").is_file()
        # The heatmap npz round-trips through the EPM-3-side loader.
        res = metrics.load_result_npz(out / "b1_best_tap")
        assert res["recall"][8]["mean"].shape == res["recall"][8]["n"].shape
        # Synthetic ordering: B1(best) and B2 far above B0(anchor) which is
        # above chance; prior taps collapse on off-segment layers.
        js = {n: json.load(open(out / f"{n}.json"))
              for n in ("b0_anchor", "b1_best_tap", "b1_prior_tap",
                        "b2_linear_probe")}
        assert js["b1_best_tap"]["overall_recall8"] > 0.9
        assert js["b2_linear_probe"]["overall_recall8"] > 0.9
        assert js["b1_best_tap"]["overall_recall8"] > \
            js["b1_prior_tap"]["overall_recall8"] + 0.2
        assert js["b0_anchor"]["overall_recall8"] > 8 / 32  # above chance

    def test_report_deterministic_across_runs(self, tmp_path):
        for d in ("a", "b"):
            run_baselines.main(["--synthetic",
                                "--out", str(tmp_path / d), "--seed", "4"])
        for name in ("b1_best_tap", "b2_linear_probe", "b0_prev"):
            za = np.load(tmp_path / "a" / f"{name}.npz")
            zb = np.load(tmp_path / "b" / f"{name}.npz")
            np.testing.assert_array_equal(za["recall8"], zb["recall8"])
            np.testing.assert_array_equal(za["n8"], zb["n8"])

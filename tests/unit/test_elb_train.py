"""EPM-3 unit tests: tools/elb_train/train.py (PLAN.md Phase 29 EPM-3
verification gates).

Gates covered here:
  1. overfit sanity — planted-linear synthetic, IN-SAMPLE recall@8 >= 0.99;
  2. recoverability — true routers + planted orthogonal feature map, the
     trained Tier-1 adapter reaches recall ~1 HELD-OUT and matches the B2
     closed-form skyline (the model class contains the truth);
  3. W plumbing — zero-weight cells produce exactly zero gradient (loss
     path); rank schedules follow W under the budget (capacity path is
     also covered in test_elb_model.py);
  4. resume — bitwise-identical eval after save->load; a resumed run
     reproduces the uninterrupted run bit-for-bit (loss continuity is a
     corollary and asserted explicitly);
  5. determinism — two runs, same seed => identical final weights and
     metrics;
  plus the direct-head arm (decision D), the memory-mapped shard reader,
  token-mismatch weighting semantics (TD-EPM-REJECTED-LABELS), the
  standardize stats pass, and the --evaluate entrypoint's EPM-2 artifact
  contract.

Small dims throughout (hidden 8-16, 32 experts, <= 6 layers); the gate
training runs take a few seconds each on CPU.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

import numpy as np
import pytest
import torch

_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "tools"))

from elb_train import baselines, dataset, glm_router, metrics  # noqa: E402
from elb_train import model as model_mod  # noqa: E402
from elb_train import synth_corpus, train as train_mod  # noqa: E402


# ── config helpers ───────────────────────────────────────────────────────────

def gate_cfg(out_dir, **over):
    """The planted-linear synthetic gate config (module docstring)."""
    cfg = {
        "seed": 0, "device": "cpu", "out_dir": str(out_dir),
        "data": {"synthetic": {
            "n_seqs": 16, "blocks_per_seq": 8, "gamma": 4, "n_taps": 5,
            "hidden": 16, "moe_layers": [3, 4, 5, 10, 24, 40],
            "n_experts": 32, "rho": 0.9, "tap_noise": 0.0,
            "feature_map": "linear", "tap_assignment": "prior",
            "seed": 1},
            "held_out_fraction": 0.25, "batch_blocks": 16},
        "model": {"normalization": {"kind": "none"},
                  "adapter": {"r_base": 16,
                              "delta": {"enabled": True,
                                        "budget_params": 20000,
                                        "max_rank": 8}}},
        "wmap": {"kind": "directive", "accept_curve": [0.55, 0.4, 0.25]},
        "loss": {"kl_weight": 1.0, "bce_weight": 0.0, "mse_weight": 1.0},
        "optim": {"lr": 0.02, "weight_decay": 0.0, "warmup_steps": 20,
                  "max_steps": 1200, "clip_grad_norm": 1.0},
        "eval": {"every_steps": 0}, "checkpoint": {"every_steps": 0},
        "log_every": 400,
    }
    return train_mod._deep_update(
        train_mod._deep_update(train_mod.DEFAULT_CONFIG, cfg), over)


def tiny_cfg(out_dir, **over):
    """Fast config for resume/determinism mechanics (not accuracy)."""
    return gate_cfg(out_dir, **train_mod._deep_update({
        "data": {"synthetic": {"n_seqs": 6, "blocks_per_seq": 4,
                               "gamma": 3, "hidden": 8,
                               "moe_layers": [3, 4, 10],
                               "n_experts": 16, "seed": 2},
                 "batch_blocks": 4},
        "model": {"adapter": {"r_base": 4,
                              "delta": {"enabled": True,
                                        "budget_params": 2000,
                                        "max_rank": 4}}},
        "optim": {"max_steps": 40, "warmup_steps": 4},
        "log_every": 1,
    }, over))


@pytest.fixture(scope="module")
def gate_run(tmp_path_factory):
    """One trained gate run shared by the accuracy-gate tests."""
    out = tmp_path_factory.mktemp("epm3-gate")
    tr = train_mod.Trainer(gate_cfg(out))
    tr.train()
    return tr


# ── shard mmap reader ────────────────────────────────────────────────────────

class TestShardMmap:
    def test_mmap_matches_np_load(self, tmp_path):
        synth_corpus.generate_synthetic_run(
            tmp_path / "dump", n_seqs=3, blocks_per_seq=3, gamma=3,
            hidden=8, moe_layers=(3, 4), n_experts=16, seed=0)
        dataset.assemble_shards(tmp_path / "dump", tmp_path / "shards")
        path = tmp_path / "shards" / "shard_00000.npz"
        mm = train_mod.open_shard_mmap(path)
        ref = np.load(path)
        assert set(mm) == set(ref.files)
        for k in ref.files:
            np.testing.assert_array_equal(np.asarray(mm[k]), ref[k],
                                          err_msg=k)


# ── per-cell weights (gate 3: W plumbing, loss path) ─────────────────────────

def _fake_batch(b=2, g=3, j=2, e=8, k_top=4, accepted=(1, -1),
                match=(True, False)):
    rng = np.random.default_rng(0)
    return {
        "labels_logits": rng.standard_normal((b, g, j, e)).astype(
            np.float16),
        "labels_top_ids": rng.integers(0, e, (b, g, j, k_top)).astype(
            np.int32),
        "label_mask": np.ones((b, g), bool),
        "accepted_len": np.asarray(accepted, np.int32),
        "tokens_match": np.asarray(match, bool),
    }


class TestCellWeights:
    def test_token_match_semantics(self):
        # block 0: manifest joined, tokens_match, accepted_len=1
        #   -> k <= 1 matched (incl. the first REJECTED draft feed), k=2 not
        # block 1: no manifest (accepted_len=-1) -> unknown provenance
        batch = _fake_batch()
        m = train_mod.token_match_mask(batch, unknown_matched=True)
        np.testing.assert_array_equal(m[0], [True, True, False])
        np.testing.assert_array_equal(m[1], [True, True, True])
        m2 = train_mod.token_match_mask(batch, unknown_matched=False)
        np.testing.assert_array_equal(m2[1], [False, False, False])
        # tokens_match=False poisons the whole block
        batch["accepted_len"][1] = 2
        m3 = train_mod.token_match_mask(batch, unknown_matched=True)
        np.testing.assert_array_equal(m3[1], [False, False, False])

    def test_cell_weights_compose_wmap_mask_and_mismatch(self):
        batch = _fake_batch()
        batch["label_mask"][0, 2] = False
        w = torch.tensor([[1.0, 0.5], [0.4, 0.2], [0.1, 0.05]])
        cw = train_mod.cell_weights(
            batch, w, {"token_mismatch_weight": 0.25,
                       "unknown_provenance_matched": True}, "cpu")
        assert cw.shape == (2, 3, 2)
        # block 0, k=0/1 matched -> W rows; k=2 unlabeled -> 0
        np.testing.assert_allclose(cw[0, 0], [1.0, 0.5])
        np.testing.assert_allclose(cw[0, 1], [0.4, 0.2])
        np.testing.assert_allclose(cw[0, 2], [0.0, 0.0])
        # block 1 unknown->matched, all labeled -> plain W
        np.testing.assert_allclose(cw[1, 2].numpy(), [0.1, 0.05],
                                   rtol=1e-6)
        # mismatch down-weighting: make block 0 k=2 labeled-but-mismatched
        batch["label_mask"][0, 2] = True
        cw2 = train_mod.cell_weights(
            batch, w, {"token_mismatch_weight": 0.25,
                       "unknown_provenance_matched": True}, "cpu")
        np.testing.assert_allclose(cw2[0, 2].numpy(),
                                   [0.1 * 0.25, 0.05 * 0.25], rtol=1e-6)

    def test_zero_weight_cells_produce_zero_grad(self):
        """Gate 3 (loss path): positions with zero cell weight contribute
        EXACTLY zero gradient — per-position delta adapters of masked
        positions stay untouched."""
        dims = model_mod.Dims(hidden=8, n_taps=2, n_experts=16,
                              max_gamma=3,
                              moe_layers=np.asarray([3, 4], np.int32))
        rng = np.random.default_rng(0)
        bank = glm_router.RouterBank(
            moe_layers=dims.moe_layers,
            weight=rng.standard_normal((2, 16, 8)).astype(np.float32),
            bias=np.zeros((2, 16), np.float32))
        cfg = {"arm": "router", "normalization": {"kind": "none"},
               "taps": {"mode": "fixed", "map": 0},
               "adapter": {"r_base": 4,
                           "delta": {"enabled": True,
                                     "budget_params": 2000,
                                     "min_rank": 1, "max_rank": 4}}}
        m = model_mod.build_model(cfg, dims, {"kind": "uniform"}, bank,
                                  seed=0)
        # non-zero up-projections so gradients could flow if weighted
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn(p.shape, generator=torch.
                                          Generator().manual_seed(1)))
        batch = {
            "labels_logits": rng.standard_normal(
                (2, 3, 2, 16)).astype(np.float16),
            "labels_top_ids": rng.integers(0, 16, (2, 3, 2, 8)).astype(
                np.int32),
            "label_mask": np.array([[True, False, False],
                                    [True, False, False]]),
            "accepted_len": np.asarray([2, 2], np.int32),
            "tokens_match": np.asarray([True, True], bool),
        }
        x = torch.randn(2, 3, 2, 8,
                        generator=torch.Generator().manual_seed(2))
        w_cell = train_mod.cell_weights(batch, m.wmap,
                                        {"token_mismatch_weight": 1.0},
                                        "cpu")
        loss, _ = train_mod.compute_loss(
            m(x), batch, w_cell,
            {"kl_weight": 1.0, "bce_weight": 0.5, "mse_weight": 0.1})
        loss.backward()
        for key, p in m.delta.down.items():
            g = p.grad
            if key.startswith("k0_"):
                assert g is not None and g.abs().sum() > 0, key
            else:
                assert g is None or torch.all(g == 0), \
                    f"{key}: masked position leaked gradient"
        # explicit W column zero -> that layer's adapter gets no gradient
        m.zero_grad(set_to_none=True)
        w0 = m.wmap.clone()
        w0[:, 1] = 0.0
        batch["label_mask"][:] = True
        w_cell = (torch.as_tensor(batch["label_mask"],
                                  dtype=torch.float32)[:, :, None]
                  * w0[None])
        loss, _ = train_mod.compute_loss(
            m(x), batch, w_cell, {"kl_weight": 1.0, "bce_weight": 0.0})
        loss.backward()
        # per_layer groups: base adapter group 1 == layer 1 (zero-weight)
        assert torch.all(m.base_down.grad[1] == 0)
        assert m.base_down.grad[0].abs().sum() > 0


class TestLoss:
    def test_kl_zero_iff_pred_matches_labels(self):
        batch = _fake_batch()
        pred = torch.as_tensor(
            np.ascontiguousarray(batch["labels_logits"]),
            dtype=torch.float32)
        w = torch.ones(3, 2)
        cw = train_mod.cell_weights(batch, w,
                                    {"token_mismatch_weight": 1.0}, "cpu")
        loss, comps = train_mod.compute_loss(
            pred, batch, cw, {"kl_weight": 1.0, "bce_weight": 0.0})
        assert comps["kl"] == pytest.approx(0.0, abs=1e-6)
        loss2, comps2 = train_mod.compute_loss(
            pred + torch.randn(pred.shape,
                               generator=torch.Generator().manual_seed(3)),
            batch, cw, {"kl_weight": 1.0, "bce_weight": 0.0})
        assert comps2["kl"] > 0.01

    def test_bce_rewards_member_logits(self):
        batch = _fake_batch()
        e = batch["labels_logits"].shape[-1]
        member = np.full((2, 3, 2, e), -5.0, np.float32)
        for b in range(2):
            for g in range(3):
                for j in range(2):
                    member[b, g, j, batch["labels_top_ids"][b, g, j]] = 5.0
        cw = train_mod.cell_weights(batch, torch.ones(3, 2),
                                    {"token_mismatch_weight": 1.0}, "cpu")
        _, good = train_mod.compute_loss(
            torch.as_tensor(member), batch, cw,
            {"kl_weight": 0.0, "bce_weight": 1.0})
        _, bad = train_mod.compute_loss(
            torch.as_tensor(-member), batch, cw,
            {"kl_weight": 0.0, "bce_weight": 1.0})
        assert good["bce"] < 0.05 < bad["bce"]

    def test_all_masked_returns_zero_loss(self):
        batch = _fake_batch()
        batch["label_mask"][:] = False
        cw = train_mod.cell_weights(batch, torch.ones(3, 2),
                                    {"token_mismatch_weight": 1.0}, "cpu")
        loss, comps = train_mod.compute_loss(
            torch.zeros(2, 3, 2, 8), batch, cw, {"kl_weight": 1.0})
        assert float(loss) == 0.0 and comps["wsum"] == 0.0


class TestLrSchedule:
    def test_warmup_cosine_constant(self):
        cfg = {"lr": 1.0, "warmup_steps": 10, "schedule": "cosine",
               "max_steps": 110, "min_lr_ratio": 0.0}
        assert train_mod.lr_at(0, cfg) == pytest.approx(0.1)
        assert train_mod.lr_at(9, cfg) == pytest.approx(1.0)
        assert train_mod.lr_at(10, cfg) == pytest.approx(1.0)
        assert train_mod.lr_at(60, cfg) == pytest.approx(0.5, abs=0.01)
        assert train_mod.lr_at(110, cfg) == pytest.approx(0.0, abs=1e-6)
        const = {**cfg, "schedule": "constant"}
        assert train_mod.lr_at(105, const) == pytest.approx(1.0)


# ── accuracy gates (shared trained run) ──────────────────────────────────────

class TestAccuracyGates:
    def test_overfit_sanity_in_sample_recall(self, gate_run):
        """Gate 1: training drives IN-SAMPLE (train split) recall@8 to
        ~1 on the planted-linear synthetic (Tier 0 is ~0.2 here)."""
        res = gate_run.eval_and_save("gate_train", split="train")
        r8, n = metrics.overall_recall(res, 8)
        assert n > 1000
        assert r8 >= 0.99, f"in-sample recall@8 {r8:.4f} < 0.99"

    def test_recoverability_held_out_matches_b2_skyline(self, gate_run):
        """Gate 2: with the true routers + planted orthogonal map the
        trained adapter reaches recall ~1 held-out and matches the B2
        closed-form linear skyline (the model class contains the
        truth)."""
        res = gate_run.eval_and_save("gate_held", split="held_out")
        r8, n = metrics.overall_recall(res, 8)
        assert n > 200
        assert r8 >= 0.985, f"held-out recall@8 {r8:.4f} < 0.985"
        shards = gate_run.cfg["data"]["shards"]
        idx = dataset.load_index(shards)
        taps = glm_router.aux_prior_tap(idx["moe_layers"])
        probe = baselines.train_b2(shards, gate_run.train_keys, taps,
                                   ridge_lambda=1e-8)
        bank = glm_router.RouterBank.from_npz(
            pathlib.Path(gate_run.out) / "routers.npz")
        resb = baselines.run_b2(shards, probe, bank, gate_run.held_keys)
        b2, _ = metrics.overall_recall(resb, 8)
        assert r8 >= b2 - 0.02, \
            f"trained {r8:.4f} vs B2 skyline {b2:.4f}"
        # And Tier 0 (init) really was far below — the adapter is
        # load-bearing on this corpus, not a refinement.
        res0 = baselines.run_b1(shards, bank, gate_run.held_keys, taps)
        t0, _ = metrics.overall_recall(res0, 8)
        assert t0 < 0.5

    def test_evaluate_entrypoint_emits_epm2_artifact_contract(
            self, gate_run, tmp_path):
        """--evaluate on the run's latest checkpoint produces the
        metrics.py npz/json contract (run_baselines-comparable)."""
        cfg_path = tmp_path / "cfg.json"
        cfg = dict(gate_run.cfg)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        rc = train_mod.main(["--config", str(cfg_path),
                             "--out", str(gate_run.out),
                             "--evaluate", "--eval-split", "held_out",
                             "--eval-name", "eval_cli"])
        assert rc == 0
        z = np.load(pathlib.Path(gate_run.out) / "eval_cli.npz")
        for key in ("recall8", "n8", "recall16", "recall32",
                    "n_missing_pred", "union_coverage", "union_n",
                    "calib_bin_edges", "calib_bin_n", "moe_layers"):
            assert key in z, key
        with open(pathlib.Path(gate_run.out) / "eval_cli.json",
                  encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["overall_recall8"] >= 0.985
        assert summary["meta"]["split"] == "held_out"
        # CLI eval must be bitwise-identical to the in-process eval of
        # the same weights (save -> load round trip changed nothing).
        z0 = np.load(pathlib.Path(gate_run.out) / "gate_held.npz")
        np.testing.assert_array_equal(z["recall8"], z0["recall8"])
        np.testing.assert_array_equal(z["calib_bin_n"],
                                      z0["calib_bin_n"])

    def test_direct_head_arm_trains(self, tmp_path):
        """Decision (D) flagged arm: the direct low-rank head (no router
        replica) trains to high held-out recall on the same corpus."""
        cfg = gate_cfg(tmp_path / "direct", **{
            "model": {"arm": "direct",
                      "direct_head": {"r_base": 16, "logit_bias": True,
                                      "delta": {"enabled": True,
                                                "budget_params": 20000,
                                                "max_rank": 8}}},
            "optim": {"max_steps": 600}})
        tr = train_mod.Trainer(cfg)
        tr.train()
        res = tr.eval_and_save("gate_held", split="held_out")
        r8, _ = metrics.overall_recall(res, 8)
        assert r8 >= 0.95, f"direct-head held-out recall@8 {r8:.4f}"


# ── resume + determinism (gates 4/5) ─────────────────────────────────────────

def _final_state(tr):
    return {k: v.clone() for k, v in tr.model.state_dict().items()}


class TestResumeAndDeterminism:
    def test_resume_is_bitwise_and_loss_continuous(self, tmp_path):
        # uninterrupted 40-step run
        tr_a = train_mod.Trainer(tiny_cfg(tmp_path / "a"))
        tr_a.train()
        state_a = _final_state(tr_a)
        losses_a = [json.loads(l) for l in
                    open(tmp_path / "a" / "train_log.jsonl")]
        # interrupted: 20 steps, checkpoint, fresh Trainer, resume, 40
        tr_b1 = train_mod.Trainer(tiny_cfg(tmp_path / "b"))
        tr_b1.train(max_steps=20)
        ckpt = tmp_path / "b" / "ckpt_step0000020"
        assert ckpt.is_dir()
        tr_b2 = train_mod.Trainer(tiny_cfg(tmp_path / "b"))
        tr_b2.resume(ckpt)
        assert tr_b2.state.step == 20
        # bitwise-identical eval after save -> load
        x = torch.randn(1, 3, 5, 8,
                        generator=torch.Generator().manual_seed(0))
        ids1, sc1 = tr_b1.model.predict(x)
        ids2, sc2 = tr_b2.model.predict(x)
        np.testing.assert_array_equal(ids1, ids2)
        np.testing.assert_array_equal(sc1, sc2)
        tr_b2.train()
        state_b = _final_state(tr_b2)
        assert set(state_a) == set(state_b)
        for k in state_a:
            assert torch.equal(state_a[k], state_b[k]), \
                f"{k}: resumed run diverged from uninterrupted run"
        # loss continuity: the resumed run's step-21..40 losses equal the
        # uninterrupted run's (no spike; bitwise-identical trajectory)
        losses_b = [json.loads(l) for l in
                    open(tmp_path / "b" / "train_log.jsonl")]
        by_step_a = {r["step"]: r["loss"] for r in losses_a
                     if "loss" in r}
        by_step_b = {r["step"]: r["loss"] for r in losses_b
                     if "loss" in r}
        for s in range(21, 41):
            assert by_step_b[s] == pytest.approx(by_step_a[s],
                                                 rel=1e-6), s

    def test_two_runs_same_seed_identical_metrics(self, tmp_path):
        results = []
        for name in ("r1", "r2"):
            tr = train_mod.Trainer(tiny_cfg(tmp_path / name))
            tr.train()
            res = tr.eval_and_save("final", split="held_out")
            results.append((_final_state(tr),
                            metrics.overall_recall(res, 8)))
        state1, m1 = results[0]
        state2, m2 = results[1]
        assert m1 == m2
        for k in state1:
            assert torch.equal(state1[k], state2[k]), k

    def test_different_seed_differs(self, tmp_path):
        tr1 = train_mod.Trainer(tiny_cfg(tmp_path / "s0"))
        tr1.train(max_steps=5)
        tr2 = train_mod.Trainer(tiny_cfg(tmp_path / "s1", seed=1))
        tr2.train(max_steps=5)
        s1, s2 = _final_state(tr1), _final_state(tr2)
        assert any(not torch.equal(s1[k], s2[k]) for k in s1)


# ── standardize stats pass ───────────────────────────────────────────────────

class TestStandardize:
    def test_trainer_fills_stats_from_train_split(self, tmp_path):
        cfg = tiny_cfg(tmp_path / "std", **{
            "model": {"normalization": {"kind": "standardize"}}})
        tr = train_mod.Trainer(cfg)
        assert int(tr.model.norm.stats_set) == 1
        # cross-check vs a direct computation over the same rows
        mean, std = train_mod.compute_standardize_stats(tr.store,
                                                        tr.train_rows)
        np.testing.assert_allclose(tr.model.norm.mean.numpy(), mean,
                                   rtol=1e-5)
        tr.train(max_steps=3)  # trains with the normalizer in place

    def test_checkpoint_carries_stats(self, tmp_path):
        cfg = tiny_cfg(tmp_path / "std2", **{
            "model": {"normalization": {"kind": "standardize"}},
            "optim": {"max_steps": 4}})
        tr = train_mod.Trainer(cfg)
        tr.train()
        ckpt = tmp_path / "std2" / "ckpt_step0000004"
        bank = glm_router.RouterBank.from_npz(tmp_path / "std2"
                                              / "routers.npz")
        m2, _ = model_mod.load_model(ckpt / "model", bank)
        assert int(m2.norm.stats_set) == 1
        assert torch.equal(m2.norm.mean, tr.model.norm.mean)
        assert torch.equal(m2.norm.inv_std, tr.model.norm.inv_std)


# ── misc mechanics ───────────────────────────────────────────────────────────

class TestMechanics:
    def test_config_overrides(self, tmp_path):
        p = tmp_path / "c.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"seed": 3, "optim": {"lr": 0.5}}, f)
        cfg = train_mod.load_config(p, ["optim.max_steps=7",
                                        "model.arm=direct",
                                        "data.in_sample=true"])
        assert cfg["seed"] == 3 and cfg["optim"]["lr"] == 0.5
        assert cfg["optim"]["max_steps"] == 7
        assert cfg["model"]["arm"] == "direct"
        assert cfg["data"]["in_sample"] is True
        # defaults survive the deep-merge
        assert cfg["loss"]["kl_weight"] == 1.0

    def test_checkpoint_pruning_keeps_latest(self, tmp_path):
        cfg = tiny_cfg(tmp_path / "pr", **{
            "optim": {"max_steps": 6},
            "checkpoint": {"every_steps": 2, "keep": 2}})
        tr = train_mod.Trainer(cfg)
        tr.train()
        dirs = sorted(p.name for p in (tmp_path / "pr").glob("ckpt_*"))
        assert dirs == ["ckpt_step0000004", "ckpt_step0000006"]
        latest = (tmp_path / "pr" / "latest").read_text().strip()
        assert latest == "ckpt_step0000006"

    def test_degenerate_split_requires_in_sample(self, tmp_path):
        cfg = tiny_cfg(tmp_path / "deg")
        cfg["data"]["synthetic"]["n_seqs"] = 1
        with pytest.raises(ValueError, match="in_sample"):
            train_mod.Trainer(cfg)
        cfg["data"]["in_sample"] = True
        shutil.rmtree(tmp_path / "deg")
        tr = train_mod.Trainer(cfg)
        assert tr.train_keys == tr.held_keys


class TestLogitsStoredGuard:
    def test_kl_on_logitless_corpus_fails_closed(self, tmp_path):
        """EPM-5 disk-bounded corpora (store_logits=False) must refuse a
        KL-weighted loss — it would distill against all-zero logits."""
        from elb_train import dataset as ds
        from test_elb_dataset import make_dump
        dump = tmp_path / "dump"
        dump.mkdir()
        make_dump(dump, seqs=(1, 2, 3, 4), blocks_per_seq=3, gamma=3)
        shards = tmp_path / "shards"
        ds.assemble_shards(dump, shards, store_logits=False, compress=True)
        cfg = tiny_cfg(tmp_path / "out", **{
            "data": {"synthetic": None, "shards": str(shards),
                     "in_sample": True},
            "loss": {"kl_weight": 1.0, "bce_weight": 0.25},
        })
        with pytest.raises(ValueError, match="logits_stored"):
            train_mod.Trainer(cfg)
        # BCE-primary is accepted on the same corpus (direct arm — no
        # router bank needed on a real-shard corpus).
        cfg2 = tiny_cfg(tmp_path / "out2", **{
            "data": {"synthetic": None, "shards": str(shards),
                     "in_sample": True},
            "loss": {"kl_weight": 0.0, "bce_weight": 1.0},
            "model": {"arm": "direct",
                      "direct_head": {"r_base": 4, "logit_bias": True,
                                      "use_router_bias_for_selection":
                                          False,
                                      "delta": {"enabled": False}}},
        })
        train_mod.Trainer(cfg2)  # constructs without raising

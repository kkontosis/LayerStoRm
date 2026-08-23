"""EPM-3 unit tests: tools/elb_train/{wmap,model}.py.

W[k][j] directive properties (PLAN.md Phase 29 EPM-3 verify gates):
k=0 row maximal everywhere; k=1 early-layer boost above its own baseline;
monotone survival taper; capacity allocation follows W ordering under the
parameter budget.  Model: init-is-Tier-0 (LoRA zero-init up + residual ->
the router arm reproduces the frozen-router probe exactly at init),
INV-EPM-SIDE (router replicas are non-trainable, untouched by training,
and never serialized), save/load bitwise round trip, both output arms,
tap fusion, normalization variants.  Small dims throughout.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
import torch

_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "tools"))

from elb_train import glm_router, wmap  # noqa: E402
from elb_train import model as model_mod  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def make_dims(h=16, l=5, e=32, g=4, layers=(3, 4, 5, 10, 24, 40)):
    return model_mod.Dims(hidden=h, n_taps=l, n_experts=e, max_gamma=g,
                          moe_layers=np.asarray(layers, np.int32))


def make_bank(dims, seed=0):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((dims.n_layers, dims.n_experts, dims.hidden))
         / np.sqrt(dims.hidden)).astype(np.float32)
    b = (0.02 * rng.standard_normal((dims.n_layers,
                                     dims.n_experts))).astype(np.float32)
    return glm_router.RouterBank(moe_layers=dims.moe_layers, weight=w,
                                 bias=b)


def base_cfg(**adapter):
    return {"arm": "router",
            "normalization": {"kind": "none"},
            "taps": {"mode": "fixed", "map": "prior"},
            "layer_groups": "per_layer",
            "adapter": {"r_base": 8, "residual": True,
                        "activation": "none",
                        "delta": {"enabled": False}, **adapter}}


def rand_features(dims, b=2, g=None, seed=0):
    rng = np.random.default_rng(seed)
    g = g or dims.max_gamma
    return torch.as_tensor(rng.standard_normal(
        (b, g, dims.n_taps, dims.hidden)).astype(np.float32))


# ── W[k][j] value map ────────────────────────────────────────────────────────

class TestWMap:
    def test_survival_is_cumprod_of_accept_curve(self):
        s = wmap.survival_from_curve((0.5, 0.4, 0.2), 4)
        np.testing.assert_allclose(s, [1.0, 0.5, 0.2, 0.04])

    def test_survival_curve_extends_with_last_entry(self):
        s = wmap.survival_from_curve((0.5,), 4)
        np.testing.assert_allclose(s, [1.0, 0.5, 0.25, 0.125])

    def test_directive_defaults_shape_and_ordering(self):
        layers = list(range(3, 78))  # 75 GLM-5.2 MoE layers
        w = wmap.build_wmap({"kind": "directive"}, 7, layers)
        assert w.shape == (7, 75)
        # k=0 row maximal on ALL layers.
        assert np.all(w[0] == 1.0)
        assert np.all(w[0:1] >= w), "k=0 must dominate every other row"
        # k=1 boosted on its first 4 layers (absolute 3..6) ABOVE its own
        # non-boosted baseline.
        assert np.all(w[1, :4] > w[1, 4:].max())
        assert np.all(w[1, :4] == pytest.approx(0.8))
        # Monotone taper over positions (per layer, k >= 1 non-increasing;
        # the floor may flatten the tail but never inverts it).
        for j in range(75):
            col = w[1:, j]
            assert np.all(np.diff(col) <= 1e-7)
        # Depth taper: k >= 1 rows non-increasing over layer index
        # (outside the k=1 boost region).
        assert np.all(np.diff(w[2]) <= 1e-7)
        assert np.all(np.diff(w[1, 4:]) <= 1e-7)
        # Survival composition: W[2][0] = 1.0 * 0.43*0.28 * taper[0]=1.
        assert w[2, 0] == pytest.approx(0.43 * 0.28, rel=1e-5)
        # Floor.
        assert np.all(w >= 0.02 - 1e-9)

    def test_uniform_and_explicit(self):
        u = wmap.build_wmap({"kind": "uniform", "value": 0.5}, 3, [3, 4])
        assert np.all(u == 0.5) and u.shape == (3, 2)
        m = [[1.0, 0.5], [0.2, 0.1]]
        e = wmap.build_wmap({"kind": "explicit", "matrix": m}, 2, [3, 4])
        np.testing.assert_allclose(e, m)
        with pytest.raises(ValueError):
            wmap.build_wmap({"kind": "explicit", "matrix": m}, 3, [3, 4])

    def test_spec_serialization_round_trip(self):
        spec = wmap.WMapSpec(kind="directive", accept_curve=(0.5, 0.3),
                             k1_early_value=0.7, floor=0.01)
        spec2 = wmap.WMapSpec.from_dict(
            json.loads(json.dumps(spec.to_dict())))
        assert spec2 == spec
        np.testing.assert_array_equal(spec.build(4, [3, 4, 5]),
                                      spec2.build(4, [3, 4, 5]))

    def test_spec_validation(self):
        with pytest.raises(ValueError):
            wmap.WMapSpec(kind="nope")
        with pytest.raises(ValueError):
            wmap.WMapSpec(floor=1.5)  # >= k0_value
        with pytest.raises(ValueError):
            wmap.survival_from_curve((1.2,), 3)


class TestAllocateRanks:
    def test_budget_respected_and_ordering_follows_w(self):
        w = wmap.build_wmap({"kind": "directive"}, 5, list(range(3, 15)))
        gol = np.arange(12)
        budget = 40_000
        r = wmap.allocate_ranks(w, gol, budget_params=budget, in_dim=32,
                                out_dim=32, max_rank=1000)
        assert int(r.sum()) * 64 <= budget
        # ordering: higher W (== v here, one layer per group) -> >= rank
        v = w
        flat_v = v.ravel()
        flat_r = r.ravel()
        order = np.argsort(-flat_v, kind="stable")
        assert np.all(np.diff(flat_r[order]) <= 0) or np.all(
            flat_r[order][:-1] >= flat_r[order][1:])
        # k=0 row gets the most capacity.
        assert r[0].min() >= r[1:].max()

    def test_group_value_is_summed_over_layers(self):
        w = np.ones((1, 4), np.float32)
        gol = np.array([0, 0, 0, 1])  # group 0 serves 3 layers
        r = wmap.allocate_ranks(w, gol, budget_params=8 * (4 + 4),
                                in_dim=4, out_dim=4, max_rank=100)
        assert r[0, 0] == 6 and r[0, 1] == 2  # 3:1 value split of 8 units

    def test_max_rank_saturation_under_spends(self):
        w = np.ones((1, 2), np.float32)
        r = wmap.allocate_ranks(w, [0, 1], budget_params=1000, in_dim=1,
                                out_dim=1, max_rank=4)
        assert np.all(r == 4)  # capped, budget under-spent

    def test_min_rank_over_budget_raises(self):
        w = np.ones((2, 2), np.float32)
        with pytest.raises(ValueError):
            wmap.allocate_ranks(w, [0, 1], budget_params=10, in_dim=8,
                                out_dim=8, min_rank=4)

    def test_deterministic(self):
        w = np.random.default_rng(0).random((4, 6)).astype(np.float32)
        kw = dict(budget_params=5000, in_dim=16, out_dim=16, max_rank=64)
        r1 = wmap.allocate_ranks(w, np.arange(6), **kw)
        r2 = wmap.allocate_ranks(w, np.arange(6), **kw)
        np.testing.assert_array_equal(r1, r2)

    def test_zero_value_cells_get_zero_rank(self):
        w = np.array([[1.0, 1.0], [0.0, 0.0]], np.float32)
        r = wmap.allocate_ranks(w, [0, 1], budget_params=1000, in_dim=2,
                                out_dim=2, max_rank=1000)
        assert np.all(r[1] == 0) and np.all(r[0] > 0)


# ── model ────────────────────────────────────────────────────────────────────

class TestModelInit:
    def test_router_arm_init_is_tier0(self):
        """LoRA zero-init up + residual => at init the model IS the
        Tier-0 frozen-router probe (B1) exactly."""
        dims = make_dims()
        bank = make_bank(dims)
        m = model_mod.build_model(base_cfg(), dims,
                                  {"kind": "uniform"}, bank, seed=0)
        x = rand_features(dims)
        logits = m(x).detach().numpy()
        taps = glm_router.aux_prior_tap(dims.moe_layers)
        # per-layer: frozen router_j applied to the prior tap's raw hidden
        ref = np.stack([bank.logits(x.numpy()[:, :, taps[j]])[..., j, :]
                        for j in range(dims.n_layers)], axis=2)
        np.testing.assert_allclose(logits, ref, rtol=1e-4, atol=1e-5)

    def test_direct_arm_init_emits_zero_logits(self):
        dims = make_dims()
        cfg = {"arm": "direct", "normalization": {"kind": "none"},
               "taps": {"mode": "fixed", "map": "prior"},
               "direct_head": {"r_base": 8, "logit_bias": True,
                               "delta": {"enabled": False}}}
        m = model_mod.build_model(cfg, dims, {"kind": "uniform"},
                                  make_bank(dims), seed=0)
        out = m(rand_features(dims)).detach()
        assert torch.all(out == 0)

    def test_forward_shapes_and_partial_gamma(self):
        dims = make_dims()
        cfg = base_cfg(delta={"enabled": True, "budget_params": 4000,
                              "max_rank": 8})
        m = model_mod.build_model(cfg, dims, {"kind": "directive"},
                                  make_bank(dims), seed=0)
        out = m(rand_features(dims, b=3))
        assert out.shape == (3, dims.max_gamma, dims.n_layers,
                             dims.n_experts)
        out2 = m(rand_features(dims, b=3, g=2))
        assert out2.shape == (3, 2, dims.n_layers, dims.n_experts)
        with pytest.raises(ValueError):
            m(rand_features(dims, g=dims.max_gamma + 1))

    def test_tap_fusion_init_matches_fixed_prior_tap(self):
        dims = make_dims()
        bank = make_bank(dims)
        fixed = model_mod.build_model(base_cfg(), dims,
                                      {"kind": "uniform"}, bank, seed=0)
        cfg = base_cfg()
        cfg["taps"] = {"mode": "fusion", "rank": 4, "init_gain": 14.0,
                       "map": "prior"}
        fused = model_mod.build_model(cfg, dims, {"kind": "uniform"},
                                      bank, seed=0)
        x = rand_features(dims)
        np.testing.assert_allclose(fused(x).detach(), fixed(x).detach(),
                                   rtol=1e-3, atol=1e-3)

    def test_layer_group_sharing_shapes(self):
        dims = make_dims()
        for groups, n_g in (("per_layer", 6), ("per_segment", 3),
                            ("single", 1),
                            ([[3, 4, 5, 10], [24, 40]], 2)):
            cfg = base_cfg()
            cfg["layer_groups"] = groups
            m = model_mod.build_model(cfg, dims, {"kind": "uniform"},
                                      make_bank(dims), seed=0)
            assert m.base_down.shape[0] == n_g
            assert m(rand_features(dims)).shape == (
                2, dims.max_gamma, dims.n_layers, dims.n_experts)

    def test_delta_ranks_follow_wmap_capacity(self):
        dims = make_dims()
        cfg = base_cfg(delta={"enabled": True, "budget_params": 30000,
                              "max_rank": 1000})
        m = model_mod.build_model(cfg, dims, {"kind": "directive"},
                                  make_bank(dims), seed=0)
        ranks = np.asarray(m.delta_rank_plan())
        assert ranks.shape == (dims.max_gamma, dims.n_layers)
        assert ranks[0].min() >= ranks[1:].max()
        total = ranks.sum() * (dims.hidden + dims.hidden)
        assert total <= 30000

    def test_activation_arm(self):
        dims = make_dims()
        cfg = base_cfg(activation="silu")
        m = model_mod.build_model(cfg, dims, {"kind": "uniform"},
                                  make_bank(dims), seed=0)
        assert m(rand_features(dims)).shape[-1] == dims.n_experts
        with pytest.raises(ValueError):
            model_mod.build_model(base_cfg(activation="nope"), dims,
                                  {"kind": "uniform"}, make_bank(dims),
                                  seed=0)


class TestNormalization:
    def test_rmsnorm_normalizes_rows(self):
        dims = make_dims()
        norm = model_mod.InputNorm({"kind": "rmsnorm", "learnable": False},
                                   dims)
        x = rand_features(dims) * 7.0
        y = norm(x)
        rms = y.pow(2).mean(-1).sqrt()
        np.testing.assert_allclose(rms.numpy(), 1.0, rtol=1e-3)

    def test_standardize_requires_stats_then_applies_them(self):
        dims = make_dims()
        norm = model_mod.InputNorm({"kind": "standardize", "eps": 0.0},
                                   dims)
        x = rand_features(dims)
        with pytest.raises(RuntimeError):
            norm(x)
        mean = np.full((dims.n_taps, dims.hidden), 2.0, np.float32)
        std = np.full((dims.n_taps, dims.hidden), 4.0, np.float32)
        norm.set_standardize_stats(mean, std)
        np.testing.assert_allclose(norm(x).numpy(),
                                   (x.numpy() - 2.0) / 4.0, rtol=1e-5)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            model_mod.InputNorm({"kind": "zscore"}, make_dims())


class TestInvEpmSide:
    def test_router_buffers_frozen_and_never_serialized(self, tmp_path):
        dims = make_dims()
        bank = make_bank(dims)
        cfg = base_cfg(delta={"enabled": True, "budget_params": 4000,
                              "max_rank": 8})
        m = model_mod.build_model(cfg, dims, {"kind": "directive"}, bank,
                                  seed=0)
        assert not m.router_weight.requires_grad
        before = m.router_weight.clone()
        opt = torch.optim.AdamW(m.parameters(), lr=0.1)
        x = rand_features(dims)
        m(x).pow(2).mean().backward()
        opt.step()
        assert torch.equal(m.router_weight, before)
        assert torch.equal(m.router_bias,
                           torch.as_tensor(bank.bias))
        model_mod.save_model(m, tmp_path / "ck")
        saved = model_mod.load_safetensors(tmp_path / "ck.safetensors")
        assert not any("router" in k for k in saved)

    def test_optimizer_step_changes_only_trainable_params(self):
        dims = make_dims()
        m = model_mod.build_model(base_cfg(), dims, {"kind": "uniform"},
                                  make_bank(dims), seed=0)
        bufs = {k: v.clone() for k, v in m.named_buffers()}
        opt = torch.optim.AdamW(m.parameters(), lr=0.1)
        m(rand_features(dims)).pow(2).mean().backward()
        opt.step()
        for k, v in m.named_buffers():
            assert torch.equal(v, bufs[k]), f"buffer {k} changed"


class TestCheckpointRoundTrip:
    def test_save_load_bitwise(self, tmp_path):
        dims = make_dims()
        bank = make_bank(dims)
        cfg = base_cfg(delta={"enabled": True, "budget_params": 6000,
                              "max_rank": 8})
        cfg["normalization"] = {"kind": "rmsnorm", "learnable": True}
        m = model_mod.build_model(cfg, dims, {"kind": "directive"}, bank,
                                  seed=7)
        # perturb away from init so the round trip is non-trivial
        opt = torch.optim.AdamW(m.parameters(), lr=0.05)
        for _ in range(3):
            opt.zero_grad()
            m(rand_features(dims)).pow(2).mean().backward()
            opt.step()
        model_mod.save_model(m, tmp_path / "ck",
                             sidecar_extra={"note": "test"})
        m2, sidecar = model_mod.load_model(tmp_path / "ck", bank)
        for (k1, v1), (k2, v2) in zip(m.state_dict().items(),
                                      m2.state_dict().items()):
            assert k1 == k2 and torch.equal(v1, v2), k1
        x = rand_features(dims, b=3, seed=5)
        ids1, sc1 = m.predict(x)
        ids2, sc2 = m2.predict(x)
        np.testing.assert_array_equal(ids1, ids2)
        np.testing.assert_array_equal(sc1, sc2)
        assert sidecar["note"] == "test"
        assert sidecar["arm"] == "router"
        assert np.asarray(sidecar["wmap_matrix"]).shape == (
            dims.max_gamma, dims.n_layers)

    def test_safetensors_io_round_trip_dtypes(self, tmp_path):
        tensors = {
            "a": torch.randn(3, 4),
            "b": torch.randn(5).to(torch.bfloat16),
            "c": torch.arange(6, dtype=torch.int64).reshape(2, 3),
            "d": torch.tensor([1.5, -2.0], dtype=torch.float16),
        }
        model_mod.save_safetensors(tensors, tmp_path / "t.safetensors")
        back = model_mod.load_safetensors(tmp_path / "t.safetensors")
        assert set(back) == set(tensors)
        for k in tensors:
            assert back[k].dtype == tensors[k].dtype
            assert torch.equal(back[k], tensors[k])

    def test_predict_matches_noaux_tc_semantics(self):
        """predict() ranks with sigmoid+bias via glm_router — verify the
        biased selection against a direct noaux_tc call on the model's
        own logits."""
        dims = make_dims()
        bank = make_bank(dims)
        m = model_mod.build_model(base_cfg(), dims, {"kind": "uniform"},
                                  bank, seed=0)
        x = rand_features(dims)
        ids, _ = m.predict(x, m_rank=8)
        logits = m(x).detach().numpy()
        ref_ids, _ = glm_router.noaux_tc_topk(logits, bank.bias, topk=8)
        np.testing.assert_array_equal(ids, ref_ids)


class TestConfigResolution:
    def test_tap_map_variants(self):
        dims = make_dims()
        prior = model_mod.resolve_tap_map({"map": "prior"}, dims)
        np.testing.assert_array_equal(
            prior, glm_router.aux_prior_tap(dims.moe_layers))
        np.testing.assert_array_equal(
            model_mod.resolve_tap_map({"map": 2}, dims),
            np.full(dims.n_layers, 2))
        explicit = [0, 1, 2, 3, 4, 0]
        np.testing.assert_array_equal(
            model_mod.resolve_tap_map({"map": explicit}, dims), explicit)
        with pytest.raises(ValueError):
            model_mod.resolve_tap_map({"map": [9] * dims.n_layers}, dims)

    def test_explicit_layer_groups_validation(self):
        dims = make_dims()
        with pytest.raises(ValueError):  # unassigned layer
            model_mod.resolve_layer_groups([[3, 4]], dims,
                                           np.zeros(6, np.int32))
        with pytest.raises(ValueError):  # duplicate
            model_mod.resolve_layer_groups(
                [[3, 4, 5, 10, 24, 40], [3]], dims, np.zeros(6, np.int32))


class TestPriorHybridArm:
    """b0_prev-hybrid: beta_j * prev_membership added to trained logits."""

    def _cfg(self, per="layer", init_beta=3.0):
        return {"arm": "direct", "normalization": {"kind": "none"},
                "taps": {"mode": "fixed", "map": "prior"},
                "layer_groups": "per_layer",
                "direct_head": {"r_base": 4, "logit_bias": True,
                                "use_router_bias_for_selection": False,
                                "delta": {"enabled": False}},
                "prior": {"enabled": True, "init_beta": init_beta,
                          "per": per}}

    def test_prior_is_noop_without_membership(self):
        dims = make_dims()
        m = model_mod.build_model(self._cfg(), dims, {"kind": "uniform"},
                                  None, seed=0)
        x = rand_features(dims)
        assert torch.allclose(m(x), m(x, None))

    def test_prior_adds_beta_per_layer(self):
        dims = make_dims()
        m = model_mod.build_model(self._cfg(per="layer"), dims,
                                  {"kind": "uniform"}, None, seed=0)
        with torch.no_grad():
            m.prior_beta.copy_(torch.arange(dims.n_layers,
                                            dtype=torch.float32))
        x = rand_features(dims)
        mem = torch.zeros(x.shape[0], x.shape[1], dims.n_layers,
                          dims.n_experts)
        mem[..., 0] = 1.0                                 # expert 0 = member
        base, prior = m(x), m(x, mem)
        # expert 0 got +beta_j; others unchanged.
        for j in range(dims.n_layers):
            assert torch.allclose(prior[..., j, 0],
                                  base[..., j, 0] + float(j))
        assert torch.allclose(prior[..., 1:], base[..., 1:])

    def test_scalar_prior(self):
        dims = make_dims()
        m = model_mod.build_model(self._cfg(per="scalar", init_beta=2.0),
                                  dims, {"kind": "uniform"}, None, seed=0)
        x = rand_features(dims)
        mem = torch.zeros(x.shape[0], x.shape[1], dims.n_layers,
                          dims.n_experts)
        mem[..., 5] = 1.0
        d = m(x, mem)[..., 5] - m(x)[..., 5]
        assert torch.allclose(d, torch.full_like(d, 2.0))

    def test_prior_beta_is_trainable_and_saves(self, tmp_path):
        dims = make_dims()
        m = model_mod.build_model(self._cfg(), dims, {"kind": "uniform"},
                                  None, seed=0)
        assert m.prior_beta.requires_grad
        # a gradient flows into beta through a membership term
        x = rand_features(dims)
        mem = torch.zeros(x.shape[0], x.shape[1], dims.n_layers,
                          dims.n_experts)
        mem[..., 2] = 1.0
        m(x, mem).sum().backward()
        assert m.prior_beta.grad is not None
        assert torch.any(m.prior_beta.grad != 0)
        with torch.no_grad():
            m.prior_beta.copy_(torch.linspace(0.5, 4.0, dims.n_layers))
        model_mod.save_model(m, tmp_path / "model")
        m2, _ = model_mod.load_model(tmp_path / "model", None)
        assert m2.prior_enabled
        assert torch.allclose(m2.prior_beta, m.prior_beta)

    def test_disabled_prior_has_no_beta(self):
        dims = make_dims()
        cfg = self._cfg()
        cfg["prior"] = {"enabled": False}
        m = model_mod.build_model(cfg, dims, {"kind": "uniform"}, None,
                                  seed=0)
        assert m.prior_beta is None
        x = rand_features(dims)
        mem = torch.ones(x.shape[0], x.shape[1], dims.n_layers,
                         dims.n_experts)
        assert torch.allclose(m(x), m(x, mem))            # ignored

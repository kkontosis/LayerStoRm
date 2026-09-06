"""Autoconfig solver tests (TD-AUTOCONFIG-HARDWARE-FIT).

The acceptance test (AUTOCONFIG §9): on this box's replica descriptor, with
the champion-implied levers, the solver must reproduce
recipes/glm52_serve_champion.json with every divergence classified and none
unexplained. Fully hermetic: fake /proc+/sys, synthetic calibration and
draft checkpoint, byte constants measured once from the real artifacts.
"""

import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from test_autoconfig_detect import BOX_GPUS, build_fake_tree, fake_calibration

from autoconfig import engine_constraints as ec, sizing
from autoconfig.calibration import load_calibration, accept_calibration
from autoconfig.compare import compare
from autoconfig.explain import Infeasible
from autoconfig.hwdetect import detect_hardware
from autoconfig.modelshape import ModelShape
from autoconfig.solver import (PREFERENCES, AutoconfigKnobs, Attempt,
                               DraftCandidate, Levers, Solver, _order_key,
                               load_draft_candidate)

REPO = os.path.join(os.path.dirname(__file__), "..", "..")

# Measured once from the real artifacts (2026-08-30); the solver receives
# them through the same interfaces the CLI resolves them by.
SLOT_BYTES = 27623424        # GLM-5.2 prepack manifest slot_size_bytes
PINNED_BYTES = int(19.59 * (1 << 30))  # GGUF non-expert tensor bytes
CHAMPION_LEVERS = Levers(vram_expert_ratio=0.0627,
                         total_active_context_tokens=51200)

# The classified-and-allowed champion divergences (AUTOCONFIG §9): the
# solver metadata section, the hand-rounded RAM total, and the measured
# HBM fraction_free correction. Anything else failing this set is a
# solver regression.
ALLOWED_DIVERGENCE_PREFIXES = (
    "autoconfig",
    "hardware.system_ram_gb",
    "memory.cross_node_spill.per_node[",
    # the champion omits the field and inherits the schema default; a derived
    # recipe states it (house style: readable stand-alone) and explains it
    "gpu_loader.calibration_mode",
    # (removed 2026-08-31: "hardware.dcp_indexer_mode".  The divergence that
    #  entry explained is GONE, not absorbed — the champion A/B's PRICE for
    #  `local` (~8-9% served prefill for 43.5 MiB/rank) landed as registry row
    #  `local-indexer-prefill-cost`, so the solver now prefers `replicated`
    #  while those bytes are affordable and derives exactly what the champion
    #  recipe says.  `local` is still reachable — as a CAPACITY escalation the
    #  E1 fit takes on its own, with the price named (TestIndexerModeCost).)
)


def write_fake_draft(tmp_path):
    d = tmp_path / "draft.dspark"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "block_size": 16,
        "speculators_config": {"proposal_methods": [{"speculative_tokens": 15}]},
        "transformer_layer_config": {
            "num_hidden_layers": 5, "num_key_value_heads": 64, "head_dim": 64,
            "hidden_size": 6144, "vocab_size": 154880},
    }))
    header = {}
    off = 0
    def add(name, shape, dtype="BF16"):
        nonlocal off
        n = 1
        for s in shape:
            n *= s
        nbytes = n * 2
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    add("model.embed_tokens.weight", (154880, 6144))
    add("lm_head.weight", (154880, 6144))
    for l in range(5):
        add(f"model.layers.{l}.self_attn.qkv_proj.weight", (12288, 6144))
        add(f"model.layers.{l}.mlp.gate_up_proj.weight", (24576, 6144))
        add(f"model.layers.{l}.mlp.down_proj.weight", (6144, 12288))
        add(f"model.layers.{l}.input_layernorm.weight", (6144,))
    hb = json.dumps(header).encode()
    with open(d / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
    return str(d)


@pytest.fixture()
def champion():
    with open(os.path.join(REPO, "recipes", "glm52_serve_champion.json")) as f:
        return json.load(f)


def make_solver(tmp_path, champion, levers=CHAMPION_LEVERS, bar1_5090=32,
                with_cal=True, with_draft=True, knobs=AutoconfigKnobs(),
                shape=None):
    gpus = [(b, m, u, (bar1_5090 if "5090" in m else 16), n)
            for b, m, u, _, n in BOX_GPUS]
    proc, sysd = build_fake_tree(str(tmp_path / "tree"), gpus=gpus)
    hw = detect_hardware(proc_root=proc, sys_root=sysd)
    shape = shape or ModelShape.from_model_section(champion["model"])
    cal = rej = None
    if with_cal:
        cal = load_calibration(fake_calibration(tmp_path))
        rej = accept_calibration(cal, hw, [2, 3, 0, 1],
                                 shape.hidden_size, shape.moe_intermediate_size)
        assert rej == []
    draft = None
    if with_draft:
        dpath = write_fake_draft(tmp_path)
        draft = load_draft_candidate(os.path.basename(dpath), str(tmp_path))
        draft = DraftCandidate(**{**draft.__dict__,
                                  "checkpoint_path": champion["speculation"]["dspark"]["checkpoint_path"]})
    return Solver(hw, shape, champion, levers, knobs, cal=cal,
                  cal_reject_reasons=tuple(rej or ()),
                  expert_slot_bytes=SLOT_BYTES, slot_provenance="prepack manifest",
                  non_expert_pinned_bytes=PINNED_BYTES,
                  pinned_provenance="gguf headers", draft=draft)


class TestChampionAcceptance:
    def test_reproduces_champion_shape(self, tmp_path, champion):
        res = make_solver(tmp_path, champion).solve()
        r = res.recipe
        # spot-assert the load-bearing derivations so a failure names the knob
        assert [g["id"] for g in r["hardware"]["gpus"]] == [2, 3, 0, 1]
        assert [g["vram_gb"] for g in r["hardware"]["gpus"]] == [30, 30, 15, 15]
        assert [g["vram_allocation_gb"]["expert_streaming"]
                for g in r["hardware"]["gpus"]] == [4.0, 4.0, 11.5, 11.5]
        assert r["hardware"]["tp_array"] == [0, 1]
        assert r["hardware"]["dcp_kv_mode"] == "sharded"
        # The engine gate forcing `replicated` fell (TD-KVT-LOCAL-INDEXER-
        # UNBLOCK, 2026-08-30) but the champion A/B priced the flip: `local`
        # costs ~8-9% of served prefill for 43.5 MiB/rank of indexer-K.  That
        # price is registry row `local-indexer-prefill-cost`, so the solver
        # buys prefill with VRAM the box is not using and lands back ON the
        # champion recipe — `local` stays reachable as a capacity escalation.
        assert r["hardware"]["dcp_indexer_mode"] == "replicated"
        assert r["parallelism"]["tensor_parallelism"] == 2
        assert r["memory"]["kv_cache"]["max_pages_per_gpu"] == 20480
        assert r["memory"]["kv_cache"]["page_growth_chunk_tokens"] == 16
        assert r["memory"]["kv_tiering"] == {"enabled": True,
                                             "hot_buffer_slots": 2048,
                                             "host_to_device_ratio": 8.0,
                                             "tiered_prefill": True}
        assert r["memory"]["pinned_layers"]["dense_ffn_layers"] == [0, 1, 2]
        assert r["memory"]["vram_safety_margin_gb"] == 2.25
        assert r["serving"]["max_concurrent_requests"] == 2
        assert r["serving"]["max_sequence_length"] == 25600
        assert r["serving"]["prefix_cache"]["max_cached_tokens"] == 65536
        assert r["_internal-prefix_cache"]["max_entry_tokens"] == 25600
        assert r["speculation"]["method"] == "dspark"
        assert r["speculation"]["dspark"]["draft_gpus"] == [0, 1]
        assert r["speculation"]["dspark"]["draft_weights_quant"] == "nvfp4"
        assert r["speculation"]["dspark"]["block_size"] == 16
        assert r["speculation"]["dspark"]["speculative_tokens"] == 15
        assert r["compute"]["attention_backend"] == "turboquant_mla"
        assert res.degradations == []

    def test_every_divergence_classified(self, tmp_path, champion):
        res = make_solver(tmp_path, champion).solve()
        divs = compare(champion, res.recipe, res.explanations)
        unexplained = [d for d in divs if d.klass == "unexplained"]
        assert unexplained == [], [f"{d.path}: {d.reference}->{d.derived}"
                                   for d in unexplained]
        for d in divs:
            assert d.path.startswith(ALLOWED_DIVERGENCE_PREFIXES), \
                f"new divergence vs champion: {d.path} ({d.reference} -> {d.derived})"

    def test_expert_slot_arithmetic(self, tmp_path, champion):
        res = make_solver(tmp_path, champion).solve()
        by_ord = {p.ordinal: p for p in res.gpu_plans}
        assert by_ord[2].expert_slots == 155  # 4.0 GiB / 27623424
        assert by_ord[0].expert_slots == 447  # 11.5 GiB / 27623424
        total = sum(p.expert_slots for p in res.gpu_plans)
        assert abs(total / (256 * 75) - 0.0627) < 0.002


class TestDegradationAndRefusal:
    def test_draft_dropped_before_capacity(self, tmp_path, champion):
        # a 29-GiB-BAR "5090" cannot host pinned+draft+experts+KV; the ladder
        # must shed the draft FIRST and keep the user's 2 x 25600 ask intact
        res = make_solver(tmp_path, champion, bar1_5090=29).solve()
        assert any("draft dropped" in d for d in res.degradations)
        assert res.recipe["speculation"]["method"] == "none"
        assert res.recipe["serving"]["max_concurrent_requests"] == 2
        assert res.recipe["serving"]["max_sequence_length"] == 25600

    def test_ratio_exceeds_vram_refused_with_achievable(self, tmp_path, champion):
        with pytest.raises(Infeasible) as ei:
            make_solver(tmp_path, champion,
                        levers=Levers(0.5, 51200)).solve()
        e = ei.value
        assert e.constraint_id in ("expert-ratio-exceeds-vram", "vram-carve-overflow")
        assert "vram_expert_ratio" in e.suggestion

    def test_tiny_box_refuses_naming_binding_gpu(self, tmp_path, champion):
        gpus = [BOX_GPUS[2], BOX_GPUS[3]]  # only the two 5090s...
        gpus = [(b, m, u, 16, n) for b, m, u, _, n in gpus]  # ...at 16 GiB
        proc, sysd = build_fake_tree(str(tmp_path / "tiny"), gpus=gpus)
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        shape = ModelShape.from_model_section(champion["model"])
        sv = Solver(hw, shape, champion, Levers(0.01, 1048576 + 102400),
                    cal=None, expert_slot_bytes=SLOT_BYTES,
                    non_expert_pinned_bytes=PINNED_BYTES)
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert "gpu" in ei.value.binding

    def test_no_calibration_is_provisional_and_self_healing(self, tmp_path, champion):
        res = make_solver(tmp_path, champion, with_cal=False).solve()
        assert any("PROVISIONAL" in w for w in res.warnings)
        assert res.recipe["gpu_loader"]["calibration_mode"] == "full"


class TestConstraintsAsData:
    def test_default_derivation_weighs_the_price_with_tiering(self, tmp_path,
                                                              champion):
        """TD-KVT-LOCAL-INDEXER-UNBLOCK resolved (2026-08-30): the engine's
        replicated-only tiering construction gate was a merge artifact and is
        gone; the registry row was DELETED (the one-edit change the row
        existed for).  Tiering is therefore ON under EITHER indexer mode
        (INV-KVT-20), and the mode is decided by the measured price row
        `local-indexer-prefill-cost` instead — on this box `replicated`."""
        res = make_solver(tmp_path, champion).solve()
        assert res.recipe["hardware"]["dcp_indexer_mode"] == "replicated"
        assert res.recipe["memory"]["kv_tiering"]["enabled"] is True
        e = res.explanations.get("hardware.dcp_indexer_mode")
        assert "TD-KVT-LOCAL-INDEXER-UNBLOCK" in (e.refs or ())
        assert "local-indexer-prefill-cost" in (e.refs or ())

    def test_injected_gate_row_forces_replicated(self, tmp_path, champion,
                                                 monkeypatch):
        """Constraint-as-data, exercised in the other direction: should a
        tiering x indexer-mode gate ever RETURN to the engine, adding back a
        registry row with this id must flip the derivation to replicated —
        data alone, no solver-logic change (the lookup survives in
        Solver.dcp_modes for exactly this)."""
        row = ec.ConstraintRow(
            id="local-indexer-disables-tiering",
            kind="engine_gate",
            statement="synthetic re-introduction for the data-flip test",
            ticket="TD-SYNTHETIC",
            site="tests/unit/test_autoconfig_solver.py",
        )
        monkeypatch.setattr(ec, "ENGINE_CONSTRAINTS",
                            ec.ENGINE_CONSTRAINTS + (row,))
        # exercised on the CONTESTED box, where the derivation without the
        # gate is `local` (TestIndexerModeCost) — so this pins the gate, not
        # the price row's default.  With `local` off the menu the fit has to
        # pay for the mode in capacity instead: the draft goes.
        sv = _contested_solver(tmp_path, champion)
        res = sv.solve()
        assert res.recipe["hardware"]["dcp_indexer_mode"] == "replicated"
        assert res.recipe["memory"]["kv_tiering"]["enabled"] is True
        e = res.explanations.get("hardware.dcp_indexer_mode")
        assert "local-indexer-disables-tiering" in (e.refs or ())
        assert res.degradations, "the gate must cost capacity, not be free"


def _contested_solver(tmp_path, champion, ctx=409600):
    """The champion box asked for 8x the champion's active context: the
    indexer-K share (525 slabs / 544 MiB per max-length sequence at
    replicated) no longer fits beside experts+KV, so the indexer-K bytes are
    GENUINELY contested — the case registry row `local-indexer-prefill-cost`
    exists to decide."""
    base = dict(champion)
    base["serving"] = {k: v for k, v in champion["serving"].items()
                       if k != "max_sequence_length"}
    sv = make_solver(tmp_path, base, levers=Levers(0.0627, ctx))
    sv.base = base
    return sv


class TestIndexerModeCost:
    """Registry row `local-indexer-prefill-cost` (champion A/B 2026-08-30):
    `local` halves the per-rank indexer-K share and costs ~8-9% of served
    prefill.  The mode is therefore a CAPACITY decision with a known price —
    the same "capacity, not speed" shape as the glm5next TP row."""

    def test_champion_prefers_replicated_and_names_the_price(self, tmp_path,
                                                             champion):
        res = make_solver(tmp_path, champion).solve()
        assert res.recipe["hardware"]["dcp_indexer_mode"] == "replicated"
        e = res.explanations.get("hardware.dcp_indexer_mode")
        assert e.kind == "measured"
        # the WHY: the payoff it declined, in the boot log's own unit, and
        # the price it bought with VRAM it is not using
        assert "84 slabs / 87.1 MiB -> 42 slabs / 43.5 MiB" in e.because
        assert "8-9% of served prefill" in e.because
        assert "AFFORDABLE" in e.because
        assert "cross-rank merge" in e.because
        assert "local-indexer-prefill-cost" in e.refs
        assert "kvtlocal_ab" in e.measurement

    def test_solver_sizing_reproduces_the_measured_payoff(self, tmp_path,
                                                          champion):
        """The row's VRAM payoff is not a constant pasted into a string: the
        solver's own sizing model derives it (4 indexer pages x 21 DSA
        computing layers = 84 slabs replicated, 42 under the dcp=2
        ceil-divide) and agrees with the boot log to the MiB."""
        sv = make_solver(tmp_path, champion)
        sv.solve()
        r_slabs, r_bytes = sv._indexer_share("replicated", 2, 25600)
        l_slabs, l_bytes = sv._indexer_share("local", 2, 25600)
        assert (r_slabs, l_slabs) == (84, 42)
        assert abs(r_bytes / (1 << 20) - 87.1) < 0.1
        assert abs(l_bytes / (1 << 20) - 43.5) < 0.1

    def test_escalates_to_local_for_capacity_not_speed(self, tmp_path,
                                                       champion):
        res = _contested_solver(tmp_path, champion).solve()
        assert res.recipe["hardware"]["dcp_indexer_mode"] == "local"
        e = res.explanations.get("hardware.dcp_indexer_mode")
        assert e.kind == "searched"
        assert "CAPACITY, NOT SPEED" in e.because
        # it must name the price it just paid, and what forced it
        assert "PRICE is ~8-9% of served prefill" in e.because
        assert "infeasible" in e.because
        assert "MiB/rank reclaimed" in e.because
        # and the mode is spent BEFORE the user's ask or the draft: the
        # ladder keeps both at the escalation rung
        assert res.degradations == []
        assert res.recipe["speculation"]["method"] == "dspark"
        assert res.recipe["memory"]["kv_tiering"]["enabled"] is True

    def test_deleting_the_row_moves_the_derivation(self, tmp_path, champion,
                                                   monkeypatch):
        """The delete-the-row property (AUTOCONFIG §4): re-measure and find
        the cost gone, delete this ONE row, and the solver goes back to
        taking `local` wherever nothing forbids it — no solver edit."""
        monkeypatch.setattr(ec, "ENGINE_CONSTRAINTS",
                            tuple(r for r in ec.ENGINE_CONSTRAINTS
                                  if r.id != "local-indexer-prefill-cost"))
        res = make_solver(tmp_path, champion).solve()
        assert res.recipe["hardware"]["dcp_indexer_mode"] == "local"
        assert res.recipe["memory"]["kv_tiering"]["enabled"] is True
        e = res.explanations.get("hardware.dcp_indexer_mode")
        assert e.kind == "heuristic"          # the pre-row VRAM rationale
        assert "local-indexer-prefill-cost" not in (e.refs or ())

    def test_the_row_states_its_evidence_bound(self):
        row = ec.get("local-indexer-prefill-cost")
        assert row.kind == "measured" and row.scope == ""
        # the four ladder legs, with their arms
        for n in ("45.15", "41.56", "43.88", "39.95", "43.85", "40.04",
                  "42.73", "38.82"):
            assert n in row.statement
        # payoff, mechanism, preconditions, provenance
        assert "84 slabs (87.1 MiB) -> 42 slabs (43.5 MiB)" in row.statement
        assert "B=64 rows" in row.statement
        assert "decode in-noise" in row.statement
        assert "deterministic_ep_combine ON" in row.statement
        assert "DET-TOPK-TIES" in row.statement and "degraded=0" in row.statement
        assert "kvtlocal_ab_replicated.json" in row.site
        assert "kvtlocal_ab_local.json" in row.site
        # the bound itself, stated the way glm5next-tp1-default states its
        # own: ONE box, ONE model, ONE ladder — and what would falsify it
        assert "EVIDENCE BOUND" in row.statement
        assert "ONE box" in row.statement and "ONE model" in row.statement
        assert "does NOT establish" in row.statement
        assert "FALSIFIED BY" in row.statement


class TestLeverSemantics:
    def test_active_context_split_default_pairing(self, tmp_path, champion):
        base = dict(champion)
        base["serving"] = {k: v for k, v in champion["serving"].items()
                           if k != "max_sequence_length"}
        sv = make_solver(tmp_path, base)
        sv.base = base
        res = sv.solve()
        assert res.recipe["serving"]["max_concurrent_requests"] == 2
        assert res.recipe["serving"]["max_sequence_length"] == 25600

    def test_base_max_seq_carried(self, tmp_path, champion):
        res = make_solver(tmp_path, champion,
                          levers=Levers(0.0627, 76800)).solve()
        # base recipe pins 25600; lever 76800 -> 3 concurrent
        assert res.recipe["serving"]["max_sequence_length"] == 25600
        assert res.recipe["serving"]["max_concurrent_requests"] == 3

    def test_ratio_none_residual_policy(self, tmp_path, champion):
        res = make_solver(tmp_path, champion,
                          levers=Levers(None, 51200)).solve()
        e = res.explanations.get("autoconfig.vram_expert_ratio")
        assert e is not None and e.value > 0.05  # hosts filled + TP residual

def _prefer(sv, prefer):
    """Same solver, one lever changed (Levers is frozen)."""
    sv.levers = Levers(sv.levers.vram_expert_ratio,
                       sv.levers.total_active_context_tokens, prefer)
    return sv


def _derivation(sv):
    """Everything a reader of the artifacts would see."""
    res = sv.solve()
    return (res.recipe, res.explanations.render_markdown(),
            list(res.warnings), list(res.degradations))


class TestPreferenceKnob:
    """AUTOCONFIG §2.3 — `prefer` orders the E1 search and does nothing else.

    The lever exists because the shipped ladder encodes value judgements
    nobody chose: WHICH priced slowdown is spent before the user's ask is a
    policy, not a fact.  It is deliberately NOT a weight — tok/s and MiB have
    no exchange rate (TD-AUTOCONFIG-SPEED-BUDGET), so the knob may only
    reorder."""

    # ---- the regression guard: `balanced` is the shipped derivation --------

    def test_balanced_is_the_default_and_a_no_op(self, tmp_path, champion):
        """Naming the default must change nothing, field for field: recipe,
        explanation table, warnings and degradations."""
        assert Levers().prefer == "balanced"
        mks = (lambda t: make_solver(t, champion),
               lambda t: make_solver(t, champion, bar1_5090=29),
               lambda t: _contested_solver(t, champion))
        for i, mk in enumerate(mks):
            a = _derivation(mk(tmp_path / f"a{i}"))
            b = _derivation(_prefer(mk(tmp_path / f"b{i}"), "balanced"))
            assert a == b

    def test_balanced_keeps_the_champion_acceptance(self, tmp_path, champion):
        """The §9 acceptance under the named default: same 6 divergences,
        none unexplained, and NO preference row in the sidecar."""
        res = _prefer(make_solver(tmp_path, champion), "balanced").solve()
        divs = compare(champion, res.recipe, res.explanations)
        assert [d.path for d in divs if d.klass == "unexplained"] == []
        assert all(d.path.startswith(ALLOWED_DIVERGENCE_PREFIXES) for d in divs)
        assert res.recipe["hardware"]["dcp_indexer_mode"] == "replicated"
        assert res.recipe["speculation"]["method"] == "dspark"
        assert res.degradations == []
        assert res.explanations.get("autoconfig.prefer") is None

    # ---- the property that makes the knob safe ----------------------------

    def test_every_preference_walks_the_same_attempt_set(self, tmp_path,
                                                         champion):
        """The never-refuse property, structurally: a preference is a SORT
        KEY over one lattice, so the set of candidate fits it may accept is
        preference-independent by construction.  If this ever stops holding,
        a preference could refuse where another fits."""
        sv = _contested_solver(tmp_path, champion)
        ordered = sv.order_gpus()
        tps = sv.tp_plan(ordered)
        with sv._quiet():
            setups = {tp: sv._tp_setup(ordered, tp) for tp in tps}
        conc0, max_seq0 = sv.active_split()
        lattice = sv._lattice(tps, setups, conc0, max_seq0)
        assert lattice and len(set(lattice)) == len(lattice)
        orders = {p: [tuple(sorted(lattice, key=lambda a: _order_key(p, a)))]
                  for p in PREFERENCES}
        # same points, different order
        assert len({frozenset(v[0]) for v in orders.values()}) == 1
        assert orders["speed"][0] != orders["balanced"][0]
        assert all(len(v[0]) == len(lattice) for v in orders.values())

    def test_no_preference_causes_a_refusal_another_avoids(self, tmp_path,
                                                           champion):
        """The same property end to end, over boxes that span comfortable,
        contested and impossible.  Feasibility must be a property of the BOX,
        never of the preference."""
        cases = [
            ("comfortable", lambda t, p: _prefer(make_solver(t, champion), p)),
            ("tight-bar", lambda t, p: _prefer(
                make_solver(t, champion, bar1_5090=29), p)),
            ("contested", lambda t, p: _prefer(
                _contested_solver(t, champion), p)),
            ("very-contested", lambda t, p: _prefer(
                _contested_solver(t, champion, ctx=819200), p)),
            ("impossible-ratio", lambda t, p: _prefer(
                make_solver(t, champion, levers=Levers(0.5, 51200)), p)),
        ]
        seen_fit = seen_refusal = False
        for name, mk in cases:
            fits = {}
            for i, p in enumerate(PREFERENCES):
                try:
                    mk(tmp_path / f"{name}{i}", p).solve()
                    fits[p] = True
                except Infeasible:
                    fits[p] = False
            assert len(set(fits.values())) == 1, (name, fits)
            seen_fit |= fits["speed"]
            seen_refusal |= not fits["speed"]
        assert seen_fit and seen_refusal, "the sweep must exercise both sides"

    def test_speed_still_takes_the_priced_option_when_it_is_the_only_path(
            self, tmp_path):
        """`speed` refuses to BUY capacity, not to SERVE.  On a glm5_next box
        where TP=1 does not fit at any rung, the TP escalation (priced at
        ~0.5-2% of the decode wall) is the only feasible path and every
        preference must take it."""
        if not os.path.exists(GLM5NEXT_CONFIG):
            pytest.skip("glm5_next reference config absent")
        with open(GLM5NEXT_CONFIG) as f:
            shape = ModelShape.from_model_section(json.load(f)["model"])
        for i, p in enumerate(PREFERENCES):
            proc, sysd = build_fake_tree(str(tmp_path / f"only{i}"))
            hw = detect_hardware(proc_root=proc, sys_root=sysd)
            sv = Solver(hw, shape, {"model": shape.raw}, Levers(0.02, 51200, p),
                        expert_slot_bytes=SLOT_BYTES,
                        non_expert_pinned_bytes=26 * (1 << 30))
            res = sv.solve()
            assert res.recipe["parallelism"]["tensor_parallelism"] == 2, p
            e = sv.ex.get("parallelism.tensor_parallelism")
            assert "CAPACITY, NOT SPEED" in e.because

    # ---- what each value actually does ------------------------------------

    def test_speed_sheds_capacity_before_buying_the_indexer_price(
            self, tmp_path, champion):
        """The contested box: `balanced` keeps the whole 2 x 204800 ask and
        pays ~8-9% of served prefill for `local`; `speed` refuses to buy a
        measured slowdown while a capacity lever is still unspent."""
        bal = _contested_solver(tmp_path / "b", champion).solve()
        assert bal.recipe["hardware"]["dcp_indexer_mode"] == "local"
        assert bal.recipe["serving"]["max_concurrent_requests"] == 2
        assert bal.degradations == []

        sv = _prefer(_contested_solver(tmp_path / "s", champion), "speed")
        fast = sv.solve()
        assert fast.recipe["hardware"]["dcp_indexer_mode"] == "replicated"
        assert fast.degradations, "speed must pay in capacity instead"
        # the ask is smaller, the draft survives, and no price was paid
        assert (fast.recipe["serving"]["max_concurrent_requests"]
                * fast.recipe["serving"]["max_sequence_length"]
                < bal.recipe["serving"]["max_concurrent_requests"]
                * bal.recipe["serving"]["max_sequence_length"])
        assert fast.recipe["speculation"]["method"] == "dspark"
        e = fast.explanations.get("autoconfig.prefer")
        assert e is not None and e.value == "speed" and e.kind == "searched"
        assert "spends every free capacity lever before it buys a priced" in e.because
        assert "~8-9% of served prefill" in e.because
        assert "local-indexer-prefill-cost" in e.refs

    def test_capacity_spends_the_priced_option_to_protect_the_ask(self, tmp_path):
        """The mirror image, on the arch that carries a second priced axis:
        `balanced` sheds concurrency at TP=1 rather than pay for TP; with
        prefer=capacity the box buys TP=2 (~0.5-2% of the decode wall) and
        keeps the user's 2 x 204800."""
        if not os.path.exists(GLM5NEXT_CONFIG):
            pytest.skip("glm5_next reference config absent")
        with open(GLM5NEXT_CONFIG) as f:
            shape = ModelShape.from_model_section(json.load(f)["model"])
        out = {}
        for i, p in enumerate(PREFERENCES):
            proc, sysd = build_fake_tree(str(tmp_path / f"cap{i}"))
            hw = detect_hardware(proc_root=proc, sys_root=sysd)
            sv = Solver(hw, shape, {"model": shape.raw},
                        Levers(0.02, 409600, p), expert_slot_bytes=SLOT_BYTES,
                        non_expert_pinned_bytes=25 * (1 << 30))
            out[p] = sv.solve()
        for p in ("balanced", "speed"):
            assert out[p].recipe["parallelism"]["tensor_parallelism"] == 1
            assert out[p].recipe["serving"]["max_concurrent_requests"] == 1
            # 204800 (the full per-request ask): turboquant_mla's 258 B
            # NoPE rows (vs snapmla's 516 B, 2.0x) buy the whole context —
            # under the deleted `glm5next-tq-backend-unwired` snapmla era
            # this was 102400.  The BEHAVIOUR under test is the ordering,
            # not the number — hence the assertion below reads it back.
            assert out[p].recipe["serving"]["max_sequence_length"] == 204800
            assert out[p].degradations
        cap = out["capacity"]
        assert cap.recipe["parallelism"]["tensor_parallelism"] == 2
        assert cap.recipe["serving"]["max_concurrent_requests"] == 2
        assert cap.recipe["serving"]["max_sequence_length"] == 204800
        assert cap.degradations == []
        e = cap.explanations.get("autoconfig.prefer")
        assert e is not None and e.value == "capacity"
        assert "spends every priced slowdown before it touches" in e.because
        assert "escalated to TP=2 early because prefer=capacity" in e.because
        assert "~0.5-2% of the decode wall" in e.because
        assert "glm5next-tp1-default" in e.refs
        # and it says what balanced would have served instead
        bal = out["balanced"].recipe["serving"]["max_sequence_length"]
        assert f"where `balanced` serves 1 x {bal}" in e.because

    # ---- the explanation contract -----------------------------------------

    def test_the_explanation_is_silent_when_the_preference_changed_nothing(
            self, tmp_path, champion):
        """A knob that narrates itself for doing nothing is noise.  On a box
        with slack every preference derives the same recipe, so the sidecar
        must not mention the lever at all."""
        base = _derivation(make_solver(tmp_path / "d", champion))
        for i, p in enumerate(PREFERENCES):
            sv = _prefer(make_solver(tmp_path / f"q{i}", champion), p)
            res = sv.solve()
            assert res.explanations.get("autoconfig.prefer") is None, p
            assert res.recipe == base[0], p
            assert "autoconfig.prefer" not in res.explanations.render_markdown()

    def test_an_unknown_preference_is_refused_at_construction(self, tmp_path,
                                                              champion):
        with pytest.raises(ValueError) as ei:
            make_solver(tmp_path, champion,
                        levers=Levers(0.0627, 51200, "fastest"))
        assert "prefer='fastest'" in str(ei.value)
        assert "speed/balanced/capacity" in str(ei.value)

    # ---- constraints stay DATA under every preference ---------------------

    def test_deleting_the_price_row_moves_every_preference(self, tmp_path,
                                                           champion, monkeypatch):
        """The delete-a-row property (§4) must survive the knob: with the
        measured price gone there is no priced axis left to order, so all
        three preferences take `local` and none of them explains a
        preference effect."""
        monkeypatch.setattr(ec, "ENGINE_CONSTRAINTS",
                            tuple(r for r in ec.ENGINE_CONSTRAINTS
                                  if r.id != "local-indexer-prefill-cost"))
        for i, p in enumerate(PREFERENCES):
            sv = _prefer(_contested_solver(tmp_path / f"del{i}", champion), p)
            res = sv.solve()
            assert res.recipe["hardware"]["dcp_indexer_mode"] == "local", p
            assert res.explanations.get("autoconfig.prefer") is None, p


GLM5NEXT_CONFIG = os.path.join(REPO, "test-data", "config", "glm53_flash_gguf.json")


@pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                    reason="glm5_next reference config absent")
class TestGlm5NextTensorParallel:
    """INV-KDA-TP (GF3.10): TP is correct on glm5_next but decode-neutral, so
    TP=1 is the default and the head-divisible ceiling is a CAPACITY fallback.
    A throughput finding must never turn into a refusal."""

    def _shape(self):
        with open(GLM5NEXT_CONFIG) as f:
            return ModelShape.from_model_section(json.load(f)["model"])

    def _solver(self, tmp_path, pinned_gib, name="box"):
        proc, sysd = build_fake_tree(str(tmp_path / name))
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        shape = self._shape()
        return Solver(hw, shape, {"model": shape.raw}, Levers(0.02, 51200),
                      expert_slot_bytes=SLOT_BYTES,
                      non_expert_pinned_bytes=int(pinned_gib * (1 << 30)))

    def test_tp1_is_the_measured_default(self, tmp_path):
        sv = self._solver(tmp_path, 10)
        res = sv.solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 1
        assert res.recipe["hardware"]["tp_array"] == [0]
        # the second flagship becomes an expert host, not a TP peer
        assert res.recipe["hardware"]["gpus"][1]["roles"] == ["expert_streaming"]
        e = sv.ex.get("parallelism.tensor_parallelism")
        assert e.kind == "measured"
        assert "latency-floored" in e.because
        assert "glm5next-tp1-default" in e.refs

    def test_escalates_to_the_ceiling_for_capacity_not_speed(self, tmp_path):
        # 26 GiB of pinned weights does not leave room for KV+state at TP=1;
        # sharding is then the only way to serve the ask at all
        sv = self._solver(tmp_path, 26)
        res = sv.solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 2
        e = sv.ex.get("parallelism.tensor_parallelism")
        assert e.kind == "searched"
        assert "CAPACITY, NOT SPEED" in e.because
        # and it still tells the operator what that costs
        assert "0.5-2%" in e.because

    def test_deleting_the_row_restores_the_plain_ceiling(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ec, "ENGINE_CONSTRAINTS",
                            tuple(r for r in ec.ENGINE_CONSTRAINTS
                                  if r.id != "glm5next-tp1-default"))
        sv = self._solver(tmp_path, 10)
        res = sv.solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 2
        assert sv.ex.get("parallelism.tensor_parallelism").kind == "closed_form"

    def test_the_row_states_what_would_falsify_it(self):
        row = ec.get("glm5next-tp1-default")
        assert row.kind == "measured" and row.scope == "glm5_next"
        # the prefill evidence is ONE small shape; a reader must see the bound
        assert "141-token" in row.statement
        assert "falsify" in row.statement
        assert "glm53_flash.md" in row.site

    def test_the_champion_arch_is_untouched(self, tmp_path, champion):
        # the row is scoped to glm5_next; MLA+DSA archs keep the ceiling
        res = make_solver(tmp_path, champion).solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 2


@pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                    reason="glm5_next reference config absent")
class TestGlm5NextAttentionBackendIsTurboquant:
    """TD-GLM5-TQ-BACKEND-UNWIRED RESOLVED (2026-09-01): glm5_next serves
    turboquant_mla end-to-end (tp=1 live verification + GLM53_TQ=1 golden
    arm), so the temporary `glm5next-tq-backend-unwired` engine-gate row is
    DELETED and the derivation is back on the family template.  The earlier
    "TQ generates worse" evidence was the EP4 combine defect
    (TD-GLM53-EP4-DEGENERATE-GENERATION), not TQ.  The row-lookup MECHANISM
    in Solver.attention_backend() stays: re-introducing an engine-gate row
    with that id must flip the derivation back to snapmla (the data-flip
    direction is pinned below, the TD-KVT-LOCAL-INDEXER-UNBLOCK pattern)."""

    def _solver(self, tmp_path, name="tqbox"):
        with open(GLM5NEXT_CONFIG) as f:
            shape = ModelShape.from_model_section(json.load(f)["model"])
        proc, sysd = build_fake_tree(str(tmp_path / name))
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        return Solver(hw, shape, {"model": shape.raw}, Levers(0.02, 51200),
                      expert_slot_bytes=SLOT_BYTES,
                      non_expert_pinned_bytes=10 * (1 << 30))

    def test_derives_turboquant_with_this_geometry_byte_argument(self, tmp_path):
        sv = self._solver(tmp_path)
        res = sv.solve()
        assert res.recipe["compute"]["attention_backend"] == "turboquant_mla"
        e = sv.ex.get("compute.attention_backend")
        assert e.kind == "measured"
        # glm5_next NoPE geometry: 258 vs 516 B/row (2.0x) — the numbers are
        # COMPUTED from the shape, never quoted from GLM-5.2's 386/644 row
        assert "258 B" in e.because
        assert "516 B" in e.because
        assert "386" not in e.because
        assert any("TD-GLM5-TQ-BACKEND-UNWIRED" in r for r in e.refs)

    def test_the_whole_byte_model_follows_the_served_backend(self, tmp_path):
        sv = self._solver(tmp_path)
        assert sv.attention_backend() == "turboquant_mla"

    def test_reintroducing_the_gate_row_moves_the_derivation_back(
            self, tmp_path, monkeypatch):
        # The registry stays DATA: if TQ regresses, adding the engine-gate
        # row back (same id) must flip glm5_next to snapmla with no code
        # change — and the explain sheet must say TEMPORARY again.
        row = ec.ConstraintRow(
            id="glm5next-tq-backend-unwired",
            kind="engine_gate",
            scope="glm5_next",
            statement="TEMPORARY (delete this row, do not edit it): "
                      "glm5_next serves on attention_backend snapmla ONLY.",
            ticket="TD-GLM5-TQ-BACKEND-UNWIRED",
            site="test reintroduction")
        monkeypatch.setattr(ec, "ENGINE_CONSTRAINTS",
                            ec.ENGINE_CONSTRAINTS + (row,))
        sv = self._solver(tmp_path, name="tqbox2")
        res = sv.solve()
        assert res.recipe["compute"]["attention_backend"] == "snapmla"
        e = sv.ex.get("compute.attention_backend")
        assert e.kind == "template"
        assert "glm5next-tq-backend-unwired" in e.refs
        # and the byte model follows the gate, not the template
        assert sv.attention_backend() == "snapmla"

    def test_the_row_is_gone(self):
        # the resolved ticket retired its row IN THE SLICE THAT RESOLVED IT
        # (the TD-AUTOCONFIG-STALE-KDA-CARVE-ROW lesson)
        assert ec.get("glm5next-tq-backend-unwired") is None

    def test_glm52_champion_keeps_turboquant(self, tmp_path, champion):
        # the glm5_next row never existed for MLA+DSA: the champion's
        # measured choice must not move in either direction
        res = make_solver(tmp_path, champion).solve()
        assert res.recipe["compute"]["attention_backend"] == "turboquant_mla"


@pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                    reason="glm5_next reference config absent")
class TestMappedKdaAdmissibility:
    """TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA + TD-AUTOCONFIG-MAXSEQ-IGNORES-
    MAPPED-KDA: since the mapped-state default (2026-08-31) the KDA recurrent
    state is a TENANT of the shared kv_main slab pool, not a dedicated carve.
    Two things follow, and both are pinned here.

    (1) The per-request charge is the SLAB-PADDED demand, not the slot: the
        pool hands out whole slabs, so 150.2 MiB/request against a 145.6 MiB
        slot, and sizing at the slot under-charges every request.
    (2) KV pages, the FULL-LENGTH indexer-K reservation and that state all
        draw on the ONE pool at admission — so a max_sequence_length may only
        be advertised if the pool holds their SUM for one max-length request.
        The GF3 1M arm advertised 500,000 and refused at 474,880.

    Registry row `kda-state-mapped-tenant` carries (1); deleting it must send
    the charge back to the carve model with no solver edit (§4)."""

    def _shape(self):
        with open(GLM5NEXT_CONFIG) as f:
            return ModelShape.from_model_section(json.load(f)["model"])

    def _solver(self, tmp_path, name="adm", ctx=51200, pinned_gib=10,
                bar1=32, ratio=0.02):
        gpus = [(b, m, u, (bar1 if "5090" in m else 16), n)
                for b, m, u, _, n in BOX_GPUS]
        proc, sysd = build_fake_tree(str(tmp_path / name), gpus=gpus)
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        shape = self._shape()
        return Solver(hw, shape, {"model": shape.raw}, Levers(ratio, ctx),
                      expert_slot_bytes=SLOT_BYTES,
                      non_expert_pinned_bytes=int(pinned_gib * (1 << 30)))

    def _slab_bytes(self, sv):
        return sizing.slab_geometry(
            sv.shape, sv.kv_quant, sv.attention_backend(),
            sv.k.page_size_tokens, sv.k.indexer_k_page_size_tokens).slab_bytes

    # ---- (1) the charge --------------------------------------------------

    def test_the_state_is_charged_slab_padded_as_shared_pool_capacity(
            self, tmp_path):
        sv = self._solver(tmp_path)
        res = sv.solve()
        plan = res.gpu_plans[0]
        slots = sizing.kda_policy_slots(
            res.recipe["serving"]["max_concurrent_requests"], True,
            sv.k.prefix_cache_max_entries)
        mapped = slots * sizing.kda_mapped_demand_bytes(
            sv.shape, res.recipe["parallelism"]["tensor_parallelism"],
            self._slab_bytes(sv))
        assert plan.kda_mapped is True
        assert plan.kda_bytes == mapped
        # and it is NOT the slot model: that is the 4.6 MiB/request the
        # stale carve row was hiding
        carve = slots * sizing.kda_slot_bytes(
            sv.shape, res.recipe["parallelism"]["tensor_parallelism"])
        assert carve < plan.kda_bytes

    def test_deleting_the_row_reverts_to_the_carve_slot_model(
            self, tmp_path, monkeypatch):
        """The delete-a-row property (§4): the carve survives as the engine's
        off-path (LS_KDA_STATE_MAPPED=0). Should it become the default again,
        deleting this ONE row must move the charge back — data, not code."""
        monkeypatch.setattr(ec, "ENGINE_CONSTRAINTS",
                            tuple(r for r in ec.ENGINE_CONSTRAINTS
                                  if r.id != "kda-state-mapped-tenant"))
        sv = self._solver(tmp_path, name="carve")
        res = sv.solve()
        plan = res.gpu_plans[0]
        slots = sizing.kda_policy_slots(
            res.recipe["serving"]["max_concurrent_requests"], True,
            sv.k.prefix_cache_max_entries)
        assert plan.kda_mapped is False
        assert plan.kda_bytes == slots * sizing.kda_slot_bytes(
            sv.shape, res.recipe["parallelism"]["tensor_parallelism"])
        # a carve is not pool capacity, so it charges no state pages either
        assert plan.adm_state_pages == 0

    # ---- (2) the admissibility arithmetic --------------------------------

    def test_the_plan_carries_the_decomposed_admission_demand(self, tmp_path):
        """A single ceiling number cannot say WHICH tenant bound it — the
        misreading that cost the GF3 arm a ladder rung — so the plan carries
        the three terms and their sum."""
        sv = self._solver(tmp_path, name="terms")
        res = sv.solve()
        p = res.gpu_plans[0]
        assert p.adm_pool_pages > 0
        assert p.adm_kv_pages > 0 and p.adm_idx_pages > 0
        assert p.adm_state_pages > 0          # mapped: the state IS a tenant
        assert p.adm_demand_pages == (p.adm_kv_pages + p.adm_idx_pages
                                      + p.adm_state_pages)
        # the ceiling is the mirror's own binary search over those inputs,
        # not a second derivation that could drift from it
        assert p.admissible_max_seq == sizing.admissible_context_tokens(
            p.adm_pool_pages, sv.k.page_size_tokens,
            sv.shape.engine_kv_pool_layers(), sv.shape.num_dsa_computing_layers,
            sv.k.indexer_k_page_size_tokens,
            sizing.slab_geometry(sv.shape, sv.kv_quant, sv.attention_backend(),
                                 sv.k.page_size_tokens,
                                 sv.k.indexer_k_page_size_tokens).pages_per_slab,
            p.adm_state_pages)

    def test_the_explain_sheet_always_shows_the_margin(self, tmp_path):
        """Shown even when it clears comfortably: a 0.66% miss has to be
        visible BEFORE the boot, and the engine prints the same figures at
        boot for enginecheck to compare against."""
        res = self._solver(tmp_path, name="explain").solve()
        e = res.explanations.get("autoconfig.context_admissibility")
        assert e is not None and e.kind == "closed_form"
        p = res.gpu_plans[0]
        assert str(p.adm_demand_pages) in e.because
        assert str(p.adm_pool_pages) in e.because
        assert "admissible context (single request)" in e.because
        assert "kda-state-mapped-tenant" in e.refs
        assert "autoconfig.context_admissibility" in \
            res.explanations.render_markdown()

    def test_the_emitted_context_is_never_inadmissible(self, tmp_path):
        """The invariant the whole slice exists for, swept over boxes that
        span comfortable, VRAM-clamped and impossible: whatever the ladder
        does — serve the ask, shed concurrency, halve max_seq, or refuse —
        the recipe it EMITS must never advertise a context its own pool
        cannot admit for a single request.

        NOTE for a future reader: the reduce branch in solve() does not fire
        on today's fit model, and that is a property of the sizing, not a
        gap in this test. The pool is sized at `conc` KV shares and
        `policy_slots` state shares while admission charges ONE of each, so
        the pool is over-provisioned against a single request by
        construction, and a pool too small to hold even that raises
        `kv-demand-exceeds-vram` first and the lattice sheds. The branch is
        the guard for the day that stops being true (an elastic or
        lent-out pool); the invariant below is what must hold either way,
        so it is what is asserted."""
        cases = [
            ("comfortable", dict(ctx=51200, pinned_gib=10)),
            ("big-ask", dict(ctx=819200, pinned_gib=10)),
            ("clamped", dict(ctx=2097152, pinned_gib=24)),
            ("tight-bar", dict(ctx=819200, pinned_gib=22, bar1=20)),
            ("experts-hungry", dict(ctx=409600, pinned_gib=14, ratio=0.2)),
            ("impossible", dict(ctx=3145728, pinned_gib=26, bar1=17)),
        ]
        fits = refusals = 0
        for name, kw in cases:
            try:
                res = self._solver(tmp_path, name=name, **kw).solve()
            except Infeasible:
                refusals += 1
                continue
            fits += 1
            p = res.gpu_plans[0]
            if p.admissible_max_seq <= 0:
                continue          # tiering/V4 pools are not page-modelled
            assert res.recipe["serving"]["max_sequence_length"] <= \
                p.admissible_max_seq, name
            assert p.adm_demand_pages <= p.adm_pool_pages, name
        assert fits and refusals, "the sweep must exercise both sides"


V4_CONFIG = os.path.join(REPO, "test-data", "config",
                         "deepseek_v4_flash_gguf.json")


class TestAccuracyLever:
    """AUTOCONFIG §2.4 / TD-AUTOCONFIG-ACCURACY-LEVER — the 4th lever.

    `accuracy` FLOORS the family's numerics ladder
    (templates.ACCURACY_LADDER); it is not a weight — tok/s and perplexity
    have no honest exchange rate (the TD-AUTOCONFIG-SPEED-BUDGET argument),
    so the lever orders options and never trades them arithmetically.  The
    default `standard` tier is proven byte-identical to the pre-lever
    derivation the `prefer` way (scratchpad/accuracy_lever/NOOP_PROOF.txt:
    git-archive pre-tree, 12 scenarios, full-derivation diff)."""

    def _accuracy(self, sv, tier):
        sv.levers = Levers(sv.levers.vram_expert_ratio,
                           sv.levers.total_active_context_tokens,
                           sv.levers.prefer, tier)
        return sv

    def _glm5(self, tmp_path, levers, pinned_gib=25):
        with open(GLM5NEXT_CONFIG) as f:
            shape = ModelShape.from_model_section(json.load(f)["model"])
        proc, sysd = build_fake_tree(str(tmp_path / "t"))
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        return Solver(hw, shape, {"model": shape.raw}, levers,
                      expert_slot_bytes=SLOT_BYTES,
                      non_expert_pinned_bytes=pinned_gib * (1 << 30))

    # ---- the regression guard: `standard` is the shipped derivation -------

    def test_standard_is_the_default_and_a_no_op(self, tmp_path, champion):
        """Naming the default must change nothing, field for field —
        recipe, explanation table, warnings and degradations (the
        cross-TREE byte-identity is pinned by NOOP_PROOF.txt; this pins
        the in-tree property so it cannot rot)."""
        assert Levers().accuracy == "standard"
        mks = (lambda t: make_solver(t, champion),
               lambda t: _contested_solver(t, champion),
               lambda t: make_solver(t, champion, with_draft=False))
        for i, mk in enumerate(mks):
            a = _derivation(mk(tmp_path / f"a{i}"))
            b = _derivation(self._accuracy(mk(tmp_path / f"b{i}"), "standard"))
            assert a == b
        res = self._accuracy(make_solver(tmp_path / "c", champion),
                             "standard").solve()
        assert res.explanations.get("autoconfig.accuracy") is None

    # ---- what each tier does ----------------------------------------------

    @pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                        reason="glm5_next reference config absent")
    def test_high_floors_glm5next_at_snapmla_and_names_both_prices(
            self, tmp_path):
        """accuracy=high on glm5_next: the backend moves to the all-FP8
        KV path and BOTH sides of the trade are named — the measured
        accuracy basis (registry row) and the computed 258->516 B/row
        capacity price."""
        sv = self._glm5(tmp_path, Levers(0.02, 51200, "balanced", "high"))
        res = sv.solve()
        assert res.recipe["compute"]["attention_backend"] == "snapmla"
        e = res.explanations.get("compute.attention_backend")
        assert e is not None and e.kind == "measured"
        assert "accuracy=high" in e.because
        assert "516 B/row" in e.because and "258 B/row" in e.because
        assert any("mla-tq-vs-snapmla-accuracy" in r for r in e.refs)
        a = res.explanations.get("autoconfig.accuracy")
        assert a is not None and a.value == "high"
        assert "`turboquant_mla` -> `snapmla`" in a.because
        assert "never a weight" in a.because

    @pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                        reason="glm5_next reference config absent")
    def test_high_pays_its_capacity_price_through_the_ordinary_fit(
            self, tmp_path):
        """The 2.0x KV rows halve what the same ask serves: the standard
        tier's 409600-token ask serves 1 x 204800 under TQ (the
        TD-GLM5-TQ-BACKEND-UNWIRED capacity fixture); accuracy=high pays
        for snapmla rows with exactly half the context — through the
        NORMAL E1 degradation ladder, not through any accuracy-specific
        arithmetic."""
        std = self._glm5(tmp_path / "s", Levers(0.02, 409600)).solve()
        assert std.recipe["serving"]["max_sequence_length"] == 204800
        hi = self._glm5(tmp_path / "h",
                        Levers(0.02, 409600, "balanced", "high")).solve()
        assert hi.recipe["compute"]["attention_backend"] == "snapmla"
        assert hi.recipe["serving"]["max_sequence_length"] == 102400
        assert hi.degradations

    def test_compact_is_silent_where_it_changes_nothing(self, tmp_path,
                                                        champion):
        """On the MLA families the template already sits at the compact
        end (turboquant_mla), so accuracy=compact derives the identical
        recipe and the sidecar must not mention the lever at all."""
        base = _derivation(make_solver(tmp_path / "a", champion))
        res = self._accuracy(make_solver(tmp_path / "b", champion),
                             "compact").solve()
        assert res.recipe == base[0]
        assert res.explanations.get("autoconfig.accuracy") is None
        assert "autoconfig.accuracy" not in res.explanations.render_markdown()

    @pytest.mark.skipif(not os.path.exists(V4_CONFIG),
                        reason="deepseek_v4 reference config absent")
    def test_v4_ladder_maps_all_three_tiers(self, tmp_path):
        """V4 has a genuine 3-way ladder: csa_hca (all-FP8, the native
        trained format) > csa_hca_tq_mix (template) > csa_hca_tq (all-TQ).
        The tier mapping is pinned at the _backend_for_tier level; the
        basis on V4 is STRUCTURAL (nested quantization), so the explain
        kind must be heuristic, never measured."""
        with open(V4_CONFIG) as f:
            shape = ModelShape.from_model_section(json.load(f)["model"])
        proc, sysd = build_fake_tree(str(tmp_path / "t"))
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        def sv(tier):
            return Solver(hw, shape, {"model": shape.raw},
                          Levers(0.02, 51200, "balanced", tier),
                          expert_slot_bytes=SLOT_BYTES,
                          non_expert_pinned_bytes=20 * (1 << 30))
        assert sv("standard")._backend_for_tier("standard") == "csa_hca_tq_mix"
        assert sv("high")._backend_for_tier("high") == "csa_hca"
        assert sv("compact")._backend_for_tier("compact") == "csa_hca_tq"

    # ---- the honest stub --------------------------------------------------

    def test_superior_is_defined_but_refuses(self, tmp_path, champion):
        """`superior` (KV bf16, no TQ, everything maximal) is DEFINED and
        NOT implemented: the engine builds no kFull codec arm
        (kv_codec.h:12-14), so the tier refuses at derive time naming that
        fact — it must never be silently approximated by a lesser tier."""
        sv = make_solver(tmp_path, champion,
                         levers=Levers(0.0627, 51200, "balanced", "superior"))
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert ei.value.constraint_id == "accuracy-superior-not-implemented"
        assert "kFull" in str(ei.value)
        assert "kv_codec.h" in str(ei.value)
        assert "accuracy=high" in str(ei.value)

    def test_an_unknown_tier_is_refused_at_construction(self, tmp_path,
                                                        champion):
        with pytest.raises(ValueError) as ei:
            make_solver(tmp_path, champion,
                        levers=Levers(0.0627, 51200, "balanced", "maximal"))
        assert "accuracy='maximal'" in str(ei.value)
        assert "compact/standard/high/superior" in str(ei.value)

    # ---- constraints stay DATA under every tier ---------------------------

    @pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                        reason="glm5_next reference config absent")
    def test_a_reintroduced_gate_row_filters_the_menu_for_every_tier(
            self, tmp_path, monkeypatch):
        """If TQ regresses and the engine-gate row returns, turboquant_mla
        leaves the MENU: every tier lands on snapmla and the lever goes
        silent (no tier can change an outcome the menu no longer offers).
        The pre-lever data-flip test (TestGlm5NextAttentionBackendIs-
        Turboquant) still pins the standard tier's direction."""
        row = ec.ConstraintRow(
            id="glm5next-tq-backend-unwired", kind="engine_gate",
            scope="glm5_next", statement="test reintroduction", ticket="T")
        monkeypatch.setattr(ec, "ENGINE_CONSTRAINTS",
                            ec.ENGINE_CONSTRAINTS + (row,))
        for tier in ("compact", "standard", "high"):
            sv = self._glm5(tmp_path / tier,
                            Levers(0.02, 51200, "balanced", tier))
            res = sv.solve()
            assert res.recipe["compute"]["attention_backend"] == "snapmla", tier
            assert res.explanations.get("autoconfig.accuracy") is None, tier

    @pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                        reason="glm5_next reference config absent")
    def test_deleting_the_measured_row_downgrades_the_basis_not_the_floor(
            self, tmp_path, monkeypatch):
        """The delete-a-row property stays meaningful: without the measured
        accuracy row the tier still floors (the precision ordering is
        structural) but the explain HONESTLY downgrades to heuristic and
        says the trade is unpriced."""
        monkeypatch.setattr(ec, "ENGINE_CONSTRAINTS",
                            tuple(r for r in ec.ENGINE_CONSTRAINTS
                                  if r.id != "mla-tq-vs-snapmla-accuracy"))
        sv = self._glm5(tmp_path, Levers(0.02, 51200, "balanced", "high"))
        res = sv.solve()
        assert res.recipe["compute"]["attention_backend"] == "snapmla"
        e = res.explanations.get("compute.attention_backend")
        assert e is not None and e.kind == "heuristic"
        assert "unpriced" in e.because


# ---------------------------------------------------------------------------
# TD-AUTOCONFIG-PINNED-CONSTRAINTS


from autoconfig.pins import Pin, PinSet  # noqa: E402


def _pins(d):
    return PinSet({p: Pin(p, v, "test") for p, v in d.items()})


def _with_pins(sv, d):
    sv.pins = _pins(d)
    return sv


class TestPinnedConstraints:
    """TD-AUTOCONFIG-PINNED-CONSTRAINTS — pins are a THIRD input class:
    not the identity base (--config, freely overwritten), not a lever (an
    ask the ladder degrades) — a HARD constraint that survives every E1
    rung. A pin REMOVES an axis from the lattice; an infeasible pin set
    refuses NAMING the pin; the sheet marks pinned fields `pinned` (told,
    not derived); a pin contradicting a measured registry row derives
    LOUDLY, one contradicting an engine gate refuses.
    Design: spec/plans/AUTOCONFIG_PINNED_CONSTRAINTS.md."""

    def _glm5(self, tmp_path, pinned_gib=10, ctx=51200, name="g5"):
        with open(GLM5NEXT_CONFIG) as f:
            shape = ModelShape.from_model_section(json.load(f)["model"])
        proc, sysd = build_fake_tree(str(tmp_path / name))
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        return Solver(hw, shape, {"model": shape.raw}, Levers(0.02, ctx),
                      expert_slot_bytes=SLOT_BYTES,
                      non_expert_pinned_bytes=int(pinned_gib * (1 << 30)))

    def _lattice_of(self, sv):
        ordered = sv.order_gpus()
        tps = sv.tp_plan(ordered)
        with sv._quiet():
            setups = {tp: sv._tp_setup(ordered, tp) for tp in tps}
        c0, m0 = sv.active_split()
        return sv._lattice(tps, setups, c0, m0)

    # ---- no pins is a byte-level no-op (design §8) ------------------------

    def test_no_pins_is_a_no_op(self, tmp_path, champion):
        """An EMPTY pin set must normalise to None and derive identically —
        the cross-tree byte proof is scratchpad/pinned_constraints/
        NOOP_PROOF.txt; this is the in-tree property."""
        a = _derivation(make_solver(tmp_path / "a", champion))
        sv = make_solver(tmp_path / "b", champion)
        sv.pins = None
        b = _derivation(sv)
        sv2 = make_solver(tmp_path / "c", champion)
        sv2.pins = None if len(_pins({})) == 0 else _pins({})
        c = _derivation(sv2)
        assert a == b == c
        assert "| pinned |" not in a[1]

    # ---- (b) the sheet never claims to have derived what it was told ------

    def test_backend_pin_is_marked_pinned_and_loud(self, tmp_path, champion):
        """The champion's standard tier derives turboquant_mla; pinning
        snapmla must carry through recipe + byte model, be marked `pinned`
        (never re-narrated as measured/searched), and LOUDLY quote the
        measured accuracy row it moves against (a)."""
        sv = _with_pins(make_solver(tmp_path, champion),
                        {"compute.attention_backend": "snapmla"})
        res = sv.solve()
        assert res.recipe["compute"]["attention_backend"] == "snapmla"
        e = res.explanations.get("compute.attention_backend")
        assert e is not None and e.kind == "pinned"
        assert "told, not derived" in e.because
        assert "TD-AUTOCONFIG-PINNED-CONSTRAINTS" in e.refs
        # loud: the measured row is quoted with the pin, in the warnings
        w = [w for w in res.warnings if "mla-tq-vs-snapmla-accuracy" in w]
        assert w and "compute.attention_backend" in w[0]
        assert "override" in w[0].lower()
        # the accuracy lever changed nothing (the axis is removed): silent
        assert res.explanations.get("autoconfig.accuracy") is None
        # the sheet renders the pinned kind
        assert "| pinned |" in res.explanations.render_markdown()

    def test_every_pinned_path_stays_pinned_in_the_sheet(self, tmp_path,
                                                         champion):
        pins = {"compute.attention_backend": "snapmla",
                "hardware.dcp_indexer_mode": "local",
                "serving.max_sequence_length": 25600,
                "serving.max_concurrent_requests": 2,
                "speculation.method": "none",
                "hardware.dcp_kv_mode": "sharded"}
        sv = _with_pins(make_solver(tmp_path, champion), pins)
        res = sv.solve()
        for path in pins:
            e = res.explanations.get(path)
            assert e is not None and e.kind == "pinned", path
        # and the recipe carries every pin
        assert res.recipe["compute"]["attention_backend"] == "snapmla"
        assert res.recipe["hardware"]["dcp_indexer_mode"] == "local"
        assert res.recipe["hardware"]["dcp_kv_mode"] == "sharded"
        assert res.recipe["serving"]["max_sequence_length"] == 25600
        assert res.recipe["serving"]["max_concurrent_requests"] == 2
        assert res.recipe["speculation"]["method"] == "none"

    # ---- (a) measured rows: overridable, LOUDLY ---------------------------

    def test_idx_pin_local_overrides_the_measured_row_loudly(self, tmp_path,
                                                             champion):
        """The ticket's own use case: forcing dcp_indexer_mode=local to
        measure its cost. The solver's measured default is replicated
        (TestIndexerModeCost); the pin lands local at the FIRST rung and
        the warning quotes both the pin and the priced row."""
        sv = _with_pins(make_solver(tmp_path, champion),
                        {"hardware.dcp_indexer_mode": "local"})
        res = sv.solve()
        assert res.recipe["hardware"]["dcp_indexer_mode"] == "local"
        e = res.explanations.get("hardware.dcp_indexer_mode")
        assert e is not None and e.kind == "pinned"   # immutability held
        w = [w for w in res.warnings if "local-indexer-prefill-cost" in w]
        assert w and "8-9%" in w[0] and "hardware.dcp_indexer_mode" in w[0]
        # not a degradation: the pin is an instruction, not a shed
        assert not res.degradations

    def test_tp_pin_on_glm5next_quotes_the_measured_row(self, tmp_path):
        sv = _with_pins(self._glm5(tmp_path),
                        {"parallelism.tensor_parallelism": 2})
        res = sv.solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 2
        assert res.recipe["hardware"]["tp_array"] == [0, 1]
        e = res.explanations.get("parallelism.tensor_parallelism")
        assert e is not None and e.kind == "pinned"
        w = [w for w in res.warnings if "glm5next-tp1-default" in w]
        assert w and "parallelism.tensor_parallelism=2" in w[0]

    def test_tiering_pin_flips_glm5next_and_quotes_the_row(self, tmp_path):
        sv = _with_pins(self._glm5(tmp_path),
                        {"memory.kv_tiering.enabled": True})
        res = sv.solve()
        assert res.recipe["memory"]["kv_tiering"]["enabled"] is True
        e = res.explanations.get("memory.kv_tiering.enabled")
        assert e is not None and e.kind == "pinned"
        w = [w for w in res.warnings if "glm5next-tiering-default-off" in w]
        assert w and "memory.kv_tiering.enabled=true" in w[0]

    def test_deleting_the_measured_row_silences_the_quote(self, tmp_path,
                                                          champion,
                                                          monkeypatch):
        """Delete-a-row stays meaningful: the pin still applies, the
        override warning dies with the row it quoted."""
        monkeypatch.setattr(ec, "ENGINE_CONSTRAINTS",
                            tuple(r for r in ec.ENGINE_CONSTRAINTS
                                  if r.id != "local-indexer-prefill-cost"))
        sv = _with_pins(make_solver(tmp_path, champion),
                        {"hardware.dcp_indexer_mode": "local"})
        res = sv.solve()
        assert res.recipe["hardware"]["dcp_indexer_mode"] == "local"
        assert not [w for w in res.warnings
                    if "local-indexer-prefill-cost" in w]

    # ---- (c) a pin REMOVES an axis, and composes with prefer/accuracy -----

    def test_max_seq_pin_removes_the_halving_rungs(self, tmp_path, champion):
        sv = _with_pins(make_solver(tmp_path, champion),
                        {"serving.max_sequence_length": 25600})
        lat = self._lattice_of(sv)
        assert lat and all(a.max_seq == 25600 for a in lat)
        # concurrency (unpinned) still sheds
        assert {a.conc for a in lat} == {1, 2}

    def test_conc_pin_removes_the_shed_rungs(self, tmp_path, champion):
        sv = _with_pins(make_solver(tmp_path, champion),
                        {"serving.max_concurrent_requests": 2})
        lat = self._lattice_of(sv)
        assert lat and all(a.conc == 2 for a in lat)
        # the halving rungs (unpinned axis) keep the pinned concurrency
        assert len({a.max_seq for a in lat}) > 1

    def test_pinned_lattice_is_identical_across_preferences(self, tmp_path,
                                                            champion):
        """Per-pin-set attempt-set identity (design §5): for a FIXED pin
        set every preference walks the same lattice, so `prefer` still
        cannot cause a refusal another preference would avoid."""
        pins = {"hardware.dcp_indexer_mode": "local",
                "serving.max_concurrent_requests": 2}
        sets = []
        for i, p in enumerate(PREFERENCES):
            sv = _with_pins(_prefer(
                make_solver(tmp_path / f"p{i}", champion), p), pins)
            sets.append(frozenset(self._lattice_of(sv)))
        assert sets[0] == sets[1] == sets[2]
        assert all(a.idx_mode == "local" for a in sets[0])

    def test_accuracy_floor_conflict_refuses_naming_both(self, tmp_path,
                                                         champion):
        """The interesting corner (c): accuracy=high floors at snapmla; a
        turboquant_mla pin makes the floor unsatisfiable. Two hard
        statements about ONE axis: refuse, never guess."""
        sv = _with_pins(
            make_solver(tmp_path, champion,
                        levers=Levers(0.0627, 51200, "balanced", "high")),
            {"compute.attention_backend": "turboquant_mla"})
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert ei.value.constraint_id == "pin-conflicts-accuracy-floor"
        s = str(ei.value)
        assert "turboquant_mla" in s and "snapmla" in s
        assert "accuracy=high" in s

    def test_accuracy_floor_agreement_is_silent(self, tmp_path, champion):
        """accuracy=high + pin snapmla agree: derive, mark pinned, and the
        tier row stays silent (it changed nothing — the axis is removed)."""
        sv = _with_pins(
            make_solver(tmp_path, champion,
                        levers=Levers(0.0627, 51200, "balanced", "high")),
            {"compute.attention_backend": "snapmla"})
        res = sv.solve()
        assert res.recipe["compute"]["attention_backend"] == "snapmla"
        assert res.explanations.get(
            "compute.attention_backend").kind == "pinned"
        assert res.explanations.get("autoconfig.accuracy") is None

    def test_standard_tier_never_conflicts_with_a_backend_pin(self, tmp_path,
                                                              champion):
        """`standard` is the no-preference tier: the pin wins (loudly, via
        the measured-row warning) instead of refusing."""
        sv = _with_pins(make_solver(tmp_path, champion),
                        {"compute.attention_backend": "snapmla"})
        res = sv.solve()   # must not raise
        assert res.recipe["compute"]["attention_backend"] == "snapmla"

    # ---- engine gates and geometry are NOT overridable --------------------

    @pytest.mark.skipif(not os.path.exists(V4_CONFIG),
                        reason="v4 reference config absent")
    def test_v4_sharded_kv_pin_refuses_quoting_the_gate(self, tmp_path):
        with open(V4_CONFIG) as f:
            shape = ModelShape.from_model_section(json.load(f)["model"])
        proc, sysd = build_fake_tree(str(tmp_path / "t"))
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        sv = Solver(hw, shape, {"model": shape.raw}, Levers(0.02, 51200),
                    expert_slot_bytes=SLOT_BYTES,
                    non_expert_pinned_bytes=20 * (1 << 30),
                    pins=_pins({"hardware.dcp_kv_mode": "sharded"}))
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert ei.value.constraint_id == "pinned-kv-mode-engine-gated"
        assert "v4-no-sharded-kv" in str(ei.value)

    def test_unserveable_backend_pin_refuses_with_the_partition(
            self, tmp_path, champion):
        sv = _with_pins(make_solver(tmp_path, champion),
                        {"compute.attention_backend": "csa_hca"})
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert ei.value.constraint_id == "pinned-backend-not-serveable"
        assert "config_validator.cpp:510-534" in str(ei.value)

    def test_tp_pin_violating_geometry_refuses_naming_the_pin(
            self, tmp_path, champion):
        sv = _with_pins(make_solver(tmp_path, champion),
                        {"parallelism.tensor_parallelism": 3})
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert "parallelism.tensor_parallelism=3" in str(ei.value)

    def test_max_seq_pin_beyond_model_geometry_refuses(self, tmp_path,
                                                       champion):
        sv = _with_pins(make_solver(tmp_path, champion),
                        {"serving.max_sequence_length": 2 ** 21})
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert ei.value.constraint_id == "pinned-max-seq-exceeds-model"

    # ---- the draft axis ---------------------------------------------------

    def test_method_none_pin_never_charges_the_draft(self, tmp_path,
                                                     champion):
        sv = _with_pins(make_solver(tmp_path, champion),
                        {"speculation.method": "none"})
        res = sv.solve()
        assert res.recipe["speculation"]["method"] == "none"
        # no charge, no ghost row, no bogus degradation note
        assert res.explanations.get("speculation.dspark.draft_gpus") is None
        assert not any("draft dropped" in d for d in res.degradations)
        assert all(p.draft_bytes == 0 for p in res.gpu_plans)

    def test_method_dspark_pin_without_a_checkpoint_refuses(self, tmp_path,
                                                            champion):
        sv = _with_pins(make_solver(tmp_path, champion, with_draft=False),
                        {"speculation.method": "dspark"})
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert ei.value.constraint_id == "pinned-method-needs-draft"

    def _glm5_draft(self, tmp_path, ctx, pinned_gib):
        with open(GLM5NEXT_CONFIG) as f:
            shape = ModelShape.from_model_section(json.load(f)["model"])
        proc, sysd = build_fake_tree(str(tmp_path / "t"))
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        dpath = write_fake_draft(tmp_path)
        draft = load_draft_candidate(os.path.basename(dpath), str(tmp_path))
        return Solver(hw, shape, {"model": shape.raw}, Levers(0.02, ctx),
                      expert_slot_bytes=SLOT_BYTES,
                      non_expert_pinned_bytes=int(pinned_gib * (1 << 30)),
                      draft=draft)

    def test_method_dspark_pin_sheds_capacity_instead_of_the_draft(
            self, tmp_path):
        """On this glm5 box the unpinned ladder drops the draft FIRST and
        keeps the ask whole. Pinning dspark removes that axis: the draft
        SURVIVES and the fit sheds the ask instead."""
        res0 = self._glm5_draft(tmp_path / "u", 2097152, 14).solve()
        assert any("draft dropped" in d for d in res0.degradations), \
            "setup: the unpinned ladder must be the draft-dropping one"
        sv = _with_pins(self._glm5_draft(tmp_path / "p", 2097152, 14),
                        {"speculation.method": "dspark"})
        res = sv.solve()
        assert res.recipe["speculation"]["method"] == "dspark"
        assert any("concurrency" in d for d in res.degradations)
        assert not any("draft dropped" in d for d in res.degradations)

    # ---- (d) infeasibility names WHICH pin --------------------------------

    def test_infeasible_pin_is_named_not_relaxed(self, tmp_path, champion):
        """The 29-GiB-BAR box only fits by DROPPING the draft (the ladder's
        first shed). Pinning dspark forbids exactly that shed, so nothing
        fits — and the refusal must name the pin and prove a fit exists
        without it, never silently drop the pinned draft."""
        sv = _with_pins(make_solver(tmp_path, champion, bar1_5090=29),
                        {"speculation.method": "dspark"})
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert ei.value.constraint_id == "pinned-constraint-binds"
        s = str(ei.value)
        assert "speculation.method='dspark'" in s
        assert "a fit EXISTS without" in s
        assert "never quietly relaxed" in s

    def test_boxwide_infeasibility_keeps_the_ordinary_refusal(
            self, tmp_path, champion):
        """When the box is infeasible REGARDLESS of pins, the refusal is
        the ordinary preference-independent one — pins must not claim
        credit for a wall they did not build."""
        sv = _with_pins(make_solver(tmp_path, champion,
                                    levers=Levers(0.5, 51200)),
                        {"hardware.dcp_indexer_mode": "local"})
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert ei.value.constraint_id == "expert-ratio-exceeds-vram"

    def _glm5_adm(self, tmp_path, name):
        """A glm5 shape where the reduce branch genuinely fires: no prefix
        entries (the state pool share stops over-provisioning the pool
        against one request) and a 2 x 1M ask (probe-verified: unpinned,
        this reduces max_seq at the DEFAULT speculation fraction)."""
        with open(GLM5NEXT_CONFIG) as f:
            shape = ModelShape.from_model_section(json.load(f)["model"])
        proc, sysd = build_fake_tree(str(tmp_path / name))
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        return Solver(hw, shape, {"model": shape.raw}, Levers(0.02, 2097152),
                      AutoconfigKnobs(prefix_cache_max_entries=0),
                      expert_slot_bytes=SLOT_BYTES,
                      non_expert_pinned_bytes=22 * (1 << 30))

    def test_pinned_max_seq_refuses_at_the_admissibility_ceiling(
            self, tmp_path):
        """The reduce loop must never shave a pinned value (design §6) —
        first proving the same shape DOES reduce when unpinned."""
        res = self._glm5_adm(tmp_path, "u").solve()
        assert any("admissibility ceiling" in d for d in res.degradations), \
            "setup must reach the reduce branch"
        sv = _with_pins(self._glm5_adm(tmp_path, "p"),
                        {"serving.max_sequence_length": 1048576})
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert ei.value.constraint_id == "pinned-max-seq-not-admissible"
        s = str(ei.value)
        assert "kda-state-mapped-tenant" in s
        assert "mapped KDA state" in s and "indexer reservation" in s


class TestEmittedFieldHygiene:
    """TD-AUTOCONFIG-NO-SERVING-SURFACE (incl. the baked constants): every
    emitted field is derived-and-explained, carried-and-labelled, or
    ABSENT.  A template constant masquerading as a decision
    (max_batch_size without a row, speculation.enabled=true beside
    method='none', dcp_kv_mode='sharded' at tp=1 where there is no DCP to
    shard) is the failure class; the inverse failure is the solver
    silently DROPPING the GF3.13 serving surface and the GF3.15
    superchunk win on glm5_next."""

    def _glm5(self, tmp_path, base_extra=None, levers=None, pins=None):
        with open(GLM5NEXT_CONFIG) as f:
            shape = ModelShape.from_model_section(json.load(f)["model"])
        proc, sysd = build_fake_tree(str(tmp_path / "t"))
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        base = {"model": shape.raw}
        if base_extra:
            base.update(base_extra)
        sv = Solver(hw, shape, base, levers or Levers(0.02, 51200),
                    expert_slot_bytes=SLOT_BYTES,
                    non_expert_pinned_bytes=25 * (1 << 30))
        if pins:
            sv = _with_pins(sv, pins)
        return sv

    @pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                        reason="glm5_next reference config absent")
    def test_glm5next_serving_surface_and_template_constants(self, tmp_path):
        res = self._glm5(tmp_path).solve()
        r = res.recipe
        assert r["parallelism"]["tensor_parallelism"] == 1
        # the GF3.13 serving surface, from the family template, each with
        # its own row
        assert r["serving"]["tokenizer_mode"] == "glm5_next"
        assert r["serving"]["tool_call_parser"] == "glm47"
        assert r["serving"]["reasoning_parser"] == "glm45"
        assert r["serving"]["enable_auto_tool_choice"] is True
        for key in ("tokenizer_mode", "tool_call_parser",
                    "reasoning_parser", "enable_auto_tool_choice"):
            e = res.explanations.get(f"serving.{key}")
            assert e is not None and e.kind == "template", key
            assert "TD-AUTOCONFIG-NO-SERVING-SURFACE" in e.refs
        # the GF3.15 superchunk win, requested (the engine derives the
        # effective capacity elastically)
        assert r["compute"]["prefill_superchunk_tokens"] == 512
        e = res.explanations.get("compute.prefill_superchunk_tokens")
        assert e is not None and e.kind == "template"
        assert "35.9" in e.because and "38.1" in e.because
        # the GF3 schedule width, with a row
        assert r["orchestrator"]["max_batch_size"] == 512
        e = res.explanations.get("orchestrator.max_batch_size")
        assert e is not None and e.kind == "template"
        assert "recipes/glm53flash_serve.json" in e.refs

    @pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                        reason="glm5_next reference config absent")
    def test_speculation_enabled_follows_the_method(self, tmp_path):
        res = self._glm5(tmp_path).solve()   # no draft discoverable
        assert res.recipe["speculation"]["method"] == "none"
        assert res.recipe["speculation"]["enabled"] is False
        e = res.explanations.get("speculation.enabled")
        assert e is not None and e.kind == "closed_form"
        assert "P6" in e.because

    def test_speculation_enabled_true_with_dspark(self, tmp_path, champion):
        res = make_solver(tmp_path, champion).solve()
        assert res.recipe["speculation"]["method"] == "dspark"
        assert res.recipe["speculation"]["enabled"] is True

    @pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                        reason="glm5_next reference config absent")
    def test_inert_dcp_fields_absent_below_dcp2(self, tmp_path):
        res = self._glm5(tmp_path).solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 1
        assert "dcp_kv_mode" not in res.recipe["hardware"]
        assert "dcp_indexer_mode" not in res.recipe["hardware"]

    def test_dcp_fields_present_at_dcp2(self, tmp_path, champion):
        res = make_solver(tmp_path, champion).solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 2
        assert res.recipe["hardware"]["dcp_kv_mode"] == "sharded"
        assert res.recipe["hardware"]["dcp_indexer_mode"] == "replicated"

    @pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                        reason="glm5_next reference config absent")
    def test_pinned_dcp_field_is_emitted_even_when_inert(self, tmp_path):
        res = self._glm5(
            tmp_path, pins={"hardware.dcp_kv_mode": "replicated"}).solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 1
        assert res.recipe["hardware"]["dcp_kv_mode"] == "replicated"
        e = res.explanations.get("hardware.dcp_kv_mode")
        assert e is not None and e.kind == "pinned"
        assert any("INERT" in w for w in res.warnings)
        # the unpinned sibling stays absent
        assert "dcp_indexer_mode" not in res.recipe["hardware"]

    @pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                        reason="glm5_next reference config absent")
    def test_base_serving_surface_is_carried_over_the_template(self, tmp_path):
        res = self._glm5(tmp_path, base_extra={
            "serving": {"tool_call_parser": "custom_parser"}}).solve()
        assert res.recipe["serving"]["tool_call_parser"] == "custom_parser"
        e = res.explanations.get("serving.tool_call_parser")
        assert e is not None and e.kind == "carried"
        # the other three still come from the family template
        assert res.recipe["serving"]["tokenizer_mode"] == "glm5_next"
        assert res.explanations.get(
            "serving.tokenizer_mode").kind == "template"

    def test_champion_families_emit_no_serving_surface(self, tmp_path,
                                                       champion):
        """mla_dsa carries no serving-surface template and the champion
        base recipe says nothing — absent, never invented."""
        res = make_solver(tmp_path, champion).solve()
        for key in ("tokenizer_mode", "tool_call_parser",
                    "reasoning_parser", "enable_auto_tool_choice"):
            assert key not in res.recipe["serving"], key
        assert "prefill_superchunk_tokens" not in res.recipe["compute"]
        assert res.recipe["orchestrator"]["max_batch_size"] == 64
        e = res.explanations.get("orchestrator.max_batch_size")
        assert e is not None and e.kind == "template"


class TestDraftIdentity:
    """TD-AUTOCONFIG-DRAFT-IDENTITY — a .dspark checkpoint is matched to the
    model (hidden size, vocab/tokenizer, aux layer taps) BEFORE it is ever
    costed; a foreign one refuses, never gets silently priced."""

    def _foreign_shape(self, champion):
        """A GLM-5.3-Flash-like target: hidden 4096, 45 layers, other vocab —
        the live incident's geometry (the GLM-5.2 speculator was returned
        for it and costed at 9.14 GiB/rank)."""
        from dataclasses import replace
        shape = ModelShape.from_model_section(champion["model"])
        return replace(shape, hidden_size=4096, vocab_size=151552,
                       num_hidden_layers=45)

    def test_matching_draft_has_no_errors(self, tmp_path, champion):
        from autoconfig.solver import draft_identity_errors
        dpath = write_fake_draft(tmp_path)
        draft = load_draft_candidate(os.path.basename(dpath), str(tmp_path))
        shape = ModelShape.from_model_section(champion["model"])
        assert draft_identity_errors(draft, shape) == []

    def test_foreign_draft_names_every_mismatch(self, tmp_path, champion):
        """GLM-5.2 speculator vs a GLM-5.3-Flash-shaped target: hidden 6144
        vs 4096, vocab 154880 vs 151552, aux taps [8,23,39,55,70] reading
        layers a 45-layer model does not have."""
        from autoconfig.solver import draft_identity_errors
        d = tmp_path / "spec.dspark"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "aux_hidden_state_layer_ids": [8, 23, 39, 55, 70],
            "draft_vocab_size": 154880,
            "target_hidden_size": None,
            "transformer_layer_config": {
                "num_hidden_layers": 5, "hidden_size": 6144,
                "vocab_size": 154880},
        }))
        draft = load_draft_candidate("spec.dspark", str(tmp_path))
        errs = draft_identity_errors(draft, self._foreign_shape(champion))
        joined = "; ".join(errs)
        assert len(errs) == 3
        assert "6144" in joined and "4096" in joined          # hidden
        assert "154880" in joined and "151552" in joined      # vocab
        assert "[55, 70]" in joined and "45" in joined        # aux taps

    def test_identity_fields_are_parsed_from_the_checkpoint(self, tmp_path):
        d = tmp_path / "spec.dspark"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "aux_hidden_state_layer_ids": [1, 2, 3],
            "draft_vocab_size": 100352,
            "target_hidden_size": 4096,
            "transformer_layer_config": {"hidden_size": 2048,
                                         "vocab_size": 100352},
        }))
        draft = load_draft_candidate("spec.dspark", str(tmp_path))
        assert draft.aux_layer_ids == (1, 2, 3)
        assert draft.draft_vocab == 100352       # top-level draft_vocab_size
        assert draft.target_hidden == 4096       # trained-against target
        assert draft.draft_hidden == 2048        # its own transformer

    def test_target_hidden_size_null_falls_back_to_draft_hidden(self, tmp_path,
                                                                champion):
        from autoconfig.solver import draft_identity_errors
        dpath = write_fake_draft(tmp_path)   # target_hidden absent -> 0
        draft = load_draft_candidate(os.path.basename(dpath), str(tmp_path))
        assert draft.target_hidden == 0
        # against the matching (6144) target: the fallback compares
        # draft_hidden itself
        shape = ModelShape.from_model_section(champion["model"])
        assert draft_identity_errors(draft, shape) == []
        errs = draft_identity_errors(draft, self._foreign_shape(champion))
        assert any("6144" in e and "4096" in e for e in errs)


from autoconfig import combo_constraints as cc  # noqa: E402


@pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                    reason="glm5_next reference config absent")
class TestComboConstraints:
    """TD-AUTOCONFIG-COMBO-CONSTRAINTS — the combination-keyed per-model
    registry (model_constraints.json, shipped DATA read at import).  First
    row: arch == glm5_next AND tensor_parallelism > 1 => dcp_kv_mode =
    replicated (the engine fail-closes sharded KV for the arch,
    engine.cpp:637 — a solver that does not know that derives a recipe that
    cannot boot, TD-AUTOCONFIG-GLM5NEXT-TP-SHARDED-KV).  A row is consulted
    only when its combination applies; deleting it moves the derivation
    with no code edit; a contradicting pin REFUSES quoting both."""

    def _glm5(self, tmp_path, pins=None, pinned_gib=10):
        with open(GLM5NEXT_CONFIG) as f:
            shape = ModelShape.from_model_section(json.load(f)["model"])
        proc, sysd = build_fake_tree(str(tmp_path / "t"))
        hw = detect_hardware(proc_root=proc, sys_root=sysd)
        sv = Solver(hw, shape, {"model": shape.raw}, Levers(0.02, 51200),
                    expert_slot_bytes=SLOT_BYTES,
                    non_expert_pinned_bytes=int(pinned_gib * (1 << 30)))
        if pins:
            sv = _with_pins(sv, pins)
        return sv

    # ---- the row fires only where its combination applies ------------------

    def test_tp2_derivation_forces_replicated_with_engine_gate_row(
            self, tmp_path):
        sv = self._glm5(tmp_path,
                        pins={"parallelism.tensor_parallelism": 2})
        res = sv.solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 2
        assert res.recipe["hardware"]["dcp_kv_mode"] == "replicated"
        e = res.explanations.get("hardware.dcp_kv_mode")
        assert e is not None
        # (b) its OWN kind: an engine-current fact, neither measured nor
        # heuristic, citing the enforcing engine site and the ticket
        assert e.kind == "engine_gate"
        assert "glm5next-no-sharded-kv" in e.refs
        assert any("engine.cpp:637" in r for r in e.refs)
        assert any("TD-AUTOCONFIG-GLM5NEXT-TP-SHARDED-KV" in r
                   for r in e.refs)
        assert "arch == glm5_next AND tensor_parallelism > 1" in e.because

    def test_tp1_derivation_never_consults_the_row(self, tmp_path):
        # the combination does not apply at tp=1: the dcp fields stay
        # absent and no engine_gate row is written — nothing over-constrained
        res = self._glm5(tmp_path).solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 1
        assert "dcp_kv_mode" not in res.recipe["hardware"]
        e = res.explanations.get("hardware.dcp_kv_mode")
        # the standing "omitted below dcp=2" row, not the engine gate
        assert e.kind != "engine_gate"
        assert "glm5next-no-sharded-kv" not in e.refs

    def test_capacity_escalated_tp_also_hits_the_gate(self, tmp_path):
        # unpinned: 26 GiB of pinned weights forces the E1 fit up to tp=2
        # (capacity, not speed) — the gate must catch the ESCALATED tp too
        sv = self._glm5(tmp_path, pinned_gib=26)
        res = sv.solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 2
        assert res.recipe["hardware"]["dcp_kv_mode"] == "replicated"
        assert res.explanations.get("hardware.dcp_kv_mode").kind == \
            "engine_gate"

    # ---- (a) rows stay DATA: deleting one moves the derivation -------------

    def test_deleting_the_row_moves_the_derivation(self, tmp_path,
                                                   monkeypatch):
        monkeypatch.setattr(cc, "COMBO_CONSTRAINTS",
                            tuple(r for r in cc.COMBO_CONSTRAINTS
                                  if r.id != "glm5next-no-sharded-kv"))
        sv = self._glm5(tmp_path,
                        pins={"parallelism.tensor_parallelism": 2})
        res = sv.solve()
        # the gate gone, the solver goes back to sharded — but the +4.6%
        # measurement is arch-scoped to glm_moe_dsa (symptom TD (c)), so
        # glm5_next takes it as an UNPRICED structural default
        assert res.recipe["hardware"]["dcp_kv_mode"] == "sharded"
        e = res.explanations.get("hardware.dcp_kv_mode")
        assert e.kind == "heuristic"
        assert "UNPRICED" in e.because

    # ---- (c)/(d) pins: refusal, never silent override ----------------------

    def test_sharded_pin_with_tp2_pin_refuses_quoting_both(self, tmp_path):
        sv = self._glm5(tmp_path,
                        pins={"parallelism.tensor_parallelism": 2,
                              "hardware.dcp_kv_mode": "sharded"})
        with pytest.raises(Infeasible) as ei:
            sv.solve()
        assert ei.value.constraint_id == "pinned-kv-mode-engine-gated"
        msg = str(ei.value)
        assert "glm5next-no-sharded-kv" in msg
        assert "hardware.dcp_kv_mode='sharded'" in msg
        assert "parallelism.tensor_parallelism=2" in msg
        assert "engine.cpp:637" in msg

    def test_sharded_pin_alone_adapts_around_by_dropping_the_gated_lane(
            self, tmp_path):
        # tp unpinned: the pin removes the tp=2 escalation lane (where the
        # gate fires) LOUDLY and the derivation proceeds at tp=1, where the
        # combination does not apply and the pinned field is inert
        sv = self._glm5(tmp_path,
                        pins={"hardware.dcp_kv_mode": "sharded"})
        res = sv.solve()
        assert res.recipe["parallelism"]["tensor_parallelism"] == 1
        assert res.recipe["hardware"]["dcp_kv_mode"] == "sharded"
        assert any("removes tp=2" in w and "glm5next-no-sharded-kv" in w
                   for w in res.warnings)

    def test_replicated_pin_agrees_with_the_gate_without_override_noise(
            self, tmp_path):
        # the pin AGREES with the engine gate: kind stays pinned and no
        # "OVERRIDES measured sharded-kv-beats-replicated" warning fires —
        # replicated is mandatory here, not an override
        sv = self._glm5(tmp_path,
                        pins={"parallelism.tensor_parallelism": 2,
                              "hardware.dcp_kv_mode": "replicated"})
        res = sv.solve()
        assert res.recipe["hardware"]["dcp_kv_mode"] == "replicated"
        assert res.explanations.get("hardware.dcp_kv_mode").kind == "pinned"
        assert not any("sharded-kv-beats-replicated" in w
                       for w in res.warnings)

    # ---- the champion arch is untouched (scoping + gate) -------------------

    def test_champion_keeps_measured_sharded(self, tmp_path, champion):
        res = make_solver(tmp_path, champion).solve()
        assert res.recipe["hardware"]["dcp_kv_mode"] == "sharded"
        e = res.explanations.get("hardware.dcp_kv_mode")
        assert e.kind == "measured"
        assert "sharded-kv-beats-replicated" in e.refs

    def test_measured_row_is_arch_scoped(self):
        # symptom TD (c): the +4.6% row can never price an arch it was not
        # measured on (or one that cannot shard at all)
        row = ec.get("sharded-kv-beats-replicated")
        assert row.scope == "glm_moe_dsa"
        assert "ARCH-SCOPED" in row.statement

    # ---- registry hygiene: shipped data, loud loader -----------------------

    def test_the_row_is_engine_current_fact_with_ticket_and_site(self):
        row = cc.get("glm5next-no-sharded-kv")
        assert row is not None
        assert row.kind == "engine_gate"
        assert row.ticket == "TD-AUTOCONFIG-GLM5NEXT-TP-SHARDED-KV"
        assert "engine.cpp:637" in row.site
        assert row.when_text() == \
            "arch == glm5_next AND tensor_parallelism > 1"
        assert row.forced("hardware.dcp_kv_mode") == "replicated"

    def _write(self, tmp_path, doc):
        p = tmp_path / "mc.json"
        p.write_text(json.dumps(doc))
        return str(p)

    def test_loader_refuses_unknown_predicate_key(self, tmp_path):
        # fail-closed: a predicate this build cannot evaluate must never be
        # silently skipped (it would un-fire a gate and emit a non-booting
        # recipe). A GPU predicate is a NEW PREDICATE_KEYS entry, not a
        # new mechanism.
        doc = {"rows": [{"id": "x", "kind": "engine_gate",
                         "when": {"gpu_class": "rtx5090"},
                         "force": {"a.b": 1}, "statement": "s"}]}
        with pytest.raises(ValueError, match="unknown predicate key"):
            cc._load(self._write(tmp_path, doc))

    def test_loader_refuses_unknown_operator(self, tmp_path):
        doc = {"rows": [{"id": "x", "kind": "engine_gate",
                         "when": {"tensor_parallelism": {"between": [1, 4]}},
                         "force": {"a.b": 1}, "statement": "s"}]}
        with pytest.raises(ValueError, match="unknown operator"):
            cc._load(self._write(tmp_path, doc))

    def test_loader_refuses_measured_kind(self, tmp_path):
        # a combination row is an engine-current fact — never measured,
        # never heuristic
        doc = {"rows": [{"id": "x", "kind": "measured",
                         "when": {"arch": "glm5_next"},
                         "force": {"a.b": 1}, "statement": "s"}]}
        with pytest.raises(ValueError, match="engine-current fact"):
            cc._load(self._write(tmp_path, doc))

    def test_match_refuses_a_ctx_missing_a_needed_predicate(self):
        row = cc.get("glm5next-no-sharded-kv")
        with pytest.raises(ValueError, match="needs predicate"):
            cc.matches(row, {"arch": "glm5_next"})

    def test_shipped_registry_loads_and_validates(self):
        # the module imported => the shipped JSON passed the loud loader;
        # re-load explicitly so a data edit that breaks validation fails
        # HERE and not at first derivation
        rows = cc._load()
        assert any(r.id == "glm5next-no-sharded-kv" for r in rows)

"""Long-context autoconfig derivations (P-30 / P-31).

Two tickets meet here and fund each other:

  TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH — at deep asks the allocations
  that live OUTSIDE the VramAllocator block (block tables, prefill
  staging, logits, KVT union staging, the MoE persistent set) stop being
  noise; the solver must size them from the ENGINE's own arithmetic and
  fund them by raising the TP-rank margin, or the boot dies at
  command_dispatcher.cpp:1363 with the recipe still claiming a fit.

  TD-AUTOCONFIG-STRIDE-NOT-DERIVED — the superchunk stride was a copied
  512; P-30 measured +54-62% fresh prefill at S=2048, so the stride is
  now the largest that fits the per-device-class MoE-big check.

The sizing pins below are the load-bearing part: they are transcriptions
of engine arithmetic, and a transcription is only worth what its
ground truth is worth (P-30 step 4 measured spill: 246 / 987 / 1975 MiB
at S = 512 / 2048 / 4096 on the GLM-5.3-Flash shape).  The derivation
tests then pin what the solver does with them on the standard fake box.
Hermetic: fake /proc+/sys, no engine, no GPU, no weights.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from test_autoconfig_detect import build_fake_tree

from autoconfig import sizing
from autoconfig.explain import Infeasible
from autoconfig.hwdetect import detect_hardware
from autoconfig.modelshape import ModelShape
from autoconfig.pins import Pin, PinSet
from autoconfig.solver import Levers, Solver

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
GLM53_RECIPE = os.path.join(REPO, "recipes", "glm53flash_serve.json")
GLM5NEXT_CONFIG = os.path.join(REPO, "test-data", "config",
                               "glm53_flash_gguf.json")

SLOT_BYTES = 27623424        # GLM-5.2 prepack manifest slot_size_bytes


def _shape(path):
    with open(path) as f:
        return ModelShape.from_model_section(json.load(f)["model"])


def _pins(d):
    return PinSet({p: Pin(p, v, "test") for p, v in d.items()})


def _glm5(tmp_path, name, ctx, pinned_gib, pins=None, levers=None):
    shape = _shape(GLM5NEXT_CONFIG)
    proc, sysd = build_fake_tree(str(tmp_path / name))
    hw = detect_hardware(proc_root=proc, sys_root=sysd)
    sv = Solver(hw, shape, {"model": shape.raw}, levers or Levers(0.02, ctx),
                expert_slot_bytes=SLOT_BYTES,
                non_expert_pinned_bytes=int(pinned_gib * (1 << 30)))
    if pins:
        sv.pins = _pins(pins)
    return sv


@pytest.mark.skipif(not os.path.exists(GLM53_RECIPE),
                    reason="glm53flash reference recipe absent")
class TestSizingGroundTruths:
    """The two P-31 transcriptions, against measured engine numbers.

    A sizing function that drifts from the engine it transcribes is worse
    than no function at all: the recipe would promise a fit the boot
    cannot honour.  Both pins are on the REAL GLM-5.3-Flash shape (the
    shipped serving recipe's model section), so a geometry change moves
    them and a transcription bug fails them."""

    def _shape(self):
        return _shape(GLM53_RECIPE)

    def test_block_tables_at_one_million_tokens(self):
        """dev_block_tables = (layers + nextn) x B x ceil(max_seq/page) x 4 B
        (command_dispatcher.cpp:1359) — 1472 MiB EXACTLY at 1M/B=128, the
        term that forces the schedule width down from 512."""
        s = self._shape()
        assert sizing.block_tables_bytes(s, 1048576, 128, 16) \
            == 1472 * sizing.MIB
        # and it is linear in B: the halvings the solver walks are exact
        assert sizing.block_tables_bytes(s, 1048576, 512, 16) \
            == 4 * 1472 * sizing.MIB

    def test_moe_big_transient_matches_the_measured_spill(self):
        """MoE-big transient at tp=2 with GGUF weights, against the P-30
        step 4 measured spill (246 / 987 / 1975 MiB at S = 512 / 2048 /
        4096) — the term the derived stride is bought with."""
        s = self._shape()
        def mib(chunk):
            return sizing.moe_big_transient_bytes(
                s, chunk, 2, True) / sizing.MIB
        assert mib(512) == pytest.approx(246.90894317626953, rel=1e-12)
        assert mib(2048) == pytest.approx(987.6291580200195, rel=1e-12)
        assert mib(4096) == pytest.approx(1975.2561111450195, rel=1e-12)
        # the measured spill anchors, to the MiB the engine reported
        for chunk, measured in ((512, 246), (2048, 987), (4096, 1975)):
            assert int(mib(chunk)) == measured


@pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                    reason="glm5_next reference config absent")
class TestMillionTokenDerivation:
    """The whole P-31 long-context surface at ONE point: 2 x 1M pinned on
    the standard 4-GPU fake box (2 x 5090 attention + 2 x 5080 expert
    hosts).  Every field below is a DERIVATION that did not exist before
    P-31; pinning them together is what proves they fund each other (the
    stride needs margin, the margin needs the block-table reduction, the
    ask needs tiering)."""

    def _res(self, tmp_path):
        sv = _glm5(tmp_path, "m", 2 * 1048576, 7,
                   pins={"serving.max_sequence_length": 1048576,
                         "serving.max_concurrent_requests": 2})
        return sv, sv.solve()

    def test_the_ask_is_served_whole(self, tmp_path):
        sv, res = self._res(tmp_path)
        r = res.recipe
        assert r["serving"]["max_sequence_length"] == 1048576
        assert r["serving"]["max_concurrent_requests"] == 2
        assert res.degradations == []
        # the head-divisible ceiling is ordered FIRST at >= 262144: at this
        # depth attention is the measured majority of the prefill wall
        assert r["parallelism"]["tensor_parallelism"] == 2
        e = res.explanations.get("parallelism.tensor_parallelism")
        assert e is not None and "ordered FIRST" in e.because
        # and the ask only fits because tiering escalated
        assert r["memory"]["kv_tiering"]["enabled"] is True
        e = res.explanations.get("memory.kv_tiering.enabled")
        assert e is not None and e.kind == "searched"
        assert "conc-ask-exceeds-untiered-pool" in e.because

    def test_the_schedule_width_is_reduced_to_fit_the_block_tables(
            self, tmp_path):
        sv, res = self._res(tmp_path)
        assert res.recipe["orchestrator"]["max_batch_size"] == 128
        e = res.explanations.get("orchestrator.max_batch_size")
        assert e is not None and e.kind == "closed_form"
        assert "REDUCED from the family template's 512" in e.because
        assert "TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH" in e.refs

    def test_the_stride_is_derived_and_the_two_moe_knobs_move_together(
            self, tmp_path):
        sv, res = self._res(tmp_path)
        c = res.recipe["compute"]
        sc = c["prefill_superchunk_tokens"]
        assert sc >= 512 and sc % 512 == 0
        assert c["moe_big_chunk_tokens"] == sc   # EP beyond the TP group
        e = res.explanations.get("compute.prefill_superchunk_tokens")
        assert e is not None and e.kind == "searched"
        assert "TD-AUTOCONFIG-STRIDE-NOT-DERIVED" in e.refs
        # at this depth the post-check allocations no longer fit the
        # engine's 1024 MiB default headroom, so it is raised and said
        assert c["moe_big_fit_headroom_mb"] > 1024
        e = res.explanations.get("compute.moe_big_fit_headroom_mb")
        assert e is not None and e.kind == "closed_form"
        assert "command_dispatcher.cpp:1363" in e.because

    def test_only_the_tp_ranks_pay_the_raised_margin(self, tmp_path):
        """The runtime scratch lives on the ATTENTION hosts; charging the
        expert-only devices for it would evict experts to fund nothing."""
        sv, res = self._res(tmp_path)
        gpus = res.recipe["hardware"]["gpus"]
        tp = res.recipe["parallelism"]["tensor_parallelism"]
        for i, g in enumerate(gpus[:tp]):
            m = g["vram_allocation_gb"]["safety_margin_gb"]
            assert m > 2.25, g["id"]
            e = res.explanations.get(
                f"hardware.gpus[{i}].vram_allocation_gb.safety_margin_gb")
            assert e is not None and e.kind == "closed_form"
            assert "TD-AUTOCONFIG-LONGCTX-RUNTIME-SCRATCH" in e.refs
            assert "dev_block_tables" in e.because
        for i, g in enumerate(gpus[tp:], start=tp):
            assert "safety_margin_gb" not in g["vram_allocation_gb"], g["id"]
            assert res.explanations.get(
                f"hardware.gpus[{i}].vram_allocation_gb.safety_margin_gb") \
                is None
            # the expert hosts keep doing their job
            assert g["vram_allocation_gb"]["expert_streaming"] > 0, g["id"]

    def test_the_derivation_is_deterministic(self, tmp_path):
        a = self._res(tmp_path)[1].recipe
        b = _glm5(tmp_path, "m2", 2 * 1048576, 7,
                  pins={"serving.max_sequence_length": 1048576,
                        "serving.max_concurrent_requests": 2}).solve().recipe
        assert a == b


@pytest.mark.skipif(not os.path.exists(GLM5NEXT_CONFIG),
                    reason="glm5_next reference config absent")
class TestConcAdmissionEscalation:
    """P-31: the ask is CONC SIMULTANEOUS max-length requests (§2.2's own
    definition), so the untiered kMain pool must hold their SUM — a pool
    that holds one is a fit the engine refuses at admission time, hundreds
    of thousands of tokens into the second prefill.  The check raises
    `conc-ask-exceeds-untiered-pool`; KV tiering (INV-KVT-16 windowed
    admission) is the capacity escalation that answers it.

    The never-refuse property must survive the new axis: with tiering
    PINNED off the same box degrades the ask instead of refusing it."""

    ASK = 2 * 409600      # 2 x 409600, deep enough to force the ceiling

    def test_the_escalation_keeps_the_whole_ask(self, tmp_path):
        res = _glm5(tmp_path, "e", self.ASK, 20).solve()
        r = res.recipe
        assert r["serving"]["max_concurrent_requests"] == 2
        assert r["serving"]["max_sequence_length"] == 409600
        assert res.degradations == []
        assert r["memory"]["kv_tiering"]["enabled"] is True
        assert r["memory"]["kv_tiering"]["tiered_prefill"] is True
        e = res.explanations.get("memory.kv_tiering.enabled")
        assert e is not None and e.kind == "searched"
        assert "conc-ask-exceeds-untiered-pool" in e.because
        assert "ESCALATED from the measured glm5_next OFF default for " \
               "CAPACITY" in e.because
        assert "glm5next-tiering-default-off" in e.refs

    def test_pinning_tiering_off_degrades_but_never_refuses(self, tmp_path):
        """The pin removes the axis (design §3), so the same ask must be
        SHED — a preference-independent degradation, never an Infeasible."""
        res = _glm5(tmp_path, "p", self.ASK, 20,
                    pins={"memory.kv_tiering.enabled": False}).solve()
        r = res.recipe
        assert r["memory"]["kv_tiering"]["enabled"] is False
        assert res.degradations, "the ask must be paid for somewhere"
        assert (r["serving"]["max_concurrent_requests"]
                * r["serving"]["max_sequence_length"]) < self.ASK
        assert r["serving"]["max_sequence_length"] >= 51200

    def test_the_untiered_default_is_kept_and_explained_below_the_bound(
            self, tmp_path):
        """The escalation is not a new default: while the untiered pool
        holds the ask, the measured OFF row still owns the decision and
        says the escalation was not needed."""
        res = _glm5(tmp_path, "d", 2 * 51200, 10).solve()
        assert res.recipe["memory"]["kv_tiering"]["enabled"] is False
        e = res.explanations.get("memory.kv_tiering.enabled")
        assert e is not None and e.kind == "measured"
        assert "the capacity escalation was not needed" in e.because
        assert "glm5next-tiering-default-off" in e.refs

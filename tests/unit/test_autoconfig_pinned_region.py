"""The solver's model of the engine, held against the engine's own boot log.

TD-AUTOCONFIG-PINNED-BYTES-UPPER-BOUND: the solver summed the checkpoint's
STORED non-expert bytes (9384.7 MiB) while the engine reserved a slot-by-slot
REGION (16073.0 MiB) — a 41.6% miss that made the default lever path derive an
unbootable recipe. Its sibling tickets say the fix is a boot-log cross-check,
not a patched constant, so these tests pin BOTH: the transcription, and the
comparator that keeps it honest.

Two arms of the same model are asserted, because two engines exist:
  * upper-bound arm  = the pre-GF3.15 engine, whose figure is in the committed
                       boot log (16073.0 MiB). Needs no checkpoint.
  * checkpoint arm   = the post-GF3.15 engine, which sizes glm5_next+GGUF
                       attention from the file's real k-quant widths
                       (10736.3 MiB). Needs the GGUF; skipped without it.
Asserting only one would let the other rot.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from autoconfig import enginecheck, gguf_meta, sizing
from autoconfig.modelshape import ModelShape

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RECIPE = os.path.join(REPO, "scratchpad", "gf3_ladder",
                      "glm53-ctx100k.autoconfig.json")
BOOT_LOG = os.path.join(REPO, "scratchpad", "gf3_ladder",
                        "glm53-ctx100k.bench.benchmark-logs",
                        "benchmark_serve.log")
MIB = 1 << 20

# The engine's figure in that committed log (pre-GF3.15 boot).
LOG_PINNED_MIB = 16073.0
# What a post-GF3.15 engine reserves for the same recipe: the same plan with
# the attention slots sized from the checkpoint's real Q8_0 widths.
CHECKPOINT_PINNED_MIB = 10736.3


def _recipe():
    if not os.path.exists(RECIPE):
        pytest.skip(f"committed recipe {RECIPE} absent")
    with open(RECIPE) as f:
        return json.load(f)


def _weights(recipe):
    p = recipe["model"]["weights_path"]
    if not os.path.exists(p):
        pytest.skip(f"checkpoint {p} not on this box")
    return p


def _layout(arm, with_weights):
    r = _recipe()
    mem = r["memory"]
    return gguf_meta.pinned_region_layout(
        r["model"], r["quantization"],
        r["parallelism"]["tensor_parallelism"],
        _weights(r) if with_weights else "",
        mem["pinned_layers"], mem["tp_mode_per_layer"], arm)


# ─────────────────────────────────────────────── the modelled pinned region

class TestPinnedRegion:
    def test_upper_bound_arm_reproduces_the_committed_boot_log(self):
        """The pre-GF3.15 arm must land on the engine's own printed figure.

        This is the arm the committed log was produced by, so equality here is
        a real assertion about the transcription — not a constant chosen to
        make a test pass. Needs no checkpoint: the bound is config-only."""
        lay = _layout("upper_bound", with_weights=False)
        assert lay.total_bytes / MIB == pytest.approx(LOG_PINNED_MIB, abs=0.05)
        if os.path.exists(BOOT_LOG):
            figures = enginecheck.read_boot_log(BOOT_LOG)
            assert figures.tp_pinned_mib == pytest.approx(
                lay.total_bytes / MIB, abs=1.0)

    def test_upper_bound_arm_declares_itself_a_bound(self):
        lay = _layout("upper_bound", with_weights=False)
        assert not lay.exact
        assert lay.direction == "over"
        assert "upper bound" in lay.provenance

    def test_checkpoint_arm_sizes_attention_from_the_real_widths(self):
        lay = _layout("auto", with_weights=True)
        assert lay.total_bytes / MIB == pytest.approx(CHECKPOINT_PINNED_MIB,
                                                      abs=0.05)
        assert lay.exact and lay.direction == "exact"

    def test_the_two_arms_differ_only_in_attention(self):
        """The GF3.15 fix touches the attention slots and nothing else — if a
        second component moves, one of the two arms is wrong."""
        ub = _layout("upper_bound", with_weights=False).by_component
        ck = _layout("auto", with_weights=True).by_component
        assert set(ub) == set(ck)
        moved = {k for k in ub if ub[k] != ck[k]}
        assert moved == {"attention"}
        # ~6.5 GiB on this model: 34 KDA layers' [8192,4096] projections.
        assert (ub["attention"] - ck["attention"]) / MIB == pytest.approx(
            5336.7, abs=1.0)

    def test_stored_bytes_are_NOT_the_region(self):
        """The bug in one line: what the checkpoint stores is far below what
        the engine reserves, in both arms."""
        r = _recipe()
        known, unknown = gguf_meta.non_expert_bytes(_weights(r))
        stored_mib = (known + unknown) / MIB
        assert stored_mib == pytest.approx(9384.7, abs=1.0)
        assert stored_mib < CHECKPOINT_PINNED_MIB < LOG_PINNED_MIB

    def test_an_unmodelled_architecture_refuses_rather_than_guesses(self):
        r = _recipe()
        v4 = dict(r["model"], architecture="deepseek_v4")
        with pytest.raises(gguf_meta.UnmodelledArchitecture):
            gguf_meta.pinned_region_layout(v4, r["quantization"], 1)

    def test_an_unknown_weight_quant_refuses_rather_than_guesses(self):
        with pytest.raises(gguf_meta.UnmodelledArchitecture):
            gguf_meta.weight_bytes_per_element("int8_hypothetical")


# ──────────────────────────────────────────────────── the boot-log parser

# One boot's worth of the lines this module reads, for the committed ctx100k
# recipe (glm5_next, snapmla, tp=1, max_sequence_length 49984). It carries the
# admissible-ceiling line too, because a fixture that OMITS a line teaches the
# cross-check nothing: the two admissibility rows would report UNCHECKED and
# every "all rows agree" assertion would pass vacuously — the exact failure
# mode enginecheck's honesty rule exists to prevent.
FIXTURE_LOG = """\
[info] GGUF generic weights: built mixed expert QuantInterface (gate=gguf_q5_k)
[info] VramAllocator: S1 slab geometry — slab 272448 B = 33 kMain pages x 8256 B (holds one 272384 B indexer page, sliver 64 B/slab)
[info] KDA state pool on GPU 0: MAPPED over the shared slab region (no dedicated carve; TD-KDA-STATE-MAPPED-SLABS). Per-request demand = 34 linear layers x 17 slabs (4.28 MiB/layer unit, 150.2 MiB/request incl. slab padding vs 145.6 MiB slot); claimed at seq_create.
[info] VramAllocator: GPU 0 (hw id 2, TP) budget: total 30720.0 MiB = pinned 16073.0 + margin 2304.0 + indexer_k 0.0 + kv[main 2559.2 spec 88.5] + expert 8704.0
[info] VramAllocator: GPU 0 admissible context (single request) = 443552 tokens — max_sequence_length 49984 needs 59334 kMain pages (KV 37488 + indexer 2772 + KDA state 19074) vs pool 373527 pages
[info] VramAllocator: GPU 1 (hw id 3, expert-only) budget: total 30720.0 MiB = pinned 0.0 + margin 2304.0 + expert 27136.0
[info] VramAllocator GPU 2: 30720 MiB total, 28416 MiB allocated (safety margin 2304 MiB)
[info]   pinned weights    16073.0 MiB  | kv_main         2559.2 MiB (260007 pages)
[info]   kda_state (GF3.8)     0.0 MiB (0 slots x 145.6 MiB — per-REQUEST recurrent+conv state)
[info] VramAllocator GPU 3: 30720 MiB total, 28416 MiB allocated (safety margin 2304 MiB)
[info]   pinned weights        0.0 MiB  | kv_main            0.0 MiB (0 pages)
"""

TQ_SLAB_LINE = ("[info] VramAllocator: S1 slab geometry — slab 272448 B = "
                "66 kMain pages x 4128 B (holds one 272384 B indexer page)\n")


class TestBootLogParser:
    def test_reads_every_checked_figure(self):
        f = enginecheck.parse_boot_log(FIXTURE_LOG, "fixture")
        assert f.tp_pinned_mib == 16073.0
        assert [b.index for b in f.budgets] == [0, 1]
        assert [b.role for b in f.budgets] == ["TP", "expert-only"]
        assert f.budgets[0].hw_id == 2
        assert f.summary_pinned_mib == {2: 16073.0, 3: 0.0}
        assert (f.kv_bytes_per_page, f.slab_pages, f.slab_bytes) == \
            (8256, 33, 272448)
        assert f.kda_slot_mib == 145.6
        assert f.kda_request_mib == 150.2
        assert (f.kda_linear_layers, f.kda_slabs_per_layer) == (34, 17)
        # TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA: the ceiling AND its three
        # component terms, because a ceiling alone cannot say WHICH tenant
        # bound it — the GF3 1M arm was misread as state-only for exactly
        # that reason.
        assert f.admissible_ctx_tokens == 443552
        assert f.admission_demand_pages == 59334
        assert f.admission_kv_pages == 37488
        assert f.admission_indexer_pages == 2772
        assert f.admission_state_pages == 19074
        assert f.admission_pool_pages == 373527
        assert (f.admission_kv_pages + f.admission_indexer_pages
                + f.admission_state_pages) == f.admission_demand_pages

    def test_unreported_figures_stay_None_not_zero(self):
        """A model with no DSA prints no slab line — the difference between
        'the engine said 0' and 'the engine said nothing' is the whole point."""
        f = enginecheck.parse_boot_log(
            "[info] VramAllocator: GPU 0 (hw id 0, TP) budget: total 30720.0 "
            "MiB = pinned 1024.0 + margin 2304.0\n")
        assert f.tp_pinned_mib == 1024.0
        assert f.kv_bytes_per_page is None
        assert f.kda_slot_mib is None
        # an engine too old to print its admissible ceiling must leave the
        # rows UNCHECKED, never let 0 read as "no context is admissible"
        assert f.admissible_ctx_tokens is None
        assert f.admission_demand_pages is None
        assert f.admission_kv_pages is None
        assert f.admission_indexer_pages is None
        assert f.admission_state_pages is None
        assert f.admission_pool_pages is None

    def test_reads_the_committed_boot_log(self):
        if not os.path.exists(BOOT_LOG):
            pytest.skip("committed boot log absent")
        f = enginecheck.read_boot_log(BOOT_LOG)
        assert f.tp_pinned_mib == LOG_PINNED_MIB
        assert f.kv_bytes_per_page == 8256      # snapmla: 516 B/token x 16
        assert f.kda_slot_mib == 145.6


# ──────────────────────────────────────────────────── the drift comparator

class TestCrossCheck:
    def _ctx(self, recipe=None):
        r = recipe or _recipe()
        return enginecheck.CheckContext(r, r["model"]["weights_path"])

    def test_the_rows_that_agree_today_are_locked_in(self):
        """KV row bytes, KDA slot and KDA mapped demand all reproduce the
        engine exactly on the committed log — that is the state this test
        exists to defend, now that the attention backend is flippable again."""
        f = enginecheck.parse_boot_log(FIXTURE_LOG, "fixture")
        by = {r.row.name: r for r in enginecheck.cross_check(f, self._ctx())}
        for name in ("KV bytes per page", "KDA state slot",
                     "KDA mapped demand per request"):
            assert by[name].status == "agree", by[name].describe()

    def test_kv_row_bytes_track_the_attention_backend(self):
        """turboquant_mla 258 B/token vs snapmla 516 is a 2x disagreement the
        solver must not be able to make silently (TD-GLM5-TQ-BACKEND-UNWIRED
        deleted the row that pinned the backend)."""
        r = _recipe()
        tq = json.loads(json.dumps(r))
        tq["compute"]["attention_backend"] = "turboquant_mla"
        tq_log = enginecheck.parse_boot_log(
            FIXTURE_LOG.replace(
                "slab 272448 B = 33 kMain pages x 8256 B",
                "slab 272448 B = 66 kMain pages x 4128 B"), "tq")
        by = {c.row.name: c for c in enginecheck.cross_check(
            tq_log, enginecheck.CheckContext(tq, ""))}
        assert by["KV bytes per page"].status == "agree"
        assert by["KV bytes per page"].engine == 4128

        # ... and the snapmla solver model against a TQ engine must DIVERGE.
        by = {c.row.name: c for c in enginecheck.cross_check(
            tq_log, enginecheck.CheckContext(r, ""))}
        assert by["KV bytes per page"].status == "diverge"
        assert by["KV bytes per page"].fatal

    def test_agreement_passes_and_says_nothing_alarming(self):
        f = enginecheck.parse_boot_log(FIXTURE_LOG, "fixture")
        r = _recipe()
        _weights(r)   # the pinned row needs the checkpoint to be exact
        results = enginecheck.cross_check(f, self._ctx(r), (
            enginecheck.CHECKS[1], enginecheck.CHECKS[2], enginecheck.CHECKS[3]))
        assert all(c.status == "agree" for c in results)
        assert not any(c.fatal for c in results)

    def test_a_500_mib_divergence_fails_with_both_numbers(self):
        r = _recipe()
        _weights(r)
        shifted = FIXTURE_LOG.replace("pinned 16073.0",
                                      f"pinned {CHECKPOINT_PINNED_MIB + 500:.1f}")
        f = enginecheck.parse_boot_log(shifted, "fixture")
        res = enginecheck.cross_check(f, self._ctx(r), (enginecheck.CHECKS[0],))
        assert res[0].status == "diverge" and res[0].fatal
        msg = res[0].describe()
        assert f"{CHECKPOINT_PINNED_MIB:g}" in msg
        assert f"{CHECKPOINT_PINNED_MIB + 500:g}" in msg
        assert "500" in msg                       # the delta
        assert "gguf_meta.pinned_region_layout" in msg    # the solver site
        assert "compute_pinned_layout" in msg             # the engine site

    def test_check_boot_log_raises_on_drift(self, tmp_path):
        r = _recipe()
        _weights(r)
        log = tmp_path / "boot.log"
        log.write_text(FIXTURE_LOG.replace(
            "pinned 16073.0", f"pinned {CHECKPOINT_PINNED_MIB + 500:.1f}"))
        with pytest.raises(enginecheck.EngineModelDrift) as e:
            enginecheck.check_boot_log(str(log), r, r["model"]["weights_path"],
                                       log=lambda *_a: None)
        assert "pinned region" in str(e.value)

    def test_check_boot_log_passes_when_the_model_is_right(self, tmp_path):
        r = _recipe()
        _weights(r)
        log = tmp_path / "boot.log"
        log.write_text(FIXTURE_LOG.replace(
            "pinned 16073.0", f"pinned {CHECKPOINT_PINNED_MIB:.1f}"))
        out = enginecheck.check_boot_log(str(log), r,
                                         r["model"]["weights_path"],
                                         log=lambda *_a: None)
        assert all(c.status == "agree" for c in out)

    def test_an_inadmissible_advertisement_is_fatal(self, tmp_path):
        """The row the ticket exists for. The GF3 1M arm advertised a context
        its own admission path refused; whatever the upstream cause (pinned
        drift, a KV-row flip, a new pool tenant), it lands HERE as a violated
        bound — advertised 49984 against a 42000-token ceiling — and a
        violated bound is a hard failure, not a warning."""
        r = _recipe()
        _weights(r)
        short = FIXTURE_LOG.replace("= 443552 tokens", "= 42000 tokens")
        f = enginecheck.parse_boot_log(short, "fixture")
        by = {c.row.name: c for c in enginecheck.cross_check(f, self._ctx(r))}
        row = by["advertised max_sequence_length vs admissible ceiling"]
        assert row.status == "diverge"
        assert row.fatal
        assert row.solver.value == 49984 and row.engine == 42000

        log = tmp_path / "boot.log"
        log.write_text(short.replace(
            "pinned 16073.0", f"pinned {CHECKPOINT_PINNED_MIB:.1f}"))
        with pytest.raises(enginecheck.EngineModelDrift) as e:
            enginecheck.check_boot_log(str(log), r, r["model"]["weights_path"],
                                       log=lambda *_a: None)
        assert "advertised max_sequence_length" in str(e.value)

    def test_headroom_under_the_ceiling_is_agreement_not_a_warning(self):
        """The inequality row's other half: an advertisement far BELOW the
        ceiling is the healthy state. An estimate row would flag the gap as
        drift; this row must not, or every well-sized recipe would nag."""
        f = enginecheck.parse_boot_log(FIXTURE_LOG, "fixture")
        by = {c.row.name: c for c in enginecheck.cross_check(f, self._ctx())}
        row = by["advertised max_sequence_length vs admissible ceiling"]
        assert row.status == "agree" and not row.fatal
        assert row.solver.value < row.engine        # 49984 << 443552

    def test_a_wrong_admission_demand_is_fatal(self):
        """The demand row is pure geometry — no VRAM model, no policy — so
        any disagreement is a transcription bug in the mirror and must fail
        hard rather than warn."""
        r = _recipe()
        _weights(r)
        shifted = FIXTURE_LOG.replace("needs 59334 kMain pages",
                                      "needs 59434 kMain pages")
        f = enginecheck.parse_boot_log(shifted, "fixture")
        by = {c.row.name: c for c in enginecheck.cross_check(f, self._ctx(r))}
        row = by["single-request admission demand at max_seq"]
        assert row.status == "diverge" and row.fatal
        assert row.solver.value == 59334 and row.engine == 59434
        assert "sizing.single_request_demand_pages" in row.describe()

    def test_a_log_without_the_line_reports_UNCHECKED_not_agreement(self,
                                                                    tmp_path):
        """An engine too old to print its ceiling must leave BOTH rows
        unchecked — and check_boot_log must not raise on them. Silence is the
        one thing that must never read as agreement (enginecheck's honesty
        rule), and it must not read as failure either."""
        r = _recipe()
        _weights(r)
        older = "\n".join(l for l in FIXTURE_LOG.splitlines()
                          if "admissible context" not in l) + "\n"
        older = older.replace("pinned 16073.0",
                              f"pinned {CHECKPOINT_PINNED_MIB:.1f}")
        log = tmp_path / "boot.log"
        log.write_text(older)
        out = enginecheck.check_boot_log(str(log), r,
                                         r["model"]["weights_path"],
                                         log=lambda *_a: None)
        by = {c.row.name: c for c in out}
        for name in ("single-request admission demand at max_seq",
                     "advertised max_sequence_length vs admissible ceiling"):
            assert by[name].status == "unchecked", by[name].describe()
            assert not by[name].fatal
            assert "not agreement" in by[name].describe()

    def test_an_unmodelled_row_reports_UNCHECKED_not_agreement(self):
        r = _recipe()
        v4 = json.loads(json.dumps(r))
        v4["model"]["architecture"] = "deepseek_v4"
        f = enginecheck.parse_boot_log(FIXTURE_LOG, "fixture")
        res = enginecheck.cross_check(f, enginecheck.CheckContext(v4, ""),
                                      (enginecheck.CHECKS[0],))
        assert res[0].status == "unchecked"
        assert not res[0].fatal
        assert "not modelled" in res[0].describe()
        assert "not agreement" in res[0].describe()


# ─────────────────────────────────────────────── the mapped-KDA sizing row

class TestKdaMappedDemand:
    def test_slab_padded_demand_matches_the_engine(self):
        """`34 linear layers x 17 slabs ... 150.2 MiB/request incl. slab
        padding vs 145.6 MiB slot` — the 4.6 MiB the slot figure misses is
        exactly TD-AUTOCONFIG-MAXSEQ-IGNORES-MAPPED-KDA's gap."""
        r = _recipe()
        s = ModelShape.from_model_section(r["model"])
        f = enginecheck.parse_boot_log(FIXTURE_LOG, "fixture")
        geo = sizing.slab_geometry(
            s, r["quantization"]["kv_cache"], r["compute"]["attention_backend"],
            r["memory"]["kv_cache"]["page_size_tokens"],
            r["memory"]["kv_cache"]["indexer_k_page_size_tokens"])
        assert geo.slab_bytes == f.slab_bytes
        demand = sizing.kda_mapped_demand_bytes(s, 1, geo.slab_bytes)
        assert demand / MIB == pytest.approx(f.kda_request_mib, abs=0.05)
        assert demand == (f.kda_linear_layers * f.kda_slabs_per_layer
                          * f.slab_bytes)
        assert sizing.kda_slot_bytes(s, 1) < demand


# ─────────────────────────────── the GF3 1M arm, reduced to its arithmetic

class TestSingleRequestAdmissibility:
    """TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA pinned as numbers, from the
    committed ladder logs.

    The 1M arm advertised max_sequence_length 500000 and REFUSED the request
    that reached 474880+1024 tokens. The ticket read that refusal as a KDA
    state shortfall — 75 slabs the pool could not find. The mirror says the
    state is only one tenant of three: since the mapped default, KV pages,
    the FULL-LENGTH indexer-K reservation and the state all draw on the ONE
    kMain pool, and the real deficit is an order of magnitude larger than the
    state-only reading. Getting that wrong is how a solver advertises a
    context its own engine refuses, so these numbers are pinned.

    Geometry is the arm's own (snapmla, tp=1): 16-token/8256 B kMain pages,
    8192-token indexer pages, 33 kMain pages per S1 slab, 12 kMain-bearing
    layers, and 34 linear layers x 17 slabs x 33 = 19074 pages of mapped
    state."""

    POOL_PAGES = 373527          # the arm's kMain pool, boot-log verified
    GEO = dict(page_size_tokens=16, kv_pool_layers=12, dsa_computing_layers=12,
               indexer_k_page_size_tokens=8192, pages_per_slab=33,
               state_pages=19074)

    def _demand(self, tokens):
        return sizing.single_request_demand_pages(tokens, **self.GEO)

    def test_the_refused_request_needed_more_than_the_pool_holds(self):
        assert self._demand(475904) == 399366          # 474880 + 1024
        assert self._demand(475904) > self.POOL_PAGES

    def test_the_deficit_is_not_the_state_shortfall_the_ticket_saw(self):
        """75 slabs of missing state is 2475 kMain pages; the pool was short
        25839. The gap between those two numbers IS the ticket: the indexer
        reservation rides the same pool and nobody was charging it."""
        deficit = self._demand(475904) - self.POOL_PAGES
        assert deficit == 25839
        assert deficit > 75 * self.GEO["pages_per_slab"] * 10

    def test_the_rung_that_served_fits(self):
        """The ladder's next rung down went through — the mirror has to agree
        with the engine on the SERVING side too, or it would refuse contexts
        the box can hold."""
        assert self._demand(427008) == 360318          # 425984 + 1024
        assert self._demand(427008) <= self.POOL_PAGES

    def test_the_ceiling_is_the_largest_context_that_fits(self):
        ceiling = sizing.admissible_context_tokens(self.POOL_PAGES, **self.GEO)
        assert ceiling == 443552
        # exact at the boundary, in both directions: one token more does not
        # fit, so the ceiling is the ceiling and not a rounded-down guess
        assert self._demand(ceiling) <= self.POOL_PAGES
        assert self._demand(ceiling + 1) > self.POOL_PAGES

    def test_the_three_tenants_sum_to_the_demand(self):
        """The decomposition the boot line prints, so a future reader can
        tell WHICH tenant bound the ceiling — the thing the ticket could not
        tell from a single number."""
        tokens = 475904
        kv = sizing.single_request_demand_pages(
            tokens, self.GEO["page_size_tokens"], self.GEO["kv_pool_layers"],
            0, 0, 0, 0)
        idx = sizing.single_request_demand_pages(
            tokens, self.GEO["page_size_tokens"], self.GEO["kv_pool_layers"],
            self.GEO["dsa_computing_layers"],
            self.GEO["indexer_k_page_size_tokens"],
            self.GEO["pages_per_slab"], 0) - kv
        assert kv + idx + self.GEO["state_pages"] == self._demand(tokens)
        # and the indexer term is no rounding error next to the state
        assert idx > self.GEO["state_pages"]

"""TD-AUTOCONFIG-COMPARE-CLASSES — the acceptance instrument's classifier.

`unexplained` used to conflate three different things: a real solver gap,
the REFERENCE omitting a field the derived recipe states, and a
family-template copy the reference hand-deviates from.  It also could not
recurse into arrays whose LENGTHS differ (1 GPU vs 4), so a
wholly-justified shape change read as one opaque unexplained leaf.  These
tests pin the split and the shape-change handling.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from autoconfig.compare import (CLASS_ORDER, Divergence, class_counts,  # noqa: E402
                                compare, render_report, summary_line)
from autoconfig.explain import ExplanationSet  # noqa: E402
from autoconfig import templates  # noqa: E402


def _by_path(divs):
    return {d.path: d for d in divs}


# ---------------------------------------------------------------- classes

class TestClasses:
    def test_reference_omits_is_not_unexplained(self):
        """The derived recipe STATES what a hand recipe inherits as a
        schema default — that describes the reference, not a solver bug."""
        ref = {"memory": {}}
        der = {"memory": {"vram_safety_margin_gb": 2.25}}
        divs = compare(ref, der, ExplanationSet())
        d = _by_path(divs)["memory.vram_safety_margin_gb"]
        assert d.klass == "reference-omits"
        assert "schema default" in d.why

    def test_reference_omits_quotes_the_explanation_when_one_exists(self):
        ex = ExplanationSet()
        ex.add("memory.vram_safety_margin_gb", 2.25, "heuristic",
               "champion-proven margin")
        divs = compare({"memory": {}},
                       {"memory": {"vram_safety_margin_gb": 2.25}}, ex)
        d = _by_path(divs)["memory.vram_safety_margin_gb"]
        assert d.klass == "reference-omits"
        assert "champion-proven margin" in d.why

    def test_explained_template_copy_classifies_template_copy(self):
        """A kind=template Explanation row is an explanation — 'no
        explanation covers this path' was a lie for these rows."""
        ex = ExplanationSet()
        ex.add("orchestrator.max_batch_size", 512, "template",
               "family serving template", refs=("recipes/x.json",))
        divs = compare({"orchestrator": {"max_batch_size": 64}},
                       {"orchestrator": {"max_batch_size": 512}}, ex)
        d = _by_path(divs)["orchestrator.max_batch_size"]
        assert d.klass == "template-copy"
        assert "family serving template" in d.why

    def test_carried_rows_classify_template_copy(self):
        ex = ExplanationSet()
        ex.add("serving.tool_call_parser", "glm47", "carried",
               "carried from the base recipe")
        divs = compare({"serving": {"tool_call_parser": "other"}},
                       {"serving": {"tool_call_parser": "glm47"}}, ex)
        assert _by_path(divs)["serving.tool_call_parser"].klass == "template-copy"

    def test_bare_template_data_copy_with_no_row_classifies_template_copy(self):
        """Solver._assemble copies COMMON sections wholesale without
        per-leaf rows; a reference deviating there is the reference's
        hand-tuning, not a solver gap."""
        tv = templates.template_view("glm5_next")
        want = tv["prefetch"]["fusion_weights"]["prescope_alpha"]
        ref = {"prefetch": {"fusion_weights": {"prescope_alpha": 0.9}}}
        der = {"prefetch": {"fusion_weights": {"prescope_alpha": want}}}
        divs = compare(ref, der, ExplanationSet(), template=tv)
        d = _by_path(divs)["prefetch.fusion_weights.prescope_alpha"]
        assert d.klass == "template-copy"
        assert "no Explanation row" in d.why

    def test_derived_value_off_template_stays_unexplained(self):
        """The template fallback matches VALUES, not paths: a derived value
        that differs from the template too is still a solver gap."""
        tv = templates.template_view("glm5_next")
        ref = {"prefetch": {"fusion_weights": {"prescope_alpha": 0.9}}}
        der = {"prefetch": {"fusion_weights": {"prescope_alpha": 0.123}}}
        divs = compare(ref, der, ExplanationSet(), template=tv)
        assert _by_path(divs)[
            "prefetch.fusion_weights.prescope_alpha"].klass == "unexplained"

    def test_any_derivation_row_is_an_explanation(self):
        """heuristic/closed_form rows without refs are explanations too —
        the old refs-or-measurement gate mislabelled them 'no explanation
        covers this path'."""
        ex = ExplanationSet()
        ex.add("serving.prefix_cache.max_cached_tokens", 131072, "heuristic",
               "active-context lever rounded up to a 64k granule")
        divs = compare(
            {"serving": {"prefix_cache": {"max_cached_tokens": 65536}}},
            {"serving": {"prefix_cache": {"max_cached_tokens": 131072}}}, ex)
        d = _by_path(divs)["serving.prefix_cache.max_cached_tokens"]
        assert d.klass == "solver-choice"
        assert "[heuristic]" in d.why

    def test_real_gap_still_reads_unexplained(self):
        divs = compare({"compute": {"foo": 1}}, {"compute": {"foo": 2}},
                       ExplanationSet(),
                       template=templates.template_view("glm5_next"))
        assert _by_path(divs)["compute.foo"].klass == "unexplained"

    def test_derived_omission_reads_unexplained(self):
        """The solver DROPPING a field the reference sets is a gap (the
        reference-omits symmetry does not apply)."""
        divs = compare({"serving": {"enable_auto_tool_choice": True}},
                       {"serving": {}}, ExplanationSet())
        d = _by_path(divs)["serving.enable_auto_tool_choice"]
        assert d.klass == "unexplained"
        assert "OMITS" in d.why


# ------------------------------------------------------- array lengths

class TestArrayLengths:
    def test_length_change_emits_length_row_and_recurses(self):
        """1 GPU vs 4: the shape decision is ONE row classified through the
        array path's Explanation; matching indices still diff per-field."""
        ex = ExplanationSet()
        ex.add("hardware.gpus", "4 entries", "closed_form",
               "one entry per detected GPU", refs=("AUTOCONFIG §5 E1",))
        ref = {"hardware": {"gpus": [{"id": 0, "vram_gb": 30}]}}
        der = {"hardware": {"gpus": [{"id": 0, "vram_gb": 28},
                                     {"id": 1, "vram_gb": 30},
                                     {"id": 2, "vram_gb": 30},
                                     {"id": 3, "vram_gb": 30}]}}
        divs = _by_path(compare(ref, der, ex))
        assert divs["hardware.gpus.length"].klass == "solver-choice"
        assert divs["hardware.gpus.length"].reference == 1
        assert divs["hardware.gpus.length"].derived == 4
        # index 0 still recursed per-field
        assert divs["hardware.gpus[0].vram_gb"].reference == 30
        assert divs["hardware.gpus[0].vram_gb"].derived == 28

    def test_one_sided_elements_inherit_the_length_class(self):
        ex = ExplanationSet()
        ex.add("hardware.gpus", "4 entries", "closed_form",
               "one entry per detected GPU", refs=("AUTOCONFIG §5 E1",))
        ref = {"hardware": {"gpus": [{"id": 0}]}}
        der = {"hardware": {"gpus": [{"id": 0}, {"id": 1}, {"id": 2}]}}
        divs = _by_path(compare(ref, der, ex))
        for i in (1, 2):
            d = divs[f"hardware.gpus[{i}]"]
            assert d.klass == "solver-choice"
            assert "hardware.gpus.length" in d.why

    def test_unjustified_length_change_reads_unexplained_once(self):
        ref = {"things": [{"a": 1}]}
        der = {"things": [{"a": 1}, {"a": 2}]}
        divs = _by_path(compare(ref, der, ExplanationSet()))
        assert divs["things.length"].klass == "unexplained"
        assert divs["things[1]"].klass == "unexplained"
        assert "things.length" in divs["things[1]"].why


# ---------------------------------------------------------- reporting

class TestReport:
    def test_summary_counts_every_class(self):
        divs = [Divergence("a", 1, 2, k, "w") for k in CLASS_ORDER]
        counts = class_counts(divs)
        assert all(counts[k] == 1 for k in CLASS_ORDER)
        line = summary_line(divs)
        for k in CLASS_ORDER:
            assert k in line

    def test_report_states_acceptance_per_class(self):
        divs = [Divergence("x.y", 1, 2, "unexplained", "gap"),
                Divergence("z", "<missing>", 3, "reference-omits", "omit")]
        rep = render_report("champ.json", divs)
        assert "1 unexplained" in rep
        assert "1 reference-omits" in rep
        assert "Only `unexplained` gates acceptance" in rep


# --------------------------------------------------- template view data

class TestTemplateView:
    def test_view_mirrors_assemble_copies(self):
        """The sections _assemble copies verbatim must resolve through the
        view — keep this in lock-step with Solver._assemble."""
        for arch in ("glm_moe_dsa", "deepseek_v4", "glm5_next"):
            tv = templates.template_view(arch)
            assert tv["prefetch"] == templates.COMMON["prefetch"]
            assert tv["speculation"] == templates.COMMON["speculation_scaffold"]
            assert tv["parallelism"] == templates.COMMON["parallelism"]
            assert tv["transfer"] == templates.COMMON["transfer"]
            assert tv["compute"]["cuda_graphs"] == \
                templates.COMMON["compute_cuda_graphs"]
            assert tv["compute"]["gemm"] == templates.COMMON["compute_gemm"]
            assert tv["memory"]["expert_cache"] == \
                templates.COMMON["expert_cache"]
            assert tv["memory"]["numa"] == templates.COMMON["numa"]
        assert templates.template_view("glm5_next")[
            "orchestrator"]["max_batch_size"] == 512
        assert templates.template_view("glm_moe_dsa")[
            "orchestrator"]["max_batch_size"] == 64
        assert templates.template_view("glm5_next")[
            "serving"]["tool_call_parser"] == "glm47"

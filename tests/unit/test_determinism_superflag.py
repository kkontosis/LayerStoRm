"""Determinism superflags — Python apply semantics, C++<->Python registry
parity (both axes), the two-flag hierarchy, and the tree-scan negative
control.

The scan is the tripwire the P-29 campaign needed: it walks every engine-runtime
source dir for env reads of LS_*/LAYERSTORM_* names and FAILS if any name is
missing from the C++ registry (src/core/determinism.cpp) — so a new
trajectory-forking flag cannot silently escape BOTH flags.  Four such flags
were added in the two days before this landed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from orchestrator import determinism as det

REPO = Path(__file__).resolve().parents[2]
CPP_REGISTRY = REPO / "src" / "core" / "determinism.cpp"

# Matches one C++ registry row:
#   {"ENV", A::kAxis, C::kClass, "forced"|nullptr, K::kKind, "exact"|nullptr, "evidence"},
_CPP_ROW = re.compile(
    r'\{"(?P<env>(?:LS|LAYERSTORM)_[A-Z0-9_]+)",\s*A::(?P<axis>k\w+),\s*'
    r'C::(?P<cls>k\w+),\s*(?P<forced>"[^"]*"|nullptr),\s*K::(?P<conf>k\w+),'
    r'\s*(?P<exact>"[^"]*"|nullptr),')

# Env-name read/consume sites: any call taking the quoted name as an
# argument — getenv("X"), os.environ.get("X"), environ["X"], env_double("X",
# ...).  Deliberately wide; a false positive is a one-line registry entry.
_READ = re.compile(r'[\(\[]\s*"((?:LS|LAYERSTORM)_[A-Z0-9_]+)"')

# Engine-runtime source roots (tools/tests/benchmarks excluded: not serving
# knobs).  deps dirs may be absent/uninitialized in some checkouts — skipped.
_SCAN_ROOTS = [
    "src",
    "python/orchestrator",
    "python/cli",
    "python/bridge",
    "deps/LayerStoRmKernels/csrc",
    "deps/LayerStoRmGemmKernels/csrc",
    "deps/LayerStoRmExpertKernels/csrc",
    "deps/LayerStoRmCpuExpertKernels/csrc",
]
_EXTS = {".cpp", ".h", ".hpp", ".cu", ".cuh", ".py"}


def _cpp_rows() -> dict[str, tuple[str, str, str | None, str, str | None]]:
    text = CPP_REGISTRY.read_text(encoding="utf-8")
    rows: dict[str, tuple[str, str, str | None, str, str | None]] = {}
    for m in _CPP_ROW.finditer(text):
        forced = None if m["forced"] == "nullptr" else m["forced"].strip('"')
        exact = None if m["exact"] == "nullptr" else m["exact"].strip('"')
        assert m["env"] not in rows, f"duplicate C++ row {m['env']}"
        rows[m["env"]] = (m["axis"], m["cls"], forced, m["conf"], exact)
    return rows


def _scan_env_names(roots: list[Path]) -> set[str]:
    names: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.suffix not in _EXTS or not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            names.update(_READ.findall(text))
    return names


# ── Registry parity (C++ is the single source of truth for the full set) ────

def test_cpp_registry_parses_and_is_large():
    rows = _cpp_rows()
    assert len(rows) > 150, "C++ registry parse regressed (or table shrank)"
    # THE exemplar row: deterministic run-to-run, reference-forking.
    assert rows["LS_TQ_SPLITKV"] == (
        "kReference", "kForceOff", "0", "kIfTruthy", None)
    # An enforcer on the run-to-run axis.
    assert rows["LAYERSTORM_DETERMINISTIC_EP_COMBINE"] == (
        "kRunToRun", "kForceOn", "1", "kIfFalsy", None)
    assert rows["LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION"] == (
        "kReference", "kForceOn", "bf16", "kIfExact", "fp32")
    assert rows["LS_SPEC_GOVERNOR"][0] == "kRunToRun"


def test_python_mirror_matches_cpp_actionable_rows():
    cpp = _cpp_rows()
    py = {k.env: (k.axis, k.cls, k.forced, k.conflict, k.conflict_exact)
          for k in det.REGISTRY}
    # Every Python row exists in C++ with identical semantics (both axes).
    for env, row in py.items():
        assert env in cpp, f"{env} in Python registry but not in C++"
        assert cpp[env] == row, (
            f"{env}: C++ {cpp[env]} != Python {row} — keep the two registries "
            "in lockstep (src/core/determinism.cpp <-> "
            "python/orchestrator/determinism.py)")
    # Every ACTIONABLE C++ row (anything but audited-neutral) exists in Python.
    for env, (_axis, cls, *_rest) in cpp.items():
        if cls != "kNeutral":
            assert env in py, (
                f"{env} is {cls} in C++ but missing from the Python mirror")


# ── The tree scan: no unregistered knob, anywhere ───────────────────────────

def test_every_env_knob_in_tree_is_registered():
    registered = set(_cpp_rows())
    found = _scan_env_names([REPO / r for r in _SCAN_ROOTS])
    assert found, "scan found nothing — roots moved? fix _SCAN_ROOTS"
    unregistered = sorted(found - registered)
    assert not unregistered, (
        "env knob(s) read by engine-runtime code but MISSING from the "
        "determinism registry — classify each on BOTH axes in "
        "src/core/determinism.cpp (and python/orchestrator/determinism.py if "
        f"actionable) with evidence before landing: {unregistered}")


def test_scan_negative_control_detects_unregistered_knob(tmp_path):
    """The scan must actually catch a fresh unregistered flag."""
    f = tmp_path / "sneaky.cpp"
    f.write_text('static bool on = std::getenv("LS_TOTALLY_NEW_FORK_FLAG");\n')
    found = _scan_env_names([tmp_path])
    assert "LS_TOTALLY_NEW_FORK_FLAG" in found
    assert "LS_TOTALLY_NEW_FORK_FLAG" not in set(_cpp_rows())


# ── mode resolution + the hierarchy ordering ────────────────────────────────

def test_mode_requested_env_overrides_config_per_flag():
    m = det.mode_requested
    assert m({}, {}) == det.MODE_OFF
    assert m({"compute": {"deterministic": True}}, {}) == det.MODE_RUN_TO_RUN
    assert m({"compute": {"reference_trajectory_identity": True}}, {}) \
        == det.MODE_REFERENCE
    assert m({}, {"LS_DETERMINISTIC": "1"}) == det.MODE_RUN_TO_RUN
    assert m({}, {"LS_REFERENCE_TRAJECTORY_IDENTITY": "1"}) \
        == det.MODE_REFERENCE
    # env "0" suppresses the same flag's config value.
    assert m({"compute": {"deterministic": True}},
             {"LS_DETERMINISTIC": "0"}) == det.MODE_OFF
    assert m({"compute": {"reference_trajectory_identity": True}},
             {"LS_REFERENCE_TRAJECTORY_IDENTITY": "0"}) == det.MODE_OFF
    # flag 2 off does not cancel an independently requested flag 1.
    assert m({"compute": {"deterministic": True,
                          "reference_trajectory_identity": True}},
             {"LS_REFERENCE_TRAJECTORY_IDENTITY": "0"}) \
        == det.MODE_RUN_TO_RUN


# ── apply semantics (injected env; os.environ untouched) ────────────────────

def _apply(env: dict[str, str], cfg=None):
    written: dict[str, str] = {}
    lines = det.apply_deterministic_mode(
        cfg, environ=env, setter=lambda k, v: written.__setitem__(k, v))
    return written, lines


def test_off_mode_is_a_noop():
    written, lines = _apply({"LS_TQ_SPLITKV": "1"})
    assert written == {} and lines == []


def test_flag1_forces_only_run_to_run_rows_and_splitkv_survives():
    written, lines = _apply({"LS_DETERMINISTIC": "1", "LS_TQ_SPLITKV": "1"})
    assert written == {
        "LAYERSTORM_DETERMINISTIC_REDUCE": "1",
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE": "1",
        "LS_SPEC_GOVERNOR": "0",
    }
    # The exemplar: deterministic-but-forking pin SURVIVES flag 1.
    assert "LS_TQ_SPLITKV" not in written
    assert any("RUN-TO-RUN DETERMINISM" in l for l in lines)
    assert any("LS_TQ_SPLITKV) are permitted" in l for l in lines)


def test_flag2_forces_both_axes():
    written, lines = _apply({"LS_REFERENCE_TRAJECTORY_IDENTITY": "1"})
    assert written == {
        "LAYERSTORM_DETERMINISTIC_REDUCE": "1",
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE": "1",
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION": "bf16",
        "LS_TQ_SPLITKV": "0",
        "LS_SNAPMLA_FP8_DECODE": "0",
        "LS_ORCH_SUBGRID_MIDEDGE": "0",
        "LS_SPEC_GOVERNOR": "0",
    }
    assert any("REFERENCE-TRAJECTORY IDENTITY (implies run-to-run" in l
               for l in lines)
    assert any("FORCED (reference) LS_TQ_SPLITKV=0" in l for l in lines)
    assert any("FORCED (run-to-run) LS_SPEC_GOVERNOR=0" in l for l in lines)
    assert any("DET-TOPK-TIES" in l for l in lines)
    assert any("'reference' is RELATIVE" in l for l in lines)


def test_hierarchy_suppression_refused():
    with pytest.raises(det.DeterminismConflictError) as ei:
        _apply({"LS_REFERENCE_TRAJECTORY_IDENTITY": "1",
                "LS_DETERMINISTIC": "0"})
    assert "IMPLIES" in str(ei.value)
    # ...and in config form too.
    with pytest.raises(det.DeterminismConflictError):
        _apply({}, {"compute": {"reference_trajectory_identity": True,
                                "deterministic": False}})


def test_apply_conforming_pin_logged_not_conflicting():
    written, lines = _apply({"LS_REFERENCE_TRAJECTORY_IDENTITY": "1",
                             "LS_TQ_SPLITKV": "0"})
    assert written["LS_TQ_SPLITKV"] == "0"
    assert any("LS_TQ_SPLITKV=0 already pinned" in l for l in lines)


def test_apply_refuses_on_conflicting_pin_and_lists_all():
    with pytest.raises(det.DeterminismConflictError) as ei:
        _apply({"LS_REFERENCE_TRAJECTORY_IDENTITY": "1",
                "LS_TQ_SPLITKV": "1",
                "LS_SNAPMLA_FP8_DECODE": "1",
                "LS_PINNED_EXACT_WIDTHS": "0"})
    msg = str(ei.value)
    assert "LS_TQ_SPLITKV" in msg
    assert "LS_SNAPMLA_FP8_DECODE" in msg
    assert "LS_PINNED_EXACT_WIDTHS" in msg
    assert "REFUSING" in msg


def test_flag1_ignores_reference_axis_pins():
    # fp32 payload + FP8 decode + exact-widths=0 are all reference-axis pins:
    # under flag 1 alone they are permitted (deterministic, just forking).
    written, _ = _apply({"LS_DETERMINISTIC": "1",
                         "LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION": "fp32",
                         "LS_SNAPMLA_FP8_DECODE": "1",
                         "LS_PINNED_EXACT_WIDTHS": "0"})
    assert "LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION" not in written
    # But the run-to-run-axis pin (CPU expert) is refused under flag 1 too.
    with pytest.raises(det.DeterminismConflictError):
        _apply({"LS_DETERMINISTIC": "1", "LS_CPU_EXPERT": "1"})


def test_apply_neutral_pins_never_refuse():
    written, _ = _apply({"LS_REFERENCE_TRAJECTORY_IDENTITY": "1",
                         "LS_PERF_TRACE": "16000000",
                         "LS_LOADER_SHADOW": "0",
                         "LS_KDA_XRAY": "1"})
    assert "LS_PERF_TRACE" not in written  # untouched, not conflicting


def test_apply_refuses_explicit_config_contradictions():
    # Run-to-run-axis config contradictions refuse under BOTH modes.
    for arm in ({"deterministic": True},
                {"reference_trajectory_identity": True}):
        with pytest.raises(det.DeterminismConflictError):
            _apply({}, {"compute": {**arm, "deterministic_ep_combine": False}})
        with pytest.raises(det.DeterminismConflictError):
            _apply({}, {"compute": {**arm, "deterministic_reduce": False}})
    # Reference-axis config contradictions refuse only under flag 2.
    for cfg in (
        {"compute": {"reference_trajectory_identity": True,
                     "deterministic_ep_combine_precision": "fp32"}},
        {"compute": {"reference_trajectory_identity": True},
         "_internal-prefix_cache": {"subgrid_mid_edge": True}},
    ):
        with pytest.raises(det.DeterminismConflictError):
            _apply({}, cfg)
    written, _ = _apply({}, {
        "compute": {"deterministic": True,
                    "deterministic_ep_combine_precision": "fp32"},
        "_internal-prefix_cache": {"subgrid_mid_edge": True}})
    assert written["LAYERSTORM_DETERMINISTIC_EP_COMBINE"] == "1"
    # A schema DEFAULT (key absent) is never a pin — forcing past defaults is
    # the whole point of the superflags.
    written, _ = _apply({}, {"compute": {"reference_trajectory_identity": True}})
    assert written["LS_TQ_SPLITKV"] == "0"


def test_snapmla_fp8_exact_semantics():
    # Since the P-29 step-14 OQ-7 default flip the parser is default-ON with
    # "anything but exactly '0' = ON": any truthy pin ("1", "true") is a
    # contrary ON pin and must be refused in reference mode; ="0" conforms
    # with the forced canonical value and is written through unrefused.
    with pytest.raises(det.DeterminismConflictError):
        _apply({"LS_REFERENCE_TRAJECTORY_IDENTITY": "1",
                "LS_SNAPMLA_FP8_DECODE": "true"})
    with pytest.raises(det.DeterminismConflictError):
        _apply({"LS_REFERENCE_TRAJECTORY_IDENTITY": "1",
                "LS_SNAPMLA_FP8_DECODE": "1"})
    written, _ = _apply({"LS_REFERENCE_TRAJECTORY_IDENTITY": "1",
                         "LS_SNAPMLA_FP8_DECODE": "0"})
    assert written["LS_SNAPMLA_FP8_DECODE"] == "0"

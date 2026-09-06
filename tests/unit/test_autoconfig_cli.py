"""Autoconfig CLI/persistence tests: opt-in flow, fingerprint reuse,
explain sidecar (TD-AUTOCONFIG-HARDWARE-FIT). Hermetic: injected hardware
descriptor, no live detection, no weights."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from test_autoconfig_detect import build_fake_tree

import pytest

from autoconfig.cli import build_arg_parser, run, default_output_path
from autoconfig.hwdetect import detect_hardware

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _hw_json(tmp_path):
    proc, sysd = build_fake_tree(str(tmp_path / "tree"))
    hw = detect_hardware(proc_root=proc, sys_root=sysd)
    p = tmp_path / "hw.json"
    p.write_text(json.dumps(hw.to_json()))
    return str(p)


def _base(tmp_path):
    with open(os.path.join(REPO, "recipes", "glm52_serve_champion.json")) as f:
        base = json.load(f)
    p = tmp_path / "base.json"
    p.write_text(json.dumps(base))
    return str(p)


class TestPreferLever:
    """AUTOCONFIG §2.3 — the third human-facing lever, surfaced exactly like
    the other two: an optional flag with a good default."""

    def test_the_flag_is_choice_validated(self):
        ap = build_arg_parser()
        assert ap.parse_args([]).prefer is None            # unset -> balanced
        assert ap.parse_args(["--prefer", "capacity"]).prefer == "capacity"
        with pytest.raises(SystemExit):
            ap.parse_args(["--prefer", "fastest"])

    def test_the_flag_reaches_the_solver_and_is_silent_when_inert(
            self, tmp_path, capsys):
        """On a box with slack the preference changes nothing, so the
        derivation must be identical and the sidecar must not mention it."""
        base = _base(tmp_path)
        hw = _hw_json(tmp_path)
        outs = {}
        for p in ("balanced", "speed", "capacity"):
            out = str(tmp_path / f"{p}.json")
            assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                       hardware_json=hw, prefer=p) == 0
            outs[p] = (json.loads(open(out).read()),
                       open(out.replace(".json", ".explain.md")).read())
        def body(r):     # minus the solver metadata (output_path differs)
            return {k: v for k, v in r.items() if k != "autoconfig"}
        assert body(outs["speed"][0]) == body(outs["balanced"][0])
        assert body(outs["capacity"][0]) == body(outs["balanced"][0])
        assert all("autoconfig.prefer" not in ex for _, ex in outs.values())

    def test_the_recipe_carries_prefer_only_when_non_default(self, tmp_path):
        """The schema promotion this class long pinned as owed landed
        (TD-KVXP-SCHEMA-KNOBS window, 2026-09-02): `autoconfig.prefer` is a
        declared property, and the WRITE side now carries it — but ONLY
        when non-default, so a default derivation's recipe is byte-identical
        to the pre-promotion one while a lever-carrying recipe re-applies
        its preference on every --autoconfig boot (the read side was live
        all along)."""
        base_path = _base(tmp_path)
        base = json.loads(open(base_path).read())
        base.setdefault("autoconfig", {})["prefer"] = "capacity"
        open(base_path, "w").write(json.dumps(base))
        out = str(tmp_path / "derived.json")
        assert run(base_path, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=_hw_json(tmp_path)) == 0
        # carried forward: a re-derivation from the emitted recipe alone
        # keeps the preference
        assert json.loads(open(out).read())["autoconfig"]["prefer"] \
            == "capacity"
        # and the DEFAULT lever is still never emitted — the non-emission
        # contract survives for the default, keeping default derivations
        # byte-identical
        base2_path = _base(tmp_path)
        out2 = str(tmp_path / "derived-default.json")
        assert run(base2_path, out2, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=_hw_json(tmp_path)) == 0
        assert "prefer" not in json.loads(open(out2).read())["autoconfig"]


class TestAccuracyLever:
    """AUTOCONFIG §2.4 — the fourth human-facing lever
    (TD-AUTOCONFIG-ACCURACY-LEVER), surfaced exactly like `prefer`: an
    optional choice-validated flag with a good default, read from a base
    recipe's `autoconfig` section and (since the 2026-09-02 schema window)
    written back exactly when non-default."""

    def test_the_flag_is_choice_validated(self):
        ap = build_arg_parser()
        assert ap.parse_args([]).accuracy is None          # unset -> standard
        assert ap.parse_args(["--accuracy", "high"]).accuracy == "high"
        with pytest.raises(SystemExit):
            ap.parse_args(["--accuracy", "maximal"])

    def test_standard_reaches_the_solver_and_is_a_no_op(self, tmp_path):
        """Naming the default tier changes nothing: identical recipe body
        and no `autoconfig.accuracy` row in the sidecar."""
        base = _base(tmp_path)
        hw = _hw_json(tmp_path)
        outs = {}
        for t in (None, "standard", "compact"):
            out = str(tmp_path / f"acc-{t}.json")
            assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                       hardware_json=hw, accuracy=t) == 0
            outs[t] = (json.loads(open(out).read()),
                       open(out.replace(".json", ".explain.md")).read())
        def body(r):
            return {k: v for k, v in r.items() if k != "autoconfig"}
        assert body(outs["standard"][0]) == body(outs[None][0])
        assert body(outs["compact"][0]) == body(outs[None][0])
        assert all("autoconfig.accuracy" not in ex for _, ex in outs.values())

    def test_superior_refuses_with_exit_code_3(self, tmp_path, capsys):
        """The defined-but-not-implemented tier is a first-class refusal
        (§6): exit 3 and the missing kFull codec arm named on stderr."""
        base = _base(tmp_path)
        rc = run(base, str(tmp_path / "sup.json"), 0.0627, 51200,
                 repo_root=str(tmp_path), hardware_json=_hw_json(tmp_path),
                 accuracy="superior")
        assert rc == 3
        err = capsys.readouterr().err
        assert "accuracy-superior-not-implemented" in err
        assert "kFull" in err

    def test_the_recipe_carries_accuracy_only_when_non_default(
            self, tmp_path):
        """Same promotion as `prefer` (schema property landed 2026-09-02):
        a non-default tier is carried in the emitted recipe so it survives
        re-derivation; the default tier is still never emitted (default
        derivations stay byte-identical). The read side was live all
        along."""
        base_path = _base(tmp_path)
        base = json.loads(open(base_path).read())
        base.setdefault("autoconfig", {})["accuracy"] = "compact"
        open(base_path, "w").write(json.dumps(base))
        out = str(tmp_path / "derived.json")
        assert run(base_path, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=_hw_json(tmp_path)) == 0
        assert json.loads(open(out).read())["autoconfig"]["accuracy"] \
            == "compact"
        base2_path = _base(tmp_path)
        out2 = str(tmp_path / "derived-default.json")
        assert run(base2_path, out2, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=_hw_json(tmp_path), accuracy="standard") == 0
        assert "accuracy" not in json.loads(open(out2).read())["autoconfig"]


class TestCliFlow:
    def test_persist_recipe_and_explain_sidecar(self, tmp_path, capsys):
        base = _base(tmp_path)
        out = str(tmp_path / "derived.json")
        rc = run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                 hardware_json=_hw_json(tmp_path))
        assert rc == 0
        recipe = json.loads(open(out).read())
        # a normal, readable recipe: full section inventory + self-description
        for sec in ("model", "quantization", "hardware", "memory", "serving",
                    "speculation", "compute", "autoconfig"):
            assert sec in recipe
        assert recipe["autoconfig"]["enabled"] is True
        assert recipe["autoconfig"]["fingerprint"].startswith("hwfp1-")
        ex = open(out.replace(".json", ".explain.md")).read()
        assert "| path | value | kind |" in ex
        assert "vram_expert_ratio" in ex
        # every speed-derived choice is provisional without a calibration
        assert recipe["gpu_loader"]["calibration_mode"] == "full"

    def test_fingerprint_reuse_no_rederive(self, tmp_path, capsys):
        base = _base(tmp_path)
        out = str(tmp_path / "derived.json")
        hw = _hw_json(tmp_path)
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=hw) == 0
        before = open(out).read()
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=hw) == 0
        assert "reusing" in capsys.readouterr().out
        assert open(out).read() == before  # untouched
        # --redetect forces a fresh derivation
        assert run(base, out, 0.0627, 51200, redetect=True,
                   repo_root=str(tmp_path), hardware_json=hw) == 0

    def test_fingerprint_change_rederives(self, tmp_path):
        base = _base(tmp_path)
        out = str(tmp_path / "derived.json")
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=_hw_json(tmp_path)) == 0
        # re-slot the cards into Gen4: fingerprint must miss -> re-derive
        proc, sysd = build_fake_tree(str(tmp_path / "tree2"),
                                     pcie_max="16.0 GT/s PCIe")
        hw2 = detect_hardware(proc_root=proc, sys_root=sysd)
        p2 = tmp_path / "hw2.json"
        p2.write_text(json.dumps(hw2.to_json()))
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=str(p2)) == 0
        fp = json.loads(open(out).read())["autoconfig"]["fingerprint"]
        assert fp.startswith("hwfp1-")

    def test_infeasible_exit_code(self, tmp_path):
        base = _base(tmp_path)
        rc = run(base, str(tmp_path / "x.json"), 0.9, 51200,
                 repo_root=str(tmp_path), hardware_json=_hw_json(tmp_path))
        assert rc == 3

    def test_default_output_path(self):
        assert default_output_path("/a/b/base.json") == "/a/b/base.autoconfig.json"


class TestPinParsing:
    """TD-AUTOCONFIG-PINNED-CONSTRAINTS parse-time refusals: unknown paths
    list the menu, contradictions and bad values refuse, unmodeled
    speculation methods refuse naming the gap (honest stub)."""

    def test_inline_and_file_spellings_parse_and_flatten(self, tmp_path):
        from autoconfig.pins import parse_pin_args
        f = tmp_path / "pins.json"
        f.write_text(json.dumps(
            {"hardware": {"dcp_indexer_mode": "local"},
             "serving.max_sequence_length": 25600}))
        ps = parse_pin_args(["compute.attention_backend=snapmla", str(f)])
        assert ps.get("compute.attention_backend").value == "snapmla"
        assert ps.get("hardware.dcp_indexer_mode").value == "local"
        assert ps.get("serving.max_sequence_length").value == 25600
        # inline RHS is JSON-parsed: booleans and ints work
        ps2 = parse_pin_args(["memory.kv_tiering.enabled=true",
                              "parallelism.tensor_parallelism=2"])
        assert ps2.get("memory.kv_tiering.enabled").value is True
        assert ps2.get("parallelism.tensor_parallelism").value == 2

    def test_unknown_path_refuses_listing_the_menu(self):
        from autoconfig.pins import parse_pin_args
        with pytest.raises(ValueError) as ei:
            parse_pin_args(["orchestrator.max_batch_size=32"])
        s = str(ei.value)
        assert "not a pinnable field" in s
        assert "compute.attention_backend" in s     # the menu is listed
        assert "decision surface" in s

    def test_contradictory_duplicate_refuses(self, tmp_path):
        from autoconfig.pins import parse_pin_args
        with pytest.raises(ValueError) as ei:
            parse_pin_args(["hardware.dcp_indexer_mode=local",
                            "hardware.dcp_indexer_mode=replicated"])
        assert "pinned twice" in str(ei.value)
        # the same value twice is not a contradiction
        ps = parse_pin_args(["hardware.dcp_indexer_mode=local",
                             "hardware.dcp_indexer_mode=local"])
        assert len(ps) == 1

    def test_bad_enum_value_refuses(self):
        from autoconfig.pins import parse_pin_args
        with pytest.raises(ValueError) as ei:
            parse_pin_args(["hardware.dcp_kv_mode=striped"])
        assert "replicated/sharded" in str(ei.value)

    def test_unmodeled_speculation_method_refuses_naming_the_gap(self):
        from autoconfig.pins import parse_pin_args
        with pytest.raises(ValueError) as ei:
            parse_pin_args(["speculation.method=mtp"])
        assert "no sizing model" in str(ei.value)


class TestPinFlag:
    """The --pin surface: CLI-only BY DESIGN — unlike prefer/accuracy
    (schema-promoted 2026-09-02), a pin file is a path argument, the
    emitted recipe never carries pins (a self-pinned recipe would re-apply
    stale hard constraints on every --autoconfig boot), and pins always
    force a re-derivation."""

    def test_the_flag_is_repeatable_and_validated_early(self, capsys):
        ap = build_arg_parser()
        assert ap.parse_args([]).pin is None
        args = ap.parse_args(["--pin", "a=1", "--pin", "b=2"])
        assert args.pin == ["a=1", "b=2"]
        from autoconfig.cli import main
        rc = main(["--config", "nonexistent.json",
                   "--pin", "bogus.path=1"])
        assert rc == 2      # parse fails FAST, before anything is touched
        assert "not a pinnable field" in capsys.readouterr().err

    def test_pins_reach_the_solver_and_are_marked_pinned(self, tmp_path):
        from autoconfig.pins import parse_pin_args
        base = _base(tmp_path)
        out = str(tmp_path / "o.json")
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=_hw_json(tmp_path),
                   pins=parse_pin_args(
                       ["compute.attention_backend=snapmla"])) == 0
        rec = json.loads(open(out).read())
        assert rec["compute"]["attention_backend"] == "snapmla"
        ex = open(out.replace(".json", ".explain.md")).read()
        assert "| pinned |" in ex
        assert "told, not derived" in ex
        # LOUD: the measured-row override lands in the sidecar warnings
        assert "mla-tq-vs-snapmla-accuracy" in ex

    def test_the_recipe_never_carries_pins(self, tmp_path):
        """The non-emission pin, exactly like prefer/accuracy: a recipe
        that self-pinned would re-apply stale hard constraints on every
        --autoconfig boot."""
        from autoconfig.pins import parse_pin_args
        base = _base(tmp_path)
        out = str(tmp_path / "o.json")
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=_hw_json(tmp_path),
                   pins=parse_pin_args(
                       ["hardware.dcp_indexer_mode=local"])) == 0
        assert "pins" not in json.loads(open(out).read())["autoconfig"]
        # and a pin-free run's autoconfig block is unchanged too
        out2 = str(tmp_path / "p.json")
        assert run(base, out2, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=_hw_json(tmp_path)) == 0
        assert "pins" not in json.loads(open(out2).read())["autoconfig"]

    def test_pins_force_rederivation_past_the_fingerprint_reuse(
            self, tmp_path, capsys):
        """Reusing a stored config would silently ignore a HARD constraint
        — the one silent-failure mode a pin must never have."""
        from autoconfig.pins import parse_pin_args
        base = _base(tmp_path)
        out = str(tmp_path / "o.json")
        hw = _hw_json(tmp_path)
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=hw) == 0
        assert json.loads(open(out).read())["hardware"][
            "dcp_indexer_mode"] == "replicated"
        capsys.readouterr()
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=hw,
                   pins=parse_pin_args(
                       ["hardware.dcp_indexer_mode=local"])) == 0
        assert "reusing" not in capsys.readouterr().out
        assert json.loads(open(out).read())["hardware"][
            "dcp_indexer_mode"] == "local"

    def test_benchmark_of_an_existing_recipe_rejects_pins(self, tmp_path,
                                                          capsys):
        """--benchmark --config runs no derivation, so a pin there would be
        silently ignored — refuse instead."""
        from autoconfig.cli import main
        rc = main(["--benchmark", "--config", _base(tmp_path),
                   "--pin", "hardware.dcp_indexer_mode=local"])
        assert rc == 2
        assert "no derivation" in capsys.readouterr().err


class TestLegacyLeverReuse:
    """TD-AUTOCONFIG-LEGACY-CONFIG-MODE-IGNORES-LEVERS (b): the reuse key is
    hardware fingerprint + LEVERS. Three --active-context values must yield
    three derivations, never one cached recipe with a success message."""

    def test_a_differing_lever_busts_the_reuse_cache(self, tmp_path, capsys):
        base = _base(tmp_path)
        out = str(tmp_path / "derived.json")
        hw = _hw_json(tmp_path)
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=hw) == 0
        capsys.readouterr()
        # same levers: reused
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=hw) == 0
        assert "reusing" in capsys.readouterr().out
        # different active-context: MUST re-derive, and say why
        assert run(base, out, 0.0627, 25600, repo_root=str(tmp_path),
                   hardware_json=hw) == 0
        got = capsys.readouterr().out
        assert "reusing" not in got
        assert "NOT the requested levers" in got
        assert json.loads(open(out).read())["autoconfig"][
            "total_active_context_tokens"] == 25600

    def test_a_differing_prefer_busts_the_reuse_cache(self, tmp_path, capsys):
        base = _base(tmp_path)
        out = str(tmp_path / "derived.json")
        hw = _hw_json(tmp_path)
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=hw) == 0
        capsys.readouterr()
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=hw, prefer="capacity") == 0
        got = capsys.readouterr().out
        assert "reusing" not in got
        assert "NOT the requested levers" in got
        # the non-default lever is now carried by the recipe, so a THIRD run
        # asking for the same thing reuses again
        assert run(base, out, 0.0627, 51200, repo_root=str(tmp_path),
                   hardware_json=hw, prefer="capacity") == 0
        assert "reusing" in capsys.readouterr().out


class TestLegacyModeRejectsAutoRunFlags:
    """TD-AUTOCONFIG-LEGACY-CONFIG-MODE-IGNORES-LEVERS (a): legacy --config
    mode (no --model) is derive-only; auto-run flags it cannot honor REFUSE
    loudly instead of being silently discarded (writing beside the input
    while --out-dir points elsewhere, etc.)."""

    @pytest.mark.parametrize("flags", [
        ["--name", "prod"],
        ["--out-dir", "/tmp/elsewhere"],
        ["--calibration", "/tmp/cal.json"],
        ["--trained", "/tmp/trained.json"],
        ["--placement-table", "/tmp/freq.csv"],
        ["--draft", "none"],
        ["--reset"],
    ])
    def test_each_autorun_flag_refuses(self, tmp_path, capsys, flags):
        from autoconfig.cli import main
        rc = main(["--config", _base(tmp_path)] + flags)
        assert rc == 2
        err = capsys.readouterr().err
        assert flags[0] in err
        assert "--model" in err       # the message names the way out

    def test_the_honored_legacy_flags_still_run(self, tmp_path, capsys):
        from autoconfig.cli import main
        out = str(tmp_path / "o.json")
        rc = main(["--config", _base(tmp_path), "--out", out,
                   "--vram-expert-ratio", "0.0627",
                   "--active-context", "51200",
                   "--repo-root", str(tmp_path),
                   "--hardware", _hw_json(tmp_path)])
        assert rc == 0
        assert os.path.exists(out)


class TestLegacyDraftIdentity:
    """TD-AUTOCONFIG-DRAFT-IDENTITY through the legacy path: the champion
    recipe NAMES its speculator; a foreign checkpoint at that path refuses,
    a missing one derives without a draft OUT LOUD."""

    def test_recipe_named_foreign_draft_refuses(self, tmp_path, capsys):
        base = _base(tmp_path)
        ck = json.loads(open(base).read())[
            "speculation"]["dspark"]["checkpoint_path"]
        d = tmp_path / ck
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({
            "aux_hidden_state_layer_ids": [8, 23, 39, 55, 70],
            "draft_vocab_size": 154880,
            "transformer_layer_config": {"num_hidden_layers": 5,
                                         "hidden_size": 4096,   # != 6144
                                         "vocab_size": 154880},
        }))
        rc = run(base, str(tmp_path / "o.json"), 0.0627, 51200,
                 repo_root=str(tmp_path), hardware_json=_hw_json(tmp_path))
        assert rc == 3
        err = capsys.readouterr().err
        assert "draft-identity-mismatch" in err
        assert "never priced" in err

    def test_recipe_named_missing_draft_warns_and_derives(self, tmp_path,
                                                          capsys):
        # The precondition must be a checkpoint that exists NOWHERE:
        # load_draft_candidate deliberately falls back to CWD when repo_root
        # misses (`for root in (repo_root, ".")`), so a fake repo_root alone
        # still resolves the real in-tree speculator.
        with open(os.path.join(REPO, "recipes",
                               "glm52_serve_champion.json")) as f:
            base = json.load(f)
        base["speculation"]["dspark"]["checkpoint_path"] = \
            "test-data/NO-SUCH-speculator.dspark"
        bp = tmp_path / "base_missing_draft.json"
        bp.write_text(json.dumps(base))
        rc = run(str(bp), str(tmp_path / "o.json"), 0.0627, 51200,
                 repo_root=str(tmp_path), hardware_json=_hw_json(tmp_path))
        assert rc == 0
        got = capsys.readouterr()
        assert "does not exist on this box" in got.err
        assert "WITHOUT a draft" in got.err
        # and the warning is in the explain sidecar, not just on stderr
        assert "WITHOUT a draft" in open(
            str(tmp_path / "o.explain.md")).read()

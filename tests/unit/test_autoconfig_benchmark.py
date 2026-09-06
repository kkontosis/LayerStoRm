"""Autoconfig BENCHMARK mode tests (spec/AUTO_RUN.md 'benchmark').

Hermetic: a fake serve process that emits the engine's real ``[orch-stats]``
line shape, a synthetic recipe, a synthetic corpus. No engine, no GPU, no
weights — the mode's contract is what is under test:

  * the ladder SCALES to the recipe's own max_sequence_length;
  * every number carries the dossier-4b preconditions and the warm/cold
    boot state;
  * a degraded sample is DISCARDED from every aggregate and said so;
  * so is a sample whose TEXT is degenerate while the counters are clean
    (TD-GLM53-EP4-DEGENERATE-GENERATION);
  * the engine is ALWAYS stopped, including when a leg fails;
  * two engines are refused before anything boots;
  * the emitted CSV concatenates across arms without a join.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "python"))

from autoconfig import benchmark, cli, pipeline           # noqa: E402
from autoconfig.benchmark import (BenchmarkError, BenchmarkOptions,  # noqa: E402
                                  PromptCorpus, boot_evidence,
                                  config_knobs, discard_reason,
                                  ladder_targets, ledger_lines,
                                  summarize_samples, text_health,
                                  to_csv_rows)
from autoconfig.enginerun import EngineEnv                # noqa: E402
from autoconfig.explain import Infeasible                 # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STATS_LINE = ("  [orch-stats] req={req} tokens={tokens} finish=length "
              "rounds={tokens} acc=0.0000 (0/0) prefix_hit={hit} "
              "prefill_ms={pf:.1f} decode_ms={dm:.1f} "
              "decode_tok_s={dts:.3f} moe_degraded_layers={deg} "
              "degraded_retries=0 indexer_dense_steps={dense}")

WARM_BOOT_LOG = [
    "[info] arena_attach: early worker adopted + registered the persistent "
    "arena (6 node(s), 277.3 GB) overlapped with rank init",
    "[info] ArenaCache: scan — 12096 adoptable, 1 stale, 0 interrupted",
    "[info] arena_attach: adopted 12096 warm slot(s) — preload covers only "
    "the gaps",
    "[info] live prepack: nothing to build (12096 adopted warm, 0 skipped)",
]


def make_recipe(tmp_path, **over):
    cfg = {
        "model": {"architecture": "glm-dsa", "weights_path": "w/model.gguf",
                  "num_hidden_layers": 79, "n_routed_experts": 256},
        "quantization": {"weights": "gguf", "kv_cache": "fp8_e4m3"},
        "hardware": {
            "dcp_enabled": True, "dcp_indexer_mode": "replicated",
            "dcp_kv_mode": "sharded",
            "gpus": [
                {"id": 0, "type": "rtx5090",
                 "roles": ["attention", "resident", "expert_streaming"]},
                {"id": 1, "type": "rtx5090",
                 "roles": ["attention", "resident", "expert_streaming"]},
                {"id": 2, "type": "rtx5080", "roles": ["expert_streaming"]},
                {"id": 3, "type": "rtx5080", "roles": ["expert_streaming"]},
            ],
        },
        "parallelism": {"tensor_parallelism": 2},
        "memory": {"kv_tiering": {"enabled": True},
                   "arena_placement": {"freq_table": "t.csv"},
                   "arena_attach": {"persist": True}},
        "compute": {"attention_backend": "snapmla",
                    "deterministic_ep_combine": True},
        "orchestrator": {"max_batch_size": 64, "prefill_chunk_tokens": 64},
        "speculation": {"method": "dspark",
                        "dspark": {"checkpoint_path": "d/spec.dspark"}},
        "serving": {"max_sequence_length": 25600,
                    "max_concurrent_requests": 32,
                    "tokenizer_path": "auto",
                    "prefix_cache": {"enabled": True}},
        "gpu_loader": {"calibration_path": "cal_trained.json"},
        "autoconfig": {"fingerprint": "fp-abc"},
    }
    for k, v in over.items():
        cfg[k] = v
    p = tmp_path / "arm.autoconfig.json"
    p.write_text(json.dumps(cfg, indent=1))
    return str(p), cfg


class FakeProc:
    """A serve.py stand-in that speaks the engine's own log line."""

    instances: list = []

    # per-request degradation script: index -> (degraded, dense)
    script: dict = {}
    fail_on: int = -1
    decode_tok_s = 6.691
    prefill_ms = 1000.0

    def __init__(self, config_path, env, log_path, extra_env=None, port=0,
                 tokenizer_path=""):
        self.log_path = log_path
        self.config_path = config_path
        self.started = self.stopped = False
        self.req = 0
        self.prompts: list = []
        FakeProc.instances.append(self)

    def _append(self, text):
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)) or ".",
                    exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(text + "\n")

    def start(self):
        self.started = True
        for line in WARM_BOOT_LOG:
            self._append(line)

    def wait_ready(self, timeout_s, poll_s=2.0):
        return None

    def model_id(self):
        return "fake-model"

    def complete(self, model_id, prompt, max_tokens, timeout_s):
        self.req += 1
        if self.req == FakeProc.fail_on:
            raise OSError("simulated engine failure mid-leg")
        n = len(prompt) if isinstance(prompt, list) else len(prompt) // 4
        deg, dense = FakeProc.script.get(self.req, (0, 0))
        self.prompts.append(prompt)
        # a decode-heavy request reports a decode wall; a prefill probe's
        # 8 tokens still report one (the engine always does)
        self._append(STATS_LINE.format(
            req=self.req, tokens=max_tokens, hit=0,
            pf=FakeProc.prefill_ms * max(n, 1) / 1000.0,
            dm=1000.0 * max_tokens / FakeProc.decode_tok_s,
            dts=FakeProc.decode_tok_s, deg=deg, dense=dense))
        return {"choices": [{"text": "x" * 10}],
                "usage": {"prompt_tokens": n, "completion_tokens": max_tokens,
                          "total_tokens": n + max_tokens}}

    def stop(self, timeout_s=180.0):
        self.stopped = True
        return 0


@pytest.fixture
def quiet_box(monkeypatch):
    FakeProc.instances = []
    FakeProc.script = {}
    FakeProc.fail_on = -1
    monkeypatch.setattr(benchmark, "engine_pids", lambda: [])
    monkeypatch.setattr(benchmark, "gpu_compute_apps", lambda: [])
    monkeypatch.setattr(benchmark, "arena_holder_pids", lambda: [4242])
    monkeypatch.setattr(benchmark, "gpu_inventory", lambda: [
        {"index": 0, "name": "NVIDIA GeForce RTX 5090",
         "memory_total_mib": 32607}])
    monkeypatch.setattr(benchmark, "det_topk_ties_state", lambda root: {
        "state": "present", "head": "25105fe", "required": "8228e71",
        "detail": "deps/LayerStoRmKernels HEAD 25105fe contains 8228e71"})
    monkeypatch.setattr(benchmark, "build_corpus",
                        lambda root, cfg, opts, log=print: PromptCorpus(
                            ids=list(range(1000, 1000 + 50000)),
                            source="synthetic", files=("synthetic",),
                            sha256="deadbeef"))
    monkeypatch.setattr(benchmark, "ServeProcess", FakeProc)


# ------------------------------------------------------------ the ladder

class TestLadderScalesToTheConfig:
    def test_champion_shaped_context(self):
        rungs = ladder_targets(25600, gen_tokens=8)
        assert rungs == sorted(rungs)
        assert len(rungs) == 4
        assert all(r % 64 == 0 for r in rungs)
        assert all(r < 25600 for r in rungs)
        assert rungs[-1] > 24000        # the TOP is what long ctx is for

    def test_a_million_token_config_gets_a_million_token_ladder(self):
        rungs = ladder_targets(1_048_576, gen_tokens=8)
        assert rungs[-1] > 900_000
        assert rungs[0] > 100_000       # ...and no vestigial 8k rung
        assert all(r < 1_048_576 for r in rungs)

    def test_rungs_halve(self):
        rungs = ladder_targets(1_048_576, steps=4)
        for a, b in zip(rungs, rungs[1:]):
            assert 1.9 < b / a < 2.1

    def test_floor_truncates_a_small_context(self):
        # top = (4096-8)*0.95 -> 3840 on the 64-grid; the third rung (960)
        # is under the floor, so the ladder stops at two
        assert ladder_targets(4096, floor=1024) == [1920, 3840]

    def test_no_max_sequence_length_is_a_refusal_not_a_guess(self):
        with pytest.raises(BenchmarkError) as e:
            ladder_targets(0)
        assert "max_sequence_length" in str(e.value)


# --------------------------------------------------- the arm's identity

class TestKnobsTravelWithTheNumbers:
    def test_every_axis_a_chart_needs(self, tmp_path):
        _, cfg = make_recipe(tmp_path)
        k = config_knobs(cfg)
        assert k["max_sequence_length"] == 25600
        assert k["max_concurrent_requests"] == 32
        assert k["gpu_count"] == 4
        assert k["tensor_parallelism"] == 2
        assert k["expert_parallelism"] == 4
        assert k["dcp_indexer_mode"] == "replicated"
        assert k["dcp_kv_mode"] == "sharded"
        assert k["kv_tiering"] is True
        assert k["has_draft"] is True and k["draft"] == "spec.dspark"
        assert k["placement_table"] is True

    def test_a_draftless_recipe_says_so(self, tmp_path):
        _, cfg = make_recipe(tmp_path)
        cfg["speculation"] = {"method": "none"}
        k = config_knobs(cfg)
        assert k["has_draft"] is False and k["draft"] == ""

    def test_an_omitted_key_records_the_SCHEMA_default_not_a_zero(self):
        """A recipe may legally omit these; recording 0 would mislabel a
        chart axis, and a 0 max_sequence_length would refuse a ladder the
        engine would happily serve."""
        k = config_knobs({})
        assert k["max_sequence_length"] == 32768
        assert k["max_concurrent_requests"] == 32
        assert k["max_batch_size"] == 64
        assert k["prefill_chunk_tokens"] == 64
        assert k["tensor_parallelism"] == 2
        assert k["dcp_indexer_mode"] == "replicated"
        assert k["dcp_kv_mode"] == "sharded"
        assert k["attention_backend"] == "snapmla"

    def test_expert_parallelism_follows_the_roles(self, tmp_path):
        _, cfg = make_recipe(tmp_path)
        cfg["hardware"]["gpus"][2]["roles"] = ["attention"]
        assert config_knobs(cfg)["expert_parallelism"] == 3


# ------------------------------------------------------ the stats source

class TestOrchStatsIsTheSource:
    def test_the_real_line_parses(self, tmp_path):
        log = tmp_path / "s.log"
        log.write_text(
            "boot\n" + STATS_LINE.format(req=7, tokens=300, hit=12, pf=26240.0,
                                         dm=44900.0, dts=6.691, deg=0,
                                         dense=0) + "\n")
        st = benchmark.OrchStatsTail(str(log)).next_stats(timeout_s=1)
        assert st["request_id"] == 7
        assert st["prefill_ms"] == 26240.0
        assert st["decode_tok_s"] == 6.691
        assert st["prefix_hit_tokens"] == 12
        assert st["moe_degraded_layers"] == 0

    def test_a_missing_line_is_loud_not_a_zero(self, tmp_path):
        log = tmp_path / "s.log"
        log.write_text("boot only\n")
        with pytest.raises(BenchmarkError) as e:
            benchmark.OrchStatsTail(str(log)).next_stats(timeout_s=0.2)
        assert "no honest number" in str(e.value)


class TestDossier4bDiscardRules:
    def test_degraded_is_discarded_with_a_reason(self):
        r = discard_reason({"moe_degraded_layers": 3, "indexer_dense_steps": 0})
        assert "INCOMPLETE expert set" in r

    def test_dense_indexer_is_discarded_as_a_bug_witness(self):
        r = discard_reason({"moe_degraded_layers": 0, "indexer_dense_steps": 5})
        assert "BUG witness" in r

    def test_a_clean_sample_is_kept(self):
        assert discard_reason({"moe_degraded_layers": 0,
                               "indexer_dense_steps": 0}) == ""

    # TD-GLM53-EP4-DEGENERATE-GENERATION: the counters said "healthy" while
    # the engine echoed its own prompt.  These are the two real completions
    # from that defect's evidence file, verbatim.
    CLEAN = {"moe_degraded_layers": 0, "indexer_dense_steps": 0}
    LOOP = (" was a little girl who was a little girl who was a little girl "
            "who was a little girl who was a little girl who was a little "
            "girl who was a little girl who")
    ECHO_PROMPT = "The capital of France is"
    ECHO = (" Paris, and the capital of France is the capital of France is "
            "the capital of France is the capital of France is the capital "
            "of France is")
    GOOD = (" loved to play with her toys. One day, she found a mysterious "
            "box in her room. When she opened it, she saw a magical world "
            "inside, full of talking animals and floating islands that "
            "shimmered in the afternoon light.")

    def test_a_looping_completion_is_discarded_though_counters_are_clean(self):
        h = text_health("Once upon a time, there was a little girl who",
                        self.LOOP)
        assert h["repetition_ratio"] < benchmark.DEGENERATE_REPETITION_RATIO
        r = discard_reason(self.CLEAN, h)
        assert "LOOPS" in r and "repetition_ratio" in r

    def test_a_prompt_echo_is_discarded_though_counters_are_clean(self):
        h = text_health(self.ECHO_PROMPT, self.ECHO)
        r = discard_reason(self.CLEAN, h)
        assert r, "an engine that re-emits its prompt is not a serving rate"

    def test_a_healthy_completion_is_kept(self):
        h = text_health("Once upon a time, there was a little girl who",
                        self.GOOD)
        assert h["repetition_ratio"] == 1.0 and h["prompt_echo_ratio"] == 0.0
        assert discard_reason(self.CLEAN, h) == ""

    def test_short_completions_are_never_judged_on_text(self):
        # Below the word floor the ratios are noise -- never discard on them.
        h = text_health("hi", " hi hi hi hi hi")
        assert h["words"] < benchmark.DEGENERATE_MIN_WORDS
        assert discard_reason(self.CLEAN, h) == ""

    def test_token_id_prompts_report_no_echo_but_still_catch_loops(self):
        h = text_health([1, 2, 3], self.LOOP)
        assert h["prompt_echo_ratio"] is None
        assert "LOOPS" in discard_reason(self.CLEAN, h)

    # The generation leg's own prompt, 2026-09-01 (GF3 ladder re-run): the
    # HEALTHY, EP-fixed engine looped on the old instruction prompt in all 5
    # samples while the ladder rungs of the SAME BOOT read as fluent prose.
    # /v1/completions applies no chat template, so a finished imperative is
    # continued the only way its surface form invites — by restating itself.
    HEALTHY_ENGINE_LOOP = (
        " Then explain, step by step and in complete sentences, how a "
        "mixture-of-experts transformer routes each token to its experts "
        "during decoding, and why the expert weights must be fetched before "
        "the layer can finish." * 5)
    OLD_INSTRUCTION_PROMPT = (
        "Explain, step by step and in complete sentences, how a "
        "mixture-of-experts transformer routes each token to its experts "
        "during decoding, and why the expert weights must be fetched before "
        "the layer can finish.")

    def test_the_generation_prompt_is_a_continuation_seed(self):
        """A raw-completion benchmark prompt must be UNFINISHED.

        The gate is not wrong to discard a restating completion — it is
        wrong to hand a base-mode endpoint a finished imperative and then
        call the result a serving rate.  A degeneracy gate no healthy engine
        can pass measures the prompt (TD-AUTOCONFIG-BENCHMARK-RAW-COMPLETION).
        """
        assert not benchmark.GEN_PROMPT.rstrip().endswith((".", "?", "!"))
        h = text_health(benchmark.GEN_PROMPT, self.HEALTHY_ENGINE_LOOP)
        # the loop the OLD prompt produced would still be caught
        assert "LOOPS" in discard_reason(self.CLEAN, h)

    def test_the_old_instruction_prompt_is_what_tripped_the_gate(self):
        h = text_health(self.OLD_INSTRUCTION_PROMPT, self.HEALTHY_ENGINE_LOOP)
        assert h["repetition_ratio"] <= benchmark.DEGENERATE_REPETITION_RATIO
        assert h["prompt_echo_ratio"] >= benchmark.DEGENERATE_ECHO_RATIO
        assert discard_reason(self.CLEAN, h)

    def test_aggregates_exclude_discarded_samples(self):
        s = [{"index": 1, "v": 10.0, "discarded": False, "discard_reason": ""},
             {"index": 2, "v": 99.0, "discarded": True,
              "discard_reason": "degraded"},
             {"index": 3, "v": 12.0, "discarded": False, "discard_reason": ""}]
        out = summarize_samples(s, "v")
        assert out["values"] == [10.0, 12.0] and out["n"] == 2
        assert out["median"] == 11.0
        assert out["discarded"] == [{"index": 2, "reason": "degraded"}]


# --------------------------------------------------------- warm vs cold

class TestWarmColdIsRecorded:
    def test_a_warm_attach_is_read_off_the_engine_log(self):
        ev = boot_evidence(WARM_BOOT_LOG)
        assert ev["arena"] == "warm"
        assert ev["warm_slots_adopted"] == 12096

    def test_a_private_arena_is_its_own_state_not_cold(self):
        # A process-private arena is NOT a cold holder store: it is a small
        # throwaway arena, and its fetch behaviour (hence its tok/s) is not
        # the recipe's.  Collapsing the two into "cold" is what let a 28 GB
        # private-arena boot's 2.3 tok/s read as a decode datum (2026-08-31).
        ev = boot_evidence(["arena_attach: holder unreachable — "
                            "process-private arena"])
        assert ev["arena"] == "process-private"
        assert ev["process_private"] is True

    def test_a_cold_build_reports_its_prepack_cost_and_full_coverage(self):
        ev = boot_evidence([
            "[info] live prepack: building 12096 slot(s) over 6 node(s) with "
            "33 worker(s) (0 adopted warm, 0 skipped-full)",
            "[info] live prepack: filled 12096 slot(s) (207.5 GB) in 432.34 s "
            "(0.48 GB/s), 0 failed, 0 adopted warm, 0 skipped-full"])
        assert ev["arena"] == "cold"
        assert ev["slots_built"] == 12096
        assert ev["prepack_gb"] == 207.5
        assert ev["prepack_seconds"] == 432.34
        assert ev["arena_coverage"] == 1.0

    def test_a_short_arena_is_caught_by_coverage(self):
        # 10467 of 12096 slots had nowhere to go: this boot served from an
        # arena that does not hold the model, and every number it produces
        # is arena-starved rather than representative of the recipe.
        ev = boot_evidence([
            "[info] live prepack: filled 1629 slot(s) (27.9 GB) in 7.05 s "
            "(3.97 GB/s), 0 failed, 0 adopted warm, 10467 skipped-full"])
        assert ev["arena_coverage"] == 0.1347
        assert ev["slots_skipped_full"] == 10467

    def test_a_partly_warm_boot_is_neither_warm_nor_cold(self):
        ev = boot_evidence([
            "[info] live prepack: filled 1000 slot(s) (17.2 GB) in 40.10 s "
            "(0.43 GB/s), 0 failed, 11096 adopted warm, 0 skipped-full"])
        assert ev["arena"] == "partial"
        assert ev["warm_slots_adopted"] == 11096
        assert ev["slots_built"] == 1000

    def test_the_det_combine_boot_notice_overrides_the_config(self):
        ev = boot_evidence(["  [orch] NOTICE: deterministic_ep_combine is "
                            "OFF -- the routed EP combine follows..."])
        assert ev["deterministic_ep_combine_notice"]


# ------------------------------------------------------------- the guard

class TestNeverTwoEngines:
    def test_a_live_serve_process_refuses_before_anything_boots(
            self, monkeypatch):
        monkeypatch.setattr(benchmark, "engine_pids", lambda: [1234])
        monkeypatch.setattr(benchmark, "gpu_compute_apps", lambda: [])
        monkeypatch.setattr(benchmark, "arena_holder_pids", lambda: [])
        with pytest.raises(BenchmarkError) as e:
            benchmark.refuse_if_box_busy()
        assert "Never two engines" in str(e.value)

    def test_a_gpu_compute_app_the_name_guard_cannot_see(self, monkeypatch):
        monkeypatch.setattr(benchmark, "engine_pids", lambda: [])
        monkeypatch.setattr(benchmark, "gpu_compute_apps",
                            lambda: [{"pid": 99, "used_mib": 13700,
                                      "gpu": "GPU-x"}])
        monkeypatch.setattr(benchmark, "arena_holder_pids", lambda: [])
        with pytest.raises(BenchmarkError):
            benchmark.refuse_if_box_busy()

    def test_the_override_exists_and_still_records_what_it_overrode(
            self, monkeypatch):
        monkeypatch.setattr(benchmark, "engine_pids", lambda: [1234])
        monkeypatch.setattr(benchmark, "gpu_compute_apps", lambda: [])
        monkeypatch.setattr(benchmark, "arena_holder_pids", lambda: [7])
        st = benchmark.refuse_if_box_busy(allow=True)
        assert st["serve_pids"] == [1234] and st["arena_holder_pids"] == [7]


# ------------------------------------------------------------ end to end

@pytest.mark.usefixtures("quiet_box")
class TestTheRunItself:
    def _run(self, tmp_path, **kw):
        cfg_path, _ = make_recipe(tmp_path)
        opts = BenchmarkOptions(
            label="arm-A", ladder=(1024, 2048), repeats=2, gen_runs=3,
            gen_tokens=32, **kw)
        rec = benchmark.run_benchmark(
            cfg_path, EngineEnv(repo_root=REPO), str(tmp_path / "logs"),
            opts, log=lambda *a: None)
        return rec, cfg_path

    def test_it_measures_both_legs_and_stops_the_engine(self, tmp_path):
        rec, _ = self._run(tmp_path)
        assert rec["aborted"] == ""
        assert rec["generation"]["tok_s"]["n"] == 3
        assert rec["generation"]["tok_s"]["median"] == pytest.approx(6.691)
        assert rec["generation"]["tok_s"]["values"] == [6.691] * 3
        assert [r["target_tokens"] for r in rec["ladder"]] == [1024, 2048]
        assert all(r["prefill_tok_s"]["n"] == 2 for r in rec["ladder"])
        assert FakeProc.instances[-1].stopped is True
        assert rec["conditions"]["engines_still_running"] == []

    def test_the_warmup_is_measured_but_never_aggregated(self, tmp_path):
        rec, _ = self._run(tmp_path)
        kinds = [s["kind"] for s in rec["samples"]]
        assert kinds[0] == "warmup"
        assert len(rec["samples"]) == 1 + 3 + 2 * 2
        assert all(row[3] != "warmup" for row in to_csv_rows(rec)[1:])

    def test_ladder_prompts_are_exact_and_share_no_prefix(self, tmp_path):
        rec, _ = self._run(tmp_path)
        proc = FakeProc.instances[-1]
        ladder = [p for p in proc.prompts
                  if isinstance(p, list) and len(p) in (1024, 2048)]
        assert len(ladder) == 4
        by_len = {}
        for p in ladder:
            by_len.setdefault(len(p), []).append(p)
        for reps in by_len.values():
            # different ring offsets => a different first token => the
            # prefix cache cannot answer the repeat
            assert len({r[0] for r in reps}) == len(reps)

    def test_a_degraded_sample_is_discarded_and_said_so(self, tmp_path):
        FakeProc.script = {3: (2, 0)}     # 2nd generation run degrades
        rec, _ = self._run(tmp_path)
        gen = rec["generation"]["tok_s"]
        assert gen["n"] == 2
        assert gen["discarded"] and "INCOMPLETE expert set" in \
            gen["discarded"][0]["reason"]
        assert rec["conditions"]["preconditions_4b"]["degraded_samples"] == 1

    def test_every_4b_precondition_is_on_the_record(self, tmp_path):
        rec, _ = self._run(tmp_path)
        p4 = rec["conditions"]["preconditions_4b"]
        assert p4["deterministic_ep_combine"] == "on"
        assert p4["det_topk_ties_kernel"]["state"] == "present"
        assert p4["subgrid_mid_edge"] == "off"
        assert p4["degraded_samples"] == 0
        assert p4["dense_indexer_samples"] == 0

    def test_warm_cold_and_boot_mode_are_on_the_record(self, tmp_path):
        rec, _ = self._run(tmp_path)
        boot = rec["conditions"]["boot"]
        assert boot["arena"] == "warm"
        assert boot["warm_slots_adopted"] == 12096
        assert boot["mode"] == "holder-attached"
        assert boot["holder_pids_before"] == [4242]
        assert isinstance(boot["boot_wall_s"], float)

    def test_identity_across_generation_runs_is_reported(self, tmp_path):
        rec, _ = self._run(tmp_path)
        assert rec["generation"]["identical_outputs"] is True
        assert rec["generation"]["identity_valid"] is True

    def test_a_failure_mid_leg_still_stops_the_engine_and_reports(
            self, tmp_path):
        FakeProc.fail_on = 3
        rec, _ = self._run(tmp_path)
        assert rec["aborted"].startswith("OSError")
        assert FakeProc.instances[-1].stopped is True
        assert len(rec["samples"]) == 2       # warm-up + one generation run

    def test_a_rung_above_max_sequence_length_is_refused(self, tmp_path):
        cfg_path, _ = make_recipe(tmp_path)
        with pytest.raises(BenchmarkError) as e:
            benchmark.run_benchmark(
                cfg_path, EngineEnv(repo_root=REPO), str(tmp_path / "l"),
                BenchmarkOptions(ladder=(25600,)), log=lambda *a: None)
        assert "max_sequence_length" in str(e.value)
        assert not FakeProc.instances   # refused BEFORE any boot

    def test_skip_switches(self, tmp_path):
        rec, _ = self._run(tmp_path, skip_generation=True)
        assert rec["generation"] == {}
        assert len(rec["ladder"]) == 2


# ------------------------------------------------------------ artifacts

@pytest.mark.usefixtures("quiet_box")
class TestArtifacts:
    def test_csv_is_long_format_with_the_knobs_on_every_row(self, tmp_path):
        cfg_path, _ = make_recipe(tmp_path)
        rec = benchmark.run_benchmark(
            cfg_path, EngineEnv(repo_root=REPO), str(tmp_path / "logs"),
            BenchmarkOptions(label="arm-A", ladder=(1024,), repeats=2,
                             gen_runs=2, gen_tokens=16),
            log=lambda *a: None)
        rows = to_csv_rows(rec)
        head, body = rows[0], rows[1:]
        for col in ("arm_label", "measurement", "target_tokens",
                    "prefill_tok_s", "decode_tok_s", "moe_degraded_layers",
                    "arena", "deterministic_ep_combine",
                    "det_topk_ties_kernel", "max_sequence_length",
                    "dcp_indexer_mode", "kv_tiering", "has_draft",
                    "expert_parallelism"):
            assert col in head
        assert len(body) == 4          # 2 generation + 2 prefill, no warm-up
        # every row carries the arm identity => two arms' CSVs concatenate
        assert {r[head.index("arm_label")] for r in body} == {"arm-A"}
        assert {r[head.index("max_sequence_length")] for r in body} == {25600}

    def test_artifacts_land_beside_the_recipe(self, tmp_path):
        cfg_path, _ = make_recipe(tmp_path)
        rec = benchmark.run_benchmark(
            cfg_path, EngineEnv(repo_root=REPO), str(tmp_path / "logs"),
            BenchmarkOptions(ladder=(1024,), repeats=1, gen_runs=1,
                             gen_tokens=8), log=lambda *a: None)
        out, _ = pipeline.benchmark_paths(cfg_path)
        written = benchmark.write_artifacts(rec, out, log=lambda *a: None)
        assert os.path.exists(written["json"])
        assert os.path.exists(written["csv"])
        assert os.path.exists(written["ledger"])
        assert out.endswith("arm.autoconfig.benchmark.json")
        loaded = json.load(open(written["json"]))
        assert loaded["schema"] == "layerstorm.autoconfig.benchmark/1"

    def test_ledger_snippet_carries_number_regime_and_preconditions(
            self, tmp_path):
        cfg_path, _ = make_recipe(tmp_path)
        rec = benchmark.run_benchmark(
            cfg_path, EngineEnv(repo_root=REPO), str(tmp_path / "logs"),
            BenchmarkOptions(label="glm53-ep4", ladder=(1024,), repeats=2,
                             gen_runs=2, gen_tokens=16), log=lambda *a: None)
        text = ledger_lines(rec)
        assert "glm53-ep4" in text
        assert "decode B=1 greedy" in text and "tok/s median" in text
        assert "served prefill @1024" in text
        assert "det_ep_combine=on" in text
        assert "DET-TOPK-TIES=present" in text
        assert "degraded=0" in text
        assert "arena=warm" in text


# ------------------------------------------------------------------ CLI

class TestCliSurface:
    def test_benchmark_without_a_recipe_is_a_usage_error(self, capsys):
        assert cli.main(["--benchmark"]) == 2
        assert "--config" in capsys.readouterr().err

    def test_benchmark_on_a_missing_config_refuses(self, capsys):
        assert cli.main(["--benchmark", "--config", "/nope/x.json"]) == 3
        assert "benchmark-config-missing" in capsys.readouterr().err

    def test_benchmark_with_config_measures_instead_of_deriving(
            self, tmp_path, monkeypatch):
        cfg_path, _ = make_recipe(tmp_path)
        seen = {}

        def fake(config_path, repo_root, opts, log_dir="", log=print):
            seen.update(config=config_path, label=opts.label,
                        ladder=opts.ladder, repeats=opts.repeats)
            return {"aborted": ""}

        monkeypatch.setattr(cli, "run_benchmark_mode", fake)
        rc = cli.main(["--benchmark", "--config", cfg_path,
                       "--benchmark-label", "arm-B",
                       "--benchmark-ladder", "1024,4096",
                       "--benchmark-repeats", "5"])
        assert rc == 0
        assert seen["config"] == cfg_path and seen["label"] == "arm-B"
        assert seen["ladder"] == (1024, 4096) and seen["repeats"] == 5

    def test_an_aborted_benchmark_exits_nonzero(self, tmp_path, monkeypatch):
        cfg_path, _ = make_recipe(tmp_path)
        monkeypatch.setattr(
            cli, "run_benchmark_mode",
            lambda *a, **k: {"aborted": "EngineRunError: boom"})
        assert cli.main(["--benchmark", "--config", cfg_path]) == 6

    def test_legacy_config_derivation_is_untouched_without_benchmark(
            self, tmp_path, monkeypatch):
        cfg_path, _ = make_recipe(tmp_path)
        called = {}
        monkeypatch.setattr(cli, "run",
                            lambda *a, **k: called.setdefault("n", 1) and 0)
        cli.main(["--config", cfg_path])
        assert called == {"n": 1}


class TestBenchmarkPaths:
    def test_defaults_sit_beside_the_recipe(self, tmp_path):
        out, logs = pipeline.benchmark_paths(str(tmp_path / "a.json"))
        assert out.endswith("a.benchmark.json")
        assert logs.endswith("a.benchmark-logs")

    def test_an_explicit_out_wins(self, tmp_path):
        out, _ = pipeline.benchmark_paths(str(tmp_path / "a.json"),
                                          str(tmp_path / "z.json"))
        assert out.endswith("z.json")

    def test_the_engine_log_follows_the_results_not_the_recipe(self, tmp_path):
        """TD-AUTOCONFIG-BENCHMARK-LOG-DIR-COLLIDES: two runs of ONE recipe
        with different --benchmark-out must not share (and overwrite) one
        engine-log dir, or a ledger row cites the log of a different run."""
        cfg = str(tmp_path / "recipe.json")
        _, a = pipeline.benchmark_paths(cfg, str(tmp_path / "arm-a.json"))
        _, b = pipeline.benchmark_paths(cfg, str(tmp_path / "arm-b.json"))
        assert a != b
        assert a.endswith("arm-a-logs") and b.endswith("arm-b-logs")

    def test_an_explicit_log_dir_still_wins(self, tmp_path):
        _, logs = pipeline.benchmark_paths(str(tmp_path / "a.json"),
                                           str(tmp_path / "z.json"),
                                           str(tmp_path / "elsewhere"))
        assert logs.endswith("elsewhere")


class TestPromptCorpusRing:
    def test_token_slices_are_exact_and_wrap(self):
        c = PromptCorpus(ids=list(range(10)))
        assert c.slice(0, 4) == [0, 1, 2, 3]
        assert c.slice(8, 4) == [8, 9, 0, 1]
        assert len(c.slice(3, 25)) == 25

    def test_wraps_are_counted_so_self_similar_prompts_are_visible(self):
        c = PromptCorpus(ids=list(range(100)))
        assert c.wraps_for(50) == 0
        assert c.wraps_for(250) == 2

    def test_text_mode_slices_by_chars(self):
        c = PromptCorpus(text="abcdefghij" * 10)
        assert len(c.slice(0, 10)) == 40      # 4 chars/token estimate
        assert len(c.slice(95, 10)) == 40     # wraps


class TestRealBuildProbe:
    def test_det_topk_ties_probe_answers_for_this_checkout(self):
        st = benchmark.det_topk_ties_state(REPO)
        assert st["state"] in ("present", "absent", "unknown")
        assert st["detail"]

    def test_an_uninitialised_submodule_is_unknown_not_absent(self, tmp_path):
        """Caught live: an empty deps/LayerStoRmKernels inside a git work
        tree makes `git -C` answer for the PARENT repo, which reported the
        parent's HEAD as a kernels commit and called the kernel ABSENT.
        A build precondition decided from the wrong history is worse than
        an admitted unknown."""
        import subprocess
        root = tmp_path / "checkout"
        (root / "deps" / "LayerStoRmKernels").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        st = benchmark.det_topk_ties_state(str(root))
        assert st["state"] == "unknown"
        assert "not a checked-out submodule" in st["detail"]

    def test_a_missing_submodule_directory_is_unknown(self, tmp_path):
        st = benchmark.det_topk_ties_state(str(tmp_path))
        assert st["state"] == "unknown"

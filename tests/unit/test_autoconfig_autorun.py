"""Auto-run pipeline tests (spec/AUTO_RUN.md): model probing from bare
weights, artifact naming/reuse, the measured-step guards, and the emitted
recipe's schema validity. Hermetic — a synthetic GGUF, an injected hardware
descriptor, and a synthetic calibration; no engine, no GPU, no weights.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from test_autoconfig_detect import BOX_GPUS, build_fake_tree, write_fake_gguf

from autoconfig import enginerun, modelprobe, pipeline
from autoconfig.explain import Infeasible
from autoconfig.hwdetect import detect_hardware
from autoconfig.pipeline import AutoRunOptions, plan_paths, run_auto

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# A GLM-5.2-shaped metadata block (the real values, so the probe's arithmetic
# is pinned against a model we can check by hand).
GLM_KV = {
    "general.architecture": "glm-dsa",
    "general.name": "Glm-5.2",
    "glm-dsa.block_count": 79,             # 78 hidden + 1 MTP
    "glm-dsa.context_length": 1048576,
    "glm-dsa.embedding_length": 6144,
    "glm-dsa.feed_forward_length": 12288,
    "glm-dsa.attention.head_count": 64,
    "glm-dsa.attention.head_count_kv": 1,  # MLA: not the recipe's value
    "glm-dsa.rope.freq_base": 8000000.0,
    "glm-dsa.attention.layer_norm_rms_epsilon": 1e-05,
    "glm-dsa.expert_count": 256,
    "glm-dsa.expert_used_count": 8,
    "glm-dsa.expert_group_count": 1,
    "glm-dsa.expert_group_used_count": 1,
    "glm-dsa.expert_gating_func": 2,
    "glm-dsa.leading_dense_block_count": 3,
    "glm-dsa.vocab_size": 154880,
    "glm-dsa.attention.q_lora_rank": 2048,
    "glm-dsa.attention.kv_lora_rank": 512,
    "glm-dsa.attention.key_length_mla": 256,
    "glm-dsa.attention.value_length_mla": 256,
    "glm-dsa.expert_feed_forward_length": 2048,
    "glm-dsa.expert_shared_count": 1,
    "glm-dsa.expert_weights_scale": 2.5,
    "glm-dsa.expert_weights_norm": True,
    "glm-dsa.rope.dimension_count": 64,
    "glm-dsa.nextn_predict_layers": 1,
    "glm-dsa.attention.indexer.head_count": 32,
    "glm-dsa.attention.indexer.key_length": 128,
    "glm-dsa.attention.indexer.top_k": 2048,
}

EXPERT_TENSORS = [
    ("blk.3.ffn_gate_exps.weight", (6144, 2048, 256), 13),
    ("blk.3.ffn_up_exps.weight", (6144, 2048, 256), 13),
    ("blk.3.ffn_down_exps.weight", (2048, 6144, 256), 14),
]


def make_model_dir(tmp_path, name="GLM-5.2-GGUF-Q4_K_XL", kv=None):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "GLM-5.2-UD-Q4_K_XL-00001-of-00002.gguf"
    write_fake_gguf(str(p), EXPERT_TENSORS, kv if kv is not None else GLM_KV)
    write_fake_gguf(str(d / "GLM-5.2-UD-Q4_K_XL-00002-of-00002.gguf"), [], {})
    return str(d), str(p)


def make_calibration(path, hw, n=6144, k=2048):
    """A synthetic artifact in loader_constants.h schema shape whose device
    UUIDs are this (fake) box's — i.e. one the accept predicate takes."""
    gpus = sorted(hw.gpus, key=lambda g: (-g.vram_mib, g.ordinal))
    devices = [{"position": i, "numa_node": g.numa_node, "name": g.name,
                "uuid": g.uuid, "xfer_lat_us": 30.0,
                "compute": {"a_us": 40.0, "b_us": 5.0, "P": 64}}
               for i, g in enumerate(gpus)]
    banks = [{"node": nd.node, "egress_us": 20.0, "contention": 1.0,
              "is_hbm": nd.is_hbm_bank} for nd in hw.numa_nodes]
    matrix = [[{"tier": 0 if b["node"] == d["numa_node"] else 1,
                "rate_us": 500.0} for d in devices] for b in banks]
    with open(path, "w") as f:
        json.dump({"version": 2, "source": "calibrated",
                   "expert_bytes": 27623424.0,
                   "compute_N": 2 * k, "compute_K": n,
                   "devices": devices, "banks": banks, "matrix": matrix}, f)
    return path


def write_fake_trace(path, rounds=10, layers=(3, 4, 5, 6), experts=(0, 1, 2)):
    """A perf_trace.csv in the engine's column shape (ns, stage, gpu, seq,
    key, tok) holding `rounds` decode layer-sweeps: one MoE window per layer
    (stage 12 enter / 15 finalize-exit) with dispatched expert H2D copies
    (stage 4) inside it, keyed layer<<16|expert."""
    ns, seq = 0, 0
    with open(path, "w") as f:
        f.write("ns,stage,gpu,seq,key,tok\n")
        for _ in range(rounds):
            for layer in layers:
                seq += 1
                ns += 100
                f.write(f"{ns},12,0,{seq},{layer},0\n")        # kMoeEnter
                for e in experts:
                    ns += 1
                    f.write(f"{ns},4,0,{seq},{(layer << 16) | e},0\n")
                ns += 1
                f.write(f"{ns},15,0,{seq},0,0\n")              # kMoeFinalizeExit
    return path


@pytest.fixture()
def box(tmp_path):
    proc, sysd = build_fake_tree(str(tmp_path / "tree"))
    hw = detect_hardware(proc_root=proc, sys_root=sysd)
    p = tmp_path / "hw.json"
    p.write_text(json.dumps(hw.to_json()))
    return hw, str(p)


class TestModelProbe:
    def test_gguf_metadata_yields_the_recipe_model_section(self, tmp_path):
        d, shard = make_model_dir(tmp_path)
        src = modelprobe.probe_model(d, str(tmp_path))
        m = src.model_section
        assert m["architecture"] == "glm_moe_dsa"
        # block_count counts the MTP block; num_hidden_layers does not
        assert (m["num_hidden_layers"], m["num_nextn_predict_layers"]) == (78, 1)
        # MLA split: key_length_mla - rope_dim, and KV heads follow the
        # attention heads (GGUF's head_count_kv=1 is the MLA latent, not ours)
        assert (m["qk_nope_head_dim"], m["qk_rope_head_dim"]) == (192, 64)
        assert m["num_key_value_heads"] == 64
        assert (m["kv_lora_rank"], m["v_head_dim"]) == (512, 256)
        assert m["gating_score_fn"] == "sigmoid"
        # the profile fills what GGUF cannot express — getting index_topk_freq
        # wrong turns 21 indexer-computing layers into 79
        assert (m["index_topk_freq"], m["index_skip_topk_offset"]) == (4, 3)
        assert m["rope_interleave"] is True
        assert src.weights_path.endswith("-00001-of-00002.gguf")
        assert src.source == "gguf-metadata"
        assert src.display_name == "glm-5.2"

    def test_shard_one_is_picked_from_a_directory(self, tmp_path):
        d, shard = make_model_dir(tmp_path)
        assert modelprobe.first_shard(d) == shard

    def test_unknown_architecture_refuses(self, tmp_path):
        kv = dict(GLM_KV, **{"general.architecture": "llama-9000"})
        d, _ = make_model_dir(tmp_path, kv=kv)
        with pytest.raises(Infeasible) as e:
            modelprobe.probe_model(d, str(tmp_path))
        assert e.value.constraint_id == "unknown-gguf-architecture"
        assert "--config" in e.value.suggestion

    def test_hf_config_beside_the_weights_wins_over_the_profile(self, tmp_path):
        d, _ = make_model_dir(tmp_path)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"model_type": "glm_moe_dsa", "num_hidden_layers": 12,
                       "hidden_size": 6144, "scoring_func": "sigmoid",
                       "rope_parameters": {"rope_theta": 8000000.0,
                                           "rope_type": "default"}}, f)
        src = modelprobe.probe_model(d, str(tmp_path))
        assert src.source == "hf-config"
        assert src.model_section["num_hidden_layers"] == 12
        assert src.model_section["gating_score_fn"] == "sigmoid"
        assert src.model_section["rope_theta"] == 8000000.0

    def test_tokenizer_discovered_at_a_prefix_sibling(self, tmp_path):
        d, shard = make_model_dir(tmp_path)
        tok = tmp_path / "GLM-5.2"
        tok.mkdir()
        (tok / "tokenizer.json").write_text("{}")
        assert modelprobe.find_tokenizer_dir(shard) == str(tok)
        src = modelprobe.probe_model(d, str(tmp_path))
        assert src.tokenizer_path == "GLM-5.2"

    def test_tokenizer_absent_stays_auto(self, tmp_path):
        d, _ = make_model_dir(tmp_path)
        assert modelprobe.probe_model(d, str(tmp_path)).tokenizer_path == "auto"

    def test_prepack_discovered_by_manifest_identity_not_by_name(self, tmp_path):
        d, shard = make_model_dir(tmp_path)
        for name, src_path in (("other-prepacked", "/somewhere/else.gguf"),
                               ("GLM-5.2-prepacked", shard)):
            p = tmp_path / name
            p.mkdir()
            (p / "manifest.json").write_text(json.dumps(
                {"source_model_path": src_path, "slot": {"slot_size_bytes": 27623424}}))
        got = modelprobe.probe_model(d, str(tmp_path))
        assert got.prepacked_dir == "GLM-5.2-prepacked"
        assert got.live_prepack is False

    def test_draft_discovery(self, tmp_path):
        d, shard = make_model_dir(tmp_path)
        ck = tmp_path / "GLM-5.2-speculator.dspark"
        ck.mkdir()
        (ck / "config.json").write_text("{}")
        assert modelprobe.find_draft_checkpoint(shard, str(tmp_path)) == \
            "GLM-5.2-speculator.dspark"


class TestArtifactNaming:
    def _opts(self, tmp_path, **kw):
        d, _ = make_model_dir(tmp_path)
        return AutoRunOptions(model_path=d, repo_root=str(tmp_path),
                              out_dir=str(tmp_path / "out"), **kw), d

    def test_default_stem_is_the_model_name(self, tmp_path):
        opts, d = self._opts(tmp_path)
        src = modelprobe.probe_model(d, str(tmp_path))
        p = plan_paths(opts, src)
        assert p.stem == "glm-5.2"
        assert p.config_path.endswith("out/glm-5.2.autoconfig.json")
        # one config per model, and its sidecars share the stem
        assert p.explain_path.endswith("glm-5.2.autoconfig.explain.md")
        assert p.autorun_path.endswith("glm-5.2.autoconfig.autorun.json")

    def test_name_and_prefix_overrides(self, tmp_path):
        opts, d = self._opts(tmp_path, name="prod")
        src = modelprobe.probe_model(d, str(tmp_path))
        assert plan_paths(opts, src).stem == "prod"
        opts2, _ = self._opts(tmp_path, prefix="lab-")
        assert plan_paths(opts2, src).stem == "lab-glm-5.2"

    def test_calibration_lives_beside_the_weights_as_a_bare_filename(self, tmp_path):
        opts, d = self._opts(tmp_path)
        src = modelprobe.probe_model(d, str(tmp_path))
        p = plan_paths(opts, src)
        assert os.path.dirname(p.calibration_abs) == d
        # engine.cpp:500-512 joins any relative path to the weights dir, so a
        # weights-adjacent artifact must be written as a BARE filename
        assert p.calibration_cfg == "gpu_loader_calibration_glm-5.2.json"
        assert p.trained_cfg == "gpu_loader_calibration_glm-5.2_trained.json"

    def test_supplied_calibration_elsewhere_becomes_absolute(self, tmp_path):
        cal = tmp_path / "elsewhere" / "cal.json"
        cal.parent.mkdir()
        cal.write_text("{}")
        opts, d = self._opts(tmp_path, calibration=str(cal))
        src = modelprobe.probe_model(d, str(tmp_path))
        p = plan_paths(opts, src)
        assert p.calibration_cfg == str(cal)
        assert p.trained_abs.endswith("cal_trained.json")


class TestAutoRunFlow:
    def _run(self, tmp_path, box, monkeypatch, **kw):
        hw, hw_json = box
        d, shard = make_model_dir(tmp_path)
        cal = make_calibration(str(tmp_path / "cal.json"), hw)
        trained = make_calibration(str(tmp_path / "cal_trained.json"), hw)
        def _no_engine(*a, **k):
            raise AssertionError("no engine boot may happen with artifacts supplied")
        monkeypatch.setattr(pipeline, "run_calibration_boot", _no_engine)
        monkeypatch.setattr(pipeline, "run_training_boot", _no_engine)
        opts = AutoRunOptions(
            model_path=d, repo_root=str(tmp_path), out_dir=str(tmp_path / "out"),
            hardware_json=hw_json, calibration=cal, trained=trained,
            vram_expert_ratio=0.0627, total_active_context_tokens=51200,
            # steps 1-3 are what these cases are about; step 4 would otherwise
            # (correctly) boot to finish the pipeline
            skip_placement=True, **kw)
        return run_auto(opts, log=lambda *a: None), opts, d

    def test_supplied_artifacts_skip_both_measured_steps(self, tmp_path, box, monkeypatch):
        res, opts, d = self._run(tmp_path, box, monkeypatch)
        by_step = {s.step: s for s in res.steps}
        assert by_step["calibrate"].action == "supplied"
        assert by_step["train"].action == "supplied"
        assert by_step["derive"].action == "measured"
        assert os.path.exists(res.config_path)

    def test_the_recipe_points_at_the_TRAINED_artifact(self, tmp_path, box, monkeypatch):
        res, _, _ = self._run(tmp_path, box, monkeypatch)
        recipe = json.load(open(res.config_path))
        # step 3 is the second, better iteration of the constants — serving
        # must use it, not the raw hardware calibration
        assert recipe["gpu_loader"]["calibration_path"].endswith("cal_trained.json")
        assert recipe["gpu_loader"]["calibration_mode"] == "loaded"

    def test_identity_is_derived_not_asked_for(self, tmp_path, box, monkeypatch):
        res, _, _ = self._run(tmp_path, box, monkeypatch)
        recipe = json.load(open(res.config_path))
        assert recipe["model"]["architecture"] == "glm_moe_dsa"
        assert recipe["model"]["weights_path"].endswith("-00001-of-00002.gguf")
        assert recipe["quantization"]["weights"] == "gguf"
        assert recipe["serving"]["max_sequence_length"] == 25600

    def test_sidecars_and_serve_command(self, tmp_path, box, monkeypatch):
        res, opts, _ = self._run(tmp_path, box, monkeypatch)
        stem = res.config_path[: -len(".json")]
        assert "| path | value | kind |" in open(stem + ".explain.md").read()
        assert "## Auto-run steps" in open(stem + ".explain.md").read()
        book = json.load(open(stem + ".autorun.json"))
        assert book["model"]["architecture"] == "glm_moe_dsa"
        # all four levers are recorded, so a re-run is reproducible from the
        # book alone (the config's own `autoconfig` section carries
        # `prefer`/`accuracy` only when non-default — schema-promoted
        # 2026-09-02 — so the book stays the one place ALL levers appear)
        assert book["levers"] == {"vram_expert_ratio": 0.0627,
                                  "total_active_context_tokens": 51200,
                                  "prefer": "balanced",
                                  "accuracy": "standard"}
        assert book["artifacts"]["trained"].endswith("cal_trained.json")
        assert [s["action"] for s in book["steps"]] == \
            ["supplied", "measured", "supplied", "skipped"]
        # the whole point: one line the user runs afterwards
        assert "serve.py --config" in res.serve_command
        assert res.serve_command.startswith("cd ")
        # TD-AUTOCONFIG-SERVE-CMD-DEVICE-ORDER: the recipe's gpus[].id are
        # PCI-ordered, so the pasted command must pin the ordering the
        # pipeline's own boots use — omitting it lands a 5090-sized carve
        # on a 5080 (device_alloc failed)
        assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in res.serve_command


    def test_a_current_config_is_reused_verbatim(self, tmp_path, box, monkeypatch):
        """Tuned once, not every run — and hand edits survive."""
        res, opts, d = self._run(tmp_path, box, monkeypatch)
        recipe = json.load(open(res.config_path))
        recipe["serving"]["port"] = 9999          # a hand edit
        with open(res.config_path, "w") as f:
            json.dump(recipe, f)
        again = run_auto(opts, log=lambda *a: None)
        assert json.load(open(again.config_path))["serving"]["port"] == 9999
        assert [s.action for s in again.steps] == ["reused"] * 4

    def test_an_explicit_lever_change_busts_the_reuse(self, tmp_path, box,
                                                      monkeypatch):
        """TD-AUTOCONFIG-LEGACY-CONFIG-MODE-IGNORES-LEVERS applies to the
        auto-run reuse gate too: a lever the user passed must change the
        output or refuse — never be discarded by a fingerprint-only cache."""
        res, opts, d = self._run(tmp_path, box, monkeypatch)
        lines = []
        again = run_auto(pipeline.AutoRunOptions(
            **dict(opts.__dict__, total_active_context_tokens=25600)),
            log=lambda *a: lines.append(" ".join(str(x) for x in a)))
        assert {s.step: s for s in again.steps}["derive"].action == "measured"
        assert any("NOT the requested levers" in l for l in lines)
        assert json.load(open(again.config_path))["autoconfig"][
            "total_active_context_tokens"] == 25600
        # an UNSET lever never busts the cache — tuned once, not every run
        third = run_auto(pipeline.AutoRunOptions(
            **dict(opts.__dict__, vram_expert_ratio=None,
                   total_active_context_tokens=None)), log=lambda *a: None)
        assert [s.action for s in third.steps] == ["reused"] * 4

    def test_redetect_re_derives_over_a_hand_edit(self, tmp_path, box, monkeypatch):
        res, opts, d = self._run(tmp_path, box, monkeypatch)
        recipe = json.load(open(res.config_path))
        recipe["serving"]["port"] = 9999
        with open(res.config_path, "w") as f:
            json.dump(recipe, f)
        again = run_auto(pipeline.AutoRunOptions(
            **dict(opts.__dict__, redetect=True)), log=lambda *a: None)
        assert json.load(open(again.config_path))["serving"]["port"] == 8000

    def test_skip_training_leaves_the_baseline_calibration(self, tmp_path, box, monkeypatch):
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        cal = make_calibration(str(tmp_path / "cal.json"), hw)
        monkeypatch.setattr(pipeline, "run_training_boot",
                            lambda *a, **k: pytest.fail("training must not run"))
        res = run_auto(AutoRunOptions(
            model_path=d, repo_root=str(tmp_path), out_dir=str(tmp_path / "out"),
            hardware_json=hw_json, calibration=cal, skip_training=True,
            skip_placement=True), log=lambda *a: None)
        recipe = json.load(open(res.config_path))
        assert recipe["gpu_loader"]["calibration_path"] == cal
        assert [s.action for s in res.steps][2] == "skipped"

    def test_missing_supplied_artifact_refuses(self, tmp_path, box):
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        with pytest.raises(Infeasible) as e:
            run_auto(AutoRunOptions(model_path=d, repo_root=str(tmp_path),
                                    out_dir=str(tmp_path / "out"),
                                    hardware_json=hw_json,
                                    calibration=str(tmp_path / "nope.json")),
                     log=lambda *a: None)
        assert e.value.constraint_id == "supplied-calibration-missing"

    def test_supplied_artifact_plus_reset_refuses(self, tmp_path, box, monkeypatch):
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        cal = make_calibration(str(tmp_path / "cal.json"), hw)
        with pytest.raises(Infeasible) as e:
            run_auto(AutoRunOptions(
                model_path=d, repo_root=str(tmp_path),
                out_dir=str(tmp_path / "out"), hardware_json=hw_json,
                calibration=cal, reset=True), log=lambda *a: None)
        assert e.value.constraint_id == "conflicting-flags"

    def test_reset_forces_a_fresh_measurement(self, tmp_path, box, monkeypatch):
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        cal = os.path.join(d, "gpu_loader_calibration_glm-5.2.json")
        make_calibration(cal, hw)
        calls = []

        def fake_boot(config_path, env, log_path, artifact_path, **k):
            calls.append(artifact_path)
            make_calibration(artifact_path, hw)   # the engine writes it
        monkeypatch.setattr(pipeline, "run_calibration_boot", fake_boot)
        res = run_auto(AutoRunOptions(
            model_path=d, repo_root=str(tmp_path), out_dir=str(tmp_path / "out"),
            hardware_json=hw_json, reset=True, skip_training=True,
            skip_placement=True), log=lambda *a: None)
        assert calls == [cal]
        assert [s.action for s in res.steps][0] == "measured"
        # the provisional recipe used for the calibrating boot is cleaned up
        assert not os.path.exists(res.config_path[: -len(".json")]
                                  + ".provisional.json")

    def test_a_rejected_calibration_is_not_silently_used(self, tmp_path, box, monkeypatch):
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        # right box, WRONG model dims (the engine's own accept predicate)
        cal = make_calibration(str(tmp_path / "cal.json"), hw, n=4096, k=1408)
        with pytest.raises(Infeasible) as e:
            run_auto(AutoRunOptions(
                model_path=d, repo_root=str(tmp_path),
                out_dir=str(tmp_path / "out"), hardware_json=hw_json,
                calibration=cal), log=lambda *a: None)
        assert e.value.constraint_id == "supplied-calibration-rejected"


class TestMeasuredStepGuards:
    def test_calibration_refuses_against_a_warm_arena_holder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(enginerun, "arena_holder_pids", lambda: [4242])
        with pytest.raises(enginerun.EngineRunError) as e:
            enginerun.run_calibration_boot(
                "cfg.json", enginerun.EngineEnv(repo_root=REPO),
                str(tmp_path / "log"), str(tmp_path / "cal.json"))
        # registry row full-calibration-needs-cold-boot: measured, not guessed
        assert "cold" in str(e.value).lower()
        assert "4242" in str(e.value)

    def test_serve_subprocess_command_shape(self, tmp_path):
        env = enginerun.EngineEnv(repo_root=REPO)
        proc = enginerun.ServeProcess("cfg.json", env, str(tmp_path / "l.log"),
                                      port=1234, tokenizer_path="tok")
        cmd = proc.command()
        assert cmd[1].endswith(os.path.join("python", "cli", "serve.py"))
        assert cmd[cmd.index("--port") + 1] == "1234"
        assert cmd[cmd.index("--tokenizer-path") + 1] == "tok"
        e = env.base_env()
        assert e["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.path.join(REPO, "python") in e["PYTHONPATH"]

    def test_training_env_carries_the_capture_switches(self, tmp_path, monkeypatch):
        seen = {}

        class FakeProc:
            def __init__(self, cfg, env, log, extra_env=None, **k):
                seen.update(extra_env or {})

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return None

            def wait_ready(self, *a, **k):
                return None

            def decode(self, prompt, max_tokens, timeout):
                open(seen["LS_LOADER_SHADOW_DUMP"], "w").write("{}\n")
                open(seen["LS_PERF_TRACE_OUT"], "w").write("x\n")
                seen["tokens"] = max_tokens
                return {"choices": [{"text": "ok"}]}

        monkeypatch.setattr(enginerun, "ServeProcess", FakeProc)
        enginerun.run_training_boot(
            "cfg.json", enginerun.EngineEnv(repo_root=REPO),
            str(tmp_path / "t.log"), str(tmp_path / "dump.jsonl"),
            str(tmp_path / "trace.csv"))
        assert seen["LS_LOADER_SHADOW"] == "1"
        assert seen["LS_PERF_TRACE"] == "1"
        assert seen["tokens"] == enginerun.TRAIN_TOKENS == 100


class TestEmittedRecipeIsSchemaValid:
    def test_validates_against_config_schema(self, tmp_path, box, monkeypatch):
        jsonschema = pytest.importorskip("jsonschema")
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        cal = make_calibration(str(tmp_path / "cal.json"), hw)
        res = run_auto(AutoRunOptions(
            model_path=d, repo_root=str(tmp_path), out_dir=str(tmp_path / "out"),
            hardware_json=hw_json, calibration=cal, skip_training=True,
            skip_placement=True, vram_expert_ratio=0.0627,
            total_active_context_tokens=51200), log=lambda *a: None)

        def inline(node):
            if isinstance(node, dict):
                ref = node.get("$ref", "")
                if ref.startswith("internal-schemas/"):
                    with open(os.path.join(REPO, "config", ref)) as f:
                        sub = json.load(f)
                    sub.pop("$schema", None)
                    sub.pop("$id", None)
                    return inline(dict(sub, **{k: v for k, v in node.items()
                                               if k != "$ref"}))
                return {k: inline(v) for k, v in node.items()}
            if isinstance(node, list):
                return [inline(v) for v in node]
            return node

        with open(os.path.join(REPO, "config", "schema.json")) as f:
            schema = inline(json.load(f))
        errors = list(jsonschema.Draft202012Validator(schema).iter_errors(
            json.load(open(res.config_path))))
        assert not errors, [f"{list(e.path)}: {e.message}" for e in errors[:5]]


@pytest.mark.skipif(
    not os.path.exists(os.path.join(
        REPO, "test-data/GLM-5.2-GGUF-Q4_K_XL/"
              "GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf")),
    reason="machine-data-gated: the GLM-5.2 GGUF is not present")
class TestRealModelMatchesTheChampion:
    def test_probe_reproduces_the_champion_model_section(self):
        """The acceptance claim of --model: the identity the champion states
        by hand is recoverable from the weights alone."""
        src = modelprobe.probe_model(
            "test-data/GLM-5.2-GGUF-Q4_K_XL", REPO)
        with open(os.path.join(REPO, "recipes/glm52_serve_champion.json")) as f:
            champion = json.load(f)["model"]
        assert src.model_section == champion
        assert src.prepacked_dir == "test-data/GLM-5.2-prepacked"
        assert src.tokenizer_path == "test-data/GLM-5.2"


class TestPlacementFit:
    """Step 4: the arena's demand-fetch table is FITTED, not invented — from
    the trace step 3's decode already produced."""

    def _prepared(self, tmp_path, box, monkeypatch, **kw):
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        cal = make_calibration(str(tmp_path / "cal.json"), hw)

        def fake_training(config_path, env, log_path, dump_path, trace_path,
                          **k):
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            open(dump_path, "w").write("{}\n")
            write_fake_trace(trace_path)
            return {"choices": [{"text": "ok"}]}

        def fake_trainer(repo_root, in_calib, dump, trace, out_calib, log,
                         **k):
            make_calibration(out_calib, hw)
        monkeypatch.setattr(pipeline, "run_training_boot", fake_training)
        monkeypatch.setattr(pipeline, "run_trainer_apply", fake_trainer)
        monkeypatch.setattr(pipeline, "run_capture_boot",
                            lambda *a, **k: pytest.fail("no capture boot is "
                                                        "needed after step 3"))
        return AutoRunOptions(
            model_path=d, repo_root=str(tmp_path), out_dir=str(tmp_path / "out"),
            hardware_json=hw_json, calibration=cal,
            vram_expert_ratio=0.0627, total_active_context_tokens=51200, **kw)

    def test_fitted_from_the_training_trace_by_the_real_fitter(
            self, tmp_path, box, monkeypatch):
        opts = self._prepared(tmp_path, box, monkeypatch)
        res = run_auto(opts, log=lambda *a: None)
        by_step = {s.step: s for s in res.steps}
        assert by_step["place"].action == "measured"
        table = by_step["place"].artifact
        assert os.path.basename(table) == "arena_placement_glm-5.2.csv"
        rows = [l for l in open(table).read().splitlines()
                if l and not l.startswith("#")]
        # 10 sweeps, 3 warm-up excluded -> 7 counted fetches per (layer,expert)
        assert sorted(rows) == sorted(f"{l},{e},7" for l in (3, 4, 5, 6)
                                      for e in (0, 1, 2))
        recipe = json.load(open(res.config_path))
        assert recipe["memory"]["arena_placement"]["freq_table"] == table
        # and it is explained as measured, with the cold-rebuild consequence
        why = open(res.config_path[: -len(".json")] + ".explain.md").read()
        assert "rebuilds the warm store ONCE" in why

    def test_an_existing_table_is_reused_not_refitted(
            self, tmp_path, box, monkeypatch):
        opts = self._prepared(tmp_path, box, monkeypatch)
        res = run_auto(opts, log=lambda *a: None)
        table = {s.step: s for s in res.steps}["place"].artifact
        stamp = os.path.getmtime(table)
        monkeypatch.setattr(pipeline, "run_freq_table_fit",
                            lambda *a, **k: pytest.fail("must not refit"))
        again = run_auto(pipeline.AutoRunOptions(
            **dict(opts.__dict__, redetect=True)), log=lambda *a: None)
        assert {s.step: s for s in again.steps}["place"].action == "reused"
        assert os.path.getmtime(table) == stamp

    def test_supplied_table_is_carried_verbatim(self, tmp_path, box, monkeypatch):
        supplied = tmp_path / "measured_freq.csv"
        supplied.write_text("3,0,9\n")
        opts = self._prepared(tmp_path, box, monkeypatch,
                              placement_table=str(supplied))
        monkeypatch.setattr(pipeline, "run_freq_table_fit",
                            lambda *a, **k: pytest.fail("must not refit"))
        res = run_auto(opts, log=lambda *a: None)
        recipe = json.load(open(res.config_path))
        assert recipe["memory"]["arena_placement"]["freq_table"] == str(supplied)
        assert {s.step: s for s in res.steps}["place"].action == "supplied"

    def test_skip_placement_explains_the_absence(self, tmp_path, box, monkeypatch):
        opts = self._prepared(tmp_path, box, monkeypatch, skip_placement=True)
        monkeypatch.setattr(pipeline, "run_freq_table_fit",
                            lambda *a, **k: pytest.fail("must not fit"))
        res = run_auto(opts, log=lambda *a: None)
        recipe = json.load(open(res.config_path))
        assert "arena_placement" not in recipe["memory"]
        why = open(res.config_path[: -len(".json")] + ".explain.md").read()
        assert "online migrator" in why

    def test_supplied_artifacts_still_finish_the_pipeline(
            self, tmp_path, box, monkeypatch):
        """Supplying an artifact means "do not redo THIS one" — never "do not
        finish the rest". With calibration and trained both supplied, step 4
        still boots to capture the trace it needs, and leaves the supplied
        trained artifact untouched."""
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        cal = make_calibration(str(tmp_path / "cal.json"), hw)
        trained = make_calibration(str(tmp_path / "cal_trained.json"), hw)
        trained_stamp = os.path.getmtime(trained)
        captured = []

        def fake_capture(config_path, env, log_path, trace_path, **k):
            captured.append(trace_path)
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            write_fake_trace(trace_path)
        for name in ("run_calibration_boot", "run_training_boot",
                     "run_trainer_apply"):
            monkeypatch.setattr(pipeline, name,
                                lambda *a, **k: pytest.fail(
                                    "a supplied step must not be redone"))
        monkeypatch.setattr(pipeline, "run_capture_boot", fake_capture)
        lines = []
        res = run_auto(AutoRunOptions(
            model_path=d, repo_root=str(tmp_path), out_dir=str(tmp_path / "out"),
            hardware_json=hw_json, calibration=cal, trained=trained),
            log=lambda *a: lines.append(" ".join(str(x) for x in a)))
        assert len(captured) == 1
        assert {s.step: s for s in res.steps}["place"].action == "measured"
        recipe = json.load(open(res.config_path))
        assert recipe["memory"]["arena_placement"]["freq_table"].endswith(
            "arena_placement_glm-5.2.csv")
        # the supplied trained artifact is untouched, and the output SAYS so
        assert os.path.getmtime(trained) == trained_stamp
        assert recipe["gpu_loader"]["calibration_path"] == trained
        assert any("REUSED, not re-fitted" in l for l in lines)

    def test_skip_placement_is_the_only_way_to_end_without_a_table(
            self, tmp_path, box, monkeypatch):
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        cal = make_calibration(str(tmp_path / "cal.json"), hw)
        trained = make_calibration(str(tmp_path / "cal_trained.json"), hw)
        for name in ("run_calibration_boot", "run_training_boot",
                     "run_capture_boot", "run_freq_table_fit"):
            monkeypatch.setattr(pipeline, name,
                                lambda *a, **k: pytest.fail("no work at all"))
        res = run_auto(AutoRunOptions(
            model_path=d, repo_root=str(tmp_path), out_dir=str(tmp_path / "out"),
            hardware_json=hw_json, calibration=cal, trained=trained,
            skip_placement=True), log=lambda *a: None)
        assert {s.step: s for s in res.steps}["place"].action == "skipped"
        assert "arena_placement" not in json.load(open(res.config_path))["memory"]

    def test_skip_training_no_longer_suppresses_placement(
            self, tmp_path, box, monkeypatch):
        """--skip-training skips step 3 ONLY; step 4 still finishes."""
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        cal = make_calibration(str(tmp_path / "cal.json"), hw)
        captured = []

        def fake_capture(config_path, env, log_path, trace_path, **k):
            captured.append(trace_path)
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            write_fake_trace(trace_path)
        monkeypatch.setattr(pipeline, "run_capture_boot", fake_capture)
        monkeypatch.setattr(pipeline, "run_training_boot",
                            lambda *a, **k: pytest.fail("training was skipped"))
        res = run_auto(AutoRunOptions(
            model_path=d, repo_root=str(tmp_path), out_dir=str(tmp_path / "out"),
            hardware_json=hw_json, calibration=cal, skip_training=True),
            log=lambda *a: None)
        actions = {s.step: s.action for s in res.steps}
        assert actions["train"] == "skipped" and actions["place"] == "measured"
        assert len(captured) == 1
        # step 3 skipped => the recipe still names the untrained calibration
        assert json.load(open(res.config_path))["gpu_loader"][
            "calibration_path"] == cal

    def test_a_config_without_a_table_is_not_current(
            self, tmp_path, box, monkeypatch):
        """An unfinished pipeline must not short-circuit as 'already current'."""
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        cal = make_calibration(str(tmp_path / "cal.json"), hw)
        trained = make_calibration(str(tmp_path / "cal_trained.json"), hw)
        base = dict(model_path=d, repo_root=str(tmp_path),
                    out_dir=str(tmp_path / "out"), hardware_json=hw_json,
                    calibration=cal, trained=trained)
        run_auto(AutoRunOptions(skip_placement=True, **base), log=lambda *a: None)
        captured = []

        def fake_capture(config_path, env, log_path, trace_path, **k):
            captured.append(trace_path)
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            write_fake_trace(trace_path)
        monkeypatch.setattr(pipeline, "run_capture_boot", fake_capture)
        res = run_auto(AutoRunOptions(**base), log=lambda *a: None)
        assert len(captured) == 1
        assert {s.step: s for s in res.steps}["place"].action == "measured"

    def test_supplied_table_plus_refit_refuses(self, tmp_path, box, monkeypatch):
        supplied = tmp_path / "measured_freq.csv"
        supplied.write_text("3,0,9\n")
        opts = self._prepared(tmp_path, box, monkeypatch,
                              placement_table=str(supplied),
                              refit_placement=True)
        with pytest.raises(Infeasible) as e:
            run_auto(opts, log=lambda *a: None)
        assert e.value.constraint_id == "conflicting-flags"


class TestDraftIdentityFlow:
    """TD-AUTOCONFIG-DRAFT-IDENTITY in the pipeline: discovery SKIPS a
    foreign .dspark out loud (never wires it in, never prices it), an
    explicit --draft that is foreign or missing REFUSES before any boot."""

    FOREIGN = {   # a draft for some OTHER model than the fake GLM-5.2
        "aux_hidden_state_layer_ids": [8, 23, 39, 55, 80],   # 80 >= 78 layers
        "draft_vocab_size": 151552,                          # != 154880
        "transformer_layer_config": {"num_hidden_layers": 5,
                                     "hidden_size": 4096,    # != 6144
                                     "vocab_size": 151552},
    }
    MATCHING = {  # GLM-5.2's own geometry (the fake GGUF's)
        "aux_hidden_state_layer_ids": [8, 23, 39, 55, 70],
        "draft_vocab_size": 154880,
        "transformer_layer_config": {"num_hidden_layers": 5,
                                     "hidden_size": 6144,
                                     "vocab_size": 154880},
    }

    def _run(self, tmp_path, box, monkeypatch, draft_cfg=None, **kw):
        hw, hw_json = box
        d, shard = make_model_dir(tmp_path)
        if draft_cfg is not None:
            self._write_draft_in(d, draft_cfg)
        cal = make_calibration(str(tmp_path / "cal.json"), hw)
        trained = make_calibration(str(tmp_path / "cal_trained.json"), hw)
        monkeypatch.setattr(pipeline, "run_calibration_boot",
                            lambda *a, **k: pytest.fail("no boot"))
        monkeypatch.setattr(pipeline, "run_training_boot",
                            lambda *a, **k: pytest.fail("no boot"))
        lines = []
        res = run_auto(AutoRunOptions(
            model_path=d, repo_root=str(tmp_path),
            out_dir=str(tmp_path / "out"), hardware_json=hw_json,
            calibration=cal, trained=trained,
            vram_expert_ratio=0.0627, total_active_context_tokens=51200,
            skip_placement=True, **kw),
            log=lambda *a: lines.append(" ".join(str(x) for x in a)))
        return res, lines

    def _write_draft_in(self, model_dir, cfg):
        import pathlib
        d = pathlib.Path(model_dir) / "spec.dspark"
        d.mkdir()
        (d / "config.json").write_text(json.dumps(cfg))
        return str(d)

    def test_discovery_skips_a_foreign_draft_out_loud(self, tmp_path, box,
                                                      monkeypatch):
        res, lines = self._run(tmp_path, box, monkeypatch,
                               draft_cfg=self.FOREIGN)
        recipe = json.load(open(res.config_path))
        assert "checkpoint_path" not in json.dumps(
            recipe.get("speculation") or {})
        rejected = [l for l in lines if "REJECTED" in l]
        assert len(rejected) == 1
        assert "spec.dspark" in rejected[0]
        assert "never priced" in rejected[0]
        assert "4096" in rejected[0] and "6144" in rejected[0]

    def test_discovery_takes_a_matching_draft(self, tmp_path, box,
                                              monkeypatch):
        res, lines = self._run(tmp_path, box, monkeypatch,
                               draft_cfg=self.MATCHING)
        assert any("identity-matched" in l for l in lines)
        assert not any("REJECTED" in l for l in lines)

    def test_explicit_foreign_draft_refuses(self, tmp_path, box, monkeypatch):
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        foreign = self._write_draft_in(str(tmp_path), self.FOREIGN)
        with pytest.raises(Infeasible) as e:
            run_auto(AutoRunOptions(
                model_path=d, repo_root=str(tmp_path),
                out_dir=str(tmp_path / "out"), hardware_json=hw_json,
                draft_path="spec.dspark"), log=lambda *a: None)
        assert e.value.constraint_id == "draft-identity-mismatch"
        assert "never priced" in str(e.value)

    def test_explicit_missing_draft_refuses(self, tmp_path, box):
        hw, hw_json = box
        d, _ = make_model_dir(tmp_path)
        with pytest.raises(Infeasible) as e:
            run_auto(AutoRunOptions(
                model_path=d, repo_root=str(tmp_path),
                out_dir=str(tmp_path / "out"), hardware_json=hw_json,
                draft_path="nope.dspark"), log=lambda *a: None)
        assert e.value.constraint_id == "draft-checkpoint-missing"

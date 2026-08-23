"""#91 / TD-ORCH-ENGINE-DECODE-E2E — orchestrator-driven REAL-engine decode.

THE first Python-orchestrator ↔ real-engine end-to-end: start_engine (real
CUDA backends, daemon thread) → build_engine_loop (SPSC rings + sideband +
StateSnapshot attach) → submit the GGUF golden's prompt → the orchestrator
teacher-forces the whole prompt (real prefill, token i at position i) and
greedily decodes through the PRODUCTION seams (fused-gate routing export →
E_CMD_FETCH_AND_RUN_MOE per MoE layer; OUTPUT_HEAD argmax readback).

GATE: the first generated token must equal the C++ golden's ground truth —
12089 (" Paris") for "The capital of France is" on the GLM-5.2 Q4_K_XL
GGUF (llama.cpp-verified, see tests/integration/glm52_gguf_golden_test.cpp).

Run (opt-in — loads the 436 GB GGUF, takes tens of minutes cold):
  ORCH_E2E=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
    .venv/bin/python -m pytest tests/integration/test_orch_engine_decode.py -s

TD-DSPARK-PY-LOOP-E2E: test_orchestrator_real_engine_dspark_lossless drives
the Python DSpark loop (_dispatch_dspark_step → sequential early-stop
verify) against the real 3-GPU dspark engine — plain-AR baseline loop first,
then a speculation_depth>0 DSpark loop on the SAME engine; the generated
tokens must be IDENTICAL (INV-DSPARK-LOSSLESS through the Python loop).

Run (3 GPUs: default CUDA fastest-first ordering puts the 5090 TP pair at
visible 0,1 and a 5080 draft at 2 — see spec/DEBUG.md):
  ORCH_E2E_DSPARK=1 CUDA_VISIBLE_DEVICES=0,1,2 \
    .venv/bin/python -m pytest \
    tests/integration/test_orch_engine_decode.py -k dspark -s
Optional: GLM52_PREPACKED=1 (keeper-parity pinned expert arenas — much
faster decode), ORCH_E2E_DSPARK_REQUIRE_ACCEPT=1 (hard-fail when nothing is
accepted; default reports only — the toy 5-token prompt is OOD for the
drafter, see TD-DSPARK-ACCEPT-SHORTPROMPT's realistic-prompt measurement).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_CONFIG = _PROJECT_ROOT / "test-data" / "config" / "glm_5_2_gguf.json"
_GGUF = (_PROJECT_ROOT / "test-data" / "GLM-5.2-GGUF-Q4_K_XL"
         / "GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf")

# llama.cpp ground truth on THIS GGUF (glm52_gguf_golden_test.cpp).
PROMPT_TOKENS = [785, 6722, 315, 9621, 374]   # "The capital of France is"
GOLDEN_NEXT = 12089                            # " Paris"

GEN_TOKENS = 3          # first asserted == GOLDEN_NEXT; rest recorded
RUN_DEADLINE_S = 5400   # cold expert streaming from NVMe is slow

_DSPARK_CKPT = _PROJECT_ROOT / "test-data" / "GLM-5.2-speculator.dspark"


def _gpu_free_mb() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.split()
        return [int(m) for m in out]
    except Exception:
        return []


def _two_big_gpus_free() -> bool:
    """Two visible GPUs with >= 26 GB free (TP=2 keeper shape)."""
    return sum(1 for m in _gpu_free_mb() if m >= 26_000) >= 2


def _dspark_gpus_free() -> bool:
    """TP pair (>= 26 GB free x2) plus a draft GPU (>= 11 GB free)."""
    free = sorted(_gpu_free_mb(), reverse=True)
    return (len(free) >= 3 and free[0] >= 26_000 and free[1] >= 26_000
            and free[2] >= 11_000)


def _base_engine_config() -> dict:
    """The glm52_gguf_golden_test.start_engine config shape (TP=2 5090s)."""
    with open(_CONFIG) as f:
        cfg = json.load(f)
    cfg["model"]["weights_path"] = str(_GGUF)
    cfg["quantization"]["gguf_strategy"] = "int"
    cfg["hardware"]["gpus"] = [
        {"id": 0, "type": "rtx5090", "vram_gb": 30,
         "pcie_gen": 5, "pcie_width": 16},
        {"id": 1, "type": "rtx5090", "vram_gb": 30,
         "pcie_gen": 5, "pcie_width": 16},
    ]
    cfg["hardware"]["tp_array"] = [0, 1]
    cfg["parallelism"]["tensor_parallelism"] = 2
    cfg["memory"]["vram_safety_margin_gb"] = 3.0
    cfg["memory"]["kv_cache"]["max_pages_per_gpu"] = 2048
    cfg["memory"]["kv_cache"]["page_growth_chunk_tokens"] = 16

    # GLM52_PREPACKED=1: keeper-parity pinned expert arenas (same block as
    # glm52_gguf_golden_test.start_engine) — decode becomes hit-mostly
    # instead of cold GGUF streaming.
    prepacked = _PROJECT_ROOT / "test-data" / "GLM-5.2-prepacked"
    if (os.environ.get("GLM52_PREPACKED") == "1"
            and (prepacked / "manifest.json").exists()):
        arena_frac = 0.76
        af = os.environ.get("GLM52_ARENA_FRACTION")
        if af:
            try:
                v = float(af)
                if 0.0 < v <= 1.0:
                    arena_frac = v
            except ValueError:
                pass
        cfg.setdefault("preprocessing", {})["prepacked_dir"] = str(prepacked)
        cfg["memory"]["preload_expert_buffers"] = True
        cfg["memory"]["pin_host_expert_pool"] = True
        cfg["memory"]["pin_host_expert_pool_direct_load"] = True
        cfg["memory"]["pin_host_expert_pool_preload"] = True
        cfg["memory"]["pin_host_expert_pool_direct_o_direct"] = True
        # P-24b: tests default to a process-private arena (never touch the
        # persistent holder store); GLM52_ARENA_ATTACH=1 opts back in.
        cfg["memory"]["arena_attach"] = {
            "enabled": os.environ.get("GLM52_ARENA_ATTACH") == "1"}
        cfg["memory"]["pin_host_expert_pool_sizing"] = {
            "mode": "fraction_total", "value": arena_frac}
        cfg["memory"]["cross_node_spill"] = {
            "enabled": True,
            "nodes": [{"node": 0, "weight": 1}, {"node": 1, "weight": 1}],
            "sizing_mode": "fraction_total",
            "sizing_value": arena_frac,
        }
    return cfg


def _run_request(loop, request_id: int, prompt: list[int], gen_tokens: int,
                 deadline_s: float, t0: float, tag: str) -> tuple[list[int],
                                                                  str]:
    """Submit one request and drive run_one_cycle until completion.

    Prints tokens as they stream and, on completion, the DECODE rate
    (tokens/s from the first to the last generated token — prefill
    excluded), so speculation phases can be compared on the same engine.
    """
    from orchestrator.loop.orchestrator_loop import InferenceRequest

    results: list = []
    tokens_seen: list[int] = []
    token_times: list[float] = []

    def on_token(rid, token_id, lp):
        tokens_seen.append(token_id)
        token_times.append(time.monotonic())
        print(f"[{tag}] token[{len(tokens_seen) - 1}] = {token_id} "
              f"(+{time.monotonic() - t0:.0f}s)", flush=True)

    def on_complete(rid, gen, reason, lp):
        results.append((rid, list(gen), reason))

    loop.submit_request(InferenceRequest(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        gpu=0,
        max_tokens=gen_tokens,
        on_token=on_token,
        on_complete=on_complete,
    ))
    deadline = time.monotonic() + deadline_s
    while not results and time.monotonic() < deadline:
        loop.run_one_cycle()
    assert results, (f"[{tag}] request {request_id} did not finish within "
                     f"{deadline_s}s (tokens so far: {tokens_seen})")
    rid, gen, reason = results[0]
    assert rid == request_id
    if len(token_times) >= 2 and token_times[-1] > token_times[0]:
        rate = (len(token_times) - 1) / (token_times[-1] - token_times[0])
        print(f"[{tag}] decode rate: {rate:.3f} tok/s "
              f"({len(token_times)} tokens)", flush=True)
    # Settle: drain trailing completions (e.g. the finalize E_CMD_SEQ_FREE
    # ack) so a subsequently attached loop never sees this loop's stale
    # completions.
    for _ in range(50):
        loop.run_one_cycle()
    return gen, reason


@pytest.mark.skipif(os.environ.get("ORCH_E2E") != "1",
                    reason="set ORCH_E2E=1 to run (real 436 GB GGUF engine)")
@pytest.mark.skipif(not _GGUF.exists(), reason="GLM-5.2 GGUF not present")
@pytest.mark.skipif(not _two_big_gpus_free(),
                    reason="need TWO GPUs with >= 26 GB free (TP=2)")
def test_orchestrator_real_engine_greedy_decode():
    import layerstorm_engine
    from orchestrator.engine_glue import build_engine_loop
    from orchestrator.loop.orchestrator_loop import OrchestratorConfig

    # ── Config: identical shape to glm52_gguf_golden_test.start_engine ──
    cfg = _base_engine_config()
    cfg_path = "/tmp/orch_e2e_glm52_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[orch-e2e] starting engine (config={cfg_path}) ...", flush=True)
    t0 = time.monotonic()
    info = layerstorm_engine.start_engine(cfg_path)
    try:
        print(f"[orch-e2e] engine up in {time.monotonic() - t0:.0f}s: "
              f"layers={info.num_layers} experts={info.num_experts} "
              f"gpus={info.num_gpus}", flush=True)

        buffer_ids = layerstorm_engine.query_buffer_ids()
        loop, handles = build_engine_loop(
            info,
            buffer_ids=buffer_ids,
            # Pure greedy AR decode (speculation off) — the e2e gate.
            speculation_depth=0,
            # Throttle idle spin; the YIELD wait breaks early on any
            # completion so per-command latency is unaffected.
            orchestrator_config=OrchestratorConfig(cycle_budget_us=200.0),
        )
        assert handles.metadata.hidden_buf_id != 0, "hidden buf not found"
        assert handles.metadata.logits_buf_id != 0, "logits buf not found"

        gen, reason = _run_request(
            loop, 1, PROMPT_TOKENS, GEN_TOKENS, RUN_DEADLINE_S, t0,
            "orch-e2e")
        print(f"[orch-e2e] DONE reason={reason} tokens={gen} "
              f"({time.monotonic() - t0:.0f}s total)", flush=True)
        assert reason == "length", f"unexpected finish reason: {reason}"
        assert len(gen) == GEN_TOKENS
        # THE GATE: greedy next token matches the C++ golden / llama.cpp.
        assert gen[0] == GOLDEN_NEXT, (
            f"orchestrator-driven decode produced {gen[0]}, "
            f"expected {GOLDEN_NEXT} (' Paris')")
    finally:
        layerstorm_engine.stop_engine()
        try:
            os.remove(cfg_path)
        except OSError:
            pass


MTP_GEN_TOKENS = 12      # long enough for several speculation rounds


@pytest.mark.skipif(os.environ.get("ORCH_E2E_MTP") != "1",
                    reason="set ORCH_E2E_MTP=1 to run (real GLM-5.2 engine, "
                           "TP=2)")
@pytest.mark.skipif(not _GGUF.exists(), reason="GLM-5.2 GGUF not present")
@pytest.mark.skipif(not _two_big_gpus_free(),
                    reason="need TWO GPUs with >= 26 GB free (TP=2)")
def test_orchestrator_real_engine_mtp_lossless():
    """MTP batched verification vs the real engine — lossless + throughput.

    Three loops on the SAME engine (speculation.method=mtp so the MTP
    layer's weights are live):
      Phase A: plain-AR baseline (speculation_depth=0).
      Phase B: MTP with SEQUENTIAL verify (batched_verify=False — the
               MtpLossless golden's reference schedule).
      Phase C: MTP with BATCHED verify (ONE V-token forward per round —
               the throughput-bearing mode).

    GATES: token identity across all three (INV-MTP-LOSSLESS through the
    Python loop, both verify shapes) + the batched path actually engaged
    (loop._batched_verify_rounds > 0) + drafts were proposed.  Decode
    rates are printed per phase for the throughput comparison.
    """
    import layerstorm_engine
    from orchestrator.engine_glue import build_engine_loop
    from orchestrator.loop.orchestrator_loop import OrchestratorConfig
    from orchestrator.mtp_draft import MtpDraftConfig

    cfg = _base_engine_config()
    cfg["speculation"]["method"] = "mtp"
    cfg["speculation"]["enabled"] = True
    cfg["speculation"]["mtp"] = {"enabled": True}
    cfg_path = "/tmp/orch_e2e_glm52_mtp_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    depth = int(os.environ.get("ORCH_E2E_MTP_DEPTH", "2"))
    gen_tokens = int(os.environ.get("ORCH_E2E_MTP_TOKENS",
                                    str(MTP_GEN_TOKENS)))

    print(f"\n[orch-mtp] starting TP=2 mtp engine (depth={depth}, "
          f"config={cfg_path}) ...", flush=True)
    t0 = time.monotonic()
    info = layerstorm_engine.start_engine(cfg_path)
    try:
        print(f"[orch-mtp] engine up in {time.monotonic() - t0:.0f}s: "
              f"layers={info.num_layers} experts={info.num_experts} "
              f"gpus={info.num_gpus} moe_batch_capacity="
              f"{getattr(info, 'moe_batch_capacity', 0)}", flush=True)
        buffer_ids = layerstorm_engine.query_buffer_ids()

        # ── Phase A: baseline plain-AR loop (speculation off) ────────────
        loop_a, handles_a = build_engine_loop(
            info,
            buffer_ids=buffer_ids,
            speculation_depth=0,
            orchestrator_config=OrchestratorConfig(cycle_budget_us=200.0),
        )
        assert handles_a.metadata.hidden_buf_id != 0, "hidden buf not found"
        baseline, reason_a = _run_request(
            loop_a, 1, PROMPT_TOKENS, gen_tokens, RUN_DEADLINE_S,
            t0, "orch-mtp-base")
        print(f"[orch-mtp] baseline ({reason_a}): {baseline}", flush=True)
        assert reason_a == "length"
        assert len(baseline) == gen_tokens
        assert baseline[0] == GOLDEN_NEXT, (
            f"baseline produced {baseline[0]}, expected {GOLDEN_NEXT}")

        # ── Phase B: MTP with SEQUENTIAL verify ──────────────────────────
        loop_s, _ = build_engine_loop(
            info,
            buffer_ids=buffer_ids,
            speculation_depth=depth,
            mtp_config=MtpDraftConfig(enabled=True, max_depth=depth,
                                      batched_verify=False),
            num_mtp_layers=1,
            orchestrator_config=OrchestratorConfig(cycle_budget_us=200.0),
        )
        seq_tokens, reason_s = _run_request(
            loop_s, 2, PROMPT_TOKENS, gen_tokens, RUN_DEADLINE_S,
            t0, "orch-mtp-seq")
        draft_s = loop_s._mtp_draft  # noqa: SLF001 — test introspection
        print(f"[orch-mtp] sequential ({reason_s}): {seq_tokens} "
              f"drafts={draft_s.total_steps} "
              f"accepted={draft_s.total_accepted}", flush=True)
        assert reason_s == "length"
        assert seq_tokens == baseline, (
            "INV-MTP-LOSSLESS violated (sequential verify): "
            f"{seq_tokens} != {baseline}")
        assert loop_s._batched_verify_rounds == 0  # noqa: SLF001

        # ── Phase C: MTP with BATCHED verify ─────────────────────────────
        loop_b, _ = build_engine_loop(
            info,
            buffer_ids=buffer_ids,
            speculation_depth=depth,
            mtp_config=MtpDraftConfig(enabled=True, max_depth=depth,
                                      batched_verify=True),
            num_mtp_layers=1,
            orchestrator_config=OrchestratorConfig(cycle_budget_us=200.0),
        )
        bat_tokens, reason_b = _run_request(
            loop_b, 3, PROMPT_TOKENS, gen_tokens, RUN_DEADLINE_S,
            t0, "orch-mtp-batched")
        draft_b = loop_b._mtp_draft  # noqa: SLF001 — test introspection
        rounds_b = loop_b._batched_verify_rounds  # noqa: SLF001
        print(f"[orch-mtp] batched   ({reason_b}): {bat_tokens} "
              f"drafts={draft_b.total_steps} "
              f"accepted={draft_b.total_accepted} "
              f"batched_rounds={rounds_b} "
              f"acceptance_rate={draft_b.acceptance_rate:.4f}", flush=True)
        assert reason_b == "length"

        # ── THE LOSSLESS GATE (INV-MTP-LOSSLESS via the Python loop) ─────
        assert bat_tokens == baseline, (
            "INV-MTP-LOSSLESS violated (batched verify): "
            f"{bat_tokens} != {baseline}")

        # ── Mechanics: the batched path actually ran + drafted ───────────
        assert rounds_b > 0, "batched verify never engaged (fallback?)"
        assert draft_b.total_steps > 0, "MTP loop proposed no drafts"
        # The MtpLossless C++ golden accepts >= 1 on this prompt; assert
        # the same bar so the batched round is real speculation.
        assert draft_b.total_accepted >= 1, (
            "batched MTP accepted nothing on the golden prompt")
    finally:
        layerstorm_engine.stop_engine()
        try:
            os.remove(cfg_path)
        except OSError:
            pass


DSPARK_GEN_TOKENS = 8   # same bar as Glm52GgufGolden.DsparkLossless


@pytest.mark.skipif(os.environ.get("ORCH_E2E_DSPARK") != "1",
                    reason="set ORCH_E2E_DSPARK=1 to run (real 3-GPU dspark "
                           "engine)")
@pytest.mark.skipif(not _GGUF.exists(), reason="GLM-5.2 GGUF not present")
@pytest.mark.skipif(not (_DSPARK_CKPT / "model.safetensors").exists(),
                    reason="DSpark speculator checkpoint not present")
@pytest.mark.skipif(not _dspark_gpus_free(),
                    reason="need TWO GPUs >= 26 GB free (TP=2) + a third "
                           ">= 11 GB free (draft)")
def test_orchestrator_real_engine_dspark_lossless():
    """TD-DSPARK-PY-LOOP-E2E: the Python DSpark loop vs the real engine.

    Phase A: plain-AR loop (speculation_depth=0) generates the baseline.
    Phase B: a second loop on the SAME engine with speculation_depth=γ and
    the DsparkDraft planner armed — _start_drafting dispatches ONE
    D_CMD_RUN_DSPARK_STEP per round, _handle_dspark_completion parses the
    sideband readback (ids + c_k), and the sequential early-stop verify
    feeds accepted tokens on the main sequence (INV-DSPARK-AUX keeps the
    draft context in lockstep; the pos-0 prompt feed re-arms it onto the
    new sequence).  Ring cursors live in the shared IPC block, so the
    second loop resumes exactly where the first stopped.

    GATES: token identity with the baseline (INV-DSPARK-LOSSLESS through
    the Python loop) + the drafter actually ran (rounds > 0, proposed > 0).
    Acceptance is REPORTED; accepted > 0 is asserted only under
    ORCH_E2E_DSPARK_REQUIRE_ACCEPT=1 — the toy 5-token prompt is OOD for
    the drafter (TD-DSPARK-ACCEPT-SHORTPROMPT: the realistic-prompt golden
    DsparkAcceptRealistic owns the in-distribution acceptance gate).
    """
    import layerstorm_engine
    from orchestrator.dspark_draft import DsparkDraftConfig
    from orchestrator.engine_glue import build_engine_loop
    from orchestrator.loop.orchestrator_loop import OrchestratorConfig

    # γ from the checkpoint (model-agnostic, mirrors the C++ fixture):
    # shipped ckpt block 8 / spec 7, glm-5.2-dspark-preview 16 / 15.
    with open(_DSPARK_CKPT / "config.json") as f:
        ckpt_cfg = json.load(f)
    block_size = int(ckpt_cfg["block_size"])
    spec_tokens = int(ckpt_cfg["speculators_config"]["proposal_methods"][0]
                      ["speculative_tokens"])

    # ── Config: the GLM52_DSPARK 3-GPU shape (TP pair + 5080 draft) ─────
    cfg = _base_engine_config()
    cfg["hardware"]["gpus"].append(
        {"id": 2, "type": "rtx5080", "vram_gb": 15,
         "pcie_gen": 5, "pcie_width": 16})
    cfg["speculation"]["method"] = "dspark"
    cfg["speculation"]["enabled"] = True
    cfg["speculation"]["dspark"] = {
        "checkpoint_path": str(_DSPARK_CKPT),
        "draft_gpus": [2],
        "block_size": block_size,
        "speculative_tokens": spec_tokens,
        "confidence_enabled": True,
    }
    cfg_path = "/tmp/orch_e2e_glm52_dspark_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # ── TD-DSPARK-SUPERCHUNK-CAPTURE regression knob ─────────────────────
    # ORCH_E2E_DSPARK_PROMPT_TOKENS=<n> (n > 64) swaps the toy prompt for
    # the first n canonical longctx prompt tokens, so the orchestrator's
    # DEFAULT superchunk prefill runs multiple 64-token sub-chunks per layer
    # (chunk-major aux capture).  Pre-fix, EVERY >64-token prompt silently
    # disabled drafting (the aux capture gapped/skipped sub-chunks >= 2), so
    # the proposed>0 gate below fails pre-fix.  The golden-next assertion
    # only applies to the toy prompt.
    prompt = list(PROMPT_TOKENS)
    golden_first: int | None = GOLDEN_NEXT
    n_prompt = int(os.environ.get("ORCH_E2E_DSPARK_PROMPT_TOKENS", "0") or 0)
    # ORCH_E2E_DSPARK_SUBCHUNK=<n> optionally shrinks the orchestrator's
    # prefill sub-chunk so the prompt spans MULTIPLE sub-chunks (chunk-major
    # aux capture) without needing a >64-token prompt — the >64-token
    # multi-sub-chunk regime currently trips the UNRELATED
    # TD-ORCH-ELM-COMPLETION-LIVELOCK first (e.g. PROMPT_TOKENS=61
    # SUBCHUNK=32 exercises the K=2 capture cleanly).
    n_subchunk = int(os.environ.get("ORCH_E2E_DSPARK_SUBCHUNK", "0") or 0)
    if n_prompt > 0:
        tok_file = (_PROJECT_ROOT / "test-data" / "prompts"
                    / "glm52_longctx_tokens3.txt")
        toks = [int(t) for t in tok_file.read_text().split()][:n_prompt]
        assert len(toks) == n_prompt, (
            f"prompt file has only {len(toks)} tokens, wanted {n_prompt}")
        prompt = toks
        golden_first = None
        print(f"[orch-dspark] long-prompt mode: {n_prompt} prompt tokens "
              f"(superchunk prefill, TD-DSPARK-SUPERCHUNK-CAPTURE"
              + (f"; sub-chunk {n_subchunk}" if n_subchunk else "") + ")",
              flush=True)

    print(f"\n[orch-dspark] starting 3-GPU dspark engine "
          f"(gamma={spec_tokens}/{block_size}, config={cfg_path}) ...",
          flush=True)
    t0 = time.monotonic()
    info = layerstorm_engine.start_engine(cfg_path)
    try:
        print(f"[orch-dspark] engine up in {time.monotonic() - t0:.0f}s: "
              f"layers={info.num_layers} experts={info.num_experts} "
              f"gpus={info.num_gpus}", flush=True)
        buffer_ids = layerstorm_engine.query_buffer_ids()

        # ── Phase A: baseline plain-AR loop (speculation off) ───────────
        _orch_kwargs = ({"prefill_subchunk_tokens": n_subchunk}
                        if n_subchunk > 0 else {})
        loop_a, handles_a = build_engine_loop(
            info,
            buffer_ids=buffer_ids,
            speculation_depth=0,
            orchestrator_config=OrchestratorConfig(cycle_budget_us=200.0,
                                                   **_orch_kwargs),
        )
        assert handles_a.metadata.hidden_buf_id != 0, "hidden buf not found"
        baseline, reason_a = _run_request(
            loop_a, 1, prompt, DSPARK_GEN_TOKENS, RUN_DEADLINE_S,
            t0, "orch-dspark-base")
        print(f"[orch-dspark] baseline ({reason_a}): {baseline}", flush=True)
        assert reason_a == "length"
        assert len(baseline) == DSPARK_GEN_TOKENS
        if golden_first is not None:
            assert baseline[0] == golden_first, (
                f"baseline produced {baseline[0]}, expected {golden_first}")

        # ── Phase B: DSpark loop on the same engine ──────────────────────
        loop_b, _ = build_engine_loop(
            info,
            buffer_ids=buffer_ids,
            speculation_depth=spec_tokens,
            dspark_config=DsparkDraftConfig(
                enabled=True,
                block_size=block_size,
                speculative_tokens=spec_tokens,
                confidence_enabled=True,
            ),
            orchestrator_config=OrchestratorConfig(cycle_budget_us=200.0,
                                                   **_orch_kwargs),
        )
        spec, reason_b = _run_request(
            loop_b, 2, prompt, DSPARK_GEN_TOKENS, RUN_DEADLINE_S,
            t0, "orch-dspark-spec")
        print(f"[orch-dspark] dspark   ({reason_b}): {spec}", flush=True)
        assert reason_b == "length"
        assert len(spec) == DSPARK_GEN_TOKENS

        # ── THE LOSSLESS GATE (INV-DSPARK-LOSSLESS via the Python loop) ──
        assert spec == baseline, (
            "INV-DSPARK-LOSSLESS violated through the Python loop: "
            f"dspark {spec} != baseline {baseline}")

        # ── Drafter mechanics + acceptance report ────────────────────────
        draft = loop_b._dspark_draft  # noqa: SLF001 — test introspection
        print(f"[orch-dspark] rounds={draft.total_rounds} "
              f"proposed={draft.total_proposed} "
              f"accepted={draft.total_accepted} "
              f"acceptance_rate={draft.acceptance_rate:.4f}", flush=True)
        assert draft.total_rounds > 0, "DSpark loop never drafted a round"
        assert draft.total_proposed > 0, "DSpark loop proposed no tokens"
        if os.environ.get("ORCH_E2E_DSPARK_REQUIRE_ACCEPT") == "1":
            assert draft.total_accepted > 0, (
                "DSpark accepted nothing (toy prompt is OOD — see "
                "TD-DSPARK-ACCEPT-SHORTPROMPT)")
    finally:
        layerstorm_engine.stop_engine()
        try:
            os.remove(cfg_path)
        except OSError:
            pass

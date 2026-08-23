"""V4-mode orchestrator tests (ticket I; chunked prefill 2026-08-21).

deepseek_v4 boots the successor orchestrator ARCHITECTURE-AWARE:
split/ACT command flow (use_far=False, route_arm "act" — the ticket-H
golden harness shape), prefix cache OFF, first_moe_layer 0
(all-MoE: the bridge's dense branch never fires), CHUNKED prefill at
512-token chunks (TD-V4-CHUNK-PREFILL resolved — the engine's V4 chunk
arm; the teacher-forced per-token feed is retained as a debug arm),
200 s expert-fetch deadline.

Driven against the scripted FakeDaemon from test_bridge_ring — no CUDA.
"""

from __future__ import annotations

import json

from test_bridge_ring import (  # the scripted-daemon harness
    NUM_LAYERS,
    VOCAB,
    FakeInfo,
    _finish,
    _make,
    chain,
)

from orchestrator.orchestrator import (
    InferenceRequest,
    Orchestrator,
    PrefixCacheConfig,
    SpeculationConfig,
)
from orchestrator.types import EngineMetadata


def _meta(eos: tuple[int, ...] = ()) -> EngineMetadata:
    return EngineMetadata(
        num_gpus=4, num_moe_layers=NUM_LAYERS, num_experts=64,
        num_layers=NUM_LAYERS, expert_bytes=0, kv_bytes_per_page=0,
        eos_token_ids=eos, vocab_size=VOCAB, moe_batch_capacity=512)


def _v4_orch(eos: tuple[int, ...] = (), *, teacher_forced: bool = False,
             superchunk: bool = False, prefill_chunk: int = 512):
    """Orchestrator wired the way Orchestrator.boot wires deepseek_v4.

    ``teacher_forced=True`` exercises the RETAINED per-token debug arm
    (the ticket-H lock-step shape); the boot default is superchunk
    prefill (SC port — sub-chunked attention + one MOE_BIG per layer);
    ``superchunk=False`` keeps the P1 chunked path for its own tests."""
    bridge, daemon, _ = _make(use_far=False, route_arm="act")
    bridge.first_moe_layer = 0            # V4-Flash is all-MoE
    bridge.decode_timeout_us = 200_000_000
    orch = Orchestrator(
        bridge, metadata=_meta(eos),
        speculation=SpeculationConfig(),   # plain-decode harness (spec-on
                                           # V4 serving rides ticket J's
                                           # micro-chunk verify engine-side)
        prefix_cache=PrefixCacheConfig(enabled=False),
        teacher_forced_prefill=teacher_forced,
        prefill_chunk=prefill_chunk,
        prefill_superchunk=superchunk)
    return orch, daemon


class _Sink:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.done: tuple | None = None

    def on_token(self, rid: int, tok: int, logp) -> None:
        self.tokens.append(tok)

    def on_complete(self, rid: int, tokens: list[int], reason: str,
                    logp) -> None:
        assert self.done is None
        self.done = (rid, list(tokens), reason)


def _serve(orch: Orchestrator, req: InferenceRequest) -> None:
    orch.submit_request(req)
    assert orch._serve_next() is True


def test_v4_chunked_prompt_feed_lossless():
    prompt = chain(11, 5)
    orch, daemon = _v4_orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=6,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length"
        # Lossless: the same greedy chain the teacher-forced arm produces.
        assert tokens == chain(prompt[-1], 6)
        assert sink.tokens == tokens
        assert daemon.far_layers == 0
        assert daemon.reef_routes == 0
        assert daemon.dense_moes == 0
        # Chunked prefill: ONE multi-row embed for prompt[:-1], then one
        # single-token embed per decode step.
        assert daemon.embed_calls == [len(prompt) - 1] + [1] * 6
        # One expert-union FETCH_AND_RUN per layer for the chunk, then one
        # per layer per decode step.
        assert daemon.fetch_moes == NUM_LAYERS * (1 + 6)
    finally:
        _finish(daemon)


def test_v4_superchunk_prompt_feed_lossless():
    # SC (superchunk port): the boot-default V4 arm — sub-chunked
    # embedding/attention at row_offset + ONE FETCH_AND_RUN_MOE_BIG per
    # layer over the whole prompt body. Same greedy chain, MOE_BIG rows =
    # the full body, per-layer union validated inside the FakeDaemon.
    prompt = chain(11, 8)                  # body = 7 rows
    orch, daemon = _v4_orch(superchunk=True, prefill_chunk=2)
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=5,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length"
        assert tokens == chain(prompt[-1], 5)
        assert sink.tokens == tokens
        assert daemon.far_layers == 0
        assert daemon.reef_routes == 0
        assert daemon.dense_moes == 0
        # Superchunk: embedding sub-chunks (2,2,2,1) then one per decode.
        assert daemon.embed_calls == [2, 2, 2, 1] + [1] * 5
        # ONE MOE_BIG per layer over the whole 7-row body...
        assert daemon.fetch_moe_bigs == NUM_LAYERS
        assert daemon.moe_big_rows == [7] * NUM_LAYERS
        # ...and plain FETCH_AND_RUN only for the decode steps.
        assert daemon.fetch_moes == NUM_LAYERS * 5
    finally:
        _finish(daemon)


def test_v4_superchunk_splits_at_capacity():
    # Body larger than moe_batch_capacity → multiple superchunks.
    prompt = chain(7, 12)                  # body = 11 rows
    orch, daemon = _v4_orch(superchunk=True, prefill_chunk=3)
    orch.bridge.moe_batch_capacity = 6     # force a 6+5 superchunk split
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=9, prompt_token_ids=prompt, max_tokens=2,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, _ = sink.done
        assert tokens == chain(prompt[-1], 2)
        # Superchunk 1: rows 6 (subs 3,3); superchunk 2: rows 5 (subs 3,2).
        assert daemon.embed_calls == [3, 3, 3, 2] + [1] * 2
        assert daemon.moe_big_rows == [6] * NUM_LAYERS + [5] * NUM_LAYERS
    finally:
        _finish(daemon)


def test_v4_teacher_forced_prompt_feed_lossless():
    # The RETAINED debug arm (teacher_forced_prefill=True).
    prompt = chain(11, 5)
    orch, daemon = _v4_orch(teacher_forced=True)
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=6,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "length"
        # Same greedy chain the chunked-prefill path would produce.
        assert tokens == chain(prompt[-1], 6)
        assert sink.tokens == tokens
        # Split/ACT arm only: no FAR fused commands, no REEF service,
        # no dense-MoE branch (first_moe_layer 0 → all layers routed).
        assert daemon.far_layers == 0
        assert daemon.reef_routes == 0
        assert daemon.dense_moes == 0
        # Teacher-forced feed: every embed is a single token — one per
        # prompt-feed step (len-1) plus one per decode step.
        assert daemon.embed_calls == [1] * (len(prompt) - 1 + 6)
        # Every step ran FETCH_AND_RUN_MOE on all layers.
        assert daemon.fetch_moes == NUM_LAYERS * (len(prompt) - 1 + 6)
    finally:
        _finish(daemon)


def test_v4_single_token_prompt_no_prefill_steps():
    orch, daemon = _v4_orch()
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=2, prompt_token_ids=[4321], max_tokens=3,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert tokens == chain(4321, 3)
        assert daemon.embed_calls == [1, 1, 1]     # decode steps only
    finally:
        _finish(daemon)


def test_v4_eos_stop():
    ref = chain(11, 8)
    orch, daemon = _v4_orch(eos=(ref[4],))
    try:
        sink = _Sink()
        _serve(orch, InferenceRequest(
            request_id=3, prompt_token_ids=[11], max_tokens=0,
            on_token=sink.on_token, on_complete=sink.on_complete))
        _, tokens, reason = sink.done
        assert reason == "stop"
        assert tokens == ref[:5]
    finally:
        _finish(daemon)


# ---------------------------------------------------------------------------
# Boot-level architecture awareness (fake engine module, no daemon —
# boot only constructs; no ring traffic).
# ---------------------------------------------------------------------------


class _FakeEngineModule:
    def __init__(self, info: FakeInfo) -> None:
        self._info = info
        self.stopped = False

    def start_engine(self, config_path: str):
        return self._info

    def query_buffer_ids(self):
        return {"hidden_state.attn.rank0": 7, "logits_scratch.pos0": 9}

    def stop_engine(self) -> None:
        self.stopped = True


_V4_BOOT_CFG = {
    "model": {"architecture": "deepseek_v4", "vocab_size": 129280,
              "first_k_dense_replace": 0},
    # Ticket J: a configured dspark method arms V4 speculation (the engine
    # runs the dflash draft; verify rides the V4 micro-chunk arm). FAR/REEF/
    # prefix cache stay forced OFF for V4 regardless.
    "speculation": {"enabled": True, "method": "dspark",
                    "dspark": {"speculative_tokens": 4}},
    "gpu_loader": {"enabled": True},
    "serving": {"prefix_cache": {"enabled": True}},
}


def _boot(cfg: dict, tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return Orchestrator.boot(str(p), engine_module=_FakeEngineModule(
        FakeInfo()))


def test_boot_deepseek_v4_arch_awareness(tmp_path):
    orch = _boot(_V4_BOOT_CFG, tmp_path)
    b = orch.bridge
    assert b.use_far is False
    assert b.route_arm == "act"
    assert b.first_moe_layer == 0
    assert b.decode_timeout_us == 200_000_000
    assert b.vocab_size == 129280
    # Ticket J: V4 speculation follows the config (dspark armed above).
    assert orch.spec.enabled is True and orch.spec.gamma == 4
    # TD-V4-SERVE-PREFIX resolved: fork clones complete V4 state, so the
    # prefix cache is enabled for V4 (superchunk-grid registration).
    assert orch.prefix_cache is not None
    # TD-V4-CHUNK-PREFILL resolved: V4 boots chunked (512-token chunks).
    assert orch._teacher_forced_prefill is False
    assert orch._prefill_chunk == 512
    # SC (superchunk port): V4 boots superchunk the prompt body (MOE_BIG
    # per layer over up to moe_batch_capacity rows, 512-row sub-chunks).
    assert orch._prefill_superchunk is True
    # TD-V4-SPEC-PREFILL-CTX RESOLVED: the engine fires the final aux tap
    # at the last layer's MoE finalize for headless chunks, so V4 chunked
    # prefill arms the dflash draft context like GLM.
    assert orch._chunk_prefill_arms_draft is True


def test_boot_deepseek_v4_spec_off_without_dspark(tmp_path):
    cfg = json.loads(json.dumps(_V4_BOOT_CFG))
    cfg["speculation"] = {"enabled": False}
    orch = _boot(cfg, tmp_path)
    assert orch.spec.enabled is False
    assert orch.bridge.use_far is False and orch.bridge.route_arm == "act"


def test_boot_deepseek_v4_spec_depth_zero_forces_plain(tmp_path):
    orch = _boot(_V4_BOOT_CFG, tmp_path)
    assert orch.spec.enabled is True
    p = tmp_path / "cfg2.json"
    p.write_text(json.dumps(_V4_BOOT_CFG))
    from orchestrator.orchestrator import Orchestrator as O
    orch2 = O.boot(str(p), engine_module=_FakeEngineModule(FakeInfo()),
                   speculation_depth=0)
    assert orch2.spec.enabled is False


def test_boot_glm_shape_unchanged(tmp_path):
    cfg = json.loads(json.dumps(_V4_BOOT_CFG))
    cfg["model"]["architecture"] = "glm_moe_dsa"
    cfg["model"]["first_k_dense_replace"] = 3
    orch = _boot(cfg, tmp_path)
    b = orch.bridge
    assert b.use_far is True
    assert b.route_arm == "reef"          # gpu_loader.enabled
    assert b.first_moe_layer == 3
    assert b.decode_timeout_us == 5_000_000
    assert orch.spec.enabled is True and orch.spec.gamma == 4
    assert orch.prefix_cache is not None
    assert orch._teacher_forced_prefill is False
    assert orch._prefill_chunk == 64      # GLM attention sub-chunk rows
    assert orch._chunk_prefill_arms_draft is True   # GLM chunks arm drafts
    # GLM mini-superchunks DEFAULT ON since the 2026-08-23 green light
    # (TD-SERVE-SC-TRAJECTORY): stride bounded at the single-shot MoE
    # chunk capacity — max(moe_big_chunk_tokens [512], max_batch_size).
    assert orch._prefill_superchunk is True
    assert orch._prefill_sc_stride == 512


def test_boot_glm_superchunk_kill_switch(tmp_path, monkeypatch):
    # LS_ORCH_NO_SC=1 restores the chunk-64 prefill path (and the
    # pre-flip served trajectory) — the TD-SERVE-SC-TRAJECTORY contract.
    monkeypatch.setenv("LS_ORCH_NO_SC", "1")
    cfg = json.loads(json.dumps(_V4_BOOT_CFG))
    cfg["model"]["architecture"] = "glm_moe_dsa"
    cfg["model"]["first_k_dense_replace"] = 3
    orch = _boot(cfg, tmp_path)
    assert orch._prefill_superchunk is False
    assert orch._prefill_chunk == 64


def test_metadata_attention_type_for_layer():
    # V4-8: per-layer attention type helper from EngineInfo export.
    m = _meta()
    assert m.attention_type_for_layer(0) == 0          # non-V4: empty
    import dataclasses
    m2 = dataclasses.replace(m, v4_attention_types=(0, 0, 1, 2, 1))
    assert [m2.attention_type_for_layer(i) for i in range(6)] == \
        [0, 0, 1, 2, 1, 0]

"""dsp52 through the raw C++/Python ring bridge — NO orchestrator.

Reproduces ``Dsp52Test.SpeculativeHundredTokenDecodeFetchAndRun_FullFit_
EP4_GLM52`` (tests/integration/dsp52_test.cpp — keeper52 + DSpark
speculative decoding) driven end-to-end from Python over the SPSC
command/completion rings + sideband region (``python/bridge``),
deliberately bypassing the Python orchestrator loop
(``orchestrator_loop``/``engine_glue``): this bridge layer is the intended
basis for its successor.  The only orchestrator import is the pure ctypes
protocol mirror (via ``bridge.protocol``).

THE DEFAULT RUN IS THE CHAMPION ARM (user decision 2026-08-18): batched
verify + overlap, conf 0.1, γ from the preview ckpt (15), nvfp4 draft
sharded on the 5090s, prompt-fed tokens3@512 + teacher-forced r1
trajectory, daemon-side I8-ACT placement, sharded KV + TQ + tiering, and
whatever the GLM-5.2 config json defaults carry (M3 arena placement table,
``speculation.dspark.verify_chunk_prefetch``).  Bare invocation::

  DSP52_PY=1 .venv/bin/python -m pytest \
      tests/integration/test_dsp52_bridge.py -s

measures the same arm-corpus as the C++ champion runs (~10.4 tok/s C++;
the bridge carries a measured host-overhead deficit — see
scratchpad/GAP_EXPERIMENTS_9_1.md era numbers).  The C++ test's
non-champion arms (plain DSP52_SPEC=0 loop, sequential verify, the
test-side LRU/affinity/REEF-orch placement models, ELB/UPART/oracle/REF
diagnostics, KV/backend bisect overrides) are deliberately NOT exposed
here — the bridge library (``bridge.spec_decode``) still implements
plain/seq arms and they stay covered by tests/unit/test_bridge_ring.py.

Hot path: the Cython module ``bridge._fastbridge`` accelerates ring
writes, the completion poll and the sideband bulk stores when built
(``python/bridge/build_fastbridge.py``); the test prints which path is
live.  ``LS_BRIDGE_NO_CYTHON=1`` forces the pure-ctypes fallback.

GATES:
  * rounds > 0, no NaN top1_prob/entropy, every command status 0
    (CMP_ERROR raises), all tokens in-vocab.
  * DSP52_PY_REF=<file> — token-identity "happy match" against a C++
    reference trajectory recorded with the DSP52_DUMP_TOKENS knob of
    dsp52_test.cpp on a MATCHED recipe.
  * DSP52_PY_REF_STATS=<acc>,<tau>,<hit> — optional numeric fingerprints
    (hit is placement-arm-specific; "-" skips a field).

Champion-path knobs (same semantics as dsp52_test.cpp):
  KEEPER52_TOKENS=<n>          decode length (default 100)
  LS_KEEPER_SEED_TOKEN=<t>     bare-seed token (default 15234; only used
                               when the prompt is disabled)
  DSP52_PROMPT=<file|none>     prompt token file (default the canonical
                               corpus test-data/prompts/
                               glm52_longctx_tokens3.txt; "none"/"off"/"0"
                               = bare-seed)
  DSP52_PROMPT_TOKENS=<n>      prompt cap (default 512; 0 = whole file)
  DSP52_FORCE_TRAJ=<file|none> teacher-forced trajectory (default
                               test-data/prompts/dsp52_forced_traj_r1.txt;
                               "none"/"off"/"0" = free-run commits)
  DSP52_GAMMA=<n>              effective draft length (<= ckpt gamma)
  DSP52_CONF_THRESH=<f>        DSP-9 truncation (default 0.1 = champion)
  DSP52_CKPT=<dir>             DSpark checkpoint override
  DSP52_QUANT=<fmt>            draft requant (default nvfp4 = champion)
  DSP52_DRAFT_GPU=<p[,p]>      draft host position(s) (default 0,1 = the
                               TP=2 shard on the 5090s)
  DSP52_PREFETCH=1|0           lever-4 OVERRIDE (source of truth = config
                               speculation.dspark.verify_chunk_prefetch,
                               default false)
  KEEPER52_EXPERT_CACHE_GB / KEEPER52_EXPERT_CACHE_GB_5080 /
  KEEPER52_STABLE_FRAC / KEEPER52_ARENA_FRACTION /
  GLM52_KV_TIERING / GLM52_KVT_RATIO / GLM52_ARENA_ATTACH /
  LS_LOADER_CALIB              config-input parameterization (json fields)
  DSP52_DUMP_TOKENS=<file>     dump THIS run's committed trajectory
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import time

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_CONFIG = _PROJECT_ROOT / "test-data" / "config" / "glm_5_2_gguf.json"
_GGUF = (_PROJECT_ROOT / "test-data" / "GLM-5.2-GGUF-Q4_K_XL"
         / "GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf")
_PREPACKED = _PROJECT_ROOT / "test-data" / "GLM-5.2-prepacked"
_DSPARK_CKPT = _PROJECT_ROOT / "test-data" / "GLM-5.2-speculator.dspark"
_PROMPT_DEFAULT = (_PROJECT_ROOT / "test-data" / "prompts"
                   / "glm52_longctx_tokens3.txt")
_FORCE_TRAJ_DEFAULT = (_PROJECT_ROOT / "test-data" / "prompts"
                       / "dsp52_forced_traj_r1.txt")

SEED_TOKEN = 15234
NUM_TOKENS = 100
_OFF = ("none", "off", "0")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v:
        try:
            return float(v)
        except ValueError:
            pass
    return default


def _env_path(name: str, default: pathlib.Path) -> pathlib.Path | None:
    """Champion-fixture path knob: unset → default; 'none'/'off'/'0' →
    disabled; anything else → that path."""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if v.lower() in _OFF:
        return None
    return pathlib.Path(v)


def _dsp52_prefetch(cfg: dict) -> bool:
    """Lever-4 verify-chunk prefetch (dsp52_test.cpp dsp52_prefetch() parity).

    Source of truth = the engine config field
    speculation.dspark.verify_chunk_prefetch (schema default false: measured
    -1.0% champion wall, and the perfect-predictor oracle loses too on this
    DMA-bound box). The test-interface env DSP52_PREFETCH OVERRIDES it when
    set: non-"0" forces ON, "0" forces OFF; unset defers to the config.
    """
    v = os.environ.get("DSP52_PREFETCH")
    if v:
        return v != "0"
    return bool(cfg.get("speculation", {}).get("dspark", {})
                .get("verify_chunk_prefetch", False))


def _probe_gpus() -> list[tuple[int, int, int]]:
    """[(nvidia-smi index, total MiB, free MiB)] — PCI_BUS_ID order.

    With CUDA_DEVICE_ORDER=PCI_BUS_ID the nvidia-smi index IS the CUDA
    ordinal (test_keeper52_reef.py contract) — no CUDA touch from Python.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return gpus


def _ep4_gpus_ok() -> bool:
    srt = sorted(_probe_gpus(), key=lambda g: g[1], reverse=True)
    if len(srt) < 4:
        return False
    return (srt[1][1] >= 28 * 1024 and srt[1][2] >= 28_000
            and srt[3][1] >= 14 * 1024 and srt[3][2] >= 14_000)


def _dsp52_ckpt_dir() -> pathlib.Path:
    ck = os.environ.get("DSP52_CKPT")
    return pathlib.Path(ck) if ck else _DSPARK_CKPT


def _dsp52_draft_gpus() -> list[int]:
    """Champion default: the TP=2 nvfp4 shard on both 5090s (positions
    0,1 — ≡ the C++ DSP52_SHARD=1 / DSP52_DRAFT_GPU=0,1 arm)."""
    e = os.environ.get("DSP52_DRAFT_GPU")
    if not e:
        return [0, 1]
    out: list[int] = []
    for tok in e.split(","):
        tok = tok.strip()
        if tok:
            p = int(tok)
            if 0 <= p <= 3 and p not in out:
                out.append(p)
    return out or [0, 1]


def _load_token_file(path: str | pathlib.Path) -> list[int]:
    with open(path) as f:
        raw = f.read().strip()
    try:
        arr = json.loads(raw)
        return [int(x) for x in arr]
    except (json.JSONDecodeError, TypeError, ValueError):
        return [int(x) for x in raw.replace(",", " ").split()]


def _build_dsp52_config() -> tuple[dict, int, int]:
    """Faithful port of dsp52_test.cpp build_dsp52_config (EP=4 GPU arm,
    champion deltas only). Starts from the GLM-5.2 default json — every
    field the champion does not override (arena placement table,
    verify_chunk_prefetch, indexer mode, ...) is inherited from there.

    Returns (config, ckpt_gamma, ckpt_block)."""
    with open(_CONFIG) as f:
        cfg = json.load(f)

    cfg["model"]["weights_path"] = str(_GGUF)
    cfg["quantization"]["gguf_strategy"] = "int"

    # ── EP=4 hardware block (INV-MOE-EP-XTP), 5090s-first ────────────────
    expert_cache_gb = _env_float("KEEPER52_EXPERT_CACHE_GB", 4.5)
    expert_cache_gb_5080 = _env_float("KEEPER52_EXPERT_CACHE_GB_5080", 11.0)
    stable_frac = _env_float("KEEPER52_STABLE_FRAC", 0.95)

    quant = os.environ.get("DSP52_QUANT", "nvfp4") or "nvfp4"
    draft_positions = _dsp52_draft_gpus()
    sharded_draft = len(draft_positions) > 1
    # Draft-host expert windows: the proven per-card occupancies from
    # TD-DSPARK-DRAFT-QUANT/-SHARD (dsp52_test.cpp build_dsp52_config).
    if sharded_draft:
        draft_cache_gb = {"fp8_e4m3": 2.5, "nvfp4": 3.25}.get(quant, 1.25)
    elif draft_positions[0] < 2:
        draft_cache_gb = {"fp8_e4m3": 0.5, "nvfp4": 1.75}.get(quant, 4.0)
    else:
        draft_cache_gb = 4.0

    probed = sorted(_probe_gpus(), key=lambda g: g[1], reverse=True)
    assert len(probed) >= 4, "EP=4 needs 4 GPUs"
    gpus = []
    for pos in range(4):
        big = pos < 2
        g: dict = {
            "id": probed[pos][0],
            "type": "rtx5090" if big else "rtx5080",
            "vram_gb": 30 if big else 15,
            "pcie_gen": 5, "pcie_width": 16,
            "roles": (["attention", "resident", "expert_streaming"]
                      if big else ["expert_streaming"]),
        }
        hosts_draft = pos in draft_positions
        win = (draft_cache_gb if hosts_draft
               else (expert_cache_gb if big else expert_cache_gb_5080))
        if win > 0.0:
            g["vram_allocation_gb"] = {
                "expert_streaming": win,
                "stable_zone_fraction": stable_frac,
            }
        gpus.append(g)
    cfg["hardware"]["gpus"] = gpus
    cfg["hardware"]["tp_array"] = [0, 1]
    cfg["parallelism"]["tensor_parallelism"] = 2

    # ── DSpark speculation block (ckpt-derived gamma/block) ──────────────
    ckpt = _dsp52_ckpt_dir()
    spec = cfg.setdefault("speculation", {})
    spec["method"] = "dspark"
    spec["enabled"] = True
    dsp = spec.setdefault("dspark", {})
    dsp["checkpoint_path"] = str(ckpt)
    dsp["confidence_enabled"] = _env_float("DSP52_CONF_THRESH", 0.1) > 0.0
    dsp["draft_gpus"] = draft_positions
    if quant != "bf16":
        dsp["draft_weights_quant"] = quant
    gamma, block = 7, 8
    try:
        with open(ckpt / "config.json") as f:
            cj = json.load(f)
        if "block_size" in cj:
            dsp["block_size"] = cj["block_size"]
            block = int(cj["block_size"])
        pm = (cj.get("speculators_config", {})
              .get("proposal_methods") or [])
        if pm and "speculative_tokens" in pm[0]:
            dsp["speculative_tokens"] = pm[0]["speculative_tokens"]
            gamma = int(pm[0]["speculative_tokens"])
    except OSError:
        pass  # engine init fails loud on a broken ckpt anyway

    # ── keeper52 feature deltas (KVS + TQ + tiering) — the champion shape;
    # the GLM json base is deliberately vanilla (snapmla, no tiering). ────
    cfg["hardware"]["dcp_kv_mode"] = "sharded"
    cfg["hardware"]["dcp_indexer_mode"] = "replicated"
    cfg["memory"]["kv_cache"]["dcp_chunk_size"] = 16
    cfg["compute"]["attention_backend"] = "turboquant_mla"
    hot_slots = int(_env_float("GLM52_KV_TIERING", 512))
    cold_ratio = _env_float("GLM52_KVT_RATIO", 16.0)
    cfg["memory"]["kv_tiering"] = {
        "enabled": True,
        "hot_buffer_slots": hot_slots if hot_slots > 0 else 512,
        "host_to_device_ratio": cold_ratio if cold_ratio >= 1.0 else 16.0,
    }
    cfg["memory"]["vram_safety_margin_gb"] = 3.0
    cfg["memory"]["kv_cache"]["max_pages_per_gpu"] = 4096
    cfg["memory"]["kv_cache"]["page_growth_chunk_tokens"] = 16

    # ── full-fit O_DIRECT arena (arena_attach default ON) ────────────────
    arena_frac = _env_float("KEEPER52_ARENA_FRACTION", 0.90)
    if not (0.0 < arena_frac <= 1.0):
        arena_frac = 0.90
    cfg.setdefault("preprocessing", {})["prepacked_dir"] = str(_PREPACKED)
    cfg["memory"]["preload_expert_buffers"] = True
    cfg["memory"]["pin_host_expert_pool"] = True
    cfg["memory"]["pin_host_expert_pool_direct_load"] = True
    # persist (2026-08-18 post-incident): a mismatched boot fails loud
    # instead of wiping the holder store; GLM52_ARENA_PERSIST=0 opts a run
    # out for an intentional wipe + cold rebuild.
    cfg["memory"]["arena_attach"] = {
        "enabled": os.environ.get("GLM52_ARENA_ATTACH") != "0",
        "persist": os.environ.get("GLM52_ARENA_PERSIST") != "0"}
    cfg["memory"]["pin_host_expert_pool_preload"] = True
    cfg["memory"]["pin_host_expert_pool_direct_o_direct"] = True
    cfg["memory"]["pin_host_expert_pool_sizing"] = {
        "mode": "fraction_total", "value": arena_frac}
    cfg["memory"]["cross_node_spill"] = {
        "enabled": True,
        "nodes": [{"node": n, "weight": 1} for n in (0, 1, 4, 5, 6, 7)],
        "sizing_mode": "fraction_total",
        "sizing_value": arena_frac,
        "per_node": [{"node": n, "mode": "fraction_free", "value": 0.80}
                     for n in (4, 5, 6, 7)],
    }

    # ── I8 loader wiring (the ACT arm env is set in _setup_env) ──────────
    cfg["gpu_loader"] = {
        "enabled": True,
        "calibration_path": os.environ.get(
            "LS_LOADER_CALIB", "gpu_loader_calibration_ep4x4.json"),
    }
    return cfg, gamma, block


def _setup_env() -> None:
    """Pre-boot env contract (no-overwrite, C++ setenv(...,0) parity).

    Placement arm = REEF over IPC (E_CMD_REEF_ROUTE: the daemon-hosted
    ReefOrch service solves placement + fills the 13c-2.0 victim map;
    FETCH ships with have_evict_map=1 — the C++ champion REEF-arm
    contract). Like the C++ KEEPER52_REEF_ORCH arm, the I8 shadow is OFF
    (the REEF service owns placement). IPC-region pinning (DSP52_BOOST
    lever 2) is no longer exported here — it is config default-on
    (compute.ipc_pin); LS_IPC_PIN stays available as an override."""
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "LS_LOADER_SHADOW": "0",
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE": "1",
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION": "bf16",
        "GLM52_ARENA_ATTACH": "1",
        "LS_LOADER_CALIB": "gpu_loader_calibration_ep4x4.json",
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)


_setup_env()


@pytest.mark.skipif(os.environ.get("DSP52_PY") != "1",
                    reason="set DSP52_PY=1 to run (4-GPU GLM-5.2 dsp52)")
@pytest.mark.skipif(not _CONFIG.exists(), reason="GLM-5.2 config missing")
@pytest.mark.skipif(not _GGUF.exists(), reason="GLM-5.2 GGUF not present")
@pytest.mark.skipif(not (_PREPACKED / "manifest.json").exists(),
                    reason="GLM-5.2 prepacked set not present")
@pytest.mark.skipif(not (_dsp52_ckpt_dir() / "config.json").exists(),
                    reason="DSpark checkpoint not present")
@pytest.mark.skipif(not _ep4_gpus_ok(),
                    reason="EP=4 needs 4 idle GPUs (2x >=28 GiB + 2x >=14)")
def test_dsp52_bridge_speculative_hundred_token_decode():
    import layerstorm_engine
    from bridge.ring_bridge import EngineBridge, fastbridge_active
    from bridge.spec_decode import SpecStats, run_speculative_loop

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        pytest.skip("dsp52 bridge requires CUDA_VISIBLE_DEVICES=0,1,2,3 "
                    "(nvidia-smi index == CUDA ordinal contract)")
    if os.environ.get("KEEPER52_EP", "4") != "4":
        pytest.skip("dsp52 requires the EP=4 shape")

    num_tokens = NUM_TOKENS
    nt = os.environ.get("KEEPER52_TOKENS")
    if nt and nt.isdigit() and int(nt) > 0:
        num_tokens = int(nt)
    seed_token = SEED_TOKEN
    st_env = os.environ.get("LS_KEEPER_SEED_TOKEN")
    if st_env and st_env.isdigit():
        seed_token = int(st_env)

    conf_thresh = _env_float("DSP52_CONF_THRESH", 0.1)

    # Champion corpus defaults (env overrides; "none" disables).
    prompt: list[int] | None = None
    prompt_path = _env_path("DSP52_PROMPT", _PROMPT_DEFAULT)
    if prompt_path is not None:
        prompt = _load_token_file(prompt_path)
        cap = int(_env_float("DSP52_PROMPT_TOKENS", 512))
        if 0 < cap < len(prompt):
            prompt = prompt[:cap]
        assert len(prompt) >= 2, "DSP52_PROMPT too short"
    forced: list[int] | None = None
    forced_path = _env_path("DSP52_FORCE_TRAJ", _FORCE_TRAJ_DEFAULT)
    if forced_path is not None:
        forced = _load_token_file(forced_path)
        assert forced, "DSP52_FORCE_TRAJ empty"

    cfg, ckpt_gamma, ckpt_block = _build_dsp52_config()
    # Lever 4: config-first (speculation.dspark.verify_chunk_prefetch),
    # DSP52_PREFETCH overrides — resolved after the config is built.
    prefetch = _dsp52_prefetch(cfg)
    vocab_size = int(cfg["model"]["vocab_size"])
    first_moe_layer = int(cfg["model"].get("first_k_dense_replace", 3))
    gamma = ckpt_gamma
    g_env = int(_env_float("DSP52_GAMMA", 0))
    if g_env > 0:
        gamma = min(g_env, ckpt_gamma)
    if prompt:
        for t in prompt:
            assert 0 <= t < vocab_size, "DSP52_PROMPT id out of vocab"
    if forced:
        for t in forced:
            assert 0 <= t < vocab_size, "DSP52_FORCE_TRAJ id out of vocab"

    cfg_path = "/tmp/dsp52_bridge_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[dsp52-bridge] EP=4 boot (champion arm, ckpt gamma="
          f"{ckpt_gamma}/block {ckpt_block}, config={cfg_path}) ...",
          flush=True)
    t0 = time.monotonic()
    info = layerstorm_engine.start_engine(cfg_path)
    try:
        print(f"[dsp52-bridge] engine up in {time.monotonic() - t0:.0f}s: "
              f"layers={info.num_layers} experts={info.num_experts} "
              f"gpus={info.num_gpus} moe_cap={info.moe_batch_capacity}",
              flush=True)
        assert info.num_gpus == 4, "EP=4 engine must expose 4 GPUs"

        buffer_ids = layerstorm_engine.query_buffer_ids()
        hidden_buf = logits_buf = 0
        for name, bid in buffer_ids.items():
            if name.startswith("hidden_state.attn.rank0"):
                hidden_buf = int(bid)
            elif name.startswith("logits_scratch.pos0"):
                logits_buf = int(bid)
        assert hidden_buf and logits_buf, "hidden/logits buffers not found"

        # route_arm "reef" (default): the daemon ReefOrch service + victim
        # map. Fused E_CMD_FAR_FORWARD_LAYER is the DEFAULT layer driver
        # (9.76-9.83 vs 9.42 split-path); BRIDGE_FAR=0 falls back to the
        # split REEF_ROUTE+FETCH chain.
        # BRIDGE_BURST=0 disables the pipelined FAR sweep (send-all,
        # collect-in-order) for A/B against the serial per-layer form.
        bridge = EngineBridge(info, vocab_size=vocab_size,
                              first_moe_layer=first_moe_layer,
                              hidden_buf_id=hidden_buf,
                              logits_buf_id=logits_buf,
                              use_far=os.environ.get("BRIDGE_FAR",
                                                     "1") != "0",
                              far_burst=os.environ.get("BRIDGE_BURST",
                                                       "1") != "0")
        print(f"[dsp52-bridge] fastbridge (Cython) "
              f"{'ACTIVE' if fastbridge_active() else 'absent — '
                 'pure-ctypes fallback'}", flush=True)

        h2d_state = {"base": 0}

        def snap_h2d():
            h2d_state["base"] = layerstorm_engine.h2d_path_stats()[
                "phase_count"]

        print(f"\n=== dsp52-bridge SpeculativeHundredTokenDecode "
              f"(GLM-5.2 GGUF EP=4 CHAMPION: batched+overlap gamma={gamma} "
              f"conf={conf_thresh} prefetch={int(prefetch)} "
              f"prompt={'y' if prompt else 'bare'} "
              f"forced={'y' if forced else 'free'}, I8-ACT arm, "
              f"FETCH_AND_RUN_MOE, TQ + sharded-KV + tiering) ===",
              flush=True)

        st = SpecStats()
        committed = run_speculative_loop(
            bridge, 1, num_tokens, seed_token, None, st,
            gamma=gamma, conf_thresh=conf_thresh, vb="batched",
            overlap=True, prefetch=prefetch, prompt=prompt,
            forced=forced, h2d_base_cb=snap_h2d)
        assert len(committed) >= num_tokens, "loop under-committed"
        misses = (layerstorm_engine.h2d_path_stats()["phase_count"]
                  - h2d_state["base"])
        hit_rate = (1.0 - misses / st.lookups) if st.lookups else 0.0
        acc = st.accepted / st.proposed if st.proposed else 0.0
        tau = (len(committed) - 1) / st.rounds if st.rounds else 0.0
        wall_toks = (1000.0 * len(committed) / st.wall_ms
                     if st.wall_ms > 0 else 0.0)
        nr = max(1, st.rounds)
        print(f"\n  ── speculation ──\n"
              f"  rounds={st.rounds}  proposed={st.proposed}  "
              f"accepted={st.accepted}  acceptance={acc:.4f}\n"
              f"  tau (tokens/round incl bonus)={tau:.3f}  "
              f"committed={len(committed)}\n"
              f"  wall: {st.wall_ms:.1f} ms  -> {wall_toks:.3f} tok/s "
              f"over COMMITTED tokens\n"
              f"  draft: {st.draft_ms / nr:.3f} ms/round  "
              f"verify: {st.verify_ms / nr:.3f} ms/round  "
              f"(attn {st.attn_ms / nr:.3f}, moe {st.moe_ms / nr:.3f})"
              f"\n  overlap plain: {st.overlap_plain_ms / nr:.3f} "
              f"ms/round  fallback_rounds={st.fallback_rounds}  "
              f"trunc_slots={st.trunc_slots}\n"
              f"  cache: lookups={st.lookups} misses={misses} "
              f"hit_rate={hit_rate:.4f}", flush=True)
        assert st.rounds > 0, "speculation never ran a round"
        assert st.nan_count == 0, "NaN prob/entropy during speculation"

        assert all(0 <= t < vocab_size for t in committed), \
            "invalid committed token"
        print(f"  tokens: {committed}", flush=True)

        # ── DSP52_DUMP_TOKENS: record this run's committed trajectory ──
        dump = os.environ.get("DSP52_DUMP_TOKENS")
        if dump:
            with open(dump, "w") as f:
                f.write(" ".join(str(t) for t in committed) + "\n")
            print(f"  [dump] wrote {len(committed)} tokens -> {dump}",
                  flush=True)

        # ── THE GATE: token identity vs a C++ reference trajectory ──────
        ref_path = os.environ.get("DSP52_PY_REF")
        if ref_path:
            ref = _load_token_file(ref_path)
            n = min(len(ref), len(committed))
            assert n > 0, "empty DSP52_PY_REF"
            mism = [(i, committed[i], ref[i]) for i in range(n)
                    if committed[i] != ref[i]]
            assert not mism, (
                f"trajectory diverged from the C++ reference at "
                f"{len(mism)}/{n} positions; first: idx={mism[0][0]} "
                f"py={mism[0][1]} cpp={mism[0][2]}")
            print(f"  [gate] HAPPY MATCH: {n} committed tokens identical "
                  f"to the C++ reference ({ref_path})", flush=True)

        # ── optional numeric fingerprints ───────────────────────────────
        ref_stats = os.environ.get("DSP52_PY_REF_STATS")
        if ref_stats:
            parts = [p.strip() for p in ref_stats.split(",")]
            assert len(parts) == 3, "DSP52_PY_REF_STATS = acc,tau,hit"
            if parts[0] != "-":
                assert f"{acc:.4f}" == f"{float(parts[0]):.4f}", (
                    f"acceptance {acc:.4f} != ref {parts[0]}")
            if parts[1] != "-":
                assert f"{tau:.3f}" == f"{float(parts[1]):.3f}", (
                    f"tau {tau:.3f} != ref {parts[1]}")
            if parts[2] != "-":
                assert f"{hit_rate:.4f}" == f"{float(parts[2]):.4f}", (
                    f"hit_rate {hit_rate:.4f} != ref {parts[2]} "
                    f"(placement-arm-specific)")
            print("  [gate] numeric fingerprints matched", flush=True)
    finally:
        layerstorm_engine.stop_engine()
        try:
            os.remove(cfg_path)
        except OSError:
            pass


# ── CPU-only structural check (no CUDA, no weights): config-builder parity
# with the C++ Dsp52ConfigCpu clone — asserts the keeper52 feature deltas +
# the DSpark speculation block, and the bridge/protocol layout self-check
# (import of bridge.ring_bridge runs it) plus the Cython/pure-Python parity
# of the routing-union dedup when the fast path is built. ────────────────────


def test_dsp52_bridge_config_and_layout_cpu():
    from bridge import ring_bridge  # runs _layout_selfcheck on import
    from bridge.ring_bridge import fastbridge_active

    if not _CONFIG.exists():
        pytest.skip("GLM-5.2 config missing")
    if len(_probe_gpus()) < 4:
        pytest.skip("config builder needs the 4-GPU probe")

    cfg, gamma, block = _build_dsp52_config()
    # Champion feature deltas are hard-set (no bisect env surface here).
    assert cfg["compute"]["attention_backend"] == "turboquant_mla"
    assert cfg["hardware"]["dcp_kv_mode"] == "sharded"
    assert cfg["memory"]["kv_tiering"]["enabled"] is True
    assert cfg["speculation"]["method"] == "dspark"
    assert cfg["speculation"]["enabled"] is True
    assert cfg["speculation"]["dspark"]["checkpoint_path"]
    # Champion draft defaults: nvfp4 TP=2 shard on the 5090s.
    if not os.environ.get("DSP52_QUANT"):
        assert cfg["speculation"]["dspark"]["draft_weights_quant"] == "nvfp4"
    if not os.environ.get("DSP52_DRAFT_GPU"):
        assert cfg["speculation"]["dspark"]["draft_gpus"] == [0, 1]
    # The json-inherited fields flow through untouched.
    assert cfg["memory"]["arena_placement"]["freq_table"]
    assert cfg["hardware"]["dcp_indexer_mode"] == "replicated"
    # M3b online placement: config field, schema default TRUE (json carries
    # only non-defaults — absence == engaged; LS_ARENA_PLACE_ONLINE=0 is
    # the explicit static-arm override).
    assert "online" not in cfg["memory"]["arena_placement"]
    # Lever-4 verify-chunk prefetch: config-first, schema default OFF (the
    # json carries only non-default values, so the key is absent here).
    assert "verify_chunk_prefetch" not in cfg["speculation"]["dspark"]
    pf_env = os.environ.get("DSP52_PREFETCH")
    if pf_env:
        # Override-wins: env decides regardless of the config value.
        want = pf_env != "0"
        assert _dsp52_prefetch(cfg) is want
        cfg_on = json.loads(json.dumps(cfg))
        cfg_on["speculation"]["dspark"]["verify_chunk_prefetch"] = True
        assert _dsp52_prefetch(cfg_on) is want
    else:
        assert _dsp52_prefetch(cfg) is False
        cfg_on = json.loads(json.dumps(cfg))
        cfg_on["speculation"]["dspark"]["verify_chunk_prefetch"] = True
        assert _dsp52_prefetch(cfg_on) is True
    assert gamma >= 1
    if (_dsp52_ckpt_dir() / "config.json").exists():
        assert block == gamma + 1, \
            "bonus-anchor layout: block = 1 + speculative_tokens"

    # Champion corpus fixtures exist and parse (the DEFAULT run uses them).
    if _PROMPT_DEFAULT.exists():
        assert len(_load_token_file(_PROMPT_DEFAULT)) >= 512
    if _FORCE_TRAJ_DEFAULT.exists():
        assert len(_load_token_file(_FORCE_TRAJ_DEFAULT)) >= 100

    # Verify-chunk capacity guards (IPC constants, INV-DSPARK-ANCHOR).
    from bridge.protocol import (MAX_BATCH_DESCRIPTORS,
                                 MAX_EXPERT_PREFETCH,
                                 MAX_SIDEBAND_TOKEN_IDS)
    R = gamma + 1
    assert R <= MAX_BATCH_DESCRIPTORS
    assert R <= MAX_SIDEBAND_TOKEN_IDS
    topk = int(cfg["model"].get("num_experts_per_tok", 8))
    assert R * topk <= MAX_EXPERT_PREFETCH

    # Cython/pure parity of the routing-union dedup (when built).
    if fastbridge_active():
        import ctypes

        from bridge import _fastbridge
        vals = [3, -1, 7, 3, 300, 5, 7, 0, 5, 1]
        arr = (ctypes.c_int32 * len(vals))(*vals)
        got = _fastbridge.routing_union(
            ctypes.addressof(arr), len(vals), 256)
        assert got == [3, 7, 5, 0, 1]

    # GpuLru eviction-model smoke: capacity-2 victim selection matches the
    # 13c-2.0 semantics (LRU-back victim, needed-now guard). The LRU arm
    # lives in the bridge LIBRARY (tests/unit/test_bridge_ring.py covers
    # the loops); the integration test above always runs the I8-ACT arm.
    from bridge.ring_bridge import GpuLru
    lru = GpuLru(2)
    for e in (1, 2):
        k = (0, e)
        lru.resident.add(k)
        lru.touch(k)
    # New layer-0 miss (expert 9): victim = LRU back (expert 1).
    class _B:  # minimal shim exposing what fill_moe_entries needs
        moe_gpus = 1  # ticket J: fill_moe_entries reads moe_gpus (== num_gpus on GLM)
        _write_moe_entries = staticmethod(
            lambda layer, experts, assigns, victims: len(experts))
    victims_seen = {}
    def _capture(layer, experts, assigns, victims):
        victims_seen["v"] = victims
        return len(experts)
    b = _B()
    b._write_moe_entries = _capture
    ring_bridge.EngineBridge.fill_moe_entries(b, 0, [9], [lru])
    assert victims_seen["v"][0] == (0, 1, 0)
    assert (0, 9) in lru.resident and (0, 1) not in lru.resident

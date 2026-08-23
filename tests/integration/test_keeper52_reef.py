"""keeper52-REEF through the Python orchestrator — the C++ keeper's port.

Reproduces ``Keeper52Test.HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52``
(tests/integration/keeper52_test.cpp) at EP=4 with the REEF routing path —
the I8-ACT champion stack — driven end-to-end by the Python orchestrator
(``start_engine`` -> ``build_engine_loop`` -> 100-token greedy decode over
the production FETCH_AND_RUN_MOE seam).

REEF path (user decision — the ONLY routing arm ported):
  * The orchestrator sends the PLAIN baseline per-expert targets — the same
    ``gpu_idx = e % num_gpus`` split the C++ keeper's non-affinity arm sends
    (orchestrator_loop._dispatch_work_item already does exactly this) — and
    NO eviction map (``have_evict_map=0``; the writer leaves the field 0).
  * The daemon does ALL routing/eviction via the I8 loader:
    ``LS_LOADER_SHADOW=1 LS_LOADER_ACT=1
    LS_LOADER_CALIB=gpu_loader_calibration_4gpu_hbm.json`` (all other loader
    knobs stay at their landed defaults).
  * Not sending the C++ keeper's 13c-2.0 LRU victim map is trajectory-neutral
    UNDER REEF: with the loader shadow-armed the EvictScoreBoard is built on
    the FIRST FETCH command (shadow_solve_and_log -> ensure_loader_stats_
    boards, before the evict step), so decode (num_seqs==1) always takes the
    inline ensure_residents make-room path and the orchestrator victim map is
    never read (dispatch_moe.cpp inline_evict_owns_makeroom).

Fixture parity with the C++ keeper (EP=4 shape):
  * GLM-5.2 GGUF Q4_K_XL + prepacked expert set, arena fraction 0.90,
    cross-node spill on nodes 0,1 + HBM banks 4-7 (fraction_free 0.80).
  * TQ MLA + sharded DCP KV + HiSparse tiering (hot 512, ratio 16).
  * EP=4: 5090s = TP/attention ranks (expert window 5.5 GB), 5080s =
    expert-only ranks (window 12 GB), stable_zone_fraction 0.95.
  * Seed token 15234, 100 greedy tokens, deterministic EP combine (bf16) —
    the canonical same-trace gate the C++ REEF fingerprint was drawn under.
  * GLM52_ARENA_ATTACH=1 (defaulted ON here — MANDATORY on the box: booting
    detached while the holder pins its ~504 GB store OOM-SIGKILLs at arena
    build, exit 137).

GATE (canonical trace, deterministic per config): cache hit_rate must equal
the C++ REEF reference fingerprint 0.7340 EXACTLY (4-decimal print parity).
tok/s is REPORTED against the 8.11 tok/s C++ reference — Python orchestrator
overhead is an expected, interesting result, not a failure.

Run (needs the 4-GPU box idle + the GLM-5.2 assets; ~2-3 min warm):
  KEEPER52_PY=1 .venv/bin/python -m pytest \
      tests/integration/test_keeper52_reef.py -s

Optional env (same semantics as the C++ keeper):
  KEEPER52_TOKENS=<n>            decode length override (PROFILING ONLY —
                                 the 0.7340 fingerprint gate is skipped)
  KEEPER52_EXPERT_CACHE_GB       5090 expert window GB (default 5.5)
  KEEPER52_EXPERT_CACHE_GB_5080  5080 expert window GB (default 12)
  KEEPER52_STABLE_FRAC           stable-zone fraction (default 0.95)
  KEEPER52_ARENA_FRACTION        host-arena fraction_total (default 0.90)
  GLM52_KV_TIERING / GLM52_KVT_RATIO  tiering hot slots / cold ratio
  LS_KEEPER_SEED_TOKEN           decode seed token (default 15234)
  LS_KEEPER_DUMP_ALL=1           print every step (default: 0th + each 10th)
  LS_LOADER_CALIB                calibration JSON (default 4gpu_hbm)
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
_PREPACKED = _PROJECT_ROOT / "test-data" / "GLM-5.2-prepacked"

# The established keeper decode seed (keeper52_test.cpp).
SEED_TOKEN = 15234
NUM_TOKENS = 100

# C++ REEF reference on this box (spec/CHANGELOG.md 2026-07-18 fingerprint
# restoration: "REEF hit 0.7340 bit-exact"; wall ~8.11 tok/s).
REEF_REFERENCE_HIT = "0.7340"
REEF_REFERENCE_TOKS = 8.11

RUN_DEADLINE_S = 3600  # cold arena load can take several minutes


def _probe_gpus() -> list[tuple[int, int, int]]:
    """[(nvidia-smi index, total MiB, free MiB)] — PCI_BUS_ID order.

    With CUDA_DEVICE_ORDER=PCI_BUS_ID the nvidia-smi index IS the CUDA
    ordinal, so this mirrors the C++ probe_sm120_gpus() without touching
    CUDA from Python.
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
    """EP=4 shape: 2 idle GPUs >= 28 GiB (TP 5090s) + 2 more >= 14 GiB."""
    free_sorted = sorted(_probe_gpus(), key=lambda g: g[1], reverse=True)
    if len(free_sorted) < 4:
        return False
    return (free_sorted[1][1] >= 28 * 1024 and free_sorted[1][2] >= 28_000
            and free_sorted[3][1] >= 14 * 1024
            and free_sorted[3][2] >= 14_000)


def _load_force_traj(path: str) -> list[int]:
    """Load a teacher-forcing token list (JSON array, or whitespace/comma
    separated ints).  Milestone B matched-trajectory A/B: BOTH OFF and ON
    commit these exact tokens so the fetch/compute workload is identical
    despite the sub-ULP CPU-vs-GPU SwiGLU argmax drift."""
    with open(path) as f:
        raw = f.read().strip()
    try:
        arr = json.loads(raw)
        return [int(x) for x in arr]
    except (json.JSONDecodeError, TypeError, ValueError):
        return [int(x) for x in raw.replace(",", " ").split()]


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v:
        try:
            return float(v)
        except ValueError:
            pass
    return default


def _build_keeper52_ep4_config() -> dict:
    """Faithful port of keeper52_test.cpp build_keeper52_config (EP=4 arm).

    Every knob mirrors the C++ builder; deltas are marked REEF (loader env
    contract) — there is no affinity/eviction-map machinery here by design.
    """
    with open(_CONFIG) as f:
        cfg = json.load(f)

    cfg["model"]["weights_path"] = str(_GGUF)
    cfg["quantization"]["gguf_strategy"] = "int"

    # ── EP=4 hardware block (INV-MOE-EP-XTP): gpus ORDERED 5090s-first —
    # ids are live CUDA ordinals; positions 0,1 = 5090 TP ranks, 2,3 = 5080
    # expert-only ranks. Python's stable largest-first sort reproduces the
    # C++ std::sort result for this 2+2 shape ([2,3,0,1] on the dev box).
    expert_cache_gb = _env_float("KEEPER52_EXPERT_CACHE_GB", 5.5)
    expert_cache_gb_5080 = _env_float("KEEPER52_EXPERT_CACHE_GB_5080", 12.0)
    stable_frac = _env_float("KEEPER52_STABLE_FRAC", 0.95)
    probed = sorted(_probe_gpus(), key=lambda g: g[1], reverse=True)
    assert len(probed) >= 4, "EP=4 needs 4 GPUs"
    gpus = []
    for pos in range(4):
        big = pos < 2
        g = {
            "id": probed[pos][0],
            "type": "rtx5090" if big else "rtx5080",
            "vram_gb": 30 if big else 15,
            "pcie_gen": 5, "pcie_width": 16,
            "roles": (["attention", "resident", "expert_streaming"]
                      if big else ["expert_streaming"]),
        }
        win = expert_cache_gb if big else expert_cache_gb_5080
        if win > 0.0:
            g["vram_allocation_gb"] = {
                "expert_streaming": win,
                "stable_zone_fraction": stable_frac,
            }
        gpus.append(g)
    cfg["hardware"]["gpus"] = gpus
    cfg["hardware"]["tp_array"] = [0, 1]
    cfg["parallelism"]["tensor_parallelism"] = 2

    # ── Feature 4: sharded DCP KV (indexer replicated) ──
    cfg["hardware"]["dcp_kv_mode"] = "sharded"
    cfg["hardware"]["dcp_indexer_mode"] = "replicated"
    cfg["memory"]["kv_cache"]["dcp_chunk_size"] = 16

    # ── Feature 5: TurboQuant 4-bit MLA ──
    cfg["compute"]["attention_backend"] = "turboquant_mla"

    # ── Feature 3: HiSparse DSA-guided KV tiering ──
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

    # ── Feature 2 + inherited full-fit O_DIRECT overlapped-register arena ──
    arena_frac = _env_float("KEEPER52_ARENA_FRACTION", 0.90)
    if not (0.0 < arena_frac <= 1.0):
        arena_frac = 0.90
    cfg.setdefault("preprocessing", {})["prepacked_dir"] = str(_PREPACKED)
    cfg["memory"]["preload_expert_buffers"] = True
    cfg["memory"]["pin_host_expert_pool"] = True
    cfg["memory"]["pin_host_expert_pool_direct_load"] = True
    # P-24b holder attach — GLM52_ARENA_ATTACH is defaulted to "1" by the
    # test (MANDATORY on the box, see module docstring).
    cfg["memory"]["arena_attach"] = {
        "enabled": os.environ.get("GLM52_ARENA_ATTACH") == "1"}
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

    # ── REEF: I8 loader-solver wiring (identical to the C++ SetUp) ──
    cfg["gpu_loader"] = {
        "enabled": True,
        "calibration_path": os.environ.get(
            "LS_LOADER_CALIB", "gpu_loader_calibration_4gpu_hbm.json"),
    }

    # ── C-6 CPU expert offload (Milestone A) — opt-in via KEEPER52_CPU_EXPERT=
    # <numa_node> (default UNSET ⇒ no CPU device ⇒ byte-identical champion).
    # Pairs with the runtime gates LS_CPU_EXPERT=1 + LS_CPU_EXPERT_FORCE=... .
    cpu_node = os.environ.get("KEEPER52_CPU_EXPERT")
    if cpu_node is not None and cpu_node.strip() != "":
        dev = {"numa_node": int(cpu_node)}
        # Milestone B: bound the host FFN's NUMA-node parallelism so a LARGE
        # forced offload is measured under a controlled thread budget (the
        # I8 solver would gate CPU use by a cost model; forced offload does
        # not, so surface the thread knob).  0/unset => all physical cores.
        mt = os.environ.get("KEEPER52_CPU_EXPERT_THREADS")
        if mt and mt.isdigit() and int(mt) > 0:
            dev["max_threads"] = int(mt)
        cfg["hardware"]["cpu_expert_devices"] = [dev]
    return cfg


def _setup_reef_env() -> None:
    """The REEF env contract — set BEFORE start_engine (no-overwrite, like
    the C++ fixture's setenv(..., 0)). The daemon reads these at boot."""
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        # REEF = shadow solve + ACT reroute + the 4-GPU HBM calibration.
        "LS_LOADER_SHADOW": "1",
        "LS_LOADER_ACT": "1",
        "LS_LOADER_CALIB": "gpu_loader_calibration_4gpu_hbm.json",
        # Canonical placement-invariant EP combine (keeper harness default
        # since 2026-07-18): one routing trajectory per config — the 0.7340
        # fingerprint is drawn under this gate.
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE": "1",
        "LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION": "bf16",
        # MANDATORY while the arena holder pins the box's store (exit-137
        # OOM otherwise). Explicit GLM52_ARENA_ATTACH=0 is respected but
        # unsupported on this box.
        "GLM52_ARENA_ATTACH": "1",
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)


_setup_reef_env()


@pytest.mark.skipif(os.environ.get("KEEPER52_PY") != "1",
                    reason="set KEEPER52_PY=1 to run (4-GPU GLM-5.2 keeper)")
@pytest.mark.skipif(not _CONFIG.exists(), reason="GLM-5.2 config missing")
@pytest.mark.skipif(not _GGUF.exists(), reason="GLM-5.2 GGUF not present")
@pytest.mark.skipif(not (_PREPACKED / "manifest.json").exists(),
                    reason="GLM-5.2 prepacked set not present")
@pytest.mark.skipif(not _ep4_gpus_ok(),
                    reason="EP=4 needs 4 idle GPUs (2x >=28 GiB + 2x >=14)")
def test_keeper52_reef_hundred_token_decode():
    import layerstorm_engine
    from orchestrator.engine_glue import build_engine_loop
    from orchestrator.loop.orchestrator_loop import (InferenceRequest,
                                                     OrchestratorConfig)

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        pytest.skip("keeper52-REEF requires CUDA_VISIBLE_DEVICES=0,1,2,3 "
                    "(nvidia-smi index == CUDA ordinal contract)")
    if os.environ.get("KEEPER52_EP", "4") != "4":
        pytest.skip("only the EP=4 REEF arm is ported (KEEPER52_EP=4)")

    num_tokens = NUM_TOKENS
    nt = os.environ.get("KEEPER52_TOKENS")
    if nt and nt.isdigit() and int(nt) > 0:
        num_tokens = int(nt)

    seed_token = SEED_TOKEN
    st = os.environ.get("LS_KEEPER_SEED_TOKEN")
    if st and st.isdigit():
        seed_token = int(st)

    cfg = _build_keeper52_ep4_config()
    vocab_size = int(cfg["model"]["vocab_size"])
    cfg_path = "/tmp/keeper52_py_reef_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[keeper52-py] REEF EP=4 boot (config={cfg_path}) ...",
          flush=True)
    t0 = time.monotonic()
    info = layerstorm_engine.start_engine(cfg_path)
    try:
        print(f"[keeper52-py] engine up in {time.monotonic() - t0:.0f}s: "
              f"layers={info.num_layers} experts={info.num_experts} "
              f"gpus={info.num_gpus}", flush=True)
        assert info.num_gpus == 4, "EP=4 engine must expose 4 GPUs"

        buffer_ids = layerstorm_engine.query_buffer_ids()
        loop, handles = build_engine_loop(
            info,
            buffer_ids=buffer_ids,
            vocab_size=vocab_size,
            speculation_depth=0,          # pure greedy AR — the keeper shape
            orchestrator_config=OrchestratorConfig(cycle_budget_us=200.0),
        )
        assert handles.metadata.hidden_buf_id != 0, "hidden buf not found"
        assert handles.metadata.logits_buf_id != 0, "logits buf not found"

        # ── Milestone B TEACHER-FORCING (KEEPER52_FORCE_TRAJ=<file>) ─────────
        # Override the committed greedy token with the forced list so OFF and
        # ON decode the IDENTICAL trajectory (matched fetch/compute workload).
        # Pure greedy AR (speculation_depth=0): _extract_token_id fires once
        # per generated token, call k == generation step k.  The real readback
        # still runs (the model computes normally); only the COMMITTED id is
        # replaced, and the next EMBEDDING feeds token_history[-1].  Class-level
        # patch (loop mixes in _SpeculationMixin); restored in the finally.
        force_path = os.environ.get("KEEPER52_FORCE_TRAJ")
        forced_ids: list[int] = (
            _load_force_traj(force_path) if force_path else [])
        loop_cls = type(loop)
        orig_extract = loop_cls._extract_token_id
        force_state = {"i": 0, "match": 0}
        if forced_ids:
            def _forced_extract(self_, payload, _orig=orig_extract):
                real = _orig(self_, payload)
                if real is None:
                    return None
                k = force_state["i"]
                force_state["i"] += 1
                if k < len(forced_ids):
                    if real == forced_ids[k]:
                        force_state["match"] += 1
                    return int(forced_ids[k])
                return real
            loop_cls._extract_token_id = _forced_extract
            print(f"[keeper52-py] TEACHER-FORCING {len(forced_ids)} tokens "
                  f"from {force_path}", flush=True)

        # ── lookups counter: Σ expert_count over every FETCH_AND_RUN_MOE —
        # the exact denominator the C++ keeper accumulates (moe_lookups).
        # Test-side CLASS-level wrap (CommandWriter has __slots__, so an
        # instance patch is rejected); no orchestrator change — REEF sends
        # plain targets. Restored in the finally block.
        from orchestrator.command_writer import CommandWriter
        lookups = {"n": 0, "cmds": 0}
        orig_farm = CommandWriter.fetch_and_run_moe

        def counting_farm(self_, *args, **kwargs):
            lookups["n"] += int(kwargs.get("expert_count", 0))
            lookups["cmds"] += 1
            return orig_farm(self_, *args, **kwargs)

        CommandWriter.fetch_and_run_moe = counting_farm

        # Per-seed H2D delta base (engine counters are cumulative).
        h2d_base = layerstorm_engine.h2d_path_stats()
        assert h2d_base, "h2d_path_stats unavailable (engine not running?)"

        # ── the 100-token decode: prompt = [seed]; the single prompt token
        # feeds through the per-token chain at pos 0 (superchunk body = 0)
        # and its OUTPUT_HEAD yields generated token #1 — byte-parity with
        # the C++ keeper's step 0. ──
        results: list = []
        tokens_seen: list[int] = []
        token_times: list[float] = []
        dump_all = os.environ.get("LS_KEEPER_DUMP_ALL") is not None

        def on_token(rid, token_id, lp):
            tokens_seen.append(token_id)
            token_times.append(time.monotonic())
            i = len(tokens_seen) - 1
            if dump_all or i == 0 or (i + 1) % 10 == 0:
                dt = (token_times[-1] - token_times[-2]) if i else 0.0
                print(f"  step {i:3d}: token={token_id:6d}  "
                      f"(+{dt * 1000.0:.0f} ms)", flush=True)

        def on_complete(rid, gen, reason, lp):
            results.append((rid, list(gen), reason))

        print(f"\n=== keeper52-py REEF HundredTokenDecodeFetchAndRun "
              f"(GLM-5.2 GGUF, EP=4 e%4 baseline targets, I8-ACT loader "
              f"routing, TQ + sharded-KV + HiSparse tiering) ===",
              flush=True)
        drive_t0 = time.monotonic()
        loop.submit_request(InferenceRequest(
            request_id=1,
            prompt_token_ids=[seed_token],
            gpu=0,
            max_tokens=num_tokens,
            on_token=on_token,
            on_complete=on_complete,
        ))
        deadline = time.monotonic() + RUN_DEADLINE_S
        while not results and time.monotonic() < deadline:
            loop.run_one_cycle()
        assert results, (f"request did not finish in {RUN_DEADLINE_S}s "
                         f"({len(tokens_seen)} tokens so far)")
        # Drain trailing completions (finalize E_CMD_SEQ_FREE ack).
        for _ in range(50):
            loop.run_one_cycle()

        rid, gen, reason = results[0]
        assert rid == 1 and reason == "length", f"finish: {reason}"
        assert len(gen) == num_tokens
        assert all(0 <= t < vocab_size for t in gen), "invalid token id"

        # ── stats (same accounting as the C++ keeper) ──
        h2d = layerstorm_engine.h2d_path_stats()
        misses = h2d["phase_count"] - h2d_base["phase_count"]
        dma_ns = h2d["dma_wait_ns"] - h2d_base["dma_wait_ns"]
        cpy_ns = h2d["memcpy_ns"] - h2d_base["memcpy_ns"]
        hit_rate = (1.0 - misses / lookups["n"]) if lookups["n"] else 0.0

        total_s = token_times[-1] - drive_t0
        avg_ms = total_s * 1000.0 / num_tokens
        steady_ms = ((token_times[-1] - token_times[0]) * 1000.0
                     / (num_tokens - 1)) if num_tokens > 1 else avg_ms
        print(f"\n  Total: {total_s * 1000.0:.1f} ms for {num_tokens} "
              f"tokens\n  Avg: {avg_ms:.3f} ms/token "
              f"({1000.0 / avg_ms:.3f} tok/s)\n"
              f"  SteadyState(excl tok0): {steady_ms:.3f} ms/token "
              f"({1000.0 / steady_ms:.3f} tok/s)", flush=True)
        print(f"  cache: lookups={lookups['n']} misses={misses} "
              f"hit_rate={hit_rate:.4f} "
              f"(fetched {misses // max(num_tokens, 1)}/token, "
              f"{lookups['cmds']} MoE cmds)", flush=True)
        if misses:
            print(f"  H2D x-ray: direct={h2d['direct'] - h2d_base['direct']} "
                  f"staged={h2d['staged'] - h2d_base['staged']}  "
                  f"DMA+detect {dma_ns / 1e3 / misses:.1f} us/xfer  "
                  f"host-copy {cpy_ns / 1e3 / misses:.1f} us/xfer",
                  flush=True)
        print(f"  vs C++ REEF reference: hit {REEF_REFERENCE_HIT} @ "
              f"{REEF_REFERENCE_TOKS} tok/s "
              f"(delta {1000.0 / steady_ms - REEF_REFERENCE_TOKS:+.2f} "
              f"tok/s steady)", flush=True)
        print(f"  tokens: {gen}", flush=True)

        # ── Milestone B: dump the decoded tokens (KEEPER52_DUMP_TOKENS=<file>)
        # so the OFF baseline trajectory can teacher-force the ON run.
        dump_path = os.environ.get("KEEPER52_DUMP_TOKENS")
        if dump_path:
            with open(dump_path, "w") as f:
                json.dump(gen, f)
            print(f"  [dump] wrote {len(gen)} tokens -> {dump_path}",
                  flush=True)
        if forced_ids:
            print(f"  [force] model argmax matched forced token "
                  f"{force_state['match']}/{min(len(forced_ids), num_tokens)} "
                  f"(mismatches = committed-forced, model-argmax diverged)",
                  flush=True)

        # ── THE GATE: trajectory parity with the C++ REEF canonical trace.
        # Deterministic per config (same-trace combine gate) — an exact
        # 4-decimal hit_rate match proves the port is faithful.  SKIPPED when
        # the CPU-expert offload or teacher-forcing is engaged (a different
        # config / trajectory by construction — this is the Milestone B wall,
        # not the canonical-trace fingerprint gate).
        cpu_expert_on = bool(os.environ.get("KEEPER52_CPU_EXPERT", "").strip())
        if (num_tokens == NUM_TOKENS and seed_token == SEED_TOKEN
                and not forced_ids and not cpu_expert_on):
            assert f"{hit_rate:.4f}" == REEF_REFERENCE_HIT, (
                f"REEF trajectory diverged: Python hit {hit_rate:.4f} != "
                f"C++ reference {REEF_REFERENCE_HIT} "
                f"(lookups={lookups['n']} misses={misses})")
    finally:
        try:
            from orchestrator.command_writer import CommandWriter
            CommandWriter.fetch_and_run_moe = orig_farm
        except (ImportError, NameError):
            pass
        try:
            loop_cls._extract_token_id = orig_extract
        except NameError:
            pass
        layerstorm_engine.stop_engine()
        try:
            os.remove(cfg_path)
        except OSError:
            pass

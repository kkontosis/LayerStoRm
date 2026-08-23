"""EPM corpus collection driver (TD-EPM-CORPUS / EPM-4 Part 1 + EPM-5).

Drives the REAL dspark engine through the Python orchestrator loop — the
only existing driver that (a) takes arbitrary prompt text, (b) generates
hundreds of tokens per prompt, and (c) writes the EPM block manifest
(accepted_len / tokens_match provenance) beside the daemon's EPMB/EPMR
records. No engine code is touched. The loop is tests/integration/
test_orch_engine_decode.py's, with DsparkDraftConfig speculation.

Two engine-boot recipes (--recipe):

* ``champion`` (DEFAULT — the EPM-5 collection shape, M6 §4 GO decision):
  mirrors the dsp52 champion boot (tests/integration/dsp52_test.cpp
  build_dsp52_config EP=4 + the canonical recipe env,
  test-data/prompts/README.md): 4 GPUs largest-first (5090 pair =
  attention/resident/expert window, 5080 pair = expert-ONLY ranks),
  TP=2 + EP=4 (ep_gpu_indices [0,1,2,3]), TurboQuant MLA attention,
  SHARDED KV (the 2026-08-17 champion verdict — replicated loses 4.6%;
  dsa_sparse_prefill dense-fallbacks under it) + replicated indexer,
  KV tiering,
  arena-holder attach (warm 504 GB store adopt), preview γ15 checkpoint
  (the default test-data symlink), nvfp4 SHARDED draft on the 5090 TP
  pair (3.25 GB expert window each; 5080s keep 11.0), and the engine env
  set: LS_IPC_PIN, LS_ARENA_PLACE_ONLINE, LS_INDEXER_REWIND,
  LS_CHUNK_SMALLM, deterministic EP combine (bf16).
  NOT portable from the C++ champion (test-side machinery, measured as
  found): chunk/batched verify (the Python dspark path verifies with
  SEQUENTIAL early-stop feeds — speculation.py _finish_dspark_drafting),
  draft-overlap, prev-union prefetch, REEF test-side placement, and the
  DSP-9 conf-0.1 truncation (inert under sequential verify: rejected
  drafts are never fed, so there is nothing to truncate;
  confidence_enabled stays ON — the c_k labels feed the manifest).

* ``legacy`` (the EPM-4 shape): 3 GPUs (2x5090 TP target + 1x5080
  expert-free draft host), mirrors glm52_gguf_golden_test.cpp's
  GLM52_DSPARK=1 config.

One engine RUN per dump dir (dataset.py contract: engine seq ids restart
per run); multiple SEQUENCES per run are fine and are exactly what the
INV-EPM-DATA sequence split needs. Requests are submitted strictly
sequentially (dspark routing capture is decode batch-1, TD-EPM-BATCH).

Run (from the repo root, GPUs free, arena holder armed with the warm
online-placement store):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python3 tools/elb_train/collect_corpus.py \
      --dump-dir build/epm5-corpus/run_1 --prompts tools/elb_train/prompts.json
(legacy: CUDA_VISIBLE_DEVICES=2,3,0 ... --recipe legacy)
Optional: --gen-tokens N (override per-prompt max), --budget-s S (stop
submitting new prompts after S seconds of wall time), --smoke (1 short
prompt, 12 tokens — pipeline validation only).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "python"))
_BUILD_PY = _PROJECT_ROOT / "build" / "python"
if _BUILD_PY.exists():
    sys.path.insert(0, str(_BUILD_PY))

_CONFIG = _PROJECT_ROOT / "test-data" / "config" / "glm_5_2_gguf.json"
_GGUF = (_PROJECT_ROOT / "test-data" / "GLM-5.2-GGUF-Q4_K_XL"
         / "GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf")
_DSPARK_CKPT = _PROJECT_ROOT / "test-data" / "GLM-5.2-speculator.dspark"
_TOKENIZER = _PROJECT_ROOT / "test-data" / "GLM-5.2" / "tokenizer.json"


def checkpoint_block_shape(ckpt: pathlib.Path) -> tuple[int, int]:
    """(block_size, gamma) from the checkpoint's own config.json — the
    engine fails closed on a mismatch (the test-data symlink can point at
    either the shipped block-8 checkpoint or the gamma=16 preview)."""
    with open(ckpt / "config.json") as f:
        c = json.load(f)
    block = int(c["block_size"])
    gamma = int(c.get("speculative_tokens", block - 1))
    return block, gamma


# The canonical champion recipe's ENGINE-side env (test-data/prompts/
# README.md; the DSP52_* test-side knobs have no meaning here).  Applied
# with setdefault so an explicit caller env always wins.
_CHAMPION_ENV = {
    "LS_IPC_PIN": "1",
    "LS_ARENA_PLACE_ONLINE": "1",
    "LS_INDEXER_REWIND": "1",
    "LS_CHUNK_SMALLM": "1",
    "LAYERSTORM_DETERMINISTIC_EP_COMBINE": "1",
    "LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION": "bf16",
}


def probe_gpus_largest_first() -> list[dict]:
    """Visible CUDA ordinals sorted largest-VRAM-first (the dsp52
    probe_sm120_gpus order) via nvidia-smi, honoring CUDA_VISIBLE_DEVICES
    (which must be PCI-ordered — set CUDA_DEVICE_ORDER=PCI_BUS_ID)."""
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout
    phys = []
    for line in out.strip().splitlines():
        idx, name, mem = [t.strip() for t in line.split(",")]
        phys.append({"index": int(idx), "name": name, "mem_mib": int(mem)})
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        order = [int(t) for t in vis.split(",") if t.strip()]
        by_idx = {g["index"]: g for g in phys}
        gpus = [dict(by_idx[p], ordinal=v) for v, p in enumerate(order)]
    else:
        gpus = [dict(g, ordinal=g["index"]) for g in phys]
    gpus.sort(key=lambda g: -g["mem_mib"])  # stable: ties keep PCI order
    return gpus


def build_engine_config(
        dump_dir: str, ckpt: pathlib.Path, recipe: str) -> tuple[str, dict]:
    with open(_CONFIG) as f:
        cfg = json.load(f)
    cfg["model"]["weights_path"] = str(_GGUF)
    cfg["quantization"]["gguf_strategy"] = "int"
    block, gamma = checkpoint_block_shape(ckpt)

    if recipe == "champion":
        # ── dsp52 champion EP=4 shape (build_dsp52_config mirror) ──────
        gpus = probe_gpus_largest_first()
        if len(gpus) < 4:
            raise SystemExit(f"champion recipe needs 4 visible GPUs, "
                             f"found {len(gpus)}")
        all_roles = ["attention", "resident", "expert_streaming"]
        expert_only = ["expert_streaming"]
        draft_positions = (0, 1)     # nvfp4 TP=2 draft shard on the 5090s
        hw = []
        for p in range(4):
            big = p < 2              # largest-first: 0,1 = the 5090s
            # Windows: sharded-nvfp4 draft hosts 3.25 GB each (proven
            # per-card occupancy, TD-DSPARK-DRAFT-SHARD); expert-only
            # 5080s keep the keeper 11.0 default.
            win = 3.25 if p in draft_positions else (4.5 if big else 11.0)
            hw.append({
                "id": gpus[p]["ordinal"],
                "type": "rtx5090" if big else "rtx5080",
                "vram_gb": 30 if big else 15,
                "pcie_gen": 5, "pcie_width": 16,
                "roles": all_roles if big else expert_only,
                "vram_allocation_gb": {
                    "expert_streaming": win,
                    "stable_zone_fraction": 0.95,   # EP=4 keeper default
                },
            })
        cfg["hardware"]["gpus"] = hw
        cfg["hardware"]["tp_array"] = [0, 1]
        # Champion recipe (2026-08-17, scratchpad/GAP_EXPERIMENTS_9_1.md):
        # sharded KV — replicated measured -4.6% (doubles KvTiering
        # materialization); sharded is also the schema default now.
        cfg["hardware"]["dcp_kv_mode"] = "sharded"
        cfg["hardware"]["dcp_indexer_mode"] = "replicated"
        cfg["parallelism"]["tensor_parallelism"] = 2
        # EP=4: ALL ranks own routed experts (INV-MOE-EP-XTP); the draft
        # shard hosts (0,1) still carry expert windows in this shape.
        cfg["_internal-orchestrator"] = {"ep_gpu_indices": [0, 1, 2, 3]}
        cfg["compute"]["attention_backend"] = "turboquant_mla"
        # Inert under sharded KV (sparse chunks dense-fallback,
        # TD-SPARSE-PREFILL-KVS); kept for a future replicated A/B.
        cfg["compute"]["dsa_sparse_prefill"] = True
        cfg["memory"]["kv_cache"]["dcp_chunk_size"] = 16
        cfg["memory"]["kv_tiering"] = {
            "enabled": True, "hot_buffer_slots": 512,
            "host_to_device_ratio": 16.0}
        # P-24b persistent arena: adopt the holder's warm store (the
        # canonical online-placement layout).  The arena geometry below
        # (fraction 0.9 + spill nodes) is the base config's — identical
        # to the keeper/dsp52 recipe, so the store adopts warm.
        cfg["memory"]["arena_attach"] = {"enabled": True}
        drafts = {
            "checkpoint_path": str(ckpt),
            "block_size": block,
            "speculative_tokens": gamma,
            "draft_gpus": list(draft_positions),
            "draft_weights_quant": "nvfp4",
            "confidence_enabled": True,
            # Recorded for provenance; inert under sequential verify
            # (see module docstring).
            "confidence_threshold": 0.1,
            "epm_dump_dir": dump_dir,
        }
    else:
        # ── legacy EPM-4 shape (glm52_gguf_golden_test GLM52_DSPARK=1) ─
        cfg["hardware"]["gpus"] = [
            {"id": 0, "type": "rtx5090", "vram_gb": 30,
             "pcie_gen": 5, "pcie_width": 16},
            {"id": 1, "type": "rtx5090", "vram_gb": 30,
             "pcie_gen": 5, "pcie_width": 16},
            # Draft GPU (visible ordinal 2 under CUDA_VISIBLE_DEVICES=2,3,0).
            {"id": 2, "type": "rtx5080", "vram_gb": 15,
             "pcie_gen": 5, "pcie_width": 16},
        ]
        cfg["hardware"]["tp_array"] = [0, 1]
        cfg["parallelism"]["tensor_parallelism"] = 2
        drafts = {
            "checkpoint_path": str(ckpt),
            "block_size": block,
            "speculative_tokens": gamma,
            "draft_gpus": [2],
            "confidence_enabled": True,
            "epm_dump_dir": dump_dir,
        }

    cfg.setdefault("speculation", {})
    cfg["speculation"]["method"] = "dspark"
    cfg["speculation"]["enabled"] = True
    cfg["speculation"].setdefault("dspark", {}).update(drafts)
    cfg["memory"]["vram_safety_margin_gb"] = 3.0
    # Longer sequences than the golden's ~10 tokens: 8192 physical pages
    # covers 78 layers x ceil(~1600/16) logical pages.
    cfg["memory"]["kv_cache"]["max_pages_per_gpu"] = 8192
    cfg["memory"]["kv_cache"]["page_growth_chunk_tokens"] = 16
    cfg_path = "/tmp/epm4_collect_glm52_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg_path, cfg


def tokenize_prompts(prompts: list[dict]) -> list[dict]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(_TOKENIZER))
    out = []
    for p in prompts:
        ids = tok.encode(p["text"], add_special_tokens=False).ids
        out.append({**p, "token_ids": ids})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump-dir", required=True,
                    help="fresh dump dir (ONE engine run per dir)")
    ap.add_argument("--prompts", required=True,
                    help="JSON list of {name, text, gen_tokens}")
    ap.add_argument("--gen-tokens", type=int, default=0,
                    help="override every prompt's gen_tokens")
    ap.add_argument("--budget-s", type=float, default=7200.0,
                    help="stop submitting NEW prompts after this wall time")
    ap.add_argument("--smoke", action="store_true",
                    help="single 12-token request (pipeline validation)")
    ap.add_argument("--dspark-ckpt", type=pathlib.Path, default=_DSPARK_CKPT,
                    help="dspark checkpoint dir (block_size/gamma are read "
                         "from its config.json)")
    ap.add_argument("--recipe", choices=("champion", "legacy"),
                    default="champion",
                    help="engine boot recipe (default champion = the dsp52 "
                         "EP=4 canonical stack, needs 4 GPUs + the warm "
                         "arena holder; legacy = the 3-GPU EPM-4 shape)")
    ap.add_argument("--prefill-subchunk-tokens", type=int, default=0,
                    help="override OrchestratorConfig.prefill_subchunk_"
                         "tokens (0 = loop default 64). Tuning knob only: "
                         "since TD-DSPARK-SUPERCHUNK-CAPTURE was resolved, "
                         "the dspark aux capture ingests multi-sub-chunk "
                         "superchunk prefills chunk-major, so long prompts "
                         "draft at the default 64 and the old subchunk-512 "
                         "workaround is unnecessary")
    ap.add_argument("--deadline-per-token-s", type=float, default=15.0,
                    help="per-request deadline = max(1800, this * "
                         "gen_tokens + 600) seconds")
    args = ap.parse_args(argv)
    ckpt = args.dspark_ckpt.resolve()
    block_size, gamma = checkpoint_block_shape(ckpt)

    dump_dir = pathlib.Path(args.dump_dir).resolve()
    if dump_dir.exists() and any(dump_dir.iterdir()):
        print(f"refusing non-empty dump dir {dump_dir}", file=sys.stderr)
        return 2
    dump_dir.mkdir(parents=True, exist_ok=True)

    with open(args.prompts) as f:
        prompts = json.load(f)
    if args.smoke:
        prompts = prompts[:1]
        for p in prompts:
            p["gen_tokens"] = 12
    if args.gen_tokens > 0:
        for p in prompts:
            p["gen_tokens"] = args.gen_tokens
    prompts = tokenize_prompts(prompts)

    # Champion recipe: canonical engine env (explicit caller env wins).
    if args.recipe == "champion":
        for k, v in _CHAMPION_ENV.items():
            os.environ.setdefault(k, v)

    # Provenance beside the raw records (small; the study dir copies it).
    with open(dump_dir / "provenance.json", "w") as f:
        json.dump({
            "driver": "tools/elb_train/collect_corpus.py",
            "recipe": args.recipe,
            "recipe_env": ({k: os.environ[k] for k in _CHAMPION_ENV}
                           if args.recipe == "champion" else {}),
            "engine_config_base": str(_CONFIG),
            "gguf": str(_GGUF),
            "dspark_checkpoint": str(ckpt),
            "tokenizer": str(_TOKENIZER),
            "block_size": block_size,
            "gamma": gamma,
            "prompts": [{k: p[k] for k in
                         ("name", "text", "gen_tokens", "token_ids")}
                        for p in prompts],
        }, f, indent=2)

    # Daemon + orchestrator both resolve LS_EPM_DUMP (set before
    # start_engine so the C++ side sees it too; config carries it as well).
    os.environ["LS_EPM_DUMP"] = str(dump_dir)

    cfg_path, cfg = build_engine_config(str(dump_dir), ckpt, args.recipe)

    import layerstorm_engine
    from orchestrator.dspark_draft import DsparkDraftConfig
    from orchestrator.engine_glue import (
        build_engine_loop,
        ep_gpu_indices_from_config,
    )
    from orchestrator.loop.orchestrator_loop import (
        InferenceRequest,
        OrchestratorConfig,
    )

    print(f"[epm4-collect] starting engine (config={cfg_path}) ...",
          flush=True)
    t0 = time.monotonic()
    info = layerstorm_engine.start_engine(cfg_path)
    results: list[dict] = []
    try:
        print(f"[epm4-collect] engine up in {time.monotonic() - t0:.0f}s: "
              f"layers={info.num_layers} experts={info.num_experts} "
              f"gpus={info.num_gpus}", flush=True)

        loop, handles = build_engine_loop(
            info,
            buffer_ids=layerstorm_engine.query_buffer_ids(),
            speculation_depth=gamma,
            dspark_config=DsparkDraftConfig(
                enabled=True,
                block_size=block_size,
                speculative_tokens=gamma,
                confidence_enabled=True,
                epm_dump_dir=str(dump_dir),
            ),
            # EP owners = the TP pair only (topology-derived fallback) —
            # the draft GPU (position 2) hosts no expert cache; including
            # it deadline-burns every MoE layer (INV-FAR;
            # OrchestratorConfig.ep_gpu_indices).  A config-supplied
            # _internal-orchestrator.ep_gpu_indices wins when present.
            orchestrator_config=OrchestratorConfig(
                cycle_budget_us=200.0,
                ep_gpu_indices=ep_gpu_indices_from_config(
                    cfg,
                    fallback=tuple(cfg["hardware"]["tp_array"])),
                **({"prefill_subchunk_tokens":
                    args.prefill_subchunk_tokens}
                   if args.prefill_subchunk_tokens > 0 else {})),
        )
        assert handles.metadata.hidden_buf_id != 0
        assert handles.metadata.logits_buf_id != 0

        budget_end = time.monotonic() + args.budget_s
        for rid, p in enumerate(prompts, start=1):
            if time.monotonic() > budget_end:
                print(f"[epm4-collect] wall budget reached — skipping "
                      f"remaining {len(prompts) - rid + 1} prompts",
                      flush=True)
                break
            done: list = []
            n_tok = [0]

            def on_token(_rid, token_id, _lp, _n=n_tok, _p=p):
                _n[0] += 1
                if _n[0] % 50 == 0:
                    print(f"[epm4-collect]   {_p['name']}: {_n[0]} tokens "
                          f"(+{time.monotonic() - t0:.0f}s)", flush=True)

            def on_complete(_rid, gen_tokens, reason, _lp, _done=done):
                _done.append((list(gen_tokens), reason))

            ts = time.monotonic()
            print(f"[epm4-collect] prompt '{p['name']}' "
                  f"({len(p['token_ids'])} prompt tokens, "
                  f"gen {p['gen_tokens']}) ...", flush=True)
            loop.submit_request(InferenceRequest(
                request_id=rid,
                prompt_token_ids=list(p["token_ids"]),
                gpu=0,
                max_tokens=int(p["gen_tokens"]),
                on_token=on_token,
                on_complete=on_complete,
            ))
            # Per-request deadline: generous, but bounded so one stuck
            # request cannot eat the whole campaign.
            deadline = time.monotonic() + max(
                1800.0,
                args.deadline_per_token_s * p["gen_tokens"] + 600.0)
            while not done and time.monotonic() < deadline:
                loop.run_one_cycle()
            if not done:
                print(f"[epm4-collect] TIMEOUT on '{p['name']}' — aborting "
                      f"further collection (engine state uncertain)",
                      flush=True)
                break
            gen, reason = done[0]
            dt = time.monotonic() - ts
            results.append({
                "name": p["name"], "request_id": rid, "reason": reason,
                "gen_tokens": len(gen), "seconds": round(dt, 1),
                "tok_per_s": round(len(gen) / max(dt, 1e-9), 3),
                # Generated ids: health gates (n-gram loop detection on
                # long gens) + provenance/forced-traj reuse.
                "gen_token_ids": [int(t) for t in gen],
            })
            print(f"[epm4-collect] '{p['name']}' done: {len(gen)} tokens "
                  f"in {dt:.0f}s ({len(gen)/max(dt,1e-9):.2f} tok/s), "
                  f"reason={reason}", flush=True)
    finally:
        layerstorm_engine.stop_engine()
        with open(dump_dir / "collection_results.json", "w") as f:
            json.dump(results, f, indent=2)

    manifest = dump_dir / "manifest.jsonl"
    n_blocks = 0
    if manifest.exists():
        with open(manifest) as f:
            n_blocks = sum(1 for _ in f)
    print(f"[epm4-collect] run complete: {len(results)} sequences, "
          f"{n_blocks} manifest blocks, dump at {dump_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Engine invocations the auto-run pipeline needs (AUTO_RUN steps 1 and 3).

Two of the pipeline's steps cannot be computed — they must be MEASURED on the
box, which means booting the engine:

  step 1  CALIBRATE  boot once with ``gpu_loader.calibration_mode: full``;
                     the engine measures per-device transfer/compute curves
                     and persists the artifact itself
                     (loader_calibration.cpp:1039 load_or_calibrate_with).
  step 3  TRAIN      boot the derived recipe with the loader SHADOW solver and
                     the perf trace on, decode ~100 tokens through the real
                     serving path, then fit the measured wall against the
                     solver's predictions and bake corrected constants
                     (tools/loader_xray/trainer_apply.py).
  step 4  PLACE      fit the per-(layer,expert) demand-fetch table the host
                     arena places by, from step 3's perf trace
                     (tools/loader_xray/freq_table.py). CPU-only when step 3
                     ran; otherwise it needs a capture boot of its own.

Everything here shells out to the same `python/cli/serve.py` a user runs, so
the measurement happens on the production path, not a test binary. No CUDA is
touched in-process (INV-GPU-1): we start a subprocess and speak HTTP to it.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .enginecheck import check_boot_log

# The engine's own boot log voice — these prefixes are what an operator greps.
LOG_PREFIX = "autoconfig:"

# The checkout this package ships in. Tool SCRIPTS live here, not under the
# user's --repo-root (which is a data root: where repo-relative weight paths
# resolve). The two are the same directory in normal use.
PACKAGE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

# A short, deterministic prompt: the training fit wants ~100 DECODE steps
# through the routed-MoE fetch path, not a long prefill.
TRAIN_PROMPT = ("Explain, step by step and in complete sentences, how a "
                "mixture-of-experts transformer routes each token to its "
                "experts during decoding.")
TRAIN_TOKENS = 100


class EngineRunError(RuntimeError):
    """A measured step failed. Carries the log path — the engine's own boot
    log is the authority on why (dossier: read the log, not the exit code)."""

    def __init__(self, message: str, log_path: str = "") -> None:
        super().__init__(message + (f" — see {log_path}" if log_path else ""))
        self.log_path = log_path


@dataclass(frozen=True)
class EngineEnv:
    """How to invoke this checkout's serve stack."""
    repo_root: str
    python: str = ""            # "" -> .venv/bin/python if present, else sys.executable
    cuda_visible_devices: str = ""   # "" -> inherit the caller's setting

    def interpreter(self) -> str:
        if self.python:
            return self.python
        venv = os.path.join(self.repo_root, ".venv", "bin", "python")
        return venv if os.path.exists(venv) else sys.executable

    def base_env(self) -> dict:
        env = dict(os.environ)
        py_paths = [os.path.join(self.repo_root, "build", "python"),
                    os.path.join(self.repo_root, "python")]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join(py_paths + ([existing] if existing else []))
        # PCI_BUS_ID ordering is what hwdetect assumes (ordinal == PCI order)
        # and what every recipe on this project pins.
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        # Deterministic EP combine: greedy decode reproducible run-to-run
        # (recipes/serve_guided.sh ships the same two).
        env.setdefault("LAYERSTORM_DETERMINISTIC_EP_COMBINE", "1")
        env.setdefault("LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION", "bf16")
        return env


def free_port() -> int:
    """An ephemeral port for autoconfig's own boots — never collide with a
    server the operator already has on the recipe's port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def arena_holder_pids() -> list[int]:
    """PIDs of a running `layerstorm_arena_holder`.

    Registry row ``full-calibration-needs-cold-boot``: a calibrating boot
    allocates a ~32 GiB NUMA-bound host footprint early, and against a warm
    holder store the GPU-local node is already ~90% pinned — the kernel
    node-constrained OOM-kills the engine. Detected, not assumed."""
    pids: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        if "layerstorm_arena_holder" in cmd:
            pids.append(int(entry))
    return pids


def engine_pids() -> list[int]:
    """PIDs of a LayerStoRm serving engine already on this box.

    Dossier 1: NEVER two engines — the second one OOM-kills the machine.
    Dossier 12: a `pgrep -f`/`pkill -f` pattern inside a compound command
    matches the command's OWN wrapper shell, so this reads /proc directly
    and never shells out.  The match is deliberately WIDER than `serve.py`:
    a gate harness driving the engine in-process (scratchpad/*_gate.py,
    tools/serve_determinism_probe.py, orch_drive.py) is just as much an
    engine, and a guard that misses it lets two lines share the box.
    """
    # NOT the arena holder: it is not an engine, and a warm holder is the
    # DESIRED state for a benchmark boot (dossier 2).
    needles = ("cli/serve.py", "serve_determinism_probe.py", "orch_drive.py",
               "_gate.py")
    me = os.getpid()
    pids: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace").replace("\0", " ")
        except OSError:
            continue
        if not cmd.strip():
            continue
        if any(n in cmd for n in needles):
            pids.append(pid)
    return pids


def gpu_compute_apps() -> list[dict]:
    """Processes the DRIVER says are holding GPU memory right now.

    The process-name guard cannot see a foreign framework, a stale engine
    under another name, or another agent's harness; the driver can."""
    try:
        out = subprocess.run(  # noqa: S603
            ["nvidia-smi",
             "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    apps: list[dict] = []
    for line in out.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 2 or not f[0].isdigit():
            continue
        try:
            used = int(float(f[1]))
        except ValueError:
            used = 0
        apps.append({"pid": int(f[0]), "used_mib": used,
                     "gpu": f[2] if len(f) > 2 else ""})
    return apps


def _get(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
        return r.status, r.read()


def _post_json(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,  # noqa: S310
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return json.loads(r.read().decode())


class ServeProcess:
    """A serve.py subprocess: boot, wait for /health, drive it, stop it."""

    def __init__(self, config_path: str, env: EngineEnv, log_path: str,
                 extra_env: dict | None = None, port: int = 0,
                 tokenizer_path: str = "") -> None:
        self.config_path = config_path
        self.env = env
        self.log_path = log_path
        self.port = port or free_port()
        self.extra_env = dict(extra_env or {})
        self.tokenizer_path = tokenizer_path
        self.proc: subprocess.Popen | None = None
        self._log = None

    def command(self) -> list[str]:
        cmd = [self.env.interpreter(),
               os.path.join(self.env.repo_root, "python", "cli", "serve.py"),
               "--config", self.config_path,
               "--host", "127.0.0.1", "--port", str(self.port)]
        if self.tokenizer_path:
            cmd += ["--tokenizer-path", self.tokenizer_path]
        return cmd

    def start(self) -> None:
        env = self.env.base_env()
        env.update(self.extra_env)
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)) or ".",
                    exist_ok=True)
        self._log = open(self.log_path, "w")
        self._log.write("# " + " ".join(self.command()) + "\n")
        for k in sorted(self.extra_env):
            self._log.write(f"# env {k}={self.extra_env[k]}\n")
        self._log.flush()
        self.proc = subprocess.Popen(  # noqa: S603
            self.command(), cwd=self.env.repo_root, env=env,
            stdout=self._log, stderr=subprocess.STDOUT,
            start_new_session=True)

    def wait_ready(self, timeout_s: float, poll_s: float = 2.0) -> None:
        """Poll /health until the stack serves. A cold boot is ~3.5 min and a
        FULL calibration adds ~3 min, so the default budget is generous; a
        process that EXITS fails immediately rather than burning the budget."""
        assert self.proc is not None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                raise EngineRunError(
                    f"serve exited with status {rc} before becoming ready",
                    self.log_path)
            try:
                status, _ = _get(f"http://127.0.0.1:{self.port}/health")
                if status == 200:
                    return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(poll_s)
        raise EngineRunError(
            f"serve did not report healthy within {timeout_s:.0f}s",
            self.log_path)

    def model_id(self) -> str:
        status, body = _get(f"http://127.0.0.1:{self.port}/v1/models", timeout=10)
        if status != 200:
            raise EngineRunError("GET /v1/models failed", self.log_path)
        data = json.loads(body.decode())
        items = data.get("data") or []
        if not items:
            raise EngineRunError("/v1/models returned no model", self.log_path)
        return str(items[0].get("id", ""))

    def decode(self, prompt: str, max_tokens: int, timeout_s: float) -> dict:
        """One greedy completion — the measured decode the trainer fits."""
        return self.complete(self.model_id(), prompt, max_tokens, timeout_s)

    def complete(self, model_id: str, prompt, max_tokens: int,
                 timeout_s: float) -> dict:
        """One greedy completion against an ALREADY-RESOLVED model id.

        ``prompt`` is text OR a list of token ids (the OpenAI-compatible
        surface accepts both: CompletionRequest.prompt is ``str |
        list[int]``).  Token ids are what the benchmark's prefill ladder
        uses — they hit an EXACT prompt length, which text cannot.  Taking
        the model id as an argument keeps a measured request from paying an
        extra /v1/models round trip inside its own timing window."""
        return _post_json(
            f"http://127.0.0.1:{self.port}/v1/completions",
            {"model": model_id, "prompt": prompt,
             "max_tokens": max_tokens, "temperature": 0.0, "stream": False},
            timeout=timeout_s)

    def stop(self, timeout_s: float = 180.0) -> int:
        """SIGINT the process group: serve.py's run_stack() catches
        KeyboardInterrupt and calls stack.shutdown(), which is what flushes
        the shadow dump and the perf trace. Escalates if it hangs."""
        if self.proc is None:
            return 0
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self.proc.wait(timeout=30)
        rc = self.proc.returncode or 0
        if self._log is not None:
            self._log.close()
            self._log = None
        return rc

    def __enter__(self) -> "ServeProcess":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# ------------------------------------------------------------------ steps

def cross_check_boot(config_path: str, env: EngineEnv, log_path: str,
                     log=print) -> list:
    """Hold the solver's model of the engine against the engine's own boot log.

    EVERY boot this module performs runs this, and it lives HERE rather than at
    the pipeline's call sites for one reason: the function that produced the log
    is the one that can never forget to check it. A future measured step gets
    the cross-check by construction.

    Drift is a HARD failure (``EngineModelDrift``), not a warning, for exact
    rows — the same judgement ``run_calibration_boot`` already applies to a
    stale artifact mtime: when the engine demonstrably did something other than
    what we predicted, the recipe we are about to emit is derived from a wrong
    model of it, and emitting it anyway is how TD-AUTOCONFIG-PINNED-BYTES-
    UPPER-BOUND produced an unbootable default path in the first place. A row
    whose solver value is only a STATED BOUND warns when it errs in its own
    direction and fails when the bound is violated (enginecheck.CheckResult).
    """
    # Unreadable INPUTS are not drift and must not fail the step that produced
    # them: the boot's own guards already own "did this run at all". Report it
    # loudly as UNCHECKED instead — the one thing this module may never do is
    # let silence read as agreement.
    try:
        with open(config_path) as f:
            recipe = json.load(f)
    except (OSError, ValueError) as e:
        log(f"{LOG_PREFIX} engine cross-check UNCHECKED — cannot read the "
            f"recipe {config_path}: {e}")
        return []
    if not os.path.exists(log_path):
        log(f"{LOG_PREFIX} engine cross-check UNCHECKED — no boot log at "
            f"{log_path}")
        return []
    weights = (recipe.get("model") or {}).get("weights_path", "")
    weights_abs = ""
    for root in (env.repo_root, "."):
        cand = os.path.join(root, weights)
        if weights and os.path.exists(cand):
            weights_abs = cand
            break
    return check_boot_log(log_path, recipe, weights_abs, log=log)


def run_calibration_boot(config_path: str, env: EngineEnv, log_path: str,
                         artifact_path: str, tokenizer_path: str = "",
                         timeout_s: float = 2400.0,
                         allow_warm_holder: bool = False) -> None:
    """AUTO_RUN step 1: boot once so the engine measures and persists.

    The recipe passed here must already carry ``calibration_mode: full`` (the
    solver emits it whenever no artifact is accepted). Success is the
    ARTIFACT existing afterwards — a boot that serves without writing it means
    the engine loaded a different path than we predicted."""
    pids = arena_holder_pids()
    if pids and not allow_warm_holder:
        raise EngineRunError(
            "an arena holder is running (pids " + ", ".join(map(str, pids))
            + ") and a calibrating boot must run COLD: the ~32 GiB "
            "NUMA-bound calibration footprint node-OOMs against a warm "
            "holder store (registry row full-calibration-needs-cold-boot). "
            "Kill it, wait for `free -g` shared to drain, then re-run "
            "(or pass --allow-warm-holder to override)")
    before = os.path.getmtime(artifact_path) if os.path.exists(artifact_path) else 0.0
    proc = ServeProcess(config_path, env, log_path, tokenizer_path=tokenizer_path)
    with proc:
        proc.wait_ready(timeout_s)
    if not os.path.exists(artifact_path):
        raise EngineRunError(
            f"calibration boot finished but {artifact_path} was not written",
            log_path)
    if os.path.getmtime(artifact_path) <= before:
        raise EngineRunError(
            f"calibration boot did not refresh {artifact_path} (stale mtime) "
            "— the engine resolved a different calibration_path than "
            "predicted (engine.cpp:500-512 joins relative paths to the "
            "weights dir)", log_path)
    cross_check_boot(config_path, env, log_path)


def run_training_boot(config_path: str, env: EngineEnv, log_path: str,
                      dump_path: str, trace_path: str,
                      tokenizer_path: str = "",
                      prompt: str = TRAIN_PROMPT, max_tokens: int = TRAIN_TOKENS,
                      timeout_s: float = 2400.0,
                      decode_timeout_s: float = 900.0) -> dict:
    """AUTO_RUN step 3a: the measured 100-token decode.

    LS_LOADER_SHADOW makes the loader solver run in shadow (behaviour-neutral,
    INV-LOADER-XRAY-REKEY) and dump one record per solve; LS_PERF_TRACE
    captures the real fetch/DMA/compute timings the fit regresses against."""
    for p in (dump_path, trace_path):
        os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
        if os.path.exists(p):
            os.remove(p)
    extra = {
        "LS_LOADER_SHADOW": "1",
        "LS_LOADER_SHADOW_DUMP": dump_path,
        "LS_PERF_TRACE": "1",
        "LS_PERF_TRACE_OUT": trace_path,
    }
    proc = ServeProcess(config_path, env, log_path, extra_env=extra,
                        tokenizer_path=tokenizer_path)
    with proc:
        proc.wait_ready(timeout_s)
        result = proc.decode(prompt, max_tokens, decode_timeout_s)
    for p, what in ((dump_path, "shadow dump"), (trace_path, "perf trace")):
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            raise EngineRunError(
                f"training decode produced no {what} at {p}", log_path)
    cross_check_boot(config_path, env, log_path)
    return result


def run_trainer_apply(repo_root: str, in_calib: str, dump_path: str,
                      trace_path: str, out_calib: str, log_path: str,
                      model: str = "current", python: str = "") -> None:
    """AUTO_RUN step 3b: fit + bake (CPU-only).

    Same invocation the gtest TearDown hook uses
    (keeper52_test.cpp:925-968) — one trainer, one schema, so a trained
    artifact from autoconfig and one from the benchmark path are the same
    kind of file."""
    xray = os.path.join(PACKAGE_ROOT, "tools", "loader_xray")
    cmd = [python or sys.executable, os.path.join(xray, "trainer_apply.py"),
           "--in-calib", in_calib, "--model-jsonl", dump_path,
           "--trace", trace_path, "--model", model, "--out-calib", out_calib]
    with open(log_path, "w") as log:
        log.write("# " + " ".join(cmd) + "\n")
        log.flush()
        # trainer_apply.py imports its siblings (joiner/model/trainer) by
        # bare name — it must run with tools/loader_xray on sys.path.
        env = dict(os.environ)
        env["PYTHONPATH"] = xray + (":" + env["PYTHONPATH"]
                                    if env.get("PYTHONPATH") else "")
        rc = subprocess.call(cmd, cwd=xray, env=env,  # noqa: S603
                             stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        raise EngineRunError(f"trainer_apply.py exited {rc}", log_path)
    if not os.path.exists(out_calib):
        raise EngineRunError(
            f"trainer_apply.py reported success but {out_calib} is missing",
            log_path)

def run_capture_boot(config_path: str, env: EngineEnv, log_path: str,
                     trace_path: str, tokenizer_path: str = "",
                     prompt: str = TRAIN_PROMPT, max_tokens: int = TRAIN_TOKENS,
                     timeout_s: float = 2400.0,
                     decode_timeout_s: float = 900.0) -> dict:
    """A decode that captures ONLY the perf trace (AUTO_RUN step 4 without a
    step 3 to ride on). Same workload as the training decode; no shadow
    solver, so nothing about the run is diagnostic-weighted."""
    os.makedirs(os.path.dirname(os.path.abspath(trace_path)) or ".", exist_ok=True)
    if os.path.exists(trace_path):
        os.remove(trace_path)
    proc = ServeProcess(config_path, env, log_path,
                        extra_env={"LS_PERF_TRACE": "1",
                                   "LS_PERF_TRACE_OUT": trace_path},
                        tokenizer_path=tokenizer_path)
    with proc:
        proc.wait_ready(timeout_s)
        result = proc.decode(prompt, max_tokens, decode_timeout_s)
    if not os.path.exists(trace_path) or os.path.getsize(trace_path) == 0:
        raise EngineRunError(f"capture decode produced no perf trace at "
                             f"{trace_path}", log_path)
    cross_check_boot(config_path, env, log_path)
    return result


def run_freq_table_fit(repo_root: str, out_csv: str, traces: list,
                       log_path: str, python: str = "") -> None:
    """AUTO_RUN step 4: trace -> per-(layer,expert) demand-fetch table
    (CPU-only).

    ``tools/loader_xray/freq_table.py`` counts DISPATCHED expert H2D copies
    per key, excluding prefill and warm-up sweeps — the quantity the M3
    placement policy optimises (bytes on the wire per key), not routing
    frequency. Multiple traces accumulate; we pass what we captured."""
    script = os.path.join(PACKAGE_ROOT, "tools", "loader_xray", "freq_table.py")
    cmd = [python or sys.executable, script, out_csv] + list(traces)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(log_path, "w") as log:
        log.write("# " + " ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.call(cmd, cwd=PACKAGE_ROOT,  # noqa: S603
                             stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        raise EngineRunError(f"freq_table.py exited {rc}", log_path)
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        raise EngineRunError(
            f"freq_table.py wrote no table at {out_csv} — the trace may hold "
            "no counted decode fetches (all prefill / warm-up sweeps)",
            log_path)

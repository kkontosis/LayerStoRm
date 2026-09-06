"""Cross-check the solver's model of the engine against the engine's OWN boot log.

Three tickets filed in three days were the SAME bug wearing different clothes:
the solver models a quantity the engine computes differently, nothing notices,
and the recipe is wrong in a way that only shows up as a dead boot or a refused
request.

  TD-AUTOCONFIG-PINNED-BYTES-UPPER-BOUND   pinned region: solver summed STORED
      bytes (9384.7 MiB), engine reserved a REGION (16073.0 MiB). -41.6%.
  TD-AUTOCONFIG-MAXSEQ-IGNORES-MAPPED-KDA  mapped-KDA state: the solver's
      admissibility arithmetic and the engine's slab-padded per-request demand.
  TD-GLM5-TQ-BACKEND-UNWIRED (fallout)     KV row bytes: with the attention
      backend flippable again, snapmla 516 B/token vs turboquant_mla 258 is a
      2x disagreement waiting to happen.

Both owning tickets say the same thing: fix it with a BOOT-LOG CROSS-CHECK, not
by patching constants. So this module is a TABLE, not a pair of one-offs — the
fourth member of the family should be a row here, not new code:

    CHECKS = (CheckRow(name, unit, tolerance, solver_site, engine_site,
                       solve=<solver quantity>, observe=<boot-log field>), ...)

The engine prints every one of these figures itself at boot. Parsing them back
costs nothing (the log already exists on every measured step) and turns "the
solver believes X" into an assertion against the only authority that matters.

HONESTY RULE. A row's ``solve`` returns a :class:`SolverFigure` that says
whether the number is byte-EXACT against the engine or a stated-direction
BOUND. An exact row that diverges is a hard failure. A bound row that diverges
in its stated direction is a warning; one that diverges the WRONG way is a hard
failure too, because a violated bound is exactly the dangerous case. A row that
cannot be computed for this architecture returns ``None`` and is reported as
UNCHECKED — silence must never read as agreement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from . import gguf_meta, sizing
from .modelshape import ModelShape

MIB = 1 << 20


# ───────────────────────────────────────────────────── the engine's boot log

@dataclass(frozen=True)
class GpuBudget:
    """One ``VramAllocator: GPU <i> (hw id <h>, <role>) budget:`` line."""
    index: int                 # logical GPU index (recipe hardware.gpus order)
    hw_id: int                 # CUDA ordinal under CUDA_DEVICE_ORDER=PCI_BUS_ID
    role: str                  # "TP" | "expert-only"
    total_mib: float
    pinned_mib: float


@dataclass(frozen=True)
class BootFigures:
    """Every engine-computed figure this module knows how to read back.

    Fields are ``None`` when the log did not print them (a model with no DSA
    prints no slab geometry; a model with no linear-attention layers prints no
    KDA pool line). ``None`` means UNOBSERVED, never zero."""
    source: str = ""
    budgets: tuple = ()                     # tuple[GpuBudget, ...]
    summary_pinned_mib: dict = field(default_factory=dict)   # hw_id -> MiB
    kv_bytes_per_page: int | None = None
    slab_pages: int | None = None
    slab_bytes: int | None = None
    kda_slot_mib: float | None = None
    kda_request_mib: float | None = None
    kda_linear_layers: int | None = None
    kda_slabs_per_layer: int | None = None
    # TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA: the engine's own single-request
    # admissible-context ceiling and its component arithmetic, read off the
    # 'admissible context (single request)' boot line (first TP GPU).
    admissible_ctx_tokens: int | None = None
    admission_demand_pages: int | None = None
    admission_kv_pages: int | None = None
    admission_indexer_pages: int | None = None
    admission_state_pages: int | None = None
    admission_pool_pages: int | None = None

    @property
    def tp_pinned_mib(self) -> float | None:
        """The pinned figure of a TP rank — every TP rank carries the same
        region, so any of them answers for the recipe."""
        tp = [b.pinned_mib for b in self.budgets if b.role == "TP"]
        if tp:
            return max(tp)
        # No role marker (older log format): fall back to the largest pinned
        # figure the summary block reported.
        return max(self.summary_pinned_mib.values()) if self.summary_pinned_mib else None


# `VramAllocator: GPU 0 (hw id 2, TP) budget: total 30720.0 MiB = pinned 16073.0 + margin ...`
_BUDGET_RE = re.compile(
    r"VramAllocator: GPU (\d+) \(hw id (\d+), ([^)]*)\) budget: "
    r"total ([\d.]+) MiB = pinned ([\d.]+)")
# `VramAllocator GPU 2: 30720 MiB total, 28416 MiB allocated ...`
_SUMMARY_GPU_RE = re.compile(r"VramAllocator GPU (\d+): ")
# `  pinned weights    16073.0 MiB  | kv_main ...`
_SUMMARY_PINNED_RE = re.compile(r"pinned weights\s+([\d.]+) MiB")
# `VramAllocator: S1 slab geometry — slab 272448 B = 33 kMain pages x 8256 B ...`
_SLAB_RE = re.compile(
    r"S1 slab geometry .* slab (\d+) B = (\d+) kMain pages x (\d+) B")
# `kda_state (GF3.8)  0.0 MiB (0 slots x 145.6 MiB — ...)`
_KDA_SLOT_RE = re.compile(r"kda_state \(GF3\.8\)\s+[\d.]+ MiB \(\d+ slots x ([\d.]+) MiB")
# `Per-request demand = 34 linear layers x 17 slabs (4.28 MiB/layer unit, 150.2 MiB/request ...`
_KDA_DEMAND_RE = re.compile(
    r"Per-request demand = (\d+) linear layers x (\d+) slabs "
    r"\([\d.]+ MiB/layer unit, ([\d.]+) MiB/request")
# `VramAllocator: GPU 0 admissible context (single request) = 470096 tokens
#  — max_sequence_length 500000 needs 380270 kMain pages (KV 375000 +
#  indexer 8712 + KDA state 19074) vs pool 373527 pages`
_ADMISSIBLE_RE = re.compile(
    r"admissible context \(single request\) = (\d+) tokens .* "
    r"max_sequence_length (\d+) needs (\d+) kMain pages "
    r"\(KV (\d+) \+ indexer (\d+) \+ KDA state (\d+)\) "
    r"vs pool (\d+) pages")


def parse_boot_log(text: str, source: str = "") -> BootFigures:
    """Read every engine-computed figure this module checks out of one boot log.

    The engine writes these lines itself (vram_allocator.cpp's budget/summary
    reporting); they are the authority the solver is measured against."""
    budgets: list[GpuBudget] = []
    summary: dict = {}
    kv_bpp = slab_pages = slab_bytes = None
    kda_slot = kda_request = None
    kda_layers = kda_slabs = None
    adm = None       # (ceiling, demand, kv, idx, state, pool)
    current_hw: int | None = None
    for line in text.splitlines():
        m = _BUDGET_RE.search(line)
        if m:
            budgets.append(GpuBudget(index=int(m.group(1)), hw_id=int(m.group(2)),
                                     role=m.group(3).strip(),
                                     total_mib=float(m.group(4)),
                                     pinned_mib=float(m.group(5))))
            continue
        m = _SUMMARY_GPU_RE.search(line)
        if m:
            current_hw = int(m.group(1))
            continue
        m = _SUMMARY_PINNED_RE.search(line)
        if m and current_hw is not None:
            summary[current_hw] = float(m.group(1))
            continue
        m = _SLAB_RE.search(line)
        if m:
            slab_bytes = int(m.group(1))
            slab_pages = int(m.group(2))
            kv_bpp = int(m.group(3))
            continue
        m = _KDA_SLOT_RE.search(line)
        if m:
            kda_slot = float(m.group(1))
            continue
        m = _KDA_DEMAND_RE.search(line)
        if m:
            kda_layers = int(m.group(1))
            kda_slabs = int(m.group(2))
            kda_request = float(m.group(3))
            continue
        m = _ADMISSIBLE_RE.search(line)
        if m and adm is None:   # first TP GPU answers for the recipe
            adm = (int(m.group(1)), int(m.group(3)), int(m.group(4)),
                   int(m.group(5)), int(m.group(6)), int(m.group(7)))
    return BootFigures(source=source, budgets=tuple(budgets),
                       summary_pinned_mib=summary, kv_bytes_per_page=kv_bpp,
                       slab_pages=slab_pages, slab_bytes=slab_bytes,
                       kda_slot_mib=kda_slot, kda_request_mib=kda_request,
                       kda_linear_layers=kda_layers,
                       kda_slabs_per_layer=kda_slabs,
                       admissible_ctx_tokens=adm[0] if adm else None,
                       admission_demand_pages=adm[1] if adm else None,
                       admission_kv_pages=adm[2] if adm else None,
                       admission_indexer_pages=adm[3] if adm else None,
                       admission_state_pages=adm[4] if adm else None,
                       admission_pool_pages=adm[5] if adm else None)


def read_boot_log(path: str) -> BootFigures:
    with open(path, errors="replace") as f:
        return parse_boot_log(f.read(), path)


# ─────────────────────────────────────────────────────────── the check table

@dataclass(frozen=True)
class CheckContext:
    """What a row's ``solve`` gets: the recipe that was BOOTED, plus the
    weights path so checkpoint-derived quantities can be recomputed."""
    recipe: dict
    weights_path: str = ""

    @property
    def model(self) -> dict:
        return self.recipe.get("model") or {}

    @property
    def shape(self) -> ModelShape:
        # P-29 step 13: from the WHOLE booted recipe — speculation.method "mtp"
        # arms the engine's extended MoE census, and a check that planned
        # the un-extended one would mis-read the boot log.
        return ModelShape.from_config(self.recipe)

    @property
    def tensor_parallelism(self) -> int:
        return max(1, int((self.recipe.get("parallelism") or {})
                          .get("tensor_parallelism", 1) or 1))


@dataclass(frozen=True)
class SolverFigure:
    value: float
    exact: bool = True
    direction: str = "exact"      # "exact" | "over" | "under"
    note: str = ""


@dataclass(frozen=True)
class CheckRow:
    name: str
    unit: str
    tolerance: float              # in `unit`; the log's own print precision
    solver_site: str
    engine_site: str
    solve: Callable
    observe: Callable
    # An INEQUALITY row checks a CONSTRAINT, not an estimate: the solver
    # value is a different quantity (e.g. the ADVERTISED max_seq) that must
    # sit on the stated side of the engine figure (solve's `direction`:
    # "under" = solver <= engine). Headroom is then AGREEMENT — the healthy
    # state, not a divergence to warn about — and a violated bound is the
    # hard failure. Estimate rows keep the original semantics: a bound that
    # errs in its own direction stays a logged warning (conservative, not
    # wrong), because there the solver value CLAIMS to track the engine's.
    inequality: bool = False


@dataclass(frozen=True)
class CheckResult:
    row: CheckRow
    status: str                   # "agree" | "diverge" | "unchecked"
    solver: SolverFigure | None = None
    engine: float | None = None
    reason: str = ""              # why unchecked

    @property
    def delta(self) -> float:
        if self.solver is None or self.engine is None:
            return 0.0
        return self.solver.value - self.engine

    @property
    def fatal(self) -> bool:
        """A divergence is FATAL when the solver claimed exactness, or when a
        stated bound was violated (solver said "over" and came in under, or
        vice versa). A bound that erred in its own stated direction is a
        warning: it is conservative, not wrong."""
        if self.status != "diverge":
            return False
        if self.solver is None or self.solver.exact:
            return True
        if self.solver.direction == "over":
            return self.delta < 0
        if self.solver.direction == "under":
            return self.delta > 0
        return True

    def describe(self) -> str:
        if self.status == "unchecked":
            return (f"{self.row.name}: UNCHECKED — {self.reason} "
                    f"(nothing was compared; this is not agreement)")
        assert self.solver is not None and self.engine is not None
        sign = "+" if self.delta >= 0 else "-"
        claim = ("EXACT" if self.solver.exact
                 else f"bound (stated {self.solver.direction})")
        head = "AGREE" if self.status == "agree" else "DIVERGE"
        out = (f"{self.row.name}: {head} — solver {self.solver.value:g} "
               f"{self.row.unit} [{claim}] vs engine {self.engine:g} "
               f"{self.row.unit}, delta {sign}{abs(self.delta):g} "
               f"{self.row.unit} (tolerance {self.row.tolerance:g})")
        if self.status == "diverge":
            out += (f"\n    solver site: {self.row.solver_site}"
                    f"\n    engine site: {self.row.engine_site}")
            if self.solver.note:
                out += f"\n    solver note: {self.solver.note}"
        return out


# ── rows ────────────────────────────────────────────────────────────────────

def _solve_pinned(ctx: CheckContext) -> SolverFigure | None:
    mem = ctx.recipe.get("memory") or {}
    lay = gguf_meta.pinned_region_layout(
        ctx.model, ctx.recipe.get("quantization") or {},
        ctx.tensor_parallelism, ctx.weights_path,
        mem.get("pinned_layers"), mem.get("tp_mode_per_layer"))
    return SolverFigure(value=lay.total_bytes / MIB, exact=lay.exact,
                        direction=lay.direction, note=lay.provenance)


def _solve_kv_page(ctx: CheckContext) -> SolverFigure | None:
    s = ctx.shape
    if s.is_v4:
        return None      # V4 has no uniform per-token size (sizing.py throws)
    backend = str((ctx.recipe.get("compute") or {}).get("attention_backend", ""))
    if not backend:
        return None
    page = int(((ctx.recipe.get("memory") or {}).get("kv_cache") or {})
               .get("page_size_tokens", 0) or 0)
    if page <= 0:
        return None
    kv_quant = str((ctx.recipe.get("quantization") or {}).get("kv_cache", "fp8_e4m3"))
    return SolverFigure(
        value=float(sizing.kv_bytes_per_page(s, kv_quant, backend, page)))


def _solve_kda_slot(ctx: CheckContext) -> SolverFigure | None:
    s = ctx.shape
    if s.num_linear_attention_layers <= 0:
        return None
    return SolverFigure(
        value=sizing.kda_slot_bytes(s, ctx.tensor_parallelism) / MIB)


def _solve_kda_request(ctx: CheckContext) -> SolverFigure | None:
    """The MAPPED per-request demand (TD-KDA-STATE-MAPPED-SLABS): slab-padded,
    which is what the engine actually claims — 150.2 MiB/request against a
    145.6 MiB slot on GLM-5.3-Flash. This is the quantity
    TD-AUTOCONFIG-MAXSEQ-IGNORES-MAPPED-KDA says the admissibility arithmetic
    must use."""
    s = ctx.shape
    if s.num_linear_attention_layers <= 0:
        return None
    kv = (ctx.recipe.get("memory") or {}).get("kv_cache") or {}
    backend = str((ctx.recipe.get("compute") or {}).get("attention_backend", ""))
    page = int(kv.get("page_size_tokens", 0) or 0)
    idx_page = int(kv.get("indexer_k_page_size_tokens", 0) or 0)
    if not backend or page <= 0 or idx_page <= 0:
        return None
    kv_quant = str((ctx.recipe.get("quantization") or {}).get("kv_cache", "fp8_e4m3"))
    geo = sizing.slab_geometry(s, kv_quant, backend, page, idx_page)
    if geo is None:
        return None
    demand = sizing.kda_mapped_demand_bytes(s, ctx.tensor_parallelism,
                                            geo.slab_bytes)
    return SolverFigure(value=demand / MIB)




def _admission_geometry(ctx: CheckContext):
    """Shared inputs of the two admissibility rows; None when the recipe
    does not carry enough geometry (then the rows report UNCHECKED)."""
    sshape = ctx.shape
    if sshape.is_v4:
        return None
    kv = (ctx.recipe.get("memory") or {}).get("kv_cache") or {}
    backend = str((ctx.recipe.get("compute") or {}).get("attention_backend", ""))
    page = int(kv.get("page_size_tokens", 0) or 0)
    idx_page = int(kv.get("indexer_k_page_size_tokens", 0) or 0)
    max_seq = int((ctx.recipe.get("serving") or {})
                  .get("max_sequence_length", 0) or 0)
    if not backend or page <= 0 or max_seq <= 0:
        return None
    kv_quant = str((ctx.recipe.get("quantization") or {}).get("kv_cache",
                                                             "fp8_e4m3"))
    geo = None
    if sshape.has_dsa and idx_page > 0:
        geo = sizing.slab_geometry(sshape, kv_quant, backend, page, idx_page)
    pps = geo.pages_per_slab if geo else 0
    mapped = bool((ctx.recipe.get("_internal-kda_state") or {})
                  .get("mapped", True))
    state_pages = (sizing.kda_mapped_state_pages(
                       sshape, ctx.tensor_parallelism,
                       geo.slab_bytes if geo else 0, pps)
                   if mapped and geo else 0)
    hw = ctx.recipe.get("hardware") or {}
    tp = ctx.tensor_parallelism
    kv_shard = tp if (tp >= 2 and str(hw.get("dcp_kv_mode", "")) == "sharded")         else 1
    idx_shard = tp if (tp >= 2
                       and str(hw.get("dcp_indexer_mode", "")) == "local") else 1
    chunk = int(kv.get("dcp_chunk_size", 0) or 0)
    dsa_layers = (sshape.num_dsa_computing_layers
                  if (sshape.has_dsa and pps > 0) else 0)
    return (max_seq, page, sshape.engine_kv_pool_layers(), dsa_layers,
            idx_page, pps, state_pages, kv_shard, chunk, idx_shard)


def _solve_admission_demand(ctx: CheckContext) -> SolverFigure | None:
    """One max-length request's whole-life kMain-page demand — the sum the
    engine prints on the 'admissible context (single request)' line
    (TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA). Pure geometry, no VRAM model:
    a divergence here is a transcription bug, never a sizing decision."""
    g = _admission_geometry(ctx)
    if g is None:
        return None
    (max_seq, page, layers, dsa_layers, idx_page, pps, state_pages,
     kv_shard, chunk, idx_shard) = g
    return SolverFigure(value=float(sizing.single_request_demand_pages(
        max_seq, page, layers, dsa_layers, idx_page, pps, state_pages,
        kv_shard, chunk, idx_shard)))


def _solve_advertised_max_seq(ctx: CheckContext) -> SolverFigure | None:
    """The recipe's ADVERTISED serving.max_sequence_length, checked as a
    bound against the engine's own admissible ceiling: advertised must be
    <= the ceiling (direction 'under'), or the recipe promises a context
    its own admission path refuses — the GF3 1M-arm defect
    (TD-AUTOCONFIG-MAXSEQ-IGNORES-MAPPED-KDA). This is the row that
    catches ANY model error upstream (pinned drift, KV-row drift, a new
    tenant): whatever the cause, an inadmissible advertisement lands
    here as a violated bound = hard failure."""
    max_seq = int((ctx.recipe.get("serving") or {})
                  .get("max_sequence_length", 0) or 0)
    if max_seq <= 0:
        return None
    return SolverFigure(
        value=float(max_seq), exact=False, direction="under",
        note="advertised context; must not exceed the engine's admissible "
             "single-request ceiling")


CHECKS: tuple = (
    CheckRow(
        name="pinned region (per TP rank)", unit="MiB", tolerance=1.0,
        solver_site="autoconfig.gguf_meta.pinned_region_layout",
        engine_site="compute_pinned_layout (src/model/pinned_region_layout.cpp:565)",
        solve=_solve_pinned, observe=lambda f: f.tp_pinned_mib),
    CheckRow(
        name="KV bytes per page", unit="B", tolerance=0.0,
        solver_site="autoconfig.sizing.kv_bytes_per_page",
        engine_site="VramAllocator::kv_bytes_per_page (src/memory/vram_allocator.cpp:70-75)",
        solve=_solve_kv_page, observe=lambda f: f.kv_bytes_per_page),
    CheckRow(
        name="KDA state slot", unit="MiB", tolerance=0.05,
        solver_site="autoconfig.sizing.kda_slot_bytes",
        engine_site="compute_kda_state_layout (src/memory/vram_allocator.cpp:236-277)",
        solve=_solve_kda_slot, observe=lambda f: f.kda_slot_mib),
    CheckRow(
        name="KDA mapped demand per request", unit="MiB", tolerance=0.05,
        solver_site="autoconfig.sizing.kda_mapped_demand_bytes",
        engine_site="mapped KDA slab claim (src/memory/vram_allocator.cpp, "
                    "TD-KDA-STATE-MAPPED-SLABS)",
        solve=_solve_kda_request, observe=lambda f: f.kda_request_mib),
    CheckRow(
        name="single-request admission demand at max_seq", unit="pages",
        tolerance=0.0,
        solver_site="autoconfig.sizing.single_request_demand_pages",
        engine_site="admissible-ceiling boot line (src/core/memory/"
                    "vram_allocator.cpp, TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA)",
        solve=_solve_admission_demand,
        observe=lambda f: f.admission_demand_pages),
    CheckRow(
        name="advertised max_sequence_length vs admissible ceiling",
        unit="tokens", tolerance=0.0,
        solver_site="recipe serving.max_sequence_length "
                    "(autoconfig.solver reduce-or-refuse)",
        engine_site="admissible-ceiling boot line (src/core/memory/"
                    "vram_allocator.cpp, TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA)",
        solve=_solve_advertised_max_seq,
        observe=lambda f: f.admissible_ctx_tokens, inequality=True),
)


# ── running the table ───────────────────────────────────────────────────────

def cross_check(figures: BootFigures, ctx: CheckContext,
                rows: tuple = CHECKS) -> list:
    """Run every row. Returns one CheckResult per row, in table order."""
    out: list = []
    for row in rows:
        engine = row.observe(figures)
        if engine is None:
            out.append(CheckResult(row, "unchecked", reason=(
                "the engine's boot log did not report this figure "
                f"({figures.source or 'log'})")))
            continue
        try:
            solver = row.solve(ctx)
        except gguf_meta.UnmodelledArchitecture as e:
            out.append(CheckResult(row, "unchecked", engine=float(engine),
                                   reason=f"not modelled for this model: {e}"))
            continue
        if solver is None:
            out.append(CheckResult(row, "unchecked", engine=float(engine),
                                   reason="the solver derives no such quantity "
                                          "for this model"))
            continue
        if row.inequality:
            ok = (solver.value <= float(engine) + row.tolerance
                  if solver.direction == "under"
                  else solver.value >= float(engine) - row.tolerance)
            out.append(CheckResult(row, "agree" if ok else "diverge",
                                   solver=solver, engine=float(engine)))
            continue
        agree = abs(solver.value - float(engine)) <= row.tolerance
        out.append(CheckResult(row, "agree" if agree else "diverge",
                               solver=solver, engine=float(engine)))
    return out


def format_report(results: list, source: str = "") -> str:
    head = f"engine cross-check ({source})" if source else "engine cross-check"
    return "\n".join([head] + ["  " + r.describe() for r in results])


class EngineModelDrift(RuntimeError):
    """The solver's model of the engine and the engine's own boot log disagree.

    Raised, not warned, because the pipeline's whole output is a recipe derived
    from that model: a recipe fitted to a wrong number is exactly the artifact
    this pipeline must never emit (the same reason ``run_calibration_boot``
    raises when the engine resolved a different calibration path than
    predicted). Whoever sees this fixes the transcription or the engine, not
    the constant."""


def check_boot_log(log_path: str, recipe: dict, weights_path: str = "",
                   log=print, rows: tuple = CHECKS) -> list:
    """Cross-check one boot's log against the recipe that produced it.

    Logs every row's verdict, warns on a conservative bound that drifted, and
    RAISES :class:`EngineModelDrift` on any fatal divergence (see
    ``CheckResult.fatal``)."""
    figures = read_boot_log(log_path)
    results = cross_check(figures, CheckContext(recipe, weights_path), rows)
    fatal = [r for r in results if r.fatal]
    for r in results:
        if r.status == "agree":
            continue
        log("autoconfig: " + r.describe().replace("\n", "\n           "))
    if fatal:
        raise EngineModelDrift(
            "the solver's model of the engine disagrees with the engine's own "
            f"boot log ({log_path}):\n" +
            "\n".join("  " + r.describe() for r in fatal))
    return results

"""Combination-keyed per-model constraint registry
(TD-AUTOCONFIG-COMBO-CONSTRAINTS) — DATA, not logic.

`engine_constraints.py` holds rules keyed by kind + a single arch scope.
This registry holds the rules that bind a COMBINATION of choices — the
first: *arch == glm5_next AND tensor_parallelism > 1 ⇒ hardware.dcp_kv_mode
= replicated* (the engine fail-closes on sequence-sharded KV for the arch,
src/daemon/engine.cpp:637, and a solver that does not know that emits a
recipe that cannot boot — TD-AUTOCONFIG-GLM5NEXT-TP-SHARDED-KV).

The rows live in ``model_constraints.json`` BESIDE THIS FILE — shipped
package data, not user config, loaded once at import and validated loudly.
JSON rather than a Python tuple because a combination row is the purest
form of the registry discipline: deleting one is a data edit no linter,
import or reviewer of code ever sees, the file is diffable/greppable on its
own, and a future non-Python consumer (the engine, tooling) can read the
same bytes. Import-time load keeps it as static as compiling it in.

Semantics the solver relies on:

* a row is CONSULTED only when every predicate in ``when`` matches the
  derivation's context — a tp=1 glm5_next derivation never sees the row,
  so nothing is over-constrained (the user's design note, verbatim);
* predicate keys are a CLOSED SET (``PREDICATE_KEYS``): an unknown key in
  the JSON refuses at import (a silently-skipped predicate would emit
  non-booting recipes — fail-closed, never fail-open), and a context that
  omits a key a row needs refuses at match time for the same reason.  A
  GPU predicate ("card class X AND feature Y ⇒ Z") is a NEW KEY here plus
  the ctx entry that feeds it — not a new mechanism;
* a firing row is an ENGINE-CURRENT FACT: it explains with its own kind
  (``engine_gate``) citing the enforcing code site, and a contradicting
  ``--pin`` REFUSES quoting both — never a silent override
  (TD-AUTOCONFIG-PINNED-CONSTRAINTS discipline).

Deleting a row re-derives the config with the rule gone, with no code
edit — the delete-the-row property tests keep that meaningful.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping

_JSON_PATH = os.path.join(os.path.dirname(__file__), "model_constraints.json")

# A combination row states what the engine DOES (grep the site), or what a
# validator rejects. Never `measured`/`heuristic`: re-measuring cannot
# change it; only an engine commit can, and that commit deletes the row.
KINDS = ("engine_gate", "validator")

# The closed predicate vocabulary. Every solver call site must supply ALL
# of these in its ctx (cheap, and keeps match-time failures impossible for
# shipped rows); adding a GPU predicate = appending here + at the ctx
# builders (solver._combo_ctx).
PREDICATE_KEYS = ("arch", "tensor_parallelism")

_OPS = ("eq", "ne", "gt", "ge", "lt", "le", "in")


@dataclass(frozen=True)
class ComboRow:
    id: str
    kind: str                              # engine_gate | validator
    when: "tuple[tuple[str, object], ...]"   # predicate key -> condition
    force: "tuple[tuple[str, object], ...]"  # dotted path -> forced value
    statement: str
    ticket: str        # the TD whose resolution deletes this row
    site: str          # enforcing code site — grep it before trusting

    def forces(self, path: str) -> bool:
        return any(p == path for p, _ in self.force)

    def forced(self, path: str) -> object:
        for p, v in self.force:
            if p == path:
                return v
        raise KeyError(f"combo row '{self.id}' does not force {path}")

    def when_text(self) -> str:
        """Human form for explain/refusal lines:
        ``arch == glm5_next AND tensor_parallelism > 1``."""
        sym = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=",
               "lt": "<", "le": "<="}
        parts = []
        for key, cond in self.when:
            if isinstance(cond, dict):
                for op, want in cond.items():
                    if op == "in":
                        parts.append(f"{key} in {list(want)}")
                    else:
                        parts.append(f"{key} {sym[op]} {want!r}"
                                     if isinstance(want, str)
                                     else f"{key} {sym[op]} {want}")
            elif isinstance(cond, list):
                parts.append(f"{key} in {cond}")
            else:
                parts.append(f"{key} == {cond}")
        return " AND ".join(parts)


def _check_condition(row_id: str, key: str, cond: object) -> None:
    if isinstance(cond, dict):
        if not cond:
            raise ValueError(f"model_constraints.json row '{row_id}': "
                             f"empty operator object for predicate {key!r}")
        for op in cond:
            if op not in _OPS:
                raise ValueError(
                    f"model_constraints.json row '{row_id}': unknown "
                    f"operator {op!r} for predicate {key!r} (known: "
                    f"{', '.join(_OPS)}) — refusing to load a registry "
                    "this build cannot evaluate (fail-closed)")
    elif not isinstance(cond, (str, int, float, bool, list)):
        raise ValueError(f"model_constraints.json row '{row_id}': predicate "
                         f"{key!r} condition must be a scalar, list, or "
                         f"operator object, got {type(cond).__name__}")


def _load(path: str = _JSON_PATH) -> "tuple[ComboRow, ...]":
    with open(path) as f:
        doc = json.load(f)
    rows_raw = doc.get("rows")
    if not isinstance(rows_raw, list):
        raise ValueError("model_constraints.json: top-level 'rows' list "
                         "missing")
    rows: list[ComboRow] = []
    seen: set[str] = set()
    for r in rows_raw:
        missing = [k for k in ("id", "kind", "when", "force", "statement")
                   if k not in r]
        if missing:
            raise ValueError(f"model_constraints.json row {r.get('id', '?')!r}"
                             f": missing field(s) {missing}")
        rid = str(r["id"])
        if rid in seen:
            raise ValueError(f"model_constraints.json: duplicate row id "
                             f"{rid!r}")
        seen.add(rid)
        if r["kind"] not in KINDS:
            raise ValueError(
                f"model_constraints.json row '{rid}': kind {r['kind']!r} "
                f"not in {KINDS} — a combination row is an engine-current "
                "fact, never measured/heuristic")
        when = r["when"]
        if not isinstance(when, dict) or not when:
            raise ValueError(f"model_constraints.json row '{rid}': 'when' "
                             "must be a non-empty object (an unconditional "
                             "rule belongs in engine_constraints.py)")
        for key, cond in when.items():
            if key not in PREDICATE_KEYS:
                raise ValueError(
                    f"model_constraints.json row '{rid}': unknown predicate "
                    f"key {key!r} (known: {', '.join(PREDICATE_KEYS)}) — a "
                    "new predicate is a new PREDICATE_KEYS entry plus its "
                    "ctx feed, and this build does not have it; refusing to "
                    "load rather than silently skipping (fail-closed)")
            _check_condition(rid, key, cond)
        force = r["force"]
        if not isinstance(force, dict) or not force:
            raise ValueError(f"model_constraints.json row '{rid}': 'force' "
                             "must be a non-empty object of dotted recipe "
                             "paths to values")
        rows.append(ComboRow(
            id=rid, kind=str(r["kind"]),
            when=tuple(when.items()), force=tuple(force.items()),
            statement=str(r["statement"]), ticket=str(r.get("ticket", "")),
            site=str(r.get("site", ""))))
    return tuple(rows)


COMBO_CONSTRAINTS: "tuple[ComboRow, ...]" = _load()


def _match_one(cond: object, actual: object) -> bool:
    if isinstance(cond, dict):
        for op, want in cond.items():
            if op == "eq" and not actual == want:
                return False
            if op == "ne" and not actual != want:
                return False
            if op == "gt" and not actual > want:      # type: ignore[operator]
                return False
            if op == "ge" and not actual >= want:     # type: ignore[operator]
                return False
            if op == "lt" and not actual < want:      # type: ignore[operator]
                return False
            if op == "le" and not actual <= want:     # type: ignore[operator]
                return False
            if op == "in" and actual not in want:     # type: ignore[operator]
                return False
        return True
    if isinstance(cond, list):
        return actual in cond
    return actual == cond


def matches(row: ComboRow, ctx: Mapping[str, object]) -> bool:
    """True when EVERY predicate in the row's combination holds in ctx.

    A ctx missing a key the row names REFUSES: silently treating an
    unevaluable predicate as false would un-fire a gate and emit a
    non-booting recipe (fail-closed, same rule as unknown keys at load)."""
    for key, cond in row.when:
        if key not in ctx:
            raise ValueError(
                f"combo row '{row.id}' needs predicate {key!r} but the "
                f"caller's ctx carries only {sorted(ctx)} — every consult "
                "site must supply all PREDICATE_KEYS")
        if not _match_one(cond, ctx[key]):
            return False
    return True


def forcing(path: str, ctx: Mapping[str, object]) -> "ComboRow | None":
    """The row that forces `path` under this combination, or None.

    Two firing rows forcing the same path to different values is a registry
    inconsistency and refuses loudly — never a tie silently broken."""
    hit: "ComboRow | None" = None
    for row in COMBO_CONSTRAINTS:
        if row.forces(path) and matches(row, ctx):
            if hit is not None and hit.forced(path) != row.forced(path):
                raise ValueError(
                    f"model_constraints.json: rows '{hit.id}' and "
                    f"'{row.id}' both fire for {path} with different "
                    "values — fix the registry, the solver refuses to pick")
            hit = hit or row
    return hit


def get(row_id: str) -> "ComboRow | None":
    """Row lookup; None when the row has been deleted (rule no longer holds)."""
    for row in COMBO_CONSTRAINTS:
        if row.id == row_id:
            return row
    return None


def active_for(ctx: Mapping[str, object]) -> "tuple[ComboRow, ...]":
    return tuple(r for r in COMBO_CONSTRAINTS if matches(r, ctx))

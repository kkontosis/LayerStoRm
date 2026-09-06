"""Explainability output (AUTOCONFIG §8).

Every derived field carries an Explanation: the value, HOW it was produced
(closed_form / heuristic / searched / template / carried), the constraint or
measurement that produced it, and citations. This extends the boot-log
pattern of vram_allocator.cpp (indexer pool / V4 side tiers) — a number no
one can trace back to a constraint is a number no one can debug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

KINDS = ("closed_form", "heuristic", "searched", "template", "carried", "measured",
         "pinned", "engine_gate")
# engine_gate: the value is FORCED by an engine-current fact (a combination
# row in model_constraints.json or an engine_constraints gate) — not derived,
# not measured, not a heuristic; the row cites the enforcing code site and
# the ticket whose resolution deletes it (TD-AUTOCONFIG-COMBO-CONSTRAINTS).


@dataclass
class Explanation:
    path: str            # dotted config path, e.g. "memory.kv_tiering.hot_buffer_slots"
    value: object
    kind: str            # one of KINDS
    because: str         # the constraint sentence, boot-log voice
    refs: tuple[str, ...] = ()    # spec/ticket/registry-row/code citations
    measurement: str = ""         # the measured input it rests on, when any

    def log_line(self) -> str:
        parts = [f"autoconfig: {self.path} = {self.value!r} [{self.kind}] — {self.because}"]
        if self.measurement:
            parts.append(f" (measured: {self.measurement})")
        if self.refs:
            parts.append(f" [{'; '.join(self.refs)}]")
        return "".join(parts)


class ExplanationSet:
    def __init__(self) -> None:
        self._by_path: dict[str, Explanation] = {}
        self.muted = False   # see Solver._quiet (AUTOCONFIG §5 E1)
        self._pinned: set[str] = set()   # immutable paths (kind=pinned)

    def add(self, path: str, value: object, kind: str, because: str,
            refs: tuple[str, ...] = (), measurement: str = "") -> None:
        assert kind in KINDS, kind
        if self.muted:
            # a SEARCH probe, not a derivation: the E1 ladder evaluates many
            # candidate fits and only the winning one's rationale belongs in
            # the record. Muting is explicit so a rejected candidate can
            # never leave a stray line behind (it used to be "the last call
            # wins", which only held while the search was a single loop).
            return
        if path in self._pinned:
            # the sheet never claims to have derived what it was told
            # (TD-AUTOCONFIG-PINNED-CONSTRAINTS (b)): once a path is pinned,
            # no derivation stage can re-narrate it as closed_form/searched/
            # measured/heuristic — same discipline shape as `muted`.
            return
        self._by_path[path] = Explanation(path, value, kind, because, refs, measurement)

    def pin(self, path: str, value: object, because: str,
            refs: tuple[str, ...] = ()) -> None:
        """Record a user pin (kind=pinned) and make the path IMMUTABLE."""
        self._by_path[path] = Explanation(path, value, "pinned", because, refs, "")
        self._pinned.add(path)

    def get(self, path: str) -> Explanation | None:
        return self._by_path.get(path)

    def all(self) -> list[Explanation]:
        return [self._by_path[k] for k in sorted(self._by_path)]

    def log_lines(self) -> list[str]:
        return [e.log_line() for e in self.all()]

    def render_markdown(self, title: str = "Autoconfig derivation") -> str:
        out = [f"# {title}", "",
               "| path | value | kind | because | measurement | refs |",
               "|---|---|---|---|---|---|"]
        for e in self.all():
            val = str(e.value)
            if len(val) > 60:
                val = val[:57] + "..."
            out.append("| `{}` | `{}` | {} | {} | {} | {} |".format(
                e.path, val, e.kind,
                e.because.replace("|", "\\|"),
                e.measurement.replace("|", "\\|"),
                "; ".join(e.refs).replace("|", "\\|")))
        out.append("")
        return "\n".join(out)


@dataclass
class Infeasible(Exception):
    """AUTOCONFIG §6 — refusal is a first-class output."""
    constraint_id: str
    binding: str          # which GPU / node / resource bound
    requested: str
    affordable: str
    suggestion: str

    def __str__(self) -> str:
        return (f"autoconfig REFUSES: {self.constraint_id} binds on {self.binding} — "
                f"requested {self.requested}, affordable {self.affordable}. "
                f"{self.suggestion}")

"""Champion comparison (AUTOCONFIG §9) — the acceptance instrument.

Diffs a derived recipe against a reference field-by-field and classifies
every divergence (TD-AUTOCONFIG-COMPARE-CLASSES: `unexplained` used to
conflate three different situations — a real solver gap, the REFERENCE
omitting a field the derived recipe states, and a family-template copy the
reference hand-deviates from; the first is a bug, the other two are not):

  match           — identical (not reported)
  derived-equal   — different bytes, same derived meaning (documented paths)
  solver-choice   — the solver diverges ON PURPOSE and its ExplanationSet
                    says why (any derivation row: measured/searched/
                    closed_form/heuristic/pinned)
  template-copy   — the derived value is a family-template copy
                    (kind=template/carried row, or a value matching
                    templates.template_view with no Explanation row); the
                    REFERENCE deviates from the template — review the
                    reference's hand-tuning, not the solver
  reference-omits — the reference recipe omits the field entirely (it
                    inherits a schema default the derived recipe states
                    explicitly); nothing diverges in meaning
  unexplained     — no row and no template covers the path: a solver bug
                    until proven a discovery (AUTOCONFIG §9)

Array LENGTH differences (e.g. 1 GPU vs 4) no longer collapse into one
opaque whole-array leaf: the walker emits a `<path>.length` divergence
(classified through the array path's own Explanation) plus per-element
rows; one-sided elements inherit the length row's class — a
wholly-justified shape change reads as its justification, and an
unjustified one reads as ONE unexplained length row.

Only `unexplained` gates acceptance (CLI exit 4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .explain import ExplanationSet

_MISSING = "<missing>"

# Paths whose divergence is definitionally benign (identity/meta, not sizing).
DERIVED_EQUAL_PATHS = {
    "hardware.system_ram_gb": (
        "informational field — pool sizing reads per-node meminfo at boot; "
        "the champion hand-rounds (512), the solver emits the detected total"),
    "autoconfig": "solver metadata section; absent from hand recipes",
}


@dataclass
class Divergence:
    path: str
    reference: object
    derived: object
    klass: str          # derived-equal | solver-choice | template-copy |
                        # reference-omits | unexplained
    why: str


def _walk(path: str, ref: object, der: object, out: list) -> None:
    if isinstance(ref, dict) and isinstance(der, dict):
        for k in sorted(set(ref) | set(der)):
            _walk(f"{path}.{k}" if path else k,
                  ref.get(k, _MISSING), der.get(k, _MISSING), out)
    elif isinstance(ref, list) and isinstance(der, list) and \
            all(isinstance(x, dict) for x in ref + der) and (ref or der):
        if len(ref) != len(der):
            # the shape decision is its own divergence, classified through
            # the ARRAY path's Explanation (TD-AUTOCONFIG-COMPARE-CLASSES:
            # a 1-GPU-vs-4 recipe used to read as one opaque unexplained
            # whole-array leaf)
            out.append((f"{path}.length", len(ref), len(der)))
        for i in range(max(len(ref), len(der))):
            r = ref[i] if i < len(ref) else _MISSING
            d = der[i] if i < len(der) else _MISSING
            if isinstance(r, dict) and isinstance(d, dict):
                _walk(f"{path}[{i}]", r, d, out)
            elif r != d:
                out.append((f"{path}[{i}]", r, d))
    elif ref != der:
        out.append((path, ref, der))


def _probe_explanation(ex: ExplanationSet, path: str):
    """Leaf-first, then progressively shorter prefixes with list indices
    stripped and the (hbm) qualifier probed (explanations may be keyed one
    level above the leaf the diff walker reaches).  A `<path>.length` row
    probes the array path itself."""
    e = ex.get(path)
    if e is not None:
        return e
    norm = re.sub(r"\[\d+\]", "", path)
    if norm.endswith(".length"):
        norm = norm[: -len(".length")]
    parts = norm.split(".")
    for cut in range(len(parts), 0, -1):
        probe = ".".join(parts[:cut])
        e = ex.get(probe) or ex.get(probe + "(hbm)")
        if e is not None:
            return e
    return None


def _template_value(template: dict | None, path: str):
    """Look the dotted path up in the template view (dict-only descent;
    list indices are stripped — template data carries no per-item arrays).
    Returns (found, value)."""
    if not template:
        return False, None
    node: object = template
    for part in re.sub(r"\[\d+\]", "", path).split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _classify(path: str, ref: object, der: object, ex: ExplanationSet,
              template: dict | None) -> Divergence:
    top = path.split(".")[0].split("[")[0]
    for p, why in DERIVED_EQUAL_PATHS.items():
        if path == p or top == p:
            return Divergence(path, ref, der, "derived-equal", why)
    e = _probe_explanation(ex, path)
    if ref == _MISSING and der != _MISSING:
        why = ("the reference omits this field — it inherits the schema "
               "default; the derived recipe states it explicitly "
               "(readable stand-alone, like the champion)")
        if e is not None:
            why += f" [{e.kind}] {e.because}"
        return Divergence(path, ref, der, "reference-omits", why)
    if e is not None:
        if e.kind in ("template", "carried"):
            return Divergence(
                path, ref, der, "template-copy",
                f"[{e.kind}] {e.because}"
                + (f" [{'; '.join(e.refs)}]" if e.refs else ""))
        return Divergence(
            path, ref, der, "solver-choice",
            f"[{e.kind}] {e.because}"
            + (f" [{'; '.join(e.refs)}]" if e.refs else ""))
    found, tval = _template_value(template, path)
    if found and tval == der:
        return Divergence(
            path, ref, der, "template-copy",
            "family-template copy with no Explanation row "
            "(templates.template_view) — the reference deviates from the "
            "template; review the reference's hand-tuning, not the solver")
    if der == _MISSING:
        return Divergence(
            path, ref, der, "unexplained",
            "the derived recipe OMITS a field the reference sets, and no "
            "explanation covers the omission — solver bug until proven a "
            "discovery")
    return Divergence(path, ref, der, "unexplained",
                      "no explanation and no template covers this path — "
                      "solver bug until proven a discovery")


def compare(reference: dict, derived: dict, ex: ExplanationSet,
            template: dict | None = None) -> list[Divergence]:
    """``template``: recipe-coordinate view of the family-template data the
    solver copies verbatim (templates.template_view(architecture)); None
    disables the template-copy fallback (rows with kind=template still
    classify)."""
    raw: list[tuple[str, object, object]] = []
    _walk("", reference, derived, raw)
    result: list[Divergence] = []
    by_path: dict[str, Divergence] = {}
    for path, ref, der in raw:
        d = _classify(path, ref, der, ex, template)
        # one-sided array elements are the length decision's consequence:
        # they inherit its class instead of exploding into per-element
        # unexplained rows (TD-AUTOCONFIG-COMPARE-CLASSES)
        m = re.fullmatch(r"(.*)\[\d+\]", path)
        if m and (ref == _MISSING or der == _MISSING):
            parent = by_path.get(f"{m.group(1)}.length")
            if parent is not None:
                d = Divergence(path, ref, der, parent.klass,
                               f"array shape change — see `{parent.path}`")
        result.append(d)
        by_path[path] = d
    return result


CLASS_ORDER = ("unexplained", "solver-choice", "template-copy",
               "reference-omits", "derived-equal")


def class_counts(divs: list[Divergence]) -> dict[str, int]:
    counts = {k: 0 for k in CLASS_ORDER}
    for d in divs:
        counts[d.klass] = counts.get(d.klass, 0) + 1
    return counts


def summary_line(divs: list[Divergence]) -> str:
    counts = class_counts(divs)
    per = ", ".join(f"{counts[k]} {k}" for k in CLASS_ORDER if counts[k])
    return f"{len(divs)} divergence(s): {per or 'none'}."


def render_report(reference_name: str, divs: list[Divergence]) -> str:
    out = [f"# Champion comparison vs {reference_name}", ""]
    if not divs:
        out.append("Field-for-field identical.")
        return "\n".join(out)
    out.append(summary_line(divs))
    out.append("Only `unexplained` gates acceptance (a solver bug until "
               "proven a discovery); `reference-omits` and `template-copy` "
               "describe the REFERENCE, not the solver.")
    out += ["", "| path | reference | derived | class | why |", "|---|---|---|---|---|"]
    for d in divs:
        out.append("| `{}` | `{}` | `{}` | **{}** | {} |".format(
            d.path, str(d.reference)[:48], str(d.derived)[:48], d.klass,
            d.why.replace("|", "\\|")))
    out.append("")
    return "\n".join(out)

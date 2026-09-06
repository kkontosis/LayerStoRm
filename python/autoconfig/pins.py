"""Pinned constraints (TD-AUTOCONFIG-PINNED-CONSTRAINTS) — a THIRD input
class beside the base config and the levers.

A base config is an IDENTITY source the derivation overwrites; a lever is an
ASK the E1 ladder may degrade; a pin is NEITHER: a hard constraint that
survives every E1 rung. It REMOVES an axis from the search lattice rather
than reordering it, and when no fit exists WITH the pin the solver refuses
naming the pin — never quietly relaxes it.

Only the solver's DECISION surface is pinnable (the whitelist below): a pin
on a field the sizing model never routes through would be either an inert
overlay (hand-edit the derived recipe instead) or silently inconsistent with
the derivation (sized for X, emitted Y) — the exact class of leak the ticket
exists to end. Unknown paths refuse at parse time, listing the menu.

CLI-only surface (`--pin`), like `prefer` and `accuracy` before it: schema
work is serialized in this campaign, and a pin FILE is a path argument, so
no `config/schema.json` change is owed. The emitted recipe's `autoconfig`
block never carries pins (pinned by test).

Design: spec/plans/AUTOCONFIG_PINNED_CONSTRAINTS.md; AUTOCONFIG §2.5.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Pin:
    path: str       # dotted recipe path, e.g. "compute.attention_backend"
    value: object
    source: str     # where the pin came from: file basename or "--pin arg"


def _enum(*allowed: str):
    def check(v: object) -> str:
        if not isinstance(v, str) or v not in allowed:
            return f"one of {'/'.join(allowed)}"
        return ""
    return check


def _pos_int(v: object) -> str:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return "a positive integer"
    return ""


def _bool(v: object) -> str:
    if not isinstance(v, bool):
        return "true or false"
    return ""


def _backend(v: object) -> str:
    # any string parses; family serveability is shape-dependent and checked
    # at solve time against the config_validator partition
    if not isinstance(v, str) or not v:
        return "an attention backend name"
    return ""


def _spec_method(v: object) -> str:
    if not isinstance(v, str):
        return "a speculation method name"
    if v not in ("none", "dspark"):
        # honest stub, refused at PARSE time: the solver has no sizing model
        # for the other methods (mtp, self_speculative, ...) and must never
        # price what it cannot size (design §3/§9)
        return ("'none' or 'dspark' — the solver has no sizing model for "
                f"method {v!r} and refuses to pin what it cannot size "
                "(use a hand recipe for other methods)")
    return ""


# The solver's decision surface — path -> (validator, what the pin fixes).
# Order matters: it is the deterministic order of refusal attribution.
PINNABLE: dict = {
    "compute.attention_backend":
        (_backend, "the numerics choice (accuracy ladder axis)"),
    "hardware.dcp_indexer_mode":
        (_enum("replicated", "local"), "the E1 indexer-mode axis (P5)"),
    "hardware.dcp_kv_mode":
        (_enum("replicated", "sharded"), "the P4 DCP KV mode choice"),
    "parallelism.tensor_parallelism":
        (_pos_int, "the E1 TP axis (P2)"),
    "serving.max_sequence_length":
        (_pos_int, "the cap axis' sequence half (K1; no halving rungs)"),
    "serving.max_concurrent_requests":
        (_pos_int, "the cap axis' concurrency half (K1; no shed rungs)"),
    "speculation.method":
        (_spec_method, "the draft axis (P6; none=off only, dspark=on only)"),
    "memory.kv_tiering.enabled":
        (_bool, "the K3 KV-tiering decision"),
}


def _menu() -> str:
    return ", ".join(PINNABLE)


class PinSet:
    """Ordered (PINNABLE order) immutable set of pins, keyed by path."""

    def __init__(self, pins: dict[str, Pin]):
        self._pins = {p: pins[p] for p in PINNABLE if p in pins}

    def get(self, path: str) -> Pin | None:
        return self._pins.get(path)

    def paths(self) -> tuple[str, ...]:
        return tuple(self._pins)

    def items(self):
        return self._pins.items()

    def __len__(self) -> int:
        return len(self._pins)

    def describe(self) -> str:
        return ", ".join(f"{p}={pin.value!r}" for p, pin in self._pins.items())


def _flatten(obj: dict, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten(v, path))
        else:
            out.append((path, v))
    return out


def _validate(path: str, value: object) -> None:
    entry = PINNABLE.get(path)
    if entry is None:
        raise ValueError(
            f"--pin {path}: not a pinnable field. Pinnable = the solver's "
            f"decision surface: {_menu()}. Anything else would be either an "
            "inert overlay (hand-edit the derived recipe, or use --config "
            "for identity inputs) or silently inconsistent with the sizing "
            "model (design: spec/plans/AUTOCONFIG_PINNED_CONSTRAINTS.md §3)")
    want = entry[0](value)
    if want:
        raise ValueError(f"--pin {path}={value!r}: expected {want}")


def parse_pin_args(args: list[str]) -> PinSet:
    """Parse the repeatable --pin arguments: each is a JSON file path or an
    inline `dotted.path=value` (RHS json-parsed, falling back to string).
    Duplicate paths with different values refuse — a self-contradictory pin
    set is the user's error, not a tie to break."""
    pins: dict[str, Pin] = {}

    def put(path: str, value: object, source: str) -> None:
        _validate(path, value)
        prev = pins.get(path)
        if prev is not None and prev.value != value:
            raise ValueError(
                f"--pin {path}: pinned twice with different values "
                f"({prev.value!r} from {prev.source}, {value!r} from "
                f"{source}) — a pin is a hard constraint, drop one")
        pins[path] = Pin(path, value, source)

    for arg in args:
        if os.path.exists(arg) or (arg.endswith(".json") and "=" not in arg):
            if not os.path.exists(arg):
                raise ValueError(f"--pin {arg}: file not found")
            with open(arg) as f:
                try:
                    obj = json.load(f)
                except ValueError as e:
                    raise ValueError(f"--pin {arg}: not valid JSON ({e})")
            if not isinstance(obj, dict):
                raise ValueError(f"--pin {arg}: expected a JSON object of "
                                 "config paths to values")
            for path, value in _flatten(obj):
                put(path, value, os.path.basename(arg))
        elif "=" in arg:
            path, _, raw = arg.partition("=")
            try:
                value: object = json.loads(raw)
            except ValueError:
                value = raw   # bare string, e.g. snapmla
            put(path.strip(), value, f"--pin {arg}")
        else:
            raise ValueError(
                f"--pin {arg}: neither an existing JSON file nor an inline "
                f"dotted.path=value pin (pinnable paths: {_menu()})")
    return PinSet(pins)

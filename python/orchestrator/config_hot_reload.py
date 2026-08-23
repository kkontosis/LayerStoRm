"""ConfigHotReload — mutable config object for live parameter updates.

Orchestrator modules read attributes directly each cycle. The orchestrator
loop or an external control path mutates values via update(). Changes take
effect on the next read (no locking needed — single-threaded orchestrator).

For Python→C++ sync, pending_daemon_updates() returns dirty field entries
that the orchestrator loop sends via CMD_CONFIG_UPDATE.
"""

from __future__ import annotations

import struct
from typing import Any

from orchestrator._gen_changeable_fields import CHANGEABLE_FIELDS


class ConfigHotReload:
    def __init__(self) -> None:
        self._fields: dict[str, _FieldMeta] = {}
        self._by_id: dict[int, str] = {}
        self._dirty: set[str] = set()

        for field_id, dotted_path, py_type, default_val in CHANGEABLE_FIELDS:
            attr = dotted_path.replace(".", "_").replace("-", "_")
            meta = _FieldMeta(
                field_id=field_id,
                dotted_path=dotted_path,
                attr=attr,
                py_type=py_type,
                value=default_val,
            )
            self._fields[dotted_path] = meta
            self._by_id[field_id] = dotted_path
            object.__setattr__(self, attr, default_val)

    def update(self, dotted_path: str, value: Any) -> None:
        meta = self._fields.get(dotted_path)
        if meta is None:
            raise KeyError(
                f"'{dotted_path}' is not a changeable field"
            )
        meta.value = meta.py_type(value)
        object.__setattr__(self, meta.attr, meta.value)
        self._dirty.add(dotted_path)

    def get(self, dotted_path: str) -> Any:
        meta = self._fields.get(dotted_path)
        if meta is None:
            raise KeyError(f"'{dotted_path}' is not a changeable field")
        return meta.value

    def pending_daemon_updates(self) -> list[tuple[int, int, int]]:
        """Return (field_id, value_type, raw_value) tuples for dirty fields."""
        result: list[tuple[int, int, int]] = []
        for path in self._dirty:
            meta = self._fields[path]
            vtype, raw = _encode_value(meta.py_type, meta.value)
            result.append((meta.field_id, vtype, raw))
        return result

    def clear_pending(self) -> None:
        self._dirty.clear()

    @property
    def has_pending(self) -> bool:
        return len(self._dirty) > 0

    @property
    def changeable_paths(self) -> list[str]:
        return list(self._fields.keys())


class _FieldMeta:
    __slots__ = ("field_id", "dotted_path", "attr", "py_type", "value")

    def __init__(
        self,
        field_id: int,
        dotted_path: str,
        attr: str,
        py_type: type,
        value: Any,
    ) -> None:
        self.field_id = field_id
        self.dotted_path = dotted_path
        self.attr = attr
        self.py_type = py_type
        self.value = value


def _encode_value(py_type: type, value: Any) -> tuple[int, int]:
    """Encode a Python value into (value_type, raw_uint32)."""
    if py_type is bool:
        return (0, 1 if value else 0)
    if py_type is int:
        return (1, value & 0xFFFFFFFF)
    if py_type is float:
        raw_bytes = struct.pack("<f", float(value))
        raw_u32 = struct.unpack("<I", raw_bytes)[0]
        return (2, raw_u32)
    raise TypeError(f"Unsupported type: {py_type}")

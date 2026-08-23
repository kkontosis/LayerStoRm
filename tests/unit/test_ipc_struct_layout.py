"""Cross-language IPC envelope struct layout consistency tests.

Validates that Python ctypes struct definitions match C++ struct layouts
(field offsets, field sizes, total sizeof) for the 8 IPC envelope structs:
IpcHeader, RingHeader, Command, Completion, GpuSnapshot, RequestAcceptance,
StateSnapshot, EngineInfo.

A mismatch here means IPC data will be read as garbage — silent corruption
with no crash.

The C++ side exposes layout metadata via layerstorm_engine.ipc_struct_layout(),
computed at compile time with offsetof(). This test compares against the
Python ctypes definitions in orchestrator.shm_protocol.

See also: test_ipc_payload_layout.py (command/completion payloads),
          test_ipc_sideband_layout.py (sideband structs, IPC-8b).
"""

import ctypes

import pytest

import layerstorm_engine
from orchestrator.shm_protocol import (
    Command,
    Completion,
    EngineInfo,
    GpuSnapshot,
    IpcHeader,
    RequestAcceptance,
    RingHeader,
    StateSnapshot,
)

# Map C++ struct names to Python ctypes classes
_PY_STRUCTS = {
    "IpcHeader":         IpcHeader,
    "RingHeader":        RingHeader,
    "Command":           Command,
    "Completion":        Completion,
    "GpuSnapshot":       GpuSnapshot,
    "RequestAcceptance": RequestAcceptance,
    "StateSnapshot":     StateSnapshot,
    "EngineInfo":        EngineInfo,
}


def _py_field_map(cls):
    """Build {field_name: (offset, size)} from a ctypes Structure."""
    result = {}
    for name, ctype in cls._fields_:
        if name.startswith("_pad"):
            continue  # Skip padding fields — names may differ across languages
        field_descriptor = getattr(cls, name)
        result[name] = (field_descriptor.offset, ctypes.sizeof(ctype))
    return result


@pytest.fixture(scope="module")
def cpp_layout():
    return layerstorm_engine.ipc_struct_layout()


class TestIpcStructLayout:
    """Validate every IPC envelope struct's sizeof and field offsets match C++."""

    def test_all_structs_present(self, cpp_layout):
        """C++ exports layout for every struct Python mirrors."""
        for name in _PY_STRUCTS:
            assert name in cpp_layout, f"C++ missing layout for {name}"

    @pytest.mark.parametrize("struct_name", list(_PY_STRUCTS.keys()))
    def test_sizeof_matches(self, cpp_layout, struct_name):
        cpp = cpp_layout[struct_name]
        py_cls = _PY_STRUCTS[struct_name]
        assert ctypes.sizeof(py_cls) == cpp["sizeof"], (
            f"{struct_name}: sizeof mismatch — "
            f"C++={cpp['sizeof']}, Python={ctypes.sizeof(py_cls)}"
        )

    @pytest.mark.parametrize("struct_name", list(_PY_STRUCTS.keys()))
    def test_field_offsets_match(self, cpp_layout, struct_name):
        cpp_fields = cpp_layout[struct_name]["fields"]
        py_fields = _py_field_map(_PY_STRUCTS[struct_name])

        # Check every C++ field has a matching Python field
        for fname, (cpp_off, cpp_sz) in cpp_fields.items():
            if fname.startswith("_pad"):
                continue
            assert fname in py_fields, (
                f"{struct_name}.{fname}: present in C++ (offset={cpp_off}) "
                f"but missing in Python"
            )
            py_off, py_sz = py_fields[fname]
            assert py_off == cpp_off, (
                f"{struct_name}.{fname}: offset mismatch — "
                f"C++={cpp_off}, Python={py_off}"
            )
            assert py_sz == cpp_sz, (
                f"{struct_name}.{fname}: size mismatch — "
                f"C++={cpp_sz}, Python={py_sz}"
            )

    @pytest.mark.parametrize("struct_name", list(_PY_STRUCTS.keys()))
    def test_no_extra_python_fields(self, cpp_layout, struct_name):
        """Python doesn't have non-padding fields absent from C++."""
        cpp_fields = cpp_layout[struct_name]["fields"]
        py_fields = _py_field_map(_PY_STRUCTS[struct_name])

        for fname in py_fields:
            assert fname in cpp_fields, (
                f"{struct_name}.{fname}: present in Python but missing in C++ — "
                f"likely a Python-only field or renamed field"
            )

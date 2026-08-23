"""Cross-language IPC sideband struct layout consistency tests (IPC-8b).

Validates that Python ctypes sideband entry struct definitions match C++
struct layouts (field offsets, field sizes, total sizeof) and that
sub-region offsets/sizes/max-entries agree across languages.

See also: test_ipc_struct_layout.py (envelope structs),
          test_ipc_payload_layout.py (command/completion payloads).
"""

import ctypes

import pytest

import layerstorm_engine
from orchestrator.shm_protocol import (
    BatchDescriptorEntry,
    ExpertEvictionEntry,
    ExpertPrefetchEntry,
    NvmeReadEntry,
    ReserveBatchEntry,
    TransferBatchEntry,
    SIDEBAND_BATCH_DESCRIPTOR_OFF,
    SIDEBAND_BATCH_DESCRIPTOR_SIZE,
    SIDEBAND_EXPERT_EVICTION_OFF,
    SIDEBAND_EXPERT_EVICTION_SIZE,
    SIDEBAND_EXPERT_PREFETCH_OFF,
    SIDEBAND_EXPERT_PREFETCH_SIZE,
    SIDEBAND_NVME_READ_OFF,
    SIDEBAND_NVME_READ_SIZE,
    SIDEBAND_RESERVE_BATCH_OFF,
    SIDEBAND_RESERVE_BATCH_SIZE,
    SIDEBAND_TOKEN_IDS_OFF,
    SIDEBAND_TOKEN_IDS_SIZE,
    SIDEBAND_TOTAL_SIZE,
    SIDEBAND_TRANSFER_BATCH_OFF,
    SIDEBAND_TRANSFER_BATCH_SIZE,
    MAX_BATCH_DESCRIPTORS,
    MAX_EXPERT_EVICTION,
    MAX_EXPERT_PREFETCH,
    MAX_NVME_READ_BATCH,
    MAX_RESERVE_BATCH,
    MAX_SIDEBAND_TOKEN_IDS,
    MAX_TRANSFER_BATCH,
)

# Map C++ struct names to Python ctypes classes.
_PY_SIDEBAND_STRUCTS = {
    "BatchDescriptorEntry": BatchDescriptorEntry,
    "ExpertPrefetchEntry":  ExpertPrefetchEntry,
    "ExpertEvictionEntry":  ExpertEvictionEntry,
    "TransferBatchEntry":   TransferBatchEntry,
    "ReserveBatchEntry":    ReserveBatchEntry,
    "NvmeReadEntry":        NvmeReadEntry,
}

# Python sub-region constants: {name: (offset, size, max_entries)}.
_PY_REGIONS = {
    "batch_descriptor": (SIDEBAND_BATCH_DESCRIPTOR_OFF,
                         SIDEBAND_BATCH_DESCRIPTOR_SIZE,
                         MAX_BATCH_DESCRIPTORS),
    "expert_prefetch":  (SIDEBAND_EXPERT_PREFETCH_OFF,
                         SIDEBAND_EXPERT_PREFETCH_SIZE,
                         MAX_EXPERT_PREFETCH),
    "expert_eviction":  (SIDEBAND_EXPERT_EVICTION_OFF,
                         SIDEBAND_EXPERT_EVICTION_SIZE,
                         MAX_EXPERT_EVICTION),
    "transfer_batch":   (SIDEBAND_TRANSFER_BATCH_OFF,
                         SIDEBAND_TRANSFER_BATCH_SIZE,
                         MAX_TRANSFER_BATCH),
    "reserve_batch":    (SIDEBAND_RESERVE_BATCH_OFF,
                         SIDEBAND_RESERVE_BATCH_SIZE,
                         MAX_RESERVE_BATCH),
    "nvme_read":        (SIDEBAND_NVME_READ_OFF,
                         SIDEBAND_NVME_READ_SIZE,
                         MAX_NVME_READ_BATCH),
    "token_ids":        (SIDEBAND_TOKEN_IDS_OFF,
                         SIDEBAND_TOKEN_IDS_SIZE,
                         MAX_SIDEBAND_TOKEN_IDS),
}


def _py_field_map(cls):
    """Build {field_name: (offset, size)} from a ctypes Structure."""
    result = {}
    for name, ctype in cls._fields_:
        if name.startswith("_pad"):
            continue
        field_descriptor = getattr(cls, name)
        result[name] = (field_descriptor.offset, ctypes.sizeof(ctype))
    return result


@pytest.fixture(scope="module")
def cpp_layout():
    return layerstorm_engine.ipc_sideband_layout()


# ── Entry struct layout validation ───────────────────────────────────────────

class TestSidebandStructLayout:
    """Validate sideband entry struct layouts match C++."""

    @pytest.mark.parametrize("name", list(_PY_SIDEBAND_STRUCTS.keys()))
    def test_sizeof_matches(self, cpp_layout, name):
        cpp = cpp_layout[name]
        py_cls = _PY_SIDEBAND_STRUCTS[name]
        assert ctypes.sizeof(py_cls) == cpp["sizeof"], (
            f"{name}: sizeof mismatch -- C++={cpp['sizeof']}, "
            f"Python={ctypes.sizeof(py_cls)}"
        )

    @pytest.mark.parametrize("name", list(_PY_SIDEBAND_STRUCTS.keys()))
    def test_field_offsets_match(self, cpp_layout, name):
        cpp_fields = cpp_layout[name]["fields"]
        py_fields = _py_field_map(_PY_SIDEBAND_STRUCTS[name])

        for fname, (cpp_off, cpp_sz) in cpp_fields.items():
            if fname.startswith("_pad"):
                continue
            assert fname in py_fields, (
                f"{name}.{fname}: present in C++ (offset={cpp_off}) "
                f"but missing in Python"
            )
            py_off, py_sz = py_fields[fname]
            assert py_off == cpp_off, (
                f"{name}.{fname}: offset mismatch -- C++={cpp_off}, "
                f"Python={py_off}"
            )
            assert py_sz == cpp_sz, (
                f"{name}.{fname}: size mismatch -- C++={cpp_sz}, "
                f"Python={py_sz}"
            )

    @pytest.mark.parametrize("name", list(_PY_SIDEBAND_STRUCTS.keys()))
    def test_no_extra_python_fields(self, cpp_layout, name):
        """Python should not define fields absent from C++."""
        cpp_fields = set(cpp_layout[name]["fields"].keys())
        py_fields = set(_py_field_map(_PY_SIDEBAND_STRUCTS[name]).keys())
        extra = py_fields - cpp_fields
        assert not extra, (
            f"{name}: Python has extra fields not in C++: {extra}"
        )


# ── Sub-region layout validation ─────────────────────────────────────────────

class TestSidebandRegionLayout:
    """Validate sideband sub-region offsets, sizes, and max entries match C++."""

    @pytest.mark.parametrize("name", list(_PY_REGIONS.keys()))
    def test_region_offset(self, cpp_layout, name):
        cpp_off, _, _ = cpp_layout["_regions"][name]
        py_off, _, _ = _PY_REGIONS[name]
        assert py_off == cpp_off, (
            f"Region '{name}': offset mismatch -- C++={cpp_off}, Python={py_off}"
        )

    @pytest.mark.parametrize("name", list(_PY_REGIONS.keys()))
    def test_region_size(self, cpp_layout, name):
        _, cpp_sz, _ = cpp_layout["_regions"][name]
        _, py_sz, _ = _PY_REGIONS[name]
        assert py_sz == cpp_sz, (
            f"Region '{name}': size mismatch -- C++={cpp_sz}, Python={py_sz}"
        )

    @pytest.mark.parametrize("name", list(_PY_REGIONS.keys()))
    def test_region_max_entries(self, cpp_layout, name):
        _, _, cpp_max = cpp_layout["_regions"][name]
        _, _, py_max = _PY_REGIONS[name]
        assert py_max == cpp_max, (
            f"Region '{name}': max_entries mismatch -- C++={cpp_max}, "
            f"Python={py_max}"
        )

    def test_all_regions_present(self, cpp_layout):
        """All Python regions have C++ counterparts and vice versa."""
        cpp_names = set(cpp_layout["_regions"].keys())
        py_names = set(_PY_REGIONS.keys())
        assert cpp_names == py_names, (
            f"Region name mismatch: C++ has {cpp_names - py_names} extra, "
            f"Python has {py_names - cpp_names} extra"
        )


# ── Total size validation ────────────────────────────────────────────────────

class TestSidebandTotalSize:
    def test_total_size_matches(self, cpp_layout):
        assert SIDEBAND_TOTAL_SIZE == cpp_layout["_total_size"], (
            f"Sideband total size mismatch -- C++={cpp_layout['_total_size']}, "
            f"Python={SIDEBAND_TOTAL_SIZE}"
        )

    def test_total_size_is_96272(self, cpp_layout):
        # 8 command regions (→ 26624, end of spec-checkpoint) + F-4 routing-export
        # slot (65552) + F-7 seam-checkpoint (4096) = 96272.
        assert cpp_layout["_total_size"] == 96272
        assert SIDEBAND_TOTAL_SIZE == 96272

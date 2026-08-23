"""Cross-language IPC command/completion payload layout consistency tests.

Validates that Python ctypes payload struct definitions match C++ struct
layouts (field offsets, field sizes, total sizeof) for:
  - 21 Command payload types (TransferPayload..NumaMigratePayload)
  - 3 Completion payload types (TransferCompletion..ErrorCompletion)

C++ payloads are anonymous structs inside a union at offset 16 within
Command/Completion. The C++ exporter reports offsets relative to the
payload start (union base), matching the Python standalone struct layout.

See also: test_ipc_struct_layout.py (envelope structs),
          test_ipc_sideband_layout.py (sideband structs, IPC-8b).
"""

import ctypes

import pytest

import layerstorm_engine
from orchestrator import shm_protocol


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
    return layerstorm_engine.ipc_payload_layout()


# ── Command payloads ────────────────────────────────────────────────────────

_PY_CMD_PAYLOADS = {
    "TransferPayload":        shm_protocol.TransferPayload,
    "CacheReservePayload":    shm_protocol.CacheReservePayload,
    "CacheOpPayload":         shm_protocol.CacheOpPayload,
    "AttentionPayload":       shm_protocol.AttentionPayload,
    "GatingPayload":          shm_protocol.GatingPayload,
    "ExpertFfnPayload":       shm_protocol.ExpertFfnPayload,
    "EmbeddingLookupPayload": shm_protocol.EmbeddingLookupPayload,
    "OutputHeadPayload":      shm_protocol.OutputHeadPayload,
    "RmsnormPayload":         shm_protocol.RmsnormPayload,
    "SwigluPayload":          shm_protocol.SwigluPayload,
    "MoePermutePayload":      shm_protocol.MoePermutePayload,
    "MoeUnpermutePayload":    shm_protocol.MoeUnpermutePayload,
    "DcpCorrectionPayload":   shm_protocol.DcpCorrectionPayload,
    "NcclAllreducePayload":   shm_protocol.NcclAllreducePayload,
    "DynamicFp8QuantPayload": shm_protocol.DynamicFp8QuantPayload,
    "SampleTokensPayload":    shm_protocol.SampleTokensPayload,
    "PrescopePayload":        shm_protocol.PrescopePayload,
    "ProbePayload":           shm_protocol.ProbePayload,
    "GraphPayload":           shm_protocol.GraphPayload,
    "EventPayload":           shm_protocol.EventPayload,
    "AffinityHintsPayload":   shm_protocol.AffinityHintsPayload,
    "NumaMigratePayload":     shm_protocol.NumaMigratePayload,
    "SeqCreatePayload":       shm_protocol.SeqCreatePayload,
    "SeqFreePayload":         shm_protocol.SeqFreePayload,
    "SeqForkPayload":         shm_protocol.SeqForkPayload,
    "NvmeReadPayload":        shm_protocol.NvmeReadPayload,
    "NvmeWritePayload":       shm_protocol.NvmeWritePayload,
    "NvmeEvictHostPayload":   shm_protocol.NvmeEvictHostPayload,
    "CancelTransferPayload":  shm_protocol.CancelTransferPayload,
    "RunAttentionPayload":       shm_protocol.RunAttentionPayload,
    "RunMoePayload":             shm_protocol.RunMoePayload,
    "PrefetchBatchPayload":      shm_protocol.PrefetchBatchPayload,
    "EvictBatchPayload":         shm_protocol.EvictBatchPayload,
    "NvmeBatchReadPayload":      shm_protocol.NvmeBatchReadPayload,
    "PrefetchExpertPayload":     shm_protocol.PrefetchExpertPayload,
    "EvictToHostPayload":        shm_protocol.EvictToHostPayload,
    "StageExpertPayload":        shm_protocol.StageExpertPayload,
    "RunPrefetchProbePayload":   shm_protocol.RunPrefetchProbePayload,
    "RunAdapterForwardPayload":  shm_protocol.RunAdapterForwardPayload,
    "SlowEvictToHostPayload":    shm_protocol.SlowEvictToHostPayload,
    "MtpStepPayload":            shm_protocol.MtpStepPayload,
    "SelfSpecForwardPayload":    shm_protocol.SelfSpecForwardPayload,
    "ConfigUpdatePayload":       shm_protocol.ConfigUpdatePayload,
}


class TestCommandPayloadLayout:
    """Validate Command payload struct layouts match C++."""

    @pytest.mark.parametrize("name", list(_PY_CMD_PAYLOADS.keys()))
    def test_sizeof_matches(self, cpp_layout, name):
        cpp = cpp_layout[name]
        py_cls = _PY_CMD_PAYLOADS[name]
        assert ctypes.sizeof(py_cls) == cpp["sizeof"], (
            f"{name}: sizeof mismatch — C++={cpp['sizeof']}, Python={ctypes.sizeof(py_cls)}"
        )

    @pytest.mark.parametrize("name", list(_PY_CMD_PAYLOADS.keys()))
    def test_field_offsets_match(self, cpp_layout, name):
        cpp_fields = cpp_layout[name]["fields"]
        py_fields = _py_field_map(_PY_CMD_PAYLOADS[name])

        for fname, (cpp_off, cpp_sz) in cpp_fields.items():
            if fname.startswith("_pad"):
                continue
            assert fname in py_fields, (
                f"{name}.{fname}: present in C++ (offset={cpp_off}) but missing in Python"
            )
            py_off, py_sz = py_fields[fname]
            assert py_off == cpp_off, (
                f"{name}.{fname}: offset mismatch — C++={cpp_off}, Python={py_off}"
            )
            assert py_sz == cpp_sz, (
                f"{name}.{fname}: size mismatch — C++={cpp_sz}, Python={py_sz}"
            )


# ── Completion payloads ─────────────────────────────────────────────────────

_PY_CMP_PAYLOADS = {
    "TransferCompletionPayload": shm_protocol.TransferCompletionPayload,
    "ComputeCompletionPayload":  shm_protocol.ComputeCompletionPayload,
    "ErrorCompletionPayload":    shm_protocol.ErrorCompletionPayload,
    "SeqOpCompletionPayload":    shm_protocol.SeqOpCompletionPayload,
    "NvmeCompletionPayload":     shm_protocol.NvmeCompletionPayload,
    "CancelCompletionPayload":   shm_protocol.CancelCompletionPayload,
    "ElmExpertCompletionPayload":   shm_protocol.ElmExpertCompletionPayload,
    "ElmProgressCompletionPayload": shm_protocol.ElmProgressCompletionPayload,
    "CheckpointCompletionPayload":  shm_protocol.CheckpointCompletionPayload,
}


class TestCompletionPayloadLayout:
    """Validate Completion payload struct layouts match C++."""

    @pytest.mark.parametrize("name", list(_PY_CMP_PAYLOADS.keys()))
    def test_sizeof_matches(self, cpp_layout, name):
        cpp = cpp_layout[name]
        py_cls = _PY_CMP_PAYLOADS[name]
        assert ctypes.sizeof(py_cls) == cpp["sizeof"], (
            f"{name}: sizeof mismatch — C++={cpp['sizeof']}, Python={ctypes.sizeof(py_cls)}"
        )

    @pytest.mark.parametrize("name", list(_PY_CMP_PAYLOADS.keys()))
    def test_field_offsets_match(self, cpp_layout, name):
        cpp_fields = cpp_layout[name]["fields"]
        py_fields = _py_field_map(_PY_CMP_PAYLOADS[name])

        for fname, (cpp_off, cpp_sz) in cpp_fields.items():
            if fname.startswith("_pad"):
                continue
            assert fname in py_fields, (
                f"{name}.{fname}: present in C++ (offset={cpp_off}) but missing in Python"
            )
            py_off, py_sz = py_fields[fname]
            assert py_off == cpp_off, (
                f"{name}.{fname}: offset mismatch — C++={cpp_off}, Python={py_off}"
            )
            assert py_sz == cpp_sz, (
                f"{name}.{fname}: size mismatch — C++={cpp_sz}, Python={py_sz}"
            )

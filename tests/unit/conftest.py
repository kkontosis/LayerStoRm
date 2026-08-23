import ctypes
import pathlib
import sys

import pytest

# Add python/ directory to path for orchestrator imports.
_project_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "python"))

# Add build directory for compiled pybind11 modules (layerstorm_engine, etc.).
_build_dir = _project_root / "build" / "python"
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

from orchestrator.shm_protocol import StateSnapshot
from orchestrator.types import EngineMetadata, GpuConfig


@pytest.fixture
def empty_snapshot():
    """Zero-initialized StateSnapshot with its backing buffer."""
    buf = bytearray(ctypes.sizeof(StateSnapshot))
    return StateSnapshot.from_buffer(buf), buf


@pytest.fixture
def mock_engine_metadata():
    """Realistic EngineMetadata for a 2x5090 + 2x5080 config."""
    return EngineMetadata(
        num_gpus=4,
        num_moe_layers=58,
        num_experts=256,
        num_layers=61,
        expert_bytes=2_359_296,
        kv_bytes_per_page=644 * 64,
        gpus=(
            GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                      vram_bytes=32 * 1024**3),
            GpuConfig(position=1, gpu_type="rtx5090", is_tp=True,
                      vram_bytes=32 * 1024**3),
            GpuConfig(position=2, gpu_type="rtx5080", is_tp=False,
                      vram_bytes=16 * 1024**3),
            GpuConfig(position=3, gpu_type="rtx5080", is_tp=False,
                      vram_bytes=16 * 1024**3),
        ),
    )

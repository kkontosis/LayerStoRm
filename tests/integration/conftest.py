"""Python-path setup for integration pytest modules (mirrors tests/unit)."""

import pathlib
import sys

_project_root = pathlib.Path(__file__).resolve().parent.parent.parent

# Add python/ directory to path for orchestrator imports.
sys.path.insert(0, str(_project_root / "python"))

# Add build directory for compiled pybind11 modules (layerstorm_engine).
_build_dir = _project_root / "build" / "python"
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

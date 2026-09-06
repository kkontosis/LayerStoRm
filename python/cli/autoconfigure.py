#!/usr/bin/env python3
"""CLI autoconfigure — point it at a model, get a config and a serve command.

    python python/cli/autoconfigure.py --model /path/to/model

Probes the model (GGUF metadata / HF config), calibrates this box, derives a
full serving recipe, trains the loader constants on a real 100-token decode,
and prints the serve command. Levers are optional flags
(--vram-expert-ratio, --active-context); measured artifacts are reused unless
--reset/--retrain ask otherwise.

Advanced/legacy: --config <recipe> derives from an existing recipe's identity
(what serve.py --autoconfig calls).

User guide: spec/AUTO_RUN.md. Design: docs/AUTOCONFIG_MODEL.md.
"""

import os
import sys

# NOTE: this file must NOT be named autoconfig.py — a module named
# "autoconfig" in the script dir shadows the python/autoconfig package
# for any process whose sys.path has python/cli/ first (serve.py under
# serve_guided.sh, whose exported PYTHONPATH defeats the dedupe-guarded
# re-insertion). Found by the first hardware boot leg, 2026-08-30.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autoconfig.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

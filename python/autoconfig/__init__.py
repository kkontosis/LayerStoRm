"""Autoconfig — hardware-fit config derivation (TD-AUTOCONFIG-HARDWARE-FIT).

Opt-in, default OFF. Detects the hardware (CPU-only, never touches the GPUs),
derives a full serving recipe from the four human-facing levers
(vram_expert_ratio, total_active_context_tokens, prefer, accuracy), persists
it as a normal schema-valid recipe file, and explains every derived number.
Beside the levers (asks the fit may degrade) and the base config (an
identity source), `--pin` supplies HARD constraints the solver treats as
fixed and solves around — pins.py; AUTOCONFIG §2.5.

Spec: docs/AUTOCONFIG_MODEL.md. Code comments cite it as AUTOCONFIG §n.
"""

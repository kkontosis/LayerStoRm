"""LayerStoRm C++/Python ring bridge — orchestrator-free engine driver.

A minimal, self-contained layer over the engine's IPC surfaces (SPSC
command/completion rings + sideband region + seqlock snapshot) exposed by
the ``layerstorm_engine`` pybind module.  It deliberately does NOT import
the ``orchestrator`` event-loop machinery (``orchestrator_loop`` /
``engine_glue``): the only shared dependency is the pure protocol mirror
(``orchestrator.shm_protocol``), re-exported through ``bridge.protocol`` as
the single swap point should the protocol module move when the current
orchestrator is deprecated.

Modules:
  protocol    — ctypes struct/constant re-exports (the IPC contract)
  ring_bridge — EngineBridge: rings, sideband, per-step command chains
  spec_decode — plain + DSpark-speculative decode loops over EngineBridge
  _fastbridge — optional Cython hot path (ring ops, completion poll,
                sideband bulk writes); pure-Python fallback is automatic
"""

from bridge.ring_bridge import BridgeError, EngineBridge, GpuLru  # noqa: F401
from bridge.spec_decode import (  # noqa: F401
    PlainAgg,
    SpecStats,
    run_plain_loop,
    run_speculative_loop,
)

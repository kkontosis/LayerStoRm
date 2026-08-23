"""IPC protocol re-exports for the ring bridge.

The bridge's ONLY dependency on the ``orchestrator`` package is the pure
protocol mirror ``orchestrator.shm_protocol`` (ctypes structs + layout
constants matching src/daemon/ipc_protocol.h byte-for-byte).  Everything
the bridge needs is re-exported here so that when the current orchestrator
package is deprecated, moving the protocol module means editing exactly
one import line.

Layout ground truth remains the C++ header; ``ring_bridge`` cross-checks
the offsets it packs against these ctypes structs at import time, and the
pybind ``ipc_struct_layout()/ipc_payload_layout()`` introspection remains
the authoritative runtime check.
"""

from orchestrator.shm_protocol import (  # noqa: F401
    # ── envelope structs ──
    Command,
    Completion,
    RingHeader,
    BatchDescriptorEntry,
    ExpertPrefetchEntry,
    ExpertEvictionEntry,
    RoutingExportHeader,
    # ── slot / capacity constants ──
    CMD_SLOT_BYTES,
    CMP_SLOT_BYTES,
    MAX_BATCH_DESCRIPTORS,
    MAX_EXPERT_PREFETCH,
    MAX_EXPERT_EVICTION,
    MAX_SIDEBAND_TOKEN_IDS,
    # ── command types the bridge drives ──
    CMD_EMBEDDING_LOOKUP,
    CMD_OUTPUT_HEAD,
    CMD_SAMPLE_TOKENS,
    CMD_SEQ_CREATE,
    CMD_SEQ_FREE,
    CMD_SEQ_FORK,
    D_B_CMD_RUN_ATTENTION,
    D_B_CMD_RUN_MOE,
    D_B_CMD_PREFETCH_BATCH,
    D_CMD_RUN_DSPARK_STEP,
    E_CMD_FETCH_AND_RUN_MOE,
    E_CMD_FETCH_AND_RUN_MOE_BIG,
    E_CMD_REEF_ROUTE,
    E_CMD_FAR_FORWARD_LAYER,
    # ── completion types ──
    CMP_COMPUTE_DONE,
    CMP_CHECKPOINT,
    CMP_ERROR,
    CMP_SEQ_OP_DONE,
    # ── sideband sub-region offsets (relative to sideband base) ──
    SIDEBAND_BATCH_DESCRIPTOR_OFF,
    SIDEBAND_EXPERT_PREFETCH_OFF,
    SIDEBAND_EXPERT_EVICTION_OFF,
    SIDEBAND_TOKEN_IDS_OFF,
    SIDEBAND_SPEC_CHECKPOINT_OFF,
    SIDEBAND_ROUTING_EXPORT_OFF,
    SIDEBAND_ROUTING_EXPORT_INDICES_OFF,
)

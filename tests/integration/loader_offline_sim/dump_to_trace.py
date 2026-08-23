#!/usr/bin/env python3
# Convert an LS_LOADER_SHADOW_DUMP JSONL (one record per MoE solve) into the
# offline-sim fixtures routed_trace.csv (5-col) + oracle_baseline.csv (9-col).
# token is derived from solve order: layer_idx runs 3..60 per token (MoE starts
# at first_k_dense_replace=3); a layer_idx <= prev marks a new token.
import json, sys

src = sys.argv[1]
trace_out = sys.argv[2]
oracle_out = sys.argv[3]

records = []
with open(src) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

trace_rows = []   # token,layer,slot,expert_idx,bank
oracle_rows = []  # token,layer,slot,expert_idx,bank,oj,j,cached0,cached1
token = 0
prev_layer = None
for rec in records:
    layer = rec["layer_idx"]
    if prev_layer is not None and layer <= prev_layer:
        token += 1
    prev_layer = layer
    for slot, e in enumerate(rec["experts"]):
        ei = e["expert_idx"]
        bank = e["bank"]
        cached = e.get("cached", [0, 0])
        c0 = cached[0] if len(cached) > 0 else 0
        c1 = cached[1] if len(cached) > 1 else 0
        oj = e.get("oj", -1)
        j = e.get("j", -1)
        trace_rows.append((token, layer, slot, ei, bank))
        oracle_rows.append((token, layer, slot, ei, bank, oj, j, c0, c1))

with open(trace_out, "w") as f:
    f.write("token,layer,slot,expert_idx,bank\n")
    for r in trace_rows:
        f.write(",".join(map(str, r)) + "\n")
with open(oracle_out, "w") as f:
    f.write("token,layer,slot,expert_idx,bank,oj,j,cached0,cached1\n")
    for r in oracle_rows:
        f.write(",".join(map(str, r)) + "\n")

tokens = (max(r[0] for r in trace_rows) + 1) if trace_rows else 0
print(f"records={len(records)} rows={len(trace_rows)} tokens={tokens}")

"""elb_train — EPM (Expert-Routing Prediction Model) offline tooling.

Phase 29 (PLAN.md EPM-1..5; authoritative design: spec/MoE-SpeQ_NOTES.md).
Out of the engine build: pure Python/numpy (PyTorch only where training
needs it, EPM-3+). EPM-1 ships `dataset` (raw-dump readers, shard
writer/reader, sequence-level splits, stats). EPM-2 ships `metrics`
(coverage-aware eval harness), `glm_router` (noaux_tc CPU reference +
standalone safetensors router loader), `baselines` (B0/B1/B2),
`synth_corpus` (known-process synthetic dumps), `run_baselines`
(study-dir report entrypoint). EPM-3 ships `wmap` (W[k][j] value map +
budgeted rank allocation), `model` (EpmPredictor: Tier-1 side adapters ->
frozen router replicas, direct-head/5-tap-fusion arms, safetensors +
sidecar checkpoints), `train` (config-driven deterministic trainer on
cached features; KL/BCE/MSE losses weighted by W), `configs/`.
"""

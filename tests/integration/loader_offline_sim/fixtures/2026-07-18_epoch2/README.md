# Epoch-2 fixtures (2026-07-18, post-LASTPASSER re-baseline)

Frozen snapshots of the I8 fitted/tuned artifacts as they stood when the P-25
campaign closed — kept available verbatim BEFORE the P-26 unified deployment
fit retrains them. Never edit in place; a refit writes NEW artifacts.

- `exposed_params_v2_reef_4gpu.json` — M2v2 exposed-wall params (fit by
  `tools/loader_xray/exposed_model.py --form v2` on the REEF 4-GPU capture;
  wall R² 0.821 / total 0.735; config-scoped — does NOT generalize). Copy of
  `../../assets/exposed_params_v2_reef_4gpu.json` at commit 6cebc682.
- `policy_params_reef_4gpu.json` — the freq/reuse policy weights validated by
  the epoch-2 regrid (12-run grid through KEEPER52_REEF_ORCH; ledger UPDATE 9):
  freq_w=60, freq_decay=0.1, reuse_w=2000, reuse_tau=300 — plateau optimum,
  identical to the pre-rebase tuning.

Context fingerprints for this epoch (gate-on canonical trace): REEF-ORCH
0.7327 (champion, 8.815-8.935 tok/s), engine-REEF 0.7328, affinity/board-LRU/E0
0.7233. Pre-rebase fingerprints (0.7340 etc.) are a CLOSED epoch.

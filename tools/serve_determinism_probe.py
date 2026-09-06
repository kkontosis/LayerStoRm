#!/usr/bin/env python
"""Served-prefill run-to-run determinism probe (TD-SERVE-PREFILL-NONDET-RUN-TO-RUN).

Boots a serving recipe, runs a WARM-UP request (the first request of a boot
may finalize MoE layers degraded — TD-V4-FIRSTREQ-COLD-SHARE-OVER-CAPACITY —
and is never a control), then REPS identical greedy (temperature 0) requests
of the same PROBE-token prompt back-to-back with the prefix cache disabled,
and reports:

  - per-request tokens + RequestStats.moe_degraded_layers (a degraded
    request is excluded from the verdict — its output is computed with an
    incomplete expert set and is not identity-comparable);
  - pairwise first-divergence indices;
  - the verdict: ALL IDENTICAL or not.

Determinism contract this probe verifies (see the TD entry):
  With compute.deterministic_ep_combine ON (or env
  LAYERSTORM_DETERMINISTIC_EP_COMBINE=1) and single-shot MoE batches
  (serving strides <= the single-shot chunk capacity — both production
  serving stacks), a served greedy prefill+decode at temperature 0 is
  reproducible within a boot at equal shapes.  With the flag OFF the routed
  EP combine is the legacy mode-0 placement-DEPENDENT partial sum
  (residency changes the FP reduction grouping run to run) and this probe
  is EXPECTED to fail — that configuration is a documented
  throughput/determinism tradeoff, and no token-identity gate may run on it.

  FOURTH precondition (TD-GLM52-CHAMPION-GREEDY-NONDET-RUN-TO-RUN,
  2026-09-02): on a speculative greedy arm the SpecRoundGovernor picks each
  round's shape (plain step vs draft+verify chunk) from wall-clock EMAs,
  and verify-chunk argmax is the batched forward's own output (a different
  FP shape from the width-1 plain step — INV-DSPARK-LOSSLESS B>1 clause),
  so near-tie tokens follow the wall-clock-random shape sequence.  This
  probe therefore DEFAULTS LS_SPEC_GOVERNOR=0 for its own process (export
  LS_SPEC_GOVERNOR explicitly to override and measure the governor's
  trajectory noise instead).

Env contract (mirrors the r12/nondet gate scripts):
  GATE_CFG     recipe json (required)
  GATE_CORPUS  whitespace-separated token-id corpus file (required)
  GATE_PROBE   probe prompt length in tokens   (default 6624)
  GATE_GEN     generated tokens per request    (default 16)
  GATE_REPS    identical repetitions           (default 3)
  GATE_WARMUP  warm-up prompt length           (default 1024)
  GATE_OUT     optional json results path

Run recipes (this box):
  V4  : scratchpad/nondet_v4_detcombine.sh (warm arena attach, ~25 s boot)
  GLM : scratchpad/nondet_glm_noise.sh (holder swap + cold boot ~350 s)
Exit code: 0 = all comparable runs identical, 1 = divergence, 2 = no
comparable runs (everything degraded).
"""
import json
import os
import sys
import threading

from orchestrator.orchestrator import (InferenceRequest, Orchestrator,
                                       SamplingParams)


def main():
    cfg = os.environ["GATE_CFG"]
    corpus = [int(x) for x in open(os.environ["GATE_CORPUS"]).read().split()]
    probe = int(os.environ.get("GATE_PROBE", "6624"))
    gen = int(os.environ.get("GATE_GEN", "16"))
    reps = int(os.environ.get("GATE_REPS", "3"))
    warmup = int(os.environ.get("GATE_WARMUP", "1024"))
    if len(corpus) < probe:
        sys.exit(f"corpus too short: {len(corpus)} < {probe}")

    # §4b precondition 4: the round-shape governor converts wall-clock noise
    # into trajectory choice on speculative greedy arms — identity verdicts
    # need it OFF (explicit LS_SPEC_GOVERNOR in the env overrides).
    gov = os.environ.setdefault("LS_SPEC_GOVERNOR", "0")
    print(f"[probe] LS_SPEC_GOVERNOR={gov}"
          + ("" if gov == "0" else "  (governor armed: trajectory noise is "
             "EXPECTED on a speculative arm — this is not an identity gate)"),
          flush=True)

    orch = Orchestrator.boot(cfg, eos_token_ids=())
    orch.prefix_cache = None          # uncached: no holders are ever created
    det = getattr(orch, "deterministic_ep_combine", True)
    print(f"[probe] deterministic_ep_combine={det}"
          + ("" if det else "  (mode-0 placement-dependent combine: "
             "divergence is EXPECTED — this arm measures the noise floor)"),
          flush=True)
    threading.Thread(target=orch.run, daemon=True).start()
    rid = [0]

    def run(prompt, n, tag):
        rid[0] += 1
        done, out = threading.Event(), {}
        orch.submit_request(InferenceRequest(
            request_id=rid[0], prompt_token_ids=list(prompt), max_tokens=n,
            sampling=SamplingParams(temperature=0.0),
            on_complete=lambda i, t, r, lp, error="": (
                out.update(t=list(t), e=error), done.set())))
        done.wait(3600)
        st = orch.last_stats
        print(f"[probe] {tag}: prefill={st.prefill_ms:9.1f}ms "
              f"degraded={st.moe_degraded_layers} toks={out.get('t')}",
              flush=True)
        return out.get("t"), st.moe_degraded_layers

    run(corpus[:warmup], 2, "warmup")
    runs = [run(corpus[:probe], gen, f"U{k + 1}") for k in range(reps)]

    comparable = [(i, t) for i, (t, d) in enumerate(runs) if d == 0 and t]
    print("", flush=True)
    for i, (t, d) in enumerate(runs):
        if d:
            print(f"[probe] U{i + 1} EXCLUDED from verdict: "
                  f"moe_degraded_layers={d}", flush=True)
    allsame = True
    for a in range(len(comparable)):
        for b in range(a + 1, len(comparable)):
            ia, ta = comparable[a]
            ib, tb = comparable[b]
            same = ta == tb
            allsame = allsame and same
            div = next((k for k, (x, y) in enumerate(zip(ta, tb))
                        if x != y), None)
            print(f"[probe] U{ia + 1} vs U{ib + 1}: same={same} "
                  f"first_div={div}", flush=True)
    if out := os.environ.get("GATE_OUT"):
        with open(out, "w") as f:
            json.dump({"runs": [{"tokens": t, "degraded": d}
                                for t, d in runs],
                       "identical": allsame,
                       "comparable": len(comparable)}, f, indent=1)
    if not comparable or len(comparable) < 2:
        print("[probe] VERDICT: NO COMPARABLE RUNS", flush=True)
        code = 2
    else:
        print(f"[probe] VERDICT: "
              f"{'IDENTICAL' if allsame else 'DIVERGENT'} over "
              f"{len(comparable)} comparable run(s)"
              + ("" if det or allsame
                 else "  [expected: deterministic_ep_combine is OFF]"),
              flush=True)
        code = 0 if allsame else 1
    orch.stop_engine()
    os._exit(code)


if __name__ == "__main__":
    main()

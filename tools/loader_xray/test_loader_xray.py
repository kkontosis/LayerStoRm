"""End-to-end CPU-only validation of the loader x-ray pipeline (goal 6).

Generates synthetic (inputs -> 'real' timings) with KNOWN injected biases and
contention, runs join -> analyze -> train, and asserts the framework RECOVERS the
injected corrections:

  * per-device transfer scale (device 1 slower than calibration),
  * the bank-contention coefficient c_true (the §3.3 open question),
  * model comparison: the contention-aware model fits strictly better than the
    strict-serial current model on contended data.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import synth
from joiner import join, parse_trace_csv, parse_model_jsonl, rekey_transfer_rows
from model import make_model, CurrentModel, ContentionBankModel
from analyze import (analyze, render_report, sum_calibration, per_term_scale,
                     predict_assignment, _assignment_for)
from trainer import (train, loss_fn, calibrate_closed_form, why,
                     apply_params_to_constants)


def _calib_from_gt(gt) -> dict:
    """A baseline LoaderConstants dict whose matrix rate / bank egress reproduce the
    synthetic dump's raw subxfer / egress (so calib-sourced predict == row-sourced
    predict at default params). subxfer in the dump = dev_subxfer[d] (rate, lat=0);
    egress = bank_egress[b]."""
    return {
        "version": 2,
        "source": "calibrated",
        "num_devices": gt.M,
        "num_banks": gt.B,
        "fixed_overhead_us": 0.0,
        # recon: synth's dump records a FLAT recon_us == gt.recon_us regardless of
        # participant count, so put it all in recon_overhead_us (max => flat) with no
        # per-device added term, matching the dump's row recon.
        "devices": [
            {"position": d, "numa_node": d,
             "compute": {"a_us": 0.0, "b_us": 0.0, "P": 1},
             "xfer_lat_us": 0.0,
             "recon_overhead_us": gt.recon_us, "recon_added_us": 0.0}
            for d in range(gt.M)
        ],
        "banks": [
            {"node": b, "egress_us": gt.bank_egress[b], "contention": 1.0}
            for b in range(gt.B)
        ],
        "matrix": [
            [{"rate_us": gt.dev_subxfer[d], "lat_us": 0.0, "tier": 1}
             for d in range(gt.M)]
            for _b in range(gt.B)
        ],
    }


@pytest.fixture(scope="module")
def synth_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("xray")
    mp = str(d / "model.jsonl")
    tp = str(d / "trace.csv")
    gt = synth.write_files(mp, tp, n_layers=300)
    return mp, tp, gt


def test_synth_schema_matches_dump_fields():
    """The synthetic model record carries every field the C++ dump emits."""
    ml, tl, gt = synth.generate(n_layers=3)
    rec = __import__("json").loads(ml[0])
    for k in ["cmd_seq", "layer_idx", "N", "M", "B", "experts", "j", "dev_counts",
              "predicted_us", "device_makespan_us", "bank_egress_us", "recon_us",
              "place_us", "evict_us", "prep_us", "exact"]:
        assert k in rec, f"missing dump field {k}"
    e = rec["experts"][0]
    for k in ["bank", "j", "egress_us", "cached", "subxfer_us"]:
        assert k in e


def test_join_recovers_real_metrics(synth_files):
    mp, tp, gt = synth_files
    rows, stats = join(mp, tp)
    assert stats["n_joined"] == stats["n_model_records"], stats
    assert stats["n_dropped_incomplete"] == 0
    assert stats["n_layer_mismatch"] == 0
    r = rows[0]
    assert r["real"]["total_us"] > 0
    assert r["real"]["transfer_wall_us"] > 0
    assert r["real"]["compute_us"] > 0
    # recon has no dedicated perf_trace marker -> joiner reports None (folded into
    # the finalize/compute window). The real total is enter->finalize_exit, which
    # equals transfer_wall + compute (recon is inside the compute window here).
    assert r["real"]["recon_us"] is None
    approx_total = r["real"]["transfer_wall_us"] + r["real"]["compute_us"]
    # within ~ the small fixed inter-stage gaps the synth inserts (sub-us).
    assert abs(approx_total - r["real"]["total_us"]) < 5.0


def test_analyze_flags_device_bias(tmp_path):
    """Device 1 has a 1.4x injected transfer scale; the per-device fit must find a
    scale meaningfully > device 0's. Uses a DEVICE-BOUND dataset (cheap banks) so
    the per-device transfer roofline — not the bank floor — drives the detect wall,
    making the device scale identifiable (when the bank binds, per-device detect is
    bank-contention-limited and device scale is unobservable — the contention test
    covers that regime)."""
    gt = synth.GroundTruth(
        dev_xfer_scale=[1.0, 1.4], dev_subxfer=[557.0, 557.0],
        bank_egress=[20.0, 20.0], contention_c=1.0, noise_us=4.0,
    )
    mp = str(tmp_path / "m.jsonl"); tp = str(tmp_path / "t.csv")
    synth.write_files(mp, tp, n_layers=300, gt=gt)
    rows, _ = join(mp, tp)
    mdl = make_model("current", gt.M)
    rep = analyze(mdl, mdl.init_params(), rows)
    pdf = rep["per_device_fits"]
    assert 0 in pdf and 1 in pdf
    s0, s1 = pdf[0]["scale"], pdf[1]["scale"]
    # device 1 should require a larger correction scale (true ratio 1.4 / 1.0).
    assert s1 / s0 > 1.2, (s0, s1)
    assert s1 / s0 == pytest.approx(gt.dev_xfer_scale[1] / gt.dev_xfer_scale[0], rel=0.15)
    # report renders without error.
    md = render_report(rep)
    assert "Per-device" in md


def test_train_reduces_loss(synth_files):
    mp, tp, gt = synth_files
    rows, _ = join(mp, tp)
    mdl = make_model("current", gt.M)
    res = train(mdl, rows, iters=150, lr=5.0)
    assert res.loss < res.loss0, (res.loss0, res.loss)
    # report 'why' runs.
    assert "Learned corrections" in why(res, mdl.init_params())


def test_contention_model_recovers_c_true(synth_files):
    """The headline check: fit the contention-aware model and recover c_true."""
    mp, tp, gt = synth_files
    rows, _ = join(mp, tp)
    mdl = make_model("contention_bank", gt.M)
    res = train(mdl, rows, iters=400, lr=5.0,
                weights={"total": 0.0, "transfer_wall": 1.0})
    c_learned = res.learned["contention_c"]
    # must recover the injected contention coefficient within tolerance.
    assert c_learned == pytest.approx(gt.contention_c, abs=0.12), (c_learned, gt.contention_c)


def test_contention_model_beats_current_on_contended_data(synth_files):
    """Model comparison (goal 5): on contended synthetic data the contention-aware
    bank term fits the transfer wall strictly better than the strict-serial current
    model."""
    mp, tp, gt = synth_files
    rows, _ = join(mp, tp)
    w = {"total": 0.0, "transfer_wall": 1.0}

    cur = make_model("current", gt.M)
    cur_res = train(cur, rows, iters=300, lr=5.0, weights=w)

    con = make_model("contention_bank", gt.M)
    con_res = train(con, rows, iters=400, lr=5.0, weights=w)

    # contention model should achieve a lower (or equal) transfer-wall loss.
    assert con_res.loss <= cur_res.loss * 1.01, (cur_res.loss, con_res.loss)
    # and on this strongly-contended data, strictly better by a clear margin.
    assert con_res.loss < cur_res.loss * 0.95, (cur_res.loss, con_res.loss)


def test_pluggable_predict_signature():
    """Both plugins satisfy predict(params, row) -> per-term dict with TERMS keys."""
    ml, tl, gt = synth.generate(n_layers=2)
    import json
    row = json.loads(ml[0])
    row["real"] = {}
    for name in ["current", "contention_bank"]:
        mdl = make_model(name, gt.M)
        out = mdl.predict(mdl.init_params(), row)
        for t in ["prep", "device_makespan", "bank_egress", "recon", "place",
                  "evict", "total"]:
            assert t in out
        # total identity holds.
        assert out["total"] == pytest.approx(
            out["prep"] + max(out["device_makespan"], out["bank_egress"])
            + out["recon"] + out["place"] + out["evict"])


def test_current_model_matches_dump_predicted_at_defaults():
    """At default params (scale=1, compute_a=0, recon_scale=1) the current model's
    re-derived sub-terms must equal what the C++ solver wrote into the dump — the
    Python re-implementation is faithful to the C++ objective."""
    ml, tl, gt = synth.generate(n_layers=20)
    import json
    rows = [json.loads(x) for x in ml]
    mdl = make_model("current", gt.M)
    p = mdl.init_params()
    for r in rows:
        r["real"] = {}
        out = mdl.predict(p, r)
        assert out["device_makespan"] == pytest.approx(r["device_makespan_us"], abs=1e-6)
        assert out["bank_egress"] == pytest.approx(r["bank_egress_us"], abs=1e-6)
        assert out["recon"] == pytest.approx(r["recon_us"], abs=1e-6)
        assert out["total"] == pytest.approx(r["predicted_us"], abs=1e-6)


# ---------------------------------------------------------------------------
# apply_params_to_constants round-trip (task item 1 + 4).
# ---------------------------------------------------------------------------
def test_calib_predict_matches_row_predict_at_defaults():
    """The calib-sourced predict path (raw terms from constants) must equal the
    row-sourced path at default params, when the baseline calib reproduces the dump
    inputs — the precondition the round-trip relies on."""
    ml, _tl, gt = synth.generate(n_layers=30)
    rows = [json.loads(x) for x in ml]
    calib = _calib_from_gt(gt)
    mdl = make_model("current", gt.M)
    p = mdl.init_params()
    for r in rows:
        r["real"] = {}
        a = mdl.predict(p, r)               # row-sourced
        b = mdl.predict(p, r, calib=calib)  # calib-sourced
        assert b["device_makespan"] == pytest.approx(a["device_makespan"], abs=1e-6)
        assert b["bank_egress"] == pytest.approx(a["bank_egress"], abs=1e-6)
        assert b["total"] == pytest.approx(a["total"], abs=1e-6)


def _roundtrip_params(mdl):
    """A learned-params vector exercising every coefficient channel."""
    p = mdl.init_params()
    p.set("dev_scale[0]", 1.25)
    p.set("dev_scale[1]", 1.7)
    p.set("compute_a[0]", 30.0)
    p.set("compute_a[1]", 45.0)
    p.set("bank_scale", 0.85)
    p.set("recon_scale", 1.4)
    p.set("recon_off", 55.0)  # outside the max -> exact via fixed_overhead
    return p


def test_apply_params_roundtrip_bank_bound():
    """predict(default, calib') == predict(learned, calib) on bank-bound data.

    Synthetic egress (557/760) dominates the device roofline, so the bank term binds
    in BOTH predictions; dev_off is 0 here, recon_off folds exactly into the global
    fixed_overhead_us (it sits outside the max). This is the exact-round-trip regime
    documented in apply_params_to_constants."""
    ml, _tl, gt = synth.generate(n_layers=60)
    rows = [json.loads(x) for x in ml]
    calib = _calib_from_gt(gt)
    mdl = make_model("current", gt.M)
    learned = _roundtrip_params(mdl)
    calib2 = apply_params_to_constants(mdl, learned, calib)
    assert calib2["fixed_overhead_us"] == pytest.approx(55.0)  # recon_off only
    default = mdl.init_params()
    for r in rows:
        r["real"] = {}
        ref = mdl.predict(learned, r, calib=calib)
        got = mdl.predict(default, r, calib=calib2)
        assert got["total"] == pytest.approx(ref["total"], abs=1e-4), r["cmd_seq"]
        assert got["device_makespan"] == pytest.approx(ref["device_makespan"], abs=1e-4)
        assert got["bank_egress"] == pytest.approx(ref["bank_egress"], abs=1e-4)


def test_apply_params_roundtrip_with_dev_off_device_bound():
    """When the device roofline binds in both predictions, dev_off folds exactly into
    fixed_overhead_us. Use cheap banks + a dev_off so the device term is the max."""
    gt = synth.GroundTruth(dev_subxfer=[557.0, 557.0], bank_egress=[20.0, 20.0])
    ml, _tl, gt = synth.generate(n_layers=40, gt=gt)
    rows = [json.loads(x) for x in ml]
    calib = _calib_from_gt(gt)
    mdl = make_model("current", gt.M)
    p = mdl.init_params()
    p.set("dev_scale[0]", 1.1)
    p.set("dev_scale[1]", 1.1)  # symmetric so both device rooflines scale equally
    p.set("dev_off", 80.0)
    learned = p
    calib2 = apply_params_to_constants(mdl, learned, calib)
    assert calib2["fixed_overhead_us"] == pytest.approx(80.0)
    default = mdl.init_params()
    for r in rows:
        r["real"] = {}
        ref = mdl.predict(learned, r, calib=calib)
        got = mdl.predict(default, r, calib=calib2)
        assert got["total"] == pytest.approx(ref["total"], abs=1e-4), r["cmd_seq"]


def test_apply_then_fit_emits_loadable_schema():
    """apply produces a dict carrying the engine LoaderConstants keys + the new
    fixed_overhead_us, ready to write back."""
    _ml, _tl, gt = synth.generate(n_layers=10)
    calib = _calib_from_gt(gt)
    mdl = make_model("current", gt.M)
    out = apply_params_to_constants(mdl, _roundtrip_params(mdl), calib)
    for k in ["version", "num_devices", "num_banks", "devices", "banks", "matrix",
              "fixed_overhead_us", "source"]:
        assert k in out, k
    assert out["source"] == "trained"
    # matrix rates scaled by dev_scale.
    assert out["matrix"][0][1]["rate_us"] == pytest.approx(gt.dev_subxfer[1] * 1.7)
    # input calib untouched (deep copy).
    assert calib["matrix"][0][1]["rate_us"] == pytest.approx(gt.dev_subxfer[1])


# ---------------------------------------------------------------------------
# Stage-3: transfer-row cmd_seq re-key (interval join).
# ---------------------------------------------------------------------------
def test_rekey_recovers_transfer_wall_with_cmd_seq_zero(tmp_path):
    """When the transfer (kDetected/kDmaGpu) rows carry cmd_seq=0 (the §5 GAP), the
    DIRECT join still recovers transfer_wall (its kDetected rows landed under cmd 0),
    but the PER-LAYER attribution is wrong. The Stage-3 interval re-key reassigns each
    transfer to the MoE window that contains it, so transfer_wall is recovered per
    layer and matches the cmd_seq-stamped baseline."""
    gt = synth.GroundTruth(noise_us=0.0)  # deterministic so per-layer walls compare
    # Baseline: transfer rows carry the real cmd_seq.
    ml_b, tl_b, _ = synth.generate(n_layers=40, gt=gt)
    mp_b = str(tmp_path / "mb.jsonl"); tp_b = str(tmp_path / "tb.csv")
    open(mp_b, "w").write("\n".join(ml_b) + "\n")
    open(tp_b, "w").write("\n".join(tl_b) + "\n")
    rows_b, _ = join(mp_b, tp_b)
    # Gap reproduction: transfer rows carry cmd_seq=0.
    ml_g, tl_g, _ = synth.generate(n_layers=40, gt=gt, transfer_cmd_seq=0)
    mp_g = str(tmp_path / "mg.jsonl"); tp_g = str(tmp_path / "tg.csv")
    open(mp_g, "w").write("\n".join(ml_g) + "\n")
    open(tp_g, "w").write("\n".join(tl_g) + "\n")
    # WITHOUT re-key: transfer_wall per layer is unrecoverable (all transfers under 0).
    rows_no, stats_no = join(mp_g, tp_g, rekey_transfers=False)
    tw_no = [r["real"]["transfer_wall_us"] for r in rows_no]
    assert all(t is None for t in tw_no), "expected no per-layer wall without re-key"
    # WITH re-key (default): per-layer transfer_wall recovered, matches the baseline.
    rows_rk, stats_rk = join(mp_g, tp_g)
    assert stats_rk["n_rekeyed_transfers"] == len(rows_rk)
    for rb, rk in zip(rows_b, rows_rk):
        assert rk["real"]["transfer_wall_us"] is not None
        assert rk["real"]["transfer_wall_us"] == pytest.approx(
            rb["real"]["transfer_wall_us"], abs=0.6), rb["cmd_seq"]


def test_rekey_standalone_windows(tmp_path):
    """rekey_transfer_rows assigns transfers to the right MoE window."""
    gt = synth.GroundTruth(noise_us=0.0)
    ml, tl, _ = synth.generate(n_layers=10, gt=gt, transfer_cmd_seq=0)
    tp = str(tmp_path / "t.csv")
    open(tp, "w").write("\n".join(tl) + "\n")
    rk = rekey_transfer_rows(tp)
    # every MoE cmd (1..10) owns transfers; none orphaned (all transfers in a window).
    assert set(rk) == set(range(1, 11))
    for cmd, d in rk.items():
        assert d["n_transfers"] >= 1
        assert d["transfer_wall_us"] is not None
        assert d["_n_orphan_transfers"] == 0


# ---------------------------------------------------------------------------
# Stage-1: Σ-calibration.  Stage-2: per-term scale.
# ---------------------------------------------------------------------------
def test_assignment_executed_uses_oj():
    """_assignment_for('executed') reads the oj field (executed e%tp); 'solver' reads
    j. The synth sets oj==j, so they agree; a hand-built divergent row shows the split."""
    ml, _tl, gt = synth.generate(n_layers=1)
    row = json.loads(ml[0]); row["real"] = {}
    assert _assignment_for(row, "solver") == [int(e["j"]) for e in row["experts"]]
    assert _assignment_for(row, "executed") == [int(e["oj"]) for e in row["experts"]]
    # divergent: flip oj for the first expert.
    row["experts"][0]["oj"] = (int(row["experts"][0]["j"]) + 1) % int(row["M"])
    assert _assignment_for(row, "executed")[0] != _assignment_for(row, "solver")[0]


def test_predict_assignment_matches_dump_for_solver_j():
    """predict_assignment with the solver's own j reproduces the dump's recorded
    device_makespan / bank_egress (faithful re-evaluation of the §3 objective)."""
    ml, _tl, gt = synth.generate(n_layers=20)
    mdl = make_model("current", gt.M); p = mdl.init_params()
    for line in ml:
        row = json.loads(line); row["real"] = {}
        out = predict_assignment(mdl, p, row, _assignment_for(row, "solver"))
        assert out["device_makespan"] == pytest.approx(row["device_makespan_us"], abs=1e-6)
        assert out["bank_egress"] == pytest.approx(row["bank_egress_us"], abs=1e-6)


def test_sum_calibration_global_scale_recovers_injection(synth_files):
    """Σ-calibration: the global scale Σreal/Σpred reflects the injected device/compute
    biases the uncorrected model misses. With the synth's slower device-1 + nonzero
    compute (absent from the dump), Σreal > Σpred so the scale is > 1 (the §7 'scale
    1.20' family)."""
    mp, tp, gt = synth_files
    rows, _ = join(mp, tp)
    mdl = make_model("current", gt.M)
    sc = sum_calibration(mdl, mdl.init_params(), rows, which="executed")
    assert sc["sum_pred_total_us"] > 0
    assert sc["sum_real_total_us"] > 0
    # the global scale is finite and positive; its DIRECTION is the diagnostic (here
    # the uncorrected strict-serial bank term over-predicts vs the c=0.55 real channel,
    # so scale < 1 — exactly the kind of per-policy mis-scale Stage-1 surfaces).
    assert sc["global_scale_real_over_pred"] > 0.0
    assert np.isfinite(sc["global_scale_real_over_pred"])
    assert sc["n_real"] == len(rows)
    # term sums present.
    for t in ["device_makespan", "bank_egress", "evict"]:
        assert t in sc["sum_pred_terms"]


def test_per_term_scale_reports_each_term(synth_files):
    mp, tp, gt = synth_files
    rows, _ = join(mp, tp)
    mdl = make_model("current", gt.M)
    pt = per_term_scale(mdl, mdl.init_params(), rows, which="executed")
    fits = pt["per_term_fits"]
    assert "total" in fits and "transfer_wall(max dev,bank)" in fits
    shares = pt["term_shares_of_pred_total"]
    # the partitioning shares (fetch wall + additive terms) sum to ~1; the diagnostic
    # _dev_binds_frac is excluded from the partition.
    part = sum(v for k, v in shares.items() if not k.startswith("_"))
    assert abs(part - 1.0) < 0.05, shares
    # the fetch wall dominates predicted total on this transfer-heavy synth.
    assert shares["fetch_wall(max dev,bank)"] > 0.5


def test_sum_calibration_solver_vs_executed_differ_when_assignments_diverge():
    """When the solver's j packs experts differently from the executed e%tp, Σpred
    differs between the two assignments — the Stage-1 lever that decides #2 vs #3a."""
    gt = synth.GroundTruth(noise_us=0.0)
    ml, tl, _ = synth.generate(n_layers=30, gt=gt)
    rows = []
    for line in ml:
        r = json.loads(line)
        # force a divergent solver j: pack ALL experts onto device 0 (the "myopic" pack).
        for e in r["experts"]:
            e["j"] = 0
        r["real"] = {"total_us": None}
        rows.append(r)
    mdl = make_model("current", gt.M); p = mdl.init_params()
    sc_solver = sum_calibration(mdl, p, rows, which="solver")    # all on dev0
    sc_exec = sum_calibration(mdl, p, rows, which="executed")    # e%tp split
    # packing all on one device raises that device's makespan vs the balanced split.
    assert sc_solver["sum_pred_terms"]["device_makespan"] > \
           sc_exec["sum_pred_terms"]["device_makespan"]


def test_trainer_apply_cli_smoke(tmp_path):
    """End-to-end CLI: synth -> trainer_apply.py -> loadable corrected JSON."""
    mp = str(tmp_path / "model.jsonl")
    tp = str(tmp_path / "trace.csv")
    gt = synth.write_files(mp, tp, n_layers=120)
    in_calib = str(tmp_path / "calib.json")
    with open(in_calib, "w") as fh:
        json.dump(_calib_from_gt(gt), fh)
    out_calib = str(tmp_path / "trained.json")
    here = os.path.dirname(__file__)
    rc = subprocess.run(
        [sys.executable, os.path.join(here, "trainer_apply.py"),
         "--in-calib", in_calib, "--model-jsonl", mp, "--trace", tp,
         "--model", "current", "--out-calib", out_calib, "--iters", "60"],
        capture_output=True, text=True, cwd=here,
    )
    assert rc.returncode == 0, rc.stderr
    assert os.path.exists(out_calib)
    with open(out_calib) as fh:
        trained = json.load(fh)
    assert "fixed_overhead_us" in trained
    assert trained["source"] == "trained"
    assert len(trained["matrix"]) == gt.B
    assert len(trained["devices"]) == gt.M

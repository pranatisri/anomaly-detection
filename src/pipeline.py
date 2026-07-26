"""End-to-end run: signals -> fusion -> alerts -> imbalance-aware metrics.

Fit and evaluation use DIFFERENT SEEDS by default. A model fitted and scored on one
dataset tells you nothing about whether it learned behaviour or memorised a sample.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import evaluate as ev
from config import (
    ATTACK_TYPES,
    INSIDER_DRIFT,
    LEVEL_DAILY_BUDGET,
    LEVELS,
    NORMAL,
    assert_feature_frame,
)
from detector import Detector, LEVEL_SIGNALS, explain, predict_type, risk_band
from features import SIGNAL_NAMES, extract

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load(tag: str, data_dir: str = DATA):
    d = os.path.abspath(data_dir)
    ev_df = pd.read_parquet(os.path.join(d, "events_%s.parquet" % tag))
    lb = pd.read_parquet(os.path.join(d, "labels_%s.parquet" % tag))
    cp = pd.read_parquet(os.path.join(d, "campaigns_%s.parquet" % tag))
    return ev_df, lb, cp


def _rank_p_by_cohort(raw: np.ndarray, cohort: np.ndarray) -> np.ndarray:
    """Empirical upper-tail rank of each score within its own cohort, on live data.

    Contamination is 0.5-3%, so the ranks of the benign bulk are essentially unaffected
    by the anomalies mixed in; this needs no labels.
    """
    raw = np.asarray(raw, dtype=float)
    coh = np.asarray(cohort).astype(str)
    out = np.empty(raw.shape, dtype=float)
    for c in np.unique(coh):
        m = coh == c
        v = raw[m]
        order = np.argsort(np.argsort(-v))       # 0 = most extreme
        out[m] = (1.0 + order) / (1.0 + len(v))
    return out


def score_dataset(det: Detector, events: pd.DataFrame, sig: pd.DataFrame
                  ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, np.ndarray]]:
    """Produce a scored frame per level, plus a per-event max-across-levels score."""
    base = events[["event_id", "entity_id", "entity_type", "source_ip", "timestamp"]].merge(
        sig[["event_id"]], on="event_id", how="right")

    scored: Dict[str, pd.DataFrame] = {}
    zmap: Dict[str, np.ndarray] = {}
    per_level_scores = []
    for lv in LEVELS:
        cal, raw, z = det.score(sig, lv)
        zmap[lv] = z
        df = base.copy()
        df["score"] = raw          # within-level ranking: full resolution
        df["score_cal"] = cal      # analyst-facing probability
        # Per-event tail probability, needed for the alert-level multiple-comparisons
        # correction in evaluate.group_alerts.
        # Per-event tail probability, re-centred on the LIVE population per cohort.
        #
        # Taking p from the training reference alone leaves a systematic offset when the
        # live stream differs at all from the fitting seed: mean benign z sat at +0.13 to
        # +0.18 instead of 0. That looks negligible until it meets Stouffer, which scales
        # it by sqrt(n) -- so a 200-event alert inherits +2.0 of pure offset while a
        # 5-event alert inherits +0.28, and the queue sorts by entity busyness again.
        #
        # Ranking p within the live stream's own cohort makes it uniform by construction,
        # so what accumulates is deviation from what that cohort is doing NOW. It uses no
        # labels, and it is what a deployed detector does anyway when it recalibrates
        # against a trailing window of live traffic.
        p_train = det.fusions[lv].raw_to_p(raw, base["entity_type"].to_numpy())
        df["p"] = _rank_p_by_cohort(raw, base["entity_type"].to_numpy())
        df["p_train"] = p_train
        scored[lv] = df
        per_level_scores.append(cal)

    # The analyst works all three queues, so an event's overall risk is the strongest
    # case any level makes for it. All three are calibrated probabilities, so max is
    # meaningful rather than an apples-to-oranges comparison.
    combined = base.copy()
    combined["score"] = np.max(np.column_stack(per_level_scores), axis=1)

    # Banding is done PER LEVEL against that level's own threshold, then combined.
    # A single global threshold cannot work: theta lives in each level's raw space, and
    # comparing one level's theta against a max-of-calibrated score put every event in
    # the LOW band regardless of what it did.
    band_rank = np.zeros(len(combined), dtype=int)
    for lv in LEVELS:
        th = det.fusions[lv].theta_
        if th is None:
            continue
        r = scored[lv]["score"].to_numpy()
        lvl_rank = np.where(r >= th, 2, np.where(r >= 0.5 * th, 1, 0))
        band_rank = np.maximum(band_rank, lvl_rank)
    combined["band"] = np.array(["LOW", "MEDIUM", "HIGH"])[band_rank]
    return scored, combined, zmap


def _english(scope: str, why: List[Dict[str, object]], when) -> str:
    """One plain sentence. Two z-score bullets are not an explanation an analyst can act on."""
    if not why:
        return "%s flagged on diffuse evidence with no single dominant signal, %s." % (
            scope, str(when)[:16])
    parts = [str(w["text"]) for w in why[:3]]
    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = "%s and %s" % (parts[0], parts[1])
    else:
        body = "%s, %s, and %s" % (parts[0], parts[1], parts[2])
    return "Flagged: %s - %s, on %s." % (scope, body, str(when)[:16])


def top_alerts_with_explanations(det: Detector, scored: Dict[str, pd.DataFrame],
                                 sig: pd.DataFrame, events: pd.DataFrame,
                                 zmap: Dict[str, np.ndarray], level: str = "entity",
                                 n: int = 10,
                                 only_ids: Optional[Sequence[str]] = None,
                                 alerts: Optional[pd.DataFrame] = None,
                                 assign: Optional[pd.DataFrame] = None
                                 ) -> List[Dict[str, object]]:
    """Rank alerts and attach L6 explanations. Shared by the CLI and the dashboard.

    The explanation is a sort over already-named quantities, not a post-hoc attribution
    over an opaque score. That is the whole return on building the detector in layers.

    `only_ids` restricts the work to specific alerts, and `alerts`/`assign` let a caller
    pass in a grouping it has already computed. Both matter for the dashboard: it
    displays 40 alerts but there are ~58,000 of them, and explaining all of them takes
    over ten minutes -- during which the page renders its header and then appears to hang
    with empty tabs.
    """
    if alerts is None or assign is None:
        alerts, assign = ev.group_alerts(scored[level], level)
    if only_ids is not None:
        want = set(only_ids)
        alerts = alerts[alerts["alert_id"].isin(want)]
    alerts = alerts.sort_values(["alert_score", "alert_id"], ascending=[False, True]).head(n)

    fu = det.fusions[level]
    names = fu.signals
    idx = {e: i for i, e in enumerate(sig["event_id"].to_numpy())}
    roles = dict(zip(events["entity_id"], events["role"]))
    keep_ids = set(alerts["alert_id"])
    a2e: Dict[str, List[str]] = {}
    for e, a in zip(assign["event_id"].to_numpy(), assign["alert_id"].to_numpy()):
        if a in keep_ids:
            a2e.setdefault(a, []).append(e)
    # Column-major numpy view: sig.iloc[row] per alert is a pandas row construction and
    # dominates the runtime once there is more than a handful of alerts.
    sig_cols = {nm: sig[nm].to_numpy() for nm in names if nm in sig.columns}
    cold_col = sig["_cold_start"].to_numpy() if "_cold_start" in sig.columns else None

    theta = fu.theta_ or 0.0
    out: List[Dict[str, object]] = []
    for _, row in alerts.iterrows():
        members = a2e.get(row["alert_id"], [])
        if not members:
            continue
        rows = [idx[m] for m in members if m in idx]
        if not rows:
            continue
        # Explain the single strongest member event -- the one the analyst opens first.
        best = max(rows, key=lambda i: zmap[level][i].max())
        raw_vals = {nm: float(sig_cols[nm][best]) for nm in sig_cols}
        scope = str(row["scope_key"])
        cold = bool(cold_col[best]) if cold_col is not None else False
        zrow = zmap[level][best]
        why = explain(zrow, names, raw_vals, roles.get(scope, ""), top_k=5)
        # Type attribution needs the evidence from EVERY level, not just this one.
        # credential_stuffing is characterised by IP-level fan-out and low_and_slow by
        # long-window signals, so scoring the rules against entity-level z alone made
        # those two types unreachable -- 49 low_and_slow alerts were attributed to
        # lateral_movement purely because the signals that distinguish them were absent.
        # Evidence for ATTRIBUTION is the max over ALL member events, per signal --
        # not the vector of one representative event.
        #
        # Using a single "best" event (chosen by max entity-level z) systematically lost
        # credential stuffing: its defining evidence is IP fan-out across many entities,
        # which peaks on different events than the entity-level maximum. 13 stuffing
        # campaigns were attributed to brute_force despite the two being cleanly
        # separable on the raw signals (fan-out 12.0 vs 1.4, entity fail-rate 0.08 vs 27).
        ridx = np.asarray(rows, dtype=int)
        all_z: Dict[str, float] = {}
        for lv2 in LEVELS:
            fu2 = det.fusions[lv2]
            block = zmap[lv2][ridx]
            mx = block.max(axis=0)
            for j2, nm2 in enumerate(fu2.signals):
                all_z[nm2] = float(mx[j2])
        ptype, pconf = predict_type(all_z)
        summary = _english(scope, why, row["start_ts"])
        out.append({
            "alert_id": row["alert_id"], "level": level, "scope_key": scope,
            "score": float(row["alert_score"]),
            "band": risk_band(float(row["alert_score"]), theta),
            "start_ts": row["start_ts"], "end_ts": row["end_ts"],
            "n_events": int(row["n_events"]),
            "cold_start": cold,
            "n_history": int(sig["_n_history"].to_numpy()[best]) if "_n_history" in sig.columns else 0,
            "why": why,
            "pred_type": ptype,
            "pred_confidence": pconf,
            "summary": summary,
        })
    return out


def run(fit_tag: str, eval_tag: str, data_dir: str = DATA, calibrate: bool = True,
        verbose: bool = True, explain_n: int = 0) -> Dict[str, object]:
    t0 = time.time()
    fit_events, fit_labels, _ = load(fit_tag, data_dir)
    assert_feature_frame(fit_events, "detector(fit)")

    if verbose:
        print("extracting signals: fit set %s (%d events)" % (fit_tag, len(fit_events)))
    fit_sig, _ = extract(fit_events)

    y_fit = (fit_events[["event_id"]]
             .merge(fit_labels[["event_id", "label"]], on="event_id", how="left")["label"]
             .fillna(NORMAL).isin(ATTACK_TYPES).to_numpy().astype(int))
    y_fit = (fit_sig[["event_id"]].merge(
        fit_labels[["event_id", "label"]], on="event_id", how="left")["label"]
        .fillna(NORMAL).isin(ATTACK_TYPES).to_numpy().astype(int))

    fit_cohort = (fit_sig[["event_id"]].merge(
        fit_events[["event_id", "entity_type"]], on="event_id", how="left")["entity_type"]
        .fillna("?").to_numpy())
    det = Detector().fit(fit_sig, y_fit if calibrate else None, calibrate=calibrate,
                         cohort=fit_cohort)

    if eval_tag == fit_tag:
        ev_events, ev_labels, ev_camps = fit_events, fit_labels, load(fit_tag, data_dir)[2]
        ev_sig = fit_sig
    else:
        ev_events, ev_labels, ev_camps = load(eval_tag, data_dir)
        assert_feature_frame(ev_events, "detector(eval)")
        if verbose:
            print("extracting signals: eval set %s (%d events)" % (eval_tag, len(ev_events)))
        ev_sig, _ = extract(ev_events)

    scored, combined, zmap = score_dataset(det, ev_events, ev_sig)

    ts = pd.to_datetime(ev_events["timestamp"])
    n_days = max(1.0, (ts.max() - ts.min()).total_seconds() / 86400.0)

    alerts_by_level, assign_by_level = {}, {}
    for lv in LEVELS:
        a, asg = ev.group_alerts(scored[lv], lv)
        alerts_by_level[lv] = a
        assign_by_level[lv] = asg

    # The analyst budget scales with the size of the estate, not with a fixed constant.
    # A flat 50 alerts/day for a 200-entity network is not a realistic SOC workload, and
    # it caps Incident_Precision at n_campaigns/K for reasons that have nothing to do
    # with the detector.
    n_entities = int(ev_events["entity_id"].nunique())
    budgets = {
        "entity": max(2, int(round(0.03 * n_entities))),
        "ip": max(1, int(round(0.01 * n_entities))),
        "long": max(1, int(round(0.01 * n_entities))),
    }

    event_m = ev.event_level_metrics(combined[["event_id", "score"]], ev_labels)
    inc_m = ev.incident_metrics(alerts_by_level, assign_by_level, ev_labels, ev_camps,
                                n_days=n_days, budgets=budgets,
                                event_times=ev_events[["event_id", "timestamp"]])
    fp = ev.fp_breakdown_by_confounder(combined[["event_id", "score"]], ev_labels)
    theta = det.fusions["entity"].theta_
    bands = ev.insider_drift_band_report(combined[["event_id", "score", "band"]], ev_labels, theta)

    out = {
        "fit_tag": fit_tag, "eval_tag": eval_tag, "n_days": n_days,
        "event": event_m, "incident": inc_m,
        "fp_breakdown": fp, "insider_bands": bands,
        "theta_entity": theta, "seconds": round(time.time() - t0, 1),
        "held_out": fit_tag != eval_tag,
    }

    if verbose:
        print()
        print(ev.format_headline(event_m, inc_m))
        print()
        print("PER-TYPE campaign recall (denominator = campaigns of that type)")
        for k, v in sorted(inc_m["per_type"].items()):
            ttd = v.get("median_ttd_hours")
            print("  %-22s %5.3f  (%d/%d)%s" % (
                k, v["recall"], v["n_detected"], v["n_campaigns"],
                "   median TTD %6.1f h" % ttd if ttd is not None else ""))
        print()
        print("FALSE POSITIVES by engineered benign behaviour (top 1%% budget)")
        for _, r in fp.head(8).iterrows():
            print("  %-28s %4d alerts  %5.1f%% of budget" % (
                r["confounder"], r["n_false_positives"], 100 * r["share_of_alert_budget"]))
        print()
        print("insider_drift band distribution (target: MEDIUM, not HIGH)")
        for lab, dist in bands["bands"].items():
            print("  %-16s LOW %.2f  MEDIUM %.2f  HIGH %.2f"
                  % (lab, dist["LOW"], dist["MEDIUM"], dist["HIGH"]))
        print("  targets met: %s" % bands["targets_met"])
        print()
        print("elapsed %.1fs   held-out=%s" % (out["seconds"], out["held_out"]))

    if explain_n:
        print()
        print("=" * 74)
        print("L6 EXPLANATION -- top %d entity-level alerts as an analyst sees them" % explain_n)
        print("=" * 74)
        lab = dict(zip(ev_labels["event_id"], ev_labels["label"]))
        al = top_alerts_with_explanations(det, scored, ev_sig, ev_events, zmap,
                                          "entity", explain_n)
        asg = ev.group_alerts(scored["entity"], "entity")[1]
        a2e = {}
        for e, a in zip(asg["event_id"].to_numpy(), asg["alert_id"].to_numpy()):
            a2e.setdefault(a, []).append(e)
        for a in al:
            truth = {lab.get(m, NORMAL) for m in a2e.get(a["alert_id"], [])} - {NORMAL}
            print()
            print("[%s] %-14s score %.3f  %d events  %s%s" % (
                a["band"], a["scope_key"], a["score"], a["n_events"],
                str(a["start_ts"])[:16],
                "  LOW CONFIDENCE - COLD START" if a["cold_start"] else ""))
            for w in a["why"]:
                print("     - %s   [z=%.1f]" % (w["text"], w["z"]))
            print("     ground truth: %s" % (", ".join(sorted(truth)) if truth else "benign"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the detection pipeline end to end.")
    ap.add_argument("--fit", default="seed1_delta05")
    ap.add_argument("--eval", dest="eval_tag", default="seed3_delta05")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--explain", type=int, default=0,
                    help="render the top N alerts with L6 explanations")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="unsupervised-only ablation: skip the 1-D calibration map")
    args = ap.parse_args()
    run(args.fit, args.eval_tag, args.data, calibrate=not args.no_calibrate,
        explain_n=args.explain)


if __name__ == "__main__":
    main()

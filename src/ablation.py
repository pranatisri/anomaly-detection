"""Layer ablation: does the layered architecture actually earn its keep?

Section 3 of the report asserts that L0-L3 each contribute. This measures it. Each layer
is disabled by removing its signals from the fusion and refitting everything downstream --
reference distributions, weights, correlation correction, calibration and threshold -- so
the comparison is against a detector genuinely built without that layer, not one that has
the layer and ignores it at scoring time.

The headline metric is Precision@1%, because it is the one that is neither budget-bounded
(incident precision) nor confounded by prevalence (incident recall).
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import evaluate as ev
from config import ATTACK_TYPES, LEVELS, NORMAL, assert_feature_frame
from features import SIGNAL_BY_NAME, extract

# Which named signals belong to which architectural layer. `burst_ratio` sits with L0:
# it is a deterministic rate comparison, not a learned baseline.
LAYERS: Dict[str, Tuple[str, ...]] = {
    "L0 deterministic rules": (
        "geo_velocity", "fail_rate_entity", "fail_rate_ip", "fingerprint_mismatch",
        "ip_fail_ratio", "burst_ratio",
    ),
    "L1 per-entity baseline": (
        "hour_surprisal", "ip_novelty", "country_novelty", "resource_surprisal",
        "duration_z", "ncmd_z", "peer_incongruence", "auth_method_novelty",
    ),
    "L2 sequence": ("cmd_surprisal",),
    "L3 graph / long window": (
        "ip_entity_fanout", "new_resource_rate_7d", "offhours_rate_7d",
        "breadth_ratio_7d", "uncorroborated_new_edges_7d", "corroboration_7d",
    ),
}


def _score_with(disabled: Sequence[str], fit_sig, fit_events, fit_labels,
                ev_sig, ev_events, ev_labels, ev_camps) -> Dict[str, float]:
    """Fit and score a detector built WITHOUT the named signals."""
    import detector as D

    saved = {lv: (list(D.Fusion(lv).signals), list(D.Fusion(lv).mitigators))
             for lv in LEVELS}
    orig_level_signals = {lv: list(D.LEVEL_SIGNALS[lv]) for lv in LEVELS}
    try:
        drop = set(disabled)
        for lv in LEVELS:
            D.LEVEL_SIGNALS[lv] = [s for s in orig_level_signals[lv] if s not in drop]
        if not any(D.LEVEL_SIGNALS[lv] for lv in LEVELS):
            return {}

        from pipeline import score_dataset

        y = (fit_sig[["event_id"]].merge(fit_labels[["event_id", "label"]],
                                         on="event_id", how="left")["label"]
             .fillna(NORMAL).isin(ATTACK_TYPES).to_numpy().astype(int))
        cohort = (fit_sig[["event_id"]].merge(fit_events[["event_id", "entity_type"]],
                                              on="event_id", how="left")["entity_type"]
                  .fillna("?").to_numpy())
        det = D.Detector().fit(fit_sig, y, cohort=cohort)
        scored, combined, _ = score_dataset(det, ev_events, ev_sig)

        em = ev.event_level_metrics(combined[["event_id", "score"]], ev_labels)
        ts = pd.to_datetime(ev_events["timestamp"])
        n_days = max(1.0, (ts.max() - ts.min()).total_seconds() / 86400.0)
        n_ent = int(ev_events["entity_id"].nunique())
        budgets = {"entity": max(2, int(round(0.03 * n_ent))),
                   "ip": max(1, int(round(0.01 * n_ent))),
                   "long": max(1, int(round(0.01 * n_ent)))}
        alerts, assigns = {}, {}
        for lv in LEVELS:
            a, g = ev.group_alerts(scored[lv], lv)
            alerts[lv], assigns[lv] = a, g
        im = ev.incident_metrics(alerts, assigns, ev_labels, ev_camps, n_days=n_days,
                                 budgets=budgets,
                                 event_times=ev_events[["event_id", "timestamp"]])
        out = {
            "precision_at_1pct": em["precision_at_budget"],
            "r_precision": em["r_precision"],
            "pr_auc": em["pr_auc"],
            "incident_recall": im["incident_recall_at_k"],
        }
        for k, v in im["per_type"].items():
            out["recall_%s" % k] = v["recall"]
        return out
    finally:
        for lv in LEVELS:
            D.LEVEL_SIGNALS[lv] = orig_level_signals[lv]


def run(fit_tag: str, eval_tag: str) -> pd.DataFrame:
    from pipeline import DATA, load

    fit_events, fit_labels, _ = load(fit_tag, DATA)
    assert_feature_frame(fit_events, "ablation(fit)")
    fit_sig, _ = extract(fit_events)
    ev_events, ev_labels, ev_camps = load(eval_tag, DATA)
    assert_feature_frame(ev_events, "ablation(eval)")
    ev_sig, _ = extract(ev_events)

    rows = []
    full = _score_with([], fit_sig, fit_events, fit_labels, ev_sig, ev_events,
                       ev_labels, ev_camps)
    full["config"] = "full (all layers)"
    full["removed"] = "-"
    rows.append(full)
    print("  full model              Precision@1%% %.3f" % full["precision_at_1pct"])

    for name, sigs in LAYERS.items():
        r = _score_with(sigs, fit_sig, fit_events, fit_labels, ev_sig, ev_events,
                        ev_labels, ev_camps)
        if not r:
            continue
        r["config"] = "without %s" % name
        r["removed"] = "%d signals" % len(sigs)
        rows.append(r)
        print("  without %-22s Precision@1%% %.3f  (%+.3f)"
              % (name, r["precision_at_1pct"],
                 r["precision_at_1pct"] - full["precision_at_1pct"]))

    df = pd.DataFrame(rows)
    base = df.iloc[0]
    for m in ("precision_at_1pct", "r_precision", "pr_auc", "incident_recall"):
        df["d_" + m] = df[m] - base[m]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer ablation.")
    ap.add_argument("--fit", default="seed1_delta05")
    ap.add_argument("--eval", dest="eval_tag", default="seed3_delta05")
    args = ap.parse_args()

    print("Ablating layers (each refits the whole downstream pipeline)...")
    df = run(args.fit, args.eval_tag)

    import os
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "ablation.csv")
    df.to_csv(os.path.abspath(out), index=False)

    print()
    print("=" * 86)
    print("LAYER ABLATION   fit=%s  eval=%s" % (args.fit, args.eval_tag))
    print("=" * 86)
    cols = ["config", "precision_at_1pct", "r_precision", "pr_auc", "incident_recall"]
    print(df[cols].to_string(index=False, float_format=lambda v: "%.3f" % v))
    print()
    print("Change vs full model:")
    for _, r in df.iloc[1:].iterrows():
        print("  %-32s Precision@1%% %+.3f   PR-AUC %+.3f   R-Prec %+.3f"
              % (r["config"], r["d_precision_at_1pct"], r["d_pr_auc"], r["d_r_precision"]))
    print()
    tc = [c for c in df.columns if c.startswith("recall_")]
    if tc:
        print("Per-attack-type campaign recall by configuration:")
        show = df[["config"] + sorted(tc)].rename(
            columns=lambda c: c.replace("recall_", ""))
        print(show.to_string(index=False, float_format=lambda v: "%.2f" % v))
    print("=" * 86)


if __name__ == "__main__":
    main()

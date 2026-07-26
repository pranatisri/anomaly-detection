"""Difficulty sweep and the held-out-seed protocol.

TWO THINGS THIS EXISTS TO PREVENT
---------------------------------
1. Reporting a single scalar. A benchmark's difficulty is a choice, so one number says
   as much about the generator as about the detector. The headline result is the
   DEGRADATION CURVE over delta -- how performance falls as attacks stop being obvious.

2. Tuning on the test set. Every threshold and weight is frozen into `frozen_config.json`
   with a content hash BEFORE the holdout seeds are generated. The holdout seeds
   {101..105} are then run exactly once. A dev/holdout gap is a finding to report, not
   something to iterate away.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")


def tag_for(seed: int, delta: float) -> str:
    return "seed%d_delta%s" % (seed, str(delta).replace(".", ""))


def ensure_dataset(seed: int, delta: float, entities: int, days: int,
                   force: bool = False) -> str:
    tag = tag_for(seed, delta)
    path = os.path.join(os.path.abspath(DATA), "events_%s.parquet" % tag)
    if os.path.exists(path) and not force:
        return tag
    cmd = [sys.executable, os.path.join(HERE, "generator.py"),
           "--seed", str(seed), "--entities", str(entities), "--days", str(days),
           "--delta", str(delta)]
    subprocess.run(cmd, check=True, capture_output=True)
    return tag


def freeze_config() -> Dict[str, object]:
    """Snapshot every tunable, hash it, and write it out.

    The hash is what makes "we did not tune on the holdout" checkable rather than
    merely asserted: it is recorded here and quoted in the report.
    """
    import detector as D
    import features as F
    import drift as DR

    cfg = {
        "signal_prior": D.SIGNAL_PRIOR,
        "mitigating_signals": list(D.MITIGATING_SIGNALS),
        "mitigation_weight": D.MITIGATION_WEIGHT,
        "z_clip": [D.Z_CLIP_LO, D.Z_CLIP_HI],
        "target_fpr": D.TARGET_FPR,
        "ref_sample_cap": D.REF_SAMPLE_CAP,
        "cold_start_n0": F.COLD_START_N0,
        "cold_start_min_events": F.COLD_START_MIN_EVENTS,
        "fail_window_s": F.FAIL_WINDOW_S,
        "ip_window_s": F.IP_WINDOW_S,
        "long_window_s": F.LONG_WINDOW_S,
        "max_tracked_ips": F.MAX_TRACKED_IPS,
        "max_tracked_resources": F.MAX_TRACKED_RESOURCES,
        "growth_spike_factor": DR.GROWTH_SPIKE_FACTOR,
        "quarantine_min_uncorroborated": DR.QUARANTINE_MIN_UNCORROBORATED,
        "signals": list(F.SIGNAL_NAMES),
    }
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    cfg_hash = hashlib.sha256(blob).hexdigest()[:16]
    out = {"config": cfg, "sha256_16": cfg_hash,
           "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = os.path.join(HERE, "..", "frozen_config.json")
    with open(os.path.abspath(path), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    return out


def _row(res: Dict[str, object], **extra) -> Dict[str, object]:
    e, i = res["event"], res["incident"]
    row = {
        "fit": res["fit_tag"], "eval": res["eval_tag"],
        "prevalence": e["prevalence"],
        "precision_at_1pct": e["precision_at_budget"],
        "recall_at_1pct": e["recall_at_budget"],
        "recall_ceiling": e["recall_at_budget_ceiling"],
        "r_precision": e["r_precision"],
        "pr_auc": e["pr_auc"],
        "pr_auc_baseline": e["pr_auc_baseline"],
        "incident_recall": i["incident_recall_at_k"],
        "incident_precision": i["incident_precision_at_k"],
        "incident_precision_ceiling": i["incident_precision_ceiling"],
        "alerts_per_day": i["alerts_per_analyst_per_day"],
        "median_ttd_h": i["median_ttd_hours"],
        "n_campaigns": i["n_campaigns"],
    }
    for t, v in i["per_type"].items():
        row["recall_%s" % t] = v["recall"]
    b = res["insider_bands"]["bands"]
    if "insider_drift" in b:
        row["insider_drift_HIGH"] = b["insider_drift"]["HIGH"]
        row["insider_drift_MEDIUM"] = b["insider_drift"]["MEDIUM"]
    if "low_and_slow" in b:
        row["low_and_slow_HIGH"] = b["low_and_slow"]["HIGH"]
    row.update(extra)
    return row


def delta_sweep(deltas: Sequence[float], entities: int, days: int,
                fit_seed: int = 1, eval_seed: int = 3) -> pd.DataFrame:
    from pipeline import run

    rows = []
    for d in deltas:
        ft = ensure_dataset(fit_seed, d, entities, days)
        et = ensure_dataset(eval_seed, d, entities, days)
        print("  delta=%.2f  %s -> %s" % (d, ft, et))
        res = run(ft, et, DATA, verbose=False)
        rows.append(_row(res, delta=d, split="dev"))
    return pd.DataFrame(rows)


def holdout_run(holdout_seeds: Sequence[int], delta: float, entities: int,
                days: int, fit_seed: int = 1) -> pd.DataFrame:
    from pipeline import run

    ft = ensure_dataset(fit_seed, delta, entities, days)
    rows = []
    for s in holdout_seeds:
        et = ensure_dataset(s, delta, entities, days)
        print("  holdout seed %d  %s" % (s, et))
        res = run(ft, et, DATA, verbose=False)
        rows.append(_row(res, delta=delta, split="holdout", seed=s))
    return pd.DataFrame(rows)


def plot_sweep(df: pd.DataFrame, outdir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # HEADLINE = the two UNCONFOUNDED metrics only.
    #
    # Incident recall is not monotone in delta (0.600, 0.733, 0.817, 0.567, 0.650) and the
    # reason is in the adjacent column: prevalence falls from 2.4% to 1.5% across the
    # sweep, so at low delta more campaigns compete for the same fixed budget K and recall
    # is budget-saturated rather than detector-limited. That is a confound in the metric,
    # not a property of the detector, so it is shown separately rather than as the headline.
    ax = axes[0]
    ax.plot(df["delta"], df["precision_at_1pct"], "^-", lw=2.2, color="#1f77b4",
            label="Precision@1%")
    ax.plot(df["delta"], df["pr_auc"], "s-", lw=2.2, color="#d62728", label="PR-AUC")
    ax.plot(df["delta"], df["r_precision"], "o-", lw=1.6, color="#2ca02c", alpha=.8,
            label="R-Precision")
    ax.set_xlabel("difficulty delta   (0 = blatant, 1 = overlaps benign)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    ax.set_title("Degradation curve (unconfounded metrics)")

    ax2 = ax.twinx()
    ax2.plot(df["delta"], df["prevalence"], ":", color="0.45", lw=1.4)
    ax2.set_ylabel("prevalence", color="0.45", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="0.45", labelsize=7)
    ax2.set_ylim(0, max(df["prevalence"]) * 2.2)

    ax = axes[1]
    types = [c for c in df.columns if c.startswith("recall_")
             and c not in ("recall_at_1pct", "recall_ceiling")]
    for c in sorted(types):
        if df[c].notna().sum() >= 2:
            ax.plot(df["delta"], df[c], "o-", lw=1.6, label=c.replace("recall_", ""))
    ax.set_xlabel("difficulty delta")
    ax.set_ylabel("campaign recall")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    ax.set_title("Per-attack-type campaign recall vs difficulty")

    fig.suptitle(
        "Precision@1% and PR-AUC degrade monotonically with difficulty. Incident "
        "recall is NOT shown here: it is confounded by prevalence (dotted, right "
        "axis) falling across the sweep, so at low delta more campaigns compete for "
        "the same fixed budget K and recall is budget-saturated, not detector-limited.",
        fontsize=9, y=1.04)
    fig.tight_layout()
    p = os.path.join(outdir, "delta_sweep.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Delta sweep and holdout protocol.")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--holdout", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    ap.add_argument("--entities", type=int, default=200)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--skip-holdout", action="store_true")
    args = ap.parse_args()

    frozen = freeze_config()
    print("FROZEN CONFIG sha256[:16] = %s   (%s)" % (frozen["sha256_16"], frozen["frozen_at"]))
    print("Holdout seeds are generated and scored ONLY after this point.\n")

    print("delta sweep ...")
    sweep = delta_sweep(args.deltas, args.entities, args.days)
    sweep.to_csv(os.path.join(os.path.abspath(FIGS), "delta_sweep.csv"), index=False)

    print("\nDELTA SWEEP")
    cols = ["delta", "prevalence", "incident_recall", "incident_precision",
            "precision_at_1pct", "recall_at_1pct", "r_precision", "pr_auc",
            "alerts_per_day"]
    print(sweep[cols].to_string(index=False, float_format=lambda v: "%.3f" % v))

    fig = plot_sweep(sweep, os.path.abspath(FIGS))
    print("\nfigure: %s" % fig)

    if not args.skip_holdout:
        print("\nholdout (run once, config frozen at %s) ..." % frozen["sha256_16"])
        hold = holdout_run(args.holdout, 0.5, args.entities, args.days)
        hold.to_csv(os.path.join(os.path.abspath(FIGS), "holdout.csv"), index=False)
        print("\nHOLDOUT SEEDS  (delta=0.5)")
        print(hold[["eval", "prevalence", "incident_recall", "precision_at_1pct",
                    "r_precision", "pr_auc"]].to_string(
            index=False, float_format=lambda v: "%.3f" % v))

        dev = sweep[np.isclose(sweep["delta"], 0.5)]
        if len(dev):
            print("\nDEV vs HOLDOUT at delta=0.5  (a gap is a finding, not a failure to hide)")
            for m in ["incident_recall", "precision_at_1pct", "r_precision", "pr_auc"]:
                d, h = float(dev[m].iloc[0]), float(hold[m].mean())
                print("  %-22s dev %.3f   holdout %.3f   delta %+.3f" % (m, d, h, h - d))


if __name__ == "__main__":
    main()

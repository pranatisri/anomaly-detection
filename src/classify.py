"""L5 anomaly-type classifier and the L6 explanation renderer.

L5 sees ONLY the evidence vector -- the z-scores of the named L0-L3 signals -- never raw
features. That restriction matters because L5 is the most inherently circular component
in the system: seven attack types produced by roughly seven generator knobs, each wired
to roughly one signal. Given raw fields it would learn the generator's wiring diagram and
post ~0.95 accuracy that means nothing.

So the headline number here is deliberately NOT accuracy. It is the gap between the
7-class and the 6-class macro-F1, where the 6-class variant merges `low_and_slow` and
`insider_drift` into `gradual_footprint_expansion`. That gap measures exactly how much of
the classifier's confusion is the one genuinely ambiguous pair -- which is a more honest
and more useful result than a high score.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import evaluate as ev
from config import ATTACK_TYPES, INSIDER_DRIFT, LEVELS, NORMAL
from detector import Detector, explain
from features import SIGNAL_BY_NAME

MERGED_CLASS = "gradual_footprint_expansion"
AMBIGUOUS_PAIR = ("low_and_slow", INSIDER_DRIFT)


def build_evidence(det: Detector, sig: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Concatenate the per-level z-vectors into one evidence matrix.

    These are named quantities with units, not learned embeddings, which is what keeps
    the classifier's inputs auditable.
    """
    blocks, names = [], []
    for lv in LEVELS:
        fu = det.fusions[lv]
        cols = fu.signals + fu.mitigators
        blocks.append(fu.to_z(sig, cols))
        names.extend("%s.%s" % (lv, c) for c in cols)
    return np.column_stack(blocks), names


def _fit_predict(Xtr, ytr, Xte, seed: int = 0):
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.07,
                               class_weight="balanced", random_state=seed, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    m.fit(Xtr, ytr)
    return m, m.predict(Xte)


def _report(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, object]:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return {
        "labels": list(labels),
        "confusion": cm,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(labels),
                                   zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=list(labels),
                                        zero_division=0, digits=3),
    }


def _merge(labels: Sequence[str]) -> List[str]:
    return [MERGED_CLASS if x in AMBIGUOUS_PAIR else x for x in labels]


def format_confusion(res: Dict[str, object]) -> str:
    labs = res["labels"]
    cm = res["confusion"]
    w = max(len(x) for x in labs) + 1
    head = " " * (w + 2) + " ".join("%6s" % x[:6] for x in labs)
    lines = [head]
    for i, lab in enumerate(labs):
        lines.append("%-*s  %s" % (w, lab, " ".join("%6d" % v for v in cm[i])))
    return "\n".join(lines)


def run(fit_tag: str, eval_tag: str, data_dir: Optional[str] = None,
        detected_only: bool = True) -> Dict[str, object]:
    from pipeline import DATA, load, score_dataset
    from features import extract
    from config import assert_feature_frame

    data_dir = data_dir or DATA
    fit_events, fit_labels, _ = load(fit_tag, data_dir)
    assert_feature_frame(fit_events, "classifier(fit)")
    fit_sig, _ = extract(fit_events)

    y_fit_bin = (fit_sig[["event_id"]].merge(fit_labels[["event_id", "label"]],
                                             on="event_id", how="left")["label"]
                 .fillna(NORMAL).isin(ATTACK_TYPES).to_numpy().astype(int))
    det = Detector().fit(fit_sig, y_fit_bin)

    ev_events, ev_labels, ev_camps = load(eval_tag, data_dir)
    assert_feature_frame(ev_events, "classifier(eval)")
    ev_sig, _ = extract(ev_events)

    Xtr, names = build_evidence(det, fit_sig)
    Xte, _ = build_evidence(det, ev_sig)

    ytr = (fit_sig[["event_id"]].merge(fit_labels[["event_id", "label"]],
                                       on="event_id", how="left")["label"]
           .fillna(NORMAL).to_numpy())
    yte = (ev_sig[["event_id"]].merge(ev_labels[["event_id", "label"]],
                                      on="event_id", how="left")["label"]
           .fillna(NORMAL).to_numpy())

    # Train only on anomalous events: L5's job is "which KIND", not "is it anomalous".
    tr_mask = ytr != NORMAL
    te_mask = yte != NORMAL

    if detected_only:
        # "Detected" means the events of campaigns the detector actually surfaced --
        # NOT the top-1% of events by score.
        #
        # Using the raw top-1% makes the evaluation set 86% credential stuffing and
        # leaves lateral_movement and impossible_travel with zero test rows, because
        # burst attacks emit hundreds of events while impossible travel emits three.
        # That is the same density skew that makes event-level recall gameable, and it
        # would make this confusion matrix a report on one attack type.
        scored, combined, _ = score_dataset(det, ev_events, ev_sig)
        ts = pd.to_datetime(ev_events["timestamp"])
        n_days = max(1.0, (ts.max() - ts.min()).total_seconds() / 86400.0)
        n_entities = int(ev_events["entity_id"].nunique())
        budgets = {"entity": max(2, int(round(0.03 * n_entities))),
                   "ip": max(1, int(round(0.01 * n_entities))),
                   "long": max(1, int(round(0.01 * n_entities)))}
        alerts, assigns = {}, {}
        for lv in LEVELS:
            a, asg = ev.group_alerts(scored[lv], lv)
            alerts[lv], assigns[lv] = a, asg
        inc = ev.incident_metrics(alerts, assigns, ev_labels, ev_camps, n_days=n_days,
                                  budgets=budgets,
                                  event_times=ev_events[["event_id", "timestamp"]])
        det_camps = set()
        for lv in LEVELS:
            attrib = ev.attribute_alerts(assigns[lv], ev_labels)
            top = alerts[lv].sort_values(["alert_score", "alert_id"],
                                         ascending=[False, True]).head(
                max(1, int(round(budgets[lv] * n_days))))
            det_camps |= set(attrib[attrib["alert_id"].isin(set(top["alert_id"]))]["campaign_id"])
        cmap = ev_sig[["event_id"]].merge(ev_labels[["event_id", "campaign_id"]],
                                          on="event_id", how="left")["campaign_id"]
        te_mask = te_mask & cmap.isin(det_camps).to_numpy()

    if tr_mask.sum() < 30 or te_mask.sum() < 20:
        return {"error": "insufficient anomalous events (train=%d, test=%d)"
                % (tr_mask.sum(), te_mask.sum())}

    _, pred = _fit_predict(Xtr[tr_mask], ytr[tr_mask], Xte[te_mask])

    # Macro-average only over classes actually present in the evaluation set (as truth or
    # as a prediction). A class that is absent from both scores F1 = 0 and silently drags
    # the macro down -- and because the merged view has one fewer class, that artefact
    # lands unevenly on the two variants and inflates the very delta this reports.
    # Seeds deliberately omit whole attack types (CM8), so this is the normal case.
    absent = sorted(set(ytr[tr_mask]) - (set(yte[te_mask]) | set(pred)))
    present7 = sorted(set(yte[te_mask]) | set(pred))
    r7 = _report(yte[te_mask], pred, present7)

    ym, pm = _merge(yte[te_mask]), _merge(pred)
    present6 = sorted(set(ym) | set(pm))
    r6 = _report(ym, pm, present6)

    sep = ev.separation_auc(
        pd.DataFrame({"event_id": ev_sig["event_id"],
                      "peer_incongruence": ev_sig["peer_incongruence"],
                      "score": 0.0}),
        ev_labels, "peer_incongruence")

    return {
        "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
        "detected_only": detected_only,
        "seven": r7, "six": r6,
        "macro_f1_delta": r6["macro_f1"] - r7["macro_f1"],
        "separation_auc": sep,
        "evidence_dims": len(names),
        "absent_classes": absent,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="L5 anomaly-type classifier.")
    ap.add_argument("--fit", default="seed1_delta05")
    ap.add_argument("--eval", dest="eval_tag", default="seed3_delta05")
    ap.add_argument("--all-events", action="store_true",
                    help="score every anomalous event, not only detected ones")
    args = ap.parse_args()

    res = run(args.fit, args.eval_tag, detected_only=not args.all_events)
    if "error" in res:
        print("SKIPPED: %s" % res["error"])
        return

    print("=" * 74)
    print("L5 ANOMALY-TYPE CLASSIFICATION  (evidence vector only, %d dims)"
          % res["evidence_dims"])
    print("train %d anomalous events -> test %d %s"
          % (res["n_train"], res["n_test"],
             "(detected only)" if res["detected_only"] else "(all)"))
    print("=" * 74)
    print()
    print("7-CLASS  macro-F1 %.3f" % res["seven"]["macro_f1"])
    print(format_confusion(res["seven"]))
    print()
    print("6-CLASS  macro-F1 %.3f   (low_and_slow + insider_drift merged)"
          % res["six"]["macro_f1"])
    print(format_confusion(res["six"]))
    print()
    if res.get("absent_classes"):
        print("classes absent from this eval set (excluded from macro): %s"
              % ", ".join(res["absent_classes"]))
        print()
    print("MACRO-F1 DELTA  %+.3f" % res["macro_f1_delta"])
    print("  The share of the classifier's confusion attributable to the one")
    print("  genuinely ambiguous pair. This is the headline, not accuracy.")
    print()
    s = res["separation_auc"]
    print("low_and_slow vs insider_drift separability, best single discriminator")
    print("  peer_incongruence ROC-AUC %.3f  [%.3f, %.3f]  (n=%d)"
          % (s["auc"], s["ci_low"], s["ci_high"], s["n"]))
    print("  0.5 would mean the two are inseparable on this signal. Reported as measured.")
    print("=" * 74)


if __name__ == "__main__":
    main()

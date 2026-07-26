"""Hand-computed checks on the metric definitions.

Every expected number here was worked out by hand and is written in the assertion, so
that if the metric implementation ever drifts, the test says what the answer should have
been rather than just that something changed.

Runs standalone (`python tests/test_evaluate.py`) or under pytest.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config import INSIDER_DRIFT, LEVEL_ENTITY, LEVEL_IP, NORMAL  # noqa: E402
import evaluate as ev  # noqa: E402

T0 = pd.Timestamp("2026-01-01 00:00:00")


def _mins(m):
    return T0 + pd.Timedelta(minutes=m)


# --------------------------------------------------------------------------------------


def test_top2mean():
    assert ev.top2mean([5.0]) == 5.0                    # single value -> itself
    assert ev.top2mean([1.0, 2.0, 3.0]) == 2.5          # (3+2)/2
    assert ev.top2mean([0.1, 0.9]) == 0.5               # (0.9+0.1)/2
    assert ev.top2mean([4.0, 4.0, 0.0]) == 4.0
    assert np.isnan(ev.top2mean([]))
    # NaNs are dropped, not propagated.
    assert ev.top2mean([np.nan, 2.0, 6.0]) == 4.0


def test_group_alerts_merge_window():
    """Merge window is 60 min. A@0 and A@30 merge; A@120 is 90 min later so it splits."""
    scored = pd.DataFrame({
        "event_id": ["e1", "e2", "e3", "e4"],
        "entity_id": ["A", "A", "A", "B"],
        "timestamp": [_mins(0), _mins(30), _mins(120), _mins(0)],
        "score": [0.1, 0.9, 0.5, 0.3],
    })
    alerts, assign = ev.group_alerts(scored, LEVEL_ENTITY)

    assert len(alerts) == 3, "expected A{e1,e2}, A{e3}, B{e4}"

    by_scope = {}
    for _, r in alerts.iterrows():
        by_scope.setdefault(r["scope_key"], []).append(r)

    a_alerts = sorted(by_scope["A"], key=lambda r: r["start_ts"])
    assert a_alerts[0]["n_events"] == 2
    assert a_alerts[0]["alert_score"] == 0.5      # top2mean([0.1, 0.9])
    assert a_alerts[1]["n_events"] == 1
    assert a_alerts[1]["alert_score"] == 0.5      # single member e3
    assert by_scope["B"][0]["alert_score"] == 0.3

    # e1 and e2 share an alert id; e3 does not.
    m = dict(zip(assign["event_id"], assign["alert_id"]))
    assert m["e1"] == m["e2"]
    assert m["e3"] != m["e1"]
    assert m["e4"] != m["e1"]


def test_group_alerts_boundary_is_not_strict():
    """A gap exactly equal to the window merges; only a gap strictly greater splits."""
    scored = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "entity_id": ["A", "A"],
        "timestamp": [_mins(0), _mins(60)],
        "score": [1.0, 1.0],
    })
    alerts, _ = ev.group_alerts(scored, LEVEL_ENTITY)
    assert len(alerts) == 1

    scored.loc[1, "timestamp"] = _mins(61)
    alerts, _ = ev.group_alerts(scored, LEVEL_ENTITY)
    assert len(alerts) == 2


def test_event_level_metrics_hand_computed():
    """100 events, scores strictly descending by index, attacks at e001, e005, e050.

    budget_fraction = 0.10  ->  k = 10, so S = {e000 .. e009}
      |S n A| = 2 (e001, e005)      -> Precision@10% = 2/10 = 0.2
      |A|     = 3                   -> Recall@10%    = 2/3
      prevalence = 3/100 = 0.03     -> ceiling = min(1, 0.10/0.03) = 1.0
      R-Precision: k = |A| = 3, S = {e000, e001, e002}, 1 attack -> 1/3
    """
    n = 100
    ids = ["e%03d" % i for i in range(n)]
    scored = pd.DataFrame({
        "event_id": ids,
        "score": [1.0 - i / n for i in range(n)],
    })
    labels = pd.DataFrame({"event_id": ids, "label": [NORMAL] * n})
    for i in (1, 5, 50):
        labels.loc[i, "label"] = "brute_force"

    m = ev.event_level_metrics(scored, labels, budget_fraction=0.10)

    assert m["k"] == 10
    assert abs(m["precision_at_budget"] - 0.2) < 1e-12
    assert abs(m["recall_at_budget"] - 2.0 / 3.0) < 1e-12
    assert abs(m["prevalence"] - 0.03) < 1e-12
    assert abs(m["recall_at_budget_ceiling"] - 1.0) < 1e-12
    assert abs(m["r_precision"] - 1.0 / 3.0) < 1e-12
    assert abs(m["pr_auc_baseline"] - 0.03) < 1e-12


def test_recall_ceiling_artefact_is_reported():
    """At 3% prevalence with a 1% budget, Recall@1% cannot exceed 1/3.

    This is the artefact that makes a raw Recall@1% number misleading on its own. A
    PERFECT detector -- every attack ranked above every benign event -- still tops out
    at the ceiling, so the test asserts the ceiling is both reported and actually binding.
    """
    n = 1000
    ids = ["e%04d" % i for i in range(n)]
    labels = pd.DataFrame({"event_id": ids, "label": [NORMAL] * n})
    labels.loc[:29, "label"] = "brute_force"          # 30 attacks -> prevalence 3%

    # Perfect ranking: all 30 attacks score above every benign event.
    scored = pd.DataFrame({
        "event_id": ids,
        "score": [1.0] * 30 + [0.0] * (n - 30),
    })
    m = ev.event_level_metrics(scored, labels, budget_fraction=0.01)

    assert m["k"] == 10
    assert abs(m["prevalence"] - 0.03) < 1e-12
    assert abs(m["recall_at_budget_ceiling"] - 1.0 / 3.0) < 1e-12
    # Even a perfect detector only reaches the ceiling, never 1.0.
    assert abs(m["recall_at_budget"] - 10.0 / 30.0) < 1e-12
    # R-Precision has no such ceiling and correctly reports perfection.
    assert abs(m["r_precision"] - 1.0) < 1e-12


def test_insider_drift_is_not_a_positive():
    """insider_drift stays in the scoring universe but is never counted as an attack."""
    ids = ["e%02d" % i for i in range(10)]
    scored = pd.DataFrame({"event_id": ids, "score": [1.0 - i / 10 for i in range(10)]})
    labels = pd.DataFrame({"event_id": ids, "label": [NORMAL] * 10})
    labels.loc[0, "label"] = INSIDER_DRIFT      # ranked top
    labels.loc[1, "label"] = "lateral_movement"

    m = ev.event_level_metrics(scored, labels, budget_fraction=0.20)   # k = 2

    assert m["n_attack_events"] == 1.0                 # insider_drift excluded from A
    assert m["n_insider_drift_events"] == 1.0
    assert m["n_events"] == 10.0                       # but still in the universe N
    assert abs(m["precision_at_budget"] - 0.5) < 1e-12  # 1 of the 2 reviewed is a real attack
    assert abs(m["recall_at_budget"] - 1.0) < 1e-12
    # It consumed half the analyst's budget, and that cost is reported.
    assert abs(m["insider_drift_share_of_budget"] - 0.5) < 1e-12


def test_attribute_alerts_majority_and_tiebreak():
    """An alert is attributed to whichever campaign contributed most of its events."""
    assign = pd.DataFrame({
        "event_id": ["a", "b", "c", "d"],
        "alert_id": ["entity:1"] * 4,
        "level": LEVEL_ENTITY,
    })
    labels = pd.DataFrame({
        "event_id": ["a", "b", "c", "d"],
        "label": ["brute_force", "brute_force", "lateral_movement", NORMAL],
        "campaign_id": ["C1", "C1", "C2", None],
    })
    got = ev.attribute_alerts(assign, labels)
    assert len(got) == 1
    assert got.iloc[0]["campaign_id"] == "C1", got.iloc[0].to_dict()   # 2 events beats 1
    assert got.iloc[0]["n_member_events"] == 2

    # Majority wins regardless of ordering: make C2 the larger contributor.
    labels.loc[1, "label"] = "lateral_movement"
    labels.loc[1, "campaign_id"] = "C2"                # now C1:1 (a), C2:2 (b, c)
    got = ev.attribute_alerts(assign, labels)
    assert got.iloc[0]["campaign_id"] == "C2", got.iloc[0].to_dict()
    assert got.iloc[0]["n_member_events"] == 2

    # Genuine 1-1 tie -> lexicographically first campaign_id, deterministically.
    labels.loc[2, "label"] = NORMAL                    # now C1:1 (a), C2:1 (b)
    labels.loc[2, "campaign_id"] = None
    got = ev.attribute_alerts(assign, labels)
    assert got.iloc[0]["campaign_id"] == "C1", got.iloc[0].to_dict()
    assert got.iloc[0]["n_member_events"] == 1

    # An alert with no attack members is attributed to nothing.
    labels["label"] = NORMAL
    labels["campaign_id"] = None
    assert ev.attribute_alerts(assign, labels).empty


def test_incident_recall_counts_each_campaign_once():
    """The density-gaming fix: a 50-event campaign and a 1-event campaign each count 1.

    C1 (brute_force) emits 50 events on entity A inside one merge window -> 1 alert.
    C2 (impossible_travel) emits 1 event on entity B                     -> 1 alert.
    Both are reviewed, so incident recall is 2/2 = 1.0 even though C1 outweighs C2
    50:1 at the event level.
    """
    rows = []
    for i in range(50):
        rows.append(("c1_%02d" % i, "A", _mins(i % 30), 0.9, "brute_force", "C1"))
    rows.append(("c2_00", "B", _mins(0), 0.8, "impossible_travel", "C2"))
    for i in range(200):
        rows.append(("n_%03d" % i, "N%d" % i, _mins(0), 0.1, NORMAL, None))

    scored = pd.DataFrame(rows, columns=["event_id", "entity_id", "timestamp", "score", "label", "campaign_id"])
    labels = scored[["event_id", "label", "campaign_id"]].copy()
    scored = scored[["event_id", "entity_id", "timestamp", "score"]]

    campaigns = pd.DataFrame({
        "campaign_id": ["C1", "C2"],
        "campaign_type": ["brute_force", "impossible_travel"],
        "start_ts": [_mins(0), _mins(0)],
        "end_ts": [_mins(29), _mins(0)],
    })

    alerts, assign = ev.group_alerts(scored, LEVEL_ENTITY)
    inc = ev.incident_metrics(
        {LEVEL_ENTITY: alerts}, {LEVEL_ENTITY: assign},
        labels, campaigns, n_days=1.0, budgets={LEVEL_ENTITY: 5},
    )

    assert inc["n_campaigns"] == 2.0
    assert inc["n_campaigns_detected"] == 2.0
    assert inc["incident_recall_at_k"] == 1.0
    assert inc["per_type"]["brute_force"]["recall"] == 1.0
    assert inc["per_type"]["impossible_travel"]["recall"] == 1.0
    # Each campaign contributes exactly 1 to the denominator.
    assert inc["per_type"]["brute_force"]["n_campaigns"] == 1.0


def test_redundant_alerts_count_as_workload_not_as_detections():
    """Two alerts on one campaign = 1 TP + 1 redundant, but 2 alerts of analyst workload."""
    rows = [
        ("a1", "A", _mins(0), 0.9, "brute_force", "C1"),
        ("a2", "A", _mins(500), 0.9, "brute_force", "C1"),   # far outside the 60 min window
    ]
    for i in range(50):
        rows.append(("n%02d" % i, "N%d" % i, _mins(0), 0.1, NORMAL, None))

    scored = pd.DataFrame(rows, columns=["event_id", "entity_id", "timestamp", "score", "label", "campaign_id"])
    labels = scored[["event_id", "label", "campaign_id"]].copy()
    scored = scored[["event_id", "entity_id", "timestamp", "score"]]

    campaigns = pd.DataFrame({
        "campaign_id": ["C1"], "campaign_type": ["brute_force"],
        "start_ts": [_mins(0)], "end_ts": [_mins(500)],
    })

    alerts, assign = ev.group_alerts(scored, LEVEL_ENTITY)
    inc = ev.incident_metrics(
        {LEVEL_ENTITY: alerts}, {LEVEL_ENTITY: assign},
        labels, campaigns, n_days=1.0, budgets={LEVEL_ENTITY: 4},
    )

    assert inc["n_campaigns_detected"] == 1.0        # detection quality: lenient
    assert inc["redundant_alerts"] == 1.0
    assert inc["n_reviewed_alerts"] == 4.0           # workload: strict, budget fully spent
    assert inc["alerts_per_analyst_per_day"] == 4.0
    assert inc["incident_precision_at_k"] == 1.0 / 4.0


def test_time_to_detect_penalises_late_detection():
    """hit@1 alone would score a 20-day-late catch as a win; TTD is what exposes it.

    The campaign's early events score BELOW the benign baseline, so they never enter the
    reviewed set -- exactly the low_and_slow failure mode. Only the final event is caught.
    """
    rows = [("early", "A", _mins(0), 0.01, "low_and_slow", "C1"),
            ("late", "A", _mins(20 * 24 * 60), 0.99, "low_and_slow", "C1")]
    for i in range(50):
        rows.append(("n%02d" % i, "N%d" % i, _mins(0), 0.05, NORMAL, None))

    scored = pd.DataFrame(rows, columns=["event_id", "entity_id", "timestamp", "score", "label", "campaign_id"])
    labels = scored[["event_id", "label", "campaign_id"]].copy()
    scored = scored[["event_id", "entity_id", "timestamp", "score"]]

    campaigns = pd.DataFrame({
        "campaign_id": ["C1"], "campaign_type": ["low_and_slow"],
        "start_ts": [_mins(0)], "end_ts": [_mins(20 * 24 * 60)],
    })

    alerts, assign = ev.group_alerts(scored, LEVEL_ENTITY)
    inc = ev.incident_metrics(
        {LEVEL_ENTITY: alerts}, {LEVEL_ENTITY: assign},
        labels, campaigns, n_days=20.0, budgets={LEVEL_ENTITY: 1},
    )

    assert inc["incident_recall_at_k"] == 1.0                 # counted as detected...
    assert abs(inc["median_ttd_hours"] - 480.0) < 1e-6        # ...but 20 days late
    assert inc["frac_detected_early"] == 0.0                  # past the campaign midpoint


def test_fp_breakdown_attributes_cost_to_confounders():
    """False positives are broken down by which engineered benign behaviour caused them."""
    ids = ["e%02d" % i for i in range(20)]
    scored = pd.DataFrame({"event_id": ids, "score": [1.0 - i / 20 for i in range(20)]})
    labels = pd.DataFrame({
        "event_id": ids,
        "label": [NORMAL] * 20,
        "confounder": [None] * 20,
    })
    labels.loc[0, "confounder"] = "business_travel"
    labels.loc[1, "confounder"] = "business_travel"
    labels.loc[2, "confounder"] = "os_patch"
    labels.loc[3, "label"] = "impossible_travel"      # a genuine attack, not an FP

    out = ev.fp_breakdown_by_confounder(scored, labels, budget_fraction=0.25)  # k = 5

    row = out[out["confounder"] == "business_travel"].iloc[0]
    assert row["n_false_positives"] == 2
    assert abs(row["share_of_alert_budget"] - 2.0 / 5.0) < 1e-12
    assert out[out["confounder"] == "os_patch"].iloc[0]["n_false_positives"] == 1
    # The real attack in the top-k is not counted as a false positive.
    assert out["n_false_positives"].sum() == 4


def test_insider_drift_band_targets():
    """The edge case's objective is MEDIUM, not recall."""
    ids = ["e%02d" % i for i in range(20)]
    # theta = 0.8, so HIGH >= 0.8, MEDIUM in [0.4, 0.8), LOW < 0.4.
    scores = [0.5] * 10 + [0.9] * 10
    labels = pd.DataFrame({
        "event_id": ids,
        "label": [INSIDER_DRIFT] * 10 + ["low_and_slow"] * 10,
    })
    scored = pd.DataFrame({"event_id": ids, "score": scores})

    rep = ev.insider_drift_band_report(scored, labels, theta=0.8)

    assert rep["bands"][INSIDER_DRIFT]["MEDIUM"] == 1.0
    assert rep["bands"][INSIDER_DRIFT]["HIGH"] == 0.0
    assert rep["bands"]["low_and_slow"]["HIGH"] == 1.0
    assert all(rep["targets_met"].values())


def test_ip_level_grouping_catches_credential_stuffing_shape():
    """Many entities, one IP -- invisible per-entity, obvious at the IP level.

    Ten different entities each emit a single event from one shared IP. At entity level
    that is ten unrelated singleton alerts; at IP level it is one alert with ten members.
    This is why the levels are scored separately rather than pooled.
    """
    rows = [("s%02d" % i, "user%02d" % i, "10.0.0.9", _mins(i), 0.6) for i in range(10)]
    scored = pd.DataFrame(rows, columns=["event_id", "entity_id", "source_ip", "timestamp", "score"])

    ent_alerts, _ = ev.group_alerts(scored, LEVEL_ENTITY)
    ip_alerts, _ = ev.group_alerts(scored, LEVEL_IP)

    assert len(ent_alerts) == 10
    assert len(ip_alerts) == 1
    assert ip_alerts.iloc[0]["n_events"] == 10


# --------------------------------------------------------------------------------------


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS  %s" % t.__name__)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL  %s\n        %s: %s" % (t.__name__, type(exc).__name__, exc))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

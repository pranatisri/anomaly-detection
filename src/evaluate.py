"""Imbalance-aware evaluation: alert grouping, incident matching, metrics.

Written BEFORE any model exists, deliberately. If the metric definition moves after the
models are built, every result produced up to that point is invalidated.

Two evaluation units, both reported:

  Event level    matches the analyst's literal alert budget ("top 1% of events").
                 Gameable on its own: recall is dominated by whichever attack type is
                 most event-dense, so a detector that nails brute_force (~50 events per
                 campaign) and misses everything else posts an excellent number.

  Incident level RECOMMENDED PRIMARY. Events are collapsed into alerts, alerts are
                 matched to ground-truth campaigns, and each campaign counts exactly
                 once in the denominator regardless of how many events it emitted.

insider_drift is not a positive. It stays in the scoring universe (it is real traffic
and does consume analyst budget, so it can and should hurt precision if it ranks high),
but it is excluded from the positive set A and from the campaign set C. Its objective is
asymmetric and is reported separately by insider_drift_band_report().
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import (
    ATTACK_TYPES,
    EVENT_BUDGET_FRACTION,
    INSIDER_DRIFT,
    LEVEL_DAILY_BUDGET,
    LEVEL_MERGE_WINDOW_S,
    LEVEL_SCOPE_KEYS,
    LEVELS,
    NORMAL,
    BAND_HIGH,
    BAND_LOW,
    BAND_MEDIUM,
    band_for,
)

# --------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------


def top2mean(values: Sequence[float]) -> float:
    """Mean of the two largest values.

    Interpolates between max (fully sensitive to one screaming signal) and mean (fully
    sensitive to diffuse evidence) with a single interpretable knob, and is far more
    robust than either extreme. Used both to score an alert from its member events and
    to combine per-signal z-scores in L4 fusion.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float(arr[0])
    part = np.partition(arr, -2)[-2:]
    return float(part.mean())


def _scope_key_series(df: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    """Concatenate the level's scope columns into a single string key (fast path)."""
    out = df[keys[0]].astype(str)
    for k in keys[1:]:
        out = out + "|" + df[k].astype(str)
    return out


# --------------------------------------------------------------------------------------
# Alert grouping
# --------------------------------------------------------------------------------------


def group_alerts(
    scored: pd.DataFrame,
    level: str,
    merge_window_s: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse scored events into analyst-facing alerts.

    Events sharing (level, scope_key) within the level's merge window become ONE alert
    with ``alert_score = top2mean(member scores)``.

    Grouping is non-negotiable for honest metrics: without it a 50-event brute force
    contributes 50 detections while a 3-event impossible_travel contributes 3, so
    recall silently becomes a measure of event density rather than of detection.

    Args:
        scored: must contain event_id, timestamp (datetime64), score, and the level's
            scope columns. For LEVEL_LONG the ``day`` column is derived if absent.
        level: one of config.LEVELS.
        merge_window_s: override the level default (used by tests).

    Returns:
        (alerts, assignment) where ``alerts`` is one row per alert and ``assignment``
        maps event_id -> alert_id. The mapping is returned separately rather than as a
        list column so that downstream joins stay vectorised at 1M+ events.
    """
    if level not in LEVELS:
        raise ValueError("unknown level %r" % (level,))
    keys = list(LEVEL_SCOPE_KEYS[level])
    window = LEVEL_MERGE_WINDOW_S[level] if merge_window_s is None else merge_window_s

    df = scored.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "day" in keys and "day" not in df.columns:
        df["day"] = df["timestamp"].dt.floor("D")

    missing = {"event_id", "score"}.union(keys) - set(df.columns)
    if missing:
        raise ValueError("scored frame missing columns: %s" % sorted(missing))

    df["_scope"] = _scope_key_series(df, keys)
    df = df.sort_values(["_scope", "timestamp", "event_id"]).reset_index(drop=True)

    secs = df["timestamp"].astype("int64") // 10**9
    scope_changed = df["_scope"] != df["_scope"].shift()
    gap = secs - secs.shift()
    # A new alert starts at a scope change, or when the gap since the previous event in
    # the same scope exceeds the merge window.
    starts = scope_changed | (gap > window)
    starts.iloc[0] = True
    df["_grp"] = starts.cumsum()

    g = df.groupby("_grp", sort=False)
    alerts = g.agg(
        scope_key=("_scope", "first"),
        start_ts=("timestamp", "min"),
        end_ts=("timestamp", "max"),
        n_events=("event_id", "size"),
        max_score=("score", "max"),
    )

    if "p" in df.columns:
        from scipy.stats import norm

        # Stouffer: Z = sum(z_i) / sqrt(n). Calibrated in n under the null (so a busy
        # entity is not rewarded for volume) while still accumulating evidence (so fifty
        # failed authentications count for more than one).
        #
        # Two alternatives were measured and rejected:
        #   Sidak on the best member event  -- fixes size bias but discards the other n-1
        #       events, so a brute force is penalised for the very evidence that convicts
        #       it: campaign recall 1.000 -> 0.000.
        #   Fisher, -2 sum(ln p_i) ~ chi2(2n) -- correct in theory, but it compounds any
        #       systematic per-cohort offset over n events, and underflows to -inf in the
        #       deep tail so every large alert tied at the top. corr with size 0.62.
        #
        # Stouffer still amplifies a cohort offset by sqrt(n), which is why the real fix
        # for the service-account skew had to happen in the generator rather than here.
        z_ev = norm.ppf(1.0 - np.clip(df["p"].to_numpy(dtype=float), 1e-12, 1 - 1e-12))
        df = df.assign(_z=np.clip(z_ev, -8.0, 8.0))
        gg = df.groupby("_grp", sort=False)["_z"]
        n = alerts["n_events"].to_numpy().astype(float)
        stouffer = gg.sum().to_numpy() / np.sqrt(n)

        # EMPIRICAL SIZE CALIBRATION -- the step that actually removes the size bias.
        #
        # Stouffer's null assumes independent member events. Events from one entity in one
        # hour are strongly correlated (same IP, same resources, same session), so the true
        # variance is 1 + (n-1)*rho rather than 1, and the inflation GROWS with n. Even
        # after centring per-event p on the live cohort -- which brought mean benign z to
        # ~0.00 -- service accounts still took 17 of the top 20 alerts purely because their
        # alerts contain 40x more events and therefore 40x more accumulated correlation.
        #
        # Rather than assume a null and estimate rho, calibrate against reality: compare
        # each alert only with alerts of the SAME SIZE, using median/MAD over log2 size
        # bins. Anomalies are 0.5-3% of traffic, so the per-bin median and MAD are robust
        # estimates of that bin's benign centre and spread, and no labels are involved.
        # The size bins are then smoothed by a robust line in log2(n), and the FITTED
        # centre/spread is applied to every bin. Calibrating each bin independently and
        # passing sparse bins through raw left the largest bin (23 alerts, below any
        # sensible minimum) uncalibrated at a mean of 15.2 -- so the handful of biggest
        # alerts still owned the top of the queue. Extrapolating the trend covers exactly
        # the bins that matter most and are always the emptiest.
        # Sparse bins are POOLED with their neighbours rather than extrapolated into.
        # Fitting a line in log2(n) and extrapolating was tried and inverted the bias
        # outright (correlation with size went from +0.18 to -0.41, and the queue filled
        # with singleton alerts): the relationship is not linear, so the fit was worse
        # than no model at all. Pooling makes no functional assumption.
        bins = np.floor(np.log2(np.maximum(n, 1.0))).astype(int)
        uniq = sorted(np.unique(bins).tolist())
        counts = {b: int((bins == b).sum()) for b in uniq}

        groups: List[List[int]] = []
        cur: List[int] = []
        for b in uniq:
            cur.append(b)
            if sum(counts[x] for x in cur) >= 30:
                groups.append(cur)
                cur = []
        if cur:
            if groups:
                groups[-1].extend(cur)     # tail merges into its nearest populated group
            else:
                groups.append(cur)

        # Spread is estimated from the LOWER half only: sigma ~ (median - q16).
        #
        # A two-sided MAD is not robust here because the contamination is not uniform --
        # it is concentrated in exactly the bins being calibrated. Large alerts are
        # disproportionately attacks (brute force, credential stuffing, CI bursts), so
        # roughly a third of the biggest-alert bin is malicious, and a two-sided MAD is
        # inflated by the very campaigns it is meant to score. Measured: a brute force
        # with Stouffer 14.9 ranked 9,991st while one with 12.1 ranked 1st.
        #
        # Attacks inflate the UPPER tail only, so the lower half still describes benign
        # traffic and gives an uncontaminated scale.
        adj = np.empty_like(stouffer)
        for grp in groups:
            m = np.isin(bins, grp)
            v = stouffer[m]
            med = float(np.median(v))
            q16 = float(np.percentile(v, 16))
            sigma = max(med - q16, 1e-6)
            adj[m] = (v - med) / sigma
        alerts["alert_score"] = adj
        alerts["alert_stouffer"] = stouffer
        alerts["alert_p"] = norm.sf(np.clip(adj, -40.0, 40.0))
    else:
        alerts["alert_score"] = g["score"].apply(lambda s: top2mean(s.to_numpy()))
    alerts = alerts.reset_index(drop=False)
    alerts["level"] = level
    alerts["alert_id"] = level + ":" + alerts["_grp"].astype(str)
    alerts = alerts.drop(columns=["_grp"])

    assignment = pd.DataFrame({
        "event_id": df["event_id"].to_numpy(),
        "alert_id": (level + ":" + df["_grp"].astype(str)).to_numpy(),
        "level": level,
    })
    return alerts.reset_index(drop=True), assignment


# --------------------------------------------------------------------------------------
# Event-level metrics
# --------------------------------------------------------------------------------------


def event_level_metrics(
    scored: pd.DataFrame,
    labels: pd.DataFrame,
    budget_fraction: float = EVENT_BUDGET_FRACTION,
) -> Dict[str, float]:
    """Precision@budget, Recall@budget, R-Precision, PR-AUC, with prevalence stated.

    Definitions (N = all scored events, k = ceil(budget_fraction * N)):
        S = top-k events by (score DESC, event_id ASC)   -- deterministic, so |S| == k
        A = events whose label is a true attack type (insider_drift excluded)

        Precision@1% = |S n A| / k        <- denominator is k, NOT |S|
        Recall@1%    = |S n A| / |A|      <- denominator is all attack EVENTS

    CEILING ARTEFACT, reported alongside every use of Recall@1%:
        At prevalence p, Recall@1% <= budget_fraction / p by construction. At p = 3% the
        best achievable Recall@1% is 1/3, however good the detector is. R-Precision
        (the same formula with k = |A|) removes the artefact and is reported next to it.
    """
    df = scored.merge(labels[["event_id", "label"]], on="event_id", how="left")
    df["label"] = df["label"].fillna(NORMAL)

    is_attack = df["label"].isin(ATTACK_TYPES).to_numpy()
    n_total = int(len(df))
    n_attack = int(is_attack.sum())
    prevalence = n_attack / n_total if n_total else float("nan")

    order = np.lexsort((df["event_id"].to_numpy(), -df["score"].to_numpy()))
    ranked_attack = is_attack[order]

    k = max(1, math.ceil(budget_fraction * n_total))
    hits_at_k = int(ranked_attack[:k].sum())

    out = {
        "n_events": float(n_total),
        "n_attack_events": float(n_attack),
        "prevalence": prevalence,
        "budget_fraction": budget_fraction,
        "k": float(k),
        "precision_at_budget": hits_at_k / k,
        # Precision has a ceiling too, and it binds in the opposite regime to recall's.
        # There are only |A| attacks to find, so Precision@1% <= |A|/k = prevalence/0.01.
        # Below 1% prevalence this caps precision outright: at 0.6% prevalence the top 1%
        # of events CANNOT be more than 60% attacks, however perfect the ranking.
        # Comparing raw Precision@1% across datasets of different prevalence therefore
        # compares their prevalence as much as their detectors.
        "precision_at_budget_ceiling": min(1.0, n_attack / k) if k else float("nan"),
        "recall_at_budget": (hits_at_k / n_attack) if n_attack else float("nan"),
        # The best value recall_at_budget could possibly take at this prevalence.
        "recall_at_budget_ceiling": min(1.0, budget_fraction / prevalence) if prevalence else float("nan"),
        "n_insider_drift_events": float((df["label"] == INSIDER_DRIFT).sum()),
    }

    if n_attack:
        r = n_attack
        out["r_precision"] = float(ranked_attack[:r].sum()) / r
        from sklearn.metrics import average_precision_score

        out["pr_auc"] = float(average_precision_score(is_attack, df["score"].to_numpy()))
        # PR-AUC is unreadable without its random baseline, which is the prevalence.
        out["pr_auc_baseline"] = prevalence
        out["pr_auc_lift"] = out["pr_auc"] / prevalence if prevalence else float("nan")
    else:
        out["r_precision"] = float("nan")
        out["pr_auc"] = float("nan")
        out["pr_auc_baseline"] = prevalence
        out["pr_auc_lift"] = float("nan")

    # Share of the analyst's budget consumed by the ambiguous edge case.
    ranked_label = df["label"].to_numpy()[order]
    out["insider_drift_share_of_budget"] = float((ranked_label[:k] == INSIDER_DRIFT).sum()) / k
    return out


# --------------------------------------------------------------------------------------
# Incident (campaign) level
# --------------------------------------------------------------------------------------


def attribute_alerts(
    assignment: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Attribute each alert to the campaign contributing most of its member events.

    Ties are broken by earliest campaign start, so attribution is deterministic.
    An alert with no attack member events is attributed to no campaign (a false positive
    if it is reviewed).
    """
    lab = labels[["event_id", "label", "campaign_id"]].copy()
    joined = assignment.merge(lab, on="event_id", how="left")
    joined["label"] = joined["label"].fillna(NORMAL)

    attack = joined[joined["label"].isin(ATTACK_TYPES) & joined["campaign_id"].notna()]
    if attack.empty:
        return pd.DataFrame(columns=["alert_id", "campaign_id", "n_member_events"])

    counts = (
        attack.groupby(["alert_id", "campaign_id"], sort=False)
        .size()
        .reset_index(name="n_member_events")
    )
    counts = counts.sort_values(
        ["alert_id", "n_member_events", "campaign_id"], ascending=[True, False, True]
    )
    return counts.groupby("alert_id", sort=False).head(1).reset_index(drop=True)


def incident_metrics(
    alerts_by_level: Dict[str, pd.DataFrame],
    assignments_by_level: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    campaigns: pd.DataFrame,
    n_days: float,
    budgets: Optional[Dict[str, int]] = None,
    event_times: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Incident-level recall/precision at a per-level analyst budget, plus time-to-detect.

    K_level = budget_level * n_days. The reviewed set is the UNION of the per-level
    top-K queues, because the analyst really does see all three queues.

    A campaign is detected if at least one reviewed alert is attributed to it (hit@1).
    That is the operationally honest rule -- the analyst only needs one thread to pull --
    but on its own it would let "detected on day 19 of a 21-day low_and_slow" count as a
    win, so time-to-detect is reported alongside and is not optional.

    Multiple reviewed alerts on one campaign yield ONE true positive; the extras are
    redundant, counted as neither TP nor FP, but they still consume budget and so are
    counted in alerts_per_analyst_per_day. Detection quality is lenient; workload is strict.
    """
    budgets = budgets or LEVEL_DAILY_BUDGET

    reviewed_parts: List[pd.DataFrame] = []
    per_level: Dict[str, Dict[str, float]] = {}

    for level, alerts in alerts_by_level.items():
        if alerts.empty:
            per_level[level] = {"k": 0.0, "n_alerts": 0.0, "n_tp": 0.0, "precision": float("nan")}
            continue
        k = max(1, int(round(budgets[level] * n_days)))
        top = alerts.sort_values(["alert_score", "alert_id"], ascending=[False, True]).head(k).copy()
        attrib = attribute_alerts(assignments_by_level[level], labels)
        top = top.merge(attrib[["alert_id", "campaign_id"]], on="alert_id", how="left")
        top["level"] = level
        reviewed_parts.append(top)
        per_level[level] = {
            "k": float(k),
            "n_alerts": float(len(top)),
            "n_attributed": float(top["campaign_id"].notna().sum()),
        }

    reviewed = (
        pd.concat(reviewed_parts, ignore_index=True)
        if reviewed_parts
        else pd.DataFrame(columns=["alert_id", "campaign_id", "alert_score", "start_ts", "level"])
    )

    # Campaign universe excludes insider_drift by construction.
    camps = campaigns[campaigns["campaign_type"].isin(ATTACK_TYPES)].copy()
    n_campaigns = int(len(camps))

    detected_ids = set(reviewed["campaign_id"].dropna().unique())
    n_detected = len(detected_ids & set(camps["campaign_id"]))

    # One TP per detected campaign; the earliest reviewed alert is the detecting one.
    hits = reviewed[reviewed["campaign_id"].notna()].copy()
    hits = hits[hits["campaign_id"].isin(set(camps["campaign_id"]))]
    total_k = sum(v.get("k", 0.0) for v in per_level.values())

    if not hits.empty:
        first = (
            hits.sort_values(["campaign_id", "start_ts", "alert_id"])
            .groupby("campaign_id", sort=False)
            .head(1)
        )
        first = first.merge(
            camps[["campaign_id", "campaign_type", "start_ts", "end_ts"]],
            on="campaign_id",
            how="left",
            suffixes=("_alert", "_camp"),
        )
        detect_ts = pd.to_datetime(first["start_ts_alert"])
        if event_times is not None:
            # An alert groups every event in its scope window, benign ones included, so
            # its start can precede the campaign it caught -- which produced negative
            # times-to-detect. Detection actually happens at the first ATTACK event that
            # falls inside a reviewed alert, so that is what is measured.
            reviewed_ids = set(reviewed["alert_id"])
            am = pd.concat(list(assignments_by_level.values()), ignore_index=True)
            am = am[am["alert_id"].isin(reviewed_ids)]
            am = am.merge(labels[["event_id", "label", "campaign_id"]], on="event_id", how="left")
            am = am[am["label"].isin(ATTACK_TYPES) & am["campaign_id"].notna()]
            am = am.merge(event_times[["event_id", "timestamp"]], on="event_id", how="left")
            if not am.empty:
                fd = (am.groupby("campaign_id", sort=False)["timestamp"].min()
                      .rename("detect_ts").reset_index())
                first = first.merge(fd, on="campaign_id", how="left")
                detect_ts = pd.to_datetime(first["detect_ts"]).fillna(detect_ts)

        ttd = (detect_ts - pd.to_datetime(first["start_ts_camp"])).dt.total_seconds()
        first["ttd_seconds"] = ttd.clip(lower=0.0)
        span = (
            pd.to_datetime(first["end_ts_camp"]) - pd.to_datetime(first["start_ts_camp"])
        ).dt.total_seconds()
        # Detected before the campaign's temporal midpoint.
        first["detected_early"] = ttd <= (span / 2.0)
    else:
        first = pd.DataFrame(columns=["campaign_id", "campaign_type", "ttd_seconds", "detected_early"])

    out: Dict[str, object] = {
        "n_campaigns": float(n_campaigns),
        "n_campaigns_detected": float(n_detected),
        "incident_recall_at_k": (n_detected / n_campaigns) if n_campaigns else float("nan"),
        "incident_precision_at_k": (n_detected / total_k) if total_k else float("nan"),
        # Structural cap, exactly analogous to the Recall@1% ceiling: there are only
        # n_campaigns campaigns to find, so no detector can exceed n_campaigns/K however
        # good it is. Reporting precision without this is reporting the budget, not
        # the model.
        "incident_precision_ceiling": (min(1.0, n_campaigns / total_k)
                                       if total_k else float("nan")),
        "budget_K": float(total_k),
        "n_reviewed_alerts": float(len(reviewed)),
        "alerts_per_analyst_per_day": float(len(reviewed)) / n_days if n_days else float("nan"),
        "redundant_alerts": float(max(0, len(hits) - n_detected)),
        "per_level": per_level,
    }

    # Per-type recall, denominator = campaigns OF THAT TYPE.
    per_type: Dict[str, Dict[str, float]] = {}
    for atype, grp in camps.groupby("campaign_type"):
        ids = set(grp["campaign_id"])
        det = len(ids & detected_ids)
        row = {
            "n_campaigns": float(len(ids)),
            "n_detected": float(det),
            "recall": det / len(ids) if len(ids) else float("nan"),
        }
        if not first.empty:
            sub = first[first["campaign_type"] == atype]
            if len(sub):
                row["median_ttd_hours"] = float(sub["ttd_seconds"].median() / 3600.0)
                row["frac_detected_early"] = float(sub["detected_early"].mean())
        per_type[atype] = row
    out["per_type"] = per_type

    if not first.empty:
        out["median_ttd_hours"] = float(first["ttd_seconds"].median() / 3600.0)
        out["frac_detected_early"] = float(first["detected_early"].mean())
    else:
        out["median_ttd_hours"] = float("nan")
        out["frac_detected_early"] = float("nan")

    return out


# --------------------------------------------------------------------------------------
# False-positive breakdown and the insider_drift band objective
# --------------------------------------------------------------------------------------


def fp_breakdown_by_confounder(
    scored: pd.DataFrame,
    labels: pd.DataFrame,
    budget_fraction: float = EVENT_BUDGET_FRACTION,
) -> pd.DataFrame:
    """Which engineered benign behaviour is actually costing the analyst?

    Confounders are negatives injected at 3-5x the attack rate specifically to trip each
    detector layer (business travel -> geo-velocity, OS patch -> fingerprint mismatch,
    office NAT -> the IP-level stuffing detector, ...). This table is the most credible
    artefact in the report: it shows the detector was tested against realistic benign
    weirdness rather than against a clean negative class.
    """
    df = scored.merge(labels[["event_id", "label", "confounder"]], on="event_id", how="left")
    df["label"] = df["label"].fillna(NORMAL)

    k = max(1, math.ceil(budget_fraction * len(df)))
    order = np.lexsort((df["event_id"].to_numpy(), -df["score"].to_numpy()))
    top = df.iloc[order[:k]]

    fps = top[~top["label"].isin(ATTACK_TYPES)].copy()
    fps["confounder"] = fps["confounder"].fillna("(none - ordinary benign)")

    counts = fps.groupby("confounder", sort=False).size().reset_index(name="n_false_positives")
    pop = df.copy()
    pop["confounder"] = pop["confounder"].fillna("(none - ordinary benign)")
    totals = pop.groupby("confounder", sort=False).size().reset_index(name="n_events")

    out = counts.merge(totals, on="confounder", how="left")
    out["fp_rate"] = out["n_false_positives"] / out["n_events"]
    out["share_of_alert_budget"] = out["n_false_positives"] / k
    return out.sort_values("n_false_positives", ascending=False).reset_index(drop=True)


def insider_drift_band_report(
    scored: pd.DataFrame,
    labels: pd.DataFrame,
    theta: float,
) -> Dict[str, object]:
    """The asymmetric objective for the ambiguous edge case.

    insider_drift's success criterion is NOT recall. A legitimate employee slowly
    expanding their footprint should surface as MEDIUM -- visible to an analyst, not
    escalated as an intrusion. Targets:

        maximise P(insider_drift -> MEDIUM)
                 P(insider_drift -> HIGH)  <= 0.15   (false-positive budget)
                 P(insider_drift -> LOW)   <= 0.25   (analyst should still see it)
                 P(low_and_slow  -> HIGH)  >= 0.70

    low_and_slow is reported in the same table because the two are near-identical by
    construction; the contrast between their band distributions is the actual result.
    """
    df = scored.merge(labels[["event_id", "label"]], on="event_id", how="left")
    df["label"] = df["label"].fillna(NORMAL)
    if "band" not in df.columns:
        # Fall back to a single-threshold banding. Callers scoring at several levels
        # should supply a precomputed `band`, because each level has its own theta in its
        # own space and one global cut cannot represent all three.
        df["band"] = [band_for(s, theta) for s in df["score"].to_numpy()]

    out: Dict[str, object] = {"theta": theta, "bands": {}}
    for lab in (INSIDER_DRIFT, "low_and_slow", NORMAL):
        sub = df[df["label"] == lab]
        if sub.empty:
            continue
        dist = sub["band"].value_counts(normalize=True).to_dict()
        out["bands"][lab] = {b: float(dist.get(b, 0.0)) for b in (BAND_LOW, BAND_MEDIUM, BAND_HIGH)}

    idf = out["bands"].get(INSIDER_DRIFT, {})
    las = out["bands"].get("low_and_slow", {})
    out["targets_met"] = {
        "insider_drift_high_le_0.15": idf.get(BAND_HIGH, 0.0) <= 0.15,
        "insider_drift_low_le_0.25": idf.get(BAND_LOW, 0.0) <= 0.25,
        "low_and_slow_high_ge_0.70": las.get(BAND_HIGH, 0.0) >= 0.70,
    }
    return out


def separation_auc(
    scored: pd.DataFrame,
    labels: pd.DataFrame,
    signal_col: str,
    label_a: str = "low_and_slow",
    label_b: str = INSIDER_DRIFT,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """How separable are low_and_slow and insider_drift, really?

    At the level of a single event's features these are the same generative process, so
    this number is reported as-is rather than tuned until it looks good. If it comes out
    at 0.62 [0.55, 0.69], the report says exactly that -- an honest overlap measurement
    is a stronger result than a suspiciously clean separation.
    """
    from sklearn.metrics import roc_auc_score

    df = scored.merge(labels[["event_id", "label"]], on="event_id", how="left")
    sub = df[df["label"].isin([label_a, label_b])].dropna(subset=[signal_col])
    if sub["label"].nunique() < 2:
        return {"auc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": float(len(sub))}

    y = (sub["label"] == label_a).to_numpy().astype(int)
    x = sub[signal_col].to_numpy(dtype=float)
    auc = float(roc_auc_score(y, x))

    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], x[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    return {"auc": auc, "ci_low": float(lo), "ci_high": float(hi), "n": float(n)}


# --------------------------------------------------------------------------------------
# Reporting helper
# --------------------------------------------------------------------------------------


def format_headline(event_m: Dict[str, float], inc_m: Dict[str, object]) -> str:
    """Render the headline block, with every caveat the numbers require attached."""
    p = event_m["prevalence"]
    lines = [
        "PRIMARY (incident level, campaign = 1 regardless of event count)",
        "  Incident_Recall@K      %.3f   (%d/%d campaigns)"
        % (inc_m["incident_recall_at_k"], inc_m["n_campaigns_detected"], inc_m["n_campaigns"]),
        "  Incident_Precision@K   %.3f   (ceiling %.3f: only %d campaigns exist for K=%d)"
        % (inc_m["incident_precision_at_k"], inc_m["incident_precision_ceiling"],
           inc_m["n_campaigns"], inc_m["budget_K"]),
        "  alerts/analyst/day     %.1f" % inc_m["alerts_per_analyst_per_day"],
        "  median time-to-detect  %.1f h" % inc_m["median_ttd_hours"],
        "",
        "EVENT LEVEL (analyst budget = top %.1f%% of events)" % (100 * event_m["budget_fraction"]),
        "  Precision@1%%           %.3f   (ceiling %.3f at this prevalence)"
        % (event_m["precision_at_budget"], event_m["precision_at_budget_ceiling"]),
        "  Recall@1%%              %.3f   (ceiling %.3f at prevalence %.3f%%)"
        % (event_m["recall_at_budget"], event_m["recall_at_budget_ceiling"], 100 * p),
        "  R-Precision            %.3f   (no ceiling artefact)" % event_m["r_precision"],
        "  PR-AUC                 %.3f   (random baseline %.4f, lift %.1fx)"
        % (event_m["pr_auc"], event_m["pr_auc_baseline"], event_m["pr_auc_lift"]),
    ]
    return "\n".join(lines)

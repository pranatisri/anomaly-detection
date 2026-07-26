"""Adversarial artefact audit (CM9) -- a deliverable, not a formality.

The failure mode this exists to catch: a synthetic benchmark where attacks are
identifiable from incidental properties of how they were WRITTEN rather than from how
they BEHAVE. If attack rows were appended in a block, or produced by a second code path
with different float formatting or a different MAC string style, then a model can score
near-perfectly while learning nothing about intrusion detection -- and every metric in
the report becomes meaningless.

Method: hand a gradient-boosted classifier ONLY features that carry no legitimate
behavioural information, and check it cannot predict the label.

    GATE features       row position in file, string lengths, float decimal precision,
                        a random hash bucket of the source IP, field-presence flags.
                        None of these can legitimately indicate an intrusion.
                        REQUIREMENT: ROC-AUC <= 0.60.

    DIAGNOSTIC features adds absolute timestamp and command-sequence length. These are
                        reported but NOT gated, because they carry real signal by design
                        (low_and_slow is deliberately off-hours; lateral movement really
                        does issue longer command sequences). Treating them as artefacts
                        would flag intended behaviour as leakage.

Second check: a per-value exclusivity scan. Any categorical value that appears at least
`min_support` times and is followed by an attack label 100% of the time is an oracle --
one rule would solve the benchmark. Those must not exist.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ATTACK_TYPES, NORMAL

GATE_AUC_MAX = 0.60
N_AUDIT_SPLITS = 7   # gate on the MEAN over this many grouped splits


def _decimals(x: float) -> int:
    """Number of digits after the decimal point -- a pure formatting artefact."""
    s = repr(float(x))
    return len(s.split(".")[1]) if "." in s else 0


def build_artefact_features(events: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Construct artefact-only features. Returns (frame, gate_cols, diagnostic_cols)."""
    n = len(events)
    f = pd.DataFrame(index=np.arange(n))

    # Position in the file. If attacks were appended as a block rather than interleaved
    # and globally re-sorted, this alone would be a near-perfect predictor.
    f["row_index"] = np.arange(n)
    f["row_index_frac"] = np.arange(n) / max(1, n - 1)

    # String lengths of OPAQUE IDENTIFIERS. These are pure formatting: an entity id or a
    # MAC address carries no meaning in its length, so if length predicts the label, two
    # code paths formatted values differently.
    for col in ("entity_id", "source_ip", "device_mac"):
        if col in events.columns:
            f["len_%s" % col] = events[col].astype(str).str.len().to_numpy()

    # Float formatting precision -- differs if two code paths produced the numbers.
    if "session_duration" in events.columns:
        f["dec_session_duration"] = [_decimals(v) for v in events["session_duration"].to_numpy()]
    for col in ("geo_lat", "geo_lon"):
        if col in events.columns:
            f["dec_%s" % col] = [_decimals(v) for v in events[col].to_numpy()]

    # A random bucketing of the IP. Carries no ordering or semantic information; if it
    # predicts the label, attacker IPs came from a disjoint pool (CM4 violation).
    if "source_ip" in events.columns:
        f["ip_hash_bucket"] = events["source_ip"].astype(str).map(lambda s: hash(s) % 997).to_numpy()
        f["ip_octet1"] = events["source_ip"].astype(str).str.split(".").str[0].astype(int).to_numpy()

    # Field presence. Only informative if missingness was injected asymmetrically.
    for col in ("geo_lat", "geo_country", "device_mac", "command_sequence"):
        if col in events.columns:
            f["present_%s" % col] = events[col].notna().astype(int).to_numpy()

    gate_cols = list(f.columns)

    # --- diagnostic only, deliberately excluded from the gate ---
    # String lengths of SEMANTIC categoricals are content proxies, not artefacts: the
    # length of `resource_accessed` stands in for *which* resource, and which resource
    # was touched is exactly what a behavioural detector is supposed to weigh. Brute
    # force really is authentication traffic, so `sso/read` (8 chars) really is
    # over-represented among attacks. Gating on these would mean demanding the generator
    # erase true signal, so they are measured and reported, not gated.
    for col in ("resource_accessed", "device_os", "device_firmware", "geo_country",
                "auth_method"):
        if col in events.columns:
            f["len_%s" % col] = events[col].astype(str).str.len().to_numpy()
    if "timestamp" in events.columns:
        ts = pd.to_datetime(events["timestamp"])
        f["abs_timestamp"] = ts.astype("int64").to_numpy() / 1e9
    if "command_sequence" in events.columns:
        f["n_commands"] = events["command_sequence"].map(
            lambda v: len(v) if isinstance(v, (list, np.ndarray)) else 0).to_numpy()
    diag_cols = [c for c in f.columns if c not in gate_cols]
    return f, gate_cols, diag_cols


def _auc(X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray] = None,
         seed: int = 0) -> float:
    """ROC-AUC of an artefact-only model, on a CAMPAIGN-GROUPED split.

    The split must be grouped, and this matters more than it looks. Attacks are
    campaigns: one brute force puts ~50 events on a single source IP. Under a plain
    random split the same IP lands in train and test, so any high-cardinality identifier
    (IP, MAC, device, entity) scores near 1.0 purely by memorising which campaign an
    event came from. That is contamination in the AUDIT, not evidence of a generator
    leak, and reading it as one sends you chasing a phantom.

    Grouping by campaign forces the question we actually care about: do artefacts
    generalise from known attacks to UNSEEN ones? If yes, the generator is leaking.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.metrics import roc_auc_score

    if y.sum() < 20 or (~y.astype(bool)).sum() < 20:
        return float("nan")

    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=seed)
        tr, te = next(gss.split(X, y, groups=groups))
        if y[tr].sum() < 10 or y[te].sum() < 10:
            return float("nan")
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=seed,
                                              stratify=y)
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=250, num_leaves=63, learning_rate=0.08,
            class_weight="balanced", random_state=seed, verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=250, random_state=seed)
    model.fit(Xtr, ytr)
    return float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))


# Vocabulary fields: a small shared alphabet every entity draws from. If one *value*
# here is attack-exclusive, the generator has invented an attack-only token and a single
# rule solves the benchmark. These are GATED.
VOCAB_COLUMNS: Tuple[str, ...] = ("resource_accessed", "auth_method", "device_os",
                                  "device_protocol", "geo_country", "resource_type",
                                  "device_firmware")

# Identifier fields: high-cardinality handles for specific infrastructure. Per-value
# exclusivity is EXPECTED here and is not a leak -- a compromised host's IP genuinely is
# 100% malicious, in this dataset and in reality. What would be a leak is attacker
# identifiers being drawn from a structurally separate pool, and that is caught by the
# campaign-grouped artefact model instead (ip_hash_bucket / ip_octet1). Reported, not gated.
IDENTIFIER_COLUMNS: Tuple[str, ...] = ("source_ip", "device_mac", "entity_id", "device_id")


def exclusivity_scan(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    min_support: int = 5,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Find categorical values that are perfect attack oracles.

    A value with decent support and P(attack | value) == 1.0 means a single rule solves
    the benchmark. In a realistic dataset attackers reuse the resources, auth methods and
    platforms benign traffic also uses, so no such VOCABULARY value should exist.
    """
    columns = list(columns) if columns is not None else list(VOCAB_COLUMNS)
    df = events.merge(labels[["event_id", "label"]], on="event_id", how="left")
    df["label"] = df["label"].fillna(NORMAL)
    is_attack = df["label"].isin(ATTACK_TYPES)

    out: List[dict] = []
    for col in columns:
        if col not in df.columns:
            continue
        g = df.groupby(df[col].astype(str))
        support = g.size()
        rate = is_attack.groupby(df[col].astype(str)).mean()
        for val, sup in support.items():
            if sup >= min_support and rate[val] >= 1.0:
                out.append({"column": col, "value": val, "support": int(sup),
                            "p_attack_given_value": float(rate[val])})
    return pd.DataFrame(out, columns=["column", "value", "support", "p_attack_given_value"])


def run_audit(events: pd.DataFrame, labels: pd.DataFrame, seed: int = 0) -> Dict[str, object]:
    order = events[["event_id"]].merge(
        labels[["event_id", "label", "campaign_id"]], on="event_id", how="left")
    y = order["label"].fillna(NORMAL).isin(ATTACK_TYPES).to_numpy().astype(int)

    # Group = the campaign for attack events, the entity for benign ones. No campaign
    # may appear on both sides of the split.
    groups = order["campaign_id"].fillna("").to_numpy().astype(object)
    ent = events["entity_id"].to_numpy().astype(object)
    groups = np.where(groups == "", np.array(["ent:" + str(e) for e in ent], dtype=object), groups)

    feats, gate_cols, diag_cols = build_artefact_features(events)

    # Average over several grouped splits. A single split is far too noisy to adjudicate
    # this gate: measured across 10 splits the artefact AUC had sd ~0.07, so one draw
    # reported 0.6036 (a FAIL) for a dataset whose mean was 0.42 and whose worst
    # individual feature was 0.57. Gating on a point estimate with that much variance
    # decides pass/fail by which random_state was hard-coded.
    gate_runs = [_auc(feats[gate_cols], y, groups=groups, seed=seed + k)
                 for k in range(N_AUDIT_SPLITS)]
    gate_runs = [v for v in gate_runs if not np.isnan(v)]
    gate_auc = float(np.mean(gate_runs)) if gate_runs else float("nan")
    gate_sd = float(np.std(gate_runs)) if len(gate_runs) > 1 else 0.0
    diag_auc = (float(np.mean([_auc(feats[gate_cols + diag_cols], y, groups=groups,
                                    seed=seed + k) for k in range(N_AUDIT_SPLITS)]))
                if diag_cols else float("nan"))

    per_feature: Dict[str, float] = {}
    for c in gate_cols:
        per_feature[c] = _auc(feats[[c]], y, groups=groups, seed=seed)

    excl = exclusivity_scan(events, labels, columns=VOCAB_COLUMNS)
    excl_ident = exclusivity_scan(events, labels, columns=IDENTIFIER_COLUMNS)

    passed = (np.isnan(gate_auc) or gate_auc <= GATE_AUC_MAX) and excl.empty
    return {
        "gate_auc": gate_auc,
        "gate_auc_sd": gate_sd,
        "gate_auc_runs": len(gate_runs),
        "gate_auc_max": GATE_AUC_MAX,
        "diagnostic_auc": diag_auc,
        "per_feature_auc": per_feature,
        "exclusivity_violations": excl,
        "identifier_exclusivity": excl_ident,
        "n_attack_events": int(y.sum()),
        "passed": bool(passed),
    }


def format_report(res: Dict[str, object]) -> str:
    lines = ["=" * 74, "ADVERSARIAL ARTEFACT AUDIT (CM9)", "=" * 74]
    lines.append("attack events                  %d" % res["n_attack_events"])
    lines.append("")
    lines.append("GATE  artefact-only ROC-AUC    %.4f +/- %.4f  (mean of %d grouped "
                 "splits; must be <= %.2f)"
                 % (res["gate_auc"], res.get("gate_auc_sd", 0.0),
                    res.get("gate_auc_runs", 1), res["gate_auc_max"]))
    lines.append("      %s" % ("PASS - artefacts do not identify attacks"
                               if res["gate_auc"] <= res["gate_auc_max"]
                               else "FAIL - the generator is leaking; fix and regenerate"))
    lines.append("")
    lines.append("DIAG  + timestamp & cmd length %.4f   (not gated: real signal by design)"
                 % res["diagnostic_auc"])
    lines.append("")
    lines.append("Worst individual artefact features:")
    pf = sorted(res["per_feature_auc"].items(),
                key=lambda kv: (-kv[1] if not np.isnan(kv[1]) else 0))
    for name, auc in pf[:8]:
        flag = "  <-- investigate" if (not np.isnan(auc) and auc > res["gate_auc_max"]) else ""
        lines.append("    %-26s %.4f%s" % (name, auc, flag))
    lines.append("")
    excl = res["exclusivity_violations"]
    if len(excl):
        lines.append("VOCABULARY EXCLUSIVITY (GATED) - P(attack|value) == 1.0, support >= 5:")
        for _, r in excl.head(20).iterrows():
            lines.append("    %-20s %-28s support=%d" % (r["column"], r["value"], r["support"]))
        lines.append("    -> an attack-only token exists. One rule solves this. Fix the generator.")
    else:
        lines.append("VOCABULARY EXCLUSIVITY (GATED) clean - no attack-only token exists")

    ident = res.get("identifier_exclusivity")
    if ident is not None:
        n = len(ident)
        lines.append("IDENTIFIER EXCLUSIVITY (INFO)  %d high-cardinality value(s) attack-only"
                     % n)
        lines.append("    expected and not gated: a compromised host's IP is 100%% malicious in")
        lines.append("    reality too. Generalisation across campaigns is what the gate tests.")
    lines.append("")
    lines.append("OVERALL: %s" % ("PASS" if res["passed"] else "FAIL"))
    lines.append("=" * 74)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the adversarial artefact audit.")
    ap.add_argument("--tag", default="seed1_delta05")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = ap.parse_args()

    d = os.path.abspath(args.data)
    events = pd.read_parquet(os.path.join(d, "events_%s.parquet" % args.tag))
    labels = pd.read_parquet(os.path.join(d, "labels_%s.parquet" % args.tag))

    res = run_audit(events, labels)
    print(format_report(res))
    return 0 if res["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())

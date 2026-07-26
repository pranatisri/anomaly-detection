"""Shared constants and the label-isolation contract.

This module is imported by both the generator and the detector, so it must contain
NO generator parameters. Anything that describes how attacks are synthesised belongs
in generator.py and must never be importable from the detector side (CM3).
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

# --------------------------------------------------------------------------------------
# Attack taxonomy
# --------------------------------------------------------------------------------------

NORMAL = "normal"

ATTACK_TYPES: Tuple[str, ...] = (
    "brute_force",
    "credential_stuffing",
    "impossible_travel",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow",
)

# insider_drift is deliberately NOT in ATTACK_TYPES. It is a legitimate-but-ambiguous
# edge case used for false-positive tuning, and is excluded from the positive class in
# both the numerator and the denominator of every recall metric. It has its own
# asymmetric 3-band objective instead (see evaluate.insider_drift_band_report).
INSIDER_DRIFT = "insider_drift"

ALL_LABELS: Tuple[str, ...] = (NORMAL,) + ATTACK_TYPES + (INSIDER_DRIFT,)

# Benign confounder tags. These are NEGATIVES -- label == NORMAL -- but each one is
# engineered to trip a specific detector layer. False positives are broken down by these
# tags, which is the most informative table in the report (CM5).
CONFOUNDERS: Tuple[str, ...] = (
    "business_travel",       # trips L0 geo-velocity
    "password_reset_storm",  # trips L0 failed-auth rate
    "cert_retry_loop",       # trips L0 failed-auth rate
    "os_patch",              # trips L0 fingerprint mismatch
    "device_refresh",        # trips L0 fingerprint mismatch
    "oncall_rotation",       # trips L1 habitual-hour
    "maintenance_window",    # trips L1 habitual-hour
    "project_onboarding",    # trips L1 novelty + L3 new-edge
    "ci_automation_burst",   # trips L2 surprisal + session duration
    "shared_jump_host",      # trips the IP-level credential-stuffing detector
    "office_nat",            # trips the IP-level credential-stuffing detector
    "credential_rotation",   # trips auth_method change
)

ENTITY_TYPES: Tuple[str, ...] = ("user", "service_account", "edge_device")

# --------------------------------------------------------------------------------------
# Aggregation levels
# --------------------------------------------------------------------------------------
# Entity-event, source_ip-window and entity-long-window are different units with
# different base rates and different analyst workflows. They are NEVER pooled into a
# single ranking -- each gets its own calibration, its own threshold and its own budget.
#
# This split is also structurally required: credential_stuffing is an IP-level
# phenomenon (many entities, few IPs) and low_and_slow needs a multi-day window. A
# purely per-event per-entity detector misses both by construction.

LEVEL_ENTITY = "entity"
LEVEL_IP = "ip"
LEVEL_LONG = "long"

LEVELS: Tuple[str, ...] = (LEVEL_ENTITY, LEVEL_IP, LEVEL_LONG)

# scope_key column(s) used to group events into one alert, per level.
LEVEL_SCOPE_KEYS: Dict[str, Tuple[str, ...]] = {
    LEVEL_ENTITY: ("entity_id",),
    LEVEL_IP: ("source_ip",),
    LEVEL_LONG: ("entity_id", "day"),
}

# Events sharing a scope_key within this many seconds collapse into ONE alert.
LEVEL_MERGE_WINDOW_S: Dict[str, int] = {
    LEVEL_ENTITY: 60 * 60,
    LEVEL_IP: 60 * 60,
    LEVEL_LONG: 24 * 60 * 60,
}

# Alerts an analyst can review per day, per level. Used for Incident_*@K and for
# "alerts per analyst per day".
LEVEL_DAILY_BUDGET: Dict[str, int] = {
    LEVEL_ENTITY: 30,
    LEVEL_IP: 10,
    LEVEL_LONG: 10,
}

# --------------------------------------------------------------------------------------
# Label isolation (CM2)
# --------------------------------------------------------------------------------------
# The detector may only ever see these columns. Enforced by an explicit whitelist with
# no wildcards, checked at the detector entrypoint. Convention is not sufficient: the
# failure mode this guards against (campaign_id or attack_phase silently riding along in
# the feature frame) is invisible and catastrophic.

FEATURE_COLUMNS: FrozenSet[str] = frozenset({
    "event_id",
    "entity_id",
    "entity_type",
    "role",            # organisational fact, drives benign assignment too (CM3 exception)
    "timestamp",
    "source_ip",
    "geo_lat",
    "geo_lon",
    "geo_country",
    "resource_accessed",
    "resource_type",
    "auth_method",
    "auth_result",
    "session_duration",
    "command_sequence",
    "device_id",
    "device_os",
    "device_firmware",
    "device_mac",
    "device_protocol",
})

# Columns that live ONLY in the sealed label file, keyed by event_id. Loaded by eval/
# and by nothing else.
LABEL_COLUMNS: FrozenSet[str] = frozenset({
    "event_id",
    "label",
    "campaign_id",
    "campaign_type",
    "confounder",
    "is_cold_start",
})

# Anything matching these is a label-adjacent trap: it must never reach the detector.
FORBIDDEN_SUBSTRINGS: Tuple[str, ...] = (
    "label", "campaign", "attack", "anomal", "seed", "difficulty", "delta",
    "is_synthetic", "ground_truth", "confounder", "phase",
)


class LabelLeakError(AssertionError):
    """Raised when the detector is handed something it must never see."""


def assert_feature_frame(df, where: str = "detector") -> None:
    """Guard the detector entrypoint. Fails loudly rather than silently inflating metrics.

    Checks three things:
      1. no column outside the explicit whitelist
      2. no label-adjacent column name, even if somehow whitelisted
      3. the generator module is not loaded in this process
    """
    import sys

    cols = set(map(str, df.columns))

    extra = cols - set(FEATURE_COLUMNS)
    if extra:
        raise LabelLeakError(
            "%s received non-whitelisted columns: %s" % (where, sorted(extra))
        )

    for col in cols:
        low = col.lower()
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in low:
                raise LabelLeakError(
                    "%s received label-adjacent column %r (matched %r)" % (where, col, bad)
                )

    leaked = [m for m in sys.modules if m.split(".")[-1] == "generator"]
    if leaked:
        raise LabelLeakError(
            "%s: generator module is importable in this process (%s); the detector must "
            "not be able to read generator parameters" % (where, leaked)
        )


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

EVENT_BUDGET_FRACTION = 0.01   # the "top 1% of events" alert budget
DELTA_SWEEP: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
DEV_SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
HOLDOUT_SEEDS: Tuple[int, ...] = (101, 102, 103, 104, 105)

# Risk bands. insider_drift's objective is to land in MEDIUM, not to be recalled.
BAND_LOW, BAND_MEDIUM, BAND_HIGH = "LOW", "MEDIUM", "HIGH"
BANDS: Tuple[str, ...] = (BAND_LOW, BAND_MEDIUM, BAND_HIGH)


def band_for(score: float, theta: float) -> str:
    """Map a calibrated score to an analyst-facing band, relative to the level's threshold."""
    if score >= theta:
        return BAND_HIGH
    if score >= 0.5 * theta:
        return BAND_MEDIUM
    return BAND_LOW

"""L4 fusion, calibration, thresholding and the L6 explanation layer.

Fusion is deliberately mostly UNSUPERVISED. Labels touch the pipeline at exactly one
point: a 1-D monotone map from the fused score to a probability. That restriction is the
whole argument for using them at all -- a monotone scalar transform cannot re-weight
features, so it cannot memorise which signal the generator wired to which attack type.
It can only stretch the axis so that "top 1%" means something probabilistic.

Naive summing would double-count. L0 geo_velocity, L1 country_novelty and the IP-change
signal all fire together on impossible travel, so a plain sum reports one event as three
independent alarms. Correlation-corrected Stouffer divides by sqrt(w' Sigma w), which
inflates the denominator exactly when signals are redundant.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import LEVEL_ENTITY, LEVEL_IP, LEVEL_LONG, LEVELS
from features import SIGNAL_BY_NAME, SIGNAL_NAMES, Signal

# Which signals feed which aggregation level. The levels are scored SEPARATELY and never
# pooled: credential stuffing is an IP-level phenomenon (many entities, few IPs) and
# low_and_slow only exists over days. A single per-event, per-entity ranking misses both
# by construction, no matter how good the model is.
LEVEL_SIGNALS: Dict[str, List[str]] = {
    LEVEL_ENTITY: [s.name for s in SIGNAL_BY_NAME.values() if s.level == "entity"],
    LEVEL_IP: [s.name for s in SIGNAL_BY_NAME.values() if s.level == "ip"],
    LEVEL_LONG: [s.name for s in SIGNAL_BY_NAME.values() if s.level == "long"],
}

# Analyst priors: how much a signal is trusted a priori. NOT learned from labels --
# learning these would be learning the generator's wiring diagram.
SIGNAL_PRIOR: Dict[str, float] = {
    "geo_velocity": 1.0, "fail_rate_entity": 0.9, "fingerprint_mismatch": 0.9,
    "burst_ratio": 0.9,
    "cmd_surprisal": 0.7, "uncorroborated_new_edges_7d": 0.9,
    "hour_surprisal": 0.6, "ip_novelty": 0.5, "country_novelty": 0.7,
    "resource_surprisal": 0.8, "duration_z": 0.4, "ncmd_z": 0.4,
    "peer_incongruence": 0.8, "auth_method_novelty": 0.5,
    "fail_rate_ip": 0.9, "ip_entity_fanout": 1.0, "ip_fail_ratio": 0.9,
    "new_resource_rate_7d": 0.8, "offhours_rate_7d": 0.6, "breadth_ratio_7d": 0.7,
}

# MITIGATING signals are evidence FOR benignity: they are subtracted from the fused
# score instead of added to it.
#
# Without this the fusion has no exculpatory path at all. Every aggregator here is
# monotone increasing -- Stouffer sums, top2mean takes the two LARGEST z -- so a signal
# saying "five teammates gained this same access in the same week" simply cannot lower a
# score, and `insider_drift` sat at 47% HIGH against a <=15% target. An analyst reasons
# in both directions; the fusion has to as well.
#
# The weight is an analyst prior, fixed a priori. It is deliberately NOT fitted on labels
# -- tuning it against the insider_drift band targets would be exactly the memorisation
# this architecture exists to avoid.
MITIGATING_SIGNALS: Tuple[str, ...] = ("corroboration_7d",)
MITIGATION_WEIGHT = 0.75

Z_CLIP_LO, Z_CLIP_HI = -4.0, 6.0
TARGET_FPR = 0.01
REF_SAMPLE_CAP = 120_000


def top2mean_rows(z: np.ndarray) -> np.ndarray:
    """Row-wise mean of the two largest values.

    Stouffer aggregates diffuse evidence (low_and_slow: many small signals). Max
    responds to one screaming signal (impossible travel). top2mean sits between them,
    and taking max(Stouffer, top2mean) keeps both behaviours instead of trading one off.
    """
    if z.shape[1] == 1:
        return z[:, 0]
    part = np.partition(z, -2, axis=1)[:, -2:]
    return part.mean(axis=1)


class Fusion:
    """Per-level fusion: signal -> tail probability -> z -> correlated combination."""

    def __init__(self, level: str, rng_seed: int = 0) -> None:
        self.level = level
        self.signals: List[str] = [s for s in LEVEL_SIGNALS[level]
                                   if s not in MITIGATING_SIGNALS]
        # Mitigation is scoped to the level whose evidence it actually speaks to.
        # Applying corroboration across all levels also suppressed brute force and
        # lateral movement (recall 1.000 -> 0.929 / 0.846), because it subtracted from
        # authentication-burst evidence that cohort corroboration says nothing about.
        # Peer-relative reasoning for the entity level is handled instead inside
        # resource_surprisal, which is now an EXCESS over the role expectation.
        self.mitigators: List[str] = [s for s in LEVEL_SIGNALS[level]
                                      if s in MITIGATING_SIGNALS]
        self.ref: Dict[str, np.ndarray] = {}
        self.ztab: Dict[str, np.ndarray] = {}
        self.weights: Optional[np.ndarray] = None
        self.sigma_scale: float = 1.0
        self.calibrator = None
        self.theta_: Optional[float] = None
        self.rng = np.random.default_rng(rng_seed)

    # -- reference ---------------------------------------------------------------
    def fit_reference(self, sig: pd.DataFrame, unflagged: Optional[np.ndarray] = None) -> "Fusion":
        """Build the reference distribution from UNFLAGGED events only.

        Unflagged, never "labelled normal": selecting the reference set by label would be
        leakage. Early on everything is unflagged, which is the correct bootstrap.
        """
        m = np.ones(len(sig), dtype=bool) if unflagged is None else np.asarray(unflagged, dtype=bool)
        sub = sig[m]
        if len(sub) > REF_SAMPLE_CAP:
            idx = self.rng.choice(len(sub), REF_SAMPLE_CAP, replace=False)
            sub = sub.iloc[idx]
        for name in self.signals + self.mitigators:
            r = np.sort(sub[name].to_numpy(dtype=float))
            self.ref[name] = r
            # Precomputed z lookup. z depends on the reference rank alone, so the whole
            # mapping can be tabulated once: z_table[k] is the score for "k reference
            # values >= this one". Calling scipy's norm.ppf per signal per event was the
            # dominant cost on the streaming path -- 20 scalar ppf calls per event.
            n = len(r)
            p = (1.0 + np.arange(n + 1)) / (1.0 + n)
            self.ztab[name] = np.clip(norm.ppf(1.0 - np.clip(p, 1e-9, 1 - 1e-9)),
                                      Z_CLIP_LO, Z_CLIP_HI)

        z = self.to_z(sub)
        # Reliability: a signal that alarms far more often than the target FPR on
        # reference traffic is downweighted. Purely unsupervised.
        w = []
        for j, name in enumerate(self.signals):
            fpr = float((z[:, j] > 2.33).mean())          # 2.33 ~ 1% one-sided
            w.append(SIGNAL_PRIOR.get(name, 0.5) / (1.0 + fpr / TARGET_FPR))
        self.weights = np.asarray(w, dtype=float)

        # Correlation-corrected denominator. Redundant signals inflate w'Sigma w and so
        # stop being counted twice.
        with np.errstate(invalid="ignore"):
            sigma = np.corrcoef(z, rowvar=False)
        sigma = np.nan_to_num(sigma, nan=0.0)
        if sigma.ndim == 0:
            sigma = np.array([[1.0]])
        np.fill_diagonal(sigma, 1.0)
        if self.weights.size == 0:
            self.weights = np.zeros(z.shape[1], dtype=float)
        if sigma.shape[0] != self.weights.shape[0]:
            sigma = np.eye(self.weights.shape[0])
        denom = float(self.weights @ sigma @ self.weights)
        self.sigma_scale = math.sqrt(max(denom, 1e-9))
        return self

    def to_z(self, sig: pd.DataFrame, names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Empirical upper-tail probability against the reference, mapped to a z-score."""
        use = self.signals if names is None else list(names)
        if not use:
            # A level can legitimately end up with no signals -- the ablation disables a
            # whole layer, which empties the long-window level. Return a single all-zero
            # column so the level scores flat rather than crashing on an empty stack.
            return np.zeros((len(sig), 1), dtype=float)
        cols = []
        for name in use:
            r = self.ref[name]
            s = sig[name].to_numpy(dtype=float)
            n_ge = len(r) - np.searchsorted(r, s, side="left")
            cols.append(self.ztab[name][n_ge])
        return np.column_stack(cols)

    def to_z_one(self, sig: Dict[str, float], names: Optional[Sequence[str]] = None
                 ) -> np.ndarray:
        """Single-event z-vector, without building a DataFrame.

        The batch path scores ~10k events/sec; wrapping each event in a one-row
        DataFrame to reuse it dropped the streaming path to 500/sec. The arithmetic is
        identical -- one searchsorted per signal -- so the 20x was pure pandas
        construction overhead on the hot path.
        """
        out = []
        for name in (self.signals if names is None else names):
            r = self.ref[name]
            n_ge = len(r) - int(np.searchsorted(r, float(sig.get(name, 0.0)), side="left"))
            out.append(float(self.ztab[name][n_ge]))
        return np.asarray(out, dtype=float)

    def raw_score_one(self, sig: Dict[str, float]) -> Tuple[float, np.ndarray]:
        z = self.to_z_one(sig)
        stouffer = float(z @ self.weights) / self.sigma_scale
        if z.size >= 2:
            part = np.partition(z, -2)[-2:]
            raw = max(stouffer, float(part.mean()))
        else:
            raw = max(stouffer, float(z[0]) if z.size else 0.0)
        if self.mitigators:
            zm = self.to_z_one(sig, self.mitigators)
            raw -= MITIGATION_WEIGHT * float(np.clip(zm, 0.0, None).max())
        return raw, z

    def raw_score(self, sig: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        z = self.to_z(sig)
        stouffer = (z @ self.weights) / self.sigma_scale
        raw = np.maximum(stouffer, top2mean_rows(z))
        if self.mitigators:
            zm = self.to_z(sig, self.mitigators)
            # Only positive mitigation counts: strong corroboration lowers the score,
            # but its absence must not silently raise one. "No evidence of innocence"
            # is not evidence of guilt -- uncorroborated_new_edges_7d already carries
            # that case explicitly, as an aggregating signal.
            raw = raw - MITIGATION_WEIGHT * np.clip(zm, 0.0, None).max(axis=1)
        return raw, z

    # -- calibration --------------------------------------------------------------
    def fit_calibration(self, raw: np.ndarray, y: np.ndarray) -> "Fusion":
        """The ONLY place labels enter: a monotone 1-D map raw -> P(anomaly)."""
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y).astype(int)
        if y.sum() < 1000:
            lr = LogisticRegression()
            lr.fit(raw.reshape(-1, 1), y)
            self.calibrator = ("platt", lr)
        else:
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
            iso.fit(raw, y)
            self.calibrator = ("isotonic", iso)
        return self

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        """Map the fused score to a probability, preserving a strict total order.

        Isotonic regression is a STEP function: large blocks of events collapse onto an
        identical probability. The top of the alert queue then contains hundreds of ties
        broken by alert id, i.e. alphabetically -- which is not a ranking at all, and put
        six benign service accounts at the head of the queue on the first run.

        The fix is an infinitesimal rank-order tiebreak on the underlying raw score. It
        is monotone (so calibration is preserved to within 1e-6) and it restores the
        finer ordering the evidence already contains inside each isotonic plateau.
        """
        if self.calibrator is None:
            return raw
        kind, model = self.calibrator
        if kind == "platt":
            return model.predict_proba(raw.reshape(-1, 1))[:, 1]
        p = model.predict(raw)
        order = np.argsort(np.argsort(raw))
        return p + 1e-6 * (order / max(1, len(raw) - 1))

    # -- operating point ----------------------------------------------------------
    def fit_raw_reference(self, raw_ref: np.ndarray,
                          cohort: Optional[np.ndarray] = None) -> "Fusion":
        """Reference distribution of the FUSED score, stratified BY COHORT.

        Stratification is not a refinement, it is load-bearing. Cohorts have genuinely
        different score distributions -- service accounts here are the loosest baselines
        in the estate, so their events score systematically below p=0.5 even when
        entirely benign. Any statistic that accumulates evidence over an alert's events
        then compounds that offset: with a single global reference, Fisher's combined
        p-value correlated 0.62 with alert size and handed 17 of the top 20 alerts to
        service accounts, which are 29.8% of traffic.

        Calibrating per cohort makes benign p approximately uniform WITHIN each cohort,
        so what accumulates is deviation from that cohort's own normal. This is the same
        thing as ranking by exceedance over a per-cohort threshold, folded into the
        p-value so everything downstream inherits it.
        """
        raw_ref = np.asarray(raw_ref, dtype=float)
        self.raw_ref_ = np.sort(raw_ref)
        self.raw_ref_by_cohort_: Dict[str, np.ndarray] = {}
        if cohort is not None:
            coh = np.asarray(cohort).astype(str)
            for c in np.unique(coh):
                v = np.sort(raw_ref[coh == c])
                if len(v) >= 200:          # too few to calibrate: fall back to global
                    self.raw_ref_by_cohort_[c] = v
        return self

    def raw_to_p(self, raw: np.ndarray, cohort: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-event upper-tail probability of the fused score against reference traffic.

        Alert-level significance needs a p-value, not a z: an alert spanning n events gets
        n chances to throw a high score, and correcting for that is impossible without
        each event's tail probability.
        """
        raw = np.asarray(raw, dtype=float)
        byc = getattr(self, "raw_ref_by_cohort_", None)
        if cohort is None or not byc:
            r = getattr(self, "raw_ref_", None)
            if r is None or len(r) == 0:
                return np.full(raw.shape, 0.5)
            n_ge = len(r) - np.searchsorted(r, raw, side="left")
            return (1.0 + n_ge) / (1.0 + len(r))

        coh = np.asarray(cohort).astype(str)
        out = np.empty(raw.shape, dtype=float)
        glob = self.raw_ref_
        for c in np.unique(coh):
            m = coh == c
            r = byc.get(c, glob)
            n_ge = len(r) - np.searchsorted(r, raw[m], side="left")
            out[m] = (1.0 + n_ge) / (1.0 + len(r))
        return out

    def fit_threshold(self, score_ref: np.ndarray, fpr: float = TARGET_FPR) -> float:
        """theta = the (1-fpr) quantile of the score over REFERENCE-NORMAL traffic.

        Not a daily quantile of the mixed stream. A quantile of the mixed stream moves
        whenever attack volume moves, so "top 1%" would mean something different every
        day; and on a zero-attack day it manufactures a full budget of false positives.
        A fixed false-positive rate on the negative class is stable by construction.
        """
        self.theta_ = float(np.quantile(score_ref, 1.0 - fpr))
        return self.theta_


# --------------------------------------------------------------------------------------
# L6 explanation
# --------------------------------------------------------------------------------------


def explain(z_row: np.ndarray, signal_names: Sequence[str], raw_row: Dict[str, float],
            role: str = "", top_k: int = 3) -> List[Dict[str, object]]:
    """Rank the contributing signals for one event and render analyst-facing text.

    This is why the architecture is layered. Every fused input is already a named,
    signed, unit-ed quantity, so the explanation is a sort -- not a post-hoc attribution
    over an opaque scorer.
    """
    order = np.argsort(-z_row)
    out: List[Dict[str, object]] = []
    for j in order[:top_k]:
        name = signal_names[j]
        if z_row[j] <= 0.5:
            continue
        sg: Signal = SIGNAL_BY_NAME[name]
        val = float(raw_row.get(name, 0.0))
        try:
            text = sg.template.format(v=val, role=role or "peer")
        except (KeyError, IndexError):
            text = "%s = %.2f" % (name, val)
        out.append({"signal": name, "z": float(z_row[j]), "value": val, "text": text})
    return out


# Evidence -> attack-type attribution. Each type is defined by the signals that
# characterise it, and the predicted type is whichever rule captures the most of the
# alert's actual evidence.
#
# This is deliberately RULE-BASED rather than learned. A classifier trained on labels
# from our own generator would be learning our injection code: seven attack types are
# produced by roughly seven knobs, each wired to roughly one signal, so it would recover
# the wiring diagram and report near-perfect accuracy that means nothing. Attributing
# from named evidence is both more honest and consistent with keeping the fusion
# unsupervised. (`src/classify.py` does fit a model on the evidence vector, for the
# confusion matrix and the 7-vs-6-class comparison; this is the operational path.)
# Signals are WEIGHTED within each rule, and each rule is scored by its weighted MEAN
# rather than its sum -- otherwise a rule simply wins by listing more signals.
#
# The weights encode what actually defines each type. Brute force and credential stuffing
# both produce a failing source IP, so `fail_rate_ip` cannot separate them; what does is
# that stuffing sprays MANY entities from few IPs while brute force hammers one. Hence
# `ip_entity_fanout` carries the decisive weight for stuffing, and `burst_ratio` (rate
# against the entity's own norm) for brute force.
TYPE_RULES: Tuple[Tuple[str, Dict[str, float]], ...] = (
    ("impossible_travel",   {"geo_velocity": 2.0, "country_novelty": 1.0}),
    ("brute_force",         {"fail_rate_entity": 1.5, "burst_ratio": 1.5,
                             "ip_entity_fanout": -1.0}),
    ("credential_stuffing", {"ip_entity_fanout": 2.5, "fail_rate_ip": 1.0,
                             "ip_fail_ratio": 1.0}),
    ("device_spoofing",     {"fingerprint_mismatch": 2.0}),
    # lateral_movement vs low_and_slow both show unusual resource access; what separates
    # them is TEMPO. Lateral movement is a burst -- an operator working a session, with
    # anomalous command sequences. low_and_slow is gradual accumulation, off-hours, with
    # no burst at all. The long-window signals carry that distinction, so they are
    # weighted above the resource signals the two classes share.
    # lateral_movement vs low_and_slow both show unusual resource access; what separates
    # them is TEMPO -- lateral movement is a burst with anomalous command sequences,
    # low_and_slow is gradual and off-hours.
    #
    # A rebalance onto the strongest event-level discriminators (burst_ratio 0.81 vs 1.65,
    # cmd_surprisal 3.93 vs 5.23) was tried and made things WORSE: low_and_slow fell to
    # 1/67 and overall accuracy 0.608 -> 0.547. The reason is the max-over-members
    # aggregation the attribution uses: taking the MAX of burst_ratio across an alert's
    # events washes out precisely the "never bursts" property that defines low_and_slow.
    # Max helps credential stuffing (fan-out peaks on some event) and hurts tempo.
    ("lateral_movement",    {"cmd_surprisal": 2.0, "burst_ratio": 1.5,
                             "resource_surprisal": 0.6, "peer_incongruence": 0.6,
                             "offhours_rate_7d": -0.8}),
    ("low_and_slow",        {"offhours_rate_7d": 2.0, "new_resource_rate_7d": 2.0,
                             "uncorroborated_new_edges_7d": 1.5,
                             "breadth_ratio_7d": 1.0,
                             "burst_ratio": -1.5, "cmd_surprisal": -0.8}),
)


def predict_type(z_by_name: Dict[str, float], min_z: float = 1.0
                 ) -> Tuple[str, float]:
    """Attribute an alert to an attack type from its evidence.

    Returns (type, confidence) where confidence is the winning rule's share of the total
    positive evidence. Low confidence is meaningful and is shown to the analyst rather
    than hidden: diffuse evidence genuinely does not identify a type.
    """
    pos = {k: max(0.0, v) for k, v in z_by_name.items() if v >= min_z}
    if not pos:
        return "unclassified", 0.0

    # Weighted SUM, not mean. Normalising by rule size penalises multi-signal rules and
    # let single-signal `device_spoofing` (one term, weight 2.0) outscore a rule whose
    # evidence was spread across four -- accuracy fell 0.533 -> 0.467. Accumulated
    # evidence is the point; the weights already stop a rule winning on breadth alone.
    scores = {}
    for name, weights in TYPE_RULES:
        scores[name] = sum(w * pos.get(s, 0.0) for s, w in weights.items())

    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return "unclassified", 0.0
    tot = sum(v for v in scores.values() if v > 0)
    return best, (scores[best] / tot if tot > 0 else 0.0)


def risk_band(score: float, theta: float) -> str:
    if score >= theta:
        return "HIGH"
    if score >= 0.5 * theta:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


class Detector:
    """L0-L1 signals -> per-level fusion -> calibrated score + explanation."""

    def __init__(self, seed: int = 0) -> None:
        self.fusions: Dict[str, Fusion] = {lv: Fusion(lv, seed) for lv in LEVELS}
        self.z_cache: Dict[str, np.ndarray] = {}

    def fit(self, sig: pd.DataFrame, y: Optional[np.ndarray] = None,
            calibrate: bool = True, refine_rounds: int = 2,
            cohort: Optional[np.ndarray] = None) -> "Detector":
        """Fit reference distributions, weights, calibration and threshold.

        The reference is refined iteratively. Round 1 treats all traffic as unflagged --
        the correct bootstrap, since nothing has been scored yet. Later rounds rebuild it
        from events the previous round did NOT flag, so attacks stop contaminating the
        very distribution they are being measured against. No labels are involved: the
        mask comes from the detector's own output.
        """
        for lv, fu in self.fusions.items():
            unflagged: Optional[np.ndarray] = None
            for _ in range(max(1, refine_rounds)):
                fu.fit_reference(sig, unflagged)
                raw, _ = fu.raw_score(sig)
                unflagged = raw < np.quantile(raw, 1.0 - TARGET_FPR)
            if calibrate and y is not None:
                fu.fit_calibration(raw, y)
            # theta lives in RAW space, matching what alerts are ranked by.
            #
            # The quantile is taken over ALL events, not over the unflagged subset. Taking
            # the 99th percentile OF the bottom 99% yields the 98th percentile overall, so
            # a threshold advertised as 1% FPR actually admitted 3.9% of events. The
            # unflagged mask is still what builds the reference DISTRIBUTIONS, where
            # excluding attacks is correct; it is wrong for the operating point.
            ref_scores = raw
            ref_mask = unflagged if (unflagged is not None and unflagged.any()) else np.ones(len(raw), bool)
            fu.fit_raw_reference(raw[ref_mask],
                                 cohort[ref_mask] if cohort is not None else None)
            fu.fit_threshold(ref_scores)
        return self

    def score(self, sig: pd.DataFrame, level: str
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (calibrated, raw, z).

        Both are needed and they are used for different things:

          raw         ranks alerts WITHIN a level. Calibration is monotone, so raw gives
                      the same order but with full resolution -- isotonic's step function
                      collapses large blocks onto one probability (0.211 here), which
                      left the top of the queue effectively unranked.
          calibrated  compares ACROSS levels, where raw z-scales are not commensurable,
                      and gives the analyst a probability rather than a bare z.
        """
        fu = self.fusions[level]
        raw, z = fu.raw_score(sig)
        return fu.calibrate(raw), raw, z

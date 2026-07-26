"""L0 deterministic rules and L1 per-entity behavioural baselines.

Computed in ONE streaming pass, deliberately. The evaluation path and the real-time
`score_event` path run the same code, so a latency number measured here is a latency
number that means something, and there is no chance of batch/stream skew.

Every signal is a NAMED, SIGNED, UNIT-ED quantity. That is what makes the explanation
layer nearly free: an alert can say "geo-velocity 4,200 km/h" rather than "feature 37 had
a high SHAP value". A monolithic scorer would have to reconstruct this after the fact,
slowly and vaguely.

State is bounded per entity (capped dictionaries, time-pruned deques), which is what
makes O(1)-per-event streaming honest rather than aspirational.

CAUSALITY: every signal is computed from the entity's history BEFORE the current event,
then the state is updated. Fitting a profile over an entity's whole timeline and scoring
events against it would be leakage that inflates or deflates results unpredictably, and
cannot be repaired after the fact.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sequence import NGramSurprisal, event_tokens

# --------------------------------------------------------------------------------------
# Signal registry -- names, levels and analyst-facing templates for L6
# --------------------------------------------------------------------------------------


class Signal:
    def __init__(self, name: str, level: str, template: str, unit: str = "") -> None:
        self.name = name
        self.level = level
        self.template = template
        self.unit = unit


SIGNALS: Tuple[Signal, ...] = (
    # ---- L0: deterministic, exact, perfectly explainable ----
    Signal("geo_velocity", "entity", "travel at {v:,.0f} km/h since previous access", "km/h"),
    Signal("fail_rate_entity", "entity", "{v:.0f} failed authentications in 15 min", "count"),
    Signal("fail_rate_ip", "ip", "{v:.0f} failed authentications from this IP in 15 min", "count"),
    Signal("fingerprint_mismatch", "entity", "{v:.0f} device fingerprint field(s) changed", "fields"),
    # Activity burst RELATIVE TO THIS ENTITY'S OWN habitual rate -- not absolute volume.
    # A service account emitting 200 events an hour is doing its job; an account that
    # normally emits two doing the same thing is the attack. Absolute size cannot tell
    # these apart, and normalising it away destroys brute force, whose entire signature
    # is rate.
    Signal("burst_ratio", "entity", "{v:.0f}x this entity's normal event rate", "ratio"),
    # ---- L1: per-entity statistical baseline ----
    Signal("hour_surprisal", "entity", "access at an unusual hour ({v:.1f} bits)", "bits"),
    Signal("ip_novelty", "entity", "source IP never seen for this entity", "flag"),
    Signal("country_novelty", "entity", "first access from this country", "flag"),
    Signal("resource_surprisal", "entity", "unusual resource for this entity ({v:.1f} bits)", "bits"),
    Signal("duration_z", "entity", "session duration {v:.1f} sigma from normal", "sigma"),
    Signal("ncmd_z", "entity", "command count {v:.1f} sigma from normal", "sigma"),
    Signal("peer_incongruence", "entity", "resource is unusual for the {role} role ({v:.1f} bits)", "bits"),
    Signal("auth_method_novelty", "entity", "authentication method changed", "flag"),
    # ---- IP level: credential stuffing is invisible per-entity ----
    Signal("ip_entity_fanout", "ip", "{v:.0f} distinct entities from this IP in 1 h", "entities"),
    Signal("ip_fail_ratio", "ip", "{v:.0%} of this IP's recent attempts failed", "ratio"),
    # ---- L2 sequence ----
    Signal("cmd_surprisal", "entity", "unusual action sequence ({v:.1f} bits/token)", "bits"),
    # ---- long window: low_and_slow needs days, not seconds ----
    Signal("new_resource_rate_7d", "long", "{v:.0f} previously unseen resources in 7 days", "count"),
    Signal("offhours_rate_7d", "long", "{v:.0f} off-hours accesses in 7 days", "count"),
    Signal("breadth_ratio_7d", "long", "{v:.0%} of new-resource accesses were single touches", "ratio"),
    # ---- L3 entity-resource graph ----
    # The discriminator that separates a legitimate insider expanding their footprint
    # from an attacker doing the same thing. Legitimate access growth is rarely a
    # singleton: a whole team is onboarded within a fortnight. Attacker growth is alone.
    Signal("uncorroborated_new_edges_7d", "long",
           "{v:.0f} new resource(s) in 7 days that no {role} peer also newly accessed", "count"),
    # MITIGATING: evidence FOR benignity. Subtracted from the risk score rather than
    # added, because an analyst who sees five teammates granted the same access in the
    # same week concludes "onboarding", not "intrusion". Without an exculpatory path the
    # fusion can only ever accumulate suspicion.
    Signal("corroboration_7d", "long",
           "{v:.0f} new resource(s) also newly accessed by 2+ {role} peers", "count"),
)

SIGNAL_NAMES: Tuple[str, ...] = tuple(s.name for s in SIGNALS)
SIGNAL_BY_NAME: Dict[str, Signal] = {s.name: s for s in SIGNALS}

# Cold-start shrinkage: weight on the entity's own history is n/(n+N0).
COLD_START_N0 = 50.0
COLD_START_MIN_EVENTS = 25       # below this, alerts are marked LOW CONFIDENCE

MAX_TRACKED_IPS = 96
MAX_TRACKED_RESOURCES = 384
FAIL_WINDOW_S = 15 * 60
IP_WINDOW_S = 60 * 60
LONG_WINDOW_S = 7 * 24 * 3600


def _prune(dq: Deque[Tuple[float, object]], now: float, window: float) -> None:
    while dq and now - dq[0][0] > window:
        dq.popleft()


def _cap(counts: Dict[str, float], cap: int) -> None:
    """Keep per-entity state bounded: drop the least-used keys when over capacity."""
    if len(counts) <= cap:
        return
    for k in sorted(counts, key=counts.get)[: len(counts) - cap]:
        del counts[k]


EARTH_R_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------------------
# Per-entity state
# --------------------------------------------------------------------------------------


class EntityState:
    __slots__ = ("n", "hours", "ips", "countries", "resources", "res_total",
                 "dur_mean", "dur_m2", "ncmd_mean", "ncmd_m2",
                 "last_lat", "last_lon", "last_ts", "devices", "auth_methods",
                 "fail_times", "new_res_times", "offhours_times", "res_touch", "ev_times",
                 "first_ts", "quarantined", "pending",
                 "seq_n", "seq_mean", "seq_m2")

    def __init__(self) -> None:
        self.n = 0.0
        self.hours = [0.0] * 24
        self.ips: Dict[str, float] = {}
        self.countries: Dict[str, float] = {}
        self.resources: Dict[str, float] = {}
        self.res_total = 0.0
        self.dur_mean = 0.0
        self.dur_m2 = 0.0
        self.ncmd_mean = 0.0
        self.ncmd_m2 = 0.0
        self.last_lat: Optional[float] = None
        self.last_lon: Optional[float] = None
        self.last_ts: Optional[float] = None
        self.devices: Dict[str, Dict[Tuple[str, str, str, str], float]] = {}
        self.auth_methods: Dict[str, float] = {}
        self.fail_times: Deque[Tuple[float, object]] = deque()
        self.new_res_times: Deque[Tuple[float, object]] = deque()
        self.offhours_times: Deque[Tuple[float, object]] = deque()
        self.ev_times: Deque[Tuple[float, object]] = deque()
        self.res_touch: Dict[str, float] = {}
        self.first_ts: Optional[float] = None
        # Drift defence (exercised in the poisoning experiment).
        self.quarantined = False
        self.pending: Dict[str, float] = {}
        # L2 running surprisal stats (Welford), for per-entity normalisation.
        self.seq_n = 0.0
        self.seq_mean = 0.0
        self.seq_m2 = 0.0


class IPState:
    __slots__ = ("entity_times", "attempt_times", "fail_times")

    def __init__(self) -> None:
        self.entity_times: Deque[Tuple[float, object]] = deque()
        self.attempt_times: Deque[Tuple[float, object]] = deque()
        self.fail_times: Deque[Tuple[float, object]] = deque()


# --------------------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------------------


class FeatureExtractor:
    """Single-pass, causal, bounded-state signal computation.

    Priors (hour-of-day by entity_type, resource distribution by role) are learned
    ONLINE from the stream itself. They are population statistics the detector is
    entitled to compute; nothing is read from the generator.
    """

    def __init__(self, learn_baseline: bool = True, use_sequence: bool = True) -> None:
        self.entities: Dict[str, EntityState] = {}
        self.ips: Dict[str, IPState] = {}
        self.type_hours: Dict[str, List[float]] = {}
        self.type_n: Dict[str, float] = {}
        self.role_res: Dict[str, Dict[str, float]] = {}
        self.role_n: Dict[str, float] = {}
        self.learn_baseline = learn_baseline
        # L2
        self.use_sequence = use_sequence
        self.ngram = NGramSurprisal() if use_sequence else None
        # L3: entity-resource bipartite graph. res_first_seen[resource][entity] = when
        # that edge first appeared, which is what makes cohort corroboration computable.
        self.res_first_seen: Dict[str, Dict[str, float]] = {}
        self.entity_role: Dict[str, str] = {}
        self.role_members: Dict[str, set] = {}

    def _corroboration(self, st: "EntityState", eid: str, role: str, ts: float
                       ) -> Tuple[float, float]:
        """New edges this entity acquired recently that no same-role peer also acquired.

        Causal: only peers whose first access to the resource is already in the past
        count. A legitimate onboarding still scores low because teammates are typically
        granted access before or alongside the individual.
        """
        n_alone = 0.0
        n_corr = 0.0
        for t_new, res in st.new_res_times:
            owners = self.res_first_seen.get(res)
            if not owners:
                n_alone += 1.0
                continue
            peers = 0
            for other, t_first in owners.items():
                if other == eid or self.entity_role.get(other) != role:
                    continue
                if abs(t_first - t_new) <= LONG_WINDOW_S:
                    peers += 1
                    if peers >= 2:
                        break
            if peers >= 2:
                n_corr += 1.0
            else:
                n_alone += 1.0
        return n_alone, n_corr

    # -- priors -------------------------------------------------------------------
    def _hour_prior(self, etype: str, hour: int) -> float:
        h = self.type_hours.get(etype)
        if not h:
            return 1.0 / 24.0
        tot = self.type_n.get(etype, 0.0)
        return (h[hour] + 1.0) / (tot + 24.0)

    def _role_res_prior(self, role: str, resource: str) -> float:
        d = self.role_res.get(role)
        if not d:
            return 1e-3
        tot = self.role_n.get(role, 0.0)
        return (d.get(resource, 0.0) + 0.5) / (tot + 0.5 * (len(d) + 1))

    # -- main ---------------------------------------------------------------------
    def process(self, ev: dict, update: bool = True, update_decider=None) -> Dict[str, float]:
        """Score one event against history, then fold it into the baselines.

        `update=False` scores without learning -- used for poisoning resistance, where a
        quarantined entity must stop contributing to its own baseline.
        """
        eid = ev["entity_id"]
        st = self.entities.get(eid)
        if st is None:
            st = self.entities[eid] = EntityState()

        ts = ev["_ts"]
        hour = ev["_hour"]
        etype = ev["entity_type"]
        role = ev["role"]
        res = ev["resource_accessed"]
        ip = ev["source_ip"]
        n = st.n
        lam = n / (n + COLD_START_N0)          # empirical-Bayes shrinkage weight

        sig: Dict[str, float] = {}

        # ---------------- L0 ----------------
        v = 0.0
        if st.last_ts is not None and st.last_lat is not None:
            dt_h = max((ts - st.last_ts) / 3600.0, 1.0 / 60.0)
            km = haversine_km(st.last_lat, st.last_lon, ev["geo_lat"], ev["geo_lon"])
            if km > 25.0:                      # ignore GeoIP jitter within a metro area
                v = km / dt_h
        sig["geo_velocity"] = min(v, 20000.0)

        _prune(st.fail_times, ts, FAIL_WINDOW_S)
        sig["fail_rate_entity"] = float(len(st.fail_times))

        _prune(st.ev_times, ts, FAIL_WINDOW_S)
        recent = float(len(st.ev_times))
        if st.first_ts is not None and n >= 20:
            elapsed = max(ts - st.first_ts, 1.0)
            expected = max(n * (FAIL_WINDOW_S / elapsed), 0.5)   # this entity's own rate
            sig["burst_ratio"] = recent / expected
        else:
            sig["burst_ratio"] = 0.0

        ipst = self.ips.get(ip)
        if ipst is None:
            ipst = self.ips[ip] = IPState()
        _prune(ipst.fail_times, ts, FAIL_WINDOW_S)
        _prune(ipst.attempt_times, ts, IP_WINDOW_S)
        _prune(ipst.entity_times, ts, IP_WINDOW_S)
        sig["fail_rate_ip"] = float(len(ipst.fail_times))
        sig["ip_entity_fanout"] = float(len({e for _, e in ipst.entity_times}))
        att = len(ipst.attempt_times)
        sig["ip_fail_ratio"] = (len(ipst.fail_times) / att) if att >= 5 else 0.0

        # Compare against the device's MODAL fingerprint, not the last one seen. Storing
        # last-seen means the first spoofed event overwrites the baseline and every
        # subsequent event in the same campaign looks consistent -- which cut
        # device_spoofing campaign recall to 0.14. A real asset inventory holds a stable
        # expected fingerprint, and the modal observation is the unsupervised equivalent.
        seen = st.devices.get(ev["device_id"])
        cur = (ev["device_os"], ev["device_mac"], ev["device_firmware"], ev["device_protocol"])
        if not seen:
            sig["fingerprint_mismatch"] = 0.0
        else:
            modal = max(seen, key=seen.get)
            sig["fingerprint_mismatch"] = float(sum(1 for a, b in zip(modal, cur) if a != b))

        # ---------------- L1 ----------------
        p_ent_hour = (st.hours[hour] + 0.5) / (n + 12.0) if n > 0 else 1.0 / 24.0
        p_hour = lam * p_ent_hour + (1 - lam) * self._hour_prior(etype, hour)
        sig["hour_surprisal"] = -math.log2(max(p_hour, 1e-9))

        # Novelty flags are scaled by the shrinkage weight: for an entity with almost no
        # history, "never seen before" is nearly vacuous and must not fire at full
        # strength. This is what stops cold-start entities flooding the alert budget.
        seen_ip = st.ips.get(ip, 0.0)
        sig["ip_novelty"] = lam if seen_ip == 0.0 else 0.0

        sig["country_novelty"] = (1.0 * lam) if ev["geo_country"] not in st.countries else 0.0
        sig["auth_method_novelty"] = (1.0 * lam) if (
            st.auth_methods and ev["auth_method"] not in st.auth_methods) else 0.0

        p_ent_res = ((st.resources.get(res, 0.0) + 0.5) / (st.res_total + 0.5 * 64)
                     if st.res_total > 0 else 1e-3)
        p_res = lam * p_ent_res + (1 - lam) * self._role_res_prior(role, res)
        # Peer-relative: how unusual is this resource for the ROLE, independent of the
        # individual? This is the strongest legitimate discriminator between an insider
        # drifting toward their team's tools and an attacker reaching outward.
        role_bits = -math.log2(max(self._role_res_prior(role, res), 1e-9))
        sig["peer_incongruence"] = role_bits
        # EXCESS surprisal: how much more surprising is this resource for THIS entity
        # than it already is for their whole role?
        #
        # Raw per-entity surprisal treats "new to you but standard for your team" and
        # "new to you and unheard-of for your team" identically, which is precisely the
        # insider_drift / lateral_movement confusion. Subtracting the role expectation
        # encodes the joint condition that actually matters: unusual for you AND unusual
        # for your peers. Floored at 0 -- a resource being *common* for the role is not
        # evidence of innocence for an entity that has never touched it.
        sig["resource_surprisal"] = max(0.0, -math.log2(max(p_res, 1e-9)) - role_bits)

        if n >= 5:
            sd = math.sqrt(max(st.dur_m2 / max(n - 1, 1), 1e-6))
            sig["duration_z"] = abs(math.log1p(ev["session_duration"]) - st.dur_mean) / max(sd, 0.15)
            sc = math.sqrt(max(st.ncmd_m2 / max(n - 1, 1), 1e-6))
            sig["ncmd_z"] = abs(ev["_ncmd"] - st.ncmd_mean) / max(sc, 0.75)
        else:
            sig["duration_z"] = 0.0
            sig["ncmd_z"] = 0.0

        # ---------------- long window ----------------
        _prune(st.new_res_times, ts, LONG_WINDOW_S)
        _prune(st.offhours_times, ts, LONG_WINDOW_S)
        sig["new_resource_rate_7d"] = float(len(st.new_res_times))
        sig["offhours_rate_7d"] = float(len(st.offhours_times))
        # Breadth vs depth: reconnaissance touches many resources once each; a person
        # settling into a new project returns to the same few repeatedly.
        recent_new = [r for _, r in st.new_res_times]
        if recent_new:
            singles = sum(1 for r in set(recent_new) if st.res_touch.get(r, 0.0) <= 1.0)
            sig["breadth_ratio_7d"] = singles / max(1, len(set(recent_new)))
        else:
            sig["breadth_ratio_7d"] = 0.0

        n_alone, n_corr = self._corroboration(st, eid, role, ts)
        sig["uncorroborated_new_edges_7d"] = n_alone
        sig["corroboration_7d"] = n_corr

        # ---------------- L2 sequence ----------------
        toks: List[str] = []
        if self.ngram is not None:
            toks = event_tokens(ev)
            s_bits = self.ngram.surprisal(eid, etype, toks)
            # Normalised against this entity's OWN history, with a floor on the spread.
            # Ultra-repetitive service accounts have near-zero variance, and without the
            # floor their z explodes and one account floods the entire alert budget.
            if st.seq_n >= 8:
                sd = math.sqrt(max(st.seq_m2 / max(st.seq_n - 1, 1), 1e-6))
                sig["cmd_surprisal"] = (s_bits - st.seq_mean) / max(sd, 0.5)
            else:
                sig["cmd_surprisal"] = 0.0
        else:
            sig["cmd_surprisal"] = 0.0

        cold = n < COLD_START_MIN_EVENTS
        sig["_n_history"] = n
        sig["_cold_start"] = 1.0 if cold else 0.0

        # ---------------- update ----------------
        # The decision to LEARN from an event is taken after scoring it, in one pass.
        # Scoring and updating cannot be split into two calls: the windowed deques below
        # are appended unconditionally, so a second pass would double-count them.
        if update_decider is not None:
            update = bool(update_decider(sig, ev, st))

        # OBSERVATION state always advances, even for a quarantined entity.
        #
        # This split matters more than it looks. The windows below (new_res_times,
        # offhours_times, last position) are what the DETECTION signals read; the
        # baseline (habitual hours, known IPs, known resources, duration stats) is what
        # "normal" means. Freezing both together made quarantine self-defeating: a
        # quarantined attacker stopped accruing new-resource events, so
        # new_resource_rate_7d fell to zero and its score DROPPED. Quarantine made the
        # attack invisible instead of protecting the baseline from it.
        if res not in st.resources and (not st.new_res_times or st.new_res_times[-1][1] != res):
            st.new_res_times.append((ts, res))
        if p_hour < 0.012:
            st.offhours_times.append((ts, 1))
        st.res_touch[res] = st.res_touch.get(res, 0.0) + 1.0
        st.last_lat, st.last_lon, st.last_ts = ev["geo_lat"], ev["geo_lon"], ts
        st.ev_times.append((ts, 1))
        if st.first_ts is None:
            st.first_ts = ts
        _cap(st.res_touch, MAX_TRACKED_RESOURCES)

        if update and not st.quarantined:
            self._update(st, ev, ts, hour, etype, role, res, ip, p_hour)
            if self.ngram is not None and toks:
                self.ngram.update(eid, etype, toks)
                b = self.ngram.surprisal(eid, etype, toks)
                d = b - st.seq_mean
                st.seq_n += 1.0
                st.seq_mean += d / st.seq_n
                st.seq_m2 += d * (b - st.seq_mean)
            self.entity_role[eid] = role
            owners = self.res_first_seen.get(res)
            if owners is None:
                owners = self.res_first_seen[res] = {}
            if eid not in owners:
                owners[eid] = ts
        if ev["auth_result"] == "failure":
            st.fail_times.append((ts, 1))
            ipst.fail_times.append((ts, 1))
        ipst.attempt_times.append((ts, 1))
        ipst.entity_times.append((ts, eid))
        return sig

    def _update(self, st, ev, ts, hour, etype, role, res, ip, p_hour) -> None:
        n = st.n
        if st.first_ts is None:
            st.first_ts = ts
        st.hours[hour] += 1.0
        st.ips[ip] = st.ips.get(ip, 0.0) + 1.0
        st.countries[ev["geo_country"]] = st.countries.get(ev["geo_country"], 0.0) + 1.0
        st.auth_methods[ev["auth_method"]] = st.auth_methods.get(ev["auth_method"], 0.0) + 1.0
        st.resources[res] = st.resources.get(res, 0.0) + 1.0
        st.res_total += 1.0
        fp = (ev["device_os"], ev["device_mac"], ev["device_firmware"], ev["device_protocol"])
        seen = st.devices.get(ev["device_id"])
        if seen is None:
            seen = st.devices[ev["device_id"]] = {}
        seen[fp] = seen.get(fp, 0.0) + 1.0
        if len(seen) > 6:                      # keep per-device state bounded
            for k in sorted(seen, key=seen.get)[: len(seen) - 6]:
                del seen[k]
        # Welford, on log duration (durations are lognormal by construction).
        x = math.log1p(ev["session_duration"])
        d = x - st.dur_mean
        st.dur_mean += d / (n + 1)
        st.dur_m2 += d * (x - st.dur_mean)
        c = float(ev["_ncmd"])
        dc = c - st.ncmd_mean
        st.ncmd_mean += dc / (n + 1)
        st.ncmd_m2 += dc * (c - st.ncmd_mean)
        st.n = n + 1

        _cap(st.ips, MAX_TRACKED_IPS)
        _cap(st.resources, MAX_TRACKED_RESOURCES)

        if self.learn_baseline:
            h = self.type_hours.get(etype)
            if h is None:
                h = self.type_hours[etype] = [0.0] * 24
            h[hour] += 1.0
            self.type_n[etype] = self.type_n.get(etype, 0.0) + 1.0
            rr = self.role_res.get(role)
            if rr is None:
                rr = self.role_res[role] = {}
            rr[res] = rr.get(res, 0.0) + 1.0
            self.role_n[role] = self.role_n.get(role, 0.0) + 1.0


# --------------------------------------------------------------------------------------
# Batch driver
# --------------------------------------------------------------------------------------


def prepare(events: pd.DataFrame) -> List[dict]:
    """Convert the event frame into lightweight dicts, sorted causally by time."""
    df = events.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])
    recs = df.to_dict("records")
    secs = (ts.astype("int64") // 10**9).to_numpy()
    hours = ts.dt.hour.to_numpy()
    for i, r in enumerate(recs):
        r["_ts"] = float(secs[i])
        r["_hour"] = int(hours[i])
        cs = r.get("command_sequence")
        r["_ncmd"] = len(cs) if isinstance(cs, (list, np.ndarray)) else 0
    return recs


def extract(events: pd.DataFrame, extractor: Optional[FeatureExtractor] = None
            ) -> Tuple[pd.DataFrame, FeatureExtractor]:
    """Run the streaming extractor over a whole dataset and return a signal frame."""
    fx = extractor or FeatureExtractor()
    recs = prepare(events)
    out: List[Dict[str, float]] = []
    for r in recs:
        s = fx.process(r)
        s["event_id"] = r["event_id"]
        out.append(s)
    sig = pd.DataFrame(out)
    keep = ["event_id"] + list(SIGNAL_NAMES) + ["_n_history", "_cold_start"]
    return sig[keep], fx

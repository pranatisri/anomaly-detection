"""Synthetic access-log generator with injected attack taxonomy.

DESIGN RULE (CM1): there is exactly ONE function that turns intent into an event --
``emit_event``. Benign traffic, benign confounders and attacks all go through it. An
attack changes only the *parameters* handed to it, never the mechanism. This is what
stops attacks being identifiable by incidental artefacts (float precision, MAC string
format, timestamp jitter, or position in the file) rather than by behaviour.

The generator writes two files:

    events.parquet   only config.FEATURE_COLUMNS -- what the detector may see
    labels.parquet   label, campaign_id, campaign_type, confounder -- sealed, eval only

Documented behavioural assumptions
----------------------------------
* Entities have stable habitual hours, a home city, a small set of home IPs, a role-
  conditioned resource affinity, and one device. Humans are weekday- and daytime-biased;
  service accounts run continuously with a nightly batch; edge devices are flat and
  narrow.
* "Normal" is per-entity, not global. A 3 a.m. login is unremarkable for a service
  account and notable for a salesperson.
* Attacks are campaigns, not points: they have a start, an end, and many member events.
* Benign life is messy. Confounders are injected at 3-5x the attack rate specifically so
  that "unusual" and "malicious" are not synonyms in this dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import (
    CONFOUNDERS,
    FEATURE_COLUMNS,
    INSIDER_DRIFT,
    LABEL_COLUMNS,
    NORMAL,
)
from world import (
    CITIES,
    COMMANDS_BY_TYPE,
    EDGE_FUNCTIONS,
    HIGH_VALUE,
    Device,
    EntityProfile,
    IPPools,
    build_ip_pools,
    build_resources,
    haversine_km,
    make_entities,
    sample_commands,
)

SECOND = np.timedelta64(1, "s")


def lerp(a: float, b: float, t: float) -> float:
    """Interpolate an attack knob from blatant (t=0) to overlapping-with-benign (t=1)."""
    return float(a + (b - a) * t)


def _pick(cum: np.ndarray, u: float) -> int:
    """O(log n) categorical sample from a precomputed cumulative distribution."""
    return int(np.searchsorted(cum, u * cum[-1]))


# --------------------------------------------------------------------------------------
# The single emission path
# --------------------------------------------------------------------------------------


@dataclass
class EmitParams:
    """Everything a caller may override. Anything left None is drawn from the entity's
    own habitual profile, so benign and malicious events are produced by identical code."""
    timestamp: pd.Timestamp
    source_ip: Optional[str] = None
    geo: Optional[Tuple[float, float, str]] = None
    resource: Optional[str] = None
    auth_method: Optional[str] = None
    auth_result: Optional[str] = None
    duration_mult: float = 1.0
    n_commands: Optional[int] = None
    recon_bias: float = 0.02          # benign baseline: recon verbs are never exclusive
    device: Optional[Device] = None
    # Metadata only -- has no influence on any observable field.
    label: str = NORMAL
    campaign_id: Optional[str] = None
    campaign_type: Optional[str] = None
    confounder: Optional[str] = None


class Emitter:
    """Holds the shared world state and produces events. One instance per dataset."""

    def __init__(
        self,
        rng: np.random.Generator,
        resources: List[str],
        rtype: Dict[str, str],
        pools: IPPools,
        window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ) -> None:
        self.rng = rng
        self.resources = resources
        self.rtype = rtype
        self.pools = pools
        self.rows: List[dict] = []
        self._cum_cache: Dict[str, np.ndarray] = {}
        # Events outside the observation window are dropped rather than emitted. Campaign
        # helpers scatter events with relative offsets (peers onboarding at -7 days, a
        # 30-day exfiltration starting on day 118) and without clipping those produced a
        # handful of stray rows before and after the main window -- boundary days holding
        # 3 events instead of 3000, which is itself a tell.
        self.window = window
        self.n_clipped = 0
        self._reg_ips: Optional[List[str]] = None
        self._reg_cum: Optional[np.ndarray] = None

    def build_ip_registry(self) -> None:
        """Snapshot the addresses that actually carry benign traffic, by volume.

        Called once after the benign baseline exists. Attack traffic then draws mostly
        from this registry, which makes the attacker's pool mix identical to the real
        one by construction. Hand-picking pool weights instead left the first octet of
        the source IP predicting the label at 0.68 AUC, because a hand-chosen mix is
        never quite the empirical mix. It is also the realistic story: attackers work
        from compromised hosts, shared VPN egress and proxies that real users also use.
        """
        from collections import Counter
        c = Counter(r["source_ip"] for r in self.rows)
        if not c:
            return
        self._reg_ips = list(c.keys())
        w = np.asarray([c[i] for i in self._reg_ips], dtype=float)
        self._reg_cum = np.cumsum(w / w.sum())

    def any_ip(self, prof: EntityProfile, registry_prob: float = 0.75) -> str:
        """Draw from the shared allocator (CM4).

        Attack traffic uses this, and so does benign traffic, so no pool is
        attack-exclusive and a hash bucket of the source IP carries no label information.
        """
        rng = self.rng
        if self._reg_cum is not None and rng.random() < registry_prob:
            return self._reg_ips[_pick(self._reg_cum, rng.random())]
        r = rng.random()
        if r < 0.35:
            pool = self.pools.cloud
        elif r < 0.65:
            pool = self.pools.consumer[prof.home_city]
        elif r < 0.85:
            pool = self.pools.office[prof.home_city]
        else:
            pool = self.pools.vpn[prof.home_city]
        return pool[int(rng.integers(0, len(pool)))]

    def attack_ip(self, prof: EntityProfile, reuse_prob: float = 0.35) -> str:
        """Attackers frequently operate from infrastructure the victim already uses --
        a stolen session on the corporate VPN, or a foothold on the same office range."""
        if self.rng.random() < reuse_prob and prof.home_ips:
            return prof.home_ips[int(self.rng.integers(0, len(prof.home_ips)))]
        return self.any_ip(prof)

    def resource_cum(self, prof: EntityProfile) -> np.ndarray:
        c = self._cum_cache.get(prof.entity_id)
        if c is None:
            c = np.cumsum(prof.resource_p)
            self._cum_cache[prof.entity_id] = c
        return c

    def emit(self, prof: EntityProfile, p: EmitParams) -> Optional[dict]:
        rng = self.rng

        if self.window is not None:
            if p.timestamp < self.window[0] or p.timestamp >= self.window[1]:
                self.n_clipped += 1
                return None

        resource = p.resource
        if resource is None:
            resource = self.resources[_pick(self.resource_cum(prof), rng.random())]
        rt = self.rtype.get(resource, "endpoint")

        if p.geo is not None:
            lat, lon, country = p.geo
        else:
            # Ordinary jitter around the home city (GeoIP is not exact).
            lat = prof.home_lat + rng.normal(0, 0.09)
            lon = prof.home_lon + rng.normal(0, 0.09)
            country = prof.home_country

        ip = p.source_ip
        if ip is None:
            if rng.random() < prof.new_ip_rate:
                # Benign sessions do sometimes arrive from a brand-new address, which is
                # what keeps "unseen IP" a weak signal rather than an oracle (CM4).
                ip = self.any_ip(prof)
            else:
                ip = prof.home_ips[int(rng.integers(0, len(prof.home_ips)))]

        auth_method = p.auth_method or prof.auth_method
        if p.auth_result is not None:
            auth_result = p.auth_result
        else:
            auth_result = "failure" if rng.random() < prof.fail_rate else "success"

        dur = float(rng.lognormal(prof.duration_mu, prof.duration_sigma) * p.duration_mult)
        dur = max(0.4, min(dur, 86400.0))

        n_cmd = p.n_commands
        if n_cmd is None:
            n_cmd = int(1 + rng.poisson(prof.cmd_len_lambda))
        cmds = sample_commands(rng, rt, n_cmd, recon_bias=p.recon_bias)

        dev = p.device or prof.device
        nonce = int(rng.integers(0, 2**62))
        raw = "%s|%s|%s|%s|%d" % (prof.entity_id, p.timestamp, resource, ip, nonce)
        eid = hashlib.blake2b(raw.encode("utf-8"), digest_size=10).hexdigest()

        row = {
            "event_id": eid,
            "entity_id": prof.entity_id,
            "entity_type": prof.entity_type,
            "role": prof.role,
            "timestamp": p.timestamp,
            "source_ip": ip,
            "geo_lat": round(lat, 4),
            "geo_lon": round(lon, 4),
            "geo_country": country,
            "resource_accessed": resource,
            "resource_type": rt,
            "auth_method": auth_method,
            "auth_result": auth_result,
            "session_duration": round(dur, 2),
            "command_sequence": cmds,
            "device_id": dev.device_id,
            "device_os": dev.os,
            "device_firmware": dev.firmware,
            "device_mac": dev.mac,
            "device_protocol": dev.protocol,
            # sealed
            "label": p.label,
            "campaign_id": p.campaign_id,
            "campaign_type": p.campaign_type,
            "confounder": p.confounder,
        }
        self.rows.append(row)
        prof.known_resources.add(resource)
        return row


# --------------------------------------------------------------------------------------
# Benign baseline
# --------------------------------------------------------------------------------------


def generate_benign(em: Emitter, profiles: Sequence[EntityProfile], start: pd.Timestamp,
                    n_days: int) -> None:
    """Per-entity habitual behaviour, sampled with noise."""
    rng = em.rng
    for prof in profiles:
        hour_cum = np.cumsum(prof.hour_weights)
        for day in range(prof.first_day, n_days):
            dow = (start + pd.Timedelta(days=day)).dayofweek
            expected = prof.events_per_day * prof.weekday_weights[dow] * 7.0
            n = int(rng.poisson(max(0.05, expected)))
            if n <= 0:
                continue
            us = rng.random(n)
            mins = rng.integers(0, 3600, size=n)
            auth = rng.random(n)
            for j in range(n):
                hour = _pick(hour_cum, us[j])
                ts = start + pd.Timedelta(days=day, hours=int(hour), seconds=int(mins[j]))
                if auth[j] < 0.22:
                    # Every session begins with an authentication, so sso/read is the
                    # single highest-volume resource in the estate. Without this it was
                    # touched almost only by brute force and credential stuffing, making
                    # the resource name itself a give-away -- the audit caught it as a
                    # 0.83-AUC signal on resource-string LENGTH.
                    em.emit(prof, EmitParams(timestamp=ts, resource="sso/read",
                                             n_commands=int(1 + (rng.random() < 0.3))))
                else:
                    em.emit(prof, EmitParams(timestamp=ts))


# --------------------------------------------------------------------------------------
# Benign confounders (CM5) -- negatives engineered to trip each detector layer
# --------------------------------------------------------------------------------------


def _far_city(rng: np.random.Generator, prof: EntityProfile, min_km: float = 2000.0):
    for _ in range(30):
        name, lat, lon, country = CITIES[int(rng.integers(0, len(CITIES)))]
        if haversine_km(prof.home_lat, prof.home_lon, lat, lon) >= min_km:
            return name, lat, lon, country
    return CITIES[0]


def _peers(rng, profiles, prof, k):
    same = [p for p in profiles if p.role == prof.role and p.entity_id != prof.entity_id
            and p.entity_type == prof.entity_type]
    if not same:
        return []
    k = min(k, len(same))
    idx = rng.choice(len(same), size=k, replace=False)
    return [same[int(i)] for i in np.atleast_1d(idx)]


def _fleet_device(rng: np.random.Generator, profiles: Sequence[EntityProfile],
                  prof: EntityProfile, idx: int) -> Device:
    """Mint a replacement device whose OS/firmware already exist in the fleet.

    make_device() invents a firmware string like `fw-2.9.15` from a 800-way random space,
    so a device minted only inside a campaign carries a version no other device runs --
    an attack-only token. The audit caught exactly that at 58 events, 100% malicious.

    This is used by low_and_slow, insider_drift AND the benign device_refresh confounder,
    which matters: if only the attack cloned and the benign paths minted, the asymmetry
    would simply be a tell in the opposite direction.
    """
    from world import make_device
    dev = make_device(rng, prof.entity_type, idx)
    same = [q for q in profiles if q.entity_type == prof.entity_type
            and q.entity_id != prof.entity_id]
    if same:
        donor = same[int(rng.integers(0, len(same)))].device
        dev = Device(device_id=dev.device_id, os=donor.os, firmware=donor.firmware,
                     mac=dev.mac, protocol=donor.protocol)
    return dev


def inject_confounder(em: Emitter, kind: str, prof: EntityProfile,
                      profiles: Sequence[EntityProfile], start: pd.Timestamp,
                      n_days: int) -> None:
    """Benign behaviour that looks wrong. Label stays NORMAL; `confounder` records which."""
    rng = em.rng
    day0 = _start_day(rng, prof, n_days)

    t0 = start + pd.Timedelta(days=day0)
    P = lambda **kw: EmitParams(label=NORMAL, confounder=kind, **kw)  # noqa: E731

    if kind == "business_travel":
        # Trips L0 geo-velocity. Effective speed ~800 km/h including airport time, which
        # is exactly where the hard impossible_travel cases sit at delta=1.
        _, lat, lon, country = _far_city(rng, prof)
        dist = haversine_km(prof.home_lat, prof.home_lon, lat, lon)
        hours = dist / rng.normal(800.0, 120.0)
        vpn_ip = str(rng.choice(em.pools.vpn[prof.home_city]))
        em.emit(prof, P(timestamp=t0))
        tarr = t0 + pd.Timedelta(hours=float(max(1.0, hours)))
        for d in range(int(rng.integers(2, 6))):
            for _ in range(int(rng.integers(2, 8))):
                ts = tarr + pd.Timedelta(days=d, hours=float(rng.uniform(0, 12)))
                em.emit(prof, P(timestamp=ts, geo=(lat + rng.normal(0, .05),
                                                   lon + rng.normal(0, .05), country),
                                source_ip=vpn_ip))

    elif kind == "password_reset_storm":
        # Trips L0 failed-auth rate: a human who forgot their password.
        for i in range(int(rng.integers(5, 14))):
            ts = t0 + pd.Timedelta(minutes=float(rng.uniform(0, 25)))
            em.emit(prof, P(timestamp=ts, auth_result="failure", resource="sso/read",
                            n_commands=1))
        em.emit(prof, P(timestamp=t0 + pd.Timedelta(minutes=30), auth_result="success",
                        resource="sso/read"))

    elif kind == "cert_retry_loop":
        # Trips L0 failed-auth rate: an expired certificate retrying on a timer.
        for i in range(int(rng.integers(20, 70))):
            ts = t0 + pd.Timedelta(minutes=float(i * rng.uniform(1.5, 4.0)))
            em.emit(prof, P(timestamp=ts, auth_result="failure",
                            auth_method="certificate", n_commands=1))

    elif kind in ("os_patch", "device_refresh"):
        # Trips L0 fingerprint mismatch. os_patch keeps the MAC (same hardware);
        # device_refresh changes everything (new laptop) -- the same observable shape as
        # device_spoofing, which is the point.
        d = prof.device
        if kind == "os_patch":
            newdev = Device(device_id=d.device_id,
                            os=_bump_version(rng, d.os),
                            firmware=_bump_version(rng, d.firmware),
                            mac=d.mac, protocol=d.protocol)
        else:
            newdev = _fleet_device(rng, profiles, prof, int(rng.integers(90000, 99999)))
        for d_ in range(int(rng.integers(3, 10))):
            for _ in range(int(rng.integers(1, 5))):
                ts = t0 + pd.Timedelta(days=d_, hours=float(rng.uniform(0, 14)))
                em.emit(prof, P(timestamp=ts, device=newdev))
        prof.device = newdev

    elif kind in ("oncall_rotation", "maintenance_window"):
        # Trips L1 habitual-hour: legitimate work at 3 a.m.
        for d_ in range(int(rng.integers(4, 9))):
            for _ in range(int(rng.integers(2, 7))):
                ts = t0 + pd.Timedelta(days=d_, hours=float(rng.uniform(0.5, 5.0)))
                em.emit(prof, P(timestamp=ts))

    elif kind == "project_onboarding":
        # Trips L1 novelty and L3 new-edge -- and is CORROBORATED, which is the signal
        # that legitimately separates it from lateral movement.
        peers = _peers(rng, profiles, prof, 6)
        pool = [r for r in em.resources if r not in prof.known_resources]
        if not pool:
            return
        newres = [str(x) for x in rng.choice(pool, size=min(5, len(pool)), replace=False)]
        for d_ in range(int(rng.integers(6, 15))):
            for r in newres:
                if rng.random() < 0.55:
                    ts = t0 + pd.Timedelta(days=d_, hours=float(rng.uniform(8, 19)))
                    em.emit(prof, P(timestamp=ts, resource=r))
        for pe in peers:                     # the rest of the team, same fortnight
            for r in newres:
                if rng.random() < 0.5:
                    ts = t0 + pd.Timedelta(days=float(rng.uniform(-7, 7)),
                                           hours=float(rng.uniform(8, 19)))
                    em.emit(pe, P(timestamp=ts, resource=r))

    elif kind == "ci_automation_burst":
        # Trips L2 surprisal and session-duration: a release pipeline.
        for i in range(int(rng.integers(60, 200))):
            ts = t0 + pd.Timedelta(minutes=float(i * rng.uniform(0.2, 1.2)))
            em.emit(prof, P(timestamp=ts, duration_mult=float(rng.uniform(2, 8)),
                            n_commands=int(rng.integers(8, 20))))

    elif kind in ("shared_jump_host", "office_nat"):
        # Trips the IP-level credential-stuffing detector: many entities, one IP, but
        # all succeeding. Volume, not failure, is what makes this benign.
        ip = (str(rng.choice(em.pools.cloud)) if kind == "shared_jump_host"
              else str(rng.choice(em.pools.office[prof.home_city])))
        crowd = _peers(rng, profiles, prof, int(rng.integers(12, 40))) + [prof]
        for pe in crowd:
            for _ in range(int(rng.integers(1, 6))):
                ts = t0 + pd.Timedelta(hours=float(rng.uniform(0, 10)))
                em.emit(pe, P(timestamp=ts, source_ip=ip, auth_result="success"))

    elif kind == "credential_rotation":
        # Trips auth_method-change logic: a service account moving password -> certificate.
        newm = "certificate" if prof.auth_method != "certificate" else "token"
        for d_ in range(int(rng.integers(2, 6))):
            for _ in range(int(rng.integers(3, 10))):
                ts = t0 + pd.Timedelta(days=d_, hours=float(rng.uniform(0, 24)))
                em.emit(prof, P(timestamp=ts, auth_method=newm))
        prof.auth_method = newm


def _bump_version(rng: np.random.Generator, s: str) -> str:
    """Nudge a version string the way a patch would, keeping the product identity."""
    if s == "n/a":
        return s
    parts = s.split()
    if len(parts) >= 2 and any(ch.isdigit() for ch in parts[-1]):
        tail = parts[-1].split(".")
        if tail[-1].isdigit():
            tail[-1] = str(int(tail[-1]) + int(rng.integers(1, 4)))
            parts[-1] = ".".join(tail)
            return " ".join(parts)
    return s + "-p%d" % rng.integers(1, 9)


# --------------------------------------------------------------------------------------
# Attack campaigns
# --------------------------------------------------------------------------------------


def stratified_start_days(rng: np.random.Generator, n_campaigns: int, n_days: int,
                          day_weights: Optional[np.ndarray] = None) -> List[int]:
    """One start day per campaign, spread evenly across the window then shuffled.

    Drawing start days iid-uniform leaves gaps when campaigns are sparse. With 35
    campaigns over 62 days one run happened to place none in the first week, and
    "attacks never happen before day 7" is a real, generalisable pattern: row position in
    the file predicted the label at 0.77 AUC and the artefact audit failed.

    Stratifying guarantees coverage regardless of how few campaigns a seed draws, which
    removes the failure mode rather than relying on it not recurring.
    """
    lo, hi = 0, max(2, n_days - 1)
    if n_campaigns <= 0:
        return []

    if day_weights is not None and np.sum(day_weights) > 0:
        # Place campaigns in proportion to BENIGN TRAFFIC VOLUME, not uniformly over the
        # calendar.
        #
        # Benign activity follows a weekly cycle -- users drop to ~15% of weekday volume
        # at weekends -- but calendar-uniform attack placement does not. The result was a
        # weekend attack RATE 4.07x the weekday rate (Sunday 3.8% vs Monday 0.11%), which
        # is a real, generalisable pattern an artefact model can learn: row position in
        # the file scored 0.63 AUC. Attacks now track the traffic they hide in.
        w = np.asarray(day_weights, dtype=float).clip(min=0.0)
        cdf = np.cumsum(w) / w.sum()
        qs = (np.arange(n_campaigns) + rng.random(n_campaigns)) / n_campaigns
        days = [int(np.clip(np.searchsorted(cdf, q), lo, hi)) for q in qs]
    else:
        edges = np.linspace(lo, hi, n_campaigns + 1)
        days = [int(np.clip(rng.uniform(edges[i], edges[i + 1]), lo, hi))
                for i in range(n_campaigns)]
    rng.shuffle(days)
    return days


def _start_day(rng: np.random.Generator, prof: EntityProfile, n_days: int) -> int:
    """Uniform start day across the WHOLE observation window.

    Campaigns are deliberately allowed to start late and be truncated by the end of the
    window -- any real dataset contains attacks still in progress at the cut-off.

    The obvious alternative, reserving enough room for a campaign's full span, is what
    the first version did, and it was wrong: it pushed every 30-day low_and_slow into the
    first third of the window and made attack density decay from 2.2% on day 6 to 0.1% by
    day 66. Row position in the file then predicted the label at 0.88 AUC.
    """
    lo = max(prof.first_day, 1)
    hi = max(lo + 1, n_days - 2)
    return int(rng.integers(lo, hi))


@dataclass
class CampaignRec:
    campaign_id: str
    campaign_type: str
    scope_level: str
    scope_key: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    n_events: int = 0


def inject_attack(em: Emitter, kind: str, cid: str, prof: EntityProfile,
                  profiles: Sequence[EntityProfile], start: pd.Timestamp,
                  n_days: int, delta: float,
                  day0_override: Optional[int] = None) -> Optional[CampaignRec]:
    """Inject one attack campaign. `delta` interpolates blatant -> benign-overlapping."""
    rng = em.rng
    before = len(em.rows)
    day0 = _start_day(rng, prof, n_days) if day0_override is None else max(
        int(day0_override), int(prof.first_day))
    t0 = start + pd.Timedelta(days=day0, hours=float(rng.uniform(0, 24)))
    P = lambda **kw: EmitParams(label=kind, campaign_id=cid, campaign_type=kind, **kw)  # noqa: E731
    scope_level, scope_key = "entity", prof.entity_id

    if kind == "brute_force":
        # delta=0: 60 attempts/min (obvious). delta=1: 3/min (looks like a reset storm).
        rate = lerp(60.0, 3.0, delta)
        n = int(rng.integers(25, 90))
        ip = em.attack_ip(prof, reuse_prob=0.15)
        for i in range(n):
            ts = t0 + pd.Timedelta(minutes=float(i / max(rate, 0.5)))
            em.emit(prof, P(timestamp=ts, source_ip=ip, auth_result="failure",
                            resource="sso/read", n_commands=1))
        if rng.random() < 0.35:               # occasionally they get in
            em.emit(prof, P(timestamp=t0 + pd.Timedelta(minutes=float(n / max(rate, .5)) + 1),
                            source_ip=ip, auth_result="success", resource="sso/read"))

    elif kind == "credential_stuffing":
        # An IP-LEVEL phenomenon: many entities, few source IPs, high failure rate.
        # Invisible to a per-entity detector, which is why the IP level exists.
        n_ips = max(1, int(lerp(1, 4, delta)))
        ips = [em.any_ip(prof) for _ in range(n_ips)]
        # Clamp to the population: a small estate has fewer entities than the nominal
        # 20-70 victim fan-out, and sampling without replacement would raise.
        n_vic = int(min(rng.integers(20, 70), max(2, len(profiles) - 1)))
        victims = list(rng.choice(len(profiles), size=n_vic, replace=False))
        succ_rate = lerp(0.02, 0.18, delta)
        span_h = lerp(1.5, 9.0, delta)
        for vi in victims:
            v = profiles[int(vi)]
            for _ in range(int(rng.integers(1, 5))):
                ts = t0 + pd.Timedelta(hours=float(rng.uniform(0, span_h)))
                em.emit(v, P(timestamp=ts, source_ip=str(rng.choice(ips)),
                             auth_result="success" if rng.random() < succ_rate else "failure",
                             resource="sso/read", n_commands=1))
        scope_level, scope_key = "ip", ips[0]

    elif kind == "impossible_travel":
        # delta=0: 3000 km/h (physically impossible). delta=1: ~900 km/h, which overlaps
        # the business_travel confounder's N(800, 120) distribution.
        kmh = lerp(3000.0, 900.0, delta)
        _, lat, lon, country = _far_city(rng, prof)
        dist = haversine_km(prof.home_lat, prof.home_lon, lat, lon)
        gap_h = dist / kmh
        em.emit(prof, P(timestamp=t0))
        t1 = t0 + pd.Timedelta(hours=float(gap_h))
        ip = em.attack_ip(prof, reuse_prob=0.10)
        for _ in range(int(rng.integers(2, 7))):
            ts = t1 + pd.Timedelta(minutes=float(rng.uniform(0, 90)))
            em.emit(prof, P(timestamp=ts, geo=(lat, lon, country), source_ip=ip))

    elif kind == "lateral_movement":
        # Unusual BREADTH plus recon verbs, reaching peer-inconsistent high-value targets.
        breadth = int(lerp(22, 7, delta))
        recon = lerp(0.55, 0.12, delta)
        pool = [r for r in em.resources if r not in prof.known_resources]
        if not pool:
            return None
        targets = [str(x) for x in rng.choice(pool, size=min(breadth, len(pool)), replace=False)]
        targets += [h for h in HIGH_VALUE if rng.random() < 0.5]
        ip = em.attack_ip(prof, reuse_prob=0.40)
        for i, r in enumerate(targets):
            ts = t0 + pd.Timedelta(minutes=float(i * rng.uniform(2, 25)))
            em.emit(prof, P(timestamp=ts, resource=r, source_ip=ip, recon_bias=recon,
                            n_commands=int(rng.integers(2, 9))))

    elif kind == "device_spoofing":
        # delta=0: OS, firmware, MAC and protocol all wrong. delta=1: one field differs,
        # indistinguishable in shape from the os_patch confounder.
        nfields = max(1, int(round(lerp(4, 1, delta))))
        d = prof.device
        from world import make_device
        alt = make_device(rng, prof.entity_type, int(rng.integers(80000, 89999)))
        os_ = alt.os if nfields >= 1 else d.os
        mac = alt.mac if nfields >= 2 else d.mac
        fw = alt.firmware if nfields >= 3 else d.firmware
        proto = alt.protocol if nfields >= 4 else d.protocol
        # MAC cloning is a real spoofing technique, and it also keeps freshly-minted MACs
        # from being attack-exclusive -- the audit flagged specific MACs as oracles.
        if nfields >= 2 and rng.random() < 0.5:
            victim = profiles[int(rng.integers(0, len(profiles)))]
            mac = victim.device.mac
        # Firmware is cloned from a real device of the same type rather than invented.
        # A minted version string appears nowhere else in the fleet, which made it an
        # attack-only token: the audit caught `fw-2.9.15` at 58 events, 100% malicious.
        # A spoofer claiming a version no device has ever run is also just bad tradecraft.
        if nfields >= 3:
            same = [q for q in profiles
                    if q.entity_type == prof.entity_type and q.device.firmware != "n/a"]
            if same:
                fw = same[int(rng.integers(0, len(same)))].device.firmware
        spoof = Device(device_id=d.device_id, os=os_, firmware=fw, mac=mac, protocol=proto)
        ip = em.attack_ip(prof, reuse_prob=0.30)
        for i in range(int(rng.integers(6, 25))):
            ts = t0 + pd.Timedelta(hours=float(i * rng.uniform(0.3, 3.0)))
            em.emit(prof, P(timestamp=ts, device=spoof, source_ip=ip))

    elif kind == "low_and_slow":
        # Gradual, small, off-hours accumulation. Deliberately BELOW any per-event
        # threshold: this is the attack that per-event baseline filtering cannot stop,
        # and the reason drift defence has to work at entity level.
        span_days = int(lerp(28, 14, delta))
        per_day = lerp(4.0, 1.4, delta)
        pool = [r for r in em.resources if r not in prof.known_resources]
        peer_bad = [r for r in pool if r in HIGH_VALUE] + pool[: max(1, len(pool) // 3)]
        if not peer_bad:
            return None
        # As delta rises the attacker also blends in targets that look ordinary for the
        # victim's role -- a patient adversary picks unremarkable objectives.
        #
        # Without this, target selection is the ONE axis on which low_and_slow and
        # insider_drift never converge, no matter how high delta goes, and a single
        # peer-congruence feature separated them at 0.97 AUC. That is a planted tell of
        # exactly the kind the artefact audit exists to prevent: the two classes are
        # supposed to become genuinely ambiguous at delta=1, not stay trivially separable.
        peer_like: List[str] = []
        for pe in _peers(rng, profiles, prof, 8):
            peer_like.extend([r for r in list(pe.known_resources)[:20]
                              if r not in prof.known_resources])
        blend_p = 0.7 * delta
        # Same nuisance shifts insider_drift gets, at the same rate (D4): if only the
        # attack came with a new device, that would be a planted tell, not a signal.
        dev = prof.device
        if rng.random() < 0.35:
            dev = _fleet_device(rng, profiles, prof, int(rng.integers(70000, 79999)))
        ip = em.any_ip(prof) if rng.random() < 0.40 else None
        for d_ in range(span_days):
            k = int(rng.poisson(per_day))
            for _ in range(k):
                if peer_like and rng.random() < blend_p:
                    r = str(rng.choice(peer_like))     # looks ordinary for the role
                else:
                    r = str(rng.choice(peer_bad))      # breadth-first: ~1 touch each
                ts = t0 + pd.Timedelta(days=d_, hours=float(rng.uniform(0.5, 5.5)))
                em.emit(prof, P(timestamp=ts, resource=r, source_ip=ip, device=dev,
                                n_commands=int(rng.integers(1, 3))))

    else:
        raise ValueError("unknown attack kind %r" % kind)

    added = em.rows[before:]
    if not added:
        return None
    ts_all = [r["timestamp"] for r in added]
    return CampaignRec(cid, kind, scope_level, scope_key, min(ts_all), max(ts_all), len(added))


def inject_insider_drift(em: Emitter, cid: str, prof: EntityProfile,
                         profiles: Sequence[EntityProfile], start: pd.Timestamp,
                         n_days: int, delta: float,
                         day0_override: Optional[int] = None) -> Optional[CampaignRec]:
    """The ambiguous edge case: a legitimate employee slowly expanding their footprint.

    Near-identical to low_and_slow by construction. What legitimately differs:
      * targets move TOWARD the role centroid, not toward peer-inconsistent high value
      * depth, not breadth -- few resources touched repeatedly
      * cohort-corroborated: teammates gain the same access in the same fortnight
      * bursty, work-calendar-shaped ramp rather than a machine-regular one
    It receives the SAME nuisance shifts (new device, new IP) at the same rate, so those
    cannot be used as a shortcut.
    """
    rng = em.rng
    before = len(em.rows)
    day0 = _start_day(rng, prof, n_days) if day0_override is None else max(
        int(day0_override), int(prof.first_day))
    t0 = start + pd.Timedelta(days=day0)
    P = lambda **kw: EmitParams(label=INSIDER_DRIFT, campaign_id=cid,  # noqa: E731
                                campaign_type=INSIDER_DRIFT, **kw)

    # Peer-consistent targets: things this person's colleagues already use.
    peers = _peers(rng, profiles, prof, 8)
    peer_res: List[str] = []
    for pe in peers:
        peer_res.extend(list(pe.known_resources)[:20])
    pool = [r for r in dict.fromkeys(peer_res) if r not in prof.known_resources]
    if len(pool) < 2:
        return None
    targets = [str(x) for x in rng.choice(pool, size=min(4, len(pool)), replace=False)]

    dev = prof.device
    if rng.random() < 0.35:
        dev = _fleet_device(rng, profiles, prof, int(rng.integers(60000, 69999)))
    ip = em.any_ip(prof) if rng.random() < 0.40 else None

    span_days = int(lerp(28, 14, delta))
    for d_ in range(span_days):
        dow = (t0 + pd.Timedelta(days=d_)).dayofweek
        if dow >= 5 and rng.random() < 0.8:      # human ramps follow the work calendar
            continue
        k = int(rng.poisson(2.6 * rng.uniform(0.3, 2.2)))   # bursty, not machine-regular
        for _ in range(k):
            r = str(rng.choice(targets))                    # depth: same few, repeatedly
            ts = t0 + pd.Timedelta(days=d_, hours=float(rng.uniform(8, 19)))
            em.emit(prof, P(timestamp=ts, resource=r, source_ip=ip, device=dev))

    # Corroboration: the rest of the team gains the same access around the same time.
    for pe in peers[:4]:
        for r in targets:
            if rng.random() < 0.45:
                ts = t0 + pd.Timedelta(days=float(rng.uniform(-6, 10)),
                                       hours=float(rng.uniform(8, 19)))
                em.emit(pe, EmitParams(timestamp=ts, resource=r, label=NORMAL))

    added = em.rows[before:]
    if not added:
        return None
    ts_all = [r["timestamp"] for r in added]
    return CampaignRec(cid, INSIDER_DRIFT, "entity", prof.entity_id,
                       min(ts_all), max(ts_all), len(added))


# --------------------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------------------

ATTACK_KINDS = ("brute_force", "credential_stuffing", "impossible_travel",
                "lateral_movement", "device_spoofing", "low_and_slow")


def generate(
    seed: int,
    n_entities: int = 800,
    n_days: int = 120,
    delta: float = 0.5,
    confounder_multiplier: float = 4.0,
    target_contamination: Optional[float] = None,
    randomise: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Generate one dataset.

    Args:
        seed: RNG seed. DEV seeds are 1-5; HOLDOUT seeds are 101-105 and are run once.
        delta: difficulty. 0 = blatant attacks, 1 = attacks that overlap benign behaviour.
        confounder_multiplier: benign confounder campaigns per attack campaign (CM5).
        randomise: vary population size, contamination and the attack roster by seed so
            that nothing about the benchmark can be hardcoded by the detector (CM8).

    Returns:
        (events, labels, campaigns, meta)
    """
    rng = np.random.default_rng(seed)

    if randomise:
        n_entities = int(n_entities * rng.uniform(0.85, 1.15))
        n_days = int(n_days * rng.uniform(0.9, 1.1))
        if target_contamination is None:
            target_contamination = float(rng.uniform(0.005, 0.03))
        kinds = list(ATTACK_KINDS)
        # Some seeds deliberately omit a whole attack type, so we can measure the pure
        # false-positive rate of the detector that hunts it.
        if rng.random() < 0.35:
            kinds.remove(str(rng.choice(kinds)))
    else:
        target_contamination = target_contamination or 0.015
        kinds = list(ATTACK_KINDS)

    start = pd.Timestamp("2026-01-05 00:00:00")
    # Victims are drawn in proportion to how much traffic they generate, not uniformly
    # over entities. Uniform selection made attacks 74% user-generated while benign
    # traffic was only 54% user, because service accounts and edge devices emit far more
    # events per entity. The population mismatch leaked through every entity-type proxy
    # (OS string, auth method), and the artefact audit picked it up.
    resources, rtype, role_dist = build_resources(rng)
    pools = build_ip_pools(rng, [c[0] for c in CITIES])
    profiles = make_entities(rng, n_entities, n_days, resources, role_dist, pools)
    em = Emitter(rng, resources, rtype, pools,
                 window=(start, start + pd.Timedelta(days=n_days)))

    t_start = time.time()
    generate_benign(em, profiles, start, n_days)
    n_benign = len(em.rows)

    em.build_ip_registry()

    # Benign events per day -- the density attacks must follow.
    _bd = np.zeros(n_days + 2, dtype=float)
    for r in em.rows:
        di = int((r["timestamp"] - start).days)
        if 0 <= di < len(_bd):
            _bd[di] += 1.0

    volume = np.array([p.events_per_day * max(1, n_days - p.first_day) for p in profiles])
    victim_cum = np.cumsum(volume / volume.sum())

    def pick_victim(restrict_type: Optional[str] = None) -> EntityProfile:
        if restrict_type is not None:
            cands = [p for p in profiles if p.entity_type == restrict_type]
            if cands:
                w = np.array([p.events_per_day for p in cands])
                return cands[_pick(np.cumsum(w / w.sum()), rng.random())]
        return profiles[_pick(victim_cum, rng.random())]

    # Size the attack budget from the target contamination and the observed benign volume.
    approx_events_per_campaign = 50.0   # measured mean; `actual_contamination` is reported
    n_attack_campaigns = max(len(kinds),
                             int(n_benign * target_contamination / approx_events_per_campaign))
    n_conf_campaigns = int(n_attack_campaigns * confounder_multiplier)

    # Confounders first: they are part of "normal life" and should be in the baseline.
    for i in range(n_conf_campaigns):
        # Pick the VICTIM first (weighted by traffic), then a confounder type that suits
        # it. Choosing the type first concentrated load: four of twelve types were
        # service-account-only, but service accounts are ~16% of the population, so each
        # one carried several times a user's confounder burden. That inflated the whole
        # cohort's baseline variance and hence its alert rate.
        prof = pick_victim()
        if prof.entity_type == "service_account":
            pool = ("cert_retry_loop", "ci_automation_burst", "credential_rotation",
                    "maintenance_window", "os_patch", "device_refresh")
        elif prof.entity_type == "edge_device":
            pool = ("os_patch", "device_refresh", "maintenance_window", "office_nat")
        else:
            pool = ("business_travel", "password_reset_storm", "os_patch",
                    "device_refresh", "oncall_rotation", "project_onboarding",
                    "shared_jump_host", "office_nat")
        kind = str(rng.choice(pool))
        inject_confounder(em, kind, prof, profiles, start, n_days)

    campaigns: List[CampaignRec] = []
    atk_days = stratified_start_days(rng, n_attack_campaigns, n_days, _bd)
    for i in range(n_attack_campaigns):
        kind = kinds[i % len(kinds)]
        rec = inject_attack(em, kind, "CAMP-%05d" % i, pick_victim(), profiles,
                            start, n_days, delta, day0_override=atk_days[i])
        if rec is not None:
            campaigns.append(rec)

    # insider_drift: the edge case, injected at roughly the attack rate so the two
    # gradual-expansion classes are comparably represented.
    n_drift = max(2, n_attack_campaigns // 3)
    drift_days = stratified_start_days(rng, n_drift, n_days, _bd)
    for i in range(n_drift):
        rec = inject_insider_drift(em, "DRIFT-%05d" % i, pick_victim(), profiles,
                                   start, n_days, delta, day0_override=drift_days[i])
        if rec is not None:
            campaigns.append(rec)

    df = pd.DataFrame(em.rows)
    # Global re-sort and content-hash ids, so neither row order nor id carries any signal.
    df = df.sort_values(["timestamp", "event_id"]).reset_index(drop=True)

    cold = {p.entity_id for p in profiles if p.first_day > 0}
    labels = df[["event_id", "label", "campaign_id", "campaign_type", "confounder"]].copy()
    labels["is_cold_start"] = df["entity_id"].isin(cold).to_numpy()

    events = df[[c for c in df.columns if c in FEATURE_COLUMNS]].copy()
    missing = set(FEATURE_COLUMNS) - set(events.columns)
    if missing:
        raise RuntimeError("generator failed to produce feature columns: %s" % sorted(missing))

    camp_df = pd.DataFrame([c.__dict__ for c in campaigns]) if campaigns else pd.DataFrame(
        columns=["campaign_id", "campaign_type", "scope_level", "scope_key",
                 "start_ts", "end_ts", "n_events"])

    n_attack_events = int(labels["label"].isin(ATTACK_KINDS).sum())
    meta = {
        "seed": seed,
        "delta": delta,
        "n_entities": len(profiles),
        "n_days": n_days,
        "n_events": len(df),
        "n_attack_events": n_attack_events,
        "actual_contamination": n_attack_events / max(1, len(df)),
        "target_contamination": target_contamination,
        "n_campaigns": len(campaigns),
        "n_confounder_campaigns": n_conf_campaigns,
        "confounder_multiplier": confounder_multiplier,
        "attack_kinds_present": sorted(set(c.campaign_type for c in campaigns)),
        "n_cold_start_entities": len(cold),
        "gen_seconds": round(time.time() - t_start, 1),
    }
    return events, labels, camp_df, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic access-log dataset.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--entities", type=int, default=800)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--confounder-multiplier", type=float, default=4.0)
    ap.add_argument("--no-randomise", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = ap.parse_args()

    ev, lb, camps, meta = generate(
        seed=args.seed, n_entities=args.entities, n_days=args.days, delta=args.delta,
        confounder_multiplier=args.confounder_multiplier, randomise=not args.no_randomise,
    )
    tag = "seed%d_delta%s" % (args.seed, str(args.delta).replace(".", ""))
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    ev.to_parquet(os.path.join(outdir, "events_%s.parquet" % tag), index=False)
    lb.to_parquet(os.path.join(outdir, "labels_%s.parquet" % tag), index=False)
    camps.to_parquet(os.path.join(outdir, "campaigns_%s.parquet" % tag), index=False)

    print("wrote %s" % tag)
    for k, v in meta.items():
        print("  %-26s %s" % (k, v))


if __name__ == "__main__":
    main()

"""The simulated organisation: roles, resources, geography, IP space, devices.

Everything here is *structural* — it describes an organisation, not an attack. Role
structure in particular is deliberately visible to the detector (config.FEATURE_COLUMNS
includes `role`), because an enterprise really does have an HR system and peer-relative
reasoning is a legitimate detection technique. The rule that makes that safe is that
roles drive **benign** resource assignment too; they are not something attached only to
attacked entities.

Per-entity behavioural parameters (habitual hours, home geo, resource affinity, command
transition matrix, device fingerprint) live in EntityProfile and are HIDDEN — never
serialised into the event stream. See config.CM3 notes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------------------

# (name, lat, lon, country)
CITIES: Tuple[Tuple[str, float, float, str], ...] = (
    ("mumbai", 19.076, 72.877, "IN"),
    ("pune", 18.520, 73.856, "IN"),
    ("bengaluru", 12.972, 77.595, "IN"),
    ("hyderabad", 17.385, 78.487, "IN"),
    ("delhi", 28.614, 77.209, "IN"),
    ("london", 51.507, -0.128, "GB"),
    ("frankfurt", 50.110, 8.682, "DE"),
    ("amsterdam", 52.370, 4.895, "NL"),
    ("dublin", 53.350, -6.260, "IE"),
    ("new_york", 40.713, -74.006, "US"),
    ("chicago", 41.878, -87.630, "US"),
    ("san_jose", 37.339, -121.895, "US"),
    ("austin", 30.267, -97.743, "US"),
    ("singapore", 1.352, 103.820, "SG"),
    ("sydney", -33.868, 151.209, "AU"),
    ("tokyo", 35.690, 139.692, "JP"),
    ("dubai", 25.205, 55.271, "AE"),
    ("sao_paulo", -23.551, -46.633, "BR"),
)

EARTH_R_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance. Used for geo-velocity in both the generator and L0."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------------------
# Roles and resources
# --------------------------------------------------------------------------------------

ROLES: Tuple[str, ...] = (
    "engineering", "finance", "hr", "sales", "operations", "security", "support",
)

RESOURCE_TYPES: Tuple[str, ...] = ("file", "endpoint", "port", "device_function")

# Resource families per role. Each family expands into several concrete resources.
_ROLE_FAMILIES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "engineering": (("git", "endpoint"), ("ci", "endpoint"), ("artifact", "file"),
                    ("staging_db", "port"), ("wiki", "endpoint")),
    "finance":     (("ledger", "endpoint"), ("invoices", "file"), ("payroll", "file"),
                    ("tax_reports", "file"), ("erp", "endpoint")),
    "hr":          (("employee_records", "file"), ("recruiting", "endpoint"),
                    ("benefits", "endpoint"), ("policies", "file")),
    "sales":       (("crm", "endpoint"), ("quotes", "file"), ("contracts", "file"),
                    ("pricing", "file")),
    "operations":  (("monitoring", "endpoint"), ("runbooks", "file"),
                    ("inventory", "endpoint"), ("scheduler", "port")),
    "security":    (("siem", "endpoint"), ("vault", "port"), ("ids_console", "endpoint"),
                    ("audit_logs", "file")),
    "support":     (("tickets", "endpoint"), ("kb", "endpoint"), ("customer_data", "file")),
}

# Resources every role touches. Their presence is what stops "unseen resource" from
# being a perfect oracle, and gives lateral movement somewhere plausible to start.
_SHARED_FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("sso", "endpoint"), ("mail", "endpoint"), ("intranet", "endpoint"),
    ("file_share", "file"), ("print", "device_function"), ("vpn", "port"),
)

# High-value resources: the plausible objective of an intrusion. They belong to real
# roles (finance/hr/security own them legitimately), so accessing them is only
# suspicious relative to the accessing entity's own role.
HIGH_VALUE: Tuple[str, ...] = (
    "payroll/export", "employee_records/bulk", "vault/read", "audit_logs/purge",
    "tax_reports/archive", "customer_data/export", "ledger/adjust",
)

EDGE_FUNCTIONS: Tuple[str, ...] = (
    "sensor/read", "actuator/write", "telemetry/push", "firmware/status",
    "config/get", "diagnostics/run", "valve/set", "meter/report",
)

AUTH_METHODS: Tuple[str, ...] = ("password", "token", "certificate", "biometric")
PROTOCOLS: Tuple[str, ...] = ("https", "ssh", "rdp", "mqtt", "modbus", "ldap")

OS_USER: Tuple[str, ...] = ("Windows 11 23H2", "Windows 11 24H2", "macOS 14.5",
                            "macOS 15.1", "Ubuntu 22.04", "Ubuntu 24.04")
OS_SERVER: Tuple[str, ...] = ("RHEL 9.4", "Ubuntu Server 22.04", "Debian 12",
                              "Windows Server 2022")
OS_EDGE: Tuple[str, ...] = ("OpenWrt 23.05", "Yocto 4.0", "FreeRTOS 10.5",
                            "Zephyr 3.6", "BuildRoot 2024.02")


def build_resources(rng: np.random.Generator) -> Tuple[List[str], Dict[str, str], Dict[str, np.ndarray]]:
    """Build the resource catalogue and P(resource | role).

    Returns:
        resources: ordered catalogue
        rtype: resource -> resource_type
        role_dist: role -> probability vector over `resources`

    The distribution is deliberately NOT disjoint across roles: every role has a small
    tail of probability on other roles' resources. Without that tail, "resource outside
    the role's set" would be a perfect attack oracle rather than a weak signal.
    """
    resources: List[str] = []
    rtype: Dict[str, str] = {}
    owner: Dict[str, Optional[str]] = {}

    for role, fams in _ROLE_FAMILIES.items():
        for fam, typ in fams:
            for suffix in ("read", "list", "write", "search"):
                name = "%s/%s" % (fam, suffix)
                if name not in rtype:
                    resources.append(name)
                    rtype[name] = typ
                    owner[name] = role
    for fam, typ in _SHARED_FAMILIES:
        for suffix in ("read", "list", "write"):
            name = "%s/%s" % (fam, suffix)
            if name not in rtype:
                resources.append(name)
                rtype[name] = typ
                owner[name] = None
    for name in HIGH_VALUE:
        if name not in rtype:
            resources.append(name)
            rtype[name] = "file"
            fam = name.split("/")[0]
            owner[name] = {
                "payroll": "hr", "employee_records": "hr", "vault": "security",
                "audit_logs": "security", "tax_reports": "finance",
                "customer_data": "support", "ledger": "finance",
            }.get(fam)
    for name in EDGE_FUNCTIONS:
        resources.append(name)
        rtype[name] = "device_function"
        owner[name] = "operations"

    idx = {r: i for i, r in enumerate(resources)}
    role_dist: Dict[str, np.ndarray] = {}
    for role in ROLES:
        w = np.full(len(resources), 0.02)          # non-zero tail everywhere
        for r in resources:
            o = owner[r]
            if o == role:
                w[idx[r]] = 6.0                    # core to this role
            elif o is None:
                w[idx[r]] = 3.0                    # shared services
        # A little per-role jitter so roles are not interchangeable.
        w = w * rng.uniform(0.7, 1.3, size=w.shape)
        role_dist[role] = w / w.sum()
    return resources, rtype, role_dist


# --------------------------------------------------------------------------------------
# IP space
# --------------------------------------------------------------------------------------


@dataclass
class IPPools:
    """A single shared IP allocator (CM4).

    Attacker traffic is drawn from the SAME pools as benign traffic, including NAT'd
    office ranges and consumer ISP blocks, and some campaigns deliberately reuse the
    victim's own known-good IP. Symmetrically some benign sessions arrive from brand-new
    IPs. If attacker IPs came from a disjoint pool, any "unseen IP" flag would be a
    perfect oracle and every reported metric would be meaningless.
    """
    office: Dict[str, List[str]]      # city -> NAT egress IPs (shared by many entities)
    consumer: Dict[str, List[str]]    # city -> home broadband range
    vpn: Dict[str, List[str]]         # city -> VPN egress
    cloud: List[str]                  # datacentre / hosting ranges

    def all_ips(self) -> List[str]:
        out: List[str] = list(self.cloud)
        for d in (self.office, self.consumer, self.vpn):
            for v in d.values():
                out.extend(v)
        return out


def build_ip_pools(rng: np.random.Generator, cities: Sequence[str]) -> IPPools:
    def block(a: int, b: int, n: int) -> List[str]:
        return ["%d.%d.%d.%d" % (a, b, rng.integers(0, 256), rng.integers(1, 255))
                for _ in range(n)]

    office, consumer, vpn = {}, {}, {}
    for i, c in enumerate(cities):
        office[c] = block(10, 10 + i, 3)        # few IPs, many entities -> NAT
        consumer[c] = block(49, 30 + i, 60)
        vpn[c] = block(103, 20 + i, 4)
    cloud = block(172, 67, 40) + block(64, 90, 40)
    return IPPools(office=office, consumer=consumer, vpn=vpn, cloud=cloud)


# --------------------------------------------------------------------------------------
# Entity profiles (HIDDEN — never serialised into the event stream)
# --------------------------------------------------------------------------------------


@dataclass
class Device:
    device_id: str
    os: str
    firmware: str
    mac: str
    protocol: str


@dataclass
class EntityProfile:
    """Per-entity behavioural parameters. NONE of this reaches the detector.

    The detector must re-estimate all of it from observed events; handing any of it over
    directly would be sufficient-statistic inversion (circularity path C1) and would make
    recall trend to 1.0 purely as history accumulates.
    """
    entity_id: str
    entity_type: str
    role: str
    home_city: str
    home_lat: float
    home_lon: float
    home_country: str

    hour_weights: np.ndarray          # 24-vector, habitual login hours
    weekday_weights: np.ndarray       # 7-vector
    events_per_day: float
    resource_p: np.ndarray            # affinity over the resource catalogue
    auth_method: str
    duration_mu: float                # lognormal
    duration_sigma: float
    cmd_len_lambda: float
    home_ips: List[str]
    device: Device
    first_day: int = 0                # >0 means the entity joins mid-window (cold start)
    new_ip_rate: float = 0.02         # benign sessions from a brand-new IP
    fail_rate: float = 0.02

    # Mutable simulation state (still hidden).
    known_resources: set = field(default_factory=set)


def _mac(rng: np.random.Generator) -> str:
    return ":".join("%02x" % rng.integers(0, 256) for _ in range(6))


def make_device(rng: np.random.Generator, entity_type: str, idx: int) -> Device:
    if entity_type == "user":
        os_ = str(rng.choice(OS_USER))
        proto = str(rng.choice(("https", "ssh", "rdp"), p=(0.7, 0.2, 0.1)))
        fw = "n/a"
    elif entity_type == "service_account":
        os_ = str(rng.choice(OS_SERVER))
        proto = str(rng.choice(("https", "ssh", "ldap"), p=(0.6, 0.3, 0.1)))
        fw = "n/a"
    else:
        os_ = str(rng.choice(OS_EDGE))
        proto = str(rng.choice(("mqtt", "modbus", "https"), p=(0.5, 0.3, 0.2)))
        fw = "fw-%d.%d.%d" % (rng.integers(1, 5), rng.integers(0, 10), rng.integers(0, 20))
    return Device(device_id="dev-%05d" % idx, os=os_, firmware=fw, mac=_mac(rng), protocol=proto)


def _hour_weights(rng: np.random.Generator, entity_type: str) -> np.ndarray:
    h = np.arange(24)
    if entity_type == "user":
        centre = rng.normal(11.0, 2.0)
        w = np.exp(-0.5 * ((h - centre) / rng.uniform(2.0, 3.2)) ** 2)
        w += 0.55 * np.exp(-0.5 * ((h - (centre + 5.5)) / 2.0) ** 2)   # afternoon peak
        w += 0.02                                                       # thin night tail
    elif entity_type == "service_account":
        w = np.full(24, 1.0)                                            # round the clock
        w[rng.integers(0, 24)] += 3.0                                   # a nightly batch
    else:
        w = np.full(24, 1.0)                                            # edge devices: flat
    return w / w.sum()


def make_entities(
    rng: np.random.Generator,
    n_entities: int,
    n_days: int,
    resources: Sequence[str],
    role_dist: Dict[str, np.ndarray],
    pools: IPPools,
    cold_start_frac: float = 0.06,
) -> List[EntityProfile]:
    """Build the entity population with hidden behavioural profiles."""
    n_res = len(resources)
    profiles: List[EntityProfile] = []
    type_p = (0.72, 0.16, 0.12)     # user / service_account / edge_device

    for i in range(n_entities):
        etype = str(rng.choice(("user", "service_account", "edge_device"), p=type_p))
        role = str(rng.choice(ROLES))
        city_i = int(rng.integers(0, len(CITIES)))
        cname, clat, clon, ccountry = CITIES[city_i]

        # Resource affinity: role distribution, sharpened per entity so individuals have
        # habits narrower than their role's, then renormalised.
        base = role_dist[role].copy()
        # Service accounts are the TIGHTEST baselines in any real estate: a handful of
        # endpoints, one or two source IPs, machine-regular timing. Giving them the same
        # profile spread as humans made them the loosest instead -- one had 40 distinct
        # resources and 42 distinct source IPs -- and a loose profile produces a
        # high-variance baseline, which produces systematically elevated anomaly scores.
        # Measured: mean benign per-event z was 0.281 for service accounts against 0.052
        # for users, and since alert scores accumulate over an alert's events, that offset
        # was multiplied by sqrt(n) and handed them 17 of the top 20 alerts.
        exponent = rng.uniform(4.5, 7.0) if etype == "service_account" else rng.uniform(1.6, 2.6)
        sharp = base ** exponent
        # Retention probability scales with the role's affinity, so an entity reliably
        # keeps its role's core resources (including the high-value ones its role
        # legitimately owns) and only sometimes keeps the tail. A flat retention mask
        # here made HIGH_VALUE resources effectively attack-exclusive, which the
        # artefact audit correctly flagged as an oracle.
        keep_p = np.clip(base / base.max(), 0.06, 0.92)
        if etype == "service_account":
            keep_p = keep_p * 0.10          # a handful of endpoints, not dozens
        keep = rng.random(n_res) < keep_p
        sharp = sharp * (0.05 + keep)
        if etype == "edge_device":
            mask = np.zeros(n_res)
            edge_idx = [j for j, r in enumerate(resources) if r in EDGE_FUNCTIONS]
            mask[edge_idx] = 1.0
            sharp = sharp * 0.02 + mask * rng.uniform(0.5, 1.5, size=n_res)
        # Relative floor. An ABSOLUTE floor of 1e-9 silently destroyed the sharpening:
        # base values are ~1e-2, so base**6 is ~1e-14 and the floor clamped nearly every
        # resource to the same value -- producing an almost UNIFORM distribution. Service
        # accounts ended up touching 148 distinct resources, the opposite of intended.
        sharp = sharp / max(sharp.max(), 1e-300)
        sharp = np.maximum(sharp, 1e-6)
        resource_p = sharp / sharp.sum()

        # Rates are calibrated so the medium default (800 entities x 120 days) lands near
        # 1M events, which is what the CPU-only budget supports end to end.
        if etype == "user":
            rate = float(rng.uniform(4, 18))
            auth = str(rng.choice(("password", "token", "biometric"), p=(0.55, 0.35, 0.10)))
            dmu, dsig = rng.uniform(4.8, 6.2), rng.uniform(0.5, 0.9)
            cmdlam = rng.uniform(3.0, 7.0)
            wd = np.array([1.0, 1.0, 1.0, 1.0, 0.95, 0.18, 0.12])
        elif etype == "service_account":
            rate = float(rng.uniform(10, 36))
            auth = str(rng.choice(("token", "certificate"), p=(0.6, 0.4)))
            # Machine-regular: tight session durations and tight command counts, so the
            # baseline is genuinely narrow rather than merely centred.
            dmu, dsig = rng.uniform(2.5, 4.0), rng.uniform(0.10, 0.22)
            cmdlam = rng.uniform(2.0, 3.0)
            wd = np.ones(7)
        else:
            rate = float(rng.uniform(6, 24))
            auth = str(rng.choice(("certificate", "token"), p=(0.75, 0.25)))
            dmu, dsig = rng.uniform(1.5, 3.0), rng.uniform(0.2, 0.5)
            cmdlam = rng.uniform(1.0, 2.5)
            wd = np.ones(7)
        wd = wd * rng.uniform(0.9, 1.1, size=7)

        # Home IPs: office NAT (shared with colleagues) and/or consumer broadband.
        # Every pool must carry real benign traffic. If attackers were the only users of
        # the cloud/hosting ranges, a hash bucket of the source IP would be a perfect
        # oracle -- which is exactly what the artefact audit caught on the first run.
        ips: List[str] = []
        if etype == "user":
            ips += list(rng.choice(pools.office[cname], size=1))
            if rng.random() < 0.7:
                ips += list(rng.choice(pools.consumer[cname], size=int(rng.integers(1, 3))))
            if rng.random() < 0.40:      # remote access via a corporate proxy / gateway
                ips += list(rng.choice(pools.cloud, size=1))
            if rng.random() < 0.30:
                ips += list(rng.choice(pools.vpn[cname], size=1))
        elif etype == "service_account":
            # One or two fixed addresses. No roaming, no office NAT.
            ips += list(rng.choice(pools.cloud, size=int(rng.integers(1, 3))))
        else:
            # A physical device sits on one gateway. It does not roam.
            ips += list(rng.choice(pools.office[cname], size=1))

        first_day = 0
        if rng.random() < cold_start_frac:
            first_day = int(rng.integers(int(n_days * 0.45), int(n_days * 0.92)))

        profiles.append(EntityProfile(
            entity_id="%s-%05d" % ({"user": "usr", "service_account": "svc",
                                    "edge_device": "dev"}[etype], i),
            entity_type=etype,
            role=role,
            home_city=cname, home_lat=clat, home_lon=clon, home_country=ccountry,
            hour_weights=_hour_weights(rng, etype),
            weekday_weights=wd / wd.sum(),
            events_per_day=rate,
            resource_p=resource_p,
            auth_method=auth,
            duration_mu=dmu, duration_sigma=dsig,
            cmd_len_lambda=cmdlam,
            home_ips=[str(x) for x in ips],
            device=make_device(rng, etype, i),
            first_day=first_day,
            # Only humans roam. A per-event new-IP rate of 0.005-0.05 over ~900 events
            # gave edge devices a median of 24 distinct source IPs (worst: 53) -- the
            # same realism defect service accounts had, and for the same reason: a rate
            # tuned for people applied to machines.
            new_ip_rate=float(rng.uniform(0.005, 0.05) if etype == "user"
                              else rng.uniform(0.0003, 0.003)),
            fail_rate=float(rng.uniform(0.005, 0.04) if etype == "user"
                            else rng.uniform(0.001, 0.008)),
        ))
    return profiles


# --------------------------------------------------------------------------------------
# Command vocabulary
# --------------------------------------------------------------------------------------

# Ordinary actions, by resource type. Recon-flavoured verbs (whoami, net_group, ...) are
# available to BENIGN traffic too, at low probability -- otherwise their appearance would
# be a planted attack tell rather than a signal the model has to weigh.
COMMANDS_BY_TYPE: Dict[str, Tuple[str, ...]] = {
    "file":            ("open", "read", "list", "search", "download", "save", "close",
                        "copy", "rename", "stat"),
    "endpoint":        ("login", "query", "view", "update", "export", "search", "logout",
                        "refresh", "post"),
    "port":            ("connect", "auth", "select", "fetch", "commit", "disconnect",
                        "ping"),
    "device_function": ("poll", "read", "report", "ack", "calibrate", "reset", "status"),
}

RECON_COMMANDS: Tuple[str, ...] = (
    "whoami", "net_group_domain", "list_shares", "enum_users", "list_acl",
    "port_scan", "dump_creds", "list_admins",
)


def sample_commands(
    rng: np.random.Generator,
    resource_type: str,
    n: int,
    recon_bias: float = 0.0,
) -> List[str]:
    """Sample an ordered command list.

    `recon_bias` in [0, 1] raises the probability of recon-flavoured verbs. Benign
    traffic uses a small non-zero bias so these tokens are never attack-exclusive.
    """
    base = COMMANDS_BY_TYPE.get(resource_type, COMMANDS_BY_TYPE["endpoint"])
    n = max(1, n)
    # Index with integers rather than rng.choice: choice() coerces the sequence to an
    # array on every call, which dominates runtime at ~1M events x ~5 commands each.
    idx = rng.integers(0, len(base), size=n)
    if recon_bias <= 0.0:
        return [base[i] for i in idx]
    draws = rng.random(n)
    ridx = rng.integers(0, len(RECON_COMMANDS), size=n)
    return [RECON_COMMANDS[ridx[j]] if draws[j] < recon_bias else base[idx[j]]
            for j in range(n)]

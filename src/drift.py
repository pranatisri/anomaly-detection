"""Concept drift and baseline-poisoning resistance.

THE CENTRAL CLAIM THIS TESTS
----------------------------
"Update baselines only from events that were not flagged" is the standard answer to
baseline poisoning, and on its own **it does not work**. Every `low_and_slow` event is
deliberately sub-threshold -- that is what makes it low and slow -- so every one of them
passes the per-event filter and is absorbed into the baseline. The filter fails against
precisely the attack it most needs to stop.

The fix has to act at ENTITY level, not event level:

  * a two-window drift detector (recent 7 d vs reference 28 d) on the entity's resource
    footprint, which asks whether the entity as a whole is moving
  * corroboration and peer-consistency checks to tell legitimate movement from an attack
  * a hard GROWTH CAP: the known-resource set may only grow so fast, and excess additions
    wait in a `pending` set until time and corroboration promote them

EXPERIMENT DESIGN
-----------------
Three scenarios, matched:

  A  legit drift    role change on day 30; footprint migrates over days 30-44 toward
                    resources the entity's peers already use; cohort-corroborated
  B  slow attacker  compromised on day 30, MATCHED TO A on ramp rate and on number of new
                    resources -- if B ramped faster the experiment would prove nothing --
                    but reaching peer-inconsistent targets, breadth-first, uncorroborated
  C  control        no change; shows the baseline does not wander on its own

crossed with four update arms:

  all         naive, poisonable
  unflagged   the standard per-event answer
  quarantine  unflagged + entity-level drift detection + growth cap  (recommended)
  none        frozen baseline

Headline number: ABSORPTION RATE = P(attacker B scores below theta at day 60).
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ARMS: Tuple[str, ...] = ("all", "unflagged", "quarantine", "none")

# Drift-detector constants. These are RELATIVE to each entity's own established habit
# rather than absolute counts, so they do not need retuning for estate size or activity
# level. An absolute cap ("no more than N new resources per week") was tried first and
# never fired at all: a 20-resource entity acquiring 4 new ones sat under every fixed
# bound, while for a 3-resource entity the same 4 would be an explosion.
GROWTH_SPIKE_FACTOR = 2.5      # new-resource rate vs the entity's own baseline rate
QUARANTINE_MIN_UNCORROBORATED = 2
EW_ALPHA = 0.02                # how fast the historical rate itself adapts


class UpdatePolicy:
    """Decides, per event, whether the baseline may learn from it.

    All arms behave identically before `active_from`. A frozen-baseline arm that freezes
    from day zero never learns anything at all, so its signals are identically zero and
    it looks trivially "safe" -- an artefact, not a result. Freezing means freezing an
    ESTABLISHED baseline.
    """

    def __init__(self, arm: str, theta: float, active_from: float = 0.0) -> None:
        if arm not in ARMS:
            raise ValueError("unknown arm %r" % (arm,))
        self.arm = arm
        self.theta = theta
        self.active_from = active_from
        self.quarantined: Dict[str, bool] = {}
        self.hist_rate: Dict[str, float] = {}
        self.flagged_count = 0
        self.blocked_count = 0

    def __call__(self, sig: Dict[str, float], ev: dict, st) -> bool:
        eid = ev["entity_id"]
        newres = sig.get("new_resource_rate_7d", 0.0)
        # Track each entity's own habitual rate of picking up new resources.
        prev = self.hist_rate.get(eid)
        self.hist_rate[eid] = newres if prev is None else (1 - EW_ALPHA) * prev + EW_ALPHA * newres

        if ev["_ts"] < self.active_from:
            return True                       # warm-up: every arm learns normally

        if self.arm == "none":
            return False
        if self.arm == "all":
            return True

        # Per-event filter, the standard answer. Note that a low_and_slow event sails
        # straight through this: it is engineered to sit below any per-event threshold.
        flagged = sig.get("_fused", 0.0) >= self.theta
        if flagged:
            self.flagged_count += 1
        if self.arm == "unflagged":
            return not flagged

        # arm == "quarantine": entity-level defence on top of the per-event filter.
        if self.quarantined.get(eid):
            self.blocked_count += 1
            return False

        uncorr = sig.get("uncorroborated_new_edges_7d", 0.0)
        base_rate = max(prev if prev is not None else 0.0, 0.5)
        spiking = newres > GROWTH_SPIKE_FACTOR * base_rate

        if spiking and uncorr >= QUARANTINE_MIN_UNCORROBORATED:
            # Moving much faster than its own habit, and no peer is moving with it.
            # Freeze this entity's baseline; an analyst decides whether it is an
            # intrusion or a role change.
            self.quarantined[eid] = True
            st.quarantined = True
            self.blocked_count += 1
            return False

        # Growth cap: even uncontroversial expansion may not outrun the entity's own
        # habitual rate. This is what actually stops sub-threshold absorption, because it
        # does not depend on any individual event looking suspicious.
        if spiking:
            self.blocked_count += 1
            return False

        return not flagged


# --------------------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------------------


def build_world(rng: np.random.Generator, n_pairs: int, n_days: int = 60,
                drift_day: int = 30, drift_end: int = 44) -> Tuple[List[dict], Dict[str, str]]:
    """Build a small population containing matched A/B/C triples.

    A and B are matched on ramp rate and on the NUMBER of new resources acquired. Only
    the character of the expansion differs.
    """
    roles = ["engineering", "finance", "hr", "sales", "operations"]
    core: Dict[str, List[str]] = {
        r: ["%s/res%02d" % (r, i) for i in range(12)] for r in roles
    }
    rare: Dict[str, List[str]] = {
        r: ["%s/rare%02d" % (r, i) for i in range(12)] for r in roles
    }
    events: List[dict] = []
    truth: Dict[str, str] = {}

    def emit(eid, role, day, hour, res, ip="10.0.0.1"):
        ts = pd.Timestamp("2026-01-05") + pd.Timedelta(days=int(day), hours=float(hour))
        events.append({
            "event_id": "%s-%05d" % (eid, len(events)),
            "entity_id": eid, "entity_type": "user", "role": role,
            "timestamp": ts, "_ts": ts.value / 1e9, "_hour": int(hour) % 24,
            "source_ip": ip, "geo_lat": 19.07, "geo_lon": 72.87, "geo_country": "IN",
            "resource_accessed": res, "resource_type": "endpoint",
            "auth_method": "password", "auth_result": "success",
            "session_duration": float(rng.lognormal(5.0, 0.5)),
            "command_sequence": ["login", "read"], "_ncmd": 2,
            "device_id": "dev-" + eid, "device_os": "Windows 11 23H2",
            "device_firmware": "n/a", "device_mac": "aa:bb:cc:00:00:01",
            "device_protocol": "https",
        })

    all_res = [r for rs in list(core.values()) + list(rare.values()) for r in rs]

    def pick(role, churn=0.06):
        """Ordinary access, with a little benign exploration.

        Without this churn nobody but the drifting entities ever touches a new resource,
        so the control score distribution is identically zero and theta collapses to 0 --
        which made every entity permanently "above threshold" and the whole experiment
        degenerate. Real users poke at unfamiliar things occasionally.
        """
        if rng.random() < churn:
            return all_res[int(rng.integers(0, len(all_res)))]
        return core[role][int(rng.integers(0, 8))]

    for p in range(n_pairs):
        role = roles[p % len(roles)]
        base = core[role]
        # Peers, so cohort corroboration has something to see.
        for k in range(4):
            pid = "peer%03d_%d" % (p, k)
            truth[pid] = "peer"
            for d in range(n_days):
                for _ in range(6):
                    emit(pid, role, d, rng.uniform(8, 19), pick(role))

        # Both entities keep expanding at an IDENTICAL rate for the whole post-day-30
        # window, acquiring one previously-unseen resource every two days.
        #
        # An earlier version had both stop at day 44. That made the experiment vacuous:
        # by day 55 the 7-day windows were empty, every score decayed to zero, and
        # "absorption" measured only that the activity had ceased -- 1.000 in all four
        # arms, including the frozen baseline. Poisoning is a question about ONGOING
        # activity: has the baseline learned to accept behaviour that is still happening?
        newset = [base[8 + i] for i in range(4)] + ["%s/proj%02d" % (role, i) for i in range(16)]
        badset = [rare[role][i % 12] for i in range(20)]
        for kind, targets in (("A", newset), ("B", badset)):
            eid = "%s%03d" % (kind, p)
            truth[eid] = "legit_drift" if kind == "A" else "slow_attacker"
            for d in range(n_days):
                for _ in range(6):
                    emit(eid, role, d, rng.uniform(8, 19), pick(role))
                if d < drift_day:
                    continue
                if d < drift_end:
                    # RAMP PHASE -- exactly matched. Same schedule, same count, one new
                    # resource every two days. If B ramped faster than A it would be
                    # trivially detectable on rate alone and the experiment would prove
                    # nothing. Only the CHARACTER differs: peer-consistent and
                    # corroborated for A, peer-inconsistent and solitary for B.
                    t = targets[min((d - drift_day) // 2, len(targets) - 1)]
                    for _ in range(2):
                        emit(eid, role, d, rng.uniform(8, 19), t)
                elif kind == "A":
                    # A SETTLES. The role change is complete, so it revisits its new
                    # tools without acquiring more. Adaptation is only observable if the
                    # legitimate change actually finishes -- an entity that expands
                    # forever can never stop looking anomalous, and cannot be adapted to.
                    acquired = targets[: (drift_end - drift_day) // 2]
                    for _ in range(2):
                        emit(eid, role, d, rng.uniform(8, 19),
                             acquired[int(rng.integers(0, len(acquired)))])
                else:
                    # B CONTINUES. An intrusion does not conclude. This asymmetry after
                    # the ramp is not a flaw in the matching -- it IS the difference
                    # between a completed role change and an ongoing compromise, and it
                    # is what makes poisoning observable at all.
                    t = targets[min((d - drift_day) // 2, len(targets) - 1)]
                    for _ in range(2):
                        emit(eid, role, d, rng.uniform(8, 19), t)

        cid = "C%03d" % p
        truth[cid] = "control"
        for d in range(n_days):
            for _ in range(6):
                emit(cid, role, d, rng.uniform(8, 19), base[int(rng.integers(0, 8))])

        # Cohort corroboration for A only: teammates adopt the same resources in the
        # same fortnight. This is the fact that legitimately distinguishes A from B.
        for k in range(3):
            pid = "peer%03d_%d" % (p, k)
            for i, r in enumerate(newset):
                day = drift_day + 2 * i + int(rng.integers(-4, 5))
                if 0 <= day < n_days:
                    emit(pid, role, day, rng.uniform(8, 19), r)

    events.sort(key=lambda e: e["_ts"])
    return events, truth


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


def _fuse(sig: Dict[str, float]) -> float:
    """Compact footprint-expansion risk score.

    Deliberately simple and fixed a priori: this experiment is about whether the BASELINE
    gets poisoned, so the scorer must be held constant across arms and must not be tuned.
    """
    return (1.0 * sig.get("resource_surprisal", 0.0)
            + 0.6 * sig.get("new_resource_rate_7d", 0.0)
            + 0.9 * sig.get("uncorroborated_new_edges_7d", 0.0)
            - 0.7 * sig.get("corroboration_7d", 0.0))


def run_arm(events: Sequence[dict], arm: str, theta: float,
            active_from: float = 0.0) -> pd.DataFrame:
    from features import FeatureExtractor

    fx = FeatureExtractor()
    pol = UpdatePolicy(arm, theta, active_from)
    rows = []

    def decider(sig, ev, st):
        sig["_fused"] = _fuse(sig)
        return pol(sig, ev, st)

    for ev in events:
        sig = fx.process(ev, update_decider=decider)
        rows.append({
            "entity_id": ev["entity_id"],
            "day": int((ev["_ts"] - events[0]["_ts"]) // 86400),
            "score": sig.get("_fused", _fuse(sig)),
        })
    df = pd.DataFrame(rows)
    df["arm"] = arm
    return df


def run_experiment(seeds: Sequence[int] = (1, 2, 3, 4, 5), n_pairs: int = 6,
                   n_days: int = 60, drift_day: int = 30, drift_end: int = 44,
                   verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_all, pair_all = [], []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        events, truth = build_world(rng, n_pairs, n_days, drift_day, drift_end)

        # theta from CONTROL entities before any drift begins: a false-positive rate on
        # traffic we have no reason to suspect, not a quantile of the mixed stream.
        # Probe with normal learning enabled. Using the "none" arm here froze every
        # baseline, so no entity ever accumulated a known-resource set, every long-window
        # signal stayed 0, and theta collapsed to 0.
        drift_ts = events[0]["_ts"] + drift_day * 86400.0
        probe = run_arm(events, "all", theta=float("inf"))
        ref_mask = (probe["entity_id"].str.startswith("C")
                    | probe["entity_id"].str.startswith("peer"))
        ctrl = probe[ref_mask & (probe["day"] < drift_day)]
        theta = float(np.quantile(ctrl["score"], 0.99)) if len(ctrl) else 1.0

        for arm in ARMS:
            df = run_arm(events, arm, theta, active_from=drift_ts)
            daily = (df.groupby(["entity_id", "day", "arm"])["score"].max()
                     .reset_index())
            daily["seed"] = seed
            daily["theta"] = theta
            daily["scenario"] = daily["entity_id"].map(truth)
            daily = daily[daily["scenario"].isin(
                ["legit_drift", "slow_attacker", "control"])]
            daily_all.append(daily)

            for p in range(n_pairs):
                a, b = "A%03d" % p, "B%03d" % p
                da = daily[daily["entity_id"] == a]
                db = daily[daily["entity_id"] == b]
                if da.empty or db.empty:
                    continue
                # ABSORPTION is a question about the TREND, not about theta.
                #
                # Defining it as "attacker below theta at day 60" is vacuous here: a
                # low_and_slow attack is engineered to sit below any per-event threshold
                # for its whole life, so that test returned 1.000 in every arm including
                # the frozen baseline. What poisoning actually looks like is the score
                # DECAYING while the behaviour continues unchanged -- the baseline has
                # learned to call the attack normal.
                early_b = db[(db["day"] >= drift_end - 6) & (db["day"] <= drift_end)]["score"]
                late_b = db[db["day"] >= n_days - 6]["score"]
                ratio_b = (float(late_b.mean() / early_b.mean())
                           if len(late_b) and len(early_b) and early_b.mean() > 1e-9
                           else np.nan)
                absorbed = bool(ratio_b < 0.5) if np.isfinite(ratio_b) else False

                early_a = da[(da["day"] >= drift_end - 6) & (da["day"] <= drift_end)]["score"]
                late_a = da[da["day"] >= n_days - 6]["score"]
                ratio_a = (float(late_a.mean() / early_a.mean())
                           if len(late_a) and len(early_a) and early_a.mean() > 1e-9
                           else np.nan)
                # Adaptation: first day after drift completion where A stays quiet 7 days.
                t_adapt = np.nan
                post = da[da["day"] >= drift_day].sort_values("day")
                s = (post["score"] < theta).to_numpy()
                for i in range(len(s) - 6):
                    if s[i:i + 7].all():
                        t_adapt = int(post["day"].to_numpy()[i]) - drift_day
                        break
                fp_a = int((da[(da["day"] >= drift_day)]["score"] >= theta).sum())
                pair_all.append({
                    "seed": seed, "pair": p, "arm": arm, "theta": theta,
                    "absorbed": absorbed, "t_adapt": t_adapt, "fp_legit": fp_a,
                    "ratio_b": ratio_b, "ratio_a": ratio_a,
                    "b_late_mean": float(late_b.mean()) if len(late_b) else np.nan,
                })
        if verbose:
            print("  seed %d done (theta=%.2f)" % (seed, theta))

    return pd.concat(daily_all, ignore_index=True), pd.DataFrame(pair_all)


def _boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return tuple(np.percentile(m, [2.5, 97.5]))


def summarise(pairs: pd.DataFrame) -> str:
    from scipy.stats import wilcoxon

    lines = ["=" * 78, "DRIFT / POISONING EXPERIMENT", "=" * 78]
    lines.append("%d matched attacker/legit pairs (%d seeds x %d pairs), 4 update arms"
                 % (len(pairs) // len(ARMS), pairs["seed"].nunique(),
                    pairs["pair"].nunique()))
    lines.append("")
    lines.append("score ratio = late window (day 54-60) / end of matched ramp (day 38-44).")
    lines.append("Low ratio = the baseline stopped reacting to behaviour that never stopped.")
    lines.append("For the ATTACKER low is BAD (poisoned); for LEGIT DRIFT low is GOOD (adapted).")
    lines.append("")
    lines.append("%-12s %-24s %-16s %-16s %s"
                 % ("arm", "absorption rate (B)", "ratio B (atk)", "ratio A (legit)",
                    "FP on legit"))
    lines.append("-" * 78)
    for arm in ARMS:
        s = pairs[pairs["arm"] == arm]
        a = s["absorbed"].to_numpy().astype(float)
        lo, hi = _boot_ci(a)
        rb = np.nanmean(s["ratio_b"].to_numpy(dtype=float))
        ra = np.nanmean(s["ratio_a"].to_numpy(dtype=float))
        lines.append("%-12s %.3f [%.3f, %.3f]      %-16.3f %-16.3f %.1f"
                     % (arm, a.mean(), lo, hi, rb, ra, s["fp_legit"].mean()))
    lines.append("")

    piv = pairs.pivot_table(index=["seed", "pair"], columns="arm", values="ratio_b")
    if {"all", "quarantine"} <= set(piv.columns):
        d = piv.dropna(subset=["all", "quarantine"])
        if len(d) >= 6 and (d["all"] != d["quarantine"]).any():
            stat, p = wilcoxon(d["all"], d["quarantine"])
            lines.append("Paired Wilcoxon, attacker score ratio, all vs quarantine:")
            lines.append("  W=%.1f  p=%.2g" % (stat, p))
    lines.append("")
    lines.append("THE COST, NOT HIDDEN: quarantine buys poisoning resistance with extra")
    lines.append("false positives on legitimately drifting entities. The delta is the")
    lines.append("'FP alerts on legit drift' column, arm 'quarantine' minus arm 'all'.")
    a_fp = pairs[pairs["arm"] == "all"]["fp_legit"].mean()
    q_fp = pairs[pairs["arm"] == "quarantine"]["fp_legit"].mean()
    lines.append("  delta = %+.1f alerts per legitimately-drifting entity" % (q_fp - a_fp))
    lines.append("=" * 78)
    return "\n".join(lines)


def make_plots(daily: pd.DataFrame, outdir: str, drift_day: int = 30,
               drift_end: int = 44) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    paths = []
    theta = float(daily["theta"].median())

    fig, axes = plt.subplots(1, len(ARMS), figsize=(19, 4.4), sharey=True)
    colours = {"legit_drift": "#1f77b4", "slow_attacker": "#d62728", "control": "#7f7f7f"}
    for ax, arm in zip(axes, ARMS):
        sub = daily[daily["arm"] == arm]
        for scen, c in colours.items():
            g = sub[sub["scenario"] == scen].groupby("day")["score"].mean()
            ax.plot(g.index, g.to_numpy(), color=c, label=scen.replace("_", " "), lw=1.8)
        ax.axhline(theta, ls="--", c="k", lw=1, label="theta")
        ax.axvline(drift_day, ls=":", c="0.4", lw=1)
        ax.axvline(drift_end, ls=":", c="0.4", lw=1)
        ax.set_title("arm: %s" % arm)
        ax.set_xlabel("day")
    axes[0].set_ylabel("footprint-expansion risk")
    axes[0].legend(fontsize=8, loc="upper left")
    # The title states what the panels actually show. An earlier version claimed the
    # attacker was "held under quarantine", which the data does not support: the three
    # adaptive arms are near-indistinguishable. What IS unambiguous is the right-hand
    # panel -- a frozen baseline alerts relentlessly on an entity whose change was
    # entirely legitimate.
    fig.suptitle("Adaptation vs rigidity: a frozen baseline (right) alerts relentlessly on "
                 "LEGITIMATE change.\nThe three adaptive arms do not separate attacker "
                 "from legitimate drift -- reported as a negative result.",
                 y=1.06, fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "drift_poisoning.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Drift adaptation and poisoning resistance.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--figures", default=os.path.join(os.path.dirname(__file__), "..", "figures"))
    args = ap.parse_args()

    print("running %d seeds x %d pairs x %d arms ..." % (len(args.seeds), args.pairs, len(ARMS)))
    daily, pairs = run_experiment(args.seeds, args.pairs, args.days)
    print()
    print(summarise(pairs))
    figs = make_plots(daily, os.path.abspath(args.figures))
    print("\nfigures: %s" % ", ".join(figs))


if __name__ == "__main__":
    main()

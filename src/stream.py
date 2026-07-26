"""Real-time scoring path and its latency benchmark.

`score_event(event) -> Alert` with O(1) state per entity. This is not a separate
implementation: it calls the same FeatureExtractor the batch evaluation uses, so the
latency measured here is the latency of the thing that was actually evaluated, and there
is no possibility of batch/stream skew.

Per-entity state is bounded by construction -- capped dictionaries and time-pruned deques
in features.EntityState -- so memory is O(entities), not O(events). That is what makes
"O(1) per event" a property of the code rather than an aspiration.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import LEVELS
from detector import Detector, explain, risk_band
from features import FeatureExtractor, SIGNAL_NAMES, prepare


@dataclass
class Alert:
    event_id: str
    entity_id: str
    timestamp: pd.Timestamp
    score: float
    band: str
    level: str
    cold_start: bool
    why: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["timestamp"] = str(self.timestamp)
        return d


class StreamScorer:
    """Online scorer. One instance holds all per-entity state."""

    def __init__(self, det: Detector) -> None:
        self.det = det
        self.fx = FeatureExtractor()
        self.n_scored = 0

    def score_event(self, ev: dict) -> Alert:
        sig = self.fx.process(ev)

        best_level, best_raw, best_z = LEVELS[0], -1e9, None
        for lv in LEVELS:
            fu = self.det.fusions[lv]
            raw, z = fu.raw_score_one(sig)
            if raw > best_raw:
                best_level, best_raw, best_z = lv, raw, z
        best_cal = float(self.det.fusions[best_level].calibrate(
            np.asarray([best_raw]))[0])

        fu = self.det.fusions[best_level]
        theta = fu.theta_ or 0.0
        raw_vals = {n: float(sig.get(n, 0.0)) for n in fu.signals}
        self.n_scored += 1
        return Alert(
            event_id=ev["event_id"], entity_id=ev["entity_id"],
            timestamp=ev["timestamp"], score=best_cal,
            band=risk_band(best_raw, theta), level=best_level,
            cold_start=bool(sig.get("_cold_start", 0.0)),
            why=explain(best_z, fu.signals, raw_vals, ev.get("role", "")),
        )

    def state_size_bytes(self) -> int:
        """Serialised size of all per-entity state -- the thing a real deployment would
        keep in RocksDB or Redis."""
        return len(pickle.dumps(self.fx.entities, protocol=4))


def benchmark(det: Detector, events: pd.DataFrame, n: int = 20000,
              warmup: int = 2000) -> Dict[str, float]:
    recs = prepare(events)
    sc = StreamScorer(det)

    for r in recs[:warmup]:
        sc.score_event(r)

    sample = recs[warmup:warmup + n]
    lat = np.empty(len(sample), dtype=float)
    t0 = time.perf_counter()
    for i, r in enumerate(sample):
        s = time.perf_counter()
        sc.score_event(r)
        lat[i] = (time.perf_counter() - s) * 1000.0
    wall = time.perf_counter() - t0

    n_entities = len(sc.fx.entities)
    state = sc.state_size_bytes()
    return {
        "n_events": float(len(sample)),
        "wall_seconds": wall,
        "events_per_second": len(sample) / wall,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "max_ms": float(lat.max()),
        "n_entities": float(n_entities),
        "state_bytes": float(state),
        "state_bytes_per_entity": float(state / max(1, n_entities)),
    }


ARCHITECTURE = """
PRODUCTION MAPPING
------------------

   auth / API / device logs
             |
             v
   [ ingest ]  Kafka topic, partitioned BY entity_id
             |                so all of one entity's events land on one worker
             v                and per-entity state is never shared across workers
   [ stream processor ]  Flink / Kafka Streams, one task per partition
             |            runs FeatureExtractor.process() unchanged
             v
   [ state store ]  RocksDB (embedded) or Redis
             |       per-entity: hour histogram, known IPs/resources/devices,
             |       duration stats, windowed deques. Bounded, serialisable.
             v
   [ scorer ]  Fusion per level -> calibrated score -> theta -> band
             |
             v
   [ alert queue ]  ranked, deduplicated into alerts by (level, scope_key)
                    within the merge window, then presented to the analyst

Partitioning by entity_id is what makes this scale horizontally: entity state is
strictly local to a partition, so workers never coordinate. The IP-level detector is
the exception -- credential stuffing is by definition cross-entity -- so it needs a
second stream keyed by source_ip. That is the standard co-partitioning trade-off and
it is why the levels are kept separate in the design rather than fused into one score.

Reference distributions and the calibration map are broadcast state, recomputed
offline (nightly) and pushed to workers; they change slowly and must not be updated
per event.
"""


def main() -> None:
    from pipeline import DATA, load
    from features import extract
    from config import ATTACK_TYPES, NORMAL, assert_feature_frame

    ap = argparse.ArgumentParser(description="Streaming latency benchmark.")
    ap.add_argument("--fit", default="seed1_delta05")
    ap.add_argument("--eval", dest="eval_tag", default="seed3_delta05")
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()

    fit_events, fit_labels, _ = load(args.fit, DATA)
    assert_feature_frame(fit_events, "stream(fit)")
    fit_sig, _ = extract(fit_events)
    y = (fit_sig[["event_id"]].merge(fit_labels[["event_id", "label"]],
                                     on="event_id", how="left")["label"]
         .fillna(NORMAL).isin(ATTACK_TYPES).to_numpy().astype(int))
    det = Detector().fit(fit_sig, y)

    ev_events, _, _ = load(args.eval_tag, DATA)
    res = benchmark(det, ev_events, n=args.n)

    print("=" * 66)
    print("STREAMING LATENCY  (single process, Python 3.9, 12-core CPU)")
    print("=" * 66)
    print("  events scored          %d" % res["n_events"])
    print("  throughput             %.0f events/sec" % res["events_per_second"])
    print("  p50 latency            %.3f ms" % res["p50_ms"])
    print("  p95 latency            %.3f ms" % res["p95_ms"])
    print("  p99 latency            %.3f ms" % res["p99_ms"])
    print("  max latency            %.3f ms" % res["max_ms"])
    print("  entities tracked       %d" % res["n_entities"])
    print("  total state            %.1f MB" % (res["state_bytes"] / 1e6))
    print("  state per entity       %.1f kB" % (res["state_bytes_per_entity"] / 1e3))
    print(ARCHITECTURE)

    out = os.path.join(os.path.dirname(__file__), "..", "figures", "latency.json")
    with open(os.path.abspath(out), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()

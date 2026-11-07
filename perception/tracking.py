"""Multi-frame hotspot tracker: georeferenced detections -> fused Hotspot entities.

At 9 Hz a seeker sees the same spot in dozens of consecutive frames; without tracking each frame would be a
"new" hotspot. Association is greedy nearest-neighbour inside a gate (default 5 m, see georef.py error budget).

Fusion ([ASSUMPTION] defaults):
  * position     : confidence-weighted running mean
  * confidence   : log-odds sum with evidence_weight < 1, because consecutive frames are strongly correlated
                   (10 frames of the same blob are not 10 independent pieces of evidence); clamped to [0.01, 0.99]
  * peak_temp_c  : max seen;  est_area_m2: exponential moving average
  * growth rate  : least-squares slope of area vs time once >= 3 samples spanning >= 30 s
Status stays DETECTED; CONFIRMED comes from an INVESTIGATE task (workstreams A/B), not from the tracker.
IDs are deterministic: f"{id_prefix}-H{n}".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from contracts import Hotspot, HotspotStatus, Position

__all__ = ["Observation", "HotspotTracker"]


@dataclass(frozen=True)
class Observation:
    position: Position
    confidence: float
    peak_temp_c: float
    area_m2: float
    timestamp: float


@dataclass
class _Track:
    hotspot_id: str
    x: float
    y: float
    z: float
    w: float
    log_odds: float
    peak: float
    area: float
    first_seen: float
    last_seen: float
    n: int = 1
    history: list[tuple[float, float]] = field(default_factory=list)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class HotspotTracker:
    def __init__(self, id_prefix: str, gate_m: float = 5.0, evidence_weight: float = 0.3,
                 area_alpha: float = 0.3, max_age_s: float | None = None):
        self.id_prefix, self.gate_m, self.evidence_weight = id_prefix, gate_m, evidence_weight
        self.area_alpha, self.max_age_s = area_alpha, max_age_s
        self.tracks: list[_Track] = []
        self._next = 1

    def _to_hotspot(self, t: _Track) -> Hotspot:
        conf = float(np.clip(1 / (1 + math.exp(-t.log_odds)), 0.01, 0.99))
        return Hotspot(
            hotspot_id=t.hotspot_id, position=Position(x=t.x, y=t.y, z=t.z),
            peak_temp_c=float(np.clip(t.peak, -50, 1500)), est_area_m2=max(t.area, 0.0), confidence=conf,
            status=HotspotStatus.DETECTED, observations=t.n, growth_rate_m2_per_min=self._growth(t),
            reported_by=self.id_prefix, first_seen=t.first_seen, last_seen=t.last_seen,
        )

    @staticmethod
    def _growth(t: _Track) -> float | None:
        if len(t.history) < 3 or t.history[-1][0] - t.history[0][0] < 30.0:
            return None
        ts, areas = np.array(t.history).T
        slope = np.polyfit(ts, areas, 1)[0]  # m^2 per second
        return float(slope * 60.0)

    def update(self, observations: list[Observation]) -> list[Hotspot]:
        """Fuse one frame's observations. Returns the hotspots that changed (new or updated)."""
        changed: dict[str, _Track] = {}
        used: set[int] = set()
        for ob in sorted(observations, key=lambda o: -o.confidence):
            best, best_d = None, self.gate_m
            for i, t in enumerate(self.tracks):
                if i in used:
                    continue  # one observation per track per frame
                d = math.hypot(t.x - ob.position.x, t.y - ob.position.y)
                if d <= best_d:
                    best, best_d = i, d
            if best is None:
                t = _Track(hotspot_id=f"{self.id_prefix}-H{self._next}", x=ob.position.x, y=ob.position.y,
                           z=ob.position.z, w=ob.confidence, log_odds=_logit(ob.confidence), peak=ob.peak_temp_c,
                           area=ob.area_m2, first_seen=ob.timestamp, last_seen=ob.timestamp,
                           history=[(ob.timestamp, ob.area_m2)])
                self._next += 1
                self.tracks.append(t)
                used.add(len(self.tracks) - 1)
            else:
                t = self.tracks[best]
                used.add(best)
                w = max(ob.confidence, 1e-3)
                t.x, t.y, t.z = ((t.x * t.w + ob.position.x * w) / (t.w + w), (t.y * t.w + ob.position.y * w) / (t.w + w),
                                 (t.z * t.w + ob.position.z * w) / (t.w + w))
                t.w += w
                t.log_odds = float(np.clip(t.log_odds + self.evidence_weight * _logit(ob.confidence), -9, 9))
                t.peak = max(t.peak, ob.peak_temp_c)
                t.area = (1 - self.area_alpha) * t.area + self.area_alpha * ob.area_m2
                t.last_seen = max(t.last_seen, ob.timestamp)
                t.n += 1
                t.history.append((ob.timestamp, ob.area_m2))
                t.history = t.history[-200:]
            changed[t.hotspot_id] = t
        return [self._to_hotspot(t) for t in changed.values()]

    def prune(self, now: float) -> None:
        if self.max_age_s is not None:
            self.tracks = [t for t in self.tracks if now - t.last_seen <= self.max_age_s]

    def hotspots(self) -> list[Hotspot]:
        return [self._to_hotspot(t) for t in self.tracks]

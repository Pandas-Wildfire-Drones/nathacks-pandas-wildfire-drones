"""Task allocation: a max-heap of fires by priority, each served by the extinguisher with the lowest refill-aware ETA.

Cost of drone d serving fire f (seconds):
    has water:   |d -> f| / v
    needs water: min over refill points s of ( |d -> s| + |s -> f| ) / v + fill_s
The refill point is chosen PER TASK (best detour), not "nearest to the drone": a lake slightly behind the drone
but right next to the fire beats one next to the drone on the far side. With only the base as a refill point this
degrades to "go home, then go to the fire", every drone costs the same once it is home, and the allocator is plain
highest-priority-first (the lakeless baseline the brief asked for).

Round (every leader tick, all idle drones):
    pop fires in priority order; for each, give it as many drones as it still needs (litres / payload, capped
    at max_per_fire), each time picking the idle drone with the lowest ETA; stop when no drone is idle.
Fires keep their place: every fire that still needs water is pushed back for the next round. Entries are lazily
invalidated (a version counter) so updating a priority is O(log n).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from contracts import Position, RefillTargetKind

__all__ = ["Assignment", "DroneView", "FireQueue", "FireView", "RefillPoint", "allocate", "best_refill", "eta_s"]


@dataclass(frozen=True)
class RefillPoint:
    point_id: str
    kind: RefillTargetKind
    position: Position
    fill_s: float = 0.0     # hook-up + pickup time at this point


@dataclass
class DroneView:
    drone_id: str
    position: Position
    water_l: float
    capacity_l: float
    speed_mps: float


@dataclass
class FireView:
    fire_id: str
    position: Position
    priority: float           # higher = more urgent
    need_l: float             # litres still unassigned
    max_drones: int = 1


@dataclass(frozen=True)
class Assignment:
    drone_id: str
    fire_id: str
    volume_l: float
    refill: RefillPoint | None   # where to fill first (None = fly direct)
    eta_s: float


def _d(a: Position, b: Position) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def best_refill(src: Position, dst: Position | None, points: list[RefillPoint], speed_mps: float = 1.0
                ) -> tuple[RefillPoint | None, float]:
    """Refill point minimising src -> s -> dst (or src -> s when dst is unknown). Returns (point, seconds)."""
    best, best_t = None, math.inf
    for s in points:
        t = (_d(src, s.position) + (_d(s.position, dst) if dst is not None else 0.0)) / speed_mps + s.fill_s
        if t < best_t:
            best, best_t = s, t
    return best, best_t


def needs_refill(d: DroneView, volume_l: float, min_useful_l: float) -> bool:
    return d.water_l < min(volume_l, max(min_useful_l, 1e-6))


def eta_s(d: DroneView, target: Position, volume_l: float, points: list[RefillPoint], min_useful_l: float
          ) -> tuple[float, RefillPoint | None]:
    if not needs_refill(d, volume_l, min_useful_l) or not points:
        return _d(d.position, target) / d.speed_mps, None
    s, t = best_refill(d.position, target, points, d.speed_mps)
    return t, s


class FireQueue:
    """Max-heap of fire ids keyed by priority, with lazy updates."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, str, int]] = []
        self._version: dict[str, int] = {}   # monotonic per fire: stale heap entries can never match again
        self._live: dict[str, int] = {}      # fire -> version of its one valid heap entry
        self._seq = 0

    def push(self, fire_id: str, priority: float) -> None:
        """Insert, or re-key if already queued (the old entry becomes stale)."""
        v = self._version.get(fire_id, 0) + 1
        self._version[fire_id] = v
        self._live[fire_id] = v
        self._seq += 1
        heapq.heappush(self._heap, (-priority, self._seq, fire_id, v))
        if len(self._heap) > 64 and len(self._heap) > 4 * len(self._live):   # drop stale entries
            self._heap = [e for e in self._heap if self._live.get(e[2]) == e[3]]
            heapq.heapify(self._heap)

    def discard(self, fire_id: str) -> None:
        self._live.pop(fire_id, None)

    def pop(self) -> tuple[str, float] | None:
        while self._heap:
            neg, _, fid, v = heapq.heappop(self._heap)
            if self._live.get(fid) == v:
                del self._live[fid]
                return fid, -neg
        return None

    def __contains__(self, fire_id: str) -> bool:
        return fire_id in self._live

    def __len__(self) -> int:
        return len(self._live)


def allocate(queue: FireQueue, fires: dict[str, FireView], idle: list[DroneView], points: list[RefillPoint],
             min_useful_l: float = 1.0) -> list[Assignment]:
    """One allocation round. Pops fires from `queue` in priority order and pushes back those still needing water."""
    out: list[Assignment] = []
    free = {d.drone_id: d for d in idle}
    keep: list[tuple[str, float]] = []
    while free:
        item = queue.pop()
        if item is None:
            break
        fid, prio = item
        f = fires.get(fid)
        if f is None or f.need_l <= 0:
            continue
        need = f.need_l
        taken = 0
        while need > 1e-6 and taken < f.max_drones and free:
            best = None
            for d in free.values():
                vol = min(need, d.capacity_l)
                t, s = eta_s(d, f.position, vol, points, min_useful_l)
                key = (t, d.drone_id)
                if best is None or key < best[0]:
                    best = (key, d, s, vol)
            (t, _), d, s, vol = best
            if s is None:
                vol = min(vol, d.water_l) if d.water_l > 0 else vol
            out.append(Assignment(d.drone_id, fid, max(vol, 1.0), s, t))
            del free[d.drone_id]
            need -= vol
            taken += 1
        keep.append((fid, prio))
    for fid, prio in keep:
        queue.push(fid, prio)
    return out

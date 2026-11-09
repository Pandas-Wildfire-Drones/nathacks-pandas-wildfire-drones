"""Legacy swarm-brain stubs. The autonomous swarm (watch / DEMO / play's shadow run) now uses the real brain in
`core/` (core.leader.SwarmLeader, core.allocation, core.zones, core.priority). What is left here only backs the
human-mode bot commander (sim/bots.HumanCommander subclasses StubLeader) and its nearest-station refill choice.

Three seams, each a Protocol:
  ZoneAllocator   arena + seekers -> one search Zone per seeker            stub: equal horizontal bands
  TaskAssigner    hotspots + idle extinguishers -> (drone, hotspot, litres) stub: greedy nearest, no look-ahead
  RefillAssigner  RefillRequest -> station                                 stub: nearest station

`StubLeader` is the bus-facing leader role around them: it listens to hotspot reports, heartbeats, task completions
and refill requests, and talks to drones with directed HotspotReport + TaskAssign (epoch-stamped) and RefillGrant.
Hotspot confirmation is also a stub: a report at/above the CNN's precision-first confirm threshold counts as
CONFIRMED (the real flow is a seeker INVESTIGATE pass with perception.confirm.HotspotConfirmer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

from contracts import (
    SUB_ALL_HEARTBEATS, SUB_ALL_HOTSPOTS, DroneKind, Heartbeat, Hotspot, HotspotReport, HotspotStatus,
    MessageType, Position, RefillGrant, RefillRequest, RefillStation, RefillTargetKind, SwarmMessage, Task,
    TaskAssign, TaskComplete, TaskStatus, TaskType, Zone, sub_my_commands, topic_for,
)
from contracts.topics import ROUTES

__all__ = ["ExtView", "ZoneAllocator", "TaskAssigner", "RefillAssigner", "DummyZoneAllocator",
           "DummyTaskAssigner", "DummyRefillAssigner", "TrackedHotspot", "StubLeader"]


@dataclass
class ExtView:
    """What the leader knows about one extinguisher (from heartbeats)."""

    drone_id: str
    position: Position
    water_l: float
    capacity_l: float
    busy: bool


# ------------------------------------------------------------------------------------------------------------------
class ZoneAllocator(Protocol):
    def allocate(self, arena_radius_m: float, seeker_ids: list[str]) -> dict[str, Zone]: ...


class TaskAssigner(Protocol):
    def assign(self, hotspots: list["TrackedHotspot"], drones: list[ExtView], now: float
               ) -> list[tuple[str, str, float]]: ...


class RefillAssigner(Protocol):
    def choose(self, request: RefillRequest, stations: list[RefillStation]) -> RefillStation | None: ...


def task_volume(need_l: float, d: ExtView, min_l: float = 20.0) -> float:
    """Litres for one task: what the drone carries if that's a useful amount (no needless refill), else a full tank
    (autonomy then requests a refill itself before flying out)."""
    if d.water_l >= min_l:
        return max(1.0, min(need_l, d.water_l))
    return max(1.0, min(need_l, d.capacity_l))


# ------------------------------------------------------------------------------------------------------------------
class DummyZoneAllocator:
    """STUB: cut the arena into equal-height horizontal bands, one per seeker (polygon = band ∩ circle)."""

    def __init__(self, margin_m: float = 6.0, arc_points: int = 10):
        self.margin_m, self.arc_points = margin_m, arc_points

    def allocate(self, arena_radius_m: float, seeker_ids: list[str]) -> dict[str, Zone]:
        R = arena_radius_m - self.margin_m
        n = max(len(seeker_ids), 1)
        out = {}
        for i, sid in enumerate(seeker_ids):
            y0, y1 = -R + 2 * R * i / n, -R + 2 * R * (i + 1) / n
            pts = []
            for k in range(self.arc_points + 1):          # east arc, south -> north
                y = y0 + (y1 - y0) * k / self.arc_points
                pts.append(Position(x=math.sqrt(max(R * R - y * y, 0.0)), y=y))
            for k in range(self.arc_points + 1):          # west arc, north -> south
                y = y1 - (y1 - y0) * k / self.arc_points
                pts.append(Position(x=-math.sqrt(max(R * R - y * y, 0.0)), y=y))
            out[sid] = Zone(zone_id=f"Z{i + 1}", boundary=pts, assigned_to=sid)
        return out


class DummyTaskAssigner:
    """STUB: each idle drone takes the nearest unfinished hotspot. No priorities, no look-ahead, no bundling."""

    def assign(self, hotspots, drones, now):
        out = []
        remaining = {h.hotspot_id: h.remaining_l for h in hotspots}
        for d in sorted(drones, key=lambda d: d.drone_id):
            if d.busy:
                continue
            best, best_d = None, float("inf")
            for h in hotspots:
                if remaining[h.hotspot_id] <= 0:
                    continue
                dist = math.hypot(h.hotspot.position.x - d.position.x, h.hotspot.position.y - d.position.y)
                if dist < best_d:
                    best, best_d = h, dist
            if best is None:
                continue
            vol = task_volume(remaining[best.hotspot_id], d)
            remaining[best.hotspot_id] = 0.0       # one drone per hotspot per round
            out.append((d.drone_id, best.hotspot_id, vol))
        return out


class DummyRefillAssigner:
    """STUB: nearest station with a free bay (stations only: basins are off in the game)."""

    def choose(self, request, stations):
        best = None
        for s in stations:
            d = math.hypot(s.position.x - request.position.x, s.position.y - request.position.y)
            if best is None or d < best[0]:
                best = (d, s)
        return best[1] if best else None


# ------------------------------------------------------------------------------------------------------------------
@dataclass
class TrackedHotspot:
    hotspot: Hotspot
    need_l: float
    delivered_l: float = 0.0
    reserved_l: float = 0.0
    verify_until: float = -1.0          # waiting for the dropping drone's VERIFY report until this time
    sources: set[str] = field(default_factory=set)

    @property
    def hotspot_id(self) -> str:
        return self.hotspot.hotspot_id

    @property
    def remaining_l(self) -> float:
        return self.need_l - self.delivered_l - self.reserved_l


@dataclass
class _Assignment:
    task: Task
    hotspot_id: str
    volume_l: float
    sent_at: float
    started: bool = False


class StubLeader:
    """Leader role for one world. `assigner`/`refill` are the swappable swarm-brain pieces."""

    def __init__(self, leader_id: str, bus, clock, *, assigner: TaskAssigner, refill: RefillAssigner,
                 stations: list[RefillStation], confirm_threshold: float, l_per_m2: float, margin: float,
                 min_volume_l: float, merge_radius_m: float = 6.0, epoch: int = 1,
                 geofence_radius_m: float | None = None, log: Callable[[str], None] | None = None):
        self.geofence_radius_m = geofence_radius_m
        self.id, self.bus, self.clock = leader_id, bus, clock
        self.assigner, self.refill = assigner, refill
        self.stations = stations
        self.confirm_threshold = confirm_threshold
        self.l_per_m2, self.margin, self.min_volume_l = l_per_m2, margin, min_volume_l
        self.merge_radius_m = merge_radius_m
        self.epoch = epoch
        self.log = log or (lambda s: None)
        self.hotspots: dict[str, TrackedHotspot] = {}
        self._alias: dict[str, str] = {}
        self.ext: dict[str, ExtView] = {}
        self.assignments: dict[str, _Assignment] = {}   # drone_id -> active assignment
        self.active = False
        self.seq = 0
        self.n_tasks = 0
        self.last_assign_t = -1e9
        bus.subscribe(SUB_ALL_HOTSPOTS, self._on_hotspot)
        bus.subscribe(SUB_ALL_HEARTBEATS, self._on_heartbeat)
        bus.subscribe(ROUTES[MessageType.TASK_COMPLETE], self._on_complete)
        bus.subscribe(ROUTES[MessageType.REFILL_REQUEST], self._on_refill)

    # ---- inputs ------------------------------------------------------------------------------------------------
    def _need(self, h: Hotspot) -> float:
        return max(self.min_volume_l, h.est_area_m2 * self.l_per_m2 * self.margin)

    def _on_hotspot(self, msg: SwarmMessage) -> None:
        h = msg.body.hotspot
        if self.geofence_radius_m is not None and math.hypot(h.position.x, h.position.y) > self.geofence_radius_m:
            return   # the camera sees past the geofence; nothing out there can be tasked
        if h.hotspot_id in self.hotspots and msg.sender_id in self.ext:
            self._on_verify(h)
            return
        key = self._alias.get(h.hotspot_id)
        if key is None:
            for k, t in self.hotspots.items():
                p = t.hotspot.position
                if math.hypot(p.x - h.position.x, p.y - h.position.y) <= self.merge_radius_m:
                    key = k
                    break
        if key is None:
            key = h.hotspot_id
            status = HotspotStatus.CONFIRMED if h.confidence >= self.confirm_threshold else HotspotStatus.DETECTED
            self.hotspots[key] = TrackedHotspot(h.model_copy(update={"status": status}), self._need(h))
        else:
            t = self.hotspots[key]
            old = t.hotspot
            conf = max(old.confidence, h.confidence)
            status = old.status
            if status is HotspotStatus.DETECTED and conf >= self.confirm_threshold:
                status = HotspotStatus.CONFIRMED
            area = max(old.est_area_m2, h.est_area_m2)
            t.hotspot = old.model_copy(update={
                "confidence": conf, "status": status, "est_area_m2": area,
                "peak_temp_c": max(old.peak_temp_c, h.peak_temp_c), "last_seen": max(old.last_seen, h.last_seen),
                "observations": old.observations + 1})
            if t.delivered_l == 0:
                t.need_l = max(t.need_l, self._need(t.hotspot))
        self._alias[h.hotspot_id] = key
        self.hotspots[key].sources.add(msg.sender_id)

    def _on_verify(self, h: Hotspot) -> None:
        """VERIFY report from the extinguisher that just dropped (perception.confirm.SuppressionChecker)."""
        t = self.hotspots[h.hotspot_id]
        t.verify_until = -1.0
        if h.status is HotspotStatus.SUPPRESSED:
            t.hotspot = t.hotspot.model_copy(update={"status": HotspotStatus.SUPPRESSED, "assigned_to": None,
                                                     "last_seen": h.last_seen})
            return
        t.hotspot = t.hotspot.model_copy(update={"status": HotspotStatus.CONFIRMED, "assigned_to": None,
                                                 "position": h.position, "est_area_m2": h.est_area_m2,
                                                 "last_seen": h.last_seen})
        t.need_l = t.delivered_l + self.reserved_total(t) + self._need(h)

    @staticmethod
    def reserved_total(t: "TrackedHotspot") -> float:
        return t.reserved_l

    def _on_heartbeat(self, msg: SwarmMessage) -> None:
        s = msg.body.state
        if s.kind is not DroneKind.EXTINGUISHER:
            return
        busy = s.current_task_id is not None or s.drone_id in self.assignments
        a = self.assignments.get(s.drone_id)
        if a is not None:
            if s.current_task_id == a.task.task_id:
                a.started = True
            elif (a.started or self.clock.now() - a.sent_at > 3.0) and s.current_task_id is None:
                # adapter rejected or dropped it without a TaskComplete: free the drone and the litres
                self._release(s.drone_id, delivered=0.0)
                busy = False
        self.ext[s.drone_id] = ExtView(s.drone_id, s.position, s.water_remaining_l or 0.0, 170.0, busy)

    def _on_complete(self, msg: SwarmMessage) -> None:
        b: TaskComplete = msg.body
        self._release(b.drone_id, delivered=b.water_used_l or 0.0)
        if not b.success and b.notes:
            self.log(f"{b.drone_id}: {b.notes}")

    def _release(self, drone_id: str, delivered: float) -> None:
        a = self.assignments.pop(drone_id, None)
        if a is None:
            return
        t = self.hotspots.get(a.hotspot_id)
        if t is not None:
            t.reserved_l = max(0.0, t.reserved_l - a.volume_l)
            t.delivered_l += delivered
            if delivered > 0:
                t.verify_until = self.clock.now() + 8.0   # the dropping drone re-images it before anyone else goes
        if drone_id in self.ext:
            self.ext[drone_id].busy = False

    def _on_refill(self, msg: SwarmMessage) -> None:
        req: RefillRequest = msg.body
        st = self.refill.choose(req, self.stations)
        if st is None:
            return
        grant = RefillGrant(drone_id=req.drone_id, target_kind=RefillTargetKind.STATION, target_id=st.station_id,
                            position=st.position, epoch=self.epoch)
        self._send(grant, receiver=req.drone_id)

    # ---- outputs -----------------------------------------------------------------------------------------------
    def _send(self, body, receiver: str | None = None, topic: str | None = None) -> None:
        self.seq += 1
        msg = SwarmMessage.create(self.id, body, self.clock.now(), seq=self.seq, receiver_id=receiver,
                                  epoch=self.epoch)
        self.bus.publish(topic or topic_for(msg), msg)

    def assign(self, drone_id: str, hotspot_id: str, volume_l: float) -> None:
        t = self.hotspots[hotspot_id]
        now = self.clock.now()
        self.n_tasks += 1
        task = Task(task_id=f"{self.id}-T{self.n_tasks}", type=TaskType.EXTINGUISH, priority=0.8,
                    status=TaskStatus.ASSIGNED, assigned_to=drone_id, target_hotspot_id=hotspot_id,
                    created_at=now, epoch=self.epoch, volume_l=max(volume_l, 1.0))
        h = t.hotspot.model_copy(update={"status": HotspotStatus.ASSIGNED, "assigned_to": drone_id,
                                         "position": Position(x=t.hotspot.position.x, y=t.hotspot.position.y, z=0.0)})
        # the autonomy adapter learns its target only from a HotspotReport, so send it first (same topic, in order)
        self._send(HotspotReport(hotspot=h), receiver=drone_id, topic=sub_my_commands(drone_id))
        self._send(TaskAssign(task=task, epoch=self.epoch), receiver=drone_id)
        t.reserved_l += task.volume_l
        self.assignments[drone_id] = _Assignment(task, hotspot_id, task.volume_l, now)
        if drone_id in self.ext:
            self.ext[drone_id].busy = True

    def assignable(self) -> list[TrackedHotspot]:
        """Confirmed hotspots that still need water and have no drone on them (one drone per hotspot at a time:
        identical goals make autonomy's CBS replan endlessly)."""
        now = self.clock.now()
        return [t for t in self.hotspots.values()
                if t.hotspot.status in (HotspotStatus.CONFIRMED, HotspotStatus.ASSIGNED) and t.remaining_l > 0
                and t.reserved_l <= 0 and t.verify_until < now]

    def tick(self, now: float) -> None:
        if not self.active or now - self.last_assign_t < 0.5:
            return
        self.last_assign_t = now
        idle = [d for d in self.ext.values() if not d.busy and d.drone_id not in self.assignments]
        if not idle:
            return
        for drone_id, hid, vol in self.assigner.assign(self.assignable(), idle, now):
            if hid in self.hotspots and drone_id not in self.assignments:
                self.assign(drone_id, hid, vol)

    def preview_plan(self) -> list[tuple[str, str, float]]:
        """What the assigner would do right now (shown on the planning screen)."""
        idle = [ExtView(d.drone_id, d.position, d.water_l, d.capacity_l, False) for d in self.ext.values()]
        return self.assigner.assign(self.assignable(), idle, self.clock.now())

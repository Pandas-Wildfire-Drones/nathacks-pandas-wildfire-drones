"""SwarmLeader: the leader role (workstream A) on the bus. Pure logic: contracts + stdlib, Bus/Clock protocols only.

Inputs (subscriptions)                     Outputs (directed to drone/{id}/cmd unless noted)
  drone/*/hotspots   seeker + VERIFY reports   HotspotReport  target picture for the drone (sent before the task)
  drone/*/heartbeat  where everyone is         TaskAssign     EXTINGUISH (volume_l), REFILL (stand-by prefill),
  swarm/tasks/complete                                        INVESTIGATE (seeker close-range confirmation)
  swarm/refill/requests                        RefillGrant    best refill point for the drone's current task
  ground/commands    operator decisions

Flow per fire:
  1. seeker reports merge into one TrackedFire (merge radius) as DETECTED.
     CONFIRMATION (cfg.investigate): the nearest free seeker gets an INVESTIGATE task; it descends and runs
     perception.confirm.HotspotConfirmer over several frames (precision-first threshold, mean probability):
        CONFIRMED -> fire is real; FALSE_POSITIVE -> closed; inconclusive -> REVIEW: the seeker sends a ReviewReport
        to the ground base and the leader waits for GroundCommand(CONFIRM_FIRE | MARK_FALSE_POSITIVE).
     The swarm never depends on the ground link: no answer within review_timeout_s -> the leader confirms on its
     own (recall-first, decision 0004: a missed fire costs more than a wasted drop). A human MARK_FALSE_POSITIVE or
     CONFIRM_OUT always wins. With cfg.dispatch_on_detect, extinguishers may already fly to a DETECTED fire; their
     drop is gated on the confirmation (the target picture they hold says DETECTED until the leader re-sends it).
     (cfg.investigate = False keeps the old rule: confidence >= confirm threshold counts as CONFIRMED.)
  2. priority = core.priority.fire_priority (size x growth x temperature x confidence); the fire goes on a max-heap.
  3. every tick, idle extinguishers are handed out by core.allocation.allocate: highest priority first, each slot
     to the drone with the lowest refill-aware ETA (drone -> best refill point -> fire).
  4. the dropping drone re-images the spot (VERIFY): SUPPRESSED closes the fire; still hot -> new area/position,
     need and priority are recomputed and the fire goes back on the heap (same priority formula).
  5. a later seeker detection on a SUPPRESSED fire (rekindle) re-opens it.
Refill requests are answered with the point minimising drone -> point -> task target (lakes when enabled, else base).
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from contracts import (
    SUB_ALL_HEARTBEATS,
    SUB_ALL_HOTSPOTS,
    SUB_GROUND_COMMANDS,
    CommandKind,
    DroneKind,
    GroundCommand,
    Hotspot,
    HotspotReport,
    HotspotStatus,
    MessageType,
    Position,
    RefillGrant,
    RefillRequest,
    RefillTargetKind,
    SwarmMessage,
    Task,
    TaskAssign,
    TaskComplete,
    TaskStatus,
    TaskType,
    sub_my_commands,
    topic_for,
)
from contracts.topics import ROUTES
from core.allocation import (
    Assignment,
    DroneView,
    FireQueue,
    FireView,
    RefillPoint,
    allocate,
    best_refill,
)
from core.priority import PriorityParams, fire_priority, normalized_priority

__all__ = ["ExtState", "LeaderConfig", "SeekerState", "SwarmLeader", "TrackedFire"]

_GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


@dataclass
class LeaderConfig:
    confirm_threshold: float
    capacity_l: float = 170.0           # extinguisher payload (litres of loaded agent)
    speed_mps: float = 20.0             # extinguisher cruise, for ETAs
    l_per_m2: float = 10.0              # water needed per m2 of burning area (PLACEHOLDER, contracts)
    agent_mult: float = 1.0             # litres of water 1 L of the loaded agent replaces (foam 3x, ASSUMPTION)
    margin: float = 1.4                 # need = area x l/m2 / agent_mult x margin
    min_volume_l: float = 25.0
    min_useful_l: float = 20.0          # below this a drone refills before flying a task (clamped to 90 % of capacity)
    max_per_fire: int = 1               # concurrent drones per fire (1 for the CBS-planned heavy extinguishers)
    merge_radius_m: float = 6.0
    verify_timeout_s: float = 8.0
    geofence_radius_m: float | None = None
    prefill_idle: bool = False          # idle + empty drones top up at their nearest refill point while waiting
    stage_idle: bool = False            # idle + loaded drones pre-position, spread evenly over the refill points
    tick_s: float = 0.5
    fire_time_scale: float = 1.0        # [GAME-SCALE] sim seconds run this many fire seconds (growth units)
    priority: PriorityParams = field(default_factory=PriorityParams)
    # --- confirmation --------------------------------------------------------------------------------------------
    investigate: bool = True            # a seeker INVESTIGATE pass decides CONFIRMED (False: confidence rule)
    dispatch_on_detect: bool = False    # extinguishers may fly to DETECTED fires; the drop waits for confirmation
    investigate_timeout_s: float = 45.0  # investigator silent this long -> re-assign
    review_timeout_s: float = 20.0      # no ground decision by then -> confirm autonomously (recall-first)


@dataclass
class TrackedFire:
    hotspot: Hotspot
    need_l: float
    raw_priority: float = 0.0
    priority: float = 0.0
    delivered_l: float = 0.0
    inflight: dict[str, float] = field(default_factory=dict)
    samples: list[tuple[float, float]] = field(default_factory=list)   # (t, area) from seekers
    growth_m2_min: float | None = None
    verify_until: float = -1.0
    suppressed_at: float | None = None
    drops: int = 0
    reopened: int = 0
    sources: set[str] = field(default_factory=set)
    investigator: str | None = None     # seeker running the INVESTIGATE task
    investigate_task: str | None = None
    investigate_since: float = -1.0
    review_since: float | None = None   # inconclusive close-range look: waiting for the ground base
    close_range: Hotspot | None = None  # the investigating seeker's report (position / area / mean p)
    confirmed_by: str | None = None     # "investigate" | "ground" | "timeout" | "detector"

    @property
    def hotspot_id(self) -> str:
        return self.hotspot.hotspot_id

    @property
    def reserved_l(self) -> float:
        return sum(self.inflight.values())

    @property
    def remaining_l(self) -> float:
        return self.need_l - self.delivered_l - self.reserved_l


@dataclass
class ExtState:
    drone_id: str
    position: Position
    water_l: float
    hb_task: str | None = None


@dataclass
class SeekerState:
    drone_id: str
    position: Position
    hb_task: str | None = None


@dataclass
class _Assignment:
    task: Task
    fire_id: str
    volume_l: float
    sent_at: float
    eta_s: float
    refill: RefillPoint | None
    started: bool = False


class SwarmLeader:
    def __init__(self, leader_id: str, bus, clock, cfg: LeaderConfig, refill_points: list[RefillPoint],
                 epoch: int = 1, log: Callable[[str], None] | None = None):
        self.id, self.bus, self.clock, self.cfg = leader_id, bus, clock, cfg
        self.refill_points = list(refill_points)
        self.epoch = epoch
        self.log = log or (lambda s: None)
        self.hotspots: dict[str, TrackedFire] = {}   # name kept for sim/game.py (fires keyed by hotspot id)
        self._alias: dict[str, str] = {}
        self.ext: dict[str, ExtState] = {}
        self.seekers: set[str] = set()
        self.seeker_state: dict[str, SeekerState] = {}
        self.investigations: dict[str, str] = {}      # INVESTIGATE task id -> fire id
        self.scouting = False                         # investigations run (seek phase) even before `active`
        self.assignments: dict[str, _Assignment] = {}
        self.standby: dict[str, str] = {}             # drone -> prefill / staging REFILL task id
        self.staged: dict[str, str] = {}              # drone -> refill point it waits at (until its next task)
        self.queue = FireQueue()
        self.active = False
        self.seq = 0
        self.n_tasks = 0
        self.last_tick = -1e9
        self.stats: Counter = Counter()
        self.grants: Counter = Counter()              # refill grants per point id
        bus.subscribe(SUB_ALL_HOTSPOTS, self._on_hotspot)
        bus.subscribe(SUB_ALL_HEARTBEATS, self._on_heartbeat)
        bus.subscribe(ROUTES[MessageType.TASK_COMPLETE], self._on_complete)
        bus.subscribe(ROUTES[MessageType.REFILL_REQUEST], self._on_refill)
        bus.subscribe(SUB_GROUND_COMMANDS, self._on_ground)

    @property
    def useful_l(self) -> float:
        """Litres below which a drone refills before a task (never more than the tank holds)."""
        return min(self.cfg.min_useful_l, 0.9 * self.cfg.capacity_l)

    # ---- fire bookkeeping ----------------------------------------------------------------------------------------
    def _need(self, area_m2: float) -> float:
        c = self.cfg
        return max(c.min_volume_l, area_m2 * c.l_per_m2 / c.agent_mult * c.margin)

    def _reprioritise(self, f: TrackedFire) -> None:
        f.raw_priority = fire_priority(f.hotspot, self.cfg.priority, growth_m2_min=f.growth_m2_min)
        f.priority = normalized_priority(f.raw_priority, self.cfg.priority)
        f.hotspot = f.hotspot.model_copy(update={"priority_score": f.raw_priority})
        if self._assignable(f):
            self.queue.push(f.hotspot_id, f.priority)
        else:
            self.queue.discard(f.hotspot_id)

    def _confirmed(self, f: TrackedFire) -> bool:
        return f.hotspot.status in (HotspotStatus.CONFIRMED, HotspotStatus.ASSIGNED)

    def _assignable(self, f: TrackedFire, now: float | None = None) -> bool:
        now = self.clock.now() if now is None else now
        ok = self._confirmed(f) or (self.cfg.investigate and self.cfg.dispatch_on_detect
                                    and f.hotspot.status is HotspotStatus.DETECTED)
        if not ok or f.remaining_l <= 0:
            return False
        if self.cfg.max_per_fire <= 1 and (f.inflight or f.verify_until >= now):
            return False   # one drone at a time: identical goals make CBS replan endlessly; wait for its VERIFY
        return len(f.inflight) < self.cfg.max_per_fire

    def _growth(self, f: TrackedFire) -> None:
        s = f.samples
        if len(s) < 3 or s[-1][0] - s[0][0] < 20.0:
            return
        n = len(s)
        mt = sum(t for t, _ in s) / n
        ma = sum(a for _, a in s) / n
        den = sum((t - mt) ** 2 for t, _ in s)
        if den <= 0:
            return
        slope = sum((t - mt) * (a - ma) for t, a in s) / den          # m2 per sim second
        f.growth_m2_min = slope * 60.0 / max(self.cfg.fire_time_scale, 1e-6)

    def _on_hotspot(self, msg: SwarmMessage) -> None:
        h: Hotspot = msg.body.hotspot
        if self.cfg.geofence_radius_m is not None and math.hypot(h.position.x, h.position.y) > self.cfg.geofence_radius_m:
            return   # the camera sees past the geofence; nothing out there can be tasked
        if msg.sender_id in self.ext and h.hotspot_id in self.hotspots:
            self._on_verify(msg.sender_id, h)
            return
        f0 = self.hotspots.get(h.hotspot_id)
        # the investigating seeker's close-range report (status != DETECTED: its tracker reports are always DETECTED)
        if f0 is not None and f0.investigator == msg.sender_id and h.status is not HotspotStatus.DETECTED:
            f0.close_range = h            # the verdict itself arrives with the INVESTIGATE TaskComplete
            return
        self.seekers.add(msg.sender_id)
        now = self.clock.now()
        key = self._alias.get(h.hotspot_id)
        if key is None:
            for k, t in self.hotspots.items():
                p = t.hotspot.position
                if math.hypot(p.x - h.position.x, p.y - h.position.y) <= self.cfg.merge_radius_m:
                    key = k
                    break
        if key is None:
            key = h.hotspot_id
            status = (HotspotStatus.CONFIRMED if not self.cfg.investigate and h.confidence >= self.cfg.confirm_threshold
                      else HotspotStatus.DETECTED)
            f = TrackedFire(h.model_copy(update={"status": status, "position": Position(x=h.position.x,
                                                                                        y=h.position.y, z=0.0)}),
                            self._need(h.est_area_m2))
            self.hotspots[key] = f
            if status is HotspotStatus.CONFIRMED:
                self.stats["confirmed"] += 1
                f.confirmed_by = "detector"
        else:
            f = self.hotspots[key]
            old = f.hotspot
            conf = max(old.confidence, h.confidence)
            status = old.status
            if status in (HotspotStatus.FALSE_POSITIVE, HotspotStatus.VERIFIED_OUT):
                self._alias[h.hotspot_id] = key       # closed (rejected / declared out): don't re-open it
                return
            if not self.cfg.investigate and status is HotspotStatus.DETECTED and conf >= self.cfg.confirm_threshold:
                status = HotspotStatus.CONFIRMED
                self.stats["confirmed"] += 1
                f.confirmed_by = "detector"
            if (status is HotspotStatus.SUPPRESSED and h.confidence >= self.cfg.confirm_threshold
                    and f.suppressed_at is not None and h.last_seen > f.suppressed_at + 10.0):
                status = HotspotStatus.CONFIRMED            # rekindled: a seeker sees heat again
                f.suppressed_at = None
                f.reopened += 1
                f.need_l = f.delivered_l + f.reserved_l + self._need(h.est_area_m2)
                self.stats["reopened"] += 1
                self.log(f"{key}: rekindled, back in the queue")
            f.hotspot = old.model_copy(update={
                "confidence": conf, "status": status, "est_area_m2": max(old.est_area_m2, h.est_area_m2)
                if f.delivered_l == 0 else h.est_area_m2,
                "peak_temp_c": max(old.peak_temp_c, h.peak_temp_c), "last_seen": max(old.last_seen, h.last_seen),
                "observations": old.observations + 1})
            if f.delivered_l == 0:
                f.need_l = max(f.need_l, self._need(f.hotspot.est_area_m2))
        self._alias[h.hotspot_id] = key
        f.sources.add(msg.sender_id)
        if h.growth_rate_m2_per_min is not None:
            f.growth_m2_min = h.growth_rate_m2_per_min / max(self.cfg.fire_time_scale, 1e-6)
        if not f.samples or now - f.samples[-1][0] >= 3.0:
            f.samples.append((now, h.est_area_m2))
            f.samples = f.samples[-12:]
            if h.growth_rate_m2_per_min is None and f.delivered_l == 0:
                self._growth(f)
        self._reprioritise(f)

    def _on_verify(self, drone_id: str, h: Hotspot) -> None:
        """VERIFY report from the extinguisher that just dropped (perception.confirm.SuppressionChecker)."""
        f = self.hotspots[h.hotspot_id]
        f.verify_until = -1.0
        if h.status is HotspotStatus.SUPPRESSED:
            if f.inflight:                                   # others still on their way: let them go elsewhere
                for d in list(f.inflight):
                    self._release(d, delivered=0.0)
                    self.staged.pop(d, None)          # re-staged (or re-tasked) on the next tick
            f.hotspot = f.hotspot.model_copy(update={"status": HotspotStatus.SUPPRESSED, "assigned_to": None,
                                                     "est_area_m2": 0.0, "last_seen": h.last_seen})
            f.suppressed_at = self.clock.now()
            self.stats["suppressed"] += 1
            self.queue.discard(f.hotspot_id)
            self.log(f"{f.hotspot_id}: suppressed after {f.drops} drop(s), {f.delivered_l:.0f} L")
            return
        self.stats["verify_hot"] += 1
        f.hotspot = f.hotspot.model_copy(update={"status": HotspotStatus.CONFIRMED, "assigned_to": None,
                                                 "position": Position(x=h.position.x, y=h.position.y, z=0.0),
                                                 "est_area_m2": h.est_area_m2, "last_seen": h.last_seen})
        f.need_l = f.delivered_l + f.reserved_l + self._need(h.est_area_m2)
        self._reprioritise(f)                                # back on the heap, same formula

    # ---- confirmation ------------------------------------------------------------------------------------------
    def _confirm(self, f: TrackedFire, by: str) -> None:
        if f.hotspot.status is not HotspotStatus.DETECTED:
            return
        upd = {"status": HotspotStatus.CONFIRMED}
        cr = f.close_range
        if cr is not None:
            upd.update(position=Position(x=cr.position.x, y=cr.position.y, z=0.0),
                       est_area_m2=max(cr.est_area_m2, f.hotspot.est_area_m2) if cr.est_area_m2 else
                       f.hotspot.est_area_m2, confidence=max(cr.confidence, f.hotspot.confidence))
        f.hotspot = f.hotspot.model_copy(update=upd)
        f.review_since = None
        f.confirmed_by = by
        self.stats["confirmed"] += 1
        self.stats[f"confirmed_{by}"] += 1
        if f.delivered_l == 0:
            f.need_l = max(f.need_l, self._need(f.hotspot.est_area_m2))
        if f.inflight:                                         # dispatched on detection: clear their drop
            f.hotspot = f.hotspot.model_copy(update={"status": HotspotStatus.ASSIGNED})
            for d in f.inflight:
                h = f.hotspot.model_copy(update={"assigned_to": d})
                self._send(HotspotReport(hotspot=h), receiver=d, topic=sub_my_commands(d))
        self.log(f"{f.hotspot_id}: CONFIRMED ({by})")
        self._reprioritise(f)

    def _reject(self, f: TrackedFire, by: str) -> None:
        if f.hotspot.status is HotspotStatus.FALSE_POSITIVE:
            return
        for d in list(f.inflight):                             # recall anyone already on the way
            self._release(d, delivered=0.0)
            self.staged.pop(d, None)
        f.hotspot = f.hotspot.model_copy(update={"status": HotspotStatus.FALSE_POSITIVE, "assigned_to": None})
        f.review_since = None
        self.queue.discard(f.hotspot_id)
        self.stats["false_positive"] += 1
        self.stats[f"false_positive_{by}"] += 1
        self.log(f"{f.hotspot_id}: FALSE POSITIVE ({by})")

    def _on_investigated(self, f: TrackedFire, b: TaskComplete) -> None:
        f.investigator, f.investigate_task = None, None
        self.stats["investigated"] += 1
        if b.result_status is HotspotStatus.CONFIRMED:
            self._confirm(f, "investigate")
        elif b.result_status is HotspotStatus.FALSE_POSITIVE:
            self._reject(f, "investigate")
        elif f.hotspot.status is HotspotStatus.DETECTED:     # inconclusive: the seeker escalated to the ground
            f.review_since = self.clock.now()
            self.stats["review"] += 1
            self.log(f"{f.hotspot_id}: inconclusive at close range, waiting for the ground base")

    def _on_ground(self, msg: SwarmMessage) -> None:
        g = msg.body
        if not isinstance(g, GroundCommand) or g.target_id is None:
            return
        fid = self._alias.get(g.target_id, g.target_id)
        f = self.hotspots.get(fid)
        if f is None:
            return
        if g.command is CommandKind.CONFIRM_FIRE:
            self.stats["ground_fire"] += 1
            self._confirm(f, "ground")
        elif g.command is CommandKind.MARK_FALSE_POSITIVE:
            self.stats["ground_not_fire"] += 1
            self._reject(f, "ground")
        elif g.command is CommandKind.CONFIRM_OUT:           # humans declare fires out (hard rule 10)
            for d in list(f.inflight):
                self._release(d, delivered=0.0)
            f.hotspot = f.hotspot.model_copy(update={"status": HotspotStatus.VERIFIED_OUT, "assigned_to": None})
            self.queue.discard(fid)

    def _assign_investigations(self, now: float) -> None:
        for f in self.hotspots.values():                      # timeouts first
            if f.investigator is not None and now - f.investigate_since > self.cfg.investigate_timeout_s:
                self.investigations.pop(f.investigate_task, None)
                f.investigator, f.investigate_task = None, None
                self.stats["investigate_timeout"] += 1
            if (f.review_since is not None and f.hotspot.status is HotspotStatus.DETECTED
                    and now - f.review_since > self.cfg.review_timeout_s):
                self.stats["review_timeout"] += 1
                self._confirm(f, "timeout")
        busy = {f.investigator for f in self.hotspots.values() if f.investigator}
        free = [s for s in self.seeker_state.values() if s.drone_id not in busy
                and not (s.hb_task or "").startswith(f"{self.id}-")]
        todo = sorted((f for f in self.hotspots.values() if f.hotspot.status is HotspotStatus.DETECTED
                       and f.investigator is None and f.review_since is None), key=lambda f: -f.priority)
        for f in todo:
            if not free:
                return
            p = f.hotspot.position
            s = min(free, key=lambda s: (math.hypot(s.position.x - p.x, s.position.y - p.y), s.drone_id))
            free.remove(s)
            self.n_tasks += 1
            task = Task(task_id=f"{self.id}-T{self.n_tasks}", type=TaskType.INVESTIGATE,
                        priority=round(f.priority, 4), status=TaskStatus.ASSIGNED, assigned_to=s.drone_id,
                        target_hotspot_id=f.hotspot_id, created_at=now, epoch=self.epoch)
            self._send(HotspotReport(hotspot=f.hotspot), receiver=s.drone_id, topic=sub_my_commands(s.drone_id))
            self._send(TaskAssign(task=task, epoch=self.epoch), receiver=s.drone_id)
            f.investigator, f.investigate_task, f.investigate_since = s.drone_id, task.task_id, now
            self.investigations[task.task_id] = f.hotspot_id
            self.stats["investigate_tasks"] += 1

    # ---- drones --------------------------------------------------------------------------------------------------
    def _on_heartbeat(self, msg: SwarmMessage) -> None:
        s = msg.body.state
        if s.kind is DroneKind.SEEKER:
            st = self.seeker_state.get(s.drone_id)
            if st is None:
                st = self.seeker_state[s.drone_id] = SeekerState(s.drone_id, s.position)
            st.position, st.hb_task = s.position, s.current_task_id
            return
        if s.kind is not DroneKind.EXTINGUISHER:
            return
        e = self.ext.get(s.drone_id)
        if e is None:
            e = self.ext[s.drone_id] = ExtState(s.drone_id, s.position, s.water_remaining_l or 0.0)
        e.position, e.water_l, e.hb_task = s.position, s.water_remaining_l or 0.0, s.current_task_id
        a = self.assignments.get(s.drone_id)
        if a is not None:
            if s.current_task_id == a.task.task_id:
                a.started = True
            elif (a.started or self.clock.now() - a.sent_at > 3.0) and s.current_task_id is None:
                self._release(s.drone_id, delivered=0.0)     # executor dropped it without a TaskComplete

    def busy(self, drone_id: str) -> bool:
        if drone_id in self.assignments:
            return True
        t = self.ext[drone_id].hb_task
        return t is not None and not t.startswith(f"{self.id}-")   # e.g. VERIFY-... after a drop

    def _on_complete(self, msg: SwarmMessage) -> None:
        b: TaskComplete = msg.body
        fid = self.investigations.pop(b.task_id, None)
        if fid is not None:
            if fid in self.hotspots and self.hotspots[fid].investigate_task == b.task_id:
                self._on_investigated(self.hotspots[fid], b)
            return
        if self.standby.get(b.drone_id) == b.task_id:
            self.standby.pop(b.drone_id, None)
            return
        a = self.assignments.get(b.drone_id)
        if a is None or a.task.task_id != b.task_id:
            return
        self._release(b.drone_id, delivered=b.water_used_l or 0.0)
        if not b.success and b.notes:
            self.log(f"{b.drone_id}: {b.notes}")

    def _release(self, drone_id: str, delivered: float) -> None:
        a = self.assignments.pop(drone_id, None)
        if a is None:
            return
        f = self.hotspots.get(a.fire_id)
        if f is None:
            return
        f.inflight.pop(drone_id, None)
        if delivered > 0:
            f.delivered_l += delivered
            f.drops += 1
            self.stats["drops"] += 1
            f.verify_until = self.clock.now() + self.cfg.verify_timeout_s
        if not f.inflight and f.hotspot.status is HotspotStatus.ASSIGNED:
            f.hotspot = f.hotspot.model_copy(update={"status": HotspotStatus.CONFIRMED, "assigned_to": None})
        self._reprioritise(f)

    def _on_refill(self, msg: SwarmMessage) -> None:
        req: RefillRequest = msg.body
        a = self.assignments.get(req.drone_id)
        target = self.hotspots[a.fire_id].hotspot.position if a is not None and a.fire_id in self.hotspots else None
        pts = [p for p in self.refill_points if p.kind is RefillTargetKind.STATION] if req.needs_fuel \
            else self.refill_points
        pt, _ = best_refill(req.position, target, pts, self.cfg.speed_mps)
        if pt is None:
            return
        self.grants[pt.point_id] += 1
        self.stats[f"refill_{pt.kind.value.lower()}"] += 1
        self._send(RefillGrant(drone_id=req.drone_id, target_kind=pt.kind, target_id=pt.point_id,
                               position=pt.position, epoch=self.epoch), receiver=req.drone_id)

    # ---- outputs -------------------------------------------------------------------------------------------------
    def _send(self, body, receiver: str | None = None, topic: str | None = None) -> None:
        self.seq += 1
        msg = SwarmMessage.create(self.id, body, self.clock.now(), seq=self.seq, receiver_id=receiver,
                                  epoch=self.epoch)
        self.bus.publish(topic or topic_for(msg), msg)

    def _aim(self, f: TrackedFire) -> Position:
        """Concurrent drones on one fire aim at different parts of it (golden-angle spiral inside its radius)."""
        p = f.hotspot.position
        k = len(f.inflight)
        if self.cfg.max_per_fire <= 1 or k == 0:
            return Position(x=p.x, y=p.y, z=0.0)
        r = 0.55 * math.sqrt(max(f.hotspot.est_area_m2, 0.5) / math.pi) * math.sqrt(k / self.cfg.max_per_fire)
        return Position(x=p.x + r * math.cos(k * _GOLDEN), y=p.y + r * math.sin(k * _GOLDEN), z=0.0)

    def _assign(self, a: Assignment) -> None:
        f = self.hotspots[a.fire_id]
        now = self.clock.now()
        self.n_tasks += 1
        task = Task(task_id=f"{self.id}-T{self.n_tasks}", type=TaskType.EXTINGUISH, priority=round(f.priority, 4),
                    status=TaskStatus.ASSIGNED, assigned_to=a.drone_id, target_hotspot_id=a.fire_id,
                    created_at=now, epoch=self.epoch, volume_l=max(a.volume_l, 1.0))
        confirmed = self._confirmed(f)
        h = f.hotspot.model_copy(update={"status": HotspotStatus.ASSIGNED if confirmed else HotspotStatus.DETECTED,
                                         "assigned_to": a.drone_id, "position": self._aim(f)})
        # executors learn the target only from a HotspotReport, so send it first (same topic, in order)
        self._send(HotspotReport(hotspot=h), receiver=a.drone_id, topic=sub_my_commands(a.drone_id))
        self._send(TaskAssign(task=task, epoch=self.epoch), receiver=a.drone_id)
        f.inflight[a.drone_id] = task.volume_l
        if confirmed:
            f.hotspot = f.hotspot.model_copy(update={"status": HotspotStatus.ASSIGNED, "assigned_to": a.drone_id})
        self.assignments[a.drone_id] = _Assignment(task, a.fire_id, task.volume_l, now, a.eta_s, a.refill)
        self.standby.pop(a.drone_id, None)
        self.staged.pop(a.drone_id, None)

    def _standby(self, drone_id: str) -> None:
        """Idle + empty: top up at the nearest refill point, so the next task costs travel only."""
        if not self.cfg.prefill_idle or drone_id in self.standby or drone_id not in self.ext:
            return
        pt, _ = best_refill(self.ext[drone_id].position, None, self.refill_points, self.cfg.speed_mps)
        if pt is not None:
            self._refill_task(drone_id, pt)
            self.stats["prefill"] += 1

    def _stage(self, drone_id: str) -> None:
        """Idle + loaded with nothing in the queue: pre-position at the refill point with the fewest drones waiting
        (ties: nearest). Fires found later anywhere on the map then have a loaded drone and water close by.
        With the base as the only refill point (no lakes) this is a no-op: everyone waits at the base."""
        if drone_id in self.standby or drone_id not in self.ext or not self.refill_points:
            return
        pos = self.ext[drone_id].position
        counts = Counter(self.staged.values())
        pt = min(self.refill_points, key=lambda p: (counts[p.point_id],
                                                    math.hypot(p.position.x - pos.x, p.position.y - pos.y)))
        self._refill_task(drone_id, pt)
        self.stats["staged"] += 1

    def _refill_task(self, drone_id: str, pt: RefillPoint) -> None:
        """REFILL task to `pt` (fill up and wait there). The next EXTINGUISH task supersedes it."""
        self.n_tasks += 1
        kw = ({"target_station_id": pt.point_id} if pt.kind is RefillTargetKind.STATION
              else {"target_water_source_id": pt.point_id})
        task = Task(task_id=f"{self.id}-T{self.n_tasks}", type=TaskType.REFILL, priority=0.1,
                    status=TaskStatus.ASSIGNED, assigned_to=drone_id, created_at=self.clock.now(),
                    epoch=self.epoch, **kw)
        self._send(TaskAssign(task=task, epoch=self.epoch), receiver=drone_id)
        self.standby[drone_id] = task.task_id
        self.staged[drone_id] = pt.point_id

    def _views(self) -> tuple[dict[str, FireView], list[DroneView]]:
        fires = {}
        for fid, f in self.hotspots.items():
            if self._assignable(f):
                fires[fid] = FireView(fid, f.hotspot.position, f.priority, f.remaining_l,
                                      self.cfg.max_per_fire - len(f.inflight))
        idle = [DroneView(e.drone_id, e.position, e.water_l, self.cfg.capacity_l, self.cfg.speed_mps)
                for e in sorted(self.ext.values(), key=lambda e: e.drone_id) if not self.busy(e.drone_id)]
        return fires, idle

    def tick(self, now: float) -> None:
        if not (self.active or self.scouting) or now - self.last_tick < self.cfg.tick_s:
            return
        self.last_tick = now
        if self.cfg.investigate:
            self._assign_investigations(now)
        if not self.active:
            return
        for fid, f in self.hotspots.items():      # fires whose VERIFY window just expired rejoin the heap
            if fid not in self.queue and self._assignable(f, now):
                self.queue.push(fid, f.priority)
        fires, idle = self._views()
        if not idle:
            return
        for a in allocate(self.queue, fires, idle, self.refill_points, self.useful_l):
            self._assign(a)
        if self.cfg.prefill_idle or self.cfg.stage_idle:
            for d in idle:
                if d.drone_id in self.assignments or d.drone_id in self.standby:
                    continue
                if d.water_l < self.useful_l:
                    self._standby(d.drone_id)
                elif self.cfg.stage_idle and d.drone_id not in self.staged:
                    self._stage(d.drone_id)

    # ---- views for the UI ----------------------------------------------------------------------------------------
    def preview_plan(self) -> list[tuple[str, str, float]]:
        """What the allocator would do right now with every drone free (planning screen)."""
        fires = {fid: FireView(fid, f.hotspot.position, f.priority, f.need_l - f.delivered_l, self.cfg.max_per_fire)
                 for fid, f in self.hotspots.items()
                 if f.hotspot.status in (HotspotStatus.CONFIRMED, HotspotStatus.ASSIGNED)}
        q = FireQueue()
        for fid, fv in fires.items():
            q.push(fid, fv.priority)
        drones = [DroneView(e.drone_id, e.position, e.water_l, self.cfg.capacity_l, self.cfg.speed_mps)
                  for e in sorted(self.ext.values(), key=lambda e: e.drone_id)]
        return [(a.drone_id, a.fire_id, a.volume_l) for a in allocate(q, fires, drones, self.refill_points,
                                                                      self.useful_l)]

    def queue_view(self, n: int = 8) -> list[tuple[str, float, float]]:
        """Top of the priority queue: (fire id, priority 0..1, litres still unassigned)."""
        rows = [(fid, f.priority, max(f.remaining_l, 0.0)) for fid, f in self.hotspots.items()
                if f.hotspot.status in (HotspotStatus.CONFIRMED, HotspotStatus.ASSIGNED) and f.remaining_l > 0]
        return sorted(rows, key=lambda r: -r[1])[:n]

"""DEMO-mode extinguisher: a small field-refill attacker that mixes foam concentrate into water picked up anywhere.

It carries ~15 L per drop, so it refills after EVERY drop: at the base, or at any lake the leader validated. The
concentrate tank is not modelled (at < 1 % dose it lasts dozens of loads; the brief says assume it is always there).

Execution only; every decision comes from the leader (core.leader.SwarmLeader) over the bus, the same protocol the
autonomy adapter uses for the heavy extinguishers:
    HotspotReport (target) + TaskAssign(EXTINGUISH, volume_l)   -> fly there (RefillRequest first if not loaded)
    RefillGrant                                                 -> fill at that point, then go on with the task
    TaskAssign(REFILL)                                          -> stand-by top-up; a later TaskAssign supersedes it
It reports TaskComplete after the drop, re-images the spot (VerifyPass: real SuppressionChecker), then publishes
the VERIFY HotspotReport. Heartbeats at 2 Hz.

Navigation is direct, at a per-drone transit altitude layer (vertical separation). That stands in for PX4's own
position control / collision avoidance; we deliberately skip the CBS planner here (GitHub issue #21).
"""
from __future__ import annotations

import math

from contracts import (
    DroneKind,
    DroneState,
    DroneStatus,
    ExtinguishingAgent,
    Heartbeat,
    Hotspot,
    HotspotReport,
    Position,
    Powertrain,
    RefillGrant,
    RefillRequest,
    RefillTargetKind,
    SwarmMessage,
    SwarmRole,
    Task,
    TaskAssign,
    TaskComplete,
    TaskType,
    sub_my_commands,
)
from sim.agents import _SENSORS_OK, VerifyPass, _Publisher
from sim.vehicles import SimAutopilot, SimPayload

__all__ = ["STATUS_TEXT", "AttackerAgent"]

STATUS_TEXT = {
    "IDLE": "standing by", "TO_FILL": "flying to refill", "FILL": "refilling", "TO_FIRE": "flying to fire",
    "DROP": "dropping", "VERIFY": "verifying the drop", "WAIT": "waiting for a refill grant",
}


class AttackerAgent:
    def __init__(self, drone_id: str, index: int, world, ap: SimAutopilot, payload: SimPayload,
                 refill_points: dict[str, tuple[Position, RefillTargetKind, str]]):
        self.id, self.index, self.world, self.ap, self.payload = drone_id, index, world, ap, payload
        self.points = refill_points            # point id -> (position, kind, display name)
        cfg = world.cfg
        self.cfg = cfg
        self.spec = cfg.extinguisher_spec
        self.layer_z = cfg.transit_agl_m + 4.0 * (index % 4)       # vertical separation between neighbours
        self.pub = _Publisher(drone_id, world.bus, world.clock)
        self.verify = VerifyPass(drone_id, world, self.spec, crop=128, first_frame_s=0.3, timeout_s=2.5)
        self.mode = "IDLE"
        self.task: Task | None = None
        self.targets: dict[str, Hotspot] = {}
        self.target: Hotspot | None = None
        self.aim: Position | None = None
        self.fill_point: tuple[str, Position, RefillTargetKind] | None = None
        self.fill_t = 0.0
        self.drop_used = 0.0
        self.max_epoch = 0
        self.next_hb_t = 0.0
        self.stats = {"drops": 0, "fills_lake": 0, "fills_base": 0, "litres": 0.0}
        a = 2.0 * math.pi * ((index * 0.618) % 1.0)
        self.slot = (math.cos(a), math.sin(a))                   # spread drones over a refill point
        world.bus.subscribe(sub_my_commands(drone_id), self._on_cmd)

    # ---- bus -----------------------------------------------------------------------------------------------------
    def _on_cmd(self, msg: SwarmMessage) -> None:
        if msg.epoch is not None:
            if msg.epoch < self.max_epoch:
                return                                           # stale leader (hard rule 7)
            self.max_epoch = msg.epoch
        b = msg.body
        if isinstance(b, HotspotReport):
            self.targets[b.hotspot.hotspot_id] = b.hotspot
        elif isinstance(b, TaskAssign):
            self._accept(b.task)
        elif isinstance(b, RefillGrant) and self.mode == "WAIT":
            pos = b.position
            self.fill_point = (b.target_id, pos, b.target_kind)
            self._go_fill()

    def _accept(self, task: Task) -> None:
        if self.mode in ("DROP", "VERIFY"):
            # finish the drop / re-image first; the leader already freed us, so take the new task right after
            self._queued = task
            return
        self.task = task
        self._queued = None
        if task.type is TaskType.REFILL:
            pid = task.target_water_source_id or task.target_station_id
            pos, kind, _ = self.points[pid]
            self.fill_point = (pid, pos, kind)
            self.target = None
            self._go_fill()
            return
        self.target = self.targets.get(task.target_hotspot_id)
        if self.target is None:
            self._complete(False, 0.0, "no target picture for task")
            return
        self.aim = self._aim_point(self.target.position)
        if self.payload.water_l + 1e-6 < min(task.volume_l or self.payload.capacity_l, self.payload.capacity_l) * 0.98:
            if self.mode in ("FILL", "TO_FILL") and self.fill_point is not None:
                return                                          # already on the way to water: keep going
            self.mode = "WAIT"
            self.pub.publish(RefillRequest(drone_id=self.id, position=self.ap.position(),
                                           water_remaining_l=self.payload.water_l,
                                           agent=ExtinguishingAgent.CLASS_A_FOAM))
        else:
            self._go_fire()

    _queued: Task | None = None

    def _complete(self, success: bool, used: float, notes: str | None = None) -> None:
        if self.task is not None:
            self.pub.publish(TaskComplete(task_id=self.task.task_id, drone_id=self.id, success=success,
                                          water_used_l=used, notes=notes))
        self.task = None

    # ---- navigation ----------------------------------------------------------------------------------------------
    def _aim_point(self, p: Position) -> Position:
        """Aim upwind of the target so the falling water drifts onto it (flight controller wind estimate)."""
        cfg, fire = self.cfg, self.world.fire
        fall_s = math.sqrt(2.0 * cfg.drop_agl_m / 9.81)
        k = cfg.wind_speed_mps * fall_s * cfg.drop_drift_factor
        return Position(x=p.x - fire.wind_dir[0] * k, y=p.y - fire.wind_dir[1] * k, z=0.0)

    def _hdist(self, p: Position) -> float:
        return math.hypot(self.ap.x - p.x, self.ap.y - p.y)

    def _go_fill(self) -> None:
        self.mode = "TO_FILL"
        self.fill_t = 0.0

    def _go_fire(self) -> None:
        self.mode = "TO_FIRE"

    def _fill_spot(self) -> Position:
        _, pos, kind = self.fill_point
        r = 4.0 if kind is RefillTargetKind.STATION else 12.0
        return Position(x=pos.x + r * self.slot[0], y=pos.y + r * self.slot[1], z=0.0)

    # ---- loop ----------------------------------------------------------------------------------------------------
    def tick(self, now: float, dt: float) -> None:
        ap, cfg = self.ap, self.cfg
        m = self.mode
        if m == "TO_FILL":
            spot = self._fill_spot()
            d = self._hdist(spot)
            if d > ap.speed_mps() ** 2 / (2 * self.spec.max_accel_mps2) + 12.0:
                ap.goto(Position(x=spot.x, y=spot.y, z=self.layer_z))
            else:
                ap.goto(Position(x=spot.x, y=spot.y, z=cfg.attacker_fill_agl_m))
                if d < 1.5 and ap.z <= cfg.attacker_fill_agl_m + 0.6:
                    self.mode = "FILL"
        elif m == "FILL":
            self.fill_t += dt
            if self.fill_t >= cfg.attacker_fill_s or self.payload.water_l >= self.payload.capacity_l - 1e-6:
                if self.payload.water_l < self.payload.capacity_l - 1e-6:
                    self.payload.fill(self.payload.capacity_l, ExtinguishingAgent.CLASS_A_FOAM)
                    key = "fills_base" if self.fill_point[2] is RefillTargetKind.STATION else "fills_lake"
                    self.stats[key] += 1
                if self.task is not None and self.task.type is TaskType.EXTINGUISH:
                    self._go_fire()
                else:
                    self._complete(True, 0.0)                    # stand-by refill done
                    self.mode = "IDLE"
                    ap.goto(Position(x=ap.x, y=ap.y, z=self.layer_z))
        elif m == "TO_FIRE":
            aim = self.aim
            d = self._hdist(aim)
            stop = ap.speed_mps() ** 2 / (2 * self.spec.max_accel_mps2) + 12.0
            if d > stop:
                ap.goto(Position(x=aim.x, y=aim.y, z=self.layer_z), self.spec.cruise_speed_mps)
            else:
                ap.goto(Position(x=aim.x, y=aim.y, z=cfg.drop_agl_m))
                if d < 2.5 and ap.z <= cfg.drop_agl_m + 2.0 and ap.speed_mps() < 3.0:
                    self.mode = "DROP"
                    self.drop_used = 0.0
        elif m == "DROP":
            rate = self.payload.capacity_l / max(cfg.attacker_drop_s, 0.1)
            used = self.payload.release(min(rate * dt, self.payload.water_l), ap.position())
            self.drop_used += used
            if self.payload.water_l <= 1e-6:
                self.stats["drops"] += 1
                self.stats["litres"] += self.drop_used
                target = self.target
                self._complete(True, self.drop_used)
                ap.goto(Position(x=ap.x, y=ap.y, z=cfg.drop_agl_m + 5.0))
                if target is not None:
                    self.verify.start(target, now)
                    self.mode = "VERIFY"
                else:
                    self.mode = "IDLE"
        elif m == "VERIFY":
            report = self.verify.step(now, ap)
            if report is not None:
                self.pub.publish(HotspotReport(hotspot=report))
                self.mode = "IDLE"
                self.target = None
                ap.goto(Position(x=ap.x, y=ap.y, z=self.layer_z))
                if self._queued is not None:
                    t, self._queued = self._queued, None
                    self._accept(t)
        ap.update(dt)
        if now >= self.next_hb_t:
            self.next_hb_t = now + 0.5
            self.pub.publish(self._heartbeat(now))

    # ---- telemetry -----------------------------------------------------------------------------------------------
    def _heartbeat(self, now: float):
        ap, spec = self.ap, self.spec
        status = {"IDLE": DroneStatus.IDLE, "TO_FILL": DroneStatus.TRANSIT, "FILL": DroneStatus.REFILLING,
                  "TO_FIRE": DroneStatus.TRANSIT, "DROP": DroneStatus.EXTINGUISHING,
                  "VERIFY": DroneStatus.EXTINGUISHING, "WAIT": DroneStatus.IDLE}[self.mode]
        task_id = (self.task.task_id if self.task is not None
                   else f"VERIFY-{self.verify.hotspot.hotspot_id}" if self.verify.active else None)
        st = DroneState(drone_id=self.id, kind=DroneKind.EXTINGUISHER, position=ap.position(),
                        heading_deg=ap.heading_deg(), speed_mps=ap.speed_mps(), powertrain=Powertrain.HYBRID,
                        battery_pct=100.0, fuel_remaining_kg=ap.fuel_kg,
                        fuel_pct=100.0 * ap.fuel_kg / spec.fuel_capacity_kg, generator_output_kw=0.0,
                        water_remaining_l=self.payload.water_l, agent=ExtinguishingAgent.CLASS_A_FOAM,
                        status=status, current_task_id=task_id, leader_id="leader", epoch=self.max_epoch,
                        role=SwarmRole.FOLLOWER, sensors_ok=dict(_SENSORS_OK), link_quality=1.0, timestamp=now)
        return Heartbeat(state=st)

    def activity(self) -> str:
        m = self.mode
        tid = self.target.hotspot_id if self.target is not None else None
        prio = f" · priority {self.task.priority:.2f}" if self.task is not None and self.task.priority else ""
        if m in ("TO_FILL", "FILL") and self.fill_point is not None:
            name = self.points.get(self.fill_point[0], (None, None, self.fill_point[0]))[2]
            what = "Refilling at" if m == "FILL" else "Flying to"
            return f"{what} {name}" + (f" → then {tid}{prio}" if tid else " (stand-by top-up)")
        if m == "TO_FIRE" and tid:
            return f"Flying to {tid} with {self.payload.water_l:.0f} L foam{prio}"
        if m == "DROP" and tid:
            return f"Dropping {self.payload.capacity_l:.0f} L on {tid}"
        if m == "VERIFY" and self.verify.hotspot is not None:
            return f"Verifying {self.verify.hotspot.hotspot_id} with its thermal camera"
        if m == "WAIT":
            return "Asking the leader where to refill"
        return f"Standing by · {self.payload.water_l:.0f} L aboard"

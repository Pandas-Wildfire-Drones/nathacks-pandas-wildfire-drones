"""Drone agents: thin glue between the sim and the REAL perception / autonomy code.

SeekerAgent        flies a lawnmower (autonomy.coverage_search) or follows the player's pose; every camera tick it
                   renders a ThermalFrame and runs perception.pipeline.SeekerPerception (FUSED: threshold + CNN)
                   in a worker thread; hotspots are published as HotspotReport after a fixed latency.
ExtinguisherAgent  autonomy.swarm_adapter.SwarmAutonomyAdapter + DroneController (+ MissionCoordinator/CBS,
                   water-drop gate, refill) on a SimAutopilot / SimPayload.
PlayerExtinguisher the human's extinguisher: pose from the client, water released while Space is held.
"""
from __future__ import annotations

import math
import zlib
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field

import numpy as np

from autonomy.drone_controller import DroneController
from autonomy.mission_coordinator import MissionPlanningContext
from autonomy.swarm_adapter import SwarmAutonomyAdapter
from autonomy.water_drop import WaterDropLimits
from autonomy.coverage_search import generate_coverage_waypoints
from contracts import (
    CameraModel, DroneKind, DroneState, DroneStatus, ExtinguishingAgent, GridSpec, Heartbeat, HotspotReport,
    HotspotStatus, Pose, Position, Powertrain, RefillStation, SwarmMessage, SwarmRole, ThermalFrame, Zone, topic_for,
)
from perception.confirm import SuppressionChecker
from perception.frame_context import FrameContext
from perception.pipeline import SeekerPerception
from sim import thermal
from sim.vehicles import SimAutopilot, SimPayload

__all__ = ["SeekerAgent", "ExtinguisherAgent", "PlayerExtinguisher", "planning_context"]

_SENSORS_OK = {"thermal": True, "lidar": True, "gnss_rtk": True, "rangefinder": True, "comms": True,
               "compute": True, "payload": True, "power": True}


def _status_for(v: float, default: DroneStatus) -> DroneStatus:
    return default if v > 0.5 else DroneStatus.IDLE


class _Publisher:
    def __init__(self, drone_id: str, bus, clock):
        self.drone_id, self.bus, self.clock, self.seq = drone_id, bus, clock, 0

    def publish(self, body, receiver: str | None = None, epoch: int | None = None) -> None:
        self.seq += 1
        msg = SwarmMessage.create(self.drone_id, body, self.clock.now(), seq=self.seq, receiver_id=receiver,
                                  epoch=epoch)
        self.bus.publish(topic_for(msg), msg)


# ------------------------------------------------------------------------------------------------------------------
@dataclass
class _Pending:
    due: float
    future: Future
    timestamp: float


class SeekerAgent:
    def __init__(self, drone_id: str, world, autopilot: SimAutopilot, perception: SeekerPerception,
                 executor: Executor, player: bool = False):
        self.id, self.world, self.ap, self.perception = drone_id, world, autopilot, perception
        self.executor, self.player = executor, player
        self.pub = _Publisher(drone_id, world.bus, world.clock)
        self.waypoints: list[Position] = []
        self.wp_i = 0
        self.zone: Zone | None = None
        self.searching = False
        self.next_frame_t = 0.0
        self.next_hb_t = 0.0
        self.pending: list[_Pending] = []
        self._prev: Future | None = None
        self.frame_i = 0
        self.last_preview: dict | None = None
        self.n_frames = 0
        spec = world.cfg.seeker_spec
        self.camera = spec.thermal.camera
        self.saturation_c = spec.thermal.saturation_temp_c
        self.netd_c = spec.thermal.netd_mk / 1000.0

    # lawnmower over the assigned zone, using the autonomy coverage generator (scan lines along x)
    def assign_zone(self, zone: Zone, agl_m: float, overlap: float) -> None:
        self.zone = zone
        spacing = self.camera.swath_m(agl_m) * (1.0 - overlap)
        # the generator scans horizontal lines; our camera swath is across-track, so spacing = swath works
        self.waypoints = generate_coverage_waypoints(zone, spacing, z_m=agl_m)
        self.wp_i = 0

    def start_search(self) -> None:
        self.searching = True
        if self.waypoints and not self.player:
            self.ap.goto(self.waypoints[0], self.world.cfg.lawnmower_speed_mps)

    def stop_search(self, rtb: bool = True) -> None:
        self.searching = False
        if rtb and not self.player:
            self.ap.goto(Position(x=self.ap.start.x, y=self.ap.start.y, z=self.world.cfg.transit_agl_m))

    def tick(self, now: float, dt: float) -> None:
        cfg = self.world.cfg
        if not self.player:
            if self.searching and self.waypoints and self.ap.at_target(2.0):
                self.wp_i += 1
                if self.wp_i < len(self.waypoints):
                    self.ap.goto(self.waypoints[self.wp_i], cfg.lawnmower_speed_mps)
                else:
                    self.searching = False   # done: hold over the last point
            self.ap.update(dt)
        # camera
        if self.searching and now >= self.next_frame_t:
            self.next_frame_t = now + 1.0 / cfg.perception_hz
            self._capture(now)
        self._collect(now)
        if now >= self.next_hb_t:
            self.next_hb_t = now + 0.5
            self.pub.publish(Heartbeat(state=self.state(now)))

    def _capture(self, now: float) -> None:
        p = self.ap.position()
        agl = p.z - 0.0
        if agl < 5.0:
            return
        pose = Pose(position=p, yaw_deg=self.ap.heading_deg())
        self.frame_i += 1
        job = thermal.prepare(self.world.static, self.world.fire, pose, self.camera,
                              noise_seed=zlib.crc32(f"{self.id}:{self.frame_i}".encode()),
                              netd_c=self.netd_c, saturation_c=self.saturation_c)
        friendly = tuple(a.ap.position() for a in self.world.all_drone_agents() if a is not self)
        ctx = FrameContext(agl_m=agl, saturation_c=self.saturation_c, friendly_positions=friendly)
        fut = self.executor.submit(self._run, job, now, ctx, self._prev)
        self._prev = fut
        self.pending.append(_Pending(now + self.world.cfg.perception_latency_s, fut, now))

    def _run(self, job, now, ctx, prev: Future | None):
        temp = thermal.render(self.world.static, job, self.world.cfg.fire_cell_m)
        if prev is not None:      # SeekerPerception (tracker, map) is stateful: process this seeker's frames in order
            prev.result()
        frame = ThermalFrame(drone_id=self.id, timestamp=now, temp_c=temp, pose=job.pose, camera=job.camera)
        hotspots = self.perception.process(frame, ctx)
        return temp, hotspots, job.pose

    def _collect(self, now: float) -> None:
        while self.pending and (self.pending[0].due <= now or not self.searching and self.pending[0].future.done()):
            item = self.pending.pop(0)
            temp, hotspots, pose = item.future.result()   # blocks if the worker is late: keeps runs deterministic
            self.n_frames += 1
            for h in hotspots:
                self.pub.publish(HotspotReport(hotspot=h))
            self.last_preview = {"temp": temp, "t": item.timestamp, "pose": pose}

    def drain(self) -> None:
        for item in self.pending:
            item.future.result()
        self._collect(float("inf"))

    def state(self, now: float) -> DroneState:
        ap = self.ap
        return DroneState(drone_id=self.id, kind=DroneKind.SEEKER, position=ap.position(),
                          heading_deg=ap.heading_deg(), speed_mps=ap.speed_mps(), powertrain=Powertrain.ELECTRIC,
                          battery_pct=ap.battery_pct(), endurance_remaining_s=ap.endurance_remaining_s(),
                          status=DroneStatus.SCANNING if self.searching else DroneStatus.RETURNING,
                          sensors_ok=dict(_SENSORS_OK, payload=True), link_quality=1.0, timestamp=now, epoch=1)


# ------------------------------------------------------------------------------------------------------------------
def planning_context(cfg, base: Position, band: tuple[float, float] | None = None) -> MissionPlanningContext:
    """Occupancy grid for autonomy's CBS/A*: free inside the arena circle, blocked outside (geofence)."""
    R, res = cfg.arena_radius_m, cfg.plan_cell_m
    n = int(math.ceil(2 * R / res))
    grid = GridSpec(origin=Position(x=-R, y=-R, z=cfg.transit_agl_m), resolution_m=res, rows=n, cols=n)
    occ = []
    for r in range(n):
        row = []
        for c in range(n):
            x, y = -R + (c + 0.5) * res, -R + (r + 0.5) * res
            row.append(0 if math.hypot(x, y) <= R - 2 else 1)
        occ.append(row)
    return MissionPlanningContext(occupancy_grid=occ, grid_spec=grid, base_position=base,
                                  platform_spec=cfg.extinguisher_spec, minimum_separation_m=6.0, time_step_s=1.0,
                                  altitude_band_m=band)


class ExtinguisherAgent:
    def __init__(self, drone_id: str, world, autopilot: SimAutopilot, payload: SimPayload, station: RefillStation,
                 context: MissionPlanningContext, limits: WaterDropLimits):
        self.id, self.world, self.ap, self.payload = drone_id, world, autopilot, payload
        self.station, self.limits = station, limits
        self.pub = _Publisher(drone_id, world.bus, world.clock)
        spec = world.cfg.extinguisher_spec
        self.state_obj = DroneState(
            drone_id=drone_id, kind=DroneKind.EXTINGUISHER, position=autopilot.position(),
            powertrain=Powertrain.HYBRID, battery_pct=100.0, fuel_remaining_kg=autopilot.fuel_kg,
            fuel_pct=100.0, generator_output_kw=0.0, water_remaining_l=payload.water_l,
            agent=ExtinguishingAgent.WATER, sensors_ok=dict(_SENSORS_OK), link_quality=1.0,
            timestamp=world.clock.now(), epoch=1, leader_id="leader", role=SwarmRole.FOLLOWER)
        self.adapter = SwarmAutonomyAdapter(
            self.state_obj, DroneController(limits, autopilot, payload), world.bus, planning_context=context,
            altitude_agl_provider=lambda d: d.position.z - 0.0,
            refill_target_resolver=lambda g: station if g.target_id == station.station_id else None)
        self.spec = spec
        self.next_hb_t = 0.0
        self.last_status = "IDLE"
        self.controllers_used = 1
        # VERIFY after each drop with the extinguisher's own nadir thermal (aim/verify camera, context.md 6.3).
        # A centre crop of the 320x256 / 50 deg camera keeps the same GSD at a fraction of the render cost.
        full = spec.thermal.camera
        crop = 192
        hfov = math.degrees(2 * math.atan(math.tan(math.radians(full.hfov_deg) / 2) * crop / full.width_px))
        self.verify_camera = CameraModel(width_px=crop, height_px=crop, hfov_deg=hfov)
        self.verify: dict | None = None
        self.verify_seq = 0

    def _sync(self, now: float) -> None:
        s, ap = self.state_obj, self.ap
        s.position = ap.position()
        s.heading_deg = ap.heading_deg()
        s.speed_mps = ap.speed_mps()
        s.fuel_remaining_kg = ap.fuel_kg
        s.fuel_pct = 100.0 * ap.fuel_kg / self.spec.fuel_capacity_kg
        s.water_remaining_l = self.payload.water_l
        s.endurance_remaining_s = ap.endurance_remaining_s()
        task = self.adapter.state.current_task
        s.current_task_id = (task.task_id if task is not None
                             else f"VERIFY-{self.verify['hotspot'].hotspot_id}" if self.verify is not None else None)
        s.status = (DroneStatus.REFILLING if self.adapter.state.refill_request_outstanding
                    else DroneStatus.EXTINGUISHING if task is not None
                    else _status_for(ap.speed_mps(), DroneStatus.TRANSIT))
        s.timestamp = now

    def _verify_step(self, now: float) -> None:
        v = self.verify
        if now < v["next_t"]:
            return
        v["next_t"] = now + 0.5
        p = self.ap.position()
        pose = Pose(position=p, yaw_deg=self.ap.heading_deg())
        self.verify_seq += 1
        job = thermal.prepare(self.world.static, self.world.fire, pose, self.verify_camera,
                              noise_seed=zlib.crc32(f"{self.id}:v{self.verify_seq}".encode()),
                              netd_c=self.spec.thermal.netd_mk / 1000.0, saturation_c=self.spec.thermal.saturation_temp_c)
        temp = thermal.render(self.world.static, job, self.world.cfg.fire_cell_m)
        frame = ThermalFrame(drone_id=self.id, timestamp=now, temp_c=temp, pose=pose, camera=self.verify_camera)
        gsd = self.verify_camera.ground_sample_distance_m(max(p.z, 1.0))
        h = v["hotspot"]
        v["checker"].add_frame(frame, h.position, gsd)
        v["last"] = (temp, gsd, p)
        if len(v["checker"]._hot) < v["checker"].min_frames and now < v["deadline"]:
            return
        res = v["checker"].result()
        temp, gsd, p = v["last"]
        hot = temp >= float(np.median(temp)) + v["checker"].hot_margin_c
        area = float(hot.sum()) * gsd * gsd
        if res.status is HotspotStatus.SUPPRESSED and area < 1.0:
            report = h.model_copy(update={"status": HotspotStatus.SUPPRESSED, "est_area_m2": 0.0,
                                          "last_seen": now, "reported_by": self.id})
        else:   # still hot (here or at the fire's edge): re-aim at the remaining heat
            vv, uu = np.nonzero(hot)
            pos = h.position
            if len(uu):
                xs, ys = thermal.ground_coords(pose, self.verify_camera, us=[float(np.mean(uu))],
                                               vs=[float(np.mean(vv))])
                pos = Position(x=float(xs[0, 0]), y=float(ys[0, 0]), z=0.0)
            report = h.model_copy(update={"status": HotspotStatus.CONFIRMED, "position": pos,
                                          "est_area_m2": max(area, 0.5), "assigned_to": None,
                                          "last_seen": now, "reported_by": self.id})
        self.pub.publish(HotspotReport(hotspot=report))
        self.verify = None

    def tick(self, now: float, dt: float) -> None:
        self.ap.update(dt)
        self._sync(now)
        target_before = self.adapter.state.current_hotspot
        result = self.adapter.step()
        if result is not None:
            self.last_status = result.status
            if result.status == "DONE" and result.water_used_l > 0 and target_before is not None:
                self.verify = {"hotspot": target_before, "checker": SuppressionChecker(radius_m=3.0),
                               "next_t": now + 0.6, "deadline": now + 5.0}
        if self.verify is not None:
            self._verify_step(now)
        # DroneController is single-use (its terminal result never clears): give the adapter a fresh one
        ctl = self.adapter.controller
        if self.adapter.state.current_task is None and not ctl.ready_for_mission:
            self.adapter.controller = DroneController(self.limits, self.ap, self.payload)
            self.controllers_used += 1
        self._sync(now)
        if now >= self.next_hb_t:
            self.next_hb_t = now + 0.5
            self.pub.publish(Heartbeat(state=self.state_obj.model_copy()))


@dataclass
class PlayerExtinguisher:
    """The human's extinguisher. Not autonomous: the player flies it and holds Space to release water."""

    drone_id: str
    ap: SimAutopilot
    payload: SimPayload
    release_lps: float
    refill_lps: float
    base: Position
    dropping: bool = False
    refilling: bool = False
    stats: dict = field(default_factory=lambda: {"dropped_l": 0.0})

    def tick(self, now: float, dt: float, apply_water) -> None:
        if self.dropping and self.payload.water_l > 0:
            vol = min(self.release_lps * dt, self.payload.water_l)
            self.payload.water_l -= vol
            self.stats["dropped_l"] += vol
            p = self.ap.position()
            apply_water(p.x, p.y, vol, p.z)
        p = self.ap.position()
        near_pad = math.hypot(p.x - self.base.x, p.y - self.base.y) < 10.0 and p.z < 14.0
        self.refilling = near_pad and self.payload.water_l < self.payload.capacity_l - 0.01
        if self.refilling:
            self.payload.fill(self.refill_lps * dt)

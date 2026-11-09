"""One simulated world: static map + fire + bus + drones + leader. A session runs up to three of them in lockstep:

  human     player seeker + player extinguisher + 9 bot extinguishers (HumanCommander)
  swarm     N lawnmower seekers + 10 extinguishers (StubLeader with the stub swarm-brain)
  baseline  fire only (no drones): the denominator of "% fire left"
"""
from __future__ import annotations

import math
from collections import deque
from concurrent.futures import Executor

import numpy as np

from autonomy.water_drop import WaterDropLimits
from contracts import (
    SUB_GROUND_ALERTS, ExtinguishingAgent, GridSpec, Position, RefillStation,
)
from perception.detectors import FusedDetector, load_classifier
from perception.pipeline import SeekerPerception
from sim.agents import ExtinguisherAgent, PlayerExtinguisher, SeekerAgent, planning_context
from sim.brain_stub import DummyRefillAssigner, DummyTaskAssigner, DummyZoneAllocator, StubLeader
from sim.bots import HumanCommander
from sim.bus import SimBus, SimClock
from sim.config import GameConfig
from sim.fire import FireModel
from sim.static import WorldStatic
from sim.vehicles import DwellZone, SimAutopilot, SimPayload

__all__ = ["World", "MODEL_PATH"]

MODEL_PATH = "perception/models/fire_cnn_mix.onnx"   # perception RESULTS.md: recommended model (mix_v2)


def _pads(base: Position, n: int, spacing: float = 7.0) -> list[Position]:
    cols = 5
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        out.append(Position(x=base.x + (c - (cols - 1) / 2) * spacing, y=base.y + 6.0 + r * spacing, z=0.0))
    return out


class World:
    def __init__(self, cfg: GameConfig, static: WorldStatic, kind: str, executor: Executor | None,
                 n_seekers: int = 1, model_path: str = MODEL_PATH):
        self.cfg, self.static, self.kind = cfg, static, kind
        self.clock = SimClock()
        self.bus = SimBus(self.clock)
        self.fire = FireModel(static, cfg, seed=static.seed)
        self.next_fire_t = cfg.fire_step_s
        self.seekers: list[SeekerAgent] = []
        self.exts: list[ExtinguisherAgent] = []
        self.player_ext: PlayerExtinguisher | None = None
        self.leader: StubLeader | None = None
        self.drops: deque = deque(maxlen=64)
        self.drop_seq = 0
        self.log: deque = deque(maxlen=40)
        self.stats = {"dropped_l": 0.0, "on_fire_l": 0.0, "drops": 0}
        base = Position(x=cfg.base_xy[0], y=cfg.base_xy[1], z=0.0)
        self.base = base
        self.station = RefillStation(station_id="S1", position=Position(x=base.x, y=base.y - 8.0, z=0.0),
                                     water_available_l=1e6, fill_rate_lpm=600.0, total_bays=10, free_bays=10,
                                     fuel_available_kg=5000.0, agents_available=[ExtinguishingAgent.WATER])
        if kind == "baseline":
            return

        R = cfg.arena_radius_m
        n_map = int(math.ceil(2 * R / cfg.map_cell_m))
        self.grid = GridSpec(origin=Position(x=-R, y=-R), resolution_m=cfg.map_cell_m, rows=n_map, cols=n_map)
        classifier = load_classifier(model_path)
        self.confirm_threshold = float(getattr(classifier, "confirm_threshold", None) or classifier.threshold)
        sspec, espec = cfg.seeker_spec, cfg.extinguisher_spec
        n_s = 1 if kind == "human" else max(1, n_seekers)
        for i in range(n_s):
            sid = f"seeker-{i + 1}"
            # both sides start hovering at 40 m over the base (the player's spawn), so the seek clock is fair
            start = Position(x=base.x - 14.0 + 7.0 * i, y=base.y - 2.0, z=40.0)
            ap = SimAutopilot(sspec, start, min_agl_m=cfg.seeker_min_agl_m, transit_z=cfg.transit_agl_m,
                              vertical_speed_mps=6.0, accel_mps2=4.0)
            perc = SeekerPerception(sid, FusedDetector(classifier), grid=self.grid, altimeter_max_m=200.0)
            self.seekers.append(SeekerAgent(sid, self, ap, perc, executor, player=(kind == "human")))

        limits = WaterDropLimits(min_confidence=0.5, max_distance_m=3.0, min_altitude_agl_m=cfg.ext_min_agl_m,
                                 max_altitude_agl_m=cfg.drop_max_effective_agl_m, max_speed_mps=3.0)
        dwell = [DwellZone(self.station.position, 6.0, cfg.refill_dwell_s)]
        pads = _pads(base, cfg.n_extinguishers)
        autopilots = {}
        for i in range(cfg.n_extinguishers):
            eid = f"ext-{i + 1}"
            ctx = planning_context(cfg, base)
            ap = SimAutopilot(espec, pads[i], min_agl_m=cfg.drop_agl_m, transit_z=cfg.transit_agl_m,
                              vertical_speed_mps=cfg.vertical_speed_mps, dwell_zones=dwell)
            payload = SimPayload(espec.water_capacity_l, espec.water_capacity_l)
            ap.water_kg = (lambda p=payload: p.water_l)
            payload.on_release = self._release_cb(ap)
            if kind == "human" and i == 0:
                ap.min_agl_m = cfg.ext_min_agl_m
                self.player_ext = PlayerExtinguisher(
                    eid, ap, payload, release_lps=espec.water_capacity_l / espec.drop_release_s,
                    refill_lps=espec.water_capacity_l / cfg.refill_dwell_s, base=base)
                continue
            agent = ExtinguisherAgent(eid, self, ap, payload, self.station, ctx, limits)
            self.exts.append(agent)
            autopilots[eid] = ap

        common = dict(refill=DummyRefillAssigner(), stations=[self.station],
                      confirm_threshold=self.confirm_threshold, l_per_m2=cfg.suppression_l_per_m2,
                      margin=cfg.hotspot_volume_margin, min_volume_l=cfg.min_task_volume_l,
                      geofence_radius_m=cfg.arena_radius_m - 4.0, log=self.log.append)
        if kind == "human":
            self.leader = HumanCommander("commander", self.bus, self.clock, autopilots=autopilots,
                                         transit_z=cfg.transit_agl_m, **common)
        else:
            self.leader = StubLeader("leader", self.bus, self.clock, assigner=DummyTaskAssigner(), **common)
        self.bus.subscribe(SUB_GROUND_ALERTS, self._on_alert)

    # ------------------------------------------------------------------------------------------------------------
    def _release_cb(self, ap: SimAutopilot):
        def cb(litres: float) -> None:
            self.apply_water(ap.x, ap.y, litres, ap.z)
        return cb

    def apply_water(self, x: float, y: float, litres: float, agl: float) -> None:
        on = self.fire.apply_water(x, y, litres, agl)
        self.stats["dropped_l"] += litres
        self.stats["on_fire_l"] += on
        if litres >= 1.0 or not self.drops or self.clock.now() - self.drops[-1][1] > 0.25:
            self.drop_seq += 1
            self.stats["drops"] += 1
            r = math.sqrt(litres / self.cfg.suppression_l_per_m2 / math.pi) + self.cfg.drop_spread_per_m * agl
            self.drops.append((self.drop_seq, self.clock.now(), x, y, r, litres))

    def _on_alert(self, msg) -> None:
        a = msg.body.alert
        self.log.append(f"{a.subject_id}: {a.message}")

    def all_drone_agents(self):
        return list(self.seekers) + list(self.exts)

    # ---- phases ------------------------------------------------------------------------------------------------
    def begin_seek(self) -> None:
        cfg = self.cfg
        if self.kind == "swarm":
            zones = DummyZoneAllocator().allocate(cfg.arena_radius_m, [s.id for s in self.seekers])
            for s in self.seekers:
                s.assign_zone(zones[s.id], cfg.lawnmower_agl_m, cfg.lawnmower_overlap)
                s.start_search()
        elif self.kind == "human":
            for s in self.seekers:
                s.searching = True

    def end_seek(self) -> None:
        for s in self.seekers:
            s.stop_search(rtb=True)
        for s in self.seekers:
            s.drain()

    def begin_extinguish(self, plan: dict | None = None) -> None:
        if self.leader is None:
            return
        if isinstance(self.leader, HumanCommander):
            self.leader.set_plan(plan or {})
            self.leader.start()
        else:
            self.leader.active = True

    # ---- stepping ----------------------------------------------------------------------------------------------
    def step(self, dt: float) -> None:
        self.clock.t += dt
        now = self.clock.t
        self.bus.pump(now)
        for s in self.seekers:
            s.tick(now, dt)
        for e in self.exts:
            e.tick(now, dt)
        if self.player_ext is not None:
            self.player_ext.tick(now, dt, self.apply_water)
        if self.leader is not None:
            self.leader.tick(now)
        while now >= self.next_fire_t - 1e-9:
            self.fire.step(self.cfg.fire_step_s * self.cfg.fire_time_scale)
            self.next_fire_t += self.cfg.fire_step_s

    # ---- views -------------------------------------------------------------------------------------------------
    def coverage(self) -> np.ndarray:
        """Cells any seeker camera has actually covered (perception log-odds map, last_seen)."""
        if not self.seekers:
            return np.zeros((1, 1), bool)
        cov = None
        for s in self.seekers:
            seen = ~np.isnan(s.perception.map.last_seen)
            cov = seen if cov is None else (cov | seen)
        return cov

    def coverage_frac(self) -> float:
        cov = self.coverage()
        if cov.size <= 1:
            return 0.0
        g = self.grid
        yy, xx = (np.mgrid[:g.rows, :g.cols] + 0.5) * g.resolution_m - self.cfg.arena_radius_m
        inside = np.hypot(xx, yy) <= self.cfg.arena_radius_m
        return float((cov & inside).sum() / inside.sum())

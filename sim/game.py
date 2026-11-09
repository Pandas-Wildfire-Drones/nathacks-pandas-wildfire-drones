"""Game session: phases, the human / swarm / baseline worlds in lockstep, scoring and client snapshots.

play   human world (player) + swarm world (shadow, same seed) + baseline; the player sees the human world
watch  swarm world + baseline; spectate the autonomous system
demo   DEMO mode: large map, seekers + field-refill attackers at the same time for cfg.demo_s, + baseline
Score: fire_left_% = burning intensity-area(world) / burning intensity-area(baseline) x 100 at the end (lower = better).
"""
from __future__ import annotations

import base64
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from contracts import HotspotStatus
from perception.georef import world_to_pixel
from sim.attacker import STATUS_TEXT
from sim.config import DEMO_MAPS, GameConfig, demo_config
from sim.static import build_static
from sim.world import World

__all__ = ["PHASES", "Session"]

PHASES = ("BRIEFING", "SEEK", "PLAN", "EXTINGUISH", "SCORE")
DEMO_PHASES = ("DEMO", "SCORE")
_STATUS = {HotspotStatus.DETECTED: 0, HotspotStatus.CONFIRMED: 1, HotspotStatus.ASSIGNED: 2,
           HotspotStatus.SUPPRESSED: 3, HotspotStatus.FALSE_POSITIVE: 4, HotspotStatus.VERIFIED_OUT: 3}

_EXECUTOR: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="perception")
    return _EXECUTOR


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


class Session:
    def __init__(self, map_id: str = "p1", seed: int = 1, mode: str = "play", swarm_seekers: int = 1,
                 cfg: GameConfig | None = None):
        self.map_id, self.seed, self.mode = map_id, int(seed), mode
        self.follow_id: str | None = None
        ex = _executor()
        if mode == "demo":
            info = DEMO_MAPS.get(map_id, DEMO_MAPS["demo-lakes-1"])
            cfg = cfg if cfg is not None and cfg.demo else demo_config(wet=info["wet"])
            cfg = replace(cfg, lakes_wet=info["wet"])
            self.cfg = cfg
            # wet and dry variants of a map share the layout seed: identical fires, only the water differs
            self.static = build_static(cfg, info["tiles"], info["layout"] * 1000 + self.seed)
            self.title = info["title"]
            self.swarm = World(cfg, self.static, "demo", ex)
            self.human = None
            self.phases = DEMO_PHASES
        else:
            cfg = cfg or GameConfig()
            cfg = replace(cfg, swarm_seekers=int(max(1, min(4, swarm_seekers))))
            self.cfg = cfg
            self.static = build_static(cfg, map_id, self.seed)
            self.title = self.static.map.title
            self.swarm = World(cfg, self.static, "swarm", ex, n_seekers=cfg.swarm_seekers)
            self.human = World(cfg, self.static, "human", ex) if mode == "play" else None
            self.phases = PHASES
        self.baseline = World(cfg, self.static, "baseline", None)
        self.view = self.human if self.human is not None else self.swarm
        self.worlds = [w for w in (self.human, self.swarm, self.baseline) if w is not None]
        self.phase_i = 0
        self.phase_t = 0.0
        self.t = 0.0
        self.paused = False
        self.durations = {"BRIEFING": cfg.briefing_s, "SEEK": cfg.seek_s, "PLAN": cfg.plan_s,
                          "EXTINGUISH": cfg.extinguish_s, "DEMO": cfg.demo_s, "SCORE": float("inf")}
        self.plan: dict[str, dict] = {}
        self.swarm_plan: list = []
        self.result: dict | None = None
        self.metric_at_ext_start: dict[str, float] = {}
        self.snap_i = 0
        self._last_drop_seq = 0
        if self.human is not None:
            s = self.human.seekers[0]
            s.ap.set_pose(s.ap.start.x, s.ap.start.y, 40.0, 0.0, 0.0)
        if mode == "demo":                # seekers and extinguishers start together
            self.swarm.begin_seek()
            self.swarm.leader.active = True
            self.metric_at_ext_start = {w.kind: w.fire.fire_metric() for w in self.worlds}

    # ---- phases ------------------------------------------------------------------------------------------------
    @property
    def phase(self) -> str:
        return self.phases[self.phase_i]

    def phase_left(self) -> float:
        d = self.durations[self.phase]
        return max(0.0, d - self.phase_t) if math.isfinite(d) else 0.0   # JSON has no Infinity

    def advance(self) -> None:
        if self.phase == "SCORE":
            return
        old = self.phase
        if old == "DEMO":
            for s in self.swarm.seekers:
                s.drain()
        self.phase_i += 1
        self.phase_t = 0.0
        new = self.phase
        if old == "BRIEFING":
            for w in self.worlds:
                w.begin_seek()
        elif old == "SEEK":
            for w in self.worlds:
                w.end_seek()
            if self.swarm.leader is not None:
                self.swarm_plan = self.swarm.leader.preview_plan()
        elif old == "PLAN":
            self.metric_at_ext_start = {w.kind: w.fire.fire_metric() for w in self.worlds}
            if self.human is not None:
                self.human.begin_extinguish(self.plan)
                pe = self.human.player_ext
                pe.ap.set_pose(pe.ap.start.x, pe.ap.start.y, 20.0, 0.0, 0.0)
            self.swarm.begin_extinguish()
        if new == "SCORE":
            self.result = self.score()

    def skip(self) -> None:
        """Briefing skip / 'launch now' on the planning screen (both worlds share the clock, so this stays fair)."""
        if self.phase in ("BRIEFING", "PLAN"):
            self.advance()

    def step(self) -> None:
        if self.paused or self.phase == "SCORE":
            return
        dt = 1.0 / self.cfg.tick_hz
        for w in self.worlds:
            w.step(dt)
        self.t += dt
        self.phase_t += dt
        if self.phase_t >= self.durations[self.phase] - 1e-9:
            self.advance()

    # ---- player ------------------------------------------------------------------------------------------------
    def player_input(self, msg: dict) -> None:
        if self.human is None:
            return
        cfg = self.cfg
        x, y, z = float(msg.get("x", 0)), float(msg.get("y", 0)), float(msg.get("z", 40))
        r = math.hypot(x, y)
        if r > cfg.arena_radius_m:
            x, y = x * cfg.arena_radius_m / r, y * cfg.arena_radius_m / r
        yaw, v = float(msg.get("yaw", 0)), float(msg.get("v", 0))
        if self.phase in ("BRIEFING", "SEEK"):
            z = min(max(z, cfg.seeker_min_agl_m if self.phase == "SEEK" else 0.0), cfg.seeker_max_agl_m)
            self.human.seekers[0].ap.set_pose(x, y, z, yaw, v)
        elif self.phase == "EXTINGUISH":
            z = min(max(z, 0.0), cfg.ext_max_agl_m)
            pe = self.human.player_ext
            pe.ap.set_pose(x, y, z, yaw, v)
            pe.dropping = bool(msg.get("drop", False))

    def set_plan(self, plan: dict) -> None:
        self.plan = {str(k): v for k, v in plan.items()}

    # ---- scoring -----------------------------------------------------------------------------------------------
    def _detected(self, w: World) -> tuple[int, int]:
        if w.leader is None:
            return 0, 0
        hs = [t.hotspot.position for t in w.leader.hotspots.values()]
        fires = sum(1 for f in self.static.fires if any(math.hypot(p.x - f.x, p.y - f.y) < 10 for p in hs))
        decoys = sum(1 for d in self.static.decoys if any(math.hypot(p.x - d.x, p.y - d.y) < 6 for p in hs))
        return fires, decoys

    def score(self) -> dict:
        base = max(self.baseline.fire.fire_metric(), 1e-6)
        out = {"baseline_m2": round(base, 1), "fires_total": len(self.static.fires),
               "decoys_total": len(self.static.decoys)}
        if self.mode == "demo":
            out["demo"] = self.demo_stats(final=True)
            return out
        for key, w in (("human", self.human), ("swarm", self.swarm)):
            if w is None:
                continue
            fires, decoys = self._detected(w)
            out[key] = {
                "fire_left_pct": round(100.0 * w.fire.fire_metric() / base, 1),
                "fire_m2": round(w.fire.fire_metric(), 1),
                "coverage_pct": round(100.0 * w.coverage_frac(), 1),
                "fires_detected": fires, "decoys_reported": decoys,
                "water_dropped_l": round(w.stats["dropped_l"]),
                "water_on_fire_l": round(w.stats["on_fire_l"]),
                "tasks": w.leader.n_tasks if w.leader else 0,
            }
        return out

    def demo_stats(self, final: bool = False) -> dict:
        w = self.swarm
        ld = w.leader
        base = max(self.baseline.fire.fire_metric(), 1e-6)
        left = 100.0 * w.fire.fire_metric() / base
        fills_lake = sum(a.stats["fills_lake"] for a in w.attackers)
        fills_base = sum(a.stats["fills_base"] for a in w.attackers)
        out = {
            "fire_fought_pct": round(max(0.0, 100.0 - left), 1), "fire_left_pct": round(left, 1),
            "fire_m2": round(w.fire.fire_metric(), 1), "baseline_m2": round(base, 1),
            "drops": int(ld.stats["drops"]), "fills_lake": fills_lake, "fills_base": fills_base,
            "suppressed": int(ld.stats["suppressed"]), "verify_hot": int(ld.stats["verify_hot"]),
            "reopened": int(ld.stats["reopened"]), "confirmed": int(ld.stats["confirmed"]),
            "queue": len(ld.queue_view(999)), "tasks": ld.n_tasks,
            "water_dropped_l": round(w.stats["dropped_l"]), "water_on_fire_l": round(w.stats["on_fire_l"]),
            "seekers": len(w.seekers), "attackers": len(w.attackers), "lakes": len(w.water_sources),
        }
        if final:
            fires, decoys = self._detected(w)
            out.update({"coverage_pct": round(100.0 * w.coverage_frac(), 1), "fires_detected": fires,
                        "decoys_reported": decoys, "fires_total": len(self.static.fires),
                        "wet": bool(self.cfg.lakes_wet)})
        return out

    def activity(self, w: World) -> dict[str, str]:
        out = {}
        for s in w.seekers:
            z = w.zones.get(s.id)
            n = sum(1 for t in (w.leader.hotspots.values() if w.leader else []) if s.id in t.sources)
            what = (f"Lawnmower over zone {z.zone_id} (pass {s.passes + 1}) at {s.ap.z:.0f} m"
                    if s.searching and z is not None else "Returning to base")
            out[s.id] = f"{what} · {n} hotspots reported"
        for a in w.attackers:
            out[a.id] = a.activity()
        return out

    # ---- client messages ---------------------------------------------------------------------------------------
    def init_message(self) -> dict:
        st, cfg = self.static, self.cfg
        prev = st.ground_preview(cfg.map_cell_m)
        w = self.view
        return {
            "type": "init", "mode": self.mode, "map": self.map_id, "map_title": self.title, "seed": self.seed,
            "real_data": st.map.real, "swarm_seekers": cfg.swarm_seekers,
            "arena_radius": cfg.arena_radius_m, "base": [w.base.x, w.base.y],
            "station": [w.station.position.x, w.station.position.y],
            "pads": [[e.ap.start.x, e.ap.start.y] for e in w.exts] +
                    ([[w.player_ext.ap.start.x, w.player_ext.ap.start.y]] if w.player_ext else []),
            "trees": [[round(t.x, 1), round(t.y, 1), round(t.h, 1), round(t.r, 2), t.kind] for t in st.trees],
            "decoys": [[d.kind, round(d.x, 1), round(d.y, 1), round(d.yaw_deg), round(d.area_m2, 1)]
                       for d in st.decoys],
            "ground": {"n": int(prev.shape[0]), "cell": cfg.map_cell_m, "data": _b64(prev)},
            "raster_cell": cfg.map_cell_m,
            "phases": {k: (v if math.isfinite(v) else -1) for k, v in self.durations.items()},
            "limits": {"seeker": [cfg.seeker_min_agl_m, cfg.seeker_max_agl_m],
                       "ext": [cfg.ext_min_agl_m, cfg.ext_max_agl_m], "drop_max_agl": cfg.drop_max_effective_agl_m},
            "speeds": {"seeker": [cfg.seeker_spec.cruise_speed_mps, cfg.seeker_spec.max_speed_mps,
                                  cfg.seeker_spec.max_accel_mps2],
                       "ext": [cfg.extinguisher_spec.cruise_speed_mps, cfg.extinguisher_spec.max_speed_mps,
                               cfg.extinguisher_spec.max_accel_mps2]},
            "camera": {"hfov": cfg.seeker_spec.thermal.camera.hfov_deg, "w": 640, "h": 512},
            "water_capacity": cfg.extinguisher_spec.water_capacity_l,
            "bots": [e.id for e in w.exts],
            "player_ext": w.player_ext.drone_id if w.player_ext else None,
            "lakes": [[lk.lake_id, lk.wet, round(lk.r, 1), [[round(x, 1), round(y, 1)] for x, y in lk.outline()]]
                      for lk in st.lakes],
            "n_seekers": len(w.seekers), "n_exts": len(w.exts) + len(w.attackers) + (1 if w.player_ext else 0),
            "zones": {sid: [[round(p.x, 1), round(p.y, 1)] for p in z.boundary] for sid, z in w.zones.items()},
            "status_text": STATUS_TEXT,
        }

    def _drones(self, w: World) -> list:
        out = []
        for s in w.seekers:
            p = s.ap
            out.append([s.id, "s", round(p.x, 2), round(p.y, 2), round(p.z, 2), round(p.yaw, 1), 0,
                        "SCAN" if s.searching else "RTB", s.player])
        for e in w.exts:
            p = e.ap
            out.append([e.id, "e", round(p.x, 2), round(p.y, 2), round(p.z, 2), round(p.yaw, 1),
                        round(e.payload.water_l), e.last_status, False])
        for a in w.attackers:
            p = a.ap
            out.append([a.id, "e", round(p.x, 2), round(p.y, 2), round(p.z, 2), round(p.yaw, 1),
                        round(a.payload.water_l, 1), a.mode, False])
        if w.player_ext is not None:
            p = w.player_ext.ap
            out.append([w.player_ext.drone_id, "e", round(p.x, 2), round(p.y, 2), round(p.z, 2), round(p.yaw, 1),
                        round(w.player_ext.payload.water_l), "REFILL" if w.player_ext.refilling else "PLAYER", True])
        return out

    def _hotspots(self, w: World) -> list:
        if w.leader is None:
            return []
        return [[t.hotspot_id, round(t.hotspot.position.x, 1), round(t.hotspot.position.y, 1),
                 round(t.hotspot.est_area_m2, 1), round(t.hotspot.confidence, 2), _STATUS.get(t.hotspot.status, 0),
                 round(max(t.remaining_l, 0)), round(getattr(t, "priority", 0.0), 3)]
                for t in w.leader.hotspots.values()]

    def _thermal(self, w: World) -> dict | None:
        seeker = next((s for s in w.seekers if s.id == self.follow_id), w.seekers[0] if w.seekers else None)
        if seeker is None or seeker.last_preview is None:
            return None
        pv = seeker.last_preview
        temp = pv["temp"][::4, ::4]
        lo = float(np.percentile(temp, 2))
        hi = max(lo + 8.0, min(float(temp.max()), 300.0))
        img = np.clip((temp - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        boxes = []
        for h in seeker.perception.tracker.hotspots():
            if abs(h.last_seen - pv["t"]) > 1e-6:
                continue
            uv = world_to_pixel(h.position, pv["pose"], seeker.camera)
            if uv is not None:
                boxes.append([round(uv[0] / 4, 1), round(uv[1] / 4, 1), round(h.confidence, 2)])
        return {"w": int(img.shape[1]), "h": int(img.shape[0]), "data": _b64(img), "lo": round(lo, 1),
                "hi": round(hi, 1), "t": round(pv["t"], 2), "boxes": boxes, "id": seeker.id,
                "tmax": round(float(pv["temp"].max()), 1)}

    def snapshot(self) -> dict:
        self.snap_i += 1
        w = self.view
        cfg = self.cfg
        snap = {"type": "state", "t": round(self.t, 2), "phase": self.phase, "phase_left": round(self.phase_left(), 2),
                "paused": self.paused, "drones": self._drones(w)}
        if self.snap_i % 2 == 0 or self.phase == "PLAN":
            heat = w.fire.raster(cfg.map_cell_m, "heat")
            snap["fire"] = _b64((np.clip(heat, 0, 1) * 255).astype(np.uint8))
            snap["fire_n"] = int(heat.shape[0])
        if self.snap_i % 8 == 0:
            burnt = w.fire.raster(cfg.map_cell_m, "burnt")
            snap["burnt"] = _b64((np.clip(burnt, 0, 1) * 255).astype(np.uint8))
        if self.snap_i % 4 == 0 or self.phase == "PLAN":
            cov = w.coverage()
            if cov.size > 1:
                snap["cov"] = _b64(cov.astype(np.uint8))
                snap["cov_n"] = int(cov.shape[0])
                snap["cov_pct"] = round(100.0 * w.coverage_frac(), 1)
            snap["hotspots"] = self._hotspots(w)
        if self.snap_i % 5 == 0:
            th = self._thermal(w)
            if th is not None:
                snap["thermal"] = th
        drops = [list(d) for d in w.drops if d[0] > self._last_drop_seq]
        if drops:
            self._last_drop_seq = drops[-1][0]
            snap["drops"] = [[d[0], round(d[2], 1), round(d[3], 1), round(d[4], 2), round(d[5], 1)] for d in drops]
        if self.snap_i % 6 == 0:
            snap["log"] = list(w.log)[-6:]
            snap["fire_m2"] = round(w.fire.fire_metric(), 1)
            snap["baseline_m2"] = round(self.baseline.fire.fire_metric(), 1)
            snap["water_dropped"] = round(w.stats["dropped_l"])
        if self.phase in ("PLAN", "EXTINGUISH") and self.mode == "watch":
            snap["plan"] = [[d, h] for d, h, _ in self.swarm_plan]
        if self.mode == "demo":
            if self.snap_i % 3 == 0:
                snap["acts"] = self.activity(w)
                snap["links"] = [[d, a.fire_id] for d, a in w.leader.assignments.items()]
            if self.snap_i % 6 == 0:
                snap["demo"] = self.demo_stats()
                snap["queue"] = [[f, round(p, 3), round(l)] for f, p, l in w.leader.queue_view(8)]
        if self.phase == "SCORE" and self.result is not None:
            snap["score"] = self.result
        return snap

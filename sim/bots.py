"""Human-mode bot behaviour (game logic, not swarm-brain): the player's plan anchors each bot to a spot or a fire.

At EXTINGUISH start each bot flies straight to its anchor (crude direct navigation at transit altitude). Once there,
it repeatedly takes the nearest known hotspot within `radius_m` of its anchor that still needs water. Tasks go to the
bot through the same bus messages as the swarm, so execution is the real autonomy adapter in both modes.
Hotspots the player's seeker never detected don't exist for the bots.
"""
from __future__ import annotations

import math

from contracts import Position
from sim.brain_stub import StubLeader, task_volume

__all__ = ["AnchorAssigner", "HumanCommander"]


class AnchorAssigner:
    def __init__(self, radius_m: float = 45.0):
        self.radius_m = radius_m
        self.anchors: dict[str, tuple[float, float]] = {}
        self.first_target: dict[str, str] = {}
        self.arrived: set[str] = set()

    def assign(self, hotspots, drones, now):
        out = []
        remaining = {h.hotspot_id: h.remaining_l for h in hotspots}
        by_id = {h.hotspot_id: h for h in hotspots}
        for d in sorted(drones, key=lambda d: d.drone_id):
            if d.busy or d.drone_id not in self.anchors:
                continue
            first = self.first_target.pop(d.drone_id, None)
            if first is not None and first in by_id and remaining[first] > 0:
                vol = task_volume(remaining[first], d)
                remaining[first] = 0.0
                out.append((d.drone_id, first, vol))
                continue
            if d.drone_id not in self.arrived:
                continue
            ax, ay = self.anchors[d.drone_id]
            best, best_d = None, float("inf")
            for h in hotspots:
                if remaining[h.hotspot_id] <= 0:
                    continue
                p = h.hotspot.position
                if math.hypot(p.x - ax, p.y - ay) > self.radius_m:
                    continue
                dist = math.hypot(p.x - d.position.x, p.y - d.position.y)
                if dist < best_d:
                    best, best_d = h, dist
            if best is not None:
                vol = task_volume(remaining[best.hotspot_id], d)
                remaining[best.hotspot_id] = 0.0
                out.append((d.drone_id, best.hotspot_id, vol))
        return out


class HumanCommander(StubLeader):
    def __init__(self, *args, autopilots: dict, transit_z: float, **kw):
        self.anchor_assigner = AnchorAssigner()
        super().__init__(*args, assigner=self.anchor_assigner, **kw)
        self.autopilots = autopilots
        self.transit_z = transit_z
        self.plan: dict[str, dict] = {}

    def set_plan(self, plan: dict[str, dict]) -> None:
        """plan: drone_id -> {"x": .., "y": ..} (area) or {"hotspot": id} (a detected fire)."""
        self.plan = dict(plan)

    def start(self) -> None:
        a = self.anchor_assigner
        for drone_id, ap in self.autopilots.items():
            item = self.plan.get(drone_id)
            if item is None:
                continue
            hid = item.get("hotspot")
            if hid is not None and hid in self.hotspots:
                p = self.hotspots[hid].hotspot.position
                a.anchors[drone_id] = (p.x, p.y)
                a.first_target[drone_id] = hid
                a.arrived.add(drone_id)        # goes straight to the fire through autonomy
            elif "x" in item and "y" in item:
                a.anchors[drone_id] = (float(item["x"]), float(item["y"]))
                ap.goto(Position(x=float(item["x"]), y=float(item["y"]), z=self.transit_z))
        self.active = True

    def tick(self, now: float) -> None:
        a = self.anchor_assigner
        for drone_id, (x, y) in a.anchors.items():
            if drone_id in a.arrived:
                continue
            ap = self.autopilots[drone_id]
            if math.hypot(ap.x - x, ap.y - y) < 4.0:
                a.arrived.add(drone_id)
        super().tick(now)

"""SimAutopilot / SimPayload: Level-0 implementations of contracts.interfaces.Autopilot and PayloadSystem.

Kinematics are deliberately simple [ASSUMPTION, MechE to replace]: horizontal speed is acceleration-limited,
direction changes instantly, vertical speed is limited separately.

Autopilot behaviours worth knowing (they stand in for what PX4 would do):
  * Minimum-AGL floor: a goto target below ground + min_agl_m is raised to it (terrain-following floor). Autonomy
    flies to `hotspot.position` literally (z = ground), so this is what makes it hover at drop height.
  * Fly-through waypoints: targets at/above transit altitude are mission waypoints with an acceptance radius
    (no braking); lower targets (approach, drop, refill pad) are stop points (brake to a hover).
  * Dwell zones: at a refill pad `at_target()` only becomes True after hovering dwell_s there (hook-up + pumping).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from contracts import ExtinguishingAgent, PlatformSpec, Position, Powertrain

__all__ = ["DwellZone", "SimAutopilot", "SimPayload"]


@dataclass
class DwellZone:
    position: Position
    radius_m: float
    dwell_s: float


@dataclass
class SimAutopilot:
    spec: PlatformSpec
    start: Position
    min_agl_m: float
    transit_z: float
    vertical_speed_mps: float = 5.0
    ground_z: float = 0.0
    dwell_zones: list[DwellZone] = field(default_factory=list)
    accel_mps2: float | None = None
    # state
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    v: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    target: Position | None = None
    target_speed: float | None = None
    fly_through: bool = False
    dwell_t: float = 0.0
    fuel_kg: float | None = None
    battery_wh: float | None = None
    water_kg: Callable[[], float] = lambda: 0.0

    def __post_init__(self):
        self.x, self.y, self.z = self.start.x, self.start.y, self.start.z
        if self.spec.powertrain is Powertrain.HYBRID:
            self.fuel_kg = float(self.spec.fuel_capacity_kg)
        else:
            self.battery_wh = float(self.spec.battery_wh)

    # ---- Autopilot protocol ------------------------------------------------------------------------------------
    def position(self) -> Position:
        return Position(x=self.x, y=self.y, z=self.z)

    def heading_deg(self) -> float:
        return self.yaw % 360.0

    def speed_mps(self) -> float:
        return abs(self.v)

    def battery_pct(self) -> float:
        if self.battery_wh is None:
            return 100.0  # hybrid buffer kept full by the generator
        return max(0.0, 100.0 * self.battery_wh / self.spec.battery_wh)

    def endurance_remaining_s(self) -> float:
        if self.spec.powertrain is Powertrain.HYBRID:
            return self.spec.hybrid_endurance_s(water_l=self.water_kg(), fuel_kg=self.fuel_kg, reserve=True)
        usable = self.battery_wh - self.spec.battery_wh * (1 - self.spec.battery_usable_frac)
        return max(0.0, usable / self.spec.cruise_power_w * 3600.0)

    def goto(self, target: Position, speed_mps: float | None = None) -> None:
        z = max(target.z, self.ground_z + self.min_agl_m)
        eff = Position(x=target.x, y=target.y, z=z)
        if self.target is None or eff != self.target:
            self.dwell_t = 0.0
        self.target = eff
        self.target_speed = speed_mps
        self.fly_through = z >= self.transit_z - 0.5

    def hold(self) -> None:
        self.target = Position(x=self.x, y=self.y, z=self.z)
        self.fly_through = False

    def return_to_launch(self) -> None:
        self.goto(Position(x=self.start.x, y=self.start.y, z=self.ground_z))

    def at_target(self, tolerance_m: float = 2.0) -> bool:
        if self.target is None:
            return True
        d = math.dist((self.x, self.y, self.z), (self.target.x, self.target.y, self.target.z))
        if d > tolerance_m:
            return False
        zone = self._dwell_zone()
        return zone is None or self.dwell_t >= zone.dwell_s

    # ---- simulation --------------------------------------------------------------------------------------------
    def _dwell_zone(self) -> DwellZone | None:
        if self.target is None:
            return None
        for zn in self.dwell_zones:
            if math.hypot(self.target.x - zn.position.x, self.target.y - zn.position.y) <= zn.radius_m:
                return zn
        return None

    def update(self, dt: float) -> None:
        spec = self.spec
        a = self.accel_mps2 or spec.max_accel_mps2
        if self.target is not None:
            dx, dy = self.target.x - self.x, self.target.y - self.y
            dist = math.hypot(dx, dy)
            vmax = min(self.target_speed or spec.cruise_speed_mps, spec.max_speed_mps)
            if self.fly_through:
                v_des = vmax
            else:
                v_des = min(vmax, math.sqrt(2.0 * a * max(dist - 0.2, 0.0)))
            if self.v < v_des:
                self.v = min(v_des, self.v + a * dt)
            else:
                self.v = max(v_des, self.v - a * 1.5 * dt)
            step = self.v * dt
            if dist <= max(step, 1e-6):
                self.x, self.y = self.target.x, self.target.y
                if not self.fly_through:
                    self.v = 0.0
            elif dist > 1e-6:
                self.x += dx / dist * step
                self.y += dy / dist * step
                if step > 0.02:
                    want = math.degrees(math.atan2(dx, dy)) % 360.0
                    diff = (want - self.yaw + 540.0) % 360.0 - 180.0
                    self.yaw = (self.yaw + max(-120 * dt, min(120 * dt, diff))) % 360.0
            dz = self.target.z - self.z
            self.vz = max(-self.vertical_speed_mps, min(self.vertical_speed_mps, dz / 0.7))
            self.z += self.vz * dt
            if abs(self.target.z - self.z) < 0.05:
                self.z = self.target.z
            zone = self._dwell_zone()
            if zone is not None and math.dist((self.x, self.y, self.z),
                                               (self.target.x, self.target.y, self.target.z)) <= 2.0:
                self.dwell_t += dt
        # energy
        if self.fuel_kg is not None:
            mass = spec.empty_mass_kg + self.water_kg() + self.fuel_kg
            self.fuel_kg = max(0.0, self.fuel_kg - spec.fuel_burn_kg_h(mass) / 3600.0 * dt)
        elif self.battery_wh is not None:
            p = spec.cruise_power_w if self.v > 1.0 else spec.hover_power_w
            self.battery_wh = max(0.0, self.battery_wh - p * dt / 3600.0)

    def set_pose(self, x: float, y: float, z: float, yaw: float, v: float) -> None:
        """Player-driven drones: pose comes from the client."""
        self.x, self.y, self.z, self.yaw, self.v = x, y, z, yaw % 360.0, v
        self.target = None


@dataclass
class SimPayload:
    capacity_l: float
    water_l: float
    agent_kind: ExtinguishingAgent = ExtinguishingAgent.WATER
    on_release: Callable[[float], None] | None = None  # litres -> applies them at the drone's actual position

    def water_remaining_l(self) -> float:
        return self.water_l

    def agent(self) -> ExtinguishingAgent:
        return self.agent_kind

    def release(self, volume_l: float, at: Position) -> float:
        amount = max(0.0, min(volume_l, self.water_l))
        self.water_l -= amount
        if amount > 0 and self.on_release is not None:
            self.on_release(amount)
        return amount

    def fill(self, volume_l: float, agent: ExtinguishingAgent = ExtinguishingAgent.WATER) -> float:
        amount = max(0.0, min(volume_l, self.capacity_l - self.water_l))
        self.water_l += amount
        self.agent_kind = agent
        return amount

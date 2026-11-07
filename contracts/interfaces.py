"""Interfaces (Protocols) between layers. Drone logic depends ONLY on these, never on ROS/PX4/FastAPI.

Fidelity levels are different implementations of the same interfaces:
  Level 0 (abstract sim):  SimBus / SimClock / SimAutopilot / SimPayload / FakeDetector / SimThermalSensor
  Level 1 (PX4 SITL):      same Bus/Clock; Px4Autopilot (MAVSDK); Gazebo or sim-supplied sensors
  Level 2 (real drone):    Ros2Bus or MqttBus; Px4Autopilot; real camera drivers + OnnxDetector

Execution model (recommended): single-threaded, tick-based and deterministic.
The simulator calls `tick(now, dt)` on every Tickable each step; the SimBus delivers queued
messages (applying latency/loss) at the start of each tick.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from .enums import DetectorKind, ExtinguishingAgent
from .geometry import Pose, Position
from .messages import SwarmMessage
from .sensing import CameraModel, Detection, Obstacle, ThermalFrame

__all__ = [
    "MessageHandler", "Subscription", "Clock", "Bus", "Tickable", "Autopilot", "PayloadSystem",
    "PowerSystem", "Detector", "Georeferencer", "ThermalSensor", "ObstacleSensor",
]

MessageHandler = Callable[[SwarmMessage], None]


@runtime_checkable
class Subscription(Protocol):
    def cancel(self) -> None: ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        """Seconds. Sim time in the simulator, wall-clock epoch seconds on hardware."""
        ...


@runtime_checkable
class Bus(Protocol):
    """Pub/sub. Implementations MUST serialize with encode()/decode() and use topic_matches()."""

    def publish(self, topic: str, msg: SwarmMessage) -> None: ...

    def subscribe(self, pattern: str, handler: MessageHandler) -> Subscription: ...


@runtime_checkable
class Tickable(Protocol):
    def tick(self, now: float, dt: float) -> None: ...


@runtime_checkable
class Autopilot(Protocol):
    """Flight control. goto() is a REQUEST: arrival is observed via position()/at_target(), not assumed."""

    def position(self) -> Position: ...
    def heading_deg(self) -> float: ...
    def speed_mps(self) -> float: ...
    def battery_pct(self) -> float: ...
    def endurance_remaining_s(self) -> float: ...
    def goto(self, target: Position, speed_mps: float | None = None) -> None: ...
    def hold(self) -> None: ...
    def return_to_launch(self) -> None: ...
    def at_target(self, tolerance_m: float = 2.0) -> bool: ...


@runtime_checkable
class PayloadSystem(Protocol):
    """Water/agent tank on an EXTINGUISHER. `release` supports partial drops (several hotspots per sortie)."""

    def water_remaining_l(self) -> float: ...
    def agent(self) -> ExtinguishingAgent: ...
    def release(self, volume_l: float, at: Position) -> float:
        """Drop up to volume_l at `at`; returns litres actually released."""
        ...
    def fill(self, volume_l: float, agent: ExtinguishingAgent = ExtinguishingAgent.WATER) -> float:
        """Add up to volume_l (capped by capacity); returns litres actually added."""
        ...


@runtime_checkable
class PowerSystem(Protocol):
    """Energy state. ELECTRIC: pack only. HYBRID: fuel + generator + battery buffer (fuel_* are None when ELECTRIC)."""

    def battery_pct(self) -> float: ...
    def fuel_remaining_kg(self) -> float | None: ...
    def generator_output_kw(self) -> float | None: ...
    def endurance_remaining_s(self) -> float:
        """Estimate at the CURRENT mass and flight mode; must account for the fuel/battery reserve."""
        ...


@runtime_checkable
class Detector(Protocol):
    """Image -> detections. Implementations: THRESHOLD (classical), ML (ONNX), FUSED, FAKE."""

    kind: DetectorKind

    def detect(self, frame: ThermalFrame) -> list[Detection]: ...


@runtime_checkable
class Georeferencer(Protocol):
    """Pixel -> world. Returns None if the ray misses the ground (e.g. above horizon)."""

    def pixel_to_world(self, u: float, v: float, pose: Pose, camera: CameraModel) -> Position | None: ...


@runtime_checkable
class ThermalSensor(Protocol):
    def capture(self) -> ThermalFrame: ...


@runtime_checkable
class ObstacleSensor(Protocol):
    def obstacles(self) -> list[Obstacle]: ...

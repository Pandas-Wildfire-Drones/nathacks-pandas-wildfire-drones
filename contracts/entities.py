"""Domain entities: the nouns of the system.

Entities are mutable (they change state over a mission). Anything sent over the bus is
wrapped in a frozen message body (see messages.py) and serialized, so receivers never
share Python objects with senders.

Design notes
  * Hotspot.priority_score is COMPUTED BY THE LEADER (the formula lives in core/, not here).
  * HotspotStatus.VERIFIED_OUT is set only via a ground-confirmed command: humans declare fires out.
  * Map cells use log-odds (see GridCellUpdate); convert with p = 1 / (1 + exp(-log_odds)).
"""
from __future__ import annotations

from pydantic import Field, model_validator

from .base import Contract, FrozenContract
from .enums import (
    AlertKind, DroneKind, DroneStatus, ExtinguishingAgent, HotspotStatus, Powertrain, Severity, SwarmRole,
    TaskStatus, TaskType, WaterSourceKind, WaterSourceOrigin, WaterSourceStatus,
)
from .geometry import Position, point_in_polygon

__all__ = [
    "DroneState", "Hotspot", "Zone", "Task", "WaterSource", "RefillStation",
    "GridCellUpdate", "Alert",
]


class DroneState(Contract):
    """Telemetry snapshot. Carried inside HEARTBEAT; the ground base builds its health view from it."""

    drone_id: str
    kind: DroneKind
    role: SwarmRole = SwarmRole.FOLLOWER
    position: Position
    heading_deg: float = Field(default=0.0, ge=0, lt=360)
    speed_mps: float = Field(default=0.0, ge=0)
    powertrain: Powertrain = Powertrain.ELECTRIC
    battery_pct: float = Field(ge=0, le=100)  # ELECTRIC: the main pack. HYBRID: the buffer (emergency landing reserve)
    endurance_remaining_s: float | None = Field(default=None, ge=0)  # drone's own estimate (fuel or battery)
    # HYBRID only (required for HYBRID, forbidden for ELECTRIC): fuel is what limits an extinguisher sortie.
    fuel_remaining_kg: float | None = Field(default=None, ge=0)
    fuel_pct: float | None = Field(default=None, ge=0, le=100)
    generator_output_kw: float | None = Field(default=None, ge=0)
    water_remaining_l: float | None = Field(default=None, ge=0)      # EXTINGUISHER only
    agent: ExtinguishingAgent | None = None                          # what the tank currently holds
    status: DroneStatus = DroneStatus.IDLE
    current_task_id: str | None = None
    leader_id: str | None = None   # who this drone currently believes is leader
    epoch: int = Field(default=0, ge=0)  # leader epoch this drone believes in
    # Health flags keyed by sensor/subsystem name, e.g. {"thermal": True, "lidar": True,
    # "gnss_rtk": False, "comms": True, "payload": True}. Names are defined in config.py (SENSOR_NAMES).
    sensors_ok: dict[str, bool] = Field(default_factory=dict)
    link_quality: float | None = Field(default=None, ge=0, le=1)
    timestamp: float

    @model_validator(mode="after")
    def _powertrain_fields(self) -> "DroneState":
        fuel = (self.fuel_remaining_kg, self.fuel_pct)
        if self.powertrain is Powertrain.HYBRID and any(f is None for f in fuel):
            raise ValueError("HYBRID drones must report fuel_remaining_kg and fuel_pct")
        if self.powertrain is Powertrain.ELECTRIC and (any(f is not None for f in fuel)
                                                       or self.generator_output_kw is not None):
            raise ValueError("ELECTRIC drones must not report fuel or generator fields")
        return self


class Hotspot(Contract):
    hotspot_id: str
    position: Position
    peak_temp_c: float = Field(ge=-50, le=1500)
    est_area_m2: float = Field(default=0.0, ge=0)
    confidence: float = Field(ge=0, le=1)
    status: HotspotStatus = HotspotStatus.DETECTED
    priority_score: float = Field(default=0.0, ge=0)  # set by the leader
    observations: int = Field(default=1, ge=1)        # how many detections fused into this hotspot
    growth_rate_m2_per_min: float | None = None       # from tracking; None until estimated
    assigned_to: str | None = None
    reported_by: str | None = None
    first_seen: float
    last_seen: float

    @model_validator(mode="after")
    def _times_ordered(self) -> "Hotspot":
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must be >= first_seen")
        return self


class Zone(Contract):
    """A search region assigned to a seeker."""

    zone_id: str
    boundary: list[Position] = Field(min_length=3)  # polygon in local x/y
    assigned_to: str | None = None
    last_scanned: float | None = None

    def contains(self, pos: Position) -> bool:
        return point_in_polygon(pos.x, pos.y, self.boundary)


class Task(Contract):
    task_id: str
    type: TaskType
    priority: float = Field(default=0.5, ge=0, le=1)  # 1 = most urgent
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str | None = None
    target_hotspot_id: str | None = None
    target_zone_id: str | None = None
    target_water_source_id: str | None = None
    target_station_id: str | None = None
    created_at: float
    deadline: float | None = None
    epoch: int = Field(default=0, ge=0)  # leader epoch that issued/assigned it
    # EXTINGUISH only: litres (of the loaded agent) to release on this hotspot. One sortie can serve several
    # hotspots because the tank is much bigger than a small spot fire needs. None = "all you have".
    volume_l: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _has_required_target(self) -> "Task":
        t = self.type
        if t is TaskType.SCAN_ZONE and not self.target_zone_id:
            raise ValueError("SCAN_ZONE requires target_zone_id")
        if t in (TaskType.INVESTIGATE, TaskType.EXTINGUISH, TaskType.VERIFY) and not self.target_hotspot_id:
            raise ValueError(f"{t} requires target_hotspot_id")
        if t is TaskType.REFILL and not (self.target_station_id or self.target_water_source_id):
            raise ValueError("REFILL requires target_station_id or target_water_source_id")
        if t is TaskType.SURVEY_WATER and not (self.target_water_source_id or self.target_zone_id):
            raise ValueError("SURVEY_WATER requires target_water_source_id or target_zone_id")
        return self


class WaterSource(Contract):
    """A natural or man-made water body usable for refilling (pond, lake, river, tank)."""

    source_id: str
    position: Position  # intake point (where a drone would hover)
    kind: WaterSourceKind
    origin: WaterSourceOrigin
    status: WaterSourceStatus = WaterSourceStatus.CANDIDATE
    est_volume_l: float | None = Field(default=None, ge=0)
    surface_area_m2: float | None = Field(default=None, ge=0)
    approach_clearance_m: float | None = Field(default=None, ge=0)  # free radius around the intake point
    confidence: float = Field(default=0.5, ge=0, le=1)
    reported_by: str | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0


class RefillStation(Contract):
    """Primary (powered, possibly mobile) refill point with bays and an optional dosing system."""

    station_id: str
    position: Position
    water_available_l: float = Field(ge=0)
    fill_rate_lpm: float = Field(gt=0)
    total_bays: int = Field(default=1, ge=1)
    free_bays: int = Field(default=1, ge=0)
    agents_available: list[ExtinguishingAgent] = Field(
        default_factory=lambda: [ExtinguishingAgent.WATER]
    )
    is_mobile: bool = False
    # HYBRID drones can only refuel at a station (never at a water source). 0 = no fuel service here.
    fuel_available_kg: float = Field(default=0.0, ge=0)
    fuel_rate_kg_per_min: float = Field(default=20.0, gt=0)
    battery_swap_s: float | None = Field(default=None, gt=0)  # None = no swap service (seeker packs)

    @model_validator(mode="after")
    def _bays_consistent(self) -> "RefillStation":
        if self.free_bays > self.total_bays:
            raise ValueError("free_bays cannot exceed total_bays")
        return self


class GridCellUpdate(FrozenContract):
    """Sparse map update. log_odds = ln(p/(1-p)) of 'fire present'; 0 means unknown (p = 0.5)."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    log_odds: float = Field(ge=-10, le=10)
    last_seen: float


class Alert(Contract):
    """Human-facing event for the ground base UI."""

    alert_id: str
    severity: Severity
    kind: AlertKind
    subject_id: str | None = None  # drone/hotspot/source the alert is about
    message: str
    position: Position | None = None
    timestamp: float

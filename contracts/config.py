"""Platform, sensor, agent and scenario configuration.

EVERY NUMBER HERE IS AN ASSUMPTION unless its comment says SOURCED. Rationale and sources live in
context.md (sections 6-8). Change defaults in ONE place (here), then add a changelog line to context.md.
The mech E owns the physical values; CS agents should not "tune" them to make the demo look better.
"""
from __future__ import annotations

from pydantic import Field, model_validator

from .base import Contract
from .entities import RefillStation, WaterSource
from .enums import (
    DroneKind, ExtinguishingAgent, Powertrain, WaterSourceKind, WaterSourceOrigin, WaterSourceStatus,
)
from .geometry import GeoOrigin, GridSpec, Position
from .sensing import CameraModel

__all__ = [
    "SENSOR_NAMES", "ThermalCameraSpec", "LidarSpec", "CommsSpec", "PlatformSpec", "AgentSpec",
    "FireSpec", "DecoySpec", "WindSpec", "ScenarioConfig", "DEFAULT_AGENT_SPECS",
    "default_seeker_spec", "default_extinguisher_spec", "example_scenario",
]

# Keys used in DroneState.sensors_ok and SimFault(DEGRADE_SENSOR).
# "power" = battery pack / fuel system / generator health (both powertrains). "payload" = tank/valve (extinguisher).
SENSOR_NAMES = ("thermal", "lidar", "gnss_rtk", "rangefinder", "comms", "compute", "payload", "power")


class ThermalCameraSpec(Contract):
    camera: CameraModel
    netd_mk: float = Field(default=50.0, gt=0)          # noise-equivalent delta-T, millikelvin
    frame_rate_hz: float = Field(default=9.0, gt=0)     # ASSUMPTION: export-friendly cores are often limited; verify
    saturation_temp_c: float = 550.0                    # ASSUMPTION: low-gain radiometric range; fires can exceed this
    mass_kg: float = Field(default=0.025, gt=0)         # SOURCED: Boson+ ~21-23 g with lens
    power_w: float = Field(default=1.0, gt=0)           # SOURCED: Boson from ~0.5 W; dual thermal+visible module 1.8 W


class LidarSpec(Contract):
    range_m: float = Field(default=40.0, gt=0)          # SOURCED: Livox Mid-360, typical on 10% reflectivity target
    fov_h_deg: float = 360.0                            # SOURCED
    fov_v_deg: float = 59.0                             # SOURCED
    points_per_s: int = 200_000                         # SOURCED
    mass_kg: float = 0.265                              # SOURCED
    power_w: float = 6.5                                # SOURCED (average)


class CommsSpec(Contract):
    """Drone-to-drone and drone-to-ground radio model used by the simulator's lossy bus."""

    range_m: float = Field(default=3000.0, gt=0)        # ASSUMPTION: mesh radio, line of sight
    ground_link_range_m: float = Field(default=5000.0, gt=0)
    latency_s: float = Field(default=0.05, ge=0)
    latency_jitter_s: float = Field(default=0.02, ge=0)
    loss_prob_near: float = Field(default=0.01, ge=0, le=1)
    loss_prob_edge: float = Field(default=0.5, ge=0, le=1)   # loss probability at range_m; interpolate between
    bandwidth_kbps: float = Field(default=500.0, gt=0)       # why raw frames never travel on the bus


class PlatformSpec(Contract):
    """One airframe + powertrain + sensor set. See context.md section 6 for the math behind the defaults.

    ELECTRIC (seekers): energy = battery_wh * battery_usable_frac; power from hover_power_w / cruise_power_w.
    HYBRID (extinguishers): fuel + generator, with a small battery BUFFER for peaks and emergency landing.
        Fuel burn follows burn(m) = fuel_burn_kg_h_at_mtow * (m / max_takeoff_kg)^1.5 (hover-power scaling at
        constant disk loading), calibrated to ACC Thunder Wasp GT figures (context.md section 6.2).
    """

    kind: DroneKind
    name: str
    powertrain: Powertrain = Powertrain.ELECTRIC
    search_capable: bool = True          # False = does not run area search; thermal is for aim/verify only
    empty_mass_kg: float = Field(gt=0)   # includes battery (ELECTRIC) or engine + generator + buffer battery (HYBRID); EXCLUDES fuel
    max_takeoff_kg: float = Field(gt=0)
    water_capacity_l: float = Field(default=0.0, ge=0)  # 1 L of water ~ 1 kg
    cruise_speed_mps: float = Field(gt=0)
    max_speed_mps: float = Field(gt=0)
    max_accel_mps2: float = Field(default=3.0, gt=0)    # also used as braking decel for stopping distance
    endurance_s: float = Field(gt=0)                    # RATED, at full mass, to empty (no reserve); see methods below
    search_altitude_agl_m: float = Field(gt=0)          # extinguishers: transit altitude (they do not search)
    drop_height_agl_m: float = Field(default=10.0, gt=0)
    drop_release_s: float = Field(default=8.0, gt=0)    # time to empty the tank
    refill_time_s: float = Field(default=120.0, gt=0)   # hook-up + fill (+ dosing)
    avionics_mass_kg: float = Field(gt=0)
    avionics_power_w: float = Field(gt=0)
    thermal: ThermalCameraSpec
    lidar: LidarSpec | None = None
    has_rtk_gnss: bool = True
    sensor_names: list[str] = Field(default_factory=lambda: list(SENSOR_NAMES))

    # --- ELECTRIC energy model ---
    battery_wh: float | None = Field(default=None, gt=0)
    battery_usable_frac: float = Field(default=0.8, gt=0, le=1)
    hover_power_w: float | None = Field(default=None, gt=0)   # total electric incl. avionics
    cruise_power_w: float | None = Field(default=None, gt=0)  # total electric incl. avionics

    # --- HYBRID energy model ---
    fuel_capacity_kg: float | None = Field(default=None, gt=0)
    fuel_density_kg_per_l: float = Field(default=0.8, gt=0)   # Jet A-1 / diesel class
    fuel_reserve_frac: float = Field(default=0.15, ge=0, lt=1)  # never plan below this fraction of capacity
    fuel_burn_kg_h_at_mtow: float | None = Field(default=None, gt=0)
    battery_buffer_wh: float | None = Field(default=None, gt=0)  # emergency-landing / peak buffer

    @model_validator(mode="after")
    def _physics_sane(self) -> "PlatformSpec":
        fuel = self.fuel_capacity_kg or 0.0
        if self.empty_mass_kg + self.water_capacity_l + fuel > self.max_takeoff_kg + 1e-9:
            raise ValueError("empty_mass_kg + water_capacity_l + fuel_capacity_kg exceeds max_takeoff_kg")
        if self.cruise_speed_mps > self.max_speed_mps:
            raise ValueError("cruise_speed_mps exceeds max_speed_mps")
        unknown = set(self.sensor_names) - set(SENSOR_NAMES)
        if unknown:
            raise ValueError(f"unknown sensor names: {sorted(unknown)}")
        if self.powertrain is Powertrain.ELECTRIC:
            if None in (self.battery_wh, self.hover_power_w, self.cruise_power_w):
                raise ValueError("ELECTRIC needs battery_wh, hover_power_w, cruise_power_w")
            if self.fuel_capacity_kg is not None or self.fuel_burn_kg_h_at_mtow is not None:
                raise ValueError("ELECTRIC must not define fuel fields")
        else:
            if None in (self.fuel_capacity_kg, self.fuel_burn_kg_h_at_mtow, self.battery_buffer_wh):
                raise ValueError("HYBRID needs fuel_capacity_kg, fuel_burn_kg_h_at_mtow, battery_buffer_wh")
        return self

    def stopping_distance_m(self, speed_mps: float) -> float:
        """v^2 / (2a): distance needed to stop from `speed_mps` at max_accel_mps2."""
        return speed_mps ** 2 / (2 * self.max_accel_mps2)

    # ---- ELECTRIC helpers ----
    def electric_endurance_s(self, hover: bool = False) -> float:
        if self.powertrain is not Powertrain.ELECTRIC:
            raise ValueError("electric_endurance_s is for ELECTRIC platforms")
        power = self.hover_power_w if hover else self.cruise_power_w
        return self.battery_wh * self.battery_usable_frac / power * 3600.0

    # ---- HYBRID helpers ----
    def fuel_burn_kg_h(self, mass_kg: float) -> float:
        """Burn rate at total mass `mass_kg` (hover-power scaling, exponent 1.5)."""
        if self.powertrain is not Powertrain.HYBRID:
            raise ValueError("fuel_burn_kg_h is for HYBRID platforms")
        return self.fuel_burn_kg_h_at_mtow * (mass_kg / self.max_takeoff_kg) ** 1.5

    def hybrid_endurance_s(self, water_l: float | None = None, fuel_kg: float | None = None,
                           reserve: bool = False) -> float:
        """Closed-form flight time while carrying `water_l` the whole time, burning `fuel_kg` (default: full tank).

        With dm/dt = -b0 (m/M)^1.5 the substitution u = m^-0.5 gives u(t) = u0 + 0.5 b0 M^-1.5 t, hence
        t = 2 M^1.5 (m1^-0.5 - m0^-0.5) / b0. `reserve=True` stops at fuel_reserve_frac of capacity.
        Conservative: real sorties drop water, so mass (and burn) falls.
        """
        if self.powertrain is not Powertrain.HYBRID:
            raise ValueError("hybrid_endurance_s is for HYBRID platforms")
        w = self.water_capacity_l if water_l is None else water_l
        f0 = self.fuel_capacity_kg if fuel_kg is None else fuel_kg
        f1 = self.fuel_capacity_kg * self.fuel_reserve_frac if reserve else 0.0
        m0 = self.empty_mass_kg + w + f0
        m1 = self.empty_mass_kg + w + f1
        b0 = self.fuel_burn_kg_h_at_mtow
        hours = 2.0 * self.max_takeoff_kg ** 1.5 * (m1 ** -0.5 - m0 ** -0.5) / b0
        return hours * 3600.0


class AgentSpec(Contract):
    """Extinguishing agent model. `water_equiv_multiplier`: litres of plain water that 1 L of this agent replaces."""

    agent: ExtinguishingAgent
    water_equiv_multiplier: float = Field(gt=0)
    concentrate_ratio: float | None = Field(default=None, ge=0, le=1)  # fraction of concentrate in solution
    needs_dosing: bool                  # True = needs an onboard/inline proportioner at fill time
    aquatic_buffer_m: float = Field(default=0.0, ge=0)  # no aerial application within this distance of waterways
    note: str = ""


# SOURCED/ASSUMED values, see context.md section 7.
DEFAULT_AGENT_SPECS: dict[ExtinguishingAgent, AgentSpec] = {
    ExtinguishingAgent.WATER: AgentSpec(
        agent=ExtinguishingAgent.WATER, water_equiv_multiplier=1.0, needs_dosing=False,
        note="Baseline. Compatible with any refill source.",
    ),
    ExtinguishingAgent.CLASS_A_FOAM: AgentSpec(
        agent=ExtinguishingAgent.CLASS_A_FOAM, water_equiv_multiplier=3.0, concentrate_ratio=0.003,
        needs_dosing=True, aquatic_buffer_m=91.44,
        note="Mix 0.1-1% (SOURCED). Manufacturer claims 3-5x vs plain water; independent evidence is mixed "
             "(ASSUMPTION 3x). US guideline: no aerial foam/retardant within 300 ft (91.44 m) of waterways "
             "(VERIFY Alberta/Canada equivalent).",
    ),
    ExtinguishingAgent.GEL: AgentSpec(
        agent=ExtinguishingAgent.GEL, water_equiv_multiplier=1.5, needs_dosing=True, aquatic_buffer_m=91.44,
        note="PLACEHOLDER multiplier, no source. Gel consistency is sensitive to water hardness (SOURCED), "
             "so basin water quality matters.",
    ),
}


def default_seeker_spec() -> PlatformSpec:
    """Light electric multirotor scout. ASSUMPTIONS (context.md 6.3): 10 kg MTOW, 4 kg pack at 250 Wh/kg,
    4 x 26" props. Hover ~1.07 kW, cruise ~0.92 kW (both incl. 48 W avionics) -> ~52 min. The option of a
    VTOL fixed-wing seeker (~2.5 h) is an [OPEN] question in context.md."""
    return PlatformSpec(
        kind=DroneKind.SEEKER, name="Seeker (electric multirotor, 10 kg)", powertrain=Powertrain.ELECTRIC,
        search_capable=True,
        empty_mass_kg=10.0, max_takeoff_kg=10.0, water_capacity_l=0.0,   # empty includes 4 kg battery + 1 kg avionics
        cruise_speed_mps=15.0, max_speed_mps=20.0, max_accel_mps2=4.0, endurance_s=3150.0,
        search_altitude_agl_m=80.0, avionics_mass_kg=1.0, avionics_power_w=48.0,
        thermal=ThermalCameraSpec(camera=CameraModel(width_px=640, height_px=512, hfov_deg=32.0)),
        lidar=LidarSpec(),
        sensor_names=["thermal", "lidar", "gnss_rtk", "rangefinder", "comms", "compute", "power"],
        battery_wh=1000.0, battery_usable_frac=0.8, hover_power_w=1070.0, cruise_power_w=915.0,
    )


def default_extinguisher_spec() -> PlatformSpec:
    """Ultra-heavy hybrid, Thunder-Wasp-GT class. SOURCED anchors: MTOW 800 kg, empty 375 kg, up to 248 kg fuel,
    400 kg payload, cruise 70-90 km/h (ACC dossier). ASSUMED: +45 kg for hybrid drivetrain and 5 kWh buffer
    (empty 420), tank 170 L, fuel 210 kg, burn 52 kg/h at MTOW (80 kg/h turbine fitted to ACC x 0.65 genset
    factor). Gives ~5.1 h to dry / ~4.1 h with 15% reserve at full mass (hybrid_endurance_s())."""
    return PlatformSpec(
        kind=DroneKind.EXTINGUISHER, name="Extinguisher (hybrid ultra-heavy, 170 L)", powertrain=Powertrain.HYBRID,
        search_capable=False,
        empty_mass_kg=420.0, max_takeoff_kg=800.0, water_capacity_l=170.0,
        cruise_speed_mps=20.0, max_speed_mps=25.0, max_accel_mps2=2.0, endurance_s=18000.0,
        search_altitude_agl_m=120.0,   # transit altitude: above canopy + margin; stopping dist at 20 m/s is 100 m
        drop_height_agl_m=15.0, drop_release_s=8.0, refill_time_s=120.0,
        avionics_mass_kg=1.2, avionics_power_w=35.0,
        thermal=ThermalCameraSpec(camera=CameraModel(width_px=320, height_px=256, hfov_deg=50.0)),
        lidar=LidarSpec(),   # last-mile only: 40 m range cannot protect a 20 m/s heavy drone (100 m to stop)
        sensor_names=["thermal", "lidar", "gnss_rtk", "rangefinder", "comms", "compute", "payload", "power"],
        fuel_capacity_kg=210.0, fuel_density_kg_per_l=0.8, fuel_reserve_frac=0.15,
        fuel_burn_kg_h_at_mtow=52.0, battery_buffer_wh=5000.0,
    )


class FireSpec(Contract):
    fire_id: str
    position: Position
    ignition_time_s: float = Field(default=0.0, ge=0)
    initial_area_m2: float = Field(gt=0)
    peak_temp_c: float = Field(default=450.0, ge=-50, le=1500)
    growth_rate_m2_per_min: float = Field(default=0.1, ge=0)  # night fire behaviour is low; derive from CWFIS


class DecoySpec(Contract):
    """Heat source that is NOT a fire (warm rock, vehicle, permitted campfire). Detectors must not waste water on it."""

    decoy_id: str
    position: Position
    temp_c: float
    area_m2: float = Field(gt=0)
    kind: str = "warm_rock"


class WindSpec(Contract):
    speed_mps: float = Field(default=2.0, ge=0)
    from_deg: float = Field(default=270.0, ge=0, lt=360)  # compass direction wind blows FROM


def _default_origin() -> GeoOrigin:
    # PLACEHOLDER site (central Alberta). Replace with the real scenario location.
    return GeoOrigin(lat_deg=53.5, lon_deg=-113.5, utm_epsg=32612)


def _default_grid() -> GridSpec:
    # 4 km x 4 km at 2 m cells = 2000 x 2000 (4M cells = 16 MB per float32 layer)
    return GridSpec(origin=Position(x=-2000.0, y=-2000.0), resolution_m=2.0, rows=2000, cols=2000)


class ScenarioConfig(Contract):
    name: str = "default"
    seed: int = 0
    duration_s: float = Field(default=3600.0, gt=0)
    time_scale: float = Field(default=1.0, gt=0)  # sim seconds per wall second
    night: bool = True
    ambient_temp_c: float = 8.0
    wind: WindSpec = Field(default_factory=WindSpec)
    origin: GeoOrigin = Field(default_factory=_default_origin)
    grid: GridSpec = Field(default_factory=_default_grid)
    base_position: Position = Field(default_factory=lambda: Position(x=0.0, y=0.0, z=0.0))

    n_seekers: int = Field(default=6, ge=0)        # ~52 min sorties, so plan for battery swaps
    n_extinguishers: int = Field(default=2, ge=0)  # ultra-heavy: few, big, fuel-limited
    seeker_spec: PlatformSpec = Field(default_factory=default_seeker_spec)
    extinguisher_spec: PlatformSpec = Field(default_factory=default_extinguisher_spec)

    agent: ExtinguishingAgent = ExtinguishingAgent.WATER
    agent_specs: dict[ExtinguishingAgent, AgentSpec] = Field(default_factory=lambda: dict(DEFAULT_AGENT_SPECS))
    suppression_l_per_m2: float = Field(default=10.0, gt=0)  # PLACEHOLDER: mech E must source/derive (fuel, burn depth)

    use_water_sources: bool = True  # feature flag: allow refilling from natural water sources
    water_sources: list[WaterSource] = Field(default_factory=list)  # prior map (CANDIDATE) and/or known sources
    stations: list[RefillStation] = Field(default_factory=list)

    fires: list[FireSpec] = Field(default_factory=list)
    decoys: list[DecoySpec] = Field(default_factory=list)
    comms: CommsSpec = Field(default_factory=CommsSpec)

    @model_validator(mode="after")
    def _consistent(self) -> "ScenarioConfig":
        if not self.grid.contains(self.base_position):
            raise ValueError("base_position must lie inside the grid")
        for label, ids in (
            ("fire_id", [f.fire_id for f in self.fires]),
            ("decoy_id", [d.decoy_id for d in self.decoys]),
            ("source_id", [w.source_id for w in self.water_sources]),
            ("station_id", [s.station_id for s in self.stations]),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} in scenario")
        if self.agent not in self.agent_specs:
            raise ValueError("agent has no entry in agent_specs")
        if self.seeker_spec.kind is not DroneKind.SEEKER or self.extinguisher_spec.kind is not DroneKind.EXTINGUISHER:
            raise ValueError("seeker_spec/extinguisher_spec have the wrong DroneKind")
        return self


def example_scenario() -> ScenarioConfig:
    """Small, valid scenario for sim/UI developers to start from. Positions are local meters."""
    return ScenarioConfig(
        name="example_night_initial_attack",
        seed=42,
        fires=[
            FireSpec(fire_id="F1", position=Position(x=800, y=1200), ignition_time_s=0,
                     initial_area_m2=2.0, peak_temp_c=450, growth_rate_m2_per_min=0.10),
            FireSpec(fire_id="F2", position=Position(x=-1500, y=600), ignition_time_s=300,
                     initial_area_m2=1.0, peak_temp_c=300, growth_rate_m2_per_min=0.05),
            FireSpec(fire_id="F3", position=Position(x=300, y=-1700), ignition_time_s=900,
                     initial_area_m2=4.0, peak_temp_c=600, growth_rate_m2_per_min=0.20),
        ],
        decoys=[DecoySpec(decoy_id="D1", position=Position(x=-400, y=900), temp_c=35.0, area_m2=6.0)],
        water_sources=[
            WaterSource(source_id="W1", position=Position(x=1000, y=900), kind=WaterSourceKind.POND,
                        origin=WaterSourceOrigin.PRIOR_MAP, status=WaterSourceStatus.CANDIDATE,
                        est_volume_l=2_000_000, confidence=0.6),
            WaterSource(source_id="W2", position=Position(x=-1800, y=-900), kind=WaterSourceKind.LAKE,
                        origin=WaterSourceOrigin.PRIOR_MAP, status=WaterSourceStatus.CANDIDATE,
                        est_volume_l=50_000_000, confidence=0.8),
        ],
        stations=[
            RefillStation(station_id="S1", position=Position(x=0, y=0), water_available_l=20_000,
                          fill_rate_lpm=300.0, total_bays=2, free_bays=2,
                          fuel_available_kg=1500.0, fuel_rate_kg_per_min=30.0, battery_swap_s=120.0,
                          agents_available=[ExtinguishingAgent.WATER, ExtinguishingAgent.CLASS_A_FOAM]),
        ],
    )

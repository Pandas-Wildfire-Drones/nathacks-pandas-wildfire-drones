"""Game configuration. Physical defaults come from contracts/config.py; game-only knobs are tagged [GAME-SCALE].

[GAME-SCALE] values exist only to make a ~3.5 minute demo readable on a small arena. They are NOT physics claims and
the MechE (workstream E) owns replacing them. Everything else should be read from the contracts specs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from contracts import (
    DEFAULT_AGENT_SPECS, CameraModel, DroneKind, ExtinguishingAgent, LidarSpec, PlatformSpec, Powertrain,
    ThermalCameraSpec, default_extinguisher_spec, default_seeker_spec,
)

__all__ = ["FireType", "DecoyType", "GameConfig", "FIRE_TYPES", "DECOY_TYPES", "DEMO_FIRE_TYPES", "DEMO_MAPS",
           "attacker_spec", "demo_config"]


@dataclass(frozen=True)
class FireType:
    """One class of spot fire. ROS = rate of spread in REAL m/min (multiplied by fire_time_scale in game)."""

    name: str
    count: int
    area_m2: tuple[float, float]        # initial burning area range
    ros_m_min: tuple[float, float]      # head-fire rate of spread, real m/min [ASSUMPTION, night = low]
    temp_scale: tuple[float, float]     # multiplies (real FLAME 3 fire texture - ground): 1.0 = flaming
    residence_min: tuple[float, float]  # how long one cell burns before it is out (real minutes)
    water_mult: float = 1.0             # litres needed per m2 relative to suppression_l_per_m2
    wind_k: float = 1.5                 # wind elongation strength (0 = circular)


# [ASSUMPTION] my defaults, see exclude/simulator_task.md section 7. MechE should own these.
# Counts scale with the arena (radius 220 m = 1.7x the old 170 m area).
FIRE_TYPES: tuple[FireType, ...] = (
    FireType("SMOULDER", 6, (0.3, 1.5), (0.02, 0.05), (0.35, 0.5), (20.0, 40.0), 1.0, 0.5),
    FireType("CREEPING", 8, (1.0, 4.0), (0.08, 0.18), (0.75, 0.9), (4.0, 8.0), 1.0, 1.5),
    FireType("ACTIVE_SPOT", 5, (3.0, 8.0), (0.2, 0.4), (0.95, 1.0), (2.0, 4.0), 1.0, 2.5),
    FireType("HOLDOVER", 3, (0.5, 1.5), (0.0, 0.01), (0.6, 0.8), (600.0, 900.0), 2.5, 0.0),
)

# DEMO mode (large maps): more fires, and a wider spread of size / rate of spread / temperature so the priority
# queue has something to sort. [ASSUMPTION / GAME-SCALE], MechE should own these.
DEMO_FIRE_TYPES: tuple[FireType, ...] = (
    FireType("SMOULDER", 12, (0.5, 2.0), (0.02, 0.06), (0.35, 0.5), (20.0, 40.0), 1.0, 0.5),
    FireType("CREEPING", 14, (1.0, 4.0), (0.10, 0.25), (0.75, 0.9), (4.0, 8.0), 1.0, 1.5),
    FireType("ACTIVE_SPOT", 9, (2.5, 7.0), (0.30, 0.60), (0.95, 1.0), (2.0, 4.0), 1.0, 2.5),
    FireType("HOLDOVER", 5, (0.5, 1.5), (0.0, 0.01), (0.6, 0.8), (600.0, 900.0), 2.5, 0.0),
)


@dataclass(frozen=True)
class DecoyType:
    name: str
    count: int
    temp_c: tuple[float, float]
    area_m2: tuple[float, float]


DECOY_TYPES: tuple[DecoyType, ...] = (
    DecoyType("warm_rock", 3, (38.0, 55.0), (2.0, 6.0)),
    DecoyType("vehicle", 1, (70.0, 105.0), (8.0, 9.0)),
    DecoyType("cabin", 2, (35.0, 45.0), (30.0, 40.0)),
)

DEMO_DECOY_TYPES: tuple[DecoyType, ...] = (
    DecoyType("warm_rock", 6, (38.0, 55.0), (2.0, 6.0)),
    DecoyType("vehicle", 3, (70.0, 105.0), (8.0, 9.0)),
    DecoyType("cabin", 3, (35.0, 45.0), (30.0, 40.0)),
)


@dataclass(frozen=True)
class GameConfig:
    # --- arena -------------------------------------------------------------------------------------------------
    arena_radius_m: float = 220.0          # [GAME-SCALE] circle; context.md default grid is 4 x 4 km
    base_xy: tuple[float, float] = (0.0, -195.0)
    world_res_m: float = 0.15              # thermal raster (FLAME 3 NADIR is 0.13-0.16 m/px)
    fire_cell_m: float = 0.5               # fire model cell
    map_cell_m: float = 2.0                # perception log-odds map + snapshot rasters
    plan_cell_m: float = 20.0              # autonomy CBS/A* planning grid (no obstacles at transit altitude)
    n_trees: int = 700
    fire_clusters: int = 3
    cluster_r_m: float = 28.0
    # lakes (DEMO maps): natural refill points. lakes_wet=False keeps the same basins dry (identical fire layout)
    n_lakes: int = 0
    lake_r_m: tuple[float, float] = (30.0, 70.0)
    lakes_wet: bool = True
    use_water_sources: bool = True         # leader may send extinguishers to validated lakes (contracts flag)

    # --- phases (wall seconds; total <= 4 min) ---------------------------------------------------------------
    briefing_s: float = 8.0
    seek_s: float = 90.0
    plan_s: float = 30.0
    extinguish_s: float = 90.0

    # --- loop -------------------------------------------------------------------------------------------------
    tick_hz: float = 20.0
    perception_hz: float = 2.0             # real camera is 9 Hz; 2 Hz keeps CPU sane with the CNN in the loop
    perception_latency_s: float = 0.4      # frame -> hotspot available (deterministic, the sim waits if needed)
    snapshot_hz: float = 15.0

    # --- fire ---------------------------------------------------------------------------------------------------
    fire_time_scale: float = 3.0           # [GAME-SCALE] fire-seconds per wall second
    fire_step_s: float = 0.5               # wall seconds between fire updates
    suppression_l_per_m2: float = 10.0     # PLACEHOLDER from contracts ScenarioConfig
    wet_block_l_per_m2: float = 2.0        # wetness that fully blocks ignition [ASSUMPTION]
    evaporation_l_per_m2_min: float = 0.3  # real minutes [ASSUMPTION]
    wind_speed_mps: float = 3.0
    wind_from_deg: float = 250.0           # compass, wind blows FROM here
    fires: tuple[FireType, ...] = FIRE_TYPES
    decoys: tuple[DecoyType, ...] = DECOY_TYPES

    # --- drones -------------------------------------------------------------------------------------------------
    n_extinguishers: int = 19              # human: player + 18 bots (+ the seeker = 20 drones); swarm: 19
    swarm_seekers: int = 1                 # D3: configurable, default = the human's 1
    seeker_min_agl_m: float = 30.0
    seeker_max_agl_m: float = 150.0
    ext_min_agl_m: float = 8.0
    ext_max_agl_m: float = 120.0
    transit_agl_m: float = 40.0            # [GAME-SCALE] context.md says >= 120 m; trees here are <= 24 m
    vertical_speed_mps: float = 5.0        # [ASSUMPTION]
    lawnmower_agl_m: float = 150.0         # swarm seeker search altitude = the player's ceiling (swath 86 m)
    lawnmower_overlap: float = 0.05
    lawnmower_speed_mps: float = 20.0      # seeker max speed (the player's boost speed)
    drop_agl_m: float = 15.0               # extinguisher spec drop height
    drop_max_effective_agl_m: float = 30.0
    drop_spread_per_m: float = 0.05        # footprint radius grows with drop height [ASSUMPTION, MechE]
    drop_drift_factor: float = 0.3         # drift = factor x wind x fall time (water column lags the wind) [ASSUMPTION]
    refill_dwell_s: float = 20.0           # [GAME-SCALE] spec refill_time_s is 120 s
    hotspot_volume_margin: float = 1.4     # litres per task = est_area x l/m2 x margin (stub assigner)
    min_task_volume_l: float = 25.0

    # --- swarm brain (core/) ---------------------------------------------------------------------------------------
    max_per_fire: int = 1                  # heavy CBS-planned extinguishers: one at a time per fire
    prefill_idle: bool = False
    stage_idle: bool = False
    agent_mult: float = 1.0                # water-equivalent of 1 L of the loaded agent (contracts AgentSpec)

    # --- DEMO mode -------------------------------------------------------------------------------------------------
    demo: bool = False
    demo_s: float = 120.0                  # seekers and extinguishers run together for this long
    demo_seekers: int = 6
    loop_search: bool = False              # seekers re-fly their zone (reversed) when done: re-observation -> growth
    attacker_fill_s: float = 5.0           # [GAME-SCALE] pickup at a lake or the base (hover + pump + dose)
    attacker_drop_s: float = 1.5           # release time for one load
    attacker_fill_agl_m: float = 3.0       # hover height over the water while pumping [OPEN intake method]

    seeker_spec: object = field(default_factory=default_seeker_spec)
    extinguisher_spec: object = field(default_factory=default_extinguisher_spec)

    @property
    def arena_size_m(self) -> float:
        return 2.0 * self.arena_radius_m

    @property
    def total_s(self) -> float:
        if self.demo:
            return self.demo_s
        return self.briefing_s + self.seek_s + self.plan_s + self.extinguish_s


def attacker_spec() -> PlatformSpec:
    """Small field-refill attacker (DEMO mode). Carries ~15 L per drop (team assumption: 10-20 kg suppressant
    payload), mixes foam concentrate into water it picks up at ANY refill point (lake or base), so it refills after
    every drop. Masses and burn are PLACEHOLDERS for the PlatformSpec validator, not a design: MechE (E) owns them."""
    return PlatformSpec(
        kind=DroneKind.EXTINGUISHER, name="Attacker (field-refill, ~15 L)", powertrain=Powertrain.HYBRID,
        search_capable=False, empty_mass_kg=32.0, max_takeoff_kg=50.0, water_capacity_l=15.0,
        cruise_speed_mps=20.0, max_speed_mps=24.0, max_accel_mps2=4.0, endurance_s=5 * 3600.0,
        search_altitude_agl_m=40.0, drop_height_agl_m=15.0, drop_release_s=1.5, refill_time_s=5.0,
        avionics_mass_kg=0.8, avionics_power_w=30.0,
        thermal=ThermalCameraSpec(camera=CameraModel(width_px=320, height_px=256, hfov_deg=50.0)),
        lidar=LidarSpec(),
        sensor_names=["thermal", "lidar", "gnss_rtk", "rangefinder", "comms", "compute", "payload", "power"],
        fuel_capacity_kg=3.0, fuel_reserve_frac=0.3, fuel_burn_kg_h_at_mtow=0.5, battery_buffer_wh=1000.0,
    )


# DEMO maps: (thermal tiles, lakes wet?, layout seed). Same layout seed => identical fires; only the water differs.
DEMO_MAPS: dict[str, dict] = {
    "demo-lakes-1": {"tiles": "p1", "wet": True, "layout": 11, "title": "DEMO · Lakes · Hanna Hammock 1"},
    "demo-dry-1": {"tiles": "p1", "wet": False, "layout": 11, "title": "DEMO · Dry basins · Hanna Hammock 1"},
    "demo-lakes-2": {"tiles": "p2", "wet": True, "layout": 23, "title": "DEMO · Lakes · Hanna Hammock 2"},
    "demo-dry-2": {"tiles": "p2", "wet": False, "layout": 23, "title": "DEMO · Dry basins · Hanna Hammock 2"},
}


def demo_config(wet: bool = True, **over) -> GameConfig:
    """Large DEMO map. [GAME-SCALE] numbers chosen to run smoothly on a laptop (see sim/README.md)."""
    cfg = GameConfig(
        demo=True, arena_radius_m=450.0, base_xy=(0.0, -420.0), world_res_m=0.3, map_cell_m=3.0,
        n_trees=2200, fire_clusters=6, cluster_r_m=60.0, n_lakes=6, lake_r_m=(25.0, 50.0), lakes_wet=wet,
        fires=DEMO_FIRE_TYPES, decoys=DEMO_DECOY_TYPES, fire_time_scale=3.0, perception_hz=1.5,
        demo_seekers=8, n_extinguishers=40, loop_search=True, max_per_fire=6, prefill_idle=True, stage_idle=True,
        transit_agl_m=30.0, vertical_speed_mps=8.0,
        agent_mult=DEFAULT_AGENT_SPECS[ExtinguishingAgent.CLASS_A_FOAM].water_equiv_multiplier,
        hotspot_volume_margin=3.5, min_task_volume_l=10.0, extinguisher_spec=attacker_spec(),
    )
    return replace(cfg, **over) if over else cfg

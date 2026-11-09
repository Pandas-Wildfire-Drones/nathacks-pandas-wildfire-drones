"""Game configuration. Physical defaults come from contracts/config.py; game-only knobs are tagged [GAME-SCALE].

[GAME-SCALE] values exist only to make a ~3.5 minute demo readable on a small arena. They are NOT physics claims and
the MechE (workstream E) owns replacing them. Everything else should be read from the contracts specs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from contracts import default_extinguisher_spec, default_seeker_spec

__all__ = ["FireType", "DecoyType", "GameConfig", "FIRE_TYPES", "DECOY_TYPES"]


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
FIRE_TYPES: tuple[FireType, ...] = (
    FireType("SMOULDER", 4, (0.3, 1.5), (0.02, 0.05), (0.35, 0.5), (20.0, 40.0), 1.0, 0.5),
    FireType("CREEPING", 5, (1.0, 4.0), (0.08, 0.18), (0.75, 0.9), (4.0, 8.0), 1.0, 1.5),
    FireType("ACTIVE_SPOT", 3, (3.0, 8.0), (0.2, 0.4), (0.95, 1.0), (2.0, 4.0), 1.0, 2.5),
    FireType("HOLDOVER", 2, (0.5, 1.5), (0.0, 0.01), (0.6, 0.8), (600.0, 900.0), 2.5, 0.0),
)


@dataclass(frozen=True)
class DecoyType:
    name: str
    count: int
    temp_c: tuple[float, float]
    area_m2: tuple[float, float]


DECOY_TYPES: tuple[DecoyType, ...] = (
    DecoyType("warm_rock", 2, (38.0, 55.0), (2.0, 6.0)),
    DecoyType("vehicle", 1, (70.0, 105.0), (8.0, 9.0)),
    DecoyType("cabin", 1, (35.0, 45.0), (30.0, 40.0)),
)


@dataclass(frozen=True)
class GameConfig:
    # --- arena -------------------------------------------------------------------------------------------------
    arena_radius_m: float = 170.0          # [GAME-SCALE] circle; context.md default grid is 4 x 4 km
    base_xy: tuple[float, float] = (0.0, -150.0)
    world_res_m: float = 0.15              # thermal raster (FLAME 3 NADIR is 0.13-0.16 m/px)
    fire_cell_m: float = 0.5               # fire model cell
    map_cell_m: float = 2.0                # perception log-odds map + snapshot rasters
    plan_cell_m: float = 20.0              # autonomy CBS/A* planning grid (no obstacles at transit altitude)
    n_trees: int = 420

    # --- phases (wall seconds; total <= 4 min) ---------------------------------------------------------------
    briefing_s: float = 8.0
    seek_s: float = 75.0
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
    n_extinguishers: int = 10              # human: player + 9 bots; swarm: 10
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

    seeker_spec: object = field(default_factory=default_seeker_spec)
    extinguisher_spec: object = field(default_factory=default_extinguisher_spec)

    @property
    def arena_size_m(self) -> float:
        return 2.0 * self.arena_radius_m

    @property
    def total_s(self) -> float:
        return self.briefing_s + self.seek_s + self.plan_s + self.extinguish_s

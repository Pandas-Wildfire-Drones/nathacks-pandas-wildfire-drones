"""Coordinates autonomy decisions for one drone."""

from contracts.entities import DroneState, Hotspot, Task, Zone
from contracts.geometry import GridSpec, Position

from autonomy.coverage_search import generate_coverage_waypoints
from autonomy.deconfliction import detect_conflicts
from autonomy.path_planning import find_path
from autonomy.resource_manager import assess_mission_resources
from autonomy.water_drop import (
    DropDecision,
    WaterDropLimits,
    evaluate_water_drop,
)


class DroneController:
    """Coordinates planning and safety checks for an individual drone."""

    def __init__(self, drop_limits: WaterDropLimits):
        self.drop_limits = drop_limits

    def plan_route(
        self,
        occupancy_grid: list[list[int]],
        grid_spec: GridSpec,
        start: Position,
        goal: Position,
    ) -> list[Position] | None:
        """Return an A* route, or None if no route exists."""
        return find_path(occupancy_grid, grid_spec, start, goal)

    def plan_coverage(
        self,
        zone: Zone,
        spacing_m: float,
        *,
        z_m: float | None = None,
    ) -> list[Position]:
        """Generate search waypoints inside an assigned zone."""
        return generate_coverage_waypoints(
            zone,
            spacing_m,
            z_m=z_m,
        )

    def check_resources(
        self,
        drone: DroneState,
        outbound_distance_m: float,
        return_distance_m: float,
        cruise_speed_mps: float,
        **resource_estimates,
    ):
        """Assess whether the drone can finish its mission and retain reserves."""
        return assess_mission_resources(
            drone=drone,
            outbound_distance_m=outbound_distance_m,
            return_distance_m=return_distance_m,
            cruise_speed_mps=cruise_speed_mps,
            **resource_estimates,
        )

    def check_path_conflicts(
        self,
        proposed_path,
        other_paths,
        minimum_separation_m: float,
    ):
        """Check a proposed 3D path against other drones' paths."""
        return detect_conflicts(
            proposed_path,
            other_paths,
            minimum_separation_m,
        )

    def check_drop(
        self,
        drone: DroneState,
        hotspot: Hotspot,
        task: Task,
        altitude_agl_m: float,
    ) -> DropDecision:
        """Return a safe-release decision for this drone and task."""
        return evaluate_water_drop(
            drone=drone,
            hotspot=hotspot,
            task=task,
            altitude_agl_m=altitude_agl_m,
            limits=self.drop_limits,
        )

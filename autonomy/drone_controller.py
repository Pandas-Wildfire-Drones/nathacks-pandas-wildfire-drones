"""DroneController coordinates the individual drone's autonomy calculations."""
from autonomy.path_planning import find_path
from autonomy.resource_manager import assess_mission_resources
from autonomy.water_drop import evaluate_water_drop
from autonomy.deconfliction import detect_conflicts


class DroneController:
    """Coordinates the individual drone's autonomy calculations."""

    def plan_route(self, grid, current_cell, target_cell):
        return find_path(grid, current_cell, target_cell)

    def check_resources(self, **resource_inputs):
        return assess_mission_resources(**resource_inputs)

    def check_drop_conditions(self, **drop_inputs):
        return evaluate_water_drop(**drop_inputs)

    def check_path_conflicts(
        self, proposed_path, other_paths, minimum_separation_m
    ):
        return detect_conflicts(
            proposed_path,
            other_paths,
            minimum_separation_m,
        )

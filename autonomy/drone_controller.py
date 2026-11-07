"""Coordinates planning, safety and execution for one drone."""

from dataclasses import dataclass
from math import isfinite

from contracts.entities import DroneState, Hotspot, Task, Zone
from contracts.enums import (
    DroneKind,
    HotspotStatus,
    TaskStatus,
    TaskType,
)
from contracts.geometry import GridSpec, Position
from contracts.interfaces import Autopilot, PayloadSystem

from autonomy.coverage_search import generate_coverage_waypoints
from autonomy.deconfliction import detect_conflicts
from autonomy.path_planning import find_path
from autonomy.resource_manager import assess_mission_resources
from autonomy.water_drop import (
    DropDecision,
    WaterDropLimits,
    evaluate_water_drop,
)


@dataclass(frozen=True)
class MissionExecutionResult:
    """Outcome of one step in an extinguisher mission."""

    status: str
    success: bool
    water_used_l: float
    reason: str


class DroneController:
    """Coordinates planning and mission execution for an individual drone."""

    def __init__(
        self,
        drop_limits: WaterDropLimits,
        autopilot: Autopilot | None = None,
        payload: PayloadSystem | None = None,
    ):
        self.drop_limits = drop_limits
        self.autopilot = autopilot
        self.payload = payload

        self._active_task_id: str | None = None
        self._terminal_result: MissionExecutionResult | None = None

    # ---------------------------------------------------------
    # Existing planning functions
    # ---------------------------------------------------------

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
        return generate_coverage_waypoints(zone, spacing_m, z_m=z_m)

    def check_resources(
        self,
        drone: DroneState,
        outbound_distance_m: float,
        return_distance_m: float,
        cruise_speed_mps: float,
        **resource_estimates,
    ):
        """Assess mission resources and reserve margins."""
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

    # ---------------------------------------------------------
    # Mission execution
    # ---------------------------------------------------------

    def begin_extinguish(
        self,
        drone: DroneState,
        hotspot: Hotspot,
        task: Task,
        cruise_speed_mps: float | None = None,
    ) -> MissionExecutionResult:
        """Validate an assigned extinguishing task and request navigation."""

        if self.autopilot is None or self.payload is None:
            raise RuntimeError(
                "Mission execution requires an autopilot and payload"
            )

        if self._active_task_id is not None and self._terminal_result is None:
            raise RuntimeError("Another mission is already active")

        if drone.kind is not DroneKind.EXTINGUISHER:
            raise ValueError("Only extinguisher drones can execute drop missions")

        if task.type is not TaskType.EXTINGUISH:
            raise ValueError("Task is not an extinguishing task")

        if task.status not in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            raise ValueError("Task is not active")

        if task.assigned_to != drone.drone_id:
            raise ValueError("Task is assigned to another drone")

        if task.epoch != drone.epoch:
            raise ValueError("Task epoch does not match drone epoch")

        if task.target_hotspot_id != hotspot.hotspot_id:
            raise ValueError("Task targets a different hotspot")

        if (
            hotspot.assigned_to is not None
            and hotspot.assigned_to != drone.drone_id
        ):
            raise ValueError("Hotspot is assigned to another drone")

        if hotspot.status not in (
            HotspotStatus.CONFIRMED,
            HotspotStatus.ASSIGNED,
        ):
            raise ValueError("Hotspot has not been confirmed")

        if self.payload.water_remaining_l() <= 0:
            raise ValueError("Extinguisher has no water")

        self._active_task_id = task.task_id
        self._terminal_result = None

        self.autopilot.goto(
            hotspot.position,
            speed_mps=cruise_speed_mps,
        )

        return MissionExecutionResult(
            status="TRANSIT",
            success=False,
            water_used_l=0.0,
            reason="Navigation requested; waiting for arrival",
        )

    def step_extinguish(
        self,
        drone: DroneState,
        hotspot: Hotspot,
        task: Task,
        altitude_agl_m: float,
    ) -> MissionExecutionResult:
        """Advance an extinguishing mission by one simulation step.

        This method never assumes goto() means arrival. It checks at_target()
        and uses the latest autopilot and payload telemetry before releasing.
        """

        if self.autopilot is None or self.payload is None:
            raise RuntimeError(
                "Mission execution requires an autopilot and payload"
            )

        if self._active_task_id != task.task_id:
            raise ValueError("This task has not been started by this controller")

        if self._terminal_result is not None:
            return self._terminal_result

        if not self.autopilot.at_target():
            return MissionExecutionResult(
                status="TRANSIT",
                success=False,
                water_used_l=0.0,
                reason="Drone has not reached the hotspot",
            )

        # Use live telemetry rather than stale values from the original
        # DroneState snapshot.
        current_drone = drone.model_copy(
            update={
                "position": self.autopilot.position(),
                "speed_mps": self.autopilot.speed_mps(),
                "water_remaining_l": self.payload.water_remaining_l(),
            }
        )

        decision = self.check_drop(
            current_drone,
            hotspot,
            task,
            altitude_agl_m,
        )

        if not decision.should_drop:
            self.autopilot.hold()

            self._terminal_result = MissionExecutionResult(
                status="REJECTED",
                success=False,
                water_used_l=0.0,
                reason=decision.reason,
            )
            return self._terminal_result

        actual_l = self.payload.release(
            decision.water_to_release_liters,
            at=self.autopilot.position(),
        )

        if not isfinite(actual_l) or actual_l < 0:
            raise ValueError("Payload returned an invalid release amount")

        if actual_l > decision.water_to_release_liters + 1e-6:
            raise ValueError("Payload released more water than requested")

        self.autopilot.hold()

        if actual_l <= 0:
            self._terminal_result = MissionExecutionResult(
                status="FAILED",
                success=False,
                water_used_l=0.0,
                reason="Payload did not release any water",
            )
            return self._terminal_result

        self._terminal_result = MissionExecutionResult(
            status="DONE",
            success=True,
            water_used_l=actual_l,
            reason="Water release completed",
        )
        return self._terminal_result

"""Coordinates planning, safety and execution for one drone."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Coordinates planning, safety and mission execution for one drone."""

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
    # Planning
    # ---------------------------------------------------------

    def plan_route(
        self,
        occupancy_grid: list[list[int]],
        grid_spec: GridSpec,
        start: Position,
        goal: Position,
    ) -> list[Position] | None:
        """Return an A* route, or None if no route exists."""
        return find_path(
            occupancy_grid,
            grid_spec,
            start,
            goal,
        )

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
        """Assess mission resources and reserve margins."""
        return assess_mission_resources(
            drone=drone,
            outbound_distance_m=outbound_distance_m,
            return_distance_m=return_distance_m,
            cruise_speed_mps=cruise_speed_mps,
            **resource_estimates,
        )

    def evaluate_return_to_base(
        self,
        drone: DroneState,
        outbound_distance_m: float,
        return_distance_m: float,
        cruise_speed_mps: float,
        *,
        battery_use_per_second: float = 0,
        hover_seconds: float = 0,
        hover_battery_use_per_second: float = 0,
        required_drop_liters: float = 0,
        reserve_fraction: float = 0.20,
        fuel_use_kg_per_second: float = 0,
        fuel_reserve_kg: float = 0,
    ):
        """
        Assess whether the drone can complete the mission
        while preserving its resource reserves.
        """
        assessment = self.check_resources(
            drone,
            outbound_distance_m,
            return_distance_m,
            cruise_speed_mps,
            battery_use_per_second=battery_use_per_second,
            hover_seconds=hover_seconds,
            hover_battery_use_per_second=hover_battery_use_per_second,
            required_drop_liters=required_drop_liters,
            reserve_fraction=reserve_fraction,
            fuel_use_kg_per_second=fuel_use_kg_per_second,
            fuel_reserve_kg=fuel_reserve_kg,
        )

        if assessment.should_return and self.autopilot is not None:
            self.autopilot.return_to_launch()

        return assessment

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

    # ---------------------------------------------------------
    # Water-drop safety
    # ---------------------------------------------------------

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
        """
        Start an extinguishing mission.

        Validation failures that indicate an invalid command/assignment
        raise ValueError. A valid mission transitions to TRANSIT.
        """

        if self.autopilot is None or self.payload is None:
            raise RuntimeError(
                "Mission execution requires an autopilot and payload"
            )

        if self._active_task_id is not None:
            raise RuntimeError(
                f"Mission task {self._active_task_id} is already active"
            )

        if self._terminal_result is not None:
            raise RuntimeError("Previous mission has already completed")

        if drone.kind is not DroneKind.EXTINGUISHER:
            raise ValueError("Drone is not an extinguisher")

        if task.type is not TaskType.EXTINGUISH:
            raise ValueError("Task is not an extinguishing task")

        if task.status not in (
            TaskStatus.ASSIGNED,
            TaskStatus.IN_PROGRESS,
        ):
            raise ValueError(
                "Task is not assigned or in progress"
            )

        if task.assigned_to != drone.drone_id:
            raise ValueError(
                "Task is assigned to another drone"
            )

        if task.epoch != drone.epoch:
            raise ValueError(
                "Task epoch does not match drone epoch"
            )

        if task.target_hotspot_id != hotspot.hotspot_id:
            raise ValueError(
                "Task does not target this hotspot"
            )

        if (
            hotspot.assigned_to is not None
            and hotspot.assigned_to != drone.drone_id
        ):
            raise ValueError(
                "Hotspot is assigned to another drone"
            )

        if hotspot.status not in (
            HotspotStatus.CONFIRMED,
            HotspotStatus.ASSIGNED,
        ):
            raise ValueError(
                "Hotspot is not confirmed for extinguishing"
            )

        self._active_task_id = task.task_id

        self.autopilot.goto(
            hotspot.position,
            speed_mps=cruise_speed_mps,
        )

        return MissionExecutionResult(
            status="TRANSIT",
            success=True,
            water_used_l=0.0,
            reason="Navigating to assigned hotspot",
        )

    def step_extinguish(
        self,
        drone: DroneState,
        hotspot: Hotspot,
        task: Task,
        *,
        altitude_agl_m: float,
    ) -> MissionExecutionResult:
        """
        Advance an active extinguishing mission by one step.

        The final water-release decision is made only after arrival and
        all safety gates pass.
        """

        if self.autopilot is None or self.payload is None:
            raise RuntimeError(
                "Mission execution requires an autopilot and payload"
            )

        # Once a mission has completed, return the exact same terminal
        # result and never release payload again.
        if self._terminal_result is not None:
            return self._terminal_result

        if self._active_task_id != task.task_id:
            return MissionExecutionResult(
                status="REJECTED",
                success=False,
                water_used_l=0.0,
                reason="No active mission for this task",
            )

        # Do not release while the drone is still travelling.
        if not self.autopilot.at_target(
            tolerance_m=self.drop_limits.max_distance_m
        ):
            return MissionExecutionResult(
                status="TRANSIT",
                success=True,
                water_used_l=0.0,
                reason="Drone has not reached the hotspot",
            )

        # Final drop gate. This must happen immediately before release.
        decision = self.check_drop(
            drone,
            hotspot,
            task,
            altitude_agl_m,
        )

        if not decision.should_drop:
            self.autopilot.hold()

            result = MissionExecutionResult(
                status="REJECTED",
                success=False,
                water_used_l=0.0,
                reason=decision.reason,
            )

            return result

        released = self.payload.release(
            decision.water_to_release_liters,
            hotspot.position,
        )

        if released is None:
            released_l = decision.water_to_release_liters
        else:
            released_l = float(released)

        if released_l <= 0:
            result = MissionExecutionResult(
                status="FAILED",
                success=False,
                water_used_l=0.0,
                reason="Payload release produced no water",
            )

            self._terminal_result = result
            self._active_task_id = None

            return result

        result = MissionExecutionResult(
            status="DONE",
            success=True,
            water_used_l=released_l,
            reason="Water released successfully",
        )

        self._terminal_result = result
        self._active_task_id = None

        return result

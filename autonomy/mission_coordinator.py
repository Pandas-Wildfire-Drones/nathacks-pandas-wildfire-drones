"""Coordinate task assignment, resources, CBS routing and path claims."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

from autonomy.cbs_planner import PathRequest, path_duration_s, plan_paths
from autonomy.deconfliction import detect_conflicts
from autonomy.path_claim_manager import PathClaimManager
from autonomy.path_planning import find_path
from autonomy.resource_manager import ResourceAssessment, assess_mission_resources
from contracts.config import PlatformSpec
from contracts.entities import DroneState, Hotspot, Task
from contracts.enums import HotspotStatus, Powertrain, TaskStatus, TaskType
from contracts.geometry import GridSpec, Position
from contracts.messages import PathClaim, SwarmMessage


@dataclass(frozen=True)
class MissionPlan:
    """Locally validated plan for one assigned mission."""

    task_id: str
    drone_id: str
    waypoints: list[Position]
    duration_s: float
    time_step_s: float
    water_required_l: float


@dataclass(frozen=True)
class MissionPlanningResult:
    """Result of local mission planning."""

    accepted: bool
    reason: str
    plan: MissionPlan | None = None
    resource_assessment: ResourceAssessment | None = None


@dataclass(frozen=True)
class MissionPlanningContext:
    """Runtime map and aircraft data required to plan an assigned mission."""

    occupancy_grid: list[list[int]]
    grid_spec: GridSpec
    base_position: Position
    platform_spec: PlatformSpec
    minimum_separation_m: float
    time_step_s: float
    altitude_band_m: tuple[float, float] | None = None


class MissionCoordinator:
    """Connect task assignments to resources, CBS and path claims."""

    def __init__(
        self,
        drone: DroneState,
        path_claim_manager: PathClaimManager,
    ) -> None:
        self.drone = drone
        self.path_claim_manager = path_claim_manager

    def check_resources_for_task(
        self,
        task: Task,
        *,
        outbound_distance_m: float,
        return_distance_m: float,
        cruise_speed_mps: float,
        battery_use_per_second: float = 0.0,
        hover_seconds: float = 0.0,
        hover_battery_use_per_second: float = 0.0,
        fuel_use_kg_per_second: float = 0.0,
        fuel_reserve_kg: float = 0.0,
        reserve_fraction: float = 0.20,
    ):
        """Check whether this drone can safely perform the assigned task."""

        required_water = (
            task.volume_l
            if task.volume_l is not None
            else (self.drone.water_remaining_l or 0.0)
        )

        return assess_mission_resources(
            drone=self.drone,
            outbound_distance_m=outbound_distance_m,
            return_distance_m=return_distance_m,
            cruise_speed_mps=cruise_speed_mps,
            battery_use_per_second=battery_use_per_second,
            hover_seconds=hover_seconds,
            hover_battery_use_per_second=hover_battery_use_per_second,
            fuel_use_kg_per_second=fuel_use_kg_per_second,
            fuel_reserve_kg=fuel_reserve_kg,
            required_drop_liters=required_water,
            reserve_fraction=reserve_fraction,
        )

    def plan_task(
        self,
        task: Task,
        hotspot: Hotspot,
        context: MissionPlanningContext,
        *,
        start_time: float | None = None,
    ) -> MissionPlanningResult:
        """Check resources before CBS, then recheck against the planned route."""

        if task.type is not TaskType.EXTINGUISH:
            return MissionPlanningResult(
                accepted=False,
                reason="Task is not an extinguishing task",
            )

        if task.assigned_to != self.drone.drone_id:
            return MissionPlanningResult(
                accepted=False,
                reason="Task is assigned to another drone",
            )

        if task.epoch != self.drone.epoch:
            return MissionPlanningResult(
                accepted=False,
                reason="Task epoch does not match drone epoch",
            )

        if task.status not in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            return MissionPlanningResult(
                accepted=False,
                reason="Task is not assigned or in progress",
            )

        if task.target_hotspot_id != hotspot.hotspot_id:
            return MissionPlanningResult(
                accepted=False,
                reason="Task does not target this hotspot",
            )

        if (
            hotspot.assigned_to is not None
            and hotspot.assigned_to != self.drone.drone_id
        ):
            return MissionPlanningResult(
                accepted=False,
                reason="Hotspot is assigned to another drone",
            )

        if hotspot.status not in (
            HotspotStatus.CONFIRMED,
            HotspotStatus.ASSIGNED,
        ):
            return MissionPlanningResult(
                accepted=False,
                reason="Hotspot is not confirmed for extinguishing",
            )

        if (
            context.platform_spec.kind is not self.drone.kind
            or context.platform_spec.powertrain is not self.drone.powertrain
        ):
            return MissionPlanningResult(
                accepted=False,
                reason="Aircraft model does not match drone kind/powertrain",
            )

        cruise_speed_mps = context.platform_spec.cruise_speed_mps
        outbound_distance_m = self.drone.position.distance_to(
            hotspot.position,
        )
        return_distance_m = hotspot.position.distance_to(
            context.base_position,
        )

        try:
            resource_estimates = self._resource_estimates(
                context.platform_spec,
            )
            preliminary = self.check_resources_for_task(
                task,
                outbound_distance_m=outbound_distance_m,
                return_distance_m=return_distance_m,
                cruise_speed_mps=cruise_speed_mps,
                hover_seconds=context.platform_spec.drop_release_s,
                **resource_estimates,
            )
        except ValueError as exc:
            return MissionPlanningResult(
                accepted=False,
                reason=f"Resource check failed: {exc}",
            )

        if not preliminary.can_complete_mission:
            return MissionPlanningResult(
                accepted=False,
                reason="Resource check failed: "
                + "; ".join(preliminary.reasons),
                resource_assessment=preliminary,
            )

        try:
            planned = self.plan_single_task(
                task,
                hotspot.position,
                context.occupancy_grid,
                context.grid_spec,
                minimum_separation_m=context.minimum_separation_m,
                time_step_s=context.time_step_s,
                cruise_speed_mps=cruise_speed_mps,
                altitude_band_m=context.altitude_band_m,
                start_time=start_time,
            )
        except ValueError as exc:
            return MissionPlanningResult(
                accepted=False,
                reason=f"CBS planning failed: {exc}",
                resource_assessment=preliminary,
            )

        if not planned.accepted or planned.plan is None:
            return MissionPlanningResult(
                accepted=False,
                reason=planned.reason,
                resource_assessment=preliminary,
            )

        planned_waypoints = list(planned.plan.waypoints)
        if planned_waypoints[-1] != hotspot.position:
            planned_waypoints.append(hotspot.position)
            planned = MissionPlanningResult(
                accepted=True,
                reason=planned.reason,
                plan=replace(
                    planned.plan,
                    waypoints=planned_waypoints,
                    duration_s=(
                        planned.plan.duration_s + context.time_step_s
                    ),
                ),
                resource_assessment=planned.resource_assessment,
            )

        route_distance_m = sum(
            first.distance_to(second)
            for first, second in zip(
                planned.plan.waypoints,
                planned.plan.waypoints[1:],
            )
        )
        return_route = find_path(
            context.occupancy_grid,
            context.grid_spec,
            hotspot.position,
            context.base_position,
        )
        if return_route is None:
            return MissionPlanningResult(
                accepted=False,
                reason="No safe route back to base is available",
                resource_assessment=preliminary,
            )
        return_distance_m = sum(
            first.distance_to(second)
            for first, second in zip(
                return_route,
                return_route[1:],
            )
        )
        exact_assessment = self.check_resources_for_task(
            task,
            outbound_distance_m=route_distance_m,
            return_distance_m=return_distance_m,
            cruise_speed_mps=cruise_speed_mps,
            hover_seconds=context.platform_spec.drop_release_s,
            **resource_estimates,
        )

        if not exact_assessment.can_complete_mission:
            return MissionPlanningResult(
                accepted=False,
                reason="Resource check failed for planned route: "
                + "; ".join(exact_assessment.reasons),
                resource_assessment=exact_assessment,
            )

        return MissionPlanningResult(
            accepted=True,
            reason=planned.reason,
            plan=planned.plan,
            resource_assessment=exact_assessment,
        )

    def _resource_estimates(
        self,
        platform_spec: PlatformSpec,
    ) -> dict[str, float]:
        if self.drone.powertrain is Powertrain.ELECTRIC:
            if (
                platform_spec.cruise_power_w is None
                or platform_spec.battery_wh is None
                or platform_spec.hover_power_w is None
            ):
                raise ValueError(
                    "Electric aircraft model is missing battery estimates"
                )
            return {
                "battery_use_per_second": (
                    platform_spec.cruise_power_w
                    / (platform_spec.battery_wh * 3600.0)
                ),
                "hover_battery_use_per_second": (
                    platform_spec.hover_power_w
                    / (platform_spec.battery_wh * 3600.0)
                ),
                "reserve_fraction": 1.0 - platform_spec.battery_usable_frac,
            }

        if (
            platform_spec.fuel_burn_kg_h_at_mtow is None
            or platform_spec.fuel_capacity_kg is None
        ):
            raise ValueError(
                "Hybrid aircraft model is missing fuel estimates"
            )

        return {
            # MTOW burn is conservative when exact sortie mass is unavailable.
            "fuel_use_kg_per_second": (
                platform_spec.fuel_burn_kg_h_at_mtow / 3600.0
            ),
            "fuel_reserve_kg": (
                platform_spec.fuel_capacity_kg
                * platform_spec.fuel_reserve_frac
            ),
        }

    def plan_single_task(
        self,
        task: Task,
        goal: Position,
        occupancy_grid: list[list[int]],
        grid_spec: GridSpec,
        *,
        minimum_separation_m: float,
        time_step_s: float,
        cruise_speed_mps: float,
        altitude_band_m: tuple[float, float] | None = None,
        start_time: float | None = None,
    ) -> MissionPlanningResult:
        """Create a CBS route for this drone's assigned task."""

        if task.type is not TaskType.EXTINGUISH:
            return MissionPlanningResult(
                accepted=False,
                reason="Task is not an extinguishing task",
            )

        if task.status not in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            return MissionPlanningResult(
                accepted=False,
                reason="Task is not assigned or in progress",
            )

        if task.assigned_to != self.drone.drone_id:
            return MissionPlanningResult(
                accepted=False,
                reason="Task is assigned to another drone",
            )

        if task.epoch != self.drone.epoch:
            return MissionPlanningResult(
                accepted=False,
                reason="Task epoch does not match drone epoch",
            )

        request = PathRequest(
            drone_id=self.drone.drone_id,
            start=self.drone.position,
            goal=goal,
            speed_mps=cruise_speed_mps,
            start_time=(
                self.drone.timestamp if start_time is None else start_time
            ),
            altitude_band_m=altitude_band_m,
        )

        paths = plan_paths(
            [request],
            occupancy_grid,
            grid_spec,
            minimum_separation_m=minimum_separation_m,
            time_step_s=time_step_s,
        )

        if paths is None or self.drone.drone_id not in paths:
            return MissionPlanningResult(
                accepted=False,
                reason="CBS could not produce a safe route",
            )

        waypoints = paths[self.drone.drone_id]

        duration_s = path_duration_s(
            waypoints,
            time_step_s=time_step_s,
        )

        plan = MissionPlan(
            task_id=task.task_id,
            drone_id=self.drone.drone_id,
            waypoints=waypoints,
            duration_s=duration_s,
            time_step_s=time_step_s,
            water_required_l=(
                task.volume_l
                if task.volume_l is not None
                else (self.drone.water_remaining_l or 0.0)
            ),
        )

        return MissionPlanningResult(
            accepted=True,
            reason="CBS route planned successfully",
            plan=plan,
        )

    def publish_plan(
        self,
        plan: MissionPlan,
        *,
        other_claims: list[PathClaim],
        minimum_separation_m: float,
        valid_from: float | None = None,
        altitude_band_m: tuple[float, float] | None = None,
    ) -> SwarmMessage | None:
        """Publish only a route that is clear of overlapping remote claims."""

        if plan.drone_id != self.drone.drone_id:
            return None

        if altitude_band_m is not None and any(
            not altitude_band_m[0] <= waypoint.z <= altitude_band_m[1]
            for waypoint in plan.waypoints
        ):
            raise ValueError(
                "Planned route leaves the assigned altitude band"
            )

        route_start = (
            self.drone.timestamp if valid_from is None else valid_from
        )
        route_end = route_start + max(
            plan.time_step_s,
            plan.duration_s,
        )

        for claim in other_claims:
            overlap_start = max(route_start, claim.valid_from)
            overlap_end = min(route_end, claim.valid_until)
            if overlap_start > overlap_end:
                continue

            if (
                altitude_band_m is not None
                and claim.altitude_band_m is not None
                and (
                    altitude_band_m[0] >= claim.altitude_band_m[1]
                    or claim.altitude_band_m[0] >= altitude_band_m[1]
                )
            ):
                continue

            claim_time_step = (
                (claim.valid_until - claim.valid_from)
                / max(1, len(claim.waypoints) - 1)
            )
            sample_step = min(plan.time_step_s, claim_time_step)
            sample_count = max(
                1,
                ceil((overlap_end - overlap_start) / sample_step),
            )
            times = [
                overlap_start
                + (overlap_end - overlap_start) * index / sample_count
                for index in range(sample_count + 1)
            ]
            proposed_samples = [
                self._position_on_path(
                    plan.waypoints,
                    route_start,
                    route_end,
                    timestamp,
                )
                for timestamp in times
            ]
            claimed_samples = [
                self._position_on_path(
                    claim.waypoints,
                    claim.valid_from,
                    claim.valid_until,
                    timestamp,
                )
                for timestamp in times
            ]
            if detect_conflicts(
                proposed_samples,
                [claimed_samples],
                minimum_separation_m,
            ):
                return None

        return self.path_claim_manager.publish_from_planned_path(
            self.drone,
            plan.waypoints,
            valid_from=route_start,
            time_step_s=plan.time_step_s,
            altitude_band_m=altitude_band_m,
        )

    @staticmethod
    def _position_on_path(
        waypoints: list[Position],
        valid_from: float,
        valid_until: float,
        timestamp: float,
    ) -> Position:
        if len(waypoints) == 1:
            return waypoints[0]

        interval = (valid_until - valid_from) / (len(waypoints) - 1)
        path_offset = min(
            max(0.0, timestamp - valid_from),
            valid_until - valid_from,
        )
        segment = min(
            int(path_offset / interval),
            len(waypoints) - 2,
        )
        fraction = (path_offset - segment * interval) / interval
        start = waypoints[segment]
        end = waypoints[segment + 1]

        return Position(
            x=start.x + (end.x - start.x) * fraction,
            y=start.y + (end.y - start.y) * fraction,
            z=start.z + (end.z - start.z) * fraction,
        )

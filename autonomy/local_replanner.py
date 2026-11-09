"""Local multi-drone replanning orchestration."""

from __future__ import annotations

from autonomy.cbs_planner import PathRequest, plan_paths
from autonomy.conflict_groups import (
    PlannedDroneRoute,
    build_conflict_groups,
)
from contracts.geometry import GridSpec, Position


def replan_conflict_groups(
    requests: list[PathRequest],
    existing_routes: dict[str, list[Position]],
    occupancy_grid: list[list[int]],
    grid_spec: GridSpec,
    *,
    minimum_separation_m: float,
    conflict_group_threshold_m: float,
    time_step_s: float = 0.1,
    max_cbs_nodes: int = 100,
    max_timesteps: int = 500,
) -> dict[str, list[Position]] | None:
    """Replan only connected groups of potentially conflicting drones."""

    request_by_id = {
        request.drone_id: request
        for request in requests
    }

    routes = []

    for request in requests:
        current_route = existing_routes.get(request.drone_id)

        if current_route is None:
            return None

        routes.append(
            PlannedDroneRoute(
                drone_id=request.drone_id,
                waypoints=tuple(current_route),
                altitude_band_m=request.altitude_band_m,
            )
        )

    groups = build_conflict_groups(
        routes,
        horizontal_threshold_m=conflict_group_threshold_m,
    )

    result = dict(existing_routes)

    for group in groups:
        if len(group) <= 1:
            continue

        group_requests = [
            request_by_id[drone_id]
            for drone_id in sorted(group)
        ]

        group_result = plan_paths(
            group_requests,
            occupancy_grid,
            grid_spec,
            minimum_separation_m=minimum_separation_m,
            time_step_s=time_step_s,
            max_cbs_nodes=max_cbs_nodes,
            max_timesteps=max_timesteps,
        )

        if group_result is None:
            return None

        result.update(group_result)

    return result

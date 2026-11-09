"""Time-aware multi-drone path planning using Conflict-Based Search."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import isfinite, sqrt

from contracts.geometry import GridIndex, GridSpec, Position


@dataclass(frozen=True)
class PathRequest:
    """Planning request for one drone.

    The altitude band is assumed to have been assigned by a higher-level
    lane/airspace manager. This planner chooses the midpoint of that band
    as its nominal planning altitude.
    """

    drone_id: str
    start: Position
    goal: Position
    speed_mps: float
    start_time: float = 0.0
    altitude_band_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.drone_id:
            raise ValueError("drone_id must not be empty")

        if not isfinite(self.speed_mps) or self.speed_mps <= 0:
            raise ValueError("speed_mps must be finite and greater than zero")

        if not isfinite(self.start_time):
            raise ValueError("start_time must be finite")

        if self.altitude_band_m is not None:
            lower, upper = self.altitude_band_m

            if not isfinite(lower) or not isfinite(upper):
                raise ValueError("altitude band must be finite")

            if lower >= upper:
                raise ValueError(
                    "altitude band lower bound must be less than upper bound"
                )


@dataclass(frozen=True)
class CBSConflict:
    """A conflict between two planned agents."""

    drone_a: str
    drone_b: str
    timestep: int
    distance_m: float
    kind: str


@dataclass(frozen=True)
class _Constraint:
    """A vertex or edge constraint for one drone."""

    drone_id: str
    timestep: int
    cell: GridIndex
    previous_cell: GridIndex | None = None


@dataclass(frozen=True)
class _Node:
    """One node in the high-level CBS search tree."""

    constraints: tuple[_Constraint, ...]
    paths: tuple[tuple[str, tuple[Position, ...]], ...]


def _heuristic(a: GridIndex, b: GridIndex) -> float:
    row_diff = abs(a.row - b.row)
    col_diff = abs(a.col - b.col)

    diagonal = min(row_diff, col_diff)
    straight = max(row_diff, col_diff) - diagonal

    return straight + diagonal * sqrt(2.0)


def _neighbors(
    current: GridIndex,
    occupancy_grid: list[list[int]],
    grid_spec: GridSpec,
) -> list[GridIndex]:
    """Return legal 8-connected spatial neighbors."""

    result: list[GridIndex] = []

    for row_step, col_step in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ):
        row = current.row + row_step
        col = current.col + col_step

        if not (0 <= row < grid_spec.rows and 0 <= col < grid_spec.cols):
            continue

        if occupancy_grid[row][col] != 0:
            continue

        # Prevent diagonal corner cutting.
        if row_step != 0 and col_step != 0:
            if occupancy_grid[current.row][col] != 0:
                continue

            if occupancy_grid[row][current.col] != 0:
                continue

        result.append(GridIndex(row=row, col=col))

    return result


def _step_distance_m(
    current: GridIndex,
    neighbor: GridIndex,
    grid_spec: GridSpec,
) -> float:
    diagonal = (
        current.row != neighbor.row
        and current.col != neighbor.col
    )

    return grid_spec.resolution_m * (sqrt(2.0) if diagonal else 1.0)


def _altitude_for(request: PathRequest, grid_spec: GridSpec) -> float:
    """Return the planner's nominal flight altitude."""

    if request.altitude_band_m is None:
        # The grid origin z is scenario-datum height. We deliberately do
        # not reinterpret it as AGL.
        return grid_spec.origin.z

    lower, upper = request.altitude_band_m
    return (lower + upper) / 2.0


def _position_for(
    request: PathRequest,
    cell: GridIndex,
    grid_spec: GridSpec,
) -> Position:
    """Convert a grid cell to a position at the assigned altitude."""

    world = grid_spec.cell_to_world(cell)
    altitude = _altitude_for(request, grid_spec)

    return Position(
        x=world.x,
        y=world.y,
        z=altitude,
    )


def _constraint_sets(
    constraints: tuple[_Constraint, ...],
    drone_id: str,
) -> tuple[
    set[tuple[GridIndex, int]],
    set[tuple[GridIndex, GridIndex, int]],
]:
    """Return vertex and edge constraints for one drone."""

    vertex_constraints: set[tuple[GridIndex, int]] = set()
    edge_constraints: set[tuple[GridIndex, GridIndex, int]] = set()

    for constraint in constraints:
        if constraint.drone_id != drone_id:
            continue

        if constraint.previous_cell is None:
            vertex_constraints.add(
                (constraint.cell, constraint.timestep)
            )
        else:
            edge_constraints.add(
                (
                    constraint.previous_cell,
                    constraint.cell,
                    constraint.timestep,
                )
            )

    return vertex_constraints, edge_constraints


def _reconstruct(
    came_from: dict[
        tuple[GridIndex, int],
        tuple[GridIndex, int],
    ],
    state: tuple[GridIndex, int],
) -> list[tuple[GridIndex, int]]:
    path = [state]

    while state in came_from:
        state = came_from[state]
        path.append(state)

    path.reverse()
    return path


def _low_level_plan(
    request: PathRequest,
    occupancy_grid: list[list[int]],
    grid_spec: GridSpec,
    constraints: tuple[_Constraint, ...],
    time_step_s: float,
    max_timesteps: int,
) -> tuple[Position, ...] | None:
    """Run time-expanded A* for one drone."""

    start_cell = grid_spec.world_to_cell(request.start)
    goal_cell = grid_spec.world_to_cell(request.goal)

    if start_cell is None:
        raise ValueError(
            f"Start position for {request.drone_id} is outside the grid"
        )

    if goal_cell is None:
        raise ValueError(
            f"Goal position for {request.drone_id} is outside the grid"
        )

    if occupancy_grid[start_cell.row][start_cell.col] != 0:
        raise ValueError(
            f"Start position for {request.drone_id} is blocked"
        )

    if occupancy_grid[goal_cell.row][goal_cell.col] != 0:
        raise ValueError(
            f"Goal position for {request.drone_id} is blocked"
        )

    vertex_constraints, edge_constraints = _constraint_sets(
        constraints,
        request.drone_id,
    )

    start_state = (start_cell, 0)

    if start_state in vertex_constraints:
        return None

    frontier = []
    tie_breaker = count()

    heappush(
        frontier,
        (
            _heuristic(start_cell, goal_cell),
            next(tie_breaker),
            start_state,
        ),
    )

    came_from: dict[
        tuple[GridIndex, int],
        tuple[GridIndex, int],
    ] = {}

    costs = {start_state: 0.0}

    while frontier:
        _, _, state = heappop(frontier)

        cell, timestep = state

        if cell == goal_cell:
            timed_cells = _reconstruct(came_from, state)

            return tuple(
                _position_for(request, path_cell, grid_spec)
                for path_cell, _ in timed_cells
            )

        if timestep >= max_timesteps:
            continue

        next_timestep = timestep + 1

        candidates = _neighbors(
            cell,
            occupancy_grid,
            grid_spec,
        )

        # Waiting is a real time-expanded action. It is not "free":
        # the drone remains physically at this cell for one full
        # planning interval, and the conflict checker still evaluates
        # that occupied position.
        candidates.append(cell)

        for neighbor in candidates:
            if (neighbor, next_timestep) in vertex_constraints:
                continue

            if (
                cell,
                neighbor,
                next_timestep,
            ) in edge_constraints:
                continue

            distance_m = _step_distance_m(
                cell,
                neighbor,
                grid_spec,
            )

            travel_time_s = distance_m / request.speed_mps

            # A single grid transition must fit inside one synchronized
            # planning interval.
            if travel_time_s > time_step_s:
                continue

            next_state = (neighbor, next_timestep)

            movement_cost = (
                travel_time_s
                if neighbor != cell
                else time_step_s
            )

            new_cost = costs[state] + movement_cost

            if new_cost >= costs.get(next_state, float("inf")):
                continue

            costs[next_state] = new_cost
            came_from[next_state] = state

            remaining_distance = (
                _heuristic(neighbor, goal_cell)
                * grid_spec.resolution_m
            )

            heuristic_time = (
                remaining_distance / request.speed_mps
            )

            priority = new_cost + heuristic_time

            heappush(
                frontier,
                (
                    priority,
                    next(tie_breaker),
                    next_state,
                ),
            )

    return None


def _distance_at(
    path: tuple[Position, ...],
    timestep: int,
) -> Position:
    """Hold the drone at its final position after arrival."""

    return path[min(timestep, len(path) - 1)]


def _paths_have_altitude_compatibility(
    point_a: Position,
    point_b: Position,
) -> bool:
    """Return whether the vertical separation is relevant to collision."""

    return True


def _detect_conflict(
    paths: dict[str, tuple[Position, ...]],
    minimum_separation_m: float,
) -> CBSConflict | None:
    """Return the earliest synchronized vertex or edge conflict."""

    drone_ids = sorted(paths)

    for index, drone_a in enumerate(drone_ids):
        for drone_b in drone_ids[index + 1:]:
            path_a = paths[drone_a]
            path_b = paths[drone_b]

            shared_steps = max(
                len(path_a),
                len(path_b),
            )

            for timestep in range(shared_steps):
                point_a = _distance_at(path_a, timestep)
                point_b = _distance_at(path_b, timestep)

                if not _paths_have_altitude_compatibility(
                    point_a,
                    point_b,
                ):
                    continue

                distance_m = point_a.distance_to(point_b)

                if distance_m < minimum_separation_m:
                    return CBSConflict(
                        drone_a=drone_a,
                        drone_b=drone_b,
                        timestep=timestep,
                        distance_m=distance_m,
                        kind="vertex",
                    )

            # Detect direct edge swaps:
            #
            # A: cell 1 -> cell 2
            # B: cell 2 -> cell 1
            #
            # They can avoid sharing a discrete vertex while still
            # occupying the same corridor at the same time.
            shared_edges = min(
                len(path_a),
                len(path_b),
            ) - 1

            for timestep in range(shared_edges):
                a_from = path_a[timestep]
                a_to = path_a[timestep + 1]
                b_from = path_b[timestep]
                b_to = path_b[timestep + 1]

                if (
                    a_from.x == b_to.x
                    and a_from.y == b_to.y
                    and a_from.z == b_to.z
                    and a_to.x == b_from.x
                    and a_to.y == b_from.y
                    and a_to.z == b_from.z
                ):
                    return CBSConflict(
                        drone_a=drone_a,
                        drone_b=drone_b,
                        timestep=timestep + 1,
                        distance_m=0.0,
                        kind="edge",
                    )

    return None


def _count_conflicts(
    paths: dict[str, tuple[Position, ...]],
    minimum_separation_m: float,
) -> int:
    """Count vertex conflicts for CBS node prioritization."""

    ids = sorted(paths)
    total = 0

    for index, drone_a in enumerate(ids):
        for drone_b in ids[index + 1:]:
            path_a = paths[drone_a]
            path_b = paths[drone_b]

            shared_steps = max(
                len(path_a),
                len(path_b),
            )

            for timestep in range(shared_steps):
                point_a = _distance_at(path_a, timestep)
                point_b = _distance_at(path_b, timestep)

                if (
                    point_a.distance_to(point_b)
                    < minimum_separation_m
                ):
                    total += 1

    return total


def _paths_tuple(
    paths: dict[str, tuple[Position, ...]],
) -> tuple[tuple[str, tuple[Position, ...]], ...]:
    return tuple(
        (drone_id, paths[drone_id])
        for drone_id in sorted(paths)
    )


def _paths_dict(
    paths: tuple[tuple[str, tuple[Position, ...]], ...],
) -> dict[str, tuple[Position, ...]]:
    return dict(paths)


def plan_paths(
    requests: list[PathRequest],
    occupancy_grid: list[list[int]],
    grid_spec: GridSpec,
    *,
    minimum_separation_m: float,
    time_step_s: float = 0.1,
    max_cbs_nodes: int = 100,
    max_timesteps: int = 500,
) -> dict[str, list[Position]] | None:
    """Plan synchronized conflict-free paths using bounded CBS.

    `time_step_s` is the shared planning clock. Every path index
    represents one planning interval.

    `start_time` is preserved as mission-time metadata by the request;
    the returned path itself is relative to timestep zero. The caller
    can publish the corresponding PathClaim using:

        valid_from = request.start_time
        valid_until = request.start_time + path_duration

    Returns None when the bounded CBS search cannot find a solution.
    """

    if not requests:
        return {}

    if len({
        request.drone_id
        for request in requests
    }) != len(requests):
        raise ValueError("Drone IDs must be unique")

    if (
        not isfinite(minimum_separation_m)
        or minimum_separation_m < 0
    ):
        raise ValueError(
            "minimum_separation_m must be finite and non-negative"
        )

    if (
        not isfinite(time_step_s)
        or time_step_s <= 0
    ):
        raise ValueError(
            "time_step_s must be finite and greater than zero"
        )

    if max_cbs_nodes <= 0:
        raise ValueError(
            "max_cbs_nodes must be greater than zero"
        )

    if max_timesteps <= 0:
        raise ValueError(
            "max_timesteps must be greater than zero"
        )

    if len(occupancy_grid) != grid_spec.rows:
        raise ValueError("Occupancy grid row count does not match GridSpec")

    if any(
        len(row) != grid_spec.cols
        for row in occupancy_grid
    ):
        raise ValueError(
            "Occupancy grid column count does not match GridSpec"
        )

    initial_paths: dict[str, tuple[Position, ...]] = {}

    for request in requests:
        path = _low_level_plan(
            request=request,
            occupancy_grid=occupancy_grid,
            grid_spec=grid_spec,
            constraints=(),
            time_step_s=time_step_s,
            max_timesteps=max_timesteps,
        )

        if path is None:
            return None

        initial_paths[request.drone_id] = path

    root = _Node(
        constraints=(),
        paths=_paths_tuple(initial_paths),
    )

    frontier = []
    tie_breaker = count()

    heappush(
        frontier,
        (
            _count_conflicts(
                initial_paths,
                minimum_separation_m,
            ),
            next(tie_breaker),
            root,
        ),
    )

    request_by_id = {
        request.drone_id: request
        for request in requests
    }

    expanded = 0

    while frontier and expanded < max_cbs_nodes:
        _, _, node = heappop(frontier)
        expanded += 1

        paths = _paths_dict(node.paths)

        conflict = _detect_conflict(
            paths,
            minimum_separation_m,
        )

        if conflict is None:
            return {
                drone_id: list(path)
                for drone_id, path in node.paths
            }

        for constrained_drone in (
            conflict.drone_a,
            conflict.drone_b,
        ):
            request = request_by_id[constrained_drone]

            path = paths[constrained_drone]

            current_index = min(
                conflict.timestep,
                len(path) - 1,
            )

            current_cell = grid_spec.world_to_cell(
                path[current_index]
            )

            if current_cell is None:
                continue

            previous_cell = None

            if (
                conflict.kind == "edge"
                and current_index > 0
            ):
                previous_cell = grid_spec.world_to_cell(
                    path[current_index - 1]
                )

            constraint = _Constraint(
                drone_id=constrained_drone,
                timestep=conflict.timestep,
                cell=current_cell,
                previous_cell=previous_cell,
            )

            if constraint in node.constraints:
                continue

            new_constraints = (
                node.constraints + (constraint,)
            )

            new_path = _low_level_plan(
                request=request,
                occupancy_grid=occupancy_grid,
                grid_spec=grid_spec,
                constraints=new_constraints,
                time_step_s=time_step_s,
                max_timesteps=max_timesteps,
            )

            if new_path is None:
                continue

            new_paths = dict(paths)
            new_paths[constrained_drone] = new_path

            child = _Node(
                constraints=new_constraints,
                paths=_paths_tuple(new_paths),
            )

            heappush(
                frontier,
                (
                    _count_conflicts(
                        new_paths,
                        minimum_separation_m,
                    ),
                    next(tie_breaker),
                    child,
                ),
            )

    return None


def path_duration_s(
    path: list[Position] | tuple[Position, ...],
    *,
    time_step_s: float,
) -> float:
    """Return the synchronized duration represented by a planned path."""

    if not path:
        return 0.0

    if not isfinite(time_step_s) or time_step_s <= 0:
        raise ValueError(
            "time_step_s must be finite and greater than zero"
        )

    return (len(path) - 1) * time_step_s

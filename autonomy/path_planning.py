"""A* pathfinding using the team's shared grid and geometry contracts."""

from heapq import heappop, heappush
from itertools import count
from math import sqrt

from contracts.geometry import GridIndex, GridSpec, Position


def _heuristic(a: GridIndex, b: GridIndex) -> float:
    """Octile distance for movement in eight directions."""
    row_diff = abs(a.row - b.row)
    col_diff = abs(a.col - b.col)
    diagonal = min(row_diff, col_diff)
    straight = max(row_diff, col_diff) - diagonal
    return straight + diagonal * sqrt(2)


def _get_neighbors(
    current: GridIndex,
    occupancy_grid: list[list[int]],
    grid_spec: GridSpec,
) -> list[GridIndex]:
    """Return free neighboring cells without cutting obstacle corners."""
    neighbors = []

    for row_step, col_step in (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ):
        row = current.row + row_step
        col = current.col + col_step

        if not (0 <= row < grid_spec.rows and 0 <= col < grid_spec.cols):
            continue

        if occupancy_grid[row][col] != 0:
            continue

        # A diagonal move is allowed only if both adjacent
        # orthogonal cells are free.
        if row_step != 0 and col_step != 0:
            if occupancy_grid[current.row][col] != 0:
                continue
            if occupancy_grid[row][current.col] != 0:
                continue

        neighbors.append(GridIndex(row=row, col=col))

    return neighbors


def find_path(
    occupancy_grid: list[list[int]],
    grid_spec: GridSpec,
    start: Position,
    goal: Position,
) -> list[Position] | None:
    """
    Find a shortest route through a 2D occupancy grid using A*.

    Occupancy convention:
        0 = free cell
        non-zero = blocked cell

    Inputs:
        occupancy_grid: rows indexed as [row][col]
        grid_spec: shared physical grid definition
        start: starting world-coordinate Position
        goal: destination world-coordinate Position

    Returns:
        A list of world-coordinate Positions, including start and goal,
        or None if no route exists.

    Note:
        Returned positions are at cell centers and use the grid origin's z.
    """
    if len(occupancy_grid) != grid_spec.rows:
        raise ValueError("Occupancy grid row count does not match GridSpec")

    if any(len(row) != grid_spec.cols for row in occupancy_grid):
        raise ValueError("Occupancy grid column count does not match GridSpec")

    start_cell = grid_spec.world_to_cell(start)
    goal_cell = grid_spec.world_to_cell(goal)

    if start_cell is None:
        raise ValueError("Start position is outside the grid")

    if goal_cell is None:
        raise ValueError("Goal position is outside the grid")

    if occupancy_grid[start_cell.row][start_cell.col] != 0:
        raise ValueError("Start position is in a blocked cell")

    if occupancy_grid[goal_cell.row][goal_cell.col] != 0:
        raise ValueError("Goal position is in a blocked cell")

    frontier = []
    tie_breaker = count()

    heappush(
        frontier,
        (_heuristic(start_cell, goal_cell), next(tie_breaker), start_cell),
    )

    came_from = {}
    costs = {start_cell: 0.0}

    while frontier:
        _, _, current = heappop(frontier)

        if current == goal_cell:
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            path.reverse()
            return [grid_spec.cell_to_world(cell) for cell in path]

        for neighbor in _get_neighbors(current, occupancy_grid, grid_spec):
            diagonal = (
                neighbor.row != current.row
                and neighbor.col != current.col
            )
            step_cost = sqrt(2) if diagonal else 1.0
            new_cost = costs[current] + step_cost

            if new_cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = new_cost
                came_from[neighbor] = current

                priority = new_cost + _heuristic(neighbor, goal_cell)

                heappush(
                    frontier,
                    (priority, next(tie_breaker), neighbor),
                )

    return None
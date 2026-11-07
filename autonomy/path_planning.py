"""A* pathfinding algorithm for 2D occupancy grids.S"""

from heapq import heappush, heappop
from math import sqrt


def heuristic(a, b):
    """Octile distance: an admissible heuristic for 8-way movement."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)


def get_neighbors(position, grid):
    x, y = position
    height = len(grid)
    width = len(grid[0])
    neighbors = []

    for dx, dy in [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]:
        nx, ny = x + dx, y + dy

        if not (0 <= nx < width and 0 <= ny < height):
            continue
        if grid[ny][nx] != 0:
            continue

        # Prevent cutting diagonally through obstacle corners.
        if dx != 0 and dy != 0:
            if grid[y][nx] != 0 or grid[ny][x] != 0:
                continue

        neighbors.append((nx, ny))

    return neighbors


def find_path(grid, start, goal):
    """
    A* pathfinding on a 2D occupancy grid.
    0 = free cell; 1 = blocked cell.
    Coordinates are grid indices (x, y), not metres.
    Returns a list of cells, or None if no route exists.
    """
    if not grid or not grid[0]:
        raise ValueError("Grid cannot be empty")

    width = len(grid[0])
    height = len(grid)

    if any(len(row) != width for row in grid):
        raise ValueError("Grid rows must have equal lengths")

    for name, (x, y) in [("start", start), ("goal", goal)]:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{name} is outside the grid")
        if grid[y][x] != 0:
            raise ValueError(f"{name} is blocked")

    frontier = []
    heappush(frontier, (heuristic(start, goal), start))
    came_from = {}
    costs = {start: 0}

    while frontier:
        _, current = heappop(frontier)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor in get_neighbors(current, grid):
            diagonal = (
                neighbor[0] != current[0]
                and neighbor[1] != current[1]
            )
            step_cost = sqrt(2) if diagonal else 1.0
            new_cost = costs[current] + step_cost

            if new_cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + heuristic(neighbor, goal)
                heappush(frontier, (priority, neighbor))

    return None             
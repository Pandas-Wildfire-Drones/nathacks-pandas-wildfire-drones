import pytest

from autonomy.path_planning import find_path
from contracts.geometry import GridSpec, Position


def make_grid(rows, cols):
    """Create a one-metre grid with its south-west corner at (0, 0)."""
    return GridSpec(
        origin=Position(x=0, y=0, z=0),
        resolution_m=1,
        rows=rows,
        cols=cols,
    )


def world_position(row, col):
    """Return the center of a cell in a one-metre grid."""
    return Position(
        x=col + 0.5,
        y=row + 0.5,
        z=0,
    )


def test_finds_path():
    grid = [[0, 0, 0] for _ in range(3)]
    spec = make_grid(3, 3)

    path = find_path(
        grid,
        spec,
        world_position(0, 0),
        world_position(2, 2),
    )

    assert path is not None
    assert path[0] == world_position(0, 0)
    assert path[-1] == world_position(2, 2)


def test_avoids_obstacles():
    grid = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    spec = make_grid(3, 3)

    path = find_path(
        grid,
        spec,
        world_position(0, 0),
        world_position(0, 2),
    )

    assert path is not None

    for position in path:
        cell = spec.world_to_cell(position)
        assert cell is not None
        assert grid[cell.row][cell.col] == 0


def test_no_route_returns_none():
    grid = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
    spec = make_grid(3, 3)

    path = find_path(
        grid,
        spec,
        world_position(0, 0),
        world_position(0, 2),
    )

    assert path is None


def test_rejects_start_outside_grid():
    grid = [[0, 0], [0, 0]]
    spec = make_grid(2, 2)

    with pytest.raises(ValueError, match="Start position is outside"):
        find_path(
            grid,
            spec,
            Position(x=-1, y=0.5),
            world_position(0, 1),
        )


def test_rejects_blocked_start():
    grid = [
        [1, 0],
        [0, 0],
    ]
    spec = make_grid(2, 2)

    with pytest.raises(ValueError, match="Start position is in a blocked cell"):
        find_path(
            grid,
            spec,
            world_position(0, 0),
            world_position(1, 1),
        )


def test_does_not_cut_diagonal_obstacle_corner():
    grid = [
        [0, 1],
        [1, 0],
    ]
    spec = make_grid(2, 2)

    path = find_path(
        grid,
        spec,
        world_position(0, 0),
        world_position(1, 1),
    )

    assert path is None


def test_rejects_grid_shape_mismatch():
    grid = [
        [0, 0],
        [0, 0],
    ]
    spec = make_grid(3, 3)

    with pytest.raises(ValueError, match="row count"):
        find_path(
            grid,
            spec,
            world_position(0, 0),
            world_position(1, 1),
        )

from autonomy.path_planning import find_path


def test_finds_path():
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    path = find_path(grid, (0, 0), (2, 2))
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)


def test_avoids_obstacles():
    grid = [[0, 1, 0], [0, 1, 0], [0, 0, 0]]
    path = find_path(grid, (0, 0), (2, 0))
    assert path is not None
    assert all(grid[y][x] == 0 for x, y in path)


def test_no_route_returns_none():
    grid = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert find_path(grid, (0, 0), (2, 0)) is None

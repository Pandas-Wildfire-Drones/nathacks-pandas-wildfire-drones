import pytest

from autonomy.cbs_planner import (
    PathRequest,
    path_duration_s,
    plan_paths,
)
from contracts.geometry import GridSpec, Position


def make_grid(rows, cols):
    return GridSpec(
        origin=Position(x=0, y=0, z=0),
        resolution_m=1,
        rows=rows,
        cols=cols,
    )


def cell(row, col, z=0):
    return Position(
        x=col + 0.5,
        y=row + 0.5,
        z=z,
    )


def test_plans_single_drone():
    grid = [[0, 0, 0] for _ in range(3)]

    requests = [
        PathRequest(
            drone_id="d1",
            start=cell(0, 0),
            goal=cell(0, 2),
            speed_mps=10,
        )
    ]

    paths = plan_paths(
        requests,
        grid,
        make_grid(3, 3),
        minimum_separation_m=2,
    )

    assert paths is not None
    assert paths["d1"][0] == cell(0, 0)
    assert paths["d1"][-1] == cell(0, 2)


def test_altitude_band_controls_planner_altitude():
    grid = [[0, 0, 0]]

    requests = [
        PathRequest(
            drone_id="d1",
            start=cell(0, 0),
            goal=cell(0, 2),
            speed_mps=10,
            altitude_band_m=(10, 20),
        )
    ]

    paths = plan_paths(
        requests,
        grid,
        make_grid(1, 3),
        minimum_separation_m=2,
    )

    assert paths is not None

    for position in paths["d1"]:
        assert position.z == pytest.approx(15.0)


def test_different_altitudes_are_not_a_conflict():
    grid = [[0, 0, 0]]

    requests = [
        PathRequest(
            drone_id="low",
            start=cell(0, 0, 10),
            goal=cell(0, 2, 10),
            speed_mps=10,
            altitude_band_m=(10, 20),
        ),
        PathRequest(
            drone_id="high",
            start=cell(0, 0, 40),
            goal=cell(0, 2, 40),
            speed_mps=10,
            altitude_band_m=(40, 50),
        ),
    ]

    paths = plan_paths(
        requests,
        grid,
        make_grid(1, 3),
        minimum_separation_m=10,
    )

    assert paths is not None

    for low, high in zip(
        paths["low"],
        paths["high"],
    ):
        assert low.distance_to(high) >= 10


def test_cbs_resolves_crossing_paths():
    grid = [[0, 0, 0] for _ in range(3)]

    requests = [
        PathRequest(
            drone_id="eastbound",
            start=cell(1, 0),
            goal=cell(1, 2),
            speed_mps=1,
        ),
        PathRequest(
            drone_id="westbound",
            start=cell(1, 2),
            goal=cell(1, 0),
            speed_mps=1,
        ),
    ]

    paths = plan_paths(
        requests,
        grid,
        make_grid(3, 3),
        minimum_separation_m=1.1,
        time_step_s=1.0,
        max_cbs_nodes=100,
        max_timesteps=20,
    )

    assert paths is not None

    assert paths["eastbound"][-1] == cell(1, 2)
    assert paths["westbound"][-1] == cell(1, 0)

    shared_steps = max(
        len(paths["eastbound"]),
        len(paths["westbound"]),
    )

    for timestep in range(shared_steps):
        east = paths["eastbound"][
            min(timestep, len(paths["eastbound"]) - 1)
        ]
        west = paths["westbound"][
            min(timestep, len(paths["westbound"]) - 1)
        ]

        assert east.distance_to(west) >= 1.1


def test_blocked_grid_returns_no_path():
    grid = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]

    request = PathRequest(
        drone_id="d1",
        start=cell(0, 0),
        goal=cell(0, 2),
        speed_mps=10,
    )

    paths = plan_paths(
        [request],
        grid,
        make_grid(3, 3),
        minimum_separation_m=1,
    )

    assert paths is None


def test_duplicate_drone_ids_are_rejected():
    grid = [[0, 0]]

    requests = [
        PathRequest(
            drone_id="d1",
            start=cell(0, 0),
            goal=cell(0, 1),
            speed_mps=10,
        ),
        PathRequest(
            drone_id="d1",
            start=cell(0, 1),
            goal=cell(0, 0),
            speed_mps=10,
        ),
    ]

    with pytest.raises(ValueError, match="Drone IDs must be unique"):
        plan_paths(
            requests,
            grid,
            make_grid(1, 2),
            minimum_separation_m=1,
        )


def test_invalid_altitude_band_is_rejected():
    with pytest.raises(
        ValueError,
        match="lower bound must be less than upper bound",
    ):
        PathRequest(
            drone_id="d1",
            start=cell(0, 0),
            goal=cell(0, 1),
            speed_mps=10,
            altitude_band_m=(20, 10),
        )


def test_invalid_speed_is_rejected():
    with pytest.raises(
        ValueError,
        match="speed_mps must be finite and greater than zero",
    ):
        PathRequest(
            drone_id="d1",
            start=cell(0, 0),
            goal=cell(0, 1),
            speed_mps=0,
        )


def test_path_duration_uses_planning_clock():
    path = [
        cell(0, 0),
        cell(0, 1),
        cell(0, 2),
    ]

    assert path_duration_s(
        path,
        time_step_s=0.1,
    ) == pytest.approx(0.2)

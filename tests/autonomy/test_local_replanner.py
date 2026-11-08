from autonomy.cbs_planner import PathRequest
from autonomy.local_replanner import replan_conflict_groups
from contracts.geometry import GridSpec, Position


def cell(row, col, z=0):
    return Position(
        x=col + 0.5,
        y=row + 0.5,
        z=z,
    )


def make_grid(rows, cols):
    return GridSpec(
        origin=Position(x=0, y=0, z=0),
        resolution_m=1,
        rows=rows,
        cols=cols,
    )


def test_singleton_routes_are_not_replanned():
    grid = [[0, 0, 0] for _ in range(3)]

    requests = [
        PathRequest(
            drone_id="d1",
            start=cell(0, 0),
            goal=cell(0, 2),
            speed_mps=10,
        ),
        PathRequest(
            drone_id="d2",
            start=cell(2, 0),
            goal=cell(2, 2),
            speed_mps=10,
        ),
    ]

    existing = {
        "d1": [
            cell(0, 0),
            cell(0, 1),
            cell(0, 2),
        ],
        "d2": [
            cell(2, 0),
            cell(2, 1),
            cell(2, 2),
        ],
    }

    result = replan_conflict_groups(
        requests,
        existing,
        grid,
        make_grid(3, 3),
        minimum_separation_m=1,
        conflict_group_threshold_m=1,
    )

    assert result == existing


def test_conflicting_group_is_replanned():
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

    existing = {
        "eastbound": [
            cell(1, 0),
            cell(1, 1),
            cell(1, 2),
        ],
        "westbound": [
            cell(1, 2),
            cell(1, 1),
            cell(1, 0),
        ],
    }

    result = replan_conflict_groups(
        requests,
        existing,
        grid,
        make_grid(3, 3),
        minimum_separation_m=1.1,
        conflict_group_threshold_m=2,
        time_step_s=1,
        max_cbs_nodes=100,
        max_timesteps=20,
    )

    assert result is not None

    assert result["eastbound"][-1] == cell(1, 2)
    assert result["westbound"][-1] == cell(1, 0)

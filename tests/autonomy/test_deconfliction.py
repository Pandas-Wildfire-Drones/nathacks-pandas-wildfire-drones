import pytest

from autonomy.deconfliction import Conflict, detect_conflicts
from contracts.geometry import Position


def test_detects_close_drones():
    proposed = [(0, 0, 10), (1, 0, 10)]
    other = [(0, 1, 10), (1, 1, 10)]

    conflicts = detect_conflicts(proposed, [other], 2)

    assert len(conflicts) == 2
    assert conflicts[0] == Conflict(timestep=0, distance_m=1)
    assert conflicts[1] == Conflict(timestep=1, distance_m=1)


def test_no_conflict_when_altitude_separation_is_sufficient():
    proposed = [(0, 0, 10), (1, 0, 10)]
    other = [(0, 0, 30), (1, 0, 30)]

    conflicts = detect_conflicts(proposed, [other], 5)

    assert conflicts == []


def test_detects_conflict_using_position_contracts():
    proposed = [
        Position(x=0, y=0, z=10),
        Position(x=5, y=0, z=10),
    ]
    other = [
        Position(x=0, y=0.5, z=10),
        Position(x=5, y=0.5, z=10),
    ]

    conflicts = detect_conflicts(proposed, [other], 2)

    assert len(conflicts) == 2


def test_exact_separation_threshold_is_not_a_conflict():
    proposed = [(0, 0, 10)]
    other = [(3, 0, 10)]

    conflicts = detect_conflicts(proposed, [other], 3)

    assert conflicts == []


def test_detects_conflicts_against_multiple_drones():
    proposed = [(0, 0, 10)]
    other_paths = [
        [(0, 1, 10)],
        [(0, 20, 10)],
        [(0, 0, 11)],
    ]

    conflicts = detect_conflicts(proposed, other_paths, 2)

    assert len(conflicts) == 2


def test_only_compares_shared_timesteps():
    proposed = [
        (0, 0, 10),
        (1, 0, 10),
        (2, 0, 10),
    ]
    other = [
        (0, 0, 10),
    ]

    conflicts = detect_conflicts(proposed, [other], 2)

    assert len(conflicts) == 1
    assert conflicts[0].timestep == 0


def test_empty_paths_have_no_conflicts():
    assert detect_conflicts([], [[(0, 0, 0)]], 2) == []
    assert detect_conflicts([(0, 0, 0)], [[]], 2) == []


@pytest.mark.parametrize("separation", [-1, float("inf"), float("nan")])
def test_rejects_invalid_minimum_separation(separation):
    with pytest.raises(ValueError):
        detect_conflicts([(0, 0, 0)], [], separation)


def test_rejects_points_with_wrong_number_of_coordinates():
    with pytest.raises(ValueError):
        detect_conflicts([(0, 0)], [], 2)


def test_rejects_non_finite_path_coordinates():
    with pytest.raises(ValueError):
        detect_conflicts([(0, 0, float("nan"))], [], 2)

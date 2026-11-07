
import pytest
from autonomy.coverage_search import generate_coverage_waypoints


def test_generates_lawnmower_waypoints():
    points = generate_coverage_waypoints(0, 0, 10, 10, 5)

    assert points[0] == (0, 0)
    assert points[1] == (10, 0)
    assert points[2] == (10, 5)
    assert points[3] == (0, 5)


def test_invalid_spacing():
    with pytest.raises(ValueError):
        generate_coverage_waypoints(0, 0, 10, 10, 0)

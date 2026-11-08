import pytest

from autonomy.coverage_search import generate_coverage_waypoints
from contracts.entities import Zone
from contracts.geometry import Position


def make_zone(points):
    return Zone(
        zone_id="zone_test",
        boundary=[
            Position(x=x, y=y, z=0)
            for x, y in points
        ],
    )


def test_generates_lawnmower_waypoints_for_rectangle():
    zone = make_zone([
        (0, 0),
        (10, 0),
        (10, 10),
        (0, 10),
    ])

    points = generate_coverage_waypoints(zone, spacing_m=5)

    assert len(points) == 4

    assert points[0].y == pytest.approx(2.5)
    assert points[1].y == pytest.approx(2.5)
    assert points[2].y == pytest.approx(7.5)
    assert points[3].y == pytest.approx(7.5)

    # First row travels left to right.
    assert points[0].x < points[1].x

    # Second row travels right to left.
    assert points[2].x > points[3].x


def test_all_rectangle_waypoints_are_inside_zone():
    zone = make_zone([
        (0, 0),
        (10, 0),
        (10, 10),
        (0, 10),
    ])

    points = generate_coverage_waypoints(zone, spacing_m=2)

    assert points
    assert all(zone.contains(point) for point in points)


def test_supports_concave_polygon():
    # U-shaped polygon with two separate sections at y=5.
    zone = make_zone([
        (0, 0),
        (10, 0),
        (10, 10),
        (7, 10),
        (7, 3),
        (3, 3),
        (3, 10),
        (0, 10),
    ])

    points = generate_coverage_waypoints(zone, spacing_m=2)

    assert points
    assert all(zone.contains(point) for point in points)

    row_points = [
        point for point in points
        if point.y == pytest.approx(5)
    ]

    # Two separate sections, with two endpoints each.
    assert len(row_points) == 4


def test_uses_requested_altitude():
    zone = make_zone([
        (0, 0),
        (10, 0),
        (10, 10),
        (0, 10),
    ])

    points = generate_coverage_waypoints(
        zone,
        spacing_m=5,
        z_m=25,
    )

    assert points
    assert all(point.z == 25 for point in points)


def test_defaults_to_boundary_altitude():
    zone = Zone(
        zone_id="zone_altitude",
        boundary=[
            Position(x=0, y=0, z=12),
            Position(x=10, y=0, z=12),
            Position(x=10, y=10, z=12),
            Position(x=0, y=10, z=12),
        ],
    )

    points = generate_coverage_waypoints(zone, spacing_m=5)

    assert points
    assert all(point.z == 12 for point in points)


@pytest.mark.parametrize(
    "spacing",
    [0, -1, float("inf"), float("nan")],
)
def test_rejects_invalid_spacing(spacing):
    zone = make_zone([
        (0, 0),
        (10, 0),
        (10, 10),
        (0, 10),
    ])

    with pytest.raises(ValueError):
        generate_coverage_waypoints(zone, spacing)


def test_rejects_zone_without_positive_height():
    zone = make_zone([
        (0, 0),
        (10, 0),
        (5, 0),
    ])

    with pytest.raises(ValueError):
        generate_coverage_waypoints(zone, spacing_m=2)


def test_rejects_zone_without_positive_width():
    zone = make_zone([
        (5, 0),
        (5, 10),
        (5, 5),
    ])

    with pytest.raises(ValueError):
        generate_coverage_waypoints(zone, spacing_m=2)


def test_rejects_non_finite_altitude():
    zone = make_zone([
        (0, 0),
        (10, 0),
        (10, 10),
        (0, 10),
    ])

    with pytest.raises(ValueError):
        generate_coverage_waypoints(
            zone,
            spacing_m=2,
            z_m=float("nan"),
        )
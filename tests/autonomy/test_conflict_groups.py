import pytest

from autonomy.conflict_groups import (
    PlannedDroneRoute,
    build_conflict_groups,
    routes_may_conflict,
)
from contracts.geometry import Position


def point(x, y, z):
    return Position(x=x, y=y, z=z)


def route(drone_id, points, altitude_band=None):
    return PlannedDroneRoute(
        drone_id=drone_id,
        waypoints=tuple(points),
        altitude_band_m=altitude_band,
    )


def test_nearby_routes_are_in_same_group():
    routes = [
        route(
            "d1",
            [point(0, 0, 10), point(10, 0, 10)],
            (10, 20),
        ),
        route(
            "d2",
            [point(0, 5, 10), point(10, 5, 10)],
            (10, 20),
        ),
    ]

    groups = build_conflict_groups(
        routes,
        horizontal_threshold_m=10,
    )

    assert len(groups) == 1
    assert groups[0] == {"d1", "d2"}


def test_far_apart_routes_are_separate():
    routes = [
        route(
            "d1",
            [point(0, 0, 10), point(10, 0, 10)],
            (10, 20),
        ),
        route(
            "d2",
            [point(100, 100, 10), point(110, 100, 10)],
            (10, 20),
        ),
    ]

    groups = build_conflict_groups(
        routes,
        horizontal_threshold_m=10,
    )

    assert len(groups) == 2
    assert {frozenset(group) for group in groups} == {
        frozenset({"d1"}),
        frozenset({"d2"}),
    }


def test_non_overlapping_altitude_bands_are_not_grouped():
    routes = [
        route(
            "low",
            [point(0, 0, 10), point(10, 0, 10)],
            (10, 20),
        ),
        route(
            "high",
            [point(0, 0, 40), point(10, 0, 40)],
            (40, 50),
        ),
    ]

    assert not routes_may_conflict(
        routes[0],
        routes[1],
        horizontal_threshold_m=100,
    )

    groups = build_conflict_groups(
        routes,
        horizontal_threshold_m=100,
    )

    assert len(groups) == 2


def test_missing_altitude_is_conservative():
    routes = [
        route(
            "d1",
            [point(0, 0, 10), point(10, 0, 10)],
        ),
        route(
            "d2",
            [point(0, 5, 40), point(10, 5, 40)],
            (40, 50),
        ),
    ]

    assert routes_may_conflict(
        routes[0],
        routes[1],
        horizontal_threshold_m=10,
    )


def test_transitive_conflicts_form_one_group():
    routes = [
        route("d1", [point(0, 0, 10)], (10, 20)),
        route("d2", [point(5, 0, 10)], (10, 20)),
        route("d3", [point(10, 0, 10)], (10, 20)),
    ]

    groups = build_conflict_groups(
        routes,
        horizontal_threshold_m=6,
    )

    assert len(groups) == 1
    assert groups[0] == {"d1", "d2", "d3"}


def test_same_drone_does_not_conflict_with_itself():
    first = route(
        "d1",
        [point(0, 0, 10)],
        (10, 20),
    )

    assert not routes_may_conflict(
        first,
        first,
        horizontal_threshold_m=100,
    )


def test_empty_route_is_rejected():
    with pytest.raises(
        ValueError,
        match="waypoints must not be empty",
    ):
        PlannedDroneRoute(
            drone_id="d1",
            waypoints=(),
        )


def test_invalid_threshold_is_rejected():
    routes = [
        route("d1", [point(0, 0, 10)]),
        route("d2", [point(1, 0, 10)]),
    ]

    with pytest.raises(
        ValueError,
        match="horizontal_threshold_m",
    ):
        routes_may_conflict(
            routes[0],
            routes[1],
            horizontal_threshold_m=-1,
        )

"""Identify drones that require local multi-drone replanning."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from contracts.geometry import Position


@dataclass(frozen=True)
class PlannedDroneRoute:
    """Minimal route information needed for conflict grouping."""

    drone_id: str
    waypoints: tuple[Position, ...]
    altitude_band_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _validate_route(self)


def _validate_route(route: PlannedDroneRoute) -> None:
    if not route.drone_id:
        raise ValueError("drone_id must not be empty")

    if not route.waypoints:
        raise ValueError("waypoints must not be empty")

    if route.altitude_band_m is not None:
        lower, upper = route.altitude_band_m

        if not isfinite(lower) or not isfinite(upper):
            raise ValueError("altitude band must be finite")

        if lower >= upper:
            raise ValueError(
                "altitude band lower bound must be less than upper bound"
            )


def _altitude_bands_overlap(
    first: tuple[float, float] | None,
    second: tuple[float, float] | None,
) -> bool:
    """Return whether two altitude bands can occupy the same vertical space."""

    if first is None or second is None:
        return True

    first_lower, first_upper = first
    second_lower, second_upper = second

    return (
        first_lower < second_upper
        and second_lower < first_upper
    )


def _minimum_route_distance(
    first: PlannedDroneRoute,
    second: PlannedDroneRoute,
) -> float:
    """Return minimum horizontal distance between route segments."""

    minimum = float("inf")

    for first_point in first.waypoints:
        for second_point in second.waypoints:
            distance = first_point.distance_to(
                second_point,
                horizontal=True,
            )

            if distance < minimum:
                minimum = distance

    for first_start, first_end in zip(
        first.waypoints,
        first.waypoints[1:],
    ):
        for second_start, second_end in zip(
            second.waypoints,
            second.waypoints[1:],
        ):
            if _segments_intersect(
                first_start,
                first_end,
                second_start,
                second_end,
            ):
                return 0.0

            minimum = min(
                minimum,
                _point_to_segment_distance(
                    first_start,
                    second_start,
                    second_end,
                ),
                _point_to_segment_distance(
                    first_end,
                    second_start,
                    second_end,
                ),
                _point_to_segment_distance(
                    second_start,
                    first_start,
                    first_end,
                ),
                _point_to_segment_distance(
                    second_end,
                    first_start,
                    first_end,
                ),
            )

    return minimum


def _point_to_segment_distance(
    point: Position,
    start: Position,
    end: Position,
) -> float:
    segment_x = end.x - start.x
    segment_y = end.y - start.y
    length_squared = segment_x * segment_x + segment_y * segment_y
    if length_squared == 0:
        return point.distance_to(start, horizontal=True)

    projection = (
        (point.x - start.x) * segment_x
        + (point.y - start.y) * segment_y
    ) / length_squared
    projection = min(1.0, max(0.0, projection))
    closest = Position(
        x=start.x + projection * segment_x,
        y=start.y + projection * segment_y,
        z=0,
    )
    return point.distance_to(closest, horizontal=True)


def _segments_intersect(
    first_start: Position,
    first_end: Position,
    second_start: Position,
    second_end: Position,
) -> bool:
    def orientation(a: Position, b: Position, c: Position) -> float:
        return (
            (b.x - a.x) * (c.y - a.y)
            - (b.y - a.y) * (c.x - a.x)
        )

    def within(value: float, lower: float, upper: float) -> bool:
        return min(lower, upper) <= value <= max(lower, upper)

    first_orientation = orientation(first_start, first_end, second_start)
    second_orientation = orientation(first_start, first_end, second_end)
    third_orientation = orientation(second_start, second_end, first_start)
    fourth_orientation = orientation(second_start, second_end, first_end)

    if (
        first_orientation == 0
        and within(second_start.x, first_start.x, first_end.x)
        and within(second_start.y, first_start.y, first_end.y)
    ):
        return True
    if (
        second_orientation == 0
        and within(second_end.x, first_start.x, first_end.x)
        and within(second_end.y, first_start.y, first_end.y)
    ):
        return True
    if (
        third_orientation == 0
        and within(first_start.x, second_start.x, second_end.x)
        and within(first_start.y, second_start.y, second_end.y)
    ):
        return True
    if (
        fourth_orientation == 0
        and within(first_end.x, second_start.x, second_end.x)
        and within(first_end.y, second_start.y, second_end.y)
    ):
        return True

    first_pair_crosses = (
        first_orientation > 0 > second_orientation
        or second_orientation > 0 > first_orientation
    )
    second_pair_crosses = (
        third_orientation > 0 > fourth_orientation
        or fourth_orientation > 0 > third_orientation
    )
    return first_pair_crosses and second_pair_crosses


def routes_may_conflict(
    first: PlannedDroneRoute,
    second: PlannedDroneRoute,
    *,
    horizontal_threshold_m: float,
) -> bool:
    """Return whether two routes belong in the same local group.

    This is only a conservative candidate filter. It is not the final
    collision decision.
    """

    if (
        not isfinite(horizontal_threshold_m)
        or horizontal_threshold_m < 0
    ):
        raise ValueError(
            "horizontal_threshold_m must be finite and non-negative"
        )

    _validate_route(first)
    _validate_route(second)

    if first.drone_id == second.drone_id:
        return False

    if not _altitude_bands_overlap(
        first.altitude_band_m,
        second.altitude_band_m,
    ):
        return False

    return (
        _minimum_route_distance(first, second)
        < horizontal_threshold_m
    )


def build_conflict_groups(
    routes: list[PlannedDroneRoute],
    *,
    horizontal_threshold_m: float,
) -> list[set[str]]:
    """Build connected local conflict groups."""

    if (
        not isfinite(horizontal_threshold_m)
        or horizontal_threshold_m < 0
    ):
        raise ValueError(
            "horizontal_threshold_m must be finite and non-negative"
        )

    for route in routes:
        _validate_route(route)

    adjacency: dict[str, set[str]] = {
        route.drone_id: set()
        for route in routes
    }

    for index, first in enumerate(routes):
        for second in routes[index + 1:]:
            if routes_may_conflict(
                first,
                second,
                horizontal_threshold_m=horizontal_threshold_m,
            ):
                adjacency[first.drone_id].add(second.drone_id)
                adjacency[second.drone_id].add(first.drone_id)

    groups: list[set[str]] = []
    unvisited = set(adjacency)

    while unvisited:
        start = next(iter(unvisited))
        stack = [start]
        group: set[str] = set()

        while stack:
            drone_id = stack.pop()

            if drone_id not in unvisited:
                continue

            unvisited.remove(drone_id)
            group.add(drone_id)
            stack.extend(adjacency[drone_id] - group)

        groups.append(group)

    return groups

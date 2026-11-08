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
    """Return minimum horizontal distance between route waypoints."""

    minimum = float("inf")

    for first_point in first.waypoints:
        for second_point in second.waypoints:
            distance = first_point.distance_to(
                second_point,
                horizontal=True,
            )

            if distance < minimum:
                minimum = distance

    return minimum


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

"""Detect unsafe proximity between synchronized 3D drone paths."""

from dataclasses import dataclass
from math import dist, isfinite

from contracts.geometry import Position


@dataclass(frozen=True)
class Conflict:
    """A timestep where two planned drone positions are too close."""

    timestep: int
    distance_m: float


def _coordinates(point) -> tuple[float, float, float]:
    """Normalize a Position or an (x, y, z) tuple into coordinates."""
    if isinstance(point, Position):
        coordinates = (point.x, point.y, point.z)
    else:
        try:
            if len(point) != 3:
                raise ValueError("Each path point must have exactly 3 coordinates")
            coordinates = tuple(float(value) for value in point)
        except TypeError as exc:
            raise ValueError(
                "Each path point must be a Position or an (x, y, z) tuple"
            ) from exc

    if not all(isfinite(value) for value in coordinates):
        raise ValueError("Path coordinates must be finite")

    return coordinates


def detect_conflicts(
    proposed_path,
    other_paths,
    minimum_separation_m,
) -> list[Conflict]:
    """Detect close approaches between synchronized 3D paths.

    Path points are compared at matching timestep indices. Each point may
    be a shared Position contract or an (x, y, z) tuple, in metres.

    Only overlapping timesteps are compared. A conflict is reported when
    3D distance is strictly less than minimum_separation_m.
    """
    if not isfinite(minimum_separation_m) or minimum_separation_m < 0:
        raise ValueError("Minimum separation must be finite and non-negative")

    proposed = [_coordinates(point) for point in proposed_path]
    conflicts = []

    for other_path in other_paths:
        other = [_coordinates(point) for point in other_path]
        shared_steps = min(len(proposed), len(other))

        for timestep in range(shared_steps):
            distance_m = dist(proposed[timestep], other[timestep])

            if distance_m < minimum_separation_m:
                conflicts.append(
                    Conflict(
                        timestep=timestep,
                        distance_m=distance_m,
                    )
                )

    return conflicts
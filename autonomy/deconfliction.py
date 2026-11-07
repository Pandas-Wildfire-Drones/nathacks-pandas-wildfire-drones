"""Checks whether two drones' planned paths come too close to each other. Each path point is (x, y, altitude) in metres. The paths must be sampled at the same time intervals."""

from dataclasses import dataclass
from math import dist


@dataclass
class Conflict:
    timestep: int
    distance_m: float


def detect_conflicts(
    proposed_path,
    other_paths,
    minimum_separation_m,
):
    """
    Compare a proposed 3D path against other synchronized paths.

    Each point is (x, y, altitude), in metres.
    Returns a list of detected conflicts.
    """
    if minimum_separation_m < 0:
        raise ValueError("Minimum separation cannot be negative")

    conflicts = []

    for other_path in other_paths:
        shared_steps = min(len(proposed_path), len(other_path))

        for timestep in range(shared_steps):
            distance = dist(
                proposed_path[timestep],
                other_path[timestep],
            )

            if distance < minimum_separation_m:
                conflicts.append(
                    Conflict(timestep=timestep, distance_m=distance)
                )

    return conflicts

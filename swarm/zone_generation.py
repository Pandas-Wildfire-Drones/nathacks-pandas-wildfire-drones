"""Generate rectangular search zones in the swarm's local ENU coordinate frame."""

from __future__ import annotations

import math

from contracts import Position, Zone

__all__ = ["generate_grid_zones", "zone_center"]


def zone_center(zone: Zone) -> Position:
    """Return a representative target point from a zone's polygon vertices."""
    return Position(
        x=sum(point.x for point in zone.boundary) / len(zone.boundary),
        y=sum(point.y for point in zone.boundary) / len(zone.boundary),
        z=sum(point.z for point in zone.boundary) / len(zone.boundary),
    )


def generate_grid_zones(
    southwest: Position,
    northeast: Position,
    *,
    zone_width_m: float,
    zone_height_m: float | None = None,
    zone_id_prefix: str = "zone",
) -> tuple[Zone, ...]:
    """Tile a rectangular operational area with clipped edge zones.

    The caller must first convert geographic boundaries to the scenario's local
    ENU frame. This routine accepts meters, not longitude/latitude.
    """
    height_m = zone_width_m if zone_height_m is None else zone_height_m
    if not math.isfinite(zone_width_m) or zone_width_m <= 0:
        raise ValueError("zone_width_m must be finite and > 0")
    if not math.isfinite(height_m) or height_m <= 0:
        raise ValueError("zone_height_m must be finite and > 0")
    if not zone_id_prefix or any(
        char in zone_id_prefix for char in "/#*+\t\r\n "
    ):
        raise ValueError("zone_id_prefix must be non-empty and topic-safe")
    if southwest.x >= northeast.x or southwest.y >= northeast.y:
        raise ValueError("northeast must be strictly north and east of southwest")
    if southwest.z != northeast.z:
        raise ValueError("southwest and northeast must have the same z coordinate")

    width_m = northeast.x - southwest.x
    area_height_m = northeast.y - southwest.y
    columns_float = width_m / zone_width_m
    rows_float = area_height_m / height_m
    if not math.isfinite(columns_float) or not math.isfinite(rows_float):
        raise ValueError("zone dimensions produce a non-finite grid size")
    columns = math.ceil(columns_float)
    rows = math.ceil(rows_float)

    zones: list[Zone] = []
    for row in range(rows):
        south = southwest.y + row * height_m
        north = min(south + height_m, northeast.y)
        for column in range(columns):
            west = southwest.x + column * zone_width_m
            east = min(west + zone_width_m, northeast.x)
            zones.append(
                Zone(
                    zone_id=f"{zone_id_prefix}-r{row:04d}-c{column:04d}",
                    boundary=[
                        Position(x=west, y=south, z=southwest.z),
                        Position(x=east, y=south, z=southwest.z),
                        Position(x=east, y=north, z=southwest.z),
                        Position(x=west, y=north, z=southwest.z),
                    ],
                )
            )
    return tuple(zones)

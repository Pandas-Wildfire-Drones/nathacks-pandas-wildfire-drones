"""Coordinates, poses and grids.

CONVENTIONS - every workstream must follow these:
  * Local frame is ENU in meters: x = east, y = north, z = up.
  * Origin is ScenarioConfig.origin (a GeoOrigin). lat/lon exist only at the edges
    (data import, ground-base map display); never inside the swarm core.
  * z is height above the scenario datum, NOT above ground (AGL).
  * Headings/yaw are compass degrees clockwise from north (0 = N, 90 = E).
  * Grids are indexed [row, col]; row 0 is the SOUTH edge (flip when plotting).
  * PX4 uses NED/FRD axes: convert inside the PX4 adapter only.
"""
from __future__ import annotations

import math
from typing import Annotated

from pydantic import Field

from .base import FrozenContract

__all__ = ["Meters", "Position", "Pose", "GeoOrigin", "GridIndex", "GridSpec", "point_in_polygon"]

Meters = Annotated[float, Field(allow_inf_nan=False)]


class Position(FrozenContract):
    x: Meters  # meters east of scenario origin
    y: Meters  # meters north of scenario origin
    z: Meters = 0.0  # meters above scenario datum (NOT AGL)

    def distance_to(self, other: "Position", horizontal: bool = False) -> float:
        dz = 0.0 if horizontal else other.z - self.z
        return math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2 + dz ** 2)

    def offset(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Position":
        return Position(x=self.x + dx, y=self.y + dy, z=self.z + dz)


class Pose(FrozenContract):
    position: Position
    yaw_deg: float = 0.0    # compass heading, clockwise from north
    pitch_deg: float = 0.0  # positive = nose up
    roll_deg: float = 0.0   # positive = right side down


class GeoOrigin(FrozenContract):
    """Where local (0, 0) sits on Earth. Used only at the edges (data import, map display)."""

    lat_deg: float = Field(ge=-90, le=90)
    lon_deg: float = Field(ge=-180, le=180)
    utm_epsg: int  # e.g. 32612 = WGS84 / UTM zone 12N
    datum_elevation_m: float = 0.0


class GridIndex(FrozenContract):
    row: int = Field(ge=0)  # 0 = southern edge
    col: int = Field(ge=0)  # 0 = western edge


class GridSpec(FrozenContract):
    """Shared definition of the map grid. Everyone indexes numpy arrays as [row, col]."""

    origin: Position  # south-west corner of cell (0, 0)
    resolution_m: float = Field(gt=0)
    rows: int = Field(gt=0)  # extent along +y (north)
    cols: int = Field(gt=0)  # extent along +x (east)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def width_m(self) -> float:
        return self.cols * self.resolution_m

    @property
    def height_m(self) -> float:
        return self.rows * self.resolution_m

    def world_to_cell(self, pos: Position) -> GridIndex | None:
        """Cell containing `pos`, or None if outside the grid."""
        col = math.floor((pos.x - self.origin.x) / self.resolution_m)
        row = math.floor((pos.y - self.origin.y) / self.resolution_m)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return GridIndex(row=row, col=col)
        return None

    def cell_to_world(self, idx: GridIndex) -> Position:
        """Center of the cell (z = grid origin z)."""
        return Position(
            x=self.origin.x + (idx.col + 0.5) * self.resolution_m,
            y=self.origin.y + (idx.row + 0.5) * self.resolution_m,
            z=self.origin.z,
        )

    def contains(self, pos: Position) -> bool:
        return self.world_to_cell(pos) is not None


def point_in_polygon(x: float, y: float, polygon: list[Position]) -> bool:
    """Ray-casting point-in-polygon test on the x/y plane."""
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[j].x, polygon[j].y
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside

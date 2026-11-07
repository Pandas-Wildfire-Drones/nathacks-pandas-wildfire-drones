"""Terrain height (DEM) sampling for georeferencing and height-above-ground (AGL).

Without terrain, georef assumes flat ground at the datum. On a slope that misplaces hotspots: a height error dz
shifts a point at the image edge by ~dz * tan(16 deg) for the seeker (5 m DEM error -> ~1.4 m), more for the
extinguisher's 50 deg lens. It also corrupts the GSD, and with it area estimates and scale-normalized patches.

DemSampler takes a height array already resampled onto the scenario grid ([row, col], row 0 = south, heights in
metres above the scenario datum, same convention as Position.z). Loading GeoTIFFs (rasterio) belongs at the edges
(sim/ or tools/), not here (hard rule 1).
"""
from __future__ import annotations

import math

import numpy as np

from contracts import GridSpec

__all__ = ["DemSampler"]


class DemSampler:
    """Bilinear height lookup at cell centres; clamps to the edge outside the grid. Callable: dem(x, y) -> z."""

    def __init__(self, heights: np.ndarray, grid: GridSpec):
        if heights.shape != grid.shape:
            raise ValueError(f"heights {heights.shape} != grid {grid.shape}")
        self.h = np.asarray(heights, dtype=np.float64)
        self.grid = grid

    @classmethod
    def flat(cls, grid: GridSpec, z: float = 0.0) -> "DemSampler":
        return cls(np.full(grid.shape, z), grid)

    def __call__(self, x: float, y: float) -> float:
        g = self.grid
        fc = (x - g.origin.x) / g.resolution_m - 0.5
        fr = (y - g.origin.y) / g.resolution_m - 0.5
        fc = min(max(fc, 0.0), g.cols - 1.0)
        fr = min(max(fr, 0.0), g.rows - 1.0)
        c0, r0 = int(math.floor(fc)), int(math.floor(fr))
        c1, r1 = min(c0 + 1, g.cols - 1), min(r0 + 1, g.rows - 1)
        tc, tr = fc - c0, fr - r0
        top = self.h[r0, c0] * (1 - tc) + self.h[r0, c1] * tc
        bot = self.h[r1, c0] * (1 - tc) + self.h[r1, c1] * tc
        return float(top * (1 - tr) + bot * tr)

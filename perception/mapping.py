"""Log-odds "fire present" occupancy map (the ImagingMap of the class diagram, as NumPy layers).

Layers, indexed [row, col] with row 0 = SOUTH (contracts.GridSpec):
  log_odds  float32, 0 = unknown (p = 0.5), clamped to [-10, 10] like GridCellUpdate
  last_seen float32 seconds, NaN = never imaged

Inverse sensor model per frame ([ASSUMPTION] numbers, tune with the evaluation harness):
  * every cell inside the frame footprint and NOT near a detection: += l_miss (evidence of no fire)
  * every cell within hit_radius_m of a detection:                  += logit(0.5 + 0.4 * confidence)
Seen-but-empty cells decay towards "no fire", so the leader can tell "searched, nothing" from "never searched".

Wire format: pop_updates() -> sparse GridCellUpdate list for MapSync (hard rule 3: no raw frames on the bus).
"""
from __future__ import annotations

import math

import numpy as np

from contracts import GridCellUpdate, GridSpec, Position

__all__ = ["LogOddsMap", "logit", "prob"]


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def prob(log_odds: np.ndarray | float):
    return 1.0 / (1.0 + np.exp(-np.asarray(log_odds)))


def _points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly: list[Position]) -> np.ndarray:
    inside = np.zeros(xs.shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = poly[i].x, poly[i].y, poly[j].x, poly[j].y
        crosses = (yi > ys) != (yj > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (xj - xi) * (ys - yi) / (yj - yi) + xi
        inside ^= crosses & (xs < x_at)
        j = i
    return inside


class LogOddsMap:
    def __init__(self, grid: GridSpec, l_miss: float = logit(0.4), hit_radius_m: float = 3.0,
                 l_min: float = -10.0, l_max: float = 10.0):
        self.grid = grid
        self.log_odds = np.zeros(grid.shape, dtype=np.float32)
        self.last_seen = np.full(grid.shape, np.nan, dtype=np.float32)
        self._dirty = np.zeros(grid.shape, dtype=bool)
        self.l_miss, self.hit_radius_m, self.l_min, self.l_max = l_miss, hit_radius_m, l_min, l_max

    def _cell_window(self, xs_min, xs_max, ys_min, ys_max):
        g = self.grid
        c0 = max(int(math.floor((xs_min - g.origin.x) / g.resolution_m)), 0)
        c1 = min(int(math.floor((xs_max - g.origin.x) / g.resolution_m)) + 1, g.cols)
        r0 = max(int(math.floor((ys_min - g.origin.y) / g.resolution_m)), 0)
        r1 = min(int(math.floor((ys_max - g.origin.y) / g.resolution_m)) + 1, g.rows)
        return r0, r1, c0, c1

    def _centres(self, r0, r1, c0, c1):
        g = self.grid
        cols = g.origin.x + (np.arange(c0, c1) + 0.5) * g.resolution_m
        rows = g.origin.y + (np.arange(r0, r1) + 0.5) * g.resolution_m
        return np.meshgrid(cols, rows)  # (xs, ys) with shape (r1-r0, c1-c0)

    def update(self, footprint: list[Position] | None, hits: list[tuple[Position, float]], now: float) -> None:
        """footprint: ground polygon of the frame (perception.georef.footprint_corners). hits: (position, confidence)."""
        hit_mask_global: dict[tuple[int, int, int, int], np.ndarray] = {}
        for pos, conf in hits:
            rr = self.hit_radius_m
            win = self._cell_window(pos.x - rr, pos.x + rr, pos.y - rr, pos.y + rr)
            r0, r1, c0, c1 = win
            if r1 <= r0 or c1 <= c0:
                continue
            xs, ys = self._centres(*win)
            m = (xs - pos.x) ** 2 + (ys - pos.y) ** 2 <= rr ** 2
            self.log_odds[r0:r1, c0:c1][m] += logit(0.5 + 0.4 * float(conf))
            self.last_seen[r0:r1, c0:c1][m] = now
            self._dirty[r0:r1, c0:c1] |= m
            hit_mask_global[win] = m
        if footprint:
            xs_all, ys_all = [p.x for p in footprint], [p.y for p in footprint]
            win = self._cell_window(min(xs_all), max(xs_all), min(ys_all), max(ys_all))
            r0, r1, c0, c1 = win
            if r1 > r0 and c1 > c0:
                xs, ys = self._centres(*win)
                m = _points_in_polygon(xs, ys, footprint)
                for (hr0, hr1, hc0, hc1), hm in hit_mask_global.items():  # don't add miss evidence on hit cells
                    a0, a1, b0, b1 = max(hr0, r0), min(hr1, r1), max(hc0, c0), min(hc1, c1)
                    if a1 > a0 and b1 > b0:
                        m[a0 - r0:a1 - r0, b0 - c0:b1 - c0] &= ~hm[a0 - hr0:a1 - hr0, b0 - hc0:b1 - hc0]
                self.log_odds[r0:r1, c0:c1][m] += self.l_miss
                self.last_seen[r0:r1, c0:c1][m] = now
                self._dirty[r0:r1, c0:c1] |= m
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)

    def probability(self) -> np.ndarray:
        return prob(self.log_odds)

    def pop_updates(self) -> list[GridCellUpdate]:
        rows, cols = np.nonzero(self._dirty)
        out = [GridCellUpdate(row=int(r), col=int(c), log_odds=float(self.log_odds[r, c]),
                              last_seen=float(self.last_seen[r, c])) for r, c in zip(rows, cols)]
        self._dirty[:] = False
        return out

    def apply_updates(self, cells: list[GridCellUpdate]) -> None:
        """Receiver side (leader / ground base): overwrite with newer cells."""
        for c in cells:
            prev = self.last_seen[c.row, c.col]
            if np.isnan(prev) or c.last_seen >= prev:
                self.log_odds[c.row, c.col] = c.log_odds
                self.last_seen[c.row, c.col] = c.last_seen

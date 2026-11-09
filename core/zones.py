"""Seeker search zones for a circular arena: columns x bands that every seeker finishes at about the same time.

The lawnmower (autonomy.coverage_search) flies east-west scan lines spaced one camera swath apart, so a zone is a
group of consecutive scan lines inside one column. A seeker's work, in metres flown, is

    transit (base -> start of its first line)  +  sum of its line lengths  +  turns between lines

and we minimise the WORST seeker's work (everyone starts at the same base; the search is done when the slowest
seeker is done):
  1. try 1..4 equal-area vertical columns; give each column a share of the seekers proportional to its area;
  2. inside a column, split its scan lines into contiguous bands by exact min-max linear partition (bisect on the
     finishing distance T, greedy fill from the south). Boundaries land on whole line spacings, so nobody flies a
     wasted half swath;
  3. keep the column count with the lowest worst-case work.
Columns matter on big maps: with ~900 m lines and few lines per seeker, horizontal bands alone can't balance
(one line is too coarse a unit), splitting lines in half or thirds can. Zones are the circle clipped to each cell.
"""
from __future__ import annotations

import math

from contracts import Position, Zone

__all__ = ["EqualTimeZoneAllocator", "band_edges", "column_edges", "strip_area", "zone_cells"]


def _F(y: float, R: float) -> float:
    y = min(max(y, -R), R)
    return y * math.sqrt(max(R * R - y * y, 0.0)) + R * R * math.asin(y / R)


def strip_area(y0: float, y1: float, R: float) -> float:
    """Area of {(x, y) in circle radius R : y0 <= y <= y1} (same function for vertical strips by symmetry)."""
    return max(_F(y1, R) - _F(y0, R), 0.0)      # integral of the chord 2 sqrt(R^2 - y^2)


def column_edges(c: int, R: float) -> list[float]:
    """c equal-area vertical columns."""
    total = math.pi * R * R
    edges = [-R]
    for k in range(1, c):
        lo, hi = -R, R
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if strip_area(-R, mid, R) < total * k / c:
                lo = mid
            else:
                hi = mid
        edges.append(hi)
    return edges + [R]


def _lines(R: float, spacing_m: float, x0: float, x1: float) -> list[tuple[float, float, float]]:
    """(y, x_start, length) of the scan lines inside column [x0, x1]; y aligned at -R + spacing/2 + k spacing."""
    out = []
    y = -R + spacing_m / 2
    while y < R:
        h = math.sqrt(max(R * R - y * y, 0.0))
        a, b = max(-h, x0), min(h, x1)
        if b - a > 1.0:
            out.append((y, a, b - a))
        y += spacing_m
    return out


def _transit(base: Position, y: float, xa: float, length: float) -> float:
    return math.hypot(min(max(base.x, xa), xa + length) - base.x, y - base.y)


def band_edges(n: int, R: float, spacing_m: float, base: Position, x0: float | None = None,
               x1: float | None = None) -> tuple[list[float], float]:
    """Split column [x0, x1] (default: the whole circle) into n bands. Returns (n + 1 y-edges, worst cost m)."""
    x0 = -R if x0 is None else x0
    x1 = R if x1 is None else x1
    lines = _lines(R, spacing_m, x0, x1)
    L = len(lines)
    if L == 0:
        return [-R + 2 * R * i / n for i in range(n + 1)], 0.0

    def run(i: int, j: int) -> float:
        y, xa, ln = lines[i]
        return _transit(base, y, xa, ln) + sum(q[2] for q in lines[i:j]) + spacing_m * (j - i - 1)

    def groups(T: float) -> list[int] | None:
        starts, i = [], 0
        while i < L:
            if len(starts) == n:
                return None
            starts.append(i)
            if run(i, i + 1) > T:
                return None
            j = i + 1
            while j < L and run(i, j + 1) <= T:
                j += 1
            i = j
        return starts

    lo, hi = 0.0, run(0, L) + 4 * R
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if groups(mid) is None:
            lo = mid
        else:
            hi = mid
    starts = groups(hi)
    while len(starts) < n:                  # fewer groups than seekers: split the biggest group
        ends = starts[1:] + [L]
        k = max(range(len(starts)), key=lambda q: ends[q] - starts[q])
        if ends[k] - starts[k] < 2:
            break
        starts.insert(k + 1, (starts[k] + ends[k]) // 2)
    ends = starts[1:] + [L]
    worst = max(run(a, b) for a, b in zip(starts, ends))
    edges = [-R] + [-R + spacing_m * s for s in starts[1:]] + [R]
    while len(edges) < n + 1:               # more seekers than lines: duplicate thin bands at the top
        edges.insert(-1, 0.5 * (edges[-2] + edges[-1]))
    return edges, worst


def _shares(n: int, c: int) -> list[int]:
    """Seekers per equal-area column (largest remainder; every column gets at least one)."""
    base = [n // c] * c
    for k in range(n - sum(base)):
        base[(c // 2 + (k + 1) // 2 * (-1) ** k) % c] += 1      # extra seekers go to the middle columns first
    return base


def zone_cells(n: int, R: float, spacing_m: float, base: Position, max_columns: int = 4
               ) -> tuple[list[tuple[float, float, float, float]], float]:
    """Best columns x bands layout: list of (x0, x1, y0, y1) cells (one per seeker) and its worst cost (m)."""
    best = None
    for c in range(1, min(n, max_columns) + 1):
        xe = column_edges(c, R)
        cells, worst = [], 0.0
        for k, m in enumerate(_shares(n, c)):
            ye, w = band_edges(m, R, spacing_m, base, xe[k], xe[k + 1])
            worst = max(worst, w)
            cells += [(xe[k], xe[k + 1], ye[i], ye[i + 1]) for i in range(m)]
        if best is None or worst < best[1] - 1e-6:
            best = (cells, worst)
    return best


def _clip(poly: list[tuple[float, float]], inside, cross) -> list[tuple[float, float]]:
    out = []
    for i, cur in enumerate(poly):
        prev = poly[i - 1]
        if inside(cur):
            if not inside(prev):
                out.append(cross(prev, cur))
            out.append(cur)
        elif inside(prev):
            out.append(cross(prev, cur))
    return out


def _cell_polygon(R: float, x0: float, x1: float, y0: float, y1: float, n: int = 160) -> list[Position]:
    """Circle (radius R) clipped to the rectangle [x0, x1] x [y0, y1] (Sutherland-Hodgman)."""
    poly = [(R * math.cos(2 * math.pi * k / n), R * math.sin(2 * math.pi * k / n)) for k in range(n)]

    def cx(v):
        return lambda p, q: (v, p[1] + (q[1] - p[1]) * (v - p[0]) / (q[0] - p[0]))

    def cy(v):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (v - p[1]) / (q[1] - p[1]), v)

    for inside, cross in ((lambda p: p[0] >= x0, cx(x0)), (lambda p: p[0] <= x1, cx(x1)),
                          (lambda p: p[1] >= y0, cy(y0)), (lambda p: p[1] <= y1, cy(y1))):
        poly = _clip(poly, inside, cross)
        if not poly:
            break
    return [Position(x=x, y=y) for x, y in poly]


class EqualTimeZoneAllocator:
    """ZoneAllocator: one zone per seeker (circle inset by `margin_m`, clipped to its column x band cell)."""

    def __init__(self, spacing_m: float, base: Position, margin_m: float = 6.0, max_columns: int = 4):
        self.spacing_m, self.base, self.margin_m, self.max_columns = spacing_m, base, margin_m, max_columns
        self.worst_m = 0.0

    def allocate(self, arena_radius_m: float, seeker_ids: list[str]) -> dict[str, Zone]:
        R = arena_radius_m - self.margin_m
        cells, self.worst_m = zone_cells(max(len(seeker_ids), 1), R, self.spacing_m, self.base, self.max_columns)
        out = {}
        for i, sid in enumerate(seeker_ids):
            pts = _cell_polygon(R, *cells[i])
            out[sid] = Zone(zone_id=f"Z{i + 1}", boundary=pts, assigned_to=sid)
        return out

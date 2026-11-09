"""Static world for one (map, seed): thermal rasters, fire seeds, decoys and trees. Shared read-only by all worlds.

Coordinates: ENU metres, origin = arena centre. Rasters are indexed [row, col] with row 0 = SOUTH edge
(context.md convention), cell (0, 0) has its south-west corner at (-R, -R).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from sim.config import GameConfig
from sim.maps import MapData, load_map

__all__ = ["FireSeed", "DecoySeed", "Tree", "LakeSeed", "WorldStatic", "build_static"]


@dataclass(frozen=True)
class FireSeed:
    fire_id: str
    kind: str
    x: float
    y: float
    area_m2: float
    ros_m_min: float
    temp_scale: float
    residence_min: float
    water_mult: float
    wind_k: float
    template: int
    scale: float


@dataclass(frozen=True)
class DecoySeed:
    decoy_id: str
    kind: str
    x: float
    y: float
    temp_c: float
    area_m2: float
    yaw_deg: float


@dataclass(frozen=True)
class Tree:
    x: float
    y: float
    h: float
    r: float
    kind: int  # 0 conifer, 1 broadleaf


@dataclass(frozen=True)
class LakeSeed:
    """A lake (refill point) or, on the dry variant of the same map, a dry basin. Shape: a wobbly circle."""

    lake_id: str
    x: float
    y: float
    r: float
    harm: tuple[tuple[float, float], ...]   # (relative amplitude, phase) of harmonics k = 2, 3, 4
    wet: bool

    @property
    def max_r(self) -> float:
        return self.r * (1.0 + sum(abs(a) for a, _ in self.harm))

    def radius_at(self, theta):
        out = 1.0
        for k, (a, ph) in enumerate(self.harm, start=2):
            out = out + a * np.cos(k * theta + ph)
        return self.r * out

    def contains(self, x, y, buffer_m: float = 0.0):
        dx, dy = np.asarray(x) - self.x, np.asarray(y) - self.y
        return np.hypot(dx, dy) < self.radius_at(np.arctan2(dy, dx)) + buffer_m

    def outline(self, n: int = 56) -> list[tuple[float, float]]:
        th = np.linspace(0, 2 * math.pi, n, endpoint=False)
        r = self.radius_at(th)
        return [(float(self.x + rr * math.cos(t)), float(self.y + rr * math.sin(t))) for t, rr in zip(th, r, strict=True)]


@dataclass
class WorldStatic:
    cfg: GameConfig
    map: MapData
    seed: int
    res: float
    n_px: int
    bg: np.ndarray        # °C, (n_px, n_px)
    char: np.ndarray      # °C
    fire_tex: np.ndarray  # °C (real flame peaks + pasted real fire components)
    fires: list[FireSeed]
    decoys: list[DecoySeed]
    trees: list[Tree]
    fuel: np.ndarray      # fire-grid fuel load 0..1
    init_mask: list[tuple[np.ndarray, np.ndarray, np.ndarray]]  # per fire: (rows, cols, burning fraction) on the fire grid
    lakes: list[LakeSeed] = field(default_factory=list)

    def in_lake(self, x: float, y: float, buffer_m: float = 0.0) -> bool:
        return any(bool(lk.contains(x, y, buffer_m)) for lk in self.lakes)

    @property
    def radius(self) -> float:
        return self.cfg.arena_radius_m

    def world_to_px(self, x, y):
        R = self.radius
        return (np.asarray(y) + R) / self.res - 0.5, (np.asarray(x) + R) / self.res - 0.5  # (row, col) float

    def ground_preview(self, cell_m: float = 2.0) -> np.ndarray:
        """Background on exactly the map grid (same cells as coverage / fire snapshots), 0..255."""
        n = int(math.ceil(2 * self.radius / cell_m))
        k = cell_m / self.res
        a = ndimage.zoom(ndimage.gaussian_filter(self.bg, k / 2.5), n / self.n_px, order=1)[:n, :n]
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        return np.clip((a - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8)


def _mosaic(tiles: list[np.ndarray], n: int, rng: np.random.Generator, normalize: bool) -> np.ndarray:
    out = np.empty((n, n), dtype=np.float32)
    ref = float(np.median(np.concatenate([t.ravel() for t in tiles]))) if normalize else 0.0
    r = 0
    while r < n:
        c = 0
        th_row = None
        while c < n:
            t = tiles[int(rng.integers(len(tiles)))]
            if rng.random() < 0.5:
                t = t[::-1]
            if rng.random() < 0.5:
                t = t[:, ::-1]
            if normalize:
                t = t - np.median(t) + ref
            h = min(t.shape[0], n - r)
            w = min(t.shape[1], n - c)
            th_row = h if th_row is None else min(th_row, h)
            out[r:r + h, c:c + w] = t[:h, :w]
            c += w
        r += th_row
    return out


def _place_points(rng, n, R, min_r_from_base, base, min_sep, existing, max_r, clusters=None, cluster_frac=0.0,
                  cluster_r=28.0, forbid=None):
    pts = []
    tries = 0
    while len(pts) < n and tries < 20000:
        tries += 1
        if clusters and rng.random() < cluster_frac:
            cx, cy = clusters[int(rng.integers(len(clusters)))]
            a, d = rng.uniform(0, 2 * math.pi), cluster_r * math.sqrt(rng.random())
            x, y = cx + d * math.cos(a), cy + d * math.sin(a)
        else:
            a, d = rng.uniform(0, 2 * math.pi), max_r * math.sqrt(rng.random())
            x, y = d * math.cos(a), d * math.sin(a)
        if math.hypot(x, y) > max_r or math.hypot(x - base[0], y - base[1]) < min_r_from_base:
            continue
        if any(math.hypot(x - px, y - py) < min_sep for px, py in pts + existing):
            continue
        if forbid is not None and forbid(x, y):
            continue
        pts.append((x, y))
    return pts


def _paste(dst: np.ndarray, src: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    a0, b0 = max(r0, 0), max(c0, 0)
    a1, b1 = min(r0 + h, dst.shape[0]), min(c0 + w, dst.shape[1])
    if a1 <= a0 or b1 <= b0:
        return
    s = (slice(a0 - r0, a1 - r0), slice(b0 - c0, b1 - c0))
    region = dst[a0:a1, b0:b1]
    region[mask[s]] = src[s][mask[s]]


def build_static(cfg: GameConfig, map_id: str, seed: int) -> WorldStatic:
    md = load_map(map_id, cfg.world_res_m)
    rng = np.random.default_rng(seed)
    R, res = cfg.arena_radius_m, md.world_res_m
    n = int(math.ceil(2 * R / res))
    bg = _mosaic([t.bg for t in md.tiles], n, rng, normalize=True)
    char = _mosaic([t.char for t in md.tiles], n, rng, normalize=False)
    fire_tex = _mosaic([t.fire for t in md.tiles], n, rng, normalize=False)
    char = np.maximum(char, bg + 3.0)
    base = cfg.base_xy

    # ---- lakes (own random stream, so the wet and dry variants of a map share one layout) ----------------------
    lakes: list[LakeSeed] = []
    if cfg.n_lakes > 0:
        lrng = np.random.default_rng([seed, 7])
        rmax = cfg.lake_r_m[1] * 1.3
        for i, (x, y) in enumerate(_place_points(lrng, cfg.n_lakes, R, rmax + 70.0, base, 2 * rmax + 40.0, [],
                                                 R - rmax - 25.0)):
            harm = tuple((float(lrng.uniform(0.03, 0.10)), float(lrng.uniform(0, 2 * math.pi))) for _ in range(3))
            lakes.append(LakeSeed(f"L{i + 1}", x, y, float(lrng.uniform(*cfg.lake_r_m)), harm, cfg.lakes_wet))
    in_lake = (lambda x, y, b=8.0: any(bool(lk.contains(x, y, b)) for lk in lakes)) if lakes else None

    # ---- fire seeds -----------------------------------------------------------------------------------------
    clusters = _place_points(rng, cfg.fire_clusters, R, 70.0, base, 60.0, [], R - 45.0, forbid=in_lake)
    fires: list[FireSeed] = []
    taken: list[tuple[float, float]] = []
    k = 0
    for ft in cfg.fires:
        pts = _place_points(rng, ft.count, R, 40.0, base, 11.0, taken, R - 18.0, clusters, 0.55,
                            cluster_r=cfg.cluster_r_m, forbid=in_lake)
        for x, y in pts:
            area = float(rng.uniform(*ft.area_m2))
            ratio = md.tarea / area
            cand = np.nonzero((ratio > 0.3) & (ratio < 3.0))[0]
            ti = int(rng.choice(cand)) if len(cand) else int(np.argmin(np.abs(np.log(ratio))))
            scale = float(np.clip(math.sqrt(area / max(float(md.tarea[ti]), 1e-3)), 0.25, 3.0))
            fires.append(FireSeed(f"F{k + 1:02d}", ft.name, x, y, area, float(rng.uniform(*ft.ros_m_min)),
                                  float(rng.uniform(*ft.temp_scale)), float(rng.uniform(*ft.residence_min)),
                                  ft.water_mult, ft.wind_k, ti, scale))
            taken.append((x, y))
            k += 1

    # ---- paste real fire components into the flame texture + initial burning masks on the fire grid ------------
    cell = cfg.fire_cell_m
    nf = int(math.ceil(2 * R / cell))
    init_masks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for f in fires:
        m = md.tmask[f.template].astype(np.float32)
        t = md.ttemp[f.template].astype(np.float32)
        if abs(f.scale - 1.0) > 1e-3:
            m = ndimage.zoom(m, f.scale, order=1)
            t = ndimage.zoom(t, f.scale, order=1)
        rot = int(rng.integers(4))
        m, t = np.rot90(m, rot), np.rot90(t, rot)
        mb = m >= 0.5
        h, w = mb.shape
        rr, cc = (f.y + R) / res, (f.x + R) / res
        r0, c0 = int(round(rr - h / 2)), int(round(cc - w / 2))
        _paste(fire_tex, t, mb & (t > 120.0), r0, c0)
        # coverage fraction on the fire grid
        k_ = cell / res
        full = np.zeros((nf, nf), dtype=np.float32)
        ys, xs = np.nonzero(mb)
        if len(ys):
            gr = ((r0 + ys + 0.5) / k_).astype(int)
            gc = ((c0 + xs + 0.5) / k_).astype(int)
            ok = (gr >= 0) & (gr < nf) & (gc >= 0) & (gc < nf)
            np.add.at(full, (gr[ok], gc[ok]), 1.0 / (k_ * k_))
        full = np.clip(full, 0.0, 1.0)
        full[full < 0.25] = 0.0
        if not full.any():  # tiny fire: at least its centre cell
            full[int((f.y + R) / cell), int((f.x + R) / cell)] = 1.0
        rr_, cc_ = np.nonzero(full)
        init_masks.append((rr_.astype(np.int32), cc_.astype(np.int32), full[rr_, cc_]))

    # ---- decoys (static hot objects in the ground raster) -------------------------------------------------------
    decoys: list[DecoySeed] = []
    j = 0
    for dt in cfg.decoys:
        pts = _place_points(rng, dt.count, R, 35.0, base, 16.0, taken, R - 20.0, forbid=in_lake)
        for x, y in pts:
            decoys.append(DecoySeed(f"D{j + 1}", dt.name, x, y, float(rng.uniform(*dt.temp_c)),
                                    float(rng.uniform(*dt.area_m2)), float(rng.uniform(0, 360))))
            taken.append((x, y))
            j += 1
    yy, xx = np.mgrid[-60:61, -60:61] * res
    for d in decoys:
        layer = np.zeros(xx.shape, np.float32)
        c, s = math.cos(math.radians(d.yaw_deg)), math.sin(math.radians(d.yaw_deg))
        u, v = xx * c - yy * s, xx * s + yy * c
        if d.kind == "warm_rock":
            r = math.sqrt(d.area_m2 / math.pi)
            shape = np.hypot(u, v * 1.3) < r
            layer[shape] = d.temp_c
        elif d.kind == "vehicle":
            body = (np.abs(u) < 1.0) & (np.abs(v) < 2.25)
            layer[body] = bg.mean() + 6.0
            engine = (np.abs(u) < 0.7) & (v > 1.2) & (v < 2.2)
            layer[engine] = d.temp_c
        else:  # cabin: warm roof + hot chimney
            half = math.sqrt(d.area_m2) / 2
            roof = (np.abs(u) < half) & (np.abs(v) < half)
            layer[roof] = d.temp_c
            chim = (np.abs(u - half * 0.5) < 0.35) & (np.abs(v - half * 0.3) < 0.35)
            layer[chim] = 95.0
        mask = layer > 0
        layer = ndimage.gaussian_filter(np.where(mask, layer, 0), 0.8)
        weight = ndimage.gaussian_filter(mask.astype(np.float32), 0.8)
        h = layer.shape[0]
        r0 = int(round((d.y + R) / res - h / 2))
        c0 = int(round((d.x + R) / res - h / 2))
        a0, b0 = max(r0, 0), max(c0, 0)
        a1, b1 = min(r0 + h, n), min(c0 + h, n)
        if a1 > a0 and b1 > b0:
            sl = (slice(a0 - r0, a1 - r0), slice(b0 - c0, b1 - c0))
            wgt = np.clip(weight[sl], 0, 1)
            val = np.divide(layer[sl], np.maximum(weight[sl], 1e-6))
            bg[a0:a1, b0:b1] = bg[a0:a1, b0:b1] * (1 - wgt) + val * wgt

    # ---- trees (visual + a little thermal shading) -------------------------------------------------------------
    trees: list[Tree] = []
    tries = 0
    while len(trees) < cfg.n_trees and tries < cfg.n_trees * 30:
        tries += 1
        a, d = rng.uniform(0, 2 * math.pi), (R + 30) * math.sqrt(rng.random())
        x, y = d * math.cos(a), d * math.sin(a)
        if math.hypot(x - base[0], y - base[1]) < 28:
            continue
        if any(math.hypot(x - px, y - py) < 7.0 for px, py in taken):
            continue
        if in_lake is not None and in_lake(x, y, 3.0):
            continue
        kind = 0 if rng.random() < 0.78 else 1
        h = float(rng.uniform(9, 24) if kind == 0 else rng.uniform(7, 15))
        trees.append(Tree(x, y, h, float(h * rng.uniform(0.2, 0.28)), kind))

    fuel = 0.75 + 0.25 * ndimage.gaussian_filter(rng.normal(size=(nf, nf)).astype(np.float32), 6) * 6
    yy, xx = (np.mgrid[:nf, :nf] + 0.5) * cell - R
    fuel = np.clip(fuel, 0.35, 1.0).astype(np.float32)
    fuel[np.hypot(xx, yy) > R - 2] = 0.0
    if lakes:
        _paint_lakes(lakes, bg, char, fire_tex, fuel, R, res, cell, rng)
    return WorldStatic(cfg, md, seed, res, n, bg, char, fire_tex, fires, decoys, trees, fuel, init_masks, lakes)


def _paint_lakes(lakes, bg, char, fire_tex, fuel, R, res, cell, rng) -> None:
    """Lakes: no fuel; thermally a smooth surface a little WARMER than the land at night ([VERIFY], context.md 8).
    Dry basins: no fuel, bare mud slightly cooler than the forest floor, little texture."""
    ref = float(np.median(bg))
    for lk in lakes:
        m = lk.max_r + 2.0
        r0, r1 = max(int((lk.y - m + R) / res), 0), min(int((lk.y + m + R) / res) + 1, bg.shape[0])
        c0, c1 = max(int((lk.x - m + R) / res), 0), min(int((lk.x + m + R) / res) + 1, bg.shape[1])
        yy, xx = (np.mgrid[r0:r1, c0:c1] + 0.5) * res - R
        inside = lk.contains(xx, yy)
        edge = ndimage.gaussian_filter(inside.astype(np.float32), 1.5 / res)   # soft shoreline
        sl = (slice(r0, r1), slice(c0, c1))
        if lk.wet:
            surf = ref + 2.5 + 0.25 * ndimage.gaussian_filter(rng.normal(size=inside.shape), 4.0 / res) * 8
        else:
            surf = 0.3 * bg[sl] + 0.7 * (ref - 1.0)
        for a in (bg, char, fire_tex):
            a[sl] = a[sl] * (1 - edge) + surf * edge
        f0, f1 = max(int((lk.y - m + R) / cell), 0), min(int((lk.y + m + R) / cell) + 1, fuel.shape[0])
        g0, g1 = max(int((lk.x - m + R) / cell), 0), min(int((lk.x + m + R) / cell) + 1, fuel.shape[1])
        fy, fx = (np.mgrid[f0:f1, g0:g1] + 0.5) * cell - R
        fuel[f0:f1, g0:g1][lk.contains(fx, fy, 1.0)] = 0.0

"""Simple fire physics: a stochastic cellular automaton on a 0.5 m grid. [ASSUMPTION] numbers, MechE to replace.

Per cell: heat (0..1, "fraction of the cell actively burning"), fuel (0..1), wet (L/m2 of water still on it),
plus the properties of the fire that ignited it (ROS, temperature scale, residence time, water need, wind factor).

  spread    an unburning cell with fuel ignites at rate  sum_nb heat_nb * ROS_nb * wind(dir) / (dist * calib)
            (ROS in m/s). Wetness blocks ignition linearly up to wet_block_l_per_m2.
  burning   heat ramps up to 1, fuel is consumed over the residence time, then heat decays (burnt out -> char)
  water     each L/m2 removes 1 / (suppression_l_per_m2 * water_mult) of heat and leaves wetness that evaporates

Determinism: the random field for step k is drawn from default_rng([seed, k]) over the FULL grid, so identical
worlds (human / swarm / baseline) evolve identically wherever no water has landed. Only windows around active
fire are updated, so cost scales with fire size, not arena size.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from sim.config import GameConfig
from sim.static import WorldStatic

__all__ = ["FireModel"]

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
SPREAD_CALIB = 2.9      # calibrated: a windless circular fire's radius grows at ~ROS (tests/sim/test_fire.py)
RAMP_S = 20.0           # fire-seconds for a new cell to reach full heat
COOL_S = 90.0           # fire-seconds for a burnt-out cell's heat to decay


class FireModel:
    def __init__(self, static: WorldStatic, cfg: GameConfig, seed: int):
        self.cfg, self.static, self.seed = cfg, static, seed
        R, cell = cfg.arena_radius_m, cfg.fire_cell_m
        self.cell = cell
        self.n = n = int(math.ceil(2 * R / cell))
        self.heat = np.zeros((n, n), np.float32)
        self.fuel = static.fuel.copy()
        self.wet = np.zeros((n, n), np.float32)
        self.ignited = np.zeros((n, n), bool)
        self.fire_id = np.full((n, n), -1, np.int16)
        self.ros = np.zeros((n, n), np.float32)        # m/s (real)
        self.tscale = np.zeros((n, n), np.float32)
        self.resid = np.ones((n, n), np.float32)       # fire-seconds
        self.wmult = np.ones((n, n), np.float32)
        self.windk = np.zeros((n, n), np.float32)
        self.step_index = 0
        self.fire_seconds = 0.0
        to = math.radians(cfg.wind_from_deg + 180.0)
        self.wind_dir = (math.sin(to), math.cos(to))   # (east, north) unit vector the wind blows TOWARD
        self.wind_norm = min(cfg.wind_speed_mps / 5.0, 1.5)
        self.water_on_fire_l = 0.0
        self.water_total_l = 0.0
        for i, f in enumerate(static.fires):
            rr, cc, vv = static.init_mask[i]
            self.heat[rr, cc] = np.maximum(self.heat[rr, cc], 0.6 + 0.4 * vv)
            self.ignited[rr, cc] = True
            self.fire_id[rr, cc] = i
            self.ros[rr, cc] = f.ros_m_min / 60.0
            self.tscale[rr, cc] = f.temp_scale
            self.resid[rr, cc] = f.residence_min * 60.0
            self.wmult[rr, cc] = f.water_mult
            self.windk[rr, cc] = f.wind_k
            self.fuel[rr, cc] = np.maximum(self.fuel[rr, cc], 0.6)

    # ------------------------------------------------------------------------------------------------------------
    def _windows(self) -> list[tuple[slice, slice]]:
        active = self.fire_id + 1
        active[(self.heat <= 0) & (self.wet <= 0)] = 0
        objs = ndimage.find_objects(active)
        boxes = []
        for o in objs:
            if o is None:
                continue
            boxes.append([max(o[0].start - 3, 0), min(o[0].stop + 3, self.n),
                          max(o[1].start - 3, 0), min(o[1].stop + 3, self.n)])
        merged = True
        while merged and len(boxes) > 1:   # merge overlapping boxes so no cell is updated twice
            merged = False
            out = []
            while boxes:
                b = boxes.pop()
                for o in boxes:
                    if b[0] < o[1] and o[0] < b[1] and b[2] < o[3] and o[2] < b[3]:
                        o[0], o[1], o[2], o[3] = min(o[0], b[0]), max(o[1], b[1]), min(o[2], b[2]), max(o[3], b[3])
                        merged = True
                        break
                else:
                    out.append(b)
            boxes = out
        return [(slice(b[0], b[1]), slice(b[2], b[3])) for b in boxes]

    def step(self, dt_fire: float) -> None:
        """Advance by dt_fire fire-seconds."""
        cfg = self.cfg
        rand = np.random.default_rng([self.seed, self.step_index]).random((self.n, self.n), dtype=np.float32)
        self.step_index += 1
        self.fire_seconds += dt_fire
        wx, wy = self.wind_dir
        evap = cfg.evaporation_l_per_m2_min * dt_fire / 60.0
        for rs, cs in self._windows():
            heat = self.heat[rs, cs]
            fuel = self.fuel[rs, cs]
            wet = self.wet[rs, cs]
            ign = self.ignited[rs, cs]
            fid = self.fire_id[rs, cs]
            ros, tsc, res, wm, wk = (self.ros[rs, cs], self.tscale[rs, cs], self.resid[rs, cs],
                                     self.wmult[rs, cs], self.windk[rs, cs])
            hp = np.pad(heat, 1)
            props = [np.pad(a, 1) for a in (ros, tsc, res, wm, wk)]
            fidp = np.pad(fid, 1, constant_values=-1)
            h, w = heat.shape
            rate = np.zeros_like(heat)
            best = np.zeros_like(heat)
            src = np.zeros((h, w, 2), np.int32)
            for dr, dc in _NB:
                # neighbour at (r+dr, c+dc) spreads INTO (r, c): direction of spread = (-dc, -dr) in (east, north)
                nb_h = hp[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
                nb_ros = props[0][1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
                nb_wk = props[4][1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
                dist = math.hypot(dr, dc) * self.cell
                cosang = (-dc * wx + -dr * wy) / math.hypot(dr, dc)
                wind = np.maximum(1.0 + nb_wk * self.wind_norm * cosang, 0.08)
                contrib = nb_h * nb_ros * wind / (dist * SPREAD_CALIB)
                rate += contrib
                better = contrib > best
                best = np.where(better, contrib, best)
                src[better] = (dr, dc)
            can = (fuel > 0.05) & (heat <= 0)   # quenched cells with fuel left can rekindle once dry
            block = np.clip(wet / cfg.wet_block_l_per_m2, 0.0, 1.0)
            p = (1.0 - np.exp(-rate * dt_fire)) * (1.0 - block)
            new = can & (rand[rs, cs] < p) & (rate > 0)
            if new.any():
                rr, cc = np.nonzero(new)
                sr, sc = src[rr, cc, 0] + rr + 1, src[rr, cc, 1] + cc + 1
                for a, pa in zip((ros, tsc, res, wm, wk), props):
                    a[rr, cc] = pa[sr, sc]
                fid[rr, cc] = fidp[sr, sc]
                heat[rr, cc] = 0.12
                ign[rr, cc] = True
            burning = heat > 0
            has_fuel = fuel > 0.02
            # ramp up while fuel lasts (suppressed by wetness), consume fuel, decay when out of fuel
            heat_up = burning & has_fuel
            heat[heat_up] = np.minimum(1.0, heat[heat_up] + dt_fire / RAMP_S * (1.0 - block[heat_up]))
            fuel[burning] = np.maximum(0.0, fuel[burning] - heat[burning] * dt_fire / res[burning])
            out = burning & ~has_fuel
            heat[out] = np.maximum(0.0, heat[out] - dt_fire / COOL_S)
            np.subtract(wet, evap, out=wet)
            np.maximum(wet, 0.0, out=wet)

    # ------------------------------------------------------------------------------------------------------------
    def apply_water(self, x: float, y: float, litres: float, agl_m: float) -> float:
        """Drop `litres` centred at (x, y). Returns litres that landed on burning cells."""
        cfg = self.cfg
        if litres <= 0:
            return 0.0
        self.water_total_l += litres
        eff = 1.0 if agl_m <= cfg.drop_max_effective_agl_m else max(0.15, cfg.drop_max_effective_agl_m / agl_m)
        area = litres / cfg.suppression_l_per_m2
        radius = math.sqrt(area / math.pi) + cfg.drop_spread_per_m * max(agl_m, 0.0)
        fall_s = math.sqrt(2.0 * max(agl_m, 1.0) / 9.81)
        x += self.wind_dir[0] * cfg.wind_speed_mps * fall_s * cfg.drop_drift_factor
        y += self.wind_dir[1] * cfg.wind_speed_mps * fall_s * cfg.drop_drift_factor
        R, cell = cfg.arena_radius_m, self.cell
        rc = int(math.ceil(radius * 1.6 / cell)) + 1
        r0, c0 = int((y + R) / cell), int((x + R) / cell)
        a0, a1 = max(r0 - rc, 0), min(r0 + rc + 1, self.n)
        b0, b1 = max(c0 - rc, 0), min(c0 + rc + 1, self.n)
        if a1 <= a0 or b1 <= b0:
            return 0.0
        yy, xx = np.mgrid[a0:a1, b0:b1]
        d2 = ((xx + 0.5) * cell - R - x) ** 2 + ((yy + 0.5) * cell - R - y) ** 2
        wgt = np.exp(-d2 / (2 * (radius / 1.2) ** 2))
        wgt /= wgt.sum() * cell * cell
        lpm2 = (litres * eff * wgt).astype(np.float32)          # L/m2 per cell
        sl = (slice(a0, a1), slice(b0, b1))
        heat = self.heat[sl]
        on_fire = float((lpm2 * (heat > 0)).sum() * cell * cell)
        knock = lpm2 / (cfg.suppression_l_per_m2 * self.wmult[sl])
        heat -= knock
        np.maximum(heat, 0.0, out=heat)
        self.wet[sl] += lpm2
        self.water_on_fire_l += on_fire
        return on_fire

    # ------------------------------------------------------------------------------------------------------------
    def fire_metric(self) -> float:
        """Burning intensity-area, m2 (sum heat x cell area)."""
        return float(self.heat.sum() * self.cell * self.cell)

    def per_fire_metric(self) -> np.ndarray:
        n = len(self.static.fires)
        m = self.fire_id >= 0
        return np.bincount(self.fire_id[m].astype(np.int64), weights=self.heat[m], minlength=n) * self.cell ** 2

    def burnt_fraction(self) -> np.ndarray:
        return np.where(self.ignited, 1.0 - np.clip(self.fuel, 0, 1), 0.0).astype(np.float32)

    def raster(self, cell_m: float, what: str = "heat") -> np.ndarray:
        """Downsample (max) to cell_m for snapshots."""
        k = max(1, int(round(cell_m / self.cell)))
        a = self.heat if what == "heat" else self.burnt_fraction()
        n = self.n // k
        return a[: n * k, : n * k].reshape(n, k, n, k).max((1, 3))

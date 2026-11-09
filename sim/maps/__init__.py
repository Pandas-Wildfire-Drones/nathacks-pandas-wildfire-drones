"""Map loading. Real maps come from FLAME 3 NADIR (`python -m sim.maps.build`); `synth` works anywhere."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from sim.maps.paths import default_data_dir, default_maps_dir

__all__ = ["Tile", "MapData", "available_maps", "load_map", "MAP_TITLES"]

MAP_TITLES = {
    "p1": "Hanna Hammock · test plot",
    "p2": "Hanna Hammock · plot 2",
    "synth": "Synthetic night (no dataset)",
}


@dataclass
class Tile:
    bg: np.ndarray     # °C, float32, at world_res_m
    char: np.ndarray   # °C
    fire: np.ndarray   # °C, real burning-pixel temperatures


@dataclass
class MapData:
    map_id: str
    title: str
    world_res_m: float
    tiles: list[Tile]
    tmask: np.ndarray            # (N, S, S) uint8 real fire component masks
    ttemp: np.ndarray            # (N, S, S) float16 °C crops
    tarea: np.ndarray            # (N,) m2
    meta: dict = field(default_factory=dict)
    real: bool = True


def _map_file(map_id: str) -> Path:
    return default_maps_dir() / f"{map_id}.npz"


def available_maps() -> list[dict]:
    out = []
    for mid in ("p1", "p2"):
        ok = _map_file(mid).exists() or default_data_dir() is not None
        out.append({"id": mid, "title": MAP_TITLES[mid], "available": ok})
    out.append({"id": "synth", "title": MAP_TITLES["synth"], "available": True})
    return out


def _load_npz(path: Path) -> MapData:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    tiles, tm, tt, ta = [], [], [], []
    for i in range(meta["n_tiles"]):
        tiles.append(Tile(z[f"bg_{i}"].astype(np.float32), z[f"char_{i}"].astype(np.float32),
                          z[f"fire_{i}"].astype(np.float32)))
        if len(z[f"tarea_{i}"]):
            tm.append(z[f"tmask_{i}"])
            tt.append(z[f"ttemp_{i}"])
            ta.append(z[f"tarea_{i}"])
    return MapData(meta["map_id"], meta["title"], meta["world_res_m"], tiles,
                   np.concatenate(tm), np.concatenate(tt), np.concatenate(ta), meta, real=True)


def _synthetic(world_res_m: float, seed: int = 7) -> MapData:
    """Fallback for machines without FLAME 3: smooth night ground + synthetic fire textures and blobs."""
    rng = np.random.default_rng(seed)
    n = int(80 / world_res_m)
    bg = 8.0 + 1.6 * ndimage.gaussian_filter(rng.normal(size=(n, n)), 18) * 18
    bg += 0.6 * ndimage.gaussian_filter(rng.normal(size=(n, n)), 2) * 4
    char = bg + 9.0 + 2.0 * ndimage.gaussian_filter(rng.normal(size=(n, n)), 3) * 5
    fire = 380.0 + 140.0 * ndimage.gaussian_filter(rng.normal(size=(n, n)), 4) * 7
    fire = np.clip(fire, 160.0, 620.0)
    s = 128
    N = 40
    tmask = np.zeros((N, s, s), np.uint8)
    ttemp = np.zeros((N, s, s), np.float16)
    tarea = np.zeros(N, np.float32)
    yy, xx = np.mgrid[:s, :s] - s / 2
    for i in range(N):
        r = rng.uniform(4, 30)
        noise = ndimage.gaussian_filter(rng.normal(size=(s, s)), 4) * 12
        m = (np.hypot(xx * rng.uniform(0.6, 1.4), yy) + noise) < r
        tmask[i] = m
        ttemp[i] = np.where(m, np.clip(250 + 300 * ndimage.gaussian_filter(rng.random((s, s)), 3), 120, 600), 0)
        tarea[i] = m.sum() * world_res_m ** 2
    keep = tarea > 0.2
    return MapData("synth", MAP_TITLES["synth"], world_res_m,
                   [Tile(bg.astype(np.float32), char.astype(np.float32), fire.astype(np.float32))],
                   tmask[keep], ttemp[keep], tarea[keep], {"source": "synthetic"}, real=False)


_CACHE: dict[str, MapData] = {}


def load_map(map_id: str, world_res_m: float = 0.15) -> MapData:
    """Load (and on first use, build) a map. Falls back to `synth` if the FLAME 3 data isn't on this machine."""
    if map_id in _CACHE:
        return _CACHE[map_id]
    if map_id == "synth":
        md = _synthetic(world_res_m)
    else:
        path = _map_file(map_id)
        if not path.exists():
            data = default_data_dir()
            if data is None:
                md = _synthetic(world_res_m)
                md.meta["fallback_for"] = map_id
                _CACHE[map_id] = md
                return md
            from sim.maps.build import build_map
            build_map(map_id, data, default_maps_dir(create=True), world_res_m)
        md = _load_npz(path)
    _CACHE[map_id] = md
    return md

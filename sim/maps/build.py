"""FLAME 3 NADIR (Hanna Hammock) -> sim map files ("reverse matched" real thermal data).

Input: the output of `python -m training.build_nadir` (exclude/data/flame3_nadir_prepared): every-3rd-frame
georeferenced radiometric frames (°C), per-pixel masks (0 not fire, 1 fire, 2 ignore) and summary_<stack>.npz
(per-pixel peak, t_peak, post-peak floor). All frames of one stack share one map grid, so each pixel has a history.

Per stack we cut the largest fully valid rectangle (geo-TIFFs are rotated: corners are nodata) and keep:
  bg     per-pixel pre-burn ground temperature = min over the stack (unburned ground, real clutter)
  char   post-peak floor of burned pixels (real char / burnt-out ground), nearest-filled elsewhere
  fire   a texture of REAL burning-pixel temperatures (mask == 1) from the frame with the most fire, nearest-filled
  templates  real fire components (mask + °C crop) from many frames: initial shapes of the sim fires
Everything is resampled to the sim's world resolution (default 0.15 m/px) before saving.

    python -m sim.maps.build                       # builds p1 and p2 into exclude/sim_maps/
    python -m sim.maps.build --data <prepared dir> --out <dir> --maps p1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from sim.maps.paths import default_data_dir, default_maps_dir

__all__ = ["MAP_STACKS", "build_map", "main"]

# map id -> (title, stacks). p1 is the CNN's TEST plot (never trained on): the honest demo map.
MAP_STACKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "p1": ("Hanna Hammock · test plot (FLAME 3 NADIR p1)", ("p1",)),
    "p2": ("Hanna Hammock · plot 2 (FLAME 3 NADIR p2a/p2b)", ("p2a", "p2b")),
}
TEMPLATE_SIZE = 128
MAX_TEMPLATES_PER_STACK = 60


def _largest_rectangle(valid: np.ndarray) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1) of the largest all-True axis-aligned rectangle (histogram method)."""
    h, w = valid.shape
    heights = np.zeros(w, dtype=np.int64)
    best = (0, 0, 0, 0, 0)
    for r in range(h):
        heights = np.where(valid[r], heights + 1, 0)
        stack: list[int] = []
        for c in range(w + 1):
            cur = heights[c] if c < w else 0
            while stack and heights[stack[-1]] >= cur:
                top = stack.pop()
                height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                area = height * (c - left)
                if area > best[0]:
                    best = (area, r - height + 1, r + 1, left, c)
            stack.append(c)
    _, r0, r1, c0, c1 = best
    return r0, r1, c0, c1


def _nearest_fill(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    if known.all():
        return values
    if not known.any():
        raise ValueError("nothing to fill from")
    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    return values[tuple(idx)]


def _load_stack(data: Path, stack: str):
    frames = sorted((data / "frames").glob(f"{stack}_*.npy"))
    if not frames:
        raise FileNotFoundError(f"no frames for stack {stack} in {data / 'frames'}")
    temps = np.stack([np.load(f).astype(np.float32) for f in frames])
    masks = np.stack([np.load(data / "masks" / f.name) for f in frames])
    summary = np.load(data / f"summary_{stack}.npz")
    return temps, masks, summary


def _resample(a: np.ndarray, factor: float, order: int) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3:
        return a
    return ndimage.zoom(a, factor, order=order, mode="nearest", grid_mode=True)


def build_stack(data: Path, stack: str, world_res_m: float, rng: np.random.Generator) -> dict:
    temps, masks, summary = _load_stack(data, stack)
    gsd = float(summary["gsd_m"])
    # nodata pixels were filled with each frame's median -> exactly equal to it. Temperatures are quantized (0.1 C),
    # so real pixels can equal the median too: only median-valued regions connected to the image border are nodata.
    medians = np.median(temps.reshape(len(temps), -1), axis=1)
    fill = np.zeros(temps.shape, dtype=bool)
    for i in range(len(temps)):
        eq = temps[i] == medians[i]
        lab, _ = ndimage.label(eq)
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        border = border[border > 0]
        fill[i] = np.isin(lab, border)
    valid = fill.mean(0) < 0.1
    valid = ndimage.binary_erosion(valid, iterations=4)
    r0, r1, c0, c1 = _largest_rectangle(valid)
    sl = (slice(r0, r1), slice(c0, c1))

    bg = np.where(fill, np.inf, temps).min(0)
    bg = np.where(np.isfinite(bg), bg, np.median(temps)).astype(np.float32)

    peak, floor = summary["peak"], summary["floor"]
    burned = (peak >= 250.0) & valid
    plateau = float(summary["plateau"])
    # Pixels that never cooled below the char plateau during the stack are residual heat (smouldering logs) in
    # the real data. The sim's fire model doesn't know about them, so inpaint them: the only heat sources in the
    # sim are its own fires and decoys. Clutter below the plateau (sun-warmed ground, char) is kept as is.
    hot = ndimage.binary_dilation(bg > plateau + 5.0, iterations=2)
    if hot.any() and (~hot).any():
        bg = _nearest_fill(bg, ~hot)
    cooled = burned & (floor <= plateau + 20.0)   # stacks can end mid-burn: only pixels that reached the char plateau
    if cooled.sum() > 100:
        char = _nearest_fill(np.where(cooled, floor, 0.0).astype(np.float32), cooled)
    else:
        char = np.full(bg.shape, plateau, dtype=np.float32)

    fire_px = (masks == 1) & ~fill
    best = int(fire_px.reshape(len(temps), -1).sum(1).argmax())
    # flame texture: each burned pixel's REAL peak temperature (the flaming moment), nearest-filled elsewhere
    fire = _nearest_fill(np.where(burned, peak, 0.0).astype(np.float32), burned)

    # templates: real fire components, every 3rd prepared frame
    templates_m, templates_t, areas = [], [], []
    for i in range(0, len(temps), 3):
        lab, n = ndimage.label(fire_px[i], structure=np.ones((3, 3)))
        if n == 0:
            continue
        objs = ndimage.find_objects(lab)
        sizes = np.bincount(lab.ravel())
        for k, box in enumerate(objs, start=1):
            if box is None or not (25 <= sizes[k] <= 4000):
                continue
            (a0, a1), (b0, b1) = (box[0].start, box[0].stop), (box[1].start, box[1].stop)
            if a0 < r0 or a1 > r1 or b0 < c0 or b1 > c1:
                continue
            if max(a1 - a0, b1 - b0) > 110:
                continue
            m = (lab[box] == k)
            t = temps[i][box]
            templates_m.append(m)
            templates_t.append(t)
            areas.append(float(sizes[k]) * gsd * gsd)
    if templates_m:
        order = rng.permutation(len(templates_m))[:MAX_TEMPLATES_PER_STACK * 4]
        # keep a spread of sizes
        order = sorted(order, key=lambda j: areas[j])
        pick = np.linspace(0, len(order) - 1, min(MAX_TEMPLATES_PER_STACK, len(order))).astype(int)
        chosen = [order[j] for j in pick]
    else:
        chosen = []

    factor = gsd / world_res_m
    out_m = np.zeros((len(chosen), TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)
    out_t = np.zeros((len(chosen), TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.float16)
    out_area = np.zeros(len(chosen), dtype=np.float32)
    for j, idx in enumerate(chosen):
        m = _resample(templates_m[idx].astype(np.float32), factor, 1) >= 0.5
        t = _resample(templates_t[idx], factor, 1)
        h, w = m.shape
        h, w = min(h, TEMPLATE_SIZE), min(w, TEMPLATE_SIZE)
        oy, ox = (TEMPLATE_SIZE - h) // 2, (TEMPLATE_SIZE - w) // 2
        out_m[j, oy:oy + h, ox:ox + w] = m[:h, :w]
        out_t[j, oy:oy + h, ox:ox + w] = t[:h, :w]
        out_area[j] = float(m.sum()) * world_res_m ** 2

    return {
        "bg": _resample(bg[sl], factor, 1).astype(np.float32),
        "char": _resample(char[sl], factor, 1).astype(np.float16),
        "fire": _resample(fire[sl], factor, 1).astype(np.float16),
        "tmask": out_m, "ttemp": out_t, "tarea": out_area,
        "meta": {"stack": stack, "gsd_m": gsd, "rect": [int(r0), int(r1), int(c0), int(c1)],
                 "frames": int(len(temps)), "fire_frame": best, "templates": len(chosen),
                 "bg_median_c": float(np.median(bg[sl])), "char_median_c": float(np.median(char[sl]))},
    }


def build_map(map_id: str, data: Path, out: Path, world_res_m: float = 0.15, seed: int = 0) -> Path:
    title, stacks = MAP_STACKS[map_id]
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    metas = []
    for i, s in enumerate(stacks):
        d = build_stack(data, s, world_res_m, rng)
        for k in ("bg", "char", "fire", "tmask", "ttemp", "tarea"):
            arrays[f"{k}_{i}"] = d[k]
        metas.append(d["meta"])
    meta = {"map_id": map_id, "title": title, "world_res_m": world_res_m, "n_tiles": len(stacks),
            "source": "FLAME 3 NADIR thermal, Hanna Hammock (Hopkins et al. 2024, cite as in training/FLAME3.md)",
            "stacks": metas}
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{map_id}.npz"
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None, help="flame3_nadir_prepared directory")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--maps", nargs="*", default=list(MAP_STACKS))
    ap.add_argument("--res", type=float, default=0.15)
    a = ap.parse_args(argv)
    data = a.data or default_data_dir()
    if data is None:
        raise SystemExit("FLAME 3 NADIR prepared data not found; pass --data (see training/build_nadir.py)")
    out = a.out or default_maps_dir(create=True)
    for m in a.maps:
        p = build_map(m, data, out, a.res)
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

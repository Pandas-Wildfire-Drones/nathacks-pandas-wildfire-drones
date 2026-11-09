"""FLAME 3 NADIR (Hanna Hammock) -> frames + per-pixel masks labelled from each pixel's OWN HISTORY. No hand labels.

Why this works: the georeferenced thermal TIFFs of one plot are stabilized and share one map grid (dataset README),
so every pixel has a temperature time series through the burn: ~30 C unburned -> 300-500 C as the front passes ->
cooling to a "char plateau" (~45-50 C: black char in the sun is warmer than unburned ground). A frame alone can't
say whether a 70 C pixel is a cooling ember or sun-heated char; its history can.

Per pixel (whole stack):   peak = max over time, t_peak = frame of the peak,
                           floor = 10th percentile of the frames AFTER the peak (where it cools to),
                           burned = peak >= burn_c
Per frame t:               P = running max up to t (what has happened so far), T = current temperature
  FIRE (1)    T >= active_c                                      flaming / glowing now
              burned so far (P >= burn_c) and T - floor >= residual_dc   still well above where it ends up = residual
                                                                  heat (smouldering, cooling embers)
  NOT FIRE(0) never heated so far (P < cold_peak_c) and T < cold_now_c          unburned fuel, trees, ground
              burned, at least settle_frames after its peak, T - floor < out_dc, and floor <= char plateau +
              char_margin_c                                       burnt out: warm char, no combustion
  IGNORE (2)  everything else (pre-heating ahead of the front, mid-cooling, frame borders/nodata)
All thresholds are [ASSUMPTION] defaults chosen from the plot-1 time series (see the notebook); they're CLI options.
Only FIRE above active_c is a plain temperature rule, and it's physical: in daylight the sun can't heat ground to it.

Outputs (out/):  frames/<stack>_<IRX>.npy  float32 C, nodata filled with the frame median
                 masks/<stack>_<IRX>.npy   uint8 0/1/2 (perception.datasets.MASK_IGNORE = 2)
                 summary_<stack>.npz       peak, t_peak, floor maps (for plots)
                 manifest.csv              group = stack + 30-frame time block; split: plot 1 = test, plot 2 = train
Frames are 3-5 s apart, so only every --every-th frame is written (labels still use ALL frames).

    python -m training.build_nadir --root exclude/data/flame3_nadir --out exclude/data/flame3_nadir_prepared
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from perception.datasets import MASK_IGNORE, FrameRecord, write_manifest

__all__ = ["LabelRules", "STACKS", "load_stack", "history_maps", "label_frame", "detect_saturation", "build"]

# (stack id, folder under "Hanna Hammock", split). Plot 2 was flown in two parts with a 9 min pause, each its own
# registration (README), so they are separate stacks.
STACKS = (("p1", "plot 1/duringburn/geo_thermal_tiff_celsius", "test"),
          ("p2a", "plot 2/duringburn/geo_thermal_tiff_celsius/part1", "train"),
          ("p2b", "plot 2/duringburn/geo_thermal_tiff_celsius/part2", "train"))
LOW_GAIN_SATURATION_C = 550.0   # SOURCED: FLAME 3 Table III, Autel XT709 low gain 0-550 C


@dataclass(frozen=True)
class LabelRules:
    active_c: float = 120.0       # FIRE: burning now
    burn_c: float = 250.0         # a pixel whose peak reached this has burned
    residual_dc: float = 25.0     # FIRE: still this far above its own post-peak floor
    cold_peak_c: float = 60.0     # NOT FIRE: never above this so far ...
    cold_now_c: float = 50.0      # ... and below this now
    out_dc: float = 8.0           # NOT FIRE (burnt out): within this of its floor ...
    settle_frames: int = 20       # ... at least this many frames after its peak ...
    char_margin_c: float = 15.0   # ... and its floor is near the scene's char plateau (not a steady smoulder)


def _irx(path: Path) -> str:
    m = re.search(r"IRX_(\d+)", path.name)
    return m.group(1) if m else path.stem


def load_stack(folder: Path) -> tuple[list[Path], np.ndarray, float]:
    """(files, stack (T, H, W) float32 with NaN = nodata, GSD in m from the GeoTIFF transform)."""
    import tifffile

    files = sorted(folder.glob("*.TIFF")) + sorted(folder.glob("*.tif"))
    files = sorted(set(files), key=lambda p: int(_irx(p)) if _irx(p).isdigit() else 0)
    if not files:
        raise SystemExit(f"no TIFFs in {folder}")
    with tifffile.TiffFile(files[0]) as t:
        a, b, _, _, d, e = t.pages[0].tags["ModelTransformationTag"].value[:6]
    gsd = math.sqrt(abs(a * e - b * d))
    st = np.stack([tifffile.imread(f).astype(np.float32) for f in files])
    st[st <= 0] = np.nan   # warped borders are written as 0 C
    return files, st, gsd


def history_maps(st: np.ndarray, rows_per_chunk: int = 64) -> dict[str, np.ndarray]:
    """peak, t_peak, floor (10th pct after the peak) per pixel. Chunked over rows to bound memory."""
    n, h, w = st.shape
    peak = np.full((h, w), np.nan, np.float32)
    t_peak = np.zeros((h, w), np.int32)
    floor = np.full((h, w), np.nan, np.float32)
    idx = np.arange(n)[:, None, None]
    for r0 in range(0, h, rows_per_chunk):
        s = st[:, r0:r0 + rows_per_chunk]
        valid = np.isfinite(s).any(0)
        filled = np.where(np.isfinite(s), s, -np.inf)
        tp = filled.argmax(0)
        pk = np.where(valid, filled.max(0), np.nan)
        after = np.where(idx > tp[None], s, np.nan)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fl = np.nanpercentile(after, 10, axis=0)
        fl = np.where(np.isfinite(fl), fl, pk)   # peak in the last frame: no "after" yet
        peak[r0:r0 + rows_per_chunk], t_peak[r0:r0 + rows_per_chunk], floor[r0:r0 + rows_per_chunk] = pk, tp, fl
    return {"peak": peak, "t_peak": t_peak, "floor": floor}


def char_plateau(maps: dict[str, np.ndarray], rules: LabelRules) -> float:
    burned = maps["peak"] >= rules.burn_c
    return float(np.nanmedian(maps["floor"][burned])) if burned.any() else float("nan")


def label_frame(t: int, temp: np.ndarray, running_max: np.ndarray, maps: dict[str, np.ndarray], plateau: float,
                rules: LabelRules = LabelRules()) -> np.ndarray:
    """uint8 mask (0 not fire, 1 fire, 2 ignore) for frame t. temp/running_max may hold NaN (nodata -> ignore)."""
    floor, t_peak = maps["floor"], maps["t_peak"]
    with np.errstate(invalid="ignore"):
        burned_so_far = running_max >= rules.burn_c
        excess = temp - floor
        fire = (temp >= rules.active_c) | (burned_so_far & (excess >= rules.residual_dc))
        unburned = (running_max < rules.cold_peak_c) & (temp < rules.cold_now_c)
        burnt_out = (burned_so_far & (t - t_peak >= rules.settle_frames) & (excess < rules.out_dc)
                     & (floor <= plateau + rules.char_margin_c))
    m = np.full(temp.shape, MASK_IGNORE, np.uint8)
    m[(unburned | burnt_out) & ~fire] = 0
    m[fire] = 1
    m[~np.isfinite(temp)] = MASK_IGNORE
    return m


def detect_saturation(temp: np.ndarray, min_px: int = 20, high_gain_max_c: float = 260.0) -> float:
    """High-gain frames clip hard (~197 C on this camera): many pixels sit exactly at the max."""
    mx = float(np.nanmax(temp))
    n_at = int((temp >= mx - 0.05).sum())
    return mx if (n_at >= min_px and mx < high_gain_max_c) else LOW_GAIN_SATURATION_C


def build(root: str | Path, out: str | Path, every: int = 3, block: int = 30, rules: LabelRules = LabelRules(),
          stacks=STACKS) -> list[FrameRecord]:
    root, out = Path(root), Path(out)
    base = root / "Hanna Hammock" if (root / "Hanna Hammock").exists() else root
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    recs: list[FrameRecord] = []
    info = {"rules": asdict(rules), "every": every, "block": block, "stacks": {}}
    for sid, folder, split in stacks:
        files, st, gsd = load_stack(base / folder)
        maps = history_maps(st)
        plateau = char_plateau(maps, rules)
        np.savez_compressed(out / f"summary_{sid}.npz", plateau=plateau, gsd_m=gsd, **maps)
        running = np.full(st.shape[1:], -np.inf, np.float32)
        counts = np.zeros(3, np.int64)
        for t in range(len(files)):
            running = np.fmax(running, st[t])
            if t % every:
                continue
            temp = st[t]
            mask = label_frame(t, temp, running, maps, plateau, rules)
            counts += np.bincount(mask.ravel(), minlength=3)[:3]
            filled = np.where(np.isfinite(temp), temp, np.nanmedian(temp)).astype(np.float32)
            name = f"{sid}_{_irx(files[t])}"
            np.save(out / "frames" / f"{name}.npy", filled)
            np.save(out / "masks" / f"{name}.npy", mask)
            recs.append(FrameRecord(frame_path=str(out / "frames" / f"{name}.npy"),
                                    mask_path=str(out / "masks" / f"{name}.npy"), group=f"nadir_{sid}_b{t // block:02d}",
                                    label=int((mask == 1).any()), split=split, source=f"flame3_nadir:{sid}",
                                    gsd_m=round(gsd, 4), saturation_c=detect_saturation(filled), timestamp=float(t)))
        info["stacks"][sid] = {"frames": len(files), "written": int(np.ceil(len(files) / every)), "gsd_m": gsd,
                               "char_plateau_c": plateau, "split": split,
                               "pixel_share": dict(zip(("not_fire", "fire", "ignore"), (counts / counts.sum()).round(4).tolist()))}
        print(sid, info["stacks"][sid], flush=True)
    write_manifest(out / "manifest.csv", recs)
    (out / "build_info.json").write_text(json.dumps(info, indent=2))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="exclude/data/flame3_nadir")
    ap.add_argument("--out", default="exclude/data/flame3_nadir_prepared")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--block", type=int, default=30, help="frames per CV group (time block)")
    for f, v in asdict(LabelRules()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()
    rules = LabelRules(**{f: getattr(a, f) for f in asdict(LabelRules())})
    recs = build(a.root, a.out, a.every, a.block, rules)
    print(f"{len(recs)} frames -> {a.out}/manifest.csv")


if __name__ == "__main__":
    main()

"""Build manifest.csv for the FLAME 3 computer-vision subset (Sycan Marsh, DJI M30T), from the files as shipped.

    python -m training.build_manifest_flame3 --root exclude/data/flame3_sycan_marsh_raw --out exclude/data/flame3/manifest.csv
    python -m training.build_manifest_flame3 --root exclude/data/flame3_sycan_marsh_raw --inspect   # print, write nothing

LAYOUT (checked on the download, 2026-11-07). Each sample is a quartet with the SAME file stem in four folders:
    <root>/{Fire,No Fire}/Thermal/Celsius TIFF/00001.TIFF   640x512 float, degrees C          -> frame_path
                          Thermal/Raw JPG/00001.JPG         palette image (never used for training)
                          RGB/Raw/00001.JPG                 4000x3000 wide camera, carries the DJI XMP below
                          RGB/Corrected FOV/00001.JPG       640x512 crop of the raw RGB matched to the thermal FOV
  * Stems restart in each class folder (Fire/00001 and No Fire/00001 are different photos), so pairing must use the
    folder AND the stem.
  * Stems are NOT in time order: the numbering is shuffled (Spearman(stem, time) ~ 0). Time comes from EXIF.

WHAT IS READ PER FRAME
  * EXIF DateTimeOriginal (local time) -> `timestamp` (used for ordering, flights and prev/next frames).
  * DJI XMP of RGB/Raw: RelativeAltitude (m above TAKEOFF), GimbalPitch/Yaw (vs horizon / north), LRFTarget* (laser
    rangefinder: slant distance, lat/lon and altitude of the ground point at the image centre).
  * View geometry: pitch_deg = GimbalPitchDegree; agl_m = RelativeAltitude - LRFTargetAlt (height above the ground the
    image centre looks at; the LRF agrees with pitch x height to 0.01 deg). Without an LRF return, the day's median
    terrain offset is used instead. focal_px: --focal-px (M30T thermal, see FLAME3.md "Cameras, gain and GSD").
    Every frame is OBLIQUE (pitch -46..+12 deg, median -15): GSD varies row by row (perception.datasets.gsd_rows).
    gsd_m is written as the CENTRE-row value for reference only.
  * Gain mode from the TIFF itself. The M30T clips at two levels (measured on this subset): high gain floor -40.6 /
    clip 187.4 C, low gain floor -22.8 / clip 608.7 C. A frame showing either floor or clip gets that gain; others take
    the nearest frame in time (same day, within --gain-gap-s). -> saturation_c.

GROUPS (splits and CV folds never cut a group). One group = one DAY x one --cell-m ground cell of the image-centre
ground point, so frames looking at the same ground on the same day stay together (consecutive frames are 3 s apart
and near-duplicates). A Fire and a No-Fire frame of the same ground on the same day share a group.

LABELS = the folder (Fire 1 / No Fire 0). FLAME 3's folders come from a max-temperature heuristic plus manual review
(training/FLAME3.md): trust "No Fire" as backgrounds; "Fire" says the frame contains fire, not which blob.

Also writes frames_meta.csv next to the manifest: every raw value above plus tmin/tmax, saturated-pixel counts,
lat/lon, and the corrected-FOV RGB path. Use it for dataset review (issue #8) and plots.

The NADIR plot set (Hanna Hammock, Autel) has another layout and camera: add a sibling adapter, don't bend this one.

Dataset citation (requested by the FLAME 3 authors; also in training/FLAME3.md):
  Hopkins, B., ONeill, L., Marinaccio, M., Rowell, E., Parsons, R., Flanary, S., Nazim I, Seielstad C, Afghah, F.
  (2024). FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management.
  arXiv preprint arXiv:2412.02831.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from perception.datasets import FrameRecord, write_manifest

__all__ = ["ShotMeta", "read_xmp", "read_capture_time", "detect_gain", "scan", "assign_gains", "build_records"]

CLASS_DIRS = {"Fire": 1, "No Fire": 0}
THERMAL_SUB = Path("Thermal") / "Celsius TIFF"
RGB_RAW_SUB = Path("RGB") / "Raw"
RGB_FOV_SUB = Path("RGB") / "Corrected FOV"

# M30T thermal, 640x512. DJI lists DFOV 61 deg -> focal = 409.8 px / tan(30.5 deg) = 696 px (HFOV 49.4 deg), which
# reproduces the paper's 17.6 cm/px at 122 m. Measured on this subset: the corrected-FOV RGB covers 0.65 of the wide
# camera's width -> ~684-690 px. EXIF 9.1 mm / 12 um would give 758 px; the two measurements disagree with it.
M30T_FOCAL_PX = 696.0
GAINS = {  # name: (floor_c, clip_c) measured on the Sycan subset: thousands of pixels pile up exactly here
    "high": (-40.6, 187.4),
    "low": (-22.8, 608.7),
}
_XMP_KEYS = ("RelativeAltitude", "AbsoluteAltitude", "GimbalPitchDegree", "GimbalYawDegree", "GimbalRollDegree",
             "FlightPitchDegree", "FlightRollDegree", "FlightYawDegree", "GpsLatitude", "GpsLongitude",
             "LRFTargetDistance", "LRFTargetLat", "LRFTargetLon", "LRFTargetAlt", "LRFTargetAbsAlt")


@dataclass
class ShotMeta:
    cls: str
    stem: str
    thermal: Path
    rgb_raw: Path | None
    rgb_fov: Path | None
    time: dt.datetime | None
    xmp: dict[str, float] = field(default_factory=dict)
    tmin: float = math.nan
    tmax: float = math.nan
    p999: float = math.nan
    n_at_max: int = 0
    gain: str = ""

    @property
    def label(self) -> int:
        return CLASS_DIRS[self.cls]

    @property
    def lrf_ok(self) -> bool:
        d = self.xmp.get("LRFTargetDistance", 0.0)
        return 0.0 < d < 1500.0


def read_xmp(path: Path, keys=_XMP_KEYS) -> dict[str, float]:
    """DJI XMP attributes (dji:Key="value") from the first 256 kB of a JPEG. Missing keys are left out."""
    raw = path.read_bytes()[:262144]
    i, j = raw.find(b"<x:xmpmeta"), raw.find(b"</x:xmpmeta>")
    if i < 0 or j < 0:
        return {}
    x = raw[i:j].decode("utf-8", "replace")
    out = {}
    for k in keys:
        m = re.search(r'dji:%s="([^"]*)"' % k, x) or re.search(r"<dji:%s>([^<]*)<" % k, x)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                pass
    return out


def read_capture_time(path: Path) -> dt.datetime | None:
    """EXIF DateTimeOriginal (falls back to DateTime). Camera local time, no timezone."""
    from PIL import Image

    try:
        ex = Image.open(path).getexif()
        s = ex.get_ifd(0x8769).get(36867) or ex.get(306)
        return dt.datetime.strptime(str(s).strip(), "%Y:%m:%d %H:%M:%S") if s else None
    except Exception:
        return None


def detect_gain(tmin: float, tmax: float, gains=GAINS, tol: float = 0.05) -> str:
    """Gain mode from the floor/clip values a frame touches; '' if it touches neither."""
    for name, (floor, clip) in gains.items():
        if abs(tmin - floor) <= tol or abs(tmax - clip) <= tol:
            return name
    hi = max(gains.values(), key=lambda g: g[1])
    lo_clip = min(g[1] for g in gains.values())
    if tmax > lo_clip + 1.0:          # hotter than the lower clip -> must be the wide-range mode
        return next(n for n, g in gains.items() if g == hi)
    return ""


def scan(root: str | Path, with_stats: bool = True) -> list[ShotMeta]:
    import tifffile

    root = Path(root)
    out: list[ShotMeta] = []
    for cls in CLASS_DIRS:
        tdir = root / cls / THERMAL_SUB
        if not tdir.is_dir():
            raise FileNotFoundError(f"{tdir} missing: is --root the folder that holds 'Fire' and 'No Fire'?")
        for t in sorted(p for p in tdir.iterdir() if p.suffix.lower() in (".tif", ".tiff")):
            def pair(sub: Path) -> Path | None:
                for ext in (".JPG", ".jpg", ".jpeg", ".JPEG"):
                    q = root / cls / sub / (t.stem + ext)
                    if q.exists():
                        return q
                return None

            raw, fov = pair(RGB_RAW_SUB), pair(RGB_FOV_SUB)
            src = raw or fov
            s = ShotMeta(cls, t.stem, t, raw, fov, read_capture_time(src) if src else None,
                         read_xmp(raw) if raw else {})
            if with_stats:
                a = tifffile.imread(t).astype(np.float32)
                s.tmin, s.tmax = float(a.min()), float(a.max())
                s.p999 = float(np.percentile(a, 99.9))
                s.n_at_max = int((a >= s.tmax - 0.05).sum())
                s.gain = detect_gain(s.tmin, s.tmax)
            out.append(s)
    return out


def assign_gains(shots: list[ShotMeta], gap_s: float = 120.0) -> None:
    """Fill unknown gains per SESSION: consecutive frames (all classes) no more than gap_s apart. Gain is a camera
    setting, so an unknown frame takes its session's gain; if the session switched gain, the nearest known frame's."""
    timed = sorted((s for s in shots if s.time), key=lambda s: s.time)
    sessions: list[list[ShotMeta]] = []
    for s in timed:
        if sessions and (s.time - sessions[-1][-1].time).total_seconds() <= gap_s:
            sessions[-1].append(s)
        else:
            sessions.append([s])
    for ses in sessions:
        known = [s for s in ses if s.gain]
        if not known:
            continue
        for s in ses:
            if not s.gain:
                s.gain = min(known, key=lambda k: abs((k.time - s.time).total_seconds())).gain


def _enu(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    return (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0)), (lat - lat0) * 110_540.0


def build_records(shots: list[ShotMeta], focal_px: float = M30T_FOCAL_PX, cell_m: float = 250.0,
                  source: str = "flame3_sycan", height_px: int = 512) -> tuple[list[FrameRecord], list[dict]]:
    """FrameRecords (+ one meta dict per frame), sorted by capture time."""
    shots = sorted(shots, key=lambda s: (s.time or dt.datetime.min, s.cls, s.stem))
    def lrf_terrain(s: ShotMeta) -> float | None:
        """Ground height (vs takeoff) under the image centre from the LRF, if plausible. The LRF sometimes returns
        a far ridge or nonsense (e.g. -233 m); flat-ground GSD is meaningless then, so fall back to the day median."""
        x = s.xmp
        if not (s.lrf_ok and "LRFTargetAlt" in x and "RelativeAltitude" in x):
            return None
        t = x["LRFTargetAlt"]
        return t if -30.0 <= t <= 0.8 * x["RelativeAltitude"] else None

    by_day: dict[str, list[float]] = defaultdict(list)
    for s in shots:
        if s.time and lrf_terrain(s) is not None:
            by_day[s.time.date().isoformat()].append(lrf_terrain(s))
    terrain = {d: float(np.median(v)) for d, v in by_day.items()}
    lats = [s.xmp["GpsLatitude"] for s in shots if "GpsLatitude" in s.xmp]
    lons = [s.xmp["GpsLongitude"] for s in shots if "GpsLongitude" in s.xmp]
    lat0, lon0 = (math.floor(min(lats) * 100) / 100, math.floor(min(lons) * 100) / 100) if lats else (0.0, 0.0)
    t0 = min((s.time for s in shots if s.time), default=None)
    sat = {n: c for n, (_, c) in GAINS.items()}

    recs, meta = [], []
    for s in shots:
        day = s.time.date().isoformat() if s.time else "nodate"
        x = s.xmp
        pitch = x.get("GimbalPitchDegree")
        rel = x.get("RelativeAltitude")
        lt = lrf_terrain(s)
        if lt is not None:
            agl, agl_src = rel - lt, "lrf"
        elif rel is not None:
            agl, agl_src = rel - terrain.get(day, 0.0), "day_median_terrain" if s.lrf_ok is False else "lrf_rejected"
        else:
            agl, agl_src = None, "none"
        if agl is not None and agl < 5.0:
            agl, agl_src = None, f"{agl_src}:implausible"
        # ground point of the image centre: LRF if it returned, else flat-ground projection, else the drone itself
        if s.lrf_ok and "LRFTargetLat" in x:
            e, n = _enu(x["LRFTargetLat"], x["LRFTargetLon"], lat0, lon0)
        elif "GpsLatitude" in x:
            e, n = _enu(x["GpsLatitude"], x["GpsLongitude"], lat0, lon0)
            if agl and pitch is not None and pitch < -3:
                d = min(agl / math.tan(math.radians(-pitch)), 1000.0)
                yaw = math.radians(x.get("GimbalYawDegree", 0.0))
                e, n = e + d * math.sin(yaw), n + d * math.cos(yaw)
        else:
            e = n = 0.0
        group = f"{day}_c{int(e // cell_m):+d}{int(n // cell_m):+d}"
        geom = agl is not None and pitch is not None
        rec = FrameRecord(
            frame_path=str(s.thermal), group=group, label=s.label, source=source,
            agl_m=round(agl, 2) if geom else None, pitch_deg=pitch if geom else None,
            focal_px=focal_px if geom else None, saturation_c=sat.get(s.gain),
            rgb_path=str(s.rgb_raw or s.rgb_fov or ""),
            timestamp=(s.time - t0).total_seconds() if s.time and t0 else None)
        if geom:
            rec = replace(rec, gsd_m=round(rec.gsd_at(height_px / 2, height_px), 4))
        recs.append(rec)
        meta.append({"frame_path": str(s.thermal), "class": s.cls, "stem": s.stem, "group": group,
                     "time": s.time.isoformat() if s.time else "", "gain": s.gain, "saturation_c": sat.get(s.gain, ""),
                     "tmin": round(s.tmin, 2), "tmax": round(s.tmax, 2), "p99_9": round(s.p999, 2),
                     "n_px_at_max": s.n_at_max, "agl_m": rec.agl_m if geom else "", "agl_source": agl_src,
                     "gsd_centre_m": rec.gsd_m if geom else "", "east_m": round(e, 1), "north_m": round(n, 1),
                     "rgb_raw": str(s.rgb_raw or ""), "rgb_fov": str(s.rgb_fov or ""),
                     **{k: x.get(k, "") for k in _XMP_KEYS}})
    return recs, meta


def _write_meta(path: Path, meta: list[dict], base: Path) -> None:
    def rel(p: str) -> str:
        if not p:
            return ""
        q = Path(p).resolve()
        return str(q.relative_to(base)) if q.is_relative_to(base) else str(q)

    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta[0]))
        w.writeheader()
        for m in meta:
            w.writerow({**m, **{k: rel(m[k]) for k in ("frame_path", "rgb_raw", "rgb_fov")}})


def summarize(recs: list[FrameRecord], meta: list[dict]) -> str:
    lines = [f"{len(recs)} frames, {len({r.group for r in recs})} groups"]
    for lab, name in ((1, "Fire"), (0, "No Fire")):
        rs = [r for r in recs if r.label == lab]
        ms = [m for m in meta if m["class"] == name]
        if not rs:
            continue
        pitch = np.array([r.pitch_deg for r in rs if r.pitch_deg is not None])
        agl = np.array([r.agl_m for r in rs if r.agl_m is not None])
        gsd = np.array([r.gsd_m for r in rs if r.gsd_m is not None])
        tmax = np.array([m["tmax"] for m in ms], dtype=float)
        lines += [f"  {name}: {len(rs)} frames, {len({r.group for r in rs})} groups, gains {dict(Counter(m['gain'] or '?' for m in ms))}",
                  f"    pitch deg  min/median/max {pitch.min():.1f} / {np.median(pitch):.1f} / {pitch.max():.1f}",
                  f"    agl m      min/median/max {agl.min():.1f} / {np.median(agl):.1f} / {agl.max():.1f}"
                  f"   (sources {dict(Counter(m['agl_source'] for m in ms))})",
                  f"    centre GSD m/px  5%/median/95% {np.percentile(gsd, 5):.3f} / {np.median(gsd):.3f} / {np.percentile(gsd, 95):.3f}",
                  f"    max T C    min/median/max {tmax.min():.1f} / {np.median(tmax):.1f} / {tmax.max():.1f};"
                  f"  in 80-200: {int(((tmax >= 80) & (tmax < 200)).sum())}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder holding 'Fire' and 'No Fire'")
    ap.add_argument("--out", help="manifest.csv to write (frames_meta.csv goes next to it)")
    ap.add_argument("--focal-px", type=float, default=M30T_FOCAL_PX)
    ap.add_argument("--cell-m", type=float, default=250.0, help="ground cell size for groups")
    ap.add_argument("--gain-gap-s", type=float, default=120.0)
    ap.add_argument("--source", default="flame3_sycan")
    ap.add_argument("--inspect", action="store_true", help="print the summary, write nothing")
    a = ap.parse_args()
    shots = scan(a.root)
    assign_gains(shots, a.gain_gap_s)
    recs, meta = build_records(shots, a.focal_px, a.cell_m, a.source)
    print(summarize(recs, meta))
    if a.inspect or not a.out:
        return
    out = Path(a.out)
    write_manifest(out, recs)
    _write_meta(out.with_name("frames_meta.csv"), meta, out.parent.resolve())
    print(f"wrote {out} and {out.with_name('frames_meta.csv')}")


if __name__ == "__main__":
    main()

"""Synthetic radiometric night thermal frames with ground-truth fire masks.

Why: (1) run the whole train -> export -> evaluate loop before real data arrives; (2) the "sim-generated test set"
of context.md section 10 (altitude/GSD, sub-pixel spots, decoys that real datasets lack); (3) domain randomization.
Shared with workstream E (thermal-frame synthesizer). Every physical number is an [ASSUMPTION] unless noted.

Physics kept on purpose:
  * SUB-PIXEL DILUTION: each pixel mixes fire and ground in RADIANCE (Planck at 10 um, LWIR band centre),
    then converts back to temperature. A 0.05 m^2 ember at 500 C in a 7 cm pixel reads far below 500 C.
  * Night background: ambient +- smooth terrain variation; water bodies a few C warmer than land at night [VERIFY].
  * Decoys: warm rocks (25-45 C), vehicle engines/exhaust (40-90 C: these pass a 60 C absolute threshold!),
    animals/people (30-38 C). The verifier must learn to reject them.
  * Optics: Gaussian PSF (sigma 0.6 px), NETD 50 mK noise (ThermalCameraSpec), saturation at 550 C.

CLI:  python -m perception.synth --out exclude/data/synth --flights 12 --frames-per-flight 40 --seed 0
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

__all__ = ["SynthObject", "SynthParams", "planck_radiance", "radiance_to_temp", "synth_frame", "random_objects",
           "generate_dataset"]

from .radiometry import planck_radiance, radiance_to_temp  # noqa: E402  (re-exported for callers)


@dataclass(frozen=True)
class SynthObject:
    kind: str           # "fire" | "rock" | "vehicle" | "animal"
    u: float
    v: float
    area_m2: float
    temp_c: float       # core temperature
    aspect: float = 1.0
    angle_deg: float = 0.0

    @property
    def is_fire(self) -> bool:
        return self.kind == "fire"


@dataclass(frozen=True)
class SynthParams:
    height_px: int = 256
    width_px: int = 256
    gsd_m: float = 0.072          # 640 px, 32 deg HFOV at 80 m AGL (context.md 6.5)
    ambient_c: float = 8.0
    terrain_sigma_c: float = 1.5
    netd_c: float = 0.05
    psf_sigma_px: float = 0.6
    saturation_c: float = 550.0
    supersample: int = 8


def _background(rng, p: SynthParams) -> np.ndarray:
    h, w = p.height_px, p.width_px
    smooth = ndimage.gaussian_filter(rng.normal(size=(h, w)), sigma=rng.uniform(8, 30))
    smooth = smooth / (smooth.std() + 1e-9) * p.terrain_sigma_c
    fine = ndimage.gaussian_filter(rng.normal(size=(h, w)), 1.5)
    bg = p.ambient_c + smooth + 0.3 * fine / (fine.std() + 1e-9)
    if rng.random() < 0.25:  # a pond/stream: slightly warmer than land at night [VERIFY]
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(10, 50)
        bg = np.where((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2, p.ambient_c + rng.uniform(2, 5), bg)
    return bg.astype(np.float64)


def _render_object(obj: SynthObject, p: SynthParams):
    """Return (r0, c0, fill_fraction, mean_radiance_inside) for the pixel window around obj."""
    s = p.supersample
    area_px = obj.area_m2 / p.gsd_m ** 2
    a = math.sqrt(area_px * obj.aspect / math.pi)   # semi-axes in pixels
    b = math.sqrt(area_px / (obj.aspect * math.pi))
    rad = int(math.ceil(max(a, b))) + 2
    r0, c0 = int(math.floor(obj.v)) - rad, int(math.floor(obj.u)) - rad
    n = 2 * rad + 1
    sub = (np.arange(n * s) + 0.5) / s
    yy, xx = np.meshgrid(sub + r0, sub + c0, indexing="ij")
    th = math.radians(obj.angle_deg)
    dx, dy = xx - obj.u, yy - obj.v
    xr, yr = dx * math.cos(th) + dy * math.sin(th), -dx * math.sin(th) + dy * math.cos(th)
    rr2 = (xr / max(a, 1e-6)) ** 2 + (yr / max(b, 1e-6)) ** 2
    inside = rr2 <= 1.0
    if obj.is_fire:  # hot core, cooler rim (~40% of core temp at the edge)
        t = obj.temp_c * (0.4 + 0.6 * (1 - np.clip(rr2, 0, 1)))
    else:
        t = np.full(rr2.shape, obj.temp_c)
    radiance = np.where(inside, planck_radiance(t), 0.0)
    fill = inside.reshape(n, s, n, s).mean(axis=(1, 3))
    rad_sum = radiance.reshape(n, s, n, s).sum(axis=(1, 3)) / (s * s)
    return r0, c0, fill, rad_sum


def synth_frame(rng: np.random.Generator, objects: list[SynthObject], p: SynthParams = SynthParams()):
    """Returns (temp_c float32 HxW, fire_mask bool HxW). Mask = pixels with any fire fill (fill > 0)."""
    h, w = p.height_px, p.width_px
    bg = _background(rng, p)
    radiance = planck_radiance(bg)
    mask = np.zeros((h, w), dtype=bool)
    for obj in objects:
        r0, c0, fill, rad_in = _render_object(obj, p)
        n = fill.shape[0]
        rs, cs = slice(max(r0, 0), min(r0 + n, h)), slice(max(c0, 0), min(c0 + n, w))
        if rs.start >= rs.stop or cs.start >= cs.stop:
            continue
        fr = slice(rs.start - r0, rs.stop - r0)
        fc = slice(cs.start - c0, cs.stop - c0)
        f, ri = fill[fr, fc], rad_in[fr, fc]
        radiance[rs, cs] = radiance[rs, cs] * (1 - f) + ri  # radiance mixing = sub-pixel dilution
        if obj.is_fire:
            mask[rs, cs] |= f > 0
    radiance = ndimage.gaussian_filter(radiance, p.psf_sigma_px)
    temp = radiance_to_temp(radiance) + rng.normal(0, p.netd_c, size=(h, w))
    return np.clip(temp, -40, p.saturation_c).astype(np.float32), mask


def random_objects(rng: np.random.Generator, p: SynthParams, fire_prob: float = 0.6) -> list[SynthObject]:
    objs: list[SynthObject] = []
    m = 12

    def pos():
        return float(rng.uniform(m, p.width_px - m)), float(rng.uniform(m, p.height_px - m))

    if rng.random() < fire_prob:
        for _ in range(rng.integers(1, 4)):
            u, v = pos()
            area = float(np.exp(rng.uniform(np.log(0.01), np.log(4.0))))  # embers to ~4 m^2 spot fires
            objs.append(SynthObject("fire", u, v, area, float(rng.uniform(150, 700)), float(rng.uniform(1, 2.5)),
                                    float(rng.uniform(0, 180))))
    for _ in range(rng.poisson(1.2)):
        u, v = pos()
        kind = rng.choice(["rock", "vehicle", "animal"], p=[0.5, 0.25, 0.25])
        if kind == "rock":
            o = SynthObject("rock", u, v, float(rng.uniform(0.3, 6)), float(rng.uniform(25, 45)), float(rng.uniform(1, 2)))
        elif kind == "vehicle":
            o = SynthObject("vehicle", u, v, float(rng.uniform(1, 8)), float(rng.uniform(40, 90)), 2.2,
                            float(rng.uniform(0, 180)))
        else:
            o = SynthObject("animal", u, v, float(rng.uniform(0.2, 1.0)), float(rng.uniform(30, 38)), 2.0,
                            float(rng.uniform(0, 180)))
        objs.append(o)
    return objs


def generate_dataset(out_dir: str | Path, flights: int = 12, frames_per_flight: int = 40, seed: int = 0,
                     size_px: int = 256, agl_range_m: tuple[float, float] = (25.0, 150.0)) -> Path:
    """Writes frames/*.npy, masks/*.npy and manifest.csv. Each 'flight' has its own altitude and ambient,
    so group-wise splits behave like real flights. Altitude spans close-range confirmation (25-40 m) up to
    high search (150 m) with the 640 px / 32 deg seeker camera, so one scale-normalized model covers both."""
    from .datasets import FrameRecord, write_manifest

    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    recs = []
    for f in range(flights):
        agl = float(rng.uniform(*agl_range_m))
        gsd = 2 * agl * math.tan(math.radians(16)) / 640
        p = SynthParams(height_px=size_px, width_px=size_px, gsd_m=gsd, ambient_c=float(rng.uniform(-2, 16)),
                        terrain_sigma_c=float(rng.uniform(0.5, 3.0)))
        for i in range(frames_per_flight):
            temp, mask = synth_frame(rng, random_objects(rng, p), p)
            stem = f"f{f:03d}_{i:04d}"
            np.save(out / "frames" / f"{stem}.npy", temp)
            np.save(out / "masks" / f"{stem}.npy", mask.astype(np.uint8))
            recs.append(FrameRecord(frame_path=str(out / "frames" / f"{stem}.npy"), group=f"synth-flight-{f:03d}",
                                    mask_path=str(out / "masks" / f"{stem}.npy"), label=int(mask.any()),
                                    source="synth", gsd_m=gsd))
    write_manifest(out / "manifest.csv", recs)
    return out / "manifest.csv"


def _main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic radiometric thermal dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flights", type=int, default=12)
    ap.add_argument("--frames-per-flight", type=int, default=40)
    ap.add_argument("--size-px", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agl-range", type=float, nargs=2, default=(25.0, 150.0), metavar=("MIN_M", "MAX_M"))
    a = ap.parse_args()
    print(generate_dataset(a.out, a.flights, a.frames_per_flight, a.seed, a.size_px, tuple(a.agl_range)))


if __name__ == "__main__":
    _main()

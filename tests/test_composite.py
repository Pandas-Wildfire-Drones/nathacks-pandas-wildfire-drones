"""training.composite: synthetic fires on real (here: stand-in) No-Fire backgrounds."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from perception.datasets import FrameRecord, load_frame, load_manifest, load_mask, write_manifest
from perception.synth import SynthObject, planck_radiance, radiance_to_temp
from training.composite import band_of, build_composites, composite


def _background(seed: int, h: int = 96, w: int = 96, ambient: float = 10.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (ambient + ndimage.gaussian_filter(rng.normal(size=(h, w)), 4) * 3).astype(np.float32)


def test_uniform_object_mixes_in_radiance_and_leaves_rest_untouched():
    bg = _background(0)
    rock = SynthObject("rock", 48.0, 48.0, area_m2=1.0, temp_c=40.0)          # 100 px at 0.1 m GSD
    temp, mask, peaks = composite(bg, [rock], gsd_m=0.1, psf_sigma_px=0.0)
    assert not mask.any()                                                      # decoys never enter the fire mask
    assert temp[48, 48] == pytest.approx(40.0, abs=1e-3)                       # fully covered pixel = object temp
    assert peaks[0] == pytest.approx(40.0, abs=1e-3)
    far = np.ones_like(mask)
    far[30:67, 30:67] = False
    assert np.array_equal(temp[far], bg[far])                                  # bit-identical away from the object
    # an edge pixel is a RADIANCE mix of rock and ground, not a temperature mix
    r, c = np.argwhere((temp != bg) & (np.abs(temp - 40.0) > 0.5))[0]
    lo, hi = sorted((float(bg[r, c]), 40.0))
    assert lo < temp[r, c] < hi


def test_subpixel_fire_is_diluted_and_masked():
    bg = _background(1)
    ember = SynthObject("fire", 40.3, 55.7, area_m2=0.002, temp_c=500.0)     # 0.2 of a 0.01 m^2 pixel
    temp, mask, peaks = composite(bg, [ember], gsd_m=0.1)
    assert mask.any() and mask.sum() <= 4
    assert bg.max() < peaks[0] < 500.0
    # the dilution follows Planck: at most the brightest case of a fully-hot fill of 0.2
    upper = radiance_to_temp(0.8 * planck_radiance(float(bg.max())) + 0.2 * planck_radiance(500.0))
    assert peaks[0] <= upper + 1e-6


def test_band_of():
    assert band_of(50) == "<80" and band_of(80) == "80-200" and band_of(199.9) == "80-200" and band_of(250) == ">200"
    assert band_of(math.nan) == "nan"


def _manifest(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    recs = []
    frames = {"cool_a": (_background(2), 0), "cool_b": (_background(3), 0),
              "hot_nofire": (_background(4) + 100, 0), "fire": (_background(5), 1)}
    for i, (name, (arr, label)) in enumerate(frames.items()):
        np.save(src / f"{name}.npy", arr)
        recs.append(FrameRecord(frame_path=str(src / f"{name}.npy"), group=f"burn-{i % 2}", label=label, source="flame3"))
    write_manifest(src / "manifest.csv", recs)
    return src / "manifest.csv"


def test_build_composites_end_to_end(tmp_path: Path):
    man = _manifest(tmp_path)
    s = build_composites(man, tmp_path / "out", per_background=3, gsd_m=0.1, seed=0)
    assert s["backgrounds"] == {"used": 2, "too_hot": 1, "labelled_fire": 1}
    recs = load_manifest(s["manifest"])
    comps = [r for r in recs if r.source == "composite:flame3"]
    assert len(comps) == 6 and len(recs) == 8                                   # + the 2 backgrounds as negatives
    assert {r.group for r in recs} <= {"burn-0", "burn-1"}                      # groups preserved -> no split leak
    assert all(r.gsd_m == 0.1 for r in recs)                                    # consistent -> scale-normalized patches
    for r in comps:                                                             # no --decoys -> every composite has fire
        m = load_mask(r)
        assert r.label == 1 and m.any()
        assert load_frame(r)[m].max() > float(np.load(Path(man).parent / "cool_a.npy").min())
    with (tmp_path / "out" / "objects.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows and all(row["band"] in ("<80", "80-200", ">200") for row in rows if row["kind"] == "fire")
    assert sum(s["fires_by_observed_band"].values()) == sum(row["kind"] == "fire" for row in rows)


def test_build_composites_is_deterministic_and_needs_gsd(tmp_path: Path):
    man = _manifest(tmp_path)
    a = build_composites(man, tmp_path / "a", per_background=2, gsd_m=0.1, decoys=True, seed=7)
    b = build_composites(man, tmp_path / "b", per_background=2, gsd_m=0.1, decoys=True, seed=7)
    assert a["fires_by_observed_band"] == b["fires_by_observed_band"]
    for fa in sorted((tmp_path / "a" / "frames").glob("*.npy")):
        assert np.array_equal(np.load(fa), np.load(tmp_path / "b" / "frames" / fa.name))
    with pytest.raises(ValueError, match="gsd"):
        build_composites(man, tmp_path / "c")


def test_composites_feed_patch_builder(tmp_path: Path):
    from training.patch_data import build_patch_set

    s = build_composites(_manifest(tmp_path), tmp_path / "out", per_background=4, gsd_m=0.1, seed=1)
    ps = build_patch_set(load_manifest(s["manifest"]))
    assert ps.y.sum() > 0 and (ps.y == 0).sum() > 0

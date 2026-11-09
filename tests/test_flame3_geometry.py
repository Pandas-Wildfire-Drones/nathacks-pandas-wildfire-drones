"""Oblique view geometry, the FLAME 3 manifest builder, oblique composites and label-tool pairing/clipping."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from perception.datasets import FrameRecord, depression_rows, load_manifest, write_manifest
from perception.patches import extract_patches, per_centre_gsd
from perception.synth import SynthObject


# ---------- geometry ----------

def test_nadir_geometry_matches_pinhole_gsd():
    rec = FrameRecord(frame_path="x.npy", group="g", label=0, agl_m=80.0, pitch_deg=-90.0, focal_px=700.0)
    gsd, sin = rec.gsd_rows(512)
    assert gsd[256] == pytest.approx(80.0 / 700.0, rel=1e-3)          # centre: agl / focal
    assert np.all(gsd >= gsd[256] - 1e-12) and np.all(sin <= 1.0)    # off-axis rows are a bit coarser, never finer


def test_oblique_geometry_centre_rows_and_horizon_clamp():
    rec = FrameRecord(frame_path="x.npy", group="g", label=0, agl_m=60.0, pitch_deg=-15.0, focal_px=696.0)
    dep = np.degrees(depression_rows(512, rec.pitch_deg, rec.focal_px))
    assert dep[256] == pytest.approx(15.0, abs=0.1) and dep[-1] > dep[0]   # lower rows look further down
    gsd, sin = rec.gsd_rows(512)
    assert gsd[256] == pytest.approx(60.0 / math.sin(math.radians(15.0)) / 696.0, rel=0.01)
    assert gsd[-1] < gsd[256] < gsd[0]                                     # coarser toward the horizon
    assert dep[0] < 0 and np.isfinite(gsd).all()                           # above the horizon: clamped, finite
    assert rec.gsd_at(256.4, 512) == pytest.approx(gsd[256])
    assert FrameRecord(frame_path="x", group="g", label=0, gsd_m=0.1).gsd_at(10, 512) == 0.1
    assert FrameRecord(frame_path="x", group="g", label=0).gsd_rows(512) is None


def test_manifest_roundtrip_new_columns(tmp_path: Path):
    (tmp_path / "rgb").mkdir()
    rec = FrameRecord(frame_path=str(tmp_path / "a.npy"), group="g", label=1, agl_m=55.5, pitch_deg=-20.0,
                      focal_px=696.0, saturation_c=187.4, rgb_path=str(tmp_path / "rgb" / "a.jpg"), timestamp=12.0)
    write_manifest(tmp_path / "m.csv", [rec])
    assert "rgb/a.jpg" in (tmp_path / "m.csv").read_text()                # stored relative
    back = load_manifest(tmp_path / "m.csv")[0]
    assert back.has_geometry and back.saturation_c == 187.4 and back.timestamp == 12.0
    assert Path(back.rgb_path) == tmp_path / "rgb" / "a.jpg"


def test_per_centre_gsd_in_patches():
    t = np.full((64, 64), 10.0, np.float32)
    t[30:34, 30:34] = 300.0
    a = extract_patches(t, [(32, 32), (32, 32)], gsd_m=[0.05, 0.2])
    assert a.shape[0] == 2 and not np.allclose(a[0], a[1])               # different ground windows
    assert np.allclose(extract_patches(t, [(32, 32)], gsd_m=0.05)[0], a[0])
    with pytest.raises(ValueError):
        per_centre_gsd([0.1], 2)


# ---------- manifest builder on a fake FLAME 3 tree ----------

def _jpeg(path: Path, when: str, xmp: dict[str, float] | None = None) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.zeros((30, 40, 3), np.uint8))
    ex = Image.Exif()
    ex[306] = when
    img.save(path, exif=ex)
    if xmp:   # appended after the JPEG data; read_xmp scans raw bytes, like the DJI APP1 block
        attrs = " ".join(f'dji:{k}="{v:+.3f}"' for k, v in xmp.items())
        with path.open("ab") as fh:
            fh.write(f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description {attrs}/></x:xmpmeta>'.encode())


def _tiff(path: Path, arr: np.ndarray) -> None:
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, arr.astype(np.float32))


def _fake_flame3(root: Path) -> None:
    base = dict(RelativeAltitude=70.0, GimbalPitchDegree=-20.0, GimbalYawDegree=0.0, GpsLatitude=42.85,
                GpsLongitude=-121.15, LRFTargetDistance=180.0, LRFTargetLat=42.8515, LRFTargetLon=-121.15,
                LRFTargetAlt=8.0)
    frames = {   # (class, stem): (time, tmin, tmax)
        ("Fire", "00001"): ("2022:10:25 17:00:09", -40.6, 187.4),   # high gain, clipped
        ("Fire", "00002"): ("2022:10:25 17:00:03", 2.0, 450.0),     # hotter than the high-gain clip -> low gain
        ("No Fire", "00001"): ("2022:10:25 17:00:06", 3.0, 25.0),   # unknown gain -> its session's nearest frame
    }
    for (cls, stem), (when, lo, hi) in frames.items():
        a = np.full((16, 20), 10.0)
        a[0, 0], a[-1, -1] = lo, hi
        _tiff(root / cls / "Thermal" / "Celsius TIFF" / f"{stem}.TIFF", a)
        _jpeg(root / cls / "RGB" / "Raw" / f"{stem}.JPG", when, base)
        _jpeg(root / cls / "RGB" / "Corrected FOV" / f"{stem}.JPG", when)


def test_build_manifest_flame3_layout_pairing_gain_geometry(tmp_path: Path):
    from training.build_manifest_flame3 import assign_gains, build_records, scan

    _fake_flame3(tmp_path / "raw")
    shots = scan(tmp_path / "raw")
    assert len(shots) == 3
    assigned = {(s.cls, s.stem): s.gain for s in shots}
    assert assigned[("Fire", "00001")] == "high" and assigned[("Fire", "00002")] == "low"
    assert assigned[("No Fire", "00001")] == ""
    assign_gains(shots)
    nofire = next(s for s in shots if s.cls == "No Fire")
    assert nofire.gain in ("high", "low")                                 # nearest frame (3 s either side)
    recs, meta = build_records(shots, height_px=16)
    assert [r.timestamp for r in recs] == [0.0, 3.0, 6.0]                 # sorted by capture time, not by stem
    for r in recs:                                                        # stems collide across classes: pair by folder
        cls = "No Fire" if r.label == 0 else "Fire"
        assert f"/{cls}/RGB/Raw/" in r.rgb_path and Path(r.rgb_path).stem == Path(r.frame_path).stem
        assert r.agl_m == pytest.approx(62.0) and r.pitch_deg == -20.0 and r.has_geometry
    assert len({r.group for r in recs}) == 1                              # same ground, same day
    assert {r.saturation_c for r in recs if r.label == 1} == {187.4, 608.7}
    assert all(m["agl_source"] == "lrf" for m in meta)


# ---------- oblique composites ----------

def test_to_pixel_equivalent_foreshortens_vertically():
    from training.composite import to_pixel_equivalent

    disc = SynthObject("fire", 10, 10, area_m2=1.0, temp_c=400.0, aspect=1.0, angle_deg=0.0)
    assert to_pixel_equivalent(disc, 0.1, 1.0, 0.1) == disc
    o = to_pixel_equivalent(disc, 0.2, 0.25, 0.1)
    area_px = o.area_m2 / 0.1 ** 2
    assert area_px == pytest.approx(1.0 / 0.2 ** 2 * 0.25, rel=1e-6)     # (1 m^2 / gsd^2) x sin(depression)
    assert o.aspect == pytest.approx(4.0, rel=1e-6)                       # squashed into a flat ellipse...
    assert min(o.angle_deg, 180 - o.angle_deg) == pytest.approx(0.0, abs=1e-6)   # ...lying horizontally


def test_oblique_composites_stay_on_ground_and_use_clip_mode(tmp_path: Path):
    from training.composite import build_composites, ground_pixels

    rng = np.random.default_rng(0)
    bg = (12.0 + rng.normal(0, 0.3, (128, 160))).astype(np.float32)
    bg[:20] = -40.6                                                       # sky band at the top
    np.save(tmp_path / "bg.npy", bg)
    rec = FrameRecord(frame_path=str(tmp_path / "bg.npy"), group="g0", label=0, agl_m=60.0, pitch_deg=-12.0,
                      focal_px=170.0, saturation_c=187.4)
    write_manifest(tmp_path / "m.csv", [rec])
    valid = ground_pixels(bg, rec)
    rows = np.unique(valid // 160)
    dep = np.degrees(depression_rows(128, -12.0, 170.0))
    assert rows.min() >= 20 and dep[rows].min() >= 10.0                   # not sky, not near the horizon
    s = build_composites(tmp_path / "m.csv", tmp_path / "out", per_background=4, seed=1, core_range_c=(400, 600),
                         area_range_m2=(2.0, 4.0))
    import csv
    objs = list(csv.DictReader((tmp_path / "out" / "objects.csv").open()))
    assert objs and all(int(float(o["v"])) in set(rows.tolist()) for o in objs)
    assert max(float(o["observed_peak_c"]) for o in objs) > 187.4          # camera clip (550), not the frame's
    build_composites(tmp_path / "m.csv", tmp_path / "out2", per_background=4, seed=1, core_range_c=(400, 600),
                     area_range_m2=(2.0, 4.0), clip_mode="frame")
    objs2 = list(csv.DictReader((tmp_path / "out2" / "objects.csv").open()))
    assert max(float(o["observed_peak_c"]) for o in objs2) <= 187.4 + 1e-3
    back = load_manifest(s["manifest"])
    assert all(r.has_geometry for r in back)                              # geometry travels to the patch builder


# ---------- label tool: pairing from the manifest, clipped blobs, ordering ----------

def test_label_tool_uses_manifest_rgb_skips_clipped_and_orders_by_time(tmp_path: Path):
    from training.label_tool import find_candidates, load_candidates, write_candidates

    recs = []
    for k, (cls, ts) in enumerate((("Fire", 20.0), ("Fire", 10.0), ("No Fire", 15.0))):
        d = tmp_path / cls
        d.mkdir(exist_ok=True)
        t = np.full((64, 64), 10.0, np.float32)
        if cls == "Fire":
            t[20:23, 20:23] = 150.0                                       # in band
            t[40:43, 45:48] = 187.4                                       # at the high-gain clip
        np.save(d / "00001.npy" if k != 1 else d / "00002.npy", t)
        fp = d / ("00001.npy" if k != 1 else "00002.npy")
        recs.append(FrameRecord(frame_path=str(fp), group="g0", label=int(cls == "Fire"), saturation_c=187.4,
                                rgb_path=str(d / f"{fp.stem}.jpg"), timestamp=ts))
    write_manifest(tmp_path / "m.csv", recs)
    cands = find_candidates(load_manifest(tmp_path / "m.csv"), rgb_box=(0.1, 0.1, 0.9, 0.9))
    assert len(cands) == 2 and all(c.peak_c < 187.0 for c in cands)       # clipped blobs skipped
    assert all(c.rgb_path.endswith(".jpg") for c in cands)
    assert len({c.cand_id for c in cands}) == 2
    late = next(c for c in cands if c.frame_path.endswith("Fire/00001.npy"))
    assert late.prev_path.endswith("No Fire/00001.npy") and late.next_path == ""   # time order, not file order
    with_clip = find_candidates(load_manifest(tmp_path / "m.csv"), include_clipped=True)
    assert len(with_clip) == 4
    write_candidates(tmp_path / "lab" / "c.csv", cands)
    assert load_candidates(tmp_path / "lab" / "c.csv")[0].rgb_box == pytest.approx((0.1, 0.1, 0.9, 0.9))

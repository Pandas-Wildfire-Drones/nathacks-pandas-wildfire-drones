"""Perception unit tests. Need numpy + scipy; ONNX tests skip when onnxruntime/torch are missing."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from contracts import CameraModel, GridSpec, Pose, Position, ThermalFrame  # noqa: E402
from perception.detectors import (  # noqa: E402
    ConstantClassifier, FakeDetector, FusedDetector, ThresholdDetector, TruthSource,
)
from perception.evaluation import (  # noqa: E402
    average_precision, binary_metrics, choose_threshold_recall_first, gt_objects_from_mask, match_detections, roc_auc,
)
from perception.georef import FlatGroundGeoreferencer, TerrainGeoreferencer, footprint_corners, world_to_pixel  # noqa: E402
from perception.mapping import LogOddsMap  # noqa: E402
from perception.patches import PATCH_SPEC, extract_patch, extract_patches, normalize_patch  # noqa: E402
from perception.pipeline import SeekerPerception  # noqa: E402
from perception.results_doc import update_section  # noqa: E402
from perception.synth import SynthObject, SynthParams, planck_radiance, radiance_to_temp, synth_frame  # noqa: E402
from perception.tracking import HotspotTracker, Observation  # noqa: E402

CAM = CameraModel(width_px=640, height_px=512, hfov_deg=32.0)


def nadir_pose(x=0.0, y=0.0, z=80.0, yaw=0.0) -> Pose:
    return Pose(position=Position(x=x, y=y, z=z), yaw_deg=yaw)


# ---------- georeferencing ----------

def test_nadir_centre_pixel_hits_point_below():
    p = FlatGroundGeoreferencer().pixel_to_world(320, 256, nadir_pose(10, 20), CAM)
    assert p is not None and abs(p.x - 10) < 1e-6 and abs(p.y - 20) < 1e-6 and p.z == 0.0


def test_image_top_is_north_and_right_is_east_at_yaw0():
    g = FlatGroundGeoreferencer()
    top = g.pixel_to_world(320, 0, nadir_pose(), CAM)
    right = g.pixel_to_world(640, 256, nadir_pose(), CAM)
    assert top.y > 0 and abs(top.x) < 1e-6
    assert right.x > 0 and abs(right.y) < 1e-6
    assert math.isclose(2 * right.x, CAM.swath_m(80), rel_tol=1e-9)  # matches CameraModel footprint math


def test_yaw_90_rotates_image_top_to_east():
    top = FlatGroundGeoreferencer().pixel_to_world(320, 0, nadir_pose(yaw=90), CAM)
    assert top.x > 0 and abs(top.y) < 1e-6


@pytest.mark.parametrize("pose", [nadir_pose(5, -3, 100, 37), Pose(position=Position(x=0, y=0, z=90), yaw_deg=200,
                                                                     pitch_deg=4, roll_deg=-3)])
def test_world_to_pixel_round_trip(pose):
    g = FlatGroundGeoreferencer()
    for u, v in [(100.5, 50.5), (320, 256), (600, 500)]:
        w = g.pixel_to_world(u, v, pose, CAM)
        uv = world_to_pixel(w, pose, CAM)
        assert uv is not None and abs(uv[0] - u) < 1e-6 and abs(uv[1] - v) < 1e-6


def test_terrain_georef_matches_flat_on_flat_terrain():
    flat = FlatGroundGeoreferencer(ground_z=12.0).pixel_to_world(500, 100, nadir_pose(z=92), CAM)
    terr = TerrainGeoreferencer(lambda x, y: 12.0, step_m=0.5).pixel_to_world(500, 100, nadir_pose(z=92), CAM)
    assert terr.distance_to(flat) < 1e-6


def test_forward_camera_above_horizon_returns_none():
    cam = CameraModel(width_px=640, height_px=512, hfov_deg=32.0, mount_pitch_deg=0.0)
    assert FlatGroundGeoreferencer().pixel_to_world(320, 0, nadir_pose(), cam) is None


# ---------- patches ----------

def test_extract_patch_is_centred_and_padded():
    img = np.zeros((10, 10), np.float32)
    img[5, 5] = 100.0
    p = extract_patch(img, 5, 5, 8)
    assert p.shape == (8, 8) and p[4, 4] == 100.0
    assert extract_patch(img, 0, 0, 8).shape == (8, 8)  # edge padding
    assert normalize_patch(p).shape == (2, 8, 8)
    assert extract_patches(img, []).shape == (0, 2, PATCH_SPEC.size_px, PATCH_SPEC.size_px)


# ---------- synthetic frames + threshold detector ----------

def test_subpixel_dilution_reads_cooler():
    mixed = radiance_to_temp(0.2 * planck_radiance(500.0) + 0.8 * planck_radiance(8.0))
    assert 8.0 < mixed < 500.0 * 0.5


def test_threshold_detector_finds_fire_and_proposes_decoy():
    rng = np.random.default_rng(0)
    p = SynthParams()
    objs = [SynthObject("fire", 60, 60, 0.5, 400), SynthObject("rock", 180, 180, 3.0, 35)]
    temp, mask = synth_frame(rng, objs, p)
    blobs = ThresholdDetector().blobs(temp)
    gts = gt_objects_from_mask(mask)
    m = match_detections([(b.u, b.v) for b in blobs], gts)
    assert len(gts) == 1 and m["tp"] == 1 and m["fn"] == 0
    assert m["fp"] == 1  # the warm rock is proposed: rejecting it is the verifier's job


def test_threshold_detector_quiet_on_empty_night():
    temp, _ = synth_frame(np.random.default_rng(1), [], SynthParams())
    assert ThresholdDetector().blobs(temp) == []


def test_fused_detector_threshold_and_hot_bypass():
    rng = np.random.default_rng(0)
    temp, _ = synth_frame(rng, [SynthObject("fire", 60, 60, 1.0, 500), SynthObject("rock", 180, 180, 3.0, 35)],
                          SynthParams())
    frame = ThermalFrame("s1", 0.0, temp, nadir_pose(), CAM)
    reject_all = FusedDetector(ConstantClassifier(p=0.0, threshold=0.5))
    dets = reject_all.detect(frame)
    assert len(dets) == 1 and dets[0].peak_temp_c >= 300  # only the bypassed hot fire survives
    assert FusedDetector(ConstantClassifier(p=0.0), hot_bypass_c=None).detect(frame) == []
    assert len(FusedDetector(ConstantClassifier(p=1.0)).detect(frame)) == 2


# ---------- metrics ----------

def test_binary_metrics_known_values():
    m = binary_metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert math.isclose(m["precision"], 2 / 3) and math.isclose(m["recall"], 2 / 3) and math.isclose(m["f1"], 2 / 3)


def test_recall_first_threshold():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    s = np.array([.9, .8, .7, .6, .65, .3, .2, .1, .05, .5])
    r = choose_threshold_recall_first(y, s, 0.8)
    assert r["met_target"] and r["recall"] >= 0.8 and r["threshold"] == 0.6
    full = choose_threshold_recall_first(y, s, 1.0)
    assert full["recall"] == 1.0 and full["threshold"] == 0.05


def test_ap_and_auc_perfect_ranking():
    y, s = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    assert average_precision(y, s) == 1.0 and roc_auc(y, s) == 1.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_object_matching_counts_duplicates_separately():
    mask = np.zeros((50, 50), bool)
    mask[10:13, 10:13] = True
    mask[40, 40] = True
    gts = gt_objects_from_mask(mask)
    m = match_detections([(11.5, 11.5), (12, 11), (25, 25)], gts)
    assert (m["tp"], m["fp"], m["fn"], m["duplicates"]) == (1, 1, 1, 1)


# ---------- tracking + mapping + pipeline ----------

def test_tracker_merges_repeat_detections_and_estimates_growth():
    tr = HotspotTracker("s1", gate_m=5)
    for k in range(5):
        hs = tr.update([Observation(Position(x=100 + 0.3 * k, y=50), 0.8, 400, 1.0 + 0.5 * k, timestamp=20.0 * k)])
    assert len(tr.tracks) == 1 and hs[0].observations == 5 and hs[0].hotspot_id == "s1-H1"
    assert hs[0].confidence > 0.8 and hs[0].growth_rate_m2_per_min == pytest.approx(1.5, rel=0.05)
    tr.update([Observation(Position(x=300, y=50), 0.6, 300, 1.0, 120.0)])
    assert len(tr.tracks) == 2


def test_log_odds_map_hits_and_misses():
    grid = GridSpec(origin=Position(x=-100, y=-100), resolution_m=2.0, rows=100, cols=100)
    m = LogOddsMap(grid)
    pose = nadir_pose()
    fp = footprint_corners(pose, CAM, FlatGroundGeoreferencer())
    m.update(fp, [(Position(x=10, y=10), 0.9)], now=1.0)
    hit = grid.world_to_cell(Position(x=10, y=10))
    miss = grid.world_to_cell(Position(x=-10, y=-10))
    far = grid.world_to_cell(Position(x=-90, y=-90))
    assert m.log_odds[hit.row, hit.col] > 0 > m.log_odds[miss.row, miss.col]
    assert m.log_odds[far.row, far.col] == 0 and np.isnan(m.last_seen[far.row, far.col])
    ups = m.pop_updates()
    assert ups and m.pop_updates() == []
    other = LogOddsMap(grid)
    other.apply_updates(ups)
    assert np.allclose(other.log_odds, m.log_odds)


def test_fake_detector_deterministic_and_pipeline_end_to_end():
    truth = [TruthSource("F1", Position(x=5, y=5), 2.0, 450.0), TruthSource("D1", Position(x=-5, y=-5), 4.0, 35.0, False)]

    def run():
        det = FakeDetector(lambda now: truth, seed=7, p_decoy=0.0, clutter_rate=0.0)
        grid = GridSpec(origin=Position(x=-200, y=-200), resolution_m=2.0, rows=200, cols=200)
        perc = SeekerPerception("s1", det, grid=grid)
        out = []
        for k in range(10):
            out = perc.process(ThermalFrame("s1", float(k), np.zeros((0, 0)), nadir_pose(), CAM))
        return out, perc

    a, perc = run()
    b, _ = run()
    assert [h.model_dump() for h in a] == [h.model_dump() for h in b]
    assert len(perc.tracker.tracks) == 1
    h = perc.tracker.hotspots()[0]
    assert h.position.distance_to(Position(x=5, y=5), horizontal=True) < 1.0


def test_results_doc_section_replace(tmp_path: Path):
    doc = tmp_path / "RESULTS.md"
    update_section(doc, "a", "first", {"x": 1})
    update_section(doc, "b", "other")
    update_section(doc, "a", "second")
    text = doc.read_text()
    assert "second" in text and "first" not in text and "other" in text
    assert (tmp_path / "results" / "a.json").exists()


# ---------- ONNX round trip (optional deps) ----------

def test_cnn_onnx_export_and_fused_inference(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    import json

    from perception.detectors import OnnxCnnClassifier
    from training.model import TinyFireNet
    from training.train_cnn import check_onnx_parity, export_onnx

    torch.manual_seed(0)
    model = TinyFireNet()
    path = tmp_path / "m.onnx"
    export_onnx(model, path)
    x = np.random.default_rng(0).normal(10, 3, (8, 32, 32)).astype(np.float32)
    assert check_onnx_parity(model, path, x) < 1e-3
    path.with_suffix(".json").write_text(json.dumps({"model_kind": "cnn", "patch_spec": PATCH_SPEC.to_dict(),
                                                     "threshold": 0.5}))
    clf = OnnxCnnClassifier(path)
    temp, _ = synth_frame(np.random.default_rng(0), [SynthObject("fire", 60, 60, 1.0, 400)], SynthParams())
    probs = clf.predict_proba(temp, [(60, 60), (100, 100), (10, 200)])
    assert probs.shape == (3,) and np.all((probs >= 0) & (probs <= 1))


# ---------- mask audit (training/check_masks.py) ----------

def test_mask_audit_flags_threshold_masks_but_not_physical_masks():
    from training.check_masks import threshold_fit, verdict

    rng = np.random.default_rng(3)
    p = SynthParams()
    thr_rows, phys_rows = [], []
    for k in range(12):
        temp, mask = synth_frame(rng, [SynthObject("fire", 60 + 5 * k, 80, 0.3 + 0.2 * k, 250 + 30 * k),
                                       SynthObject("rock", 180, 180, 3.0, 35)], p)
        thr_rows.append(threshold_fit(temp, temp >= 60.0))
        phys_rows.append(threshold_fit(temp, mask))
    assert verdict(thr_rows).startswith("THRESHOLD-GENERATED")
    assert not verdict(phys_rows).startswith("THRESHOLD-GENERATED")

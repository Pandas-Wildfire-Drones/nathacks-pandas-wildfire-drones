"""Perception unit tests. Need numpy + scipy; ONNX tests skip when onnxruntime/torch are missing."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from contracts import CameraModel, GridSpec, HotspotStatus, Pose, Position, ThermalFrame  # noqa: E402
from perception.detectors import (  # noqa: E402
    ConstantClassifier, ExclusionZone, FakeDetector, FusedDetector, HotBypassRule, ThresholdDetector, TruthSource,
)
from perception.evaluation import (  # noqa: E402
    average_precision, binary_metrics, choose_threshold_recall_first, gt_objects_from_mask, match_detections, roc_auc,
)
from perception.georef import FlatGroundGeoreferencer, TerrainGeoreferencer, footprint_corners, world_to_pixel  # noqa: E402
from perception.mapping import LogOddsMap  # noqa: E402
from perception.patches import (  # noqa: E402
    PATCH_SPEC, extract_patch, extract_patch_scaled, extract_patches, normalize_patch,
)
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


def test_hot_bypass_rule_saturation_single_pixel_and_exclusion():
    rng = np.random.default_rng(0)
    p = SynthParams()
    # 700 C core clips at the 550 C saturation -> many pixels >= 0.9 x 550; the 35 C rock never bypasses
    temp, _ = synth_frame(rng, [SynthObject("fire", 60, 60, 1.0, 700), SynthObject("rock", 180, 180, 3.0, 35)], p)
    frame = ThermalFrame("s1", 0.0, temp, nadir_pose(), CAM)
    reject_all = FusedDetector(ConstantClassifier(p=0.0, threshold=0.5))
    dets = reject_all.detect(frame, gsd_m=p.gsd_m)
    assert len(dets) == 1 and dets[0].peak_temp_c >= 495
    assert "saturated" in reject_all.last_bypass_reasons
    # an exclusion zone (friendly drone) removes the bypass privilege, so the CNN's "no" stands
    zone = ExclusionZone(dets[0].u, dets[0].v, 20.0, "friendly_drone")
    assert reject_all.detect(frame, gsd_m=p.gsd_m, exclusions=[zone]) == []
    assert "excluded:friendly_drone" in reject_all.last_bypass_reasons
    # high gain mode (saturation ~150 C): trigger follows the camera, not a fixed number
    assert HotBypassRule().with_saturation(150.0).trigger_c == pytest.approx(135.0)
    assert len(FusedDetector(ConstantClassifier(p=1.0)).detect(frame, gsd_m=p.gsd_m)) == 2
    # a single hot (dead) pixel must not bypass
    cold = np.full((64, 64), 8.0, np.float32)
    cold[30, 30] = 549.0
    det = FusedDetector(ConstantClassifier(p=0.0))
    assert det.detect(ThermalFrame("s1", 0.0, cold, nadir_pose(), CAM), gsd_m=0.072) == []


def test_hot_bypass_oversize_blob():
    temp = np.full((128, 128), 8.0, np.float32)
    temp[20:100, 20:100] = 200.0  # 80 px wide hot area, far larger than the 32 px patch window
    det = FusedDetector(ConstantClassifier(p=0.0))
    dets = det.detect(ThermalFrame("s1", 0.0, temp, nadir_pose(), CAM), gsd_m=0.072)
    assert len(dets) == 1 and det.last_bypass_reasons == ["oversize"]
    temp[20:100, 20:100] = 40.0  # same size but only warm (sun-warmed rock / vehicle): no bypass
    assert det.detect(ThermalFrame("s1", 0.0, temp, nadir_pose(), CAM), gsd_m=0.072) == []


def test_scale_normalized_patch_is_altitude_invariant():
    """The same 1 m^2 fire imaged at 80 m and 30 m AGL gives similar normalized patches when scale-normalized."""
    obj = SynthObject("fire", 128, 128, 1.0, 400)
    patches = {}
    for agl in (80.0, 30.0):
        gsd = CAM.ground_sample_distance_m(agl)
        prm = SynthParams(gsd_m=gsd, terrain_sigma_c=0.01, netd_c=0.0)
        temp, _ = synth_frame(np.random.default_rng(1), [obj], prm)
        patches[agl] = (extract_patch_scaled(temp, 128, 128, gsd), extract_patch(temp, 128, 128))
    hot = lambda x: (x > 60).sum()  # noqa: E731
    scaled_ratio = hot(patches[30][0]) / hot(patches[80][0])
    plain_ratio = hot(patches[30][1]) / hot(patches[80][1])
    assert 0.6 < scaled_ratio < 1.6       # same ground window -> similar hot area
    assert plain_ratio > 4                # plain crop: ~(80/30)^2 = 7x more hot pixels up close


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


# ---------- thresholds for the confirm stage ----------

def test_precision_first_threshold():
    from perception.evaluation import choose_threshold_precision_first

    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    s = np.array([.9, .8, .7, .6, .65, .3, .2, .1, .05, .5])
    r = choose_threshold_precision_first(y, s, 1.0)
    assert r["met_target"] and r["precision"] == 1.0 and r["threshold"] == 0.7 and r["recall"] == 0.6


# ---------- sensor wiring ----------

def test_lens_distortion_round_trip_and_georef_correction():
    from perception.distortion import LensDistortion

    lens = LensDistortion(k1=-0.1, k2=0.01, p1=0.001)
    u, v = lens.distort_uv(600.0, 40.0, CAM)
    assert (u, v) != (600.0, 40.0)
    uu, vv = lens.undistort_uv(u, v, CAM)
    assert abs(uu - 600.0) < 1e-6 and abs(vv - 40.0) < 1e-6
    pose, truth = nadir_pose(), Position(x=20.0, y=16.0)
    obs = world_to_pixel(truth, pose, CAM, lens)                      # what a real lens records
    naive = FlatGroundGeoreferencer().pixel_to_world(*obs, pose, CAM)  # pinhole assumption: wrong at the edge
    fixed = FlatGroundGeoreferencer(distortion=lens).pixel_to_world(*obs, pose, CAM)
    assert naive.distance_to(truth) > 0.2 and fixed.distance_to(truth) < 1e-4


def test_pose_buffer_interpolates_and_rejects_stale():
    from perception.frame_context import PoseBuffer

    pb = PoseBuffer(max_gap_s=0.5, max_extrapolation_s=0.05)
    pb.add(0.0, Pose(position=Position(x=0, y=0, z=80), yaw_deg=350))
    pb.add(0.1, Pose(position=Position(x=1.5, y=0, z=80), yaw_deg=10))
    mid = pb.at(0.05)
    assert mid.position.x == pytest.approx(0.75) and mid.yaw_deg == pytest.approx(0.0, abs=1e-9)  # wraps through N
    assert pb.at(0.12) is not None and pb.at(0.3) is None and pb.at(-1.0) is None


def test_dem_sampler_bilinear_and_terrain_georef():
    from perception.terrain import DemSampler

    grid = GridSpec(origin=Position(x=-100, y=-100), resolution_m=10.0, rows=20, cols=20)
    xs = -100 + (np.arange(20) + 0.5) * 10
    heights = np.tile(0.1 * xs, (20, 1))  # 10% slope rising to the east
    dem = DemSampler(heights, grid)
    assert dem(5.0, 0.0) == pytest.approx(0.5)
    pose = nadir_pose(z=80.0)
    on_dem = TerrainGeoreferencer(dem, step_m=0.25).pixel_to_world(640, 256, pose, CAM)
    flat = FlatGroundGeoreferencer(0.0).pixel_to_world(640, 256, pose, CAM)
    assert abs(on_dem.z - dem(on_dem.x, on_dem.y)) < 0.05
    assert on_dem.x < flat.x  # rising ground meets the ray earlier


def _fake_pipeline(**kw):
    truth = [TruthSource("F1", Position(x=5, y=5), 2.0, 450.0)]
    det = FakeDetector(lambda now: truth, seed=3, p_decoy=0.0, clutter_rate=0.0)
    grid = GridSpec(origin=Position(x=-200, y=-200), resolution_m=2.0, rows=200, cols=200)
    return SeekerPerception("s1", det, grid=grid, **kw)


def test_pipeline_skips_ffc_and_bad_thermal_without_writing_misses():
    from perception.frame_context import FrameContext

    perc = _fake_pipeline()
    f = ThermalFrame("s1", 1.0, np.zeros((0, 0)), nadir_pose(), CAM)
    assert perc.process(f, FrameContext(ffc_active=True)) == []
    assert perc.process(f, FrameContext(thermal_ok=False)) == []
    assert perc.map.pop_updates() == [] and perc.stats.skipped_ffc == 1 and perc.stats.skipped_thermal == 1
    assert perc.process(f, FrameContext()) and perc.map.pop_updates()


def test_pipeline_rtk_loss_widens_gate_and_uses_altimeter():
    from perception.frame_context import FrameContext

    perc = _fake_pipeline()
    f = ThermalFrame("s1", 1.0, np.zeros((0, 0)), nadir_pose(), CAM)
    perc.process(f, FrameContext())
    # simulate a 6 m GNSS jump without RTK: still the same hotspot (gate 5 m + 3 m), not a duplicate
    jumped = ThermalFrame("s1", 2.0, np.zeros((0, 0)), nadir_pose(x=-6.0), CAM)
    perc.process(jumped, FrameContext(gnss_rtk_ok=False, agl_m=80.0))
    assert len(perc.tracker.tracks) == 1 and perc.stats.rtk_lost_frames == 1 and perc.stats.altimeter_used == 1
    perc.process(f, FrameContext(agl_m=500.0))  # beyond the altimeter's rated range -> ignored
    assert perc.stats.altimeter_used == 1


def test_pipeline_pose_buffer_used_at_exposure_time():
    from perception.frame_context import PoseBuffer

    pb = PoseBuffer()
    pb.add(0.4, nadir_pose(x=0.0))
    pb.add(0.6, nadir_pose(x=3.0))
    perc = _fake_pipeline(pose_buffer=pb)
    stale_pose_in_frame = nadir_pose(x=999.0)  # must be replaced by the buffered pose at t = 0.5
    out = perc.process(ThermalFrame("s1", 0.5, np.zeros((0, 0)), stale_pose_in_frame, CAM))
    assert out and out[0].position.distance_to(Position(x=5, y=5), horizontal=True) < 1.0
    assert perc.process(ThermalFrame("s1", 5.0, np.zeros((0, 0)), nadir_pose(), CAM)) == []
    assert perc.stats.skipped_stale_pose == 1


# ---------- close-range confirm / verify ----------

def test_confirmer_confirms_fire_and_rejects_decoy():
    from perception.confirm import HotspotConfirmer

    p = SynthParams(gsd_m=CAM.ground_sample_distance_m(30.0))
    rng = np.random.default_rng(5)
    clf_fire = ConstantClassifier(p=0.97, threshold=0.3, confirm_threshold=0.9)
    conf = HotspotConfirmer(clf_fire, min_frames=5)
    for _ in range(6):
        temp, _ = synth_frame(rng, [SynthObject("fire", 128, 128, 0.5, 400)], p)
        conf.add_uv(temp, (128.0, 128.0), p.gsd_m)
    assert conf.result().status is HotspotStatus.CONFIRMED
    rock = HotspotConfirmer(ConstantClassifier(p=0.05, threshold=0.3, confirm_threshold=0.9), min_frames=5)
    for _ in range(6):
        temp, _ = synth_frame(rng, [SynthObject("rock", 128, 128, 3.0, 35)], p)
        rock.add_uv(temp, (128.0, 128.0), p.gsd_m)
    r = rock.result()
    assert r.status is HotspotStatus.FALSE_POSITIVE and r.seen_frac == 1.0
    assert HotspotConfirmer(clf_fire).result().status is None  # too few frames


def test_suppression_checker():
    from perception.confirm import SuppressionChecker

    p = SynthParams(gsd_m=CAM.ground_sample_distance_m(15.0))
    rng = np.random.default_rng(6)
    still = SuppressionChecker()
    out = SuppressionChecker()
    for _ in range(6):
        hot, _ = synth_frame(rng, [SynthObject("fire", 128, 128, 0.3, 300)], p)
        cold, _ = synth_frame(rng, [], p)
        still.add_uv(hot, (128.0, 128.0), p.gsd_m)
        out.add_uv(cold, (128.0, 128.0), p.gsd_m)
    assert out.result().status is HotspotStatus.SUPPRESSED
    assert still.result().status is None and "still hot" in still.result().reason


def test_confirm_threshold_never_below_search_threshold():
    from training.cv import evaluate_oof

    y = np.array([1] * 50 + [0] * 50)
    oof = np.r_[np.linspace(0.6, 1.0, 50), np.linspace(0.0, 0.3, 50)]  # perfectly separable, "easy" data
    folds = [(np.arange(100) % 2 == k, np.nonzero(np.arange(100) % 2 == k)[0]) for k in (0, 1)]
    cv = evaluate_oof(y, oof, folds, 0.95, 0.98)
    assert cv["confirm_threshold"]["threshold"] >= cv["final_threshold"]["threshold"]

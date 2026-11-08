"""Ignore masks, NADIR history labels, Sycan weak masks, hard-example sampling, point view, view pooling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perception.datasets import MASK_IGNORE, FrameRecord, load_ignore, load_mask
from perception.detectors.classifiers import ConstantClassifier
from perception.evaluation import BLOB_IGNORE, blob_label, drop_ignored
from training.build_nadir import LabelRules, detect_saturation, history_maps, label_frame
from training.build_sycan_weak import assign_splits, weak_mask
from training.compare import score_manifest, score_points
from training.eval_sets import concat
from training.patch_data import build_patch_set, sample_hard_points


def test_ignore_mask_roundtrip_and_blob_label(tmp_path: Path):
    m = np.zeros((10, 10), np.uint8)
    m[1:3, 1:3] = 1
    m[6:8, 6:8] = MASK_IGNORE
    np.save(tmp_path / "m.npy", m)
    rec = FrameRecord(frame_path="x", mask_path=str(tmp_path / "m.npy"), group="g")
    fire, ig = load_mask(rec), load_ignore(rec)
    assert fire.sum() == 4 and ig.sum() == 4 and not (fire & ig).any()
    assert blob_label((1, 1, 3, 3), fire, ignore=ig) == 1
    assert blob_label((6, 6, 8, 8), fire, ignore=ig) == BLOB_IGNORE
    assert blob_label((6, 6, 8, 8), fire) == 0                      # without the ignore mask: old behaviour
    assert drop_ignored([(2.0, 2.0), (6.5, 6.5)], ig) == [(2.0, 2.0)]
    np.save(tmp_path / "old.npy", (m == 1).astype(np.uint8) * 255)   # 0/255 masks keep meaning "fire"
    assert load_mask(FrameRecord(frame_path="x", mask_path=str(tmp_path / "old.npy"), group="g")).sum() == 4


def _burn_stack() -> np.ndarray:
    """3 pixels over 60 frames: (0,0) never burns; (0,1) burns at t=10 and cools to a 45 C plateau;
    (0,2) burns at t=10 and keeps smouldering at ~110 C."""
    n = 60
    st = np.full((n, 1, 3), 30.0, np.float32)
    t = np.arange(n)
    cool = 45 + 355 * np.exp(-np.maximum(t - 10, 0) / 6.0)
    st[10:, 0, 1] = cool[10:]
    st[10:, 0, 2] = np.maximum(110.0, cool[10:])
    return st


def test_history_labels_follow_physics():
    st = _burn_stack()
    maps = history_maps(st)
    assert maps["t_peak"][0, 1] == 10 and maps["floor"][0, 1] == pytest.approx(45.0, abs=0.5)
    plateau = 45.0
    running = np.maximum.accumulate(st, axis=0)
    lab = lambda t: label_frame(t, st[t], running[t], maps, plateau, LabelRules())[0]  # noqa: E731
    assert list(lab(5)) == [0, 0, 0]                    # all unburned before the front
    assert lab(10)[1] == 1 and lab(10)[2] == 1          # flaming
    assert lab(55)[1] == 0                              # burnt out: at its plateau long after the peak -> char
    assert lab(55)[2] == MASK_IGNORE                    # steady 110 C: floor far above the plateau, never "burnt out"
    assert lab(55)[0] == 0
    nan = st[20].copy()
    nan[0, 0] = np.nan
    assert label_frame(20, nan, running[20], maps, plateau)[0, 0] == MASK_IGNORE   # nodata


def test_detect_saturation_high_gain():
    a = np.full((50, 50), 30.0, np.float32)
    a[:10, :10] = 197.0
    assert detect_saturation(a) == 197.0
    a[0, 0] = 400.0
    assert detect_saturation(a) == 550.0


def test_weak_mask_and_split():
    t = np.array([[30.0, 150.0, 187.4, 250.0]], np.float32)
    assert weak_mask(t, False, 187.4).tolist() == [[0, 0, 0, 0]]
    assert weak_mask(t, True, 187.4).tolist() == [[MASK_IGNORE, MASK_IGNORE, 1, 1]]   # clip counts as hot
    assert weak_mask(t, True, 608.7).tolist() == [[MASK_IGNORE, MASK_IGNORE, MASK_IGNORE, 1]]
    recs = [FrameRecord(frame_path=f"f{i}", group=g, label=1) for i, g in enumerate("abcdefgh")]
    s = assign_splits(recs, {"a": "test", "b": "train"})
    assert s["a"] == "test" and s["b"] == "train" and list(s.values()).count("test") >= 2


def _nadir_like(tmp_path: Path) -> list[FrameRecord]:
    t = np.full((80, 80), 30.0, np.float32)
    m = np.zeros((80, 80), np.uint8)
    t[10:12, 10:12] = 300.0
    m[10:12, 10:12] = 1
    t[10:12, 30:32] = 80.0          # residual ember
    m[10:12, 30:32] = 1
    t[50:70, 50:70] = 48.0          # warm char, far from fire
    t[40:44, 10:14] = 90.0          # cooling, unknown
    m[40:44, 10:14] = MASK_IGNORE
    np.save(tmp_path / "f.npy", t)
    np.save(tmp_path / "m.npy", m)
    return [FrameRecord(frame_path=str(tmp_path / "f.npy"), mask_path=str(tmp_path / "m.npy"), group="nadir_x_b00",
                        label=1, split="test", gsd_m=0.15, saturation_c=550.0)]


def test_hard_sampling_and_patch_set(tmp_path: Path):
    rec = _nadir_like(tmp_path)[0]
    temp, mask, ig = np.load(rec.frame_path), load_mask(rec), load_ignore(rec)
    pts = sample_hard_points(temp, mask, ig, rec, 10, 4, np.random.default_rng(0))
    negs = [uv for uv, lab in pts if lab == 0]
    assert len(negs) == 10 and all(50 <= v < 70 and 50 <= u < 70 for u, v in negs)   # only the clean warm char
    poss = [uv for uv, lab in pts if lab == 1]
    assert len(poss) == 4 and all(temp[int(v), int(u)] <= 80.0 for u, v in poss)     # cooler half of the fire
    ps = build_patch_set([rec], hard_neg_per_frame=5, hard_pos_per_frame=3, random_neg_per_frame=0)
    assert ps.stats["by_origin"]["hard_neg"] == 5 and ps.stats["by_origin"]["hard_pos"] == 3
    assert ps.stats["proposals_skipped_ignore"] >= 1                                 # the 90 C blob is ignored


def test_point_view_and_pooling(tmp_path: Path):
    recs = _nadir_like(tmp_path)
    clf = ConstantClassifier(p=0.5)
    pts = score_points(recs, clf, n_neg=6, n_pos=4)
    assert pts.y.sum() == 4 and (pts.y == 0).sum() == 6
    assert set(pts.label_text) <= {"warm_not_fire", "fire<120C", "fire>=120C"}
    props = score_manifest(recs, clf)
    assert BLOB_IGNORE not in props.y.tolist()
    pooled = concat({"a": props, "b": props})          # same blobs twice -> counted once
    assert len(pooled) == len(props) and all(g.startswith("a:") for g in pooled.group)
    pooled2 = concat({"pts": pts, "props": props})
    assert len(pts) <= len(pooled2) <= len(pts) + len(props) and set(pooled2.view) == {"pts", "props"}

"""perception.triage (FIRE / REVIEW / NOT_FIRE) and training.compare (three approaches on the same candidates)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from perception.datasets import FrameRecord, load_manifest, write_manifest
from perception.detectors import HotBypassRule
from perception.detectors.classifiers import ConstantClassifier
from perception.triage import Triage, TriageBands, suggest_bands, sweep_bands, triage_table
from training.compare import compare, default_bands, default_search_thresholds, markdown, score_handlabels, score_manifest
from training.label_tool import find_candidates, write_candidates


def test_bands_classify_edges_and_validation():
    b = TriageBands(0.2, 0.8)
    assert b.classify(0.1999) is Triage.NOT_FIRE
    assert b.classify(0.2) is Triage.REVIEW          # low edge belongs to REVIEW
    assert b.classify(0.8) is Triage.FIRE            # high edge belongs to FIRE
    s = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    assert list(b.classify_many(s)) == [b.classify(x) for x in s]
    assert TriageBands(0.5, 0.5).classify(0.49) is Triage.NOT_FIRE   # empty REVIEW band = plain threshold
    with pytest.raises(ValueError):
        TriageBands(0.9, 0.1)
    clf = ConstantClassifier(threshold=0.5, confirm_threshold=0.9, reject_threshold=0.1)
    assert TriageBands.from_classifier(clf) == TriageBands(0.1, 0.9)


def test_triage_table_counts_and_rates():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.05, 0.5, 0.9, 0.95, 0.05, 0.1, 0.5, 0.99])
    t = triage_table(y, s, TriageBands(0.2, 0.8))
    assert t["counts"] == {"FIRE": {"fire": 2, "not_fire": 1}, "REVIEW": {"fire": 1, "not_fire": 1},
                           "NOT_FIRE": {"fire": 1, "not_fire": 2}}
    assert t["missed_fire_rate"] == pytest.approx(0.25)
    assert t["review_rate"] == pytest.approx(2 / 8)
    assert t["fire_precision"] == pytest.approx(2 / 3)
    assert t["false_alarm_rate"] == pytest.approx(1 / 4)


def test_sweep_matches_table_and_suggest_bands():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    s = np.clip(np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.15, 400), 0, 1)
    grid = np.linspace(0, 1, 11)
    sw = sweep_bands(y, s, grid, grid)
    i, j = 3, 8
    t = triage_table(y, s, TriageBands(grid[i], grid[j]))
    assert sw["missed_fire_rate"][i, j] == pytest.approx(t["missed_fire_rate"])
    assert sw["review_rate"][i, j] == pytest.approx(t["review_rate"])
    assert sw["fire_precision"][i, j] == pytest.approx(t["fire_precision"])
    assert np.isnan(sw["review_rate"][8, 3])          # low > high is not a valid band
    sb = suggest_bands(y, s, max_missed_fire_rate=0.01, min_fire_precision=0.95)
    assert triage_table(y, s, sb["bands"])["missed_fire_rate"] <= 0.01 + 1e-12


class _PeakClassifier(ConstantClassifier):
    """p = centre temperature / 1000: hotter blobs score higher (stands in for a real model)."""

    def predict_proba(self, temp_c, centers_uv, gsd_m=None):
        return np.array([min(float(temp_c[int(v), int(u)]) / 1000.0, 1.0) for u, v in centers_uv], dtype=np.float32)


def _masked_frames(tmp_path: Path) -> list[FrameRecord]:
    """One frame: a 300 C fire (masked), a 70 C non-fire spot, and a masked fire that reads cold (proposer misses it)."""
    t = np.full((64, 64), 10.0, np.float32)
    m = np.zeros((64, 64), np.uint8)
    t[10:13, 10:13] = 300.0
    m[10:13, 10:13] = 1
    t[40:42, 40:42] = 70.0
    m[50:52, 20:22] = 1
    np.save(tmp_path / "f.npy", t)
    np.save(tmp_path / "f_mask.npy", m)
    recs = [FrameRecord(frame_path=str(tmp_path / "f.npy"), mask_path=str(tmp_path / "f_mask.npy"), group="g0",
                        label=1, split="test", gsd_m=0.1, saturation_c=310.0)]
    write_manifest(tmp_path / "manifest.csv", recs)
    return load_manifest(tmp_path / "manifest.csv", split="test")


def test_score_manifest_three_approaches_and_bypass(tmp_path: Path):
    clf = _PeakClassifier(threshold=0.2, confirm_threshold=0.25, reject_threshold=0.05)
    sc = score_manifest(_masked_frames(tmp_path), clf, HotBypassRule(saturation_c=550.0))
    assert len(sc) == 2 and sc.y.tolist().count(1) == 1
    assert sc.n_objects == 2 and sc.missed_objects == 1
    fire = int(np.flatnonzero(sc.y == 1)[0])
    # per-frame saturation 310 C -> trigger 279 C -> the 300 C fire bypasses; the 70 C spot does not
    assert sc.bypass[fire] == "saturated" and not sc.bypassed[1 - fire]
    assert sc.p_pipeline[fire] >= 0.9 and sc.p_model[fire] == pytest.approx(0.3)
    assert np.allclose(sc.scores("algorithmic"), sc.peak_c)
    res = compare(sc, default_bands(clf), default_search_thresholds(clf), n_boot=20)
    assert set(res["approaches"]) == {"algorithmic", "model", "pipeline"}
    assert res["approaches"]["pipeline"]["search"]["recall"] == 1.0
    assert "never touched 1 of 2" in markdown(res, "unit")
    sub = sc.subset(sc.y == 1)
    assert len(sub) == 1 and sub.bypass == ["saturated"]


def test_score_manifest_needs_masks(tmp_path: Path):
    np.save(tmp_path / "f.npy", np.zeros((8, 8), np.float32))
    with pytest.raises(ValueError):
        score_manifest([FrameRecord(frame_path=str(tmp_path / "f.npy"), group="g", label=0)], ConstantClassifier())


def test_score_handlabels_drops_unsure(tmp_path: Path):
    t = np.full((64, 64), 10.0, np.float32)
    t[20:23, 20:23] = 150.0
    t[40:43, 40:43] = 120.0
    recs = []
    for i in range(2):
        np.save(tmp_path / f"IR_{i}.npy", t)
        recs.append(FrameRecord(frame_path=str(tmp_path / f"IR_{i}.npy"), group=f"g{i}", label=1, gsd_m=0.1))
    cands = find_candidates(recs)
    assert len(cands) == 4
    write_candidates(tmp_path / "candidates.csv", cands)
    labs = ["active", "not_fire", "unsure", "smoulder"]
    with (tmp_path / "labels_x.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cand_id", "label", "labeler", "rgb_available", "labelled_at"])
        for c, lab in zip(cands, labs):
            w.writerow([c.cand_id, lab, "x", 0, ""])
    sc = score_handlabels(tmp_path / "candidates.csv", tmp_path / "labels_x.csv", _PeakClassifier())
    assert len(sc) == 3 and sorted(sc.label_text) == ["active", "not_fire", "smoulder"]
    assert sc.y.tolist() == [int(x != "not_fire") for x in sc.label_text]
    assert all(b is None for b in sc.bypass)          # 150 C is below 0.9 x 550 C and the blob is small
    assert np.allclose(sc.p_model, sc.peak_c / 1000.0)

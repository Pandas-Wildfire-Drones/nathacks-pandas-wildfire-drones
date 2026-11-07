"""training.label_tool: candidate selection, keyboard session logic, agreement, scoring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perception.datasets import FrameRecord, write_manifest
from perception.detectors.classifiers import ConstantClassifier
from training.label_tool import (
    LabelSession, agreement, cohen_kappa, find_candidates, labelled_targets, load_candidates, load_labels,
    predict_candidates, write_candidates,
)


def _frames(tmp_path: Path) -> Path:
    """Group g0: 3 fire frames with one 150 C spot (in band) and one 400 C spot (out of band); g1: one No-Fire frame."""
    d = tmp_path / "data"
    (d / "rgb").mkdir(parents=True)
    recs = []
    for i in range(3):
        t = np.full((64, 64), 10.0, np.float32)
        t[20:23, 20:23] = 150.0
        t[40:43, 45:48] = 400.0
        np.save(d / f"IR_{i:03d}.npy", t)
        np.save(d / "rgb" / f"RGB_{i:03d}.npy", np.zeros(1))         # not an image: must be ignored
        recs.append(FrameRecord(frame_path=str(d / f"IR_{i:03d}.npy"), group="g0", label=1, gsd_m=0.1))
    from PIL import Image
    for i in (0, 1):                                                  # RGB for frames 0 and 1 only
        Image.fromarray(np.full((128, 128, 3), 90, np.uint8)).save(d / "rgb" / f"RGB_{i:03d}.png")
    np.save(d / "cold.npy", np.full((64, 64), 5.0, np.float32))
    recs.append(FrameRecord(frame_path=str(d / "cold.npy"), group="g1", label=0))
    write_manifest(d / "manifest.csv", recs)
    return d


def test_candidates_band_filter_rgb_pairing_and_roundtrip(tmp_path: Path):
    d = _frames(tmp_path)
    from perception.datasets import load_manifest

    cands = find_candidates(load_manifest(d / "manifest.csv"), rgb_dir=d / "rgb", rgb_replace=[("IR_", "RGB_")])
    assert len(cands) == 3 and {c.group for c in cands} == {"g0"}
    assert all(80 <= c.peak_c < 200 and abs(c.u - 21) < 1 and abs(c.v - 21) < 1 for c in cands)
    assert sum(bool(c.rgb_path) for c in cands) == 2 and all(c.rgb_path.endswith(".png") for c in cands if c.rgb_path)
    mid = next(c for c in cands if c.frame_path.endswith("IR_001.npy"))
    assert mid.prev_path.endswith("IR_000.npy") and mid.next_path.endswith("IR_002.npy")
    write_candidates(tmp_path / "labels" / "candidates.csv", cands)
    back = load_candidates(tmp_path / "labels" / "candidates.csv")
    assert [c.cand_id for c in back] == [c.cand_id for c in cands]
    assert all(Path(c.frame_path).exists() for c in back) and back[0].gsd_m == 0.1


def test_max_per_group_caps_sampling(tmp_path: Path):
    d = _frames(tmp_path)
    from perception.datasets import load_manifest

    assert len(find_candidates(load_manifest(d / "manifest.csv"), max_per_group=2)) == 2


def test_session_keys_save_resume_and_render(tmp_path: Path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from perception.datasets import load_manifest

    d = _frames(tmp_path)
    cands = find_candidates(load_manifest(d / "manifest.csv"), rgb_dir=d / "rgb", rgb_replace=[("IR_", "RGB_")])
    cpath = tmp_path / "labels" / "candidates.csv"
    write_candidates(cpath, cands)
    lpath = tmp_path / "labels" / "labels_a.csv"
    s = LabelSession(cpath, lpath, "a")
    fig = plt.figure()
    for c in s.cands:                                  # every candidate renders (with and without RGB)
        s.render(fig)
        s.i += 1
    s.i = 0
    assert s.handle_key("a") and s.i == 1
    assert s.handle_key("x") and s.i == 1              # unknown keys are ignored
    assert s.handle_key("left") and s.i == 0
    assert s.handle_key("n") and s.i == 1              # relabel #0, then jump to the first unlabelled (#1)
    assert load_labels(lpath)[s.cands[0].cand_id]["label"] == "not_fire"
    assert s.handle_key("s") and s.i == 2
    assert not s.handle_key("q")                       # q saves and closes
    resumed = LabelSession(cpath, lpath, "a")
    assert resumed.i == 2                              # resumes at the first unlabelled candidate
    assert not resumed.handle_key("n") and resumed.done
    resumed.render(fig)                                # "all labelled" screen
    assert [r["label"] for r in load_labels(lpath).values()] == ["not_fire", "smoulder", "not_fire"]
    plt.close(fig)


def test_kappa_agreement_and_scoring(tmp_path: Path):
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    assert cohen_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) == pytest.approx(-1.0)
    import csv
    for name, labs in (("x", ["active", "not_fire", "smoulder"]), ("y", ["smoulder", "not_fire", "unsure"])):
        with (tmp_path / f"{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cand_id", "label", "labeler", "rgb_available", "labelled_at"])
            for k, lab in enumerate(labs):
                w.writerow([f"c{k}", lab, name, 1, ""])
    ag = agreement(tmp_path / "x.csv", tmp_path / "y.csv")
    assert ag["n_common"] == 2 and ag["kappa_fire_vs_not"] == pytest.approx(1.0)

    d = _frames(tmp_path)
    from perception.datasets import load_manifest

    cands = find_candidates(load_manifest(d / "manifest.csv"))
    labels = {cands[0].cand_id: {"label": "active"}, cands[1].cand_id: {"label": "unsure"},
              cands[2].cand_id: {"label": "not_fire"}}
    kept, y = labelled_targets(cands, labels)
    assert len(kept) == 2 and y.tolist() == [1, 0]
    p = predict_candidates(ConstantClassifier(p=0.7), kept)
    assert p.shape == (2,) and np.allclose(p, 0.7)

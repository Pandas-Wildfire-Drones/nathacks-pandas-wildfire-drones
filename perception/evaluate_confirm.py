"""Evaluate close-range confirmation (INVESTIGATE) on synthetic hover sequences. Writes a RESULTS.md section.

    python -m perception.evaluate_confirm --model scale=perception/models/fire_cnn_synth.onnx \
        [--model plain=perception/models/fire_cnn_synth_plain.onnx] --targets 300 --key confirm_synth

For each target (50% fires 0.01-4 m^2 at 150-700 C, 50% decoys: warm rocks, vehicles, animals), a seeker hovers at
25-40 m AGL and takes --frames frames (hover drift ~1 px, fresh noise each frame). HotspotConfirmer decides
CONFIRMED / FALSE_POSITIVE / inconclusive. The same target seen once from 80 m with the SEARCH threshold is the
baseline. Comparing a scale-normalized model with a plain-crop one answers "does the confirm stage need its own
model, or does GSD normalization handle the altitude change?" (docs/decisions/0004).

SYNTHETIC ONLY: background is regenerated per frame (real hover frames are more correlated), decoys are cooler than
fires by construction. Treat as a behaviour check, not a performance claim.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from contracts import CameraModel, HotspotStatus

from .confirm import HotspotConfirmer
from .detectors import HotBypassRule, ThresholdDetector, load_classifier
from .results_doc import update_section
from .synth import SynthObject, SynthParams, random_objects, synth_frame

CAM = CameraModel(width_px=640, height_px=512, hfov_deg=32.0)  # seeker camera (contracts default_seeker_spec)
C = 128.0  # target pixel (frame centre)


def _target(rng) -> SynthObject:
    if rng.random() < 0.5:
        area = float(np.exp(rng.uniform(np.log(0.01), np.log(4.0))))
        return SynthObject("fire", C, C, area, float(rng.uniform(150, 700)), float(rng.uniform(1, 2.5)),
                           float(rng.uniform(0, 180)))
    kind = rng.choice(["rock", "vehicle", "animal"], p=[0.5, 0.25, 0.25])
    if kind == "rock":
        return SynthObject("rock", C, C, float(rng.uniform(0.3, 6)), float(rng.uniform(25, 45)), float(rng.uniform(1, 2)))
    if kind == "vehicle":
        return SynthObject("vehicle", C, C, float(rng.uniform(1, 8)), float(rng.uniform(40, 90)), 2.2,
                           float(rng.uniform(0, 180)))
    return SynthObject("animal", C, C, float(rng.uniform(0.2, 1.0)), float(rng.uniform(30, 38)), 2.0,
                       float(rng.uniform(0, 180)))


def _clutter(rng, p: SynthParams) -> list[SynthObject]:
    return [o for o in random_objects(rng, p, fire_prob=0.0) if math.hypot(o.u - C, o.v - C) > 45]


def _size_bucket(o: SynthObject) -> str:
    if not o.is_fire:
        return "decoy"
    return "fire <0.1 m²" if o.area_m2 < 0.1 else "fire 0.1-1 m²" if o.area_m2 < 1 else "fire >1 m²"


def _search_once(clf, proposer, rng, target, saturation_c) -> bool:
    gsd = CAM.ground_sample_distance_m(80.0)
    p = SynthParams(gsd_m=gsd)
    temp, _ = synth_frame(rng, [target] + _clutter(rng, p), p)
    near = [b for b in proposer.blobs(temp) if (b.u - C) ** 2 + (b.v - C) ** 2 <= (3.0 / gsd) ** 2]
    if not near:
        return False
    b = max(near, key=lambda x: x.peak_temp_c)
    return bool(clf.predict_proba(temp, [(b.u, b.v)], gsd)[0] >= clf.threshold)


def run(models: dict[str, object], targets: int, frames: int, agl_range, seed: int, saturation_c: float) -> dict:
    proposer = ThresholdDetector()
    out = {}
    for name, clf in models.items():
        rng = np.random.default_rng(seed)  # same targets for every model
        tally = defaultdict(Counter)
        search = defaultdict(Counter)
        for _ in range(targets):
            tgt = _target(rng)
            bucket = _size_bucket(tgt)
            agl = float(rng.uniform(*agl_range))
            p = SynthParams(gsd_m=CAM.ground_sample_distance_m(agl))
            clutter = _clutter(rng, p)
            conf = HotspotConfirmer(clf, proposer, HotBypassRule(saturation_c=saturation_c),
                                    min_frames=frames, max_frames=frames)
            for _f in range(frames):
                du, dv = rng.normal(0, 1.0, 2)
                obj = SynthObject(tgt.kind, C + du, C + dv, tgt.area_m2, tgt.temp_c, tgt.aspect, tgt.angle_deg)
                temp, _ = synth_frame(rng, [obj] + clutter, p)
                conf.add_uv(temp, (C, C), p.gsd_m, saturation_c)
            r = conf.result()
            tally[bucket][r.status.value if r.status else "INCONCLUSIVE"] += 1
            search[bucket]["detected" if _search_once(clf, proposer, rng, tgt, saturation_c) else "missed"] += 1
        fires = sum((tally[b] for b in tally if b != "decoy"), Counter())
        dec = tally["decoy"]
        tp, fn_rej = fires["CONFIRMED"], fires["FALSE_POSITIVE"]
        fp, tn = dec["CONFIRMED"], dec["FALSE_POSITIVE"]
        out[name] = {
            "by_bucket": {b: dict(c) for b, c in sorted(tally.items())},
            "search_80m_by_bucket": {b: dict(c) for b, c in sorted(search.items())},
            "confirm_precision": tp / (tp + fp) if tp + fp else float("nan"),
            "confirm_recall_of_fires": tp / max(sum(fires.values()), 1),
            "fires_wrongly_rejected": fn_rej, "decoys_confirmed": fp, "decoys_rejected": tn,
            "inconclusive": fires["INCONCLUSIVE"] + dec["INCONCLUSIVE"],
            "thresholds": {"confirm": clf.confirm_threshold, "reject": getattr(clf, "reject_threshold", clf.threshold)},
            "scale_normalized": getattr(clf, "scale_normalized", None),
        }
    return out


def _markdown(res: dict, targets: int, frames: int, agl_range) -> str:
    lines = [f"### Close-range confirmation (INVESTIGATE), synthetic · {targets} targets, {frames} frames each, "
             f"{agl_range[0]:.0f}-{agl_range[1]:.0f} m AGL", "",
             "Synthetic behaviour check (decoys are cooler than fires by construction; background regenerated per frame). "
             "Not a real-world performance claim.", "",
             "| model | confirm precision | fires confirmed | fires wrongly rejected | decoys confirmed | decoys rejected | inconclusive | thresholds (confirm / reject) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        lines.append(f"| {name} (scale-normalized: {r['scale_normalized']}) | {r['confirm_precision']:.3f} | "
                     f"{r['confirm_recall_of_fires']:.1%} | {r['fires_wrongly_rejected']} | {r['decoys_confirmed']} | "
                     f"{r['decoys_rejected']} | {r['inconclusive']} | "
                     f"{r['thresholds']['confirm']:.3f} / {r['thresholds']['reject']:.3f} |")
    lines += ["", "**By target type** (close-range outcome · single 80 m search frame):", "",
              "| model | target | CONFIRMED | FALSE_POSITIVE | INCONCLUSIVE | search @80 m detected / missed |",
              "|---|---|---|---|---|---|"]
    for name, r in res.items():
        for b, c in r["by_bucket"].items():
            s = r["search_80m_by_bucket"].get(b, {})
            lines.append(f"| {name} | {b} | {c.get('CONFIRMED', 0)} | {c.get('FALSE_POSITIVE', 0)} | "
                         f"{c.get('INCONCLUSIVE', 0)} | {s.get('detected', 0)} / {s.get('missed', 0)} |")
    return "\n".join(lines)


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="name=path/to/model.onnx (repeatable)")
    ap.add_argument("--targets", type=int, default=300)
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--confirm-agl", type=float, nargs=2, default=(25.0, 40.0))
    ap.add_argument("--saturation-c", type=float, default=550.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key", default="confirm_synth")
    ap.add_argument("--results-doc", default=str(Path(__file__).with_name("RESULTS.md")))
    a = ap.parse_args()
    models = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        models[name] = load_classifier(path)
    res = run(models, a.targets, a.frames, tuple(a.confirm_agl), a.seed, a.saturation_c)
    md = _markdown(res, a.targets, a.frames, a.confirm_agl)
    update_section(a.results_doc, a.key, md, {"results": res, "targets": a.targets, "frames": a.frames,
                                              "confirm_agl": list(a.confirm_agl), "status_values": [s.value for s in HotspotStatus]})
    print(md)


if __name__ == "__main__":
    _main()

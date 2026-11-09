# perception/ (workstream C)

Drone-side perception. Imports only `contracts`, numpy and scipy (hard rule 1); onnxruntime/tifffile load lazily.

| Module | What |
|---|---|
| `detectors/threshold.py` | `ThresholdDetector`: classical, recall-first blob proposals (absolute temperature OR local contrast) |
| `detectors/fused.py` | `FusedDetector`: the mixed approach (proposals → verifier → detections) + `HotBypassRule` (decision 0003) |
| `detectors/classifiers.py` | `OnnxCnnClassifier` (**primary**), `OnnxGbmClassifier` (**stretch**), `ConstantClassifier` (stub), `load_classifier()` |
| `detectors/fake.py` | `FakeDetector`: sim ground truth with misses, decoy hits and clutter (deterministic). **Use this from day one** |
| `georef.py` | pixel ↔ world (flat ground or DEM ray-march), frame footprint |
| `tracking.py` | `HotspotTracker`: detections over frames → `Hotspot` entities |
| `mapping.py` | `LogOddsMap`: NumPy log-odds layers → `GridCellUpdate` deltas for `MapSync` |
| `pipeline.py` | `SeekerPerception`: one call per frame, all of the above, with sensor context |
| `reporting.py` | `PerceptionReporter`: publishes hotspot reports and sends REVIEW-band candidates to `ground/reviews` |
| `frame_context.py` | `FrameContext` (altimeter, RTK/thermal health, FFC, gain saturation, friendly drones) + `PoseBuffer` (pose at exposure time) |
| `terrain.py` / `distortion.py` | `DemSampler` (terrain height) / `LensDistortion` (Brown–Conrady) |
| `confirm.py` | close range: `HotspotConfirmer` (INVESTIGATE) and `SuppressionChecker` (VERIFY after a drop) |
| `evaluate_confirm.py` | synthetic hover-sequence evaluation of the confirm stage |
| `radiometry.py` | Planck temperature ↔ radiance (used whenever resolution changes) |
| `patches.py` / `features.py` | preprocessing shared with training (versioned) |
| `triage.py` | `TriageBands`: score → FIRE / REVIEW / NOT_FIRE, `triage_table`, `sweep_bands`, `suggest_bands` (comparison: `training/compare.py` + `training/compare_approaches.ipynb`) |
| `evaluation.py` / `evaluate.py` | metrics (F1, recall-first threshold, object matching, CIs) and the test-split evaluator |
| `synth.py` | synthetic radiometric night frames with exact masks (sub-pixel dilution, decoys) |
| `RESULTS.md`, `results/*.json` | evaluated numbers (written by scripts) |

## Using it from the sim or the drone loop (workstreams A/B)

```python
from perception.detectors import FakeDetector, FusedDetector, TruthSource, load_classifier
from perception.pipeline import SeekerPerception
from perception.reporting import PerceptionReporter
from perception.triage import TriageBands

# day one: ground-truth-driven detector (sim supplies current fire/decoy state)
det = FakeDetector(lambda now: [TruthSource("F1", fire_pos, area_m2=2.0, temp_c=450.0)], seed=scenario.seed)
# later: the real mixed detector
det = FusedDetector(load_classifier("perception/models/fire_cnn.onnx"))

perc = SeekerPerception(drone_id="seeker-1", detector=det, grid=scenario.grid,
                        dem=dem_sampler, pose_buffer=pose_buffer)        # both optional
reporter = PerceptionReporter(perc, bus, TriageBands(0.2, 0.8))  # calibrate for fused hotspot confidence
pose_buffer.add(t, autopilot_pose)                                      # every autopilot tick (50-200 Hz)
ctx = FrameContext(agl_m=altimeter_m, gnss_rtk_ok=state.sensors_ok.get("gnss_rtk", True),
                   ffc_active=camera.ffc_in_progress, friendly_positions=other_drone_positions)
changed_hotspots = reporter.process(thermal_frame, ctx)  # -> HotspotReport + one ground review per REVIEW track
cells = perc.map.pop_updates()                       # -> MapSync.cells

# INVESTIGATE task (seeker hovering at ~30 m over a candidate hotspot)
conf = HotspotConfirmer(load_classifier("perception/models/fire_cnn.onnx"))
conf.add_frame(frame, hotspot.position, gsd_m)       # each frame; then
result = conf.result()                               # CONFIRMED / FALSE_POSITIVE / None (keep looking)
# VERIFY task after a drop (seeker or extinguisher camera)
chk = SuppressionChecker(); chk.add_frame(frame, hotspot.position, gsd_m); chk.result()  # SUPPRESSED or None
```

Design decisions with demo talking points: `docs/decisions/` (0001 altitude vs lidar, 0002 sensor wiring and lens
distortion, 0003 hot-blob bypass, 0004 close-range confirmation).

Training and data: `exclude/training_README.md` and `exclude/DATA_SOURCES.md` (local only: `exclude/` is gitignored).

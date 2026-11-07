# perception/ (workstream C)

Drone-side perception. Imports only `contracts`, numpy and scipy (hard rule 1); onnxruntime/tifffile load lazily.

| Module | What |
|---|---|
| `detectors/threshold.py` | `ThresholdDetector`: classical, recall-first blob proposals (absolute temperature OR local contrast) |
| `detectors/fused.py` | `FusedDetector`: the mixed approach (proposals → verifier → detections, hot-bypass ≥ 300 °C) |
| `detectors/classifiers.py` | `OnnxCnnClassifier` (**primary**), `OnnxGbmClassifier` (**stretch**), `ConstantClassifier` (stub), `load_classifier()` |
| `detectors/fake.py` | `FakeDetector`: sim ground truth with misses, decoy hits and clutter (deterministic). **Use this from day one** |
| `georef.py` | pixel ↔ world (flat ground or DEM ray-march), frame footprint |
| `tracking.py` | `HotspotTracker`: detections over frames → `Hotspot` entities |
| `mapping.py` | `LogOddsMap`: NumPy log-odds layers → `GridCellUpdate` deltas for `MapSync` |
| `pipeline.py` | `SeekerPerception`: one call per frame, all of the above |
| `patches.py` / `features.py` | preprocessing shared with training (versioned) |
| `evaluation.py` / `evaluate.py` | metrics (F1, recall-first threshold, object matching, CIs) and the test-split evaluator |
| `synth.py` | synthetic radiometric night frames with exact masks (sub-pixel dilution, decoys) |
| `RESULTS.md`, `results/*.json` | evaluated numbers (written by scripts) |

## Using it from the sim or the drone loop (workstreams A/B)

```python
from perception.detectors import FakeDetector, FusedDetector, TruthSource, load_classifier
from perception.pipeline import SeekerPerception

# day one: ground-truth-driven detector (sim supplies current fire/decoy state)
det = FakeDetector(lambda now: [TruthSource("F1", fire_pos, area_m2=2.0, temp_c=450.0)], seed=scenario.seed)
# later: the real mixed detector
det = FusedDetector(load_classifier("perception/models/fire_cnn.onnx"))

perc = SeekerPerception(drone_id="seeker-1", detector=det, grid=scenario.grid)
changed_hotspots = perc.process(thermal_frame)   # -> publish each as HotspotReport
cells = perc.map.pop_updates()                   # -> MapSync.cells
```

Training and data: see `training/README.md` and `training/DATA_SOURCES.md`.

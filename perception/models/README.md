# Shipped verifier models

Commit only final, evaluated models here (each is ~0.1–1 MB). Every `.onnx` must have its sidecar `.json`
(same stem) written by the training script: it holds the patch-spec version, the recall-first threshold and the
CV summary, and `perception.detectors.load_classifier()` refuses a model without it.

| File | What | Produced by |
|---|---|---|
| `fire_cnn.onnx` + `.json` | **PRIMARY** CNN patch verifier | `python -m training.train_cnn ... --export-to perception/models/fire_cnn.onnx` |
| `fire_cnn_flame3comp.onnx` + `.json` | CNN trained on FLAME 3 Sycan **composites** (real No-Fire backgrounds + synthetic fires/decoys), scale-normalized. Test object F1 0.940 on held-out composites, 0.950 on the synthetic night test (RESULTS.md). Candidate for PRIMARY once checked on the hand-labelled hard band | `bash training/retrain_flame3.sh` (copied from `exclude/runs/cnn_flame3`) |
| `fire_cnn_mix.onnx` + `.json` | **Recommended (2026-11-08).** CNN trained on FLAME 3 NADIR plot 2 (history labels) + Sycan composites + Sycan weak labels, with hard examples sampled from the masks (`mix_v2`). Pooled test (NADIR plot 1 + composite + CV subset): pipeline F1 0.998 vs 0.952 for `fire_cnn_flame3comp`, 0 % fires missed vs 3.5 %; NADIR CNN recall 0.997 vs 0.73. Thresholds: reject 0.177, search = confirm 0.935 | `training/build_nadir.py` + `build_sycan_weak.py` + `eval_sets.write_mix_manifest`, then `train_cnn --hard-neg-per-frame 12 --hard-pos-per-frame 4 --max-gt-per-frame 6` (see `exclude/runs/cnn_mix_v2`) |
| `fire_gbm.onnx` + `.json` | **[STRETCH]** GBM feature verifier | `python -m training.train_gbm ... --export-to perception/models/fire_gbm.onnx` |

Models trained on synthetic data only are for wiring tests. Name them `*_synth.onnx` and don't present their
numbers as real-data results.

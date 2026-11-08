# Shipped verifier models

Commit only final, evaluated models here (each is ~0.1–1 MB). Every `.onnx` must have its sidecar `.json`
(same stem) written by the training script: it holds the patch-spec version, the recall-first threshold and the
CV summary, and `perception.detectors.load_classifier()` refuses a model without it.

| File | What | Produced by |
|---|---|---|
| `fire_cnn.onnx` + `.json` | **PRIMARY** CNN patch verifier | `python -m training.train_cnn ... --export-to perception/models/fire_cnn.onnx` |
| `fire_gbm.onnx` + `.json` | **[STRETCH]** GBM feature verifier | `python -m training.train_gbm ... --export-to perception/models/fire_gbm.onnx` |

Models trained on synthetic data only are for wiring tests. Name them `*_synth.onnx` and don't present their
numbers as real-data results.

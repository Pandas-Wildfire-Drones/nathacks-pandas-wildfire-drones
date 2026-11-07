"""Patch verifiers used by FusedDetector: given a frame and blob centres, return p(fire) per blob.

  * OnnxCnnClassifier  - PRIMARY. Small CNN trained in PyTorch (training/train_cnn.py), exported to ONNX.
  * OnnxGbmClassifier  - [STRETCH]. Gradient boosting on hand-made features (training/train_gbm.py).
  * ConstantClassifier - test/demo stub (lets the pipeline run before any model exists).

Every exported model ships with a sidecar JSON (same stem, `.json`) written by the training code:
    {"model_kind": "cnn" | "gbm", "patch_spec": {...}, "feature_version": "1" (gbm only),
     "threshold": 0.31, "threshold_rule": "recall>=0.95 then max precision", "input_name": "patches", ...}
The loader checks the patch-spec version so a model never runs on preprocessing it was not trained with.

onnxruntime is imported lazily (hard rule 1: perception imports only contracts/stdlib/numpy/scipy at module level).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..features import FEATURE_VERSION, features_at
from ..patches import PATCH_SPEC, PatchSpec, extract_patches

__all__ = [
    "PatchClassifier", "ConstantClassifier", "OnnxCnnClassifier", "OnnxGbmClassifier",
    "load_metadata", "load_classifier",
]


@runtime_checkable
class PatchClassifier(Protocol):
    threshold: float  # decision threshold on p(fire), chosen recall-first on validation data
    name: str

    def predict_proba(self, temp_c: np.ndarray, centers_uv: list[tuple[float, float]]) -> np.ndarray:
        """p(fire) in [0, 1] for each centre, shape (N,)."""
        ...


class ConstantClassifier:
    """Returns a fixed probability. Use for wiring tests, never for reported numbers."""

    def __init__(self, p: float = 1.0, threshold: float = 0.5):
        self.p, self.threshold, self.name = p, threshold, "constant"

    def predict_proba(self, temp_c, centers_uv):
        return np.full(len(centers_uv), self.p, dtype=np.float32)


def load_metadata(model_path: str | Path) -> dict:
    meta_path = Path(model_path).with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing model metadata sidecar {meta_path}")
    return json.loads(meta_path.read_text())


def _session(model_path: str | Path):
    import onnxruntime as ort  # lazy: optional dependency

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # deterministic + predictable latency
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class OnnxCnnClassifier:
    """PRIMARY verifier. ONNX input (N, C, P, P) float32 -> output (N, 1) logit."""

    def __init__(self, model_path: str | Path, threshold: float | None = None, batch_size: int = 256):
        self.meta = load_metadata(model_path)
        if self.meta.get("model_kind") != "cnn":
            raise ValueError(f"{model_path} is not a CNN model (model_kind={self.meta.get('model_kind')})")
        spec_d = self.meta["patch_spec"]
        if spec_d["version"] != PATCH_SPEC.version:
            raise ValueError(f"model trained with patch spec v{spec_d['version']}, code has v{PATCH_SPEC.version}")
        self.spec = PatchSpec(**spec_d)
        self.session = _session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = float(self.meta["threshold"] if threshold is None else threshold)
        self.batch_size = batch_size
        self.name = f"cnn:{Path(model_path).stem}"

    def predict_patches(self, x: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(x), self.batch_size):
            logits = self.session.run(None, {self.input_name: x[i:i + self.batch_size]})[0]
            out.append(_sigmoid(np.asarray(logits, dtype=np.float32).reshape(-1)))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def predict_proba(self, temp_c, centers_uv):
        return self.predict_patches(extract_patches(temp_c, centers_uv, self.spec))


class OnnxGbmClassifier:
    """[STRETCH] GBM verifier exported with skl2onnx. ONNX input (N, F) features -> probabilities."""

    def __init__(self, model_path: str | Path, threshold: float | None = None):
        self.meta = load_metadata(model_path)
        if self.meta.get("model_kind") != "gbm":
            raise ValueError(f"{model_path} is not a GBM model")
        if self.meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError("feature version mismatch between model and perception/features.py")
        self.spec = PatchSpec(**self.meta["patch_spec"])
        self.session = _session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = float(self.meta["threshold"] if threshold is None else threshold)
        self.name = f"gbm:{Path(model_path).stem}"

    def predict_features(self, feats: np.ndarray) -> np.ndarray:
        if len(feats) == 0:
            return np.zeros(0, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: feats.astype(np.float32)})
        probs = outputs[1]  # skl2onnx classifier: [labels, probabilities]
        if isinstance(probs, list):  # zipmap output: list of {class: prob}
            return np.array([p[1] for p in probs], dtype=np.float32)
        return np.asarray(probs, dtype=np.float32)[:, 1]

    def predict_proba(self, temp_c, centers_uv):
        return self.predict_features(features_at(temp_c, centers_uv, self.spec))


def load_classifier(model_path: str | Path, threshold: float | None = None) -> PatchClassifier:
    kind = load_metadata(model_path).get("model_kind")
    if kind == "cnn":
        return OnnxCnnClassifier(model_path, threshold)
    if kind == "gbm":
        return OnnxGbmClassifier(model_path, threshold)
    raise ValueError(f"unknown model_kind {kind!r}")

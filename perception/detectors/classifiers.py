"""Patch verifiers used by FusedDetector and HotspotConfirmer: frame + blob centres -> p(fire) per blob.

  * OnnxCnnClassifier  - PRIMARY. Small CNN trained in PyTorch (training/train_cnn.py), exported to ONNX.
  * OnnxGbmClassifier  - [STRETCH]. Gradient boosting on hand-made features (training/train_gbm.py).
  * ConstantClassifier - test/demo stub (lets the pipeline run before any model exists).

Every exported model ships with a sidecar JSON (same stem, `.json`) written by the training code:
    {"model_kind": "cnn" | "gbm", "patch_spec": {...}, "scale_normalized": true,
     "threshold": 0.31          # SEARCH operating point: recall-first (recall >= 0.95, then max precision)
     "confirm_threshold": 0.93  # CONFIRM operating point: precision-first (precision >= 0.98, then max recall)
     ...}
Two operating points, one model: searching should not miss fires; confirming (which can send an extinguisher)
should not cry wolf. See docs/decisions/0004-close-range-confirmation.md.
The loader checks the patch-spec version so a model never runs on preprocessing it was not trained with.

onnxruntime is imported lazily (hard rule 1: perception imports only contracts/stdlib/numpy/scipy at module level).
"""
from __future__ import annotations

import json
import warnings
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
    threshold: float          # search operating point (recall-first)
    confirm_threshold: float  # close-range CONFIRMED above this (precision-first, never below `threshold`)
    reject_threshold: float   # close-range FALSE_POSITIVE below this (keeps recall >= 0.99); between = inconclusive
    name: str

    def predict_proba(self, temp_c: np.ndarray, centers_uv: list[tuple[float, float]],
                      gsd_m: float | None = None) -> np.ndarray:
        """p(fire) in [0, 1] for each centre, shape (N,). gsd_m = metres per pixel of this frame, if known: a scalar
        (nadir drone camera) or one value per centre (oblique dataset frames, perception.patches.GsdArg)."""
        ...


class ConstantClassifier:
    """Returns a fixed probability. Use for wiring tests, never for reported numbers."""

    def __init__(self, p: float = 1.0, threshold: float = 0.5, confirm_threshold: float = 0.9,
                 reject_threshold: float | None = None):
        self.p, self.threshold, self.confirm_threshold, self.name = p, threshold, confirm_threshold, "constant"
        self.reject_threshold = threshold if reject_threshold is None else reject_threshold
        self.spec = PATCH_SPEC
        self.scale_normalized = True

    def predict_proba(self, temp_c, centers_uv, gsd_m=None):
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


class _Base:
    def _init_common(self, model_path, threshold, confirm_threshold, kind):
        self.meta = load_metadata(model_path)
        if self.meta.get("model_kind") != kind:
            raise ValueError(f"{model_path} is not a {kind} model (model_kind={self.meta.get('model_kind')})")
        spec_d = self.meta["patch_spec"]
        if spec_d["version"] != PATCH_SPEC.version:
            raise ValueError(f"model trained with patch spec v{spec_d['version']}, code has v{PATCH_SPEC.version}: retrain")
        self.spec = PatchSpec(**spec_d)
        self.scale_normalized = bool(self.meta.get("scale_normalized", False))
        self.session = _session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = float(self.meta["threshold"] if threshold is None else threshold)
        ct = self.meta.get("confirm_threshold", max(self.threshold, 0.5))
        self.confirm_threshold = float(ct if confirm_threshold is None else confirm_threshold)
        self.confirm_threshold = max(self.confirm_threshold, self.threshold)  # confirm is never laxer than search
        self.reject_threshold = min(float(self.meta.get("reject_threshold", self.threshold)), self.threshold)
        self.name = f"{kind}:{Path(model_path).stem}"
        self._warned = False

    def _gsd(self, gsd_m):
        """Use the GSD only if the model was trained scale-normalized (inputs must match training)."""
        if not self.scale_normalized:
            return None
        if gsd_m is None and not self._warned:
            warnings.warn(f"{self.name} was trained on scale-normalized patches but no GSD was given; "
                          "falling back to plain crops (accuracy will drop away from the training altitude)")
            self._warned = True
        return gsd_m


class OnnxCnnClassifier(_Base):
    """PRIMARY verifier. ONNX input (N, C, P, P) float32 -> output (N, 1) logit."""

    def __init__(self, model_path: str | Path, threshold: float | None = None,
                 confirm_threshold: float | None = None, batch_size: int = 256):
        self._init_common(model_path, threshold, confirm_threshold, "cnn")
        self.batch_size = batch_size

    def predict_patches(self, x: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(x), self.batch_size):
            logits = self.session.run(None, {self.input_name: x[i:i + self.batch_size]})[0]
            out.append(_sigmoid(np.asarray(logits, dtype=np.float32).reshape(-1)))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def predict_proba(self, temp_c, centers_uv, gsd_m=None):
        return self.predict_patches(extract_patches(temp_c, centers_uv, self.spec, self._gsd(gsd_m)))


class OnnxGbmClassifier(_Base):
    """[STRETCH] GBM verifier exported with skl2onnx. ONNX input (N, F) features -> probabilities."""

    def __init__(self, model_path: str | Path, threshold: float | None = None, confirm_threshold: float | None = None):
        self._init_common(model_path, threshold, confirm_threshold, "gbm")
        if self.meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError("feature version mismatch between model and perception/features.py: retrain")

    def predict_features(self, feats: np.ndarray) -> np.ndarray:
        if len(feats) == 0:
            return np.zeros(0, dtype=np.float32)
        probs = self.session.run(None, {self.input_name: feats.astype(np.float32)})[1]  # [labels, probabilities]
        if isinstance(probs, list):  # zipmap output: list of {class: prob}
            return np.array([p[1] for p in probs], dtype=np.float32)
        return np.asarray(probs, dtype=np.float32)[:, 1]

    def predict_proba(self, temp_c, centers_uv, gsd_m=None):
        return self.predict_features(features_at(temp_c, centers_uv, self.spec, self._gsd(gsd_m)))


def load_classifier(model_path: str | Path, threshold: float | None = None) -> PatchClassifier:
    kind = load_metadata(model_path).get("model_kind")
    if kind == "cnn":
        return OnnxCnnClassifier(model_path, threshold)
    if kind == "gbm":
        return OnnxGbmClassifier(model_path, threshold)
    raise ValueError(f"unknown model_kind {kind!r}")

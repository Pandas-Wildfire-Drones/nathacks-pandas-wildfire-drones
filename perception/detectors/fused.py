"""The mixed algorithmic + ML detector (DetectorKind.FUSED) chosen in Dev Report #2.

    frame --ThresholdDetector (recall stage)--> blobs --PatchClassifier (precision stage)--> detections

Decision per blob:
  * keep if classifier p(fire) >= classifier.threshold (chosen recall-first on validation data), or
  * keep regardless if peak >= hot_bypass_c: a blob that hot is unlikely to be a decoy, and the ML must not be
    able to veto it ([ASSUMPTION] safety rule; set hot_bypass_c=None to measure the classifier alone).
Detection.confidence = classifier probability (or max(p, bypass_confidence) for bypassed blobs).
"""
from __future__ import annotations

import numpy as np

from contracts import Detection, DetectorKind, ThermalFrame

from .classifiers import PatchClassifier
from .threshold import Blob, ThresholdDetector

__all__ = ["FusedDetector"]


class FusedDetector:
    kind = DetectorKind.FUSED

    def __init__(self, classifier: PatchClassifier, proposer: ThresholdDetector | None = None,
                 threshold: float | None = None, hot_bypass_c: float | None = 300.0,
                 bypass_confidence: float = 0.9):
        self.classifier = classifier
        self.proposer = proposer or ThresholdDetector()
        self.threshold = classifier.threshold if threshold is None else threshold
        self.hot_bypass_c = hot_bypass_c
        self.bypass_confidence = bypass_confidence

    def score_blobs(self, temp_c: np.ndarray) -> tuple[list[Blob], np.ndarray]:
        """All proposals with their classifier probabilities (used by evaluation to sweep thresholds)."""
        blobs = self.proposer.blobs(temp_c)
        probs = self.classifier.predict_proba(temp_c, [(b.u, b.v) for b in blobs])
        return blobs, np.asarray(probs, dtype=np.float32)

    def decide(self, blobs: list[Blob], probs: np.ndarray, threshold: float | None = None) -> list[Detection]:
        thr = self.threshold if threshold is None else threshold
        out = []
        for b, p in zip(blobs, probs):
            bypass = self.hot_bypass_c is not None and b.peak_temp_c >= self.hot_bypass_c
            if p >= thr or bypass:
                conf = max(float(p), self.bypass_confidence) if bypass else float(p)
                out.append(Detection(u=b.u, v=b.v, area_px=b.area_px, peak_temp_c=b.peak_temp_c,
                                     mean_temp_c=b.mean_temp_c, confidence=min(conf, 1.0),
                                     source=DetectorKind.FUSED, bbox=b.bbox))
        return out

    def detect(self, frame: ThermalFrame) -> list[Detection]:
        blobs, probs = self.score_blobs(frame.temp_c)
        return self.decide(blobs, probs)

"""The mixed algorithmic + ML detector (DetectorKind.FUSED) chosen in Dev Report #2.

    frame --ThresholdDetector (recall stage)--> blobs --PatchClassifier (precision stage)--> detections

Decision per blob: keep if classifier p(fire) >= threshold (recall-first, from training CV), OR if the
HOT-BLOB BYPASS fires. The bypass exists so the ML can never veto an obvious fire; it does not skip
confirmation (a bypassed blob is still only DETECTED and gets an INVESTIGATE pass before any water).
See docs/decisions/0003-hot-blob-bypass.md for the reasoning.

HotBypassRule (replaces the old fixed ">= 300 C" rule):
  * "saturated": >= min_hot_px pixels at >= sat_frac x the camera's saturation for its CURRENT gain mode.
    Tied to the sensor, not a magic number: Boson+ low gain tops out ~350 C, high gain ~150 C [S23], so a fixed
    300 C either never fires (high gain) or fires only just below saturation (low gain). min_hot_px stops a single
    hot/dead pixel from bypassing.
  * "oversize": the blob spans >= oversize_frac of the patch window AND its peak is >= oversize_min_c. The CNN's
    contrast channel compares the centre with the patch border; when the fire is bigger than the patch, the border
    is hot too and the CNN is blind. Size alone is NOT evidence of fire (a 3 m^2 sun-warmed rock or a vehicle is
    large too), hence the temperature floor: [ASSUMPTION] 150 C sits above night-time rocks/animals (25-45 C) and
    most vehicle bodies, and far below smouldering peaks (450-700 C [S24]).
  * Exclusion zones (friendly drones, known industrial heat such as gas flares) REMOVE THE BYPASS, never the
    detection: the blob still goes to the CNN. (An extinguisher hovering over the very fire it is fighting must not
    hide that fire.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts import Detection, DetectorKind, ThermalCameraSpec, ThermalFrame

from ..patches import patch_window_px, per_centre_gsd
from .classifiers import PatchClassifier
from .threshold import Blob, ThresholdDetector

__all__ = ["HotBypassRule", "ExclusionZone", "FusedDetector"]


@dataclass(frozen=True)
class ExclusionZone:
    """Image-space circle where the bypass is disabled. reason: 'friendly_drone', 'industrial', ..."""

    u: float
    v: float
    radius_px: float
    reason: str = "excluded"


@dataclass(frozen=True)
class HotBypassRule:
    saturation_c: float = 550.0   # [ASSUMPTION] matches ThermalCameraSpec default; set from the real camera + gain mode
    sat_frac: float = 0.9
    min_hot_px: int = 3
    oversize_frac: float = 0.75
    oversize_min_c: float = 150.0
    enabled: bool = True

    @property
    def trigger_c(self) -> float:
        return self.sat_frac * self.saturation_c

    @classmethod
    def from_camera(cls, spec: ThermalCameraSpec, **kw) -> "HotBypassRule":
        return cls(saturation_c=spec.saturation_temp_c, **kw)

    def with_saturation(self, saturation_c: float | None) -> "HotBypassRule":
        if saturation_c is None or saturation_c == self.saturation_c:
            return self
        return HotBypassRule(saturation_c, self.sat_frac, self.min_hot_px, self.oversize_frac, self.oversize_min_c,
                             self.enabled)

    def reason(self, temp_c: np.ndarray, blob: Blob, window_px: float,
               exclusions: list[ExclusionZone] | None = None) -> str | None:
        """'saturated' | 'oversize' | 'excluded:<why>' | None. Only the first two grant a bypass."""
        if not self.enabled:
            return None
        u0, v0, u1, v1 = (int(x) for x in blob.bbox)
        n_hot = int((temp_c[v0:v1, u0:u1] >= self.trigger_c).sum())
        size = max(u1 - u0, v1 - v0)
        big_and_hot = size >= self.oversize_frac * window_px and blob.peak_temp_c >= self.oversize_min_c
        why = "saturated" if n_hot >= self.min_hot_px else "oversize" if big_and_hot else None
        if why and exclusions:
            for z in exclusions:
                if (blob.u - z.u) ** 2 + (blob.v - z.v) ** 2 <= z.radius_px ** 2:
                    return f"excluded:{z.reason}"
        return why


class FusedDetector:
    kind = DetectorKind.FUSED

    def __init__(self, classifier: PatchClassifier, proposer: ThresholdDetector | None = None,
                 threshold: float | None = None, bypass: HotBypassRule | None = HotBypassRule(),
                 bypass_confidence: float = 0.9, ground_z: float = 0.0):
        self.classifier = classifier
        self.proposer = proposer or ThresholdDetector()
        self.threshold = classifier.threshold if threshold is None else threshold
        self.bypass = bypass or HotBypassRule(enabled=False)
        self.bypass_confidence = bypass_confidence
        self.ground_z = ground_z
        self.last_bypass_reasons: list[str | None] = []

    def _window_px(self, gsd_m: float | None) -> float:
        spec = getattr(self.classifier, "spec", None)
        scale_ok = getattr(self.classifier, "scale_normalized", False)
        return patch_window_px(gsd_m if scale_ok else None, spec) if spec is not None else 32.0

    @staticmethod
    def _blob_gsd(gsd_m, blobs: list[Blob]):
        """gsd_m may be a scalar (nadir), one value per blob, or a callable v -> GSD (oblique offline frames)."""
        if callable(gsd_m):
            return np.array([gsd_m(b.v) for b in blobs], dtype=np.float64)
        return gsd_m

    def score_blobs(self, temp_c: np.ndarray, gsd_m=None) -> tuple[list[Blob], np.ndarray]:
        """All proposals with their classifier probabilities (used by evaluation to sweep thresholds)."""
        blobs = self.proposer.blobs(temp_c)
        probs = self.classifier.predict_proba(temp_c, [(b.u, b.v) for b in blobs], self._blob_gsd(gsd_m, blobs))
        return blobs, np.asarray(probs, dtype=np.float32)

    def bypass_reasons(self, temp_c, blobs, gsd_m=None, exclusions=None, saturation_c=None) -> list[str | None]:
        rule = self.bypass.with_saturation(saturation_c)
        gs = per_centre_gsd(self._blob_gsd(gsd_m, blobs), len(blobs))
        return [rule.reason(temp_c, b, self._window_px(g), exclusions) for b, g in zip(blobs, gs)]

    def decide(self, temp_c: np.ndarray, blobs: list[Blob], probs: np.ndarray, gsd_m=None,
               exclusions: list[ExclusionZone] | None = None, threshold: float | None = None,
               saturation_c: float | None = None) -> list[Detection]:
        thr = self.threshold if threshold is None else threshold
        reasons = self.bypass_reasons(temp_c, blobs, gsd_m, exclusions, saturation_c)
        self.last_bypass_reasons = reasons
        out = []
        for b, p, why in zip(blobs, probs, reasons):
            bypass = why in ("saturated", "oversize")
            if p >= thr or bypass:
                conf = max(float(p), self.bypass_confidence) if bypass else float(p)
                out.append(Detection(u=b.u, v=b.v, area_px=b.area_px, peak_temp_c=b.peak_temp_c,
                                     mean_temp_c=b.mean_temp_c, confidence=min(conf, 1.0),
                                     source=DetectorKind.FUSED, bbox=b.bbox))
        return out

    def detect(self, frame: ThermalFrame, gsd_m: float | None = None, exclusions: list[ExclusionZone] | None = None,
               saturation_c: float | None = None) -> list[Detection]:
        """Detector protocol. Extra optional args come from SeekerPerception (FrameContext); without them the GSD
        is estimated from the pose height above `ground_z` (nadir camera)."""
        if gsd_m is None:
            gsd_m = frame.camera.ground_sample_distance_m(max(frame.pose.position.z - self.ground_z, 0.1))
        blobs, probs = self.score_blobs(frame.temp_c, gsd_m)
        return self.decide(frame.temp_c, blobs, probs, gsd_m, exclusions, saturation_c=saturation_c)

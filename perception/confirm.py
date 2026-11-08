"""Close-range perception tasks: INVESTIGATE (is this candidate a real fire?) and VERIFY (is it out after a drop?).

Why separate from search (docs/decisions/0004):
  * DIFFERENT QUESTION, DIFFERENT COST. Search must not miss fires (recall-first threshold). A confirmation can
    send an 800 kg extinguisher, so it must not cry wolf: it uses the model's PRECISION-FIRST `confirm_threshold`.
  * MORE EVIDENCE. The seeker hovers, so it sees the target in many frames. Scores are averaged over frames
    (mean probability, not noisy-OR: consecutive frames are correlated, not independent).
  * SAME MODEL. Patches are scale-normalized by GSD (perception/patches.py), so the CNN sees a 2.3 m ground window
    at 80 m search altitude and at 30 m confirm altitude alike. A separate confirm model only pays off if real
    close-range data shows the shared model failing there (measure it with perception.evaluate_confirm).
  * VERIFY IS PHYSICS, NOT ML. After a drop the question is "is anything near the target still hot?", answered with
    temperatures against the local background. Works with either drone's camera (seeker 640x512 or the
    extinguisher's 320x256). It reports SUPPRESSED at most: only humans declare VERIFIED_OUT (hard rule 10).

Both classes have an image-space API (add_uv: the caller knows where the target is in the frame) and a frame API
(add_frame: projects the target's world position with the pose, optional lens distortion).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from contracts import HotspotStatus, Position, ThermalFrame

from .detectors.classifiers import PatchClassifier
from .detectors.fused import HotBypassRule
from .detectors.threshold import ThresholdDetector
from .distortion import LensDistortion
from .georef import world_to_pixel
from .patches import patch_window_px

__all__ = ["ConfirmResult", "HotspotConfirmer", "SuppressionResult", "SuppressionChecker"]


@dataclass
class ConfirmResult:
    status: HotspotStatus | None  # CONFIRMED, FALSE_POSITIVE, or None = inconclusive (keep looking / escalate)
    n_frames: int
    mean_p: float
    seen_frac: float              # fraction of frames with a hot blob near the target
    median_peak_c: float | None
    reason: str


class HotspotConfirmer:
    """INVESTIGATE task. Feed frames while hovering over the candidate; read `result()` after each frame."""

    def __init__(self, classifier: PatchClassifier, proposer: ThresholdDetector | None = None,
                 bypass: HotBypassRule | None = HotBypassRule(), confirm_threshold: float | None = None,
                 reject_threshold: float | None = None, min_frames: int = 5, max_frames: int = 45,
                 search_radius_m: float = 3.0, min_seen_frac: float = 0.5):
        self.classifier = classifier
        self.proposer = proposer or ThresholdDetector()
        self.bypass = bypass or HotBypassRule(enabled=False)
        self.confirm_threshold = (getattr(classifier, "confirm_threshold", 0.9)
                                  if confirm_threshold is None else confirm_threshold)
        # Reject only below the model's reject threshold (chosen to keep recall >= 0.99): wrongly rejecting a real fire
        # is the dangerous error. Between reject and confirm -> inconclusive -> keep observing / escalate to a human.
        self.reject_threshold = (getattr(classifier, "reject_threshold", classifier.threshold)
                                 if reject_threshold is None else reject_threshold)
        # Confirm must be at least as strict as search (guards old sidecars / hand-set values).
        self.confirm_threshold = max(self.confirm_threshold, self.reject_threshold)
        self.min_frames, self.max_frames = min_frames, max_frames
        self.search_radius_m, self.min_seen_frac = search_radius_m, min_seen_frac
        self.reset()

    def reset(self) -> None:
        self._p: list[float] = []
        self._seen: list[bool] = []
        self._peaks: list[float] = []

    def add_uv(self, temp_c: np.ndarray, uv: tuple[float, float], gsd_m: float,
               saturation_c: float | None = None) -> None:
        """One frame where the target is expected at pixel uv. No hot blob within search_radius_m counts as p = 0."""
        blobs = self.proposer.blobs(temp_c)
        r_px = self.search_radius_m / gsd_m
        near = [b for b in blobs if (b.u - uv[0]) ** 2 + (b.v - uv[1]) ** 2 <= r_px ** 2]
        if not near:
            self._p.append(0.0)
            self._seen.append(False)
            return
        b = max(near, key=lambda x: x.peak_temp_c)
        p = float(self.classifier.predict_proba(temp_c, [(b.u, b.v)], gsd_m)[0])
        spec = getattr(self.classifier, "spec", None)
        window = patch_window_px(gsd_m if getattr(self.classifier, "scale_normalized", False) else None, spec) \
            if spec is not None else 32.0
        if self.bypass.with_saturation(saturation_c).reason(temp_c, b, window) in ("saturated", "oversize"):
            p = max(p, 0.9)
        self._p.append(p)
        self._seen.append(True)
        self._peaks.append(b.peak_temp_c)

    def add_frame(self, frame: ThermalFrame, target: Position, gsd_m: float,
                  distortion: LensDistortion | None = None, saturation_c: float | None = None) -> bool:
        """Returns False (frame ignored) if the target is outside this frame."""
        uv = world_to_pixel(target, frame.pose, frame.camera, distortion)
        if uv is None:
            return False
        self.add_uv(frame.temp_c, uv, gsd_m, saturation_c)
        return True

    def result(self) -> ConfirmResult:
        n = len(self._p)
        mean_p = float(np.mean(self._p)) if n else 0.0
        seen = float(np.mean(self._seen)) if n else 0.0
        peak = float(np.median(self._peaks)) if self._peaks else None
        if n < self.min_frames:
            return ConfirmResult(None, n, mean_p, seen, peak, f"need {self.min_frames} frames")
        if mean_p >= self.confirm_threshold and seen >= self.min_seen_frac:
            return ConfirmResult(HotspotStatus.CONFIRMED, n, mean_p, seen, peak, "precision-first threshold met")
        if mean_p < self.reject_threshold:
            return ConfirmResult(HotspotStatus.FALSE_POSITIVE, n, mean_p, seen, peak,
                                 "below the search threshold at close range")
        if n >= self.max_frames:
            return ConfirmResult(None, n, mean_p, seen, peak, "inconclusive after max frames: escalate to ground")
        return ConfirmResult(None, n, mean_p, seen, peak, "inconclusive: keep observing")


@dataclass
class SuppressionResult:
    status: HotspotStatus | None  # SUPPRESSED, or None = still hot / inconclusive
    n_frames: int
    hot_frac: float
    max_excess_c: float | None    # hottest reading above local background near the target
    reason: str


@dataclass
class SuppressionChecker:
    """VERIFY task after a drop. 'Hot' = something within radius_m of the target reads >= hot_margin_c above the
    frame's median background. [ASSUMPTION] margins; wet ground and steam cool quickly, smouldering roots don't."""

    radius_m: float = 2.0
    hot_margin_c: float = 15.0
    min_frames: int = 5
    suppressed_max_hot_frac: float = 0.1
    still_hot_min_frac: float = 0.5
    _hot: list[bool] = field(default_factory=list)
    _excess: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self._hot.clear()
        self._excess.clear()

    def add_uv(self, temp_c: np.ndarray, uv: tuple[float, float], gsd_m: float) -> None:
        r = max(int(np.ceil(self.radius_m / gsd_m)), 1)
        h, w = temp_c.shape
        u, v = int(uv[0]), int(uv[1])
        win = temp_c[max(v - r, 0):min(v + r + 1, h), max(u - r, 0):min(u + r + 1, w)]
        if win.size == 0:
            return
        excess = float(win.max() - np.median(temp_c))
        self._excess.append(excess)
        self._hot.append(excess >= self.hot_margin_c)

    def add_frame(self, frame: ThermalFrame, target: Position, gsd_m: float,
                  distortion: LensDistortion | None = None) -> bool:
        uv = world_to_pixel(target, frame.pose, frame.camera, distortion)
        if uv is None:
            return False
        self.add_uv(frame.temp_c, uv, gsd_m)
        return True

    def result(self) -> SuppressionResult:
        n = len(self._hot)
        frac = float(np.mean(self._hot)) if n else 0.0
        mx = max(self._excess) if self._excess else None
        if n < self.min_frames:
            return SuppressionResult(None, n, frac, mx, f"need {self.min_frames} frames")
        if frac <= self.suppressed_max_hot_frac:
            return SuppressionResult(HotspotStatus.SUPPRESSED, n, frac, mx,
                                     "cold: report SUPPRESSED (humans confirm VERIFIED_OUT)")
        if frac >= self.still_hot_min_frac:
            return SuppressionResult(None, n, frac, mx, "still hot: request another drop")
        return SuppressionResult(None, n, frac, mx, "intermittent heat: keep observing")

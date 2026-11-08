"""Turn labelled frames into labelled patches for the verifier, plus augmentation.

Key design choice: training patches are centred on the SAME classical proposals the FusedDetector will
produce at inference (ThresholdDetector.blobs). The verifier therefore learns exactly the decision it makes on
the drone: "is this proposed blob a fire?", including hard negatives (warm rocks, engines, animals).

Labelling rules per frame:
  mask present          proposals labelled by mask overlap (perception.evaluation.blob_label);
                        + `gt_per_object` jittered patches centred on each mask component (extra positives);
                        + `random_neg_per_frame` random patches away from fire (easy negatives).
  no mask, label = 0    every proposal is negative (+ random negatives).
  no mask, label = 1    ambiguous: a fire frame also contains warm non-fire blobs. Skipped unless
                        weak_labels=True, which labels all its proposals positive (noisy; prefer masks).

Scale normalization: if EVERY frame has a gsd_m (manifest column), patches are fixed-ground-size windows
(perception.patches.extract_patch_scaled) with a random +-20% GSD jitter per patch (altitude/AGL error), and the
model is marked scale_normalized. If any frame lacks a GSD, all patches are plain pixel crops (mixing the two
would teach the model two different scales).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from perception.datasets import FrameRecord, frame_label, load_frame, load_mask
from perception.detectors import ThresholdDetector
from perception.evaluation import blob_label, gt_objects_from_mask
from perception.patches import PATCH_SPEC, PatchSpec, extract_patch_scaled

__all__ = ["PatchSet", "build_patch_set", "augment_batch"]


@dataclass
class PatchSet:
    patches_c: np.ndarray   # (N, P, P) float32 degrees C (raw, NOT normalized: augmentation happens in C)
    y: np.ndarray           # (N,) int8
    groups: np.ndarray      # (N,) str: flight/burn id, used for GroupKFold
    origin: np.ndarray      # (N,) str: "proposal" | "gt_center" | "random_bg"
    stats: dict
    scale_normalized: bool = False

    def save(self, path) -> None:
        np.savez_compressed(path, patches_c=self.patches_c, y=self.y, groups=self.groups, origin=self.origin,
                            scale_normalized=np.array([self.scale_normalized]))

    @classmethod
    def load(cls, path) -> "PatchSet":
        d = np.load(path, allow_pickle=False)
        sn = bool(d["scale_normalized"][0]) if "scale_normalized" in d else False
        return cls(d["patches_c"], d["y"], d["groups"], d["origin"], {"loaded_from": str(path)}, sn)


def build_patch_set(records: list[FrameRecord], spec: PatchSpec = PATCH_SPEC, proposer: ThresholdDetector | None = None,
                    weak_labels: bool = False, gt_per_object: int = 2, gt_jitter_px: float = 2.0,
                    random_neg_per_frame: int = 1, seed: int = 0, gsd_jitter: float = 0.2,
                    scale_normalize: bool = True) -> PatchSet:
    """scale_normalize=False forces plain pixel crops even when GSDs are known (the A/B in decision 0004)."""
    rng = np.random.default_rng(seed)
    proposer = proposer or ThresholdDetector()
    scale_normalized = scale_normalize and all(r.gsd_m is not None for r in records)
    if scale_normalize and not scale_normalized and any(r.gsd_m is not None for r in records):
        print("[patch_data] some frames lack gsd_m: using plain pixel crops for ALL frames")

    def crop(temp, u, v, gsd):
        if not scale_normalized:
            return extract_patch_scaled(temp, u, v, None, spec)
        return extract_patch_scaled(temp, u, v, gsd * float(rng.uniform(1 - gsd_jitter, 1 + gsd_jitter)), spec)

    P, y, g, o = [], [], [], []
    skipped_ambiguous = 0
    for rec in records:
        temp = load_frame(rec)
        mask = load_mask(rec)
        label = frame_label(rec, mask)
        blobs = proposer.blobs(temp)
        if mask is None and label == 1 and not weak_labels:
            skipped_ambiguous += 1
            continue
        for b in blobs:
            lab = blob_label(b.bbox, mask) if mask is not None else label
            P.append(crop(temp, b.u, b.v, rec.gsd_m)); y.append(lab); g.append(rec.group); o.append("proposal")
        if mask is not None:
            for gt in gt_objects_from_mask(mask):
                for _ in range(gt_per_object):
                    du, dv = rng.normal(0, gt_jitter_px, 2)
                    P.append(crop(temp, gt.u + du, gt.v + dv, rec.gsd_m))
                    y.append(1); g.append(rec.group); o.append("gt_center")
        h, w = temp.shape
        avoid = ndimage.binary_dilation(mask, iterations=spec.size_px // 2) if mask is not None and label else None
        for _ in range(random_neg_per_frame):
            for _try in range(20):
                u, v = rng.uniform(0, w), rng.uniform(0, h)
                if avoid is None or not avoid[int(v), int(u)]:
                    P.append(crop(temp, u, v, rec.gsd_m)); y.append(0); g.append(rec.group); o.append("random_bg")
                    break
    if not P:
        raise ValueError("no patches built: check the manifest (labels/masks) and the threshold parameters")
    ps = PatchSet(np.stack(P).astype(np.float32), np.array(y, dtype=np.int8), np.array(g), np.array(o), {},
                  scale_normalized)
    ps.stats = {
        "frames": len(records), "frames_skipped_ambiguous": skipped_ambiguous, "patches": int(len(ps.y)),
        "positives": int(ps.y.sum()), "negatives": int((ps.y == 0).sum()), "groups": int(len(set(ps.groups))),
        "by_origin": {k: int((ps.origin == k).sum()) for k in ("proposal", "gt_center", "random_bg")},
        "proposal_positives": int(ps.y[ps.origin == "proposal"].sum()),
        "proposal_negatives_hard": int((ps.y[ps.origin == "proposal"] == 0).sum()),
        "scale_normalized": scale_normalized,
    }
    return ps


def augment_batch(x_c: np.ndarray, rng: np.random.Generator, ambient_sd_c: float = 3.0, noise_max_c: float = 0.2,
                  degrade_prob: float = 0.3) -> np.ndarray:
    """Augment raw degrees-C patches (N, P, P). Nadir camera => rotation/flip invariant.
      * rot90 / flip                     (heading-independent)
      * global offset N(0, ambient_sd)   (calibration drift, warmer/colder nights)
      * Gaussian noise up to noise_max   (NETD varies by core and temperature)
      * 2x resolution degradation        (higher altitude / wide lens -> coarser GSD; approximates sub-pixel
                                          dilution in C, the radiance-correct version lives in perception/synth.py)
    """
    x = x_c.copy()
    n, p, _ = x.shape
    k = rng.integers(0, 4, n)
    for kk in range(1, 4):
        sel = k == kk
        if sel.any():
            x[sel] = np.rot90(x[sel], kk, axes=(1, 2))
    flip = rng.random(n) < 0.5
    x[flip] = x[flip, :, ::-1]
    x += rng.normal(0, ambient_sd_c, (n, 1, 1)).astype(np.float32)
    x += (rng.normal(0, 1, x.shape) * rng.uniform(0, noise_max_c, (n, 1, 1))).astype(np.float32)
    deg = rng.random(n) < degrade_prob
    if deg.any() and p % 2 == 0:
        d = x[deg].reshape(-1, p // 2, 2, p // 2, 2).mean(axis=(2, 4))
        x[deg] = np.repeat(np.repeat(d, 2, axis=1), 2, axis=2)
    return x

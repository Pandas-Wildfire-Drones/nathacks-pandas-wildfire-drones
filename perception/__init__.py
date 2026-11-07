"""Perception (workstream C): thermal detection, georeferencing, mapping, tracking and evaluation.

Hard rule 1 (context.md section 4): modules here import only `contracts`, the stdlib, numpy and scipy.
Optional runtimes (onnxruntime, tifffile) are imported lazily inside the few functions that need them,
so the swarm core and the simulator can import perception without installing ML packages.

Pipeline (mixed algorithmic + ML, DetectorKind.FUSED):

    ThermalFrame --ThresholdDetector--> candidate blobs (recall-first, classical)
                 --PatchClassifier-->   p(fire) per blob (CNN in ONNX; GBM = STRETCH), scale-normalized by GSD
                 --FusedDetector-->     Detection list (image coords)
                 --Georeferencer-->     world Position per detection
                 --HotspotTracker-->    Hotspot entities (fused over frames)
                 --LogOddsMap-->        GridCellUpdate deltas for MapSync

    Close range (INVESTIGATE / VERIFY tasks): perception/confirm.py, same CNN, separate thresholds, multi-frame.
"""
from .patches import (
    PATCH_SPEC, PatchSpec, extract_patch, extract_patch_scaled, extract_patches, normalize_patch, normalize_patches,
)

__all__ = ["PATCH_SPEC", "PatchSpec", "extract_patch", "extract_patch_scaled", "extract_patches", "normalize_patch",
           "normalize_patches"]

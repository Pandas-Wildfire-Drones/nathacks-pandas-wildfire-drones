"""Build a manifest.csv for FLAME 3 (or any folder of radiometric thermal TIFFs).

[VERIFY after download] The folder layout below is a GUESS from the paper and dataset page: each sample is a
"quartet" (raw RGB, raw thermal JPG, corrected-FOV RGB, Celsius thermal TIFF); the Kaggle CV subset sorts them
into "Fire" / "No Fire". Run with --inspect first, check the printout, then adjust the patterns.

    python -m training.build_manifest_flame3 --root exclude/data/flame3_raw --inspect
    python -m training.build_manifest_flame3 --root exclude/data/flame3_raw --out exclude/data/flame3/manifest.csv \
        --group-by sequence --block 50 [--mask-dir exclude/data/flame3_raw/masks]

Grouping (matters a lot, context.md section 10):
  --group-by parent    one group per parent folder (use when folders = flights/burns)
  --group-by sequence  sort by filename (time order) and cut into blocks of --block consecutive frames.
                       Use for the single-burn Kaggle subset: neighbouring frames stay in the same group.
Labels: from the nearest folder name ("No Fire"/"NoFire"/"no_fire" -> 0, "Fire" -> 1), or from masks if given.
  FLAME 3's Fire/No-Fire labels are a max-temperature heuristic (<~80 C no fire, >~200 C fire) plus manual review
  of the band in between; any pixel masks should be assumed threshold-derived. See training/FLAME3.md.

Dataset citation (requested by the FLAME 3 authors; also in training/FLAME3.md):
  Hopkins, B., ONeill, L., Marinaccio, M., Rowell, E., Parsons, R., Flanary, S., Nazim I, Seielstad C, Afghah, F.
  (2024). FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management.
  arXiv preprint arXiv:2412.02831.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from perception.datasets import FrameRecord, write_manifest

NEG = re.compile(r"no[\s_-]*fire", re.I)
POS = re.compile(r"fire", re.I)


def folder_label(path: Path) -> int | None:
    for part in reversed(path.parts[:-1]):
        if NEG.search(part):
            return 0
        if POS.search(part):
            return 1
    return None


def find_tiffs(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in (".tif", ".tiff") and re.search(pattern, str(p), re.I))


def inspect(paths: list[Path], n: int = 5) -> None:
    import tifffile

    print(f"{len(paths)} TIFFs found. Sample:")
    for p in paths[: n]:
        a = tifffile.imread(p).astype(np.float64)
        print(f"  {p}  shape={a.shape} dtype={a.dtype} min={a.min():.1f} median={np.median(a):.1f} max={a.max():.1f}"
              f"  label={folder_label(p)}")
    print("Check: values must be degrees C (night ground ~ -5..20, fire 100+). Values ~27000-30000 mean centi-Kelvin;"
          " ~270-300 mean Kelvin: convert before training (add a conversion step, don't guess silently).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out")
    ap.add_argument("--pattern", default=r"therm|tif", help="regex a TIFF path must match (filters out RGB TIFFs)")
    ap.add_argument("--group-by", choices=["parent", "sequence"], default="sequence")
    ap.add_argument("--block", type=int, default=50, help="frames per group for --group-by sequence")
    ap.add_argument("--mask-dir", help="folder of masks with the same file stem as the frames")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    paths = find_tiffs(root, a.pattern)
    if a.inspect or not a.out:
        inspect(paths)
        return
    masks = {}
    if a.mask_dir:
        masks = {p.stem: p for p in Path(a.mask_dir).rglob("*") if p.suffix.lower() in (".png", ".tif", ".tiff", ".npy")}
    recs = []
    for i, p in enumerate(paths):
        lab = folder_label(p)
        mask = masks.get(p.stem)
        if lab is None and mask is None:
            print(f"skip (no label): {p}")
            continue
        group = p.parent.name if a.group_by == "parent" else f"{folder_label(p)}-block{i // a.block:04d}"
        recs.append(FrameRecord(frame_path=str(p), group=group, mask_path=str(mask) if mask else "",
                                label=lab, source="flame3"))
    write_manifest(a.out, recs)
    print(f"wrote {len(recs)} rows, {len({r.group for r in recs})} groups -> {a.out}")


if __name__ == "__main__":
    main()

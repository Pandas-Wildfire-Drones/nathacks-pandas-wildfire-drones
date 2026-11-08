"""Assign a group-wise train/test split to a manifest (whole flights/burns go to one side).

    python -m training.make_split --manifest exclude/data/flame3/manifest.csv --test-frac 0.2 --seed 0
    -> exclude/data/flame3/manifest_split.csv

Run once per dataset and keep the seed fixed: every later training run and the final evaluation use this file.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from perception.datasets import assign_group_split, frame_label, load_manifest, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    recs = assign_group_split(load_manifest(a.manifest), a.test_frac, a.seed)
    out = Path(a.out or Path(a.manifest).with_name("manifest_split.csv"))
    write_manifest(out, recs)
    for split in ("train", "test"):
        rs = [r for r in recs if r.split == split]
        labels = Counter(frame_label(r) for r in rs)
        print(f"{split}: {len(rs)} frames, {len({r.group for r in rs})} groups, fire={labels[1]} no_fire={labels[0]}")
    n_groups = len({r.group for r in recs})
    if n_groups < 6:
        print(f"WARNING: only {n_groups} groups. CV and the test estimate will be noisy; consider finer groups "
              "(e.g. --group-by sequence in the manifest builder) or more data.")
    print(out)


if __name__ == "__main__":
    main()

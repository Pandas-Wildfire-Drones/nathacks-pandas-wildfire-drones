"""Hand-label a small REAL test set in the 80-200 C band, judged from the paired RGB photo (training/FLAME3.md).

Why: FLAME 3 says which FRAMES contain fire, never which BLOB. Our verifier decides per blob, and the 80-200 C band
(smouldering spots, sun-warmed logs and rocks, cooling ash) is where that decision matters and where FLAME 3's labels
are a heuristic or missing. A person looking at the RGB photo is an independent judge, because the detector never
sees RGB. This set is for TESTING only; training uses composites (training/composite.py).

Three steps:
  1. candidates   python -m training.label_tool candidates --manifest exclude/data/flame3/manifest.csv \
                      --out exclude/labels/candidates.csv --rgb-box 0.1638 0.1511 0.8004 0.8289 [--n 300]
     Runs the ThresholdDetector on Fire frames and keeps blobs whose peak is in [--band-lo, --band-hi). Then it samples
     --n of them, spread across groups: consecutive frames are near-duplicates, so --max-per-group caps each group.
     CLIPPED blobs (peak within --clip-margin-c of the frame's saturation_c) are skipped unless --include-clipped:
     their true peak is unknown (>= the clip), so they are not "in the band". On FLAME 3 Sycan this matters: the
     high-gain mode clips at 187.4 C, and every Fire frame whose max is in 80-200 C is a high-gain frame.
     RGB pairing: the manifest's rgb_path (training/build_manifest_flame3.py writes the 4000x3000 raw photo). Without
     it: --rgb-dir, matched by file stem after optional --rgb-replace OLD NEW substitutions.
     --rgb-box x0 y0 x1 y1: where the thermal field of view sits inside the RGB image, as fractions of its width and
     height. Default 0 0 1 1 = an FOV-corrected pair. FLAME 3 Sycan raw RGB (M30T wide camera), measured by matching
     the "Corrected FOV" crops on 12 frames: 0.1638 0.1511 0.8004 0.8289 (sd 0.004, i.e. +-2-3 thermal px).
  2. label        python -m training.label_tool label --candidates exclude/labels/candidates.csv --labeler alice
     Window: thermal context, thermal close-up, RGB of the same spot, previous and next frame. Keys:
       a = active fire (flame / glow)          s = smouldering / residual heat (burnt, no flame)
       n = not fire (rock, log, soil, person, vehicle...)       u = unsure (excluded from metrics)
       left or b = back                         q = save and quit
     Saved after every key; re-running resumes. The temperature scale is hidden by default (--show-temp shows it),
     so the label comes from the photo and not from "it's hot, so it's fire".
  3. agree        python -m training.label_tool agree exclude/labels/labels_alice.csv exclude/labels/labels_bob.csv
     Cohen's kappa (3-class and fire/not-fire) on the candidates both people labelled. Everyone sees the same order,
     so a second person only needs to do the first ~50.
Scoring a model on the labels: training/train_verifier.ipynb, section "Hand-labelled hard band".
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from perception.datasets import FrameRecord, frame_label, load_frame, load_manifest
from perception.detectors import ThresholdDetector

__all__ = ["Candidate", "LABELS", "find_candidates", "write_candidates", "load_candidates", "load_labels",
           "LabelSession", "cohen_kappa", "agreement", "labelled_targets", "predict_candidates"]

LABELS = {"a": "active", "s": "smoulder", "n": "not_fire", "u": "unsure"}
_IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class Candidate:
    cand_id: str
    frame_path: str
    group: str
    u: float
    v: float
    peak_c: float
    area_px: int
    gsd_m: float | None = None
    rgb_path: str = ""
    prev_path: str = ""
    next_path: str = ""
    saturation_c: float | None = None
    rgb_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)   # thermal FOV inside the RGB, fractions


# ---------- step 1: candidates ----------

def _rgb_index(rgb_dir: str | Path | None) -> dict[str, Path]:
    if not rgb_dir:
        return {}
    idx: dict[str, Path] = {}
    for p in sorted(Path(rgb_dir).rglob("*")):
        if p.suffix.lower() in _IMG_EXT:
            idx.setdefault(p.stem.lower(), p)
    return idx


def _rgb_for(frame_path: str, idx: dict[str, Path], replace: list[tuple[str, str]]) -> str:
    stem = Path(frame_path).stem
    for old, new in replace:
        stem = stem.replace(old, new)
    hit = idx.get(stem.lower())
    return str(hit) if hit is not None and str(hit) != str(Path(frame_path)) else ""


def _time_key(r: FrameRecord):
    """Capture order: timestamp when the manifest has it (FLAME 3 file numbers are shuffled), else the path."""
    return (0, r.timestamp, r.frame_path) if r.timestamp is not None else (1, 0.0, r.frame_path)


def _cand_id(r: FrameRecord, j: int) -> str:
    """Unique across class folders: FLAME 3 stems restart in Fire/ and No Fire/."""
    parts = Path(r.frame_path).parts
    cls = next((p for p in reversed(parts[:-1]) if p.lower().replace(" ", "") in ("fire", "nofire")), "")
    return f"{cls.replace(' ', '')}_{Path(r.frame_path).stem}#{j}" if cls else f"{Path(r.frame_path).stem}#{j}"


def find_candidates(records: list[FrameRecord], band: tuple[float, float] = (80.0, 200.0), n: int = 300,
                    max_per_group: int = 15, rgb_dir: str | Path | None = None,
                    rgb_replace: list[tuple[str, str]] | None = None, detector: ThresholdDetector | None = None,
                    seed: int = 0, include_clipped: bool = False, clip_margin_c: float = 0.5,
                    rgb_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> list[Candidate]:
    """Band blobs from Fire frames (label 1), sampled round-robin over groups in a seeded order."""
    det = detector or ThresholdDetector()
    idx = _rgb_index(rgb_dir)
    rep = rgb_replace or []
    by_group: dict[str, list[FrameRecord]] = defaultdict(list)
    for r in records:
        by_group[r.group].append(r)
    pool: dict[str, list[Candidate]] = defaultdict(list)
    for g, recs in by_group.items():
        recs = sorted(recs, key=_time_key)
        for i, r in enumerate(recs):
            if frame_label(r) != 1:
                continue
            temp = load_frame(r)
            for j, b in enumerate(det.blobs(temp)):
                if not band[0] <= b.peak_temp_c < band[1]:
                    continue
                if (not include_clipped and r.saturation_c is not None
                        and b.peak_temp_c >= r.saturation_c - clip_margin_c):
                    continue
                gsd = r.gsd_at(b.v, temp.shape[0]) if r.has_gsd else None
                pool[g].append(Candidate(
                    cand_id=_cand_id(r, j), frame_path=r.frame_path, group=g,
                    u=round(float(b.u), 2), v=round(float(b.v), 2), peak_c=round(float(b.peak_temp_c), 2),
                    area_px=int(b.area_px), gsd_m=None if gsd is None else round(gsd, 4),
                    rgb_path=r.rgb_path or _rgb_for(r.frame_path, idx, rep),
                    prev_path=recs[i - 1].frame_path if i > 0 else "",
                    next_path=recs[i + 1].frame_path if i + 1 < len(recs) else "",
                    saturation_c=r.saturation_c, rgb_box=tuple(rgb_box)))
    rng = np.random.default_rng(seed)
    for g in pool:
        rng.shuffle(pool[g])
        pool[g] = pool[g][:max_per_group]
    groups = sorted(pool)
    rng.shuffle(groups)
    out: list[Candidate] = []
    while len(out) < n and any(pool[g] for g in groups):
        for g in groups:
            if pool[g] and len(out) < n:
                out.append(pool[g].pop())
    return out


def _rel(p: str, base: Path) -> str:
    if not p:
        return ""
    q = Path(p).resolve()
    return str(q.relative_to(base)) if q.is_relative_to(base) else str(q)


def write_candidates(path: str | Path, cands: list[Candidate]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.parent.resolve()
    names = [f.name for f in fields(Candidate)]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        for c in cands:
            d = asdict(c)
            for k in ("frame_path", "rgb_path", "prev_path", "next_path"):
                d[k] = _rel(d[k], base)
            d["rgb_box"] = " ".join(f"{x:.4f}" for x in c.rgb_box)
            w.writerow({k: ("" if v is None else v) for k, v in d.items()})


def load_candidates(path: str | Path) -> list[Candidate]:
    path = Path(path)
    base = path.parent

    def res(p: str) -> str:
        return "" if not p else str(p if Path(p).is_absolute() else base / p)

    with path.open(newline="") as fh:
        return [Candidate(cand_id=r["cand_id"], frame_path=res(r["frame_path"]), group=r["group"], u=float(r["u"]),
                          v=float(r["v"]), peak_c=float(r["peak_c"]), area_px=int(r["area_px"]),
                          gsd_m=float(r["gsd_m"]) if r.get("gsd_m") else None, rgb_path=res(r.get("rgb_path", "")),
                          prev_path=res(r.get("prev_path", "")), next_path=res(r.get("next_path", "")),
                          saturation_c=float(r["saturation_c"]) if r.get("saturation_c") else None,
                          rgb_box=tuple(float(x) for x in r["rgb_box"].split()) if r.get("rgb_box")
                          else (0.0, 0.0, 1.0, 1.0))
                for r in csv.DictReader(fh)]


# ---------- step 2: labelling ----------

_LABEL_COLS = ["cand_id", "label", "labeler", "rgb_available", "labelled_at"]


def load_labels(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {r["cand_id"]: r for r in csv.DictReader(fh)}


def _crop(a: np.ndarray, u: float, v: float, half: int) -> np.ndarray:
    """Window centred on (u, v), edge-padded so the blob stays centred near image borders."""
    pad = np.pad(a, [(half, half), (half, half)] + [(0, 0)] * (a.ndim - 2), mode="edge")
    r, c = int(round(v)) + half, int(round(u)) + half
    return pad[r - half:r + half + 1, c - half:c + half + 1]


class LabelSession:
    """Keyboard labelling over a candidates CSV. `handle_key` holds the logic; `run` only wires it to matplotlib."""

    def __init__(self, candidates_csv: str | Path, labels_csv: str | Path, labeler: str, show_temp: bool = False,
                 context_px: int = 48, close_px: int = 16, rgb_margin_px: int = 20):
        self.cands = load_candidates(candidates_csv)
        self.labels_csv = Path(labels_csv)
        self.labeler = labeler
        self.show_temp = show_temp
        self.context_px, self.close_px, self.rgb_margin_px = context_px, close_px, rgb_margin_px
        self.labels = load_labels(self.labels_csv)
        self.i = next((k for k, c in enumerate(self.cands) if c.cand_id not in self.labels), len(self.cands))
        self._cache: dict[str, np.ndarray] = {}
        self._rgb_cache: dict[str, np.ndarray] = {}

    def _rgb(self, path: str) -> np.ndarray:
        if path not in self._rgb_cache:
            if len(self._rgb_cache) >= 3:                        # 4000x3000 photos are 36 MB each
                self._rgb_cache.pop(next(iter(self._rgb_cache)))
            self._rgb_cache[path] = _load_rgb(path)
        return self._rgb_cache[path]

    @property
    def done(self) -> bool:
        return self.i >= len(self.cands)

    def save(self) -> None:
        self.labels_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.labels_csv.with_suffix(".tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_LABEL_COLS)
            w.writeheader()
            order = {c.cand_id: k for k, c in enumerate(self.cands)}
            for row in sorted(self.labels.values(), key=lambda r: order.get(r["cand_id"], 1 << 30)):
                w.writerow({k: row.get(k, "") for k in _LABEL_COLS})
        tmp.replace(self.labels_csv)   # atomic: a crash never leaves a half-written file

    def handle_key(self, key: str) -> bool:
        """Apply one key. Returns False when the session should close."""
        if key == "q":
            self.save()
            return False
        if key in ("left", "b"):
            self.i = max(self.i - 1, 0)
            return True
        if key in LABELS and not self.done:
            c = self.cands[self.i]
            self.labels[c.cand_id] = {"cand_id": c.cand_id, "label": LABELS[key], "labeler": self.labeler,
                                      "rgb_available": int(bool(c.rgb_path)),
                                      "labelled_at": dt.datetime.now().isoformat(timespec="seconds")}
            self.save()
            self.i += 1
            nxt = next((k for k in range(self.i, len(self.cands)) if self.cands[k].cand_id not in self.labels), None)
            self.i = len(self.cands) if nxt is None else nxt
        return not self.done

    def _thermal(self, path: str) -> np.ndarray | None:
        if not path:
            return None
        if path not in self._cache:
            if len(self._cache) >= 6:                            # current + neighbours, a few steps back
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = load_frame(FrameRecord(frame_path=path, group="", label=0))
        return self._cache[path]

    def render(self, fig) -> None:
        fig.clf()
        if self.done:
            fig.text(0.5, 0.5, f"All {len(self.cands)} candidates labelled. Press q.", ha="center", va="center")
            return
        c = self.cands[self.i]
        temp = self._thermal(c.frame_path)
        gs = fig.add_gridspec(2, 3)
        ctx = _crop(temp, c.u, c.v, self.context_px)
        close = _crop(temp, c.u, c.v, self.close_px)
        lo = float(np.percentile(ctx, 5))
        hi = max(float(ctx.max()), lo + 1.0)
        for ax_pos, img, title in ((gs[0, 0], ctx, "thermal: context"), (gs[0, 1], close, "thermal: close-up")):
            ax = fig.add_subplot(ax_pos)
            im = ax.imshow(img, cmap="inferno", vmin=lo, vmax=hi)
            h = img.shape[0] // 2
            ax.add_patch(_circle(fig, (h, h), max(3, h // 6)))
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            if self.show_temp:
                fig.colorbar(im, ax=ax, fraction=0.046, label="°C")
        ax = fig.add_subplot(gs[0, 2])
        if c.rgb_path:
            rgb = self._rgb(c.rgb_path)
            x0, y0, x1, y1 = c.rgb_box   # thermal FOV inside the RGB image (fractions)
            sx = (x1 - x0) * rgb.shape[1] / temp.shape[1]   # RGB pixels per thermal pixel
            sy = (y1 - y0) * rgb.shape[0] / temp.shape[0]
            ru, rv = x0 * rgb.shape[1] + (c.u + 0.5) * sx, y0 * rgb.shape[0] + (c.v + 0.5) * sy
            half = int(round((self.context_px + self.rgb_margin_px) * max(sx, sy)))
            ax.imshow(_crop(rgb, ru, rv, half))
            ax.add_patch(_circle(fig, (half, half), max(4, int(self.close_px * max(sx, sy)))))
            ax.set_title(f"RGB (±{self.rgb_margin_px} px misalignment margin)", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no RGB match\n(label 'u' unless obvious)", ha="center", va="center")
        ax.set_xticks([]), ax.set_yticks([])
        for k, (path, title) in enumerate(((c.prev_path, "previous frame, same pixels"), (c.next_path, "next frame, same pixels"))):
            ax = fig.add_subplot(gs[1, k])
            t = self._thermal(path) if path else None
            if t is not None and t.shape == temp.shape:
                ax.imshow(_crop(t, c.u, c.v, self.close_px), cmap="inferno", vmin=lo, vmax=hi)
            else:
                ax.text(0.5, 0.5, "none", ha="center", va="center")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
        ax = fig.add_subplot(gs[1, 2])
        ax.axis("off")
        n_done = sum(1 for x in self.cands if x.cand_id in self.labels)
        prev = self.labels.get(c.cand_id, {}).get("label", "-")
        peak = f"peak {c.peak_c:.0f} °C · " if self.show_temp else ""
        ax.text(0, 1, f"{c.cand_id}\ngroup {c.group}\n{peak}{c.area_px} px\n\n"
                      f"{n_done}/{len(self.cands)} labelled · current: {prev}\n\n"
                      "a  active fire (flame / glow)\ns  smouldering / residual heat\nn  not fire\nu  unsure\n"
                      "b / ←  back      q  save + quit", va="top", family="monospace", fontsize=9)

    def run(self) -> None:  # pragma: no cover - interactive
        import matplotlib.pyplot as plt

        for k in ("a", "s", "n", "u", "b", "q"):           # stop matplotlib's own shortcuts from stealing keys
            for name, keys in list(plt.rcParams.items()):
                if name.startswith("keymap.") and k in keys:
                    plt.rcParams[name] = [x for x in keys if x != k]
        fig = plt.figure(figsize=(12, 7))

        def on_key(event):
            if not self.handle_key(event.key or ""):
                plt.close(fig)
                return
            self.render(fig)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key)
        self.render(fig)
        plt.show()
        self.save()
        print(f"saved {len(self.labels)} labels to {self.labels_csv}")


def _circle(fig, xy, r):
    from matplotlib.patches import Circle

    return Circle(xy, r, fill=False, ec="cyan", lw=1.2)


def _load_rgb(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


# ---------- step 3: agreement + scoring helpers ----------

def cohen_kappa(a: list[str], b: list[str]) -> float:
    if not a:
        return float("nan")
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in cats) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(labels_a: str | Path, labels_b: str | Path, positive=("active", "smoulder")) -> dict:
    """Kappa on candidates both labelled (unsure excluded). >0.8 = strong, 0.6-0.8 = usable, <0.6 = redefine labels."""
    la, lb = load_labels(labels_a), load_labels(labels_b)
    common = [k for k in la if k in lb and "unsure" not in (la[k]["label"], lb[k]["label"])]
    a = [la[k]["label"] for k in common]
    b = [lb[k]["label"] for k in common]
    fa = ["fire" if x in positive else "not" for x in a]
    fb = ["fire" if x in positive else "not" for x in b]
    return {"n_common": len(common), "kappa_3class": cohen_kappa(a, b), "kappa_fire_vs_not": cohen_kappa(fa, fb),
            "confusion": dict(Counter(zip(a, b))), "agree_frac": float(np.mean([x == y for x, y in zip(a, b)]))
            if common else float("nan")}


def labelled_targets(cands: list[Candidate], labels: dict[str, dict], positive=("active", "smoulder")):
    """(candidates, y) for labelled, non-unsure candidates. `positive` decides what counts as fire."""
    keep = [(c, int(labels[c.cand_id]["label"] in positive)) for c in cands
            if c.cand_id in labels and labels[c.cand_id]["label"] != "unsure"]
    return [c for c, _ in keep], np.array([y for _, y in keep], dtype=int)


def predict_candidates(classifier, cands: list[Candidate]) -> np.ndarray:
    """p(fire) per candidate from any PatchClassifier, loading each frame once."""
    out = np.zeros(len(cands), dtype=np.float32)
    by_frame: dict[str, list[int]] = defaultdict(list)
    for k, c in enumerate(cands):
        by_frame[c.frame_path].append(k)
    for path, ks in by_frame.items():
        temp = load_frame(FrameRecord(frame_path=path, group="", label=0))
        centers = [(cands[k].u, cands[k].v) for k in ks]
        gsds = [cands[k].gsd_m for k in ks]           # per candidate: oblique frames change GSD row by row
        gsd = None if any(g is None for g in gsds) else gsds
        try:
            out[ks] = classifier.predict_proba(temp, centers, gsd_m=gsd)
        except TypeError:   # classifiers from before PatchSpec v2 take no GSD
            out[ks] = classifier.predict_proba(temp, centers)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--manifest", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--rgb-dir")
    c.add_argument("--rgb-replace", nargs=2, action="append", default=[], metavar=("OLD", "NEW"),
                   help="substitute in the thermal stem to get the RGB stem (repeatable)")
    c.add_argument("--band-lo", type=float, default=80.0)
    c.add_argument("--band-hi", type=float, default=200.0)
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--max-per-group", type=int, default=15)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--include-clipped", action="store_true", help="keep blobs whose peak is at the gain clip")
    c.add_argument("--clip-margin-c", type=float, default=0.5)
    c.add_argument("--rgb-box", type=float, nargs=4, default=(0.0, 0.0, 1.0, 1.0), metavar=("X0", "Y0", "X1", "Y1"),
                   help="thermal FOV inside the RGB image, fractions (FLAME 3 Sycan raw: 0.1638 0.1511 0.8004 0.8289)")
    l_ = sub.add_parser("label")
    l_.add_argument("--candidates", required=True)
    l_.add_argument("--labeler", required=True)
    l_.add_argument("--labels", help="default: labels_<labeler>.csv next to the candidates")
    l_.add_argument("--show-temp", action="store_true")
    g = sub.add_parser("agree")
    g.add_argument("labels_a")
    g.add_argument("labels_b")
    a = ap.parse_args()
    if a.cmd == "candidates":
        cands = find_candidates(load_manifest(a.manifest), (a.band_lo, a.band_hi), a.n, a.max_per_group, a.rgb_dir,
                                [tuple(x) for x in a.rgb_replace], seed=a.seed, include_clipped=a.include_clipped,
                                clip_margin_c=a.clip_margin_c, rgb_box=tuple(a.rgb_box))
        write_candidates(a.out, cands)
        with_rgb = sum(bool(x.rgb_path) for x in cands)
        print(f"{len(cands)} candidates from {len({x.group for x in cands})} groups -> {a.out}; "
              f"RGB matched for {with_rgb}/{len(cands)}")
        if with_rgb < len(cands):
            print("  unmatched RGB: check the file naming and pass --rgb-replace OLD NEW")
    elif a.cmd == "label":
        labels = a.labels or str(Path(a.candidates).with_name(f"labels_{a.labeler}.csv"))
        LabelSession(a.candidates, labels, a.labeler, a.show_temp).run()
    else:
        for k, v in agreement(a.labels_a, a.labels_b).items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

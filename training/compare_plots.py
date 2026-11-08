"""Plots for training/compare_approaches.ipynb (matplotlib; ipywidgets optional for the threshold sliders).

Colours (validated categorical palette, fixed order, never cycled):
  classes     not fire = blue, fire = orange
  approaches  algorithmic = aqua solid, CNN alone = yellow dashed, full pipeline = magenta dotted
Line styles double the colour so identity never depends on colour alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from perception.datasets import FrameRecord, load_frame
from perception.evaluation import _pr_curve
from perception.triage import Triage, TriageBands, sweep_bands, triage_table

from .compare import APPROACHES, ScoredCandidates

__all__ = ["CLASS_COLORS", "APPROACH_STYLE", "APPROACH_NAMES", "plot_triage", "plot_curves", "plot_sweep",
           "plot_reliability", "plot_errors", "triage_panel", "triage_summary"]

CLASS_COLORS = {"not_fire": "#2a78d6", "fire": "#eb6834"}
APPROACH_STYLE = {"algorithmic": ("#1baf7a", "-"), "model": ("#eda100", "--"), "pipeline": ("#e87ba4", ":")}
APPROACH_NAMES = {"algorithmic": "algorithmic (peak °C)", "model": "CNN alone", "pipeline": "full pipeline"}
_INK, _MUTED, _GRID, _REVIEW_FILL = "#2b2b29", "#6b6a64", "#e4e3de", "#efeee9"


def _style(ax) -> None:
    ax.grid(True, color=_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelcolor=_INK)


def _xlabel(approach: str) -> str:
    return "blob peak temperature (°C)" if approach == "algorithmic" else "p(fire) (sigmoid of CNN logit)"


def triage_summary(sc: ScoredCandidates, approach: str, bands: TriageBands) -> str:
    t = triage_table(sc.y, sc.scores(approach), bands)
    c = t["counts"]
    return (f"{APPROACH_NAMES[approach]}: NOT_FIRE < {bands.not_fire_below:.3g} ≤ REVIEW < {bands.fire_at_or_above:.3g}"
            f" ≤ FIRE\n  fires missed (in NOT_FIRE): {c['NOT_FIRE']['fire']}/{t['n_fire']}"
            f"   review: {c['REVIEW']['fire'] + c['REVIEW']['not_fire']}/{t['n']} ({t['review_rate']:.1%})"
            f"   FIRE precision: {t['fire_precision']:.3f}"
            f"   non-fires auto-FIRE: {c['FIRE']['not_fire']}/{t['n_not_fire']}")


def plot_triage(sc: ScoredCandidates, approach: str, bands: TriageBands, axes=None):
    """Left: score histogram per class with the three bands. Right: counts per band and class."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    ax, bx = axes
    s = sc.scores(approach)
    lo_x = float(np.min(s)) if approach == "algorithmic" else 0.0
    hi_x = float(max(np.percentile(s, 99.5), bands.fire_at_or_above * 1.1)) if approach == "algorithmic" else 1.0
    bins = np.linspace(lo_x, hi_x, 41)
    ax.axvspan(bands.not_fire_below, bands.fire_at_or_above, color=_REVIEW_FILL, zorder=0)
    for cls, name in ((0, "not_fire"), (1, "fire")):
        ax.hist(np.clip(s[sc.y == cls], lo_x, hi_x), bins=bins, histtype="step", lw=2, color=CLASS_COLORS[name],
                label=f"{name.replace('_', ' ')} (n={int((sc.y == cls).sum())})", log=len(sc) > 500)
    for x in (bands.not_fire_below, bands.fire_at_or_above):
        ax.axvline(x, color=_MUTED, ls="--", lw=1)
    top = ax.get_ylim()[1]
    for x, txt in (((lo_x + bands.not_fire_below) / 2, "NOT_FIRE"),
                   ((bands.not_fire_below + bands.fire_at_or_above) / 2, "REVIEW"),
                   ((bands.fire_at_or_above + hi_x) / 2, "FIRE")):
        ax.text(x, top, txt, ha="center", va="top", fontsize=8, color=_MUTED)
    ax.set_xlabel(_xlabel(approach))
    ax.set_ylabel("candidates")
    ax.set_title(APPROACH_NAMES[approach], loc="left", fontsize=10, color=_INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    _style(ax)

    t = triage_table(sc.y, s, bands)
    xs = np.arange(3)
    order = [Triage.NOT_FIRE, Triage.REVIEW, Triage.FIRE]
    for k, name in enumerate(("not_fire", "fire")):
        vals = [t["counts"][b.value][name] for b in order]
        bars = bx.bar(xs + (k - 0.5) * 0.36, vals, width=0.32, color=CLASS_COLORS[name],
                      label=name.replace("_", " "))
        bx.bar_label(bars, fontsize=8, color=_INK, padding=2)
    bx.set_xticks(xs, [b.value for b in order])
    bx.set_ylabel("candidates")
    bx.set_title("where each class lands", loc="left", fontsize=10, color=_INK)
    bx.legend(frameon=False, fontsize=8)
    _style(bx)
    return axes


def plot_curves(sc: ScoredCandidates, axes=None):
    """ROC and precision-recall for the three scores on the same candidates."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax, bx = axes
    y = sc.y.astype(bool)
    for a in APPROACHES:
        s = sc.scores(a).astype(np.float64)
        col, ls = APPROACH_STYLE[a]
        order = np.argsort(-s, kind="stable")
        tpr = np.r_[0, np.cumsum(y[order]) / max(y.sum(), 1)]
        fpr = np.r_[0, np.cumsum(~y[order]) / max((~y).sum(), 1)]
        ax.plot(fpr, tpr, color=col, ls=ls, lw=2, label=APPROACH_NAMES[a])
        _, rec, prec = _pr_curve(y, s)
        bx.plot(rec, prec, color=col, ls=ls, lw=2, label=APPROACH_NAMES[a])
    ax.plot([0, 1], [0, 1], color=_MUTED, lw=1, ls=(0, (1, 3)))
    ax.text(0.55, 0.45, "chance", color=_MUTED, fontsize=8, rotation=35)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_title("ROC", loc="left", fontsize=10)
    bx.axhline(y.mean(), color=_MUTED, lw=1, ls=(0, (1, 3)))
    bx.text(0.02, y.mean() + 0.01, f"base rate {y.mean():.2f}", color=_MUTED, fontsize=8)
    bx.set_xlabel("recall")
    bx.set_ylabel("precision")
    bx.set_title("precision-recall", loc="left", fontsize=10)
    for a_ in axes:
        a_.legend(frameon=False, fontsize=8)
        _style(a_)
    return axes


def plot_sweep(sc: ScoredCandidates, approach: str = "model", max_missed: float = 0.02, n: int = 41, axes=None):
    """Heatmaps over (low, high): fires missed and review workload. Dashed contour = missed == max_missed."""
    import matplotlib.pyplot as plt

    s = sc.scores(approach)
    grid = (np.linspace(np.min(s), np.percentile(s, 99.5), n) if approach == "algorithmic"
            else np.linspace(0, 1, n))
    sw = sweep_bands(sc.y, s, grid, grid)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ext = [grid[0], grid[-1], grid[0], grid[-1]]
    for ax, key, title, cmap in ((axes[0], "missed_fire_rate", "fires missed (NOT_FIRE)", "Oranges"),
                                 (axes[1], "review_rate", "review rate (workload)", "Blues")):
        im = ax.imshow(sw[key].T, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=0,
                       vmax=max(np.nanmax(sw[key]), 1e-9))
        ax.contour(grid, grid, sw["missed_fire_rate"].T, levels=[max_missed + 1e-9], colors=_INK, linewidths=1.5, linestyles="--")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel(f"NOT_FIRE below ({'°C' if approach == 'algorithmic' else 'p'})")
        ax.set_ylabel(f"FIRE at or above ({'°C' if approach == 'algorithmic' else 'p'})")
        ax.set_title(f"{title} · {APPROACH_NAMES[approach]}", loc="left", fontsize=10)
    axes[0].text(0.02, 0.98, f"dashed line: missed = {max_missed:.0%}\n(left of it is safer)", transform=axes[0].transAxes,
                 va="top", fontsize=8, color=_INK)
    return axes


def plot_reliability(sc: ScoredCandidates, approach: str = "model", n_bins: int = 10, ax=None):
    """Is p(fire) a probability? Points on the diagonal = calibrated. pos_weight training pushes points below it."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.4))
    s = sc.scores(approach)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(s[m].mean()), ys.append(sc.y[m].mean()), ns.append(int(m.sum()))
    col, _ = APPROACH_STYLE[approach]
    ax.plot([0, 1], [0, 1], color=_MUTED, lw=1, ls=(0, (1, 3)))
    ax.plot(xs, ys, color=col, lw=2, marker="o", ms=7, mec="white", mew=1.5)
    for x, y_, k in zip(xs, ys, ns):
        ax.annotate(f"n={k}", (x, y_), textcoords="offset points", xytext=(4, -10), fontsize=7, color=_MUTED)
    ax.set_xlabel("mean p(fire) in bin")
    ax.set_ylabel("fraction that is fire")
    ax.set_title(f"reliability · {APPROACH_NAMES[approach]}", loc="left", fontsize=10)
    _style(ax)
    return ax


def _crop(img: np.ndarray, u: float, v: float, half: int) -> np.ndarray:
    h, w = img.shape[:2]
    u, v = int(u), int(v)
    return img[max(v - half, 0):min(v + half, h), max(u - half, 0):min(u + half, w)]


def plot_errors(sc: ScoredCandidates, approach: str, bands: TriageBands, kind: str = "missed", n: int = 8,
                half_px: int = 24):
    """kind='missed': fires that land in NOT_FIRE (lowest score first). kind='false_fire': non-fires in FIRE
    (highest first). Thermal crop on top, RGB crop below when the source has paired RGB (hand labels)."""
    import matplotlib.pyplot as plt

    s = sc.scores(approach)
    t = bands.classify_many(s)
    if kind == "missed":
        idx = np.flatnonzero((t == Triage.NOT_FIRE) & (sc.y == 1))
        idx = idx[np.argsort(s[idx])]
    else:
        idx = np.flatnonzero((t == Triage.FIRE) & (sc.y == 0))
        idx = idx[np.argsort(-s[idx])]
    idx = idx[:n]
    if len(idx) == 0:
        print(f"no {kind} cases for {APPROACH_NAMES[approach]} at these bands")
        return None
    has_rgb = sc.source.startswith("handlabels") and sc.rgb_path and all(Path(sc.rgb_path[i]).exists() for i in idx)
    rows = 2 if has_rgb else 1
    fig, axes = plt.subplots(rows, len(idx), figsize=(2.1 * len(idx), 2.3 * rows), squeeze=False)
    for k, i in enumerate(idx):
        temp = load_frame(FrameRecord(frame_path=sc.frame_path[i], group="", label=0))
        ax = axes[0, k]
        ax.imshow(_crop(temp, sc.u[i], sc.v[i], half_px), cmap="inferno")
        score = f"{s[i]:.0f}°C" if approach == "algorithmic" else f"p={s[i]:.2f}"
        ax.set_title(f"{sc.label_text[i]} · {score}\n{sc.peak_c[i]:.0f}°C peak"
                     + (f" · {sc.bypass[i]}" if sc.bypass[i] else ""), fontsize=7)
        if has_rgb:
            from PIL import Image

            rgb = np.asarray(Image.open(sc.rgb_path[i]).convert("RGB"))
            x0, y0, x1, y1 = sc.rgb_box[i]
            sx = (x1 - x0) * rgb.shape[1] / temp.shape[1]
            sy = (y1 - y0) * rgb.shape[0] / temp.shape[0]
            axes[1, k].imshow(_crop(rgb, x0 * rgb.shape[1] + (sc.u[i] + 0.5) * sx,
                                    y0 * rgb.shape[0] + (sc.v[i] + 0.5) * sy, int(half_px * max(sx, sy))))
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    what = "fires sent to NOT_FIRE" if kind == "missed" else "non-fires sent to FIRE"
    fig.suptitle(f"{APPROACH_NAMES[approach]}: {what} ({len(idx)} shown)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return axes


def triage_panel(sc: ScoredCandidates, approach: str, bands: TriageBands) -> None:
    """Two sliders (NOT_FIRE below, FIRE at or above) redrawing plot_triage. Static plot if ipywidgets is missing."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    def draw(low: float, high: float) -> None:
        b = TriageBands(min(low, high), max(low, high))
        plot_triage(sc, approach, b)
        plt.show()
        print(triage_summary(sc, approach, b))

    try:
        import ipywidgets as w
    except ImportError:
        print("ipywidgets not installed (pip install -e .[train]): static plot at the configured bands")
        draw(bands.not_fire_below, bands.fire_at_or_above)
        return
    s = sc.scores(approach)
    if approach == "algorithmic":
        mn, mx, step, fmt = float(np.floor(np.min(s))), float(np.ceil(max(np.percentile(s, 99.5), 300))), 1.0, ".0f"
    else:
        mn, mx, step, fmt = 0.0, 1.0, 0.005, ".3f"
    kw = dict(min=mn, max=mx, step=step, continuous_update=False, readout_format=fmt,
              style={"description_width": "initial"}, layout=w.Layout(width="45%"))
    lo = w.FloatSlider(value=bands.not_fire_below, description="NOT_FIRE below", **kw)
    hi = w.FloatSlider(value=bands.fire_at_or_above, description="FIRE at or above", **kw)
    display(w.HBox([lo, hi]), w.interactive_output(draw, {"low": lo, "high": hi}))


# ---------- models x views ----------

def matrix_markdown(results: dict, metric_cols=("roc_auc", "precision", "recall", "f1", "missed", "review")) -> str:
    """results[model][view] = training.compare.compare(...) output. One row per view x approach, one column group per
    model. precision/recall/f1 = the single search threshold used on the drone today; missed/review = triage bands."""
    models = list(results)
    views = list(dict.fromkeys(v for m in models for v in results[m]))
    head = "| view | approach | " + " | ".join(f"{m}<br>{c}" for m in models for c in metric_cols) + " |"
    lines = [head, "|" + "---|" * (2 + len(models) * len(metric_cols))]

    def val(r, a, c):
        x = r["approaches"][a]
        t = x["triage"]
        v = {"roc_auc": x["roc_auc"], "precision": x["search"]["precision"], "recall": x["search"]["recall"],
             "f1": x["search"]["f1"], "missed": t["missed_fire_rate"], "review": t["review_rate"]}[c]
        return "-" if v != v else f"{v:.3f}"

    for v in views:
        for a in APPROACHES:
            cells = []
            for m in models:
                r = results[m].get(v)
                cells += [val(r, a, c) if r else "" for c in metric_cols]
            n = next((results[m][v] for m in models if v in results[m]), None)
            lab = f"{v} (n={n['n']}, fire={n['n_fire']})" if (n and a == APPROACHES[0]) else ""
            lines.append(f"| {lab} | {APPROACH_NAMES[a]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_matrix(results: dict, metric: str = "f1", axes=None):
    """Small multiples, one panel per model: x = test view, dots = approaches (fixed colours), y = metric at the
    search threshold (f1 / recall / precision) or 'roc_auc'."""
    import matplotlib.pyplot as plt

    models = list(results)
    views = list(dict.fromkeys(v for m in models for v in results[m]))
    if axes is None:
        _, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 3.8), sharey=True, squeeze=False)
        axes = axes[0]
    markers = {"algorithmic": "o", "model": "s", "pipeline": "D"}
    for ax, m in zip(axes, models):
        xs = np.arange(len(views))
        for k, a in enumerate(APPROACHES):
            col, _ = APPROACH_STYLE[a]
            ys = []
            for v in views:
                r = results[m].get(v)
                x = r["approaches"][a] if r else None
                ys.append(np.nan if x is None else (x["roc_auc"] if metric == "roc_auc" else x["search"][metric]))
            ax.plot(xs + (k - 1) * 0.18, ys, ls="none", marker=markers[a], ms=8, color=col, mec="white", mew=1.5,
                    label=APPROACH_NAMES[a])
        ax.set_xticks(xs, views, rotation=25, ha="right", fontsize=8)
        ax.set_title(m, loc="left", fontsize=10)
        ax.set_ylim(min(0.4, ax.get_ylim()[0]), 1.02)
        _style(ax)
    axes[0].set_ylabel(metric.replace("_", " ") + ("" if metric == "roc_auc" else " @ search threshold"))
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    return axes

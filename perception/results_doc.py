"""Write evaluation results into perception/RESULTS.md (human-readable) + perception/results/<key>.json (machine).

Each script owns one section, delimited by markers, and replaces only that section:
    <!-- BEGIN:cnn_cv --> ... <!-- END:cnn_cv -->
The JSON copy is what the demo/frontend (workstream D) should read; the markdown is for the team and pitch.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = ["update_section", "fmt_metric_table"]

_HEADER = "# Perception results\n\nAuto-generated sections; edit only the text outside the markers.\n"


def update_section(doc_path: str | Path, key: str, markdown: str, data: dict | None = None) -> None:
    doc = Path(doc_path)
    text = doc.read_text() if doc.exists() else _HEADER
    block = f"<!-- BEGIN:{key} -->\n{markdown.rstrip()}\n<!-- END:{key} -->"
    pattern = re.compile(rf"<!-- BEGIN:{re.escape(key)} -->.*?<!-- END:{re.escape(key)} -->", re.S)
    text = pattern.sub(lambda _: block, text) if pattern.search(text) else text.rstrip() + "\n\n" + block + "\n"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(text)
    if data is not None:
        jdir = doc.parent / "results"
        jdir.mkdir(exist_ok=True)
        (jdir / f"{key}.json").write_text(json.dumps(data, indent=2, default=float))


def _f(x) -> str:
    if isinstance(x, dict) and "mean" in x:
        return f"{x['mean']:.3f} ± {x['std']:.3f}"
    if isinstance(x, float):
        return "n/a" if x != x else f"{x:.3f}"
    return str(x)


def fmt_metric_table(rows: list[tuple[str, dict]], cols=("precision", "recall", "f1", "f2", "accuracy")) -> str:
    present = [c for c in cols if any(c in r for _, r in rows)]
    lines = ["| | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for name, r in rows:
        lines.append(f"| {name} | " + " | ".join(_f(r.get(c, "")) for c in present) + " |")
    return "\n".join(lines)

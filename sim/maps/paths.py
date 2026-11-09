"""Where map data lives. exclude/ is gitignored, and in a git worktree it may only exist in the main checkout."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

__all__ = ["repo_roots", "default_data_dir", "default_maps_dir"]

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]


def repo_roots() -> list[Path]:
    """This checkout first, then the main checkout (when running from a git worktree)."""
    roots = [_REPO]
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=_REPO, capture_output=True,
                                text=True, timeout=5).stdout.strip()
        if common:
            main = (_REPO / common).resolve().parent
            if main not in roots:
                roots.append(main)
    except (OSError, subprocess.SubprocessError):
        pass
    return roots


def default_data_dir() -> Path | None:
    env = os.environ.get("SIM_FLAME3_NADIR")
    if env:
        return Path(env)
    for root in repo_roots():
        p = root / "exclude" / "data" / "flame3_nadir_prepared"
        if (p / "frames").is_dir():
            return p
    return None


def default_maps_dir(create: bool = False) -> Path:
    env = os.environ.get("SIM_MAPS_DIR")
    if env:
        return Path(env)
    for root in repo_roots():
        p = root / "exclude" / "sim_maps"
        if p.is_dir():
            return p
    p = repo_roots()[-1] / "exclude" / "sim_maps"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p

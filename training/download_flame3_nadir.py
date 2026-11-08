"""Download the THERMAL part of the FLAME 3 NADIR thermal plot subset (Hanna Hammock) from Kaggle.

The Kaggle dataset is one ~35 GB zip, almost all 8000x6000 RGB. We only need the radiometric thermal frames (~1 GB):
  plot {1,2}/duringburn/raw_thermal_tiff/IRX_*.TIFF        int16 counts: celsius = DN * 0.1 - 275 (dataset README)
  plot {1,2}/duringburn/geo_thermal_tiff_celsius/...       stabilized + georeferenced, already celsius
  plot {1,2}/plot*_gcp_coords.txt, GCPs/CSV, metadata/, README   small
  --rgb-per-plot N raw RGB frames per plot (default 3, ~24 MB each) for eyeballing only
Pre/post-burn folders hold RGB, orthos and point clouds only (no thermal), so they are skipped.

How: ONE authenticated API call returns a signed storage URL for the zip. The zip is then read with HTTP Range
requests: the central directory first, then only the wanted members. Per-file API downloads (kagglehub
`dataset_download(path=...)`) hit Kaggle's API rate limit (HTTP 429) after a few dozen files; storage reads don't.

Auth: KAGGLE_API_TOKEN from .env (gitignored). The signed URL is never printed.
    set -a; . ./.env; set +a
    python -m training.download_flame3_nadir --out exclude/data/flame3_nadir [--workers 6] [--rgb-per-plot 3]
Resumable: members already on disk with the right size are skipped. Names contain spaces ("Hanna Hammock/plot 1"):
they are handled as Python strings, never through a shell. Standard library only.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DOWNLOAD_API = "https://www.kaggle.com/api/v1/datasets/download/brycehopkins/flame-3-nadir-thermal-plot-subset"
_WANT = re.compile(
    r"(^|/)Hanna Hammock/("
    r"plot [12]/duringburn/(raw_thermal_tiff|geo_thermal_tiff_celsius)/.*\.tiff?"
    r"|plot [12]/[^/]+\.txt|GCPs/CSV/[^/]+|metadata/[^/]+|[^/]+\.rtf)$", re.IGNORECASE)
_RGB = re.compile(r"(^|/)Hanna Hammock/plot ([12])/duringburn/raw_rgb_jpg/[^/]+\.jpe?g$", re.IGNORECASE)


def signed_url(token: str) -> str:
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    req = urllib.request.Request(DOWNLOAD_API, headers={"Authorization": f"Bearer {token}"})
    try:
        urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307) and e.headers.get("Location"):
            return e.headers["Location"]
        raise SystemExit(f"Kaggle download API returned HTTP {e.code} (429 = rate limited: wait a few minutes)")
    raise SystemExit("Kaggle download API did not redirect to a storage URL")


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP Range requests, with a small block cache (enough for zipfile)."""

    def __init__(self, url: str, block: int = 1 << 20, retries: int = 5):
        self.url, self.block, self.retries, self.pos = url, block, retries, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.size = int(head.headers["Content-Length"])
        self._cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, start: int, end: int) -> bytes:   # inclusive end
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return b""

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n > 4 * self.block:   # big member payloads: one direct request, no caching
            data = self._get(self.pos, self.pos + n - 1)
        else:
            out = bytearray()
            p = self.pos
            while len(out) < n:
                b = p // self.block
                if b not in self._cache:
                    if len(self._cache) > 64:
                        self._cache.clear()
                    s = b * self.block
                    self._cache[b] = self._get(s, min(s + self.block, self.size) - 1)
                chunk = self._cache[b][p - b * self.block:]
                take = chunk[:n - len(out)]
                out += take
                p += len(take)
            data = bytes(out)
        self.pos += len(data)
        return data

    def readinto(self, buf) -> int:
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)


def select(infos: list[zipfile.ZipInfo], rgb_per_plot: int) -> list[zipfile.ZipInfo]:
    keep = [i for i in infos if not i.is_dir() and _WANT.search(i.filename)]
    for plot in ("1", "2"):
        rgb = sorted((i for i in infos if (m := _RGB.search(i.filename)) and m.group(2) == plot),
                     key=lambda i: i.filename)
        if rgb and rgb_per_plot > 0:   # spread over the burn: start ... end
            idx = sorted({round(k * (len(rgb) - 1) / max(rgb_per_plot - 1, 1)) for k in range(rgb_per_plot)})
            keep += [rgb[k] for k in idx]
    return keep


def _dest(out: Path, name: str) -> Path:
    i = name.find("Hanna Hammock/")
    rel = Path(name[i:] if i >= 0 else name)
    if rel.is_absolute() or ".." in rel.parts:   # never write outside --out
        raise ValueError(f"unsafe member name: {name!r}")
    return out / rel


def _extract(url: str, names: list[str], out: Path) -> dict[str, int]:
    """One worker: its own HTTP file + ZipFile (zipfile is not thread-safe on a shared handle)."""
    stats = {"ok": 0, "skip": 0, "failed": 0}
    with zipfile.ZipFile(HttpRangeFile(url)) as zf:
        for name in names:
            info = zf.getinfo(name)
            dst = _dest(out, name)
            if dst.exists() and dst.stat().st_size == info.file_size:
                stats["skip"] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            try:
                with zf.open(info) as src, tmp.open("wb") as fh:
                    shutil.copyfileobj(src, fh, 1 << 20)
                tmp.replace(dst)
                stats["ok"] += 1
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                print(f"  FAILED {name}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="exclude/data/flame3_nadir")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rgb-per-plot", type=int, default=3)
    a = ap.parse_args()
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise SystemExit("KAGGLE_API_TOKEN not set: set -a; . ./.env; set +a")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    url = signed_url(token)
    with zipfile.ZipFile(HttpRangeFile(url)) as zf:
        todo = select(zf.infolist(), a.rgb_per_plot)
    print(f"{len(todo)} members, {sum(i.file_size for i in todo) / 1e9:.2f} GB -> {out}", flush=True)
    names = [i.filename for i in todo]
    chunks = [names[k::a.workers] for k in range(a.workers)]
    total = {"ok": 0, "skip": 0, "failed": 0}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for st in ex.map(lambda c: _extract(url, c, out), chunks):
            for k, v in st.items():
                total[k] += v
    print("done", total, flush=True)
    if total["failed"]:
        print("re-run the same command to retry the failures")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Refresh the contracts block inside context.md from the files in contracts/.

context.md is the source of truth for everything EXCEPT the code between the markers
    <!-- CONTRACTS:BEGIN --> ... <!-- CONTRACTS:END -->
which this script regenerates. Run from the repo root after any change to contracts/:

    python tools/sync_contracts.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = [
    "contracts/__init__.py", "contracts/base.py", "contracts/enums.py", "contracts/geometry.py",
    "contracts/sensing.py", "contracts/entities.py", "contracts/messages.py", "contracts/topics.py",
    "contracts/interfaces.py", "contracts/config.py",
]
BEGIN, END = "<!-- CONTRACTS:BEGIN -->", "<!-- CONTRACTS:END -->"


def main() -> int:
    doc_path = ROOT / "context.md"
    doc = doc_path.read_text(encoding="utf-8")
    if BEGIN not in doc or END not in doc:
        print("markers not found in context.md", file=sys.stderr)
        return 1
    parts = []
    for rel in FILES:
        code = (ROOT / rel).read_text(encoding="utf-8").rstrip()
        n_lines = code.count("\n") + 1
        parts.append(f"#### `{rel}` ({n_lines} lines)\n\n```python\n{code}\n```\n")
    block = f"{BEGIN}\n\n" + "\n".join(parts) + f"\n{END}"
    new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _m: block, doc, flags=re.S)
    doc_path.write_text(new, encoding="utf-8")
    print(f"synced {len(FILES)} files into context.md ({len(block.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

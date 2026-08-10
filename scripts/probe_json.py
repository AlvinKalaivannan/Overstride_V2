"""Sketch the structure of a single session JSON. Phase 0 step C5 reconnaissance.

Usage: python scripts/probe_json.py <path-to-json> [max_depth]
"""

import json
import sys
from pathlib import Path


def describe(node, name: str, depth: int, max_depth: int, out: list) -> None:
    pad = "  " * depth
    if isinstance(node, dict):
        out.append(f"{pad}{name}: dict({len(node)}) keys={list(node)[:14]}")
        if depth < max_depth:
            for k, v in node.items():
                describe(v, str(k), depth + 1, max_depth, out)
    elif isinstance(node, list):
        inner = node
        dims = []
        while isinstance(inner, list):
            dims.append(len(inner))
            if not inner:
                break
            inner = inner[0]
        out.append(f"{pad}{name}: list shape~{dims} leaf={type(inner).__name__}")
        # Recurse into the first element only when it is a container of containers,
        # so we sketch structure without walking millions of floats.
        if depth < max_depth and node and isinstance(node[0], dict):
            describe(node[0], f"{name}[0]", depth + 1, max_depth, out)
    else:
        val = repr(node)
        out.append(f"{pad}{name}: {type(node).__name__} = {val[:70]}")


def main() -> int:
    path = Path(sys.argv[1])
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    print(f"file: {path.name}  {path.stat().st_size / 1e6:.2f} MB")
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    out: list[str] = []
    describe(data, "<root>", 0, max_depth, out)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

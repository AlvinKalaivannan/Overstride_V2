"""Phase 15 -- fetch the MoVi slice from Scholars Portal Dataverse.

Licence and accession are recorded in `docs/data-inventory.md` 8.5, committed
BEFORE this script was first run (commit 3e0ab69), per the standing rule. The
pipeline was frozen earlier still (64fb0bc), so `git log` against the file
mtimes shows the ordering freeze -> licence -> download.

Only 1.20 GB of the record's 290.2 GB is taken: the round-S Visual3D mocap, the
three calibrated Point Grey clips that can carry an MAE, and the two hand-held
phone clips that answer section 7.5 of `results/phase11.md`.

Nothing here is published. The licence forbids distribution of the Data in any
form; what leaves this machine is angle statistics, which are not the Data.

Run: .venv/Scripts/python.exe scripts/phase15_fetch_movi.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "movi" / "raw"
MANIFEST = REPO / "results" / "phase15_movi_manifest.json"

API = "https://borealisdata.ca/api/access/datafile/{id}"
DOI = "doi:10.5683/SP2/JRHDRN"

# (datafile id, served filename, expected MB from the Dataverse file listing)
SLICE = [
    (92073, "S_V3D.tar", 590.8),
    (85637, "S2_PG1_Subject_45_L.avi", 94.5),
    (85559, "S2_PG1_Subject_74_L.avi", 93.7),
    (85654, "S2_PG1_Subject_80_L.avi", 97.6),
    (86001, "S2_CP1_Subject_22.mp4", 167.9),
    (86078, "S2_CP2_Subject_22.mp4", 153.0),
]

CHUNK = 1 << 20


def fetch(fid: int, name: str, dest: Path) -> dict:
    """Stream one datafile to disk, hashing as it goes.

    Written to a `.part` file and renamed only on completion, so an interrupted
    download can never be mistaken for a complete one by a later stage.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    n = 0
    t0 = time.time()
    req = urllib.request.Request(API.format(id=fid),
                                 headers={"User-Agent": "overstride-phase15"})
    with urllib.request.urlopen(req, timeout=300) as r, open(part, "wb") as f:
        declared = r.headers.get("Content-Length")
        while True:
            b = r.read(CHUNK)
            if not b:
                break
            f.write(b)
            h.update(b)
            n += len(b)
    part.replace(dest)
    dt = time.time() - t0
    print(f"    {name}: {n / 1e6:.1f} MB in {dt:.0f}s "
          f"({n / 1e6 / max(dt, 1e-9):.1f} MB/s)")
    return {"datafile_id": fid, "filename": name, "bytes": n,
            "content_length_header": declared,
            "sha256": h.hexdigest(), "seconds": round(dt, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    total = sum(mb for _, _, mb in SLICE)
    print(f"=== MoVi slice: {len(SLICE)} files, ~{total:.0f} MB ===")
    print(f"  {DOI} -> {RAW}")
    if args.dry_run:
        for fid, name, mb in SLICE:
            print(f"    would fetch {fid:>6}  {name:<28} {mb:>7.1f} MB")
        return 0

    rows = []
    for fid, name, mb in SLICE:
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 0.9 * mb * 1e6:
            print(f"    {name}: already present, skipping")
            rows.append({"datafile_id": fid, "filename": name,
                         "bytes": dest.stat().st_size, "skipped": True})
            continue
        rows.append(fetch(fid, name, dest))

    MANIFEST.write_text(json.dumps(
        {"doi": DOI, "api": API,
         "note": "sizes and hashes only -- no MoVi data is committed",
         "files": rows}, indent=2), encoding="utf-8")
    got = sum(r["bytes"] for r in rows)
    print(f"\n  {got / 1e6:.1f} MB on disk")
    print(f"wrote {MANIFEST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

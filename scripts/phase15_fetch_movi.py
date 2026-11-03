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
import tarfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "movi" / "raw"
MANIFEST = REPO / "results" / "phase15_movi_manifest.json"

API = "https://borealisdata.ca/api/access/datafile/{id}"
DOI = "doi:10.5683/SP2/JRHDRN"

# (datafile id, served filename, expected MB from the Dataverse file listing)
#
# ROUND S IS THE WRONG ROUND, and that was established by the pre-registered
# checks rather than guessed: `S_V3D.tar` ships 12 physical markers -- head,
# wrists, ankles -- with no hip, knee or pelvis, so check 3 (marker set ->
# H36M-17) cannot pass for the two joints this phase scores. It also ships no
# `motions_list` and no frame flags, so it cannot even be cut into movements.
# Kept here as the record of how that was found. See commit fe247a8 and
# `docs/data-inventory.md` 8.5.
SLICE_S = [
    (92073, "S_V3D.tar", 590.8),
    (85637, "S2_PG1_Subject_45_L.avi", 94.5),
    (85559, "S2_PG1_Subject_74_L.avi", 93.7),
    (85654, "S2_PG1_Subject_80_L.avi", 97.6),
    (86001, "S2_CP1_Subject_22.mp4", 167.9),
    (86078, "S2_CP2_Subject_22.mp4", 153.0),
]

# ROUND F is what phase 15 runs on: 87 markers including the ASIS and PSIS pairs
# that define H36M's own hip, a `motions_list` naming jogging, and `flags30` /
# `flags120` giving the same movement boundaries in video and in mocap frames.
#
# PG1 is the SIDE-ON camera (view_ratio 0.720 against AthleticsPose's own median
# of 0.751, travel azimuth 88.6 deg) and PG2 is near-frontal (0.468, 5.3 deg),
# so one subject gets both and the phase 11 viewpoint curve becomes measurable
# on a lab the pipeline has never seen. CP1 is the hand-held iPhone 7 and
# answers section 7.5; it has no calibration, hence the conditional already
# frozen into `results/phase15.md`.
#
# Subjects are 1, 10 and 11 -- the first three in the mocap archive's own
# lexicographic order, chosen for streaming cost and for nothing else. No
# measurement informed the choice, because none existed when it was made.
SLICE_F = [
    (85268, "F_PG1_Subject_1_L.avi", 145.3),
    (85238, "F_PG1_Subject_10_L.avi", 250.6),
    (85274, "F_PG1_Subject_11_L.avi", 300.9),
    (85331, "F_PG2_Subject_1_L.avi", 132.4),
    (86002, "F_CP1_Subject_1.mp4", 309.6),
]

# The DIAGNOSTIC set, designated in commit f549bfd BEFORE it was downloaded:
# Subject_12, Subject_13, Subject_15, PG1 only. Spent on arrival -- used to
# diagnose the hip bias and the detector split, never to show that a fix works.
# Subject_14 is skipped: it has no round F video of any kind.
#
# The SEALED TEST set (Subject_16, _17, _18, _19, _2) is deliberately NOT here.
# It is named and costed in docs/data-inventory.md 8.5 and must stay untouched
# until a fix is frozen. Do not add it to this list to "save a trip".
SLICE_F_DIAG = [
    (85270, "F_PG1_Subject_12_L.avi", 247.0),
    (85234, "F_PG1_Subject_13_L.avi", 260.0),
    (85255, "F_PG1_Subject_15_L.avi", 234.0),
]
DIAG_SUBJECTS = ("Subject_12", "Subject_13", "Subject_15")

# The round F mocap is one 1.81 GB tar, but it is SEQUENTIAL: members are laid
# out in archive order, so a ranged GET of a prefix yields the first few without
# transferring the rest. Subjects 1, 10 and 11 are members 1-3.
MOCAP_TAR = (128299, "F_Subjects_1_45.tar", 1814.6)
MOCAP_PREFIX_MB = 150.0
MOCAP_WANT = 3

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


def fetch_mocap_prefix(dest_dir: Path, n_want: int = MOCAP_WANT) -> list[dict]:
    """Pull the first `n_want` subjects out of the round F mocap tar.

    A ranged GET of the archive's first ~150 MB, read with tarfile in stream
    mode, then the connection is done with. The remaining 1.7 GB is never
    transferred -- the subjects wanted happen to be the archive's first members.
    """
    fid, name, _mb = MOCAP_TAR
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = dest_dir / f"prefix_{fid}.tarpart"
    # Enough bytes to reach `n_want` members. The measured cumulative size to
    # the 7th member is ~290 MB; 40 MB of headroom covers the variation, and
    # anything not needed is simply never read.
    need_mb = max(MOCAP_PREFIX_MB, 40.0 * n_want + 60.0)
    n = int(need_mb * 1e6)
    req = urllib.request.Request(
        API.format(id=fid), headers={"User-Agent": "overstride-phase15",
                                     "Range": f"bytes=0-{n - 1}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r, open(prefix, "wb") as f:
        got = 0
        while (b := r.read(CHUNK)):
            f.write(b)
            got += len(b)
    print(f"    {name}: {got / 1e6:.1f} MB prefix in {time.time() - t0:.0f}s "
          f"(archive is {_mb:.0f} MB; the rest is not transferred)")

    # Each member goes to a `.part` and is renamed only once it is complete.
    # The prefix cuts off mid-member by construction, so without this the last
    # member lands as a TRUNCATED .mat with a plausible name -- which is worse
    # than no file at all, because a later stage would load it.
    rows = []
    try:
        with tarfile.open(prefix, mode="r|") as t:
            for m in t:
                if not m.isfile():
                    continue
                out = dest_dir / Path(m.name).name
                part = out.with_suffix(out.suffix + ".part")
                h = hashlib.sha256()
                try:
                    with t.extractfile(m) as src, open(part, "wb") as dst:
                        while (b := src.read(CHUNK)):
                            dst.write(b)
                            h.update(b)
                except tarfile.ReadError:
                    part.unlink(missing_ok=True)
                    print(f"      {out.name}: INCOMPLETE in this prefix "
                          f"(needs {m.size / 1e6:.1f} MB) -- discarded; raise "
                          f"MOCAP_PREFIX_MB to reach it")
                    break
                if part.stat().st_size != m.size:
                    part.unlink(missing_ok=True)
                    print(f"      {out.name}: short read, discarded")
                    break
                part.replace(out)
                rows.append({"datafile_id": fid, "from_archive": name,
                             "filename": out.name, "bytes": out.stat().st_size,
                             "sha256": h.hexdigest()})
                print(f"      {out.name}: {out.stat().st_size / 1e6:.1f} MB")
                if len(rows) >= n_want:
                    break
    finally:
        prefix.unlink(missing_ok=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", choices=["F", "S"], default="F",
                    help="F is what phase 15 runs on; S is the wrong round")
    ap.add_argument("--diagnostic", action="store_true",
                    help="fetch the DIAGNOSTIC set (Subject_12/13/15, PG1 "
                         "only) designated in commit f549bfd")
    ap.add_argument("--no-mocap", action="store_true",
                    help="skip the mocap tar prefix (video only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.diagnostic:
        if args.round != "F":
            raise SystemExit("--diagnostic is round F only")
        slice_ = SLICE_F_DIAG
        # Subject_15 is the 7th member in the archive's lexicographic order,
        # so the prefix has to run that far. 1, 10, 11 are already on disk and
        # are re-extracted harmlessly; 14 has no video but still occupies a slot.
        want_members = 7
    else:
        slice_ = SLICE_F if args.round == "F" else SLICE_S
        want_members = MOCAP_WANT
    raw = RAW / f"round_{args.round.lower()}"
    raw.mkdir(parents=True, exist_ok=True)
    total = sum(mb for _, _, mb in slice_)
    print(f"=== MoVi round {args.round}: {len(slice_)} video files, "
          f"~{total:.0f} MB ===")
    print(f"  {DOI} -> {raw}")
    if args.dry_run:
        for fid, name, mb in slice_:
            print(f"    would fetch {fid:>6}  {name:<28} {mb:>7.1f} MB")
        if args.round == "F" and not args.no_mocap:
            need = max(MOCAP_PREFIX_MB, 40.0 * want_members + 60.0)
            print(f"    would range-fetch {need:.0f} MB of {MOCAP_TAR[1]} "
                  f"for the first {want_members} archive members")
        return 0

    rows = []
    if args.round == "F" and not args.no_mocap:
        rows += fetch_mocap_prefix(raw / "mocap", want_members)
    for fid, name, mb in slice_:
        dest = raw / name
        if dest.exists() and dest.stat().st_size > 0.9 * mb * 1e6:
            print(f"    {name}: already present, skipping")
            rows.append({"datafile_id": fid, "filename": name,
                         "bytes": dest.stat().st_size, "skipped": True})
            continue
        rows.append(fetch(fid, name, dest))

    man = MANIFEST.with_name(f"phase15_movi_manifest_{args.round.lower()}.json")
    man.write_text(json.dumps(
        {"doi": DOI, "api": API, "round": args.round,
         "note": "sizes and hashes only -- no MoVi data is committed",
         "files": rows}, indent=2), encoding="utf-8")
    got = sum(r["bytes"] for r in rows)
    print(f"\n  {got / 1e6:.1f} MB on disk")
    print(f"wrote {man.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

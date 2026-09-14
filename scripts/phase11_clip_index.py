"""Phase 11 (B1) step 1 -- map AthletePose3D's windowed clips back to their actions.

THE PROBLEM
-----------
`pose_3d.zip` ships two things that do not reference each other:

  frame_81/{train,test}/NNNNNNNN.pkl   clips, each {data_input, data_label} only
  train.pkl / valid.pkl               per-frame records WITH action labels

B1 is about RUNNING, and the clips carry no action label, so the running subset
cannot be selected without a mapping between the two.

WHY NOT RECONSTRUCT THEIR SEGMENTATION
--------------------------------------
The obvious route -- group frames by video and cut non-overlapping 81-frame
windows -- gives 3,143 clips against the 3,067 actually shipped for test, and the
videos are not even contiguous in `valid.pkl` (1,010 runs over 826 distinct
videos). Guessing the remaining rule would put an unverified assumption
underneath every downstream number.

WHAT THIS DOES INSTEAD
----------------------
Matches each clip to its source frame by CONTENT, needing no assumption about how
the clips were cut. The normalisation was recovered exactly (max |diff| =
0.000000 on the first clip):

    x_norm = x / video_width * 2 - 1
    y_norm = y / video_width * 2 - video_height / video_width

NEAREST NEIGHBOUR, NOT AN EXACT KEY. A first attempt hashed the rounded
coordinates and matched only 1,355 of 3,067 -- including a clip already verified
identical by hand. The cause was the hashing, not the data: rounding to a fixed
number of decimals splits values sitting on a rounding boundary, so two
numerically identical coordinates can produce different keys. A KD-tree over the
34-dimensional normalised first frame has no such failure mode, and an accepted
match must still agree to within TOL.

BOTH SPLITS ARE INDEXED, AND THAT IS DELIBERATE
-----------------------------------------------
AthletePose3D's train/valid split exists to partition ITS OWN models' training.
It says nothing about this project, whose lifting weights were trained on
AthleticsPose -- verified in phase 10's B0 check -- and which has never loaded an
AP3D-trained model; `scripts/phase11_fetch_ap3d.py` deletes `model_params/`
outright for that reason.

So from here every AthletePose3D frame is held-out data, and restricting to their
"valid" split would buy no independence while costing most of the running cohort:
valid carries running subject S2 alone, train carries S3 and S4. Indexing both is
what makes a per-subject figure possible, and the plan requires per-subject rather
than pooled MAE precisely because the running cohort is this small.

Run: .venv/Scripts/python.exe scripts/phase11_clip_index.py
"""

from __future__ import annotations

import collections
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ZIP = REPO / "data" / "athletepose3d" / "pose_3d.zip"
OUT = REPO / "data" / "derived" / "ap3d_clip_index.json"
ROOT = "pose_3d_v3"
TOL = 1e-5         # max per-coordinate disagreement accepted as the same frame

SPLITS = (("test", "valid"), ("train", "train"))


def norm_xy(joints: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalised 2D of one frame, flattened to a 34-vector."""
    xy = np.empty((joints.shape[0], 2), dtype=np.float64)
    xy[:, 0] = joints[:, 0] / w * 2.0 - 1.0
    xy[:, 1] = joints[:, 1] / w * 2.0 - h / w
    return xy.ravel()


def match_split(z, tree, v, names, split, index, unmatched) -> float:
    """Match one split's clips to their source frames. Returns worst accepted error."""
    worst = 0.0
    for n in names:
        with z.open(n) as f:
            c = pickle.load(f)
        q = np.asarray(c["data_input"][0, :, :2], dtype=np.float64).ravel()
        dist, j = tree.query(q, k=1)
        # cKDTree gives Euclidean distance over 34 dims; convert to a
        # per-coordinate bound so TOL means what it says.
        per_coord = float(dist) / np.sqrt(q.size)
        if per_coord > TOL:
            unmatched.append(f"{split}/{Path(n).stem} "
                             f"(nearest {per_coord:.2e} > {TOL:.0e})")
            continue
        worst = max(worst, per_coord)
        e = v[int(j)]
        index[f"{split}/{Path(n).stem}"] = {
            "split": split, "src_idx": int(j), "action": e["action"],
            "subject": e["subject"], "subaction": e["subaction"],
            "cameraid": e["cameraid"], "fps": float(e["fps"]),
            "video_width": int(e["video_width"]),
            "video_height": int(e["video_height"]),
            "zip_entry": n,
        }
    return worst


def main() -> int:
    if not ZIP.exists():
        print(f"ERROR: {ZIP} not found")
        return 2
    from scipy.spatial import cKDTree

    z = zipfile.ZipFile(ZIP)
    index: dict[str, dict] = {}
    unmatched: list[str] = []
    worst = 0.0
    n_clips = 0

    for split, pkl in SPLITS:
        print(f"\nindexing {pkl}.pkl ...")
        with z.open(f"{ROOT}/{pkl}.pkl") as f:
            v = pickle.load(f)
        mat = np.empty((len(v), 34), dtype=np.float64)
        for i, e in enumerate(v):
            mat[i] = norm_xy(np.asarray(e["joint_3d_image"]),
                             e["video_width"], e["video_height"])
        tree = cKDTree(mat)
        print(f"  {len(v)} frame records | KD-tree {mat.shape}")

        names = sorted(n for n in z.namelist()
                       if n.startswith(f"{ROOT}/frame_81/{split}/")
                       and n.endswith(".pkl"))
        n_clips += len(names)
        print(f"  matching {len(names)} {split} clips ...")
        worst = max(worst, match_split(z, tree, v, names, split, index, unmatched))

    print(f"\nmatched {len(index)} of {n_clips} | unmatched {len(unmatched)}")
    print(f"worst ACCEPTED per-coordinate error: {worst:.2e} (tolerance {TOL:.0e})")
    for n in unmatched[:5]:
        print(f"    {n}")

    print("\nclips per action:")
    by_action = collections.Counter(r["action"] for r in index.values())
    for a, n in sorted(by_action.items()):
        rows = [r for r in index.values() if r["action"] == a]
        print(f"  {a:<5} {n:>5} clips | fps {sorted({r['fps'] for r in rows})} | "
              f"subjects {sorted({r['subject'] for r in rows})} | "
              f"{len({r['cameraid'] for r in rows})} cameras")

    rm = [r for r in index.values() if r["action"] == "rm"]
    print(f"\nRUNNING cohort for B1: {len(rm)} clips / "
          f"{len({r['subject'] for r in rm})} subjects / "
          f"{len({r['cameraid'] for r in rm})} cameras")
    per_subj = collections.Counter(r["subject"] for r in rm)
    for s, n in sorted(per_subj.items()):
        print(f"  {s}: {n} clips")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source_zip": ZIP.name, "root": ROOT,
        "n_clips": n_clips, "n_matched": len(index),
        "n_unmatched": len(unmatched), "unmatched": unmatched[:50],
        "worst_accepted_per_coord_error": worst, "tolerance": TOL,
        "normalisation": "x/w*2-1, y/w*2-h/w",
        "both_splits_indexed_because":
            "our lifter was trained on AthleticsPose (phase 10 B0), never on "
            "AP3D, so both AP3D splits are held-out data here",
        "by_action": dict(by_action),
        "running_subjects": sorted({r["subject"] for r in rm}),
        "clips": index}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

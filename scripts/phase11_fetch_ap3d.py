"""Phase 11 (B1) step 0 -- fetch AthletePose3D.

AthletePose3D (Yeung et al., CVSports at CVPR 2025) is distributed only as a
public Google Drive folder reached through a read-the-licence page, with no
direct or API URL, so `gdown` is required rather than curl.

LICENCE. The licence page states: "By downloading the files, you agree to the
license terms", and the terms are NON-COMMERCIAL SCIENTIFIC RESEARCH ONLY with a
citation requirement. That is already the footing this project operates on --
`CLAUDE.md` lists AthletePose3D as research-only under "Out of scope: any
commercial framing" -- so downloading adds no obligation the project was not
already bound by.

THE ERRATUM THIS MUST RESPECT
-----------------------------
The repository carries this notice:

  "11/07/2025 - Erratum: A preprocessing mistake occurred in one camera angle of
   the running motions (3D). The pose_3d.zip files, pre-trained model, and
   corresponding code have been corrected - please re-download them."

B1 is entirely about RUNNING and rests on the 3D ground truth, so a stale
pose_3d.zip would silently corrupt the one measurement B1 exists to make. Google
Drive preserves each file's modified time and gdown writes it through, so the
download is checked against the erratum date rather than assumed current.

WHAT IS DELIBERATELY NOT FETCHED
--------------------------------
`model_params/` holds AthletePose3D's own trained checkpoints, including
`motionagformer-s-ap3d.pth.tr`. Those are NOT downloaded and must not be used to
produce any number here: phase 10's B0 check established that this project's
lifting checkpoints were trained on AthleticsPose, which is what makes
AthletePose3D an INDEPENDENT test set. Evaluating an AP3D-trained model on AP3D
data would be train-on-test and would destroy exactly the property B1 needs.

Run: .venv/Scripts/python.exe scripts/phase11_fetch_ap3d.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "data" / "athletepose3d"

# Erratum date from the repository notice. Anything older than this for
# pose_3d.zip is the uncorrected release and must not be used.
ERRATUM = dt.date(2025, 7, 11)      # 11/07/2025, read as DD/MM/YYYY
ERRATUM_ALT = dt.date(2025, 11, 7)  # ...or MM/DD/YYYY. Both are checked.

WANTED = {
    "cam_param.json": "1C6ljWV5kJyTciIKNb0ZRre3f0cNh1Ll8",
    "pose_3d.zip": "1PQerwftEKoOwqG-dhvadz_gaGxZx0D2y",
    "pose_2d.zip": "13ISVY8G_NxrwLWFdxyTlOUhSyxVJsd93",
    "data.zip": "1xnQDxvTjS9D9eYJMWsizbsfHSvxTnCxp",
}

FOLDER = "10YnMJAluiscnLkrdiluIeehNetdry5Ft"

# Downloaded by the folder fallback but NOT usable here. model_params/ holds
# AthletePose3D's own trained weights -- their 2D detector (moganet_b_ap2d) and
# their lifters (motionagformer-s-ap3d, TCPFormer_ap3d). Phase 10's B0 check
# established that this project's weights were trained on AthleticsPose, which is
# precisely what makes AthletePose3D an independent test set. Using any AP3D-
# trained model on AP3D data would be train-on-test and would destroy that
# property, so these are deleted rather than left lying next to the data.
PURGE = ("model_params",)


def main() -> int:
    import gdown

    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, fid in WANTED.items():
        out = DEST / name
        if out.exists() and out.stat().st_size > 0:
            print(f"  have {name} ({out.stat().st_size / 1e6:.1f} MB), skipping")
        else:
            print(f"\n--- {name}")
            # Google Drive enforces a per-file share quota ("Too many users have
            # viewed or downloaded this file recently"), which is transient and
            # unrelated to this machine. One file hitting it must not abort the
            # rest -- record and carry on.
            try:
                gdown.download(id=fid, output=str(out), quiet=False, resume=True)
            except Exception as e:  # noqa: BLE001 - recorded, never silent
                quota = "Too many users" in str(e)
                print(f"  FAILED ({type(e).__name__}): "
                      f"{'Drive share quota -- retry in a few hours' if quota else e}")
                manifest[name] = {"drive_id": fid, "failed": True,
                                  "quota_limited": quota,
                                  "error": type(e).__name__}
                continue
        if not out.exists() or out.stat().st_size == 0:
            print(f"  FAILED: {name} produced no file")
            manifest[name] = {"drive_id": fid, "failed": True}
            continue
        st = out.stat()
        mtime = dt.datetime.fromtimestamp(st.st_mtime)
        manifest[name] = {"drive_id": fid, "bytes": st.st_size,
                          "drive_mtime": mtime.isoformat()}
        print(f"  {name}: {st.st_size / 1e6:.1f} MB, Drive mtime {mtime.date()}")

    # --- folder fallback ---------------------------------------------------
    # Per-file download is refused for pose_2d.zip and data.zip ("Cannot retrieve
    # the public link... or have had many accesses"), while pose_3d.zip of 1.46 GB
    # succeeds -- so it is per-file share state, not a size or gdown limit. The
    # folder endpoint is a different code path and does serve them.
    missing = [n for n in WANTED if not (DEST / n).exists()
               or (DEST / n).stat().st_size == 0]
    if missing:
        print(f"\n=== per-file download failed for {missing}; "
              f"falling back to the folder endpoint ===")
        try:
            gdown.download_folder(url=f"https://drive.google.com/drive/folders/{FOLDER}",
                                  output=str(DEST), quiet=False, resume=True)
        except Exception as e:  # noqa: BLE001
            print(f"  folder fallback also failed: {type(e).__name__}: {e}")
        for name in WANTED:
            out = DEST / name
            if out.exists() and out.stat().st_size:
                st = out.stat()
                manifest[name] = {
                    "drive_id": WANTED[name], "bytes": st.st_size,
                    "drive_mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(),
                    "via": "folder endpoint"}

    for rel in PURGE:
        d = DEST / rel
        if d.exists():
            import shutil
            n = sum(1 for _ in d.rglob("*") if _.is_file())
            shutil.rmtree(d)
            print(f"\npurged {rel}/ ({n} files) -- AP3D-trained weights must "
                  f"not be used on AP3D data (train-on-test)")

    # --- the erratum check -------------------------------------------------
    p3 = manifest.get("pose_3d.zip")
    if p3:
        d = dt.date.fromisoformat(p3["drive_mtime"][:10])
        ok = d >= min(ERRATUM, ERRATUM_ALT)
        p3["postdates_erratum"] = bool(ok)
        print(f"\n=== erratum check ===")
        print(f"  pose_3d.zip Drive mtime : {d}")
        print(f"  erratum announced       : {ERRATUM} or {ERRATUM_ALT}")
        verdict = ("post-dates the erratum -- corrected release" if ok
                   else "PREDATES the erratum -- DO NOT USE, re-download")
        print(f"  VERDICT: {verdict}")
        if not ok:
            print("  The running 3D ground truth may be the uncorrected version.")
            print("  B1 must not proceed on this file.")

    (REPO / "results" / "phase11_fetch.json").write_text(
        json.dumps({"source": "AthletePose3D, Yeung et al., CVSports at CVPR 2025",
                    "drive_folder": "10YnMJAluiscnLkrdiluIeehNetdry5Ft",
                    "licence": "non-commercial scientific research only",
                    "retrieved": dt.date.today().isoformat(),
                    "model_params_deliberately_not_fetched": True,
                    "files": manifest}, indent=2), encoding="utf-8")
    print(f"\nwrote results/phase11_fetch.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

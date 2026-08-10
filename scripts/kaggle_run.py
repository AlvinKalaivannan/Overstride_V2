"""Push the phase 3 notebook to Kaggle, run it on GPU, and fetch the output.

Kaggle runs under YOUR account, so this needs an API token that only you can
create. It is a one-time step:

  1. kaggle.com -> avatar -> Settings -> API -> "Create New Token".
     NOTE: this no longer downloads a file. It shows an opaque token, KGAT_...
  2. Store it either as the KAGGLE_API_TOKEN environment variable, or in
     C:\\Users\\<you>\\.kaggle\\access_token  (no extension).
     The legacy kaggle.json ({"username","key"}) still works if you have one.
  3. uv add kaggle          (needs a recent release; older ones predate KGAT_)

The token is a credential -- do not paste it into a chat window or commit it.
.gitignore covers .kaggle/, kaggle.json and access_token.

Then:
  .venv/Scripts/python.exe scripts/kaggle_run.py push     # dataset + kernel
  .venv/Scripts/python.exe scripts/kaggle_run.py status   # poll
  .venv/Scripts/python.exe scripts/kaggle_run.py fetch    # download output

The notebook itself needs GPU and Internet; both are set in the kernel metadata
below rather than left to the UI, so a forgotten toggle cannot silently produce a
CPU run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "data" / "kaggle_stage"          # gitignored (under data/)
SLUG_DATA = "overstride-scripts"
# Kaggle derives the live slug from the TITLE, not from the id you supply -- an
# id of "overstride-phase3" with title "Overstride phase 3" silently lands at
# "overstride-phase-3", and status/fetch then query a kernel that does not exist.
# Keep these two in agreement.
SLUG_KERNEL = "overstride-phase-3"
TITLE_KERNEL = "Overstride phase 3"


def _auth_source() -> str:
    """Which credential the client will use. Never returns the value itself."""
    import os
    kdir = Path.home() / ".kaggle"
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN env var"
    if (kdir / "access_token").exists():
        return "~/.kaggle/access_token"
    if (kdir / "kaggle.json").exists():
        return "~/.kaggle/kaggle.json (legacy)"
    raise SystemExit(
        "No Kaggle credential found.\n"
        "  kaggle.com -> Settings -> API -> Create New Token (shows a KGAT_ token;\n"
        "  it does NOT download a file), then set KAGGLE_API_TOKEN or write it to\n"
        "  ~/.kaggle/access_token. Do not paste it into a chat window."
    )


def username() -> str:
    """Resolve the account name. The KGAT_ token carries no username, so this
    tries explicit sources first and only then asks the authenticated client.

    Guessing is not an option: a wrong username creates the dataset and kernel in
    a namespace the caller does not own, and the push still looks like it worked.
    """
    import os
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    for i, a in enumerate(sys.argv):
        if a == "--user" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        user = api.get_config_value("username")
        if user:
            return str(user)
    except Exception as exc:  # noqa: BLE001 - reported, never guessed around
        raise SystemExit(f"Could not authenticate to Kaggle: {exc}") from exc
    raise SystemExit(
        "Authenticated, but no username could be resolved.\n"
        "Pass it explicitly:  kaggle_run.py push --user <your-kaggle-username>"
    )


def kaggle(*args: str) -> int:
    exe = shutil.which("kaggle")
    cmd = [exe, *args] if exe else [sys.executable, "-m", "kaggle", *args]
    print("$", " ".join(cmd[-len(args) - 1:]))
    return subprocess.run(cmd).returncode


def cmd_push() -> int:
    user = username()

    # --- dataset: the single script that has to cross ---------------------
    d = STAGE / "dataset"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "scripts" / "phase3_angles.py", d / "phase3_angles.py")
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "overstride-scripts",
        "id": f"{user}/{SLUG_DATA}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")

    rc = kaggle("datasets", "create", "-p", str(d), "--dir-mode", "zip")
    if rc != 0:
        print("create failed (dataset probably exists) -> pushing a new version")
        kaggle("datasets", "version", "-p", str(d), "-m",
               "update phase3_angles", "--dir-mode", "zip")

    # --- kernel ------------------------------------------------------------
    k = STAGE / "kernel"
    k.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "notebooks" / "phase3_kaggle.ipynb",
                k / "phase3-kaggle.ipynb")
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{SLUG_KERNEL}",
        "title": TITLE_KERNEL,
        "code_file": "phase3-kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "",
        "model_sources": [],
        "dataset_sources": [f"{user}/{SLUG_DATA}"],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2), encoding="utf-8")

    rc = kaggle("kernels", "push", "-p", str(k))
    print(f"\nkernel: https://www.kaggle.com/code/{user}/{SLUG_KERNEL}")
    print("It queues, then runs. Poll with:  scripts/kaggle_run.py status")
    return rc


def cmd_status() -> int:
    return kaggle("kernels", "status", f"{username()}/{SLUG_KERNEL}")


def cmd_fetch() -> int:
    out = STAGE / "output"
    out.mkdir(parents=True, exist_ok=True)
    rc = kaggle("kernels", "output", f"{username()}/{SLUG_KERNEL}",
                "-p", str(out))
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}  {f.stat().st_size/1e3:.1f} KB")
    print(f"\nlook for phase3_angle_errors.json under {out}")
    return rc


if __name__ == "__main__":
    actions = {"push": cmd_push, "status": cmd_status, "fetch": cmd_fetch}
    if len(sys.argv) < 2 or sys.argv[1] not in actions:
        raise SystemExit(
            f"usage: kaggle_run.py {{{'|'.join(actions)}}} [--user <name>]")
    print(f"auth: {_auth_source()}   account: {username()}")
    raise SystemExit(actions[sys.argv[1]]())

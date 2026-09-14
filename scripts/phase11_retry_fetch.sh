#!/usr/bin/env bash
# Patiently resume the AthletePose3D download.
#
# Google Drive serves these files in bursts and then refuses: one attempt pulled
# 3.3 GB of data.zip before cutting off. gdown writes a .part file and resumes,
# so progress survives each refusal and the only thing needed is patience.
#
# A generous interval, deliberately: this is someone else's share quota, and
# hammering it is both rude and counter-productive.
#
# Run detached; it exits once both files are present.

cd /c/Users/alvin/Overstride_V2 || exit 1
PY=.venv/Scripts/python.exe
DEST=data/athletepose3d
INTERVAL=600          # 10 minutes
MAX_ATTEMPTS=72       # ~12 hours

for i in $(seq 1 $MAX_ATTEMPTS); do
  have_data=0; have_2d=0
  [ -s "$DEST/data.zip" ] && have_data=1
  [ -s "$DEST/pose_2d.zip" ] && have_2d=1
  if [ $have_data -eq 1 ] && [ $have_2d -eq 1 ]; then
    echo "[$(date +%H:%M:%S)] both files present after $((i-1)) retries"
    exit 0
  fi
  part=$(ls -la "$DEST"/*.part 2>/dev/null | awk '{s+=$5} END {print s+0}')
  echo "[$(date +%H:%M:%S)] attempt $i/$MAX_ATTEMPTS | data.zip=$have_data pose_2d.zip=$have_2d | partial bytes=$part"
  "$PY" scripts/phase11_fetch_ap3d.py >/dev/null 2>&1
  sleep $INTERVAL
done
echo "[$(date +%H:%M:%S)] gave up after $MAX_ATTEMPTS attempts"
exit 1

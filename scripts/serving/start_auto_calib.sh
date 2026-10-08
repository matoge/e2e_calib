#!/bin/bash
# Idempotent launcher — if the calib API is already answering on :8501, do
# nothing; otherwise (fresh reboot, crash, docker prune, someone killed it)
# spin it back up in the background.
#
# Intended for user crontab (no sudo, no systemd):
#     */2 * * * *  /home/hfunaya/git/e2e_calib/scripts/serving/start_auto_calib.sh
#     @reboot      /home/hfunaya/git/e2e_calib/scripts/serving/start_auto_calib.sh
#
# The 2-minute cadence means recovery is within 2 min of a crash; @reboot is
# a nice-to-have that shortens boot recovery further.
set -euo pipefail

PORT=${PORT:-8501}
LOG=${LOG:-/home/hfunaya/git/e2e_calib/scripts/_watch/auto_calib_server.log}
CKPT=${CKPT:-/home/hfunaya/git/e2e_calib/experiments/kmwv_s3_ba40_512r256_0901_1344}
PY=/home/hfunaya/.pyenv/versions/3.10.4/bin/python
SRV=/home/hfunaya/git/e2e_calib/scripts/serving/auto_calib_server.py

# Alive? — /health returning JSON with model
if curl -sfS --max-time 3 "http://127.0.0.1:${PORT}/health" \
     | grep -q '"model"'; then
    exit 0
fi

# Not alive — start it. Log rotates via >> (append), size TBD later.
mkdir -p "$(dirname "$LOG")"
echo "[$(date '+%F %T')] launching auto_calib_server on :${PORT}" >> "$LOG"

# nohup + setsid + disown so parent cron shell can exit cleanly
setsid nohup "$PY" "$SRV" --port "$PORT" --ckpt "$CKPT" \
    >> "$LOG" 2>&1 </dev/null &
disown || true

# wait briefly + confirm
sleep 4
if curl -sfS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "[$(date '+%F %T')] ✓ up on :${PORT}" >> "$LOG"
else
    echo "[$(date '+%F %T')] ✗ failed to come up, see log above" >> "$LOG"
    exit 1
fi

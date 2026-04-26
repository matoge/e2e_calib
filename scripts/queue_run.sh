#!/usr/bin/env bash
# Sequential experiment runner. Usage: queue_run.sh <queuefile>
# Each line in queuefile is "name|extra args" (passed to train_cross_frame.py)
# Common args + result rsync handled here.
set -u
QFILE="${1:?queuefile required}"
LOG_DIR="/tmp/queue_logs"
mkdir -p "$LOG_DIR"
COMMON="--model unified --full \
  --scenes-root /mnt/nvme6t/pandaset_39 --train-frac 0.8 --cameras front_camera \
  --img-size 64 --max-points 256 --batch-size 64 --lr 1e-3 --epochs 30 --log-every 1 \
  --baseline-min 1 --baseline-max 20 --sigma-ypr 1.0 --sigma-t 0.2 \
  --crop-min 128 --crop-max 256 --num-workers 16 --virtual-epoch 10000 --val-pool-size 2000 \
  --no-uvd --clearml --clearml-project e2e_calib/cross-frame"
PY="${PY:-python}"

while IFS='|' read -r NAME EXTRA; do
    NAME=$(echo "$NAME" | xargs)
    [ -z "$NAME" ] && continue
    [[ "$NAME" =~ ^# ]] && continue
    LOG="$LOG_DIR/${NAME}.log"
    echo "[$(date '+%H:%M:%S')] >>> launching $NAME with: $EXTRA" | tee -a "$LOG"
    cd ~/git/e2e_calib
    $PY train_cross_frame.py --name "$NAME" $COMMON $EXTRA >> "$LOG" 2>&1
    rc=$?
    echo "[$(date '+%H:%M:%S')] <<< $NAME finished rc=$rc" | tee -a "$LOG"
done < "$QFILE"

echo "[$(date '+%H:%M:%S')] queue done"

#!/bin/bash
# Standard single-stage training for kamikado + woven_v3_cal_ok caches.
# One command, no S1→S3 chain, no manual handoff. Reproducible by anyone.
#
# ep0-10:  points-only (ba_weight=0) — μ-head warms up without BA pressure
# ep10-30: BA weight ramps 0→0.05       — InfoHead2x2 slowly takes over GN weights
# ep30-60: full BA (ba_weight=0.05)     — joint μ+InfoHead refinement
#
# Caches (frame-level, NOT --tile):
#   /raid/.../cache_v5/kamikado_v3_full
#   /raid/.../cache_v5/woven_v3_cal_ok  (built from CAL OK sequences only)
#
# Usage:
#   NAME=run_$(date +%m%d_%H%M) GPUS='4,5,6,7,8,9,10,11' PORT=29514 \
#       scripts/train_kmwv_calok.sh
#
# All env vars are optional with defaults below.
set -euo pipefail

NAME="${NAME:-kmwv_calok_$(date +%m%d_%H%M)}"
GPUS="${GPUS:-4,5,6,7,8,9,10,11}"
PORT="${PORT:-29514}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-3e-4}"
BA_WEIGHT="${BA_WEIGHT:-0.05}"
BA_WARMUP_START="${BA_WARMUP_START:-10}"
BA_WARMUP_END="${BA_WARMUP_END:-30}"
CACHE_KM="${CACHE_KM:-/raid/home/hfunaya/cache_v5/kamikado_v3_full}"
CACHE_WV="${CACHE_WV:-/raid/home/hfunaya/cache_v5/woven_v3_cal_ok}"
IMAGE="${IMAGE:-e2e-calib-train:np2}"
NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')

echo "[kick] name=$NAME  gpus=$GPUS ($NUM_GPUS procs)  epochs=$EPOCHS"
echo "[kick] BA weight=$BA_WEIGHT  warmup ep$BA_WARMUP_START → ep$BA_WARMUP_END"

docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" --network host \
    -v /home/hfunaya/git/e2e_calib:/workspace -v /raid:/raid -v /home/hfunaya:/home/hfunaya -v /mnt/fsx:/mnt/fsx \
    -v /mnt/fsx/.clearml_agent.dgx2-gpu.cfg:/root/clearml.conf:ro \
    --shm-size=64g --ipc=host \
    -e PYTHONPATH=/workspace -e CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) \
    -w /workspace "$IMAGE" \
    bash -c "accelerate launch --main_process_port $PORT --num_processes=$NUM_GPUS \
      --mixed_precision=fp16 datasets/train_cnd2_ddp.py \
      --name $NAME \
      --cache $CACHE_KM,$CACHE_WV \
      --per-cache-oversample $CACHE_KM:40,$CACHE_WV:40 \
      --per-cache-crop-px    $CACHE_KM:512,$CACHE_WV:512 \
      --epochs $EPOCHS --eval-every 5 --batch-size 4 --img-size 256 --grid-n 16 \
      --workers 4 --val-fraction 0.1 --scene-split \
      --n-iter 4 --lr $LR \
      --rot-deg 0.5 --t-m 0.20 \
      --min-crop-px 256 --max-crop-px 256 \
      --use-info-head --share-pert --crop-grid \
      --grid-iw 3840 --grid-ih 2160 --grid-frac 0.0 --oversample 40 \
      --ba-loss --ba-iter 4 --ba-damping 1e-3 \
      --ba-weight $BA_WEIGHT --ba-loss-type nll --ba-w-source infohead \
      --ba-warmup-start $BA_WARMUP_START --ba-warmup-end $BA_WARMUP_END \
      --clearml --clearml-project e2e_calib/calib \
      --why 'Single-stage km+wv_cal_ok training. ep0-$BA_WARMUP_START: pts-only. ep$BA_WARMUP_START-$BA_WARMUP_END: BA ramp. ep$BA_WARMUP_END-$EPOCHS: full BA.'"

echo "[kick] docker: $NAME"
echo "[kick] watch: docker logs -f $NAME"
echo "[kick] train.log: experiments/$NAME/train.log"

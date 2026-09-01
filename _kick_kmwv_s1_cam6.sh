#!/bin/bash
# S1 (points-only) but resumed from the 6-cam nuScenes ckpt (cam6_250x2_100ep)
# instead of the front-only one (front_670x3). Same recipe as _kick_kmwv_s1_pts_512.sh
# otherwise — 512x512 crop → 256x256 model input, 30 ep.
set -euo pipefail
NAME="kmwv_s1cam6_pts_512r256_$(date +%m%d_%H%M)"

KM=/raid/home/hfunaya/cache_v5/kamikado_v3_full
WV=/raid/home/hfunaya/cache_v5/woven_v3_full
CACHES="${KM},${WV}"
OS_MAP="${KM}:10,${WV}:8"
CROP_MAP="${KM}:512,${WV}:512"

WHY="S1 points-only warmup for km+wv joint, resumed from the 6-cam nuScenes ckpt (cam6_250x2_100ep) instead of front_670x3. Same otherwise: 512x512 per-cache crop → img_size 256 → grid_n=16, 30 ep, no BA, sigma_ypr=1.0deg, sigma_t=0.20m. Purpose: compare the front-only vs 6-cam warmstart — does the wider viewpoint prior speed up convergence on fisheye km+wv, or does the previous front_670x3 → S3 chain (F=1 0.020°, F=32 0.0067°) already saturate?"

CUDA_DEVICES=1,2,3,4,5,6,7,8 \
infra/submit_clearml_task.sh \
  --name        "$NAME" \
  --script      datasets/train_cnd2_ddp.py \
  --queue       dgx2 \
  --num-gpus    8 \
  --image       e2e-calib-train:np2 \
  --project     e2e_calib/calib \
  --args "--cache ${CACHES} --per-cache-oversample ${OS_MAP} --per-cache-crop-px ${CROP_MAP} --resume-ckpt experiments/cam6_250x2_100ep/best_model.pt --start-epoch 0 --epochs 30 --eval-every 5 --batch-size 4 --img-size 256 --grid-n 16 --workers 4 --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 1.0 --t-m 0.20 --min-crop-px 256 --max-crop-px 256 --use-info-head --clearml --clearml-project e2e_calib/calib --why \"$WHY\""

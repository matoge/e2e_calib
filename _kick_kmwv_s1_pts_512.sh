#!/bin/bash
# S1: km+wv joint, POINTS-ONLY (gaussian2d_nll), no BA.
# Per-cache crop 512x512 → resize img_size=256 → grid16 で nuScenes 条件に揃える。
# Native 3840x2160、512-tile 覆いは 8x5 = 40 windows/frame (S3 で使う。S1 は BA 無しなので
# per-cache-oversample=8/10 のランダムサンプリング)。
# Report: docs/2026-08-31_nuscenes-calibration.md §学習をどう進めたか 段1
set -euo pipefail
NAME="kmwv_s1_pts_512r256_$(date +%m%d_%H%M)"

KM=/raid/home/hfunaya/cache_v5/kamikado_v3_full
WV=/raid/home/hfunaya/cache_v5/woven_v3_full
CACHES="${KM},${WV}"
OS_MAP="${KM}:10,${WV}:8"
CROP_MAP="${KM}:512,${WV}:512"

WHY="S1 points-only warmup for km+wv joint (fisheye tss4_fcm, native 3840x2160). Following 2026-08-31 nuScenes report staged recipe: stage 1 = gaussian2d_nll only, NO BA (untrained InfoHead makes BA loss 5285 / grad 2e6 → diverges). Per-cache crop 512x512 → resize img_size=256 → grid_n=16, matches nuScenes report input scale (4K natives resized to nuScenes-equivalent 256). 30 epochs, eval-every=5, sigma_ypr=1.0deg, sigma_t=0.20m, n_iter=4. Resume from front_670x3 (nuScenes CAM_FRONT) as init. No --ba-loss, no --share-pert."

CUDA_DEVICES=1,2,3,4,5,6,7,8 \
infra/submit_clearml_task.sh \
  --name        "$NAME" \
  --script      datasets/train_cnd2_ddp.py \
  --queue       dgx2 \
  --num-gpus    8 \
  --image       e2e-calib-train:np2 \
  --project     e2e_calib/calib \
  --args "--cache ${CACHES} --per-cache-oversample ${OS_MAP} --per-cache-crop-px ${CROP_MAP} --resume-ckpt experiments/front_670x3/best_model.pt --start-epoch 0 --epochs 30 --eval-every 5 --batch-size 4 --img-size 256 --grid-n 16 --workers 4 --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 1.0 --t-m 0.20 --min-crop-px 256 --max-crop-px 256 --use-info-head --clearml --clearml-project e2e_calib/calib --why \"$WHY\""

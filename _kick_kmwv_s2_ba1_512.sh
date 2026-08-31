#!/bin/bash
# S2: km+wv joint, 1-tile BA (W=sigma), resume S1.
# Per-cache 512x512 → resize img_size=256 → grid16 で nuScenes 条件に揃える。
# 1 窓 GN、share-pert なし、W は sigma head から。
# Report: docs/2026-08-31_nuscenes-calibration.md §学習をどう進めたか 段2
#
# 使い方: S1_CKPT に S1 の best_model.pt を渡してから kick。
#   S1_CKPT=experiments/kmwv_s1_pts_512r256_<TS>/best_model.pt bash _kick_kmwv_s2_ba1_512.sh
set -euo pipefail
S1_CKPT="${S1_CKPT:?set S1_CKPT=experiments/kmwv_s1_pts_512r256_<TS>/best_model.pt}"
[ -f "$S1_CKPT" ] || { echo "S1_CKPT not found: $S1_CKPT" >&2; exit 1; }

NAME="kmwv_s2_ba1_512r256_$(date +%m%d_%H%M)"

KM=/raid/home/hfunaya/cache_v5/kamikado_v3_full
WV=/raid/home/hfunaya/cache_v5/woven_v3_full
CACHES="${KM},${WV}"
OS_MAP="${KM}:10,${WV}:8"
CROP_MAP="${KM}:512,${WV}:512"

WHY="S2 1-tile BA for km+wv joint. Resume from S1 ($S1_CKPT). ba_w_source=sigma (info from per-point sigma head; InfoHead2x2 still stays cold — it takes over in S3). BA warmup 0→0.05 over ep0-10 to avoid the 'untrained InfoHead → 2e6 grad' failure. share-pert OFF (each window solves its own GN). 20 epochs, eval-every=5, per-cache 512→256, grid16."

CUDA_DEVICES=1,2,3,4,5,6,7,8 \
infra/submit_clearml_task.sh \
  --name        "$NAME" \
  --script      datasets/train_cnd2_ddp.py \
  --queue       dgx2 \
  --num-gpus    8 \
  --image       e2e-calib-train:np2 \
  --project     e2e_calib/calib \
  --args "--cache ${CACHES} --per-cache-oversample ${OS_MAP} --per-cache-crop-px ${CROP_MAP} --resume-ckpt ${S1_CKPT} --start-epoch 0 --epochs 20 --eval-every 5 --batch-size 4 --img-size 256 --grid-n 16 --workers 4 --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 1.0 --t-m 0.20 --min-crop-px 256 --max-crop-px 256 --use-info-head --ba-loss --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-w-source sigma --ba-warmup-start 0 --ba-warmup-end 10 --clearml --clearml-project e2e_calib/calib --why \"$WHY\""

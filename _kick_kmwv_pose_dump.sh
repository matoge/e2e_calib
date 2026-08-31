#!/bin/bash
# Val-only pose dump: resume S3 ckpt, run 1 epoch with --dump-pose to save
# per-frame (delta_pred, delta_gt, H) into experiments/<name>/pose_dump_ep001.pt.
# Consumed offline by scripts/eval/frame_fusion.py.
set -euo pipefail
NAME="kmwv_pose_dump_$(date +%m%d_%H%M)"

KM=/raid/home/hfunaya/cache_v5/kamikado_v3_full
WV=/raid/home/hfunaya/cache_v5/woven_v3_full
S3_CKPT=/home/hfunaya/git/e2e_calib/experiments/kmwv_s3_ba40_512r256_0831_0325/best_model.pt

WHY="Val-only pose dump for offline F=1..32 fusion (chi2 gate + inv-var pool + over-dispersion k). Resume from S3 best_model.pt, 1 epoch, no training update — writes pose_dump_ep001.pt with per-frame (delta_pred, delta_gt, H) that scripts/eval/frame_fusion.py consumes to produce the multi-frame sweep. 40-tile crop-grid (per-cache 512), scene-split held-out val."

CUDA_DEVICES=1,2,3,4,5,6,7,8 \
infra/submit_clearml_task.sh \
  --name        "$NAME" \
  --script      datasets/train_cnd2_ddp.py \
  --queue       dgx2 \
  --num-gpus    8 \
  --image       e2e-calib-train:np2 \
  --project     e2e_calib/calib \
  --args "--cache ${KM},${WV} --per-cache-oversample ${KM}:40,${WV}:40 --per-cache-crop-px ${KM}:512,${WV}:512 --resume-ckpt ${S3_CKPT} --start-epoch 0 --epochs 1 --eval-every 1 --batch-size 4 --img-size 256 --grid-n 16 --workers 4 --val-fraction 0.1 --scene-split --n-iter 4 --lr 1e-6 --rot-deg 0.5 --t-m 0.20 --min-crop-px 256 --max-crop-px 256 --use-info-head --share-pert --crop-grid --grid-iw 3840 --grid-ih 2160 --grid-frac 0.0 --oversample 40 --ba-loss --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-w-source infohead --ba-warmup-start 0 --ba-warmup-end 0 --dump-pose --clearml --clearml-project e2e_calib/calib --why \"$WHY\""

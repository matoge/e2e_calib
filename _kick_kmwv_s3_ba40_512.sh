#!/bin/bash
# S3: km+wv joint, FUSED 40-tile BA (W=InfoHead), resume S2.
# Per-cache 512x512 → resize img_size=256 → grid16。
# share-pert: 1 frame の 40 tile が同じ δ を持ち、1 つの GN で融合。
# crop-grid: 決定的タイル配置で画像全体を覆う (grid_iw=3840, grid_ih=2160)。
# per-cache-oversample=40 で各フレーム 40 tile emit。
# Report: docs/2026-08-31_nuscenes-calibration.md §学習をどう進めたか 段3
#
# 使い方: S2_CKPT に S2 の best_model.pt を渡してから kick。
#   S2_CKPT=experiments/kmwv_s2_ba1_512r256_<TS>/best_model.pt bash _kick_kmwv_s3_ba40_512.sh
set -euo pipefail
S2_CKPT="${S2_CKPT:?set S2_CKPT=experiments/kmwv_s2_ba1_512r256_<TS>/best_model.pt}"
[ -f "$S2_CKPT" ] || { echo "S2_CKPT not found: $S2_CKPT" >&2; exit 1; }

NAME="kmwv_s3_ba40_512r256_$(date +%m%d_%H%M)"

KM=/raid/home/hfunaya/cache_v5/kamikado_v3_full
WV=/raid/home/hfunaya/cache_v5/woven_v3_full
CACHES="${KM},${WV}"
# 3840/512=8, 2160/512=ceil(4.22)=5 → 40 windows/frame。crop_grid が自動導出する
# が、per-cache-oversample を明示して override を避ける。
OS_MAP="${KM}:40,${WV}:40"
CROP_MAP="${KM}:512,${WV}:512"

WHY="S3 fused 40-tile BA for km+wv joint. Resume from S2 ($S2_CKPT). ba_w_source=infohead (InfoHead2x2 now takes over the GN weights, sigma stays on gaussian2d_nll — the split that let nuScenes drop chi2/6 3113→1.0). share-pert ON + crop-grid ON: 1 frame の 40 tile が同じ δ を持ち 1 つの GN に融合。40 = ceil(3840/512)*ceil(2160/512) = 8*5。BA warmup 0→0.05 over ep0-10。50 epochs, eval-every=5, per-cache 512→256, grid16。"

CUDA_DEVICES=1,2,3,4,5,6,7,8 \
infra/submit_clearml_task.sh \
  --name        "$NAME" \
  --script      datasets/train_cnd2_ddp.py \
  --queue       dgx2 \
  --num-gpus    8 \
  --image       e2e-calib-train:np2 \
  --project     e2e_calib/calib \
  --args "--cache ${CACHES} --per-cache-oversample ${OS_MAP} --per-cache-crop-px ${CROP_MAP} --resume-ckpt ${S2_CKPT} --start-epoch 0 --epochs 50 --eval-every 5 --batch-size 4 --img-size 256 --grid-n 16 --workers 4 --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.20 --min-crop-px 256 --max-crop-px 256 --use-info-head --share-pert --crop-grid --grid-iw 3840 --grid-ih 2160 --grid-frac 0.0 --oversample 40 --ba-loss --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-w-source infohead --ba-warmup-start 0 --ba-warmup-end 10 --clearml --clearml-project e2e_calib/calib --why \"$WHY\""

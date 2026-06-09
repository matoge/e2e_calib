#!/bin/bash
# CND2 mixed mode 50ep on DGX4 (16 GPU available after llama70b stop):
#   pair (pose+calib_pert via cross-frame): PS, NS, Waymo
#   calib (single-frame): kamikado, woven_sequence, tss4 (u_band=0.8)
#
# Pair side runs forward_cross_frame with --calib-pert + --pair-bidir;
# calib side runs the legacy single-frame calib head. Both share the same
# model parameters; epoch loop alternates one full pass over each side.
#
# eval-every=10. Per-cache oversample sqrt(100k/n)>=4.
set -euo pipefail
NAME="cnd2_mixed_6ds_50ep_$(date +%m%d_%H%M)"

PS=/raid/home/hfunaya/cache_v5/pandaset_v3_full
KM=/raid/home/hfunaya/cache_v5/kamikado_v3_full
WV=/raid/home/hfunaya/cache_v5/woven_v3_full
TS=/raid/home/hfunaya/cache_v5/tss4_v3_full_iter1kb4_yaw3
NS=/raid/home/hfunaya/cache_v5/ns_v3_full
WM=/raid/home/hfunaya/cache_v5/waymo_v3_full
CACHES="${PS},${KM},${WV},${TS},${NS},${WM}"
OS_MAP="${PS}:4,${KM}:10,${WV}:8,${TS}:4,${NS}:4,${WM}:4"
MODE_MAP="${PS}:pair,${NS}:pair,${WM}:pair,${KM}:calib,${WV}:calib,${TS}:calib"

WHY="CND2 MIXED mode 50ep on 6 caches (DGX4 16 GPU). Pair side (PS+NS+Waymo): pose learning via forward_cross_frame + bidir + calib_pert ε_calib_A multi-task. Calib side (kamikado+woven+tss4): single-frame calib head only. Per-cache os: kami10 woven8 tss4-4 PS-4 ns-4 wm-4. eval-every=10, batch=24, img=128, grid=16, n_iter=4, sigma_ypr=1.0deg sigma_t=0.20m. Pair-side u_band=0; calib-side u_band only on tss4 (0.8). Goal: confirm pose+calib joint training across pinhole+fisheye crop diversity converges, baseline for calibration drift recovery on a CND1-like multi-dataset stack."

CUDA_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
infra/submit_clearml_task.sh \
  --name        "$NAME" \
  --script      datasets/train_cnd2_ddp.py \
  --queue       dgx4 \
  --num-gpus    16 \
  --image       e2e-calib-train:np2 \
  --project     e2e_calib/cross-frame \
  --args "--cache ${CACHES} --per-cache-oversample ${OS_MAP} --per-cache-mode ${MODE_MAP} --pair-bidir --calib-pert --w-calib 1.0 --w-pose 1.0 --epochs 50 --eval-every 10 --batch-size 24 --img-size 128 --grid-n 16 --workers 8 --val-fraction 0.1 --n-iter 4 --rot-deg 1.0 --t-m 0.20 --min-crop-px 128 --max-crop-px 512 --u-band ${TS}:0.8 --use-info-head --clearml --clearml-project e2e_calib/cross-frame --why \"$WHY\""

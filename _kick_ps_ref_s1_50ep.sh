#!/bin/bash
# PandaSet reference run (stage 1, tiles, no BA) — the setting to reproduce on another machine.
# Same as ps_cn2_fourier10_s1_50ep_nearest_cam on yokohama0 (2026-10-10):
#   CalibNet2, scratch, PandaSet front camera only, scene split (cache meta), val fixed by --val-seed,
#   fixed 256 px crops (no resize), query = point nearest the camera in each 16x16 cell, 24 points/cell,
#   own-cell neighbourhood, Fourier features on PointMLP3 (L=10), 50 epochs, 8 windows/frame, batch 4 frames.
# yokohama0 result at ep50 (val 400 frames): va_nll 4.000  va_mse 7.153 px  POSE rot 0.1958 deg  t 0.0718 m
#   (same without --point-mlp-fourier-n-freq: 3.985 / 7.230 / 0.2014 / 0.0757)
# Usage:  C=/path/to/cache_root ./_kick_ps_ref_s1_50ep.sh      (expects $C/pandaset_pad256)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD
: "${C:=/mnt/ssd2t/work/e2e_calib/cache}"
: "${NAME:=ps_ref_s1_50ep}"
python -u datasets/train_cnd2_ddp.py \
  --cache $C/pandaset_pad256 \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 8 --epochs 50 --eval-every 10 \
  --cross-attn deform --ref-mode query --rep-strategy nearest_cam --k-per-cell 24 --frustum-nb own \
  --crop-pivot image --point-mlp-fourier-n-freq 10 \
  --name $NAME \
  --why "PandaSet reference (CalibNet2, scratch, 256 fixed, nearest_cam, 24/cell own, Fourier L=10, 50 ep). yokohama0 ep50: va_nll 4.000 va_mse 7.153 rot 0.1958 t 0.0718"

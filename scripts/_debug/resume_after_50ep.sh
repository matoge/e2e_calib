#!/bin/bash
# Wait for both DGX1 (n=4) and DGX2 (n=2) to finish 50ep training, then
# kick the resume runs with init_from=<that ckpt>, crop 256-512, 150 more
# epochs.
set -euo pipefail

while docker ps --filter name=km_wv_wm_dgx2_n2 --format '{{.Status}}' | grep -q Up; do sleep 60; done
echo "[$(date)] DGX2 base done"
while ssh dgx1 "docker ps --filter name=km_wv_wm_dgx1_n4 --format '{{.Status}}' | grep -q Up"; do sleep 60; done
echo "[$(date)] DGX1 base done"

docker run -d --name km_wv_wm_dgx2_n2_resume \
  --gpus '"device=8,9,10,11,12,13,14,15"' --shm-size=128g \
  --ulimit nofile=1048576:1048576 \
  -v /home/hfunaya/git/e2e_calib:/workspace \
  -v /home/hfunaya/cache/kamikado_v3_tiled:/cache/kamikado_v3_tiled:ro \
  -v /home/hfunaya/cache/woven_v3_tile:/cache/woven_v3_tile:ro \
  -v /home/hfunaya/clearml/data/cache/waymo_v3_tiled_i:/cache/waymo_v3_tiled_i:ro \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  e2e-calib-train:np2 \
  bash -c 'cd /workspace && accelerate launch --num_processes=8 --mixed_precision=fp16 scripts/training/train_ps_v3_ddp.py --name km_wv_wm_dgx2_n2_v4_resume --cache /cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i --epochs 150 --batch-size 192 --workers 8 --oversample 4 --n-layers 2 --val-size 800 --min-crop-px 256 --max-crop-px 512 --init-from km_wv_wm_dgx2_n2_v4 --deform-mode sl --clearml --why DGX2_n2_resume_crop256-512'

ssh dgx1 "docker run -d --name km_wv_wm_dgx1_n4_resume \
  --gpus '\"device=0,1,2,3,4,5,6,7\"' --shm-size=128g \
  --ulimit nofile=1048576:1048576 \
  -v /home/hfunaya/git/e2e_calib:/workspace \
  -v /raid/home/hfunaya/cache/kamikado_v3_tiled:/cache/kamikado_v3_tiled:ro \
  -v /raid/home/hfunaya/cache/woven_v3_tile:/cache/woven_v3_tile:ro \
  -v /raid/home/hfunaya/cache/waymo_v3_tiled_i:/cache/waymo_v3_tiled_i:ro \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  e2e-calib-train:np2 \
  bash -c 'cd /workspace && accelerate launch --num_processes=8 --mixed_precision=fp16 scripts/training/train_ps_v3_ddp.py --name km_wv_wm_dgx1_n4_v4_resume --cache /cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i --epochs 150 --batch-size 192 --workers 8 --oversample 4 --n-layers 4 --val-size 800 --min-crop-px 256 --max-crop-px 512 --init-from km_wv_wm_dgx1_n4_v4 --deform-mode sl --clearml --why DGX1_n4_resume_crop256-512'"
echo "[$(date)] both resume kicked"

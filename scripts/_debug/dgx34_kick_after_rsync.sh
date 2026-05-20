#!/bin/bash
# Wait for kw + wm rsyncs to dgx3/dgx4 then enqueue tasks via clearml-task.
set -euo pipefail
export CLEARML_CONFIG_FILE=/home/hfunaya/clearml-dgx2.conf

echo "[$(date)] waiting for rsync to dgx3..."
while ps -ef | grep -E "[r]sync.*dgx3:/raid" | grep -q .; do
  sleep 60
done
echo "[$(date)] dgx3 rsync done."

echo "[$(date)] waiting for rsync to dgx4..."
while ps -ef | grep -E "[r]sync.*dgx4:/raid" | grep -q .; do
  sleep 60
done
echo "[$(date)] dgx4 rsync done."

# Approach: docker run -d directly on dgx3/dgx4 (the agent isn't strictly
# needed for our docker-launch model — we already have the image+cache+config).
# This keeps things consistent with DGX1/DGX2 style.

echo "[$(date)] kicking DGX3 (n=2 img=192) ..."
ssh dgx3 "docker rm -f km_wv_wm_dgx3_n2_img192 2>/dev/null; docker run -d --name km_wv_wm_dgx3_n2_img192 \
  --gpus all --shm-size=128g --ulimit nofile=1048576:1048576 \
  -v /home/hfunaya/git/e2e_calib:/workspace \
  -v /raid/home/hfunaya/cache/kamikado_v3_tiled:/cache/kamikado_v3_tiled:ro \
  -v /raid/home/hfunaya/cache/woven_v3_tile:/cache/woven_v3_tile:ro \
  -v /raid/home/hfunaya/cache/waymo_v3_tiled_i:/cache/waymo_v3_tiled_i:ro \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  e2e-calib-train:np2 \
  bash -c 'cd /workspace && accelerate launch --num_processes=8 --mixed_precision=fp16 scripts/training/train_ps_v3_ddp.py --name km_wv_wm_dgx3_n2_img192_200ep --cache /cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i --epochs 200 --batch-size 96 --workers 8 --oversample 4 --n-layers 2 --img-size 192 --grid-n 24 --min-crop-px 192 --max-crop-px 512 --deform-mode sl --clearml --why DGX3_3ds_n2_img192_grid24'"
echo "[$(date)] DGX3 kicked."

echo "[$(date)] kicking DGX4 (n=4 img=192) ..."
ssh dgx4 "docker rm -f km_wv_wm_dgx4_n4_img192 2>/dev/null; docker run -d --name km_wv_wm_dgx4_n4_img192 \
  --gpus all --shm-size=128g --ulimit nofile=1048576:1048576 \
  -v /home/hfunaya/git/e2e_calib:/workspace \
  -v /raid/home/hfunaya/cache/kamikado_v3_tiled:/cache/kamikado_v3_tiled:ro \
  -v /raid/home/hfunaya/cache/woven_v3_tile:/cache/woven_v3_tile:ro \
  -v /raid/home/hfunaya/cache/waymo_v3_tiled_i:/cache/waymo_v3_tiled_i:ro \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  e2e-calib-train:np2 \
  bash -c 'cd /workspace && accelerate launch --num_processes=8 --mixed_precision=fp16 scripts/training/train_ps_v3_ddp.py --name km_wv_wm_dgx4_n4_img192_200ep --cache /cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i --epochs 200 --batch-size 96 --workers 8 --oversample 4 --n-layers 4 --img-size 192 --grid-n 24 --min-crop-px 192 --max-crop-px 512 --deform-mode sl --clearml --why DGX4_3ds_n4_img192_grid24'"
echo "[$(date)] DGX4 kicked."

#!/bin/bash
# After current km_wv_wm_dgx{1,2}_n{2,4}_img128 trains finish, kick the
# img=192/grid=24 follow-up runs (same n_layers per machine).
set -euo pipefail

dgx2_name=km_wv_wm_dgx2_n2
dgx1_name=km_wv_wm_dgx1_n4

echo "[$(date)] waiting for $dgx2_name on DGX2 ..."
while docker ps --filter name=$dgx2_name --format '{{.Status}}' | grep -q Up ; do
  sleep 60
done
echo "[$(date)] $dgx2_name finished. Kicking img=192 follow-up on DGX2..."
docker rm -f km_wv_wm_dgx2_n2_img192 2>/dev/null || true
docker run -d --name km_wv_wm_dgx2_n2_img192 \
  --gpus '"device=8,9,10,11,12,13,14,15"' --shm-size=128g \
  --ulimit nofile=1048576:1048576 \
  -v /home/hfunaya/git/e2e_calib:/workspace \
  -v /home/hfunaya/cache/kamikado_v3_tiled:/cache/kamikado_v3_tiled:ro \
  -v /home/hfunaya/cache/woven_v3_tile:/cache/woven_v3_tile:ro \
  -v /home/hfunaya/clearml/data/cache/waymo_v3_tiled_i:/cache/waymo_v3_tiled_i:ro \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  e2e-calib-train:np2 \
  bash -c "cd /workspace && accelerate launch --num_processes=8 --mixed_precision=fp16 scripts/training/train_ps_v3_ddp.py --name km_wv_wm_dgx2_n2_img192_200ep --cache /cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i --epochs 200 --batch-size 96 --workers 8 --oversample 4 --n-layers 2 --img-size 192 --grid-n 24 --min-crop-px 192 --max-crop-px 512 --deform-mode sl --clearml --why DGX2_3ds_n2_img192_grid24"
echo "[$(date)] DGX2 img=192 kicked."

echo "[$(date)] waiting for $dgx1_name on DGX1 ..."
while ssh dgx1 "docker ps --filter name=$dgx1_name --format '{{.Status}}' | grep -q Up" ; do
  sleep 60
done
echo "[$(date)] $dgx1_name finished. Kicking img=192 follow-up on DGX1..."
ssh dgx1 "docker rm -f km_wv_wm_dgx1_n4_img192 2>/dev/null; docker run -d --name km_wv_wm_dgx1_n4_img192 \
  --gpus '\"device=0,1,2,3,4,5,6,7\"' --shm-size=128g \
  --ulimit nofile=1048576:1048576 \
  -v /home/hfunaya/git/e2e_calib:/workspace \
  -v /raid/home/hfunaya/cache/kamikado_v3_tiled:/cache/kamikado_v3_tiled:ro \
  -v /raid/home/hfunaya/cache/woven_v3_tile:/cache/woven_v3_tile:ro \
  -v /raid/home/hfunaya/cache/waymo_v3_tiled_i:/cache/waymo_v3_tiled_i:ro \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  e2e-calib-train:np2 \
  bash -c 'cd /workspace && accelerate launch --num_processes=8 --mixed_precision=fp16 scripts/training/train_ps_v3_ddp.py --name km_wv_wm_dgx1_n4_img192_200ep --cache /cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i --epochs 200 --batch-size 96 --workers 8 --oversample 4 --n-layers 4 --img-size 192 --grid-n 24 --min-crop-px 192 --max-crop-px 512 --deform-mode sl --clearml --why DGX1_3ds_n4_img192_grid24'"
echo "[$(date)] DGX1 img=192 kicked."

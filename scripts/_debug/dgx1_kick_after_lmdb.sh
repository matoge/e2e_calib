#!/bin/bash
set -euo pipefail
echo "[$(date)] waiting for waymo LMDB rsync ..."
while ps -ef | grep -E "[r]sync.*waymo_v3_tiled_i" | grep -q .; do
  sleep 30
done
echo "[$(date)] rsync done; verifying..."
sleep 10
ssh dgx1 "ls -lh /raid/home/hfunaya/cache/waymo_v3_tiled_i/data.lmdb/data.mdb /raid/home/hfunaya/cache/waymo_v3_tiled_i/meta.pt"
echo "[$(date)] kicking DGX1 n=4 train..."
ssh dgx1 "docker rm -f km_wv_wm_dgx1_n4 2>/dev/null; docker run -d --name km_wv_wm_dgx1_n4 \
  --gpus '\"device=0,1,2,3,4,5,6,7\"' --shm-size=128g \
  --ulimit nofile=1048576:1048576 \
  -v /home/hfunaya/git/e2e_calib:/workspace \
  -v /raid/home/hfunaya/cache/kamikado_v3_tiled:/cache/kamikado_v3_tiled:ro \
  -v /raid/home/hfunaya/cache/woven_v3_tile:/cache/woven_v3_tile:ro \
  -v /raid/home/hfunaya/cache/waymo_v3_tiled_i:/cache/waymo_v3_tiled_i:ro \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  e2e-calib-train:np2 \
  bash -c 'cd /workspace && accelerate launch --num_processes=8 --mixed_precision=fp16 scripts/training/train_ps_v3_ddp.py --name km_wv_wm_dgx1_n4_img128_200ep --cache /cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i --epochs 200 --batch-size 192 --workers 8 --oversample 4 --n-layers 4 --deform-mode sl --clearml --why DGX1_3ds_n4_img128_default'"
echo "[$(date)] kicked."

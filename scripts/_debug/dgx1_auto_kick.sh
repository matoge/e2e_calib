#!/bin/bash
set -euo pipefail
IMG=e2e-calib-train:np2

echo "[$(date)] waiting for rsync to finish ..."
while ps -ef | grep -E "[r]sync.*chacha20" | grep -q .; do
  sleep 30
done
echo "[$(date)] rsync done."

if ! ssh dgx1 "docker images $IMG --format '{{.ID}}' | head -1 | grep -q ." ; then
  echo "[$(date)] image $IMG missing on dgx1, transferring..."
  docker save "$IMG" | ssh -c chacha20-poly1305@openssh.com dgx1 'docker load'
  echo "[$(date)] image loaded."
else
  echo "[$(date)] image already on dgx1."
fi

echo "[$(date)] kicking training on dgx1 ..."
ssh dgx1 "docker rm -f km_wv_wm_8gpu 2>/dev/null; docker run -d --name km_wv_wm_8gpu \
  --gpus '\"device=0,1,2,3,4,5,6,7\"' --shm-size=128g \
  --ulimit nofile=1048576:1048576 \
  -v /home/hfunaya/git/e2e_calib:/workspace \
  -v /raid/home/hfunaya/cache/kamikado_v3_tiled:/cache/kamikado_v3_tiled:ro \
  -v /raid/home/hfunaya/cache/woven_v3_tile:/cache/woven_v3_tile:ro \
  -v /raid/home/hfunaya/cache/waymo_v3_tiled_i:/cache/waymo_v3_tiled_i:ro \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  $IMG \
  bash -c 'cd /workspace && accelerate launch --num_processes=8 --mixed_precision=fp16 scripts/training/train_ps_v3_ddp.py --name km_wv_wm_dgx1_8gpu_200ep_os4 --cache /cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i --epochs 200 --batch-size 384 --workers 8 --oversample 4 --min-crop-px 128 --max-crop-px 256 --deform-mode sl --clearml --why kamikado+woven+waymo_8GPU_bs384_os4_200ep_DGX1'"
echo "[$(date)] kicked."

#!/bin/bash
# Enqueue 2 tasks on the DGX2 ClearML server, one per dgx3/dgx4 queue.
# The clearml-agent on each host picks it up, pulls e2e-calib-train:np2,
# mounts the cache from /raid/home/hfunaya/cache, and runs the trainer.
set -euo pipefail
export CLEARML_CONFIG_FILE=/home/hfunaya/clearml-dgx2.conf

cd /home/hfunaya/git/e2e_calib

# DGX3 = n=2 img=192 (matches DGX2 n=2 img=128 baseline scaled up)
docker run --rm \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  -v /home/hfunaya/git/e2e_calib:/workspace \
  e2e-calib-train:np2 \
  clearml-task \
    --project "e2e_calib" \
    --name km_wv_wm_dgx3_n2_img192_200ep \
    --queue dgx3 \
    --docker e2e-calib-train:np2 \
    --docker_args "--gpus all --shm-size=128g --ulimit nofile=1048576:1048576 -v /raid/home/hfunaya/cache/kamikado_v3_tiled:/cache/kamikado_v3_tiled:ro -v /raid/home/hfunaya/cache/woven_v3_tile:/cache/woven_v3_tile:ro -v /raid/home/hfunaya/cache/waymo_v3_tiled_i:/cache/waymo_v3_tiled_i:ro" \
    --script /workspace/scripts/training/train_ps_v3_ddp.py \
    --args \
      cache=/cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i \
      epochs=200 batch_size=96 num_workers=8 oversample=4 n_layers=2 \
      img_size=192 grid_n=24 min_crop_px=192 max_crop_px=512 deform_mode=sl

docker run --rm \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  -v /home/hfunaya/git/e2e_calib:/workspace \
  e2e-calib-train:np2 \
  clearml-task \
    --project "e2e_calib" \
    --name km_wv_wm_dgx4_n4_img192_200ep \
    --queue dgx4 \
    --docker e2e-calib-train:np2 \
    --docker_args "--gpus all --shm-size=128g --ulimit nofile=1048576:1048576 -v /raid/home/hfunaya/cache/kamikado_v3_tiled:/cache/kamikado_v3_tiled:ro -v /raid/home/hfunaya/cache/woven_v3_tile:/cache/woven_v3_tile:ro -v /raid/home/hfunaya/cache/waymo_v3_tiled_i:/cache/waymo_v3_tiled_i:ro" \
    --script /workspace/scripts/training/train_ps_v3_ddp.py \
    --args \
      cache=/cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i \
      epochs=200 batch_size=96 num_workers=8 oversample=4 n_layers=4 \
      img_size=192 grid_n=24 min_crop_px=192 max_crop_px=512 deform_mode=sl

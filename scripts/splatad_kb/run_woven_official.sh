#!/bin/bash
# Run gsplat 1.5.3 official simple_trainer.py on woven_sequence via WovenParser.
# Usage:
#   bash run_woven_official.sh <seq_dir> <out_dir> [iters] [gpu]
set -euo pipefail
SEQ="${1:?seq_dir required}"
OUT="${2:?out_dir required}"
ITERS="${3:-5000}"
GPU="${4:-4}"
TS=$(date +%m%d_%H%M)
NAME="woven_kb_official_${TS}"

# Mount the host repo at /host_e2e_calib so simple_trainer.py can
# find woven_parser.py via WOVEN:... + sys.path.
docker run -d --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$GPU" -e CUDA_VISIBLE_DEVICES=0 \
  -e WOVEN_RECALIB_JSON=/loom/backend/assets/woven_sequence/llinking_26/recalibration.json \
  -e CLEARML_BIND=1 \
  -e CLEARML_PROJECT=e2e_calib/splat_kb \
  -e CLEARML_TASK_NAME="$NAME" \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  --shm-size=16g --ipc=host \
  --name "$NAME" \
  -v /raid/home/hfunaya/gsplat_examples/gsplat:/repo:ro \
  -v /home/hfunaya/git/e2e_calib:/host_e2e_calib:ro \
  -v /home/hfunaya/git/loom:/loom:ro \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -v /mnt/ecp-perception:/mnt/ecp-perception:ro \
  -v /raid/home/hfunaya:/raid \
  -v /raid/home/hfunaya/torch_extensions_cache:/root/.cache/torch_extensions \
  e2e-calib-splatkb:v1-examples \
  bash -lc "cd /repo/examples && python simple_trainer.py mcmc \
      --eval_steps 2000 4000 6000 8000 10000 \
      --tb_every 100 \
      --disable_viewer --data_factor 1 \
      --strategy.cap-max 1500000 \
      --opacity_reg 0.001 --init_scale 0.5 \
      --camera_model fisheye --with_ut --with_eval3d \
      --render_traj_path skip \
      --max_steps $ITERS \
      --data_dir 'WOVEN:$SEQ#248#/host_e2e_calib/scripts/webui_kb_fit/_outputs/_masks/baked_pylon_seq' \
      --result_dir '$OUT'"
echo "name=$NAME"
echo "tail:  docker logs -f $NAME"
echo "out:   $OUT"

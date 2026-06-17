#!/usr/bin/env bash
# End-to-end woven_sequence -> GS training.
#   1. (skip if exist) extract tar.gz, lidar_deskew, SAM3 mask, bake mask
#   2. Kick gsplat 1.5.3 simple_trainer with WovenParserPinhole
#
# Usage:
#   run_woven_gs.sh <tar.gz path> [GPU] [TAG]
#   run_woven_gs.sh <seq_dir>     [GPU] [TAG]   # if already extracted
#
set -euo pipefail
INPUT=${1:?tar.gz or seq dir required}
GPU=${2:-4}
TAG=${3:-raw02}

REPO=/home/hfunaya/git/e2e_calib
PY=/home/hfunaya/.pyenv/versions/3.10.4/bin/python
SEQ_ROOT=/raid/home/hfunaya/woven_sequence_extracted_raw02

# Resolve SEQ_DIR
if [[ -d "$INPUT" ]]; then
  SEQ_DIR="$INPUT"
elif [[ "$INPUT" == *.tar.gz ]]; then
  SEQ_NAME=$(basename "$INPUT" .tar.gz)
  SEQ_DIR="$SEQ_ROOT/$SEQ_NAME"
  mkdir -p "$SEQ_ROOT"
  if [[ ! -d "$SEQ_DIR/tss4_fcm" ]]; then
    echo "[extract] $INPUT -> $SEQ_DIR"
    tar -xzf "$INPUT" -C "$SEQ_ROOT"
  fi
else
  echo "input must be a tar.gz or extracted seq dir: $INPUT"; exit 1
fi
SEQ_NAME=$(basename "$SEQ_DIR")
N_FCM=$(ls "$SEQ_DIR/tss4_fcm" | wc -l)
N_VLS=$(ls "$SEQ_DIR/vls128" | wc -l)
echo "[seq] $SEQ_NAME  fcm=$N_FCM vls=$N_VLS"

LOG_DIR=/raid/home/hfunaya/_preprocess_logs
mkdir -p "$LOG_DIR"

# 1. lidar deskew
if [[ ! -d "$SEQ_DIR/vls128_rear_axle" ]] || [[ -z $(ls "$SEQ_DIR/vls128_rear_axle" 2>/dev/null) ]]; then
  echo "[1/4] lidar_deskew"
  $PY "$REPO/scripts/webui_kb_fit/lidar_deskew.py" \
      --raw-seq-dir "$SEQ_DIR" --out-seq-dir "$SEQ_DIR"
  if [[ -d "$SEQ_DIR/vls128_rear_axle_deskew" ]] && [[ ! -d "$SEQ_DIR/vls128_rear_axle" ]]; then
    mv "$SEQ_DIR/vls128_rear_axle_deskew" "$SEQ_DIR/vls128_rear_axle"
  fi
else
  echo "[1/4] vls128_rear_axle exists ($(ls "$SEQ_DIR/vls128_rear_axle" | wc -l) npz)"
fi

# 2. SAM3 mask (host pyenv 3.10.4 with transformers 5.6 dev)
if [[ ! -d "$SEQ_DIR/_sam3/inst" ]] || [[ -z $(ls "$SEQ_DIR/_sam3/inst" 2>/dev/null) ]]; then
  echo "[2/4] SAM3 (GPU $GPU)"
  HF_TOKEN=$HF_READ_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
    $PY "$REPO/scripts/webui_kb_fit/sam3_dynamic_mask.py" \
        --seq-dir "$SEQ_DIR" --out-dir "$SEQ_DIR/_sam3" \
    | tee "$LOG_DIR/sam3_${SEQ_NAME}.log"
else
  echo "[2/4] SAM3 inst exists ($(ls "$SEQ_DIR/_sam3/inst" | wc -l) files)"
fi

# 3. bake masks (fisheye-native, full-res 3840x1952)
if [[ ! -d "$SEQ_DIR/_masks_baked" ]] || [[ -z $(ls "$SEQ_DIR/_masks_baked" 2>/dev/null) ]]; then
  echo "[3/4] bake masks (fisheye)"
  $PY "$REPO/scripts/splatad_kb/bake_masks.py" \
      --seq-dir "$SEQ_DIR" \
      --out-dir "$SEQ_DIR/_masks_baked" \
      --inst-dir "$SEQ_DIR/_sam3/inst" \
      --vehicle 248
else
  echo "[3/4] _masks_baked exists ($(ls "$SEQ_DIR/_masks_baked" | wc -l) png)"
fi

# 4. GS training (woven_parser, fisheye native)
TS=$(date +%m%d_%H%M)
NAME="${TAG}_${SEQ_NAME}_${TS}"
RESULT_DIR=/raid/home/hfunaya/_splat_kb/$NAME
echo "[4/4] kick GS training: $NAME on GPU $GPU"
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --runtime=nvidia --shm-size=16g --name "$NAME" \
  -e NVIDIA_VISIBLE_DEVICES=$GPU -e CUDA_VISIBLE_DEVICES=0 \
  -e WOVEN_RECALIB_JSON=/recalib.json \
  -e CLEARML_BIND=1 \
  -e CLEARML_PROJECT=e2e_calib/splat_kb \
  -e CLEARML_TASK_NAME="$NAME" \
  -v "$REPO":/host_e2e_calib \
  -v /home/hfunaya/git/loom/backend/assets/woven_sequence/llinking_26/recalibration.json:/recalib.json:ro \
  -v /raid/home/hfunaya:/raid/home/hfunaya \
  -v /raid/home/hfunaya/torch_extensions_cache:/root/.cache/torch_extensions \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -v /raid/home/hfunaya/gsplat_examples/gsplat:/repo:ro \
  -w /repo/examples \
  e2e-calib-splatkb:v1-examples \
  bash -lc "cd /repo/examples && python simple_trainer.py mcmc \
      --eval_steps 7000 15000 30000 \
      --tb_every 100 --disable_viewer --data_factor 1 \
      --strategy.cap-max 5000000 --opacity_reg 0.001 --init_scale 0.5 \
      --camera_model fisheye --with_ut --with_eval3d --render_traj_path skip \
      --pose_opt --pose_opt_lr 1e-5 --pose_opt_reg 1e-6 \
      --depth_loss --depth_lambda 0.5 \
      --scale_reg 0.01 \
      --max_steps 30000 \
      --data_dir 'WOVEN:$SEQ_DIR#248#$SEQ_DIR/_masks_baked' \
      --result_dir $RESULT_DIR"
echo "container=$NAME"
echo "result_dir=$RESULT_DIR"
echo "log: docker logs $NAME"

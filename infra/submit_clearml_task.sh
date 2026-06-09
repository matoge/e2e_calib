#!/bin/bash
# infra/submit_clearml_task.sh
#
# ClearML queue submitter — minimal. agent が docker run した container 内で、
# bind-mount された repo に cd して `accelerate launch <script> <args>` を
# 一発叩くだけ。launcher も argparse 変換もしない。
#
# 使い方:
#   ./infra/submit_clearml_task.sh \
#       --name km_only_15deg_06m \
#       --script scripts/training/train_ps_v3_ddp.py \
#       --queue dgx2 \
#       --num-gpus 8 \
#       --image e2e-calib-train:np2 \
#       --args "--cache /home/hfunaya/cache_v4/kamikado_v3_tiled --epochs 40 ..."
#
# GPU 限定: CUDA_DEVICES=4,5,6,7,8,9,10,11 を env で渡す
#   → bash_setup_script の中で CUDA_VISIBLE_DEVICES=$CUDA_DEVICES を立てる。
#     docker --gpus は agent が勝手に "all" を挿すのでそれに任せる。

set -euo pipefail

NAME=""
PROJECT="e2e_calib/calib"
SCRIPT=""
QUEUE="dgx2"
ARGS=""
DOCKER_IMAGE="e2e-calib-train:np2"
NUM_PROCESSES=8

usage() {
  cat <<EOF
Usage: $0 --name <run_name> --script <path/to/train.py> --queue <q> --num-gpus N [options]

Required:
  --name     NAME      ClearML task 名
  --script   PATH      実行する python script (リポジトリルートからの相対)
  --queue    QUEUE     ClearML queue (dgx1 / dgx2 / dgx3 / dgx4)

Optional:
  --project  PROJECT   (default: e2e_calib)
  --args     "STR"     script に渡す追加 CLI 引数 (クォートして一塊で)
  --image    IMAGE     docker image (default: e2e-calib-train:np2)
  --num-gpus N         accelerate --num_processes (default: 8)

Env:
  CUDA_DEVICES=a,b,c   container 内で CUDA_VISIBLE_DEVICES に流す
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)         NAME="$2";          shift 2 ;;
    --project)      PROJECT="$2";       shift 2 ;;
    --script)       SCRIPT="$2";        shift 2 ;;
    --queue)        QUEUE="$2";         shift 2 ;;
    --args)         ARGS="$2";          shift 2 ;;
    --image)        DOCKER_IMAGE="$2";  shift 2 ;;
    --num-gpus)     NUM_PROCESSES="$2"; shift 2 ;;
    -h|--help)      usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

[[ -z "$NAME"   ]] && { echo "[err] --name required" >&2;   usage; }
[[ -z "$SCRIPT" ]] && { echo "[err] --script required" >&2; usage; }

# Auto-inject --name into the python script args so the on-disk exp dir
# (`experiments/<CFG['name']>/`) matches the ClearML task name. Prepended
# so a caller who explicitly passes --name in --args still wins (argparse
# keeps the last occurrence).
ENTRY="accelerate launch --num_processes=${NUM_PROCESSES} --mixed_precision=fp16 ${SCRIPT} --name ${NAME} ${ARGS}"

CUDA_LINE="true"
if [[ -n "${CUDA_DEVICES:-}" ]]; then
  CUDA_LINE="export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
  echo "[info] pinning GPUs via CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
fi

BASH_SETUP="$(pwd)/infra/.cml_setup.sh"
# Bind-mount mode (legacy/working): host has the repo at /home/hfunaya/git/
# e2e_calib (rsynced before submit), agent docker mounts it at /workspace,
# bash_setup just `cd /workspace && exec ...`. One-liner: agent v3 joins
# every newline with `;` which breaks multi-line bash control flow.
cat > "$BASH_SETUP" <<EOSETUP
${CUDA_LINE} && cd /workspace && exec ${ENTRY}
EOSETUP

echo "[info] submitting:"
echo "       name : $NAME"
echo "       queue: $QUEUE"
echo "       image: $DOCKER_IMAGE"
echo "       entry: $ENTRY"
echo "[info] bash_setup_script:"
sed 's/^/    /' "$BASH_SETUP"

/home/hfunaya/.pyenv/versions/3.10.4/bin/clearml-task \
  --project     "$PROJECT" \
  --name        "$NAME" \
  --queue       "$QUEUE" \
  --docker      "$DOCKER_IMAGE" \
  --docker_args "--shm-size=64g --network host -v /home/hfunaya/git/e2e_calib:/workspace -v /home/hfunaya:/home/hfunaya -v /mnt/fsx:/mnt/fsx -v /raid:/raid -v /dev/shm:/dev/shm --ipc=host -e PYTHONPATH=/workspace" \
  --docker_bash_setup_script "$BASH_SETUP" \
  --script      "$SCRIPT" \
  --skip-repo-detection \
  --skip-python-env-install \
  --force-no-requirements \
  --cwd         "." \
  --packages    ""

rm -f "$BASH_SETUP"

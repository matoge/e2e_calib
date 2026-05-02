#!/bin/bash
# infra/submit_clearml_task.sh
#
# e2e_calib の学習を ClearML の queue に投入する薄ラッパー。
#
# 前提:
#   - clearml がインストール済み (pip install clearml)
#   - ~/clearml.conf に https://clearml.budda.site の credentials
#   - agent 側 (dgx2-gpu 等) で infra/Dockerfile.train がビルド済み
#     もしくは docker hub に push 済み (今は local build 前提)
#
# 使い方:
#   # Waymo 200ep DDP (frustum ON)
#   ./infra/submit_clearml_task.sh \
#       --name wm_ddp_v721_frustum \
#       --script scripts/training/train_ps_v3_ddp.py \
#       --queue dgx2-gpu \
#       --args "--config configs/waymo/v721_frustum.yaml --epochs 200 --bs 64"
#
#   # queue を dgx1-gpu にしたい場合:
#   ./infra/submit_clearml_task.sh ... --queue dgx1-gpu
#
# 仕組み:
#   clearml-task を呼ぶ。 git の現在の commit を clone してもらい、
#   agent 側でイメージ e2e-calib-train:local で実行する。
#   accelerate launch は script 側で自前で書いてもいいが、
#   ClearML が分散を管理したい場合は --docker-args で --shm-size と
#   --gpus all を必ず渡す。

set -euo pipefail

# デフォルト値
NAME=""
PROJECT="e2e_calib"
SCRIPT=""
QUEUE="dgx2-gpu"
ARGS=""
DOCKER_IMAGE="e2e-calib-train:local"
NUM_PROCESSES=8  # GPU 数。 dgx1/2/3/4 は全部 8GPU 前提

usage() {
  cat <<EOF
Usage: $0 --name <run_name> --script <path/to/train.py> [options]

Required:
  --name     NAME      ClearML task 名 (例: wm_ddp_v721_frustum)
  --script   PATH      実行する python script (リポジトリルートからの相対)

Optional:
  --project  PROJECT   (default: e2e_calib)
  --queue    QUEUE     (default: dgx2-gpu; dgx1-gpu / dgx3-gpu / dgx4-gpu も可)
  --args     "STR"     script に渡す追加 CLI 引数 (クォートして一塊で)
  --image    IMAGE     docker image (default: e2e-calib-train:local)
  --num-gpus N         accelerate --num_processes (default: 8)
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
[[ ! -f "$SCRIPT" ]] && { echo "[err] script not found: $SCRIPT" >&2; exit 2; }

# accelerate で包む entry point。 script 側で accelerate.Accelerator() を
# 使っている前提。 torchrun を自前で書きたい場合は下を書き換え。
ENTRY="accelerate launch --num_processes=${NUM_PROCESSES} --mixed_precision=bf16 ${SCRIPT} ${ARGS}"

echo "[info] submitting ClearML task"
echo "       name    : $NAME"
echo "       project : $PROJECT"
echo "       queue   : $QUEUE"
echo "       image   : $DOCKER_IMAGE"
echo "       entry   : $ENTRY"

# clearml-task CLI
# --docker-args:
#   --shm-size=64g  DDP で NCCL shared memory をケチると落ちる
#   --gpus all      container に全 GPU 見せる
#   -v /mnt/fsx:/mnt/fsx  データセットキャッシュ
#   -v /dev/shm:/dev/shm  /dev/shm 上のキャッシュも共有
clearml-task \
  --project  "$PROJECT" \
  --name     "$NAME" \
  --queue    "$QUEUE" \
  --docker   "$DOCKER_IMAGE" \
  --docker-args "--shm-size=64g --gpus all -v /mnt/fsx:/mnt/fsx -v /dev/shm:/dev/shm --ipc=host" \
  --script   "$SCRIPT" \
  --args     $ARGS \
  --skip-task-init

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
QUEUE="dgx1-gpu"        # default dgx1-gpu (host-pip agent v3.0.0 in docker mode).
                         # dgx2-gpu / dgx3-gpu / dgx4-gpu は --queue で切替。
ARGS=""
DOCKER_IMAGE="e2e-calib-train:local"
NUM_PROCESSES=4          # GPU 数。 smoke/short run は 4 GPU、 200ep 本番は 8 に上げる
USE_LAUNCHER=1           # 1 → scripts/training/launch_ddp_ps.py を噛ませて
                         #     container 内で accelerate launch --num_processes=N を
                         #     起動する。 clearml-task は `python <script>` を直叩き
                         #     するので、 ここを 0 にすると DDP は effectively
                         #     num_processes=1 で回る (比較/デバッグ時のみ)。

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
    --no-launcher)  USE_LAUNCHER=0;     shift 1 ;;
    -h|--help)      usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

[[ -z "$NAME"   ]] && { echo "[err] --name required" >&2;   usage; }
[[ -z "$SCRIPT" ]] && { echo "[err] --script required" >&2; usage; }
# script 存在チェックは repo が手元にある場合のみ。
# clearml-task は --repo 指定があれば agent 側で clone するので、
# 家や CI からの submit で "repo 持ってないが投げる" 場合は -f チェックを skip。
if [[ -f "$SCRIPT" ]]; then
  :
elif [[ "${ALLOW_REMOTE_ONLY:-0}" == "1" ]]; then
  echo "[info] script not present locally (ALLOW_REMOTE_ONLY=1): $SCRIPT — assuming agent will clone" >&2
else
  echo "[err] script not found: $SCRIPT" >&2
  echo "      (set ALLOW_REMOTE_ONLY=1 to skip local existence check,"      >&2
  echo "       e.g. when submitting from a machine that does not have the repo cloned)" >&2
  exit 2
fi

# --- DDP launcher auto-wrap ---------------------------------------------------
# clearml-task v3 は --script をそのまま `python <script>` で叩くので、
# accelerate launch が発火しない (結果 num_processes=1)。 USE_LAUNCHER=1 なら
# scripts/training/launch_ddp_ps.py を代わりに --script に渡し、
# --args に num-gpus=N / target-script=<orig> を先頭追記する。
#
# - launcher を噛ませない方がいいケース (= 単一プロセス script / デバッグ) は
#   --no-launcher で opt-out。
# - orig script を `target-script=<path>` で渡すのは、 launcher 側でそのまま
#   accelerate launch ... <target-script> ... に渡す argparse 契約。
ORIG_SCRIPT="$SCRIPT"
LAUNCHER_PREFIX_ARGS=""
if [[ "$USE_LAUNCHER" == "1" ]]; then
  LAUNCHER_PATH="scripts/training/launch_ddp_ps.py"
  if [[ -f "$LAUNCHER_PATH" ]]; then
    :
  elif [[ "${ALLOW_REMOTE_ONLY:-0}" == "1" ]]; then
    echo "[info] launcher not present locally (ALLOW_REMOTE_ONLY=1): $LAUNCHER_PATH — assuming agent will clone" >&2
  else
    echo "[err] launcher not found: $LAUNCHER_PATH" >&2
    echo "      (disable with --no-launcher to submit $SCRIPT directly," >&2
    echo "       or set ALLOW_REMOTE_ONLY=1 if submitting without a local repo)" >&2
    exit 2
  fi
  SCRIPT="$LAUNCHER_PATH"
  LAUNCHER_PREFIX_ARGS="--num-gpus $NUM_PROCESSES --target-script $ORIG_SCRIPT"
  echo "[info] launcher wrap: $ORIG_SCRIPT → $SCRIPT (num_processes=$NUM_PROCESSES)"
fi

# PS の cache デフォルト自動注入 (ARGS の --cache が無い場合のみ)。
#   dgx1-4 すべて Lustre (/mnt/fsx) マウント済みなので全 queue 共通で
#   /mnt/fsx/tmp/hfunaya/cache/pandaset_v3_full を使う。
#   (以前 "dgx1 だけ Lustre 無し" と書いていたが誤り。2026-05-03 訂正。)
# --args 側で明示的に --cache 渡せば override される。
if [[ "$ORIG_SCRIPT" == *train_ps_v3_ddp.py* || "$ORIG_SCRIPT" == scripts/training/train_ps_v3_ddp.py ]]; then
  if [[ "$ARGS" != *"--cache"* ]]; then
    DEFAULT_PS_CACHE="/mnt/fsx/tmp/hfunaya/cache/pandaset_v3_full"
    ARGS="--cache $DEFAULT_PS_CACHE $ARGS"
    echo "[info] auto-inject: --cache $DEFAULT_PS_CACHE"
  fi
fi
# prepend launcher-specific flags (if any)
if [[ -n "$LAUNCHER_PREFIX_ARGS" ]]; then
  ARGS="$LAUNCHER_PREFIX_ARGS $ARGS"
fi

# clearml-task v3 `--args` は argparse 形式 (key=value スペース区切り)。
# なので "--config X --epochs 2 --smoke" を "config=X epochs=2 smoke=True" に変換する。
# flag-only (--smoke) は store_true 扱いで value=True を付ける。
CT_ARGS=""
tokens=( $ARGS )
i=0
while [[ $i -lt ${#tokens[@]} ]]; do
  t="${tokens[$i]}"
  if [[ "$t" == --* ]]; then
    key="${t#--}"
    next="${tokens[$((i+1))]:-}"
    if [[ -z "$next" || "$next" == --* ]]; then
      CT_ARGS="$CT_ARGS ${key}=True"
      i=$((i+1))
    else
      CT_ARGS="$CT_ARGS ${key}=${next}"
      i=$((i+2))
    fi
  else
    i=$((i+1))
  fi
done

# accelerate で包む entry point。 script 側で accelerate.Accelerator() を
# 使っている前提。 torchrun を自前で書きたい場合は下を書き換え。
ENTRY="accelerate launch --num_processes=${NUM_PROCESSES} --mixed_precision=bf16 ${SCRIPT} ${ARGS}"

echo "[info] submitting ClearML task"
echo "       name    : $NAME"
echo "       project : $PROJECT"
echo "       queue   : $QUEUE"
echo "       image   : $DOCKER_IMAGE"
echo "       entry   : $ENTRY"
echo "       ct_args : $CT_ARGS"

# clearml-task CLI
# 注意点 (2026-05-03 実地調査):
#   - `--docker_args` は underscore。 hyphen 版 (`--docker-args`) は
#     installed clearml-task では unrecognized arg になる。
#   - `--cwd .` を省くと populate.py:186 で self.cwd=None → TypeError。
#   - `--packages` が無いと repo の requirements.txt が無いケースで落ちる。
#     agent image 側で入ってる前提なら空文字でよいが、互換性のため clearml を明示。
# docker_args の中身:
#   --shm-size=64g  DDP で NCCL shared memory をケチると落ちる
#   --gpus all      container に全 GPU 見せる (accelerate 側で num_processes で制限)
#   -v /mnt/fsx:/mnt/fsx  データセットキャッシュ (dgx1-4 全て Lustre マウント済み)
#   -v /dev/shm:/dev/shm  /dev/shm 上のキャッシュも共有
#   -e CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1
#       container 内 python env をそのまま使う。 nvcr.io/nvidia/pytorch:24.02
#       base image は pip freeze すると `aiohttp @ file:///rapids/...` みたいな
#       ローカルパス参照を吐くため、 agent が新しい venv を作って再インストール
#       しようとすると Local file not found で落ちる。 image の site-packages
#       を丸ごと使う前提で SKIP_PIP_VENV_INSTALL=1 を立てる (agent 0.17.1 /
#        v3.0.0 両方サポート)。
#   -e CLEARML_AGENT_GIT_USER / GIT_PASS (optional)
#       SSH 経由の git@github.com:matoge/... clone 用の PAT を持たせたい場合。
#       既に image 側に ~/.ssh が焼いてあれば不要。
clearml-task \
  --project     "$PROJECT" \
  --name        "$NAME" \
  --queue       "$QUEUE" \
  --docker      "$DOCKER_IMAGE" \
  --docker_args "--shm-size=64g --gpus all -v /mnt/fsx:/mnt/fsx -v /dev/shm:/dev/shm --ipc=host -e CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 -e PYTHONPATH=/workspace" \
  --script      "$SCRIPT" \
  --cwd         "." \
  --packages    "clearml" \
  --args        $CT_ARGS \
  --skip-task-init

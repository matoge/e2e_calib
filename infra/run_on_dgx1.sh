#!/bin/bash
# infra/run_on_dgx1.sh
#
# One-liner wrapper to launch e2e_calib experiments on the dgx1 ClearML queue.
#
# プリセット (どれも 4 GPU / bf16 / accelerate launch):
#   ps_smoke   -- PandaSet 2 epoch / bs=16 / train_size=200 val_size=40
#                 パイプラインが通るか確認するだけの短距離走。 ~5 min。
#   ps_short   -- PandaSet 20 epoch / bs=64 — バイアス・loss 曲線の初期チェック
#   ps_100     -- PandaSet 100 epoch / bs=64 (v9_3layer_rgb_randdepth 系)
#   ps_200_frustum / ps_200_nofrustum
#              -- ablation pair: frustum on vs off, 200 ep, bs=64
#
# 使い方:
#   ./infra/run_on_dgx1.sh ps_smoke
#   ./infra/run_on_dgx1.sh ps_100 --name ps_v721_dgx1  # name 上書き
#   ./infra/run_on_dgx1.sh custom --args "--epochs 50 --no-frustum" --name ps_foo
#
# 環境:
#   - SSH 不要 (このスクリプトを dgx1 の上で動かす前提)。 laptop から投げたい
#     なら `ssh dgx1 './infra/run_on_dgx1.sh ps_smoke'` でよい。
#   - venv: /home/hfunaya/venv_clearml (clearml-task 3.0.0 / clearml 2.1.6)。
#   - agent: dgx1 host で `dgx1-gpu` queue を watch している (docker mode)。
#
# 注意:
#   docker image e2e-calib-train:local が dgx1 に存在する前提。 無ければ
#       ssh dgx1 'cd ~/git/e2e_calib && docker build -f infra/Dockerfile.train -t e2e-calib-train:local .'
#   を一度だけ走らせる。

set -euo pipefail

# --- locate repo root regardless of where this script is invoked --------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

# --- pick up the venv that has clearml-task 3.0.0 -----------------------------
# (on dgx1 the host-pip agent runs from /home/hfunaya/venv_clearml; we reuse it)
if [[ -z "${VIRTUAL_ENV:-}" && -d /home/hfunaya/venv_clearml ]]; then
  # shellcheck disable=SC1091
  source /home/hfunaya/venv_clearml/bin/activate
fi

if ! command -v clearml-task >/dev/null; then
  echo "[err] clearml-task not on PATH. Activate /home/hfunaya/venv_clearml or pip-install clearml>=1.16." >&2
  exit 3
fi

PRESET="${1:-}"
[[ -z "$PRESET" ]] && { echo "usage: $0 <preset> [--name NAME] [--args \"...\"]"; exit 2; }
shift || true

# extra --name / --args overrides
NAME_OVERRIDE=""
ARGS_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME_OVERRIDE="$2"; shift 2 ;;
    --args) ARGS_OVERRIDE="$2"; shift 2 ;;
    *) echo "unknown override: $1" >&2; exit 2 ;;
  esac
done

SCRIPT="scripts/training/train_ps_v3_ddp.py"
QUEUE="dgx1-gpu"
NUM_GPUS=4

case "$PRESET" in
  ps_smoke)
    NAME="${NAME_OVERRIDE:-ps_dgx1_smoke}"
    ARGS="${ARGS_OVERRIDE:---epochs 2 --batch-size 16 --train-size 200 --val-size 40 --why dgx1-pipeline-smoke}"
    ;;
  ps_short)
    NAME="${NAME_OVERRIDE:-ps_dgx1_short20}"
    ARGS="${ARGS_OVERRIDE:---epochs 20 --batch-size 64 --why dgx1-short-20ep}"
    ;;
  ps_100)
    NAME="${NAME_OVERRIDE:-ps_dgx1_100ep}"
    ARGS="${ARGS_OVERRIDE:---epochs 100 --batch-size 64 --why dgx1-100ep-baseline}"
    ;;
  ps_200_frustum)
    NAME="${NAME_OVERRIDE:-ps_dgx1_v721_frustum}"
    ARGS="${ARGS_OVERRIDE:---epochs 200 --batch-size 64 --why dgx1-v721-frustum-on}"
    ;;
  ps_200_nofrustum)
    NAME="${NAME_OVERRIDE:-ps_dgx1_v721b_nofrustum}"
    ARGS="${ARGS_OVERRIDE:---epochs 200 --batch-size 64 --no-frustum --why dgx1-v721b-frustum-off}"
    ;;
  custom)
    [[ -z "$NAME_OVERRIDE" ]] && { echo "[err] custom preset requires --name" >&2; exit 2; }
    [[ -z "$ARGS_OVERRIDE" ]] && { echo "[err] custom preset requires --args" >&2; exit 2; }
    NAME="$NAME_OVERRIDE"
    ARGS="$ARGS_OVERRIDE"
    ;;
  *)
    echo "[err] unknown preset: $PRESET" >&2
    echo "      known: ps_smoke ps_short ps_100 ps_200_frustum ps_200_nofrustum custom" >&2
    exit 2
    ;;
esac

echo "============================================================"
echo "preset  : $PRESET"
echo "name    : $NAME"
echo "queue   : $QUEUE"
echo "gpus    : $NUM_GPUS"
echo "args    : $ARGS"
echo "============================================================"

exec ./infra/submit_clearml_task.sh \
  --name     "$NAME" \
  --script   "$SCRIPT" \
  --queue    "$QUEUE" \
  --num-gpus "$NUM_GPUS" \
  --args     "$ARGS"

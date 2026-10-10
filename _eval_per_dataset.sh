#!/bin/bash
# 各データセットの val を別々に評価 (--eval-only: 学習なし、val 1 回)。分割・窓の設定は all5_s1_50ep と同じ。
# 使い方: ./_eval_per_dataset.sh <ckpt> <tag>
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
PY=/home/hiro/miniconda3/envs/neurad/bin/python
C=/mnt/ssd2t/work/e2e_calib/cache
CK=$1; TAG=$2
for D in pandaset_pad256 ns_850x4_6cam_pad256 waymo_v5_5cam_z2_pad256 kamikado_pad256 tss4_pad256; do
  EXTRA=""
  case $D in
    kamikado_pad256|tss4_pad256) EXTRA="--per-cache-crop-px $C/$D:512-1024";;
    ns_850x4_6cam_pad256|waymo_v5_5cam_z2_pad256) EXTRA="--frame-stride $C/$D:2";;
  esac
  $PY -u datasets/train_cnd2_ddp.py --cache $C/$D $EXTRA \
    --min-crop-px 256 --max-crop-px 512 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
    --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
    --no-with-ba --oversample 8 --cross-attn deform --ref-mode query --rep-strategy random_train --crop-pivot image \
    --resume-ckpt $CK --start-epoch 49 --epochs 50 --eval-only \
    --name eval_${TAG}_${D} --why "per-dataset val of $CK" 2>&1 | grep -E "ep050/50|train=.*val=" | sed "s#^#$TAG $D  #"
done

#!/bin/bash
# km_s1_512_from_nsps (段 1) が終わったら、その最良の重みから段 2 (BA あり)。設定は nsps_s2_pad256 と同じ (30 エポック)。
set -euo pipefail
cd "$(dirname "$0")"
while [ -e /proc/2898676 ]; do sleep 60; done
grep -q "ep020/20" experiments/km_s1_512_from_nsps/train.log || { echo "段 1 が 20 エポック終わっていない。段 2 は始めない"; exit 1; }
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py --cache $C/kamikado_pad256 \
  --min-crop-px 512 --max-crop-px 512 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --with-ba --resume-ckpt experiments/km_s1_512_from_nsps/best_model.pt --start-epoch 0 --epochs 30 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-warmup-start 0 --ba-warmup-end 10 \
  --name km_s2_512_from_nsps \
  --why "段2: km_s1_512_from_nsps (nsps_s1_pad256 → kamikado 段 1) の最良の重みから。窓は画像全体の格子 (512 px → 256、8x5 = 40 窓)、同じずれ、GN で 1 つのポーズ、InfoHead に BA の損失 (重み 0.05、0〜10 エポックで立ち上げ)。他は nsps_s2_pad256 と同じ。"

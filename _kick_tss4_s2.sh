#!/bin/bash
# TSS4 (woven vehicle 248、魚眼 3840x1952、8 シーケンス) を段 2 (BA あり) から。nsps_s2_pad256 (PandaSet+nuScenes 段 2) の重みから。
# 窓は 512 px を 256 にリサイズ、画像全体の格子 (8x4 = 32 窓)。他は nsps_s2_pad256 と同じ。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py --cache $C/tss4_pad256 \
  --min-crop-px 512 --max-crop-px 512 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --with-ba --resume-ckpt experiments/nsps_s2_pad256/best_model.pt --start-epoch 0 --epochs 30 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-warmup-start 0 --ba-warmup-end 10 \
  --name tss4_s2_from_nsps \
  --why "TSS4 (8 シーケンス、train 350 / val 50、POSLV を合成したキャッシュ tss4_pad256) を段 2 (BA あり) から。nsps_s2_pad256 の重みから。窓は 512 px → 256、画像全体の格子。"

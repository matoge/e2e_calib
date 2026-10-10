#!/bin/bash
# Waymo 5 カメラ (build_waymo_v4、深度 2 m 未満は除外) だけで段 1 (タイル、BA なし)。nsps_s1_pad256 の重みから。
# 窓は 256〜512 px をランダムに切り出して 256 にリサイズ、画像の画素から一様に置く (--crop-pivot image)。
set -euo pipefail
while ! grep -q DONE /mnt/ssd2t/work/e2e_calib/build_waymo_v4_5cam_z2_pad256.log 2>/dev/null; do sleep 30; done
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py --cache $C/waymo_v4_5cam_z2_pad256 \
  --min-crop-px 256 --max-crop-px 512 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 16 --epochs 12 --eval-every 3 \
  --cross-attn deform --ref-mode query --rep-strategy random_train --crop-pivot image \
  --resume-ckpt experiments/nsps_s1_pad256/best_model.pt --start-epoch 0 \
  --name waymo_z2_s1_from_nsps \
  --why "Waymo 5 カメラだけ (build_waymo_v4、深度 2 m 未満を除外、train 約 1 万) で段 1 (タイル、BA なし)。nsps_s1_pad256 の重みから。窓は 256〜512 px をランダムに切り出して 256 に (256 だと小さすぎるかもしれないので)。1 フレーム 16 窓、12 エポック。"

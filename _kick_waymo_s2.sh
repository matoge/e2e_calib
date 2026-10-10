#!/bin/bash
# Waymo 5 カメラ (build_waymo_v4、train 9985 / val 995) だけで BA あり。nsps_s2_pad256 (PandaSet+nuScenes 段 2) の重みから。
# 4 エポックだと学習率の立ち上げの途中で終わったので (all5_s2)、12 エポック。
set -euo pipefail
while ! grep -q DONE /mnt/ssd2t/work/e2e_calib/build_waymo_v4_5cam_z2_pad256.log 2>/dev/null; do sleep 30; done
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py --cache $C/waymo_v4_5cam_z2_pad256 \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --epochs 12 --eval-every 3 \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --with-ba --resume-ckpt experiments/nsps_s2_pad256/best_model.pt --start-epoch 0 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-warmup-start 0 --ba-warmup-end 2 \
  --name waymo_z2_s2_from_nsps \
  --why "Waymo 5 カメラだけ (build_waymo_v4: 実 LiDAR 点を自前のピンホールで投影、深度 2 m 未満は除外、train 9985 / val 995) で BA あり。nsps_s2_pad256 の重みから。all5_s2 が収束しなかったので、Waymo 単体で学習できるかを見る。12 エポック。"

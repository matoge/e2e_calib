#!/bin/bash
# ps_s1_own24_rt (段1、20 エポック) が終わったら、その最良の重みから段 2 (BA あり) を始める。
set -euo pipefail
cd "$(dirname "$0")"
L=experiments/ps_s1_own24_rt/train.log
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
python -u datasets/train_cnd2_ddp.py --cache /mnt/ssd2t/work/e2e_calib/cache/pandaset_v3_full \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --with-ba --resume-ckpt experiments/ps_s1_own24_rt/best_model.pt --start-epoch 0 --epochs 30 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-warmup-start 0 --ba-warmup-end 10 \
  --name ps_s2_own24_rt \
  --why "段2: ps_s1_own24_rt (段1、Deformable 参照点=自分の uv、局所エンコーダ 自分のセル 24 点、学習時クエリ ランダム) の最良の重みから。40 窓の格子に同じずれ、GN で 1 つのポーズ、InfoHead に BA の損失 (重み 0.05、10 エポックで立ち上げ)。"

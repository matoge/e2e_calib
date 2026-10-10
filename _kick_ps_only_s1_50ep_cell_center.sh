#!/bin/bash
# PandaSet だけで段 1 (タイル、BA なし)、50 エポック。スクラッチ (旧モデルとの比較のため重みは使わない)。all5_s1_resume28 と同じ設定で、PandaSet 単独の性能と比べるため。
# 窓は画像の画素から一様、256 px 固定でリサイズなし (旧モデル ps_old_simple_s256_* と同じ)。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py \
  --cache $C/pandaset_pad256 \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 8 --epochs 50 --eval-every 10 \
  --cross-attn deform --ref-mode query --rep-strategy cell_center --crop-pivot image \
  --start-epoch 0 \
  --name ps_only_s1_50ep_cell_center \
  --why "rep-strategy=cell_center (ps_only_s1_50ep の random_train との比較)。PandaSet 単独 (2720) を段 1、50 エポック、スクラッチ。all5_s1_resume28 と同じ窓設定 (256 px 固定、1 フレーム 8 窓)。混ぜたときの PandaSet の性能と比べるため。"

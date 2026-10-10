#!/bin/bash
# PandaSet だけで段 1 (タイル、BA なし)、50 エポック。nsps_s1_pad256 の重みから。all5_s1_resume28 と同じ設定で、PandaSet 単独の性能と比べるため。
# 窓は画像の画素から一様、大きさはランダム: ピンホール 256〜512 px、魚眼の 4K (kamikado / TSS4) は同じ画角の 512〜1024 px、全部 256 にリサイズ。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py \
  --cache $C/pandaset_pad256 \
  --min-crop-px 256 --max-crop-px 512 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 8 --epochs 50 --eval-every 10 \
  --cross-attn deform --ref-mode query --rep-strategy random_train --crop-pivot image \
  --resume-ckpt experiments/nsps_s1_pad256/best_model.pt --start-epoch 0 \
  --name ps_only_s1_50ep \
  --why "PandaSet 単独 (2720) を段 1、50 エポック、nsps_s1_pad256 から。all5_s1_resume28 と同じ窓設定 (256〜512 px、1 フレーム 8 窓)。混ぜたときの PandaSet の性能と比べるため。"

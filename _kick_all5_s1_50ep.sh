#!/bin/bash
# 5 データセットを一緒に段 1 (タイル、BA なし)、50 エポック。nsps_s1_pad256 の重みから。
# 窓は画像の画素から一様、大きさはランダム: ピンホール 256〜512 px、魚眼の 4K (kamikado / TSS4) は同じ画角の 512〜1024 px、全部 256 にリサイズ。
# nuScenes 6 カメラと Waymo は 1/2 (--frame-stride)。1 フレーム 8 窓 (毎エポック引き直す)。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py \
  --cache $C/pandaset_pad256,$C/ns_850x4_6cam_pad256,$C/waymo_v4_5cam_z2_pad256,$C/kamikado_pad256,$C/tss4_pad256 \
  --per-cache-crop-px $C/kamikado_pad256:512-1024,$C/tss4_pad256:512-1024 \
  --frame-stride $C/ns_850x4_6cam_pad256:2,$C/waymo_v4_5cam_z2_pad256:2 \
  --min-crop-px 256 --max-crop-px 512 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 8 --epochs 50 --eval-every 10 \
  --cross-attn deform --ref-mode query --rep-strategy random_train --crop-pivot image \
  --resume-ckpt experiments/nsps_s1_pad256/best_model.pt --start-epoch 0 \
  --name all5_s1_50ep \
  --why "PandaSet (2720) + nuScenes 6 カメラ (1/2、9177) + Waymo 5 カメラ (深度 2 m 未満除外、1/2、4993) + kamikado (683) + TSS4 (350) を一緒に段 1 (タイル、BA なし)、50 エポック。nsps_s1_pad256 の重みから。窓の大きさはランダム (ピンホール 256〜512 px、魚眼 4K は 512〜1024 px)、256 にリサイズ、画像の画素から一様に置く。1 フレーム 8 窓。"

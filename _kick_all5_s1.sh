#!/bin/bash
# nuScenes 6 カメラ + Waymo 5 カメラ + PandaSet + kamikado + TSS4 を一緒に段 1 (タイル、BA なし)。nsps_s1_pad256 の重みから。
# 窓は画像の画素から一様 (--crop-pivot image)。ピンホールは 256 px、魚眼 (kamikado / TSS4) は 512 px を 256 に。
set -euo pipefail
cd "$(dirname "$0")"
while ! grep -q DONE /mnt/ssd2t/work/e2e_calib/build_waymo_v4_5cam_pad256.log; do sleep 30; done
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py \
  --cache $C/pandaset_pad256,$C/ns_850x4_6cam_pad256,$C/waymo_v4_5cam_pad256,$C/kamikado_pad256,$C/tss4_pad256 \
  --per-cache-crop-px $C/kamikado_pad256:512,$C/tss4_pad256:512 \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 16 --epochs 8 --eval-every 2 \
  --cross-attn deform --ref-mode query --rep-strategy random_train --crop-pivot image \
  --resume-ckpt experiments/nsps_s1_pad256/best_model.pt --start-epoch 0 \
  --name all5_s1_from_nsps \
  --why "nuScenes 6 カメラ (18354) + Waymo 5 カメラ (build_waymo_v4、約 1 万) + PandaSet (2720) + kamikado (683) + TSS4 (350) を一緒に段 1 (タイル、BA なし)。nsps_s1_pad256 の重みから。窓は画像から一様に置く。1 フレーム 16 窓、8 エポック (朝までに終わる長さ)。"

#!/bin/bash
# nuScenes 6 カメラ + Waymo 5 カメラ + PandaSet + kamikado + TSS4 を一緒に BA あり。
# kmtss4_s2_from_nsps/last_model.pt (nsps_s2_pad256 → kamikado+TSS4 BA、魚眼の GN 修正後) から。
# 窓は画像全体の格子 (ピンホール 256 px、魚眼 512 px を 256 に)、1 フレーム 40 窓 (足りない分は複製で w_active=0)。
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
  --clearml --clearml-project e2e_calib/calib --epochs 4 --eval-every 4 \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --with-ba --resume-ckpt experiments/kmtss4_s2_from_nsps/last_model.pt --start-epoch 0 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-warmup-start 0 --ba-warmup-end 1 \
  --name all5_s2 \
  --why "nuScenes 6 カメラ (18354) + Waymo 5 カメラ (build_waymo_v4、約 1 万) + PandaSet (2720) + kamikado (683) + TSS4 (350) を一緒に BA あり。kmtss4_s2_from_nsps/last_model.pt から (既に BA で学習済みなので立ち上げは 1 エポック)。4 エポック (朝までに終わる長さ)、val は最後だけ。"

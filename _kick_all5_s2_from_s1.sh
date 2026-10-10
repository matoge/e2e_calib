#!/bin/bash
# 5 データセットを一緒に段 2 (BA あり)。all5_s1_resume28/last_model.pt (段 1、BA なし) から。
# 窓は画像全体の格子、固定の大きさ: ピンホール 256 px、魚眼 4K (kamikado / TSS4) は 512 px、全部 256 にリサイズ
# (段 1 のランダム範囲 256〜512 / 512〜1024 の下端)。nuScenes 6 カメラと Waymo は 1/2。
# 10 エポック (LR は 5 エポック warmup → cosine)。BA の重みは ep1 は 0、ep1→3 で 0.05 まで線形。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py \
  --cache $C/pandaset_pad256,$C/ns_850x4_6cam_pad256,$C/waymo_v5_5cam_z2_pad256,$C/kamikado_pad256,$C/tss4_pad256 \
  --per-cache-crop-px $C/kamikado_pad256:512,$C/tss4_pad256:512 \
  --frame-stride $C/ns_850x4_6cam_pad256:2,$C/waymo_v5_5cam_z2_pad256:2 \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --epochs 10 --eval-every 5 \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --with-ba --resume-ckpt experiments/all5_s1_resume28/last_model.pt --start-epoch 0 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-warmup-start 1 --ba-warmup-end 3 \
  --name all5_s2_from_s1 \
  --why "PandaSet + nuScenes 6 カメラ (1/2) + Waymo v5 (公式投影、1/2) + kamikado + TSS4 を段 2 (BA あり)。all5_s1_resume28 (段 1) から。窓は固定 256 px (魚眼 512 px) の格子。10 エポック、BA 重みは ep1→3 で 0→0.05。"

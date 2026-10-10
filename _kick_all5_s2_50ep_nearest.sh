#!/bin/bash
# 5 データセットを段 2 (BA あり) で 50 エポック。all5_s2_from_s1/last_model.pt (段 2 を 10 エポック) から続ける。
# クエリ: nearest_cam (セルで一番カメラに近い点)。1 セル 24 点、近傍は own (どちらも train_cnd2_ddp の既定値を明示)。
# 窓は格子、ピンホール 256 px、魚眼 4K (kamikado / TSS4) 512 px、256 にリサイズ。nuScenes 6 カメラと Waymo v5 は 1/2。
# BA で学習済みの重みなので BA の重み 0.05 を 1 エポック目から (ramp なし)。LR は 5 エポック warmup → cosine。
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
  --clearml --clearml-project e2e_calib/calib --epochs 50 --eval-every 5 \
  --cross-attn deform --ref-mode query --rep-strategy nearest_cam --k-per-cell 24 --frustum-nb own \
  --with-ba --resume-ckpt experiments/all5_s2_from_s1/last_model.pt --start-epoch 0 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-warmup-start 0 --ba-warmup-end 0 \
  --name all5_s2_50ep_nearest \
  --why "5 データセット (PandaSet + nuScenes 6 カメラ 1/2 + Waymo v5 1/2 + kamikado + TSS4) を段 2 (BA) で 50 エポック。all5_s2_from_s1/last_model.pt から。クエリ nearest_cam、1 セル 24 点、近傍 own。窓は格子 256 px (魚眼 512 px)。BA 重み 0.05 を最初から。"

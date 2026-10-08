#!/bin/bash
# kamikado + TSS4 (手で合わせたキャリブ) を段 2 (BA あり) から。nsps_s2_pad256 の重みから。
# GN は魚眼を KB で解く (41ee556)。窓は 512 px → 256、画像全体の格子 (kamikado 40 窓、TSS4 32 窓を複製で 40 に)。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py --cache $C/kamikado_pad256,$C/tss4_pad256 \
  --min-crop-px 512 --max-crop-px 512 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --with-ba --resume-ckpt experiments/nsps_s2_pad256/best_model.pt --start-epoch 0 --epochs 30 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-warmup-start 0 --ba-warmup-end 10 \
  --name kmtss4_s2_from_nsps \
  --why "kamikado (5 シーン 683) + TSS4 (7 シーケンス 350、手で合わせたキャリブで作り直したキャッシュ) を段 2 (BA あり) から。nsps_s2_pad256 の重みから。GN は魚眼を Kannala-Brandt で解く (41ee556 以前はピンホールで、魚眼の正解の残差で 13〜18 px 残っていた)。"

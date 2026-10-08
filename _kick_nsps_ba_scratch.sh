#!/bin/bash
# with-BA で最初から学習する (段 1 を飛ばす)。キャッシュと設定は nsps_s2_pad256 と同じ、エポックは段 1 + 段 2 の 50。
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py --cache $C/ns_850x4_pad256,$C/pandaset_pad256 \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --with-ba --epochs 50 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-warmup-start 0 --ba-warmup-end 10 \
  --name nsps_ba_scratch_pad256 \
  --why "with-BA で最初から (段 1 なし)。学習でも推論と同じ画像全体の格子 (nuScenes 28 窓 + PandaSet 40 窓、足りない分は複製で w_active=0)。BA の損失の重みは 0.05、0〜10 エポックで立ち上げ。nsps_s2_pad256 (段 1 20 エポック → 段 2 30 エポック) と比べる。"

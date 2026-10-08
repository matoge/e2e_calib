#!/bin/bash
# kamikado + TSS4 を一緒に段 1 (タイル、BA なし)。nsps_s1_pad256 の重みから。窓は 512 px → 256、画像から一様に置く (--crop-pivot image)。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py --cache $C/kamikado_pad256,$C/tss4_pad256 \
  --min-crop-px 512 --max-crop-px 512 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 40 --epochs 20 \
  --cross-attn deform --ref-mode query --rep-strategy random_train --crop-pivot image \
  --resume-ckpt experiments/nsps_s1_pad256/best_model.pt --start-epoch 0 \
  --name kmtss4_s1_from_nsps \
  --why "kamikado (5 シーン 683 フレーム) + TSS4 (7 シーケンス 350 フレーム) を一緒に段 1 (タイル、BA なし)。nsps_s1_pad256 の重みから。窓は 512 px → 256、画像の画素から一様に置き点が足りなければ引き直す (--crop-pivot image)。val は kamikado 1 シーン + TSS4 1 シーケンス。"

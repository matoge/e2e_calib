#!/bin/bash
# kamikado (FCM 魚眼 3840x2160、6 シーン) だけで段 1 (BA なし)。窓は 512 px を切り出して 256 にリサイズ。
# 他は ps_s1_own24_rt と同じ。キャッシュは build_kamikado_full_lmdb.py (intensity /255、画像の外 256 px まで点)。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py --cache $C/kamikado_pad256 \
  --min-crop-px 512 --max-crop-px 512 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 40 --epochs 20 \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --name km_s1_512 \
  --why "kamikado だけ (6 シーン、train 683 / val 98 フレーム、val は d005_3000_3020 の 1 シーン)。3840x2160 魚眼なので窓は 512 px を切り出して 256 にリサイズ (セル 1 つ = 元画像 32 px)。段 1 (BA なし)、他は ps_s1_own24_rt と同じ (Deformable 参照点=自分の uv、自分のセル 24 点、学習時クエリ ランダム)。"

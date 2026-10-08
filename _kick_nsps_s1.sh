#!/bin/bash
# nuScenes (ns_850x4_pad256、CAM_FRONT 850 シーン) + PandaSet (pandaset_pad256) を混ぜた段 1 (BA なし)。設定は ps_s1_own24_rt と同じ。
# キャッシュは _build_pad256_caches.sh (intensity を [0,1] に正規化、画像の外 256 px まで点を残す)。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py --cache $C/ns_850x4_pad256,$C/pandaset_pad256 \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 40 --epochs 20 \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --name nsps_s1_pad256 \
  --why "nuScenes ns_850x4_pad256 (850 シーン CAM_FRONT 1600x900、train 3059 / val 340) + PandaSet pandaset_pad256 (39 シーン 1920x1080、2720 / 400)。前の run から変えたこと: クエリの深度をずらしたポーズで測る (GT の混入の修正)、intensity を 0〜255→[0,1] に正規化 (以前は nuScenes が全点 0、PandaSet は clip で 94〜98% が 1)、キャッシュに画像の外 256 px までの点、候補点の前絞りを GT ±64 px → ±256 px・z>0.1、点の損失から複製の窓を除外、GN の ρ の tanh 二重掛けを修正。他は ps_s1_own24_rt と同じ (256 px、Deformable 参照点=自分の uv、自分のセル 24 点、学習時クエリ ランダム)。"

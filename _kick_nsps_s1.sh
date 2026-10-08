#!/bin/bash
# nuScenes (ns_850x4、CAM_FRONT 850 シーン) + PandaSet (pandaset_v3_full) を混ぜた段 1 (BA なし)。設定は ps_s1_own24_rt と同じ。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py --cache $C/ns_850x4,$C/pandaset_v3_full \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 40 --epochs 20 \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --name nsps_s1_dfix \
  --why "nuScenes ns_850x4 (850 シーン CAM_FRONT 1600x900、train 3059 / val 340 フレーム) + PandaSet pandaset_v3_full (39 シーン 1920x1080、2720 / 400) を混ぜる。クエリの深度をずらしたポーズで測る修正 (以前は GT のカメラからの距離で t·p̂ が読めた) の後。他は ps_s1_own24_rt と同じ (256 px、Deformable 参照点=自分の uv、自分のセル 24 点、学習時クエリ ランダム)。"

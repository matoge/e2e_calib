#!/bin/bash
# PandaSet を GT 漏れ無しで二段学習する。段1 が終わると段2 がその重みから続く。
#
# 漏れ: build_crop のクエリ代表点のセル割当てが 2026-05-31 から GT 投影で、
# 正解位置がセル中心に一番近い点をクエリに選んでいた。同じ ckpt でセル割当てを
# 摂動後の投影に替えると rot 誤差 0.0049 -> 0.2268 deg (注入 0.2292 deg)。
# 学習時の val 0.0056 deg はこの漏れで出ていた。今回からセル割当ては uv_off_c。
#
#   段1  --no-with-ba  窓ごとに別の δ、点ごとの gaussian2d_nll のみ
#   段2  --with-ba     格子 40 窓に同じ δ、GN で融合、BA loss を InfoHead へ
set -euo pipefail
cd "$(dirname "$0")"
PS=/mnt/ssd2t/work/e2e_calib/cache/pandaset_v3_full
COMMON="--cache $PS --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16
        --batch-size 4 --workers 6 --val-fraction 0.1 --scene-split
        --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 --eval-every 10
        --clearml --clearml-project e2e_calib/calib"
export PYTHONPATH=$PWD

python -u datasets/train_cnd2_ddp.py $COMMON --name ps_s1_noleak --no-with-ba \
  --oversample 40 --epochs 60 \
  --why "段1 (GT 漏れ修正後)。タイルのみ、窓ごとに別の δ、点ごとの NLL のみ。スクラッチ。"

python -u datasets/train_cnd2_ddp.py $COMMON --name ps_s2_noleak --with-ba \
  --resume-ckpt experiments/ps_s1_noleak/best_model.pt --start-epoch 0 --epochs 60 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll \
  --ba-warmup-start 0 --ba-warmup-end 10 \
  --why "段2 (GT 漏れ修正後)。ps_s1_noleak から resume。格子 40 窓融合 + InfoHead。"

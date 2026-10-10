#!/bin/bash
# 古いシンプルなモデル (CalibNetDepth: ConvNeXt + 点 MLP + FrustumLocalEncoder + sl ブロック 4 段、
# ブロックごとに Δuv を出して位置を更新)。RoPE・info head・frame pose なし。
# PandaSet だけ、シーン分割 (キャッシュの meta、val のシーンは学習に入らない)、窓は 256 px 固定でリサイズなし。
# クエリ: cell_center (cell_center = セル中心に一番近い点 (5 月と同じ)、nearest_cam = 一番カメラに近い点)。50 エポック、1 フレーム 8 窓。スクラッチ (この形の重みはない)。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
/home/hiro/miniconda3/envs/neurad/bin/python -u scripts/training/train_ps_v3_ddp.py \
  --name ps_old_simple_s256_cell_center --cache /mnt/ssd2t/work/e2e_calib/cache/pandaset_pad256 \
  --n-layers 4 --convnext --deform-mode sl --img-size 256 --min-crop-px 256 --max-crop-px 256 \
  --rot-deg 0.5 --t-m 0.2 --epochs 50 --oversample 8 --batch-size 32 --workers 6 \
  --scene-split --rep-strategy cell_center --no-ba-eval --clearml

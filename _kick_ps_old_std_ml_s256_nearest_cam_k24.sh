#!/bin/bash
# 古いシンプルなモデル (CalibNetDepth: ConvNeXt + 点 MLP + FrustumLocalEncoder + sl ブロック 4 段、
# ブロックごとに Δuv を出して位置を更新)。RoPE・info head・frame pose なし。
# PandaSet だけ、シーン分割 (キャッシュの meta、val のシーンは学習に入らない)、窓は 256 px 固定でリサイズなし。
# 普通の Deformable DETR デコーダの形: ml (各層が粗・細の両方)、層ごとに別の重み、参照点を毎層更新 (detach なし)、σ は各層自身、全層に損失 (deep supervision)。
# 学習の損失は 4 層の NLL の平均。ログの nll と mse は最後の層の値 (他の run と比べられる)。
# クエリ: nearest_cam (一番カメラに近い点)。1 セル 24 点。ps_old_simple_s256_nearest_cam_k24 との違いは ml と std_decoder。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
/home/hiro/miniconda3/envs/neurad/bin/python -u scripts/training/train_ps_v3_ddp.py \
  --name ps_old_std_ml_s256_nearest_cam_k24 --cache /mnt/ssd2t/work/e2e_calib/cache/pandaset_pad256 \
  --n-layers 4 --convnext --deform-mode ml --img-size 256 --min-crop-px 256 --max-crop-px 256 \
  --rot-deg 0.5 --t-m 0.2 --epochs 50 --oversample 8 --batch-size 32 --workers 6 \
  --scene-split --rep-strategy nearest_cam --std-decoder --k-per-cell 24 --no-ba-eval --clearml

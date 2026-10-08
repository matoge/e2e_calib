#!/bin/bash
# nsps_s1_pad256 (段 1) が終わったら、その最良の重みから段 2 (BA あり) を始める。設定は ps_s2_own24_rt と同じ。
set -euo pipefail
cd "$(dirname "$0")"
while [ -e /proc/710375 ]; do sleep 60; done
grep -q "ep020/20" experiments/nsps_s1_pad256/train.log || { echo "段 1 が 20 エポック終わっていない。段 2 は始めない"; exit 1; }
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
C=/mnt/ssd2t/work/e2e_calib/cache
python -u datasets/train_cnd2_ddp.py --cache $C/ns_850x4_pad256,$C/pandaset_pad256 \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib \
  --cross-attn deform --ref-mode query --rep-strategy random_train \
  --with-ba --resume-ckpt experiments/nsps_s1_pad256/best_model.pt --start-epoch 0 --epochs 30 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll --ba-warmup-start 0 --ba-warmup-end 10 \
  --name nsps_s2_pad256 \
  --why "段2: nsps_s1_pad256 の最良の重みから。nuScenes 28 窓 + PandaSet 40 窓の格子 (足りない分は複製で w_active=0、BA と点の損失から除外)、同じずれ、GN で 1 つのポーズ、InfoHead に BA の損失 (重み 0.05、10 エポックで立ち上げ)。"

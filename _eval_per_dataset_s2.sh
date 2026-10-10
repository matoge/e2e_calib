#!/bin/bash
# 段 2 (BA あり) のチェックポイントを、データセットごとの val で別々に評価する (--eval-only: 学習なし、val 1 回)。
# 窓は all5_s2_50ep_nearest と同じ格子 (ピンホール 256 px、魚眼 512 px)、クエリ nearest_cam、1 セル 24 点、own。
# --dump-pose で val の 1 フレームごとの (δ_pred, δ_gt, H) を experiments/eval_s2_<tag>_<cache>/pose_dump_ep050.pt に残す。
# 使い方: ./_eval_per_dataset_s2.sh <ckpt> <tag>
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
PY=/home/hiro/miniconda3/envs/neurad/bin/python
C=/mnt/ssd2t/work/e2e_calib/cache
CK=$1; TAG=$2
for D in waymo_v5_5cam_z2_pad256 pandaset_pad256 ns_850x4_6cam_pad256 kamikado_pad256 tss4_pad256; do
  EXTRA=""
  case $D in
    kamikado_pad256|tss4_pad256) EXTRA="--per-cache-crop-px $C/$D:512";;
    ns_850x4_6cam_pad256|waymo_v5_5cam_z2_pad256) EXTRA="--frame-stride $C/$D:2";;
  esac
  $PY -u datasets/train_cnd2_ddp.py --cache $C/$D $EXTRA \
    --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 2 --workers 4 \
    --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
    --cross-attn deform --ref-mode query --rep-strategy nearest_cam --k-per-cell 24 --frustum-nb own \
    --with-ba --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll \
    --resume-ckpt $CK --start-epoch 49 --epochs 50 --eval-only --dump-pose \
    --name eval_s2_${TAG}_${D} --why "per-dataset BA val of $CK" 2>&1 | grep -E "ep050/50|pose dump|Traceback|Error" | sed "s#^#$TAG $D  #"
done

#!/bin/bash
# 段1 で、クロスアテンションの違いだけを比べる (ps_s1_refq と同じ条件、10 エポック)。
#   A ps_s1_daq  : MSDeformAttn、参照点 = 投影したクエリの (u,v) そのもの (ref_proj なし)
#   B ps_s1_full : nn.MultiheadAttention で KV 全体を見る (参照点なし)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD
PS=/mnt/ssd2t/work/e2e_calib/cache/pandaset_v3_full
COMMON="--cache $PS --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16
        --batch-size 4 --workers 6 --val-fraction 0.1 --scene-split
        --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 --eval-every 1
        --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 40 --epochs 10"
python -u datasets/train_cnd2_ddp.py $COMMON --name ps_s1_daq --cross-attn deform --ref-mode query \
  --why "段1 A: DA、参照点=投影位置そのもの (ref_proj なし)。refq/full と同条件 10ep。"
python -u datasets/train_cnd2_ddp.py $COMMON --name ps_s1_full --cross-attn full \
  --why "段1 B: 全体クロスアテンション (参照点なし)。refq/daq と同条件 10ep。"

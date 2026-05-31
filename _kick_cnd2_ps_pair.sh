#!/bin/bash
# Kick CalibNet2 cross-frame (pair_mode) training on PandaSet FULL LMDB.
# PoseEmb at stack midpoint (= state-space disentanglement).
# Target: reproduce 旧 cross_frame.py の PS 0.72 px、できれば improve.
set -euo pipefail
NAME="cnd2_ps_pair_$(date +%m%d_%H%M)"

WHY=$(cat <<'WHY_END'
CalibNet2 cross-frame kick #1 — state-space disentanglement.

Goal: reproduce 旧 cross_frame.py の PandaSet val 0.72 px sub-pixel result. できれば improve.

Architecture diff vs old cross_frame.py:
  - PoseEmb は block stack の途中で 1 回 RoPE(R_AB) 作用 (旧は入口 additive)
  - frame token = world / pose_emb = ego motion で状態空間が直交分解される
  - block 1 種類で stack、KV always own (A 段は KV_A、B 段は KV_B)
  - 重み完全共有 (calib R=I は cross-frame の特殊ケース)

PS の GT pose は systematic bias が小さいことを旧 cross_frame.py の 0.72 px 結果が示唆 — LLN で吸収可能な regime。CalibNet2 でも同等以上が出るはず。HAT 摂動 σ_ypr=1.0° / σ_t=0.2m で評価。

50 epoch、img_size=128、grid_n=16、batch_size=32 (per-rank、global=256 on 8 GPU)、oversample=2、AdamW lr=1e-3。

期待値:
- ep1 SPS が見えること
- val_mse がエポック追って 9 px → 5 px → ... 1 px 級に降下
- 50ep 終盤で sub-2px、できれば sub-1px
WHY_END
)

infra/submit_clearml_task.sh \
  --name        "$NAME" \
  --script      datasets/train_cnd2_ddp.py \
  --queue       dgx2 \
  --num-gpus    8 \
  --image       e2e-calib-train:np2 \
  --project     e2e_calib/cross-frame \
  --args "--cache /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full --pair-mode --pair-stride 10 --epochs 50 --batch-size 32 --img-size 128 --grid-n 16 --oversample 2 --workers 8 --val-fraction 0.1 --n-iter 4 --rot-deg 1.0 --t-m 0.20 --use-info-head --clearml --clearml-project e2e_calib/cross-frame --why \"$WHY\""

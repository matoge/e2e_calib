#!/bin/bash
# kick #3 SMOKE: per-layer KV pyramid + layer-specific weights (A/B共有),
# kick3 schedule = [coarse, coarse, fine, super_fine] with n_points 4/4/4/8.
# 10 epochs on dgx3 (16 GPUs).
set -euo pipefail
NAME="cnd2_ps_pair_kick3_smoke10_$(date +%m%d_%H%M)"

WHY=$(cat <<'WHY_END'
CalibNet2 cross-frame kick #3 — per-layer KV pyramid (10ep smoke).

Diff vs kick #2 (oversample=16 / n_iter=4 / shared block / fixed [coarse, fine, lidar] KV):
  - n_iter=4 still, but 4 LAYER-SPECIFIC blocks (no longer shared across iters)
  - A/B共有 is preserved: same ModuleList replayed on KV_A and KV_B
  - Per-layer KV pyramid (kick3 schedule):
      L0: image=coarse(8x8)  + lidar(16x16)  n_points=4
      L1: image=coarse(8x8)  + lidar(16x16)  n_points=4
      L2: image=fine(16x16)  + lidar(16x16)  n_points=4
      L3: image=super_fine(32x32 = stem-level) + lidar(16x16)  n_points=8
  - Goes coarse → fine → super-fine to give the last layer a sub-pixel-resolution
    image map for final refinement; n_points doubled at the last layer to spread
    DA samples wider on the higher-resolution map.

10ep / oversample=8 (vs kick #2 50ep oversample=16): ~1/10 of kick #2 compute.
Goal: confirm the new model trains stably and beats kick #2's ep10 va_mse.

Settings:
  * PandaSet FULL LMDB / pair-mode / pair-stride=10 (bidirectional ±[1..10])
  * img_size=128, grid_n=16, n_heads=4
  * HAT 摂動 σ_ypr=1.0° / σ_t=0.20m
  * batch_size per-rank=32, global=512 on 16 GPU
  * oversample=8 / epochs=10 / AdamW lr=1e-3 cosine→1e-6
  * use_info_head=True (Gaussian NLL with σ²)

期待値 (vs kick #2 ep10 va_mse=3.12):
  * ep10 va_mse < 2.5 px なら採用 (kick #3 50ep 本番化)
  * 軌道が悪ければ 10ep で打ち切り、別 schedule (例: 全層 fine) を試す
WHY_END
)

infra/submit_clearml_task.sh \
  --name        "$NAME" \
  --script      datasets/train_cnd2_ddp.py \
  --queue       dgx3 \
  --num-gpus    16 \
  --image       e2e-calib-train:np2 \
  --project     e2e_calib/cross-frame \
  --args "--cache /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full --pair-mode --pair-stride 10 --epochs 10 --batch-size 32 --img-size 128 --grid-n 16 --oversample 8 --workers 8 --val-fraction 0.1 --n-iter 4 --rot-deg 1.0 --t-m 0.20 --use-info-head --kv-schedule kick3 --clearml --clearml-project e2e_calib/cross-frame --why \"$WHY\""

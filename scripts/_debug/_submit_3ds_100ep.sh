#!/bin/bash
# 3-DS run: woven + kamikado + waymo, 100ep, img256/grid32/pe, ±1.5°/±0.2m
# oversample 4 (waymo auto-overridden to 1 in train script).
# 12 GPU on dgx2 (CUDA_VISIBLE_DEVICES=4..15).
set -euo pipefail
cd /home/hfunaya/git/e2e_calib

NAME="km_wv_wm_15deg_20cm_img256_grid32_pe_100ep_dgx2_12gpu"
CACHES="/home/hfunaya/cache_v4/woven_v3_tile,/home/hfunaya/cache_v4/kamikado_v3_tiled,/home/hfunaya/cache_v4/waymo_v3_tiled_i"

CUDA_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15 \
./infra/submit_clearml_task.sh \
  --name   "$NAME" \
  --script scripts/training/train_ps_v3_ddp.py \
  --queue  dgx2 \
  --num-gpus 12 \
  --image  e2e-calib-train:np2 \
  --args "--cache $CACHES \
          --img-size 256 --grid-n 32 --use-pose-emb \
          --rot-deg 1.5 --t-m 0.2 \
          --oversample 4 \
          --epochs 100 --batch-size 32 --workers 8 \
          --min-crop-px 128 --max-crop-px 512 \
          --convnext --deform-mode sl \
          --n-layers 4 \
          --ba-eval-start-ep 1 --ba-eval-every 10 --ba-eval-n-seeds 4 --ba-eval-n-inst 200 \
          --clearml \
          --why '3-DS フル mix 100ep run。動機: km-only / 50ep / img128 では BA ω 残差 0.45 px (200×512) が下げ止まり、scene 多様性不足 + img/grid 解像度不足 + 学習量不足の 3 因子のうちどれが効くか分離できない。本 run は (a) woven+kami+waymo の 3-DS ConcatDataset で scene 多様性を 10× 以上に増やし、(b) img_size=256 + grid_n=32 で frustum セルを fx 換算 sub-px 帯まで細かくし、(c) 100ep 長尺で σ-head が cosine warmup 後にどこまで sharpening するかを見る。oversample は woven=4 / kami=4 / waymo=1 (waymo 5M tiles が支配的になりすぎないよう train script 側で auto-override)。摂動は train↔BA eval 同分布 ±1.5°/±0.20m。BA eval: kamikado fisheye idx=17、3-level (±0.5°,±0.05m / ±1.0°,±0.10m / ±1.5°,±0.20m)、K=4 seeds × n_inst=200、cs=256 を 4 象限に切って 800 tiles で shared GN (2026-05-22 主結果と同規模)、ep1 (sanity) + 10 ep ごと + 最終 ep に scalar+overlay PNG を ClearML に upload (毎 ep だと 800-tile × 3-level × 4-seed = 9600 forward pass で重すぎ)。期待: ep100 で BA[r1.5_t0.2] ω が < 5 px@fx、val NLL が ~4 台、未到達なら次は cs=512 / pose_emb refactor (intrinsic-only Linear(1→D)) / longer run のどれかへ分岐。前回 v2 (km-only) は ep2 で val NLL 5.74 / BA r1.0 6.8 px、これを 3-DS で再現できるかが ep10 までの sanity gate。'"

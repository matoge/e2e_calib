#!/bin/bash
# DGX3 8GPU: PandaSet full-frame LMDB を img256 / grid32 で 100ep 回す。
# DGX4 (img128/grid16) と同時走で、解像度を上げると ω/t calib 残差が
# subpixel に近づくか (memory project_resolution_hypothesis_512 の延長) を検証。
# pair_mode=False (single-frame calib のみ)。
#
# Reproduce:
#   PATH=/home/hfunaya/.pyenv/versions/3.10.4/bin:$PATH \
#       bash infra/.kick_dgx3_pandaset_full_img256.sh
#   → ClearML web (http://172.16.200.185:8082)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="ps_full_n4_img256_grid32_dgx3_100ep"

read -r -d '' WHY <<'WHY_EOF' || true
DGX3 8GPU で PandaSetCalibDatasetFull (/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full、inst 8240 frame、scene/cam ごとに 1 inst = 1 fullsweep+1 fullimage) を img_size=256 / grid_n=32 / min/max_crop_px=512/1024 で 100ep 回し、DGX4 並走 (km_wv_wm の img128/grid16 同条件、ps_full_n4_img128_parity_dgx4_100ep, task 7f6bd62b217f4089a751d8bc70c5506d) と並行で解像度倍増の effect を測る。動機: 既存 200×800 sub-pixel 仮説 (project_resolution_hypothesis_512: 128 入力 fx 換算 1.6 px → 4× 解像度で σ⁻² 16× → sub-pixel 射程) を full-frame dataset 経路で再現できるかのゲート。差分は img_size 128→256, grid_n 16→32, min/max_crop_px 256/512→512/1024 のみ。loss/model/eval は既存と同一: n_layers=4 ConvNeXt deform_mode=ml frustum_dense=True use_pose_emb=True (PoseEmb=zeros = intrinsic-only) max_rot_deg=1.5 max_offset_m=0.20 oversample=4 batch=32/rank (img256 は VRAM 倍以上、bs を半分に) lr=1e-3 cosine→1e-6 BA-eval cs=256 npi=4 every 10ep。期待: ep1 train loss 5-6、sps/rank ≥ 50 (img256 は img128 比で ~50% TFLOPs)、ep20 val_nll が DGX4 img128 baseline と同等 or 若干良、ep100 で val_nll < 1 / BA omega < 0.1deg / t < 0.03m が出れば 256 入力の sub-pixel 効果 confirmed = 次は img512 / cross-frame に進む素地。失敗時 = OOM (bs を 16 に下げる)、もしくは img256 で grid32 だと per-cell pt 数が不足、を疑う。並走 DGX2 (200ep ml_dense_pe img128) と DGX4 (100ep ps_full img128 parity) と合わせて img128 vs img256 の calib 軌跡比較が一発で取れる。
WHY_EOF

WHY_QUOTED=$(printf '%q' "$WHY")

ARGS="--cache /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full \
--epochs 100 --batch-size 32 --img-size 256 --n-layers 4 --convnext --grid-n 32 \
--deform-mode ml --frustum-dense --use-pose-emb \
--rot-deg 1.5 --t-m 0.20 --oversample 4 --workers 8 --min-crop-px 512 --max-crop-px 1024 \
--lr 0.001 --lr-min 1e-06 \
--ba-eval-start-ep 1 --ba-eval-every 10 --ba-eval-n-seeds 4 --ba-eval-n-inst 200 \
--clearml --why $WHY_QUOTED"

exec ./infra/submit_clearml_task.sh \
  --name     "$RUN_NAME" \
  --script   scripts/training/train_ps_v3_ddp.py \
  --queue    dgx3 \
  --num-gpus 8 \
  --image    e2e-calib-train:np2 \
  --args     "$ARGS"

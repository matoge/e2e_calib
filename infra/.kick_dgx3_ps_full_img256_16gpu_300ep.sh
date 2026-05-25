#!/bin/bash
# DGX3 16GPU: PandaSet full-frame img256/grid32 を 300ep に延長 + GPU 倍増。
# 元 run = ps_full_n4_img256_grid32_dgx3_100ep (8GPU/100ep) が良かったので、
# 同条件で (1) 16GPU に倍増、(2) 100→300ep に延長。差分はそれだけ。
#
# Reproduce:
#   PATH=/home/hfunaya/.pyenv/versions/3.10.4/bin:$PATH \
#       bash infra/.kick_dgx3_ps_full_img256_16gpu_300ep.sh
#   → ClearML web (http://172.16.200.185:8082)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="ps_full_n4_img256_grid32_dgx3_300ep_16gpu"

read -r -d '' WHY <<'WHY_EOF' || true
DGX3 16GPU で PandaSetCalibDatasetFull (/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full、inst 8240 frame、scene/cam ごとに 1 inst = 1 fullsweep+1 fullimage) を img_size=256 / grid_n=32 / min/max_crop_px=512/1024 で 300ep 回す。前回 ps_full_n4_img256_grid32_dgx3_100ep (8GPU/100ep) でユーザ判定「無茶苦茶性能良い」だったため、(1) GPU 8→16 に倍増し global batch を 256→512 に上げる、(2) 100→300ep に学習延長、の 2 軸変更。lr/モデル/データ/loss は既存と完全同一: n_layers=4 ConvNeXt deform_mode=ml frustum_dense=True use_pose_emb=True (PoseEmb=zeros = intrinsic-only) max_rot_deg=1.5 max_offset_m=0.20 oversample=4 batch=32/rank min/max_crop_px=512/1024 lr=1e-3 cosine→1e-6 BA-eval cs=256 npi=4 every 10ep。global batch 倍増に対し lr を据え置くのは、既存 100ep が lr=1e-3 で素直に収束しており、安全側 (sqrt scaling だと 1.4e-3 だが今回は性能維持優先) で再現性を取りに行くため。期待: ep1 train loss 5-6, sps/rank ~50 (img256, V100 32GB, bs32)、ep30 で 100ep run の ep100 程度 (val_nll 1 前後 / BA omega 0.1deg / t 0.03m)、ep100 で更に半減、ep300 で plateau に到達して img128 grid16 baseline 比 4x 解像度の sub-pixel 効果を上限まで搾り取れる見込み (project_resolution_hypothesis_512: 128 入力 fx 換算 1.6 px → 4× 解像度で σ⁻² 16× → sub-pixel)。これが順当に伸びれば次は img512 / cross-frame に進む素地。失敗時 = OOM (bs を 16 に下げる)、もしくは lr 据え置きが大きすぎて ep10 内で発散 (lr=7e-4 に下げて再走)、を疑う。並走 DGX2 200ep ml_dense_pe img128 / DGX4 100ep ps_full img128 parity と合わせて img128 vs img256 vs img256/300ep の calib 軌跡比較が一発で取れる。
WHY_EOF

WHY_QUOTED=$(printf '%q' "$WHY")

ARGS="--cache /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full \
--epochs 300 --batch-size 32 --img-size 256 --n-layers 4 --convnext --grid-n 32 \
--deform-mode ml --frustum-dense --use-pose-emb \
--rot-deg 1.5 --t-m 0.20 --oversample 4 --workers 8 --min-crop-px 512 --max-crop-px 1024 \
--lr 0.001 --lr-min 1e-06 \
--ba-eval-start-ep 1 --ba-eval-every 10 --ba-eval-n-seeds 4 --ba-eval-n-inst 200 \
--clearml --why $WHY_QUOTED"

exec ./infra/submit_clearml_task.sh \
  --name     "$RUN_NAME" \
  --script   scripts/training/train_ps_v3_ddp.py \
  --queue    dgx3 \
  --num-gpus 16 \
  --image    e2e-calib-train:np2 \
  --args     "$ARGS"

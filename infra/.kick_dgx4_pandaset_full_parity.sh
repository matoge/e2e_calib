#!/bin/bash
# DGX4 8GPU: PandaSet full-frame LMDB を素のキャリブで 100ep 回し、tile-mode との parity を取る。
# pair_mode=False (single-frame calib のみ)。dataset 切替の parity を測ってから cross-frame 拡張に進む。
#
# Reproduce:
#   PATH=/home/hfunaya/.pyenv/versions/3.10.4/bin:$PATH \
#       bash infra/.kick_dgx4_pandaset_full_parity.sh
#   → ClearML web (http://172.16.200.185:8082)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="ps_full_n4_img128_parity_dgx4_100ep"

read -r -d '' WHY <<'WHY_EOF' || true
DGX4 8GPU で PandaSetCalibDatasetFull (新しい full-frame LMDB /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full、inst 8240 frame、scene/cam ごとに 1 inst = 1 fullsweep+1 fullimage) を pair_mode=False のままで 100ep 回し、現行 tile-mode (km_wv_wm_n4_img128_grid16_repro_dgx1_8gpu_direct と同条件) との calib parity を測る。動機: cross-frame 拡張 (forward_pair で同じ重みを (A solo, B solo, A→B E2E) の 3 回叩き、dpose_AB を pose_emb で渡す) に進む前に、データロード経路を tile-mode → full-frame に切り替えても val NLL が ~1 BA omega ~0.13deg t ~0.04m を保てることを確認するゲート。差分は dataset cache のみ (single PandaSet, full-frame inst から build_window が同じ 256-512 px sub-crop を切る)。loss/model/eval は tile-mode 既存と完全同一: n_layers=4 ConvNeXt img_size=128 grid_n=16 deform_mode=ml frustum_dense=True use_pose_emb=True (PoseEmb=zeros = intrinsic-only) max_rot_deg=1.5 max_offset_m=0.20 oversample=4 min/max_crop_px=256/512 batch=64/rank lr=1e-3 cosine→1e-6 BA-eval cs=256 npi=4 every 10ep。期待: ep1 train loss 5-6 (smoke と一致)、sps/rank ≥ 100、ep20 val_nll が 7-8 程度 (DGX1 baseline 同等)、ep100 で val_nll < 2 / BA omega < 0.2deg。これが parity OK なら次に pair_mode=True を入れて collate_pair を流し A-only loss で同じ収束軌跡が出ることを確認、その後 forward_pair の 3 ロス全部 (A solo + B solo + A→B E2E) に拡張する。失敗時 = full-frame inst の build_window が tile-mode と違う pivot 分布を吐いてる、もしくは meta.pt の split が偏ってる、を疑う。
WHY_EOF

WHY_QUOTED=$(printf '%q' "$WHY")

ARGS="--cache /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full \
--epochs 100 --batch-size 64 --img-size 128 --n-layers 4 --convnext --grid-n 16 \
--deform-mode ml --frustum-dense --use-pose-emb \
--rot-deg 1.5 --t-m 0.20 --oversample 4 --workers 8 --min-crop-px 256 --max-crop-px 512 \
--lr 0.001 --lr-min 1e-06 \
--ba-eval-start-ep 1 --ba-eval-every 10 --ba-eval-n-seeds 4 --ba-eval-n-inst 200 \
--clearml --why $WHY_QUOTED"

exec ./infra/submit_clearml_task.sh \
  --name     "$RUN_NAME" \
  --script   scripts/training/train_ps_v3_ddp.py \
  --queue    dgx4 \
  --num-gpus 8 \
  --image    e2e-calib-train:np2 \
  --args     "$ARGS"

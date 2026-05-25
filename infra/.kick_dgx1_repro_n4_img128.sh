#!/bin/bash
# 現サーバーモデル (experiments/km_wv_wm_dgx2_n4_img128_8gpu_HEAD/best_model.pt) を
# DGX1 16GPU で完全再現するための kick。code は HEAD (commit 74b85f6) 状態 —
# InfoHead/Beta-NLL/bound の細工は全部 revert 済み、純粋な sigma-head 経路。
#
# 元 DGX2 8GPU bs=128 → global=1024。DGX1 8GPU per-rank bs=128 で global=1024
# を保つ。16GPU で agent が拾わない ticket があったので 8GPU に固定。
#
# Reproduce (この script は ClearML 経由が default。直 docker run は使わない):
#   (A) ClearML queue=dgx1 経由 [DEFAULT, この script が submit する経路]:
#       PATH=/home/hfunaya/.pyenv/versions/3.10.4/bin:$PATH \
#           bash infra/.kick_dgx1_repro_n4_img128.sh
#       → ClearML web (http://172.16.200.185:8082) で task 進行を見る。
#       agent が queue を拾わない時は agent を直すこと (直 docker run に逃げない、
#       ClearML 上の対照基盤がなくなる)。memory: feedback_clearml_default.md。
#
#   (B) 単一 GPU の sanity smoke (新規 host):
#       cd /home/hfunaya/git/e2e_calib
#       /home/hfunaya/.pyenv/versions/3.10.4/bin/python -m accelerate.commands.launch \
#         --num_processes=1 --mixed_precision=fp16 \
#         scripts/training/train_ps_v3_ddp.py --epochs 1 --batch-size 8 \
#         --img-size 128 --n-layers 4 --convnext --grid-n 16 --deform-mode sl \
#         --rot-deg 1.5 --t-m 0.20 --oversample 1 --workers 2 \
#         --min-crop-px 256 --max-crop-px 512 \
#         --cache /home/hfunaya/cache/kamikado_v3_tiled
#
# 期待値 (ep1):
#   train loss 6.x → 5.x (sigma-head NLL = ½r²/σ² + log σ)
#   sps/rank 100 前後 (warm-up 中は 30→150 で増える)
#   val_nll は同オーダー
# 200ep 完走後の参考値 (元 DGX2 run):
#   val_nll ≈ 6, BA omega ≈ 0.13deg, BA t ≈ 0.04m
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="km_wv_wm_n4_img128_grid16_repro_dgx1_16gpu"

read -r -d '' WHY <<'WHY_EOF' || true
ClearML 上の現 calib API サーバーが提供してるモデル (experiments/km_wv_wm_dgx2_n4_img128_8gpu_HEAD/best_model.pt) を DGX1 16GPU で完全再現する。元 run は DGX2 8GPU bs=128 global=1024、本 run は DGX1 16GPU per-rank bs=64 global=1024 で再現。conf 完全コピー: n_layers=4, img_size=128, ConvNeXt, oversample=4, min/max_crop_px=256/512, grid_n=16, max_rot_deg=0.5, max_offset_m=0.20, use_pose_emb なし, use_info_head なし (sigma-head)。動機: 直前の InfoHead bound + Beta-NLL run (ef81ab9f) は ep1 train loss が 1500-6000 で「安定化したのに 1000× 化」したと判断不能なまま走らせてしまったため一旦 revert。ベースラインとしてサーバーモデル相当を「DGX1 でも同じ ep1 train loss / val_nll が出る」ことを先に確認する。期待: ep1 で train loss 50-500 (sigma-head は r²/σ² の中央値 ~10-100 オーダー) + val_nll 6-12 程度、200ep 完走で BA omega ≈ 0.13deg / val_nll ≈ 6 まで降りる (DGX2 8GPU run の HEAD 値の再現)。これが出たら次の改善 (Cross-Frame, multi-frame) に進める基準点になる。
WHY_EOF

WHY_QUOTED=$(printf '%q' "$WHY")

ARGS="--cache /home/hfunaya/cache_v4/kamikado_v3_tiled,/home/hfunaya/cache_v4/woven_v3_tile,/home/hfunaya/cache_v4/waymo_v3_tiled_i \
--epochs 200 --batch-size 64 --img-size 128 --n-layers 4 --convnext --grid-n 16 --deform-mode sl \
--rot-deg 1.5 --t-m 0.20 --oversample 4 --workers 8 --min-crop-px 256 --max-crop-px 512 \
--lr 0.001 --lr-min 1e-06 \
--ba-eval-start-ep 1 --ba-eval-every 10 --ba-eval-n-seeds 4 --ba-eval-n-inst 200 \
--clearml --why $WHY_QUOTED"

exec ./infra/submit_clearml_task.sh \
  --name     "$RUN_NAME" \
  --script   scripts/training/train_ps_v3_ddp.py \
  --queue    dgx1 \
  --num-gpus 8 \
  --image    e2e-calib-train:np2 \
  --args     "$ARGS"

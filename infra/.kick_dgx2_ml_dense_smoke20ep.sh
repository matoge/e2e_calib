#!/bin/bash
# DGX2 並走 smoke: deform_mode=ml + frustum_dense=True + use_pose_emb=True で
# 既存 calib タスク (km+wv+wm 3-DS, n4 img128 grid16) が 20ep で収束方向に
# 動くかを確認する。これが OK なら cross-frame の重み共有拡張 (Δpose=0 →
# calib 退化、Δpose≠0 → cross-frame、KV に target 画像 coarse/fine + target
# LiDAR dense map の 3 level を deformable-ML attn で同時に attend) に進める。
# 落ちたら ml + frustum_dense の組み合わせ自体を先に潰す。
#
# 元サーバーモデル (deform_mode=sl, frustum_dense なし) との差:
#   - sl → ml: 全 block が (coarse_feat, fine_feat, [lidar_dense]) 全 level を
#             同時に deformable attn で見るようになる。level_embed (learnable)
#             で level 区別。
#   - frustum_dense=True: FrustumLocalEncoder.forward_dense が gh*gw=256 の
#             dense LiDAR cell map を出して 3rd KV level に並べる。空 cell も
#             cell_uv_embed (learnable) で UV 位置情報だけは残る。
#   - use_pose_emb=True: PoseEmb([SE3, log_vfp]) → D-dim bias を Q/KV に加算。
#             今は SE3=zeros (calib モード) なので intrinsic-only emb として
#             働く。cross-frame 拡張時はここに Δpose を流す経路として再利用。
#
# 期待 (20ep, 並走 baseline は dgx1 の sl 構成):
#   - ep1 train loss ~ 5-6 (sigma-head NLL)、sps/rank が落ち過ぎないか
#   - ep20 val_nll が dgx1 baseline ep20 と同等オーダー (~7-8 程度) まで来るか
#   - BA omega の trajectory が dgx1 と同等 or それ以上に降りるか

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="km_wv_wm_n4_img128_ml_dense_pe_dgx2_20ep_smoke"

read -r -d '' WHY <<'WHY_EOF' || true
DGX1 で並走中の baseline (sl + frustum_dense なし、現サーバーモデル相当、km_wv_wm_n4_img128_grid16_repro_dgx1_16gpu) と同じデータ・同じ摂動・同じ epoch 上限の 20ep で、deform_mode='ml' (全 block が coarse/fine 両 level を同時に deformable-ML attn で見る) + frustum_dense=True (FrustumLocalEncoder.forward_dense が出す gh*gw=256 の dense LiDAR cell map、空 cell は cell_uv_embed のみで UV 位置情報を保持、を 3rd KV level に追加) + use_pose_emb=True (PoseEmb([SE3, log_vfp]) D-dim bias を Q/KV に加算、現状 SE3=zeros で intrinsic-only emb) の組み合わせが「現 calib タスクで」収束するかを先に確認する。これが OK なら cross-frame 拡張: Frame1→Frame2 で同じ CalibNetDepth forward を 2 回呼ぶ (重み 100% 共有)、Frame2 の Q に PoseEmb([Δpose_pert, log_vfp]) を加算、Frame2 KV = (img2_coarse, img2_fine, lidar_dense2)、head は既存の duv+W に Δd+log_σ_d (1D NLL) を追加、Δpose=0 で calib に連続退化、loss は gaussian2d_nll(duv) + 1D NLL(Δd) の dual、に進む。期待: ep1 train loss 5-6 (sigma-head NLL = ½r²/σ² + log σ)、sps/rank 100 前後 (ml は per-block の attn 計算が 2-3× になるので sl の 80-90% を覚悟)、ep20 val_nll が DGX1 baseline ep20 と同等 (7-8 程度) まで来れば cross-frame 拡張に進む。降りなかったら ml + frustum_dense の組み合わせ自体を先に潰す (sl + frustum_dense=False に戻して何が壊したかの差分検証)。
WHY_EOF

WHY_QUOTED=$(printf '%q' "$WHY")

ARGS="--cache /home/hfunaya/cache_v4/kamikado_v3_tiled,/home/hfunaya/cache_v4/woven_v3_tile,/home/hfunaya/cache_v4/waymo_v3_tiled_i \
--epochs 20 --batch-size 64 --img-size 128 --n-layers 4 --convnext --grid-n 16 \
--deform-mode ml --frustum-dense --use-pose-emb \
--rot-deg 1.5 --t-m 0.20 --oversample 4 --workers 8 --min-crop-px 256 --max-crop-px 512 \
--lr 0.001 --lr-min 1e-06 \
--ba-eval-start-ep 1 --ba-eval-every 5 --ba-eval-n-seeds 4 --ba-eval-n-inst 200 \
--clearml --why $WHY_QUOTED"

exec ./infra/submit_clearml_task.sh \
  --name     "$RUN_NAME" \
  --script   scripts/training/train_ps_v3_ddp.py \
  --queue    dgx2 \
  --num-gpus 8 \
  --image    e2e-calib-train:np2 \
  --args     "$ARGS"

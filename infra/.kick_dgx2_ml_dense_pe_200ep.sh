#!/bin/bash
# DGX2 ベースライン 200ep: deform_mode=ml + frustum_dense=True + use_pose_emb=True
# (smoke 20ep run `km_wv_wm_n4_img128_ml_dense_pe_dgx2_20ep_smoke` の 200ep 拡張)。
# smoke が ep4 val_nll=3.05 → step 1675 train loss 1.7-2.0 まで降りており calib
# モードでは収束方向が確認できたので、これを今後の対照基盤として 200ep 完走させる。
# 重みは smoke を resume せず新品から (smoke は途中停止 + lr schedule が 20ep 想定で
# 200ep の cosine と整合しないため、フルで回し直し)。
#
# Reproduce:
#   PATH=/home/hfunaya/.pyenv/versions/3.10.4/bin:$PATH \
#       bash infra/.kick_dgx2_ml_dense_pe_200ep.sh
#   → ClearML web (http://172.16.200.185:8082) で task 進行を見る。
#
# 期待 (200ep):
#   - ep1 train loss ~ 5-6 (sigma-head NLL = ½r²/σ² + log σ)
#   - ep20 val_nll は DGX1 baseline (sl, pose_emb なし) ep20 と同等オーダー
#   - ep200 完走で val_nll ~1 付近まで降りる見込み (smoke ep4 の trajectory から外挿)
#   - BA omega ≈ 0.13deg / t ≈ 0.04m が DGX1 baseline と同等 or それ以上
# OK なら cross-frame 拡張 (Frame1→Frame2 重み共有、Δpose を pose_emb 経由で
# 流す、head に Δd+log_σ_d 追加) に進む。

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep"

read -r -d '' WHY <<'WHY_EOF' || true
DGX2 で 20ep smoke (km_wv_wm_n4_img128_ml_dense_pe_dgx2_20ep_smoke) が ep4 val_nll=3.05、step 1675 train loss 1.7-2.0 まで降りて calib モードでの収束方向が確認できたため、今後の cross-frame 拡張の対照基盤としてこの構成 (deform_mode='ml' で全 block が coarse/fine 両 level を同時に deformable-ML attn で見る + frustum_dense=True で FrustumLocalEncoder.forward_dense が出す gh*gw=256 の dense LiDAR cell map を 3rd KV level に追加 + use_pose_emb=True で PoseEmb([SE3, log_vfp]) の D=128 bias を Q/KV に加算、現状 SE3=zeros で intrinsic-only emb として働く) を 200ep 完走させる。重みは smoke を resume せず新品から (smoke は途中停止 + lr schedule が 20ep 想定で 200ep の cosine と整合しない)。並走中: DGX1 8GPU baseline (km_wv_wm_n4_img128_grid16_repro_dgx1_8gpu_direct, sl + pose_emb なし、現サーバーモデル相当の 3-DS calib) と DGX1 8GPU Phase A1 自己教師 pose_emb (km_wv_wm_n4_img128_pose_emb_selfsup_dgx1_8gpu_direct, sl + δ=δ1+δ2 split で δ1 を pose_emb に渡し δ2 残差だけ NLL 学習)。conf 完全に smoke と同一: n_layers=4 img_size=128 ConvNeXt grid_n=16 oversample=4 min/max_crop_px=256/512 max_rot_deg=1.5 max_offset_m=0.20。期待: ep1 train loss 5-6、sps/rank 100 前後 (ml は per-block の attn 計算が 2-3× で sl の 80-90% を覚悟、smoke では sps/rank=438 出てたので fp16 + dense は問題なし)、ep20 val_nll が DGX1 baseline ep20 と同等 (~7-8 程度)、ep200 完走で val_nll ~1 付近 + BA omega ≈ 0.13deg / t ≈ 0.04m。これが今後の cross-frame 拡張 (Frame1→Frame2 で同じ CalibNetDepth forward を 2 回呼ぶ、重み 100% 共有、Frame2 の Q に PoseEmb([Δpose_pert, log_vfp]) を加算、Frame2 KV = (img2_coarse, img2_fine, lidar_dense2)、head は既存の duv+W に Δd+log_σ_d (1D NLL) を追加、Δpose=0 で calib に連続退化、loss = gaussian2d_nll(duv) + 1D NLL(Δd) の dual) の対照基盤になる。
WHY_EOF

WHY_QUOTED=$(printf '%q' "$WHY")

ARGS="--cache /home/hfunaya/cache_v4/kamikado_v3_tiled,/home/hfunaya/cache_v4/woven_v3_tile,/home/hfunaya/cache_v4/waymo_v3_tiled_i \
--epochs 200 --batch-size 64 --img-size 128 --n-layers 4 --convnext --grid-n 16 \
--deform-mode ml --frustum-dense --use-pose-emb \
--rot-deg 1.5 --t-m 0.20 --oversample 4 --workers 8 --min-crop-px 256 --max-crop-px 512 \
--lr 0.001 --lr-min 1e-06 \
--ba-eval-start-ep 1 --ba-eval-every 10 --ba-eval-n-seeds 4 --ba-eval-n-inst 200 \
--clearml --why $WHY_QUOTED"

exec ./infra/submit_clearml_task.sh \
  --name     "$RUN_NAME" \
  --script   scripts/training/train_ps_v3_ddp.py \
  --queue    dgx2 \
  --num-gpus 8 \
  --image    e2e-calib-train:np2 \
  --args     "$ARGS"

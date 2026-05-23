#!/bin/bash
# DGX1 GPU 8-15 並走 (baseline `km_wv_wm_n4_img128_grid16_repro_dgx1_8gpu_direct`
# が 0-7 を占有中)。Phase A1 自己教師 pose_emb 学習: δ = δ1 + δ2 split で
# δ1 を pose_emb に「ヒント」として食わせ、δ2 だけ NLL で残差回帰させる。
#
# 構成は baseline と同じ (km+wv+wm 3-DS, n4 img128 grid16 ConvNeXt sl
# deform_mode, oversample=4, min/max_crop_px=256/512) に --use-pose-emb と
# --pose-emb-self-sup を足しただけ。これにより:
#   * dataset は δ1 (±1.0°/±0.20m) と δ2 (同じレンジ) を独立 sample、合成 δ で
#     入力を投影。
#   * true_uvd は δ1 適用後の uv (uv_pre)。target = true - dist = -δ2 起因の
#     reproj shift だけ。
#   * model.forward(pose_emb_se3=δ1) で pose_emb がヒントを線形変換 → D=128 bias
#     を Q + KV に broadcast 加算。
#   * δ1=0 で legacy calib に厳密に退化 (構造的同一性が保証される)。
#
# 期待 (ep1):
#   train loss は baseline より HIGHER オーダーになり得る (δ2 だけが target だが
#   pose_emb がランダム初期化なのでヒントが当面ノイズ→残差信号が薄まる)。だが
#   2-3ep で baseline と同等付近に降りるはず (pose_emb が "δ1 をそのまま反映する
#   shift" を学習するのは線形変換 1 段でほぼ自明な map)。
# 期待 (ep20):
#   * baseline と同等以下の val_nll に到達。
#   * pose_emb が「δ1 を予測 uv にどれだけ反映するか」を学んでいることが、
#     δ1 ≠ 0 vs δ1 = 0 で eval した時の予測差で確認できる。
#   * これが OK なら Phase B/C で δ1 のレンジを実フレーム移動スケール
#     (~1m vertical for 30 km/h × 0.1s) に拡張、PandaSet で本物の cross-frame に。
# 落ちたら Plan B (deformable attn の reference point shift) へ。
#
# Reproduce:
#   PATH=/home/hfunaya/.pyenv/versions/3.10.4/bin:$PATH \
#       bash infra/.kick_dgx1_pose_emb_selfsup.sh
#   → ClearML web (http://172.16.200.185:8082) で task 進行を見る。
#
# 注: ClearML 経由が default。直 docker run には逃げない (memory:
# feedback_clearml_default.md)。

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="km_wv_wm_n4_img128_pose_emb_selfsup_dgx1_8gpu"

read -r -d '' WHY <<'WHY_EOF' || true
DGX1 で並走中の baseline (km_wv_wm_n4_img128_grid16_repro_dgx1_8gpu_direct, sl + frustum_dense なし、現サーバーモデル相当の 3-DS calib) と同じ data・同じ epoch 上限の 200ep で、自己教師 pose_emb 学習を試す。アイデア: 摂動を δ = δ1 + δ2 に分け (各 ±1.0°/±0.20m, total ±2°/±0.40m)、δ1 を pose_emb に「ヒント」として渡し、target は δ1 適用後の uv (uv_pre)、つまり残差 = -δ2 起因の reproj shift だけになる。pose_emb (D=128 線形変換 + broadcast bias on Q + img/lidar KV) は「与えられた δ1 が画像トークン位置をどう動かしたか」の線形写像を、δ1 を入れずに学んでいた現状から、δ1 を直接条件付けして学ぶ形に変わる。δ1=0 で legacy calib に厳密に退化 (構造的同一性が保てる)。動機: 直近の DGX2 smoke (km_wv_wm_n4_img128_ml_dense_pe_dgx2_20ep_smoke, ml + frustum_dense + pose_emb) は ep4 val_nll=3.05 で baseline trajectory に乗っており calib モードでは pose_emb が機能している。ここから cross-frame 拡張に進む前段として、pose_emb が「外部から与えられた pose 量」を実際に内部に取り込めることを単一フレーム calib タスク内で確かめておく。期待: ep1 train loss 5-7 (sigma-head NLL = ½r²/σ² + log σ、δ2 だけが target だが pose_emb がランダム初期化なので最初は baseline より高い可能性)、2-3ep で baseline と同等付近に降り、ep20 val_nll が baseline ep20 (~7-8 想定) と同等以下に到達。その後 200ep 完走で BA omega が baseline と同等。次フェーズで δ1 を 1m 縦並進スケールに拡張し、PandaSet/Waymo の連続フレームでの本物の cross-frame inference に進む。落ちた場合の plan B は deformable attn の reference point を pose_emb 出力で shift する形に変更 (現状の broadcast 加算→ref-point shift)。
WHY_EOF

WHY_QUOTED=$(printf '%q' "$WHY")

ARGS="--cache /home/hfunaya/cache_v4/kamikado_v3_tiled,/home/hfunaya/cache_v4/woven_v3_tile,/home/hfunaya/cache_v4/waymo_v3_tiled_i \
--epochs 200 --batch-size 64 --img-size 128 --n-layers 4 --convnext --grid-n 16 --deform-mode sl \
--use-pose-emb --pose-emb-self-sup \
--rot-deg 1.0 --t-m 0.20 --oversample 4 --workers 8 --min-crop-px 256 --max-crop-px 512 \
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

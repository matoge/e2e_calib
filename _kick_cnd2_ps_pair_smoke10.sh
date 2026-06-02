#!/bin/bash
# CND2 cross-frame smoke: 10 epoch, oversample=8 で convergence shape を確認
# ── kick #1 (50ep va_mse=1.61) は ep20 で sub-2.5px に達してたので、
# oversample 上げて epoch あたりのデータ量を 4× にすれば 10ep でほぼ同等の
# 収束カーブが踏める。SE(3) PoseEmb を入れた今、kick #1 と break しない
# ことを 30 分で確認するのが目的。
set -euo pipefail
NAME="cnd2_ps_pair_se3_smoke10_$(date +%m%d_%H%M)"

WHY=$(cat <<'WHY_END'
CND2 cross-frame SE(3) smoke — 10ep with oversample=8 (= kick #1 50ep
relative data-volume baseline at 50/(50/4·10/2)=8x oversample).

Goal:
  * Confirm preflight vis_check_pair_getitem panel (ConnectionPatch dotted
    lines = same world point pts_A[i] in A vs B image) renders correctly
    on production training task — A↔B correspondence after pts_A B-projection
    fix (commit bf328de) and force_sub_idx alignment (commit 8599e3a-ish).
  * Confirm SE(3) PoseEmb (translation_mlp zero-init → no break) doesn't
    diverge in the first 5 epochs.
  * Confirm va_mse降下シェイプ: kick #1 ep5≈4.8 / ep10≈3.1 px。今回 10ep ですべて
    同等 or それより良い軌跡が出れば SE(3) extension が無害と確定。

設定 vs kick #1:
  * epochs 50 → 10
  * oversample 2 → 8 (= 1ep あたり ~4× データ量)
  * 残りすべて kick #1 と同一: img128 / grid16 / pair-stride10 / batch32 /
    n_iter4 / σ_ypr=1.0° / σ_t=0.20m / use_info_head

走行時間目安:
  * kick #1 1ep ≈ 180s on dgx2 8×V100 → 10ep × oversample 4× ≈ 60min
WHY_END
)

CUDA_DEVICES=0,1,2,3,4,5,6,7 \
infra/submit_clearml_task.sh \
  --name        "$NAME" \
  --script      datasets/train_cnd2_ddp.py \
  --queue       dgx2 \
  --num-gpus    8 \
  --image       e2e-calib-train:np2 \
  --project     e2e_calib/cross-frame \
  --args "--cache /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full --pair-mode --pair-stride 10 --epochs 10 --batch-size 32 --img-size 128 --grid-n 16 --oversample 8 --workers 8 --val-fraction 0.1 --n-iter 4 --rot-deg 1.0 --t-m 0.20 --use-info-head --clearml --clearml-project e2e_calib/cross-frame --why \"$WHY\""

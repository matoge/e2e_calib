#!/bin/bash
# Kick CalibNet2 single-frame calib baseline on PandaSet FULL LMDB.
# Pairs with _kick_cnd2_ps_pair.sh: same data / arch / hyper-params,
# only --pair-mode is OFF. Provides the calib reference val_mse curve to
# compare against the cross-frame run.
set -euo pipefail
NAME="cnd2_ps_calib_$(date +%m%d_%H%M)"

WHY=$(cat <<'WHY_END'
CalibNet2 single-frame calib baseline kick on PandaSet FULL LMDB.

Companion to cnd2_ps_pair_*: identical settings except --pair-mode is OFF.
Goal: reference val_nll / val_mse curves for the legacy single-frame calib
path to compare against the cross-frame run on the same data.

設定: 50 epoch、img_size=128、grid_n=16、batch_size=32 (per-rank、global=256
on 8 GPU)、oversample=2、AdamW lr=1e-3、HAT 摂動 σ_ypr=1.0° / σ_t=0.2m。

期待:
- ep1 SPS が見えること
- 50ep 終盤で val_mse が cnd2 既存 baseline (~2.5 NLL) 周辺に着地
- pair 版と曲線を比較して、cross-frame 化で精度がどれだけ動くか測る
WHY_END
)

infra/submit_clearml_task.sh \
  --name        "$NAME" \
  --script      datasets/train_cnd2_ddp.py \
  --queue       dgx4 \
  --num-gpus    16 \
  --image       e2e-calib-train:np2 \
  --project     e2e_calib/cross-frame \
  --args "--cache /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full --epochs 50 --batch-size 32 --img-size 128 --grid-n 16 --oversample 2 --workers 8 --val-fraction 0.1 --n-iter 4 --rot-deg 1.0 --t-m 0.20 --use-info-head --clearml --clearml-project e2e_calib/cross-frame --why \"$WHY\""

# kamikado + woven_v3_cal_ok 標準訓練手順

2026-10-03 ／ CalibNet2 single-stage

---

## 要約

CalibNet2 を kamikado + woven_v3_cal_ok 連結 cache で訓練する。
これまでの S1→S3 chain は中間 ckpt の受け渡しが必要で、再現性が低かった。
BA warmup を slow ramp にすれば 1 コマンドで完結する:

- **ep 0-10**: pts-only (`ba_weight = 0`)。μ-head と log_sigma を gaussian2d_nll で粗くフィット。
- **ep 10-30**: BA weight を 0 → 0.05 まで滑らかに増やす。InfoHead2x2 が cold start から徐々に GN 重みを担う。
- **ep 30-60**: full BA (`ba_weight = 0.05`)。両 head joint refinement。

## コマンド

```bash
scripts/train_kmwv_calok.sh
# → container 名 kmwv_calok_<MMDD_HHMM>、GPU 4-11、60 ep
```

環境変数で上書き可:

```bash
NAME=myrun GPUS='4,5,6,7' PORT=29514 EPOCHS=80 \
    scripts/train_kmwv_calok.sh
```

## 前提

- Cache は **frame-level** でビルド (`build_woven_sequence_v3.py` の `--tile` を**付けない**)。
  tile 化すると `_plan_grid` が 1 cell しか返せず、share_pert = 40 copy fusion が
  degenerate → BA が ep1 で chi2r 8000+ に跳ね NaN になる (
  [docs/2026-10-03_kmwv_calok_training.md](./2026-10-03_kmwv_calok_training.md) で
  発覚した事故)。`datasets/pandaset_full.py` の guard がこれを検出して abort する。
- woven_v3_cal_ok は loom frontend の **CAL OK タグ付きシーケンスのみ** から build:
  11 seqs (ip607×2, ip651×3, ip654×5, unilab×1、2018 frames)。

## 出力

- `experiments/<name>/train.log` — epoch 毎メトリクス
- `experiments/<name>/best_model.pt` — val_nll min
- `experiments/<name>/last_model.pt` — 最新 (NaN では更新スキップ)
- ClearML (`e2e_calib/calib` project) に自動登録

## ハイパーパラメタ一覧

| 項目 | 値 | 備考 |
|---|---|---|
| epochs | 60 | ep0-10 pts, 10-30 ramp, 30-60 full |
| lr | 3e-4 → cosine → 1e-6 | 標準 |
| ba_weight | 0.05 (@ep30+) | 大きすぎると NaN、小さすぎると sigma 校正不足 |
| ba_warmup | ep10 → ep30 | 10 ep 以上の pts-only warmup 必須 |
| ba_iter | 4 | GN 反復 |
| ba_damping | 1e-3 | LM 緩和 |
| mixed_precision | fp16 | model forward のみ、BA は fp64 自動キャスト |
| rot_deg | 0.5 | train pert range |
| t_m | 0.20 | train pert range |
| oversample | 40 | fuse group size (= `grid_iw/cs × grid_ih/cs` = 8×5) |
| grid_iw × grid_ih | 3840 × 2160 | 原画像 size |
| per_cache_crop_px | 512 | 両 cache (fisheye native) |

## 期待性能

- val POSE rot `<0.05°` by ep20
- chi2r → 1.0 by ep15
- val NLL `<1.6` by ep20

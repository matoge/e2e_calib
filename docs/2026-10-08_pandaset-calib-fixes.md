# PandaSet：学習・推論・API を同じ経路で回す（2026-10-07〜08 の修正と結果）

## 結論

- PandaSet の段 1（BA なし）→ 段 2（BA あり）を、学習・推論（`infer_calib.py`）・API（`/api/eval_frame`）の同じ経路で回した。
- 外でずらしたポーズ（回転 各軸 ±0.5°、並進 各軸 ±0.2 m）を渡すと、val 40 フレームの補正後の中央値は次のとおり。
  - ヨー 0.027°、ピッチ 0.029°、ロール 0.065°（PandaSet 前方カメラ f=1970 px で約 1 px）
  - x 0.010 m、y 0.013 m、z 0.020 m
- 正しいポーズを渡しても、補正後に同じくらいの誤差（ヨー 0.026°、ピッチ 0.039°、ロール 0.063°）が残る。
- 一番大きいのはロールと z（前後）。val の p90 は、ロール 0.148°、z 0.075 m。

## 1. 結果（ps_s2_own24_rt、本当の推論の経路）

40 フレームずつ、補正後の誤差の絶対値。カメラ座標で、ヨー＝y 軸（下）まわり、ピッチ＝x 軸（右）まわり、ロール＝光軸まわり、x/y/z はカメラ中心のずれ。

| | ヨー | ピッチ | ロール | x | y | z |
|---|---|---|---|---|---|---|
| val 補正前 中央 | 0.189° | 0.323° | 0.276° | 0.102 m | 0.082 m | 0.114 m |
| **val 補正後 中央** | **0.027°** | **0.029°** | **0.065°** | **0.010 m** | **0.013 m** | **0.020 m** |
| val 補正後 p90 | 0.065° | 0.060° | 0.148° | 0.026 m | 0.030 m | 0.075 m |
| val ずれなし 補正後 中央 | 0.026° | 0.039° | 0.063° | 0.009 m | 0.016 m | 0.018 m |
| val ずれなし 補正後 p90 | 0.078° | 0.100° | 0.140° | 0.020 m | 0.033 m | 0.046 m |
| train 補正後 中央 | 0.021° | 0.020° | 0.050° | 0.010 m | 0.013 m | 0.010 m |

API（`/api/eval_frame`）で val の 1 フレームを 3 通りのずれで呼んだ結果（段 1 の重み）:

| 入れたずれ | 補正前（測地 / 距離） | 補正後 |
|---|---|---|
| なし | 0.000° / 0.000 m | 0.143° / 0.043 m |
| (0.3, −0.2, 0.25)° / (0.1, −0.05, 0.15) m | 0.438° / 0.187 m | 0.159° / 0.048 m |
| (−0.45, 0.4, −0.1)° / (−0.18, 0.12, 0.05) m | 0.611° / 0.222 m | 0.104° / 0.037 m |

学習の val の推移（段 2、学習ログの「POSE rot」＝回転の 3 軸の絶対値の平均。測地距離ではない）:

| ep | 回転（3 軸平均） | 並進 | chi2r | σ |
|---|---|---|---|---|
| 1 | 0.098° | 0.045 m | 22719 | 4.5 px |
| 10 | 0.056 | 0.021 | 0.9 | 5.0 |
| 30 | 0.045 | 0.014 | 2.6 | 3.1 |

train のログ（同じ定義）は ep30 で 0.025°。同じ評価コードで train のシーンを測ると 0.036°、val のシーンは 0.046°。

## 2. 直したこと

| 何が問題だったか | 直し方 | ファイル |
|---|---|---|
| セルの代表点（クエリ）を GT の投影で選んでいた（2026-05-31〜）。モデルが座標から答えを読めた | ずらした後の投影（`uv_off_c`）で選ぶ | `datasets/pandaset_full.py` |
| CalibNet2 の Deformable の参照点を `sigmoid(Linear(q))` で作っていて、自分の位置を見ていなかった（学習後も自分から 42〜100 px 離れていた） | `--ref-mode query`：参照点＝各点の uv そのもの。ずれは DA 内のサンプリング位置だけが学ぶ | `models/calibnet2.py` |
| 局所エンコーダが周り 3×3 セルからランダムに 16 点、1 セル最大 8 点 | `--frustum-nb own --k-per-cell 24`（既定）：自分のセルの点を全部、1 セル最大 24 点 | `models/model_depth.py`、`train_cnd2_ddp.py` |
| 学習でも val でもクエリがセル中心に一番近い点だけ | `--rep-strategy random_train`：学習時だけセル内ランダム、val・推論は中心に一番近い点 | `datasets/pandaset_full.py` |
| 格子の窓のうち 1 枚でも点が足りないと、フレームごと引き直していた（学習では黙って別フレームに、推論では "no valid window after 1024 re-rolls" で落ちる） | 足りない窓は有効な窓の複製で埋め、`w_active=0`（BA に参加しない） | `datasets/pandaset_full.py` |
| val に理由の記録なしで `center_band=0.5`（縦の中央 50% の行だけ）が付いていた | 外した。val も画像全体から窓を取る | `train_cnd2_ddp.py` |
| val の窓とずれが毎エポック変わる（段 1 は numpy のグローバル乱数） | `--val-seed`（既定 20261008）で val の `__getitem__` の乱数を固定 | `pandaset_full.py`、`train_cnd2_ddp.py` |
| ClearML のタスク一覧の Iterations が秒数（180 など）になる | 起動直後に iteration 1 で lr を送る | `train_cnd2_ddp.py`、`train_grid_depth.py` |
| ClearML の報告用プロセス（fork）が `Task.init` で止まり、数値が 1 件も届かないことがあった | `~/clearml.conf` に `sdk.development.report_use_subprocess: false`（各マシンで設定） | — |
| "Converting mask without torch.bool dtype" が val のたびに大量に出る | torch 2.0.0 の不具合（bool のマスクでも出る）。この警告だけ表示しない | `train_cnd2_ddp.py` |
| eval が 10 エポックごと | 既定を 5 エポックごとに | `train_cnd2_ddp.py` |
| 誤差が測地距離か 3 軸平均かで混乱 | `pose_error_axes`：ヨー・ピッチ・ロール・x・y・z を出す。API の応答にも `error_before_axes` / `error_after_axes` | `scripts/inference/infer_calib.py`、`services/calib_api/server.py` |

## 3. 学習のしかた

```bash
# 段 1（BA なし）。--frustum-nb own --k-per-cell 24 は既定
python -u datasets/train_cnd2_ddp.py --cache <pandaset_v3_full> \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 40 --epochs 20 \
  --cross-attn deform --ref-mode query --rep-strategy random_train --name <段1の名前>

# 段 2（BA あり）。段 1 の最良の重みから
python -u datasets/train_cnd2_ddp.py <段 1 と同じ引数から --no-with-ba と --oversample を外す> \
  --with-ba --resume-ckpt experiments/<段1の名前>/best_model.pt --start-epoch 0 --epochs 30 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll \
  --ba-warmup-start 0 --ba-warmup-end 10 --name <段2の名前>
```

- 段 2 の実例は `_kick_ps_s2_after_rt.sh`。
- 時間（RTX 3090 1 枚）：段 1 は 1 エポック約 3.2 分、段 2 は約 3.5〜4 分。
- 環境：今回の最後の run は `sam3` 環境（torch 2.11）。`sam3` に libturbojpeg が無いので `TURBOJPEG_LIB=<libturbojpeg.so.0 のパス>` を渡した。API（uvicorn、FastAPI）は `neurad` 環境（torch 2.0）で動かした。torch 2.11 で保存した重みは 2.0 で読める。

## 4. 推論・API の確かめ方

```bash
# 本当の推論の経路で 40 フレーム（外でずらしたポーズを渡す）。ZERO=1 でずれなし
CKPT_EXP=<段2の名前> N=40 python tests/test_infer_pandaset.py

# API
E2E_EXP=<段2の名前> python -m uvicorn services.calib_api.server:app --port 5092
curl -X POST localhost:5092/api/eval_frame -F image=@image.jpg -F points=@points.txt \
     -F calib=@calib.json -F rot_deg='[0.3,-0.2,0.25]' -F t_m='[0.1,-0.05,0.15]'
```

### Web ページで試す

```bash
E2E_EXP=ps_s2_own24_rt python -m uvicorn services.calib_api.server:app --host 0.0.0.0 --port 5092
# ブラウザで http://<host>:5092/calibrate
```

- 「PandaSet の val から選ぶ」：フレーム番号（0〜399）を入れるかランダムに選び、ずらす量（回転 ZYX の度、並進 カメラ軸の m）を入れて「ずらして直す」。キャッシュは `E2E_PS_CACHE`（既定 `/mnt/ssd2t/work/e2e_calib/cache/pandaset_v3_full`）。
- 補正前・補正後の誤差（ヨー・ピッチ・ロール・x・y・z）の表と、画像に LiDAR の点を重ねた図（緑＝正しいポーズ、赤＝ずらしたポーズ、水色＝補正後）が出る。
- 自分のデータ（画像・点群・calib JSON）をアップロードしても同じことができる。
- 例：val #123（シーン 042、フレーム 43）、ずれ 回転 [0.4, −0.3, 0.4]°・並進 [0.1, −0.1, 0.15] m → 補正後 ヨー +0.024°、ピッチ −0.045°、ロール +0.138°、x −0.009 m、y −0.026 m、z +0.011 m。

CLI では、`--rot` と `--t` でずれを軸ごとに指定できる。

```bash
python scripts/inference/infer_calib.py ps_s2_own24_rt --image image.jpg --points points.txt \
    --calib calib.json --rot '[0.3,-0.2,0.4]' --t '[0.05,-0.1,0.15]'
```

### 直す前後（CLI）

PandaSet の val を番号で選び、ずらして直し、重ねた図を保存する。図の左が補正前（赤＝ずらしたポーズ、緑＝正しいポーズ）、右が補正後（赤＝ずらしたポーズ、緑＝正しいポーズ、水色＝補正後。水色を一番上に描くので、緑と重なったところは水色だけ見える）。

```bash
python scripts/inference/infer_calib.py ps_s2_own24_rt --pandaset-val 123 \
    --rot '[0.4,-0.3,0.4]' --t '[0.1,-0.1,0.15]' --overlay out.png
```

| フレーム | | ヨー | ピッチ | ロール | x | y | z |
|---|---|---|---|---|---|---|---|
| val 250（シーン 015/10）ずれ 回転 [−0.45, 0.4, −0.1]°・並進 [−0.18, 0.12, 0.05] m | 補正前 | +0.399° | −0.103° | −0.451° | −0.180 m | +0.120 m | +0.050 m |
| | 補正後 | −0.025° | +0.015° | **+0.098°** | +0.007 m | +0.002 m | −0.011 m |
| val 123（シーン 042/43）ずれ 回転 [0.4, −0.3, 0.4]°・並進 [0.1, −0.1, 0.15] m | 補正前 | −0.303° | +0.398° | +0.398° | +0.100 m | −0.100 m | +0.150 m |
| | 補正後 | +0.024° | −0.045° | **+0.138°** | −0.009 m | −0.026 m | +0.011 m |
| val 250、ずれなし | 補正前 | 0 | 0 | 0 | 0 | 0 | 0 |
| | 補正後 | −0.034° | −0.001° | **+0.087°** | +0.010 m | −0.009 m | −0.020 m |

3 例とも補正後に一番大きく残るのはロール。ずれなしで入れても、ロール 0.087°・z 0.020 m 動く。

val 250、ずれあり:

![](_figs/2026-10-08/cli_val250_shift.jpg)

val 123、ずれあり（API の `/api/pandaset/eval_image` も同じ関数 `render_overlay` で描く）:

![](_figs/2026-10-08/cli_val123_shift.jpg)

val 250、ずれなし:

![](_figs/2026-10-08/cli_val250_zero.jpg)

### 直す前後（API）

```bash
# 前後を重ねた PNG が返る。誤差はヘッダ X-Error-Before / X-Error-After / X-Frame（JSON）
curl -X POST localhost:5092/api/pandaset/eval_image -F i=123 \
     -F rot_deg='[0.4,-0.3,0.4]' -F t_m='[0.1,-0.1,0.15]' -D - -o out.png

# 同じものを JSON で（軸ごとの誤差 error_before_axes / error_after_axes と、点の投影 overlay）
curl -X POST localhost:5092/api/pandaset/eval -F i=123 \
     -F rot_deg='[0.4,-0.3,0.4]' -F t_m='[0.1,-0.1,0.15]'

# 自分のデータで前後の PNG
curl -X POST localhost:5092/api/eval_frame_image -F image=@image.jpg -F points=@points.txt \
     -F calib=@calib.json -F rot_deg='[0.3,-0.2,0.25]' -F t_m='[0.1,-0.05,0.15]' -o out.png
```

## 5. まだ残っていること

- 正しいポーズを渡しても補正後に誤差が残る（ロール 0.063°、z 0.018 m）。
- データセットの候補点の前絞りが、GT の投影 ±64 px で決まっている。ずれが 64 px を超える近い点（並進 0.2 m、深度 5 m で約 79 px）は GT 依存で落ちる。`z > 0.5` の判定も GT のポーズの深度。
- val は 5 シーン（400 フレーム）、train は 34 シーン（2,720 フレーム）。キャッシュに 39 シーンしか入っていない。
- 学習中の train の値は、クエリがランダム・格子がずれた条件で、学習途中の重みのエポック平均。val と同じ条件ではない。
- ClearML の val の可視化は、フレームを全体から等間隔に、窓を点の一番多いものに選ぶよう直した（次の run から）。

## 関連

- 人工データでの切り分け: [2026-10-08_toy-bench-ladder.md](2026-10-08_toy-bench-ladder.md)（Deformable で参照点を自分の位置にすると、格子の点のデータで 7.03 → 1.66 px）

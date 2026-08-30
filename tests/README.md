# システムテスト — 実験の前に必ず通すこと

## 鉄則

**テストを書く → 通ることを確認する → それから実験する。**

2026-08-28 に一晩溶かした原因がこれを守らなかったこと。壊れたデータの上で
Covariance Intersection・変量効果・MINQUE・D-最適実験計画を 5 時間試して、全部無駄だった。
原因は摂動を掛ける座標系が混ざっていたことで、テストを先に書いていれば最初の 10 分で出ていた。

**実験を回す前に必ず流す。PASS しなければ実験してはいけない。**

```bash
CACHE=/path/to/ns_v3_notile python tests/system_test.py
echo $?     # 0 = PASS, 1 = FAIL
```

## 何を保証するか

データローダから GN ソルバーまでの経路を、1 サンプルずつ通して各段を検証する。
モデルは要らない（S8 だけ `CKPT` を渡すと動く）。

| stage | 検証内容 | 合格条件 |
|---|---|---|
| **S1** | 生フレームと GT 投影を画像に描く | 目視（DEBUG に PNG） |
| **S2** | キャッシュの `uv_full` と、`(pts−cam_pos)@R_gt` を投影したものが一致するか | < 0.05 px |
| **S3** | クロップ枠が画像内に収まっているか | 完全に内側 |
| **S4** | 摂動を掛けた投影 | 目視 |
| **S5** | クロップ内の点と、セル代表選択の結果 | 占有セル数 == 代表点数 |
| **S6** | クロップ → 入力解像度のスケール | 窓とオリジナルで一致 |
| **S7** | **6-DOF 復元**: 注入した δ を GT 残差から解き直せるか | rot 0.0000° / t 0.000000 m |
| **S8** | 学習済みモデル（`CKPT` を渡したときだけ） | base より改善 |
| **S9** | **共有 δ の複数窓**: 1 フレームから 6 窓、同じ δ、融合して復元できるか | 各窓・融合とも rot < 0.01° / t < 0.005 m、`Z>0.5` 通過率 > 99% |
| **S10** | **BA loss が本当に G 窓を融合しているか**: `_ba_pose_loss` に入る点数を数える | 融合後の点数 == G × 1窓の点数 |
| **S11** | **スケール不変性**: 256〜512 の違うサイズのクロップが同じポーズに融合するか | 各窓・融合とも rot < 0.01° / t < 0.005 m、`cs` が `max_crop_px` 以内 |
| **S12** | **BA loss は暖機が要る**: 未学習モデルでの BA loss と勾配の大きさを記録 | 有限であること + `frac_pd > 0.5`。値は毎回表示され、暖機の必要性が見える |

### S7 と S9 が本体

S7 は「注入した δ が、データローダが出す `pts_cam_orig` / `duv_orig` / `K_orig` から
そのまま解き直せるか」を見る。ここが通らなければ、学習ターゲットもポーズ loss も無意味。

S9 は「同じ δ を複数窓に掛けて融合したときに、同じ δ が戻るか」を見る。
座標系が混ざっていると窓ごとに違う量になるので、ここで落ちる。

## 過去にこのテストが捕まえたバグ

| バグ | 落ちた stage | 症状 |
|---|---|---|
| `pts_cam_orig` が world 座標 | S9 | 6 窓中 4 窓が nan、残り 2 窓も rot 0.57° / t 2.46 m |
| 摂動の並進が world 系（回転はカメラ系） | S9 | 融合 t err 0.5539 m |
| `Z > 0.5` ガードが world z（高さ）を通していた | S9 | 通過率 54%（正しくは 100%） |
| `u0` にクリップが無い | S3 | クロップが画像外に 12% |
| 混合バッチ分岐が常に発火し 28 窓融合を学習していなかった | S10 | `group=4` なのにソルバーが 1 窓分 (146 点) しか見ていない |
| 深度分岐が `max_crop_px` を無視して 768 まで広げる | S11 | `min=256 max=512` の指定に対し `cs` が 618〜695 px |
| スクラッチに BA loss を掛けて発散 | S12 | 未学習で loss 3.1e4 / 勾配 2.0e7（`tr_nll` が ep18 で 3.1e14 → ep19 で NaN） |

### BA loss は必ず暖機してから

1 バッチでの実測（256 px, group=4）:

| チェックポイント | BA loss | max\|grad\| |
|---|---|---|
| 未学習 | 7.2e+04 | **1.8e+07** |
| `head_ns200`（BA なしで暖機） | 1.3e+04 | 1.0e+06 |
| `cam_fuse28`（BA ありで学習済み） | **−20.5** | **1.1e+03** |

`−½logdet H` は σ を縮める方向に効くので、μ がまだランダムだと二次項が巨大になる。
**`--ba-loss` はスクラッチから掛けてはいけない。** まず BA なしで暖機して resume する。

## 実行結果 (2026-08-29, カメラ系修正後)

```
--- checks ---
  S2_cache_uv      5/5 pass   cached uv vs reprojected: max 0.0085 px
  S3_box_inside    5/5 pass   crop box (863,172,256) inside 1600x900
  S5_reps          5/5 pass   26 reps for 26 occupied cells
  S6_scale         5/5 pass   scale consistent: window 14.732 x 1.000 == orig 14.732
  S7_fit           5/5 pass   residual after one rigid pose: 0.0000 px
  S7_rot           5/5 pass   rotation error vs injected: 0.0000 deg
  S7_xfer          5/5 pass   recovered delta maps GT->perturbed: 0.000000 m
  S7_gtalg         5/5 pass   analytic (R_true,t_true) maps GT->perturbed: 0.000000000 m
  S7_t             5/5 pass   translation error vs injected: 0.000000 m
FAILED: 0

S9  scene-0061 f0  6 windows, shared delta |ypr| 0.5315 deg  |t| 0.2835 m
win  npts   Z>0.5    fit px  rot err deg    t err m
  0   140  100.0%    0.0000       0.0000     0.0000
  1   125  100.0%    0.0000       0.0000     0.0000
  2   120  100.0%    0.0000       0.0000     0.0000
  3   121  100.0%    0.0000       0.0000     0.0000
  4    78  100.0%    0.0001       0.0000     0.0000
  5   142  100.0%    0.0000       0.0000     0.0000
FUSED (info-form, 6 windows): rot err 0.0000 deg  t err 0.0000 m
FAILED: 0

S10  group=4, 149 pts/window -> solver saw 596 (fused)

S12  UNTRAINED model: BA loss 3.139e+04   max|grad| 1.997e+07
     (a trained one gives ~-20 and ~1e3 -- warm-start before enabling --ba-loss)

S11  crops 256..512 -> input 512px, 8 windows, shared delta |ypr| 0.7087 deg |t| 0.2126 m
win   cs  scale  npts   rot err     t err
  0  363  1.410   154    0.0000    0.0000
  1  466  1.099   162    0.0000    0.0000
  2  378  1.354   131    0.0000    0.0000
  3  491  1.043   171    0.0000    0.0000
  4  340  1.506    86    0.0000    0.0000
  5  418  1.225   210    0.0000    0.0000
  6  387  1.323   182    0.0000    0.0000
  7  476  1.076   221    0.0000    0.0000
FUSED across sizes 340..491 px: rot 0.0000 deg  t 0.0000 m

==============================================================
SYSTEM TEST: PASS   (0 failing checks)
==============================================================
```

world 系だった頃は S9 の fit が 0.0005〜0.0027 px 残っていた。カメラ系にしてから
**6 窓すべて 0.0000 px** になっている。

## 回帰確認

`IMG=256 CROP=256` と `IMG=512 CROP=512` の両方で PASS（exit 0）を確認してから実験に進むこと。

```
IMG=256 exit=0  SYSTEM TEST: PASS   (0 failing checks)
IMG=512 exit=0  SYSTEM TEST: PASS   (0 failing checks)
```

## 環境変数

| 変数 | 既定 | 説明 |
|---|---|---|
| `CACHE` | (必須) | v3 LMDB キャッシュのパス |
| `WDIR` | リポジトリ | import 元 |
| `TAG` | `run` | DEBUG 出力のサブディレクトリ名 |
| `IMG` / `CROP` | 256 / 256 | 入力解像度・クロップ辺 |
| `ROT` / `TM` | 0.5 / 0.05 | 摂動の半幅（deg / m） |
| `N` | 5 | ダンプするサンプル数 |
| `CKPT` | — | 渡すと S8（学習済みモデル）も走る |
| `SCRATCH` | `/tmp/e2e_calib_systest` | DEBUG 出力先 |

## テストを足すとき

新しい機構を入れたら、**その機構を保証する stage を先に足す**。
「動いたっぽい」で進めない。落ちる条件（`chk(...)` の第 2 引数）を必ず書く。

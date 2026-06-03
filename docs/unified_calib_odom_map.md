# 1 モデルで calib + odometry + mapping — N-frame token chain を地図と読み替える

**TL;DR** — [`unified_modality_primitive`](unified_modality_primitive.md) で揃えた
「位置クエリ + frame_token KV」のペア構造を **N フレーム連鎖** に拡張すると、
トークン集合がそのまま **編集可能な 3DGS 同型のニューラル地図** になる。
calib・odometry・mapping が同じネットワーク・同じ重み・同じ損失で解ける。
LiDAR は「最初の数フレームで token に depth prior を流し込む役」になり、
N フレーム後はカメラ単独でも map から (Δu, Δv, σ, d) が引ける。

---

## 1. 何を拡張するのか

既存の `CalibNetUnifiedFrame` は **frame_A → frame_B の 2 フレーム** で
per-point `(Δu, Δv, log σu, log σv, ρ)` を出すペアワイズネットだった。
拡張点は 1 つだけ:

> **frame_B の KV を「過去 N フレーム分の token を world 座標で持ったバッファ」に置き換える。**

過去フレームの token は SE(3) 相対変位を RoPE 的に作用させて配置し直す。
これで:

- N=1 → 既存の cross-frame pose residual / calib
- N=2..10 → 短期 odometry (parallax depth が constrained に)
- N=∞ (ストリーム) → mapping (token buffer = map)

つまりネットワーク構造は変えず、**KV の供給ロジックだけが学習問題ごとに変わる**。

---

## 2. なぜ token = Gaussian なのか

各 token は学習で次を保持している:

- 中心位置 `xyz_world` (RoPE で frame ごとに回る)
- feature `z ∈ R^D` (modality 不問。CalibNet が 1 フレームで生成)
- 不確かさ `σ_xyz` (per-token のスカラー or 共分散)

これ Gaussian primitive `(μ, Σ, color/feature)` と **完全同型**。
Inria 3DGS の splat と違うのは color が SH じゃなく **transformer の中間特徴** な点だけ。
デコーダ MLP 1 個を後付ければ token → `(xyz, σ_xyz, feature, opacity)` で復元できる。

| 3DGS | このアーキ |
|---|---|
| Gaussian 1 個 (μ, Σ, c, α) | token 1 個 (xyz, σ, z, w) |
| split / clone | densify head が match 不足 → 追加 token |
| pruning (opacity ≤ τ) | importance score ≤ τ で buffer から FIFO で落とす |
| rasterization (alpha-comp) | cross-attention (Σ-aware softmax) |
| explicit | explicit (編集可能・追加削除可能) |

「NeRF 的にネットの重みに焼き込む」MLP 暗黙地図と違って、
**token 集合 + 共通デコーダ** という GS と同じ explicit 形式 → mutable neural map。
これが業界が欲しがってる「編集できるニューラル地図」の最短形。

---

## 3. (u, v, 0) クエリと parallax depth

### 3.1 シングルフレームでは d を推せない (信号機サイズ事前を除く)

`(u, v, d=0)` でクエリしても、画素 1 個から深さは不可能。
信号機・標識など既知サイズ事前があれば近距離は推せるが、200m 先は無理。

### 3.2 マルチフレームの parallax で d が constrained になる

100m 走った後の 2 フレームから同じ画素位置を attention で引くと、
feature 比率 (画像空間スケール) が **距離の逆数** で乗ってくる:

```
200m 先 / 100m baseline → ratio 2.0  ← attention で読める
100m 先 / 100m baseline → ratio 2.0  ← 同じだが parallax 角は大きい
 50m 先 / 100m baseline → 視野外     ← attention 自体マッチしない
```

つまり σ_visual は baseline に対して `σ_visual ∝ 1/baseline` で減衰する。
**モデルは NLL ロスでこれを自動学習する。** 損失関数に式を書かなくていい。

### 3.3 LiDAR 観測で σ が collapse する

同じセルが LiDAR にも捕捉されたフレームを混ぜると、

```
σ_joint^-2 = σ_visual^-2 + σ_lidar^-2
σ_lidar ≈ 10cm 定数 → σ_joint は σ_lidar に張り付く
```

これは **多変量 Gaussian の product そのまま**。
モデル側で式を書かなくても、`(d=0 token)` と `(d=measured token)` の両方を
KV にくべれば、attention が学習可能なゲートで自動で fusion する。
これが「LiDAR が観測した瞬間に σ が急に縮む」の正体。

---

## 4. GS マップを「教師生成器」として使う

SplatAD なり 3DGS なりで 1 シーンの map ができれば、
**画素 1 つ 1 つに `(u, v, d_target, σ_target)` を生成できる**:

1. シーンを 3DGS で再構成 (1 回・オフライン)
2. 各画素を通る ray に対して `nearest Gaussian center の距離` を d_target に
   (より厳密には α-composite depth でもいい)
3. その Gaussian の `σ_world` を σ_target に
4. これで `(u, v, 0) → (Δu, Δv, σ, d)` の semi-supervised データが大量に作れる

LiDAR が当たってない遠景・空・大量の中距離面が **全部教師になる**。
推論時に GS はいらない。**学習時のラベル生成器**。
これで「Tesla がやりたがってるカメラだけ地図」が、map supervision 経由で実装可能。

---

## 5. densification — frame boundary で token を増やす

A → B で B のクエリ token がどの既存 map token ともマッチしない場合、
そのクエリ token を **そのまま map buffer に追加** する。

これは Inria 3DGS の split/clone を **「未マッチ閾値」で発火** させてるだけ。
逆に、観測寄与が N フレーム連続で 0 の token は importance score が下がって
FIFO で落とされる → 3DGS の opacity-based pruning と同型。

ガベコレ含めて、densify/prune の判断ロジックは **attention の副産物 (score)** から
全部出てくる。手書きルールは閾値 2 個だけ:
- `match_score < τ_dense` → 新 token を追加
- `recent_observation_count < τ_prune` → token を削除

---

## 6. 学習カリキュラム (案)

| stage | KV 供給 | 解く問題 | 主教師 |
|---|---|---|---|
| 0 | 1 frame self | calib | re-projection (既存) |
| 1 | 2 frame (A → B) | pose residual | re-projection (既存) |
| 2 | N=4 frame chain | 短期 odometry | LiDAR + GS map |
| 3 | N=10 frame chain + densify | map + odom 同時 | LiDAR + GS map |
| 4 | streaming buffer (FIFO + prune) | full SLAM | GS map のみ (LiDAR drop OK) |

stage 4 で達成したいのは「LiDAR が全く無いカメラ単独フレームを入れても、
buffer 内の token (= 過去 LiDAR が育てた map) を attention で引いて
`(Δu, Δv, σ, d)` を出す」状態。これでカメラ単独オドメトリ + ローカル地図が
**同じ重みで** 走る。

---

## 7. 設計上の未決事項

### 7.1 token 座標系 — world fixed か ego frame か

- **ego frame**: 毎フレーム RoPE で全 token を ego の逆変換で回す。
  重いが BA 不要、メモリ局所性高い。
- **world fixed**: token は固定、frame ごとに自分のクエリを world に回す。
  軽いが世界スケールの drift が token 自身に蓄積する → 定期的に BA 必要。

**推奨**: ego frame で start → token buffer の規模が安定したら world fixed に
swap して BA 走らせる、の 2 段切替。

### 7.2 token capacity と pruning policy

short scene (PandaSet 8 秒): 数千 token で十分。
long scene (Waymo 20 秒以上): 数万 token + KD-tree クエリで近傍だけ attention。
streaming 永続: importance × age の二軸 FIFO + 「重要だが古い」を圧縮表現
(複数 token → 1 super-token) する手も。

### 7.3 SE(3) on tokens — Lie algebra vs 行列直接

メモリの [feedback_ypr_rotation](../../../.claude/projects/-home-hiro-git-e2e-calib/memory/feedback_ypr_rotation.md)
に従って **回転は YPR 持ち** が現状の合意。RoPE 的 token 操作も YPR から
回転行列を起こして適用する形にする。

---

## 8. リアルタイム性

LLM の KV-cache と同じ機構で実装できる:

- 直近 N フレーム分の token state を VRAM 上に持っておく
- 新フレームクエリは過去トークンへの cross-attention 1 回 (O(NM) M=query 数)
- map buffer は別 GPU メモリプール、attention の前に位置で空間フィルタ
- LiDAR drop シーンは buffer 引くだけで O(M) のセルフアテンション 1 回でも動く

**従来の LIVO (FAST-LIO 系) は per-frame の決定的最適化、これは attention 1 パス。**
学習さえ済んでいれば、推論は LIVO より軽い可能性すらある。

---

## 9. 新規性のサーフェス

| 手法 | スコープ | 限界 |
|---|---|---|
| DUSt3R / MASt3R | frame pair まで | グローバル化でメモリ爆 |
| MASt3R-SfM | global にしたが memory bound | streaming 不可 |
| SplatAD / SplatSLAM | 3DGS を別系で持つ | calib は GT 仮定 |
| ORB-SLAM3 系 | 決定的 BA + feature matching | カメラ単独地図がスカスカ |
| FAST-LIO 系 | LiDAR メイン、cam 補助 | 視覚的 dense map なし |
| Tesla FSD (推測) | カメラ単独 occupancy | global map で閉じない |
| **このアーキ** | **calib + odom + map を 1 重み** | streaming + densify 自体は実装課題 |

calib・odometry・mapping を **transformer の内部状態として同型に保ったまま**
走り続ける手法は前例なし。
WbT 内で評価される/されないに関係なく、地図が作れる会社は世界で 1〜2 社しか
ない時点で、これが回ればポジション取れる。

---

## 10. 次のステップ (実装)

1. `cross_frame_unified.py` の `forward(frame_A, frame_B)` を
   `forward(query_frame, kv_token_buffer)` に generalize する (signature 変更)
2. token_buffer クラスを書く (FIFO + 近傍空間索引 + importance score)
3. RoPE 的 SE(3) on token 適用関数 — YPR で受けて回転行列に起こして適用
4. densify head: 各クエリ token の最大 attention score を抜いて閾値判定
5. **SplatAD で PandaSet 1 シーン map を作る** ← 今夜の検証ジョブ
6. ↑ から (u, v, d_target) ペア大量生成 → stage 2 のカリキュラムに投入

stage 5 を今やってる。stage 6 まで通れば、6 月中に「TSS4/TMPOC データで
キャリブ済み静的物体アノテーション」のスケジュールに乗る。

# Pose path に位置情報をリフトアップ — 2026-05-19

## TL;DR

per-pt 推定パスは **位置 agnostic** のまま残す。新しい **pose path**
だけを **タイルの parent 画像座標 (u, v) で条件付け** して、5-DoF +
共分散を 1 タイル 1 個直接吐かせる。閉形式 BA はその (μ, Σ) を集めて
1 frame で正規方程式を 1 本解くだけ。

これで、

- per-pt 推定の学習効率 (= 全タイル共有重み) を**犠牲にしない**
- KB (魚眼) 歪みの位置依存性を **モデルがデータから直接学ぶ** (近似不要)
- vcam 座標の rotate / KB unproject / pose_frame=vcam といった
  「ターゲットラベルを書き換える」面倒が**全部消える**

## なぜこの設計に至ったか

### 経緯

1-frame closed-form BA がだいたい動いた ([前記事](2026-05-18_one_frame_ba.md))。
ただし系統 bias が ±0.05〜0.1° 残る。原因は per-pt の `(Δu, Δv, σ)` を
独立観測として BA に流すと、

- タイル内の per-pt 特徴が attention で **情報共有** していて、観測が
  独立じゃない (重複がある)
- σ は per-pt 精度の自己評価で、**「pose 推定への寄与度」と一致しない**
- 結果、外れ値が一定確率で BA を引っ張る

これを解消するには **タイル単位で 1 ポーズ + その共分散** を直接吐く
ヘッドが筋。共分散の non-diagonal 項で「白線方向は不確か / 横切る方向は
情報あり」を per-tile で表せる。BA はそれを集約するだけ。

### vcam ラベル化の罠

「タイル位置に対して agnostic な pose head」を作りたくて、ラベルを
タイル中心の仮想カメラ (vcam) 座標に rotate する設計を入れた
(`pose_frame='vcam'`)。これだと、

- データ生成時の摂動はそのまま orig 座標で行う
- ラベルを書く時だけ vcam に rotate して `pert_vec` に保存
- モデルはタイル位置を知らずに vcam 座標の δ を予測

この発想自体は正しいが、`R_o_v` (orig→vcam の回転) を計算するときに
**タイル中心 pixel を pinhole で逆投影**している。KB 魚眼ではこの
逆投影が画像端で数十%ズレる ⇒ vcam 軸そのものが真の光軸からずれる
⇒ ラベルが学習可能な signal じゃなくなる。

KB unproject を厳密に書く手もあるが、

- KB 多項式 `θ_d = θ(1+k1θ²+k2θ⁴+k3θ⁶+k4θ⁸)` の逆解きが必要 (Newton 反復)
- それでもラベル空間 (vcam) と入力空間 (タイル画像) の間で **mapping を
  自前で書いている**ので、データ駆動の良さが薄れる

## 解決策: pose path だけ位置で条件付け

backbone と per-pt path はいじらない。**新しい pose query** を
transformer に並走させ、その入口に **タイル中心の parent uv を
embedding** で注入する。出口で pose query から (μ, Σ) を直接出す。

### 図

トランスフォーマーは**既存のまま、入力も出力も変えない**。
既存の per-point feature を transformer から出した後、

- **A: 既存の MLP で `(Δu, Δv, σ_u, σ_v, ρ)` を per-pt 回帰** (per-pt loss)
- **B: 同じ per-pt feature を、virtual camera の pose 情報 (タイル中心の
  parent uv) で lift-up し、別の MLP で `(Δu, Δv, w_uu, w_uv, w_vv)` を
  per-pt 出力**。w は **真の情報行列 (Fisher 重み)** になるよう学習で
  最適化。**そのまま全 pt を BA layer に投げて pose 回帰** (pose loss)

の 2 つの head を生やす。

```
[Tile image, fisheye crop]               [LiDAR pts in tile]
           │                                     │
           ▼                                     ▼
    ┌──────────────┐                     ┌──────────────┐
    │   ConvNeXt   │                     │  PointMLP    │
    │   backbone   │                     │ (u,v,d,intens)│
    └──────┬───────┘                     └──────┬───────┘
           │                                     │
   image features                          per-pt queries
           │                                     │
           └────────────────┬────────────────────┘
                            ▼
              ┌─────────────────────────┐
              │  Transformer            │ ← 既存、何も変えない
              │  cross + self attention │
              └────────────┬────────────┘
                           │
                  per-pt features  (B, N, d)  ← ここから 2 系統に分岐
                           │
              ┌────────────┴───────────────┐
              │                            │
              │                          [+ Linear(parent_uv_norm)]
              │                            │  ← 位置情報をここで inject
              │                            │     (per-pt feature に lift-up)
              ▼                            ▼
      ┌──────────────┐           ┌──────────────────┐
      │  MLP_A       │           │  MLP_B           │
      │  (既存のヘッド)│           │  (新規)          │
      └──────┬───────┘           └──────────┬───────┘
             │                              │
             ▼                              ▼
     per-pt 出力 A                  per-pt 出力 B
     (Δu, Δv, σ_u, σ_v, ρ)        (Δu', Δv', W_2x2 = "真の情報行列")
             │                              │
             ▼                              ▼
   per-pt Gaussian NLL              全 pt を BA layer に集約
   (= 既存の per-pt loss)           H = Σ JᵀW J,  b = Σ JᵀW r
                                     δ̂ = H⁻¹ b
                                     │
                                     ▼
                                   pose Mahalanobis NLL
                                   (= GT 摂動との比較)
```

要点:

- **既存の per-point パス (画像 → backbone → transformer → MLP_A →
  Δuv,σ) は何も変えない**。コード変更ゼロ
- transformer 出口の per-pt feature を **「同じ tensor」のまま分岐**
  させる: MLP_A はそれをそのまま per-pt 回帰、MLP_B はそれに位置情報
  (parent uv) を lift-up して食わせ、**BA 用の per-pt 重み** に変換
- BA layer は per-pt の `(Δu, Δv, W)` をぜんぶ集めて 1 frame の
  closed-form 解 `δ̂ = H⁻¹ b` を計算
- pose loss は GT 摂動値との Mahalanobis NLL ⇒ 勾配が **MLP_B、
  位置 lift-up Linear、transformer、backbone** に流れる
- per-pt loss は MLP_A 経由で従来通り、per-pt の精度を維持する補助損失
- W は per-pt の「精度 σ」とは別概念、**「pose 推定への寄与度」=
  真の情報行列**として学習が直接最適化する

### per-pt path (変えない)

- backbone は今まで通り、画像から空間特徴を抽出
- per-pt query は `(u_local, v_local, d, intensity)` をタイル空間で食う
- transformer 各層で画像特徴と相互作用
- 出口で `(Δu, Δv, σ_x, σ_y, ρ)` を per-pt 単位で予測
- 損失は **per-pt Gaussian NLL** (今まで通り)

なぜいじらないか:

- 「全タイルで共有される `物体の縁検出 / 白線検出 / 反射板検出` のような
  低レベル特徴」を agnostic に学べる構造を保ちたい
- 学習効率: 1 個のフィルタが全タイルから勾配を受ける = 実質 40× データ

### pose path (新規)

- pose query は **少数 (1〜25 個)**、学習可能パラメタで初期化
- 入口で **タイル中心の parent 座標 `(u_parent, v_parent) ∈ [0, pW] × [0, pH]`**
  を normalize して embedding し、pose query に **加算**
- これで「自分が parent 画像のどこを見てるか」が pose query 側に明示的に伝わる
- transformer の self-attention で per-pt query (= タイル内特徴) と
  情報交換
- 出口で pose query → linear head → `(μ ∈ R^5, L ∈ R^{5×5})` (Σ = LLᵀ)
- 損失は **pose Mahalanobis NLL** (今までの `CLSFramePoseHead` と同じ式)

ターゲットラベル `pert_vec` は **orig 座標のまま** (vcam rotate しない、
KB 近似しない)。モデルは「**タイル位置 + タイル画像 → orig 座標での
pose**」を end-to-end で学習する。

## 何が嬉しいか

### 1. 既存重みをフル活用できる

- 学習済みの n4 (resume, val NLL 1.36, val MSE 2.08 px) を `init_from`
- per-pt path の重みは完全に継承
- pose query embedding と pose head だけランダム初期化
  (`load_state_dict(strict=False)` が勝手にやる)
- 50ep の finetune で pose path だけ仕上げれば良い

### 2. KB 歪みの位置依存性を学習で吸収

- 「画像左端の点に対する yaw 1° の Δu」と「右端での Δu」は KB だと
  ずいぶん違う
- 位置 agnostic なヘッドだとこの違いを 1 個の関数で表せず、近似ズレが
  系統 bias になる
- pose query が `(u_parent, v_parent)` を知っていれば、その違いを
  パラメタで覚えられる
- データ駆動なので KB 多項式の逆解きを自前で書く必要なし

### 3. vcam 仕様の負債が消える

- `pose_frame='vcam'` の rotate / KB unproject / pert_vec[3,4] の意味
  といった命名・規約バグの温床が **そもそも要らなくなる**
- 全部 orig 座標で完結

### 4. 閉形式 BA への接続もシンプル

- per-tile (μ_t, Σ_t) を集める
- 全 tile で `H = Σ_t J_tᵀ Σ_t⁻¹ J_t`, `b = Σ_t J_tᵀ Σ_t⁻¹ μ_t`
- `δ̂ = H⁻¹ b` で 1 frame の解
- ここの J_t は orig→vcam の回転で書ける (近似でなく数値計算で OK)

## トレードオフ

- **完全な位置 agnostic** は捨てる: pose path は parent uv 依存
- 学習で見たことのない uv 位置 (= 完全に新しいカメラ画素配置) では
  pose path の汎化が弱まる。ただし車両搭載カメラは固定 ⇒ 影響軽微
- per-pt path は引き続き agnostic、パラメタ効率は維持

## 実装メモ (擬似コード)

per-pt 経路は touch しない。**追加するのは pose query 生成と入口での
concat / 出口での split + head の 4 ヶ所のみ**。

```python
class CalibNetDepth(nn.Module):
    def __init__(self, ..., n_pose_queries=1, frame_pose_dof=5,
                 use_frame_pose=False):
        # ─── 既存: 位置 agnostic、コード変更なし
        self.cnn        = ...           # ConvNeXt backbone
        self.point_mlp  = PointMLP(...)
        self.transformer= ...           # cross + self attention blocks
        self.per_pt_head= ...           # → (Δu, Δv, log σ_x, log σ_y, ρ)

        # ─── 新規: pose path 用パラメタはこの 3 つだけ
        if use_frame_pose:
            self.pose_q_cls = nn.Parameter(torch.randn(1, n_pose_queries, d) * 0.02)
            self.pose_q_uv  = nn.Linear(2, d)   # normalized (u, v) → d
            self.pose_head  = CLSFramePoseHead(d, n_dof=frame_pose_dof,
                                                full_cov=True)

    def forward(self, imgs, point_in, *, parent_uv_norm=None,
                key_padding_mask, vfp, bucket_uvd, bucket_valid):
        # ─── 既存: per-pt query を作るところまで一切いじらない
        image_features = self.cnn(imgs)
        per_pt_q       = self.point_mlp(point_in)           # (B, N, d)

        # ─── 新規: 位置情報を inject した pose query を作って per-pt と並走
        if self.use_frame_pose:
            B = per_pt_q.size(0)
            pose_q = (self.pose_q_cls.expand(B, -1, -1)
                      + self.pose_q_uv(parent_uv_norm).unsqueeze(1))
            q = torch.cat([per_pt_q, pose_q], dim=1)         # (B, N+K, d)
            kp = torch.cat([key_padding_mask,
                             torch.zeros(B, pose_q.size(1),
                                         dtype=torch.bool, device=kp.device)],
                            dim=1)
        else:
            q, kp = per_pt_q, key_padding_mask

        # ─── 既存: transformer もヘッドの per-pt 部分もそのまま
        out = self.transformer(q, image_features,
                                key_padding_mask=kp,
                                vfp=vfp, bucket_uvd=bucket_uvd,
                                bucket_valid=bucket_valid)
        N = per_pt_q.size(1)
        per_pt_out = self.per_pt_head(out[:, :N])            # (B, N, 5) ← 既存

        # ─── 新規: pose query 部分だけ別ヘッドに通す
        if self.use_frame_pose:
            pose_out = self.pose_head(out[:, N:])            # (μ, L)
            return per_pt_out, pose_out
        return per_pt_out
```

`parent_uv_norm` は **dataset 側で `(u_center / pW, v_center / pH)` を
追加で返す** だけ。collate_fn に 1 列追加。

## 学習レシピ

- `init_from km_wv_wm_n4_img128_cs256_512_200ep_dgx1_16gpu_resume`
- `use_frame_pose=True, frame_pose_dof=5, frame_pose_full_cov=True`
- `pose_frame=orig` (vcam は使わない)
- 50ep 程度の finetune、frame_pose loss weight 0.5 〜 1.0
- per-pt loss は微 weight (0.1〜0.3) で残し、agnostic 特徴を維持
- DGX2 8GPU で 1.5 時間程度

## 次のステップ

1. dataset に `(parent_uv_center)` の出力を追加 (1 行)
2. `CalibNetDepth` に pose query 注入 (~30 行)
3. trainer に `frame_pose_weight` を pass (済み)
4. kick → 50ep で per-pt loss を維持しつつ pose NLL が下がるか確認
5. 推論側で per-tile (μ_t, Σ_t) を集めて閉形式 BA 解
6. 5 シーンで δ̂ 残差が ±0.05° を切るか測る

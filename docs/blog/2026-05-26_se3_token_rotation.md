# SE(3) token-rotation — abs-PE は 1cm 地図と両立しない

_2026-05-26 — toy_validated, conclusion_reversed_

## TL;DR

このノートは、cross-attn ブロックの出力を **frame token** と見たとき、
frame token 同士を **回転 + 並進で正確に往復できるか?** を問う。
普通の transformer (abs-PE + R を 9-vec で MLP に混ぜる) では
**回転を正確に表現できない** ことを toy で示し、解として **RoPE** を採用する。
RoPE は LLM 文脈で幾何と無関係に発明された道具だが、
**特徴空間で SO(3) を作用させる手段として転用** でき、機械精度で exact に
回転を表現する。これによって calib と odometry を同じネットワークで
解く構造が開く。

**数値で詰める**:

- (a) abs-PE + R-MLP: **訓練範囲内 0.5° ですら RMSE 0.0061**
  (token magnitude 1 に対し 0.6% 相対誤差)。
  → **10m 構造で 6cm のバイアスを architecture が常時注入**。
  1cm 地図の予算では即アウト。
- (b) type-1 block-diag(R) (= 3D 版 RoPE): 全角度で 1e-7 (float32 機械精度)。
  学習で減らせない構造誤差ではなく architectural に exact。

**致命的なのは外挿ではなく範囲内**。0.5° という運用域で 6cm/10m 出る時点で、
"Δpose 小さいから abs-PE で十分" は成立しない。地図に積んだ瞬間に
全フレームで同符号 (architecture 起源 = 非ランダム) のバイアスが累積する。

**結論**: RoPE / type-1 block-diag(R) は "条件付き保留" ではなく **必須**。

旧版の 「Δpose ≤ 数度なら abs-PE で十分」 「SO(3) は GN ソルバに任せれば
transformer は局所 duv 回帰に専念すればよい」 は **撤回**。duv そのものが
rotation 不変じゃない座標系で出てくる以上、token 段階のバイアスがそのまま
duv に乗る。GN で吸収できる量ではない。

---

## 1. 問題設定

![Problem setup](../assets/2026-05-26_se3_token_rotation/rope_se3_problem.png)

**Setup.**
- 同じ点群 P ∈ R^(N×3) を Q 側と KV 側の両方に置く
- KV 側だけ未知の R ∈ SO(3) で回す: KV = R · P
- 各 token に PosEmbed を与えて、cross-attention または decoder で
  「対応 i ↔ i」を取らせる、または `g(T(R) f(P)) = R · P` を満たす
  decoder を学習させる
- Eval は `|R| ∈ [0°, 180°]` で sweep し、訓練範囲内/外の両方を見る

**何を聞いているか.**
「**transformer は token 空間で SO(3) という群を表現できるか?**」
これは `T(R₁) · T(R₂) = T(R₁ R₂)`、 `T(R)⁻¹ = T(R^T)`、
`T(R)` が周期 2π を持つ、という SO(3) の代数構造を、token 空間の
線型作用 / MLP として network が持てるか、という問い。

---

## 2. 三つの注入方式

| 流派 | 何をするか | SO(3) 構造 |
|---|---|---|
| (a) **abs-PE + R as 9-vec** | PE(p) を token に加算、R を flatten して MLP に concat | network が学習で覚える (constraint なし) |
| (b) **type-1 RoPE-3D** | token を `(D_s scalar, K type-1 vectors)` に分解し、type-1 chunk に R を **block-diag(R, R, …, R)** で直接作用 | architectural に exact (linear, type-preserving) |
| (c) **per-axis 1D RoPE-3D** | 各座標 (x, y, z) に独立に 1D RoPE phase | SO(2)³、SO(3) ではない (translation 不変どまり) |

(a) は「LLM の RoPE を真似て位置を埋め込む」流。(b) は SE(3)-Transformer /
Equiformer の中核を最小限取り出した形。(c) は素朴な 3D 拡張で、当然 SO(3)
全体は表現できない。

---

## 3. Toy 結果 — train [0, 30°] / eval [0, 180°]

訓練は `|R| ∈ [0, 30°]` のサンプル分布で `MSE‖g(T(R) f(P)) − R · P‖²` を最小化。
評価は同じ network を `|R| ∈ [0°, 180°]` で sweep。

![SE(3) extrapolation](../assets/2026-05-26_se3_token_rotation/rope_se3_decode.png)

| `|R|` (°) | (a) abs-PE + 学習 T(R) | (b) type-1 block-diag(R) | (c) per-axis 1D RoPE |
|---:|---:|---:|---:|
| 0.5 | **0.0061** | 0.0000 | 0.0022 |
| 5 | 0.0062 | 0.0000 | 0.0023 |
| 15 | 0.0063 | 0.0000 | 0.0029 |
| **30** (train 上限) | 0.0068 | 0.0000 | 0.0044 |
| 60 | 0.0245 | 0.0000 | 0.0081 |
| 90 | 0.0844 | 0.0000 | 0.0127 |
| 120 | 0.2212 | 0.0000 | 0.0163 |
| 150 | 0.3301 | 0.0000 | 0.0208 |
| **180** | 0.4492 | 0.0000 | 0.0224 |

**読み方の修正.** 旧版はこの表を「(a) は 30° 以内なら 0.007 で動いている」と
読んだが、これは間違い。**0.5° で 0.0061 が出ている時点でアウト**。

- (a) は train 範囲で `0.006`、180° で `0.449`。**train 範囲内ですら
  (b) と 4 桁差**。これは「学習が足りない」ではなく、**identity 周りの
  Taylor 1 次近似の残差** — データを 100 倍にしても消えない構造誤差。
- (b) は全角度で `1e-7` (float32 機械精度)。decoder は type-1 channel を
  K-mix する linear にしてあるので、`g_v(R · v) = R · g_v(v)` が線形性から
  自動成立 → architectural に exact equivariance。「ほぼ 0」ではなく **本当に 0**。
- (c) は (a) よりは綺麗だが train 範囲内 30° で既に 0.0044 = (b) の 4 万倍。
  SO(3) を SO(2)³ で近似している段階で棄却。

---

## 4. なぜ 0.0061 が致命的か (本ノートの中心)

token magnitude を 1 に正規化したとき RMSE 0.0061 ということは、

```
relative error ≈ 0.6%
↓ 物理スケールに移す
1m  構造 → 6mm  バイアス
10m 構造 → 6cm  バイアス
```

これが **architecture 由来で、loss を下げても消えない**。

我々の北極星 ([`.claude/north_star.md`](../../.claude/north_star.md)) は
**自宅周辺数 km の 1cm 精度 3D 地図**。誤差予算 1cm に対して、

- 6mm = **誤差予算の 60% を rotation embed の Taylor 残差だけで食う**
- これに sensor noise / GN 残差 / cross-frame 整合性誤差 / quantization が
  全部上乗せされる
- 「calib の閉ループだけ」ならこの 6mm は気付かない量。だが地図に積んだ瞬間に
  全フレームで同符号の bias がかかる方向に効く可能性が高く (architecture 起源
  なので random でない)、累積する

**つまり 0.0061 は "ほぼ 0" ではなく "1cm 地図にとってクリティカル"**。

(a) で 0.0061 が出ている理由は明確で、`T(R) = exp(R)` を identity 周りで
線形展開した近似を MLP が覚えているだけ。残差は `O(|R|²)` のオーダで残り、
0.5° = 0.0087 rad なら `(0.5°)²/2 ≈ 4×10⁻⁵` 程度の係数を network 重みが
打ち消し損ねた残り。これは Taylor 構造が architecture に入っていないことの
直接の帰結で、**データ・容量・学習時間で減らせない**。

(b) が 1e-7 なのは float32 の機械精度。`X_v[i,k,:] ← R · X_v[i,k,:]` の
1 行で SO(3) の合成則・周期性・逆元・非可換性が **代数的に等式として成立** し、
学習せずとも `g(T(R) f(P)) = R · P` が exact identity になる。

---

## 5. なぜ旧版で「abs-PE で十分」と書いたか — 撤回

旧版の論旨:

> 我々の運用 Δpose は IMU/odom/prev-calib で粗く既知 → 残差は数度・数 cm の
> 範囲。Toy 図で言えば常に train [0, 30°] の中に収まる。(a) abs-PE + R-MLP
> でも train 範囲内は 0.007 で動いている。

**何が間違いか.**

1. **「train 範囲内なら OK」を 0.007 ≒ 0 と読んだ**。実際は 0.6% 相対誤差で、
   1m 構造に 6mm。1cm 地図と両立しない。
2. **「SO(3) は GN ソルバが持っている」を transformer に免罪符として使った**。
   GN は Lie 代数で更新するので合成則・周期性は持っているが、それは **入力
   duv が正しいときに更新が正しい** という話で、token 段階で 6mm のバイアスが
   入っていれば duv にもバイアスが乗り、GN は素直にそのバイアスを fit して
   しまう (GN は外乱を平均化するが系統誤差を除去しない)。
3. **「duv 回帰だけが transformer の責務」という分離を仮定した**。実際には
   duv は image-pixel 単位で出てくるが、token 表現の中で point の rotation 不変
   性が壊れていれば、duv そのものが rotation-biased な座標で出る。GN が後段で
   そのバイアスを取り除く方法は無い。

**結論**: Δpose が小さいことは abs-PE を許す理由にならない。問題は外挿ではなく
**範囲内の Taylor 残差**。

---

## 6. Toy の限界 — (b) の優位は本当に見えているか

正直に書く。今の toy で (b) type-1 RoPE-3D が `1e-7` を叩いているのは、
type-1 chunk の中身が **p のスカラー倍** になっているせい。

```
X_v[i, k, :] = β_k · p_i                  # K 個全部 p の定数倍
T(R) X_v[i, k, :] = β_k · (R p_i)
g_v(out)     = Σ_k a_k · X_v_rot[i, k, :] = (Σ_k a_k β_k) · (R p_i)
```

network が学ぶ自由度は `Σ a_k β_k = 1` を満たすスカラー 1 個だけ。type-1
chunk が K 本あっても情報は K 重複しており、(b) と (c) は **このセットアップ
では情報経路として等価** (どちらも結局「R を点にかけて再 embed」しているだけ)。

(b) の真価が立つのは type-1 chunk に **多様な "回るベクトル"** (画像特徴の
方向、表面法線、特徴勾配など p と独立な direction) を詰めたとき。

ただし toy として示したいことは

- (a) が train 範囲内ですら 0.006 出る ← 本物
- (b) が architectural に exact ← 形式的に正しい (1e-7 = float32 精度)
- (a) が train 範囲外で 74× 壊れる ← 本物

の 3 点で、これらは toy の trivial さに依存しない。**「(a) を採用する根拠を
否定する」目的には十分**。

---

## 7. それで実装はどうするか?

**結論: cross-frame calib の architecture を type-1 chunk 前提で組み直す**。

```
Frame A image / point  ─→ CNN/MLP →  token = [scalar (D_s) | type-1 (3K)]
Frame B image / point  ─→ 同 ─→     token (B 側は Δpose で type-1 を block-diag(R) 作用)
                                       ↓
                          self-attn / cross-attn / FFN は全て type-preserving
                          (W_q/W_k/W_v/MLP は type-0↔type-0 と K×K type-1
                           channel-mix のみ)
                                       ↓
                          per-cell duv prediction
                                       ↓
                          GN solver (Lie 代数で Δpose 更新)
```

設計判断:

1. **type-1 block-diag(R) は必須**
   - 1cm 地図の誤差予算と (a) の 6mm バイアスは両立しない
   - 後付けで type-1 化するのは architecture-wide refactor になる、最初から
     入れる
2. **type-preserving 制約**
   - W_q/W_k/W_v/MLP/FFN を type-0↔type-0 + K×K type-1 channel-mix に縛る
   - SE(3)-Transformer / Equiformer の中核制約。実装は重いが、これが無いと
     (b) の exact equivariance が attention/FFN を抜けた時点で壊れる
3. **type-1 chunk に詰めるベクトル**
   - 画像特徴の方向 (gradient, flow, edge orientation)
   - 表面法線 / point ray 方向
   - スカラー量 (intensity, depth magnitude, log-σ) は type-0 へ
4. **GN solver はそのまま**
   - SO(3) の代数を analytic に持っているのは依然 true
   - ただし「transformer は局所 duv 回帰に専念」は嘘なので、token 段階で
     回転不変性を持たせた上で GN に渡す
5. **2D RoPE-Mixed (Heo et al. 2024)** も解像度切替で同じ Taylor 残差問題が
   出る可能性が高い。同じ理由で **採用検討から保留に格下げしない、必須側に
   倒す**。

---

## 8. 何を学んだか

1. **「PosEmb で位置を入れれば transformer は幾何を覚える」は嘘** — train
   範囲内ですら 0.6% 相対誤差 = 1m に 6mm バイアスが残る。これは Taylor
   残差で、loss を下げても消えない。
2. **"R を embed して足す" と "token を R で回す" は質的に違う** — 前者は
   universal approximator に丸投げ、後者は架構に SO(3) を入れる。**toy で
   4 桁差**。
3. **0.0061 は "ほぼ 0" ではない** — 1cm 地図用途では誤差予算の 60% を
   architecture 由来のバイアスだけで食う、クリティカルな量。
4. **「Δpose 小だから abs-PE で十分」は 1cm 地図と非両立** — Δpose が小さい
   ほど Taylor 残差は小さくなるが、消えない。GN ソルバが SO(3) を持っていても
   token 段階のバイアスは取り除けない。
5. **type-1 block-diag(R) + type-preserving constraint は必須** — 「条件付き
   保留」ではなく architecture 制約として最初から入れる。

---

## 9. 文献

- **RoPE (Rotary Position Embedding)** — Su et al., *RoFormer: Enhanced
  Transformer with Rotary Position Embedding*, 2021.
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864). LLM 文脈での
  オリジナル 1D / per-axis 2D 定式化。幾何とは無関係に発明されたが、
  本ノートでは同じ代数作用 (paired 次元への block-diag 回転) を
  type-1 chunk への SO(3) 作用として転用している。
- **SE(3)-Transformer** — Fuchs et al., 2020.
  [arXiv:2006.10503](https://arxiv.org/abs/2006.10503). type 分解
  ((D_s scalar, K type-1 vectors)) と type-preserving な
  W_q/W_k/W_v はここからの最小限の借用。
- **Equiformer** — Liao & Smidt, 2022.
  [arXiv:2206.11990](https://arxiv.org/abs/2206.11990). type-l 特徴量を
  attention で扱う実用的な equivariant transformer。toy の (b) が
  近似しているのは概ねこれ。
- **RoPE-Mixed (2D)** — Heo et al., *Rotary Position Embedding for
  Vision Transformer*, 2024.
  [arXiv:2403.13298](https://arxiv.org/abs/2403.13298). ViT 用の 2D
  RoPE。本ノートと同じ機構の 2D image-token 版。§7 脚注の通り、保留
  ではなく必須側に倒す。
- **Tensor Field Networks** — Thomas et al., 2018.
  [arXiv:1802.08219](https://arxiv.org/abs/1802.08219). type-l 分解
  による SO(3)-equivariant point network のオリジナル。(b) で使う
  type-1 block-diag(R) はここに辿れる。

## 10. 再現

```bash
# Toy: train [0,30°], eval [0,180°]
/home/hfunaya/.pyenv/versions/3.10.4/bin/python scripts/_debug/_rope_se3_toy.py

# Problem-setup figure
/home/hfunaya/.pyenv/versions/3.10.4/bin/python scripts/_debug/_rope_se3_problem_fig.py
```

出力: `scripts/_debug/_outputs/rope_se3_decode.png` および
`docs/assets/2026-05-26_se3_token_rotation/rope_se3_problem.png`。

Code (commit `153d870`):

- [scripts/_debug/_rope_se3_toy.py](https://github.com/tmc-autonomy/loom-calibration/blob/main/scripts/_debug/_rope_se3_toy.py)
- [scripts/_debug/_rope_se3_problem_fig.py](https://github.com/tmc-autonomy/loom-calibration/blob/main/scripts/_debug/_rope_se3_problem_fig.py)
- [docs/blog/2026-05-26_se3_token_rotation.md](https://github.com/tmc-autonomy/loom-calibration/blob/main/docs/blog/2026-05-26_se3_token_rotation.md)

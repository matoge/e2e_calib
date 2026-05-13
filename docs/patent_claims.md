# 特許請求の範囲 / Patent Claims

発明: **マルチモーダル統合前の残留不確実性予測に基づく決定論的トークン凝縮アーキテクチャ**
Invention: *Deterministic Token Condensation Architecture Based on Predicted Residual Uncertainty Prior to Multimodal Fusion*

---

## 日本語版クレーム

### 【請求項1】（独立項：基本構成）

第1のセンサモダリティと第2のセンサモダリティから取得されたデータを統合して推論を行う情報処理方法であって、

(a) 前記第1のセンサモダリティから抽出された複数の局所特徴量を取得するステップと、

(b) 平均推定値と不確実性指標とを出力する予測ヘッドを用いて、前記複数の局所特徴量の各々について、前記第2のセンサモダリティとの統合処理後にもなお残存すると予測される残留不確実性を算出するステップであって、後続のステップ(c)における選別は、前記平均推定値を用いず前記残留不確実性のみに基づいて行われる、ステップと、

(c) 前記残留不確実性が高い特徴量を優先的に含むサブセットを、決定論的手順により前記複数の局所特徴量から選別するステップと、

(d) 前記第2のセンサモダリティとの統合処理を、前記選別されたサブセットに対してのみ実行し、選別されなかった局所特徴量に対しては前記統合処理を実行しないステップと、

を備える情報処理方法。

---

### 【請求項2】（決定論的多様性サンプリング）

請求項1において、前記ステップ(c)の決定論的手順は、

- 既に選別された特徴量からの空間距離または特徴空間距離の最小値と、
- 当該特徴量の前記残留不確実性と、

の積または関数を最大化するように、特徴量を反復的に選別する手順であることを特徴とする、情報処理方法。

---

### 【請求項3】（WFPSの具体的なスコア定義）

請求項2において、前記反復的選別における第 i 特徴量の選別スコア S_i が、

$$S_i = \sigma_i^2 \times \min_{j \in \text{Selected}} \| p_i - p_j \|$$

（σ_i² は第 i 特徴量の残留不確実性、p_i は第 i 特徴量の位置、Selected は既に選別された特徴量のインデックス集合）

として定義されることを特徴とする、情報処理方法。

---

### 【請求項4】（学習スキーム：Aleatoric Loss）

請求項1から3のいずれか一項において、前記予測ヘッドは、

$$L_{aleatoric} = \sum_i \left( \frac{\|y_i - \hat{y}_i\|^2}{2\sigma_i^2} + \frac{1}{2}\log\sigma_i^2 \right)$$

の形式の損失関数を用いて、前記ステップ(c)の選別結果と独立に、前記複数の局所特徴量の全てに対して教師あり学習されることを特徴とする、情報処理方法。

---

### 【請求項5】（Straight-Through Estimator）

請求項1から4のいずれか一項において、前記情報処理方法の学習時において、

- 順伝播時には前記ステップ(c)で選別されたサブセットのみを後段に出力し、
- 逆伝播時には前記統合処理後の損失からの勾配を、前記選別されたサブセットの重みを介して前記予測ヘッドへ伝播させる、

ことを特徴とする、情報処理方法。

---

### 【請求項6】（第1モダリティ＝LiDAR、第2モダリティ＝カメラ）

請求項1から5のいずれか一項において、前記第1のセンサモダリティがLiDARセンサであり、前記第2のセンサモダリティがカメラセンサであることを特徴とする、情報処理方法。

---

### 【請求項7】（マルチセンサ拡張）

請求項1から5のいずれか一項において、前記第1のセンサモダリティおよび前記第2のセンサモダリティは、LiDAR、カメラ、レーダー、サーマルカメラ、IMUからなる群より独立に選択されることを特徴とする、情報処理方法。

---

### 【請求項8】（ハードゲートによる計算削減）

請求項1から7のいずれか一項において、前記ステップ(d)の統合処理は Cross-Attention 演算を含み、選別されなかった局所特徴量に対する Cross-Attention の計算が物理的に実行されないことにより、前記Cross-Attentionの演算量が O(N²) から O(K²)（ただし K は前記サブセットの要素数、N は前記複数の局所特徴量の要素数、K << N）に削減されることを特徴とする、情報処理方法。

---

### 【請求項9】（情報処理装置）

請求項1から8のいずれか一項に記載の情報処理方法を実行するように構成された、特徴抽出部、予測ヘッド部、決定論的選別部、および統合処理部を含む情報処理装置。

---

### 【請求項10】（プログラム）

請求項1から8のいずれか一項に記載の情報処理方法を、コンピュータに実行させるためのプログラム。

---

## English Version Claims

### Claim 1 (Independent — Core Architecture)

A method for information processing that integrates data acquired from a first sensor modality and a second sensor modality, comprising:

(a) obtaining a plurality of local features extracted from the first sensor modality;

(b) computing, via a prediction head that outputs a mean estimate and an uncertainty indicator, for each of the plurality of local features, a predicted residual uncertainty representing uncertainty that is expected to remain even after fusion with the second sensor modality, wherein the selection in subsequent step (c) is performed based on said residual uncertainty alone and not on said mean estimate;

(c) selecting, by a deterministic procedure, a subset of the plurality of local features that preferentially includes features with high residual uncertainty;

(d) performing fusion processing with the second sensor modality only on the selected subset, and not performing said fusion processing on the unselected local features.

---

### Claim 2 (Deterministic Diversity Sampling)

The method of Claim 1, wherein the deterministic procedure in step (c) iteratively selects features so as to maximize a product or function of:

- a minimum spatial or feature-space distance from features already selected, and
- the residual uncertainty of the candidate feature.

---

### Claim 3 (Explicit WFPS Score)

The method of Claim 2, wherein the iterative selection score S_i for the i-th feature is defined as:

$$S_i = \sigma_i^2 \times \min_{j \in \text{Selected}} \| p_i - p_j \|$$

where σ_i² is the residual uncertainty of the i-th feature, p_i is the position of the i-th feature, and Selected is the index set of already-selected features.

---

### Claim 4 (Training Scheme — Aleatoric Loss)

The method of any one of Claims 1 to 3, wherein the prediction head is trained, independently of the selection result of step (c), using a loss function of the form

$$L_{aleatoric} = \sum_i \left( \frac{\|y_i - \hat{y}_i\|^2}{2\sigma_i^2} + \frac{1}{2}\log\sigma_i^2 \right)$$

applied to all of said plurality of local features.

---

### Claim 5 (Straight-Through Estimator)

The method of any one of Claims 1 to 4, wherein during training:

- in the forward pass, only the subset selected in step (c) is propagated to subsequent processing; and
- in the backward pass, gradients from the loss computed after fusion processing are propagated to the prediction head through the weights of said selected subset.

---

### Claim 6 (LiDAR + Camera Specialization)

The method of any one of Claims 1 to 5, wherein the first sensor modality is a LiDAR sensor and the second sensor modality is a camera sensor.

---

### Claim 7 (Multi-Sensor Generalization)

The method of any one of Claims 1 to 5, wherein the first and second sensor modalities are each independently selected from the group consisting of LiDAR, camera, radar, thermal camera, and IMU.

---

### Claim 8 (Hard-Gate Cost Reduction)

The method of any one of Claims 1 to 7, wherein the fusion processing in step (d) comprises a Cross-Attention operation, and Cross-Attention is not physically computed for unselected local features, thereby reducing the Cross-Attention cost from O(N²) to O(K²), where K is the cardinality of said selected subset and N is the cardinality of said plurality of local features, with K << N.

---

### Claim 9 (Apparatus)

An information processing apparatus comprising a feature extraction unit, a prediction head unit, a deterministic selection unit, and a fusion processing unit, configured to execute the method of any one of Claims 1 to 8.

---

### Claim 10 (Program)

A program for causing a computer to execute the method of any one of Claims 1 to 8.

---

## Claim Strategy Summary / クレーム戦略まとめ

**独立クレーム1の4本柱**（これらの組み合わせが新規性のコア）:

1. **Pre-Fusion Gating**（統合前ゲート）: ステップ(d)で「選別されたサブセットに対してのみ」統合処理を実行
2. **Variance-Only**（分散のみ）: ステップ(b)で「平均推定値を用いず」残留不確実性のみで選別
3. **Residual Uncertainty**（残留不確実性）: 統合後に「残ると予測される」不確実性を予測（DETR系の「答えに近いもの」とは逆極性）
4. **Deterministic Selection**（決定論的選別）: ランダム性を排除（車載機能安全と整合）

**従属クレームの守備範囲**:

- Claims 2–3: WFPS の具体的な数式に至る防御層
- Claims 4–5: 学習スキーム（Aleatoric Loss + STE）に至る防御層
- Claims 6–7: モダリティ組み合わせに対する一般化
- Claim 8: ハードゲートによる計算量削減の効果
- Claims 9–10: 装置・プログラム形式

**最大の引用予想と防衛**:
RT-DETR の "uncertainty-minimal query selection" が最大の引用候補。Claim 1 の (b) で「**残存する**」（= 統合後にもなお残る）と「**第2のセンサモダリティとの統合処理後**」を明示することで、RT-DETR（単一モダリティ・統合前ではない・極性逆）から綺麗に切り分けられる。

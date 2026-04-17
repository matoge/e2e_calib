# CLAUDE.md

## クイックスタート

```bash
# デモサーバー起動
python app.py
# → http://localhost:5001

# Grid+Depth実験（メイン）
#   config_grid_depth.py を編集してから:
python train_grid_depth.py   # → experiments/{name}/ に保存

# その他学習スクリプト
python train.py          # 1物体
python train_multi.py    # 2物体
python train_cov.py      # 共分散
python train_depth.py    # Depth-aware (旧)

# 可視化
python vis.py            # → result.png
python vis_cov.py        # → result_cov.png
python vis_depth.py      # → result_depth.png
```

## Grid+Depth実験ワークフロー

`config_grid_depth.py` を編集 → `python train_grid_depth.py` で実行。
結果は `experiments/{name}/` に保存（`best_model.pt`, `train.log`, `config.py`）。

```python
# config_grid_depth.py の主なパラメータ
CFG = dict(
    name         = "vXX_...",     # 実験名 → experiments/{name}/
    n_layers     = 2,             # 2 or 3 (coarse→fine or coarse→coarse→fine)
    self_first   = False,         # True = self-attn → cross-attn (悪化する、v10で確認)
    kv_self_attn = False,         # True = 画像トークンにself-attnしてからcross-attn
    img_size     = 64,
    in_channels  = 3,             # RGB
    epochs       = 100,
    batch_size   = 64,
    lr           = 1e-3,
    lr_min       = 1e-6,
    train_size   = 8000,
    val_size     = 800,
    max_offset   = 16.0,
    random_depths = True,         # 全グループの深度をランダム化（v9+）
)
```

## 実験履歴（CalibNetDepth / GridDepthDataset）

| 実験名 | 変更点 | val NLL | obj NLL | 備考 |
|--------|--------|---------|---------|------|
| v7 | 2層 baseline | - | - | |
| v8 | 3層 + pole形状追加 | ~5.1 | ~2.x | |
| v9 `v9_3layer_rgb_randdepth` | ランダム深度・BGギャップ修正 | **5.069** | **1.453** | ベスト |
| v10 `v10_3layer_tfdecoder` | TransformerDecoder (img SA→pt SA→CA) | ~5.37 | ~4.6 | 大幅悪化 |
| v11 `v11_2layer_crossfirst` | 2層・v9アーキテクチャ戻し | 学習中 | - | |
| v12 `v12_2layer_kvsa` | 2層・画像KVにself-attn追加 | 学習中 | - | cross-firstは維持 |

**重要な知見:**
- `self_first=True`（self-attn → cross-attn）は大幅に悪化する
- cross-attn先が自然: 点がまず自分の位置(U,V,D)で画像をクエリしてから点間通信
- v10のTransformerDecoder構造は全然ダメだった（obj NLL 1.4 → 4.6）

## アーキテクチャ概要

- `model.py` → `CalibNet`: Coarse+Fine CrossAttention、Self-Attention付き
- `model_cov.py` → `CalibNetCov` + `gaussian2d_nll` + `CrossAttentionBlockCov` + `TransformerDecoderBlock`
- `model_depth.py` → `CalibNetDepth`: 3ch (U,V,D) 入力、`kv_self_attn`オプション付き
- `dataset.py` → `GridDepthDataset`: 矩形・円・ポール形状、ランダム深度

### CalibNetDepthの構造（n_layers=2, cross-first）

```
入力: 画像(B,3,64,64) + 点群(B,N,3)[U,V,D]

CNN → coarse_feat(B,D,H/8,W/8), fine_feat(B,D,H/16,W/16)
PointMLP(U,V,D) → q(B,N,D)

Layer1: [kv_self_attn(coarse)] → cross-attn(q→coarse) → self-attn(点間) → FFN → raw_c
Layer2: q+raw_c でクエリ位置をずらす → [kv_self_attn(fine)] → cross-attn(→fine) → self-attn → FFN → raw_f

出力: clamp(raw_c + raw_f) → (B,N,5)[tx,ty,log_sx,log_sy,rho]
```

## API エンドポイント

- `GET /api/sample?seed=N&mode=single|multi|depth`
- `GET /api/model_status`

## チェックポイント（旧モデル）

| ファイル | モデル | val loss |
|---------|--------|---------|
| `best_model.pt` | CalibNet (1物体) | 0.18 px |
| `best_model_multi.pt` | CalibNet (2物体) | 0.41 px |
| `best_model_cov.pt` | CalibNetCov | NLL −0.289 |
| `experiments/v9_3layer_rgb_randdepth/best_model.pt` | CalibNetDepth v9 | NLL 5.069 |

## 注意

- `.pt` ファイル: `sim3d_train.pt`, `sim3d_val.pt` 以外は `.gitignore` 済み
- `torch.compile(max-autotune)` で初回起動が遅い（〜5分）
- CUDA必須（CPUでも動くが遅い）
- BGの検出: 深度が最大のグループ（`gi == len(groups) - 1`）

# Auto-mode 進捗ログ — 2026-04-27 PM

ユーザー外出中の自走メモ。終わったもの / 走ってるもの / 待ち行列。

## ✅ 完了

| 内容 | 結果 | コミット |
|---|---|---|
| Unified arch 同モデルで cam-LiDAR calib + cross-frame 同時動作 | v303 0.67px / v304 進行中 | 6094a19 |
| 1 model N problems 設計 doc | `docs/unified_modality_primitive.md` | 8c3a18c |
| Map-side frame_token + LIVO 比較 (SLAM 拡張) | doc 同上 §6 | 07f8054 |
| **Cam-Radar PoC** (nuScenes) | v305 val_err 0.61px / val_nll −0.17 | f59b64a |
| **6DIR バグ修正** (step_n loss = sum*0.5 → mean) | v201 が v200 に劣る原因特定 + 修正 | b7b03c4 |
| Cam-Radar 結果 doc 反映 | 3 modality × 2 dataset 表 | c452eda |

## 🔄 走行中

### yokohama (5080)
- **v306** combined cam-LiDAR calib (PandaSet 103 + DDAD 200 + Waymo 797、all cams、4649 pairs)
  - ep8: val_err 1.72px (base 7.56), val_nll 1.32 → 順調

### sakurai (5070Ti)
- **v304** PandaSet 103 cross-frame uv-only-Q (ep27/30、val_err 2.56px)
  - 残り 3 epoch (~ 4 分)

### Background
- **AV2 sensor sync** to mininas: train 687/830GB ≈ 83%、val/test 完了

## 📋 待ち行列

### yokohama (v306 後)
1. **v307** combined pair (4 dataset = panda + ddad + waymo + **nuScenes**、all cams)
2. **v308** nuScenes cam-Radar calib all-cam (front_cam → all-cam scale-up)
3. **v310** nuScenes cam-LiDAR calib (calibration の dataset 多様性検証)

### sakurai (v304 後)
1. **v309** PandaSet 39 N=3 6-direction with **6DIR loss fix** (b7b03c4)
   - 比較対象: v200 (val 2.27 px, 2-frame pair) / v201 (val 4.91 px, バグあり 6-dir)

## 🔜 まだやってない

- **AV2 PandaSet-format 変換** — train sync 終わり次第
- **multi-camera calib** smoke test (modality flag swap、コード修正最小)
- **map-side frame_token** PoC (永続地図 → frame_token render → cross-attn) — SLAM kernel
- **学習済みモデルの量子化 + edge benchmark** (Hailo-8 / Coral / Jetson)

## 注意 / 知見

- **6DIR は実は pair より弱い** が empirical fact。3 つの仮説で順に検証:
  1. composition-consistent 摂動 → 既に修正済み (v201 で確認、4.91 px)
  2. loss スケール (`sum*0.5` → `mean`) → v309 で逆効果 (6.65 px、AB grad 1/6 weight)
  3. **結論**: 6-direction supervision そのものが M 関連の方向経由で
     共有 encoder を destabilize → pair (v200=2.27px) より構造的に弱い。
  - v309 後に loss scaling は v201 の `sum * 0.5` に revert。
- **Combined dataset は per-dataset best に追いつかない**:
  - cam-LiDAR calib: v303 (PandaSet 103 単独) 0.67 < v306 (combined all-cam) 1.00
  - cross-frame: v304 (PandaSet 103 単独) 2.54 < v307b (combined 3-ds) 2.63
  - 仮説: 各 ds のキャリブ精度・センサー特性の差異を吸収しきれてない。
    Per-dataset fine-tune が必要かも。

- **All-cam scaling は OK**: v306 は 1100 scenes × all-cam = 4649 (scene,cam) で
  1.00 px、v308 nuScenes all-cam で 0.71 px。データ量は劣化要因じゃない。

- **3 modality (LiDAR / Radar / mm) × 2 dataset family** で同モデル動作確認。
  論文にしてもよさそうだが、ユーザーは "papers as pebbles" 主義なので
  artifact (動くシステム) に集中。

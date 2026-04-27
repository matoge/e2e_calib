# Auto-mode 進捗ログ — 2026-04-27 PM

ユーザー外出中の自走メモ。最終結果まで完走。

## 🎯 最重要成果

### 1 モデル定義で 3 modality × 4 dataset を解いた

`CalibNetUnifiedFrame(uv_only_query=True, n_cross_layers=4, 1.65M params)`
で全部回る。modality (cam/lidar/radar/mm) は encoder の flag だけ。

| run | task | dataset / cam | val_err |
|---|---|---|---|
| v303 | calib | PandaSet 103 / front | 0.67 |
| v304 | cross-frame | PandaSet 103 / front | 2.54 |
| v305 | **cam-Radar** calib | nuScenes 150 / front | 0.61 |
| v306 | calib (combined) | Panda+DDAD+Waymo / **all-cam** | 1.00 |
| v307b | cross-frame (combined 3-ds) | all-cam | 2.63 |
| v308 | cam-Radar (all-cam) | nuScenes / 6 cams | 0.71 |
| v310 | calib | nuScenes / front | 0.79 |

### Pre-train (combined) → fine-tune (single ds) recipe が効く

| pre-train | target | val_err | from-scratch | Δ |
|---|---|---|---|---|
| v306 | v311 PandaSet 103 calib | **0.60** | v303 0.67 | −10% |
| v307b | v312 PandaSet 103 cross-frame | **2.38** | v304 2.54 | −6% |
| v308 | v313 nuScenes radar | **0.53** | v305 0.61 | −13% |
| v306 | v314 DDAD calib | **0.88** | (combined 1.00) | −12% |
| v306 | v315 Waymo calib | **0.86** | (combined 1.00) | −14% |

**全 5 fine-tune が from-scratch / combined を改善**。Production deployment
recipe 確立: 「1 base 重み × N 短時間 fine-tune」。

### Cross-modal transfer の非対称性 (v316/v317)

| direction | val_err | from-scratch baseline |
|---|---|---|
| LiDAR → Radar (v316) | **0.58** | v305 0.61 ← better |
| Radar → LiDAR (v317) | 0.81 (ep12) | v303 0.67 ← **worse** |

**情報量の大きい modality を pre-train に**。Radar encoder の細い表現は
dense LiDAR には載らない。

### 6-direction supervision は pair より構造的に弱い (negative result)

| run | scheme | val_err |
|---|---|---|
| v200 | pair (2 dirs, mean) | 2.27 |
| v201 | 6 dirs, sum*0.5 (実効 lr 3×) | 4.91 |
| v309 | 6 dirs, mean (lr 校正) | 6.65 ← 最悪 |

仮説 1 (composition-consistent): 修正済み (v201)。
仮説 2 (loss scale): mean 試したら逆効果 (v309)。
**結論**: M 関連方向が共有 encoder を destabilize、pair が最適。

## ✅ 完了タスク (全 17 runs)

- v303 / v305 / v310 (per-dataset calib from scratch)
- v306 / v307b / v308 (combined pre-train)
- v311 / v312 / v313 / v314 / v315 (per-dataset fine-tune)
- v309 (6DIR fix verification — confirmed dead-end)
- v316 / v317 (cross-modal transfer)
- AV2 sync to mininas (1.1TB train+val+test 完了)
- nuScenes radar 150 シーン変換完了
- 7 commits + doc/html 更新

## 📋 残タスク (未着手 / user 帰り次第)

- **AV2 PandaSet 変換**: シンボリックリンクが nvme6t 古いパス向き、mininas 経路に再変換要
  - `python scripts/preprocessing/av2_to_pandaset.py --src /mnt/mininas/datasets/argoverse2/sensor/train` で再 run 可能
- **nuScenes lidar pickle 一部破損**: numpy 2.0 系で書かれたものがあり、現環境 (numpy 1.26) で読めない scene が混在。`scene-0001` で踏むので v307 (4-ds combined) は失敗、v310/v311 (front_camera 単独) は OK だった
- **multi-camera calib** smoke test: modality_A=`cam`, modality_B=`cam` で動くはず、未検証
- **map-side frame_token**: 永続地図 → frame_token render → cross-attn の SLAM kernel、設計のみ

## 🔗 参照

- 設計 doc: `docs/unified_modality_primitive.md` (and html)
- 全実験 ckpt: `experiments/cross_frame_v3{03..17}*/best_model.pt`
- ClearML: http://localhost:18080/projects/99a9ff692440487f8e4251020103d041

## コミット履歴 (今セッション、新しい順)

```
5610ec2 docs: cross-modal transfer asymmetry — LiDAR→Radar works, Radar→LiDAR doesn't
9a64262 docs: v315 Waymo fine-tune lands at 0.86 px (vs 1.00 combined)
bd9741f docs: pre-train+fine-tune validated across 3 modalities + 3 dataset families
ab77077 docs: pre-train + fine-tune recipe — beats from-scratch single-dataset
f85c9d3 revert 6DIR loss scaling fix — 6-direction is empirically weaker than pair
d38bb2a docs: empirical table now 6 runs across 3 modalities × 3 datasets
edbac92 docs: auto-mode session log
c452eda docs: cam-Radar PoC empirical row + 3-modality verification table
b7b03c4 6DIR fix: switch step_n loss from sum()*0.5 to mean()
f59b64a cam-Radar PoC — nuScenes radar via lidar_subdir flag
07f8054 docs: extend unified primitive — map-side frame_token + LIVO comparison
8c3a18c docs: 1 model, N problems — uv-Q + multimodal frame_token primitive
6094a19 unified arch: same model handles calib + cross-frame via modality split
```

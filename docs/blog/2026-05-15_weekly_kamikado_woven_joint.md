# 週次レポート (2026-05-09 〜 2026-05-15)

E2E_CALIB σ-net、社内データ統合と Joint training 拡張。

---

## P1 — 上門さん TSS4 (kamikado) データ学習成功

社内 TSS4 (front camera 1ch + VLS-128 LiDAR、4-ch intensity 付き) cache
で σ-net を 100 epoch 学習。**frame-level split** (= scene を超えた frame で
train/val 切る) で過学習しないことを確認。

### 設定 — [config](../../experiments/tss4_20260514_intensity_4ch_100ep_framesplit/config.py)

| key | val |
|---|---|
| cache | `tss4_v3_tiled` (V3 tile cache、4-ch intensity 同梱) |
| frames | 795 (train 716 / val 79、seed=42 で random split) |
| instances | train **229,120** / val 25,280 (tile 展開後) |
| arch | ConvNeXt + frustum + deform_sl + 4-layer cross-attention |
| epochs / batch | 100 / 256 (RTX 5080、bf16) |
| oversample | 8 |
| 学習時間 | 198 min (3.3h) |

### 結果

**best val NLL = 2.1348** (ep 100)。

cosine schedule で lr 3e-4 → 1e-7、val 単調減少。`obj NLL = 0` は TSS4 cache に
obj annotation が無いため (bg-only NLL = pt NLL = 2.135)。

| ep | train_nll | val_nll | lr |
|---:|---:|---:|---:|
| 1 | 4.5 | 4.2 | warmup |
| 10 | 3.1 | 2.8 | 3.0e-4 |
| 50 | 2.5 | 2.3 | 1.5e-4 |
| 100 | 2.33 | **2.135** | 1.0e-7 |

### 可視化サンプル (ep 100 val)

左 = 元 frame 上での tile 位置 (赤枠 = model が見た crop)。
右 = 同 tile crop に LiDAR 点を投影。σ 楕円は今の vis preset では出ない
ので別途 best_model.pt から viz 走らせる必要あり。

![val 00](../../experiments/tss4_20260514_intensity_4ch_100ep_framesplit/vis_ep100/val_00_idx023682.png)
![val 01](../../experiments/tss4_20260514_intensity_4ch_100ep_framesplit/vis_ep100/val_01_idx019684.png)
![val 02](../../experiments/tss4_20260514_intensity_4ch_100ep_framesplit/vis_ep100/val_02_idx001577.png)
![val 03](../../experiments/tss4_20260514_intensity_4ch_100ep_framesplit/vis_ep100/val_03_idx031377.png)

→ TSS4 cache だけでも 5cm 級の点単位ずれ予測ができる、社内データ単体での
σ-net 適用に道が開けた。

<div style="page-break-after: always;"></div>

---

## P2 — Loom (WOVEN) データ学習対応 + Joint training 拡張

Loom チームの 11-camera + VLS-128 cache (`woven_v3_tile_v1`、35,881 tile、
8GB) を joint training に組み込む準備。

### Joint training 構成

```
cache list = [
  zod_v3_tiled_clean_i,    # ZOD (Volvo + Zenseact)、front cam、4-ch intensity
  tss4_v3_tiled,           # TSS4 (kamikado 提供)、4-ch intensity
  pandaset_v3_tiled_i,     # PandaSet (Hesai)、4-ch intensity
  # woven_v3_tile_v1,      # 追加予定 (現在 cache 復旧待ち)
]
rep_strategy = nearest_cam        # multi-cam joint 用
oversample   = 4 per cache
```

### 現状: `zod_tss4_ps_20260515_joint_repnear_os4_50ep_lrmin1e6`

3 cache 合成で **1,562,880 train inst**、val 8000、50 epoch、RTX 5080 1枚で
~14h 想定。lr_min は 1e-6 (前回 1e-7 だと最後 9 epoch で停止していた)。

| ep | train_nll | val_nll | save |
|---:|---:|---:|---|
| 1 | 5.462 | 5.079 | ★ saved |
| 3 | 4.748 | (val skip) | |
| 5 | 4.287 | 4.152 | ★ saved |
| 7 | 3.940 | 4.152 | (進行中) |
| 50 | (予定) | TBD | |

`val_every=5` で 5 epoch ごとに validate。前回 os=1 run は val 3.211 で
saturate だったので、os=4 で容量増 + lr_min=1e-6 で末尾 flat tail 取りに行く。

### WOVEN cache 復旧

ClearML 上に `woven_v3_tile_v1` task は **completed** 表示だが、state.json
(10MB) のみで data zip chunk **未 upload** の壊れた状態 (旧 ministar 死亡時に
upload 中だった疑い)。state.json 内に 35,881 ファイルの (path, hash, size,
artifact_name) があるので、元 raw frame + lidar から再ビルドが必要。

→ 元 raw データの場所 (Loom 側 PC) 確認 + 再 cache build スクリプト走らせる
のが Next Step。

<div style="page-break-after: always;"></div>

---

## P3 — インフラ整備

### Waymo V3 tile cache build

Waymo Open Dataset (798 segments) を V3 tile + 4-ch intensity 形式で
cache 化。

- 並列 4 worker、`max_tasks_per_child=1` で worker recycle (glibc heap 断片化
  対策)
- `_done_<seg>.flag` の touch で resume 可、途中 crash しても巻き直し効く
- **進捗: 595 / 798 (75%)、6h32m 経過、+20GB 出力、残り ~2.3h** (今夜中完了見込み)

完了次第 joint training の 4th cache に追加可能。

### ClearML server 復旧 + fileserver patch

旧 ministar 死亡で:
- api_server LAN IP `192.168.1.5 → 192.168.1.9` 移動 (`~/clearml.conf` 修正)
- fileserver の `POST /<path>` route patch を bind-mount で永続化
  (`/home/hiro/clearml-server/fileserver.py` → 容器内 `/opt/.../fileserver.py`)
- 検証: `curl -F file=@x http://localhost:18081/any/path` → 200 (旧 405 解消)

### Cache の ClearML Dataset 化

学習 cache を ClearML Datasets として登録 (再 ministar 化に備えたバックアップ)。
**Cloudflare Tunnel の 100MB body 制限** で 512MB zip chunk が 413 で弾かれた
ので、`CLEARML_FILES_HOST=http://192.168.1.9:18081` で LAN IP 直叩きに切替。

| dataset | id | size | status |
|---|---|---:|---|
| `pandaset_v3_tiled_i` | `04cfeb7c92d6405795c3c72f74b12acc` | 22 GB | ✅ closed/finalized |
| `zod_v3_tiled_clean_i` | `16a2ddd08fdf4f50b3d03ca7cdcb19d6` | 128 GB | 🔄 upload 中 (~50GB 着) |

### 次週の TODO

1. WOVEN cache 復旧 (元 raw データから再 build) → 4-cache joint training へ
2. Waymo cache 完了確認 + joint training 5th cache 追加
3. joint os4 50ep 完了 → val NLL を ZOD baseline と比較、lr 末尾の影響評価
4. Cross-frame σ-net (VCPE + KV concat、UV-emb-only query) の 1-week milestone
   実装着手

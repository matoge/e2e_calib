# V3 Cache Build Guide

会社サーバ (Xeon Platinum 8168 × 2 / V100 / NVMe) で `pandaset_v3_full` / `waymo_v3_full` / `nuscenes_v3_full` をゼロから作るための手順。

ハイレベル方針：
- **PandaSet は front_camera のみ** (V2 の側面カメラは 31-75ms 時刻オフセット、calib 学習に不適)
- **Waymo は LCP (lidar_camera_projection) 経由でピクセル単位クリーン投影**、5cam 全部
- **nuScenes は 6cam 全部、key-frame 2Hz**
- 全キャッシュ共通スキーマ (jpg_bytes インライン + 3D pts + K + cuboids)、`PandaSetCalibDatasetFull` 一本で読める

---

## 0. 依存

```bash
# system
sudo apt install libturbojpeg
# python
pip install torch numpy pandas pyarrow scipy pillow PyTurboJPEG nuscenes-devkit
# gcloud SDK (Waymo の LCP 取得に必要)
curl https://sdk.cloud.google.com | bash && exec -l $SHELL
gcloud auth login   # 認証は anonymous でも OK (waymo open dataset は public)
```

GPU は不要 (CPU only build)。出力は instance ごとの `.pt`、合計 ~110GB。

---

## 1. ソースデータ DL

### PandaSet (44.5GB)
HuggingFace から:
```bash
huggingface-cli download --repo-type dataset georghess/pandaset \
  --local-dir /data/pandaset_zip
unzip /data/pandaset_zip/pandaset.zip -d /data/pandaset/
# → /data/pandaset/<scene_id>/{camera/, lidar/, annotations/, meta/}
```

### Waymo Open Dataset v2 (training)
- `lidar/*.parquet` (range images per laser × frame)
- `camera_image/*.parquet`
- `camera_calibration/*.parquet`
- `vehicle_pose/*.parquet`

```bash
mkdir -p /data/waymo_v2/training
for sub in lidar camera_image camera_calibration vehicle_pose; do
  gcloud storage cp -r "gs://waymo_open_dataset_v_2_0_0/training/$sub" \
    /data/waymo_v2/training/
done
```

LCP (lidar_camera_projection) は build 中にも自動 DL されるが**先取り並列 DL したほうが圧倒的に速い**：
```bash
gcloud storage cp -n "gs://waymo_open_dataset_v_2_0_0/training/lidar_camera_projection/*.parquet" \
  /data/waymo_lcp/training/
# 798 segs × ~85MB ≈ 68GB、並列 DL で 20-30 分
```

`datasets/waymo_lcp.py` の `ensure_lcp()` 既定パスは `/mnt/nvme6t/waymo_lcp` なので、別パスに置く場合は `WAYMO_LCP_DIR` 環境変数で上書き、または build_waymo_v3.py 起動時に渡す引数を追加（現在は hardcode）。

### nuScenes trainval (300GB+)
nuscenes.org でアカウント作って trainval01-10 + meta + lidarseg ダウンロード。展開して：
```
/data/nuscenes/
  ├ samples/
  ├ sweeps/
  ├ v1.0-trainval/
  └ ...
```

---

## 2. キャッシュ build

各 build スクリプトは ProcessPoolExecutor で並列化、`--workers` でコア数指定。**物理コア数 ≦ workers** を推奨 (HT スレッド回しても JPEG decode で cache 汚染、伸びない)。

### 2.1 PandaSet (front camera のみ、1Hz)

```bash
python scripts/preprocessing/build_pandaset_full_v3.py \
  --src     /data/pandaset \
  --out     /data/cache/pandaset_v3_full \
  --cams    front_camera \
  --stride  5 \
  --workers 16
```

- 103 scenes × 80 frames / stride 5 ≈ **2.7k insts、1.5GB**
- ETA **~10 分**
- ⚠️ `.pkl` と `.pkl.gz` の重複に注意 — build_pandaset_full_v3.py の `_frame_files()` で `.pkl` 優先する fix 済み (2026-04-30 の bug)

### 2.2 Waymo (全 5cam、1Hz)

LCP 先取り済の前提で：

```bash
python scripts/preprocessing/build_waymo_v3.py \
  --out     /data/cache/waymo_v3_full \
  --cams    1,2,3,4,5 \
  --stride  10 \
  --workers 8
```

- 798 segs × 20 frames × 5 cam ≈ **80k insts、~80GB**
- ETA **~3-4h** (96-core 機 / LCP cached 済 / NVMe)
- LCP 未 DL 時は +30 分の serial DL が混じって遅い
- `--max-segs N` で smoke run 可 (= 100 segs / 30 min)

### 2.3 nuScenes (全 6cam、2Hz key-frames)

```bash
python scripts/preprocessing/build_nuscenes_v3.py \
  --src     /data/nuscenes \
  --version v1.0-trainval \
  --out     /data/cache/nuscenes_v3_full \
  --stride  2 \
  --workers 24
```

- 850 scenes × 40 keyframes × 6 cam ≈ **100k insts、28GB**
- ETA **~1h** (NAS-bound — ローカル NVMe にデータ置けば 30 分)

### (おまけ) DDAD (1Hz)

```bash
python scripts/preprocessing/build_ddad_v3.py \
  --src     /data/ddad/ddad_train_val \
  --out     /data/cache/ddad_v3_full \
  --stride  10 \
  --workers 16
```

- 200 scenes × 20 frames × 6 cam ≈ **24k insts、~10GB**
- cuboids は `[]` (DDAD は 3D box ラベル無し)

---

## 3. 出力スキーマ (instance .pt)

各 `<gid:08d>.pt` は dict：

```python
{
  'jpg_bytes' : bytes,                   # raw JPEG (元解像度)
  'IH', 'IW'  : int,                     # 画像 height/width
  'pts'       : (N, 3) float32 tensor,   # 3D points (詳細下記)
  'cam_pos'   : (3,)   float32,
  'R_gt'      : (3, 3) float32,          # cam→world (Waymo は identity)
  'T_gt'      : (4, 4) float32,          # world→cam (Waymo は identity)
  'K_full'    : (3, 3) float32,          # pinhole intrinsic
  'cuboids'   : [                         # 3D ボックス、Waymo/DDAD は空
    {'pos': (3,), 'dims': (3,), 'yaw': float, 'cls': str?},
    ...
  ],
  'scene'     : str,                     # provenance
  'cam'       : int|str,
  'frame'     : int,
}
```

`pts` の座標系：
- **Waymo**: cam frame (LCP の `(u,v,depth)` から逆射影で復元)、`R_gt`/`T_gt` は identity
- **PandaSet / nuScenes / DDAD**: world frame、`T_gt = inv(T_cam_to_world)` を別途持つ

`PandaSetCalibDatasetFull` (`datasets/pandaset_full.py`) はこれをロードして:
1. `T_gt @ pts` で cam frame に持っていく
2. ランダムな摂動 `(δR, δt)` をかけて再投影 → モデル入力
3. GT との残差 `(Δu, Δv)` を学習

---

## 4. 検証 (build 後の sanity check)

```bash
# 全 inst のうち 10 枚ランダム + フル画像 + crop ボックスを描画
python scripts/visualization/vis_pretrain.py \
  --cache /data/cache/pandaset_v3_full --n 10

# Waymo は cam ごとに分けて出力
python scripts/visualization/vis_pretrain_waymo.py \
  --cache /data/cache/waymo_v3_full --n-per-cam 5
```

→ `<cache>/vis_pretrain/*.png` 確認、lidar が画像にちゃんと乗ってれば OK。

---

## 5. 学習 (combined cache)

`train_ps_v3.py` は単一 `--cache` 前提。複数キャッシュ統合する場合は **symlink farm** が一番楽：

```bash
DST=/data/cache/combined_v3
mkdir -p $DST/inst
# prefix で衝突回避
for i in /data/cache/pandaset_v3_full/inst/*.pt; do
  ln -s "$i" "$DST/inst/p_$(basename $i)"
done
for i in /data/cache/nuscenes_v3_full/inst/*.pt; do
  ln -s "$i" "$DST/inst/n_$(basename $i)"
done
for i in /data/cache/waymo_v3_full/inst/*.pt; do
  ln -s "$i" "$DST/inst/w_$(basename $i)"
done

# meta.pt も結合
python -c "
import torch
parts = ['/data/cache/pandaset_v3_full/meta.pt',
         '/data/cache/nuscenes_v3_full/meta.pt',
         '/data/cache/waymo_v3_full/meta.pt']
prefixes = ['p_', 'n_', 'w_']
train, val = [], []
for p, pre in zip(parts, prefixes):
    m = torch.load(p, weights_only=False)
    train += [pre + f for f in m['train']]
    val   += [pre + f for f in m['val']]
torch.save({'train': train, 'val': val}, '$DST/meta.pt')
print(f'combined: train={len(train)} val={len(val)}')
"

python scripts/training/train_ps_v3.py \
  --cache $DST \
  --name  v600_combined_clean3_v100 \
  --workers 24 --batch-size 64 --epochs 50 \
  --img-size 128 --grid-n 16
```

---

## 6. 高速化 tips

- **TurboJPEG 必須**: `pip install PyTurboJPEG` & `apt install libturbojpeg`。無いと PIL fallback で 5-10x 遅くなる
- **キャッシュは NVMe / tmpfs に**: 110GB が RAM に乗るなら `mount -t tmpfs -o size=130G tmpfs /tmpfs_cache && cp -r ...` で 2-3x 加速
- **DataLoader workers**: 物理コア数まで。Xeon Platinum 8168 (48c) なら 24-32 が sweet spot
- **forkserver 必須**: `multiprocessing.set_start_method('forkserver', force=True)` (train_ps_v3.py では既設定済)
- **kill 後の orphan 確認**: `pgrep -af pt_data_worker | wc -l`、leak してたら `pkill -9 -f pt_data_worker`

---

## 7. トラブル

| 症状 | 原因 | 対処 |
|---|---|---|
| Waymo build 1 セグ 5 分以上 | LCP DL がシリアル | `gcloud storage cp -n` で並列先取り |
| OOM during build | workers × max-frames × jpg_bytes | `--workers` 半減 or `--max-frames` 制限 |
| PandaSet で点が画像に乗らない (val_07 みたいなやつ) | `.pkl/.pkl.gz` 重複で frame index ずれ | 既 fix 済 (`_frame_files()` で `.pkl` 優先)、古い cache は再 build |
| `clearml-data get` で停止 | files_server URL が cloudflared 経由でない | 会社マシン `~/clearml.conf` の files_server を auth 済 URL に書き換え |

---

## 8. 参考

- 元のデバッグ履歴と各 fix の経緯: 2026-04 の commit log (`waymo: fix [CameraImageComponent].pose interpretation` 周辺)
- アーキテクチャ詳細: `experiments/ps_v504r_convnext_deform_l4/config.py` (val_nll 1.94、obj 1.75)
- 校正残差ネットの一般化: `docs/unified_modality_primitive.md`

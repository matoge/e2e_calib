# Waymo Open Dataset v2 → PandaSet-layout

`waymo_to_pandaset.py` 変換器の実装ノート。**`[CameraImageComponent].pose` の解釈に落とし穴あり**。

## V3 cache: `build_waymo_v3.py`

直接 v2 parquet を読む V3 build。`waymo_to_pandaset.py` を介さない。
PandaSet-compat schema (`scripts/preprocessing/build_pandaset_full_v3.py` と同一)、
`pts` は cam frame、`T_gt = identity`、`R_gt = I`、`cam_pos = 0`。

### Full-frame mode (1 inst/frame/cam)

```bash
python scripts/preprocessing/build_waymo_v3.py \
    --workers 8 --cams 1,2,3,4,5 --stride 10 \
    --out /path/to/waymo_v3_full
```

各 inst:
- `jpg_bytes` (~400KB)、`IH=1280, IW=1920`
- `pts` / `pts_cam` (N, 3) cam-frame、`uv_full` (N, 2)、`z_cam` (N,)、`is_obj` (N,)
- `cuboids` (cam-frame AABB list)、`K_full` (3, 3)

### Tile mode (sliding window、5×3=15 tile/frame for front)

`--tile` で 1 frame を 5×3 tile (512×512、stride=384、`y_start=200` で空切り) に分割。
`PandaSetCalibDatasetFull` がそのまま読める形式 (`tile_u0, tile_v0, in_box` 付き)。

```bash
python scripts/preprocessing/build_waymo_v3.py --tile --stride 10 \
    --workers 32 --cams 1,2,3,4,5 \
    --out /path/to/waymo_v3_tiled \
    [--tile-w 512 --tile-h 512 --tile-stride 384 --tile-pad 64
     --tile-y-start 200 --tile-jpg-q 90]
```

ストレージ: 5 cam × 798 seg × ~20 frame × ~15 tile/frame ≈ 1.2M tile inst (~120GB)。
parquet IO 重いので NVMe 必須 (HDD だと 5h+ かかる)。1TB メモリ機なら `--workers 32` 余裕。
ヘルパー: `_tile_split.py::cut_inst_to_tiles` を共有 (PS / NS / Waymo build 全部 同実装)。

## ⚠ 落とし穴: `[CameraImageComponent].pose` は cam pose ではない

Waymo OD v2 parquet `camera_image` 内の各 row には:
- `[CameraImageComponent].pose.transform` (4×4)
- `[CameraImageComponent].pose_timestamp` (秒)

…という pose 関連フィールドがあるが、**`pose.transform` は `T_world_from_camera` ではない**。実体は **`T_world_from_vehicle` を `pose_timestamp` で interpolate した値** = エゴ車体の世界姿勢、カメラ姿勢ではない。

検証 (一致 = 0.0m / 0.0rad):
```python
T_image_pose      = np.array(row['[CameraImageComponent].pose.transform']).reshape(4,4)
T_world_from_veh  = vehicle_pose @ pose_timestamp   # interpolated
np.allclose(T_image_pose, T_world_from_veh)         # → True
```

正しいカメラ世界姿勢は静的 `T_veh_from_camera_static` を掛ける必要あり:
```python
T_world_from_wcam_at_shutter = T_image_pose @ T_veh_from_camera_static
```

過去 commit でこの掛け算が抜けてて、`waymo_ps/*/camera/*/poses.json` 全数 cam mount 分 (~2m) ずれてた。フロントカメラだと yaw rotation 軸との位置合わせで症状がほぼ縮退して気付きにくい (translation のみ ~2m off)、サイドカメラで実距離 0.1m 付近の物体まで見えなくなって発覚。

修正後 (`scripts/preprocessing/waymo_to_pandaset.py:227-237` 周辺):
```python
T_world_from_veh_shutter = np.array(img_row['[CameraImageComponent].pose.transform']).reshape(4,4)
_, _, _, _, T_veh_from_wcam_static = cam_intr[cid]
T_world_from_wcam = T_world_from_veh_shutter @ T_veh_from_wcam_static
T_world_from_cam  = _T_world_from_opencvcam(T_world_from_wcam)   # OpenCV 軸へ
```

## サイドカメラの残差 (~10-30 px)

修正後でも side cam に小さい残差が残る。原因:

1. **TOP_LIDAR 360° スイープ ~100ms** — frame 内でも azimuth ごとに数十 ms の時間差
2. **vehicle_pose が 10Hz のみ** — フレーム間 SLERP 補間で旋回時 5-10cm 残る
3. **cam rolling shutter** — 1 フレームの上下で 30-50ms、1ポーズ近似で残る
4. **静的 calib 残差** — Waymo 公開 calib で 0.1° 程度報告

(1) を緩和するには per-azimuth motion compensation が必要だが、**v2 parquet には per-row pose チャネル無し** (v1 TFRecord の `range_image_top_pose` は v2 で削除)。`vis_waymo_raw_5cam.py --motion-comp` では `t_capture(col) = frame_ts + col * (sweep_period/W)` の線形近似で補間してる — 経験上 cam shutter timestamp と完全には合わず ~10-30px 残差。

`cross_frame` 学習タスクではこの残差は σ で吸収可能なので問題にならないが、calib 用途で点群と画像のピクセル単位整合が必要な場合は注意。

## 診断スクリプト

`scripts/visualization/vis_waymo_raw_5cam.py` — Waymo 生 parquet から直接 5 cam projection。

```bash
# 全 5 LiDAR + 両 return + per-azimuth motion-comp
python scripts/visualization/vis_waymo_raw_5cam.py --seg-idx 5 --frame 30 \
    --all-lasers --use-return2 --motion-comp
```

出力 `vis_waymo_raw_5cam.png` の各カメラ title に `vis_pts / Δt` 表示。

## 既知の影響範囲

`waymo_to_pandaset.py` の修正前に preprocess された `/mnt/nvme6t/waymo_ps/*` は全部 cam pose ズレてる。Waymo 含む cross_frame 系実験 (v112, v300, v301, v302, v306, v307, v315, v320 など) は wrong pose で学習済み。再評価 or 再学習が必要。

# WovenSequence データの扱い (TSS4 FCM fisheye + VLS128)

> 2026-05-15: devpc が死んで未 push の可視化スクリプトを失った件を契機に整理。
> 「自前で投影を書き直さない」「アノテのVERIFY状態をフィルタする」「困ったらこのファイルを読む」が目的。

## データレイアウト

ローカル: `~/git/loom/backend/assets/woven_sequence/<batch>/<task>/<sequence_dir>/`

例: `llinking_27/tf_long2/sequence=ip654_1337941440921107425_16943630305775105398_1749030654176-1749030664176/`

| パス | 中身 |
|---|---|
| `tss4_fcm/<frame_id>.jpg` | 3840×2160 fisheye 画像 |
| `tss4_fcm_rect/...` | 同 rectified 版 |
| `vls128_rear_axle/<frame_id>.npz` | LiDAR (rear_axle FLU)。`xs, ys, zs, intensity, azimuth, elevation, ...` |
| `saved_annotations/<frame_id>.json` | per-frame 3D box list |
| `setting-ip*.json` | calib (Kannala-Brandt + extrinsics) |
| `metadata.json` | poses (`gicped_poses` 優先) と `camera_delays_ms` |
| `K_rect.json` | rectified pinhole 用 K |

`<frame_id>` = `<index>_<timestamp_ns>` (例: `0050_1749030625599869952`)

## 座標系

- **Vehicle / LiDAR (rear_axle)**: ISO8855 FLU. `x=fwd, y=left, z=up`
- **Camera (出力 FRD)**: `x=right, y=down, z=fwd`
- 変換: `R_TO_RDF = [[0,-1,0],[0,0,-1],[1,0,0]]` を介す
  (`build_woven_sequence_v3.py` の `_camera_calib_fcm` 参照)

## 投影パイプライン (絶対自分で書き直さない)

backend の `simple_api._compute_projected_points` と完全に揃えてある実装が
`scripts/preprocessing/build_woven_sequence_v3.py` にある。これを **そのまま import して使う**。

```python
from scripts.preprocessing.build_woven_sequence_v3 import (
    _load_setting,
    _camera_calib_fcm,
    _load_metadata,
    _get_poses,
    _camera_delay_ms_for_frame,
    _pose_at_camera_time,
    _load_pts_intensity,
    _lidar_to_cam_at_camera_time,
    _project_kannala,
)

setting = _load_setting(SEQ)
K, dist, R_cam_from_veh, t_cam_in_veh, W, H, delay_default = _camera_calib_fcm(setting)
metadata = _load_metadata(SEQ)
frame_ids, poses = _get_poses(metadata)

# 1 フレーム
idx = ...   # int, frame_ids 上のインデックス
fid = frame_ids[idx]
pts_veh, intensity = _load_pts_intensity(SEQ, fid)
delay_ms = _camera_delay_ms_for_frame(metadata, fid, delay_default)
pose_curr   = poses[fid]
pose_camera = _pose_at_camera_time(poses, frame_ids, idx, delay_ms)
pts_cam = _lidar_to_cam_at_camera_time(
    pts_veh, pose_curr, pose_camera, R_cam_from_veh, t_cam_in_veh)
uv = _project_kannala(pts_cam, K, dist)   # (N, 2)
```

**ハマりポイント** (過去にやらかした):

- `R_cam_from_veh` の符号や軸順を勘で書くと左右逆になる。`_camera_calib_fcm` を使う
- `camera_delay_ms` (33ms 程度) を無視すると動的物体が数 px ズレて気持ち悪い → `_pose_at_camera_time` を必ず通す
- LiDAR 点は `xs, ys, zs` の3列を `np.stack(axis=1)` した `(N,3)` で渡す。`_load_pts_intensity` がやってくれる

## アノテーション (`saved_annotations/<frame>.json`)

```
{
  "details": [
    {
      "type": "box",
      "label": "car" | "truck" | "sign" | "traffic_body" | "traffic_light_bulb"
                | "lanemarker_edge" | "delineator" | "pedestrian" | ...,
      "object_id": 17,
      "cuboid_uuid": "...",
      "bbox": {...},                 # 2D image bbox (使わなくてOK)
      "left", "top", "width", "height": float,   # 2D image bbox 別表現
      "not_in_lidar": false,
      "edit_history": {
        "action": "...",
        "timestamp": "2025-11-20T09:20:39.388904",
        "user": "hiroyuki.funaya"     # ← VERIFY フィルタの判定キー
      },
      "attributes": {
        "3dbb_rear_axle": {           # rear_axle FLU の 3D box (LiDAR と同系)
          "center_meter":  [x, y, z],
          "size_meter":    [l, w, h],
          "direction":     [cos_yaw, sin_yaw],
          "timestamp_ns":  ...
        },
        "3dbb_rear_axle_camera_frame": {...},   # camera shutter 時刻補正後 (build_v3 はこれを優先)
        "projected_3d_corners": [[u,v]×8]        # backend が事前計算した画像座標の 8 角
      }
    },
    ...
  ]
}
```

### VERIFY フィルタ (`edit_history.user`)

ip654 / llinking_27 のサンプル統計 (1シーケンスの 30 フレーム集計):

| user | 件数 | 性質 |
|---|---|---|
| `automation`       | 761 | 自動 (未VERIFY) |
| `automation_yolo`  | 368 | YOLO 自動 (~VERIFY扱い) |
| `hiroyuki.funaya`  | 201 | 人手 VERIFY |
| `automation_gicp`  |  68 | GICP 自動 (未VERIFY) |

→ **VERIFY 済み box だけを学習・可視化に使いたい場合**:

```python
ALLOWED = {'hiroyuki.funaya', 'yolo', 'automation_yolo'}

def last_user(eh):
    if isinstance(eh, dict):  return eh.get('user')
    if isinstance(eh, list) and eh:
        return eh[-1].get('user') if isinstance(eh[-1], dict) else None
    return None

dets = [d for d in ann['details']
        if d.get('type') == 'box'
        and last_user(d.get('edit_history')) in ALLOWED]
```

label 別 VERIFY 済み比率は `car` で ~80%、`sign / traffic_body / lanemarker_edge` 系は
未VERIFYの方が圧倒的に多い (= 自動検出のまま放置)。学習時はこれをそのまま信じない。

## box 内点判定 (rear_axle FLU)

`build_woven_sequence_v3._is_obj_per_point` と同等。最小実装:

```python
def points_in_box(pts, center, size, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
    local = (R @ (pts - center).T).T
    return np.all(np.abs(local) <= size * 0.5, axis=1)

# yaw は direction から
yaw = np.arctan2(direction[1], direction[0])
```

## 可視化スクリプト

| ファイル | 何をするか |
|---|---|
| `scripts/visualization/vis_woven_proj_only.py` | 投影パイプの動作確認用。LiDAR を画像に重畳するだけ。box は描かない |
| `scripts/visualization/vis_woven_bbox.py`      | 投影 + 3D box 8 角ワイヤフレーム + box 内点を緑、それ以外を depth カラー |
| `scripts/visualization/vis_cross_frame_woven.py` | CalibNetCrossFrame の zero-shot 可視化 (要 ckpt) |

実行例:

```bash
# 投影だけ (まずこれで投影が合ってるか確認)
python3 scripts/visualization/vis_woven_proj_only.py
# → out/vis_woven_proj_only.png

# box + 内点緑 (VERIFY 済みだけ表示, デフォ4フレーム)
python3 scripts/visualization/vis_woven_bbox.py \
  --out out/vis_woven_bbox.png

# 全 box (フィルタ無効)
python3 scripts/visualization/vis_woven_bbox.py --users all --out out/all.png

# car だけ
python3 scripts/visualization/vis_woven_bbox.py --labels car --out out/cars.png

# 別シーケンス
python3 scripts/visualization/vis_woven_bbox.py \
  --seq ~/git/loom/backend/assets/woven_sequence/llinking_27/tf_long2/sequence=ip607_... \
  --out out/other.png
```

## 前処理 (tile cache)

`scripts/preprocessing/build_woven_sequence_v3.py` が
ZOD/PandaSet/Kamikado と同じ V3 タイル形式 (`pts/uv_full/is_obj/...`) で吐く。
学習側 (`scripts/training/train_ps_v3.py`) はこれを区別せず食べられる。

## 引き継ぎメモ (やってはいけないこと)

1. **投影を自分で書かない**。`build_woven_sequence_v3` から import する
2. **camera_delay を無視しない**。`_pose_at_camera_time` を必ず通す
3. **未VERIFY (`automation` / `automation_gicp`) の box を学習に混ぜない**。
   特に `sign / traffic_body / lanemarker_edge` は自動検出ノイズが多い
4. **`projected_3d_corners` (annotation 内の事前投影) と Kannala 投影は座標系が一致**してる。
   両方使って差分を取れば投影が正しいかセルフチェックできる

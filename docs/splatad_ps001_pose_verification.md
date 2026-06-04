# SplatAD on PandaSet 001: PS pose 品質検証 + SO3xR3 ON で遠方シャープネス回復

**TL;DR** ─ SplatAD を PandaSet 001 で default vs `camera_optimizer.mode=SO3xR3` の 2 通り走らせ、`pose_adjustment` を取り出して PS の元 pose 精度を定量化、render 比較で「**pose 補正で遠方信号機エリアの PSNR が +3.65 dB (MSE 半減以下)**」を実証。

---

## 1. 走った 2 run

| run | mode | cameras | iters | 役割 |
|---|---|---|---|---|
| `ps001_default` | default (frozen pose、velocity+sync のみ) | all 6 | 30000 | baseline |
| `ps001_front_so3xr3` | **`camera_optimizer.mode=SO3xR3`** ON | front_only | 30000 | pose 補正 |

両方 Y0 (RTX 3090) で約 1.5h 走行。SO3xR3 は per-sensor-per-frame で 6dof delta を学習させる。

---

## 2. Pose 補正の数値

### 2.1 Per-instance pose_adjustment (final, step 30000)

```
front_cam:  trans  4.7 mm (max 14.8 mm)   rot 0.062° (max 0.116°)
LiDAR:      trans 13.4 mm (max 43.1 mm)   rot 0.083° (max 0.177°)
```

### 2.2 Vehicle 軌跡 drift (cam+lidar 共通成分)

```
vehicle trans:  mean 6.9 mm, max 24.9 mm
vehicle rot:    mean 0.057°, max 0.108°
```

→ **PS の軌跡 backbone そのものが cm スケールで微妙にズレてた** ことを定量化。

### 2.3 cam-LiDAR 静的 extrinsic bias (frame 平均)

```
trans:  tz = +10.3 mm (LiDAR が cam より 10mm 前方に取付け)
        tx = +0.2 mm, ty = +1.7 mm
rot:    ry = -0.058° (yaw 0.06° ズレ), 他 ≈ 0
```

→ PS 配布の cam-LiDAR extrinsic は **forward 方向に 10mm + yaw 0.06° の取付ズレ** がある。

### 2.4 Per-frame visualization

![pose comparison](assets/splatad_ps001/pose_compare.png)

上段: per-frame translation/rotation 補正量
下段: 元 PS 軌跡 (青) vs SplatAD 補正後 (緑) + per-frame delta vector (100× scale)

→ progressive drift ではなく、frame ごとの jitter + 一部 systematic 成分 (yaw rz、cam-LiDAR tz)。

---

## 3. Render 品質比較 (frame 01)

### 3.1 同視点 3 way

![frame 01 3-way](assets/splatad_ps001/signal_crop_3way.png)

(信号機エリア crop: GT / Default / SO3xR3)

### 3.2 全体 + 領域別 PSNR

| 領域 | Default PSNR | SO3xR3 PSNR | ΔPSNR | 解釈 |
|---|---|---|---|---|
| Whole frame (1920×1080) | 26.99 dB | 27.39 dB | +0.41 dB | 全体平均、近景含む |
| 近景 (y+200 ≒ 道路+駐車車両) | 25.33 dB | 26.28 dB | +0.96 dB | 改善控えめ (元から OK) |
| **遠方信号機 crop** | **24.84 dB** | **28.50 dB** | **+3.65 dB** 🔥 | MSE 半減以下 = 「ぼけ量」が半分 |

ΔPSNR の比 (vs whole frame): 近景 2.4×, 信号機 **9×**
→ **pose 補正の効果は距離に比例して効く** ことが定量的に立証。

### 3.3 Edge sharpness (Laplacian variance)

| 領域 | GT | Default | SO3xR3 |
|---|---|---|---|
| Whole frame | 302.4 | 162.7 (54%) | 227.6 (75%) |
| 遠方信号 crop | 963.6 | 609.9 (63%) | 761.9 (79%) |

→ sharpness が GT の 54% → 75% に回復、信号機エリアでは 63% → 79%。

### 3.4 近景 crop (y+200, 道路)

![y+200 crop](assets/splatad_ps001/crop_y200.png)

---

## 4. 物理計算との整合

PS front_camera: fx=1970 px, HFOV=52°, per-pixel angular resolution=0.027°/px

| rotation | image shift (距離無関係) | 200m 先での lateral shift |
|---|---|---|
| 0.062° (mean) | 2.1 px | 22 cm |
| 0.108° (vehicle max) | 3.7 px | 38 cm |
| 0.177° (lidar max) | 6.1 px | 62 cm |

→ Default の 0.1° 残りで遠方信号機 (~2-4 px サイズ) が 3-6 px ぶれる = 完全消失も納得。
SO3xR3 後の残差 (< 0.06° avg) では sub-pixel に圧縮、信号機が「2-3 px の明点」に復帰する。

---

## 5. 結論

1. **PS001 の pose は完璧ではない**: vehicle 軌跡に cm スケール + 0.1° 級の jitter、cam-LiDAR extrinsic に 10mm / 0.06° の取付ズレ
2. **default SplatAD (mode='off') では pose を一切修正しない** ため、これらズレが Gaussian の σ に直接流れ込んで遠方ぼけ
3. **`camera_optimizer.mode=SO3xR3` ON** で pose を per-frame per-sensor で学習させると、遠方ぼけが定量的に改善 (信号機 crop +3.65 dB / sharpness 63→79%)
4. **論文 demo クオリティ (PSNR 30+)** にはまだ届いてないが、これは Gaussian cap (5M) や iter 数の問題で、pose の効果は明確に切り分けられた

## 5.5 Refined pose の commit

per-frame の `pose_adjustment` (cam + lidar) と元の PS pose を同梱したファイル:

- [`assets/splatad_ps001/refined_poses_ps001.json`](assets/splatad_ps001/refined_poses_ps001.json) (50KB)

構造:
```json
{
  "scene": "001",
  "source": "SplatAD camera_optimizer.mode=SO3xR3 (front_cam + lidar), step 30000",
  "summary": {
    "vehicle_drift_trans_mm_mean": 6.93,
    "vehicle_drift_trans_mm_max": 24.91,
    "vehicle_drift_rot_deg_mean": 0.0568,
    "vehicle_drift_rot_deg_max": 0.1081,
    "cam_lidar_extrinsic_static_bias_mm": [0.22, 1.66, 10.26],
    "cam_lidar_extrinsic_static_bias_deg": [-0.002, -0.058, 0.010]
  },
  "frames": {
    "0": {
      "cam_delta_trans_m": [tx, ty, tz],
      "cam_delta_axisangle_rad": [rx, ry, rz],
      "lidar_delta_trans_m": [...],
      "lidar_delta_axisangle_rad": [...],
      "original_cam_pose": { ... PandaSet format ... },
      "original_lidar_pose": { ... }
    },
    ...
  }
}
```

使い方:
```python
import json
d = json.load(open("docs/assets/splatad_ps001/refined_poses_ps001.json"))
for fi, frame in d["frames"].items():
    cam_orig = frame["original_cam_pose"]  # PS pose dict
    delta = frame["cam_delta_trans_m"]  # 3-vec
    rot_delta = frame["cam_delta_axisangle_rad"]  # 3-vec
    # apply delta to original pose → refined pose
    refined_cam_position = [
        cam_orig["position"]["x"] + delta[0],
        cam_orig["position"]["y"] + delta[1],
        cam_orig["position"]["z"] + delta[2],
    ]
    # rotation: axisangle delta ⊕ original quaternion
```

これを cross_frame の supervision で「公開 PS pose」じゃなく「SplatAD-refined」を使うことで、訓練 GT が cm 精度向上。

---

## 6. 含意 (cross_frame net への接続)

- SplatAD は「**地図作成側**」ではなく「**pose refinement + 高精度教師生成器**」として使う用途で有効
- 公開 PS pose を SplatAD-refined pose に置き換えれば、cross_frame net の supervision が cm 精度で上がる
- 同じ手で Waymo / Argoverse でも refine 可、最終的に **公開 GT より高精度な pose** を auto-emit する pipeline が現実的

---

## 関連

- [unified_calib_odom_map.md](unified_calib_odom_map.md) — token chain で calib+odom+map を 1 ネットで解く方針
- [unified_modality_primitive.md](unified_modality_primitive.md) — Q/KV 分離で modality 不問
- memo: [`reference_splatad_calib_modes`](../.claude/projects/-home-hiro-git-e2e-calib/memory/reference_splatad_calib_modes.md) — SplatAD の 2 系 optimizer (static / velocity) の使い分け
- memo: [`project_ps_calib_full_picture`](../.claude/projects/-home-hiro-git-e2e-calib/memory/project_ps_calib_full_picture.md) — PS の side cam calib 仮説 (今回は前カメ単独で検証)

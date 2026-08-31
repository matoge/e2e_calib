# Using the kamikado + WovenSequence checkpoint

[English](2026-08-31_kmwv-usage.en.md)

学習の中身は [レポート](2026-08-31_kmwv-calibration.md) 参照。ここは「重みを取ってきてどう使うか」だけ。

## 取得

```bash
git clone <this repo>
cd e2e_calib
git lfs pull -I experiments/kmwv_s3_ba40_512r256_0831_0325/best_model.pt
```

`.pt` ファイルは LFS (`.gitattributes` の `*.pt filter=lfs`)。上のコマンドで **その 1 ファイルだけ** 取ってくる（他の実験の重みも欲しければ `git lfs pull` で全部）。

## 前提

- PyTorch 2.3 以上（学習は `nvcr.io/nvidia/pytorch:24.02-py3` = torch 2.3.0a0）
- Python 3.10
- LMDB キャッシュ (`cache_v5/{kamikado,woven}_v3_full`) は同 host 上に想定。推論だけなら不要。

## 最短の呼び出し

```python
import torch
from pathlib import Path
import importlib.util
from models.calibnet2 import CalibNet2

exp = Path('experiments/kmwv_s3_ba40_512r256_0831_0325')

# config.py に全 CLI 引数がそのまま保存されている
spec = importlib.util.spec_from_file_location('_cfg', exp / 'config.py')
mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cfg  = mod.CFG

model = CalibNet2(img_size=cfg['img_size'],
                  grid_n=cfg['grid_n'],
                  n_iter=cfg['n_iter'],
                  use_info_head=cfg['use_info_head']).cuda().eval()

sd = torch.load(exp / 'best_model.pt', map_location='cpu', weights_only=False)
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not missing and not unexpected, (missing, unexpected)
```

**config.py を必ず読むこと。** ckpt と `n_iter` / `grid_n` / `use_info_head` がズレると state_dict が壊れて silent に精度が落ちる（見た目 shape は合うが読み手が違う）。

## 入力

| 引数 | 形 | 意味 |
|---|---|---|
| `image` | (B, 3, 256, 256) | uint8 でも float でもよいが、モデル側は 0–1 float で正規化される |
| `point_in` | (B, N, 4) | 各点 `[u/img_size, v/img_size, depth_m, intensity]` (摂動後の投影 uv、depth はメートル) |
| `bucket_uvd` | (B, N, K, 3) | 各点の近傍 K 点 (frustum context) |
| `bucket_valid` | (B, N, K) | bool |
| `key_padding_mask` | (B, N) | True = padding |

`point_in[:, :, :2]` は「摂動後の投影位置」。model はここから **戻すべき差分 (μ, σ) と情報行列 W** を返す。

## 出力

```python
with torch.no_grad():
    per_pt, W = model(image, point_in, bucket_uvd=bucket_uvd,
                      bucket_valid=bucket_valid,
                      key_padding_mask=key_padding_mask)
# per_pt: (B, N, 5) = (mu_u, mu_v, log_sx, log_sy, rho) in px
# W:      (B, N, 2, 2) — 各点の 2×2 情報行列 (Cholesky で PSD 保証)
```

**mu が「元の投影から差し引くべき量」、W が「その差分がどのくらい信用できるか」。** 6-DOF は自分で GN する（[`scripts/ba/gn_pose.py`](../scripts/ba/gn_pose.py)、学習パラメータゼロ）。

## 6-DOF まで解きたい場合

一番シンプルな呼び方 (28 or 40 窓のクロップを 1 フレームで融合):

```python
from scripts.ba.gn_pose import solve_gn_pose

# per_pt, W は上の forward で得たもの
# pts_cam: (M, 3) LiDAR 点の cam 座標 (メートル)
# K:       (3, 3) intrinsics
# uv_hat:  (M, 2) 摂動後の投影 uv (per_pt の μ を足せば元に戻す)
delta6, H6 = solve_gn_pose(
    pts_cam=pts_cam, K=K,
    uv_hat=uv_hat,           # 摂動後の投影
    delta_uv=per_pt[..., :2],  # 戻すべき量
    W=W,                     # 2×2 情報行列
    iters=4, damping=1e-3,
)
# delta6: (6,) 6-DOF 補正 (rot 3 + trans 3)
# H6:     (6, 6) 情報行列。fp64 で解けている。
```

`H6` を N フレームで足せば multi-frame fusion (レポート §複数フレーム合成 参照)。

## 複数フレーム合成

[`scripts/eval/frame_fusion.py`](../scripts/eval/frame_fusion.py) が全部やる:

```bash
python scripts/eval/frame_fusion.py \
    experiments/kmwv_pose_dump_0831_1249/pose_dump_ep001.pt \
    --out  docs/assets/kmwv_quick/fusion.json \
    --plot docs/assets/kmwv_quick/fusion.png
```

`pose_dump_epNNN.pt` は `datasets/train_cnd2_ddp.py --dump-pose` で val 1 epoch 走らせて出す。`sum` / `gate3` / `CI` の 3 規則を F ∈ {1,2,4,8,16,32} で sweep して JSON と PNG を書き出す。

## 学習し直したいとき

- **キャッシュ**: `/raid/home/hfunaya/cache_v5/{kamikado,woven}_v3_full` (LMDB packed)。build 手順は `scripts/preprocessing/BUILD_COMMANDS.md`
- **段階学習**: `_kick_kmwv_s1_pts_512.sh` → 収束後 `_kick_kmwv_s3_ba40_512.sh` (S2_CKPT を S1 の best_model.pt にする)。
- **他のデータセットに拡張**: `--per-cache-crop-px <path>:<size>` を足せば cache ごとに crop 側を変えられる。crop 512 → img 256 で nuScenes と同じ入力条件、crop 256 → img 256 なら 1600×900 の nuScenes 本体と同じ扱い。

## 動作確認

```bash
# Docker で 1 val epoch だけ回して pose_dump を出す (~1.5 min on 8× V100)
bash _kick_kmwv_pose_dump.sh   # ClearML task が立つ

# 手元で fusion (pose_dump さえあれば docker 不要)
python scripts/eval/frame_fusion.py \
    experiments/<pose_dump_run>/pose_dump_ep001.pt --plot fusion.png
```

期待値は `docs/2026-08-31_kmwv-calibration.md §複数フレーム合成` の table。F=1 で 0.025°/6.2mm、F=32 で 0.007°/1.4mm 前後に落ちれば ckpt は正しく動いている。

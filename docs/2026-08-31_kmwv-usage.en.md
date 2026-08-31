# Using the kamikado + WovenSequence checkpoint

[日本語](2026-08-31_kmwv-usage.md)

Training details are in the [report](2026-08-31_kmwv-calibration.en.md). This document only covers "how to pull the weights and run them."

## Fetch

```bash
git clone <this repo>
cd e2e_calib
git lfs pull -I experiments/kmwv_s3_ba40_512r256_0831_0325/best_model.pt
```

`.pt` files are tracked by LFS (`.gitattributes` has `*.pt filter=lfs`). The command above fetches **only that one file**; drop `-I ...` to grab every experiment's weights.

## Prereqs

- PyTorch ≥ 2.3 (training used `nvcr.io/nvidia/pytorch:24.02-py3` = torch 2.3.0a0)
- Python 3.10
- The LMDB caches (`cache_v5/{kamikado,woven}_v3_full`) are assumed to be on the same host. Not needed for pure inference.

## Minimal load

```python
import torch
from pathlib import Path
import importlib.util
from models.calibnet2 import CalibNet2

exp = Path('experiments/kmwv_s3_ba40_512r256_0831_0325')

# config.py stores every CLI arg the ckpt was trained with
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

**Always read `config.py`.** Mismatched `n_iter` / `grid_n` / `use_info_head` between the ckpt and the freshly built model silently degrades accuracy — the state_dict shape can still match but the layers won't line up.

## Inputs

| arg | shape | meaning |
|---|---|---|
| `image` | (B, 3, 256, 256) | uint8 or float; the model normalises to 0–1 internally |
| `point_in` | (B, N, 4) | each point `[u/img_size, v/img_size, depth_m, intensity]` (uv is the *perturbed* projection, depth is metres) |
| `bucket_uvd` | (B, N, K, 3) | K neighbouring LiDAR points (frustum context) |
| `bucket_valid` | (B, N, K) | bool |
| `key_padding_mask` | (B, N) | True = padding |

`point_in[:, :, :2]` is the projection *after* the perturbation. The model returns the **correction (μ, σ) plus per-point information matrix W** — how much to move each point back, and how much to trust that move.

## Outputs

```python
with torch.no_grad():
    per_pt, W = model(image, point_in, bucket_uvd=bucket_uvd,
                      bucket_valid=bucket_valid,
                      key_padding_mask=key_padding_mask)
# per_pt: (B, N, 5) = (mu_u, mu_v, log_sx, log_sy, rho) in pixels
# W:      (B, N, 2, 2) — per-point 2×2 information matrix (Cholesky, PSD)
```

**mu is the offset to subtract from the observed uv, W is how strongly to weight that offset.** The 6-DOF is solved outside the net ([`scripts/ba/gn_pose.py`](../scripts/ba/gn_pose.py), zero learnable params).

## Solving to 6-DOF

Simplest usage — fuse 28 or 40 crop windows of a single frame:

```python
from scripts.ba.gn_pose import solve_gn_pose

# per_pt, W from the forward above
# pts_cam: (M, 3) LiDAR points in camera frame (metres)
# K:       (3, 3) intrinsics
# uv_hat:  (M, 2) perturbed projection uv (add per_pt's μ to get back to GT)
delta6, H6 = solve_gn_pose(
    pts_cam=pts_cam, K=K,
    uv_hat=uv_hat,           # perturbed projections
    delta_uv=per_pt[..., :2],  # per-point correction
    W=W,                     # per-point 2×2 info
    iters=4, damping=1e-3,
)
# delta6: (6,) 6-DOF correction (rot 3 + trans 3)
# H6:     (6, 6) information matrix, solved in fp64.
```

Adding `H6` across N frames gives multi-frame fusion (report §Multi-frame fusion).

## Multi-frame fusion

[`scripts/eval/frame_fusion.py`](../scripts/eval/frame_fusion.py) handles it end-to-end:

```bash
python scripts/eval/frame_fusion.py \
    experiments/kmwv_pose_dump_0831_1249/pose_dump_ep001.pt \
    --out  docs/assets/kmwv_quick/fusion.json \
    --plot docs/assets/kmwv_quick/fusion.png
```

The `pose_dump_epNNN.pt` file is produced by running `datasets/train_cnd2_ddp.py --dump-pose` for one val epoch. The fusion script sweeps F ∈ {1,2,4,8,16,32} under three rules (`sum` / `gate3` / `CI`) and writes both a JSON and a PNG.

## Retraining

- **Caches**: `/raid/home/hfunaya/cache_v5/{kamikado,woven}_v3_full` (LMDB packed). Build instructions: `scripts/preprocessing/BUILD_COMMANDS.md`.
- **Staged recipe**: `_kick_kmwv_s1_pts_512.sh` → once converged, `_kick_kmwv_s3_ba40_512.sh` with `S2_CKPT=<S1 best_model.pt>`.
- **Extending to other datasets**: use `--per-cache-crop-px <path>:<size>` to override the crop size per cache. crop 512 → img 256 matches the nuScenes-report input; crop 256 → img 256 matches nuScenes native 1600 × 900.

## Sanity check

```bash
# Docker: run 1 val epoch, dump per-frame poses (~1.5 min on 8× V100)
bash _kick_kmwv_pose_dump.sh   # submits a ClearML task

# Local (no docker needed once the dump exists)
python scripts/eval/frame_fusion.py \
    experiments/<pose_dump_run>/pose_dump_ep001.pt --plot fusion.png
```

Expected: `docs/2026-08-31_kmwv-calibration.en.md §Multi-frame fusion` — F=1 lands at ~0.025° / 6.2 mm, F=32 at ~0.007° / 1.4 mm. If your numbers match, the ckpt loaded correctly.

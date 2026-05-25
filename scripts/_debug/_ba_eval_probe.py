"""Probe: does the model produce non-zero duv predictions on a simple ±1° / ±0.10m pert?"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.eval import eval_shared_256x800 as _ks
from scripts.eval.eval_ps_shared_256x800 import _build_model, _load_cfg

DEV = torch.device('cuda')
exp = REPO / 'experiments/ps_full_n4_img256_grid32_dgx3_100ep'
cfg = _load_cfg(exp)
model = _build_model(cfg).to(DEV)
sd = torch.load(exp / 'best_model.pt', map_location=DEV, weights_only=False)
if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
model.load_state_dict(sd); model.eval()

ds = PandaSetCalibDatasetFull(
    cache_dir='/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full', split='val',
    img_size=cfg['img_size'], min_crop_px=128, max_crop_px=512,
    max_offset_m=0.0, max_rot_deg=0.0,
    oversample=1, grid_n=cfg.get('grid_n', 16), center_band=0.0, preload=False)
inst = ds._load_inst(17)
ypr = np.array([0.0, 0.0, 0.5], dtype=np.float64)  # roll 0.5°
t   = np.array([0.0, 0.0, 0.0], dtype=np.float64)
wins = []
for u0, v0 in [(0,0),(256,0),(0,256),(256,256)]:
    w = _ks._build_subwin(ds, inst, t, ypr, u0=u0, v0=v0, cs=256)
    if w is not None: wins.append(w)
print(f'wins: {len(wins)}')
moved = [t.to(DEV) if torch.is_tensor(t) else t for t in collate_full(wins)]
(imgs, true_uvd, dist_uvd, pad_mask, vfp,
 bucket_uvd, bucket_valid, _,
 pts_cam_orig, duv_orig, K_orig, cs_b, _) = moved
valid = ~pad_mask
print(f'imgs={imgs.shape} dist_uvd={dist_uvd.shape} dist_uvd[..., :3].abs.mean={dist_uvd[..., :3].abs().mean():.3f}')
print(f'duv_orig.norm.mean (target Δuv in orig-px) = {duv_orig[valid].norm(dim=-1).mean():.3f}px')

use_intensity = getattr(model, 'use_intensity', True)
if use_intensity:
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
else:
    point_in = dist_uvd[..., :3]

with torch.no_grad():
    out = model(imgs.float()/255.0, point_in, key_padding_mask=pad_mask, vfp=vfp,
                bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
per_pt = out[0] if isinstance(out, tuple) else out
duv_local = per_pt[..., :2]
duv_local_v = duv_local[valid]
print(f'duv_pred_local norm: mean={duv_local_v.norm(dim=-1).mean():.4f}px max={duv_local_v.norm(dim=-1).max():.4f}px')
scale_l2o = (cs_b / float(cfg['img_size'])).reshape(-1,1,1)
duv_pred_orig = duv_local * scale_l2o
print(f'scale_l2o per-tile = {scale_l2o.flatten()[:8]}')
print(f'duv_pred_orig norm (orig-px): mean={duv_pred_orig[valid].norm(dim=-1).mean():.4f}px max={duv_pred_orig[valid].norm(dim=-1).max():.4f}px')

# compare to oracle duv in orig-px
duv_pred_orig_v = duv_pred_orig[valid]
duv_orig_v = duv_orig[valid]
print(f'oracle duv_orig norm: mean={duv_orig_v.norm(dim=-1).mean():.4f}px')
err = (duv_pred_orig_v - duv_orig_v).norm(dim=-1)
print(f'pred - oracle norm: mean={err.mean():.4f}px median={err.median():.4f}px')

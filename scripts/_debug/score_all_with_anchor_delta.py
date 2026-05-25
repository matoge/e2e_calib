"""Solve δ̂ ONCE with B=800 anchored on idx=17, then SCORE every val+train
fisheye instance (no overlay rendering). Just numbers → results.json so we
can pick visually distinct sub-pixel candidates afterwards.
"""
from __future__ import annotations
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.eval.eval_shared_256x800 as ess
from datasets.pandaset_full import PandaSetCalibDatasetFull
from scripts.util.projection import project_kannala as proj_kb


def _score(inst, ypr_t, t_t, delta_np):
    """Return (pert_mean_px, corr_mean_px) — same metric the overlay prints,
    but skip matplotlib entirely.  Works on the parent tile coordinate system.
    """
    if 'jpg_bytes' in inst:
        parent = np.array(Image.open(io.BytesIO(inst['jpg_bytes'])).convert('RGB'))
    else:
        parent = np.asarray(inst['img'])
        if parent.ndim == 3 and parent.shape[0] in (1, 3):
            parent = np.transpose(parent, (1, 2, 0))
    pH, pW = parent.shape[:2]
    K_full  = inst['K_full'].numpy().astype(np.float64)
    dist_kb = inst['distortion'].numpy().astype(np.float64)
    pts     = inst['pts'].numpy().astype(np.float64)
    tile_u0 = float(inst.get('tile_u0', 0))
    tile_v0 = float(inst.get('tile_v0', 0))

    R_pert = R.from_euler('zyx', ypr_t, degrees=True).as_matrix()
    pts_pert = (pts - t_t) @ R_pert
    R_d = R.from_rotvec(np.deg2rad(delta_np[:3])).as_matrix()
    pts_corr = pts_pert @ R_d.T + delta_np[3:]

    def _project(P):
        uv = proj_kb(P, K_full, dist_kb)
        return uv - np.array([tile_u0, tile_v0])

    uv_gt   = _project(pts)
    uv_pert = _project(pts_pert)
    uv_corr = _project(pts_corr)

    def _in(uv, z):
        return ((z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < pW)
                & (uv[:, 1] >= 0) & (uv[:, 1] < pH))

    def _mean(uv_a, uv_b, z):
        m = _in(uv_a, z) & _in(uv_b, z)
        if not m.any():
            return float('nan'), 0
        return float(np.linalg.norm(uv_a[m] - uv_b[m], axis=1).mean()), int(m.sum())

    p_mean, n_p = _mean(uv_pert, uv_gt,  pts[:, 2])
    c_mean, n_c = _mean(uv_corr, uv_gt,  pts_corr[:, 2])
    return p_mean, c_mean, n_c


def main():
    cfg = ess._load_cfg()
    ds = PandaSetCalibDatasetFull(
        cache_dir=ess.CACHE, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    model = ess._build_model(cfg).to(ess.DEVICE)
    sd = torch.load(ess.CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False); model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    inst0 = ds._load_inst(17)
    dist_one = inst0['distortion'].clone().detach().to(torch.float32).reshape(1, 4)
    rng = np.random.RandomState(1007)
    ypr_t, t_t = ess._draw_pert(rng, rot_deg=0.5, t_m=0.05)
    rng_b = np.random.RandomState(1008)
    delta, B, _ = ess._solve_one(
        model, ds, target_idx=17, n_inst=200, cs=256, n_per_inst=4,
        rng=rng_b, ypr_target=ypr_t, t_target=t_t, dist_one=dist_one,
        cfg=cfg, label='anchor')
    delta_np = delta.cpu().numpy()
    print(f'[anchor] B={B}  delta={delta_np.tolist()}', flush=True)
    print(f'[anchor] ypr_t={ypr_t.tolist()}  t_t={t_t.tolist()}', flush=True)

    out = Path('scripts/_debug/score_all_anchor17')
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'delta.npy',  delta_np)
    np.savez(out / 'target.npz', ypr=ypr_t, t=t_t)

    rows = []
    for split in ('val', 'train'):
        ds_s = PandaSetCalibDatasetFull(
            cache_dir=ess.CACHE, split=split,
            img_size=cfg['img_size'],
            min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
            max_offset_m=0.0, max_rot_deg=0.0,
            oversample=1, grid_n=cfg.get('grid_n', 16),
            center_band=0.0, preload=False)
        N = len(ds_s.fnames)
        print(f'[{split}] N={N}', flush=True)
        t0 = time.time()
        for idx in range(N):
            try:
                inst = ds_s._load_inst(idx)
            except Exception:
                continue
            if not inst.get('is_fisheye', False) or 'distortion' not in inst:
                continue
            try:
                p_mean, c_mean, n_c = _score(inst, ypr_t, t_t, delta_np)
            except Exception:
                continue
            rows.append((split, idx, p_mean, c_mean, n_c))
            if idx % 500 == 0:
                print(f'  [{split} {idx:6d}/{N}] pert={p_mean:6.2f}  corr={c_mean:6.2f}  '
                      f'n={n_c}  t={time.time()-t0:.1f}s', flush=True)
        print(f'[{split}] done {time.time()-t0:.1f}s  rows={len(rows)}', flush=True)

    with open(out / 'results.json', 'w') as fp:
        json.dump(rows, fp)
    sub = sorted([r for r in rows if r[3] == r[3]], key=lambda r: r[3])
    print('=== top 50 by corr_mean ===')
    for s, i, p, c, n in sub[:50]:
        print(f'  {s} idx={i:6d}  pert={p:6.2f}  corr={c:6.2f}  n={n}')
    print(f'wrote {out/"results.json"}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()

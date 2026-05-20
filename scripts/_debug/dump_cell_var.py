"""Sanity check for the 16 px plateau hypothesis.

For one tile, sample many random δ_pert (ω_x, ω_y in ±1°), compute the
oracle Δuv per point, and group points by their 32×32 token cell. The
token-gather averages all points in a cell, so the per-cell std of Δuv
is a lower bound on |Δuv − oracle| that any cell-based predictor can
achieve. If that bound matches the observed ~16 px plateau, the gather
is the bottleneck and per-point query (or finer grid / bilinear) fixes
it.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import importlib.util, sys as _s
spec = importlib.util.spec_from_file_location(
    'rd', str(REPO / 'scripts' / '_debug' / 'overfit_1image_random_delta.py'))
rd = importlib.util.module_from_spec(spec); _s.modules['rd'] = rd
spec.loader.exec_module(rd)

from scripts.ba.ba_kb_jac import project_kb

cf = rd.load_frame(rd.m.SCENE, rd.m.FRAME)
H_full, W_full = cf.img.shape[:2]
K = cf.K.astype(np.float64); dist = np.asarray(cf.dist, np.float64)
pts_cam = cf.pts_cam.astype(np.float64)
cu, cv = W_full // 2, H_full // 2 + rd.m.DV
u0 = cu - rd.m.TILE // 2; v0 = cv - rd.m.TILE // 2
u1 = u0 + rd.m.TILE;       v1 = v0 + rd.m.TILE
TILE = rd.m.TILE

rng = np.random.RandomState(7)

# Within-cell variance under FIXED δ (≈ how badly cells alias different
# points sharing the same Δuv-target).
def per_delta_within_cell(omx, omy, G):
    d = rd.make_delta6(omx, omy)
    uv_p, z_p, pc_in = rd.perturb_pool(pts_cam, K, dist, d, u0, v0, u1, v1)
    uv_l = uv_p.copy(); uv_l[:, 0] -= u0; uv_l[:, 1] -= v0
    duv_o = project_kb(pc_in, K, dist) - uv_p           # (N, 2)
    cell = TILE / G
    gx = np.clip((uv_l[:, 0] / cell).astype(int), 0, G - 1)
    gy = np.clip((uv_l[:, 1] / cell).astype(int), 0, G - 1)
    cid = gy * G + gx
    # within-cell residual after subtracting the cell mean.
    resid_per_pt = np.zeros_like(duv_o)
    for c in np.unique(cid):
        m = cid == c
        resid_per_pt[m] = duv_o[m] - duv_o[m].mean(axis=0, keepdims=True)
    rms = float(np.sqrt((resid_per_pt ** 2).sum(axis=1).mean()))
    occ = np.bincount(cid, minlength=G * G)
    return rms, occ, len(uv_p)


# Cross-δ variance at FIXED token cell (≈ same cell sees many different
# Δuv-targets across the random-δ training stream).
def cross_delta_within_cell(n_delta, G):
    cell = TILE / G
    bucket: dict[int, list[np.ndarray]] = {}
    n_pts_total = 0
    for _ in range(n_delta):
        a = rng.uniform(-rd.DELTA_RANGE_DEG, rd.DELTA_RANGE_DEG)
        b = rng.uniform(-rd.DELTA_RANGE_DEG, rd.DELTA_RANGE_DEG)
        d = rd.make_delta6(a, b)
        uv_p, z_p, pc_in = rd.perturb_pool(pts_cam, K, dist, d, u0, v0, u1, v1)
        if len(uv_p) < 50:
            continue
        uv_l = uv_p.copy(); uv_l[:, 0] -= u0; uv_l[:, 1] -= v0
        duv_o = project_kb(pc_in, K, dist) - uv_p
        gx = np.clip((uv_l[:, 0] / cell).astype(int), 0, G - 1)
        gy = np.clip((uv_l[:, 1] / cell).astype(int), 0, G - 1)
        cid = gy * G + gx
        for c, du in zip(cid, duv_o):
            bucket.setdefault(int(c), []).append(du)
        n_pts_total += len(uv_p)
    # for each cell with ≥2 entries, pull RMS deviation from cell mean.
    sq, n = 0.0, 0
    n_cells_seen = 0
    occ_hist = []
    for c, dus in bucket.items():
        if len(dus) < 2:
            continue
        arr = np.stack(dus, axis=0)
        m = arr.mean(axis=0, keepdims=True)
        sq += ((arr - m) ** 2).sum()
        n  += arr.shape[0] * 2
        n_cells_seen += 1
        occ_hist.append(arr.shape[0])
    rms_xd = float(np.sqrt(sq / max(n, 1)) * np.sqrt(2))   # ≈ √sum-of-squares per (Δu,Δv) jointly
    return rms_xd, np.asarray(occ_hist), n_pts_total, n_cells_seen


print('─── within-cell residual (FIXED δ): same cell, multiple points ──')
print(f'{"δ":<14} {"G":>4} {"N_pts":>6} {"cells":>5} {"max occ":>8} {"mean occ":>9} {"RMS resid (px)":>16}')
for omx, omy in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.5), (-0.8, 1.2)]:
    for G in (32, 64, 128):
        rms, occ, n = per_delta_within_cell(omx, omy, G)
        nz = occ[occ > 0]
        print(f'({omx:+.1f},{omy:+.1f})   {G:>4} {n:>6d} {len(nz):>5d} '
               f'{occ.max():>8d} {nz.mean():>9.2f} {rms:>16.3f}')

print()
print('─── cross-δ residual at FIXED cell (random-δ moving target) ──')
print(f'{"G":>4} {"n_δ":>6} {"cells used":>11} {"max stack":>10} {"mean stack":>11} {"RMS resid (px)":>16}')
for G in (32, 64, 128):
    rms_xd, occ_hist, n_pts, n_cells = cross_delta_within_cell(64, G)
    print(f'{G:>4} {64:>6d} {n_cells:>11d} {int(occ_hist.max()):>10d} '
           f'{occ_hist.mean():>11.2f} {rms_xd:>16.3f}')
